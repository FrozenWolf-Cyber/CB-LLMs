"""
Two-phase steerability benchmark: (1) Llama-3-8B + CBL generate texts, release GPU;
(2) Qwen3.5 ~27B GGUF as LLM judge (thinking-friendly chat template), one 1–10 score per text.

Designed for a single A100 80GB: after phase 1, only the GGUF judge should occupy VRAM.

Requires: ``pip install llama-cpp-python`` (CUDA build recommended), plus usual torch/transformers.

Example (local GGUF)::

    python resume_qwen35_gguf_judge_metric.py \\
      --run_ids_pickle all_wandb_runs.pkl \\
      --dataset SetFit/sst2 \\
      --wandb_entity YOUR_ENTITY \\
      --judge_gguf_path /path/to/Qwen3.5-27B-Q8_0.gguf \\
      --gen_device cuda:0

Example (HuggingFace from_pretrained — auto-downloads)::

    python resume_qwen35_gguf_judge_metric.py \\
      --run_ids_pickle all_wandb_runs.pkl \\
      --dataset SetFit/sst2 \\
      --wandb_entity YOUR_ENTITY \\
      --judge_gguf_path "unsloth/Qwen3.5-27B-GGUF::Qwen3.5-27B-Q8_0.gguf" \\
      --gen_device cuda:0

Optional: persist full judge outputs (reasoning + answer) per W&B run::

    --judge_log_dir ./judge_logs \\
    --judge_log_name qwen35_27b_q8

``--judge_log_name`` defaults to a slug of the GGUF filename; use a distinct name when comparing judges.

VRAM / GGUF notes (80GB A100)
-----------------------------
- Llama-3-8B bf16 + LoRA + CBL is released before loading the judge; peak judge VRAM is from GGUF.
- **Recommended quant**: Q8_0 (~28.6 GB) is the sweet spot for A100 80GB — accurate and fits
  comfortably. Q6_K (~22.5 GB) or Q5_K_M (~19.6 GB) also work well.
- **Avoid BF16**: True BF16 GGUF is ~54 GB weights alone, is split across 2 shards, and with
  KV-cache can OOM on 80 GB. Only use if you have >80 GB VRAM headroom.
- **Token budget**: ``--judge_max_tokens`` caps *generation* (reasoning + final line). Default 2048
  reduces truncation before ``SCORE: N``; lower if OOM.
- **Prompt size**: ``--judge_text_max_chars`` truncates the candidate text so prompt+thinking stays
  in context.

Thinking mode (Qwen3.5)
-----------------------
Qwen3.5 thinks by default (``<think>...</think>`` block before the answer). The ``enable_thinking``
Jinja template variable controls this. This script passes it directly to the chat template handler.

**Important**: Qwen3.5 does NOT support the ``/think`` / ``/no_think`` user-message suffixes
that Qwen3 uses — those are ignored or treated as literal text.

llama.cpp notes: https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re

import numpy as np
import torch
import wandb

try:
    import config_finegrained as CFG
except ImportError:
    import config as CFG
from transformers import AutoTokenizer, LlamaConfig

from steerability_cache import (
    save_all_steerability_texts,
    sanitize_concept_slug,
    steerability_output_root,
)
from resume_steerability_test import (
    find_eval_checkpoint,
    generate_steerability_texts,
    get_llama_vocab_weight,
    infer_run_layout,
    load_model_and_cbl,
    set_seed,
)


# Rubric: steerability (concept alignment) is primary; fluency is secondary (see Qwen thinking docs link in module docstring).
JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for *steerability*: a language model was intervened so its generation should align with one target concept.

Reason briefly first (any built-in thinking channel is fine), then give your judgment. **Keep reasoning short:** aim for a few tight sentences or a tiny bullet list—do not write long numbered outlines or essays. You must leave enough of your output budget for the mandatory score line.

Scoring rubric for ONE integer 1–10 (higher = better *steerability*):
- **Primary (~70–80% of the score):** Does the text clearly reflect, argue for, or exemplify the named concept? Would a reader correctly infer the concept from the text alone? Penalize off-topic, contradictory, or concept-orthogonal content heavily.
- **Secondary (~20–30%):** Basic readability: grossly ungrammatical or incoherent text may cap the score even if fragments hint at the concept, because alignment cannot be trusted.

Do NOT reward length, style, or eloquence beyond what is needed to judge concept alignment.

Your **final line of the entire reply** MUST be exactly (no extra characters on that line):
SCORE: N
where N is an integer from 1 through 10. Nothing may follow that line."""


def _sanitize_path_component(s: str, max_len: int = 120) -> str:
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s).strip())
    t = t.strip("_") or "unnamed"
    return t[:max_len]


def default_judge_log_name_from_gguf(judge_gguf_path: str) -> str:
    """Filesystem-safe label derived from GGUF path or ``repo::file.gguf`` spec."""
    _, _, fn = judge_gguf_path.partition("::")
    base = fn.strip() if fn else judge_gguf_path
    base = os.path.basename(base.rstrip("/"))
    if base.lower().endswith(".gguf"):
        base = base[: -5]
    return _sanitize_path_component(base, max_len=80)


def write_judge_reasoning_log(
    *,
    judge_log_run_dir: str,
    concept_idx: int,
    concept_name: str,
    sample_idx: int,
    run_id: str,
    enable_thinking: bool,
    parsed_score: int | None,
    raw_assistant_output: str,
) -> None:
    """
    Append-free log: one UTF-8 file per sample. Overwrites on re-run (not a cache).
    Body is the full model output (thinking tags + visible answer).
    """
    sub = os.path.join(
        judge_log_run_dir,
        f"c{concept_idx:03d}_{sanitize_concept_slug(concept_name)}",
    )
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"sample_{sample_idx:04d}.txt")
    score_line = "null" if parsed_score is None else str(parsed_score)
    header = (
        "# judge_reasoning_log v1\n"
        f"run_id: {run_id}\n"
        f"concept_idx: {concept_idx}\n"
        f"concept_name: {json.dumps(concept_name, ensure_ascii=False)}\n"
        f"sample_idx: {sample_idx}\n"
        f"parsed_score: {score_line}\n"
        f"enable_thinking: {str(enable_thinking).lower()}\n"
        "---\n"
    )
    body = raw_assistant_output if raw_assistant_output is not None else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(body)


def _release_generation_vram():
    """Drop cached Llama lm_head weights from resume_steerability_test and clear CUDA cache."""
    import resume_steerability_test as rst

    rst._CACHED_LLAMA_VOCAB_WEIGHT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def parse_score_1_to_10(assistant_text: str):
    """Parse SCORE: N (1..10) from judge output; tolerates thinking / markdown."""
    if not assistant_text:
        return None
    t = assistant_text.strip()
    for m in reversed(list(re.finditer(r"(?i)SCORE:\s*(\d{1,2})\b", t))):
        v = int(m.group(1))
        if 1 <= v <= 10:
            return v
    for line in reversed(t.splitlines()):
        line = line.strip()
        m = re.match(r"^(\d{1,2})$", line)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                return v
    return None


def load_llama_cpp_llm(
    gguf_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_batch: int,
    verbose: bool,
):
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is required for GGUF judging. "
            "Install e.g. `pip install llama-cpp-python` (use a CUDA wheel/build for GPU)."
        ) from e

    common_kwargs = dict(n_ctx=n_ctx, n_batch=n_batch, n_gpu_layers=n_gpu_layers, verbose=verbose)

    if os.path.isfile(gguf_path):
        print(f"Loading GGUF judge (local): {gguf_path}")
        print(f"  n_ctx={n_ctx} n_gpu_layers={n_gpu_layers} n_batch={n_batch}")
        return Llama(model_path=gguf_path, **common_kwargs)

    if "/" in gguf_path and not os.path.sep == "/" or not os.path.exists(gguf_path):
        repo_id, _, filename = gguf_path.partition("::")
        if not filename:
            repo_id = gguf_path
            filename = None
        print(f"Loading GGUF judge via from_pretrained: repo_id={repo_id} filename={filename}")
        print(f"  n_ctx={n_ctx} n_gpu_layers={n_gpu_layers} n_batch={n_batch}")

        fp_kwargs: dict = {}
        if filename:
            fp_kwargs["filename"] = filename
            m = re.match(r"^(.*)-(\d+)-of-(\d+)\.gguf$", filename)
            if m:
                prefix, _idx, total = m.group(1), int(m.group(2)), int(m.group(3))
                if total > 1:
                    fp_kwargs["additional_files"] = [
                        f"{prefix}-{str(i).zfill(len(m.group(2)))}-of-{m.group(3)}.gguf"
                        for i in range(2, total + 1)
                    ]
        else:
            fp_kwargs["filename"] = "*.gguf"

        return Llama.from_pretrained(repo_id=repo_id, **fp_kwargs, **common_kwargs)

    raise FileNotFoundError(f"GGUF not found (local file or HF repo): {gguf_path}")


def _strip_think_blocks(text: str) -> str:
    """Strip Qwen/llama.cpp thinking wrappers so SCORE parsing sees the visible answer."""
    t = text or ""
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<redacted_thinking>.*?</redacted_thinking>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"</redacted_thinking>", "", t, flags=re.DOTALL)
    return t.strip()


def _get_chat_handler(llm):
    """Resolve the chat completion handler from a Llama instance.

    Calling the handler directly (instead of create_chat_completion) lets us
    forward extra kwargs (e.g. ``enable_thinking``) through the Jinja2 chat
    template renderer.
    """
    from llama_cpp import llama_chat_format

    return (
        llm.chat_handler
        or llm._chat_handlers.get(llm.chat_format)
        or llama_chat_format.get_chat_completion_handler(llm.chat_format)
    )


def judge_one(
    llm,
    concept_name: str,
    text: str,
    max_chars: int,
    max_tokens: int,
    enable_thinking: bool = True,
) -> tuple[int | None, str]:
    body = text[:max_chars] if text else ""
    user_msg = (
        f'Target concept (the generator was *steered* toward this concept): "{concept_name}"\n\n'
        f"Generated text to evaluate (single sample):\n{body}\n\n"
        "Give one overall 1–10 score for *steerability* (concept alignment), using the rubric in the system message. "
        "Be concise: short reasoning, then your **last line** must be exactly `SCORE: N` with N from 1 to 10."
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Low temperature helps follow the SCORE: N contract; Qwen3.5 defaults are higher for chat.
    judge_temperature = 0.2
    handler = _get_chat_handler(llm)
    try:
        out = handler(
            llama=llm,
            messages=messages,
            max_tokens=max_tokens,
            temperature=judge_temperature,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=judge_temperature,
        )

    content = ""
    try:
        content = out["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        content = str(out)

    cleaned = _strip_think_blocks(content)
    score = parse_score_1_to_10(cleaned)
    return score, content


def run_judge_on_texts(
    llm,
    decoded_texts_by_concept,
    concept_set,
    max_chars: int,
    max_tokens: int,
    enable_thinking: bool = True,
    judge_log_run_dir: str | None = None,
    wandb_run_id: str = "",
):
    all_scores: list[float] = []
    per_concept: dict = {}
    parse_ok = 0
    parse_fail = 0

    for concept_idx, concept_name in enumerate(concept_set):
        texts = decoded_texts_by_concept[concept_idx] if concept_idx < len(decoded_texts_by_concept) else []
        if not texts:
            per_concept[concept_name] = {"n": 0, "judge_mean_1_10": float("nan"), "judge_std_1_10": 0.0}
            continue

        scores_this: list[float] = []
        for b, t in enumerate(texts):
            s, raw = judge_one(llm, concept_name, t, max_chars, max_tokens, enable_thinking=enable_thinking)
            if judge_log_run_dir:
                write_judge_reasoning_log(
                    judge_log_run_dir=judge_log_run_dir,
                    concept_idx=concept_idx,
                    concept_name=concept_name,
                    sample_idx=b,
                    run_id=wandb_run_id,
                    enable_thinking=enable_thinking,
                    parsed_score=s,
                    raw_assistant_output=raw or "",
                )
            if s is None:
                parse_fail += 1
                wandb.log(
                    {
                        f"judge_sample_{concept_name}_{b + 1}": t[:2000],
                        f"judge_score_{concept_name}_{b + 1}": -1.0,
                        f"judge_raw_tail_{concept_name}_{b + 1}": (raw or "")[-1500:],
                    }
                )
            else:
                parse_ok += 1
                all_scores.append(float(s))
                scores_this.append(float(s))
                wandb.log(
                    {
                        f"judge_sample_{concept_name}_{b + 1}": t[:2000],
                        f"judge_score_{concept_name}_{b + 1}": float(s),
                        f"judge_raw_tail_{concept_name}_{b + 1}": (raw or "")[-1500:],
                    }
                )

        if scores_this:
            a = np.array(scores_this, dtype=np.float64)
            per_concept[concept_name] = {
                "n": len(scores_this),
                "judge_mean_1_10": float(a.mean()),
                "judge_std_1_10": float(a.std()) if a.size > 1 else 0.0,
            }
        else:
            per_concept[concept_name] = {"n": len(texts), "judge_mean_1_10": float("nan"), "judge_std_1_10": 0.0}

    if not all_scores:
        g_mean, g_std = float("nan"), 0.0
    else:
        a = np.array(all_scores, dtype=np.float64)
        g_mean, g_std = float(a.mean()), float(a.std()) if a.size > 1 else 0.0

    return {
        "judge_mean_1_10": g_mean,
        "judge_std_1_10": g_std,
        "judge_total_n_scored": len(all_scores),
        "judge_parse_ok": parse_ok,
        "judge_parse_fail": parse_fail,
        "per_concept": per_concept,
    }


def process_run(
    run_id: str,
    expected_dataset: str,
    seed: int,
    wandb_project: str,
    wandb_entity: str | None,
    judge_gguf_path: str,
    judge_n_ctx: int,
    judge_max_tokens: int,
    judge_n_gpu_layers: int,
    judge_n_batch: int,
    judge_text_max_chars: int,
    judge_chat_template_kwargs_json: str | None,
    judge_disable_thinking: bool,
    judge_llama_verbose: bool,
    gen_device_str: str,
    judge_log_dir: str | None = None,
    judge_log_name: str | None = None,
    samples_per_concept: int | None = None,
    run_idx: int | None = None,
    total_runs: int | None = None,
):
    set_seed(seed)

    print(f"\n{'='*60}")
    if run_idx is not None and total_runs is not None:
        print(f"Processing run {run_idx}/{total_runs}: {run_id} (seed={seed})")
    else:
        print(f"Processing run: {run_id} (seed={seed})")
    print(f"{'='*60}")

    api = wandb.Api()
    run_path = f"{wandb_entity}/{wandb_project}/{run_id}" if wandb_entity else f"{wandb_project}/{run_id}"
    try:
        original_run = api.run(run_path)
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return None

    run_config = original_run.config
    dataset = run_config.get("dataset", "SetFit/sst2")
    discrimination_loss = run_config.get("discrimination_loss", 1.0)
    arch_type = run_config.get("arch_type", None)
    residual_dim = run_config.get("residual_dim", 768)
    add_llama_logits = bool(run_config.get("add_llama_logits", False))

    if dataset != expected_dataset:
        print(f"SKIPPING run {run_id}: dataset mismatch ('{dataset}' vs expected '{expected_dataset}').")
        return None

    run_type, ckpt_prefix = infer_run_layout(run_id, dataset, run_config)
    if run_type is None or ckpt_prefix is None:
        print(f"Could not infer checkpoint layout for run {run_id}")
        return None

    peft_path, cbl_path, best_epoch, is_low_score = find_eval_checkpoint(ckpt_prefix, run_type, dataset)
    if best_epoch is None:
        print(f"No model weights found for run {run_id}")
        return None

    enable_thinking = not judge_disable_thinking
    if judge_chat_template_kwargs_json:
        parsed = json.loads(judge_chat_template_kwargs_json)
        if "enable_thinking" in parsed:
            enable_thinking = bool(parsed["enable_thinking"])

    wandb.init(project=wandb_project, entity=wandb_entity, id=run_id, resume="must")

    gen_device = torch.device(gen_device_str)
    # resume_steerability_test uses module-level `device` for generation
    import resume_steerability_test as rst

    rst.device = gen_device

    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    concept_set = CFG.concept_set.get(dataset, CFG.concepts_from_labels[dataset])
    n_samples = max(1, samples_per_concept) if samples_per_concept is not None else max(1, 100 // len(concept_set))
    print(f"Samples per concept: {n_samples}" + (" (from --samples_per_concept)" if samples_per_concept is not None else " (default: 100 // num_concepts)"))

    if arch_type is not None:
        d_load = 1.0 if arch_type == "non_residual" else 0.0
    else:
        d_load = discrimination_loss

    decoded_texts_by_concept = None
    results: dict = {}
    preLM, cbl = None, None
    steer_dir = steerability_output_root(ckpt_prefix, best_epoch, is_low_score)
    print(f"Steerability sample cache: {steer_dir}")

    # ----- Phase 1: Llama 8B + CBL, batched steerability gen -----
    try:
        print(f"\n[Phase 1] Generation on {gen_device} — epoch {best_epoch}")
        preLM, cbl = load_model_and_cbl(
            peft_path, cbl_path, config, concept_set, tokenizer, d_load, residual_dim
        )
        llama_vocab_weight = get_llama_vocab_weight() if add_llama_logits else None
        decoded_texts_by_concept = generate_steerability_texts(
            preLM,
            cbl,
            tokenizer,
            concept_set,
            dataset,
            samples_per_concept=n_samples,
            print_k=3,
            llama_vocab_weight=llama_vocab_weight,
            steerability_cache_dir=steer_dir,
            steerability_cache_seed=seed,
        )
        save_all_steerability_texts(steer_dir, seed, concept_set, decoded_texts_by_concept)
    except Exception as e:
        print(f"[Phase 1] Error: {e}")
        import traceback

        traceback.print_exc()
        wandb.finish()
        return None
    finally:
        if preLM is not None:
            del preLM
        if cbl is not None:
            del cbl
        _release_generation_vram()
        print("[Phase 1] Released Llama/CBL (and lm_head cache).")

    # ----- Phase 2: GGUF judge only -----
    llm = None
    try:
        print(f"\n[Phase 2] GGUF judge — n_ctx={judge_n_ctx} max_tokens={judge_max_tokens}")
        llm = load_llama_cpp_llm(
            judge_gguf_path,
            n_ctx=judge_n_ctx,
            n_gpu_layers=judge_n_gpu_layers,
            n_batch=judge_n_batch,
            verbose=judge_llama_verbose,
        )
        judge_log_run_dir = None
        if judge_log_dir:
            jname = judge_log_name or default_judge_log_name_from_gguf(judge_gguf_path)
            judge_log_run_dir = os.path.join(
                os.path.abspath(judge_log_dir),
                jname,
                _sanitize_path_component(run_id),
            )
            os.makedirs(judge_log_run_dir, exist_ok=True)
            print(f"Judge reasoning logs (overwrite each run): {judge_log_run_dir}")

        metrics = run_judge_on_texts(
            llm,
            decoded_texts_by_concept,
            concept_set,
            max_chars=judge_text_max_chars,
            max_tokens=judge_max_tokens,
            enable_thinking=enable_thinking,
            judge_log_run_dir=judge_log_run_dir,
            wandb_run_id=run_id,
        )

        def _fmt(x):
            return f"{x:.4f}" if x == x else "nan"

        print(
            f"  judge_mean_1_10={_fmt(metrics['judge_mean_1_10'])} "
            f"judge_std_1_10={_fmt(metrics['judge_std_1_10'])} "
            f"n_scored={metrics['judge_total_n_scored']} "
            f"parse_ok={metrics['judge_parse_ok']} fail={metrics['judge_parse_fail']}"
        )

        log_payload = {
            "judge_mean_1_10": metrics["judge_mean_1_10"],
            "judge_std_1_10": metrics["judge_std_1_10"],
            "judge_total_n_scored": metrics["judge_total_n_scored"],
            "judge_parse_ok": metrics["judge_parse_ok"],
            "judge_parse_fail": metrics["judge_parse_fail"],
            "judge_metric_epoch": best_epoch,
            "judge_metric_run_type": run_type,
            "judge_metric_low_score_checkpoint": is_low_score,
            "judge_gguf_path": judge_gguf_path,
            "judge_n_ctx": judge_n_ctx,
            "judge_max_tokens": judge_max_tokens,
        }
        for cname, row in metrics["per_concept"].items():
            safe = cname.replace(" ", "_").replace("/", "_")[:80]
            if row["n"] > 0:
                log_payload[f"judge_mean_1_10_{safe}"] = row["judge_mean_1_10"]
                log_payload[f"judge_std_1_10_{safe}"] = row["judge_std_1_10"]

        wandb.log(log_payload)
        results = {
            **metrics,
            "epoch": best_epoch,
            "run_type": run_type,
            "low_score_checkpoint": is_low_score,
        }
        _js = {
            "judge_mean_1_10": metrics["judge_mean_1_10"],
            "judge_std_1_10": metrics["judge_std_1_10"],
            "judge_total_n_scored": metrics["judge_total_n_scored"],
            "judge_parse_ok": metrics["judge_parse_ok"],
            "judge_parse_fail": metrics["judge_parse_fail"],
            "per_concept": metrics["per_concept"],
        }
        wandb.log({"judge_metric_results_summary": _js})
    except Exception as e:
        print(f"[Phase 2] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wandb.finish()
    print(f"\nCompleted run {run_id}")
    return results if results else None


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    p = argparse.ArgumentParser(description="Steerability gen (Llama 8B) then Qwen GGUF 1–10 judge.")
    p.add_argument("--run_ids_pickle", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default="cbm-generation-new")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen_device", type=str, default="cuda:0", help="CUDA device for Llama-3-8B + CBL generation.")
    p.add_argument(
        "--samples_per_concept",
        type=int,
        default=None,
        help="Steerability samples per concept. If omitted, max(1, 100 // num_concepts).",
    )
    p.add_argument(
        "--judge_gguf_path",
        type=str,
        required=True,
        help=(
            "Local path to GGUF, or HF spec 'repo_id::filename' for auto-download. "
            "Q8_0 recommended (~28.6 GB, fits A100 80GB). "
            "E.g. /data/Qwen3.5-27B-Q8_0.gguf or 'unsloth/Qwen3.5-27B-GGUF::Qwen3.5-27B-Q8_0.gguf'."
        ),
    )
    p.add_argument(
        "--judge_n_ctx",
        type=int,
        default=6144,
        help="Context length for GGUF. 6144 is a practical default on 80GB after freeing 8B; lower if OOM.",
    )
    p.add_argument(
        "--judge_max_tokens",
        type=int,
        default=2048,
        help="Max new tokens per judgment (thinking + SCORE line). 2048 avoids truncating verbose judges; reduce if OOM.",
    )
    p.add_argument("--judge_n_gpu_layers", type=int, default=-1, help="-1 = all layers on GPU (llama.cpp).")
    p.add_argument("--judge_n_batch", type=int, default=256, help="llama.cpp physical batch size.")
    p.add_argument(
        "--judge_text_max_chars",
        type=int,
        default=1200,
        help="Truncate generated text in the judge prompt to save ctx for thinking.",
    )
    p.add_argument(
        "--judge_chat_template_kwargs_json",
        type=str,
        default=None,
        help='Override enable_thinking via JSON, e.g. \'{"enable_thinking": false}\'. Takes precedence over --judge_disable_thinking.',
    )
    p.add_argument(
        "--judge_disable_thinking",
        action="store_true",
        help="Set enable_thinking false (ignored if --judge_chat_template_kwargs_json is set).",
    )
    p.add_argument("--judge_llama_verbose", action="store_true", help="Verbose llama.cpp loader.")
    p.add_argument(
        "--judge_log_dir",
        type=str,
        default=None,
        help=(
            "If set, write full judge outputs (reasoning + answer) to disk under "
            "``{dir}/{judge_name}/{wandb_run_id}/cXXX_slug/sample_YYYY.txt`` (overwrites on re-run; not a cache)."
        ),
    )
    p.add_argument(
        "--judge_log_name",
        type=str,
        default=None,
        help=(
            "Subfolder name for this judge (e.g. qwen35_27b_q8 vs another GGUF). "
            "Default: derived from --judge_gguf_path basename."
        ),
    )
    args = p.parse_args()

    with open(args.run_ids_pickle, "rb") as f:
        run_ids = pickle.load(f)

    print(f"Runs: {len(run_ids)} | gen_device={args.gen_device} | GGUF={args.judge_gguf_path}")

    all_out = {}
    for idx, rid in enumerate(run_ids, start=1):
        all_out[rid] = process_run(
            rid,
            args.dataset,
            args.seed,
            args.wandb_project,
            args.wandb_entity,
            args.judge_gguf_path,
            args.judge_n_ctx,
            args.judge_max_tokens,
            args.judge_n_gpu_layers,
            args.judge_n_batch,
            args.judge_text_max_chars,
            args.judge_chat_template_kwargs_json,
            args.judge_disable_thinking,
            args.judge_llama_verbose,
            args.gen_device,
            judge_log_dir=args.judge_log_dir,
            judge_log_name=args.judge_log_name,
            samples_per_concept=args.samples_per_concept,
            run_idx=idx,
            total_runs=len(run_ids),
        )

    print("\n" + "=" * 60)
    for rid, res in all_out.items():
        m = res.get("judge_mean_1_10") if res else None
        print(f"{rid}: judge_mean_1_10={m}")


if __name__ == "__main__":
    main()
