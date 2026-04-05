"""
Resume wandb runs and score steerability generations with LMUnit (same judge path as
``train_grpo_finegrained_llm`` for ``grpo_reward_mode=llm``).

For each generated text we run **three** LMUnit passes (steer / grammar / combined unit tests),
matching the templates and ``generate`` + parse flow in ``_lmunit_forward_scores``. Unlike
training ``compute_llm_rewards_batch``, there is **no batch min–max**; we report means (and stds)
of the per-sample scores in ``[0, 1]`` from ``(rating - 1) / 4`` (failed parses → 0.0).

Mirrors ``resume_rm_metric_test.py`` for checkpoint discovery, generation, and wandb resume.
"""
import argparse
import os
import pickle

import numpy as np
import torch
import wandb

try:
    import config_finegrained as CFG
except ImportError:
    import config as CFG
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from steerability_cache import save_all_steerability_texts, steerability_output_root
from resume_steerability_test import (
    find_eval_checkpoint,
    generate_steerability_texts,
    get_llama_vocab_weight,
    infer_run_layout,
    load_model_and_cbl,
    set_seed,
)
from train_grpo_finegrained_llm import (
    _LMUNIT_QUERY_TEMPLATE,
    _LMUNIT_UNIT_TEST_COMBINED,
    _LMUNIT_UNIT_TEST_GRAMMAR,
    _LMUNIT_UNIT_TEST_STEER,
    _lmunit_forward_scores,
    cuda_gc,
)


def load_lmunit_judge(model_name: str, device: torch.device):
    """Same loader as train_grpo_finegrained_llm when grpo_reward_mode=llm."""
    print(f"Loading LMUnit judge: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    print(f"  LMUnit device: {device}")
    return model, tokenizer


def run_lmunit_metrics_from_texts(
    decoded_texts_by_concept,
    concept_set,
    lmunit_model,
    lmunit_tokenizer,
    max_text_len: int,
    max_new_tokens: int,
    debug: bool,
):
    """
    Three passes per concept (steer, grammar, combined). Returns pooled means of r01 scores.
    """
    all_steer, all_gram, all_comb = [], [], []
    per_concept = {}

    for concept_idx, concept_name in enumerate(concept_set):
        texts = decoded_texts_by_concept[concept_idx] if concept_idx < len(decoded_texts_by_concept) else []
        if not texts:
            per_concept[concept_name] = {
                "n": 0,
                "lmunit_steer_mean_01": float("nan"),
                "lmunit_grammar_mean_01": float("nan"),
                "lmunit_combined_mean_01": float("nan"),
            }
            continue

        query = _LMUNIT_QUERY_TEMPLATE.format(concept_name=concept_name)
        ut_steer = _LMUNIT_UNIT_TEST_STEER.format(concept_name=concept_name)
        ut_both = _LMUNIT_UNIT_TEST_COMBINED.format(concept_name=concept_name)

        dbg = debug and concept_idx == 0

        s01_s, s15_s, _ = _lmunit_forward_scores(
            lmunit_model,
            lmunit_tokenizer,
            texts,
            query,
            ut_steer,
            max_text_len,
            max_new_tokens,
            dbg,
            debug_tag="steer",
        )
        cuda_gc()

        s01_g, s15_g, _ = _lmunit_forward_scores(
            lmunit_model,
            lmunit_tokenizer,
            texts,
            query,
            _LMUNIT_UNIT_TEST_GRAMMAR,
            max_text_len,
            max_new_tokens,
            dbg,
            debug_tag="grammar",
        )
        cuda_gc()

        s01_c, s15_c, _ = _lmunit_forward_scores(
            lmunit_model,
            lmunit_tokenizer,
            texts,
            query,
            ut_both,
            max_text_len,
            max_new_tokens,
            dbg,
            debug_tag="combined",
        )
        cuda_gc()

        all_steer.extend(s01_s)
        all_gram.extend(s01_g)
        all_comb.extend(s01_c)

        per_concept[concept_name] = {
            "n": len(texts),
            "lmunit_steer_mean_01": float(np.mean(s01_s)) if s01_s else float("nan"),
            "lmunit_grammar_mean_01": float(np.mean(s01_g)) if s01_g else float("nan"),
            "lmunit_combined_mean_01": float(np.mean(s01_c)) if s01_c else float("nan"),
        }

        for b, (t, rs, rg, rc, v5s, v5g, v5c) in enumerate(
            zip(texts, s01_s, s01_g, s01_c, s15_s, s15_g, s15_c)
        ):
            wandb.log(
                {
                    f"lmunit_sample_{concept_name}_{b + 1}": t,
                    f"lmunit_steer_01_{concept_name}_{b + 1}": rs,
                    f"lmunit_grammar_01_{concept_name}_{b + 1}": rg,
                    f"lmunit_combined_01_{concept_name}_{b + 1}": rc,
                    f"lmunit_steer_15_{concept_name}_{b + 1}": v5s if v5s is not None else -1.0,
                    f"lmunit_grammar_15_{concept_name}_{b + 1}": v5g if v5g is not None else -1.0,
                    f"lmunit_combined_15_{concept_name}_{b + 1}": v5c if v5c is not None else -1.0,
                }
            )

    def _ms(xs):
        if not xs:
            return float("nan"), 0.0
        a = np.array(xs, dtype=np.float64)
        return float(a.mean()), float(a.std()) if a.size > 1 else 0.0

    ms, ss = _ms(all_steer)
    mg, sg = _ms(all_gram)
    mc, sc = _ms(all_comb)
    n = len(all_steer)

    return {
        "lmunit_steer_mean_01": ms,
        "lmunit_steer_std_01": ss,
        "lmunit_grammar_mean_01": mg,
        "lmunit_grammar_std_01": sg,
        "lmunit_combined_mean_01": mc,
        "lmunit_combined_std_01": sc,
        "lmunit_total_n": n,
        "per_concept": per_concept,
    }


def process_run(
    run_id,
    expected_dataset,
    seed,
    wandb_project,
    wandb_entity,
    lmunit_model_name,
    lmunit_max_text_len,
    lmunit_max_new_tokens,
    lmunit_device_str,
    debug_lmunit,
    samples_per_concept=None,
    run_idx=None,
    total_runs=None,
):
    set_seed(seed)

    print(f"\n{'='*60}")
    if run_idx is not None and total_runs is not None:
        print(f"Processing run {run_idx}/{total_runs}: {run_id} (seed={seed})")
    else:
        print(f"Processing run: {run_id} (seed={seed})")
    print(f"{'='*60}")

    api = wandb.Api()
    if wandb_entity:
        run_path = f"{wandb_entity}/{wandb_project}/{run_id}"
    else:
        run_path = f"{wandb_project}/{run_id}"

    try:
        original_run = api.run(run_path)
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return None

    run_config = original_run.config
    print(f"Run config: {run_config}")

    dataset = run_config.get("dataset", "SetFit/sst2")
    discrimination_loss = run_config.get("discrimination_loss", 1.0)
    arch_type = run_config.get("arch_type", None)
    residual_dim = run_config.get("residual_dim", 768)
    add_llama_logits = bool(run_config.get("add_llama_logits", False))
    print(f"Add llama logits: {add_llama_logits}")

    if dataset != expected_dataset:
        print(f"SKIPPING run {run_id}: dataset mismatch. Run used '{dataset}' but expected '{expected_dataset}'.")
        return None

    run_type, ckpt_prefix = infer_run_layout(run_id, dataset, run_config)
    if run_type is None or ckpt_prefix is None:
        print(f"Could not infer checkpoint layout for run {run_id}")
        return None

    print(f"Detected run type: {run_type}")
    print(f"Checkpoint prefix: {ckpt_prefix}")

    peft_path, cbl_path, best_epoch, is_low_score = find_eval_checkpoint(ckpt_prefix, run_type, dataset)
    print(f"Evaluation epoch: {best_epoch} (low_score={is_low_score})")

    if best_epoch is None:
        print(f"No model weights found for run {run_id}")
        return None

    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token

    concept_set = CFG.concept_set.get(dataset, CFG.concepts_from_labels[dataset])
    print(f"Concept len: {len(concept_set)}")
    n_samples = max(1, samples_per_concept) if samples_per_concept is not None else max(1, 100 // len(concept_set))
    print(f"Samples per concept: {n_samples}" + (" (from --samples_per_concept)" if samples_per_concept is not None else " (default: 100 // num_concepts)"))

    lm_device = torch.device(lmunit_device_str)
    lmunit_model, lmunit_tokenizer = load_lmunit_judge(lmunit_model_name, lm_device)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="must",
    )

    results = {}
    print(f"\nLMUnit steerability benchmark — epoch {best_epoch} (no batch min–max)")
    print(f"  max_text_len={lmunit_max_text_len} max_new_tokens={lmunit_max_new_tokens}")

    steer_dir = steerability_output_root(ckpt_prefix, best_epoch, is_low_score)
    print(f"Steerability sample cache: {steer_dir}")

    if arch_type is not None:
        discrimination_loss_for_loading = 1.0 if arch_type == "non_residual" else 0.0
    else:
        discrimination_loss_for_loading = discrimination_loss

    try:
        preLM, cbl = load_model_and_cbl(
            peft_path,
            cbl_path,
            config,
            concept_set,
            tokenizer,
            discrimination_loss_for_loading,
            residual_dim,
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

        metrics = run_lmunit_metrics_from_texts(
            decoded_texts_by_concept,
            concept_set,
            lmunit_model,
            lmunit_tokenizer,
            max_text_len=lmunit_max_text_len,
            max_new_tokens=lmunit_max_new_tokens,
            debug=debug_lmunit,
        )

        def _fmt(x):
            return f"{x:.4f}" if x == x else "nan"  # NaN != NaN

        print(
            f"  lmunit_steer_mean_01={_fmt(metrics['lmunit_steer_mean_01'])} "
            f"lmunit_grammar_mean_01={_fmt(metrics['lmunit_grammar_mean_01'])} "
            f"lmunit_combined_mean_01={_fmt(metrics['lmunit_combined_mean_01'])} (n={metrics['lmunit_total_n']})"
        )

        log_payload = {
            "lmunit_steer_mean_01": metrics["lmunit_steer_mean_01"],
            "lmunit_steer_std_01": metrics["lmunit_steer_std_01"],
            "lmunit_grammar_mean_01": metrics["lmunit_grammar_mean_01"],
            "lmunit_grammar_std_01": metrics["lmunit_grammar_std_01"],
            "lmunit_combined_mean_01": metrics["lmunit_combined_mean_01"],
            "lmunit_combined_std_01": metrics["lmunit_combined_std_01"],
            "lmunit_total_n": metrics["lmunit_total_n"],
            "lmunit_metric_epoch": best_epoch,
            "lmunit_metric_run_type": run_type,
            "lmunit_metric_low_score_checkpoint": is_low_score,
            "lmunit_model_name": lmunit_model_name,
        }
        for cname, row in metrics["per_concept"].items():
            safe = cname.replace(" ", "_").replace("/", "_")[:80]
            if row["n"] > 0:
                log_payload[f"lmunit_steer_mean_01_{safe}"] = row["lmunit_steer_mean_01"]
                log_payload[f"lmunit_grammar_mean_01_{safe}"] = row["lmunit_grammar_mean_01"]
                log_payload[f"lmunit_combined_mean_01_{safe}"] = row["lmunit_combined_mean_01"]

        wandb.log(log_payload)

        results = {
            **metrics,
            "epoch": best_epoch,
            "run_type": run_type,
            "low_score_checkpoint": is_low_score,
        }

        save_all_steerability_texts(steer_dir, seed, concept_set, decoded_texts_by_concept)

        del preLM, cbl
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during LMUnit metric test (epoch={best_epoch}): {e}")
        import traceback

        traceback.print_exc()

    if results:
        summary = {
            "lmunit_steer_mean_01": results.get("lmunit_steer_mean_01"),
            "lmunit_grammar_mean_01": results.get("lmunit_grammar_mean_01"),
            "lmunit_combined_mean_01": results.get("lmunit_combined_mean_01"),
            "lmunit_total_n": results.get("lmunit_total_n"),
            "epoch": results.get("epoch"),
            "per_concept": results.get("per_concept"),
        }
        wandb.log({"lmunit_metric_results_summary": summary})

    del lmunit_model
    cuda_gc()
    wandb.finish()

    print(f"\nCompleted processing run {run_id}")
    if results:
        print(
            f"  steer_01={results.get('lmunit_steer_mean_01')} "
            f"grammar_01={results.get('lmunit_grammar_mean_01')} "
            f"combined_01={results.get('lmunit_combined_mean_01')}"
        )

    return results


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        description="Resume wandb runs; LMUnit scores (steer / grammar / combined) on steerability samples."
    )
    parser.add_argument("--run_ids_pickle", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="cbm-generation-new")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples_per_concept",
        type=int,
        default=None,
        help="Steerability samples per concept. If omitted, max(1, 100 // num_concepts).",
    )
    parser.add_argument(
        "--lmunit_model_name",
        type=str,
        default="ContextualAI/LMUnit-qwen2.5-72b",
        help="HF causal LM judge (same default as --grpo_llm_judge_model in train_grpo_finegrained_llm).",
    )
    parser.add_argument(
        "--lmunit_max_text_len",
        type=int,
        default=200,
        help="Max chars of trajectory in LMUnit user message (matches training default).",
    )
    parser.add_argument(
        "--lmunit_max_new_tokens",
        type=int,
        default=64,
        help="Generation budget for LMUnit score decode.",
    )
    parser.add_argument(
        "--lmunit_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for LMUnit judge (policy/LLaMA stays on default cuda from resume_steerability_test).",
    )
    parser.add_argument(
        "--debug_lmunit",
        action="store_true",
        help="Print first-sample prompts/decodes for the first concept only.",
    )
    args = parser.parse_args()

    with open(args.run_ids_pickle, "rb") as f:
        run_ids = pickle.load(f)

    print(f"Loaded {len(run_ids)} run IDs from {args.run_ids_pickle}")
    print(f"LMUnit: {args.lmunit_model_name} | device={args.lmunit_device}")

    all_results = {}
    total_runs = len(run_ids)
    for idx, run_id in enumerate(run_ids, start=1):
        print(f"\nStarting run {idx}/{total_runs}. Runs left after this: {total_runs - idx}")
        try:
            out = process_run(
                run_id,
                args.dataset,
                args.seed,
                args.wandb_project,
                args.wandb_entity,
                args.lmunit_model_name,
                args.lmunit_max_text_len,
                args.lmunit_max_new_tokens,
                args.lmunit_device,
                args.debug_lmunit,
                samples_per_concept=args.samples_per_concept,
                run_idx=idx,
                total_runs=total_runs,
            )
            all_results[run_id] = out
        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All runs — lmunit_*_mean_01 (steer / grammar / combined):")
    print("=" * 60)
    for rid, res in all_results.items():
        if not res:
            print(f"{rid}: None")
            continue
        print(
            f"{rid}: steer={res.get('lmunit_steer_mean_01')} "
            f"gram={res.get('lmunit_grammar_mean_01')} "
            f"comb={res.get('lmunit_combined_mean_01')}"
        )


if __name__ == "__main__":
    main()
