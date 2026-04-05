"""
Resume wandb runs and score steerability generations with a Skywork-style RM for benchmarking.

Uses the same chat templates / user prompts as train_grpo_finegrained_llm (relevance, grammar,
combined), but **no batch min–max**: metrics are sequence-classification logits **clipped to
[-100, 100]**, then averaged across all generated trajectories (logged per criterion as ``rm_*``).

Mirrors resume_steerability_test.py for checkpoint discovery, generation, and wandb resume.
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaConfig

from steerability_cache import save_all_steerability_texts, steerability_output_root
from resume_steerability_test import (
    find_eval_checkpoint,
    generate_steerability_texts,
    get_llama_vocab_weight,
    infer_run_layout,
    load_model_and_cbl,
    set_seed,
)

# Same user turns as train_grpo_finegrained_llm.compute_reward_model_scores (for comparability).
RM_USER_RELEVANCE = "Write a text about the concept: {concept_name}"
RM_USER_GRAMMAR = "Write a grammatically correct and fluent paragraph."
RM_USER_TOGETHER = "Write a grammatically correct and fluent text about the concept: {concept_name}"

RM_LOGIT_CLIP_MIN = -100.0
RM_LOGIT_CLIP_MAX = 100.0


def _make_formatted(rm_tokenizer, user_turn: str, response_text: str, max_text_len: int) -> str:
    conv = [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": response_text[:max_text_len]},
    ]
    formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
    if rm_tokenizer.bos_token and formatted.startswith(rm_tokenizer.bos_token):
        formatted = formatted[len(rm_tokenizer.bos_token) :]
    return formatted


def _raw_logits_for_texts(
    rm_model,
    rm_tokenizer,
    texts,
    user_turn: str,
    device: torch.device,
    rm_batch_size: int,
    max_text_len: int,
):
    """Single RM criterion: one user prompt for all texts; returns logits[:, 0] clipped to [-100, 100]."""
    if not texts:
        return []
    formatted = [_make_formatted(rm_tokenizer, user_turn, t, max_text_len) for t in texts]
    chunk = rm_batch_size if rm_batch_size > 0 else len(formatted)
    all_scores = []
    for start in range(0, len(formatted), chunk):
        chunk_list = formatted[start : start + chunk]
        tokenized = rm_tokenizer(
            chunk_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        with torch.no_grad():
            logits = rm_model(**tokenized).logits
        clipped = logits[:, 0].float().clamp(RM_LOGIT_CLIP_MIN, RM_LOGIT_CLIP_MAX)
        all_scores.extend(clipped.detach().cpu().tolist())
        del tokenized, logits
    return all_scores


def load_reward_model(rm_model_name: str, rm_device: torch.device):
    """Load Skywork-style sequence classification RM (same loader as train_grpo_finegrained_llm)."""
    print(f"Loading reward model: {rm_model_name} ...")
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
    _kwargs = dict(torch_dtype=torch.bfloat16, num_labels=1)
    try:
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            rm_model_name,
            attn_implementation="flash_attention_2",
            **_kwargs,
        )
        print("  Loaded RM with flash_attention_2.")
    except Exception as fa2_err:
        print(f"  flash_attention_2 unavailable ({fa2_err}), falling back to eager attention.")
        rm_model = AutoModelForSequenceClassification.from_pretrained(rm_model_name, **_kwargs)
    rm_model.eval()
    for p in rm_model.parameters():
        p.requires_grad = False
    rm_model.to(rm_device)
    print(f"  RM device: {rm_device}")
    return rm_model, rm_tokenizer


def run_rm_metrics_from_texts(
    decoded_texts_by_concept,
    concept_set,
    rm_model,
    rm_tokenizer,
    rm_device,
    rm_batch_size,
    rm_max_text_len,
):
    """
    For each concept, score generations under three RM prompts (relevance, grammar, together).
    Logits are clipped to [RM_LOGIT_CLIP_MIN, RM_LOGIT_CLIP_MAX]; global metrics = mean (and std) over all trajectories.
    """
    all_rel, all_gram, all_tog = [], [], []
    per_concept = {}

    for concept_idx, concept_name in enumerate(concept_set):
        texts = decoded_texts_by_concept[concept_idx] if concept_idx < len(decoded_texts_by_concept) else []
        if not texts:
            per_concept[concept_name] = {
                "n": 0,
                "rm_relevance_mean": float("nan"),
                "rm_grammar_mean": float("nan"),
                "rm_together_mean": float("nan"),
            }
            continue

        u_rel = RM_USER_RELEVANCE.format(concept_name=concept_name)
        u_tog = RM_USER_TOGETHER.format(concept_name=concept_name)

        rel = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, u_rel, rm_device, rm_batch_size, rm_max_text_len
        )
        gram = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, RM_USER_GRAMMAR, rm_device, rm_batch_size, rm_max_text_len
        )
        tog = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, u_tog, rm_device, rm_batch_size, rm_max_text_len
        )

        all_rel.extend(rel)
        all_gram.extend(gram)
        all_tog.extend(tog)

        per_concept[concept_name] = {
            "n": len(texts),
            "rm_relevance_mean": float(np.mean(rel)) if rel else float("nan"),
            "rm_grammar_mean": float(np.mean(gram)) if gram else float("nan"),
            "rm_together_mean": float(np.mean(tog)) if tog else float("nan"),
        }

        for b, (t, r, g, o) in enumerate(zip(texts, rel, gram, tog)):
            wandb.log(
                {
                    f"rm_sample_{concept_name}_{b + 1}": t,
                    f"rm_relevance_logit_{concept_name}_{b + 1}": r,
                    f"rm_grammar_logit_{concept_name}_{b + 1}": g,
                    f"rm_together_logit_{concept_name}_{b + 1}": o,
                }
            )

    def _ms(xs):
        if not xs:
            return float("nan"), 0.0
        a = np.array(xs, dtype=np.float64)
        return float(a.mean()), float(a.std()) if a.size > 1 else 0.0

    r_m, r_s = _ms(all_rel)
    g_m, g_s = _ms(all_gram)
    t_m, t_s = _ms(all_tog)
    n = len(all_rel)

    return {
        "rm_relevance_mean": r_m,
        "rm_relevance_std": r_s,
        "rm_grammar_mean": g_m,
        "rm_grammar_std": g_s,
        "rm_together_mean": t_m,
        "rm_together_std": t_s,
        "rm_total_n": n,
        "per_concept": per_concept,
    }


def process_run(
    run_id,
    expected_dataset,
    seed,
    wandb_project,
    wandb_entity,
    rm_model_name,
    rm_batch_size,
    rm_max_text_len,
    rm_device_str,
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

    rm_device = torch.device(rm_device_str)
    rm_model, rm_tokenizer = load_reward_model(rm_model_name, rm_device)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="must",
    )

    results = {}
    print(f"\nRM steerability benchmark — epoch {best_epoch} (logits clipped to [{RM_LOGIT_CLIP_MIN}, {RM_LOGIT_CLIP_MAX}], no min–max)")
    print(f"  rm_batch_size={rm_batch_size} max_text_len={rm_max_text_len}")

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

        metrics = run_rm_metrics_from_texts(
            decoded_texts_by_concept,
            concept_set,
            rm_model,
            rm_tokenizer,
            rm_device,
            rm_batch_size=rm_batch_size,
            rm_max_text_len=rm_max_text_len,
        )

        print(
            f"  rm_relevance_mean={metrics['rm_relevance_mean']:.4f} "
            f"rm_grammar_mean={metrics['rm_grammar_mean']:.4f} "
            f"rm_together_mean={metrics['rm_together_mean']:.4f} (n={metrics['rm_total_n']})"
        )

        log_payload = {
            "rm_relevance_mean": metrics["rm_relevance_mean"],
            "rm_relevance_std": metrics["rm_relevance_std"],
            "rm_grammar_mean": metrics["rm_grammar_mean"],
            "rm_grammar_std": metrics["rm_grammar_std"],
            "rm_together_mean": metrics["rm_together_mean"],
            "rm_together_std": metrics["rm_together_std"],
            "rm_total_n": metrics["rm_total_n"],
            "rm_metric_epoch": best_epoch,
            "rm_metric_run_type": run_type,
            "rm_metric_low_score_checkpoint": is_low_score,
            "rm_model_name": rm_model_name,
        }
        for cname, row in metrics["per_concept"].items():
            safe = cname.replace(" ", "_").replace("/", "_")[:80]
            if row["n"] > 0:
                log_payload[f"rm_relevance_mean_{safe}"] = row["rm_relevance_mean"]
                log_payload[f"rm_grammar_mean_{safe}"] = row["rm_grammar_mean"]
                log_payload[f"rm_together_mean_{safe}"] = row["rm_together_mean"]

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
        print(f"Error during RM metric test (epoch={best_epoch}): {e}")
        import traceback

        traceback.print_exc()

    if results:
        summary = {
            "rm_relevance_mean": results.get("rm_relevance_mean"),
            "rm_grammar_mean": results.get("rm_grammar_mean"),
            "rm_together_mean": results.get("rm_together_mean"),
            "rm_total_n": results.get("rm_total_n"),
            "epoch": results.get("epoch"),
            "per_concept": results.get("per_concept"),
        }
        wandb.log({"rm_metric_results_summary": summary})

    wandb.finish()

    print(f"\nCompleted processing run {run_id}")
    if results:
        print(
            f"  relevance={results.get('rm_relevance_mean')} "
            f"grammar={results.get('rm_grammar_mean')} "
            f"together={results.get('rm_together_mean')}"
        )

    return results


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        description="Resume wandb runs; log RM logits clipped to [-100,100] (relevance / grammar / together) on steerability samples."
    )
    parser.add_argument("--run_ids_pickle", type=str, required=True, help="Pickle file with list of wandb run IDs")
    parser.add_argument("--wandb_project", type=str, default="cbm-generation-new")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Expected dataset tag (must match the run). e.g. SetFit/sst2, ag_news",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples_per_concept",
        type=int,
        default=None,
        help="Steerability samples per concept. If omitted, max(1, 100 // num_concepts).",
    )
    parser.add_argument(
        "--rm_model_name",
        type=str,
        default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        help="HF id for sequence-classification reward model.",
    )
    parser.add_argument("--rm_batch_size", type=int, default=0, help="0 = score all texts per chunk in one forward.")
    parser.add_argument("--rm_max_text_len", type=int, default=500)
    parser.add_argument(
        "--rm_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the reward model (e.g. cuda:1).",
    )
    args = parser.parse_args()

    with open(args.run_ids_pickle, "rb") as f:
        run_ids = pickle.load(f)

    print(f"Loaded {len(run_ids)} run IDs from {args.run_ids_pickle}")
    print(f"Run IDs: {run_ids}")
    print(f"Expected dataset: {args.dataset}")
    print(
        f"RM: {args.rm_model_name} | logits clipped to [{RM_LOGIT_CLIP_MIN}, {RM_LOGIT_CLIP_MAX}] "
        f"(relevance, grammar, together) | rm_device={args.rm_device}"
    )

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
                args.rm_model_name,
                args.rm_batch_size,
                args.rm_max_text_len,
                args.rm_device,
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
    print("All runs — rm_relevance_mean / rm_grammar_mean / rm_together_mean (clipped logits):")
    print("=" * 60)
    for rid, res in all_results.items():
        if not res:
            print(f"{rid}: None")
            continue
        print(
            f"{rid}: rel={res.get('rm_relevance_mean')} "
            f"gram={res.get('rm_grammar_mean')} "
            f"tog={res.get('rm_together_mean')}"
        )


if __name__ == "__main__":
    main()
