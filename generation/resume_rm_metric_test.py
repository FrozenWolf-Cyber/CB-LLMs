"""
Resume wandb runs and score steerability generations with the same Skywork RM path as
train_grpo_finegrained_llm (compute_reward_model_scores).

Mirrors resume_steerability_test.py: run_ids pickle, checkpoint discovery, generation,
wandb resume="must", and summary logging — but replaces RoBERTa steerability accuracy
with reward-model statistics.

Checkpoint layout matches train_combined_finegrained.py, train_grpo_finegrained.py, and
train_grpo_finegrained_llm.py (``from_pretained_llama3_lora_{cbm|grpo}_{run_id}/``).
Uses ``config_finegrained`` when present (same concept_set as config.py today).

Not covered: runs with no saved checkpoints, wrong working directory (paths are
relative to cwd), ``--dataset`` not equal to the run's dataset string, or very old
GRPO runs missing ``arch_type`` (falls back to ``discrimination_loss`` for CBL vs
CBLResidual — usually OK for default residual GRPO).
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

from resume_steerability_test import (
    device,
    find_eval_checkpoint,
    generate_steerability_texts,
    get_llama_vocab_weight,
    infer_run_layout,
    load_model_and_cbl,
    set_seed,
)
from train_grpo_finegrained_llm import compute_reward_model_scores


def load_reward_model(rm_model_name: str, rm_device: torch.device):
    """Load Skywork-style sequence classification RM (same as train_grpo_finegrained_llm)."""
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
    criteria_mode,
    rm_batch_size,
    rm_max_text_len,
    debug=False,
):
    """
    Score each concept's generations with the RM. Min–max is within each concept's batch
    (same as GRPO training).
    Returns dict with per-concept lists and aggregates.
    """
    per_concept = {}
    all_rewards = []

    for concept_idx, concept_name in enumerate(concept_set):
        texts = decoded_texts_by_concept[concept_idx] if concept_idx < len(decoded_texts_by_concept) else []
        if not texts:
            per_concept[concept_name] = {"mean": float("nan"), "std": float("nan"), "n": 0, "rewards": []}
            continue

        rewards, raw_list = compute_reward_model_scores(
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            texts=texts,
            concept_name=concept_name,
            device=rm_device,
            criteria_mode=criteria_mode,
            rm_batch_size=rm_batch_size,
            max_text_len=rm_max_text_len,
            debug=debug and concept_idx == 0,
        )

        arr = np.array(rewards, dtype=np.float64)
        per_concept[concept_name] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size > 1 else 0.0,
            "n": int(arr.size),
            "rewards": rewards,
            "raw_scores": raw_list,
        }
        all_rewards.extend(rewards)

        for b, (text, r) in enumerate(zip(texts, rewards)):
            wandb.log({
                f"rm_metric_sample_{concept_name}_{b + 1}": text,
                f"rm_metric_reward_{concept_name}_{b + 1}": r,
            })

    overall_mean = float(np.mean(all_rewards)) if all_rewards else float("nan")
    overall_std = float(np.std(all_rewards)) if len(all_rewards) > 1 else 0.0

    return {
        "rm_metric_mean_reward": overall_mean,
        "rm_metric_std_reward": overall_std,
        "rm_metric_total_n": len(all_rewards),
        "per_concept": per_concept,
    }


def process_run(
    run_id,
    expected_dataset,
    seed,
    wandb_project,
    wandb_entity,
    rm_model_name,
    rm_criteria_mode,
    rm_batch_size,
    rm_max_text_len,
    rm_device_str,
    debug_rm,
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

    rm_device = torch.device(rm_device_str)
    rm_model, rm_tokenizer = load_reward_model(rm_model_name, rm_device)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="must",
    )

    results = {}
    print(f"\nRM metric — epoch {best_epoch}")
    print(f"  criteria_mode={rm_criteria_mode} rm_batch_size={rm_batch_size} max_text_len={rm_max_text_len}")

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

        samples_per_concept = max(1, 100 // len(concept_set))
        llama_vocab_weight = get_llama_vocab_weight() if add_llama_logits else None
        decoded_texts_by_concept = generate_steerability_texts(
            preLM,
            cbl,
            tokenizer,
            concept_set,
            dataset,
            samples_per_concept=samples_per_concept,
            print_k=3,
            llama_vocab_weight=llama_vocab_weight,
        )

        metrics = run_rm_metrics_from_texts(
            decoded_texts_by_concept,
            concept_set,
            rm_model,
            rm_tokenizer,
            rm_device,
            criteria_mode=rm_criteria_mode,
            rm_batch_size=rm_batch_size,
            rm_max_text_len=rm_max_text_len,
            debug=debug_rm,
        )

        print(f"RM metric mean reward (pooled): {metrics['rm_metric_mean_reward']}")
        for cname, row in metrics["per_concept"].items():
            print(f"  {cname}: mean={row['mean']:.4f} std={row['std']:.4f} n={row['n']}")

        log_payload = {
            "rm_metric_mean_reward": metrics["rm_metric_mean_reward"],
            "rm_metric_std_reward": metrics["rm_metric_std_reward"],
            "rm_metric_total_n": metrics["rm_metric_total_n"],
            "rm_metric_epoch": best_epoch,
            "rm_metric_run_type": run_type,
            "rm_metric_low_score_checkpoint": is_low_score,
            "rm_metric_criteria_mode": rm_criteria_mode,
            "rm_model_name": rm_model_name,
        }
        for cname, row in metrics["per_concept"].items():
            safe = cname.replace(" ", "_").replace("/", "_")[:80]
            log_payload[f"rm_metric_per_concept_mean_{safe}"] = row["mean"]
            log_payload[f"rm_metric_per_concept_std_{safe}"] = row["std"]

        wandb.log(log_payload)

        results = {
            **metrics,
            "epoch": best_epoch,
            "run_type": run_type,
            "low_score_checkpoint": is_low_score,
        }

        del preLM, cbl
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during RM metric test (epoch={best_epoch}): {e}")
        import traceback

        traceback.print_exc()

    if results:
        # Compact summary for tables (strip large nested lists)
        summary = {
            "rm_metric_mean_reward": results.get("rm_metric_mean_reward"),
            "rm_metric_std_reward": results.get("rm_metric_std_reward"),
            "rm_metric_total_n": results.get("rm_metric_total_n"),
            "epoch": results.get("epoch"),
            "per_concept_means": {k: v["mean"] for k, v in results.get("per_concept", {}).items()},
        }
        wandb.log({"rm_metric_results_summary": summary})

    wandb.finish()

    print(f"\nCompleted processing run {run_id}")
    print(f"Results summary: {results.get('rm_metric_mean_reward') if results else None}")

    return results


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description="Resume wandb runs and log Skywork RM metrics on steerability samples.")
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
        "--rm_model_name",
        type=str,
        default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        help="HF id for sequence-classification reward model (same default as train_grpo_finegrained_llm).",
    )
    parser.add_argument(
        "--rm_criteria_mode",
        type=str,
        default="separate",
        choices=["separate", "together", "separate_hybrid", "relevance_only"],
        help="Same modes as --rm_criteria_mode in train_grpo_finegrained_llm.",
    )
    parser.add_argument("--rm_batch_size", type=int, default=0, help="0 = score all texts per concept in one batch.")
    parser.add_argument("--rm_max_text_len", type=int, default=500)
    parser.add_argument(
        "--rm_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the reward model (e.g. cuda:1 to free cuda:0 for LLaMA).",
    )
    parser.add_argument("--debug_rm", action="store_true", help="Verbose RM debug on first concept only.")
    args = parser.parse_args()

    with open(args.run_ids_pickle, "rb") as f:
        run_ids = pickle.load(f)

    print(f"Loaded {len(run_ids)} run IDs from {args.run_ids_pickle}")
    print(f"Run IDs: {run_ids}")
    print(f"Expected dataset: {args.dataset}")
    print(f"RM: {args.rm_model_name} | criteria={args.rm_criteria_mode} | rm_device={args.rm_device}")

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
                args.rm_criteria_mode,
                args.rm_batch_size,
                args.rm_max_text_len,
                args.rm_device,
                args.debug_rm,
                run_idx=idx,
                total_runs=total_runs,
            )
            all_results[run_id] = out
        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All RM metric results (mean reward):")
    print("=" * 60)
    for rid, res in all_results.items():
        m = res.get("rm_metric_mean_reward") if res else None
        print(f"{rid}: {m}")


if __name__ == "__main__":
    main()
