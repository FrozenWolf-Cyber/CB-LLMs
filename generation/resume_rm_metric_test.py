"""
Resume wandb runs and score steerability generations with a Skywork-style RM for benchmarking.

Uses the same chat templates / user prompts as train_grpo_finegrained_llm (relevance, grammar,
combined), but **no batch min-max**: metrics are sequence-classification logits **clipped to
[-100, 100]**, then averaged across all generated trajectories (logged per criterion as ``rm_*``).

All evaluation logic is centralized in eval_metrics.py.
"""
import argparse
import os
import pickle

import torch
import wandb

try:
    import config_finegrained as CFG
except ImportError:
    import config as CFG
from transformers import LlamaConfig, AutoTokenizer

import shutil

from steerability_cache import save_all_steerability_texts, steerability_output_root
from eval_metrics import (
    RM_LOGIT_CLIP_MIN,
    RM_LOGIT_CLIP_MAX,
    find_eval_checkpoint,
    generate_steerability_texts,
    get_llama_vocab_weight,
    infer_run_layout,
    load_model_and_cbl,
    load_reward_model,
    run_rm_metrics,
    set_seed,
)


def process_run(
    run_id,
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
    interventions_per_batch=1,
    use_label_concepts=False,
    clear_cache=False,
    no_wandb=False,
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
    print(f"Dataset (from W&B run config): {dataset}")
    discrimination_loss = run_config.get("discrimination_loss", 1.0)
    arch_type = run_config.get("arch_type", None)
    residual_dim = run_config.get("residual_dim", 768)
    add_llama_logits = bool(run_config.get("add_llama_logits", False))
    print(f"Add llama logits: {add_llama_logits}")

    gen_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if use_label_concepts:
        concept_set = CFG.concepts_from_labels[dataset]
        print("Concept source: class labels (CFG.concepts_from_labels)")
    else:
        concept_set = CFG.concept_set[dataset]
        print("Concept source: fine-grained concepts (CFG.concept_set)")
    print(f"Concept len: {len(concept_set)}")
    n_samples = max(1, samples_per_concept) if samples_per_concept is not None else max(1, 100 // len(concept_set))
    print(f"Samples per concept: {n_samples}" + (" (from --samples_per_concept)" if samples_per_concept is not None else " (default: 100 // num_concepts)"))

    rm_device = torch.device(rm_device_str)
    rm_model, rm_tokenizer = load_reward_model(rm_model_name, rm_device)

    if no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=run_id,
            resume="must",
            save_code=False,
            settings=wandb.Settings(
                console="off",
                disable_git=True,
                _disable_stats=True,
            ),
        )

    results = {}
    print(f"\nRM steerability benchmark — epoch {best_epoch} (logits clipped to [{RM_LOGIT_CLIP_MIN}, {RM_LOGIT_CLIP_MAX}], no min-max)")
    print(f"  rm_batch_size={rm_batch_size} max_text_len={rm_max_text_len}")

    steer_dir = steerability_output_root(ckpt_prefix, best_epoch, is_low_score)
    print(f"Steerability sample cache: {steer_dir}")

    if clear_cache and os.path.isdir(steer_dir):
        print(f"--clear_cache: removing steerability cache at {steer_dir}")
        shutil.rmtree(steer_dir)
        print("  Cache cleared. Texts will be regenerated.")

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
            gen_device,
        )

        llama_vocab_weight = get_llama_vocab_weight(gen_device) if add_llama_logits else None
        decoded_texts_by_concept = generate_steerability_texts(
            preLM,
            cbl,
            tokenizer,
            concept_set,
            dataset,
            gen_device,
            samples_per_concept=n_samples,
            print_k=3,
            llama_vocab_weight=llama_vocab_weight,
            steerability_cache_dir=steer_dir,
            steerability_cache_seed=seed,
            interventions_per_batch=interventions_per_batch,
        )

        metrics = run_rm_metrics(
            decoded_texts_by_concept,
            concept_set,
            rm_model,
            rm_tokenizer,
            rm_device,
            rm_batch_size=rm_batch_size,
            rm_max_text_len=rm_max_text_len,
        )

        log_payload = {
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
    parser.add_argument("--run_ids_pickle", type=str, default=None, help="Pickle file with list of wandb run IDs")
    parser.add_argument("--run_id", type=str, default=None, help="Single wandb run ID (alternative to --run_ids_pickle)")
    parser.add_argument("--wandb_project", type=str, default="cbm-generation-new")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Deprecated: ignored. Dataset is read from each run's W&B config['dataset']. Kept for backward-compatible CLIs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples_per_concept",
        type=int,
        default=50,
        help="Steerability samples per concept. Default 50.",
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
        "--interventions_per_batch",
        type=int,
        default=50,
        help="Number of concept interventions to batch together during generation. (default: 50)",
    )
    parser.add_argument(
        "--use_label_concepts",
        action="store_true",
        help=(
            "Use class-label concepts from CFG.concepts_from_labels (for train_combined.py-style runs). "
            "If not set, uses fine-grained concepts from CFG.concept_set."
        ),
    )
    parser.add_argument(
        "--rm_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the reward model (e.g. cuda:1).",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Delete cached steerability texts and regenerate from scratch.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging (skip resume, all wandb.log calls become no-ops).",
    )
    args = parser.parse_args()

    if args.run_id is not None:
        run_ids = [args.run_id]
        print(f"Using single run ID: {args.run_id}")
    elif args.run_ids_pickle is not None:
        with open(args.run_ids_pickle, "rb") as f:
            run_ids = pickle.load(f)
        print(f"Loaded {len(run_ids)} run IDs from {args.run_ids_pickle}")
    else:
        parser.error("Must provide either --run_id or --run_ids_pickle")

    print(f"Run IDs: {run_ids}")
    if args.dataset is not None:
        print("Note: --dataset is ignored; using each run's W&B config['dataset'].")
    print(
        f"RM: {args.rm_model_name} | logits clipped to [{RM_LOGIT_CLIP_MIN}, {RM_LOGIT_CLIP_MAX}] "
        f"(relevance, grammar, together) | rm_device={args.rm_device}"
    )
    print(
        "Concept mode: "
        + ("class labels (train_combined.py compatible)" if args.use_label_concepts else "fine-grained concepts")
    )

    all_results = {}
    total_runs = len(run_ids)
    for idx, run_id in enumerate(run_ids, start=1):
        print(f"\nStarting run {idx}/{total_runs}. Runs left after this: {total_runs - idx}")
        try:
            out = process_run(
                run_id,
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
                interventions_per_batch=args.interventions_per_batch,
                use_label_concepts=args.use_label_concepts,
                clear_cache=args.clear_cache,
                no_wandb=args.no_wandb,
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
