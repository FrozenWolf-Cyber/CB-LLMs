"""CB-LLMs generation evaluation: resume W&B runs and compute perplexity.

This script reads a pickle of W&B run IDs, resumes each run, loads the best
checkpoint (same selection logic as training), generates cached perplexity texts,
then computes perplexity via ``eval_metrics.compute_perplexity``.

The perplexity computation uses the ``evaluate`` library which loads its own
LLM, so we explicitly free the training model from GPU before scoring.
"""
import argparse
import gc
import os
import pickle

import torch
import wandb

try:
    import config_finegrained as CFG
except ImportError:
    import config as CFG
from transformers import LlamaConfig, AutoTokenizer

from eval_metrics import (
    _perplexity_cache_path,
    find_eval_checkpoint,
    generate_perplexity_texts,
    compute_perplexity,
    get_llama_vocab_weight,
    infer_run_layout,
    load_model_and_cbl,
    set_seed,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--run_ids_pickle", type=str, default=None, help="Path to pickle file containing list of wandb run IDs")
parser.add_argument("--run_id", type=str, default=None, help="Single wandb run ID (alternative to --run_ids_pickle)")
parser.add_argument("--wandb_project", type=str, default="cbm-generation-new", help="Wandb project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset to test (must match the dataset used in the wandb run). e.g., 'SetFit/sst2', 'ag_news', 'yelp_polarity', 'dbpedia_14'")
parser.add_argument("--seed", type=int, default=42, help="Random seed for perplexity text generation")
parser.add_argument(
    "--perplexity_n_samples",
    type=int,
    default=100,
    help="Number of generated texts to use for perplexity. Default 100.",
)
parser.add_argument(
    "--clear_cache",
    action="store_true",
    help="Delete cached perplexity texts and regenerate from scratch.",
)
parser.add_argument(
    "--no_wandb",
    action="store_true",
    help="Disable wandb logging (skip resume, all wandb.log calls become no-ops).",
)


def process_run(
    run_id,
    expected_dataset,
    seed,
    wandb_project,
    wandb_entity=None,
    perplexity_n_samples=100,
    run_idx=None,
    total_runs=None,
    clear_cache=False,
    no_wandb=False,
):
    """
    Process a single wandb run: load config, load best checkpoint, generate cached texts,
    and compute perplexity metrics.
    """
    # Reseed at the start of every run for reproducibility
    set_seed(seed)
    
    print(f"\n{'='*60}")
    if run_idx is not None and total_runs is not None:
        runs_left = total_runs - run_idx
        print(f"Processing run {run_idx}/{total_runs}: {run_id} (seed={seed}, remaining={runs_left})")
    else:
        print(f"Processing run: {run_id} (seed={seed})")
    print(f"{'='*60}")
    
    # Initialize wandb API and get run config
    api = wandb.Api()
    if wandb_entity:
        run_path = f"{wandb_entity}/{wandb_project}/{run_id}"
    else:
        run_path = f"{wandb_project}/{run_id}"
    
    try:
        original_run = api.run(run_path)
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return
    
    # Extract config from the run
    run_config = original_run.config
    print(f"Run config: {run_config}")
    
    # Extract necessary parameters
    dataset = run_config.get('dataset', 'SetFit/sst2')
    discrimination_loss = run_config.get('discrimination_loss', 1.0)
    arch_type = run_config.get('arch_type', None)
    residual_dim = run_config.get('residual_dim', 768)

    add_llama_logits = bool(run_config.get('add_llama_logits', False))
    print(f"Add llama logits: {add_llama_logits} (source=wandb_config_default_false)")
    
    print(f"Dataset: {dataset}, Perplexity seed: {seed}")
    
    # Validate dataset matches what we expect
    # if dataset != expected_dataset:
    #     print(f"SKIPPING run {run_id}: dataset mismatch. Run used '{dataset}' but expected '{expected_dataset}'.")
    #     return
    
    # Detect run layout and checkpoint prefix
    run_type, ckpt_prefix = infer_run_layout(run_id, dataset, run_config)
    if run_type is None or ckpt_prefix is None:
        print(f"Could not infer checkpoint layout for run {run_id}")
        return

    print(f"Detected run type: {run_type}")
    print(f"Checkpoint prefix: {ckpt_prefix}")

    peft_path, cbl_path, best_epoch, is_low_score = find_eval_checkpoint(ckpt_prefix, run_type, dataset)
    print(f"Evaluation epoch: {best_epoch} (low_score={is_low_score})")

    if best_epoch is None:
        print(f"No model weights found for run {run_id}")
        return
    
    # Setup tokenizers and config
    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Match train_combined_finegrained.py: concept_set drives CBL output dimension.
    concept_set = CFG.concept_set.get(dataset, CFG.concepts_from_labels[dataset])
    print(f"Concept len: {len(concept_set)}")
    print(f"Perplexity n_samples: {perplexity_n_samples}")

    if no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=run_id,
            resume="must"
        )
    
    # Run perplexity evaluation (single selected checkpoint) and log keys exactly as training scripts
    results = {}
    print(f"\nEvaluating epoch {best_epoch}...")
    print(f"Loading model from: {peft_path}")
    print(f"Loading CBL from: {cbl_path}")

    # Select CBL architecture robustly across train_combined.py and train_grpo.py variants.
    if arch_type is not None:
        discrimination_loss_for_loading = 1.0 if arch_type == "non_residual" else 0.0
    else:
        discrimination_loss_for_loading = discrimination_loss

    # Use the run prefix for caching generated perplexity texts.
    cache_dir = os.path.normpath(str(ckpt_prefix).rstrip("/"))
    print(f"Perplexity text cache dir: {cache_dir}")

    if clear_cache:
        cache_file = _perplexity_cache_path(cache_dir, seed)
        if os.path.isfile(cache_file):
            print(f"--clear_cache: removing perplexity cache at {cache_file}")
            os.remove(cache_file)
            print("  Cache cleared. Texts will be regenerated.")
        else:
            print(f"--clear_cache: no existing cache file at {cache_file}")

    try:
        preLM, cbl = load_model_and_cbl(
            peft_path, cbl_path, config, concept_set, tokenizer,
            discrimination_loss_for_loading, residual_dim, device,
        )

        llama_vocab_weight = get_llama_vocab_weight(device) if add_llama_logits else None
        run_name = f"{run_id}_epoch{best_epoch}"

        ppl_texts = generate_perplexity_texts(
            cbl,
            preLM,
            tokenizer,
            seed,
            device,
            n_samples=perplexity_n_samples,
            cache_dir=cache_dir,
            run_name=run_name,
            llama_vocab_weight=llama_vocab_weight,
        )

        # Free model from GPU before evaluate loads its own LLM.
        del preLM, cbl
        gc.collect()
        torch.cuda.empty_cache()

        ppl_results = compute_perplexity(ppl_texts)
        results.update({
            "perplexity": {
                **ppl_results,
                "epoch": best_epoch,
                "run_type": run_type,
                "low_score_checkpoint": is_low_score,
            }
        })

    except Exception as e:
        print(f"Error evaluating epoch {best_epoch}: {e}")
        import traceback
        traceback.print_exc()
    
    # Log summary
    if results:
        wandb.log({"perplexity_results_summary": results})
    
    # Finish the resumed run
    wandb.finish()
    
    print(f"\nCompleted processing run {run_id}")
    print(f"Results: {results}")
    
    return results


def main():
    args = parser.parse_args()
    
    if args.run_id is not None:
        run_ids = [args.run_id]
        print(f"Using single run ID: {args.run_id}")
    elif args.run_ids_pickle is not None:
        with open(args.run_ids_pickle, 'rb') as f:
            run_ids = pickle.load(f)
        print(f"Loaded {len(run_ids)} run IDs from {args.run_ids_pickle}")
    else:
        parser.error("Must provide either --run_id or --run_ids_pickle")

    print(f"Run IDs: {run_ids}")
    
    print(f"Expected dataset: {args.dataset}")
    print(f"Perplexity seed: {args.seed}")
    print(f"perplexity_n_samples: {args.perplexity_n_samples}")
    
    # Process each run
    all_results = {}
    total_runs = len(run_ids)
    for idx, run_id in enumerate(run_ids, start=1):
        runs_left_after_this = total_runs - idx
        print(f"\nStarting run {idx}/{total_runs}. Runs left after this: {runs_left_after_this}")
        try:
            results = process_run(
                run_id,
                args.dataset,
                args.seed,
                args.wandb_project,
                args.wandb_entity,
                perplexity_n_samples=args.perplexity_n_samples,
                run_idx=idx,
                total_runs=total_runs,
                clear_cache=args.clear_cache,
                no_wandb=args.no_wandb,
            )
            all_results[run_id] = results
        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All results:")
    print("="*60)
    for run_id, results in all_results.items():
        print(f"{run_id}: {results}")


if __name__ == "__main__":
    main()
