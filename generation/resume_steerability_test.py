"""
Resume wandb runs and run steerability tests with RoBERTa classifiers.

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
from transformers import LlamaConfig, AutoTokenizer, RobertaTokenizerFast

from steerability_cache import save_all_steerability_texts, steerability_output_root
from eval_metrics import (
    find_eval_checkpoint,
    generate_steerability_texts,
    get_llama_vocab_weight,
    infer_run_layout,
    load_model_and_cbl,
    run_steerability_test_from_texts,
    score_steerability_roberta,
    set_seed,
)
from modules import Roberta_classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--run_ids_pickle", type=str, required=True, help="Path to pickle file containing list of wandb run IDs")
parser.add_argument("--wandb_project", type=str, default="cbm-generation-new", help="Wandb project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)")
parser.add_argument("--classifier_weight_suffixes", type=str, default="_seed42,_seed123,_seed456",
                    help="Comma-separated list of classifier weight suffixes to test (e.g., '_seed42,_seed123,_seed456')")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset to test (must match the dataset used in the wandb run). e.g., 'SetFit/sst2', 'ag_news', 'yelp_polarity', 'dbpedia_14'")
parser.add_argument("--seed", type=int, default=42, help="Random seed for steerability test generation")
parser.add_argument(
    "--samples_per_concept",
    type=int,
    default=50,
    help="Steerability samples per concept. Default 50.",
)
parser.add_argument(
    "--interventions_per_batch",
    type=int,
    default=50,
    help="Number of concept interventions to batch together during generation. "
         "Higher values reduce sequential autoregressive loops but increase VRAM usage. (default: 50)",
)


def process_run(run_id, classifier_suffixes, expected_dataset, seed, wandb_project, wandb_entity=None,
                samples_per_concept=None, run_idx=None, total_runs=None, interventions_per_batch=1):
    """
    Process a single wandb run: load config, load models, run steerability tests, and log results.
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
    
    print(f"Dataset: {dataset}, Steerability seed: {seed}")
    
    # Validate dataset matches what we expect
    if dataset != expected_dataset:
        print(f"SKIPPING run {run_id}: dataset mismatch. Run used '{dataset}' but expected '{expected_dataset}'.")
        return
    
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
    
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    # Match train_combined_finegrained.py: concept_set drives CBL output dimension.
    concept_set = CFG.concept_set.get(dataset, CFG.concepts_from_labels[dataset])
    print(f"Concept len: {len(concept_set)}")
    n_samples = max(1, samples_per_concept) if samples_per_concept is not None else max(1, 100 // len(concept_set))
    print(f"Samples per concept: {n_samples}" + (" (from --samples_per_concept)" if samples_per_concept is not None else " (default: 100 // num_concepts)"))
    
    # Load multiple roberta classifiers for steerability test (same order as training scripts)
    classifiers = {}
    classifier_paths = [dataset.replace('/', '_') + "_classifier.pt"]
    for suffix in classifier_suffixes:
        classifier_path = dataset.replace('/', '_') + f"_classifier{suffix}.pt"
        classifier_paths.append(classifier_path)

    for clf_idx, classifier_path in enumerate(classifier_paths):
        if not os.path.exists(classifier_path):
            print(f"Warning: Classifier not found at {classifier_path}, skipping...")
            continue
        
        classifier = Roberta_classifier(len(concept_set)).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        try:
            classifier = torch.compile(classifier)
        except Exception as compile_err:
            print(f"Warning: torch.compile failed for {classifier_path}, using eager mode: {compile_err}")
        classifiers[clf_idx] = classifier
        print(f"Loaded classifier from {classifier_path}")
    
    if not classifiers:
        print(f"No classifiers found for run {run_id}")
        return

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="must"
    )
    
    # Run steerability test (single selected checkpoint) and log keys exactly as training scripts
    results = {}
    print(f"\nTesting epoch {best_epoch}...")
    print(f"Loading model from: {peft_path}")
    print(f"Loading CBL from: {cbl_path}")

    # Select CBL architecture robustly across train_combined.py and train_grpo.py variants.
    if arch_type is not None:
        discrimination_loss_for_loading = 1.0 if arch_type == "non_residual" else 0.0
    else:
        discrimination_loss_for_loading = discrimination_loss

    steer_dir = steerability_output_root(ckpt_prefix, best_epoch, is_low_score)
    print(f"Steerability sample cache: {steer_dir}")

    try:
        preLM, cbl = load_model_and_cbl(
            peft_path, cbl_path, config, concept_set, tokenizer,
            discrimination_loss_for_loading, residual_dim, device,
        )

        llama_vocab_weight = get_llama_vocab_weight(device) if add_llama_logits else None
        decoded_texts_by_concept = generate_steerability_texts(
            preLM,
            cbl,
            tokenizer,
            concept_set,
            dataset,
            device,
            samples_per_concept=n_samples,
            print_k=3,
            llama_vocab_weight=llama_vocab_weight,
            steerability_cache_dir=steer_dir,
            steerability_cache_seed=seed,
            interventions_per_batch=interventions_per_batch,
        )

        for clf_idx, classifier in classifiers.items():
            print(f"\n  Testing with classifier idx={clf_idx}...")

            acc = score_steerability_roberta(
                decoded_texts_by_concept,
                roberta_tokenizer,
                classifier,
                concept_set,
                device,
            )

            print(f"  Steerability test accuracy (epoch={best_epoch}, classifier_idx={clf_idx}): {acc}")

            # Use exactly the key names in train_combined.py / train_grpo.py.
            if clf_idx == 0:
                log_key = "steerability_test_accuracy"
            else:
                log_key = f"steerability_test_accuracy_{clf_idx}"

            wandb.log({log_key: acc})
            results[log_key] = {
                "accuracy": acc,
                "epoch": best_epoch,
                "classifier_idx": clf_idx,
                "run_type": run_type,
                "low_score_checkpoint": is_low_score,
            }

        save_all_steerability_texts(steer_dir, seed, concept_set, decoded_texts_by_concept)

        del preLM, cbl
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error testing epoch {best_epoch}: {e}")
        import traceback
        traceback.print_exc()
    
    # Log summary
    if results:
        wandb.log({"steerability_results_summary": results})
    
    # Finish the resumed run
    wandb.finish()
    
    print(f"\nCompleted processing run {run_id}")
    print(f"Results: {results}")
    
    return results


def main():
    args = parser.parse_args()
    
    # Load run IDs from pickle file
    with open(args.run_ids_pickle, 'rb') as f:
        run_ids = pickle.load(f)
    
    print(f"Loaded {len(run_ids)} run IDs from {args.run_ids_pickle}")
    print(f"Run IDs: {run_ids}")
    
    # Parse classifier weight suffixes
    classifier_suffixes = [s.strip() for s in args.classifier_weight_suffixes.split(',')]
    print(f"Classifier weight suffixes to test: {classifier_suffixes}")
    
    print(f"Expected dataset: {args.dataset}")
    print(f"Steerability seed: {args.seed}")
    print(f"samples_per_concept arg: {args.samples_per_concept}")
    print(f"interventions_per_batch: {args.interventions_per_batch}")
    
    # Process each run
    all_results = {}
    total_runs = len(run_ids)
    for idx, run_id in enumerate(run_ids, start=1):
        runs_left_after_this = total_runs - idx
        print(f"\nStarting run {idx}/{total_runs}. Runs left after this: {runs_left_after_this}")
        try:
            results = process_run(
                run_id,
                classifier_suffixes,
                args.dataset,
                args.seed,
                args.wandb_project,
                args.wandb_entity,
                samples_per_concept=args.samples_per_concept,
                run_idx=idx,
                total_runs=total_runs,
                interventions_per_batch=args.interventions_per_batch,
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
