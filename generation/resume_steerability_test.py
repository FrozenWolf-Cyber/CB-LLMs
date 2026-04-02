import argparse
import os
import pickle
import glob
import torch
import numpy as np
import evaluate
from tqdm.auto import tqdm
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast, AutoModelForCausalLM
from modules import CBLResidual, CBL, Roberta_classifier
import wandb


_CACHED_LLAMA_VOCAB_WEIGHT = None


def get_llama_vocab_weight():
    global _CACHED_LLAMA_VOCAB_WEIGHT
    if _CACHED_LLAMA_VOCAB_WEIGHT is not None:
        return _CACHED_LLAMA_VOCAB_WEIGHT

    lm_head_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B',
        torch_dtype=torch.bfloat16,
    ).to(device)
    _CACHED_LLAMA_VOCAB_WEIGHT = lm_head_model.get_output_embeddings().weight.detach()
    del lm_head_model
    torch.cuda.empty_cache()
    return _CACHED_LLAMA_VOCAB_WEIGHT


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def infer_run_layout(run_id, dataset, run_config):
    """
    Infer whether this run is from train_combined.py (cbm) or train_grpo.py (grpo),
    and return the corresponding checkpoint prefix.
    """
    d_name = dataset.replace('/', '_')
    cbm_prefix = f"./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/"
    grpo_prefix = f"./from_pretained_llama3_lora_grpo_{run_id}/{d_name}/"

    cbm_exists = os.path.isdir(cbm_prefix)
    grpo_exists = os.path.isdir(grpo_prefix)

    if cbm_exists and not grpo_exists:
        return "cbm", cbm_prefix
    if grpo_exists and not cbm_exists:
        return "grpo", grpo_prefix

    # If both/neither exist, use config hints.
    if "grpo_epochs" in run_config and "pretrained_run_id" in run_config:
        return "grpo", grpo_prefix
    if "discrimination_loss" in run_config:
        return "cbm", cbm_prefix

    # Final fallback preference.
    if cbm_exists:
        return "cbm", cbm_prefix
    if grpo_exists:
        return "grpo", grpo_prefix
    return None, None


def parse_epoch_from_path(path, marker):
    basename = os.path.basename(path)
    try:
        return int(basename.replace(marker, "").replace(".pt", ""))
    except Exception:
        return None


def find_eval_checkpoint(prefix, run_type, dataset):
    """
    Return checkpoint paths (peft_path, cbl_path, epoch, is_low_score) for evaluation.

    train_combined.py:
      - SetFit/sst2: prefers cbl_epoch_{best}.pt, fallback cbl_low_score_epoch_{last}.pt
      - others: usually cbl_epoch_{last}.pt

    train_grpo.py:
      - uses cbl_epoch_{epoch}.pt
    """
    if not os.path.isdir(prefix):
        return None, None, None, None

    cbl_best_files = sorted(glob.glob(os.path.join(prefix, "cbl_epoch_*.pt")))
    cbl_low_files = sorted(glob.glob(os.path.join(prefix, "cbl_low_score_epoch_*.pt")))

    best_epoch = None
    is_low_score = False

    if cbl_best_files:
        epochs = [parse_epoch_from_path(f, "cbl_epoch_") for f in cbl_best_files]
        epochs = [e for e in epochs if e is not None]
        if epochs:
            best_epoch = max(epochs)
            is_low_score = False

    # For combined on SetFit/sst2, fallback to low_score if no best checkpoint exists.
    if best_epoch is None and cbl_low_files:
        low_epochs = [parse_epoch_from_path(f, "cbl_low_score_epoch_") for f in cbl_low_files]
        low_epochs = [e for e in low_epochs if e is not None]
        if low_epochs:
            best_epoch = max(low_epochs)
            is_low_score = True

    if best_epoch is None:
        return None, None, None, None

    if is_low_score:
        peft_path = os.path.join(prefix, f"llama3_low_score_epoch_{best_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_low_score_epoch_{best_epoch}.pt")
    else:
        peft_path = os.path.join(prefix, f"llama3_epoch_{best_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_epoch_{best_epoch}.pt")

    if not os.path.isdir(peft_path):
        return None, None, None, None
    if not os.path.isfile(cbl_path):
        return None, None, None, None

    return peft_path, cbl_path, best_epoch, is_low_score


def generate_steerability_texts(preLM, cbl, tokenizer, concept_set, dataset, samples_per_concept, print_k=3, llama_vocab_weight=None):
    """
    Generate steerability texts with batched generation for each concept.
    Returns: list of lists, shape (num_concepts, samples_per_concept)
    """
    if dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100

    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    special_tokens_mask = torch.tensor([128000, 128001]).to(device)
    all_texts = []

    chunk_size = 25

    with torch.no_grad():
        for concept_idx in tqdm(range(len(concept_set)), desc="Steerability generation"):
            v = [0] * len(concept_set)
            v[concept_idx] = intervention_value
            concept_texts = []
            generated_so_far = 0

            while generated_so_far < samples_per_concept:
                current_batch = min(chunk_size, samples_per_concept - generated_so_far)
                text_ids_batch, _ = cbl.generate_batch(
                    input_ids,
                    preLM,
                    num_samples=current_batch,
                    intervene=v,
                    length=50,
                    llama_vocab_weight=llama_vocab_weight,
                )

                for b in range(current_batch):
                    decoded_text_ids = tokenizer.decode(
                        text_ids_batch[b][~torch.isin(text_ids_batch[b], special_tokens_mask)]
                    )
                    concept_texts.append(decoded_text_ids)
                    # Keep key format aligned with train scripts
                    wandb.log({
                        f"steerability_sample_{concept_set[concept_idx]}_{generated_so_far + b + 1}": decoded_text_ids
                    })

                generated_so_far += current_batch

            print(f"Concept '{concept_set[concept_idx]}' sample preview:")
            for k in range(min(print_k, len(concept_texts))):
                print(f"  [{k+1}] {concept_texts[k]}")

            all_texts.append(concept_texts)

    return all_texts


def run_steerability_test_from_texts(decoded_texts_by_concept, roberta_tokenizer, classifier, concept_set):
    """
    Score already generated texts with one classifier and return accuracy dict.
    """
    pred = []
    ref = []
    acc = evaluate.load("accuracy")

    with torch.no_grad():
        for concept_idx, concept_texts in enumerate(tqdm(decoded_texts_by_concept, desc="Steerability scoring")):
            if len(concept_texts) == 0:
                continue

            roberta_enc = roberta_tokenizer(
                concept_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            roberta_input = {
                "input_ids": roberta_enc["input_ids"],
                "attention_mask": roberta_enc["attention_mask"],
            }
            logits = classifier(roberta_input)
            pred.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            ref.extend([concept_idx] * len(concept_texts))

    acc.add_batch(predictions=np.array(pred), references=np.array(ref))
    return acc.compute()


def load_model_and_cbl(peft_path, cbl_path, config, concept_set, tokenizer, discrimination_loss, residual_dim=768):
    """
    Load the LLaMA model with LoRA adapter and CBL module.
    """
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM.load_adapter(peft_path)
    preLM.eval()
    
    if discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), residual_dim, tokenizer).to(device)
    
    state_dict = torch.load(cbl_path, map_location=device)
    try:
        cbl.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: strict load_state_dict failed for {cbl_path}: {e}")
        incompatible = cbl.load_state_dict(state_dict, strict=False)
        print(
            "Falling back to strict=False: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    cbl.eval()
    
    return preLM, cbl


def process_run(run_id, classifier_suffixes, expected_dataset, seed, wandb_project, wandb_entity=None,
                run_idx=None, total_runs=None):
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
    
    # Resume wandb run
    resumed_run = wandb.init(
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

    try:
        preLM, cbl = load_model_and_cbl(
            peft_path, cbl_path, config, concept_set, tokenizer,
            discrimination_loss_for_loading, residual_dim
        )

        # Generate once (batched), then reuse texts for all classifiers.
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

        for clf_idx, classifier in classifiers.items():
            print(f"\n  Testing with classifier idx={clf_idx}...")

            acc = run_steerability_test_from_texts(
                decoded_texts_by_concept,
                roberta_tokenizer,
                classifier,
                concept_set,
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
                run_idx=idx,
                total_runs=total_runs,
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
