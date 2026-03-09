import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBLResidual, CBL, Roberta_classifier
import wandb


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


def get_model_path_for_run(run_id, dataset, epoch, is_best=False):
    """
    Construct the model path based on run_id, dataset, and epoch.
    For SetFit/sst2, best model doesn't have 'low_score' prefix.
    For others, we use last epoch.
    """
    d_name = dataset.replace('/', '_')
    prefix = f"./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/"
    
    model_name = "llama3"
    cbl_name = "cbl"
    
    if is_best:
        # Best model path (no low_score prefix)
        peft_path = prefix + model_name + "_epoch_" + str(epoch)
        cbl_path = prefix + cbl_name + "_epoch_" + str(epoch) + ".pt"
    else:
        # Check if it's a low_score epoch or regular
        best_path = prefix + model_name + "_epoch_" + str(epoch)
        low_score_path = prefix + model_name + "_low_score_epoch_" + str(epoch)
        
        if os.path.exists(best_path):
            peft_path = best_path
            cbl_path = prefix + cbl_name + "_epoch_" + str(epoch) + ".pt"
        elif os.path.exists(low_score_path):
            peft_path = low_score_path
            cbl_path = prefix + cbl_name + "_low_score_epoch_" + str(epoch) + ".pt"
        else:
            return None, None
    
    return peft_path, cbl_path


def find_best_epoch(run_id, dataset):
    """
    Find the best epoch for a given run.
    For SetFit/sst2: find the epoch with best validation loss (the one without 'low_score' prefix)
    For others: return the last available epoch
    """
    d_name = dataset.replace('/', '_')
    prefix = f"./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/"
    
    if not os.path.exists(prefix):
        print(f"Warning: Path {prefix} does not exist")
        return None
    
    model_name = "llama3"
    epochs = CFG.epoch[dataset]
    
    if dataset == 'SetFit/sst2':
        # For SetFit/sst2, find the best epoch (the one saved without 'low_score')
        for e in range(epochs, 0, -1):
            best_path = prefix + model_name + "_epoch_" + str(e)
            if os.path.exists(best_path):
                return e
        # Fallback: check for any available epoch
        for e in range(epochs, 0, -1):
            low_score_path = prefix + model_name + "_low_score_epoch_" + str(e)
            if os.path.exists(low_score_path):
                print(f"Warning: Only found low_score model for epoch {e}")
                return e
    else:
        # For other datasets, return the last available epoch
        for e in range(epochs, 0, -1):
            epoch_path = prefix + model_name + "_epoch_" + str(e)
            if os.path.exists(epoch_path):
                return e
    
    return None


def find_all_available_epochs(run_id, dataset):
    """
    Find all available epochs for a given run.
    """
    d_name = dataset.replace('/', '_')
    prefix = f"./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/"
    
    if not os.path.exists(prefix):
        return []
    
    model_name = "llama3"
    epochs = CFG.epoch[dataset]
    available_epochs = []
    
    for e in range(1, epochs + 1):
        best_path = prefix + model_name + "_epoch_" + str(e)
        low_score_path = prefix + model_name + "_low_score_epoch_" + str(e)
        if os.path.exists(best_path) or os.path.exists(low_score_path):
            available_epochs.append(e)
    
    return available_epochs


def run_steerability_test(preLM, cbl, tokenizer, roberta_tokenizer, classifier, concept_set, dataset, seed=42):
    """
    Run steerability test and return accuracy.
    """
    set_seed(seed)
    
    if dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100
    
    pred = []
    text = []
    acc = evaluate.load("accuracy")
    
    with torch.no_grad():
        for i in tqdm(range(100 // len(concept_set)), desc="Steerability test"):
            input_ids = torch.tensor([tokenizer.encode("")]).to(device)
            for j in range(len(concept_set)):
                v = [0] * len(concept_set)
                v[j] = intervention_value
                text_ids, _ = cbl.generate(input_ids, preLM, intervene=v)
                decoded_text_ids = tokenizer.decode(
                    text_ids[0][~torch.isin(text_ids[0], torch.tensor([128000, 128001]).to(device))]
                )
                text.append(decoded_text_ids)
                roberta_text_ids = torch.tensor([roberta_tokenizer.encode(decoded_text_ids)]).to(device)
                roberta_input = {
                    "input_ids": roberta_text_ids,
                    "attention_mask": torch.tensor([[1] * roberta_text_ids.shape[1]]).to(device)
                }
                logits = classifier(roberta_input)
                pred.append(logits)
    
    pred = torch.cat(pred, dim=0).detach().cpu()
    pred = np.argmax(pred.numpy(), axis=-1)
    acc.add_batch(predictions=pred, references=list(range(len(concept_set))) * (100 // len(concept_set)))
    
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
    
    cbl.load_state_dict(torch.load(cbl_path, map_location=device), strict=False)
    ### warn missing keys:
    for name, param in cbl.named_parameters():
        if name not in cbl.state_dict():
            print(f"Warning: {name} is in the model but not in the state dict.")
    for name in cbl.state_dict():
        if name not in cbl.named_parameters():
            print(f"Warning: {name} is in the state dict but not in the model.")
    cbl.eval()
    
    return preLM, cbl


def process_run(run_id, classifier_suffixes, expected_dataset, seed, wandb_project, wandb_entity=None):
    """
    Process a single wandb run: load config, load models, run steerability tests, and log results.
    """
    # Reseed at the start of every run for reproducibility
    set_seed(seed)
    
    print(f"\n{'='*60}")
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
    residual_dim = run_config.get('residual_dim', 768)
    
    print(f"Dataset: {dataset}, Steerability seed: {seed}")
    
    # Validate dataset matches what we expect
    if dataset != expected_dataset:
        print(f"SKIPPING run {run_id}: dataset mismatch. Run used '{dataset}' but expected '{expected_dataset}'.")
        return
    
    # Find best epoch (validation-based for SetFit/sst2, last epoch for others)
    best_epoch = find_best_epoch(run_id, dataset)
    print(f"Best epoch: {best_epoch}")
    
    if best_epoch is None:
        print(f"No model weights found for run {run_id}")
        return
    
    # Only test the best epoch
    epochs_to_test = [best_epoch]
    
    print(f"Epoch to test: {epochs_to_test}")
    
    # Setup tokenizers and config
    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token
    
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    concept_set = CFG.concepts_from_labels[dataset]
    print(f"Concept set length: {len(concept_set)}")
    
    # Load multiple roberta classifiers for steerability test
    classifiers = {}
    for suffix in classifier_suffixes:
        classifier_path = dataset.replace('/', '_') + f"_classifier{suffix}.pt"
        if not os.path.exists(classifier_path):
            print(f"Warning: Classifier not found at {classifier_path}, skipping...")
            continue
        
        classifier = Roberta_classifier(len(concept_set)).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        classifiers[suffix] = classifier
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
    
    # Run steerability test for each epoch and each classifier
    results = {}
    test_idx = 1
    for epoch_idx, epoch in enumerate(epochs_to_test, start=1):
        print(f"\nTesting epoch {epoch} (epoch_idx {epoch_idx})...")
        
        # Get model paths
        is_best = (epoch == best_epoch) if dataset == 'SetFit/sst2' else True
        peft_path, cbl_path = get_model_path_for_run(run_id, dataset, epoch, is_best=is_best)
        
        if peft_path is None or not os.path.exists(peft_path):
            # Try alternative path
            peft_path, cbl_path = get_model_path_for_run(run_id, dataset, epoch, is_best=False)
        
        if peft_path is None or not os.path.exists(peft_path):
            print(f"Model path not found for epoch {epoch}")
            continue
        
        if not os.path.exists(cbl_path):
            print(f"CBL path not found: {cbl_path}")
            continue
        
        print(f"Loading model from: {peft_path}")
        print(f"Loading CBL from: {cbl_path}")
        
        try:
            # Load models
            preLM, cbl = load_model_and_cbl(
                peft_path, cbl_path, config, concept_set, tokenizer,
                discrimination_loss, residual_dim
            )
            
            # Run steerability test with each classifier
            for suffix, classifier in classifiers.items():
                print(f"\n  Testing with classifier{suffix}...")
                
                acc = run_steerability_test(
                    preLM, cbl, tokenizer, roberta_tokenizer, classifier,
                    concept_set, dataset, seed
                )
                
                print(f"  Steerability test accuracy (epoch={epoch}, classifier{suffix}): {acc}")
                
                # Log to wandb with unique index
                log_key = f"steerability_test_accuracy_{test_idx}"
                wandb.log({
                    log_key: acc, 
                    f"epoch_for_steerability_{test_idx}": epoch,
                    f"classifier_suffix_for_steerability_{test_idx}": suffix
                })
                results[log_key] = {
                    "accuracy": acc, 
                    "epoch": epoch, 
                    "classifier_suffix": suffix
                }
                test_idx += 1
            
            # Clean up
            del preLM, cbl
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
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
    for run_id in run_ids:
        try:
            results = process_run(
                run_id, 
                classifier_suffixes,
                args.dataset,
                args.seed,
                args.wandb_project, 
                args.wandb_entity
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
