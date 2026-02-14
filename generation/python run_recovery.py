import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
import config as CFG
from transformers import LlamaConfig, AutoTokenizer, RobertaTokenizerFast
from modules import Roberta_classifier
from module_intervention import CustomLlamaModel, CustomLlamaForCausalLM, eos_pooling
import wandb
import glob
import pickle
import gc

# Argument parsing for the recovery script
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True, help="The WandB run ID (e.g., vq5w2wo0)")
parser.add_argument("--entity", type=str, default="frozenwolf", help="WandB entity/username")
parser.add_argument("--project", type=str, default="cbm-generation-new", help="WandB project name")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
cmd_args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def build_loaders(encoded_text, batch_size, num_workers, mode):
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, encoded_text):
            self.encoded_text = encoded_text

        def __getitem__(self, idx):
            t = {key: torch.tensor(values[idx]) for key, values in self.encoded_text.items()}
            return t

        def __len__(self):
            return len(self.encoded_text['input_ids'])

    dataset = ClassificationDataset(encoded_text)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

def main():
    print(f"🚀 Recovering Run ID: {cmd_args.run_id}")
    
    # 1. Connect to WandB and fetch Config
    api = wandb.Api()
    try:
        run = api.run(f"{cmd_args.entity}/{cmd_args.project}/{cmd_args.run_id}")
    except Exception as e:
        print(f"Error fetching run from WandB: {e}")
        return

    # Convert WandB config dict to a Namespace object to mimic 'args'
    class ArgsNamespace:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    # Update config with any missing defaults if necessary
    config_dict = run.config
    # Ensure lists are lists (sometimes wandb stores them as strings if not careful, but usually fine)
    args = ArgsNamespace(**config_dict)
    
    print(f"Loaded config for dataset: {args.dataset}")
    set_seed(args.seed)
    
    # Initialize a new WandB run for Evaluation logging (don't overwrite the old one)
    wandb.init(
        project="cbm-generation-new", 
        id=cmd_args.run_id, 
        resume="must"
    )
    # 2. Prepare Data (Only Test Data needed usually, but we load needed parts)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading test dataset...")
    test_dataset = load_dataset(args.dataset, split='test')
    
    # Preprocessing logic from original script
    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "")
            example["text"] = example["text"].replace("#39;", "'")
            return example
        test_dataset = test_dataset.map(replace_bad_string)

    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    
    # Remove columns
    cols_to_remove = [CFG.example_name[args.dataset]]
    if args.dataset == 'SetFit/sst2': cols_to_remove.append('label_text')
    if args.dataset == 'dbpedia_14': cols_to_remove.append('title')
    encoded_test_dataset = encoded_test_dataset.remove_columns(cols_to_remove)
    
    concept_set = CFG.concepts_from_labels[args.dataset]
    test_loader = build_loaders(encoded_test_dataset, args.batch_size, args.num_workers, mode="test")

    # 3. Initialize Model Structure
    print("Initializing Model...")
    preLM = CustomLlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
    
    # Create intermediate layer using args from WandB
    preLM.create_intermediate(
        args.intermediate_loc, 
        len(concept_set), 
        intermediate_sizes=args.intermediate_sizes, 
        skip_dropout=args.skip_dropout
    )
    preLM.to(cmd_args.device)
    
    preLM_generator = CustomLlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
    preLM_generator.model = preLM
    preLM_generator.lm_head.to(cmd_args.device)

    # 4. Locate and Load Weights
    d_name = args.dataset.replace('/', '_')
    # Reconstruct path based on: "./from_pretained_llama3_lora_cbm_" + run_name + "/" + d_name + "/"
    save_dir = f"./from_pretained_llama3_lora_cbm_{cmd_args.run_id}/{d_name}/"
    
    # Try to find best epoch from WandB summary
    best_epoch = run.summary.get("best_model_epoch")
    
    model_path = None
    if best_epoch:
        potential_path = os.path.join(save_dir, f"llama3_epoch_{best_epoch}")
        if os.path.exists(potential_path):
            model_path = potential_path
            print(f"Found best model from summary: {model_path}")
    
    # Fallback: Find the latest epoch file if best not found or summary missing
    if model_path is None:
        print("Best epoch not found in summary, searching directory...")
        files = glob.glob(os.path.join(save_dir, "llama3_epoch_*"))
        if not files:
            raise FileNotFoundError(f"No model checkpoints found in {save_dir}")
        # Sort by modification time or extract epoch number
        model_path = max(files, key=os.path.getctime)
        print(f"Loading latest checkpoint: {model_path}")

    # Load state dict
    state_dict = torch.load(model_path, map_location=cmd_args.device)
    preLM.intermediate.load_state_dict(state_dict)
    
    preLM.eval()
    preLM.intermediate.eval()
    preLM_generator.eval()

    # ==============================================================================
    # 5. EXECUTE EVALUATION (Steerability, Concept Pred, Perplexity)
    # ==============================================================================

    # --- A. STEERABILITY ---
    print("\n=== Running Steerability Test ===")
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    classifier_path = args.dataset.replace('/', '_') + "_classifier.pt"
    
    if not os.path.exists(classifier_path):
        print(f"Warning: Classifier not found at {classifier_path}. Skipping Steerability.")
    else:
        classifier = Roberta_classifier(len(concept_set)).to(cmd_args.device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=cmd_args.device))
        
        pred = []
        text = []
        acc_metric = evaluate.load("accuracy")

        with torch.no_grad():
            # Running fewer iterations for speed if needed, or keeping original 100 // len
            iterations = 100 // len(concept_set)
            for i in tqdm(range(iterations), desc="Steerability"):
                input_ids = torch.tensor([tokenizer.encode("")]).to(cmd_args.device)
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                
                for j in range(len(concept_set)):
                    v = [0] * len(concept_set)
                    v[j] = 1
                    B, T = input_ids.shape
                    intervene_tensor = torch.tensor(v, device=cmd_args.device).view(1, 1, -1).expand(B, T, len(concept_set))

                    preLM_generator.model.intervene = intervene_tensor
                    preLM_generator.model.intervention_margin = args.intervention_margin
                    preLM_generator.model.intervention_spread = args.intervention_spread
                    
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output_tokens = preLM_generator.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                            max_new_tokens=100,
                            temperature=0.7,
                            top_k=100,
                            top_p=0.9,
                            repetition_penalty=1.5,
                            pad_token_id=128001
                        )
                    preLM_generator.model.intervene = None
                    decoded_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                    text.append(decoded_text)

                    encoded_input = roberta_tokenizer(
                        decoded_text, return_tensors='pt', truncation=True, max_length=512
                    ).to(cmd_args.device)

                    roberta_input = {"input_ids": encoded_input["input_ids"], "attention_mask": encoded_input["attention_mask"]}
                    logits = classifier(roberta_input)
                    pred.append(logits)

        pred = torch.cat(pred, dim=0).detach().cpu()
        pred_labels = np.argmax(pred.numpy(), axis=-1)
        refs = list(range(len(concept_set))) * iterations
        
        # Ensure lengths match (in case of uneven loops)
        min_len = min(len(pred_labels), len(refs))
        accuracy = acc_metric.compute(predictions=pred_labels[:min_len], references=refs[:min_len])
        
        print("Steerability test accuracy:", accuracy)
        wandb.log({"steerability_test_accuracy": accuracy['accuracy']})

    # --- B. CONCEPT PREDICTION ---
    print("\n=== Running Concept Prediction Test ===")
    metric = evaluate.load("accuracy")
    concept_predictions = []
    
    for batch in tqdm(test_loader, desc="Concept Pred"):
        batch = {k: v.to(cmd_args.device) for k, v in batch.items()}
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pre_hidden_states, causal_mask, position_embeddings = (
                    preLM.firsthalf_forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                )
                concepts, skips = preLM.intermediate.encode(pre_hidden_states)
        concept_predictions.append(eos_pooling(concepts, batch["attention_mask"]))
        
    concept_predictions = torch.cat(concept_predictions, dim=0).detach().cpu()
    pred_labels = np.argmax(concept_predictions.numpy(), axis=-1)
    
    # Create references list matching the predictions size
    # Note: dataset length might slightly differ from loader loop due to batch dropping/size
    references = encoded_test_dataset["label"][:len(pred_labels)]
    
    acc = metric.compute(predictions=pred_labels, references=references)
    print("Concept prediction accuracy:", acc)
    wandb.log({"concept_prediction_accuracy": acc['accuracy']})

    # --- C. PERPLEXITY ---
    print("\n=== Running Perplexity Test ===")
    set_seed(args.seed)
    
    pred_texts = []
    input_ids = torch.tensor([tokenizer.encode("")]).to(cmd_args.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    for i in tqdm(range(100), desc="Generating for PPL"):
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_ids = preLM_generator.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=100,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    pad_token_id=128001
                )
            decoded = tokenizer.decode(text_ids[0], skip_special_tokens=True)
            pred_texts.append(decoded)

    # Save generated texts
    if not os.path.exists("perplexity_text"):
        os.makedirs("perplexity_text")
    pickle.dump(pred_texts, open(f"perplexity_text/{cmd_args.run_id}_generated_texts_{args.seed}.pkl", "wb"))

    # Free up memory before loading the metric model (perplexity loads a model internally)
    del preLM, preLM_generator
    torch.cuda.empty_cache()
    gc.collect()

    # Calculate Perplexity (Using standard library approach)
    # Note: We filter for length > 0
    clean_preds = [p for p in pred_texts if len(p.strip()) > 0]
    
    print("Calculating Perplexity (this loads a new model)...")
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    
    # 1. Under 30 tokens (approx) check
    short_preds = [p for p in clean_preds if len(p.split()) < 30] # Rough heuristic from original
    if short_preds:
        try:
            ppl_short = perplexity_metric.compute(predictions=short_preds, model_id='meta-llama/Meta-Llama-3-8B', device=cmd_args.device)['mean_perplexity']
            print(f"Perplexity (Short texts): {ppl_short}")
            wandb.log({"perplexity_under_30_tokens": ppl_short})
        except Exception as e:
            print(f"Error calc short perplexity: {e}")
            
    # 2. All tokens
    try:
        ppl_all = perplexity_metric.compute(predictions=clean_preds, model_id='meta-llama/Meta-Llama-3-8B', device=cmd_args.device)['mean_perplexity']
        print(f"Perplexity (All texts): {ppl_all}")
        wandb.log({"perplexity_all_tokens": ppl_all})
    except Exception as e:
        print(f"Error calc all perplexity: {e}")

    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    main()