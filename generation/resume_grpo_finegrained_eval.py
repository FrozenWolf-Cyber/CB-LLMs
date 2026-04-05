import argparse
import os
import pickle
import glob
import torch
import numpy as np
import evaluate
from tqdm.auto import tqdm
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast
from modules import CBLResidual, CBL, Roberta_classifier
import wandb
import time
import torch.nn.functional as F
from datasets import load_dataset
from utils import mean_pooling, get_labels, cos_sim_cubed
from steerability_cache import load_concept_samples, sample_file_path, save_all_steerability_texts, steerability_output_root, write_sample


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
parser.add_argument(
    "--samples_per_concept",
    type=int,
    default=None,
    help="Steerability generations per concept in GRPO eval. If omitted, uses max(1, 100 // num_concepts) like resume_steerability_test.",
)


def find_eval_checkpoint(run_id, dataset):
    """
    Return checkpoint paths (peft_path, cbl_path, epoch) for evaluation for a GRPO run.
    """
    d_name = dataset.replace('/', '_')
    prefix = f"./from_pretained_llama3_lora_grpo_{run_id}/{d_name}/"
    
    if not os.path.isdir(prefix):
        return None, None, None

    cbl_files = sorted(glob.glob(os.path.join(prefix, "cbl_epoch_*.pt")))
    
    if not cbl_files:
        return None, None, None

    latest_cbl_path = cbl_files[-1]
    try:
        epoch = int(os.path.basename(latest_cbl_path).replace("cbl_epoch_", "").replace(".pt", ""))
    except ValueError:
        return None, None, None

    peft_path = os.path.join(prefix, f"llama3_epoch_{epoch}")
    cbl_path = os.path.join(prefix, f"cbl_epoch_{epoch}.pt")

    if not os.path.isdir(peft_path) or not os.path.isfile(cbl_path):
        return None, None, None

    return peft_path, cbl_path, epoch


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


def load_model_and_cbl(peft_path, cbl_path, config, concept_set, tokenizer, arch_type, residual_dim=768):
    """
    Load the LLaMA model with LoRA adapter and CBL module.
    """
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM.load_adapter(peft_path)
    preLM.eval()
    
    if arch_type == "non_residual":
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), residual_dim, tokenizer).to(device)
    
    cbl.load_state_dict(torch.load(cbl_path, map_location=device), strict=False)
    cbl.eval()
    
    return preLM, cbl


def run_evaluation(
    preLM,
    cbl,
    tokenizer,
    concept_set,
    dataset,
    run_config,
    classifier_suffixes,
    run_name,
    seed,
    num_steerability_samples=None,
    steerability_cache_dir=None,
    steerability_cache_seed=42,
):
    """
    Run all evaluations from train_grpo_finegrained.py.
    """
    results = {}
    
    # 1. Steerability Test (from train_grpo_finegrained.py)
    print("\nRunning Steerability Test...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    sim_model.eval()

    encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=350)
    encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}
    concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"])
    concept_features = mean_pooling(concept_features.last_hidden_state, encoded_c["attention_mask"])
    concept_features = F.normalize(concept_features, p=2, dim=1)

    if dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100
        
    text = []
    cos_sim_cubed_values = []
    softmax_values = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0
    total_evals = 0

    n_steer = (
        max(1, num_steerability_samples)
        if num_steerability_samples is not None
        else max(1, 100 // len(concept_set))
    )
    print(
        f"Steerability samples per concept: {n_steer}"
        + (" (from --samples_per_concept)" if num_steerability_samples is not None else " (default: 100 // num_concepts)")
    )
    if steerability_cache_dir:
        print(f"Steerability sample cache: {steerability_cache_dir}")
    gen_input = torch.tensor([tokenizer.encode("")]).to(device)
    special_tokens_mask = torch.tensor([128000, 128001]).to(device)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    steer_concept_indices = range(len(concept_set))
    chunk_size = 25
    cseed = steerability_cache_seed
    steer_texts_snapshot = [[] for _ in range(len(concept_set))]

    with torch.no_grad():
        for j in tqdm(steer_concept_indices, desc="Steerability concepts"):
            v = [0] * len(concept_set)
            v[j] = intervention_value
            cname = concept_set[j]
            slots = load_concept_samples(steerability_cache_dir, cseed, j, cname, n_steer)
            pos = 0
            while pos < n_steer:
                if slots[pos] is not None:
                    pos += 1
                    continue
                end = pos
                while end < n_steer and slots[end] is None:
                    end += 1
                gen_pos = pos
                while gen_pos < end:
                    current_batch = min(chunk_size, end - gen_pos)
                    text_ids_batch, _ = cbl.generate_batch(
                        gen_input,
                        preLM,
                        num_samples=current_batch,
                        intervene=v,
                        length=50,
                    )
                    for b in range(current_batch):
                        sample_idx = gen_pos + b
                        tokens = text_ids_batch[b][~torch.isin(text_ids_batch[b], special_tokens_mask)]
                        decoded = tokenizer.decode(tokens)
                        slots[sample_idx] = decoded
                        if steerability_cache_dir:
                            write_sample(
                                sample_file_path(steerability_cache_dir, j, cname, cseed, sample_idx),
                                decoded,
                            )
                        text.append(decoded)
                    gen_pos += current_batch
                pos = end

            decoded_texts = [slots[k] for k in range(n_steer)]
            steer_texts_snapshot[j] = decoded_texts
            for idx, dt in enumerate(decoded_texts):
                wandb.log({f"steerability_sample_{cname}_{idx + 1}": dt})

            generated_c = tokenizer_sim(
                decoded_texts, padding=True, truncation=True, max_length=350, return_tensors="pt"
            )
            generated_c = {k: v.to(device) for k, v in generated_c.items()}
            generated_features = sim_model(
                input_ids=generated_c["input_ids"], attention_mask=generated_c["attention_mask"]
            )
            generated_features = mean_pooling(
                generated_features.last_hidden_state, generated_c["attention_mask"]
            )
            generated_features = F.normalize(generated_features, p=2, dim=1)

            sims = generated_features @ concept_features.T
            v_tensor = torch.tensor(v).to(device).unsqueeze(0).expand(sims.size(0), -1)

            cos_vals = cos_sim_cubed(sims, v_tensor.float(), reduce=False)
            cos_sim_cubed_values.extend(cos_vals.detach().cpu().tolist())

            targets = torch.full((sims.size(0),), j, dtype=torch.long, device=device)
            ce_vals = ce_loss_fn(sims, targets)
            softmax_values.extend(ce_vals.detach().cpu().tolist())

            sorted_indices = torch.argsort(sims, dim=1, descending=True)
            top1_correct += (sorted_indices[:, 0] == j).sum().item()
            top3_correct += (sorted_indices[:, :3] == j).any(dim=1).sum().item()
            top5_correct += (sorted_indices[:, :5] == j).any(dim=1).sum().item()
            top10_correct += (sorted_indices[:, :10] == j).any(dim=1).sum().item()
            top20_correct += (sorted_indices[:, :20] == j).any(dim=1).sum().item()
            total_evals += sims.size(0)

    if steerability_cache_dir:
        save_all_steerability_texts(steerability_cache_dir, cseed, concept_set, steer_texts_snapshot)

    steer_results = {
        "steerability_cos_sim_cubed": sum(cos_sim_cubed_values) / len(cos_sim_cubed_values),
        "steerability_softmax": sum(softmax_values) / len(softmax_values),
        "steerability_top1_acc": top1_correct / total_evals,
        "steerability_top3_acc": top3_correct / total_evals,
        "steerability_top5_acc": top5_correct / total_evals,
        "steerability_top10_acc": top10_correct / total_evals,
        "steerability_top20_acc": top20_correct / total_evals,
    }
    wandb.log(steer_results)
    results.update(steer_results)
    print(f"Steerability Top-1 Acc: {steer_results['steerability_top1_acc']}")

    # 2. Concept Prediction Evaluation (from train_grpo_finegrained.py)
    print("\nRunning Concept Prediction Evaluation...")
    labeling = run_config.get('labeling', 'mpnet')
    d_name = dataset.replace('/', '_')
    
    if not os.path.exists(f"./{labeling}/{d_name}") and os.path.exists(f"./{labeling}_acs/{d_name}"):
        labeling = f"{labeling}_acs"
        
    label_prefix = f"./" # Assuming labels are in the root, adjust if needed
    if labeling in ['mpnet', 'simcse', 'angle', 'llm', 'mpnet_acs', 'simcse_acs', 'angle_acs', 'llm_acs']:
        label_prefix = os.path.join(label_prefix, labeling)
    label_prefix = os.path.join(label_prefix, d_name)

    test_sim_path = os.path.join(label_prefix, "concept_labels_test.npy")
    
    if os.path.exists(test_sim_path):
        test_dataset = load_dataset(dataset, split='test')
        if dataset != 'SetFit/sst2':
            test_dataset = test_dataset.rename_column(CFG.label_name[dataset], 'label')
        if dataset == 'ag_news':
            test_dataset = test_dataset.select(range(1000))

        encoded_test_dataset = test_dataset.map(
            lambda e: tokenizer(e[CFG.example_name[dataset]], padding='max_length', truncation=True, max_length=350),
            batched=True, batch_size=len(test_dataset)
        )
        
        from utils import eos_pooling
        class EvalDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                row = self.encodings[idx]
                item = {key: torch.tensor(row[key]) for key in ['input_ids', 'attention_mask']}
                return item, 0 # dummy label
            def __len__(self):
                return len(self.encodings)

        test_loader = torch.utils.data.DataLoader(
            EvalDataset(encoded_test_dataset), batch_size=4, shuffle=False
        )

        concept_predictions = []
        for batch, _ in tqdm(test_loader, total=len(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                concepts, _, _, _ = cbl(features.float())
            pooled_concepts = eos_pooling(concepts, batch["attention_mask"])
            concept_predictions.append(pooled_concepts.detach().cpu())
        
        concept_predictions = torch.cat(concept_predictions, dim=0)
        
        test_similarity_np = np.load(test_sim_path)
        test_similarity = torch.tensor(test_similarity_np, dtype=torch.float32)

        if test_similarity.shape == concept_predictions.shape:
            test_cos_sim = cos_sim_cubed(concept_predictions, test_similarity)
            test_cos_loss = -test_cos_sim.item()

            pred_norm = F.normalize(concept_predictions, p=2, dim=-1)
            label_norm = F.normalize(test_similarity, p=2, dim=-1)
            test_cos_raw = (pred_norm * label_norm).sum(dim=-1).mean().item()

            true_concepts = torch.argmax(test_similarity, dim=-1)
            pred_sorted = torch.argsort(concept_predictions, dim=-1, descending=True)

            topk_list = [1, 3, 5, 10, 20]
            topk_hits = {k: 0 for k in topk_list}
            topk_iou_sums = {k: 0.0 for k in topk_list}
            total = concept_predictions.size(0)

            for i in range(total):
                gt_idx = true_concepts[i].item()
                row = pred_sorted[i]
                for k in topk_list:
                    k_clipped = min(k, row.size(0))
                    gt_topk = torch.topk(test_similarity[i], k=k_clipped, dim=-1).indices.tolist()
                    pred_topk = row[:k_clipped].tolist()
                    gt_set = set(gt_topk)
                    pred_set = set(pred_topk)
                    inter = len(gt_set & pred_set)
                    union = len(gt_set | pred_set)
                    if union > 0:
                        topk_iou_sums[k] += inter / union
                for k in topk_list:
                    if k <= row.size(0) and gt_idx in row[:k].tolist():
                        topk_hits[k] += 1
            
            topk_acc = {f"test_concept_top{k}_acc": topk_hits[k] / total for k in topk_list}
            topk_iou = {f"test_concept_top{k}_iou": topk_iou_sums[k] / total for k in topk_list}

            concept_pred_results = {
                "test_concept_cosine_similarity": float(test_cos_sim.item()),
                "test_concept_cosine_loss": float(test_cos_loss),
                "test_concept_cosine_raw": float(test_cos_raw),
                **topk_acc,
                **topk_iou,
            }
            wandb.log(concept_pred_results)
            results.update(concept_pred_results)
            print(f"Test concept cosine similarity (raw): {test_cos_raw:.4f}")
        else:
            print(f"[WARN] Shape mismatch for concept prediction. Skipping.")
    else:
        print(f"[WARN] {test_sim_path} not found. Skipping cosine-similarity-based concept evaluation.")


    # 3. Weight Analysis (from train_grpo_finegrained.py)
    print("\nRunning Weight Analysis...")
    # This part is tricky because cbl.fc doesn't exist in CBLResidual.
    # We will try to access it, and if it fails, we assume it's a residual model.
    try:
        w = cbl.fc.weight.data[:, :len(concept_set)].T
        for i in tqdm(range(len(concept_set)), desc="Weight Analysis"):
            top_values, top_ids = torch.topk(w[i], k=10)
            # Log or print, but avoid excessive printing in a loop
        
        sparsity = (w > 1e-6).count_nonzero() / w.numel()
        wandb.log({"concept_weight_sparsity": sparsity})
        results["concept_weight_sparsity"] = sparsity.item()
        print(f"  Concept weight sparsity: {sparsity.item()}")
    except AttributeError:
        print("  Skipping weight analysis for this model architecture (likely residual).")


    # 4. Perplexity (from train_grpo_finegrained.py)
    print("\nRunning Perplexity Calculation...")
    set_seed(seed)
    
    pred = []
    c = 0
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    for i in tqdm(range(100), desc="Perplexity Generation"):
        with torch.no_grad():
            text_ids, _ = cbl.generate(input_ids, preLM)
            decoded_text = tokenizer.decode(text_ids[0], skip_special_tokens=True)
            pred.append(decoded_text)
            if len(decoded_text.split()) <= 30 and len(decoded_text.split()) > 0:
                c += 1
                perplexity_metric.add_batch(predictions=[decoded_text])

    # Save generated texts
    os.makedirs("perplexity_text", exist_ok=True)
    safe_run_name = run_name.replace('/', '_')
    pickle.dump(pred, open(f"perplexity_text/{safe_run_name}_generated_texts_{seed}.pkl", "wb"))

    if c > 0:
        ppl_under_30 = perplexity_metric.compute(model_id='meta-llama/Meta-Llama-3-8B')['mean_perplexity']
        wandb.log({"perplexity_under_30_tokens": ppl_under_30})
        results["perplexity_under_30_tokens"] = ppl_under_30
        print(f"  Perplexity (under 30 tokens): {ppl_under_30}")
    else:
        wandb.log({"perplexity_under_30_tokens": None})
        print("  No generated texts under 30 tokens to compute perplexity.")

    perplexity_all = evaluate.load("perplexity", module_type="metric")
    # Filter out empty strings which cause errors in perplexity calculation
    pred_non_empty = [p for p in pred if p.strip()]
    if pred_non_empty:
        perplexity_all.add_batch(predictions=pred_non_empty)
        ppl_all = perplexity_all.compute(model_id='meta-llama/Meta-Llama-3-8B')['mean_perplexity']
        wandb.log({"perplexity_all_tokens": ppl_all})
        results["perplexity_all_tokens"] = ppl_all
        print(f"  Perplexity (all tokens): {ppl_all}")
    else:
        wandb.log({"perplexity_all_tokens": None})
        print("  No non-empty generated texts to compute perplexity for all tokens.")

    return results


def process_run(run_id, classifier_suffixes, expected_dataset, seed, wandb_project, wandb_entity=None,
                samples_per_concept=None, run_idx=None, total_runs=None):
    """
    Process a single wandb run: load config, load models, run evaluations, and log results.
    """
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
        return
    
    run_config = original_run.config
    print(f"Run config: {run_config}")
    
    dataset = run_config.get('dataset', 'SetFit/sst2')
    arch_type = run_config.get('arch_type', 'residual')
    residual_dim = run_config.get('residual_dim', 768)
    
    if dataset != expected_dataset:
        print(f"SKIPPING run {run_id}: dataset mismatch. Run used '{dataset}' but expected '{expected_dataset}'.")
        return
    
    peft_path, cbl_path, best_epoch = find_eval_checkpoint(run_id, dataset)
    print(f"Evaluation epoch: {best_epoch}")

    if best_epoch is None:
        print(f"No model weights found for run {run_id}")
        return

    d_name = dataset.replace("/", "_")
    grpo_prefix = os.path.join(".", f"from_pretained_llama3_lora_grpo_{run_id}", d_name)
    steer_dir = steerability_output_root(grpo_prefix, best_epoch, False)
    
    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token
    
    concept_set = CFG.concept_set[dataset]
    print(f"Concept set length: {len(concept_set)}")
    
    resumed_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_id,
        resume="must"
    )
    
    results = {}
    print(f"\nTesting epoch {best_epoch}...")
    print(f"Loading model from: {peft_path}")
    print(f"Loading CBL from: {cbl_path}")

    try:
        preLM, cbl = load_model_and_cbl(
            peft_path, cbl_path, config, concept_set, tokenizer,
            arch_type, residual_dim
        )

        eval_results = run_evaluation(
            preLM,
            cbl,
            tokenizer,
            concept_set,
            dataset,
            run_config,
            classifier_suffixes,
            resumed_run.name,
            seed,
            num_steerability_samples=samples_per_concept,
            steerability_cache_dir=steer_dir,
            steerability_cache_seed=seed,
        )
        results.update(eval_results)

        del preLM, cbl
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error testing epoch {best_epoch}: {e}")
        import traceback
        traceback.print_exc()
    
    if results:
        wandb.log({"evaluation_summary": results})
    
    wandb.finish()
    
    print(f"\nCompleted processing run {run_id}")
    print(f"Results: {results}")
    
    return results


def main():
    args = parser.parse_args()
    
    with open(args.run_ids_pickle, 'rb') as f:
        run_ids = pickle.load(f)
    
    print(f"Loaded {len(run_ids)} run IDs from {args.run_ids_pickle}")
    
    classifier_suffixes = [s.strip() for s in args.classifier_weight_suffixes.split(',')]
    
    all_results = {}
    total_runs = len(run_ids)
    for idx, run_id in enumerate(run_ids, start=1):
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
