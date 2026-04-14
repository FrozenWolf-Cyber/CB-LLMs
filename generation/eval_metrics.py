"""
Centralized evaluation metrics for CB-LLMs generation.

All eval/test functions used by train scripts and resume scripts live here.
Supports caching for steerability text generation and perplexity text generation.

Default metrics (called by train scripts after training):
  - Perplexity (under 30 tokens + all tokens)
  - Steerability (RoBERTa classifiers or MPNet similarity)
  - Concept accuracy (hard labels or cosine similarity)
  - RM rewards (relevance, grammar, together)
"""
from __future__ import annotations

import gc
import glob
import os
import pickle
from typing import Dict, List, Optional, Sequence

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaConfig,
    LlamaModel,
    RobertaTokenizerFast,
)

from modules import CBL, CBLResidual, Roberta_classifier
from steerability_cache import (
    load_concept_samples,
    save_all_steerability_texts,
    steerability_output_root,
    write_samples_batch,
)
from utils import cos_sim_cubed, eos_pooling, mean_pooling


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


_CACHED_LLAMA_VOCAB_WEIGHT = None


def get_llama_vocab_weight(device):
    global _CACHED_LLAMA_VOCAB_WEIGHT
    if _CACHED_LLAMA_VOCAB_WEIGHT is not None:
        return _CACHED_LLAMA_VOCAB_WEIGHT
    lm_head_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16,
    ).to(device)
    _CACHED_LLAMA_VOCAB_WEIGHT = lm_head_model.get_output_embeddings().weight.detach()
    del lm_head_model
    torch.cuda.empty_cache()
    return _CACHED_LLAMA_VOCAB_WEIGHT


def release_llama_vocab_weight():
    global _CACHED_LLAMA_VOCAB_WEIGHT
    _CACHED_LLAMA_VOCAB_WEIGHT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_intervention_value(dataset: str) -> int:
    return 150 if dataset == "dbpedia_14" else 100


# ═══════════════════════════════════════════════════════════════
# Checkpoint Discovery
# ═══════════════════════════════════════════════════════════════

def infer_run_layout(run_id, dataset, run_config):
    d_name = dataset.replace("/", "_")
    cbm_prefix = f"./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/"
    grpo_prefix = f"./from_pretained_llama3_lora_grpo_{run_id}/{d_name}/"

    cbm_exists = os.path.isdir(cbm_prefix)
    grpo_exists = os.path.isdir(grpo_prefix)

    if cbm_exists and not grpo_exists:
        return "cbm", cbm_prefix
    if grpo_exists and not cbm_exists:
        return "grpo", grpo_prefix
    if "grpo_epochs" in run_config and "pretrained_run_id" in run_config:
        return "grpo", grpo_prefix
    if "discrimination_loss" in run_config:
        return "cbm", cbm_prefix
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


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model_and_cbl(
    peft_path, cbl_path, config, concept_set, tokenizer,
    discrimination_loss, residual_dim, device,
):
    preLM = LlamaModel.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16,
    ).to(device)
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
            f"Falling back to strict=False: "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    cbl.eval()
    return preLM, cbl


# ═══════════════════════════════════════════════════════════════
# Steerability Text Generation (with disk caching)
# ═══════════════════════════════════════════════════════════════

def generate_steerability_texts(
    preLM,
    cbl,
    tokenizer,
    concept_set,
    dataset,
    device,
    samples_per_concept,
    print_k=3,
    llama_vocab_weight=None,
    keep_other_concepts=False,
    steerability_cache_dir=None,
    steerability_cache_seed=42,
    interventions_per_batch=1,
):
    """
    Generate steered texts for each concept with caching.

    Returns list-of-lists: ``decoded_texts_by_concept[concept_idx][sample_idx]``.
    """
    intervention_value = get_intervention_value(dataset)
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    special_tokens_mask = torch.tensor([128000, 128001]).to(device)
    num_concepts = len(concept_set)
    chunk_size = 25
    cseed = steerability_cache_seed

    all_slots: list[list[str | None]] = []
    for concept_idx in range(num_concepts):
        cname = concept_set[concept_idx]
        slots = load_concept_samples(
            steerability_cache_dir, cseed, concept_idx, cname, samples_per_concept,
        )
        all_slots.append(slots)

    with torch.no_grad():
        if interventions_per_batch <= 1:
            for concept_idx in tqdm(range(num_concepts), desc="Steerability generation"):
                v = [0] * num_concepts
                v[concept_idx] = intervention_value
                cname = concept_set[concept_idx]
                slots = all_slots[concept_idx]
                pos = 0
                while pos < samples_per_concept:
                    if slots[pos] is not None:
                        pos += 1
                        continue
                    end = pos
                    while end < samples_per_concept and slots[end] is None:
                        end += 1
                    gen_pos = pos
                    while gen_pos < end:
                        current_batch = min(chunk_size, end - gen_pos)
                        text_ids_batch, _ = cbl.generate_batch(
                            input_ids, preLM,
                            num_samples=current_batch,
                            intervene=v, length=50,
                            keep_other_concepts=keep_other_concepts,
                            llama_vocab_weight=llama_vocab_weight,
                        )
                        pending_writes: list[tuple] = []
                        for b in range(current_batch):
                            sample_idx = gen_pos + b
                            decoded = tokenizer.decode(
                                text_ids_batch[b][~torch.isin(text_ids_batch[b], special_tokens_mask)]
                            )
                            slots[sample_idx] = decoded
                            if steerability_cache_dir:
                                pending_writes.append(
                                    (concept_idx, cname, cseed, sample_idx, decoded)
                                )
                        if pending_writes:
                            write_samples_batch(steerability_cache_dir, pending_writes)
                        gen_pos += current_batch
                    pos = end
        else:
            for group_start in tqdm(
                range(0, num_concepts, interventions_per_batch),
                desc=f"Steerability generation (x{interventions_per_batch} concepts/batch)",
            ):
                group_end = min(group_start + interventions_per_batch, num_concepts)
                group_indices = list(range(group_start, group_end))

                needs_gen: list[int] = []
                missing_indices: dict[int, list[int]] = {}
                for ci in group_indices:
                    missing = [i for i in range(samples_per_concept) if all_slots[ci][i] is None]
                    if missing:
                        needs_gen.append(ci)
                        missing_indices[ci] = missing

                if not needs_gen:
                    continue

                interventions = []
                for ci in needs_gen:
                    v = [0] * num_concepts
                    v[ci] = intervention_value
                    interventions.append(v)

                max_missing = max(len(missing_indices[ci]) for ci in needs_gen)
                gen_offset = 0
                while gen_offset < max_missing:
                    current_chunk = min(chunk_size, max_missing - gen_offset)
                    text_ids_batch, _ = cbl.generate_multi_concept_batch(
                        input_ids, preLM,
                        interventions=interventions,
                        samples_per_intervention=current_chunk,
                        length=50,
                        keep_other_concepts=keep_other_concepts,
                        llama_vocab_weight=llama_vocab_weight,
                    )
                    pending_writes = []
                    for g, ci in enumerate(needs_gen):
                        row_start = g * current_chunk
                        mi = missing_indices[ci]
                        cname = concept_set[ci]
                        for b in range(current_chunk):
                            abs_idx = gen_offset + b
                            if abs_idx >= len(mi):
                                continue
                            sample_idx = mi[abs_idx]
                            decoded = tokenizer.decode(
                                text_ids_batch[row_start + b][
                                    ~torch.isin(text_ids_batch[row_start + b], special_tokens_mask)
                                ]
                            )
                            all_slots[ci][sample_idx] = decoded
                            if steerability_cache_dir:
                                pending_writes.append(
                                    (ci, cname, cseed, sample_idx, decoded)
                                )
                    if pending_writes:
                        write_samples_batch(steerability_cache_dir, pending_writes)
                    gen_offset += current_chunk

    all_texts: list[list[str]] = []
    for concept_idx in range(num_concepts):
        cname = concept_set[concept_idx]
        concept_texts = [all_slots[concept_idx][k] for k in range(samples_per_concept)]
        for idx, t in enumerate(concept_texts):
            wandb.log({f"steerability_sample_{cname}_{idx + 1}": t})
        if print_k > 0:
            print(f"Concept '{cname}' sample preview:")
            for k in range(min(print_k, len(concept_texts))):
                print(f"  [{k+1}] {concept_texts[k]}")
        all_texts.append(concept_texts)

    return all_texts


# ═══════════════════════════════════════════════════════════════
# Steerability Evaluation: RoBERTa Classifiers
# ═══════════════════════════════════════════════════════════════

def score_steerability_roberta(
    decoded_texts_by_concept, roberta_tokenizer, classifier, concept_set, device,
):
    """Score steerability texts with a single RoBERTa classifier. Returns accuracy dict."""
    pred, ref = [], []
    acc = evaluate.load("accuracy")
    with torch.no_grad():
        for concept_idx, concept_texts in enumerate(
            tqdm(decoded_texts_by_concept, desc="Steerability scoring")
        ):
            if not concept_texts:
                continue
            roberta_enc = roberta_tokenizer(
                concept_texts, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
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


# Backward-compatible alias used by resume_steerability_test
run_steerability_test_from_texts = score_steerability_roberta


def run_steerability_roberta(
    decoded_texts_by_concept,
    concept_set,
    dataset,
    device,
    classifier_weight_suffixes=("_seed42", "_seed123", "_seed456"),
):
    """Run steerability eval with multiple RoBERTa classifiers. Returns dict of accuracies."""
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    d_name = dataset.replace("/", "_")
    classifier_paths = [f"{d_name}_classifier.pt"]
    for suffix in classifier_weight_suffixes:
        classifier_paths.append(f"{d_name}_classifier{suffix}.pt")

    results = {}
    for clf_idx, classifier_path in enumerate(classifier_paths):
        if not os.path.exists(classifier_path):
            print(f"Warning: Classifier not found at {classifier_path}, skipping...")
            continue
        print(f"Testing steerability with classifier: {classifier_path}")
        classifier = Roberta_classifier(len(concept_set)).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        try:
            classifier = torch.compile(classifier)
        except Exception:
            pass

        acc = score_steerability_roberta(
            decoded_texts_by_concept, roberta_tokenizer, classifier, concept_set, device,
        )
        log_key = "steerability_test_accuracy" if clf_idx == 0 else f"steerability_test_accuracy_{clf_idx}"
        print(f"  {log_key}: {acc}")
        wandb.log({log_key: acc})
        results[log_key] = acc

        del classifier
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════
# Steerability Evaluation: MPNet Similarity
# ═══════════════════════════════════════════════════════════════

def run_steerability_mpnet(
    decoded_texts_by_concept, concept_set, intervention_value, max_length, device,
):
    """Run steerability eval using MPNet sentence similarity. Returns dict of metrics."""
    tokenizer_sim = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    sim_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    sim_model.eval()

    encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=max_length)
    encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}
    concept_features = sim_model(
        input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"],
    )
    concept_features = mean_pooling(concept_features.last_hidden_state, encoded_c["attention_mask"])
    concept_features = F.normalize(concept_features, p=2, dim=1)

    cos_sim_cubed_values: list[float] = []
    softmax_values: list[float] = []
    top1_correct = top3_correct = top5_correct = top10_correct = top20_correct = 0
    total_evals = 0
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for j in tqdm(range(len(concept_set)), desc="Steerability MPNet scoring"):
            decoded_texts = decoded_texts_by_concept[j]
            if not decoded_texts:
                continue

            v = [0] * len(concept_set)
            v[j] = intervention_value

            generated_c = tokenizer_sim(
                decoded_texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            generated_c = {k: v_t.to(device) for k, v_t in generated_c.items()}
            generated_features = sim_model(
                input_ids=generated_c["input_ids"],
                attention_mask=generated_c["attention_mask"],
            )
            generated_features = mean_pooling(
                generated_features.last_hidden_state, generated_c["attention_mask"],
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

    del sim_model
    torch.cuda.empty_cache()

    metrics = {
        "steerability_cos_sim_cubed": (
            sum(cos_sim_cubed_values) / len(cos_sim_cubed_values)
            if cos_sim_cubed_values else float("nan")
        ),
        "steerability_softmax": (
            sum(softmax_values) / len(softmax_values)
            if softmax_values else float("nan")
        ),
        "steerability_top1_acc": top1_correct / total_evals if total_evals else 0.0,
        "steerability_top3_acc": top3_correct / total_evals if total_evals else 0.0,
        "steerability_top5_acc": top5_correct / total_evals if total_evals else 0.0,
        "steerability_top10_acc": top10_correct / total_evals if total_evals else 0.0,
        "steerability_top20_acc": top20_correct / total_evals if total_evals else 0.0,
    }
    wandb.log(metrics)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


# ═══════════════════════════════════════════════════════════════
# Concept Accuracy: Hard Labels (train_combined.py style)
# ═══════════════════════════════════════════════════════════════

def run_concept_accuracy_labels(preLM, cbl, test_loader, concept_set, encoded_test_dataset, device):
    """Concept prediction accuracy using argmax (hard labels). Returns accuracy dict."""
    print("eval concepts...")
    metric = evaluate.load("accuracy")
    concept_predictions = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            ).last_hidden_state
            concepts, _, _, _ = cbl(features.float())
        concept_predictions.append(eos_pooling(concepts, batch["attention_mask"]))
    concept_predictions = torch.cat(concept_predictions, dim=0).detach().cpu()
    pred = np.argmax(concept_predictions.numpy(), axis=-1)
    metric.add_batch(predictions=pred, references=encoded_test_dataset["label"])
    acc = metric.compute()
    print(f"Concept prediction accuracy: {acc}")
    wandb.log({"concept_prediction_accuracy": acc})
    return acc


# ═══════════════════════════════════════════════════════════════
# Concept Accuracy: Cosine Similarity (train_combined_finegrained.py style)
# ═══════════════════════════════════════════════════════════════

def run_concept_accuracy_cosine(preLM, cbl, test_loader, concept_set, label_prefix, device):
    """Concept prediction accuracy using cosine similarity to ACS labels. Returns dict."""
    print("eval concepts (cosine similarity to MPNet labels)...")
    concept_predictions = []
    for batch, _ in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            ).last_hidden_state
            concepts, _, _, _ = cbl(features.float())
        pooled_concepts = eos_pooling(concepts, batch["attention_mask"])
        concept_predictions.append(pooled_concepts.detach().cpu())
    concept_predictions = torch.cat(concept_predictions, dim=0)

    test_sim_path = os.path.join(label_prefix, "concept_labels_test.npy")
    if not os.path.exists(test_sim_path):
        print(f"[WARN] {test_sim_path} not found. Skipping cosine concept evaluation.")
        return {}

    test_similarity_np = np.load(test_sim_path)
    test_similarity = torch.tensor(test_similarity_np, dtype=torch.float32)

    if test_similarity.shape != concept_predictions.shape:
        print(
            f"[WARN] Shape mismatch: predictions {tuple(concept_predictions.shape)} "
            f"vs labels {tuple(test_similarity.shape)}."
        )
        return {}

    test_cos_sim = cos_sim_cubed(concept_predictions, test_similarity)
    test_cos_loss = -test_cos_sim.item()

    pred_norm = F.normalize(concept_predictions, p=2, dim=-1)
    label_norm = F.normalize(test_similarity, p=2, dim=-1)
    test_cos_raw = (pred_norm * label_norm).sum(dim=-1).mean().item()

    print(f"Test concept cosine similarity (cos_sim_cubed): {test_cos_sim.item():.4f}")
    print(f"Test concept cosine loss: {test_cos_loss:.4f}")
    print(f"Test concept cosine similarity (raw): {test_cos_raw:.4f}")

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
            gt_set, pred_set = set(gt_topk), set(pred_topk)
            inter = len(gt_set & pred_set)
            union = len(gt_set | pred_set)
            if union > 0:
                topk_iou_sums[k] += inter / union
            if gt_idx in row[:k].tolist():
                topk_hits[k] += 1

    topk_acc = {f"test_concept_top{k}_acc": topk_hits[k] / total for k in topk_list}
    topk_iou = {f"test_concept_top{k}_iou": topk_iou_sums[k] / total for k in topk_list}

    for k in topk_list:
        print(f"Test concept Top-{k} Acc: {topk_acc[f'test_concept_top{k}_acc']:.4f}")
        print(f"Test concept Top-{k} IoU: {topk_iou[f'test_concept_top{k}_iou']:.4f}")

    metrics = {
        "test_concept_cosine_similarity": float(test_cos_sim.item()),
        "test_concept_cosine_loss": float(test_cos_loss),
        "test_concept_cosine_raw": float(test_cos_raw),
        **topk_acc,
        **topk_iou,
    }
    wandb.log(metrics)
    return metrics


# ═══════════════════════════════════════════════════════════════
# Weight Analysis
# ═══════════════════════════════════════════════════════════════

def run_weight_analysis(cbl, concept_set, tokenizer):
    """Print and log top tokens per concept neuron and sparsity."""
    print("Top tokens for each concept neuron:")
    w = cbl.fc.weight.data[:, : len(concept_set)].T
    for i in tqdm(range(len(concept_set))):
        top_values, top_ids = torch.topk(w[i], k=10)
        print(f"Neuron: {concept_set[i]}")
        print("Top 10 tokens with highest weight:")
        for j in range(10):
            print(
                f"Neuron: {concept_set[i]} "
                f"[{round(float(top_values.detach().cpu()[j]), 3)}] "
                f"{tokenizer.decode(top_ids[j])}"
            )

    sparsity = (w > 1e-6).count_nonzero() / w.numel()
    print(f"Sparsity of concept weight matrix: {sparsity}")
    wandb.log({"concept_weight_sparsity": sparsity})


# ═══════════════════════════════════════════════════════════════
# Perplexity (split: generate texts, then compute metric)
# ═══════════════════════════════════════════════════════════════

def _perplexity_cache_path(cache_dir, seed):
    return os.path.join(cache_dir, f"perplexity_texts_seed{seed}.pkl")


def generate_perplexity_texts(
    cbl, preLM, tokenizer, seed, device,
    n_samples=100, cache_dir=None, run_name=None,
    llama_vocab_weight=None,
):
    """Generate free (un-intervened) texts for perplexity. Supports caching."""
    set_seed(seed)
    pred: list[str] = []
    cached = False

    if cache_dir:
        cache_path = _perplexity_cache_path(cache_dir, seed)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                pred = pickle.load(f)
            if len(pred) >= n_samples:
                pred = pred[:n_samples]
                cached = True
                print(f"Loaded {len(pred)} cached perplexity texts from {cache_path}")

    if not cached:
        input_ids = torch.tensor([tokenizer.encode("")]).to(device)
        for _ in tqdm(range(n_samples), desc="Generating perplexity texts"):
            with torch.no_grad():
                text_ids, _ = cbl.generate(input_ids, preLM, llama_vocab_weight=llama_vocab_weight)
                pred.append(tokenizer.decode(text_ids[0], skip_special_tokens=True))

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(_perplexity_cache_path(cache_dir, seed), "wb") as f:
                pickle.dump(pred, f)
            print(f"Saved perplexity texts to cache")

    if run_name:
        os.makedirs("perplexity_text", exist_ok=True)
        with open(f"perplexity_text/{run_name}_generated_texts_{seed}.pkl", "wb") as f:
            pickle.dump(pred, f)

    print("Some generated texts:")
    for i in range(min(5, len(pred))):
        print(pred[i])

    return pred


def compute_perplexity(texts: list[str]) -> dict:
    """Compute perplexity (under-30 tokens and all tokens) from pre-generated texts.

    This function loads a fresh LLM via the ``evaluate`` library, so the training
    model should be freed from GPU beforehand.
    """
    results = {}

    c = 0
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    for p in texts:
        if len(p.split()) > 30:
            continue
        c += 1
        perplexity_metric.add_batch(predictions=[p])

    if c > 0:
        ppl_short = perplexity_metric.compute(
            model_id="meta-llama/Meta-Llama-3-8B", max_length=100,
        )["mean_perplexity"]
        print(f"Perplexity (under 30 tokens): {ppl_short}")
        wandb.log({"perplexity_under_30_tokens": ppl_short})
        results["perplexity_under_30_tokens"] = ppl_short
    else:
        print("No generated texts under 30 tokens to compute perplexity.")
        wandb.log({"perplexity_under_30_tokens": None})

    perplexity_all = evaluate.load("perplexity", module_type="metric")
    for p in texts:
        perplexity_all.add_batch(predictions=[p])
    ppl_all = perplexity_all.compute(
        model_id="meta-llama/Meta-Llama-3-8B", max_length=100,
    )["mean_perplexity"]
    print(f"Perplexity (all tokens): {ppl_all}")
    wandb.log({"perplexity_all_tokens": ppl_all})
    results["perplexity_all_tokens"] = ppl_all

    return results


# ═══════════════════════════════════════════════════════════════
# RM (Reward Model) Metrics
# ═══════════════════════════════════════════════════════════════

RM_USER_RELEVANCE = "Write a text about the concept: {concept_name}"
RM_USER_GRAMMAR = "Write a grammatically correct and fluent paragraph."
RM_USER_TOGETHER = "Write a grammatically correct and fluent text about the concept: {concept_name}"
RM_LOGIT_CLIP_MIN = -100.0
RM_LOGIT_CLIP_MAX = 100.0


def load_reward_model(rm_model_name: str, rm_device: torch.device):
    """Load a Skywork-style sequence-classification RM."""
    print(f"Loading reward model: {rm_model_name} ...")
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
    _kwargs = dict(torch_dtype=torch.bfloat16, num_labels=1)
    try:
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            rm_model_name, attn_implementation="flash_attention_2", **_kwargs,
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


def _make_rm_formatted(rm_tokenizer, user_turn: str, response_text: str, max_text_len: int) -> str:
    conv = [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": response_text[:max_text_len]},
    ]
    formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
    if rm_tokenizer.bos_token and formatted.startswith(rm_tokenizer.bos_token):
        formatted = formatted[len(rm_tokenizer.bos_token):]
    return formatted


def _raw_logits_for_texts(
    rm_model, rm_tokenizer, texts, user_turn: str,
    device: torch.device, rm_batch_size: int, max_text_len: int,
):
    if not texts:
        return []
    formatted = [_make_rm_formatted(rm_tokenizer, user_turn, t, max_text_len) for t in texts]
    chunk = rm_batch_size if rm_batch_size > 0 else len(formatted)
    all_scores: list[float] = []
    for start in range(0, len(formatted), chunk):
        chunk_list = formatted[start : start + chunk]
        tokenized = rm_tokenizer(
            chunk_list, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)
        with torch.no_grad():
            logits = rm_model(**tokenized).logits
        clipped = logits[:, 0].float().clamp(RM_LOGIT_CLIP_MIN, RM_LOGIT_CLIP_MAX)
        all_scores.extend(clipped.detach().cpu().tolist())
        del tokenized, logits
    return all_scores


def run_rm_metrics(
    decoded_texts_by_concept,
    concept_set,
    rm_model,
    rm_tokenizer,
    rm_device,
    rm_batch_size=0,
    rm_max_text_len=500,
):
    """Score steerability texts with RM (relevance, grammar, together).

    Returns dict with global means/stds and per-concept breakdown.
    """
    all_rel, all_gram, all_tog = [], [], []
    per_concept: dict = {}

    for concept_idx, concept_name in enumerate(concept_set):
        texts = (
            decoded_texts_by_concept[concept_idx]
            if concept_idx < len(decoded_texts_by_concept) else []
        )
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
            rm_model, rm_tokenizer, texts, u_rel, rm_device, rm_batch_size, rm_max_text_len,
        )
        gram = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, RM_USER_GRAMMAR, rm_device, rm_batch_size, rm_max_text_len,
        )
        tog = _raw_logits_for_texts(
            rm_model, rm_tokenizer, texts, u_tog, rm_device, rm_batch_size, rm_max_text_len,
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
            wandb.log({
                f"rm_sample_{concept_name}_{b + 1}": t,
                f"rm_relevance_logit_{concept_name}_{b + 1}": r,
                f"rm_grammar_logit_{concept_name}_{b + 1}": g,
                f"rm_together_logit_{concept_name}_{b + 1}": o,
            })

    def _ms(xs):
        if not xs:
            return float("nan"), 0.0
        a = np.array(xs, dtype=np.float64)
        return float(a.mean()), float(a.std()) if a.size > 1 else 0.0

    r_m, r_s = _ms(all_rel)
    g_m, g_s = _ms(all_gram)
    t_m, t_s = _ms(all_tog)

    global_metrics = {
        "rm_relevance_mean": r_m, "rm_relevance_std": r_s,
        "rm_grammar_mean": g_m, "rm_grammar_std": g_s,
        "rm_together_mean": t_m, "rm_together_std": t_s,
        "rm_total_n": len(all_rel),
    }
    wandb.log(global_metrics)
    print(
        f"  rm_relevance_mean={r_m:.4f} rm_grammar_mean={g_m:.4f} "
        f"rm_together_mean={t_m:.4f} (n={len(all_rel)})"
    )

    return {**global_metrics, "per_concept": per_concept}
