import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset
import config_finegrained as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBLResidual, CBL, Roberta_classifier
import time
from module_intervention import amplify_intervention
from utils import elastic_net_penalty, mean_pooling, eos_pooling, get_labels, cos_sim_cubed
from steerability_cache import save_all_steerability_texts, steerability_output_root
from eval_metrics import (
    set_seed,
    get_intervention_value,
    generate_steerability_texts,
    run_steerability_mpnet,
    run_concept_accuracy_cosine,
    run_weight_analysis,
    generate_perplexity_texts,
    compute_perplexity,
    load_reward_model,
    run_rm_metrics,
)
import wandb


def _num_rows_encoded(encoded):
    """Return number of examples for either a HF Dataset or a dict-of-columns."""
    if isinstance(encoded, dict):
        if "input_ids" in encoded:
            return len(encoded["input_ids"])
        if "label" in encoded:
            return len(encoded["label"])
        first_key = next(iter(encoded.keys()))
        return len(encoded[first_key])
    return len(encoded)


def _truncate_encoded(encoded, n: int):
    if isinstance(encoded, dict):
        return {k: v[:n] for k, v in encoded.items()}
    return encoded.select(range(n))


def _align_similarity_and_encoded(similarity: np.ndarray, encoded, split_name: str):
    n_sim = int(similarity.shape[0])
    n_data = int(_num_rows_encoded(encoded))
    if n_sim == n_data:
        return similarity, encoded

    n = min(n_sim, n_data)
    print(
        f"WARNING: {split_name} concept-label rows ({n_sim}) != {split_name} dataset rows ({n_data}). "
        f"Truncating both to {n}. "
        "(This usually means the dataset is filtered/subsampled in this script but the concept labels were generated "
        "for the original split.)"
    )
    return similarity[:n], _truncate_encoded(encoded, n)


def build_intervened_concepts_from_similarity(
    concepts: torch.Tensor,
    batch_sim: torch.Tensor,
    intervention_value: float,
    keep_other_concepts: bool,
    use_topk: bool,
    topk_k: int,
    log_wandb: bool = False,
    wandb_prefix: str = "train",
) -> torch.Tensor:
    """Construct an intervention concept tensor for `cbl.intervene`.

        Current behavior (defaults):
            - pick top-1 concept per example (argmax over batch_sim)
            - set that concept to `intervention_value` for all time steps
            - set all other concepts to 0

        Optional strategies:
            - keep_other_concepts=True: start from `concepts` and only overwrite targeted dims
            - use_topk=True: stochastic top-K mode.
                    For each example:
                        1) take the top-K concepts by `batch_sim`
                        2) softmax the K scores into probabilities
                        3) sample a rank r ~ Categorical(probs)
                        4) intervene on the top-(r+1) concepts (a prefix of the top-K)

    Args:
        concepts: (B, T, C)
        batch_sim: (B, C)
    """
    if concepts.dim() != 3:
        raise ValueError(f"Expected concepts to have shape (B, T, C); got {tuple(concepts.shape)}")
    if batch_sim.dim() != 2:
        raise ValueError(f"Expected batch_sim to have shape (B, C); got {tuple(batch_sim.shape)}")
    if concepts.size(0) != batch_sim.size(0) or concepts.size(-1) != batch_sim.size(-1):
        raise ValueError(
            f"Shape mismatch: concepts {tuple(concepts.shape)} vs batch_sim {tuple(batch_sim.shape)}"
        )

    if keep_other_concepts:
        intervened = concepts.detach().clone()
    else:
        intervened = torch.zeros_like(concepts)

    value = float(intervention_value)
    B = concepts.size(0)
    C = concepts.size(-1)

    if not use_topk:
        indices = torch.argmax(batch_sim, dim=-1)  # (B,)
        for b in range(B):
            intervened[b, :, int(indices[b].item())] = value
        return intervened

    k = int(topk_k)
    if k <= 0:
        k = 1
    k = min(k, C)

    # Stochastic top-K cutoff selection per example.
    sampled_ranks = []
    sampled_probs = []
    for b in range(B):
        topk = torch.topk(batch_sim[b], k=k, dim=-1)
        topk_scores = topk.values  # (k,)
        topk_indices = topk.indices  # (k,)

        probs = F.softmax(topk_scores, dim=-1)
        sampled_rank = torch.multinomial(probs, num_samples=1).item()  # int in [0, k-1]
        cutoff = int(sampled_rank) + 1
        selected = topk_indices[:cutoff].tolist()

        sampled_ranks.append(int(sampled_rank))
        sampled_probs.append(float(probs[int(sampled_rank)].item()))

        for idx in selected:
            intervened[b, :, int(idx)] = value

    if log_wandb and len(sampled_ranks) > 0 and wandb.run is not None:
        # Log a small amount of debug info per training step.
        # `sampled_rank` corresponds to choosing top-(rank+1) concepts among the top-K.
        mean_rank = float(sum(sampled_ranks) / len(sampled_ranks))
        mean_prob = float(sum(sampled_probs) / len(sampled_probs))
        wandb.log(
            {
                f"{wandb_prefix}_intervention_topk_sampled_rank_b0": sampled_ranks[0],
                f"{wandb_prefix}_intervention_topk_sampled_prob_b0": sampled_probs[0],
                f"{wandb_prefix}_intervention_topk_sampled_rank_mean": mean_rank,
                f"{wandb_prefix}_intervention_topk_sampled_prob_mean": mean_prob,
            },
            commit=False,
        )
    return intervened
    

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epoch_multiplier", type=int, default=1, help="Epoch multiplier to increase total training steps (for debugging).")
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--samples_per_concept",
    type=int,
    default=50,
    help="Steerability evaluation: samples per concept. Default 50.",
)
parser.add_argument(
    "--train_size",
    type=int,
    default=100000,
    help="For non-SetFit/sst2 datasets, optionally subsample the train split to this many examples (roughly class-balanced, order-preserving). Set <=0 to disable.",
)
parser.add_argument("--discrimination_loss", type=float, default=1.0)
parser.add_argument("--neg_entropy_loss", type=float, default=1.0)
parser.add_argument("--concept_loss", type=float, default=1.0)
parser.add_argument("--word_loss", type=float, default=1.0)
parser.add_argument("--elastic_net_alpha", type=float, default=1.0)
parser.add_argument("--residual_dim", type=int, default=768)
parser.add_argument("--orthogonal_loss_weight", type=float, default=0)
parser.add_argument("--residual_penalty_weight", type=float, default=0)
parser.add_argument("--DEBUG", action='store_true', help="If set, use a smaller subset of data for quick debugging.")
parser.add_argument("--intervention_gen_loss", type=float, default=0.0)
parser.add_argument("--intervention_margin", type=float, default=10.0)
parser.add_argument("--no_detach_intervention", action='store_true', help="If set, do not detach unsup during intervention generation loss computation.")
parser.add_argument("--intervention_spread", type=float, default=2.0)
parser.add_argument(
    "--intervention_keep_other_concepts",
    action="store_true",
    help="If set, intervention overwrites only the selected concept(s) and keeps all other concept activations as-is (instead of setting them to 0).",
)
parser.add_argument(
    "--intervention_topk_concepts",
    action="store_true",
    help=(
        "If set, use stochastic top-K selection when constructing the intervention tensor: "
        "take the top-K concepts by similarity, softmax their scores, sample a rank, and intervene on the top-(rank+1) concepts. "
        "If not set, intervene on only the top-1 concept."
    ),
)
parser.add_argument(
    "--intervention_topk_k",
    type=int,
    default=3,
    help="K for --intervention_topk_concepts (default 3).",
)
parser.add_argument("--classifier_weight_suffixes", type=str, default="_seed42,_seed123,_seed456", 
                    help="Comma-separated list of classifier weight suffixes to test (e.g., '_seed42,_seed123,_seed456')")
parser.add_argument("--grpo_loss_weight", type=float, default=0.0, help="Weight for GRPO steerability loss. 0 disables it.")
parser.add_argument("--grpo_warmup_steps", type=int, default=300, help="Global training steps before GRPO kicks in.")
parser.add_argument("--grpo_num_trajectories", type=int, default=4, help="Number of rollouts (G) per GRPO step.")
parser.add_argument("--grpo_gen_length", type=int, default=100, help="Max generation length for GRPO rollouts.")
parser.add_argument("--grpo_clip_advantage", type=float, default=5.0, help="Clip GRPO advantages to [-clip, clip].")
parser.add_argument("--automatic_concept_correction", action='store_true', help="If set, automatically set concept labels to 0 for concepts that are not present in the example according to the ground truth label. This is a form of training intervention to correct mislabeled concepts.")
parser.add_argument("--concept_loss_type", type=str, default="cosine_cubed", help="Type of concept loss to use: 'cosine_cubed' or 'ce'.")
parser.add_argument("--labeling", type=str, default="mpnet", help="mpnet, angle, simcse, llm")
parser.add_argument("--use_last_epoch", action='store_true', help="If set, load the classifier from the last epoch instead of the best epoch based on validation loss.")
parser.add_argument(
    "--add_llama_logits",
    action="store_true",
    help=(
        "If set, add the original Llama vocab projection logits (from the backbone hidden states) to the CBL/CBLResidual logits. "
        "This keeps CBL unchanged (no extra parameters) and acts like a residual-on-logits."
    ),
)
parser.add_argument("--rm_model_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    help="HF id for sequence-classification reward model.")
parser.add_argument("--rm_batch_size", type=int, default=0, help="0 = score all texts per chunk in one forward.")
parser.add_argument("--rm_max_text_len", type=int, default=500)
parser.add_argument("--skip_rm", action="store_true", help="Skip RM reward evaluation after training.")


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text, s):
        self.encoded_text = encoded_text
        self.s = s


    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encoded_text.items()}
        y = torch.FloatTensor(self.s[idx])
        return t, y

    def __len__(self):
        return len(self.encoded_text['input_ids'])


def build_loaders(encoded_text, s, mode):
    dataset = ClassificationDataset(encoded_text, s)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    set_seed(args.seed)

    wandb.init(project="cbm-generation-new", name=f"finegrained-{args.dataset}-seed{args.seed}",
               config=vars(args))
    
    run_name = wandb.run.id
    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    test_dataset = load_dataset(args.dataset, split='test')
    if args.dataset == 'SetFit/sst2':
        val_dataset = load_dataset(args.dataset, split='validation')

    # If we subsample, do it via indices on the *original* train split so that
    # precomputed concept label matrices (generated on the original split) can
    # be subset using the same indices.
    train_select_indices = None
    original_train_len = len(train_dataset)
    if args.dataset != 'SetFit/sst2' and args.train_size and args.train_size > 0 and original_train_len > args.train_size:
        class_count = CFG.class_num[args.dataset]
        per_class = args.train_size // class_count
        if per_class <= 0:
            raise ValueError(f"train_size={args.train_size} is too small for class_count={class_count}.")
        labels = np.asarray(train_dataset["label"], dtype=np.int64)
        selected = []
        for class_id in range(class_count):
            class_indices = np.flatnonzero(labels == class_id)
            if len(class_indices) == 0:
                raise ValueError(f"No examples found for label={class_id} in dataset {args.dataset}.")
            selected.extend(class_indices[:per_class].tolist())
        # Keep original ordering to match how concept labels were generated.
        selected = sorted(selected)
        train_select_indices = selected
        train_dataset = train_dataset.select(train_select_indices)

    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "")
            example["text"] = example["text"].replace("#39;", "'")
            return example
        train_dataset = train_dataset.map(replace_bad_string)

    print("training data len: ", len(train_dataset))
    if args.dataset == 'SetFit/sst2':
        print("val data len: ", len(val_dataset))

    print("tokenizing...")

    lora_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                  "down_proj"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)

    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token

    encoded_train_dataset = train_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True,
        batch_size=len(train_dataset))
    encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['title'])
    encoded_train_dataset = encoded_train_dataset[:len(encoded_train_dataset)]

    if args.dataset == 'SetFit/sst2':
        encoded_val_dataset = val_dataset.map(
            lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True,
            batch_size=len(val_dataset))
        encoded_val_dataset = encoded_val_dataset.remove_columns([CFG.example_name[args.dataset]])
        if args.dataset == 'SetFit/sst2':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['label_text'])
        if args.dataset == 'dbpedia_14':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['title'])
        encoded_val_dataset = encoded_val_dataset[:len(encoded_val_dataset)]


    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "")
            example["text"] = example["text"].replace("#39;", "'")
            return example
        test_dataset = test_dataset.map(replace_bad_string)

    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    concept_set = CFG.concept_set[args.dataset]
    print("concept len: ", len(concept_set))  # concept_set: list of strings, len = num_concepts

    d_name = args.dataset.replace('/', '_')
    label_prefix = "./"
    if args.labeling == 'mpnet':
        label_prefix += "mpnet_acs"
    elif args.labeling == 'simcse':
        label_prefix += "simcse_acs"
    elif args.labeling == 'angle':
        label_prefix += "angle_acs"
    elif args.labeling == 'llm':
        label_prefix += "llm_labeling"

    label_prefix += "/"
    label_prefix += d_name
    label_prefix += "/"
    
    print(f"Loading concept labels from: {label_prefix}")
    train_similarity = np.load(label_prefix + "/concept_labels_train.npy")  # (N_train, num_concepts)
    print("train_similarity shape: ", train_similarity.shape)
    if args.dataset == 'SetFit/sst2':
        val_similarity = np.load(label_prefix + "/concept_labels_val.npy")  # (N_val, num_concepts)

    # If we subsampled the train set by indices, subset the full-split concept labels identically.
    if train_select_indices is not None and int(train_similarity.shape[0]) == int(original_train_len):
        train_similarity = train_similarity[train_select_indices]
        print("train_similarity shape after subsample: ", train_similarity.shape)

    # Align label matrices with the (potentially subsampled) encoded datasets.
    train_similarity, encoded_train_dataset = _align_similarity_and_encoded(
        train_similarity, encoded_train_dataset, split_name="train"
    )
    if args.dataset == 'SetFit/sst2':
        val_similarity, encoded_val_dataset = _align_similarity_and_encoded(
            val_similarity, encoded_val_dataset, split_name="val"
        )

    # Basic shape sanity checks.
    if train_similarity.ndim != 2 or train_similarity.shape[1] != len(concept_set):
        raise ValueError(
            f"Unexpected train_similarity shape {train_similarity.shape}; expected (N, {len(concept_set)}). "
            f"Check {label_prefix}/concept_labels_train.npy and config_finegrained.concept_set for {args.dataset}."
        )
    if args.dataset == 'SetFit/sst2':
        if val_similarity.ndim != 2 or val_similarity.shape[1] != len(concept_set):
            raise ValueError(
                f"Unexpected val_similarity shape {val_similarity.shape}; expected (N, {len(concept_set)}). "
                f"Check {label_prefix}/concept_labels_val.npy and config_finegrained.concept_set for {args.dataset}."
            )

    if args.automatic_concept_correction:
        start = time.time()
        print("training intervention...")
        train_labels = np.asarray(encoded_train_dataset["label"], dtype=np.int64)
        train_similarity = np.maximum(train_similarity, 0.0)
        for j in range(len(concept_set)):
            allowed_label = get_labels(j, args.dataset)
            train_similarity[train_labels != allowed_label, j] = 0.0
        
        if args.dataset == 'SetFit/sst2':
            val_labels = np.asarray(encoded_val_dataset["label"], dtype=np.int64)
            val_similarity = np.maximum(val_similarity, 0.0)
            for j in range(len(concept_set)):
                allowed_label = get_labels(j, args.dataset)
                val_similarity[val_labels != allowed_label, j] = 0.0
        end = time.time()
        print("time of training intervention:", (end - start) / 3600, "hours")

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")
    if args.dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, val_similarity, mode="valid")
    
    test_similarity = np.zeros((len(encoded_test_dataset["label"]), 1), dtype=np.float32)
    test_loader = build_loaders(encoded_test_dataset, test_similarity, mode="test")

    print("preparing backbone")
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=5e-5)

    llama_vocab_weight = None
    if args.add_llama_logits:
        # IMPORTANT: For Llama-3, lm_head weights are not necessarily tied to input embeddings.
        # We therefore grab the *output* projection (lm_head) weights from a CausalLM head.
        # This does not add parameters to CBL; it's just an external tensor used in forward.
        lm_head_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3-8B',
            torch_dtype=torch.bfloat16,
        ).to(device)
        llama_vocab_weight = lm_head_model.get_output_embeddings().weight.detach()
        del lm_head_model
    
    if args.discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=5e-5)
    print("preparing classifier")
    total_params = sum(p.numel() for p in preLM.parameters())
    trainable_params = sum(p.numel() for p in preLM.parameters() if p.requires_grad)
    cbl_params = sum(p.numel() for p in cbl.parameters())
    trainable_params += cbl_params
    total_params += cbl_params
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} = {trainable_params/total_params:.4f} of total")
    wandb.log({"trainable_parameters": trainable_params, "trainable_ratio": trainable_params/total_params})
    
    classifier = torch.nn.Linear(args.residual_dim, len(concept_set)).to(device)
    
    if args.discrimination_loss > 0:
        opt_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)


    if args.dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100


    print("start training...")
    best_loss = float('inf')
    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    prefix += "./from_pretained_llama3_lora_cbm_" + run_name
    prefix += "/"
    prefix += d_name
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "llama3"
    cbl_name = "cbl"



    start = time.time()
    best_epoch = -1
    epochs = CFG.epoch[args.dataset]*args.epoch_multiplier
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        cbl.train()
        classifier.train()
        training_losses = {
            "concept_loss": [],
            "word_loss": [],
            "neg_entropy_loss": [],
            "reg_loss": [],
            "orthogonal_loss": [],
            "residual_penalty_loss": [],
            "intervention_gen_loss": [],
        }

        
        for i, (batch, batch_sim) in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_sim = batch_sim.to(device)

            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            llama_logits = F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
            concepts, unsup, vocabs, matched_unsup = cbl(features.float(), llama_logits=llama_logits)
            # print("concepts shape in training loop:", concepts.shape)
            # print("elastic_net_alphaunsup shape in training loop:", unsup.shape)
            # print("vocabs shape in training loop:", vocabs.shape)
            
            mask = (batch["attention_mask"][:, :-1] != 0).reshape(-1) # (B * (seq_len - 1))
            c_slice = concepts[:, :-1, :].contiguous().view(-1, concepts.shape[-1]) # (B * (seq_len - 1), C)
            batch_sim_slice = batch_sim.unsqueeze(1).expand(-1, concepts.shape[1] - 1, -1).contiguous().view(-1, batch_sim.shape[-1])
            
            valid_c = c_slice[mask]          # (N_valid, C)
            valid_sim = batch_sim_slice[mask]  # (N_valid, C)

            if args.concept_loss_type == "cosine_cubed":
                # Cosine-similarity-based concept loss against soft ACS labels
                concept_loss = -cos_sim_cubed(valid_c, valid_sim)
            elif args.concept_loss_type == "ce":
                # Cross-entropy concept loss using hard labels from ACS top concept
                hard_targets = torch.argmax(valid_sim, dim=-1)  # (N_valid,)
                concept_loss = torch.nn.CrossEntropyLoss()(valid_c, hard_targets)
            else:
                raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")
            word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
            loss = args.concept_loss * concept_loss + word_loss*args.word_loss
            reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
            
            if matched_unsup is not None:
                orthogonal_loss = torch.cosine_similarity(concepts, matched_unsup, dim=-1).mean().abs() ## TODO: check shape
                loss += args.orthogonal_loss_weight * orthogonal_loss
                training_losses["orthogonal_loss"].append(orthogonal_loss.detach().cpu().numpy())
            
            if args.residual_penalty_weight > 0:
                residual_contrib = cbl.compute_residual_contrib(unsup)
                residual_penalty = torch.mean(torch.abs(residual_contrib)) ## TODO: check logic
                loss += args.residual_penalty_weight * residual_penalty
                training_losses["residual_penalty_loss"].append(residual_penalty.detach().cpu().numpy())
                
            if args.intervention_gen_loss > 0:
                ### concepts shapes: (B, seq_len, concept_dim)
                if args.dataset == "dbpedia_14":
                    intervention_value = 150
                else:
                    intervention_value = 100

                intervened_concept = build_intervened_concepts_from_similarity(
                    concepts=concepts,
                    batch_sim=batch_sim,
                    intervention_value=intervention_value,
                    keep_other_concepts=args.intervention_keep_other_concepts,
                    use_topk=args.intervention_topk_concepts,
                    topk_k=args.intervention_topk_k,
                    log_wandb=args.intervention_topk_concepts,
                    wandb_prefix="train",
                )
                    
                # print("intervened_concept shape: ", intervened_concept.shape, intervened_concept.max(), intervened_concept.min())
                llama_logits_for_intervene = None
                if llama_logits is not None:
                    llama_logits_for_intervene = llama_logits if args.no_detach_intervention else llama_logits.detach()

                if args.no_detach_intervention:
                    vocab = cbl.intervene(unsup, intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
                else:
                    vocab = cbl.intervene(unsup.detach(), intervened_concept.detach(), llama_logits=llama_logits_for_intervene)
                intervention_gen_loss = torch.nn.CrossEntropyLoss()(vocab[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
                loss += args.intervention_gen_loss * intervention_gen_loss
                training_losses["intervention_gen_loss"].append(intervention_gen_loss.detach().cpu().numpy())
                
            loss += args.elastic_net_alpha * reg
            
            
            
            opt_prelm.zero_grad()
            opt_cbl.zero_grad()
            loss.backward()
            opt_prelm.step()
            opt_cbl.step()

            if args.discrimination_loss > 0:
                classification = classifier(mean_pooling(unsup.detach(), batch["attention_mask"]))

                # Probe loss: train the classifier to predict finegrained concept similarities from unsup.
                # This keeps the probe consistent with the concept supervision and avoids class labels.
                if args.concept_loss_type == "cosine_cubed":
                    discrimination_loss = -cos_sim_cubed(classification, batch_sim)
                elif args.concept_loss_type == "ce":
                    hard_targets = torch.argmax(batch_sim, dim=-1)
                    discrimination_loss = torch.nn.CrossEntropyLoss()(classification, hard_targets)
                else:
                    raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")
                opt_classifier.zero_grad()
                (args.discrimination_loss * discrimination_loss).backward(inputs=list(classifier.parameters()))
                opt_classifier.step()

            if args.neg_entropy_loss > 0:
                _, unsup, _, _ = cbl(features.detach().float())
                classification = classifier(mean_pooling(unsup, batch["attention_mask"]))
                p = F.softmax(classification, dim=-1)
                neg_entropy_loss = torch.sum(p * torch.log(p), dim=-1).mean()
                opt_cbl.zero_grad()
                (args.neg_entropy_loss * neg_entropy_loss).backward(inputs=list(cbl.unsup.parameters()))
                opt_cbl.step()
                training_losses["neg_entropy_loss"].append(neg_entropy_loss.detach().cpu().numpy())


            training_losses["concept_loss"].append(concept_loss.detach().cpu().numpy())
            training_losses["word_loss"].append(word_loss.detach().cpu().numpy())
            
            training_losses["reg_loss"].append(reg.detach().cpu().numpy())
            
            log = {}
            for key in training_losses.keys():
                if len(training_losses[key]) > 0:
                    print(f"{key}: {training_losses[key][-1]}", end=" ")
                    log[key] = training_losses[key][-1]
            # print(" | batch ", i+1, " / ", len(train_loader), end="\r")
            
            
            log["epoch"] = e + 1
            log["batch"] = i + 1
            wandb.log(log)
            
            if args.DEBUG and i >= 2:
                break
            
            
        avg_metrics = {}
        for key in training_losses.keys():
            if len(training_losses[key]) > 0:
                avg_metrics[key] = sum(training_losses[key]) / len(training_losses[key])
        print("Epoch ", e + 1, " training losses: ", avg_metrics)
        wandb.log({f"avg_{k}": avg_metrics[k] for k in avg_metrics.keys()})

        if args.dataset == 'SetFit/sst2':
            preLM.eval()
            cbl.eval()
            val_losses = {
                "val_concept_loss": [],
                "val_word_loss": [],
                "val_neg_entropy_loss": [],
                "val_reg_loss": [],
                "val_residual_penalty_loss": [],
                "val_orthogonal_loss": [],
                "val_intervention_gen_loss": []
            }
            for i, (batch, batch_sim) in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_sim = batch_sim.to(device)

                word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
                with torch.no_grad():
                    features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    llama_logits = F.linear(features, llama_vocab_weight) if llama_vocab_weight is not None else None
                    concepts, unsup, vocabs, matched_unsup = cbl(features.float(), llama_logits=llama_logits)
                    classification = classifier(mean_pooling(unsup, batch["attention_mask"]))
                
                mask = (batch["attention_mask"][:, :-1] != 0).reshape(-1)
                c_slice = concepts[:, :-1, :].contiguous().view(-1, concepts.shape[-1])
                batch_sim_slice = batch_sim.unsqueeze(1).expand(-1, concepts.shape[1] - 1, -1).contiguous().view(-1, batch_sim.shape[-1])
                valid_c = c_slice[mask]
                valid_sim = batch_sim_slice[mask]

                if args.concept_loss_type == "cosine_cubed":
                    concept_loss = -cos_sim_cubed(valid_c, valid_sim)
                elif args.concept_loss_type == "ce":
                    hard_targets = torch.argmax(valid_sim, dim=-1)
                    concept_loss = torch.nn.CrossEntropyLoss()(valid_c, hard_targets)
                else:
                    raise ValueError(f"Unknown concept_loss_type: {args.concept_loss_type}")
                word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
                p = F.softmax(classification, dim=-1)
                
                if args.residual_penalty_weight > 0:
                    residual_contrib = cbl.compute_residual_contrib(unsup)
                    residual_penalty = torch.mean(torch.abs(residual_contrib)) ## TODO: check logic
                    val_losses["val_residual_penalty_loss"].append(residual_penalty.detach().cpu().numpy())
                    
                if matched_unsup is not None:
                    orthogonal_loss = torch.cosine_similarity(concepts, matched_unsup, dim=-1).mean().abs() ## TODO: check shape
                    val_losses["val_orthogonal_loss"].append(orthogonal_loss.detach().cpu().numpy())
                
                if args.intervention_gen_loss > 0:
                    if args.dataset == "dbpedia_14":
                        intervention_value = 150
                    else:
                        intervention_value = 100

                    intervened_concept = build_intervened_concepts_from_similarity(
                        concepts=concepts,
                        batch_sim=batch_sim,
                        intervention_value=intervention_value,
                        keep_other_concepts=args.intervention_keep_other_concepts,
                        use_topk=args.intervention_topk_concepts,
                        topk_k=args.intervention_topk_k,
                    )
                    vocab = cbl.intervene(unsup.detach(), intervened_concept.detach(), llama_logits=llama_logits)
                    intervention_gen_loss = torch.nn.CrossEntropyLoss()(vocab[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
                    val_losses["val_intervention_gen_loss"].append(intervention_gen_loss.detach().cpu().numpy())
                
                neg_entropy_loss = torch.sum(p * torch.log(p), dim=-1).mean()
                reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
                val_losses["val_concept_loss"].append(concept_loss.detach().cpu().numpy())
                val_losses["val_word_loss"].append(word_loss.detach().cpu().numpy())
                val_losses["val_neg_entropy_loss"].append(neg_entropy_loss.detach().cpu().numpy())
                val_losses["val_reg_loss"].append(reg.detach().cpu().numpy())
                
                if args.DEBUG and i >= 2:
                    break
                
            avg_val_loss = {}
            for key in val_losses.keys():
                if len(val_losses[key]) > 0:
                    avg_val_loss[key] = sum(val_losses[key]) / len(val_losses[key])
            print("Epoch ", e + 1, " validation losses: ", avg_val_loss)
            wandb.log({f"avg_{k}": avg_val_loss[k] for k in avg_val_loss.keys()})
            avg_val_concept_loss = avg_val_loss["val_concept_loss"]
            avg_val_word_loss = avg_val_loss["val_word_loss"]


            avg_val_loss = avg_val_concept_loss + avg_val_word_loss
            if (avg_val_loss < best_loss) or (args.use_last_epoch):
                best_epoch = e + 1
                print("save model")
                best_loss = avg_val_loss
                preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
                torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")
                wandb.log({"best_model_epoch": e + 1})
            else:
                preLM.save_pretrained(prefix + model_name + "_low_score_epoch_" + str(e + 1))
                torch.save(cbl.state_dict(), prefix + cbl_name + "_low_score_epoch_" + str(e + 1) + ".pt")
        else:
            print("save model")
            preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
            torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")

        if args.DEBUG:
            break

    end = time.time()
    print("time of training CBM:", (end - start) / 3600, "hours")
    
    ## delete training objects and free GPU before evaluation
    import gc
    if llama_vocab_weight is not None:
        del llama_vocab_weight
        llama_vocab_weight = None
    del preLM, cbl, classifier, opt_prelm, opt_cbl
    
    if args.discrimination_loss > 0:
        del opt_classifier
    gc.collect()
    torch.cuda.empty_cache()
    
    ## lOAD BEST MODEL AND
    if best_epoch == -1:
        best_epoch = epochs
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    peft_path = prefix + model_name + "_epoch_" + str(best_epoch)
    preLM.load_adapter(peft_path)
    preLM.eval()

    llama_vocab_weight = None
    if args.add_llama_logits:
        from eval_metrics import get_llama_vocab_weight
        llama_vocab_weight = get_llama_vocab_weight(device)

    if args.discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    cbl.load_state_dict(torch.load(prefix + cbl_name + "_epoch_" + str(best_epoch) + ".pt", map_location=device))
    cbl.eval()

    # ── Configure evaluation ──
    intervention_value = get_intervention_value(args.dataset)
    num_steerability_samples = (
        max(1, args.samples_per_concept)
        if args.samples_per_concept is not None
        else max(1, 100 // len(concept_set))
    )
    steer_root = steerability_output_root(os.path.normpath(prefix.rstrip("/")), best_epoch, False)
    print(f"Steerability sample cache: {steer_root}")

    # ── Generate steerability texts (cached) ──
    set_seed(args.seed)
    decoded_texts_by_concept = generate_steerability_texts(
        preLM, cbl, tokenizer, concept_set, args.dataset, device,
        samples_per_concept=num_steerability_samples,
        llama_vocab_weight=llama_vocab_weight,
        keep_other_concepts=args.intervention_keep_other_concepts,
        steerability_cache_dir=steer_root,
        steerability_cache_seed=args.seed,
        interventions_per_batch=50,
    )

    # ── Generate perplexity texts (cached) ──
    ppl_texts = generate_perplexity_texts(
        cbl, preLM, tokenizer, args.seed, device,
        cache_dir=prefix, run_name=run_name,
        llama_vocab_weight=llama_vocab_weight,
    )

    # ── Concept accuracy (cosine similarity) ──
    run_concept_accuracy_cosine(preLM, cbl, test_loader, concept_set, label_prefix, device)

    # ── Weight analysis ──
    run_weight_analysis(cbl, concept_set, tokenizer)

    # ── Free model from GPU ──
    del preLM, cbl
    if llama_vocab_weight is not None:
        from eval_metrics import release_llama_vocab_weight
        release_llama_vocab_weight()
        llama_vocab_weight = None
    gc.collect()
    torch.cuda.empty_cache()

    # ── Steerability scoring (MPNet similarity) ──
    run_steerability_mpnet(
        decoded_texts_by_concept, concept_set,
        intervention_value, args.max_length, device,
    )

    # ── Perplexity computation (evaluate library loads its own LLM) ──
    compute_perplexity(ppl_texts)

    # ── RM reward scoring (optional) ──
    if not args.skip_rm:
        try:
            rm_model, rm_tokenizer_rm = load_reward_model(args.rm_model_name, device)
            run_rm_metrics(
                decoded_texts_by_concept, concept_set,
                rm_model, rm_tokenizer_rm, device,
                rm_batch_size=args.rm_batch_size,
                rm_max_text_len=args.rm_max_text_len,
            )
            del rm_model, rm_tokenizer_rm
            torch.cuda.empty_cache()
        except Exception as rm_err:
            print(f"RM evaluation failed (non-fatal): {rm_err}")

    # ── Save steerability text cache ──
    save_all_steerability_texts(steer_root, args.seed, concept_set, decoded_texts_by_concept)
    