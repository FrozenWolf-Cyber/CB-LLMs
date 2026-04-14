"""
GRPO (Group Relative Policy Optimization) training for CB-LLMs.

Proper GRPO algorithm with KL-divergence penalty to prevent deviation from the
reference (pretrained) model:

    1. Load a pretrained model from train_combined.py checkpoint (via wandb run_id or folder path).
       This becomes the FROZEN reference policy π_ref.
    2. Create a trainable copy π_θ (same architecture, same weights, but trainable).
    3. For each training step:
       a. Sample a random concept index c.
       b. Generate G trajectories from π_θ with intervention on concept c.
       c. Score each trajectory with RoBERTa classifier ensemble → rewards r_1..r_G.
       d. Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + eps), then clip.
       e. For trajectories with non-zero advantage:
          - Compute log π_θ(tokens | intervention)   (with grad, teacher-forced).
          - Compute log π_ref(tokens | intervention)  (no grad, teacher-forced).
          - KL ≈ log π_θ - log π_ref                  (per-token, averaged over sequence).
          - Policy loss = -A_i * log π_θ
          - Total loss = mean(policy_loss) + β * mean(KL)
       f. Update π_θ.
    4. Run the same evaluation suite as train_combined.py (steerability, concept prediction,
       weight analysis, perplexity).

Usage:
    python train_grpo.py --dataset SetFit/sst2 --pretrained_run_id <wandb_run_id>
    python train_grpo.py --dataset SetFit/sst2 --pretrained_path <folder_path>
"""
import argparse
import os
import gc
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
import time
from module_intervention import amplify_intervention
from utils import elastic_net_penalty, mean_pooling, eos_pooling
from steerability_cache import save_all_steerability_texts, steerability_output_root
from eval_metrics import (
    set_seed,
    get_intervention_value,
    generate_steerability_texts,
    run_steerability_roberta,
    run_steerability_mpnet,
    run_concept_accuracy_labels,
    run_weight_analysis,
    generate_perplexity_texts,
    compute_perplexity,
    load_reward_model,
    run_rm_metrics,
)
import wandb
import glob
import copy


parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--discrimination_loss", type=float, default=0.0)
parser.add_argument("--arch_type", type=str, default="residual", choices=["residual", "non_residual"])
parser.add_argument("--residual_dim", type=int, default=768)
parser.add_argument("--DEBUG", action='store_true', help="If set, use a smaller subset of data for quick debugging.")
parser.add_argument("--classifier_weight_suffixes", type=str, default="_seed42,_seed123,_seed456", 
                    help="Comma-separated list of classifier weight suffixes to test (e.g., '_seed42,_seed123,_seed456')")

# ---- Pretrained model loading ----
parser.add_argument("--pretrained_run_id", type=str, default=None,
                    help="Wandb run_id from train_combined.py. Will auto-locate the saved checkpoint folder.")
parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Exact folder path to pretrained checkpoint (the prefix dir containing llama3_epoch_* and cbl_epoch_*.pt). "
                         "If given, --pretrained_run_id is ignored.")

# ---- GRPO hyperparameters ----
parser.add_argument("--grpo_epochs", type=int, default=1, help="Number of GRPO training epochs over the dataset.")
parser.add_argument("--grpo_loss_weight", type=float, default=1.0, help="Weight for the GRPO policy gradient loss.")
parser.add_argument("--grpo_kl_weight", type=float, default=0.1, help="Weight β for KL divergence penalty against reference model.")
parser.add_argument("--grpo_num_trajectories", type=int, default=4, help="Number of rollouts (G) per GRPO step.")
parser.add_argument("--grpo_gen_length", type=int, default=100, help="Max generation length for GRPO rollouts.")
parser.add_argument("--grpo_clip_advantage", type=float, default=5.0, help="Clip GRPO advantages to [-clip, clip].")
parser.add_argument("--grpo_lr", type=float, default=1e-5, help="Learning rate for GRPO fine-tuning.")
parser.add_argument("--grpo_steps_per_epoch", type=int, default=-1, help="Max GRPO steps per epoch. -1 = full dataset.")
parser.add_argument("--concept_distill_weight", type=float, default=0.0, help="Weight for concept prediction distillation loss (CE between policy and reference model concepts on real data). 0 disables it.")

# ---- Evaluation hyperparameters ----
parser.add_argument("--samples_per_concept", type=int, default=50,
                    help="Steerability samples per concept. Default 50.")
parser.add_argument("--rm_model_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    help="Reward model for RM evaluation.")
parser.add_argument("--rm_batch_size", type=int, default=4,
                    help="Batch size for RM evaluation.")
parser.add_argument("--rm_max_text_len", type=int, default=1024,
                    help="Max text length for RM evaluation.")
parser.add_argument("--skip_rm", action="store_true",
                    help="Skip RM reward evaluation after training.")

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text


    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encoded_text.items()}
        return t

    def __len__(self):
        return len(self.encoded_text['input_ids'])


def build_loaders(encoded_text, mode):
    dataset = ClassificationDataset(encoded_text)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader


def find_pretrained_checkpoint(run_id, dataset):
    """Locate the pretrained checkpoint folder from a wandb run_id.
    
    Convention from train_combined.py:
        prefix = ./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/
        Best epoch:  llama3_epoch_{N}  and  cbl_epoch_{N}.pt
        Other epoch: llama3_low_score_epoch_{N}  and  cbl_low_score_epoch_{N}.pt
    
    Strategy: prefer best epoch (no 'low_score' in name), fallback to last epoch.
    """
    d_name = dataset.replace('/', '_')
    prefix = os.path.join(".", f"from_pretained_llama3_lora_cbm_{run_id}", d_name)
    if not os.path.isdir(prefix):
        raise FileNotFoundError(f"Checkpoint directory not found: {prefix}")
    
    # Find all cbl checkpoint files
    cbl_files = sorted(glob.glob(os.path.join(prefix, "cbl_epoch_*.pt")))
    cbl_low_files = sorted(glob.glob(os.path.join(prefix, "cbl_low_score_epoch_*.pt")))
    
    # Best epoch files (no 'low_score')
    best_cbl_files = [f for f in cbl_files if "low_score" not in f]
    
    if best_cbl_files:
        # Pick the highest best epoch
        best_epoch = max(
            int(os.path.basename(f).replace("cbl_epoch_", "").replace(".pt", ""))
            for f in best_cbl_files
        )
        peft_path = os.path.join(prefix, f"llama3_epoch_{best_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_epoch_{best_epoch}.pt")
    else:
        # No best epoch found, use the last low_score epoch
        if not cbl_low_files:
            raise FileNotFoundError(f"No checkpoint files found in {prefix}")
        last_epoch = max(
            int(os.path.basename(f).replace("cbl_low_score_epoch_", "").replace(".pt", ""))
            for f in cbl_low_files
        )
        peft_path = os.path.join(prefix, f"llama3_low_score_epoch_{last_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_low_score_epoch_{last_epoch}.pt")
    
    if not os.path.isdir(peft_path):
        raise FileNotFoundError(f"LoRA adapter directory not found: {peft_path}")
    if not os.path.isfile(cbl_path):
        raise FileNotFoundError(f"CBL checkpoint not found: {cbl_path}")
    
    print(f"Found pretrained checkpoint: peft={peft_path}, cbl={cbl_path}")
    return peft_path, cbl_path


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    set_seed(args.seed)

    # Validate: need either run_id or path
    if args.pretrained_run_id is None and args.pretrained_path is None:
        raise ValueError("Must provide either --pretrained_run_id or --pretrained_path")

    wandb.init(project="cbm-generation-new", name=f"grpo-{args.dataset}-seed{args.seed}-{args.pretrained_run_id}",
               config=vars(args))
    
    run_name = wandb.run.id
    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    test_dataset = load_dataset(args.dataset, split='test')
    if args.dataset == 'SetFit/sst2':
        val_dataset = load_dataset(args.dataset, split='validation')

    if args.dataset != 'SetFit/sst2':
        d_list = []
        for i in range(CFG.class_num[args.dataset]):
            d_list.append(
                train_dataset.filter(lambda e: e['label'] == i).select(range(100000 // CFG.class_num[args.dataset])))
        train_dataset = concatenate_datasets(d_list)

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

    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="train")
    if args.dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")
    test_loader = build_loaders(encoded_test_dataset, mode="test")

    # ======================================================================
    # Load pretrained checkpoint from train_combined.py
    # ======================================================================
    if args.pretrained_path is not None:
        # Direct folder path: expect llama3_epoch_* and cbl_epoch_*.pt inside
        pretrained_prefix = args.pretrained_path.rstrip("/")
        # Auto-detect best/last epoch
        cbl_files = sorted(glob.glob(os.path.join(pretrained_prefix, "cbl_epoch_*.pt")))
        cbl_low_files = sorted(glob.glob(os.path.join(pretrained_prefix, "cbl_low_score_epoch_*.pt")))
        best_cbl_files = [f for f in cbl_files if "low_score" not in f]
        if best_cbl_files:
            best_epoch = max(
                int(os.path.basename(f).replace("cbl_epoch_", "").replace(".pt", ""))
                for f in best_cbl_files
            )
            peft_path = os.path.join(pretrained_prefix, f"llama3_epoch_{best_epoch}")
            cbl_path = os.path.join(pretrained_prefix, f"cbl_epoch_{best_epoch}.pt")
        elif cbl_low_files:
            last_epoch = max(
                int(os.path.basename(f).replace("cbl_low_score_epoch_", "").replace(".pt", ""))
                for f in cbl_low_files
            )
            peft_path = os.path.join(pretrained_prefix, f"llama3_low_score_epoch_{last_epoch}")
            cbl_path = os.path.join(pretrained_prefix, f"cbl_low_score_epoch_{last_epoch}.pt")
        else:
            raise FileNotFoundError(f"No checkpoint files found in {pretrained_prefix}")
        print(f"Using pretrained checkpoint: peft={peft_path}, cbl={cbl_path}")
    else:
        peft_path, cbl_path = find_pretrained_checkpoint(args.pretrained_run_id, args.dataset)

    wandb.log({"pretrained_peft_path": peft_path, "pretrained_cbl_path": cbl_path})

    # ======================================================================
    # Build REFERENCE model (frozen π_ref) and POLICY model (trainable π_θ)
    # ======================================================================
    print("preparing reference model (frozen)...")
    ref_preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    ref_preLM.load_adapter(peft_path)
    ref_preLM.eval()
    for p in ref_preLM.parameters():
        p.requires_grad = False

    if args.arch_type == "non_residual":
        ref_cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        ref_cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    ref_cbl.load_state_dict(torch.load(cbl_path, map_location=device), strict=False)  # strict=False to allow missing keys if arch differs
    ref_cbl.eval()
    for p in ref_cbl.parameters():
        p.requires_grad = False

    print("preparing policy model (trainable)...")
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    # Load the pretrained adapter weights into the new PEFT model
    # We need to load state dict from the saved adapter
    from peft import set_peft_model_state_dict
    import safetensors
    adapter_weights_path = os.path.join(peft_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(peft_path, "adapter_model.bin")
        adapter_state = torch.load(adapter_weights_path, map_location=device)
    else:
        from safetensors.torch import load_file
        adapter_state = load_file(adapter_weights_path, device=str(device))
    set_peft_model_state_dict(preLM, adapter_state)
    
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=args.grpo_lr)
    
    if args.arch_type == "non_residual":
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device), strict=False)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=args.grpo_lr)

    print("preparing classifier")
    total_params = sum(p.numel() for p in preLM.parameters())
    trainable_params = sum(p.numel() for p in preLM.parameters() if p.requires_grad)
    cbl_params = sum(p.numel() for p in cbl.parameters())
    trainable_params += cbl_params
    total_params += cbl_params
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} = {trainable_params/total_params:.4f} of total")
    wandb.log({"trainable_parameters": trainable_params, "trainable_ratio": trainable_params/total_params})

    # Set intervention value once (used by GRPO)
    if args.dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100

    # Load and compile RoBERTa classifiers for GRPO reward scoring
    grpo_classifiers = []
    roberta_tokenizer_grpo = RobertaTokenizerFast.from_pretrained('roberta-base')
    d_grpo = args.dataset.replace('/', '_')
    # Default classifier + seeded classifiers
    grpo_clf_paths = [d_grpo + "_classifier.pt"]
    clf_suffixes = [s.strip() for s in args.classifier_weight_suffixes.split(',')]
    for suffix in clf_suffixes:
        grpo_clf_paths.append(d_grpo + f"_classifier{suffix}.pt")
    for clf_path in grpo_clf_paths:
        clf = Roberta_classifier(len(concept_set)).to(device)
        clf.load_state_dict(torch.load(clf_path, map_location=device))
        clf.eval()
        for p in clf.parameters():
            p.requires_grad = False
        try:
            clf = torch.compile(clf)
        except Exception as compile_err:
            print(f"  Warning: torch.compile failed for {clf_path}, using eager mode: {compile_err}")
        grpo_classifiers.append(clf)
    print(f"Loaded {len(grpo_classifiers)} RoBERTa classifiers for GRPO (compiled)")

    print("start GRPO training...")
    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    prefix += "./from_pretained_llama3_lora_grpo_" + run_name
    prefix += "/"
    prefix += d_name
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "llama3"
    cbl_name = "cbl"

    # ======= DEBUG: Test batched generate + reward scoring =======
    def _debug_generate_and_score(label, intervene, target_concept_idx):
        """Generate num_test trajectories, score with RoBERTa ensemble, compute advantages."""
        print("=" * 60)
        print(f"DEBUG: {label}")
        print("=" * 60)
        with torch.no_grad():
            batch_ids, _ = cbl.generate_batch(
                test_input, preLM, num_samples=num_test,
                intervene=intervene, length=50
            )
        decoded_texts = []
        for g in range(num_test):
            tokens = batch_ids[g][~torch.isin(batch_ids[g], special_tokens)]
            text = tokenizer.decode(tokens)
            decoded_texts.append(text)

        # Score with RoBERTa ensemble for each concept
        rewards_per_concept = {}
        for c_idx in range(len(concept_set)):
            rewards = [0.0] * num_test
            non_empty = [g for g, t in enumerate(decoded_texts) if len(t.strip()) > 0]
            if non_empty:
                non_empty_texts = [decoded_texts[g] for g in non_empty]
                rob_enc = roberta_tokenizer_grpo(
                    non_empty_texts, return_tensors='pt', truncation=True,
                    max_length=512, padding=True
                ).to(device)
                rob_input = {"input_ids": rob_enc["input_ids"], "attention_mask": rob_enc["attention_mask"]}
                total_probs = torch.zeros(len(non_empty_texts), device=device)
                for grpo_clf in grpo_classifiers:
                    clf_logits = grpo_clf(rob_input)
                    probs = F.softmax(clf_logits, dim=-1)[:, c_idx]
                    total_probs += probs
                total_probs /= len(grpo_classifiers)
                for idx_ne, g in enumerate(non_empty):
                    rewards[g] = total_probs[idx_ne].item()
            rewards_per_concept[c_idx] = rewards

        # Print trajectories with rewards
        for g in range(num_test):
            reward_str = ", ".join([f"{concept_set[c]}={rewards_per_concept[c][g]:.3f}" for c in range(len(concept_set))])
            print(f"  [{g+1}/{num_test}] ({len(decoded_texts[g].split())} words): {decoded_texts[g][:200]}")
            print(f"    Rewards: {reward_str}")

        # Compute advantages for the target concept
        rewards_t = torch.tensor(rewards_per_concept[target_concept_idx], device=device, dtype=torch.float32)
        if rewards_t.std() > 1e-8:
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards_t)
        advantages = advantages.clamp(-args.grpo_clip_advantage, args.grpo_clip_advantage)
        print(f"  Target concept: {concept_set[target_concept_idx]} (idx={target_concept_idx})")
        print(f"  Rewards:    {[round(r, 4) for r in rewards_per_concept[target_concept_idx]]}")
        print(f"  Mean reward: {rewards_t.mean().item():.4f}, Std: {rewards_t.std().item():.4f}")
        print(f"  Advantages: {[round(a, 4) for a in advantages.tolist()]}")
        print(f"  Non-zero advantages: {(advantages.abs() > 1e-8).sum().item()}/{num_test}")
        print("=" * 60)

    if args.DEBUG:
        test_input = torch.tensor([tokenizer.encode("")]).to(device)
        num_test = min(args.grpo_num_trajectories, 8)
        special_tokens = torch.tensor([128000, 128001]).to(device)
        preLM.eval()
        cbl.eval()
        with torch.no_grad():
            # --- No intervention ---
            _debug_generate_and_score(
                label="generate_batch (no intervention)",
                intervene=None,
                target_concept_idx=0
            )

            # --- Intervention on first concept ---
            test_intervene_0 = [0] * len(concept_set)
            test_intervene_0[0] = intervention_value
            _debug_generate_and_score(
                label=f"generate_batch (intervene concept 0 = '{concept_set[0]}', value={intervention_value})",
                intervene=test_intervene_0,
                target_concept_idx=0
            )

            # --- Intervention on last concept ---
            last_idx = len(concept_set) - 1
            test_intervene_last = [0] * len(concept_set)
            test_intervene_last[last_idx] = intervention_value
            _debug_generate_and_score(
                label=f"generate_batch (intervene concept {last_idx} = '{concept_set[last_idx]}', value={intervention_value})",
                intervene=test_intervene_last,
                target_concept_idx=last_idx
            )
        preLM.train()
        cbl.train()

    start = time.time()
    epochs = args.grpo_epochs
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        cbl.train()
        training_losses = {
            "grpo_policy_loss": [],
            "grpo_kl_loss": [],
            "grpo_total_loss": [],
            "grpo_mean_reward": [],
            "non_zero_grpo_advantages": [],
            "concept_distill_loss": []
        }

        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.grpo_steps_per_epoch > 0 and i >= args.grpo_steps_per_epoch:
                break

            # ======= GRPO STEP =======
            grpo_concept_idx = torch.randint(0, len(concept_set), (1,)).item()
            grpo_intervene = [0] * len(concept_set)
            grpo_intervene[grpo_concept_idx] = intervention_value

            # Phase 1: Generate G trajectories in parallel & score with RoBERTa ensemble (no grad)
            gen_input = torch.tensor([tokenizer.encode("")]).to(device)
            special_tokens_mask = torch.tensor([128000, 128001]).to(device)

            preLM.eval()
            cbl.eval()
            with torch.no_grad():
                # Batched generation — all trajectories at once
                text_ids_batch, _ = cbl.generate_batch(
                    gen_input, preLM, num_samples=args.grpo_num_trajectories,
                    intervene=grpo_intervene, length=args.grpo_gen_length
                )

                # Decode all trajectories and re-encode
                generated_seqs = []
                decoded_texts = []
                for g in range(args.grpo_num_trajectories):
                    decoded = tokenizer.decode(
                        text_ids_batch[g][~torch.isin(text_ids_batch[g], special_tokens_mask)]
                    )
                    re_encoded = torch.tensor([tokenizer.encode(decoded)]).to(device)
                    generated_seqs.append(re_encoded.detach())
                    decoded_texts.append(decoded)

                # Batched reward scoring with RoBERTa ensemble
                grpo_rewards = [0.0] * args.grpo_num_trajectories
                non_empty_indices = [g for g, t in enumerate(decoded_texts) if len(t.strip()) > 0]
                if non_empty_indices:
                    non_empty_texts = [decoded_texts[g] for g in non_empty_indices]
                    rob_enc = roberta_tokenizer_grpo(
                        non_empty_texts, return_tensors='pt', truncation=True,
                        max_length=512, padding=True
                    ).to(device)
                    rob_input = {"input_ids": rob_enc["input_ids"], "attention_mask": rob_enc["attention_mask"]}

                    total_probs = torch.zeros(len(non_empty_texts), device=device)
                    for grpo_clf in grpo_classifiers:
                        clf_logits = grpo_clf(rob_input)
                        probs = F.softmax(clf_logits, dim=-1)[:, grpo_concept_idx]
                        total_probs += probs
                    total_probs /= len(grpo_classifiers)

                    for idx, g in enumerate(non_empty_indices):
                        grpo_rewards[g] = total_probs[idx].item()

                if args.DEBUG:
                    for g in range(args.grpo_num_trajectories):
                        print(f"GRPO Trajectory {g+1}/{args.grpo_num_trajectories}:")
                        print("  Decoded text:", decoded_texts[g])
                        print("  Reward:", grpo_rewards[g])
                        print("-" * 50)
                        
                ## log the lowest and highest reward text+reward to wandb for debugging
                wandb.log({
                    "grpo_debug_lowest_reward": min(grpo_rewards),
                    "grpo_debug_highest_reward": max(grpo_rewards),
                    "grpo_debug_lowest_text": decoded_texts[grpo_rewards.index(min(grpo_rewards))],
                    "grpo_debug_highest_text": decoded_texts[grpo_rewards.index(max(grpo_rewards))]
                })

            preLM.train()
            cbl.train()

            # Phase 2: Group-relative advantages
            grpo_rewards_t = torch.tensor(grpo_rewards, device=device, dtype=torch.float32)
            if grpo_rewards_t.std() > 1e-8:
                grpo_advantages = (grpo_rewards_t - grpo_rewards_t.mean()) / (grpo_rewards_t.std() + 1e-8)
            else:
                grpo_advantages = torch.zeros_like(grpo_rewards_t)
            grpo_advantages = grpo_advantages.clamp(-args.grpo_clip_advantage, args.grpo_clip_advantage)

            # Phase 3: Recompute log-probs WITH gradient + KL against reference (batched teacher-forced)
            valid_indices = [g for g in range(args.grpo_num_trajectories)
                             if generated_seqs[g].shape[1] > 1 and grpo_advantages[g].abs().item() > 1e-8]

            if valid_indices:
                max_seq_len = max(generated_seqs[g].shape[1] for g in valid_indices)
                padded_inputs = []
                padded_targets = []
                attn_masks = []
                valid_advantages = []

                for g in valid_indices:
                    seq = generated_seqs[g]  # (1, seq_len)
                    seq_len = seq.shape[1] - 1
                    pad_len = max_seq_len - 1 - seq_len
                    seq_input = seq[:, :-1]
                    seq_target = seq[:, 1:]
                    if pad_len > 0:
                        seq_input = F.pad(seq_input, (0, pad_len), value=tokenizer.pad_token_id)
                        seq_target = F.pad(seq_target, (0, pad_len), value=0)
                        attn = torch.cat([torch.ones(1, seq_len, device=device),
                                          torch.zeros(1, pad_len, device=device)], dim=1)
                    else:
                        attn = torch.ones(1, seq_len, device=device)
                    padded_inputs.append(seq_input)
                    padded_targets.append(seq_target)
                    attn_masks.append(attn)
                    valid_advantages.append(grpo_advantages[g])

                batch_input = torch.cat(padded_inputs, dim=0)     # (V, max_seq_len-1)
                batch_target = torch.cat(padded_targets, dim=0)   # (V, max_seq_len-1)
                batch_attn = torch.cat(attn_masks, dim=0).long()  # (V, max_seq_len-1)
                valid_adv = torch.stack(valid_advantages)          # (V,)

                # --- Policy model forward (with grad) ---
                feats_g = preLM(input_ids=batch_input, attention_mask=batch_attn).last_hidden_state
                concepts_g, unsup_g, _, _ = cbl(feats_g.float())

                intervened_g = torch.zeros(len(valid_indices), batch_input.shape[1], len(concept_set), device=device)
                intervened_g[:, :, grpo_concept_idx] = intervention_value

                vocab_logits_g = cbl.intervene(unsup_g, intervened_g)
                log_probs_g = F.log_softmax(vocab_logits_g, dim=-1)
                token_log_probs_g = log_probs_g.gather(2, batch_target.unsqueeze(-1)).squeeze(-1)
                # Mask out padded positions
                token_log_probs_g = token_log_probs_g * batch_attn.float()
                mean_log_probs = token_log_probs_g.sum(dim=1) / batch_attn.sum(dim=1).float()

                # --- Reference model forward (no grad) ---
                with torch.no_grad():
                    ref_feats_g = ref_preLM(input_ids=batch_input, attention_mask=batch_attn).last_hidden_state
                    ref_concepts_g, ref_unsup_g, _, _ = ref_cbl(ref_feats_g.float())

                    ref_intervened_g = torch.zeros(len(valid_indices), batch_input.shape[1], len(concept_set), device=device)
                    ref_intervened_g[:, :, grpo_concept_idx] = intervention_value

                    ref_vocab_logits_g = ref_cbl.intervene(ref_unsup_g, ref_intervened_g)
                    ref_log_probs_g = F.log_softmax(ref_vocab_logits_g, dim=-1)
                    
                # --- KL divergence: full-vocab KL(π_θ || π_ref) per token, always ≥ 0 ---
                # KL(π_θ || π_ref) = Σ_a π_θ(a) [log π_θ(a) - log π_ref(a)]
                policy_probs_g = F.softmax(vocab_logits_g, dim=-1)  # (V, seq_len, vocab_size)
                kl_per_token = (policy_probs_g * (log_probs_g - ref_log_probs_g)).sum(dim=-1)  # (V, seq_len)
                kl_per_token = kl_per_token * batch_attn.float()
                kl_per_seq = kl_per_token.sum(dim=1) / batch_attn.sum(dim=1).float()
                kl_loss = kl_per_seq.mean()

                # --- Policy gradient loss ---
                policy_loss = (-valid_adv * mean_log_probs).mean()

                # --- Concept prediction distillation loss ---
                # concepts_g are ReLU'd (used as logits), ref_concepts_g → softmax → soft targets
                # Only on non-padded positions (batch_attn)
                concept_distill_loss = torch.tensor(0.0, device=device)
                if args.concept_distill_weight > 0:
                    concept_mask_flat = batch_attn.reshape(-1).bool()  # (V * seq_len,)
                    policy_concept_flat = concepts_g.reshape(-1, len(concept_set))  # (V * seq_len, C)
                    with torch.no_grad():
                        ref_concept_targets = F.softmax(ref_concepts_g.reshape(-1, len(concept_set)), dim=-1)
                    concept_distill_loss = F.cross_entropy(
                        policy_concept_flat[concept_mask_flat],
                        ref_concept_targets[concept_mask_flat]
                    )

                # --- Total GRPO loss ---
                grpo_total_loss = args.grpo_loss_weight * policy_loss + args.grpo_kl_weight * kl_loss + args.concept_distill_weight * concept_distill_loss

                opt_prelm.zero_grad()
                opt_cbl.zero_grad()
                grpo_total_loss.backward()
                opt_prelm.step()
                opt_cbl.step()

                training_losses["grpo_policy_loss"].append(policy_loss.detach().cpu().numpy())
                training_losses["grpo_kl_loss"].append(kl_loss.detach().cpu().numpy())
                training_losses["concept_distill_loss"].append(concept_distill_loss.detach().cpu().numpy())
                training_losses["grpo_total_loss"].append(grpo_total_loss.detach().cpu().numpy())
                training_losses["grpo_mean_reward"].append(grpo_rewards_t.mean().item())
                
            training_losses["non_zero_grpo_advantages"].append((grpo_advantages.abs() > 1e-8).sum().item())
            if args.DEBUG:
                print(f"GRPO debug - rewards: {grpo_rewards}, advantages: {grpo_advantages.tolist()}")

            
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

        # Save after each epoch
        print("save model")
        preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
        torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")

        if args.DEBUG:
            break

    end = time.time()
    print("time of GRPO training:", (end - start) / 3600, "hours")
    
    # ── Cleanup training models ──
    del ref_preLM, ref_cbl
    del opt_prelm, opt_cbl
    for clf in grpo_classifiers:
        del clf
    del grpo_classifiers
    gc.collect()
    torch.cuda.empty_cache()

    # ── Load best (latest) epoch for evaluation ──
    best_epoch = epochs
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    peft_path_eval = prefix + model_name + "_epoch_" + str(best_epoch)
    preLM.load_adapter(peft_path_eval)
    preLM.eval()
    if args.arch_type == "non_residual":
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    cbl.load_state_dict(torch.load(prefix + cbl_name + "_epoch_" + str(best_epoch) + ".pt", map_location=device), strict=False)
    cbl.eval()

    # ── Configure evaluation ──
    n_samples = (
        max(1, args.samples_per_concept)
        if args.samples_per_concept is not None
        else max(1, 100 // len(concept_set))
    )
    steer_root = steerability_output_root(os.path.normpath(prefix.rstrip("/")), best_epoch, False)
    classifier_suffixes = [s.strip() for s in args.classifier_weight_suffixes.split(',')]

    # ── Generate steerability texts (cached) ──
    set_seed(args.seed)
    decoded_texts_by_concept = generate_steerability_texts(
        preLM, cbl, tokenizer, concept_set, args.dataset, device,
        samples_per_concept=n_samples,
        steerability_cache_dir=steer_root,
        steerability_cache_seed=args.seed,
        interventions_per_batch=50,
    )

    # ── Generate perplexity texts (cached) ──
    ppl_texts = generate_perplexity_texts(
        cbl, preLM, tokenizer, args.seed, device,
        cache_dir=prefix, run_name=run_name,
    )

    # ── Concept prediction accuracy ──
    run_concept_accuracy_labels(preLM, cbl, test_loader, concept_set, encoded_test_dataset, device)

    # ── Weight analysis ──
    run_weight_analysis(cbl, concept_set, tokenizer)

    # ── Free model from GPU ──
    del preLM, cbl
    gc.collect()
    torch.cuda.empty_cache()

    # ── Steerability scoring (RoBERTa classifiers — small models) ──
    run_steerability_roberta(
        decoded_texts_by_concept, concept_set, args.dataset, device,
        classifier_weight_suffixes=classifier_suffixes,
    )

    # ── Steerability scoring (MPNet cosine similarity top-k) ──
    intervention_value_eval = get_intervention_value(args.dataset)
    run_steerability_mpnet(
        decoded_texts_by_concept, concept_set,
        intervention_value_eval, args.max_length, device,
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

