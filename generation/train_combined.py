import argparse
import os
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
import wandb
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
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
parser.add_argument("--classifier_weight_suffixes", type=str, default="_seed42,_seed123,_seed456", 
                    help="Comma-separated list of classifier weight suffixes to test (e.g., '_seed42,_seed123,_seed456')")
parser.add_argument("--grpo_loss_weight", type=float, default=0.0, help="Weight for GRPO steerability loss. 0 disables it.")
parser.add_argument("--grpo_warmup_steps", type=int, default=300, help="Global training steps before GRPO kicks in.")
parser.add_argument("--grpo_num_trajectories", type=int, default=4, help="Number of rollouts (G) per GRPO step.")
parser.add_argument("--grpo_gen_length", type=int, default=100, help="Max generation length for GRPO rollouts.")
parser.add_argument("--grpo_clip_advantage", type=float, default=5.0, help="Clip GRPO advantages to [-clip, clip].")

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



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    set_seed(args.seed)

    wandb.init(project="cbm-generation-new", name=f"train-{args.dataset}-seed{args.seed}",
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

    print("preparing backbone")
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=5e-5)
    
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

    # Set intervention value once (used by both intervention_gen_loss and GRPO)
    if args.dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100

    # Load and compile RoBERTa classifiers for GRPO reward scoring
    grpo_classifiers = []
    if args.grpo_loss_weight > 0:
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
    epochs = CFG.epoch[args.dataset]
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
            "grpo_loss": [],
            "grpo_mean_reward": [],
            "non_zero_grpo_advantages": []
        }

        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            concepts, unsup, vocabs, matched_unsup = cbl(features.float())
            # print("concepts shape in training loop:", concepts.shape)
            # print("elastic_net_alphaunsup shape in training loop:", unsup.shape)
            # print("vocabs shape in training loop:", vocabs.shape)
            
            concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label.reshape(-1))
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
                concept_label_raw = batch["label"].view(-1, 1) ## shape: (B, 1)
                
                if args.dataset == "dbpedia_14":
                    intervention_value = 150
                else:
                    intervention_value = 100
                    
                intervened_concept = torch.zeros_like(concepts, device=device) ## shape: (B, seq_len, concept_dim)
                for b in range(concepts.shape[0]):
                    intervened_concept[b, :, concept_label_raw[b].item()] = intervention_value
                    
                # print("intervened_concept shape: ", intervened_concept.shape, intervened_concept.max(), intervened_concept.min())
                if args.no_detach_intervention:
                    vocab = cbl.intervene(unsup, intervened_concept.detach())
                else:
                    vocab = cbl.intervene(unsup.detach(), intervened_concept.detach())
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
                discrimination_loss = torch.nn.CrossEntropyLoss()(classification, batch["label"])
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

            # ======= GRPO STEERABILITY LOSS (activated after warmup) =======
            grpo_step = e * len(train_loader) + i
            if args.DEBUG or (args.grpo_loss_weight > 0 and grpo_step >= args.grpo_warmup_steps):
                grpo_concept_idx = torch.randint(0, len(concept_set), (1,)).item()
                grpo_intervene = [0] * len(concept_set)
                grpo_intervene[grpo_concept_idx] = intervention_value

                # Phase 1: Generate G trajectories & score with RoBERTa ensemble (no grad)
                generated_seqs = []
                grpo_rewards = []
                gen_input = torch.tensor([tokenizer.encode("")]).to(device)

                preLM.eval()
                cbl.eval()
                with torch.no_grad():
                    for g in range(args.grpo_num_trajectories):
                        text_ids, _ = cbl.generate(
                            gen_input, preLM, intervene=grpo_intervene,
                            length=args.grpo_gen_length
                        )

                        decoded = tokenizer.decode(
                            text_ids[0][~torch.isin(text_ids[0], torch.tensor([128000, 128001]).to(device))]
                        )
                        # Re-encode the decoded text to get clean token IDs (without special tokens)
                        re_encoded = torch.tensor([tokenizer.encode(decoded)]).to(device)
                        generated_seqs.append(re_encoded.detach())

                        if len(decoded.strip()) == 0:
                            grpo_rewards.append(0.0)
                            continue

                        rob_enc = roberta_tokenizer_grpo(
                            decoded, return_tensors='pt', truncation=True, max_length=512
                        ).to(device)
                        rob_input = {"input_ids": rob_enc["input_ids"], "attention_mask": rob_enc["attention_mask"]}

                        reward = 0.0
                        for grpo_clf in grpo_classifiers:
                            logits = grpo_clf(rob_input)
                            prob = F.softmax(logits, dim=-1)[0, grpo_concept_idx].item()
                            reward += prob
                        reward /= len(grpo_classifiers)
                        grpo_rewards.append(reward)
                        
                        if args.DEBUG:
                            print(f"GRPO Trajectory {g+1}/{args.grpo_num_trajectories}:")
                            print("  Decoded text:", decoded)
                            print("  Reward:", reward)
                            print("-" * 50)

                preLM.train()
                cbl.train()

                # Phase 2: Group-relative advantages
                grpo_rewards_t = torch.tensor(grpo_rewards, device=device, dtype=torch.float32)
                if grpo_rewards_t.std() > 1e-8:
                    grpo_advantages = (grpo_rewards_t - grpo_rewards_t.mean()) / (grpo_rewards_t.std() + 1e-8)
                else:
                    grpo_advantages = torch.zeros_like(grpo_rewards_t)
                grpo_advantages = grpo_advantages.clamp(-args.grpo_clip_advantage, args.grpo_clip_advantage)

                # Phase 3: Recompute log-probs WITH gradient (teacher-forced, single fwd per traj)
                grpo_loss_parts = []
                for g in range(args.grpo_num_trajectories):
                    seq = generated_seqs[g]  # (1, seq_len)
                    if seq.shape[1] <= 1 or grpo_advantages[g].abs().item() < 1e-8:
                        continue

                    seq_input = seq[:, :-1]
                    seq_target = seq[:, 1:]
                    seq_attn = torch.ones_like(seq_input)

                    feats_g = preLM(input_ids=seq_input, attention_mask=seq_attn).last_hidden_state
                    _, unsup_g, _, _ = cbl(feats_g.float())

                    intervened_g = torch.zeros(1, seq_input.shape[1], len(concept_set), device=device)
                    intervened_g[:, :, grpo_concept_idx] = intervention_value

                    vocab_logits_g = cbl.intervene(unsup_g, intervened_g)
                    log_probs_g = F.log_softmax(vocab_logits_g, dim=-1)
                    token_log_probs_g = log_probs_g.gather(2, seq_target.unsqueeze(-1)).squeeze(-1)
                    mean_log_prob_g = token_log_probs_g.mean()

                    grpo_loss_parts.append(-grpo_advantages[g] * mean_log_prob_g)

                if len(grpo_loss_parts) > 0:
                    grpo_loss_val = torch.stack(grpo_loss_parts).mean()
                    opt_prelm.zero_grad()
                    opt_cbl.zero_grad()
                    (args.grpo_loss_weight * grpo_loss_val).backward()
                    opt_prelm.step()
                    opt_cbl.step()
                    training_losses["grpo_loss"].append(grpo_loss_val.detach().cpu().numpy())
                    training_losses["grpo_mean_reward"].append(grpo_rewards_t.mean().item())
                    
                training_losses["non_zero_grpo_advantages"].append((grpo_advantages.abs() > 1e-8).sum().item())
                if args.DEBUG:
                    print(f"GRPO debug - rewards: {grpo_rewards}, advantages: {grpo_advantages.tolist()}")


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
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
                word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
                with torch.no_grad():
                    features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    concepts, unsup, vocabs, matched_unsup = cbl(features.float())
                    classification = classifier(mean_pooling(unsup, batch["attention_mask"]))
                concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label.reshape(-1))
                word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
                discrimination_loss = torch.nn.CrossEntropyLoss()(classification, batch["label"])
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
                    # print("concept_label shape: ", concept_label.shape, concepts.shape)
                    intervened_concept = torch.zeros_like(concepts, device=device)

                    # ---- BEFORE ----
                    # print("Before intervention:")
                    # print("  min:", intervened_concept.min().item())
                    # print("  max:", intervened_concept.max().item())

                    # Apply intervention
                    seq_len = concept_label.size(1)
                    mask = (concept_label == 1)

                    intervened_concept[:, :seq_len, 1] = mask.float() * intervention_value

                    # ---- Counter AFTER intervention ----
                    # counter = (intervened_concept[:, :, 1] == intervention_value).sum().item()

                    # print("Counter:", counter)
                    # print("After intervention:")
                    # print("  min:", intervened_concept.min().item())
                    # print("  max:", intervened_concept.max().item())

                    # # Optional: how many positions activated
                    # print("Activated positions:", mask.sum().item())
                    # print("-" * 50)
                    vocab = cbl.intervene(unsup.detach(), intervened_concept.detach())
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
            if avg_val_loss < best_loss:
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
    
    ## delete previous models to save space
    import gc
    del preLM, cbl, classifier, opt_prelm, opt_cbl
    
    if args.discrimination_loss > 0:
        del opt_classifier
    torch.cuda.empty_cache()
    gc.collect()
    
    ## lOAD BEST MODEL AND
    if best_epoch == -1:
        best_epoch = epochs
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    peft_path = prefix + model_name + "_epoch_" + str(best_epoch)
    preLM.load_adapter(peft_path)
    preLM.eval()
    if args.discrimination_loss > 0:
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    cbl.load_state_dict(torch.load(prefix + cbl_name + "_epoch_" + str(best_epoch) + ".pt", map_location=device))
    cbl.eval()
        
    
    
    
    ### TEST STEERABILITY AFTER TRAINING

    
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    classifier_path = args.dataset.replace('/', '_') + "_classifier.pt"
    ## this is default.
    
    classifier_paths = [classifier_path]
    ### three more classifiers with different random seeds for training
    classifier_suffixes = [s.strip() for s in args.classifier_weight_suffixes.split(',')]
    print(f"Classifier weight suffixes to test: {classifier_suffixes}")
    for suffix in classifier_suffixes:
        classifier_paths.append(args.dataset.replace('/', '_') + f"_classifier{suffix}.pt")
    
    for clf_idx, classifier_path in enumerate(classifier_paths):
        print(f"Testing steerability with classifier weights from: {classifier_path}")
        classifier = Roberta_classifier(len(concept_set)).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()

        set_seed(args.seed)

        if args.dataset == "dbpedia_14":
            intervention_value = 150
        else:
            intervention_value = 100
        pred = []
        text = []
        acc = evaluate.load("accuracy")
        with torch.no_grad():
            for i in tqdm(range(100 // len(concept_set))):
                print("example", str(i), end="\r")
                with torch.no_grad():
                    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
                    for j in range(len(concept_set)):
                        v = [0] * len(concept_set)
                        v[j] = intervention_value
                        text_ids, _ = cbl.generate(input_ids, preLM, intervene=v)
                        decoded_text_ids = tokenizer.decode(text_ids[0][~torch.isin(text_ids[0], torch.tensor([128000, 128001]).to(device))])
                        text.append(decoded_text_ids)
                        roberta_text_ids = torch.tensor([roberta_tokenizer.encode(decoded_text_ids)]).to(device)
                        roberta_input = {"input_ids": roberta_text_ids, "attention_mask": torch.tensor([[1]*roberta_text_ids.shape[1]]).to(device)}
                        logits = classifier(roberta_input)
                        pred.append(logits)
            pred = torch.cat(pred, dim=0).detach().cpu()
            pred = np.argmax(pred.numpy(), axis=-1)
            acc.add_batch(predictions=pred, references=list(range(len(concept_set)))*(100 // len(concept_set)))

        print("Steerability test accuracy:")
        acc = acc.compute()
        if clf_idx == 0:
            wandb.log({"steerability_test_accuracy": acc})
        else:
            wandb.log({f"steerability_test_accuracy_{clf_idx}": acc})
        print(acc)
    
    
    
    
    ### TEST CONCEPT PREDICTION AFTER TRAINING
    print("eval concepts...")
    metric = evaluate.load("accuracy")
    concept_predictions = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            concepts, _, _, _ = cbl(features.float())
        concept_predictions.append(eos_pooling(concepts, batch["attention_mask"]))
    concept_predictions = torch.cat(concept_predictions, dim=0).detach().cpu()
    pred = np.argmax(concept_predictions.numpy(), axis=-1)
    metric.add_batch(predictions=pred, references=encoded_test_dataset["label"])
    print("Concept prediction accuracy:")
    acc = metric.compute()
    print(acc)
    wandb.log({"concept_prediction_accuracy": acc})
    
    
    
    #### TEST WEIGHT
    print("Top tokens for each concept neuron:")
    w = cbl.fc.weight.data[:, :len(concept_set)].T
    for i in tqdm(range(len(concept_set))):
        top_values, top_ids = torch.topk(w[i], k=10)
        print("Neuron: ", concept_set[i])
        print("Top 10 tokens with highest weight:")
        for j in range(10):
            print("Neuron:", concept_set[i], "[",round(float(top_values.detach().cpu()[j]), 3), "]", tokenizer.decode(top_ids[j]))

    print("Sparsity of concept weight matrix:")
    print((w > 1e-6).count_nonzero() / w.numel())
    wandb.log({"concept_weight_sparsity": (w > 1e-6).count_nonzero() / w.numel()})
    
    
    
    #### TEST PERPLEXITY AFTER TRAINING
    print("Test perplexity after training:")
    set_seed(args.seed)
    
    pred = []
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    for i in tqdm(range(100)):
        print("example", str(i), end="\r")
        with torch.no_grad():
            text_ids, _ = cbl.generate(input_ids, preLM)
            pred.append(tokenizer.decode(text_ids[0], skip_special_tokens=True ))
            if len(pred[-1].split()) > 30:
                continue
            perplexity.add_batch(predictions=[pred[i]])

        ## print some generated texts
    print("Some generated texts:")
    for i in range(5):
        print(pred[i])
    import pickle
    if "perplexity_text" not in os.listdir("./"):
        try:
            os.mkdir("perplexity_text")
        except:
            pass
    pickle.dump(pred, open(f"perplexity_text/{run_name}_generated_texts_{args.seed}.pkl", "wb"))
    del preLM
    del cbl
    gc.collect()
    torch.cuda.empty_cache()

    print("Perplexity: (under 30 tokens)")
    perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
    print(perplexity)
    wandb.log({"perplexity_under_30_tokens": perplexity})
    
    print("Now for all tokens:")
    perplexity = evaluate.load("perplexity", module_type="metric")
    for p in pred:
        perplexity.add_batch(predictions=[p])
    perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
    print(perplexity)
    wandb.log({"perplexity_all_tokens": perplexity})
        
    