import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
import config_finegrained as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast, AutoModel
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBLResidual, CBL, Roberta_classifier
import time
from module_intervention import amplify_intervention
from utils import elastic_net_penalty, mean_pooling, eos_pooling, get_labels, cos_sim_cubed
import wandb
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epoch_multiplier", type=int, default=1, help="Epoch multiplier to increase total training steps (for debugging).")
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
parser.add_argument("--automatic_concept_correction", action='store_true', help="If set, automatically set concept labels to 0 for concepts that are not present in the example according to the ground truth label. This is a form of training intervention to correct mislabeled concepts.")
parser.add_argument("--concept_loss_type", type=str, default="cosine_cubed", help="Type of concept loss to use: 'cosine_cubed' or 'ce'.")
parser.add_argument("--labeling", type=str, default="mpnet", help="mpnet, angle, simcse, llm")
parser.add_argument("--use_last_epoch", action='store_true', help="If set, load the classifier from the last epoch instead of the best epoch based on validation loss.")


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
    if args.dataset == 'SetFit/sst2':
        val_similarity = np.load(label_prefix + "/concept_labels_val.npy")  # (N_val, num_concepts)

    if args.automatic_concept_correction:
        start = time.time()
        print("training intervention...")
        for i in range(train_similarity.shape[0]):
            for j in range(len(concept_set)):
                if get_labels(j, args.dataset) != encoded_train_dataset["label"][i]:
                    train_similarity[i][j] = 0.0
                else:
                    if train_similarity[i][j] < 0.0:
                        train_similarity[i][j] = 0.0
        
        if args.dataset == 'SetFit/sst2':
            for i in range(val_similarity.shape[0]):
                for j in range(len(concept_set)):
                    if get_labels(j, args.dataset) != encoded_val_dataset["label"][i]:
                        val_similarity[i][j] = 0.0
                    else:
                        if val_similarity[i][j] < 0.0:
                            val_similarity[i][j] = 0.0
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

    # Load MPNET for GRPO reward scoring
    if args.grpo_loss_weight > 0:
        from transformers import AutoTokenizer, AutoModel
        tokenizer_sim_grpo = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        sim_model_grpo = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
        sim_model_grpo.eval()
        for p in sim_model_grpo.parameters():
            p.requires_grad = False
        
        encoded_c_grpo = tokenizer_sim_grpo(concept_set, padding=True, truncation=True, max_length=args.max_length)
        encoded_c_grpo = {k: torch.tensor(v).to(device) for k, v in encoded_c_grpo.items()}
        concept_features_grpo = sim_model_grpo(input_ids=encoded_c_grpo["input_ids"], attention_mask=encoded_c_grpo["attention_mask"])
        concept_features_grpo = mean_pooling(concept_features_grpo.last_hidden_state, encoded_c_grpo["attention_mask"])
        concept_features_grpo = F.normalize(concept_features_grpo, p=2, dim=1)
        print("Loaded MPNET for GRPO similarity scoring.")

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

    # ======= DEBUG: Test batched generate =======
    if args.DEBUG:
        print("=" * 60)
        print("DEBUG: Testing generate_batch (no intervention)")
        print("=" * 60)
        test_input = torch.tensor([tokenizer.encode("")]).to(device)
        num_test = min(args.grpo_num_trajectories, 8)
        preLM.eval()
        cbl.eval()
        with torch.no_grad():
            batch_ids, _ = cbl.generate_batch(
                test_input, preLM, num_samples=num_test,
                intervene=None, length=50
            )
        special_tokens = torch.tensor([128000, 128001]).to(device)
        for g in range(num_test):
            tokens = batch_ids[g][~torch.isin(batch_ids[g], special_tokens)]
            text = tokenizer.decode(tokens)
            print(f"  [{g+1}/{num_test}] ({len(tokens)} tokens): {text}")
        print("=" * 60)
        preLM.train()
        cbl.train()

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
            "grpo_loss": [],
            "grpo_mean_reward": [],
            "non_zero_grpo_advantages": []
        }

        
        for i, (batch, batch_sim) in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_sim = batch_sim.to(device)

            concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            concepts, unsup, vocabs, matched_unsup = cbl(features.float())
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
                        generated_c_grpo = tokenizer_sim_grpo(non_empty_texts, return_tensors='pt', truncation=True, max_length=args.max_length, padding=True).to(device)
                        generated_features_grpo = sim_model_grpo(input_ids=generated_c_grpo["input_ids"], attention_mask=generated_c_grpo["attention_mask"])
                        generated_features_grpo = mean_pooling(generated_features_grpo.last_hidden_state, generated_c_grpo["attention_mask"])
                        generated_features_grpo = F.normalize(generated_features_grpo, p=2, dim=1)
                        
                        sims_grpo = generated_features_grpo @ concept_features_grpo.T
                        v_target_grpo = [0] * len(concept_set)
                        v_target_grpo[grpo_concept_idx] = 1.0
                        v_tensor_grpo = torch.tensor(v_target_grpo).to(device).unsqueeze(0).expand(len(non_empty_texts), -1)

                        for idx, g in enumerate(non_empty_indices):
                            r = cos_sim_cubed(sims_grpo[idx:idx+1], v_tensor_grpo[idx:idx+1].float()).item()
                            grpo_rewards[g] = r

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

                # Phase 3: Recompute log-probs WITH gradient (batched teacher-forced)
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

                    feats_g = preLM(input_ids=batch_input, attention_mask=batch_attn).last_hidden_state
                    _, unsup_g, _, _ = cbl(feats_g.float())

                    intervened_g = torch.zeros(len(valid_indices), batch_input.shape[1], len(concept_set), device=device)
                    intervened_g[:, :, grpo_concept_idx] = intervention_value

                    vocab_logits_g = cbl.intervene(unsup_g, intervened_g)
                    log_probs_g = F.log_softmax(vocab_logits_g, dim=-1)
                    token_log_probs_g = log_probs_g.gather(2, batch_target.unsqueeze(-1)).squeeze(-1)
                    # Mask out padded positions
                    token_log_probs_g = token_log_probs_g * batch_attn.float()
                    mean_log_probs = token_log_probs_g.sum(dim=1) / batch_attn.sum(dim=1).float()

                    grpo_loss_val = (-valid_adv * mean_log_probs).mean()
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
            for i, (batch, batch_sim) in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_sim = batch_sim.to(device)

                concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
                word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
                with torch.no_grad():
                    features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    concepts, unsup, vocabs, matched_unsup = cbl(features.float())
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
    
    from transformers import AutoTokenizer, AutoModel
    tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    sim_model.eval()

    encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=args.max_length)
    encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}
    concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"])
    concept_features = mean_pooling(concept_features.last_hidden_state, encoded_c["attention_mask"])
    concept_features = F.normalize(concept_features, p=2, dim=1)

    if args.dataset == "dbpedia_14":
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
    
    # Use batched generation for steerability evaluation: for each concept
    # generate 50 samples at once with `generate_batch`.
    num_steerability_samples = 50  # scalar
    gen_input = torch.tensor([tokenizer.encode("")]).to(device)  # (1, prompt_len)
    special_tokens_mask = torch.tensor([128000, 128001]).to(device)  # (2,)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")  # CE over classes

    with torch.no_grad():
        for j in tqdm(range(len(concept_set)), desc="Steerability concepts"):
            v = [0] * len(concept_set)  # (C,)
            v[j] = intervention_value

            # Batched generation: `num_steerability_samples` samples for this concept
            text_ids_batch, _ = cbl.generate_batch(
                gen_input,
                preLM,
                num_samples=num_steerability_samples,
                intervene=v,
                length=50,
            )  # (num_steerability_samples, gen_len)

            # Decode and collect texts for MPNet scoring
            decoded_texts = []  # list of length B=num_steerability_samples
            for g in range(num_steerability_samples):
                tokens = text_ids_batch[g][~torch.isin(text_ids_batch[g], special_tokens_mask)]
                decoded = tokenizer.decode(tokens)
                decoded_texts.append(decoded)
                text.append(decoded)
            
            ### print some decoded texts for debugging
            ### print some decoded texts for debugging
            # print(f"Steerability evaluation for concept '{concept_set[j]}':")
            for idx in range(len(decoded_texts)):
                # print(f"  Sample {idx+1}: {decoded_texts[idx]}")
                wandb.log({f"steerability_sample_{concept_set[j]}_{idx+1}": decoded_texts[idx]})
            # Batched similarity scoring with MPNet
            generated_c = tokenizer_sim(
                decoded_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )  # dict: input_ids -> (B, L), attention_mask -> (B, L)
            generated_c = {k: v.to(device) for k, v in generated_c.items()}
            generated_features = sim_model(
                input_ids=generated_c["input_ids"],
                attention_mask=generated_c["attention_mask"],
            )  # last_hidden_state: (B, L, H)
            generated_features = mean_pooling(
                generated_features.last_hidden_state,
                generated_c["attention_mask"],
            )  # (B, H)
            generated_features = F.normalize(generated_features, p=2, dim=1)  # (B, H)

            sims = generated_features @ concept_features.T  # (B, C)
            v_tensor = torch.tensor(v).to(device).unsqueeze(0).expand(sims.size(0), -1)  # (B, C)

            # cos_sim_cubed per sample (no reduction)
            cos_vals = cos_sim_cubed(sims, v_tensor.float(), reduce=False)  # (B,)
            cos_sim_cubed_values.extend(cos_vals.detach().cpu().tolist())

            # Cross-entropy loss per sample w.r.t. true concept j
            targets = torch.full((sims.size(0),), j, dtype=torch.long, device=device)  # (B,)
            ce_vals = ce_loss_fn(sims, targets)  # (B,)
            softmax_values.extend(ce_vals.detach().cpu().tolist())

            # Top-k accuracy counts (vectorized over the batch)
            sorted_indices = torch.argsort(sims, dim=1, descending=True)  # (B, C)
            top1_correct += (sorted_indices[:, 0] == j).sum().item()
            top3_correct += (sorted_indices[:, :3] == j).any(dim=1).sum().item()
            top5_correct += (sorted_indices[:, :5] == j).any(dim=1).sum().item()
            top10_correct += (sorted_indices[:, :10] == j).any(dim=1).sum().item()
            top20_correct += (sorted_indices[:, :20] == j).any(dim=1).sum().item()
            total_evals += sims.size(0)

    wandb.log({
        "steerability_cos_sim_cubed": sum(cos_sim_cubed_values) / len(cos_sim_cubed_values),
        "steerability_softmax": sum(softmax_values) / len(softmax_values),
        "steerability_top1_acc": top1_correct / total_evals,
        "steerability_top3_acc": top3_correct / total_evals,
        "steerability_top5_acc": top5_correct / total_evals,
        "steerability_top10_acc": top10_correct / total_evals,
        "steerability_top20_acc": top20_correct / total_evals,
    })
    
    print(f"Steerability Top-1 Acc: {top1_correct / total_evals}")
    print(f"Steerability Top-3 Acc: {top3_correct / total_evals}")
    print(f"Steerability Top-5 Acc: {top5_correct / total_evals}")
    print(f"Steerability Top-10 Acc: {top10_correct / total_evals}")
    print(f"Steerability Top-20 Acc: {top20_correct / total_evals}")
    
    
    ### TEST CONCEPT PREDICTION AFTER TRAINING (COSINE-SIMILARITY-BASED)
    print("eval concepts (cosine similarity to MPNet labels)...")
    concept_predictions = []  # list of tensors, each (B, num_concepts)

    for batch, _ in tqdm(test_loader, total=len(test_loader)):
        # batch["input_ids"]: (B, seq_len), batch["attention_mask"]: (B, seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state  # (B, seq_len, hidden_dim)
            concepts, _, _, _ = cbl(features.float())  # concepts: (B, seq_len, num_concepts)
        pooled_concepts = eos_pooling(concepts, batch["attention_mask"])  # (B, num_concepts)
        concept_predictions.append(pooled_concepts.detach().cpu())

    # concept_predictions: (N_test, num_concepts)
    concept_predictions = torch.cat(concept_predictions, dim=0)  # (N_test, num_concepts)

    # Load test-time MPNet/ACS concept label vectors, if available
    test_sim_path = label_prefix + "/concept_labels_test.npy"
    if os.path.exists(test_sim_path):
        test_similarity_np = np.load(test_sim_path)  # (N_test, num_concepts)
        test_similarity = torch.tensor(test_similarity_np, dtype=torch.float32)  # (N_test, num_concepts)

        if test_similarity.shape != concept_predictions.shape:
            print("[WARN] Shape mismatch between concept_predictions",
                  f"{tuple(concept_predictions.shape)} and test_similarity {tuple(test_similarity.shape)}.")
            print("       Skipping cosine-similarity-based concept evaluation.")
        else:
            # Average cosine similarity over test set (same objective as training cos_sim_cubed)
            # concept_predictions: (N_test, num_concepts)
            # test_similarity:     (N_test, num_concepts)
            test_cos_sim = cos_sim_cubed(concept_predictions, test_similarity)  # scalar
            test_cos_loss = -test_cos_sim.item()

            print(f"Test concept cosine similarity (cos_sim_cubed): {test_cos_sim.item():.4f}")
            print(f"Test concept cosine loss: {test_cos_loss:.4f}")

            # --- Concept-level top-k accuracy w.r.t. ACS labels ---
            # Ground-truth concept per example: top concept from test_similarity
            # true_concepts: (N_test,)
            true_concepts = torch.argmax(test_similarity, dim=-1)  # indices in [0, num_concepts-1]

            # Predicted ranking over concepts per example
            # pred_sorted: (N_test, num_concepts), each row is concept indices sorted by predicted score
            pred_sorted = torch.argsort(concept_predictions, dim=-1, descending=True)

            topk_list = [1, 3, 5, 10, 20]
            topk_hits = {k: 0 for k in topk_list}
            total = concept_predictions.size(0)

            for i in range(total):
                gt_idx = true_concepts[i].item()
                row = pred_sorted[i]
                for k in topk_list:
                    if k <= row.size(0) and gt_idx in row[:k].tolist():
                        topk_hits[k] += 1

            topk_acc = {f"test_concept_top{k}_acc": topk_hits[k] / total for k in topk_list}

            for k in topk_list:
                print(f"Test concept Top-{k} Acc (w.r.t. ACS top concept): {topk_acc[f'test_concept_top{k}_acc']:.4f}")

            wandb.log({
                "test_concept_cosine_similarity": float(test_cos_sim.item()),
                "test_concept_cosine_loss": float(test_cos_loss),
                **topk_acc,
            })
    else:
        print(f"[WARN] {test_sim_path} not found. Skipping cosine-similarity-based concept evaluation.")
    
    
    
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
    c = 0
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    for i in tqdm(range(100)):
        print("example", str(i), end="\r")
        with torch.no_grad():
            text_ids, _ = cbl.generate(input_ids, preLM)
            pred.append(tokenizer.decode(text_ids[0], skip_special_tokens=True ))
            if len(pred[-1].split()) > 30:
                continue
            c += 1
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
    if c > 0:
        perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
        print(perplexity)
        wandb.log({"perplexity_under_30_tokens": perplexity})
    else:
        print("No generated texts under 30 tokens to compute perplexity.")
        wandb.log({"perplexity_under_30_tokens": None})
    
    print("Now for all tokens:")
    perplexity = evaluate.load("perplexity", module_type="metric")
    for p in pred:
        perplexity.add_batch(predictions=[p])
    perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
    print(perplexity)
    wandb.log({"perplexity_all_tokens": perplexity})
        
    