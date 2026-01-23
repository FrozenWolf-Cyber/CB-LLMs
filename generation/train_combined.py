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
parser.add_argument("--elastic_net_alpha", type=float, default=1.0)
parser.add_argument("--residual_dim", type=int, default=768)
parser.add_argument("--orthogonal_loss_weight", type=float, default=0)
parser.add_argument("--residual_penalty_weight", type=float, default=0)
parser.add_argument("--DEBUG", action='store_true', help="If set, use a smaller subset of data for quick debugging.")


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

    wandb.init(project="cbm-generation", name=f"train-{args.dataset}-seed{args.seed}",
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
    
    classifier = torch.nn.Linear(args.residual_dim, len(concept_set)).to(device)
    
    if args.discrimination_loss > 0:
        opt_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)

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
            loss = args.concept_loss * concept_loss + word_loss
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
                    orthogonal_loss = torch.cosine_similarity(concepts, matched_unsup, dim=-1).mean() ## TODO: check shape
                    val_losses["val_orthogonal_loss"].append(orthogonal_loss.detach().cpu().numpy())
                
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
    classifier = Roberta_classifier(len(concept_set)).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    

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
    wandb.log({"steerability_test_accuracy": acc})
    
    
    
    
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
            pred.append(tokenizer.decode(text_ids[0]))
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
        
    