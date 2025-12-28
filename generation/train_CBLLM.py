import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, concatenate_datasets
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBL
import time
from utils import elastic_net_penalty, mean_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)

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

    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
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

    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="train")
    if args.dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")


    print("preparing backbone")
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=5e-5)
    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=5e-5)
    print("preparing classifier")
    classifier = torch.nn.Linear(768, len(concept_set)).to(device)
    opt_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    print("start training...")
    best_loss = float('inf')
    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    prefix += "./from_pretained_llama3_lora_cbm"
    prefix += "/"
    prefix += d_name
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "llama3"
    cbl_name = "cbl"

    start = time.time()
    epochs = CFG.epoch[args.dataset]
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        cbl.train()
        classifier.train()
        training_concept_loss = []
        training_word_loss = []
        training_neg_entropy_loss = []
        training_reg_loss = []
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            concepts, unsup, vocabs = cbl(features.float())
            concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label.reshape(-1))
            word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
            loss = concept_loss + word_loss
            reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
            loss += 1.0 * reg
            opt_prelm.zero_grad()
            opt_cbl.zero_grad()
            loss.backward()
            opt_prelm.step()
            opt_cbl.step()

            classification = classifier(mean_pooling(unsup.detach(), batch["attention_mask"]))
            discrimination_loss = torch.nn.CrossEntropyLoss()(classification, batch["label"])
            opt_classifier.zero_grad()
            discrimination_loss.backward(inputs=list(classifier.parameters()))
            opt_classifier.step()

            _, unsup, _ = cbl(features.detach().float())
            classification = classifier(mean_pooling(unsup, batch["attention_mask"]))
            p = F.softmax(classification, dim=-1)
            neg_entropy_loss = torch.sum(p * torch.log(p), dim=-1).mean()
            opt_cbl.zero_grad()
            neg_entropy_loss.backward(inputs=list(cbl.unsup.parameters()))
            opt_cbl.step()

            print("batch", str(i), "concept loss:", concept_loss.detach().cpu().numpy(), "word loss:", word_loss.detach().cpu().numpy(), "neg e loss:", neg_entropy_loss.detach().cpu().numpy(), "reg loss:", reg.detach().cpu().numpy(), end="\r")
            training_concept_loss.append(concept_loss.detach().cpu().numpy())
            training_word_loss.append(word_loss.detach().cpu().numpy())
            training_neg_entropy_loss.append(neg_entropy_loss.detach().cpu().numpy())
            training_reg_loss.append(reg.detach().cpu().numpy())
        avg_training_concept_loss = sum(training_concept_loss)/len(training_concept_loss)
        avg_training_word_loss = sum(training_word_loss) / len(training_word_loss)
        avg_training_neg_entropy_loss = sum(training_neg_entropy_loss) / len(training_neg_entropy_loss)
        avg_training_reg_loss = sum(training_reg_loss)/len(training_reg_loss)
        print("training concept loss:", avg_training_concept_loss, "training word loss:", avg_training_word_loss, "training neg e loss:", avg_training_neg_entropy_loss, "training reg loss: ", avg_training_reg_loss)


        if args.dataset == 'SetFit/sst2':
            preLM.eval()
            cbl.eval()
            val_concept_loss = []
            val_word_loss = []
            val_neg_entropy_loss = []
            val_reg_loss = []
            for i, batch in enumerate(val_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
                word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
                with torch.no_grad():
                    features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    concepts, unsup, vocabs = cbl(features.float())
                    classification = classifier(mean_pooling(unsup, batch["attention_mask"]))
                concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label.reshape(-1))
                word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label.reshape(-1))
                discrimination_loss = torch.nn.CrossEntropyLoss()(classification, batch["label"])
                p = F.softmax(classification, dim=-1)
                neg_entropy_loss = torch.sum(p * torch.log(p), dim=-1).mean()
                reg = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
                val_concept_loss.append(concept_loss.detach().cpu().numpy())
                val_word_loss.append(word_loss.detach().cpu().numpy())
                val_neg_entropy_loss.append(neg_entropy_loss.detach().cpu().numpy())
                val_reg_loss.append(reg.detach().cpu().numpy())
            avg_val_concept_loss = sum(val_concept_loss) / len(val_concept_loss)
            avg_val_word_loss = sum(val_word_loss) / len(val_word_loss)
            avg_val_neg_entropy_loss = sum(val_neg_entropy_loss) / len(val_neg_entropy_loss)
            avg_val_reg_loss = sum(val_reg_loss) / len(val_reg_loss)
            print("val concept loss:", avg_val_concept_loss, "val word loss:", avg_val_word_loss, "val neg e loss:", avg_val_neg_entropy_loss, "val reg loss: ", avg_val_reg_loss)

            avg_val_loss = avg_val_concept_loss + avg_val_word_loss
            if avg_val_loss < best_loss:
                print("save model")
                best_loss = avg_val_loss
                preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
                torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")
            else:
                preLM.save_pretrained(prefix + model_name + "_low_score_epoch_" + str(e + 1))
                torch.save(cbl.state_dict(), prefix + cbl_name + "_low_score_epoch_" + str(e + 1) + ".pt")
        else:
            print("save model")
            preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
            torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")

    end = time.time()
    print("time of training CBM:", (end - start) / 3600, "hours")
