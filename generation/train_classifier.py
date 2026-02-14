import argparse
import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset, concatenate_datasets
import config as CFG
from modules import Roberta_classifier
import wandb
import copy

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=0)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encode_roberta):
        self.encode_roberta = encode_roberta

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encode_roberta.items()}

        return t

    def __len__(self):
        return len(self.encode_roberta['input_ids'])

def build_loaders(encode_roberta, mode):
    dataset = ClassificationDataset(encode_roberta)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

def get_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    wandb.init(project="CB-LLMs-steer", name=f"{args.dataset}_class", config=vars(args))

    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    print("tokenizing...")

    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "")
            example["text"] = example["text"].replace("#39;", "'")
            return example
        train_dataset = train_dataset.map(replace_bad_string)

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    encoded_train_dataset = train_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True,
        batch_size=len(train_dataset))
    encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['title'])
    encoded_train_dataset = encoded_train_dataset[:len(encoded_train_dataset)]

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="train")

    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("preparing classifier...")
    classifier = Roberta_classifier(len(concept_set)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-6)


    print("start training...")
    best_loss = float('inf')

    epochs = CFG.epoch[args.dataset]

    for e in range(epochs):
        print("Epoch ", e+1, ":")
        classifier.train()
        training_loss, training_acc = [], []
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = classifier(batch)
            loss = torch.nn.CrossEntropyLoss()(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = get_accuracy(logits, batch["label"])
            training_loss.append(loss.item()); training_acc.append(acc.item())
            print(f"batch {i} loss: {loss.item():.4f} acc: {acc.item():.4f}", end="\r")

        metrics = {"train_loss": np.mean(training_loss), "train_acc": np.mean(training_acc), "epoch": e+1}
        print(f"\ntrain loss: {metrics['train_loss']:.4f} | train acc: {metrics['train_acc']:.4f}")
        wandb.log(metrics)
        torch.save(classifier.state_dict(), args.dataset.replace('/', '_') + "_classifier.pt")
    
    wandb.finish()
