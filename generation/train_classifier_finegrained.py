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

parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--labeling", type=str, default="mpnet")

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encode_roberta, s):
        self.encode_roberta = encode_roberta
        self.s = s

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encode_roberta.items()}
        y = torch.FloatTensor(self.s[idx])
        return t, y

    def __len__(self):
        return len(self.encode_roberta['input_ids'])

def build_loaders(encode_roberta, s, mode):
    dataset = ClassificationDataset(encode_roberta, s)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

def get_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    actuals = torch.argmax(targets, dim=1)
    return (preds == actuals).float().mean()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    
    wandb.init(project="CB-LLM-steer", name=f"{args.dataset}_finegrained", config=vars(args))

    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    val_dataset = load_dataset(args.dataset, split='validation' if args.dataset == 'SetFit/sst2' else 'test')

    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "").replace("#39;", "'")
            return example
        train_dataset = train_dataset.map(replace_bad_string)
        val_dataset = val_dataset.map(replace_bad_string)

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    def tokenize_fn(e):
        return tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length)

    encoded_train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=len(train_dataset))
    encoded_val_dataset = val_dataset.map(tokenize_fn, batched=True, batch_size=len(val_dataset))

    for ds in [encoded_train_dataset, encoded_val_dataset]:
        ds.remove_columns([CFG.example_name[args.dataset]])
        if args.dataset == 'SetFit/sst2': ds.remove_columns(['label_text'])
        if args.dataset == 'dbpedia_14': ds.remove_columns(['title'])

    d_name = args.dataset.replace('/', '_')
    prefix = f"./{args.labeling}_acs/{d_name}/"
    train_similarity = np.load(os.path.join(prefix, "concept_labels_train.npy"))
    val_similarity = np.load(os.path.join(prefix, "concept_labels_val.npy"))

    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")
    val_loader = build_loaders(encoded_val_dataset, val_similarity, mode="val")

    concept_set = CFG.concept_set[args.dataset]
    classifier = Roberta_classifier(len(concept_set)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-6)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = CFG.epoch[args.dataset]
    best_val_acc = -1
    for e in range(epochs):
        classifier.train()
        training_loss, training_acc = [], []
        for i, (batch, targets) in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = targets.to(device)
            logits = classifier(batch)
            loss = criterion(logits, F.softmax(targets, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = get_accuracy(logits, targets)
            training_loss.append(loss.item())
            training_acc.append(acc.item())
            print(f"batch {i} loss: {loss.item():.4f} acc: {acc.item():.4f}", end="\r")

        classifier.eval()
        val_loss, val_acc = [], []
        with torch.no_grad():
            for batch, targets in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                targets = targets.to(device)
                logits = classifier(batch)
                loss = criterion(logits, F.softmax(targets, dim=1))
                acc = get_accuracy(logits, targets)
                val_loss.append(loss.item())
                val_acc.append(acc.item())

        metrics = {
            "train_loss": np.mean(training_loss),
            "train_acc": np.mean(training_acc),
            "val_loss": np.mean(val_loss),
            "val_acc": np.mean(val_acc),
            "epoch": e + 1
        }
        print(f"\nEpoch {e+1}: Train Loss {metrics['train_loss']:.4f} Val Acc {metrics['val_acc']:.4f}")
        wandb.log(metrics)
        if metrics['val_acc'] > best_val_acc:
            best_val_acc = metrics['val_acc']
            torch.save(classifier.state_dict(), f"{d_name}_finegrained_classifier.pt")

    wandb.finish()