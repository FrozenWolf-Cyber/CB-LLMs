import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel
from datasets import load_dataset
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
    
    wandb.init(project="CB-LLMs-steer", name=f"{args.dataset}_finegrained", config=vars(args))

    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')

    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "").replace("#39;", "'")
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

    print("Len of dataset: ", len(encoded_train_dataset))
    d_name = args.dataset.replace('/', '_')
    prefix = f"./{args.labeling}_acs/{d_name}/"
    train_similarity = np.load(os.path.join(prefix, "concept_labels_train.npy"))

    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")
    print("concept len: ", len(CFG.concept_set[args.dataset]))
    print("Test batch: ", next(iter(train_loader)))
    concept_set = CFG.concept_set[args.dataset]
    print("concept are: ", concept_set)
    classifier = Roberta_classifier(len(concept_set)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-6)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = CFG.epoch[args.dataset]
    for e in range(min(epochs, 10)):  # Limit to 10 epochs for debugging
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

        metrics = {
            "train_loss": np.mean(training_loss),
            "train_acc": np.mean(training_acc),
            "epoch": e + 1
        }
        print(f"\nEpoch {e+1}: Train Loss {metrics['train_loss']:.4f} Train Acc {metrics['train_acc']:.4f}")
        wandb.log(metrics)

    torch.save(classifier.state_dict(), f"{d_name}_finegrained_classifier.pt")
    wandb.finish()
