import argparse
import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset, concatenate_datasets
import config as CFG
# from modules import CBL, RobertaCBL, GPT2CBL, RobertaCBLResidual
from module_combined import RobertaCBL, RobertaCBLResidual
from utils import cos_sim_cubed, get_labels, eos_pooling
import time
import evaluate
import wandb
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--backbone", type=str, default="roberta", help="roberta or gpt2")
parser.add_argument('--tune_cbl_only', action=argparse.BooleanOptionalAction)
parser.add_argument('--automatic_concept_correction', action=argparse.BooleanOptionalAction)
parser.add_argument("--labeling", type=str, default="mpnet", help="mpnet, angle, simcse, llm")
parser.add_argument("--cbl_only_batch_size", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--residual_ratio", type=float, default=0)
parser.add_argument("--orthogonal_loss_weight", type=float, default=0)

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
    if args.tune_cbl_only:
        batch_size = args.cbl_only_batch_size
    else:
        batch_size = args.batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

def init_metrics():
    metrics = evaluate.combine([
        evaluate.load("accuracy"),
        evaluate.load("f1"),
        evaluate.load("precision"),
        evaluate.load("recall"),
        evaluate.load("confusion_matrix"),
    ])
    return metrics

def metric_eval(metrics, prefix="train"):
    cm = metrics.pop('confusion_matrix')
    metrics['TN'] = cm[0][0]
    metrics['FP'] = cm[0][1]
    metrics['FN'] = cm[1][0]
    metrics['TP'] = cm[1][1]
    metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics

@torch.no_grad()
def concept_activation_error_rate(feature, label):
    feature = F.relu(feature)
    error_rate = []
    for i in range(feature.T.size(0)):
        error = 0
        total = 0
        value, s = feature.T[i].topk(5)
        for j in range(5):
            if value[j] > 1.0:
                total += 1
                if get_labels(i, args.dataset) != label[s[j]]:
                    error += 1
        if total != 0:
            error_rate.append(error/total)
    return sum(error_rate) / len(error_rate)
      

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    wandb.init(project="CB-LLMs", 
               name=f"train_CBL_{args.dataset}_{args.backbone}_{args.labeling}_tunecblonly_{args.tune_cbl_only}_acc_{args.automatic_concept_correction}",
                config=vars(args),
                )
    run_name = wandb.run.name
    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    test_dataset = load_dataset(args.dataset, split='test')
    if args.dataset == 'SetFit/sst2':
        val_dataset = load_dataset(args.dataset, split='validation')
    print("training data len: ", len(train_dataset))
    if args.dataset == 'SetFit/sst2':
        print("val data len: ", len(val_dataset))
    print("tokenizing...")

    if args.labeling == 'llm':
        d_list = []
        for i in range(CFG.class_num[args.dataset]):
            d_list.append(
                train_dataset.filter(lambda e: e['label'] == i).select(range(1000 // CFG.class_num[args.dataset])))
        train_dataset = concatenate_datasets(d_list)
        if args.dataset == 'SetFit/sst2':
            d_list = []
            for i in range(CFG.class_num[args.dataset]):
                d_list.append(
                    val_dataset.filter(lambda e: e['label'] == i).select(range(80 // CFG.class_num[args.dataset])))
            val_dataset = concatenate_datasets(d_list)

        print("training labeled data len: ", len(train_dataset))
        if args.dataset == 'SetFit/sst2':
            print("val labeled data len: ", len(val_dataset))

    if args.backbone == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif args.backbone == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")

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

    encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]
    
    concept_set = CFG.concept_set[args.dataset]
    print("concept len: ", len(concept_set))

    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    if args.labeling == 'mpnet':
        prefix += "mpnet_acs"
    elif args.labeling == 'simcse':
        prefix += "simcse_acs"
    elif args.labeling == 'angle':
        prefix += "angle_acs"
    elif args.labeling == 'llm':
        prefix += "llm_labeling"

    prefix += "/"
    prefix += d_name
    prefix += "/"
    train_similarity = np.load(prefix + "/concept_labels_train.npy")
    if args.dataset == 'SetFit/sst2':
        val_similarity = np.load(prefix + "/concept_labels_val.npy")


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
        print("time of trainng intervention:", (end - start) / 3600, "hours")

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")
    if args.dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, val_similarity, mode="valid")
     ## dummy placeholder float numbers for test
    test_similarity = np.zeros((len(encoded_test_dataset["label"]),1))
    
    test_loader = build_loaders(encoded_test_dataset, test_similarity, mode="test")
    next(iter(test_loader))
    if args.backbone == 'roberta':
        if args.tune_cbl_only:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(cbl.parameters(), lr=1e-4)
        else:
            print("preparing backbone(roberta)+CBL...")
            # backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            if args.residual_ratio==0:
                backbone_cbl = RobertaCBL(len(concept_set), args.dropout, CFG.class_num[args.dataset]).to(device)
            else:
                residual_size = max(int(args.residual_ratio*len(concept_set)), 1)
                print("Residual Size", residual_size)
                backbone_cbl = RobertaCBLResidual(len(concept_set), residual_size, args.dropout, CFG.class_num[args.dataset]).to(device)
            
            optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=5e-6)
    elif args.backbone == 'gpt2':
        if args.tune_cbl_only:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(cbl.parameters(), lr=1e-4)
        else:
            print("preparing backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
            optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=5e-6)
    else:
        raise Exception("backbone should be roberta or gpt2")

    print("start training...")
    best_loss = float('inf')

    if args.backbone == 'roberta':
        prefix += 'roberta_cbm'
    elif args.backbone == 'gpt2':
        prefix += 'gpt2_cbm'
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "cbl"
    if args.tune_cbl_only:
        model_name += "_no_backbone"
    if args.automatic_concept_correction:
        model_name += "_acc"

    start = time.time()
    if args.labeling == 'llm':
        epochs = 10
    else:
        epochs = CFG.cbl_epochs[args.dataset]
        
    CE_criterion = torch.nn.CrossEntropyLoss()
    
    for e in range(epochs):
        metrics = init_metrics()
        print("Epoch ", e+1, ":")
        if args.tune_cbl_only:
            cbl.train()
        else:
            backbone_cbl.train()
        training_loss = {"training_loss_total": 0, 
                         "training_loss_clf": 0, 
                         "training_loss_concept_similarity": 0,
                         "training_loss_orthogonal": 0}
        cbl_feature = []
        cbl_labels = []
        for i, batch in enumerate(train_loader):
            batch_text, batch_sim = batch[0], batch[1]
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            batch_sim = batch_sim.to(device)

            if args.tune_cbl_only:
                with torch.no_grad():
                    LM_features = preLM(input_ids=batch_text["input_ids"], attention_mask=batch_text["attention_mask"]).last_hidden_state
                    if args.backbone == 'roberta':
                        LM_features = LM_features[:, 0, :]
                    elif args.backbone == 'gpt2':
                        LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                    else:
                        raise Exception("backbone should be roberta or gpt2")
                cbl_features = cbl(LM_features)
            else:
                if args.residual_ratio == 0:
                    cbl_features, pred = backbone_cbl(batch_text)
                else:
                    cbl_features, feature_residual, pred = backbone_cbl(batch_text)
            
            loss = -cos_sim_cubed(cbl_features, batch_sim)
            clf_loss = CE_criterion(pred, batch_text["label"])
            orthogonal_loss = 0
            if args.orthogonal_loss_weight>0: ## cosin similarity between concept features and residual features
                orthogonal_loss = F.cosine_similarity(cbl_features, feature_residual, dim=-1).mean()
                
            metrics.add_batch(predictions=torch.argmax(pred, dim=-1).cpu(), references=batch_text["label"].cpu())
            total_loss = loss + clf_loss + args.orthogonal_loss_weight*orthogonal_loss
            

            
            cbl_feature.append(cbl_features.detach().cpu())
            cbl_labels.append(batch_text["label"].detach().cpu())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            wandb.log({"clf_loss": clf_loss.detach().cpu().numpy(),
                       "concept_similarity_loss": loss.detach().cpu().numpy(),
                       "batch_training_loss": total_loss.detach().cpu().numpy(),
                       "orthogonal_loss": orthogonal_loss.detach().cpu().numpy() if args.orthogonal_loss_weight>0 else 0,
                       "epoch": e+1, "batch": i})
            
            print(f"batch {i}/{len(train_loader)} total loss: ", total_loss.detach().cpu().numpy(),
                  f" clf loss: {clf_loss.detach().cpu().numpy()}",
                  f" concept similarity loss: {loss.detach().cpu().numpy()}", end="\r")
            
            training_loss["training_loss_total"] += total_loss.detach().cpu().numpy()
            training_loss["training_loss_clf"] += clf_loss.detach().cpu().numpy()
            training_loss["training_loss_concept_similarity"] += loss.detach().cpu().numpy()
            training_loss["training_loss_orthogonal"] += orthogonal_loss.detach().cpu().numpy()
            
        metrics_result = metric_eval(metrics.compute(), prefix="train")
        for k in training_loss.keys():
            training_loss[k] /= len(train_loader)
            metrics_result[k] = training_loss[k]
        
        metrics_result["train_concept_activation_error_rate"] =  concept_activation_error_rate(torch.cat(cbl_feature, dim=0), torch.cat(cbl_labels, dim=0))
        
        wandb.log(metrics_result)
        print("metrics: ", metrics_result)


        if args.dataset == 'SetFit/sst2':
            if args.tune_cbl_only:
                cbl.eval()
            else:
                backbone_cbl.eval()
            val_loss = {"val_loss_total_loss": 0,
                        "val_loss_clf_loss": 0,
                        "val_loss_concept_similarity_loss": 0,
                        "val_loss_orthogonal_loss": 0}
            
            val_metrics = init_metrics()
            cbl_feature = []
            cbl_labels = []
            for batch in val_loader:
                batch_text, batch_sim = batch[0], batch[1]
                batch_text = {k: v.to(device) for k, v in batch_text.items()}
                batch_sim = batch_sim.to(device)
                with torch.no_grad():
                    if args.tune_cbl_only:
                        LM_features = preLM(input_ids=batch_text["input_ids"], attention_mask=batch_text["attention_mask"]).last_hidden_state
                        if args.backbone == 'roberta':
                            LM_features = LM_features[:, 0, :]
                        elif args.backbone == 'gpt2':
                            LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                        else:
                            raise Exception("backbone should be roberta or gpt2")
                        cbl_features = cbl(LM_features)
                    else:
                        if args.residual_ratio == 0:
                            cbl_features, pred = backbone_cbl(batch_text)
                        else:
                            cbl_features, feature_residual, pred = backbone_cbl(batch_text)
                    loss = -cos_sim_cubed(cbl_features, batch_sim)
                    clf_loss = CE_criterion(pred, batch_text["label"])
                    orthogonal_loss = 0
                    if args.orthogonal_loss_weight>0: ## cosin similarity between concept features and residual features
                        orthogonal_loss = F.cosine_similarity(cbl_features, feature_residual, dim=-1).mean()
                    
                    total_loss = loss + clf_loss + orthogonal_loss
                    val_loss["val_total_loss"] += total_loss.detach().cpu().numpy()
                    val_loss["val_clf_loss"] += clf_loss.detach().cpu().numpy()
                    val_loss["val_concept_similarity_loss"] += loss.detach().cpu().numpy()
                    val_loss["val_orthogonal_loss"] += orthogonal_loss.detach().cpu().numpy()
                    
                    cbl_feature.append(cbl_features.detach().cpu())
                    cbl_labels.append(batch_text["label"].detach().cpu())
                    val_metrics.add_batch(predictions=torch.argmax(pred, dim=-1).cpu(), references=batch_text["label"].cpu())
             
            val_metrics_result = metric_eval(val_metrics.compute(), prefix="val")
                   
            for k in val_loss.keys():
                val_loss[k] /= len(val_loader)
                val_metrics_result[k] = val_loss[k]
                 
            val_concept_act_error_rate = concept_activation_error_rate(torch.cat(cbl_feature, dim=0), torch.cat(cbl_labels, dim=0))
            wandb.log(val_metrics_result)
            
            avg_val_loss = val_loss["val_total_loss"]
            
            if avg_val_loss < best_loss:
                print("save model")
                best_loss = avg_val_loss
                if args.tune_cbl_only:
                    torch.save(cbl.state_dict(), prefix + model_name + f"-{run_name}.pt")
                else:
                    torch.save(backbone_cbl.state_dict(), prefix + model_name + f"-{run_name}.pt")
        else:
            print("save model")
            if args.tune_cbl_only:
                torch.save(cbl.state_dict(), prefix + model_name + f"-{run_name}.pt")
            else:
                torch.save(backbone_cbl.state_dict(), prefix + model_name + f"-{run_name}.pt")

    end = time.time()
    print("time of training CBL:", (end - start) / 3600, "hours")
    metric = init_metrics()
    FL_test_features = []
    for i, batch in enumerate(test_loader):
        batch_text, _ = batch[0], batch[1]
        batch_text = {k: v.to(device) for k, v in batch_text.items()}
        with torch.no_grad():
            if args.tune_cbl_only:
                LM_features = preLM(input_ids=batch_text["input_ids"], attention_mask=batch_text["attention_mask"]).last_hidden_state
                if args.backbone == 'roberta':
                    LM_features = LM_features[:, 0, :]
                elif args.backbone == 'gpt2':
                    LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                else:
                    raise Exception("backbone should be roberta or gpt2")
                cbl_features = cbl(LM_features)
            else:
                if args.residual_ratio == 0:
                    cbl_features, pred = backbone_cbl(batch_text)
                else:
                    cbl_features, feature_residual, pred = backbone_cbl(batch_text)
                    
        FL_test_features.append(cbl_features)
                    
        metric.add_batch(predictions=torch.argmax(pred, dim=-1).cpu(), references=batch_text["label"].cpu())
    
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()
    label = encoded_test_dataset["label"]
    
    error_rate = concept_activation_error_rate(test_c, label)
    
    
    m = metric_eval(metric.compute(), prefix="test")
    m["test_concept_activation_error_rate"] = error_rate
    print("Test results: ", m)
    wandb.log(m)
    
    wandb.finish()