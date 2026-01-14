import argparse
import gc
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
parser.add_argument("--peft_path", type=str, default="from_pretained_llama3_lora_cbm/SetFit_sst2/llama3")
parser.add_argument("--cbl_path", type=str, default="from_pretained_llama3_lora_cbm/SetFit_sst2/cbl.pt")
parser.add_argument("--num_runs", type=int, default=1000)
parser.add_argument("--output_path", type=str, default="perplexity_text/generated_texts_XXXX.pkl")


args = parser.parse_args()
set_seed(args.seed)

lora_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                              "down_proj"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)
config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
tokenizer.pad_token = tokenizer.eos_token

concept_set = CFG.concepts_from_labels[args.dataset]
preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
preLM.load_adapter(args.peft_path)
preLM.eval()
if args.discrimination_loss > 0:
    cbl = CBL(config, len(concept_set), tokenizer).to(device)
else:
    cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    
cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
cbl.eval()


pred = []
perplexity = evaluate.load("perplexity", module_type="metric")
input_ids = torch.tensor([tokenizer.encode("")]).to(device)
for i in range(args.num_runs):
    print("example", str(i), end="\r")
    with torch.no_grad():
        text_ids, _ = cbl.generate(input_ids, preLM)
        pred.append(tokenizer.decode(text_ids[0]))

import pickle
pickle.dump(pred, open(args.output_path, "wb"))
del preLM
del cbl
gc.collect()
torch.cuda.empty_cache()

    
