import torch
import torch.nn.functional as F
import math

import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
import gc


import argparse
import os
import torch
import numpy as np
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast
from datasets import load_dataset, concatenate_datasets
from modules import CBL, Roberta_classifier
from utils import eos_pooling
import evaluate
import time
import gc

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--max_length", type=int, default=1024)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')


    import pickle
    pred_2 = pickle.load(open("generated_texts_3.pkl", "rb"))
    temp = []
    for p in pred_2:
        if len(p.split()) < 25:
            temp.append(p)
            
    print("Filtered length:", len(temp))
    pred = temp
            


    perplexity = evaluate.load("perplexity", module_type="metric")
    
    for i in range(len(pred)):
        perplexity.add_batch(predictions=[pred[i]])
        
    with torch.no_grad():
        print(perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity'])

    