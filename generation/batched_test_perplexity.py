import torch
import torch.nn.functional as F
import math

def compute_perplexity_single(model, tokenizer, text, device, max_length=1024):
    """
    Compute perplexity for a single text sequence.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.last_hidden_state @ model.embed_tokens.weight.T

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    loss = loss * shift_mask.view(-1)
    total_loss = loss.sum()
    total_tokens = shift_mask.sum()

    ppl = torch.exp(total_loss / total_tokens)
    return ppl.item()



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

    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("preparing backbone")
    peft_path = "from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/llama3"
    cbl_path = "from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/cbl.pt"
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM.load_adapter(peft_path)
    preLM.eval()
    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    cbl.eval()

    pred = []
    perplexities = []
    
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    
    for i in range(100):
        print("example", i, end="\r")
    
        with torch.no_grad():
            text_ids, _ = cbl.generate(input_ids, preLM)
            text = tokenizer.decode(text_ids[0], skip_special_tokens=True)
            pred.append(text)
    
        ppl = compute_perplexity_single(
            model=preLM,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_length=100,
        )
        perplexities.append(ppl)

    del preLM
    del cbl
    gc.collect()
    torch.cuda.empty_cache()

    mean_ppl = sum(perplexities) / len(perplexities)
    print("Mean perplexity:", mean_ppl)
