import argparse
import os
import torch
import numpy as np
import config as CFG
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    RobertaTokenizerFast,
)
from datasets import load_dataset, concatenate_datasets
from modules import CBL, Roberta_classifier
from utils import eos_pooling
import time
import gc

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--max_length", type=int, default=1024)


def compute_perplexity_batched(
    texts,
    model,
    tokenizer,
    batch_size=4,
    max_length=100,
):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(
                input_ids=enc.input_ids,
                labels=enc.input_ids,
            )

            num_tokens = enc.input_ids.numel()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    return float(np.exp(total_loss / total_tokens))



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len:", len(concept_set))

    print("preparing backbone")
    peft_path = (
        "from_pretained_llama3_lora_cbm/"
        + args.dataset.replace("/", "_")
        + "/llama3"
    )
    cbl_path = (
        "from_pretained_llama3_lora_cbm/"
        + args.dataset.replace("/", "_")
        + "/cbl.pt"
    )


    preLM = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.bfloat16,
    ).to(device)

    preLM.load_adapter(peft_path)
    preLM.eval()

    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    cbl.eval()

    pred = []
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)

    for i in range(100):
        print("example", i, end="\r")
        with torch.no_grad():
            text_ids, _ = cbl.generate(input_ids, preLM)
            pred.append(tokenizer.decode(text_ids[0]))

    ppl = compute_perplexity_batched(
        pred,
        preLM,
        tokenizer,
        batch_size=2,
        max_length=100,
    )
    print("\nMean Perplexity:", ppl)


    del preLM
    del cbl
    gc.collect()
    torch.cuda.empty_cache()
