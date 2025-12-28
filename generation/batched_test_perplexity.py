import torch
import torch.nn.functional as F
import math

import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer
import gc

def compute_perplexity_streaming(
    texts,
    model,
    tokenizer,
    max_length=100,
):
    """
    Streaming perplexity (batch size = 1), evaluate-compatible.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(model.device)

            outputs = model(
                input_ids=enc.input_ids,
                labels=enc.input_ids,
            )

            # HF loss is mean over tokens
            loss = outputs.loss.item()
            num_tokens = enc.input_ids.numel()

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))



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
    # tokenizer.pad_token = tokenizer.eos_token

    # roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


    # concept_set = CFG.concepts_from_labels[args.dataset]
    # print("concept len: ", len(concept_set))

    # print("preparing backbone")
    # peft_path = "from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/llama3"
    # cbl_path = "from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/cbl.pt"
    # preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    # preLM.load_adapter(peft_path)
    # preLM.eval()
    # cbl = CBL(config, len(concept_set), tokenizer).to(device)
    # cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    # cbl.eval()

    # pred = []
    # input_ids = torch.tensor([tokenizer.encode("")]).to(device)

    # for i in range(100):
    #     print("example", i, end="\r")

    #     with torch.no_grad():
    #         text_ids, _ = cbl.generate(input_ids, preLM)
    #         text = tokenizer.decode(
    #             text_ids[0],
    #             skip_special_tokens=True
    #         )
    #         print(text)
    #         pred.append(text)



    # import pickle
    # pickle.dump(pred, open("generated_texts.pkl", "wb"))
    # del preLM
    # del cbl
    # gc.collect()
    # torch.cuda.empty_cache()

    import pickle
    pred = pickle.load(open("generated_texts_2.pkl", "rb"))
    
    from transformers import LlamaForCausalLM


    # preLM_lm = LlamaForCausalLM.from_pretrained(
    #     "meta-llama/Meta-Llama-3-8B",
    #     torch_dtype=torch.bfloat16
    # ).to(device)

    # preLM_lm.eval()


    # mean_ppl = compute_perplexity_streaming(
    #     pred,
    #     model=preLM_lm,  
    #     tokenizer=tokenizer,
    #     max_length=100,
    # )
    
    # print("Mean perplexity:", mean_ppl)

    perplexity = evaluate.load("perplexity", module_type="metric")
    
    for i in range(100):
        perplexity.add_batch(predictions=[pred[i]])
        
    with torch.no_grad():
        print(perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity'])

    