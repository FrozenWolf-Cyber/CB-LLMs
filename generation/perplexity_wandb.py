import torch
import evaluate
import argparse
import os


## maintain a json file to make note which runs have been done already
import json
file = "perplexity_text/perplexity_wandb_log.json"

runs_available = []
if "perplexity_wandb_log.json" in os.listdir("perplexity_text/"):
    with open(file, "r") as f:
        runs_done = json.load(f)
else:
    runs_done = {}


for i in os.listdir("perplexity_text/"):
    if "generated_texts_" in i and i.endswith(".pkl"):
        run_name = i.split("_")[0]
        print("Checking run:", run_name)
        if run_name in runs_done.keys():
            print("Run already done, skipping.")
            continue
        
        runs_available.append(i)


    

print("Runs available:", runs_available)

for run_file in runs_available:
    ## run python perplexity_calc.py --path perplexity_text/generated_texts_XXXX.pkl
    print("Processing file:", run_file)
    print("Command:", f"python perplexity_calc.py --path perplexity_text/{run_file}")
    result = os.system(f"python perplexity_calc.py --path perplexity_text/{run_file}")
    print(f"Command output status: {result}")