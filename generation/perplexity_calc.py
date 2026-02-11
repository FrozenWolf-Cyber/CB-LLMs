import torch
import evaluate
import wandb
import argparse
import os

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--path", type=str)
parser.add_argument("--prefix", type=str, default="")

    

if __name__ == "__main__":
    args = parser.parse_args()
    run_name = args.path.split("/")[-1].split("_")[0]
    print("Run name:", run_name)
    with wandb.init(entity="frozenwolf", project="cbm-generation-new", id=run_name, resume="must") as run:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    
    
        import pickle
        pred_2 = pickle.load(open(args.path, "rb"))
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
            perp = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
            
            print("Perplexity under 30 tokens:", perp)
            wandb.log({f"{args.prefix}perplexity_under_30_tokens": perp})
            

        import json
        file = "perplexity_text/perplexity_wandb_log.json"
        if "perplexity_wandb_log.json" in os.listdir("perplexity_text/"):
            with open(file, "r") as f:
                runs_done = json.load(f)
        else:
            runs_done = {}
            
        runs_done[run_name] = True
        with open(file, "w") as f:
            json.dump(runs_done, f)
            print("Logged run to json.")
            
        print("Calculating perplexity for all lengths...")

    
        temp = []
        for p in pred_2:
            temp.append(p)
            
        print("Total length:", len(temp))
        pred = temp
        perplexity = evaluate.load("perplexity", module_type="metric")
        
        for i in range(len(pred)):
            perplexity.add_batch(predictions=[pred[i]])
            
        with torch.no_grad():
            perp = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
            
            print("Perplexity all lengths:", perp)
            wandb.log({f"{args.prefix}perplexity_all_tokens": perp})
    