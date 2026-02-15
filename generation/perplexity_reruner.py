import wandb
import argparse
import os
import tqdm
from config import epoch
os.environ["MKL_THREADING_LAYER"] = "GNU"
parser = argparse.ArgumentParser()


parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_runs", type=int, default=1000)
parser.add_argument("--last_epoch_only", action="store_true", default=False)
args = parser.parse_args()
api = wandb.Api()

run = api.runs("frozenwolf/cbm-generation-new")
runs = list(run)

# >>> r[0].config
# {'seed': 42, 'DEBUG': False, 'dataset': 'SetFit/sst2', 'batch_size': 4, 'max_length': 350, 'num_workers': 0, 'concept_loss': 1, 'residual_dim': 768, 'neg_entropy_loss': 0, 'elastic_net_alpha': 1, 'discrimination_loss': 0, 'orthogonal_loss_weight': 0, 'residual_penalty_weight': 0}
# >>> r[0].id
# '0qinar9i'

### model path - from_pretained_llama3_lora_cbm_01w93yoj/
# ls from_pretained_llama3_lora_cbm_01w93yoj/SetFit_sst2/
# cbl_epoch_1.pt  cbl_epoch_2.pt  cbl_low_score_epoch_3.pt  llama3_epoch_1  llama3_epoch_2  llama3_low_score_epoch_3

tag = str(args.num_runs) + "_runs"
## get previously done runs:
previously_done_runs = []
for i in os.listdir("perplexity_text/"):
    if tag in i and i.endswith(".pkl"):
        if args.last_epoch_only:
            if "last_epoch_only" not in i:
                continue
        run_name = i.split("_")[0]
        
        if f"{args.num_runs}_runs.pkl" not in i:
            continue
        
        previously_done_runs.append(run_name)
## get all available run in current dir:
print("Previously done runs:", previously_done_runs)
chosen_runs = []

for local_runs in os.listdir("."):
    if "from_pretained_llama3_lora_cbm_" in local_runs:
        run_name = local_runs.split("from_pretained_llama3_lora_cbm_")[1].split("/")[0]
        if run_name in previously_done_runs:
            print("Run already done, skipping:", run_name)
            continue

        # print("Checking run:", run_name)
        found = False
        for r in runs:
            if "intervention" in r.displayName:
                print("Skipping intervention run:", r.run_name)
                continue
            if r.id == run_name:
                # print("Found matching wandb run, will be evaluated:", run_name)
                found = True
                best_epoch = -1
                subdir = local_runs + "/" + r.config["dataset"].replace('/', '_')
                if args.last_epoch_only:
                    best_epoch = epoch[r.config["dataset"]]
                    
                else:
                    ## best checkpoint will be one with largest epoch number but no "low_score" in the name
                    
                    best_epoch = -1
                    available_ckpt = os.listdir(subdir)
                    for ckpt in available_ckpt:
                        if "low_score" in ckpt:
                            continue
                        if "epoch_" in ckpt:
                            epoch_num = int(ckpt.split("epoch_")[1].split(".pt")[0])
                            if epoch_num > best_epoch:
                                best_epoch = epoch_num

                    if best_epoch == -1:
                        # print("No valid checkpoint found for run, skipping:", run_name)
                        break
                
                ## get peft_path
                if args.last_epoch_only:
                    # cbl_low_score_epoch_3.pt
                    cbl_path = subdir + "/cbl_low_score_epoch_" + str(best_epoch) + ".pt"
                    peft_path = subdir + "/llama3_low_score_epoch_" + str(best_epoch)
                else:
                    # cbl_epoch_2.pt
                    cbl_path = subdir + "/cbl_epoch_" + str(best_epoch) + ".pt"
                    peft_path = subdir + "/llama3_epoch_" + str(best_epoch)
                    
                
                r.config["peft_path"] = peft_path
                r.config["cbl_path"] = cbl_path
                ## get cbl_path

                chosen_runs.append(r)
                print("Chosen run:", run_name, "cbl_path:", cbl_path, "peft_path:", peft_path)
                break

        if not found:
            # print("No matching wandb run found for, will be skipped:", run_name)
            continue

print("Total number of chosen runs to be evaluated:", len(chosen_runs))
### need to run the inference for the chosen runs 1000 times and save it at perplexity_text/5uwsnbyp_generated_texts_<seed>_1000_runs.pkl
for chosen_run in tqdm.tqdm(chosen_runs):
    print("Running inference for run:", chosen_run.id)
    hparam = chosen_run.config
    print("Hyperparameters:", hparam)
    del hparam['DEBUG']
    del hparam['seed']
    hparam_str = ""
    for key in hparam.keys():
        hparam_str += " --" + str(key) + " " + str(hparam[key])
    cmd = "python inference.py --seed " + str(args.seed) + " --num_runs " + str(args.num_runs) + f" --output_path perplexity_text/{chosen_run.id}_generated_texts_" + ("last_epoch_only" if args.last_epoch_only else "") + str(args.seed) + "_" + str(args.num_runs) + "_runs.pkl" + hparam_str
    print("inference Command:", cmd)

    eval_cmd = f"python perplexity_calc.py --path perplexity_text/{chosen_run.id}_generated_texts_" + ("last_epoch_only" if args.last_epoch_only else "") + str(args.seed) + "_" + str(args.num_runs) + f"_runs.pkl --prefix {args.num_runs}_"
    eval_cmd += "last_epoch_only" if args.last_epoch_only else "" ## prefix adjustment
    print("eval Command:", eval_cmd)
    
    os.system(cmd)
    ### need to run the perplexity calculation for each of the generated files
    #  os.system(f"python perplexity_calc.py --path perplexity_text/{run_file}")
    print("Calculating perplexity for run:", chosen_run.id)
    result = os.system(eval_cmd)
    print(f"Perplexity calculation command output status: {result}")
    print("Finished processing run:", chosen_run.id)
    
    
## there ar some runs that have inference done but not perplexity calculation, need to check and run those as well
unfinished_runs = []
list(filter(lambda x: "1000_perplexity_under_30_tokens" in x.config, runs))

for r in runs:
    if "1000_perplexity_under_30_tokens" not in r._attrs["summaryMetrics"]:
        unfinished_runs.append(r)
        
print("Total number of unfinished runs to be evaluated for perplexity:", len(unfinished_runs))

for unfinished_run in tqdm.tqdm(unfinished_runs):
    print("Calculating perplexity for unfinished run:", unfinished_run.id)
    eval_cmd = f"python perplexity_calc.py --path perplexity_text/{unfinished_run.id}_generated_texts_" + ("last_epoch_only" if args.last_epoch_only else "") + str(args.seed) + "_" + str(args.num_runs) + f"_runs.pkl --prefix {args.num_runs}_"
    eval_cmd += "last_epoch_only" if args.last_epoch_only else "" ## prefix adjustment
    print("eval Command:", eval_cmd)
    
    result = os.system(eval_cmd)
    print(f"Perplexity calculation command output status: {result}")
    print("Finished processing unfinished run:", unfinished_run.id)