import argparse
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
from module_intervention import CustomLlamaModel, CustomLlamaForCausalLM, amplify_intervention, compute_training_losses
import time
import random
# from torch.cuda.amp import autocast, GradScaler
from utils import elastic_net_penalty, mean_pooling, eos_pooling
import wandb
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--intermediate_loc", type=int, default=24)
parser.add_argument("--DEBUG", action='store_true', help="If set, use a smaller subset of data for quick debugging.")
parser.add_argument("--concept_loss", type=float, default=1.0)

parser.add_argument("--intermediate_recon_loss", type=float, default=1.0)
parser.add_argument("--generation_recon_loss", type=float, default=1.0)

parser.add_argument("--cyclic_loss", type=float, default=1.0)
parser.add_argument("--intervention_gen_loss", type=float, default=1.0)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--intervention_margin", type=float, default=10.0)
parser.add_argument("--intervention_spread", type=float, default=2.0)
parser.add_argument(
    "--intermediate_sizes", 
    type=int, 
    nargs="+", 
    default=[2048, 1024, 512, 128],
    help="List of hidden dimensions for the U-Net bottleneck"
)
parser.add_argument(
    "--skip_dropout", 
    type=float, 
    default=0.0, 
    help="Dropout rate for skip connections in the U-Net"
)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text


    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encoded_text.items()}
        return t

    def __len__(self):
        return len(self.encoded_text['input_ids'])


def build_loaders(encoded_text, mode):
    dataset = ClassificationDataset(encoded_text)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    set_seed(args.seed)
    scaler = torch.amp.GradScaler("cuda")

    wandb.init(project="cbm-generation-new", name=f"intervention-{args.dataset}-{args.seed}",
               config=vars(args))
    
    run_name = wandb.run.id
    print("loading data...")
    train_dataset = load_dataset(args.dataset, split='train')
    test_dataset = load_dataset(args.dataset, split='test')
    if args.dataset == 'SetFit/sst2':
        val_dataset = load_dataset(args.dataset, split='validation')

    if args.dataset != 'SetFit/sst2':
        d_list = []
        for i in range(CFG.class_num[args.dataset]):
            d_list.append(
                train_dataset.filter(lambda e: e['label'] == i).select(range(100000 // CFG.class_num[args.dataset])))
        train_dataset = concatenate_datasets(d_list)

    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "")
            example["text"] = example["text"].replace("#39;", "'")
            return example
        train_dataset = train_dataset.map(replace_bad_string)

    print("training data len: ", len(train_dataset))
    if args.dataset == 'SetFit/sst2':
        print("val data len: ", len(val_dataset))

    print("tokenizing...")

    lora_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                  "down_proj"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)

    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token

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


    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "")
            example["text"] = example["text"].replace("#39;", "'")
            return example
        test_dataset = test_dataset.map(replace_bad_string)

    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, mode="train")
    if args.dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")
    test_loader = build_loaders(encoded_test_dataset, mode="test")

    print("preparing backbone")
    # preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = CustomLlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
    preLM.create_intermediate(args.intermediate_loc, len(concept_set), intermediate_sizes=args.intermediate_sizes, skip_dropout=args.skip_dropout)
    preLM.to(device)
    
    preLM_generator = CustomLlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
    preLM_generator.model = preLM
    preLM_generator.lm_head.to(device)
    
    ## print numel
    total_params = sum(p.numel() for p in preLM.parameters())
    trainable_params = sum(p.numel() for p in preLM.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} = {trainable_params/total_params:.4f} of total")
    wandb.log({"trainable_parameters": trainable_params, "trainable_ratio": trainable_params/total_params})
    
    
    # preLM = get_peft_model(preLM, lora_config)
    # preLM.print_trainable_parameters()
    # lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(preLM.intermediate.parameters(), lr=args.lr)
    
    # if args.discrimination_loss > 0:
    #     cbl = CBL(config, len(concept_set), tokenizer).to(device)
    # else:
    #     cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    # opt_cbl = torch.optim.Adam(cbl.parameters(), lr=5e-5)
    print("preparing classifier")
    
    
    print("start training...")
    best_loss = float('inf')
    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    prefix += "./from_pretained_llama3_lora_cbm_" + run_name
    prefix += "/"
    prefix += d_name
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "llama3"
    cbl_name = "cbl"

    start = time.time()
    best_epoch = -1
    epochs = args.epoch
    
    
    reconstr_crit = torch.nn.MSELoss()
    CSE_crit = torch.nn.CrossEntropyLoss()
            





    
    
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        preLM_generator.train()
        preLM.intermediate.train()
        # cbl.train()
        training_losses = {
                "loss": [],
                "concept_loss": [],
                "intermediate_reconstruction_loss": [],
                "generation_reconstruction_loss": [],
                "cyclic_concept_loss": [],
                "intervened_generation_loss": [],
            }

        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            gate_logs = {}
            for i, gate_param in enumerate(preLM.intermediate.gates):
                    # .item() is crucial to get the scalar value
                    gate_logs[f"gates/skip_layer_{i}"] = gate_param.item()

                # Log the final output gate (how much the whole bottleneck affects Llama)
            gate_logs["gates/final_residual_gate"] = preLM.intermediate.final_gate.item()


            batch = {k: v.to(device) for k, v in batch.items()}
            concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
            word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
            
            with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16):
                loss_dict = compute_training_losses(
                                   batch=batch,
                                   preLM=preLM,
                                   preLM_generator=preLM_generator,
                                   reconstr_crit=reconstr_crit,
                                   CSE_crit=CSE_crit,
                                   concept_label=concept_label,
                                   word_label=word_label,
                                   concept_set=concept_set,
                                   config=config,
                                   args=args,
                               )
                
                scaler.scale(loss_dict["loss"]).backward()
                scaler.step(opt_prelm)
                scaler.update()
                opt_prelm.zero_grad()


            log = {}

            for k, v in loss_dict.items():
                training_losses[k].append(v.detach().cpu().item())
                log[k]  = training_losses[k][-1]
            
            log["epoch"] = e + 1
            log["batch"] = i + 1
            
            log = {**log, **gate_logs}
            
            wandb.log(log)
            
            if args.DEBUG and i >= 2:
                break
            
            
        avg_metrics = {}
        for key in training_losses.keys():
            if len(training_losses[key]) > 0:
                avg_metrics[key] = sum(training_losses[key]) / len(training_losses[key])
        print("Epoch ", e + 1, " training losses: ", avg_metrics)
        wandb.log({f"avg_{k}": avg_metrics[k] for k in avg_metrics.keys()})


        if args.dataset == 'SetFit/sst2':
            preLM.eval()
            preLM_generator.eval()
            preLM.intermediate.eval()

            val_losses = {
                "val_loss": [],
                "val_concept_loss": [],
                "val_intermediate_reconstruction_loss": [],
                "val_generation_reconstruction_loss": [],
                "val_cyclic_concept_loss": [],
                "val_intervened_generation_loss": [],
            }

            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = {k: v.to(device) for k, v in batch.items()}

                concept_label = torch.where(
                    batch["attention_mask"][:, :-1] == 0,
                    -100,
                    batch["label"].view(-1, 1),
                )

                word_label = torch.where(
                    batch["attention_mask"][:, :-1] == 0,
                    -100,
                    batch["input_ids"][:, 1:],
                )

                with torch.no_grad():
                    with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16):
                        loss_dict = compute_training_losses(
                        batch=batch,
                        preLM=preLM,
                        preLM_generator=preLM_generator,
                        reconstr_crit=reconstr_crit,
                        CSE_crit=CSE_crit,
                        concept_label=concept_label,
                        word_label=word_label,
                        concept_set=concept_set,
                        config=config,
                        args=args,
                    )


                for k, v in loss_dict.items():
                    val_losses[f"val_{k}"].append(v.detach().cpu().item())
                

                if args.DEBUG and i >= 2:
                    break
 

            avg_val_loss = {}
            for key in val_losses:
                if len(val_losses[key]) > 0:
                    avg_val_loss[key] = sum(val_losses[key]) / len(val_losses[key])

            print(f"Epoch {e+1} validation losses:", avg_val_loss)

            wandb.log({f"avg_{k}": v for k, v in avg_val_loss.items()})

            avg_val_concept_loss = avg_val_loss["val_concept_loss"]
            avg_val_total_loss = avg_val_loss["val_loss"]



            avg_val_loss = avg_val_total_loss
            if avg_val_loss < best_loss:
                best_epoch = e + 1
                print("save model")
                best_loss = avg_val_loss
                torch.save(preLM.intermediate.state_dict(), prefix + model_name + "_epoch_" + str(e + 1))
                wandb.log({"best_model_epoch": e + 1})
            else:
                torch.save(preLM.intermediate.state_dict(), prefix + model_name + "_low_score_epoch_" + str(e + 1))
        else:
            print("save model")
            torch.save(preLM.intermediate.state_dict(), prefix + model_name + "_epoch_" + str(e + 1))

        if args.DEBUG:
            break

    end = time.time()
    print("time of training CBM:", (end - start) / 3600, "hours")
    
    ## delete previous models to save space
    import gc
    del preLM, opt_prelm, loss_dict
    

    torch.cuda.empty_cache()
    gc.collect()
    preLM = CustomLlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16)
    preLM.create_intermediate(args.intermediate_loc, len(concept_set))
    preLM.to(device)
    ## lOAD BEST MODEL AND
    best_path = prefix + model_name + "_epoch_" + str(best_epoch)
    state_dict = torch.load(best_path, map_location=device)  # or "cuda" if needed
    preLM.intermediate.load_state_dict(state_dict)
    preLM.eval()
    preLM.intermediate.eval()
    preLM_generator.model = preLM
    preLM_generator.eval()

    
    ### TEST STEERABILITY AFTER TRAINING
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    classifier_path = args.dataset.replace('/', '_') + "_classifier.pt"
    classifier = Roberta_classifier(len(concept_set)).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    


    pred = []
    text = []

    acc_metric = evaluate.load("accuracy")

    with torch.no_grad():
        for i in tqdm(range(100 // len(concept_set))):
            print("example", str(i), end="\r")
            input_ids = torch.tensor([tokenizer.encode("")]).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            for j in range(len(concept_set)):
                # original vector: one-hot for the concept
                v = [0] * len(concept_set)  # all concepts suppressed
                v[j] = 1                    # activate target concept j
                B, T = input_ids.shape
                intervene_tensor = torch.tensor(v, device=device).view(1, 1, -1).expand(B, T, len(concept_set))

                preLM_generator.model.intervene = intervene_tensor
                preLM_generator.model.intervention_margin = args.intervention_margin
                preLM_generator.model.intervention_spread = args.intervention_spread
                print("Gen")
                with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16):
                    output_tokens = preLM_generator.generate(
                                        input_ids,
                                        attention_mask=attention_mask,       # must pass if padding exists
                                        use_cache=True,
                                        max_new_tokens=100,                  # instead of length
                                        temperature=0.7,                     # temp -> temperature
                                        top_k=100,                           # topk -> top_k
                                        top_p=0.9,                           # topp -> top_p
                                        repetition_penalty=1.5,
                                        pad_token_id=128001                   # set pad_token_id to avoid warnings
                                    )
                preLM_generator.model.intervene = None
                decoded_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                print(f"Class {concept_set[j]}: {decoded_text}")
                text.append(decoded_text)

                encoded_input = roberta_tokenizer(
                    decoded_text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=512
                ).to(device)

                roberta_input = {
                    "input_ids": encoded_input["input_ids"], 
                    "attention_mask": encoded_input["attention_mask"]
                }

                logits = classifier(roberta_input)
                pred.append(logits)


    pred = torch.cat(pred, dim=0).detach().cpu()
    pred_labels = np.argmax(pred.numpy(), axis=-1)

    refs = list(range(len(concept_set))) * (100 // len(concept_set))
    acc_metric.add_batch(predictions=pred_labels, references=refs)
    accuracy = acc_metric.compute()

    print("Steerability test accuracy:", accuracy)
    wandb.log({"steerability_test_accuracy": accuracy})

    
    
    
    ### TEST CONCEPT PREDICTION AFTER TRAINING
    print("eval concepts...")
    metric = evaluate.load("accuracy")
    concept_predictions = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16):
                pre_hidden_states, causal_mask, position_embeddings = (
                    preLM.firsthalf_forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                )
                concepts, skips = preLM.intermediate.encode(pre_hidden_states)

        concept_predictions.append(eos_pooling(concepts, batch["attention_mask"]))
    concept_predictions = torch.cat(concept_predictions, dim=0).detach().cpu()
    pred = np.argmax(concept_predictions.numpy(), axis=-1)
    metric.add_batch(predictions=pred, references=encoded_test_dataset["label"])
    print("Concept prediction accuracy:")
    acc = metric.compute()
    print(acc)
    wandb.log({"concept_prediction_accuracy": acc})
    
    
    
    #### TEST PERPLEXITY AFTER TRAINING
    print("Test perplexity after training:")
    set_seed(args.seed)
    
    pred = []
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    for i in tqdm(range(100)):
        print("example", str(i), end="\r")
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16):
                text_ids = preLM_generator.generate(
                                        input_ids,
                                        attention_mask=attention_mask,       # must pass if padding exists
                                        use_cache=True,
                                        max_new_tokens=100,                  # instead of length
                                        temperature=0.7,                     # temp -> temperature
                                        top_k=100,                           # topk -> top_k
                                        top_p=0.9,                           # topp -> top_p
                                        repetition_penalty=1.5,
                                        pad_token_id=128001                   # set pad_token_id to avoid warnings
                                    )
            pred.append(tokenizer.decode(text_ids[0], skip_special_tokens=True ))
            if len(pred[-1].split()) > 30:
                continue
            perplexity.add_batch(predictions=[pred[i]])

        ## print some generated texts
    print("Some generated texts:")
    for i in range(5):
        print(pred[i])
    import pickle
    if "perplexity_text" not in os.listdir("./"):
        try:
            os.mkdir("perplexity_text")
        except:
            pass
    pickle.dump(pred, open(f"perplexity_text/{run_name}_generated_texts_{args.seed}.pkl", "wb"))
    del preLM
    gc.collect()
    torch.cuda.empty_cache()

    print("Perplexity: (under 30 tokens)")
    perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
    print(perplexity)
    wandb.log({"perplexity_under_30_tokens": perplexity})
    
    print("Now for all tokens:")
    perplexity = evaluate.load("perplexity", module_type="metric")
    for p in pred:
        perplexity.add_batch(predictions=[p])
    perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
    print(perplexity)
    wandb.log({"perplexity_all_tokens": perplexity})
        
    