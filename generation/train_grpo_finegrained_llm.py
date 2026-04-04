"""
GRPO (Group Relative Policy Optimization) training for CB-LLMs.

Proper GRPO algorithm with KL-divergence penalty to prevent deviation from the
reference (pretrained) model:

    1. Load a pretrained model from train_combined.py checkpoint (via wandb run_id or folder path).
       This becomes the FROZEN reference policy π_ref.
    2. Create a trainable copy π_θ (same architecture, same weights, but trainable).
    3. For each training step:
       a. Sample a random concept index c.
       b. Generate G trajectories from π_θ with intervention on concept c.
       c. Score each trajectory with RoBERTa classifier ensemble → rewards r_1..r_G.
       d. Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + eps), then clip.
       e. For trajectories with non-zero advantage:
          - Compute log π_θ(tokens | intervention)   (with grad, teacher-forced).
          - Compute log π_ref(tokens | intervention)  (no grad, teacher-forced).
          - KL ≈ log π_θ - log π_ref                  (per-token, averaged over sequence).
          - Policy loss = -A_i * log π_θ
          - Total loss = mean(policy_loss) + β * mean(KL)
       f. Update π_θ.
    4. Run the same evaluation suite as train_combined.py (steerability, concept prediction,
       weight analysis, perplexity).

Usage:
    python train_grpo.py --dataset SetFit/sst2 --pretrained_run_id <wandb_run_id>
    python train_grpo.py --dataset SetFit/sst2 --pretrained_path <folder_path>

Reward modes:
    cosine        : cos_sim_cubed vs one-hot target (MPNet embeddings)
    aggressive    : inverse-rank * similarity (MPNet embeddings)
    llm           : base Llama-3-8B as judge (steerability + coherence + grammar)
    reward_model  : Skywork-Reward-V2-Llama-3.1-8B as judge.
                    Controlled by two extra flags:
                      --rm_criteria_mode {separate, together, separate_hybrid}
                        separate        : score relevance + grammar with two RM prompts;
                                          sigmoid each → multiply.
                        together        : single combined RM prompt; one sigmoid score.
                        separate_hybrid : RM scores grammar only (sigmoid); steerability
                                          is computed with MPNet cosine (same as cosine mode);
                                          final reward = grammar_sigmoid × cosine_reward.
                      --rm_batch_size N : number of trajectories per RM forward pass
                                          (default 0 = all at once).
"""
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
import config_finegrained as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBLResidual, CBL, Roberta_classifier
import time
from module_intervention import amplify_intervention
from utils import elastic_net_penalty, mean_pooling, eos_pooling, get_labels, cos_sim_cubed
import wandb
import glob
import copy
import gc


def cuda_gc():
    """Best-effort memory cleanup for long-running GPU training.

    Notes:
      - `gc.collect()` only helps once Python refs are dropped.
      - `torch.cuda.empty_cache()` releases cached blocks back to the allocator.
      - Calling this too frequently can slow training, but reduces peak VRAM.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

parser = argparse.ArgumentParser()


parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=4)

parser.add_argument("--epoch_multiplier", type=int, default=1, help="Epoch multiplier to increase total training steps (for debugging).")
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--discrimination_loss", type=float, default=0.0)
parser.add_argument("--arch_type", type=str, default="residual", choices=["residual", "non_residual"])
parser.add_argument("--residual_dim", type=int, default=768)
parser.add_argument("--DEBUG", action='store_true', help="If set, use a smaller subset of data for quick debugging.")
parser.add_argument("--classifier_weight_suffixes", type=str, default="_seed42,_seed123,_seed456", 
                    help="Comma-separated list of classifier weight suffixes to test (e.g., '_seed42,_seed123,_seed456')")
parser.add_argument("--labeling", type=str, default="mpnet", help="mpnet, angle, simcse, llm")
parser.add_argument("--automatic_concept_correction", action='store_true', help="If set, automatically set concept labels to 0 for concepts that are not present in the example according to the ground truth label. This is a form of training intervention to correct mislabeled concepts.")

# ---- Pretrained model loading ----
parser.add_argument("--pretrained_run_id", type=str, default=None,
                    help="Wandb run_id from train_combined.py. Will auto-locate the saved checkpoint folder.")
parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Exact folder path to pretrained checkpoint (the prefix dir containing llama3_epoch_* and cbl_epoch_*.pt). "
                         "If given, --pretrained_run_id is ignored.")

# ---- GRPO hyperparameters ----
parser.add_argument("--grpo_epochs", type=int, default=1, help="Number of GRPO training epochs over the dataset.")
parser.add_argument("--grpo_loss_weight", type=float, default=1.0, help="Weight for the GRPO policy gradient loss.")
parser.add_argument("--grpo_kl_weight", type=float, default=0.1, help="Weight β for KL divergence penalty against reference model.")
parser.add_argument("--grpo_num_trajectories", type=int, default=4, help="Number of rollouts (G) per GRPO step.")
parser.add_argument("--grpo_gen_length", type=int, default=100, help="Max generation length for GRPO rollouts.")
parser.add_argument("--grpo_clip_advantage", type=float, default=5.0, help="Clip GRPO advantages to [-clip, clip].")
parser.add_argument("--grpo_lr", type=float, default=1e-5, help="Learning rate for GRPO fine-tuning.")
parser.add_argument("--grpo_steps_per_concept", type=int, default=-1,
                    help="If >0, override the effective number of epochs so that total GRPO steps scale as: original_epochs * num_concepts * grpo_steps_per_concept.")
parser.add_argument("--concept_distill_weight", type=float, default=0.0, help="Weight for concept prediction distillation loss (CE between policy and reference model concepts on real data). 0 disables it.")
parser.add_argument("--grpo_reward_mode", type=str, default="cosine",
                    choices=["cosine", "aggressive", "llm", "reward_model"],
                    help=(
                        "Reward type for GRPO:\n"
                        "  cosine       : cos_sim_cubed vs one-hot target (MPNet).\n"
                        "  aggressive   : inverse-rank * similarity (MPNet).\n"
                        "  llm          : base Llama-3-8B as few-shot JSON judge.\n"
                        "  reward_model : Skywork-Reward-V2-Llama-3.1-8B as judge.\n"
                        "                 See --rm_criteria_mode and --rm_batch_mode."
                    ))
parser.add_argument("--grpo_llm_max_text_len", type=int, default=200,
                    help="Max chars of each generated trajectory passed to the LLM judge prompt (used when --grpo_reward_mode llm).")

# ---- NEW: Reward-model reward mode flags ----
parser.add_argument("--rm_model_name", type=str,
                    default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    help="HuggingFace model id for the reward model (used when --grpo_reward_mode reward_model).")
parser.add_argument("--rm_criteria_mode", type=str, default="separate",
                    choices=["separate", "together", "separate_hybrid"],
                    help=(
                        "How to construct RM prompts (reward_model mode only).\n"
                        "  separate        : two RM prompts per trajectory — relevance + grammar.\n"
                        "                    Sigmoid each and multiply.\n"
                        "  together        : single combined RM prompt; one sigmoid score.\n"
                        "  separate_hybrid : RM scores grammar only (sigmoid); MPNet cosine\n"
                        "                    (same as cosine mode) scores steerability;\n"
                        "                    final reward = grammar_sigmoid x cosine_reward."
                    ))
parser.add_argument("--rm_batch_size", type=int, default=0,
                    help="Number of trajectories per RM forward pass. "
                         "0 (default) = score all trajectories in a single batched call.")
parser.add_argument("--rm_max_text_len", type=int, default=500,
                    help="Max chars of each trajectory passed into the RM prompt (reward_model mode).")
parser.add_argument("--rm_device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                help="Device to keep the reward/judge model on. "
                    "Used for --grpo_reward_mode reward_model (Skywork RM) and llm (base Llama judge). "
                    "Set to 'cuda:1', 'cuda:2', etc. to keep it resident on a specific GPU.")
parser.add_argument("--ref_device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                help="Device to keep the KL reference model (pi_ref) on. "
                    "Set to 'cuda:2', etc. to put the reference on a different GPU. "
                    "In llm reward mode, this may load a separate base Llama copy for pi_ref.")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the model on.")
# ---- Dataset mixing ----
parser.add_argument("--grpo_mix_dataset", action="store_true",
                    help="If set, replace grpo_mix_k of the G generated trajectories with real training texts from the current batch.")
parser.add_argument("--grpo_mix_k", type=int, default=1,
                    help="Number of generated trajectories to replace with real training texts when --grpo_mix_dataset is active.")
parser.add_argument("--grpo_active_concepts_k", type=int, default=-1,
                    help="If >0, restrict GRPO to intervening on only the first K concepts (by index) while still loading all concepts for evaluation.")
parser.add_argument("--grpo_subset_renorm", action="store_true",
                    help="If set, restrict reward computation to a top-K subset of concepts (plus the target concept) and renormalize similarities before applying the chosen reward mode.")
parser.add_argument("--grpo_subset_topk", type=int, default=20,
                    help="Top-K cutoff used when --grpo_subset_renorm is enabled (default 20).")

device = None  # set in main after parsing args

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text, s):
        self.encoded_text = encoded_text
        self.s = s


    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encoded_text.items()}
        y = torch.FloatTensor(self.s[idx])
        return t, y

    def __len__(self):
        return len(self.encoded_text['input_ids'])


def build_loaders(encoded_text, s, mode):
    dataset = ClassificationDataset(encoded_text, s)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=max(args.batch_size, args.grpo_mix_k), num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

def find_pretrained_checkpoint(run_id, dataset):
    """Locate the pretrained checkpoint folder from a wandb run_id.
    
    Convention from train_combined.py:
        prefix = ./from_pretained_llama3_lora_cbm_{run_id}/{d_name}/
        Best epoch:  llama3_epoch_{N}  and  cbl_epoch_{N}.pt
        Other epoch: llama3_low_score_epoch_{N}  and  cbl_low_score_epoch_{N}.pt
    
    Strategy: prefer best epoch (no 'low_score' in name), fallback to last epoch.
    """
    d_name = dataset.replace('/', '_')
    prefix = os.path.join(".", f"from_pretained_llama3_lora_cbm_{run_id}", d_name)
    if not os.path.isdir(prefix):
        raise FileNotFoundError(f"Checkpoint directory not found: {prefix}")
    
    # Find all cbl checkpoint files
    cbl_files = sorted(glob.glob(os.path.join(prefix, "cbl_epoch_*.pt")))
    cbl_low_files = sorted(glob.glob(os.path.join(prefix, "cbl_low_score_epoch_*.pt")))
    
    # Best epoch files (no 'low_score')
    best_cbl_files = [f for f in cbl_files if "low_score" not in f]
    
    if best_cbl_files:
        # Pick the highest best epoch
        best_epoch = max(
            int(os.path.basename(f).replace("cbl_epoch_", "").replace(".pt", ""))
            for f in best_cbl_files
        )
        peft_path = os.path.join(prefix, f"llama3_epoch_{best_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_epoch_{best_epoch}.pt")
    else:
        # No best epoch found, use the last low_score epoch
        if not cbl_low_files:
            raise FileNotFoundError(f"No checkpoint files found in {prefix}")
        last_epoch = max(
            int(os.path.basename(f).replace("cbl_low_score_epoch_", "").replace(".pt", ""))
            for f in cbl_low_files
        )
        peft_path = os.path.join(prefix, f"llama3_low_score_epoch_{last_epoch}")
        cbl_path = os.path.join(prefix, f"cbl_low_score_epoch_{last_epoch}.pt")
    
    if not os.path.isdir(peft_path):
        raise FileNotFoundError(f"LoRA adapter directory not found: {peft_path}")
    if not os.path.isfile(cbl_path):
        raise FileNotFoundError(f"CBL checkpoint not found: {cbl_path}")
    
    print(f"Found pretrained checkpoint: peft={peft_path}, cbl={cbl_path}")
    return peft_path, cbl_path


def compute_llm_rewards_batch(base_lm_model, base_lm_tokenizer, texts, concept_name, device,
                               max_text_len=200, max_new_tokens=512, debug=False):
    """Compute LLM-as-judge rewards using few-shot completion prompting.

    Uses two few-shot demonstrations so that base (non-instruct) Llama-3-8B reliably
    completes the JSON array pattern without instruction-following capability.
    The prompt ends with '[' (opening of the scores array) so greedy decoding
    simply continues the established pattern.

    Args:
        base_lm_model     : AutoModelForCausalLM (frozen, on device).
        base_lm_tokenizer : matching tokenizer (pad_token must be set).
        texts             : list[str] — trajectories to score (caller handles filtering;
                            no internal .strip() filtering to avoid index misalignment).
        concept_name      : str, the target concept label for concept_relevance scoring.
        device            : torch.device.
        max_text_len      : max chars of each trajectory included in the prompt.
        max_new_tokens    : token budget for the model's JSON completion.
        debug             : if True, print the full input prompt and raw model output.

    Returns:
        rewards    : list[float] same length as `texts`, values in [0, 1].
        raw_scores : list[dict|None] — parsed per-trajectory score dicts (for logging).
    """
    import json, re

    cuda_gc()

    n = len(texts)
    rewards = [0.0] * n
    raw_scores = [None] * n

    if n == 0:
        cuda_gc()
        return rewards, raw_scores

    # Build numbered trajectory block — no internal filtering so indices stay aligned
    trajectory_block = ""
    for idx, t in enumerate(texts):
        trunc = t[:max_text_len].replace('"', "'").replace("\n", " ")
        trajectory_block += f"({idx + 1}) {trunc}\n"

    # Few-shot completion prompt.
    # Two demonstrations teach the pattern; prompt ends with '[' so the
    # base model just continues the completion.
# OLD
# Three entries per demo with varied non-alternating scores to prevent
    # the model from learning a positional pattern (9,0,9,0,...).
    few_shot = (
        'Concept: "technology"\n'
        '(1) The smartphone revolution transformed how billions communicate daily.\n'
        '(2) She baked a rich chocolate cake with extra frosting for the party.\n'
        '(3) Engineers debated the tradeoffs of the new processor architecture.\n'
        'Scores: [{"concept_relevance":9,"coherence":8,"grammar":9},{"concept_relevance":1,"coherence":8,"grammar":9},{"concept_relevance":7,"coherence":7,"grammar":8}]\n\n'
        'Concept: "nature"\n'
        '(1) Towering oaks and mossy boulders lined the winding forest trail.\n'
        '(2) The quarterly earnings report exceeded analyst expectations by 12 percent.\n'
        '(3) A gentle rain softened the dusty summer soil.\n'
        'Scores: [{"concept_relevance":9,"coherence":9,"grammar":8},{"concept_relevance":0,"coherence":8,"grammar":8},{"concept_relevance":8,"coherence":8,"grammar":9}]\n\n'
    )

    prompt = (
        f"Score each text for concept_relevance, coherence, and grammar (integers 0-9).\n"
        f"Output exactly {n} JSON objects then close the array with ].\n\n"
        f"{few_shot}"
        f'Concept: "{concept_name}"\n'
        f"{trajectory_block}"
        f'Scores: [{{"'
    )
    if debug:
        print(f"\n[LLM reward DEBUG] ===== INPUT PROMPT =====\n{prompt}\n{'='*60}")

    # Dynamic budget: leave max_new_tokens of headroom in context window
    # model_max_length on base Llama-3 is a huge sentinel (~1e30), so cap it.
    model_max = getattr(base_lm_tokenizer, 'model_max_length', 8192)
    model_max = min(model_max, 8192)
    max_prompt_tokens = max(512, model_max - max_new_tokens)
    enc = base_lm_tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=max_prompt_tokens
    ).to(device)

    with torch.inference_mode():
        out_ids = base_lm_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=base_lm_tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens (strip prompt prefix)
    new_ids = out_ids[0][enc["input_ids"].shape[1]:]
    response = base_lm_tokenizer.decode(new_ids, skip_special_tokens=True)

    if debug:
        print(f"[LLM reward DEBUG] ===== RAW MODEL OUTPUT =====\n{response}\n{'='*60}")

    # Prepend the '[' we held back — the model completed the array body
    candidate = '[{"' + response

    parsed = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Tolerate trailing text: grab the first complete [...] block
        match = re.search(r'\[.*?\]', candidate, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    parse_ok = parsed is not None and isinstance(parsed, list) and len(parsed) == n
    if parse_ok:
        for idx, entry in enumerate(parsed):
            if isinstance(entry, dict):
                cr = max(0.0, min(9.0, float(entry.get("concept_relevance", 0)))) / 9.0
                co = max(0.0, min(9.0, float(entry.get("coherence", 0)))) / 9.0
                gr = max(0.0, min(9.0, float(entry.get("grammar", 0)))) / 9.0
                rewards[idx] = (cr + co + gr) / 3.0
                raw_scores[idx] = {"concept_relevance": cr, "coherence": co, "grammar": gr}
    else:
        got_len = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
        print(f"[LLM reward] JSON parse failed or wrong length (got {got_len}, expected {n}).")
        print(f"[LLM reward] candidate[:400]: {candidate[:400]}")

    del enc, out_ids, new_ids, candidate, response
    if 'match' in locals():
        del match
    del trajectory_block, prompt, few_shot
    cuda_gc()

    return rewards, raw_scores


# =============================================================================
# NEW: Skywork reward model scoring
# =============================================================================

def compute_reward_model_scores(
    rm_model,
    rm_tokenizer,
    texts,
    concept_name,
    device,
    criteria_mode="separate",
    rm_batch_size=0,
    max_text_len=500,
    debug=False,
):
    """Score trajectories with a Skywork-style sequence-classification reward model.

    Raw RM logits span a wide numeric range (e.g. 3 → 23). A sigmoid maps them
    to (0, 1), consistent with other reward modes and GRPO advantage normalization.

    Args:
        rm_model       : AutoModelForSequenceClassification, already on `device`.
        rm_tokenizer   : matching tokenizer.
        texts          : list[str] — decoded trajectories. Indices are preserved;
                         no internal filtering so the caller's index alignment holds.
        concept_name   : str — the concept the policy was steered toward.
        device         : torch.device — the main training device (RM is already here).
        criteria_mode  : "separate", "together", or "separate_hybrid".
            separate        : two RM prompts per trajectory (relevance + grammar);
                              sigmoid each and multiply.
            together        : single combined RM prompt; sigmoid of one score.
            separate_hybrid : RM scores grammar only (sigmoid). Steerability
                              (cosine) is computed externally by the caller and
                              multiplied in. This function returns grammar sigmoids
                              in `rewards`; the caller is responsible for the cosine
                              multiplication. raw_scores_list carries "grammar_score".
        rm_batch_size  : int — trajectories per RM forward pass. 0 = all at once.
        max_text_len   : int — max chars of each trajectory in the RM prompt.
        debug          : bool — print prompts/scores for the first call.

    Returns:
        rewards          : list[float], same length as `texts`, values in (0, 1).
                           For separate_hybrid, these are grammar-only sigmoids.
        raw_scores_list  : list[dict|None] — per-trajectory raw RM scores.
            Keys: "rm_score" (all), "relevance_score"/"grammar_score" (separate/hybrid).
    """
    n = len(texts)
    if debug:
        print(f"\n[RM reward DEBUG] compute_reward_model_scores called with {n} trajectories, criteria_mode={criteria_mode}, rm_batch_size={rm_batch_size}")
        print(f"  concept_name: {concept_name}")    
    rewards = [0.0] * n
    raw_scores_list = [None] * n

    if n == 0:
        return rewards, raw_scores_list

    # ------------------------------------------------------------------
    # Prompt templates — assistant turn is always the trajectory text.
    # ------------------------------------------------------------------
    relevance_user = f"Write a text about the concept: {concept_name}"
    grammar_user   = "Write a grammatically correct and fluent paragraph."
    combined_user  = f"Write a grammatically correct and fluent text about the concept: {concept_name}"

    def _make_formatted(user_turn, response_text):
        conv = [
            {"role": "user",      "content": user_turn},
            {"role": "assistant", "content": response_text[:max_text_len]},
        ]
        formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
        if rm_tokenizer.bos_token and formatted.startswith(rm_tokenizer.bos_token):
            formatted = formatted[len(rm_tokenizer.bos_token):]
        return formatted

    def _sigmoid(x: float) -> float:
        return float(torch.sigmoid(torch.tensor(x)).item())

    def _score_formatted(formatted_list):
        """Batch-score pre-formatted strings; chunk size controlled by rm_batch_size."""
        chunk = rm_batch_size if rm_batch_size > 0 else len(formatted_list)
        all_scores = []
        for start in range(0, len(formatted_list), chunk):
            chunk_list = formatted_list[start: start + chunk]
            tokenized = rm_tokenizer(
                chunk_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)
            with torch.no_grad():
                logits = rm_model(**tokenized).logits  # (chunk, 1)
            all_scores.extend(logits[:, 0].tolist())
            del tokenized, logits
        return all_scores

    # ------------------------------------------------------------------
    # Criteria mode dispatch
    # ------------------------------------------------------------------
    if criteria_mode == "separate":
        rel_formatted  = [_make_formatted(relevance_user, t) for t in texts]
        gram_formatted = [_make_formatted(grammar_user,   t) for t in texts]

        if debug:
            print(f"[RM reward DEBUG] criteria=separate, rm_batch_size={rm_batch_size}")
            print(f"  relevance prompt[0]:\n{rel_formatted[0][:400]}")
            print(f"  grammar prompt[0]:\n{gram_formatted[0][:400]}")

        rel_scores  = _score_formatted(rel_formatted)
        gram_scores = _score_formatted(gram_formatted)

        for i in range(n):
            r_rel  = _sigmoid(rel_scores[i])
            r_gram = _sigmoid(gram_scores[i])
            rewards[i] = r_rel * r_gram
            raw_scores_list[i] = {
                "relevance_score": rel_scores[i],
                "grammar_score":   gram_scores[i],
                "rm_score":        (rel_scores[i] + gram_scores[i]) / 2.0,
            }
            if debug:
                print(f"  [{i}] rel={rel_scores[i]:.3f}→{r_rel:.3f}  "
                      f"gram={gram_scores[i]:.3f}→{r_gram:.3f}  "
                      f"reward={rewards[i]:.4f}")

    elif criteria_mode == "together":
        comb_formatted = [_make_formatted(combined_user, t) for t in texts]

        if debug:
            print(f"[RM reward DEBUG] criteria=together, rm_batch_size={rm_batch_size}")
            print(f"  combined prompt[0]:\n{comb_formatted[0][:400]}")

        comb_scores = _score_formatted(comb_formatted)

        for i in range(n):
            rewards[i] = _sigmoid(comb_scores[i])
            raw_scores_list[i] = {"rm_score": comb_scores[i]}

            if debug:
                print(f"  [{i}] score={comb_scores[i]:.3f}→{rewards[i]:.4f}")

    else:  # separate_hybrid — grammar RM only; caller multiplies cosine reward
        gram_formatted = [_make_formatted(grammar_user, t) for t in texts]

        if debug:
            print(f"[RM reward DEBUG] criteria=separate_hybrid, rm_batch_size={rm_batch_size}")
            print(f"  grammar prompt[0]:\n{gram_formatted[0][:400]}")

        gram_scores = _score_formatted(gram_formatted)

        for i in range(n):
            r_gram = _sigmoid(gram_scores[i])
            rewards[i] = r_gram  # cosine factor applied by caller
            raw_scores_list[i] = {
                "grammar_score": gram_scores[i],
                "rm_score":      gram_scores[i],
            }
            if debug:
                print(f"  [{i}] gram={gram_scores[i]:.3f}→{r_gram:.3f} (cosine TBD by caller)")

    cuda_gc()
    return rewards, raw_scores_list


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()
    device = torch.device(args.device)
    reward_device = torch.device(args.rm_device)
    ref_device = torch.device(args.ref_device)
    set_seed(args.seed)

    # Validate: need either run_id or path
    if args.pretrained_run_id is None and args.pretrained_path is None:
        raise ValueError("Must provide either --pretrained_run_id or --pretrained_path")

    wandb.init(project="cbm-generation-new", name=f"grpo-finegrained-{args.dataset}-seed{args.seed}-{args.pretrained_run_id}",
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

    concept_set = CFG.concept_set[args.dataset]
    print("concept len: ", len(concept_set))  # concept_set: list of strings, len = num_concepts

    # Optionally cap the active concept set used for GRPO interventions and some evaluations.
    # We still load all concepts, but only indices in active_concept_indices will be used as targets.
    if args.grpo_active_concepts_k is not None and args.grpo_active_concepts_k > 0:
        k_cap = min(args.grpo_active_concepts_k, len(concept_set))
        active_concept_indices = list(range(k_cap))
        print(f"[GRPO] Using only first {k_cap} concepts as active GRPO targets.")
    else:
        active_concept_indices = list(range(len(concept_set)))

    d_name = args.dataset.replace('/', '_')
    label_prefix = "./"
    if args.labeling == 'mpnet':
        label_prefix += "mpnet_acs"
    elif args.labeling == 'simcse':
        label_prefix += "simcse_acs"
    elif args.labeling == 'angle':
        label_prefix += "angle_acs"
    elif args.labeling == 'llm':
        label_prefix += "llm_labeling"

    label_prefix += "/"
    label_prefix += d_name
    label_prefix += "/"
    
    print(f"Loading concept labels from: {label_prefix}")
    train_similarity = np.load(label_prefix + "/concept_labels_train.npy")  # (N_train, num_concepts)
    if args.dataset == 'SetFit/sst2':
        val_similarity = np.load(label_prefix + "/concept_labels_val.npy")  # (N_val, num_concepts)

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
        print("time of training intervention:", (end - start) / 3600, "hours")

    print("creating loader...")
    train_loader = build_loaders(encoded_train_dataset, train_similarity, mode="train")
    if args.dataset == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, val_similarity, mode="valid")
    
    test_similarity = np.zeros((len(encoded_test_dataset["label"]), 1), dtype=np.float32)
    test_loader = build_loaders(encoded_test_dataset, test_similarity, mode="test")


    # ======================================================================
    # Load pretrained checkpoint from train_combined.py
    # ======================================================================
    if args.pretrained_path is not None:
        # Direct folder path: expect llama3_epoch_* and cbl_epoch_*.pt inside
        pretrained_prefix = args.pretrained_path.rstrip("/")
        # Auto-detect best/last epoch
        cbl_files = sorted(glob.glob(os.path.join(pretrained_prefix, "cbl_epoch_*.pt")))
        cbl_low_files = sorted(glob.glob(os.path.join(pretrained_prefix, "cbl_low_score_epoch_*.pt")))
        best_cbl_files = [f for f in cbl_files if "low_score" not in f]
        if best_cbl_files:
            best_epoch = max(
                int(os.path.basename(f).replace("cbl_epoch_", "").replace(".pt", ""))
                for f in best_cbl_files
            )
            peft_path = os.path.join(pretrained_prefix, f"llama3_epoch_{best_epoch}")
            cbl_path = os.path.join(pretrained_prefix, f"cbl_epoch_{best_epoch}.pt")
        elif cbl_low_files:
            last_epoch = max(
                int(os.path.basename(f).replace("cbl_low_score_epoch_", "").replace(".pt", ""))
                for f in cbl_low_files
            )
            peft_path = os.path.join(pretrained_prefix, f"llama3_low_score_epoch_{last_epoch}")
            cbl_path = os.path.join(pretrained_prefix, f"cbl_low_score_epoch_{last_epoch}.pt")
        else:
            raise FileNotFoundError(f"No checkpoint files found in {pretrained_prefix}")
        print(f"Using pretrained checkpoint: peft={peft_path}, cbl={cbl_path}")
    else:
        peft_path, cbl_path = find_pretrained_checkpoint(args.pretrained_run_id, args.dataset)

    wandb.log({"pretrained_peft_path": peft_path, "pretrained_cbl_path": cbl_path})

    # ======================================================================
    # Build REFERENCE model (frozen π_ref) and POLICY model (trainable π_θ)
    # ======================================================================
    # In 'llm' reward mode the KL reference is base Llama-3-8B (AutoModelForCausalLM),
    # which is loaded later as `base_lm_model`.  The CBL-based ref_preLM / ref_cbl are
    # NOT loaded to save memory.
    ref_preLM = None
    ref_cbl = None
    if args.grpo_reward_mode != "llm":
        print("preparing reference model (frozen)...")
        ref_preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(ref_device)
        ref_preLM.load_adapter(peft_path)
        ref_preLM.eval()
        for p in ref_preLM.parameters():
            p.requires_grad = False

        if args.arch_type == "non_residual":
            ref_cbl = CBL(config, len(concept_set), tokenizer).to(ref_device)
        else:
            ref_cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(ref_device)
        ref_cbl.load_state_dict(torch.load(cbl_path, map_location=ref_device), strict=False)
        ref_cbl.eval()
        for p in ref_cbl.parameters():
            p.requires_grad = False
    else:
        print("Skipping CBL reference model load (llm reward mode uses base_lm_model as π_ref).")

    print("preparing policy model (trainable)...")
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM = get_peft_model(preLM, lora_config)
    # Load the pretrained adapter weights into the new PEFT model
    # We need to load state dict from the saved adapter
    from peft import set_peft_model_state_dict
    import safetensors
    adapter_weights_path = os.path.join(peft_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(peft_path, "adapter_model.bin")
        adapter_state = torch.load(adapter_weights_path, map_location=device)
    else:
        from safetensors.torch import load_file
        adapter_state = load_file(adapter_weights_path, device=str(device))
    set_peft_model_state_dict(preLM, adapter_state)
    
    preLM.print_trainable_parameters()
    lora_layers = filter(lambda p: p.requires_grad, preLM.parameters())
    opt_prelm = torch.optim.Adam(lora_layers, lr=args.grpo_lr)
    
    if args.arch_type == "non_residual":
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device), strict=False)
    opt_cbl = torch.optim.Adam(cbl.parameters(), lr=args.grpo_lr)

    print("preparing classifier")
    total_params = sum(p.numel() for p in preLM.parameters())
    trainable_params = sum(p.numel() for p in preLM.parameters() if p.requires_grad)
    cbl_params = sum(p.numel() for p in cbl.parameters())
    trainable_params += cbl_params
    total_params += cbl_params
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} = {trainable_params/total_params:.4f} of total")
    wandb.log({"trainable_parameters": trainable_params, "trainable_ratio": trainable_params/total_params})

    # Set intervention value once (used by GRPO)
    if args.dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100

    # Load MPNET for GRPO reward scoring (used by cosine/aggressive; also for
    # steerability metrics in reward_model mode).
    from transformers import AutoTokenizer, AutoModel
    tokenizer_sim_grpo = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model_grpo = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    sim_model_grpo.eval()
    for p in sim_model_grpo.parameters():
        p.requires_grad = False
    try:
        sim_model_grpo = torch.compile(sim_model_grpo, mode="reduce-overhead")
        print("Compiled sim_model_grpo (MPNet) for reward scoring.")
    except Exception as compile_err:
        print(f"  [GRPO] Warning: torch.compile failed for sim_model_grpo, using eager mode: {compile_err}")
    
    encoded_c_grpo = tokenizer_sim_grpo(concept_set, padding=True, truncation=True, max_length=args.max_length)
    encoded_c_grpo = {k: torch.tensor(v).to(device) for k, v in encoded_c_grpo.items()}  # input_ids: (C, L_mpnet)
    concept_features_grpo = sim_model_grpo(input_ids=encoded_c_grpo["input_ids"], attention_mask=encoded_c_grpo["attention_mask"])  # last_hidden_state: (C, L_mpnet, D)
    concept_features_grpo = mean_pooling(concept_features_grpo.last_hidden_state, encoded_c_grpo["attention_mask"])  # (C, D)
    concept_features_grpo = F.normalize(concept_features_grpo, p=2, dim=1)  # (C, D) L2-normalized
    print("Loaded MPNET for GRPO similarity scoring.")

    # Load RoBERTa concept classifiers for GRPO on-the-fly ACS-style filtering
    # grpo_classifiers = []
    # roberta_tokenizer_grpo = RobertaTokenizerFast.from_pretrained('roberta-base')
    # d_grpo = args.dataset.replace('/', '_')
    # grpo_clf_paths = [d_grpo + "_classifier.pt"]
    # clf_suffixes = [s.strip() for s in args.classifier_weight_suffixes.split(',')]
    # for suffix in clf_suffixes:
    #     grpo_clf_paths.append(d_grpo + f"_classifier{suffix}.pt")

    # for clf_path in grpo_clf_paths:
    #     if not os.path.exists(clf_path):
    #         print(f"[GRPO] Warning: classifier weights not found at {clf_path}, skipping.")
    #         continue
    #     try:
    #         clf = Roberta_classifier(len(concept_set)).to(device)
    #         clf.load_state_dict(torch.load(clf_path, map_location=device))
    #         clf.eval()
    #         for p in clf.parameters():
    #             p.requires_grad = False
    #         try:
    #             clf = torch.compile(clf)
    #         except Exception as compile_err:
    #             print(f"  [GRPO] Warning: torch.compile failed for {clf_path}, using eager mode: {compile_err}")
    #         grpo_classifiers.append(clf)
    #     except Exception as e:
    #         print(f"[GRPO] Warning: failed to load classifier from {clf_path}: {e}")

    # print(f"Loaded {len(grpo_classifiers)} RoBERTa classifiers for GRPO concept filtering.")

    # ---- Base LLM (AutoModelForCausalLM) for 'llm' reward mode ----
    # This single model serves TWO roles:
    #   (a) reward judge  — generates JSON scores for all trajectories in one pass
    #   (b) KL reference  — frozen π_ref for the KL divergence penalty
    # NOTE: In `llm` reward mode we keep the base LLM resident on `--rm_device`
    # (and optionally a separate KL reference copy on `--ref_device`).
    base_lm_model = None
    base_lm_ref_model = None
    base_lm_tokenizer = None
    if args.grpo_reward_mode == "llm":
        print("Loading base Llama-3-8B (AutoModelForCausalLM) for LLM reward + KL reference...")
        from transformers import AutoModelForCausalLM
        base_lm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
        base_lm_tokenizer.pad_token = base_lm_tokenizer.eos_token
        base_lm_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16
        )
        base_lm_model.eval()
        for p in base_lm_model.parameters():
            p.requires_grad = False
        base_lm_model.to(reward_device)
        print(f"base_lm_model loaded on {reward_device} (kept resident; no CPU offload).")

        # Optional separate copy for KL reference if ref_device differs.
        # This is intentional: it lets you place reward judge and pi_ref on different GPUs.
        if ref_device == reward_device:
            base_lm_ref_model = base_lm_model
        else:
            base_lm_ref_model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16
            )
            base_lm_ref_model.eval()
            for p in base_lm_ref_model.parameters():
                p.requires_grad = False
            base_lm_ref_model.to(ref_device)
            print(f"base_lm_ref_model loaded on {ref_device} (separate copy for KL reference).")

        # When offloading between CPU/GPU, compiling the reference forward tends to recompile
        # or be device-sensitive. Keep it in eager mode for stability.
        base_lm_logits_model = None
    else:
        base_lm_logits_model = None

    # =========================================================================
    # NEW: Load Skywork reward model for 'reward_model' reward mode
    # =========================================================================
    rm_model = None
    rm_tokenizer_rm = None
    rm_idle_device = None  # set below when reward_model mode is active
    if args.grpo_reward_mode == "reward_model":
        print(f"Loading reward model: {args.rm_model_name} ...")
        from transformers import AutoModelForSequenceClassification

        rm_tokenizer_rm = AutoTokenizer.from_pretrained(args.rm_model_name)

        # Try flash_attention_2 first (requires flash-attn package); fall back to eager.
        _rm_load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            num_labels=1,
        )
        try:
            rm_model = AutoModelForSequenceClassification.from_pretrained(
                args.rm_model_name,
                attn_implementation="flash_attention_2",
                **_rm_load_kwargs,
            )
            print("  Loaded RM with flash_attention_2.")
        except Exception as fa2_err:
            print(f"  flash_attention_2 unavailable ({fa2_err}), falling back to eager attention.")
            rm_model = AutoModelForSequenceClassification.from_pretrained(
                args.rm_model_name,
                **_rm_load_kwargs,
            )

        rm_model.eval()
        for p in rm_model.parameters():
            p.requires_grad = False

        # Keep the RM resident on its configured device.
        rm_idle_device = torch.device(args.rm_device)
        rm_model.to(rm_idle_device)
        print(f"  RM device: {rm_idle_device} (kept resident)")
        print(f"  rm_criteria_mode = {args.rm_criteria_mode}")
        print(f"  rm_batch_size    = {args.rm_batch_size} (0 = all at once)")

    print("start GRPO training...")
    d_name = args.dataset.replace('/', '_')
    prefix = "./"
    prefix += "./from_pretained_llama3_lora_grpo_" + run_name
    prefix += "/"
    prefix += d_name
    prefix += "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    model_name = "llama3"
    cbl_name = "cbl"

    # ======= DEBUG: Test batched generate + reward scoring =======
    def _debug_generate_and_score(label, intervene, target_concept_idx):
        """Generate num_test trajectories, score with RoBERTa ensemble, compute advantages."""
        print("=" * 60)
        print(f"DEBUG: {label}")
        print("=" * 60)
        with torch.no_grad():
            batch_ids, _ = cbl.generate_batch(
                test_input, preLM, num_samples=num_test,
                intervene=intervene, length=50
            )
        decoded_texts = []
        for g in range(num_test):
            tokens = batch_ids[g][~torch.isin(batch_ids[g], special_tokens)]
            text = tokenizer.decode(tokens)
            decoded_texts.append(text)

        # Score with MPNET for each concept
        rewards_per_concept = {}
        non_empty = [g for g, t in enumerate(decoded_texts) if len(t.strip()) > 0]
        if non_empty:
            non_empty_texts = [decoded_texts[g] for g in non_empty]
            generated_c_grpo = tokenizer_sim_grpo(non_empty_texts, return_tensors='pt', truncation=True, max_length=args.max_length, padding=True).to(device)
            generated_features_grpo = sim_model_grpo(input_ids=generated_c_grpo["input_ids"], attention_mask=generated_c_grpo["attention_mask"])
            generated_features_grpo = mean_pooling(generated_features_grpo.last_hidden_state, generated_c_grpo["attention_mask"])
            generated_features_grpo = F.normalize(generated_features_grpo, p=2, dim=1)
            sims_grpo = generated_features_grpo @ concept_features_grpo.T
            
            for c_idx in range(len(concept_set)):
                rewards = [0.0] * num_test
                v_target_grpo = [0] * len(concept_set)
                v_target_grpo[c_idx] = 1.0
                v_tensor_grpo = torch.tensor(v_target_grpo).to(device).unsqueeze(0).expand(len(non_empty_texts), -1)
                
                for idx_ne, g in enumerate(non_empty):
                    rewards[g] = cos_sim_cubed(sims_grpo[idx_ne:idx_ne+1], v_tensor_grpo[idx_ne:idx_ne+1].float()).item()
                rewards_per_concept[c_idx] = rewards
        else:
            for c_idx in range(len(concept_set)):
                rewards_per_concept[c_idx] = [0.0] * num_test

        # Print trajectories with rewards
        for g in range(num_test):
            reward_str = ", ".join([f"{concept_set[c]}={rewards_per_concept[c][g]:.3f}" for c in range(len(concept_set))])
            print(f"  [{g+1}/{num_test}] ({len(decoded_texts[g].split())} words): {decoded_texts[g][:200]}")
            print(f"    Rewards: {reward_str}")

        # Compute advantages for the target concept
        rewards_t = torch.tensor(rewards_per_concept[target_concept_idx], device=device, dtype=torch.float32)
        if rewards_t.std() > 1e-8:
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards_t)
        advantages = advantages.clamp(-args.grpo_clip_advantage, args.grpo_clip_advantage)
        print(f"  Target concept: {concept_set[target_concept_idx]} (idx={target_concept_idx})")
        print(f"  Rewards:    {[round(r, 4) for r in rewards_per_concept[target_concept_idx]]}")
        print(f"  Mean reward: {rewards_t.mean().item():.4f}, Std: {rewards_t.std().item():.4f}")
        print(f"  Advantages: {[round(a, 4) for a in advantages.tolist()]}")
        print(f"  Non-zero advantages: {(advantages.abs() > 1e-8).sum().item()}/{num_test}")
        print("=" * 60)

    if args.DEBUG:
        test_input = torch.tensor([tokenizer.encode("")]).to(device)
        num_test = min(args.grpo_num_trajectories, 8)
        special_tokens = torch.tensor([128000, 128001]).to(device)
        preLM.eval()
        cbl.eval()
        with torch.no_grad():
            # --- No intervention ---
            _debug_generate_and_score(
                label="generate_batch (no intervention)",
                intervene=None,
                target_concept_idx=0
            )

            # --- Intervention on first concept ---
            test_intervene_0 = [0] * len(concept_set)
            test_intervene_0[0] = intervention_value
            _debug_generate_and_score(
                label=f"generate_batch (intervene concept 0 = '{concept_set[0]}', value={intervention_value})",
                intervene=test_intervene_0,
                target_concept_idx=0
            )

            # --- Intervention on last concept ---
            last_idx = len(concept_set) - 1
            test_intervene_last = [0] * len(concept_set)
            test_intervene_last[last_idx] = intervention_value
            _debug_generate_and_score(
                label=f"generate_batch (intervene concept {last_idx} = '{concept_set[last_idx]}', value={intervention_value})",
                intervene=test_intervene_last,
                target_concept_idx=last_idx
            )
        preLM.train()
        cbl.train()

    start = time.time()
    # Base number of epochs
    epochs = CFG.epoch[args.dataset] * args.epoch_multiplier
    total_steps = epochs * len(train_loader)
    print(f"Initial epochs: {epochs}, total GRPO steps: {total_steps} (before adjustment)")
    if args.grpo_steps_per_concept > 0:
        target_total_steps = len(active_concept_indices) * args.grpo_steps_per_concept
        if total_steps < target_total_steps:
            import math
            epochs = math.ceil(target_total_steps / len(train_loader))
            total_steps = epochs * len(train_loader)
            print(
                f"Adjusting epochs to {epochs} to meet target total GRPO steps of {target_total_steps} "
                f"with {len(active_concept_indices)} active concepts. New total GRPO steps: {total_steps}."
            )

    # Track what we actually saved so post-training reload can't point at non-existent epochs.
    last_saved_epoch = 0
            
    for e in range(epochs):
        print("Epoch ", e+1, ":")
        preLM.train()
        cbl.train()
        training_losses = {
            "grpo_policy_loss": [],
            "grpo_kl_loss": [],
            "grpo_total_loss": [],
            "grpo_mean_reward": [],
            "non_zero_grpo_advantages": [],
            "concept_distill_loss": [],
            "grpo_steer_top1": [],
            "grpo_steer_top3": [],
            "grpo_steer_top5": [],
            "grpo_steer_top20": [],
            "grpo_steer_cos_sim": [],
            "grpo_reg_loss": [],
        }


        for i, (batch, batch_sim) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if e*len(train_loader) + i >= total_steps:
                print(f"Reached total step limit of {total_steps}, ending training.")
                break
            # ======= GRPO STEP =======
            # Cycle through active_concept_indices rather than all concepts
            grpo_concept_idx = active_concept_indices[(i + e * len(train_loader)) % len(active_concept_indices)]
            grpo_intervene = [0] * len(concept_set)
            grpo_intervene[grpo_concept_idx] = intervention_value

            # Phase 1: Generate G trajectories in parallel & score with RoBERTa ensemble (no grad)
            gen_input = torch.tensor([tokenizer.encode("")]).to(device)
            special_tokens_mask = torch.tensor([128000, 128001]).to(device)

            preLM.eval()
            cbl.eval()
            with torch.no_grad():
                # Batched generation — all trajectories at once
                text_ids_batch, _ = cbl.generate_batch(
                    gen_input, preLM, num_samples=args.grpo_num_trajectories,
                    intervene=grpo_intervene, length=args.grpo_gen_length
                )

                # Decode all trajectories and re-encode
                generated_seqs = []
                decoded_texts = []
                for g in range(args.grpo_num_trajectories):
                    decoded = tokenizer.decode(
                        text_ids_batch[g][~torch.isin(text_ids_batch[g], special_tokens_mask)]
                    )
                    re_encoded = torch.tensor([tokenizer.encode(decoded)]).to(device)
                    generated_seqs.append(re_encoded.detach())
                    decoded_texts.append(decoded)

                # ---- Dataset mixing ----
                # Replace the last grpo_mix_k trajectories with real training texts from the
                # current batch so the model also receives gradient signal on authentic text.
                if args.grpo_mix_dataset and args.grpo_mix_k > 0:
                    mix_k = min(args.grpo_mix_k, args.grpo_num_trajectories, batch["input_ids"].shape[0])
                    # Decode raw training ids (strip padding & special tokens)
                    for mix_i in range(mix_k):
                        real_ids = batch["input_ids"][mix_i].to(device)  # move to GPU
                        # Remove padding (id == pad_token_id) and EOS / BOS special tokens
                        real_ids_clean = real_ids[
                            (real_ids != tokenizer.pad_token_id) &
                            ~torch.isin(real_ids, special_tokens_mask)
                        ]
                        real_text = tokenizer.decode(real_ids_clean, skip_special_tokens=True)
                        replace_idx = args.grpo_num_trajectories - mix_k + mix_i
                        decoded_texts[replace_idx] = real_text
                        re_encoded = torch.tensor([tokenizer.encode(real_text)]).to(device)
                        generated_seqs[replace_idx] = re_encoded.detach()

                # Batched reward scoring with MPNET
                grpo_rewards = [0.0] * args.grpo_num_trajectories
                non_empty_indices = [g for g, t in enumerate(decoded_texts) if len(t.strip()) > 0]
                if non_empty_indices:
                    non_empty_texts = [decoded_texts[g] for g in non_empty_indices]

                    if args.grpo_reward_mode == "llm":
                        # ---- LLM-as-judge reward ----
                        cuda_gc()
                        # Print prompt + output for the first 3 steps to verify parsing
                        _llm_debug = args.DEBUG
                        llm_rewards, llm_raw_scores = compute_llm_rewards_batch(
                            base_lm_model, base_lm_tokenizer,
                            non_empty_texts,
                            concept_name=concept_set[grpo_concept_idx],
                            device=reward_device,
                            max_text_len=args.grpo_llm_max_text_len,
                            max_new_tokens=512,
                            debug=_llm_debug,
                        )
                        cuda_gc()
                        for rank, g in enumerate(non_empty_indices):
                            grpo_rewards[g] = llm_rewards[rank]

                        # Free reward intermediates ASAP
                        del llm_rewards

                        # Log parsed criteria scores for the highest-reward trajectory.
                        # Important: `llm_raw_scores` is aligned to `non_empty_texts` (ranked 0..N-1),
                        # while `grpo_rewards` is indexed by the original trajectory id `g` (0..G-1).
                        # Pick the best trajectory using a simple loop (clear + avoids rank/index confusion).
                        best_rank = 0
                        best_g = non_empty_indices[0]
                        best_reward = grpo_rewards[best_g]
                        for rank, g in enumerate(non_empty_indices):
                            r = grpo_rewards[g]
                            if r > best_reward:
                                best_reward = r
                                best_rank = rank
                                best_g = g
                        best_raw = llm_raw_scores[best_rank] or {}
                        wandb.log({
                            "llm_reward_best_concept_relevance": best_raw.get("concept_relevance", 0.0),
                            "llm_reward_best_coherence": best_raw.get("coherence", 0.0),
                            "llm_reward_best_grammar": best_raw.get("grammar", 0.0),
                        })

                        del llm_raw_scores, best_raw
                        cuda_gc()

                        # Steerability metrics: not applicable without MPNet sims — append 0 placeholders
                        training_losses["grpo_steer_top1"].append(0.0)
                        training_losses["grpo_steer_top3"].append(0.0)
                        training_losses["grpo_steer_top5"].append(0.0)
                        training_losses["grpo_steer_top20"].append(0.0)
                        training_losses["grpo_steer_cos_sim"].append(float(np.mean([grpo_rewards[g] for g in non_empty_indices])))

                    elif args.grpo_reward_mode == "reward_model":
                        # ---- MPNet + (optional RoBERTa) reward (cosine / aggressive) ----
                        cuda_gc()

                        # RM stays resident on rm_idle_device (args.rm_device)

                        _rm_debug = args.DEBUG and (e == 0) and (i < 2)  # only debug the first 2 batches of the first epoch
                        rm_rewards, rm_raw_scores = compute_reward_model_scores(
                            rm_model=rm_model,
                            rm_tokenizer=rm_tokenizer_rm,
                            texts=non_empty_texts,
                            concept_name=concept_set[grpo_concept_idx],
                            device=rm_idle_device,
                            criteria_mode=args.rm_criteria_mode,
                            rm_batch_size=args.rm_batch_size,
                            max_text_len=args.rm_max_text_len,
                            debug=_rm_debug,
                        )
                        cuda_gc()

                        # MPNet cosine — always computed for steerability metrics;
                        # also used as the steerability factor in separate_hybrid.
                        gen_c_rm = tokenizer_sim_grpo(
                            non_empty_texts, return_tensors='pt',
                            truncation=True, max_length=args.max_length, padding=True
                        ).to(device)
                        gen_feats_rm = sim_model_grpo(
                            input_ids=gen_c_rm["input_ids"],
                            attention_mask=gen_c_rm["attention_mask"]
                        )
                        gen_feats_rm = mean_pooling(gen_feats_rm.last_hidden_state, gen_c_rm["attention_mask"])
                        gen_feats_rm = F.normalize(gen_feats_rm, p=2, dim=1)
                        sims_rm = gen_feats_rm @ concept_features_grpo.T  # (N_non_empty, C)

                        v_target_rm = torch.zeros(len(non_empty_texts), len(concept_set), device=device)
                        v_target_rm[:, grpo_concept_idx] = 1.0
                        cosine_rewards_rm = cos_sim_cubed(sims_rm, v_target_rm.float(), reduce=False)  # (N_non_empty,)

                        # Steerability metrics from MPNet (shared across all sub-modes)
                        sorted_indices_rm = torch.argsort(sims_rm, dim=1, descending=True)
                        steer_top1  = (sorted_indices_rm[:, 0]   == grpo_concept_idx).float().mean().item()
                        steer_top3  = (sorted_indices_rm[:, :3]  == grpo_concept_idx).any(dim=1).float().mean().item()
                        steer_top5  = (sorted_indices_rm[:, :5]  == grpo_concept_idx).any(dim=1).float().mean().item()
                        steer_top20 = (sorted_indices_rm[:, :20] == grpo_concept_idx).any(dim=1).float().mean().item()
                        steer_cos_mean_rm = cosine_rewards_rm.mean().item()

                        training_losses["grpo_steer_top1"].append(steer_top1)
                        training_losses["grpo_steer_top3"].append(steer_top3)
                        training_losses["grpo_steer_top5"].append(steer_top5)
                        training_losses["grpo_steer_top20"].append(steer_top20)
                        training_losses["grpo_steer_cos_sim"].append(steer_cos_mean_rm)

                        # Assemble final per-trajectory rewards
                        for rank, g in enumerate(non_empty_indices):
                            if args.rm_criteria_mode == "separate_hybrid":
                                # grammar sigmoid (from RM) × cosine steerability (from MPNet)
                                grpo_rewards[g] = rm_rewards[rank] * cosine_rewards_rm[rank].item()
                            else:
                                # separate or together: RM score is already the full reward
                                grpo_rewards[g] = rm_rewards[rank]

                        # Log RM scores for the best trajectory
                        best_rm_rank = int(np.argmax([grpo_rewards[g] for g in non_empty_indices]))
                        best_rm_raw  = rm_raw_scores[best_rm_rank] or {}
                        rm_log = {"rm_reward_best_score": best_rm_raw.get("rm_score", 0.0)}
                        if args.rm_criteria_mode in ("separate", "separate_hybrid"):
                            rm_log["rm_reward_best_grammar"]   = best_rm_raw.get("grammar_score", 0.0)
                        if args.rm_criteria_mode == "separate":
                            rm_log["rm_reward_best_relevance"] = best_rm_raw.get("relevance_score", 0.0)
                        if args.rm_criteria_mode == "separate_hybrid":
                            rm_log["rm_reward_best_cosine"] = cosine_rewards_rm[best_rm_rank].item()
                        wandb.log(rm_log)

                        del gen_c_rm, gen_feats_rm, sims_rm, v_target_rm, sorted_indices_rm
                        del cosine_rewards_rm, rm_rewards, rm_raw_scores, best_rm_raw
                        cuda_gc()
                    # =================================================================

                    else:
                        # ---- MPNet cosine / aggressive reward ----
                        cuda_gc()
                        generated_c_grpo = tokenizer_sim_grpo(non_empty_texts, return_tensors='pt', truncation=True, max_length=args.max_length, padding=True).to(device)
                        generated_features_grpo = sim_model_grpo(input_ids=generated_c_grpo["input_ids"], attention_mask=generated_c_grpo["attention_mask"])
                        generated_features_grpo = mean_pooling(generated_features_grpo.last_hidden_state, generated_c_grpo["attention_mask"])
                        generated_features_grpo = F.normalize(generated_features_grpo, p=2, dim=1)

                        sims_grpo = generated_features_grpo @ concept_features_grpo.T
                        v_target_grpo = [0] * len(concept_set)
                        v_target_grpo[grpo_concept_idx] = 1.0
                        v_tensor_grpo = torch.tensor(v_target_grpo).to(device).unsqueeze(0).expand(len(non_empty_texts), -1)

                        if args.grpo_subset_renorm:
                            K = min(args.grpo_subset_topk, sims_grpo.size(1))
                            sims_mod = torch.zeros_like(sims_grpo)
                            for row_idx in range(sims_grpo.size(0)):
                                sims_vec = sims_grpo[row_idx]
                                top_vals, top_idx = torch.topk(sims_vec, K)
                                # Ensure target concept is included in the subset
                                if grpo_concept_idx not in top_idx:
                                    min_pos = torch.argmin(top_vals)
                                    top_idx[min_pos] = grpo_concept_idx
                                    top_vals[min_pos] = sims_vec[grpo_concept_idx]
                                # L2-normalize the subset to reduce scale issues
                                top_vals = top_vals / (top_vals.norm(p=2) + 1e-8)
                                sims_mod[row_idx, top_idx] = top_vals
                            sims_grpo = sims_mod

                        # DEBUG: confirm shapes for GRPO reward computation
                        if args.DEBUG and e == 0 and i == 0:
                            print("[GRPO DEBUG] generated_c_grpo[input_ids] shape:", generated_c_grpo["input_ids"].shape)
                            print("[GRPO DEBUG] generated_features_grpo shape:", generated_features_grpo.shape)
                            print("[GRPO DEBUG] concept_features_grpo shape:", concept_features_grpo.shape)
                            print("[GRPO DEBUG] sims_grpo shape:", sims_grpo.shape)
                            print("[GRPO DEBUG] v_tensor_grpo shape:", v_tensor_grpo.shape)

                        # --- Steerability metrics for this GRPO step (using MPNet sims) ---
                        # Top-k accuracy of target concept under MPNet similarities
                        sorted_indices = torch.argsort(sims_grpo, dim=1, descending=True)  # (N_non_empty, C)
                        steer_top1 = (sorted_indices[:, 0] == grpo_concept_idx).float().mean().item()
                        steer_top3 = (sorted_indices[:, :3] == grpo_concept_idx).any(dim=1).float().mean().item()
                        steer_top5 = (sorted_indices[:, :5] == grpo_concept_idx).any(dim=1).float().mean().item()
                        steer_top20 = (sorted_indices[:, :20] == grpo_concept_idx).any(dim=1).float().mean().item()
                        # Per-sample cosine-sim-cubed between sims and one-hot target (no reduction)
                        steer_cos_vals = cos_sim_cubed(sims_grpo, v_tensor_grpo.float(), reduce=False)  # (N_non_empty,)
                        steer_cos_mean = steer_cos_vals.mean().item()

                        training_losses["grpo_steer_top1"].append(steer_top1)
                        training_losses["grpo_steer_top3"].append(steer_top3)
                        training_losses["grpo_steer_top5"].append(steer_top5)
                        training_losses["grpo_steer_top20"].append(steer_top20)
                        training_losses["grpo_steer_cos_sim"].append(steer_cos_mean)

                        # Compute GRPO rewards per trajectory according to selected reward mode
                        max_k_cutoff = 20  # only trajectories where target is within top-20 get positive reward in aggressive mode
                        for idx, g in enumerate(non_empty_indices):
                            sims_vec = sims_grpo[idx]  # (C,)

                            if args.grpo_reward_mode == "cosine":
                                # Original reward: cosine-similarity-cubed between full distribution and one-hot target
                                r = cos_sim_cubed(sims_vec.unsqueeze(0), v_tensor_grpo[idx:idx+1].float()).item()
                            else:  # aggressive inverse-rank * similarity reward
                                # Rank of target concept (1-based)
                                sorted_vals, sorted_idx = torch.sort(sims_vec, descending=True)
                                # position where sorted_idx == grpo_concept_idx
                                rank_tensor = (sorted_idx == grpo_concept_idx).nonzero(as_tuple=False)
                                if rank_tensor.numel() == 0:
                                    # target concept not found (should not happen), give zero reward
                                    r = 0.0
                                else:
                                    rank = int(rank_tensor[0].item()) + 1  # 1-based rank
                                    if rank > max_k_cutoff:
                                        # Outside top-k cutoff → zero reward
                                        r = 0.0
                                    else:
                                        sim_j = float(sims_vec[grpo_concept_idx].item())
                                        sim_j = max(sim_j, 0.0)  # only reward positive similarity
                                        # Option C: inverse-rank * similarity
                                        r = (1.0 / rank) * sim_j

                            grpo_rewards[g] = r

                        # Drop MPNet intermediates ASAP
                        del generated_c_grpo, generated_features_grpo, sims_grpo, v_tensor_grpo
                        cuda_gc()

                if args.DEBUG:
                    for g in range(args.grpo_num_trajectories):
                        print(f"GRPO Trajectory {g+1}/{args.grpo_num_trajectories}:")
                        print("  Decoded text:", decoded_texts[g])
                        print("  Reward:", grpo_rewards[g])
                        print("-" * 50)
                        
                ## log the lowest and highest reward text+reward to wandb for debugging
                wandb.log({
                    "grpo_debug_lowest_reward": min(grpo_rewards),
                    "grpo_debug_highest_reward": max(grpo_rewards),
                    "grpo_debug_lowest_text": decoded_texts[grpo_rewards.index(min(grpo_rewards))],
                    "grpo_debug_highest_text": decoded_texts[grpo_rewards.index(max(grpo_rewards))],
                    "grpo_concept_idx": grpo_concept_idx,
                })

            preLM.train()
            cbl.train()

            # Phase 2: Group-relative advantages
            grpo_rewards_t = torch.tensor(grpo_rewards, device=device, dtype=torch.float32)
            if grpo_rewards_t.std() > 1e-8:
                grpo_advantages = (grpo_rewards_t - grpo_rewards_t.mean()) / (grpo_rewards_t.std() + 1e-8)
            else:
                grpo_advantages = torch.zeros_like(grpo_rewards_t)
            grpo_advantages = grpo_advantages.clamp(-args.grpo_clip_advantage, args.grpo_clip_advantage)

            # Phase 3: Recompute log-probs WITH gradient + KL against reference (batched teacher-forced)
            valid_indices = [g for g in range(args.grpo_num_trajectories)
                             if generated_seqs[g].shape[1] > 1 and grpo_advantages[g].abs().item() > 1e-8]

            if valid_indices:
                max_seq_len = max(generated_seqs[g].shape[1] for g in valid_indices)
                padded_inputs = []
                padded_targets = []
                attn_masks = []
                valid_advantages = []

                for g in valid_indices:
                    seq = generated_seqs[g]  # (1, seq_len)
                    seq_len = seq.shape[1] - 1
                    pad_len = max_seq_len - 1 - seq_len
                    seq_input = seq[:, :-1]
                    seq_target = seq[:, 1:]
                    if pad_len > 0:
                        seq_input = F.pad(seq_input, (0, pad_len), value=tokenizer.pad_token_id)
                        seq_target = F.pad(seq_target, (0, pad_len), value=0)
                        attn = torch.cat([torch.ones(1, seq_len, device=device),
                                          torch.zeros(1, pad_len, device=device)], dim=1)
                    else:
                        attn = torch.ones(1, seq_len, device=device)
                    padded_inputs.append(seq_input)
                    padded_targets.append(seq_target)
                    attn_masks.append(attn)
                    valid_advantages.append(grpo_advantages[g])

                batch_input = torch.cat(padded_inputs, dim=0)     # (V, T)
                batch_target = torch.cat(padded_targets, dim=0)   # (V, T)
                batch_attn = torch.cat(attn_masks, dim=0).long()  # (V, T)
                valid_adv = torch.stack(valid_advantages)          # (V,)

                # --- Policy model forward (with grad) ---
                feats_g = preLM(input_ids=batch_input, attention_mask=batch_attn).last_hidden_state  # (V, T, H)
                concepts_g, unsup_g, _, _ = cbl(feats_g.float())  # concepts_g: (V, T, C), unsup_g: (V, T, H_cbl)

                intervened_g = torch.zeros(len(valid_indices), batch_input.shape[1], len(concept_set), device=device)  # (V, T, C)
                intervened_g[:, :, grpo_concept_idx] = intervention_value

                vocab_logits_g = cbl.intervene(unsup_g, intervened_g)  # (V, T, vocab_size)
                log_probs_g = F.log_softmax(vocab_logits_g, dim=-1)
                token_log_probs_g = log_probs_g.gather(2, batch_target.unsqueeze(-1)).squeeze(-1)
                # Mask out padded positions
                token_log_probs_g = token_log_probs_g * batch_attn.float()  # (V, T)
                mean_log_probs = token_log_probs_g.sum(dim=1) / batch_attn.sum(dim=1).float()  # (V,)

                # --- Reference model forward (no grad) ---
                with torch.no_grad():
                    if args.grpo_reward_mode == "llm":
                        # Use base Llama-3-8B (AutoModelForCausalLM) as KL reference.
                        # It produces vocab logits directly without any CBL intervention.
                        cuda_gc()
                        ref_llm = base_lm_ref_model if base_lm_ref_model is not None else base_lm_model
                        if ref_device != device:
                            ref_batch_input = batch_input.to(ref_device, non_blocking=True)
                            ref_batch_attn = batch_attn.to(ref_device, non_blocking=True)
                        else:
                            ref_batch_input = batch_input
                            ref_batch_attn = batch_attn
                        ref_vocab_logits_g = ref_llm(
                            input_ids=ref_batch_input, attention_mask=ref_batch_attn
                        ).logits  # (V, T, vocab_size)
                        ref_log_probs_g = F.log_softmax(ref_vocab_logits_g, dim=-1)
                        if ref_log_probs_g.device != device:
                            ref_log_probs_g = ref_log_probs_g.to(device, non_blocking=True)
                        del ref_vocab_logits_g
                        if ref_device != device:
                            del ref_batch_input, ref_batch_attn
                        cuda_gc()
                    else:
                        if ref_device != device:
                            ref_batch_input = batch_input.to(ref_device, non_blocking=True)
                            ref_batch_attn = batch_attn.to(ref_device, non_blocking=True)
                        else:
                            ref_batch_input = batch_input
                            ref_batch_attn = batch_attn

                        ref_feats_g = ref_preLM(input_ids=ref_batch_input, attention_mask=ref_batch_attn).last_hidden_state
                        ref_concepts_g, ref_unsup_g, _, _ = ref_cbl(ref_feats_g.float())

                        ref_intervened_g = torch.zeros(
                            len(valid_indices), ref_batch_input.shape[1], len(concept_set), device=ref_device
                        )
                        ref_intervened_g[:, :, grpo_concept_idx] = intervention_value

                        ref_vocab_logits_g = ref_cbl.intervene(ref_unsup_g, ref_intervened_g)
                        ref_log_probs_g = F.log_softmax(ref_vocab_logits_g, dim=-1)
                        if ref_log_probs_g.device != device:
                            ref_log_probs_g = ref_log_probs_g.to(device, non_blocking=True)

                        # Free reference intermediates (keep ref_concepts_g for optional distillation)
                        del ref_feats_g, ref_unsup_g, ref_intervened_g, ref_vocab_logits_g
                        if ref_device != device:
                            del ref_batch_input, ref_batch_attn
                        cuda_gc()
                    
                # --- KL divergence: full-vocab KL(π_θ || π_ref) per token, always ≥ 0 ---
                # KL(π_θ || π_ref) = Σ_a π_θ(a) [log π_θ(a) - log π_ref(a)]
                policy_probs_g = F.softmax(vocab_logits_g, dim=-1)  # (V, seq_len, vocab_size)
                kl_per_token = (policy_probs_g * (log_probs_g - ref_log_probs_g)).sum(dim=-1)  # (V, seq_len)
                kl_per_token = kl_per_token * batch_attn.float()
                kl_per_seq = kl_per_token.sum(dim=1) / batch_attn.sum(dim=1).float()
                kl_loss = kl_per_seq.mean()

                # --- Policy gradient loss ---
                policy_loss = (-valid_adv * mean_log_probs).mean()

                # --- Concept prediction distillation loss ---
                # concepts_g are ReLU'd (used as logits), ref_concepts_g → softmax → soft targets
                # Only on non-padded positions (batch_attn)
                # Note: concept distillation requires ref_cbl (not available in 'llm' reward mode).
                concept_distill_loss = torch.tensor(0.0, device=device)
                if args.concept_distill_weight > 0 and args.grpo_reward_mode != "llm":
                    concept_mask_flat = batch_attn.reshape(-1).bool()  # (V * seq_len,)
                    policy_concept_flat = concepts_g.reshape(-1, len(concept_set))  # (V * seq_len, C)
                    with torch.no_grad():
                        _ref_concepts_for_distill = ref_concepts_g
                        if _ref_concepts_for_distill.device != device:
                            _ref_concepts_for_distill = _ref_concepts_for_distill.to(device, non_blocking=True)
                        ref_concept_targets = F.softmax(_ref_concepts_for_distill.reshape(-1, len(concept_set)), dim=-1)
                    concept_distill_loss = F.cross_entropy(
                        policy_concept_flat[concept_mask_flat],
                        ref_concept_targets[concept_mask_flat]
                    )
                if args.grpo_reward_mode != "llm":
                    # Done with reference concept logits/targets
                    try:
                        del ref_concepts_g, ref_concept_targets
                    except Exception:
                        pass
                    cuda_gc()

                # --- Total GRPO loss ---
                grpo_total_loss = args.grpo_loss_weight * policy_loss + args.grpo_kl_weight * kl_loss + args.concept_distill_weight * concept_distill_loss

                # Add sparsity regularization on concept weights to encourage sparse concepts
                reg_loss = elastic_net_penalty(cbl.fc.weight[:, :len(concept_set)])
                grpo_total_loss = grpo_total_loss + reg_loss

                opt_prelm.zero_grad()
                opt_cbl.zero_grad()
                grpo_total_loss.backward()
                opt_prelm.step()
                opt_cbl.step()

                training_losses["grpo_policy_loss"].append(policy_loss.detach().cpu().numpy())
                training_losses["grpo_kl_loss"].append(kl_loss.detach().cpu().numpy())
                training_losses["concept_distill_loss"].append(concept_distill_loss.detach().cpu().numpy())
                training_losses["grpo_total_loss"].append(grpo_total_loss.detach().cpu().numpy())
                training_losses["grpo_mean_reward"].append(grpo_rewards_t.mean().item())
                training_losses["grpo_reg_loss"].append(reg_loss.detach().cpu().numpy())

                # Drop big per-step tensors (helps keep peak VRAM down)
                del feats_g, concepts_g, unsup_g, intervened_g
                del vocab_logits_g, log_probs_g, token_log_probs_g, mean_log_probs
                del ref_log_probs_g, policy_probs_g, kl_per_token, kl_per_seq
                del batch_input, batch_target, batch_attn, valid_adv
                del padded_inputs, padded_targets, attn_masks, valid_advantages
                cuda_gc()

            # Drop generation/reward temporaries for this step
            try:
                del text_ids_batch
            except Exception:
                pass
            try:
                del decoded_texts, non_empty_indices, non_empty_texts
            except Exception:
                pass
            try:
                del generated_seqs
            except Exception:
                pass
            try:
                del gen_input, special_tokens_mask
            except Exception:
                pass
            cuda_gc()
                
            training_losses["non_zero_grpo_advantages"].append((grpo_advantages.abs() > 1e-8).sum().item())
            if args.DEBUG:
                print(f"GRPO debug - rewards: {grpo_rewards}, advantages: {grpo_advantages.tolist()}")

            
            log = {}
            for key in training_losses.keys():
                if len(training_losses[key]) > 0:
                    print(f"{key}: {training_losses[key][-1]}", end=" ")
                    log[key] = training_losses[key][-1]
            # print(" | batch ", i+1, " / ", len(train_loader), end="\r")
            
            
            log["epoch"] = e + 1
            log["batch"] = i + 1
            wandb.log(log)
            
            if args.DEBUG and i >= 2:
                break
            
            
        avg_metrics = {}
        for key in training_losses.keys():
            if len(training_losses[key]) > 0:
                avg_metrics[key] = sum(training_losses[key]) / len(training_losses[key])
        print("Epoch ", e + 1, " training losses: ", avg_metrics)
        wandb.log({f"avg_{k}": avg_metrics[k] for k in avg_metrics.keys()})

        # Save after each epoch
        print("save model")
        preLM.save_pretrained(prefix + model_name + "_epoch_" + str(e + 1))
        torch.save(cbl.state_dict(), prefix + cbl_name + "_epoch_" + str(e + 1) + ".pt")
        last_saved_epoch = e + 1

        if args.DEBUG:
            break
        
        if (e+1)*len(train_loader) >= total_steps:
            print(f"Reached total step limit of {total_steps}, ending training.")
            break

    end = time.time()
    print("time of training GRPO:", (end - start) / 3600, "hours")
    
    ## delete previous models to save space
    del preLM, cbl, opt_prelm, opt_cbl
    try:
        del ref_preLM, ref_cbl
    except:
        pass
    # Free the reward model if loaded
    if rm_model is not None:
        del rm_model
    try:
        if base_lm_ref_model is not None and base_lm_ref_model is not base_lm_model:
            del base_lm_ref_model
    except Exception:
        pass
    if base_lm_model is not None:
        del base_lm_model

    cuda_gc()
    
    ## lOAD BEST MODEL AND
    # GRPO doesn't track a validation metric here; evaluate the best (latest) checkpoint we actually saved.
    best_epoch = last_saved_epoch if last_saved_epoch > 0 else epochs
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    peft_path = prefix + model_name + "_epoch_" + str(best_epoch)
    preLM.load_adapter(peft_path)
    preLM.eval()
    if args.arch_type == "non_residual":
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
    else:
        cbl = CBLResidual(config, len(concept_set), args.residual_dim, tokenizer).to(device)
    cbl.load_state_dict(torch.load(prefix + cbl_name + "_epoch_" + str(best_epoch) + ".pt", map_location=device))
    cbl.eval()
        
    
    
    
    
    ### TEST STEERABILITY AFTER TRAINING
    
    from transformers import AutoTokenizer, AutoModel
    tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    sim_model.eval()

    encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=args.max_length)
    encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}
    concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"])
    concept_features = mean_pooling(concept_features.last_hidden_state, encoded_c["attention_mask"])
    concept_features = F.normalize(concept_features, p=2, dim=1)

    if args.dataset == "dbpedia_14":
        intervention_value = 150
    else:
        intervention_value = 100
        
    text = []
    cos_sim_cubed_values = []
    softmax_values = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0
    total_evals = 0
    
    # Use batched generation for steerability evaluation: for each concept
    # generate 50 samples at once with `generate_batch`.
    num_steerability_samples = 50  # scalar
    gen_input = torch.tensor([tokenizer.encode("")]).to(device)  # (1, prompt_len)
    special_tokens_mask = torch.tensor([128000, 128001]).to(device)  # (2,)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")  # CE over classes

    # Use either the full concept set or the capped active subset for steerability testing
    steer_concept_indices = active_concept_indices if args.grpo_active_concepts_k is not None and args.grpo_active_concepts_k > 0 else range(len(concept_set))

    with torch.no_grad():
        for j in tqdm(steer_concept_indices, desc="Steerability concepts"):
            v = [0] * len(concept_set)  # (C,)
            v[j] = intervention_value

            # Batched generation: `num_steerability_samples` samples for this concept
            text_ids_batch, _ = cbl.generate_batch(
                gen_input,
                preLM,
                num_samples=num_steerability_samples,
                intervene=v,
                length=50,
            )  # (num_steerability_samples, gen_len)

            # Decode and collect texts for MPNet scoring
            decoded_texts = []  # list of length B=num_steerability_samples
            for g in range(num_steerability_samples):
                tokens = text_ids_batch[g][~torch.isin(text_ids_batch[g], special_tokens_mask)]
                decoded = tokenizer.decode(tokens)
                decoded_texts.append(decoded)
                text.append(decoded)
            
            ### print some decoded texts for debugging
            # print(f"Steerability evaluation for concept '{concept_set[j]}':")
            for idx in range(len(decoded_texts)):
                # print(f"  Sample {idx+1}: {decoded_texts[idx]}")
                wandb.log({f"steerability_sample_{concept_set[j]}_{idx+1}": decoded_texts[idx]})
            # Batched similarity scoring with MPNet
            generated_c = tokenizer_sim(
                decoded_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )  # dict: input_ids -> (B, L), attention_mask -> (B, L)
            generated_c = {k: v.to(device) for k, v in generated_c.items()}
            generated_features = sim_model(
                input_ids=generated_c["input_ids"],
                attention_mask=generated_c["attention_mask"],
            )  # last_hidden_state: (B, L, H)
            generated_features = mean_pooling(
                generated_features.last_hidden_state,
                generated_c["attention_mask"],
            )  # (B, H)
            generated_features = F.normalize(generated_features, p=2, dim=1)  # (B, H)

            sims = generated_features @ concept_features.T  # (B, C)
            v_tensor = torch.tensor(v).to(device).unsqueeze(0).expand(sims.size(0), -1)  # (B, C)

            # cos_sim_cubed per sample (no reduction)
            cos_vals = cos_sim_cubed(sims, v_tensor.float(), reduce=False)  # (B,)
            cos_sim_cubed_values.extend(cos_vals.detach().cpu().tolist())

            # Cross-entropy loss per sample w.r.t. true concept j
            targets = torch.full((sims.size(0),), j, dtype=torch.long, device=device)  # (B,)
            ce_vals = ce_loss_fn(sims, targets)  # (B,)
            softmax_values.extend(ce_vals.detach().cpu().tolist())

            # Top-k accuracy counts (vectorized over the batch)
            sorted_indices = torch.argsort(sims, dim=1, descending=True)  # (B, C)
            top1_correct += (sorted_indices[:, 0] == j).sum().item()
            top3_correct += (sorted_indices[:, :3] == j).any(dim=1).sum().item()
            top5_correct += (sorted_indices[:, :5] == j).any(dim=1).sum().item()
            top10_correct += (sorted_indices[:, :10] == j).any(dim=1).sum().item()
            top20_correct += (sorted_indices[:, :20] == j).any(dim=1).sum().item()
            total_evals += sims.size(0)

    wandb.log({
        "steerability_cos_sim_cubed": sum(cos_sim_cubed_values) / len(cos_sim_cubed_values),
        "steerability_softmax": sum(softmax_values) / len(softmax_values),
        "steerability_top1_acc": top1_correct / total_evals,
        "steerability_top3_acc": top3_correct / total_evals,
        "steerability_top5_acc": top5_correct / total_evals,
        "steerability_top10_acc": top10_correct / total_evals,
        "steerability_top20_acc": top20_correct / total_evals,
    })
    
    print(f"Steerability Top-1 Acc: {top1_correct / total_evals}")
    print(f"Steerability Top-3 Acc: {top3_correct / total_evals}")
    print(f"Steerability Top-5 Acc: {top5_correct / total_evals}")
    print(f"Steerability Top-10 Acc: {top10_correct / total_evals}")
    print(f"Steerability Top-20 Acc: {top20_correct / total_evals}")
    
    
    ### TEST CONCEPT PREDICTION AFTER TRAINING (COSINE-SIMILARITY-BASED)
    print("eval concepts (cosine similarity to MPNet labels)...")
    concept_predictions = []  # list of tensors, each (B, num_concepts)

    for batch, _ in tqdm(test_loader, total=len(test_loader)):
        # batch["input_ids"]: (B, seq_len), batch["attention_mask"]: (B, seq_len)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state  # (B, seq_len, hidden_dim)
            concepts, _, _, _ = cbl(features.float())  # concepts: (B, seq_len, num_concepts)
        pooled_concepts = eos_pooling(concepts, batch["attention_mask"])  # (B, num_concepts)
        concept_predictions.append(pooled_concepts.detach().cpu())

    # concept_predictions: (N_test, num_concepts)
    concept_predictions = torch.cat(concept_predictions, dim=0)  # (N_test, num_concepts)

    # Load test-time MPNet/ACS concept label vectors, if available
    test_sim_path = label_prefix + "/concept_labels_test.npy"
    if os.path.exists(test_sim_path):
        test_similarity_np = np.load(test_sim_path)  # (N_test, num_concepts)
        test_similarity = torch.tensor(test_similarity_np, dtype=torch.float32)  # (N_test, num_concepts)

        if test_similarity.shape != concept_predictions.shape:
            print("[WARN] Shape mismatch between concept_predictions",
                  f"{tuple(concept_predictions.shape)} and test_similarity {tuple(test_similarity.shape)}.")
            print("       Skipping cosine-similarity-based concept evaluation.")
        else:
            # Average cosine similarity over test set (same objective as training cos_sim_cubed)
            # concept_predictions: (N_test, num_concepts)
            # test_similarity:     (N_test, num_concepts)
            test_cos_sim = cos_sim_cubed(concept_predictions, test_similarity)  # scalar
            test_cos_loss = -test_cos_sim.item()

            print(f"Test concept cosine similarity (cos_sim_cubed): {test_cos_sim.item():.4f}")
            print(f"Test concept cosine loss: {test_cos_loss:.4f}")

            # --- Raw (non-cubed) cosine similarity between prediction and ACS vectors ---
            pred_norm = F.normalize(concept_predictions, p=2, dim=-1)
            label_norm = F.normalize(test_similarity, p=2, dim=-1)
            test_cos_raw = (pred_norm * label_norm).sum(dim=-1).mean().item()

            print(f"Test concept cosine similarity (raw): {test_cos_raw:.4f}")

            # --- Concept-level top-k accuracy w.r.t. ACS labels ---
            # Ground-truth concept per example: top concept from test_similarity
            # true_concepts: (N_test,)
            true_concepts = torch.argmax(test_similarity, dim=-1)  # indices in [0, num_concepts-1]

            # Predicted ranking over concepts per example
            # pred_sorted: (N_test, num_concepts), each row is concept indices sorted by predicted score
            pred_sorted = torch.argsort(concept_predictions, dim=-1, descending=True)

            topk_list = [1, 3, 5, 10, 20]
            topk_hits = {k: 0 for k in topk_list}
            topk_iou_sums = {k: 0.0 for k in topk_list}
            total = concept_predictions.size(0)

            for i in range(total):
                gt_idx = true_concepts[i].item()
                row = pred_sorted[i]
                # For IoU, also collect top-k sets from GT similarity and predictions
                for k in topk_list:
                    k_clipped = min(k, row.size(0))
                    gt_topk = torch.topk(test_similarity[i], k=k_clipped, dim=-1).indices.tolist()
                    pred_topk = row[:k_clipped].tolist()
                    gt_set = set(gt_topk)
                    pred_set = set(pred_topk)
                    inter = len(gt_set & pred_set)
                    union = len(gt_set | pred_set)
                    if union > 0:
                        topk_iou_sums[k] += inter / union
                for k in topk_list:
                    if k <= row.size(0) and gt_idx in row[:k].tolist():
                        topk_hits[k] += 1

            topk_acc = {f"test_concept_top{k}_acc": topk_hits[k] / total for k in topk_list}
            topk_iou = {f"test_concept_top{k}_iou": topk_iou_sums[k] / total for k in topk_list}

            for k in topk_list:
                print(f"Test concept Top-{k} Acc (w.r.t. ACS top concept): {topk_acc[f'test_concept_top{k}_acc']:.4f}")
                print(f"Test concept Top-{k} IoU (GT vs pred top-k concepts): {topk_iou[f'test_concept_top{k}_iou']:.4f}")

            wandb.log({
                "test_concept_cosine_similarity": float(test_cos_sim.item()),
                "test_concept_cosine_loss": float(test_cos_loss),
                "test_concept_cosine_raw": float(test_cos_raw),
                **topk_acc,
                **topk_iou,
            })
    else:
        print(f"[WARN] {test_sim_path} not found. Skipping cosine-similarity-based concept evaluation.")
    
    
    
    #### TEST WEIGHT
    print("Top tokens for each concept neuron:")
    w = cbl.fc.weight.data[:, :len(concept_set)].T
    for i in tqdm(range(len(concept_set))):
        top_values, top_ids = torch.topk(w[i], k=10)
        print("Neuron: ", concept_set[i])
        print("Top 10 tokens with highest weight:")
        for j in range(10):
            print("Neuron:", concept_set[i], "[",round(float(top_values.detach().cpu()[j]), 3), "]", tokenizer.decode(top_ids[j]))

    print("Sparsity of concept weight matrix:")
    print((w > 1e-6).count_nonzero() / w.numel())
    wandb.log({"concept_weight_sparsity": (w > 1e-6).count_nonzero() / w.numel()})
    
    
    
    #### TEST PERPLEXITY AFTER TRAINING
    print("Test perplexity after training:")
    set_seed(args.seed)
    
    pred = []
    c = 0
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_ids = torch.tensor([tokenizer.encode("")]).to(device)
    for i in tqdm(range(100)):
        print("example", str(i), end="\r")
        with torch.no_grad():
            text_ids, _ = cbl.generate(input_ids, preLM)
            pred.append(tokenizer.decode(text_ids[0], skip_special_tokens=True ))
            if len(pred[-1].split()) > 30:
                continue
            c += 1
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
    del cbl
    gc.collect()
    torch.cuda.empty_cache()

    print("Perplexity: (under 30 tokens)")
    if c > 0:
        perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
        print(perplexity)
        wandb.log({"perplexity_under_30_tokens": perplexity})
    else:
        print("No generated texts under 30 tokens to compute perplexity.")
        wandb.log({"perplexity_under_30_tokens": None})
    
    print("Now for all tokens:")
    perplexity = evaluate.load("perplexity", module_type="metric")
    for p in pred:
        perplexity.add_batch(predictions=[p])
    perplexity = perplexity.compute(model_id='meta-llama/Meta-Llama-3-8B', max_length=100)['mean_perplexity']
    print(perplexity)
    wandb.log({"perplexity_all_tokens": perplexity})
        
    