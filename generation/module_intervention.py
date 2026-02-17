import torch
from torch import nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model, GPT2TokenizerFast, RobertaModel
import torch.nn.functional as F
from utils import top_k_top_p_filtering
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.generation import GenerationMixin

class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class LlamaUNetBottleneck(nn.Module):
    ## TODO: Improve this skipping. See if good balance be found
    def __init__(self, hidden_size=4096, intermediate_sizes=[1024, 512, 256], concept_size=14, skip_dropout=0.3, gate=True):
        super().__init__()
        self.act = nn.SiLU()
        self.concept_size = concept_size
        self.encoders = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        self.skip_dropout = nn.Dropout(p=skip_dropout)
        self.gate = gate
        
        current_dim = hidden_size
        for next_dim in intermediate_sizes:
            self.encoders.append(nn.Linear(current_dim, next_dim))
            self.enc_norms.append(nn.LayerNorm(next_dim))
            current_dim = next_dim
            
        # Final bridge to the concept bottleneck
        self.concept_enc = nn.Linear(current_dim, concept_size)
        self.concept_norm = nn.LayerNorm(concept_size)
        
        # --- Expansive Path (Decoder Stack) ---
        self.decoders = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        self.gates = nn.ParameterList() # Gates for hierarchical skip connections
        
        # Reverse the intermediate sizes for the upward path
        reversed_sizes = list(reversed(intermediate_sizes))
        
        current_dim = concept_size
        for next_dim in reversed_sizes:
            self.decoders.append(nn.Linear(current_dim, next_dim))
            self.dec_norms.append(nn.LayerNorm(next_dim))
            self.gates.append(nn.Parameter(torch.zeros(1)))
            current_dim = next_dim
            
        self.final_proj = nn.Linear(current_dim, hidden_size)
        self.final_gate = nn.Parameter(torch.zeros(1))
        

    def encode(self, x):
        skips = []
        h = x
        for enc, norm in zip(self.encoders, self.enc_norms):
            h = norm(self.act(enc(h)))
            skips.append(h)
            
        concepts = self.concept_norm(self.concept_enc(h))
        return concepts, skips

    def decode(self, x_original, concepts, skips):
        skips = skips[::-1] 
        h = concepts
        
        for i, (dec, norm, gate) in enumerate(zip(self.decoders, self.dec_norms, self.gates)):
            h_up = self.act(dec(h))
            
            skip_connection = self.skip_dropout(skips[i])
            
            if self.gate:
                h = norm(h_up + (gate * skip_connection))
            else:
                h = norm(h_up + skip_connection)
            
        x_reconstructed = self.final_proj(h)
        
        if not self.gate:
            output = x_original + x_reconstructed
        else:
            output = x_original + (self.final_gate * x_reconstructed)
        
        return output

    def forward(self, x):
        concepts, skips = self.encode(x)
        return self.decode(x, concepts, skips)
        
    
class CustomLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.peft_forward = False
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        print("CONFIG", config.num_hidden_layers)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.intervene = None
        self.skip_first_half = False
        self.post_init()

        # freeze everything except new layer
        for p in self.parameters():
            p.requires_grad = False

    def create_intermediate(self, where, concept_size, intermediate_sizes=[1024, 512, 256], debug=False, skip_dropout=0.0, gate=True):
        self.where = where
        self.debug = debug
        self.concept_size = concept_size
        hidden_size = self.config.hidden_size
        # your intermediate layer
        if debug:
            self.intermediate = nn.Identity()
        else:
            self.intermediate = LlamaUNetBottleneck(hidden_size=hidden_size,
                                                    intermediate_sizes=intermediate_sizes, #4096/8 = 512
                                                    concept_size=concept_size,
                                                    skip_dropout=skip_dropout,
                                                    gate=gate
                                                    )

        self.add_module("intermediate", self.intermediate)
        for p in self.intermediate.parameters():
            p.requires_grad = True

    def firsthalf_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # print("Gradient tracking in first half forward:", torch.is_grad_enabled())
        with torch.set_grad_enabled(self.peft_forward): # If using PEFT, we want gradients to flow through the first half to the intermediate module. Otherwise, we keep it in no_grad mode.
            # print("PEFT forward mode:", self.peft_forward, )
            if inputs_embeds is None:
                inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position: torch.Tensor = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)


            for idx, decoder_layer in enumerate(self.layers[:self.where]):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )


        if self.debug:
            print(f"[LOG] TRIGGERED INTERMEDIATE AT LAYER {self.where}")
            print(hidden_states.shape)

        # hidden_states, concepts = self.intermediate(hidden_states)
        
        
        return hidden_states, causal_mask, position_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        multiple_intervene = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        
        # Trainable layer
        hidden_states, causal_mask, position_embeddings = \
        self.firsthalf_forward(input_ids,
                               attention_mask,
                               position_ids,
                               past_key_values,
                               inputs_embeds,
                               cache_position,
                               use_cache)
        
        concepts, skips = self.intermediate.encode(hidden_states)
        
        if self.intervene is not None:
            concepts = amplify_intervention(
                    concepts,
                    self.intervene,
                    margin=self.intervention_margin,
                    spread=self.intervention_spread,
                )

        hidden_states = self.intermediate.decode(hidden_states, concepts, skips)
        
        output = self.secondhalf_forward(hidden_states,
        causal_mask,
        position_embeddings,
        position_ids,
        past_key_values,
        cache_position,
        **kwargs
        )
        
        output.concepts = concepts ### TODO: IS THIS THE BEST WAY TO SHARE CONCEPTS?
        
        return output
        
    
    def secondhalf_forward(
        self,
        hidden_states,
        causal_mask,
        position_embeddings,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        
        
        # Subsequent layers (must track gradients to pass them backward)
        for idx in range(self.where, self.config.num_hidden_layers):
            decoder_layer = self.layers[idx]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

def amplify_intervention(original_concepts, labels, margin=10.0, spread=2.0):
    """
    Applies per-concept intervention while preserving original values for
    padding or untargeted tokens.

    Args:
        original_concepts : Tensor (B, T, C)
        labels            : Tensor (B, T, 1)
                            values in {1, 0, -100}
        margin            : float, distance added beyond observed extrema
        spread            : float, minimum absolute magnitude for extrema

    Returns:
        intervened_concepts : Tensor (B, T, C)
    """

    # ---- 1. Compute per-concept extrema (no gradients) ----
    with torch.no_grad():
        # per-concept max/min across batch and time
        v_max = original_concepts.amax(dim=(0, 1), keepdim=True)
        v_min = original_concepts.amin(dim=(0, 1), keepdim=True)

        # ensure minimum activation spread
        v_max = torch.clamp(v_max, min=spread)
        v_min = torch.clamp(v_min, max=-spread)

        high_target = v_max + margin   # (1, 1, C)
        low_target  = v_min - margin   # (1, 1, C)

    # ---- 2. Start from original concepts ----
    intervened = original_concepts.clone().detach()

    # ---- 3. Apply intervention only where labels specify ----
    pos_mask = (labels == 1)   # (B, T, 1)
    neg_mask = (labels == 0)   # (B, T, 1)

    intervened = torch.where(pos_mask, high_target, intervened)
    intervened = torch.where(neg_mask, low_target, intervened)

    # ---- 4. Return detached tensor (intervention treated as constant) ----
    return intervened.detach()




class CustomLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, model=None):
        super().__init__(config)
        self.model = CustomLlamaModel(config) if model is None else model 
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        for p in self.lm_head.parameters():
            p.requires_grad = False

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def compute_training_losses(
    batch,
    preLM:CustomLlamaModel,
    preLM_generator:CustomLlamaForCausalLM,
    reconstr_crit,
    CSE_crit,
    concept_label,
    word_label,
    concept_set,
    config,
    args,
):
    """
    Computes all training losses for the concept-intervention model.

    Expected tensor shapes:
        input_ids          : (B, T)
        attention_mask     : (B, T)
        concept_label      : (B, T-1)
        word_label         : (B, T-1)
    """

    # ============================================================
    # 1. FIRST HALF FORWARD (encoder side)
    # ============================================================

    # pre_hidden_states : (B, T, H)
    # concepts          : (B, T, C)
    # skips             : model dependent
    # causal_mask       : (B, 1, T, T) or compatible
    # position_embeddings : (B, T, H)
    pre_hidden_states, causal_mask, position_embeddings = (
        preLM.firsthalf_forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    )
    pre_hidden_states = pre_hidden_states.detach() 
    ## technically pre_hidden_states wouldn't need detach since firsthalf is no_grad 

    # Encode into concept space
    # concepts : (B, T, C)
    concepts, skips = preLM.intermediate.encode(pre_hidden_states)

    # Decode back to hidden space
    # post_hidden_states : (B, T, H)
    post_hidden_states = preLM.intermediate.decode(
        pre_hidden_states, concepts, skips
    )

    # ============================================================
    # 2. CONCEPT LOSS
    # ============================================================

# Check magnitude of concepts
    # print(f"Concepts Mean: {concepts.mean().item()}, Max: {concepts.max().item()}")
    # print(f"Concepts GT Mean: {concept_label.float().mean().item()}, Max: {concept_label.float().max().item()}")
    # concepts[:, :-1, :] : (B, T-1, C)
    # reshape -> (B*(T-1), C)
    # concept_label -> (B*(T-1))
    concept_loss = CSE_crit(
        concepts[:, :-1, :].reshape(-1, len(concept_set)),
        concept_label.reshape(-1),
    )

    # ============================================================
    # 3. RECONSTRUCTION LOSSES
    # ============================================================

    # ------------------------------------------------------------
    # 3.1 Encoder-decoder reconstruction
    # ------------------------------------------------------------
    # both tensors : (B, T-1, H)
    intermediate_reconstruction_loss = reconstr_crit(
        post_hidden_states[:, :-1, :],
        pre_hidden_states[:, :-1, :].detach(),
    )

    # ------------------------------------------------------------
    # 3.2 Generation reconstruction (KL divergence)
    # ------------------------------------------------------------

    # hidden states after decoder
    # (B, T, H)
    vocabs_reconstr = preLM.secondhalf_forward(
        post_hidden_states,
        causal_mask,
        position_embeddings,
    ).last_hidden_state

    # original hidden states
    # (B, T, H)
    vocabs_org = preLM.secondhalf_forward(
        pre_hidden_states,
        causal_mask,
        position_embeddings,
    ).last_hidden_state

    vocabs_org = vocabs_org.detach()

    # logits -> log probs
    # (B, T, V)
    log_probs_org = F.log_softmax(
        preLM_generator.lm_head(vocabs_org), dim=-1
    )
    log_probs_reconstr = F.log_softmax(
        preLM_generator.lm_head(vocabs_reconstr), dim=-1
    )

    # KL over vocabulary dimension
    # input shapes : (B, T-1, V)
    generation_reconstruction_loss = F.kl_div(
        log_probs_reconstr[:, :-1, :],
        log_probs_org[:, :-1, :],
        log_target=True,
        reduction="batchmean",
    )

    # ============================================================
    # 4. INTERVENTION LOSSES
    # ============================================================

    # ------------------------------------------------------------
    # 4.0 Prepare labels for intervention
    # ------------------------------------------------------------

    # concept_label_expanded : (B, T-1, 1)
    concept_label_expanded = concept_label.unsqueeze(-1)

    # pad last timestep so shapes match concepts
    # pad : (B, 1, 1)
    pad = torch.full(
        (concept_label_expanded.size(0), 1, 1),
        -100,
        device=concept_label_expanded.device,
        dtype=concept_label_expanded.dtype,
    )

    # (B, T, 1)
    concept_label_expanded = torch.cat(
        [concept_label_expanded, pad], dim=1
    )

    # ------------------------------------------------------------
    # 4.1 Apply intervention in concept space
    # ------------------------------------------------------------

    # intervened_concept : (B, T, C)
    intervened_concept = amplify_intervention(
        concepts,
        concept_label_expanded,
        margin=args.intervention_margin,
        spread=args.intervention_spread,
    )

    # ------------------------------------------------------------
    # 4.2 Cyclic consistency loss
    # ------------------------------------------------------------

    # decode intervened concepts
    # (B, T, H)
    post_hidden_states_interven = preLM.intermediate.decode(
        pre_hidden_states,
        intervened_concept,
        skips,
    )

    # re-encode
    # cyclic_concepts : (B, T, C)
    cyclic_concepts, skips = preLM.intermediate.encode(
        post_hidden_states_interven
    )

    # MSE between intervened and re-encoded concepts
    # both : (B, T-1, C)
    cyclic_concept_loss = reconstr_crit(
        cyclic_concepts[:, :-1, :],
        intervened_concept[:, :-1, :],
    )

    # ------------------------------------------------------------
    # 4.3 Intervention generation loss
    # ------------------------------------------------------------

    # hidden states after intervention
    # (B, T, H)
    vocabs_intervened = preLM.secondhalf_forward(
        post_hidden_states_interven,
        causal_mask,
        position_embeddings,
    ).last_hidden_state

    # logits : (B, T, V)
    vocabs_intervened = preLM_generator.lm_head(vocabs_intervened)

    # CE loss
    # reshape -> (B*(T-1), V)
    intervened_generation_loss = CSE_crit(
        vocabs_intervened[:, :-1, :].reshape(-1, config.vocab_size),
        word_label.reshape(-1),
    )

    # ============================================================
    # 5. TOTAL LOSS
    # ============================================================

    loss = (
        args.concept_loss * concept_loss
        + args.intermediate_recon_loss * intermediate_reconstruction_loss
        + args.generation_recon_loss * generation_reconstruction_loss
        + args.cyclic_loss * cyclic_concept_loss
        + args.intervention_gen_loss * intervened_generation_loss
    )

    return {
        "loss": loss,
        "concept_loss": concept_loss,
        "intermediate_reconstruction_loss": intermediate_reconstruction_loss,
        "generation_reconstruction_loss": generation_reconstruction_loss,
        "cyclic_concept_loss": cyclic_concept_loss,
        "intervened_generation_loss": intervened_generation_loss,
    }

from transformers import LlamaConfig, LlamaModel, AutoTokenizer, RobertaTokenizerFast
import evaluate
from modules import CBLResidual, CBL, Roberta_classifier
from utils import elastic_net_penalty, mean_pooling, eos_pooling

@torch.no_grad()
def evaluate_steerability_and_concepts(preLM, preLM_generator, tokenizer, concept_set, args, loader=None, device="cuda"):
    """
    Performs Steerability (generation) and Concept Prediction (classification) tests.
    Loads and deletes RoBERTa internally to save VRAM.
    """
    results = {}
    preLM.eval()
    preLM_generator.eval()
    device_str = str(device)
    
    # --- 1. STEERABILITY TEST ---
    print("Running Steerability Test...")
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    classifier_path = args.dataset.replace('/', '_') + "_classifier.pt"
    classifier = Roberta_classifier(len(concept_set)).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    acc_metric = evaluate.load("accuracy")
    pred_logits = []
    
    # We use BS=1 for generation as requested
    num_samples_per_concept = max(1, 100 // len(concept_set))
    
    with torch.no_grad():
        for i in range(num_samples_per_concept):
            # Start with empty prompt or BOS
            input_ids = torch.tensor([[tokenizer.bos_token_id if tokenizer.bos_token_id else 128000]]).to(device)
            attention_mask = torch.ones_like(input_ids)
            
            for j in range(len(concept_set)):
                # Intervene: One-hot vector for target concept
                v = torch.zeros(len(concept_set), device=device)
                v[j] = 1.0
                intervene_tensor = v.view(1, 1, -1) # [BS=1, Seq=1, Dim]

                preLM_generator.model.intervene = intervene_tensor
                preLM_generator.model.intervention_margin = args.intervention_margin
                preLM_generator.model.intervention_spread = args.intervention_spread
                
                with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16):
                    output_tokens = generate(
                        preLM_generator, input_ids, preLM=preLM, length=100, temp=0.7, topk=100, topp=0.9
                    )
                
                preLM_generator.model.intervene = None # Reset
                decoded_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

                # Classify with RoBERTa
                encoded_input = roberta_tokenizer(decoded_text, return_tensors='pt', truncation=True, max_length=512).to(device)
                logits = classifier({"input_ids": encoded_input["input_ids"], "attention_mask": encoded_input["attention_mask"]})
                pred_logits.append(logits)

    pred_labels = torch.cat(pred_logits, dim=0).argmax(dim=-1).cpu().numpy()
    refs = list(range(len(concept_set))) * num_samples_per_concept
    steer_acc = acc_metric.compute(predictions=pred_labels, references=refs[:len(pred_labels)])['accuracy']
    results['steerability_acc'] = steer_acc

    # CLEANUP RoBERTa immediately
    del classifier, roberta_tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. CONCEPT PREDICTION TEST ---
    if loader is not None:
        print("Running Concept Prediction Test...")
        concept_metric = evaluate.load("accuracy")
        all_preds = []
        all_refs = []
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast(device_type=device_str, dtype=torch.bfloat16):
                    pre_h, _, _ = preLM.firsthalf_forward(batch["input_ids"], batch["attention_mask"])
                    concepts, _ = preLM.intermediate.encode(pre_h)
                
                pooled_concepts = eos_pooling(concepts, batch["attention_mask"])
                all_preds.append(pooled_concepts.argmax(dim=-1).cpu())
                all_refs.append(batch["label"].cpu())
        
        concept_acc = concept_metric.compute(predictions=torch.cat(all_preds), references=torch.cat(all_refs))['accuracy']
        results['concept_acc'] = concept_acc

    return results


import torch
import torch.nn.functional as F

def generate(self, ids, preLM, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5, eos_token_id=128001):
    past_key_values = None
    device = ids.device
    
    for i in range(length):
        # 1. Forward pass through the backbone (using KV-cache for efficiency)
        # Only pass the last token if we have a cache
        inputs = ids[:, -1:] if past_key_values is not None else ids
        outputs = preLM(inputs, past_key_values=past_key_values, use_cache=True)
        
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state # [batch, seq_len, hidden_dim]
        
        # 2. Project to vocabulary (lm_head)
        # Taking only the last position's logits
        logits = self.lm_head(hidden_states[:, -1, :]) # [batch, vocab_size]
        
        # 3. Apply Repetition Penalty (using your specific logic)
        # Note: This logic assumes batch_size=1 based on your 'ids[0]' indexing
        existing_tokens = ids[0]
        score = logits[0, existing_tokens]
        # If score is negative, make it more negative; if positive, dampen it
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits[0, existing_tokens] = score
        
        # 4. Scaling and Filtering
        next_token_logits = logits / temp
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
        
        # 5. Sample next token
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 6. Update sequence
        ids = torch.cat((ids, next_token), dim=-1)
        
        # 7. Check for EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
            
    return ids

import torch
from transformers import AutoTokenizer, LlamaConfig

# --- ASSUMING YOUR CustomLlamaModel AND LlamaForCausalLM CLASSES ARE DEFINED ABOVE ---

def test_llama3_custom_architecture():
    model_id = 'meta-llama/Meta-Llama-3-8B'
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. Load the model using YOUR custom class
    # We use strict=False so it ignores the fact that 'model.intermediate' is missing in Meta's weights
    print("Loading custom model with pretrained weights (this may take a minute)...")
    model = CustomLlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Setup Debugging
    model.model.debug = True
    model.model.where = 16  # Place the intermediate layer in the middle (Llama-3-8B has 32 layers)
    model.eval()

    # 3. INITIALIZATION CRITICAL STEP
    # Because 'intermediate' weights weren't in the checkpoint, they are currently RANDOM.
    # Random weights in the middle of a model = Gibberish.
    # We must initialize it to Identity so it doesn't break the pre-trained logic yet.
    print(model.model.intermediate)
    with torch.no_grad():
        if isinstance(model.model.intermediate, torch.nn.Linear):
            model.model.intermediate = nn.Identity()
            print("[INFO] Initialized intermediate layer to Identity Matrix.")

    # 4. Run Generation
    prompt = "The tallest mountain in the world is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("\n--- Starting Generation ---")
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False,
        use_cache=True
    )

    result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(f"\nResult: {result}")
    print("--- Test Finished ---")

if __name__ == "__main__":
    test_llama3_custom_architecture()