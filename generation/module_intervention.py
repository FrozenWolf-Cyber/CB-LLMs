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
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention
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


class CustomLlamaModel(LlamaModel):
    def __init__(self, config, debug=False, where=24):
        super().__init__(config)
        self.where = where
        self.debug = debug
        # your intermediate layer
        if debug:
            self.intermediate = nn.Identity()
        else:
            self.intermediate = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False
        )

        # freeze everything except new layer
        for p in self.parameters():
            p.requires_grad = False

        for p in self.intermediate.parameters():
            p.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
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

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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


class CustomLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
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