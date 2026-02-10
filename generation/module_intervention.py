import torch
from torch import nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model, GPT2TokenizerFast, RobertaModel
import torch.nn.functional as F
from utils import top_k_top_p_filtering
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        with torch.no_grad():
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




        with torch.no_grad():
            for idx, decoder_layer in enumerate(self.layers[:self.where]):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]


        if self.debug:
            print(f"[LOG] TRIGGERED INTERMEDIATE AT LAYER {self.where}")
            print(hidden_states.shape)

        # Trainable layer
        hidden_states = self.intermediate(hidden_states)

        # Subsequent layers (must track gradients to pass them backward)
        for idx in range(self.where, self.config.num_hidden_layers):
            decoder_layer = self.layers[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
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
from transformers import LlamaConfig, AutoTokenizer

# Assuming your CustomLlamaModel and LlamaForCausalLM classes are defined above...

def test_custom_model_debug():
    print("--- Starting Test ---")
    
    # 1. Setup a tiny configuration to save memory/time
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4, # Small number of layers
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    
    # 2. Initialize the model with debug=True
    # We set 'where=2' so it triggers in the middle of our 4 layers
    model = LlamaForCausalLM(config)
    model.model.debug = True
    model.model.where = 2
    model.eval() # Generation usually happens in eval mode

    # 3. Prepare a dummy input
    # Note: In a real scenario, you'd use a tokenizer, 
    # but for a shape test, raw tensors work.
    input_ids = torch.randint(0, config.vocab_size, (1, 10)) 
    
    print(f"Running generate() with debug=True and where={model.model.where}...")
    
    # 4. Run generate
    # max_new_tokens=2 will trigger the forward pass twice
    output = model.generate(
        input_ids, 
        max_new_tokens=2, 
        do_sample=False,
        use_cache=True # This tests if your cache logic in forward holds up
    )

    print("--- Test Completed ---")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_custom_model_debug()