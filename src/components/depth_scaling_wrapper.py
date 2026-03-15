import functools
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# Patch helpers adapted from the upstream Qwen2/LLaMA decoder implementations.
import torch
import torch.nn as nn
from typing import Optional
from transformers.utils.deprecation import deprecate_kwarg

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
# from transformers.utils.generic import TransformersKwargs
# from transformers.utils.fx_utils import Unpack


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def scaled_forward_qwen2_llama3(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        depth_scaling: Optional[torch.Tensor] = None,                     
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
    if depth_scaling is None:
        depth_scaling = torch.tensor(1.0)
    depth_scaling = depth_scaling.to(hidden_states.device)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention                                                                               
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    # hidden_states = residual + hidden_states
    hidden_states = residual + depth_scaling * hidden_states
    # Fully Connected                                                                              
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    #hidden_states = residual + hidden_states
    hidden_states = residual + depth_scaling * hidden_states
    return hidden_states

def update_depth_scaling(model: nn.Module, model_name: str, depth_alpha: Optional[torch.Tensor] = None) -> nn.Module:
    if depth_alpha is None:
        depth_alpha = torch.tensor(1.0)
    # depth_scaling = torch.tensor(1/24.0)
    depth_scaling = torch.tensor(1/4) # TODO: update depending on model!
    depth_scaling = depth_scaling**depth_alpha
    forward_function = get_forward_function(model_name)
    decoder_layer_class = get_decoder_layer_class(model_name)
    new_forward = functools.partialmethod(forward_function, depth_scaling=depth_scaling)
    # breakpoint()
    for layer in model.model.layers:

        layer.forward = new_forward.__get__(layer, decoder_layer_class)
    return model
    
def get_decoder_layer_class(model_name: str):
    if "Qwen" in model_name:
        return Qwen2DecoderLayer
    elif "Llama-3.2" in model_name:
        return LlamaDecoderLayer
    else:
        raise ValueError(f"Model name {model_name} not supported")

def get_forward_function(model_name: str):
    if "Qwen" in model_name:
        return scaled_forward_qwen2_llama3
    elif "Llama-3.2" in model_name:
        return scaled_forward_qwen2_llama3
    else:
        raise ValueError(f"Model name {model_name} not supported")

if __name__ == "__main__":
    update_depth_scaling()
