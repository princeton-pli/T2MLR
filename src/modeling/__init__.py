from .tinyllama import TinyLlamaConfig, TinyLlamaForCausalLM
from .rnnlm import RNNLMConfig, RNNLMForCausalLM

__all__ = [
    "TinyLlamaConfig",
    "TinyLlamaForCausalLM",
    "RNNLMConfig",
    "RNNLMForCausalLM",
]

# Register custom models with the HF auto classes so `AutoModelForCausalLM.from_pretrained`
# can reload checkpoints produced by this repo.
try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("rnnlm", RNNLMConfig)
    AutoModelForCausalLM.register(RNNLMConfig, RNNLMForCausalLM)
except Exception:
    # Registration is best-effort; training can still instantiate via direct classes.
    pass
