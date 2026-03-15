from __future__ import annotations

import torch
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class _NoPositionalRotaryEmbedding(nn.Module):
    """Return unity cosines and zero sines while matching LLaMA's RoPE API."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.attention_scaling = 1.0
        self.head_dim = config.hidden_size // config.num_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = position_ids.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        cos = torch.ones(batch, seq_len, self.head_dim, device=device, dtype=dtype)
        sin = torch.zeros(batch, seq_len, self.head_dim, device=device, dtype=dtype)
        return cos, sin

    def _set_triton_flash_attn(self, _use_flash: bool) -> None:
        # Interface compatibility; nothing to configure.
        return None


class TinyLlamaConfig(LlamaConfig):
    """Tiny LLaMA-style causal LM config with positional encoding enabled (RoPE).

    This mirrors the tiny defaults previously used by the repo's small models. Use
    `--model_name_or_path tinyllama` (with `--from_pretrained False`) to instantiate it.
    """

    model_type = "tinyllama"

    def __init__(self, **kwargs):
        # Default to a small NanoGPT-like transformer.
        # Note: total parameter count depends heavily on vocab_size provided at runtime.
        defaults = dict(
            vocab_size=32000,  # overridden in train.py to match tokenizer
            hidden_size=256,
            intermediate_size=1024,  # 4 * hidden_size (GPT-style MLP width)
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            rope_theta=10000.0,  # standard RoPE scale; enables positional encoding
            rms_norm_eps=1e-5,
            initializer_range=0.02,
            hidden_act="silu",
            tie_word_embeddings=False,
            # If True, positional encoding is disabled by swapping rotary embeddings.
            disable_positional_encoding=False,
        )
        defaults.update(kwargs)
        defaults["model_type"] = "tinyllama"
        super().__init__(**defaults)


class TinyLlamaForCausalLM(LlamaForCausalLM):
    """Tiny LLaMA-style LM with optional RoPE disabling."""

    config_class = TinyLlamaConfig

    def __init__(self, config: TinyLlamaConfig):
        super().__init__(config)
        if bool(getattr(config, "disable_positional_encoding", False)):
            self._disable_rotary_embeddings()
        self.post_init()

    def _disable_rotary_embeddings(self) -> None:
        no_rope = _NoPositionalRotaryEmbedding(self.config)
        self.model.rotary_emb = no_rope
        for layer in self.model.layers:
            layer.self_attn.rotary_emb = no_rope

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "TinyLlamaForCausalLM does not ship with pretrained weights. Instantiate from config instead."
        )


