from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any, Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin


class RNNLMConfig(PretrainedConfig):
    model_type = "rnnlm"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 256,
        num_layers: int = 4,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        use_cache: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        # HF generation/cache utils often expect `num_hidden_layers`.
        self.num_hidden_layers = int(num_layers)
        self.rnn_type = str(rnn_type).lower()
        self.dropout = float(dropout)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        # RNNs don't use transformer KV caches; keep HF generation on the non-cache path by default.
        self.use_cache = bool(use_cache)

        if self.rnn_type not in ("gru", "lstm", "rnn"):
            raise ValueError(f"Unsupported rnn_type={rnn_type!r}. Use one of: gru, lstm, rnn.")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive.")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative.")


class RNNLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = RNNLMConfig
    base_model_prefix = "rnnlm"
    supports_gradient_checkpointing = False
    _supports_cache_class = False
    # HF Transformers (newer versions) expects a dict mapping {target_param_name: source_param_name}.
    # This is used by `post_init()` / `get_expanded_tied_weights_keys()` to build `all_tied_weights_keys`.
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config: RNNLMConfig):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        if config.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif config.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.RNN(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tie embeddings if requested. Note: metadata for tying is stored in `_tied_weights_keys` (dict).
        if config.tie_word_embeddings:
            self.tie_weights(recompute_mapping=True)

        self.post_init()

    def tie_weights(self, missing_keys=None, recompute_mapping: bool = True, **kwargs):
        """
        Tie the input and output embeddings.

        We delegate to `PreTrainedModel.tie_weights()` so HF can keep `all_tied_weights_keys` consistent for
        save/load and `from_pretrained` (which may call this with `missing_keys`).
        
        Note: PreTrainedModel.tie_weights() doesn't accept any arguments, so we accept them for
        compatibility but don't pass them to the parent.
        """
        return super().tie_weights()

    # --- HF embedding helpers ---
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        if self.config.tie_word_embeddings and hasattr(self, "lm_head"):
            self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None):
        if new_num_tokens is None:
            return self.get_input_embeddings()
        old_num_tokens, old_dim = self.embed_tokens.weight.shape
        if pad_to_multiple_of is not None and pad_to_multiple_of > 0:
            new_num_tokens = int(((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of)
        new_num_tokens = int(new_num_tokens)
        if new_num_tokens == old_num_tokens:
            return self.get_input_embeddings()

        new_embed = nn.Embedding(new_num_tokens, old_dim).to(self.embed_tokens.weight.device)
        new_embed.weight.data.normal_(mean=0.0, std=0.02)
        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_embed.weight.data[:num_to_copy] = self.embed_tokens.weight.data[:num_to_copy]
        self.embed_tokens = new_embed

        self.config.vocab_size = new_num_tokens
        # keep head tied if requested; else rebuild head
        if self.config.tie_word_embeddings:
            self.tie_weights()
        else:
            new_head = nn.Linear(old_dim, new_num_tokens, bias=False).to(self.lm_head.weight.device)
            new_head.weight.data.normal_(mean=0.0, std=0.02)
            new_head.weight.data[:num_to_copy] = self.lm_head.weight.data[:num_to_copy]
            self.lm_head = new_head
            # Clear tied weights keys when untied
            self._tied_weights_keys = {}
        return self.get_input_embeddings()

    # --- Generation caching ---
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # For RNNs, we need to handle padding properly to avoid processing padding tokens
        # which would corrupt the hidden state. We use pack_padded_sequence when attention_mask is available.
        emb = self.embed_tokens(input_ids)

        h0 = None
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            h0 = past_key_values[0]

        # Use pack_padded_sequence to skip padding tokens if attention_mask is provided
        # This is important for correct evaluation and training with batched sequences
        use_packing = attention_mask is not None and past_key_values is None
        if use_packing:
            # Compute lengths from attention_mask (batch_size,)
            lengths = attention_mask.sum(dim=1).cpu()
            # Pack sequences to skip padding tokens
            packed_emb = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        else:
            packed_emb = emb

        if self.config.rnn_type == "lstm":
            # LSTM expects (h0, c0). We store both in past_key_values.
            if past_key_values is not None and len(past_key_values) >= 2 and past_key_values[1] is not None:
                h0 = (past_key_values[0], past_key_values[1])
            if use_packing:
                packed_outputs, hn = self.rnn(packed_emb, h0) if h0 is not None else self.rnn(packed_emb)
                # Unpack the sequence back to (batch, seq_len, hidden_size)
                outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=emb.size(1))
            else:
                outputs, hn = self.rnn(emb, h0) if h0 is not None else self.rnn(emb)
            next_past: Optional[Tuple[torch.Tensor, ...]] = (hn[0], hn[1]) if (use_cache or use_cache is None) else None
        else:
            if use_packing:
                packed_outputs, hn = self.rnn(packed_emb, h0) if h0 is not None else self.rnn(packed_emb)
                # Unpack the sequence back to (batch, seq_len, hidden_size)
                outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=emb.size(1))
            else:
                outputs, hn = self.rnn(emb, h0) if h0 is not None else self.rnn(emb)
            next_past = (hn,) if (use_cache or use_cache is None) else None

        outputs = self.dropout(outputs)
        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_past,
            hidden_states=None,
            attentions=None,
        )


