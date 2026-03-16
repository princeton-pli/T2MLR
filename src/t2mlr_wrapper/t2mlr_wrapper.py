import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt
from safetensors.torch import load_file as safe_load_file
from typing import Optional, Tuple, List, Union
from collections import OrderedDict
import os
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, GenerationMixin, AutoConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import get_last_checkpoint

from .block_wrapper import apply_block_wrapper, BlockWrapper
from .t2mlr_config import T2MLRConfig
from .t2mlr_gate_zoo import get_t2mlr_mixing_module_class
from .model_io_utils import (
    load_base_model_from_config,
    load_weights_for_model,
    fetch_hidden_size,
    resolve_dtype,
    load_t2mlr_config_with_fallback,
)

from modeling.tinyllama import TinyLlamaConfig, TinyLlamaForCausalLM

from components.all_arguments import T2MLRArguments
from components.t2mlr_utils import split_batch_by_recurrent_flow
from dataclasses import dataclass

from IPython import embed

import logging

LOGGING_FORMAT = (
    "%(asctime)s | %(levelname)-6s | [%(name)s] | "
    "%(funcName)s:%(lineno)d - %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S' 
)

logger = logging.getLogger(__name__)

@dataclass
class T2MLROutput(ModelOutput):
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    past_key_values: Optional[Tuple[torch.Tensor]] = None
    logits: Optional[torch.Tensor] = None

class T2MLRWrapper(PreTrainedModel, GenerationMixin):
    config_class = T2MLRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []  # Will be inherited from base model
    _supports_cache_class = False  # Disable new cache classes for compatibility
    
    def __init__(
        self,
        config: T2MLRConfig,
        base_model: Optional[PreTrainedModel] = None
    ):
        """
        Initialize the T2MLR wrapper for continuous chain of thought.
        
        Args:
            config: T2MLRConfig instance containing all configuration
            model: The base HuggingFace model to wrap (optional if loading from pretrained)
        """
        super().__init__(config)
        self.config = config

        # Create the base model from config if not provided
        if base_model is None:
            logger.info("Creating base model from config")
            base_model = load_base_model_from_config(self.config)

        # Fetch the list of transformer blocks from the base model
        base_layers = self.get_layers_from_model(base_model)
        self.hidden_size = fetch_hidden_size(base_model)
        self.num_layers = len(base_layers)
        logger.info(f"Base Model Hidden Size: {self.hidden_size}; Total number of layers: {self.num_layers}")
        
        # Configure the recurrence range
        self.t2mlr_enabled = self.config.t2mlr_enabled
        self.l_start = self.config.l_start
        # Allow Python-style negative indexing for l_end (e.g., -1 -> last layer)
        self.l_end = self.config.l_end if self.config.l_end >= 0 else self.num_layers + self.config.l_end
        assert self.l_start < self.l_end and self.l_end < self.num_layers, (
            f"Invalid layer indices: l_start ({self.l_start}) must be less than l_end ({self.l_end}) "
            f"and both must be within [0, {self.num_layers - 1}]"
        )
        # Deprecated: gate tracing is handled via record_gating_stats/mixing_module_logs.
        self._gate_trace_records: Optional[List[dict]] = None
        logger.info("Initializing T2MLRWrapper with l_start: {}, l_end: {}".format(self.l_start, self.l_end))
        
        # Initialize the recurrent mixing module
        assert 'recurrent_mixing_module_name' in self.config, f"recurrent_mixing_module_name must be set in the config (chose from {get_t2mlr_mixing_module_class(None).keys()})"
        self.recurrent_mixing_module_name = self.config.recurrent_mixing_module_name
        logger.info(f"Initializing T2MLR Mixing Module: {self.recurrent_mixing_module_name}")
        mixing_module_cls = get_t2mlr_mixing_module_class(self.recurrent_mixing_module_name)
        t2mlr_mixing_module = mixing_module_cls.from_config(config, hidden_size=self.hidden_size, dtype=resolve_dtype(self.config.base_config["dtype"]))

        if self.config.freeze_base_model:
            logger.info("Freezing base model parameters; T2MLR adapters remain trainable.")
            for param in base_model.parameters():
                param.requires_grad = False

        self.t2mlr_model = apply_block_wrapper(
            base_model,
            t2mlr_mixing_module,
            l_start=self.l_start,
        )

        # Copy important attributes from base model
        self.config.base_config = base_model.config.to_dict()
        self.config.base_model_type = base_model.config.model_type
        
        # Inherit split modules from base model if available
        if hasattr(base_model.__class__, '_no_split_modules'):
            self._no_split_modules = base_model.__class__._no_split_modules
        
        # Cache for simple recurrent forward (like inference_wrapper)
        self.recurrent_cache = None
        # Shared gating buffer for generate() runs
        self.active_gate_buffer: Optional[dict] = None
        # Flag for automatic control flow generation during .generate()
        self.auto_control_flow_generation = False
        # When True, prompt tokens are treated as recurrent during auto control flow generation.
        self.control_flow_all_recurrent = False

        # State for optional skip-to-l_end mode (populated when we inject recurrent input at l_start).
        self._t2mlr_skip_recurrent_embedding: Optional[torch.Tensor] = None
        self._t2mlr_skip_control_flows: Optional[torch.Tensor] = None
        self._l_end_skip_hook_handle = None
        self._register_l_end_skip_hook()

    def _register_l_end_skip_hook(self) -> None:
        if self._l_end_skip_hook_handle is not None:
            return
        try:
            self._l_end_skip_hook_handle = self.layers[self.l_end].register_forward_hook(self._l_end_skip_hook)
        except Exception as e:
            logger.warning("Failed to register l_end skip hook: %s", e)
            self._l_end_skip_hook_handle = None

    def _l_end_skip_hook(self, module: nn.Module, inputs, output):
        if not bool(getattr(self.config, "recurrent_skip_to_l_end", False)):
            return output

        recurrent_embedding = getattr(self, "_t2mlr_skip_recurrent_embedding", None)
        control_flows = getattr(self, "_t2mlr_skip_control_flows", None)
        if recurrent_embedding is None or control_flows is None:
            return output

        try:
            if not bool(torch.any(control_flows > 1).item()):
                return output
        except Exception:
            return output

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        # Align devices/dtypes (layer output dictates).
        if recurrent_embedding.device != hidden.device:
            recurrent_embedding = recurrent_embedding.to(hidden.device)
        if recurrent_embedding.dtype != hidden.dtype:
            recurrent_embedding = recurrent_embedding.to(dtype=hidden.dtype)
        if control_flows.device != hidden.device:
            control_flows = control_flows.to(hidden.device)

        if bool(getattr(self.config, "recurrent_skip_to_l_end_detach", False)):
            recurrent_embedding = recurrent_embedding.detach()

        # Build (B, S, 1) mask for recurrent positions.
        if control_flows.ndim != 2:
            return output
        if control_flows.shape[1] == 1 and hidden.shape[1] != 1:
            control_mask = (control_flows > 1).expand(-1, hidden.shape[1])
        elif control_flows.shape[1] == hidden.shape[1]:
            control_mask = (control_flows > 1)
        else:
            return output
        control_mask = control_mask.unsqueeze(-1).to(dtype=hidden.dtype)

        # Broadcast recurrent embedding across sequence if needed.
        if recurrent_embedding.ndim != 3:
            return output
        if recurrent_embedding.shape[0] != hidden.shape[0] or recurrent_embedding.shape[2] != hidden.shape[2]:
            return output
        if recurrent_embedding.shape[1] == 1 and hidden.shape[1] != 1:
            recurrent_embedding = recurrent_embedding.expand(-1, hidden.shape[1], -1)
        elif recurrent_embedding.shape[1] != hidden.shape[1]:
            return output

        weight = float(getattr(self.config, "recurrent_skip_to_l_end_weight", 1.0))
        skip = recurrent_embedding * weight

        if bool(getattr(self.config, "recurrent_skip_to_l_end_post_norm", False)):
            try:
                eps = float(getattr(self.config, "recurrent_skip_to_l_end_post_norm_eps", 1e-6))
                clamp = float(getattr(self.config, "recurrent_skip_to_l_end_post_norm_clamp", 5.0))
                clamp = max(clamp, 1.0)
                y = skip.detach().float()
                rms_y = torch.sqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
                scale = (1.0 / rms_y).clamp(min=1.0 / clamp, max=clamp).to(dtype=skip.dtype)
                skip = skip * scale
            except Exception:
                pass

        hidden = hidden + (skip * control_mask)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        if isinstance(output, list):
            out_list = list(output)
            out_list[0] = hidden
            return out_list
        return hidden

    def get_layers_from_model(self, model):
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Unsupported model architecture: cannot locate transformer blocks.")

    @property
    def layers(self):
        return self.get_layers_from_model(self.t2mlr_model)

    @property
    def model(self):
        """
        Expose the underlying HF "base model" under the conventional `.model` attribute
        without registering it as a child module (avoids duplicate keys in state_dict).
        """
        return getattr(self.t2mlr_model, "model", self.t2mlr_model)

    @property
    def lm_head(self):
        """
        Expose the language modeling head under the conventional `.lm_head` attribute
        without registering it as a child module (avoids duplicate keys in state_dict).
        """
        head = getattr(self.t2mlr_model, "lm_head", None)
        if head is not None:
            return head
        try:
            return self.t2mlr_model.get_output_embeddings()
        except Exception:
            return None


    # Legacy methods for loading checkpoints

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Override to sanitize checkpoint keys (e.g., torch.compile adds `_orig_mod.` prefixes).
        """
        # cleaned_state_dict = self._clean_state_dict_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    
    @classmethod
    def from_pretrained_with_t2mlr(cls, model_name_or_path: str, **kwargs):
        """
        Minimal loader:
          - If T2MLR config exists at path, rebuild base model, wrap, and load weights.
          - Otherwise error (no t2mlr_args-based wrapping here).
        """
        attn_impl_override = kwargs.pop("attn_impl", None)
        try:
            t2mlr_config = load_t2mlr_config_with_fallback(model_name_or_path)
            logger.info(f"Successfully loaded T2MLR config from {t2mlr_config.name_or_path}")
        except Exception as e:
            raise Exception(f"Failed to load T2MLR config from {model_name_or_path}") from e

        if "dtype" in kwargs and getattr(t2mlr_config, "dtype", None) is not None:
            t2mlr_config.dtype = kwargs["dtype"]
            logger.info(f"Set dtype to {kwargs['dtype']}")

        if attn_impl_override is not None:
            try:
                setattr(t2mlr_config, "attn_impl", attn_impl_override)
                base_cfg = getattr(t2mlr_config, "base_config", None)
                if isinstance(base_cfg, dict):
                    base_cfg["attn_impl"] = attn_impl_override
            except Exception:
                logger.warning("Could not apply attn_impl override to t2mlr_config; continuing with existing value.")

        base_model = load_base_model_from_config(t2mlr_config)
        model = cls(t2mlr_config, base_model)
        load_weights_for_model(model, model_name_or_path, strict=False)

        device_map = kwargs.get("device_map", None)
        if device_map is not None:
            from accelerate import dispatch_model, infer_auto_device_map
            if device_map == "auto":
                device_map = infer_auto_device_map(model)
            model = dispatch_model(model, device_map=device_map)

        return model
    
    @classmethod
    def from_base_model(cls, base_model: PreTrainedModel, t2mlr_args: T2MLRArguments):
        """Wrap an existing model instance with T2MLR."""
        t2mlr_config = T2MLRConfig.from_base_config(base_model.config, t2mlr_args)
        return cls(t2mlr_config, base_model)
    
    def set_recurrent_input(self, block_id, recurrent_embedding, control_flows, mixing_module_log_buffer) -> nn.Module:
        target_layer = self.layers[block_id]
        
        # Ensure recurrent embedding is on the same device as the target layer (for tensor/pipeline parallel)
        try:
            target_param = next(target_layer.parameters())
            target_device = target_param.device
        except StopIteration:
            # Fallback in unlikely case the layer has no parameters
            target_device = recurrent_embedding.device

        if recurrent_embedding.device != target_device:
            recurrent_embedding = recurrent_embedding.to(target_device)

        # Ensure control flows tensor is on the same device as the target layer
        if control_flows is not None and hasattr(control_flows, 'device') and control_flows.device != target_device:
            control_flows = control_flows.to(target_device)

        # Cache the injected recurrent embedding for optional skip-to-l_end mode (pre-gate).
        self._t2mlr_skip_recurrent_embedding = recurrent_embedding
        self._t2mlr_skip_control_flows = control_flows
        
        target_layer.set_recurrent_input(
            recurrent_embedding,
            control_flows,
            mixing_module_log_buffer
        )
        return self.t2mlr_model
    
    def reset_recurrent_input(self):
        for layer in self.layers:
            if isinstance(layer, BlockWrapper):
                layer.set_recurrent_input(None, None, None)
        self._t2mlr_skip_recurrent_embedding = None
        self._t2mlr_skip_control_flows = None

    # Gate trace helpers removed: use `generate(..., record_gating_stats=True)` and read `outputs.mixing_module_logs`.
    
    # Required PreTrainedModel delegation methods
    def get_input_embeddings(self):
        """Get the input embeddings from the wrapped model."""
        return self.t2mlr_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set the input embeddings of the wrapped model."""
        self.t2mlr_model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """Get the output embeddings from the wrapped model."""
        return self.t2mlr_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings of the wrapped model."""
        self.t2mlr_model.set_output_embeddings(new_embeddings)
    
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None):
        """Resize token embeddings of the wrapped model."""
        return self.t2mlr_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
    
    def tie_weights(self):
        """Tie weights in the wrapped model."""
        self.t2mlr_model.tie_weights()
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder the cache for beam search."""

        # Fall back to the model's _reorder_cache if it exists
        if hasattr(self.t2mlr_model, '_reorder_cache'):
            return self.t2mlr_model._reorder_cache(past_key_values, beam_idx)
        if hasattr(self.t2mlr_model.__class__, '_reorder_cache'):
            return self.t2mlr_model.__class__._reorder_cache(past_key_values, beam_idx)
        
        if past_key_values is None:
            return past_key_values
        if isinstance(past_key_values, tuple):
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                for layer_past in past_key_values
            )
        return past_key_values
    
    def can_generate(self):
        """Check if the model can generate sequences."""
        return True

    def eval(self):
        """Set the model to evaluation mode."""
        super().eval()
        self.t2mlr_model.eval()
        self.config.batch_forward = False
        return self
    
    def train(self, *args, **kwargs):
        """Set the model to training mode."""
        super().train(*args, **kwargs)
        self.t2mlr_model.train(*args, **kwargs)
        self.config.batch_forward = True
        return self

    @staticmethod
    def _coerce_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dtype == torch.bool or attention_mask.is_floating_point():
            return attention_mask
        return attention_mask.to(torch.bool)
        
    def forward(
            self,
            input_ids,
            control_flows=None,
            attention_mask=None,
            past_key_values=None,
            hidden_states=None,
            record_gating_stats: bool = False,
            prompt_recurrence_bfad: int = -1, # if -1, use infinite recurrence (exact sequence forward)
            **kwargs
        ) -> Union[T2MLROutput, CausalLMOutputWithPast]:

        logger.debug("Forward pass !!!!\n")
        # During generation we may want to record gate stats, but cannot always pass custom kwargs
        # through `transformers.generate()` (torch.compile / wrappers can cause kwargs validation to
        # reject unknown model_kwargs). Treat an active buffer as an implicit "record stats" flag.
        if (not record_gating_stats) and (getattr(self, "active_gate_buffer", None) is not None):
            record_gating_stats = True

        attention_mask = self._coerce_attention_mask(attention_mask)

        # Internal flag toggled by generate() to force simple recurrence
        force_simple = getattr(self, "_force_simple_recurrent", False)
        if control_flows is None:
            if self.config.t2mlr_enabled:
                raise ValueError(
                    "T2MLR is enabled but `control_flows` was not provided. "
                    "Pass `control_flows` explicitly (shape like `input_ids`) or disable T2MLR."
                )

        # If T2MLR is not enabled or there is no recurrence, do a regular forward pass
        if not self.config.t2mlr_enabled or (control_flows is not None and torch.max(control_flows) <= 1):
            logger.debug("Regular forward pass")
            # During generation, we still want to seed the recurrent cache from the prompt so
            # the first recurrent token can mix with the last prompt token's l_end state.
            seed_recurrent_cache = bool(
                getattr(self, "_force_simple_recurrent", False)
                and getattr(self, "auto_control_flow_generation", False)
                and (past_key_values is None or self._is_empty_cache(past_key_values))
                and kwargs.get("use_cache", False)
                and self.config.t2mlr_enabled
            )

            captured_hidden = {}
            hook_handle = None
            if seed_recurrent_cache:
                def _capture_l_end_hidden(module, inputs, output):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    captured_hidden[self.l_end] = hidden
                hook_handle = self.layers[self.l_end].register_forward_hook(_capture_l_end_hidden)

            try:
                outputs = self.t2mlr_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs
                )
            finally:
                if hook_handle is not None:
                    hook_handle.remove()

            if seed_recurrent_cache and self.l_end in captured_hidden:
                recurrent_connection = captured_hidden[self.l_end][:, -1:, :]
                if self.config.connection_detach:
                    recurrent_connection = recurrent_connection.detach()
                self.recurrent_cache = recurrent_connection.clone()

            return outputs
        
        # If batch forward is enabled, do a batch approximate forward pass.
        # Skip when force_simple is set (e.g. during generate()) since
        # batch_approximate_forward does not support KV caching.
        elif self.config.batch_forward and not force_simple:
            return self.batch_approximate_forward(
                input_ids,
                control_flows,
                attention_mask,
                self.config.batch_forward_approximate_depth,
                past_key_values,
                hidden_states,
                record_gating_stats=record_gating_stats,
                **kwargs
            )

        elif input_ids.shape[1] > 1:
            # This will be used when we have a multi-token prefill and want to use
            # Batch approximate forward to compute the hidden states for the prefill
            logger.debug("Batched prefill forward for multi-token sequences")
            # print(kwargs['use_cache'])
            if prompt_recurrence_bfad == -1:
                return self.exact_sequence_recurrent_forward(
                    input_ids=input_ids,
                    control_flows=control_flows,
                    attention_mask=attention_mask,
                    record_gating_stats=record_gating_stats,
                    **kwargs
                )
            else:
                logger.debug("Doing batch approximate forward")
                return self.batch_approximate_forward(
                    input_ids,
                    control_flows,
                    attention_mask,
                    min(prompt_recurrence_bfad, input_ids.shape[1] + 10),
                    past_key_values,
                    hidden_states,
                    training_flag=False,
                    record_gating_stats=record_gating_stats,
                    **kwargs
                )
        else:
            assert not self.training or force_simple, "Simple recurrent forward is only supported in evaluation mode, please set batch_forward to True if you want to train with T2MLR"
            # For multi-token sequences, prefer exact recurrence unless generate() set force-simple.
            if (not force_simple):
                logger.debug("Exact sequence recurrent forward")
                return self.exact_sequence_recurrent_forward(
                    input_ids=input_ids,
                    control_flows=control_flows,
                    attention_mask=attention_mask,
                    record_gating_stats=record_gating_stats,
                    **kwargs,
                )
            logger.debug("Simple recurrent forward")
            return self.simple_recurrent_forward(
                input_ids=input_ids,
                control_flows=control_flows,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                record_gating_stats=record_gating_stats,
                **kwargs,
            )
    
    def simple_recurrent_forward(
            self,
            input_ids,
            control_flows,
            attention_mask,
            past_key_values=None,
            hidden_states=None,
            record_gating_stats: bool = False,
            **kwargs
        ) -> Union[T2MLROutput, CausalLMOutputWithPast]:
        """
        Simple recurrent forward pass similar to inference_wrapper approach.
        Uses internal cache to capture hidden states from l_end and inject at l_start.
        """
        gate_buffer = None
        if record_gating_stats:
            if self.active_gate_buffer is None:
                self.active_gate_buffer = {}
            gate_buffer = self.active_gate_buffer

        prev_recurrent_cache = self.recurrent_cache

        # If we have cached states from previous token and need recurrence, inject them
        if self.recurrent_cache is not None and torch.any(control_flows > 1):
            # IMPORTANT: BlockWrapper now applies mixing where (control_flow > 1).
            # We pass the raw control_flows (containing 1s for prompt, 2s for recurrence)
            # so BlockWrapper can distinguish them.
            self.set_recurrent_input(
                block_id=self.l_start,
                recurrent_embedding=self.recurrent_cache,
                control_flows=control_flows,
                mixing_module_log_buffer=gate_buffer,
            )
        
        # Remove duplicate arguments from kwargs
        kwargs.pop('use_cache', None)
        kwargs.pop('output_hidden_states', None)
        captured_hidden = {}

        # FlashAttention varlen does not allow zero-length rows. When left padding causes
        # attention_mask rows with all zeros (e.g., early prefill steps), temporarily
        # fall back to eager attention for this forward.
        use_eager_attention = False
        mask_for_check = attention_mask
        if mask_for_check is not None and torch.is_tensor(mask_for_check) and mask_for_check.ndim == 2:
            mask_for_check = self._coerce_attention_mask(mask_for_check)
            zero_len = (mask_for_check.sum(dim=1) == 0).any().item()
            use_eager_attention = bool(zero_len)

        attn_impl_attr = None
        orig_attn_impl = None
        model_cfg = getattr(self.t2mlr_model, "config", None)
        if use_eager_attention and model_cfg is not None:
            if hasattr(model_cfg, "_attn_implementation"):
                attn_impl_attr = "_attn_implementation"
            elif hasattr(model_cfg, "attn_implementation"):
                attn_impl_attr = "attn_implementation"
            if attn_impl_attr is not None:
                orig_attn_impl = getattr(model_cfg, attn_impl_attr, None)
                if orig_attn_impl != "eager":
                    setattr(model_cfg, attn_impl_attr, "eager")

        # Create a hook to capture the l_end hidden state
        def _capture_l_end_hidden(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured_hidden[self.l_end] = hidden

        # Always capture l_end hidden state.
        #
        # Rationale: `simple_recurrent_forward()` may be used during prefill (prompt-only tokens with
        # control_flow<=1). In that case we still want to seed `self.recurrent_cache` from the last
        # prompt token so the first recurrent decode step can mix with it.
        hook_handle = self.layers[self.l_end].register_forward_hook(_capture_l_end_hidden)
        try:
            output = self.t2mlr_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,  # KV cache must be enabled for T2MLR to work at inference time.
                output_hidden_states=False,
                **kwargs,
            )
        finally:
            hook_handle.remove()
            if attn_impl_attr is not None and model_cfg is not None:
                try:
                    setattr(model_cfg, attn_impl_attr, orig_attn_impl)
                except Exception:
                    pass

        if self.l_end not in captured_hidden:
            raise RuntimeError("Failed to capture l_end hidden state during simple recurrent forward.")
        recurrent_connection = captured_hidden[self.l_end][:, -1:, :]

        # Optional residual from previous recurrent embedding -> next recurrent embedding (cache).
        if (
            prev_recurrent_cache is not None
            and bool(getattr(self.config, "recurrent_residual_to_recurrent_cache", False))
            and (not bool(getattr(self.config, "recurrent_skip_to_l_end", False)))
            and bool(torch.any(control_flows > 1).item())
        ):
            residual = prev_recurrent_cache
            if bool(getattr(self.config, "recurrent_residual_to_recurrent_cache_detach", False)):
                residual = residual.detach()
            if residual.device != recurrent_connection.device:
                residual = residual.to(recurrent_connection.device)
            if residual.dtype != recurrent_connection.dtype:
                residual = residual.to(dtype=recurrent_connection.dtype)
            if residual.shape == recurrent_connection.shape:
                weight = float(getattr(self.config, "recurrent_residual_to_recurrent_cache_weight", 1.0))
                if weight != 0.0:
                    recurrent_connection = recurrent_connection + (weight * residual)

                if bool(getattr(self.config, "recurrent_residual_to_recurrent_cache_post_norm", False)):
                    try:
                        eps = float(getattr(self.config, "recurrent_residual_to_recurrent_cache_post_norm_eps", 1e-6))
                        clamp = float(getattr(self.config, "recurrent_residual_to_recurrent_cache_post_norm_clamp", 5.0))
                        clamp = max(clamp, 1.0)
                        y = recurrent_connection.detach().float()
                        rms_y = torch.sqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
                        scale = (1.0 / rms_y).clamp(min=1.0 / clamp, max=clamp).to(dtype=recurrent_connection.dtype)
                        recurrent_connection = recurrent_connection * scale
                    except Exception:
                        pass

        if self.config.connection_detach:
            recurrent_connection = recurrent_connection.detach()
        self.recurrent_cache = recurrent_connection.clone()
        # print("[Simple Recurrent Forward] Recurrent embedding: ", recurrent_connection[0, 0, :10])
        
        # Reset recurrent input for next call
        self.reset_recurrent_input()
        
        if record_gating_stats and isinstance(self.layers[self.l_start], BlockWrapper):
            buffer = gate_buffer if gate_buffer is not None else self.layers[self.l_start].mixing_module_log_buffer
            setattr(output, "mixing_module_logs", buffer if buffer is not None else {})

        return output

    def exact_sequence_recurrent_forward(
            self,
            input_ids: torch.Tensor,
            control_flows: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            record_gating_stats: bool = False,
            **kwargs
        ) -> CausalLMOutputWithPast:
        """
        Exact full-sequence recurrence with support for mixed control-flow prompts.

        - Recurrent positions (control_flow > 1): processed token-by-token via `simple_recurrent_forward()`.
        - Non-recurrent contiguous spans where *all batch items* have control_flow <= 1: processed in one
          parallel forward pass through the base model (KV-cache aware).

        Semantics:
          - Feeds one token at a time with KV cache enabled.
          - Uses `simple_recurrent_forward()` to inject the previous token's recurrent state
            (captured at `l_end`) into `l_start` on the next step.

        Note: This is significantly slower than a single full-sequence forward.
        """
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError(f"Expected `input_ids` with shape (B, S), got {None if input_ids is None else tuple(input_ids.shape)}")
        if control_flows is None or control_flows.shape != input_ids.shape:
            raise ValueError(f"Expected `control_flows` with shape {tuple(input_ids.shape)}, got {None if control_flows is None else tuple(control_flows.shape)}")

        labels = kwargs.pop("labels", None)
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(f"Expected `labels` with shape {tuple(input_ids.shape)}, got {tuple(labels.shape)}")

        # New sequence: do not carry recurrent state across sequences/batches.
        self.recurrent_cache = None
        # Define the initial recurrent state (h_{-1}) as zeros so the first recurrent token
        # can still apply mixing (matches batch_approximate_forward's shifted-cache semantics).
        if bool(torch.any(control_flows[:, 0] > 1).item()):
            try:
                p = next(self.parameters())
                init_device = p.device
                init_dtype = p.dtype
            except StopIteration:
                init_device = input_ids.device
                init_dtype = torch.float32
            self.recurrent_cache = torch.zeros(
                (input_ids.shape[0], 1, self.hidden_size),
                device=init_device,
                dtype=init_dtype,
            )

        past_key_values = None
        logits_chunks: List[torch.Tensor] = []

        loss_sum = None
        loss_count = None

        batch_size, seq_len = input_ids.shape
        ignore_index = -100

        cache_position_full = kwargs.pop("cache_position", None)
        position_ids_full = kwargs.pop("position_ids", None)

        # print("control flows: ", control_flows)

        t = 0
        while t < seq_len:
            # Fast path: a whole non-recurrent span shared across the batch.
            all_nonrecur_here = bool(torch.all(control_flows[:, t] <= 1).item())
            if all_nonrecur_here:
                t_end = t + 1
                while t_end < seq_len and bool(torch.all(control_flows[:, t_end] <= 1).item()):
                    t_end += 1

                # No T2MLR mixing in non-recurrent spans, but still seed recurrence from l_end.
                self.recurrent_cache = None
                self.reset_recurrent_input()

                chunk_input_ids = input_ids[:, t:t_end]
                chunk_attention_mask = attention_mask[:, :t_end] if attention_mask is not None else None
                chunk_position_ids = position_ids_full[:, t:t_end] if position_ids_full is not None else None
                chunk_cache_position = cache_position_full[t:t_end] if cache_position_full is not None else None

                kwargs.pop("use_cache", None)
                kwargs.pop("output_hidden_states", None)

                captured_hidden = {}

                def _capture_l_end_hidden(module, inputs, output):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    captured_hidden[self.l_end] = hidden

                hook_handle = self.layers[self.l_end].register_forward_hook(_capture_l_end_hidden)
                try:
                    out = self.t2mlr_model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=False,
                        cache_position=chunk_cache_position,
                        position_ids=chunk_position_ids,
                        **kwargs,
                    )
                finally:
                    hook_handle.remove()

                past_key_values = getattr(out, "past_key_values", None)
                chunk_logits = out.logits
                logits_chunks.append(chunk_logits)

                # Seed recurrent cache from the last token of this (non-recurrent) span.
                if self.l_end in captured_hidden:
                    recurrent_connection = captured_hidden[self.l_end][:, -1:, :]
                    if self.config.connection_detach:
                        recurrent_connection = recurrent_connection.detach()
                    self.recurrent_cache = recurrent_connection.clone()

                if labels is not None:
                    max_p = min(t_end - 1, seq_len - 2)
                    if max_p >= t:
                        logits_for_loss = chunk_logits[:, : (max_p - t + 1), :].float()
                        targets = labels[:, (t + 1) : (max_p + 2)]
                        step_loss_sum = F.cross_entropy(
                            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                            targets.reshape(-1),
                            ignore_index=ignore_index,
                            reduction="sum",
                        )
                        step_loss_count = (targets.reshape(-1) != ignore_index).sum()
                        loss_sum = step_loss_sum if loss_sum is None else (loss_sum + step_loss_sum)
                        loss_count = step_loss_count if loss_count is None else (loss_count + step_loss_count)

                t = t_end
                continue

            # Slow path: recurrent (or mixed) position; process token-by-token.
            step_input_ids = input_ids[:, t : t + 1]
            step_control = control_flows[:, t : t + 1]
            step_attention_mask = attention_mask[:, : t + 1] if attention_mask is not None else None
            step_position_ids = position_ids_full[:, t : t + 1] if position_ids_full is not None else None
            step_cache_position = cache_position_full[t : t + 1] if cache_position_full is not None else None

            step_out = self.simple_recurrent_forward(
                input_ids=step_input_ids,
                control_flows=step_control,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                hidden_states=None,
                record_gating_stats=record_gating_stats,
                cache_position=step_cache_position,
                position_ids=step_position_ids,
                **kwargs,
            )

            past_key_values = getattr(step_out, "past_key_values", None)
            step_logits = step_out.logits
            logits_chunks.append(step_logits)

            if labels is not None and t + 1 < seq_len:
                targets = labels[:, t + 1]
                step_loss_sum = F.cross_entropy(
                    step_logits[:, 0, :].float(),
                    targets,
                    ignore_index=ignore_index,
                    reduction="sum",
                )
                step_loss_count = (targets != ignore_index).sum()
                loss_sum = step_loss_sum if loss_sum is None else (loss_sum + step_loss_sum)
                loss_count = step_loss_count if loss_count is None else (loss_count + step_loss_count)

            t += 1

        logits = torch.cat(logits_chunks, dim=1) if logits_chunks else input_ids.new_zeros((batch_size, 0, 0))
        loss = None
        if loss_sum is not None and loss_count is not None:
            loss = loss_sum / torch.clamp(loss_count, min=1)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    def post_process_gating_stats(self):
        """
        Post process the gating logs to convert them to a dictionary.
        """
        if self.active_gate_buffer is None:
            return
        for key, stats in self.active_gate_buffer.items():
            # Concatenate the stats along the sequence length dimension
            self.active_gate_buffer[key] = np.concatenate(stats, axis=1)
        # print(f"Post processed gating stats: {self.active_gate_buffer}")
    
    # Generation support methods
    def generate(self, *args, auto_control_flow: bool = True, record_gating_stats: bool = False, **kwargs):
        """
        Generate sequences using the model with automatic T2MLR control flow management.
        
        Args:
            auto_control_flow (bool): If True, automatically manage control flows during generation.
                - Prompt tokens get control_flow=1
                - Generated tokens get control_flow=2 (enables recurrence)
                Set to False to provide manual control_flow via kwargs.
            record_gating_stats (bool): If True, collect gating logs across the full generation.
            **kwargs: Standard generation arguments (see transformers.GenerationMixin.generate)
        
        Returns:
            Generated token IDs
            
        Example:
            >>> outputs = model.generate(input_ids, max_new_tokens=50, auto_control_flow=True)
        """
        # Set up shared gating buffer for the full generate call
        self.active_gate_buffer = {} if record_gating_stats else None
        self.auto_control_flow_generation = auto_control_flow and self.config.t2mlr_enabled and "control_flows" not in kwargs
        
        # Disable cache_implementation if not already specified (for compatibility)
        if 'cache_implementation' not in kwargs:
            kwargs['cache_implementation'] = None
        
        # Reset recurrent cache before generation
        self.recurrent_cache = None

        # During generate, force simple recurrent forward (skip exact sequence loop)
        self._force_simple_recurrent = True
        # NOTE: Do NOT pass `record_gating_stats` down into `transformers.generate()` as a model_kwarg.
        # Some wrappers (e.g., torch.compile) lose the explicit forward() signature, and Transformers'
        # kwargs validation will raise on unknown keys. We instead use `self.active_gate_buffer` as the
        # implicit flag inside forward().
        out = super().generate(*args, **kwargs)
        self._force_simple_recurrent = False


        # Attach the gating buffer to the output if record_gating_stats is True
        if record_gating_stats:
            self.post_process_gating_stats()
            assert hasattr(out, "__setattr__"), "Output must have __setattr__ attribute to attach gating buffer"
            setattr(out, "mixing_module_logs", self.active_gate_buffer if self.active_gate_buffer is not None else {})
            # Important: avoid leaking a post-processed (ndarray) buffer into subsequent forward() calls.
            # forward() treats a non-None active_gate_buffer as an implicit "record stats" flag, which can
            # cause BlockWrapper.update_gating_buffer() to append into an ndarray and crash.
            self.active_gate_buffer = None

        return out
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        control_flows: Optional[torch.Tensor] = None,
        recurrence_in_prompt: bool = False,
        prompt_recurrence_start_offset: int = -1, # if -1, set all tokens to be recurrence
        **model_kwargs
    ):
        """
        Prepare inputs for generation, including T2MLR-specific control flows.
        
        This method is called by generate() at each step to prepare inputs.
        """
        # Get manual control flow if provided
        manual_control_flow = control_flows
        
        # Get base model's prepared inputs
        base_inputs = self.t2mlr_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **model_kwargs
        )

        # If T2MLR is disabled, just return base inputs
        if not self.t2mlr_enabled:
            if manual_control_flow is not None:
                base_inputs["control_flows"] = manual_control_flow
            return base_inputs
        
        # Handle control flow
        auto_mode = self.auto_control_flow_generation
        control_flow_tensor = manual_control_flow

        
        if control_flow_tensor is None and auto_mode:
            tokens = base_inputs.get("input_ids", input_ids)
            if tokens is None:
                raise ValueError(
                    "Automatic T2MLR generation requires `input_ids`. "
                    "Provide `control_flows` manually when using inputs_embeds."
                )
            
            # Determine control flow value based on whether we're processing prompt or generating
            is_prompt = past_key_values is None or self._is_empty_cache(past_key_values)
            prompt_value = 2 if bool(getattr(self, "control_flow_all_recurrent", False)) else 1
            fill_value = prompt_value if is_prompt else 2  # 1/2 for prompt, 2 for generation
            control_flow_tensor = tokens.new_full(tokens.shape, fill_value)

            if recurrence_in_prompt and is_prompt:
                if prompt_recurrence_start_offset < 0:
                    control_flow_tensor[:, :] = 2
                else:
                    recurrence_start = tokens.shape[1] - prompt_recurrence_start_offset
                    if recurrence_start > 0:
                        control_flow_tensor[:, recurrence_start:] = 2
                    elif recurrence_start == 0:
                        control_flow_tensor[:, :] = 2
        
        if control_flow_tensor is None:
            control_flow_tensor = torch.tensor(control_flows, device=input_ids.device, dtype=input_ids.dtype)

        # Zero-out control flows on padding positions to avoid T2MLR mixing on padded tokens.
        mask = base_inputs.get("attention_mask", attention_mask)
        if mask is not None and torch.is_tensor(mask):
            mask = self._coerce_attention_mask(mask)
            if mask.dtype != torch.bool:
                mask = mask > 0
            if mask.device != control_flow_tensor.device:
                mask = mask.to(control_flow_tensor.device)
            # During decoding, attention_mask can be full-length while control_flows is last-token only.
            if mask.shape != control_flow_tensor.shape:
                if (
                    mask.ndim == 2
                    and control_flow_tensor.ndim == 2
                    and mask.shape[0] == control_flow_tensor.shape[0]
                    and mask.shape[1] >= control_flow_tensor.shape[1]
                ):
                    mask = mask[:, -control_flow_tensor.shape[1]:]
                else:
                    mask = None
            if mask is not None:
                control_flow_tensor = control_flow_tensor.clone()
                control_flow_tensor = control_flow_tensor.masked_fill(~mask, 0)

        logger.debug("Control flow tensor: ", control_flow_tensor)
        
        base_inputs["control_flows"] = control_flow_tensor
        return base_inputs
    
    def _is_empty_cache(self, past_key_values):
        """Check if KV cache is empty."""
        if past_key_values is None:
            return True

        # Handle Cache objects (e.g. DynamicCache) created by HF generate()
        if hasattr(past_key_values, "get_seq_length"):
            try:
                return past_key_values.get_seq_length() == 0
            except Exception:
                pass

        # Handle legacy (list/tuple) cache formats
        if isinstance(past_key_values, (list, tuple)):
            if len(past_key_values) == 0:
                return True
            first_layer = past_key_values[0]
            if isinstance(first_layer, (list, tuple)) and len(first_layer) > 0:
                first_kv = first_layer[0]
                if isinstance(first_kv, torch.Tensor):
                    return first_kv.shape[-2] == 0  # Check sequence length dimension

        return False
        
    def batch_cache_shift(self, cache):
        # Shift the cache by one token to the right, fill in all-zeros for the first token
        first_token_placeholder = torch.zeros_like(cache[:, :1, :])
        shifted_cache = torch.cat([first_token_placeholder, cache[:, :-1, :]], dim=1)
        return shifted_cache

    @staticmethod
    def _apply_packed_boundary_zeros(shifted_cache: torch.Tensor, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """
        For packed sequences (multiple segments concatenated along time), prevent T2MLR recurrent cache
        from leaking across segment boundaries by zeroing the cache at every segment-start token.

        We treat any token position where position_ids == 0 as a segment start.
        """
        if position_ids is None:
            return shifted_cache
        try:
            # position_ids is (B, S)
            boundary = (position_ids == 0)
            if boundary.any():
                shifted_cache = shifted_cache.clone()
                shifted_cache[boundary] = 0
        except Exception as e:
            # Warn if boundary zeroing fails, as this is critical for packed training stability.
            logger.warning(f"Failed to apply packed boundary zeros: {e}. Check position_ids shape/compatibility.")
            return shifted_cache
        return shifted_cache
    
    def batch_approximate_forward(
            self,
            input_ids,
            control_flows,
            attention_mask,
            approximate_depth: int,
            past_key_values=None,
            hidden_states=None,
            training_flag: bool = True,
            record_gating_stats: bool = False,
            **kwargs
        ) -> Union[T2MLROutput, CausalLMOutputWithPast]:

        gate_buffer = None
        if record_gating_stats:
            if self.active_gate_buffer is None:
                self.active_gate_buffer = {}
            gate_buffer = self.active_gate_buffer
        
        if training_flag:
            assert past_key_values is None, "In batch approximate forward training we do not support kv caching, currently getting: " + str(past_key_values)
        assert hidden_states is None, "There should be no previous hidden states in batch approximate forward"
        assert approximate_depth > 0, "Approximate depth must be greater than 0"

        # Clear any stale recurrent cache left on BlockWrapper by a previous
        # call's checkpoint recomputation (which runs set_recurrent_input
        # during backward and never resets it).
        self.reset_recurrent_input()

        # Track recurrence using raw control flows (0=padding, 1=prompt, 2=recurrence).
        # BlockWrapper will only mix when control_flow > 1.
        recurrence_tracker = control_flows.clone()

        # Packed-sequence boundary handling (FlashAttention varlen packing uses position_ids resets).
        position_ids = kwargs.get("position_ids", None)
        # if position_ids is not None:
        # TODO: Fix packing, should be before feeding in the data instead of here
        #     try:
        #         boundary = (position_ids == 0)
        #         if boundary.any():
        #             recurrence_tracker = recurrence_tracker.clone()
        #             recurrence_tracker[boundary] = 0
        #     except Exception:
        #         pass

        # Do a full regular forward pass and cache the hidden states before reaching l_start
        # Cache the forward pass arguments before reaching l_start
        captured_args = {}
        captured_hidden = {}
        
        def capture_layer_inputs(module, args, kwargs):
            # Extract the hidden states explicitly (first positional arg)
            if isinstance(args, tuple) and len(args) >= 1:
                captured_args['hidden_states'] = args[0]
            else:
                captured_args['hidden_states'] = kwargs.get('hidden_states')
            captured_args['kwargs'] = dict(kwargs) if kwargs is not None else {}
 
        hook_handle = self.layers[self.l_start].register_forward_pre_hook(
            capture_layer_inputs, with_kwargs=True
        )

        def capture_l_end(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured_hidden[self.l_end] = hidden

        hook_l_end = self.layers[self.l_end].register_forward_hook(capture_l_end)

        output_dummy = self.t2mlr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            **kwargs
        )
        hook_handle.remove()
        hook_l_end.remove()

        # If the batch backward approximate depth is 0, return the dummy output (first forward pass)
        if self.config.batch_forward_approximate_depth == 0:
            assert training_flag, "Batch forward approximate depth is 0, but training flag is False"
            return output_dummy

        # Store captured arguments for later use or inspection
        l_start_input_cache = captured_args
        filtered_kwargs = {
            k: v for k, v in l_start_input_cache['kwargs'].items() 
            if k not in ['past_key_values', 'past_key_value', 'cache_position', 'position_ids']
        }

        # Cache the hidden states after layer l_end for the first recursion
        if self.l_end not in captured_hidden:
            raise RuntimeError("Failed to capture l_end hidden state during batch approximate forward.")
        
        # Shift the cache by one token to the right, prepare for the first recursion
        initial_cache_raw = captured_hidden[self.l_end]
        
        # If backward depth is less than forward depth, we need to detach the first cache anyway
        if self.config.connection_detach or self.config.batch_backward_approximate_depth < self.config.batch_forward_approximate_depth:
            initial_cache_raw = initial_cache_raw.detach()

        initial_cache_shifted = self.batch_cache_shift(initial_cache_raw).clone()
        initial_cache_shifted = self._apply_packed_boundary_zeros(initial_cache_shifted, position_ids)
        l_end_output_caches = [initial_cache_shifted]
        logger.debug(f"Batch forward at depth 1 completed")

        recur_mask = (recurrence_tracker > 1).unsqueeze(-1).to(dtype=l_end_output_caches[-1].dtype)
        
        # Do batch parallel forward for approximate depth times
        for d in range(approximate_depth - 1):

            # Set the recurrent input of l_start (wrapped layer) from the latest l_end output cache
            self.set_recurrent_input(
                block_id=self.l_start,
                recurrent_embedding=l_end_output_caches[-1],
                control_flows=recurrence_tracker,
                mixing_module_log_buffer={} if record_gating_stats else None,
            )

            # --- BFA gradient checkpointing: wrap recurrent-layer forward ---
            _hs_input = l_start_input_cache['hidden_states'].clone()
            if getattr(self.config, "bfa_gradient_checkpointing", False) and torch.is_grad_enabled():
                # We must re-set the recurrent input inside the checkpointed fn
                # so that both the original forward and the recomputation see
                # identical BlockWrapper state.
                _recurrent_emb = l_end_output_caches[-1]
                _recurrence_trk = recurrence_tracker

                def _ckpt_bfa_fn(hs, rec_emb, rec_trk):
                    self.set_recurrent_input(
                        block_id=self.l_start,
                        recurrent_embedding=rec_emb,
                        control_flows=rec_trk,
                        mixing_module_log_buffer=None,
                    )
                    return self.batch_recurrent_layers_forward(hs, **filtered_kwargs)

                l_end_output_cache = torch_ckpt.checkpoint(
                    _ckpt_bfa_fn,
                    _hs_input,
                    _recurrent_emb,
                    _recurrence_trk,
                    use_reentrant=False,
                ).clone()
            else:
                l_end_output_cache = self.batch_recurrent_layers_forward(
                    _hs_input,
                    **filtered_kwargs
                ).clone()


            # Optional residual from previous recurrent embedding -> next recurrent embedding (cache).
            # Here, the recurrent embedding used for this pass is exactly `l_end_output_caches[-1]`.
            if bool(getattr(self.config, "recurrent_residual_to_recurrent_cache", False)) and (not bool(getattr(self.config, "recurrent_skip_to_l_end", False))):
                residual = l_end_output_caches[-1]
                if bool(getattr(self.config, "recurrent_residual_to_recurrent_cache_detach", False)):
                    residual = residual.detach()
                if residual.device != l_end_output_cache.device:
                    residual = residual.to(l_end_output_cache.device)
                if residual.dtype != l_end_output_cache.dtype:
                    residual = residual.to(dtype=l_end_output_cache.dtype)
                if residual.shape == l_end_output_cache.shape:
                    weight = float(getattr(self.config, "recurrent_residual_to_recurrent_cache_weight", 1.0))
                    if weight != 0.0:
                        # only update the positions where recurrence is happening
                        merged_recurrent_cache = l_end_output_cache + (weight * residual)

                    if bool(getattr(self.config, "recurrent_residual_to_recurrent_cache_post_norm", False)):
                        try:
                            eps = float(getattr(self.config, "recurrent_residual_to_recurrent_cache_post_norm_eps", 1e-6))
                            clamp = float(getattr(self.config, "recurrent_residual_to_recurrent_cache_post_norm_clamp", 5.0))
                            clamp = max(clamp, 1.0)
                            y = merged_recurrent_cache.detach().float()
                            rms_y = torch.sqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
                            scale = (1.0 / rms_y).clamp(min=1.0 / clamp, max=clamp).to(dtype=merged_recurrent_cache.dtype)
                            merged_recurrent_cache = merged_recurrent_cache * scale
                        except Exception:
                            print("Failed to post normalize merged recurrent cache, skipping")
                            pass
                    l_end_output_cache = merged_recurrent_cache * recur_mask + l_end_output_cache * (1 - recur_mask)


            # When d = 0, one recurrence is done, so we have D - 1 - d remaining steps to go
            remaining_steps = approximate_depth - 1 - d

            if remaining_steps == 1:
                logger.debug("[BFA Prefill] Recurrent embedding: ", l_end_output_cache[0, :, :5])
            # Debug-only (avoid spamming training logs).
            logger.debug("Remaining steps: %s", remaining_steps)
            logger.debug("Batch backward approximate depth: %s", self.config.batch_backward_approximate_depth)
            logger.debug("Connection detach: %s", self.config.connection_detach)

            # If the remaining steps is greater than the batch backward approximate depth, we need to detach the cache
            # As a sanity check, if backward depth is 0, which means we detach all the time, then when d = approximate_depth - 2 (last step)
            # We would have remaining steps = D - 1 - (D - 2) = 1 > 0, which means we would detach the cache on the last step
            if self.config.connection_detach or remaining_steps > self.config.batch_backward_approximate_depth:
                # Keep detachment behavior, but don't print during training.
                l_end_output_cache = l_end_output_cache.detach()

            self.reset_recurrent_input()

            try:
                l_end_output_cache_shifted = self.batch_cache_shift(l_end_output_cache)
                l_end_output_cache_shifted = self._apply_packed_boundary_zeros(l_end_output_cache_shifted, position_ids)
            except Exception as e:
                logger.error(f"Error shifting cache: {e} during batch forward at depth {d}")
                raise e
            
            l_end_output_caches.append(l_end_output_cache_shifted)

            # --- Memory-efficient cache: drop old detached references ---
            if getattr(self.config, "bfa_memory_efficient_cache", False):
                # Only the latest cache is needed for the next iteration.
                # Earlier caches that were detached can be freed.  Non-detached
                # tensors remain alive via the autograd graph even without a
                # Python reference.
                for ci in range(len(l_end_output_caches) - 1):
                    l_end_output_caches[ci] = None

        self.set_recurrent_input(
            block_id=self.l_start,
            recurrent_embedding=l_end_output_caches[-1],
            control_flows=recurrence_tracker,
            mixing_module_log_buffer=gate_buffer if record_gating_stats else None,
        )

        kwargs.pop("use_cache", None)
        outputs = self.t2mlr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=not training_flag,
            **kwargs
        )
        kwargs['use_cache'] = not training_flag
        self.reset_recurrent_input()

        if not training_flag:
            self.recurrent_cache = l_end_output_cache[:, -1:, :]

        if record_gating_stats and isinstance(self.layers[self.l_start], BlockWrapper):
            buffer = gate_buffer if gate_buffer is not None else self.layers[self.l_start].mixing_module_log_buffer
            setattr(outputs, "mixing_module_logs", buffer if buffer is not None else {})

        return outputs
        
    def batch_recurrent_layers_forward(self, hidden_states, **kwargs):

        # Loop through the layers from l_start to l_end + 1 to perform the forward pass
        printed = False
        for layer in self.layers[self.l_start:self.l_end + 1]:
            out = layer(hidden_states, **kwargs)
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out
        return hidden_states

    @staticmethod
    def _fix_state_dict_key_on_load_legacy(key: str) -> Tuple[str, bool]:
        """
        Strip common wrapper prefixes (e.g., torch.compile/Accelerate `_orig_mod.`) then
        delegate to the base implementation for legacy renames.
        """
        new_key = key
        changed = False

        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
            changed = True
        if "._orig_mod." in new_key:
            new_key = new_key.replace("._orig_mod.", ".")
            changed = True

        base_key, base_changed = PreTrainedModel._fix_state_dict_key_on_load(new_key)
        return base_key, changed or base_changed
    
    @staticmethod
    def _clean_state_dict_keys_legacy(state_dict: Union[dict, OrderedDict]) -> OrderedDict:
        """
        Normalize checkpoint keys before loading (e.g., strip `_orig_mod.` from torch.compile).
        """
        metadata = getattr(state_dict, "_metadata", None)
        cleaned_state_dict = OrderedDict()

        for key, value in state_dict.items():
            new_key, _ = T2MLRWrapper._fix_state_dict_key_on_load_legacy(key)
            cleaned_state_dict[new_key] = value

        if metadata is not None:
            cleaned_state_dict._metadata = metadata
        return cleaned_state_dict
    
    
