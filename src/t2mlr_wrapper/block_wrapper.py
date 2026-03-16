"""
Block Wrapper for Transformer Models
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from .t2mlr_gate_zoo import T2MLR_Mixing_Module

logger = logging.getLogger(__name__)

class BlockWrapper(nn.Module):
    """
    Lightweight wrapper that injects recurrent information before forwarding to the base block.
    """

    def __init__(
        self,
        model: nn.Module,
        t2mlr_mixing_module: T2MLR_Mixing_Module,
    ):
        super().__init__()
        self.model = model
        self.t2mlr_mixing_module = t2mlr_mixing_module
        self.recurrent_cache = None
        self.control_flow = None
        self.mixing_module_log_buffer = None

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped model for compatibility with different architectures.
        This allows the wrapper to expose attributes like 'attention_type' (Qwen) or other
        model-specific attributes that may be accessed during forward passes.
        """
        # First, try nn.Module's attribute resolution (finds registered modules/params)
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass

        # Fallback: delegate to the wrapped model via the module registry
        modules = object.__getattribute__(self, "_modules")
        wrapped_model = modules.get("model", None)
        if wrapped_model is not None and hasattr(wrapped_model, name):
            return getattr(wrapped_model, name)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def set_recurrent_input(self, recurrent_cache, control_flow, mixing_module_log_buffer):
        self.recurrent_cache = recurrent_cache
        self.control_flow = control_flow
        self.mixing_module_log_buffer = mixing_module_log_buffer

    def update_gating_buffer(self, mixing_module_log, control_flow):
        if self.mixing_module_log_buffer is None or mixing_module_log is None:
            return
        
        assert isinstance(mixing_module_log, dict), f"Mixing module log must be a dictionary, got {type(mixing_module_log)}"
        for key, stats in mixing_module_log.items():
            if key not in self.mixing_module_log_buffer:
                self.mixing_module_log_buffer[key] = []
            # Defensive: buffer entries should be list-like (one ndarray per step). Some callers
            # post-process logs in-place (np.concatenate), which turns the list into an ndarray.
            # Convert back to a list so we can keep appending without crashing.
            if isinstance(self.mixing_module_log_buffer.get(key), np.ndarray):
                self.mixing_module_log_buffer[key] = [self.mixing_module_log_buffer[key]]
            assert stats.shape[:2] == self.control_flow.shape[:2], f"Mixing module log value must have the same batch and sequence length as control flow, got {stats.shape[:2]} and {self.control_flow.shape[:2]} for key {key}"
            
            stats = stats * (control_flow > 1)
            stats = stats.detach().float().cpu().numpy()
            self.mixing_module_log_buffer[key].append(stats)

    def apply_t2mlr_mixing(self, hidden_states):
        assert self.control_flow is not None, "Control flow must be provided when applying T2MLR mixing module."
        control_flow = self.control_flow.to(hidden_states.device).unsqueeze(-1)
        recurrent_cache = self.recurrent_cache.to(hidden_states.device)

        assert recurrent_cache.shape == hidden_states.shape
        assert control_flow.shape[:2] == hidden_states.shape[:2], f"Control flow and hidden states must have the same shape, got {control_flow.shape[:2]} and {hidden_states.shape[:2]}"

        mixed_states, mixing_module_log = self.t2mlr_mixing_module(hidden_states, recurrent_cache)
        # Optional: log hidden-state norms through the same gating buffer mechanism.
        # This is intentionally shape-compatible with `update_gating_buffer()` and can be aggregated
        # by the same trainer logging code.
        if self.mixing_module_log_buffer is not None:
            try:
                # Per-token L2 norms (loggable as (B, T, 1))
                h_in = hidden_states.detach().float().norm(dim=-1, keepdim=True)
                h_rec = recurrent_cache.detach().float().norm(dim=-1, keepdim=True)
                h_out = mixed_states.detach().float().norm(dim=-1, keepdim=True)
                if mixing_module_log is None:
                    mixing_module_log = {}
                mixing_module_log["hidden_norm/input"] = h_in
                mixing_module_log["hidden_norm/recurrent"] = h_rec
                mixing_module_log["hidden_norm/mixed"] = h_out
            except Exception:
                # Best-effort logging only; do not affect forward.
                pass
        # Update the gating buffer if it is not None
        self.update_gating_buffer(mixing_module_log, control_flow)
        return mixed_states * (control_flow > 1) + hidden_states * (control_flow <= 1)

    def forward(self, hidden_states, *args, **kwargs):
        # print(f"Hidden states shape: {hidden_states.shape}")
        if self.recurrent_cache is not None:
            # print(f"Applying T2MLR mixing to hidden states: {hidden_states.shape}")
            hidden_states = self.apply_t2mlr_mixing(hidden_states)
        
        # Hidden states are already on the correct device from the model's forward pass
        # No need to call .to() which would iterate through all parameters
        forward_result = self.model.forward(hidden_states, *args, **kwargs)
        # print(f"Forward result shape: {forward_result.shape}")
        return forward_result

def apply_block_wrapper(
    model,
    t2mlr_mixing_module: T2MLR_Mixing_Module,
    l_start: int = 0,
):
    """Wrap the transformer block at l_start with BlockWrapper.

    Supports:
    - GPT-NeoX-like: model.gpt_neox.layers
    - LLaMA-like: model.model.layers
    - GPT-2-like: model.transformer.h
    
    Args:
        model: The transformer model to wrap.
        t2mlr_mixing_module: Pre-constructed mixing module instance.
        l_start: Layer index to wrap.
    
    Returns:
        The model with the specified layer wrapped.
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        container = model.gpt_neox
        attr_name = "layers"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        container = model.model
        attr_name = "layers"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        container = model.transformer
        attr_name = "h"
    else:
        raise AttributeError(
            "Unsupported model architecture: expected model.gpt_neox.layers, model.model.layers, or transformer.h"
        )

    layers = getattr(container, attr_name)
    layer = layers[l_start]
    logger.debug("Applying block wrapper to layer %d (l_start)", l_start)
    assert not isinstance(layer, BlockWrapper), f"Layer {l_start} is already a BlockWrapper"

    layers[l_start] = BlockWrapper(layer, t2mlr_mixing_module)
    return model
