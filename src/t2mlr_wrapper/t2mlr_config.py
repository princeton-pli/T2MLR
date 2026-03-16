"""
T2MLR Configuration class for PreTrainedModel integration.
"""

import json
from dataclasses import asdict, dataclass
from typing import Optional, Union, Dict, Any

from transformers import PretrainedConfig


@dataclass
class T2MLRSettings:
    # Core T2MLR settings
    t2mlr_enabled: bool = True
    l_start: int = 0
    l_end: int = -1

    # Batch approximate forward settings
    batch_forward: bool = False
    batch_forward_approximate_depth: int = 1
    batch_backward_approximate_depth: int = 10000000
    eval_batch_forward: bool = False
    
    # Backward Pass settings
    connection_detach: bool = False
    recurrent_weight: float = 1.0
    # Residual on recurrent cache update (previous recurrent -> next recurrent cache)
    recurrent_residual_to_recurrent_cache: bool = False
    recurrent_residual_to_recurrent_cache_weight: float = 1.0
    recurrent_residual_to_recurrent_cache_detach: bool = False
    recurrent_residual_to_recurrent_cache_post_norm: bool = False
    recurrent_residual_to_recurrent_cache_post_norm_eps: float = 1e-6
    recurrent_residual_to_recurrent_cache_post_norm_clamp: float = 5.0

    # Alternative skip target: apply the skip directly to the residual stream at l_end.
    #
    # When enabled, the recurrent embedding injected at l_start (pre-gate) is added to the
    # hidden state output of layer l_end for recurrent positions (control_flow > 1). The
    # recurrent cache is then captured from this modified l_end output, so we do not also
    # add the skip again during cache update.
    recurrent_skip_to_l_end: bool = False
    recurrent_skip_to_l_end_weight: float = 1.0
    recurrent_skip_to_l_end_detach: bool = False
    recurrent_skip_to_l_end_post_norm: bool = False
    recurrent_skip_to_l_end_post_norm_eps: float = 1e-6
    recurrent_skip_to_l_end_post_norm_clamp: float = 5.0

    # Mixing module settings
    recurrent_mixing_module_name: str = "none"

    # Optional: pre-normalize both input and recurrent streams before mixing.
    # This separates magnitude normalization from importance weighting (gamma).
    pre_norm_streams: bool = False
    pre_norm_type: str = "rmsnorm"  # 'rmsnorm' or 'layernorm'

    # Optional: normalize the gate output (representation being added to residual stream) using RMS norm.
    post_norm: bool = False
    post_norm_eps: float = 1e-6
    post_norm_clamp: float = 5.0

    # Constant weight gate settings
    recurrent_alpha: float = 0.5

    # Projection adapter settings
    use_recurrent_projection: bool = False
    recurrent_projection_dim: Optional[int] = None
    # Recurrent-state transform type: 'auto' preserves historical behavior.
    recurrent_state_proj_type: str = "auto"  # 'auto'|'linear'|'mlp'
    # Recurrent-state MLP adapter hyperparameters (used when recurrent_state_proj_type='mlp')
    recurrent_state_mlp_hidden_dim: Optional[int] = None
    recurrent_state_mlp_num_layers: int = 2
    recurrent_state_mlp_activation: str = "gelu"
    recurrent_state_mlp_dropout: float = 0.0
    use_learnable_gate: bool = False
    use_learnable_recurrent_gate: Optional[bool] = None
    use_learnable_input_gate: Optional[bool] = None
    # Gate safety/normalization (used by 'gated', 'alpha_coupled', 'var_norm_gated' mixers)
    # If True, fail-fast when gate values contain NaN/Inf instead of sanitizing/clamping.
    raise_on_nonfinite_gates: bool = False
    # If True (only for 'gated' mixer), renormalize (recurrent_gate + input_gate) == 1 per element.
    normalize_gates: bool = False
    # Extra kwargs for the selected mixing module (merged into the module constructor kwargs).
    # Must be a flat dict; unknown keys should error at module construction time.
    mixing_module_kwargs: Optional[Dict[str, Any]] = None
    recurrent_gate_init: float = 0.2
    input_gate_init: float = 0.8
    gate_weight_init_std: float = 1e-3
    # Learnable gate architecture settings
    gate_proj_type: str = "linear"  # 'linear' or 'mlp'
    gate_mlp_hidden_dim: Optional[int] = None
    gate_mlp_num_layers: int = 2
    gate_mlp_activation: str = "gelu"
    gate_mlp_dropout: float = 0.0
    concat_recurrent: bool = False
    freeze_base_model: bool = False
    use_recurrent_weight_curriculum: bool = False
    recurrent_weight_curriculum_start: float = 0.0
    recurrent_weight_curriculum_end: float = 1.0
    recurrent_weight_curriculum_schedule: str = "linear"
    recurrent_weight_curriculum_warmup_steps: Optional[int] = None
    recurrent_weight_curriculum_warmup_ratio: Optional[float] = None

    # --- Memory optimization settings for BFA ---
    # Gradient checkpointing for intermediate BFA recurrent-layer passes.
    # When True, wraps batch_recurrent_layers_forward in torch.utils.checkpoint
    # during BFA iterations (not the initial/final full model forward).
    # Saves activation memory at the cost of one recompute per BFA iteration.
    bfa_gradient_checkpointing: bool = False
    # Drop intermediate cache references from l_end_output_caches list.
    # Only the latest cache is kept in Python; earlier detached caches are freed.
    # Autograd still retains non-detached tensors via the gradient graph.
    bfa_memory_efficient_cache: bool = False


class T2MLRConfig(PretrainedConfig):
    """
    Configuration class for T2MLR (Recurrent Chain-of-Thought) models.
    
    This class stores configuration for the T2MLR wrapper, which adds recurrent
    connections between transformer layers.
    """

    # IMPORTANT: this must be a stable, T2MLR-specific identifier.
    # Older versions incorrectly used the wrapped base model type here, which breaks
    # checkpoint detection and can cause AutoModel loading failures.
    model_type = "t2mlr"
    
    def __init__(
        self,
        base_config: Optional[dict] = None,
        base_model_type: Optional[str] = None,
        t2mlr_settings: Optional[Union[T2MLRSettings, dict]] = None,
        **kwargs
    ):
        """
        Initialize T2MLR configuration.
        
        Args:
            base_config: Configuration dictionary of the wrapped base model
            base_model_type: Model type of the base model (e.g., "llama", "gpt2")
            t2mlr_settings: T2MLRSettings or dict; values override defaults
            **kwargs: Additional T2MLR-specific parameters or PretrainedConfig kwargs
        """
        t2mlr_overrides = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key in T2MLRSettings.__dataclass_fields__
        }
        t2mlr_values = self._build_t2mlr_settings(t2mlr_settings, t2mlr_overrides)

        super().__init__(**kwargs)
        self.base_config = base_config
        # Keep the wrapped base model type separately (e.g., "llama", "gpt2").
        if base_model_type is None and isinstance(base_config, dict):
            try:
                base_model_type = base_config.get("model_type", None)
            except Exception:
                base_model_type = None
        self.base_model_type = base_model_type
        for key, value in t2mlr_values.items():
            setattr(self, key, value)

    @classmethod
    def from_base_config(cls, base_config: PretrainedConfig, t2mlr_args=None, **kwargs):
        """
        Create T2MLRConfig from a base model config.
        
        Args:
            base_config: Configuration of the base model
            t2mlr_args: T2MLRArguments dataclass instance
            **kwargs: Additional T2MLR-specific parameters
        """
        # Convert base config to dict for storage
        base_config_dict = base_config.to_dict()
        
        config_kwargs = {
            "base_config": base_config_dict,
            "base_model_type": base_config.model_type,
        }
        
        t2mlr_kwargs = cls._t2mlr_kwargs_from_args(t2mlr_args)
        config_kwargs.update(kwargs)
        
        return cls(t2mlr_settings=t2mlr_kwargs, **config_kwargs)
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        """
        output = super().to_dict()
        return output
    
    @staticmethod
    def _build_t2mlr_settings(t2mlr_settings=None, overrides=None):
        data = asdict(T2MLRSettings())
        if t2mlr_settings is not None:
            data.update(asdict(t2mlr_settings) if isinstance(t2mlr_settings, T2MLRSettings) else dict(t2mlr_settings))
        if overrides:
            data.update(overrides)

        use_gate = data["use_learnable_gate"]
        if data["use_learnable_recurrent_gate"] is None:
            data["use_learnable_recurrent_gate"] = use_gate
        if data["use_learnable_input_gate"] is None:
            data["use_learnable_input_gate"] = use_gate
        return data

    @classmethod
    def _t2mlr_kwargs_from_args(cls, t2mlr_args):
        if t2mlr_args is None:
            return {}
        fields = T2MLRSettings.__dataclass_fields__.keys()
        out = {name: getattr(t2mlr_args, name) for name in fields if hasattr(t2mlr_args, name)}
        
        # If recurrent_gate_init is None, keep it as None (allows random bias init)
        # Only default if it's truly missing (not explicitly set to None)
        # Note: This means None will pass through to the mixing module for random init
        # Allow CLI to provide mixing_module_kwargs as a JSON string.
        mmk = out.get("mixing_module_kwargs", None)
        if isinstance(mmk, str):
            s = mmk.strip()
            if s == "":
                out["mixing_module_kwargs"] = None
            else:
                import sys
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError as e:
                    # Enhanced error message with the actual string
                    print(f"[ERROR mixing_module_kwargs] JSON decode failed: {e.msg} at pos {e.pos}", file=sys.stderr)
                    print(f"[ERROR mixing_module_kwargs] String (repr): {repr(s)}", file=sys.stderr)
                    print(f"[ERROR mixing_module_kwargs] String (hex): {s.encode('utf-8').hex()}", file=sys.stderr)
                    raise ValueError(
                        f"Invalid JSON for mixing_module_kwargs. "
                        f"Expected a flat dict, e.g. '{{\"normalize_gates\": true}}'. "
                        f"Received (repr): {repr(s)}, length: {len(s)}, error at position {e.pos}: {e.msg}"
                    ) from e
                except Exception as e:
                    raise ValueError(
                        f"Invalid JSON for mixing_module_kwargs. "
                        f"Expected a flat dict, e.g. '{{\"normalize_gates\": true}}'. "
                        f"Received (repr): {repr(s)}, length: {len(s)}, error: {e}"
                    ) from e
                if parsed is None:
                    out["mixing_module_kwargs"] = None
                elif not isinstance(parsed, dict):
                    raise ValueError(
                        "mixing_module_kwargs must be a JSON object (flat dict), e.g. '{\"normalize_gates\": true}'."
                    )
                else:
                    # Enforce *flat* dict: no nested dict/list values.
                    for k, v in parsed.items():
                        if isinstance(v, (dict, list)):
                            raise ValueError(
                                f"mixing_module_kwargs must be a flat dict; key '{k}' has non-scalar value type {type(v).__name__}."
                            )
                    out["mixing_module_kwargs"] = parsed
        return out
    
    def __getattr__(self, name):
        """
        Delegate attribute access to base_config if attribute not found in T2MLR config.
        This allows the config to expose base model attributes like num_hidden_layers.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # Try to get from base_config
            base_config = super().__getattribute__('base_config')
            if base_config is not None and isinstance(base_config, dict):
                # Direct hit
                if name in base_config:
                    return base_config[name]
                # Common aliases across model families (e.g., GPT-2, LLaMA, GPT-Neo)
                alias_map = {
                    "num_hidden_layers": ["num_hidden_layers", "n_layer", "num_layers"],
                    "hidden_size": ["hidden_size", "n_embd", "d_model"],
                    "num_attention_heads": ["num_attention_heads", "n_head"],
                    "intermediate_size": ["intermediate_size", "n_inner"],
                }
                if name in alias_map:
                    for alt in alias_map[name]:
                        if alt in base_config:
                            return base_config[alt]
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
