import torch
from torch import nn
from typing import Tuple, Dict, Type, Optional, Any, List
import json
import numpy as np

import logging
logger = logging.getLogger(__name__)

def _resolve_activation_cls(name: str) -> Type[nn.Module]:
    key = (name or "gelu").strip().lower()
    if key in {"gelu"}:
        return nn.GELU
    if key in {"relu"}:
        return nn.ReLU
    if key in {"silu", "swish"}:
        return nn.SiLU
    if key in {"tanh"}:
        return nn.Tanh
    raise ValueError(f"Unsupported gate_mlp_activation '{name}'. Expected one of: gelu, relu, silu, tanh.")


def _build_gate_projector(
    *,
    input_dim: int,
    output_dim: int,
    proj_type: str,
    mlp_hidden_dim: Optional[int],
    mlp_num_layers: int,
    mlp_activation: str,
    mlp_dropout: float,
    weight_init_gain: float,
    bias_init_prob: Optional[float],
    dtype: torch.dtype,
) -> nn.Module:
    """
    Build a gate projector. For linear: Linear(input_dim->output_dim).
    For mlp: (Linear -> act -> dropout) x (L-1) with final Linear to output_dim.

    If bias_init_prob is provided, initialize the final Linear bias such that sigmoid(bias)=bias_init_prob.
    """
    proj_key = (proj_type or "linear").strip().lower()
    if proj_key not in {"linear", "mlp"}:
        raise ValueError(f"Unsupported gate_proj_type '{proj_type}'. Expected 'linear' or 'mlp'.")

    if proj_key == "linear" or int(mlp_num_layers) <= 1:
        proj = nn.Linear(input_dim, output_dim, bias=True, dtype=dtype)
        if weight_init_gain > 0:
            nn.init.xavier_normal_(proj.weight, gain=weight_init_gain)
        else:
            nn.init.zeros_(proj.weight)
        if bias_init_prob is not None:
            with torch.no_grad():
                bias = torch.logit(torch.tensor(_clamp_probability(bias_init_prob), dtype=dtype), eps=1e-6)
                proj.bias.fill_(bias.item())
        return proj

    if mlp_hidden_dim is None:
        mlp_hidden_dim = output_dim
    if mlp_hidden_dim <= 0:
        raise ValueError("gate_mlp_hidden_dim must be positive when gate_proj_type='mlp'.")
    if mlp_num_layers < 2:
        raise ValueError("gate_mlp_num_layers must be >= 2 when gate_proj_type='mlp'.")
    if mlp_dropout < 0.0 or mlp_dropout >= 1.0:
        raise ValueError("gate_mlp_dropout must be in [0.0, 1.0).")

    activation_cls = _resolve_activation_cls(mlp_activation)
    layers: List[nn.Module] = []
    # First layer
    layers.append(nn.Linear(input_dim, mlp_hidden_dim, bias=True, dtype=dtype))
    layers.append(activation_cls())
    if mlp_dropout > 0:
        layers.append(nn.Dropout(p=float(mlp_dropout)))
    # Hidden layers (if any)
    for _ in range(int(mlp_num_layers) - 2):
        layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=True, dtype=dtype))
        layers.append(activation_cls())
        if mlp_dropout > 0:
            layers.append(nn.Dropout(p=float(mlp_dropout)))
    # Output layer
    layers.append(nn.Linear(mlp_hidden_dim, output_dim, bias=True, dtype=dtype))
    proj = nn.Sequential(*layers)

    # Init weights for all Linear layers
    for m in proj.modules():
        if isinstance(m, nn.Linear):
            if weight_init_gain > 0:
                nn.init.xavier_normal_(m.weight, gain=weight_init_gain)
            else:
                nn.init.zeros_(m.weight)

    # Bias-init on final Linear layer (pre-sigmoid)
    if bias_init_prob is not None:
        final_linear = None
        for m in reversed(list(proj.modules())):
            if isinstance(m, nn.Linear):
                final_linear = m
                break
        if final_linear is not None:
            with torch.no_grad():
                bias = torch.logit(torch.tensor(_clamp_probability(bias_init_prob), dtype=dtype), eps=1e-6)
                final_linear.bias.fill_(bias.item())

    return proj


def _ensure_module_device_dtype(module: Optional[nn.Module], *, device: torch.device, dtype: torch.dtype) -> None:
    """
    Best-effort move a module to the target device/dtype.
    Supports both nn.Linear and nn.Sequential (MLP) projectors.
    """
    if module is None:
        return
    try:
        p = next(module.parameters())
    except StopIteration:
        return
    if p.device != device or p.dtype != dtype:
        module.to(device=device, dtype=dtype)

class T2MLR_Mixing_Module(nn.Module):
    """
    Base class for T2MLR mixing modules.
    Handles 2D and 3D tensors; mixing operations are on the last dimension.

    Subclasses should:
      1. Set CONFIG_KEYS to list required/optional config attribute names.
      2. Override from_config() only if custom construction logic is needed.
    """
    CONFIG_KEYS: List[str] = []
    # Keys that are accepted by all mixing modules (and stored on the base class).
    BASE_CONFIG_KEYS: List[str] = [
        "hidden_size",
        "pre_norm_streams",
        "pre_norm_type",
        "post_norm",
        "post_norm_eps",
        "post_norm_clamp",
    ]

    @classmethod
    def from_config(cls, config, **override_kwargs) -> "T2MLR_Mixing_Module":
        """
        Factory method to construct module from config.
        
        Args:
            config: T2MLRConfig instance containing all configuration
            **override_kwargs: Additional configuration overrides
        """
        all_kwargs = config.to_dict()
        all_kwargs.update(override_kwargs)

        # Optional: allow a free-form dict of mixing-module-specific kwargs.
        # Must be a *flat* dict: {"normalize_gates": true, ...}.
        module_kwargs = all_kwargs.get("mixing_module_kwargs", None)
        # If it arrives as a JSON string (e.g., loaded from a config file), parse defensively.
        if isinstance(module_kwargs, str):
            s = module_kwargs.strip()
            if s:
                try:
                    module_kwargs = json.loads(s)
                except Exception:
                    module_kwargs = None
            else:
                module_kwargs = None
        if module_kwargs is not None:
            if not isinstance(module_kwargs, dict):
                raise ValueError(
                    "mixing_module_kwargs must be a dict (flat kwargs). "
                    "Example: {'normalize_gates': True}."
                )
            # Enforce *flat* dict.
            for k, v in module_kwargs.items():
                if isinstance(v, (dict, list)):
                    raise ValueError(
                        f"mixing_module_kwargs must be a flat dict; key '{k}' has non-scalar value type {type(v).__name__}."
                    )
            # Reject incompatible keys early (before module __init__).
            supported = set(cls.CONFIG_KEYS) | set(getattr(cls, "BASE_CONFIG_KEYS", []))
            unsupported = [k for k in module_kwargs.keys() if k not in supported]
            if unsupported:
                raise ValueError(
                    f"Incompatible mixing_module_kwargs for {cls.__name__}: {unsupported}. "
                    f"Supported keys: {sorted(list(supported))}"
                )
            all_kwargs.update(module_kwargs)
        # Pull supported kwargs (module-specific + base keys).
        wanted = list(cls.CONFIG_KEYS) + list(getattr(cls, "BASE_CONFIG_KEYS", []))
        kwargs = {k: all_kwargs[k] for k in wanted if k in all_kwargs}
        logger.info(f"Initializing T2MLR Mixing Module: {cls.__name__} with kwargs: {kwargs}")
        return cls(**kwargs)

    def __init__(
        self,
        hidden_size: int = None,
        pre_norm_streams: bool = False,
        pre_norm_type: str = "rmsnorm",
        post_norm: bool = False,
        post_norm_eps: float = 1e-6,
        post_norm_clamp: float = 5.0,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.pre_norm_streams = bool(pre_norm_streams)
        self.pre_norm_type = (pre_norm_type or "rmsnorm").strip().lower()
        self.post_norm = bool(post_norm)
        self.post_norm_eps = float(post_norm_eps)
        self.post_norm_clamp = float(post_norm_clamp)
        
        # Pre-normalization modules for input and recurrent streams
        self.norm_x = None
        self.norm_r = None
        if self.pre_norm_streams:
            if hidden_size is None:
                raise ValueError("hidden_size must be provided when pre_norm_streams=True")
            if self.pre_norm_type == "rmsnorm":
                self.norm_x = nn.RMSNorm(hidden_size, dtype=dtype)
                self.norm_r = nn.RMSNorm(hidden_size, dtype=dtype)
            elif self.pre_norm_type == "layernorm":
                self.norm_x = nn.LayerNorm(hidden_size, dtype=dtype)
                self.norm_r = nn.LayerNorm(hidden_size, dtype=dtype)
            else:
                raise ValueError(f"Unsupported pre_norm_type '{pre_norm_type}'. Expected 'rmsnorm' or 'layernorm'.")

    def _apply_pre_norm(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply pre-normalization to both input and recurrent streams.
        This separates magnitude normalization from importance weighting (gamma).
        
        Args:
            x: Input stream (full_hidden_states)
            r: Recurrent stream (recurrent_cache or projected_recurrent)
            
        Returns:
            Tuple of (x_normed, r_normed) if pre_norm_streams is enabled,
            otherwise (x, r) unchanged.
        """
        if not self.pre_norm_streams or self.norm_x is None or self.norm_r is None:
            return x, r
        
        target_device = x.device
        target_dtype = x.dtype
        
        # Ensure norm modules are on the correct device/dtype
        _ensure_module_device_dtype(self.norm_x, device=target_device, dtype=target_dtype)
        _ensure_module_device_dtype(self.norm_r, device=target_device, dtype=target_dtype)
        
        x_normed = self.norm_x(x)
        r_normed = self.norm_r(r)
        
        return x_normed, r_normed

    def _apply_post_norm(self, y: torch.Tensor) -> torch.Tensor:
        """
        Optional post-normalization of the gate output (the representation being added to residual stream).
        Normalizes y using RMS norm to stabilize the contribution to the residual stream.
        
        Args:
            y: The mixed representation to normalize (B, T, H) or (B*T, H)
            
        Returns:
            Normalized y if post_norm is enabled, otherwise y unchanged.
        """
        if not self.post_norm:
            return y
        
        try:
            eps = self.post_norm_eps
            clamp = max(self.post_norm_clamp, 1.0)
            # Compute RMS norm from detached y (so this doesn't add extra gradient coupling)
            y_detached = y.detach().float()
            rms_y = torch.sqrt(torch.mean(y_detached * y_detached, dim=-1, keepdim=True) + eps)
            # Normalize to unit RMS, then optionally clamp the scale factor
            # This keeps the magnitude of y controlled before adding to residual
            scale = (1.0 / rms_y).clamp(min=1.0 / clamp, max=clamp).to(dtype=y.dtype)
            return y * scale
        except Exception:
            # Best-effort normalization; if it fails, return y unchanged
            return y

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        module_log = None
        mixed_hidden_states = full_hidden_states    
        return mixed_hidden_states, module_log


# Registry to store gate implementations
_MIXING_MODULE_REGISTRY: Dict[str, Type[T2MLR_Mixing_Module]] = {}

def register_t2mlr_mixing_module(name: str):
    """Decorator to register a T2MLR mixing module class."""
    def decorator(cls: Type[T2MLR_Mixing_Module]):
        _MIXING_MODULE_REGISTRY[name] = cls
        return cls
    return decorator

def get_t2mlr_mixing_module_class(name: str) -> Type[T2MLR_Mixing_Module]:
    if name not in _MIXING_MODULE_REGISTRY:
        raise ValueError(f"T2MLR Mixing Module '{name}' not found. Available: {list(_MIXING_MODULE_REGISTRY.keys())}")
    return _MIXING_MODULE_REGISTRY[name]


@register_t2mlr_mixing_module("none")
class T2MLR_No_Mixing_Module(T2MLR_Mixing_Module):
    """No mixing - passes through full_hidden_states unchanged."""
    CONFIG_KEYS: List[str] = []

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        return full_hidden_states, None


@register_t2mlr_mixing_module("constant_weight")
class T2MLR_Constant_Weight_Mixing_Module(T2MLR_Mixing_Module):
    """Mixing gate with a fixed alpha: out = (1-alpha)*input + alpha*recurrent."""
    CONFIG_KEYS: List[str] = ["recurrent_alpha", "use_learnable_gate", "dtype"]

    def __init__(
        self,
        recurrent_alpha: float = 0.1,
        use_learnable_gate: bool = True,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.use_learnable_gate = use_learnable_gate
        # FSDP requires parameters to be at least 1D; keep scalar semantics via shape (1,)
        alpha_tensor = torch.tensor([recurrent_alpha], dtype=dtype)
        if use_learnable_gate:
            self.recurrent_alpha = nn.Parameter(alpha_tensor)
        else:
            # Register as buffer so it's saved/loaded but not trained
            self.register_buffer("recurrent_alpha", alpha_tensor)

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Apply pre-normalization to separate magnitude from importance
        x_for_mix, r_for_mix = self._apply_pre_norm(full_hidden_states, recurrent_cache)
        
        # NOTE: `recurrent_alpha` is an unconstrained parameter; clamp at runtime for stability.
        # Without this, alpha can drift outside [0, 1] and amplify activations (loss spikes).
        alpha = self.recurrent_alpha.to(full_hidden_states.device).clamp(0.0, 1.0)
        mixed = x_for_mix * (1 - alpha) + r_for_mix * alpha

        gate_stats = (
            torch.ones(
                full_hidden_states.shape[0],
                full_hidden_states.shape[1],
                1,
                dtype=full_hidden_states.dtype,
                device=full_hidden_states.device,
            )
            * alpha
        )
        input_gate_stats = 1.0 - gate_stats
        return mixed, {"recurrent_gate": gate_stats, "input_gate": input_gate_stats}


def _clamp_probability(value: float) -> float:
    prob = float(value)
    return float(min(max(prob, 1e-4), 1 - 1e-4))


def _gate_log_tensor(
    gate: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Normalize a gate tensor to be loggable by `BlockWrapper.update_gating_buffer()`,
    which expects stats.shape[:2] == (B, T).

    - Scalar gates ([]) are expanded to (B, T, 1)
    - (B, T) gates become (B, T, 1)
    - (B, T, ...) gates are left unchanged
    """
    if gate.device != device or gate.dtype != dtype:
        gate = gate.to(device=device, dtype=dtype)
    if gate.ndim == 0:
        return gate.view(1, 1, 1).expand(batch_size, seq_len, 1)
    if gate.ndim == 1:
        # Ambiguous; treat as per-hidden or per-batch vector. For logging, prefer safe broadcast.
        # If it matches hidden dim, broadcasting to (B, T, H) is possible, but we don't know H here.
        # Fallback: expand as a scalar-like gate over (B, T, 1) if size==1, else raise.
        if gate.numel() == 1:
            return gate.reshape(1, 1, 1).expand(batch_size, seq_len, 1)
        raise ValueError(f"Unsupported 1D gate shape for logging: {tuple(gate.shape)}")
    if gate.ndim == 2:
        if gate.shape[0] == batch_size and gate.shape[1] == seq_len:
            return gate.unsqueeze(-1)
        raise ValueError(f"Unsupported 2D gate shape for logging: {tuple(gate.shape)} (expected {(batch_size, seq_len)})")
    # gate.ndim >= 3
    return gate


def _raise_if_nonfinite(t: torch.Tensor, *, name: str) -> None:
    """
    Raise if `t` contains NaN/Inf (fail-fast to catch instabilities early).
    """
    if torch.isfinite(t).all():
        return
    raise FloatingPointError(f"Non-finite detected in {name}")


class RecurrentStateMLP(nn.Module):
    """Residual bottleneck MLP adapter for the recurrent state (initializes close to identity)."""
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        num_layers: int = 2,
        activation_cls: Type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        init_scale: float = 1e-4,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if bottleneck_size <= 0:
            raise ValueError("bottleneck_size must be positive for residual projection adapters")
        if int(num_layers) < 2:
            raise ValueError("num_layers must be >= 2 for RecurrentStateMLP")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)

        layers: List[nn.Module] = []
        layers.append(nn.Linear(hidden_size, bottleneck_size, bias=False, dtype=dtype))
        layers.append(activation_cls())
        if self.dropout_p > 0:
            layers.append(nn.Dropout(p=self.dropout_p))
        # Extra bottleneck layers (if any)
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(bottleneck_size, bottleneck_size, bias=False, dtype=dtype))
            layers.append(activation_cls())
            if self.dropout_p > 0:
                layers.append(nn.Dropout(p=self.dropout_p))
        layers.append(nn.Linear(bottleneck_size, hidden_size, bias=False, dtype=dtype))
        self.net = nn.Sequential(*layers)
        self.scaling = nn.Parameter(torch.tensor(init_scale, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    # QR decomposition for orthogonal init is not implemented for BF16 on CPU
                    w_32 = m.weight.to(torch.float32)
                    nn.init.orthogonal_(w_32)
                    m.weight.copy_(w_32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.net(inputs)
        return inputs + self.scaling * residual


class T2MLR_Base_Projected_Mixer(T2MLR_Mixing_Module):
    """Base class for mixers that may project the recurrent state."""
    CONFIG_KEYS: List[str] = [
        "hidden_size",
        "recurrent_projection_dim",
        "use_recurrent_projection",
        "recurrent_state_proj_type",
        "recurrent_state_mlp_hidden_dim",
        "recurrent_state_mlp_num_layers",
        "recurrent_state_mlp_activation",
        "recurrent_state_mlp_dropout",
        "dtype",
    ]

    def __init__(
        self,
        hidden_size: int,
        recurrent_projection_dim: int = None,
        use_recurrent_projection: bool = False,
        recurrent_state_proj_type: str = "auto",
        recurrent_state_mlp_hidden_dim: Optional[int] = None,
        recurrent_state_mlp_num_layers: int = 2,
        recurrent_state_mlp_activation: str = "gelu",
        recurrent_state_mlp_dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(hidden_size=hidden_size, **kwargs)
        self.hidden_size = hidden_size
        if recurrent_projection_dim is None:
            recurrent_projection_dim = hidden_size
        self.recurrent_projection_dim = recurrent_projection_dim
        self.recurrent_projection = None
        if use_recurrent_projection:
            proj_type = (recurrent_state_proj_type or "auto").strip().lower()
            if proj_type not in {"auto", "linear", "mlp"}:
                raise ValueError(
                    "Unsupported recurrent_state_proj_type "
                    f"'{recurrent_state_proj_type}'. Expected one of: auto, linear, mlp."
                )
            # Back-compat default: infer from dimension.
            if proj_type == "auto":
                proj_type = "linear" if recurrent_projection_dim == hidden_size else "mlp"

            if proj_type == "linear":
                self.recurrent_projection = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
            else:
                bottleneck = recurrent_projection_dim
                if recurrent_state_mlp_hidden_dim is not None:
                    bottleneck = int(recurrent_state_mlp_hidden_dim)
                activation_cls = _resolve_activation_cls(recurrent_state_mlp_activation)
                self.recurrent_projection = RecurrentStateMLP(
                    hidden_size,
                    bottleneck,
                    num_layers=int(recurrent_state_mlp_num_layers),
                    activation_cls=activation_cls,
                    dropout=float(recurrent_state_mlp_dropout),
                    dtype=dtype,
                )
    
    def project_recurrent(
        self,
        recurrent_cache: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device
    ) -> torch.Tensor:
        projected = recurrent_cache
        if recurrent_cache.dtype != target_dtype or recurrent_cache.device != target_device:
            projected = recurrent_cache.to(dtype=target_dtype, device=target_device)
        if self.recurrent_projection is not None:
            _ensure_module_device_dtype(self.recurrent_projection, device=target_device, dtype=target_dtype)
            projected = self.recurrent_projection(projected)
        return projected


@register_t2mlr_mixing_module("concat")
class T2MLR_Concat_Mixing_Module(T2MLR_Base_Projected_Mixer):
    """Concatenates input and recurrent, then projects back to hidden_size."""
    CONFIG_KEYS: List[str] = [
        "hidden_size",
        "recurrent_projection_dim",
        "use_recurrent_projection",
        "recurrent_state_proj_type",
        "dtype",
    ]

    def __init__(
        self,
        hidden_size: int,
        recurrent_projection_dim: int = None,
        use_recurrent_projection: bool = False,
        recurrent_state_proj_type: str = "auto",
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(
            hidden_size,
            recurrent_projection_dim,
            use_recurrent_projection,
            recurrent_state_proj_type=recurrent_state_proj_type,
            dtype=dtype,
            **kwargs,
        )
        self.concat_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False, dtype=dtype)
        with torch.no_grad():
            self.concat_projection.weight[:, :hidden_size] = torch.eye(hidden_size)
            self.concat_projection.weight[:, hidden_size:] = torch.zeros(hidden_size, hidden_size)

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        target_dtype = full_hidden_states.dtype
        target_device = full_hidden_states.device
        projected_recurrent = self.project_recurrent(recurrent_cache, target_dtype, target_device)
        
        # Apply pre-normalization to separate magnitude from importance
        x_for_mix, r_for_mix = self._apply_pre_norm(full_hidden_states, projected_recurrent)
        
        _ensure_module_device_dtype(self.concat_projection, device=target_device, dtype=target_dtype)
        concatenated = torch.cat([x_for_mix, r_for_mix], dim=-1)
        mixed_hidden_states = self.concat_projection(concatenated)
        return mixed_hidden_states, None


@register_t2mlr_mixing_module("gated")
class T2MLR_Gated_Mixing_Module(T2MLR_Base_Projected_Mixer):
    """Learnable gating mechanism for mixing input and recurrent states."""
    CONFIG_KEYS: List[str] = [
        "hidden_size",
        "recurrent_projection_dim",
        "use_recurrent_projection",
        "recurrent_state_proj_type",
        "recurrent_weight",
        "orig_weight",
        "use_learnable_recurrent_gate",
        "use_learnable_input_gate",
        "disable_x_branch",
        "gate_input_detach",
        "recurrent_gate_input_detach",
        "input_gate_input_detach",
        "normalize_gates",
        "raise_on_nonfinite_gates",
        "recurrent_gate_init",
        "input_gate_init",
        "rezero_gamma_recurrent_gate_init",
        "rezero_gamma_input_gate_init",
        "gate_weight_init_std",
        "gate_proj_type",
        "gate_mlp_hidden_dim",
        "gate_mlp_num_layers",
        "gate_mlp_activation",
        "gate_mlp_dropout",
        "use_rezero_residual",
        "rezero_gamma_scalar",
        "rezero_gamma_init",
        "dtype",
    ]

    def __init__(
        self, 
        hidden_size: int, 
        recurrent_projection_dim: int = None, 
        use_recurrent_projection: bool = False,
        recurrent_state_proj_type: str = "auto",
        recurrent_weight: Optional[float] = None,
        orig_weight: Optional[float] = None,
        use_learnable_recurrent_gate: bool = False,
        use_learnable_input_gate: bool = False,
        disable_x_branch: bool = False,
        gate_input_detach: Optional[bool] = None,
        recurrent_gate_input_detach: Optional[bool] = None,
        input_gate_input_detach: Optional[bool] = None,
        normalize_gates: bool = False,
        raise_on_nonfinite_gates: bool = False,
        recurrent_gate_init: Optional[float] = None,
        input_gate_init: Optional[float] = None,
        rezero_gamma_recurrent_gate_init: Optional[float] = 0.0,
        rezero_gamma_input_gate_init: Optional[float] = 0.0,
        gate_weight_init_std: float = 1.0,
        gate_proj_type: str = "linear",
        gate_mlp_hidden_dim: Optional[int] = None,
        gate_mlp_num_layers: int = 2,
        gate_mlp_activation: str = "gelu",
        gate_mlp_dropout: float = 0.0,
        use_rezero_residual: bool = True,
        rezero_gamma_scalar: bool = True,
        rezero_gamma_init: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        super().__init__(
            hidden_size,
            recurrent_projection_dim,
            use_recurrent_projection,
            recurrent_state_proj_type=recurrent_state_proj_type,
            dtype=dtype,
            **kwargs,
        )
        
        self.use_learnable_recurrent_gate = bool(use_learnable_recurrent_gate)
        self.use_learnable_input_gate = bool(use_learnable_input_gate)
        self.disable_x_branch = bool(disable_x_branch)
        # Back-compat: accept unified gate_input_detach and map to both gates.
        # Per-gate flags (if provided) take precedence over the unified alias.
        base_gate_input_detach = bool(gate_input_detach) if gate_input_detach is not None else False
        self.recurrent_gate_input_detach = (
            bool(recurrent_gate_input_detach)
            if recurrent_gate_input_detach is not None
            else base_gate_input_detach
        )
        self.input_gate_input_detach = (
            bool(input_gate_input_detach)
            if input_gate_input_detach is not None
            else base_gate_input_detach
        )

        # If True, renormalize gates so recurrent_gate + input_gate = 1 (in fp32).
        # This prevents amplification when both gates are learned independently.
        self.normalize_gates = bool(normalize_gates)
        # If True, immediately raise if gates contain NaN/Inf (instead of sanitizing them).
        self.raise_on_nonfinite_gates = bool(raise_on_nonfinite_gates)
        self.use_rezero_residual = bool(use_rezero_residual)
        self.rezero_gamma_scalar = bool(rezero_gamma_scalar)

        if rezero_gamma_recurrent_gate_init is None:
            rezero_gamma_recurrent_gate_init = 0.0
        if rezero_gamma_input_gate_init is None:
            rezero_gamma_input_gate_init = 0.0
        if rezero_gamma_init is None:
            rezero_gamma_init = 0.0

        if self.rezero_gamma_scalar:
            recurrent_gamma_init = float(rezero_gamma_recurrent_gate_init if rezero_gamma_recurrent_gate_init is not None else rezero_gamma_init)
            input_gamma_init = float(rezero_gamma_input_gate_init if rezero_gamma_input_gate_init is not None else rezero_gamma_init)
            # Keep gamma params in fp32 to avoid quantization/step effects in low-precision training.
            self.rezero_gamma_recurrent_gate = nn.Parameter(
                torch.tensor([recurrent_gamma_init], dtype=torch.float32)
            )
            self.rezero_gamma_input_gate = nn.Parameter(
                torch.tensor([input_gamma_init], dtype=torch.float32)
            )
        else:
            recurrent_gamma_init = float(rezero_gamma_recurrent_gate_init if rezero_gamma_recurrent_gate_init is not None else rezero_gamma_init)
            input_gamma_init = float(rezero_gamma_input_gate_init if rezero_gamma_input_gate_init is not None else rezero_gamma_init)
            # Keep gamma params in fp32 to avoid quantization/step effects in low-precision training.
            self.rezero_gamma_recurrent_gate = nn.Parameter(
                torch.full((hidden_size,), recurrent_gamma_init, dtype=torch.float32)
            )
            self.rezero_gamma_input_gate = nn.Parameter(
                torch.full((hidden_size,), input_gamma_init, dtype=torch.float32)
            )
        
        gate_input_dim = 2 * hidden_size
        
        # Handle None for random initialization (similar to alpha_coupled)
        if recurrent_gate_init is not None:
            recurrent_gate_init = float(recurrent_gate_init)
        
        if input_gate_init is not None:
            input_gate_init = float(input_gate_init)
        
        self.recurrent_gate_proj = None
        if use_learnable_recurrent_gate:
            self.recurrent_gate_proj = _build_gate_projector(
                input_dim=gate_input_dim,
                output_dim=hidden_size,
                proj_type=gate_proj_type,
                mlp_hidden_dim=gate_mlp_hidden_dim,
                mlp_num_layers=gate_mlp_num_layers,
                mlp_activation=gate_mlp_activation,
                mlp_dropout=gate_mlp_dropout,
                weight_init_gain=float(gate_weight_init_std or 0.0),
                bias_init_prob=recurrent_gate_init,  # None = random, float = targeted
                dtype=dtype,
            )

        self.input_gate_proj = None
        if use_learnable_input_gate:
            self.input_gate_proj = _build_gate_projector(
                input_dim=gate_input_dim,
                output_dim=hidden_size,
                proj_type=gate_proj_type,
                mlp_hidden_dim=gate_mlp_hidden_dim,
                mlp_num_layers=gate_mlp_num_layers,
                mlp_activation=gate_mlp_activation,
                mlp_dropout=gate_mlp_dropout,
                weight_init_gain=float(gate_weight_init_std or 0.0),
                bias_init_prob=input_gate_init,  # None = random, float = targeted
                dtype=dtype,
            )

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        module_log: Dict[str, Any] = {}
        recurrent_weight = kwargs.get('recurrent_weight', 1.0)
        orig_weight = kwargs.get('orig_weight', 0.0)

        target_dtype = full_hidden_states.dtype
        target_device = full_hidden_states.device
        
        projected_recurrent = self.project_recurrent(recurrent_cache, target_dtype, target_device)
        
        # Apply pre-normalization to separate magnitude from importance
        x_for_mix, r_for_mix = self._apply_pre_norm(full_hidden_states, projected_recurrent)
        
        gate_features = None
        if self.use_learnable_recurrent_gate or self.use_learnable_input_gate:
            gate_features = torch.cat([x_for_mix, r_for_mix], dim=-1)

        if self.use_learnable_recurrent_gate:
            _ensure_module_device_dtype(self.recurrent_gate_proj, device=target_device, dtype=target_dtype)
            recurrent_gate_logits = (
                self.recurrent_gate_proj(gate_features)
                if not self.recurrent_gate_input_detach
                else self.recurrent_gate_proj(gate_features.detach())
            )
            recurrent_gate = torch.sigmoid(recurrent_gate_logits)
        else:
            if torch.is_tensor(recurrent_weight):
                recurrent_gate = recurrent_weight.to(dtype=target_dtype, device=target_device)
            else:
                recurrent_gate = torch.tensor(recurrent_weight, dtype=target_dtype, device=target_device)
            
        if self.use_learnable_input_gate:
            _ensure_module_device_dtype(self.input_gate_proj, device=target_device, dtype=target_dtype)
            if self.disable_x_branch:
                input_gate = torch.zeros((), dtype=target_dtype, device=target_device)
            else:
                input_gate_logits = (
                    self.input_gate_proj(gate_features)
                    if not self.input_gate_input_detach
                    else self.input_gate_proj(gate_features.detach())
                )
                input_gate = torch.sigmoid(input_gate_logits)
        else:
            if torch.is_tensor(orig_weight):
                input_gate = (
                    torch.zeros((), dtype=target_dtype, device=target_device)
                    if self.disable_x_branch
                    else orig_weight.to(dtype=target_dtype, device=target_device)
                )
            else:
                input_gate = (
                    torch.tensor(0.0, dtype=target_dtype, device=target_device)
                    if self.disable_x_branch
                    else torch.tensor(orig_weight, dtype=target_dtype, device=target_device)
                )

        if self.raise_on_nonfinite_gates:
            _raise_if_nonfinite(recurrent_gate.to(dtype=torch.float32), name="T2MLR_Gated_Mixing_Module.recurrent_gate")
            _raise_if_nonfinite(input_gate.to(dtype=torch.float32), name="T2MLR_Gated_Mixing_Module.input_gate")

        if (not self.disable_x_branch) and self.normalize_gates:
            # Renormalize in fp32 to avoid amplification (sum > 1).
            denom = (recurrent_gate.to(dtype=torch.float32) + input_gate.to(dtype=torch.float32)).clamp(min=1e-6).detach().float()
            recurrent_gate = (recurrent_gate.to(dtype=torch.float32) / denom).to(dtype=target_dtype)
            input_gate = (input_gate.to(dtype=torch.float32) / denom).to(dtype=target_dtype)

        gamma_recurrent = 1.0
        gamma_input = 1.0
        if self.use_rezero_residual:
            # Bound gamma to [-1, 1] via tanh parameterization.
            gamma_recurrent = torch.tanh(
                self.rezero_gamma_recurrent_gate.to(device=target_device, dtype=torch.float32)
            ).to(dtype=target_dtype)
            gamma_input = torch.tanh(
                self.rezero_gamma_input_gate.to(device=target_device, dtype=torch.float32)
            ).to(dtype=target_dtype)

        # Base gated mixing (this is the "mixed representation" y).
        # Use pre-normalized versions if pre_norm_streams is enabled
        y = r_for_mix * recurrent_gate * gamma_recurrent
        if not self.disable_x_branch:
            y = y + (x_for_mix * input_gate * gamma_input)

        # Optional post-normalization of the gate output
        y = self._apply_post_norm(y)

        mixed_hidden_states = full_hidden_states + y if self.use_rezero_residual else y

        # Expose gates for logging/debugging. These are the raw (unmasked) per-token gates;
        # downstream `BlockWrapper.update_gating_buffer()` masks by `control_flow > 0`.
        bsz, seqlen = full_hidden_states.shape[0], full_hidden_states.shape[1]
        if torch.is_tensor(recurrent_gate):
            module_log["recurrent_gate"] = _gate_log_tensor(
                recurrent_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
            )
        if torch.is_tensor(input_gate):
            module_log["input_gate"] = _gate_log_tensor(
                input_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
            )
        if self.use_rezero_residual:
            # Log ReZero gamma parameters regardless of whether gates are learned; gamma still affects mixing.
            # Expand to (B, T, 1) for scalar gamma and (B, T, H) for per-dim gamma.
            if self.rezero_gamma_recurrent_gate is not None:
                g = torch.tanh(
                    self.rezero_gamma_recurrent_gate.to(device=target_device, dtype=torch.float32)
                ).to(dtype=target_dtype)
                if g.ndim == 1 and g.numel() == 1:
                    g = g.view(1, 1, 1).expand(bsz, seqlen, 1)
                elif g.ndim == 1 and g.numel() == self.hidden_size:
                    g = g.view(1, 1, self.hidden_size).expand(bsz, seqlen, self.hidden_size)
                module_log["rezero_gamma_recurrent_gate"] = _gate_log_tensor(
                    g, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
                )
            if (not self.disable_x_branch) and self.rezero_gamma_input_gate is not None:
                g = torch.tanh(
                    self.rezero_gamma_input_gate.to(device=target_device, dtype=torch.float32)
                ).to(dtype=target_dtype)
                if g.ndim == 1 and g.numel() == 1:
                    g = g.view(1, 1, 1).expand(bsz, seqlen, 1)
                elif g.ndim == 1 and g.numel() == self.hidden_size:
                    g = g.view(1, 1, self.hidden_size).expand(bsz, seqlen, self.hidden_size)
                module_log["rezero_gamma_input_gate"] = _gate_log_tensor(
                    g, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
                )

        return mixed_hidden_states, module_log


@register_t2mlr_mixing_module("var_norm_gated")
class T2MLR_VarNorm_Gated_Mixing_Module(T2MLR_Base_Projected_Mixer):
    """
    Coupled gating mechanism that enforces variance-normalized mixing:

        alpha in (0, 1)
        w_recurrent = sqrt(alpha)
        w_input     = sqrt(1 - alpha)
        out = w_recurrent * recurrent + w_input * input

    This couples the two gates by construction (w_recurrent^2 + w_input^2 = 1).
    """
    CONFIG_KEYS: List[str] = [
        "hidden_size",
        "recurrent_projection_dim",
        "use_recurrent_projection",
        "recurrent_state_proj_type",
        "use_learnable_gate",
        "gate_input_detach",
        "raise_on_nonfinite_gates",
        "recurrent_gate_init",
        "gate_weight_init_std",
        "gate_proj_type",
        "gate_mlp_hidden_dim",
        "gate_mlp_num_layers",
        "gate_mlp_activation",
        "gate_mlp_dropout",
        "use_rezero_residual",
        "rezero_gamma_scalar",
        "rezero_gamma_init",
        "dtype",
    ]

    def __init__(
        self,
        hidden_size: int,
        recurrent_projection_dim: int = None,
        use_recurrent_projection: bool = False,
        recurrent_state_proj_type: str = "auto",
        use_learnable_gate: bool = False,
        gate_input_detach: bool = False,
        raise_on_nonfinite_gates: bool = True,
        recurrent_gate_init: float = None,
        gate_weight_init_std: float = 1.0,
        gate_proj_type: str = "linear",
        gate_mlp_hidden_dim: Optional[int] = None,
        gate_mlp_num_layers: int = 2,
        gate_mlp_activation: str = "gelu",
        gate_mlp_dropout: float = 0.0,
        use_rezero_residual: bool = True,
        rezero_gamma_scalar: bool = True,
        rezero_gamma_init: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            recurrent_projection_dim,
            use_recurrent_projection,
            recurrent_state_proj_type=recurrent_state_proj_type,
            dtype=dtype,
            **kwargs,
        )
        # Back-compat: older configs may still pass `use_learnable_alpha_gate`; treat it as `use_learnable_gate`.
        legacy_alpha_flag = bool(kwargs.pop("use_learnable_alpha_gate", False))
        if legacy_alpha_flag and not use_learnable_gate:
            logger.warning(
                "T2MLR_VarNorm_Gated_Mixing_Module: received deprecated `use_learnable_alpha_gate=True`; "
                "use `use_learnable_gate` instead."
            )
        self.use_learnable_alpha_gate = bool(use_learnable_gate or legacy_alpha_flag)
        self.gate_input_detach = gate_input_detach
        self.raise_on_nonfinite_gates = bool(raise_on_nonfinite_gates)

        self.use_rezero_residual = bool(use_rezero_residual)
        self.rezero_gamma_scalar = bool(rezero_gamma_scalar)

        # ReZero-style residual scaling parameter, initialized to 0.
        # Can be scalar (standard ReZero) or vector (per-dimension scaling).
        # FSDP requires parameters to be at least 1D; use shape (1,) for scalar semantics.
        # FSDP also requires uniform dtype within wrapped modules; use module dtype.
        if self.use_rezero_residual:
            if self.rezero_gamma_scalar:
                # Keep gamma params in fp32 to avoid quantization/step effects in low-precision training.
                self.rezero_gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            else:
                # Keep gamma params in fp32 to avoid quantization/step effects in low-precision training.
                self.rezero_gamma = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        else:
            self.rezero_gamma = None
        if self.rezero_gamma is not None and rezero_gamma_init is not None:
            with torch.no_grad():
                self.rezero_gamma.fill_(float(rezero_gamma_init))

        # Back-compat: accept deprecated `alpha_init` kwarg
        alpha_init = kwargs.pop("alpha_init", None)
        if recurrent_gate_init is None:
            recurrent_gate_init = alpha_init if alpha_init is not None else 0.2

        self.alpha_gate_proj = None
        if self.use_learnable_alpha_gate:
            gate_input_dim = 2 * hidden_size
            self.alpha_gate_proj = _build_gate_projector(
                input_dim=gate_input_dim,
                output_dim=hidden_size,
                proj_type=gate_proj_type,
                mlp_hidden_dim=gate_mlp_hidden_dim,
                mlp_num_layers=gate_mlp_num_layers,
                mlp_activation=gate_mlp_activation,
                mlp_dropout=gate_mlp_dropout,
                weight_init_gain=float(gate_weight_init_std or 0.0),
                bias_init_prob=recurrent_gate_init,
                dtype=dtype,
            )

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        module_log: Dict[str, Any] = {}

        target_dtype = full_hidden_states.dtype
        target_device = full_hidden_states.device

        projected_recurrent = self.project_recurrent(recurrent_cache, target_dtype, target_device)
        
        # Apply pre-normalization to separate magnitude from importance
        x_for_mix, r_for_mix = self._apply_pre_norm(full_hidden_states, projected_recurrent)

        if self.use_learnable_alpha_gate:
            _ensure_module_device_dtype(self.alpha_gate_proj, device=target_device, dtype=target_dtype)
            gate_features = torch.cat([x_for_mix, r_for_mix], dim=-1)
            alpha = torch.sigmoid(self.alpha_gate_proj(gate_features)) if not self.gate_input_detach else torch.sigmoid(self.alpha_gate_proj(gate_features.detach()))
        else:
            # Backwards-compatible kwarg name: use `recurrent_weight` as alpha in (0, 1).
            alpha_value = kwargs.get("recurrent_weight", 0.5)
            if torch.is_tensor(alpha_value):
                alpha = alpha_value.to(dtype=target_dtype, device=target_device)
            else:
                alpha = torch.tensor(alpha_value, dtype=target_dtype, device=target_device)

        # IMPORTANT (bf16 stability):
        # In bfloat16, values very close to 1.0 can round to *exactly* 1.0. If we compute
        # `sqrt(1 - alpha)` in bf16 and `alpha` becomes 1.0, the backward of sqrt hits an
        # infinite slope at 0, which can produce NaN gradients. To avoid this, we do the
        # clamp + sqrt in float32 and only then cast the gates back to `target_dtype`.
        alpha_f32 = alpha.to(dtype=torch.float32)
        if self.raise_on_nonfinite_gates:
            _raise_if_nonfinite(alpha_f32, name="T2MLR_VarNorm_Gated_Mixing_Module.alpha")
        # If alpha is already non-finite, it will poison everything downstream; make it safe.
        alpha_f32 = torch.nan_to_num(alpha_f32, nan=0.5, posinf=1.0, neginf=0.0)
        # Clamp for numerical stability: avoid sqrt(0) and large gradients near the boundaries.
        alpha_f32 = alpha_f32.clamp(min=1e-4, max=1.0 - 1e-4)

        recurrent_gate = torch.sqrt(alpha_f32).to(dtype=target_dtype)
        input_gate = torch.sqrt(1.0 - alpha_f32).to(dtype=target_dtype)

        # Base var_norm_gated mixing (this is the "mixed representation" y).
        # Use pre-normalized versions if pre_norm_streams is enabled
        y = r_for_mix * recurrent_gate + x_for_mix * input_gate

        # Optional post-normalization of the gate output
        y = self._apply_post_norm(y)

        # Optional ReZero residual: out = x + gamma * y
        if self.use_rezero_residual:
            gamma = torch.tanh(self.rezero_gamma.to(device=target_device, dtype=torch.float32))
            mixed_hidden_states = full_hidden_states + (gamma.to(dtype=target_dtype) * y)
        else:
            mixed_hidden_states = y

        # Expose gates for logging/debugging. These are the raw (unmasked) per-token gates;
        # downstream `BlockWrapper.update_gating_buffer()` masks by `control_flow > 0`.
        bsz, seqlen = full_hidden_states.shape[0], full_hidden_states.shape[1]
        if torch.is_tensor(recurrent_gate):
            module_log["recurrent_gate"] = _gate_log_tensor(
                recurrent_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
            )
        if torch.is_tensor(input_gate):
            module_log["input_gate"] = _gate_log_tensor(
                input_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
            )
        if self.use_rezero_residual:
            if self.rezero_gamma_scalar:
                # Scalar: expand to (B, T, 1) for logging
                gamma_val = torch.tanh(
                    self.rezero_gamma.to(device=target_device, dtype=torch.float32)
                ).view(1, 1, 1).expand(bsz, seqlen, 1)
            else:
                # Vector: expand to (B, T, H) for logging
                gamma_val = torch.tanh(
                    self.rezero_gamma.to(device=target_device, dtype=torch.float32)
                ).view(1, 1, self.hidden_size).expand(bsz, seqlen, self.hidden_size)
            module_log["rezero_gamma"] = _gate_log_tensor(
                gamma_val, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
            )

        return mixed_hidden_states, module_log


@register_t2mlr_mixing_module("alpha_coupled")
class T2MLR_Alpha_Coupled_Mixing_Module(T2MLR_Base_Projected_Mixer):
    """
    Simple coupled gating with a single alpha in (0, 1):

        recurrent_gate = alpha
        input_gate     = 1 - alpha
        out = alpha * recurrent + (1 - alpha) * input

    Alpha can be learned from (input, recurrent) features or provided via `recurrent_weight`.
    """
    CONFIG_KEYS: List[str] = [
        "hidden_size",
        "recurrent_projection_dim",
        "use_recurrent_projection",
        "recurrent_state_proj_type",
        "use_learnable_gate",
        "gate_input_detach",
        "raise_on_nonfinite_gates",
        "recurrent_gate_init",
        "gate_weight_init_std",
        "gate_proj_type",
        "gate_mlp_hidden_dim",
        "gate_mlp_num_layers",
        "gate_mlp_activation",
        "gate_mlp_dropout",
        "use_rezero_residual",
        "rezero_gamma_scalar",
        "rezero_gamma_init",
        "dtype",
    ]

    def __init__(
        self,
        hidden_size: int,
        recurrent_projection_dim: int = None,
        use_recurrent_projection: bool = False,
        recurrent_state_proj_type: str = "auto",
        use_learnable_gate: bool = False,
        gate_input_detach: bool = False,
        raise_on_nonfinite_gates: bool = False,
        recurrent_gate_init: float = None,
        gate_weight_init_std: float = 1.0,
        gate_proj_type: str = "linear",
        gate_mlp_hidden_dim: Optional[int] = None,
        gate_mlp_num_layers: int = 2,
        gate_mlp_activation: str = "gelu",
        gate_mlp_dropout: float = 0.0,
        use_rezero_residual: bool = True,
        rezero_gamma_scalar: bool = True,
        rezero_gamma_init: Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            recurrent_projection_dim,
            use_recurrent_projection,
            recurrent_state_proj_type=recurrent_state_proj_type,
            dtype=dtype,
            **kwargs,
        )
        # Back-compat: older configs may still pass `use_learnable_alpha_gate`; treat it as `use_learnable_gate`.
        legacy_alpha_flag = bool(kwargs.pop("use_learnable_alpha_gate", False))
        if legacy_alpha_flag and not use_learnable_gate:
            logger.warning(
                "T2MLR_Alpha_Coupled_Mixing_Module: received deprecated `use_learnable_alpha_gate=True`; "
                "use `use_learnable_gate` instead."
            )
        self.use_learnable_alpha_gate = bool(use_learnable_gate or legacy_alpha_flag)
        self.gate_input_detach = gate_input_detach
        self.raise_on_nonfinite_gates = bool(raise_on_nonfinite_gates)
        self.use_rezero_residual = bool(use_rezero_residual)
        self.rezero_gamma_scalar = bool(rezero_gamma_scalar)

        # ReZero-style residual scaling parameter, initialized to 0.
        # Can be scalar (standard ReZero) or vector (per-dimension scaling).
        # FSDP requires parameters to be at least 1D; use shape (1,) for scalar semantics.
        # FSDP also requires uniform dtype within wrapped modules; use module dtype.
        if self.rezero_gamma_scalar:
            # Keep gamma params in fp32 to avoid quantization/step effects in low-precision training.
            self.rezero_gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            # Keep gamma params in fp32 to avoid quantization/step effects in low-precision training.
            self.rezero_gamma = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        if rezero_gamma_init is not None:
            with torch.no_grad():
                self.rezero_gamma.fill_(float(rezero_gamma_init))

        # Back-compat: accept deprecated `alpha_init` kwarg
        alpha_init = kwargs.pop("alpha_init", None)
        # If recurrent_gate_init is None, use random bias init (bias_init_prob=None)
        # Otherwise, use it to target a specific probability
        if recurrent_gate_init is None:
            # Check if alpha_init was provided as fallback
            if alpha_init is not None:
                recurrent_gate_init = alpha_init
            else:
                # None means random initialization (bias_init_prob=None)
                recurrent_gate_init = None
        else:
            # Explicit value provided, use it for targeted init
            recurrent_gate_init = float(recurrent_gate_init)

        self.alpha_gate_proj = None
        if self.use_learnable_alpha_gate:
            gate_input_dim = 2 * hidden_size
            self.alpha_gate_proj = _build_gate_projector(
                input_dim=gate_input_dim,
                output_dim=hidden_size,
                proj_type=gate_proj_type,
                mlp_hidden_dim=gate_mlp_hidden_dim,
                mlp_num_layers=gate_mlp_num_layers,
                mlp_activation=gate_mlp_activation,
                mlp_dropout=gate_mlp_dropout,
                weight_init_gain=float(gate_weight_init_std or 0.0),
                bias_init_prob=recurrent_gate_init,  # None = random, float = targeted
                dtype=dtype,
            )

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        module_log: Dict[str, Any] = {}

        target_dtype = full_hidden_states.dtype
        target_device = full_hidden_states.device

        projected_recurrent = self.project_recurrent(recurrent_cache, target_dtype, target_device)
        
        # Apply pre-normalization to separate magnitude from importance
        x_for_mix, r_for_mix = self._apply_pre_norm(full_hidden_states, projected_recurrent)

        if self.use_learnable_alpha_gate:
            _ensure_module_device_dtype(self.alpha_gate_proj, device=target_device, dtype=target_dtype)
            gate_features = torch.cat([x_for_mix, r_for_mix], dim=-1)
            alpha = torch.sigmoid(self.alpha_gate_proj(gate_features)) if not self.gate_input_detach else torch.sigmoid(self.alpha_gate_proj(gate_features.detach()))
        else:
            # Backwards-compatible kwarg name: use `recurrent_weight` as alpha in (0, 1).
            alpha_value = kwargs.get("recurrent_weight", 0.5)
            if torch.is_tensor(alpha_value):
                alpha = alpha_value.to(dtype=target_dtype, device=target_device)
            else:
                alpha = torch.tensor(alpha_value, dtype=target_dtype, device=target_device)

        # Match var_norm_gated's numeric safety: clamp in float32 so low-precision dtypes
        # (e.g., bf16) don't introduce non-finite gates, and so NaN/Inf alphas don't
        # poison downstream mixing.
        alpha_f32 = alpha.to(dtype=torch.float32)
        if self.raise_on_nonfinite_gates:
            _raise_if_nonfinite(alpha_f32, name="T2MLR_Alpha_Coupled_Mixing_Module.alpha")
        alpha_f32 = torch.nan_to_num(alpha_f32, nan=0.5, posinf=1.0, neginf=0.0)
        alpha_f32 = alpha_f32.clamp(min=1e-4, max=1.0 - 1e-4)

        recurrent_gate = alpha_f32.to(dtype=target_dtype)
        input_gate = (1.0 - alpha_f32).to(dtype=target_dtype)

        # Base alpha_coupled mixing (this is the "mixed representation" y).
        # Use pre-normalized versions if pre_norm_streams is enabled
        y = r_for_mix * recurrent_gate + x_for_mix * input_gate

        # Optional post-normalization of the gate output
        y = self._apply_post_norm(y)

        # Optional ReZero residual: out = x + gamma * y
        if self.use_rezero_residual:
            gamma = torch.tanh(self.rezero_gamma.to(device=target_device, dtype=torch.float32))
            mixed_hidden_states = full_hidden_states + (gamma.to(dtype=target_dtype) * y)
        else:
            mixed_hidden_states = y

        # Expose gates for logging/debugging. These are the raw (unmasked) per-token gates;
        # downstream `BlockWrapper.update_gating_buffer()` masks by `control_flow > 0`.
        bsz, seqlen = full_hidden_states.shape[0], full_hidden_states.shape[1]
        module_log["recurrent_gate"] = _gate_log_tensor(
            recurrent_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
        )
        module_log["input_gate"] = _gate_log_tensor(
            input_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
        )
        if self.use_rezero_residual:
            if self.rezero_gamma_scalar:
                # Scalar: expand to (B, T, 1) for logging
                gamma_val = torch.tanh(
                    self.rezero_gamma.to(device=target_device, dtype=torch.float32)
                ).view(1, 1, 1).expand(bsz, seqlen, 1)
            else:
                # Vector: expand to (B, T, H) for logging
                gamma_val = torch.tanh(
                    self.rezero_gamma.to(device=target_device, dtype=torch.float32)
                ).view(1, 1, self.hidden_size).expand(bsz, seqlen, self.hidden_size)
            module_log["rezero_gamma"] = _gate_log_tensor(
                gamma_val, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
            )

        return mixed_hidden_states, module_log


@register_t2mlr_mixing_module("exponential_rotational_gated")
class T2MLR_Exponential_Rotational_Gated_Mixing_Module(T2MLR_Mixing_Module):
    """
    Exponential Rotational Gating mechanism:
    
    r_t = sigmoid(W_a * input + b_a)
    i_t = sigmoid(W_x * input + b_x)
    a_t = exp(-c * softplus(Lambda) * r_t)
    out = a_t * current + sqrt(1 - a_t^2) * (i_t * recurrent)

    Gate values initialized to uniformly random in (-1/sqrt(d), 1/sqrt(d)) where d is the dimension of the input.
    """
    CONFIG_KEYS: List[str] = [
        "hidden_size",
        "concat_recurrent",
        "exponential_rotational_gate_c_value",
        "exponential_rotational_gate_r_min",
        "exponential_rotational_gate_r_max",
        "exponential_rotational_gate_sqrt_eps",
        "exponential_rotational_gate_input_gate_input_detach",
        "exponential_rotational_gate_recurrent_gate_input_detach",
        "dtype",
    ]

    def __init__(
        self,
        hidden_size: int,
        concat_recurrent: bool = False,
        exponential_rotational_gate_c_value: float = 8.0,
        exponential_rotational_gate_r_min: float = 0.99,
        exponential_rotational_gate_r_max: float = 0.999,
        exponential_rotational_gate_sqrt_eps: float = 1e-8,
        exponential_rotational_gate_input_gate_input_detach: bool = False,
        exponential_rotational_gate_recurrent_gate_input_detach: bool = False,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(hidden_size=hidden_size, dtype=dtype, **kwargs)
        self.concat_recurrent = concat_recurrent

        print(f"exponential_rotational_gate_input_gate_input_detach: {exponential_rotational_gate_input_gate_input_detach}")
        print(f"exponential_rotational_gate_recurrent_gate_input_detach: {exponential_rotational_gate_recurrent_gate_input_detach}")

        self.input_gate_input_detach = exponential_rotational_gate_input_gate_input_detach
        self.recurrent_gate_input_detach = exponential_rotational_gate_recurrent_gate_input_detach

        self.c_value = exponential_rotational_gate_c_value
        self.r_min = exponential_rotational_gate_r_min
        self.r_max = exponential_rotational_gate_r_max
        self.eps = exponential_rotational_gate_sqrt_eps

        logger.info(f"Initializing Lambda with shape {hidden_size}, concat_recurrent={concat_recurrent}, r_min={self.r_min}, r_max={self.r_max}, c_value={self.c_value}, eps={self.eps}")
        logger.info(f"Input gate input detach: {self.input_gate_input_detach}, recurrent gate input detach: {self.recurrent_gate_input_detach}")
        input_dim = 2 * hidden_size if concat_recurrent else hidden_size
        # r_t projector (W_a)
        self.erg_r_proj = nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype)
        # i_t projector (W_x)
        self.erg_i_proj = nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype)

        # initialize linear layer weights to uniformly random in (-1/sqrt(d), 1/sqrt(d)) where d is the dimension of the input.
        self.erg_r_proj.weight.data.uniform_(-1.0 / torch.sqrt(torch.tensor(input_dim, dtype=dtype)), 1.0 / torch.sqrt(torch.tensor(input_dim, dtype=dtype)))
        self.erg_i_proj.weight.data.uniform_(-1.0 / torch.sqrt(torch.tensor(input_dim, dtype=dtype)), 1.0 / torch.sqrt(torch.tensor(input_dim, dtype=dtype)))
        
        # Lambda parameter
        lambda_init = self.initialize_lambda(hidden_size)
        self.erg_lambda_param = nn.Parameter(lambda_init)
    
    def initialize_lambda(self, hidden_size: int):
        """Initialize Lambda such that exp(-c * softplus(Lambda)) is uniformly distributed in [r_min, r_max]"""
        """Notes: p = exp(-c * log(1 + exp(lambda)) => lambda = log(p^(-1/c) - 1)"""
        
        post_lambda = torch.rand(hidden_size) * (self.r_max - self.r_min) + self.r_min
        lambda_init = torch.log(torch.pow(post_lambda, -1/self.c_value) - 1.0)
        return lambda_init

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        module_log: Dict[str, Any] = {}
        target_dtype = full_hidden_states.dtype
        target_device = full_hidden_states.device
        
        # Apply pre-normalization to separate magnitude from importance
        x_for_mix, r_for_mix = self._apply_pre_norm(full_hidden_states, recurrent_cache)

        gate_inputs = torch.cat([x_for_mix, r_for_mix], dim=-1) if self.concat_recurrent else x_for_mix
        # print(f"gate_inputs: {gate_inputs.shape}")
        
        # Ensure parameters are on correct device/dtype
        _ensure_module_device_dtype(self.erg_r_proj, device=target_device, dtype=target_dtype)
        _ensure_module_device_dtype(self.erg_i_proj, device=target_device, dtype=target_dtype)
        # gate_inputs = gate_inputs.detach()
        
        # Compute r_t and i_t
        r_t = torch.sigmoid(self.erg_r_proj(gate_inputs)) if not self.recurrent_gate_input_detach else torch.sigmoid(self.erg_r_proj(gate_inputs.detach()))
        i_t = torch.sigmoid(self.erg_i_proj(gate_inputs)) if not self.input_gate_input_detach else torch.sigmoid(self.erg_i_proj(gate_inputs.detach()))

        # print(f"r_t: {r_t.mean()}")
        # print(f"i_t: {i_t.mean()}")
        
        # Lambda on device
        a_t = torch.exp(-self.c_value * torch.nn.functional.softplus(self.erg_lambda_param) * r_t)
        # print(f"erg_lambda: {self.erg_lambda_param}")
        a_t = a_t.to(target_dtype)
        coeff_recurrent = torch.sqrt(1.0 - a_t.pow(2) + self.eps).to(target_dtype)

        # print(f"current:\t{a_t.mean()}")
        # print(f"recurrent:\t{coeff_recurrent.mean()}")
        
        # e_{t+1} = a_t * e_{t+1} + sqrt(1 - a_t^2) * (i_t * h_{t+1})
        # Use pre-normalized versions if pre_norm_streams is enabled
        mixed_hidden_states = a_t * x_for_mix + coeff_recurrent * (i_t * r_for_mix)
        
        # Optional post-normalization of the gate output
        mixed_hidden_states = self._apply_post_norm(mixed_hidden_states)
        
        # Logging
        # Use size() instead of shape[] for Dynamo compatibility
        bsz = full_hidden_states.size(0)
        seqlen = full_hidden_states.size(1)
        module_log["r_t"] = _gate_log_tensor(r_t, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device)
        module_log["i_t"] = _gate_log_tensor(i_t, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device)
        module_log["a_t"] = _gate_log_tensor(a_t, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device)
        module_log["coeff_recurrent"] = _gate_log_tensor(coeff_recurrent, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device)
        
        return mixed_hidden_states, module_log


# NOTE: Disabled per request (commented out): two-stage gate.
# @register_t2mlr_mixing_module("two_stage")
# class T2MLR_TwoStage_Mixing_Module(T2MLR_Base_Projected_Mixer):
#     """
#     Two-stage gate mixing module.
    
#     Formula:
#         out = (1 - m) * input + m * (g * recurrent + (1 - g) * input)
#             = input + m * g * (recurrent - input)
    
#     Where:
#         g = sigmoid(gate_proj(cat(input, recurrent)))  (Standard sigmoid gate)
#         m = activation(beta_proj(cat(input, recurrent))) (Linear "turn-on" gate)
    
#     The second gate 'm' is designed to be linear (or near linear) at initialization 0,
#     allowing easier optimization flow ("non-vanishing slope at 0").
#     """
#     CONFIG_KEYS: List[str] = [
#         "hidden_size",
#         "recurrent_projection_dim",
#         "use_recurrent_projection",
#         "recurrent_state_proj_type",
#         "use_learnable_g_gate",
#         "use_learnable_m_gate",
#         "g_gate_input_detach",
#         "m_gate_input_detach",
#         "raise_on_nonfinite_gates",
#         "recurrent_gate_init",
#         "m_gate_activation",
#         "gate_weight_init_std",
#         "gate_proj_type",
#         "gate_mlp_hidden_dim",
#         "gate_mlp_num_layers",
#         "gate_mlp_activation",
#         "gate_mlp_dropout",
#         "dtype",
#     ]

#     def __init__(
#         self,
#         hidden_size: int,
#         recurrent_projection_dim: int = None,
#         use_recurrent_projection: bool = False,
#         recurrent_state_proj_type: str = "auto",
#         use_learnable_g_gate: bool = True,
#         use_learnable_m_gate: bool = True,
#         g_gate_input_detach: bool = False,
#         m_gate_input_detach: bool = False,
#         raise_on_nonfinite_gates: bool = False,
#         recurrent_gate_init: float = 0.0,
#         m_gate_activation: str = "softplus",
#         gate_weight_init_std: float = 1.0,
#         gate_proj_type: str = "linear",
#         gate_mlp_hidden_dim: Optional[int] = None,
#         gate_mlp_num_layers: int = 2,
#         gate_mlp_activation: str = "gelu",
#         gate_mlp_dropout: float = 0.0,
#         dtype: torch.dtype = torch.float32,
#         **kwargs,
#     ):
#         super().__init__(
#             hidden_size,
#             recurrent_projection_dim,
#             use_recurrent_projection,
#             recurrent_state_proj_type=recurrent_state_proj_type,
#             dtype=dtype,
#             **kwargs,
#         )
#         self.use_learnable_g_gate = bool(use_learnable_g_gate)
#         self.use_learnable_m_gate = bool(use_learnable_m_gate)
#         self.g_gate_input_detach = g_gate_input_detach
#         self.m_gate_input_detach = m_gate_input_detach
#         self.raise_on_nonfinite_gates = bool(raise_on_nonfinite_gates)
        
#         self.m_gate_activation = (m_gate_activation or "softplus").strip().lower()
#         if self.m_gate_activation not in {"linear", "tanh", "softplus", "clamp"}:
#             raise ValueError(f"Unsupported m_gate_activation '{m_gate_activation}'. Expected: linear, tanh, softplus, clamp.")

#         gate_input_dim = 2 * hidden_size

#         # 1. First gate 'g' (sigmoid)
#         # "Complete random init like in standard practice" -> bias_init_prob=None
#         self.g_gate_proj = None
#         if self.use_learnable_g_gate:
#             self.g_gate_proj = _build_gate_projector(
#                 input_dim=gate_input_dim,
#                 output_dim=hidden_size,
#                 proj_type=gate_proj_type,
#                 mlp_hidden_dim=gate_mlp_hidden_dim,
#                 mlp_num_layers=gate_mlp_num_layers,
#                 mlp_activation=gate_mlp_activation,
#                 mlp_dropout=gate_mlp_dropout,
#                 weight_init_gain=float(gate_weight_init_std or 0.0),
#                 bias_init_prob=None, 
#                 dtype=dtype,
#             )

#         # 2. Second gate 'm' (target init via recurrent_gate_init)
#         self.m_gate_proj = None
#         if self.use_learnable_m_gate:
#             # We use _build_gate_projector but we handle bias init manually since it expects prob for sigmoid
#             self.m_gate_proj = _build_gate_projector(
#                 input_dim=gate_input_dim,
#                 output_dim=hidden_size,
#                 proj_type=gate_proj_type,
#                 mlp_hidden_dim=gate_mlp_hidden_dim,
#                 mlp_num_layers=gate_mlp_num_layers,
#                 mlp_activation=gate_mlp_activation,
#                 mlp_dropout=gate_mlp_dropout,
#                 weight_init_gain=float(gate_weight_init_std or 0.0),
#                 bias_init_prob=None, # Manual init below
#                 dtype=dtype,
#             )
            
#             # Determine bias init for m based on recurrent_gate_init target
#             target_val = float(recurrent_gate_init)
#             m_bias_init = 0.0
            
#             if self.m_gate_activation == "softplus":
#                 # softplus(b) = target -> b = ln(exp(target) - 1)
#                 if target_val > 1e-6:
#                     m_bias_init = float(np.log(np.expm1(target_val)))
#                 else:
#                     m_bias_init = -10.0 # Close to 0
#             elif self.m_gate_activation == "linear":
#                 m_bias_init = target_val
#             elif self.m_gate_activation == "tanh":
#                 # tanh(b) = target -> b = atanh(target)
#                 t_clamped = min(max(target_val, -0.999), 0.999)
#                 m_bias_init = float(np.arctanh(t_clamped))
#             elif self.m_gate_activation == "clamp":
#                 m_bias_init = target_val

#             with torch.no_grad():
#                 # Find final linear
#                 final_linear = None
#                 if isinstance(self.m_gate_proj, nn.Linear):
#                     final_linear = self.m_gate_proj
#                 else:
#                     for m in reversed(list(self.m_gate_proj.modules())):
#                         if isinstance(m, nn.Linear):
#                             final_linear = m
#                             break
#                 if final_linear is not None:
#                     final_linear.bias.fill_(m_bias_init)

#     def forward(
#         self,
#         full_hidden_states: torch.Tensor,
#         recurrent_cache: torch.Tensor,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
#         module_log: Dict[str, Any] = {}
#         target_dtype = full_hidden_states.dtype
#         target_device = full_hidden_states.device
        
#         projected_recurrent = self.project_recurrent(recurrent_cache, target_dtype, target_device)
        
#         gate_features = None
#         if self.use_learnable_g_gate or self.use_learnable_m_gate:
#             gate_features = torch.cat([full_hidden_states, projected_recurrent], dim=-1)

#         # Compute 'g' gate (sigmoid)
#         if self.use_learnable_g_gate:
#             _ensure_module_device_dtype(self.g_gate_proj, device=target_device, dtype=target_dtype)
#             g_feat = gate_features if not self.g_gate_input_detach else gate_features.detach()
#             g_gate = torch.sigmoid(self.g_gate_proj(g_feat))
#         else:
#             # Fallback to scalar
#             val = kwargs.get("g_gate", 0.5)
#             if torch.is_tensor(val):
#                 g_gate = val.to(dtype=target_dtype, device=target_device)
#             else:
#                 g_gate = torch.tensor(val, dtype=target_dtype, device=target_device)

#         # Compute 'm' gate (linear/other)
#         if self.use_learnable_m_gate:
#             _ensure_module_device_dtype(self.m_gate_proj, device=target_device, dtype=target_dtype)
#             m_feat = gate_features if not self.m_gate_input_detach else gate_features.detach()
#             m_pre = self.m_gate_proj(m_feat)
            
#             if self.m_gate_activation == "linear":
#                 m_gate = m_pre
#             elif self.m_gate_activation == "tanh":
#                 m_gate = torch.tanh(m_pre)
#             elif self.m_gate_activation == "softplus":
#                 m_gate = torch.nn.functional.softplus(m_pre)
#             elif self.m_gate_activation == "clamp":
#                 # Soft constraint 0-1
#                 m_gate = m_pre.clamp(min=0.0, max=1.0)
#             else:
#                 m_gate = m_pre
#         else:
#             val = kwargs.get("m_gate", 0.0)
#             if torch.is_tensor(val):
#                 m_gate = val.to(dtype=target_dtype, device=target_device)
#             else:
#                 m_gate = torch.tensor(val, dtype=target_dtype, device=target_device)

#         # Numeric safety
#         def _stabilize(t, name, min_val=None, max_val=None):
#             t_f32 = t.to(torch.float32)
#             if self.raise_on_nonfinite_gates:
#                 _raise_if_nonfinite(t_f32, name=name)
#             t_f32 = torch.nan_to_num(t_f32, nan=0.0 if min_val is None else min_val, posinf=1.0, neginf=-1.0)
#             if min_val is not None:
#                 t_f32 = t_f32.clamp(min=min_val)
#             if max_val is not None:
#                 t_f32 = t_f32.clamp(max=max_val)
#             return t_f32.to(target_dtype)

#         # g is probability, clamp to (0, 1)
#         g_gate = _stabilize(g_gate, "g_gate", 1e-4, 1.0 - 1e-4)
        
#         # m is potentially unbounded or clamped
#         # If linear/tanh, we just check for non-finite.
#         # If clamp/softplus, we know bounds.
#         m_gate = _stabilize(m_gate, "m_gate") 

#         # out = input + m * g * (recurrent - input)
#         diff = projected_recurrent - full_hidden_states
#         mixed_hidden_states = full_hidden_states + m_gate * g_gate * diff
        
#         # Logging
#         bsz, seqlen = full_hidden_states.shape[0], full_hidden_states.shape[1]
        
#         # Effective recurrent/input gates for unified logging/curriculum.
#         # T2MLR_TwoStage implements out = (1 - m*g)*input + (m*g)*recurrent.
#         recurrent_gate = m_gate * g_gate
#         input_gate = 1.0 - recurrent_gate

#         module_log["recurrent_gate"] = _gate_log_tensor(
#             recurrent_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
#         )
#         module_log["input_gate"] = _gate_log_tensor(
#             input_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
#         )
        
#         # Also log the individual stage gates for specialized analysis.
#         module_log["g_gate"] = _gate_log_tensor(
#             g_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
#         )
#         module_log["m_gate"] = _gate_log_tensor(
#             m_gate, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
#         )

#         return mixed_hidden_states, module_log

# import math
# from typing import Any, Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn

# NOTE: Disabled per request (commented out): attention-based mixing gate.
# @register_t2mlr_mixing_module("two_source_attn_mix_rezero")
# class T2MLR_TwoSource_Attn_Mix_ReZero_Mixing_Module(T2MLR_Mixing_Module):
#     """
#     Variant 1: 2-source softmax mixing across SOURCES (x vs r), multi-head.

#     - Query depends on BOTH x and r (optional concat).
#     - Keys from x and r.
#     - Values from x and r (so alpha_x is used explicitly).
#     - Residual ReZero: out = x + gamma * Wo(mix), gamma init = 0.

#     Shapes:
#       x, r: [B, T, D]
#       logits: [B, T, H, 2]
#       alpha_x/alpha_r: [B, T, H]
#     """

#     CONFIG_KEYS: List[str] = [
#         "hidden_size",
#         "num_heads",
#         "concat_recurrent_for_query",
#         "attn_temperature",
#         "recurrent_logit_bias",
#         "use_layernorm",
#         "logit_clamp_value",
#         "per_head_gamma",
#         "share_value_proj",
#         "dtype",
#     ]

#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int = 16,
#         concat_recurrent_for_query: bool = True,
#         attn_temperature: float = 1.0,
#         recurrent_logit_bias: float = 0.0,
#         use_layernorm: bool = True,
#         logit_clamp_value: float = 0.0,  # 0 => off; else clamp logits to [-v, v]
#         per_head_gamma: bool = True,
#         share_value_proj: bool = False,  # if True, use one Wv for both x and r
#         dtype: torch.dtype = torch.bfloat16,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         assert hidden_size % num_heads == 0, (
#             f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
#         )

#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads

#         self.concat_recurrent_for_query = bool(concat_recurrent_for_query)
#         self.attn_temperature = float(attn_temperature)
#         self.recurrent_logit_bias = float(recurrent_logit_bias)
#         self.use_layernorm = bool(use_layernorm)
#         self.logit_clamp_value = float(logit_clamp_value)
#         self.per_head_gamma = bool(per_head_gamma)
#         self.share_value_proj = bool(share_value_proj)

#         q_in_dim = (2 * hidden_size) if self.concat_recurrent_for_query else hidden_size

#         # Optional normalization (good for bf16 stability)
#         self.ln_x = nn.LayerNorm(hidden_size, elementwise_affine=True, dtype=dtype) if use_layernorm else None
#         self.ln_r = nn.LayerNorm(hidden_size, elementwise_affine=True, dtype=dtype) if use_layernorm else None
#         self.ln_q = nn.LayerNorm(q_in_dim, elementwise_affine=True, dtype=dtype) if use_layernorm else None

#         # Projections
#         self.Wq = nn.Linear(q_in_dim, hidden_size, bias=False, dtype=dtype)

#         # Shared key projection (simple + stable)
#         self.Wk = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

#         # Values: either shared or separate
#         if self.share_value_proj:
#             self.Wv = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
#             self.Wv_x = None
#             self.Wv_r = None
#         else:
#             self.Wv_x = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
#             self.Wv_r = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
#             self.Wv = None

#         self.Wo = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

#         # ReZero gate: start OFF but learnable immediately
#         if self.per_head_gamma:
#             self.gamma = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))  # [H]
#         else:
#             self.gamma = nn.Parameter(torch.zeros((), dtype=torch.float32))         # scalar

#         # Constant recurrent logit bias in fp32
#         self.register_buffer(
#             "bias_rec",
#             torch.tensor(self.recurrent_logit_bias, dtype=torch.float32),
#         )

#     def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
#         # [B,T,D] -> [B,T,H,dh]
#         # Input is guaranteed to be 3D (handled at forward level)
#         # Use reshape with explicit shape tuple to avoid Dynamo issues
#         # Access shape elements directly (more stable than size() for Dynamo)
#         if len(x.shape) != 3:
#             raise ValueError(f"_split_heads expects 3D tensor, got {len(x.shape)}D tensor with shape {x.shape}")
#         B = x.shape[0]
#         T = x.shape[1]
#         return x.view(B, T, self.num_heads, self.head_dim)

#     def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
#         # [B,T,H,dh] -> [B,T,D]
#         # Use size() instead of shape[] indexing for better Dynamo compatibility
#         B = x.size(0)
#         T = x.size(1)
#         H = x.size(2)
#         dh = x.size(3)
#         return x.reshape(B, T, H * dh)

#     def forward(
#         self,
#         full_hidden_states: torch.Tensor,   # x: [B,T,D] or [T,D] during profiling
#         recurrent_cache: torch.Tensor,      # r: [B,T,D] or [T,D] during profiling
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:

#         module_log: Dict[str, Any] = {}
#         x = full_hidden_states
#         r = recurrent_cache

#         # Handle 2D case during profiling: ensure 3D for processing
#         # Store original shape for restoration (works for both 2D and 3D)
#         original_x_shape = x.shape
#         original_ndim = len(original_x_shape)
        
#         # Normalize to 3D for internal processing
#         if original_ndim == 2:
#             x = x.unsqueeze(0)  # [T, D] -> [1, T, D]
#             r = r.unsqueeze(0)  # [T, D] -> [1, T, D]
#         elif original_ndim != 3:
#             raise ValueError(f"forward expects 2D or 3D tensor, got {original_ndim}D")

#         target_dtype = x.dtype
#         target_device = x.device

#         # Ensure module params are on correct device/dtype (your codebase style)
#         _ensure_module_device_dtype(self.Wq, device=target_device, dtype=target_dtype)
#         _ensure_module_device_dtype(self.Wk, device=target_device, dtype=target_dtype)
#         if self.Wv is not None:
#             _ensure_module_device_dtype(self.Wv, device=target_device, dtype=target_dtype)
#         else:
#             _ensure_module_device_dtype(self.Wv_x, device=target_device, dtype=target_dtype)
#             _ensure_module_device_dtype(self.Wv_r, device=target_device, dtype=target_dtype)
#         _ensure_module_device_dtype(self.Wo, device=target_device, dtype=target_dtype)

#         if self.ln_x is not None:
#             _ensure_module_device_dtype(self.ln_x, device=target_device, dtype=target_dtype)
#             _ensure_module_device_dtype(self.ln_r, device=target_device, dtype=target_dtype)
#             _ensure_module_device_dtype(self.ln_q, device=target_device, dtype=target_dtype)

#         # Normalize streams
#         x_n = self.ln_x(x) if self.ln_x is not None else x
#         r_n = self.ln_r(r) if self.ln_r is not None else r

#         # Query depends on BOTH x and r (optional concat)
#         if self.concat_recurrent_for_query:
#             q_in = torch.cat([x_n, r_n], dim=-1)
#         else:
#             q_in = x_n
#         q_in = self.ln_q(q_in) if self.ln_q is not None else q_in

#         # Projections
#         q  = self._split_heads(self.Wq(q_in))   # [B,T,H,dh]
#         kx = self._split_heads(self.Wk(x_n))    # [B,T,H,dh]
#         kr = self._split_heads(self.Wk(r_n))    # [B,T,H,dh]

#         if self.Wv is not None:
#             vx = self._split_heads(self.Wv(x_n))  # [B,T,H,dh]
#             vr = self._split_heads(self.Wv(r_n))  # [B,T,H,dh]
#         else:
#             vx = self._split_heads(self.Wv_x(x_n))
#             vr = self._split_heads(self.Wv_r(r_n))

#         # Dot products per head (source attention)
#         scale = 1.0 / math.sqrt(self.head_dim)
#         lx = (q * kx).sum(dim=-1) * scale    # [B,T,H]
#         lr = (q * kr).sum(dim=-1) * scale    # [B,T,H]
#         lr = lr + self.bias_rec.to(lr.device)

#         # Softmax over 2 sources (fp32)
#         logits = torch.stack([lx, lr], dim=-1).float()  # [B,T,H,2]
#         if self.attn_temperature != 1.0:
#             logits = logits / float(self.attn_temperature)
#         if self.logit_clamp_value and self.logit_clamp_value > 0:
#             v = float(self.logit_clamp_value)
#             logits = logits.clamp(min=-v, max=v)

#         weights = torch.softmax(logits, dim=-1)  # [B,T,H,2] fp32
#         alpha_x = weights[..., 0].to(target_dtype)  # [B,T,H]
#         alpha_r = weights[..., 1].to(target_dtype)  # [B,T,H]

#         # Variant 1: mix values from BOTH streams
#         mix = alpha_x.unsqueeze(-1) * vx + alpha_r.unsqueeze(-1) * vr  # [B,T,H,dh]
#         mix = self.Wo(self._merge_heads(mix))                          # [B,T,D]

#         # ReZero residual: out = x + gamma * mix  (identity at init)
#         if self.per_head_gamma:
#             # apply per-head gamma in head space before merge
#             gamma = self.gamma.to(device=target_device, dtype=torch.float32).view(1, 1, self.num_heads, 1)
#             mix_h = self._split_heads(mix)                 # [B,T,H,dh]
#             mix_h = mix_h * gamma.to(mix_h.dtype)
#             mix = self._merge_heads(mix_h)                 # [B,T,D]
#             out = x + mix
#         else:
#             gamma = self.gamma.to(device=target_device, dtype=torch.float32)
#             out = x + (gamma.to(dtype=target_dtype) * mix)

#         # Logging
#         # Use size() instead of shape[] for Dynamo compatibility
#         bsz = x.size(0)
#         seqlen = x.size(1)
#         module_log["alpha_x"] = _gate_log_tensor(alpha_x, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device)
#         module_log["alpha_r"] = _gate_log_tensor(alpha_r, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device)
        
#         if self.per_head_gamma:
#             # [H] -> [1, 1, H] -> [B, T, H]
#             gamma_val = self.gamma.view(1, 1, -1).expand(bsz, seqlen, -1)
#             module_log["gamma_per_head"] = _gate_log_tensor(
#                 gamma_val, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
#             )
#         else:
#             # scalar -> [1, 1, 1] -> [B, T, 1]
#             gamma_val = self.gamma.view(1, 1, 1).expand(bsz, seqlen, 1)
#             module_log["gamma"] = _gate_log_tensor(
#                 gamma_val, batch_size=bsz, seq_len=seqlen, dtype=target_dtype, device=target_device
#             )

#         # Restore original shape to match input
#         # Use reshape with original shape tuple for Dynamo compatibility
#         if original_ndim == 2:
#             # [1, T, D] -> [T, D] - use view with unpacked shape tuple
#             out = out.view(*original_x_shape)

#         return out, module_log

#----------------------------------------------------------#
# Baseline mixing modules (CODI, Coconut, etc.)
#----------------------------------------------------------#

@register_t2mlr_mixing_module("baseline_coconut")
class T2MLR_Coconut_Mixing_Module(T2MLR_Mixing_Module):
    """Full mixing - passes through recurrent cache unchanged."""
    CONFIG_KEYS: List[str] = []

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Apply pre-normalization (only to recurrent since that's what we return)
        _, r_for_mix = self._apply_pre_norm(full_hidden_states, recurrent_cache)
        mixed_hidden_states = r_for_mix
        # Optional post-normalization of the gate output
        mixed_hidden_states = self._apply_post_norm(mixed_hidden_states)
        return mixed_hidden_states, None

@register_t2mlr_mixing_module("baseline_codi")
class T2MLR_CODI_Mixing_Module(T2MLR_Mixing_Module):
    """Output projected recurrent cache only"""
    CONFIG_KEYS: List[str] = ["hidden_size", "bottleneck_size", "activation_cls", "dtype"]

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        activation_cls: Type[nn.Module] = nn.GELU,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        super().__init__(hidden_size=hidden_size, dtype=dtype, **kwargs)

        self.recurrent_projection = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size, dtype=dtype),
            activation_cls(),
            nn.Linear(bottleneck_size, hidden_size, dtype=dtype),
            nn.LayerNorm(hidden_size, dtype=dtype),
        )

    def forward(
        self,
        full_hidden_states: torch.Tensor,
        recurrent_cache: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Apply pre-normalization to recurrent before projection
        _, r_for_mix = self._apply_pre_norm(full_hidden_states, recurrent_cache)
        projected_recurrent = self.recurrent_projection(r_for_mix)
        # Optional post-normalization of the gate output
        mixed_hidden_states = self._apply_post_norm(projected_recurrent)
        return mixed_hidden_states, None
