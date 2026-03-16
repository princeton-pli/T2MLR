"""
Reusable modules for mixing recurrent and input embeddings.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def resolve_mixer_type(mixer_type: Optional[str], concat_recurrent: bool = False) -> str:
    """
    Normalize mixer type names and fall back to concat flag for backward compatibility.
    """
    if mixer_type is None:
        return "concat" if concat_recurrent else "gated_sum"
    normalized = mixer_type.strip().lower()
    alias_map = {
        "gated": "gated_sum",
        "gated_sum": "gated_sum",
        "sum": "gated_sum",
        "concat": "concat",
        "concatenate": "concat",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported recurrent mixer type '{mixer_type}'.")
    return alias_map[normalized]


@dataclass
class RecurrentMixerConfig:
    hidden_size: int
    projection_dim: int
    mixer_type: str = "gated_sum"
    use_recurrent_projection: bool = False
    use_learnable_recurrent_gate: bool = False
    use_learnable_input_gate: bool = False
    recurrent_gate_init: float = 0.2
    input_gate_init: float = 0.8
    gate_weight_init_std: float = 1e-3


class RecurrentStateMLP(nn.Module):
    """Residual bottleneck MLP adapter for the recurrent state (initializes close to identity)."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        activation_cls: type[nn.Module] = nn.GELU,
        init_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if bottleneck_size <= 0:
            raise ValueError("bottleneck_size must be positive for residual projection adapters")

        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.activation = activation_cls()
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=False)
        self.scaling = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # `nn.init.orthogonal_` uses QR decomposition; on some backends (notably CPU),
        # QR isn't implemented for bf16. Initialize in fp32 and cast back.
        with torch.no_grad():
            for w in (self.down.weight, self.up.weight):
                tmp = torch.empty_like(w, dtype=torch.float32, device=w.device)
                nn.init.orthogonal_(tmp)
                w.copy_(tmp.to(dtype=w.dtype))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.activation(self.down(inputs)))
        # Keep `scaling` in fp32, and do the residual scaling in fp32 even under autocast.
        out_f32 = inputs.to(dtype=torch.float32) + self.scaling * residual.to(dtype=torch.float32)
        return out_f32.to(dtype=inputs.dtype)


class RecurrentInputMixer(nn.Module):
    """
    Combine recurrent and input embeddings using configurable strategies.
    """

    def __init__(
        self,
        config: RecurrentMixerConfig,
        block_id: int,
        reference_dtype: Optional[torch.dtype] = None,
        reference_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_id = block_id
        self.reference_dtype = reference_dtype or torch.float32
        self.reference_device = reference_device
        self._gate_capture_buffer = None

        self.mixer_type = resolve_mixer_type(config.mixer_type)
        self.use_recurrent_projection = config.use_recurrent_projection
        self.use_learnable_recurrent_gate = config.use_learnable_recurrent_gate
        self.use_learnable_input_gate = config.use_learnable_input_gate
        self.recurrent_gate_init = self._clamp_probability(config.recurrent_gate_init)
        self.input_gate_init = self._clamp_probability(config.input_gate_init)
        self.gate_weight_init_std = float(config.gate_weight_init_std or 0.0)

        self.recurrent_projection = None
        if self.use_recurrent_projection:
            if config.projection_dim == config.hidden_size:
                self.recurrent_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                nn.init.eye_(self.recurrent_projection.weight)
            else:
                self.recurrent_projection = RecurrentStateMLP(config.hidden_size, config.projection_dim)
            self._ensure_module_device_dtype(self.recurrent_projection, self.reference_dtype, self.reference_device)

        self.concat_projection = None
        if self.mixer_type == "concat":
            self.concat_projection = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
            with torch.no_grad():
                self.concat_projection.weight[:, :config.hidden_size] = torch.eye(config.hidden_size)
                self.concat_projection.weight[:, config.hidden_size:] = torch.zeros(config.hidden_size, config.hidden_size)
            self._ensure_module_device_dtype(self.concat_projection, self.reference_dtype, self.reference_device)

        gate_input_dim = 2 * config.hidden_size
        self.recurrent_gate_proj = None
        if self.mixer_type == "gated_sum" and self.use_learnable_recurrent_gate:
            self.recurrent_gate_proj = nn.Linear(gate_input_dim, config.hidden_size, bias=True)
            if self.gate_weight_init_std > 0:
                nn.init.xavier_normal_(self.recurrent_gate_proj.weight, gain=self.gate_weight_init_std)
            else:
                nn.init.zeros_(self.recurrent_gate_proj.weight)
            with torch.no_grad():
                bias = torch.logit(torch.tensor(self.recurrent_gate_init, dtype=torch.float32), eps=1e-6)
                self.recurrent_gate_proj.bias.fill_(bias.item())
            self._ensure_module_device_dtype(self.recurrent_gate_proj, self.reference_dtype, self.reference_device)

        self.input_gate_proj = None
        if self.mixer_type == "gated_sum" and self.use_learnable_input_gate:
            self.input_gate_proj = nn.Linear(gate_input_dim, config.hidden_size, bias=True)
            if self.gate_weight_init_std > 0:
                nn.init.xavier_normal_(self.input_gate_proj.weight, gain=self.gate_weight_init_std)
            else:
                nn.init.zeros_(self.input_gate_proj.weight)
            with torch.no_grad():
                bias = torch.logit(torch.tensor(self.input_gate_init, dtype=torch.float32), eps=1e-6)
                self.input_gate_proj.bias.fill_(bias.item())
            self._ensure_module_device_dtype(self.input_gate_proj, self.reference_dtype, self.reference_device)

    @property
    def is_concat(self) -> bool:
        return self.mixer_type == "concat"

    def set_gate_capture_buffer(self, buffer) -> None:
        self._gate_capture_buffer = buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_embedding: torch.Tensor,
        control_flow: torch.Tensor,
        recurrent_weight: Union[torch.Tensor, float] = 1.0,
        orig_weight: Union[torch.Tensor, float] = 0.0,
    ) -> torch.Tensor:
        if recurrent_embedding is None:
            return hidden_states

        if hidden_states.shape[1] != recurrent_embedding.shape[1]:
            raise ValueError(
                "The hidden_states and recurrent_embedding must have the same number of features, "
                f"right now they have {hidden_states.shape[1]} and {recurrent_embedding.shape[1]}"
            )

        target_dtype = hidden_states.dtype
        target_device = hidden_states.device
        projected_embedding = self._project_recurrent_embedding(recurrent_embedding, target_dtype, target_device)

        if control_flow is None:
            raise ValueError("Control flow must be provided when mixing recurrent inputs.")

        if self.mixer_type == "concat":
            return self._concat_mix(hidden_states, projected_embedding, target_dtype, target_device)

        return self._gated_sum_mix(
            hidden_states=hidden_states,
            recurrent_embedding=projected_embedding,
            control_flow=control_flow,
            recurrent_weight=recurrent_weight,
            orig_weight=orig_weight,
            target_dtype=target_dtype,
            target_device=target_device,
        )

    def _project_recurrent_embedding(
        self,
        recurrent_embedding: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        projected = recurrent_embedding
        if recurrent_embedding.dtype != target_dtype or recurrent_embedding.device != target_device:
            projected = recurrent_embedding.to(dtype=target_dtype, device=target_device)

        if self.recurrent_projection is not None:
            self._ensure_module_device_dtype(self.recurrent_projection, target_dtype, target_device)
            projected = self.recurrent_projection(projected)
        else:
            self.reference_device = target_device
            self.reference_dtype = target_dtype
        return projected

    def _concat_mix(
        self,
        hidden_states: torch.Tensor,
        recurrent_embedding: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        if self.concat_projection is None:
            raise RuntimeError("Concat projection is not initialized for concat mixer.")
        self._ensure_module_device_dtype(self.concat_projection, target_dtype, target_device)
        concatenated = torch.cat([hidden_states, recurrent_embedding], dim=-1)
        return self.concat_projection(concatenated)

    def _gated_sum_mix(
        self,
        hidden_states: torch.Tensor,
        recurrent_embedding: torch.Tensor,
        control_flow: torch.Tensor,
        recurrent_weight: Union[torch.Tensor, float],
        orig_weight: Union[torch.Tensor, float],
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        mask, inverse_mask = self._compute_masks(control_flow, target_dtype, target_device)
        recurrent_weight_vec, orig_weight_vec = self._prepare_weight_vectors(
            recurrent_weight, orig_weight, mask, inverse_mask, target_dtype, target_device
        )

        gate_features = None
        needs_gate_features = (
            (self.use_learnable_recurrent_gate and self.recurrent_gate_proj is not None)
            or (self.use_learnable_input_gate and self.input_gate_proj is not None)
        )
        if needs_gate_features:
            gate_features = torch.cat([hidden_states, recurrent_embedding], dim=-1)

        recurrent_gate, recorded_recurrent_gate = self._compute_recurrent_gate(
            gate_features,
            mask,
            recurrent_weight_vec,
            target_dtype,
            target_device,
        )
        input_gate, recorded_input_gate = self._compute_input_gate(
            gate_features,
            mask,
            inverse_mask,
            orig_weight_vec,
            target_dtype,
            target_device,
        )

        mixed_states = recurrent_embedding * recurrent_gate + hidden_states * input_gate
        self._record_gates(recorded_recurrent_gate, recorded_input_gate, control_flow)
        return mixed_states

    def _compute_recurrent_gate(
        self,
        gate_features: Optional[torch.Tensor],
        mask: torch.Tensor,
        recurrent_weight_vec: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_learnable_recurrent_gate and self.recurrent_gate_proj is not None:
            if gate_features is None:
                raise RuntimeError("Gate features must be provided when using a learnable recurrent gate.")
            self._ensure_module_device_dtype(self.recurrent_gate_proj, target_dtype, target_device)
            raw_gate = torch.sigmoid(self.recurrent_gate_proj(gate_features))
            gate = mask.unsqueeze(-1) * raw_gate
            return gate, raw_gate
        return recurrent_weight_vec, recurrent_weight_vec

    def _compute_input_gate(
        self,
        gate_features: Optional[torch.Tensor],
        mask: torch.Tensor,
        inverse_mask: torch.Tensor,
        orig_weight_vec: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_learnable_input_gate and self.input_gate_proj is not None:
            if gate_features is None:
                raise RuntimeError("Gate features must be provided when using a learnable input gate.")
            self._ensure_module_device_dtype(self.input_gate_proj, target_dtype, target_device)
            raw_gate = torch.sigmoid(self.input_gate_proj(gate_features))
            gate = mask.unsqueeze(-1) * raw_gate + inverse_mask.unsqueeze(-1)
            return gate, raw_gate
        return orig_weight_vec, orig_weight_vec

    def _prepare_weight_vectors(
        self,
        recurrent_weight: Union[torch.Tensor, float],
        orig_weight: Union[torch.Tensor, float],
        mask: torch.Tensor,
        inverse_mask: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        recurrent_weight_tensor = self._to_tensor(recurrent_weight, target_dtype, target_device)
        orig_weight_tensor = self._to_tensor(orig_weight, target_dtype, target_device)

        recurrent_weight_vec = torch.clamp(mask.unsqueeze(-1) * recurrent_weight_tensor, min=0.0)
        orig_weight_vec = torch.clamp(mask.unsqueeze(-1) * orig_weight_tensor, min=0.0) + inverse_mask.unsqueeze(-1)
        return recurrent_weight_vec, orig_weight_vec

    def _compute_masks(
        self,
        control_flow: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (control_flow > 0).to(dtype=target_dtype, device=target_device)
        inverse_mask = (control_flow <= 0).to(dtype=target_dtype, device=target_device)
        return mask, inverse_mask

    def _record_gates(
        self,
        recurrent_gate: Optional[torch.Tensor],
        input_gate: Optional[torch.Tensor],
        control_flow: Optional[torch.Tensor],
    ) -> None:
        if self._gate_capture_buffer is None:
            return
        try:
            recurrent_gate_cpu = None if recurrent_gate is None else recurrent_gate.detach().cpu().float()
            input_gate_cpu = None if input_gate is None else input_gate.detach().cpu().float()
            control_cpu = None
            if control_flow is not None and torch.is_tensor(control_flow):
                control_cpu = control_flow.detach().cpu()
            elif control_flow is not None:
                control_cpu = control_flow
            self._gate_capture_buffer.append(
                {
                    "block_id": self.block_id,
                    "gate": recurrent_gate_cpu.tolist() if recurrent_gate_cpu is not None else None,
                    "input_gate": input_gate_cpu.tolist() if input_gate_cpu is not None else None,
                    "control_flow": control_cpu.tolist() if hasattr(control_cpu, "tolist") else control_cpu,
                }
            )
        except Exception:
            logger.debug("Failed to record gate activation for block %s", self.block_id, exc_info=True)

    def _to_tensor(
        self,
        value: Union[torch.Tensor, float],
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=target_dtype, device=target_device)
        return torch.tensor(value, dtype=target_dtype, device=target_device)

    def _clamp_probability(self, value: float) -> float:
        prob = float(value)
        return float(min(max(prob, 1e-4), 1 - 1e-4))

    def _ensure_module_device_dtype(
        self,
        module: Optional[nn.Module],
        target_dtype: Optional[torch.dtype],
        target_device: Optional[torch.device],
    ) -> None:
        if module is None:
            return
        if target_dtype is None and target_device is None:
            return
        module.to(device=target_device, dtype=target_dtype)
        self.reference_device = target_device or self.reference_device
        self.reference_dtype = target_dtype or self.reference_dtype
