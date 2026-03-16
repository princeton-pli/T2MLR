import json
import logging
import re
import torch
from typing import Any, Callable, List, Optional, Tuple, Union

from transformers import Trainer
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.trainer.utils import entropy_from_logits, selective_log_softmax

from components.t2mlr_trainer import (
    is_boosted_param_name,
    _parse_weight_decay_exclusions,
    _is_weight_decay_excluded,
)

logger = logging.getLogger(__name__)


class T2MLRGRPOTrainer(GRPOTrainer):
    """GRPO variant that provides T2MLR control-flow tensors when scoring completions."""

    def __init__(self, *args, gate_lr_multiplier=None, weight_decay_exclusions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gate_lr_multiplier = gate_lr_multiplier
        self._weight_decay_exclusions_raw = weight_decay_exclusions

    def create_optimizer(self):
        """Apply per-parameter LR multipliers for gate/adapter params (mirrors T2MLRTrainer)."""
        if self.optimizer is not None:
            return self.optimizer

        gate_lr_multiplier = self._gate_lr_multiplier
        weight_decay_exclusions = _parse_weight_decay_exclusions(self._weight_decay_exclusions_raw)

        # Parse string → dict or float
        if isinstance(gate_lr_multiplier, str):
            try:
                gate_lr_multiplier = json.loads(gate_lr_multiplier)
            except json.JSONDecodeError:
                try:
                    gate_lr_multiplier = float(gate_lr_multiplier)
                except ValueError:
                    raise ValueError(
                        f"Invalid gate_lr_multiplier: {gate_lr_multiplier}. "
                        "Expected a number or a JSON dictionary."
                    )

        if isinstance(gate_lr_multiplier, dict):
            logger.info("Using per-parameter LR multipliers: %s", gate_lr_multiplier)

            key_to_names = {"None": []}
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                matched_key = None
                for key in gate_lr_multiplier:
                    if re.match(key, name):
                        matched_key = key
                if matched_key is not None:
                    key_to_names.setdefault(matched_key, []).append(name)
                else:
                    key_to_names["None"].append(name)

            optimizer_grouped_parameters = []
            for key, names in key_to_names.items():
                if not names:
                    continue
                multiplier = gate_lr_multiplier.get(key, 1.0) if key != "None" else 1.0
                decay_names, no_decay_names = [], []
                for name in names:
                    if weight_decay_exclusions and _is_weight_decay_excluded(name, weight_decay_exclusions):
                        no_decay_names.append(name)
                    else:
                        decay_names.append(name)
                if decay_names:
                    optimizer_grouped_parameters.append({
                        "params": [self.model.get_parameter(n) for n in decay_names],
                        "lr": self.args.learning_rate * multiplier,
                    })
                    logger.info("Group '%s': %d params, LR mult: %s", key, len(decay_names), multiplier)
                if no_decay_names:
                    optimizer_grouped_parameters.append({
                        "params": [self.model.get_parameter(n) for n in no_decay_names],
                        "lr": self.args.learning_rate * multiplier,
                        "weight_decay": 0.0,
                    })

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        elif gate_lr_multiplier is not None and float(gate_lr_multiplier) != 1.0:
            gate_lr_multiplier = float(gate_lr_multiplier)
            logger.info("Gate LR multiplier: %s", gate_lr_multiplier)

            gate_params, regular_params = [], []
            gate_no_decay, regular_no_decay = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if weight_decay_exclusions and _is_weight_decay_excluded(name, weight_decay_exclusions):
                    (gate_no_decay if is_boosted_param_name(name) else regular_no_decay).append(param)
                elif is_boosted_param_name(name):
                    gate_params.append(param)
                else:
                    regular_params.append(param)

            groups = [
                {"params": regular_params, "lr": self.args.learning_rate},
                {"params": gate_params, "lr": self.args.learning_rate * gate_lr_multiplier},
            ]
            if regular_no_decay:
                groups.append({"params": regular_no_decay, "lr": self.args.learning_rate, "weight_decay": 0.0})
            if gate_no_decay:
                groups.append({"params": gate_no_decay, "lr": self.args.learning_rate * gate_lr_multiplier, "weight_decay": 0.0})

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        else:
            super().create_optimizer()

        return self.optimizer

    def _model_requires_control_flow(self, model: Union[torch.nn.Module, Any]) -> bool:
        """Detect whether the underlying model expects T2MLR control flows."""
        try:
            base_model = self.accelerator.unwrap_model(model)
        except Exception:
            base_model = model
        return bool(getattr(base_model, "t2mlr_enabled", False))

    def _build_control_flows(self, attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
        """Construct T2MLR control-flow values aligned with the attention mask."""
        prompt_length = attention_mask.size(1) - logits_to_keep
        prompt_length = max(prompt_length, 0)

        control_flows = attention_mask.new_zeros(attention_mask.shape)
        if prompt_length > 0:
            prompt_mask = attention_mask[:, :prompt_length]
            control_flows[:, :prompt_length] = torch.where(
                prompt_mask > 0,
                torch.ones_like(prompt_mask),
                torch.zeros_like(prompt_mask),
            )
        if logits_to_keep > 0:
            completion_mask = attention_mask[:, prompt_length:]
            control_flows[:, prompt_length:] = torch.where(
                completion_mask > 0,
                torch.full_like(completion_mask, 2),
                torch.zeros_like(completion_mask),
            )
        return control_flows

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size: Optional[int] = None,
        compute_entropy: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        num_images: Optional[List[int]] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute log probabilities while injecting T2MLR control flows when required."""

        batch_size = batch_size or input_ids.size(0)
        needs_control_flow = self._model_requires_control_flow(model)

        if not needs_control_flow:
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
            )

        if isinstance(logits_to_keep, torch.Tensor):
            logits_to_keep = int(logits_to_keep.item())

        class _ControlFlowWrapper(torch.nn.Module):
            def __init__(self, base_model: torch.nn.Module, builder: Callable[[torch.Tensor, int], torch.Tensor], keep: int):
                super().__init__()
                self.base_model = base_model
                self.builder = builder
                self.keep = keep

            def forward(self, *args: Any, **kwargs: Any):
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is not None:
                    kwargs = dict(kwargs)
                    kwargs["control_flows"] = self.builder(attention_mask, self.keep)
                return self.base_model(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.base_model, name)

        wrapped_model = _ControlFlowWrapper(model, self._build_control_flows, logits_to_keep)

        return super()._get_per_token_logps_and_entropies(
            wrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            num_images=num_images,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            token_type_ids=token_type_ids,
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Trim padded completions so their length matches the longest true sequence."""

        output = super()._generate_and_score_completions(inputs)

        completion_mask = output.get("completion_mask")
        if completion_mask is None:
            return output

        # Determine the longest real completion (tokens up to and including EOS)
        effective_lengths = completion_mask.sum(dim=1)
        if effective_lengths.numel() == 0:
            return output

        max_effective_length = int(effective_lengths.max().item())
        if max_effective_length <= 0 or max_effective_length == completion_mask.size(1):
            return output

        slice_spec = slice(0, max_effective_length)
        output["completion_ids"] = output["completion_ids"][:, slice_spec]
        output["completion_mask"] = completion_mask[:, slice_spec]

        for key in (
            "old_per_token_logps",
            "ref_per_token_logps",
            "importance_sampling_ratio",
            "sampling_per_token_logps",
        ):
            value = output.get(key)
            if value is not None:
                output[key] = value[:, slice_spec]

        return output
