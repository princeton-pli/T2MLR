import math
import os
import tempfile
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, PreTrainedModel
from typing import Dict, Any, Optional, Union, Callable, Tuple
from components.all_arguments import TrainingArguments, RCOTArguments, DataArguments
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
from torch.utils.data.sampler import SequentialSampler, Sampler
# from transformers.utils import logging
import logging
import numpy as np
import json
import torch.distributed as dist
import torch.nn.functional as F

import re
import random

from components.curriculum_scheduler import RecurrentWeightCurriculumScheduler

import datasets
import torch

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

# from IPython import embed


def is_boosted_param_name(name: str) -> bool:
    """
    Return True if a parameter name belongs to RCOT wrapper gates or adapter projection.

    We intentionally exclude base-model MLP gates (e.g., "*.mlp.gate_proj.*").
    
    Matches ALL parameters under rcot_mixing_module to ensure all gate types are covered:
    - Gate projections: recurrent_gate_proj, input_gate_proj, alpha_gate_proj, erg_r_proj, erg_i_proj
    - ReZero parameters: rezero_gamma, gamma
    - Attention projections: Wq, Wk, Wv, Wv_x, Wv_r, Wo
    - Other gate params: erg_lambda_param, recurrent_alpha, concat_projection
    - Recurrent projections: recurrent_projection (and all its submodules)
    """
    if '.mlp.gate_proj.' in name:
        return False
    
    # Match ALL parameters under rcot_mixing_module (covers all gate types in the zoo)
    if '.rcot_mixing_module.' in name:
        return True
    
    # Also match legacy patterns for backward compatibility
    return (
        ('.recurrent_gate_proj.' in name)
        or ('.input_gate_proj.' in name)
        or ('.alpha_gate_proj.' in name)
        or ('.recurrent_projection.' in name)
        or ('concat_projection' in name)
    )

def _parse_weight_decay_exclusions(raw: Optional[Union[str, list]]) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip() != ""]
    if isinstance(raw, str):
        s = raw.strip()
        if s == "":
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x).strip() != ""]
        except json.JSONDecodeError:
            pass
        return [t for t in re.split(r"[,\s]+", s) if t != ""]
    raise ValueError("weight_decay_exclusions must be a list or string (JSON list or comma/space-separated)")

def _is_weight_decay_excluded(name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if re.match(pattern, name):
            return True
    return False

class RCOTTrainer(Trainer):
    """
    Custom trainer that supports control flow as part of the input during training.
    """
    
    def __init__(
        self,
        model: PreTrainedModel, # A model of type RCOTWrapper
        training_args: TrainingArguments,
        rcot_args: RCOTArguments,
        data_args: DataArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        train_data_collator: Optional[Callable] = None,
        eval_data_collator: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            model, 
            training_args,
            train_data_collator,
            train_dataset, 
            eval_dataset, 
            **kwargs
        )
        # Override model_accepts_loss_kwargs to False because our custom compute_loss
        # does NOT use num_items_in_batch for loss scaling. This tells the HF Trainer
        # to apply its standard gradient accumulation scaling (divide loss by GA steps).
        self.model_accepts_loss_kwargs = False
        
        # Only emit custom logs on rank 0 to avoid duplicate output
        self.verbose = self.is_world_process_zero()
        self.rcot_args = rcot_args
        self.data_args = data_args
        self.train_data_collator = train_data_collator
        self.eval_data_collator = eval_data_collator
        self.collator_configs = {}  # Dictionary to store step-specific collator configs

        # Optional: sample BFAD (batch_forward_approximate_depth) each training batch.
        self._bfad_depth_values: Optional[list[int]] = None
        self._last_bfad_depth: Optional[int] = None
        self._bfad_batch_counter: int = 0  # Counter for batches within a step
        self._bfad_last_step: int = -1  # Track last step to reset batch counter
        self._init_bfad_sampler()
        
        # Log RCOT and data arguments
        if self.verbose:
            logger.info("=" * 50)
            logger.info("RCOT Trainer Initialization")
            logger.info("=" * 50)
            logger.info(f"RCOT Arguments: {rcot_args}")
            logger.info(f"Data Arguments: {data_args}")
            logger.info("=" * 50)
        
        # Initialize curriculum scheduler for recurrent_weight if enabled
        self.recurrent_weight_scheduler = None
        if rcot_args.use_recurrent_weight_curriculum:
            logger.info("Initializing recurrent_weight curriculum scheduler")
            
            # Calculate total training steps
            if train_dataset is not None:
                num_train_samples = len(train_dataset)
                batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
                steps_per_epoch = num_train_samples // (batch_size * training_args.world_size)
                total_steps = int(steps_per_epoch * training_args.num_train_epochs)
            else:
                # Fallback: use max_steps if available
                total_steps = training_args.max_steps if training_args.max_steps > 0 else 1000
            
            # Determine warmup steps
            warmup_steps = rcot_args.recurrent_weight_curriculum_warmup_steps
            if warmup_steps is None and rcot_args.recurrent_weight_curriculum_warmup_ratio is not None:
                warmup_steps = int(total_steps * rcot_args.recurrent_weight_curriculum_warmup_ratio)
            if warmup_steps is None:
                warmup_steps = total_steps
            
            # Create scheduler
            self.recurrent_weight_scheduler = RecurrentWeightCurriculumScheduler(
                start_value=rcot_args.recurrent_weight_curriculum_start,
                end_value=rcot_args.recurrent_weight_curriculum_end,
                total_steps=total_steps,
                schedule=rcot_args.recurrent_weight_curriculum_schedule,
                warmup_steps=warmup_steps,
            )
            
            # Set initial recurrent_weight
            initial_weight = self.recurrent_weight_scheduler.get_value(0)
            self._update_model_recurrent_weight(initial_weight)
            
            logger.info(f"Curriculum scheduler initialized: {self.recurrent_weight_scheduler}")
            logger.info(f"Initial recurrent_weight: {initial_weight}")

        # Initialize schedule for pause-token replacement probability if enabled
        self.pause_token_replace_prob_scheduler = None
        replace_schedule = str(getattr(data_args, "pause_token_replace_prob_schedule", "none") or "none").strip().lower()
        replace_start = getattr(data_args, "pause_token_replace_prob", None)
        replace_end = getattr(data_args, "pause_token_replace_prob_end", None)
        if replace_schedule not in {"", "none", "linear"}:
            raise ValueError("pause_token_replace_prob_schedule must be 'none' or 'linear'.")
        if replace_schedule not in {"", "none"} or replace_end is not None:
            start_value = float(replace_start) if replace_start is not None else 0.0
            end_value = float(replace_end) if replace_end is not None else start_value

            if train_dataset is not None:
                num_train_samples = len(train_dataset)
                batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
                steps_per_epoch = num_train_samples // (batch_size * training_args.world_size)
                total_steps = int(steps_per_epoch * training_args.num_train_epochs)
            else:
                total_steps = training_args.max_steps if training_args.max_steps > 0 else 1000

            warmup_steps = getattr(data_args, "pause_token_replace_prob_warmup_steps", None)
            if warmup_steps is None:
                warmup_ratio = getattr(data_args, "pause_token_replace_prob_warmup_ratio", None)
                if warmup_ratio is not None:
                    warmup_steps = int(total_steps * warmup_ratio)
            if warmup_steps is None:
                warmup_steps = total_steps

            schedule = "linear"
            self.pause_token_replace_prob_scheduler = RecurrentWeightCurriculumScheduler(
                start_value=start_value,
                end_value=end_value,
                total_steps=total_steps,
                schedule=schedule,
                warmup_steps=warmup_steps,
            )
            initial_prob = self.pause_token_replace_prob_scheduler.get_value(0)
            self._set_collator_pause_replace_prob(initial_prob)
            logger.info(
                "Pause token replacement prob scheduler initialized: %s (start=%.4f end=%.4f)",
                self.pause_token_replace_prob_scheduler,
                start_value,
                end_value,
            )

    def _init_bfad_sampler(self) -> None:
        """
        Parse rcot_args.batch_forward_approximate_depth_values into a list of candidate depths.

        We intentionally accept a simple string interface because HFArgumentParser handles strings
        robustly across bash scripts.
        
        When a single integer N is provided, samples are drawn from MIN..N, where MIN is 
        controlled by batch_forward_approximate_depth_min (defaults to 1).
        """
        raw = getattr(self.rcot_args, "batch_forward_approximate_depth_values", None)
        if raw is None:
            return
        
        # Get the min value for range-based sampling (defaults to 1)
        min_depth = int(getattr(self.rcot_args, "batch_forward_approximate_depth_min", 1) or 1)
        if min_depth < 1:
            raise ValueError(f"batch_forward_approximate_depth_min must be >= 1, got {min_depth}")
        
        if isinstance(raw, str):
            s = raw.strip()
            if s == "":
                return
            values: list[int] = []
            # Prefer JSON for unambiguous parsing.
            try:
                parsed = json.loads(s)
                if isinstance(parsed, int):
                    # Interpret a single integer as a max depth: sample from MIN..N.
                    n = int(parsed)
                    if n <= 0:
                        raise ValueError("max depth must be > 0")
                    if min_depth > n:
                        raise ValueError(f"batch_forward_approximate_depth_min ({min_depth}) must be <= max ({n})")
                    values = list(range(min_depth, n + 1))
                elif isinstance(parsed, list):
                    values = [int(x) for x in parsed]
                else:
                    raise ValueError("batch_forward_approximate_depth_values JSON must be a list")
            except json.JSONDecodeError:
                # Fallback: comma/space-separated string like "8,16,32" or "8 16 32"
                if re.fullmatch(r"\d+", s):
                    n = int(s)
                    if n <= 0:
                        raise ValueError("max depth must be > 0")
                    if min_depth > n:
                        raise ValueError(f"batch_forward_approximate_depth_min ({min_depth}) must be <= max ({n})")
                    values = list(range(min_depth, n + 1))
                else:
                    tokens = [t.strip() for t in re.split(r"[,\s]+", s) if t.strip() != ""]
                    values = [int(t) for t in tokens]

            values = [v for v in values if int(v) > 0]
            if len(values) == 0:
                raise ValueError(
                    "batch_forward_approximate_depth_values parsed to an empty set. "
                    "Provide a positive int max (e.g. '32') or a list (e.g. '8,16,32' or '[8,16,32]')."
                )
            self._bfad_depth_values = values
        else:
            raise ValueError("batch_forward_approximate_depth_values must be a string (comma-separated or JSON list)")

    def _set_model_bfad_depth(self, depth: int) -> None:
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(base_model, "config"):
            return
        try:
            setattr(base_model.config, "batch_forward_approximate_depth", int(depth))
        except Exception:
            return

    def _sample_bfad_depth_for_step(self, step: int) -> int:
        """Sample BFAD depth for a step (legacy per-step sampling)."""
        assert self._bfad_depth_values is not None and len(self._bfad_depth_values) > 0

        sampling = str(getattr(self.rcot_args, "batch_forward_approximate_depth_sampling", "uniform") or "uniform").strip().lower()
        if sampling not in {"uniform"}:
            raise ValueError(f"Unsupported batch_forward_approximate_depth_sampling: {sampling}. Supported: 'uniform'.")

        # Deterministic across ranks: seed derived from training seed + global_step.
        seed = int(getattr(self.args, "seed", 0) or 0)
        rng = random.Random(seed + 1000003 * int(step))
        return int(rng.choice(self._bfad_depth_values))
    
    def _sample_bfad_depth_for_batch(self, step: int, batch_idx: int) -> int:
        """Sample BFAD depth for a batch (per-batch sampling)."""
        assert self._bfad_depth_values is not None and len(self._bfad_depth_values) > 0

        sampling = str(getattr(self.rcot_args, "batch_forward_approximate_depth_sampling", "uniform") or "uniform").strip().lower()
        if sampling not in {"uniform"}:
            raise ValueError(f"Unsupported batch_forward_approximate_depth_sampling: {sampling}. Supported: 'uniform'.")

        # Deterministic across ranks: seed derived from training seed + global_step + batch_idx.
        seed = int(getattr(self.args, "seed", 0) or 0)
        rng = random.Random(seed + 1000003 * int(step) + 2000007 * int(batch_idx))
        return int(rng.choice(self._bfad_depth_values))

    def create_optimizer(self):
        """
        Setup the optimizer with custom learning rates for gate and adapter parameters.
        
        If gate_lr_multiplier is set, gate projection, recurrent projection adapter,
        and ReZero gamma parameters get a higher learning rate.
        """
        if self.optimizer is None:
            gate_lr_multiplier = getattr(self.rcot_args, 'gate_lr_multiplier', None)
            weight_decay_exclusions = _parse_weight_decay_exclusions(
                getattr(self.rcot_args, "weight_decay_exclusions", None)
            )
            if self.verbose and weight_decay_exclusions:
                logger.info(f"Weight decay exclusions enabled: {weight_decay_exclusions}")

            # If it's a string, try to parse as JSON (for dict) or float (for single value)
            if isinstance(gate_lr_multiplier, str):
                try:
                    gate_lr_multiplier = json.loads(gate_lr_multiplier)
                except json.JSONDecodeError:
                    try:
                        gate_lr_multiplier = float(gate_lr_multiplier)
                    except ValueError:
                        raise ValueError(f"Invalid value for gate_lr_multiplier: {gate_lr_multiplier}. "
                                       f"Expected a number or a JSON dictionary.")

            if isinstance(gate_lr_multiplier, dict):
                if self.verbose:
                    logger.info(f"Using per-parameter learning rate multipliers: {gate_lr_multiplier}")

                key_to_names = {"None": []}

                for name, param in self.model.named_parameters():
                    matched_key = None
                    # Iterate through all patterns (last match wins)
                    # This implements the overwriting scheme: patterns listed later override earlier ones
                    # This allows users to define general rules first, then add specific overrides
                    for key, multiplier in gate_lr_multiplier.items():
                        if re.match(key, name):
                            matched_key = key

                    if matched_key is not None:
                        if matched_key not in key_to_names:
                            key_to_names[matched_key] = []
                        key_to_names[matched_key].append(name)
                    else:
                        key_to_names["None"].append(name)
                
                optimizer_grouped_parameters = []
                for key, names in key_to_names.items():
                    if not names:
                        continue
                    multiplier = gate_lr_multiplier.get(key, 1.0) if key != "None" else 1.0
                    no_decay_names = []
                    decay_names = []
                    if weight_decay_exclusions:
                        for name in names:
                            if _is_weight_decay_excluded(name, weight_decay_exclusions):
                                no_decay_names.append(name)
                            else:
                                decay_names.append(name)
                    else:
                        decay_names = names
                    if decay_names:
                        optimizer_grouped_parameters.append({
                            "params": [self.model.get_parameter(name) for name in decay_names],
                            "lr": self.args.learning_rate * multiplier,
                        })
                        if self.verbose:
                            logger.info(f"Group '{key}': {len(decay_names)} parameters, LR multiplier: {multiplier}")
                    if no_decay_names:
                        optimizer_grouped_parameters.append({
                            "params": [self.model.get_parameter(name) for name in no_decay_names],
                            "lr": self.args.learning_rate * multiplier,
                            "weight_decay": 0.0,
                        })
                        if self.verbose:
                            logger.info(
                                f"Group '{key}' (no weight decay): {len(no_decay_names)} parameters, "
                                f"LR multiplier: {multiplier}"
                            )

                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            elif gate_lr_multiplier is not None and float(gate_lr_multiplier) != 1.0:
                gate_lr_multiplier = float(gate_lr_multiplier)
                if self.verbose:
                    logger.info(f"Setting custom learning rate for gate and adapter parameters with multiplier: {gate_lr_multiplier}")

                # Separate parameters into gate/adapter and regular groups
                gate_params = []
                gate_param_names = []
                gate_no_decay_params = []
                gate_no_decay_names = []
                regular_params = []
                regular_no_decay_params = []
                
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    
                    if weight_decay_exclusions and _is_weight_decay_excluded(name, weight_decay_exclusions):
                        if is_boosted_param_name(name):
                            gate_no_decay_params.append(param)
                            gate_no_decay_names.append(name)
                        else:
                            regular_no_decay_params.append(param)
                        continue

                    # Identify ONLY RCOT wrapper gates and adapter params for LR boost
                    # - Wrapper gates live under: *.recurrent_gate_proj.*, *.input_gate_proj.*, *.alpha_gate_proj.*
                    # - Adapter projection lives under: *.recurrent_projection.*
                    # - ReZero gamma parameters: *.rcot_mixing_module.rezero_gamma, *.rcot_mixing_module.gamma
                    # Explicitly avoid base-model MLP gates like: *.mlp.gate_proj.*
                    if is_boosted_param_name(name):
                        gate_params.append(param)
                        gate_param_names.append(name)
                    else:
                        regular_params.append(param)
                
                # Create parameter groups with different learning rates
                optimizer_grouped_parameters = [
                    {
                        "params": regular_params,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": gate_params,
                        "lr": self.args.learning_rate * gate_lr_multiplier,
                    },
                ]
                if regular_no_decay_params:
                    optimizer_grouped_parameters.append({
                        "params": regular_no_decay_params,
                        "lr": self.args.learning_rate,
                        "weight_decay": 0.0,
                    })
                if gate_no_decay_params:
                    optimizer_grouped_parameters.append({
                        "params": gate_no_decay_params,
                        "lr": self.args.learning_rate * gate_lr_multiplier,
                        "weight_decay": 0.0,
                    })
                
                if self.verbose:
                    logger.info(f"Regular params: {len(regular_params)}, Boosted LR params (gates + adapters): {len(gate_params)}")
                    logger.info(f"Regular LR: {self.args.learning_rate}, Boosted LR: {self.args.learning_rate * gate_lr_multiplier}")
                    logger.info(f"Boosted LR parameter names: {gate_param_names}")
                    if gate_no_decay_names or regular_no_decay_params:
                        logger.info(
                            f"No weight decay params: {len(gate_no_decay_params) + len(regular_no_decay_params)}"
                        )
                
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            else:
                # Use default optimizer creation
                super().create_optimizer()
        
        return self.optimizer

    def get_decay_parameter_names(self, model):
        decay_parameters = super().get_decay_parameter_names(model)
        weight_decay_exclusions = _parse_weight_decay_exclusions(
            getattr(self.rcot_args, "weight_decay_exclusions", None)
        )
        if not weight_decay_exclusions:
            return decay_parameters
        return [
            name for name in decay_parameters
            if not _is_weight_decay_excluded(name, weight_decay_exclusions)
        ]
    
    def _capture_stats_from_model(self, model, control_flows):
        """Helper to extract and aggregate gate stats from the model buffer."""
        base_model = model.module if hasattr(model, "module") else model

        # Post-process if needed (e.g. concatenating list of steps)
        if hasattr(base_model, "post_process_gating_stats"):
            base_model.post_process_gating_stats()

        gate_logs = getattr(base_model, "active_gate_buffer", None) or {}
        if not gate_logs:
            # Always log when gate buffer is empty to help diagnose gate logging issues
            logger.debug(f"Gate buffer is empty at step {getattr(self.state, 'global_step', 'unknown')} - no gate stats to capture")
            return

        gate_stats = {}
        # thresholds for extreme-gate logging
        extreme_eps = float(getattr(self.args, "gate_extreme_eps", 0.01))
        range_tol = float(getattr(self.args, "gate_extreme_range_tol", 0.05))
        
        # Use detach().cpu().numpy() carefully.
        # Doing this on all ranks is fine, but we only store/log on rank 0 usually.
        # However, compute_loss happens on all.
        
        def _as_numpy(arr_like):
            if arr_like is None:
                return None
            # Handle list of arrays/tensors (from multi-step accumulation)
            if isinstance(arr_like, list):
                if len(arr_like) == 0:
                    return None
                arrs = []
                for item in arr_like:
                    if isinstance(item, torch.Tensor):
                        arrs.append(item.detach().float().cpu().numpy())
                    else:
                        arrs.append(np.asarray(item))
                # Concatenate along time dimension (axis=1) for (B, T, ...)
                return np.concatenate(arrs, axis=1) if len(arrs) > 1 else arrs[0]
            if isinstance(arr_like, torch.Tensor):
                return arr_like.detach().float().cpu().numpy()
            return np.asarray(arr_like)

        def _masked_values(arr: np.ndarray, control_flows_np: Optional[np.ndarray]):
            """
            Return values restricted to recurrent positions (control_flow > 1) when possible.
            Uses boolean indexing (not multiply+drop-zeros) so true zeros are preserved.
            """
            if arr is None:
                return None
            if control_flows_np is None:
                return arr
            mask = (control_flows_np > 1)  # (B, T)
            if arr.ndim >= 2 and arr.shape[0] == mask.shape[0] and arr.shape[1] == mask.shape[1]:
                # arr: (B, T) or (B, T, ...)
                return arr[mask]
            return arr

        def _compute_stats(arr_like, name_prefix: str):
            if arr_like is None:
                return
            
            arr = _as_numpy(arr_like)
            if arr is None:
                return

            cf_np = None
            if control_flows is not None:
                cf_np = control_flows.detach().cpu().numpy() if isinstance(control_flows, torch.Tensor) else np.asarray(control_flows)

            masked = _masked_values(arr, cf_np)
            if masked is None:
                return

            # masked is either:
            # - for arr shape (B, T): masked -> (N,)
            # - for arr shape (B, T, H...): masked -> (N, H...)
            # - otherwise: masked is arr unchanged
            vals_flat = np.asarray(masked).reshape(-1)

            # Basic stats
            if vals_flat.size > 0:
                gate_stats[f"{name_prefix}_mean"] = float(vals_flat.mean())
                gate_stats[f"{name_prefix}_std"] = float(vals_flat.std())
                gate_stats[f"{name_prefix}_min"] = float(vals_flat.min())
                gate_stats[f"{name_prefix}_max"] = float(vals_flat.max())

            # Extreme gate stats (only when the tensor looks gate-like in [0, 1])
            # This avoids producing misleading metrics for non-gate tensors (e.g., rezero gamma, norms).
            if vals_flat.size > 0:
                in_range = (vals_flat >= (-range_tol)) & (vals_flat <= (1.0 + range_tol))
                gate_like_share = float(in_range.mean())
                gate_stats[f"{name_prefix}_gate_like_share"] = gate_like_share
                if gate_like_share >= 0.99:
                    near0 = (vals_flat <= extreme_eps)
                    near1 = (vals_flat >= (1.0 - extreme_eps))
                    extreme = near0 | near1
                    gate_stats[f"{name_prefix}_near0_share"] = float(near0.mean())
                    gate_stats[f"{name_prefix}_near1_share"] = float(near1.mean())
                    gate_stats[f"{name_prefix}_extreme_share"] = float(extreme.mean())

                    # Per-dimension extreme shares if the original tensor had a gate dimension.
                    # For (B, T, H...) => masked has shape (N, H...)
                    masked_arr = np.asarray(masked)
                    if masked_arr.ndim >= 2:
                        # Collapse any trailing dims into one "gate dim" axis.
                        n = masked_arr.shape[0]
                        d = int(np.prod(masked_arr.shape[1:]))
                        if n > 0 and d > 1:
                            masked_2d = masked_arr.reshape(n, d)
                            extreme_2d = (masked_2d <= extreme_eps) | (masked_2d >= (1.0 - extreme_eps))
                            per_dim = extreme_2d.mean(axis=0)  # (d,)
                            gate_stats[f"{name_prefix}_extreme_dim_mean"] = float(per_dim.mean())
                            gate_stats[f"{name_prefix}_extreme_dim_p90"] = float(np.quantile(per_dim, 0.90))
                            gate_stats[f"{name_prefix}_extreme_dim_max"] = float(per_dim.max())

        # Compute stats for every logged key (ensures compatibility across all gates in the zoo).
        for k, v in gate_logs.items():
            # Keep the existing naming convention under the 'gate/' namespace.
            safe_k = str(k)
            _compute_stats(v, f"gate/{safe_k}")

        # Derived hidden-norm ratios (helps diagnose norm mismatch directly)
        in_mean = gate_stats.get("gate/hidden_norm/input_mean", None)
        rec_mean = gate_stats.get("gate/hidden_norm/recurrent_mean", None)
        out_mean = gate_stats.get("gate/hidden_norm/mixed_mean", None)
        if in_mean is not None and rec_mean is not None:
            gate_stats["gate/hidden_norm/recurrent_to_input_mean_ratio"] = float(rec_mean / (in_mean + 1e-12))
        if in_mean is not None and out_mean is not None:
            gate_stats["gate/hidden_norm/mixed_to_input_mean_ratio"] = float(out_mean / (in_mean + 1e-12))

        # Store in self for the next log() call
        self._current_gate_stats = gate_stats
        
        if self.verbose and gate_stats:
            logger.debug(f"Captured gate stats: {list(gate_stats.keys())}")
        elif self.verbose:
            logger.debug("No gate stats computed (empty arrays or no valid values)")
        
        # Clean up buffer on model
        setattr(base_model, "active_gate_buffer", None)

    def set_collator_step(self, step: int):
        self.data_collator.set_preprocess_features_config({'global_step': step})
        self.data_collator.set_postprocess_features_config({'global_step': step})
    
    def _update_model_recurrent_weight(self, new_weight: float):
        """Update the model's recurrent_weight to a new value."""
        if hasattr(self.model, 'recurrent_weight'):
            self.model.recurrent_weight = new_weight
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'recurrent_weight'):
            # Handle wrapped models (e.g., DataParallel, DistributedDataParallel)
            self.model.module.recurrent_weight = new_weight
        else:
            logger.warning("Could not update recurrent_weight: model does not have this attribute")
    
    def _get_current_recurrent_weight(self) -> Optional[float]:
        """Get the current recurrent_weight from the model."""
        if hasattr(self.model, 'recurrent_weight'):
            return self.model.recurrent_weight
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'recurrent_weight'):
            return self.model.module.recurrent_weight
        return None

    def _set_collator_pause_replace_prob(self, prob: float) -> None:
        for collator in (getattr(self, "data_collator", None), getattr(self, "train_data_collator", None)):
            if collator is None:
                continue
            if hasattr(collator, "pause_token_replace_prob"):
                setattr(collator, "pause_token_replace_prob", float(prob))

    def _resolve_resume_checkpoint_dir(self, resume_from_checkpoint: Optional[Union[str, bool]]) -> Optional[str]:
        if not resume_from_checkpoint:
            return None
        if isinstance(resume_from_checkpoint, bool):
            if not resume_from_checkpoint:
                return None
            return get_last_checkpoint(self.args.output_dir)
        return str(resume_from_checkpoint)

    def _patch_resume_trainer_state_batch_size(self, checkpoint_dir: str) -> None:
        state_path = os.path.join(checkpoint_dir, TRAINER_STATE_NAME)
        desired_batch_size = int(self.args.train_batch_size)

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to read trainer state at {state_path}: {exc}") from exc

        old_batch_size = state.get("train_batch_size", None)
        if old_batch_size == desired_batch_size:
            return

        state["train_batch_size"] = desired_batch_size
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f"{TRAINER_STATE_NAME}.", suffix=".tmp", dir=checkpoint_dir
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp_path, state_path)
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to patch train_batch_size in {state_path} to {desired_batch_size}: {exc}"
            ) from exc

        logger.info(
            "Patched resume trainer state batch size: %s -> %s (%s)",
            old_batch_size,
            desired_batch_size,
            state_path,
        )

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None, *args, **kwargs):
        checkpoint_dir = self._resolve_resume_checkpoint_dir(resume_from_checkpoint)
        if checkpoint_dir:
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:
                    self._patch_resume_trainer_state_batch_size(checkpoint_dir)
                dist.barrier()
            else:
                self._patch_resume_trainer_state_batch_size(checkpoint_dir)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

    def _get_collator_pause_replace_prob(self) -> Optional[float]:
        for collator in (getattr(self, "data_collator", None), getattr(self, "train_data_collator", None)):
            if collator is None:
                continue
            if hasattr(collator, "pause_token_replace_prob"):
                try:
                    return float(getattr(collator, "pause_token_replace_prob"))
                except Exception:
                    return None
        return None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Loss computation for RCOT: uses the precomputed loss from RCOTWrapper when available,
        which correctly handles token-by-token processing and batch size scaling. Falls back to
        recomputing from logits if the wrapper doesn't provide a loss.
        """
        label_shift = int(getattr(self.data_args, "label_shift", 1))
        if label_shift not in (0, 1):
            raise ValueError(f"data_args.label_shift must be 0 or 1; got {label_shift}")

        # Optionally sample BFAD depth per batch (uniform over provided candidates).
        current_step = getattr(self.state, "global_step", 0)
        if self._bfad_depth_values is not None:
            # Reset batch counter if we're on a new step
            if current_step != self._bfad_last_step:
                self._bfad_batch_counter = 0
                self._bfad_last_step = current_step
            
            # Sample BFAD depth for this batch
            bfad = self._sample_bfad_depth_for_batch(current_step, self._bfad_batch_counter)
            if self._last_bfad_depth != bfad:
                self._set_model_bfad_depth(bfad)
                self._last_bfad_depth = bfad
            
            # Increment batch counter for next batch in this step
            self._bfad_batch_counter += 1

        # IMPORTANT for padding-free FlashAttention varlen:
        # Do NOT truncate `input_ids` before the forward pass, otherwise `cu_seq_lens_*` / `max_length_*` become
        # inconsistent with q/k/v shapes and may trigger illegal memory access.
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        control_flows = inputs.get("control_flows")

        position_ids = inputs.get("position_ids")

        # Optional FlashAttention varlen kwargs for padding-free packing.
        fa_kwargs = {}
        for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            if k in inputs and inputs[k] is not None:
                fa_kwargs[k] = inputs[k]

        # Gate Activity Logging Logic
        # -----------------------------------------------------------
        # If enabled and this is a logging step, instruct the model to record gate stats.
        # Note: global_step is incremented AFTER backward pass, so at forward time it's still the "old" step.
        # We want to record when the NEXT log() call will happen, which is when (global_step + 1) % logging_steps == 0.
        record_stats_this_step = False
        if getattr(self.args, "log_gate_activity", False):
            current_step = getattr(self.state, "global_step", 0)
            # Check if the NEXT step will be a logging step
            # (since log() is called after global_step increments)
            # Also always record on step 0 for initial debugging
            if (self.args.logging_steps > 0 and (current_step + 1) % self.args.logging_steps == 0) or current_step == 0:
                record_stats_this_step = True
            
            # Prepare the buffer on the model if needed
            if record_stats_this_step:
                base_model = model.module if hasattr(model, "module") else model
                if hasattr(base_model, "active_gate_buffer"):
                    setattr(base_model, "active_gate_buffer", {})

        if self.rcot_args.rcot_enabled:
            assert control_flows is not None, "Control flows are required for RCOT training"
            rcot_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                control_flows=control_flows,
                position_ids=position_ids,
                record_gating_stats=record_stats_this_step,
                **fa_kwargs,
            )
        else:
            rcot_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **fa_kwargs,
            )

        # Post-process and capture stats if we recorded them
        if record_stats_this_step and self.rcot_args.rcot_enabled:
            if self.verbose:
                logger.debug(f"Capturing gate stats at step {getattr(self.state, 'global_step', 0)}")
            self._capture_stats_from_model(model, control_flows)

        logits = rcot_output.logits
        
        # Check if model returned a precomputed loss (as a tensor, not dict or other types).
        # Note: Liger kernels with FSDP may return loss as a dict, so we must verify it's a tensor.
        model_loss = getattr(rcot_output, 'loss', None)
        if model_loss is not None and torch.is_tensor(model_loss):
            loss = model_loss
        else:
            # Compute loss from logits (standard case when labels not passed to model forward)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            if label_shift == 1 and logits.size(-2) >= 2 and labels.size(-1) >= 2:
                # Next-token prediction: logits at t predict labels at t+1.
                shifted_logits = logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.flatten())

        if not return_outputs:
            return loss

        outputs = {
            "logits": logits,
            "past_key_values": rcot_output.past_key_values,
            "hidden_states": rcot_output.hidden_states,
        }
        return loss, outputs

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Align labels with logits for next-token prediction.

        Our `compute_loss()` shifts only at loss time (logits left, labels right) for next-token prediction.
        HF Trainer's default `prediction_step()` would return the unshifted `labels`, causing
        metric misalignment. This override returns labels aligned with the logits produced by `compute_loss()`.
        """
        has_labels = "labels" in inputs and inputs["labels"] is not None
        inputs = self._prepare_inputs(inputs)

        label_shift = int(getattr(self.data_args, "label_shift", 1))
        if label_shift not in (0, 1):
            raise ValueError(f"data_args.label_shift must be 0 or 1; got {label_shift}")

        labels = None
        if has_labels:
            labels = inputs.get("labels")
            if labels is not None and label_shift == 1:
                labels = labels[..., 1:]

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.detach()
                logits = outputs["logits"]
            else:
                loss = None
                outputs = model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        if prediction_loss_only:
            return loss, None, None

        if labels is not None and label_shift == 1 and logits is not None and logits.size(-2) >= 2:
            logits = logits[..., :-1, :]

        return loss, logits.detach(), labels.detach() if labels is not None else None

    def evaluate(
        self,
        eval_dataset: Optional[Union[str, Dataset]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        if not getattr(self.args, "eval_print_examples", False):
            return metrics

        if self.is_world_process_zero():
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(self.eval_data_collator, "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(self.data_collator, "tokenizer", None)
            if tokenizer is None:
                logger.warning("eval_print_examples is enabled but no tokenizer found on trainer/collator.")
            else:
                # Pick random examples (seeded) from the eval dataset for printing.
                # This does NOT shuffle evaluation itself; it only affects which examples we log.
                dataset_obj = eval_dataset if eval_dataset is not None else self.eval_dataset
                if dataset_obj is None:
                    logger.warning("eval_print_examples is enabled but no eval dataset is available.")
                else:
                    n = int(getattr(self.args, "eval_print_examples_count", 2) or 0)
                    max_pos = int(getattr(self.args, "eval_print_max_positions", 64) or 0)
                    n = max(0, n)
                    max_pos = max(1, max_pos)
                    if n > 0:
                        # Seed defaults to training seed; offset by global_step for variety across eval calls.
                        seed = int(getattr(self.args, "seed", 0) or 0)
                        step_offset = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)

                        import random

                        rng = random.Random(seed + 1000003 * step_offset)
                        ds_len = len(dataset_obj)
                        if ds_len > 0:
                            k = min(n, ds_len)
                            picked = rng.sample(range(ds_len), k=k) if ds_len >= k else list(range(ds_len))

                            # Build a small eval dataloader for the sampled subset.
                            subset = dataset_obj.select(picked) if hasattr(dataset_obj, "select") else dataset_obj
                            dataloader = self.get_eval_dataloader(subset)
                            batch = next(iter(dataloader))

                            # Optionally capture per-token gate statistics for this print batch.
                            # We only do this for the sampled batch to keep overhead minimal.
                            control_flows = batch.get("control_flows")
                            recurrent_gate_means = None
                            input_gate_means = None
                            hidden_in_means = None
                            hidden_rec_means = None
                            hidden_out_means = None
                            gamma_recurrent_means = None
                            gamma_input_means = None
                            gamma_means = None
                            if control_flows is not None and hasattr(self.model, "active_gate_buffer"):
                                # Clear any previous buffer.
                                setattr(self.model, "active_gate_buffer", {})

                                gate_inputs = dict(batch)
                                gate_inputs = self._prepare_inputs(gate_inputs)
                                gate_inputs["record_gating_stats"] = True
                                with torch.no_grad():
                                    _ = self.model(**gate_inputs)

                                # Concatenate per-step logs into a (B, T, ...) array.
                                if hasattr(self.model, "post_process_gating_stats"):
                                    self.model.post_process_gating_stats()

                                gate_logs = getattr(self.model, "active_gate_buffer", None) or {}
                                recurrent_gate = gate_logs.get("recurrent_gate")
                                input_gate = gate_logs.get("input_gate")
                                hidden_in = gate_logs.get("hidden_norm/input")
                                hidden_rec = gate_logs.get("hidden_norm/recurrent")
                                hidden_out = gate_logs.get("hidden_norm/mixed")
                                gamma_recurrent = gate_logs.get("rezero_gamma_recurrent_gate")
                                gamma_input = gate_logs.get("rezero_gamma_input_gate")
                                gamma = gate_logs.get("rezero_gamma") or gate_logs.get("gamma")

                                # Convert to per-position scalar (mean over hidden dim when present).
                                if recurrent_gate is not None:
                                    arr = np.asarray(recurrent_gate)
                                    recurrent_gate_means = (
                                        arr.mean(axis=-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                    )
                                if input_gate is not None:
                                    arr = np.asarray(input_gate)
                                    input_gate_means = (
                                        arr.mean(axis=-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                    )
                                if hidden_in is not None:
                                    arr = np.asarray(hidden_in)
                                    hidden_in_means = arr.squeeze(-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                if hidden_rec is not None:
                                    arr = np.asarray(hidden_rec)
                                    hidden_rec_means = arr.squeeze(-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                if hidden_out is not None:
                                    arr = np.asarray(hidden_out)
                                    hidden_out_means = arr.squeeze(-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                if gamma_recurrent is not None:
                                    arr = np.asarray(gamma_recurrent)
                                    gamma_recurrent_means = (
                                        arr.mean(axis=-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                    )
                                if gamma_input is not None:
                                    arr = np.asarray(gamma_input)
                                    gamma_input_means = (
                                        arr.mean(axis=-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                    )
                                if gamma is not None:
                                    arr = np.asarray(gamma)
                                    gamma_means = (
                                        arr.mean(axis=-1) if arr.ndim >= 3 else arr.reshape(arr.shape[0], -1)
                                    )
                                # Prevent this (possibly post-processed ndarray) buffer from affecting the real eval pass below.
                                # RCOTWrapper.forward() treats a non-None active_gate_buffer as an implicit record flag.
                                setattr(self.model, "active_gate_buffer", None)

                            loss, logits, labels = self.prediction_step(
                                self.model,
                                batch,
                                prediction_loss_only=False,
                                ignore_keys=ignore_keys,
                            )

                            if logits is not None:
                                preds = logits.argmax(dim=-1)
                                input_ids = batch.get("input_ids")
                                attention_mask = batch.get("attention_mask")
                                if control_flows is None:
                                    control_flows = batch.get("control_flows")

                                n = max(0, min(k, preds.shape[0]))
                                examples_data = []

                                for i in range(n):
                                    seq_mask = attention_mask[i].bool() if attention_mask is not None else torch.ones_like(preds[i], dtype=torch.bool)
                                    positions = torch.nonzero(seq_mask, as_tuple=False).flatten().tolist()
                                    if len(positions) > max_pos:
                                        positions = positions[:max_pos]
                                    # Guard against mask/pred length mismatches to avoid CUDA index errors.
                                    max_len = int(preds.shape[1])
                                    positions = [p for p in positions if p < max_len]

                                    def _tok(ids):
                                        return tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)

                                    input_tokens = _tok(input_ids[i, positions].tolist()) if input_ids is not None else []
                                    pred_tokens = _tok(preds[i, positions].tolist())

                                    label_tokens = []
                                    if labels is not None:
                                        if labels.shape[1] == 0:
                                            logger.warning(
                                                "Eval example has empty labels (seq_len=0). idx=%s batch_shape=%s labels_shape=%s",
                                                picked[i] if i < len(picked) else "?",
                                                tuple(input_ids.shape) if input_ids is not None else None,
                                                tuple(labels.shape),
                                            )
                                        else:
                                            label_ids = labels[i, positions].tolist()
                                            label_tokens = ["<IGN>" if x == -100 else _tok([x])[0] for x in label_ids]

                                    # Control flow and per-position gate scalars.
                                    cf_values = []
                                    if control_flows is not None:
                                            cf_values = [int(x) for x in control_flows[i, positions].detach().cpu().tolist()]

                                    # Gate means are logged only for recurrent positions (control_flow > 1). Align by walking positions.
                                    rgate_values = []
                                    igate_values = []
                                    h_in_values = []
                                    h_rec_values = []
                                    h_out_values = []
                                    g_rec_values = []
                                    g_in_values = []
                                    g_values = []
                                    rptr = 0
                                    if recurrent_gate_means is not None and i < recurrent_gate_means.shape[0]:
                                        rseq = recurrent_gate_means[i].tolist()
                                    else:
                                        rseq = None
                                    if input_gate_means is not None and i < input_gate_means.shape[0]:
                                        iseq = input_gate_means[i].tolist()
                                    else:
                                        iseq = None
                                    if hidden_in_means is not None and i < hidden_in_means.shape[0]:
                                        h_in_seq = hidden_in_means[i].tolist()
                                    else:
                                        h_in_seq = None
                                    if hidden_rec_means is not None and i < hidden_rec_means.shape[0]:
                                        h_rec_seq = hidden_rec_means[i].tolist()
                                    else:
                                        h_rec_seq = None
                                    if hidden_out_means is not None and i < hidden_out_means.shape[0]:
                                        h_out_seq = hidden_out_means[i].tolist()
                                    else:
                                        h_out_seq = None
                                    if gamma_recurrent_means is not None and i < gamma_recurrent_means.shape[0]:
                                        g_rec_seq = gamma_recurrent_means[i].tolist()
                                    else:
                                        g_rec_seq = None
                                    if gamma_input_means is not None and i < gamma_input_means.shape[0]:
                                        g_in_seq = gamma_input_means[i].tolist()
                                    else:
                                        g_in_seq = None
                                    if gamma_means is not None and i < gamma_means.shape[0]:
                                        g_seq = gamma_means[i].tolist()
                                    else:
                                        g_seq = None

                                    for j, pos in enumerate(positions):
                                        cf = cf_values[j] if j < len(cf_values) else None
                                        if cf is not None and cf > 1:
                                            r_val = None
                                            i_val = None
                                            h_in_val = None
                                            h_rec_val = None
                                            h_out_val = None
                                            g_rec_val = None
                                            g_in_val = None
                                            g_val = None
                                            if rseq is not None and rptr < len(rseq):
                                                r_val = float(rseq[rptr])
                                            if iseq is not None and rptr < len(iseq):
                                                i_val = float(iseq[rptr])
                                            if h_in_seq is not None and rptr < len(h_in_seq):
                                                h_in_val = float(h_in_seq[rptr])
                                            if h_rec_seq is not None and rptr < len(h_rec_seq):
                                                h_rec_val = float(h_rec_seq[rptr])
                                            if h_out_seq is not None and rptr < len(h_out_seq):
                                                h_out_val = float(h_out_seq[rptr])
                                            if g_rec_seq is not None and rptr < len(g_rec_seq):
                                                g_rec_val = float(g_rec_seq[rptr])
                                            if g_in_seq is not None and rptr < len(g_in_seq):
                                                g_in_val = float(g_in_seq[rptr])
                                            if g_seq is not None and rptr < len(g_seq):
                                                g_val = float(g_seq[rptr])
                                            rgate_values.append(r_val)
                                            igate_values.append(i_val)
                                            h_in_values.append(h_in_val)
                                            h_rec_values.append(h_rec_val)
                                            h_out_values.append(h_out_val)
                                            g_rec_values.append(g_rec_val)
                                            g_in_values.append(g_in_val)
                                            g_values.append(g_val)
                                            rptr += 1
                                        else:
                                            g_rec_val = None
                                            g_in_val = None
                                            g_val = None
                                            if g_rec_seq is not None and j < len(g_rec_seq):
                                                g_rec_val = float(g_rec_seq[j])
                                            if g_in_seq is not None and j < len(g_in_seq):
                                                g_in_val = float(g_in_seq[j])
                                            if g_seq is not None and j < len(g_seq):
                                                g_val = float(g_seq[j])
                                            rgate_values.append(None)
                                            igate_values.append(None)
                                            h_in_values.append(None)
                                            h_rec_values.append(None)
                                            h_out_values.append(None)
                                            g_rec_values.append(g_rec_val)
                                            g_in_values.append(g_in_val)
                                            g_values.append(g_val)

                                    def _safe(tok: str) -> str:
                                        # Keep the table one-token-per-cell even if a tokenizer emits whitespace.
                                        # (e.g., SentencePiece '▁', byte-level artifacts, or accidental tabs/newlines.)
                                        tok = "" if tok is None else str(tok)
                                        # Strip leading space markers (e.g., RoBERTa/GPT-style) for cleaner logs.
                                        for marker in ("\u0120", "\u2581"):
                                            if tok.startswith(marker):
                                                tok = tok[len(marker):]
                                        return tok.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")

                                    def _fmt_gate(v: Optional[float]) -> str:
                                        if v is None:
                                            return ""
                                        return f"{float(v):.3f}"

                                    idx_w = max(len("idx"), max((len(str(p)) for p in positions), default=1))
                                    cf_w = max(len("cf"), max((len(str(v)) for v in cf_values), default=0))
                                    rgate_w = max(len("rgate"), max((len(_fmt_gate(v)) for v in rgate_values), default=0))
                                    igate_w = max(len("igate"), max((len(_fmt_gate(v)) for v in igate_values), default=0))
                                    h_in_w = max(len("h_in"), max((len(_fmt_gate(v)) for v in h_in_values), default=0))
                                    h_rec_w = max(len("h_rec"), max((len(_fmt_gate(v)) for v in h_rec_values), default=0))
                                    h_out_w = max(len("h_out"), max((len(_fmt_gate(v)) for v in h_out_values), default=0))
                                    g_rec_w = max(len("g_rec"), max((len(_fmt_gate(v)) for v in g_rec_values), default=0))
                                    g_in_w = max(len("g_in"), max((len(_fmt_gate(v)) for v in g_in_values), default=0))
                                    g_w = max(len("gamma"), max((len(_fmt_gate(v)) for v in g_values), default=0))
                                    in_w = max(len("in"), max((len(_safe(t)) for t in input_tokens), default=0))
                                    pred_w = max(len("pred"), max((len(_safe(t)) for t in pred_tokens), default=0))
                                    label_w = max(len("label"), max((len(_safe(t)) for t in label_tokens), default=0))

                                    # Use spaces (not tabs) so logs render consistently in terminals + W&B.
                                    header_parts = [
                                        f"{'idx':>{idx_w}}",
                                        f"{'cf':>{cf_w}}",
                                        f"{'rgate':>{rgate_w}}",
                                        f"{'igate':>{igate_w}}",
                                    ]
                                    if any(v is not None for v in h_in_values + h_rec_values + h_out_values):
                                        header_parts.extend(
                                            [
                                                f"{'h_in':>{h_in_w}}",
                                                f"{'h_rec':>{h_rec_w}}",
                                                f"{'h_out':>{h_out_w}}",
                                            ]
                                        )
                                    if any(v is not None for v in g_rec_values + g_in_values + g_values):
                                        header_parts.extend(
                                            [
                                                f"{'g_rec':>{g_rec_w}}",
                                                f"{'g_in':>{g_in_w}}",
                                                f"{'gamma':>{g_w}}",
                                            ]
                                        )
                                    header_parts.extend(
                                        [
                                            f"{'in':<{in_w}}",
                                            f"{'pred':<{pred_w}}",
                                            f"{'label':<{label_w}}",
                                        ]
                                    )
                                    header = "  ".join(header_parts)
                                    lines = [header]
                                    for j, pos in enumerate(positions):
                                        in_tok = _safe(input_tokens[j]) if input_tokens else ""
                                        pred_tok = _safe(pred_tokens[j]) if pred_tokens else ""
                                        lab_tok = _safe(label_tokens[j]) if label_tokens else ""
                                        cf_tok = str(cf_values[j]) if j < len(cf_values) else ""
                                        r_tok = _fmt_gate(rgate_values[j]) if j < len(rgate_values) else ""
                                        i_tok = _fmt_gate(igate_values[j]) if j < len(igate_values) else ""
                                        h_in_tok = _fmt_gate(h_in_values[j]) if j < len(h_in_values) else ""
                                        h_rec_tok = _fmt_gate(h_rec_values[j]) if j < len(h_rec_values) else ""
                                        h_out_tok = _fmt_gate(h_out_values[j]) if j < len(h_out_values) else ""
                                        g_rec_tok = _fmt_gate(g_rec_values[j]) if j < len(g_rec_values) else ""
                                        g_in_tok = _fmt_gate(g_in_values[j]) if j < len(g_in_values) else ""
                                        g_tok = _fmt_gate(g_values[j]) if j < len(g_values) else ""
                                        line_parts = [
                                            f"{str(pos):>{idx_w}}",
                                            f"{cf_tok:>{cf_w}}",
                                            f"{r_tok:>{rgate_w}}",
                                            f"{i_tok:>{igate_w}}",
                                        ]
                                        if any(v is not None for v in h_in_values + h_rec_values + h_out_values):
                                            line_parts.extend(
                                                [
                                                    f"{h_in_tok:>{h_in_w}}",
                                                    f"{h_rec_tok:>{h_rec_w}}",
                                                    f"{h_out_tok:>{h_out_w}}",
                                                ]
                                            )
                                        if any(v is not None for v in g_rec_values + g_in_values + g_values):
                                            line_parts.extend(
                                                [
                                                    f"{g_rec_tok:>{g_rec_w}}",
                                                    f"{g_in_tok:>{g_in_w}}",
                                                    f"{g_tok:>{g_w}}",
                                                ]
                                            )
                                        line_parts.extend(
                                            [
                                                f"{in_tok:<{in_w}}",
                                                f"{pred_tok:<{pred_w}}",
                                                f"{lab_tok:<{label_w}}",
                                            ]
                                        )
                                        lines.append("  ".join(line_parts))

                                    logger.info(
                                        "Eval example %d/%d (dataset_idx=%s):\n%s",
                                        i + 1,
                                        n,
                                        picked[i] if i < len(picked) else "?",
                                        "\n".join(lines),
                                    )

                                    examples_data.append(
                                    {
                                        "dataset_idx": picked[i] if i < len(picked) else None,
                                            "positions": [int(p) for p in positions],
                                            "series": {
                                                "recurrent_gate": rgate_values,
                                                "input_gate": igate_values,
                                                "hidden_in": h_in_values,
                                                "hidden_rec": h_rec_values,
                                                "hidden_out": h_out_values,
                                                "gamma_rec": g_rec_values,
                                                "gamma_in": g_in_values,
                                                "gamma": g_values,
                                            },
                                        }
                                    )

            if examples_data:
                step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
                self._log_eval_examples_to_wandb(examples_data, step, metric_key_prefix)

        # All ranks must participate in per-position loss reductions to avoid NCCL hangs.
        self._log_eval_loss_by_position(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        return metrics

    def _log_eval_loss_by_position(
        self,
        eval_dataset: Optional[Union[str, Dataset]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> None:
        dataset_obj = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset_obj is None:
            logger.warning("eval_log_position_losses is enabled but no eval dataset is available.")
            return

        dataloader = self.get_eval_dataloader(dataset_obj)
        max_len_cap = getattr(self.args, "eval_log_position_max_len", None)
        if max_len_cap is None:
            max_len_cap = getattr(self.data_args, "max_length", None)
        max_len_cap = int(max_len_cap) if max_len_cap is not None else None

        pos_sum = None
        pos_count = None

        with torch.no_grad():
            for batch in dataloader:
                _, logits, labels = self.prediction_step(
                    self.model,
                    batch,
                    prediction_loss_only=False,
                    ignore_keys=ignore_keys,
                )
                if logits is None or labels is None:
                    continue

                if max_len_cap is not None and logits.size(1) > max_len_cap:
                    logits = logits[:, :max_len_cap, :]
                    labels = labels[:, :max_len_cap]

                mask = labels != -100
                if not torch.any(mask):
                    continue

                labels_safe = labels.masked_fill(~mask, 0)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                nll = -torch.gather(log_probs, -1, labels_safe.unsqueeze(-1)).squeeze(-1)
                nll = nll * mask

                batch_sum = nll.sum(dim=0).detach().cpu().numpy()
                batch_count = mask.sum(dim=0).detach().cpu().numpy()

                if pos_sum is None:
                    pos_sum = batch_sum.astype(np.float64, copy=False)
                    pos_count = batch_count.astype(np.float64, copy=False)
                    continue

                if batch_sum.shape[0] > pos_sum.shape[0]:
                    pad = batch_sum.shape[0] - pos_sum.shape[0]
                    pos_sum = np.pad(pos_sum, (0, pad), mode="constant")
                    pos_count = np.pad(pos_count, (0, pad), mode="constant")

                pos_sum[: batch_sum.shape[0]] += batch_sum
                pos_count[: batch_count.shape[0]] += batch_count

        if pos_sum is None or pos_count is None:
            logger.warning("eval_log_position_losses ran but no valid labels were found.")
            return

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = getattr(self.args, "device", torch.device("cpu"))

        local_len = torch.tensor([pos_sum.shape[0]], device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_len, op=dist.ReduceOp.MAX)
        global_len = int(local_len.item())

        if pos_sum.shape[0] < global_len:
            pad = global_len - pos_sum.shape[0]
            pos_sum = np.pad(pos_sum, (0, pad), mode="constant")
            pos_count = np.pad(pos_count, (0, pad), mode="constant")

        pos_sum_t = torch.tensor(pos_sum, device=device, dtype=torch.float64)
        pos_count_t = torch.tensor(pos_count, device=device, dtype=torch.float64)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pos_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(pos_count_t, op=dist.ReduceOp.SUM)

        if not self.is_world_process_zero():
            return

        avg_loss = pos_sum_t / (pos_count_t + 1e-12)
        avg_loss_list = avg_loss.detach().cpu().tolist()
        count_list = pos_count_t.detach().cpu().tolist()

        step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
        output_dir = getattr(self.args, "output_dir", None) or "."
        path = os.path.join(output_dir, f"{metric_key_prefix}_loss_by_position_step{step}.json")
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "step": step,
                        "max_len_cap": max_len_cap,
                        "loss_by_position": avg_loss_list,
                        "token_counts": count_list,
                    },
                    fp,
                    indent=2,
                )
            preview_len = min(10, len(avg_loss_list))
            logger.info(
                "Saved per-position eval loss to %s (len=%d, first=%d: %s)",
                path,
                len(avg_loss_list),
                preview_len,
                [round(x, 4) for x in avg_loss_list[:preview_len]],
            )
        except Exception:
            logger.exception("Failed to save per-position eval loss to %s", path)

        self._log_eval_loss_by_position_to_wandb(avg_loss_list, count_list, step, metric_key_prefix)

    def _log_eval_examples_to_wandb(self, examples, step: int, metric_key_prefix: str) -> None:
        for entry in examples:
            dataset_idx = entry.get("dataset_idx")
            positions = entry.get("positions") or []
            series = entry.get("series") or {}
            if not positions:
                continue
            for name, values in series.items():
                if values is None:
                    continue
                title = f"{name} (idx={dataset_idx})"
                fig = self._plot_series_image(positions, values, title)
                save_path = self._save_eval_plot(fig, f"examples/{dataset_idx}", name, step, metric_key_prefix)
                if save_path:
                    logger.info("Saved eval plot: %s", save_path)

    def _log_eval_examples_to_wandb_scalars(self, examples, step: int, metric_key_prefix: str) -> None:
        """
        Log gate values per token position as plain scalars instead of tables/images.
        Uses a custom step metric (token_position) so charts use position on the x-axis.
        """
        try:
            import wandb
        except Exception:
            logger.debug("wandb not available; skipping scalar gate logging for eval examples")
            return

        run = wandb.run
        if run is None:
            logger.debug("wandb run not initialized; skipping scalar gate logging for eval examples")
            return

        # Define the shared step metric for position-based plots once per call.
        step_metric = f"{metric_key_prefix}/token_position"
        wandb.define_metric(step_metric, step_metric=step_metric)

        for entry in examples:
            positions = entry.get("positions") or []
            series = entry.get("series") or {}
            dataset_idx = entry.get("dataset_idx")
            if not positions:
                continue

            for name, values in series.items():
                if values is None:
                    continue

                metric_name = f"{metric_key_prefix}/example_{dataset_idx}/{name}"
                wandb.define_metric(metric_name, step_metric=step_metric)

                for pos, val in zip(positions, values):
                    if val is None:
                        continue
                    wandb.log(
                        {
                            step_metric: int(pos),
                            metric_name: float(val),
                            f"{metric_key_prefix}/example_{dataset_idx}/global_step": step,
                        },
                        step=step,
                    )

    def _log_eval_loss_by_position_to_wandb(self, losses, counts, step: int, metric_key_prefix: str) -> None:
        positions = list(range(len(losses)))
        fig = self._plot_series_image(positions, losses, "loss_by_position")
        save_path = self._save_eval_plot(fig, "loss_by_position", "loss_by_position", step, metric_key_prefix)
        if save_path:
            logger.info("Saved eval plot: %s", save_path)

    def _plot_series_image(self, positions, values, title: str):
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.asarray(positions, dtype=float)
        y = np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)
        if x.shape[0] != y.shape[0]:
            n = min(x.shape[0], y.shape[0])
            x = x[:n]
            y = y[:n]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, y, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("position")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def _save_eval_plot(self, fig, subdir: str, name: str, step: int, metric_key_prefix: str):
        output_dir = getattr(self.args, "output_dir", None) or "."
        safe_subdir = str(subdir).strip("/").replace("..", "_")
        safe_name = str(name).replace("/", "_").replace(" ", "_")
        plot_dir = os.path.join(output_dir, "eval_plots", metric_key_prefix, safe_subdir)
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"{safe_name}_step{step}.png"
        path = os.path.join(plot_dir, filename)
        fig.savefig(path, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return os.path.abspath(path)
    
    def training_step(self, *args, **kwargs):
        # Update collator config with the current step
        self.set_collator_step(self.state.global_step)

        # BFAD sampling is now done per-batch in compute_loss()
        # Log BFAD info periodically if enabled
        if self._bfad_depth_values is not None and self.verbose and self.args.logging_steps and (self.state.global_step % self.args.logging_steps == 0):
            logger.info("Step %s: BFAD sampling enabled (per-batch) from %s", self.state.global_step, self._bfad_depth_values)
        
        # Update recurrent_weight if curriculum is enabled
        if self.recurrent_weight_scheduler is not None:
            new_weight = self.recurrent_weight_scheduler.get_value(self.state.global_step)
            self._update_model_recurrent_weight(new_weight)
            
            # Log the current recurrent_weight periodically
            if self.state.global_step % self.args.logging_steps == 0:
                logger.debug(f"Step {self.state.global_step}: recurrent_weight = {new_weight:.4f}")

        if self.pause_token_replace_prob_scheduler is not None:
            new_prob = self.pause_token_replace_prob_scheduler.get_value(self.state.global_step)
            self._set_collator_pause_replace_prob(new_prob)
            if self.state.global_step % self.args.logging_steps == 0:
                logger.debug(f"Step {self.state.global_step}: pause_token_replace_prob = {new_prob:.4f}")
        
        return super().training_step(*args, **kwargs)
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Override log to include recurrent_weight and gate activity in metrics."""
        # Add current recurrent_weight to logs if curriculum is enabled
        if self.recurrent_weight_scheduler is not None:
            current_weight = self._get_current_recurrent_weight()
            if current_weight is not None:
                logs["recurrent_weight"] = current_weight

        if self.pause_token_replace_prob_scheduler is not None:
            current_prob = self._get_collator_pause_replace_prob()
            if current_prob is not None:
                logs["pause_token_replace_prob"] = current_prob
        
        # Pull gate stats if they were captured in compute_loss during the last step
        if getattr(self.args, "log_gate_activity", False):
            # The gate stats buffer should have been populated by compute_loss
            # in the most recent training step that aligned with logging_steps
            gate_stats = getattr(self, "_current_gate_stats", None)
            if gate_stats:
                logs.update(gate_stats)
                if self.verbose:
                    logger.debug(f"Logging gate stats at step {getattr(self.state, 'global_step', 0)}: {list(gate_stats.keys())}")
                self._current_gate_stats = None  # Clear buffer after logging
            elif self.verbose and hasattr(self.state, "global_step") and self.state.global_step % self.args.logging_steps == 0:
                # Only log warning if we're at a logging step and expected stats but didn't get them
                logger.debug("Expected gate stats at logging step but none were found")
        
        # Add perplexity to logs for TinyStories and other NLP dataset
        if self._should_log_perplexity():
            loss_key = "loss" if "loss" in logs else None
            if loss_key is None:
                # Check for eval loss
                for key in logs:
                    if "loss" in key:
                        loss_key = key
                        break
            if loss_key is not None:
                try:
                    perplexity = math.exp(logs[loss_key])
                    # Corresponding perplexity key
                    ppl_key = loss_key.replace("loss", "perplexity") if loss_key != "loss" else "perplexity"
                    logs[ppl_key] = perplexity
                except OverflowError:
                    # Loss too high, perplexity would overflow
                    ppl_key = loss_key.replace("loss", "perplexity") if loss_key != "loss" else "perplexity"
                    logs[ppl_key] = float("inf")
        
        super().log(logs, *args, **kwargs)
    
    def _should_log_perplexity(self) -> bool:
        """Check if perplexity should be logged based on dataset name."""
        if not hasattr(self, "_log_perplexity_cache"):
            dataset_name = getattr(self.data_args, "train_dataset_name", "") or ""
            dataset_name_lower = dataset_name.lower()
            self._log_perplexity_cache = (
                "tinystories" in dataset_name_lower
                or "wikitext" in dataset_name_lower
                or "fineweb" in dataset_name_lower
            )
        return self._log_perplexity_cache

    def _get_train_sampler(self, train_dataset=None) -> Sampler:

        if train_dataset is None:
            train_dataset = self.train_dataset

        if self.args.batch_gather_by_length:
            if self.verbose:
                logger.info("Gathering batch by length (using precomputed 'seq_len' if available)")
            if hasattr(train_dataset, 'column_names') and 'seq_len' in getattr(train_dataset, 'column_names'):
                lengths = train_dataset['seq_len']
            else:
                lengths = map(lambda x: len(x["input_ids"]), train_dataset)
            return LengthGroupedSampler(self.args.train_batch_size, dataset=train_dataset, lengths=lengths)

        elif self.args.batch_gather_by_nonrecur:
            if self.verbose:
                logger.info("Gathering batch by non-recurrence (strict, requires 'nonrecur_len')")
            if not (hasattr(train_dataset, 'column_names') and 'nonrecur_len' in getattr(train_dataset, 'column_names')):
                raise ValueError("'nonrecur_len' column missing. Ensure preprocessing adds it before enabling batch_gather_by_nonrecur, existing columns: " + str(getattr(train_dataset, 'column_names', [])))
            lengths = train_dataset['nonrecur_len']
            return LengthGroupedSampler(self.args.train_batch_size, dataset=train_dataset, lengths=lengths)
        
        else:
            return RandomSampler(train_dataset, replacement=False, num_samples=len(train_dataset))

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Sampler:
        return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        if self.args.eval_samples > 0:
            eval_dataset = eval_dataset.select(range(self.args.eval_samples))
        data_collator = self.eval_data_collator

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
