import os
import logging
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
import json
from typing import Dict, Any, Iterable, Tuple

from modeling.tinyllama import TinyLlamaConfig, TinyLlamaForCausalLM
from .rcot_config import RCOTConfig

logger = logging.getLogger(__name__)

def resolve_base_config(cfg_like):
    if isinstance(cfg_like, dict):
        return cfg_like
    if hasattr(cfg_like, "to_dict"):
        return cfg_like.to_dict()
    return None

def resolve_dtype(dtype):
    if dtype is None:
        return torch.float32
    if isinstance(dtype, str):
        try:
            return getattr(torch, dtype.lower())
        except AttributeError:
            raise ValueError(f"Unknown dtype: {dtype}")
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def load_base_model_from_config(config: RCOTConfig):

    # Construct the base model from the saved base_config
    base_config_data = getattr(config, "base_config", None)
    base_model_type = getattr(config, "base_model_type", None)
    if base_config_data is None:
        raise ValueError(
            "RCOTWrapper requires either a base model instance or a config containing `base_config`. "
            "Use RCOTWrapper.from_pretrained_with_rcot or RCOTConfig.from_base_config to create a valid config."
        )

    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)  # backward compatibility

    base_config_dict = resolve_base_config(base_config_data)
    if base_config_dict is None:
        raise ValueError("Unsupported base_config type when constructing base model.")

    # Attention backend selection
    # ---------------------------
    # Prefer explicit model config over environment overrides. Default to
    # flash_attention_2 when unspecified.
    #
    # Precedence:
    #  1) config.attn_impl (preferred knob for callers)
    #  2) base_config.attn_impl if present
    #  3) config.attn_implementation (older naming)
    #  4) default: "flash_attention_2"
    attn_impl = (
        getattr(config, "attn_impl", None)
        or base_config_dict.get("attn_impl")
        or getattr(config, "attn_implementation", None)
        or "flash_attention_2"
    )

    # Construct the base model from the saved base_config
    if base_model_type == "tinyllama":
        base_config = TinyLlamaConfig.from_dict(base_config_dict)
        model = TinyLlamaForCausalLM(base_config)
    else:
        # Reconstruct the concrete config class for the stored base model type
        base_config_template = AutoConfig.for_model(base_model_type)
        base_config = base_config_template.__class__.from_dict(base_config_dict)
        # Best-effort: some configs use `attn_implementation`, some use private
        # variants. We set both if they exist.
        try:
            setattr(base_config, "attn_implementation", attn_impl)
        except Exception:
            pass
        try:
            setattr(base_config, "_attn_implementation", attn_impl)
        except Exception:
            pass

        model = AutoModelForCausalLM.from_config(
            base_config,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
    
    logger.info(f"Using attention implementation: {attn_impl}, dtype: {dtype}")
    return model


def load_rcot_config_with_fallback(path: str) -> RCOTConfig:
    """
    Resolve and load RCOTConfig from a checkpoint or directory.
    Handles common FSDP layout (config.json one level up) and last-checkpoint fallback.
    """
    resolved_path = path

    # Prefer a config that lives *inside* the provided path (e.g., a concrete
    # checkpoint directory) over a parent-run root config. This avoids races
    # where the run-root `config.json` is being rewritten while checkpoint
    # configs are already stable on disk.
    if os.path.exists(os.path.join(resolved_path, "config.json")):
        # Use the path as-is (checkpoint or run root).
        pass
    elif os.path.exists(os.path.join(os.path.dirname(resolved_path), "config.json")):
        # FSDP-style layout: checkpoint shards under a run root that holds
        # the wrapper config.
        resolved_path = os.path.dirname(resolved_path)
    else:
        if os.path.isdir(resolved_path):
            last_ckpt = get_last_checkpoint(resolved_path)
            if last_ckpt is not None:
                resolved_path = last_ckpt

    rcot_config = RCOTConfig.from_pretrained(resolved_path)
    return rcot_config

def fetch_hidden_size(model):
    # Fetch the hidden size from the base model config
    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd", None)  # GPT-2
        or getattr(model.config, "d_model", None)
    )
    if hidden_size is None:
        raise ValueError("Unable to infer hidden size from base configuration for RCOT wrapper.")
    return hidden_size

def load_weights_for_model(model: nn.Module, model_name_or_path: str, strict: bool = True):
    """
    Load weights for a (possibly sharded) checkpoint directory into `model`.
    Supports:
      - model.safetensors.index.json (sharded safetensors)
      - pytorch_model.bin.index.json (sharded bin)
      - model.safetensors (single file)
      - pytorch_model.bin (single file)
    Falls back to transformers' load_sharded_checkpoint.
    """
    safe_index = os.path.join(model_name_or_path, "model.safetensors.index.json")
    pt_index = os.path.join(model_name_or_path, "pytorch_model.bin.index.json")
    safe_path = os.path.join(model_name_or_path, "model.safetensors")
    pt_path = os.path.join(model_name_or_path, "pytorch_model.bin")

    def _weight_map_from_index(index_path: str) -> Dict[str, str]:
        with open(index_path, "r") as f:
            data = json.load(f)
        weight_map = data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid sharded index file (missing weight_map): {index_path}")
        return {str(k): str(v) for k, v in weight_map.items()}

    def _unique_shard_files_from_weight_map(weight_map: Dict[str, str]) -> list[str]:
        files = sorted(set(weight_map.values()))
        if not files:
            raise ValueError("Sharded index file has empty weight_map.")
        return files

    def _iter_shards_from_weight_map(weight_map: Dict[str, str], base_dir: str) -> Iterable[str]:
        for fname in _unique_shard_files_from_weight_map(weight_map):
            yield os.path.join(base_dir, fname)

    def _load_shard_file(shard_path: str) -> Dict[str, Any]:
        if shard_path.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load_file
            return safe_load_file(shard_path)
        return torch.load(shard_path, map_location="cpu")

    def _format_key_examples(keys: set[str], max_examples: int = 10) -> str:
        if not keys:
            return "(none)"
        sorted_keys = sorted(keys)
        preview = ", ".join(sorted_keys[:max_examples])
        if len(sorted_keys) > max_examples:
            preview += ", ..."
        return preview

    def _validate_sharded_key_compatibility(weight_map: Dict[str, str]) -> None:
        checkpoint_keys = set(weight_map.keys())
        model_keys = set(model.state_dict().keys())
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Strict checkpoint load failed due to key mismatch for sharded checkpoint. "
                f"Missing keys in checkpoint: {len(missing_keys)} "
                f"(examples: {_format_key_examples(missing_keys)}). "
                f"Unexpected keys in checkpoint: {len(unexpected_keys)} "
                f"(examples: {_format_key_examples(unexpected_keys)})."
            )

    def _load_sharded_from_index(index_path: str, base_dir: str) -> None:
        weight_map = _weight_map_from_index(index_path)
        if strict:
            _validate_sharded_key_compatibility(weight_map)

        expected_keys = set(weight_map.keys())
        loaded_keys = set()
        for shard_path in _iter_shards_from_weight_map(weight_map, base_dir):
            state = _load_shard_file(shard_path)
            loaded_keys.update(state.keys())
            model.load_state_dict(state, strict=False)

        if strict:
            missing_loaded_keys = expected_keys - loaded_keys
            unexpected_loaded_keys = loaded_keys - expected_keys
            if missing_loaded_keys or unexpected_loaded_keys:
                raise RuntimeError(
                    "Strict checkpoint load failed while reading sharded files. "
                    f"Indexed keys not found in shard contents: {len(missing_loaded_keys)} "
                    f"(examples: {_format_key_examples(missing_loaded_keys)}). "
                    f"Shard keys not present in index: {len(unexpected_loaded_keys)} "
                    f"(examples: {_format_key_examples(unexpected_loaded_keys)})."
                )

    if os.path.exists(safe_index) or os.path.exists(pt_index):
        # Transformers 5.0 removed `load_sharded_checkpoint`; load shards manually from the index file.
        index_path = safe_index if os.path.exists(safe_index) else pt_index
        _load_sharded_from_index(index_path, model_name_or_path)
    elif os.path.exists(safe_path):
        state = torch.load(safe_path, map_location="cpu") if safe_path.endswith(".bin") else None
        if state is None:
            from safetensors.torch import load_file as safe_load_file
            state = safe_load_file(safe_path)
        model.load_state_dict(state, strict=strict)
    elif os.path.exists(pt_path):
        state = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state, strict=strict)
    else:
        try:
            # Back-compat: if transformers provides a helper, use it.
            from transformers.modeling_utils import load_sharded_checkpoint  # type: ignore
            load_sharded_checkpoint(model, model_name_or_path, strict=strict)
        except ValueError as e:
            # This error can happen for fsdp sharded checkpoints
            print(f"Error loading weights for model {model_name_or_path}: {e} Trying parent directory")
            from transformers.modeling_utils import load_sharded_checkpoint  # type: ignore
            load_sharded_checkpoint(model, os.path.dirname(model_name_or_path), strict=strict)
        except ImportError as e:
            raise ImportError(
                "Unable to load checkpoint weights: sharded checkpoint helper is not available "
                "and no index/single-file weights were found at the provided path."
            ) from e
