# train llama models

import torch
import os
import json
from contextlib import nullcontext
from dataclasses import asdict, fields
from typing import Optional, List, Dict, Any
import numpy as np
from itertools import permutations

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from components.all_arguments import (
    ModelArguments,
    TrainingArguments,
    DataArguments,
    RCOTArguments,
    GenerationEvalArguments,
)
from components.data_utils import rcot_collator, RCOTEvalCollator, SkipLayerEvalCollator
from components.generation_eval import run_generation_evaluation, build_rl_reward_function, should_run_perplexity_eval

from components.rcot_trainer import RCOTTrainer
# from trl import GRPOConfig
from rcot_wrapper import RCOTWrapper
from rcot_wrapper.rcot_config import RCOTConfig
from transformers import set_seed
from datasets import load_from_disk, load_dataset, Dataset
from modeling import TinyLlamaConfig, TinyLlamaForCausalLM, RNNLMConfig, RNNLMForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
parser = HfArgumentParser(
    (
        ModelArguments,
        TrainingArguments,
        DataArguments,
        RCOTArguments,
        GenerationEvalArguments,
    )
)
model_args, training_args, data_args, rcot_args, eval_args, remaining_cli = parser.parse_args_into_dataclasses(return_remaining_strings=True)

def _should_log_to_wandb(args: TrainingArguments) -> bool:
    report_to = getattr(args, "report_to", None)
    if report_to is None:
        return False
    if isinstance(report_to, str):
        lowered = report_to.strip().lower()
        if lowered in {"", "none"}:
            return False
        if lowered == "all":
            return True
        return "wandb" in {t.strip().lower() for t in lowered.split(",")}
    if isinstance(report_to, (list, tuple, set)):
        lowered = {str(t).strip().lower() for t in report_to}
        return "wandb" in lowered or "all" in lowered
    return False

def _log_wandb_config(
    *,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    rcot_args: RCOTArguments,
    eval_args: GenerationEvalArguments,
) -> None:
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"true", "1", "yes"}:
        return
    try:
        import wandb  # type: ignore
    except Exception as e:
        logging.warning("W&B not available; skipping config upload. (%s)", e)
        return

    config_payload = {
        "training": getattr(training_args, "to_dict", lambda: asdict(training_args))(),
        "model": asdict(model_args),
        "data": asdict(data_args),
        "rcot": asdict(rcot_args),
        "eval": asdict(eval_args),
    }
    try:
        if wandb.run is None:
            # Avoid creating a second W&B run; Trainer will initialize if enabled.
            logging.info("W&B run not initialized yet; skipping config upload to avoid duplicate runs.")
            return
        wandb.config.update(config_payload, allow_val_change=True)
    except Exception as e:
        logging.warning("Failed to upload config to W&B. (%s)", e)

# ---------------------------------------------------------------------------
# PyTorch Inductor TF32 warning guard (workaround)
# ---------------------------------------------------------------------------
# Some PyTorch builds error when querying `torch.backends.cuda.matmul.allow_tf32`
# if TF32 was configured via a mix of the legacy and new APIs. Inductor calls
# this getter during `torch.compile()` via `_warn_tf32_disabled()`, which can
# crash training even though TF32 is only a performance knob.
#
# Workaround: when torch.compile is enabled, override the warning helper to a
# safe no-op so compilation can proceed.
if getattr(training_args, "torch_compile", False):
    try:
        import torch._inductor.compile_fx as _compile_fx  # type: ignore

        def _noop_warn_tf32_disabled() -> None:
            return None

        _compile_fx._warn_tf32_disabled = _noop_warn_tf32_disabled  # type: ignore[attr-defined]
    except Exception:
        # Best-effort only; if the import path changes we fall back to default behavior.
        pass

if getattr(training_args, "use_liger_kernel", False):
    # Best-effort enabling of Liger kernels for training.
    try:
        import liger_kernel  # type: ignore

        logging.info("Liger kernels enabled (liger_kernel imported successfully).")
    except Exception as e:
        logging.warning(
            "--use_liger_kernel was set but Liger is not available; continuing without it. (%s)",
            e,
        )


def _maybe_apply_liger_kernels_to_model_instance(model_obj: Any, *, desc: str) -> None:
    """
    Apply Liger kernels to a *supported* HF transformer model instance.

    Important: Liger dispatches based on `model.config.model_type`. RCOT models have
    `model_type="rcot"`, which Liger doesn't support, so we must apply to the
    underlying base model instance instead (e.g., LLaMA/Qwen/...).
    """
    if not getattr(training_args, "use_liger_kernel", False):
        return
    if model_obj is None:
        return

    # Best-effort idempotency (avoid re-applying if we touch the same instance twice).
    if getattr(model_obj, "_rcot_liger_applied", False):
        return

    from liger_kernel.transformers import _apply_liger_kernel_to_instance 

    _apply_liger_kernel_to_instance(model_obj)
    setattr(model_obj, "_rcot_liger_applied", True)
    logging.info(
        "Applied Liger kernels to %s (%s, model_type=%s).",
        desc,
        model_obj.__class__.__name__,
        getattr(getattr(model_obj, "config", None), "model_type", None),
    )

training_stage = (training_args.training_stage or "sft").strip().lower()


def _is_head_process(args: TrainingArguments) -> bool:
    """Return True when running on the global rank 0 process."""
    process_index = getattr(args, "process_index", None)
    if process_index is not None:
        return process_index == 0

    local_rank = getattr(args, "local_rank", None)
    if local_rank is not None and local_rank >= 0:
        rank = os.environ.get("RANK")
        if rank is not None:
            try:
                return int(rank) == 0
            except ValueError:
                return local_rank == 0
        return local_rank == 0

    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _main_process_first(args: TrainingArguments, desc: str):
    """Attach to HF/Accelerate main_process_first when available."""
    if hasattr(args, "main_process_first"):
        return args.main_process_first(desc=desc)
    return nullcontext()


def _wait_for_everyone(args: TrainingArguments):
    """Synchronize processes if distributed training is enabled."""
    distributed_state = getattr(args, "distributed_state", None)
    if distributed_state is not None and hasattr(distributed_state, "wait_for_everyone"):
        try:
            distributed_state.wait_for_everyone()
            return
        except Exception:
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

if training_args.resume_from_checkpoint is not None and training_args.resume_from_checkpoint != 'True' and training_args.resume_from_checkpoint != 'False':
    print("Overriding output_dir to resume_from_checkpoint path")
    training_args.output_dir = training_args.resume_from_checkpoint

resume_checkpoint_path = None
resume_flag = training_args.resume_from_checkpoint
if resume_flag is not None:
    if isinstance(resume_flag, str):
        flag = resume_flag.strip().lower()
        if flag in {"true", "1", "yes"}:
            resume_checkpoint_path = get_last_checkpoint(training_args.output_dir)
            if resume_checkpoint_path is None:
                raise FileNotFoundError(
                    f"resume_from_checkpoint=True but no checkpoint found in output_dir: {training_args.output_dir}"
                )
        elif flag in {"false", "0", "no", "none"}:
            resume_checkpoint_path = None
        else:
            resume_checkpoint_path = resume_flag
    elif resume_flag is True:
        resume_checkpoint_path = get_last_checkpoint(training_args.output_dir)
        if resume_checkpoint_path is None:
            raise FileNotFoundError(
                f"resume_from_checkpoint=True but no checkpoint found in output_dir: {training_args.output_dir}"
            )

if (not training_args.do_train) and training_args.do_eval and resume_checkpoint_path:
    logging.info("Eval-only run: loading model from checkpoint: %s", resume_checkpoint_path)
    model_args.model_name_or_path = resume_checkpoint_path
    model_args.from_pretrained = True

logging.info("start training")
#if training_args.project_name != "":
wandb_dir = training_args.output_dir
os.makedirs(wandb_dir, exist_ok=True)

os.environ["WANDB_PROJECT"]=training_args.project_name
os.environ["WANDB_DIR"]=wandb_dir
# os.environ["WANDB_MODE"]="offline"
if getattr(training_args, "run_name", None):
    os.environ["WANDB_NAME"] = training_args.run_name

is_head_process = _is_head_process(training_args)
using_multiple_processes = getattr(training_args, "world_size", 1) > 1


def save_fsdp_model(trainer):
    """Save FSDP model when model is sharded. Currently not done by HF Trainer."""
    if trainer.is_fsdp_enabled:
        if ("SHARDED_STATE_DICT" in str(trainer.accelerator.state.fsdp_plugin.state_dict_type)):
            state_dict = trainer.accelerator.get_state_dict(trainer.model)     
            trainer._save(training_args.output_dir, state_dict=state_dict)

def _dump_args_to_output(args_obj, filename):
    path = os.path.join(training_args.output_dir, filename)
    with open(path, "w") as f:
        json.dump(asdict(args_obj), f, indent=2, default=str)
    logging.info("Saved %s", path)

if is_head_process:
    for _filename, _args in (
        ("model_args.json", model_args),
        ("data_args.json", data_args),
        ("eval_args.json", eval_args),
    ):
        _dump_args_to_output(_args, _filename)

# get pid
pid = os.getpid()
training_args.pid = pid
# set seed
seed = training_args.seed
training_args.remove_unused_columns = False
set_seed(seed)

tokenizer_name = (model_args.tokenizer_name_or_path or "").strip()
if tokenizer_name.lower() in {"char", "character", "s5_char", "s5-char"}:
    from components.char_tokenizer import S5CharTokenizer

    tokenizer: PreTrainedTokenizer = S5CharTokenizer()
else:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def _maybe_add_s5_tokens(tok: PreTrainedTokenizer, data_args: DataArguments) -> int:
    """Add one token per S5 action/state.

    For the S5 state-tracking task, states are also serialized using the A_* token
    family (i.e., *no* separate S_* tokens).
    """
    # The S5 character tokenizer already contains the full `<A_.....>` vocabulary.
    # Avoid calling `add_special_tokens()` which is meant for HF tokenizers.
    if tok.__class__.__name__ == "S5CharTokenizer":
        return 0

    paths = [data_args.train_data_path, data_args.eval_data_path]
    dataset_names = [data_args.train_dataset_name, data_args.eval_dataset_name]
    trigger = any(
        (
            v
            and (
                "s5_actions" in str(v).lower()
                or "s5_state" in str(v).lower()
                or "state_tracking" in str(v).lower()
            )
        )
        for v in paths + dataset_names
    )
    if not trigger:
        return 0

    action_tokens = [f"<A_{''.join(map(str, p))}>" for p in permutations((1, 2, 3, 4, 5))]
    # State-tracking reuses the A_* token family; keep only A_* specials.
    special_tokens = {"additional_special_tokens": action_tokens}
    added = tok.add_special_tokens(special_tokens)
    if added > 0:
        logging.info("Added %d S5 action/state tokens (A_*) to tokenizer vocabulary.", added)
    return added


s5_tokens_added = _maybe_add_s5_tokens(tokenizer, data_args)

if data_args.max_length is not None and data_args.max_length <= 0:
    raise ValueError("`max_length` must be a positive integer when provided.")

max_seq_length: Optional[int] = data_args.max_length

# Validate tokenizer and extract pause token ID if pause token insertion is enabled
pause_token_id = None
pause_token_mean = 0.0
if data_args.insert_pause_tokens:
    # Check if tokenizer is Llama 3
    tokenizer_name_lower = model_args.tokenizer_name_or_path.lower()
    if "llama-3" not in tokenizer_name_lower and "llama3" not in tokenizer_name_lower:
        raise ValueError(
            f"Pause token insertion requires a Llama 3 tokenizer. "
            f"Got tokenizer: {model_args.tokenizer_name_or_path}. "
            f"Expected 'llama-3' or 'llama3' in the tokenizer name."
        )
    
    # Extract pause token ID
    pause_token = "<|reserved_special_token_0|>"
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
    
    if pause_token_id == tokenizer.unk_token_id or pause_token_id < 0:
        raise ValueError(
            f"Pause token '{pause_token}' not found in tokenizer vocabulary. "
            f"Please ensure you're using a Llama 3 tokenizer with reserved special tokens."
        )
    if data_args.pause_token_mean is None:
        raise ValueError(
            "insert_pause_tokens is True but pause_token_mean was not provided. "
            "Please set --pause_token_mean to a non-negative value."
        )
    if data_args.pause_token_mean < 0:
        raise ValueError("pause_token_mean must be non-negative")

    pause_token_mean = float(data_args.pause_token_mean)

    logging.info(f"Pause token insertion enabled: '{pause_token}' (ID: {pause_token_id})")
    logging.info(f"  - Mean (per-position Poisson): {pause_token_mean}")
    logging.info(f"  - Only recurrent regions: {data_args.pause_token_only_recurrent}")
    logging.info(f"  - Random seed: {data_args.pause_token_seed}")

train_collator = None
eval_collator = None

# load model
rcot_model = None
base_model = None

model_name_key = model_args.model_name_or_path.strip().lower()
if model_name_key == "tinyllama":
    if model_args.from_pretrained:
        raise ValueError(
            f"{model_args.model_name_or_path} does not have pretrained weights; set from_pretrained=False to use it."
        )
    # Align model vocab and token IDs with the tokenizer to avoid shape mismatches
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    disable_positional_encoding = bool(getattr(model_args, "disable_positional_encoding", False))
    
    # Build TinyLlama config with optional architecture overrides
    tinyllama_kwargs = {
        "vocab_size": vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "disable_positional_encoding": disable_positional_encoding,
    }
    # Apply architecture overrides if provided
    if getattr(model_args, "tinyllama_hidden_size", None) is not None:
        tinyllama_kwargs["hidden_size"] = model_args.tinyllama_hidden_size
    if getattr(model_args, "tinyllama_num_hidden_layers", None) is not None:
        tinyllama_kwargs["num_hidden_layers"] = model_args.tinyllama_num_hidden_layers
    if getattr(model_args, "tinyllama_num_attention_heads", None) is not None:
        tinyllama_kwargs["num_attention_heads"] = model_args.tinyllama_num_attention_heads
    if getattr(model_args, "tinyllama_num_key_value_heads", None) is not None:
        tinyllama_kwargs["num_key_value_heads"] = model_args.tinyllama_num_key_value_heads
    if getattr(model_args, "tinyllama_intermediate_size", None) is not None:
        tinyllama_kwargs["intermediate_size"] = model_args.tinyllama_intermediate_size
    
    model_config = TinyLlamaConfig(**tinyllama_kwargs)
    logging.info(f"TinyLlama config: hidden_size={model_config.hidden_size}, num_hidden_layers={model_config.num_hidden_layers}, "
                 f"num_attention_heads={model_config.num_attention_heads}, num_key_value_heads={model_config.num_key_value_heads}")
    base_model = TinyLlamaForCausalLM(model_config)
    _maybe_apply_liger_kernels_to_model_instance(base_model, desc="TinyLlama base model")
    # Helpful sanity logging: confirm whether RoPE was neutralized.
    try:
        model_rotary = type(getattr(base_model.model, "rotary_emb", None)).__name__
        layer0_rotary = None
        if getattr(base_model.model, "layers", None):
            layer0_rotary = type(getattr(base_model.model.layers[0].self_attn, "rotary_emb", None)).__name__
        logging.info(
            "TinyLlama init: disable_positional_encoding=%s | model.rotary_emb=%s | layer0.self_attn.rotary_emb=%s",
            disable_positional_encoding,
            model_rotary,
            layer0_rotary,
        )
    except Exception:
        logging.info("TinyLlama init: disable_positional_encoding=%s", disable_positional_encoding)
elif model_args.model_name_or_path.strip().lower() in ("rnnlm", "rnn", "gru", "lstm"):
    if model_args.from_pretrained:
        # Allow loading a saved `rnnlm` checkpoint (or any HF checkpoint) via the generic path below.
        # Users can pass a directory path to --model_name_or_path when from_pretrained=True.
        pass
    else:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        rnn_type = model_args.rnn_type
        if model_args.model_name_or_path.strip().lower() in ("gru", "lstm", "rnn"):
            rnn_type = model_args.model_name_or_path.strip().lower()

        model_config = RNNLMConfig(
            vocab_size=vocab_size,
            hidden_size=int(model_args.rnn_hidden_size),
            num_layers=int(model_args.rnn_num_layers),
            rnn_type=str(rnn_type),
            dropout=float(model_args.rnn_dropout),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            tie_word_embeddings=True,
        )
        base_model = RNNLMForCausalLM(model_config)
else:
    is_rcot_checkpoint = False
    if model_args.from_pretrained:
        try:
            rcot_cfg = RCOTConfig.from_pretrained(model_args.model_name_or_path)
            is_rcot_checkpoint = getattr(rcot_cfg, "model_type", "") == "rcot"
        except Exception:
            is_rcot_checkpoint = False

    if model_args.from_pretrained and is_rcot_checkpoint:
        rcot_model = RCOTWrapper.from_pretrained_with_rcot(
            model_args.model_name_or_path,
            rcot_args=rcot_args,
            torch_dtype=torch.bfloat16,
            attn_impl=model_args.attn_impl,
        )
    elif model_args.from_pretrained:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=model_args.attn_impl,
            )
            _maybe_apply_liger_kernels_to_model_instance(base_model, desc="HF base model (from_pretrained)")
        except Exception:
            # The config structure now uses RCOTWrapper for all models.
            rcot_model = RCOTWrapper.from_pretrained_with_rcot(
                model_args.model_name_or_path,
                rcot_args=rcot_args,
                torch_dtype=torch.bfloat16,
                attn_impl=model_args.attn_impl,
            )
        _maybe_apply_liger_kernels_to_model_instance(base_model, desc="HF base model (from_pretrained)")
    else:
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_impl,
        )
        if "smollm2" in model_args.model_name_or_path.lower() and model_args.set_hidden_size is not None:
            # sometimes need to set hidden size to reach parameter parity.
            model_config.hidden_size = model_args.set_hidden_size
            logging.info(f"Set hidden size to {model_args.set_hidden_size} for {model_args.model_name_or_path}")

        base_model = AutoModelForCausalLM.from_config(model_config)
        _maybe_apply_liger_kernels_to_model_instance(base_model, desc="HF base model (from_config)")

if rcot_model is None:
    if base_model is None:
        raise RuntimeError("Failed to initialize base model.")

    # RNN baseline cannot be wrapped by RCOT (requires transformer blocks).
    base_type = str(getattr(base_model.config, "model_type", "") or "").lower()
    if base_type == "rnnlm":
        if rcot_args.rcot_enabled:
            raise ValueError("RCOT is not supported for `rnnlm` (non-transformer baseline). Set --rcot_enabled False.")
        rcot_model = base_model
    else:
        # Ensure attn_impl is tracked on the base config for downstream RCOT wiring
        try:
            setattr(base_model.config, "attn_impl", model_args.attn_impl)
        except Exception:
            logging.warning("Could not set attn_impl on base model config; continuing without it.")
        # Apply Liger kernels before wrapping the model in RCOT adapters/wrappers.
        _maybe_apply_liger_kernels_to_model_instance(base_model, desc="HF base model (pre-RCOT wrap)")
        rcot_model = RCOTWrapper.from_base_model(base_model, rcot_args)

# If we ended up with an RCOT wrapper (e.g., loaded from an RCOT checkpoint),
# ensure Liger kernels are applied to the underlying base model instance rather
# than the RCOT wrapper itself.
try:
    underlying = getattr(rcot_model, "rcot_model", None)
    if underlying is not None:
        _maybe_apply_liger_kernels_to_model_instance(underlying, desc="RCOT underlying base model")
except Exception:
    pass

if model_args.depth_scaling:
    from components.depth_scaling_wrapper import update_depth_scaling
    model = update_depth_scaling(rcot_model, model_args.model_name_or_path)

if s5_tokens_added > 0 and rcot_model is not None:
    rcot_model.resize_token_embeddings(len(tokenizer))
    try:
        if hasattr(rcot_model, "config") and isinstance(getattr(rcot_model, "config", None), object):
            base_cfg = getattr(rcot_model.config, "base_config", None)
            if isinstance(base_cfg, dict):
                base_cfg["vocab_size"] = len(tokenizer)
    except Exception:
        logging.warning("Failed to update RCOT base_config vocab_size; continuing.")

# load datasets (support raw jsonl paths or HF saved datasets)
def _load_path(path: str):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)
    if path.endswith(".json") or path.endswith(".jsonl"):
        # treat single file as dataset with split train
        ds = load_dataset("json", data_files=path, split="train")
        return ds
    # directory of jsonl? attempt glob
    if os.path.isdir(path):
        jsonl_files = [f for f in os.listdir(path) if f.endswith('.jsonl') or f.endswith('.json')]
        if len(jsonl_files) == 1:
            return load_dataset("json", data_files=os.path.join(path, jsonl_files[0]), split="train")
    raise ValueError(f"Unsupported dataset path format: {path}")

def _load_dataset_source(
    *,
    path: Optional[str],
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    dataset_split: Optional[str],
    default_split: str,
    dataset_role: str,
):
    if dataset_name:
        split = dataset_split or default_split
        config_label = dataset_config or "default"
        logging.info(
            "Loading %s dataset from Hugging Face hub: %s (config=%s, split=%s)",
            dataset_role,
            dataset_name,
            config_label,
            split,
        )
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, split=split)
        return load_dataset(dataset_name, split=split)
    if path:
        logging.info("Loading %s dataset from %s", dataset_role, path)
        return _load_path(path)
    raise ValueError(
        f"Insufficient information to load {dataset_role} dataset. "
        f"Please provide either dataset_name or path."
    )

def _format_gsm8k_aug_dataset(dataset, role: str, dataset_label: str):
    required_columns = {"steps", "answer"}
    if not required_columns.issubset(set(dataset.column_names)):
        return dataset

    def _format_example(example):
        answer = example.get("answer")
        if isinstance(answer, str) and ("###" in answer or "####" in answer):
            return example

        steps_field = example.get("steps")
        step_lines = []
        if isinstance(steps_field, (list, tuple)):
            for raw_step in steps_field:
                step_text = str(raw_step).strip()
                if step_text:
                    step_lines.append(step_text)
        elif isinstance(steps_field, str):
            step_text = steps_field.strip()
            if step_text:
                step_lines.append(step_text)

        reasoning = "\n".join(step_lines).strip()
        raw_answer = str(answer).strip() if answer is not None else ""

        formatted_answer = raw_answer
        if raw_answer and "###" not in raw_answer and "####" not in raw_answer:
            formatted_answer = f"### {raw_answer}"

        parts = [part for part in (reasoning, formatted_answer) if part]
        example["answer"] = "\n".join(parts) if parts else raw_answer
        return example

    map_kwargs = {
        "desc": f"Formatting gsm8k-aug reasoning for {role} dataset ({dataset_label})"
    }
    cpu_count = os.cpu_count() or 1
    if cpu_count > 1:
        map_kwargs["num_proc"] = min(8, cpu_count)

    return dataset.map(_format_example, **map_kwargs)


def _maybe_format_dataset(
    dataset,
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    role: str,
):
    if dataset is None:
        return dataset

    formatter_registry = {
        "whynlp/gsm8k-aug": _format_gsm8k_aug_dataset,
        "gsm8k-aug": _format_gsm8k_aug_dataset,
    }

    dataset_info = getattr(dataset, "info", None)
    identifiers = [
        (dataset_name or "").strip().lower(),
        getattr(dataset_info, "dataset_name", "") and getattr(dataset_info, "dataset_name").strip().lower(),
    ]

    for key in identifiers:
        if not key:
            continue
        formatter = formatter_registry.get(key)
        if formatter is None:
            continue

        dataset_label = dataset_config or dataset_name or key
        return formatter(dataset, role, dataset_label)

    return dataset

cache_exists = False
if data_args.train_tokenized_cache is not None:
    if os.path.exists(data_args.train_tokenized_cache):
        cache_exists = True
        eval_dataset = "Placeholder"

if not cache_exists:
    with _main_process_first(training_args, desc="load train dataset"):
        train_dataset = _load_dataset_source(
            path=data_args.train_data_path,
            dataset_name=data_args.train_dataset_name,
            dataset_config=data_args.train_dataset_config,
            dataset_split=data_args.train_dataset_split,
            default_split="train",
            dataset_role="train",
        )

    with _main_process_first(training_args, desc="format train dataset"):
        train_dataset = _maybe_format_dataset(
            train_dataset,
            data_args.train_dataset_name,
            data_args.train_dataset_config,
            "train",
        )

    if training_args.do_eval or training_args.do_skip_layer_eval:
        eval_dataset_name = data_args.eval_dataset_name or data_args.train_dataset_name
        eval_dataset_config = data_args.eval_dataset_config
        if eval_dataset_config is None and data_args.eval_dataset_name is None:
            eval_dataset_config = data_args.train_dataset_config
        holdout_size = getattr(data_args, "eval_holdout_size", None)
        holdout_ratio = getattr(data_args, "eval_holdout_ratio", None)
        if holdout_size is not None and holdout_ratio is not None:
            raise ValueError("Set only one of eval_holdout_size or eval_holdout_ratio.")
        if holdout_size is not None or holdout_ratio is not None:
            if data_args.eval_dataset_name or data_args.eval_data_path:
                logging.warning("Eval dataset overrides are set, but eval_holdout_* is enabled; using holdout split.")
            if not isinstance(train_dataset, Dataset):
                raise ValueError("Holdout splitting requires a map-style Dataset.")
            split_kwargs = {"seed": data_args.eval_holdout_seed or training_args.seed, "shuffle": True}
            if holdout_size is not None:
                split_kwargs["test_size"] = int(holdout_size)
            else:
                split_kwargs["test_size"] = float(holdout_ratio)
            split = train_dataset.train_test_split(**split_kwargs)
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            with _main_process_first(training_args, desc="load eval dataset"):
                eval_dataset = _load_dataset_source(
                    path=data_args.eval_data_path,
                    dataset_name=eval_dataset_name,
                    dataset_config=eval_dataset_config,
                    dataset_split=data_args.eval_dataset_split,
                    default_split="validation",
                    dataset_role="eval",
                )

            with _main_process_first(training_args, desc="format eval dataset"):
                eval_dataset = _maybe_format_dataset(
                    eval_dataset,
                    eval_dataset_name,
                    eval_dataset_config,
                    "eval",
                )
    else:
        eval_dataset = None

# preprocess datasets
is_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

rl_train_dataset: Optional[Dataset] = None
rl_eval_dataset: Optional[Dataset] = None

if training_stage == "rl":
    logging.info("Preparing datasets for GRPO training")

    response_column = eval_args.response_column
    prompt_column = eval_args.prompt_column

    def _format_conversational_dataset(dataset: Dataset, role: str) -> Dataset:
        if prompt_column not in dataset.column_names:
            raise ValueError(
                f"Dataset for {role} is missing the configured prompt column '{prompt_column}'."
            )
        if response_column not in dataset.column_names:
            raise ValueError(
                f"Dataset for {role} is missing the configured response column '{response_column}'."
            )

        def _map_example(example):
            raw_prompt = example.get(prompt_column)
            if (
                isinstance(raw_prompt, list)
                and raw_prompt
                and isinstance(raw_prompt[0], dict)
                and "role" in raw_prompt[0]
                and "content" in raw_prompt[0]
            ):
                conversation = raw_prompt
            else:
                prompt_text = "" if raw_prompt is None else str(raw_prompt)
                conversation = [{"role": "user", "content": prompt_text}]

            raw_response = example.get(response_column)
            response_text = "" if raw_response is None else str(raw_response)

            updates = {"prompt": conversation, response_column: response_text}
            if response_column != "response":
                updates["response"] = response_text
            return updates

        map_kwargs = {"desc": f"Converting {role} dataset to conversational prompts for RL"}
        cpu_count = os.cpu_count() or 1
        if cpu_count > 1:
            map_kwargs["num_proc"] = min(8, cpu_count)

        return dataset.map(_map_example, **map_kwargs)

    if training_args.do_train:
        with _main_process_first(training_args, desc="Format RL train dataset"):
            rl_train_dataset = _format_conversational_dataset(train_dataset, "train")
    if training_args.do_eval and eval_dataset is not None:
        with _main_process_first(training_args, desc="Format RL eval dataset"):
            rl_eval_dataset = _format_conversational_dataset(eval_dataset, "eval")

    if training_args.eval_samples and training_args.eval_samples > 0 and rl_eval_dataset is not None:
        subset = min(int(training_args.eval_samples), len(rl_eval_dataset))
        rl_eval_dataset = rl_eval_dataset.select(range(subset))


if training_stage != "rl":

    tokenize_kwargs = {"add_special_tokens": False}
    concat_response_to_input = bool(getattr(data_args, "concat_response_to_input", True))

    chat_template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }

    def _ensure_eos(ids):
        if tokenizer.eos_token_id is not None and (len(ids) == 0 or ids[-1] != tokenizer.eos_token_id):
            return ids + [tokenizer.eos_token_id]
        return ids

    def _maybe_get_example_control_flow(example: Dict[str, Any], prompt_len: int) -> Optional[List[int]]:
        """
        If the raw dataset provides a per-token control_flow for the prompt, use it.
        Expected: a list[int] of length == prompt_len.
        """
        cf = example.get("control_flow")
        if cf is None:
            attrs = example.get("attributes")
            if isinstance(attrs, dict):
                cf = attrs.get("control_flow")
        if cf is None:
            return None
        try:
            out = [int(x) for x in list(cf)]
        except Exception:
            return None
        if len(out) != int(prompt_len):
            raise ValueError(f"Example control_flow length {len(out)} does not match prompt_len {prompt_len}.")
        return out

    def _build_control_flow(*, prompt_len: int, response_len: int) -> List[int]:
        if prompt_len < 0 or response_len < 0:
            raise ValueError("prompt_len/response_len must be non-negative.")

        if concat_response_to_input:
            if data_args.control_flow_all_recurrent:
                return [2] * (prompt_len + response_len)
            return ([1] * prompt_len) + ([2] * response_len)

        if data_args.control_flow_all_recurrent:
            return [2] * prompt_len
        return ([2] * prompt_len) if rcot_args.rcot_enabled else ([1] * prompt_len)

    def _tokenize_prompt(prompt_text: str) -> List[int]:
        if is_chat:
            messages_prompt_only = [{"role": "user", "content": prompt_text}]
            try:
                return tokenizer.apply_chat_template(
                    messages_prompt_only,
                    **chat_template_kwargs,
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages_prompt_only, tokenize=True)
        return tokenizer(prompt_text, **tokenize_kwargs)["input_ids"]

    def _tokenize_response(response_text: str, add_eos: bool = True) -> List[int]:
        response_ids = tokenizer(response_text, **tokenize_kwargs)["input_ids"]
        return _ensure_eos(response_ids) if add_eos else response_ids

    def _align_labels_to_prompt(labels: List[int], prompt_len: int) -> List[int]:
        if prompt_len <= 0:
            return []
        if len(labels) >= prompt_len:
            return labels[:prompt_len]
        return labels + ([-100] * (prompt_len - len(labels)))

    def preprocess_train(example, prompt_col=eval_args.prompt_column, response_col=eval_args.response_column):
        prompt_text = str(example[prompt_col])
        response_text = str(example[response_col])

        prompt_ids = _tokenize_prompt(prompt_text)
        response_ids = _tokenize_response(response_text, add_eos=concat_response_to_input)
        example_cf = _maybe_get_example_control_flow(example, len(prompt_ids))
        
        if concat_response_to_input:
            input_ids = prompt_ids + response_ids
            example["input_ids"] = input_ids
            # Prompt may carry a mixed control flow; response is treated as recurrent.
            if example_cf is not None:
                example["control_flow"] = list(example_cf) + ([2] * len(response_ids))
            else:
                example["control_flow"] = _build_control_flow(prompt_len=len(prompt_ids), response_len=len(response_ids))
        else:
            # LM-style: inputs are prompt-only; labels are aligned to prompt length.
            example["input_ids"] = prompt_ids
            example["labels"] = _align_labels_to_prompt(response_ids, len(prompt_ids))
            example["control_flow"] = list(example_cf) if example_cf is not None else _build_control_flow(
                prompt_len=len(prompt_ids), response_len=len(response_ids)
            )

        example["prompt"] = prompt_text
        example["response"] = response_text
        return example

    def preprocess_eval(example, prompt_col=eval_args.prompt_column, response_col=eval_args.response_column):
        prompt_text = str(example[prompt_col])
        response_text = str(example[response_col])

        prompt_ids = _tokenize_prompt(prompt_text)
        response_ids = _tokenize_response(response_text, add_eos=concat_response_to_input)
        if not concat_response_to_input:
            response_ids = _align_labels_to_prompt(response_ids, len(prompt_ids))
        example_cf = _maybe_get_example_control_flow(example, len(prompt_ids))
        if concat_response_to_input:
            if example_cf is not None:
                example["control_flow"] = list(example_cf) + ([2] * len(response_ids))
            else:
                example["control_flow"] = _build_control_flow(prompt_len=len(prompt_ids), response_len=len(response_ids))
        else:
            example["control_flow"] = list(example_cf) if example_cf is not None else _build_control_flow(
                prompt_len=len(prompt_ids), response_len=len(response_ids)
            )

        example["input_ids"] = prompt_ids
        example["labels"] = response_ids
        example["prompt_len"] = len(prompt_ids)
        example["label_len"] = len(response_ids)

        example["prompt"] = prompt_text
        example["response"] = response_text
        return example

    def preprocess_text_only_train(example, text_col=eval_args.response_column):
        """Preprocess for text-only datasets (no prompt column).
        
        Used for language modeling on datasets like TinyStories or WikiText
        where there's just a 'text' column, not prompt-response pairs.
        """
        text = str(example[text_col])
        input_ids = tokenizer(text, **tokenize_kwargs)["input_ids"]
        input_ids = _ensure_eos(input_ids)
        
        example["input_ids"] = input_ids
        # For LM training, labels are the same as input_ids (shifted internally by the model)
        example["labels"] = input_ids.copy()
        # Control flow: all recurrent (2) since there's no prompt to distinguish
        if data_args.control_flow_all_recurrent:
            example["control_flow"] = [2] * len(input_ids)
        else:
            # First token non-recurrent, rest recurrent
            example["control_flow"] = [1] + ([2] * (len(input_ids) - 1)) if len(input_ids) > 0 else []
        
        example["prompt"] = ""
        example["response"] = text
        return example

    def preprocess_text_only_eval(example, text_col=eval_args.response_column):
        """Preprocess for text-only eval datasets (no prompt column).
        
        Used for perplexity evaluation on datasets like TinyStories or WikiText.
        """
        text = str(example[text_col])
        input_ids = tokenizer(text, **tokenize_kwargs)["input_ids"]
        input_ids = _ensure_eos(input_ids)
        
        example["input_ids"] = input_ids
        example["labels"] = input_ids.copy()
        # Control flow: all recurrent (2) since there's no prompt
        if data_args.control_flow_all_recurrent:
            example["control_flow"] = [2] * len(input_ids)
        else:
            example["control_flow"] = [1] + ([2] * (len(input_ids) - 1)) if len(input_ids) > 0 else []
        
        example["prompt_len"] = 0
        example["label_len"] = len(input_ids)
        example["prompt"] = ""
        example["response"] = text
        return example

    def _is_wikitext_title_line(text: str) -> bool:
        stripped = text.strip()
        if not (stripped.startswith("=") and stripped.endswith("=")):
            return False
        if stripped.strip("=").strip() == "":
            return False
        left = len(stripped) - len(stripped.lstrip("="))
        right = len(stripped) - len(stripped.rstrip("="))
        return left == right == 1

    def _build_lm_blocks(token_ids: List[int], max_len: Optional[int]) -> List[List[int]]:
        if not token_ids:
            return []
        if max_len is None or max_len <= 0:
            return [token_ids]
        return [token_ids[i : i + max_len] for i in range(0, len(token_ids), max_len)]

    def _tokenize_text_only_dataset(dataset: Dataset, role: str, dataset_name: Optional[str]) -> Dataset:
        is_wikitext = "wikitext" in (dataset_name or "").lower()
        eos_id = tokenizer.eos_token_id

        def _append_eos_if_needed(buffer: List[int]) -> None:
            if eos_id is None or not buffer or buffer[-1] == eos_id:
                return
            buffer.append(eos_id)

        def _tokenize_batch(examples, text_col=eval_args.response_column):
            texts = examples.get(text_col, [])
            all_ids: List[int] = []

            for text in texts:
                line = "" if text is None else str(text)
                stripped = line.strip()
                if stripped == "":
                    continue

                if is_wikitext and _is_wikitext_title_line(stripped):
                    _append_eos_if_needed(all_ids)

                ids = tokenizer(line, **tokenize_kwargs)["input_ids"]
                if ids:
                    all_ids.extend(ids)
                    if not is_wikitext:
                        _append_eos_if_needed(all_ids)

            blocks = _build_lm_blocks(all_ids, max_seq_length)
            input_ids = []
            labels = []
            control_flow = []
            prompt_len = []
            label_len = []
            prompt = []
            response = []

            for ids in blocks:
                if not ids:
                    continue
                input_ids.append(ids)
                labels.append(ids.copy())
                if data_args.control_flow_all_recurrent:
                    cf = [2] * len(ids)
                else:
                    cf = [1] + ([2] * (len(ids) - 1))
                control_flow.append(cf)
                prompt_len.append(0)
                label_len.append(len(ids))
                prompt.append("")
                response.append("")

            return {
                "input_ids": input_ids,
                "labels": labels,
                "control_flow": control_flow,
                "prompt_len": prompt_len,
                "label_len": label_len,
                "prompt": prompt,
                "response": response,
            }

        map_kwargs = {
            "desc": f"Tokenizing {role} dataset (text-only, concatenated)",
            "batched": True,
            "remove_columns": dataset.column_names,
            "num_proc": dataset_num_proc,
        }
        return dataset.map(_tokenize_batch, **map_kwargs)

    def insert_pause_tokens_preprocessing(example, idx):
        """Insert pause tokens using per-position Poisson sampling."""

        if pause_token_id is None or pause_token_mean <= 0.0:
            return example

        rng = np.random.default_rng(data_args.pause_token_seed + idx)

        original_input_ids = list(example["input_ids"])
        original_control_flow = list(example["control_flow"])

        input_ids = original_input_ids.copy()
        control_flow = original_control_flow.copy()

        valid_positions = [
            pos for pos, cf in enumerate(original_control_flow)
            if not data_args.pause_token_only_recurrent or cf > 1
        ]

        if not valid_positions:
            return example

        offset = 0
        for pos in valid_positions:
            k = rng.poisson(pause_token_mean)
            if k <= 0:
                continue

            cf_value = original_control_flow[pos]
            insert_index = pos + offset

            for _ in range(k):
                input_ids.insert(insert_index, pause_token_id)
                control_flow.insert(insert_index, cf_value)
                insert_index += 1
                offset += 1

        example["input_ids"] = input_ids
        example["control_flow"] = control_flow

        return example

    def truncate_processed_example(example):
        if max_seq_length is not None and max_seq_length > 0:
            for key in ("input_ids", "labels", "attention_mask", "control_flow"):
                if key not in example or example[key] is None:
                    continue
                seq = list(example[key])
                if len(seq) > max_seq_length:
                    print(f"Truncating sequence from {len(seq)} to {max_seq_length}")
                    seq = seq[:max_seq_length]
                example[key] = seq

        input_ids = example.get("input_ids")
        if input_ids is not None:
            example["seq_len"] = len(input_ids)

        control_flow = example.get("control_flow")
        if control_flow is not None:
            nonrecur = len(control_flow)
            for idx, value in enumerate(control_flow):
                if value > 1:
                    nonrecur = idx
                    break
            example["nonrecur_len"] = nonrecur

        if "prompt_len" in example and input_ids is not None:
            example["prompt_len"] = min(example["prompt_len"], len(input_ids))

        if "labels" in example:
            example["label_len"] = len(example["labels"])

        return example

    train_collator = rcot_collator(tokenizer, rcot_args)
    dataset_num_proc = getattr(training_args, "dataset_num_proc", None) or training_args.dataloader_num_workers

    # Determine which preprocessing function to use based on whether prompt_column is provided
    use_text_only_preprocessing = should_run_perplexity_eval(data_args) # eval_args.prompt_column == ""

    if use_text_only_preprocessing:
        logging.info("prompt_column is None - using text-only preprocessing (no prompt/response split)")
    else:
        train_preprocess_fn = preprocess_train
        eval_preprocess_fn = preprocess_eval

    # Initialize loaded_from_cache flag to track if dataset was loaded from cache
    loaded_from_cache = False

    if cache_exists or 'input_ids' not in train_dataset.column_names:
        if use_text_only_preprocessing:
            with _main_process_first(training_args, desc="Tokenizing train dataset (text-only)"):
                
                loaded_from_cache = False
                if data_args.train_tokenized_cache is not None:
                    if os.path.exists(data_args.train_tokenized_cache):
                        print(f"Loading tokenized train dataset from {data_args.train_tokenized_cache}...")
                        train_dataset = load_from_disk(data_args.train_tokenized_cache)
                        loaded_from_cache = True
                if not loaded_from_cache:
                    print(f"Tokenizing train dataset...")
                    train_dataset = _tokenize_text_only_dataset(
                        train_dataset,
                        role="train",
                        dataset_name=data_args.train_dataset_name,
                    )
                    if data_args.train_tokenized_cache is not None:
                        print(f"Saving tokenized train dataset to {data_args.train_tokenized_cache}...")
                        train_dataset.save_to_disk(data_args.train_tokenized_cache)
        else:
            with _main_process_first(training_args, desc="Tokenizing train dataset"):
                train_dataset = train_dataset.map(train_preprocess_fn, desc="Tokenizing train dataset", num_proc=dataset_num_proc)

        if (training_args.do_eval or training_args.do_skip_layer_eval) and eval_dataset is not None:
            if use_text_only_preprocessing:
                with _main_process_first(training_args, desc="Tokenizing eval dataset (text-only)"):
                    
                    loaded_from_cache = False
                    if data_args.eval_tokenized_cache is not None:
                        if os.path.exists(data_args.eval_tokenized_cache):
                            print(f"Loading tokenized eval dataset from {data_args.eval_tokenized_cache}...")
                            eval_dataset = load_from_disk(data_args.eval_tokenized_cache)
                            loaded_from_cache = True
                    if not loaded_from_cache:
                        print(f"Tokenizing eval dataset...")
                        eval_dataset = _tokenize_text_only_dataset(
                            eval_dataset,
                            role="eval",
                            dataset_name=eval_dataset_name,
                        )
                        if data_args.eval_tokenized_cache is not None:
                            print(f"Saving tokenized eval dataset to {data_args.eval_tokenized_cache}...")
                            eval_dataset.save_to_disk(data_args.eval_tokenized_cache)
            else:
                with _main_process_first(training_args, desc="Tokenizing eval dataset"):
                    eval_dataset = eval_dataset.map(eval_preprocess_fn, desc="Tokenizing eval dataset", num_proc=dataset_num_proc)
        
        if data_args.insert_pause_tokens:
            logging.info("Applying pause token insertion to training dataset...")
            with _main_process_first(training_args, desc="Inserting pause tokens"):
                train_dataset = train_dataset.map(
                    insert_pause_tokens_preprocessing,
                    with_indices=True,
                    desc="Inserting pause tokens",
                    num_proc=dataset_num_proc,
                )
            logging.info("Pause token insertion complete. Dataset cached for future runs.")
        
        # For LM-style evaluation (prompt-only inputs + per-token labels), use the standard
        # collator so the forward pass receives aligned labels and control_flows.
        # NOTE: When RCOT is enabled we *must* include `control_flows` in the eval batch,
        # otherwise the RCOT recurrent path is bypassed (control_flows defaults to ones).
        # For text-only preprocessing, always use rcot_collator since we include control_flow.
        if use_text_only_preprocessing or not concat_response_to_input:
            eval_collator = rcot_collator(tokenizer, rcot_args)
        else:
            eval_collator = rcot_collator(tokenizer, rcot_args) if rcot_args.rcot_enabled else RCOTEvalCollator(tokenizer)
    else:
        if data_args.insert_pause_tokens:
            logging.warning(
                "Pause token insertion is enabled but dataset is already preprocessed. "
                "Pause tokens will NOT be inserted. Please preprocess from raw data."
            )
        eval_collator = rcot_collator(tokenizer, rcot_args)

    # Train postprocessing is only needed when truncation is enabled.
    # When max_seq_length is None ("no truncation"), mapping over the full dataset is wasted work.

    # LM data construction already handles truncation, so we don't need to do it here.
    if max_seq_length is None:
        logging.info("Skipping train postprocessing because truncation is disabled (max_length=None).")
    else:
        if loaded_from_cache:
            logging.info("Skipping train postprocessing because dataset is loaded from cache with proper truncation.")
        else:
            map_kwargs = {"desc": f"Applying train postprocessing (truncate to {max_seq_length})"}
            if dataset_num_proc and dataset_num_proc > 1:
                map_kwargs["num_proc"] = dataset_num_proc
            with _main_process_first(training_args, desc="Applying train postprocessing"):
                train_dataset = train_dataset.map(truncate_processed_example, **map_kwargs)

else:
    train_collator = None
    eval_collator = None

# initialize trainer
if training_stage == "rl":
    from components.rcot_grpo_trainer import RCOTGRPOTrainer
    grpo_config = GRPOConfig()

    for field in fields(GRPOConfig):
        name = field.name
        if hasattr(training_args, name):
            setattr(grpo_config, name, getattr(training_args, name))

    if remaining_cli:
        grpo_parser = HfArgumentParser((GRPOConfig,))
        grpo_cli_config = grpo_parser.parse_args_into_dataclasses(args=remaining_cli)[0]
        grpo_default = GRPOConfig()
        for field in fields(GRPOConfig):
            name = field.name
            value = getattr(grpo_cli_config, name)
            default_value = getattr(grpo_default, name)
            if value != default_value:
                setattr(grpo_config, name, value)

    if not getattr(grpo_config, "output_dir", None):
        grpo_config.output_dir = training_args.output_dir

    grpo_config.remove_unused_columns = False
    reward_function = build_rl_reward_function(eval_args)

    trainer = RCOTGRPOTrainer(
        model=rcot_model,
        reward_funcs=reward_function,
        args=grpo_config,
        train_dataset=rl_train_dataset if training_args.do_train else None,
        eval_dataset=rl_eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
    )
else:
    compute_metrics = None
    preprocess_logits_for_metrics = None
    if (
        training_stage != "rl"
        and not bool(getattr(data_args, "concat_response_to_input", True))
    ):
        import numpy as np

        eval_length_values = None
        if eval_dataset is not None and hasattr(eval_dataset, "column_names") and "length" in eval_dataset.column_names:
            try:
                eval_length_values = list(eval_dataset["length"])
            except Exception:
                eval_length_values = None

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_pred):
            preds = eval_pred.predictions
            labels = eval_pred.label_ids

            preds = np.asarray(preds)
            labels = np.asarray(labels)

            valid = labels != -100
            token_total = int(valid.sum())
            if token_total == 0:
                return {"sequence_accuracy": 0.0}

            per_example_total = valid.sum(axis=1)
            per_example_correct = ((preds == labels) & valid).sum(axis=1)
            nonempty = per_example_total > 0
            seq_acc = float((per_example_correct[nonempty] == per_example_total[nonempty]).mean()) if nonempty.any() else 0.0

            metrics = {
                "sequence_accuracy": float(seq_acc),
            }

            inputs = getattr(eval_pred, "inputs", None)
            lengths = None
            if isinstance(inputs, dict) and "length" in inputs:
                lengths = np.asarray(inputs["length"])
            elif eval_length_values is not None:
                # Fallback: transformers may not propagate non-model inputs into `eval_pred.inputs`.
                lengths = eval_length_values[: labels.shape[0]]

            if lengths is not None:
                bins: Dict[tuple, list[int]] = {}
                if isinstance(lengths, list):
                    for i, v in enumerate(lengths):
                        key = tuple(v) if isinstance(v, (list, tuple)) else (int(v),)
                        bins.setdefault(key, []).append(i)
                else:
                    lengths = np.asarray(lengths)
                    if lengths.ndim == 1:
                        for i, v in enumerate(lengths.tolist()):
                            bins.setdefault((int(v),), []).append(i)
                    else:
                        for i in range(lengths.shape[0]):
                            row = lengths[i].tolist()
                            key_vals = tuple(int(x) for x in row if int(x) != -1)
                            bins.setdefault(key_vals, []).append(i)

                for key, idxs in bins.items():
                    if not idxs:
                        continue
                    idx = np.asarray(idxs, dtype=np.int64)
                    labels_L = labels[idx]
                    preds_L = preds[idx]
                    valid_L = labels_L != -100
                    token_total_L = int(valid_L.sum())
                    if token_total_L == 0:
                        continue

                    per_example_total_L = valid_L.sum(axis=1)
                    per_example_correct_L = ((preds_L == labels_L) & valid_L).sum(axis=1)
                    nonempty_L = per_example_total_L > 0
                    seq_acc_L = float((per_example_correct_L[nonempty_L] == per_example_total_L[nonempty_L]).mean()) if nonempty_L.any() else 0.0

                    key_name = "_".join(map(str, key)) if key else "empty"
                    metrics[f"sequence_accuracy_len_{key_name}"] = float(seq_acc_L)
                    metrics[f"n_len_{key_name}"] = int(len(idxs))

            return metrics

    # For debugging
    # train_dataset = train_dataset.take(200)

    trainer = RCOTTrainer(
        model=rcot_model,
        training_args=training_args,
        rcot_args=rcot_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        train_data_collator=train_collator,
        eval_data_collator=eval_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if _should_log_to_wandb(training_args):
        _log_wandb_config(
            training_args=training_args,
            model_args=model_args,
            data_args=data_args,
            rcot_args=rcot_args,
            eval_args=eval_args,
        )

if training_args.do_train:
    print("start training...")
    print("=" * 80)
    print("[TRAINING DEBUG] Dataset sizes and trainer step settings")
    if train_dataset is None:
        print("[TRAINING DEBUG] train_dataset: None")
    else:
        try:
            print(f"[TRAINING DEBUG] train_dataset length: {len(train_dataset)}")
        except Exception as exc:
            print(f"[TRAINING DEBUG] train_dataset length: <error: {exc}>")
    print(f"[TRAINING DEBUG] max_steps: {getattr(training_args, 'max_steps', None)}")
    print(f"[TRAINING DEBUG] num_train_epochs: {getattr(training_args, 'num_train_epochs', None)}")
    print(f"[TRAINING DEBUG] per_device_train_batch_size: {getattr(training_args, 'per_device_train_batch_size', None)}")
    print(f"[TRAINING DEBUG] gradient_accumulation_steps: {getattr(training_args, 'gradient_accumulation_steps', None)}")
    print(f"[TRAINING DEBUG] world_size: {getattr(training_args, 'world_size', None)}")
    print("=" * 80)

    print("[TRAINING DEBUG] resume_from_checkpoint: ", training_args.resume_from_checkpoint)
    if training_args.resume_from_checkpoint == 'False':
        training_args.resume_from_checkpoint = None
        resume_checkpoint_path = None
        if is_head_process and os.path.exists(training_args.output_dir):
            import shutil
            # This is necessary for a corner case of fsdp checkpoints saving.
            try:
                shutil.rmtree(training_args.output_dir)
            except FileNotFoundError as e:
                logging.info(f"Error deleting output directory {training_args.output_dir}: {e}")
                pass
        if using_multiple_processes:
            _wait_for_everyone(training_args)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint_path or None)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    save_fsdp_model(trainer)

if training_args.do_eval:    
    # Run perplexity evaluation for NLP datasets (TinyStories, WikiText)
    if should_run_perplexity_eval(data_args):
        logging.info("=" * 50)
        logging.info("Running Trainer Evaluation (TinyStories/WikiText)")
        logging.info("=" * 50)
        gen_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        # Old perplexity-specific evaluation kept for reference:
        # logging.info("=" * 50)
        # logging.info("Running Perplexity Evaluation")
        # logging.info("=" * 50)
        # ppl_metrics = run_perplexity_evaluation(
        #     trainer=trainer,
        #     tokenizer=tokenizer,
        #     eval_dataset=eval_dataset,
        #     rcot_args=rcot_args,
        #     training_args=training_args,
        # )
        # trainer.log_metrics("eval_ppl", ppl_metrics)
        # trainer.save_metrics("eval_ppl", ppl_metrics)
    else:
        # reward_mode already validated implicitly by usage; defaults handled in eval_args
        gen_metrics = run_generation_evaluation(
            trainer=trainer,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            rcot_args=rcot_args,
            eval_args=eval_args,
            training_args=training_args,
            model_args=model_args,
            control_flow_all_recurrent=bool(getattr(data_args, "control_flow_all_recurrent", False)),
        )
    trainer.log_metrics("eval", gen_metrics)
    trainer.save_metrics("eval", gen_metrics)


# Skip-layer evaluation: compute future token probabilities with skipped layers
if training_args.do_skip_layer_eval and eval_dataset is not None:
    from rcot_wrapper.skip_layer_inference_wrapper import SkipLayerInferenceWrapper
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    logging.info("=" * 50)
    logging.info("Starting Skip-Layer Evaluation")
    logging.info("=" * 50)
    logging.info(f"  Layers to skip: {training_args.skip_layer_eval_layers_to_skip}")
    logging.info(f"  RCOT enabled: {training_args.skip_layer_eval_rcot_enabled}")
    logging.info(f"  Batch size: {training_args.skip_layer_eval_batch_size}")
    logging.info(f"  Num future tokens: {training_args.skip_layer_eval_num_future_tokens}")
    logging.info(f"  Eval samples: {len(eval_dataset)}")
    
    # Create the skip-layer wrapper
    skip_layer_wrapper = SkipLayerInferenceWrapper(
        model=rcot_model,
        num_layers_to_skip=training_args.skip_layer_eval_layers_to_skip,
        rcot_enabled=training_args.skip_layer_eval_rcot_enabled,
    )
    logging.info(f"Created wrapper: {skip_layer_wrapper}")
    
    # Put model in eval mode
    skip_layer_wrapper.eval()
    
    # Prepare evaluation
    num_future_tokens = training_args.skip_layer_eval_num_future_tokens
    all_probs_sum = 0.0
    all_probs_per_offset = [0.0] * num_future_tokens  # Track per-offset probabilities
    total_tokens = 0
    total_tokens_per_offset = [0] * num_future_tokens
    
    # Create collator for skip-layer evaluation
    skip_layer_collator = SkipLayerEvalCollator(
        tokenizer=tokenizer,
        include_control_flows=training_args.skip_layer_eval_rcot_enabled is not False,
    )
    
    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.skip_layer_eval_batch_size,
        shuffle=False,
        collate_fn=skip_layer_collator,
        num_workers=0,
    )
    
    # Get device and dtype
    rcot_model.to("cuda")
    device = next(rcot_model.parameters()).device
    
    # Determine dtype for autocast (bf16 or fp16)
    use_bf16 = getattr(training_args, 'bf16', False)
    use_fp16 = getattr(training_args, 'fp16', False)
    use_autocast = use_bf16 or use_fp16
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    
    if use_autocast:
        logging.info(f"  Using autocast with dtype: {autocast_dtype}")

    # Evaluate
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Skip-layer eval", disable=not is_head_process):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            control_flows = batch.get("control_flow")
            if control_flows is not None:
                control_flows = control_flows.to(device)
            
            # Compute future token probabilities
            # probs shape: (batch_size, num_positions, num_future_tokens)
            # indices shape: (num_positions, num_future_tokens)
            # Use autocast for fp16/bf16 to ensure inputs are in correct dtype
            autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else nullcontext()
            # autocast_ctx = nullcontext()
            with autocast_ctx:
                probs, indices = skip_layer_wrapper.compute_future_token_probabilities(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    control_flows=control_flows,
                    return_log_probs=False,
                    num_future_tokens=num_future_tokens,
                )
            
            batch_size, num_positions, K = probs.shape
            assert K == num_future_tokens, f"Expected {num_future_tokens} future tokens, got {K}"
            
            # Create attention mask for the positions we're evaluating
            # indices[i, k] gives the position of the k-th future token for starting position i
            # We need to check if those positions are valid (not padding)
            
            # Create a mask for valid predictions: (batch_size, num_positions, num_future_tokens)
            valid_mask = torch.zeros_like(probs, dtype=torch.bool)
            
            for k in range(num_future_tokens):
                # For the k-th future token offset, check which positions are valid
                # indices[:, k] gives the target token positions for this offset
                target_positions = indices[:, k]  # (num_positions,)
                
                # Gather attention mask values for these positions
                # attention_mask shape: (batch_size, seq_len)
                # target_positions: (num_positions,) - same for all batches
                target_mask = attention_mask[:, target_positions]  # (batch_size, num_positions)
                valid_mask[:, :, k] = target_mask
            
            # Apply mask and sum probabilities
            valid_probs = probs * valid_mask.float()
            
            # Overall statistics
            all_probs_sum += valid_probs.sum().item()
            total_tokens += valid_mask.sum().item()
            
            # Per-offset statistics
            for k in range(num_future_tokens):
                all_probs_per_offset[k] += valid_probs[:, :, k].sum().item()
                total_tokens_per_offset[k] += valid_mask[:, :, k].sum().item()
    
    # Compute averages
    avg_prob = all_probs_sum / total_tokens if total_tokens > 0 else 0.0
    avg_probs_per_offset = [
        (all_probs_per_offset[k] / total_tokens_per_offset[k]) if total_tokens_per_offset[k] > 0 else 0.0
        for k in range(num_future_tokens)
    ]
    
    skip_layer_metrics = {
        "skip_layer_eval/avg_token_probability": avg_prob,
        "skip_layer_eval/total_tokens": total_tokens,
        "skip_layer_eval/layers_skipped": training_args.skip_layer_eval_layers_to_skip,
        "skip_layer_eval/rcot_enabled": skip_layer_wrapper._rcot_enabled,
        "skip_layer_eval/num_future_tokens": num_future_tokens,
    }
    
    # Add per-offset metrics
    for k in range(num_future_tokens):
        skip_layer_metrics[f"skip_layer_eval/avg_prob_offset_{k+1}"] = avg_probs_per_offset[k]
        skip_layer_metrics[f"skip_layer_eval/total_tokens_offset_{k+1}"] = total_tokens_per_offset[k]
    
    logging.info("=" * 50)
    logging.info("Skip-Layer Evaluation Results")
    logging.info("=" * 50)
    for key, value in skip_layer_metrics.items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 50)
    
    # Log and save metrics
    trainer.log_metrics("skip_layer_eval", skip_layer_metrics)
    trainer.save_metrics("skip_layer_eval", skip_layer_metrics)

if _should_log_to_wandb(training_args):
    try:
        import wandb  # type: ignore
    except Exception as e:
        logging.warning("W&B not available; skipping wandb.finish(). (%s)", e)
    else:
        if wandb.run is not None:
            wandb.finish()
