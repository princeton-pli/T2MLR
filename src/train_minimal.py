import os
import json
import logging

import torch
from typing import Optional
from dataclasses import asdict
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback

from components.all_arguments import (
    ModelArguments,
    TrainingArguments,
    DataArguments,
    RCOTArguments,
    GenerationEvalArguments,
)
from components.data_utils import rcot_collator, RCOTPaddingFreeCollator
from components.generation_eval import run_generation_evaluation
from components.rcot_trainer import RCOTTrainer
from components.dataset_preprocessing import (  # type: ignore
    RCOTCtrlFlowTokenizer,
    build_truncate_fn,
)
from components.custom_dataset_preprocessing import apply_custom_preprocessing  # pyright: ignore[reportMissingImports]
from components.custom_dataset_postprocessing import apply_custom_postprocessing  # pyright: ignore[reportMissingImports]
from components.rcot_utils import visualize_sample_control_flow  # pyright: ignore[reportMissingImports]
from rcot_wrapper import RCOTWrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _has_tokenizer_files(path: str) -> bool:
    return any(
        os.path.exists(os.path.join(path, name))
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
    )


def _resolve_tokenizer_path(tokenizer_name_or_path: Optional[str], model_name_or_path: str) -> str:
    candidate = tokenizer_name_or_path or model_name_or_path
    if os.path.isdir(candidate):
        if _has_tokenizer_files(candidate):
            return candidate
        parent = os.path.dirname(candidate.rstrip("/"))
        if parent and os.path.isdir(parent) and _has_tokenizer_files(parent):
            logger.info("Tokenizer files not found in %s; falling back to parent %s", candidate, parent)
            return parent
    return candidate


def _resolve_resume_checkpoint_path(training_args: TrainingArguments) -> Optional[str]:
    """Resolve resume_from_checkpoint for trainer state resume."""
    resume_flag = getattr(training_args, "resume_from_checkpoint", None)

    if isinstance(resume_flag, str):
        raw = resume_flag.strip()
        flag = raw.lower()
        if flag in {"", "false", "0", "no", "none"}:
            return None
        if flag in {"true", "1", "yes"}:
            return get_last_checkpoint(training_args.output_dir)
        if os.path.isdir(raw):
            latest = get_last_checkpoint(raw)
            if latest:
                return latest
            if os.path.exists(os.path.join(raw, "config.json")):
                return raw
        return raw

    if resume_flag is True:
        return get_last_checkpoint(training_args.output_dir)

    return None


def _resolve_model_weight_path(model_name_or_path: str) -> str:
    """Resolve model_name_or_path to a loadable weight path.

    Prefer run-root final weights when available (e.g., finished training runs).
    Fall back to latest checkpoint-* only when run root is not loadable yet.
    """
    if os.path.isdir(model_name_or_path):
        base = os.path.basename(model_name_or_path.rstrip("/"))
        if not base.startswith("checkpoint-"):
            if os.path.exists(os.path.join(model_name_or_path, "config.json")):
                return model_name_or_path
            latest = get_last_checkpoint(model_name_or_path)
            if latest:
                logger.info(
                    "Run root has no config.json yet; resolved model_name_or_path %s to latest checkpoint %s",
                    model_name_or_path,
                    latest,
                )
                return latest
    return model_name_or_path


class WandbConfigUploadCallback(TrainerCallback):
    """Callback to upload config to wandb when training begins."""
    
    def __init__(
        self,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        data_args: DataArguments,
        rcot_args: RCOTArguments,
        eval_args: GenerationEvalArguments,
    ):
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.rcot_args = rcot_args
        self.eval_args = eval_args
        self.config_uploaded = False
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Upload config to wandb when training begins (wandb should be initialized by now)."""
        if self.config_uploaded:
            return
        
        if os.environ.get("WANDB_DISABLED", "").strip().lower() in {"true", "1", "yes"}:
            return
        
        try:
            import wandb  # type: ignore
        except Exception as e:
            logger.warning("W&B not available; skipping config upload. (%s)", e)
            return
        
        # At this point, wandb should be initialized by the Trainer's callback system
        config_payload = {
            "training": getattr(self.training_args, "to_dict", lambda: asdict(self.training_args))(),
            "model": asdict(self.model_args),
            "data": asdict(self.data_args),
            "rcot": asdict(self.rcot_args),
            "eval": asdict(self.eval_args),
        }
        
        try:
            if wandb.run is None:
                logger.warning("W&B run not initialized in on_train_begin; skipping config upload.")
                return
            
            wandb.config.update(config_payload, allow_val_change=True)
            logger.info("Successfully uploaded all config arguments to W&B.")
            self.config_uploaded = True
        except Exception as e:
            logger.warning("Failed to upload config to W&B. (%s)", e)


def _should_log_to_wandb(training_args: TrainingArguments) -> bool:
    report_to = getattr(training_args, "report_to", None)
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


def _load_dataset(
    *,
    source: str,
    split: str,
    config: Optional[str] = None,
):
    # Check if it's a saved dataset directory (has dataset_info.json)
    if os.path.isdir(source) and os.path.exists(os.path.join(source, "dataset_info.json")):
        return load_from_disk(source)
    
    # Check if it's a JSON/JSONL file
    if os.path.isfile(source) and (source.endswith(".json") or source.endswith(".jsonl")):
        return load_dataset("json", data_files=source, split=split)
    
    # Check if it's a directory containing JSON/JSONL files
    if os.path.isdir(source):
        json_files = [f for f in os.listdir(source) if f.endswith('.jsonl') or f.endswith('.json')]
        if len(json_files) == 1:
            return load_dataset("json", data_files=os.path.join(source, json_files[0]), split=split)
        elif len(json_files) > 1:
            # Multiple JSON files - load all of them
            json_paths = [os.path.join(source, f) for f in json_files]
            return load_dataset("json", data_files=json_paths, split=split)
    
    # Fall back to HuggingFace dataset loading
    return load_dataset(source, config, split=split) if config else load_dataset(source, split=split)


def _build_pause_token_config(tokenizer: PreTrainedTokenizer, data_args: DataArguments, model_args: ModelArguments):
    insert_enabled = bool(getattr(data_args, "insert_pause_tokens", False))
    replace_prob = getattr(data_args, "pause_token_replace_prob", None)
    replace_schedule = str(getattr(data_args, "pause_token_replace_prob_schedule", "none") or "none").strip().lower()
    replace_enabled = replace_prob is not None and float(replace_prob) > 0.0
    replace_scheduled = replace_schedule not in {"", "none"}
    if not (insert_enabled or replace_enabled or replace_scheduled):
        return None

    mean = getattr(data_args, "pause_token_mean", None)
    if insert_enabled:
        if mean is None:
            raise ValueError("insert_pause_tokens is True but pause_token_mean was not provided.")
        if mean < 0:
            raise ValueError("pause_token_mean must be non-negative.")
        if mean == 0:
            mean = None

    pause_token = getattr(data_args, "pause_token_string", None) or "<|reserved_special_token_0|>"
    tokenizer_name = (
        (getattr(model_args, "tokenizer_name_or_path", None) or getattr(model_args, "model_name_or_path", "") or "")
    ).lower()
    if pause_token == "<|reserved_special_token_0|>":
        if "llama-3" not in tokenizer_name and "llama3" not in tokenizer_name:
            raise ValueError(
                f"Pause token insertion requires a Llama 3 tokenizer unless pause_token_string is set. "
                f"Got {getattr(model_args, 'tokenizer_name_or_path', None)}."
            )
    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if (unk is not None and pause_token_id == unk) or pause_token_id < 0:
        raise ValueError(f"Pause token '{pause_token}' not found in tokenizer vocabulary.")

    if replace_scheduled and replace_schedule != "linear":
        raise ValueError("pause_token_replace_prob_schedule must be 'none' or 'linear'.")

    logger.info(
        "Pause token handling enabled (collator-time): '%s' (ID: %s), mean=%s, replace_prob=%s, seed=%s",
        pause_token,
        pause_token_id,
        mean if mean is not None else 0.0,
        replace_prob if replace_prob is not None else 0.0,
        getattr(data_args, "pause_token_seed", 42),
    )
    return {
        "pause_token_id": int(pause_token_id),
        "pause_token_mean": None if mean is None else float(mean),
        "pause_token_seed": int(getattr(data_args, "pause_token_seed", 42)),
        "pause_token_only_recurrent": bool(getattr(data_args, "pause_token_only_recurrent", True)),
        "pause_token_control_flow_value": 2,
        "pause_token_replace_prob": None if replace_prob is None else float(replace_prob),
        "pause_token_replace_only_recurrent": bool(getattr(data_args, "pause_token_replace_only_recurrent", True)),
        "pause_token_replace_control_flow_value": 2,
    }


def main():
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments, DataArguments, RCOTArguments, GenerationEvalArguments)
    )
    model_args, training_args, data_args, rcot_args, eval_args = parser.parse_args_into_dataclasses()
    # Liger is optional; if enabled we import it and later apply it to the *base* model
    # (before RCOT wrapping) so dispatch sees the underlying transformer `model_type`.
    liger_requested = bool(getattr(training_args, "use_liger_kernel", False))
    liger_enabled = False
    if liger_requested:
        try:
            import liger_kernel  # type: ignore
            liger_enabled = True
            logger.info("Liger kernels enabled (liger_kernel imported successfully).")
        except Exception as e:
            logger.warning("--use_liger_kernel set but Liger unavailable; continuing without it. (%s)", e)

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

    os.makedirs(training_args.output_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = training_args.project_name
    os.environ["WANDB_DIR"] = training_args.output_dir
    if getattr(training_args, "run_name", None):
        os.environ["WANDB_NAME"] = training_args.run_name

    training_args.remove_unused_columns = False
    training_args.pid = os.getpid()
    set_seed(training_args.seed)

    tokenizer_path = _resolve_tokenizer_path(model_args.tokenizer_name_or_path, model_args.model_name_or_path)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    # Force left padding for decoder-only generation to avoid misalignment in eval/inference.
    tokenizer.padding_side = "left"

    pause_token_cfg = _build_pause_token_config(tokenizer, data_args, model_args)

    if getattr(data_args, "padding_free", False):
        train_collator = RCOTPaddingFreeCollator(
            return_flash_attn_kwargs=bool(getattr(data_args, "padding_free_return_flash_attn_kwargs", True)),
            **(pause_token_cfg or {}),
        )
    else:
        train_collator = rcot_collator(tokenizer, rcot_args, **(pause_token_cfg or {}))
    # For decoder-only generation, left-padding is required for correct results when prompts
    # are padded to a common length during evaluation/inference.
    # Skip get_labels_from_control_flow for eval data since labels are pre-set during preprocessing
    eval_collator = rcot_collator(tokenizer, rcot_args, pad_prompt_left=True, skip_labels_from_control_flow=True)

    torch_dtype = torch.bfloat16 if getattr(training_args, "bf16", False) else None
    # Try RCOT checkpoint first; fall back to base HF model when not RCOT.
    model = None
    resume_checkpoint_path = _resolve_resume_checkpoint_path(training_args)
    should_resume_from_checkpoint = resume_checkpoint_path is not None
    model_path = _resolve_model_weight_path(model_args.model_name_or_path)
    logger.info("Loading model from model_name_or_path: %s (resume=%s)", model_path, should_resume_from_checkpoint)
    if should_resume_from_checkpoint:
        logger.info("Resolved resume_from_checkpoint path: %s", resume_checkpoint_path)
    attn_impl = model_args.attn_impl

    if not torch.cuda.is_available():
        attn_impl = "sdpa"
    
    if getattr(model_args, "from_pretrained", True):
        # Load pretrained weights from model_name_or_path (resolved to latest checkpoint
        # automatically when a run directory is provided).
        def _load_pretrained_or_rcot(load_path: str):
            try:
                loaded = RCOTWrapper.from_pretrained_with_rcot(
                    load_path,
                    dtype=torch_dtype,
                    attn_impl=attn_impl,
                )
                logger.info("Loaded RCOT checkpoint with RCOTWrapper from: %s", load_path)
                return loaded
            except Exception as e:
                logger.info("RCOT load failed for %s, falling back to base model load. (%s)", load_path, e)
                return AutoModelForCausalLM.from_pretrained(
                    load_path,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl,
                )

        model = _load_pretrained_or_rcot(model_path)
    else:
        # Initialize from config only (random weights for training from scratch)
        logger.info("Initializing model from config (from_pretrained=False)")
        model_name_key = str(model_path).strip().lower()
        if model_name_key == "tinyllama":
            if getattr(model_args, "from_pretrained", False):
                raise ValueError(
                    "tinyllama does not have pretrained weights; set from_pretrained=False to use it."
                )
            from modeling.tinyllama import TinyLlamaConfig, TinyLlamaForCausalLM

            tok_vocab = getattr(tokenizer, "vocab_size", None) or 0
            tok_len = len(tokenizer)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
            bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
            vocab_size = max(tok_vocab, tok_len, pad_id + 1, bos_id + 1, eos_id + 1)
            tinyllama_kwargs = {
                "vocab_size": vocab_size,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "disable_positional_encoding": bool(getattr(model_args, "disable_positional_encoding", False)),
            }
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

            config = TinyLlamaConfig(**tinyllama_kwargs)
            if attn_impl:
                try:
                    config._attn_implementation = attn_impl
                except Exception:
                    pass
            model = TinyLlamaForCausalLM(config)
        else:
            config = AutoConfig.from_pretrained(model_path)
            if attn_impl:
                config._attn_implementation = attn_impl
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    # Apply Liger kernels to the underlying HF model instance (not the RCOT wrapper).
    if liger_enabled:
        try:
            # This liger_kernel version does not export `apply_liger_kernel_to_instance`.
            # Use the internal instance-level apply helper.
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance  # type: ignore
            _apply_liger_kernel_to_instance(model)
            logger.info("Liger kernels applied to base model")
        except Exception as e:
            logger.warning("Failed to apply Liger kernels to base model; continuing without it. (%s)", e)
    # We apply Liger manually above; ensure downstream Trainer logic does not treat it as enabled.
    # (Some Trainer/Accelerate integrations may read this flag.)
    if liger_requested:
        try:
            training_args.use_liger_kernel = False
        except Exception:
            setattr(training_args, "use_liger_kernel", False)
    try:
        setattr(model.config, "attn_impl", model_args.attn_impl)
    except Exception:
        logger.warning("Could not set attn_impl on base model config; continuing without it.")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # If model is already RCOT-wrapped, keep as is; otherwise wrap.
    if isinstance(model, RCOTWrapper):
        rcot_model = model
    else:
        rcot_model = RCOTWrapper.from_base_model(model, rcot_args)

    if data_args.max_length is not None and data_args.max_length <= 0:
        raise ValueError("`max_length` must be a positive integer when provided.")
    max_seq_length = data_args.max_length

    control_flow_all_recurrent = bool(getattr(data_args, "control_flow_all_recurrent", False))
    control_flow_split_answer = bool(getattr(data_args, "control_flow_split_answer", False))
    preprocess_train = RCOTCtrlFlowTokenizer().build_preprocess_fn(
        tokenizer,
        eval_args,
        prompt_only=False,
        control_flow_all_recurrent=control_flow_all_recurrent,
        label_mask_prompt=bool(getattr(data_args, "label_mask_prompt", False)),
        control_flow_split_answer=control_flow_split_answer,
    )
    preprocess_eval = RCOTCtrlFlowTokenizer().build_preprocess_fn(
        tokenizer,
        eval_args,
        prompt_only=True,
        control_flow_all_recurrent=control_flow_all_recurrent,
        label_mask_prompt=bool(getattr(data_args, "label_mask_prompt", False)),
        control_flow_split_answer=control_flow_split_answer,
    )
    truncate_fn = build_truncate_fn(max_seq_length)

    logger.info("Loading datasets")
    train_dataset = None
    eval_dataset = None

    if training_args.do_train:
        source = data_args.train_data_path or data_args.train_dataset_name
        if not source:
            raise ValueError("Train dataset not specified. Set --train_data_path (path or HF dataset name) or --train_dataset_name.")
        train_dataset = _load_dataset(
            source=source,
            split=getattr(data_args, "train_dataset_split", "train"),
            config=getattr(data_args, "train_dataset_config", None),
        )

    if training_args.do_eval:
        source = data_args.eval_data_path or data_args.eval_dataset_name or data_args.train_data_path or data_args.train_dataset_name
        if not source:
            raise ValueError("Eval dataset not specified. Set --eval_data_path/--eval_dataset_name (or reuse train args).")
        eval_dataset = _load_dataset(
            source=source,
            split=getattr(data_args, "eval_dataset_split", "validation"),
            config=getattr(data_args, "eval_dataset_config", None) or getattr(data_args, "train_dataset_config", None),
        )

    dataset_num_proc = training_args.dataset_num_proc or training_args.dataloader_num_workers

    if train_dataset is not None:
        train_cols = getattr(train_dataset, "column_names", [])
        train_preprocessed = "control_flow" in train_cols and "input_ids" in train_cols

        if not train_preprocessed:
            train_dataset = apply_custom_preprocessing(
                train_dataset,
                role="train",
                data_args=data_args,
                eval_args=eval_args,
            )
            train_dataset = train_dataset.map(
                preprocess_train,
                desc="Tokenizing train dataset",
                num_proc=dataset_num_proc,
            )
            train_dataset = apply_custom_postprocessing(
                train_dataset,
                role="train",
                data_args=data_args,
                eval_args=eval_args,
                tokenizer=tokenizer,
                model_args=model_args,
                num_proc=dataset_num_proc,
            )

        train_dataset = train_dataset.map(
            truncate_fn,
            desc="Applying train postprocessing",
            num_proc=dataset_num_proc,
        )
        visualize_sample_control_flow(tokenizer, train_dataset, idx=0, max_tokens=2048)

    if eval_dataset is not None:
        eval_cols = getattr(eval_dataset, "column_names", [])
        eval_preprocessed = "control_flow" in eval_cols and "input_ids" in eval_cols

        if not eval_preprocessed:
            eval_dataset = apply_custom_preprocessing(
                eval_dataset,
                role="eval",
                data_args=data_args,
                eval_args=eval_args,
            )
            eval_dataset = eval_dataset.map(
                preprocess_eval,
                desc="Tokenizing eval dataset",
                num_proc=dataset_num_proc,
            )
            eval_dataset = apply_custom_postprocessing(
                eval_dataset,
                role="eval",
                data_args=data_args,
                eval_args=eval_args,
                tokenizer=tokenizer,
                model_args=model_args,
                num_proc=dataset_num_proc,
            )

        eval_dataset = eval_dataset.map(
            truncate_fn,
            desc="Applying eval postprocessing",
            num_proc=dataset_num_proc,
        )

    # Add callback to ensure config is uploaded to wandb after training starts
    callbacks = []
    if _should_log_to_wandb(training_args):
        callbacks.append(
            WandbConfigUploadCallback(
                training_args=training_args,
                model_args=model_args,
                data_args=data_args,
                rcot_args=rcot_args,
                eval_args=eval_args,
            )
        )
    
    trainer = RCOTTrainer(
        model=rcot_model,
        training_args=training_args,
        rcot_args=rcot_args,
        data_args=data_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_data_collator=train_collator,
        eval_data_collator=eval_collator,
        callbacks=callbacks if callbacks else None,
    )

    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_train:
        logger.info("Starting training")
        if resume_checkpoint_path:
            logger.info("Resuming training from checkpoint: %s", resume_checkpoint_path)
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint_path)
        else:
            train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Ensure all processes wait for save to complete before evaluation
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Try loading back the model from the checkpoint
    logger.info("Testing loading back the model from the checkpoint")
    # rcot_model = RCOTWrapper.from_pretrained_with_rcot(training_args.output_dir)
    
    if training_args.do_eval and eval_dataset is not None:
        logger.info("Running generation evaluation")
        gen_metrics = run_generation_evaluation(
            trainer=trainer,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            rcot_args=rcot_args,
            eval_args=eval_args,
            training_args=training_args,
            model_args=model_args,
            control_flow_all_recurrent=control_flow_all_recurrent,
        )
        trainer.log_metrics("eval", gen_metrics)
        trainer.save_metrics("eval", gen_metrics)

    if _should_log_to_wandb(training_args):
        try:
            import wandb  # type: ignore
        except Exception as e:
            logger.warning("W&B not available; skipping wandb.finish(). (%s)", e)
        else:
            if wandb.run is not None:
                wandb.finish()

if __name__ == "__main__":
    main()
