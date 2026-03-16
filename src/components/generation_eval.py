import json
import os
import re
import shutil
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from datasets import Dataset
from math_verify import parse, verify  # type: ignore
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer, PreTrainedModel, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

# Optional imports for visualization
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    np = None
    plt = None

# Optional wandb import
try:
    import wandb as _wandb_module
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    _wandb_module = None

from t2mlr_wrapper.t2mlr_wrapper import T2MLRWrapper
from t2mlr_wrapper.model_io_utils import (
    load_t2mlr_config_with_fallback,
)
from components.all_arguments import GenerationEvalArguments, T2MLRArguments, TrainingArguments, ModelArguments
from t2mlr_wrapper.inference_wrapper import patch_model

import modeling  # noqa: F401

logger = logging.getLogger(__name__)


def _log_metrics_to_wandb(metrics: Dict[str, Any], prefix: str = "eval") -> bool:
    """Log metrics to wandb if it's already initialized.
    
    This function does NOT self-initialize wandb. It only logs if wandb.run
    is already active (initialized by the trainer or elsewhere).
    
    Args:
        metrics: Dictionary of metrics to log
        prefix: Prefix for metric keys (e.g., "eval", "generation_eval", "perplexity")
    
    Returns:
        True if metrics were logged successfully, False otherwise
    """
    if not HAS_WANDB or _wandb_module is None:
        logger.debug("wandb not available; skipping metric logging")
        return False
    
    run = _wandb_module.run
    if run is None:
        logger.debug("wandb run not initialized; skipping metric logging")
        return False
    
    try:
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        _wandb_module.log(prefixed_metrics)
        logger.info(f"Logged {len(metrics)} metrics to wandb with prefix '{prefix}'")
        return True
    except Exception as e:
        logger.warning(f"Failed to log metrics to wandb: {e}")
        return False


def _eval_output_dir(training_args: TrainingArguments) -> str:
    run_name = (training_args.run_name or "default").replace(os.sep, "_")
    if os.path.altsep:
        run_name = run_name.replace(os.path.altsep, "_")
    bench_name = (os.environ.get("TRAINER_EVAL_BENCH") or "").strip()
    base = training_args.output_dir.rstrip("/\\")
    if os.path.basename(base) == run_name:
        return os.path.join(base, "generation_eval")
    if bench_name:
        return os.path.join(base, run_name, bench_name)
    return os.path.join(base, run_name, "generation_eval")

def _read_config_json(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


def _get_base_config_from_dir(model_dir: str) -> Optional[Any]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    config = _read_config_json(config_path)
    return config.get("base_config")


def _resolve_model_path_for_load(model_root: str) -> str:
    """Resolve the model path for loading.
    
    Prefer run-root final weights when available (e.g., finished training runs).
    Fall back to latest checkpoint-* only when run root is not loadable yet.
    Raises RuntimeError if no valid model path can be found.
    """
    base = os.path.basename(model_root.rstrip("/"))
    config_path = os.path.join(model_root, "config.json")
    if os.path.isdir(model_root) and not base.startswith("checkpoint-"):
        if os.path.exists(config_path):
            return model_root
        latest_ckpt = get_last_checkpoint(model_root)
        if latest_ckpt:
            return latest_ckpt
    elif os.path.exists(config_path):
        return model_root
    latest_ckpt = get_last_checkpoint(model_root)
    if latest_ckpt:
        return latest_ckpt
    raise RuntimeError(
        f"Cannot find model to load: no config.json at {model_root} and no checkpoint subdirectory found. "
        "Ensure the model was saved before evaluation."
    )


def _load_inference_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> PreTrainedModel:
    """Load the saved model from disk for generation evaluation.

    Supports both:
    - T2MLR checkpoints (config.json is T2MLRConfig; loads T2MLRWrapper + base model)
    - Standard HF checkpoints (config.json is a base model config; loads AutoModelForCausalLM)
    """
    model_root = model_args.model_name_or_path
    if not os.path.exists(model_root):
        raise RuntimeError(
            f"Cannot load inference model: path does not exist: {model_root}. "
            "Ensure the model was saved to output_dir before evaluation."
        )
    logger.info("Loading inference model from %s", model_root)
    model_path = _resolve_model_path_for_load(model_root)
    ref_dtype = torch.bfloat16 if getattr(training_args, 'bf16', False) else torch.float32

    logger.info("Model path: %s", model_path)

    use_t2mlr_wrapper = True
    if os.path.isdir(model_path):
        base_config = _get_base_config_from_dir(model_path)
        if not base_config:
            use_t2mlr_wrapper = False

    if use_t2mlr_wrapper:
        t2mlr_kwargs = {"dtype": ref_dtype}
        if device.type == "cpu":
            t2mlr_kwargs["attn_impl"] = "sdpa"
        model = T2MLRWrapper.from_pretrained_with_t2mlr(model_path, **t2mlr_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=ref_dtype)

    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    return model


def _prepare_generation_kwargs(eval_args: GenerationEvalArguments) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if eval_args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = eval_args.max_new_tokens
    if eval_args.num_beams is not None:
        kwargs["num_beams"] = eval_args.num_beams
    inferred_sampling = False
    if eval_args.do_sample is None:
        # Align with common inference stacks (including vLLM): providing any sampling parameter
        # implies sampling unless explicitly disabled.
        if eval_args.top_p is not None or eval_args.top_k is not None:
            inferred_sampling = True
        elif eval_args.temperature is not None and float(eval_args.temperature) > 0:
            inferred_sampling = True

    if eval_args.do_sample is True or inferred_sampling:
        kwargs["do_sample"] = True
        if eval_args.top_p is not None:
            kwargs["top_p"] = eval_args.top_p
        if eval_args.top_k is not None:
            kwargs["top_k"] = eval_args.top_k
        if eval_args.temperature is not None:
            kwargs["temperature"] = eval_args.temperature
    elif eval_args.do_sample is False:
        kwargs["do_sample"] = False
    return kwargs




def _get_prompt(example: Dict[str, Any], prompt_column: str) -> str:
    prompt = example.get(prompt_column)
    if prompt is None:
        raise KeyError(f"Prompt column '{prompt_column}' missing in evaluation example.")
    return str(prompt)


def _get_reference(example: Dict[str, Any], response_column: str) -> str:
    reference = example.get(response_column)
    if reference is None:
        raise KeyError(f"Response column '{response_column}' missing in evaluation example.")
    return str(reference)


_PROSQA_STATEMENT_RE = re.compile(r"^(?:Every\s+)?(?P<lhs>.+?)\s+is\s+an?\s+(?P<rhs>.+?)\.\s*$", re.IGNORECASE)
_PROSQA_PREFIX_CLEAN_RE = re.compile(r"^\d+[\).:-]?\s*")
_PROSQA_LEAD_WORD_RE = re.compile(r"^(?:Therefore|Thus|Hence|So)[,\s]+", re.IGNORECASE)
_MATH_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|{[^{}]*})*)\}")


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Return the last '\\boxed{...}' payload if present."""
    matches = _MATH_BOXED_RE.findall(text)
    if not matches:
        return None
    candidate = str(matches[-1]).strip()
    return candidate or None


def _normalize_math_candidate(text: str) -> str:
    """Best-effort normalization for math equivalence checks."""
    cleaned = text.strip()
    boxed = _extract_boxed_answer(cleaned)
    if boxed is not None:
        cleaned = boxed
    # Trim trailing punctuation/newlines that commonly follow boxed answers.
    return cleaned.strip().strip(".").strip()


def _parse_math_verify(expr: str):
    """Parse an expression for math_verify, trying a couple of common wrappers."""
    last_err: Optional[Exception] = None
    for candidate in (expr, f"${expr}$"):
        try:
            return parse(candidate)
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err


def _is_equivalent_math_vllm_style(ans1: Optional[str], ans2: Optional[str]) -> Optional[bool]:
    """Match general_inference_eval's math equivalence logic.

    Returns:
      - True/False when both answers are present (parsable or not)
      - None when the model answer is missing (unparsable in vLLM terminology)
    """
    if ans1 is None:
        return None
    if ans2 is None:
        return None

    a1 = str(ans1).strip()
    a2 = str(ans2).strip()
    if a1 == a2:
        return True
    # Mirror general_inference_eval guardrail: avoid heavy parsing on long strings.
    if len(a1) > 100 or len(a2) > 100:
        return a1 == a2

    try:
        answer = parse(f"${a1}$")
        expected = parse(f"${a2}$")
        result = verify(expected, answer)
        return bool(result[0]) if isinstance(result, tuple) else bool(result)
    except Exception:
        return a1 == a2


def _compute_math_correctness_vllm_style(generated: str, reference: str) -> Optional[bool]:
    """Compute correctness for math mode to match vLLM evaluation semantics.

    - Prediction: must contain a boxed answer (last \\boxed{...}). If missing -> None (unparsable).
    - Gold: extract answer after ####/### if present, else last boxed, else stripped reference.
    - Equivalence: compare via math_verify with `$...$` wrapping (see _is_equivalent_math_vllm_style).
    """
    pred_boxed = _extract_boxed_answer(generated)
    if pred_boxed is None:
        return None
    gold_candidate = _extract_gsm8k_answer(reference) or _extract_boxed_answer(reference) or reference.strip()
    gold_candidate = gold_candidate.strip() if gold_candidate is not None else None
    return _is_equivalent_math_vllm_style(pred_boxed, gold_candidate)


def _normalize_prosqa_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    stripped = stripped.lstrip("#").strip()
    stripped = stripped.lstrip("-*•").strip()
    stripped = _PROSQA_PREFIX_CLEAN_RE.sub("", stripped)
    stripped = _PROSQA_LEAD_WORD_RE.sub("", stripped)
    # Remove special tokens that might appear after the period (e.g., <|endoftext|>, <|eot_id|>, etc.)
    # Split on common special tokens and take the first part
    for special_token in ["<|eot_id|>", "<|endoftext|>", "<|end|>"]:
        if special_token in stripped:
            stripped = stripped.split(special_token)[0].strip()
    return stripped


def _parse_prosqa_statement(line: str) -> Optional[Tuple[str, str]]:
    candidate = _normalize_prosqa_line(line)
    if not candidate:
        return None
    match = _PROSQA_STATEMENT_RE.match(candidate)
    if not match:
        return None
    lhs = match.group("lhs").strip()
    rhs = match.group("rhs").strip()
    return lhs, rhs


def _extract_prosqa_statements_and_answer(text: str) -> Tuple[List[Tuple[str, str]], Optional[Tuple[str, str]]]:
    statements: List[Tuple[str, str]] = []
    final_answer: Optional[Tuple[str, str]] = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        is_answer_line = stripped.lstrip().startswith("#")
        parsed = _parse_prosqa_statement(stripped)
        if parsed is None:
            continue
        if is_answer_line:
            final_answer = parsed
        else:
            statements.append(parsed)
    return statements, final_answer


def _prosqa_path_reward(generated: str, example: Optional[Dict[str, Any]]) -> bool:
    if example is None:
        return False

    idx_to_symbol = example.get("idx_to_symbol")
    edges = example.get("edges")
    answer = example.get("answer")

    if not isinstance(idx_to_symbol, list) or not isinstance(edges, list) or not isinstance(answer, str):
        return False

    parsed_answer = _parse_prosqa_statement(answer)
    if parsed_answer is None:
        return False
    gold_entity, gold_concept = parsed_answer

    symbol_to_idx = {str(symbol): idx for idx, symbol in enumerate(idx_to_symbol)}
    if gold_entity not in symbol_to_idx or gold_concept not in symbol_to_idx:
        return False

    edge_set: Set[Tuple[int, int]] = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        try:
            src = int(edge[0])
            dst = int(edge[1])
        except (TypeError, ValueError):
            continue
        edge_set.add((src, dst))

    if not edge_set:
        return False

    statements, explicit_answer = _extract_prosqa_statements_and_answer(generated)
    if not statements:
        return False

    inferred_answer = explicit_answer
    if inferred_answer is None:
        for stmt in reversed(statements):
            if stmt[0] == gold_entity:
                inferred_answer = stmt
                break

    if inferred_answer is None:
        return False

    pred_entity, pred_concept = inferred_answer
    if pred_entity != gold_entity or pred_concept != gold_concept:
        return False

    current_symbol = gold_entity
    for lhs, rhs in statements:
        if lhs not in symbol_to_idx or rhs not in symbol_to_idx:
            return False
        if lhs != current_symbol:
            return False
        src_idx = symbol_to_idx[lhs]
        dst_idx = symbol_to_idx[rhs]
        if (src_idx, dst_idx) not in edge_set:
            return False
        current_symbol = rhs

    if current_symbol != gold_concept:
        return False

    return True


def _pathfinding_reward(generated: str, example: Optional[Dict[str, Any]]) -> bool:
    """Validate pathfinding answer: check if the generated path is the shortest valid path.
    
    Expected format: "A -> B -> C -> D" where each transition is a valid edge.
    Also handles random path prefix format: "random_path | shortest_path"
    where only the shortest_path part (after the divider) is evaluated.
    """
    if example is None:
        return False

    idx_to_symbol = example.get("idx_to_symbol")
    edges = example.get("edges")
    start = example.get("start")
    end = example.get("end")
    expected_path_length = example.get("path_length")

    if not isinstance(idx_to_symbol, list) or not isinstance(edges, list):
        return False
    if not isinstance(start, int) or not isinstance(end, int):
        return False

    symbol_to_idx = {str(symbol): idx for idx, symbol in enumerate(idx_to_symbol)}

    # Build edge set
    edge_set: Set[Tuple[int, int]] = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        try:
            src = int(edge[0])
            dst = int(edge[1])
        except (TypeError, ValueError):
            continue
        edge_set.add((src, dst))

    if not edge_set:
        return False

    # Parse generated path: "A -> B -> C -> D"
    generated = generated.strip()
    
    # Handle random path prefix format: "random_path | shortest_path"
    # Extract only the shortest path part (after the last divider)
    if " | " in generated:
        generated = generated.split(" | ")[-1].strip()
    elif "|" in generated:
        generated = generated.split("|")[-1].strip()
    
    # Handle various separators
    if " -> " in generated:
        path_parts = [p.strip() for p in generated.split(" -> ")]
    elif "->" in generated:
        path_parts = [p.strip() for p in generated.split("->")]
    elif " - " in generated:
        path_parts = [p.strip() for p in generated.split(" - ")]
    elif "," in generated:
        path_parts = [p.strip() for p in generated.split(",")]
    else:
        path_parts = generated.split()

    if len(path_parts) < 2:
        return False

    # Convert to indices
    path_indices = []
    for part in path_parts:
        # Clean up the part
        part = part.strip().rstrip(".,;")
        if part not in symbol_to_idx:
            return False
        path_indices.append(symbol_to_idx[part])

    # Check start and end
    if path_indices[0] != start or path_indices[-1] != end:
        return False

    # Check all edges exist
    for i in range(len(path_indices) - 1):
        src, dst = path_indices[i], path_indices[i + 1]
        if (src, dst) not in edge_set:
            return False

    # Check path length matches expected shortest path
    actual_length = len(path_indices) - 1
    if expected_path_length is not None and actual_length != expected_path_length:
        return False

    return True


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    markers = ("####", "###")
    for marker in markers:
        if marker in text:
            candidate = text.rsplit(marker, 1)[-1].strip()
            if candidate:
                # Remove special tokens and padding characters that might be appended
                # Split on common special tokens and take the first part
                for special_token in ["<|eot_id|>", "<|endoftext|>", "<|end|>"]:
                    if special_token in candidate:
                        candidate = candidate.split(special_token)[0].strip()
                # Remove padding characters (common padding tokens decode to repeated characters)
                candidate = candidate.rstrip("! ").strip()
                if candidate:
                    return candidate
    # Fall back to last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        # Clean special tokens from last line too
        for special_token in ["<|eot_id|>", "<|endoftext|>", "<|end|>"]:
            if special_token in last_line:
                last_line = last_line.split(special_token)[0].strip()
        last_line = last_line.rstrip("! ").strip()
        if last_line:
            return last_line
    return None


def _normalize_gsm8k_answer(answer: str) -> str:
    cleaned = answer.strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned.replace(",", "").strip()


def _parse_gate_array(gate_list: List) -> Optional[Any]:
    """Parse nested list structure into numpy array.
    
    Handles both 2D (T, H) and 3D (B, T, H) structures.
    Returns None if numpy is not available.
    """
    if not HAS_VISUALIZATION or not gate_list:
        return None
    
    # Convert to numpy array
    arr = np.array(gate_list, dtype=np.float32)
    
    # If 2D, add batch dimension
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]  # (1, T, H)
    
    return arr


def _visualize_gate_trace_from_payload(gate_trace_payload: Dict[str, Any], output_path: str) -> bool:
    """Create visualization of gate trace from payload dictionary.
    
    Returns True if visualization was created successfully, False otherwise.
    """
    if not HAS_VISUALIZATION:
        logger.warning("Visualization libraries (numpy, matplotlib) not available. Skipping gate trace visualization.")
        return False
    
    try:
        mixing_logs = gate_trace_payload.get("mixing_module_logs", {})
        recurrent_gate_list = mixing_logs.get("recurrent_gate")
        input_gate_list = mixing_logs.get("input_gate")
        
        if recurrent_gate_list is None:
            logger.warning("No recurrent_gate found in gate trace payload")
            return False
        
        # Parse gate arrays
        recurrent_gate = _parse_gate_array(recurrent_gate_list)
        input_gate = _parse_gate_array(input_gate_list) if input_gate_list is not None else None
        
        if recurrent_gate is None:
            return False
        
        # Use first batch if multiple batches
        recurrent_gate = recurrent_gate[0]  # (T, H)
        if input_gate is not None:
            input_gate = input_gate[0]  # (T, H)
        
        T, H = recurrent_gate.shape
        
        # Extract metadata
        run_name = gate_trace_payload.get("run_name", "unknown")
        example_index = gate_trace_payload.get("example_index", -1)
        
        # Create visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. Recurrent gate mean over time
        ax1 = fig.add_subplot(gs[0, 0])
        recurrent_mean = np.mean(recurrent_gate, axis=1)  # (T,)
        ax1.plot(recurrent_mean, linewidth=1.5, color='blue', label='Recurrent gate (mean)')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Gate value')
        ax1.set_title('Recurrent Gate (Mean over Hidden Dims)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1.1])
        
        # 2. Input gate mean over time
        ax2 = fig.add_subplot(gs[0, 1])
        if input_gate is not None:
            input_mean = np.mean(input_gate, axis=1)  # (T,)
            ax2.plot(input_mean, linewidth=1.5, color='red', label='Input gate (mean)')
            ax2.set_xlabel('Time step')
            ax2.set_ylabel('Gate value')
            ax2.set_title('Input Gate (Mean over Hidden Dims)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim([0, 1.1])
        else:
            ax2.text(0.5, 0.5, 'No input gate data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Input Gate (Not Available)')
        
        # 3. Recurrent gate heatmap (sample of hidden dims)
        ax3 = fig.add_subplot(gs[1, :])
        num_dims_to_show = min(50, H)
        dim_indices = np.linspace(0, H-1, num_dims_to_show, dtype=int)
        recurrent_sample = recurrent_gate[:, dim_indices].T  # (num_dims, T)
        im3 = ax3.imshow(recurrent_sample, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel(f'Sampled Hidden Dims (showing {num_dims_to_show} of {H})')
        ax3.set_title('Recurrent Gate Heatmap (Sampled Hidden Dimensions)')
        plt.colorbar(im3, ax=ax3, label='Gate value')

        # 4. Input gate heatmap (sample of hidden dims)
        ax4 = fig.add_subplot(gs[2, :])
        if input_gate is not None:
            input_sample = input_gate[:, dim_indices].T  # (num_dims, T)
            im4 = ax4.imshow(input_sample, aspect='auto', cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
            ax4.set_xlabel('Time step')
            ax4.set_ylabel(f'Sampled Hidden Dims (showing {num_dims_to_show} of {H})')
            ax4.set_title('Input Gate Heatmap (Sampled Hidden Dimensions)')
            plt.colorbar(im4, ax=ax4, label='Gate value')
        else:
            ax4.text(0.5, 0.5, 'No input gate data', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Input Gate Heatmap (Not Available)')

        # 5. Gate statistics over time (span full width)
        ax5 = fig.add_subplot(gs[3, :])
        recurrent_p25 = np.percentile(recurrent_gate, 25, axis=1)
        recurrent_p50 = np.percentile(recurrent_gate, 50, axis=1)
        recurrent_p75 = np.percentile(recurrent_gate, 75, axis=1)
        ax5.fill_between(range(T), recurrent_p25, recurrent_p75, alpha=0.3, color='blue', label='Recurrent (25-75th percentile)')
        ax5.plot(recurrent_p50, linewidth=2, color='blue', label='Recurrent (median)')
        if input_gate is not None:
            input_p25 = np.percentile(input_gate, 25, axis=1)
            input_p50 = np.percentile(input_gate, 50, axis=1)
            input_p75 = np.percentile(input_gate, 75, axis=1)
            ax5.fill_between(range(T), input_p25, input_p75, alpha=0.3, color='red', label='Input (25-75th percentile)')
            ax5.plot(input_p50, linewidth=2, color='red', label='Input (median)')
        ax5.set_xlabel('Time step')
        ax5.set_ylabel('Gate value')
        ax5.set_title('Gate Statistics Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1.1])
        
        # Add title with metadata
        fig.suptitle(
            f'Gate Trace Visualization\n'
            f'Run: {run_name} | Example: {example_index} | T={T} | H={H}',
            fontsize=12, y=0.995
        )
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        
        logger.info(f"Gate trace visualization saved to {output_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to create gate trace visualization: {e}", exc_info=True)
        return False


def _compute_reward(generated: str, reference: str, mode: str, example: Optional[Dict[str, Any]] = None) -> float:
    generated = generated.strip()
    reference = reference.strip()
    if mode == "math":
        try:
            # Keep _compute_reward permissive for legacy callers; the main eval loop uses the
            # vLLM-style path (boxed-only prediction + `$...$` parsing).
            gold_expr = _parse_math_verify(_normalize_math_candidate(reference))
            pred_expr = _parse_math_verify(_normalize_math_candidate(generated))
            result = verify(gold_expr, pred_expr)
            is_correct = bool(result[0]) if isinstance(result, tuple) else bool(result)
        except Exception:
            is_correct = False
    elif mode == "gsm8k":
        gold_answer = _extract_gsm8k_answer(reference)
        pred_answer = _extract_gsm8k_answer(generated)
        if gold_answer is None or pred_answer is None:
            is_correct = generated == reference
        else:
            is_correct = (
                _normalize_gsm8k_answer(pred_answer)
                == _normalize_gsm8k_answer(gold_answer)
            )
    elif mode == "prosqa_path":
        is_correct = _prosqa_path_reward(generated, example)
    elif mode == "pathfinding":
        is_correct = _pathfinding_reward(generated, example)
    else:
        is_correct = generated == reference
    return 1.0 if is_correct else 0.0


def build_rl_reward_function(eval_args: GenerationEvalArguments) -> Callable[..., List[float]]:
    """Create a GRPO-compatible reward function using the canonical evaluation logic."""

    response_key = eval_args.get_eval_response_column()
    if not response_key:
        raise ValueError(
            "Eval arguments must provide a response column when building the RL reward function."
        )

    def _reward_func(*, prompts, completions, completion_ids, trainer_state=None, **kwargs) -> List[float]:
        references = kwargs.get(response_key)
        if not isinstance(references, list):
            raise ValueError(
                f"RL reward function expected the dataset to supply a list-valued '{response_key}' column."
            )

        total = len(completions)
        if len(references) < total:
            raise ValueError(
                f"Reference column '{response_key}' contains fewer entries than completions ("
                f"{len(references)} vs {total})."
            )

        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            reference_obj = references[idx]
            reference_text = "" if reference_obj is None else str(reference_obj)

            example_payload: Dict[str, Any] = {}
            for key, value in kwargs.items():
                if key == "trainer_state":
                    continue
                if isinstance(value, list):
                    if idx < len(value):
                        example_payload[key] = value[idx]
                else:
                    example_payload[key] = value

            reward_value = _compute_reward(
                completion[-1]['content'],
                reference_text,
                eval_args.reward_mode,
                example_payload,
            )
            rewards.append(float(reward_value))

        return rewards

    return _reward_func


def run_generation_evaluation(
    trainer,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    t2mlr_args: T2MLRArguments,
    eval_args: GenerationEvalArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    control_flow_all_recurrent: bool = False,
) -> Dict[str, Any]:
    device = training_args.device

    # Always load inference model from disk.
    # Never use the in-memory trainer model to avoid evaluating untrained models.
    inference_model = None
    # Determine the model path to load from.
    # - If training happened in this run, evaluate from output_dir (latest model produced this run).
    # - If no training happened, evaluate from model_name_or_path.
    if training_args.do_train:
        load_path = training_args.output_dir
        logger.info("Loading inference model from output_dir after training: %s", load_path)
    else:
        load_path = model_args.model_name_or_path
        logger.info("Loading inference model from model_name_or_path: %s", load_path)

    if not os.path.exists(load_path):
        raise RuntimeError(
            f"Cannot load inference model: path does not exist: {load_path}. "
            "Ensure the model was saved before evaluation."
        )

    old_model_path = model_args.model_name_or_path
    try:
        model_args.model_name_or_path = load_path
        inference_model = _load_inference_model(model_args, training_args, tokenizer, device)
    finally:
        model_args.model_name_or_path = old_model_path
    if isinstance(inference_model, T2MLRWrapper):
        inference_model.control_flow_all_recurrent = bool(control_flow_all_recurrent)
        logger.info(
            "Generation eval: T2MLR prompt recurrent=%s (control_flow_all_recurrent=%s)",
            inference_model.control_flow_all_recurrent,
            control_flow_all_recurrent,
        )
    else:
        logger.info(
            "Generation eval: T2MLR prompt recurrent unavailable (non-T2MLR model); control_flow_all_recurrent=%s",
            control_flow_all_recurrent,
        )
    generation_kwargs = _prepare_generation_kwargs(eval_args)
    # Transformers warns about "right padding" for decoder-only models when the last token equals `pad_token_id`.
    # Many decoder-only tokenizers set `pad_token_id == eos_token_id` (common for LLaMA-family), and our prompts
    # may legitimately end with EOS, which trips the warning even when attention_mask/padding are otherwise fine.
    # Use a "safe" pad token id for generation to avoid false positives; with attention_mask provided, this is
    # only used as a padding sentinel.
    safe_pad_token_id = tokenizer.pad_token_id
    if safe_pad_token_id is None or safe_pad_token_id == tokenizer.eos_token_id:
        safe_pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    raw_pass_values = getattr(eval_args, "pass_at_k", []) or []
    normalized_pass_values: List[int] = []
    for value in raw_pass_values:
        int_value = int(value)
        if int_value > 0:
            normalized_pass_values.append(int_value)
    pass_at_k_values = sorted(set(normalized_pass_values))
    if not pass_at_k_values:
        pass_at_k_values = [1]

    if eval_args.num_generations_per_sample is not None:
        num_generations = eval_args.num_generations_per_sample
    else:
        num_generations = max(pass_at_k_values)

    if num_generations is None or num_generations <= 0:
        raise ValueError("num_generations_per_sample must be a positive integer.")

    max_requested_k = max(pass_at_k_values) if pass_at_k_values else 1
    if num_generations < max_requested_k:
        raise ValueError(
            "num_generations_per_sample must be at least as large as the maximum pass@k value."
        )

    if trainer.is_world_process_zero():
        try:
            base_gen_cfg: Optional[GenerationConfig] = getattr(inference_model, "generation_config", None)
            if base_gen_cfg is None:
                try:
                    base_gen_cfg = GenerationConfig.from_model_config(getattr(inference_model, "config"))
                except Exception:
                    base_gen_cfg = None
            base_dict = base_gen_cfg.to_dict() if base_gen_cfg is not None else {}
            payload = json.dumps(
                {
                    "transformers_version": getattr(transformers, "__version__", None),
                    "generation_config": base_dict,
                },
                sort_keys=True,
            )
            # Use `print()` so it always lands in Slurm stdout even if logging isn't configured.
            print(f"[generation_eval] HF GenerationConfig (base): {payload}", flush=True)
            print(
                f"[generation_eval] HF generation eval settings: pass_at_k={pass_at_k_values} "
                f"resolved_num_generations={num_generations} generation_kwargs={generation_kwargs}",
                flush=True,
            )
        except Exception:
            print("[generation_eval] HF GenerationConfig (base): (failed to serialize)", flush=True)

    # Persist per-example records when requested. `save_all_generations` controls *how much*
    # we save (all generations vs only first), while `save_eval_dataset` controls *whether*
    # we write out an eval JSONL at all.
    save_records = bool(getattr(eval_args, "save_eval_dataset", False)) or bool(
        getattr(eval_args, "save_all_generations", False)
    )
    records: List[Dict[str, Any]] = []
    first_rewards: List[float] = []
    all_rewards: List[float] = []
    pass_counts = {k: 0 for k in pass_at_k_values}
    total_generated_tokens = 0
    num_examples_processed = 0
    first_unparsable_count = 0
    first_correct_count = 0
    first_incorrect_count = 0
    first_token_counts: List[int] = []
    dynamic_max_count = 0
    dynamic_max_sum = 0
    dynamic_max_min: Optional[int] = None
    dynamic_max_max: Optional[int] = None
    logged_first_effective_generation_config = False

    from torch.utils.data import DataLoader
    world_size = training_args.world_size
    process_index = training_args.process_index
    is_distributed = world_size > 1

    if trainer.is_world_process_zero():
        logger.info(f"Distributed evaluation: {'ENABLED' if is_distributed else 'DISABLED'} (world_size={world_size})")

    if is_distributed:
        # Shard the dataset manually to let each process handle a subset.
        # We use the default shard which is interleaved (contiguous=False).
        local_eval_dataset = eval_dataset.shard(num_shards=world_size, index=process_index)
        logger.info(f"[Rank {process_index}] Sharded eval dataset: {len(eval_dataset)} total -> {len(local_eval_dataset)} local samples.")
    else:
        local_eval_dataset = eval_dataset

    # Create local dataloader. Avoid trainer.get_eval_dataloader to prevent double sharding.
    eval_dataloader = DataLoader(
        local_eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=trainer.eval_data_collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        shuffle=False,
    )

    gate_trace_payload: Optional[Dict[str, Any]] = None
    target_gate_example = max(0, int(eval_args.gate_trace_example_index or 0))

    with torch.no_grad():
        example_offset = 0
        for batch in tqdm(eval_dataloader, disable=not trainer.is_world_process_zero() and not training_args.disable_tqdm, desc="Generation eval"):
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            if "input_ids" not in tensor_batch:
                raise KeyError("Evaluation dataloader batch is missing 'input_ids'.")

            batch_input_ids = tensor_batch["input_ids"]
            batch_attention_mask = tensor_batch.get("attention_mask")

            batch_size = batch_input_ids.size(0)

            example_indices: List[int] = []
            records_batch: List[Dict[str, Any]] = []
            examples_batch: List[Dict[str, Any]] = []
            prompt_texts: List[str] = []
            reference_texts: List[str] = []
            dynamic_targets: List[int] = []
            target_in_batch_idx: Optional[int] = None

            for batch_idx in range(batch_size):
                example_idx = example_offset + batch_idx
                if example_idx >= len(local_eval_dataset):
                    break

                example = local_eval_dataset[example_idx]
                # Global index for consistent tracing/logging across shards
                global_idx = example_idx * world_size + process_index if is_distributed else example_idx
                
                record = {k: example[k] for k in example.keys()} if save_records else None
                if record is not None and is_distributed:
                    record["__global_index__"] = global_idx

                prompt_text = _get_prompt(example, eval_args.get_eval_prompt_column())
                reference_text = _get_reference(example, eval_args.get_eval_response_column())

                if save_records:
                    records_batch.append(record)
                examples_batch.append(example)
                prompt_texts.append(prompt_text)
                reference_texts.append(reference_text)
                example_indices.append(global_idx)

                if (
                    eval_args.capture_gate_trace
                    and gate_trace_payload is None
                    and target_in_batch_idx is None
                    and global_idx == target_gate_example
                ):
                    target_in_batch_idx = len(example_indices) - 1

                if "max_new_tokens" not in generation_kwargs:
                    reference_tokens = tokenizer(reference_text, add_special_tokens=False)["input_ids"]
                    target_len = len(reference_tokens)
                    buffered = target_len + eval_args.target_length_buffer
                    dynamic_targets.append(max(buffered, eval_args.default_new_tokens))

            valid_batch_size = len(example_indices)
            if valid_batch_size == 0:
                break

            if valid_batch_size < batch_size:
                batch_input_ids = batch_input_ids[:valid_batch_size]
                if isinstance(batch_attention_mask, torch.Tensor):
                    batch_attention_mask = batch_attention_mask[:valid_batch_size]

            gen_kwargs = dict(generation_kwargs)
            if "max_new_tokens" not in gen_kwargs:
                computed_max = max(dynamic_targets) if dynamic_targets else eval_args.default_new_tokens
                gen_kwargs["max_new_tokens"] = computed_max
                dynamic_max_count += 1
                dynamic_max_sum += int(computed_max)
                dynamic_max_min = int(computed_max) if dynamic_max_min is None else min(dynamic_max_min, int(computed_max))
                dynamic_max_max = int(computed_max) if dynamic_max_max is None else max(dynamic_max_max, int(computed_max))

            if trainer.is_world_process_zero() and not logged_first_effective_generation_config:
                try:
                    base_gen_cfg: Optional[GenerationConfig] = getattr(inference_model, "generation_config", None)
                    if base_gen_cfg is None:
                        try:
                            base_gen_cfg = GenerationConfig.from_model_config(getattr(inference_model, "config"))
                        except Exception:
                            base_gen_cfg = None
                    cfg_dict = base_gen_cfg.to_dict() if base_gen_cfg is not None else {}
                    cfg_dict.update(gen_kwargs)
                    print(
                        "[generation_eval] HF GenerationConfig (effective for first batch): "
                        + json.dumps(cfg_dict, sort_keys=True),
                        flush=True,
                    )
                except Exception:
                    print(
                        "[generation_eval] HF GenerationConfig (effective for first batch): (failed to serialize)",
                        flush=True,
                    )
                logged_first_effective_generation_config = True

            per_example_generations: List[List[Dict[str, Any]]] = [[] for _ in range(valid_batch_size)]
            per_example_rewards: List[List[float]] = [[] for _ in range(valid_batch_size)]

            for generation_idx in range(num_generations):
                capture_gate_this_call = (
                    eval_args.capture_gate_trace
                    and gate_trace_payload is None
                    and target_in_batch_idx is not None
                    and generation_idx == 0
                )
                # T2MLRWrapper supports extra kwargs (tokenizer, record_gating_stats).
                if isinstance(inference_model, T2MLRWrapper):
                    outputs = inference_model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        tokenizer=tokenizer,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=safe_pad_token_id,
                        use_cache=True,
                        recurrence_in_prompt=bool(control_flow_all_recurrent),
                        record_gating_stats=capture_gate_this_call,
                        **gen_kwargs,
                    )
                else:
                    outputs = inference_model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=safe_pad_token_id,
                        use_cache=True,
                        **gen_kwargs,
                    )

                for i in range(valid_batch_size):
                    output_ids = outputs[i]
                    
                    # Handle left-padding: find where the actual prompt starts
                    if isinstance(batch_attention_mask, torch.Tensor):
                        # Find the first non-padding token position
                        attention_mask_i = batch_attention_mask[i]
                        nonzero_indices = torch.nonzero(attention_mask_i, as_tuple=True)[0]
                        if len(nonzero_indices) > 0:
                            prompt_start_idx = int(nonzero_indices[0].item())
                            prompt_end_idx = int(nonzero_indices[-1].item()) + 1
                        else:
                            # All padding (shouldn't happen in practice)
                            prompt_start_idx = 0
                            prompt_end_idx = len(batch_input_ids[i])
                    else:
                        # No attention mask, assume no padding
                        prompt_start_idx = 0
                        prompt_end_idx = len(batch_input_ids[i])
                    
                    # Extract generated tokens (everything after the prompt)
                    generated_ids = output_ids[prompt_end_idx:]
                    generated_tokens_with_pause = generated_ids.detach().cpu().tolist()

                    # Truncate at EOS token - model.generate() pads sequences to same length in batch
                    # so we need to stop at EOS to avoid including padding tokens
                    eos_token_id = tokenizer.eos_token_id
                    if eos_token_id is not None and eos_token_id in generated_tokens_with_pause:
                        eos_idx = generated_tokens_with_pause.index(eos_token_id)
                        generated_tokens_with_pause = generated_tokens_with_pause[:eos_idx + 1]
                    
                    # Filter out pause tokens and padding tokens (but keep EOS if present)
                    pause_token = "<|reserved_special_token_0|>"
                    pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
                    pad_token_id = tokenizer.pad_token_id
                    generated_tokens = []
                    for tid in generated_tokens_with_pause:
                        # Keep EOS token, filter out pause and padding tokens
                        if tid == eos_token_id:
                            generated_tokens.append(tid)
                        elif tid != pause_token_id and tid != pad_token_id and tid != 0:
                            generated_tokens.append(tid)
                        # Stop if we hit padding after EOS (shouldn't happen after EOS truncation, but safe)
                        elif tid == 0 or tid == pad_token_id:
                            break
                    
                    # Decode tokens, skipping special tokens during detokenization
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                    reference_text = reference_texts[i]
                    unparsable = False
                    if eval_args.reward_mode == "math":
                        correctness = _compute_math_correctness_vllm_style(generated_text, reference_text)
                        if correctness is None:
                            unparsable = True
                            reward_value = 0.0
                        else:
                            reward_value = 1.0 if correctness else 0.0
                    else:
                        reward_value = float(
                            _compute_reward(
                                generated_text,
                                reference_text,
                                eval_args.reward_mode,
                                examples_batch[i] if i < len(examples_batch) else None,
                            )
                        )

                    per_example_generations[i].append(
                        {
                            "rank": generation_idx,
                            "text": generated_text,
                            "token_ids": generated_tokens,
                            "reward": reward_value,
                            "unparsable": unparsable,
                        }
                    )
                    per_example_rewards[i].append(reward_value)
                    all_rewards.append(reward_value)
                    total_generated_tokens += len(generated_tokens)
                    if generation_idx == 0:
                        first_rewards.append(reward_value)
                        first_token_counts.append(len(generated_tokens))
                        if unparsable:
                            first_unparsable_count += 1
                        elif reward_value >= 1.0:
                            first_correct_count += 1
                        else:
                            first_incorrect_count += 1

                    if capture_gate_this_call and i == target_in_batch_idx:
                        # Extract the actual prompt tokens (excluding left padding)
                        prompt_token_ids = output_ids[prompt_start_idx:prompt_end_idx].detach().cpu().tolist()

                        mixing_logs = getattr(outputs, "mixing_module_logs", None)
                        if not isinstance(mixing_logs, dict):
                            mixing_logs = {}

                        # Prefer gated-mixer logs when available.
                        recurrent_gate = mixing_logs.get("recurrent_gate", None)
                        input_gate = mixing_logs.get("input_gate", None)

                        # Standardize to python lists (JSON-serializable)
                        recurrent_gate_list = (
                            recurrent_gate.tolist() if hasattr(recurrent_gate, "tolist") else None
                        )
                        input_gate_list = (
                            input_gate.tolist() if hasattr(input_gate, "tolist") else None
                        )

                        gate_trace_payload = {
                            "run_name": training_args.run_name,
                            "example_index": int(example_indices[i]),
                            "batch_index": int(i),
                            "prompt": {
                                "text": prompt_texts[i],
                                "token_ids": prompt_token_ids,
                            },
                            "reference": reference_texts[i],
                            "generation": {
                                "rank": generation_idx,
                                "text": generated_text,
                                "token_ids": generated_tokens,
                            },
                            "mixing_module_logs": {
                                "recurrent_gate": recurrent_gate_list,
                                "input_gate": input_gate_list,
                            },
                            "notes": (
                                "Gate trace captured via record_gating_stats/mixing_module_logs. "
                                "Values are masked by control_flow>0 during logging and concatenated across steps."
                            ),
                        }

            for i in range(valid_batch_size):
                record = records_batch[i] if save_records else None
                generation_list = per_example_generations[i]
                reward_list = per_example_rewards[i]

                pass_flags: Dict[str, bool] = {}
                for k in pass_at_k_values:
                    effective_k = min(k, len(reward_list))
                    is_success = effective_k > 0 and any(reward_list[:effective_k])
                    pass_flags[f"pass@{k}"] = is_success
                    if is_success:
                        pass_counts[k] += 1

                if generation_list:
                    first_gen = generation_list[0]
                    if save_records:
                        record["model_output_tokens"] = first_gen["token_ids"]
                        record["model_output"] = first_gen["text"]
                        record["reward"] = first_gen["reward"]
                else:
                    if save_records:
                        record["model_output_tokens"] = []
                        record["model_output"] = ""
                        record["reward"] = 0.0

                if save_records:
                    if eval_args.save_all_generations:
                        record["all_model_outputs"] = generation_list
                    record["all_rewards"] = reward_list
                    record["pass_at_k"] = pass_flags
                    record["num_generations"] = len(generation_list)
                    records.append(record)
            example_offset += valid_batch_size
            num_examples_processed += valid_batch_size
    
    if is_distributed:
        if trainer.is_world_process_zero():
            logger.info(f"Aggregating results from {world_size} processes. Waiting for all processes to finish...")
        # Aggregate metrics across all processes using torch.distributed
        k_values = sorted(pass_counts.keys())
        pass_counts_tensor = torch.tensor([pass_counts[k] for k in k_values], device=device, dtype=torch.long)
        
        metrics_tensor = torch.tensor([
            num_examples_processed,
            sum(first_rewards),
            sum(all_rewards),
            len(all_rewards),
            total_generated_tokens,
            first_unparsable_count,
            first_correct_count,
            first_incorrect_count,
        ], device=device, dtype=torch.float64)
        
        metrics_tensor = trainer.accelerator.reduce(metrics_tensor, reduction="sum")
        pass_counts_tensor = trainer.accelerator.reduce(pass_counts_tensor, reduction="sum")
        
        num_examples_processed = int(metrics_tensor[0].item())
        total_first_reward = metrics_tensor[1].item()
        total_all_reward = metrics_tensor[2].item()
        total_generations = int(metrics_tensor[3].item())
        total_generated_tokens = int(metrics_tensor[4].item())
        first_unparsable_count = int(metrics_tensor[5].item())
        first_correct_count = int(metrics_tensor[6].item())
        first_incorrect_count = int(metrics_tensor[7].item())

        # Gather per-sample token counts for the first completion so summary stats reflect the full dataset.
        # These are small (O(num_samples)) and safe to gather even when we are not saving full records.
        try:
            from accelerate.utils import gather_object

            gathered_counts = gather_object(first_token_counts)
            combined: List[int] = []
            for entry in gathered_counts:
                if isinstance(entry, list):
                    combined.extend(int(x) for x in entry)
            first_token_counts = combined
        except Exception:
            # If gathering fails, fall back to local token counts (may be per-rank only).
            pass
        
        for i, k in enumerate(k_values):
            pass_counts[k] = int(pass_counts_tensor[i].item())
            
        mean_first = total_first_reward / num_examples_processed if num_examples_processed > 0 else 0.0
        mean_all = total_all_reward / total_generations if total_generations > 0 else 0.0
        
        if save_records:
            from accelerate.utils import gather_object
            records = gather_object(records)
            # Re-sort to original order if global indices were added
            if records and "__global_index__" in records[0]:
                records.sort(key=lambda x: x["__global_index__"])

        # Gather gate trace payload if any rank captured it
        from accelerate.utils import gather_object
        all_gate_traces = gather_object([gate_trace_payload])
        # Find the first non-None trace
        gate_trace_payload = None
        for entry in all_gate_traces:
            # accelerate.gather_object may either preserve the list wrapper (entry is [payload])
            # or return the payload directly (entry is payload). Be robust to both.
            if entry is None:
                continue
            if isinstance(entry, (list, tuple)):
                if len(entry) > 0 and entry[0] is not None:
                    gate_trace_payload = entry[0]
                    break
                continue
            gate_trace_payload = entry
            break
    else:
        mean_first = (sum(first_rewards) / len(first_rewards)) if first_rewards else 0.0
        mean_all = (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        total_generations = len(all_rewards)

    metrics = {
        "num_samples": num_examples_processed,
        "generations_per_sample": num_generations,
        "total_generations": total_generations,
        "mean_reward": mean_first,
        "mean_reward_all_generations": mean_all,
        "accuracy": mean_first,
        "reward_mode": eval_args.reward_mode,
    }
    if dynamic_max_count > 0:
        metrics["dynamic_max_new_tokens_count"] = int(dynamic_max_count)
        metrics["dynamic_max_new_tokens_mean"] = float(dynamic_max_sum / dynamic_max_count)
        metrics["dynamic_max_new_tokens_min"] = int(dynamic_max_min) if dynamic_max_min is not None else 0
        metrics["dynamic_max_new_tokens_max"] = int(dynamic_max_max) if dynamic_max_max is not None else 0

    if total_generated_tokens:
        metrics["total_generated_tokens"] = total_generated_tokens
        metrics["avg_generated_tokens_per_completion"] = (
            total_generated_tokens / total_generations if total_generations else 0.0
        )

    # vLLM-style summary stats are based on the first completion per prompt (pass@1).
    if num_examples_processed:
        metrics["first_unparsable"] = first_unparsable_count
        metrics["first_correct"] = first_correct_count
        metrics["first_incorrect"] = first_incorrect_count
        metrics["first_unparsable_rate"] = first_unparsable_count / num_examples_processed
        metrics["first_correct_rate"] = first_correct_count / num_examples_processed
        metrics["first_incorrect_rate"] = first_incorrect_count / num_examples_processed

    # Token count stats (first completion per prompt).
    if first_token_counts:
        sorted_counts = sorted(first_token_counts)
        metrics["first_completion_tokens_min"] = int(sorted_counts[0])
        metrics["first_completion_tokens_max"] = int(sorted_counts[-1])
        metrics["first_completion_tokens_mean"] = float(sum(sorted_counts) / len(sorted_counts))
        metrics["first_completion_tokens_p50"] = int(sorted_counts[len(sorted_counts) // 2])
        metrics["first_completion_tokens_p90"] = int(sorted_counts[int(0.9 * (len(sorted_counts) - 1))])

    for k in pass_at_k_values:
        metrics[f"pass@{k}"] = (pass_counts[k] / num_examples_processed) if num_examples_processed else 0.0

    eval_dir = _eval_output_dir(training_args)
    if not trainer.is_world_process_zero():
        # Only Rank 0 writes final files
        return metrics

    os.makedirs(eval_dir, exist_ok=True)
    metrics_path = os.path.join(eval_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    if gate_trace_payload is not None:
        gate_trace_path = os.path.join(eval_dir, "gate_trace_example.json")
        with open(gate_trace_path, "w", encoding="utf-8") as fp:
            json.dump(gate_trace_payload, fp, indent=2)
        logger.info(f"Gate trace saved to {gate_trace_path}")
        
        # Automatically generate visualization
        gate_viz_path = os.path.join(eval_dir, "gate_trace_visualization.png")
        _visualize_gate_trace_from_payload(gate_trace_payload, gate_viz_path)

    if eval_args.save_eval_dataset and save_records and records:
        results_dataset = Dataset.from_list(records)
        dataset_dir = os.path.join(eval_dir, "dataset")
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        # results_dataset.save_to_disk(dataset_dir)
        jsonl_path = os.path.join(eval_dir, "results.jsonl")
        results_dataset.to_json(jsonl_path)

    # Print a vLLM-like one-shot summary for pass@1 (first generation).
    if eval_args.reward_mode == "math":
        total_q = int(num_examples_processed)
        if total_q > 0:
            print(
                "\\n============================================================\\n"
                "EVALUATION SUMMARY (vLLM-style, first generation)\\n"
                "============================================================\\n"
                f"Total questions: {total_q}\\n"
                f"Total responses: {total_q}\\n"
                f"Correct responses: {first_correct_count} ({(first_correct_count/total_q):.2%})\\n"
                f"Incorrect responses: {first_incorrect_count} ({(first_incorrect_count/total_q):.2%})\\n"
                f"Unparsable responses: {first_unparsable_count} ({(first_unparsable_count/total_q):.2%})\\n"
                "============================================================\\n",
                flush=True,
            )

    # Log metrics to wandb if available (only on rank 0)
    _log_metrics_to_wandb(metrics, prefix="generation_eval")

    return metrics


def run_perplexity_evaluation(
    trainer,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    t2mlr_args: T2MLRArguments,
    training_args: TrainingArguments,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate perplexity on the evaluation dataset.
    
    This evaluation computes the average cross-entropy loss over the evaluation
    dataset and converts it to perplexity (exp(loss)).
    
    Args:
        trainer: The trainer instance (used to access the model)
        tokenizer: The tokenizer
        eval_dataset: The evaluation dataset
        t2mlr_args: T2MLR configuration arguments
        training_args: Training arguments
        batch_size: Batch size for evaluation (defaults to training_args.per_device_eval_batch_size)
    
    Returns:
        Dictionary with perplexity metrics
    """
    import math
    from torch.utils.data import DataLoader
    from components.data_utils import t2mlr_collator    

    device = training_args.device
    if batch_size is None:
        batch_size = training_args.per_device_eval_batch_size or 8
    
    # Get the model
    model = getattr(trainer, "model", None)
    if model is not None:
        model = getattr(model, "module", model)  # Unwrap DDP/FSDP
    
    if model is None:
        logger.warning("No model found in trainer, cannot compute perplexity")
        return {"perplexity": float("nan"), "eval_loss": float("nan")}
    
    model.eval()
    model.to(device)
    model.config.batch_forward = False # This should already be done when calling model.eval()
    
    # Create collator - use t2mlr_collator to include control_flows when T2MLR is enabled
    collator = t2mlr_collator(tokenizer, t2mlr_args)
    
    # For distributed evaluation, shard the dataset
    world_size = training_args.world_size
    process_index = training_args.process_index
    is_distributed = world_size > 1
    
    if is_distributed:
        eval_dataset_shard = eval_dataset.shard(num_shards=world_size, index=process_index, contiguous=False)
        if trainer.is_world_process_zero():
            logger.info(f"Perplexity eval: Distributed mode, shard {process_index}/{world_size}, {len(eval_dataset_shard)} samples")
    else:
        eval_dataset_shard = eval_dataset
        logger.info(f"Perplexity eval: Single process mode, {len(eval_dataset_shard)} samples")
    
    eval_dataloader = DataLoader(
        eval_dataset_shard,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Determine dtype for autocast
    use_bf16 = getattr(training_args, 'bf16', False)
    use_fp16 = getattr(training_args, 'fp16', False)
    use_autocast = use_bf16 or use_fp16
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    
    total_loss = 0.0
    total_tokens = 0
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    with torch.no_grad():
        with FSDP.summon_full_params(model, writeback=False):
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Perplexity Eval", disable=not trainer.is_world_process_zero())):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)
                
                # Get control flows if available and T2MLR is enabled
                # t2mlr_collator uses "control_flows" (plural) as the key
                control_flows = batch.get("control_flows")
                if control_flows is not None:
                    control_flows = control_flows.to(device)
                    
                # Forward pass with autocast
                if use_autocast and autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        if t2mlr_args.t2mlr_enabled and control_flows is not None:
                            outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            control_flows=control_flows,
                            )
                        else:
                            outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            )
                else:
                    if t2mlr_args.t2mlr_enabled and control_flows is not None:
                        outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        control_flows=control_flows,
                        )
                    else:
                        outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        )
            
                logits = outputs.logits
                del outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                del logits, labels  # Free original tensors
                
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                num_tokens = (shift_labels != -100).sum().item()
                total_loss += loss.item()
                total_tokens += num_tokens
                
                del shift_logits, shift_labels, loss, input_ids, attention_mask, control_flows
                
                if hasattr(model, 'recurrent_cache'):
                    model.recurrent_cache = None
                
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
    
    # Aggregate across processes if distributed
    if is_distributed:
        import torch.distributed as dist
        
        total_loss_tensor = torch.tensor([total_loss], device=device)
        total_tokens_tensor = torch.tensor([total_tokens], device=device)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = total_loss_tensor.item()
        total_tokens = int(total_tokens_tensor.item())
    
    # Compute average loss and perplexity
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")
    else:
        avg_loss = float("nan")
        perplexity = float("nan")
    
    metrics = {
        "perplexity": perplexity,
        "eval_loss": avg_loss,
        "total_tokens": total_tokens,
    }
    
    if trainer.is_world_process_zero():
        logger.info(f"Perplexity evaluation complete:")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Perplexity: {perplexity:.2f}")
        
        # Save metrics
        # eval_dir = _eval_output_dir(training_args)
        # os.makedirs(eval_dir, exist_ok=True)
        # ppl_metrics_path = os.path.join(eval_dir, "perplexity_metrics.json")
        # with open(ppl_metrics_path, "w", encoding="utf-8") as f:
        #    json.dump(metrics, f, indent=2)
        # logger.info(f"  Metrics saved to: {ppl_metrics_path}")
        
        # Log metrics to wandb if available (only on rank 0)
        _log_metrics_to_wandb(metrics, prefix="perplexity_eval")
    
    return metrics


def should_run_perplexity_eval(data_args) -> bool:
    """
    Check if perplexity evaluation should be run based on dataset name.
    
    Returns True if train_dataset_name contains 'tinystories', 'wikitext', or 'fineweb'.
    """
    dataset_name = getattr(data_args, "train_dataset_name", "") or ""
    dataset_name_lower = dataset_name.lower()
    return (
        "tinystories" in dataset_name_lower
        or "wikitext" in dataset_name_lower
        or "fineweb" in dataset_name_lower
    )
