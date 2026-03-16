import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from transformers import PreTrainedTokenizer

from datasets import Dataset
from components.all_arguments import GenerationEvalArguments

logger = logging.getLogger(__name__)


def ensure_eos(tokenizer: PreTrainedTokenizer, ids):
    if tokenizer.eos_token_id is not None and (len(ids) == 0 or ids[-1] != tokenizer.eos_token_id):
        return ids + [tokenizer.eos_token_id]
    return ids


def _coerce_to_input_ids(tokens: Any) -> List[int]:
    if tokens is None:
        return []

    if isinstance(tokens, (list, tuple)):
        ids = list(tokens)
    else:
        # BatchEncoding/dict-like (or tokenizer output) -> prefer input_ids
        ids = tokens.get("input_ids", tokens) if hasattr(tokens, "get") else tokens

    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    # Handle accidental batching: [[...]] -> [...]
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]

    return list(ids)


class CustomCtrlFlowTokenizer:
    """Base class for custom control flow tokenization."""

    CONFIG_KEYS: List[str] = []

    @classmethod
    def from_args(cls, data_args: Any, **override_kwargs) -> "CustomCtrlFlowTokenizer":
        all_kwargs = _args_to_dict(data_args)
        all_kwargs.update(override_kwargs)
        kwargs = {k: all_kwargs[k] for k in cls.CONFIG_KEYS if k in all_kwargs} if cls.CONFIG_KEYS else all_kwargs
        return cls(**kwargs)

    def __init__(self, **kwargs):
        pass

    def build_preprocess_fn(self, tokenizer: PreTrainedTokenizer, eval_args: GenerationEvalArguments) -> Callable:
        raise NotImplementedError


_CTRL_FLOW_TOKENIZER_REGISTRY: Dict[str, Type[CustomCtrlFlowTokenizer]] = {}


def register_ctrl_flow_tokenizer(name: str) -> Callable[[Type[CustomCtrlFlowTokenizer]], Type[CustomCtrlFlowTokenizer]]:
    def decorator(cls: Type[CustomCtrlFlowTokenizer]) -> Type[CustomCtrlFlowTokenizer]:
        _CTRL_FLOW_TOKENIZER_REGISTRY[name] = cls
        return cls

    return decorator


def _args_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            return dict(vars(obj))
    try:
        return dict(vars(obj))
    except Exception:
        return {}


def apply_custom_ctrl_flow_tokenization(
    dataset: Dataset,
    *,
    role: str,
    tokenizer: PreTrainedTokenizer,
    data_args: Any,
    eval_args: GenerationEvalArguments,
    num_proc: Optional[int] = None,
    **override_kwargs,
) -> Dataset:
    name = getattr(data_args, "custom_ctrl_flow_tokenization", "t2mlr")
    if not name or name.lower() == "none":
        return dataset

    if name not in _CTRL_FLOW_TOKENIZER_REGISTRY:
        raise ValueError(
            f"CustomCtrlFlowTokenizer '{name}' not found. Available: {sorted(_CTRL_FLOW_TOKENIZER_REGISTRY.keys())}"
        )

    try:
        is_rank0 = int(os.environ.get("RANK", "0")) == 0
    except Exception:
        is_rank0 = True

    if is_rank0:
        print(f"[custom_ctrl_flow_tokenization] role={role} applying='{name}'", flush=True)

    tokenizer_obj = _CTRL_FLOW_TOKENIZER_REGISTRY[name].from_args(data_args, **override_kwargs)
    preprocess_fn = tokenizer_obj.build_preprocess_fn(tokenizer, eval_args)

    return dataset.map(
        preprocess_fn,
        desc=f"Tokenizing {role} dataset ({name})",
        num_proc=num_proc,
    )


@register_ctrl_flow_tokenizer("t2mlr")
class T2MLRCtrlFlowTokenizer(CustomCtrlFlowTokenizer):
    def build_preprocess_fn(
        self,
        tokenizer: PreTrainedTokenizer,
        eval_args: GenerationEvalArguments,
        prompt_only: bool = False,
        control_flow_all_recurrent: bool = False,
        label_mask_prompt: bool = False,
        control_flow_split_answer: bool = False,
    ) -> Callable:
        is_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
        tokenize_kwargs = {"add_special_tokens": False}

        if prompt_only:
            prompt_col = eval_args.get_eval_prompt_column()
            response_col = eval_args.get_eval_response_column()
        else:
            prompt_col = eval_args.prompt_column
            response_col = eval_args.response_column

        def _split_response_text(text: str) -> Tuple[str, str]:
            idx = text.find("####")
            marker_len = 4
            if idx < 0:
                idx = text.find("###")
                marker_len = 3
            if idx < 0:
                return text, ""
            steps = text[:idx].rstrip()
            answer = text[idx:].lstrip()
            if not answer:
                answer = text[idx: idx + marker_len]
            return steps, answer

        def preprocess(example, prompt_col=prompt_col, response_col=response_col):
            if "control_flow" in example and "input_ids" in example:
                return example

            prompt_text = str(example[prompt_col])
            response_text = str(example[response_col])
            
            # Add newline separator between prompt and response for non-chat models
            # This helps the model learn the boundary between question and COT response
            if not is_chat and not prompt_text.endswith("\n\n"):
                prompt_text = prompt_text + "\n\n"

            if is_chat:
                messages_prompt_only = [{"role": "user", "content": prompt_text}]
                try:
                    prompt_tokens = tokenizer.apply_chat_template(
                        messages_prompt_only,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                except TypeError:
                    prompt_tokens = tokenizer.apply_chat_template(messages_prompt_only, tokenize=True)

                prompt_ids = _coerce_to_input_ids(prompt_tokens)
                if control_flow_split_answer and not prompt_only and not control_flow_all_recurrent:
                    steps_text, answer_text = _split_response_text(response_text)
                    steps_ids = _coerce_to_input_ids(tokenizer(steps_text, **tokenize_kwargs)) if steps_text else []
                    answer_ids = _coerce_to_input_ids(tokenizer(answer_text, **tokenize_kwargs)) if answer_text else []
                    assistant_ids = steps_ids + answer_ids
                    assistant_ids = ensure_eos(tokenizer, assistant_ids)
                else:
                    assistant_ids = _coerce_to_input_ids(tokenizer(response_text, **tokenize_kwargs))
                    assistant_ids = ensure_eos(tokenizer, assistant_ids)

                prompt_len = len(prompt_ids)
                if prompt_only:
                    input_ids = prompt_ids
                else:
                    input_ids = prompt_ids + assistant_ids

                if control_flow_all_recurrent:
                    control_flow = [2] * len(input_ids)
                elif prompt_only:
                    control_flow = [1] * prompt_len
                elif control_flow_split_answer:
                    steps_text, answer_text = _split_response_text(response_text)
                    steps_ids = _coerce_to_input_ids(tokenizer(steps_text, **tokenize_kwargs)) if steps_text else []
                    answer_ids = _coerce_to_input_ids(tokenizer(answer_text, **tokenize_kwargs)) if answer_text else []
                    cf_steps = [2] * len(steps_ids)
                    cf_answer = [3] * len(answer_ids)
                    control_flow = [1] * prompt_len + cf_steps + cf_answer
                    if tokenizer.eos_token_id is not None and input_ids and input_ids[-1] == tokenizer.eos_token_id:
                        eos_cf = 3 if cf_answer else (2 if cf_steps else 2)
                        if len(control_flow) < len(input_ids):
                            control_flow.append(eos_cf)
                else:
                    control_flow = [1] * prompt_len + [2] * len(assistant_ids)
            else:
                prompt_ids = tokenizer(prompt_text, **tokenize_kwargs)["input_ids"]
                if control_flow_split_answer and not prompt_only and not control_flow_all_recurrent:
                    steps_text, answer_text = _split_response_text(response_text)
                    steps_ids = tokenizer(steps_text, **tokenize_kwargs)["input_ids"] if steps_text else []
                    answer_ids = tokenizer(answer_text, **tokenize_kwargs)["input_ids"] if answer_text else []
                    response_ids = steps_ids + answer_ids
                    response_ids = ensure_eos(tokenizer, response_ids)
                else:
                    response_ids = tokenizer(response_text, **tokenize_kwargs)["input_ids"]
                    response_ids = ensure_eos(tokenizer, response_ids)

                prompt_len = len(prompt_ids)
                if prompt_only:
                    input_ids = prompt_ids
                else:
                    input_ids = prompt_ids + response_ids

                if control_flow_all_recurrent:
                    control_flow = [2] * len(input_ids)
                elif prompt_only:
                    control_flow = [1] * prompt_len
                elif control_flow_split_answer:
                    steps_text, answer_text = _split_response_text(response_text)
                    steps_ids = tokenizer(steps_text, **tokenize_kwargs)["input_ids"] if steps_text else []
                    answer_ids = tokenizer(answer_text, **tokenize_kwargs)["input_ids"] if answer_text else []
                    cf_steps = [2] * len(steps_ids)
                    cf_answer = [3] * len(answer_ids)
                    control_flow = [1] * len(prompt_ids) + cf_steps + cf_answer
                    if tokenizer.eos_token_id is not None and input_ids and input_ids[-1] == tokenizer.eos_token_id:
                        eos_cf = 3 if cf_answer else (2 if cf_steps else 2)
                        if len(control_flow) < len(input_ids):
                            control_flow.append(eos_cf)
                else:
                    control_flow = [1] * len(prompt_ids) + [2] * len(response_ids)

            example["input_ids"] = input_ids
            example["control_flow"] = control_flow
            if label_mask_prompt and not prompt_only and "labels" not in example:
                example["labels"] = ([-100] * prompt_len) + input_ids[prompt_len:]
            example["prompt"] = prompt_text
            example["response"] = response_text
            example["nonrecur_len"] = prompt_len
            example["seq_len"] = len(input_ids)
            return example

        return preprocess


def build_truncate_fn(max_seq_length: Optional[int]) -> Callable:
    """Return a function that truncates sequences and recomputes lengths."""

    def truncate_processed_example(example):
        if max_seq_length is not None and max_seq_length > 0:
            for key in ("input_ids", "labels", "attention_mask", "control_flow"):
                if key not in example or example[key] is None:
                    continue
                seq = list(example[key])
                if len(seq) > max_seq_length:
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

    return truncate_processed_example


__all__ = [
    "ensure_eos",
    "apply_custom_ctrl_flow_tokenization",
    "build_truncate_fn",
    "register_ctrl_flow_tokenizer",
    "CustomCtrlFlowTokenizer",
    "T2MLRCtrlFlowTokenizer",
]
