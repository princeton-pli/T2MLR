import os
import json
import logging
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional, Type, List, Set, Tuple

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = Any  # type: ignore

logger = logging.getLogger(__name__)

def _load_prompt_template(template_path: str, template_key: str) -> str:
    with open(template_path, "r") as handle:
        templates = json.load(handle)
    if template_key not in templates:
        available = list(templates.keys())
        raise ValueError(
            f"Template key '{template_key}' not found in {template_path}. Available keys: {available}"
        )
    return templates[template_key]


def _extract_template_fields(template: str) -> Set[str]:
    pattern = r"{arg_([^}]+)}"
    matches = re.findall(pattern, template)
    return set(matches)


def _validate_template_fields(template_fields: Set[str], dataset_columns: List[str]) -> Tuple[Set[str], Set[str]]:
    dataset_columns_set = set(dataset_columns)
    valid_fields = template_fields.intersection(dataset_columns_set)
    missing_fields = template_fields - dataset_columns_set
    return valid_fields, missing_fields


def _format_template_with_row(template: str, row: Dict[str, Any], template_fields: Set[str]) -> str:
    formatted = template
    for field in template_fields:
        placeholder = f"{{arg_{field}}}"
        value = "" if row.get(field) is None else str(row.get(field))
        formatted = formatted.replace(placeholder, value)
    return formatted


class CustomDatasetPreprocessor:
    """
    Base class for custom dataset preprocessing/formatting.

    Patterned after `rcot_wrapper/rcot_gate_zoo.py`:
    - registry-driven construction
    - optional config-key filtering
    - `apply()` performs dataset -> dataset mapping
    """

    CONFIG_KEYS: List[str] = []

    @classmethod
    def from_args(cls, data_args: Any, **override_kwargs) -> "CustomDatasetPreprocessor":
        all_kwargs = _args_to_dict(data_args)
        all_kwargs.update(override_kwargs)
        if cls.CONFIG_KEYS:
            kwargs = {k: all_kwargs[k] for k in cls.CONFIG_KEYS if k in all_kwargs}
        else:
            kwargs = all_kwargs
        logger.info("Initializing CustomDatasetPreprocessor: %s", cls.__name__)
        return cls(**kwargs)

    def __init__(self, **kwargs):
        pass

    def apply(
        self,
        dataset: Dataset,
        *,
        role: str,
        data_args: Any,
        eval_args: Optional[Any] = None,
        dataset_label: Optional[str] = None,
    ) -> Dataset:
        return dataset


_PREPROCESSOR_REGISTRY: Dict[str, Type[CustomDatasetPreprocessor]] = {}


def register_custom_preprocessor(name: str) -> Callable[[Type[CustomDatasetPreprocessor]], Type[CustomDatasetPreprocessor]]:
    def decorator(cls: Type[CustomDatasetPreprocessor]) -> Type[CustomDatasetPreprocessor]:
        _PREPROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_custom_preprocessor_class(name: str) -> Type[CustomDatasetPreprocessor]:
    if name not in _PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"CustomDatasetPreprocessor '{name}' not found. Available: {sorted(_PREPROCESSOR_REGISTRY.keys())}"
        )
    return _PREPROCESSOR_REGISTRY[name]


def list_custom_preprocessors() -> List[str]:
    return sorted(_PREPROCESSOR_REGISTRY.keys())


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


def apply_custom_preprocessing(
    dataset: Dataset,
    *,
    role: str,
    data_args: Any,
    eval_args: Optional[Any] = None,
) -> Dataset:
    pipeline = getattr(data_args, "custom_dataset_preprocessing", None)
    if not pipeline:
        return dataset

    try:
        is_rank0 = int(os.environ.get("RANK", "0")) == 0
    except Exception:
        is_rank0 = True

    dataset_label = None
    if role == "train":
        dataset_label = getattr(data_args, "train_dataset_config", None) or getattr(data_args, "train_dataset_name", None)
    else:
        dataset_label = getattr(data_args, "eval_dataset_config", None) or getattr(data_args, "eval_dataset_name", None)

    names = [pipeline] if isinstance(pipeline, str) else list(pipeline)
    out = dataset
    for name in names:
        name = str(name).strip()
        if not name or name.lower() == "none":
            continue
        if name not in _PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"CustomDatasetPreprocessor '{name}' not found. Available: {sorted(_PREPROCESSOR_REGISTRY.keys())}"
            )
        if is_rank0:
            print(f"[custom_preprocessing] role={role} applying='{name}'", flush=True)
        preprocessor_cls = _PREPROCESSOR_REGISTRY[name]
        preprocessor = preprocessor_cls.from_args(data_args)
        out = preprocessor.apply(
            out,
            role=role,
            data_args=data_args,
            eval_args=eval_args,
            dataset_label=dataset_label,
        )
    return out


@register_custom_preprocessor("none")
class NoOpPreprocessor(CustomDatasetPreprocessor):
    def apply(self, dataset: Dataset, *, role: str, data_args: Any, eval_args: Optional[Any] = None, dataset_label: Optional[str] = None) -> Dataset:  # noqa: E501
        return dataset

@register_custom_preprocessor("coconut")
@register_custom_preprocessor("gsm8k_aug")
class StepsAnswerFormatter(CustomDatasetPreprocessor):
    """
    Formats datasets with steps and answer columns:
    - expects columns: `steps`, `answer`
    - creates/rewrites `response` column into: "<reasoning lines>\\n### <final>"
    Used for both GSM8K and ProsQA datasets.
    """
    
    OUTPUT_COLUMN: str = "response"
    
    def apply(
        self,
        dataset: Dataset,
        *,
        role: str,
        data_args: Any,
        eval_args: Optional[Any] = None,
        dataset_label: Optional[str] = None,
    ) -> Dataset:
        required = {"steps", "answer"}
        if not required.issubset(set(getattr(dataset, "column_names", []))):
            return dataset

        def _format_example(example: Dict[str, Any]) -> Dict[str, Any]:
            answer = example.get("answer")
            # Skip if output column already formatted
            output_value = example.get(self.OUTPUT_COLUMN)
            if isinstance(output_value, str) and ("###" in output_value or "####" in output_value):
                return example

            steps_field = example.get("steps")
            step_lines: List[str] = []
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
            example[self.OUTPUT_COLUMN] = "\n".join(parts) if parts else raw_answer
            return example

        cpu_count = os.cpu_count() or 1
        map_kwargs: Dict[str, Any] = {
            "desc": f"Formatting reasoning for {role} dataset ({dataset_label or 'unknown'})"
        }
        if cpu_count > 1:
            map_kwargs["num_proc"] = min(8, cpu_count)
        return dataset.map(_format_example, **map_kwargs)


@register_custom_preprocessor("qwen_math_prompt")
class QwenMathPromptPreprocessor(CustomDatasetPreprocessor):
    """Apply the qwen math prompt template using the vLLM-style formatting logic."""

    def apply(
        self,
        dataset: Dataset,
        *,
        role: str,
        data_args: Any,
        eval_args: Optional[Any] = None,
        dataset_label: Optional[str] = None,
    ) -> Dataset:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        template_path = os.path.join(
            base_dir, "general_inference_eval", "configs", "eval", "prompt_templates.json"
        )
        template_key = "qwen_math_prompt"

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        prompt_column = None
        if eval_args is not None:
            if role == "eval":
                if hasattr(eval_args, "get_eval_prompt_column"):
                    prompt_column = eval_args.get_eval_prompt_column()
                elif hasattr(eval_args, "prompt_column"):
                    prompt_column = eval_args.prompt_column
            elif hasattr(eval_args, "prompt_column"):
                prompt_column = eval_args.prompt_column
        if not prompt_column:
            raise ValueError("qwen_math_prompt requires a configured prompt column; none was provided.")

        template = _load_prompt_template(template_path, template_key)
        template_fields = _extract_template_fields(template)
        if template_fields != {"problem"}:
            raise ValueError(
                "qwen_math_prompt expects exactly one template field '{arg_problem}'. "
                f"Found: {sorted(template_fields)}"
            )
        dataset_columns = getattr(dataset, "column_names", [])
        if prompt_column not in dataset_columns:
            raise ValueError(
                f"Prompt column '{prompt_column}' missing from dataset. Available columns: {dataset_columns}"
            )

        def _format_example(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt_value = "" if example.get(prompt_column) is None else str(example.get(prompt_column))
            example[prompt_column] = template.replace("{arg_problem}", prompt_value)
            return example

        cpu_count = os.cpu_count() or 1
        map_kwargs: Dict[str, Any] = {
            "desc": f"Applying qwen math prompt for {role} dataset ({dataset_label or 'unknown'})"
        }
        if cpu_count > 1:
            map_kwargs["num_proc"] = min(8, cpu_count)

        return dataset.map(_format_example, **map_kwargs)
