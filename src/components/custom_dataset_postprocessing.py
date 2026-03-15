import os
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional, Type, List

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    Dataset = Any  # type: ignore

import numpy as np

logger = logging.getLogger(__name__)


class CustomDatasetPostprocessor:
    """Base class for custom dataset postprocessing (after tokenization)."""

    CONFIG_KEYS: List[str] = []

    @classmethod
    def from_args(cls, data_args: Any, **override_kwargs) -> "CustomDatasetPostprocessor":
        all_kwargs = _args_to_dict(data_args)
        all_kwargs.update(override_kwargs)
        kwargs = {k: all_kwargs[k] for k in cls.CONFIG_KEYS if k in all_kwargs} if cls.CONFIG_KEYS else all_kwargs
        logger.info("Initializing CustomDatasetPostprocessor: %s", cls.__name__)
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
    ) -> Dataset:
        return dataset


_POSTPROCESSOR_REGISTRY: Dict[str, Type[CustomDatasetPostprocessor]] = {}


def register_custom_postprocessor(name: str) -> Callable[[Type[CustomDatasetPostprocessor]], Type[CustomDatasetPostprocessor]]:
    def decorator(cls: Type[CustomDatasetPostprocessor]) -> Type[CustomDatasetPostprocessor]:
        _POSTPROCESSOR_REGISTRY[name] = cls
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


def apply_custom_postprocessing(
    dataset: Dataset,
    *,
    role: str,
    data_args: Any,
    eval_args: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    **override_kwargs,
) -> Dataset:
    names_obj = pipeline if pipeline is not None else getattr(data_args, "custom_dataset_postprocessing", None)
    if not names_obj:
        return dataset

    try:
        is_rank0 = int(os.environ.get("RANK", "0")) == 0
    except Exception:
        is_rank0 = True

    names = [names_obj] if isinstance(names_obj, str) else list(names_obj)
    out = dataset
    for name in names:
        name = str(name).strip()
        if not name or name.lower() == "none":
            continue
        if name not in _POSTPROCESSOR_REGISTRY:
            raise ValueError(
                f"CustomDatasetPostprocessor '{name}' not found. Available: {sorted(_POSTPROCESSOR_REGISTRY.keys())}"
            )
        if is_rank0:
            print(f"[custom_postprocessing] role={role} applying='{name}'", flush=True)
        out = _POSTPROCESSOR_REGISTRY[name].from_args(data_args, **override_kwargs).apply(
            out, role=role, data_args=data_args, eval_args=eval_args
        )
    return out


@register_custom_postprocessor("none")
class NoOpPostprocessor(CustomDatasetPostprocessor):
    pass


@register_custom_postprocessor("insert_pause_tokens")
class InsertPauseTokensPostprocessor(CustomDatasetPostprocessor):
    CONFIG_KEYS = [
        "tokenizer",
        "model_args",
        "num_proc",
    ]

    def __init__(
        self,
        tokenizer: Any = None,
        model_args: Any = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.num_proc = None if num_proc is None else max(int(num_proc), 1)

    @staticmethod
    def _validate_pause_token_config(tokenizer: Any, data_args: Any, model_args: Any) -> tuple[int, float]:
        if tokenizer is None or model_args is None:
            raise ValueError("insert_pause_tokens requires passing tokenizer=... and model_args=... to the postprocessor.")

        tokenizer_name = (
            (getattr(model_args, "tokenizer_name_or_path", None) or getattr(model_args, "model_name_or_path", "") or "")
        ).lower()
        if "llama-3" not in tokenizer_name and "llama3" not in tokenizer_name:
            raise ValueError(
                f"Pause token insertion requires a Llama 3 tokenizer. Got {getattr(model_args, 'tokenizer_name_or_path', None)}."
            )

        pause_token = "<|reserved_special_token_0|>"
        pause_token_id = tokenizer.convert_tokens_to_ids(pause_token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if (unk is not None and pause_token_id == unk) or pause_token_id < 0:
            raise ValueError(f"Pause token '{pause_token}' not found in tokenizer vocabulary.")

        mean = getattr(data_args, "pause_token_mean", None)
        if mean is None:
            raise ValueError("insert_pause_tokens is True but pause_token_mean was not provided.")
        if mean < 0:
            raise ValueError("pause_token_mean must be non-negative.")

        logger.info(
            "Pause token insertion enabled: '%s' (ID: %s), mean=%s, only_recurrent=%s, seed=%s",
            pause_token,
            pause_token_id,
            mean,
            getattr(data_args, "pause_token_only_recurrent", True),
            getattr(data_args, "pause_token_seed", 42),
        )
        return int(pause_token_id), float(mean)

    def apply(
        self,
        dataset: Dataset,
        *,
        role: str,
        data_args: Any,
        eval_args: Optional[Any] = None,
    ) -> Dataset:
        if not getattr(data_args, "insert_pause_tokens", False):
            return dataset
        pause_token_id, pause_token_mean = self._validate_pause_token_config(self.tokenizer, data_args, self.model_args)
        if pause_token_mean <= 0.0:
            return dataset

        seed = int(getattr(data_args, "pause_token_seed", 42))
        only_recurrent = bool(getattr(data_args, "pause_token_only_recurrent", True))

        def _insert(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            if "control_flow" not in example or "input_ids" not in example:
                return example

            rng = np.random.default_rng(seed + idx)
            orig_ids = list(example["input_ids"])
            orig_cf = list(example["control_flow"])

            valid = [
                pos
                for pos, cf in enumerate(orig_cf)
                if (not only_recurrent) or cf == 2
            ]
            if not valid:
                return example

            ids = orig_ids.copy()
            cf_out = orig_cf.copy()
            offset = 0
            for pos in valid:
                k = rng.poisson(pause_token_mean)
                if k <= 0:
                    continue
                insert_at = pos + offset
                cf_val = orig_cf[pos]
                for _ in range(int(k)):
                    ids.insert(insert_at, pause_token_id)
                    cf_out.insert(insert_at, cf_val)
                    insert_at += 1
                    offset += 1

            example["input_ids"] = ids
            example["control_flow"] = cf_out
            return example

        kwargs: Dict[str, Any] = {"with_indices": True, "desc": f"Inserting pause tokens ({role})"}
        if self.num_proc is not None:
            kwargs["num_proc"] = self.num_proc
        return dataset.map(_insert, **kwargs)


