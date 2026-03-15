"""Utility script to pre-download models, tokenizers, and datasets.

Run this once on a machine with network access to warm the local HF cache so
training/eval jobs can run offline.
"""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_TO_CACHE = [
    {"model_id": "Qwen/Qwen3-0.6B", "trust_remote_code": True},
    {"model_id": "HuggingFaceTB/SmolLM-135M", "trust_remote_code": True},
    {"model_id": "meta-llama/Llama-3.2-1B", "trust_remote_code": True},
    {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "trust_remote_code": True},
    {"model_id": "Qwen/Qwen2.5-1.5B", "trust_remote_code": True},
    {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "trust_remote_code": True},
    {"model_id": "Qwen/Qwen3-1.7B", "trust_remote_code": True},
]

DATASETS_TO_CACHE = [
    {"dataset_id": "nvidia/OpenMathReasoning", "config": "default", "split": "cot[:1%]"},
    {"dataset_id": "AI-MO/NuminaMath-CoT", "config": "default", "split": "train"},
    {"dataset_id": "whynlp/gsm8k-aug", "config": "default", "split": "test"},
]


def download_model(model_id: str, revision: Optional[str] = None, trust_remote_code: bool = False) -> None:
    print(f"[download] model: {model_id}")
    AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


def download_dataset(dataset_id: str, config: Optional[str], split: str) -> None:
    cfg_label = config or "default"
    print(f"[download] dataset: {dataset_id} (config={cfg_label}, split={split})")
    kwargs = {"split": split}
    if config:
        kwargs["name"] = config
    load_dataset(dataset_id, **kwargs)


if __name__ == "__main__":
    # Models/tokenizers to cache
    for model_kwargs in MODELS_TO_CACHE:
        download_model(**model_kwargs)

    # Datasets to cache (light subsamples to avoid huge downloads)
    for dataset_kwargs in DATASETS_TO_CACHE:
        download_dataset(**dataset_kwargs)

    print("[INFO] Download warmup complete.")
