# T2MLR: Transformer with Temporal Middle-Layer Recurrence

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-TBC-b31b1b.svg?style=flat)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Email me](https://img.shields.io/badge/Contact-6fcf97?logo=gmail)](mailto:xingyu.zhu@princeton.com)
<div>
  <img width="90%" src=assets/T2MLR_Teaser.png>
</div>
</div>

---

We introduce Transformers with Temporal Middle-Layer Recurrence (T2MLR), a generalized Transformer architecture that integrates attention and recurrence by routing a lightweight temporal pathway through the middle layers. Motivated by latent-reasoning and looped-Transformer lines of work, T2MLR injects intermediate representations from deeper layers of the previous token into earlier layers of the current token via a gated recurrent pathway, enabling iterative latent computation while preserving dense, token-level supervision.

Across natural-language pretraining and multi-hop reasoning finetuning, T2MLR consistently outperforms parameter-matched Transformer baselines at the same inference compute. Moreover, we find that looping only a middle-layer block (as little as 20% of all layers) often outperforms full-layer looping. This offers a new perspective on latent reasoning in Transformers: effective iterative refinement does not necessarily require full-stack recurrence. It can instead be achieved more effectively through targeted middle-layer recurrence.

## Overview

This repository is the release version of the T2MLR training codebase used in the paper. It contains the core T2MLR implementation (as a wrapper over the standard transformer instance) together with task-specific scripts for pretraining and supervised finetuning.

In particular, we include the necessary scripts to conduct training for:

- S5 retrieval (Section 3)
- FineWeb pretraining (Section 4.1)
- Pathfinding (Section 4.2)
- ProsQA (Section 4.2)
- Variable assignment (Section 4.2)
- GSM8K (Section 4.3)
- Future Token Prediction training (Section B)

This release is intentionally trimmed down for public use. It does not include internal experiment logs, model caches, visualization notebooks, historical task folders, or the full test suite from the original `T2MLR_Training` workspace.

## Repository Structure

```text
T2MLR/
├── src/
│   ├── train.py
│   ├── train_minimal.py
│   ├── train_ftp.py
│   ├── components/
│   ├── modeling/
│   └── t2mlr_wrapper/
├── scripts/
│   ├── fineweb/
│   ├── gsm8k/
│   ├── ftp/
│   ├── pathfinding/
│   ├── prosqa/
│   ├── s5_retrieval/
│   └── variable_assignment/
├── assets/
└── requirements.txt
```

Core entrypoints:

- `src/train.py`: general training entrypoint for pretraining and retrieval-style experiments.
- `src/train_minimal.py`: lightweight supervised finetuning entrypoint for GSM8K, pathfinding, ProsQA, and variable assignment.
- `src/train_ftp.py`: Future token prediction training entrypoint.
- `src/t2mlr_wrapper/`: T2MLR/T2MLR model wrapper and recurrence components.
- `src/components/`: shared training, evaluation, preprocessing, and argument utilities.
- `scripts/<task>/train_*.sh`: runnable examples for each task.
- `scripts/<task>/submit_*.sh`: launcher scripts for sweeps or cluster submission.
- `scripts/<task>/sweep_params.yaml`: portable example sweep configurations.

## Wrapper Structure

The core T2MLR logic lives in `src/t2mlr_wrapper/t2mlr_wrapper.py`, whose main class `T2MLRWrapper` wraps a standard Hugging Face causal language model and augments it with temporal middle-layer recurrence.

At a high level, the wrapper is organized into three layers of responsibility:

- Model construction: `T2MLRWrapper` loads or receives a base model, reads `T2MLRConfig`, resolves the recurrent layer range (`l_start`, `l_end`), and builds the selected recurrent mixing module.
- Recurrent injection: the wrapper replaces the block at `l_start` with a `BlockWrapper`, which mixes the previous token's hidden state back into the current token only when `control_flows > 1`.
- Forward-path routing: `forward()` dispatches between regular Transformer execution, batch-approximate recurrent training, exact sequence recurrence, and simple cached recurrent inference depending on the mode and input shape.

The surrounding files in `src/t2mlr_wrapper/` separate these concerns:

- `block_wrapper.py`: injects recurrent states into the wrapped Transformer block and records optional gating statistics.
- `t2mlr_gate_zoo.py`: defines the available recurrent mixing modules used to combine current and recurrent hidden states.
- `t2mlr_config.py`: stores wrapper-specific configuration on top of the base model config.
- `inference_wrapper.py` and `skip_layer_inference_wrapper.py`: helper logic for recurrent inference and skip-layer execution paths.
- `model_io_utils.py`: utilities for loading base models, configs, dtypes, and checkpoints into the wrapped model.

### Forward Mechanisms

`T2MLRWrapper.forward()` switches among several execution modes depending on whether T2MLR is enabled, whether recurrence is present in `control_flows`, whether the model is in training mode, and whether the input is a full sequence or a single decode step.

- Regular forward: used when T2MLR is disabled or when all `control_flows <= 1`. This is just the underlying Transformer forward pass, with optional seeding of the recurrent cache for later decoding.
- `batch_approximate_forward()`: the main training-time path for recurrent experiments. It runs a parallel Jacobi approximation (see section 2.4) of recurrence over the full sequence, which is much faster than exact token-by-token recurrence and is what the released training scripts use when they pass `--batch_forward True`.
- `exact_sequence_recurrent_forward()`: the exact full-sequence recurrent path. It processes recurrent tokens sequentially while still collapsing contiguous non-recurrent spans into standard parallel Transformer calls. This is the most faithful recurrence mode for evaluation or analysis, but it is slower.
- `simple_recurrent_forward()`: the cached single-step recurrent path. It carries the hidden state captured at `l_end` from the previous token and injects it at `l_start` on the next token. This is the path used during autoregressive generation and other decode-style single-token inference.

In the released tasks, the mapping is:

- FineWeb pretraining, GSM8K, Pathfinding, ProsQA, Variable Assignment, and S5 retrieval training all enable `batch_approximate_forward()` through their shell scripts.
- FTP training also forces the T2MLR backbone into `batch_approximate_forward()` during its forward pass so multi-head training stays parallel.
- Generation-style evaluation uses `simple_recurrent_forward()` through `generate()`, with automatic or manual `control_flows`.
- Exact recurrent evaluation or analysis of multi-token inputs uses `exact_sequence_recurrent_forward()` when batch-forward mode is disabled.

## Quick Start

Create an environment and install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional packages:

- `flash-attn` for FlashAttention-backed execution
- `liger-kernel` for Liger kernels
- `trl` if you plan to re-enable RL-specific training paths

## Supported Tasks

| Task | Script directory | Notes |
| --- | --- | --- |
| S5 retrieval | `scripts/s5_retrieval/` | Includes retrieval dataset generation and baseline scripts |
| FineWeb pretraining | `scripts/fineweb/` | Uses the general trainer in `src/train.py` |
| GSM8K | `scripts/gsm8k/` | SFT-style setup via `src/train_minimal.py` |
| ProsQA | `scripts/prosqa/` | Expects local train/eval data unless overridden |
| Variable assignment | `scripts/variable_assignment/` | Includes a local dataset generator |
| Pathfinding | `scripts/pathfinding/` | Includes a local dataset generator |
| Future-Token Prediction | `scripts/ftp/` | Uses `src/train_ftp.py` |

## Data Preparation

Dataset handling differs by task:

- FineWeb pretraining default to Hugging Face-hosted data or externally configured sources.
- GSM8K defaults to `whynlp/gsm8k-aug`.
- Pathfinding, ProsQA, and variable assignment expect local files under `data/` by default.
- S5 retrieval expects generated JSONL files under `data/<dataset_tag>/`.

If your data lives elsewhere, override the paths through environment variables such as `TRAIN_DATA_PATH` and `EVAL_DATA_PATH` in the provided scripts.

Generate synthetic datasets with:

```bash
python scripts/pathfinding/make_pathfinding_dataset.py --help
python scripts/variable_assignment/make_variable_assignment_dataset.py --help
python scripts/s5_retrieval/data_generation/make_s5_retrieval_dataset.py --help
```

## Running Experiments

Launch a single experiment:

```bash
bash scripts/gsm8k/train_gsm8k.sh
```

Launch an example sweep or submission script:

```bash
bash scripts/gsm8k/submit_gsm8k.sh
```

The shell scripts are written to be launched from the repository root. Most settings, including data locations, output directories, and runtime parameters, can be adjusted through environment variables or the accompanying `sweep_params.yaml` files.

## Citation
```bibtex
@inproceedings{
    cai2026tmlr,
    title={T2{MLR}: Transformer with Temporal Middle-Layer Recurrence},
    author={Ziyang Cai and Xingyu Zhu and Yihe Dong and Yinghui He and Sanjeev Arora},
    booktitle={Workshop on Latent {\&} Implicit Thinking {\textendash} Going Beyond CoT Reasoning},
    year={2026},
    url={https://openreview.net/forum?id=fQbk1EQWBO}
}
```
