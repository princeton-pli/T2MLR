# RCOT Release

Standalone release of the RCOT training code for:

- FineWeb pretraining
- GSM8K
- Medusa head training
- Pathfinding
- ProsQA
- S5 retrieval
- Variable assignment

This release intentionally excludes tests, visualization notebooks, experiment logs, model caches, and historical task folders from `RCOT_Training`.

## Layout

- `src/train.py`: general trainer used by FineWeb and S5 retrieval.
- `src/train_minimal.py`: streamlined SFT trainer used by GSM8K, pathfinding, ProsQA, and variable assignment.
- `src/train_medusa.py`: Medusa multi-head training entrypoint.
- `src/components/`, `src/modeling/`, `src/rcot_wrapper/`: shared RCOT implementation.
- `scripts/<task>/train_*.sh`: direct launch scripts.
- `scripts/<task>/submit_*.sh`: sweep submitters.
- `scripts/<task>/sweep_params.yaml`: small portable example sweeps.
- `scripts/pathfinding/make_pathfinding_dataset.py`: pathfinding dataset generator.
- `scripts/variable_assignment/make_variable_assignment_dataset.py`: variable-assignment dataset generator.
- `scripts/s5_retrieval/data_generation/make_s5_retrieval_dataset.py`: S5 retrieval dataset generator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

- `flash-attn` for FlashAttention-backed training
- `liger-kernel` for Liger kernels
- `trl` only if you later re-enable RL-specific paths

## Data

- FineWeb and Medusa default to Hugging Face datasets.
- GSM8K defaults to `whynlp/gsm8k-aug`.
- Pathfinding, ProsQA, and variable assignment expect repo-local data under `data/` by default. Override `TRAIN_DATA_PATH` and `EVAL_DATA_PATH` as needed.
- S5 retrieval expects generated JSONL files under `data/<dataset_tag>/`. Generate them with:

```bash
python scripts/s5_retrieval/data_generation/make_s5_retrieval_dataset.py --help
```

Generate pathfinding data with:

```bash
python scripts/pathfinding/make_pathfinding_dataset.py --help
```

Generate variable-assignment data with:

```bash
python scripts/variable_assignment/make_variable_assignment_dataset.py --help
```

## Usage

Run a task directly:

```bash
bash scripts/gsm8k/train_gsm8k.sh
```

Run an example sweep:

```bash
bash scripts/gsm8k/submit_gsm8k.sh
```

All train/submit scripts resolve paths relative to the repo root and can be overridden through environment variables.
