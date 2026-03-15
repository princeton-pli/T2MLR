#!/usr/bin/env bash
#SBATCH --job-name=medusa-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=pli
#SBATCH --partition=pli-c
##SBATCH --qos=pli-cp
#SBATCH --output=scripts/medusa/slurm/medusa-train-%j.out
#SBATCH --error=scripts/medusa/slurm/medusa-train-%j.err

set -euo pipefail
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ============================================================================
# PATHS AND SETUP
# ============================================================================

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
VENV_ROOT="${VENV_ROOT:-$REPO_ROOT/.venv}"

if type module >/dev/null 2>&1; then
    module load proxy/default || true
fi

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$VENV_ROOT/bin/python}"
[[ ! -x "$PYTHON_BIN" ]] && PYTHON_BIN="python"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolLM2-360M}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$MODEL_NAME_OR_PATH}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

RCOT_ENABLED="${RCOT_ENABLED:-False}"

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs/medusa}"
DATASET_SLUG="${DATASET_SLUG:-fineweb-edu}"
MODEL_SLUG="${MODEL_SLUG:-SmolLM2-360M}"
if [[ -z "${num_gpus:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        num_gpus=1
    fi
fi
echo "[INFO] num_gpus: $num_gpus"

master_port=${master_port:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}

# ============================================================================
# MEDUSA HEAD CONFIGURATION
# ============================================================================

NUM_MEDUSA_HEADS="${NUM_MEDUSA_HEADS:-4}"
MEDUSA_HEAD_HIDDEN_DIM="${MEDUSA_HEAD_HIDDEN_DIM:-}"  # Empty = use model hidden size
MEDUSA_HEAD_NUM_LAYERS="${MEDUSA_HEAD_NUM_LAYERS:-2}"  # 1 = linear, >1 = MLP
USE_RESIDUAL_CONNECTION="${USE_RESIDUAL_CONNECTION:-True}"
HIDDEN_LAYER_INDEX="${HIDDEN_LAYER_INDEX:--1}"  # -1 = last layer before LM head

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
TRAIN_DATASET_CONFIG="${TRAIN_DATASET_CONFIG:-sample-10BT}"
TRAIN_DATASET_SPLIT="${TRAIN_DATASET_SPLIT:-train}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-$TRAIN_DATASET_NAME}"
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-$TRAIN_DATASET_CONFIG}"
EVAL_DATASET_SPLIT="${EVAL_DATASET_SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"

train_tokenized_cache="${train_tokenized_cache:-$REPO_ROOT/data_cache/fineweb-edu-smollm2-135m-train-10BT}"
eval_tokenized_cache="${eval_tokenized_cache:-$REPO_ROOT/data_cache/fineweb-edu-smollm2-135m-eval-10BT}"

# Holdout split configuration (creates eval set from train data)
HOLDOUT_SIZE="${HOLDOUT_SIZE:-1000}"
HOLDOUT_RATIO="${HOLDOUT_RATIO:-}"
HOLDOUT_SEED="${HOLDOUT_SEED:-}"

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

MAX_STEPS="${MAX_STEPS:-5000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
# PER_DEVICE_TRAIN_BATCH="${PER_DEVICE_TRAIN_BATCH:-8}"
PER_DEVICE_TRAIN_BATCH="${PER_DEVICE_TRAIN_BATCH:-1}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-8}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-$((TOTAL_BATCH_SIZE / num_gpus / PER_DEVICE_TRAIN_BATCH))}"

printf "\n[Config] TOTAL_BATCH_SIZE:            %s\n" "$TOTAL_BATCH_SIZE"
printf "[Config] PER_DEVICE_TRAIN_BATCH:      %s\n" "$PER_DEVICE_TRAIN_BATCH"
printf "[Config] GRADIENT_ACCUMULATION:       %s\n\n" "$GRADIENT_ACCUMULATION"

LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-300}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
DO_TRAIN="${DO_TRAIN:-True}"
DO_EVAL="${DO_EVAL:-True}"

# Head loss weighting (JSON list, e.g., "[1.0, 0.8, 0.6, 0.4]")
HEAD_LOSS_WEIGHTS="${HEAD_LOSS_WEIGHTS:-}"

# ============================================================================
# BUILD RUN CONFIGURATION
# ============================================================================

RUN_NAME_BASE="${RUN_NAME_BASE:-""}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
HEADS_TAG="heads${NUM_MEDUSA_HEADS}_layers${MEDUSA_HEAD_NUM_LAYERS}"
LAYER_TAG="layer${HIDDEN_LAYER_INDEX}"

if [[ -n "$RUN_NAME_SUFFIX" ]]; then
    RUN_NAME="${RUN_NAME_SUFFIX}"
else
    RUN_NAME="${MODEL_SLUG}_${HEADS_TAG}_${LAYER_TAG}"
fi
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME"

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# HOLDOUT SPLIT CONFIGURATION
# ============================================================================

HOLDOUT_ARGS=()
if [[ -n "$HOLDOUT_RATIO" ]]; then
    HOLDOUT_ARGS+=(--eval_holdout_ratio "$HOLDOUT_RATIO")
elif [[ -n "$HOLDOUT_SIZE" ]]; then
    HOLDOUT_ARGS+=(--eval_holdout_size "$HOLDOUT_SIZE")
fi
if [[ -n "$HOLDOUT_SEED" ]]; then
    HOLDOUT_ARGS+=(--eval_holdout_seed "$HOLDOUT_SEED")
fi

# ============================================================================
# TRAINING COMMAND
# ============================================================================

if [ "$num_gpus" -le 1 ]; then
    LAUNCHER=("$PYTHON_BIN" "src/train_medusa.py")
else
    LAUNCHER=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$num_gpus" --master_port=${master_port} "src/train_medusa.py")
fi

PYTHON_COMMAND=(
    "${LAUNCHER[@]}"
    --do_train "$DO_TRAIN"
    --do_eval "$DO_EVAL"
    # Model args
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
    --attn_impl "$ATTN_IMPL"
    --num_medusa_heads "$NUM_MEDUSA_HEADS"
    --medusa_head_num_layers "$MEDUSA_HEAD_NUM_LAYERS"
    --use_residual_connection "$USE_RESIDUAL_CONNECTION"
    --hidden_layer_index "$HIDDEN_LAYER_INDEX"
    # Data args
    --train_dataset_name "$TRAIN_DATASET_NAME"
    --train_dataset_config "$TRAIN_DATASET_CONFIG"
    --train_dataset_split "$TRAIN_DATASET_SPLIT"
    --eval_dataset_name "$EVAL_DATASET_NAME"
    --eval_dataset_config "$EVAL_DATASET_CONFIG"
    --eval_dataset_split "$EVAL_DATASET_SPLIT"
    --max_length "$MAX_LENGTH"
    --text_column "$TEXT_COLUMN"
    "${HOLDOUT_ARGS[@]}"
    # Training args
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --max_steps "$MAX_STEPS"
    --warmup_ratio "$WARMUP_RATIO"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --lr_scheduler_type "cosine"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --logging_strategy steps
    --logging_steps "$LOGGING_STEPS"
    --eval_strategy "$EVAL_STRATEGY"
    --eval_steps "$EVAL_STEPS"
    --save_strategy steps
    --save_steps "$SAVE_STEPS"
    --save_total_limit 2
    --seed "$SEED"
    --bf16 True
    --project_name medusa_training
    --disable_tqdm True
    --report_to wandb
    --train_tokenized_cache "$train_tokenized_cache"
    --eval_tokenized_cache "$eval_tokenized_cache"
    --rcot_enabled "$RCOT_ENABLED"
)

# # Bool flags must be passed as --flag / --no_flag (HfArgumentParser uses BoolOptionalAction)
# case "${RCOT_ENABLED}" in
#   True|true|1) PYTHON_COMMAND+=(--rcot_enabled) ;;
#   *)           PYTHON_COMMAND+=(--no_rcot_enabled) ;;
# esac

# Optional arguments
[[ -n "$MEDUSA_HEAD_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--medusa_head_hidden_dim "$MEDUSA_HEAD_HIDDEN_DIM")
[[ -n "$HEAD_LOSS_WEIGHTS" ]] && PYTHON_COMMAND+=(--head_loss_weights "$HEAD_LOSS_WEIGHTS")

echo "============================================================================"
echo "[INFO] Medusa Multi-Head Training"
echo "============================================================================"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Model: $MODEL_NAME_OR_PATH"
echo "[INFO] Num Medusa heads: $NUM_MEDUSA_HEADS (predicting t+2 to t+$((NUM_MEDUSA_HEADS+1)))"
echo "[INFO] Head MLP layers: $MEDUSA_HEAD_NUM_LAYERS"
echo "[INFO] Hidden layer index: $HIDDEN_LAYER_INDEX"
echo "[INFO] Output: $OUTPUT_DIR"
echo "============================================================================"

"${PYTHON_COMMAND[@]}"

echo "[INFO] Finished: $RUN_NAME"
