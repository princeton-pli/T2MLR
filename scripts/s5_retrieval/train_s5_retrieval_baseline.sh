#!/usr/bin/env bash

# Baseline (T2MLR disabled) training for S5 retrieval.
# Prompt: dict prefix + actions; Target: values only (one char per action/state).

set -euo pipefail
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# -----------------------------
# Configuration
# -----------------------------
T2MLR_ENABLED=False
MODEL_NAME_OR_PATH="tinyllama"
TOKENIZER_NAME_OR_PATH="s5_char"
OUTPUT_BASE="$REPO_ROOT/outputs"
DATASET_TAG="s5_actions_to_values_retrieval_8_16_1_32_value_len_4"
DATA_JSON_DIR="$REPO_ROOT/data/$DATASET_TAG"

TRAIN_DATA_JSON="$DATA_JSON_DIR/train.jsonl"
VAL_DATA_JSON="$DATA_JSON_DIR/test.jsonl"

NUM_TRAIN_EPOCHS=1  # Ignored because we set max_steps
MAX_STEPS=200000
PER_DEVICE_TRAIN_BATCH=64
PER_DEVICE_EVAL_BATCH=256
GRADIENT_ACCUMULATION=1
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.06
SEED=42
EVAL_STRATEGY="steps"
EVAL_STEPS=4000

DEFAULT_NEW_TOKENS=64
TARGET_LENGTH_BUFFER=8

# -----------------------------
# Derived configuration
# -----------------------------
MODEL_SLUG="${MODEL_NAME_OR_PATH##*/}"
DATASET_SLUG="${DATA_JSON_DIR##*/}"

if [ ! -f "$TRAIN_DATA_JSON" ]; then
    echo "[ERROR] Expected training JSONL not found at $TRAIN_DATA_JSON" >&2
    exit 1
fi

if [ ! -f "$VAL_DATA_JSON" ]; then
    echo "[WARN] Validation JSONL not found at $VAL_DATA_JSON; using training split for eval" >&2
    VAL_DATA_JSON="$TRAIN_DATA_JSON"
fi

TRAIN_DATA_PATH="$TRAIN_DATA_JSON"
EVAL_DATA_PATH="$VAL_DATA_JSON"
if [ ! -f "$EVAL_DATA_PATH" ]; then
    echo "[WARN] validation split missing; using train split for evaluation" >&2
    EVAL_DATA_PATH="$TRAIN_DATA_PATH"
fi

T2MLR_TAG="t2mlr_off"
RUN_NAME="${MODEL_SLUG}_${DATASET_SLUG}_${T2MLR_TAG}"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME"
LOG_FILE="$OUTPUT_DIR/train.log"

mkdir -p "$OUTPUT_DIR"
: > "$LOG_FILE"

echo "[INFO] Starting S5 retrieval baseline training run: $RUN_NAME"

PYTHONUNBUFFERED=1 stdbuf -oL -eL python "$REPO_ROOT/src/train.py" \
    --do_train True \
    --do_eval True \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --disable_positional_encoding False \
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
    --attn_impl "flash_attention_2" \
    --from_pretrained False \
    --train_data_path "$TRAIN_DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --resume_from_checkpoint False \
    --save_total_limit 1 \
    --run_name "$RUN_NAME" \
    --batch_forward=False \
    --batch_forward_approximate_depth=1 \
    --torch_compile=True \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --group_by_length False \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta2 0.95 \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "warmup_stable_decay" \
    --lr_scheduler_kwargs '{"num_decay_steps":4000}' \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --include_inputs_for_metrics True \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --eval_print_examples True \
    --eval_print_examples_count 2 \
    --eval_print_max_positions 64 \
    --save_strategy "epoch" \
    --seed "$SEED" \
    --bf16 True \
    --project_name "t2mlr_s5_retrieval" \
    --disable_tqdm False \
    --save_only_model True \
    --concat_response_to_input False \
    --label_shift 0 \
    --t2mlr_enabled $T2MLR_ENABLED \
    --prompt_column "input" \
    --response_column "target" \
    --reward_mode "exact" \
    --default_new_tokens "$DEFAULT_NEW_TOKENS" \
    --target_length_buffer "$TARGET_LENGTH_BUFFER" \
    --do_sample False \
    | tee "$LOG_FILE"

echo "[INFO] Baseline training complete. Outputs located in $OUTPUT_DIR (logs -> $LOG_FILE)"

