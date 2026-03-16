#!/usr/bin/env bash

# RNN baseline (T2MLR disabled) training for S5 retrieval.
# Mirrors the transformer baseline launch script, but uses a lightweight LSTM/GRU causal LM.
#
# Prompt: dict prefix + actions
# Target: values only (one char per action/state)
#
# Notes:
# - Uses the same tokenizer + dataset format as the transformer baseline.
# - T2MLR must remain disabled (RNN backbone cannot be wrapped by T2MLR).

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

# Custom HF-style model implemented in src/modeling/rnnlm.py
MODEL_NAME_OR_PATH="rnnlm"   # can also be "gru" / "lstm" / "rnn"
TOKENIZER_NAME_OR_PATH="s5_char"

# RNN architecture knobs (wired via ModelArguments)
# Default to LSTM as the classic RNN LM baseline.
RNN_TYPE="${RNN_TYPE:-lstm}"
# For s5_char (vocab=186), `tinyllama` default is ~4.29M params.
# Closest LSTM match at 4 layers:
# - hidden=363 => 4,295,742 params (closest overall, not multiple-of-8)
# - hidden=360 => 4,225,680 params (~1.5% low, multiple-of-8)
RNN_HIDDEN_SIZE="${RNN_HIDDEN_SIZE:-360}"
RNN_NUM_LAYERS="${RNN_NUM_LAYERS:-4}"
RNN_DROPOUT="${RNN_DROPOUT:-0.0}"

OUTPUT_BASE="$REPO_ROOT/outputs"
DATASET_TAG="s5_actions_to_values_retrieval_8_16_1_32_value_len_4"
DATA_JSON_DIR="$REPO_ROOT/data/$DATASET_TAG"

TRAIN_DATA_JSON="$DATA_JSON_DIR/train.jsonl"
VAL_DATA_JSON="$DATA_JSON_DIR/test.jsonl"

NUM_TRAIN_EPOCHS=1  # Ignored because we set max_steps
MAX_STEPS=400000
PER_DEVICE_TRAIN_BATCH=64
PER_DEVICE_EVAL_BATCH=256
GRADIENT_ACCUMULATION=1
LEARNING_RATE=5e-3
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
MODEL_SLUG="rnnlm_${RNN_TYPE}_h${RNN_HIDDEN_SIZE}_l${RNN_NUM_LAYERS}"
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

echo "[INFO] Starting S5 retrieval RNN baseline training run: $RUN_NAME"

PYTHONUNBUFFERED=1 stdbuf -oL -eL python "$REPO_ROOT/src/train.py" \
    --do_train True \
    --do_eval True \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
    --from_pretrained False \
    --train_data_path "$TRAIN_DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --resume_from_checkpoint False \
    --save_total_limit 1 \
    --run_name "$RUN_NAME" \
    --batch_forward=False \
    --batch_forward_approximate_depth=1 \
    --torch_compile=False \
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
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 50 \
    --include_inputs_for_metrics True \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --eval_print_examples True \
    --eval_print_examples_count 1 \
    --eval_print_max_positions 256 \
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
    --rnn_type "$RNN_TYPE" \
    --rnn_hidden_size "$RNN_HIDDEN_SIZE" \
    --rnn_num_layers "$RNN_NUM_LAYERS" \
    --rnn_dropout "$RNN_DROPOUT" \
    | tee "$LOG_FILE"

echo "[INFO] RNN baseline training complete. Outputs located in $OUTPUT_DIR (logs -> $LOG_FILE)"

