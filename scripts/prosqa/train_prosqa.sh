#!/usr/bin/env bash
#SBATCH --job-name=prosqa-rcot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --gres=gpu:2
#SBATCH --time=7:59:00
#SBATCH --partition=pli-c
#SBATCH --output=scripts/prosqa/slurm/prosqa-rcot-%j.out
#SBATCH --error=scripts/prosqa/slurm/prosqa-rcot-%j.err

set -euo pipefail
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-prosqa_rcot}"

# ============================================================================
# PATHS AND SETUP
# ============================================================================

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
VENV_ROOT="${VENV_ROOT:-$REPO_ROOT/.venv}"

if type module >/dev/null 2>&1; then
    module load proxy/default || true
fi

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$VENV_ROOT/bin/python}"
[[ ! -x "$PYTHON_BIN" ]] && PYTHON_BIN="python"

# ============================================================================
# MODEL AND DATASET CONFIGURATION
# ============================================================================

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolLM2-135M}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-HuggingFaceTB/SmolLM2-135M}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
FROM_PRETRAINED="${FROM_PRETRAINED:-True}"

# # TinyLlama defaults (6 layers, 6 heads, 384 hidden dim, RoPE on)
# MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-tinyllama}"
# TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
# TINYLLAMA_HIDDEN_SIZE="${TINYLLAMA_HIDDEN_SIZE:-384}"
# TINYLLAMA_NUM_HIDDEN_LAYERS="${TINYLLAMA_NUM_HIDDEN_LAYERS:-6}"
# TINYLLAMA_NUM_ATTENTION_HEADS="${TINYLLAMA_NUM_ATTENTION_HEADS:-6}"
# TINYLLAMA_NUM_KEY_VALUE_HEADS="${TINYLLAMA_NUM_KEY_VALUE_HEADS:-6}"
# TINYLLAMA_INTERMEDIATE_SIZE="${TINYLLAMA_INTERMEDIATE_SIZE:-1536}"
# DISABLE_POSITIONAL_ENCODING="${DISABLE_POSITIONAL_ENCODING:-False}"

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
DATASET_SLUG="${DATASET_SLUG:-prosqa}"
MODEL_BASE_SLUG="${MODEL_NAME_OR_PATH##*/}"
MODEL_SLUG="${MODEL_SLUG:-$MODEL_BASE_SLUG}"
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
# TRAINING HYPERPARAMETERS
# ============================================================================

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.6}"
MAX_STEPS="${MAX_STEPS:--1}"
BATCH_BACKWARD_APPROXIMATE_DEPTH="${BATCH_BACKWARD_APPROXIMATE_DEPTH:-4}"
PER_DEVICE_TRAIN_BATCH="${PER_DEVICE_TRAIN_BATCH:-16}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-32}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-$((8 / num_gpus))}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
ADAM_BETA2="${ADAM_BETA2:-0.99}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_KW="${LR_SCHEDULER_KW:-{\"min_lr_rate\":0.01}}"
SEED="${SEED:-42}"
BATCH_FORWARD_APPROXIMATE_DEPTH="${BATCH_FORWARD_APPROXIMATE_DEPTH:-8}"
BATCH_FORWARD_TAG="bfad${BATCH_FORWARD_APPROXIMATE_DEPTH}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
TORCH_COMPILE="${TORCH_COMPILE:-False}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-True}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-$(nproc)}"

LOGGING_STEPS="${LOGGING_STEPS:-50}"
LOGGING_STRATEGY="${LOGGING_STRATEGY:-steps}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-$MAX_STEPS}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DO_TRAIN="${DO_TRAIN:-True}"
DO_EVAL="${DO_EVAL:-True}"

LOG_GATE_ACTIVITY="${LOG_GATE_ACTIVITY:-True}"
CONTROL_FLOW_ALL_RECURRENT="${CONTROL_FLOW_ALL_RECURRENT:-False}"
CONTROL_FLOW_SPLIT_ANSWER="${CONTROL_FLOW_SPLIT_ANSWER:-False}"

DEFAULT_NEW_TOKENS="${DEFAULT_NEW_TOKENS:-64}"
TARGET_LENGTH_BUFFER="${TARGET_LENGTH_BUFFER:-128}"
PASS_AT_K=(${PASS_AT_K:-1})

MAX_LENGTH="${MAX_LENGTH:-2048}"
PADDING_FREE="${PADDING_FREE:-True}"
PADDING_FREE_RETURN_FLASH_ATTN_KWARGS="${PADDING_FREE_RETURN_FLASH_ATTN_KWARGS:-True}"
LABEL_MASK_PROMPT="${LABEL_MASK_PROMPT:-False}"

INSERT_PAUSE_TOKENS="${INSERT_PAUSE_TOKENS:-False}"
PAUSE_TOKEN_MEAN="${PAUSE_TOKEN_MEAN:-0.0}"
PAUSE_TOKEN_SEED="${PAUSE_TOKEN_SEED:-42}"
PAUSE_TOKEN_ONLY_RECURRENT="${PAUSE_TOKEN_ONLY_RECURRENT:-True}"
PAUSE_TOKEN_STRING="${PAUSE_TOKEN_STRING:-}"
PAUSE_TOKEN_REPLACE_PROB="${PAUSE_TOKEN_REPLACE_PROB:-}"
PAUSE_TOKEN_REPLACE_ONLY_RECURRENT="${PAUSE_TOKEN_REPLACE_ONLY_RECURRENT:-True}"
PAUSE_TOKEN_REPLACE_PROB_END="${PAUSE_TOKEN_REPLACE_PROB_END:-}"
PAUSE_TOKEN_REPLACE_PROB_SCHEDULE="${PAUSE_TOKEN_REPLACE_PROB_SCHEDULE:-none}"
PAUSE_TOKEN_REPLACE_PROB_WARMUP_STEPS="${PAUSE_TOKEN_REPLACE_PROB_WARMUP_STEPS:-}"
PAUSE_TOKEN_REPLACE_PROB_WARMUP_RATIO="${PAUSE_TOKEN_REPLACE_PROB_WARMUP_RATIO:-}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATA_ROOT/prosqa/prosqa_hard_train.json}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATA_ROOT/prosqa/prosqa_hard_test.json}"

# ============================================================================
# RCOT FIXED PARAMETERS
# ============================================================================

CONNECTION_DETACH="${CONNECTION_DETACH:-False}"
RCOT_ENABLED="${RCOT_ENABLED:-True}"
RCOT_MIXING_MODULE_NAME="${RCOT_MIXING_MODULE_NAME:-gated}"
GATE_PROJ_TYPE="${GATE_PROJ_TYPE:-linear}"
GATE_MLP_HIDDEN_DIM="${GATE_MLP_HIDDEN_DIM:-}"
GATE_MLP_NUM_LAYERS="${GATE_MLP_NUM_LAYERS:-2}"
GATE_MLP_ACTIVATION="${GATE_MLP_ACTIVATION:-gelu}"
GATE_MLP_DROPOUT="${GATE_MLP_DROPOUT:-0.0}"
GATE_WEIGHT_INIT_STD="${GATE_WEIGHT_INIT_STD:-1e-2}"
RECURRENT_GATE_INIT="${RECURRENT_GATE_INIT:-}"
INPUT_GATE_INIT="${INPUT_GATE_INIT:-}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE:-False}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT:-1.0}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM:-True}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH:-True}"
RECURRENT_STATE_PROJ_TYPE="${RECURRENT_STATE_PROJ_TYPE:-auto}"
RECURRENT_STATE_MLP_HIDDEN_DIM="${RECURRENT_STATE_MLP_HIDDEN_DIM:-}"
RECURRENT_STATE_MLP_NUM_LAYERS="${RECURRENT_STATE_MLP_NUM_LAYERS:-2}"
RECURRENT_STATE_MLP_ACTIVATION="${RECURRENT_STATE_MLP_ACTIVATION:-gelu}"
RECURRENT_STATE_MLP_DROPOUT="${RECURRENT_STATE_MLP_DROPOUT:-0.0}"

USE_PROJECTION="${USE_PROJECTION:-on}"
PROJECTION_DIM_CHOICE="${PROJECTION_DIM_CHOICE:-auto}"
USE_GATE="${USE_GATE:-True}"
L_START="${L_START:-1}"
RECURRENT_WEIGHT="${RECURRENT_WEIGHT:-0.5}"
ORIG_WEIGHT="${ORIG_WEIGHT:-0.5}"
FREEZE_OPTION="${FREEZE_OPTION:-False}"

# ============================================================================
# BUILD RUN CONFIGURATION
# ============================================================================

PROJECTION_BOOL=$([[ "$USE_PROJECTION" == "on" ]] && echo "True" || echo "False")
PROJECTION_TAG=$([[ "$PROJECTION_BOOL" == "True" ]] && { [[ "$PROJECTION_DIM_CHOICE" == "auto" ]] && echo "proj_on_auto" || echo "proj_on_d${PROJECTION_DIM_CHOICE}"; } || echo "proj_off")
L_END=$((-L_START - 1))
WINDOW_TAG="l${L_START}_to_${L_END}"

RCOT_TAG=$([[ "$RCOT_ENABLED" == "True" ]] && echo "rcot_on" || echo "rcot_off")
# RUN_NAME_BASE_DEFAULT="${MODEL_SLUG}_${DATASET_SLUG}_${BATCH_FORWARD_TAG}_${RCOT_TAG}_${WINDOW_TAG}_${PROJECTION_TAG}"
RUN_NAME_BASE_DEFAULT="prosqa_hard_rcot"
RUN_NAME_BASE="${RUN_NAME_BASE:-$RUN_NAME_BASE_DEFAULT}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
RUN_NAME="$RUN_NAME_BASE"
if [[ -n "$RUN_NAME_SUFFIX" ]]; then
    RUN_NAME="${RUN_NAME}_${RUN_NAME_SUFFIX}"
fi
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME"

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# OPTIONAL RCOT INITIALIZATION DEFAULTS
# ============================================================================

if [[ -z "$RECURRENT_GATE_INIT" ]]; then
    RECURRENT_GATE_INIT="$RECURRENT_WEIGHT"
fi
if [[ -z "$INPUT_GATE_INIT" ]]; then
    INPUT_GATE_INIT="$ORIG_WEIGHT"
fi

# ============================================================================
# TRAINING COMMAND
# ============================================================================

if [ "$num_gpus" -le 1 ]; then
    LAUNCHER=("$PYTHON_BIN")
else
    LAUNCHER=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$num_gpus" --master_port=${master_port})
fi

PYTHON_COMMAND=(
    "${LAUNCHER[@]}" "$REPO_ROOT/src/train_minimal.py"
    --do_train "$DO_TRAIN"
    --do_eval "$DO_EVAL"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
    --from_pretrained "$FROM_PRETRAINED"
    --attn_impl "$ATTN_IMPL"
    --output_dir "$OUTPUT_DIR"
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-False}"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --run_name "$RUN_NAME"
    --batch_forward True
    --batch_forward_approximate_depth "$BATCH_FORWARD_APPROXIMATE_DEPTH"
    --batch_backward_approximate_depth "$BATCH_BACKWARD_APPROXIMATE_DEPTH"
    --torch_compile "$TORCH_COMPILE"
    --use_liger_kernel "$USE_LIGER_KERNEL"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --max_steps "$MAX_STEPS"
    --max_length "$MAX_LENGTH"
    --padding_free "$PADDING_FREE"
    --padding_free_return_flash_attn_kwargs "$PADDING_FREE_RETURN_FLASH_ATTN_KWARGS"
    --label_mask_prompt "$LABEL_MASK_PROMPT"
    --group_by_length "$GROUP_BY_LENGTH"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --dataset_num_proc "$DATASET_NUM_PROC"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --adam_beta2 "$ADAM_BETA2"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "cosine_with_min_lr"
    --lr_scheduler_kwargs "$LR_SCHEDULER_KW"
    --logging_strategy "$LOGGING_STRATEGY"
    --logging_steps "$LOGGING_STEPS"
    --eval_strategy "$EVAL_STRATEGY"
    --eval_steps "$EVAL_STEPS"
    --save_strategy "$SAVE_STRATEGY"
    --save_steps "$SAVE_STEPS"
    --seed "$SEED"
    --bf16 True
    --project_name rcot_prosqa
    --disable_tqdm False
    --save_only_model True
    --rcot_enabled "$RCOT_ENABLED"
    --recurrent_mixing_module_name "$RCOT_MIXING_MODULE_NAME"
    --l_start "$L_START"
    --l_end "$L_END"
    --recurrent_weight "$RECURRENT_WEIGHT"
    --orig_weight "$ORIG_WEIGHT"
    --connection_detach "$CONNECTION_DETACH"
    --use_recurrent_projection "$PROJECTION_BOOL"
    --recurrent_state_proj_type "$RECURRENT_STATE_PROJ_TYPE"
    --recurrent_state_mlp_num_layers "$RECURRENT_STATE_MLP_NUM_LAYERS"
    --recurrent_state_mlp_activation "$RECURRENT_STATE_MLP_ACTIVATION"
    --recurrent_state_mlp_dropout "$RECURRENT_STATE_MLP_DROPOUT"
    --use_learnable_gate "$USE_GATE"
    --gate_proj_type "$GATE_PROJ_TYPE"
    --gate_mlp_num_layers "$GATE_MLP_NUM_LAYERS"
    --gate_mlp_activation "$GATE_MLP_ACTIVATION"
    --gate_mlp_dropout "$GATE_MLP_DROPOUT"
    --log_gate_activity "$LOG_GATE_ACTIVITY"
    --control_flow_all_recurrent "$CONTROL_FLOW_ALL_RECURRENT"
    --control_flow_split_answer "$CONTROL_FLOW_SPLIT_ANSWER"
    --recurrent_residual_to_recurrent_cache "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE"
    --recurrent_residual_to_recurrent_cache_weight "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT"
    --recurrent_residual_to_recurrent_cache_post_norm "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM"
    --recurrent_residual_to_recurrent_cache_detach "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH"
    --gate_weight_init_std "$GATE_WEIGHT_INIT_STD"
    --prompt_column question
    --response_column response
    --reward_mode prosqa_path
    --freeze_base_model "$FREEZE_OPTION"
    --default_new_tokens "$DEFAULT_NEW_TOKENS"
    --target_length_buffer "$TARGET_LENGTH_BUFFER"
    --do_sample False
    --train_data_path "$TRAIN_DATA_PATH"
    --eval_data_path "$EVAL_DATA_PATH"
    --eval_dataset_split train
    --custom_dataset_preprocessing gsm8k_aug
    --capture_gate_trace "$USE_GATE"
    --insert_pause_tokens "$INSERT_PAUSE_TOKENS"
    --pause_token_mean "$PAUSE_TOKEN_MEAN"
    --pause_token_seed "$PAUSE_TOKEN_SEED"
    --pause_token_only_recurrent "$PAUSE_TOKEN_ONLY_RECURRENT"
)

PYTHON_COMMAND+=(--pass_at_k "${PASS_AT_K[@]}")

# TinyLlama architecture overrides
model_key="${MODEL_NAME_OR_PATH,,}"
if [[ "$model_key" == "tinyllama" ]]; then
    [[ -n "$TINYLLAMA_HIDDEN_SIZE" ]] && PYTHON_COMMAND+=(--tinyllama_hidden_size "$TINYLLAMA_HIDDEN_SIZE")
    [[ -n "$TINYLLAMA_NUM_HIDDEN_LAYERS" ]] && PYTHON_COMMAND+=(--tinyllama_num_hidden_layers "$TINYLLAMA_NUM_HIDDEN_LAYERS")
    [[ -n "$TINYLLAMA_NUM_ATTENTION_HEADS" ]] && PYTHON_COMMAND+=(--tinyllama_num_attention_heads "$TINYLLAMA_NUM_ATTENTION_HEADS")
    [[ -n "$TINYLLAMA_NUM_KEY_VALUE_HEADS" ]] && PYTHON_COMMAND+=(--tinyllama_num_key_value_heads "$TINYLLAMA_NUM_KEY_VALUE_HEADS")
    [[ -n "$TINYLLAMA_INTERMEDIATE_SIZE" ]] && PYTHON_COMMAND+=(--tinyllama_intermediate_size "$TINYLLAMA_INTERMEDIATE_SIZE")
    [[ -n "$DISABLE_POSITIONAL_ENCODING" ]] && PYTHON_COMMAND+=(--disable_positional_encoding "$DISABLE_POSITIONAL_ENCODING")
fi

[[ -n "${MIXING_MODULE_KWARGS-}" ]] && PYTHON_COMMAND+=(--mixing_module_kwargs "$MIXING_MODULE_KWARGS")
[[ -n "$GATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--gate_mlp_hidden_dim "$GATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_STATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--recurrent_state_mlp_hidden_dim "$RECURRENT_STATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_GATE_INIT" ]] && PYTHON_COMMAND+=(--recurrent_gate_init "$RECURRENT_GATE_INIT")
[[ -n "$INPUT_GATE_INIT" ]] && PYTHON_COMMAND+=(--input_gate_init "$INPUT_GATE_INIT")
[[ -n "${GATE_LR_MULTIPLIER-}" ]] && PYTHON_COMMAND+=(--gate_lr_multiplier "$GATE_LR_MULTIPLIER")
[[ "$PROJECTION_BOOL" == "True" && "$PROJECTION_DIM_CHOICE" != "auto" ]] && PYTHON_COMMAND+=(--recurrent_projection_dim "$PROJECTION_DIM_CHOICE")
[[ -n "$PAUSE_TOKEN_REPLACE_PROB" ]] && PYTHON_COMMAND+=(--pause_token_replace_prob "$PAUSE_TOKEN_REPLACE_PROB")
[[ -n "$PAUSE_TOKEN_REPLACE_ONLY_RECURRENT" ]] && PYTHON_COMMAND+=(--pause_token_replace_only_recurrent "$PAUSE_TOKEN_REPLACE_ONLY_RECURRENT")
[[ -n "$PAUSE_TOKEN_STRING" ]] && PYTHON_COMMAND+=(--pause_token_string "$PAUSE_TOKEN_STRING")
[[ -n "$PAUSE_TOKEN_REPLACE_PROB_END" ]] && PYTHON_COMMAND+=(--pause_token_replace_prob_end "$PAUSE_TOKEN_REPLACE_PROB_END")
[[ -n "$PAUSE_TOKEN_REPLACE_PROB_SCHEDULE" ]] && PYTHON_COMMAND+=(--pause_token_replace_prob_schedule "$PAUSE_TOKEN_REPLACE_PROB_SCHEDULE")
[[ -n "$PAUSE_TOKEN_REPLACE_PROB_WARMUP_STEPS" ]] && PYTHON_COMMAND+=(--pause_token_replace_prob_warmup_steps "$PAUSE_TOKEN_REPLACE_PROB_WARMUP_STEPS")
[[ -n "$PAUSE_TOKEN_REPLACE_PROB_WARMUP_RATIO" ]] && PYTHON_COMMAND+=(--pause_token_replace_prob_warmup_ratio "$PAUSE_TOKEN_REPLACE_PROB_WARMUP_RATIO")

echo "============================================================================"
echo "[INFO] ProsQA RCOT Training"
echo "============================================================================"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Model: $MODEL_NAME_OR_PATH"
echo "[INFO] RCOT: $RCOT_ENABLED (l_start=$L_START, l_end=$L_END)"
echo "[INFO] Output: $OUTPUT_DIR"
echo "============================================================================"

"${PYTHON_COMMAND[@]}"

echo "[INFO] Finished: $RUN_NAME"
