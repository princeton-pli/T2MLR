#!/usr/bin/env bash
#SBATCH --job-name=gsm8k-t2mlr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --gres=gpu:2
#SBATCH --time=5:59:00

#SBATCH --output=scripts/gsm8k/slurm/gsm8k-t2mlr-%j.out
#SBATCH --error=scripts/gsm8k/slurm/gsm8k-t2mlr-%j.err

set -eo pipefail
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
# MODEL AND DATASET CONFIGURATION
# ============================================================================

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$MODEL_NAME_OR_PATH}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
FROM_PRETRAINED="${FROM_PRETRAINED:-True}"

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
DATASET_SLUG="${DATASET_SLUG:-gsm8k_main}"
MODEL_BASE_SLUG="${MODEL_NAME_OR_PATH##*/}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL_BASE_SLUG}_from_pretrained}"
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

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
BATCH_BACKWARD_APPROXIMATE_DEPTH="${BATCH_BACKWARD_APPROXIMATE_DEPTH:-8}"
PER_DEVICE_TRAIN_BATCH="${PER_DEVICE_TRAIN_BATCH:-64}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-128}"
GRADIENT_ACCUMULATION_BASE="${GRADIENT_ACCUMULATION_BASE:-4}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-$((GRADIENT_ACCUMULATION_BASE / num_gpus))}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
ADAM_BETA2="${ADAM_BETA2:-0.99}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_KW="${LR_SCHEDULER_KW:-{\"min_lr_rate\":0.01}}"
SEED="${SEED:-42}"
BATCH_FORWARD_APPROXIMATE_DEPTH="${BATCH_FORWARD_APPROXIMATE_DEPTH:-16}"
BATCH_FORWARD_TAG="bfad${BATCH_FORWARD_APPROXIMATE_DEPTH}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
TORCH_COMPILE="${TORCH_COMPILE:-False}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-True}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-$(nproc)}"

LOGGING_STEPS="${LOGGING_STEPS:-20}"
LOGGING_STRATEGY="${LOGGING_STRATEGY:-steps}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-$MAX_STEPS}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DO_TRAIN="${DO_TRAIN:-True}"
DO_EVAL="${DO_EVAL:-True}"

LOG_GATE_ACTIVITY="${LOG_GATE_ACTIVITY:-True}"
CONTROL_FLOW_ALL_RECURRENT="${CONTROL_FLOW_ALL_RECURRENT:-False}"
LABEL_MASK_PROMPT="${LABEL_MASK_PROMPT:-False}"

DEFAULT_NEW_TOKENS="${DEFAULT_NEW_TOKENS:-512}"
TARGET_LENGTH_BUFFER="${TARGET_LENGTH_BUFFER:-128}"
PASS_AT_K=(${PASS_AT_K:-1})

MAX_LENGTH="${MAX_LENGTH:-4096}"
PADDING_FREE="${PADDING_FREE:-True}"
PADDING_FREE_RETURN_FLASH_ATTN_KWARGS="${PADDING_FREE_RETURN_FLASH_ATTN_KWARGS:-True}"

HF_DATASET_NAME="${HF_DATASET_NAME:-whynlp/gsm8k-aug}"
# HF_DATASET_NAME="${HF_DATASET_NAME:-whynlp/gsm8k-aug-nl}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-default}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# ============================================================================
# T2MLR FIXED PARAMETERS
# ============================================================================

CONNECTION_DETACH="${CONNECTION_DETACH:-False}"
T2MLR_ENABLED="${T2MLR_ENABLED:-True}"
T2MLR_MIXING_MODULE_NAME="${T2MLR_MIXING_MODULE_NAME:-gated}"
GATE_LR_MULTIPLIER="${GATE_LR_MULTIPLIER:-100}"
MIXING_MODULE_KWARGS="${MIXING_MODULE_KWARGS:-}"
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

INSERT_PAUSE_TOKENS="${INSERT_PAUSE_TOKENS:-False}"
PAUSE_TOKEN_MEAN="${PAUSE_TOKEN_MEAN:-1.5}"

# ============================================================================
# T2MLR SWEEP PARAMETERS (from environment, set by sweep script)
# ============================================================================

USE_PROJECTION="${USE_PROJECTION:-on}"
PROJECTION_DIM_CHOICE="${PROJECTION_DIM_CHOICE:-auto}"
USE_GATE="${USE_GATE:-on}"
L_START="${L_START:-4}"
RECURRENT_WEIGHT="${RECURRENT_WEIGHT:-0.5}"
ORIG_WEIGHT="${ORIG_WEIGHT:-0.5}"
FREEZE_OPTION="${FREEZE_OPTION:-off}"

# ============================================================================
# BUILD RUN CONFIGURATION
# ============================================================================

PROJECTION_BOOL=$([[ "$USE_PROJECTION" == "on" ]] && echo "True" || echo "False")
PROJECTION_TAG=$([[ "$PROJECTION_BOOL" == "True" ]] && { [[ "$PROJECTION_DIM_CHOICE" == "auto" ]] && echo "proj_on_auto" || echo "proj_on_d${PROJECTION_DIM_CHOICE}"; } || echo "proj_off")
GATE_BOOL=$([[ "$USE_GATE" == "on" ]] && echo "True" || echo "False")
GATE_TAG=$([[ "$GATE_BOOL" == "True" ]] && echo "gate_on" || echo "gate_off")
WEIGHT_TAG="rw${RECURRENT_WEIGHT//./p}_ow${ORIG_WEIGHT//./p}"
FREEZE_BOOL=$([[ "$FREEZE_OPTION" == "on" ]] && echo "True" || echo "False")
FREEZE_TAG=$([[ "$FREEZE_OPTION" == "on" ]] && echo "freeze_on" || echo "freeze_off")
L_END=$((-L_START - 1))
WINDOW_TAG="l${L_START}_to_${L_END}"
PAUSE_TAG=""
[[ "$INSERT_PAUSE_TOKENS" == "True" ]] && PAUSE_TAG="_pause_on_mean${PAUSE_TOKEN_MEAN//./p}"

# T2MLR_TAG=$([[ "$T2MLR_ENABLED" == "True" ]] && echo "t2mlr_on" || echo "t2mlr_off")
# RUN_NAME_BASE_DEFAULT="${MODEL_SLUG}_${DATASET_SLUG}_${BATCH_FORWARD_TAG}_${T2MLR_TAG}_${WINDOW_TAG}_${PROJECTION_TAG}_${GATE_TAG}_${WEIGHT_TAG}_${FREEZE_TAG}${PAUSE_TAG}"
# RUN_NAME_BASE="${RUN_NAME_BASE:-$RUN_NAME_BASE_DEFAULT}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
RUN_NAME="${RUN_NAME:-gsm8k_t2mlr}"
# RUN_NAME="${RUN_NAME:-gsm8k_nl_t2mlr}"

if [[ -n "$RUN_NAME_SUFFIX" ]]; then
    RUN_NAME="${RUN_NAME}_${RUN_NAME_SUFFIX}"
fi
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME"

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# OPTIONAL T2MLR INITIALIZATION DEFAULTS
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
    --project_name t2mlr_gsm8k
    --disable_tqdm False
    --save_only_model True
    --t2mlr_enabled "$T2MLR_ENABLED"
    --recurrent_mixing_module_name "$T2MLR_MIXING_MODULE_NAME"
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
    --use_learnable_gate "$GATE_BOOL"
    --gate_proj_type "$GATE_PROJ_TYPE"
    --gate_mlp_num_layers "$GATE_MLP_NUM_LAYERS"
    --gate_mlp_activation "$GATE_MLP_ACTIVATION"
    --gate_mlp_dropout "$GATE_MLP_DROPOUT"
    --log_gate_activity "$LOG_GATE_ACTIVITY"
    --control_flow_all_recurrent "$CONTROL_FLOW_ALL_RECURRENT"
    --label_mask_prompt "$LABEL_MASK_PROMPT"
    --recurrent_residual_to_recurrent_cache "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE"
    --recurrent_residual_to_recurrent_cache_weight "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT"
    --recurrent_residual_to_recurrent_cache_post_norm "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM"
    --recurrent_residual_to_recurrent_cache_detach "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH"
    --gate_weight_init_std "$GATE_WEIGHT_INIT_STD"
    --prompt_column question
    --response_column response
    --reward_mode gsm8k
    --freeze_base_model "$FREEZE_BOOL"
    --default_new_tokens "$DEFAULT_NEW_TOKENS"
    --target_length_buffer "$TARGET_LENGTH_BUFFER"
    --do_sample False
    --train_dataset_name "$HF_DATASET_NAME"
    --train_dataset_config "$HF_DATASET_CONFIG"
    --train_dataset_split "$TRAIN_SPLIT"
    --eval_dataset_name "$HF_DATASET_NAME"
    --eval_dataset_config "$HF_DATASET_CONFIG"
    --eval_dataset_split "$EVAL_SPLIT"
    --custom_dataset_preprocessing gsm8k_aug
    --capture_gate_trace "$GATE_BOOL"
    --gate_lr_multiplier "$GATE_LR_MULTIPLIER"
    --insert_pause_tokens "$INSERT_PAUSE_TOKENS"
    --pause_token_mean "$PAUSE_TOKEN_MEAN"
    --num_generations_per_sample 1
)

PYTHON_COMMAND+=(--pass_at_k "${PASS_AT_K[@]}")

[[ -n "$MIXING_MODULE_KWARGS" ]] && PYTHON_COMMAND+=(--mixing_module_kwargs "$MIXING_MODULE_KWARGS")
[[ -n "$GATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--gate_mlp_hidden_dim "$GATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_STATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--recurrent_state_mlp_hidden_dim "$RECURRENT_STATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_GATE_INIT" ]] && PYTHON_COMMAND+=(--recurrent_gate_init "$RECURRENT_GATE_INIT")
[[ -n "$INPUT_GATE_INIT" ]] && PYTHON_COMMAND+=(--input_gate_init "$INPUT_GATE_INIT")
[[ "$PROJECTION_BOOL" == "True" && "$PROJECTION_DIM_CHOICE" != "auto" ]] && PYTHON_COMMAND+=(--recurrent_projection_dim "$PROJECTION_DIM_CHOICE")

echo "============================================================================"
echo "[INFO] GSM8K T2MLR Training"
echo "============================================================================"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Model: $MODEL_NAME_OR_PATH"
echo "[INFO] T2MLR: $T2MLR_ENABLED (l_start=$L_START, l_end=$L_END)"
echo "[INFO] Output: $OUTPUT_DIR"
echo "============================================================================"

"${PYTHON_COMMAND[@]}"

rm -rf "$OUTPUT_DIR/checkpoint-$MAX_STEPS"

echo "[INFO] Finished: $RUN_NAME"
