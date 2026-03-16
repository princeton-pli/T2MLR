#!/usr/bin/env bash
#SBATCH --job-name=s5-retrieval-t2mlr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00

#SBATCH --output=scripts/s5_retrieval/slurm/s5-retrieval-%j.out
#SBATCH --error=scripts/s5_retrieval/slurm/s5-retrieval-%j.err

set -eo pipefail
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-s5_retrieval}"

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

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-tinyllama}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-s5_char}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
FROM_PRETRAINED="${FROM_PRETRAINED:-False}"

# TinyLlama architecture overrides (~10M params: 6 layers, 6 heads, 384 hidden dim)
TINYLLAMA_HIDDEN_SIZE="${TINYLLAMA_HIDDEN_SIZE:-384}"
TINYLLAMA_NUM_HIDDEN_LAYERS="${TINYLLAMA_NUM_HIDDEN_LAYERS:-6}"
TINYLLAMA_NUM_ATTENTION_HEADS="${TINYLLAMA_NUM_ATTENTION_HEADS:-6}"
TINYLLAMA_NUM_KEY_VALUE_HEADS="${TINYLLAMA_NUM_KEY_VALUE_HEADS:-6}"
TINYLLAMA_INTERMEDIATE_SIZE="${TINYLLAMA_INTERMEDIATE_SIZE:-1536}"

# RNN configuration (~10M params: 6 layers, 456 hidden)
RNN_TYPE="${RNN_TYPE:-lstm}"
RNN_HIDDEN_SIZE="${RNN_HIDDEN_SIZE:-456}"
RNN_NUM_LAYERS="${RNN_NUM_LAYERS:-6}"
RNN_DROPOUT="${RNN_DROPOUT:-0.0}"

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
# Default to linear-biased dataset for training efficiency
DATASET_TAG="${DATASET_TAG:-s5_actions_to_values_retrieval_1_32_1_48_value_len_4_linear}"
DATA_JSON_DIR="$DATA_ROOT/$DATASET_TAG"

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

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-400000}"
BATCH_BACKWARD_APPROXIMATE_DEPTH="${BATCH_BACKWARD_APPROXIMATE_DEPTH:-4}"
PER_DEVICE_TRAIN_BATCH="${PER_DEVICE_TRAIN_BATCH:-16}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-256}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-$((1 / num_gpus))}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
ADAM_BETA2="${ADAM_BETA2:-0.95}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_KW="${LR_SCHEDULER_KW:-{\"min_lr_rate\":0.01}}"
SEED="${SEED:-42}"
BATCH_FORWARD_APPROXIMATE_DEPTH="${BATCH_FORWARD_APPROXIMATE_DEPTH:-16}"
BATCH_FORWARD_TAG="bfad${BATCH_FORWARD_APPROXIMATE_DEPTH}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
TORCH_COMPILE="${TORCH_COMPILE:-False}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-True}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-$(nproc)}"

LOGGING_STEPS="${LOGGING_STEPS:-200}"
LOGGING_STRATEGY="${LOGGING_STRATEGY:-steps}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-10000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
DO_TRAIN="${DO_TRAIN:-True}"
DO_EVAL="${DO_EVAL:-True}"

LOG_GATE_ACTIVITY="${LOG_GATE_ACTIVITY:-True}"
CONTROL_FLOW_ALL_RECURRENT="${CONTROL_FLOW_ALL_RECURRENT:-False}"

DEFAULT_NEW_TOKENS="${DEFAULT_NEW_TOKENS:-64}"
TARGET_LENGTH_BUFFER="${TARGET_LENGTH_BUFFER:-8}"
PASS_AT_K=(${PASS_AT_K:-1})

MAX_LENGTH="${MAX_LENGTH:-1024}"
PADDING_FREE="${PADDING_FREE:-False}"
PADDING_FREE_RETURN_FLASH_ATTN_KWARGS="${PADDING_FREE_RETURN_FLASH_ATTN_KWARGS:-False}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATA_JSON_DIR/train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATA_JSON_DIR/test.jsonl}"

DISABLE_POSITIONAL_ENCODING="${DISABLE_POSITIONAL_ENCODING:-True}"

# ============================================================================
# T2MLR FIXED PARAMETERS
# ============================================================================

CONNECTION_DETACH="${CONNECTION_DETACH:-False}"
T2MLR_ENABLED="${T2MLR_ENABLED:-True}"
T2MLR_MIXING_MODULE_NAME="${T2MLR_MIXING_MODULE_NAME:-gated}"
GATE_PROJ_TYPE="${GATE_PROJ_TYPE:-linear}"
GATE_MLP_HIDDEN_DIM="${GATE_MLP_HIDDEN_DIM:-}"
GATE_MLP_NUM_LAYERS="${GATE_MLP_NUM_LAYERS:-2}"
GATE_MLP_ACTIVATION="${GATE_MLP_ACTIVATION:-gelu}"
GATE_MLP_DROPOUT="${GATE_MLP_DROPOUT:-0.0}"
GATE_WEIGHT_INIT_STD="${GATE_WEIGHT_INIT_STD:-1.0}"
RECURRENT_GATE_INIT="${RECURRENT_GATE_INIT:-}"
INPUT_GATE_INIT="${INPUT_GATE_INIT:-}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE:-True}"
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

T2MLR_TAG=$([[ "$T2MLR_ENABLED" == "True" ]] && echo "t2mlr_on" || echo "t2mlr_off")
RUN_NAME_BASE_DEFAULT="s5_retrieval"
RUN_NAME_BASE="${RUN_NAME_BASE:-$RUN_NAME_BASE_DEFAULT}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
RUN_NAME="$RUN_NAME_BASE"
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
    "${LAUNCHER[@]}" "$REPO_ROOT/src/train.py"
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
    --include_inputs_for_metrics True
    --eval_strategy "$EVAL_STRATEGY"
    --eval_steps "$EVAL_STEPS"
    --eval_print_examples True
    --eval_print_examples_count 2
    --eval_print_max_positions 256
    --save_strategy "$SAVE_STRATEGY"
    --save_steps "$SAVE_STEPS"
    --seed "$SEED"
    --bf16 True
    --project_name t2mlr_s5_retrieval
    --disable_tqdm False
    --save_only_model False
    --concat_response_to_input False
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
    --use_learnable_gate "$USE_GATE"
    --gate_proj_type "$GATE_PROJ_TYPE"
    --gate_mlp_num_layers "$GATE_MLP_NUM_LAYERS"
    --gate_mlp_activation "$GATE_MLP_ACTIVATION"
    --gate_mlp_dropout "$GATE_MLP_DROPOUT"
    --log_gate_activity "$LOG_GATE_ACTIVITY"
    --control_flow_all_recurrent "$CONTROL_FLOW_ALL_RECURRENT"
    --recurrent_residual_to_recurrent_cache "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE"
    --recurrent_residual_to_recurrent_cache_weight "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT"
    --recurrent_residual_to_recurrent_cache_post_norm "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM"
    --recurrent_residual_to_recurrent_cache_detach "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH"
    --gate_weight_init_std "$GATE_WEIGHT_INIT_STD"
    --prompt_column input
    --response_column target
    --reward_mode exact
    --freeze_base_model "$FREEZE_OPTION"
    --default_new_tokens "$DEFAULT_NEW_TOKENS"
    --target_length_buffer "$TARGET_LENGTH_BUFFER"
    --do_sample False
    --train_data_path "$TRAIN_DATA_PATH"
    --eval_data_path "$EVAL_DATA_PATH"
    --capture_gate_trace "$USE_GATE"
    --disable_positional_encoding "$DISABLE_POSITIONAL_ENCODING"
    --label_shift 0
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
fi

# RNN configuration
if [[ "$model_key" == "rnnlm" || "$model_key" == "lstm" || "$model_key" == "gru" || "$model_key" == "rnn" ]]; then
    PYTHON_COMMAND+=(--rnn_type "$RNN_TYPE")
    PYTHON_COMMAND+=(--rnn_hidden_size "$RNN_HIDDEN_SIZE")
    PYTHON_COMMAND+=(--rnn_num_layers "$RNN_NUM_LAYERS")
    PYTHON_COMMAND+=(--rnn_dropout "$RNN_DROPOUT")
    # RNN doesn't support T2MLR
    PYTHON_COMMAND=($(printf '%s\n' "${PYTHON_COMMAND[@]}" | grep -v "^--t2mlr_enabled" | grep -v "^--recurrent_" | grep -v "^--use_recurrent_" | grep -v "^--l_start" | grep -v "^--l_end" | grep -v "^--use_learnable_gate" | grep -v "^--gate_" | grep -v "^--log_gate_activity" | grep -v "^--control_flow_all_recurrent" | grep -v "^--connection_detach" | grep -v "^--capture_gate_trace" | grep -v "^--batch_forward" | grep -v "^--batch_backward"))
    PYTHON_COMMAND+=(--t2mlr_enabled False)
    PYTHON_COMMAND+=(--batch_forward False)
    PYTHON_COMMAND+=(--torch_compile False)
fi

[[ -n "${MIXING_MODULE_KWARGS:-}" ]] && PYTHON_COMMAND+=(--mixing_module_kwargs "$MIXING_MODULE_KWARGS")
[[ -n "$GATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--gate_mlp_hidden_dim "$GATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_STATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--recurrent_state_mlp_hidden_dim "$RECURRENT_STATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_GATE_INIT" ]] && PYTHON_COMMAND+=(--recurrent_gate_init "$RECURRENT_GATE_INIT")
[[ -n "$INPUT_GATE_INIT" ]] && PYTHON_COMMAND+=(--input_gate_init "$INPUT_GATE_INIT")
[[ -n "${GATE_LR_MULTIPLIER:-}" ]] && PYTHON_COMMAND+=(--gate_lr_multiplier "$GATE_LR_MULTIPLIER")
[[ "$PROJECTION_BOOL" == "True" && "$PROJECTION_DIM_CHOICE" != "auto" ]] && PYTHON_COMMAND+=(--recurrent_projection_dim "$PROJECTION_DIM_CHOICE")

echo "============================================================================"
echo "[INFO] S5 Retrieval Training"
echo "============================================================================"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Model: $MODEL_NAME_OR_PATH"
echo "[INFO] T2MLR: $T2MLR_ENABLED (l_start=$L_START, l_end=$L_END)"
echo "[INFO] Output: $OUTPUT_DIR"
echo "============================================================================"

"${PYTHON_COMMAND[@]}"

echo "[INFO] Finished: $RUN_NAME"
