#!/usr/bin/env bash
#SBATCH --job-name=fineweb-t2mlr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

#SBATCH --qos=pli-cp
#SBATCH --output=scripts/fineweb/slurm/fineweb-t2mlr-%j.out

set -euo pipefail
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

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
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolLM2-360M}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-HuggingFaceTB/SmolLM2-360M}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
FROM_PRETRAINED="${FROM_PRETRAINED:-False}"

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
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

FSDP_CONFIG_FILE="scripts/fineweb/fsdp_config.json"
# master_port=${master_port:-$(get_free_port)}
master_port=${master_port:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Using HuggingFace dataset - will be downloaded automatically
# TinyStories defaults (commented out):
# TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-roneneldan/TinyStories}"
# TRAIN_DATASET_SPLIT="${TRAIN_DATASET_SPLIT:-train}"
# EVAL_DATA_PATH="${EVAL_DATA_PATH:-roneneldan/TinyStories}"
# EVAL_DATASET_SPLIT="${EVAL_DATASET_SPLIT:-validation[:10000]}"
# HF_DATASET_CONFIG=${HF_DATASET_CONFIG:-"default"}

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-HuggingFaceFW/fineweb-edu}"
HOLDOUT_SIZE="${HOLDOUT_SIZE:-128}"
HOLDOUT_RATIO="${HOLDOUT_RATIO:-}"
HOLDOUT_SEED="${HOLDOUT_SEED:-}"
TRAIN_DATASET_SPLIT="${TRAIN_DATASET_SPLIT:-train}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-HuggingFaceFW/fineweb-edu}"
EVAL_DATASET_SPLIT="${EVAL_DATASET_SPLIT:-train}"
HF_DATASET_CONFIG="${HF_DATASET_CONFIG:-sample-10BT}"
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-$HF_DATASET_CONFIG}"

TRAIN_TOKENIZED_CACHE="${TRAIN_TOKENIZED_CACHE:-$REPO_ROOT/data_cache/fineweb-edu-smollm2-135m-train-10BT}"
EVAL_TOKENIZED_CACHE="${EVAL_TOKENIZED_CACHE:-$REPO_ROOT/data_cache/fineweb-edu-smollm2-135m-eval-10BT}"

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
MAX_STEPS="${MAX_STEPS:--20000}"
WARMUP_STEPS="${WARMUP_STEPS:-None}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-256}"
PER_DEVICE_TRAIN_BATCH="${PER_DEVICE_TRAIN_BATCH:-8}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-8}"

# Old hyperparameters vs. standard hyperparameters
# GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-$((32 / num_gpus))}"
# LEARNING_RATE="${LEARNING_RATE:-6e-4}"
# ADAM_BETA2="${ADAM_BETA2:-0.98}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-$((TOTAL_BATCH_SIZE / num_gpus / PER_DEVICE_TRAIN_BATCH))}"
printf "\n[Config] TOTAL_BATCH_SIZE:            %s\n" "$TOTAL_BATCH_SIZE"
printf "[Config] PER_DEVICE_TRAIN_BATCH:      %s\n" "$PER_DEVICE_TRAIN_BATCH"
printf "[Config] GRADIENT_ACCUMULATION:       %s\n\n" "$GRADIENT_ACCUMULATION"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_KW="${LR_SCHEDULER_KW:-{\"min_lr_rate\":0.001}}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

BATCH_FORWARD_APPROXIMATE_DEPTH="${BATCH_FORWARD_APPROXIMATE_DEPTH:-16}"
BATCH_BACKWARD_APPROXIMATE_DEPTH="${BATCH_BACKWARD_APPROXIMATE_DEPTH:-4}"
BATCH_FORWARD_TAG="bfad${BATCH_FORWARD_APPROXIMATE_DEPTH}_bbad${BATCH_BACKWARD_APPROXIMATE_DEPTH}"

GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-False}"
DATALOADER_NUM_WORKERS=16
TORCH_COMPILE="${TORCH_COMPILE:-False}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-True}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-$((num_gpus * 8))}"

LOGGING_STEPS="${LOGGING_STEPS:-50}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
DO_TRAIN="${DO_TRAIN:-True}"
DO_EVAL="${DO_EVAL:-True}"

# Generation eval parameters
DEFAULT_NEW_TOKENS="${DEFAULT_NEW_TOKENS:-128}"
TARGET_LENGTH_BUFFER="${TARGET_LENGTH_BUFFER:-16}"
NUM_GENERATIONS_PER_SAMPLE="${NUM_GENERATIONS_PER_SAMPLE:-1}"
PASS_AT_K=(${PASS_AT_K:-1})

# ============================================================================
# T2MLR FIXED PARAMETERS
# ============================================================================

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
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE:-False}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT:-1.0}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM:-True}"
RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH="${RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH:-False}"
RECURRENT_STATE_PROJ_TYPE="${RECURRENT_STATE_PROJ_TYPE:-linear}"
RECURRENT_STATE_MLP_HIDDEN_DIM="${RECURRENT_STATE_MLP_HIDDEN_DIM:-}"
RECURRENT_STATE_MLP_NUM_LAYERS="${RECURRENT_STATE_MLP_NUM_LAYERS:-2}"
RECURRENT_STATE_MLP_ACTIVATION="${RECURRENT_STATE_MLP_ACTIVATION:-gelu}"
RECURRENT_STATE_MLP_DROPOUT="${RECURRENT_STATE_MLP_DROPOUT:-0.0}"

# ============================================================================
# T2MLR SWEEP PARAMETERS (from environment)
# ============================================================================

USE_PROJECTION="${USE_PROJECTION:-on}"
PROJECTION_DIM_CHOICE="${PROJECTION_DIM_CHOICE:-auto}"
USE_GATE="${USE_GATE:-on}"
L_START="${L_START:-8}"  # With 16 layers, reasonable starting point
RECURRENT_WEIGHT="${RECURRENT_WEIGHT:-0.5}"
ORIG_WEIGHT="${ORIG_WEIGHT:-0.5}"

# ============================================================================
# BUILD RUN CONFIGURATION
# ============================================================================

PROJECTION_BOOL=$([[ "$USE_PROJECTION" == "on" ]] && echo "True" || echo "False")
PROJECTION_TAG=$([[ "$PROJECTION_BOOL" == "True" ]] && { [[ "$PROJECTION_DIM_CHOICE" == "auto" ]] && echo "proj_on_auto" || echo "proj_on_d${PROJECTION_DIM_CHOICE}"; } || echo "proj_off")
GATE_BOOL=$([[ "$USE_GATE" == "on" ]] && echo "True" || echo "False")
GATE_TAG=$([[ "$GATE_BOOL" == "True" ]] && echo "gate_on" || echo "gate_off")
WEIGHT_TAG="rw${RECURRENT_WEIGHT//./p}_ow${ORIG_WEIGHT//./p}"
# RUN_NAME_BASE="${RUN_NAME_BASE:-wikitext}"
RUN_NAME_BASE="${RUN_NAME_BASE:-fineweb}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
L_END=$((-L_START - 1))
WINDOW_TAG="l${L_START}_to_${L_END}"
CACHE_RESIDUAL_TAG=$([[ "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE" == "True" ]] && echo "cache_residual_on" || echo "cache_residual_off")

T2MLR_TAG=$([[ "$T2MLR_ENABLED" == "True" ]] && echo "t2mlr_on" || echo "t2mlr_off")
# RUN_NAME="${MODEL_SLUG}_${DATASET_SLUG}_${BATCH_FORWARD_TAG}_${T2MLR_TAG}_${WINDOW_TAG}_${PROJECTION_TAG}_${GATE_TAG}_${WEIGHT_TAG}_bfad${BATCH_FORWARD_APPROXIMATE_DEPTH}_bbad${BATCH_BACKWARD_APPROXIMATE_DEPTH}_${CACHE_RESIDUAL_TAG}"
# if [[ -n "$RUN_NAME_SUFFIX" ]]; then
#     RUN_NAME="${RUN_NAME}_${RUN_NAME_SUFFIX}"
# fi
RUN_NAME="${RUN_NAME_BASE}_${RUN_NAME_SUFFIX}"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_NAME"


mkdir -p "$OUTPUT_DIR"

# ============================================================================
# HOLDOUT SPLIT CONFIGURATION
# ============================================================================

HOLDOUT_ARGS=()
if [[ -n "$HOLDOUT_RATIO" ]]; then
    HOLDOUT_ARGS+=(--eval_holdout_ratio "$HOLDOUT_RATIO")
else
    HOLDOUT_ARGS+=(--eval_holdout_size "$HOLDOUT_SIZE")
fi
if [[ -n "$HOLDOUT_SEED" ]]; then
    HOLDOUT_ARGS+=(--eval_holdout_seed "$HOLDOUT_SEED")
fi

# ============================================================================
# TRAINING COMMAND
# ============================================================================

if [ "$num_gpus" -le 1 ]; then
    LAUNCHER=("$PYTHON_BIN" "src/train.py")
else
    LAUNCHER=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$num_gpus" --master_port=${master_port} "src/train.py")
fi

PYTHON_COMMAND=(
    "${LAUNCHER[@]}"
    --do_train "$DO_TRAIN"
    --do_eval "$DO_EVAL"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH"
    --from_pretrained "$FROM_PRETRAINED"
    --attn_impl "$ATTN_IMPL"
    --train_dataset_name "$TRAIN_DATA_PATH"
    --eval_dataset_name "$EVAL_DATA_PATH"
    --eval_dataset_config "$EVAL_DATASET_CONFIG"
    --eval_dataset_split "$EVAL_DATASET_SPLIT"
    --train_dataset_split "$TRAIN_DATASET_SPLIT"
    --train_tokenized_cache "$TRAIN_TOKENIZED_CACHE"
    --eval_tokenized_cache "$EVAL_TOKENIZED_CACHE"
    "${HOLDOUT_ARGS[@]}"
    --max_length "$MAX_LENGTH"
    --output_dir "$OUTPUT_DIR"
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-False}"
    --save_total_limit 20
    --run_name "$RUN_NAME"
    --batch_forward True
    --batch_forward_approximate_depth "$BATCH_FORWARD_APPROXIMATE_DEPTH"
    --batch_backward_approximate_depth "$BATCH_BACKWARD_APPROXIMATE_DEPTH"
    --torch_compile "$TORCH_COMPILE"
    --use_liger_kernel "$USE_LIGER_KERNEL"
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --max_steps "$MAX_STEPS"
    --warmup_ratio "$WARMUP_RATIO"
    --group_by_length "$GROUP_BY_LENGTH"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --dataset_num_proc "$DATASET_NUM_PROC"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --lr_scheduler_type "cosine_with_min_lr"
    --lr_scheduler_kwargs "$LR_SCHEDULER_KW"
    --adam_beta2 "$ADAM_BETA2"
    --logging_strategy steps
    --logging_steps "$LOGGING_STEPS"
    --include_inputs_for_metrics True
    --eval_strategy "$EVAL_STRATEGY"
    --eval_steps "$EVAL_STEPS"
    --eval_print_examples True
    --eval_print_examples_count 2
    --eval_print_max_positions 2048
    --save_strategy steps
    --save_steps "$SAVE_STEPS"
    --seed "$SEED"
    --bf16 True
    --padding_free False
    --padding_free_return_flash_attn_kwargs False
    --project_name t2mlr_fineweb
    --disable_tqdm True
    --save_only_model False
    --t2mlr_enabled "$T2MLR_ENABLED"
    --recurrent_mixing_module_name "$T2MLR_MIXING_MODULE_NAME"
    --l_start "$L_START"
    --l_end "$L_END"
    --recurrent_weight "$RECURRENT_WEIGHT"
    --orig_weight "$ORIG_WEIGHT"
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
    --gate_weight_init_std "$GATE_WEIGHT_INIT_STD"
    --recurrent_residual_to_recurrent_cache "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE"
    --recurrent_residual_to_recurrent_cache_weight "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_WEIGHT"
    --recurrent_residual_to_recurrent_cache_post_norm "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_POST_NORM"
    --recurrent_residual_to_recurrent_cache_detach "$RECURRENT_RESIDUAL_TO_RECURRENT_CACHE_DETACH"
    --prompt_column ""
    --response_column text
    --default_new_tokens "$DEFAULT_NEW_TOKENS"
    --target_length_buffer "$TARGET_LENGTH_BUFFER"
    --do_sample True
    --num_generations_per_sample "$NUM_GENERATIONS_PER_SAMPLE"
    --capture_gate_trace "$GATE_BOOL"
    --log_gate_activity True
    --control_flow_all_recurrent True
    --train_dataset_config "$HF_DATASET_CONFIG"
)

PYTHON_COMMAND+=(--pass_at_k "${PASS_AT_K[@]}")

[[ -n "$MIXING_MODULE_KWARGS" ]] && PYTHON_COMMAND+=(--mixing_module_kwargs "$MIXING_MODULE_KWARGS")
[[ -n "$GATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--gate_mlp_hidden_dim "$GATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_STATE_MLP_HIDDEN_DIM" ]] && PYTHON_COMMAND+=(--recurrent_state_mlp_hidden_dim "$RECURRENT_STATE_MLP_HIDDEN_DIM")
[[ -n "$RECURRENT_GATE_INIT" ]] && PYTHON_COMMAND+=(--recurrent_gate_init "$RECURRENT_GATE_INIT")
[[ -n "$INPUT_GATE_INIT" ]] && PYTHON_COMMAND+=(--input_gate_init "$INPUT_GATE_INIT")
[[ -n "$GATE_LR_MULTIPLIER" ]] && PYTHON_COMMAND+=(--gate_lr_multiplier "$GATE_LR_MULTIPLIER")
[[ "$PROJECTION_BOOL" == "True" && "$PROJECTION_DIM_CHOICE" != "auto" ]] && PYTHON_COMMAND+=(--recurrent_projection_dim "$PROJECTION_DIM_CHOICE")

echo "============================================================================"
echo "[INFO] FineWeb T2MLR Training"
echo "============================================================================"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Model: $MODEL_NAME_OR_PATH"
echo "[INFO] T2MLR: $T2MLR_ENABLED (l_start=$L_START, l_end=$L_END)"
echo "[INFO] Output: $OUTPUT_DIR"
echo "============================================================================"

"${PYTHON_COMMAND[@]}"

echo "[INFO] Finished: $RUN_NAME"
