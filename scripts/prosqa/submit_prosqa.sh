#!/usr/bin/env bash
# Simple submitter for prosqa training sweeps (train only, no eval).

set -euo pipefail
info() { echo "[INFO] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SWEEP="${SWEEP:-1}"
SUBMIT_TRAIN="${SUBMIT_TRAIN:-1}"
LOCAL_RUN="${LOCAL_RUN:-0}"
SBATCH_ARGS_BASE="${SBATCH_ARGS_BASE:-${SBATCH_ARGS:-}}"
SFT_RUN_NAME_BASE="${SFT_RUN_NAME_BASE:-prosqa_t2mlr}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/scripts/prosqa/slurm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/prosqa/train_prosqa.sh}"
SWEEP_PARAMS_YAML="${SWEEP_PARAMS_YAML:-$REPO_ROOT/scripts/prosqa/sweep_params.yaml}"

_is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

_build_run_tag() {
  local base_name="$1"
  shift
  local -a args=("$@")
  local params_str=""
  local i
  for ((i=0; i<${#args[@]}; i+=2)); do
    local name="${args[$i]}"
    local value="${args[$i+1]}"
    params_str="${params_str}${name}=${value};"
  done
  local hash
  hash="$(echo -n "$params_str" | md5sum | cut -c1-12)"
  echo "${base_name}_${hash}"
}

_parse_sbatch_args() {
  local base="${1:-}"
  local extra="${2:-}"
  local -a out=()
  local combined=""
  if [ -n "$base" ]; then
    combined="$base"
  fi
  if [ -n "$extra" ]; then
    if [ -n "$combined" ]; then
      combined="${combined} ${extra}"
    else
      combined="${extra}"
    fi
  fi
  if [ -n "$combined" ]; then
    while IFS= read -r tok; do
      [ -n "$tok" ] || continue
      out+=("$tok")
    done < <(SBATCH_ARGS_COMBINED="$combined" python3 - <<'PY'
import os, shlex
s = os.environ.get("SBATCH_ARGS_COMBINED", "") or ""
for tok in shlex.split(s):
    print(tok)
PY
    )
  fi
  printf '%s\n' "${out[@]}"
}

submit_run() {
  local run_tag="$1"
  local submit_train="$2"
  local local_run="$3"
  local log_dir="$4"
  local train_script="$5"
  local sbatch_args_base="$6"
  local sbatch_args_extra="$7"
  shift 7
  local -a run_env=("$@")
  local train_job="(skipped)"

  mkdir -p "$log_dir"
  [ -f "$train_script" ] || die "Missing train script: $train_script"

  if _is_true "$local_run"; then
    info "Running locally (LOCAL_RUN=1) for tag: $run_tag"
    if [ "${#run_env[@]}" -gt 0 ]; then
      env "${run_env[@]}" "$train_script"
    else
      "$train_script"
    fi
    return
  fi

  if _is_true "$submit_train"; then
    local -a extra_args=()
    while IFS= read -r tok; do
      [ -n "$tok" ] || continue
      extra_args+=("$tok")
    done < <(_parse_sbatch_args "$sbatch_args_base" "$sbatch_args_extra")

    if [ "${#extra_args[@]}" -gt 0 ]; then
      info "Train sbatch overrides: ${extra_args[*]}"
    fi

    if [ "${#run_env[@]}" -gt 0 ]; then
      train_job="$(env "${run_env[@]}" sbatch --export=ALL --parsable \
        --job-name="prosqa_${run_tag}" \
        --output="$log_dir/${run_tag}/train-%j.out" \
        --error="$log_dir/${run_tag}/train-%j.err" \
        "${extra_args[@]}" \
        "$train_script")"
    else
      train_job="$(sbatch --export=ALL --parsable \
        --job-name="prosqa_${run_tag}" \
        --output="$log_dir/${run_tag}/train-%j.out" \
        --error="$log_dir/${run_tag}/train-%j.err" \
        "${extra_args[@]}" \
        "$train_script")"
    fi

    [ -n "$train_job" ] || die "Train submission failed"
    info "Train job id: $train_job"
  else
    info "Skipping train submission (SUBMIT_TRAIN=0)"
  fi
}

_process_yaml_run() {
  local -a tag_parts=()
  local override_tag=""
  local set_suffix=0
  local has_env_suffix=0
  local sbatch_args_extra=""
  local submit_train="$SUBMIT_TRAIN"
  local local_run="$LOCAL_RUN"
  local log_dir="$LOG_DIR"
  local train_script="$TRAIN_SCRIPT"
  local sbatch_args_base="$SBATCH_ARGS_BASE"
  local sft_run_name_base="$SFT_RUN_NAME_BASE"
  local -a run_env=()
  local wandb_resume="${WANDB_RESUME:-allow}"
  local wandb_resume_override=""
  local has_wandb_resume_override=0

  if [ -n "${RUN_NAME_SUFFIX:-}" ]; then
    has_env_suffix=1
  fi

  local -a control_params=("RUN_TAG" "SUBMIT_TRAIN" "LOCAL_RUN" "SBATCH_ARGS" "WANDB_RESUME_RUN_ID")
  local i=1
  while [ "$i" -le "$#" ]; do
    local param_name="${!i}"
    i=$((i + 1))
    local param_value="${!i}"
    i=$((i + 1))

    if [ "$param_name" = "RUN_TAG" ]; then
      override_tag="$param_value"
      continue
    fi

    if [ "$param_name" = "WANDB_RESUME_RUN_ID" ]; then
      if [ "$has_wandb_resume_override" -eq 1 ]; then
        run_env+=("WANDB_RUN_ID=$param_value")
      else
        run_env+=("WANDB_RUN_ID=$param_value" "WANDB_RESUME=$wandb_resume")
      fi
      continue
    fi

    if [ "$param_name" = "SBATCH_ARGS" ]; then
      sbatch_args_extra="$param_value"
      continue
    fi

    if [ "$param_name" = "RUN_NAME_SUFFIX" ]; then
      set_suffix=1
    fi
    if [ "$param_name" = "WANDB_RESUME" ]; then
      wandb_resume_override="$param_value"
      has_wandb_resume_override=1
    fi

    if [ "$param_name" = "SUBMIT_TRAIN" ]; then
      submit_train="$param_value"
    fi
    if [ "$param_name" = "LOCAL_RUN" ]; then
      local_run="$param_value"
    fi
    if [ "$param_name" = "LOG_DIR" ]; then
      log_dir="$param_value"
    fi
    if [ "$param_name" = "TRAIN_SCRIPT" ]; then
      train_script="$param_value"
    fi
    if [ "$param_name" = "SBATCH_ARGS_BASE" ]; then
      sbatch_args_base="$param_value"
    fi
    if [ "$param_name" = "SFT_RUN_NAME_BASE" ]; then
      sft_run_name_base="$param_value"
    fi

    local is_control=0
    local control_param
    for control_param in "${control_params[@]}"; do
      if [ "$param_name" = "$control_param" ]; then
        is_control=1
        break
      fi
    done
    if [ "$is_control" -eq 0 ]; then
      tag_parts+=("$param_name" "$param_value")
    fi
    run_env+=("$param_name=$param_value")
  done

  local tag=""
  if [ -n "$override_tag" ]; then
    tag="$override_tag"
  else
    tag="$(_build_run_tag "$sft_run_name_base" "${tag_parts[@]}")"
  fi

  if [ "$set_suffix" -eq 0 ] && [ "$has_env_suffix" -eq 0 ]; then
    run_env+=("RUN_NAME_SUFFIX=$tag")
  fi

  submit_run "$tag" "$submit_train" "$local_run" "$log_dir" "$train_script" \
    "$sbatch_args_base" "$sbatch_args_extra" "${run_env[@]}"
}

if [ "$SWEEP" = "1" ] || [ "$SWEEP" = "true" ] || [ "$SWEEP" = "True" ]; then
  [ -f "$SWEEP_PARAMS_YAML" ] || die "Missing sweep params YAML: $SWEEP_PARAMS_YAML"

  temp_runs="$(mktemp)"
  python3 <<PYTHON_EOF > "$temp_runs" || die "Failed to parse YAML"
import yaml
import sys

try:
    with open("$SWEEP_PARAMS_YAML", 'r') as f:
        data = yaml.safe_load(f)

    if 'runs' not in data or not isinstance(data['runs'], list):
        print("Error: YAML must have a 'runs' key with a list of runs", file=sys.stderr)
        sys.exit(1)

    for run in data['runs']:
        if not isinstance(run, dict):
            continue
        for param_name, param_value in sorted(run.items()):
            print(param_name)
            print(str(param_value))
        print()
except Exception as e:
    print(f"Error parsing YAML: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

  run_args=()
  run_count=0
  while IFS= read -r line; do
    if [ -z "$line" ]; then
      if [ ${#run_args[@]} -gt 0 ]; then
        run_count=$((run_count + 1))
        (_process_yaml_run "${run_args[@]}")
        run_args=()
      fi
    else
      param_name="$line"
      IFS= read -r param_value || param_value=""
      if [ -n "$param_name" ] && [ -n "$param_value" ]; then
        run_args+=("$param_name" "$param_value")
      fi
    fi
  done < "$temp_runs"

  if [ ${#run_args[@]} -gt 0 ]; then
    run_count=$((run_count + 1))
    (_process_yaml_run "${run_args[@]}")
  fi

  rm -f "$temp_runs"
  info "Sweep submission complete. Processed $run_count run(s)."
else
  run_tag="${RUN_TAG:-run}"
  submit_run "$run_tag" "$SUBMIT_TRAIN" "$LOCAL_RUN" "$LOG_DIR" "$TRAIN_SCRIPT" \
    "$SBATCH_ARGS_BASE" ""
fi
