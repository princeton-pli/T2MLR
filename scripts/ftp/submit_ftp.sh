#!/usr/bin/env bash
# Submitter for Medusa training sweeps.

set -euo pipefail
info() { echo "[INFO] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP_NAME="${EXP_NAME:-ftp}"
SWEEP="${SWEEP:-1}"
SUBMIT_TRAIN="${SUBMIT_TRAIN:-1}"
LOCAL_RUN="${LOCAL_RUN:-0}"
SBATCH_ARGS_BASE="${SBATCH_ARGS_BASE:-${SBATCH_ARGS:-}}"
RUN_NAME_BASE="${RUN_NAME_BASE:-ftp}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/scripts/${EXP_NAME}/slurm}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/${EXP_NAME}/train_ftp.sh}"
SWEEP_PARAMS_YAML="${SWEEP_PARAMS_YAML:-$REPO_ROOT/scripts/${EXP_NAME}/sweep_params.yaml}"

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
  local -a out=()
  local combined=""
  if [ -n "${SBATCH_ARGS_BASE:-}" ]; then
    combined="${SBATCH_ARGS_BASE}"
  fi
  if [ -n "${SBATCH_ARGS_EXTRA:-}" ]; then
    if [ -n "$combined" ]; then
      combined="${combined} ${SBATCH_ARGS_EXTRA}"
    else
      combined="${SBATCH_ARGS_EXTRA}"
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
  local train_job="(skipped)"

  mkdir -p "$LOG_DIR"
  [ -f "$TRAIN_SCRIPT" ] || die "Missing train script: $TRAIN_SCRIPT"

  if _is_true "$LOCAL_RUN"; then
    info "Running locally (LOCAL_RUN=1) for tag: $run_tag"
    bash "$TRAIN_SCRIPT"
    return
  fi

  if _is_true "$SUBMIT_TRAIN"; then
    local -a extra_args=()
    while IFS= read -r tok; do
      [ -n "$tok" ] || continue
      extra_args+=("$tok")
    done < <(_parse_sbatch_args)

    if [ "${#extra_args[@]}" -gt 0 ]; then
      info "Train sbatch overrides: ${extra_args[*]}"
    fi

    mkdir -p "$LOG_DIR/${run_tag}"
    train_job="$(sbatch --export=ALL --parsable \
      --job-name="ftp_${run_tag}" \
      --output="$LOG_DIR/${run_tag}/train-%j.out" \
      --error="$LOG_DIR/${run_tag}/train-%j.err" \
      "${extra_args[@]}" \
      "$TRAIN_SCRIPT")"

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
  export SBATCH_ARGS_EXTRA=""

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
      export WANDB_RUN_ID="$param_value"
      export WANDB_RESUME="${WANDB_RESUME:-allow}"
      continue
    fi

    if [ "$param_name" = "SBATCH_ARGS" ]; then
      export SBATCH_ARGS_EXTRA="$param_value"
      continue
    fi

    if [ "$param_name" = "RUN_NAME_SUFFIX" ]; then
      set_suffix=1
    fi

    export "$param_name"="$param_value"

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
  done

  # Include t2mlr_enabled status explicitly in the run name
  local t2mlr_tag=""
  case "${T2MLR_ENABLED:-False}" in
    True|true|1) t2mlr_tag="_t2mlr_on" ;;
    *)           t2mlr_tag="_t2mlr_off" ;;
  esac

  local tag=""
  if [ -n "$override_tag" ]; then
    tag="${override_tag}${t2mlr_tag}"
  else
    tag="$(_build_run_tag "$RUN_NAME_BASE" "${tag_parts[@]}")${t2mlr_tag}"
  fi

  if [ "$set_suffix" -eq 0 ] && [ "$has_env_suffix" -eq 0 ]; then
    export RUN_NAME_SUFFIX="$tag"
  fi

  submit_run "$tag"
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
  submit_run "$run_tag"
fi
