#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  else
    PYTHON="$(command -v python)"
  fi
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "minestudio" ]]; then
  echo "minestudio conda env is not active. Current env: '${CONDA_DEFAULT_ENV:-<none>}'" >&2
  echo "Run: conda activate minestudio" >&2
  exit 1
fi

if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import yaml
import torch
import minestudio
PY
then
  echo "Active Python does not have required packages (yaml/torch/minestudio)." >&2
  echo "PYTHON=${PYTHON}" >&2
  echo "Run inside the minestudio env or override PYTHON explicitly." >&2
  exit 1
fi

TASK="${TASK:-mine_coal}"
METHODS="${METHODS:-frozen,clean_ppo,random,targeted}"

INSTANCES_PER_SPLIT="${INSTANCES_PER_SPLIT:-8}"
MAX_INSTANCE_ATTEMPTS="${MAX_INSTANCE_ATTEMPTS:-8}"
EVAL_EPISODES="${EVAL_EPISODES:-8}"
COLLECT_EPISODES="${COLLECT_EPISODES:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
COLLECT_WORLD_INSTANCES="${COLLECT_WORLD_INSTANCES:-8}"
EVAL_WORLD_INSTANCES="${EVAL_WORLD_INSTANCES:-8}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-120}"

MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

BANK_OUT_ROOT="${BANK_OUT_ROOT:-${ROOT}/outputs/evaluate_rocket/fixed_eval_bank_mines_baked_signal_local}"
COMPARE_OUT_ROOT="${COMPARE_OUT_ROOT:-${ROOT}/outputs/evaluate_rocket/procedural_mine_signal_compare_baked_local}"
REUSE_BANK_ROOT="${REUSE_BANK_ROOT:-}"

SAVE_DEBUG_ASSETS="${SAVE_DEBUG_ASSETS:-0}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
TARGETED_BOOTSTRAP_ON_SUCCESS="${TARGETED_BOOTSTRAP_ON_SUCCESS:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

sanitize_colon_file_list() {
  local raw="${1:-}"
  local parts=()
  local item
  local old_ifs="${IFS}"
  IFS=':'
  for item in ${raw}; do
    [[ -f "${item}" ]] || continue
    parts+=("${item}")
  done
  IFS="${old_ifs}"
  if (( ${#parts[@]} > 0 )); then
    (IFS=:; echo "${parts[*]}")
  fi
}

configure_torch_cuda_runtime() {
  local python_version
  python_version="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local site_pkg="${CONDA_PREFIX}/lib/python${python_version}/site-packages/nvidia"
  local lib_paths=()
  local subdir
  for subdir in \
    cublas/lib \
    cudnn/lib \
    cuda_runtime/lib \
    curand/lib \
    cusolver/lib \
    cusparse/lib \
    nvjitlink/lib
  do
    if [[ -d "${site_pkg}/${subdir}" ]]; then
      lib_paths+=("${site_pkg}/${subdir}")
    fi
  done
  if (( ${#lib_paths[@]} > 0 )); then
    local joined
    joined="$(IFS=:; echo "${lib_paths[*]}")"
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    local preload=()
    [[ -f "${site_pkg}/cublas/lib/libcublas.so.12" ]] && preload+=("${site_pkg}/cublas/lib/libcublas.so.12")
    [[ -f "${site_pkg}/cublas/lib/libcublasLt.so.12" ]] && preload+=("${site_pkg}/cublas/lib/libcublasLt.so.12")
    [[ -f "${site_pkg}/cudnn/lib/libcudnn.so.9" ]] && preload+=("${site_pkg}/cudnn/lib/libcudnn.so.9")
    if (( ${#preload[@]} > 0 )); then
      local preload_joined
      preload_joined="$(IFS=:; echo "${preload[*]}")"
      local existing_preload
      existing_preload="$(sanitize_colon_file_list "${LD_PRELOAD:-}")"
      export LD_PRELOAD="${preload_joined}${existing_preload:+:${existing_preload}}"
    fi
  fi
}

configure_torch_cuda_runtime

append_common_flags() {
  local -n _cmd_ref=$1
  if [[ "${SAVE_DEBUG_ASSETS}" == "1" ]]; then
    _cmd_ref+=(--save-debug-assets)
  fi
  if [[ "${SKIP_VIDEO}" == "1" ]]; then
    _cmd_ref+=(--skip-video)
  fi
}

format_seconds() {
  local total
  total="$(printf '%.0f' "${1:-0}")"
  if (( total < 60 )); then
    printf '%ss' "${total}"
    return
  fi
  local minutes=$(( total / 60 ))
  local seconds=$(( total % 60 ))
  if (( minutes < 60 )); then
    printf '%sm%02ds' "${minutes}" "${seconds}"
    return
  fi
  local hours=$(( minutes / 60 ))
  minutes=$(( minutes % 60 ))
  printf '%sh%02dm%02ds' "${hours}" "${minutes}" "${seconds}"
}

run_command() {
  local started_at
  started_at="$(date +%s)"
  echo "[run] $*"
  "$@"
  local ended_at
  ended_at="$(date +%s)"
  local elapsed=$(( ended_at - started_at ))
  echo "[done] elapsed_sec=${elapsed} elapsed_human=$(format_seconds "${elapsed}")"
}

print_plan() {
  local method_count=0
  IFS=',' read -r -a method_array <<< "${METHODS}"
  for method in "${method_array[@]}"; do
    method="$(echo "${method}" | xargs)"
    [[ -z "${method}" ]] && continue
    method_count=$(( method_count + 1 ))
  done
  local per_method_rollout_episodes=$(( EVAL_EPISODES + TRAIN_ITERS * (COLLECT_EPISODES + EVAL_EPISODES) + EVAL_EPISODES ))
  local total_rollout_episodes=$(( method_count * per_method_rollout_episodes ))
  local fixed_bank_instances=$(( 3 * INSTANCES_PER_SPLIT ))
  echo
  echo "==================== run plan ===================="
  echo "methods_count=${method_count}"
  echo "per_method_rollout_episodes=${per_method_rollout_episodes}"
  echo "total_rollout_episodes=${total_rollout_episodes}"
  echo "fixed_bank_instances=${fixed_bank_instances}"
  echo "max_instance_attempts=${MAX_INSTANCE_ATTEMPTS}"
  echo "video_mode=$([[ \"${SKIP_VIDEO}\" == \"1\" ]] && echo off || echo on)"
  if [[ "${SKIP_VIDEO}" == "0" ]]; then
    echo "note=video encoding is enabled; per-episode logs will now expose video_write_wall_time_sec."
  fi
}

ensure_bank() {
  if [[ -n "${REUSE_BANK_ROOT}" ]]; then
    BANK_ROOT="${REUSE_BANK_ROOT}"
    return
  fi

  mkdir -p "${BANK_OUT_ROOT}"
  local cmd=(
    "${PYTHON}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_generate_fixed_eval_bank
    --tasks "${TASK}"
    --splits train_id,ood,stress
    --instances-per-split "${INSTANCES_PER_SPLIT}"
    --max-instance-attempts "${MAX_INSTANCE_ATTEMPTS}"
    --base-seed 0
    --env-conf-dir "${SOURCE_ENV_CONF_DIR}"
    --protocol ours_v1
    --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
    --mine-anchor-mode "${MINE_ANCHOR_MODE}"
    --bake-goals
    --bake-goal-base-seed 1
    --bake-goal-episode-retries "${EPISODE_RETRIES}"
    --out-dir "${BANK_OUT_ROOT}"
  )
  if [[ "${SAVE_DEBUG_ASSETS}" == "1" ]]; then
    cmd+=(--bake-goal-save-debug-assets)
  fi

  echo
  echo "==================== generating fixed eval bank ===================="
  run_command "${cmd[@]}"

  BANK_ROOT="$(ls -td "${BANK_OUT_ROOT}"/* | head -1)"
}

run_pilot() {
  local method="$1"
  shift

  local out_dir="${COMPARE_OUT_ROOT}/${method}"
  local cmd=(
    "${PYTHON}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot
    --env-source rocket2_official
    --task-group generated_task_group_proc_compare_baked
    --task-group-path "${SOURCE_ENV_CONF_DIR}"
    --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
    --mine-anchor-mode "${MINE_ANCHOR_MODE}"
    --tasks "${TASK}"
    --auto-goal
    --eval-world-mode fixed_bank
    --eval-fixed-bank-dir "${OOD_BANK_DIR}"
    --final-eval-world-mode fixed_bank
    --final-eval-fixed-bank-dir "${STRESS_BANK_DIR}"
    --eval-world-instances "${EVAL_WORLD_INSTANCES}"
    --final-eval-world-instances "${EVAL_WORLD_INSTANCES}"
    --eval-episodes-per-task "${EVAL_EPISODES}"
    --collect-episodes-per-task "${COLLECT_EPISODES}"
    --collect-world-instances "${COLLECT_WORLD_INSTANCES}"
    --base-seed "${BASE_SEED}"
    --seed-step "${SEED_STEP}"
    --episode-retries "${EPISODE_RETRIES}"
    --step-budget-override "${STEP_BUDGET}"
    --model-path "${MODEL_PATH}"
    --ppo-epochs 1
    --ppo-learning-rate 1e-5
    --ppo-clip 0.2
    --vf-coef 0.5
    --policy-coef 1.0
    --entropy-coef 0.0
    --kl-coef 0.01
    --normalize-advantage
    --clip-vloss
    --bake-generated-goals
    --bake-goal-base-seed 1
    --bake-goal-episode-retries "${EPISODE_RETRIES}"
    --out-dir "${out_dir}"
  )
  if [[ "${SAVE_DEBUG_ASSETS}" == "1" ]]; then
    cmd+=(--bake-goal-save-debug-assets)
  fi
  append_common_flags cmd
  cmd+=("$@")

  echo
  echo "==================== ${method} ===================="
  run_command "${cmd[@]}"
}

print_summary() {
  echo
  echo "==================== summaries ===================="
  IFS=',' read -r -a method_array <<< "${METHODS}"
  for method in "${method_array[@]}"; do
    method="$(echo "${method}" | xargs)"
    [[ -z "${method}" ]] && continue
    if [[ ! -d "${COMPARE_OUT_ROOT}/${method}" ]]; then
      continue
    fi
    local run_dir
    run_dir="$(ls -td "${COMPARE_OUT_ROOT}/${method}"/* 2>/dev/null | head -1 || true)"
    if [[ -z "${run_dir}" ]]; then
      continue
    fi
    echo "---- ${method} ----"
    echo "${run_dir}"
    if [[ -f "${run_dir}/pilot_summary.json" ]]; then
      "${PYTHON}" - <<PY
import json
from pathlib import Path
path = Path(${run_dir@Q}) / "pilot_summary.json"
data = json.loads(path.read_text())
print(json.dumps({
    "baseline_summary": data.get("baseline_summary"),
    "iteration_eval_summaries": [item.get("eval_summary") for item in data.get("iteration_summaries", [])],
    "final_eval_summary": data.get("final_eval_summary"),
}, indent=2, ensure_ascii=False))
PY
    fi
  done
}

main() {
  mkdir -p "${COMPARE_OUT_ROOT}"
  cd "${ROOT}"

  echo "ROOT=${ROOT}"
  echo "PYTHON=${PYTHON}"
  echo "TASK=${TASK}"
  echo "METHODS=${METHODS}"
  echo "SOURCE_ENV_CONF_DIR=${SOURCE_ENV_CONF_DIR}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "INSTANCES_PER_SPLIT=${INSTANCES_PER_SPLIT}"
  echo "COLLECT_WORLD_INSTANCES=${COLLECT_WORLD_INSTANCES}"
  echo "EVAL_WORLD_INSTANCES=${EVAL_WORLD_INSTANCES}"
  echo "COLLECT_EPISODES=${COLLECT_EPISODES}"
  echo "EVAL_EPISODES=${EVAL_EPISODES}"
  echo "TRAIN_ITERS=${TRAIN_ITERS}"
  echo "STEP_BUDGET=${STEP_BUDGET}"
  echo "SAVE_DEBUG_ASSETS=${SAVE_DEBUG_ASSETS}"
  echo "SKIP_VIDEO=${SKIP_VIDEO}"
  echo "TARGETED_BOOTSTRAP_ON_SUCCESS=${TARGETED_BOOTSTRAP_ON_SUCCESS}"
  print_plan

  ensure_bank
  OOD_BANK_DIR="${BANK_ROOT}/ood"
  STRESS_BANK_DIR="${BANK_ROOT}/stress"

  echo "BANK_ROOT=${BANK_ROOT}"
  echo "OOD_BANK_DIR=${OOD_BANK_DIR}"
  echo "STRESS_BANK_DIR=${STRESS_BANK_DIR}"

  IFS=',' read -r -a method_array <<< "${METHODS}"
  for method in "${method_array[@]}"; do
    method="$(echo "${method}" | xargs)"
    case "${method}" in
      frozen)
        run_pilot frozen \
          --collect-world-mode random \
          --collect-factor-split clean \
          --train-iters 0
        ;;
      clean_ppo)
        run_pilot clean_ppo \
          --collect-world-mode random \
          --collect-factor-split clean \
          --train-iters "${TRAIN_ITERS}"
        ;;
      random)
        run_pilot random \
          --collect-world-mode random \
          --collect-factor-split train_id \
          --train-iters "${TRAIN_ITERS}"
        ;;
      targeted)
        targeted_args=(
          --collect-world-mode targeted
          --collect-factor-split train_id
          --targeted-review-backend heuristic
          --train-iters "${TRAIN_ITERS}"
        )
        if [[ "${TARGETED_BOOTSTRAP_ON_SUCCESS}" == "1" ]]; then
          targeted_args+=(--targeted-bootstrap-on-success)
        fi
        run_pilot targeted \
          "${targeted_args[@]}"
        ;;
      "")
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 1
        ;;
    esac
  done

  print_summary
}

main "$@"
