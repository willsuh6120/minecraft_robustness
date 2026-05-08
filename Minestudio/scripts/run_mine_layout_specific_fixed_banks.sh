#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

TASKS="${TASKS:-mine_coal}"
SPLITS="${SPLITS:-clean,train_id,ood,stress}"
INSTANCES_PER_SPLIT="${INSTANCES_PER_SPLIT:-8}"
BASE_SEED="${BASE_SEED:-1}"
ENV_CONF_DIR="${ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"
LAYOUT_CASES_CSV="${LAYOUT_CASES_CSV:-straight,offset_chamber_left,turn_left,side_alcove_left}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_layout_specific_fixed_banks_$(date +%Y%m%d_%H%M%S)}"
USE_XVFB="${USE_XVFB:-0}"
BAKE_GOALS="${BAKE_GOALS:-0}"
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-2}"
BAKE_GOAL_SAVE_DEBUG_ASSETS="${BAKE_GOAL_SAVE_DEBUG_ASSETS:-0}"
MAX_INSTANCE_ATTEMPTS="${MAX_INSTANCE_ATTEMPTS:-8}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

mkdir -p "${OUT_ROOT}"

run_py() {
  if [[ "${USE_XVFB}" == "1" ]]; then
    xvfb-run -a -s "-screen 0 1280x1024x24" "${PYTHON_BIN}" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

IFS=',' read -r -a LAYOUT_CASES <<< "${LAYOUT_CASES_CSV}"

echo "[config] OUT_ROOT=${OUT_ROOT}"
echo "[config] TASKS=${TASKS}"
echo "[config] SPLITS=${SPLITS}"
echo "[config] LAYOUT_CASES=${LAYOUT_CASES[*]}"

for layout_case in "${LAYOUT_CASES[@]}"; do
  layout_out="${OUT_ROOT}/${layout_case}"
  mkdir -p "${layout_out}"
  cmd=(
    -m minestudio.tutorials.inference.evaluate_rocket.interaction_generate_fixed_eval_bank
    --tasks "${TASKS}"
    --splits "${SPLITS}"
    --instances-per-split "${INSTANCES_PER_SPLIT}"
    --base-seed "${BASE_SEED}"
    --env-conf-dir "${ENV_CONF_DIR}"
    --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
    --mine-anchor-mode "${MINE_ANCHOR_MODE}"
    --mine-layout-case "${layout_case}"
    --max-instance-attempts "${MAX_INSTANCE_ATTEMPTS}"
    --out-dir "${layout_out}"
  )
  if [[ "${BAKE_GOALS}" == "1" ]]; then
    cmd+=(--bake-goals)
    cmd+=(--bake-goal-base-seed "${BAKE_GOAL_BASE_SEED}")
    cmd+=(--bake-goal-episode-retries "${BAKE_GOAL_EPISODE_RETRIES}")
    if [[ "${BAKE_GOAL_SAVE_DEBUG_ASSETS}" == "1" ]]; then
      cmd+=(--bake-goal-save-debug-assets)
    fi
  fi
  echo "==================== ${layout_case} ===================="
  run_py "${cmd[@]}"
done

echo "[done] generated layout-specific mine fixed banks under ${OUT_ROOT}"
