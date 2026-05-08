#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"
setup_minestudio_runtime
ensure_minestudio_engine

RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_control_vm_fresh_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/${RUN_TAG}}"
ASSET_ROOT="${ASSET_ROOT:-${OUT_ROOT}/assets}"
PATCH_ASSET_DIR_TOOL_TYPE="${PATCH_ASSET_DIR_TOOL_TYPE:-diamond_pickaxe}"
STEP_BUDGET="${STEP_BUDGET:-150}"
WARMUP_NOOP_STEPS="${WARMUP_NOOP_STEPS:-30}"

find_latest_vm_asset_dir() {
  find "${ROOT_DIR}/outputs/evaluate_rocket" \
    -maxdepth 3 \
    -type d \
    -path "*/p_obstacle_maze_t6_control_vm_fresh_*/assets/*" \
    | sort \
    | tail -n 1
}

if [[ -z "${ASSET_DIR:-}" ]]; then
  ASSET_DIR="$(find_latest_vm_asset_dir)"
fi
if [[ -z "${ASSET_DIR:-}" || ! -d "${ASSET_DIR}/eval_bank" ]]; then
  echo "No reusable VM maze_t6 asset bank found. Set ASSET_DIR explicitly to an existing assets/<timestamp> dir." >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/patch_maze_t6_asset_dir_for_rerun.py" \
  --asset-dir "${ASSET_DIR}" \
  --decision-steps "${STEP_BUDGET}" \
  --warmup-steps "${WARMUP_NOOP_STEPS}" \
  --tool-type "${PATCH_ASSET_DIR_TOOL_TYPE}"

RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE:-0}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
RUN_CONTROL_A="${RUN_CONTROL_A:-1}"
RUN_MAIN_B="${RUN_MAIN_B:-0}"

PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE:-0.125}"
BANK_WORKERS="${BANK_WORKERS:-4}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
FINAL_PROBE_WORKERS="${FINAL_PROBE_WORKERS:-4}"

RUN_TAG="${RUN_TAG}" \
OUT_ROOT="${OUT_ROOT}" \
ASSET_ROOT="${ASSET_ROOT}" \
ASSET_DIR="${ASSET_DIR}" \
RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE}" \
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL}" \
RUN_CONTROL_A="${RUN_CONTROL_A}" \
RUN_MAIN_B="${RUN_MAIN_B}" \
STEP_BUDGET="${STEP_BUDGET}" \
PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE}" \
BANK_WORKERS="${BANK_WORKERS}" \
COLLECT_WORKERS="${COLLECT_WORKERS}" \
VIDEO_WORKERS="${VIDEO_WORKERS}" \
EVAL_WORKERS="${EVAL_WORKERS}" \
FINAL_PROBE_WORKERS="${FINAL_PROBE_WORKERS}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_uniform_reward_suite.sh"
