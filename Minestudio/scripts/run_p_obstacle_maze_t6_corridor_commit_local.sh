#!/usr/bin/env bash
# corridor_commit focus mode — local run
#
# Phase 1: video probe on the first block's world (pre-training baseline behavior)
#   • 4 episodes with video + corridor annotation so you can visually verify
#     that the lane/causal-segment labels make sense BEFORE training
#
# Phase 2: training
#   • PLAN=single_left_funnel  (block 9 = left_funnel_t6, repeated 8× — hardest reliable case)
#   • ITERS_PER_BLOCK=2         same world, 2 PPO iterations before moving to next block
#   • LOSS_FOCUS_MODE=corridor_commit
#   • TRAINABLE_SCOPE=heads    (pi + value head only)
#   • PPO_LEARNING_RATE=2e-5
#
# Override any variable before calling:
#   PLAN=minimal ITERS_PER_BLOCK=1 bash scripts/run_p_obstacle_maze_t6_corridor_commit_local.sh
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${HOME}/miniconda3/envs/minestudio/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/minestudio/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

# ── Resolve ASSET_DIR ─────────────────────────────────────────────────────────
ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(
    find "${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_hard_v6_maze_t6_calibration" \
      -path "*/assets/*/eval_bank" -type d 2>/dev/null \
      | sed 's#/eval_bank$##' | sort | tail -n 1 || true
  )"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR missing or invalid. Set ASSET_DIR to the maze_t6 asset bank." >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_corridor_commit_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_maze_t6_corridor_commit/${RUN_TAG}}"
mkdir -p "${MASTER_ROOT}"

# ── Parameters ────────────────────────────────────────────────────────────────
PLAN="${PLAN:-single_left_funnel}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-2}"
LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE:-corridor_commit}"
LOSS_FOCUS_WEIGHT="${LOSS_FOCUS_WEIGHT:-4.0}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
KL_COEF="${KL_COEF:-0.1}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-heads}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
FINAL_PROBE_EPISODES="${FINAL_PROBE_EPISODES:-16}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
STEP_BUDGET="${STEP_BUDGET:-150}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

# First block world index — for single_left_funnel this is block 9 (left_funnel_t6)
VIDEO_PROBE_WORLD_IDX="${VIDEO_PROBE_WORLD_IDX:-9}"
PROBE_EPISODES="${PROBE_EPISODES:-4}"
SKIP_VIDEO_PROBE="${SKIP_VIDEO_PROBE:-0}"

echo "[corridor-commit] master_root=${MASTER_ROOT}"
echo "[corridor-commit] plan=${PLAN}  iters_per_block=${ITERS_PER_BLOCK}"
echo "[corridor-commit] loss_focus_mode=${LOSS_FOCUS_MODE}  focus_weight=${LOSS_FOCUS_WEIGHT}"

# ── Phase 1: video probe (pre-training baseline) ──────────────────────────────
if [[ "${SKIP_VIDEO_PROBE}" != "1" ]]; then
  echo
  echo "==================== Phase 1: video probe (pre-training) ===================="
  PROBE_OUT="${MASTER_ROOT}/video_probe_pretrain"
  ASSET_DIR="${ASSET_DIR}" \
  MODEL_PATH="${BASE_MODEL_PATH}" \
  WORLD_IDX="${VIDEO_PROBE_WORLD_IDX}" \
  EPISODES="${PROBE_EPISODES}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  FOCUS_WEIGHT="${LOSS_FOCUS_WEIGHT}" \
  OUT_ROOT="${PROBE_OUT}" \
  bash "${ROOT_DIR}/scripts/run_corridor_video_probe_local.sh"
  echo "[corridor-commit] pre-training probe written to ${PROBE_OUT}"
fi

# ── Phase 2: training ─────────────────────────────────────────────────────────
echo
echo "==================== Phase 2: training ===================="
LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE}" \
LOSS_FOCUS_WEIGHT="${LOSS_FOCUS_WEIGHT}" \
LOSS_FOCUS_TOP_K="${LOSS_FOCUS_TOP_K:-32}" \
PPO_LEARNING_RATE="${PPO_LEARNING_RATE}" \
KL_COEF="${KL_COEF}" \
PLAN="${PLAN}" \
TRAINABLE_SCOPE="${TRAINABLE_SCOPE}" \
COLLECT_EPISODES="${COLLECT_EPISODES}" \
EVAL_EPISODES="${EVAL_EPISODES}" \
FINAL_PROBE_EPISODES="${FINAL_PROBE_EPISODES}" \
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS}" \
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE}" \
ITERS_PER_BLOCK="${ITERS_PER_BLOCK}" \
STEP_BUDGET="${STEP_BUDGET}" \
MASTER_ROOT="${MASTER_ROOT}" \
RUN_TAG="${RUN_TAG}" \
ASSET_DIR="${ASSET_DIR}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_learning_signal_local.sh"
