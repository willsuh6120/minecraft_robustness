#!/usr/bin/env bash
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

TASK="${TASK:-mine_coal}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
FINAL_INSTANCES="${FINAL_INSTANCES:-32}"
SKIP_FINAL_BANK="${SKIP_FINAL_BANK:-0}"
BASE_SEED="${BASE_SEED:-1}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BAKE_GOALS="${BAKE_GOALS:-0}"
GOAL_BAKE_WORLD_MODE="${GOAL_BAKE_WORLD_MODE:-clean_path}"
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-2}"
P_GOAL_POSE_PROTOCOL="${P_GOAL_POSE_PROTOCOL:-center_z3}"
P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-center_z3}"
P_OBSTACLE_MATERIAL="${P_OBSTACLE_MATERIAL:-}"
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT:-0}"
PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE:-0.25}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/mine_p_straight_obstacle_assets}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
unset LD_PRELOAD || true

cmd=(
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_prepare_mine_p_obstacle_assets
  --tasks "${TASK}"
  --eval-instances "${EVAL_INSTANCES}"
  --final-instances "${FINAL_INSTANCES}"
  --base-seed "${BASE_SEED}"
  --env-conf-dir "${SOURCE_ENV_CONF_DIR}"
  --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
  --mine-anchor-mode "${MINE_ANCHOR_MODE}"
  --bank-workers "${BANK_WORKERS}"
  --goal-bake-world-mode "${GOAL_BAKE_WORLD_MODE}"
  --p-goal-pose-protocol "${P_GOAL_POSE_PROTOCOL}"
  --path-obstacle-variant-set "${P_OBSTACLE_VARIANT_SET}"
  --path-obstacle-height "${P_OBSTACLE_HEIGHT}"
  --path-progress-reward-per-zone "${PATH_PROGRESS_REWARD_PER_ZONE}"
  --bake-goal-base-seed "${BAKE_GOAL_BASE_SEED}"
  --bake-goal-episode-retries "${BAKE_GOAL_EPISODE_RETRIES}"
  --out-dir "${OUT_DIR}"
)
if [[ -n "${P_OBSTACLE_MATERIAL}" ]]; then
  cmd+=( --path-obstacle-material "${P_OBSTACLE_MATERIAL}" )
fi

if [[ "${BAKE_GOALS}" == "1" ]]; then
  cmd+=( --bake-goals )
fi
if [[ "${SKIP_FINAL_BANK}" == "1" ]]; then
  cmd+=( --skip-final-bank )
fi
if [[ "${BAKE_GOAL_SAVE_DEBUG_ASSETS:-0}" == "1" ]]; then
  cmd+=( --bake-goal-save-debug-assets )
fi

echo "[prepare-mine-p-assets] out_dir=${OUT_DIR}"
echo "[prepare-mine-p-assets] bake_goals=${BAKE_GOALS}"
echo "[prepare-mine-p-assets] goal_bake_world_mode=${GOAL_BAKE_WORLD_MODE}"
echo "[prepare-mine-p-assets] p_goal_pose_protocol=${P_GOAL_POSE_PROTOCOL}"
echo "[prepare-mine-p-assets] p_obstacle_variant_set=${P_OBSTACLE_VARIANT_SET}"
echo "[prepare-mine-p-assets] p_obstacle_material=${P_OBSTACLE_MATERIAL:-variant_default}"
echo "[prepare-mine-p-assets] p_obstacle_height=${P_OBSTACLE_HEIGHT}"
echo "[prepare-mine-p-assets] path_progress_reward_per_zone=${PATH_PROGRESS_REWARD_PER_ZONE}"
echo "[prepare-mine-p-assets] skip_final_bank=${SKIP_FINAL_BANK}"
"${cmd[@]}"
