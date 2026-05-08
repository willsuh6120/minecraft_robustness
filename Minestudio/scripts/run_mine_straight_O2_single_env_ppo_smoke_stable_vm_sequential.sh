#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_RUN_SCRIPT="${BASE_RUN_SCRIPT:-${ROOT_DIR}/scripts/run_mine_straight_O2_single_env_ppo_smoke.sh}"

export TASK_GROUP_NAME="${TASK_GROUP_NAME:-straight_O2_single_env_smoke_stable_vm}"
export OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/ppo_straight_O2_single_env_smoke_stable_vm_$(date +%Y%m%d_%H%M%S)}"

export COLLECT_EPISODES="${COLLECT_EPISODES:-50}"
export EVAL_EPISODES="${EVAL_EPISODES:-10}"
export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export STEP_BUDGET="${STEP_BUDGET:-150}"
export MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-2}"
export PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-5e-6}"
export PPO_CLIP="${PPO_CLIP:-0.1}"
export PPO_GAMMA="${PPO_GAMMA:-0.999}"
export KL_COEF="${KL_COEF:-0.05}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-5.0}"
export CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
export KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
export FINAL_EVAL_MODEL_MODE="${FINAL_EVAL_MODEL_MODE:-best_eval}"

export COLLECT_WORKERS="${COLLECT_WORKERS:-1}"
export EVAL_WORKERS="${EVAL_WORKERS:-1}"
export FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS:-1}"

bash "${BASE_RUN_SCRIPT}"
