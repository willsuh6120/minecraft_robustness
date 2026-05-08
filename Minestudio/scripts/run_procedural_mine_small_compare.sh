#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python)}"

TASK="${TASK:-mine_coal}"
BANK_ROOT="${BANK_ROOT:-${ROOT}/outputs/evaluate_rocket/fixed_eval_bank_mines_v1/20260414_030828}"
OOD_BANK_DIR="${OOD_BANK_DIR:-${BANK_ROOT}/ood}"
STRESS_BANK_DIR="${STRESS_BANK_DIR:-${BANK_ROOT}/stress}"

EVAL_EPISODES="${EVAL_EPISODES:-4}"
COLLECT_EPISODES="${COLLECT_EPISODES:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-3}"
COLLECT_WORLD_INSTANCES="${COLLECT_WORLD_INSTANCES:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
STEP_BUDGET="${STEP_BUDGET:-90}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"

MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
COMMON_OUT_ROOT="${COMMON_OUT_ROOT:-${ROOT}/outputs/evaluate_rocket/procedural_mine_small_compare}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

run_pilot() {
  METHOD="$1"
  shift
  OUT_DIR="${COMMON_OUT_ROOT}/${METHOD}"
  echo
  echo "==================== ${METHOD} ===================="
  cd "${ROOT}"
  "${PYTHON}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot \
    --env-source rocket2_official \
    --task-group generated_task_group_proc_compare \
    --task-group-path "${ROOT}/outputs/evaluate_rocket/generated_task_group/20260411_132941" \
    --mine-layout-backend procedural \
    --mine-anchor-mode "${MINE_ANCHOR_MODE}" \
    --tasks "${TASK}" \
    --auto-goal \
    --eval-world-mode fixed_bank \
    --eval-fixed-bank-dir "${OOD_BANK_DIR}" \
    --final-eval-world-mode fixed_bank \
    --final-eval-fixed-bank-dir "${STRESS_BANK_DIR}" \
    --eval-episodes-per-task "${EVAL_EPISODES}" \
    --collect-episodes-per-task "${COLLECT_EPISODES}" \
    --collect-world-instances "${COLLECT_WORLD_INSTANCES}" \
    --base-seed "${BASE_SEED}" \
    --seed-step "${SEED_STEP}" \
    --episode-retries 1 \
    --step-budget-override "${STEP_BUDGET}" \
    --save-debug-assets \
    --model-path "${MODEL_PATH}" \
    --ppo-epochs 1 \
    --ppo-learning-rate 1e-5 \
    --ppo-clip 0.2 \
    --vf-coef 0.5 \
    --policy-coef 1.0 \
    --entropy-coef 0.0 \
    --kl-coef 0.01 \
    --normalize-advantage \
    --clip-vloss \
    --out-dir "${OUT_DIR}" \
    "$@"
}

run_pilot "frozen" \
  --collect-world-mode random \
  --collect-factor-split clean \
  --train-iters 0

run_pilot "clean_ppo" \
  --collect-world-mode random \
  --collect-factor-split clean \
  --train-iters "${TRAIN_ITERS}"

run_pilot "random" \
  --collect-world-mode random \
  --collect-factor-split train_id \
  --train-iters "${TRAIN_ITERS}"

run_pilot "targeted" \
  --collect-world-mode targeted \
  --collect-factor-split train_id \
  --targeted-review-backend heuristic \
  --train-iters "${TRAIN_ITERS}"
