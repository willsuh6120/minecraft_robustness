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
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm}"
ASSET_DIR="${ASSET_DIR:-${ASSET_ROOT}/20260427_054807}"
GENERATE_ASSETS="${GENERATE_ASSETS:-0}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
FINAL_INSTANCES="${FINAL_INSTANCES:-1}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BASE_SEED="${BASE_SEED:-1}"
BAKE_GOALS="${BAKE_GOALS:-1}"
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-2}"

VARIANTS="${VARIANTS:-forward,reverse,random_a,random_b,best_single,bad_single,no_update}"
BASE_SEEDS="${BASE_SEEDS:-1}"
SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL:-1}"
SUITE_TAG="${SUITE_TAG:-auto_goal_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/o2_order_ablation_suite/${SUITE_TAG}}"

COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS:-1}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-2}"

if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ "${GENERATE_ASSETS}" == "1" || ( "${GENERATE_ASSETS}" == "auto" && ( -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ) ) ]]; then
  mkdir -p "${ASSET_ROOT}"
  echo "[o2-auto-goal-order-suite] generating O2 assets asset_root=${ASSET_ROOT}"
  cmd=(
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_prepare_mine_o2_generalization_assets
    --tasks "${TASK}"
    --eval-instances "${EVAL_INSTANCES}"
    --final-instances "${FINAL_INSTANCES}"
    --base-seed "${BASE_SEED}"
    --env-conf-dir "${SOURCE_ENV_CONF_DIR}"
    --mine-layout-backend procedural
    --mine-anchor-mode source_or_fallback
    --layout-cases straight
    --bank-workers "${BANK_WORKERS}"
    --bake-goal-base-seed "${BAKE_GOAL_BASE_SEED}"
    --bake-goal-episode-retries "${BAKE_GOAL_EPISODE_RETRIES}"
    --out-dir "${ASSET_ROOT}"
  )
  if [[ "${BAKE_GOALS}" == "1" ]]; then
    cmd+=(--bake-goals)
  fi
  "${cmd[@]}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  echo "Exact O2 repro expects the old baked-goal asset bank:" >&2
  echo "  ${ASSET_ROOT}/20260427_054807" >&2
  echo "If you intentionally want to regenerate assets, set GENERATE_ASSETS=1 explicitly." >&2
  exit 1
fi

echo "[o2-auto-goal-order-suite] asset_dir=${ASSET_DIR}"
echo "[o2-auto-goal-order-suite] bake_goals=${BAKE_GOALS}"
echo "[o2-auto-goal-order-suite] generate_assets=${GENERATE_ASSETS}"
echo "[o2-auto-goal-order-suite] exact_repro_asset=20260427_054807"
echo "[o2-auto-goal-order-suite] variants=${VARIANTS}"
echo "[o2-auto-goal-order-suite] base_seeds=${BASE_SEEDS}"
echo "[o2-auto-goal-order-suite] suite_root=${SUITE_ROOT}"
echo "[o2-auto-goal-order-suite] goal_protocol=auto_goal"
echo "[o2-auto-goal-order-suite] goal_spec=<empty>"

TASK="${TASK}" \
ASSET_DIR="${ASSET_DIR}" \
GOAL_PROTOCOL=auto_goal \
GOAL_SPEC="" \
VARIANTS="${VARIANTS}" \
BASE_SEEDS="${BASE_SEEDS}" \
SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL}" \
SUITE_ROOT="${SUITE_ROOT}" \
COLLECT_EPISODES="${COLLECT_EPISODES}" \
EVAL_EPISODES="${EVAL_EPISODES}" \
COLLECT_WORKERS="${COLLECT_WORKERS}" \
EVAL_WORKERS="${EVAL_WORKERS}" \
FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS}" \
ITERS_PER_BLOCK="${ITERS_PER_BLOCK}" \
COLLECT_VIDEO_MODE=skip \
BASELINE_VIDEO_MODE=skip \
EVAL_VIDEO_MODE=skip \
FINAL_EVAL_VIDEO_MODE=skip \
SKIP_VIDEO=1 \
bash "${ROOT_DIR}/scripts/run_o2_order_ablation_suite_vm.sh"
