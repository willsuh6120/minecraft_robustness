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
GOAL_PROTOCOL="${GOAL_PROTOCOL:-fixed_clean_front_close_v1}"
GOAL_SPEC="${GOAL_SPEC:-${ROOT_DIR}/outputs/evaluate_rocket/fixed_goal_banks/mine_coal/${GOAL_PROTOCOL}/goal_spec.json}"

SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_fixedgoal_vm}"
ASSET_DIR="${ASSET_DIR:-}"
GENERATE_ASSETS="${GENERATE_ASSETS:-auto}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
FINAL_INSTANCES="${FINAL_INSTANCES:-1}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BASE_SEED="${BASE_SEED:-1}"

COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS:-1}"
SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"

if [[ ! -f "${GOAL_SPEC}" ]]; then
  echo "[o2-fixed-goal-chain] GOAL_SPEC not found; generating fixed goal bank at ${GOAL_SPEC}"
  TASK="${TASK}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_BANK_ROOT="$(dirname "${GOAL_SPEC}")" \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  bash "${ROOT_DIR}/scripts/ensure_fixed_clean_goal_bank.sh"
fi
if [[ ! -d "${SOURCE_ENV_CONF_DIR}" ]]; then
  echo "SOURCE_ENV_CONF_DIR not found: ${SOURCE_ENV_CONF_DIR}" >&2
  exit 1
fi

if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ "${GENERATE_ASSETS}" == "1" || ( "${GENERATE_ASSETS}" == "auto" && ( -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ) ) ]]; then
  mkdir -p "${ASSET_ROOT}"
  echo "[o2-fixed-goal-chain] generating O2 assets without goal baking"
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_prepare_mine_o2_generalization_assets \
    --tasks "${TASK}" \
    --eval-instances "${EVAL_INSTANCES}" \
    --final-instances "${FINAL_INSTANCES}" \
    --base-seed "${BASE_SEED}" \
    --env-conf-dir "${SOURCE_ENV_CONF_DIR}" \
    --mine-layout-backend procedural \
    --mine-anchor-mode source_or_fallback \
    --layout-cases straight \
    --bank-workers "${BANK_WORKERS}" \
    --out-dir "${ASSET_ROOT}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  exit 1
fi

echo "[o2-fixed-goal-chain] goal_protocol=${GOAL_PROTOCOL}"
echo "[o2-fixed-goal-chain] goal_spec=${GOAL_SPEC}"
echo "[o2-fixed-goal-chain] asset_dir=${ASSET_DIR}"
echo "[o2-fixed-goal-chain] skip_all_final_eval=${SKIP_ALL_FINAL_EVAL}"

TASK="${TASK}" \
ASSET_DIR="${ASSET_DIR}" \
GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
GOAL_SPEC="${GOAL_SPEC}" \
BASE_SEED="${BASE_SEED}" \
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-${BASE_SEED}}" \
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED:-${BASE_SEED}}" \
COLLECT_EPISODES="${COLLECT_EPISODES}" \
EVAL_EPISODES="${EVAL_EPISODES}" \
COLLECT_WORKERS="${COLLECT_WORKERS}" \
EVAL_WORKERS="${EVAL_WORKERS}" \
FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS}" \
SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL}" \
COLLECT_VIDEO_MODE="${COLLECT_VIDEO_MODE:-skip}" \
BASELINE_VIDEO_MODE="${BASELINE_VIDEO_MODE:-skip}" \
EVAL_VIDEO_MODE="${EVAL_VIDEO_MODE:-skip}" \
FINAL_EVAL_VIDEO_MODE="${FINAL_EVAL_VIDEO_MODE:-skip}" \
SKIP_VIDEO="${SKIP_VIDEO}" \
EXPERIMENT_TAG="${EXPERIMENT_TAG:-fixed_clean_goal}" \
bash "${ROOT_DIR}/scripts/run_mine_o2_straight_generalization_blocks_vm.sh"
