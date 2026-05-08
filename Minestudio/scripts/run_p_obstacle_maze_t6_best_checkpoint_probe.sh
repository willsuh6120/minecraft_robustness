#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"

setup_minestudio_runtime
require_xvfb_run
ensure_minestudio_engine

RUN_ROOT="${RUN_ROOT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
TASK="${TASK:-mine_coal}"
METRIC_GROUP="${METRIC_GROUP:-full16}"
METRIC_KEY="${METRIC_KEY:-success_rate}"
EPISODES="${EPISODES:-16}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-0.0}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
ASSET_DIR="${ASSET_DIR:-}"
BANK_NAME="${BANK_NAME:-}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"

if [[ -z "${RUN_ROOT}" || -z "${EXPERIMENT_NAME}" ]]; then
  echo "RUN_ROOT and EXPERIMENT_NAME are required." >&2
  exit 1
fi

BEST_INFO_JSON="$("${PYTHON_BIN}" "${ROOT_DIR}/scripts/resolve_maze_t6_best_checkpoint.py" \
  --run-root "${RUN_ROOT}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --metric-group "${METRIC_GROUP}" \
  --metric-key "${METRIC_KEY}")"

mapfile -t BEST_INFO < <("${PYTHON_BIN}" - <<'PY' "${BEST_INFO_JSON}"
import json
import sys

payload = json.loads(sys.argv[1])
for key in (
    "best_iteration",
    "best_score",
    "best_report_path",
    "best_model_path",
    "asset_dir",
    "bank_name",
):
    print(payload.get(key, ""))
PY
)

BEST_ITERATION="${BEST_INFO[0]}"
BEST_SCORE="${BEST_INFO[1]}"
BEST_REPORT_PATH="${BEST_INFO[2]}"
BEST_MODEL_PATH="${BEST_INFO[3]}"
AUTO_ASSET_DIR="${BEST_INFO[4]}"
AUTO_BANK_NAME="${BEST_INFO[5]}"

if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="${AUTO_ASSET_DIR}"
fi
if [[ -z "${BANK_NAME}" ]]; then
  BANK_NAME="${AUTO_BANK_NAME:-eval_bank}"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "failed to resolve ASSET_DIR for RUN_ROOT=${RUN_ROOT}" >&2
  exit 1
fi
if [[ -z "${BEST_MODEL_PATH}" || ! -f "${BEST_MODEL_PATH}" ]]; then
  echo "failed to resolve best model path: ${BEST_MODEL_PATH}" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-${RUN_ROOT}/${EXPERIMENT_NAME}/best_checkpoint_probe/full16_x${EPISODES}_${METRIC_GROUP}}"
mkdir -p "${OUT_ROOT}"

printf '%s\n' "${BEST_INFO_JSON}" > "${OUT_ROOT}/best_checkpoint.json"
echo "[maze-t6-best-probe] run_root=${RUN_ROOT}"
echo "[maze-t6-best-probe] experiment=${EXPERIMENT_NAME}"
echo "[maze-t6-best-probe] best_iteration=${BEST_ITERATION} score=${BEST_SCORE}"
echo "[maze-t6-best-probe] best_report=${BEST_REPORT_PATH}"
echo "[maze-t6-best-probe] model=${BEST_MODEL_PATH}"
echo "[maze-t6-best-probe] asset_dir=${ASSET_DIR}"
echo "[maze-t6-best-probe] bank_name=${BANK_NAME}"
echo "[maze-t6-best-probe] out_root=${OUT_ROOT}"

ASSET_DIR="${ASSET_DIR}" \
BANK_NAME="${BANK_NAME}" \
TASK="${TASK}" \
MODEL_PATH="${BEST_MODEL_PATH}" \
CFG_COEF="${CFG_COEF}" \
CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
ENV_REWARD_SCALE="${ENV_REWARD_SCALE}" \
EPISODES="${EPISODES}" \
WORLD_WORKERS="${WORLD_WORKERS}" \
BASE_SEED="${BASE_SEED}" \
SEED_STEP="${SEED_STEP}" \
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED}" \
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP}" \
EPISODE_RETRIES="${EPISODE_RETRIES}" \
STEP_BUDGET="${STEP_BUDGET}" \
STOP_ON_SUCCESS="${STOP_ON_SUCCESS}" \
SKIP_VIDEO="${SKIP_VIDEO}" \
OUT_ROOT="${OUT_ROOT}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

summarize_args=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/maze_t6_uniform_reward_tools.py" summarize-probe
  --summary-json "${OUT_ROOT}/summary.json"
  --label "${EXPERIMENT_NAME} best iter_${BEST_ITERATION} full16_x${EPISODES}"
  --out-json "${OUT_ROOT}/split_report.json"
  --out-md "${OUT_ROOT}/split_report.md"
)
if [[ -n "${BASELINE_SUMMARY}" ]]; then
  summarize_args+=( --baseline-summary "${BASELINE_SUMMARY}" )
fi
"${summarize_args[@]}" >/dev/null

echo "[maze-t6-best-probe] done out_root=${OUT_ROOT}"
