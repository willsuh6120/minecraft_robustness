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
OVERNIGHT_ROOT="${OVERNIGHT_ROOT:-}"
if [[ -z "${OVERNIGHT_ROOT}" ]]; then
  OVERNIGHT_ROOT="$(
    find "${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_center_z3_overnight" \
      -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
      | sort \
      | tail -n 1 || true
  )"
fi
if [[ -z "${OVERNIGHT_ROOT}" || ! -d "${OVERNIGHT_ROOT}" ]]; then
  echo "OVERNIGHT_ROOT is missing. Set it to a completed p_obstacle_center_z3_overnight run." >&2
  exit 1
fi

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(
    find "${OVERNIGHT_ROOT}/assets" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
      | sort \
      | tail -n 1 || true
  )"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ASSET_DIR}/eval_bank/bank_manifest.json" ]]; then
  echo "Missing eval bank manifest: ${ASSET_DIR}/eval_bank/bank_manifest.json" >&2
  exit 1
fi

MODEL_MODE="${MODEL_MODE:-baseline}"
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  case "${MODEL_MODE}" in
    baseline|base|rocket2)
      MODEL_PATH="hf:phython96/ROCKET-2-1x-22w"
      ;;
    final)
      CHAIN_SUMMARY="${CHAIN_SUMMARY:-${OVERNIGHT_ROOT}/training/teacher_to_hard/seed_001/chain_summary.json}"
      MODEL_PATH="$(
        "${PYTHON_BIN}" - <<'PY' "${CHAIN_SUMMARY}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
model = ""
if path.exists():
    chain = json.loads(path.read_text(encoding="utf-8"))
    blocks = chain.get("blocks") or []
    if blocks:
        model = str(blocks[-1].get("next_model_path") or "")
print(model)
PY
      )"
      ;;
    best)
      CHAIN_SUMMARY="${CHAIN_SUMMARY:-${OVERNIGHT_ROOT}/training/teacher_to_hard/seed_001/chain_summary.json}"
      MODEL_PATH="$(
        "${PYTHON_BIN}" - <<'PY' "${CHAIN_SUMMARY}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
model = ""
best_rate = -1.0
if path.exists():
    chain = json.loads(path.read_text(encoding="utf-8"))
    for block in chain.get("blocks") or []:
        for key in ("best_eval_selection", "final_eval_model_selection"):
            sel = block.get(key) or {}
            path_value = str(sel.get("model_path") or "")
            rate = float(sel.get("auto_success_rate") or -1.0)
            if path_value and rate > best_rate:
                best_rate = rate
                model = path_value
print(model)
PY
      )"
      ;;
    *)
      echo "Unknown MODEL_MODE=${MODEL_MODE}. Use baseline, final, best, or set MODEL_PATH directly." >&2
      exit 1
      ;;
  esac
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH could not be resolved." >&2
  exit 1
fi
if [[ "${MODEL_PATH}" != hf:* && ! -f "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is not a file and not an hf: reference: ${MODEL_PATH}" >&2
  exit 1
fi

EPISODES="${EPISODES:-3}"
WORLD_WORKERS="${WORLD_WORKERS:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-${BASE_SEED}}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_center_z3_video_probe/$(date +%Y%m%d_%H%M%S)_${MODEL_MODE}}"

mkdir -p "${OUT_ROOT}"

echo "[p-center-z3-video-probe] overnight_root=${OVERNIGHT_ROOT}"
echo "[p-center-z3-video-probe] asset_dir=${ASSET_DIR}"
echo "[p-center-z3-video-probe] model_mode=${MODEL_MODE}"
echo "[p-center-z3-video-probe] model_path=${MODEL_PATH}"
echo "[p-center-z3-video-probe] episodes_per_world=${EPISODES}"
echo "[p-center-z3-video-probe] world_workers=${WORLD_WORKERS}"
echo "[p-center-z3-video-probe] video=enabled"
echo "[p-center-z3-video-probe] out_root=${OUT_ROOT}"

ASSET_DIR="${ASSET_DIR}" \
TASK="${TASK}" \
BANK_NAME=eval_bank \
EPISODES="${EPISODES}" \
WORLD_WORKERS="${WORLD_WORKERS}" \
BASE_SEED="${BASE_SEED}" \
SEED_STEP="${SEED_STEP}" \
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED}" \
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP}" \
EPISODE_RETRIES="${EPISODE_RETRIES}" \
STEP_BUDGET="${STEP_BUDGET}" \
MODEL_PATH="${MODEL_PATH}" \
CFG_COEF="${CFG_COEF:-0.0}" \
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}" \
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}" \
GOAL_SPEC="" \
STOP_ON_SUCCESS="${STOP_ON_SUCCESS}" \
SKIP_VIDEO=0 \
OUT_ROOT="${OUT_ROOT}" \
bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

find "${OUT_ROOT}" -type f -name "*_annotated.mp4" | sort > "${OUT_ROOT}/annotated_video_manifest.txt"
find "${OUT_ROOT}" -type f -name "*.mp4" | sort > "${OUT_ROOT}/video_manifest.txt"
find "${OUT_ROOT}" -type f -name "trajectory.jsonl" | sort > "${OUT_ROOT}/trajectory_manifest.txt"

echo "[p-center-z3-video-probe] summary=${OUT_ROOT}/summary.json"
echo "[p-center-z3-video-probe] annotated_videos=${OUT_ROOT}/annotated_video_manifest.txt"
echo "[p-center-z3-video-probe] videos=${OUT_ROOT}/video_manifest.txt"
echo "[p-center-z3-video-probe] trajectories=${OUT_ROOT}/trajectory_manifest.txt"
