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

XVFB_RUN="${XVFB_RUN:-$(command -v xvfb-run || true)}"
XVFB_SCREEN_ARGS="${XVFB_SCREEN_ARGS:--screen 0 1280x1024x24}"

if [[ -z "${XVFB_RUN}" ]]; then
  echo "xvfb-run not found. Install xvfb first." >&2
  exit 1
fi

TASK="${TASK:-mine_coal}"
TASK_GROUP_NAME="${TASK_GROUP_NAME:-mine_o2_straight_generalization_vm}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-auto_goal}"
GOAL_SPEC="${GOAL_SPEC:-}"

COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
BLOCK_COUNT="${BLOCK_COUNT:-5}"
BLOCK_ORDER="${BLOCK_ORDER:-}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
BAKE_GOAL_BASE_SEED="${BAKE_GOAL_BASE_SEED:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-2}"

COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS:-4}"

PPO_EPOCHS="${PPO_EPOCHS:-1}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
PPO_CLIP="${PPO_CLIP:-0.15}"
PPO_GAMMA="${PPO_GAMMA:-0.999}"
VF_COEF="${VF_COEF:-0.5}"
POLICY_COEF="${POLICY_COEF:-1.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
KL_COEF="${KL_COEF:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE:-suffix_success}"
LOSS_FOCUS_SUFFIX_LEN="${LOSS_FOCUS_SUFFIX_LEN:-32}"
LOSS_FOCUS_CONTEXT_LEN="${LOSS_FOCUS_CONTEXT_LEN:-96}"
LOSS_FOCUS_WEIGHT="${LOSS_FOCUS_WEIGHT:-4.0}"
LOSS_FOCUS_TOP_K="${LOSS_FOCUS_TOP_K:-32}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-heads}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
FINAL_EVAL_MODEL_MODE="${FINAL_EVAL_MODEL_MODE:-best_eval}"
DISABLE_UPDATES="${DISABLE_UPDATES:-0}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
SKIP_ALL_FINAL_EVAL="${SKIP_ALL_FINAL_EVAL:-0}"

COLLECT_VIDEO_MODE="${COLLECT_VIDEO_MODE:-skip}"
BASELINE_VIDEO_MODE="${BASELINE_VIDEO_MODE:-save}"
EVAL_VIDEO_MODE="${EVAL_VIDEO_MODE:-skip}"
FINAL_EVAL_VIDEO_MODE="${FINAL_EVAL_VIDEO_MODE:-save}"
NORMALIZE_ADVANTAGE="${NORMALIZE_ADVANTAGE:-1}"
CLIP_VLOSS="${CLIP_VLOSS:-1}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid. Set ASSET_DIR to a generated bank directory." >&2
  exit 1
fi

RESUME_PILOT_DIR="${RESUME_PILOT_DIR:-}"
RESUME_BLOCK_DIR=""
RESUME_BLOCK_TAG=""
RESUME_BLOCK_IDX=""
if [[ -n "${RESUME_PILOT_DIR}" ]]; then
  if [[ ! -d "${RESUME_PILOT_DIR}" ]]; then
    echo "RESUME_PILOT_DIR does not exist: ${RESUME_PILOT_DIR}" >&2
    exit 1
  fi
  RESUME_BLOCK_DIR="$(dirname "${RESUME_PILOT_DIR}")"
  RESUME_BLOCK_TAG="$(basename "${RESUME_BLOCK_DIR}")"
  if [[ ! "${RESUME_BLOCK_TAG}" =~ ^block_([0-9]+)$ ]]; then
    echo "Could not infer block tag from RESUME_PILOT_DIR: ${RESUME_PILOT_DIR}" >&2
    exit 1
  fi
  RESUME_BLOCK_IDX="$((10#${BASH_REMATCH[1]}))"
fi

if [[ -z "${OUT_ROOT:-}" ]]; then
  if [[ -n "${RESUME_PILOT_DIR}" ]]; then
    OUT_ROOT="$(dirname "$(dirname "${RESUME_BLOCK_DIR}")")"
else
    OUT_PREFIX="ppo_mine_o2_straight_generalization_blocks_vm"
    if [[ -n "${EXPERIMENT_TAG}" ]]; then
      SAFE_EXPERIMENT_TAG="${EXPERIMENT_TAG//[^A-Za-z0-9_.-]/_}"
      OUT_PREFIX="${OUT_PREFIX}_${SAFE_EXPERIMENT_TAG}"
    fi
    OUT_ROOT="${ROOT_DIR}/outputs/evaluate_rocket/${OUT_PREFIX}_$(date +%Y%m%d_%H%M%S)"
  fi
fi
BLOCK_ROOT="${OUT_ROOT}/blocks"
CHAIN_SUMMARY_PATH="${OUT_ROOT}/chain_summary.json"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

configure_torch_cuda_runtime() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    return 0
  fi
  local python_version
  python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local site_pkg="${CONDA_PREFIX}/lib/python${python_version}/site-packages/nvidia"
  local lib_paths=()
  local subdir
  for subdir in \
    cublas/lib \
    cudnn/lib \
    cuda_runtime/lib \
    curand/lib \
    cusolver/lib \
    cusparse/lib \
    nvjitlink/lib
  do
    if [[ -d "${site_pkg}/${subdir}" ]]; then
      lib_paths+=("${site_pkg}/${subdir}")
    fi
  done
  if (( ${#lib_paths[@]} == 0 )); then
    return 0
  fi
  local joined
  joined="$(IFS=:; echo "${lib_paths[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

mkdir -p "${OUT_ROOT}" "${BLOCK_ROOT}"
configure_torch_cuda_runtime

BASE_COLLECT_PLAN="${ASSET_DIR}/collect_plan.json"
COLLECT_BLOCK_PLAN_DIR="${ASSET_DIR}/collect_block_plans"
EVAL_BANK_DIR="${ASSET_DIR}/eval_bank"
FINAL_EVAL_BANK_DIR="${ASSET_DIR}/final_eval_bank"

if [[ ! -f "${BASE_COLLECT_PLAN}" ]]; then
  echo "Missing collect plan: ${BASE_COLLECT_PLAN}" >&2
  exit 1
fi
if [[ ! -d "${COLLECT_BLOCK_PLAN_DIR}" ]]; then
  echo "Missing collect block plan dir: ${COLLECT_BLOCK_PLAN_DIR}" >&2
  exit 1
fi
if [[ ! -d "${EVAL_BANK_DIR}" ]]; then
  echo "Missing eval fixed bank under ${ASSET_DIR}" >&2
  exit 1
fi
if [[ "${SKIP_ALL_FINAL_EVAL}" != "1" && ! -d "${FINAL_EVAL_BANK_DIR}" ]]; then
  echo "Missing final fixed bank under ${ASSET_DIR}; set SKIP_ALL_FINAL_EVAL=1 to run eval_bank only." >&2
  exit 1
fi

CURRENT_MODEL_PATH="${MODEL_PATH}"
if [[ -f "${CHAIN_SUMMARY_PATH}" ]]; then
  echo "[config] using existing chain summary: ${CHAIN_SUMMARY_PATH}"
else
  export CHAIN_SUMMARY_PATH
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CHAIN_SUMMARY_PATH"])
path.write_text(json.dumps({"blocks": []}, indent=2, ensure_ascii=False), encoding="utf-8")
PY
fi

export CHAIN_SUMMARY_PATH GOAL_PROTOCOL GOAL_SPEC ASSET_DIR
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

summary_path = Path(os.environ["CHAIN_SUMMARY_PATH"])
payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"blocks": []}
goal_spec = str(os.environ.get("GOAL_SPEC") or "")
goal_sha256 = ""
if goal_spec and Path(goal_spec).is_file():
    digest = hashlib.sha256()
    with Path(goal_spec).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    goal_sha256 = digest.hexdigest()
payload["goal_protocol"] = str(os.environ.get("GOAL_PROTOCOL") or "auto_goal")
payload["goal_spec_path"] = goal_spec
payload["goal_spec_sha256"] = goal_sha256
payload["asset_dir"] = str(os.environ.get("ASSET_DIR") or "")
summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "[config] ASSET_DIR=${ASSET_DIR}"
echo "[config] OUT_ROOT=${OUT_ROOT}"
echo "[config] BLOCK_COUNT=${BLOCK_COUNT}"
echo "[config] BLOCK_ORDER=${BLOCK_ORDER:-1..${BLOCK_COUNT}}"
echo "[config] ITERS_PER_BLOCK=${ITERS_PER_BLOCK}"
echo "[config] COLLECT_EPISODES=${COLLECT_EPISODES}"
echo "[config] EVAL_EPISODES=${EVAL_EPISODES}"
echo "[config] MODEL_PATH=${MODEL_PATH}"
echo "[config] GOAL_PROTOCOL=${GOAL_PROTOCOL}"
echo "[config] GOAL_SPEC=${GOAL_SPEC:-auto_goal_or_baked_task_group}"
echo "[config] DISABLE_UPDATES=${DISABLE_UPDATES}"
echo "[config] SKIP_ALL_FINAL_EVAL=${SKIP_ALL_FINAL_EVAL}"
echo "[config] UPDATE_FRAGMENT_BATCH_SIZE=${UPDATE_FRAGMENT_BATCH_SIZE}"
echo "[config] LOSS_FOCUS_MODE=${LOSS_FOCUS_MODE}"
echo "[config] COLLECT_BLOCK_PLAN_DIR=${COLLECT_BLOCK_PLAN_DIR}"
if [[ -n "${RESUME_PILOT_DIR}" ]]; then
  echo "[config] RESUME_PILOT_DIR=${RESUME_PILOT_DIR}"
  echo "[config] RESUME_BLOCK_TAG=${RESUME_BLOCK_TAG}"
fi

START_BLOCK_IDX=1
if [[ -n "${RESUME_BLOCK_IDX}" ]]; then
  START_BLOCK_IDX="${RESUME_BLOCK_IDX}"
fi

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

if [[ -n "${BLOCK_ORDER}" ]]; then
  split_csv "${BLOCK_ORDER}"
  BLOCK_SEQUENCE=( "${SPLIT_ITEMS[@]}" )
else
  BLOCK_SEQUENCE=()
  for (( BLOCK_IDX=1; BLOCK_IDX<=BLOCK_COUNT; BLOCK_IDX++ )); do
    BLOCK_SEQUENCE+=( "${BLOCK_IDX}" )
  done
fi

GOAL_ARGS=()
if [[ -n "${GOAL_SPEC}" ]]; then
  if [[ ! -e "${GOAL_SPEC}" ]]; then
    echo "GOAL_SPEC does not exist: ${GOAL_SPEC}" >&2
    exit 1
  fi
  GOAL_ARGS=(--goal-spec "${GOAL_SPEC}")
else
  GOAL_ARGS=(--auto-goal)
fi
TOTAL_CHAIN_STEPS="${#BLOCK_SEQUENCE[@]}"
if (( TOTAL_CHAIN_STEPS < 1 )); then
  echo "BLOCK_ORDER resolved to zero steps." >&2
  exit 1
fi

EFFECTIVE_MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS}"
if [[ "${DISABLE_UPDATES}" == "1" ]]; then
  EFFECTIVE_MIN_SUCCESSFUL_FRAGMENTS="$(( COLLECT_EPISODES + 1 ))"
fi

for (( CHAIN_STEP_IDX=START_BLOCK_IDX; CHAIN_STEP_IDX<=TOTAL_CHAIN_STEPS; CHAIN_STEP_IDX++ )); do
  SOURCE_BLOCK_IDX="${BLOCK_SEQUENCE[$(( CHAIN_STEP_IDX - 1 ))]}"
  if [[ ! "${SOURCE_BLOCK_IDX}" =~ ^[0-9]+$ ]]; then
    echo "Invalid block index in BLOCK_ORDER: ${SOURCE_BLOCK_IDX}" >&2
    exit 1
  fi
  if (( SOURCE_BLOCK_IDX < 1 || SOURCE_BLOCK_IDX > BLOCK_COUNT )); then
    echo "Block index out of range in BLOCK_ORDER: ${SOURCE_BLOCK_IDX}; valid range is 1..${BLOCK_COUNT}" >&2
    exit 1
  fi
  BLOCK_TAG="$(printf 'block_%03d' "${CHAIN_STEP_IDX}")"
  SOURCE_BLOCK_TAG="$(printf 'block_%03d' "${SOURCE_BLOCK_IDX}")"
  BLOCK_OUT_DIR="${BLOCK_ROOT}/${BLOCK_TAG}"
  BLOCK_PLAN_PATH="${COLLECT_BLOCK_PLAN_DIR}/${SOURCE_BLOCK_TAG}_collect_plan.json"

  mkdir -p "${BLOCK_OUT_DIR}"
  if [[ ! -f "${BLOCK_PLAN_PATH}" ]]; then
    echo "Missing collect block plan: ${BLOCK_PLAN_PATH}" >&2
    exit 1
  fi

  echo
  echo "==================== ${BLOCK_TAG} ===================="
  echo "[block] chain_step=${CHAIN_STEP_IDX}/${TOTAL_CHAIN_STEPS}"
  echo "[block] source_block=${SOURCE_BLOCK_TAG}"
  echo "[block] input_model=${CURRENT_MODEL_PATH}"
  echo "[block] collect_plan=${BLOCK_PLAN_PATH}"

  EXTRA_FINAL_EVAL_ARGS=()
  if [[ "${SKIP_ALL_FINAL_EVAL}" == "1" || CHAIN_STEP_IDX -lt TOTAL_CHAIN_STEPS ]]; then
    EXTRA_FINAL_EVAL_ARGS+=(--skip-final-eval)
  fi
  BLOCK_BASELINE_VIDEO_MODE="${BASELINE_VIDEO_MODE}"
  if (( CHAIN_STEP_IDX > 1 )); then
    BLOCK_BASELINE_VIDEO_MODE="skip"
  fi

  EXTRA_RESUME_ARGS=()
  RESUMING_THIS_BLOCK=0
  if [[ -n "${RESUME_PILOT_DIR}" && "${BLOCK_TAG}" == "${RESUME_BLOCK_TAG}" ]]; then
    EXTRA_RESUME_ARGS+=(--resume-pilot-dir "${RESUME_PILOT_DIR}")
    RESUMING_THIS_BLOCK=1
  fi

  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" \
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot \
    "${EXTRA_RESUME_ARGS[@]}" \
    --env-source rocket2_official \
    --task-group "${TASK_GROUP_NAME}_${BLOCK_TAG}" \
    --task-group-path "${SOURCE_ENV_CONF_DIR}" \
    --worldgen-source-dir "${SOURCE_ENV_CONF_DIR}" \
    --worldgen-plan-json "${BLOCK_PLAN_PATH}" \
    --mine-layout-backend "${MINE_LAYOUT_BACKEND}" \
    --mine-anchor-mode "${MINE_ANCHOR_MODE}" \
    --tasks "${TASK}" \
    "${GOAL_ARGS[@]}" \
    --collect-world-mode plan_exact \
    --eval-world-mode fixed_bank \
    --final-eval-world-mode fixed_bank \
    --collect-world-instances 1 \
    --eval-world-instances 1 \
    --final-eval-world-instances 1 \
    --collect-parallel-workers "${COLLECT_WORKERS}" \
    --eval-parallel-workers "${EVAL_WORKERS}" \
    --final-eval-parallel-workers "${FINAL_EVAL_WORKERS}" \
    --collect-episodes-per-task "${COLLECT_EPISODES}" \
    --eval-episodes-per-task "${EVAL_EPISODES}" \
    --train-iters "${ITERS_PER_BLOCK}" \
    --base-seed "${BASE_SEED}" \
    --seed-step "${SEED_STEP}" \
    --sampling-base-seed "${SAMPLING_BASE_SEED}" \
    --sampling-seed-step "${SAMPLING_SEED_STEP}" \
    --bake-goal-base-seed "${BAKE_GOAL_BASE_SEED}" \
    --bake-goal-episode-retries "${BAKE_GOAL_EPISODE_RETRIES}" \
    --episode-retries "${EPISODE_RETRIES}" \
    --step-budget-override "${STEP_BUDGET}" \
    --model-path "${CURRENT_MODEL_PATH}" \
    --cfg-coef "${CFG_COEF}" \
    --cfg-policy-mode "${CFG_POLICY_MODE}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --ppo-learning-rate "${PPO_LEARNING_RATE}" \
    --ppo-clip "${PPO_CLIP}" \
    --ppo-gamma "${PPO_GAMMA}" \
    --vf-coef "${VF_COEF}" \
    --policy-coef "${POLICY_COEF}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --kl-coef "${KL_COEF}" \
    --max-grad-norm "${MAX_GRAD_NORM}" \
    --update-fragment-batch-size "${UPDATE_FRAGMENT_BATCH_SIZE}" \
    --loss-focus-mode "${LOSS_FOCUS_MODE}" \
    --loss-focus-suffix-len "${LOSS_FOCUS_SUFFIX_LEN}" \
    --loss-focus-context-len "${LOSS_FOCUS_CONTEXT_LEN}" \
    --loss-focus-weight "${LOSS_FOCUS_WEIGHT}" \
    --loss-focus-top-k "${LOSS_FOCUS_TOP_K}" \
    --trainable-scope "${TRAINABLE_SCOPE}" \
    --min-successful-fragments "${EFFECTIVE_MIN_SUCCESSFUL_FRAGMENTS}" \
    --final-eval-model-mode "${FINAL_EVAL_MODEL_MODE}" \
    --baseline-video-mode "${BLOCK_BASELINE_VIDEO_MODE}" \
    --collect-video-mode "${COLLECT_VIDEO_MODE}" \
    --eval-video-mode "${EVAL_VIDEO_MODE}" \
    --final-eval-video-mode "${FINAL_EVAL_VIDEO_MODE}" \
    --eval-fixed-bank-dir "${EVAL_BANK_DIR}" \
    --final-eval-fixed-bank-dir "${FINAL_EVAL_BANK_DIR}" \
    --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}" \
    --kl-anchor-model-path "${KL_ANCHOR_MODEL_PATH}" \
    "${EXTRA_FINAL_EVAL_ARGS[@]}" \
    $( [[ "${NORMALIZE_ADVANTAGE}" == "1" ]] && printf '%s' "--normalize-advantage" ) \
    $( [[ "${CLIP_VLOSS}" == "1" ]] && printf '%s' "--clip-vloss" ) \
    $( [[ "${STOP_ON_SUCCESS}" == "1" ]] && printf '%s' "--stop-on-success" ) \
    $( [[ "${SKIP_VIDEO}" == "1" ]] && printf '%s' "--skip-video" ) \
    --out-dir "${BLOCK_OUT_DIR}"

  if (( RESUMING_THIS_BLOCK == 1 )); then
    BLOCK_RUN_DIR="${RESUME_PILOT_DIR}"
  else
    BLOCK_RUN_DIR="$(find "${BLOCK_OUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  fi
  if [[ -z "${BLOCK_RUN_DIR}" || ! -d "${BLOCK_RUN_DIR}" ]]; then
    echo "Could not locate block run directory under ${BLOCK_OUT_DIR}" >&2
    exit 1
  fi

  BLOCK_SUMMARY_PATH="${BLOCK_RUN_DIR}/pilot_summary.json"
  if [[ ! -f "${BLOCK_SUMMARY_PATH}" ]]; then
    echo "Missing block summary: ${BLOCK_SUMMARY_PATH}" >&2
    exit 1
  fi

  export BLOCK_SUMMARY_PATH CHAIN_SUMMARY_PATH BLOCK_TAG BLOCK_RUN_DIR SOURCE_BLOCK_TAG CHAIN_STEP_IDX DISABLE_UPDATES
  NEXT_MODEL_PATH="$("${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["BLOCK_SUMMARY_PATH"])
chain_summary_path = Path(os.environ["CHAIN_SUMMARY_PATH"])
block_tag = os.environ["BLOCK_TAG"]

summary = json.loads(summary_path.read_text(encoding="utf-8"))
best_eval = dict(summary.get("best_eval_selection") or {})
final_eval = dict(summary.get("final_eval_summary") or {})
next_model = str(best_eval.get("model_path") or summary.get("final_model_path") or "")
if not next_model:
    raise SystemExit(f"Could not resolve next model path from {summary_path}")

chain = json.loads(chain_summary_path.read_text(encoding="utf-8"))
chain.setdefault("blocks", []).append(
    {
        "block_tag": block_tag,
        "chain_step": int(os.environ.get("CHAIN_STEP_IDX", "0") or 0),
        "source_block_tag": os.environ.get("SOURCE_BLOCK_TAG", ""),
        "block_run_dir": os.environ["BLOCK_RUN_DIR"],
        "pilot_summary_path": str(summary_path),
        "next_model_path": next_model,
        "disable_updates": bool(int(os.environ.get("DISABLE_UPDATES", "0") or 0)),
        "best_eval_selection": best_eval,
        "final_eval_summary": final_eval,
    }
)
chain_summary_path.write_text(json.dumps(chain, indent=2, ensure_ascii=False), encoding="utf-8")
print(next_model)
PY
)"

  CURRENT_MODEL_PATH="${NEXT_MODEL_PATH}"
  echo "[block] next_model=${CURRENT_MODEL_PATH}"
done

echo
echo "==================== done ===================="
echo "[result] OUT_ROOT=${OUT_ROOT}"
echo "[result] CHAIN_SUMMARY_PATH=${CHAIN_SUMMARY_PATH}"
echo "[result] FINAL_MODEL_PATH=${CURRENT_MODEL_PATH}"
