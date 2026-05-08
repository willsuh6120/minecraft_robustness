#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/minestudio/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/minestudio/bin/python"
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
TASK_GROUP_NAME="${TASK_GROUP_NAME:-straight_O2_single_env_smoke}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-full}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-}"
KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-}"
RESUME_PILOT_DIR="${RESUME_PILOT_DIR:-}"
if [[ -z "${KL_ANCHOR_MODEL_PATH}" ]]; then
  KL_ANCHOR_MODEL_PATH="${MODEL_PATH}"
fi
if [[ "${CFG_POLICY_MODE}" == "frozen_base" && -z "${CFG_BASE_REF_MODEL_PATH}" ]]; then
  CFG_BASE_REF_MODEL_PATH="${KL_ANCHOR_MODEL_PATH}"
fi

COLLECT_EPISODES="${COLLECT_EPISODES:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
STEP_BUDGET="${STEP_BUDGET:-150}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-}"
PLAN_BASE_SEED="${PLAN_BASE_SEED:-20260421}"
GOAL_BASE_SEED="${GOAL_BASE_SEED:-1}"
MINE_OCCLUDER_FRONT_DEPTH="${MINE_OCCLUDER_FRONT_DEPTH:-}"
MINE_OCCLUDER_SIDE_SPAN="${MINE_OCCLUDER_SIDE_SPAN:-}"
MINE_OCCLUDER_TOP_ROW="${MINE_OCCLUDER_TOP_ROW:-}"
MINE_OCCLUDER_BOTTOM_ROW="${MINE_OCCLUDER_BOTTOM_ROW:-}"
MINE_OCCLUDER_LEAF_MATERIAL="${MINE_OCCLUDER_LEAF_MATERIAL:-}"
MINE_OCCLUDER_GLASS_MATERIAL="${MINE_OCCLUDER_GLASS_MATERIAL:-}"
COLLECT_WORKERS="${COLLECT_WORKERS:-2}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS:-${EVAL_WORKERS}}"
MAX_RESETTING_ENV_COUNT_DEFAULT="${COLLECT_WORKERS}"
if (( EVAL_WORKERS > MAX_RESETTING_ENV_COUNT_DEFAULT )); then
  MAX_RESETTING_ENV_COUNT_DEFAULT="${EVAL_WORKERS}"
fi
if (( FINAL_EVAL_WORKERS > MAX_RESETTING_ENV_COUNT_DEFAULT )); then
  MAX_RESETTING_ENV_COUNT_DEFAULT="${FINAL_EVAL_WORKERS}"
fi
if (( MAX_RESETTING_ENV_COUNT_DEFAULT < 4 )); then
  MAX_RESETTING_ENV_COUNT_DEFAULT=4
fi

PPO_EPOCHS="${PPO_EPOCHS:-1}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-1e-5}"
PPO_CLIP="${PPO_CLIP:-0.2}"
PPO_GAMMA="${PPO_GAMMA:-0.99}"
VF_COEF="${VF_COEF:-0.5}"
POLICY_COEF="${POLICY_COEF:-1.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
KL_COEF="${KL_COEF:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-1}"
LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE:-uniform}"
LOSS_FOCUS_SUFFIX_LEN="${LOSS_FOCUS_SUFFIX_LEN:-32}"
LOSS_FOCUS_CONTEXT_LEN="${LOSS_FOCUS_CONTEXT_LEN:-96}"
LOSS_FOCUS_WEIGHT="${LOSS_FOCUS_WEIGHT:-4.0}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-heads}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-0}"
FINAL_EVAL_MODEL_MODE="${FINAL_EVAL_MODEL_MODE:-last}"
NORMALIZE_ADVANTAGE="${NORMALIZE_ADVANTAGE:-1}"
CLIP_VLOSS="${CLIP_VLOSS:-1}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
BASELINE_VIDEO_MODE="${BASELINE_VIDEO_MODE:-inherit}"
COLLECT_VIDEO_MODE="${COLLECT_VIDEO_MODE:-inherit}"
EVAL_VIDEO_MODE="${EVAL_VIDEO_MODE:-inherit}"
FINAL_EVAL_VIDEO_MODE="${FINAL_EVAL_VIDEO_MODE:-inherit}"
SAVE_DEBUG_ASSETS="${SAVE_DEBUG_ASSETS:-0}"

if [[ -n "${RESUME_PILOT_DIR}" && -z "${OUT_ROOT:-}" ]]; then
  OUT_ROOT="$(cd "$(dirname "$(dirname "${RESUME_PILOT_DIR}")")" && pwd)"
fi
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/ppo_straight_O2_single_env_smoke_$(date +%Y%m%d_%H%M%S)}"
PLAN_JSON_PATH="${OUT_ROOT}/worldgen_plan.json"
WORLDGEN_ROOT="${OUT_ROOT}/worldgen"
PILOT_ROOT="${OUT_ROOT}/pilot"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export MINESTUDIO_MAX_RESETTING_ENV_COUNT="${MINESTUDIO_MAX_RESETTING_ENV_COUNT:-${MAX_RESETTING_ENV_COUNT_DEFAULT}}"
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

run_with_xvfb() {
  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" "$@"
}

find_latest_subdir() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

mkdir -p "${OUT_ROOT}" "${WORLDGEN_ROOT}" "${PILOT_ROOT}"
configure_torch_cuda_runtime

echo "[config] ROOT_DIR=${ROOT_DIR}"
echo "[config] PYTHON_BIN=${PYTHON_BIN}"
echo "[config] OUT_ROOT=${OUT_ROOT}"
echo "[config] TASK=${TASK}"
echo "[config] SOURCE_ENV_CONF_DIR=${SOURCE_ENV_CONF_DIR}"
echo "[config] MODEL_PATH=${MODEL_PATH}"
echo "[config] CFG_COEF=${CFG_COEF}"
echo "[config] CFG_POLICY_MODE=${CFG_POLICY_MODE}"
echo "[config] CFG_BASE_REF_MODEL_PATH=${CFG_BASE_REF_MODEL_PATH}"
echo "[config] KL_ANCHOR_MODEL_PATH=${KL_ANCHOR_MODEL_PATH}"
echo "[config] RESUME_PILOT_DIR=${RESUME_PILOT_DIR}"
echo "[config] COLLECT_EPISODES=${COLLECT_EPISODES}"
echo "[config] EVAL_EPISODES=${EVAL_EPISODES}"
echo "[config] TRAIN_ITERS=${TRAIN_ITERS}"
echo "[config] STEP_BUDGET=${STEP_BUDGET}"
echo "[config] BASE_SEED=${BASE_SEED}"
echo "[config] SEED_STEP=${SEED_STEP}"
echo "[config] SAMPLING_BASE_SEED=${SAMPLING_BASE_SEED}"
echo "[config] SAMPLING_SEED_STEP=${SAMPLING_SEED_STEP}"
echo "[config] MINE_OCCLUDER_FRONT_DEPTH=${MINE_OCCLUDER_FRONT_DEPTH}"
echo "[config] MINE_OCCLUDER_SIDE_SPAN=${MINE_OCCLUDER_SIDE_SPAN}"
echo "[config] MINE_OCCLUDER_TOP_ROW=${MINE_OCCLUDER_TOP_ROW}"
echo "[config] MINE_OCCLUDER_BOTTOM_ROW=${MINE_OCCLUDER_BOTTOM_ROW}"
echo "[config] COLLECT_WORKERS=${COLLECT_WORKERS}"
echo "[config] EVAL_WORKERS=${EVAL_WORKERS}"
echo "[config] FINAL_EVAL_WORKERS=${FINAL_EVAL_WORKERS}"
echo "[config] MINESTUDIO_MAX_RESETTING_ENV_COUNT=${MINESTUDIO_MAX_RESETTING_ENV_COUNT}"
echo "[config] PPO_GAMMA=${PPO_GAMMA}"
echo "[config] MAX_GRAD_NORM=${MAX_GRAD_NORM}"
echo "[config] LOSS_FOCUS_MODE=${LOSS_FOCUS_MODE}"
echo "[config] LOSS_FOCUS_SUFFIX_LEN=${LOSS_FOCUS_SUFFIX_LEN}"
echo "[config] LOSS_FOCUS_CONTEXT_LEN=${LOSS_FOCUS_CONTEXT_LEN}"
echo "[config] LOSS_FOCUS_WEIGHT=${LOSS_FOCUS_WEIGHT}"
echo "[config] TRAINABLE_SCOPE=${TRAINABLE_SCOPE}"
echo "[config] MIN_SUCCESSFUL_FRAGMENTS=${MIN_SUCCESSFUL_FRAGMENTS}"
echo "[config] FINAL_EVAL_MODEL_MODE=${FINAL_EVAL_MODEL_MODE}"
echo "[config] BASELINE_VIDEO_MODE=${BASELINE_VIDEO_MODE}"
echo "[config] COLLECT_VIDEO_MODE=${COLLECT_VIDEO_MODE}"
echo "[config] EVAL_VIDEO_MODE=${EVAL_VIDEO_MODE}"
echo "[config] FINAL_EVAL_VIDEO_MODE=${FINAL_EVAL_VIDEO_MODE}"

export PLAN_JSON_PATH TASK PLAN_BASE_SEED
export MINE_OCCLUDER_FRONT_DEPTH MINE_OCCLUDER_SIDE_SPAN
export MINE_OCCLUDER_TOP_ROW MINE_OCCLUDER_BOTTOM_ROW
export MINE_OCCLUDER_LEAF_MATERIAL MINE_OCCLUDER_GLASS_MATERIAL
if [[ -z "${RESUME_PILOT_DIR}" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

from minestudio.tutorials.inference.evaluate_rocket.interaction_world_factors import (
    classify_factor_split,
    factor_levels_to_worldgen_suggestions,
    hard_factor_count,
)

task = os.environ["TASK"]
factor_levels = {"R": 0, "H": 0, "O": 2, "C": 0, "P": 0, "A": 0}
suggestions = factor_levels_to_worldgen_suggestions(task, factor_levels)
suggestions["notes"] = (
    "Single-environment PPO smoke test: fixed straight tunnel with O2 only. "
    "The exact same baked world is reused for collect and eval every iteration."
)
suggestions["mine_blueprint_id"] = "straight_tunnel"
if str(os.environ.get("MINE_OCCLUDER_FRONT_DEPTH", "")).strip():
    suggestions["mine_occluder_front_depth"] = int(os.environ["MINE_OCCLUDER_FRONT_DEPTH"])
if str(os.environ.get("MINE_OCCLUDER_SIDE_SPAN", "")).strip():
    suggestions["mine_occluder_side_span"] = int(os.environ["MINE_OCCLUDER_SIDE_SPAN"])
if str(os.environ.get("MINE_OCCLUDER_TOP_ROW", "")).strip():
    suggestions["mine_occluder_top_row"] = [
        item.strip().lower() for item in os.environ["MINE_OCCLUDER_TOP_ROW"].split(",") if item.strip()
    ]
if str(os.environ.get("MINE_OCCLUDER_BOTTOM_ROW", "")).strip():
    suggestions["mine_occluder_bottom_row"] = [
        item.strip().lower() for item in os.environ["MINE_OCCLUDER_BOTTOM_ROW"].split(",") if item.strip()
    ]
if str(os.environ.get("MINE_OCCLUDER_LEAF_MATERIAL", "")).strip():
    suggestions["mine_occluder_leaf_material"] = str(os.environ["MINE_OCCLUDER_LEAF_MATERIAL"]).strip()
if str(os.environ.get("MINE_OCCLUDER_GLASS_MATERIAL", "")).strip():
    suggestions["mine_occluder_glass_material"] = str(os.environ["MINE_OCCLUDER_GLASS_MATERIAL"]).strip()

row = {
    "task_config_name": task,
    "task_key": task,
    "primary_failure_mode_majority": "ppo_smoke_fixed_single_env",
    "primary_factors_majority": ["O"],
    "severity_majority": 2,
    "factor_levels": factor_levels,
    "layout_seed": int(os.environ["PLAN_BASE_SEED"]),
    "template_index": 0,
    "requested_split_label": classify_factor_split(task, factor_levels),
    "computed_split_label": classify_factor_split(task, factor_levels),
    "hard_factor_count": int(hard_factor_count(task, factor_levels)),
    "trainable_with_rl_majority": True,
    "world_generation_suggestions": suggestions,
}

plan_path = Path(os.environ["PLAN_JSON_PATH"])
plan_path.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"plan_json": str(plan_path), "row": row}, ensure_ascii=False))
PY

  echo
  echo "==================== worldgen ===================="
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen \
    --plan-json "${PLAN_JSON_PATH}" \
    --env-conf-dir "${SOURCE_ENV_CONF_DIR}" \
    --out-dir "${WORLDGEN_ROOT}" \
    --mine-layout-backend "${MINE_LAYOUT_BACKEND}" \
    --mine-anchor-mode "${MINE_ANCHOR_MODE}"

  WG_DIR="$(find_latest_subdir "${WORLDGEN_ROOT}")"
  if [[ -z "${WG_DIR}" ]]; then
    echo "Failed to locate generated task-group directory under ${WORLDGEN_ROOT}" >&2
    exit 1
  fi

  echo "[worldgen] WG_DIR=${WG_DIR}"

  echo
  echo "==================== bake goals ===================="
  BAKE_CMD=(
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets
    --task-group "${TASK_GROUP_NAME}"
    --task-group-path "${WG_DIR}"
    --protocol ours_v1
    --tasks "${TASK}"
    --base-seed "${GOAL_BASE_SEED}"
    --episode-retries "${EPISODE_RETRIES}"
    --overwrite
    --refresh-task-configs
  )
  if [[ "${SAVE_DEBUG_ASSETS}" == "1" ]]; then
    BAKE_CMD+=(--save-debug-assets)
  fi
  run_with_xvfb "${BAKE_CMD[@]}"
else
  WG_DIR="$(find_latest_subdir "${WORLDGEN_ROOT}")"
  if [[ -z "${WG_DIR}" ]]; then
    echo "Failed to locate existing generated task-group directory under ${WORLDGEN_ROOT} for resume" >&2
    exit 1
  fi
  echo "[resume] Reusing WG_DIR=${WG_DIR}"
fi

echo
echo "==================== pilot ===================="
PILOT_CMD=(
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot
  --env-source rocket2_official
  --task-group "${TASK_GROUP_NAME}"
  --task-group-path "${WG_DIR}"
  --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
  --mine-anchor-mode "${MINE_ANCHOR_MODE}"
  --tasks "${TASK}"
  --auto-goal
  --collect-world-mode static
  --eval-world-mode static
  --final-eval-world-mode static
  --collect-world-instances 1
  --eval-world-instances 1
  --final-eval-world-instances 1
  --collect-parallel-workers "${COLLECT_WORKERS}"
  --eval-parallel-workers "${EVAL_WORKERS}"
  --final-eval-parallel-workers "${FINAL_EVAL_WORKERS}"
  --collect-episodes-per-task "${COLLECT_EPISODES}"
  --eval-episodes-per-task "${EVAL_EPISODES}"
  --train-iters "${TRAIN_ITERS}"
  --base-seed "${BASE_SEED}"
  --seed-step "${SEED_STEP}"
  --episode-retries "${EPISODE_RETRIES}"
  --step-budget-override "${STEP_BUDGET}"
  --model-path "${MODEL_PATH}"
  --cfg-coef "${CFG_COEF}"
  --cfg-policy-mode "${CFG_POLICY_MODE}"
  --ppo-epochs "${PPO_EPOCHS}"
  --ppo-learning-rate "${PPO_LEARNING_RATE}"
  --ppo-clip "${PPO_CLIP}"
  --ppo-gamma "${PPO_GAMMA}"
  --vf-coef "${VF_COEF}"
  --policy-coef "${POLICY_COEF}"
  --entropy-coef "${ENTROPY_COEF}"
  --kl-coef "${KL_COEF}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --update-fragment-batch-size "${UPDATE_FRAGMENT_BATCH_SIZE}"
  --loss-focus-mode "${LOSS_FOCUS_MODE}"
  --loss-focus-suffix-len "${LOSS_FOCUS_SUFFIX_LEN}"
  --loss-focus-context-len "${LOSS_FOCUS_CONTEXT_LEN}"
  --loss-focus-weight "${LOSS_FOCUS_WEIGHT}"
  --trainable-scope "${TRAINABLE_SCOPE}"
  --min-successful-fragments "${MIN_SUCCESSFUL_FRAGMENTS}"
  --final-eval-model-mode "${FINAL_EVAL_MODEL_MODE}"
  --baseline-video-mode "${BASELINE_VIDEO_MODE}"
  --collect-video-mode "${COLLECT_VIDEO_MODE}"
  --eval-video-mode "${EVAL_VIDEO_MODE}"
  --final-eval-video-mode "${FINAL_EVAL_VIDEO_MODE}"
  --out-dir "${PILOT_ROOT}"
)
if [[ -n "${SAMPLING_BASE_SEED}" ]]; then
  PILOT_CMD+=(--sampling-base-seed "${SAMPLING_BASE_SEED}")
fi
if [[ -n "${SAMPLING_SEED_STEP}" ]]; then
  PILOT_CMD+=(--sampling-seed-step "${SAMPLING_SEED_STEP}")
fi
if [[ -n "${CFG_BASE_REF_MODEL_PATH}" ]]; then
  PILOT_CMD+=(--cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}")
fi
if [[ -n "${KL_ANCHOR_MODEL_PATH}" ]]; then
  PILOT_CMD+=(--kl-anchor-model-path "${KL_ANCHOR_MODEL_PATH}")
fi
if [[ -n "${RESUME_PILOT_DIR}" ]]; then
  PILOT_CMD+=(--resume-pilot-dir "${RESUME_PILOT_DIR}")
fi
if [[ "${NORMALIZE_ADVANTAGE}" == "1" ]]; then
  PILOT_CMD+=(--normalize-advantage)
fi
if [[ "${CLIP_VLOSS}" == "1" ]]; then
  PILOT_CMD+=(--clip-vloss)
fi
if [[ "${STOP_ON_SUCCESS}" == "1" ]]; then
  PILOT_CMD+=(--stop-on-success)
fi
if [[ "${SKIP_VIDEO}" == "1" ]]; then
  PILOT_CMD+=(--skip-video)
fi
if [[ "${SAVE_DEBUG_ASSETS}" == "1" ]]; then
  PILOT_CMD+=(--save-debug-assets)
fi

run_with_xvfb "${PILOT_CMD[@]}"

if [[ -n "${RESUME_PILOT_DIR}" ]]; then
  PILOT_RUN_DIR="${RESUME_PILOT_DIR}"
else
  PILOT_RUN_DIR="$(find_latest_subdir "${PILOT_ROOT}")"
  if [[ -z "${PILOT_RUN_DIR}" ]]; then
    echo "Failed to locate pilot run directory under ${PILOT_ROOT}" >&2
    exit 1
  fi
fi

export PILOT_RUN_DIR WG_DIR OUT_ROOT
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

pilot_run_dir = Path(os.environ["PILOT_RUN_DIR"])
pilot_summary = json.loads((pilot_run_dir / "pilot_summary.json").read_text(encoding="utf-8"))

def first_row(summary_dict):
    if not isinstance(summary_dict, dict) or not summary_dict:
        return {}
    _, row = next(iter(summary_dict.items()))
    return dict(row)

curve = []
curve.append({
    "phase": "baseline",
    **first_row(pilot_summary.get("baseline_summary")),
})
for item in pilot_summary.get("iteration_summaries", []):
    curve.append({
        "phase": f"iter_{int(item.get('iteration', 0)):03d}",
        **first_row(item.get("eval_summary")),
    })
curve.append({
    "phase": "final",
    **first_row(pilot_summary.get("final_eval_summary")),
})

curve_path = pilot_run_dir / "smoke_curve.json"
curve_path.write_text(json.dumps(curve, indent=2, ensure_ascii=False), encoding="utf-8")

paths = {
    "out_root": os.environ["OUT_ROOT"],
    "world_dir": os.environ["WG_DIR"],
    "pilot_run_dir": str(pilot_run_dir),
    "pilot_summary_json": str(pilot_run_dir / "pilot_summary.json"),
    "smoke_curve_json": str(curve_path),
}
(pilot_run_dir / "smoke_paths.json").write_text(json.dumps(paths, indent=2, ensure_ascii=False), encoding="utf-8")

print("=== smoke_curve ===")
print(json.dumps(curve, indent=2, ensure_ascii=False))
print("=== smoke_paths ===")
print(json.dumps(paths, indent=2, ensure_ascii=False))
PY
