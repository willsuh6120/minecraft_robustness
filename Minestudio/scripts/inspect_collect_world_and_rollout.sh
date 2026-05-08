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

RUN_ROOT="${RUN_ROOT:-${1:-}}"
ITERATION="${ITERATION:-${2:-}}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
TASK="${TASK:-mine_coal}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/collect_world_video_inspect}"

usage() {
  cat <<'EOF'
usage:
  RUN_ROOT=/path/to/pilot_run ITERATION=1 bash scripts/inspect_collect_world_and_rollout.sh

or:
  bash scripts/inspect_collect_world_and_rollout.sh /path/to/pilot_run 1
EOF
}

if [[ -z "${RUN_ROOT}" || -z "${ITERATION}" ]]; then
  usage >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT%/}"
ITER_TAG="$(printf 'iter_%03d' "${ITERATION}")"
COLLECT_WORLD_ROOT="${RUN_ROOT}/worldgen/${ITER_TAG}_collect/instance_000"
GENERATED_ROOT="${COLLECT_WORLD_ROOT}/generated_task_groups"
PLAN_JSON="${COLLECT_WORLD_ROOT}/worldgen_plan.json"

if [[ ! -d "${GENERATED_ROOT}" ]]; then
  echo "Collect generated_task_groups not found: ${GENERATED_ROOT}" >&2
  exit 1
fi

TASK_GROUP_PATH="$(find "${GENERATED_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [[ -z "${TASK_GROUP_PATH}" || ! -d "${TASK_GROUP_PATH}" ]]; then
  echo "Could not locate collect task_group_path under ${GENERATED_ROOT}" >&2
  exit 1
fi

MANIFEST_PATH="${TASK_GROUP_PATH}/worldgen_manifest.json"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "worldgen_manifest.json not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
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

configure_torch_cuda_runtime

mkdir -p "${OUT_ROOT}"
RUN_OUT_DIR="${OUT_ROOT}/$(basename "${RUN_ROOT}")_${ITER_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_OUT_DIR}"

echo "[inspect-collect-world] run_root=${RUN_ROOT}"
echo "[inspect-collect-world] iteration=${ITERATION}"
echo "[inspect-collect-world] task_group_path=${TASK_GROUP_PATH}"
echo "[inspect-collect-world] manifest_path=${MANIFEST_PATH}"
echo "[inspect-collect-world] out_dir=${RUN_OUT_DIR}"

"${PYTHON_BIN}" - <<'PY' "${MANIFEST_PATH}" "${PLAN_JSON}"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
task = manifest["tasks"][0]
visibility = (task.get("realized_factor_metrics") or {}).get("visibility") or {}
payload = {
    "manifest_path": str(manifest_path),
    "plan_json": str(plan_path),
    "layout_seed": task.get("layout_seed"),
    "blueprint_id": task.get("blueprint_id"),
    "spawn_position": task.get("spawn_position"),
    "target_world_center": task.get("target_world_center"),
    "preferred_face": task.get("preferred_face"),
    "goal_camera_preferred_face": task.get("goal_camera_preferred_face"),
    "visibility_variant": {
        "variant_name": visibility.get("variant_name"),
        "front_depth": visibility.get("front_depth"),
        "side_span": visibility.get("side_span"),
        "side_pattern": visibility.get("side_pattern"),
        "center_material": visibility.get("center_material"),
        "left_material": visibility.get("left_material"),
        "right_material": visibility.get("right_material"),
    },
    "world_generation_suggestions": task.get("world_generation_suggestions"),
}
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

"${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" \
"${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout \
  --env-source rocket2_official \
  --protocol ours_v1 \
  --task-group "${TASK}_${ITER_TAG}_inspect" \
  --task-group-path "${TASK_GROUP_PATH}" \
  --tasks "${TASK}" \
  --episodes-per-task 1 \
  --base-seed "${BASE_SEED}" \
  --seed-step "${SEED_STEP}" \
  --sampling-base-seed "${SAMPLING_BASE_SEED}" \
  --sampling-seed-step "${SAMPLING_SEED_STEP}" \
  --episode-retries "${EPISODE_RETRIES}" \
  --step-budget-override "${STEP_BUDGET}" \
  --model-path "${MODEL_PATH}" \
  --auto-goal \
  --cfg-coef "${CFG_COEF}" \
  --cfg-policy-mode "${CFG_POLICY_MODE}" \
  --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}" \
  --stop-on-success \
  --out-dir "${RUN_OUT_DIR}"

echo "[inspect-collect-world] done"
echo "[inspect-collect-world] video_root=${RUN_OUT_DIR}"
