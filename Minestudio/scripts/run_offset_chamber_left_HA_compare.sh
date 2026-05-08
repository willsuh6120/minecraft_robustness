#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

PROTOCOL="${PROTOCOL:-ours_v1}"
TASK="${TASK:-mine_coal}"
ENV_SOURCE="${ENV_SOURCE:-rocket2_official}"
ENV_CONF_DIR="${ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
USE_XVFB="${USE_XVFB:-1}"

REFERENCE_PLAN="${REFERENCE_PLAN:-${ROOT_DIR}/outputs/evaluate_rocket/mine_single_factor_layout_probes_visualstable_20260420_182356/offset_chamber_left/A/plan_attempt_01.json}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/recheck_offset_chamber_left_A0A2_compare_$(date +%Y%m%d_%H%M%S)}"
PLAN_ROOT="${PLAN_ROOT:-${OUT_ROOT}/plans}"
LOG_PATH="${LOG_PATH:-${OUT_ROOT}/wrapper.log}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

mkdir -p "${OUT_ROOT}" "${PLAN_ROOT}"
mkdir -p "$(dirname "${LOG_PATH}")"

if [[ "${WRAPPER_TEE:-1}" == "1" && -z "${OFFSET_A_COMPARE_WRAPPER_LOG_INIT:-}" ]]; then
  export OFFSET_A_COMPARE_WRAPPER_LOG_INIT=1
  exec > >(tee -a "${LOG_PATH}") 2>&1
fi

sanitize_colon_file_list() {
  local raw="${1:-}"
  local parts=()
  local item
  local old_ifs="${IFS}"
  IFS=':'
  for item in ${raw}; do
    [[ -f "${item}" ]] || continue
    parts+=("${item}")
  done
  IFS="${old_ifs}"
  if (( ${#parts[@]} > 0 )); then
    (IFS=:; echo "${parts[*]}")
  fi
}

configure_torch_cuda_runtime() {
  local python_version
  python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local site_pkg="${CONDA_PREFIX:-}/lib/python${python_version}/site-packages/nvidia"
  [[ -d "${site_pkg}" ]] || return 0

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

  if (( ${#lib_paths[@]} > 0 )); then
    local joined
    joined="$(IFS=:; echo "${lib_paths[*]}")"
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    local preload=()
    [[ -f "${site_pkg}/cublas/lib/libcublas.so.12" ]] && preload+=("${site_pkg}/cublas/lib/libcublas.so.12")
    [[ -f "${site_pkg}/cublas/lib/libcublasLt.so.12" ]] && preload+=("${site_pkg}/cublas/lib/libcublasLt.so.12")
    [[ -f "${site_pkg}/cudnn/lib/libcudnn.so.9" ]] && preload+=("${site_pkg}/cudnn/lib/libcudnn.so.9")
    if (( ${#preload[@]} > 0 )); then
      local preload_joined
      preload_joined="$(IFS=:; echo "${preload[*]}")"
      local existing_preload
      existing_preload="$(sanitize_colon_file_list "${LD_PRELOAD:-}")"
      export LD_PRELOAD="${preload_joined}${existing_preload:+:${existing_preload}}"
    fi
  fi
}

configure_torch_cuda_runtime

run_py() {
  if [[ "${USE_XVFB}" == "1" ]]; then
    xvfb-run -a -s "-screen 0 1280x1024x24" "${PYTHON_BIN}" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

latest_subdir() {
  local base_dir="$1"
  [[ -d "${base_dir}" ]] || return 0
  find "${base_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1
}

resolve_goal_seed_dir() {
  local worldgen_dir="$1"
  local protocol_dir="${worldgen_dir}/_baked_goals/${TASK}/protocol_${PROTOCOL}"
  [[ -d "${protocol_dir}" ]] || return 0
  find "${protocol_dir}" -mindepth 1 -maxdepth 1 -type d -name 'seed_*' | sort | head -n1
}

resolve_goal_image_path() {
  local worldgen_dir="$1"
  local seed_dir
  seed_dir="$(resolve_goal_seed_dir "${worldgen_dir}")"
  [[ -n "${seed_dir}" ]] || return 0
  if [[ -f "${seed_dir}/goal_image.png" ]]; then
    echo "${seed_dir}/goal_image.png"
    return 0
  fi
  if [[ -f "${seed_dir}/selected_pose/goal_image.png" ]]; then
    echo "${seed_dir}/selected_pose/goal_image.png"
  fi
}

find_matching_rollout_dir() {
  local rollout_root="$1"
  local worldgen_dir="$2"
  [[ -d "${rollout_root}" ]] || return 0
  "${PYTHON_BIN}" - "${rollout_root}" "${worldgen_dir}" <<'PY'
import json
import sys
from pathlib import Path

rollout_root = Path(sys.argv[1])
worldgen_dir = str(Path(sys.argv[2]).resolve())
matches = []
for run_dir in sorted(rollout_root.iterdir()):
    if not run_dir.is_dir():
        continue
    meta_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "summary.json"
    if not meta_path.is_file() or not summary_path.is_file():
        continue
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    task_group_path = meta.get("task_group_path")
    if not task_group_path:
        continue
    if str(Path(task_group_path).resolve()) == worldgen_dir:
        matches.append(str(run_dir))
if matches:
    print(matches[-1])
PY
}

write_plan_variant() {
  local label="$1"
  local a_level="$2"
  local out_path="$3"

  "${PYTHON_BIN}" - "${REFERENCE_PLAN}" "${out_path}" "${label}" "${a_level}" <<'PY'
import json
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
label = sys.argv[3]
a_level = int(sys.argv[4])

rows = json.loads(src_path.read_text(encoding="utf-8"))
if not isinstance(rows, list) or not rows:
    raise SystemExit(f"invalid plan json: {src_path}")

row = dict(rows[0])
levels = dict(row.get("factor_levels") or {})
for key in ["R", "H", "O", "C", "P", "A"]:
    levels[key] = int(levels.get(key, 0) or 0)
levels["A"] = a_level
row["factor_levels"] = levels
row["severity_majority"] = a_level
row["primary_failure_mode_majority"] = f"offset_chamber_left_A{a_level}_compare"
row["primary_factors_majority"] = ["A"] if a_level > 0 else []
suggestions = dict(row.get("world_generation_suggestions") or {})
notes = suggestions.get("notes", "")
suffix = f"compare variant label={label} A={a_level}"
suggestions["notes"] = f"{notes} | {suffix}".strip(" |")
row["world_generation_suggestions"] = suggestions

out_path.write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
PY
}

run_case() {
  local label="$1"
  local a_level="$2"
  local plan_json="${PLAN_ROOT}/${label}.json"
  local case_root="${OUT_ROOT}/${label}"
  local worldgen_root="${case_root}/worldgen"
  local rollout_root="${case_root}/rollout"

  write_plan_variant "${label}" "${a_level}" "${plan_json}"

  echo
  echo "==================== ${label} ===================="
  echo "[run] plan_json=${plan_json}"
  echo "[run] A=${a_level}"

  run_py -m minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen \
    --plan-json "${plan_json}" \
    --env-conf-dir "${ENV_CONF_DIR}" \
    --mine-layout-backend procedural \
    --out-dir "${worldgen_root}"

  local worldgen_dir
  worldgen_dir="$(latest_subdir "${worldgen_root}")"
  if [[ -z "${worldgen_dir}" ]]; then
    echo "[error] missing worldgen output for label=${label}" >&2
    return 1
  fi

  run_py -m minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets \
    --task-group "offset_chamber_left_${label}_rebake" \
    --task-group-path "${worldgen_dir}" \
    --protocol "${PROTOCOL}" \
    --tasks "${TASK}" \
    --base-seed "${BASE_SEED}" \
    --episode-retries "${EPISODE_RETRIES}" \
    --save-debug-assets \
    --overwrite \
    --refresh-task-configs

  run_py -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout \
    --env-source "${ENV_SOURCE}" \
    --protocol "${PROTOCOL}" \
    --task-group "offset_chamber_left_${label}_rollout" \
    --task-group-path "${worldgen_dir}" \
    --tasks "${TASK}" \
    --episodes-per-task "${ROLLOUT_EPISODES}" \
    --base-seed "${BASE_SEED}" \
    --seed-step "${SEED_STEP}" \
    --episode-retries "${EPISODE_RETRIES}" \
    --step-budget-override "${STEP_BUDGET}" \
    --model-path "${MODEL_PATH}" \
    --out-dir "${rollout_root}" \
    --auto-goal \
    --save-debug-assets

  local goal_image
  goal_image="$(resolve_goal_image_path "${worldgen_dir}")"
  local rollout_dir
  rollout_dir="$(find_matching_rollout_dir "${rollout_root}" "${worldgen_dir}")"

  echo "[done] label=${label}"
  echo "  worldgen_dir: ${worldgen_dir}"
  echo "  goal_image: ${goal_image}"
  echo "  rollout_dir: ${rollout_dir}"
  find "${rollout_dir}" -name "*_annotated.mp4" | sort
}

echo "[log] LOG_PATH=${LOG_PATH}"
echo "[config] REFERENCE_PLAN=${REFERENCE_PLAN}"
echo "[config] OUT_ROOT=${OUT_ROOT}"

run_case "A0" 0
run_case "A2" 2

echo
echo "[summary] OUT_ROOT=${OUT_ROOT}"
find "${OUT_ROOT}" \( -name goal_image.png -o -name goal_bbox_overlay.png -o -name "*_annotated.mp4" -o -name plan_attempt_01.json -o -name "A0.json" -o -name "A2.json" \) | sort
