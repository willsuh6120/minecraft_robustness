#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK="${TASK:-mine_coal}"
PROTOCOL="${PROTOCOL:-ours_v1}"
ENV_SOURCE="${ENV_SOURCE:-rocket2_official}"
ENV_CONF_DIR="${ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

PROBE_LEVEL="${PROBE_LEVEL:-2}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
BAKE_BASE_SEED="${BAKE_BASE_SEED:-1}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-1}"
STEP_BUDGET="${STEP_BUDGET:-120}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
MAX_OFFSET_ATTEMPTS="${MAX_OFFSET_ATTEMPTS:-24}"
USE_XVFB="${USE_XVFB:-1}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_single_factor_layout_probes_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-${OUT_ROOT}/wrapper.log}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

mkdir -p "${OUT_ROOT}"
mkdir -p "$(dirname "${LOG_PATH}")"

if [[ "${WRAPPER_TEE:-1}" == "1" && -z "${PROBE_WRAPPER_LOG_INIT:-}" ]]; then
  export PROBE_WRAPPER_LOG_INIT=1
  exec > >(tee -a "${LOG_PATH}") 2>&1
fi

echo "[log] LOG_PATH=${LOG_PATH}"

SUMMARY_TSV="${OUT_ROOT}/summary.tsv"

FACTOR_CODES_CSV="${FACTOR_CODES_CSV:-R,H,O,C,P,A}"
LAYOUT_CASES_CSV="${LAYOUT_CASES_CSV:-straight,side_alcove_left,turn_left,offset_chamber_left}"

IFS=',' read -r -a FACTOR_CODES <<< "${FACTOR_CODES_CSV}"
IFS=',' read -r -a LAYOUT_CASES <<< "${LAYOUT_CASES_CSV}"

echo "[config] FACTOR_CODES=${FACTOR_CODES[*]}"
echo "[config] LAYOUT_CASES=${LAYOUT_CASES[*]}"

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

ensure_summary_tsv() {
  if [[ ! -s "${SUMMARY_TSV}" ]]; then
    printf "layout_case\tfactor_code\tlevel\tblueprint_id\ttarget_x\tworldgen_dir\tgoal_image\tgoal_bbox_overlay\trollout_dir\n" > "${SUMMARY_TSV}"
  fi
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

resolve_goal_bbox_overlay_path() {
  local worldgen_dir="$1"
  local seed_dir
  seed_dir="$(resolve_goal_seed_dir "${worldgen_dir}")"
  [[ -n "${seed_dir}" ]] || return 0
  if [[ -f "${seed_dir}/goal_bbox_overlay.png" ]]; then
    echo "${seed_dir}/goal_bbox_overlay.png"
    return 0
  fi
  if [[ -f "${seed_dir}/selected_pose/goal_bbox_overlay.png" ]]; then
    echo "${seed_dir}/selected_pose/goal_bbox_overlay.png"
  fi
}

resolve_goal_spec_path() {
  local worldgen_dir="$1"
  local seed_dir
  seed_dir="$(resolve_goal_seed_dir "${worldgen_dir}")"
  [[ -n "${seed_dir}" ]] || return 0
  if [[ -f "${seed_dir}/goal_spec.json" ]]; then
    echo "${seed_dir}/goal_spec.json"
    return 0
  fi
  if [[ -f "${seed_dir}/selected_pose/goal_spec.json" ]]; then
    echo "${seed_dir}/selected_pose/goal_spec.json"
  fi
}

find_matching_rollout_dir() {
  local probe_root="$1"
  local worldgen_dir="$2"
  local rollout_root="${probe_root}/rollout"
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

summary_upsert_row() {
  local layout_case="$1"
  local factor_code="$2"
  local level="$3"
  local blueprint_id="$4"
  local target_x="$5"
  local worldgen_dir="$6"
  local goal_image="$7"
  local goal_bbox_overlay="$8"
  local rollout_dir="$9"
  "${PYTHON_BIN}" - "${SUMMARY_TSV}" "${layout_case}" "${factor_code}" "${level}" "${blueprint_id}" "${target_x}" "${worldgen_dir}" "${goal_image}" "${goal_bbox_overlay}" "${rollout_dir}" <<'PY'
import csv
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
new_row = sys.argv[2:]
header = [
    "layout_case",
    "factor_code",
    "level",
    "blueprint_id",
    "target_x",
    "worldgen_dir",
    "goal_image",
    "goal_bbox_overlay",
    "rollout_dir",
]
rows = []
if summary_path.exists():
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        existing = list(reader)
    data_rows = existing[1:] if existing and existing[0] == header else existing
    for row in data_rows:
        if len(row) >= 2 and row[0] == new_row[0] and row[1] == new_row[1]:
            continue
        rows.append(row)
rows.append(new_row)
with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(header)
    writer.writerows(rows)
PY
}

layout_blueprint() {
  local layout_case="$1"
  case "${layout_case}" in
    straight) echo "straight_tunnel" ;;
    side_alcove_left) echo "side_alcove_left" ;;
    turn_left) echo "turn_left" ;;
    offset_chamber_left) echo "offset_chamber" ;;
    *)
      echo "unknown layout_case=${layout_case}" >&2
      return 1
      ;;
  esac
}

layout_desired_sign() {
  local layout_case="$1"
  case "${layout_case}" in
    offset_chamber_left) echo "neg" ;;
    *) echo "any" ;;
  esac
}

write_plan_json() {
  local plan_path="$1"
  local layout_case="$2"
  local blueprint_id="$3"
  local desired_sign="$4"
  local factor_code="$5"
  local level="$6"
  local layout_seed="$7"

  "${PYTHON_BIN}" - "${plan_path}" "${TASK}" "${layout_case}" "${blueprint_id}" "${desired_sign}" "${factor_code}" "${level}" "${layout_seed}" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
task_name = sys.argv[2]
layout_case = sys.argv[3]
blueprint_id = sys.argv[4]
desired_sign = sys.argv[5]
factor_code = sys.argv[6]
level = int(sys.argv[7])
layout_seed = int(sys.argv[8])

factor_levels = {"R": 0, "H": 0, "O": 0, "C": 0, "P": 0, "A": 0}
if factor_code in factor_levels:
    factor_levels[factor_code] = level

primary_factors = [factor_code] if factor_code in factor_levels and level > 0 else []
hard_factor_count = 1 if primary_factors else 0

row = {
    "task_config_name": task_name,
    "task_key": task_name,
    "primary_failure_mode_majority": f"single_factor_probe_{layout_case}_{factor_code}",
    "primary_factors_majority": primary_factors,
    "severity_majority": level,
    "factor_levels": factor_levels,
    "layout_seed": layout_seed,
    "template_index": 0,
    "requested_split_label": "train_id",
    "computed_split_label": "train_id",
    "hard_factor_count": hard_factor_count,
    "trainable_with_rl_majority": True,
    "world_generation_suggestions": {
        "mine_blueprint_id": blueprint_id,
        "notes": f"single-factor probe layout={layout_case} factor={factor_code} level={level} sign={desired_sign}",
    },
}
plan_path.write_text(json.dumps([row], ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

extract_target_x() {
  local yaml_path="$1"
  "${PYTHON_BIN}" - "${yaml_path}" <<'PY'
import sys
import yaml

data = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
print(int(data["procedural_layout"]["target_local"][0]))
PY
}

ensure_summary_tsv

run_probe() {
  local layout_case="$1"
  local factor_code="$2"
  local level="${3:-${PROBE_LEVEL}}"

  local blueprint_id
  blueprint_id="$(layout_blueprint "${layout_case}")"
  local desired_sign
  desired_sign="$(layout_desired_sign "${layout_case}")"

  local probe_root="${OUT_ROOT}/${layout_case}/${factor_code}"
  mkdir -p "${probe_root}"

  local max_attempts=1
  if [[ "${desired_sign}" != "any" ]]; then
    max_attempts="${MAX_OFFSET_ATTEMPTS}"
  fi

  local attempt
  for attempt in $(seq 1 "${max_attempts}"); do
    local attempt_tag
    attempt_tag="$(printf '%02d' "${attempt}")"
    local layout_seed=$((20000 + attempt * 97))
    local plan_json="${probe_root}/plan_attempt_${attempt_tag}.json"
    local world_out="${probe_root}/worldgen_attempt_${attempt_tag}"

    echo
    echo "==================== ${layout_case} / ${factor_code}${level} attempt ${attempt_tag} ===================="
    echo "plan=${plan_json}"

    local worldgen_dir
    worldgen_dir="$(latest_subdir "${world_out}")"
    if [[ -n "${worldgen_dir}" && -f "${worldgen_dir}/${TASK}.yaml" ]]; then
      echo "[resume] using existing worldgen_dir=${worldgen_dir}"
    else
      write_plan_json "${plan_json}" "${layout_case}" "${blueprint_id}" "${desired_sign}" "${factor_code}" "${level}" "${layout_seed}"
      "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen \
        --plan-json "${plan_json}" \
        --env-conf-dir "${ENV_CONF_DIR}" \
        --mine-layout-backend procedural \
        --mine-anchor-mode source_or_fallback \
        --out-dir "${world_out}"
      worldgen_dir="$(latest_subdir "${world_out}")"
      if [[ -z "${worldgen_dir}" ]]; then
        echo "[error] no worldgen dir for ${layout_case}/${factor_code}" >&2
        exit 1
      fi
    fi

    local yaml_path="${worldgen_dir}/${TASK}.yaml"
    local target_x
    target_x="$(extract_target_x "${yaml_path}")"

    if [[ "${desired_sign}" == "neg" && "${target_x}" -ge 0 ]]; then
      echo "[retry] ${layout_case}/${factor_code} target_local.x=${target_x}, wanted negative"
      continue
    fi

    echo "[selected] ${layout_case}/${factor_code} target_local.x=${target_x}"

    local goal_image
    goal_image="$(resolve_goal_image_path "${worldgen_dir}")"
    local goal_bbox_overlay
    goal_bbox_overlay="$(resolve_goal_bbox_overlay_path "${worldgen_dir}")"
    local goal_spec_path
    goal_spec_path="$(resolve_goal_spec_path "${worldgen_dir}")"
    if [[ -n "${goal_image}" && -n "${goal_bbox_overlay}" && -n "${goal_spec_path}" ]]; then
      echo "[resume] using existing baked goal assets"
    else
      run_py -m minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets \
        --task-group "${layout_case}_${factor_code}_bake" \
        --task-group-path "${worldgen_dir}" \
        --protocol "${PROTOCOL}" \
        --tasks "${TASK}" \
        --base-seed "${BAKE_BASE_SEED}" \
        --episode-retries "${EPISODE_RETRIES}" \
        --save-debug-assets \
        --overwrite \
        --refresh-task-configs
      goal_image="$(resolve_goal_image_path "${worldgen_dir}")"
      goal_bbox_overlay="$(resolve_goal_bbox_overlay_path "${worldgen_dir}")"
      goal_spec_path="$(resolve_goal_spec_path "${worldgen_dir}")"
    fi
    if [[ -z "${goal_image}" || -z "${goal_bbox_overlay}" || -z "${goal_spec_path}" ]]; then
      echo "[error] baked goal assets missing for ${layout_case}/${factor_code}" >&2
      return 1
    fi

    local rollout_out="${probe_root}/rollout"

    local rollout_dir
    rollout_dir="$(find_matching_rollout_dir "${probe_root}" "${worldgen_dir}")"
    if [[ -n "${rollout_dir}" ]]; then
      echo "[resume] using existing rollout_dir=${rollout_dir}"
    else
      run_py -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout \
        --env-source "${ENV_SOURCE}" \
        --protocol "${PROTOCOL}" \
        --task-group "${layout_case}_${factor_code}_rollout" \
        --task-group-path "${worldgen_dir}" \
        --tasks "${TASK}" \
        --episodes-per-task "${ROLLOUT_EPISODES}" \
        --base-seed "${BASE_SEED}" \
        --seed-step "${SEED_STEP}" \
        --episode-retries "${EPISODE_RETRIES}" \
        --step-budget-override "${STEP_BUDGET}" \
        --model-path "${MODEL_PATH}" \
        --out-dir "${rollout_out}" \
        --auto-goal \
        --save-debug-assets
      rollout_dir="$(find_matching_rollout_dir "${probe_root}" "${worldgen_dir}")"
    fi
    if [[ -z "${rollout_dir}" ]]; then
      echo "[error] rollout outputs missing for ${layout_case}/${factor_code}" >&2
      return 1
    fi

    summary_upsert_row \
      "${layout_case}" \
      "${factor_code}" \
      "${level}" \
      "${blueprint_id}" \
      "${target_x}" \
      "${worldgen_dir}" \
      "${goal_image}" \
      "${goal_bbox_overlay}" \
      "${rollout_dir}"

    echo "[done] ${layout_case}/${factor_code}"
    echo "  yaml: ${yaml_path}"
    echo "  goal_image: ${goal_image}"
    echo "  rollout_dir: ${rollout_dir}"
    find "${rollout_dir}" -name "*_annotated.mp4" | sort
    return 0
  done

  echo "[failed] ${layout_case}/${factor_code}: could not satisfy desired sign after ${max_attempts} attempts" >&2
  return 1
}

for layout_case in "${LAYOUT_CASES[@]}"; do
  if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
    run_probe "${layout_case}" "BASE" "0"
  fi
  for factor_code in "${FACTOR_CODES[@]}"; do
    run_probe "${layout_case}" "${factor_code}"
  done
done

echo
echo "[summary] OUT_ROOT=${OUT_ROOT}"
echo "[summary] TSV=${SUMMARY_TSV}"
find "${OUT_ROOT}" \( -name goal_image.png -o -name goal_bbox_overlay.png -o -name "*_annotated.mp4" \) | sort
