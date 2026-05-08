#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK="${TASK:-mine_coal}"
PROTOCOL="${PROTOCOL:-ours_v1}"
ENV_SOURCE="${ENV_SOURCE:-rocket2_official}"
ENV_CONF_DIR="${ENV_CONF_DIR:-/home/gyulab/envgen2/Minestudio/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
PYTHON_BIN="${PYTHON_BIN:-/home/gyulab/miniconda3/envs/minestudio/bin/python}"

BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
BAKE_BASE_SEED="${BAKE_BASE_SEED:-1}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
MAX_OFFSET_ATTEMPTS="${MAX_OFFSET_ATTEMPTS:-24}"

TURN_PATH_LEVEL="${TURN_PATH_LEVEL:-2}"
ALCOVE_PATH_LEVEL="${ALCOVE_PATH_LEVEL:-2}"
STRAIGHT_PATH_LEVEL="${STRAIGHT_PATH_LEVEL:-0}"
OFFSET_PATH_LEVEL="${OFFSET_PATH_LEVEL:-0}"

USE_XVFB="${USE_XVFB:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/layout_family_pack_$(date +%Y%m%d_%H%M%S)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-/home/gyulab/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-/home/gyulab/envgen2/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

mkdir -p "${OUT_ROOT}"
SUMMARY_TSV="${OUT_ROOT}/summary.tsv"
printf "case_name\tblueprint_id\ttarget_x\tworldgen_dir\tgoal_image\tgoal_bbox_overlay\trollout_dir\n" > "${SUMMARY_TSV}"

run_py() {
  if [[ "${USE_XVFB}" == "1" ]]; then
    xvfb-run -a -s "-screen 0 1280x1024x24" "${PYTHON_BIN}" "$@"
  else
    "${PYTHON_BIN}" "$@"
  fi
}

latest_subdir() {
  local base_dir="$1"
  find "${base_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1
}

write_plan_json() {
  local plan_path="$1"
  local blueprint_id="$2"
  local layout_seed="$3"
  local path_level="$4"

  "${PYTHON_BIN}" - "${plan_path}" "${TASK}" "${blueprint_id}" "${layout_seed}" "${path_level}" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
task_name = sys.argv[2]
blueprint_id = sys.argv[3]
layout_seed = int(sys.argv[4])
path_level = int(sys.argv[5])

row = {
    "task_config_name": task_name,
    "task_key": task_name,
    "primary_failure_mode_majority": "layout_family_probe",
    "primary_factors_majority": ["P"] if path_level > 0 else [],
    "severity_majority": path_level,
    "factor_levels": {"R": 0, "H": 0, "O": 0, "C": 0, "P": path_level, "A": 0},
    "layout_seed": layout_seed,
    "template_index": 0,
    "requested_split_label": "train_id",
    "computed_split_label": "train_id",
    "hard_factor_count": 1 if path_level > 0 else 0,
    "trainable_with_rl_majority": True,
    "world_generation_suggestions": {
        "mine_blueprint_id": blueprint_id,
        "notes": f"layout family probe for {blueprint_id}",
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

run_case() {
  local case_name="$1"
  local blueprint_id="$2"
  local desired_sign="$3"
  local path_level="$4"

  local case_root="${OUT_ROOT}/${case_name}"
  mkdir -p "${case_root}"

  local max_attempts=1
  if [[ "${desired_sign}" != "any" ]]; then
    max_attempts="${MAX_OFFSET_ATTEMPTS}"
  fi

  local attempt
  for attempt in $(seq 1 "${max_attempts}"); do
    local attempt_tag
    attempt_tag="$(printf '%02d' "${attempt}")"
    local layout_seed=$((10000 + attempt * 97))
    local plan_json="${case_root}/plan_attempt_${attempt_tag}.json"
    local world_out="${case_root}/worldgen_attempt_${attempt_tag}"

    write_plan_json "${plan_json}" "${blueprint_id}" "${layout_seed}" "${path_level}"

    echo
    echo "==================== ${case_name} attempt ${attempt_tag} ===================="
    echo "plan=${plan_json}"

    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen \
      --plan-json "${plan_json}" \
      --env-conf-dir "${ENV_CONF_DIR}" \
      --mine-layout-backend procedural \
      --mine-anchor-mode source_or_fallback \
      --out-dir "${world_out}"

    local worldgen_dir
    worldgen_dir="$(latest_subdir "${world_out}")"
    if [[ -z "${worldgen_dir}" ]]; then
      echo "[error] no worldgen dir produced for ${case_name}" >&2
      exit 1
    fi

    local yaml_path="${worldgen_dir}/${TASK}.yaml"
    local target_x
    target_x="$(extract_target_x "${yaml_path}")"

    if [[ "${desired_sign}" == "neg" && "${target_x}" -ge 0 ]]; then
      echo "[retry] ${case_name} target_local.x=${target_x}, wanted negative"
      continue
    fi
    if [[ "${desired_sign}" == "pos" && "${target_x}" -le 0 ]]; then
      echo "[retry] ${case_name} target_local.x=${target_x}, wanted positive"
      continue
    fi

    echo "[selected] ${case_name} target_local.x=${target_x} worldgen_dir=${worldgen_dir}"

    run_py -m minestudio.tutorials.inference.evaluate_rocket.interaction_bake_goal_assets \
      --task-group "${case_name}_bake" \
      --task-group-path "${worldgen_dir}" \
      --protocol "${PROTOCOL}" \
      --tasks "${TASK}" \
      --base-seed "${BAKE_BASE_SEED}" \
      --episode-retries "${EPISODE_RETRIES}" \
      --save-debug-assets \
      --overwrite \
      --refresh-task-configs

    local rollout_out="${case_root}/rollout"
    run_py -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout \
      --env-source "${ENV_SOURCE}" \
      --protocol "${PROTOCOL}" \
      --task-group "${case_name}_rollout" \
      --task-group-path "${worldgen_dir}" \
      --tasks "${TASK}" \
      --episodes-per-task "${ROLLOUT_EPISODES}" \
      --base-seed "${BASE_SEED}" \
      --seed-step "${SEED_STEP}" \
      --episode-retries "${EPISODE_RETRIES}" \
      --step-budget-override "${STEP_BUDGET}" \
      --model-path "${MODEL_PATH}" \
      --out-dir "${rollout_out}" \
      --auto-goal

    local rollout_dir
    rollout_dir="$(latest_subdir "${rollout_out}")"
    local goal_image
    goal_image="$(find "${worldgen_dir}/_baked_goals" -name goal_image.png | sort | head -n1)"
    local goal_bbox_overlay
    goal_bbox_overlay="$(find "${worldgen_dir}/_baked_goals" -name goal_bbox_overlay.png | sort | head -n1)"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${case_name}" \
      "${blueprint_id}" \
      "${target_x}" \
      "${worldgen_dir}" \
      "${goal_image}" \
      "${goal_bbox_overlay}" \
      "${rollout_dir}" >> "${SUMMARY_TSV}"

    echo "[done] ${case_name}"
    echo "  yaml: ${yaml_path}"
    echo "  goal_image: ${goal_image}"
    echo "  goal_bbox_overlay: ${goal_bbox_overlay}"
    echo "  rollout_dir: ${rollout_dir}"
    find "${rollout_dir}" -name "*_annotated.mp4" | sort
    return 0
  done

  echo "[failed] ${case_name}: could not satisfy desired_sign=${desired_sign} after ${max_attempts} attempts" >&2
  return 1
}

run_case "straight_tunnel"       "straight_tunnel"   "any" "${STRAIGHT_PATH_LEVEL}"
run_case "offset_chamber_left"   "offset_chamber"    "neg" "${OFFSET_PATH_LEVEL}"
run_case "offset_chamber_right"  "offset_chamber"    "pos" "${OFFSET_PATH_LEVEL}"
run_case "side_alcove_left"      "side_alcove_left"  "any" "${ALCOVE_PATH_LEVEL}"
run_case "side_alcove_right"     "side_alcove_right" "any" "${ALCOVE_PATH_LEVEL}"
run_case "turn_left"             "turn_left"         "any" "${TURN_PATH_LEVEL}"
run_case "turn_right"            "turn_right"        "any" "${TURN_PATH_LEVEL}"

echo
echo "[summary] OUT_ROOT=${OUT_ROOT}"
echo "[summary] TSV=${SUMMARY_TSV}"
find "${OUT_ROOT}" \( -name goal_image.png -o -name goal_bbox_overlay.png -o -name "*_annotated.mp4" \) | sort
