#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/gyulab/envgen2/Minestudio"
TASK="mine_coal"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-/home/gyulab/envgen2/Minestudio/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
SEED="${SEED:-1}"
STEP_BUDGET="${STEP_BUDGET:-30}"
LEVEL_R="${LEVEL_R:-2}"
LEVEL_H="${LEVEL_H:-2}"
LEVEL_O="${LEVEL_O:-2}"
LEVEL_C="${LEVEL_C:-2}"
LEVEL_P="${LEVEL_P:-2}"
LEVEL_A="${LEVEL_A:-2}"

PLAN_DIR="${ROOT_DIR}/outputs/evaluate_rocket/factor_probe_plans"
WORLD_BASE="${ROOT_DIR}/outputs/evaluate_rocket/factor_probe_worlds"
ROLLOUT_BASE="${ROOT_DIR}/outputs/evaluate_rocket/factor_probe_rollouts"

mkdir -p "${PLAN_DIR}" "${WORLD_BASE}" "${ROLLOUT_BASE}"

make_plan() {
  local factor="$1"
  local level="$2"
  local plan_path="${PLAN_DIR}/${TASK}_${factor}${level}.json"
  cat > "${plan_path}" <<JSON
[
  {
    "task_config_name": "${TASK}",
    "primary_failure_mode_majority": "manual_factor_probe",
    "primary_factors_majority": ["${factor}"],
    "severity_majority": ${level},
    "factor_levels": {
      "R": 0,
      "H": 0,
      "O": 0,
      "C": 0,
      "P": 0,
      "A": 0,
      "${factor}": ${level}
    },
    "trainable_with_rl_majority": true,
    "world_generation_suggestions": {
      "visibility": "keep",
      "distractors": "keep",
      "path_difficulty": "keep",
      "view_difficulty": "keep",
      "layout_changes": ["keep_layout"],
      "notes": "manual factor probe"
    }
  }
]
JSON
  echo "${plan_path}"
}

run_probe() {
  local factor="$1"
  local level="$2"
  local world_out="${WORLD_BASE}/${TASK}_${factor}${level}"
  local rollout_out="${ROLLOUT_BASE}/${TASK}_${factor}${level}"
  local plan_path
  plan_path="$(make_plan "${factor}" "${level}")"

  echo
  echo "==================== ${factor}${level} ===================="
  echo "plan=${plan_path}"

  python -m minestudio.tutorials.inference.evaluate_rocket.interaction_worldgen \
    --plan-json "${plan_path}" \
    --env-conf-dir "${SOURCE_ENV_CONF_DIR}" \
    --out-dir "${world_out}"

  local world_run_dir
  world_run_dir="$(ls -td "${world_out}"/* | head -1)"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MINESTUDIO_DIR=/home/gyulab/envgen2/.minestudio \
  HF_HOME=/home/gyulab/envgen2/.cache/huggingface \
  python -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout \
    --env-source rocket2_official \
    --task-group generated_task_group \
    --protocol ours_v1 \
    --task-group-path "${world_run_dir}" \
    --tasks "${TASK}" \
    --auto-goal \
    --episodes-per-task 1 \
    --base-seed "${SEED}" \
    --episode-retries 1 \
    --step-budget-override "${STEP_BUDGET}" \
    --save-debug-assets \
    --model-path hf:phython96/ROCKET-2-1x-22w \
    --out-dir "${rollout_out}"

  local rollout_run_dir
  rollout_run_dir="$(ls -td "${rollout_out}"/* | head -1)"

  echo "WORLD_RUN_DIR=${world_run_dir}"
  echo "ROLLOUT_RUN_DIR=${rollout_run_dir}"
  echo "[worldgen_manifest] ${world_run_dir}/worldgen_manifest.json"
  echo "[yaml] ${world_run_dir}/${TASK}.yaml"
  echo "[warmup_pov] ${rollout_run_dir}/${TASK}/seed_${SEED}_ep_000/rollout_after_warmup_pov.png"
  echo "[goal_overlay] ${rollout_run_dir}/${TASK}/seed_${SEED}_ep_000/selected_pose/goal_mask_overlay.png"
  echo "[selected_pose_metadata] ${rollout_run_dir}/${TASK}/seed_${SEED}_ep_000/selected_pose/metadata.json"
  echo
}

run_probe "R" "${LEVEL_R}"
run_probe "H" "${LEVEL_H}"
run_probe "O" "${LEVEL_O}"
run_probe "C" "${LEVEL_C}"
run_probe "P" "${LEVEL_P}"
run_probe "A" "${LEVEL_A}"

echo "done"
