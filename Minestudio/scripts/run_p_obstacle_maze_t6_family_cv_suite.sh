#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"

setup_minestudio_runtime
require_xvfb_run
ensure_minestudio_engine

TOOLS_PY="${ROOT_DIR}/scripts/maze_t6_uniform_reward_tools.py"

TASK="${TASK:-mine_coal}"
RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_family_cv_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/${RUN_TAG}}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"

MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

FOLD_NAME="${FOLD_NAME:-fold_a}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_waypoint_cv_${FOLD_NAME}}"
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-1.0}"

ASSET_DIR="${ASSET_DIR:-}"
BANK_VIEW_ROOT="${BANK_VIEW_ROOT:-}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"

TRAIN_ITERS="${TRAIN_ITERS:-24}"
COLLECT_EPISODES="${COLLECT_EPISODES:-72}"
COLLECT_WORLD_INSTANCES="${COLLECT_WORLD_INSTANCES:-12}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
PPO_CLIP="${PPO_CLIP:-0.15}"
PPO_GAMMA="${PPO_GAMMA:-0.999}"
VF_COEF="${VF_COEF:-0.25}"
POLICY_COEF="${POLICY_COEF:-1.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
KL_COEF="${KL_COEF:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-heads}"
NORMALIZE_ADVANTAGE="${NORMALIZE_ADVANTAGE:-1}"
CLIP_VLOSS="${CLIP_VLOSS:-1}"

VAL_BASELINE_EPISODES="${VAL_BASELINE_EPISODES:-16}"
TEST_BASELINE_EPISODES="${TEST_BASELINE_EPISODES:-32}"
CHECKPOINT_EVAL_EVERY="${CHECKPOINT_EVAL_EVERY:-4}"
VAL_EVAL_EPISODES="${VAL_EVAL_EPISODES:-16}"
BEST_TEST_EPISODES="${BEST_TEST_EPISODES:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
FINAL_PROBE_WORKERS="${FINAL_PROBE_WORKERS:-2}"
VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
BASELINE_VIDEO_EPISODES="${BASELINE_VIDEO_EPISODES:-1}"
RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE:-1}"

mkdir -p "${OUT_ROOT}"

log() {
  printf '[maze-t6-family-cv] %s\n' "$*" >&2
}

find_latest_subdir() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

resolve_asset_dir_from_source() {
  if [[ -n "${ASSET_DIR}" && -d "${ASSET_DIR}" ]]; then
    return 0
  fi
  if [[ -n "${SOURCE_RUN_ROOT}" && -d "${SOURCE_RUN_ROOT}/assets" ]]; then
    ASSET_DIR="$(find_latest_subdir "${SOURCE_RUN_ROOT}/assets")"
  fi
}

ensure_bank_views() {
    local required_rel="cv_folds/${FOLD_NAME}/train/bank_manifest.json"
    if [[ -n "${BANK_VIEW_ROOT}" && -f "${BANK_VIEW_ROOT}/${required_rel}" ]]; then
        return 0
    fi
    if [[ -n "${SOURCE_RUN_ROOT}" && -f "${SOURCE_RUN_ROOT}/bank_views/${required_rel}" ]]; then
        BANK_VIEW_ROOT="${SOURCE_RUN_ROOT}/bank_views"
        return 0
    fi
    if [[ -n "${SOURCE_RUN_ROOT}" && -f "${SOURCE_RUN_ROOT}/bank_views/splits/full16/bank_manifest.json" ]]; then
        BANK_VIEW_ROOT="${OUT_ROOT}/bank_views"
        "${PYTHON_BIN}" "${TOOLS_PY}" make-bank-views-from-manifest \
            --manifest-path "${SOURCE_RUN_ROOT}/bank_views/splits/full16/bank_manifest.json" \
            --out-dir "${BANK_VIEW_ROOT}" >/dev/null
        return 0
    fi
    resolve_asset_dir_from_source
    if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
        echo "Need ASSET_DIR or SOURCE_RUN_ROOT with assets to generate cv_folds." >&2
        exit 1
  fi
  BANK_VIEW_ROOT="${OUT_ROOT}/bank_views"
  "${PYTHON_BIN}" "${TOOLS_PY}" make-bank-views --asset-dir "${ASSET_DIR}" --out-dir "${BANK_VIEW_ROOT}" >/dev/null
}

run_probe() {
  local bank_base_dir="$1"
  local bank_name="$2"
  local label="$3"
  local model_path="$4"
  local episodes="$5"
  local workers="$6"
  local skip_video="$7"
  local out_root="$8"
  local baseline_summary="${9:-}"

  mkdir -p "${out_root}"
  log "probe bank=${bank_name} label=${label} episodes=${episodes}"
  ASSET_DIR="${bank_base_dir}" \
  BANK_NAME="${bank_name}" \
  TASK="${TASK}" \
  MODEL_PATH="${model_path}" \
  CFG_COEF="${CFG_COEF}" \
  CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
  CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
  ENV_REWARD_SCALE=0.0 \
  EPISODES="${episodes}" \
  WORLD_WORKERS="${workers}" \
  BASE_SEED="${BASE_SEED}" \
  SEED_STEP="${SEED_STEP}" \
  SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED}" \
  SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP}" \
  EPISODE_RETRIES="${EPISODE_RETRIES}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  STOP_ON_SUCCESS=1 \
  SKIP_VIDEO="${skip_video}" \
  OUT_ROOT="${out_root}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

  local summarize_args=(
    "${PYTHON_BIN}" "${TOOLS_PY}" summarize-probe
    --summary-json "${out_root}/summary.json"
    --label "${label}"
    --out-json "${out_root}/split_report.json"
    --out-md "${out_root}/split_report.md"
  )
  if [[ -n "${baseline_summary}" ]]; then
    summarize_args+=( --baseline-summary "${baseline_summary}" )
  fi
  "${summarize_args[@]}" >/dev/null
}

run_training_iteration() {
  local collect_bank_dir="$1"
  local iteration_idx="$2"
  local input_model_path="$3"
  local experiment_root="$4"
  local iter_tag
  iter_tag="$(printf 'iter_%03d' "${iteration_idx}")"
  local pilot_out="${experiment_root}/training/${iter_tag}"
  local iter_base_seed=$(( BASE_SEED + (iteration_idx - 1) * COLLECT_EPISODES ))
  local iter_sampling_seed=$(( SAMPLING_BASE_SEED + (iteration_idx - 1) * COLLECT_EPISODES ))

  mkdir -p "${pilot_out}"
  log "train fold=${FOLD_NAME} iter=${iter_tag} collect_bank=${collect_bank_dir}"

  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" \
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot \
    --env-source rocket2_official \
    --protocol ours_v1 \
    --task-group "maze_t6_${EXPERIMENT_NAME}_${iter_tag}" \
    --task-group-path "${SOURCE_ENV_CONF_DIR}" \
    --worldgen-source-dir "${SOURCE_ENV_CONF_DIR}" \
    --tasks "${TASK}" \
    --auto-goal \
    --collect-world-mode fixed_bank \
    --eval-world-mode fixed_bank \
    --final-eval-world-mode fixed_bank \
    --collect-fixed-bank-dir "${collect_bank_dir}" \
    --eval-fixed-bank-dir "${collect_bank_dir}" \
    --final-eval-fixed-bank-dir "${collect_bank_dir}" \
    --collect-world-instances "${COLLECT_WORLD_INSTANCES}" \
    --eval-world-instances "${COLLECT_WORLD_INSTANCES}" \
    --final-eval-world-instances "${COLLECT_WORLD_INSTANCES}" \
    --collect-parallel-workers "${COLLECT_WORKERS}" \
    --eval-parallel-workers 1 \
    --final-eval-parallel-workers 1 \
    --collect-episodes-per-task "${COLLECT_EPISODES}" \
    --eval-episodes-per-task 1 \
    --train-iters 1 \
    --base-seed "${iter_base_seed}" \
    --seed-step "${SEED_STEP}" \
    --sampling-base-seed "${iter_sampling_seed}" \
    --sampling-seed-step "${SAMPLING_SEED_STEP}" \
    --episode-retries "${EPISODE_RETRIES}" \
    --step-budget-override "${STEP_BUDGET}" \
    --model-path "${input_model_path}" \
    --cfg-coef "${CFG_COEF}" \
    --env-reward-scale "${ENV_REWARD_SCALE}" \
    --cfg-policy-mode "${CFG_POLICY_MODE}" \
    --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}" \
    --kl-anchor-model-path "${KL_ANCHOR_MODEL_PATH}" \
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
    --loss-focus-mode uniform \
    --trainable-scope "${TRAINABLE_SCOPE}" \
    --min-successful-fragments 0 \
    --final-eval-model-mode last \
    --collect-video-mode skip \
    --baseline-video-mode skip \
    --eval-video-mode skip \
    --final-eval-video-mode skip \
    --skip-baseline-eval \
    --skip-iteration-eval \
    --skip-final-eval \
    $( [[ "${NORMALIZE_ADVANTAGE}" == "1" ]] && printf '%s' "--normalize-advantage" ) \
    $( [[ "${CLIP_VLOSS}" == "1" ]] && printf '%s' "--clip-vloss" ) \
    --skip-video \
    --out-dir "${pilot_out}"

  local pilot_run_dir
  pilot_run_dir="$(find_latest_subdir "${pilot_out}")"
  if [[ -z "${pilot_run_dir}" || ! -d "${pilot_run_dir}" ]]; then
    echo "failed to resolve pilot run dir under ${pilot_out}" >&2
    exit 1
  fi
  local pilot_summary="${pilot_run_dir}/pilot_summary.json"
  if [[ ! -f "${pilot_summary}" ]]; then
    echo "missing pilot summary: ${pilot_summary}" >&2
    exit 1
  fi

  local next_model_path
  next_model_path="$("${PYTHON_BIN}" - <<'PY' "${pilot_summary}"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(payload.get("final_model_path") or payload.get("current_model_path") or ""))
PY
)"
  if [[ -z "${next_model_path}" || ! -f "${next_model_path}" ]]; then
    echo "invalid next model path from ${pilot_summary}: ${next_model_path}" >&2
    exit 1
  fi

  cat > "${pilot_run_dir}/suite_iteration.json" <<EOF
{
  "experiment_name": "${EXPERIMENT_NAME}",
  "fold_name": "${FOLD_NAME}",
  "iteration_idx": ${iteration_idx},
  "collect_bank_dir": "${collect_bank_dir}",
  "env_reward_scale": ${ENV_REWARD_SCALE},
  "input_model_path": "${input_model_path}",
  "next_model_path": "${next_model_path}",
  "pilot_summary": "${pilot_summary}"
}
EOF

  RUN_TRAINING_NEXT_MODEL_PATH="${next_model_path}"
}

main() {
  ensure_bank_views

  local fold_root="${BANK_VIEW_ROOT}/cv_folds/${FOLD_NAME}"
  local train_bank_dir="${fold_root}/train"
  local probe_base_dir="${fold_root}"
  if [[ ! -f "${train_bank_dir}/bank_manifest.json" ]]; then
    echo "missing train bank for ${FOLD_NAME}: ${train_bank_dir}" >&2
    exit 1
  fi

  log "source_run_root=${SOURCE_RUN_ROOT:-n/a}"
  log "bank_view_root=${BANK_VIEW_ROOT}"
  log "fold_name=${FOLD_NAME}"
  log "train_bank_dir=${train_bank_dir}"
  log "experiment=${EXPERIMENT_NAME}"
  log "env_reward_scale=${ENV_REWARD_SCALE}"
  log "out_root=${OUT_ROOT}"

  local current_model_path="${MODEL_PATH}"
  local baseline_video_root="${OUT_ROOT}/baseline/video_full16_x${BASELINE_VIDEO_EPISODES}"
  local baseline_val_root="${OUT_ROOT}/baseline/val_x${VAL_BASELINE_EPISODES}"
  local baseline_test_root="${OUT_ROOT}/baseline/test_x${TEST_BASELINE_EPISODES}"
  local baseline_val_summary=""
  local baseline_test_summary=""

  if [[ "${RUN_VIDEO_PROBE}" == "1" ]]; then
    run_probe "${BANK_VIEW_ROOT}/splits" "full16" "baseline video full16 x${BASELINE_VIDEO_EPISODES}" "${current_model_path}" "${BASELINE_VIDEO_EPISODES}" "${VIDEO_WORKERS}" 0 "${baseline_video_root}"
  fi

  run_probe "${probe_base_dir}" "val" "baseline val x${VAL_BASELINE_EPISODES}" "${current_model_path}" "${VAL_BASELINE_EPISODES}" "${EVAL_WORKERS}" 1 "${baseline_val_root}"
  baseline_val_summary="${baseline_val_root}/summary.json"

  run_probe "${probe_base_dir}" "test" "baseline test x${TEST_BASELINE_EPISODES}" "${current_model_path}" "${TEST_BASELINE_EPISODES}" "${FINAL_PROBE_WORKERS}" 1 "${baseline_test_root}"
  baseline_test_summary="${baseline_test_root}/summary.json"

  local experiment_root="${OUT_ROOT}/${EXPERIMENT_NAME}"
  mkdir -p "${experiment_root}"
  printf '%s\n' "${current_model_path}" > "${experiment_root}/initial_model_path.txt"

  local iteration_idx
  for iteration_idx in $(seq 1 "${TRAIN_ITERS}"); do
    run_training_iteration "${train_bank_dir}" "${iteration_idx}" "${current_model_path}" "${experiment_root}"
    current_model_path="${RUN_TRAINING_NEXT_MODEL_PATH}"
    printf '%s\n' "${current_model_path}" > "${experiment_root}/latest_model_path.txt"
    if (( iteration_idx % CHECKPOINT_EVAL_EVERY == 0 )); then
      local eval_root="${experiment_root}/evals/iter_$(printf '%03d' "${iteration_idx}")_val_x${VAL_EVAL_EPISODES}"
      run_probe "${probe_base_dir}" "val" "${EXPERIMENT_NAME} iter_${iteration_idx} val_x${VAL_EVAL_EPISODES}" "${current_model_path}" "${VAL_EVAL_EPISODES}" "${EVAL_WORKERS}" 1 "${eval_root}" "${baseline_val_summary}"
    fi
  done

  RUN_ROOT="${OUT_ROOT}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  METRIC_GROUP=full16 \
  METRIC_KEY=success_rate \
  ASSET_DIR="${probe_base_dir}" \
  BANK_NAME=test \
  EPISODES="${BEST_TEST_EPISODES}" \
  WORLD_WORKERS="${FINAL_PROBE_WORKERS}" \
  SKIP_VIDEO=1 \
  BASELINE_SUMMARY="${baseline_test_summary}" \
  OUT_ROOT="${experiment_root}/best_checkpoint_test/full16_x${BEST_TEST_EPISODES}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_maze_t6_best_checkpoint_probe.sh"

  cat > "${OUT_ROOT}/SUITE_SUMMARY.md" <<EOF
# Maze T6 Family CV Suite

- experiment: \`${EXPERIMENT_NAME}\`
- fold: \`${FOLD_NAME}\`
- source_run_root: \`${SOURCE_RUN_ROOT}\`
- bank_view_root: \`${BANK_VIEW_ROOT}\`
- train_bank: \`${train_bank_dir}\`
- baseline_video: \`${baseline_video_root}\`
- baseline_val: \`${baseline_val_root}/split_report.md\`
- baseline_test: \`${baseline_test_root}/split_report.md\`
- best_checkpoint_test: \`${experiment_root}/best_checkpoint_test/full16_x${BEST_TEST_EPISODES}/split_report.md\`
EOF

  log "done out_root=${OUT_ROOT}"
}

main "$@"
