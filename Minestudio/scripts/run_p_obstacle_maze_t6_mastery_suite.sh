#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"

setup_minestudio_runtime
require_xvfb_run
ensure_minestudio_engine

TOOLS_PY="${ROOT_DIR}/scripts/maze_t6_uniform_reward_tools.py"

TASK="${TASK:-mine_coal}"
RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_mastery_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/${RUN_TAG}}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"

MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_waypoint_mastery_full16}"
ENV_REWARD_SCALE="${ENV_REWARD_SCALE:-1.0}"

ASSET_DIR="${ASSET_DIR:-}"
BANK_VIEW_ROOT="${BANK_VIEW_ROOT:-}"
TRAIN_BANK_DIR="${TRAIN_BANK_DIR:-}"
PROBE_ASSET_DIR="${PROBE_ASSET_DIR:-}"
PROBE_BANK_NAME="${PROBE_BANK_NAME:-full16}"

BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"

TRAIN_ITERS="${TRAIN_ITERS:-24}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
COLLECT_WORLD_INSTANCES="${COLLECT_WORLD_INSTANCES:-16}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
COLLECT_MODE="${COLLECT_MODE:-balanced_full16}"
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

VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
BASELINE_VIDEO_EPISODES="${BASELINE_VIDEO_EPISODES:-1}"
BASELINE_EVAL_EPISODES="${BASELINE_EVAL_EPISODES:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
CHECKPOINT_EVAL_EVERY="${CHECKPOINT_EVAL_EVERY:-4}"
CHECKPOINT_EVAL_EPISODES="${CHECKPOINT_EVAL_EPISODES:-4}"
FINAL_PROBE_EPISODES="${FINAL_PROBE_EPISODES:-16}"
FINAL_PROBE_WORKERS="${FINAL_PROBE_WORKERS:-4}"

RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE:-0}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"

mkdir -p "${OUT_ROOT}"
FULL16_ROUND_ROBIN_VARIANTS=()

log() {
  printf '[maze-t6-mastery] %s\n' "$*" >&2
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
  local required_rel="splits/full16/bank_manifest.json"
  if [[ -n "${TRAIN_BANK_DIR}" && -f "${TRAIN_BANK_DIR}/bank_manifest.json" ]]; then
    BANK_VIEW_ROOT="$(cd "${TRAIN_BANK_DIR}/../.." && pwd)"
    PROBE_ASSET_DIR="${PROBE_ASSET_DIR:-${BANK_VIEW_ROOT}/splits}"
    return 0
  fi
  if [[ -n "${BANK_VIEW_ROOT}" && -f "${BANK_VIEW_ROOT}/${required_rel}" ]]; then
    TRAIN_BANK_DIR="${BANK_VIEW_ROOT}/splits/full16"
    PROBE_ASSET_DIR="${PROBE_ASSET_DIR:-${BANK_VIEW_ROOT}/splits}"
    return 0
  fi
  if [[ -n "${SOURCE_RUN_ROOT}" && -f "${SOURCE_RUN_ROOT}/bank_views/${required_rel}" ]]; then
    BANK_VIEW_ROOT="${SOURCE_RUN_ROOT}/bank_views"
    TRAIN_BANK_DIR="${BANK_VIEW_ROOT}/splits/full16"
    PROBE_ASSET_DIR="${PROBE_ASSET_DIR:-${BANK_VIEW_ROOT}/splits}"
    return 0
  fi

  resolve_asset_dir_from_source
  if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
    echo "Need ASSET_DIR or SOURCE_RUN_ROOT with assets/bank_views." >&2
    exit 1
  fi
  BANK_VIEW_ROOT="${OUT_ROOT}/bank_views"
  "${PYTHON_BIN}" "${TOOLS_PY}" make-bank-views --asset-dir "${ASSET_DIR}" --out-dir "${BANK_VIEW_ROOT}" >/dev/null
  TRAIN_BANK_DIR="${BANK_VIEW_ROOT}/splits/full16"
  PROBE_ASSET_DIR="${PROBE_ASSET_DIR:-${BANK_VIEW_ROOT}/splits}"
}

prepare_collect_schedule() {
  if [[ "${COLLECT_MODE}" != "round_robin_full16" ]]; then
    return 0
  fi
  mapfile -t FULL16_ROUND_ROBIN_VARIANTS < <(
    "${PYTHON_BIN}" - <<'PY' "${TRAIN_BANK_DIR}/bank_manifest.json"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload.get("instance_worlds") or []:
    variant_id = str(row.get("path_obstacle_variant_id") or "").strip()
    if variant_id:
        print(variant_id)
PY
  )
  if [[ "${#FULL16_ROUND_ROBIN_VARIANTS[@]}" -eq 0 ]]; then
    echo "failed to build round-robin variant schedule from ${TRAIN_BANK_DIR}/bank_manifest.json" >&2
    exit 1
  fi
}

run_probe() {
  local label="$1"
  local model_path="$2"
  local episodes="$3"
  local workers="$4"
  local skip_video="$5"
  local out_root="$6"
  local baseline_summary="${7:-}"

  mkdir -p "${out_root}"
  log "probe label=${label} episodes=${episodes} workers=${workers} skip_video=${skip_video}"
  ASSET_DIR="${PROBE_ASSET_DIR}" \
  BANK_NAME="${PROBE_BANK_NAME}" \
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
  local iteration_idx="$1"
  local input_model_path="$2"
  local experiment_root="$3"
  local iter_tag
  iter_tag="$(printf 'iter_%03d' "${iteration_idx}")"
  local pilot_out="${experiment_root}/training/${iter_tag}"
  local iter_base_seed=$(( BASE_SEED + (iteration_idx - 1) * COLLECT_EPISODES ))
  local iter_sampling_seed=$(( SAMPLING_BASE_SEED + (iteration_idx - 1) * COLLECT_EPISODES ))
  local collect_bank_dir="${TRAIN_BANK_DIR}"
  local collect_world_instances="${COLLECT_WORLD_INSTANCES}"
  local variant_label="all16"

  if [[ "${COLLECT_MODE}" == "round_robin_full16" ]]; then
    local variant_idx=$(( (iteration_idx - 1) % ${#FULL16_ROUND_ROBIN_VARIANTS[@]} ))
    variant_label="${FULL16_ROUND_ROBIN_VARIANTS[$variant_idx]}"
    collect_bank_dir="${BANK_VIEW_ROOT}/singletons/${variant_label}"
    collect_world_instances=1
    if [[ ! -f "${collect_bank_dir}/bank_manifest.json" ]]; then
      echo "missing singleton collect bank: ${collect_bank_dir}" >&2
      exit 1
    fi
  fi

  mkdir -p "${pilot_out}"
  log "train iter=${iter_tag} collect_mode=${COLLECT_MODE} variant=${variant_label} collect_bank=${collect_bank_dir} env_reward_scale=${ENV_REWARD_SCALE}"

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
    --eval-fixed-bank-dir "${TRAIN_BANK_DIR}" \
    --final-eval-fixed-bank-dir "${TRAIN_BANK_DIR}" \
    --collect-world-instances "${collect_world_instances}" \
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
  "iteration_idx": ${iteration_idx},
  "collect_bank_dir": "${collect_bank_dir}",
  "collect_mode": "${COLLECT_MODE}",
  "collect_variant_id": "${variant_label}",
  "env_reward_scale": ${ENV_REWARD_SCALE},
  "input_model_path": "${input_model_path}",
  "next_model_path": "${next_model_path}",
  "pilot_summary": "${pilot_summary}"
}
EOF

  RUN_TRAINING_NEXT_MODEL_PATH="${next_model_path}"
}

should_checkpoint_eval() {
  local iter="$1"
  if (( iter % CHECKPOINT_EVAL_EVERY == 0 )); then
    return 0
  fi
  return 1
}

write_summary() {
  local experiment_root="${OUT_ROOT}/${EXPERIMENT_NAME}"
  cat > "${OUT_ROOT}/SUITE_SUMMARY.md" <<EOF
# Maze T6 Mastery Suite

- experiment: \`${EXPERIMENT_NAME}\`
- collect_mode: \`${COLLECT_MODE}\`
- train_iters: \`${TRAIN_ITERS}\`
- collect_episodes_per_iter: \`${COLLECT_EPISODES}\`
- source_run_root: \`${SOURCE_RUN_ROOT}\`
- bank_view_root: \`${BANK_VIEW_ROOT}\`
- train_bank: \`${TRAIN_BANK_DIR}\`
- latest_model_path: \`$(cat "${experiment_root}/latest_model_path.txt")\`
- baseline_eval: \`${experiment_root}/../baseline/full16_x${BASELINE_EVAL_EPISODES}/split_report.md\`
- final_probe: \`${experiment_root}/final_probe/full16_x${FINAL_PROBE_EPISODES}/split_report.md\`
EOF
}

main() {
  ensure_bank_views
  prepare_collect_schedule

  log "source_run_root=${SOURCE_RUN_ROOT:-n/a}"
  log "bank_view_root=${BANK_VIEW_ROOT}"
  log "train_bank_dir=${TRAIN_BANK_DIR}"
  log "probe_asset_dir=${PROBE_ASSET_DIR}"
  log "experiment=${EXPERIMENT_NAME}"
  log "collect_mode=${COLLECT_MODE}"
  log "env_reward_scale=${ENV_REWARD_SCALE}"
  log "out_root=${OUT_ROOT}"

  local baseline_root="${OUT_ROOT}/baseline/full16_x${BASELINE_EVAL_EPISODES}"
  local baseline_video_root="${OUT_ROOT}/baseline/video_full16_x${BASELINE_VIDEO_EPISODES}"
  local baseline_summary=""
  local current_model_path="${MODEL_PATH}"

  if [[ "${RUN_VIDEO_PROBE}" == "1" ]]; then
    run_probe \
      "baseline video full16_x${BASELINE_VIDEO_EPISODES}" \
      "${current_model_path}" \
      "${BASELINE_VIDEO_EPISODES}" \
      "${VIDEO_WORKERS}" \
      0 \
      "${baseline_video_root}"
  fi

  if [[ "${RUN_BASELINE_EVAL}" == "1" ]]; then
    run_probe \
      "baseline full16_x${BASELINE_EVAL_EPISODES}" \
      "${current_model_path}" \
      "${BASELINE_EVAL_EPISODES}" \
      "${EVAL_WORKERS}" \
      1 \
      "${baseline_root}"
    baseline_summary="${baseline_root}/summary.json"
  fi

  local experiment_root="${OUT_ROOT}/${EXPERIMENT_NAME}"
  mkdir -p "${experiment_root}"
  printf '%s\n' "${current_model_path}" > "${experiment_root}/initial_model_path.txt"

  local iteration_idx
  for iteration_idx in $(seq 1 "${TRAIN_ITERS}"); do
    run_training_iteration "${iteration_idx}" "${current_model_path}" "${experiment_root}"
    current_model_path="${RUN_TRAINING_NEXT_MODEL_PATH}"
    printf '%s\n' "${current_model_path}" > "${experiment_root}/latest_model_path.txt"
    if should_checkpoint_eval "${iteration_idx}"; then
      local eval_root="${experiment_root}/evals/iter_$(printf '%03d' "${iteration_idx}")_full16_x${CHECKPOINT_EVAL_EPISODES}"
      run_probe \
        "${EXPERIMENT_NAME} iter_${iteration_idx} full16_x${CHECKPOINT_EVAL_EPISODES}" \
        "${current_model_path}" \
        "${CHECKPOINT_EVAL_EPISODES}" \
        "${EVAL_WORKERS}" \
        1 \
        "${eval_root}" \
        "${baseline_summary}"
    fi
  done

  local final_root="${experiment_root}/final_probe/full16_x${FINAL_PROBE_EPISODES}"
  run_probe \
    "${EXPERIMENT_NAME} final full16_x${FINAL_PROBE_EPISODES}" \
    "${current_model_path}" \
    "${FINAL_PROBE_EPISODES}" \
    "${FINAL_PROBE_WORKERS}" \
    1 \
    "${final_root}" \
    "${baseline_summary}"

  write_summary
  log "done out_root=${OUT_ROOT}"
}

main "$@"
