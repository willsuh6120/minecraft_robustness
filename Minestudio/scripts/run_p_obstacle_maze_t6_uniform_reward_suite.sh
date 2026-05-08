#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT_DIR}/scripts/_minestudio_runtime.sh"

setup_minestudio_runtime
require_xvfb_run

TOOLS_PY="${ROOT_DIR}/scripts/maze_t6_uniform_reward_tools.py"

TASK="${TASK:-mine_coal}"
RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_uniform_reward_suite_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/${RUN_TAG}}"
ASSET_ROOT="${ASSET_ROOT:-${OUT_ROOT}/assets}"
ASSET_DIR="${ASSET_DIR:-}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"

MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET:-hard_v6_maze_t6}"
P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT:-2}"
PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE:-0.125}"
BANK_WORKERS="${BANK_WORKERS:-4}"

BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"

COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
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

VIDEO_WORKERS="${VIDEO_WORKERS:-4}"
BASELINE_VIDEO_EPISODES="${BASELINE_VIDEO_EPISODES:-1}"
BASELINE_EVAL_EPISODES="${BASELINE_EVAL_EPISODES:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
FINAL_PROBE_EPISODES="${FINAL_PROBE_EPISODES:-16}"
FINAL_PROBE_WORKERS="${FINAL_PROBE_WORKERS:-4}"

RUN_VIDEO_PROBE="${RUN_VIDEO_PROBE:-0}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
RUN_CONTROL_A="${RUN_CONTROL_A:-1}"
RUN_MAIN_B="${RUN_MAIN_B:-1}"

mkdir -p "${OUT_ROOT}"

mapfile -t ROUND_ROBIN_SCHEDULE < <("${PYTHON_BIN}" "${TOOLS_PY}" field --name round_robin_schedule)

log() {
  printf '[maze-t6-uniform-suite] %s\n' "$*" >&2
}

find_latest_subdir() {
  find "$1" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

ensure_asset_dir() {
  if [[ -n "${ASSET_DIR}" && -d "${ASSET_DIR}" ]]; then
    log "using asset_dir=${ASSET_DIR}"
    return 0
  fi
  log "generating asset bank under ${ASSET_ROOT}"
  TASK="${TASK}" \
  BANK_WORKERS="${BANK_WORKERS}" \
  P_OBSTACLE_VARIANT_SET="${P_OBSTACLE_VARIANT_SET}" \
  P_OBSTACLE_HEIGHT="${P_OBSTACLE_HEIGHT}" \
  PATH_PROGRESS_REWARD_PER_ZONE="${PATH_PROGRESS_REWARD_PER_ZONE}" \
  OUT_DIR="${ASSET_ROOT}" \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  bash "${ROOT_DIR}/scripts/prepare_mine_p_auto_goal_eval_bank_local.sh"
  ASSET_DIR="$(find_latest_subdir "${ASSET_ROOT}")"
  if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
    echo "failed to resolve ASSET_DIR under ${ASSET_ROOT}" >&2
    exit 1
  fi
  log "generated asset_dir=${ASSET_DIR}"
}

prepare_bank_views() {
  BANK_VIEW_ROOT="${OUT_ROOT}/bank_views"
  "${PYTHON_BIN}" "${TOOLS_PY}" make-bank-views \
    --asset-dir "${ASSET_DIR}" \
    --out-dir "${BANK_VIEW_ROOT}" >/dev/null
  export BANK_VIEW_ROOT
  log "bank_view_root=${BANK_VIEW_ROOT}"
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
  ASSET_DIR="${ASSET_DIR}" \
  BANK_NAME=eval_bank \
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
  local experiment_name="$1"
  local env_reward_scale="$2"
  local iteration_idx="$3"
  local input_model_path="$4"
  local experiment_root="$5"

  local variant="${ROUND_ROBIN_SCHEDULE[$((iteration_idx - 1))]}"
  local iter_tag
  iter_tag="$(printf 'iter_%03d' "${iteration_idx}")"
  local pilot_out="${experiment_root}/training/${iter_tag}"
  local collect_bank_dir="${BANK_VIEW_ROOT}/singletons/${variant}"
  local iter_base_seed=$(( BASE_SEED + (iteration_idx - 1) * COLLECT_EPISODES ))
  local iter_sampling_seed=$(( SAMPLING_BASE_SEED + (iteration_idx - 1) * COLLECT_EPISODES ))

  mkdir -p "${pilot_out}"
  log "train experiment=${experiment_name} iter=${iter_tag} variant=${variant} env_reward_scale=${env_reward_scale}"

  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" \
  "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot \
    --env-source rocket2_official \
    --protocol ours_v1 \
    --task-group "maze_t6_${experiment_name}_${iter_tag}" \
    --task-group-path "${SOURCE_ENV_CONF_DIR}" \
    --worldgen-source-dir "${SOURCE_ENV_CONF_DIR}" \
    --tasks "${TASK}" \
    --auto-goal \
    --collect-world-mode fixed_bank \
    --eval-world-mode fixed_bank \
    --final-eval-world-mode fixed_bank \
    --collect-fixed-bank-dir "${collect_bank_dir}" \
    --eval-fixed-bank-dir "${BANK_VIEW_ROOT}/splits/full16" \
    --final-eval-fixed-bank-dir "${BANK_VIEW_ROOT}/splits/full16" \
    --collect-world-instances 1 \
    --eval-world-instances 1 \
    --final-eval-world-instances 1 \
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
    --env-reward-scale "${env_reward_scale}" \
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
  "experiment_name": "${experiment_name}",
  "iteration_idx": ${iteration_idx},
  "variant_id": "${variant}",
  "env_reward_scale": ${env_reward_scale},
  "input_model_path": "${input_model_path}",
  "next_model_path": "${next_model_path}",
  "collect_bank_dir": "${collect_bank_dir}",
  "pilot_summary": "${pilot_summary}"
}
EOF

  RUN_TRAINING_NEXT_MODEL_PATH="${next_model_path}"
}

checkpoint_eval_episodes() {
  case "$1" in
    4) printf '%s\n' "2" ;;
    8) printf '%s\n' "4" ;;
    12) printf '%s\n' "2" ;;
    16) printf '%s\n' "4" ;;
    *) printf '%s\n' "" ;;
  esac
}

run_experiment() {
  local experiment_name="$1"
  local env_reward_scale="$2"
  local experiment_root="${OUT_ROOT}/${experiment_name}"
  local current_model_path="${MODEL_PATH}"
  local iteration_idx

  mkdir -p "${experiment_root}"
  printf '%s\n' "${current_model_path}" > "${experiment_root}/initial_model_path.txt"

  for iteration_idx in $(seq 1 16); do
    run_training_iteration "${experiment_name}" "${env_reward_scale}" "${iteration_idx}" "${current_model_path}" "${experiment_root}"
    current_model_path="${RUN_TRAINING_NEXT_MODEL_PATH}"
    printf '%s\n' "${current_model_path}" > "${experiment_root}/latest_model_path.txt"

    local eval_episodes
    eval_episodes="$(checkpoint_eval_episodes "${iteration_idx}")"
    if [[ -n "${eval_episodes}" ]]; then
      local eval_root="${experiment_root}/evals/iter_$(printf '%03d' "${iteration_idx}")_full16_x${eval_episodes}"
      run_probe \
        "${experiment_name} iter_${iteration_idx} full16_x${eval_episodes}" \
        "${current_model_path}" \
        "${eval_episodes}" \
        "${EVAL_WORKERS}" \
        1 \
        "${eval_root}" \
        "${BASELINE_EVAL_SUMMARY}"
    fi
  done

  local final_root="${experiment_root}/final_probe/full16_x${FINAL_PROBE_EPISODES}"
  run_probe \
    "${experiment_name} final full16_x${FINAL_PROBE_EPISODES}" \
    "${current_model_path}" \
    "${FINAL_PROBE_EPISODES}" \
    "${FINAL_PROBE_WORKERS}" \
    1 \
    "${final_root}" \
    "${BASELINE_EVAL_SUMMARY}"
}

write_suite_summary() {
  "${PYTHON_BIN}" - <<'PY' "${OUT_ROOT}" "${FINAL_PROBE_EPISODES}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
final_probe_episodes = int(sys.argv[2])
rows = []
for experiment in ("control_sparse_uniform", "main_waypoint_uniform"):
    report_path = root / experiment / "final_probe" / f"full16_x{final_probe_episodes}" / "split_report.json"
    if not report_path.exists():
        continue
    report = json.loads(report_path.read_text(encoding="utf-8"))
    groups = report.get("groups") or {}
    deltas = report.get("deltas") or {}
    rows.append(
        {
            "experiment": experiment,
            "train8": groups.get("train8", {}).get("success_rate"),
            "heldout8": groups.get("heldout8", {}).get("success_rate"),
            "full16": groups.get("full16", {}).get("success_rate"),
            "delta_train8": deltas.get("delta_train8"),
            "delta_heldout8": deltas.get("delta_heldout8"),
            "delta_full16": deltas.get("delta_full16"),
            "transfer_ratio": deltas.get("transfer_ratio"),
        }
    )

def fmt_rate(value):
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"

def fmt_float(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"

lines = [
    "# Maze T6 Uniform Reward Suite",
    "",
    "| experiment | train8 | heldout8 | full16 | delta_train8 | delta_heldout8 | delta_full16 | transfer_ratio |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['experiment']} | {fmt_rate(row['train8'])} | {fmt_rate(row['heldout8'])} | "
        f"{fmt_rate(row['full16'])} | {fmt_rate(row['delta_train8'])} | {fmt_rate(row['delta_heldout8'])} | "
        f"{fmt_rate(row['delta_full16'])} | {fmt_float(row['transfer_ratio'])} |"
    )
(root / "SUITE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

ensure_asset_dir
prepare_bank_views

BASELINE_VIDEO_ROOT="${OUT_ROOT}/baseline/video_full16_x${BASELINE_VIDEO_EPISODES}"
BASELINE_EVAL_ROOT="${OUT_ROOT}/baseline/full16_x${BASELINE_EVAL_EPISODES}"
BASELINE_EVAL_SUMMARY="${BASELINE_EVAL_ROOT}/summary.json"

if [[ "${RUN_VIDEO_PROBE}" == "1" ]]; then
  run_probe \
    "baseline video full16_x${BASELINE_VIDEO_EPISODES}" \
    "${MODEL_PATH}" \
    "${BASELINE_VIDEO_EPISODES}" \
    "${VIDEO_WORKERS}" \
    0 \
    "${BASELINE_VIDEO_ROOT}"
fi

if [[ "${RUN_BASELINE_EVAL}" == "1" || ! -f "${BASELINE_EVAL_SUMMARY}" ]]; then
  run_probe \
    "baseline eval full16_x${BASELINE_EVAL_EPISODES}" \
    "${MODEL_PATH}" \
    "${BASELINE_EVAL_EPISODES}" \
    "${EVAL_WORKERS}" \
    1 \
    "${BASELINE_EVAL_ROOT}"
fi

if [[ ! -f "${BASELINE_EVAL_SUMMARY}" ]]; then
  echo "missing baseline summary: ${BASELINE_EVAL_SUMMARY}" >&2
  exit 1
fi

if [[ "${RUN_CONTROL_A}" == "1" ]]; then
  run_experiment "control_sparse_uniform" "0.0"
fi

if [[ "${RUN_MAIN_B}" == "1" ]]; then
  run_experiment "main_waypoint_uniform" "1.0"
fi

write_suite_summary
log "done out_root=${OUT_ROOT}"
