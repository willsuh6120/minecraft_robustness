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
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"

ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm}"
ASSET_DIR="${ASSET_DIR:-${ASSET_ROOT}/20260427_054807}"
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing. Set ASSET_DIR to the O2 asset directory on the VM." >&2
  exit 1
fi

EVAL_BANK_DIR="${EVAL_BANK_DIR:-${ASSET_DIR}/eval_bank}"
FINAL_EVAL_BANK_DIR="${FINAL_EVAL_BANK_DIR:-${ASSET_DIR}/final_eval_bank}"
if [[ ! -d "${EVAL_BANK_DIR}" ]]; then
  echo "Missing eval bank: ${EVAL_BANK_DIR}" >&2
  exit 1
fi

SUITE_TAG="${SUITE_TAG:-mixed_auto_goal_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/o2_mixed_collect_suite/${SUITE_TAG}}"
COLLECT_BLOCKS="${COLLECT_BLOCKS:-1,2,3,4,5}"
COLLECT_FIXED_BANK_DIR="${COLLECT_FIXED_BANK_DIR:-${SUITE_ROOT}/_mixed_collect_bank}"
PREPARE_BANK_OVERWRITE="${PREPARE_BANK_OVERWRITE:-0}"

VARIANTS="${VARIANTS:-uniform_mixed}"
BASE_SEEDS="${BASE_SEEDS:-1}"
MIXED_ITERS="${MIXED_ITERS:-10}"
COLLECT_EPISODES="${COLLECT_EPISODES:-60}"
QUOTA_COLLECT_EPISODES="${QUOTA_COLLECT_EPISODES:-160}"
QUOTA_SUCCESS_PER_WORLD="${QUOTA_SUCCESS_PER_WORLD:-2}"
QUOTA_FAILURE_PER_WORLD="${QUOTA_FAILURE_PER_WORLD:-10}"
QUOTA_MIN_SUCCESSFUL_FRAGMENTS="${QUOTA_MIN_SUCCESSFUL_FRAGMENTS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
FINAL_EVAL_WORKERS="${FINAL_EVAL_WORKERS:-1}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"

MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

BASE_SEED_STEP="${BASE_SEED_STEP:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"

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
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-heads}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
FINAL_EVAL_MODEL_MODE="${FINAL_EVAL_MODEL_MODE:-best_eval}"

NORMALIZE_ADVANTAGE="${NORMALIZE_ADVANTAGE:-1}"
CLIP_VLOSS="${CLIP_VLOSS:-1}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"

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
  for subdir in cublas/lib cudnn/lib cuda_runtime/lib curand/lib cusolver/lib cusparse/lib nvjitlink/lib; do
    if [[ -d "${site_pkg}/${subdir}" ]]; then
      lib_paths+=("${site_pkg}/${subdir}")
    fi
  done
  if (( ${#lib_paths[@]} > 0 )); then
    local joined
    joined="$(IFS=:; echo "${lib_paths[*]}")"
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

variant_min_successful_fragments() {
  case "$1" in
    uniform_no_update|no_update)
      printf '%s\n' "$(( COLLECT_EPISODES + 1 ))"
      ;;
    uniform_mixed)
      printf '%s\n' "${MIN_SUCCESSFUL_FRAGMENTS}"
      ;;
    quota_mixed|quota_balanced)
      printf '%s\n' "${QUOTA_MIN_SUCCESSFUL_FRAGMENTS}"
      ;;
    *)
      echo "Unknown O2 mixed variant: $1" >&2
      return 1
      ;;
  esac
}

variant_collect_episodes() {
  case "$1" in
    quota_mixed|quota_balanced)
      printf '%s\n' "${QUOTA_COLLECT_EPISODES}"
      ;;
    *)
      printf '%s\n' "${COLLECT_EPISODES}"
      ;;
  esac
}

mkdir -p "${SUITE_ROOT}"
configure_torch_cuda_runtime

prepare_args=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_o2_mixed_collect_bank.py"
  --asset-dir "${ASSET_DIR}"
  --env-conf-dir "${SOURCE_ENV_CONF_DIR}"
  --out-dir "${COLLECT_FIXED_BANK_DIR}"
  --blocks "${COLLECT_BLOCKS}"
  --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
  --mine-anchor-mode "${MINE_ANCHOR_MODE}"
  --python-bin "${PYTHON_BIN}"
)
if [[ "${PREPARE_BANK_OVERWRITE}" == "1" ]]; then
  prepare_args+=(--overwrite)
fi
"${prepare_args[@]}"

echo "[o2-mixed-suite] suite_root=${SUITE_ROOT}"
echo "[o2-mixed-suite] asset_dir=${ASSET_DIR}"
echo "[o2-mixed-suite] collect_fixed_bank_dir=${COLLECT_FIXED_BANK_DIR}"
echo "[o2-mixed-suite] collect_blocks=${COLLECT_BLOCKS}"
echo "[o2-mixed-suite] variants=${VARIANTS}"
echo "[o2-mixed-suite] base_seeds=${BASE_SEEDS}"
echo "[o2-mixed-suite] mixed_iters=${MIXED_ITERS}"
echo "[o2-mixed-suite] collect_episodes=${COLLECT_EPISODES}"
echo "[o2-mixed-suite] quota_collect_episodes=${QUOTA_COLLECT_EPISODES}"
echo "[o2-mixed-suite] quota_success_per_world=${QUOTA_SUCCESS_PER_WORLD}"
echo "[o2-mixed-suite] quota_failure_per_world=${QUOTA_FAILURE_PER_WORLD}"
echo "[o2-mixed-suite] eval_episodes=${EVAL_EPISODES}"
echo "[o2-mixed-suite] goal_protocol=auto_goal"

split_csv "${VARIANTS}"
VARIANT_ITEMS=( "${SPLIT_ITEMS[@]}" )
split_csv "${BASE_SEEDS}"
SEED_ITEMS=( "${SPLIT_ITEMS[@]}" )

for variant in "${VARIANT_ITEMS[@]}"; do
  variant_min_success="$(variant_min_successful_fragments "${variant}")"
  variant_collect_total="$(variant_collect_episodes "${variant}")"
  for seed in "${SEED_ITEMS[@]}"; do
    seed_tag="$(printf 'seed_%03d' "${seed}")"
    out_dir="${SUITE_ROOT}/${variant}/${seed_tag}"
    mkdir -p "${out_dir}"
    echo
    echo "==================== variant=${variant} ${seed_tag} ===================="
    echo "[o2-mixed-suite] out_dir=${out_dir}"
    echo "[o2-mixed-suite] min_successful_fragments=${variant_min_success}"
    echo "[o2-mixed-suite] collect_episodes_for_variant=${variant_collect_total}"

    cmd=(
      "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}"
      "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot
      --env-source rocket2_official
      --task-group "mine_o2_mixed_collect_${variant}"
      --task-group-path "${SOURCE_ENV_CONF_DIR}"
      --worldgen-source-dir "${SOURCE_ENV_CONF_DIR}"
      --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
      --mine-anchor-mode "${MINE_ANCHOR_MODE}"
      --tasks "${TASK}"
      --auto-goal
      --collect-world-mode fixed_bank
      --collect-fixed-bank-dir "${COLLECT_FIXED_BANK_DIR}"
      --eval-world-mode fixed_bank
      --final-eval-world-mode fixed_bank
      --collect-world-instances 5
      --eval-world-instances 1
      --final-eval-world-instances 1
      --collect-parallel-workers "${COLLECT_WORKERS}"
      --eval-parallel-workers "${EVAL_WORKERS}"
      --final-eval-parallel-workers "${FINAL_EVAL_WORKERS}"
      --collect-episodes-per-task "${variant_collect_total}"
      --eval-episodes-per-task "${EVAL_EPISODES}"
      --train-iters "${MIXED_ITERS}"
      --base-seed "${seed}"
      --seed-step "${BASE_SEED_STEP}"
      --sampling-base-seed "${seed}"
      --sampling-seed-step "${SAMPLING_SEED_STEP}"
      --episode-retries "${EPISODE_RETRIES}"
      --step-budget-override "${STEP_BUDGET}"
      --model-path "${MODEL_PATH}"
      --cfg-coef "${CFG_COEF}"
      --cfg-policy-mode "${CFG_POLICY_MODE}"
      --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}"
      --kl-anchor-model-path "${KL_ANCHOR_MODEL_PATH}"
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
      --min-successful-fragments "${variant_min_success}"
      --final-eval-model-mode "${FINAL_EVAL_MODEL_MODE}"
      --baseline-video-mode skip
      --collect-video-mode skip
      --eval-video-mode skip
      --final-eval-video-mode skip
      --eval-fixed-bank-dir "${EVAL_BANK_DIR}"
      --out-dir "${out_dir}"
    )
    case "${variant}" in
      quota_mixed|quota_balanced)
        cmd+=(
          --collect-quota-mode success_failure_per_instance
          --collect-quota-successes-per-instance "${QUOTA_SUCCESS_PER_WORLD}"
          --collect-quota-failures-per-instance "${QUOTA_FAILURE_PER_WORLD}"
        )
        ;;
    esac
    if [[ -d "${FINAL_EVAL_BANK_DIR}" ]]; then
      cmd+=(--final-eval-fixed-bank-dir "${FINAL_EVAL_BANK_DIR}")
    fi
    if [[ "${SKIP_FINAL_EVAL}" == "1" ]]; then
      cmd+=(--skip-final-eval)
    fi
    if [[ "${NORMALIZE_ADVANTAGE}" == "1" ]]; then
      cmd+=(--normalize-advantage)
    fi
    if [[ "${CLIP_VLOSS}" == "1" ]]; then
      cmd+=(--clip-vloss)
    fi
    if [[ "${STOP_ON_SUCCESS}" == "1" ]]; then
      cmd+=(--stop-on-success)
    fi
    if [[ "${SKIP_VIDEO}" == "1" ]]; then
      cmd+=(--skip-video)
    fi

    "${cmd[@]}"
  done
done

echo "[o2-mixed-suite] done suite_root=${SUITE_ROOT}"
