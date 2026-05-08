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
TASK_GROUP_PREFIX="${TASK_GROUP_PREFIX:-mine_o2_trained_collect_eval_repeats}"
SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
MINE_LAYOUT_BACKEND="${MINE_LAYOUT_BACKEND:-procedural}"
MINE_ANCHOR_MODE="${MINE_ANCHOR_MODE:-source_or_fallback}"
RUN_ROOT="${RUN_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-auto_goal}"
GOAL_SPEC="${GOAL_SPEC:-}"

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_ROOT="${ASSET_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm}"
  ASSET_DIR="$(find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

BASE_SEEDS="${BASE_SEEDS:-1,17,33}"
BLOCKS="${BLOCKS:-1,2,3,4,5}"
ITERATION="${ITERATION:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
EVAL_BANK_VIDEO_MODE="${EVAL_BANK_VIDEO_MODE:-skip}"
EVAL_BANK_SKIP_VIDEO="${EVAL_BANK_SKIP_VIDEO:-1}"
COLLECT_SKIP_VIDEO="${COLLECT_SKIP_VIDEO:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/o2_trained_collect_eval_repeats/$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<'EOF'
usage:
  RUN_ROOT=/home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_mine_o2_straight_generalization_blocks_vm_20260427_111330 \
  MODEL_PATH=/home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_mine_o2_straight_generalization_blocks_vm_20260427_111330/blocks/block_005/20260427_165139/iterations/iter_002/update/20260427_180256/model.pt \
  ASSET_DIR=/home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/mine_o2_straight_generalization_assets_vm/20260427_054807 \
  bash scripts/run_o2_trained_collect_eval_repeats.sh

optional env vars:
  BASE_SEEDS=1,17,33
  BLOCKS=1,2,3,4,5
  ITERATION=1
  EVAL_EPISODES=64
  COLLECT_EPISODES=64
  EVAL_WORKERS=4
  COLLECT_WORKERS=4
  EVAL_BANK_VIDEO_MODE=save
  EVAL_BANK_SKIP_VIDEO=0
  COLLECT_SKIP_VIDEO=0
  OUT_ROOT=/path/to/output_root
EOF
}

if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
  usage >&2
  echo "RUN_ROOT is missing or invalid." >&2
  exit 1
fi
if [[ -z "${MODEL_PATH}" || ( "${MODEL_PATH}" != hf:* && ! -f "${MODEL_PATH}" ) ]]; then
  usage >&2
  echo "MODEL_PATH is missing or invalid." >&2
  exit 1
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  usage >&2
  echo "ASSET_DIR is missing or invalid." >&2
  exit 1
fi

COLLECT_PLAN_JSON="${ASSET_DIR}/collect_plan.json"
EVAL_BANK_DIR="${ASSET_DIR}/eval_bank"

if [[ ! -d "${SOURCE_ENV_CONF_DIR}" ]]; then
  echo "SOURCE_ENV_CONF_DIR not found: ${SOURCE_ENV_CONF_DIR}" >&2
  exit 1
fi
if [[ ! -f "${COLLECT_PLAN_JSON}" ]]; then
  echo "collect_plan.json not found: ${COLLECT_PLAN_JSON}" >&2
  exit 1
fi
if [[ ! -d "${EVAL_BANK_DIR}" ]]; then
  echo "eval_bank not found under ${ASSET_DIR}" >&2
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

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

configure_torch_cuda_runtime
mkdir -p "${OUT_ROOT}"

GOAL_ARGS=()
if [[ -n "${GOAL_SPEC}" ]]; then
  if [[ ! -e "${GOAL_SPEC}" ]]; then
    echo "GOAL_SPEC does not exist: ${GOAL_SPEC}" >&2
    exit 1
  fi
  GOAL_ARGS=(--goal-spec "${GOAL_SPEC}")
else
  GOAL_ARGS=(--auto-goal)
fi
echo "[o2-trained-repeats] goal_protocol=${GOAL_PROTOCOL}"
echo "[o2-trained-repeats] goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}"

run_repeat() {
  local base_seed="$1"
  local rep_tag
  rep_tag="$(printf 'seed_%03d' "${base_seed}")"
  local rep_out="${OUT_ROOT}/${rep_tag}"
  local eval_out="${rep_out}/eval_bank"
  local collect_out="${rep_out}/collect_blocks"
  mkdir -p "${rep_out}"

  local cmd=(
    "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_ppo_pilot
    --env-source rocket2_official
    --task-group "${TASK_GROUP_PREFIX}_${rep_tag}"
    --task-group-path "${SOURCE_ENV_CONF_DIR}"
    --worldgen-source-dir "${SOURCE_ENV_CONF_DIR}"
    --worldgen-plan-json "${COLLECT_PLAN_JSON}"
    --mine-layout-backend "${MINE_LAYOUT_BACKEND}"
    --mine-anchor-mode "${MINE_ANCHOR_MODE}"
    --tasks "${TASK}"
    "${GOAL_ARGS[@]}"
    --collect-world-mode plan_exact
    --eval-world-mode fixed_bank
    --final-eval-world-mode fixed_bank
    --collect-world-instances 1
    --eval-world-instances 1
    --final-eval-world-instances 1
    --collect-parallel-workers 1
    --eval-parallel-workers "${EVAL_WORKERS}"
    --final-eval-parallel-workers 1
    --collect-episodes-per-task 1
    --eval-episodes-per-task "${EVAL_EPISODES}"
    --train-iters 0
    --base-seed "${base_seed}"
    --seed-step "${SEED_STEP}"
    --sampling-base-seed "${base_seed}"
    --sampling-seed-step "${SAMPLING_SEED_STEP}"
    --episode-retries "${EPISODE_RETRIES}"
    --step-budget-override "${STEP_BUDGET}"
    --model-path "${MODEL_PATH}"
    --cfg-coef "${CFG_COEF}"
    --cfg-policy-mode "${CFG_POLICY_MODE}"
    --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}"
    --final-eval-model-mode last
    --baseline-video-mode "${EVAL_BANK_VIDEO_MODE}"
    --collect-video-mode skip
    --eval-video-mode skip
    --final-eval-video-mode skip
    --eval-fixed-bank-dir "${EVAL_BANK_DIR}"
    --skip-final-eval
    --out-dir "${eval_out}"
  )
  if [[ "${STOP_ON_SUCCESS}" == "1" ]]; then
    cmd+=( --stop-on-success )
  fi
  if [[ "${EVAL_BANK_SKIP_VIDEO}" == "1" ]]; then
    cmd+=( --skip-video )
  fi

  echo "[o2-trained-repeats] start rep=${rep_tag} eval_bank base_seed=${base_seed}"
  "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" "${cmd[@]}"

  echo "[o2-trained-repeats] start rep=${rep_tag} collect blocks=${BLOCKS} base_seed=${base_seed}"
  RUN_ROOT="${RUN_ROOT}" \
  BLOCKS="${BLOCKS}" \
  ITERATION="${ITERATION}" \
  EPISODES="${COLLECT_EPISODES}" \
  WORKERS="${COLLECT_WORKERS}" \
  BASE_SEED="${base_seed}" \
  SEED_STEP="${SEED_STEP}" \
  SAMPLING_BASE_SEED="${base_seed}" \
  SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP}" \
  EPISODE_RETRIES="${EPISODE_RETRIES}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  MODEL_PATH="${MODEL_PATH}" \
  CFG_COEF="${CFG_COEF}" \
  CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
  CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
  STOP_ON_SUCCESS="${STOP_ON_SUCCESS}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_SPEC="${GOAL_SPEC}" \
  SKIP_VIDEO="${COLLECT_SKIP_VIDEO}" \
  OUT_ROOT="${collect_out}" \
  bash "${ROOT_DIR}/scripts/run_collect_world_bc_baseline_probe.sh"

  "${PYTHON_BIN}" - <<'PY' "${eval_out}" "${collect_out}" "${rep_out}" "${rep_tag}" "${base_seed}"
import json
import sys
from pathlib import Path

eval_out = Path(sys.argv[1])
collect_out = Path(sys.argv[2])
rep_out = Path(sys.argv[3])
rep_tag = sys.argv[4]
base_seed = int(sys.argv[5])

pilot_state_path = eval_out / "pilot_state.json"
if not pilot_state_path.exists():
    candidates = sorted(eval_out.glob("*/pilot_state.json"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate pilot_state.json under {eval_out}")
    pilot_state_path = candidates[-1]

pilot_state = json.loads(pilot_state_path.read_text(encoding="utf-8"))
baseline_summary = next(iter((pilot_state.get("baseline_summary") or {}).values()))
eval_payload = {
    "episodes": int(baseline_summary.get("episodes", 0) or 0),
    "success_rate": float(baseline_summary.get("auto_success_rate", 0.0) or 0.0),
    "mean_steps": float(baseline_summary.get("mean_steps", 0.0) or 0.0),
    "pilot_state_path": str(pilot_state_path),
}

collect_rows = []
for summary_path in sorted(collect_out.glob("block_*/block_summary.json")):
    collect_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))

payload = {
    "rep": rep_tag,
    "base_seed": base_seed,
    "model_path": str(pilot_state.get("model_path") or ""),
    "eval_bank": eval_payload,
    "collect_blocks": collect_rows,
}
(rep_out / "rep_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY
}

split_csv "${BASE_SEEDS}"
for base_seed in "${SPLIT_ITEMS[@]}"; do
  run_repeat "${base_seed}"
done

"${PYTHON_BIN}" - <<'PY' "${OUT_ROOT}"
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
rows = []
for rep_summary in sorted(out_root.glob("seed_*/rep_summary.json")):
    rows.append(json.loads(rep_summary.read_text(encoding="utf-8")))

(out_root / "summary.json").write_text(
    json.dumps({"runs": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps({"runs": rows}, ensure_ascii=False))
PY

echo "[o2-trained-repeats] done out_root=${OUT_ROOT}"
