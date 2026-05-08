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

RUN_ROOT="${RUN_ROOT:-${1:-}}"
BLOCKS="${BLOCKS:-1,2,3,4,5}"
ITERATION="${ITERATION:-1}"
TASK="${TASK:-mine_coal}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-auto_goal}"
GOAL_SPEC="${GOAL_SPEC:-}"
EPISODES="${EPISODES:-64}"
WORKERS="${WORKERS:-4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/collect_world_bc_baseline_probe/$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<'EOF'
usage:
  RUN_ROOT=/home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/ppo_mine_o2_straight_generalization_blocks_vm_20260427_111330 \
  bash scripts/run_collect_world_bc_baseline_probe.sh

optional env vars:
  BLOCKS=1,2,3,4,5
  ITERATION=1
  EPISODES=64
  WORKERS=4
  MODEL_PATH=hf:phython96/ROCKET-2-1x-22w
  OUT_ROOT=/path/to/output_root
EOF
}

if [[ -z "${RUN_ROOT}" ]]; then
  usage >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT%/}"
ITER_TAG="$(printf 'iter_%03d' "${ITERATION}")"
mkdir -p "${OUT_ROOT}"

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

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

latest_subdir() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_block_probe() {
  local block_num="$1"
  local block_tag
  block_tag="$(printf 'block_%03d' "${block_num}")"
  local block_root="${RUN_ROOT}/blocks/${block_tag}"
  if [[ ! -d "${block_root}" ]]; then
    echo "[collect-bc-probe] skip ${block_tag}: missing ${block_root}" >&2
    return 0
  fi

  local block_run_root
  block_run_root="$(latest_subdir "${block_root}")"
  if [[ -z "${block_run_root}" || ! -d "${block_run_root}" ]]; then
    echo "[collect-bc-probe] skip ${block_tag}: no timestamped run dir" >&2
    return 0
  fi

  local collect_root="${block_run_root}/worldgen/${ITER_TAG}_collect/instance_000"
  local generated_root="${collect_root}/generated_task_groups"
  if [[ ! -d "${generated_root}" ]]; then
    echo "[collect-bc-probe] skip ${block_tag}: missing ${generated_root}" >&2
    return 0
  fi

  local task_group_path
  task_group_path="$(latest_subdir "${generated_root}")"
  if [[ -z "${task_group_path}" || ! -d "${task_group_path}" ]]; then
    echo "[collect-bc-probe] skip ${block_tag}: no generated task_group path" >&2
    return 0
  fi

  local manifest_path="${task_group_path}/worldgen_manifest.json"
  local block_out="${OUT_ROOT}/${block_tag}"
  mkdir -p "${block_out}"

  echo "[collect-bc-probe] block=${block_tag}"
  echo "[collect-bc-probe] block_run_root=${block_run_root}"
  echo "[collect-bc-probe] task_group_path=${task_group_path}"
  echo "[collect-bc-probe] goal_protocol=${GOAL_PROTOCOL}"
  echo "[collect-bc-probe] goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}"

  "${PYTHON_BIN}" - <<'PY' "${manifest_path}" "${block_tag}"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
block_tag = sys.argv[2]
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = manifest["tasks"][0]
    sugg = task.get("world_generation_suggestions") or {}
    top = sugg.get("mine_occluder_top_row") or []
    bottom = sugg.get("mine_occluder_bottom_row") or []
    payload = {
        "block": block_tag,
        "layout_seed": task.get("layout_seed"),
        "mine_occluder_variant_id": sugg.get("mine_occluder_variant_id"),
        "top_row": top,
        "bottom_row": bottom,
        "blueprint_id": task.get("blueprint_id"),
        "target_world_center": task.get("target_world_center"),
    }
    print(json.dumps(payload, ensure_ascii=False))
else:
    print(json.dumps({"block": block_tag, "warning": f"missing manifest {manifest_path}"}, ensure_ascii=False))
PY

  local active_workers="${WORKERS}"
  if (( active_workers > EPISODES )); then
    active_workers="${EPISODES}"
  fi
  if (( active_workers < 1 )); then
    active_workers=1
  fi

  local base_count=$(( EPISODES / active_workers ))
  local remainder=$(( EPISODES % active_workers ))
  local episode_prefix=0
  local pids=()
  local worker_idx

  for (( worker_idx=0; worker_idx<active_workers; worker_idx++ )); do
    local worker_episodes="${base_count}"
    if (( worker_idx < remainder )); then
      worker_episodes=$(( worker_episodes + 1 ))
    fi
    local worker_base_seed=$(( BASE_SEED + episode_prefix * (SEED_STEP > 0 ? SEED_STEP : 1) ))
    local worker_sampling_seed=$(( SAMPLING_BASE_SEED + episode_prefix * (SAMPLING_SEED_STEP > 0 ? SAMPLING_SEED_STEP : 1) ))
    episode_prefix=$(( episode_prefix + worker_episodes ))
    local worker_root="${block_out}/worker_$(printf '%03d' "${worker_idx}")"
    mkdir -p "${worker_root}"
    local task_group_name
    task_group_name="$(printf '%s_%s_bcprobe_worker_%03d' "${TASK}" "${block_tag}" "${worker_idx}")"

    echo "[collect-bc-probe] start ${block_tag} worker=${worker_idx} episodes=${worker_episodes} base_seed=${worker_base_seed} sampling_seed=${worker_sampling_seed}"

    cmd=( "${PYTHON_BIN}" -m minestudio.tutorials.inference.evaluate_rocket.interaction_crossview_rollout
      --env-source rocket2_official
      --protocol ours_v1
      --task-group "${task_group_name}"
      --task-group-path "${task_group_path}"
      --tasks "${TASK}"
      --episodes-per-task "${worker_episodes}"
      --base-seed "${worker_base_seed}"
      --seed-step "${SEED_STEP}"
      --sampling-base-seed "${worker_sampling_seed}"
      --sampling-seed-step "${SAMPLING_SEED_STEP}"
      --episode-retries "${EPISODE_RETRIES}"
      --step-budget-override "${STEP_BUDGET}"
      --model-path "${MODEL_PATH}"
      --out-dir "${worker_root}"
      "${GOAL_ARGS[@]}"
      --cfg-coef "${CFG_COEF}"
      --cfg-policy-mode "${CFG_POLICY_MODE}"
      --cfg-base-ref-model-path "${CFG_BASE_REF_MODEL_PATH}"
    )
    if [[ "${STOP_ON_SUCCESS}" == "1" ]]; then
      cmd+=( --stop-on-success )
    fi
    if [[ "${SKIP_VIDEO}" == "1" ]]; then
      cmd+=( --skip-video )
    fi

    if [[ -n "${XVFB_RUN}" ]]; then
      "${XVFB_RUN}" -a -s "${XVFB_SCREEN_ARGS}" "${cmd[@]}" &
    else
      "${cmd[@]}" &
    fi
    pids+=( "$!" )
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  "${PYTHON_BIN}" - <<'PY' "${block_out}" "${block_tag}"
import csv
import json
import sys
from pathlib import Path

block_out = Path(sys.argv[1])
block_tag = sys.argv[2]
rows = []
for worker_root in sorted(block_out.glob("worker_*")):
    run_dirs = sorted([p for p in worker_root.iterdir() if p.is_dir()])
    if not run_dirs:
        continue
    ep_path = run_dirs[-1] / "episodes.csv"
    if not ep_path.exists():
        continue
    with ep_path.open() as f:
        rows.extend(list(csv.DictReader(f)))

successes = sum(r.get("auto_success", "").lower() == "true" for r in rows)
episodes = len(rows)
mean_steps = (sum(float(r.get("num_steps") or 0.0) for r in rows) / episodes) if episodes else None
payload = {
    "block": block_tag,
    "episodes": episodes,
    "successful_episodes": successes,
    "success_rate": (successes / episodes) if episodes else None,
    "mean_steps": mean_steps,
}
summary_path = block_out / "block_summary.json"
summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY
}

split_csv "${BLOCKS}"
for block_num in "${SPLIT_ITEMS[@]}"; do
  run_block_probe "${block_num}"
done

echo "[collect-bc-probe] done out_root=${OUT_ROOT}"
