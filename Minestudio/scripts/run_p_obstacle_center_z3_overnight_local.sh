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

TASK="${TASK:-mine_coal}"
RUN_TAG="${RUN_TAG:-p_obstacle_center_z3_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_center_z3_overnight/${RUN_TAG}}"

SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR:-${ROOT_DIR}/outputs/evaluate_rocket/generated_task_group/20260411_132941}"
ASSET_DIR="${ASSET_DIR:-}"
ASSET_ROOT="${ASSET_ROOT:-${MASTER_ROOT}/assets}"
GENERATE_ASSETS="${GENERATE_ASSETS:-1}"
EVAL_INSTANCES="${EVAL_INSTANCES:-16}"
BANK_WORKERS="${BANK_WORKERS:-1}"
BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES:-3}"

RUN_BASELINE="${RUN_BASELINE:-1}"
BASELINE_EPISODES="${BASELINE_EPISODES:-32}"
BASELINE_WORKERS="${BASELINE_WORKERS:-4}"

RUN_TRAINING="${RUN_TRAINING:-1}"
CHAIN_SET="${CHAIN_SET:-teacher_to_hard,no_update_teacher_to_hard}"
BASE_SEEDS="${BASE_SEEDS:-1}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-1}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
KL_COEF="${KL_COEF:-0.1}"

MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-${HOME}/envgen2/.minestudio}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
unset LD_PRELOAD || true

mkdir -p "${MASTER_ROOT}"

log() {
  echo "[p-center-z3-overnight] $*"
}

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

write_readme() {
  cat > "${MASTER_ROOT}/README.txt" <<EOF
P obstacle center_z3 overnight suite

task=${TASK}
master_root=${MASTER_ROOT}
asset_dir=${ASSET_DIR}
asset_root=${ASSET_ROOT}
generate_assets=${GENERATE_ASSETS}
eval_instances=${EVAL_INSTANCES}
baseline_episodes=${BASELINE_EPISODES}
baseline_workers=${BASELINE_WORKERS}
chain_set=${CHAIN_SET}
base_seeds=${BASE_SEEDS}
collect_episodes=${COLLECT_EPISODES}
eval_episodes=${EVAL_EPISODES}
collect_workers=${COLLECT_WORKERS}
eval_workers=${EVAL_WORKERS}
iters_per_block=${ITERS_PER_BLOCK}
step_budget=${STEP_BUDGET}

Goal protocol:
  The asset bank bakes same-world auto-goal specs.
  Training and rollout use --auto-goal, which resolves the baked goal_spec in each generated YAML.
  No fixed-clean goal_spec is passed.
EOF
}

resolve_latest_asset_dir() {
  find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true
}

generate_assets() {
  if [[ "${GENERATE_ASSETS}" != "1" && -n "${ASSET_DIR}" && -d "${ASSET_DIR}" ]]; then
    log "using existing asset_dir=${ASSET_DIR}"
    return
  fi

  mkdir -p "${ASSET_ROOT}"
  log "generating 16-world P obstacle center_z3 auto-goal bank under ${ASSET_ROOT}"
  TASK="${TASK}" \
  EVAL_INSTANCES="${EVAL_INSTANCES}" \
  FINAL_INSTANCES=0 \
  SKIP_FINAL_BANK=1 \
  BANK_WORKERS="${BANK_WORKERS}" \
  BAKE_GOAL_EPISODE_RETRIES="${BAKE_GOAL_EPISODE_RETRIES}" \
  BASE_SEED=1 \
  SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
  P_GOAL_POSE_PROTOCOL=center_z3 \
  OUT_DIR="${ASSET_ROOT}" \
  bash "${ROOT_DIR}/scripts/prepare_mine_p_auto_goal_eval_bank_local.sh"

  ASSET_DIR="$(resolve_latest_asset_dir)"
  if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
    echo "Could not resolve generated ASSET_DIR under ${ASSET_ROOT}" >&2
    exit 1
  fi
  log "generated asset_dir=${ASSET_DIR}"
}

validate_asset_dir() {
  if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
    echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${ASSET_DIR}/asset_manifest.json" ]]; then
    echo "Missing asset manifest: ${ASSET_DIR}/asset_manifest.json" >&2
    exit 1
  fi
  if [[ ! -f "${ASSET_DIR}/eval_bank/bank_manifest.json" ]]; then
    echo "Missing eval bank manifest: ${ASSET_DIR}/eval_bank/bank_manifest.json" >&2
    exit 1
  fi
  if [[ ! -d "${ASSET_DIR}/collect_block_plans" ]]; then
    echo "Missing collect_block_plans: ${ASSET_DIR}/collect_block_plans" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - <<'PY' "${ASSET_DIR}"
import json
import sys
from pathlib import Path

asset_dir = Path(sys.argv[1])
manifest = json.loads((asset_dir / "eval_bank" / "bank_manifest.json").read_text(encoding="utf-8"))
worlds = manifest.get("instance_worlds") or []
if len(worlds) != 16:
    raise SystemExit(f"Expected 16 eval worlds, found {len(worlds)}")

reserved = {(0, 3), (0, 4), (0, 5)}
required = {(0, 1), (0, 2)}
missing_goals = []
bad_rows = []
for item in worlds:
    idx = int(item.get("instance_idx", -1))
    rows = item.get("plan_rows") or []
    suggestions = (rows[0].get("world_generation_suggestions") or {}) if rows else {}
    positions = {tuple(map(int, p)) for p in suggestions.get("mine_path_obstacle_positions") or []}
    if not (positions & required):
        bad_rows.append((idx, "missing_center_blocker", sorted(positions)))
    if positions & reserved:
        bad_rows.append((idx, "reserved_cell_blocked", sorted(positions & reserved)))
    task_group = Path(str(item.get("generated_task_group_dir") or ""))
    yaml_path = task_group / "mine_coal.yaml"
    if not yaml_path.is_file():
        missing_goals.append((idx, str(yaml_path), "missing_yaml"))
        continue
    text = yaml_path.read_text(encoding="utf-8")
    if "baked_goal_spec_path:" not in text:
        missing_goals.append((idx, str(yaml_path), "missing_baked_goal_spec_path"))

if bad_rows:
    raise SystemExit(f"Invalid P obstacle geometry: {bad_rows}")
if missing_goals:
    raise SystemExit(f"Missing baked goals: {missing_goals[:5]}")
print(f"validated_asset_dir={asset_dir} worlds={len(worlds)}")
PY
}

run_baseline_probe() {
  if [[ "${RUN_BASELINE}" != "1" ]]; then
    log "skipping baseline probe"
    return
  fi

  BASELINE_OUT="${MASTER_ROOT}/baseline_rocket2_32ep"
  log "running ROCKET-2 baseline probe: 16 worlds x ${BASELINE_EPISODES} episodes"
  ASSET_DIR="${ASSET_DIR}" \
  TASK="${TASK}" \
  BANK_NAME=eval_bank \
  EPISODES="${BASELINE_EPISODES}" \
  WORLD_WORKERS="${BASELINE_WORKERS}" \
  MODEL_PATH="${MODEL_PATH}" \
  CFG_COEF="${CFG_COEF}" \
  CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
  CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
  GOAL_SPEC="" \
  BASE_SEED=1 \
  SAMPLING_BASE_SEED=1 \
  EPISODE_RETRIES=2 \
  STEP_BUDGET="${STEP_BUDGET}" \
  STOP_ON_SUCCESS=1 \
  SKIP_VIDEO=1 \
  OUT_ROOT="${BASELINE_OUT}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

  if [[ ! -f "${BASELINE_OUT}/summary.json" ]]; then
    echo "Missing baseline summary: ${BASELINE_OUT}/summary.json" >&2
    exit 1
  fi
}

build_orders_from_baseline() {
  BASELINE_OUT="${MASTER_ROOT}/baseline_rocket2_32ep"
  ORDERS_JSON="${MASTER_ROOT}/orders.json"
  ORDERS_ENV="${MASTER_ROOT}/orders.env"

  if [[ ! -f "${BASELINE_OUT}/summary.json" ]]; then
    echo "Baseline summary is required to build training order: ${BASELINE_OUT}/summary.json" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - <<'PY' "${BASELINE_OUT}/summary.json" "${ORDERS_JSON}" "${ORDERS_ENV}"
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = list(summary.get("variants") or [])
if len(rows) != 16:
    raise SystemExit(f"Expected 16 baseline variant rows, found {len(rows)}")

def rate(row):
    value = row.get("success_rate")
    return -1.0 if value is None else float(value)

def mean_steps(row):
    value = row.get("mean_steps")
    return math.inf if value is None else float(value)

easy = sorted(rows, key=lambda r: (-rate(r), mean_steps(r), int(r["instance_idx"])))
hard = sorted(rows, key=lambda r: (rate(r), -mean_steps(r), int(r["instance_idx"])))
forward = sorted(rows, key=lambda r: int(r["instance_idx"]))
reverse = list(reversed(forward))

def order(items):
    return ",".join(str(int(item["instance_idx"]) + 1) for item in items)

payload = {
    "easy_to_hard": order(easy),
    "hard_to_easy": order(hard),
    "forward": order(forward),
    "reverse": order(reverse),
    "baseline_sorted_easy": [
        {
            "block_id": int(item["instance_idx"]) + 1,
            "variant_id": item.get("variant_id"),
            "success_rate": item.get("success_rate"),
            "mean_steps": item.get("mean_steps"),
        }
        for item in easy
    ],
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
Path(sys.argv[3]).write_text(
    "\n".join(
        [
            f"EASY_TO_HARD_ORDER='{payload['easy_to_hard']}'",
            f"HARD_TO_EASY_ORDER='{payload['hard_to_easy']}'",
            f"FORWARD_ORDER='{payload['forward']}'",
            f"REVERSE_ORDER='{payload['reverse']}'",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

order_for_chain() {
  local chain_name="$1"
  # shellcheck disable=SC1090
  source "${ORDERS_ENV}"
  case "${chain_name}" in
    teacher_to_hard|easy_to_hard|no_update_teacher_to_hard|no_update_easy_to_hard)
      printf '%s\n' "${EASY_TO_HARD_ORDER}"
      ;;
    hard_to_teacher|hard_to_easy|no_update_hard_to_easy)
      printf '%s\n' "${HARD_TO_EASY_ORDER}"
      ;;
    forward|all_forward|no_update_forward|no_update_all_forward)
      printf '%s\n' "${FORWARD_ORDER}"
      ;;
    reverse|no_update_reverse)
      printf '%s\n' "${REVERSE_ORDER}"
      ;;
    *)
      echo "Unknown chain name: ${chain_name}" >&2
      return 1
      ;;
  esac
}

disable_updates_for_chain() {
  case "$1" in
    no_update_*) printf '%s\n' "1" ;;
    *) printf '%s\n' "0" ;;
  esac
}

run_training_chains() {
  if [[ "${RUN_TRAINING}" != "1" ]]; then
    log "skipping training chains"
    return
  fi

  build_orders_from_baseline
  split_csv "${CHAIN_SET}"
  local chain_items=( "${SPLIT_ITEMS[@]}" )
  split_csv "${BASE_SEEDS}"
  local seed_items=( "${SPLIT_ITEMS[@]}" )

  for chain_name in "${chain_items[@]}"; do
    local order
    local disable_updates
    order="$(order_for_chain "${chain_name}")"
    disable_updates="$(disable_updates_for_chain "${chain_name}")"
    for seed in "${seed_items[@]}"; do
      local seed_tag
      local out_root
      seed_tag="$(printf 'seed_%03d' "${seed}")"
      out_root="${MASTER_ROOT}/training/${chain_name}/${seed_tag}"
      mkdir -p "${out_root}"

      echo
      echo "==================== training chain=${chain_name} ${seed_tag} ===================="
      log "block_order=${order}"
      log "disable_updates=${disable_updates}"
      log "out_root=${out_root}"

      TASK="${TASK}" \
      TASK_GROUP_NAME="mine_p_center_z3_${chain_name}_${seed_tag}" \
      SOURCE_ENV_CONF_DIR="${SOURCE_ENV_CONF_DIR}" \
      ASSET_DIR="${ASSET_DIR}" \
      OUT_ROOT="${out_root}" \
      EXPERIMENT_TAG="p_center_z3_${chain_name}_${seed_tag}" \
      MODEL_PATH="${MODEL_PATH}" \
      CFG_COEF="${CFG_COEF}" \
      CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
      CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
      KL_ANCHOR_MODEL_PATH="${KL_ANCHOR_MODEL_PATH}" \
      GOAL_PROTOCOL=auto_goal \
      GOAL_SPEC="" \
      BLOCK_COUNT=16 \
      BLOCK_ORDER="${order}" \
      DISABLE_UPDATES="${disable_updates}" \
      SKIP_ALL_FINAL_EVAL=1 \
      BASE_SEED="${seed}" \
      SAMPLING_BASE_SEED="${seed}" \
      BAKE_GOAL_BASE_SEED="${seed}" \
      COLLECT_EPISODES="${COLLECT_EPISODES}" \
      EVAL_EPISODES="${EVAL_EPISODES}" \
      COLLECT_WORKERS="${COLLECT_WORKERS}" \
      EVAL_WORKERS="${EVAL_WORKERS}" \
      FINAL_EVAL_WORKERS=1 \
      ITERS_PER_BLOCK="${ITERS_PER_BLOCK}" \
      STEP_BUDGET="${STEP_BUDGET}" \
      MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS}" \
      UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE}" \
      PPO_LEARNING_RATE="${PPO_LEARNING_RATE}" \
      KL_COEF="${KL_COEF}" \
      COLLECT_VIDEO_MODE=skip \
      BASELINE_VIDEO_MODE=skip \
      EVAL_VIDEO_MODE=skip \
      FINAL_EVAL_VIDEO_MODE=skip \
      SKIP_VIDEO=1 \
      bash "${ROOT_DIR}/scripts/run_mine_o2_straight_generalization_blocks_vm.sh"
    done
  done
}

write_summary() {
  "${PYTHON_BIN}" - <<'PY' "${MASTER_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = [
    "# P Obstacle Center-Z3 Overnight Summary",
    "",
    f"root: `{root}`",
    "",
]

baseline_path = root / "baseline_rocket2_32ep" / "summary.json"
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = sorted(baseline.get("variants") or [], key=lambda r: int(r.get("instance_idx", 0)))
    total_eps = sum(int(r.get("episodes") or 0) for r in rows)
    total_success = sum(int(r.get("successful_episodes") or 0) for r in rows)
    rate = total_success / total_eps if total_eps else None
    lines += [
        "## Baseline",
        "",
        f"overall: `{total_success}/{total_eps}` ({100.0 * rate:.1f}%)" if rate is not None else "overall: n/a",
        "",
        "| idx | variant | success | mean steps | positions |",
        "|---:|---|---:|---:|---|",
    ]
    for row in rows:
        eps = int(row.get("episodes") or 0)
        succ = int(row.get("successful_episodes") or 0)
        sr = row.get("success_rate")
        mean_steps = row.get("mean_steps")
        lines.append(
            f"| {int(row.get('instance_idx', 0))} | {row.get('variant_id', '')} | "
            f"{succ}/{eps} ({100.0 * float(sr):.1f}%) | "
            f"{float(mean_steps):.1f} | `{row.get('positions', [])}` |"
        )
    lines.append("")

orders_path = root / "orders.json"
if orders_path.exists():
    orders = json.loads(orders_path.read_text(encoding="utf-8"))
    lines += [
        "## Orders",
        "",
        f"easy_to_hard: `{orders.get('easy_to_hard', '')}`",
        f"hard_to_easy: `{orders.get('hard_to_easy', '')}`",
        f"forward: `{orders.get('forward', '')}`",
        f"reverse: `{orders.get('reverse', '')}`",
        "",
    ]

chain_rows = []
for chain_summary_path in sorted((root / "training").glob("*/*/chain_summary.json")):
    chain = json.loads(chain_summary_path.read_text(encoding="utf-8"))
    blocks = chain.get("blocks") or []
    best_rate = None
    best_block = ""
    last_rate = None
    for block in blocks:
        sel = block.get("best_eval_selection") or {}
        rate = sel.get("auto_success_rate", sel.get("success_rate"))
        if rate is None:
            continue
        rate = float(rate)
        last_rate = rate
        if best_rate is None or rate > best_rate:
            best_rate = rate
            best_block = str(block.get("block_tag") or "")
    chain_rows.append(
        {
            "chain": str(chain_summary_path.parent.parent.relative_to(root / "training")),
            "seed": chain_summary_path.parent.name,
            "blocks": len(blocks),
            "best_rate": best_rate,
            "best_block": best_block,
            "last_rate": last_rate,
            "path": str(chain_summary_path),
        }
    )

if chain_rows:
    lines += [
        "## Training Chains",
        "",
        "| chain | seed | blocks | best eval | best block | last eval |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in chain_rows:
        def fmt(value):
            return "" if value is None else f"{100.0 * float(value):.1f}%"
        lines.append(
            f"| {row['chain']} | {row['seed']} | {row['blocks']} | "
            f"{fmt(row['best_rate'])} | {row['best_block']} | {fmt(row['last_rate'])} |"
        )
    lines.append("")

out = root / "overnight_summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY
}

on_exit() {
  local status=$?
  echo
  log "exit_status=${status}"
  log "master_root=${MASTER_ROOT}"
  if [[ -f "${MASTER_ROOT}/overnight_summary.md" ]]; then
    log "summary=${MASTER_ROOT}/overnight_summary.md"
  fi
}
trap on_exit EXIT

log "master_root=${MASTER_ROOT}"
generate_assets
write_readme
validate_asset_dir
run_baseline_probe
run_training_chains
write_summary

log "done"
