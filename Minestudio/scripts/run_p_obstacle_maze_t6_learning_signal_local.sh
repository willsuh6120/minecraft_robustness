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
RUN_TAG="${RUN_TAG:-p_obstacle_maze_t6_learning_$(date +%Y%m%d_%H%M%S)}"
MASTER_ROOT="${MASTER_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_maze_t6_learning_signal/${RUN_TAG}}"

ASSET_DIR="${ASSET_DIR:-}"
if [[ -z "${ASSET_DIR}" ]]; then
  ASSET_DIR="$(
    find "${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_hard_v6_maze_t6_calibration" \
      -path "*/assets/*/eval_bank" -type d 2>/dev/null \
      | sed 's#/eval_bank$##' \
      | sort \
      | tail -n 1 || true
  )"
fi
if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid. Set ASSET_DIR to the hard_v6_maze_t6 asset dir." >&2
  exit 1
fi
if [[ ! -d "${ASSET_DIR}/eval_bank" || ! -d "${ASSET_DIR}/collect_block_plans" ]]; then
  echo "ASSET_DIR does not look like a maze_t6 asset bank: ${ASSET_DIR}" >&2
  exit 1
fi

BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"
if [[ -z "${BASELINE_SUMMARY}" ]]; then
  CALIB_ROOT="$(dirname "$(dirname "${ASSET_DIR}")")"
  BASELINE_SUMMARY="${CALIB_ROOT}/baseline_rocket2_16ep/summary.json"
fi

BASE_SEEDS="${BASE_SEEDS:-1}"
COLLECT_EPISODES="${COLLECT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
PROBE_WORKERS="${PROBE_WORKERS:-4}"
FINAL_PROBE_EPISODES="${FINAL_PROBE_EPISODES:-16}"
RUN_FINAL_PROBES="${RUN_FINAL_PROBES:-1}"
ITERS_PER_BLOCK="${ITERS_PER_BLOCK:-1}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MIN_SUCCESSFUL_FRAGMENTS="${MIN_SUCCESSFUL_FRAGMENTS:-4}"
UPDATE_FRAGMENT_BATCH_SIZE="${UPDATE_FRAGMENT_BATCH_SIZE:-4}"
PPO_LEARNING_RATE="${PPO_LEARNING_RATE:-2e-5}"
KL_COEF="${KL_COEF:-0.1}"
LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE:-suffix_success}"
LOSS_FOCUS_TOP_K="${LOSS_FOCUS_TOP_K:-32}"

# Keep GOAL_SPEC empty by default. The maze bank was baked with same-world auto goals.
GOAL_PROTOCOL="${GOAL_PROTOCOL:-auto_goal_same_world_baked}"
GOAL_SPEC="${GOAL_SPEC:-}"

# Minimal is the default because this run is meant to find whether any P-maze PPO
# learning signal exists before spending another full night on order ablations.
PLAN="${PLAN:-minimal}"

mkdir -p "${MASTER_ROOT}"

cat > "${MASTER_ROOT}/README.txt" <<EOF
P obstacle maze_t6 learning signal suite

asset_dir=${ASSET_DIR}
baseline_summary=${BASELINE_SUMMARY}
goal_protocol=${GOAL_PROTOCOL}
goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}
plan=${PLAN}
base_seeds=${BASE_SEEDS}
collect_episodes=${COLLECT_EPISODES}
eval_episodes=${EVAL_EPISODES}
iters_per_block=${ITERS_PER_BLOCK}
final_probe_episodes=${FINAL_PROBE_EPISODES}

Block ids:
01 left_lane_z1_t6            baseline 7/16  43.8%
02 right_lane_z1_t6           baseline 7/16  43.8%
03 left_lane_z2_t6            baseline 9/16  56.2%
04 right_lane_z2_t6           baseline 7/16  43.8%
05 left_chicane_t6            baseline 6/16  37.5%
06 right_chicane_t6           baseline 3/16  18.8%
07 left_s_curve_t6            baseline 5/16  31.2%
08 right_s_curve_t6           baseline 2/16  12.5%
09 left_funnel_t6             baseline 4/16  25.0%
10 right_funnel_t6            baseline 7/16  43.8%
11 left_gate_entrance_t6      baseline 0/16   0.0%
12 right_gate_entrance_t6     baseline 1/16   6.2%
13 left_outer_detour_t6       baseline 6/16  37.5%
14 right_outer_detour_t6      baseline 6/16  37.5%
15 left_narrow_door_t6        baseline 0/16   0.0%
16 right_narrow_door_t6       baseline 2/16  12.5%

warm_medium order:
03,01,02,04,10,13,14,05

hardish_bridge order:
09,07,06,08,16,12,11,15

single_left_funnel order:
09 repeated 8 times
EOF

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

variant_order() {
  case "$1" in
    warm_medium) printf '%s\n' "3,1,2,4,10,13,14,5" ;;
    hardish_bridge) printf '%s\n' "9,7,6,8,16,12,11,15" ;;
    single_left_funnel) printf '%s\n' "9,9,9,9,9,9,9,9" ;;
    single_left_s_curve) printf '%s\n' "7,7,7,7,7,7,7,7" ;;
    no_update_warm_medium) printf '%s\n' "3,1,2,4,10,13,14,5" ;;
    all_16_easy_to_hard) printf '%s\n' "3,1,2,4,10,13,14,5,9,7,6,8,16,12,11,15" ;;
    reverse_16_hard_to_easy) printf '%s\n' "15,11,12,16,8,6,7,9,5,14,13,10,4,2,1,3" ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_disable_updates() {
  case "$1" in
    no_update_*) printf '%s\n' "1" ;;
    *) printf '%s\n' "0" ;;
  esac
}

default_variants_for_plan() {
  case "${PLAN}" in
    smoke) printf '%s\n' "single_left_funnel" ;;
    minimal) printf '%s\n' "warm_medium,single_left_funnel" ;;
    balanced) printf '%s\n' "warm_medium,hardish_bridge,single_left_funnel,no_update_warm_medium" ;;
    wide) printf '%s\n' "warm_medium,hardish_bridge,single_left_funnel,single_left_s_curve,no_update_warm_medium,all_16_easy_to_hard,reverse_16_hard_to_easy" ;;
    *) echo "Unknown PLAN=${PLAN}; expected smoke, minimal, balanced, or wide." >&2; exit 1 ;;
  esac
}

VARIANTS="${VARIANTS:-$(default_variants_for_plan)}"

echo "[maze-learning] master_root=${MASTER_ROOT}"
echo "[maze-learning] asset_dir=${ASSET_DIR}"
echo "[maze-learning] baseline_summary=${BASELINE_SUMMARY}"
echo "[maze-learning] goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}"
echo "[maze-learning] plan=${PLAN}"
echo "[maze-learning] variants=${VARIANTS}"

run_chain() {
  local variant="$1"
  local seed="$2"
  local order
  local disable_updates
  order="$(variant_order "${variant}")"
  disable_updates="$(variant_disable_updates "${variant}")"

  local seed_tag
  seed_tag="$(printf 'seed_%03d' "${seed}")"
  local out_root="${MASTER_ROOT}/training/${variant}/${seed_tag}"

  echo
  echo "==================== variant=${variant} ${seed_tag} ===================="
  echo "[maze-learning] order=${order}"
  echo "[maze-learning] disable_updates=${disable_updates}"
  echo "[maze-learning] out_root=${out_root}"

  TASK="${TASK}" \
  TASK_GROUP_NAME="mine_p_maze_t6_${variant}_${seed_tag}" \
  ASSET_DIR="${ASSET_DIR}" \
  OUT_ROOT="${out_root}" \
  EXPERIMENT_TAG="p_maze_t6_${variant}_${seed_tag}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_SPEC="${GOAL_SPEC}" \
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
  LOSS_FOCUS_MODE="${LOSS_FOCUS_MODE}" \
  LOSS_FOCUS_TOP_K="${LOSS_FOCUS_TOP_K}" \
  COLLECT_VIDEO_MODE=skip \
  BASELINE_VIDEO_MODE=skip \
  EVAL_VIDEO_MODE=skip \
  FINAL_EVAL_VIDEO_MODE=skip \
  SKIP_VIDEO=1 \
  bash "${ROOT_DIR}/scripts/run_mine_o2_straight_generalization_blocks_vm.sh"
}

run_final_probe() {
  local variant="$1"
  local seed="$2"
  local seed_tag
  seed_tag="$(printf 'seed_%03d' "${seed}")"
  local chain_root="${MASTER_ROOT}/training/${variant}/${seed_tag}"
  local chain_summary="${chain_root}/chain_summary.json"
  if [[ ! -f "${chain_summary}" ]]; then
    echo "[maze-learning] missing chain summary for final probe: ${chain_summary}" >&2
    return 1
  fi

  local model_path
  model_path="$("${PYTHON_BIN}" - <<'PY' "${chain_summary}"
import json
import sys
from pathlib import Path

chain = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
blocks = chain.get("blocks") or []
if not blocks:
    raise SystemExit("empty chain")
print(blocks[-1].get("next_model_path") or "")
PY
)"
  if [[ -z "${model_path}" || ! -f "${model_path}" ]]; then
    echo "[maze-learning] invalid final model path for ${variant}/${seed_tag}: ${model_path}" >&2
    return 1
  fi

  local probe_root="${MASTER_ROOT}/final_probe/${variant}/${seed_tag}"
  echo
  echo "==================== final_probe variant=${variant} ${seed_tag} ===================="
  echo "[maze-learning] model_path=${model_path}"
  echo "[maze-learning] probe_root=${probe_root}"

  ASSET_DIR="${ASSET_DIR}" \
  TASK="${TASK}" \
  BANK_NAME=eval_bank \
  EPISODES="${FINAL_PROBE_EPISODES}" \
  WORLD_WORKERS="${PROBE_WORKERS}" \
  MODEL_PATH="${model_path}" \
  CFG_COEF="${CFG_COEF:-0.0}" \
  CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}" \
  CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}" \
  GOAL_SPEC="${GOAL_SPEC}" \
  BASE_SEED="${BASE_SEED:-1}" \
  SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-1}" \
  EPISODE_RETRIES="${EPISODE_RETRIES:-2}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}" \
  SKIP_VIDEO=1 \
  OUT_ROOT="${probe_root}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"

  "${PYTHON_BIN}" - <<'PY' "${probe_root}/summary.json" "${BASELINE_SUMMARY}" "${probe_root}/trained_vs_baseline.md" "${variant}" "${seed_tag}"
import json
import sys
from pathlib import Path

trained_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
variant = sys.argv[4]
seed_tag = sys.argv[5]

trained = json.loads(trained_path.read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {"variants": []}

base_by_idx = {int(row["instance_idx"]): row for row in baseline.get("variants") or []}
trained_rows = sorted(trained.get("variants") or [], key=lambda row: int(row.get("instance_idx", 0)))

def pct(success, eps):
    return "" if not eps else f"{100.0 * float(success) / float(eps):.1f}%"

base_total_eps = sum(int(row.get("episodes") or 0) for row in base_by_idx.values())
base_total_success = sum(int(row.get("successful_episodes") or 0) for row in base_by_idx.values())
trained_total_eps = sum(int(row.get("episodes") or 0) for row in trained_rows)
trained_total_success = sum(int(row.get("successful_episodes") or 0) for row in trained_rows)

lines = [
    f"# P Maze-T6 Trained vs Baseline Probe: {variant}/{seed_tag}",
    "",
    f"trained_summary: `{trained_path}`",
    f"baseline_summary: `{baseline_path}`" if baseline_path.exists() else "baseline_summary: missing",
    "",
    "| model | success | rate |",
    "|---|---:|---:|",
    f"| baseline | {base_total_success}/{base_total_eps} | {pct(base_total_success, base_total_eps)} |",
    f"| trained | {trained_total_success}/{trained_total_eps} | {pct(trained_total_success, trained_total_eps)} |",
    "",
    "| idx | variant | baseline | trained | delta pp |",
    "|---:|---|---:|---:|---:|",
]

for row in trained_rows:
    idx = int(row.get("instance_idx", 0))
    base = base_by_idx.get(idx, {})
    label = str(row.get("variant_id") or base.get("variant_id") or f"instance_{idx:03d}")
    b_eps = int(base.get("episodes") or 0)
    b_succ = int(base.get("successful_episodes") or 0)
    t_eps = int(row.get("episodes") or 0)
    t_succ = int(row.get("successful_episodes") or 0)
    b_rate = b_succ / b_eps if b_eps else None
    t_rate = t_succ / t_eps if t_eps else None
    delta = "" if b_rate is None or t_rate is None else f"{100.0 * (t_rate - b_rate):+.1f}"
    lines.append(
        f"| {idx} | {label} | {b_succ}/{b_eps} ({pct(b_succ, b_eps)}) | "
        f"{t_succ}/{t_eps} ({pct(t_succ, t_eps)}) | {delta} |"
    )

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_path)
PY
}

split_csv "${VARIANTS}"
VARIANT_ITEMS=( "${SPLIT_ITEMS[@]}" )
split_csv "${BASE_SEEDS}"
SEED_ITEMS=( "${SPLIT_ITEMS[@]}" )

for variant in "${VARIANT_ITEMS[@]}"; do
  for seed in "${SEED_ITEMS[@]}"; do
    run_chain "${variant}" "${seed}"
    if [[ "${RUN_FINAL_PROBES}" == "1" ]]; then
      run_final_probe "${variant}" "${seed}"
    fi
  done
done

"${PYTHON_BIN}" - <<'PY' "${MASTER_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for chain_path in sorted((root / "training").glob("*/seed_*")):
    summary_path = chain_path / "chain_summary.json"
    if not summary_path.exists():
        continue
    chain = json.loads(summary_path.read_text(encoding="utf-8"))
    blocks = chain.get("blocks") or []
    best_rate = None
    best_block = ""
    last_rate = None
    for block in blocks:
        sel = block.get("best_eval_selection") or {}
        rate = sel.get("success_rate")
        if rate is None:
            for value in (sel.get("summary") or {}).values():
                if isinstance(value, dict) and value.get("auto_success_rate") is not None:
                    rate = value.get("auto_success_rate")
                    break
        if rate is not None:
            last_rate = float(rate)
            if best_rate is None or float(rate) > best_rate:
                best_rate = float(rate)
                best_block = str(block.get("block_tag") or "")
    rows.append(
        {
            "chain": str(chain_path.relative_to(root)),
            "blocks": len(blocks),
            "best_eval_success_rate": best_rate,
            "best_block": best_block,
            "last_eval_success_rate": last_rate,
            "final_model_path": blocks[-1].get("next_model_path", "") if blocks else "",
        }
    )

summary_json = root / "maze_learning_summary.json"
summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "# P Maze-T6 Learning Summary",
    "",
    "| chain | blocks | best internal eval | best block | last internal eval |",
    "|---|---:|---:|---|---:|",
]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{100.0 * float(value):.1f}%"
    lines.append(
        f"| {row['chain']} | {row['blocks']} | {fmt(row['best_eval_success_rate'])} | "
        f"{row['best_block']} | {fmt(row['last_eval_success_rate'])} |"
    )
for probe_md in sorted((root / "final_probe").glob("*/*/trained_vs_baseline.md")):
    lines.extend(["", f"## {probe_md.relative_to(root)}", ""])
    lines.extend(probe_md.read_text(encoding="utf-8").splitlines()[:32])

summary_md = root / "maze_learning_summary.md"
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY

echo "[maze-learning] done master_root=${MASTER_ROOT}"
echo "[maze-learning] summary=${MASTER_ROOT}/maze_learning_summary.md"
