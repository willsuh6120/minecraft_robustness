#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

TASK="${TASK:-mine_coal}"
EVAL_INSTANCES="${EVAL_INSTANCES:-8}"
EPISODES="${EPISODES:-2}"
WORKER_SET="${WORKER_SET:-1,4}"
BASE_SEED="${BASE_SEED:-1}"
SEED_STEP="${SEED_STEP:-1}"
SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED:-${BASE_SEED}}"
SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP:-1}"
EPISODE_RETRIES="${EPISODE_RETRIES:-2}"
STEP_BUDGET="${STEP_BUDGET:-150}"
MODEL_PATH="${MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
CFG_COEF="${CFG_COEF:-0.0}"
CFG_POLICY_MODE="${CFG_POLICY_MODE:-frozen_base}"
CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH:-hf:phython96/ROCKET-2-1x-22w}"
STOP_ON_SUCCESS="${STOP_ON_SUCCESS:-1}"
SKIP_VIDEO="${SKIP_VIDEO:-1}"

ASSET_DIR="${ASSET_DIR:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/evaluate_rocket/p_obstacle_worker_timing/$(date +%Y%m%d_%H%M%S)}"
ASSET_OUT_DIR="${ASSET_OUT_DIR:-${OUT_ROOT}/assets}"
GOAL_PROTOCOL="${GOAL_PROTOCOL:-fixed_clean_front_close_v1}"
DEFAULT_GOAL_SPEC="${ROOT_DIR}/outputs/evaluate_rocket/fixed_goal_banks/${TASK}/${GOAL_PROTOCOL}/goal_spec.json"
GOAL_SPEC="${GOAL_SPEC:-}"
GOAL_OVERWRITE="${GOAL_OVERWRITE:-0}"
if [[ -z "${GOAL_SPEC}" ]]; then
  GOAL_SPEC="${DEFAULT_GOAL_SPEC}"
fi
if [[ "${GOAL_OVERWRITE}" == "1" || ! -f "${GOAL_SPEC}" ]]; then
  echo "[p-obstacle-worker-timing] ensuring fixed goal bank goal_spec=${GOAL_SPEC} overwrite=${GOAL_OVERWRITE}"
  TASK="${TASK}" \
  GOAL_PROTOCOL="${GOAL_PROTOCOL}" \
  GOAL_BANK_ROOT="$(dirname "${GOAL_SPEC}")" \
  OVERWRITE="${GOAL_OVERWRITE}" \
  bash "${ROOT_DIR}/scripts/ensure_fixed_clean_goal_bank.sh"
fi
BAKE_GOALS="${BAKE_GOALS:-}"
if [[ -z "${BAKE_GOALS}" ]]; then
  if [[ -n "${GOAL_SPEC}" ]]; then
    BAKE_GOALS=0
  else
    BAKE_GOALS=1
  fi
fi
GOAL_BAKE_WORLD_MODE="${GOAL_BAKE_WORLD_MODE:-clean_path}"

mkdir -p "${OUT_ROOT}"

if [[ -z "${ASSET_DIR}" ]]; then
  echo "[p-obstacle-worker-timing] generating eval bank assets"
  TASK="${TASK}" \
  EVAL_INSTANCES="${EVAL_INSTANCES}" \
  FINAL_INSTANCES=0 \
  SKIP_FINAL_BANK=1 \
  BAKE_GOALS="${BAKE_GOALS}" \
  GOAL_BAKE_WORLD_MODE="${GOAL_BAKE_WORLD_MODE}" \
  BASE_SEED="${BASE_SEED}" \
  OUT_DIR="${ASSET_OUT_DIR}" \
  bash "${ROOT_DIR}/scripts/prepare_mine_p_straight_obstacle_assets.sh"

  ASSET_DIR="$(find "${ASSET_OUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [[ -z "${ASSET_DIR}" || ! -d "${ASSET_DIR}" ]]; then
  echo "ASSET_DIR is missing or invalid: ${ASSET_DIR}" >&2
  exit 1
fi

echo "[p-obstacle-worker-timing] asset_dir=${ASSET_DIR}"
echo "[p-obstacle-worker-timing] worker_set=${WORKER_SET}"
echo "[p-obstacle-worker-timing] episodes_per_world=${EPISODES}"
echo "[p-obstacle-worker-timing] skip_video=${SKIP_VIDEO}"
echo "[p-obstacle-worker-timing] bake_goals=${BAKE_GOALS}"
echo "[p-obstacle-worker-timing] goal_spec=${GOAL_SPEC:-auto_goal_or_baked_task_group}"
echo "[p-obstacle-worker-timing] out_root=${OUT_ROOT}"

split_csv() {
  local input="$1"
  local old_ifs="${IFS}"
  IFS=,
  read -r -a SPLIT_ITEMS <<< "${input}"
  IFS="${old_ifs}"
}

split_csv "${WORKER_SET}"
for workers in "${SPLIT_ITEMS[@]}"; do
  workers="$(echo "${workers}" | tr -d '[:space:]')"
  [[ -n "${workers}" ]] || continue
  if [[ ! "${workers}" =~ ^[0-9]+$ || "${workers}" -lt 1 ]]; then
    echo "Invalid worker count in WORKER_SET: ${workers}" >&2
    exit 1
  fi

  run_out="${OUT_ROOT}/workers_${workers}"
  mkdir -p "${run_out}"
  echo
  echo "==================== workers=${workers} ===================="
  started_ns="$(date +%s%N)"
  ASSET_DIR="${ASSET_DIR}" \
  BANK_NAME=eval_bank \
  EPISODES="${EPISODES}" \
  WORLD_WORKERS="${workers}" \
  BASE_SEED="${BASE_SEED}" \
  SEED_STEP="${SEED_STEP}" \
  SAMPLING_BASE_SEED="${SAMPLING_BASE_SEED}" \
  SAMPLING_SEED_STEP="${SAMPLING_SEED_STEP}" \
  EPISODE_RETRIES="${EPISODE_RETRIES}" \
  STEP_BUDGET="${STEP_BUDGET}" \
  MODEL_PATH="${MODEL_PATH}" \
  CFG_COEF="${CFG_COEF}" \
  CFG_POLICY_MODE="${CFG_POLICY_MODE}" \
  CFG_BASE_REF_MODEL_PATH="${CFG_BASE_REF_MODEL_PATH}" \
  GOAL_SPEC="${GOAL_SPEC}" \
  STOP_ON_SUCCESS="${STOP_ON_SUCCESS}" \
  SKIP_VIDEO="${SKIP_VIDEO}" \
  OUT_ROOT="${run_out}" \
  bash "${ROOT_DIR}/scripts/run_p_obstacle_baseline_variant_probe_local.sh"
  ended_ns="$(date +%s%N)"

  find "${run_out}" -type f -name "*_annotated.mp4" | sort > "${run_out}/annotated_video_manifest.txt"
  find "${run_out}" -type f -name "*.mp4" | sort > "${run_out}/video_manifest.txt"

  python - <<'PY' "${started_ns}" "${ended_ns}" "${run_out}/wall_time.json"
import json
import sys
from pathlib import Path

started = int(sys.argv[1])
ended = int(sys.argv[2])
elapsed = max(0.0, (ended - started) / 1_000_000_000.0)
Path(sys.argv[3]).write_text(json.dumps({"wall_time_sec": round(elapsed, 4)}, indent=2), encoding="utf-8")
print(f"[p-obstacle-worker-timing] wall_time_sec={elapsed:.4f}")
PY
done

python - <<'PY' "${OUT_ROOT}" "${ASSET_DIR}" "${WORKER_SET}" "${EPISODES}" "${SKIP_VIDEO}" "${GOAL_SPEC}" "${BAKE_GOALS}"
import csv
import json
import statistics
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
asset_dir = Path(sys.argv[2])
worker_set = [int(item.strip()) for item in str(sys.argv[3]).split(",") if item.strip()]
episodes_per_world = int(sys.argv[4])
skip_video = str(sys.argv[5]) == "1"
goal_spec = str(sys.argv[6])
bake_goals = str(sys.argv[7])

def as_float(row, key):
    try:
        return float(row.get(key) or 0.0)
    except Exception:
        return 0.0

def mean(rows, key):
    vals = [as_float(row, key) for row in rows if row.get(key) not in (None, "")]
    return round(statistics.mean(vals), 4) if vals else None

def percentile(rows, key, q):
    vals = sorted(as_float(row, key) for row in rows if row.get(key) not in (None, ""))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return round(vals[idx], 4)

def load_rows(run_root):
    rows = []
    for episodes_csv in sorted(run_root.glob("instance_*/*/episodes.csv")):
        instance_dir = episodes_csv.parents[1]
        with episodes_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["instance_dir"] = str(instance_dir)
                rows.append(row)
    return rows

summaries = []
for workers in worker_set:
    run_root = out_root / f"workers_{workers}"
    rows = load_rows(run_root)
    wall_path = run_root / "wall_time.json"
    wall = {}
    if wall_path.exists():
        wall = json.loads(wall_path.read_text(encoding="utf-8"))
    initial_rows = [
        row for row in rows
        if str(row.get("reset_kind", "")) == "initial_env" or str(row.get("episode_index", "")) == "0"
    ]
    fast_rows = [
        row for row in rows
        if str(row.get("reset_kind", "")) == "fast_reuse" or str(row.get("episode_index", "")) != "0"
    ]
    successes = sum(str(row.get("auto_success", "")).lower() == "true" for row in rows)
    instances = sorted({row.get("instance_dir", "") for row in rows if row.get("instance_dir")})
    summary = {
        "workers": workers,
        "run_root": str(run_root),
        "wall_time_sec": wall.get("wall_time_sec"),
        "worlds": len(instances),
        "episodes": len(rows),
        "episodes_per_world": episodes_per_world,
        "success_rate": round(successes / len(rows), 4) if rows else None,
        "mean_episode_wall_time_sec": mean(rows, "episode_wall_time_sec"),
        "mean_core_episode_wall_time_sec": mean(rows, "core_episode_wall_time_sec"),
        "mean_video_write_wall_time_sec": mean(rows, "video_write_wall_time_sec"),
        "initial_reset": {
            "episodes": len(initial_rows),
            "mean_reset_wall_time_sec": mean(initial_rows, "reset_wall_time_sec"),
            "p50_reset_wall_time_sec": percentile(initial_rows, "reset_wall_time_sec", 0.50),
            "max_reset_wall_time_sec": percentile(initial_rows, "reset_wall_time_sec", 1.00),
        },
        "fast_reset": {
            "episodes": len(fast_rows),
            "mean_reset_wall_time_sec": mean(fast_rows, "reset_wall_time_sec"),
            "p50_reset_wall_time_sec": percentile(fast_rows, "reset_wall_time_sec", 0.50),
            "max_reset_wall_time_sec": percentile(fast_rows, "reset_wall_time_sec", 1.00),
        },
    }
    summaries.append(summary)

by_workers = {item["workers"]: item for item in summaries}
comparison = {}
if 1 in by_workers and 4 in by_workers:
    one = by_workers[1]
    four = by_workers[4]
    def ratio(num, den):
        if num is None or den in (None, 0):
            return None
        return round(float(num) / float(den), 4)
    comparison = {
        "total_wall_speedup_workers4_vs_workers1": ratio(one.get("wall_time_sec"), four.get("wall_time_sec")),
        "episode_mean_workers4_over_workers1": ratio(four.get("mean_episode_wall_time_sec"), one.get("mean_episode_wall_time_sec")),
        "initial_reset_mean_workers4_over_workers1": ratio(
            four["initial_reset"].get("mean_reset_wall_time_sec"),
            one["initial_reset"].get("mean_reset_wall_time_sec"),
        ),
        "fast_reset_mean_workers4_over_workers1": ratio(
            four["fast_reset"].get("mean_reset_wall_time_sec"),
            one["fast_reset"].get("mean_reset_wall_time_sec"),
        ),
    }

payload = {
    "asset_dir": str(asset_dir),
    "skip_video": skip_video,
    "goal_spec": goal_spec or None,
    "bake_goals": bake_goals,
    "summaries": summaries,
    "comparison": comparison,
}
(out_root / "worker_timing_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "| workers | total wall s | episodes | mean episode s | initial reset mean s | fast reset mean s | success |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for item in summaries:
    lines.append(
        "| {workers} | {wall} | {episodes} | {episode} | {initial} | {fast} | {success} |".format(
            workers=item["workers"],
            wall=item.get("wall_time_sec"),
            episodes=item["episodes"],
            episode=item.get("mean_episode_wall_time_sec"),
            initial=item["initial_reset"].get("mean_reset_wall_time_sec"),
            fast=item["fast_reset"].get("mean_reset_wall_time_sec"),
            success=item.get("success_rate"),
        )
    )
if comparison:
    lines.append("")
    lines.append(f"workers=4 total wall speedup vs workers=1: {comparison['total_wall_speedup_workers4_vs_workers1']}x")
    lines.append(f"workers=4 initial reset mean / workers=1: {comparison['initial_reset_mean_workers4_over_workers1']}x")
    lines.append(f"workers=4 fast reset mean / workers=1: {comparison['fast_reset_mean_workers4_over_workers1']}x")
(out_root / "worker_timing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"[p-obstacle-worker-timing] summary_json={out_root / 'worker_timing_summary.json'}")
print(f"[p-obstacle-worker-timing] summary_md={out_root / 'worker_timing_summary.md'}")
PY
