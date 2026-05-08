#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FETCH_SCRIPT="${FETCH_SCRIPT:-${ROOT_DIR}/scripts/fetch_gcp_run_artifacts.sh}"

PROJECT_ID="${PROJECT_ID:-envgen-491906}"
INSTANCE_NAME="${INSTANCE_NAME:-instance-20260417-163859}"
ZONE="${ZONE:-asia-southeast1-b}"
REMOTE_USER="${REMOTE_USER:-willsuh1114}"
REMOTE_SUITE_ROOT="${REMOTE_SUITE_ROOT:-/home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/o2_order_ablation_suite}"
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT:-${ROOT_DIR}/outputs/imported_vm_runs/o2_metrics_snapshots}"
LOCAL_DEST_DIR="${LOCAL_DEST_DIR:-${LOCAL_IMPORT_ROOT}/latest}"
OVERWRITE="${OVERWRITE:-1}"

usage() {
  cat <<'EOF'
usage:
  bash scripts/fetch_vm_o2_metrics_snapshot.sh [remote_run_dir]

examples:
  # Auto-detect latest O2 order-ablation suite on VM and overwrite local latest snapshot.
  bash scripts/fetch_vm_o2_metrics_snapshot.sh

  # Fetch a specific suite/run directory.
  bash scripts/fetch_vm_o2_metrics_snapshot.sh \
    /home/willsuh1114/envgen2/Minestudio/outputs/evaluate_rocket/o2_order_ablation_suite/fixed_goal_20260430_201441

env:
  LOCAL_DEST_DIR=...   local destination; default outputs/imported_vm_runs/o2_metrics_snapshots/latest
  OVERWRITE=1          overwrite destination each fetch; default 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REMOTE_RUN_DIR="${1:-}"
if [[ -z "${REMOTE_RUN_DIR}" ]]; then
  echo "[fetch-vm-o2-metrics] auto-detecting latest suite under ${REMOTE_SUITE_ROOT}"
  REMOTE_RUN_DIR="$(
    CLOUDSDK_CORE_PROJECT="${PROJECT_ID}" \
    gcloud compute ssh "${REMOTE_USER}@${INSTANCE_NAME}" \
      --zone "${ZONE}" \
      --command "set -euo pipefail; test -d '${REMOTE_SUITE_ROOT}'; find '${REMOTE_SUITE_ROOT}' -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-"
  )"
fi

if [[ -z "${REMOTE_RUN_DIR}" ]]; then
  echo "Could not resolve REMOTE_RUN_DIR." >&2
  exit 1
fi

echo "[fetch-vm-o2-metrics] remote_run_dir=${REMOTE_RUN_DIR}"
echo "[fetch-vm-o2-metrics] local_dest_dir=${LOCAL_DEST_DIR}"

FINAL_LOCAL_DEST_DIR="${LOCAL_DEST_DIR}"
TMP_LOCAL_DEST_DIR="${FINAL_LOCAL_DEST_DIR}.tmp.$$"
if [[ -e "${FINAL_LOCAL_DEST_DIR}" && "${OVERWRITE}" != "1" ]]; then
  echo "Local destination already exists: ${FINAL_LOCAL_DEST_DIR}" >&2
  echo "Set OVERWRITE=1 to replace it, or pass a different LOCAL_DEST_DIR." >&2
  exit 1
fi
rm -rf -- "${TMP_LOCAL_DEST_DIR}"
cleanup_tmp_snapshot() {
  rm -rf -- "${TMP_LOCAL_DEST_DIR}"
}
trap cleanup_tmp_snapshot EXIT

FETCH_MODE=metrics_only \
CLOUDSDK_CORE_PROJECT="${PROJECT_ID}" \
INSTANCE_NAME="${INSTANCE_NAME}" \
ZONE="${ZONE}" \
REMOTE_USER="${REMOTE_USER}" \
LOCAL_IMPORT_ROOT="${LOCAL_IMPORT_ROOT}" \
OVERWRITE=1 \
bash "${FETCH_SCRIPT}" "${REMOTE_RUN_DIR}" "${TMP_LOCAL_DEST_DIR}"

"${PYTHON_BIN:-python}" - <<'PY' "${TMP_LOCAL_DEST_DIR}"
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def rate_from_summary_json(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = payload if isinstance(payload, list) else list(payload.values()) if isinstance(payload, dict) else []
    rates = []
    for row in rows:
        if isinstance(row, dict):
            rate = row.get("auto_success_rate", row.get("success_rate"))
            if rate is not None:
                rates.append(float(rate))
    if not rates:
        return None
    return sum(rates) / len(rates)

lines = [
    "# VM O2 Metrics Snapshot",
    "",
    f"local_root: `{root}`",
    "",
    "## Completed Chains",
    "",
    "| chain | blocks | best eval | best block | last eval |",
    "|---|---:|---:|---|---:|",
]
rows = []
for path in sorted(root.glob("*/seed_*/chain_summary.json")):
    chain = json.loads(path.read_text(encoding="utf-8"))
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
    rows.append((path.parent.parent.relative_to(root), len(blocks), best_rate, best_block, last_rate))

def fmt(value):
    return "" if value is None else f"{100.0 * float(value):.1f}%"

for chain, blocks, best_rate, best_block, last_rate in rows:
    lines.append(f"| {chain} | {blocks} | {fmt(best_rate)} | {best_block} | {fmt(last_rate)} |")

pilot_rows = []
for path in sorted(root.glob("*/seed_*/*/pilot_summary.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    iterations = list(payload.get("iteration_summaries") or [])
    best = payload.get("best_eval_selection") or {}
    best_rate = best.get("auto_success_rate", best.get("success_rate"))
    last_rate = None
    if iterations:
        last_summary = iterations[-1].get("eval_summary") or {}
        rows_for_rate = last_summary if isinstance(last_summary, list) else list(last_summary.values()) if isinstance(last_summary, dict) else []
        rates = []
        for row in rows_for_rate:
            if isinstance(row, dict):
                rate = row.get("auto_success_rate", row.get("success_rate"))
                if rate is not None:
                    rates.append(float(rate))
        if rates:
            last_rate = sum(rates) / len(rates)
    pilot_rows.append((path.parent.relative_to(root), "completed", len(iterations), best_rate, last_rate))

for path in sorted(root.glob("*/seed_*/*/pilot_state.json")):
    summary_path = path.with_name("pilot_summary.json")
    if summary_path.exists():
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    iterations = list(payload.get("iteration_summaries") or [])
    best = payload.get("best_eval_selection") or {}
    best_rate = best.get("auto_success_rate", best.get("success_rate"))
    last_rate = None
    if iterations:
        last_summary = iterations[-1].get("eval_summary") or {}
        rows_for_rate = last_summary if isinstance(last_summary, list) else list(last_summary.values()) if isinstance(last_summary, dict) else []
        rates = []
        for row in rows_for_rate:
            if isinstance(row, dict):
                rate = row.get("auto_success_rate", row.get("success_rate"))
                if rate is not None:
                    rates.append(float(rate))
        if rates:
            last_rate = sum(rates) / len(rates)
    pilot_rows.append((path.parent.relative_to(root), str(payload.get("status") or "running"), len(iterations), best_rate, last_rate))

if pilot_rows:
    lines += ["", "## Pilot Runs", "", "| run | status | iterations | best eval | last eval |", "|---|---|---:|---:|---:|"]
    for run, status, iterations, best_rate, last_rate in pilot_rows:
        lines.append(f"| {run} | {status} | {iterations} | {fmt(best_rate)} | {fmt(last_rate)} |")

lines += ["", "## Partial Block Metrics", "", "| run | phase | summaries | mean success | episodes |", "|---|---|---:|---:|---:|"]
for run_dir in sorted(root.glob("*/seed_*/blocks/block_*/*")):
    if not run_dir.is_dir():
        continue
    for phase in ("baseline", "iterations", "eval", "final_eval"):
        phase_dir = run_dir / phase
        if not phase_dir.exists():
            continue
        rates = [r for r in (rate_from_summary_json(p) for p in phase_dir.rglob("summary.json")) if r is not None]
        episode_rows = 0
        for ep_path in phase_dir.rglob("episodes.csv"):
            try:
                with ep_path.open(newline="", encoding="utf-8") as handle:
                    episode_rows += max(0, sum(1 for _ in csv.reader(handle)) - 1)
            except Exception:
                pass
        if rates or episode_rows:
            rel = run_dir.relative_to(root)
            mean = sum(rates) / len(rates) if rates else None
            lines.append(f"| {rel} | {phase} | {len(rates)} | {fmt(mean)} | {episode_rows} |")

out = root / "VM_METRICS_SUMMARY.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf -- "${FINAL_LOCAL_DEST_DIR}"
fi
mv "${TMP_LOCAL_DEST_DIR}" "${FINAL_LOCAL_DEST_DIR}"
trap - EXIT

echo "[fetch-vm-o2-metrics] summary=${LOCAL_DEST_DIR}/VM_METRICS_SUMMARY.md"
