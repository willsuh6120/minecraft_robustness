#!/usr/bin/env bash
set -euo pipefail

ACTION="report"
INCLUDE_ACTIVE=0
RECLAIM_SWAP=0
WAIT_SEC=5

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup_minestudio_leftovers.sh [--report|--term|--kill] [--include-active] [--reclaim-swap]

Default is --report and only targets orphan Minestudio/Minecraft processes
whose PPID is 1. This avoids killing a currently running rollout/training job.

Options:
  --report          Print suspected leftover processes and top swap users.
  --term            Send SIGTERM to suspected leftover process groups.
  --kill            Send SIGTERM, wait, then SIGKILL remaining suspected groups.
  --include-active  Also include non-orphan Minestudio/Minecraft processes.
  --reclaim-swap    After cleanup, run sudo swapoff -a && sudo swapon -a.
  --wait-sec N      Seconds to wait between SIGTERM and SIGKILL.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      ACTION="report"
      ;;
    --term)
      ACTION="term"
      ;;
    --kill)
      ACTION="kill"
      ;;
    --include-active)
      INCLUDE_ACTIVE=1
      ;;
    --reclaim-swap)
      RECLAIM_SWAP=1
      ;;
    --wait-sec)
      shift
      WAIT_SEC="${1:?missing value for --wait-sec}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

is_candidate_cmd() {
  local comm="$1"
  local args="$2"
  case "${args}" in
    *".minestudio/engine/build/libs/mcprec-"*".jar"*|*"mcprec-"*".jar"*|*"Minecraft/launchClient.sh"*|*"minerl/env/launchClient.sh"*|*"xvfb-run"* )
      return 0
      ;;
  esac
  case "${comm}" in
    Xvfb)
      return 0
      ;;
  esac
  return 1
}

vm_swap_kb() {
  local pid="$1"
  awk '/^VmSwap:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || echo 0
}

list_candidates() {
  ps -eo pid=,ppid=,pgid=,sid=,user=,stat=,pcpu=,pmem=,rss=,etimes=,comm=,args= |
  while read -r pid ppid pgid sid user stat pcpu pmem rss etimes comm args; do
    [[ -n "${pid:-}" ]] || continue
    [[ "${pid}" != "$$" ]] || continue
    if ! is_candidate_cmd "${comm}" "${args:-}"; then
      continue
    fi
    if [[ "${INCLUDE_ACTIVE}" != "1" && "${ppid}" != "1" ]]; then
      continue
    fi
    swap_kb="$(vm_swap_kb "${pid}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${pid}" "${ppid}" "${pgid}" "${sid}" "${user}" "${stat}" "${pcpu}" "${rss}" "${swap_kb}" "${etimes}" "${comm} ${args:-}"
  done
}

print_candidates() {
  local rows="$1"
  if [[ -z "${rows}" ]]; then
    echo "[cleanup-minestudio] no candidate leftover Minestudio/Minecraft processes"
    return
  fi
  echo "[cleanup-minestudio] candidate processes"
  printf '%8s %8s %8s %8s %-8s %-6s %7s %10s %10s %10s %s\n' \
    PID PPID PGID SID USER STAT CPU RSS_KB SWAP_KB AGE_SEC COMMAND
  printf '%s\n' "${rows}" | awk -F'\t' '{
    printf "%8s %8s %8s %8s %-8s %-6s %7s %10s %10s %10s %s\n",
      $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11
  }'
}

print_swap_top() {
  echo "[cleanup-minestudio] memory"
  free -h || true
  swapon --show || true
  echo "[cleanup-minestudio] top swap users"
  for p in /proc/[0-9]*; do
    pid="${p##*/}"
    swap_kb="$(vm_swap_kb "${pid}")"
    if [[ -n "${swap_kb}" && "${swap_kb}" -gt 0 ]]; then
      name="$(awk '/^Name:/ {$1=""; sub(/^ /,""); print}' "${p}/status" 2>/dev/null || true)"
      cmd="$(tr '\0' ' ' < "${p}/cmdline" 2>/dev/null | cut -c1-220)"
      printf '%12d %8s %-18s %s\n' "${swap_kb}" "${pid}" "${name}" "${cmd}"
    fi
  done | sort -nr | head -40
}

kill_groups() {
  local signal="$1"
  local rows="$2"
  [[ -n "${rows}" ]] || return 0
  printf '%s\n' "${rows}" | awk -F'\t' '{print $3}' | sort -n | uniq |
  while read -r pgid; do
    [[ -n "${pgid}" ]] || continue
    [[ "${pgid}" -gt 1 ]] || continue
    if [[ "${pgid}" == "$$" ]]; then
      continue
    fi
    echo "[cleanup-minestudio] kill -${signal} -${pgid}"
    kill "-${signal}" "-${pgid}" 2>/dev/null || true
  done
}

rows="$(list_candidates || true)"
print_candidates "${rows}"
print_swap_top

case "${ACTION}" in
  report)
    exit 0
    ;;
  term)
    kill_groups TERM "${rows}"
    ;;
  kill)
    kill_groups TERM "${rows}"
    sleep "${WAIT_SEC}"
    rows_after="$(list_candidates || true)"
    kill_groups KILL "${rows_after}"
    ;;
esac

if [[ "${RECLAIM_SWAP}" == "1" ]]; then
  echo "[cleanup-minestudio] reclaiming swap via sudo swapoff -a && sudo swapon -a"
  sudo swapoff -a
  sudo swapon -a
fi

echo "[cleanup-minestudio] after cleanup"
print_candidates "$(list_candidates || true)"
print_swap_top
