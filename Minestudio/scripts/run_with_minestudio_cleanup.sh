#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CLEANUP_SCRIPT="${CLEANUP_SCRIPT:-${ROOT_DIR}/scripts/cleanup_minestudio_leftovers.sh}"
TERM_WAIT_SEC="${TERM_WAIT_SEC:-8}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_with_minestudio_cleanup.sh -- <command> [args...]

Runs the command in its own process group. On Ctrl-C, TERM, or normal exit,
the wrapper kills the whole command process group and then cleans orphaned
Minestudio/Minecraft leftovers.

Example:
  bash scripts/run_with_minestudio_cleanup.sh -- bash -lc 'RUN_TAG=test bash scripts/run_p_obstacle_hard_v5_open_path_calibration_local.sh'
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

child_pid=""
cleanup_started=0

cleanup() {
  local status="$1"
  if [[ "${cleanup_started}" == "1" ]]; then
    exit "${status}"
  fi
  cleanup_started=1

  if [[ -n "${child_pid}" ]]; then
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[run-with-cleanup] stopping child process group -${child_pid}"
      kill -TERM "-${child_pid}" 2>/dev/null || true
      sleep "${TERM_WAIT_SEC}"
      kill -KILL "-${child_pid}" 2>/dev/null || true
    fi
  fi

  bash "${CLEANUP_SCRIPT}" --kill --wait-sec 2 || true
  exit "${status}"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM
trap 'cleanup $?' EXIT

setsid "$@" &
child_pid="$!"
wait "${child_pid}"
