#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# This wrapper exists only to make the intended execution environment explicit.
# The underlying rebake script is headless and works on both local Linux and VM.
MINESTUDIO_DIR="$(cd "${ROOT_DIR}/.." && pwd)/.minestudio" \
  bash "${ROOT_DIR}/scripts/rebake_goal_from_task_group_local.sh" "$@"
