#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GCLOUD="${GCLOUD:-$(command -v gcloud)}"

INSTANCE_NAME="${INSTANCE_NAME:-}"
ZONE="${ZONE:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-}"
REMOTE_TMP_DIR="${REMOTE_TMP_DIR:-/tmp}"
SYNC_MODE="${SYNC_MODE:-changed}"

if [[ -z "${GCLOUD}" ]]; then
  echo "gcloud not found in PATH." >&2
  exit 1
fi

if [[ -z "${INSTANCE_NAME}" || -z "${ZONE}" || -z "${REMOTE_USER}" ]]; then
  echo "Required env vars: INSTANCE_NAME, ZONE, REMOTE_USER" >&2
  exit 1
fi

if [[ -z "${REMOTE_REPO_ROOT}" ]]; then
  REMOTE_REPO_ROOT="/home/${REMOTE_USER}/envgen2/Minestudio"
fi

format_seconds() {
  local total
  total="$(printf '%.0f' "${1:-0}")"
  if (( total < 60 )); then
    printf '%ss' "${total}"
    return
  fi
  local minutes=$(( total / 60 ))
  local seconds=$(( total % 60 ))
  if (( minutes < 60 )); then
    printf '%sm%02ds' "${minutes}" "${seconds}"
    return
  fi
  local hours=$(( minutes / 60 ))
  minutes=$(( minutes % 60 ))
  printf '%sh%02dm%02ds' "${hours}" "${minutes}" "${seconds}"
}

log() {
  echo "[sync-gcp-code-delta] $*"
}

cleanup_tmp_dir() {
  local dir="${1:-}"
  [[ -n "${dir}" ]] || return 0
  rm -rf -- "${dir}"
}

is_syncable_path() {
  local path="$1"
  case "${path}" in
    minestudio/*|scripts/*|assets/*)
      ;;
    *)
      return 1
      ;;
  esac
  case "${path}" in
    */__pycache__|*/__pycache__/*|*.pyc|*.pyo|*.so|*.log|*.out|*.err|\
    .pytest_cache|.pytest_cache/*|assets/logs|assets/logs/*|\
    outputs/*|*.pt|*.pt.1|*.ckpt|*.mp4|*.png|*.jpg|*.jpeg|*.gif|*.tar|*.tar.gz|*.zip)
      return 1
      ;;
  esac
  return 0
}

collect_changed_paths() {
  {
    git -C "${ROOT}" diff --name-only --diff-filter=ACMR
    git -C "${ROOT}" diff --cached --name-only --diff-filter=ACMR
    git -C "${ROOT}" ls-files --others --exclude-standard
  } | sort -u | while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    is_syncable_path "${path}" || continue
    [[ -e "${ROOT}/${path}" ]] || continue
    printf '%s\n' "${path}"
  done
}

collect_explicit_paths() {
  local raw
  for raw in "$@"; do
    [[ -z "${raw}" ]] && continue
    if [[ "${raw}" = /* ]]; then
      raw="$(realpath --relative-to="${ROOT}" "${raw}")"
    fi
    [[ -e "${ROOT}/${raw}" ]] || {
      echo "Path not found under repo: ${raw}" >&2
      exit 1
    }
    if [[ -d "${ROOT}/${raw}" ]]; then
      find "${raw}" \( -type f -o -type l \) -print 2>/dev/null
    else
      printf '%s\n' "${raw}"
    fi
  done | sort -u | while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    is_syncable_path "${path}" || continue
    [[ -e "${ROOT}/${path}" ]] || continue
    printf '%s\n' "${path}"
  done
}

collect_all_syncable_paths() {
  find minestudio scripts assets \( -type f -o -type l \) -print 2>/dev/null | sort -u | while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    is_syncable_path "${path}" || continue
    [[ -e "${ROOT}/${path}" ]] || continue
    printf '%s\n' "${path}"
  done
}

main() {
  cd "${ROOT}"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap "cleanup_tmp_dir '${tmp_dir}'" EXIT

  local run_tag
  run_tag="$(date +%Y%m%d_%H%M%S)_$$"

  local manifest_path="${tmp_dir}/paths.txt"
  local archive_path="${tmp_dir}/code_delta_${run_tag}.tar.gz"
  local remote_archive="${REMOTE_TMP_DIR}/$(basename "${archive_path}")"

  if (( "$#" > 0 )); then
    collect_explicit_paths "$@" > "${manifest_path}"
  else
    case "${SYNC_MODE}" in
      changed)
        collect_changed_paths > "${manifest_path}"
        ;;
      all_code)
        collect_all_syncable_paths > "${manifest_path}"
        ;;
      *)
        echo "Unsupported SYNC_MODE: ${SYNC_MODE}" >&2
        exit 1
        ;;
    esac
  fi

  if [[ ! -s "${manifest_path}" ]]; then
    log "No syncable paths found."
    exit 0
  fi

  local path_count
  path_count="$(wc -l < "${manifest_path}" | tr -d ' ')"
  log "repo_root=${ROOT}"
  log "remote_repo_root=${REMOTE_REPO_ROOT}"
  log "instance=${INSTANCE_NAME} zone=${ZONE} remote_user=${REMOTE_USER}"
  log "path_count=${path_count}"
  log "paths:"
  sed 's/^/  - /' "${manifest_path}"

  local started_at
  started_at="$(date +%s)"
  tar -czf "${archive_path}" -C "${ROOT}" -T "${manifest_path}"
  local archive_size
  archive_size="$(du -h "${archive_path}" | awk '{print $1}')"
  log "archive=${archive_path} size=${archive_size}"

  "${GCLOUD}" compute scp \
    "${archive_path}" \
    "${REMOTE_USER}@${INSTANCE_NAME}:${remote_archive}" \
    --zone "${ZONE}"

  "${GCLOUD}" compute ssh \
    "${REMOTE_USER}@${INSTANCE_NAME}" \
    --zone "${ZONE}" \
    --command "mkdir -p '${REMOTE_REPO_ROOT}' && tar -xzf '${remote_archive}' -C '${REMOTE_REPO_ROOT}' && rm -f '${remote_archive}'"

  local ended_at
  ended_at="$(date +%s)"
  local elapsed=$(( ended_at - started_at ))
  log "done elapsed_sec=${elapsed} elapsed_human=$(format_seconds "${elapsed}")"
  log "note=delta sync only adds/overwrites files; it does not delete remote files that were removed locally."
}

main "$@"
