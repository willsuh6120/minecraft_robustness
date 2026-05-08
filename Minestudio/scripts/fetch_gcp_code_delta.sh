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
  echo "[fetch-gcp-code-delta] $*"
}

cleanup_tmp_dir() {
  local dir="${1:-}"
  [[ -n "${dir}" ]] || return 0
  rm -rf -- "${dir}"
}

normalize_explicit_path_arg() {
  local raw="$1"
  [[ -n "${raw}" ]] || return 0
  if [[ "${raw}" = /* ]]; then
    if [[ "${raw}" == "${ROOT}/"* ]]; then
      raw="${raw#${ROOT}/}"
    elif [[ "${raw}" == "${REMOTE_REPO_ROOT}/"* ]]; then
      raw="${raw#${REMOTE_REPO_ROOT}/}"
    else
      echo "Absolute path must be under local repo or remote repo: ${raw}" >&2
      exit 1
    fi
  fi
  raw="${raw#./}"
  [[ -n "${raw}" ]] || return 0
  printf '%s\n' "${raw}"
}

collect_explicit_paths() {
  local raw
  for raw in "$@"; do
    normalize_explicit_path_arg "${raw}"
  done | sort -u
}

write_remote_helper() {
  local out_path="$1"
  cat > "${out_path}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$1"
SYNC_MODE="$2"
MANIFEST_IN="$3"
MANIFEST_OUT="$4"
ARCHIVE_OUT="$5"

log() {
  echo "[fetch-gcp-code-delta:remote] $*"
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
    git -C "${REPO_ROOT}" diff --name-only --diff-filter=ACMR
    git -C "${REPO_ROOT}" diff --cached --name-only --diff-filter=ACMR
    git -C "${REPO_ROOT}" ls-files --others --exclude-standard
  } | sort -u | while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    is_syncable_path "${path}" || continue
    [[ -e "${REPO_ROOT}/${path}" ]] || continue
    printf '%s\n' "${path}"
  done
}

collect_explicit_paths() {
  while IFS= read -r raw; do
    [[ -z "${raw}" ]] && continue
    raw="${raw#./}"
    if [[ "${raw}" = /* ]]; then
      if [[ "${raw}" == "${REPO_ROOT}/"* ]]; then
        raw="${raw#${REPO_ROOT}/}"
      else
        echo "Path not under remote repo: ${raw}" >&2
        exit 1
      fi
    fi
    [[ -e "${REPO_ROOT}/${raw}" ]] || {
      echo "Path not found under remote repo: ${raw}" >&2
      exit 1
    }
    if [[ -d "${REPO_ROOT}/${raw}" ]]; then
      (
        cd "${REPO_ROOT}"
        find "${raw}" \( -type f -o -type l \) -print 2>/dev/null
      )
    else
      printf '%s\n' "${raw}"
    fi
  done < "${MANIFEST_IN}" | sort -u | while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    is_syncable_path "${path}" || continue
    [[ -e "${REPO_ROOT}/${path}" ]] || continue
    printf '%s\n' "${path}"
  done
}

collect_all_syncable_paths() {
  (
    cd "${REPO_ROOT}"
    find minestudio scripts assets \( -type f -o -type l \) -print 2>/dev/null
  ) | sort -u | while IFS= read -r path; do
    [[ -z "${path}" ]] && continue
    is_syncable_path "${path}" || continue
    [[ -e "${REPO_ROOT}/${path}" ]] || continue
    printf '%s\n' "${path}"
  done
}

main() {
  cd "${REPO_ROOT}"

  case "${SYNC_MODE}" in
    changed)
      if [[ "${MANIFEST_IN}" != "-" && -s "${MANIFEST_IN}" ]]; then
        collect_explicit_paths > "${MANIFEST_OUT}"
      else
        collect_changed_paths > "${MANIFEST_OUT}"
      fi
      ;;
    all_code)
      if [[ "${MANIFEST_IN}" != "-" && -s "${MANIFEST_IN}" ]]; then
        collect_explicit_paths > "${MANIFEST_OUT}"
      else
        collect_all_syncable_paths > "${MANIFEST_OUT}"
      fi
      ;;
    *)
      echo "Unsupported SYNC_MODE: ${SYNC_MODE}" >&2
      exit 1
      ;;
  esac

  if [[ ! -s "${MANIFEST_OUT}" ]]; then
    log "No syncable paths found."
    exit 0
  fi

  local path_count
  path_count="$(wc -l < "${MANIFEST_OUT}" | tr -d ' ')"
  log "repo_root=${REPO_ROOT}"
  log "sync_mode=${SYNC_MODE}"
  log "path_count=${path_count}"
  log "paths:"
  sed 's/^/  - /' "${MANIFEST_OUT}"

  tar -czf "${ARCHIVE_OUT}" -C "${REPO_ROOT}" -T "${MANIFEST_OUT}"
  log "archive=${ARCHIVE_OUT}"
}

main
EOF
}

main() {
  cd "${ROOT}"

  local tmp_dir
  tmp_dir="$(mktemp -d)"

  local remote_script_remote=""
  local remote_manifest_remote=""
  local remote_resolved_manifest_remote=""
  local remote_archive=""

  cleanup_remote_tmp() {
    [[ -n "${remote_script_remote}" ]] || return 0
    local cleanup_command
    cleanup_command="$(printf "rm -f %q %q %q %q" \
      "${remote_script_remote}" \
      "${remote_manifest_remote}" \
      "${remote_resolved_manifest_remote}" \
      "${remote_archive}")"
    "${GCLOUD}" compute ssh \
      "${REMOTE_USER}@${INSTANCE_NAME}" \
      --zone "${ZONE}" \
      --command "${cleanup_command}" >/dev/null 2>&1 || true
  }

  trap "cleanup_tmp_dir '${tmp_dir}'; cleanup_remote_tmp" EXIT

  local run_tag
  run_tag="$(date +%Y%m%d_%H%M%S)_$$"

  local request_manifest_local="${tmp_dir}/requested_paths.txt"
  local resolved_manifest_local="${tmp_dir}/resolved_paths.txt"
  local remote_script_local="${tmp_dir}/fetch_remote_code_delta.sh"
  local archive_local="${tmp_dir}/code_delta_${run_tag}.tar.gz"

  remote_script_remote="${REMOTE_TMP_DIR}/fetch_code_delta_${run_tag}.sh"
  remote_manifest_remote="${REMOTE_TMP_DIR}/fetch_code_requested_${run_tag}.txt"
  remote_resolved_manifest_remote="${REMOTE_TMP_DIR}/fetch_code_resolved_${run_tag}.txt"
  remote_archive="${REMOTE_TMP_DIR}/code_delta_${run_tag}.tar.gz"

  if (( "$#" > 0 )); then
    collect_explicit_paths "$@" > "${request_manifest_local}"
  fi

  write_remote_helper "${remote_script_local}"

  log "repo_root=${ROOT}"
  log "remote_repo_root=${REMOTE_REPO_ROOT}"
  log "instance=${INSTANCE_NAME} zone=${ZONE} remote_user=${REMOTE_USER}"
  log "sync_mode=${SYNC_MODE}"

  "${GCLOUD}" compute scp \
    "${remote_script_local}" \
    "${REMOTE_USER}@${INSTANCE_NAME}:${remote_script_remote}" \
    --zone "${ZONE}"

  local remote_manifest_arg="-"
  if [[ -s "${request_manifest_local}" ]]; then
    "${GCLOUD}" compute scp \
      "${request_manifest_local}" \
      "${REMOTE_USER}@${INSTANCE_NAME}:${remote_manifest_remote}" \
      --zone "${ZONE}"
    remote_manifest_arg="${remote_manifest_remote}"
    log "requested_paths:"
    sed 's/^/  - /' "${request_manifest_local}"
  fi

  local remote_command
  remote_command="$(printf "bash %q %q %q %q %q %q" \
    "${remote_script_remote}" \
    "${REMOTE_REPO_ROOT}" \
    "${SYNC_MODE}" \
    "${remote_manifest_arg}" \
    "${remote_resolved_manifest_remote}" \
    "${remote_archive}")"

  local started_at
  started_at="$(date +%s)"

  "${GCLOUD}" compute ssh \
    "${REMOTE_USER}@${INSTANCE_NAME}" \
    --zone "${ZONE}" \
    --command "${remote_command}"

  "${GCLOUD}" compute scp \
    "${REMOTE_USER}@${INSTANCE_NAME}:${remote_resolved_manifest_remote}" \
    "${resolved_manifest_local}" \
    --zone "${ZONE}"

  if [[ ! -s "${resolved_manifest_local}" ]]; then
    log "No syncable paths found."
    exit 0
  fi

  "${GCLOUD}" compute scp \
    "${REMOTE_USER}@${INSTANCE_NAME}:${remote_archive}" \
    "${archive_local}" \
    --zone "${ZONE}"

  local archive_size
  archive_size="$(du -h "${archive_local}" | awk '{print $1}')"
  log "archive=${archive_local} size=${archive_size}"

  tar -xzf "${archive_local}" -C "${ROOT}"

  local ended_at
  ended_at="$(date +%s)"
  local elapsed=$(( ended_at - started_at ))
  log "done elapsed_sec=${elapsed} elapsed_human=$(format_seconds "${elapsed}")"
  log "note=delta fetch only adds/overwrites local files; it does not delete local files that were removed on the VM."
}

main "$@"
