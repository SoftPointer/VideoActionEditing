#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

cache_token="${BERNINI_HELDOUT_RANK_CACHE_TOKEN:?set rank cache token}"
local_rank="${LOCAL_RANK:?torchrun LOCAL_RANK is required}"
global_rank="${RANK:?torchrun RANK is required}"
world_size="${WORLD_SIZE:?torchrun WORLD_SIZE is required}"
job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
step_id="${SLURM_STEP_ID:?SLURM_STEP_ID is required}"
[[ "${world_size}" == 8 && "${local_rank}" =~ ^[0-7]$ && "${global_rank}" == "${local_rank}" ]] || {
  echo "invalid WORLD8 rank identity" >&2
  exit 2
}
[[ "${job_id}" =~ ^[1-9][0-9]*$ && "${step_id}" =~ ^[0-9]+$ ]] || {
  echo "invalid Slurm step identity" >&2
  exit 2
}
[[ "${cache_token}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ && "${cache_token}" != *..* ]] || {
  echo "invalid rank cache token" >&2
  exit 2
}
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && "${scratch_parent}" != / && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || {
  echo "invalid node-local scratch parent" >&2
  exit 2
}
scratch_real="$(readlink -f -- "${scratch_parent}")"
[[ "${scratch_real}" == "${scratch_parent%/}" ]] || { echo "scratch parent canonical path differs" >&2; exit 2; }
scratch_type="$(stat -f -c '%T' -- "${scratch_real}")"
case "${scratch_type}" in nfs|nfs4|cifs|smb2|fuse*|lustre|gpfs) echo "shared filesystem forbidden for rank cache" >&2; exit 2 ;; esac
rank_prefix="${scratch_real}/preservation-${job_id}-${step_id}-${cache_token}-r${global_rank}."
rank_root="$(mktemp -d -- "${rank_prefix}XXXXXXXX")"
rank_real="$(readlink -f -- "${rank_root}")"
[[ "${rank_real}" == "${rank_root}" && "${rank_root}" == "${rank_prefix}"* ]] || { echo "unexpected rank cache root" >&2; exit 2; }
chmod 0700 "${rank_root}"
rank_identity="$(stat -c '%d:%i:%u:%a' -- "${rank_root}")"
for leaf in home tmp xdg torch-extensions triton torchinductor pycache miopen-user miopen-custom; do mkdir -m 0700 "${rank_root}/${leaf}"; done
export HOME="${rank_root}/home"
export TMPDIR="${rank_root}/tmp"
export XDG_CACHE_HOME="${rank_root}/xdg"
export TORCH_EXTENSIONS_DIR="${rank_root}/torch-extensions"
export TRITON_CACHE_DIR="${rank_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${rank_root}/torchinductor"
export PYTHONPYCACHEPREFIX="${rank_root}/pycache"
export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C LANG=C
python_bin="${BERNINI_HELDOUT_PYTHON_BIN:?set frozen Python executable}"
[[ -x "${python_bin}" ]] || { echo "invalid Python executable" >&2; exit 2; }

child_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  [[ -z "${child_pid}" ]] || kill "${child_pid}" 2>/dev/null || true
  if [[ -d "${rank_root}" && ! -L "${rank_root}" && "$(readlink -f -- "${rank_root}")" == "${rank_real}" && "$(stat -c '%d:%i:%u:%a' -- "${rank_root}")" == "${rank_identity}" ]]; then
    find "${rank_root}" -xdev -depth -mindepth 1 -delete
    rmdir "${rank_root}"
  else
    exit 70
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM HUP
"${python_bin}" -B "$@" &
child_pid=$!
set +e
wait "${child_pid}"
status=$?
set -e
child_pid=""
exit "${status}"
