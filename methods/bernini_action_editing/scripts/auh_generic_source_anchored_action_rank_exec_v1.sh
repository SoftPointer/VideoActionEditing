#!/usr/bin/env bash
# Rank-local cache isolation for one WORLD4=DP1xSP4 model.  Every rank receives
# the same trainer argv; action/actor/category never selects a physical rank.

set -Eeuo pipefail
umask 077

fail() { echo "[generic-action-rank-exec] ERROR: $*" >&2; exit 2; }

readonly cache_token="${GSA_RANK_CACHE_TOKEN:?set rank-cache token}"
readonly local_rank="${LOCAL_RANK:?torchrun LOCAL_RANK is required}"
readonly global_rank="${RANK:?torchrun RANK is required}"
readonly world_size="${WORLD_SIZE:?torchrun WORLD_SIZE is required}"
readonly job_id="${SLURM_JOB_ID:?numbered Slurm child is required}"
readonly step_id="${SLURM_STEP_ID:?numbered Slurm step is required}"
readonly python_bin="${GSA_PYTHON_BIN:?set frozen Python executable}"

[[ "${world_size}" == 4 && "${local_rank}" =~ ^[0-3]$ && "${global_rank}" == "${local_rank}" ]] || \
  fail "rank identity is not exact WORLD4 DP1xSP4"
[[ "${job_id}" =~ ^[1-9][0-9]*$ && "${step_id}" =~ ^[0-9]+$ ]] || \
  fail "Slurm child identity differs"
[[ "${cache_token}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ && "${cache_token}" != *..* ]] || \
  fail "rank-cache token differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"

readonly scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && "${scratch_parent}" != / && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "node-local scratch parent differs"
readonly scratch_real="$(readlink -f -- "${scratch_parent}")"
[[ "${scratch_real}" == "${scratch_parent%/}" ]] || fail "scratch parent is not canonical"
readonly scratch_type="$(stat -f -c '%T' -- "${scratch_real}")"
case "${scratch_type}" in
  nfs|nfs4|cifs|smb2|fuse*|lustre|gpfs) fail "shared filesystem forbidden for rank cache" ;;
esac

readonly rank_prefix="${scratch_real}/generic-action-${job_id}-${step_id}-${cache_token}-r${global_rank}."
rank_root="$(mktemp -d -- "${rank_prefix}XXXXXXXX")"
readonly rank_root
readonly rank_real="$(readlink -f -- "${rank_root}")"
[[ "${rank_real}" == "${rank_root}" && "${rank_root}" == "${rank_prefix}"* ]] || \
  fail "rank-cache root differs"
chmod 0700 "${rank_root}"
readonly rank_identity="$(stat -c '%d:%i:%u:%a' -- "${rank_root}")"
for leaf in tmp xdg torch-extensions triton torchinductor pycache miopen-user miopen-custom; do
  mkdir -m 0700 "${rank_root}/${leaf}"
done

export TMPDIR="${rank_root}/tmp"
export XDG_CACHE_HOME="${rank_root}/xdg"
export TORCH_EXTENSIONS_DIR="${rank_root}/torch-extensions"
export TRITON_CACHE_DIR="${rank_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${rank_root}/torchinductor"
export PYTHONPYCACHEPREFIX="${rank_root}/pycache"
export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export LC_ALL=C LANG=C

child_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  if [[ -d "${rank_root}" && ! -L "${rank_root}" && "$(readlink -f -- "${rank_root}")" == "${rank_real}" && "$(stat -c '%d:%i:%u:%a' -- "${rank_root}")" == "${rank_identity}" ]]; then
    find "${rank_root}" -xdev -depth -mindepth 1 -delete
    rmdir "${rank_root}"
  else
    status=70
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

