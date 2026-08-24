#!/usr/bin/env bash
# Isolate one WORLD4/SP4 checkpoint-decode rank without changing HOME.

set -Eeuo pipefail
umask 077

token="${PACKED_PRESERVATION_REVIEW_CACHE_TOKEN:?set review cache token}"
python_bin="${PACKED_PRESERVATION_REVIEW_PYTHON_BIN:?set review Python}"
local_rank="${LOCAL_RANK:?LOCAL_RANK is required}"
global_rank="${RANK:?RANK is required}"
world_size="${WORLD_SIZE:?WORLD_SIZE is required}"
job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
step_id="${SLURM_STEP_ID:?SLURM_STEP_ID is required}"
[[ "${world_size}" == 4 && "${local_rank}" =~ ^[0-3]$ && "${global_rank}" == "${local_rank}" ]] || {
  echo "invalid one-node WORLD4 rank identity" >&2; exit 2;
}
[[ "${token}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || {
  echo "invalid review cache token" >&2; exit 2;
}
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || {
  echo "invalid review Python" >&2; exit 2;
}
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && "${scratch_parent}" != / && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || {
  echo "invalid node-local scratch" >&2; exit 2;
}
scratch_real="$(readlink -f -- "${scratch_parent}")"
rank_root="$(mktemp -d -- "${scratch_real}/presv2-review-${job_id}-${step_id}-r${global_rank}.XXXXXXXX")"
rank_identity="$(stat -c '%d:%i:%u:%a' -- "${rank_root}")"
for leaf in tmp xdg hf torch-extensions triton torchinductor pycache miopen-user miopen-custom; do
  mkdir -m 0700 "${rank_root}/${leaf}"
done
export TMPDIR="${rank_root}/tmp"
export XDG_CACHE_HOME="${rank_root}/xdg"
export HF_HOME="${rank_root}/hf"
export TORCH_EXTENSIONS_DIR="${rank_root}/torch-extensions"
export TRITON_CACHE_DIR="${rank_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${rank_root}/torchinductor"
export PYTHONPYCACHEPREFIX="${rank_root}/pycache"
export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 LC_ALL=C LANG=C

child_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  [[ -z "${child_pid}" ]] || kill "${child_pid}" 2>/dev/null || true
  if [[ -d "${rank_root}" && ! -L "${rank_root}" && "$(stat -c '%d:%i:%u:%a' -- "${rank_root}")" == "${rank_identity}" ]]; then
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
