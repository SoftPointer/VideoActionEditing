#!/usr/bin/env bash
# Per-rank node-local cache wrapper for the confirmation40 native renderer.
#
# AMD COMGR must not place its temporary hipfatbin/unbundle objects on the
# shared NFS experiment tree.  torchrun invokes this wrapper with --no_python;
# each rank creates a private directory below the launcher's authenticated
# node-local scratch root, then starts the frozen Python worker inside it.

set -Eeuo pipefail
umask 077

fail() { echo "[generic-action-confirmation-rank-cache-v1] ERROR: $*" >&2; exit 2; }

readonly cache_token="${GADP_RANK_CACHE_TOKEN:?set rank cache token}"
readonly scratch_parent="${GADP_NODE_LOCAL_SCRATCH:?set node-local scratch}"
readonly expected_fstype="${GADP_NODE_LOCAL_SCRATCH_FSTYPE:?set scratch filesystem}"
readonly python_bin="${GADP_RANK_PYTHON_BIN:?set frozen Python executable}"
readonly method_root="${GADP_METHOD_ROOT:?set released method root}"
readonly resource_contract_sha="${GADP_RESOURCE_CONTRACT_SHA256:?pin r13 resource contract}"
readonly controller_sha="${GADP_CONTROLLER_SHA256:?pin confirmation controller}"
readonly local_rank="${LOCAL_RANK:?torchrun LOCAL_RANK required}"
readonly global_rank="${RANK:?torchrun RANK required}"
readonly world_size="${WORLD_SIZE:?torchrun WORLD_SIZE required}"
readonly job_id="${SLURM_JOB_ID:?Slurm job required}"
readonly step_id="${SLURM_STEP_ID:?numbered Slurm step required}"

[[ $# -ge 1 ]] || fail "worker path is required"
[[ "${cache_token}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$ ]] || fail "cache token differs"
[[ "${world_size}" == 4 && "${local_rank}" =~ ^[0-3]$ && "${global_rank}" =~ ^[0-3]$ ]] || fail "WORLD4 rank identity differs"
[[ "${global_rank}" == "${local_rank}" ]] || fail "single-node WORLD4 rank mapping differs"
[[ "${scratch_parent}" == /* && "${scratch_parent}" != / && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "node-local scratch root differs"
[[ "${method_root}" == /* && -d "${method_root}" && ! -L "${method_root}" && "$(readlink -f -- "${method_root}")" == "${method_root}" ]] || fail "released method root differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" && "$(readlink -f -- "${python_bin}")" == "${python_bin}" ]] || fail "frozen Python differs"
readonly worker="$1"
[[ "${worker}" == "${method_root}/"* && -f "${worker}" && ! -L "${worker}" && "$(readlink -f -- "${worker}")" == "${worker}" ]] || fail "worker is outside the released method root"
readonly resource_contract="${method_root}/tools/reserve4_fixed_generation_sp4_v1.py"
readonly controller="${method_root}/generic_action_confirmation_data_prep_controller_v1.py"
[[ -f "${resource_contract}" && ! -L "${resource_contract}" && "$(sha256sum "${resource_contract}" | awk '{print $1}')" == "${resource_contract_sha}" ]] || fail "r13 resource contract differs"
[[ -f "${controller}" && ! -L "${controller}" && "$(sha256sum "${controller}" | awk '{print $1}')" == "${controller_sha}" ]] || fail "confirmation controller differs"

readonly scratch_real="$(readlink -f -- "${scratch_parent}")"
[[ "${scratch_real}" == "${scratch_parent}" ]] || fail "node-local scratch canonical path differs"
readonly observed_fstype="$(stat -f -c '%T' -- "${scratch_real}")"
[[ "${observed_fstype}" == "${expected_fstype}" ]] || fail "node-local scratch filesystem changed"
case "${observed_fstype}" in
  ext2/ext3|xfs|tmpfs) ;;
  *) fail "COMGR scratch filesystem is not an allowed node-local type: ${observed_fstype}" ;;
esac

rank_root="$(mktemp -d -- "${scratch_real}/gacp-${job_id}-${step_id}-${cache_token}-r${global_rank}.XXXXXXXX")"
readonly rank_root
readonly rank_identity="$(stat -c '%d:%i:%u:%a' -- "${rank_root}")"
for leaf in tmp xdg hf torch-extensions triton torchinductor pycache miopen-user miopen-custom; do
  mkdir -m 0700 "${rank_root}/${leaf}"
done

export TMPDIR="${rank_root}/tmp"
export TMP="${rank_root}/tmp"
export TEMP="${rank_root}/tmp"
export XDG_CACHE_HOME="${rank_root}/xdg"
export HF_HOME="${rank_root}/hf"
export TORCH_EXTENSIONS_DIR="${rank_root}/torch-extensions"
export TRITON_CACHE_DIR="${rank_root}/triton"
export TORCHINDUCTOR_CACHE_DIR="${rank_root}/torchinductor"
export PYTHONPYCACHEPREFIX="${rank_root}/pycache"
export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export GADP_RANK_CACHE_FILESYSTEM="${observed_fstype}"
export GADP_RANK_CACHE_IS_NODE_LOCAL=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export LC_ALL=C LANG=C

child_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  if [[ -d "${rank_root}" && ! -L "${rank_root}" && "$(stat -c '%d:%i:%u:%a' -- "${rank_root}")" == "${rank_identity}" ]]; then
    find "${rank_root}" -xdev -depth -mindepth 1 -delete || exit 70
    rmdir "${rank_root}" || exit 70
  else
    exit 70
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM HUP

"${python_bin}" -B "${resource_contract}" assert-host-memory-monitor-live >/dev/null || fail "pre-candidate host monitor boundary failed"
"${python_bin}" -B "$@" &
child_pid=$!
set +e
wait "${child_pid}"
status=$?
set -e
child_pid=""
(( status == 0 )) || exit "${status}"
"${python_bin}" -B "${resource_contract}" assert-host-memory-monitor-live >/dev/null || fail "post-candidate host monitor boundary failed"

if [[ "${global_rank}" == 0 && "${cache_token}" == compile-smoke-* ]]; then
  readonly physical_output="${GADP_PHYSICAL_SMOKE_RECEIPT_OUTPUT:?set physical smoke receipt output}"
  candidate_output=""
  expect_output=0
  output_count=0
  for token in "$@"; do
    if (( expect_output == 1 )); then
      candidate_output="${token}"
      output_count=$((output_count + 1))
      expect_output=0
    elif [[ "${token}" == --output-dir ]]; then
      expect_output=1
    fi
  done
  [[ "${output_count}" == 1 && "${candidate_output}" == /* ]] || fail "compile-smoke candidate output argument differs"
  "${python_bin}" -B "${controller}" seal-physical-smoke-receipt \
    --plan "${GADP_UPSTREAM_PLAN:?set upstream plan}" \
    --expected-plan-sha256 "${GADP_UPSTREAM_PLAN_SHA256:?pin upstream plan}" \
    --candidate-output "${candidate_output}" \
    --r10-compile-smoke-receipt "${GADP_R10_COMPILE_SMOKE_RECEIPT:?bind r10 receipt}" \
    --expected-r10-compile-smoke-receipt-sha256 "${GADP_R10_COMPILE_SMOKE_RECEIPT_SHA256:?pin r10 receipt}" \
    --r10-generation-log "${GADP_R10_GENERATION_LOG:?bind r10 log}" \
    --expected-r10-generation-log-sha256 "${GADP_R10_GENERATION_LOG_SHA256:?pin r10 log}" \
    --output "${physical_output}" >/dev/null || fail "physical safe_open smoke receipt failed"
  [[ -f "${physical_output}" && ! -L "${physical_output}" ]] || fail "physical smoke receipt is absent"
  "${python_bin}" -B "${resource_contract}" assert-host-memory-monitor-live >/dev/null || fail "post-safe-open host monitor boundary failed"
fi
exit "${status}"
