#!/bin/bash
# Exact per-rank boundary for the one-shot PRE_D0 Level-B P2 v4 render.

set -Eeuo pipefail

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly capacity_python=/usr/bin/python3.10
readonly capacity_python_sha=11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96
readonly env_bin=/usr/bin/env
readonly env_sha=85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0
readonly base64_bin=/usr/bin/base64
readonly base64_sha=b10f8c059f50c0681c6497e7b09ebdba168e341498ae1733de9089dc8efa0898
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v4
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly rank_self="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v4.sh"
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v4.py"
readonly bootstrap_sha=1d72a1594ab52e258f0fbac5410ea1d27e5c557a12d76e88b806b3ac99794391
readonly capacity_member="${launch_root}/action_edit_level_b_p2_00435_capacity_0817_v4.py"
readonly capacity_member_sha=87fc10c580070eef660fdfeaecf18ddd997d031a009508edbcc34a263cd6c4dc
readonly expected_capacity_receipt="${experiment_root}/attempts/${tag}/STARTED/step-capacity-receipt.json"

fail() {
  printf 'Level-B P2 v4 rank wrapper refused: %s\n' "$*" >&2
  exit 97
}

[[ "${BASH_ENV:-}" == /dev/null ]] || fail "pre-script BASH_ENV boundary differs"
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C && "${LANG:-}" == C ]] || \
  fail "pre-script path or locale boundary differs"
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${ROCM_SMI_LIB_PATH:-}" ]] || \
  fail "pre-script inherited loader or Python environment differs"

require_stat_value() {
  local stat_path="$1"
  local stat_format="$2"
  local stat_expected="$3"
  local stat_label="$4"
  local stat_observed
  if ! stat_observed="$(/usr/bin/stat -c "${stat_format}" "${stat_path}")"; then
    fail "${stat_label}: stat query failed"
  fi
  [[ "${stat_observed}" == "${stat_expected}" ]] || fail "${stat_label} differs"
}

require_sha256() {
  local sha_path="$1"
  local sha_expected="$2"
  local sha_label="$3"
  local sha_observed
  if ! sha_observed="$(/usr/bin/sha256sum "${sha_path}" | /usr/bin/awk '{print $1}')"; then
    fail "${sha_label}: SHA query failed"
  fi
  [[ "${sha_observed}" =~ ^[0-9a-f]{64}$ ]] || fail "${sha_label}: SHA format differs"
  [[ "${sha_observed}" == "${sha_expected}" ]] || fail "${sha_label}: SHA differs"
}

capture_capacity_validation_base64() {
  local capture_output_name="$1"
  shift
  local capture_frame
  local capture_status
  local capture_suffix
  local capture_payload
  local capture_sentinel=__LEVEL_B_P2_00435_V4_RANK_CAPACITY_PIPESTATUS_
  if ! capture_frame="$({
    set +e
    "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
      HOME=/nonexistent/bernini-level-b-p2-00435-v4-rank \
      "${capacity_python}" -I -S -B "${capacity_member}" "$@" 2>&1 | \
      "${base64_bin}" -w0
    capture_status=("${PIPESTATUS[@]}")
    printf '%s%03d_%03d__' "${capture_sentinel}" \
      "${capture_status[0]}" "${capture_status[1]}"
    exit 0
  })"; then
    fail "rank capacity framing failed"
  fi
  [[ "${capture_frame}" =~ ${capture_sentinel}([0-9]{3})_([0-9]{3})__$ ]] || \
    fail "rank capacity frame suffix differs"
  capture_suffix="${BASH_REMATCH[0]}"
  [[ "${BASH_REMATCH[1]}" == 000 && "${BASH_REMATCH[2]}" == 000 ]] || \
    fail "rank capacity validation or encoder failed"
  capture_payload="${capture_frame%"${capture_suffix}"}"
  [[ -n "${capture_payload}" && "${capture_payload}" != *$'\n'* \
    && "${capture_payload}" =~ ^[A-Za-z0-9+/]+={0,2}$ ]] || \
    fail "rank capacity transport base64 differs"
  printf -v "${capture_output_name}" '%s' "${capture_payload}"
}

for pending_sha in "${bootstrap_sha}" "${capacity_member_sha}" \
  "${capacity_python_sha}" "${base64_sha}" "${env_sha}"; do
  [[ "${pending_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
done
[[ $# == 2 && "$1" == "${bootstrap}" && "$2" == run ]] || fail "exact bootstrap argv differs"
[[ "$0" == "${rank_self}" ]] || fail "rank wrapper absolute path differs"
[[ -x "${rank_self}" && ! -L "${rank_self}" ]] || fail "rank wrapper file differs"
require_stat_value "${rank_self}" %a 555 "rank wrapper mode"
require_stat_value "${rank_self}" %h 1 "rank wrapper link count"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
observed_hostname=
if ! observed_hostname="$(/bin/hostname -s)"; then
  fail "physical hostname query failed"
fi
readonly observed_hostname
[[ "${observed_hostname}" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm child is absent"
[[ "${WORLD_SIZE:-}" == 8 && "${LOCAL_WORLD_SIZE:-}" == 8 ]] || fail "WORLD8 differs"
[[ "${LOCAL_RANK:-}" =~ ^[0-7]$ && "${RANK:-}" =~ ^[0-7]$ ]] || fail "rank geometry differs"
[[ "${TORCHELASTIC_MAX_RESTARTS:-}" == 0 ]] || fail "elastic max restarts differs"
[[ "${TORCHELASTIC_RESTART_COUNT:-}" == 0 ]] || fail "elastic restart occurred"
[[ "${PYTHONHASHSEED:-}" == 0 ]] || fail "Python hash seed differs"
[[ "${HF_HUB_OFFLINE:-}" == 1 && "${TRANSFORMERS_OFFLINE:-}" == 1 ]] || fail "offline mode differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"
require_stat_value "${python_bin}" %a 755 "pinned Python mode"
require_stat_value "${python_bin}" %h 1 "pinned Python link count"
require_stat_value "${python_bin}" %s 31490256 "pinned Python size"
require_sha256 "${python_bin}" "${python_sha}" "pinned Python"
[[ -f "${bootstrap}" && ! -L "${bootstrap}" ]] || fail "bootstrap file differs"
require_stat_value "${bootstrap}" %a 444 "bootstrap mode"
require_stat_value "${bootstrap}" %h 1 "bootstrap link count"
require_sha256 "${bootstrap}" "${bootstrap_sha}" "bootstrap"
[[ -f "${capacity_member}" && ! -L "${capacity_member}" ]] || fail "capacity member differs"
require_stat_value "${capacity_member}" %a 444 "capacity member mode"
require_stat_value "${capacity_member}" %h 1 "capacity member link count"
require_sha256 "${capacity_member}" "${capacity_member_sha}" "capacity member"
[[ -x "${capacity_python}" && ! -L "${capacity_python}" ]] || fail "capacity Python differs"
require_stat_value "${capacity_python}" %a 755 "capacity Python mode"
require_stat_value "${capacity_python}" %h 1 "capacity Python link count"
require_stat_value "${capacity_python}" %s 5937800 "capacity Python size"
require_sha256 "${capacity_python}" "${capacity_python_sha}" "capacity Python"
[[ -x "${env_bin}" && ! -L "${env_bin}" ]] || fail "env tool differs"
require_stat_value "${env_bin}" %a 755 "env tool mode"
require_stat_value "${env_bin}" %h 1 "env tool link count"
require_stat_value "${env_bin}" %s 43976 "env tool size"
require_sha256 "${env_bin}" "${env_sha}" "env tool"
[[ -x "${base64_bin}" && ! -L "${base64_bin}" ]] || fail "base64 tool differs"
require_stat_value "${base64_bin}" %a 755 "base64 tool mode"
require_stat_value "${base64_bin}" %h 1 "base64 tool link count"
require_stat_value "${base64_bin}" %s 35336 "base64 tool size"
require_sha256 "${base64_bin}" "${base64_sha}" "base64 tool"
readonly capacity_receipt="${LEVEL_B_STEP_CAPACITY_RECEIPT:-}"
readonly capacity_receipt_sha="${LEVEL_B_STEP_CAPACITY_RECEIPT_SHA256:-}"
readonly capacity_challenge="${LEVEL_B_STEP_CAPACITY_CHALLENGE:-}"
[[ "${capacity_receipt}" == "${expected_capacity_receipt}" ]] || fail "step capacity receipt path differs"
[[ "${capacity_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "step capacity receipt SHA differs"
[[ "${capacity_challenge}" =~ ^[0-9a-f]{64}$ ]] || fail "step capacity challenge differs"
validated_capacity=
capacity_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before rank receipt validation"
capture_capacity_validation_base64 capacity_transport validate-file \
  "${capacity_receipt}" "${capacity_receipt_sha}" step "${capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after rank receipt validation"
if ! validated_capacity="$(printf '%s' "${capacity_transport}" | "${base64_bin}" -d)"; then
  fail "rank capacity transport decode failed"
fi
readonly validated_capacity
capacity_transport_roundtrip=
if ! capacity_transport_roundtrip="$(printf '%s' "${validated_capacity}" | "${base64_bin}" -w0)"; then
  fail "rank capacity transport re-encode failed"
fi
readonly capacity_transport_roundtrip
[[ "${capacity_transport_roundtrip}" == "${capacity_transport}" ]] || \
  fail "rank capacity transport contains trailing bytes"
validated_capacity_sha=
if ! validated_capacity_sha="$(printf '%s' "${validated_capacity}" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "validated step capacity receipt SHA query failed"
fi
readonly validated_capacity_sha
[[ "${validated_capacity_sha}" == "${capacity_receipt_sha}" ]] || fail "validated step capacity receipt bytes differ"
unset LEVEL_B_STEP_CAPACITY_RECEIPT
unset LEVEL_B_STEP_CAPACITY_RECEIPT_SHA256
unset LEVEL_B_STEP_CAPACITY_CHALLENGE
unset BASH_ENV ENV

exec "${python_bin}" -I -B "$@"
