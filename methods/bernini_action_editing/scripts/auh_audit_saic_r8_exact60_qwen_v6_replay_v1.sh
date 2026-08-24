#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly expected_job_id=134964
readonly expected_node=auh7-1b-gpu-283
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809
readonly stage_root="${experiment_root}/staging/allocation-134964-r8-qwen-v6-replay-45cfd756-r1"
readonly output_root="${experiment_root}/diagnostics/allocation-134964-r8-qwen-v6-replay-45cfd756-r1"
readonly source="${stage_root}/audit_saic_r8_exact60_qwen_v6_replay_v1.py"
readonly source_sha=45cfd756d5929126d591023a1c2b74b953dacd5431434ed99959e83bc53f7782
readonly qwen_source="${stage_root}/audit_saic_t2v_branch_semantics_qwen_v1.py"
readonly qwen_source_sha=bca26ff69e29ada23e35610a26810643010b2cc4d9a9707b8a068c20b53cbe66
readonly old_launcher="${stage_root}/launch_saic_fresh60_qwen_v6_69eec35.sh"
readonly old_launcher_sha=38f63226963b7d780639c3e7250916cde1f5a5e1012870d08cfcd8be03793a5a
readonly records="${experiment_root}/runs/saic-fresh60-qwen4b-v6-69eec35-r1/qwen3_vl_4b_branch_semantics_records.jsonl"
readonly records_sha=d885317804e62d9f58f183476f538f3e5dbba9f21579ddb8971ad160a48f38c4
readonly summary="${experiment_root}/runs/saic-fresh60-qwen4b-v6-69eec35-r1/qwen3_vl_4b_branch_semantics_summary.json"
readonly summary_sha=c6e5a995267ddb5779481c9837bc5458a1b1217a25ea70c2ee99d5d8d02445c7
readonly terminal="${experiment_root}/releases/saic-t2v-topup-r8-ddc8a79-r1/evidence-terminal-ddc8a79/saic-exact60-terminal-evidence-135056.json"
readonly terminal_sha=07a6ec7ccbe165d89aa8757985537ef18d62eea5d08e245e452b607dee5bd29a
readonly master="${experiment_root}/runs/t2v-events-topup-r8-ddc8a79-r1/saic-pure-t2v-event-bank-topup-receipt.json"
readonly master_sha=c5528a08fa976c0dbfb16984a35df3169c2d013a73fabd982ad45f45d5defc61
readonly deep_a="${experiment_root}/releases/saic-t2v-topup-r8-ddc8a79-r1/evidence-terminal-ddc8a79/deep-audit-sp4-a.json"
readonly deep_a_sha=2c5b47c306a7cd7895278c3bc668bc8c895328ff7c528afcab8b4ccbdd83a67e
readonly deep_b="${experiment_root}/releases/saic-t2v-topup-r8-ddc8a79-r1/evidence-terminal-ddc8a79/deep-audit-sp4-b.json"
readonly deep_b_sha=fca0e039259babae6188a8912a5990d16fe6584d9e8d8092eb02e036d83865d3
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python
readonly output="${output_root}/corrected-qwen-v6-replay-receipt.json"

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 64
fi
if [[ "${SLURM_JOB_ID:-}" != "${expected_job_id}" ]]; then
  echo "must run inside allocation ${expected_job_id}" >&2
  exit 65
fi
if [[ "${SLURMD_NODENAME:-}" != "${expected_node}" ]]; then
  echo "allocation node differs" >&2
  exit 66
fi

verify_plain_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "${label} is not a plain file: ${path}" >&2
    exit 67
  fi
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label} SHA-256 differs" >&2
    exit 68
  fi
}

verify_plain_sha256 "${source}" "${source_sha}" source
verify_plain_sha256 "${qwen_source}" "${qwen_source_sha}" qwen-v6-source
verify_plain_sha256 "${old_launcher}" "${old_launcher_sha}" old-launcher
verify_plain_sha256 "${records}" "${records_sha}" qwen-v6-records
verify_plain_sha256 "${summary}" "${summary_sha}" qwen-v6-summary
verify_plain_sha256 "${terminal}" "${terminal_sha}" terminal-evidence
verify_plain_sha256 "${master}" "${master_sha}" master-receipt
verify_plain_sha256 "${deep_a}" "${deep_a_sha}" deep-audit-sp4-a
verify_plain_sha256 "${deep_b}" "${deep_b_sha}" deep-audit-sp4-b
if [[ ! -x "${python_bin}" ]]; then
  echo "python interpreter is absent or non-executable" >&2
  exit 68
fi

if [[ -e "${output}" || -L "${output}" ]]; then
  echo "create-only output already exists: ${output}" >&2
  exit 69
fi
mkdir -p "${output_root}"

exec "${python_bin}" "${source}" \
  --old-launcher "${old_launcher}" \
  --records "${records}" \
  --summary "${summary}" \
  --terminal-evidence "${terminal}" \
  --master-receipt "${master}" \
  --deep-audit-sp4-a "${deep_a}" \
  --deep-audit-sp4-b "${deep_b}" \
  --output "${output}" \
  --expected-source-sha256 "${source_sha}"
