#!/usr/bin/env bash
# Login-node supervisor: wait without holding GPUs, then reuse the same parent.

set -Eeuo pipefail
umask 077

training_run="${PRESERVATION_CHAIN_TRAINING_RUN:?set training run}"
holder_job="${PRESERVATION_CHAIN_HOLDER_JOB:?set holder job}"
holder_node="${PRESERVATION_CHAIN_HOLDER_NODE:?set holder node}"
method_root="${PRESERVATION_CHAIN_INFER_METHOD_ROOT:?set inference method root}"
run_root="${PRESERVATION_CHAIN_INFER_RUN_ROOT:?set inference run root}"
runtime_revision="${PRESERVATION_CHAIN_RUNTIME_REVISION:?set runtime revision}"
runtime_archive_sha="${PRESERVATION_CHAIN_RUNTIME_ARCHIVE_SHA256:?set runtime archive SHA}"
base_port="${PRESERVATION_CHAIN_BASE_PORT:?set base port}"
poll_seconds="${PRESERVATION_CHAIN_POLL_SECONDS:-60}"

fail() { echo "[preservation-chain] ERROR: $*" >&2; exit 2; }
[[ -d "${training_run}" && ! -L "${training_run}" ]] || fail "training run differs"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "inference run must be fresh"
[[ "${poll_seconds}" =~ ^[0-9]+$ ]] && (( poll_seconds >= 30 && poll_seconds <= 60 )) || fail "poll interval differs"
case "${holder_job}:${holder_node}" in
  135407:auh7-1b-gpu-260|135411:auh7-1b-gpu-214) ;;
  *) fail "holder allowlist differs" ;;
esac

while [[ ! -f "${training_run}/controller.COMPLETE" ]]; do
  if [[ -f "${training_run}/controller.status" ]]; then
    child_exit="$(awk -F= '$1=="child_exit" {print $2}' "${training_run}/controller.status")"
    [[ "${child_exit}" == 0 ]] || fail "training child exited ${child_exit:-unknown}"
  fi
  job_record="$(scontrol show job -o "${holder_job}")"
  [[ "${job_record}" == *"JobState=RUNNING"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || fail "parent holder stopped while waiting"
  sleep "${poll_seconds}"
done

bundle="${training_run}/training"
[[ -f "${bundle}/receipt.json" && -f "${bundle}/adapter.safetensors" ]] || fail "completed training bundle differs"
env \
  PRESERVATION_INFER_HOLDER_JOB="${holder_job}" \
  PRESERVATION_INFER_HOLDER_NODE="${holder_node}" \
  PRESERVATION_INFER_METHOD_ROOT="${method_root}" \
  PRESERVATION_INFER_TRAINING_BUNDLE="${bundle}" \
  PRESERVATION_INFER_RUN_ROOT="${run_root}" \
  PRESERVATION_INFER_RUNTIME_REVISION="${runtime_revision}" \
  PRESERVATION_INFER_RUNTIME_ARCHIVE_SHA256="${runtime_archive_sha}" \
  PRESERVATION_INFER_BASE_PORT="${base_port}" \
  bash "${method_root}/scripts/auh_infer_preservation_residual_single_holder_v1.sh"
