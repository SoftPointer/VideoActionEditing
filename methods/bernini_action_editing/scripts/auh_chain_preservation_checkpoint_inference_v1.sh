#!/usr/bin/env bash
# Wait for one live continuous exact40 run, then infer its step20 and final40
# bundles in order.  This login-node supervisor never signals or releases the
# retained parent allocation.
set -Eeuo pipefail
umask 077

fail() { echo "[preservation-checkpoint-chain] ERROR: $*" >&2; exit 2; }

holder_job="${PRESERVATION_CHAIN_HOLDER_JOB:?set holder job}"
holder_node="${PRESERVATION_CHAIN_HOLDER_NODE:?set holder node}"
method_root="${PRESERVATION_CHAIN_METHOD_ROOT:?set sealed method root}"
training_run="${PRESERVATION_CHAIN_TRAINING_RUN:?set continuous training run}"
chain_root="${PRESERVATION_CHAIN_ROOT:?set fresh chain root}"
runtime_revision="${PRESERVATION_CHAIN_RUNTIME_REVISION:?set runtime revision}"
runtime_archive_sha="${PRESERVATION_CHAIN_RUNTIME_ARCHIVE_SHA256:?set archive SHA}"
step20_port="${PRESERVATION_CHAIN_STEP20_PORT:?set step20 port}"
step40_port="${PRESERVATION_CHAIN_STEP40_PORT:?set step40 port}"
poll_seconds="${PRESERVATION_CHAIN_POLL_SECONDS:-60}"

case "${holder_job}:${holder_node}" in
  135407:auh7-1b-gpu-260|135411:auh7-1b-gpu-214) ;;
  *) fail "holder allowlist differs" ;;
esac
for value in "${method_root}" "${training_run}" "${chain_root}"; do
  [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "path is outside user experiment root"
done
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "sealed method root differs"
[[ -d "${training_run}" && ! -L "${training_run}" ]] || fail "training run differs"
[[ ! -e "${chain_root}" && ! -L "${chain_root}" ]] || fail "chain root must be fresh"
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
[[ "${runtime_archive_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "runtime archive SHA differs"
for port in "${step20_port}" "${step40_port}"; do
  [[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 65533 )) || fail "port differs"
done
[[ "${step20_port}" != "${step40_port}" ]] || fail "checkpoint ports must differ"
[[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] && (( poll_seconds <= 3600 )) || fail "poll interval differs"

mkdir -m 0700 "${chain_root}" "${chain_root}/logs"
printf 'state=waiting_for_continuous_exact40\nparent_job=%s\nparent_node=%s\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" >"${chain_root}/chain.status"

assert_parent_running() {
  local record
  record="$(scontrol show job -o "${holder_job}")"
  [[ "${record}" == *"JobId=${holder_job} "* && "${record}" == *"JobState=RUNNING"* ]] || fail "parent is not RUNNING"
  [[ "${record}" == *"UserId=guangyi.chen"* && "${record}" == *"NodeList=${holder_node}"* ]] || fail "parent identity differs"
}

while [[ ! -f "${training_run}/controller.COMPLETE" ]]; do
  assert_parent_running
  if [[ -f "${training_run}/controller.status" ]] && \
     awk -F= '$1=="child_exit" && $2!="0" {bad=1} END {exit !bad}' "${training_run}/controller.status"; then
    fail "training controller recorded a non-zero child exit"
  fi
  sleep "${poll_seconds}"
done
assert_parent_running

step0_bundle="${training_run}/checkpoints/step-00000000"
step20_bundle="${training_run}/checkpoints/step-00000020"
step40_bundle="${training_run}/training"
for bundle in "${step0_bundle}" "${step20_bundle}" "${step40_bundle}"; do
  [[ -d "${bundle}" && ! -L "${bundle}" ]] || fail "continuous checkpoint bundle missing"
  for name in adapter.safetensors optimizer.pt history.json receipt.json; do
    [[ -f "${bundle}/${name}" && ! -L "${bundle}/${name}" ]] || fail "checkpoint bundle closure differs"
  done
done

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
"${python_bin}" -I -S -B - "${step0_bundle}/receipt.json" "${step20_bundle}/receipt.json" "${step40_bundle}/receipt.json" <<'PY'
import hashlib
import json
import sys

def read(path):
    value = json.load(open(path, encoding="ascii"))
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest")
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    assert hashlib.sha256(raw).hexdigest() == declared
    return value

zero, twenty, final = map(read, sys.argv[1:])
for value, step in ((zero, 0), (twenty, 20)):
    assert value["complete"] is True
    assert value["checkpoint_bundle"] is True
    assert value["continuous_trajectory"] is True
    assert value["trajectory_optimizer_steps"] == 40
    assert value["optimizer_steps"] == step
assert final["complete"] is True
assert final["continuous_trajectory"] is True
assert final["trajectory_optimizer_steps"] == 40
assert final["optimizer_steps"] == 40
assert final["checkpoint_steps"] == [0, 20, 40]
assert [row["optimizer_step"] for row in final["checkpoint_bundles"]] == [0, 20]
assert final["checkpoint_bundles"][0]["receipt_digest"] == zero["receipt_digest"]
assert final["checkpoint_bundles"][1]["receipt_digest"] == twenty["receipt_digest"]
PY

printf 'state=running_step20\nparent_job=%s\nparent_node=%s\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" >"${chain_root}/chain.status"
env PRESERVATION_INFER_HOLDER_JOB="${holder_job}" \
  PRESERVATION_INFER_HOLDER_NODE="${holder_node}" \
  PRESERVATION_INFER_METHOD_ROOT="${method_root}" \
  PRESERVATION_INFER_TRAINING_BUNDLE="${step20_bundle}" \
  PRESERVATION_INFER_RUN_ROOT="${chain_root}/step20" \
  PRESERVATION_INFER_RUNTIME_REVISION="${runtime_revision}" \
  PRESERVATION_INFER_RUNTIME_ARCHIVE_SHA256="${runtime_archive_sha}" \
  PRESERVATION_INFER_BASE_PORT="${step20_port}" \
  bash "${method_root}/scripts/auh_infer_preservation_residual_single_holder_v1.sh" \
  >"${chain_root}/logs/step20.log" 2>&1

printf 'state=running_step40\nparent_job=%s\nparent_node=%s\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" >"${chain_root}/chain.status"
env PRESERVATION_INFER_HOLDER_JOB="${holder_job}" \
  PRESERVATION_INFER_HOLDER_NODE="${holder_node}" \
  PRESERVATION_INFER_METHOD_ROOT="${method_root}" \
  PRESERVATION_INFER_TRAINING_BUNDLE="${step40_bundle}" \
  PRESERVATION_INFER_RUN_ROOT="${chain_root}/step40" \
  PRESERVATION_INFER_RUNTIME_REVISION="${runtime_revision}" \
  PRESERVATION_INFER_RUNTIME_ARCHIVE_SHA256="${runtime_archive_sha}" \
  PRESERVATION_INFER_BASE_PORT="${step40_port}" \
  bash "${method_root}/scripts/auh_infer_preservation_residual_single_holder_v1.sh" \
  >"${chain_root}/logs/step40.log" 2>&1

printf 'state=complete\nparent_job=%s\nparent_node=%s\nparent_not_released=true\ncheckpoints=0,20,40\n' \
  "${holder_job}" "${holder_node}" >"${chain_root}/chain.status"
printf 'COMPLETE checkpoint_chain=0,20,40 parent_retained=true\n' >"${chain_root}/chain.COMPLETE"
