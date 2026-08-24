#!/usr/bin/env bash
# Launch four pre-registered V3 arms only after V2 fitted+unseen workers finish.
# Never cancel, release, requeue, or signal parent allocation Job 140846.

set -Eeuo pipefail

job_id=140846
wait_for="${ACTION_ENDPOINT_WAIT_FOR:?set V2 unseen EVALUATION_COMPLETE path}"
root="${ACTION_ENDPOINT_EXPERIMENT_ROOT:?set fresh V3 experiment root}"
archive="${ACTION_ENDPOINT_SOURCE_ARCHIVE:?set sealed source archive}"
archive_sha="${ACTION_ENDPOINT_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
revision="${ACTION_ENDPOINT_SOURCE_REVISION:?set source revision}"
runner="${ACTION_ENDPOINT_NODE_RUNNER:?set endpoint node runner}"
manifest="${ACTION_ENDPOINT_SOURCE_MANIFEST:?set source-only manifest}"
manifest_sha="${ACTION_ENDPOINT_SOURCE_MANIFEST_SHA256:?set manifest SHA-256}"
cache="${ACTION_ENDPOINT_CACHE:?set sealed V2 teacher cache}"
cache_sha="${ACTION_ENDPOINT_CACHE_SHA256:?set V2 teacher cache SHA-256}"
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
arms=(
  endpoint_cell_band
  endpoint_consensus_band
  endpoint_consensus_trust_001
  endpoint_consensus_trust_010
)

[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ && "${cache_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ -f "${archive}" && -f "${manifest}" && -f "${cache}" && -x "${runner}" ]]
[[ ! -e "${root}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
[[ "$(sha256sum "${cache}" | awk '{print $1}')" == "${cache_sha}" ]]
required_archive_members=(
  methods/bernini_action_editing/full30_action_learning_v1.py
  methods/bernini_action_editing/self_generated_action_endpoint_consensus_v3.py
  methods/bernini_action_editing/train_lora.py
  methods/bernini_action_editing/train_self_generated_action_endpoint_consensus_v3.py
  methods/bernini_action_editing/train_self_generated_action_quotient_v1.py
  methods/bernini_action_editing/train_self_generated_action_residual_margin_v2.py
)
archive_members="$(tar -tf "${archive}")"
for member in "${required_archive_members[@]}"; do
  grep -Fxq "${member}" <<<"${archive_members}"
done

while [[ ! -f "${wait_for}" ]]; do sleep 60; done
sleep 60
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
mkdir -p "${root}/logs" "${root}/runs"

common_env=(
  ACTION_ENDPOINT_SOURCE_ARCHIVE="${archive}"
  ACTION_ENDPOINT_SOURCE_ARCHIVE_SHA256="${archive_sha}"
  ACTION_ENDPOINT_SOURCE_REVISION="${revision}"
  ACTION_ENDPOINT_SOURCE_MANIFEST="${manifest}"
  ACTION_ENDPOINT_SOURCE_MANIFEST_SHA256="${manifest_sha}"
  ACTION_ENDPOINT_CACHE="${cache}"
  ACTION_ENDPOINT_CACHE_SHA256="${cache_sha}"
  ACTION_ENDPOINT_SEED=20260817
)

# A create-only one-update canary proves the new endpoint authority, full
# frozen reference, backward path, checkpoint, and receipt before four nodes
# are occupied.  It is retained as mechanics evidence and never reused.
srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
  --nodelist=auh7-1b-gpu-246 --cpus-per-task=32 --mem=60G \
  --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
  "${common_env[@]}" ACTION_ENDPOINT_ARM=endpoint_consensus_trust_010 \
  ACTION_ENDPOINT_MAX_STEPS=1 ACTION_ENDPOINT_OUTPUT="${root}/canary" \
  "${runner}" >"${root}/logs/canary.log" 2>&1
[[ -f "${root}/canary/checkpoint-00000001/receipt.json" ]]

status=0
pids=()
for index in 0 1 2 3; do
  arm="${arms[$index]}"
  output="${root}/runs/${arm}"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${nodes[$index]}" --cpus-per-task=32 --mem=60G \
    --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${common_env[@]}" ACTION_ENDPOINT_ARM="${arm}" ACTION_ENDPOINT_MAX_STEPS=80 \
    ACTION_ENDPOINT_OUTPUT="${output}" "${runner}" \
    >"${root}/logs/train-${arm}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 ))

for arm in "${arms[@]}"; do
  for step in 10 20 40 80; do
    [[ -f "${root}/runs/${arm}/checkpoint-$(printf '%08d' "${step}")/receipt.json" ]]
  done
done
printf 'training_complete=true\nparent_allocation_cancelled=false\n' >"${root}/TRAINING_COMPLETE"
