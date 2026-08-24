#!/usr/bin/env bash
# After v1 fitted+unseen evaluation, cache and train eight residual-margin arms.
# Never cancel, release, requeue, or signal the parent allocation.

set -Eeuo pipefail

job_id=140846
wait_for="${ACTION_RESIDUAL_WAIT_FOR:?set prior EVALUATION_COMPLETE path}"
early_launch="${ACTION_RESIDUAL_EARLY_LAUNCH:-0}"
root="${ACTION_RESIDUAL_EXPERIMENT_ROOT:?set fresh residual experiment root}"
archive="${ACTION_RESIDUAL_SOURCE_ARCHIVE:?set sealed source archive}"
archive_sha="${ACTION_RESIDUAL_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
revision="${ACTION_RESIDUAL_SOURCE_REVISION:?set source revision}"
runner="${ACTION_RESIDUAL_NODE_RUNNER:?set detached node runner}"
manifest="${ACTION_RESIDUAL_SOURCE_MANIFEST:?set source-only manifest}"
manifest_sha="${ACTION_RESIDUAL_SOURCE_MANIFEST_SHA256:?set manifest SHA-256}"
cache="${root}/teacher-cache-residual-row4-slot4.pt"
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-248 auh7-1b-gpu-279 auh7-1b-gpu-279)
groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)
arms=(
  margin_005 margin_010 margin_020 margin_010_perp_010
  margin_010_perp_100 margin_010_perp_100_onset_100
  margin_010_perp_100_onset_400 margin_010_perp_100_onset_400_noop_020
)

[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "${early_launch}" == 0 || "${early_launch}" == 1 ]]
[[ -f "${archive}" && -f "${manifest}" && -x "${runner}" && ! -e "${root}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]

if [[ "${early_launch}" == 0 ]]; then
  while [[ ! -f "${wait_for}" ]]; do sleep 60; done
  sleep 60
  canary_group=0,1,2,3
else
  # The fitted/unseen rollout uses GPU 0-3 on every node.  Early launch is an
  # explicit operator-selected schedule for the otherwise idle GPU 4-7.
  # Run one arm per node at a time so two arms never share the same SP4 island.
  canary_group=4,5,6,7
  nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
  groups=(4,5,6,7 4,5,6,7 4,5,6,7 4,5,6,7 4,5,6,7 4,5,6,7 4,5,6,7 4,5,6,7)
fi
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
mkdir -p "${root}/logs" "${root}/runs"

common_env=(
  ACTION_RESIDUAL_SOURCE_ARCHIVE="${archive}"
  ACTION_RESIDUAL_SOURCE_ARCHIVE_SHA256="${archive_sha}"
  ACTION_RESIDUAL_SOURCE_REVISION="${revision}"
  ACTION_RESIDUAL_SOURCE_MANIFEST="${manifest}"
  ACTION_RESIDUAL_SOURCE_MANIFEST_SHA256="${manifest_sha}"
  ACTION_RESIDUAL_SEED=20260817
  ACTION_RESIDUAL_SLOTS=4
)

srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-246 \
  --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES="${canary_group}" \
  "${common_env[@]}" ACTION_RESIDUAL_MODE=cache ACTION_RESIDUAL_LIMIT_CELLS=1 \
  ACTION_RESIDUAL_CACHE="${root}/canary-cache-unused.pt" \
  ACTION_RESIDUAL_OUTPUT="${root}/teacher-cache-canary.pt" \
  "${runner}" >"${root}/logs/cache-canary.log" 2>&1
[[ -f "${root}/teacher-cache-canary.pt" ]]

srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-246 \
  --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES="${canary_group}" \
  "${common_env[@]}" ACTION_RESIDUAL_MODE=cache ACTION_RESIDUAL_LIMIT_CELLS=0 \
  ACTION_RESIDUAL_CACHE="${root}/full-cache-unused.pt" ACTION_RESIDUAL_OUTPUT="${cache}" \
  "${runner}" >"${root}/logs/cache-full.log" 2>&1
[[ -f "${cache}" ]]
cache_sha="$(sha256sum "${cache}" | awk '{print $1}')"

# Prove the new online frozen-reference forward and one optimizer update before
# occupying all eight GPU islands.  The create-only canary is retained.
srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-246 \
  --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES="${canary_group}" \
  "${common_env[@]}" ACTION_RESIDUAL_MODE=train ACTION_RESIDUAL_CACHE="${cache}" \
  ACTION_RESIDUAL_EXPECTED_CACHE_SHA256="${cache_sha}" \
  ACTION_RESIDUAL_OUTPUT="${root}/train-canary-margin-010" \
  ACTION_RESIDUAL_ARM=margin_010 ACTION_RESIDUAL_MAX_STEPS=1 \
  "${runner}" >"${root}/logs/train-canary-margin-010.log" 2>&1
[[ -f "${root}/train-canary-margin-010/checkpoint-00000001/receipt.json" ]]

run_round() {
  local index arm output
  local status=0
  local -a pids=()
  for index in "$@"; do
    arm="${arms[$index]}"
    output="${root}/runs/${arm}"
    [[ ! -e "${output}" ]]
    srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[$index]}" \
      --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
      env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
      ROCR_VISIBLE_DEVICES="${groups[$index]}" "${common_env[@]}" \
      ACTION_RESIDUAL_MODE=train ACTION_RESIDUAL_CACHE="${cache}" \
      ACTION_RESIDUAL_EXPECTED_CACHE_SHA256="${cache_sha}" ACTION_RESIDUAL_OUTPUT="${output}" \
      ACTION_RESIDUAL_ARM="${arm}" ACTION_RESIDUAL_MAX_STEPS=160 \
      "${runner}" >"${root}/logs/train-${arm}.log" 2>&1 &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do wait "${pids[$index]}" || status=1; done
  (( status == 0 ))
}

if [[ "${early_launch}" == 0 ]]; then
  run_round 0 1 2 3 4 5 6 7
else
  run_round 0 1 2 3
  run_round 4 5 6 7
fi
for arm in "${arms[@]}"; do
  for step in 10 20 40 80 160; do
    [[ -f "${root}/runs/${arm}/checkpoint-$(printf '%08d' "${step}")/receipt.json" ]]
  done
done
printf 'training_complete=true\ncache_sha256=%s\nparent_allocation_cancelled=false\n' \
  "${cache_sha}" >"${root}/TRAINING_COMPLETE"
