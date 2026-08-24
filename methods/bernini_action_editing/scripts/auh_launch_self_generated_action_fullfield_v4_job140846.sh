#!/usr/bin/env bash
# Validate one strong overfit control, then launch four V4 arms on Job 140846.
# Never cancel, release, requeue, or signal the parent allocation.

set -Eeuo pipefail

job_id=140846
root="${ACTION_FULLFIELD_EXPERIMENT_ROOT:?set fresh V4 experiment root}"
runner="${ACTION_FULLFIELD_NODE_RUNNER:?set V4 node runner}"
archive="${ACTION_FULLFIELD_SOURCE_ARCHIVE:?set source archive}"
archive_sha="${ACTION_FULLFIELD_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
revision="${ACTION_FULLFIELD_SOURCE_REVISION:?set source revision}"
manifest="${ACTION_FULLFIELD_SOURCE_MANIFEST:?set source manifest}"
manifest_sha="${ACTION_FULLFIELD_SOURCE_MANIFEST_SHA256:?set source manifest SHA-256}"
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
arms=(
  direct_anchor_sft
  fullfield_action_noop
  fullfield_action_noop_pcgrad_preserve
  source_carrier_sft
)

[[ ! -e "${root}" && -x "${runner}" && -f "${archive}" && -f "${manifest}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
mkdir -p "${root}/logs" "${root}/runs"

common_env=(
  ACTION_FULLFIELD_SOURCE_ARCHIVE="${archive}"
  ACTION_FULLFIELD_SOURCE_ARCHIVE_SHA256="${archive_sha}"
  ACTION_FULLFIELD_SOURCE_REVISION="${revision}"
  ACTION_FULLFIELD_SOURCE_MANIFEST="${manifest}"
  ACTION_FULLFIELD_SOURCE_MANIFEST_SHA256="${manifest_sha}"
)

# Use a real fixed-state one-video overfit control.  Increase only the number
# of retained real training records if the measured per-rank peak is <=50%.
accepted_micro=""
for micro in 1 2 3 4; do
  canary="${root}/canary-direct-anchor-micro${micro}"
  if srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist=auh7-1b-gpu-246 --cpus-per-task=32 --mem=60G \
    --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${common_env[@]}" ACTION_FULLFIELD_ARM=direct_anchor_sft \
    ACTION_FULLFIELD_MAX_STEPS=10 ACTION_FULLFIELD_MICRO_RECORDS="${micro}" \
    ACTION_FULLFIELD_OVERFIT_ROW=0 ACTION_FULLFIELD_OUTPUT="${canary}" \
    "${runner}" >"${root}/logs/canary-micro${micro}.log" 2>&1; then
    [[ -f "${canary}/checkpoint-00000010/receipt.json" ]]
    accepted_micro="${micro}"
    break
  fi
  if ! grep -Fq \
    "real training peak reserved memory is not strictly above 50%" \
    "${root}/logs/canary-micro${micro}.log"; then
    exit 1
  fi
  [[ ! -e "${canary}" ]]
done
[[ -n "${accepted_micro}" ]]
printf 'accepted_true_micro_records=%s\n' "${accepted_micro}" >"${root}/CANARY_ACCEPTED"

run_canary() {
  local index="$1" arm node output log
  arm="${arms[$index]}"
  node="${nodes[$index]}"
  output="${root}/arm-canaries/${arm}"
  log="${root}/logs/arm-canary-${arm}.log"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${node}" --cpus-per-task=32 --mem=60G \
    --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${common_env[@]}" ACTION_FULLFIELD_ARM="${arm}" \
    ACTION_FULLFIELD_MAX_STEPS=2 ACTION_FULLFIELD_MICRO_RECORDS="${accepted_micro}" \
    ACTION_FULLFIELD_OUTPUT="${output}" "${runner}" >"${log}" 2>&1
  [[ -f "${output}/checkpoint-00000002/receipt.json" ]]
  [[ -f "${output}/TRAINING_COMPLETE" ]]
}

# The overfit arm already passed ten updates.  Every other objective must prove
# that its own graph reaches all 480 LoRA tensors and passes the real >50%
# memory gate before any formal run starts.
mkdir -p "${root}/arm-canaries"
canary_status=0
canary_pids=()
for index in 1 2 3; do run_canary "${index}" & canary_pids+=("$!"); done
for pid in "${canary_pids[@]}"; do wait "${pid}" || canary_status=1; done
(( canary_status == 0 ))

run_arm() {
  local index="$1" arm node micro output log attempt_log
  arm="${arms[$index]}"
  node="${nodes[$index]}"
  for micro in 1 2 3 4; do
    (( micro >= accepted_micro )) || continue
    output="${root}/runs/${arm}"
    log="${root}/logs/train-${arm}-micro${micro}.log"
    if srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
      --nodelist="${node}" --cpus-per-task=32 --mem=60G \
      --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
      env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
      "${common_env[@]}" ACTION_FULLFIELD_ARM="${arm}" \
      ACTION_FULLFIELD_MAX_STEPS=40 ACTION_FULLFIELD_MICRO_RECORDS="${micro}" \
      ACTION_FULLFIELD_OUTPUT="${output}" "${runner}" >"${log}" 2>&1; then
      [[ -f "${output}/checkpoint-00000040/receipt.json" ]]
      return 0
    fi
    attempt_log="${log}"
    if ! grep -Fq \
      "real training peak reserved memory is not strictly above 50%" \
      "${attempt_log}"; then
      return 1
    fi
    [[ ! -e "${output}" ]]
  done
  return 1
}

status=0
pids=()
for index in 0 1 2 3; do run_arm "${index}" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 ))
for arm in "${arms[@]}"; do
  [[ -f "${root}/runs/${arm}/TRAINING_COMPLETE" ]]
  [[ -f "${root}/runs/${arm}/checkpoint-00000040/receipt.json" ]]
done
printf 'training_complete=true\nparent_allocation_cancelled=false\n' >"${root}/TRAINING_COMPLETE"
