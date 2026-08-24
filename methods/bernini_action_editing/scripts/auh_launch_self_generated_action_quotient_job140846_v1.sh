#!/usr/bin/env bash
# Canary, shared detached-teacher cache, then eight SP4 arms on all 32 GPUs.
# The parent allocation is intentionally never cancelled/released/requeued/signalled.

set -Eeuo pipefail

job_id=140846
root="${ACTION_QUOTIENT_EXPERIMENT_ROOT:?set ACTION_QUOTIENT_EXPERIMENT_ROOT}"
archive="${ACTION_QUOTIENT_SOURCE_ARCHIVE:?set ACTION_QUOTIENT_SOURCE_ARCHIVE}"
archive_sha="${ACTION_QUOTIENT_SOURCE_ARCHIVE_SHA256:?set ACTION_QUOTIENT_SOURCE_ARCHIVE_SHA256}"
revision="${ACTION_QUOTIENT_SOURCE_REVISION:?set ACTION_QUOTIENT_SOURCE_REVISION}"
runner="${ACTION_QUOTIENT_NODE_RUNNER:?set ACTION_QUOTIENT_NODE_RUNNER}"
manifest="${root}/source_only/manifest.json"
cache="${root}/teacher-cache-row4-slot4.pt"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
parquet=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1/data/action_only
anchors=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_t2v_core4_v2_20260808_17cc2c7/runs/core4_bank_v2

state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
[[ -x "${runner}" && -f "${archive}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ ! -e "${root}" ]]
mkdir -p "${root}/logs" "${root}/runs"

scratch_extract="$(mktemp -d)"
tar -xf "${archive}" -C "${scratch_extract}"
"${python_bin}" -B "${scratch_extract}/methods/bernini_action_editing/materialize_self_generated_action_quotient_v1.py" \
  --parquet-dir "${parquet}" --anchor-root "${anchors}" --output "${root}/source_only"

common_env=(
  ACTION_QUOTIENT_SOURCE_ARCHIVE="${archive}"
  ACTION_QUOTIENT_SOURCE_ARCHIVE_SHA256="${archive_sha}"
  ACTION_QUOTIENT_SOURCE_REVISION="${revision}"
  ACTION_QUOTIENT_SOURCE_MANIFEST="${manifest}"
  ACTION_QUOTIENT_SLOTS=4
)

# One real frozen-model cell proves source/teacher packing before the full cache.
srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-246 \
  --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
  "${common_env[@]}" ACTION_QUOTIENT_MODE=cache ACTION_QUOTIENT_LIMIT_CELLS=1 \
  ACTION_QUOTIENT_CACHE="${root}/canary-cache-unused.pt" \
  ACTION_QUOTIENT_OUTPUT="${root}/canary-cache-one-cell.pt" \
  "${runner}" >"${root}/logs/cache-canary.log" 2>&1
[[ -f "${root}/canary-cache-one-cell.pt" ]]

# The detached base-model codes are shared by every arm, avoiding eight copies
# of the same 96 frozen forwards.
srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-246 \
  --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
  "${common_env[@]}" ACTION_QUOTIENT_MODE=cache ACTION_QUOTIENT_LIMIT_CELLS=0 \
  ACTION_QUOTIENT_CACHE="${root}/full-cache-unused.pt" ACTION_QUOTIENT_OUTPUT="${cache}" \
  "${runner}" >"${root}/logs/cache-full.log" 2>&1
[[ -f "${cache}" ]]

nodes=(
  auh7-1b-gpu-246 auh7-1b-gpu-246
  auh7-1b-gpu-247 auh7-1b-gpu-247
  auh7-1b-gpu-248 auh7-1b-gpu-248
  auh7-1b-gpu-279 auh7-1b-gpu-279
)
groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)
arms=(
  action_only action_only_lowlr action_noop action_start
  action_nuisance action_start_nuisance
  action_start_nuisance_noop action_start_nuisance_border
)
pids=()
for index in 0 1 2 3 4 5 6 7; do
  arm="${arms[$index]}"; output="${root}/runs/${arm}"
  [[ ! -e "${output}" ]]
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[$index]}" \
    --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 \
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ROCR_VISIBLE_DEVICES="${groups[$index]}" "${common_env[@]}" \
    ACTION_QUOTIENT_MODE=train ACTION_QUOTIENT_CACHE="${cache}" \
    ACTION_QUOTIENT_OUTPUT="${output}" ACTION_QUOTIENT_ARM="${arm}" \
    ACTION_QUOTIENT_MAX_STEPS=160 "${runner}" >"${root}/logs/train-${arm}.log" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED arm=${arm} node=${nodes[$index]} gpus=${groups[$index]} pid=${pids[-1]}"
done

status=0
for index in 0 1 2 3 4 5 6 7; do
  if wait "${pids[$index]}"; then
    echo "COMPLETE arm=${arms[$index]}"
  else
    echo "FAILED arm=${arms[$index]}" >&2
    status=1
  fi
done
(( status == 0 ))
echo "ALL_TRAINING_COMPLETE parent_allocation_cancelled=false"
