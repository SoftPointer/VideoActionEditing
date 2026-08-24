#!/usr/bin/env bash
# Eight-arm anti-reward-hacking objective screen on all 32 GPUs of Job 135096.

set -Eeuo pipefail

job_id=135096
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_sha="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
launcher="${BERNINI_ACTION_NODE_LAUNCHER:?set BERNINI_ACTION_NODE_LAUNCHER}"
experiment_root="${BERNINI_ACTION_EXPERIMENT_ROOT:?set BERNINI_ACTION_EXPERIMENT_ROOT}"
run_tag="${ANTIHACK_RUN_TAG:?set ANTIHACK_RUN_TAG}"
max_steps="${ANTIHACK_MAX_STEPS:?set ANTIHACK_MAX_STEPS}"
save_every="${ANTIHACK_SAVE_EVERY:?set ANTIHACK_SAVE_EVERY}"
data_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1/data/action_only
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
bernini_archive=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591.tar.gz
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

# Pair different objective implementations on each node so their node-local
# scratch/cache paths never alias.
nodes=(
  auh7-1b-gpu-245 auh7-1b-gpu-245
  auh7-1b-gpu-246 auh7-1b-gpu-246
  auh7-1b-gpu-247 auh7-1b-gpu-247
  auh7-1b-gpu-248 auh7-1b-gpu-248
)
gpu_groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)
arms=(
  detached_rotate_w1
  ref_hardmix_w025_b3
  detached_hardmix_w1
  ref_hardmix_w05_b3
  detached_noop_w1
  detached_hardmix_pres005
  detached_incomplete_w1
  detached_hardmix_pres010
)
objectives=(
  detached_margin reference_dpo
  detached_margin reference_dpo
  detached_margin detached_margin_preservation
  detached_margin detached_margin_preservation
)
negative_schedules=(
  rotate noop_incomplete
  noop_incomplete noop_incomplete
  noop noop_incomplete
  incomplete noop_incomplete
)
preference_weights=(1.0 0.25 1.0 0.5 1.0 1.0 1.0 1.0)
dpo_betas=(10.0 3.0 10.0 3.0 10.0 10.0 10.0 10.0)
preservation_weights=(0.25 0.25 0.25 0.25 0.25 0.05 0.25 0.10)

[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]]
[[ "${save_every}" =~ ^[0-9]+$ ]]
[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]]
[[ -x "${launcher}" ]]
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_sha}" ]]
mkdir -p "${experiment_root}/logs" "${experiment_root}/${run_tag}"

for index in 0 1 2 3 4 5 6 7; do
  output="${experiment_root}/${run_tag}/${arms[$index]}-${run_tag}"
  [[ ! -e "${output}" && ! -L "${output}" ]]
done

pids=()
for index in 0 1 2 3 4 5 6 7; do
  node="${nodes[$index]}"; gpu_group="${gpu_groups[$index]}"; arm="${arms[$index]}"
  output="${experiment_root}/${run_tag}/${arm}-${run_tag}"
  log="${experiment_root}/logs/${run_tag}-${arm}.log"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${node}" --cpus-per-task=32 --mem=240G --gres=gpu:mi210:8 \
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
      ROCR_VISIBLE_DEVICES="${gpu_group}" \
      BERNINI_ACTION_SOURCE_ARCHIVE="${source_archive}" \
      BERNINI_ACTION_SOURCE_ARCHIVE_SHA256="${source_sha}" \
      BERNINI_ACTION_SOURCE_REVISION="${source_revision}" \
      BERNINI_OFFICIAL_ROOT="${bernini_root}" \
      BERNINI_OFFICIAL_ARCHIVE="${bernini_archive}" \
      BERNINI_VEOMNI_ROOT="${veomni_root}" \
      BERNINI_ACTION_CHECKPOINT="${checkpoint}" \
      BERNINI_ACTION_PARQUET_DIR="${data_root}/shards" \
      BERNINI_ACTION_DATASET_SUMMARY="${data_root}/dataset_summary.json" \
      BERNINI_ACTION_TRAIN_OUTPUT="${output}" \
      BERNINI_ACTION_OBJECTIVE="${objectives[$index]}" \
      BERNINI_ACTION_NEGATIVE_SCHEDULE="${negative_schedules[$index]}" \
      BERNINI_ACTION_MAX_STEPS="${max_steps}" \
      BERNINI_ACTION_SAVE_EVERY="${save_every}" \
      BERNINI_ACTION_LR=1e-4 \
      BERNINI_ACTION_ALLOW_REWARD_SELECTED_TARGETS=1 \
      BERNINI_ACTION_PREFERENCE_WEIGHT="${preference_weights[$index]}" \
      BERNINI_ACTION_PREFERENCE_MARGIN=0.05 \
      BERNINI_ACTION_PREFERENCE_TEMPERATURE=20.0 \
      BERNINI_ACTION_DPO_BETA="${dpo_betas[$index]}" \
      BERNINI_ACTION_PRESERVATION_WEIGHT="${preservation_weights[$index]}" \
      BERNINI_ACTION_PYTHON="${python_bin}" \
      "${launcher}" >"${log}" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED arm=${arm} objective=${objectives[$index]} negative=${negative_schedules[$index]} node=${node} physical_gpus=${gpu_group} pid=${pids[-1]}"
done

overall=0
for index in 0 1 2 3 4 5 6 7; do
  if wait "${pids[$index]}"; then
    echo "COMPLETED arm=${arms[$index]} node=${nodes[$index]} physical_gpus=${gpu_groups[$index]}"
  else
    overall=1
    echo "FAILED arm=${arms[$index]} node=${nodes[$index]}" >&2
  fi
done
(( overall == 0 ))
echo "ALL_COMPLETE run_tag=${run_tag} parent_allocation_cancelled=false"
