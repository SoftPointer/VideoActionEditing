#!/usr/bin/env bash
# Run five extra SP4 arms on the idle half-nodes of allocation 135096.
# Set CAPACITY_RUN_TAG/MAX_STEPS/SAVE_EVERY to use this for canary or formal runs.

set -Eeuo pipefail

job_id=135096
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_sha="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
launcher="${BERNINI_ACTION_NODE_LAUNCHER:?set BERNINI_ACTION_NODE_LAUNCHER}"
experiment_root="${BERNINI_ACTION_EXPERIMENT_ROOT:?set BERNINI_ACTION_EXPERIMENT_ROOT}"
run_tag="${CAPACITY_RUN_TAG:?set CAPACITY_RUN_TAG}"
max_steps="${CAPACITY_MAX_STEPS:?set CAPACITY_MAX_STEPS}"
save_every="${CAPACITY_SAVE_EVERY:?set CAPACITY_SAVE_EVERY}"
data_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1/data/action_only
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
bernini_archive=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591.tar.gz
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

nodes=(
  auh7-1b-gpu-245
  auh7-1b-gpu-245
  auh7-1b-gpu-246
  auh7-1b-gpu-247
  auh7-1b-gpu-248
)
gpu_groups=(0,1,2,3 4,5,6,7 4,5,6,7 4,5,6,7 4,5,6,7)
arms=(
  action_only_sft
  action_only_sft_preservation
  margin_noop_only
  margin_reverse_only
  margin_incomplete_only
)
objectives=(
  sft
  sft_preservation
  high_contrast_margin
  high_contrast_margin
  high_contrast_margin
)
negative_schedules=(rotate rotate noop reverse incomplete)

[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]]
[[ "${save_every}" =~ ^[0-9]+$ ]]
[[ "${run_tag}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]] || {
  echo "parent allocation preflight differs: ${state}" >&2
  exit 2
}
[[ -x "${launcher}" ]]
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_sha}" ]]
mkdir -p "${experiment_root}/logs" "${experiment_root}/${run_tag}"

for index in 0 1 2 3 4; do
  output="${experiment_root}/${run_tag}/${arms[$index]}-${run_tag}"
  [[ ! -e "${output}" && ! -L "${output}" ]] || {
    echo "create-only output exists: ${output}" >&2
    exit 2
  }
done

pids=()
for index in 0 1 2 3 4; do
  node="${nodes[$index]}"
  gpu_group="${gpu_groups[$index]}"
  arm="${arms[$index]}"
  objective="${objectives[$index]}"
  negative_schedule="${negative_schedules[$index]}"
  output="${experiment_root}/${run_tag}/${arm}-${run_tag}"
  log="${experiment_root}/logs/${run_tag}-${arm}.log"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${node}" --cpus-per-task=32 --mem=240G \
    --gres=gpu:mi210:8 \
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
      BERNINI_ACTION_OBJECTIVE="${objective}" \
      BERNINI_ACTION_NEGATIVE_SCHEDULE="${negative_schedule}" \
      BERNINI_ACTION_MAX_STEPS="${max_steps}" \
      BERNINI_ACTION_SAVE_EVERY="${save_every}" \
      BERNINI_ACTION_LR=1e-4 \
      BERNINI_ACTION_ALLOW_REWARD_SELECTED_TARGETS=1 \
      BERNINI_ACTION_PREFERENCE_WEIGHT=1.0 \
      BERNINI_ACTION_PREFERENCE_MARGIN=0.05 \
      BERNINI_ACTION_PREFERENCE_TEMPERATURE=20.0 \
      BERNINI_ACTION_DPO_BETA=10.0 \
      BERNINI_ACTION_PRESERVATION_WEIGHT=0.25 \
      BERNINI_ACTION_PYTHON="${python_bin}" \
      "${launcher}" >"${log}" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED arm=${arm} objective=${objective} negative=${negative_schedule} node=${node} physical_gpus=${gpu_group} pid=${pids[-1]}"
done

overall=0
for index in 0 1 2 3 4; do
  if wait "${pids[$index]}"; then
    echo "COMPLETED arm=${arms[$index]} node=${nodes[$index]} physical_gpus=${gpu_groups[$index]}"
  else
    status=$?
    overall=1
    echo "FAILED arm=${arms[$index]} node=${nodes[$index]} status=${status}" >&2
  fi
done
(( overall == 0 ))
echo "ALL_COMPLETE run_tag=${run_tag} parent_allocation_cancelled=false"
