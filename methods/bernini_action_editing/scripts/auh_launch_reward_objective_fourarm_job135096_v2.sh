#!/usr/bin/env bash
# Launch four matched 320-update reward-objective arms inside allocation 135096.
# Safety: never cancel, signal, release, requeue, or otherwise mutate the parent job.

set -Eeuo pipefail

job_id=135096
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_sha="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
launcher="${BERNINI_ACTION_NODE_LAUNCHER:?set BERNINI_ACTION_NODE_LAUNCHER}"
experiment_root="${BERNINI_ACTION_EXPERIMENT_ROOT:?set BERNINI_ACTION_EXPERIMENT_ROOT}"
data_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1/data
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
bernini_archive=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591.tar.gz
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

nodes=(auh7-1b-gpu-245 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248)
arms=(long_sft_control high_contrast_margin reference_dpo reference_dpo_preservation)
objectives=(sft high_contrast_margin reference_dpo reference_dpo_preservation)
datasets=(baseline action_only action_only action_only)
allow_reward=(1 1 1 1)

state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]] || {
  echo "parent allocation preflight differs: ${state}" >&2
  exit 2
}
[[ -x "${launcher}" ]]
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_sha}" ]]
mkdir -p "${experiment_root}/logs" "${experiment_root}/runs"

for index in 0 1 2 3; do
  output="${experiment_root}/runs/${arms[$index]}-u320"
  [[ ! -e "${output}" && ! -L "${output}" ]] || {
    echo "create-only output exists: ${output}" >&2
    exit 2
  }
done

pids=()
for index in 0 1 2 3; do
  arm="${arms[$index]}"
  node="${nodes[$index]}"
  dataset="${datasets[$index]}"
  output="${experiment_root}/runs/${arm}-u320"
  log="${experiment_root}/logs/train-${arm}-u320.log"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${node}" --cpus-per-task=64 --mem=500G \
    --gres=gpu:mi210:8 \
    env \
      BERNINI_ACTION_SOURCE_ARCHIVE="${source_archive}" \
      BERNINI_ACTION_SOURCE_ARCHIVE_SHA256="${source_sha}" \
      BERNINI_ACTION_SOURCE_REVISION="${source_revision}" \
      BERNINI_OFFICIAL_ROOT="${bernini_root}" \
      BERNINI_OFFICIAL_ARCHIVE="${bernini_archive}" \
      BERNINI_VEOMNI_ROOT="${veomni_root}" \
      BERNINI_ACTION_CHECKPOINT="${checkpoint}" \
      BERNINI_ACTION_PARQUET_DIR="${data_root}/${dataset}/shards" \
      BERNINI_ACTION_DATASET_SUMMARY="${data_root}/${dataset}/dataset_summary.json" \
      BERNINI_ACTION_TRAIN_OUTPUT="${output}" \
      BERNINI_ACTION_OBJECTIVE="${objectives[$index]}" \
      BERNINI_ACTION_MAX_STEPS=320 \
      BERNINI_ACTION_SAVE_EVERY=80 \
      BERNINI_ACTION_LR=1e-4 \
      BERNINI_ACTION_ALLOW_REWARD_SELECTED_TARGETS="${allow_reward[$index]}" \
      BERNINI_ACTION_PREFERENCE_WEIGHT=1.0 \
      BERNINI_ACTION_PREFERENCE_MARGIN=0.05 \
      BERNINI_ACTION_PREFERENCE_TEMPERATURE=20.0 \
      BERNINI_ACTION_DPO_BETA=10.0 \
      BERNINI_ACTION_PRESERVATION_WEIGHT=0.25 \
      BERNINI_ACTION_PYTHON="${python_bin}" \
      "${launcher}" >"${log}" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED arm=${arm} objective=${objectives[$index]} node=${node} controller_pid=${pids[-1]}"
done

overall=0
for index in 0 1 2 3; do
  if wait "${pids[$index]}"; then
    echo "COMPLETED arm=${arms[$index]} node=${nodes[$index]}"
  else
    status=$?
    overall=1
    echo "FAILED arm=${arms[$index]} node=${nodes[$index]} status=${status}" >&2
  fi
done

if (( overall != 0 )); then
  echo FAILED >"${experiment_root}/runs/TRAINING_FAILED"
  exit 1
fi
echo COMPLETE >"${experiment_root}/runs/TRAINING_COMPLETE"
echo "ALL_COMPLETE parent_allocation_cancelled=false parent_allocation_released=false"
