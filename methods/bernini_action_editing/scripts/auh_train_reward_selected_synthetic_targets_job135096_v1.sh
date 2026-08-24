#!/usr/bin/env bash
# Four matched reward-selected synthetic-target training arms inside allocation 135096.
# Safety: this script never calls scancel, scontrol release/requeue, or sends a signal.

set -Eeuo pipefail

job_id=135096
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1
source_archive="${root}/release/source-76e2858.tar"
source_sha=10d09913d20e7cc730ddacb4bf8bddbf714583de2a11327729d2051d36add84c
source_revision=76e28587bfe29d019dc8e439475d8c11b432ccdd
launcher="${root}/release/auh_train_lora_76e2858.sh"
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
bernini_archive=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591.tar.gz
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

arms=(baseline action_only preservation_only composite)
nodes=(auh7-1b-gpu-245 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248)

state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]] || {
  echo "parent allocation preflight differs: ${state}" >&2
  exit 2
}
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_sha}" ]]
[[ -x "${launcher}" ]]

pids=()
for index in 0 1 2 3; do
  arm="${arms[$index]}"
  node="${nodes[$index]}"
  output="${root}/runs/${arm}-u40"
  log="${root}/logs/train-${arm}-u40.log"
  [[ ! -e "${output}" && ! -L "${output}" ]] || {
    echo "create-only output exists: ${output}" >&2
    exit 2
  }
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
      BERNINI_ACTION_PARQUET_DIR="${root}/data/${arm}/shards" \
      BERNINI_ACTION_DATASET_SUMMARY="${root}/data/${arm}/dataset_summary.json" \
      BERNINI_ACTION_TRAIN_OUTPUT="${output}" \
      BERNINI_ACTION_MAX_STEPS=40 \
      BERNINI_ACTION_SAVE_EVERY=20 \
      BERNINI_ACTION_LR=1e-4 \
      BERNINI_ACTION_ALLOW_REWARD_SELECTED_TARGETS=1 \
      BERNINI_ACTION_PYTHON="${python_bin}" \
      "${launcher}" >"${log}" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED arm=${arm} node=${node} controller_pid=${pids[-1]}"
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
  echo FAILED >"${root}/runs/TRAINING_FAILED"
  exit 1
fi
echo COMPLETE >"${root}/runs/TRAINING_COMPLETE"
echo "ALL_COMPLETE parent_allocation_cancelled=false parent_allocation_released=false"
