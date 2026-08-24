#!/usr/bin/env bash
# Repair launch for the node245 control after the explicit dataset flag blocked r1.

set -Eeuo pipefail

job_id=135096
experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_objective_fourarm_20260815_v2
source_archive="${experiment_root}/release/reward-objective-v2-802a930.tar"
source_sha=5587a47efc7273cb98d8804f1d710001ba85b245c83eaff90b05d84833b950f5
source_revision=802a9300c9594841941a501fa59494e6ec393e15
launcher="${experiment_root}/release/auh_train_reward_objective_v2.sh"
data_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1/data/baseline
output="${experiment_root}/runs/long_sft_control-u320-r2"
log="${experiment_root}/logs/train-long_sft_control-u320-r2.log"

state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]]
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_sha}" ]]
[[ -x "${launcher}" ]]
[[ ! -e "${output}" && ! -L "${output}" ]]

srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
  --nodelist=auh7-1b-gpu-245 --cpus-per-task=64 --mem=500G \
  --gres=gpu:mi210:8 \
  env \
    BERNINI_ACTION_SOURCE_ARCHIVE="${source_archive}" \
    BERNINI_ACTION_SOURCE_ARCHIVE_SHA256="${source_sha}" \
    BERNINI_ACTION_SOURCE_REVISION="${source_revision}" \
    BERNINI_OFFICIAL_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591 \
    BERNINI_OFFICIAL_ARCHIVE=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591.tar.gz \
    BERNINI_VEOMNI_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11 \
    BERNINI_ACTION_CHECKPOINT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4 \
    BERNINI_ACTION_PARQUET_DIR="${data_root}/shards" \
    BERNINI_ACTION_DATASET_SUMMARY="${data_root}/dataset_summary.json" \
    BERNINI_ACTION_TRAIN_OUTPUT="${output}" \
    BERNINI_ACTION_OBJECTIVE=sft \
    BERNINI_ACTION_MAX_STEPS=320 \
    BERNINI_ACTION_SAVE_EVERY=80 \
    BERNINI_ACTION_LR=1e-4 \
    BERNINI_ACTION_ALLOW_REWARD_SELECTED_TARGETS=1 \
    BERNINI_ACTION_PYTHON=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 \
    "${launcher}" >"${log}" 2>&1

echo "CONTROL_COMPLETE output=${output} parent_allocation_cancelled=false"
