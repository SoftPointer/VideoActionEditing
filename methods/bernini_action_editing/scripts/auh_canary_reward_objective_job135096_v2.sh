#!/usr/bin/env bash
# One-update canaries for the three non-SFT reward objectives in allocation 135096.

set -Eeuo pipefail

job_id=135096
experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_objective_fourarm_20260815_v2
source_archive="${experiment_root}/release/reward-objective-v2-802a930.tar"
source_sha=5587a47efc7273cb98d8804f1d710001ba85b245c83eaff90b05d84833b950f5
source_revision=802a9300c9594841941a501fa59494e6ec393e15
launcher="${experiment_root}/release/auh_train_reward_objective_v2.sh"
data_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1/data/action_only
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
bernini_archive=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591.tar.gz
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248)
objectives=(high_contrast_margin reference_dpo reference_dpo_preservation)

state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]]
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_sha}" ]]
[[ -x "${launcher}" ]]

for objective in "${objectives[@]}"; do
  output="${experiment_root}/canary/${objective}-u1-r2"
  [[ ! -e "${output}" && ! -L "${output}" ]]
done

pids=()
for index in 0 1 2; do
  node="${nodes[$index]}"
  objective="${objectives[$index]}"
  output="${experiment_root}/canary/${objective}-u1-r2"
  log="${experiment_root}/logs/canary-${objective}-u1-r2.log"
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
      BERNINI_ACTION_PARQUET_DIR="${data_root}/shards" \
      BERNINI_ACTION_DATASET_SUMMARY="${data_root}/dataset_summary.json" \
      BERNINI_ACTION_TRAIN_OUTPUT="${output}" \
      BERNINI_ACTION_OBJECTIVE="${objective}" \
      BERNINI_ACTION_MAX_STEPS=1 \
      BERNINI_ACTION_SAVE_EVERY=1 \
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
  echo "CANARY_LAUNCHED objective=${objective} node=${node} pid=${pids[-1]}"
done

for index in 0 1 2; do
  wait "${pids[$index]}"
  echo "CANARY_PASS objective=${objectives[$index]} node=${nodes[$index]}"
done
