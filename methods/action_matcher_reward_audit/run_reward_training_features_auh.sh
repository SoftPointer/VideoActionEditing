#!/usr/bin/env bash
# Extract frozen-DINO diagnostics for the five post-training model outputs.
# The retained allocation is never cancelled, released, requeued, or signalled.

set -Eeuo pipefail

job_id=135096
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
model_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8
source="${root}/software/run_reward_ablation_v1.py"
manifest="${root}/manifest/reward-ablation-manifest.json"
nodes=(auh7-1b-gpu-245 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248)

[[ -f "${root}/eval/EVALUATION_COMPLETE" ]]
[[ -x "${python_bin}" && -f "${source}" && -f "${manifest}" && -d "${model_root}" ]]
mkdir -p "${root}/features" "${root}/logs/features"

pids=()
for shard in 0 1 2 3; do
  node="${nodes[$shard]}"
  runtime="/tmp/reward-training-features-${job_id}-shard${shard}"
  srun --overlap --jobid="${job_id}" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:mi210:1 --mem=128G --nodelist="${node}" \
    env \
      TMPDIR="${runtime}/tmp" XDG_CACHE_HOME="${runtime}/xdg" \
      HF_HOME="${runtime}/xdg/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
      OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
      MIOPEN_USER_DB_PATH="${runtime}/miopen-user" \
      MIOPEN_CUSTOM_CACHE_DIR="${runtime}/miopen-custom" \
      bash -lc "mkdir -p '${runtime}/tmp' '${runtime}/xdg' '${runtime}/miopen-user' '${runtime}/miopen-custom'; '${python_bin}' '${source}' extract-shard --manifest '${manifest}' --model-root '${model_root}' --shard-index '${shard}' --num-shards 4 --num-frames 8 --frame-batch-size 8 --device cuda:0 --output '${root}/features/features-shard-${shard}.pt'" \
    >"${root}/logs/features/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for shard in 0 1 2 3; do
  wait "${pids[$shard]}" || status=1
done
(( status == 0 )) || exit 1
sha256sum "${root}"/features/features-shard-*.pt >"${root}/features/SHA256SUMS"
echo COMPLETE >"${root}/features/FEATURES_COMPLETE"
echo "PASS parent_allocation_cancelled=false parent_allocation_released=false"
