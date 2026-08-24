#!/usr/bin/env bash
# Extract four frozen-DINO shards across Job 135096's four retained nodes.
# Never cancels or releases the parent allocation.

set -Eeuo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 <job-id> <experiment-root> <python> <model-root>" >&2
  exit 64
fi
job_id=$1
root=$2
python_bin=$3
model_root=$4
[[ "$job_id" =~ ^[1-9][0-9]*$ && "$root" == /* && "$python_bin" == /* && "$model_root" == /* ]] \
  || { echo "invalid argument" >&2; exit 65; }

source="$root/software/run_reward_ablation_v1.py"
manifest="$root/manifest/reward-ablation-manifest.json"
[[ -x "$python_bin" && -f "$source" && ! -L "$source" && -f "$manifest" && ! -L "$manifest" && -d "$model_root" && ! -L "$model_root" ]] \
  || { echo "feature extraction input differs" >&2; exit 66; }

mkdir -p "$root/features" "$root/logs/features"
nodes=(auh7-1b-gpu-245 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248)
pids=()
for shard in 0 1 2 3; do
  node=${nodes[$shard]}
  runtime="/tmp/reward-ablation-${job_id}-shard${shard}"
  srun --overlap --jobid="$job_id" --nodes=1 --ntasks=1 --cpus-per-task=16 \
    --gres=gpu:mi210:1 --mem=128G --nodelist="$node" \
    env \
      HOME="$runtime/home" TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/xdg" \
      HF_HOME="$runtime/xdg/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
      OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
      MIOPEN_USER_DB_PATH="$runtime/miopen-user" \
      MIOPEN_CUSTOM_CACHE_DIR="$runtime/miopen-custom" \
      bash -lc "mkdir -p '$runtime/home' '$runtime/tmp' '$runtime/xdg' '$runtime/miopen-user' '$runtime/miopen-custom'; '$python_bin' '$source' extract-shard --manifest '$manifest' --model-root '$model_root' --shard-index '$shard' --num-shards 4 --num-frames 8 --frame-batch-size 8 --device cuda:0 --output '$root/features/features-shard-$shard.pt'" \
    >"$root/logs/features/shard-${shard}.log" 2>&1 &
  pids+=("$!")
  printf 'feature shard=%s node=%s pid=%s\n' "$shard" "$node" "$!"
done

status=0
for shard in 0 1 2 3; do
  if ! wait "${pids[$shard]}"; then
    status=1
    printf 'feature FAILED shard=%s node=%s\n' "$shard" "${nodes[$shard]}" >&2
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "feature extraction failed; parent allocation unchanged" >&2
  exit "$status"
fi
sha256sum "$root"/features/features-shard-*.pt >"$root/features/SHA256SUMS"
echo "all four feature shards completed; parent_allocation_cancelled=false parent_allocation_released=false"
