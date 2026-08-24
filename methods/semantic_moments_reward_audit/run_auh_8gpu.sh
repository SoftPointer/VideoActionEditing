#!/usr/bin/env bash
# Extract one frozen-DINO SemanticMoments shard per GPU inside an existing AUH allocation.

set -euo pipefail

if [[ "$#" -lt 8 || "$#" -gt 9 ]]; then
  echo "usage: $0 <allocation-job-id> <node> <python> <run_audit.py> <manifest> <semantic-moments-root> <dinov2-root> <experiment-root> [num-frames]" >&2
  exit 64
fi

allocation_job_id="$1"
node="$2"
python_bin="$3"
audit_source="$4"
manifest="$5"
semantic_moments_root="$6"
model_root="$7"
experiment_root="$8"
num_frames="${9:-32}"

if [[ ! "$allocation_job_id" =~ ^[1-9][0-9]*$ || ! "$node" =~ ^auh[0-9A-Za-z._-]+$ ]]; then
  echo "invalid allocation or node identity" >&2
  exit 65
fi
if [[ ! "$num_frames" =~ ^[1-9][0-9]*$ || "$num_frames" -lt 2 ]]; then
  echo "num-frames must be an integer of at least two" >&2
  exit 65
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is absent: $python_bin" >&2
  exit 66
fi
for path in "$audit_source" "$manifest"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "required plain file is absent or linked: $path" >&2
    exit 66
  fi
done
for path in "$semantic_moments_root" "$model_root" "$experiment_root"; do
  if [[ ! -d "$path" || -L "$path" ]]; then
    echo "required directory is absent or linked: $path" >&2
    exit 66
  fi
done

feature_root="$experiment_root/results/features"
log_root="$experiment_root/logs/extract"
node_runtime_root="/tmp/semantic-moments-${allocation_job_id}"
mkdir -p "$feature_root" "$log_root"

# ROCm/COMGR and MIOpen create and atomically rename compiled objects.  AUH's
# shared VAST filesystem is not a safe TMPDIR for that workload, so prepare a
# private cache on the allocated node before launching the GPU workers.
srun --overlap --jobid="$allocation_job_id" --nodes=1 --ntasks=1 \
  --cpus-per-task=1 -w "$node" \
  mkdir -p \
    "$node_runtime_root/gpu0/home" "$node_runtime_root/gpu0/tmp" "$node_runtime_root/gpu0/xdg" "$node_runtime_root/gpu0/miopen-user" "$node_runtime_root/gpu0/miopen-custom" \
    "$node_runtime_root/gpu1/home" "$node_runtime_root/gpu1/tmp" "$node_runtime_root/gpu1/xdg" "$node_runtime_root/gpu1/miopen-user" "$node_runtime_root/gpu1/miopen-custom" \
    "$node_runtime_root/gpu2/home" "$node_runtime_root/gpu2/tmp" "$node_runtime_root/gpu2/xdg" "$node_runtime_root/gpu2/miopen-user" "$node_runtime_root/gpu2/miopen-custom" \
    "$node_runtime_root/gpu3/home" "$node_runtime_root/gpu3/tmp" "$node_runtime_root/gpu3/xdg" "$node_runtime_root/gpu3/miopen-user" "$node_runtime_root/gpu3/miopen-custom" \
    "$node_runtime_root/gpu4/home" "$node_runtime_root/gpu4/tmp" "$node_runtime_root/gpu4/xdg" "$node_runtime_root/gpu4/miopen-user" "$node_runtime_root/gpu4/miopen-custom" \
    "$node_runtime_root/gpu5/home" "$node_runtime_root/gpu5/tmp" "$node_runtime_root/gpu5/xdg" "$node_runtime_root/gpu5/miopen-user" "$node_runtime_root/gpu5/miopen-custom" \
    "$node_runtime_root/gpu6/home" "$node_runtime_root/gpu6/tmp" "$node_runtime_root/gpu6/xdg" "$node_runtime_root/gpu6/miopen-user" "$node_runtime_root/gpu6/miopen-custom" \
    "$node_runtime_root/gpu7/home" "$node_runtime_root/gpu7/tmp" "$node_runtime_root/gpu7/xdg" "$node_runtime_root/gpu7/miopen-user" "$node_runtime_root/gpu7/miopen-custom"

pids=()
for logical_gpu in 0 1 2 3 4 5 6 7; do
  runtime_cache="$node_runtime_root/gpu${logical_gpu}"
  srun --overlap --jobid="$allocation_job_id" --nodes=1 --ntasks=1 \
    --cpus-per-task=8 -w "$node" \
    env \
      HOME="$runtime_cache/home" \
      TMPDIR="$runtime_cache/tmp" \
      XDG_CACHE_HOME="$runtime_cache/xdg" \
      HF_HOME="$runtime_cache/xdg/huggingface" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONHASHSEED=0 \
      OMP_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 \
      MIOPEN_USER_DB_PATH="$runtime_cache/miopen-user" \
      MIOPEN_CUSTOM_CACHE_DIR="$runtime_cache/miopen-custom" \
      ROCR_VISIBLE_DEVICES="$logical_gpu" \
      "$python_bin" "$audit_source" extract-shard \
        --manifest "$manifest" \
        --semantic-moments-root "$semantic_moments_root" \
        --model-root "$model_root" \
        --shard-index "$logical_gpu" \
        --num-shards 8 \
        --num-frames "$num_frames" \
        --frame-batch-size 8 \
        --device cuda:0 \
        --output "$feature_root/features-shard-${logical_gpu}.pt" \
    >"$log_root/shard-${logical_gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "one or more extraction shards failed; allocation was not cancelled" >&2
  exit "$status"
fi

sha256sum "$feature_root"/features-shard-*.pt
echo "all eight SemanticMoments extraction shards completed"
