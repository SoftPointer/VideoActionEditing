#!/usr/bin/env bash
# Extract TEAM ViT features on all GPUs inside an existing AUH allocation.

set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: $0 <job-id> <node> <python> <runner> <manifest> <team-root> <checkpoint> <experiment-root>" >&2
  exit 64
fi

job_id="$1"
node="$2"
python_bin="$(readlink -f "$3")"
runner="$4"
manifest="$5"
team_root="$6"
checkpoint="$7"
experiment_root="$8"

if [[ ! "$job_id" =~ ^[1-9][0-9]*$ || ! "$node" =~ ^auh[0-9A-Za-z._-]+$ ]]; then
  echo "invalid allocation or node" >&2
  exit 65
fi
for path in "$python_bin" "$runner" "$manifest" "$checkpoint"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "required plain file missing or linked: $path" >&2
    exit 66
  fi
done
if [[ ! -d "$team_root" || -L "$team_root" ]]; then
  echo "TEAM root missing or linked: $team_root" >&2
  exit 66
fi

feature_root="$experiment_root/results/team-features"
log_root="$experiment_root/logs/team-extract"
node_root="/tmp/team-action-matcher-${job_id}"
mkdir -p "$feature_root" "$log_root"
srun --overlap --jobid="$job_id" -N1 -n1 -w "$node" mkdir -p \
  "$node_root/gpu0/home" "$node_root/gpu0/tmp" "$node_root/gpu0/xdg" "$node_root/gpu0/torch" \
  "$node_root/gpu1/home" "$node_root/gpu1/tmp" "$node_root/gpu1/xdg" "$node_root/gpu1/torch" \
  "$node_root/gpu2/home" "$node_root/gpu2/tmp" "$node_root/gpu2/xdg" "$node_root/gpu2/torch" \
  "$node_root/gpu3/home" "$node_root/gpu3/tmp" "$node_root/gpu3/xdg" "$node_root/gpu3/torch" \
  "$node_root/gpu4/home" "$node_root/gpu4/tmp" "$node_root/gpu4/xdg" "$node_root/gpu4/torch" \
  "$node_root/gpu5/home" "$node_root/gpu5/tmp" "$node_root/gpu5/xdg" "$node_root/gpu5/torch" \
  "$node_root/gpu6/home" "$node_root/gpu6/tmp" "$node_root/gpu6/xdg" "$node_root/gpu6/torch" \
  "$node_root/gpu7/home" "$node_root/gpu7/tmp" "$node_root/gpu7/xdg" "$node_root/gpu7/torch"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  cache="$node_root/gpu${gpu}"
  srun --overlap --jobid="$job_id" -N1 -n1 --cpus-per-task=8 -w "$node" \
    env \
      HOME="$cache/home" \
      TMPDIR="$cache/tmp" \
      XDG_CACHE_HOME="$cache/xdg" \
      HF_HOME="$cache/xdg/huggingface" \
      HF_HUB_OFFLINE=1 \
      TORCH_HOME="$cache/torch" \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      OMP_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 \
      ROCR_VISIBLE_DEVICES="$gpu" \
      "$python_bin" "$runner" extract-shard \
        --manifest "$manifest" \
        --team-root "$team_root" \
        --checkpoint "$checkpoint" \
        --shard-index "$gpu" \
        --num-shards 8 \
        --device cuda:0 \
        --output "$feature_root/team-features-shard-${gpu}.pt" \
    >"$log_root/shard-${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more TEAM feature shards failed; allocation was not cancelled" >&2
  exit "$status"
fi
sha256sum "$feature_root"/team-features-shard-*.pt
echo "all eight TEAM feature shards completed"
