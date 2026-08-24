#!/usr/bin/env bash
set -euo pipefail

[ "$(hostname -s)" = auh7-1b-gpu-247 ] || {
  echo "real-source probe flow extraction is restricted to node 247" >&2
  exit 3
}
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-crossappearance-motion-v1"
release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
source=/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/186d36f998a14e14/source.mp4
event_root="$stage/interaction_complex8_multianchor_v2_r1/e01_reach-grasp-lift-stone"
mkdir -p "$release/flows_real_event01" "$release/logs"

run_one() {
  local gpu="$1" variant="$2"
  local output="$release/flows_real_event01/${variant}.safetensors"
  local scratch="/tmp/complex8-real-event01-${variant}-g${gpu}-140846"
  mkdir -p "$scratch/cache" "$scratch/miopen-user" "$scratch/miopen-custom"
  export ROCR_VISIBLE_DEVICES="$gpu" TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
  export XDG_CACHE_HOME="$scratch/cache" MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  exec "$python_bin" -B "$source_tree/methods/bernini_action_editing/extract_anchor_raft_flow_v1.py" \
    --source "$source" --anchor "$event_root/$variant/t2v.mp4" --output "$output"
}

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run_one 0 v0 >"$release/logs/real-event01-v0-flow.log" 2>&1 & p0=$!
run_one 1 v1 >"$release/logs/real-event01-v1-flow.log" 2>&1 & p1=$!
wait "$p0"
wait "$p1"
touch "$release/REAL_EVENT01_FLOW_COMPLETE"
