#!/usr/bin/env bash
set -euo pipefail

node="$(hostname -s)"
case "$node" in
  auh7-1b-gpu-246) node_half=0 ;;
  auh7-1b-gpu-247) node_half=1 ;;
  *) echo "SGA/ANC flow extraction is restricted to nodes 246/247" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
source_tree="$stage/source-sga-anc-training-v1"
anchor_root="$stage/interaction_complex8_multianchor_v2_r1"
release="$stage/complex8_sga_anc_training_v1"
old_release="$stage/complex8_crossappearance_motion_v1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
events=(
  pour-liquid-into-cup
  reach-grasp-lift-stone
  twist-pull-mushroom
  release-harvest-into-basket
  close-door-then-drawer
  jet-ski-turn-with-wake
  tap-plant-and-rebound
  players-contact-then-separate
)
latent_heights=(74 72 74 70 72 82 82 50)
latent_widths=(50 52 50 52 52 44 44 74)
mkdir -p "$release/flows" "$release/logs"

run_gpu() {
  local gpu="$1" global event variant offset donor iid event_root source output scratch
  for global in $(seq 0 31); do
    [ $((global / 16)) -eq "$node_half" ] || continue
    [ $((global % 8)) -eq "$gpu" ] || continue
    event=$((global / 4))
    variant=$((global % 4))
    iid=$(printf 'e%02d-v%d' "$event" "$variant")
    event_root=$(printf '%s/e%02d_%s' "$anchor_root" "$event" "${events[$event]}")
    source="$event_root/v${variant}/t2v.mp4"
    test -f "$source"
    for offset in 1 2 3; do
      donor=$(((variant + offset) % 4))
      output="$release/flows/${iid}__anchor-v${donor}.safetensors"
      if [ -f "$output" ]; then
        continue
      fi
      if [ "$offset" -eq 1 ]; then
        cp "$old_release/flows_cross_exact/${iid}.safetensors" "$output"
        continue
      fi
      scratch="/tmp/complex8-sgaanc-${iid}-v${donor}-g${gpu}-${SLURM_JOB_ID:-none}"
      mkdir -p "$scratch/cache" "$scratch/miopen-user" "$scratch/miopen-custom"
      export ROCR_VISIBLE_DEVICES="$gpu" TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
      export XDG_CACHE_HOME="$scratch/cache" MIOPEN_USER_DB_PATH="$scratch/miopen-user"
      export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
      unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
      "$python_bin" -B "$source_tree/methods/bernini_action_editing/extract_anchor_raft_flow_v1.py" \
        --source "$source" \
        --anchor "$event_root/v${donor}/t2v.mp4" \
        --output "$output" \
        --latent-height "${latent_heights[$event]}" \
        --latent-width "${latent_widths[$event]}"
    done
  done
}

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_gpu "$gpu" >"$release/logs/${node}-g${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
test "$(find "$release/flows" -name '*.safetensors' | wc -l)" -ge 48
touch "$release/EXTRACT_${node}_COMPLETE"

