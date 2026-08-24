#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in
  0) current_event=1; remaining_events=(2 3) ;;
  1) current_event=5; remaining_events=(6 7) ;;
  *) exit 2 ;;
esac
node="$(hostname -s)"
case "$node" in
  auh7-1b-gpu-246) arm=dpo_only_s4 ;;
  auh7-1b-gpu-247) arm=dpo_identity005_s4 ;;
  auh7-1b-gpu-248) arm=dpo_identity015_s4 ;;
  auh7-1b-gpu-279) arm=dpo_identity010_s8 ;;
  *) echo "forbidden node outside Job 140846" >&2; exit 3 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
decode_root="$stage/interaction_complex8_large_lora_decode_v1/$arm"
worker="$stage/auh_decode_interaction_complex8_large_lora_job140846_v1.sh"
while [ ! -f "$decode_root/event_$(printf '%02d' "$current_event")/COMPLETE" ]; do sleep 15; done
for event in "${remaining_events[@]}"; do
  bash "$worker" "$1" "$event"
done
