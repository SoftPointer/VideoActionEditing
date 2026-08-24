#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0) event=0 ;; auh7-1b-gpu-246:1) event=1 ;;
  auh7-1b-gpu-247:0) event=2 ;; auh7-1b-gpu-247:1) event=3 ;;
  auh7-1b-gpu-248:0) event=4 ;; auh7-1b-gpu-248:1) event=5 ;;
  auh7-1b-gpu-279:0) event=6 ;; auh7-1b-gpu-279:1) event=7 ;;
  *) echo "forbidden node/group outside Job 140846" >&2; exit 3 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
decode_root="$stage/interaction_complex8_large_lora_decode_v1"
worker="$stage/auh_score_interaction_complex8_trained_decode_job140846_v1.sh"
while [ "$(find "$decode_root" -name COMPLETE -type f | wc -l)" -ne 32 ]; do sleep 15; done
bash "$worker" "$1" "$event"
