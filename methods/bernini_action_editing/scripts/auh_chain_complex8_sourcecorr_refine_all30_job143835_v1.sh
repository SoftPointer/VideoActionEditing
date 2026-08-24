#!/usr/bin/env bash
set -euo pipefail

if [ "$(hostname -s)" != "auh7-1b-gpu-246" ]; then
  echo "all30 refinement chain is restricted to node 246" >&2
  exit 3
fi

previous_pid="${PREVIOUS_PID:?PREVIOUS_PID is required}"
case "$previous_pid" in
  *[!0-9]*|'') echo "PREVIOUS_PID must be numeric" >&2; exit 4 ;;
esac

release=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/complex8_crossappearance_motion_v1
previous_output="$release/train_sourcecorr_refine_cross_s40_all30_lr100_q512_micro1_s512_v1"
launcher=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/source-crossappearance-sourcecorr-v5refine/methods/bernini_action_editing/scripts/auh_train_complex8_source_correspondence_refine_job140846_v1.sh

while kill -0 "$previous_pid" 2>/dev/null; do
  sleep 10
done
if [ ! -f "$previous_output/TRAINING_COMPLETE" ]; then
  echo "all30-lr100 exited without TRAINING_COMPLETE; fail closed" >&2
  exit 5
fi

echo "all30-lr100 complete; starting all30-lr500 on the same allocated GPUs"
exec env ROCR_DEVICE_SET=4,5,6,7 SOURCECORR_REFINE_LR=0.0005 bash "$launcher"
