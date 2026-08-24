#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 sgaanc|hard_sga|uniform_anc|no_gain FROZEN_EVENT" >&2
  exit 2
fi
profile="$1"
frozen_event="$2"
case "$profile" in
  sgaanc) arm=sgaanc_tau02_uniform25_gain10 ;;
  hard_sga) arm=sga_tau01_hard_gain10 ;;
  uniform_anc) arm=anc_uniform100_gain10 ;;
  no_gain) arm=sgaanc_tau02_uniform25_gain00 ;;
  *) echo "unsupported profile" >&2; exit 2 ;;
esac
case "$frozen_event" in 0|2|4|7) ;; *) echo "frozen event differs" >&2; exit 2 ;; esac

release=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/complex8_sga_anc_training_v1
train_root="$release/train_${arm}_all30_r256_micro2_s32_v2"
infer="$release/auh_infer_complex8_sga_anc_checkpoint_v1.sh"
test -f "$infer"

while [ ! -f "$train_root/TRAINING_COMPLETE" ]; do
  if ! pgrep -af '[t]rain_same_video_dense_flow_adapter_v1.py' | grep -F -- "--output $train_root" >/dev/null; then
    echo "training stopped before TRAINING_COMPLETE: $train_root" >&2
    exit 5
  fi
  sleep 20
done

for event in 0 2 4 7; do
  for step in 1 10 32; do
    output="$release/decode_v1/$profile/event_$(printf '%02d' "$event")/step_$(printf '%04d' "$step")/output.mp4"
    if [ ! -f "$output" ]; then
      env SGA_ANC_VISIBLE_DEVICES=0,1,2,3 bash "$infer" "$profile" "$step" "$event"
    fi
  done
done
frozen_output="$release/decode_v1/frozen/event_$(printf '%02d' "$frozen_event")/step_0000/output.mp4"
if [ ! -f "$frozen_output" ]; then
  env SGA_ANC_VISIBLE_DEVICES=0,1,2,3 bash "$infer" frozen 0 "$frozen_event"
fi
mkdir -p "$release/decode_v1/$profile"
printf 'profile=%s\nfrozen_event=%s\n' "$profile" "$frozen_event" \
  > "$release/decode_v1/$profile/DECODE_COMPLETE"
