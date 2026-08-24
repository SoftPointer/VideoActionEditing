#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
logs="$release/logs/correspondence_kernel_v6_earlyroute_decode"
checkpoint_experiment=online_noanchor_replay025_lr1e5_s64_v2
output_experiment=corr25_noanchor_s64_early3_r025_v6
log="$logs/noanchor_s64_early3_r025.log"
mkdir -p "$logs"
test -f "$release/train_$checkpoint_experiment/checkpoint-00000064/receipt.json"
test ! -e "$release/dynaedit_fullgrid_v2/$output_experiment"
test ! -e "$log"

nohup bash -lc "
  set -euo pipefail
  for event in 0 4; do
    srun --jobid=143808 --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-292 \
      env ONLINE_ANCHOR_DECODE_EXPERIMENT='$checkpoint_experiment' \
          ONLINE_ANCHOR_OUTPUT_EXPERIMENT='$output_experiment' \
          ONLINE_ANCHOR_DECODE_STEP=64 \
          ONLINE_ANCHOR_DECODE_TRANSPORT=correspondence_gated_hard_kernel_top25_attn_output \
          ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS=3 \
          ONLINE_ANCHOR_DECODE_STRENGTH=0.25 \
          bash '$runner' \"\$event\" no_anchor 0
  done
" >"$log" 2>&1 &
echo "pid=$! node=auh7-1b-gpu-292 output=$output_experiment"
