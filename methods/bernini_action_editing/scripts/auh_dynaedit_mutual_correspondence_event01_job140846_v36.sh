#!/usr/bin/env bash
set -euo pipefail

# Round 36 is the final correspondence canary.  It permits an anchor route
# only when current->anchor and anchor->current phase-0 feature matches agree.
# This removes Round 35's many-to-one token collapse.  Two block bands test
# whether role semantics live before or after the earlier visual-attention band.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=MUTUALCORR_QK_A050_B4_9_P16_R8
    export TRANSPORT_OVERRIDE=temporal_mutual_correspondence_contrast_qk
    export BLOCKS_OVERRIDE=4-9
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=MUTUALCORR_ATTN_A050_B4_9_P16_R8
    export TRANSPORT_OVERRIDE=temporal_mutual_correspondence_contrast_attn_output
    export BLOCKS_OVERRIDE=4-9
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=MUTUALCORR_QK_A050_B10_15_P16_R8
    export TRANSPORT_OVERRIDE=temporal_mutual_correspondence_contrast_qk
    export BLOCKS_OVERRIDE=10-15
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=MUTUALCORR_ATTN_A050_B10_15_P16_R8
    export TRANSPORT_OVERRIDE=temporal_mutual_correspondence_contrast_attn_output
    export BLOCKS_OVERRIDE=10-15
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export STRENGTH_OVERRIDE=0.50
export ANCHOR_CONTRAST_MODE=dynamic_static_same_caption
export TRANSPORT_STEPS_OVERRIDE=40
export SGA_SCORE_MODE=background_source_cosine
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.10
export PRESERVATION_OUTSIDE_SCALE=0.0
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export PRESERVATION_START_STEP=16
export PRESERVATION_RAMP_STEPS=8
export ANCHOR_SIGMA_CAP=0.8

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_mutual_correspondence_event01_v36"
mkdir -p "$OUTPUT_ROOT_OVERRIDE"
launch_lock="$OUTPUT_ROOT_OVERRIDE/.${LABEL_OVERRIDE}.launch.lock"
if ! mkdir "$launch_lock" 2>/dev/null; then
  echo "matching arm is already running; refusing a duplicate launch: $LABEL_OVERRIDE" >&2
  exit 0
fi
cleanup_lock() { rmdir "$launch_lock" 2>/dev/null || true; }
trap cleanup_lock EXIT
"$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
