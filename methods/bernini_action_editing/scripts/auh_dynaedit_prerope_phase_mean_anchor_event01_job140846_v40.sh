#!/usr/bin/env bash
set -euo pipefail

# Round 40 stop gate: apply the fixed-anchor temporal replacement before Q/K
# projection and RoPE, then retain the original target V/content stream.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=FIXEDV0_PREROPE_PHASEMEAN_QK_B0_5_P16_R8
    export BLOCKS_OVERRIDE=0-5
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=FIXEDV0_PREROPE_PHASEMEAN_QK_B4_9_P16_R8
    export BLOCKS_OVERRIDE=4-9
    ;;
  *) echo "this stop gate uses Job 140846 nodes 246-247 group 0" >&2; exit 3 ;;
esac

export TRANSPORT_OVERRIDE=hard_prerope_phase_mean_contrast_qk
export STRENGTH_OVERRIDE=1.0
export ANCHOR_CONTRAST_MODE=dynamic_static_same_caption
export ANCHOR_CANDIDATE_MODE=single_shared
export ANCHOR_GENERIC_PROMPT=1
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
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_prerope_phase_mean_anchor_event01_v40"
mkdir -p "$OUTPUT_ROOT_OVERRIDE"
lock="$OUTPUT_ROOT_OVERRIDE/.${LABEL_OVERRIDE}.launch.lock"
if ! mkdir "$lock" 2>/dev/null; then
  echo "matching arm is already running; refusing duplicate: $LABEL_OVERRIDE" >&2
  exit 0
fi
cleanup_lock() { rmdir "$lock" 2>/dev/null || true; }
trap cleanup_lock EXIT
"$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
