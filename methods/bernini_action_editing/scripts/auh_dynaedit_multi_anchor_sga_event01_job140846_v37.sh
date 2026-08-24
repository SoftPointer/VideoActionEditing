#!/usr/bin/env bash
set -euo pipefail

# Round 37 changes SGA candidate construction rather than adding another
# attention seam.  During each of the first three coarse solver steps, the
# four candidates bind v0/v1/v2/v3 respectively.  The final early-step SGA
# weights collapse both the ANC chains and the full dense anchor latents;
# later cells query that weighted anchor endpoint online.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=BANK4_SGA_BG_TRVEL_A010_P16_R8
    export ARM_OVERRIDE=AQK_SGA5
    export SGA_SCORE_MODE=background_source_cosine
    export STRENGTH_OVERRIDE=0.10
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=BANK4_AVG_TRVEL_A010_P16_R8
    export ARM_OVERRIDE=AQK_AVG5
    export SGA_SCORE_MODE=background_source_cosine
    export STRENGTH_OVERRIDE=0.10
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=BANK4_SGA_GLOBAL_TRVEL_A010_P16_R8
    export ARM_OVERRIDE=AQK_SGA5
    export SGA_SCORE_MODE=global_source_cosine
    export STRENGTH_OVERRIDE=0.10
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=BANK4_SGA_BG_TRVEL_A025_P16_R8
    export ARM_OVERRIDE=AQK_SGA5
    export SGA_SCORE_MODE=background_source_cosine
    export STRENGTH_OVERRIDE=0.25
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export ANCHOR_CANDIDATE_MODE=bank_per_candidate
export ANCHOR_GENERIC_PROMPT=1
export TRANSPORT_OVERRIDE=temporal_contrast_velocity
export ANCHOR_CONTRAST_MODE=dynamic_static_same_caption
export TRANSPORT_STEPS_OVERRIDE=40
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.10
export PRESERVATION_OUTSIDE_SCALE=0.0
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export PRESERVATION_START_STEP=16
export PRESERVATION_RAMP_STEPS=8
export ANCHOR_SIGMA_CAP=0.8

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_multi_anchor_sga_event01_v37"
mkdir -p "$OUTPUT_ROOT_OVERRIDE"
launch_lock="$OUTPUT_ROOT_OVERRIDE/.${LABEL_OVERRIDE}.launch.lock"
if ! mkdir "$launch_lock" 2>/dev/null; then
  echo "matching arm is already running; refusing a duplicate launch: $LABEL_OVERRIDE" >&2
  exit 0
fi
cleanup_lock() { rmdir "$launch_lock" 2>/dev/null || true; }
trap cleanup_lock EXIT
"$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
