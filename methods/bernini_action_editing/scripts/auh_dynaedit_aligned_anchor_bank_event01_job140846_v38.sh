#!/usr/bin/env bash
set -euo pipefail

# Round 38 keeps the Round37 four-anchor SGA bank and changes only the spatial
# coordinate of the dense dynamic-minus-static full-network velocity route.
# Each route is affine-aligned from its temporal-energy support to the source
# motion support; phase zero remains exactly zero.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=BANK4_ALIGNED_SGA_BG_TRVEL_A010_P16_R8
    export STRENGTH_OVERRIDE=0.10
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=BANK4_ALIGNED_SGA_BG_TRVEL_A025_P16_R8
    export STRENGTH_OVERRIDE=0.25
    ;;
  *) echo "this canary uses group 0 on Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export ARM_OVERRIDE=AQK_SGA5
export ANCHOR_CANDIDATE_MODE=bank_per_candidate
export ANCHOR_SPATIAL_ALIGNMENT=motion_support_affine
export ANCHOR_GENERIC_PROMPT=1
export TRANSPORT_OVERRIDE=temporal_contrast_velocity
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
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_aligned_anchor_bank_event01_v38"
mkdir -p "$OUTPUT_ROOT_OVERRIDE"
launch_lock="$OUTPUT_ROOT_OVERRIDE/.${LABEL_OVERRIDE}.launch.lock"
if ! mkdir "$launch_lock" 2>/dev/null; then
  echo "matching arm is already running; refusing a duplicate launch: $LABEL_OVERRIDE" >&2
  exit 0
fi
cleanup_lock() { rmdir "$launch_lock" 2>/dev/null || true; }
trap cleanup_lock EXIT
"$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
