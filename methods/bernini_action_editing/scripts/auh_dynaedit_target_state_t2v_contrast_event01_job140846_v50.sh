#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s)" in
  auh7-1b-gpu-246)
    label=TARGETSTATE_T2VCON_A050_EARLY8_BG_P8_R8
    transport=target_state_temporal_contrast_velocity
    ;;
  auh7-1b-gpu-247)
    label=NATIVEGATED_TARGETSTATE_T2VCON_A050_EARLY8_BG_P8_R8
    transport=native_gated_target_state_temporal_contrast_velocity
    ;;
  *) echo "Round 50 is restricted to Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_target_state_t2v_contrast_event01_v50"
export TRANSPORT_OVERRIDE="$transport"
export STRENGTH_OVERRIDE=0.50
export TRANSPORT_STEPS_OVERRIDE=8
export BLOCKS_OVERRIDE=4-9
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export ANCHOR_SPATIAL_ALIGNMENT=none
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.01
export PRESERVATION_START_STEP=8
export PRESERVATION_RAMP_STEPS=8
export SGA_SCORE_MODE=background_source_cosine
export ARM_OVERRIDE=AQK_SGA5

exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" 0
