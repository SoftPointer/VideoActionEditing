#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 sga_full|avg_full|sga_sparse25|avg_sparse25" >&2
  exit 2
fi

mode=$1
host=$(hostname -s)
case "$host:$mode" in
  auh7-1b-gpu-246:sga_full)
    label=NATIVE_ROLEWARP_STONE5_SGA_FULL_P24_R8
    transport=native_rolewarp_temporal_delta_replacement
    arm=AQK_SGA5
    gpu_group=0
    ;;
  auh7-1b-gpu-246:avg_full)
    label=NATIVE_ROLEWARP_STONE5_AVG_FULL_P24_R8
    transport=native_rolewarp_temporal_delta_replacement
    arm=AQK_AVG5
    gpu_group=1
    ;;
  auh7-1b-gpu-247:sga_sparse25)
    label=NATIVE_ROLEWARP_STONE5_SGA_SPARSE25_P24_R8
    transport=native_rolewarp_sparse25_temporal_delta_replacement
    arm=AQK_SGA5
    gpu_group=0
    ;;
  auh7-1b-gpu-247:avg_sparse25)
    label=NATIVE_ROLEWARP_STONE5_AVG_SPARSE25_P24_R8
    transport=native_rolewarp_sparse25_temporal_delta_replacement
    arm=AQK_AVG5
    gpu_group=1
    ;;
  *)
    echo "Round 60 mode $mode is not assigned to host $host" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_native_rolewarp_event01_v60"
export TRANSPORT_OVERRIDE="$transport"
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE=40
export BLOCKS_OVERRIDE=4-9
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export PRESERVATION_START_STEP=24
export PRESERVATION_RAMP_STEPS=8
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE="$arm"

exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$gpu_group"
