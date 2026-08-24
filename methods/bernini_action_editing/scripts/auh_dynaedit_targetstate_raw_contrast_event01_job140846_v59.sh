#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 raw_full|raw_sparse25|native_full|native_sparse25" >&2
  exit 2
fi

mode=$1
host=$(hostname -s)
case "$host:$mode" in
  auh7-1b-gpu-246:raw_full)
    label=TARGETSTATE_RAW_FULL_ALL40_FORCEDC0_NOPRES
    transport=targetstate_raw_delta_replacement
    gpu_group=0
    ;;
  auh7-1b-gpu-246:native_full)
    label=NATIVE_TIMED_TARGETSTATE_RAW_FULL_ALL40_FORCEDC0_NOPRES
    transport=native_targetstate_raw_delta_replacement
    gpu_group=1
    ;;
  auh7-1b-gpu-247:raw_sparse25)
    label=TARGETSTATE_RAW_SPARSE25_ALL40_FORCEDC0_NOPRES
    transport=targetstate_sparse25_raw_delta_replacement
    gpu_group=0
    ;;
  auh7-1b-gpu-247:native_sparse25)
    label=NATIVE_TIMED_TARGETSTATE_RAW_SPARSE25_ALL40_FORCEDC0_NOPRES
    transport=native_targetstate_sparse25_raw_delta_replacement
    gpu_group=1
    ;;
  *)
    echo "Round 59 mode $mode is not assigned to host $host" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_targetstate_raw_contrast_event01_v59"
export TRANSPORT_OVERRIDE="$transport"
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE=40
export BLOCKS_OVERRIDE=4-9
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=anchor_candidate0_forced
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export PRESERVATION_MODE=none
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.0
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.0
export PRESERVATION_START_STEP=0
export PRESERVATION_RAMP_STEPS=1
export SGA_SCORE_MODE=global_source_cosine
export ARM_OVERRIDE=AQK_SGA5

exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$gpu_group"
