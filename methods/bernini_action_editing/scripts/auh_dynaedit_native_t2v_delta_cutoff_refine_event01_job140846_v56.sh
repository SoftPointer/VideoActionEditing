#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "usage: $0" >&2
  exit 2
fi

host=$(hostname -s)
case "$host" in
  auh7-1b-gpu-246)
    cutoff=20
    ;;
  auh7-1b-gpu-247)
    cutoff=22
    ;;
  *)
    echo "Round 56 is restricted to Job 140846 nodes 246-247" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
printf -v padded '%02d' "$cutoff"
export LABEL_OVERRIDE="NATIVE_DELTAVEL_CUTOFF${padded}_FORCEDC0_NOPRES"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_native_t2v_delta_cutoff_refine_event01_v56"
export TRANSPORT_OVERRIDE=native_t2v_delta_velocity_replacement
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE="$cutoff"
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

exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" 0
