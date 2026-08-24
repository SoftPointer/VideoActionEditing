#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "usage: $0" >&2
  exit 2
fi

host=$(hostname -s)
case "$host" in
  auh7-1b-gpu-246)
    label=NATIVE_TEMPORAL_QUOTIENT_FULL_ALL40_FORCEDC0_NOPRES
    transport=native_t2v_temporal_delta_replacement
    ;;
  auh7-1b-gpu-247)
    label=NATIVE_TEMPORAL_QUOTIENT_SPARSE25_ALL40_FORCEDC0_NOPRES
    transport=native_t2v_sparse25_temporal_delta_replacement
    ;;
  *)
    echo "Round 57 is restricted to Job 140846 nodes 246-247" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_native_t2v_temporal_quotient_event01_v57"
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

exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" 0
