#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 all40|early8" >&2
  exit 2
fi

mode=$1
host=$(hostname -s)
case "$host:$mode" in
  auh7-1b-gpu-246:all40)
    label=NATIVE_ROLELOGIT_SGA_ALL40_B4_9_P24_R8
    transport_steps=40
    ;;
  auh7-1b-gpu-247:early8)
    label=NATIVE_ROLELOGIT_SGA_EARLY8_B4_9_P24_R8
    transport_steps=8
    ;;
  *)
    echo "Round 63 mode $mode is not assigned to host $host" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_native_rolelogit_event01_v63"
export TRANSPORT_OVERRIDE=event01_role_graph_logit_bias_attn_output
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE="$transport_steps"
export BLOCKS_OVERRIDE=4-9
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CFG_SCOPE_OVERRIDE=shared
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
export ARM_OVERRIDE=AQK_SGA5

exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" 0
