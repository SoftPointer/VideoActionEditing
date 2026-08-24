#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 value|role" >&2
  exit 2
fi

mode=$1
host=$(hostname -s)
case "$host:$mode" in
  auh7-1b-gpu-246:value)
    label=NATIVE_ANCHORREL_OBJECTV_FORCEDP1_A100_ALL40_P24_R8
    transport=event01_dynamic_source_object_value_attn_output
    ;;
  auh7-1b-gpu-247:role)
    label=NATIVE_ANCHORREL_ROLEGRAPH_SOURCEOBJ_FORCEDP1_A100_ALL40_P24_R8
    transport=event01_dynamic_role_graph_source_object_attn_output
    ;;
  *)
    echo "Round 74 mode $mode is not assigned to host $host" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_native_anchorrelative_hard_attention_event01_v74"
export TRANSPORT_OVERRIDE="$transport"
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE=40
export BLOCKS_OVERRIDE=4-9
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CFG_SCOPE_OVERRIDE=shared
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export EVENT01_FORCED_ROLE_PROPOSAL_INDEX=1
export PRESERVATION_MODE=source_motion_support_event01_object1
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export PRESERVATION_OBJECT_IDENTITY_STRENGTH=0.0
export PRESERVATION_START_STEP=24
export PRESERVATION_RAMP_STEPS=8
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE=AQK_SGA5

exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" 0
