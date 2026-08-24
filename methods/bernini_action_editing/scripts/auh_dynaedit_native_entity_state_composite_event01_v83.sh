#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 GPU_GROUP" >&2
  exit 2
fi

gpu_group=$1
host=$(hostname -s)
case "$host:$gpu_group" in
  auh7-1b-gpu-246:0)
    label=NATIVE_ENTITYSTATE_ID025_PATCHV_A100_ALL40_B4_9_P24_R8
    identity_strength=0.025
    blocks=4-9
    ;;
  auh7-1b-gpu-246:1)
    label=NATIVE_ENTITYSTATE_ID075_PATCHV_A100_ALL40_B4_9_P24_R8
    identity_strength=0.075
    blocks=4-9
    ;;
  auh7-1b-gpu-247:0)
    label=NATIVE_ENTITYSTATE_ID025_PATCHV_A100_ALL40_B8_13_P24_R8
    identity_strength=0.025
    blocks=8-13
    ;;
  auh7-1b-gpu-247:1)
    label=NATIVE_ENTITYSTATE_ID075_PATCHV_A100_ALL40_B8_13_P24_R8
    identity_strength=0.075
    blocks=8-13
    ;;
  *)
    echo "Round 83 is restricted to GPU groups 0/1 on nodes 246/247" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
source_tree="$stage/anchor_qk_sourcepatch_v1"
export DEV_OVERRIDE="$source_tree"
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_entity_state_composite_event01_v83"
export TRANSPORT_OVERRIDE=event01_dynamic_role_graph_source_patch_value_attn_output
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE=40
export BLOCKS_OVERRIDE="$blocks"
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CFG_SCOPE_OVERRIDE=shared
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export EVENT01_FORCED_ROLE_PROPOSAL_INDEX=1
export PRESERVATION_MODE=source_motion_support_event01_actor_object
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export PRESERVATION_OBJECT_IDENTITY_STRENGTH="$identity_strength"
export PRESERVATION_START_STEP=24
export PRESERVATION_RAMP_STEPS=8
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE=AQK_SGA5

exec "$source_tree/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$gpu_group"
