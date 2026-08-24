#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 observer|explicit 000|025" >&2
  exit 2
fi

mode=$1
identity=$2
host=$(hostname -s)
case "$host:$mode:$identity" in
  auh7-1b-gpu-246:observer:000) device_slot=0 ;;
  auh7-1b-gpu-246:observer:025) device_slot=1 ;;
  auh7-1b-gpu-247:explicit:000) device_slot=0 ;;
  auh7-1b-gpu-247:explicit:025) device_slot=1 ;;
  *)
    echo "Round 92 assignment differs: 246=observer, 247=explicit; ID000/025 use slots 0/1" >&2
    exit 3
    ;;
esac

case "$mode" in
  observer)
    transport=action_noop_observer_attn_output
    transport_steps=3
    forced=-1
    ;;
  explicit)
    transport=event01_dynamic_role_graph_source_patch_value_attn_output
    transport_steps=40
    forced=1
    ;;
esac
case "$identity" in 000) identity_value=0.0 ;; 025) identity_value=0.025 ;; esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
source_tree="$stage/anchor_qk_symmetry_v1"
export DEV_OVERRIDE="$source_tree"
export LABEL_OVERRIDE="NATIVE_FINALCORRIDOR_${mode^^}_ID${identity}"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_final_corridor_event01_v92"
export TRANSPORT_OVERRIDE="$transport"
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE="$transport_steps"
export BLOCKS_OVERRIDE=4-9
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CFG_SCOPE_OVERRIDE=shared
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export EVENT01_FORCED_ROLE_PROPOSAL_INDEX="$forced"
export PRESERVATION_MODE=source_motion_support_event01_actor_object
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export PRESERVATION_OBJECT_IDENTITY_STRENGTH="$identity_value"
export PRESERVATION_START_STEP=39
export PRESERVATION_RAMP_STEPS=1
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE=AQK_SGA5

exec "$source_tree/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$device_slot"
