#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 10|20 015|020" >&2
  exit 2
fi

keep=$1
fraction=$2
host=$(hostname -s)
case "$host:$keep:$fraction" in
  auh7-1b-gpu-246:10:015) device_slot=0 ;;
  auh7-1b-gpu-246:10:020) device_slot=1 ;;
  auh7-1b-gpu-247:20:015) device_slot=0 ;;
  auh7-1b-gpu-247:20:020) device_slot=1 ;;
  *)
    echo "Round 91 assignment differs: 246=K10, 247=K20; RF015/020 use slots 0/1" >&2
    exit 3
    ;;
esac

case "$keep" in 10) keep_value=0.10 ;; 20) keep_value=0.20 ;; esac
case "$fraction" in 015) residual=0.015 ;; 020) residual=0.02 ;; esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
source_tree="$stage/anchor_qk_symmetry_v1"
export DEV_OVERRIDE="$source_tree"
export LABEL_OVERRIDE="NATIVE_OBSERVER_FINALSTRONG_K${keep}_RF${fraction}"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_final_projection_event01_v91"
export TRANSPORT_OVERRIDE=action_noop_observer_attn_output
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE=3
export BLOCKS_OVERRIDE=4-9
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export INITIAL_PHASE_CLAMP=1
export ANCHOR_CFG_SCOPE_OVERRIDE=shared
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export EVENT01_FORCED_ROLE_PROPOSAL_INDEX=-1
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION="$keep_value"
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION="$residual"
export PRESERVATION_OBJECT_IDENTITY_STRENGTH=0.0
export PRESERVATION_START_STEP=39
export PRESERVATION_RAMP_STEPS=1
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE=AQK_SGA5

exec "$source_tree/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$device_slot"
