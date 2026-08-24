#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ] || [ "$1" != 0 ]; then
  echo "usage: $0 0" >&2
  exit 2
fi

host=$(hostname -s)
case "$host" in
  auh7-1b-gpu-246)
    label=NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_ALL30_P32_R8
    transport=event01_dynamic_role_graph_source_patch_value_attn_output
    ;;
  auh7-1b-gpu-247)
    label=NATIVE_EXPLSRC_SIDEALIGN_PATCHV_A100_ALL40_ALL30_P32_R8
    transport=event01_dynamic_role_graph_source_patch_value_side_aligned_attn_output
    ;;
  *)
    echo "Round 86 is restricted to nodes 246/247" >&2
    exit 3
    ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
source_tree="$stage/anchor_qk_symmetry_v1"
export DEV_OVERRIDE="$source_tree"
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_explicit_source_symmetry_all30_latepres_event01_v86"
export TRANSPORT_OVERRIDE="$transport"
export STRENGTH_OVERRIDE=1.0
export TRANSPORT_STEPS_OVERRIDE=40
export BLOCKS_OVERRIDE=0-29
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
export PRESERVATION_OBJECT_IDENTITY_STRENGTH=0.0
export PRESERVATION_START_STEP=32
export PRESERVATION_RAMP_STEPS=8
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE=AQK_SGA5

exec "$source_tree/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" 0
