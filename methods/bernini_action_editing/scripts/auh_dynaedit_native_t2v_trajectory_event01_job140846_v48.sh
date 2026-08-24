#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
host=$(hostname -s)
case "$host:$1" in
  auh7-1b-gpu-246:0)
    label=NATIVE_TRAJ_ADDATTN_A003_EARLY3_B8_13
    transport=temporal_contrast_attn_output
    strength=0.03
    steps=3
    blocks=8-13
    ;;
  auh7-1b-gpu-246:1)
    label=NATIVE_TRAJ_HARDMEAN_EARLY3_B8_13
    transport=hard_phase_mean_contrast_attn_output
    strength=1.0
    steps=3
    blocks=8-13
    ;;
  auh7-1b-gpu-247:0)
    label=NATIVE_TRAJ_HARDKERNEL_TOP10_EARLY3_B4_9
    transport=target_gated_hard_kernel_top10_attn_output
    strength=1.0
    steps=3
    blocks=4-9
    ;;
  auh7-1b-gpu-247:1)
    label=NATIVE_TRAJ_HARDKERNEL_TOP10_EARLY8_B4_9
    transport=target_gated_hard_kernel_top10_attn_output
    strength=1.0
    steps=8
    blocks=4-9
    ;;
  *) echo "Round 48 is restricted to Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export LABEL_OVERRIDE="$label"
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_native_t2v_trajectory_event01_v48"
export TRANSPORT_OVERRIDE="$transport"
export STRENGTH_OVERRIDE="$strength"
export TRANSPORT_STEPS_OVERRIDE="$steps"
export BLOCKS_OVERRIDE="$blocks"
export ANCHOR_STATE_MODE=native_t2v_trajectory
export INITIAL_NOISE_PROPOSAL_MODE=keyed_only
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
export ANCHOR_SIGMA_CAP=1.0
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.20
export PRESERVATION_OUTSIDE_SCALE=0.05
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.01
export PRESERVATION_START_STEP=8
export PRESERVATION_RAMP_STEPS=8
export SGA_SCORE_MODE=background_trust_anchor_envelope_003
export ARM_OVERRIDE=AQK_SGA5

# Every Slurm step requests exactly four GPUs.  The cgroup exposes its assigned
# physical quartet as local ordinals 0..3, independently of which half of the
# node Slurm selected.  ``$1`` chooses the scientific arm only.
exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" 0
