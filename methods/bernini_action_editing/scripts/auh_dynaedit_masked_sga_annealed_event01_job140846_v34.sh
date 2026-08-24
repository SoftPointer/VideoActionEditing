#!/usr/bin/env bash
set -euo pipefail

# Round 34 changes DynaEdit's candidate selector and preservation schedule,
# not the frozen model.  The pure-T2V dynamic-minus-static attention field is
# queried at all 40 solver steps and all 52 SGA candidate cells.  The matched
# global-SGA arm isolates the schedule; the other arms score SGA candidates
# only on their non-edit background and vary when source authority ramps in.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=FULLPATH_DYNATTN_GLOBALSGA_P8_R8
    export SGA_SCORE_MODE=global_source_cosine
    export PRESERVATION_START_STEP=8
    export PRESERVATION_RAMP_STEPS=8
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=FULLPATH_DYNATTN_BGMASKSGA_P8_R8
    export SGA_SCORE_MODE=background_source_cosine
    export PRESERVATION_START_STEP=8
    export PRESERVATION_RAMP_STEPS=8
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=FULLPATH_DYNATTN_BGMASKSGA_P16_R8
    export SGA_SCORE_MODE=background_source_cosine
    export PRESERVATION_START_STEP=16
    export PRESERVATION_RAMP_STEPS=8
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=FULLPATH_DYNATTN_BGMASKSGA_P24_R8
    export SGA_SCORE_MODE=background_source_cosine
    export PRESERVATION_START_STEP=24
    export PRESERVATION_RAMP_STEPS=8
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export TRANSPORT_OVERRIDE=temporal_contrast_attn_output
export STRENGTH_OVERRIDE=0.10
export BLOCKS_OVERRIDE=4-9
export ANCHOR_CONTRAST_MODE=dynamic_static_same_caption
export TRANSPORT_STEPS_OVERRIDE=40
export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.10
export PRESERVATION_OUTSIDE_SCALE=0.0
export PRESERVATION_DILATION=1
export PRESERVATION_RESIDUAL_FRACTION=0.015
export ANCHOR_SIGMA_CAP=0.8

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_masked_sga_annealed_event01_v34"
mkdir -p "$OUTPUT_ROOT_OVERRIDE"
launch_lock="$OUTPUT_ROOT_OVERRIDE/.${LABEL_OVERRIDE}.launch.lock"
if ! mkdir "$launch_lock" 2>/dev/null; then
  echo "matching arm is already running; refusing a duplicate launch: $LABEL_OVERRIDE" >&2
  exit 0
fi
cleanup_lock() { rmdir "$launch_lock" 2>/dev/null || true; }
trap cleanup_lock EXIT
"$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
