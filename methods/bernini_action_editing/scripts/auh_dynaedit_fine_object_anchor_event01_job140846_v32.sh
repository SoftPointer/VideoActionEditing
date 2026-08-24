#!/usr/bin/env bash
set -euo pipefail

# Fine-search the object-support threshold, then compare no-anchor against two
# real online pure-T2V anchor routes under the same preservation constraint.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=SRCMOTION_K10_RESID0125_NOANCHOR
    export PRESERVATION_RESIDUAL_FRACTION=0.0125
    export TRANSPORT_STEPS_OVERRIDE=0
    export ANCHOR_CONTRAST_MODE=caption_noop_same_video
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=SRCMOTION_K10_RESID015_NOANCHOR
    export PRESERVATION_RESIDUAL_FRACTION=0.015
    export TRANSPORT_STEPS_OVERRIDE=0
    export ANCHOR_CONTRAST_MODE=caption_noop_same_video
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=SRCMOTION_K10_RESID015_DYNQK
    export PRESERVATION_RESIDUAL_FRACTION=0.015
    export TRANSPORT_OVERRIDE=temporal_contrast_qk
    export TRANSPORT_STEPS_OVERRIDE=8
    export STRENGTH_OVERRIDE=0.10
    export BLOCKS_OVERRIDE=4-9
    export ANCHOR_CONTRAST_MODE=dynamic_static_same_caption
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=SRCMOTION_K10_RESID015_DYNVEL
    export PRESERVATION_RESIDUAL_FRACTION=0.015
    export TRANSPORT_OVERRIDE=temporal_contrast_velocity
    export TRANSPORT_STEPS_OVERRIDE=8
    export STRENGTH_OVERRIDE=0.10
    export BLOCKS_OVERRIDE=8-13
    export ANCHOR_CONTRAST_MODE=dynamic_static_same_caption
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export PRESERVATION_MODE=source_motion_support
export PRESERVATION_KEEP_FRACTION=0.10
export PRESERVATION_OUTSIDE_SCALE=0.0
export PRESERVATION_DILATION=1
export ANCHOR_SIGMA_CAP=0.8
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_fine_object_anchor_event01_v32"
exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
