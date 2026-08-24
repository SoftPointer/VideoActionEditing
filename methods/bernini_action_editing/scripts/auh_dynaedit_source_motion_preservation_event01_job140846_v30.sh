#!/usr/bin/env bash
set -euo pipefail

# Constrain the DynaEdit clean edit state with a dense support derived from all
# source latent phases.  The moving-subject/contact neighborhood remains
# editable; outside it the source video is the authority.  No low-dimensional
# reward statistic or anchor value stream is used.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=SRCMOTION_K10_D1_HARD
    export PRESERVATION_KEEP_FRACTION=0.10
    export PRESERVATION_OUTSIDE_SCALE=0.0
    export PRESERVATION_DILATION=1
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=SRCMOTION_K20_D1_HARD
    export PRESERVATION_KEEP_FRACTION=0.20
    export PRESERVATION_OUTSIDE_SCALE=0.0
    export PRESERVATION_DILATION=1
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=SRCMOTION_K30_D1_HARD
    export PRESERVATION_KEEP_FRACTION=0.30
    export PRESERVATION_OUTSIDE_SCALE=0.0
    export PRESERVATION_DILATION=1
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=SRCMOTION_K20_D1_SOFT05
    export PRESERVATION_KEEP_FRACTION=0.20
    export PRESERVATION_OUTSIDE_SCALE=0.05
    export PRESERVATION_DILATION=1
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export PRESERVATION_MODE=source_motion_support
export TRANSPORT_STEPS_OVERRIDE=0
export ANCHOR_SIGMA_CAP=1.0
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_source_motion_preservation_event01_v30"
exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
