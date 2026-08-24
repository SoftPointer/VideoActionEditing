#!/usr/bin/env bash
set -euo pipefail

# Retain the successful source-motion preservation support while admitting a
# very small per-phase residual support outside it.  This is the minimum extra
# degree of freedom needed for an initially static contacted stone to move.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=SRCMOTION_K10_RESID005
    export PRESERVATION_KEEP_FRACTION=0.10
    export PRESERVATION_RESIDUAL_FRACTION=0.005
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=SRCMOTION_K10_RESID010
    export PRESERVATION_KEEP_FRACTION=0.10
    export PRESERVATION_RESIDUAL_FRACTION=0.01
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=SRCMOTION_K10_RESID020
    export PRESERVATION_KEEP_FRACTION=0.10
    export PRESERVATION_RESIDUAL_FRACTION=0.02
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=SRCMOTION_K20_RESID010
    export PRESERVATION_KEEP_FRACTION=0.20
    export PRESERVATION_RESIDUAL_FRACTION=0.01
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export PRESERVATION_MODE=source_motion_support
export PRESERVATION_OUTSIDE_SCALE=0.0
export PRESERVATION_DILATION=1
export TRANSPORT_STEPS_OVERRIDE=0
export ANCHOR_SIGMA_CAP=1.0
export ANCHOR_CONTRAST_MODE=caption_noop_same_video
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_contact_residual_support_event01_v31"
exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
