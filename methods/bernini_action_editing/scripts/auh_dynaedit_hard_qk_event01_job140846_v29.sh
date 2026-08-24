#!/usr/bin/env bash
set -euo pipefail

# DynaEdit SGA+ANC is the outer edit ODE.  These four arms intervene at the
# actual visual self-attention seam: anchor Q/K may replace target Q/K while
# the target/source V stream remains untouched, or dynamic-minus-static Q/K
# may be added sparsely.  This directly tests motion routing without copying
# anchor appearance values.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=HARD_QK_CAP08_FIRST1_B4_9
    export TRANSPORT_OVERRIDE=hard_qk
    export TRANSPORT_STEPS_OVERRIDE=1
    export STRENGTH_OVERRIDE=1.0
    export BLOCKS_OVERRIDE=4-9
    export ANCHOR_CONTRAST_MODE=caption_noop_same_video
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=HARD_QK_CAP08_FIRST3_B4_9
    export TRANSPORT_OVERRIDE=hard_qk
    export TRANSPORT_STEPS_OVERRIDE=3
    export STRENGTH_OVERRIDE=1.0
    export BLOCKS_OVERRIDE=4-9
    export ANCHOR_CONTRAST_MODE=caption_noop_same_video
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=HARD_K_CAP08_FIRST3_B4_9
    export TRANSPORT_OVERRIDE=hard_k
    export TRANSPORT_STEPS_OVERRIDE=3
    export STRENGTH_OVERRIDE=1.0
    export BLOCKS_OVERRIDE=4-9
    export ANCHOR_CONTRAST_MODE=caption_noop_same_video
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=DYNSTATIC_QK_A010_CAP08_EARLY8_B4_9
    export TRANSPORT_OVERRIDE=temporal_contrast_qk
    export TRANSPORT_STEPS_OVERRIDE=8
    export STRENGTH_OVERRIDE=0.10
    export BLOCKS_OVERRIDE=4-9
    export ANCHOR_CONTRAST_MODE=dynamic_static_same_caption
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

export ANCHOR_SIGMA_CAP=0.8
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_hard_qk_event01_v29"
exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
