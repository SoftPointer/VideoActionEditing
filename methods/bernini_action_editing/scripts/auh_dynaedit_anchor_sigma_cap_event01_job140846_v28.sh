#!/usr/bin/env bash
set -euo pipefail

# Preserve self-generated-video information in the teacher branch even when
# the outer DynaEdit query is at maximum noise.  The wrapped Round-27 script
# keeps every other source/target/solver setting identical.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0)
    export LABEL_OVERRIDE=DYNSTATIC_VEL_A010_EARLY8_CAP08
    export ANCHOR_SIGMA_CAP=0.8
    ;;
  auh7-1b-gpu-246:1)
    export LABEL_OVERRIDE=DYNSTATIC_VEL_A010_EARLY8_CAP06
    export STRENGTH_OVERRIDE=0.10
    export ANCHOR_SIGMA_CAP=0.6
    ;;
  auh7-1b-gpu-247:0)
    export LABEL_OVERRIDE=DYNSTATIC_SELFATT_A003_EARLY8_B8_13_CAP08
    export ANCHOR_SIGMA_CAP=0.8
    ;;
  auh7-1b-gpu-247:1)
    export LABEL_OVERRIDE=DYNSTATIC_CROSSATT_A003_EARLY8_B4_9_CAP08
    export ANCHOR_SIGMA_CAP=0.8
    ;;
  *) echo "this canary uses Job 140846 nodes 246-247" >&2; exit 3 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
export OUTPUT_ROOT_OVERRIDE="$stage/dynaedit_anchor_sigma_cap_event01_v28"
exec "$stage/anchor_qk_dev_v1/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh" "$1"
