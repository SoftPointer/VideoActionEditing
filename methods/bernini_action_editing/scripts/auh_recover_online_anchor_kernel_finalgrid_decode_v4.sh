#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
training="$stage/online_anchor_attention_training_v1"
runner="$stage/source-online-anchor-attention-training-v1/methods/bernini_action_editing/scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
root="$training/dynaedit_fullgrid_v2"
logs="$training/logs/kernel_v4_finalgrid_decode_recovery"
mkdir -p "$logs"
test -f "$runner"

run_one() {
  local job="$1" node="$2" event="$3" experiment="$4" arm="$5" transport="$6" strength="$7"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --nodelist="$node" \
    env ONLINE_ANCHOR_DECODE_EXPERIMENT="$experiment" \
        ONLINE_ANCHOR_DECODE_STEP=64 \
        ONLINE_ANCHOR_DECODE_TRANSPORT="$transport" \
        ONLINE_ANCHOR_DECODE_STRENGTH="$strength" \
        bash "$runner" "$event" "$arm" 0
}

kernel=temporal_kernel_contrast_attn_output
hard=target_gated_hard_kernel_top25_attn_output

if [[ "${1:-}" == --functions-only ]]; then
  return 0 2>/dev/null || exit 0
fi

nohup bash -lc "
  set -euo pipefail
  source '$0' --functions-only
  run_one 143808 auh7-1b-gpu-268 4 kernel_actionnoop_r050_replay025_lr1e5_s64_v4 action_noop '$kernel' 0.50
  run_one 143808 auh7-1b-gpu-268 4 kernel_actionnoop_r025_replay025_lr1e5_s64_v4 action_noop '$kernel' 0.25
" >"$logs/node268_kernel50_e04_then_primary_e04.log" 2>&1 &
echo "$! node268: kernel50-E04 then primary-E04"

nohup bash -lc "
  set -euo pipefail
  source '$0' --functions-only
  run_one 143808 auh7-1b-gpu-292 0 kernel_actionnoop_r025_replay050_lr1e5_s64_v4 action_noop '$kernel' 0.25
" >"$logs/node292_replay50_e00.log" 2>&1 &
echo "$! node292: replay50-E00"

nohup bash -lc "
  set -euo pipefail
  source '$0' --functions-only
  run_one 143808 auh7-1b-gpu-315 0 kernel_dynamicstatic_r025_replay025_lr1e5_s64_v4 dynamic_static '$kernel' 0.25
  run_one 143808 auh7-1b-gpu-315 4 kernel_dynamicstatic_r025_replay025_lr1e5_s64_v4 dynamic_static '$kernel' 0.25
" >"$logs/node315_dynamicstatic_e00_then_e04.log" 2>&1 &
echo "$! node315: dynamic-static E00 then E04"

nohup bash -lc "
  set -euo pipefail
  source '$0' --functions-only
  run_one 143812 auh7-1b-gpu-293 0 hard25_actionnoop_r100_replay025_lr1e5_s64_v4 action_noop '$hard' 1.00
" >"$logs/node293_hard100_e00.log" 2>&1 &
echo "$! node293: hard100-E00"

nohup bash -lc "
  set -euo pipefail
  source '$0' --functions-only
  run_one 143811 auh7-1b-gpu-306 4 hard25_actionnoop_r100_replay025_lr1e5_s64_v4 action_noop '$hard' 1.00
" >"$logs/node306_hard100_e04.log" 2>&1 &
echo "$! node306: hard100-E04"
