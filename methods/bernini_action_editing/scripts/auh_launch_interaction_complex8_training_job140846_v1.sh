#!/usr/bin/env bash
set -euo pipefail

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
reward_root="$stage/interaction_complex8_reward_v1"
rollout_root="$stage/interaction_complex8_rv2v_candidates_v1"
manifest="$stage/interaction_complex8_preference_v1.json"
builder="$stage/build_interaction_complex8_preferences_v1.py"
worker="$stage/auh_train_interaction_complex8_large_lora_dpo_job140846_v1.sh"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
log_root="$stage/interaction_complex8_large_lora_dpo_v1_logs"

test -f "$builder"
test -x "$worker"
mkdir -p "$log_root"
while [ "$(find "$reward_root" -mindepth 2 -maxdepth 2 -name COMPLETE 2>/dev/null | wc -l)" -ne 8 ]; do
  sleep 15
done
if [ ! -f "$manifest" ]; then
  "$python_bin" -B "$builder" \
    --reward-root "$reward_root" \
    --rollout-root "$rollout_root" \
    --output "$manifest" \
    >"$log_root/preference_builder.log" 2>&1
fi
test -f "$manifest"

for node in auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279; do
  log="$log_root/${node}.log"
  pidfile="$log_root/${node}.pid"
  test ! -e "$log"
  test ! -e "$pidfile"
  ssh -o BatchMode=yes "$node" \
    "nohup bash '$worker' >'$log' 2>&1 </dev/null & echo \$! >'$pidfile'; cat '$pidfile'"
done
