#!/usr/bin/env bash
set -euo pipefail
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
worker="$stage/auh_decode_interaction_complex8_large_lora_job140846_v1.sh"
log_root="$stage/interaction_complex8_large_lora_decode_v1_logs"
test -x "$worker"
mkdir -p "$log_root"
for spec in \
  auh7-1b-gpu-246:0 auh7-1b-gpu-246:1 \
  auh7-1b-gpu-247:0 auh7-1b-gpu-247:1 \
  auh7-1b-gpu-248:0 auh7-1b-gpu-248:1 \
  auh7-1b-gpu-279:0 auh7-1b-gpu-279:1
do
  node="${spec%:*}"; group="${spec#*:}"
  log="$log_root/${node}_g${group}.log"; pidfile="$log_root/${node}_g${group}.pid"
  test ! -e "$log"; test ! -e "$pidfile"
  ssh -o BatchMode=yes "$node" \
    "nohup bash '$worker' '$group' >'$log' 2>&1 </dev/null & echo \$! >'$pidfile'; cat '$pidfile'"
done
