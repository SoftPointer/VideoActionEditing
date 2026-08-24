#!/usr/bin/env bash
set -euo pipefail

# One 4-GPU island at a time per host.  Two simultaneous model loads exceeded
# the parent allocation's host-memory budget even though VRAM was available.
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_dev_v1/auh_dynaedit_native_rolewarp_event01_job140846_v60.sh"
test -x "$runner"

run_node_246() {
  srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-246 -N1 -n1 -c64 \
    --mem=64G --gres=gpu:mi210:8 "$runner" sga_full \
    >"$stage/dynaedit_native_rolewarp_event01_v60_246_sga_full_retry.log" 2>&1
  srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-246 -N1 -n1 -c64 \
    --mem=64G --gres=gpu:mi210:8 "$runner" avg_full \
    >"$stage/dynaedit_native_rolewarp_event01_v60_246_avg_full_retry.log" 2>&1
}

run_node_247() {
  srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-247 -N1 -n1 -c64 \
    --mem=64G --gres=gpu:mi210:8 "$runner" sga_sparse25 \
    >"$stage/dynaedit_native_rolewarp_event01_v60_247_sga_sparse25_retry.log" 2>&1
  srun --jobid=140846 --overlap --exact -w auh7-1b-gpu-247 -N1 -n1 -c64 \
    --mem=64G --gres=gpu:mi210:8 "$runner" avg_sparse25 \
    >"$stage/dynaedit_native_rolewarp_event01_v60_247_avg_sparse25_retry.log" 2>&1
}

run_node_246 &
pid_246=$!
run_node_247 &
pid_247=$!
wait "$pid_246"
wait "$pid_247"
