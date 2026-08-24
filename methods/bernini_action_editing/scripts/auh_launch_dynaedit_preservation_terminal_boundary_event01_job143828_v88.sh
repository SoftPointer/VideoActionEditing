#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 observer|explicit|all" >&2
  exit 2
fi

selection=$1
job_id="${ROUND88_JOB_ID:-143828}"
stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
runner="$stage/anchor_qk_symmetry_v1/auh_dynaedit_preservation_terminal_boundary_event01_v88.sh"
output_root="$stage/dynaedit_preservation_terminal_boundary_event01_v88"

test -x "$runner"
mkdir -p "$output_root/logs"

launch_node() {
  local node=$1
  local mode=$2
  local log="$output_root/logs/${node}.log"
  srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
    --nodelist="$node" --gres=gpu:8 --cpus-per-task=64 --mem=0 \
    bash -lc "'$runner' '$mode' 37 && '$runner' '$mode' 38" >"$log" 2>&1 &
}

case "$selection" in
  observer) launch_node auh7-1b-gpu-246 observer ;;
  explicit) launch_node auh7-1b-gpu-247 explicit ;;
  all)
    launch_node auh7-1b-gpu-246 observer
    launch_node auh7-1b-gpu-247 explicit
    ;;
  *) echo "selection must be observer, explicit or all" >&2; exit 2 ;;
esac
wait

case "$selection" in
  observer) expected=(NATIVE_OBSERVER_ENVELOPE_B4_9_P37_R8 NATIVE_OBSERVER_ENVELOPE_B4_9_P38_R8) ;;
  explicit) expected=(NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P37_R8 NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P38_R8) ;;
  all) expected=(
    NATIVE_OBSERVER_ENVELOPE_B4_9_P37_R8
    NATIVE_OBSERVER_ENVELOPE_B4_9_P38_R8
    NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P37_R8
    NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P38_R8
  ) ;;
esac
for item in "${expected[@]}"; do
  test -s "$output_root/$item.mp4"
  test -s "$output_root/$item.mp4.receipt.json"
done
printf 'complete\n' > "$output_root/SWEEP_${selection}_COMPLETE"
