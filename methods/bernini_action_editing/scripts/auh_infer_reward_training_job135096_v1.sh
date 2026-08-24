#!/usr/bin/env bash
# Evaluate four trained arms plus one distributed frozen-base control per node.
# Safety: parent allocation 135096 is never cancelled, released, or signalled.

set -Eeuo pipefail

job_id=135096
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1
one="${root}/release/auh_infer_reward_training_one_v1.sh"
manifest="${root}/data/eval-manifest.json"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
arms=(baseline action_only preservation_only composite)
nodes=(auh7-1b-gpu-245 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248)

[[ -f "${root}/runs/TRAINING_COMPLETE" ]]
[[ -x "${one}" ]]
for arm in "${arms[@]}"; do
  receipt="${root}/runs/${arm}-u40/checkpoint-00000040/receipt.json"
  [[ -f "${receipt}" && ! -L "${receipt}" ]]
  [[ "$("${python_bin}" -c "import json; print(json.load(open('${receipt}'))['global_step'])")" == 40 ]]
done

payload='set -Eeuo pipefail
root="$1"; arm="$2"; frozen_index="$3"; one="$4"; manifest="$5"; python_bin="$6"
field() { "$python_bin" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"rows\"][int(sys.argv[2])][sys.argv[3]])" "$manifest" "$1" "$2"; }
run_adapter_arm() {
  for index in 0 1 2 3; do
    iid="$(field "$index" iid)"; source_video="$(field "$index" source_video_path)"; instruction="$(field "$index" instruction)"
    ROCR_VISIBLE_DEVICES=0,1,2,3 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      REWARD_EVAL_MODE=trained_adapter REWARD_EVAL_IID="$iid" \
      REWARD_EVAL_SOURCE_VIDEO="$source_video" REWARD_EVAL_INSTRUCTION="$instruction" \
      REWARD_EVAL_OUTPUT_VIDEO="$root/eval/$iid/$arm.mp4" \
      REWARD_EVAL_ADAPTER_CHECKPOINT="$root/runs/$arm-u40/checkpoint-00000040" \
      "$one" >"$root/logs/eval-$iid-$arm.r2.log" 2>&1
  done
}
run_frozen() {
  iid="$(field "$frozen_index" iid)"; source_video="$(field "$frozen_index" source_video_path)"; instruction="$(field "$frozen_index" instruction)"
  ROCR_VISIBLE_DEVICES=4,5,6,7 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    REWARD_EVAL_MODE=frozen_base REWARD_EVAL_IID="$iid" \
    REWARD_EVAL_SOURCE_VIDEO="$source_video" REWARD_EVAL_INSTRUCTION="$instruction" \
    REWARD_EVAL_OUTPUT_VIDEO="$root/eval/$iid/frozen_base.mp4" \
    "$one" >"$root/logs/eval-$iid-frozen_base.r2.log" 2>&1
}
run_adapter_arm & adapter_pid=$!
run_frozen & frozen_pid=$!
status=0
wait "$adapter_pid" || status=1
wait "$frozen_pid" || status=1
exit "$status"'

pids=()
for index in 0 1 2 3; do
  arm="${arms[$index]}"; node="${nodes[$index]}"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
    --cpus-per-task=64 --mem=500G --gres=gpu:mi210:8 \
    bash -c "${payload}" _ "${root}" "${arm}" "${index}" "${one}" "${manifest}" "${python_bin}" \
    >"${root}/logs/eval-node-${node}.r2.log" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED arm=${arm} frozen_iid_index=${index} node=${node} pid=${pids[-1]}"
done

overall=0
for index in 0 1 2 3; do
  if wait "${pids[$index]}"; then
    echo "COMPLETED arm=${arms[$index]} node=${nodes[$index]}"
  else
    overall=1
    echo "FAILED arm=${arms[$index]} node=${nodes[$index]}" >&2
  fi
done
(( overall == 0 )) || exit 1
echo COMPLETE >"${root}/eval/EVALUATION_COMPLETE"
echo "ALL_COMPLETE parent_allocation_cancelled=false parent_allocation_released=false"
