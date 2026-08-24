#!/usr/bin/env bash
# Evaluate nine 320-update reward-objective arms on four matched programs.
# Eight SP4 workers use two disjoint physical GPU islands per node.

set -Eeuo pipefail

job_id=135096
output_root="${REWARD_OBJECTIVE_EVAL_ROOT:?set REWARD_OBJECTIVE_EVAL_ROOT}"
old_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1
objective_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_objective_fourarm_20260815_v2
capacity_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_capacity_fivearm_20260815_v3
manifest="${old_root}/data/eval-manifest.json"
one="${old_root}/release/auh_infer_reward_training_one_v1.sh"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

pairs=(
  "long_sft=${objective_root}/runs/long_sft_control-u320-r2/checkpoint-00000320"
  "rotation_margin=${objective_root}/runs/high_contrast_margin-u320/checkpoint-00000320"
  "reference_dpo=${objective_root}/runs/reference_dpo-u320/checkpoint-00000320"
  "reference_dpo_preservation=${objective_root}/runs/reference_dpo_preservation-u320/checkpoint-00000320"
  "action_only_sft=${capacity_root}/formal_u320/action_only_sft-formal_u320/checkpoint-00000320"
  "action_only_sft_preservation=${capacity_root}/formal_u320/action_only_sft_preservation-formal_u320/checkpoint-00000320"
  "margin_noop_only=${capacity_root}/formal_u320/margin_noop_only-formal_u320/checkpoint-00000320"
  "margin_reverse_only=${capacity_root}/formal_u320/margin_reverse_only-formal_u320/checkpoint-00000320"
  "margin_incomplete_only=${capacity_root}/formal_u320/margin_incomplete_only-formal_u320/checkpoint-00000320"
)
nodes=(
  auh7-1b-gpu-245 auh7-1b-gpu-245
  auh7-1b-gpu-246 auh7-1b-gpu-246
  auh7-1b-gpu-247 auh7-1b-gpu-247
  auh7-1b-gpu-248 auh7-1b-gpu-248
)
groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)

state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]]
[[ -f "${manifest}" && -x "${one}" ]]
[[ ! -e "${output_root}/eval" && ! -L "${output_root}/eval" ]]
for pair in "${pairs[@]}"; do
  checkpoint="${pair#*=}"
  receipt="${checkpoint}/receipt.json"
  [[ -f "${receipt}" && ! -L "${receipt}" ]]
  [[ "$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["global_step"])' "${receipt}")" == 320 ]]
done
mkdir -p "${output_root}/logs" "${output_root}/eval"

payload='set -Eeuo pipefail
output_root="$1"; worker="$2"; group="$3"; one="$4"; manifest="$5"; python_bin="$6"; shift 6; pairs=("$@")
field() { "$python_bin" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"rows\"][int(sys.argv[2])][sys.argv[3]])" "$manifest" "$1" "$2"; }
for ((task=worker; task<36; task+=8)); do
  arm_index=$((task / 4)); row_index=$((task % 4)); pair="${pairs[$arm_index]}"
  arm="${pair%%=*}"; checkpoint="${pair#*=}"
  iid="$(field "$row_index" iid)"; source_video="$(field "$row_index" source_video_path)"; instruction="$(field "$row_index" instruction)"
  output="$output_root/eval/$iid/$arm.mp4"; log="$output_root/logs/eval-$iid-$arm.log"
  [[ ! -e "$output" && ! -L "$output" ]]
  ROCR_VISIBLE_DEVICES="$group" env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    REWARD_EVAL_MODE=trained_adapter REWARD_EVAL_IID="$iid" \
    REWARD_EVAL_SOURCE_VIDEO="$source_video" REWARD_EVAL_INSTRUCTION="$instruction" \
    REWARD_EVAL_OUTPUT_VIDEO="$output" REWARD_EVAL_ADAPTER_CHECKPOINT="$checkpoint" \
    "$one" >"$log" 2>&1
  echo "OUTPUT worker=$worker arm=$arm iid=$iid physical_gpus=$group"
done'

pids=()
for worker in 0 1 2 3 4 5 6 7; do
  node="${nodes[$worker]}"; group="${groups[$worker]}"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${node}" --cpus-per-task=32 --mem=240G --gres=gpu:mi210:8 \
    bash -c "${payload}" _ "${output_root}" "${worker}" "${group}" \
      "${one}" "${manifest}" "${python_bin}" "${pairs[@]}" \
      >"${output_root}/logs/worker-${worker}.log" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED worker=${worker} node=${node} physical_gpus=${group} pid=${pids[-1]}"
done

overall=0
for worker in 0 1 2 3 4 5 6 7; do
  if wait "${pids[$worker]}"; then
    echo "COMPLETED worker=${worker} node=${nodes[$worker]}"
  else
    overall=1
    echo "FAILED worker=${worker} node=${nodes[$worker]}" >&2
  fi
done
(( overall == 0 ))
[[ "$(find "${output_root}/eval" -type f -name '*.mp4' | wc -l)" == 36 ]]
[[ "$(find "${output_root}/eval" -type f -name '*.receipt.json' | wc -l)" == 36 ]]
echo COMPLETE >"${output_root}/eval/EVALUATION_COMPLETE"
echo "ALL_COMPLETE outputs=36 parent_allocation_cancelled=false"
