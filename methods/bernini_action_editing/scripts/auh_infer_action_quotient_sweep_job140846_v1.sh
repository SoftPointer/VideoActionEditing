#!/usr/bin/env bash
# Evaluate eight arms x five checkpoints x four fitted programs on all 32 GPUs.

set -Eeuo pipefail

job_id=140846
train_root="${ACTION_QUOTIENT_TRAIN_ROOT:?set ACTION_QUOTIENT_TRAIN_ROOT}"
eval_root="${ACTION_QUOTIENT_EVAL_ROOT:?set ACTION_QUOTIENT_EVAL_ROOT}"
one="${ACTION_QUOTIENT_EVAL_ONE:?set ACTION_QUOTIENT_EVAL_ONE}"
manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_training_pairv5_20260815_v1/data/eval-manifest.json
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

[[ ! -e "${eval_root}" && -x "${one}" && -f "${manifest}" ]]
mkdir -p "${eval_root}/logs" "${eval_root}/media"
arms=(action_only action_only_lowlr action_noop action_start action_nuisance action_start_nuisance action_start_nuisance_noop action_start_nuisance_border)
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-248 auh7-1b-gpu-279 auh7-1b-gpu-279)
groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)
checkpoints=(10 20 40 80 160)

field() { "${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])][sys.argv[3]])' "${manifest}" "$1" "$2"; }
for arm in "${arms[@]}"; do
  for step in "${checkpoints[@]}"; do printf -v checkpoint '%s/runs/%s/checkpoint-%08d' "${train_root}" "${arm}" "${step}"; [[ -f "${checkpoint}/receipt.json" ]]; done
done

payload='set -Eeuo pipefail
arm="$1"; group="$2"; train_root="$3"; eval_root="$4"; one="$5"; manifest="$6"; python_bin="$7"
field() { "$python_bin" -c '\''import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])][sys.argv[3]])'\'' "$manifest" "$1" "$2"; }
for step in 10 20 40 80 160; do
  printf -v checkpoint "%s/runs/%s/checkpoint-%08d" "$train_root" "$arm" "$step"
  for row in 0 1 2 3; do
    iid=$(field "$row" iid); source=$(field "$row" source_video_path); instruction=$(field "$row" instruction)
    output="$eval_root/media/$iid/${arm}__u${step}.mp4"; log="$eval_root/logs/${iid}__${arm}__u${step}.log"
    ROCR_VISIBLE_DEVICES="$group" env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      ACTION_QUOTIENT_EVAL_MODE=adapter ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="$source" \
      ACTION_QUOTIENT_EVAL_INSTRUCTION="$instruction" ACTION_QUOTIENT_EVAL_OUTPUT="$output" \
      ACTION_QUOTIENT_EVAL_ADAPTER="$checkpoint" "$one" >"$log" 2>&1
    echo "OUTPUT arm=$arm step=$step iid=$iid"
  done
done'

pids=()
for index in 0 1 2 3 4 5 6 7; do
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[$index]}" \
    --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 \
    bash -c "${payload}" _ "${arms[$index]}" "${groups[$index]}" "${train_root}" \
      "${eval_root}" "${one}" "${manifest}" "${python_bin}" \
      >"${eval_root}/logs/worker-${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for index in 0 1 2 3 4 5 6 7; do wait "${pids[$index]}" || status=1; done
(( status == 0 ))
[[ "$(find "${eval_root}/media" -type f -name '*.mp4' | wc -l)" == 160 ]]
echo COMPLETE >"${eval_root}/EVALUATION_COMPLETE"
