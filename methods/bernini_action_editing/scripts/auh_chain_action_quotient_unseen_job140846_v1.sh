#!/usr/bin/env bash
# After the fitted checkpoint sweep, evaluate four representative objectives
# on four unseen sources with props, retained motion, and background people.

set -Eeuo pipefail

job_id=140846
train_root="${ACTION_QUOTIENT_TRAIN_ROOT:?set ACTION_QUOTIENT_TRAIN_ROOT}"
fitted_root="${ACTION_QUOTIENT_FITTED_EVAL_ROOT:?set ACTION_QUOTIENT_FITTED_EVAL_ROOT}"
eval_root="${ACTION_QUOTIENT_UNSEEN_EVAL_ROOT:?set ACTION_QUOTIENT_UNSEEN_EVAL_ROOT}"
one="${ACTION_QUOTIENT_EVAL_ONE:?set ACTION_QUOTIENT_EVAL_ONE}"
manifest="${ACTION_QUOTIENT_UNSEEN_MANIFEST:?set ACTION_QUOTIENT_UNSEEN_MANIFEST}"
manifest_sha="${ACTION_QUOTIENT_UNSEEN_MANIFEST_SHA256:?set ACTION_QUOTIENT_UNSEEN_MANIFEST_SHA256}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
arms=(action_only action_start_nuisance action_start_nuisance_noop action_start_nuisance_border)

[[ ! -e "${eval_root}" && -x "${one}" && -f "${manifest}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
mkdir -p "${eval_root}/logs" "${eval_root}/media"

payload='set -Eeuo pipefail
row_for_base="$1"; arm="$2"; train_root="$3"; eval_root="$4"; one="$5"; manifest="$6"; python_bin="$7"
field() { "$python_bin" -c '\''import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])][sys.argv[3]])'\'' "$manifest" "$1" "$2"; }
iid=$(field "$row_for_base" iid); source=$(field "$row_for_base" source_video_path); instruction=$(field "$row_for_base" instruction)
output="$eval_root/media/$iid/frozen_base.mp4"; log="$eval_root/logs/${iid}__frozen_base.log"
ROCR_VISIBLE_DEVICES=0,1,2,3 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  ACTION_QUOTIENT_EVAL_MODE=base ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="$source" \
  ACTION_QUOTIENT_EVAL_INSTRUCTION="$instruction" ACTION_QUOTIENT_EVAL_OUTPUT="$output" \
  ACTION_QUOTIENT_EVAL_ADAPTER= "$one" >"$log" 2>&1
echo "OUTPUT condition=frozen_base iid=$iid"
for step in 80 160; do
  printf -v checkpoint "%s/runs/%s/checkpoint-%08d" "$train_root" "$arm" "$step"
  for row in 0 1 2 3; do
    iid=$(field "$row" iid); source=$(field "$row" source_video_path); instruction=$(field "$row" instruction)
    output="$eval_root/media/$iid/${arm}__u${step}.mp4"; log="$eval_root/logs/${iid}__${arm}__u${step}.log"
    ROCR_VISIBLE_DEVICES=0,1,2,3 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      ACTION_QUOTIENT_EVAL_MODE=adapter ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="$source" \
      ACTION_QUOTIENT_EVAL_INSTRUCTION="$instruction" ACTION_QUOTIENT_EVAL_OUTPUT="$output" \
      ACTION_QUOTIENT_EVAL_ADAPTER="$checkpoint" "$one" >"$log" 2>&1
    echo "OUTPUT arm=$arm step=$step iid=$iid"
  done
done
echo COMPLETE >"$eval_root/${arm}.COMPLETE"'

while [[ ! -f "${fitted_root}/EVALUATION_COMPLETE" ]]; do sleep 60; done
sleep 60
pids=()
for index in 0 1 2 3; do
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[$index]}" \
    --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 \
    bash -c "${payload}" _ "${index}" "${arms[$index]}" "${train_root}" \
      "${eval_root}" "${one}" "${manifest}" "${python_bin}" \
      >"${eval_root}/logs/worker-${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for index in 0 1 2 3; do wait "${pids[$index]}" || status=1; done
(( status == 0 ))
[[ "$(find "${eval_root}/media" -type f -name '*.mp4' | wc -l)" == 36 ]]
echo COMPLETE >"${eval_root}/EVALUATION_COMPLETE"
