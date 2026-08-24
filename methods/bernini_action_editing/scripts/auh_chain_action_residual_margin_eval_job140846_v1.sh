#!/usr/bin/env bash
# Decode every residual-margin arm/checkpoint on the four fitted sources.
# Start only after V2 training and the V1 unseen challenge release node memory.

set -Eeuo pipefail

job_id=140846
train_root="${ACTION_RESIDUAL_TRAIN_ROOT:?set ACTION_RESIDUAL_TRAIN_ROOT}"
prior_eval_root="${ACTION_RESIDUAL_PRIOR_EVAL_ROOT:?set ACTION_RESIDUAL_PRIOR_EVAL_ROOT}"
eval_root="${ACTION_RESIDUAL_EVAL_ROOT:?set ACTION_RESIDUAL_EVAL_ROOT}"
one="${ACTION_RESIDUAL_EVAL_ONE:?set ACTION_RESIDUAL_EVAL_ONE}"
manifest="${ACTION_RESIDUAL_EVAL_MANIFEST:?set ACTION_RESIDUAL_EVAL_MANIFEST}"
manifest_sha="${ACTION_RESIDUAL_EVAL_MANIFEST_SHA256:?set ACTION_RESIDUAL_EVAL_MANIFEST_SHA256}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
first_arms=(margin_005 margin_020 margin_010_perp_100 margin_010_perp_100_onset_400)
second_arms=(margin_010 margin_010_perp_010 margin_010_perp_100_onset_100 margin_010_perp_100_onset_400_noop_020)

[[ ! -e "${eval_root}" && -x "${one}" && -f "${manifest}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
while [[ ! -f "${train_root}/TRAINING_COMPLETE" || ! -f "${prior_eval_root}/EVALUATION_COMPLETE" ]]; do
  sleep 60
done
sleep 60
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
mkdir -p "${eval_root}/logs" "${eval_root}/media"

payload='set -Eeuo pipefail
first_arm="$1"; second_arm="$2"; train_root="$3"; eval_root="$4"; one="$5"; manifest="$6"; python_bin="$7"
field() { "$python_bin" -c '\''import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])][sys.argv[3]])'\'' "$manifest" "$1" "$2"; }
for arm in "$first_arm" "$second_arm"; do
  for step in 10 20 40 80 160; do
    printf -v checkpoint "%s/runs/%s/checkpoint-%08d" "$train_root" "$arm" "$step"
    [[ -f "$checkpoint/receipt.json" ]]
    for row in 0 1 2 3; do
      iid=$(field "$row" iid); source=$(field "$row" source_video_path); instruction=$(field "$row" instruction)
      output="$eval_root/media/$iid/${arm}__u${step}.mp4"; log="$eval_root/logs/${iid}__${arm}__u${step}.log"
      [[ ! -e "$output" ]]
      ROCR_VISIBLE_DEVICES=0,1,2,3 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        ACTION_QUOTIENT_EVAL_MODE=adapter ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="$source" \
        ACTION_QUOTIENT_EVAL_INSTRUCTION="$instruction" ACTION_QUOTIENT_EVAL_OUTPUT="$output" \
        ACTION_QUOTIENT_EVAL_SOURCE_ONSET_POLICY=none ACTION_QUOTIENT_EVAL_ADAPTER="$checkpoint" \
        "$one" >"$log" 2>&1
      echo "OUTPUT arm=$arm step=$step iid=$iid"
    done
  done
  echo COMPLETE >"$eval_root/${arm}.COMPLETE"
done'

pids=()
for index in 0 1 2 3; do
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[$index]}" \
    --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
    bash -c "${payload}" _ "${first_arms[$index]}" "${second_arms[$index]}" \
      "${train_root}" "${eval_root}" "${one}" "${manifest}" "${python_bin}" \
      >"${eval_root}/logs/worker-${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for index in 0 1 2 3; do wait "${pids[$index]}" || status=1; done
(( status == 0 ))
[[ "$(find "${eval_root}/media" -type f -name '*__u*.mp4' | wc -l)" == 160 ]]
printf 'evaluation_complete=true\nsource_onset_policy=none\nparent_allocation_cancelled=false\n' \
  >"${eval_root}/EVALUATION_COMPLETE"
