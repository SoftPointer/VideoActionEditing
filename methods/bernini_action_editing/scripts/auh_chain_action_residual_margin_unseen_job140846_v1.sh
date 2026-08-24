#!/usr/bin/env bash
# Evaluate four representative residual-margin arms on unseen interaction cases.

set -Eeuo pipefail

job_id=140846
train_root="${ACTION_RESIDUAL_TRAIN_ROOT:?set V2 train root}"
fitted_root="${ACTION_RESIDUAL_FITTED_EVAL_ROOT:?set V2 fitted eval root}"
eval_root="${ACTION_RESIDUAL_UNSEEN_EVAL_ROOT:?set V2 unseen eval root}"
one="${ACTION_RESIDUAL_EVAL_ONE:?set single rollout wrapper}"
manifest="${ACTION_RESIDUAL_UNSEEN_MANIFEST:?set unseen manifest}"
manifest_sha="${ACTION_RESIDUAL_UNSEEN_MANIFEST_SHA256:?set unseen manifest SHA-256}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
arms=(margin_005 margin_020 margin_010_perp_100 margin_010_perp_100_onset_400_noop_020)

[[ ! -e "${eval_root}" && -x "${one}" && -f "${manifest}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
while [[ ! -f "${fitted_root}/EVALUATION_COMPLETE" ]]; do sleep 60; done
sleep 60
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
mkdir -p "${eval_root}/logs" "${eval_root}/media"

payload='set -Eeuo pipefail
arm="$1"; train_root="$2"; eval_root="$3"; one="$4"; manifest="$5"; python_bin="$6"
field() { "$python_bin" -c '\''import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])][sys.argv[3]])'\'' "$manifest" "$1" "$2"; }
for step in 80 160; do
  printf -v checkpoint "%s/runs/%s/checkpoint-%08d" "$train_root" "$arm" "$step"
  [[ -f "$checkpoint/receipt.json" ]]
  for row in 0 1 2 3; do
    iid=$(field "$row" iid); source=$(field "$row" source_video_path); instruction=$(field "$row" instruction)
    output="$eval_root/media/$iid/${arm}__u${step}.mp4"; log="$eval_root/logs/${iid}__${arm}__u${step}.log"
    [[ ! -e "$output" ]]
    ROCR_VISIBLE_DEVICES=0,1,2,3 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      ACTION_QUOTIENT_EVAL_MODE=adapter ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="$source" \
      ACTION_QUOTIENT_EVAL_INSTRUCTION="$instruction" ACTION_QUOTIENT_EVAL_OUTPUT="$output" \
      ACTION_QUOTIENT_EVAL_ADAPTER="$checkpoint" "$one" >"$log" 2>&1
    echo "OUTPUT arm=$arm step=$step iid=$iid"
  done
done
echo COMPLETE >"$eval_root/${arm}.COMPLETE"'

pids=()
for index in 0 1 2 3; do
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[$index]}" \
    --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
    bash -c "${payload}" _ "${arms[$index]}" "${train_root}" "${eval_root}" \
      "${one}" "${manifest}" "${python_bin}" >"${eval_root}/logs/worker-${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for index in 0 1 2 3; do wait "${pids[$index]}" || status=1; done
(( status == 0 ))
[[ "$(find "${eval_root}/media" -type f -name '*__u*.mp4' | wc -l)" == 32 ]]
printf 'evaluation_complete=true\nsource_onset_policy=none\nparent_allocation_cancelled=false\n' \
  >"${eval_root}/EVALUATION_COMPLETE"
