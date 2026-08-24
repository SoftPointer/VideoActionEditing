#!/usr/bin/env bash
# Resume only missing action_only_lowlr fitted rollouts after node246 host OOM.

set -Eeuo pipefail

job_id=140846
train_root="${ACTION_QUOTIENT_TRAIN_ROOT:?set train root}"
eval_root="${ACTION_QUOTIENT_EVAL_ROOT:?set fitted eval root}"
one="${ACTION_QUOTIENT_EVAL_ONE:?set single rollout wrapper}"
manifest="${ACTION_QUOTIENT_EVAL_MANIFEST:?set fitted manifest}"
manifest_sha="${ACTION_QUOTIENT_EVAL_MANIFEST_SHA256:?set fitted manifest SHA-256}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
arm=action_only_lowlr

[[ -d "${eval_root}/media" && -d "${eval_root}/logs" && -x "${one}" && -f "${manifest}" ]]
[[ ! -e "${eval_root}/EVALUATION_COMPLETE" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]

payload='set -Eeuo pipefail
train_root="$1"; eval_root="$2"; one="$3"; manifest="$4"; python_bin="$5"; arm=action_only_lowlr
field() { "$python_bin" -c '\''import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])][sys.argv[3]])'\'' "$manifest" "$1" "$2"; }
for step in 10 20 40 80 160; do
  printf -v checkpoint "%s/runs/%s/checkpoint-%08d" "$train_root" "$arm" "$step"
  [[ -f "$checkpoint/receipt.json" ]]
  for row in 0 1 2 3; do
    iid=$(field "$row" iid); source=$(field "$row" source_video_path); instruction=$(field "$row" instruction)
    output="$eval_root/media/$iid/${arm}__u${step}.mp4"; receipt="$output.receipt.json"
    log="$eval_root/logs/${iid}__${arm}__u${step}.resume.log"
    if [[ -f "$output" && -f "$receipt" ]]; then
      echo "SKIP complete arm=$arm step=$step iid=$iid"
      continue
    fi
    [[ ! -e "$output" && ! -e "$receipt" ]]
    ROCR_VISIBLE_DEVICES=0,1,2,3 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      ACTION_QUOTIENT_EVAL_MODE=adapter ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="$source" \
      ACTION_QUOTIENT_EVAL_INSTRUCTION="$instruction" ACTION_QUOTIENT_EVAL_OUTPUT="$output" \
      ACTION_QUOTIENT_EVAL_ADAPTER="$checkpoint" "$one" >"$log" 2>&1
    [[ -f "$output" && -f "$receipt" ]]
    echo "OUTPUT arm=$arm step=$step iid=$iid"
  done
done
[[ "$(find "$eval_root/media" -type f -name "${arm}__u*.mp4" | wc -l)" == 20 ]]
echo COMPLETE >"$eval_root/${arm}.COMPLETE"'

srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-246 \
  --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
  bash -c "${payload}" _ "${train_root}" "${eval_root}" "${one}" "${manifest}" "${python_bin}" \
  >"${eval_root}/logs/worker-0-resume.log" 2>&1

while [[ "$(find "${eval_root}/media" -type f -name '*__u*.mp4' | wc -l)" != 160 ]]; do
  sleep 20
done
[[ "$(find "${eval_root}/media" -type f -name '*.mp4.receipt.json' | wc -l)" == 160 ]]
printf 'evaluation_complete=true\nresumed_arm=%s\nparent_allocation_cancelled=false\n' \
  "${arm}" >"${eval_root}/EVALUATION_COMPLETE"
