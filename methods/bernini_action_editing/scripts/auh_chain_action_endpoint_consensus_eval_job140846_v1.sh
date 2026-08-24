#!/usr/bin/env bash
# Decode all V3 arms across fitted and unseen cases after training completes.
# Never cancel, release, requeue, or signal parent allocation Job 140846.

set -Eeuo pipefail

job_id=140846
train_root="${ACTION_ENDPOINT_TRAIN_ROOT:?set V3 train root}"
eval_root="${ACTION_ENDPOINT_EVAL_ROOT:?set fresh V3 eval root}"
one="${ACTION_ENDPOINT_EVAL_ONE:?set strict single rollout wrapper}"
fitted_manifest="${ACTION_ENDPOINT_FITTED_MANIFEST:?set fitted manifest}"
fitted_sha="${ACTION_ENDPOINT_FITTED_MANIFEST_SHA256:?set fitted manifest SHA-256}"
unseen_manifest="${ACTION_ENDPOINT_UNSEEN_MANIFEST:?set unseen manifest}"
unseen_sha="${ACTION_ENDPOINT_UNSEEN_MANIFEST_SHA256:?set unseen manifest SHA-256}"
: "${ACTION_QUOTIENT_INFER_ARCHIVE:?set sealed inference archive}"
: "${ACTION_QUOTIENT_INFER_ARCHIVE_SHA256:?set inference archive SHA-256}"
: "${ACTION_QUOTIENT_INFER_REVISION:?set inference source revision}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
arms=(
  endpoint_cell_band
  endpoint_consensus_band
  endpoint_consensus_trust_001
  endpoint_consensus_trust_010
)

[[ ! -e "${eval_root}" && -x "${one}" ]]
[[ -f "${fitted_manifest}" && -f "${unseen_manifest}" ]]
[[ "${fitted_sha}" =~ ^[0-9a-f]{64}$ && "${unseen_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "${fitted_manifest}" | awk '{print $1}')" == "${fitted_sha}" ]]
[[ "$(sha256sum "${unseen_manifest}" | awk '{print $1}')" == "${unseen_sha}" ]]
while [[ ! -f "${train_root}/TRAINING_COMPLETE" ]]; do sleep 60; done
sleep 60
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
mkdir -p "${eval_root}/fitted/logs" "${eval_root}/fitted/media" \
  "${eval_root}/unseen/logs" "${eval_root}/unseen/media"

payload='set -Eeuo pipefail
arm="$1"; train_root="$2"; eval_root="$3"; one="$4"; fitted_manifest="$5"; unseen_manifest="$6"; python_bin="$7"
field() { "$python_bin" -c '\''import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])][sys.argv[3]])'\'' "$1" "$2" "$3"; }
for dataset in fitted unseen; do
  if [[ "$dataset" == fitted ]]; then manifest="$fitted_manifest"; steps=(10 20 40 80); else manifest="$unseen_manifest"; steps=(20 40 80); fi
  for step in "${steps[@]}"; do
    printf -v checkpoint "%s/runs/%s/checkpoint-%08d" "$train_root" "$arm" "$step"
    [[ -f "$checkpoint/receipt.json" ]]
    for row in 0 1 2 3; do
      iid=$(field "$manifest" "$row" iid); source=$(field "$manifest" "$row" source_video_path); instruction=$(field "$manifest" "$row" instruction)
      output="$eval_root/$dataset/media/$iid/${arm}__u${step}.mp4"; receipt="$output.receipt.json"
      log="$eval_root/$dataset/logs/${iid}__${arm}__u${step}.log"
      [[ ! -e "$output" && ! -e "$receipt" ]]
      ROCR_VISIBLE_DEVICES=0,1,2,3 env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
        ACTION_QUOTIENT_EVAL_MODE=adapter ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="$source" \
        ACTION_QUOTIENT_EVAL_INSTRUCTION="$instruction" ACTION_QUOTIENT_EVAL_OUTPUT="$output" \
        ACTION_QUOTIENT_EVAL_SOURCE_ONSET_POLICY=none ACTION_QUOTIENT_EVAL_ADAPTER="$checkpoint" \
        "$one" >"$log" 2>&1
      [[ -f "$output" && -f "$receipt" ]]
      echo "OUTPUT dataset=$dataset arm=$arm step=$step iid=$iid"
    done
  done
done
echo COMPLETE >"$eval_root/${arm}.COMPLETE"'

pids=()
for index in 0 1 2 3; do
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${nodes[$index]}" --cpus-per-task=32 --mem=64G \
    --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
    bash -c "${payload}" _ "${arms[$index]}" "${train_root}" "${eval_root}" \
      "${one}" "${fitted_manifest}" "${unseen_manifest}" "${python_bin}" \
      >"${eval_root}/worker-${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 ))

[[ "$(find "${eval_root}/fitted/media" -type f -name '*__u*.mp4' | wc -l)" == 64 ]]
[[ "$(find "${eval_root}/fitted/media" -type f -name '*.mp4.receipt.json' | wc -l)" == 64 ]]
[[ "$(find "${eval_root}/unseen/media" -type f -name '*__u*.mp4' | wc -l)" == 48 ]]
[[ "$(find "${eval_root}/unseen/media" -type f -name '*.mp4.receipt.json' | wc -l)" == 48 ]]
printf 'evaluation_complete=true\nsource_onset_policy=none\nparent_allocation_cancelled=false\n' \
  >"${eval_root}/EVALUATION_COMPLETE"
