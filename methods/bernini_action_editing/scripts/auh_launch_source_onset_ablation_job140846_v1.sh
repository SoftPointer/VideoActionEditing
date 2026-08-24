#!/usr/bin/env bash
# Isolate frozen-base source-onset policies on the fitted a66 failure.
#
# Run the two committed post-denoise policies concurrently on nodes 246--247
# while the final V2 no-op trainer remains alone on node279.  This launcher never signals,
# releases, requeues, or cancels the parent allocation.

set -Eeuo pipefail

job_id=140846
manifest="${ACTION_ONSET_MANIFEST:?set ACTION_ONSET_MANIFEST}"
manifest_sha="${ACTION_ONSET_MANIFEST_SHA256:?set ACTION_ONSET_MANIFEST_SHA256}"
output_root="${ACTION_ONSET_OUTPUT_ROOT:?set ACTION_ONSET_OUTPUT_ROOT}"
one="${ACTION_ONSET_INFER_ONE:?set ACTION_ONSET_INFER_ONE}"
archive="${ACTION_QUOTIENT_INFER_ARCHIVE:?set ACTION_QUOTIENT_INFER_ARCHIVE}"
archive_sha="${ACTION_QUOTIENT_INFER_ARCHIVE_SHA256:?set ACTION_QUOTIENT_INFER_ARCHIVE_SHA256}"
revision="${ACTION_QUOTIENT_INFER_REVISION:?set ACTION_QUOTIENT_INFER_REVISION}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247)
policies=(hard1 ramp3)

[[ -f "${manifest}" && -x "${one}" && -f "${archive}" && ! -e "${output_root}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
mkdir -p "${output_root}/media" "${output_root}/logs"

field() {
  "${python_bin}" -c 'import json,sys; rows=json.load(open(sys.argv[1]))["rows"]; row=next(x for x in rows if x["iid"]=="a66e6818e4144928"); print(row[sys.argv[2]])' "${manifest}" "$1"
}
source_video="$(field source_video_path)"
instruction="$(field instruction)"

pids=()
for index in "${!policies[@]}"; do
  node="${nodes[$index]}"
  policy="${policies[$index]}"
  output="${output_root}/media/frozen_base__${policy}.mp4"
  log="${output_root}/logs/${policy}.log"
  (
    srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
      --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 \
      env ROCR_VISIBLE_DEVICES=0,1,2,3 \
      ACTION_QUOTIENT_EVAL_MODE=base \
      ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="${source_video}" \
      ACTION_QUOTIENT_EVAL_INSTRUCTION="${instruction}" \
      ACTION_QUOTIENT_EVAL_OUTPUT="${output}" \
      ACTION_QUOTIENT_EVAL_SOURCE_ONSET_POLICY="${policy}" \
      ACTION_QUOTIENT_INFER_ARCHIVE="${archive}" \
      ACTION_QUOTIENT_INFER_ARCHIVE_SHA256="${archive_sha}" \
      ACTION_QUOTIENT_INFER_REVISION="${revision}" \
      "${one}" >"${log}" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 ))
[[ "$(find "${output_root}/media" -type f -name '*.mp4' | wc -l)" == 2 ]]
[[ "$(find "${output_root}/media" -type f -name '*.receipt.json' | wc -l)" == 2 ]]
printf 'evaluation_complete=true\nparent_allocation_cancelled=false\n' >"${output_root}/EVALUATION_COMPLETE"
