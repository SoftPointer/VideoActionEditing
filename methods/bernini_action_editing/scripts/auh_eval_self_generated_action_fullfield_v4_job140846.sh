#!/usr/bin/env bash
# Decode a matched hard-phase0 checkpoint sweep for the four V4 arms.
# Uses one SP4 inference per node after formal training completes.  Each
# process needs almost the complete 64-GiB host-memory allocation, so two
# disjoint GPU islands on one node cannot safely run concurrently.

set -Eeuo pipefail

job_id=140846
train_root="${ACTION_FULLFIELD_TRAIN_ROOT:?set completed V4 training root}"
eval_root="${ACTION_FULLFIELD_EVAL_ROOT:?set fresh/resumable V4 eval root}"
runner="${ACTION_FULLFIELD_INFER_RUNNER:?set inference runner}"
archive="${ACTION_FULLFIELD_INFER_ARCHIVE:?set inference archive}"
archive_sha="${ACTION_FULLFIELD_INFER_ARCHIVE_SHA256:?set inference archive SHA-256}"
revision="${ACTION_FULLFIELD_INFER_REVISION:?set inference source revision}"
manifest="${ACTION_FULLFIELD_SOURCE_MANIFEST:?set source manifest}"

nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
devices=(0,1,2,3 0,1,2,3 0,1,2,3 0,1,2,3)
arms=(direct_anchor_sft fullfield_action_noop \
      fullfield_action_noop_pcgrad_preserve source_carrier_sft)
steps=(5 10 20 40)
iids=(7b88a1ca1f804f41 841b5e0080a1441d a35b590961d24694 a66e6818e4144928)

[[ -x "${runner}" && -f "${archive}" && -f "${manifest}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]]
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
mkdir -p "${eval_root}/logs" "${eval_root}/media"

declare -A instructions sources
sources[7b88a1ca1f804f41]="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/7b88a1ca1f804f41/samples/7b88a1ca1f804f41/source_video.mp4"
sources[841b5e0080a1441d]="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/841b5e0080a1441d/samples/841b5e0080a1441d/source_video.mp4"
sources[a35b590961d24694]="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/a35b590961d24694/samples/a35b590961d24694/source_video.mp4"
sources[a66e6818e4144928]="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/a66e6818e4144928/samples/a66e6818e4144928/source_video.mp4"

while IFS=$'\t' read -r iid encoded; do
  instructions["${iid}"]="$(printf '%s' "${encoded}" | base64 --decode)"
done < <(python3 - "${manifest}" <<'PY'
import base64
import json
import sys

for row in json.load(open(sys.argv[1], encoding="utf-8"))["rows"]:
    encoded = base64.b64encode(row["instruction"].encode("utf-8")).decode("ascii")
    print(f"{row['iid']}\t{encoded}")
PY
)
for iid in "${iids[@]}"; do
  [[ -n "${instructions[$iid]:-}" && -f "${sources[$iid]}" ]]
done

run_eval() {
  local node="$1" visible="$2" mode="$3" iid="$4" arm="$5" step="$6"
  local output log checkpoint="" step8="" attempt
  mkdir -p "${eval_root}/media/${iid}"
  if [[ "${mode}" == base ]]; then
    output="${eval_root}/media/${iid}/frozen_base__hard1_every_step.mp4"
    log="${eval_root}/logs/${iid}__frozen_base.log"
  else
    printf -v step8 '%08d' "${step}"
    checkpoint="${train_root}/runs/${arm}/checkpoint-${step8}"
    output="${eval_root}/media/${iid}/${arm}__u${step}__hard1_every_step.mp4"
    log="${eval_root}/logs/${iid}__${arm}__u${step}.log"
    [[ -f "${checkpoint}/receipt.json" ]]
  fi
  if [[ -f "${output}" && -f "${output}.receipt.json" ]]; then return 0; fi
  [[ ! -e "${output}" && ! -e "${output}.receipt.json" ]]
  # Request the parent node's full GRES view, then bind this SP4 process to
  # one explicit disjoint physical half through ROCR_VISIBLE_DEVICES.  A
  # four-GRES child cgroup exposes only logical 0--3 and would hide 4--7.
  : >"${log}"
  for attempt in 1 2 3; do
    printf 'inference_attempt=%s\n' "${attempt}" >>"${log}"
    if srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
      --nodelist="${node}" --cpus-per-task=32 --mem=64G \
      --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
      env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
      MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072 \
      ROCR_VISIBLE_DEVICES="${visible}" \
      ACTION_QUOTIENT_EVAL_MODE="${mode}" \
      ACTION_QUOTIENT_EVAL_SOURCE_VIDEO="${sources[$iid]}" \
      ACTION_QUOTIENT_EVAL_INSTRUCTION="${instructions[$iid]}" \
      ACTION_QUOTIENT_EVAL_OUTPUT="${output}" \
      ACTION_QUOTIENT_EVAL_ADAPTER="${checkpoint}" \
      ACTION_QUOTIENT_EVAL_SOURCE_ONSET_POLICY=hard1_every_step \
      ACTION_QUOTIENT_INFER_ARCHIVE="${archive}" \
      ACTION_QUOTIENT_INFER_ARCHIVE_SHA256="${archive_sha}" \
      ACTION_QUOTIENT_INFER_REVISION="${revision}" \
      "${runner}" >>"${log}" 2>&1; then
      [[ -f "${output}" && -f "${output}.receipt.json" ]]
      return 0
    fi
    # A failed process may be retried only if it published nothing.  Never
    # overwrite or silently accept a partial video/receipt pair.
    [[ ! -e "${output}" && ! -e "${output}.receipt.json" ]] || return 1
    sleep 5
  done
  return 1
}

# Formal training is complete, so each base control owns one node's lower SP4.
base_status=0
base_pids=()
for index in 0 1 2 3; do
  slot="${index}"
  run_eval "${nodes[$slot]}" "${devices[$slot]}" base "${iids[$index]}" "" 0 &
  base_pids+=("$!")
done
for pid in "${base_pids[@]}"; do wait "${pid}" || base_status=1; done
(( base_status == 0 ))

while [[ ! -f "${train_root}/TRAINING_COMPLETE" ]]; do
  sleep 15
  state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
  [[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
done

tasks=()
for iid in "${iids[@]}"; do
  for arm in "${arms[@]}"; do
    for step in "${steps[@]}"; do tasks+=("${iid}|${arm}|${step}"); done
  done
done

for ((offset=0; offset<${#tasks[@]}; offset+=4)); do
  wave_status=0
  wave_pids=()
  for slot in 0 1 2 3; do
    index=$((offset + slot))
    (( index < ${#tasks[@]} )) || continue
    IFS='|' read -r iid arm step <<<"${tasks[$index]}"
    run_eval "${nodes[$slot]}" "${devices[$slot]}" adapter \
      "${iid}" "${arm}" "${step}" &
    wave_pids+=("$!")
  done
  for pid in "${wave_pids[@]}"; do wait "${pid}" || wave_status=1; done
  (( wave_status == 0 ))
done

expected=$((4 + ${#tasks[@]}))
observed="$(find "${eval_root}/media" -type f -name '*.mp4' | wc -l | tr -d ' ')"
[[ "${observed}" == "${expected}" ]]
printf 'evaluation_complete=true\nvideo_count=%s\nsource_onset_policy=hard1_every_step\n' \
  "${observed}" >"${eval_root}/EVALUATION_COMPLETE"
