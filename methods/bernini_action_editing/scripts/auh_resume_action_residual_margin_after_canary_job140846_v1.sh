#!/usr/bin/env bash
# Resume the formal residual-margin arms from an already sealed full cache.
# A fresh one-step canary with the exact source revision remains mandatory.

set -Eeuo pipefail

job_id=140846
root="${ACTION_RESIDUAL_EXPERIMENT_ROOT:?set residual experiment root}"
archive="${ACTION_RESIDUAL_SOURCE_ARCHIVE:?set sealed source archive}"
archive_sha="${ACTION_RESIDUAL_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
revision="${ACTION_RESIDUAL_SOURCE_REVISION:?set exact source revision}"
runner="${ACTION_RESIDUAL_NODE_RUNNER:?set detached node runner}"
manifest="${ACTION_RESIDUAL_SOURCE_MANIFEST:?set source-only manifest}"
manifest_sha="${ACTION_RESIDUAL_SOURCE_MANIFEST_SHA256:?set manifest SHA-256}"
cache="${root}/teacher-cache-residual-row4-slot4.pt"
expected_cache_sha="${ACTION_RESIDUAL_EXPECTED_CACHE_SHA256:?set sealed cache SHA-256}"
nodes=(auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248 auh7-1b-gpu-279)
arms=(
  margin_005 margin_010 margin_020 margin_010_perp_010
  margin_010_perp_100 margin_010_perp_100_onset_100
  margin_010_perp_100_onset_400 margin_010_perp_100_onset_400_noop_020
)

[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ && "${expected_cache_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ -d "${root}" && -f "${archive}" && -f "${manifest}" && -f "${cache}" && -x "${runner}" ]]
[[ ! -e "${root}/TRAINING_COMPLETE" && ! -e "${root}/train-canary-margin-010-r2" ]]
[[ "$(find "${root}/runs" -mindepth 1 -maxdepth 1 | wc -l)" == 0 ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]]
[[ "$(sha256sum "${cache}" | awk '{print $1}')" == "${expected_cache_sha}" ]]
state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]

common_env=(
  ACTION_RESIDUAL_SOURCE_ARCHIVE="${archive}"
  ACTION_RESIDUAL_SOURCE_ARCHIVE_SHA256="${archive_sha}"
  ACTION_RESIDUAL_SOURCE_REVISION="${revision}"
  ACTION_RESIDUAL_SOURCE_MANIFEST="${manifest}"
  ACTION_RESIDUAL_SOURCE_MANIFEST_SHA256="${manifest_sha}"
  ACTION_RESIDUAL_SEED=20260817
  ACTION_RESIDUAL_SLOTS=4
)

canary="${root}/train-canary-margin-010-r2"
srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-246 \
  --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=4,5,6,7 \
  "${common_env[@]}" ACTION_RESIDUAL_MODE=train ACTION_RESIDUAL_CACHE="${cache}" \
  ACTION_RESIDUAL_EXPECTED_CACHE_SHA256="${expected_cache_sha}" \
  ACTION_RESIDUAL_OUTPUT="${canary}" ACTION_RESIDUAL_ARM=margin_010 \
  ACTION_RESIDUAL_MAX_STEPS=1 "${runner}" >"${root}/logs/train-canary-margin-010-r2.log" 2>&1

receipt="${canary}/checkpoint-00000001/receipt.json"
[[ -f "${receipt}" ]]
python3 - "${receipt}" "${revision}" "${archive_sha}" "${expected_cache_sha}" <<'PY'
import json
import sys

path, revision, archive_sha, cache_sha = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
assert value["global_step"] == 1
assert value["method_source_revision"] == revision
assert value["method_source_archive_sha256"] == archive_sha
assert value["teacher_cache_sha256"] == cache_sha
assert value["training_contract"]["arm"] == "margin_010"
assert value["training_contract"]["historical_selected_target_reachable"] is False
PY

run_round() {
  local index arm output
  local status=0
  local -a pids=()
  for index in "$@"; do
    arm="${arms[$index]}"
    output="${root}/runs/${arm}"
    [[ ! -e "${output}" ]]
    srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[$index]}" \
      --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --kill-on-bad-exit=1 \
      env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=4,5,6,7 \
      "${common_env[@]}" ACTION_RESIDUAL_MODE=train ACTION_RESIDUAL_CACHE="${cache}" \
      ACTION_RESIDUAL_EXPECTED_CACHE_SHA256="${expected_cache_sha}" \
      ACTION_RESIDUAL_OUTPUT="${output}" ACTION_RESIDUAL_ARM="${arm}" \
      ACTION_RESIDUAL_MAX_STEPS=160 "${runner}" >"${root}/logs/train-${arm}.log" 2>&1 &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do wait "${pids[$index]}" || status=1; done
  (( status == 0 ))
}

run_round 0 1 2 3
run_round 4 5 6 7
for arm in "${arms[@]}"; do
  for step in 10 20 40 80 160; do
    [[ -f "${root}/runs/${arm}/checkpoint-$(printf '%08d' "${step}")/receipt.json" ]]
  done
done
printf 'training_complete=true\ncache_sha256=%s\nsource_revision=%s\nparent_allocation_cancelled=false\n' \
  "${expected_cache_sha}" "${revision}" >"${root}/TRAINING_COMPLETE"
