#!/usr/bin/env bash
# Wait for the closed full Qwen run, then produce the primary/reserve generation
# pool inside the same allocation.  No video generation is authorized here.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
node="${MOTIVE_EXISTING_SLURM_NODE:?set MOTIVE_EXISTING_SLURM_NODE}"
snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT}"
candidate_manifest="${MOTIVE_FULL_MOTION_FULL_INPUT:?set MOTIVE_FULL_MOTION_FULL_INPUT}"
qwen_root="${MOTIVE_FULL_MOTION_FULL_QWEN_ROOT:?set MOTIVE_FULL_MOTION_FULL_QWEN_ROOT}"
qwen_done="${MOTIVE_FULL_MOTION_FULL_QWEN_DONE:?set MOTIVE_FULL_MOTION_FULL_QWEN_DONE}"
output_dir="${MOTIVE_FULL_MOTION_FINAL_POOL:?set MOTIVE_FULL_MOTION_FINAL_POOL}"
python_bin="${MOTIVE_FULL_MOTION_QWEN_PYTHON:?set MOTIVE_FULL_MOTION_QWEN_PYTHON}"
wait_seconds="${MOTIVE_FULL_MOTION_QWEN_WAIT_SECONDS:-165600}"
code_root="${snapshot}/methods/motive"

fail() {
  echo "[full-motion-finalize-controller] $*" >&2
  exit 2
}

[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid job ID"
[[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node"
[[ "${wait_seconds}" =~ ^[1-9][0-9]*$ ]] || fail "invalid wait timeout"
for path in \
  "${code_root}/motive/goku_full_motion_finalize.py" \
  "${candidate_manifest}" "${python_bin}"; do
  [[ ! -L "${path}" && -e "${path}" ]] || fail "missing plain input: ${path}"
done
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] \
  || fail "final pool already exists"

deadline=$(( $(date +%s) + wait_seconds ))
while [[ ! -s "${qwen_done}" ]]; do
  (( $(date +%s) < deadline )) || fail "timed out waiting for full Qwen closure"
  sleep 20
done
grep -Fxq 'status=complete' "${qwen_done}" \
  || fail "full Qwen controller did not complete"

job_record="$(scontrol show job -o "${job_id}")"
for required in "JobState=RUNNING" "NodeList=${node}"; do
  [[ "${job_record}" == *"${required}"* ]] || fail "allocation differs: ${required}"
done
while squeue -s -j "${job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; do
  sleep 10
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}:${snapshot}"
srun \
  --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
  --exclusive --exact --kill-on-bad-exit=1 \
  --ntasks=1 --cpus-per-task=4 --mem=32G \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="${PYTHONPATH}" \
  "${python_bin}" -m motive.goku_full_motion_finalize \
  --candidate-manifest "${candidate_manifest}" \
  --qwen-dir "${qwen_root}" \
  --output-dir "${output_dir}" \
  --primary-size 256 --reserve-size 64 \
  --min-primary-multi-dynamic 64 \
  --target-signature-cap 32 --family-cap 32 \
  --require-iid 1dbe39537c984690

for artifact in \
  primary_256.jsonl reserve_64.jsonl review_candidates.jsonl summary.json done.json; do
  [[ ! -L "${output_dir}/${artifact}" && -f "${output_dir}/${artifact}" ]] \
    || fail "missing finalizer artifact: ${artifact}"
done
echo "[full-motion-finalize-controller] complete: ${output_dir}"
