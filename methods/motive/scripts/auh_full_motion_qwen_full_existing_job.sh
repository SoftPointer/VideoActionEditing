#!/usr/bin/env bash
# Detached continuation: wait for the real semantic smoke gate, then annotate
# the fixed full candidate pool using two four-GPU Qwen3-VL-32B owners.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
node="${MOTIVE_EXISTING_SLURM_NODE:?set MOTIVE_EXISTING_SLURM_NODE}"
snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT}"
smoke_gate="${MOTIVE_FULL_MOTION_SMOKE_GATE:?set MOTIVE_FULL_MOTION_SMOKE_GATE}"
input="${MOTIVE_FULL_MOTION_FULL_INPUT:?set MOTIVE_FULL_MOTION_FULL_INPUT}"
output_root="${MOTIVE_FULL_MOTION_FULL_QWEN_ROOT:?set MOTIVE_FULL_MOTION_FULL_QWEN_ROOT}"
done_output="${MOTIVE_FULL_MOTION_FULL_QWEN_DONE:?set MOTIVE_FULL_MOTION_FULL_QWEN_DONE}"
model="${MOTIVE_FULL_MOTION_QWEN_MODEL:?set MOTIVE_FULL_MOTION_QWEN_MODEL}"
python_bin="${MOTIVE_FULL_MOTION_QWEN_PYTHON:?set MOTIVE_FULL_MOTION_QWEN_PYTHON}"
wait_seconds="${MOTIVE_FULL_MOTION_GATE_WAIT_SECONDS:-43200}"
code_root="${snapshot}/methods/motive"
cache_root="/tmp/motive-goku-fullmotion-full-${job_id}"

fail() {
  echo "[full-motion-qwen-full] $*" >&2
  exit 2
}

[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid job ID"
[[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node"
[[ "${wait_seconds}" =~ ^[1-9][0-9]*$ ]] || fail "invalid gate wait"
for path in \
  "${code_root}/motive/goku_full_motion_qwen.py" \
  "${input}" "${model}/config.json" "${python_bin}"; do
  [[ ! -L "${path}" && -e "${path}" ]] || fail "missing plain input: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Qwen Python is not executable"
[[ "$(wc -l < "${input}")" == "768" ]] || fail "full input must have 768 rows"
[[ ! -e "${done_output}" && ! -L "${done_output}" ]] \
  || fail "done output already exists"
if [[ -e "${output_root}" || -L "${output_root}" ]]; then
  [[ ! -L "${output_root}" && -d "${output_root}" ]] \
    || fail "resume output root is not a plain directory"
else
  mkdir "${output_root}"
fi

deadline=$(( $(date +%s) + wait_seconds ))
while [[ ! -s "${smoke_gate}" ]]; do
  (( $(date +%s) < deadline )) || fail "timed out waiting for smoke gate"
  sleep 20
done
gate_status="$(
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${python_bin}" -c \
    'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); print(x.get("status", ""))' \
    "${smoke_gate}"
)"
[[ "${gate_status}" == "pass" ]] || fail "semantic smoke gate did not pass"

job_record="$(scontrol show job -o "${job_id}")"
for required in "JobState=RUNNING" "NodeList=${node}" "gres/gpu:mi210=8"; do
  [[ "${job_record}" == *"${required}"* ]] || fail "allocation differs: ${required}"
done
while squeue -s -j "${job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; do
  sleep 10
done

srun \
  --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
  --exclusive --exact --kill-on-bad-exit=1 \
  --ntasks=1 --cpus-per-task=1 --mem=1G \
  /usr/bin/env bash -c 'umask 077; mkdir -p "$@"' \
  motive-cache-setup \
  "${cache_root}/a/tmp" "${cache_root}/a/xdg" "${cache_root}/a/torch" \
  "${cache_root}/a/torch-kernels" "${cache_root}/a/miopen" \
  "${cache_root}/a/miopen-custom" \
  "${cache_root}/b/tmp" "${cache_root}/b/xdg" "${cache_root}/b/torch" \
  "${cache_root}/b/torch-kernels" "${cache_root}/b/miopen" \
  "${cache_root}/b/miopen-custom"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}:${snapshot}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export SLURM_EXPORT_ENV=ALL
cd "${code_root}"

run_worker() {
  local owner="$1"
  local shards="$2"
  local cache="${cache_root}/${owner}"
  srun \
    --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
    --exclusive --exact --kill-on-bad-exit=1 \
    --ntasks=1 --cpus-per-task=28 --mem=110G \
    --gpus-per-task=4 --gpu-bind=none \
    env \
      TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
      TORCH_HOME="${cache}/torch" \
      PYTORCH_KERNEL_CACHE_PATH="${cache}/torch-kernels" \
      MIOPEN_USER_DB_PATH="${cache}/miopen" \
      MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
      "${python_bin}" -m motive.goku_full_motion_qwen \
      --input "${input}" --output "${output_root}" --model "${model}" \
      --root "${input%/*}" --all-shards-sequential \
      --sequential-shards "${shards}" --shard-index 0 --num-shards 8 \
      --nframes 16 --tile-width 512 --mosaic-columns 4 \
      --max-pixels 2359296 --max-new-tokens 6144 \
      --attn-implementation sdpa --allow-errors --resume
}

echo "[full-motion-qwen-full] smoke passed; starting full768 on ${node}"
run_worker a "0,1,2,3" &
pid_a=$!
run_worker b "4,5,6,7" &
pid_b=$!
status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
(( status == 0 )) || fail "full Qwen worker failed: ${status}"

for index in 0 1 2 3 4 5 6 7; do
  shard="${output_root}/qwen_shard_$(printf '%03d' "${index}").jsonl"
  receipt="${output_root}/qwen_shard_$(printf '%03d' "${index}").receipt.json"
  [[ ! -L "${shard}" && -f "${shard}" ]] || fail "missing shard ${index}"
  [[ ! -L "${receipt}" && -f "${receipt}" ]] || fail "missing receipt ${index}"
done
temporary="${done_output}.tmp.$$"
{
  printf 'schema=motive-goku-full-motion-qwen-controller-v1\n'
  printf 'status=complete\n'
  printf 'input=%s\n' "${input}"
  printf 'input_sha256=%s\n' "$(sha256sum "${input}" | awk '{print $1}')"
  printf 'output_root=%s\n' "${output_root}"
  printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)"
} >"${temporary}"
chmod 0400 "${temporary}"
mv "${temporary}" "${done_output}"
echo "[full-motion-qwen-full] complete: ${done_output}"
