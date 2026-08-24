#!/usr/bin/env bash
# Run the full-motion eight-row semantic canary on one already allocated,
# verified-idle eight-MI210 node.  Two independent four-GPU Qwen workers own
# disjoint logical shards; the controller is safe to detach from SSH.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
node="${MOTIVE_EXISTING_SLURM_NODE:?set MOTIVE_EXISTING_SLURM_NODE}"
snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT}"
input="${MOTIVE_FULL_MOTION_SMOKE_INPUT:?set MOTIVE_FULL_MOTION_SMOKE_INPUT}"
output_root="${MOTIVE_FULL_MOTION_SMOKE_QWEN_ROOT:?set MOTIVE_FULL_MOTION_SMOKE_QWEN_ROOT}"
gate_output="${MOTIVE_FULL_MOTION_SMOKE_GATE:?set MOTIVE_FULL_MOTION_SMOKE_GATE}"
model="${MOTIVE_FULL_MOTION_QWEN_MODEL:?set MOTIVE_FULL_MOTION_QWEN_MODEL}"
python_bin="${MOTIVE_FULL_MOTION_QWEN_PYTHON:?set MOTIVE_FULL_MOTION_QWEN_PYTHON}"
canary_iid="${MOTIVE_FULL_MOTION_CANARY_IID:-1dbe39537c984690}"
code_root="${snapshot}/methods/motive"
cache_root="/tmp/motive-goku-fullmotion-smoke-${job_id}"

fail() {
  echo "[full-motion-smoke] $*" >&2
  exit 2
}

[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid job ID"
[[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node"
for path in \
  "${code_root}/motive/goku_full_motion_qwen.py" \
  "${code_root}/motive/goku_full_motion_smoke_gate.py" \
  "${input}" \
  "${model}/config.json" \
  "${python_bin}"; do
  [[ ! -L "${path}" && -e "${path}" ]] || fail "missing plain input: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Qwen Python is not executable"
[[ "$(wc -l < "${input}")" == "8" ]] || fail "smoke input must have eight rows"
if [[ -e "${output_root}" || -L "${output_root}" ]]; then
  [[ ! -L "${output_root}" && -d "${output_root}" ]] \
    || fail "resume output root is not a plain directory"
else
  mkdir "${output_root}"
fi
[[ ! -e "${gate_output}" && ! -L "${gate_output}" ]] \
  || fail "gate output already exists: ${gate_output}"

job_record="$(scontrol show job -o "${job_id}")"
for required in "JobState=RUNNING" "NodeList=${node}" "gres/gpu:mi210=8"; do
  [[ "${job_record}" == *"${required}"* ]] || fail "allocation differs: ${required}"
done
if squeue -s -j "${job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; then
  fail "allocation already has a numbered step"
fi

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
      TMPDIR="${cache}/tmp" \
      XDG_CACHE_HOME="${cache}/xdg" \
      TORCH_HOME="${cache}/torch" \
      PYTORCH_KERNEL_CACHE_PATH="${cache}/torch-kernels" \
      MIOPEN_USER_DB_PATH="${cache}/miopen" \
      MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
      "${python_bin}" -m motive.goku_full_motion_qwen \
      --input "${input}" \
      --output "${output_root}" \
      --model "${model}" \
      --root "${input%/*}" \
      --all-shards-sequential \
      --sequential-shards "${shards}" \
      --shard-index 0 --num-shards 8 \
      --nframes 16 --tile-width 512 --mosaic-columns 4 \
      --max-pixels 2359296 --max-new-tokens 6144 \
      --attn-implementation sdpa --allow-errors --resume
}

echo "[full-motion-smoke] starting job=${job_id} node=${node} input=${input}"
run_worker a "0,1,2,3" &
pid_a=$!
run_worker b "4,5,6,7" &
pid_b=$!
status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
(( status == 0 )) || fail "Qwen worker failed: ${status}"

srun \
  --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
  --exclusive --exact --kill-on-bad-exit=1 \
  --ntasks=1 --cpus-per-task=2 --mem=8G \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="${PYTHONPATH}" \
  "${python_bin}" -m motive.goku_full_motion_smoke_gate \
  --input "${input}" --qwen-root "${output_root}" \
  --output "${gate_output}" --canary-iid "${canary_iid}" \
  --minimum-hard-passes 3 --minimum-canary-dynamic-units 2

echo "[full-motion-smoke] semantic gate passed: ${gate_output}"
