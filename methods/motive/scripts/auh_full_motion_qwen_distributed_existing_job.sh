#!/usr/bin/env bash
# After the semantic smoke passes, run all eight logical Qwen shards at once on
# four independently verified-idle nodes from an existing eight-node job.
# Each node hosts two explicitly disjoint four-GPU workers (28 CPU/110G each).
# A long-lived allocation-holder step is preserved; every child step overlaps
# it, while two physical zero-VRAM audits guard admission to the GPUs.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
nodes_csv="${MOTIVE_FULL_MOTION_NODES:?set four comma-separated nodes}"
snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT}"
smoke_gate="${MOTIVE_FULL_MOTION_SMOKE_GATE:?set MOTIVE_FULL_MOTION_SMOKE_GATE}"
input="${MOTIVE_FULL_MOTION_FULL_INPUT:?set MOTIVE_FULL_MOTION_FULL_INPUT}"
output_root="${MOTIVE_FULL_MOTION_FULL_QWEN_ROOT:?set MOTIVE_FULL_MOTION_FULL_QWEN_ROOT}"
done_output="${MOTIVE_FULL_MOTION_FULL_QWEN_DONE:?set MOTIVE_FULL_MOTION_FULL_QWEN_DONE}"
model="${MOTIVE_FULL_MOTION_QWEN_MODEL:?set MOTIVE_FULL_MOTION_QWEN_MODEL}"
python_bin="${MOTIVE_FULL_MOTION_QWEN_PYTHON:?set MOTIVE_FULL_MOTION_QWEN_PYTHON}"
wait_seconds="${MOTIVE_FULL_MOTION_GATE_WAIT_SECONDS:-43200}"
code_root="${snapshot}/methods/motive"

fail() {
  echo "[full-motion-qwen-distributed] $*" >&2
  exit 2
}

[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid job ID"
[[ "${wait_seconds}" =~ ^[1-9][0-9]*$ ]] || fail "invalid gate wait"
IFS=, read -r -a nodes <<<"${nodes_csv}"
(( ${#nodes[@]} == 4 )) || fail "exactly four nodes are required"
declare -A seen_nodes=()
for node in "${nodes[@]}"; do
  [[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node: ${node}"
  [[ -z "${seen_nodes[${node}]:-}" ]] || fail "duplicate node: ${node}"
  seen_nodes["${node}"]=1
done
for path in "${code_root}/motive/goku_full_motion_qwen.py" \
  "${input}" "${model}/config.json" "${python_bin}"; do
  [[ ! -L "${path}" && -e "${path}" ]] || fail "missing plain input: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Qwen Python is not executable"
[[ "$(wc -l <"${input}" | tr -d '[:space:]')" == "768" ]] \
  || fail "full input must have 768 rows"
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
  "${python_bin}" -c \
  'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); print(x.get("status", ""))' \
  "${smoke_gate}"
)"
[[ "${gate_status}" == "pass" ]] || fail "semantic smoke gate did not pass"

job_record="$(scontrol show job -o "${job_id}")"
[[ "${job_record}" == *"JobState=RUNNING"* ]] || fail "allocation is not running"
[[ "${job_record}" == *"gres/gpu:mi210=64"* ]] \
  || fail "allocation does not contain 64 MI210 GPUs"
allocated_hosts="$(scontrol show hostnames "$(squeue -j "${job_id}" -h -o '%N')")"
for node in "${nodes[@]}"; do
  grep -Fxq "${node}" <<<"${allocated_hosts}" \
    || fail "node is outside allocation: ${node}"
done
mapfile -t existing_steps < <(squeue -s -j "${job_id}" -h -o '%i')
if (( ${#existing_steps[@]} > 0 )); then
  printf '[full-motion-qwen-distributed] preserving existing steps:'
  printf ' %s' "${existing_steps[@]}"
  printf '\n'
fi

check_idle_node() {
  local node="$1"
  # The long-lived holder step owns the allocation's memory TRES.  mem=0 is
  # required for an overlapping child step to share that already-owned node;
  # physical GPU admission is governed by the strict probes below.
  srun --overlap --jobid="${job_id}" \
    --nodelist="${node}" --nodes=1 \
    --ntasks=1 --cpus-per-task=1 --mem=0 \
    bash -lc '
      set -Eeuo pipefail
      metrics=$(mktemp)
      processes=$(mktemp)
      trap '\''rm -f -- "$metrics" "$processes"'\'' EXIT
      rocm-smi --showuse --showmemuse --showmeminfo vram --csv \
        >"$metrics" 2>/dev/null
      awk -F, '\''
        NR == 1 {
          for (column_number=1; column_number<=NF; column_number++) {
            label=$column_number
            gsub(/^[ "\t]+|[ "\t\r]+$/, "", label)
            if (label == "GPU use (%)") use_index=column_number
            if (label == "GPU Memory Allocated (VRAM%)") percent_index=column_number
            if (label == "VRAM Total Used Memory (B)") used_index=column_number
          }
          if (!use_index || !percent_index || !used_index) bad=1
          next
        }
        /^card[0-7],/ {
          card=$1
          if (seen_card[card]++) bad=1
          seen++
          use_value=$(use_index)
          percent_value=$(percent_index)
          used_value=$(used_index)
          gsub(/^[ "\t]+|[ "\t\r]+$/, "", use_value)
          gsub(/^[ "\t]+|[ "\t\r]+$/, "", percent_value)
          gsub(/^[ "\t]+|[ "\t\r]+$/, "", used_value)
          if (use_value !~ /^[0-9]+$/ || percent_value !~ /^[0-9]+$/ || used_value !~ /^[0-9]+$/) bad=1
          if ((use_value+0) != 0 || (percent_value+0) != 0 || (used_value+0) > 1073741824) bad=1
        }
        END { exit !(seen == 8 && !bad) }
      '\'' "$metrics"
      rocm-smi --showpids --csv >"$processes" 2>/dev/null
      awk -F, '\''
        NR == 1 { next }
        /^[[:space:]]*$/ { next }
        {
          gpu_flag=$3
          vram=$4
          gsub(/[ "\r]/, "", gpu_flag)
          gsub(/[ "\r]/, "", vram)
          if (gpu_flag !~ /^[0-9]+$/ || vram !~ /^[0-9]+$/) bad=1
          if ((gpu_flag+0) != 0 || (vram+0) != 0) bad=1
        }
        END { exit bad ? 1 : 0 }
      '\'' "$processes"
    '
}
for audit in 1 2; do
  for node in "${nodes[@]}"; do
    check_idle_node "${node}" || fail "GPU node is not idle: ${node}"
  done
  (( audit == 2 )) || sleep 5
done

# Slurm must be able to admit the exact two-worker geometry used below on one
# physical node.  Probe both disjoint four-GPU views concurrently: a serial
# probe would not establish that the two overlapping steps can coexist.
dual4_probe_log_0="$(mktemp)"
dual4_probe_log_1="$(mktemp)"
cleanup_dual4_probe() {
  rm -f -- "${dual4_probe_log_0}" "${dual4_probe_log_1}"
}
trap cleanup_dual4_probe EXIT HUP INT TERM

run_dual4_probe() {
  local probe_slot="$1"
  local gpu_devices="$2"
  local probe_log="$3"
  srun --overlap \
    --jobid="${job_id}" --nodelist="${nodes[0]}" --nodes=1 \
    --exact \
    --ntasks=1 --cpus-per-task=1 --mem=0 \
    --gpus-per-task=4 --gpu-bind=none \
    env \
      MOTIVE_DUAL4_PROBE_SLOT="${probe_slot}" \
      ROCR_VISIBLE_DEVICES="${gpu_devices}" \
      HIP_VISIBLE_DEVICES="${gpu_devices}" \
      CUDA_VISIBLE_DEVICES="${gpu_devices}" \
      "${python_bin}" -c '
import torch

expected_devices = 4
actual_devices = torch.cuda.device_count()
if actual_devices != expected_devices:
    raise SystemExit(
        f"expected exactly {expected_devices} visible GPUs, got {actual_devices}"
    )
allocations = []
for logical_device in range(expected_devices):
    with torch.cuda.device(logical_device):
        allocations.append(
            torch.ones(
                (256,),
                dtype=torch.float32,
                device=f"cuda:{logical_device}",
            )
        )
for logical_device, allocation in enumerate(allocations):
    if allocation.device.index != logical_device:
        raise SystemExit("logical GPU allocation mismatch")
    torch.cuda.synchronize(logical_device)
' >"${probe_log}" 2>&1
}

run_dual4_probe 0 "0,1,2,3" "${dual4_probe_log_0}" &
dual4_probe_pid_0=$!
run_dual4_probe 1 "4,5,6,7" "${dual4_probe_log_1}" &
dual4_probe_pid_1=$!
dual4_probe_status_0=0
dual4_probe_status_1=0
wait "${dual4_probe_pid_0}" || dual4_probe_status_0=$?
wait "${dual4_probe_pid_1}" || dual4_probe_status_1=$?
if (( dual4_probe_status_0 != 0 || dual4_probe_status_1 != 0 )); then
  printf '[full-motion-qwen-distributed] dual4 probe slot 0 output:\n' >&2
  sed 's/^/  /' "${dual4_probe_log_0}" >&2
  printf '[full-motion-qwen-distributed] dual4 probe slot 1 output:\n' >&2
  sed 's/^/  /' "${dual4_probe_log_1}" >&2
  cleanup_dual4_probe
  trap - EXIT HUP INT TERM
  fail "concurrent dual4 Qwen admission probe failed"
fi
cleanup_dual4_probe
trap - EXIT HUP INT TERM
echo "[full-motion-qwen-distributed] concurrent dual4 admission probe passed on ${nodes[0]}"

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

run_shard() {
  local shard_index="$1"
  local node="${nodes[$(( shard_index / 2 ))]}"
  local local_slot="$(( shard_index % 2 ))"
  local gpu_devices
  if (( local_slot == 0 )); then
    gpu_devices="0,1,2,3"
  else
    gpu_devices="4,5,6,7"
  fi
  local cache="/tmp/motive-goku-fullmotion-full-${job_id}-shard-${shard_index}"
  local output="${output_root}/qwen_shard_$(printf '%03d' "${shard_index}").jsonl"
  # Two bounded workers share one 2-TiB holder node.  A nonzero Slurm memory
  # request cannot overlap the holder's all-memory step, hence mem=0 here.
  srun --overlap \
    --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
    --exact \
    --ntasks=1 --cpus-per-task=28 --mem=0 \
    --gpus-per-task=4 --gpu-bind=none \
    env \
      ROCR_VISIBLE_DEVICES="${gpu_devices}" \
      HIP_VISIBLE_DEVICES="${gpu_devices}" \
      CUDA_VISIBLE_DEVICES="${gpu_devices}" \
      TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
      TORCH_HOME="${cache}/torch" \
      PYTORCH_KERNEL_CACHE_PATH="${cache}/torch-kernels" \
      MIOPEN_USER_DB_PATH="${cache}/miopen" \
      MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
      "${python_bin}" -m motive.goku_full_motion_qwen \
      --input "${input}" --output "${output}" --model "${model}" \
      --root "${input%/*}" --shard-index "${shard_index}" --num-shards 8 \
      --nframes 16 --tile-width 512 --mosaic-columns 4 \
      --max-pixels 2359296 --max-new-tokens 6144 \
      --attn-implementation sdpa --allow-errors --resume
}

# Cache creation happens inside each claimed GPU step, before Python starts.
run_shard_with_cache() {
  local shard_index="$1"
  local node="${nodes[$(( shard_index / 2 ))]}"
  local cache="/tmp/motive-goku-fullmotion-full-${job_id}-shard-${shard_index}"
  srun --overlap --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
    --ntasks=1 --cpus-per-task=1 --mem=0 \
    bash -c 'umask 077; mkdir -p "$1"/{tmp,xdg,torch,torch-kernels,miopen,miopen-custom}' \
    motive-cache "${cache}"
  run_shard "${shard_index}"
}

echo "[full-motion-qwen-distributed] smoke passed; nodes=${nodes_csv}"
pids=()
for shard_index in 0 1 2 3 4 5 6 7; do
  run_shard_with_cache "${shard_index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
(( status == 0 )) || fail "one or more distributed Qwen shards failed"

for index in 0 1 2 3 4 5 6 7; do
  stem="${output_root}/qwen_shard_$(printf '%03d' "${index}")"
  [[ ! -L "${stem}.jsonl" && -f "${stem}.jsonl" ]] || fail "missing shard ${index}"
  [[ ! -L "${stem}.receipt.json" && -f "${stem}.receipt.json" ]] \
    || fail "missing receipt ${index}"
done

# Stage the complete receipt in the destination directory, seal and fsync it,
# then expose it with link(2).  link(2) is the publication boundary: unlike
# mv(1), it never replaces a competitor that appears after the initial
# create-only preflight.
input_sha256="$("${python_bin}" -c '
import hashlib
import os
from pathlib import Path
import stat
import sys

path = Path(sys.argv[1])
before = path.lstat()
if not stat.S_ISREG(before.st_mode):
    raise SystemExit("Qwen controller input is not a plain file")
raw = path.read_bytes()
after = path.lstat()
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
)
if identity(before) != identity(after) or len(raw) != after.st_size:
    raise SystemExit("Qwen controller input changed while hashing")
print(hashlib.sha256(raw).hexdigest())
' "${input}")" || fail "failed to hash full Qwen input"
[[ "${input_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "full Qwen input SHA-256 is invalid"
temporary="$(mktemp "${done_output}.tmp.XXXXXX")"
cleanup_done_staging() {
  rm -f -- "${temporary}"
}
trap cleanup_done_staging EXIT HUP INT TERM
{
  printf 'schema=motive-goku-full-motion-qwen-controller-v1\n'
  printf 'status=complete\n'
  printf 'input=%s\n' "${input}"
  printf 'input_sha256=%s\n' "${input_sha256}"
  printf 'output_root=%s\n' "${output_root}"
  printf 'slurm_job_id=%s\n' "${job_id}"
  printf 'nodes=%s\n' "${nodes_csv}"
  printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)"
} >"${temporary}"
chmod 0400 "${temporary}"
if ! "${python_bin}" -c '
import os
from pathlib import Path
import stat
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
if source.parent != target.parent:
    raise SystemExit("atomic create-only publication must stay in one directory")
source_stat = source.lstat()
if not stat.S_ISREG(source_stat.st_mode) or stat.S_IMODE(source_stat.st_mode) != 0o400:
    raise SystemExit("atomic create-only publication source is not a sealed file")
descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
if os.path.lexists(target):
    raise SystemExit("atomic create-only publication target already exists")
try:
    os.link(source, target, follow_symlinks=False)
except FileExistsError as error:
    raise SystemExit("atomic create-only publication lost a race") from error
target_stat = target.lstat()
if (
    not stat.S_ISREG(target_stat.st_mode)
    or stat.S_IMODE(target_stat.st_mode) != 0o400
    or (target_stat.st_dev, target_stat.st_ino)
    != (source_stat.st_dev, source_stat.st_ino)
):
    raise SystemExit("atomic create-only publication binding differs")
directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
' "${temporary}" "${done_output}"; then
  fail "create-only done output publication failed"
fi
rm -f -- "${temporary}"
trap - EXIT HUP INT TERM
[[ ! -L "${done_output}" && -f "${done_output}" ]] \
  || fail "published done output is not a sealed plain file"
echo "[full-motion-qwen-distributed] complete: ${done_output}"
