#!/usr/bin/env bash
# Run 32 full-motion postcheck shards in four eight-way waves inside an
# existing Slurm allocation.  Four nodes each host two disjoint four-GPU
# Qwen3-VL-32B workers.  A failed shard is recorded without cancelling peers.
# Existing allocation-holder steps are preserved and explicitly overlapped;
# two physical zero-VRAM audits still gate every requested node.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
nodes_csv="${MOTIVE_FULL_MOTION_POSTCHECK_NODES:?set four comma-separated nodes}"
snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT}"
manifest_dir="${MOTIVE_FULL_MOTION_GENERATION_SHARD_DIR:?set shard directory}"
wan_shards_root="${MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT:?set MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT}"
model="${MOTIVE_FULL_MOTION_POSTCHECK_MODEL:?set MOTIVE_FULL_MOTION_POSTCHECK_MODEL}"
python_bin="${MOTIVE_FULL_MOTION_POSTCHECK_PYTHON:?set MOTIVE_FULL_MOTION_POSTCHECK_PYTHON}"
ffprobe_bin="${MOTIVE_FULL_MOTION_POSTCHECK_FFPROBE:?set MOTIVE_FULL_MOTION_POSTCHECK_FFPROBE}"
ffmpeg_bin="${MOTIVE_FULL_MOTION_POSTCHECK_FFMPEG:?set MOTIVE_FULL_MOTION_POSTCHECK_FFMPEG}"
output_root="${MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT:?set output root}"
worker_cpus="${MOTIVE_FULL_MOTION_POSTCHECK_CPUS:-28}"
idle_recheck_seconds="${MOTIVE_FULL_MOTION_POSTCHECK_IDLE_RECHECK_SECONDS:-5}"
code_root="${snapshot}/methods/motive"
log_root="${output_root}/logs"
status_path="${output_root}/dispatcher_status.tsv"
controller_receipt="${output_root}/dispatcher_receipt.json"

fail() {
  echo "[full-motion-postcheck-dispatch] $*" >&2
  exit 2
}

[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid existing job ID"
[[ "${worker_cpus}" =~ ^[1-9][0-9]*$ ]] || fail "invalid worker CPU count"
[[ "${idle_recheck_seconds}" =~ ^[1-9][0-9]*$ ]] \
  || fail "invalid idle recheck interval"
IFS=, read -r -a nodes <<<"${nodes_csv}"
(( ${#nodes[@]} == 4 )) || fail "exactly four nodes are required"
declare -A seen_nodes=()
for node in "${nodes[@]}"; do
  [[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node: ${node}"
  [[ -z "${seen_nodes[${node}]:-}" ]] || fail "duplicate node: ${node}"
  seen_nodes["${node}"]=1
done

for path in \
  "${snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/goku_full_motion_postcheck.py" \
  "${model}/config.json" \
  "${python_bin}"; do
  [[ ! -L "${path}" && -e "${path}" ]] || fail "missing plain input: ${path}"
done
[[ -x "${python_bin}" ]] || fail "postcheck Python is not executable"
[[ "${model}" == *"Qwen3-VL-32B-Instruct"* ]] \
  || fail "postcheck model must be Qwen3-VL-32B-Instruct"
for binding in "ffprobe:${ffprobe_bin}" "ffmpeg:${ffmpeg_bin}"; do
  label="${binding%%:*}"
  executable="${binding#*:}"
  [[ "${executable}" == /* && "${executable}" != "/" \
    && ! -L "${executable}" && -f "${executable}" \
    && -x "${executable}" ]] \
    || fail "${label} must be an absolute regular non-symlink executable: ${executable}"
done

hash_plain_executable() {
  "${python_bin}" -c '
import hashlib
import os
from pathlib import Path
import sys
path = Path(sys.argv[1])
if not path.is_absolute() or path == Path("/") or path.is_symlink():
    raise SystemExit("media executable path policy differs: " + str(path))
if not path.is_file() or not os.access(path, os.X_OK):
    raise SystemExit("media executable is not a regular executable: " + str(path))
before = path.stat()
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
after = path.stat()
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode
)
if identity(before) != identity(after):
    raise SystemExit("media executable changed while hashing: " + str(path))
print(digest.hexdigest())
' "$1"
}

publish_create_only() {
  local source="$1"
  local target="$2"
  local label="$3"
  "${python_bin}" -c '
import os
from pathlib import Path
import stat
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
label = sys.argv[3]
if source.parent != target.parent:
    raise SystemExit(label + " atomic create-only publication must stay in one directory")
source_stat = source.lstat()
if not stat.S_ISREG(source_stat.st_mode) or stat.S_IMODE(source_stat.st_mode) != 0o400:
    raise SystemExit(label + " atomic create-only publication source is not sealed")
descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
if os.path.lexists(target):
    raise SystemExit(label + " atomic create-only publication target already exists")
try:
    os.link(source, target, follow_symlinks=False)
except FileExistsError as error:
    raise SystemExit(label + " atomic create-only publication lost a race") from error
target_stat = target.lstat()
if (
    not stat.S_ISREG(target_stat.st_mode)
    or stat.S_IMODE(target_stat.st_mode) != 0o400
    or (target_stat.st_dev, target_stat.st_ino)
    != (source_stat.st_dev, source_stat.st_ino)
):
    raise SystemExit(label + " atomic create-only publication binding differs")
directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
' "${source}" "${target}" "${label}"
}

ffprobe_sha256="$(hash_plain_executable "${ffprobe_bin}")" \
  || fail "failed to hash ffprobe executable"
ffmpeg_sha256="$(hash_plain_executable "${ffmpeg_bin}")" \
  || fail "failed to hash ffmpeg executable"
[[ "${ffprobe_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "invalid ffprobe SHA-256"
[[ "${ffmpeg_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "invalid ffmpeg SHA-256"
for directory in "${manifest_dir}" "${wan_shards_root}"; do
  [[ "${directory}" == /* && "${directory}" != "/" \
    && ! -L "${directory}" && -d "${directory}" ]] \
    || fail "input directory is not a plain non-root absolute path: ${directory}"
done
if [[ -e "${output_root}" || -L "${output_root}" ]]; then
  [[ "${output_root}" == /* && "${output_root}" != "/" \
    && ! -L "${output_root}" && -d "${output_root}" ]] \
    || fail "resume output root is not a plain absolute directory"
else
  [[ "${output_root}" == /* && "${output_root}" != "/" \
    && -d "${output_root%/*}" && -w "${output_root%/*}" ]] \
    || fail "output root parent is unavailable"
  mkdir "${output_root}"
fi
if [[ -e "${log_root}" || -L "${log_root}" ]]; then
  [[ ! -L "${log_root}" && -d "${log_root}" ]] \
    || fail "log root is not a plain directory"
else
  mkdir "${log_root}"
fi
for publication in "${status_path}" "${controller_receipt}"; do
  [[ ! -e "${publication}" && ! -L "${publication}" ]] \
    || fail "create-only publication already exists: ${publication}"
done

for shard_index in $(seq 0 31); do
  shard_id="$(printf 'shard_%03d' "${shard_index}")"
  manifest="${manifest_dir}/${shard_id}.jsonl"
  generation_root="${wan_shards_root}/${shard_id}"
  [[ ! -L "${manifest}" && -f "${manifest}" ]] \
    || fail "missing generation manifest: ${manifest}"
  [[ "$(wc -l <"${manifest}" | tr -d '[:space:]')" == "8" ]] \
    || fail "generation manifest must contain eight rows: ${manifest}"
  [[ ! -L "${generation_root}" && -d "${generation_root}" ]] \
    || fail "missing Wan shard root: ${generation_root}"
done

job_record="$(scontrol show job -o "${job_id}")"
[[ "${job_record}" == *"JobState=RUNNING"* ]] \
  || fail "existing allocation is not running"
allocated_hosts="$(scontrol show hostnames "$(squeue -j "${job_id}" -h -o '%N')")"
for node in "${nodes[@]}"; do
  grep -Fxq "${node}" <<<"${allocated_hosts}" \
    || fail "node is outside allocation: ${node}"
done
mapfile -t existing_steps < <(squeue -s -j "${job_id}" -h -o '%i')
if (( ${#existing_steps[@]} > 0 )); then
  printf '[full-motion-postcheck-dispatch] preserving existing steps:'
  printf ' %s' "${existing_steps[@]}"
  printf '\n'
fi

check_idle_node() {
  local node="$1"
  # Holder steps reserve the allocation's complete memory TRES.  mem=0 lets
  # this overlapping probe share that reservation; GPU/PID checks remain the
  # fail-closed admission authority.
  srun --overlap --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
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
    check_idle_node "${node}" \
      || fail "node failed strict idle GPU audit ${audit}: ${node}"
  done
  (( audit == 2 )) || sleep "${idle_recheck_seconds}"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${code_root}:${snapshot}"
export SLURM_EXPORT_ENV=ALL
cd "${code_root}"

preflight_wan_closure() {
  local manifest="$1"
  local generation_root="$2"
  "${python_bin}" -c '
import hashlib
import sys
from pathlib import Path
from motive import goku_full_motion_postcheck as p
manifest = Path(sys.argv[1]).resolve(strict=True)
root = Path(sys.argv[2]).resolve(strict=True)
raw = p._stable_read(manifest, context="dispatcher manifest")
rows = p._parse_jsonl_bytes(raw, context="dispatcher manifest")
if len(rows) != 8:
    raise SystemExit("dispatcher manifest does not contain eight rows")
manifest_sha = hashlib.sha256(raw).hexdigest()
contract, _ = p._validate_run_contract(
    root,
    manifest_path=manifest,
    manifest_sha256=manifest_sha,
    manifest_rows=8,
)
p._validate_generated_manifest(
    root,
    generated_manifest_path=root / "generated_manifest.jsonl",
    generation_rows=rows,
    input_manifest_sha256=manifest_sha,
    run_contract=contract,
)
' "${manifest}" "${generation_root}"
}

prepare_slot_cache() {
  local node="$1"
  local slot="$2"
  local cache="/tmp/motive-full-motion-postcheck-${job_id}-${node}-slot-${slot}"
  srun --overlap --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
    --ntasks=1 --cpus-per-task=1 --mem=0 \
    bash -c 'umask 077; mkdir -p "$1"/{tmp,xdg,torch,torch-kernels,miopen,miopen-custom}' \
    motive-postcheck-cache "${cache}"
}
for node in "${nodes[@]}"; do
  prepare_slot_cache "${node}" 0
  prepare_slot_cache "${node}" 1
done

run_postcheck_shard() {
  local shard_index="$1"
  local slot_index="$2"
  local node="${nodes[$(( slot_index / 2 ))]}"
  local local_slot="$(( slot_index % 2 ))"
  local shard_id
  shard_id="$(printf 'shard_%03d' "${shard_index}")"
  local manifest="${manifest_dir}/${shard_id}.jsonl"
  local generation_root="${wan_shards_root}/${shard_id}"
  local output="${output_root}/postcheck_${shard_id}.jsonl"
  local cache="/tmp/motive-full-motion-postcheck-${job_id}-${node}-slot-${local_slot}"
  local gpu_devices
  if (( local_slot == 0 )); then
    gpu_devices="0,1,2,3"
  else
    gpu_devices="4,5,6,7"
  fi
  srun --overlap \
    --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
    --exact \
    --ntasks=1 --cpus-per-task="${worker_cpus}" --mem=0 \
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
      "${python_bin}" -m motive.goku_full_motion_postcheck \
      --manifest "${manifest}" \
      --generation-root "${generation_root}" \
      --output "${output}" \
      --model "${model}" \
      --ffprobe "${ffprobe_bin}" \
      --ffmpeg "${ffmpeg_bin}" \
      --nframes 24 \
      --max-pixels 1179648 \
      --max-new-tokens 4096 \
      --resume
}

status_unsorted="$(mktemp "${status_path}.unsorted.XXXXXX")"
status_tmp="$(mktemp "${status_path}.tmp.XXXXXX")"
receipt_tmp=""
cleanup_publication_staging() {
  [[ -z "${status_unsorted}" ]] || rm -f -- "${status_unsorted}"
  [[ -z "${status_tmp}" ]] || rm -f -- "${status_tmp}"
  [[ -z "${receipt_tmp}" ]] || rm -f -- "${receipt_tmp}"
}
trap cleanup_publication_staging EXIT
printf 'shard\twave\tslot\tnode\tstatus\texit_code\toutput\treceipt\n' \
  >"${status_unsorted}"
failure_count=0
completed_count=0

for wave in 0 1 2 3; do
  declare -a wave_pids=()
  declare -a wave_indices=()
  declare -a wave_slots=()
  echo "[full-motion-postcheck-dispatch] starting wave=${wave}"
  for slot in 0 1 2 3 4 5 6 7; do
    shard_index="$(( wave * 8 + slot ))"
    shard_id="$(printf 'shard_%03d' "${shard_index}")"
    manifest="${manifest_dir}/${shard_id}.jsonl"
    generation_root="${wan_shards_root}/${shard_id}"
    node="${nodes[$(( slot / 2 ))]}"
    output="${output_root}/postcheck_${shard_id}.jsonl"
    receipt="${output_root}/postcheck_${shard_id}.receipt.json"
    if ! preflight_wan_closure "${manifest}" "${generation_root}" \
      >"${log_root}/${shard_id}.preflight.out" \
      2>"${log_root}/${shard_id}.preflight.err"; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${shard_id}" "${wave}" "${slot}" "${node}" \
        preflight_failed 2 "${output}" "${receipt}" >>"${status_unsorted}"
      failure_count=$(( failure_count + 1 ))
      continue
    fi
    run_postcheck_shard "${shard_index}" "${slot}" \
      >"${log_root}/${shard_id}.out" \
      2>"${log_root}/${shard_id}.err" &
    wave_pids+=("$!")
    wave_indices+=("${shard_index}")
    wave_slots+=("${slot}")
  done

  for position in "${!wave_pids[@]}"; do
    pid="${wave_pids[${position}]}"
    shard_index="${wave_indices[${position}]}"
    slot="${wave_slots[${position}]}"
    shard_id="$(printf 'shard_%03d' "${shard_index}")"
    node="${nodes[$(( slot / 2 ))]}"
    output="${output_root}/postcheck_${shard_id}.jsonl"
    receipt="${output_root}/postcheck_${shard_id}.receipt.json"
    exit_code=0
    wait "${pid}" || exit_code=$?
    if (( exit_code == 0 )) \
      && [[ ! -L "${output}" && -f "${output}" \
        && ! -L "${receipt}" && -f "${receipt}" ]]; then
      shard_status=complete
      completed_count=$(( completed_count + 1 ))
    else
      shard_status=postcheck_failed
      failure_count=$(( failure_count + 1 ))
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${shard_id}" "${wave}" "${slot}" "${node}" "${shard_status}" \
      "${exit_code}" "${output}" "${receipt}" >>"${status_unsorted}"
  done
  echo "[full-motion-postcheck-dispatch] finished wave=${wave}"
done

{
  head -n 1 "${status_unsorted}"
  tail -n +2 "${status_unsorted}" | LC_ALL=C sort -t $'\t' -k1,1
} >"${status_tmp}"
chmod 0400 "${status_tmp}"
publish_create_only "${status_tmp}" "${status_path}" "dispatcher status" \
  || fail "create-only dispatcher status publication failed"
rm -f -- "${status_unsorted}" "${status_tmp}"
status_unsorted=""
status_tmp=""

receipt_tmp="$(mktemp "${controller_receipt}.tmp.XXXXXX")"
"${python_bin}" -c '
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
status_path = Path(sys.argv[1]).resolve(strict=True)
output_path = Path(sys.argv[2])
job_id, nodes_csv, snapshot, manifest_dir, wan_root, model = sys.argv[3:9]
ffprobe_path, expected_ffprobe_sha, ffmpeg_path, expected_ffmpeg_sha = sys.argv[9:13]

def executable_record(raw_path, expected_sha, label):
    path = Path(raw_path)
    if not path.is_absolute() or path == Path("/") or path.is_symlink():
        raise SystemExit(label + " path policy differs")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(label + " is not a regular executable")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_mode,
    )
    if identity(before) != identity(after):
        raise SystemExit(label + " changed while hashing")
    observed_sha = digest.hexdigest()
    if observed_sha != expected_sha:
        raise SystemExit(label + " changed during dispatch")
    return {"path": str(path), "sha256": observed_sha}

with status_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 32:
    raise SystemExit("dispatcher status does not contain 32 shards")
payload = {
    "schema_version": "motive-goku-full-motion-postcheck-dispatch-receipt-v2",
    "status": "complete" if all(r["status"] == "complete" for r in rows) else "partial_failure",
    "slurm_job_id": int(job_id),
    "nodes": nodes_csv.split(","),
    "source_snapshot": snapshot,
    "generation_shard_dir": manifest_dir,
    "wan_shards_root": wan_root,
    "model": model,
    "media_tools": {
        "ffprobe": executable_record(
            ffprobe_path, expected_ffprobe_sha, "ffprobe executable"
        ),
        "ffmpeg": executable_record(
            ffmpeg_path, expected_ffmpeg_sha, "ffmpeg executable"
        ),
    },
    "status_tsv": str(status_path),
    "status_tsv_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
    "completed_shards": sum(r["status"] == "complete" for r in rows),
    "failed_shards": [r["shard"] for r in rows if r["status"] != "complete"],
    "shards": rows,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
payload["receipt_digest"] = hashlib.sha256(canonical(payload)).hexdigest()
output_path.write_bytes(canonical(payload) + b"\n")
os.chmod(output_path, 0o400)
' "${status_path}" "${receipt_tmp}" "${job_id}" "${nodes_csv}" \
  "${snapshot}" "${manifest_dir}" "${wan_shards_root}" "${model}" \
  "${ffprobe_bin}" "${ffprobe_sha256}" \
  "${ffmpeg_bin}" "${ffmpeg_sha256}"
publish_create_only "${receipt_tmp}" "${controller_receipt}" \
  "dispatcher receipt" \
  || fail "create-only dispatcher receipt publication failed"
rm -f -- "${receipt_tmp}"
receipt_tmp=""
trap - EXIT

echo "[full-motion-postcheck-dispatch] completed=${completed_count} " \
  "failed=${failure_count} receipt=${controller_receipt}"
(( failure_count == 0 )) || exit 1
