#!/usr/bin/env bash
# CPU-only exact81 media diagnostics inside the user's named live allocation.

set -Eeuo pipefail
umask 077

fail() {
  echo "[saic-r6-exact28-exact81-media] ERROR: $*" >&2
  exit 2
}

if [[ "$#" -ne 14 ]]; then
  echo "usage: $0 <preflight|full> <allocation-job-id> <python-bin> <batch-source.py> <batch-source-sha256> <exact81-source.py> <exact81-source-sha256> <sealed-input-manifest> <input-manifest-sha256> <ffmpeg-bin> <ffmpeg-sha256> <ffprobe-bin> <ffprobe-sha256> <fresh-output-root>" >&2
  exit 64
fi

mode="$1"
allocation_job_id="$2"
python_bin="$3"
batch_source="$4"
batch_source_sha256="$5"
exact81_source="$6"
exact81_source_sha256="$7"
input_manifest="$8"
input_manifest_sha256="$9"
ffmpeg_bin="${10}"
ffmpeg_sha256="${11}"
ffprobe_bin="${12}"
ffprobe_sha256="${13}"
output_root="${14}"

[[ "$mode" == "preflight" || "$mode" == "full" ]] || fail "mode must be preflight or full"
[[ "$allocation_job_id" =~ ^[1-9][0-9]*$ ]] || fail "allocation JobID differs"
[[ "${SLURM_JOB_ID:-}" == "$allocation_job_id" ]] || fail "launcher is outside the named allocation"
allocated_nodes="${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}}"
[[ "$allocated_nodes" == "1" ]] || fail "launcher requires one allocated node"
allocated_cpus="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-}}"
[[ "$allocated_cpus" =~ ^[0-9]+$ ]] && (( allocated_cpus >= 32 )) || \
  fail "launcher requires an srun step with at least 32 allocated CPUs"

for value in "$python_bin" "$batch_source" "$exact81_source" "$input_manifest" \
  "$ffmpeg_bin" "$ffprobe_bin" "$output_root"; do
  [[ "$value" == /* ]] || fail "all paths must be absolute"
done
for value in "$batch_source_sha256" "$exact81_source_sha256" \
  "$input_manifest_sha256" "$ffmpeg_sha256" "$ffprobe_sha256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "one SHA-256 argument differs"
done
[[ -x "$python_bin" ]] || fail "frozen Python is not executable"
for value in "$batch_source" "$exact81_source" "$input_manifest"; do
  [[ -f "$value" && ! -L "$value" ]] || fail "one sealed source/input is not a plain file"
done
for value in "$ffmpeg_bin" "$ffprobe_bin"; do
  [[ -x "$value" && -f "$value" && ! -L "$value" ]] || fail "one portable media tool differs"
done
[[ "$(sha256sum "$batch_source" | awk '{print $1}')" == "$batch_source_sha256" ]] || \
  fail "batch source SHA-256 differs"
[[ "$(sha256sum "$exact81_source" | awk '{print $1}')" == "$exact81_source_sha256" ]] || \
  fail "exact81 source SHA-256 differs"
[[ "$(sha256sum "$input_manifest" | awk '{print $1}')" == "$input_manifest_sha256" ]] || \
  fail "sealed input manifest SHA-256 differs"
[[ "$(sha256sum "$ffmpeg_bin" | awk '{print $1}')" == "$ffmpeg_sha256" ]] || \
  fail "portable ffmpeg SHA-256 differs"
[[ "$(sha256sum "$ffprobe_bin" | awk '{print $1}')" == "$ffprobe_sha256" ]] || \
  fail "portable ffprobe SHA-256 differs"

[[ "$output_root" != / && ! -e "$output_root" && ! -L "$output_root" ]] || \
  fail "output root must be fresh, absolute, and non-root"
output_parent="$(dirname -- "$output_root")"
[[ -d "$output_parent" && ! -L "$output_parent" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "$output_parent")"
[[ "$output_root" == "$output_parent/${output_root##*/}" ]] || \
  fail "output root spelling is not canonical"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$scratch_parent" == /* && -d "$scratch_parent" && ! -L "$scratch_parent" && -w "$scratch_parent" ]] || \
  fail "allocation scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-r6-exact28-exact81-media-${SLURM_JOB_ID}.XXXXXX")"
tool_bin="$task_scratch/tool-bin"
mkdir "$tool_bin"

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-r6-exact28-exact81-media-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -L "$tool_bin/ffmpeg" ]]; then unlink "$tool_bin/ffmpeg"; fi
  if [[ -L "$tool_bin/ffprobe" ]]; then unlink "$tool_bin/ffprobe"; fi
  if [[ -d "$tool_bin" && ! -L "$tool_bin" ]]; then rmdir "$tool_bin"; fi
  if [[ -d "$task_scratch" && ! -L "$task_scratch" ]]; then rmdir "$task_scratch"; fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

ln -s -- "$ffmpeg_bin" "$tool_bin/ffmpeg"
ln -s -- "$ffprobe_bin" "$tool_bin/ffprobe"
[[ "$(realpath -e -- "$tool_bin/ffmpeg")" == "$(realpath -e -- "$ffmpeg_bin")" ]] || \
  fail "ffmpeg runtime link differs"
[[ "$(realpath -e -- "$tool_bin/ffprobe")" == "$(realpath -e -- "$ffprobe_bin")" ]] || \
  fail "ffprobe runtime link differs"

method_root="$(dirname -- "$batch_source")"
motive_root="$(realpath -e -- "$method_root/../motive")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$method_root:$motive_root"
export PATH="$tool_bin:/usr/bin:/bin"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_OPENCL_RUNTIME=disabled
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" GPU_DEVICE_ORDINAL=""
[[ "$(sha256sum "$(command -v ffmpeg)" | awk '{print $1}')" == "$ffmpeg_sha256" ]] || \
  fail "clean PATH ffmpeg differs"
[[ "$(sha256sum "$(command -v ffprobe)" | awk '{print $1}')" == "$ffprobe_sha256" ]] || \
  fail "clean PATH ffprobe differs"
ffmpeg -version >/dev/null
ffprobe -version >/dev/null

common=(
  --expected-source-sha256 "$batch_source_sha256"
  --exact81-source "$exact81_source"
  --expected-exact81-source-sha256 "$exact81_source_sha256"
  --input-manifest "$input_manifest"
  --expected-input-manifest-sha256 "$input_manifest_sha256"
  --expected-ffmpeg-sha256 "$ffmpeg_sha256"
  --expected-ffprobe-sha256 "$ffprobe_sha256"
  --output-root "$output_root"
)

if [[ "$mode" == "preflight" ]]; then
  "$python_bin" -B "$batch_source" preflight "${common[@]}"
  [[ -f "$output_root/preflight-receipt.json" && ! -L "$output_root/preflight-receipt.json" ]] || \
    fail "preflight receipt is absent"
  [[ "$(find "$output_root/diagnostics" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 1 ]] || \
    fail "preflight diagnostic count differs"
else
  "$python_bin" -B "$batch_source" full "${common[@]}" --workers 16
  [[ -f "$output_root/aggregate-receipt.json" && ! -L "$output_root/aggregate-receipt.json" ]] || \
    fail "aggregate receipt is absent"
  [[ "$(find "$output_root/diagnostics" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 28 ]] || \
    fail "full exact28 diagnostic count differs"
fi

echo "[saic-r6-exact28-exact81-media] PASS mode=$mode candidates=$([[ "$mode" == full ]] && echo 28 || echo 1) exact81=true full80=true CPU_only=true authority=zero ranking=false selection=false output=$output_root"
