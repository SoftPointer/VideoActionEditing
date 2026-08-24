#!/usr/bin/env bash
# Build the broadened pre-Qwen pool, then resume a dual-4GPU Qwen3-VL audit
# inside one already allocated and independently verified-idle MI210 node.

set -Eeuo pipefail
umask 077

existing_job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
existing_node="${MOTIVE_EXISTING_SLURM_NODE:?set MOTIVE_EXISTING_SLURM_NODE}"
source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
input_fused="${MOTIVE_GOKU_ACTION_INPUT_FUSED:?set MOTIVE_GOKU_ACTION_INPUT_FUSED}"
video_root="${MOTIVE_GOKU_ACTION_VIDEO_ROOT:?set MOTIVE_GOKU_ACTION_VIDEO_ROOT}"
run_root="${MOTIVE_GOKU_ACTION_SCALE512_ROOT:?set MOTIVE_GOKU_ACTION_SCALE512_ROOT}"
qwen_model="${MOTIVE_GOKU_ACTION_QWEN_MODEL:?set MOTIVE_GOKU_ACTION_QWEN_MODEL}"
python_bin="${MOTIVE_GOKU_ACTION_PYTHON_BIN:?set MOTIVE_GOKU_ACTION_PYTHON_BIN}"

prefilter_output="${run_root}/prefilter"
selected="${prefilter_output}/selected.jsonl"
qwen_root="${run_root}/qwen8"
# This legacy-v8 final output is only an audit checkpoint. The scalable v2
# finalizer will independently revalidate the complete Qwen shards later.
legacy_final="${run_root}/final_v8_legacy_cap128"
controller_receipt="${run_root}/qwen_controller_complete"
code_root="${source_snapshot}/methods/motive"
cache_root="/tmp/motive-goku-action-scale512-${existing_job_id}"

fail() {
  echo "[scale512-curation] $*" >&2
  exit 2
}

for value in "${existing_job_id}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "job ID must be positive"
done
[[ "${existing_node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node name"

for path in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/goku_action_anchor_prefilter.py" \
  "${code_root}/motive/goku_action_anchor_qwen.py" \
  "${code_root}/motive/goku_action_anchor_finalize.py" \
  "${input_fused}" \
  "${qwen_model}/config.json" \
  "${python_bin}"; do
  [[ ! -L "${path}" && -e "${path}" ]] || fail "missing regular input: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable: ${python_bin}"
[[ ! -L "${video_root}" && -d "${video_root}" ]] || fail "invalid video root"
[[ ! -L "${run_root}" && -d "${run_root}" ]] || fail "run root must exist"

job_record="$(scontrol show job -o "${existing_job_id}")"
for required in \
  "JobState=RUNNING" \
  "NodeList=${existing_node}" \
  "gres/gpu:mi210=8"; do
  [[ "${job_record}" == *"${required}"* ]] || fail "allocation differs: ${required}"
done
if squeue -s -j "${existing_job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; then
  fail "allocation already has a numbered Slurm step"
fi

if [[ -e "${prefilter_output}" || -L "${prefilter_output}" ]]; then
  [[ ! -L "${prefilter_output}" && -d "${prefilter_output}" ]] \
    || fail "existing prefilter path is not a plain directory"
  [[ ! -L "${selected}" && -s "${selected}" ]] \
    || fail "existing prefilter is incomplete: ${prefilter_output}"
  echo "[scale512-curation] reusing committed prefilter: ${prefilter_output}"
else
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="${code_root}:${source_snapshot}"
  export TOKENIZERS_PARALLELISM=false
  export SLURM_EXPORT_ENV=ALL
  echo "[scale512-curation] starting strict geometry prefilter on ${existing_node}"
  srun \
    --jobid="${existing_job_id}" \
    --nodelist="${existing_node}" \
    --nodes=1 \
    --exclusive \
    --exact \
    --kill-on-bad-exit=1 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=128G \
    env \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONNOUSERSITE=1 \
      PYTHONPATH="${PYTHONPATH}" \
      TOKENIZERS_PARALLELISM=false \
      "${python_bin}" -m motive.goku_action_anchor_prefilter \
        --input-fused "${input_fused}" \
        --video-root "${video_root}" \
        --output-dir "${prefilter_output}" \
        --sample-size 1280 \
        --max-per-family 512 \
        --workers 32 \
        --max-spatial-energy-entropy 1.0
fi

selected_rows="$(wc -l < "${selected}")"
[[ "${selected_rows}" =~ ^[0-9]+$ ]] || fail "invalid selected row count"
(( selected_rows >= 512 )) || fail "prefilter yielded fewer than 512 rows"
echo "[scale512-curation] prefilter rows=${selected_rows}; starting Qwen3-VL-32B"

receipt_value() {
  local key="$1"
  awk -F= -v key="${key}" '$1 == key {print substr($0, length(key) + 2)}' \
    "${controller_receipt}"
}

if [[ -e "${controller_receipt}" || -L "${controller_receipt}" ]]; then
  [[ ! -L "${controller_receipt}" && -s "${controller_receipt}" ]] \
    || fail "existing controller receipt is invalid"
  [[ "$(receipt_value schema)" == "motive-goku-action-scale512-controller-v1" ]] \
    || fail "controller receipt schema differs"
  [[ "$(receipt_value slurm_job_id)" == "${existing_job_id}" ]] \
    || fail "controller receipt job differs"
  [[ "$(receipt_value slurm_node)" == "${existing_node}" ]] \
    || fail "controller receipt node differs"
  [[ "$(receipt_value selected_rows)" == "${selected_rows}" ]] \
    || fail "controller receipt selected row count differs"
  selected_sha256="$(sha256sum "${selected}" | awk '{print $1}')"
  [[ "$(receipt_value selected_sha256)" == "${selected_sha256}" ]] \
    || fail "controller receipt selected SHA differs"
  [[ ! -L "${legacy_final}" && -d "${legacy_final}" ]] \
    || fail "completed controller lacks a plain legacy final directory"
  for artifact in \
    review_candidates.jsonl \
    proposed_128.jsonl \
    reserve_32.jsonl \
    generation_manifest.jsonl \
    summary.json \
    done.json; do
    [[ ! -L "${legacy_final}/${artifact}" && -f "${legacy_final}/${artifact}" ]] \
      || fail "completed controller lacks legacy artifact: ${artifact}"
  done
  legacy_manifest_sha256="$(sha256sum "${legacy_final}/generation_manifest.jsonl" | awk '{print $1}')"
  [[ "$(receipt_value legacy_generation_manifest_sha256)" == "${legacy_manifest_sha256}" ]] \
    || fail "controller receipt legacy manifest SHA differs"
  shopt -s nullglob
  completed_qwen_shards=("${qwen_root}"/qwen_shard_*.jsonl)
  completed_qwen_receipts=("${qwen_root}"/qwen_shard_*.receipt.json)
  (( ${#completed_qwen_shards[@]} == 8 )) \
    || fail "completed controller does not have eight Qwen shards"
  (( ${#completed_qwen_receipts[@]} == 8 )) \
    || fail "completed controller does not have eight Qwen receipts"
  echo "[scale512-curation] verified existing completed controller receipt"
  exit 0
fi

if [[ -e "${legacy_final}" || -L "${legacy_final}" ]]; then
  fail "legacy final exists without a completion receipt; refusing stale reuse"
fi

if [[ -L "${qwen_root}" || -e "${qwen_root}" && ! -d "${qwen_root}" ]]; then
  fail "Qwen output must be a non-symlink directory: ${qwen_root}"
fi
if [[ ! -d "${qwen_root}" ]]; then
  mkdir "${qwen_root}"
fi

cache_directories=()
for worker in a b; do
  cache_directories+=(
    "${cache_root}/${worker}/tmp"
    "${cache_root}/${worker}/xdg"
    "${cache_root}/${worker}/torch"
    "${cache_root}/${worker}/torch-kernels"
    "${cache_root}/${worker}/miopen"
    "${cache_root}/${worker}/miopen-custom"
  )
done
srun \
  --jobid="${existing_job_id}" \
  --nodelist="${existing_node}" \
  --nodes=1 \
  --exclusive \
  --exact \
  --kill-on-bad-exit=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=1G \
  /usr/bin/env bash -c 'umask 077; mkdir -p "$@"' \
    motive-cache-setup "${cache_directories[@]}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${code_root}:${source_snapshot}"
export SLURM_EXPORT_ENV=ALL
cd "${code_root}"

run_worker() {
  local worker="$1"
  local sequential_shards="$2"
  local worker_cache="${cache_root}/${worker}"
  srun \
    --jobid="${existing_job_id}" \
    --nodelist="${existing_node}" \
    --nodes=1 \
    --exclusive \
    --exact \
    --kill-on-bad-exit=1 \
    --ntasks=1 \
    --cpus-per-task=28 \
    --mem=110G \
    --gpus-per-task=4 \
    --gpu-bind=none \
    env \
      TMPDIR="${worker_cache}/tmp" \
      XDG_CACHE_HOME="${worker_cache}/xdg" \
      TORCH_HOME="${worker_cache}/torch" \
      PYTORCH_KERNEL_CACHE_PATH="${worker_cache}/torch-kernels" \
      MIOPEN_USER_DB_PATH="${worker_cache}/miopen" \
      MIOPEN_CUSTOM_CACHE_DIR="${worker_cache}/miopen-custom" \
      "${python_bin}" \
        -m motive.goku_action_anchor_qwen \
        --input "${selected}" \
        --output "${qwen_root}" \
        --model "${qwen_model}" \
        --all-shards-sequential \
        --sequential-shards "${sequential_shards}" \
        --shard-index 0 \
        --num-shards 8 \
        --nframes 12 \
        --max-pixels 589824 \
        --max-new-tokens 1536 \
        --attn-implementation sdpa \
        --allow-errors \
        --resume
}

run_worker a "0,1,2,3" &
worker_a_pid=$!
run_worker b "4,5,6,7" &
worker_b_pid=$!
worker_status=0
wait "${worker_a_pid}" || worker_status=$?
wait "${worker_b_pid}" || worker_status=$?
(( worker_status == 0 )) || fail "dual-4GPU Qwen worker failed: ${worker_status}"

srun \
  --jobid="${existing_job_id}" \
  --nodelist="${existing_node}" \
  --nodes=1 \
  --exclusive \
  --exact \
  --kill-on-bad-exit=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  env "${python_bin}" -m motive.goku_action_anchor_finalize \
    --input "${selected}" \
    --qwen-root "${qwen_root}" \
    --output-dir "${legacy_final}" \
    --seed 260730 \
    --allow-partial

temporary_receipt="${controller_receipt}.tmp.$$"
{
  printf 'schema=motive-goku-action-scale512-controller-v1\n'
  printf 'slurm_job_id=%s\n' "${existing_job_id}"
  printf 'slurm_node=%s\n' "${existing_node}"
  printf 'selected_rows=%s\n' "${selected_rows}"
  printf 'selected_sha256=%s\n' "$(sha256sum "${selected}" | awk '{print $1}')"
  printf 'legacy_generation_manifest_sha256=%s\n' "$(sha256sum "${legacy_final}/generation_manifest.jsonl" | awk '{print $1}')"
  printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)"
} >"${temporary_receipt}"
chmod 0400 "${temporary_receipt}"
mv "${temporary_receipt}" "${controller_receipt}"
echo "[scale512-curation] Qwen audit checkpoint complete"
