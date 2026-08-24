#!/usr/bin/env bash
# Run the frozen dual-4GPU Qwen curation worker inside an already allocated,
# verified-idle eight-GPU Slurm job.  This controller itself runs on a login
# node; every material task is an explicit step in the target allocation.

set -Eeuo pipefail
umask 077

existing_job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
existing_node="${MOTIVE_EXISTING_SLURM_NODE:?set MOTIVE_EXISTING_SLURM_NODE}"
source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
selected="${MOTIVE_GOKU_ACTION_SELECTED:?set MOTIVE_GOKU_ACTION_SELECTED}"
qwen_model="${MOTIVE_GOKU_ACTION_QWEN_MODEL:?set MOTIVE_GOKU_ACTION_QWEN_MODEL}"
qwen_root="${MOTIVE_GOKU_ACTION_QWEN_OUTPUT:?set MOTIVE_GOKU_ACTION_QWEN_OUTPUT}"
final_output="${MOTIVE_GOKU_ACTION_FINAL_OUTPUT:?set MOTIVE_GOKU_ACTION_FINAL_OUTPUT}"
python_bin="${MOTIVE_GOKU_ACTION_PYTHON_BIN:?set MOTIVE_GOKU_ACTION_PYTHON_BIN}"
controller_receipt="${MOTIVE_GOKU_ACTION_CONTROLLER_RECEIPT:?set MOTIVE_GOKU_ACTION_CONTROLLER_RECEIPT}"
nframes="${MOTIVE_GOKU_ACTION_QWEN_NFRAMES:-12}"
max_pixels="${MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS:-589824}"
max_new_tokens="${MOTIVE_GOKU_ACTION_QWEN_MAX_NEW_TOKENS:-1536}"
final_seed="${MOTIVE_GOKU_ACTION_FINAL_SEED:-260730}"
allow_partial="${MOTIVE_GOKU_ACTION_ALLOW_PARTIAL:-1}"
worker_cpus="${MOTIVE_GOKU_ACTION_WORKER_CPUS:-28}"
worker_mem="${MOTIVE_GOKU_ACTION_WORKER_MEM:-110G}"
finalizer_cpus="${MOTIVE_GOKU_ACTION_FINALIZER_CPUS:-4}"
finalizer_mem="${MOTIVE_GOKU_ACTION_FINALIZER_MEM:-16G}"
node_cache_root="${MOTIVE_GOKU_ACTION_NODE_CACHE_ROOT:-}"
code_root="${source_snapshot}/methods/motive"

if [[ ! "${existing_job_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "existing Slurm job ID must be a positive decimal integer" >&2
  exit 2
fi
if [[ ! "${existing_node}" =~ ^auh[0-9A-Za-z-]+$ ]]; then
  echo "existing Slurm node has an invalid name" >&2
  exit 2
fi
for value in "${nframes}" "${max_pixels}" "${max_new_tokens}" \
  "${worker_cpus}" "${finalizer_cpus}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "integer resource and inference values must be positive" >&2
    exit 2
  fi
done
for value in "${worker_mem}" "${finalizer_mem}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*[GM]$ ]]; then
    echo "memory values must be positive integers with G or M suffix" >&2
    exit 2
  fi
done
if [[ "${allow_partial}" != "0" && "${allow_partial}" != "1" ]]; then
  echo "allow-partial must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${node_cache_root}" && ! "${node_cache_root}" =~ ^/tmp/motive-goku-action-[A-Za-z0-9._-]+$ ]]; then
  echo "node cache root must be an isolated /tmp/motive-goku-action-* path" >&2
  exit 2
fi

for path in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${code_root}/motive/goku_action_anchor_qwen.py" \
  "${code_root}/motive/goku_action_anchor_finalize.py" \
  "${selected}" \
  "${qwen_model}/config.json" \
  "${python_bin}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "Python is not executable: ${python_bin}" >&2
  exit 2
fi
for output in "${qwen_root}" "${final_output}" "${controller_receipt}"; do
  if [[ "${output}" != /* || "${output}" == "/" || -e "${output}" || -L "${output}" ]]; then
    echo "output must be a new non-root absolute path: ${output}" >&2
    exit 2
  fi
  if [[ ! -d "${output%/*}" || ! -w "${output%/*}" ]]; then
    echo "output parent is unavailable: ${output%/*}" >&2
    exit 2
  fi
done

job_record="$(scontrol show job -o "${existing_job_id}")"
for required in \
  "JobState=RUNNING" \
  "NodeList=${existing_node}" \
  "gres/gpu:mi210=8"; do
  if [[ "${job_record}" != *"${required}"* ]]; then
    echo "existing allocation binding differs: ${required}" >&2
    exit 2
  fi
done
if squeue -s -j "${existing_job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; then
  echo "existing allocation already has a numbered Slurm step" >&2
  exit 2
fi

mkdir "${qwen_root}"
cache_root="${node_cache_root:-${qwen_root%/*}/controller_cache_${existing_job_id}}"
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
if [[ -n "${node_cache_root}" ]]; then
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
else
  mkdir -p "${cache_directories[@]}"
fi

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
    --cpus-per-task="${worker_cpus}" \
    --mem="${worker_mem}" \
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
        --nframes "${nframes}" \
        --max-pixels "${max_pixels}" \
        --max-new-tokens "${max_new_tokens}" \
        --attn-implementation sdpa \
        --allow-errors \
        --resume
}

echo "[existing-job-qwen] job=${existing_job_id} node=${existing_node} input=${selected}"
run_worker a "0,1,2,3" &
worker_a_pid=$!
run_worker b "4,5,6,7" &
worker_b_pid=$!
worker_status=0
wait "${worker_a_pid}" || worker_status=$?
wait "${worker_b_pid}" || worker_status=$?
if (( worker_status != 0 )); then
  echo "a dual-4GPU worker failed: ${worker_status}" >&2
  exit "${worker_status}"
fi

finalizer_args=(
  --input "${selected}"
  --qwen-root "${qwen_root}"
  --output-dir "${final_output}"
  --seed "${final_seed}"
)
if [[ "${allow_partial}" == "1" ]]; then
  finalizer_args+=(--allow-partial)
fi
srun \
  --jobid="${existing_job_id}" \
  --nodelist="${existing_node}" \
  --nodes=1 \
  --exclusive \
  --exact \
  --kill-on-bad-exit=1 \
  --ntasks=1 \
  --cpus-per-task="${finalizer_cpus}" \
  --mem="${finalizer_mem}" \
  env "${python_bin}" -m motive.goku_action_anchor_finalize "${finalizer_args[@]}"

for artifact in review_candidates.jsonl proposed_128.jsonl reserve_32.jsonl generation_manifest.jsonl summary.json done.json; do
  if [[ -L "${final_output}/${artifact}" || ! -f "${final_output}/${artifact}" ]]; then
    echo "missing final artifact: ${artifact}" >&2
    exit 2
  fi
done
temporary_receipt="${controller_receipt}.tmp.$$"
{
  printf 'schema=motive-goku-action-existing-allocation-controller-v1\n'
  printf 'slurm_job_id=%s\n' "${existing_job_id}"
  printf 'slurm_node=%s\n' "${existing_node}"
  printf 'selected_sha256=%s\n' "$(sha256sum "${selected}" | awk '{print $1}')"
  printf 'generation_manifest_sha256=%s\n' "$(sha256sum "${final_output}/generation_manifest.jsonl" | awk '{print $1}')"
  printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)"
} >"${temporary_receipt}"
chmod 0400 "${temporary_receipt}"
mv "${temporary_receipt}" "${controller_receipt}"
echo "[existing-job-qwen] complete"
