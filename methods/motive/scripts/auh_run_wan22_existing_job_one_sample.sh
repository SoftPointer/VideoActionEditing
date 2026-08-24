#!/usr/bin/env bash
# Run one full-quality Wan2.2 sample in an already allocated eight-GPU job.
# This is a bounded compatibility smoke; production exact-eight output uses
# the separately sized 1 TiB batch allocation.

set -Eeuo pipefail
umask 077

existing_job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
existing_node="${MOTIVE_EXISTING_SLURM_NODE:?set MOTIVE_EXISTING_SLURM_NODE}"
source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
generation_manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
signed_release="${MOTIVE_WAN22_SIGNED_RELEASE:?set MOTIVE_WAN22_SIGNED_RELEASE}"
output_root="${MOTIVE_WAN22_OUTPUT_ROOT:?set MOTIVE_WAN22_OUTPUT_ROOT}"
node_cache_root="${MOTIVE_WAN22_NODE_CACHE_ROOT:?set MOTIVE_WAN22_NODE_CACHE_ROOT}"
step_cpus="${MOTIVE_WAN22_STEP_CPUS:-60}"
step_mem="${MOTIVE_WAN22_STEP_MEM:-250G}"

if [[ ! "${existing_job_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "existing job ID must be a positive integer" >&2
  exit 2
fi
if [[ ! "${existing_node}" =~ ^auh[0-9A-Za-z-]+$ ]]; then
  echo "existing node name is invalid" >&2
  exit 2
fi
if [[ ! "${step_cpus}" =~ ^[1-9][0-9]*$ || ! "${step_mem}" =~ ^[1-9][0-9]*[GM]$ ]]; then
  echo "step CPU or memory request is invalid" >&2
  exit 2
fi
if [[ ! "${node_cache_root}" =~ ^/tmp/motive-wan22-[A-Za-z0-9._-]+$ ]]; then
  echo "node cache root must be an isolated /tmp/motive-wan22-* path" >&2
  exit 2
fi
for path in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_parallel_full.sbatch" \
  "${generation_manifest}" \
  "${signed_release}"; do
  if [[ -L "${path}" || ! -f "${path}" ]]; then
    echo "required immutable input is unavailable: ${path}" >&2
    exit 2
  fi
done
if [[ "${output_root}" != /* || "${output_root}" == "/" || -e "${output_root}" || -L "${output_root}" ]]; then
  echo "output root must be a new non-root absolute path" >&2
  exit 2
fi
if [[ ! -d "${output_root%/*}" || ! -w "${output_root%/*}" ]]; then
  echo "output parent is unavailable" >&2
  exit 2
fi

job_record="$(scontrol show job -o "${existing_job_id}")"
for required in "JobState=RUNNING" "NodeList=${existing_node}" "gres/gpu:mi210=8"; do
  if [[ "${job_record}" != *"${required}"* ]]; then
    echo "existing allocation binding differs: ${required}" >&2
    exit 2
  fi
done
if squeue -s -j "${existing_job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; then
  echo "existing allocation already has a numbered Slurm step" >&2
  exit 2
fi

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
  /usr/bin/env bash -c 'umask 077; mkdir -p "$1"' \
    motive-cache-setup "${node_cache_root}"

echo "[existing-job-wan22] job=${existing_job_id} node=${existing_node} max_new_samples=1"
srun \
  --jobid="${existing_job_id}" \
  --nodelist="${existing_node}" \
  --nodes=1 \
  --exclusive \
  --exact \
  --kill-on-bad-exit=1 \
  --ntasks=1 \
  --cpus-per-task="${step_cpus}" \
  --mem="${step_mem}" \
  --gpus-per-task=8 \
  --gpu-bind=none \
  /usr/bin/env \
    SLURM_EXPORT_ENV=ALL \
    SLURM_TMPDIR="${node_cache_root}" \
    MOTIVE_WAN22_MAX_NEW_SAMPLES=1 \
    bash "${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_parallel_full.sbatch"

committed_count="$(find "${output_root}/samples" -mindepth 2 -maxdepth 2 -name result.json -type f | wc -l | tr -d '[:space:]')"
if [[ "${committed_count}" != "1" ]]; then
  echo "one-sample smoke committed count differs: ${committed_count}" >&2
  exit 2
fi
echo "[existing-job-wan22] one full-quality sample committed"
