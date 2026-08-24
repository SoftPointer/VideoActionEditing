#!/usr/bin/env bash
# Continue a recoverable Wan2.2 exact-eight run one sample at a time inside an
# existing eight-GPU allocation. Stop before starting another sample once the
# separately sized formal job begins running.

set -Eeuo pipefail
umask 077

existing_job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
existing_node="${MOTIVE_EXISTING_SLURM_NODE:?set MOTIVE_EXISTING_SLURM_NODE}"
formal_job_id="${MOTIVE_WAN22_FORMAL_JOB_ID:?set MOTIVE_WAN22_FORMAL_JOB_ID}"
source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
generation_manifest="${MOTIVE_GOKU_ACTION_GENERATION_MANIFEST:?set MOTIVE_GOKU_ACTION_GENERATION_MANIFEST}"
signed_release="${MOTIVE_WAN22_SIGNED_RELEASE:?set MOTIVE_WAN22_SIGNED_RELEASE}"
output_root="${MOTIVE_WAN22_OUTPUT_ROOT:?set MOTIVE_WAN22_OUTPUT_ROOT}"
node_cache_root="${MOTIVE_WAN22_NODE_CACHE_ROOT:?set MOTIVE_WAN22_NODE_CACHE_ROOT}"
target_count="${MOTIVE_WAN22_TARGET_COMMITTED:-8}"
step_cpus="${MOTIVE_WAN22_STEP_CPUS:-60}"
step_mem="${MOTIVE_WAN22_STEP_MEM:-250G}"

for value in "${existing_job_id}" "${formal_job_id}" "${target_count}" "${step_cpus}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "job IDs, target count, and step CPUs must be positive integers" >&2
    exit 2
  fi
done
if (( target_count > 8 )); then
  echo "target count may not exceed the signed exact-eight release" >&2
  exit 2
fi
if [[ ! "${existing_node}" =~ ^auh[0-9A-Za-z-]+$ ]]; then
  echo "existing node name is invalid" >&2
  exit 2
fi
if [[ ! "${step_mem}" =~ ^[1-9][0-9]*[GM]$ ]]; then
  echo "step memory request is invalid" >&2
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
  "${signed_release}" \
  "${output_root}/run_contract.json"; do
  if [[ -L "${path}" || ! -f "${path}" ]]; then
    echo "required immutable or resumable input is unavailable: ${path}" >&2
    exit 2
  fi
done
if [[ "${output_root}" != /* || "${output_root}" == "/" || -L "${output_root}" || ! -d "${output_root}" ]]; then
  echo "output root must be an existing non-symlink absolute directory" >&2
  exit 2
fi
if ! grep -Fq '"max_new_samples_per_allocation": 1' "${output_root}/run_contract.json"; then
  echo "existing run contract is not the one-sample resumable contract" >&2
  exit 2
fi

job_record="$(scontrol show job -o "${existing_job_id}")"
for required in "JobState=RUNNING" "NodeList=${existing_node}" "gres/gpu:mi210=8"; do
  if [[ "${job_record}" != *"${required}"* ]]; then
    echo "existing allocation binding differs: ${required}" >&2
    exit 2
  fi
done

committed_count() {
  find "${output_root}/samples" -mindepth 2 -maxdepth 2 -name result.json \
    -type f 2>/dev/null | wc -l | tr -d '[:space:]'
}

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

while true; do
  before="$(committed_count)"
  if [[ ! "${before}" =~ ^[0-9]+$ ]]; then
    echo "committed sample count is invalid: ${before}" >&2
    exit 2
  fi
  if (( before >= target_count )); then
    echo "[existing-job-wan22-continue] target reached committed=${before}"
    exit 0
  fi

  formal_state="$(squeue -j "${formal_job_id}" -h -o '%T' | head -n 1)"
  case "${formal_state}" in
    RUNNING|COMPLETING)
      echo "[existing-job-wan22-continue] formal job ${formal_job_id} is ${formal_state}; stopping before another sample"
      exit 0
      ;;
    PENDING)
      ;;
    "")
      formal_terminal="$(sacct -j "${formal_job_id}" -X -n -P -o State | awk -F'|' 'NF {print $1; exit}')"
      case "${formal_terminal}" in
        COMPLETED*)
          echo "[existing-job-wan22-continue] formal job completed; stopping"
          exit 0
          ;;
        CANCELLED*|FAILED*|TIMEOUT*|NODE_FAIL*)
          echo "[existing-job-wan22-continue] formal backup terminal=${formal_terminal}; continuing existing allocation"
          ;;
        *)
          echo "formal job state is unavailable or unexpected: ${formal_terminal}" >&2
          exit 2
          ;;
      esac
      ;;
    *)
      echo "formal job state is unexpected: ${formal_state}" >&2
      exit 2
      ;;
  esac

  if squeue -s -j "${existing_job_id}" -h -o '%i' | grep -Eq '[.][0-9]+$'; then
    echo "existing allocation acquired another numbered step" >&2
    exit 2
  fi
  echo "[existing-job-wan22-continue] launching committed_before=${before} target=${target_count} formal_state=${formal_state}"
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

  after="$(committed_count)"
  if [[ ! "${after}" =~ ^[0-9]+$ ]] || (( after != before + 1 )); then
    echo "one-sample resume count differs: before=${before} after=${after}" >&2
    exit 2
  fi
  echo "[existing-job-wan22-continue] committed_after=${after}"
done
