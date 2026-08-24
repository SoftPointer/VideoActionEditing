#!/usr/bin/env bash
set -Eeuo pipefail

launcher="${BERNINI_SSR_LAUNCHER:?set absolute BERNINI_SSR_LAUNCHER}"
run_root="${BERNINI_SSR_RUN_ROOT:?set fresh absolute BERNINI_SSR_RUN_ROOT}"

[[ "${launcher}" == /* && -f "${launcher}" && ! -L "${launcher}" ]] || {
  echo "invalid source-self one-step launcher" >&2
  exit 2
}
[[ "${run_root}" == /* && ! -e "${run_root}" && ! -L "${run_root}" ]] || {
  echo "source-self run root must be a fresh absolute path" >&2
  exit 2
}

job_id="$(sbatch --parsable \
  --export="ALL,BERNINI_SSR_RUN_ROOT=${run_root}" \
  "${launcher}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
  echo "invalid one-step job id: ${job_id}" >&2
  exit 2
}

# Deliberately no afterok dependency: 16/64-step training and rho>0 require a
# separately authored, hash-bound scientific-gate decision.
printf 'one_step_job=%s\none_step_output=%s\nlong_training_job=NOT_SUBMITTED\nscientific_gate=CLOSED\n' \
  "${job_id}" "${run_root}"
