#!/usr/bin/env bash
# Keep two equivalent pending jobs as a scheduler race, but allow only one to
# consume resources.  Intended to run under nohup on an AUH login node.

set -Eeuo pipefail
umask 077

job_a="${1:?usage: auh_watch_slurm_race.sh JOB_A JOB_B RECEIPT [POLL_SECONDS]}"
job_b="${2:?usage: auh_watch_slurm_race.sh JOB_A JOB_B RECEIPT [POLL_SECONDS]}"
receipt="${3:?usage: auh_watch_slurm_race.sh JOB_A JOB_B RECEIPT [POLL_SECONDS]}"
poll_seconds="${4:-20}"

for value in "${job_a}" "${job_b}" "${poll_seconds}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "job IDs and poll interval must be positive decimal integers" >&2
    exit 2
  fi
done
if [[ "${job_a}" == "${job_b}" ]]; then
  echo "race job IDs must differ" >&2
  exit 2
fi
if [[ "${receipt}" != /* || "${receipt}" == "/" || -e "${receipt}" || -L "${receipt}" ]]; then
  echo "receipt must be a new non-root absolute path" >&2
  exit 2
fi
if [[ ! -d "${receipt%/*}" || ! -w "${receipt%/*}" ]]; then
  echo "receipt parent is unavailable" >&2
  exit 2
fi

state_of() {
  sacct -j "$1" -X -n -o State | awk 'NF {sub(/[+].*$/, "", $1); print $1; exit}'
}

publish() {
  local winner="$1"
  local loser="$2"
  local winner_state="$3"
  local loser_state="$4"
  local temporary="${receipt}.tmp.$$"
  {
    printf 'schema=motive-slurm-equivalent-job-race-v1\n'
    printf 'winner_job_id=%s\n' "${winner}"
    printf 'loser_job_id=%s\n' "${loser}"
    printf 'winner_state=%s\n' "${winner_state}"
    printf 'loser_state=%s\n' "${loser_state}"
    printf 'resolved_at_utc=%s\n' "$(date -u +%FT%TZ)"
  } >"${temporary}"
  chmod 0400 "${temporary}"
  mv "${temporary}" "${receipt}"
}

while true; do
  state_a="$(state_of "${job_a}")"
  state_b="$(state_of "${job_b}")"
  printf '%s job_%s=%s job_%s=%s\n' \
    "$(date -u +%FT%TZ)" "${job_a}" "${state_a}" "${job_b}" "${state_b}"

  if [[ "${state_a}" == "RUNNING" && "${state_b}" == "PENDING" ]]; then
    scancel "${job_b}"
    publish "${job_a}" "${job_b}" "${state_a}" "$(state_of "${job_b}")"
    exit 0
  fi
  if [[ "${state_b}" == "RUNNING" && "${state_a}" == "PENDING" ]]; then
    scancel "${job_a}"
    publish "${job_b}" "${job_a}" "${state_b}" "$(state_of "${job_a}")"
    exit 0
  fi
  if [[ "${state_a}" == "RUNNING" && "${state_b}" == "RUNNING" ]]; then
    # The caller orders the preferred job first.  If both cross RUNNING
    # between polls, retain that preferred implementation and stop the other.
    scancel "${job_b}"
    publish "${job_a}" "${job_b}" "${state_a}" "$(state_of "${job_b}")"
    exit 0
  fi
  # A short job can complete between polls.  Completion is also sufficient
  # evidence to cancel only an equivalent job that is still pending.
  if [[ "${state_a}" == "COMPLETED" && "${state_b}" == "PENDING" ]]; then
    scancel "${job_b}"
    publish "${job_a}" "${job_b}" "${state_a}" "$(state_of "${job_b}")"
    exit 0
  fi
  if [[ "${state_b}" == "COMPLETED" && "${state_a}" == "PENDING" ]]; then
    scancel "${job_a}"
    publish "${job_b}" "${job_a}" "${state_b}" "$(state_of "${job_a}")"
    exit 0
  fi
  # If neither job remains pending/running, there is no future duplicate to
  # suppress.  Leave a non-success status for the caller to investigate.
  if [[ "${state_a}" != "PENDING" && "${state_a}" != "RUNNING" && \
        "${state_b}" != "PENDING" && "${state_b}" != "RUNNING" ]]; then
    echo "both race jobs became terminal without a usable winner" >&2
    exit 1
  fi
  sleep "${poll_seconds}"
done
