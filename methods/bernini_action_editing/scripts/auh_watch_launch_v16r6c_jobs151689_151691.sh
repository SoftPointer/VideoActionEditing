#!/usr/bin/env bash
set -Eeuo pipefail

readonly release=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/v16r6c_two_sided_debug32_20260824_r1/release
readonly run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/v16r6c_two_sided_prefix32_run_20260824_r1
readonly launcher="${release}/auh_launch_v16r6_ab_debug32.sh"
readonly jobs=(151689 151690 151691)
readonly names=(v16r6c-dbg32-hold v16r6c-gt-dbg32 v16r6c-m128-dbg32)

fail() {
  echo "[v16r6c-watch] ERROR: $*" >&2
  exit 3
}

job_state() {
  local job="$1"
  local value=""
  value="$(squeue --jobs="${job}" --noheader --format='%T' | head -n 1)"
  if [[ -z "${value}" ]]; then
    value="$(sacct --jobs="${job}" --noheader --starttime=2026-08-24 --format=State --parsable2 | head -n 1 | cut -d'|' -f1)"
  fi
  printf '%s' "${value}"
}

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) fail "watcher must run on the AUH login host" ;;
esac
[[ -x "${launcher}" && ! -L "${launcher}" ]] || fail "authenticated C launcher is absent"

last=""
winner=""
winner_name=""
while [[ -z "${winner}" ]]; do
  states=()
  running=()
  running_names=()
  live=false
  for index in "${!jobs[@]}"; do
    state="$(job_state "${jobs[${index}]}")"
    states+=("${jobs[${index}]}=${state:-absent}")
    if [[ "${state}" == "RUNNING" ]]; then
      running+=("${jobs[${index}]}")
      running_names+=("${names[${index}]}")
    elif [[ "${state}" == "PENDING" || "${state}" == "CONFIGURING" ]]; then
      live=true
    fi
  done
  joined="${states[*]}"
  if [[ "${joined}" != "${last}" ]]; then
    echo "[v16r6c-watch] ${joined} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    last="${joined}"
  fi
  if [[ "${#running[@]}" -gt 0 ]]; then
    winner="${running[0]}"
    winner_name="${running_names[0]}"
    winner_start="$(squeue --jobs="${winner}" --noheader --format='%S' | head -n 1)"
    for index in "${!running[@]}"; do
      candidate="${running[${index}]}"
      candidate_start="$(squeue --jobs="${candidate}" --noheader --format='%S' | head -n 1)"
      if [[ "${candidate_start}" < "${winner_start}" ]]; then
        winner="${candidate}"
        winner_name="${running_names[${index}]}"
        winner_start="${candidate_start}"
      fi
    done
    break
  fi
  [[ "${live}" == true ]] || fail "all C candidates ended before launch"
  sleep 5
done

node="$(squeue --jobs="${winner}" --noheader --format='%N' | head -n 1)"
actual_name="$(squeue --jobs="${winner}" --noheader --format='%j' | head -n 1)"
[[ "${node}" =~ ^auh[0-9]+-[0-9a-z-]+$ ]] || fail "winner node syntax differs"
[[ "${actual_name}" == "${winner_name}" ]] || fail "winner job name differs"
losers=()
for candidate in "${jobs[@]}"; do
  [[ "${candidate}" == "${winner}" ]] || losers+=("${candidate}")
done
echo "[v16r6c-watch] winner=${winner}/${node}; cancelling duplicates=${losers[*]} before training"
scancel "${losers[@]}"
for loser in "${losers[@]}"; do
  for _ in $(seq 1 120); do
    loser_state="$(job_state "${loser}")"
    [[ "${loser_state}" != "PENDING" && "${loser_state}" != "CONFIGURING" && "${loser_state}" != "RUNNING" ]] && break
    sleep 0.5
  done
  [[ "${loser_state}" != "PENDING" && "${loser_state}" != "CONFIGURING" && "${loser_state}" != "RUNNING" ]] || fail "duplicate ${loser} did not cancel"
done

bash "${launcher}" \
  --release-root "${release}" \
  --run-root "${run_root}" \
  --variant c \
  --job-id "${winner}" \
  --node "${node}"
echo "[v16r6c-watch] C exact32 launcher passed; releasing holder ${winner}"
scancel "${winner}"
