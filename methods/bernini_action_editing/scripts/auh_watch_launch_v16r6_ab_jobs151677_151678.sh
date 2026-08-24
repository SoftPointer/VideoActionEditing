#!/usr/bin/env bash
set -Eeuo pipefail

readonly release=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/v16r6_ab_debug32_20260824_r4/release
readonly root_a=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/v16r6a_lr1e7_prefix32_20260824_r1
readonly root_b=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/v16r6b_route_qk_prefix32_20260824_r1
readonly job_a=151677
readonly job_b=151678
readonly job_a_gt=151681
readonly job_b_gt=151682
readonly job_a_m128=151685
readonly job_b_m128=151686
readonly launcher="${release}/auh_launch_v16r6_ab_debug32.sh"

fail() {
  echo "[v16r6-ab-watch] ERROR: $*" >&2
  exit 3
}

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) fail "watcher must run on the AUH login host" ;;
esac
[[ -x "${launcher}" && ! -L "${launcher}" ]] || fail "authenticated launcher is absent"

job_state() {
  local job="$1"
  local value=""
  value="$(squeue --jobs="${job}" --noheader --format='%T' | head -n 1)"
  if [[ -z "${value}" ]]; then
    value="$(sacct --jobs="${job}" --noheader --starttime=2026-08-24 --format=State --parsable2 | head -n 1 | cut -d'|' -f1)"
  fi
  printf '%s' "${value}"
}

launch_first_ready() {
  local variant="$1"
  local bg_job="$2"
  local gt_job="$3"
  local m128_job="$4"
  local run_root="$5"
  local bg_name="v16r6${variant}-dbg32-hold"
  local gt_name="v16r6${variant}-gt-dbg32"
  local m128_name="v16r6${variant}-m128-dbg32"
  local last_pair=""
  local bg_state=""
  local gt_state=""
  local m128_state=""
  local pair=""
  local job=""
  local loser=""
  local losers=()
  local running_jobs=()
  local running_names=()
  local expected_name=""
  local node=""
  local name=""
  while true; do
    bg_state="$(job_state "${bg_job}")"
    gt_state="$(job_state "${gt_job}")"
    m128_state="$(job_state "${m128_job}")"
    pair="bg256=${bg_state:-absent},gt256=${gt_state:-absent},bg128=${m128_state:-absent}"
    if [[ "${pair}" != "${last_pair}" ]]; then
      echo "[v16r6-ab-watch] variant=${variant} ${pair} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      last_pair="${pair}"
    fi

    running_jobs=()
    running_names=()
    if [[ "${bg_state}" == "RUNNING" ]]; then
      running_jobs+=("${bg_job}"); running_names+=("${bg_name}")
    fi
    if [[ "${gt_state}" == "RUNNING" ]]; then
      running_jobs+=("${gt_job}"); running_names+=("${gt_name}")
    fi
    if [[ "${m128_state}" == "RUNNING" ]]; then
      running_jobs+=("${m128_job}"); running_names+=("${m128_name}")
    fi
    if [[ "${#running_jobs[@]}" -gt 0 ]]; then
      job="${running_jobs[0]}"
      expected_name="${running_names[0]}"
      winner_start="$(squeue --jobs="${job}" --noheader --format='%S' | head -n 1)"
      for index in "${!running_jobs[@]}"; do
        candidate="${running_jobs[${index}]}"
        candidate_start="$(squeue --jobs="${candidate}" --noheader --format='%S' | head -n 1)"
        if [[ "${candidate_start}" < "${winner_start}" ]]; then
          job="${candidate}"
          expected_name="${running_names[${index}]}"
          winner_start="${candidate_start}"
        fi
      done
      break
    fi
    bg_live=false
    gt_live=false
    m128_live=false
    [[ "${bg_state}" == "PENDING" || "${bg_state}" == "CONFIGURING" ]] && bg_live=true
    [[ "${gt_state}" == "PENDING" || "${gt_state}" == "CONFIGURING" ]] && gt_live=true
    [[ "${m128_state}" == "PENDING" || "${m128_state}" == "CONFIGURING" ]] && m128_live=true
    [[ "${bg_live}" == true || "${gt_live}" == true || "${m128_live}" == true ]] || fail "all variant-${variant} candidates ended before launch (${pair})"
    sleep 5
  done

  node="$(squeue --jobs="${job}" --noheader --format='%N' | head -n 1)"
  name="$(squeue --jobs="${job}" --noheader --format='%j' | head -n 1)"
  [[ "${node}" =~ ^auh[0-9]+-[0-9a-z-]+$ ]] || fail "job ${job} node syntax differs"
  [[ "${name}" == "${expected_name}" ]] || fail "job ${job} name differs"
  losers=()
  for candidate in "${bg_job}" "${gt_job}" "${m128_job}"; do
    [[ "${candidate}" == "${job}" ]] || losers+=("${candidate}")
  done
  echo "[v16r6-ab-watch] variant=${variant} winner=${job}/${node}; cancelling duplicates=${losers[*]} before training"
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
    --variant "${variant}" \
    --job-id "${job}" \
    --node "${node}"
  echo "[v16r6-ab-watch] variant=${variant} job=${job} exact32 launcher passed; releasing holder"
  scancel "${job}"
}

pid_a=""
pid_b=""
cleanup() {
  local status="$?"
  set +e
  trap - EXIT
  for pid in "${pid_a:-}" "${pid_b:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT

launch_first_ready a "${job_a}" "${job_a_gt}" "${job_a_m128}" "${root_a}" &
pid_a="$!"
launch_first_ready b "${job_b}" "${job_b_gt}" "${job_b_m128}" "${root_b}" &
pid_b="$!"

set +e
wait "${pid_a}"
status_a="$?"
wait "${pid_b}"
status_b="$?"
set -e
pid_a=""
pid_b=""
[[ "${status_a}" -eq 0 && "${status_b}" -eq 0 ]] || fail "A/B launcher status A=${status_a} B=${status_b}"
echo "[v16r6-ab-watch] both exact32 diagnostics completed and holders released"
