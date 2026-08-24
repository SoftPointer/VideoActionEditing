#!/usr/bin/env bash
# One-shot capacity-gated launcher for the exact Full644 R64 profile.
# It never retries and never cancels, releases, requeues, or signals job 141620.

set -Eeuo pipefail
umask 077

readonly job_id=141620
readonly node=auh7-1b-gpu-226
readonly owner=guangyi.chen
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly python_size=31490256
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_r64_job141620_v5
readonly launcher_root="${experiment_root}/launchers/full644-exploratory-r64-job141620-v5"
readonly runner="${launcher_root}/auh_run_full644_exploratory_r64_job141620_v5.sh"
readonly runner_sha256=8c5f70ab249a6b0d38323a9df4fc93e8d84da7b7036d1f8aa306298d74bc6b3c
readonly runner_size=28011
readonly helper="${launcher_root}/full644_exploratory_r64_release_v1.py"
readonly helper_sha256=9c67f56720ec4af0f8bcb8cdddaa768ac1aa4d57a08b095b4214acb6c8c94dda
readonly helper_size=66417
readonly source_archive="${launcher_root}/full644_exploratory_r64_source_v3.tar"
readonly source_archive_sha256=12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828
readonly source_archive_size=1045504
readonly release_parent="${experiment_root}/releases"
readonly release_root="${release_parent}/full644-exploratory-r64-source-12a28ddec997"
readonly attempt_root="${experiment_root}/attempts/prelaunch-capacity-then-full644-v5"
readonly output_parent="${experiment_root}/runs"
readonly capacity_output="${output_parent}/prelaunch-capacity-sft-r64-step1-v5"
readonly full644_output="${output_parent}/full644-r64-reference-dpo-preservation-one-pass-v5"
readonly capacity_log="${attempt_root}/capacity-step.log"
readonly full644_log="${attempt_root}/full644-step.log"
readonly capacity_step_record="${attempt_root}/capacity-step-id.txt"
readonly full644_step_record="${attempt_root}/full644-step-id.txt"
readonly capacity_sacct_row="${attempt_root}/capacity-sacct-row.txt"
readonly capacity_completion="${attempt_root}/capacity-runner-completion.json"
readonly full644_completion="${attempt_root}/full644-runner-completion.json"
readonly capacity_cache_receipt="${attempt_root}/capacity-rank-cache.json"
readonly full644_cache_receipt="${attempt_root}/full644-rank-cache.json"
readonly capacity_gate="${attempt_root}/capacity-gate.json"

active_srun_pid=
pending_signal=
launched_step_id=
srun_launch_armed=0

fail() {
  printf 'Full644 job141620 launcher refused: %s\n' "$*" >&2
  exit 92
}

sha256_file() {
  /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'
}

cleanup_signal() {
  local signal_name="$1" child_status=0 job_pid background_jobs
  pending_signal="${signal_name}"
  trap '' HUP INT TERM
  if [[ "${active_srun_pid}" =~ ^[1-9][0-9]*$ ]] && /bin/kill -0 "${active_srun_pid}" 2>/dev/null; then
    /bin/kill -TERM "${active_srun_pid}" 2>/dev/null || true
    wait "${active_srun_pid}" || child_status=$?
  elif [[ "${srun_launch_armed}" == 1 ]]; then
    background_jobs="$(jobs -pr)" || {
      printf 'launcher could not enumerate armed background jobs\n' >&2
      exit 130
    }
    for job_pid in ${background_jobs}; do
      [[ "${job_pid}" =~ ^[1-9][0-9]*$ ]] || continue
      /bin/kill -TERM "${job_pid}" 2>/dev/null || true
      wait "${job_pid}" || child_status=$?
    done
  fi
  printf 'launcher interrupted by %s; direct srun status=%s\n' "${signal_name}" "${child_status}" >&2
  exit 130
}
trap 'cleanup_signal HUP' HUP
trap 'cleanup_signal INT' INT
trap 'cleanup_signal TERM' TERM

require_plain_file() {
  local path="$1" expected_sha="$2" expected_size="$3" expected_mode="$4" label="$5"
  local observed_size observed_sha observed_envelope
  [[ -f "${path}" && ! -L "${path}" ]] || fail "${label} is not one plain file"
  observed_size="$(/usr/bin/stat -c %s "${path}")" || fail "${label} size query failed"
  [[ "${observed_size}" == "${expected_size}" ]] || fail "${label} size differs"
  observed_envelope="$(/usr/bin/stat -c '%a:%h:%U' "${path}")" || fail "${label} envelope query failed"
  [[ "${observed_envelope}" == "${expected_mode}:1:${owner}" ]] || fail "${label} mode/link/owner differs"
  observed_sha="$(sha256_file "${path}")" || fail "${label} SHA query failed"
  [[ "${observed_sha}" == "${expected_sha}" ]] || fail "${label} SHA differs"
}

require_runner() {
  require_plain_file "${python_bin}" "${python_sha256}" "${python_size}" 755 Python
  require_plain_file "${runner}" "${runner_sha256}" "${runner_size}" 555 runner
  require_plain_file "${helper}" "${helper_sha256}" "${helper_size}" 444 release-helper
  require_plain_file "${source_archive}" "${source_archive_sha256}" "${source_archive_size}" 444 source-archive
}

require_outer_envelope() {
  local self_path self_envelope
  self_path="$0"
  [[ "${self_path}" == "${launcher_root}/auh_launch_full644_exploratory_r64_job141620_v5.sh" \
    && -f "${self_path}" && ! -L "${self_path}" ]] || fail "outer script path differs"
  self_envelope="$(/usr/bin/stat -c '%a:%h:%U' "${self_path}")" || fail "outer script envelope query failed"
  [[ "${self_envelope}" == "555:1:${owner}" ]] || fail "outer script mode/link/owner differs"
}

step_rows() {
  /usr/bin/timeout 15 /usr/bin/squeue --steps -h -j "${job_id}" -o '%i|%N|%u' | \
    /usr/bin/awk '{gsub(/[[:space:]]/,""); if(length($0)) print $0}'
}

require_parent_running() {
  local label="$1" projection
  projection="$(/usr/bin/timeout 15 /usr/bin/squeue -h -j "${job_id}" -o '%T|%N|%u' | \
    /usr/bin/awk '{gsub(/[[:space:]]/,""); if(length($0)) print $0}')" || \
    fail "${label}: holder projection query failed"
  [[ "${projection}" == "RUNNING|${node}|${owner}" ]] || \
    fail "${label}: holder projection differs: ${projection}"
}

require_holder_idle() {
  local label="$1" rows line step_id step_node step_user seen_batch=0 seen_extern=0 row_count=0
  require_parent_running "${label}"
  rows="$(step_rows)" || fail "${label}: Slurm step query failed"
  [[ -n "${rows}" ]] || fail "${label}: holder step projection is empty"
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    (( row_count += 1 ))
    IFS='|' read -r step_id step_node step_user extra <<EOF
${line}
EOF
    [[ -z "${extra:-}" && "${step_node}" == "${node}" && "${step_user}" == "${owner}" ]] || \
      fail "${label}: hostile step row: ${line}"
    case "${step_id}" in
      "${job_id}.batch") seen_batch=1 ;;
      "${job_id}.extern") seen_extern=1 ;;
      "${job_id}."*)
        [[ "${step_id}" =~ ^${job_id}\.[0-9]+$ ]] || fail "${label}: malformed holder step: ${step_id}"
        fail "${label}: active numbered step exists: ${step_id}"
        ;;
      *) fail "${label}: unexpected holder step: ${step_id}" ;;
    esac
  done <<EOF
${rows}
EOF
  [[ "${seen_batch}" == 1 && "${seen_extern}" == 1 && "${row_count}" == 2 ]] || fail "${label}: batch/extern closure differs"
}

await_holder_idle() {
  local label="$1" poll
  for poll in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    if ( require_holder_idle "${label}" ) 2>/dev/null; then
      return 0
    fi
    /bin/sleep 1
  done
  require_holder_idle "${label}"
}

run_step() {
  local mode="$1" log_path="$2" step_record="$3" token rc=0 line_count
  [[ -z "${pending_signal}" ]] || fail "signal arrived before ${mode}"
  [[ ! -e "${log_path}" && ! -L "${log_path}" ]] || fail "${mode} log is not fresh"
  [[ ! -e "${step_record}" && ! -L "${step_record}" ]] || fail "${mode} step record is not fresh"
  require_runner
  srun_launch_armed=1
  /usr/bin/srun --jobid="${job_id}" --nodelist="${node}" --nodes=1 --ntasks=1 \
    --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 --exclusive \
    --exact --kill-on-bad-exit=1 --export=NONE \
    --output="${log_path}" --error="${log_path}" \
    /usr/bin/env PATH=/usr/bin:/bin HOME=/nonexistent/full644-job141620 \
      LC_ALL=C LANG=C BASH_ENV=/dev/null ROCR_VISIBLE_DEVICES=0,1,2,3 \
      HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 \
      GPU_DEVICE_ORDINAL=0,1,2,3 FULL644_STEP_ID_RECORD="${step_record}" \
      /bin/bash "${runner}" "${mode}" &
  active_srun_pid=$!
  srun_launch_armed=0
  wait "${active_srun_pid}" || rc=$?
  active_srun_pid=
  [[ -z "${pending_signal}" ]] || fail "signal arrived during ${mode}"
  (( rc == 0 )) || fail "${mode} srun exited ${rc}"
  line_count="$(/usr/bin/wc -l <"${step_record}" | /usr/bin/tr -d ' ')" || fail "${mode} step record query failed"
  [[ "${line_count}" == 1 ]] || fail "${mode} emitted a non-singleton step identity"
  token="$(/bin/cat "${step_record}")" || fail "${mode} step identity read failed"
  [[ "${token}" =~ ^${job_id}\.[0-9]+$ ]] || fail "${mode} step identity differs: ${token}"
  launched_step_id="${token}"
}

capture_capacity_accounting() {
  local step_id="$1" poll row state exit_code max_rss
  [[ "${step_id}" =~ ^${job_id}\.([0-9]+)$ ]] || fail "capacity step ID differs"
  for poll in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60; do
    row="$(/usr/bin/timeout 15 /usr/bin/sacct -j "${step_id}" -n -P -o JobIDRaw,State,ExitCode,MaxRSS | \
      /usr/bin/awk -F'|' -v id="${step_id}" '$1==id {print; exit}')" || fail "capacity sacct query failed"
    IFS='|' read -r _ state exit_code max_rss _extra <<EOF
${row}
EOF
    if [[ "${state:-}" == COMPLETED && "${exit_code:-}" == 0:0 && -n "${max_rss:-}" ]]; then
      break
    fi
    /bin/sleep 1
  done
  [[ "${state:-}" == COMPLETED && "${exit_code:-}" == 0:0 && -n "${max_rss:-}" ]] || fail "capacity accounting did not settle"
  ( set -o noclobber; /usr/bin/printf '%s\n' "${row}" >"${capacity_sacct_row}" ) || fail "capacity accounting record publication failed"
  /bin/chmod 0400 "${capacity_sacct_row}"
}

audit_capacity_gate() {
  local step_number="$1"
  "${python_bin}" -I -B "${helper}" audit-capacity-gate \
    --training-output "${capacity_output}" --runner-completion "${capacity_completion}" \
    --sacct-row-file "${capacity_sacct_row}" --slurm-step-id "${step_number}" \
    --gate "${capacity_gate}" >/dev/null
}

audit_runner_completion() {
  local mode="$1" output="$2" receipt="$3" cache_receipt="$4" step_number="$5"
  "${python_bin}" -I -B "${helper}" audit-runner-completion \
    --mode "${mode}" --output "${output}" --receipt "${receipt}" \
    --cache-receipt "${cache_receipt}" --slurm-job-id "${job_id}" \
    --slurm-step-id "${step_number}" --node "${node}" >/dev/null
}

launcher_user="$(/usr/bin/id -un)" || fail "launcher user query failed"
[[ "${launcher_user}" == "${owner}" ]] || fail "launcher user differs"
[[ -d "${launcher_root}" && ! -L "${launcher_root}" ]] || fail "launcher root differs"
require_outer_envelope
require_runner
if [[ $# == 1 && "$1" == --verify-local-bundle-only ]]; then
  /bin/bash -n "${runner}" || fail "runner syntax check failed"
  "${python_bin}" -I -B "${helper}" audit-source-archive --archive "${source_archive}" >/dev/null
  require_runner
  printf 'PASS local_bundle_only=true network_used=false launch_performed=false\n'
  exit 0
fi
[[ $# == 0 ]] || fail "launcher takes no arguments except --verify-local-bundle-only"
require_holder_idle pre-create
[[ ! -e "${attempt_root}" && ! -L "${attempt_root}" ]] || fail "attempt root is not fresh"
[[ ! -e "${release_root}" && ! -L "${release_root}" ]] || fail "extracted source release is not fresh"
[[ ! -e "${capacity_output}" && ! -L "${capacity_output}" ]] || fail "capacity output is not fresh"
[[ ! -e "${full644_output}" && ! -L "${full644_output}" ]] || fail "full644 output is not fresh"
/bin/mkdir -p "${release_parent}" "${output_parent}" "${experiment_root}/attempts"
/bin/mkdir "${attempt_root}"
/bin/chmod 0700 "${release_parent}" "${attempt_root}" "${output_parent}" "${experiment_root}/attempts"
"${python_bin}" -I -B "${helper}" audit-source-archive --archive "${source_archive}" >/dev/null
"${python_bin}" -I -B "${helper}" extract-source-archive \
  --archive "${source_archive}" --output "${release_root}" >/dev/null
require_runner
require_holder_idle pre-capacity

run_step PRELAUNCH_CAPACITY_ONLY "${capacity_log}" "${capacity_step_record}"
capacity_step="${launched_step_id}"
await_holder_idle post-capacity
require_runner
"${runner}" __validate_output__ PRELAUNCH_CAPACITY_ONLY >/dev/null
capacity_step_number="${capacity_step#${job_id}.}"
audit_runner_completion capacity-smoke "${capacity_output}" "${capacity_completion}" \
  "${capacity_cache_receipt}" "${capacity_step_number}"
capture_capacity_accounting "${capacity_step}"
[[ -f "${capacity_completion}" && ! -L "${capacity_completion}" ]] || fail "capacity runner completion is absent"
"${python_bin}" -I -B "${helper}" seal-capacity-gate \
  --training-output "${capacity_output}" --runner-completion "${capacity_completion}" \
  --sacct-row-file "${capacity_sacct_row}" --slurm-step-id "${capacity_step_number}" \
  --output "${capacity_gate}" >/dev/null
audit_capacity_gate "${capacity_step_number}"

require_holder_idle pre-full644
audit_capacity_gate "${capacity_step_number}"
[[ ! -e "${full644_output}" && ! -L "${full644_output}" ]] || fail "full644 output lost freshness before launch"
run_step FULL644_EXPLORATORY "${full644_log}" "${full644_step_record}"
full644_step="${launched_step_id}"
await_holder_idle post-full644
require_runner
"${runner}" __validate_output__ FULL644_EXPLORATORY >/dev/null
[[ -f "${full644_completion}" && ! -L "${full644_completion}" ]] || fail "full644 runner completion is absent"
full644_step_number="${full644_step#${job_id}.}"
audit_runner_completion full644 "${full644_output}" "${full644_completion}" \
  "${full644_cache_receipt}" "${full644_step_number}"
printf 'PASS capacity_step=%s full644_step=%s full644_output=%s\n' \
  "${capacity_step}" "${full644_step}" "${full644_output}"
