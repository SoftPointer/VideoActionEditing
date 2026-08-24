#!/usr/bin/env bash
# Resume the fail-closed V4 evaluator until its signed 68-video marker exists.

set -Eeuo pipefail

job_id=140846
eval_root="${ACTION_FULLFIELD_EVAL_ROOT:?set resumable V4 eval root}"
controller="${ACTION_FULLFIELD_EVAL_CONTROLLER:?set V4 eval controller}"
[[ -x "${controller}" ]]
mkdir -p "${eval_root}/logs"
supervisor_log="${eval_root}/logs/eval-supervisor.log"

count_outputs() {
  find "${eval_root}/media" -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' '
}

previous="$(count_outputs)"
stagnant=0
for pass in $(seq 1 24); do
  [[ ! -f "${eval_root}/EVALUATION_COMPLETE" ]] || break
  state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
  [[ "${state}" == RUNNING\|auh7-1b-gpu-\[246-248,279\]\|gres/gpu:mi210:8 ]]
  printf 'pass=%s before=%s\n' "${pass}" "${previous}" >>"${supervisor_log}"
  set +e
  "${controller}" >>"${supervisor_log}" 2>&1
  status=$?
  set -e
  current="$(count_outputs)"
  printf 'pass=%s controller_status=%s after=%s\n' \
    "${pass}" "${status}" "${current}" >>"${supervisor_log}"
  if (( current > previous )); then
    stagnant=0
  else
    stagnant=$((stagnant + 1))
  fi
  (( stagnant < 3 )) || {
    printf 'supervisor_stopped=three_consecutive_no_progress_passes\n' \
      >>"${supervisor_log}"
    exit 1
  }
  previous="${current}"
  sleep 5
done

[[ -f "${eval_root}/EVALUATION_COMPLETE" ]]
grep -Fxq 'evaluation_complete=true' "${eval_root}/EVALUATION_COMPLETE"
grep -Fxq 'video_count=68' "${eval_root}/EVALUATION_COMPLETE"
printf 'supervisor_complete=true\n' >>"${supervisor_log}"
