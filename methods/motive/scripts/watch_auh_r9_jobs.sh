#!/usr/bin/env bash
set -Eeuo pipefail

ssh_host="${MOTIVE_R9_SSH_HOST:-auh}"
experiment_root="${MOTIVE_R9_EXPERIMENT_ROOT:?set MOTIVE_R9_EXPERIMENT_ROOT}"
job_ids_csv="${MOTIVE_R9_JOB_IDS:?set MOTIVE_R9_JOB_IDS}"
python_bin="${MOTIVE_R9_PYTHON_BIN:?set MOTIVE_R9_PYTHON_BIN}"
poll_seconds="${MOTIVE_R9_WATCH_POLL_SECONDS:-30}"
max_polls="${MOTIVE_R9_WATCH_MAX_POLLS:-960}"

if [[ ! "${experiment_root}" == \
  /vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/goku_repr_auto_r9_* ]]; then
  echo "[r9-watch] unsafe experiment root: ${experiment_root}" >&2
  exit 2
fi
if [[ ! "${job_ids_csv}" =~ ^[0-9]+,[0-9]+$ ]] \
  || [[ ! "${poll_seconds}" =~ ^[0-9]+$ ]] \
  || [[ ! "${max_polls}" =~ ^[0-9]+$ ]] \
  || ((poll_seconds < 5 || max_polls < 1)); then
  echo "[r9-watch] invalid jobs or polling policy" >&2
  exit 2
fi

IFS=, read -r -a job_ids <<< "${job_ids_csv}"
seeds=(260108835 260108836)
python_observed=0

probe_auh() {
  ssh -o ConnectTimeout=10 -o BatchMode=yes "${ssh_host}" true
}

validate_commits() {
  ssh "${ssh_host}" bash -s -- \
    "${experiment_root}" \
    "${python_bin}" \
    "${job_ids[0]}" \
    "${job_ids[1]}" <<'REMOTE'
set -Eeuo pipefail
root="$1"
python_bin="$2"
job_a="$3"
job_b="$4"
export PYTHONPATH="${root}/source_snapshot/methods/motive"
export PYTHONDONTWRITEBYTECODE=1
seeds=(260108835 260108836)
jobs=("${job_a}" "${job_b}")
for index in 0 1; do
  seed="${seeds[${index}]}"
  job="${jobs[${index}]}"
  search="${root}/seed_${seed}/representation/search_seed_${seed}"
  baseline="${root}/seed_${seed}/representation/baseline_screen"
  stdout="${root}/logs/r9_seed_${seed}_${job}.out"
  stderr="${root}/logs/r9_seed_${seed}_${job}.err"
  "${python_bin}" -c '
from pathlib import Path
from motive.r9_automated_representation_search import validate_published_search
import json
import sys
result = validate_published_search(Path(sys.argv[1]))
decision = result["summary"]["decision"]
print(json.dumps({
    "seed": int(sys.argv[2]),
    "status": decision["status"],
    "representation_gate_passed": decision["representation_gate_passed"],
    "renderer_probe_authorized": decision["renderer_probe_authorized"],
    "editor_training_authorized": decision["editor_training_authorized"],
}, sort_keys=True))
' "${search}" "${seed}"
  test -s "${baseline}/done.json"
  test -s "${stdout}"
  test -e "${stderr}"
  grep -F "[motive-r9-representation-search]" "${stdout}"
  grep -F "[motive-r7-candidate-temporal-screen]" "${stdout}"
  grep -F "[r9-controller] completed" "${stdout}"
done
REMOTE
}

for ((poll = 1; poll <= max_polls; poll++)); do
  echo \
    "[r9-watch] poll=${poll}/${max_polls} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! probe_auh; then
    echo "[r9-watch] AUH unavailable; retaining watcher state" >&2
    sleep "${poll_seconds}"
    continue
  fi

  accounting="$(
    ssh "${ssh_host}" \
      "sacct -j '${job_ids_csv}' -X -n -P -o JobIDRaw,State,ExitCode"
  )"
  printf '%s\n' "${accounting}"
  all_terminal=1
  any_failed=0
  observed=0
  running_job=
  while IFS='|' read -r job_id state exit_code; do
    [[ -n "${job_id}" ]] || continue
    state="${state%%+}"
    observed=$((observed + 1))
    case "${state}" in
      COMPLETED)
        ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|\
BOOT_FAIL|DEADLINE|REVOKED)
        any_failed=1
        ;;
      RUNNING|COMPLETING)
        all_terminal=0
        [[ -n "${running_job}" ]] || running_job="${job_id}"
        ;;
      *)
        all_terminal=0
        ;;
    esac
    if [[ "${state}" == "COMPLETED" && "${exit_code}" != "0:0" ]]; then
      any_failed=1
    fi
  done < <(printf '%s\n' "${accounting}")
  if ((observed != 2)); then
    all_terminal=0
  fi

  if ((python_observed == 0)) && [[ -n "${running_job}" ]]; then
    process_output="$(
      ssh "${ssh_host}" \
        "timeout 30s srun --overlap --jobid='${running_job}' -N1 -n1 --cpus-per-task=1 bash -lc 'ps -eo pid,ppid,etime,args | grep -E \"[p]ython.*(action_source_snapshot|motive\\.(r9_automated_representation_search|r7_candidate_temporal_screen|instruction_model_registry))\"'" \
        2>&1 || true
    )"
    printf '%s\n' "${process_output}"
    if [[ "${process_output}" == *"python"* ]]; then
      python_observed=1
      echo "[r9-watch] live Python process verified"
    fi
  fi

  if ((all_terminal == 1)); then
    if ((any_failed == 1)); then
      echo "[r9-watch] one or more jobs terminated unsuccessfully" >&2
      exit 3
    fi
    validate_commits
    echo "[r9-watch] both jobs and immutable commits verified"
    exit 0
  fi
  sleep "${poll_seconds}"
done

echo "[r9-watch] polling budget exhausted before terminal state" >&2
exit 75
