#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 1 || ( $1 != 147871 && $1 != 147873 && $1 != 147881 ) ]]; then
  echo "usage: $0 APPROVED_RUNNING_JOB_ID  # exactly 147871, 147873, or 147881" >&2
  exit 64
fi
if [[ -n ${SLURM_JOB_ID:-} || -n ${SLURM_STEP_ID:-} || -n ${SLURM_STEPID:-} ]]; then
  echo "existing-allocation launcher must run from a login shell, not inside a Slurm job/step" >&2
  exit 64
fi
job_id=$1
snapshot_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/actual_target_foundation_canary_v3_20260823/snapshot_preflip_7f3c21a9_v3r4
run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/actual_target_foundation_canary_v3_20260823/run_7f3c21a9_v3r4
prior_failed_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/actual_target_foundation_canary_v3_20260823/run_7f3c21a9_v3r3
prior_formal_log="${prior_failed_run_root}/formal.log"
prior_attempt_ledger="${prior_failed_run_root}/attempt_ledger.json"
prior_failure_closure="${prior_failed_run_root}.failure_closure_v3.json"
prior_formal_log_sha256=c00334861df242bbae38640a4680c52aa371576676fbdc0d7e8fc5e71af5a46d
prior_attempt_ledger_sha256=5dbc4952a01f8fd00eec72c9b6072de763ea6d655ae05aca1f9309f1bfdbafd7
prior_failure_closure_sha256=57740bc57523c99be751e68c16f5d2527e7ed17565506b607043d9661c05bc0f
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
controller="${snapshot_root}/actual_target_foundation_controller_v3.py"
runtime="${snapshot_root}/actual_target_foundation_runtime_v3.py"
snapshot_tool="${snapshot_root}/actual_target_foundation_snapshot_v3.py"
wrapper="${snapshot_root}/scripts/auh_actual_target_foundation_canary_rank_wrapper_v3.sh"
controller_wrapper="${snapshot_root}/scripts/auh_actual_target_foundation_canary_controller_wrapper_v3.sh"
candidate="${run_root}/candidate.json"
cache_dir="${run_root}/cache"
miopen_user_dir="${run_root}/miopen-user"
miopen_custom_cache_dir="${run_root}/miopen-custom"
miopen_scratch_closure="${run_root}/miopen_scratch_closure.json"
seal="${run_root}/completion_seal.json"
attempt_ledger="${run_root}/attempt_ledger.json"
formal_log="${run_root}/formal.log"
srun_argv_file="${run_root}/srun_argv.nul"
rank_argv_file="${run_root}/rank_argv.nul"
step_meta="${run_root}/step_meta.json"
controller_argv_file="${run_root}/controller_argv.nul"
controller_step_meta="${run_root}/controller_step_meta.json"

job_line=$(squeue -h -j "$job_id" -o '%i|%T|%u|%D|%C')
if [[ -z "$job_line" || "$job_line" != "${job_id}|RUNNING|${USER}|"* ]]; then
  echo "approved allocation is not one RUNNING job owned by the caller: $job_line" >&2
  exit 65
fi
if [[ "$run_root" == "$prior_failed_run_root" || -e "$run_root" || -L "$run_root" || ! -d "$snapshot_root" || -L "$snapshot_root" || ! -x "$wrapper" || -L "$wrapper" || ! -x "$controller_wrapper" || -L "$controller_wrapper" ]]; then
  echo "fixed run root must be absent and immutable snapshot must be present" >&2
  exit 67
fi
for prior_path in "$prior_formal_log" "$prior_attempt_ledger" "$prior_failure_closure"; do
  if [[ ! -f "$prior_path" || -L "$prior_path" || $(stat -c '%a' "$prior_path") != 444 ]]; then
    echo "prior failed V3 attempt is not preserved as a plain file" >&2
    exit 67
  fi
done
prior_formal_observed=$(sha256sum "$prior_formal_log")
prior_formal_observed=${prior_formal_observed%% *}
prior_ledger_observed=$(sha256sum "$prior_attempt_ledger")
prior_ledger_observed=${prior_ledger_observed%% *}
prior_closure_observed=$(sha256sum "$prior_failure_closure")
prior_closure_observed=${prior_closure_observed%% *}
if [[ "$prior_formal_observed" != "$prior_formal_log_sha256" || "$prior_ledger_observed" != "$prior_attempt_ledger_sha256" || "$prior_closure_observed" != "$prior_failure_closure_sha256" ]]; then
  echo "prior failed V3R3 attempt bytes differ; refusing V3R4 launch" >&2
  exit 67
fi

export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
"$python_bin" -B "$snapshot_tool" --verify --snapshot-root "$snapshot_root" --skip-original-reverify >/dev/null
# Login-side admission independently double-scans the complete frozen legacy
# snapshot and failed-run trees.  The zero-GPU seal step repeats the same scan.
"$python_bin" -B "$controller" verify-prior-closures >/dev/null
mkdir -m 700 "$run_root"
mkdir -m 700 "$cache_dir"
mkdir -m 700 "$miopen_user_dir"
mkdir -m 700 "$miopen_custom_cache_dir"
contract_digest=$("$python_bin" -B "$runtime" --print-contract | "$python_bin" -c 'import json,sys; print(json.load(sys.stdin)["digest"])')
srun_argv=(
  srun
  --jobid "$job_id"
  --exclusive
  --exact
  --immediate=60
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --gres=gpu:mi210:1
  --mem=56G
  --export=ALL,LOCAL_RANK=0,WORLD_SIZE=1
  "$wrapper"
  "$candidate"
  "$cache_dir"
  "$step_meta"
  "$rank_argv_file"
  "$snapshot_root"
  "$miopen_user_dir"
  "$miopen_custom_cache_dir"
)
"$python_bin" -B "$controller" write-nul "$srun_argv_file" -- "${srun_argv[@]}" >/dev/null

set +e
"${srun_argv[@]}" 2>&1 | tee "$formal_log"
pipe_status=("${PIPESTATUS[@]}")
set -e
if [[ ${#pipe_status[@]} -ne 2 ]]; then
  echo "existing-allocation launcher did not capture exact two-element PIPESTATUS" >&2
  exit 68
fi
srun_exit_code=${pipe_status[0]}
tee_exit_code=${pipe_status[1]}
"$python_bin" -B "$controller" freeze-file "$formal_log" >/dev/null
set +e
"$python_bin" -B "$controller" freeze-scratch \
  --miopen-user-dir "$miopen_user_dir" \
  --miopen-custom-cache-dir "$miopen_custom_cache_dir" \
  --closure "$miopen_scratch_closure" >/dev/null
scratch_freeze_exit_code=$?
set -e
if [[ $scratch_freeze_exit_code -ne 0 ]]; then
  echo "MIOpen scratch freeze failed; controller will record engineering failure" >&2
fi
if [[ $srun_exit_code -eq 0 && $tee_exit_code -eq 0 && -f "$candidate" && -d "$cache_dir" ]]; then
  set +e
  "$python_bin" -B "$controller" freeze-run-artifacts --candidate "$candidate" --cache-dir "$cache_dir" >/dev/null
  freeze_exit_code=$?
  set -e
  if [[ $freeze_exit_code -ne 0 ]]; then
    echo "candidate/cache freeze failed; controller will record engineering failure" >&2
  fi
fi

controller_argv=(
  srun
  --jobid "$job_id"
  --exclusive
  --exact
  --immediate=60
  --nodes=1
  --ntasks=1
  --cpus-per-task=1
  --gpus=0
  --mem=1G
  --export=ALL
  "$controller_wrapper"
  "$snapshot_root"
  seal
  --candidate "$candidate"
  --cache-dir "$cache_dir"
  --seal "$seal"
  --attempt-ledger "$attempt_ledger"
  --expected-contract-digest "$contract_digest"
  --srun-exit-code "$srun_exit_code"
  --tee-exit-code "$tee_exit_code"
  --expected-job-id "$job_id"
  --step-meta "$step_meta"
  --formal-log "$formal_log"
  --srun-argv "$srun_argv_file"
  --rank-argv "$rank_argv_file"
  --controller-argv "$controller_argv_file"
  --controller-step-meta "$controller_step_meta"
  --miopen-scratch-closure "$miopen_scratch_closure"
  --snapshot-root "$snapshot_root"
)
"$python_bin" -B "$controller" write-nul "$controller_argv_file" -- "${controller_argv[@]}" >/dev/null
set +e
"${controller_argv[@]}"
controller_exit_code=$?
set -e
if [[ $controller_exit_code -ne 0 ]]; then
  exit "$controller_exit_code"
fi
if [[ -f "$seal" && ! -e "$attempt_ledger" && ! -L "$seal" ]]; then
  exit 0
fi
if [[ -f "$attempt_ledger" && ! -e "$seal" && ! -L "$attempt_ledger" ]]; then
  echo "engineering attempt recorded; no valid PASS/REJECTED completion seal" >&2
  exit 70
fi
exit 71
