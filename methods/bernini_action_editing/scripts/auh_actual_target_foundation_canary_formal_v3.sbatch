#!/usr/bin/env bash
#SBATCH --job-name=actual_target_foundation_v3
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/actual_target_foundation_canary_v3_20260823/slurm_%x_%j.outer.log

set -euo pipefail
umask 077

# This entry point is deliberately non-authoritative.  V3 may run only from
# the reviewed login-side existing-allocation launcher, which binds an
# approved job id into both the compute and zero-GPU controller steps.
echo "V3 formal sbatch is not authorized; use the existing-allocation launcher" >&2
exit 64

snapshot_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/actual_target_foundation_canary_v3_20260823/snapshot_preflip_7f3c21a9_v3r4
run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/actual_target_foundation_canary_v3_20260823/run_7f3c21a9_v3r4
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

if [[ -z ${SLURM_JOB_ID:-} || -e "$run_root" || -L "$run_root" ]]; then
  echo "formal V3 requires a real Slurm job and an absent fixed run root" >&2
  exit 64
fi
if [[ ! -d "$snapshot_root" || -L "$snapshot_root" || ! -x "$wrapper" || -L "$wrapper" || ! -x "$controller_wrapper" || -L "$controller_wrapper" ]]; then
  echo "formal V3 immutable snapshot/wrapper binding differs" >&2
  exit 65
fi

export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
"$python_bin" -B "$snapshot_tool" --verify --snapshot-root "$snapshot_root" --skip-original-reverify >/dev/null
mkdir -m 700 "$run_root"
mkdir -m 700 "$cache_dir"
mkdir -m 700 "$miopen_user_dir"
mkdir -m 700 "$miopen_custom_cache_dir"

contract_digest=$("$python_bin" -B "$runtime" --print-contract | "$python_bin" -c 'import json,sys; print(json.load(sys.stdin)["digest"])')
srun_argv=(
  srun
  --nodes=1
  --ntasks=1
  --gpus-per-task=1
  --cpus-per-task=16
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
  echo "formal V3 did not capture exact two-element PIPESTATUS" >&2
  exit 66
fi
srun_exit_code=${pipe_status[0]}
tee_exit_code=${pipe_status[1]}

"$python_bin" -B "$controller" freeze-file "$formal_log" >/dev/null
"$python_bin" -B "$controller" freeze-scratch \
  --miopen-user-dir "$miopen_user_dir" \
  --miopen-custom-cache-dir "$miopen_custom_cache_dir" \
  --closure "$miopen_scratch_closure" >/dev/null
if [[ $srun_exit_code -eq 0 && $tee_exit_code -eq 0 && -f "$candidate" && -d "$cache_dir" ]]; then
  set +e
  "$python_bin" -B "$controller" freeze-run-artifacts --candidate "$candidate" --cache-dir "$cache_dir" >/dev/null
  freeze_exit_code=$?
  set -e
  if [[ $freeze_exit_code -ne 0 ]]; then
    echo "formal V3 candidate/cache freeze failed; controller will record engineering failure" >&2
  fi
fi

controller_argv=(
  srun
  --nodes=1
  --ntasks=1
  --gpus=0
  --cpus-per-task=1
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
  --expected-job-id "$SLURM_JOB_ID"
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
  echo "external V3 controller failed without an authoritative artifact" >&2
  exit "$controller_exit_code"
fi
if [[ -f "$seal" && ! -e "$attempt_ledger" && ! -L "$seal" ]]; then
  exit 0
fi
if [[ -f "$attempt_ledger" && ! -e "$seal" && ! -L "$attempt_ledger" ]]; then
  echo "V3 engineering attempt recorded; no valid PASS/REJECTED completion seal" >&2
  exit 70
fi
echo "formal V3 ended without exactly one controller artifact" >&2
exit 71
