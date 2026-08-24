#!/bin/bash -p
#SBATCH --job-name=graft-short-traj-r2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# Fresh WORLD8 / DP2xSP4 single-arm trajectory diagnostic.  A scientific
# confirmation NO-GO is a completed diagnostic outcome, not a launcher error.
# Every receipt keeps all action/training/checkpoint/scientific authority false.

case "$-" in
  *p*) ;;
  *) echo "[graft-short-traj-r2] ERROR privileged bash required" >&2; exit 2 ;;
esac
set -Eeuo pipefail
umask 077

for poison_name in ${!LD_*} ${!PYTHON*} ${!SBATCH_*}; do
  unset "${poison_name}"
done
unset BASH_ENV ENV CDPATH GLOBIGNORE HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PATH=/usr/bin:/bin LC_ALL=C LANG=C

fail() { echo "[graft-short-traj-r2] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
file_sha256() { /usr/bin/sha256sum -- "$1" | /usr/bin/cut -d' ' -f1; }

source_archive="${GRAFT_TRAJ_SOURCE_ARCHIVE:?set GRAFT_TRAJ_SOURCE_ARCHIVE}"
source_archive_sha256="${GRAFT_TRAJ_SOURCE_ARCHIVE_SHA256:?set GRAFT_TRAJ_SOURCE_ARCHIVE_SHA256}"
closure_manifest="${GRAFT_TRAJ_RUNTIME_CLOSURE:?set GRAFT_TRAJ_RUNTIME_CLOSURE}"
closure_manifest_sha256="${GRAFT_TRAJ_RUNTIME_CLOSURE_SHA256:?set GRAFT_TRAJ_RUNTIME_CLOSURE_SHA256}"
python_bin="${GRAFT_TRAJ_PYTHON_BIN:?set GRAFT_TRAJ_PYTHON_BIN}"
python_sha256="${GRAFT_TRAJ_PYTHON_SHA256:?set GRAFT_TRAJ_PYTHON_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
plan_path="${GRAFT_TRAJ_PLAN:?set GRAFT_TRAJ_PLAN}"
plan_sha256="${GRAFT_TRAJ_PLAN_SHA256:?set GRAFT_TRAJ_PLAN_SHA256}"
terminal_admission="${GRAFT_TRAJ_TERMINAL_ADMISSION:?set GRAFT_TRAJ_TERMINAL_ADMISSION}"
terminal_admission_sha256="${GRAFT_TRAJ_TERMINAL_ADMISSION_SHA256:?set GRAFT_TRAJ_TERMINAL_ADMISSION_SHA256}"
terminal_materializer_runtime_sha256="${GRAFT_TRAJ_TERMINAL_MATERIALIZER_RUNTIME_SHA256:?set GRAFT_TRAJ_TERMINAL_MATERIALIZER_RUNTIME_SHA256}"
output_root="${GRAFT_TRAJ_OUTPUT_ROOT:?set GRAFT_TRAJ_OUTPUT_ROOT}"
launcher_source="${GRAFT_TRAJ_LAUNCHER_SOURCE:?set GRAFT_TRAJ_LAUNCHER_SOURCE}"
launcher_sha256="${GRAFT_TRAJ_LAUNCHER_SHA256:?set GRAFT_TRAJ_LAUNCHER_SHA256}"
runner_sha256="${GRAFT_TRAJ_RUNNER_SHA256:?set GRAFT_TRAJ_RUNNER_SHA256}"

readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_rebinder_sha256=1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0
readonly required_terminal_materializer_implementation_sha256=4686463642a38e771c6858d1c10fc6aacb815a56e4f3eae951a336018d186cf4
readonly source_stem=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4
readonly manifest_path=${source_stem}.manifest.jsonl
readonly producer_receipt_path=${source_stem}.receipt.json
readonly execution_receipt_path=${source_stem}.execution.receipt.json
readonly submission_receipt_path=${source_stem}.submission.receipt.json
readonly manifest_sha256=0bfe9447fb3649625a4dface5c93ba0e55530285f9d63e7b15552b0bb4f9b1af
readonly producer_receipt_sha256=01a87422753c5946c84df7c26198be992793de536159b1e427d4a00ab91ffea6
readonly execution_receipt_sha256=101a9cb25a4c898be0ed1b0406a3aec866a153d12ff79c6b890fb78c1146efd6
readonly submission_receipt_sha256=35ddc72443361ee588e5bd5c852df144293036a6128ae7fc9fab20e600e95b82

for pair in \
  "$source_archive:source archive" "$closure_manifest:closure manifest" \
  "$python_bin:Python" "$bernini_root:Bernini root" "$veomni_root:VeOmni root" \
  "$checkpoint:checkpoint" "$checkpoint_manifest:checkpoint manifest" \
  "$plan_path:plan" "$terminal_admission:terminal admission" \
  "$output_root:output root" "$launcher_source:launcher"; do
  require_absolute "${pair%%:*}" "${pair#*:}"
done
for pair in \
  "$source_archive_sha256:archive" "$closure_manifest_sha256:closure" \
  "$python_sha256:Python" "$checkpoint_manifest_sha256:checkpoint manifest" \
  "$plan_sha256:plan" "$terminal_admission_sha256:terminal admission" \
  "$terminal_materializer_runtime_sha256:terminal materializer runtime" \
  "$launcher_sha256:launcher" "$runner_sha256:runner"; do
  require_sha256 "${pair%%:*}" "${pair#*:}"
done
[[ "${checkpoint_manifest_sha256}" == "${required_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest pin differs"

output_parent="$(/usr/bin/dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -e "${output_root}" ]] || fail "fresh create-only output root unavailable"
/usr/bin/mkdir -- "${output_root}"
/usr/bin/chmod 700 -- "${output_root}"

task_scratch="$(/usr/bin/mktemp -d "/tmp/graft-short-traj-r2-${SLURM_JOB_ID}.XXXXXX")"
torchrun_pid=""
publish_failure() {
  local status="$1"
  "${python_bin}" -I -S -B - "${output_root}" "${status}" "${SLURM_JOB_ID}" "${runner_sha256}" "${plan_sha256}" <<'PY' || true
import hashlib,json,os,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); status=int(sys.argv[2]); job=sys.argv[3]; runner_sha=sys.argv[4]; plan_sha=sys.argv[5]
authority={name:False for name in ("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","checkpoint_authority","publication_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed","semantic_action_editing_success_claimed")}
core={"schema_version":"bernini-graft-phase-a-short-trajectory-parent-failure-v1","status":"operational_failure","complete":False,"slurm_job_id":job,"launcher_exit_status":status,"runner_sha256":runner_sha,"plan_sha256":plan_sha,"checkpoint_written":False,"publication_performed":False,"authority":authority}
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
value={**core,"digest":hashlib.sha256(canonical(core)).hexdigest()}; payload=canonical(value)+b"\n"
fd=os.open(root/"failure.receipt.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
try:
 os.write(fd,payload); os.fsync(fd); os.fchmod(fd,0o444)
finally: os.close(fd)
PY
}
cleanup() {
  status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${torchrun_pid}" ]]; then
    /usr/bin/kill -TERM -- "-${torchrun_pid}" 2>/dev/null || true
  fi
  if (( status != 0 )); then
    publish_failure "${status}"
    echo "[graft-short-traj-r2] EXIT status=${status} scratch=${task_scratch}" >&2
  else
    /usr/bin/rm -rf -- "${task_scratch}"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM HUP

for path in "${source_archive}" "${closure_manifest}" "${checkpoint_manifest}" "${plan_path}" "${terminal_admission}" "${launcher_source}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "sealed input is not a plain file: ${path}"
done
[[ -x "${python_bin}" ]] || fail "configured Python is not executable"
[[ "$(file_sha256 "${source_archive}")" == "${source_archive_sha256}" ]] || fail "archive SHA differs"
[[ "$(file_sha256 "${closure_manifest}")" == "${closure_manifest_sha256}" ]] || fail "closure SHA differs"
[[ "$(file_sha256 "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest SHA differs"
[[ "$(file_sha256 "${plan_path}")" == "${plan_sha256}" ]] || fail "plan SHA differs"
[[ "$(file_sha256 "${terminal_admission}")" == "${terminal_admission_sha256}" ]] || fail "terminal admission SHA differs"
[[ "$(file_sha256 "${launcher_source}")" == "${launcher_sha256}" ]] || fail "launcher SHA differs"
[[ "$(file_sha256 "${python_bin}")" == "${python_sha256}" ]] || fail "Python SHA differs"

source_tree="${task_scratch}/source-tree"
/usr/bin/mkdir -- "${source_tree}"
"${python_bin}" -I -S -B - "${source_archive}" "${closure_manifest}" "${source_tree}" "${runner_sha256}" <<'PY'
import hashlib,json,os,stat,sys,tarfile
from pathlib import Path
archive,closure,out=map(Path,sys.argv[1:4]); runner_sha=sys.argv[4]
manifest=json.loads(closure.read_text(encoding="ascii")); files=manifest.get("files")
if manifest.get("schema_version")!="bernini-graft-phase-a-short-trajectory-runtime-closure-v1" or not isinstance(files,dict): raise SystemExit("closure schema differs")
expected={"methods/bernini_action_editing/"+name for name in files}
with tarfile.open(archive,"r:") as handle:
 members=handle.getmembers(); names={m.name for m in members}
 if names!=expected or any(not m.isfile() or m.issym() or m.islnk() for m in members): raise SystemExit("archive member inventory differs")
 handle.extractall(out,filter="data")
for name,wanted in files.items():
 path=out/"methods/bernini_action_editing"/name
 observed=hashlib.sha256(path.read_bytes()).hexdigest()
 if observed!=wanted: raise SystemExit("closure member SHA differs: "+name)
if files.get("run_graft_phase_a_short_trajectory_diagnostic_gpu_v1.py")!=runner_sha: raise SystemExit("runner closure pin differs")
PY

method_root="${source_tree}/methods/bernini_action_editing"
runner="${method_root}/run_graft_phase_a_short_trajectory_diagnostic_gpu_v1.py"
rebinder="${method_root}/identity_rebinder_v1.py"
[[ "$(file_sha256 "${runner}")" == "${runner_sha256}" ]] || fail "runner SHA differs"
[[ "$(file_sha256 "${rebinder}")" == "${required_rebinder_sha256}" ]] || fail "rebinder SHA differs"

[[ "$(/usr/bin/git -C "${bernini_root}" rev-parse HEAD)" == "${required_bernini_commit}" ]] || fail "Bernini commit differs"
[[ "$(/usr/bin/git -C "${veomni_root}" rev-parse HEAD)" == "${required_veomni_commit}" ]] || fail "VeOmni commit differs"

"${python_bin}" -I -B - "${method_root}" "${plan_path}" "${runner_sha256}" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import run_graft_phase_a_short_trajectory_diagnostic_gpu_v1 as runner
plan=json.loads(Path(sys.argv[2]).read_text(encoding="ascii")); runner_sha=sys.argv[3]
if plan.get("schema_version")!="bernini-graft-phase-a-short-trajectory-world8-plan-v1": raise SystemExit("plan schema differs")
if plan.get("topology")!={"allocation":"single-node-8xMI210","world_size":8,"dp_size":2,"sp_size":4}: raise SystemExit("plan topology differs")
if plan.get("trajectory_stages")!=list(runner.TRAJECTORY_STAGES) or plan.get("measurement_indices")!=list(runner.CONFIRMATION_INDICES) or plan.get("update_indices")!=list(runner.UPDATE_INDICES): raise SystemExit("plan trajectory differs")
arm=plan.get("optimizer_arm",{})
if arm!={"kind":"AdamW","learning_rate_float64_hex":runner.short_trainer.OPTIMIZER_LEARNING_RATE.hex(),"update_count":2,"single_preregistered_arm":True,"lr_or_step_sweep":False,"post_hoc_selection":False}: raise SystemExit("plan optimizer arm differs")
terminal=plan.get("terminal_coordination",{})
if terminal!={"all_pass_protocol":"all_pass_original_finish_all_ranks","any_no_go_protocol":"global_no_go_all_rank_rollback_no_finish","gate_vectors_gathered_before_original_admissions":True,"local_gate_pass_global_no_go_rolls_back":True,"mixed_family_outcomes_allowed_without_collective_divergence":True,"rollback_terminal_equals_initial_all_eight_ranks":True,"world_size":8}: raise SystemExit("plan terminal coordination differs")
if plan.get("runner_sha256")!=runner_sha or plan.get("no_checkpoint") is not True or any(plan.get("authority",{}).values()): raise SystemExit("plan authority differs")
PY

/usr/bin/mkdir -- "${output_root}/rank-logs"
rank_log_root="${output_root}/rank-logs"
master_port=$((32000 + (SLURM_JOB_ID % 20000)))
echo "[graft-short-traj-r2] WORLD8 DP2xSP4 stages=initial,after29,after38 cells=3x2 no-checkpoint"
set +e
"${python_bin}" -I -B -m torch.distributed.run \
  --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="${master_port}" \
  --max-restarts=0 --monitor-interval=1 --log-dir="${rank_log_root}" --redirects=3 \
  "${runner}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --manifest-path "${manifest_path}" \
  --producer-receipt-path "${producer_receipt_path}" \
  --execution-receipt-path "${execution_receipt_path}" \
  --submission-receipt-path "${submission_receipt_path}" \
  --terminal-admission-path "${terminal_admission}" \
  --manifest-sha256 "${manifest_sha256}" \
  --producer-receipt-sha256 "${producer_receipt_sha256}" \
  --execution-receipt-sha256 "${execution_receipt_sha256}" \
  --submission-receipt-sha256 "${submission_receipt_sha256}" \
  --terminal-admission-sha256 "${terminal_admission_sha256}" \
  --terminal-materializer-implementation-sha256 "${required_terminal_materializer_implementation_sha256}" \
  --terminal-materializer-runtime-sha256 "${terminal_materializer_runtime_sha256}" \
  --expected-checkpoint-content-manifest-sha256 "${required_checkpoint_manifest_sha256}" \
  --expected-runner-sha256 "${runner_sha256}" \
  --expected-identity-rebinder-sha256 "${required_rebinder_sha256}" \
  --expected-bernini-commit "${required_bernini_commit}" \
  --expected-veomni-commit "${required_veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${required_checkpoint_tree_sha256}" \
  --ack-two-update-diagnostic-no-checkpoint-no-scientific-claim \
  >"${output_root}/torchrun-supervisor.log" 2>&1 &
torchrun_pid=$!
wait "${torchrun_pid}"
torchrun_status=$?
torchrun_pid=""
set -e
(( torchrun_status == 0 )) || fail "WORLD8 operational failure status=${torchrun_status}"

[[ "$(/usr/bin/git -C "${bernini_root}" rev-parse HEAD)" == "${required_bernini_commit}" ]] || fail "Bernini changed across run"
[[ "$(/usr/bin/git -C "${veomni_root}" rev-parse HEAD)" == "${required_veomni_commit}" ]] || fail "VeOmni changed across run"
[[ "$(file_sha256 "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest changed across run"

"${python_bin}" -I -S -B - "${output_root}" "${rank_log_root}" "${SLURM_JOB_ID}" "${runner_sha256}" "${plan_sha256}" "${source_archive_sha256}" "${closure_manifest_sha256}" "${launcher_sha256}" <<'PY'
import hashlib,json,os,re,stat,sys
from pathlib import Path
output,logs=map(Path,sys.argv[1:3]); job,runner_sha,plan_sha,archive_sha,closure_sha,launcher_sha=sys.argv[3:]
SHA=re.compile(r"[0-9a-f]{64}\Z")
AUTHORITY=("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","checkpoint_authority","publication_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed","semantic_action_editing_success_claimed")
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def sealed(value,label):
 if not isinstance(value,dict): raise SystemExit(label+" is not mapping")
 unsigned=dict(value); claimed=unsigned.pop("digest",None)
 if not isinstance(claimed,str) or SHA.fullmatch(claimed) is None or hashlib.sha256(canonical(unsigned)).hexdigest()!=claimed: raise SystemExit(label+" digest differs")
 return claimed
stdout=list(logs.rglob("stdout.log"))
if len(stdout)!=8: raise SystemExit("exact eight rank stdout logs required")
results=[]
for path in stdout:
 if path.is_symlink() or not path.is_file() or path.stat().st_size>256*1024*1024: raise SystemExit("rank stdout differs")
 for line in path.read_bytes().splitlines():
  if not line.startswith(b"{"): continue
  try: value=json.loads(line.decode("ascii"))
  except Exception: continue
  if isinstance(value,dict) and value.get("schema_version")=="bernini-graft-phase-a-short-trajectory-world8-result-set-v1":
   if line!=canonical(value): raise SystemExit("result is not canonical ASCII JSON")
   results.append(value)
if len(results)!=1: raise SystemExit("exactly one WORLD8 trajectory result required")
result=results[0]; sealed(result,"WORLD8 result")
if result.get("status") not in {"completed_observed_scientific_no_go","completed_original_confirmation_pass"}: raise SystemExit("diagnostic status differs")
if result.get("scientific_outcome") not in {"NO_GO","ORIGINAL_GATE_PASS"}: raise SystemExit("scientific outcome differs")
if result.get("checkpoint_written") is not False or result.get("publication_performed") is not False or any(result.get(name) is not False for name in AUTHORITY): raise SystemExit("result elevated authority")
full=result.get("world8_full_results"); sealed(full,"full WORLD8")
receipts=full.get("all_eight_local_receipts"); matrices=full.get("dog_human_trajectory_matrices")
if not isinstance(receipts,list) or len(receipts)!=8 or not isinstance(matrices,list) or len(matrices)!=2: raise SystemExit("WORLD8 matrix coverage differs")
for rank,local in enumerate(receipts):
 sealed(local,f"rank{rank}")
 topology=local.get("topology",{}); rows=local.get("trajectory_matrices")
 if topology.get("rank")!=rank or topology.get("dp_arm")!=rank//4 or topology.get("sp_rank")!=rank%4 or not isinstance(rows,list) or len(rows)!=3 or local.get("trajectory_cell_count")!=6: raise SystemExit("local matrix differs")
 if local.get("checkpoint_written") is not False or any(local.get(name) is not False for name in AUTHORITY): raise SystemExit("local authority differs")
 for stage,row in zip(("initial","after_update_29","after_update_38"),rows):
  sealed(row,"matrix"); cells=row.get("cells")
  if row.get("stage")!=stage or not isinstance(cells,list) or len(cells)!=2: raise SystemExit("stage matrix differs")
  for index,cell in zip((29,38),cells):
   sealed(cell,"cell")
   if cell.get("schedule_index")!=index or cell.get("sp4_exact_consensus") is not True or cell.get("same_noise_within_correct_wrong_drop") is not True or cell.get("original_thresholds_used_without_change") is not True: raise SystemExit("cell contract differs")
authority={name:False for name in AUTHORITY}
core={"schema_version":"bernini-graft-phase-a-short-trajectory-parent-v1","status":"completed_zero_authority_diagnostic","complete":True,"slurm_job_id":job,"scientific_outcome":result["scientific_outcome"],"scientific_no_go_is_not_operational_failure":result["scientific_outcome"]=="NO_GO","topology":{"allocation":"single-node-8xMI210","world_size":8,"dp_size":2,"sp_size":4},"trajectory":{"stages":["initial","after_update_29","after_update_38"],"measurement_indices":[29,38],"cell_count_per_family":6,"dog_human_matrix_count":2,"single_preregistered_optimizer_arm":True},"provenance":{"runner_sha256":runner_sha,"plan_sha256":plan_sha,"source_archive_sha256":archive_sha,"runtime_closure_sha256":closure_sha,"launcher_sha256":launcher_sha},"runner_result_digest":result["digest"],"runner_result":result,"checkpoint_written":False,"publication_performed":False,"authority":authority}
parent={**core,"digest":hashlib.sha256(canonical(core)).hexdigest()}; payload=canonical(parent)+b"\n"
fd=os.open(output/"receipt.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
try:
 os.write(fd,payload); os.fsync(fd); os.fchmod(fd,0o444)
finally: os.close(fd)
PY

[[ -f "${output_root}/receipt.json" ]] || fail "parent diagnostic receipt absent"
echo "[graft-short-traj-r2] completed zero-authority trajectory diagnostic"
