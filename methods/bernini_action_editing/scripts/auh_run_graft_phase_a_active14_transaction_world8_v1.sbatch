#!/bin/bash -p
#SBATCH --job-name=graft-pa-a14-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=72:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# Sealed WORLD8/DP2xSP4 launcher.  The scheduler dependency is a queue gate;
# the create-only Field14 receipt is independently consumed after afterok.

case "$-" in *p*) ;; *) echo "[graft-pa-a14-v1] ERROR: privileged Bash required" >&2; exit 2;; esac
set -Eeuo pipefail
umask 077
for poison_name in ${!LD_*} ${!PYTHON*} ${!SBATCH_*}; do unset "${poison_name}"; done
unset BASH_ENV ENV CDPATH GLOBIGNORE HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PATH=/usr/bin:/bin LC_ALL=C LANG=C

fail() { echo "[graft-pa-a14-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

archive="${GRAFT_ACTIVE14_SOURCE_ARCHIVE:?set GRAFT_ACTIVE14_SOURCE_ARCHIVE}"
archive_sha="${GRAFT_ACTIVE14_SOURCE_ARCHIVE_SHA256:?set GRAFT_ACTIVE14_SOURCE_ARCHIVE_SHA256}"
closure="${GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST:?set GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST}"
closure_sha="${GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST_SHA256:?set GRAFT_ACTIVE14_RUNTIME_CLOSURE_MANIFEST_SHA256}"
python_bin="${GRAFT_ACTIVE14_PYTHON_BIN:?set GRAFT_ACTIVE14_PYTHON_BIN}"
python_sha="${GRAFT_ACTIVE14_PYTHON_SHA256:?set GRAFT_ACTIVE14_PYTHON_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
field_plan="${GRAFT_ACTIVE14_FIELD14_PLAN:?set GRAFT_ACTIVE14_FIELD14_PLAN}"
field_plan_sha="${GRAFT_ACTIVE14_FIELD14_PLAN_SHA256:?set GRAFT_ACTIVE14_FIELD14_PLAN_SHA256}"
active_plan="${GRAFT_ACTIVE14_PLAN:?set GRAFT_ACTIVE14_PLAN}"
active_plan_sha="${GRAFT_ACTIVE14_PLAN_SHA256:?set GRAFT_ACTIVE14_PLAN_SHA256}"
terminal="${GRAFT_ACTIVE14_TERMINAL_ADMISSION:?set GRAFT_ACTIVE14_TERMINAL_ADMISSION}"
terminal_sha="${GRAFT_ACTIVE14_TERMINAL_ADMISSION_SHA256:?set GRAFT_ACTIVE14_TERMINAL_ADMISSION_SHA256}"
materializer_runtime_sha="${GRAFT_ACTIVE14_TERMINAL_MATERIALIZER_RUNTIME_SHA256:?set GRAFT_ACTIVE14_TERMINAL_MATERIALIZER_RUNTIME_SHA256}"
field_receipt="${GRAFT_ACTIVE14_UPSTREAM_FIELD14_RECEIPT:?set GRAFT_ACTIVE14_UPSTREAM_FIELD14_RECEIPT}"
field_job_id="${GRAFT_ACTIVE14_UPSTREAM_FIELD14_JOB_ID:?set GRAFT_ACTIVE14_UPSTREAM_FIELD14_JOB_ID}"
output_root="${GRAFT_ACTIVE14_OUTPUT_ROOT:?set GRAFT_ACTIVE14_OUTPUT_ROOT}"
launcher_source="${GRAFT_ACTIVE14_LAUNCHER_SOURCE:?set GRAFT_ACTIVE14_LAUNCHER_SOURCE}"
launcher_sha="${GRAFT_ACTIVE14_LAUNCHER_SHA256:?set GRAFT_ACTIVE14_LAUNCHER_SHA256}"

readonly required_source_commit=037f7a8e296bf2d0bf7da63ccc9eb8b7ed6608fd
readonly required_core_sha=83290f94d92b66b8c5d15dc516dcec1ba3a492d03c7ed901d8d215cbcaec2244
readonly required_runner_sha=850d17a58096efad156990b9f6cbd478331766d95ad007dae726ceb46736d101
readonly required_field_core_sha=dbec11e9adc80171adf071bf2d79fc8ba7498b76e5740e19b077b90a0f0a5280
readonly required_field_runner_sha=41bffcf457d745b2744f321aa70f41b840a901b4e0302d722a75fc0bb08198ed
readonly required_field_source_commit=f9ef982e976ad19ed81ed075d33c9221952945e4
readonly required_short_runner_sha=4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4
readonly required_rebinder_sha=1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0
readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_checkpoint_tree_sha=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly required_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_materializer_implementation_sha=4686463642a38e771c6858d1c10fc6aacb815a56e4f3eae951a336018d186cf4
readonly release_stem=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4
readonly manifest_path=${release_stem}.manifest.jsonl
readonly producer_path=${release_stem}.receipt.json
readonly execution_path=${release_stem}.execution.receipt.json
readonly submission_path=${release_stem}.submission.receipt.json
readonly required_terminal_path=${release_stem}.terminal.admission.receipt.json
readonly manifest_sha=0bfe9447fb3649625a4dface5c93ba0e55530285f9d63e7b15552b0bb4f9b1af
readonly producer_sha=01a87422753c5946c84df7c26198be992793de536159b1e427d4a00ab91ffea6
readonly execution_sha=101a9cb25a4c898be0ed1b0406a3aec866a153d12ff79c6b890fb78c1146efd6
readonly submission_sha=35ddc72443361ee588e5bd5c852df144293036a6128ae7fc9fab20e600e95b82

for name in archive closure python_bin bernini_root veomni_root checkpoint checkpoint_manifest field_plan active_plan terminal field_receipt output_root launcher_source; do require_absolute "${!name}" "$name"; done
for name in archive_sha closure_sha python_sha checkpoint_manifest_sha field_plan_sha active_plan_sha terminal_sha materializer_runtime_sha launcher_sha; do require_sha256 "${!name}" "$name"; done
[[ "${field_job_id}" =~ ^[1-9][0-9]*$ ]] || fail "numeric Field14 job ID required"
[[ "${checkpoint_manifest_sha}" == "${required_checkpoint_manifest_sha}" ]] || fail "checkpoint manifest SHA differs"

archive="$(/usr/bin/realpath -e -- "${archive}")"
closure="$(/usr/bin/realpath -e -- "${closure}")"
python_bin="$(/usr/bin/realpath -e -- "${python_bin}")"
bernini_root="$(/usr/bin/realpath -e -- "${bernini_root}")"
veomni_root="$(/usr/bin/realpath -e -- "${veomni_root}")"
checkpoint="$(/usr/bin/realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(/usr/bin/realpath -e -- "${checkpoint_manifest}")"
field_plan="$(/usr/bin/realpath -e -- "${field_plan}")"
active_plan="$(/usr/bin/realpath -e -- "${active_plan}")"
terminal="$(/usr/bin/realpath -e -- "${terminal}")"
field_receipt="$(/usr/bin/realpath -e -- "${field_receipt}")"
launcher_source="$(/usr/bin/realpath -e -- "${launcher_source}")"
output_root="$(/usr/bin/realpath -m -- "${output_root}")"
[[ "${terminal}" == "${required_terminal_path}" ]] || fail "terminal path differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python is not a plain executable"
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do [[ -d "${path}" && ! -L "${path}" ]] || fail "plain directory required: ${path}"; done
for path in "${archive}" "${closure}" "${checkpoint_manifest}" "${field_plan}" "${active_plan}" "${terminal}" "${field_receipt}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(/usr/bin/stat -Lc '%a:%h' -- "${path}")" == 444:1 ]] || fail "sealed 0444/nlink1 input required: ${path}"
done
[[ -f "${launcher_source}" && ! -L "${launcher_source}" && "$(/usr/bin/stat -Lc '%a:%h' -- "${launcher_source}")" == 555:1 ]] || fail "sealed launcher required"
[[ ! -e "${output_root}" && ! -L "${output_root}" && "${output_root}" != / ]] || fail "output root must be fresh"
output_parent="$(/usr/bin/dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
[[ "${SLURM_JOB_ID:?SLURM_JOB_ID required}" =~ ^[0-9]+$ ]] || fail "job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
scratch="$(/usr/bin/mktemp -d "${scratch_parent%/}/graft-pa-a14-v1-${SLURM_JOB_ID}.XXXXXX")"
torchrun_pid=""
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${torchrun_pid}" =~ ^[0-9]+$ ]]; then kill "${torchrun_pid}" 2>/dev/null || true; wait "${torchrun_pid}" 2>/dev/null || true; fi
  if (( status != 0 )); then echo "[graft-pa-a14-v1] EXIT status=${status} output_retained=true scratch=${scratch}" >&2 || true; fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

file_sha256() {
  "${python_bin}" -I -S -B - "$1" <<'PY'
import hashlib,os,stat,sys
p=sys.argv[1]; fd=os.open(p,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)); h=hashlib.sha256()
try:
    before=os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1: raise SystemExit("not single regular")
    while True:
        block=os.read(fd,1024*1024)
        if not block: break
        h.update(block)
    after=os.fstat(fd)
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_mode,before.st_nlink)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_mode,after.st_nlink): raise SystemExit("changed")
finally: os.close(fd)
print(h.hexdigest())
PY
}
[[ "$(file_sha256 "${archive}")" == "${archive_sha}" ]] || fail "archive bytes differ"
[[ "$(file_sha256 "${closure}")" == "${closure_sha}" ]] || fail "closure bytes differ"
[[ "$(file_sha256 "${python_bin}")" == "${python_sha}" ]] || fail "Python bytes differ"
[[ "$(file_sha256 "${field_plan}")" == "${field_plan_sha}" ]] || fail "Field14 plan bytes differ"
[[ "$(file_sha256 "${active_plan}")" == "${active_plan_sha}" ]] || fail "active14 plan bytes differ"
[[ "$(file_sha256 "${terminal}")" == "${terminal_sha}" ]] || fail "terminal bytes differ"
[[ "$(file_sha256 "${checkpoint_manifest}")" == "${checkpoint_manifest_sha}" ]] || fail "checkpoint manifest bytes differ"
[[ "$(file_sha256 "${launcher_source}")" == "${launcher_sha}" ]] || fail "launcher bytes differ"
executed_launcher="$(/usr/bin/realpath -e -- "$0")"
[[ "$(file_sha256 "${executed_launcher}")" == "${launcher_sha}" ]] || fail "executed launcher differs"
field_receipt_sha="$(file_sha256 "${field_receipt}")"
require_sha256 "${field_receipt_sha}" field_receipt_sha

audit_vendor() {
  local root="$1" commit="$2" label="$3" status
  [[ "$(/usr/bin/git -C "${root}" rev-parse HEAD)" == "${commit}" ]] || fail "${label} commit differs"
  status="$(/usr/bin/git -C "${root}" status --porcelain=v1 --untracked-files=all)" || fail "${label} status failed"
  [[ -z "${status}" ]] || fail "${label} tree is dirty"
}
audit_vendor "${bernini_root}" "${required_bernini_commit}" Bernini-pre
audit_vendor "${veomni_root}" "${required_veomni_commit}" VeOmni-pre

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 KMP_USE_SHM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MIOPEN_USER_DB_PATH="${scratch}/cache/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/cache/torch-extensions" TRITON_CACHE_DIR="${scratch}/cache/triton" XDG_CACHE_HOME="${scratch}/cache/xdg" PYTHONPYCACHEPREFIX="${scratch}/cache/pycache" TMPDIR="${scratch}/tmp"
/usr/bin/mkdir -p -- "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TMPDIR}" "${scratch}/source-tree"
/usr/bin/cp -- "${archive}" "${scratch}/runtime-source.tar"
/usr/bin/cp -- "${closure}" "${scratch}/runtime-closure.json"

"${python_bin}" -I -S -B - "${scratch}/runtime-source.tar" "${scratch}/runtime-closure.json" "${scratch}/source-tree" "${archive_sha}" "${closure_sha}" "${required_source_commit}" "${required_core_sha}" "${required_runner_sha}" "${required_field_core_sha}" "${required_field_runner_sha}" <<'PY'
import hashlib,json,os,sys,tarfile
from pathlib import Path,PurePosixPath
archive,manifest_path,destination=map(Path,sys.argv[1:4]); archive_sha,manifest_sha,source_commit,core_sha,runner_sha,field_core_sha,field_runner_sha=sys.argv[4:]
def digest(raw): return hashlib.sha256(raw).hexdigest()
raw=manifest_path.read_bytes(); manifest=json.loads(raw.decode("ascii")); canonical=json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")+b"\n"
if digest(archive.read_bytes())!=archive_sha or digest(raw)!=manifest_sha or raw!=canonical: raise SystemExit("closure inputs differ")
if set(manifest)!={"files","root","schema_version","selection","source_git_commit"} or manifest.get("root")!="methods/bernini_action_editing" or manifest.get("schema_version")!="bernini-graft-phase-a-active14-runtime-python-closure-v1" or manifest.get("selection")!="commit-037f7a8-recursive-runtime-import-closure-v1" or manifest.get("source_git_commit")!=source_commit: raise SystemExit("closure contract differs")
files=manifest.get("files"); expected={"graft_a_lite_source_release_consumer_v1.py","graft_phase_a_field14_exact40_v1.py","graft_phase_a_native_training_closure_v1.py","graft_phase_a_native_training_closure_v2.py","identity_rebinder_v1.py","infer_lora.py","infer_native_i_axis_exact81_canary.py","infer_native_identity_generation_canary.py","infer_orderless_source_frame_set_noise_canary.py","infer_source_kv_carrier_oracle.py","infer_source_value_residual_oracle.py","inference_sigma_strata.py","native_i_axis_guidance.py","orderless_source_frame_set_noise.py","run_graft_phase_a_a_lite_short_gpu_v1.py","run_graft_phase_a_field14_exact40_gpu_v1.py","run_graft_phase_a_active14_transaction_gpu_v1.py","run_graft_phase_a_native_gpu_canary_v1.py","source_kv_replay.py","source_kv_route_batches.py","source_value_residual.py","tools/build_renderer_dataset.py","tools/materialize_vae.py","train_graft_phase_a_a_lite_short_v1.py","train_graft_phase_a_active14_transaction_v1.py","train_lora.py","tri_branch_unipc.py"}
if not isinstance(files,dict) or set(files)!=expected or len(files)!=27 or files["train_graft_phase_a_active14_transaction_v1.py"]!=core_sha or files["run_graft_phase_a_active14_transaction_gpu_v1.py"]!=runner_sha or files["graft_phase_a_field14_exact40_v1.py"]!=field_core_sha or files["run_graft_phase_a_field14_exact40_gpu_v1.py"]!=field_runner_sha: raise SystemExit("closure exact27 differs")
prefix=manifest["root"]+"/"; seen={}
with tarfile.open(archive,"r:") as handle:
    members=handle.getmembers()
    if len(members)!=27: raise SystemExit("archive must contain exact27")
    for member in members:
        path=PurePosixPath(member.name); normalized=path.as_posix().lstrip("./")
        if path.is_absolute() or ".." in path.parts or member.name!=normalized or not member.isfile() or member.issym() or member.islnk() or not normalized.startswith(prefix): raise SystemExit("unsafe archive member")
        relative=normalized[len(prefix):]; stream=handle.extractfile(member)
        if relative in seen or relative not in files or stream is None: raise SystemExit("archive inventory differs")
        payload=stream.read()
        if digest(payload)!=files[relative]: raise SystemExit("archive hash differs: "+relative)
        seen[relative]=payload
if set(seen)!=expected: raise SystemExit("archive coverage differs")
for relative,payload in sorted(seen.items()):
    target=destination/manifest["root"]/relative; target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o444)
    try:
        offset=0
        while offset<len(payload):
            wrote=os.write(fd,payload[offset:])
            if wrote<=0: raise SystemExit("write stalled")
            offset+=wrote
        os.fchmod(fd,0o444)
    finally: os.close(fd)
    compile(payload,str(target),"exec")
root=destination/manifest["root"]
for directory in sorted((p for p in root.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True): os.chmod(directory,0o555)
os.chmod(root,0o555)
PY

method_root="${scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/run_graft_phase_a_active14_transaction_gpu_v1.py"
/usr/bin/chmod -R a-w -- "${method_root}"
[[ "$(file_sha256 "${runner}")" == "${required_runner_sha}" ]] || fail "runtime runner differs"
if /usr/bin/find "${method_root}" \( -type l -o -name '__pycache__' -o -name '*.pyc' \) -print -quit | /usr/bin/grep -q .; then fail "runtime tree contains forbidden node"; fi

/usr/bin/mkdir -- "${output_root}"
output_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${output_root}")"
/usr/bin/mkdir -- "${output_root}/rank-logs"
log_root="${output_root}/rank-logs"
master_port=$((30000 + (SLURM_JOB_ID % 20000)))
echo "[graft-pa-a14-v1] WORLD8=DP2xSP4 short+field14+active14 dependency=afterok:${field_job_id}"
set +e
"${python_bin}" -I -B -m torch.distributed.run \
  --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="${master_port}" \
  --max-restarts=0 --monitor-interval=1 --log-dir="${log_root}" --redirects=3 \
  "${runner}" \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
  --manifest-path "${manifest_path}" --producer-receipt-path "${producer_path}" \
  --execution-receipt-path "${execution_path}" --submission-receipt-path "${submission_path}" \
  --terminal-admission-path "${terminal}" --manifest-sha256 "${manifest_sha}" \
  --producer-receipt-sha256 "${producer_sha}" --execution-receipt-sha256 "${execution_sha}" \
  --submission-receipt-sha256 "${submission_sha}" --terminal-admission-sha256 "${terminal_sha}" \
  --terminal-materializer-implementation-sha256 "${required_materializer_implementation_sha}" \
  --terminal-materializer-runtime-sha256 "${materializer_runtime_sha}" \
  --expected-checkpoint-content-manifest-sha256 "${required_checkpoint_manifest_sha}" \
  --expected-runner-sha256 "${required_short_runner_sha}" \
  --expected-identity-rebinder-sha256 "${required_rebinder_sha}" \
  --expected-bernini-commit "${required_bernini_commit}" --expected-veomni-commit "${required_veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${required_checkpoint_tree_sha}" \
  --plan-path "${field_plan}" --expected-plan-sha256 "${field_plan_sha}" \
  --expected-field14-core-sha256 "${required_field_core_sha}" \
  --expected-field14-runner-sha256 "${required_field_runner_sha}" \
  --expected-field14-source-commit "${required_field_source_commit}" \
  --active14-plan-path "${active_plan}" --expected-active14-plan-sha256 "${active_plan_sha}" \
  --upstream-field14-receipt-path "${field_receipt}" \
  --expected-upstream-field14-receipt-sha256 "${field_receipt_sha}" \
  --expected-upstream-field14-job-id "${field_job_id}" \
  --expected-active14-core-sha256 "${required_core_sha}" \
  --expected-active14-runner-sha256 "${required_runner_sha}" \
  --ack-two-update-diagnostic-no-checkpoint-no-scientific-claim \
  --ack-exact40-no-grad-diagnostic-no-checkpoint-no-scientific-claim \
  --ack-active14-fresh-optimizer-14-update-transaction-no-checkpoint-no-scientific-claim \
  >"${output_root}/torchrun-supervisor.log" 2>&1 &
torchrun_pid=$!
wait "${torchrun_pid}"
status=$?
torchrun_pid=""
set -e
(( status == 0 )) || fail "WORLD8 runner failed status=${status}; logs retained"

audit_vendor "${bernini_root}" "${required_bernini_commit}" Bernini-post
audit_vendor "${veomni_root}" "${required_veomni_commit}" VeOmni-post

"${python_bin}" -I -S -B - "${output_root}" "${output_identity}" "${active_plan_sha}" "${archive_sha}" "${closure_sha}" "${launcher_sha}" "${required_source_commit}" "${required_core_sha}" "${required_runner_sha}" "${field_job_id}" "${field_receipt_sha}" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); identity=tuple(map(int,sys.argv[2].split(":"))); plan_sha,archive_sha,closure_sha,launcher_sha,source_commit,core_sha,runner_sha,field_job_id,field_receipt_sha=sys.argv[3:]
AUTH=("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","checkpoint_authority","publication_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed","semantic_action_editing_success_claimed")
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def sealed(v,label):
    if not isinstance(v,dict): raise SystemExit(label+" not mapping")
    u=dict(v); d=u.pop("digest",None)
    if d!=digest(u): raise SystemExit(label+" digest differs")
def deny(v):
    if isinstance(v,dict):
        for k,item in v.items():
            if k=="authority" and (not isinstance(item,dict) or any(x is not False for x in item.values())): raise SystemExit("authority mapping differs")
            if (k in AUTH or k.endswith("_authorized") or "authority" in k) and k!="authority" and item is not False: raise SystemExit("authority differs: "+k)
            if k in {"checkpoint_written","checkpoint_payload_returned","publication_performed"} and item is not False: raise SystemExit("publication differs")
            deny(item)
    elif isinstance(v,list):
        for item in v: deny(item)
meta=root.lstat()
if not stat.S_ISDIR(meta.st_mode) or (meta.st_dev,meta.st_ino)!=identity: raise SystemExit("output identity differs")
stdout=list((root/"rank-logs").rglob("stdout.log"))
if len(stdout)!=8: raise SystemExit("exact eight stdout logs required")
admitted=[]
for path in stdout:
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>768*1024*1024: raise SystemExit("unsafe stdout")
    for line in path.read_bytes().splitlines():
        try: value=json.loads(line.decode("ascii"))
        except (UnicodeDecodeError,json.JSONDecodeError): continue
        if isinstance(value,dict) and value.get("schema_version")=="bernini-graft-phase-a-active14-gpu-runner-v1":
            if line!=canonical(value): raise SystemExit("runner result not canonical")
            admitted.append(value)
if len(admitted)!=1: raise SystemExit("exact one rank-zero result required")
result=admitted[0]; sealed(result,"active14 runner"); deny(result)
if result.get("status")!="completed_same_process_short_field14_active14_no_checkpoint" or result.get("base_unchanged") is not True or result.get("base_sha256_before")!=result.get("base_sha256_after") or result.get("dependency_afterok_is_queue_gate_only") is not True or result.get("weights_inherited_from_dependency_job") is not False: raise SystemExit("runner closure differs")
world=result.get("world8"); sealed(world,"active14 WORLD8")
if world.get("schema_version")!="bernini-graft-phase-a-active14-world8-result-v1" or world.get("rank_order")!=list(range(8)) or len(world.get("rows",[]))!=8 or world.get("all_eight_active14_transactions_completed") is not True or world.get("both_sp4_arms_parameter_consensus") is not True: raise SystemExit("WORLD8 closure differs")
commit=result.get("local_active14_commit_receipt"); sealed(commit,"active14 commit")
if commit.get("all_fourteen_updates_completed") is not True or commit.get("optimizer_contract",{}).get("schedule_indices")!=list(range(26,40)) or commit.get("transaction_committed_in_memory") is not True: raise SystemExit("active14 commit differs")
binding=result.get("source_binding"); sealed(binding,"source binding")
if binding.get("active14_core_sha256")!=core_sha or binding.get("active14_runner_sha256")!=runner_sha or binding.get("plan_sha256")!=plan_sha: raise SystemExit("source binding differs")
qualification=result.get("upstream_qualification"); sealed(qualification,"qualification")
if qualification.get("field14_job_id")!=field_job_id or qualification.get("weights_inherited_from_dependency_job") is not False: raise SystemExit("qualification differs")
core={"schema_version":"bernini-graft-phase-a-active14-world8-parent-v1","status":"completed_operational_no_checkpoint","complete":True,"pass":True,"job_id":os.environ.get("SLURM_JOB_ID"),"field14_dependency_job_id":field_job_id,"field14_receipt_file_sha256":field_receipt_sha,"runner_result_digest":result["digest"],"runtime":{"source_git_commit":source_commit,"active14_core_sha256":core_sha,"active14_runner_sha256":runner_sha,"active14_plan_sha256":plan_sha,"source_archive_sha256":archive_sha,"runtime_closure_manifest_sha256":closure_sha,"launcher_sha256":launcher_sha},"validated":{"same_process_short_field14_active14":True,"exact14_official_order":True,"all_eight_results_assembled":True,"rollback_contract_present":True,"checkpoint_written":False,"scientific_authority":False},"runner_result":result,"checkpoint_written":False,"publication_performed":False,"authority":{name:False for name in AUTH}}
receipt={**core,"receipt_digest":digest(core)}; payload=canonical(receipt)+b"\n"
fd=os.open(root/"receipt.json",os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
try:
    offset=0
    while offset<len(payload):
        wrote=os.write(fd,payload[offset:])
        if wrote<=0: raise SystemExit("receipt write stalled")
        offset+=wrote
    os.fsync(fd); os.lseek(fd,0,os.SEEK_SET)
    if os.read(fd,len(payload)+1)!=payload: raise SystemExit("receipt reread differs")
    os.fchmod(fd,0o444); os.fsync(fd)
finally: os.close(fd)
PY

/usr/bin/find "${output_root}" -type f -exec /usr/bin/chmod 0444 -- {} +
/usr/bin/find "${output_root}" -type d -exec /usr/bin/chmod 0555 -- {} +
echo "[graft-pa-a14-v1] COMPLETE receipt=${output_root}/receipt.json"
