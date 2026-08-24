#!/bin/bash -p
#SBATCH --job-name=graft-pa-full81-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=72:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# Exact29 sealed runtime launcher.  The scheduler dependency is a queue gate;
# every parameter-changing Phase-A operation is replayed in this process.

case "$-" in *p*) ;; *) echo "[graft-pa-full81-v1] ERROR: privileged Bash required" >&2; exit 2;; esac
set -Eeuo pipefail
umask 077
for poison_name in ${!LD_*} ${!PYTHON*} ${!SBATCH_*}; do unset "${poison_name}"; done
unset BASH_ENV ENV CDPATH GLOBIGNORE HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PATH=/usr/bin:/bin LC_ALL=C LANG=C

fail() { echo "[graft-pa-full81-v1] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute and non-root"; }
sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

archive="${GRAFT_FULL81_SOURCE_ARCHIVE:?}"
archive_sha="${GRAFT_FULL81_SOURCE_ARCHIVE_SHA256:?}"
closure="${GRAFT_FULL81_RUNTIME_CLOSURE:?}"
closure_sha="${GRAFT_FULL81_RUNTIME_CLOSURE_SHA256:?}"
python_bin="${GRAFT_FULL81_PYTHON_BIN:?}"
python_sha="${GRAFT_FULL81_PYTHON_SHA256:?}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?}"
veomni_root="${BERNINI_VEOMNI_ROOT:?}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?}"
checkpoint_manifest_sha="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?}"
field14_plan="${GRAFT_FULL81_FIELD14_PLAN:?}"
field14_plan_sha="${GRAFT_FULL81_FIELD14_PLAN_SHA256:?}"
active14_plan="${GRAFT_FULL81_ACTIVE14_PLAN:?}"
active14_plan_sha="${GRAFT_FULL81_ACTIVE14_PLAN_SHA256:?}"
full81_plan="${GRAFT_FULL81_PLAN:?}"
full81_plan_sha="${GRAFT_FULL81_PLAN_SHA256:?}"
terminal="${GRAFT_FULL81_TERMINAL_ADMISSION:?}"
terminal_sha="${GRAFT_FULL81_TERMINAL_ADMISSION_SHA256:?}"
materializer_runtime_sha="${GRAFT_FULL81_TERMINAL_MATERIALIZER_RUNTIME_SHA256:?}"
field14_receipt="${GRAFT_FULL81_UPSTREAM_FIELD14_RECEIPT:?}"
active14_receipt="${GRAFT_FULL81_UPSTREAM_ACTIVE14_RECEIPT:?}"
field14_job_id="${GRAFT_FULL81_UPSTREAM_FIELD14_JOB_ID:?}"
active14_job_id="${GRAFT_FULL81_ACTIVE14_JOB_ID:?}"
output_root="${GRAFT_FULL81_OUTPUT_ROOT:?}"
log_root="${GRAFT_FULL81_LOG_ROOT:?}"
launcher_source="${GRAFT_FULL81_LAUNCHER_SOURCE:?}"
launcher_sha="${GRAFT_FULL81_LAUNCHER_SHA256:?}"

readonly required_full_source_commit=579e84cc72bd7f8915126e10b4d4a726e74dd163
readonly required_full_core_sha=f6d0952ce60351a9fc31a152e203c63f54dfbbabed2d101c70041b338cf5b7dd
readonly required_full_runner_sha=d6fb1e4c195c4e089518541fb3b228dae1912d2edac30282b749e217de95aab1
readonly required_active_core_sha=83290f94d92b66b8c5d15dc516dcec1ba3a492d03c7ed901d8d215cbcaec2244
readonly required_active_runner_sha=850d17a58096efad156990b9f6cbd478331766d95ad007dae726ceb46736d101
readonly required_field_core_sha=dbec11e9adc80171adf071bf2d79fc8ba7498b76e5740e19b077b90a0f0a5280
readonly required_field_runner_sha=41bffcf457d745b2744f321aa70f41b840a901b4e0302d722a75fc0bb08198ed
readonly required_field_source_commit=f9ef982e976ad19ed81ed075d33c9221952945e4
readonly required_short_runner_sha=4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4
readonly required_field14_plan_sha=ec07f89c49dc545397d436eecbac03bb4de9126f5bf97704d41cde6a666782a2
readonly required_active14_plan_sha=e24f9c5c62049da6400f40c71becac308434f9eb56a0ae22774e7b419d4e3541
readonly required_full81_plan_sha=05c4f53279b6e23eb06e8aac8abe3ed8bc39d322d1acb9ebfdd0c2d77a44915a
readonly required_field14_job_id=133530
readonly required_active14_job_id=133534
readonly required_active14_source_commit=037f7a8e296bf2d0bf7da63ccc9eb8b7ed6608fd
readonly required_active14_launcher_sha=d896b87dbc95dbcb65b80a0d635bc1dfd577f6a30a0dfc1d726ca23e1432efdb
readonly required_active14_receipt_path=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_active14_transaction_world8_v1/runs/source-037f7a8-launcher-8b6d08b-r2/receipt.json
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
readonly manifest_sha=0bfe9447fb3649625a4dface5c93ba0e55530285f9d63e7b15552b0bb4f9b1af
readonly producer_sha=01a87422753c5946c84df7c26198be992793de536159b1e427d4a00ab91ffea6
readonly execution_sha=101a9cb25a4c898be0ed1b0406a3aec866a153d12ff79c6b890fb78c1146efd6
readonly submission_sha=35ddc72443361ee588e5bd5c852df144293036a6128ae7fc9fab20e600e95b82

for name in archive closure python_bin bernini_root veomni_root checkpoint checkpoint_manifest field14_plan active14_plan full81_plan terminal field14_receipt active14_receipt output_root log_root launcher_source; do absolute "${!name}" "$name"; done
for name in archive_sha closure_sha python_sha checkpoint_manifest_sha field14_plan_sha active14_plan_sha full81_plan_sha terminal_sha materializer_runtime_sha launcher_sha; do sha256 "${!name}" "$name"; done
[[ "${field14_job_id}" =~ ^[1-9][0-9]*$ && "${active14_job_id}" =~ ^[1-9][0-9]*$ ]] || fail "dependency job IDs differ"
[[ "${field14_job_id}" != "${active14_job_id}" ]] || fail "dependency job IDs alias"
[[ "${field14_job_id}" == "${required_field14_job_id}" && "${active14_job_id}" == "${required_active14_job_id}" ]] || fail "dependency job ID pins differ"
[[ "${active14_receipt}" == "${required_active14_receipt_path}" ]] || fail "active14 receipt path pin differs"
[[ "${SLURM_JOB_ID:?}" =~ ^[1-9][0-9]*$ ]] || fail "Slurm job ID differs"
[[ -z "${SLURM_JOB_DEPENDENCY:-}" || "${SLURM_JOB_DEPENDENCY}" == *"afterok:${active14_job_id}"* ]] || fail "scheduler afterok dependency differs"
[[ "${checkpoint_manifest_sha}" == "${required_checkpoint_manifest_sha}" ]] || fail "checkpoint manifest pin differs"
[[ "${field14_plan_sha}" == "${required_field14_plan_sha}" && "${active14_plan_sha}" == "${required_active14_plan_sha}" && "${full81_plan_sha}" == "${required_full81_plan_sha}" ]] || fail "plan SHA pins differ"

for path in "${archive}" "${closure}" "${checkpoint_manifest}" "${field14_plan}" "${active14_plan}" "${full81_plan}" "${terminal}" "${field14_receipt}" "${active14_receipt}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(/usr/bin/stat -Lc '%a:%h' -- "${path}")" == 444:1 ]] || fail "sealed 0444 input required: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do [[ -d "${path}" && ! -L "${path}" ]] || fail "plain directory required: ${path}"; done
[[ -f "${python_bin}" && -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python must be a plain executable"
[[ -f "${launcher_source}" && -x "${launcher_source}" && ! -L "${launcher_source}" && "$(/usr/bin/stat -Lc '%a:%h' -- "${launcher_source}")" == 555:1 ]] || fail "launcher source must be sealed 0555/nlink1"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "final output must be fresh"
[[ ! -e "${log_root}" && ! -L "${log_root}" ]] || fail "log root must be fresh"
[[ -d "$(/usr/bin/dirname -- "${output_root}")" && -d "$(/usr/bin/dirname -- "${log_root}")" ]] || fail "output/log parent missing"

file_sha256() {
  "${python_bin}" -I -S -B - "$1" <<'PY'
import hashlib,os,stat,sys
p=sys.argv[1]; fd=os.open(p,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)); h=hashlib.sha256()
try:
    a=os.fstat(fd)
    if not stat.S_ISREG(a.st_mode): raise SystemExit("not regular")
    while True:
        b=os.read(fd,1024*1024)
        if not b: break
        h.update(b)
    z=os.fstat(fd)
    if (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_mode,a.st_nlink)!=(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_mode,z.st_nlink): raise SystemExit("changed")
finally: os.close(fd)
print(h.hexdigest())
PY
}
field14_receipt_sha="$(file_sha256 "${field14_receipt}")"
sha256 "${field14_receipt_sha}" field14_receipt_sha
active14_receipt_sha="$(file_sha256 "${active14_receipt}")"
sha256 "${active14_receipt_sha}" active14_receipt_sha
active14_receipt_digest="$("${python_bin}" -I -S -B - "${active14_receipt}" "${active14_receipt_sha}" "${active14_job_id}" "${field14_job_id}" "${required_active14_source_commit}" "${required_active_core_sha}" "${required_active_runner_sha}" "${active14_plan_sha}" "${required_active14_launcher_sha}" "${field14_receipt_sha}" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]); file_sha,job_id,field_job,source_commit,core_sha,runner_sha,plan_sha,launcher_sha,field_receipt_sha=sys.argv[2:]
def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def sha(raw): return hashlib.sha256(raw).hexdigest()
raw=path.read_bytes(); value=json.loads(raw.decode("ascii"))
if sha(raw)!=file_sha or raw!=canonical(value)+b"\n": raise SystemExit("active14 parent bytes differ")
receipt_digest=value.get("receipt_digest"); core={key:item for key,item in value.items() if key!="receipt_digest"}
if receipt_digest!=sha(canonical(core)): raise SystemExit("active14 parent receipt digest differs")
runtime=value.get("runtime",{}); result=value.get("runner_result",{}); binding=result.get("source_binding",{}); qualification=result.get("upstream_qualification",{})
if value.get("schema_version")!="bernini-graft-phase-a-active14-world8-parent-v1" or value.get("status")!="completed_operational_no_checkpoint" or value.get("complete") is not True or value.get("pass") is not True or value.get("job_id")!=job_id or value.get("field14_dependency_job_id")!=field_job or value.get("field14_receipt_file_sha256")!=field_receipt_sha or value.get("runner_result_digest")!=result.get("digest") or value.get("checkpoint_written") is not False or value.get("publication_performed") is not False or any(value.get("authority",{}).values()): raise SystemExit("active14 parent admission differs")
if runtime.get("source_git_commit")!=source_commit or runtime.get("active14_core_sha256")!=core_sha or runtime.get("active14_runner_sha256")!=runner_sha or runtime.get("active14_plan_sha256")!=plan_sha or runtime.get("launcher_sha256")!=launcher_sha: raise SystemExit("active14 parent runtime differs")
if result.get("status")!="completed_same_process_short_field14_active14_no_checkpoint" or result.get("dependency_afterok_is_queue_gate_only") is not True or result.get("weights_inherited_from_dependency_job") is not False or result.get("checkpoint_written") is not False or result.get("publication_performed") is not False or binding.get("active14_core_sha256")!=core_sha or binding.get("active14_runner_sha256")!=runner_sha or binding.get("plan_sha256")!=plan_sha or qualification.get("field14_job_id")!=field_job or qualification.get("weights_inherited_from_dependency_job") is not False: raise SystemExit("active14 parent runner evidence differs")
print(receipt_digest)
PY
)"
sha256 "${active14_receipt_digest}" active14_receipt_digest
for pair in "${archive}:${archive_sha}" "${closure}:${closure_sha}" "${python_bin}:${python_sha}" "${checkpoint_manifest}:${checkpoint_manifest_sha}" "${field14_plan}:${field14_plan_sha}" "${active14_plan}:${active14_plan_sha}" "${full81_plan}:${full81_plan_sha}" "${terminal}:${terminal_sha}" "${field14_receipt}:${field14_receipt_sha}" "${active14_receipt}:${active14_receipt_sha}" "${launcher_source}:${launcher_sha}"; do
  path=${pair%:*}; wanted=${pair##*:}; [[ "$(file_sha256 "${path}")" == "${wanted}" ]] || fail "input SHA differs: ${path}"
done
[[ "$(file_sha256 "$(/usr/bin/realpath -e -- "$0")")" == "${launcher_sha}" ]] || fail "executed launcher SHA differs"

audit_tree() {
  local root=$1 commit=$2 label=$3 status
  [[ "$(/usr/bin/git -C "${root}" rev-parse HEAD)" == "${commit}" ]] || fail "${label} commit differs"
  status="$(/usr/bin/git -C "${root}" status --porcelain=v1 --untracked-files=all)" || fail "${label} status failed"
  [[ -z "${status}" ]] || fail "${label} tree is dirty"
}
audit_tree "${bernini_root}" "${required_bernini_commit}" Bernini-pre
audit_tree "${veomni_root}" "${required_veomni_commit}" VeOmni-pre

scratch_parent=${SLURM_TMPDIR:-/tmp}
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
scratch="$(/usr/bin/mktemp -d "${scratch_parent%/}/graft-pa-full81-${SLURM_JOB_ID}.XXXXXX")"
torchrun_pid=""
quarantine_failed_output() {
  "${python_bin}" -I -S -B - "${output_root}" "${SLURM_JOB_ID}" <<'PY'
import os,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); job=sys.argv[2]
if not root.is_absolute() or root==Path("/") or not job.isdigit(): raise SystemExit("unsafe failed-output quarantine inputs")
stage=root.with_name(f".{root.name}.staging-job{job}")
for source,kind in ((root,"postcommit"),(stage,"staging")):
    try: info=source.lstat()
    except FileNotFoundError: continue
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode): raise SystemExit("failed output is not a real directory")
    target=root.with_name(f".{root.name}.failed-launcher-{kind}-job{job}")
    if target.exists() or target.is_symlink(): raise SystemExit("failed-output quarantine target exists")
    os.rename(source,target)
    fd=os.open(root.parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try: os.fsync(fd)
    finally: os.close(fd)
PY
}
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${torchrun_pid}" =~ ^[0-9]+$ ]]; then kill "${torchrun_pid}" 2>/dev/null || true; wait "${torchrun_pid}" 2>/dev/null || true; fi
  if (( status != 0 )); then
    quarantine_failed_output || echo "[graft-pa-full81-v1] WARNING: failed output could not be quarantined" >&2
    echo "[graft-pa-full81-v1] EXIT status=${status} logs=${log_root} scratch=${scratch} output_final=${output_root}" >&2 || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 KMP_USE_SHM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_HIP_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MIOPEN_USER_DB_PATH="${scratch}/cache/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/cache/torch-extensions" TRITON_CACHE_DIR="${scratch}/cache/triton" XDG_CACHE_HOME="${scratch}/cache/xdg" PYTHONPYCACHEPREFIX="${scratch}/cache/pycache" TMPDIR="${scratch}/tmp"
/usr/bin/mkdir -p -- "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TMPDIR}" "${scratch}/source-tree"
/usr/bin/cp -- "${archive}" "${scratch}/runtime-source.tar"
/usr/bin/cp -- "${closure}" "${scratch}/runtime-closure.json"

"${python_bin}" -I -S -B - "${scratch}/runtime-source.tar" "${scratch}/runtime-closure.json" "${scratch}/source-tree" "${archive_sha}" "${closure_sha}" "${required_full_source_commit}" <<'PY'
import hashlib,json,os,sys,tarfile
from pathlib import Path,PurePosixPath
archive,manifest_path,dest=map(Path,sys.argv[1:4]); archive_sha,manifest_sha,source_commit=sys.argv[4:]
def sha(raw): return hashlib.sha256(raw).hexdigest()
raw=manifest_path.read_bytes(); manifest=json.loads(raw.decode("ascii"))
if sha(archive.read_bytes())!=archive_sha or sha(raw)!=manifest_sha or raw!=json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")+b"\n": raise SystemExit("closure inputs differ")
if set(manifest)!={"files","root","schema_version","selection","source_git_commit"} or manifest.get("root")!="methods/bernini_action_editing" or manifest.get("schema_version")!="bernini-graft-phase-a-full-exact81-runtime-python-closure-v1" or manifest.get("selection")!="commit-579e84c-recursive-runtime-import-closure-v1" or manifest.get("source_git_commit")!=source_commit: raise SystemExit("closure contract differs")
expected={
"graft_a_lite_source_release_consumer_v1.py":"13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75",
"graft_phase_a_field14_exact40_v1.py":"dbec11e9adc80171adf071bf2d79fc8ba7498b76e5740e19b077b90a0f0a5280",
"graft_phase_a_native_training_closure_v1.py":"36861e8670fc77d65b469f86aa472a10d453f9ba5fc227a62303329c4c38409a",
"graft_phase_a_native_training_closure_v2.py":"bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f",
"identity_rebinder_v1.py":"1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0",
"infer_lora.py":"ce0bb91aa1850fa4568b6441cc4be4f41db8b8dcfc2afe2d9fcc76a6fca2ebe4",
"infer_native_i_axis_exact81_canary.py":"c3373d819534ac1e08270356b3311e9ef115390932bdbaa6dca64984ab618c58",
"infer_native_identity_generation_canary.py":"a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334",
"infer_orderless_source_frame_set_noise_canary.py":"f1e0182b6f9f6e2c4da4838b4f74f1ee629541887684a07424ec912e9050921c",
"infer_source_kv_carrier_oracle.py":"fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604",
"infer_source_value_residual_oracle.py":"40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8",
"inference_sigma_strata.py":"e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
"native_i_axis_guidance.py":"6a7cb379403aca0f6010f14477300ad6af6d3ba8e4581ba1c5890e6ff6fed931",
"orderless_source_frame_set_noise.py":"b5ad305798289fcf9112ce004f0305c5fea7f206c9cd9e6b331908c2f9af7bd0",
"run_graft_phase_a_a_lite_short_gpu_v1.py":"4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4",
"run_graft_phase_a_field14_exact40_gpu_v1.py":"41bffcf457d745b2744f321aa70f41b840a901b4e0302d722a75fc0bb08198ed",
"run_graft_phase_a_native_gpu_canary_v1.py":"e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a",
"source_kv_replay.py":"45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3",
"source_kv_route_batches.py":"7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e",
"source_value_residual.py":"420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124",
"tools/build_renderer_dataset.py":"afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
"tools/materialize_vae.py":"a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
"train_graft_phase_a_a_lite_short_v1.py":"73e39048bb8836fef33516eb1aae4cbc3f9fa4ecefcfb5d2695925bcb150f7bb",
"train_lora.py":"630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
"tri_branch_unipc.py":"58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2",
"train_graft_phase_a_active14_transaction_v1.py":"83290f94d92b66b8c5d15dc516dcec1ba3a492d03c7ed901d8d215cbcaec2244",
"run_graft_phase_a_active14_transaction_gpu_v1.py":"850d17a58096efad156990b9f6cbd478331766d95ad007dae726ceb46736d101",
"graft_phase_a_full_exact81_decoded_v1.py":"f6d0952ce60351a9fc31a152e203c63f54dfbbabed2d101c70041b338cf5b7dd",
"run_graft_phase_a_full_exact81_decoded_gpu_v1.py":"d6fb1e4c195c4e089518541fb3b228dae1912d2edac30282b749e217de95aab1"}
files=manifest.get("files")
if files!=expected or len(files)!=29: raise SystemExit("closure exact29 file map differs")
prefix=manifest["root"]+"/"; payloads={}
with tarfile.open(archive,"r:") as handle:
    members=handle.getmembers()
    if len(members)!=29: raise SystemExit("archive member count differs")
    for member in members:
        p=PurePosixPath(member.name)
        if p.is_absolute() or ".." in p.parts or member.name!=p.as_posix() or not member.isfile() or member.issym() or member.islnk() or not member.name.startswith(prefix): raise SystemExit("unsafe archive member")
        rel=member.name[len(prefix):]
        if rel in payloads or rel not in files: raise SystemExit("archive inventory differs")
        stream=handle.extractfile(member); data=stream.read() if stream else None
        if data is None or sha(data)!=files[rel]: raise SystemExit("archive member SHA differs")
        compile(data,member.name,"exec"); payloads[rel]=data
if set(payloads)!=expected: raise SystemExit("archive coverage differs")
for rel,data in sorted(payloads.items()):
    target=dest/manifest["root"]/rel; target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o444)
    try:
        offset=0
        while offset<len(data): offset+=os.write(fd,data[offset:])
        os.fchmod(fd,0o444)
    finally: os.close(fd)
root=dest/manifest["root"]
for directory in sorted((p for p in root.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True): os.chmod(directory,0o555)
os.chmod(root,0o555)
PY

method_root="${scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/run_graft_phase_a_full_exact81_decoded_gpu_v1.py"
[[ "$(file_sha256 "${runner}")" == "${required_full_runner_sha}" ]] || fail "full runner extraction differs"
if /usr/bin/find "${method_root}" \( -type l -o -name '__pycache__' -o -name '*.pyc' \) -print -quit | /usr/bin/grep -q .; then fail "runtime tree contains forbidden node"; fi

checkpoint_content_snapshot() {
  "${python_bin}" -I -B - "${method_root}" "${checkpoint}" "${checkpoint_manifest}" "${checkpoint_manifest_sha}" "${required_checkpoint_tree_sha}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,checkpoint,manifest=map(Path,sys.argv[1:4]); manifest_sha,tree_sha=sys.argv[4:]
sys.path.insert(0,str(root.resolve(strict=True)))
import infer_source_kv_carrier_oracle as audit
import infer_lora as legacy
validated,_=legacy.trainer.validate_checkpoint(checkpoint)
if validated!=checkpoint.resolve(strict=True) or legacy.trainer.CHECKPOINT_TREE_SHA256!=tree_sha: raise SystemExit("checkpoint tree differs")
receipt=audit.validate_checkpoint_content(validated,manifest.resolve(strict=True),expected_manifest_sha256=manifest_sha)
payload=json.dumps(receipt,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
print(hashlib.sha256(payload).hexdigest())
PY
}
checkpoint_content_pre_sha256="$(checkpoint_content_snapshot)"
sha256 "${checkpoint_content_pre_sha256}" checkpoint_content_pre_sha256

/usr/bin/mkdir -- "${log_root}"
/usr/bin/mkdir -- "${log_root}/rank-logs"
master_port=$((30000 + (SLURM_JOB_ID % 20000)))
echo "[graft-pa-full81-v1] WORLD8=DP2xSP4 dependency=${active14_job_id} same-process replay -> exact81; no checkpoint"
set +e
"${python_bin}" -I -B -m torch.distributed.run \
  --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="${master_port}" \
  --max-restarts=0 --monitor-interval=1 --log-dir="${log_root}/rank-logs" --redirects=3 \
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
  --plan-path "${field14_plan}" --expected-plan-sha256 "${field14_plan_sha}" \
  --expected-field14-core-sha256 "${required_field_core_sha}" --expected-field14-runner-sha256 "${required_field_runner_sha}" \
  --expected-field14-source-commit "${required_field_source_commit}" \
  --active14-plan-path "${active14_plan}" --expected-active14-plan-sha256 "${active14_plan_sha}" \
  --upstream-field14-receipt-path "${field14_receipt}" --expected-upstream-field14-receipt-sha256 "${field14_receipt_sha}" \
  --expected-upstream-field14-job-id "${field14_job_id}" --expected-active14-core-sha256 "${required_active_core_sha}" \
  --expected-active14-runner-sha256 "${required_active_runner_sha}" \
  --upstream-active14-receipt-path "${active14_receipt}" --expected-upstream-active14-receipt-sha256 "${active14_receipt_sha}" \
  --full-exact81-plan-path "${full81_plan}" --expected-full-exact81-plan-sha256 "${full81_plan_sha}" \
  --expected-full-exact81-core-sha256 "${required_full_core_sha}" --expected-full-exact81-runner-sha256 "${required_full_runner_sha}" \
  --full-expected-active14-core-sha256 "${required_active_core_sha}" --full-expected-active14-runner-sha256 "${required_active_runner_sha}" \
  --full-expected-active14-source-commit "${required_active14_source_commit}" --full-expected-active14-plan-sha256 "${active14_plan_sha}" \
  --full-expected-active14-launcher-sha256 "${required_active14_launcher_sha}" \
  --expected-active14-job-id "${active14_job_id}" --full-exact81-output-dir "${output_root}" \
  --ack-two-update-diagnostic-no-checkpoint-no-scientific-claim \
  --ack-exact40-no-grad-diagnostic-no-checkpoint-no-scientific-claim \
  --ack-active14-fresh-optimizer-14-update-transaction-no-checkpoint-no-scientific-claim \
  --ack-same-process-replay-no-checkpoint-operational-only \
  >"${log_root}/torchrun-supervisor.log" 2>&1 &
torchrun_pid=$!
wait "${torchrun_pid}"
status=$?
torchrun_pid=""
set -e
(( status == 0 )) || fail "WORLD8 full exact81 runner failed status=${status}; diagnostics retained"

audit_tree "${bernini_root}" "${required_bernini_commit}" Bernini-post
audit_tree "${veomni_root}" "${required_veomni_commit}" VeOmni-post
checkpoint_content_post_sha256="$(checkpoint_content_snapshot)"
[[ "${checkpoint_content_post_sha256}" == "${checkpoint_content_pre_sha256}" ]] || fail "checkpoint content identity changed across run"

"${python_bin}" -I -B - \
  "${method_root}" "${output_root}" "${full81_plan_sha}" \
  "${required_full_source_commit}" "${required_full_core_sha}" "${required_full_runner_sha}" \
  "${required_active_core_sha}" "${required_active_runner_sha}" \
  "${checkpoint_manifest_sha}" "${checkpoint_content_pre_sha256}" "${checkpoint_content_post_sha256}" \
  "${field14_job_id}" "${active14_job_id}" "${active14_receipt}" "${active14_receipt_sha}" "${active14_receipt_digest}" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path

(method_root_text,root_text,plan_sha,source_commit,full_core_sha,full_runner_sha,
 active_core_sha,active_runner_sha,checkpoint_manifest_sha,checkpoint_pre,
 checkpoint_post,field14_job_id,active14_job_id,active14_receipt_path,
 active14_receipt_sha,active14_receipt_digest)=sys.argv[1:]
method_root=Path(method_root_text).resolve(strict=True); root=Path(root_text)
sys.path.insert(0,str(method_root))
import graft_a_lite_source_release_consumer_v1 as consumer
import graft_phase_a_full_exact81_decoded_v1 as full81
from safetensors import safe_open
import torch

def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def sha(raw): return hashlib.sha256(raw).hexdigest()
def identity(value): return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode,value.st_nlink)
def read_plain(path,label):
    before=path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o444 or before.st_nlink!=1 or before.st_size<=0 or before.st_size>512*1024*1024: raise SystemExit(label+" is not bounded sealed plain file")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)); chunks=[]
    try:
        opened=os.fstat(fd)
        if identity(opened)!=identity(before): raise SystemExit(label+" changed while opening")
        while True:
            block=os.read(fd,1024*1024)
            if not block: break
            chunks.append(block)
        after=os.fstat(fd)
        if identity(after)!=identity(opened) or identity(path.lstat())!=identity(after): raise SystemExit(label+" changed while reading")
    finally: os.close(fd)
    raw=b"".join(chunks)
    return raw,sha(raw),identity(after)

if not root.is_absolute() or root==Path("/") or root.resolve(strict=True)!=root: raise SystemExit("output root is not exact absolute path")
root_before=root.lstat()
if not stat.S_ISDIR(root_before.st_mode) or root.is_symlink() or stat.S_IMODE(root_before.st_mode)!=0o555: raise SystemExit("output root is not sealed 0555 directory")
expected_nodes={
    "receipt.json","dog","human",
    "dog/endpoint.normalized-clean-latent.safetensors","dog/decoded-exact81.mp4",
    "human/endpoint.normalized-clean-latent.safetensors","human/decoded-exact81.mp4",
}
observed={}
for node in root.rglob("*"):
    relative=node.relative_to(root).as_posix(); info=node.lstat(); observed[relative]=info
    if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)): raise SystemExit("output tree contains unsafe node")
    if stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode)!=0o555: raise SystemExit("output directory mode differs")
    if stat.S_ISREG(info.st_mode) and (stat.S_IMODE(info.st_mode)!=0o444 or info.st_nlink!=1): raise SystemExit("output artifact sealing differs")
if set(observed)!=expected_nodes: raise SystemExit("output exact artifact inventory differs")

receipt_raw,_,_=read_plain(root/"receipt.json","parent receipt")
try: parent_value=json.loads(receipt_raw.decode("ascii"))
except (UnicodeDecodeError,json.JSONDecodeError) as error: raise SystemExit("parent receipt JSON differs") from error
if receipt_raw!=canonical(parent_value)+b"\n": raise SystemExit("parent receipt is not canonical newline JSON")
parent=full81.validate_sealed_mapping(parent_value,label="full exact81 parent receipt")
full81.assert_no_elevated_authority(parent)
parent_keys={
    "schema_version","status","plan","plan_sha256","world8",
    "active14_precommit_digest","active14_commit_receipt",
    "active14_commit_receipt_digest","preparation_receipt_digest",
    "source_binding","checkpoint_content_identity","checkpoint_manifest_sha256",
    "bernini_revision","output_root","atomic_publish_protocol",
    "publish_deferred_until_active14_outer_close",
    "checkpoint_written","publication_performed","visual_semantics_evaluated","digest",
}|set(full81.AUTHORITY_FIELDS)
if set(parent)!=parent_keys or parent.get("schema_version")!="bernini-graft-phase-a-full-exact81-decoded-parent-v1" or parent.get("status")!="same_process_phase_a_replay_then_exact40_exact81_decoded" or parent.get("output_root")!=str(root) or parent.get("atomic_publish_protocol")!="sealed-stage-single-rename-after-active14-outer-close" or parent.get("publish_deferred_until_active14_outer_close") is not True or parent.get("checkpoint_written") is not False or parent.get("publication_performed") is not False or parent.get("visual_semantics_evaluated") is not False: raise SystemExit("parent receipt contract differs")
if parent.get("plan_sha256")!=plan_sha or sha(canonical(parent.get("plan"))+b"\n")!=plan_sha: raise SystemExit("embedded plan binding differs")
plan=parent["plan"]
if plan.get("dependency")!={"job_id":active14_job_id,"kind":"afterok","purpose":"queue-gate-only-no-weight-lineage","receipt_path":active14_receipt_path,"receipt_sha256_policy":"derive-from-stable-sealed-file-after-afterok"} or any(plan.get("authority",{}).values()): raise SystemExit("plan dependency/authority differs")
runtime=plan.get("runtime",{})
if runtime.get("full_exact81_core_sha256")!=full_core_sha or runtime.get("full_exact81_runner_sha256")!=full_runner_sha or runtime.get("active14_core_sha256")!=active_core_sha or runtime.get("active14_runner_sha256")!=active_runner_sha: raise SystemExit("plan runtime pins differ")

world=full81.validate_sealed_mapping(parent.get("world8"),label="full exact81 WORLD8")
recomputed=full81.assemble_world8_result(world.get("rows",[]))
if full81.canonical_json_bytes(recomputed)!=full81.canonical_json_bytes(world): raise SystemExit("WORLD8 deep recomputation differs")
if world.get("both_exact40_completed") is not True or world.get("both_exact81_decoded") is not True or world.get("all_parameter_bytes_unchanged_during_decode") is not True or world.get("checkpoint_written") is not False or world.get("publication_performed") is not False or world.get("visual_semantics_evaluated") is not False: raise SystemExit("WORLD8 operational closure differs")
precommit_digest=full81.require_sha256(parent.get("active14_precommit_digest"),label="active14 precommit digest")
upstream_matrix=world.get("rank_local_upstream_receipt_bindings")
expected_matrix=[{"global_rank":rank,**world["rows"][rank]["upstream_in_memory_receipts"]} for rank in range(full81.WORLD_SIZE)]
if upstream_matrix!=expected_matrix or world.get("rank_local_upstream_receipt_bindings_digest")!=full81.object_sha256(expected_matrix) or precommit_digest!=expected_matrix[0]["active14_precommit"]: raise SystemExit("WORLD8 rank-local upstream receipt matrix differs")
commit=full81.validate_sealed_mapping(parent.get("active14_commit_receipt"),label="active14 commit receipt")
full81.assert_no_elevated_authority(commit)
if parent.get("active14_commit_receipt_digest")!=commit["digest"] or commit.get("topology",{}).get("rank")!=0 or commit.get("same_process_short_result_digest")!=expected_matrix[0]["short"] or commit.get("same_process_field14_result_digest")!=expected_matrix[0]["field14"] or commit.get("downstream_preparation_receipt_digest")!=parent.get("preparation_receipt_digest") or commit.get("all_fourteen_updates_completed") is not True or commit.get("transaction_committed_in_memory") is not True or commit.get("optimizer_contract",{}).get("schedule_indices")!=list(range(26,40)) or commit.get("weights_inherited_from_dependency_job") is not False or commit.get("checkpoint_written") is not False or commit.get("checkpoint_payload_returned") is not False or commit.get("publication_performed") is not False: raise SystemExit("active14 commit binding differs")
binding=parent.get("source_binding")
if binding!={
    "full_exact81_core_sha256":full_core_sha,
    "full_exact81_runner_sha256":full_runner_sha,
    "active14_core_sha256":active_core_sha,
    "active14_runner_sha256":active_runner_sha,
    "active14_dependency_job_id":active14_job_id,
    "upstream_active14_parent_receipt_path":active14_receipt_path,
    "upstream_active14_parent_receipt_sha256":active14_receipt_sha,
    "upstream_active14_parent_receipt_digest":active14_receipt_digest,
    "dependency_afterok_is_queue_gate_only":True,
    "weights_inherited_from_dependency":False,
    "checkpoint_loaded_from_dependency":False,
}: raise SystemExit("parent source binding differs")
checkpoint_identity=full81.validate_sealed_mapping(parent.get("checkpoint_content_identity"),label="checkpoint identity")
if checkpoint_pre!=checkpoint_post or full81.object_sha256(checkpoint_identity.get("identity"))!=checkpoint_pre or parent.get("checkpoint_manifest_sha256")!=checkpoint_manifest_sha: raise SystemExit("checkpoint pre/post identity differs")
if source_commit!= "579e84cc72bd7f8915126e10b4d4a726e74dd163": raise SystemExit("full source commit pin differs")
if not field14_job_id.isdigit() or not active14_job_id.isdigit() or field14_job_id==active14_job_id: raise SystemExit("dependency job identities differ")

opened_ffprobe=consumer._open_frozen_ffprobe()
try:
    for arm,family in enumerate(full81.FAMILY_ORDER):
        row=world["rows"][arm*full81.SP_SIZE]
        media=row["media"]
        family_plan=plan["families"][arm]
        expected_height,expected_width=family_plan["derived_bucket_hw"]
        expected_input_hw=family_plan["source_input_hw"]
        if (family,expected_input_hw,[expected_height,expected_width]) not in (("dog",[704,736],[480,496]),("human",[896,704],[544,432])) or (media["height"],media["width"])!=(expected_height,expected_width): raise SystemExit(family+" plan/media bucket geometry differs")
        latent_record=row["artifacts"]["latent"]; video_record=row["artifacts"]["video"]
        trace=full81.validate_exact40_trace(row["exact40_trace"])
        if trace["rows"][0]["state_before"]!=row["initial_gaussian"] or trace["rows"][-1]["state_after"]!=row["endpoint"]: raise SystemExit(family+" trace endpoint chain differs")
        full81.validate_media_record(media,expected_height=expected_height,expected_width=expected_width)
        full81.validate_artifact_record(latent_record,role="normalized-clean-latent",output_root=str(root),suffix=".safetensors",endpoint_identity=row["endpoint"])
        full81.validate_artifact_record(video_record,role="decoded-exact81-video",output_root=str(root),suffix=".mp4",endpoint_identity=row["endpoint"])
        expected_latent=f"{family}/endpoint.normalized-clean-latent.safetensors"
        expected_video=f"{family}/decoded-exact81.mp4"
        if latent_record.get("relative_path")!=expected_latent or video_record.get("relative_path")!=expected_video: raise SystemExit(family+" artifact path differs")
        for record,label in ((latent_record,family+" latent"),(video_record,family+" video")):
            raw,actual_sha,_=read_plain(Path(record["path"]),label)
            if Path(record["path"])!=root/record["relative_path"] or len(raw)!=record["size_bytes"] or actual_sha!=record["sha256"]: raise SystemExit(label+" actual bytes differ")
        latent_path=Path(latent_record["path"])
        with safe_open(str(latent_path),framework="pt",device="cpu") as opened:
            if list(opened.keys())!=["normalized_clean_latent"]: raise SystemExit(family+" latent tensor key differs")
            metadata=dict(opened.metadata() or {}); tensor=opened.get_tensor("normalized_clean_latent").contiguous()
        if metadata!={"coordinate":"bernini_normalized_clean_vae_latent","frame_contract":"exact81_latent21","artifact_role":"native_sampler_proposal","source":"native_sampler_before_vae_decode"}: raise SystemExit(family+" latent metadata differs")
        expected_shape=[1,16,21,expected_height//8,expected_width//8]
        owned=tensor.detach().cpu().contiguous().clone(); storage=owned.untyped_storage(); expected_bytes=owned.numel()*owned.element_size()
        raw_tensor=bytes(storage)
        tensor_content_sha=sha(canonical({"shape":expected_shape,"dtype":"torch.float32"})+b"\0"+raw_tensor)
        if type(owned) is not torch.Tensor or owned.dtype!=torch.float32 or list(owned.shape)!=expected_shape or not bool(torch.isfinite(owned).all().item()) or storage.nbytes()!=expected_bytes or sha(raw_tensor)!=row["endpoint"]["raw_sha256"] or tensor_content_sha!=row["endpoint"]["content_sha256"]: raise SystemExit(family+" latent endpoint identity differs")
        opened_video=consumer._open_source(str(Path(video_record["path"])),label=family+" decoded video")
        try:
            probe=consumer._probe_with_frozen_ffprobe(opened_video,opened_ffprobe)
            exact=consumer._normalize_probe(probe,label=family+" decoded video")
        finally:
            os.close(opened_video.fd); os.close(opened_video.parent_fd)
        if (exact.frame_count,exact.fps_numerator,exact.fps_denominator,probe.get("reported_fps_numerator"),probe.get("reported_fps_denominator"),exact.height,exact.width)!=(81,25,1,25,1,expected_height,expected_width) or media.get("decoded_tensor_shape")!=[81,expected_height,expected_width,3] or media.get("codec_content_interpreted_for_semantics") is not False: raise SystemExit(family+" exact81/25 media differs")
finally:
    os.close(opened_ffprobe.fd)

root_after=root.lstat()
if identity(root_after)!=identity(root_before): raise SystemExit("output root changed during postflight")
print("full exact81 postflight passed: hashes+latent-shape+exact81/25+authority=false")
PY

while IFS= read -r -d '' file; do /usr/bin/chmod 0444 -- "${file}"; done < <(/usr/bin/find "${log_root}" -type f -print0)
while IFS= read -r -d '' directory; do /usr/bin/chmod 0555 -- "${directory}"; done < <(/usr/bin/find "${log_root}" -depth -type d -print0)
echo "[graft-pa-full81-v1] COMPLETE output=${output_root} logs=${log_root} checkpoint_written=false authority=false"
