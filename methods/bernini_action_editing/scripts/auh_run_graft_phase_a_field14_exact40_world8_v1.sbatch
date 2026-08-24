#!/bin/bash -p
#SBATCH --job-name=graft-pa-f14-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=48:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# Sealed WORLD8/DP2xSP4 launcher for same-process short bootstrap followed by
# the no-grad exact40 field14 sweep.  It writes rank logs and one create-only
# diagnostic receipt; the model runner has no checkpoint/output CLI.

case "$-" in *p*) ;; *) echo "[graft-pa-f14-v1] ERROR: privileged Bash required" >&2; exit 2;; esac
set -Eeuo pipefail
umask 077
for poison_name in ${!LD_*} ${!PYTHON*} ${!SBATCH_*}; do unset "${poison_name}"; done
unset BASH_ENV ENV CDPATH GLOBIGNORE HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PATH=/usr/bin:/bin LC_ALL=C LANG=C

fail() { echo "[graft-pa-f14-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

archive="${GRAFT_FIELD14_SOURCE_ARCHIVE:?set GRAFT_FIELD14_SOURCE_ARCHIVE}"
archive_sha="${GRAFT_FIELD14_SOURCE_ARCHIVE_SHA256:?set GRAFT_FIELD14_SOURCE_ARCHIVE_SHA256}"
closure="${GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST:?set GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST}"
closure_sha="${GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST_SHA256:?set GRAFT_FIELD14_RUNTIME_CLOSURE_MANIFEST_SHA256}"
python_bin="${GRAFT_FIELD14_PYTHON_BIN:?set GRAFT_FIELD14_PYTHON_BIN}"
python_sha="${GRAFT_FIELD14_PYTHON_SHA256:?set GRAFT_FIELD14_PYTHON_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
plan="${GRAFT_FIELD14_PLAN:?set GRAFT_FIELD14_PLAN}"
plan_sha="${GRAFT_FIELD14_PLAN_SHA256:?set GRAFT_FIELD14_PLAN_SHA256}"
terminal="${GRAFT_FIELD14_TERMINAL_ADMISSION:?set GRAFT_FIELD14_TERMINAL_ADMISSION}"
terminal_sha="${GRAFT_FIELD14_TERMINAL_ADMISSION_SHA256:?set GRAFT_FIELD14_TERMINAL_ADMISSION_SHA256}"
materializer_runtime_sha="${GRAFT_FIELD14_TERMINAL_MATERIALIZER_RUNTIME_SHA256:?set GRAFT_FIELD14_TERMINAL_MATERIALIZER_RUNTIME_SHA256}"
output_root="${GRAFT_FIELD14_OUTPUT_ROOT:?set GRAFT_FIELD14_OUTPUT_ROOT}"
launcher_source="${GRAFT_FIELD14_LAUNCHER_SOURCE:?set GRAFT_FIELD14_LAUNCHER_SOURCE}"
launcher_sha="${GRAFT_FIELD14_LAUNCHER_SHA256:?set GRAFT_FIELD14_LAUNCHER_SHA256}"

readonly required_plan_sha=ec07f89c49dc545397d436eecbac03bb4de9126f5bf97704d41cde6a666782a2
readonly required_core_sha=dbec11e9adc80171adf071bf2d79fc8ba7498b76e5740e19b077b90a0f0a5280
readonly required_field_runner_sha=41bffcf457d745b2744f321aa70f41b840a901b4e0302d722a75fc0bb08198ed
readonly required_field_source_commit=f9ef982e976ad19ed81ed075d33c9221952945e4
readonly required_short_runner_sha=4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4
readonly required_short_source_commit=a884d357a6c0742f751be48d226ba72c952bae76
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

for name in archive closure python_bin bernini_root veomni_root checkpoint checkpoint_manifest plan terminal output_root launcher_source; do require_absolute "${!name}" "$name"; done
for name in archive_sha closure_sha python_sha checkpoint_manifest_sha plan_sha terminal_sha materializer_runtime_sha launcher_sha; do require_sha256 "${!name}" "$name"; done
[[ "${plan_sha}" == "${required_plan_sha}" ]] || fail "plan SHA differs"
[[ "${checkpoint_manifest_sha}" == "${required_checkpoint_manifest_sha}" ]] || fail "checkpoint manifest SHA differs"

archive="$(/usr/bin/realpath -e -- "${archive}")"
closure="$(/usr/bin/realpath -e -- "${closure}")"
python_bin="$(/usr/bin/realpath -e -- "${python_bin}")"
bernini_root="$(/usr/bin/realpath -e -- "${bernini_root}")"
veomni_root="$(/usr/bin/realpath -e -- "${veomni_root}")"
checkpoint="$(/usr/bin/realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(/usr/bin/realpath -e -- "${checkpoint_manifest}")"
plan="$(/usr/bin/realpath -e -- "${plan}")"
terminal="$(/usr/bin/realpath -e -- "${terminal}")"
launcher_source="$(/usr/bin/realpath -e -- "${launcher_source}")"
output_root="$(/usr/bin/realpath -m -- "${output_root}")"
[[ "${terminal}" == "${required_terminal_path}" ]] || fail "terminal path differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python is not a plain executable"
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do [[ -d "${path}" && ! -L "${path}" ]] || fail "plain directory required: ${path}"; done
for path in "${archive}" "${closure}" "${checkpoint_manifest}" "${plan}" "${terminal}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(/usr/bin/stat -Lc '%a:%h' -- "${path}")" == 444:1 ]] || fail "sealed 0444 input required: ${path}"
done
[[ -f "${launcher_source}" && ! -L "${launcher_source}" && "$(/usr/bin/stat -Lc '%a:%h' -- "${launcher_source}")" == 555:1 ]] || fail "sealed launcher required"
[[ ! -e "${output_root}" && ! -L "${output_root}" && "${output_root}" != / ]] || fail "output root must be fresh"
output_parent="$(/usr/bin/dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
[[ "${SLURM_JOB_ID:?SLURM_JOB_ID required}" =~ ^[0-9]+$ ]] || fail "job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
scratch="$(/usr/bin/mktemp -d "${scratch_parent%/}/graft-pa-f14-v1-${SLURM_JOB_ID}.XXXXXX")"
torchrun_pid=""
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${torchrun_pid}" =~ ^[0-9]+$ ]]; then kill "${torchrun_pid}" 2>/dev/null || true; wait "${torchrun_pid}" 2>/dev/null || true; fi
  if (( status != 0 )); then echo "[graft-pa-f14-v1] EXIT status=${status} output_retained=true scratch=${scratch}" >&2 || true; fi
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
    if not stat.S_ISREG(before.st_mode): raise SystemExit("not regular")
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
[[ "$(file_sha256 "${plan}")" == "${plan_sha}" ]] || fail "plan bytes differ"
[[ "$(file_sha256 "${terminal}")" == "${terminal_sha}" ]] || fail "terminal bytes differ"
[[ "$(file_sha256 "${checkpoint_manifest}")" == "${checkpoint_manifest_sha}" ]] || fail "checkpoint manifest bytes differ"
[[ "$(file_sha256 "${launcher_source}")" == "${launcher_sha}" ]] || fail "launcher bytes differ"
executed_launcher="$(/usr/bin/realpath -e -- "$0")"
[[ "$(file_sha256 "${executed_launcher}")" == "${launcher_sha}" ]] || fail "executed launcher differs"

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

"${python_bin}" -I -S -B - "${scratch}/runtime-source.tar" "${scratch}/runtime-closure.json" "${scratch}/source-tree" "${archive_sha}" "${closure_sha}" "${required_core_sha}" "${required_field_runner_sha}" "${required_short_runner_sha}" "${required_field_source_commit}" <<'PY'
import hashlib,json,os,stat,sys,tarfile
from pathlib import Path,PurePosixPath
archive,manifest_path,destination=map(Path,sys.argv[1:4]); archive_sha,manifest_sha,core_sha,runner_sha,short_runner_sha,source_commit=sys.argv[4:]
def digest(raw): return hashlib.sha256(raw).hexdigest()
if digest(archive.read_bytes())!=archive_sha or digest(manifest_path.read_bytes())!=manifest_sha: raise SystemExit("copied closure inputs differ")
raw=manifest_path.read_bytes(); manifest=json.loads(raw.decode("ascii")); canonical=json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")+b"\n"
if raw!=canonical or set(manifest)!={"files","root","schema_version","selection","source_git_commit"} or manifest.get("root")!="methods/bernini_action_editing" or manifest.get("schema_version")!="bernini-graft-phase-a-field14-runtime-python-closure-v1" or manifest.get("selection")!="commit-f9ef982-recursive-runtime-import-closure-v1" or manifest.get("source_git_commit")!=source_commit: raise SystemExit("closure manifest contract differs")
files=manifest.get("files"); expected_names={
"graft_a_lite_source_release_consumer_v1.py","graft_phase_a_field14_exact40_v1.py","graft_phase_a_native_training_closure_v1.py","graft_phase_a_native_training_closure_v2.py","identity_rebinder_v1.py","infer_lora.py","infer_native_i_axis_exact81_canary.py","infer_native_identity_generation_canary.py","infer_orderless_source_frame_set_noise_canary.py","infer_source_kv_carrier_oracle.py","infer_source_value_residual_oracle.py","inference_sigma_strata.py","native_i_axis_guidance.py","orderless_source_frame_set_noise.py","run_graft_phase_a_a_lite_short_gpu_v1.py","run_graft_phase_a_field14_exact40_gpu_v1.py","run_graft_phase_a_native_gpu_canary_v1.py","source_kv_replay.py","source_kv_route_batches.py","source_value_residual.py","tools/build_renderer_dataset.py","tools/materialize_vae.py","train_graft_phase_a_a_lite_short_v1.py","train_lora.py","tri_branch_unipc.py"}
if not isinstance(files,dict) or set(files)!=expected_names or files["graft_phase_a_field14_exact40_v1.py"]!=core_sha or files["run_graft_phase_a_field14_exact40_gpu_v1.py"]!=runner_sha or files["run_graft_phase_a_a_lite_short_gpu_v1.py"]!=short_runner_sha or any(not isinstance(v,str) or len(v)!=64 for v in files.values()): raise SystemExit("closure exact25 file map differs")
prefix=manifest["root"]+"/"; seen={}
with tarfile.open(archive,"r:") as handle:
    members=handle.getmembers()
    if len(members)!=25: raise SystemExit("archive must contain exact25 members")
    for member in members:
        path=PurePosixPath(member.name); normalized=path.as_posix().lstrip("./")
        if path.is_absolute() or ".." in path.parts or member.name!=normalized or not member.isfile() or member.issym() or member.islnk() or member.isdev() or member.isfifo() or not normalized.startswith(prefix): raise SystemExit("unsafe archive member")
        relative=normalized[len(prefix):]
        if relative in seen or relative not in files: raise SystemExit("archive inventory differs")
        stream=handle.extractfile(member)
        if stream is None: raise SystemExit("archive member unreadable")
        payload=stream.read()
        if digest(payload)!=files[relative]: raise SystemExit("archive member hash differs: "+relative)
        seen[relative]=payload
if set(seen)!=expected_names: raise SystemExit("archive exact25 coverage differs")
for relative,payload in sorted(seen.items()):
    target=destination/manifest["root"]/relative; target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o444)
    try:
        offset=0
        while offset<len(payload):
            wrote=os.write(fd,payload[offset:])
            if wrote<=0: raise SystemExit("extract write stalled")
            offset+=wrote
        os.fchmod(fd,0o444)
    finally: os.close(fd)
    compile(payload,str(target),"exec")
root=destination/manifest["root"]
for directory in sorted((p for p in root.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True): os.chmod(directory,0o555)
os.chmod(root,0o555)
PY

method_root="${scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/run_graft_phase_a_field14_exact40_gpu_v1.py"
core="${method_root}/graft_phase_a_field14_exact40_v1.py"
rebinder="${method_root}/identity_rebinder_v1.py"
/usr/bin/chmod -R a-w -- "${method_root}"
[[ "$(file_sha256 "${runner}")" == "${required_field_runner_sha}" && "$(file_sha256 "${core}")" == "${required_core_sha}" && "$(file_sha256 "${rebinder}")" == "${required_rebinder_sha}" ]] || fail "runtime source pins differ"
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
require_sha256 "${checkpoint_content_pre_sha256}" checkpoint_content_pre_sha256

/usr/bin/mkdir -- "${output_root}"
output_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${output_root}")"
exec 8<"${output_root}" || fail "could not retain output-root descriptor"
/usr/bin/mkdir -- "${output_root}/rank-logs"
rank_log_root="${output_root}/rank-logs"
rank_log_root_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${rank_log_root}")"
exec 9<"${rank_log_root}" || fail "could not retain rank-log-root descriptor"
master_port=$((30000 + (SLURM_JOB_ID % 20000)))
echo "[graft-pa-f14-v1] WORLD8=DP2xSP4 short=29,38 exact40=0..39 no-grad no-checkpoint"
set +e
"${python_bin}" -I -B -m torch.distributed.run \
  --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="${master_port}" \
  --max-restarts=0 --monitor-interval=1 --log-dir="${rank_log_root}" --redirects=3 \
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
  --plan-path "${plan}" --expected-plan-sha256 "${plan_sha}" \
  --expected-field14-core-sha256 "${required_core_sha}" \
  --expected-field14-runner-sha256 "${required_field_runner_sha}" \
  --expected-field14-source-commit "${required_field_source_commit}" \
  --ack-two-update-diagnostic-no-checkpoint-no-scientific-claim \
  --ack-exact40-no-grad-diagnostic-no-checkpoint-no-scientific-claim \
  >"${output_root}/torchrun-supervisor.log" 2>&1 8<&- 9<&- &
torchrun_pid=$!
wait "${torchrun_pid}"
status=$?
torchrun_pid=""
set -e
(( status == 0 )) || fail "WORLD8 runner failed status=${status}; logs retained"

audit_vendor "${bernini_root}" "${required_bernini_commit}" Bernini-post
audit_vendor "${veomni_root}" "${required_veomni_commit}" VeOmni-post
checkpoint_content_post_sha256="$(checkpoint_content_snapshot)"
[[ "${checkpoint_content_post_sha256}" == "${checkpoint_content_pre_sha256}" ]] || fail "checkpoint content identity changed across run"

"${python_bin}" -I -S -B - \
  "${output_root}" "${output_identity}" 8 \
  "${rank_log_root}" "${rank_log_root_identity}" 9 \
  "${plan_sha}" "${archive_sha}" "${closure_sha}" "${launcher_sha}" \
  "${required_core_sha}" "${required_field_runner_sha}" \
  "${required_field_source_commit}" "${required_short_source_commit}" \
  "${checkpoint_content_pre_sha256}" "${checkpoint_content_post_sha256}" <<'PY'
import hashlib,json,os,re,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); expected_identity=tuple(map(int,sys.argv[2].split(":"))); inherited_root_fd=int(sys.argv[3])
logs=Path(sys.argv[4]); expected_log_identity=tuple(map(int,sys.argv[5].split(":"))); inherited_log_fd=int(sys.argv[6])
plan_sha,archive_sha,closure_sha,launcher_sha,core_sha,runner_sha,field_source_commit,short_source_commit,checkpoint_pre,checkpoint_post=sys.argv[7:]
SHA=re.compile(r"[0-9a-f]{64}\Z")
AUTHORITY=("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","checkpoint_authority","publication_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed","semantic_action_editing_success_claimed")
CONSUMER_AUTHORITY=("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed")
FIELD_ROLES=("source_noop_target_velocity","correct_atlas_noop_velocity","wrong_atlas_noop_velocity","dropped_atlas_noop_velocity","correct_atlas_action_velocity","dropped_atlas_action_velocity")
FAMILIES=("dog","human")
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def sealed(v,label):
    if not isinstance(v,dict): raise SystemExit(label+" not mapping")
    unsigned=dict(v); claimed=unsigned.pop("digest",None)
    if not isinstance(claimed,str) or SHA.fullmatch(claimed) is None or claimed!=digest(unsigned): raise SystemExit(label+" digest differs")
def require_sha(value,label):
    if not isinstance(value,str) or SHA.fullmatch(value) is None: raise SystemExit(label+" is not SHA256")
    return value
def deny(v):
    if isinstance(v,dict):
        for k,item in v.items():
            if k=="authority":
                if not isinstance(item,dict) or set(item) not in (set(AUTHORITY),set(CONSUMER_AUTHORITY)) or any(value is not False for value in item.values()): raise SystemExit("authority mapping differs")
            elif k in AUTHORITY or k.endswith("_authorized") or "authority" in k:
                if item is not False: raise SystemExit("authority-shaped value differs: "+k)
            if k in {"checkpoint_written","checkpoint_payload_returned","publication_performed"} and item is not False: raise SystemExit("artifact publication claimed")
            deny(item)
    elif isinstance(v,list):
        for item in v: deny(item)
def exact_top_authority(value,label):
    if any(value.get(name) is not False for name in AUTHORITY): raise SystemExit(label+" top-level exact12 authority differs")
root_fd=os.dup(inherited_root_fd)
root_info=os.fstat(root_fd); root_leaf=root.lstat()
if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink() or (root_info.st_dev,root_info.st_ino)!=expected_identity or (root_leaf.st_dev,root_leaf.st_ino)!=expected_identity: raise SystemExit("output root identity differs")
if logs!=root/"rank-logs": raise SystemExit("rank-log root path differs")
log_fd=os.dup(inherited_log_fd)
log_info=os.fstat(log_fd); log_leaf=logs.lstat()
if not stat.S_ISDIR(log_info.st_mode) or logs.is_symlink() or (log_info.st_dev,log_info.st_ino)!=expected_log_identity or (log_leaf.st_dev,log_leaf.st_ino)!=expected_log_identity: raise SystemExit("rank-log root identity differs")
stdout_files=[]
for node in logs.rglob("*"):
    meta=node.lstat()
    if stat.S_ISLNK(meta.st_mode) or not (stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode)): raise SystemExit("rank-log tree contains unsafe node")
    if stat.S_ISREG(meta.st_mode) and node.name=="stdout.log": stdout_files.append(node.relative_to(logs))
ranks={}
for relative in stdout_files:
    try: rank=int(relative.parent.name)
    except ValueError: raise SystemExit("rank stdout parent is not numeric")
    if rank not in range(8) or rank in ranks: raise SystemExit("rank stdout coverage differs")
    ranks[rank]=relative
if set(ranks)!=set(range(8)): raise SystemExit("exact eight rank stdout logs required")
admitted=[]
for rank,relative in sorted(ranks.items()):
    parts=relative.parts
    if not parts or parts[-1]!="stdout.log" or any(part in {"",".",".."} for part in parts): raise SystemExit("rank stdout relative path differs")
    current=os.dup(log_fd)
    try:
        for part in parts[:-1]:
            next_fd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=current)
            os.close(current); current=next_fd
        stream=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=current)
        try:
            before=os.fstat(stream)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>512*1024*1024: raise SystemExit("rank stdout is not bounded plain link-count-one")
            chunks=[]; total=0
            while True:
                block=os.read(stream,1024*1024)
                if not block: break
                total+=len(block)
                if total>512*1024*1024: raise SystemExit("rank stdout exceeds bound")
                chunks.append(block)
            after=os.fstat(stream)
            fields=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode,value.st_nlink)
            if fields(before)!=fields(after): raise SystemExit("rank stdout changed while reading")
            raw=b"".join(chunks)
        finally: os.close(stream)
    finally: os.close(current)
    for line in raw.splitlines():
        try: value=json.loads(line.decode("ascii"))
        except (UnicodeDecodeError,json.JSONDecodeError): continue
        if isinstance(value,dict) and value.get("schema_version")=="bernini-graft-phase-a-field14-exact40-gpu-runner-v1":
            if line!=canonical(value): raise SystemExit("runner JSON is not canonical")
            admitted.append((rank,value))
final_log_fd=os.fstat(log_fd); final_log_leaf=logs.lstat(); os.close(log_fd)
if (final_log_fd.st_dev,final_log_fd.st_ino)!=(log_info.st_dev,log_info.st_ino) or (final_log_leaf.st_dev,final_log_leaf.st_ino)!=expected_log_identity: raise SystemExit("rank-log root changed")
if len(admitted)!=1 or admitted[0][0]!=0: raise SystemExit("exact one rank-zero field14 result required")
result=admitted[0][1]; sealed(result,"runner result"); deny(result)
result_keys={"schema_version","status","world8","topology_receipt","source_binding","plan","plan_sha256","checkpoint_identity","initialization","local_short_result","local_field14_result","base_sha256_before","base_sha256_after","base_bytes_unchanged","base_gradients_all_none","trainable_sha256_after_short_before_sweep","trainable_sha256_after_sweep","trainable_bytes_unchanged_during_sweep","wan_diffusion_sha256","transformer_wan_sha256","runtime_versions","dependency_afterok_is_queue_gate_only","weights_inherited_from_dependency_job","checkpoint_written","checkpoint_payload_returned","publication_performed","digest"}|set(AUTHORITY)
if set(result)!=result_keys or result.get("status")!="completed_in_memory_short_then_exact40_no_checkpoint" or result.get("base_bytes_unchanged") is not True or result.get("base_gradients_all_none") is not True or result.get("trainable_bytes_unchanged_during_sweep") is not True or result.get("base_sha256_before")!=result.get("base_sha256_after") or result.get("trainable_sha256_after_short_before_sweep")!=result.get("trainable_sha256_after_sweep") or result.get("dependency_afterok_is_queue_gate_only") is not True or result.get("weights_inherited_from_dependency_job") is not False or result.get("checkpoint_written") is not False or result.get("checkpoint_payload_returned") is not False or result.get("publication_performed") is not False or result.get("plan_sha256")!=plan_sha: raise SystemExit("runner closure differs")
exact_top_authority(result,"runner result")
binding=result.get("source_binding"); sealed(binding,"source binding")
if binding.get("field14_core_sha256")!=core_sha or binding.get("field14_runner_sha256")!=runner_sha or binding.get("field14_source_commit")!=field_source_commit or binding.get("short_source_commit")!=short_source_commit or binding.get("plan_sha256")!=plan_sha: raise SystemExit("runner source binding differs")
checkpoint_identity=result.get("checkpoint_identity"); sealed(checkpoint_identity,"checkpoint identity")
if not isinstance(checkpoint_identity.get("identity"),dict) or digest(checkpoint_identity["identity"])!=checkpoint_pre or checkpoint_pre!=checkpoint_post: raise SystemExit("checkpoint full content rehash differs")
world=result.get("world8"); sealed(world,"WORLD8 result")
world_keys={"schema_version","rank_order","dp2_family_order","rows","all_eight_field14_results","arm_representatives","short_world8_full_results","all_eight_exact40_completed","all_eight_trainable_bytes_unchanged_during_sweep","all_eight_base_bytes_unchanged_entire_process","both_sp4_arms_exact_field_hash_and_metric_consensus","checkpoint_written","checkpoint_payload_returned","publication_performed","digest"}|set(AUTHORITY)
summaries=world.get("rows"); sweeps=world.get("all_eight_field14_results"); arms=world.get("arm_representatives")
if set(world)!=world_keys or world.get("schema_version")!="bernini-graft-phase-a-field14-world8-result-v1" or world.get("rank_order")!=list(range(8)) or world.get("dp2_family_order")!=list(FAMILIES) or not isinstance(summaries,list) or len(summaries)!=8 or not isinstance(sweeps,list) or len(sweeps)!=8 or not isinstance(arms,list) or len(arms)!=2 or world.get("all_eight_exact40_completed") is not True or world.get("all_eight_trainable_bytes_unchanged_during_sweep") is not True or world.get("all_eight_base_bytes_unchanged_entire_process") is not True or world.get("both_sp4_arms_exact_field_hash_and_metric_consensus") is not True or world.get("checkpoint_written") is not False or world.get("checkpoint_payload_returned") is not False or world.get("publication_performed") is not False: raise SystemExit("WORLD8 coverage differs")
exact_top_authority(world,"WORLD8 result")
sweep_keys={"schema_version","status","family","confirmation_iid","confirmation_source_sha256","wrong_owner_iid","wrong_owner_source_sha256","short_result_digest","preinstall_baseline_digest","schedule_indices","inactive_indices","active_indices","field_roles","rows","exact40_official_order","ambient_torch_no_grad","one_index_admitted_hashed_and_released_before_next","cross_index_tensor_retention","cross_index_compensation_used","cross_index_selection_used","semantic_metrics_are_diagnostic_only","checkpoint_written","checkpoint_payload_returned","publication_performed","digest"}|set(AUTHORITY)
summary_keys={"global_rank","dp_arm","sp_rank","family","short_result_digest","field14_result_digest","preinstall_baseline_digest","trainable_sha256_after_short_before_sweep","trainable_sha256_after_sweep","base_sha256_before","base_sha256_after"}
row_keys={"schedule_index","admission_digest","field_tensor_sha256","semantic_metrics_digest","provenance_digest","release_digest","all_field_tensor_objects_released"}
for rank,(summary,sweep) in enumerate(zip(summaries,sweeps)):
    arm=rank//4; sp_rank=rank%4; sealed(sweep,"rank%d exact40"%rank); exact_top_authority(sweep,"rank%d exact40"%rank)
    if set(sweep)!=sweep_keys or sweep.get("status")!="completed_in_memory_exact40_no_grad_no_checkpoint" or sweep.get("family")!=FAMILIES[arm] or sweep.get("schedule_indices")!=list(range(40)) or sweep.get("inactive_indices")!=list(range(26)) or sweep.get("active_indices")!=list(range(26,40)) or sweep.get("field_roles")!=list(FIELD_ROLES) or not isinstance(sweep.get("rows"),list) or len(sweep["rows"])!=40 or sweep.get("exact40_official_order") is not True or sweep.get("ambient_torch_no_grad") is not True or sweep.get("one_index_admitted_hashed_and_released_before_next") is not True or sweep.get("cross_index_tensor_retention") is not False or sweep.get("cross_index_compensation_used") is not False or sweep.get("cross_index_selection_used") is not False or sweep.get("semantic_metrics_are_diagnostic_only") is not True or sweep.get("checkpoint_written") is not False or sweep.get("checkpoint_payload_returned") is not False or sweep.get("publication_performed") is not False: raise SystemExit("rank%d exact40 contract differs"%rank)
    for name in ("confirmation_source_sha256","wrong_owner_source_sha256","short_result_digest","preinstall_baseline_digest"): require_sha(sweep.get(name),"rank%d %s"%(rank,name))
    if sweep.get("confirmation_source_sha256")==sweep.get("wrong_owner_source_sha256"): raise SystemExit("rank%d source identity differs"%rank)
    for index,row in enumerate(sweep["rows"]):
        if not isinstance(row,dict) or set(row)!=row_keys or row.get("schedule_index")!=index or not isinstance(row.get("field_tensor_sha256"),dict) or set(row["field_tensor_sha256"])!=set(FIELD_ROLES) or row.get("all_field_tensor_objects_released") is not True: raise SystemExit("rank%d index%d receipt schema differs"%(rank,index))
        for value in row["field_tensor_sha256"].values(): require_sha(value,"field tensor")
        for name in ("admission_digest","semantic_metrics_digest","provenance_digest","release_digest"): require_sha(row.get(name),"rank%d index%d %s"%(rank,index,name))
    if not isinstance(summary,dict) or set(summary)!=summary_keys or summary.get("global_rank")!=rank or summary.get("dp_arm")!=arm or summary.get("sp_rank")!=sp_rank or summary.get("family")!=FAMILIES[arm] or summary.get("field14_result_digest")!=sweep["digest"] or summary.get("short_result_digest")!=sweep["short_result_digest"] or summary.get("preinstall_baseline_digest")!=sweep["preinstall_baseline_digest"] or summary.get("trainable_sha256_after_short_before_sweep")!=summary.get("trainable_sha256_after_sweep") or summary.get("base_sha256_before")!=summary.get("base_sha256_after"): raise SystemExit("rank%d WORLD8 summary differs"%rank)
    deny(sweep)
arm_keys={"dp_arm","family","global_ranks","representative_global_rank","representative_field14_result","per_rank_field14_result_digests","field_hash_and_metric_consensus_digest","sp4_exact_field_hash_and_metric_consensus"}
for arm,representative_rank in enumerate((0,4)):
    representative=arms[arm]; group=sweeps[arm*4:(arm+1)*4]
    records=[[{"schedule_index":item["schedule_index"],"field_tensor_sha256":item["field_tensor_sha256"],"semantic_metrics_digest":item["semantic_metrics_digest"]} for item in sweep["rows"]] for sweep in group]
    if not isinstance(representative,dict) or set(representative)!=arm_keys or representative.get("dp_arm")!=arm or representative.get("family")!=FAMILIES[arm] or representative.get("global_ranks")!=list(range(arm*4,arm*4+4)) or representative.get("representative_global_rank")!=representative_rank or representative.get("representative_field14_result")!=sweeps[representative_rank] or representative.get("per_rank_field14_result_digests")!=[item["digest"] for item in group] or representative.get("field_hash_and_metric_consensus_digest")!=digest(records[0]) or representative.get("sp4_exact_field_hash_and_metric_consensus") is not True or any(canonical(record)!=canonical(records[0]) for record in records[1:]): raise SystemExit("arm%d SP4 exact40 consensus differs"%arm)
local=result.get("local_field14_result")
if local!=sweeps[0]: raise SystemExit("outer local exact40 is not rank0 full receipt")
short_world8=world.get("short_world8_full_results"); sealed(short_world8,"nested short WORLD8 full results"); deny(short_world8)
final_root_fd=os.fstat(root_fd); final_root_leaf=root.lstat()
if (final_root_fd.st_dev,final_root_fd.st_ino)!=expected_identity or (final_root_leaf.st_dev,final_root_leaf.st_ino)!=expected_identity: raise SystemExit("output root changed before receipt publication")
core={"schema_version":"bernini-graft-phase-a-field14-world8-parent-v2","status":"completed_diagnostic_no_checkpoint","complete":True,"pass":True,"job_id":os.environ.get("SLURM_JOB_ID"),"runner_result_digest":result["digest"],"runtime":{"field14_core_sha256":core_sha,"field14_runner_sha256":runner_sha,"field14_source_commit":field_source_commit,"short_source_commit":short_source_commit,"plan_sha256":plan_sha,"source_archive_sha256":archive_sha,"runtime_closure_manifest_sha256":closure_sha,"launcher_sha256":launcher_sha,"checkpoint_content_pre_sha256":checkpoint_pre,"checkpoint_content_post_sha256":checkpoint_post},"validated":{"same_process_short_then_exact40":True,"exact40_official_order":True,"inactive_0_25_zero_gate_preinstall_parity_producer_validated":True,"active_26_39_finite_nonzero_gate_producer_validated":True,"all_eight_full_field14_receipts_deeply_validated":True,"both_sp4_per_index_field_hash_and_metric_consensus_recomputed":True,"rank_log_root_identity_retained_from_before_torchrun":True,"rank_stdout_opened_relative_to_retained_root_with_O_NOFOLLOW":True,"output_root_identity_retained_from_creation":True,"checkpoint_content_full_rehash_pre_and_post":True,"per_index_hash_then_release":True,"cross_index_compensation_or_selection":False,"semantic_metrics_authoritative":False},"runner_result":result,"checkpoint_written":False,"checkpoint_payload_returned":False,"publication_performed":False,"authority":{name:False for name in AUTHORITY},"receipt_publication":{"create_only_O_EXCL":True,"provisional_mode":"0600","final_success_mode":"0444","mode_0444_is_terminal_success_transition":True,"canonical_ascii_json_newline":True,"output_root_identity_retained_from_creation":True}}
receipt={**core,"receipt_digest":hashlib.sha256(canonical(core)).hexdigest()}; payload=canonical(receipt)+b"\n"
fd=os.open("receipt.json",os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=root_fd)
try:
    offset=0
    while offset<len(payload):
        wrote=os.write(fd,payload[offset:])
        if wrote<=0: raise SystemExit("receipt write stalled")
        offset+=wrote
    os.fsync(fd); os.lseek(fd,0,os.SEEK_SET); observed=b""
    while True:
        block=os.read(fd,1024*1024)
        if not block: break
        observed+=block
    provisional=os.fstat(fd); path_info=os.stat("receipt.json",dir_fd=root_fd,follow_symlinks=False)
    if observed!=payload or not stat.S_ISREG(provisional.st_mode) or stat.S_IMODE(provisional.st_mode)!=0o600 or provisional.st_nlink!=1 or provisional.st_size!=len(payload) or (provisional.st_dev,provisional.st_ino)!=(path_info.st_dev,path_info.st_ino): raise SystemExit("provisional receipt transaction differs")
    os.fsync(root_fd); os.fchmod(fd,0o444); os.fsync(fd); os.fsync(root_fd)
    final=os.fstat(fd); path_info=os.stat("receipt.json",dir_fd=root_fd,follow_symlinks=False)
    if stat.S_IMODE(final.st_mode)!=0o444 or final.st_nlink!=1 or (final.st_dev,final.st_ino)!=(path_info.st_dev,path_info.st_ino): raise SystemExit("terminal receipt transition differs")
    os.close(fd); os.close(root_fd); os._exit(0)
except BaseException:
    try:
        info=os.fstat(fd); leaf=os.stat("receipt.json",dir_fd=root_fd,follow_symlinks=False)
        if (info.st_dev,info.st_ino)==(leaf.st_dev,leaf.st_ino) and stat.S_IMODE(info.st_mode)!=0o444: os.unlink("receipt.json",dir_fd=root_fd)
    except OSError: pass
    os.close(fd); os.close(root_fd); raise
PY

/usr/bin/find "${output_root}" -type f -exec /usr/bin/chmod 0444 -- {} +
/usr/bin/find "${output_root}" -type d -exec /usr/bin/chmod 0555 -- {} +
echo "[graft-pa-f14-v1] COMPLETE receipt=${output_root}/receipt.json"
