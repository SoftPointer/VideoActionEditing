#!/bin/bash -p
#SBATCH --job-name=graft-pa-short-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# Sealed single-node WORLD8/DP2/SP4 launcher for the two-update A-lite
# diagnostic.  The runner writes no checkpoint and emits its sole result on
# rank-zero stdout.  This launcher redirects every rank separately, admits
# only the exact sealed result schema, and publishes one create-only parent
# diagnostic receipt after checkpoint/vendor postflight.

case "$-" in
  *p*) ;;
  *) echo "[graft-pa-short-v1] ERROR: Bash privileged mode is required" >&2; exit 2 ;;
esac
set -Eeuo pipefail
umask 077

for poison_name in ${!LD_*} ${!PYTHON*} ${!SBATCH_*}; do
  unset "${poison_name}"
done
unset BASH_ENV ENV CDPATH GLOBIGNORE HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export PATH=/usr/bin:/bin LC_ALL=C LANG=C

fail() { echo "[graft-pa-short-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

source_archive="${GRAFT_SHORT_SOURCE_ARCHIVE:?set GRAFT_SHORT_SOURCE_ARCHIVE}"
source_archive_sha256="${GRAFT_SHORT_SOURCE_ARCHIVE_SHA256:?set GRAFT_SHORT_SOURCE_ARCHIVE_SHA256}"
closure_manifest="${GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST:?set GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST}"
closure_manifest_sha256="${GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256:?set GRAFT_SHORT_RUNTIME_CLOSURE_MANIFEST_SHA256}"
python_bin="${GRAFT_SHORT_PYTHON_BIN:?set GRAFT_SHORT_PYTHON_BIN}"
python_sha256="${GRAFT_SHORT_PYTHON_SHA256:?set GRAFT_SHORT_PYTHON_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
plan_path="${GRAFT_SHORT_PLAN:?set GRAFT_SHORT_PLAN}"
plan_sha256="${GRAFT_SHORT_PLAN_SHA256:?set GRAFT_SHORT_PLAN_SHA256}"
terminal_admission="${GRAFT_SHORT_TERMINAL_ADMISSION:?set GRAFT_SHORT_TERMINAL_ADMISSION}"
terminal_admission_sha256="${GRAFT_SHORT_TERMINAL_ADMISSION_SHA256:?set GRAFT_SHORT_TERMINAL_ADMISSION_SHA256}"
terminal_materializer_runtime_sha256="${GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256:?set GRAFT_SHORT_TERMINAL_MATERIALIZER_RUNTIME_SHA256}"
output_root="${GRAFT_SHORT_OUTPUT_ROOT:?set GRAFT_SHORT_OUTPUT_ROOT}"
launcher_source="${GRAFT_SHORT_LAUNCHER_SOURCE:?set GRAFT_SHORT_LAUNCHER_SOURCE}"
launcher_sha256="${GRAFT_SHORT_LAUNCHER_SHA256:?set GRAFT_SHORT_LAUNCHER_SHA256}"

readonly required_plan_sha256=ab2eb2e7b93341b47498184821761eb8e5c924f9dd8460284087e23a27ba34d8
readonly required_source_commit=a884d357a6c0742f751be48d226ba72c952bae76
readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_runner_sha256=4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4
readonly required_rebinder_sha256=1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0
readonly required_terminal_materializer_implementation_sha256=4686463642a38e771c6858d1c10fc6aacb815a56e4f3eae951a336018d186cf4
readonly required_release_stem=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4
readonly required_terminal_path=${required_release_stem}.terminal.admission.receipt.json
readonly manifest_path=${required_release_stem}.manifest.jsonl
readonly producer_receipt_path=${required_release_stem}.receipt.json
readonly execution_receipt_path=${required_release_stem}.execution.receipt.json
readonly submission_receipt_path=${required_release_stem}.submission.receipt.json
readonly manifest_sha256=0bfe9447fb3649625a4dface5c93ba0e55530285f9d63e7b15552b0bb4f9b1af
readonly producer_receipt_sha256=01a87422753c5946c84df7c26198be992793de536159b1e427d4a00ab91ffea6
readonly execution_receipt_sha256=101a9cb25a4c898be0ed1b0406a3aec866a153d12ff79c6b890fb78c1146efd6
readonly submission_receipt_sha256=35ddc72443361ee588e5bd5c852df144293036a6128ae7fc9fab20e600e95b82

for name in source_archive closure_manifest python_bin bernini_root veomni_root checkpoint checkpoint_manifest plan_path terminal_admission output_root launcher_source; do
  require_absolute "${!name}" "${name}"
done
for name in source_archive_sha256 closure_manifest_sha256 python_sha256 checkpoint_manifest_sha256 plan_sha256 terminal_admission_sha256 terminal_materializer_runtime_sha256 launcher_sha256; do
  require_sha256 "${!name}" "${name}"
done
[[ "${plan_sha256}" == "${required_plan_sha256}" ]] || fail "plan digest differs"
[[ "${checkpoint_manifest_sha256}" == "${required_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest authority differs"

source_archive="$(/usr/bin/realpath -e -- "${source_archive}")"
closure_manifest="$(/usr/bin/realpath -e -- "${closure_manifest}")"
python_bin="$(/usr/bin/realpath -e -- "${python_bin}")"
bernini_root="$(/usr/bin/realpath -e -- "${bernini_root}")"
veomni_root="$(/usr/bin/realpath -e -- "${veomni_root}")"
checkpoint="$(/usr/bin/realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(/usr/bin/realpath -e -- "${checkpoint_manifest}")"
plan_path="$(/usr/bin/realpath -e -- "${plan_path}")"
terminal_admission="$(/usr/bin/realpath -e -- "${terminal_admission}")"
launcher_source="$(/usr/bin/realpath -e -- "${launcher_source}")"
output_root="$(/usr/bin/realpath -m -- "${output_root}")"
[[ "${terminal_admission}" == "${required_terminal_path}" ]] || fail "terminal admission path differs from fixed plan"

[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python must be an executable plain file"
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "plain vendor/checkpoint directory required: ${path}"
done
for path in "${source_archive}" "${closure_manifest}" "${checkpoint_manifest}" "${plan_path}" "${terminal_admission}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain sealed file required: ${path}"
  [[ "$(/usr/bin/stat -Lc '%a:%h' -- "${path}")" == 444:1 ]] || fail "sealed input must be mode-0444 link-count-one: ${path}"
done
[[ -f "${launcher_source}" && ! -L "${launcher_source}" && "$(/usr/bin/stat -Lc '%a:%h' -- "${launcher_source}")" == 555:1 ]] || fail "launcher source must be plain 0555 link-count-one"
[[ ! -e "${output_root}" && ! -L "${output_root}" && "${output_root}" != / ]] || fail "output root must be fresh and non-root"
output_parent="$(/usr/bin/dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(/usr/bin/mktemp -d "${scratch_parent%/}/graft-pa-short-v1-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${task_scratch}")"
builtin cd -- "${task_scratch}"
[[ "$(/bin/pwd -P)" == "${task_scratch}" && "$(/usr/bin/stat -Lc '%d:%i' -- .)" == "${task_scratch_identity}" ]] || fail "isolated cwd identity differs"
torchrun_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${torchrun_pid}" =~ ^[0-9]+$ ]]; then
    kill "${torchrun_pid}" 2>/dev/null || true
    wait "${torchrun_pid}" 2>/dev/null || true
  fi
  # Parent mode 0444 is the final success transition.  On success, do no
  # fallible I/O after that publication.  Failure diagnostics are best effort
  # and may not replace the original exit status.
  if (( status != 0 )); then
    echo "[graft-pa-short-v1] EXIT status=${status} sibling_teardown_requested=true output_retained=true scratch_preserved=${task_scratch}" >&2 || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

file_sha256() {
  "${python_bin}" -I -S -B - "$1" <<'PY'
import hashlib, os, stat, sys
p = sys.argv[1]
fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
h = hashlib.sha256()
try:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode): raise SystemExit("not plain")
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block: break
        h.update(block)
    after = os.fstat(fd)
    fields = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink)
    if fields(before) != fields(after): raise SystemExit("changed while hashing")
finally:
    os.close(fd)
print(h.hexdigest())
PY
}

[[ "$(file_sha256 "${source_archive}")" == "${source_archive_sha256}" ]] || fail "source archive bytes differ"
[[ "$(file_sha256 "${closure_manifest}")" == "${closure_manifest_sha256}" ]] || fail "runtime closure manifest bytes differ"
[[ "$(file_sha256 "${plan_path}")" == "${plan_sha256}" ]] || fail "plan bytes differ"
[[ "$(file_sha256 "${terminal_admission}")" == "${terminal_admission_sha256}" ]] || fail "terminal admission differs from external trust anchor"
[[ "$(file_sha256 "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest bytes differ"
[[ "$(file_sha256 "${launcher_source}")" == "${launcher_sha256}" ]] || fail "launcher source bytes differ"
[[ "$(file_sha256 "${python_bin}")" == "${python_sha256}" ]] || fail "configured Python bytes differ"
executed_launcher="$(/usr/bin/realpath -e -- "$0")"
[[ -f "${executed_launcher}" && ! -L "${executed_launcher}" && "$(file_sha256 "${executed_launcher}")" == "${launcher_sha256}" ]] || fail "executed launcher differs from sealed source"

audit_vendor_tree() {
  local root="$1" expected_commit="$2" label="$3" status="" candidate="" relative=""
  local symlink_scan="${task_scratch}/vendor-symlink-${label}.$$.nul"
  local import_scan="${task_scratch}/vendor-import-${label}.$$.nul"
  [[ "$(/usr/bin/git -C "${root}" rev-parse HEAD)" == "${expected_commit}" ]] || fail "${label} commit differs"
  /usr/bin/find "${root}" -path "${root}/.git" -prune -o -type l -print0 >"${symlink_scan}" || fail "${label} symlink scan failed"
  while IFS= read -r -d '' candidate; do relative="${candidate#"${root}/"}"; fail "${label} has symlink outside .git: ${relative}"; done <"${symlink_scan}"
  status="$(/usr/bin/git -C "${root}" status --porcelain=v1 --untracked-files=all)" || fail "${label} git status audit failed"
  [[ -z "${status}" ]] || fail "${label} has tracked changes or untracked shadow files"
  /usr/bin/find "${root}" -path "${root}/.git" -prune -o \( -name '*.py' -o -name '*.pyc' -o -name '*.so' -o -name '*.pth' \) -print0 >"${import_scan}" || fail "${label} import-node scan failed"
  while IFS= read -r -d '' candidate; do
    relative="${candidate#"${root}/"}"
    [[ -f "${candidate}" && ! -L "${candidate}" ]] || fail "${label} has non-plain import node: ${relative}"
    /usr/bin/git -C "${root}" ls-files --error-unmatch -- "${relative}" >/dev/null 2>&1 || fail "${label} has untracked import shadow: ${relative}"
  done <"${import_scan}"
}
audit_vendor_tree "${bernini_root}" "${required_bernini_commit}" Bernini-pre
audit_vendor_tree "${veomni_root}" "${required_veomni_commit}" VeOmni-pre

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 KMP_USE_SHM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
/usr/bin/mkdir -p -- "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TMPDIR}"

/usr/bin/cp -- "${source_archive}" "${task_scratch}/runtime-source.tar"
/usr/bin/cp -- "${closure_manifest}" "${task_scratch}/runtime-closure.json"
/usr/bin/cp -- "${plan_path}" "${task_scratch}/plan.json"
/usr/bin/cp -- "${terminal_admission}" "${task_scratch}/terminal.json"
/usr/bin/chmod 0444 -- "${task_scratch}/runtime-source.tar" "${task_scratch}/runtime-closure.json" "${task_scratch}/plan.json" "${task_scratch}/terminal.json"
/usr/bin/mkdir -- "${task_scratch}/source-tree"

"${python_bin}" -I -S -B - \
  "${task_scratch}/runtime-source.tar" "${task_scratch}/runtime-closure.json" \
  "${task_scratch}/source-tree" "${source_archive_sha256}" "${closure_manifest_sha256}" <<'PY'
# BEGIN GRAFT_PHASE_A_SHORT_ARCHIVE_PREFLIGHT_V1
import ast, hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath

archive, manifest_path, destination = map(Path, sys.argv[1:4])
archive_sha, manifest_sha = sys.argv[4:6]
commit = "a884d357a6c0742f751be48d226ba72c952bae76"
root_name = "methods/bernini_action_editing"
expected = {
    "graft_a_lite_source_release_consumer_v1.py": "13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75",
    "graft_phase_a_native_training_closure_v1.py": "36861e8670fc77d65b469f86aa472a10d453f9ba5fc227a62303329c4c38409a",
    "graft_phase_a_native_training_closure_v2.py": "bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f",
    "identity_rebinder_v1.py": "1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0",
    "infer_lora.py": "ce0bb91aa1850fa4568b6441cc4be4f41db8b8dcfc2afe2d9fcc76a6fca2ebe4",
    "infer_native_i_axis_exact81_canary.py": "c3373d819534ac1e08270356b3311e9ef115390932bdbaa6dca64984ab618c58",
    "infer_native_identity_generation_canary.py": "a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334",
    "infer_orderless_source_frame_set_noise_canary.py": "f1e0182b6f9f6e2c4da4838b4f74f1ee629541887684a07424ec912e9050921c",
    "infer_source_kv_carrier_oracle.py": "fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604",
    "infer_source_value_residual_oracle.py": "40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8",
    "inference_sigma_strata.py": "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
    "native_i_axis_guidance.py": "6a7cb379403aca0f6010f14477300ad6af6d3ba8e4581ba1c5890e6ff6fed931",
    "orderless_source_frame_set_noise.py": "b5ad305798289fcf9112ce004f0305c5fea7f206c9cd9e6b331908c2f9af7bd0",
    "run_graft_phase_a_a_lite_short_gpu_v1.py": "4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4",
    "run_graft_phase_a_native_gpu_canary_v1.py": "e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a",
    "source_kv_replay.py": "45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3",
    "source_kv_route_batches.py": "7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e",
    "source_value_residual.py": "420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124",
    "tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "train_graft_phase_a_a_lite_short_v1.py": "73e39048bb8836fef33516eb1aae4cbc3f9fa4ecefcfb5d2695925bcb150f7bb",
    "train_lora.py": "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
    "tri_branch_unipc.py": "58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2",
}

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

if digest(archive) != archive_sha or digest(manifest_path) != manifest_sha: raise SystemExit("copied archive/manifest hash differs")
raw_manifest=manifest_path.read_bytes()
manifest=json.loads(raw_manifest.decode("ascii"))
canonical=json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")+b"\n"
if raw_manifest != canonical: raise SystemExit("closure manifest is not canonical ASCII JSON newline")
if manifest != {"files":expected,"root":root_name,"schema_version":"bernini-graft-phase-a-short-runtime-python-closure-v2","selection":"commit-a884d35-recursive-runtime-import-closure-v2","source_git_commit":commit}: raise SystemExit("runtime closure manifest differs")
prefix=root_name+"/"; seen={}
with tarfile.open(archive,"r:*") as handle:
    members=handle.getmembers()
    if len(members) != len(expected): raise SystemExit("runtime archive member count differs")
    for member in members:
        path=PurePosixPath(member.name); normalized=path.as_posix().lstrip("./")
        if path.is_absolute() or ".." in path.parts or member.name != normalized or not member.isfile() or member.issym() or member.islnk() or member.isdev() or member.isfifo() or not normalized.startswith(prefix): raise SystemExit("archive must contain exactly 23 safe regular members and no directories")
        relative=normalized[len(prefix):]
        if relative in seen or relative not in expected: raise SystemExit("archive file set/duplicate differs")
        extracted=handle.extractfile(member)
        if extracted is None: raise SystemExit("archive member unreadable")
        payload=extracted.read()
        if hashlib.sha256(payload).hexdigest()!=expected[relative]: raise SystemExit("archive committed bytes differ: "+relative)
        seen[relative]=payload
if set(seen)!=set(expected): raise SystemExit("archive lacks exact recursive closure")
for relative,payload in sorted(seen.items()):
    target=destination/root_name/relative; target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o444)
    try:
        offset=0
        while offset<len(payload):
            wrote=os.write(fd,payload[offset:])
            if wrote<=0: raise SystemExit("extraction stalled")
            offset+=wrote
        os.fchmod(fd,0o444)
    finally: os.close(fd)
reachable=set(); queue=["run_graft_phase_a_a_lite_short_gpu_v1.py"]
while queue:
    relative=queue.pop(0)
    if relative in reachable: continue
    reachable.add(relative); tree=ast.parse(seen[relative],filename=relative)
    for node in ast.walk(tree):
        names=[]
        if isinstance(node,ast.Import): names=[alias.name for alias in node.names]
        elif isinstance(node,ast.ImportFrom) and node.module: names=[node.module]
        for name in names:
            for candidate in (name.replace(".","/")+".py",name.split(".")[0]+".py"):
                if candidate in expected and candidate not in reachable: queue.append(candidate)
static_expected=set(expected)-{"tools/materialize_vae.py","tools/build_renderer_dataset.py"}
if reachable != static_expected: raise SystemExit("complete recursive static closure differs")
if b"from tools import materialize_vae" not in seen["infer_lora.py"] or b"from tools import build_renderer_dataset as raw_builder" not in seen["tools/materialize_vae.py"]: raise SystemExit("dynamic source closure edge differs")
for relative,payload in seen.items(): compile(payload,str(destination/root_name/relative),"exec")
runtime_root=destination/root_name
for directory in sorted((p for p in runtime_root.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True): os.chmod(directory,0o555)
os.chmod(runtime_root,0o555)
# END GRAFT_PHASE_A_SHORT_ARCHIVE_PREFLIGHT_V1
PY

method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/run_graft_phase_a_a_lite_short_gpu_v1.py"
rebinder="${method_root}/identity_rebinder_v1.py"
/usr/bin/chmod -R a-w -- "${method_root}"
[[ "$(file_sha256 "${runner}")" == "${required_runner_sha256}" && "$(file_sha256 "${rebinder}")" == "${required_rebinder_sha256}" ]] || fail "sealed runner/rebinder differs"
if /usr/bin/find "${method_root}" \( -type l -o -name '__pycache__' -o -name '*.pyc' \) -print -quit | /usr/bin/grep -q .; then fail "sealed source tree contains symlink/pycache"; fi

# The plan is an active gate: compare every execution-defining field to the
# sealed runner/consumer constants and every release artifact to fixed pins.
"${python_bin}" -I -B - "${method_root}" "${task_scratch}/plan.json" "${plan_sha256}" "${terminal_admission}" "${terminal_admission_sha256}" "${terminal_materializer_runtime_sha256}" <<'PY'
import hashlib,json,os,re,stat,sys
from pathlib import Path
root,plan_path=Path(sys.argv[1]),Path(sys.argv[2]); plan_sha,terminal_path,terminal_sha,runtime_sha=sys.argv[3:]
sys.path.insert(0,str(root))
import run_graft_phase_a_a_lite_short_gpu_v1 as runner
import graft_a_lite_source_release_consumer_v1 as consumer
raw=plan_path.read_bytes(); plan=json.loads(raw.decode("ascii")); canonical=json.dumps(plan,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")+b"\n"
if raw!=canonical or hashlib.sha256(raw).hexdigest()!=plan_sha: raise SystemExit("plan canonical/hash differs")
false_authority={name:False for name in runner.AUTHORITY_FIELDS}
expected_families=[{"confirmation_iid":runner.CONFIRMATION_IID_BY_DP_ARM[i],"dp_arm":i,"family":runner.FAMILY_BY_DP_ARM[i],"fit_iid":runner.FIT_IID_BY_DP_ARM[i],"same_family_wrong_iid":runner.FIT_IID_BY_DP_ARM[i]} for i in range(2)]
if (plan.get("schema_version")!="bernini-graft-phase-a-a-lite-short-world8-plan-v1" or plan.get("topology")!={"allocation":"single-node-8xMI210","dp_size":runner.DP_SIZE,"sp_size":runner.SP_SIZE,"world_size":runner.WORLD_SIZE} or plan.get("update_indices")!=list(runner.UPDATE_INDICES) or plan.get("confirmation_indices")!=list(runner.CONFIRMATION_INDICES) or plan.get("adapter_off_parity_indices")!=list(runner.ADAPTER_OFF_PARITY_INDICES) or plan.get("families")!=expected_families or plan.get("action_instructions")!={"dog":runner.ACTION_INSTRUCTION_BY_DP_ARM[0],"human":runner.ACTION_INSTRUCTION_BY_DP_ARM[1]} or plan.get("noop_instruction")!=consumer.NOOP_INSTRUCTION or plan.get("authority")!=false_authority or plan.get("no_checkpoint") is not True or plan.get("claim_scope")!="two-update-source-noop-diagnostic-no-checkpoint-no-scientific-claim"): raise SystemExit("plan does not match runner constants")
runtime=plan.get("runtime",{}); expected_runtime={"bernini_commit":"2d2b4591ac053ec25c6371b01a5a6746679e5793","consumer_sha256":runner.PINNED_CONSUMER_SOURCE_SHA256,"identity_rebinder_sha256":"1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0","native_v1_sha256":"36861e8670fc77d65b469f86aa472a10d453f9ba5fc227a62303329c4c38409a","native_v2_sha256":runner.PINNED_NATIVE_V2_SOURCE_SHA256,"runner_sha256":"4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4","short_trainer_sha256":runner.PINNED_SHORT_TRAINER_SOURCE_SHA256,"source_commit":"a884d357a6c0742f751be48d226ba72c952bae76","veomni_commit":"f90b3dc6fbb0ce693745223cc7a94064123dbf4d","world8_full_results_schema":"bernini-graft-phase-a-short-world8-full-results-v1","world8_result_schema":"bernini-graft-phase-a-short-world8-result-set-v2"}
if runtime!=expected_runtime: raise SystemExit("plan runtime pins differ")
release=plan.get("release",{}); expected_artifacts={
 "manifest":("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4.manifest.jsonl","0bfe9447fb3649625a4dface5c93ba0e55530285f9d63e7b15552b0bb4f9b1af"),
 "producer":("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4.receipt.json","01a87422753c5946c84df7c26198be992793de536159b1e427d4a00ab91ffea6"),
 "execution":("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4.execution.receipt.json","101a9cb25a4c898be0ed1b0406a3aec866a153d12ff79c6b890fb78c1146efd6"),
 "submission":("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4.submission.receipt.json","35ddc72443361ee588e5bd5c852df144293036a6128ae7fc9fab20e600e95b82")}
for label,(path,wanted) in expected_artifacts.items():
    if release.get("artifacts",{}).get(label)!={"path":path,"sha256":wanted}: raise SystemExit("plan release pin differs: "+label)
    p=Path(path)
    if p.resolve(strict=True)!=p: raise SystemExit("release path is not exact realpath")
    fd=os.open(p,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)); h=hashlib.sha256()
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o444 or before.st_nlink!=1: raise SystemExit("release artifact is not sealed")
        while True:
            block=os.read(fd,1024*1024)
            if not block: break
            h.update(block)
        after=os.fstat(fd)
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_mode,before.st_nlink)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_mode,after.st_nlink): raise SystemExit("release artifact changed")
    finally: os.close(fd)
    if h.hexdigest()!=wanted: raise SystemExit("release artifact bytes differ")
terminal=release.get("terminal_admission",{}); materializer=release.get("terminal_materializer",{})
stem="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/a_lite_source_release/10d9798_0c0aff6_c4_r1/outputs/canary4/graft_a_lite_source_canary4"
if terminal!={"path":terminal_path,"sha256_source":"external-pre-submission-pin","slurm_exit_code":"0:0","slurm_state":"COMPLETED"} or release.get("logical_output_stem")!=stem or terminal_path!=stem+".terminal.admission.receipt.json" or release.get("job_id")!="132549" or materializer!={"implementation_sha256":"4686463642a38e771c6858d1c10fc6aacb815a56e4f3eae951a336018d186cf4","runtime_sha256_source":"external-pre-submission-pin"} or re.fullmatch(r"[0-9a-f]{64}",terminal_sha) is None or re.fullmatch(r"[0-9a-f]{64}",runtime_sha) is None: raise SystemExit("external terminal admission binding differs")
runner.assert_pinned_dependencies()
PY

checkpoint_content_snapshot() {
  "${python_bin}" -I -B - "${method_root}" "${checkpoint}" "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" "${required_checkpoint_tree_sha256}" <<'PY'
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
output_root_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${output_root}")"
exec 8<"${output_root}" || fail "could not retain output-root descriptor"
/usr/bin/mkdir -- "${output_root}/rank-logs"
rank_log_root="${output_root}/rank-logs"
rank_log_root_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${rank_log_root}")"
exec 9<"${rank_log_root}" || fail "could not retain rank-log-root descriptor"
master_port=$((30000 + (SLURM_JOB_ID % 20000)))

echo "[graft-pa-short-v1] allocation=single-node-8xMI210 WORLD8=DP2xSP4 updates=29,38 confirmation=29,38 parity=0,25 no-checkpoint"
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
  --expected-runner-sha256 "${required_runner_sha256}" \
  --expected-identity-rebinder-sha256 "${required_rebinder_sha256}" \
  --expected-bernini-commit "${required_bernini_commit}" \
  --expected-veomni-commit "${required_veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${required_checkpoint_tree_sha256}" \
  --ack-two-update-diagnostic-no-checkpoint-no-scientific-claim \
  >"${output_root}/torchrun-supervisor.log" 2>&1 8<&- 9<&- &
torchrun_pid=$!
wait "${torchrun_pid}"
torchrun_status=$?
torchrun_pid=""
set -e
(( torchrun_status == 0 )) || fail "WORLD8 runner failed status=${torchrun_status}; elastic sibling teardown requested; logs retained"

audit_vendor_tree "${bernini_root}" "${required_bernini_commit}" Bernini-post
audit_vendor_tree "${veomni_root}" "${required_veomni_commit}" VeOmni-post
checkpoint_content_post_sha256="$(checkpoint_content_snapshot)"
[[ "${checkpoint_content_post_sha256}" == "${checkpoint_content_pre_sha256}" ]] || fail "checkpoint content identity changed across run"

"${python_bin}" -I -S -B - \
  "${output_root}" "${output_root_identity}" 8 \
  "${rank_log_root}" "${rank_log_root_identity}" 9 \
  "${plan_sha256}" "${source_archive_sha256}" "${closure_manifest_sha256}" \
  "${launcher_sha256}" "${terminal_admission_sha256}" "${terminal_materializer_runtime_sha256}" \
  "${checkpoint_content_pre_sha256}" "${checkpoint_content_post_sha256}" <<'PY'
# BEGIN GRAFT_PHASE_A_SHORT_WORLD8_PARENT_VALIDATOR_V1
import hashlib,json,os,re,stat,sys
from pathlib import Path
output=Path(sys.argv[1]); expected_identity=tuple(map(int,sys.argv[2].split(":"))); inherited_root_fd=int(sys.argv[3])
logs=Path(sys.argv[4]); expected_log_identity=tuple(map(int,sys.argv[5].split(":"))); inherited_log_fd=int(sys.argv[6])
plan_sha,archive_sha,closure_sha,launcher_sha,terminal_sha,materializer_runtime_sha,checkpoint_pre,checkpoint_post=sys.argv[7:]
SHA=re.compile(r"[0-9a-f]{64}\Z")
AUTHORITY=("action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","checkpoint_authority","publication_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed","semantic_action_editing_success_claimed")
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def sealed(v,label):
    if not isinstance(v,dict): raise ValueError(label+" is not a mapping")
    unsigned=dict(v); claimed=unsigned.pop("digest",None)
    if not isinstance(claimed,str) or SHA.fullmatch(claimed) is None or hashlib.sha256(canonical(unsigned)).hexdigest()!=claimed: raise ValueError(label+" digest differs")
    return claimed
def false_authority(v,label):
    if any(v.get(name) is not False for name in AUTHORITY): raise ValueError(label+" elevated authority")

FAMILIES=("dog","human")
FIT_IIDS=("7b88a1ca1f804f41","a35b590961d24694")
CONFIRMATION_IIDS=("841b5e0080a1441d","a66e6818e4144928")
CONFIRMATION_INDICES=(29,38)
PARITY_INDICES=(0,25)
PARITY_ROLES=("negative","noop_positive","action_positive")
FIELD_ROLES=("source_noop_target_velocity","correct_atlas_noop_velocity","wrong_atlas_noop_velocity","dropped_atlas_noop_velocity","correct_atlas_action_velocity","dropped_atlas_action_velocity")
GATE_NAMES=("correct_vs_wrong_noop_relative_gain","correct_vs_drop_noop_relative_gain","action_delta_correct_drop_norm_ratio","action_delta_correct_drop_cosine")
PROVENANCE_TRUE=("exactly_six_fields","all_fields_detached_fp32_finite_contiguous","all_field_storages_pairwise_disjoint","same_confirmation_source_zs_bytes_all_fields","same_native_full_source_v_pack_bytes_all_model_fields","same_noisy_target_object_and_bytes_all_fields","same_epsilon_bytes_all_fields","same_sigma_timestep_coordinate_all_fields","same_negative_condition_bytes_all_model_fields","correct_atlas_from_confirmation_row","wrong_atlas_from_same_family_fit_row","wrong_intervention_changes_only_identity_atlas_memory","drop_intervention_disables_only_identity_rebinder_residual_and_atlas_route","drop_retains_native_full_source_v_pack","noop_positive_condition_shared_across_correct_wrong_drop","action_positive_condition_shared_across_correct_drop","action_noop_pair_differs_only_in_positive_text_embedding","source_noop_target_velocity_recomputed_from_same_x_sigma_and_source_zs","native_v2v_apg_field_formula_used","same_state_tensor_identities_recomputed_byte_equal","wrong_route_receipts_differ_only_in_atlas_memory","drop_route_receipts_retain_v_branch_disable_only_rebinder","action_noop_route_receipts_equal_with_negative_raw_reuse")
PROVENANCE_FALSE=("confirmation_row_consumed_by_optimizer","wrong_atlas_is_cross_family","native_source_v_pack_dropped","negative_condition_changed_by_intervention","noise_or_coordinate_changed_by_intervention","target_video_used","generated_proposal_used","t2v_branch_used","source_retelling_used","selector_used","mask_pose_track_flow_or_motion_donor_used")
SAME_STATE_FIELDS=("confirmation_source_zs","epsilon","noisy_target_x_sigma","native_visual_pack","native_rotary_pack","sigma","timestep","negative_condition","noop_positive_condition","action_positive_condition")
ROUTE_ORDER=(("measurement","negative",False),("measurement","positive",False),("replay","negative",True),("replay","positive",True))
COORDINATES={29:(655,"3f27d446","0x1.4fa88c0000000p-1","0x1.7f027f7ae326bp-4"),38:(211,"3e58b351","0x1.b166a20000000p-3","0x1.0000000000000p+0")}
DEPENDENCY_PINS={"a_lite_consumer":"13ecb082ab3cff6f809b056c35715123be302a5c8d82a6760a7367861920ee75","native_v2_closure":"bf6a1d438183de5aa0460e729a39382e4597b3e43a4b9f1b3cdff5457439f20f","short_trainer":"73e39048bb8836fef33516eb1aae4cbc3f9fa4ecefcfb5d2695925bcb150f7bb","native_v1_gpu_runner_reuse":"e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a"}

def require(condition,message):
    if not condition: raise ValueError(message)
def require_sha(value,label):
    require(isinstance(value,str) and SHA.fullmatch(value) is not None,label+" is not SHA256")
    return value
def exact_int(value,wanted): return type(value) is int and value==wanted
def exact_float(value,wanted): return type(value) is float and value==wanted
def object_digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def recursive_denial(value,label):
    if isinstance(value,dict):
        for key,item in value.items():
            if isinstance(item,bool) and (key in AUTHORITY or key.endswith("_authorized") or "authority" in key) and item: raise ValueError(label+" elevated nested authority")
            if key in {"checkpoint_written","checkpoint_payload_returned","publication_performed"} and item is not False: raise ValueError(label+" published nested state")
            recursive_denial(item,label)
    elif isinstance(value,list):
        for item in value: recursive_denial(item,label)
def tensor_identity(value,label,dtype=None,device=None):
    dtype_bytes={"torch.bool":1,"torch.uint8":1,"torch.int8":1,"torch.float16":2,"torch.bfloat16":2,"torch.int16":2,"torch.float32":4,"torch.int32":4,"torch.float64":8,"torch.int64":8}
    require(isinstance(value,dict) and set(value)=={"shape","dtype","device_type_at_observation","finite","byte_count","raw_sha256","content_sha256"} and value.get("finite") is True and isinstance(value.get("shape"),list) and all(type(item) is int and item>0 for item in value["shape"]) and value.get("dtype") in dtype_bytes and value.get("device_type_at_observation") in {"cpu","cuda"} and type(value.get("byte_count")) is int,label+" tensor identity differs")
    elements=1
    for dimension in value["shape"]: elements*=dimension
    require(value["byte_count"]==elements*dtype_bytes[value["dtype"]],label+" byte count differs")
    if dtype is not None: require(value.get("dtype")==dtype,label+" dtype differs")
    if device is not None: require(value.get("device_type_at_observation")==device,label+" device differs")
    for name in ("raw_sha256","content_sha256"): require_sha(value.get(name),label+" "+name)
def finite_hex(value,label):
    try: observed=float.fromhex(value)
    except (TypeError,ValueError): raise ValueError(label+" is not float64 hex")
    require(observed==observed and observed not in (float("inf"),float("-inf")) and observed.hex()==value,label+" is noncanonical or nonfinite")
    return observed

def validate_parity(value,rank):
    sealed(value,f"rank{rank} parity")
    rows=value.get("rows")
    parity_keys={"schema_version","schedule_indices","branch_roles","baseline_captured_before_adapter_install","comparison_executed_after_two_updates_and_confirmation","all_installed_zero_gate_raw_bytes_equal_adapter_off","raw_dtype","rows","scheduler_unchanged","checkpoint_written","digest"}|set(AUTHORITY)
    row_keys={"schedule_index","branch_role","adapter_route_gate_float64_hex","adapter_off_raw_sha256","installed_zero_gate_raw_sha256","raw_storage_byte_exact","native_full_source_v_pack_bytes_unchanged","noisy_target_bytes_unchanged","epsilon_bytes_unchanged","sigma_timestep_unchanged","condition_bytes_unchanged","target_video_used"}
    require(set(value)==parity_keys and value.get("schema_version")=="bernini-graft-phase-a-short-adapter-off-bf16-parity-v1" and isinstance(value.get("schedule_indices"),list) and len(value["schedule_indices"])==2 and all(exact_int(item,PARITY_INDICES[position]) for position,item in enumerate(value["schedule_indices"])) and value.get("branch_roles")==list(PARITY_ROLES) and isinstance(rows,list) and len(rows)==6 and value.get("baseline_captured_before_adapter_install") is True and value.get("comparison_executed_after_two_updates_and_confirmation") is True and value.get("all_installed_zero_gate_raw_bytes_equal_adapter_off") is True and value.get("raw_dtype")=="torch.bfloat16" and value.get("scheduler_unchanged") is True and value.get("checkpoint_written") is False,f"rank{rank} adapter parity contract differs")
    expected=[(index,role) for index in PARITY_INDICES for role in PARITY_ROLES]
    require([(row.get("schedule_index"),row.get("branch_role")) for row in rows]==expected,f"rank{rank} adapter parity row order differs")
    for row in rows:
        require(set(row)==row_keys and type(row.get("schedule_index")) is int and row.get("adapter_route_gate_float64_hex")==0.0.hex() and row.get("adapter_off_raw_sha256")==row.get("installed_zero_gate_raw_sha256") and row.get("raw_storage_byte_exact") is True and row.get("native_full_source_v_pack_bytes_unchanged") is True and row.get("noisy_target_bytes_unchanged") is True and row.get("epsilon_bytes_unchanged") is True and row.get("sigma_timestep_unchanged") is True and row.get("condition_bytes_unchanged") is True and row.get("target_video_used") is False,f"rank{rank} adapter parity row differs")
        require_sha(row.get("adapter_off_raw_sha256"),f"rank{rank} adapter parity raw")
    false_authority(value,f"rank{rank} parity"); recursive_denial(value,f"rank{rank} parity")
    return value["digest"]

def validate_metric(metric,index,rank):
    sealed(metric,f"rank{rank} index{index} metric")
    metric_keys={"schema_version","schedule_index","field_roles","field_shape","field_dtype","field_device_type","field_tensor_sha256","noop_fm_loss_float64_hex","relative_gain_formula","relative_gain_float64_hex","minimum_relative_gain_float64_hex","action_delta_formula","action_delta_norm_float64_hex","action_delta_correct_drop_norm_ratio_formula","action_delta_correct_drop_norm_ratio_float64_hex","minimum_action_delta_correct_drop_norm_ratio_float64_hex","action_delta_correct_drop_cosine_float64_hex","minimum_action_delta_cosine_float64_hex","float64_tiny_hex","noncompensating_gates","noncompensating_all_pass","metrics_computed_from_six_detached_fields_by_this_core","field_origin_same_noise_state_coordinate_verified_by_this_core","digest"}|set(AUTHORITY)
    gates=metric.get("noncompensating_gates")
    require(set(metric)==metric_keys and metric.get("schema_version")=="bernini-graft-phase-a-confirmation-metrics-v1" and exact_int(metric.get("schedule_index"),index) and metric.get("field_roles")==list(FIELD_ROLES) and metric.get("field_dtype")=="torch.float32" and metric.get("field_device_type")=="cuda" and isinstance(metric.get("field_shape"),list) and metric["field_shape"] and all(type(item) is int and item>0 for item in metric["field_shape"]) and isinstance(gates,dict) and set(gates)==set(GATE_NAMES) and all(value is True for value in gates.values()) and metric.get("noncompensating_all_pass") is True and metric.get("metrics_computed_from_six_detached_fields_by_this_core") is True and metric.get("field_origin_same_noise_state_coordinate_verified_by_this_core") is False,f"rank{rank} index{index} metric gate differs")
    hashes=metric.get("field_tensor_sha256")
    require(isinstance(hashes,dict) and set(hashes)==set(FIELD_ROLES),f"rank{rank} index{index} field hashes differ")
    for role in FIELD_ROLES: require_sha(hashes[role],f"rank{rank} index{index} {role}")
    losses=metric.get("noop_fm_loss_float64_hex"); gains=metric.get("relative_gain_float64_hex"); norms=metric.get("action_delta_norm_float64_hex")
    require(isinstance(losses,dict) and set(losses)=={"correct_atlas","wrong_atlas","dropped_atlas"} and isinstance(gains,dict) and set(gains)=={"correct_vs_wrong","correct_vs_drop"} and isinstance(norms,dict) and set(norms)=={"correct_atlas","dropped_atlas"},f"rank{rank} index{index} metric components differ")
    correct_loss=finite_hex(losses["correct_atlas"],"correct loss"); wrong_loss=finite_hex(losses["wrong_atlas"],"wrong loss"); drop_loss=finite_hex(losses["dropped_atlas"],"drop loss"); tiny=finite_hex(metric.get("float64_tiny_hex"),"float64 tiny")
    wrong_gain=finite_hex(gains["correct_vs_wrong"],"wrong gain"); drop_gain=finite_hex(gains["correct_vs_drop"],"drop gain"); correct_norm=finite_hex(norms["correct_atlas"],"correct action norm"); drop_norm=finite_hex(norms["dropped_atlas"],"drop action norm"); ratio=finite_hex(metric.get("action_delta_correct_drop_norm_ratio_float64_hex"),"norm ratio"); cosine=finite_hex(metric.get("action_delta_correct_drop_cosine_float64_hex"),"cosine")
    require(correct_loss>=0.0 and wrong_loss>=0.0 and drop_loss>=0.0 and tiny==2.2250738585072014e-308 and wrong_gain==(wrong_loss-correct_loss)/max(wrong_loss,tiny) and drop_gain==(drop_loss-correct_loss)/max(drop_loss,tiny) and correct_norm>0.0 and drop_norm>0.0 and ratio==correct_norm/max(drop_norm,tiny) and -1.0<=cosine<=1.0 and metric.get("relative_gain_formula")=="(L_control-L_correct)/max(L_control,float64_tiny)" and metric.get("action_delta_formula")=="v_action-v_noop" and metric.get("action_delta_correct_drop_norm_ratio_formula")=="norm(delta_correct)/max(norm(delta_drop),float64_tiny)" and metric.get("minimum_relative_gain_float64_hex")==float(1.0e-4).hex() and metric.get("minimum_action_delta_correct_drop_norm_ratio_float64_hex")==float(0.5).hex() and metric.get("minimum_action_delta_cosine_float64_hex")==float(0.0).hex() and gates=={"correct_vs_wrong_noop_relative_gain":wrong_gain>=1.0e-4,"correct_vs_drop_noop_relative_gain":drop_gain>=1.0e-4,"action_delta_correct_drop_norm_ratio":ratio>=0.5,"action_delta_correct_drop_cosine":cosine>=0.0} and metric.get("noncompensating_all_pass") is all(gates.values()),f"rank{rank} index{index} numeric/formula gate differs")
    false_authority(metric,f"rank{rank} metric"); recursive_denial(metric,f"rank{rank} metric")
    return metric

def validate_provenance(provenance,index,arm,rank):
    sealed(provenance,f"rank{rank} index{index} provenance")
    provenance_keys={"schema_version","schedule_index","confirmation_iid","confirmation_source_sha256","wrong_owner_iid","wrong_owner_source_sha256","field_roles","field_tensor_identities","confirmation_source_state_receipt_digest","wrong_fit_source_state_receipt_digest","coordinate","epsilon","epsilon_receipt_digest","noisy_target","noisy_target_receipt_digest","confirmation_source_zs","native_visual_pack","native_rotary_pack","negative_condition","noop_positive_condition","action_positive_condition","correct_atlas","wrong_atlas","same_state_identities_before_model_fields","same_state_identities_after_all_fields","atlas_identities_before_model_fields","atlas_identities_after_all_fields","native_raw_call_order","negative_raw_reused_for_correct_noop_and_action","negative_raw_reused_for_drop_noop_and_action","route_receipts","raw_tensor_identities","ambient_torch_no_grad","digest"}|set(PROVENANCE_TRUE)|set(PROVENANCE_FALSE)|set(AUTHORITY)
    require(set(provenance)==provenance_keys and provenance.get("schema_version")=="bernini-graft-phase-a-short-six-field-provenance-v1" and exact_int(provenance.get("schedule_index"),index) and provenance.get("confirmation_iid")==CONFIRMATION_IIDS[arm] and provenance.get("wrong_owner_iid")==FIT_IIDS[arm] and provenance.get("field_roles")==list(FIELD_ROLES) and all(provenance.get(key) is True for key in PROVENANCE_TRUE) and all(provenance.get(key) is False for key in PROVENANCE_FALSE),f"rank{rank} index{index} provenance contract differs")
    for name in ("confirmation_source_sha256","wrong_owner_source_sha256","confirmation_source_state_receipt_digest","wrong_fit_source_state_receipt_digest","epsilon_receipt_digest","noisy_target_receipt_digest"): require_sha(provenance.get(name),f"rank{rank} index{index} provenance {name}")
    identities=provenance.get("field_tensor_identities")
    require(isinstance(identities,dict) and set(identities)==set(FIELD_ROLES),f"rank{rank} index{index} provenance fields differ")
    for role in FIELD_ROLES: tensor_identity(identities[role],f"rank{rank} {role}","torch.float32","cuda")
    before=provenance.get("same_state_identities_before_model_fields"); after=provenance.get("same_state_identities_after_all_fields")
    require(isinstance(before,dict) and set(before)==set(SAME_STATE_FIELDS) and before==after,f"rank{rank} index{index} same-state identities differ")
    for name in SAME_STATE_FIELDS: tensor_identity(before[name],f"rank{rank} index{index} same-state {name}")
    require(before["sigma"]["shape"]==[] and before["sigma"]["dtype"]=="torch.float32" and before["sigma"]["device_type_at_observation"]=="cpu" and before["timestep"]["shape"]==[1] and before["timestep"]["dtype"]=="torch.int64" and before["timestep"]["device_type_at_observation"]=="cuda",f"rank{rank} index{index} scalar coordinate tensor identities differ")
    aliases={"confirmation_source_zs":"confirmation_source_zs","epsilon":"epsilon","noisy_target_x_sigma":"noisy_target","native_visual_pack":"native_visual_pack","native_rotary_pack":"native_rotary_pack","negative_condition":"negative_condition","noop_positive_condition":"noop_positive_condition","action_positive_condition":"action_positive_condition"}
    require(all(provenance.get(target)==before[source] for source,target in aliases.items()),f"rank{rank} index{index} same-state aliases differ")
    coordinate=provenance.get("coordinate"); sealed(coordinate,f"rank{rank} index{index} coordinate")
    expected_timestep,expected_sigma_be,expected_sigma_hex,expected_gate_hex=COORDINATES[index]
    require(set(coordinate)=={"schema_version","schedule_index","timestep","sigma_float32_be_hex","schedule_sha256","scheduler_step_called","digest"} and coordinate.get("schema_version")=="bernini-graft-phase-a-short-coordinate-v1" and exact_int(coordinate.get("schedule_index"),index) and exact_int(coordinate.get("timestep"),expected_timestep) and coordinate.get("sigma_float32_be_hex")==expected_sigma_be and coordinate.get("schedule_sha256")=="3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03" and coordinate.get("scheduler_step_called") is False,f"rank{rank} index{index} coordinate differs")
    atlas_before=provenance.get("atlas_identities_before_model_fields"); atlas_after=provenance.get("atlas_identities_after_all_fields")
    require(isinstance(atlas_before,dict) and set(atlas_before)=={"correct_confirmation_atlas","wrong_same_family_fit_atlas"} and atlas_before==atlas_after and provenance.get("correct_atlas")==atlas_before["correct_confirmation_atlas"] and provenance.get("wrong_atlas")==atlas_before["wrong_same_family_fit_atlas"],f"rank{rank} index{index} atlas identities differ")
    for name,value in atlas_before.items(): tensor_identity(value,f"rank{rank} index{index} {name}","torch.float32","cuda")
    raw=provenance.get("raw_tensor_identities"); raw_names=("correct_negative","correct_noop","correct_action","wrong_negative","wrong_noop","drop_negative","drop_noop","drop_action")
    require(isinstance(raw,dict) and set(raw)==set(raw_names) and provenance.get("native_raw_call_order")==["correct_negative","correct_noop_positive","correct_action_positive","wrong_negative","wrong_noop_positive","drop_negative","drop_noop_positive","drop_action_positive"] and provenance.get("negative_raw_reused_for_correct_noop_and_action") is True and provenance.get("negative_raw_reused_for_drop_noop_and_action") is True and provenance.get("ambient_torch_no_grad") is True,f"rank{rank} index{index} raw forward evidence differs")
    for name in raw_names: tensor_identity(raw[name],f"rank{rank} index{index} raw {name}","torch.bfloat16","cuda")
    routes=provenance.get("route_receipts"); route_names=("correct_negative","correct_noop","correct_action","wrong_negative","wrong_noop","drop_negative","drop_noop","drop_action")
    require(isinstance(routes,dict) and set(routes)==set(route_names),f"rank{rank} index{index} route provenance differs")
    route_keys={"branch_name","total_tokens","condition_tokens","target_tokens","sequence_parallel_rank","sequence_parallel_size","sigma_hex","gate_hex","atlas_receipt_digest","source_memory_owned_by_V_VI_only","enabled","digest"}
    for name in route_names:
        sealed(routes[name],f"rank{rank} index{index} {name} route"); require(set(routes[name])==route_keys,f"rank{rank} index{index} {name} route keys differ")
    require(routes["correct_negative"]==routes["correct_noop"]==routes["correct_action"] and routes["wrong_negative"]==routes["wrong_noop"] and routes["drop_negative"]==routes["drop_noop"]==routes["drop_action"],f"rank{rank} index{index} route reuse differs")
    def without(value,*keys):
        plain=dict(value)
        for key in ("digest",*keys): plain.pop(key,None)
        return plain
    correct,routed_wrong,drop=routes["correct_negative"],routes["wrong_negative"],routes["drop_negative"]
    require_sha(correct.get("atlas_receipt_digest"),f"rank{rank} correct atlas receipt"); require_sha(routed_wrong.get("atlas_receipt_digest"),f"rank{rank} wrong atlas receipt")
    require(correct.get("branch_name")=="V" and correct.get("enabled") is True and routed_wrong.get("enabled") is True and type(correct.get("total_tokens")) is int and correct["total_tokens"]>0 and type(correct.get("condition_tokens")) is int and 0<=correct["condition_tokens"]<correct["total_tokens"] and type(correct.get("target_tokens")) is int and correct.get("target_tokens")==correct["total_tokens"]-correct["condition_tokens"] and correct["target_tokens"]>0 and exact_int(correct.get("sequence_parallel_rank"),rank%4) and exact_int(correct.get("sequence_parallel_size"),4) and correct.get("source_memory_owned_by_V_VI_only") is True and correct.get("sigma_hex")==expected_sigma_hex and correct.get("gate_hex")==expected_gate_hex and routed_wrong.get("sigma_hex")==expected_sigma_hex and routed_wrong.get("gate_hex")==expected_gate_hex and without(correct,"atlas_receipt_digest")==without(routed_wrong,"atlas_receipt_digest") and correct.get("atlas_receipt_digest")!=routed_wrong.get("atlas_receipt_digest") and without(correct,"atlas_receipt_digest","enabled","gate_hex")==without(drop,"atlas_receipt_digest","enabled","gate_hex") and drop.get("branch_name")=="V" and drop.get("enabled") is False and drop.get("atlas_receipt_digest") is None and drop.get("sigma_hex")==expected_sigma_hex and drop.get("gate_hex")==0.0.hex(),f"rank{rank} index{index} route intervention differs")
    false_authority(provenance,f"rank{rank} provenance"); recursive_denial(provenance,f"rank{rank} provenance")
    return provenance

def validate_admission(admission,metric,index,arm,rank):
    sealed(admission,f"rank{rank} index{index} admission")
    admission_keys={"schema_version","row_iid","wrong_owner_iid","schedule_index","metrics_digest","parameter_digest","base_digest","optimizer_digest","metrics","sp4_consensus_digest","no_grad","optimizer_update_performed","checkpoint_written","digest"}|set(AUTHORITY)
    for name in ("parameter_digest","base_digest","optimizer_digest"): require_sha(admission.get(name),f"rank{rank} admission {name}")
    record={"row_iid":CONFIRMATION_IIDS[arm],"wrong_owner_iid":FIT_IIDS[arm],"schedule_index":index,"metrics_digest":metric["digest"],"parameter_digest":admission["parameter_digest"],"base_digest":admission["base_digest"],"optimizer_digest":admission["optimizer_digest"]}
    require(set(admission)==admission_keys and admission.get("schema_version")=="bernini-graft-phase-a-confirmation-field-admission-v1" and exact_int(admission.get("schedule_index"),index) and all(admission.get(key)==value for key,value in record.items()) and admission.get("metrics")==metric and admission.get("sp4_consensus_digest")==object_digest(record) and admission.get("no_grad") is True and admission.get("optimizer_update_performed") is False and admission.get("checkpoint_written") is False,f"rank{rank} index{index} admission differs")
    false_authority(admission,f"rank{rank} admission"); recursive_denial(admission,f"rank{rank} admission")
    return admission["sp4_consensus_digest"]

def validate_update_route(route,ordinal,arm,rank):
    sealed(route,f"rank{rank} update route {ordinal}")
    rows=route.get("rows")
    route_keys={"schema_version","update_number","schedule_index","row_iid","row_source_sha256","fit_row_only","exact_four_native_forwards","forward_order","fresh_atlas_per_forward","measurement_atlas_detached","replay_atlas_graph_bearing_only_on_target_owner","rows","checkpoint_written","digest"}|set(AUTHORITY)
    row_keys={"ordinal","phase","role","graph_expected","schedule_index","route_gate_float64_hex","local_target_rows","adapter_graph_bearing","atlas_tokens","atlas_receipt_digest","fresh_atlas_object"}
    require(set(route)==route_keys and route.get("schema_version")=="bernini-graft-phase-a-short-update-route-v1" and exact_int(route.get("update_number"),ordinal) and exact_int(route.get("schedule_index"),CONFIRMATION_INDICES[ordinal-1]) and route.get("row_iid")==FIT_IIDS[arm] and route.get("fit_row_only") is True and route.get("exact_four_native_forwards") is True and route.get("forward_order")==[[phase,role] for phase,role,_ in ROUTE_ORDER] and route.get("fresh_atlas_per_forward") is True and route.get("measurement_atlas_detached") is True and route.get("replay_atlas_graph_bearing_only_on_target_owner") is True and route.get("checkpoint_written") is False and isinstance(rows,list) and len(rows)==4,f"rank{rank} update route {ordinal} differs")
    require_sha(route.get("row_source_sha256"),f"rank{rank} update route source")
    for index,(row,(phase,role,graph)) in enumerate(zip(rows,ROUTE_ORDER)):
        gate=finite_hex(row.get("route_gate_float64_hex"),"update route gate")
        local_rows=row.get("local_target_rows")
        require(set(row)==row_keys and exact_int(row.get("ordinal"),index) and row.get("phase")==phase and row.get("role")==role and row.get("graph_expected") is graph and exact_int(row.get("schedule_index"),CONFIRMATION_INDICES[ordinal-1]) and row.get("fresh_atlas_object") is True and row.get("route_gate_float64_hex")==COORDINATES[CONFIRMATION_INDICES[ordinal-1]][3] and gate>0.0 and type(local_rows) is int and local_rows>=0 and row.get("adapter_graph_bearing") is (graph and local_rows>0),f"rank{rank} update route row differs")
        tensor_identity(row.get("atlas_tokens"),f"rank{rank} update route atlas","torch.float32","cuda")
        require_sha(row.get("atlas_receipt_digest"),f"rank{rank} update route atlas receipt")
    require(len({(row["atlas_tokens"]["raw_sha256"],row["atlas_tokens"]["content_sha256"],row["atlas_receipt_digest"]) for row in rows})==1,f"rank{rank} update route atlas bytes differ")
    false_authority(route,f"rank{rank} update route"); recursive_denial(route,f"rank{rank} update route")
    return route["digest"]

def validate_trainer_update(update,ordinal,arm,sp_rank,rank,routing_digest):
    sealed(update,f"rank{rank} trainer update {ordinal}")
    update_keys={"schema_version","update_number","schedule_index","training_regime","dp_arm","sp_rank","row_iid","row_source_sha256","plan_digest","native_admission_digest","gradient_synchronization_digest","gradient_sync","preclip_norm_float64_hex","postclip_norm_float64_hex","max_grad_norm","optimizer","parameter_digest_before","parameter_digest_after","frozen_base_digest","parameter_world_consensus","frozen_base_world_consensus","gradients_reset_to_none_after_step","checkpoint_written","digest"}|set(AUTHORITY)
    require(set(update)==update_keys and update.get("schema_version")=="bernini-graft-phase-a-short-update-v1" and exact_int(update.get("update_number"),ordinal) and exact_int(update.get("schedule_index"),CONFIRMATION_INDICES[ordinal-1]) and update.get("training_regime")==(("bootstrap","post_bootstrap")[ordinal-1]) and exact_int(update.get("dp_arm"),arm) and exact_int(update.get("sp_rank"),sp_rank) and update.get("row_iid")==FIT_IIDS[arm] and update.get("parameter_world_consensus") is True and update.get("frozen_base_world_consensus") is True and update.get("gradients_reset_to_none_after_step") is True and update.get("checkpoint_written") is False,f"rank{rank} trainer update {ordinal} differs")
    for name in ("row_source_sha256","plan_digest","parameter_digest_before","parameter_digest_after","frozen_base_digest","native_admission_digest","gradient_synchronization_digest"): require_sha(update.get(name),f"rank{rank} update {name}")
    cell_plan={"schema_version":"bernini-graft-phase-a-short-cell-plan-v1","update_number":ordinal,"schedule_index":CONFIRMATION_INDICES[ordinal-1],"expected_regime":(("bootstrap","post_bootstrap")[ordinal-1]),"dp_arm":arm,"row_iid":FIT_IIDS[arm],"row_source_sha256":update["row_source_sha256"],"routing_digest":routing_digest,"source_owned_bytes_only":True,"source_path_input_accepted":False}
    require(update["plan_digest"]==object_digest(cell_plan),f"rank{rank} update plan digest differs")
    require(update["parameter_digest_before"]!=update["parameter_digest_after"],f"rank{rank} optimizer did not update")
    preclip=finite_hex(update.get("preclip_norm_float64_hex"),"preclip norm"); postclip=finite_hex(update.get("postclip_norm_float64_hex"),"postclip norm")
    update_optimizer=update.get("optimizer"); update_betas=update_optimizer.get("betas") if isinstance(update_optimizer,dict) else None
    optimizer_ok=isinstance(update_optimizer,dict) and set(update_optimizer)=={"kind","learning_rate","weight_decay","betas","eps","foreach"} and update_optimizer.get("kind")=="torch.optim.AdamW" and exact_float(update_optimizer.get("learning_rate"),1.0e-3) and exact_float(update_optimizer.get("weight_decay"),0.0) and isinstance(update_betas,list) and len(update_betas)==2 and exact_float(update_betas[0],0.9) and exact_float(update_betas[1],0.999) and exact_float(update_optimizer.get("eps"),1.0e-8) and update_optimizer.get("foreach") is False
    require(preclip>0.0 and postclip>0.0 and postclip<=1.0+1.0e-5 and exact_float(update.get("max_grad_norm"),1.0) and optimizer_ok,f"rank{rank} update optimizer evidence differs")
    sync=update.get("gradient_sync"); sealed(sync,f"rank{rank} update sync"); require(sync["digest"]==update["gradient_synchronization_digest"],f"rank{rank} update sync digest differs")
    category_names=("atlas_encoder","query_projection","key_projection","value_projection","output_projection"); category=sync.get("category_l2_float64_hex"); materialized=sync.get("none_materialized_parameter_names")
    require(set(sync)=={"schema_version","update_number","training_regime","none_materialized_as_true_zero_count","none_materialized_parameter_names","collective_order","category_l2_float64_hex","preclip_l2_float64_hex","gate","finite","digest"} and sync.get("schema_version")=="bernini-graft-dp2-sp4-gradient-sync-v1" and exact_int(sync.get("update_number"),ordinal) and sync.get("training_regime")==(("bootstrap","post_bootstrap")[ordinal-1]) and type(sync.get("none_materialized_as_true_zero_count")) is int and sync["none_materialized_as_true_zero_count"]>=0 and isinstance(materialized,list) and len(materialized)==sync["none_materialized_as_true_zero_count"] and all(isinstance(name,str) and name for name in materialized) and sync.get("collective_order")==["SP4_SUM","divide_by_4","DP2_SUM","divide_by_2"] and isinstance(category,dict) and set(category)==set(category_names) and sync.get("finite") is True,f"rank{rank} update sync contract differs")
    norms={name:finite_hex(category[name],"gradient category") for name in category_names}; sync_total=finite_hex(sync.get("preclip_l2_float64_hex"),"gradient total")
    require(sync_total>0.0 and abs(sync_total*sync_total-sum(value*value for value in norms.values()))<=max(1.0e-12,sync_total*sync_total*1.0e-9),f"rank{rank} update gradient norm differs")
    if ordinal==1: require(sync.get("gate")=="world8_bootstrap_output_projection_only_nonzero" and norms["output_projection"]>0.0 and all(norms[name]==0.0 for name in category_names[:-1]),f"rank{rank} bootstrap gradient gate differs")
    else: require(sync.get("gate")=="world8_post_bootstrap_all_five_categories_nonzero" and all(value>0.0 for value in norms.values()),f"rank{rank} post-bootstrap gradient gate differs")
    false_authority(update,f"rank{rank} update"); recursive_denial(update,f"rank{rank} update")
    return (update["parameter_digest_before"],update["parameter_digest_after"],update["frozen_base_digest"])

def validate_local(local,rank,summary):
    digest=sealed(local,f"rank{rank} local result"); arm=rank//4; sp_rank=rank%4
    topology=local.get("topology",{}); source=local.get("source_routing",{}); confirmation=local.get("confirmation",{}); trainer=local.get("short_trainer_receipt",{}); parity=local.get("adapter_off_parity",{})
    local_keys={"schema_version","status","complete","topology","dependency_source_sha256","source_routing","execution_trace","update_route_receipts","confirmation","adapter_off_parity","short_trainer_receipt","training_updates_executed_for_diagnostic","full_sampler_used","decoded_media_output_created","checkpoint_written","checkpoint_payload_returned","publication_performed","result_staged_in_memory_only","target_video_used","generated_proposal_used","t2v_branch_used","source_retelling_used","selector_used","mask_pose_track_flow_or_motion_donor_used","digest"}|set(AUTHORITY)
    source_keys={"routing_digest","fit_iid","confirmation_iid","fit_row_consumed_by_optimizer","confirmation_row_consumed_by_optimizer","wrong_atlas_iid","wrong_atlas_is_same_family_fit_row","owned_source_bytes_only","source_path_reopened_by_runner"}
    topology_keys={"world_size","data_parallel_size","sequence_parallel_size","rank","dp_arm","sp_rank","family"}
    topology_ok=isinstance(topology,dict) and set(topology)==topology_keys and exact_int(topology.get("world_size"),8) and exact_int(topology.get("data_parallel_size"),2) and exact_int(topology.get("sequence_parallel_size"),4) and exact_int(topology.get("rank"),rank) and exact_int(topology.get("dp_arm"),arm) and exact_int(topology.get("sp_rank"),sp_rank) and topology.get("family")==FAMILIES[arm]
    require(set(local)==local_keys and set(source)==source_keys and local.get("schema_version")=="bernini-graft-phase-a-a-lite-short-gpu-runner-v1" and local.get("status")=="completed_in_memory_diagnostic_no_checkpoint" and local.get("complete") is True and topology_ok and local.get("dependency_source_sha256")==DEPENDENCY_PINS and source.get("fit_iid")==FIT_IIDS[arm] and source.get("confirmation_iid")==CONFIRMATION_IIDS[arm] and source.get("wrong_atlas_iid")==FIT_IIDS[arm] and source.get("fit_row_consumed_by_optimizer") is True and source.get("confirmation_row_consumed_by_optimizer") is False and source.get("wrong_atlas_is_same_family_fit_row") is True and source.get("owned_source_bytes_only") is True and source.get("source_path_reopened_by_runner") is False and exact_int(local.get("training_updates_executed_for_diagnostic"),2) and local.get("checkpoint_written") is False and local.get("checkpoint_payload_returned") is False and local.get("result_staged_in_memory_only") is True and local.get("publication_performed") is False and all(local.get(key) is False for key in ("full_sampler_used","decoded_media_output_created","target_video_used","generated_proposal_used","t2v_branch_used","source_retelling_used","selector_used","mask_pose_track_flow_or_motion_donor_used")),f"rank{rank} local result contract differs")
    require_sha(source.get("routing_digest"),f"rank{rank} source routing")
    require(summary.get("result_digest")==digest,f"rank{rank} summary/local digest differs")
    sealed(trainer,f"rank{rank} trainer"); trainer_topology=trainer.get("topology",{}); trainer_source=trainer.get("source_routing",{}); optimizer=trainer.get("optimizer_contract",{}); trainer_confirmation=trainer.get("confirmation",{})
    trainer_keys={"schema_version","status","topology","dependency_pins","source_routing","optimizer_contract","updates","confirmation","initial_parameter_digest","final_parameter_digest","initial_frozen_base_digest","final_frozen_base_digest","optimizer_state_digest_after_training_and_confirmation","parameter_world_consensus","frozen_base_world_consensus","test_only","real_model_loaded_by_this_core","source_bytes_to_latent_binding_verified_by_this_core","trainer_execution_runtime_live_verified","same_process_execution_integrity_formally_proven_by_this_core","same_process_formal_security_proven_by_this_core","full_sampler_used","decoded_media_used","checkpoint_written","checkpoint_payload_returned","publication_performed","digest"}|set(AUTHORITY)
    trainer_source_keys={"routing_digest","source_release_result_digest","pinset_digest","path_free_owned_bytes_only","update_rows","confirmation_rows","local_update_iid","local_confirmation_iid","confirmation_rows_consumed_by_optimizer","authority"}
    trainer_confirmation_keys={"plan","row_iid","wrong_owner_iid","wrong_owner_is_same_family_fit_row","schedule_indices","field_roles","thresholds","per_index_metrics","sp4_consensus_digest","all_indices_noncompensating_hard_gate_passed","evaluated_under_no_grad","optimizer_state_unchanged","parameters_unchanged","frozen_base_unchanged","six_field_tensor_contract_and_metrics_authenticated_by_this_core","same_noise_x_sigma_coordinate_authenticated_by_this_core","field_origin_runtime_authenticated_by_this_core","runner_adapter_off_parity_indices","runner_adapter_off_parity_verified_by_this_core","runner_must_block_checkpoint_without_adapter_off_parity"}
    dependency_pins={"consumer_commit":"6bae78c40b531310ad89a9b47418a8ff4a81ce05","consumer_source_sha256":DEPENDENCY_PINS["a_lite_consumer"],"native_v2_source_sha256":DEPENDENCY_PINS["native_v2_closure"],"native_v2_schema":"bernini-graft-phase-a-native-training-closure-v2","dependency_runtime_surface_sha256":"1bdf3747b598c577d73132b8bc278d8a3f911da35d19236d12d706b4d476cac1","trainer_execution_runtime_sha256":"f35f621938e7a6b8bd3b9b5a6b0fb782f5ebf483939f585f561e3908f993af3c"}
    trainer_topology_keys={"world_size","data_parallel_size","sequence_parallel_size","rank","dp_arm","sp_rank"}
    trainer_topology_ok=isinstance(trainer_topology,dict) and set(trainer_topology)==trainer_topology_keys and exact_int(trainer_topology.get("world_size"),8) and exact_int(trainer_topology.get("data_parallel_size"),2) and exact_int(trainer_topology.get("sequence_parallel_size"),4) and exact_int(trainer_topology.get("rank"),rank) and exact_int(trainer_topology.get("dp_arm"),arm) and exact_int(trainer_topology.get("sp_rank"),sp_rank)
    routing_authority_keys={"action_authority","identity_authority","cross_clip_identity_authority","quality_authority","training_authority","production_authority","data_governance_authority","data_license_authority","scientific_success_claimed"}
    trainer_parity_indices=trainer_confirmation.get("runner_adapter_off_parity_indices")
    trainer_betas=optimizer.get("betas") if isinstance(optimizer,dict) else None; trainer_schedules=optimizer.get("schedule_indices") if isinstance(optimizer,dict) else None
    trainer_optimizer_ok=isinstance(optimizer,dict) and set(optimizer)=={"kind","learning_rate","weight_decay","betas","eps","max_grad_norm","steps","schedule_indices","regimes","gradient_collective_order"} and optimizer.get("kind")=="torch.optim.AdamW" and exact_float(optimizer.get("learning_rate"),1.0e-3) and exact_float(optimizer.get("weight_decay"),0.0) and isinstance(trainer_betas,list) and len(trainer_betas)==2 and exact_float(trainer_betas[0],0.9) and exact_float(trainer_betas[1],0.999) and exact_float(optimizer.get("eps"),1.0e-8) and exact_float(optimizer.get("max_grad_norm"),1.0) and exact_int(optimizer.get("steps"),2) and isinstance(trainer_schedules,list) and len(trainer_schedules)==2 and all(exact_int(item,CONFIRMATION_INDICES[position]) for position,item in enumerate(trainer_schedules)) and optimizer.get("regimes")==["bootstrap","post_bootstrap"] and optimizer.get("gradient_collective_order")==["SP4_SUM_div_4","DP2_SUM_div_2"]
    thresholds=trainer_confirmation.get("thresholds"); thresholds_ok=isinstance(thresholds,dict) and set(thresholds)=={"minimum_noop_fm_relative_gain","minimum_action_delta_correct_drop_norm_ratio","minimum_action_delta_cosine"} and exact_float(thresholds.get("minimum_noop_fm_relative_gain"),1.0e-4) and exact_float(thresholds.get("minimum_action_delta_correct_drop_norm_ratio"),0.5) and exact_float(thresholds.get("minimum_action_delta_cosine"),0.0)
    require(set(trainer)==trainer_keys and set(trainer_source)==trainer_source_keys and set(trainer_confirmation)==trainer_confirmation_keys and trainer.get("dependency_pins")==dependency_pins and trainer.get("schema_version")=="bernini-graft-phase-a-a-lite-short-training-v1" and trainer.get("status")=="completed_in_memory_orchestration" and trainer_topology_ok and trainer_source.get("routing_digest")==source["routing_digest"] and trainer_source.get("path_free_owned_bytes_only") is True and exact_int(trainer_source.get("update_rows"),2) and exact_int(trainer_source.get("confirmation_rows"),2) and trainer_source.get("local_update_iid")==FIT_IIDS[arm] and trainer_source.get("local_confirmation_iid")==CONFIRMATION_IIDS[arm] and trainer_source.get("confirmation_rows_consumed_by_optimizer") is False and isinstance(trainer_source.get("authority"),dict) and set(trainer_source["authority"])==routing_authority_keys and all(value is False for value in trainer_source["authority"].values()) and trainer_optimizer_ok and trainer_confirmation.get("plan")=="per_row_per_index_noop_fm_relative_gain_plus_action_delta_geometry" and trainer_confirmation.get("row_iid")==CONFIRMATION_IIDS[arm] and trainer_confirmation.get("wrong_owner_iid")==FIT_IIDS[arm] and trainer_confirmation.get("wrong_owner_is_same_family_fit_row") is True and trainer_confirmation.get("schedule_indices")==list(CONFIRMATION_INDICES) and trainer_confirmation.get("field_roles")==list(FIELD_ROLES) and thresholds_ok and trainer_confirmation.get("all_indices_noncompensating_hard_gate_passed") is True and all(trainer_confirmation.get(key) is True for key in ("evaluated_under_no_grad","optimizer_state_unchanged","parameters_unchanged","frozen_base_unchanged","six_field_tensor_contract_and_metrics_authenticated_by_this_core","runner_must_block_checkpoint_without_adapter_off_parity")) and all(trainer_confirmation.get(key) is False for key in ("same_noise_x_sigma_coordinate_authenticated_by_this_core","field_origin_runtime_authenticated_by_this_core","runner_adapter_off_parity_verified_by_this_core")) and isinstance(trainer_parity_indices,list) and len(trainer_parity_indices)==2 and all(exact_int(item,PARITY_INDICES[position]) for position,item in enumerate(trainer_parity_indices)) and trainer.get("parameter_world_consensus") is True and trainer.get("frozen_base_world_consensus") is True and trainer.get("test_only") is False and trainer.get("real_model_loaded_by_this_core") is False and trainer.get("source_bytes_to_latent_binding_verified_by_this_core") is False and trainer.get("trainer_execution_runtime_live_verified") is True and trainer.get("same_process_execution_integrity_formally_proven_by_this_core") is False and trainer.get("same_process_formal_security_proven_by_this_core") is False and trainer.get("full_sampler_used") is False and trainer.get("decoded_media_used") is False and trainer.get("checkpoint_written") is False and trainer.get("checkpoint_payload_returned") is False and trainer.get("publication_performed") is False,f"rank{rank} trainer contract differs")
    for name in ("routing_digest","source_release_result_digest","pinset_digest"): require_sha(trainer_source.get(name),f"rank{rank} trainer source {name}")
    require(summary.get("trainer_result_digest")==trainer["digest"],f"rank{rank} summary/trainer digest differs")
    updates=trainer.get("updates"); require(isinstance(updates,list) and len(updates)==2,f"rank{rank} trainer update coverage differs")
    update_states=[validate_trainer_update(update,ordinal,arm,sp_rank,rank,trainer_source.get("routing_digest")) for ordinal,update in enumerate(updates,1)]
    for name in ("initial_parameter_digest","final_parameter_digest","initial_frozen_base_digest","final_frozen_base_digest","optimizer_state_digest_after_training_and_confirmation"): require_sha(trainer.get(name),f"rank{rank} trainer {name}")
    require(update_states[0][0]==trainer["initial_parameter_digest"] and update_states[0][1]==update_states[1][0] and update_states[1][1]==trainer["final_parameter_digest"] and update_states[0][2]==update_states[1][2]==trainer["initial_frozen_base_digest"]==trainer["final_frozen_base_digest"],f"rank{rank} trainer state chain differs")
    metrics=trainer_confirmation.get("per_index_metrics"); consensus=trainer_confirmation.get("sp4_consensus_digest"); provenances=confirmation.get("provenance"); admissions=confirmation.get("admissions")
    require(set(confirmation)=={"schedule_indices","field_roles","provenance","admissions","exact_six_fields_per_index","same_state_interventions_verified","wrong_atlas_same_family_fit_verified","drop_disables_only_identity_rebinder_memory_verified"} and confirmation.get("schedule_indices")==list(CONFIRMATION_INDICES) and confirmation.get("field_roles")==list(FIELD_ROLES) and all(confirmation.get(key) is True for key in ("exact_six_fields_per_index","same_state_interventions_verified","wrong_atlas_same_family_fit_verified","drop_disables_only_identity_rebinder_memory_verified")) and isinstance(metrics,dict) and tuple(metrics)==tuple(map(str,CONFIRMATION_INDICES)) and isinstance(consensus,dict) and tuple(consensus)==tuple(map(str,CONFIRMATION_INDICES)) and isinstance(provenances,list) and len(provenances)==2 and isinstance(admissions,list) and len(admissions)==2,f"rank{rank} confirmation coverage differs")
    consensus_values=[]; provenance_rows=[]
    for position,index in enumerate(CONFIRMATION_INDICES):
        metric=validate_metric(metrics[str(index)],index,rank); provenance=validate_provenance(provenances[position],index,arm,rank); admitted_consensus=validate_admission(admissions[position],metric,index,arm,rank)
        require(all(metric["field_tensor_sha256"][role]==provenance["field_tensor_identities"][role]["raw_sha256"] and provenance["field_tensor_identities"][role]["shape"]==metric["field_shape"] and provenance["field_tensor_identities"][role]["dtype"]==metric["field_dtype"] and provenance["field_tensor_identities"][role]["device_type_at_observation"]==metric["field_device_type"] for role in FIELD_ROLES) and consensus[str(index)]==admitted_consensus and admissions[position].get("parameter_digest")==trainer["final_parameter_digest"] and admissions[position].get("base_digest")==trainer["final_frozen_base_digest"] and admissions[position].get("optimizer_digest")==trainer["optimizer_state_digest_after_training_and_confirmation"],f"rank{rank} index{index} trainer/admission/provenance consensus differs"); consensus_values.append(admitted_consensus); provenance_rows.append(provenance)
    parity_digest=validate_parity(parity,rank); require(summary.get("adapter_off_parity_digest")==parity_digest,f"rank{rank} summary/parity digest differs")
    routes=local.get("update_route_receipts"); require(isinstance(routes,list) and len(routes)==2,f"rank{rank} update route coverage differs"); route_digests=[validate_update_route(route,ordinal,arm,rank) for ordinal,route in enumerate(routes,1)]
    fit_source=updates[0]["row_source_sha256"]; confirmation_source=provenance_rows[0]["confirmation_source_sha256"]
    require(updates[1]["row_source_sha256"]==fit_source and all(route["row_source_sha256"]==fit_source for route in routes) and all(provenance["wrong_owner_source_sha256"]==fit_source for provenance in provenance_rows) and provenance_rows[1]["confirmation_source_sha256"]==confirmation_source and confirmation_source!=fit_source,f"rank{rank} fit/confirmation source linkage differs")
    trace=local.get("execution_trace"); expected_ops=("open_short_training","next_update_plan","make_native_v2_cell","run_update","admit_update_route_evidence","next_update_plan","make_native_v2_cell","run_update","admit_update_route_evidence","confirmation_plan","measure_six_confirmation_fields","admit_confirmation_fields","measure_six_confirmation_fields","admit_confirmation_fields","admit_adapter_off_bf16_raw_parity","finish_in_memory_short_core")
    trace_keys=({"ordinal","operation"},{"ordinal","operation","update_number","schedule_index","row_iid"},{"ordinal","operation","update_number","schedule_index"},{"ordinal","operation","update_number","schedule_index","update_receipt_digest"},{"ordinal","operation","update_number","schedule_index","route_receipt_digest"},{"ordinal","operation","update_number","schedule_index","row_iid"},{"ordinal","operation","update_number","schedule_index"},{"ordinal","operation","update_number","schedule_index","update_receipt_digest"},{"ordinal","operation","update_number","schedule_index","route_receipt_digest"},{"ordinal","operation","row_iid","wrong_owner_iid"},{"ordinal","operation","schedule_index","row_iid","wrong_owner_iid"},{"ordinal","operation","schedule_index","provenance_digest","admission_digest"},{"ordinal","operation","schedule_index","row_iid","wrong_owner_iid"},{"ordinal","operation","schedule_index","provenance_digest","admission_digest"},{"ordinal","operation","schedule_indices","parity_digest"},{"ordinal","operation","trainer_receipt_digest"})
    require(isinstance(trace,list) and len(trace)==16 and all(isinstance(item,dict) and set(item)==trace_keys[position] for position,item in enumerate(trace)) and all(exact_int(item.get("ordinal"),position) for position,item in enumerate(trace)) and tuple(item.get("operation") for item in trace)==expected_ops,f"rank{rank} execution trace order/keys differ")
    for offset,ordinal in ((1,1),(5,2)):
        index=CONFIRMATION_INDICES[ordinal-1]
        require(exact_int(trace[offset].get("update_number"),ordinal) and exact_int(trace[offset].get("schedule_index"),index) and trace[offset].get("row_iid")==FIT_IIDS[arm] and exact_int(trace[offset+1].get("update_number"),ordinal) and exact_int(trace[offset+1].get("schedule_index"),index) and exact_int(trace[offset+2].get("update_number"),ordinal) and exact_int(trace[offset+2].get("schedule_index"),index) and trace[offset+2].get("update_receipt_digest")==updates[ordinal-1]["digest"] and exact_int(trace[offset+3].get("update_number"),ordinal) and exact_int(trace[offset+3].get("schedule_index"),index) and trace[offset+3].get("route_receipt_digest")==route_digests[ordinal-1],f"rank{rank} update{ordinal} execution trace differs")
    trace_parity_indices=trace[14].get("schedule_indices")
    require(trace[9].get("row_iid")==CONFIRMATION_IIDS[arm] and trace[9].get("wrong_owner_iid")==FIT_IIDS[arm] and exact_int(trace[10].get("schedule_index"),29) and trace[10].get("row_iid")==CONFIRMATION_IIDS[arm] and trace[10].get("wrong_owner_iid")==FIT_IIDS[arm] and exact_int(trace[11].get("schedule_index"),29) and trace[11].get("provenance_digest")==provenances[0]["digest"] and trace[11].get("admission_digest")==admissions[0]["digest"] and exact_int(trace[12].get("schedule_index"),38) and trace[12].get("row_iid")==CONFIRMATION_IIDS[arm] and trace[12].get("wrong_owner_iid")==FIT_IIDS[arm] and exact_int(trace[13].get("schedule_index"),38) and trace[13].get("provenance_digest")==provenances[1]["digest"] and trace[13].get("admission_digest")==admissions[1]["digest"] and isinstance(trace_parity_indices,list) and len(trace_parity_indices)==2 and all(exact_int(item,PARITY_INDICES[position]) for position,item in enumerate(trace_parity_indices)) and trace[14].get("parity_digest")==parity_digest and trace[15].get("trainer_receipt_digest")==trainer["digest"],f"rank{rank} confirmation/parity execution trace differs")
    false_authority(local,f"rank{rank} local"); false_authority(trainer,f"rank{rank} trainer"); recursive_denial(local,f"rank{rank} local")
    return {"digest":digest,"trainer_digest":trainer["digest"],"parity_digest":parity_digest,"consensus":consensus_values,"update_parameter_states":update_states,"initial_parameter_digest":trainer.get("initial_parameter_digest"),"final_parameter_digest":trainer.get("final_parameter_digest"),"final_base_digest":trainer.get("final_frozen_base_digest"),"optimizer_state_digest":trainer.get("optimizer_state_digest_after_training_and_confirmation"),"routing_digest":source["routing_digest"],"source_release_result_digest":trainer_source["source_release_result_digest"],"pinset_digest":trainer_source["pinset_digest"],"fit_source_sha256":fit_source,"confirmation_source_sha256":confirmation_source}

def validate_world8(result):
    result_digest=sealed(result,"runner result")
    result_keys={"schema_version","status","world8_rows","world8_full_results","topology_receipt","source_binding","checkpoint_identity","initialization","local_result","base_sha256_before","base_sha256_after","base_bytes_unchanged","base_gradients_all_none","wan_diffusion_sha256","transformer_wan_sha256","runtime_versions","checkpoint_written","publication_performed","digest"}|set(AUTHORITY)
    require(set(result)==result_keys and result.get("schema_version")=="bernini-graft-phase-a-short-world8-result-set-v2" and result.get("status")=="completed_in_memory_diagnostic_no_checkpoint" and result.get("checkpoint_written") is False and result.get("publication_performed") is False and result.get("base_bytes_unchanged") is True and result.get("base_gradients_all_none") is True and result.get("base_sha256_before")==result.get("base_sha256_after"),"runner diagnostic/base closure differs")
    for name in ("base_sha256_before","wan_diffusion_sha256","transformer_wan_sha256"): require_sha(result.get(name),name)
    require(result.get("wan_diffusion_sha256")=="59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512" and result.get("transformer_wan_sha256")=="9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223","Wan runtime source pin differs")
    runtime_versions=result.get("runtime_versions"); require(isinstance(runtime_versions,dict) and set(runtime_versions)=={"torch","torch_hip","diffusers","transformers"} and all(isinstance(value,str) and value for value in runtime_versions.values()),"runtime version observation differs")
    rows=result.get("world8_rows"); full=result.get("world8_full_results")
    row_keys={"global_rank","dp_arm","sp_rank","family","result_digest","trainer_result_digest","adapter_off_parity_digest"}
    require(isinstance(rows,list) and len(rows)==8 and all(isinstance(row,dict) and set(row)==row_keys and exact_int(row.get("global_rank"),rank) for rank,row in enumerate(rows)),"WORLD8 result rows differ")
    sealed(full,"WORLD8 full results"); receipts=full.get("all_eight_full_local_receipts"); arms=full.get("arm_representatives")
    full_keys={"schema_version","rank_order","dp2_family_order","all_eight_full_local_receipts","arm_representatives","dog_and_human_exact_coverage","all_eight_confirmation_hard_gates_passed","checkpoint_written","publication_performed","digest"}|set(AUTHORITY)
    arm_keys={"dp_arm","family","global_ranks","representative_global_rank","representative_full_receipt","per_rank_result_digests","all_four_confirmation_hard_gates_passed"}
    rank_order=full.get("rank_order")
    require(set(full)==full_keys and full.get("schema_version")=="bernini-graft-phase-a-short-world8-full-results-v1" and isinstance(rank_order,list) and len(rank_order)==8 and all(exact_int(item,rank) for rank,item in enumerate(rank_order)) and full.get("dp2_family_order")==list(FAMILIES) and isinstance(receipts,list) and len(receipts)==8 and isinstance(arms,list) and len(arms)==2 and all(isinstance(arm,dict) and set(arm)==arm_keys for arm in arms) and full.get("dog_and_human_exact_coverage") is True and full.get("all_eight_confirmation_hard_gates_passed") is True and full.get("checkpoint_written") is False and full.get("publication_performed") is False,"WORLD8 full result coverage differs")
    observations=[]
    for rank,(row,local) in enumerate(zip(rows,receipts)):
        arm=rank//4; sp_rank=rank%4
        require(exact_int(row.get("dp_arm"),arm) and exact_int(row.get("sp_rank"),sp_rank) and row.get("family")==FAMILIES[arm],f"rank{rank} WORLD8 summary geometry differs")
        observations.append(validate_local(local,rank,row))
    require(result.get("local_result")==receipts[0],"outer local result is not rank0 full receipt")
    for arm,representative_rank in enumerate((0,4)):
        representative=arms[arm]; arm_observations=observations[arm*4:(arm+1)*4]
        global_ranks=representative.get("global_ranks")
        require(exact_int(representative.get("dp_arm"),arm) and representative.get("family")==FAMILIES[arm] and isinstance(global_ranks,list) and len(global_ranks)==4 and all(exact_int(item,arm*4+position) for position,item in enumerate(global_ranks)) and exact_int(representative.get("representative_global_rank"),representative_rank) and representative.get("representative_full_receipt")==receipts[representative_rank] and representative.get("per_rank_result_digests")==[item["digest"] for item in arm_observations] and representative.get("all_four_confirmation_hard_gates_passed") is True,f"arm{arm} representative differs")
        require(len({tuple(item["consensus"]) for item in arm_observations})==1,f"arm{arm} SP4 confirmation consensus differs")
        require(len({(item["fit_source_sha256"],item["confirmation_source_sha256"]) for item in arm_observations})==1,f"arm{arm} SP4 source identity differs")
    require(observations[0]["fit_source_sha256"]!=observations[4]["fit_source_sha256"] and observations[0]["confirmation_source_sha256"]!=observations[4]["confirmation_source_sha256"],"dog/human source identities must differ")
    for key in ("final_parameter_digest","final_base_digest","optimizer_state_digest"):
        values=[item[key] for item in observations]
        require(all(isinstance(value,str) and SHA.fullmatch(value) for value in values) and len(set(values))==1,"WORLD8 "+key+" consensus differs")
    for ordinal in range(2): require(len({item["update_parameter_states"][ordinal] for item in observations})==1,f"WORLD8 update{ordinal+1} parameter/base consensus differs")
    for key in ("routing_digest","source_release_result_digest","pinset_digest","initial_parameter_digest"): require(len({item[key] for item in observations})==1,"WORLD8 "+key+" differs")
    for artifact,label in ((result.get("source_binding"),"source binding"),(result.get("checkpoint_identity"),"checkpoint identity"),(result.get("initialization"),"initialization"),(result.get("topology_receipt"),"topology receipt")): sealed(artifact,label)
    source_binding=result["source_binding"]; inference_files=source_binding.get("bernini_inference_files")
    expected_inference={"bernini/pipeline.py":"c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40","bernini/cli.py":"26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf","bernini/io_utils.py":"233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a"}
    require(set(source_binding)=={"schema_version","runner_sha256","consumer_sha256","native_v2_sha256","short_trainer_sha256","native_runner_v1_sha256","identity_rebinder_sha256","bernini_commit","veomni_commit","bernini_inference_files","digest"} and source_binding.get("schema_version")=="bernini-graft-phase-a-short-source-binding-v1" and source_binding.get("runner_sha256")=="4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4" and source_binding.get("consumer_sha256")==DEPENDENCY_PINS["a_lite_consumer"] and source_binding.get("native_v2_sha256")==DEPENDENCY_PINS["native_v2_closure"] and source_binding.get("short_trainer_sha256")==DEPENDENCY_PINS["short_trainer"] and source_binding.get("native_runner_v1_sha256")==DEPENDENCY_PINS["native_v1_gpu_runner_reuse"] and source_binding.get("identity_rebinder_sha256")=="1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0" and source_binding.get("bernini_commit")=="2d2b4591ac053ec25c6371b01a5a6746679e5793" and source_binding.get("veomni_commit")=="f90b3dc6fbb0ce693745223cc7a94064123dbf4d" and inference_files==expected_inference,"WORLD8 source provenance differs")
    checkpoint=result["checkpoint_identity"]; initialization=result["initialization"]; topology=result["topology_receipt"]
    checkpoint_body=checkpoint.get("identity")
    require(set(checkpoint)=={"schema_version","identity","digest"} and checkpoint.get("schema_version")=="bernini-graft-phase-a-short-checkpoint-v1" and isinstance(checkpoint_body,dict) and set(checkpoint_body)=={"manifest_path","manifest_sha256_computed","manifest_sha256_expected","verified_file_count","every_file_sha256_verified","verified_entries_digest"} and isinstance(checkpoint_body.get("manifest_path"),str) and checkpoint_body["manifest_path"].startswith("/") and checkpoint_body.get("manifest_sha256_computed")=="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831" and checkpoint_body.get("manifest_sha256_expected")==checkpoint_body["manifest_sha256_computed"] and checkpoint_body.get("verified_file_count")==23 and checkpoint_body.get("every_file_sha256_verified") is True and isinstance(checkpoint_body.get("verified_entries_digest"),str) and SHA.fullmatch(checkpoint_body["verified_entries_digest"]) is not None and object_digest(checkpoint_body)==checkpoint_pre==checkpoint_post,"checkpoint identity contract differs")
    require(set(initialization)=={"schema_version","rank0_broadcast_before_any_adapter_forward","parameter_count","parameter_sha256","zero_output_projection_count","zero_output_projection_exact","digest"} and initialization.get("schema_version")=="bernini-graft-phase-a-short-initial-registry-v1" and initialization.get("rank0_broadcast_before_any_adapter_forward") is True and type(initialization.get("parameter_count")) is int and initialization["parameter_count"]>0 and initialization.get("parameter_sha256")==observations[0]["initial_parameter_digest"] and type(initialization.get("zero_output_projection_count")) is int and 0<initialization["zero_output_projection_count"]<=initialization["parameter_count"] and initialization.get("zero_output_projection_exact") is True,"initialization contract differs")
    sp_members=topology.get("sp_members"); dp_members=topology.get("dp_members")
    require(set(topology)=={"schema_version","world_size","dp_size","sp_size","global_rank","local_rank","dp_arm","sp_rank","sp_members","dp_members","backend","digest"} and topology.get("schema_version")=="bernini-graft-phase-a-live-world8-dp2sp4-v1" and exact_int(topology.get("world_size"),8) and exact_int(topology.get("dp_size"),2) and exact_int(topology.get("sp_size"),4) and exact_int(topology.get("global_rank"),0) and exact_int(topology.get("local_rank"),0) and exact_int(topology.get("dp_arm"),0) and exact_int(topology.get("sp_rank"),0) and isinstance(sp_members,list) and len(sp_members)==4 and all(exact_int(item,position) for position,item in enumerate(sp_members)) and isinstance(dp_members,list) and len(dp_members)==2 and exact_int(dp_members[0],0) and exact_int(dp_members[1],4) and topology.get("backend")=="nccl","rank0 topology receipt differs")
    false_authority(result,"runner result"); false_authority(full,"WORLD8 full results"); recursive_denial(result,"runner result")
    return result_digest,observations

root_fd=os.dup(inherited_root_fd)
try:
    info=os.fstat(root_fd); leaf=output.lstat()
    if not stat.S_ISDIR(info.st_mode) or output.is_symlink() or (info.st_dev,info.st_ino)!=expected_identity or (leaf.st_dev,leaf.st_ino)!=expected_identity: raise SystemExit("output root identity differs")
    if logs!=output/"rank-logs": raise SystemExit("rank log root path differs")
    log_fd=os.dup(inherited_log_fd)
    try:
        log_info=os.fstat(log_fd); log_leaf=logs.lstat()
        if not stat.S_ISDIR(log_info.st_mode) or (log_info.st_dev,log_info.st_ino)!=expected_log_identity or (log_leaf.st_dev,log_leaf.st_ino)!=expected_log_identity: raise SystemExit("rank log root identity differs")
        stdout_files=[]
        for node in logs.rglob("*"):
            meta=node.lstat()
            if stat.S_ISLNK(meta.st_mode) or not (stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode)): raise SystemExit("rank log tree contains unsafe node")
            if stat.S_ISREG(meta.st_mode) and node.name=="stdout.log": stdout_files.append(node.relative_to(logs))
        ranks={}
        for relative in stdout_files:
            try: rank=int(relative.parent.name)
            except ValueError: raise SystemExit("rank stdout parent is not numeric")
            if rank in ranks or rank not in range(8): raise SystemExit("rank stdout coverage differs")
            ranks[rank]=relative
        if set(ranks)!=set(range(8)): raise SystemExit("exact eight redirected rank stdout logs required")
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
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>256*1024*1024: raise SystemExit("rank stdout is not bounded plain link-count-one")
                    chunks=[]; total=0
                    while True:
                        block=os.read(stream,1024*1024)
                        if not block: break
                        total+=len(block)
                        if total>256*1024*1024: raise SystemExit("rank stdout exceeds bound")
                        chunks.append(block)
                    after=os.fstat(stream)
                    fields=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode,value.st_nlink)
                    if fields(before)!=fields(after): raise SystemExit("rank stdout changed while reading")
                    raw=b"".join(chunks)
                finally: os.close(stream)
            finally: os.close(current)
            for line in raw.splitlines():
                if not (line.startswith(b"{") and line.endswith(b"}")): continue
                try: value=json.loads(line.decode("ascii"))
                except Exception: continue
                if isinstance(value,dict) and value.get("schema_version")=="bernini-graft-phase-a-short-world8-result-set-v2":
                    if line!=canonical(value): raise SystemExit("runner result is not canonical ASCII JSON")
                    admitted.append((rank,value))
        final_log_fd=os.fstat(log_fd); final_log_leaf=logs.lstat()
        if (final_log_fd.st_dev,final_log_fd.st_ino)!=(log_info.st_dev,log_info.st_ino) or (final_log_leaf.st_dev,final_log_leaf.st_ino)!=expected_log_identity: raise SystemExit("rank log root changed")
    finally: os.close(log_fd)
    if len(admitted)!=1 or admitted[0][0]!=0: raise SystemExit("exactly one rank-zero result schema required; ordinary logs are never receipts")
    result=admitted[0][1]; result_digest,world8_observations=validate_world8(result); rows=result["world8_rows"]
    if checkpoint_pre!=checkpoint_post: raise SystemExit("checkpoint pre/post digest differs")
    final_root_fd=os.fstat(root_fd); final_root_leaf=output.lstat()
    if (final_root_fd.st_dev,final_root_fd.st_ino)!=expected_identity or (final_root_leaf.st_dev,final_root_leaf.st_ino)!=expected_identity: raise SystemExit("output root changed before receipt publication")
    authority={name:False for name in AUTHORITY}
    core={"schema_version":"bernini-graft-phase-a-a-lite-short-world8-parent-v2","status":"completed_diagnostic_no_checkpoint","complete":True,"pass":True,"claim_scope":"two-update-source-noop-diagnostic-no-checkpoint-no-scientific-claim","topology":{"allocation":"single-node-8xMI210","world_size":8,"dp_size":2,"sp_size":4,"world8_rows":rows,"arm_representative_global_ranks":[0,4]},"validated":{"exact_rank_zero_v2_schema_count":1,"ordinary_logs_not_receipts":True,"rank_log_root_identity_retained_from_before_torchrun":True,"all_eight_full_local_receipts_deeply_validated":True,"dog_and_human_arm_representatives_deeply_validated":True,"world8_summary_digests_cross_checked":True,"sp4_per_arm_confirmation_consensus_cross_checked":True,"dp2_world_parameter_consensus_cross_checked":True,"world8_final_optimizer_digest_observations_equal":True,"optimizer_world_consensus_proven_by_producer":False,"reported_metric_scalar_relationships_recomputed":True,"reported_cosine_threshold_gate_checked":True,"metric_values_recomputed_from_tensor_bytes":False,"confirmation_cosine_recomputed_from_tensor_bytes":False,"field_tensor_content_sha256_recomputed_from_bytes":False,"atlas_receipt_bodies_deeply_validated":False,"source_release_routing_body_deeply_validated":False,"reported_route_and_source_digests_cross_linked":True,"native_admission_body_deeply_validated":False,"native_admission_digest_only_observed":True,"rank0_topology_receipt_validated":True,"rank1_through_rank7_topology_receipts_present":False,"updates":[29,38],"confirmation_indices":[29,38],"adapter_off_parity_indices":[0,25],"same_family_wrong":True,"all_eight_noncompensating_confirmation_gates_passed":True,"all_eight_adapter_off_bf16_raw_parity_passed":True,"all_eight_update_routes_and_provenance_validated":True,"frozen_base_bytes_unchanged_and_gradients_none":True,"vendor_trees_clean_pre_and_post":True,"checkpoint_content_full_rehash_pre_and_post":True,"runner_wrote_checkpoint":False},"provenance":{"source_commit":"a884d357a6c0742f751be48d226ba72c952bae76","runner_sha256":"4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4","plan_sha256":plan_sha,"source_archive_sha256":archive_sha,"runtime_closure_manifest_sha256":closure_sha,"runtime_closure_exact_regular_members":23,"launcher_sha256":launcher_sha,"terminal_admission_sha256_external_pin":terminal_sha,"terminal_materializer_runtime_sha256_external_pin":materializer_runtime_sha,"checkpoint_content_pre_sha256":checkpoint_pre,"checkpoint_content_post_sha256":checkpoint_post,"a_lite_job_id":"132549","allocator_environment":{"requested_environment_only":{"PYTORCH_HIP_ALLOC_CONF":"expandable_segments:True","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"},"effective_runtime_configuration_verified":False}},"runner_result_digest":result_digest,"world8_validated_result_digests":[item["digest"] for item in world8_observations],"runner_result":result,"checkpoint_written":False,"publication_performed":False,"authority":authority,"receipt_publication":{"create_only_O_EXCL":True,"mode":"0444","canonical_ascii_json_newline":True,"output_root_identity_retained_from_creation":True,"rank_log_root_identity_retained_from_before_torchrun":True,"provisional_mode_before_finalization":"0600","mode_0444_is_final_success_transition":True,"mode_0444_required_to_interpret_receipt_as_success":True,"failure_absence_of_any_receipt_inode_guaranteed":False,"provisional_non_success_inode_may_survive_cleanup_failure":True}}
    parent={**core,"digest":hashlib.sha256(canonical(core)).hexdigest()}; payload=canonical(parent)+b"\n"
    receipt_fd=os.open("receipt.json",os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=root_fd)
    try:
        offset=0
        while offset<len(payload):
            wrote=os.write(receipt_fd,payload[offset:])
            if wrote<=0: raise SystemExit("parent receipt write stalled")
            offset+=wrote
        os.fsync(receipt_fd); os.lseek(receipt_fd,0,os.SEEK_SET)
        observed=b""
        while True:
            block=os.read(receipt_fd,1024*1024)
            if not block: break
            observed+=block
        receipt_info=os.fstat(receipt_fd); path_info=os.stat("receipt.json",dir_fd=root_fd,follow_symlinks=False)
        if observed!=payload or not stat.S_ISREG(receipt_info.st_mode) or stat.S_IMODE(receipt_info.st_mode)!=0o600 or receipt_info.st_nlink!=1 or receipt_info.st_size!=len(payload) or (receipt_info.st_dev,receipt_info.st_ino)!=(path_info.st_dev,path_info.st_ino): raise SystemExit("provisional parent receipt differs")
        os.fsync(root_fd)
        os.fchmod(receipt_fd,0o444)
        os._exit(0)
    except BaseException:
        try:
            receipt_info=os.fstat(receipt_fd); path_info=os.stat("receipt.json",dir_fd=root_fd,follow_symlinks=False)
            if (receipt_info.st_dev,receipt_info.st_ino)==(path_info.st_dev,path_info.st_ino) and stat.S_IMODE(receipt_info.st_mode)!=0o444: os.unlink("receipt.json",dir_fd=root_fd)
        except OSError: pass
        os.close(receipt_fd)
        raise
finally: os.close(root_fd)
# END GRAFT_PHASE_A_SHORT_WORLD8_PARENT_VALIDATOR_V1
PY
