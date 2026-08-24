#!/usr/bin/bash
#SBATCH --job-name=saic-anchor-diag-v2
#SBATCH --partition=faculty
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# Exactly one QOS and every scientific/runtime input are supplied by the
# retained-FD v2 submitter.  This wrapper is a canary only.

set -Eeuo pipefail
umask 027
export PATH=/usr/bin:/bin

expected_bash_sha256="59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4"
[[ "${BASH}" == /usr/bin/bash && \
   "$(/usr/bin/realpath -e -- /proc/$$/exe)" == /usr/bin/bash && \
   "$(/usr/bin/sha256sum /proc/$$/exe | /usr/bin/awk '{print $1}')" == \
   "${expected_bash_sha256}" ]] || {
  echo "[saic-anchor-diag-v2] ERROR: admitted compute Bash differs" >&2
  exit 2
}

fail() { echo "[saic-anchor-diag-v2] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute/non-root"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 SHA-256 differs"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 revision differs"; }
identity() { /usr/bin/stat -Lc '%d:%i' -- "$1"; }

release_manifest="${SAIC_ANCHOR_DIAG_RELEASE_MANIFEST:?set release manifest}"
release_manifest_sha256="${SAIC_ANCHOR_DIAG_RELEASE_MANIFEST_SHA256:?set release manifest SHA-256}"
release_manifest_digest="${SAIC_ANCHOR_DIAG_RELEASE_MANIFEST_DIGEST:?set release manifest digest}"
source_archive="${SAIC_ANCHOR_DIAG_SOURCE_ARCHIVE:?set source archive}"
source_archive_sha256="${SAIC_ANCHOR_DIAG_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${SAIC_ANCHOR_DIAG_SOURCE_REVISION:?set source revision}"
member_manifest="${SAIC_ANCHOR_DIAG_MEMBER_MANIFEST:?set source member manifest}"
member_manifest_sha256="${SAIC_ANCHOR_DIAG_MEMBER_MANIFEST_SHA256:?set source member manifest SHA-256}"
member_manifest_digest="${SAIC_ANCHOR_DIAG_MEMBER_MANIFEST_DIGEST:?set source member manifest digest}"
member_count="${SAIC_ANCHOR_DIAG_MEMBER_COUNT:?set source member count}"
origin_manifest="${SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST:?set source origin manifest}"
origin_manifest_sha256="${SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST_SHA256:?set source origin manifest SHA-256}"
origin_manifest_digest="${SAIC_ANCHOR_DIAG_ORIGIN_MANIFEST_DIGEST:?set source origin manifest digest}"
origin_count="${SAIC_ANCHOR_DIAG_ORIGIN_COUNT:?set source origin count}"
launcher_sha256="${SAIC_ANCHOR_DIAG_LAUNCHER_SHA256:?set launcher SHA-256}"
python_bin="${SAIC_ANCHOR_DIAG_PYTHON:?set pinned vace Python}"
python_sha256="${SAIC_ANCHOR_DIAG_PYTHON_SHA256:?set Python SHA-256}"
rendezvous_guard="${SAIC_ANCHOR_DIAG_RENDEZVOUS_GUARD:?set rendezvous guard}"
rendezvous_guard_sha256="${SAIC_ANCHOR_DIAG_RENDEZVOUS_GUARD_SHA256:?set rendezvous guard SHA-256}"
bernini_root="${SAIC_ANCHOR_DIAG_BERNINI_ROOT:?set Bernini root}"
veomni_root="${SAIC_ANCHOR_DIAG_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${SAIC_ANCHOR_DIAG_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${SAIC_ANCHOR_DIAG_CHECKPOINT_MANIFEST:?set checkpoint manifest}"
checkpoint_manifest_sha256="${SAIC_ANCHOR_DIAG_CHECKPOINT_MANIFEST_SHA256:?set checkpoint manifest SHA-256}"
source_manifest="${SAIC_ANCHOR_DIAG_SOURCE_MANIFEST:?set source-anchor manifest}"
source_manifest_sha256="${SAIC_ANCHOR_DIAG_SOURCE_MANIFEST_SHA256:?set source-anchor manifest SHA-256}"
stage_a_adapter="${SAIC_ANCHOR_DIAG_STAGE_A_ADAPTER:?set Stage-A adapter}"
stage_a_adapter_sha256="${SAIC_ANCHOR_DIAG_STAGE_A_ADAPTER_SHA256:?set Stage-A adapter SHA-256}"
stage_a_receipt="${SAIC_ANCHOR_DIAG_STAGE_A_RECEIPT:?set Stage-A receipt}"
stage_a_receipt_sha256="${SAIC_ANCHOR_DIAG_STAGE_A_RECEIPT_SHA256:?set Stage-A receipt SHA-256}"
stage_a_postflight="${SAIC_ANCHOR_DIAG_STAGE_A_POSTFLIGHT:?set Stage-A formal postflight}"
stage_a_postflight_sha256="${SAIC_ANCHOR_DIAG_STAGE_A_POSTFLIGHT_SHA256:?set Stage-A formal postflight SHA-256}"
stage_a_history="${SAIC_ANCHOR_DIAG_STAGE_A_HISTORY:?set Stage-A exact32 history}"
stage_a_history_sha256="${SAIC_ANCHOR_DIAG_STAGE_A_HISTORY_SHA256:?set Stage-A exact32 history SHA-256}"
stage_a_checkpoint_release="${SAIC_ANCHOR_DIAG_STAGE_A_CHECKPOINT_RELEASE:?set Stage-A checkpoint release receipt}"
stage_a_checkpoint_release_sha256="${SAIC_ANCHOR_DIAG_STAGE_A_CHECKPOINT_RELEASE_SHA256:?set Stage-A checkpoint release SHA-256}"
heldout_row_index="${SAIC_ANCHOR_DIAG_HELDOUT_ROW_INDEX:?set held-out row index}"
action_caption="${SAIC_ANCHOR_DIAG_ACTION_CAPTION:?set action caption file}"
action_caption_sha256="${SAIC_ANCHOR_DIAG_ACTION_CAPTION_SHA256:?set action caption SHA-256}"
visual_checkpoint="${SAIC_ANCHOR_DIAG_VISUAL_CHECKPOINT:?set frozen DINOv2 checkpoint}"
visual_manifest="${SAIC_ANCHOR_DIAG_VISUAL_MANIFEST:?set visual checkpoint manifest}"
visual_manifest_sha256="${SAIC_ANCHOR_DIAG_VISUAL_MANIFEST_SHA256:?set visual manifest SHA-256}"
visual_release="${SAIC_ANCHOR_DIAG_VISUAL_RELEASE:?set visual release manifest}"
visual_release_sha256="${SAIC_ANCHOR_DIAG_VISUAL_RELEASE_SHA256:?set visual release manifest SHA-256}"
visual_evaluator_spec="${SAIC_ANCHOR_DIAG_VISUAL_EVALUATOR_SPEC:?set visual evaluator spec}"
visual_evaluator_spec_sha256="${SAIC_ANCHOR_DIAG_VISUAL_EVALUATOR_SPEC_SHA256:?set visual evaluator spec SHA-256}"
output="${SAIC_ANCHOR_DIAG_OUTPUT:?set fresh diagnostic output}"
output_parent_device="${SAIC_ANCHOR_DIAG_OUTPUT_PARENT_DEVICE:?set output parent device}"
output_parent_inode="${SAIC_ANCHOR_DIAG_OUTPUT_PARENT_INODE:?set output parent inode}"
submission_receipt="${SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT:?set submission receipt}"
submission_receipt_device="${SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT_DEVICE:?set receipt device}"
submission_receipt_inode="${SAIC_ANCHOR_DIAG_SUBMISSION_RECEIPT_INODE:?set receipt inode}"
slurm_log_dir="${SAIC_ANCHOR_DIAG_SLURM_LOG_DIR:?set Slurm log directory}"
slurm_log_dir_device="${SAIC_ANCHOR_DIAG_SLURM_LOG_DIR_DEVICE:?set Slurm log device}"
slurm_log_dir_inode="${SAIC_ANCHOR_DIAG_SLURM_LOG_DIR_INODE:?set Slurm log inode}"
postflight="${SAIC_ANCHOR_DIAG_POSTFLIGHT:?set postflight source}"
postflight_sha256="${SAIC_ANCHOR_DIAG_POSTFLIGHT_SHA256:?set postflight SHA-256}"
seed="${SAIC_ANCHOR_DIAG_SEED:?set fixed seed}"

for name in release_manifest source_archive member_manifest origin_manifest python_bin \
  rendezvous_guard bernini_root veomni_root checkpoint checkpoint_manifest \
  source_manifest stage_a_adapter stage_a_receipt stage_a_postflight stage_a_history stage_a_checkpoint_release \
  action_caption visual_checkpoint visual_manifest visual_release \
  visual_evaluator_spec output submission_receipt slurm_log_dir postflight; do
  require_absolute "${!name}" "${name}"
done
for name in release_manifest_sha256 release_manifest_digest source_archive_sha256 \
  member_manifest_sha256 member_manifest_digest origin_manifest_sha256 \
  origin_manifest_digest launcher_sha256 python_sha256 rendezvous_guard_sha256 \
  checkpoint_manifest_sha256 source_manifest_sha256 stage_a_adapter_sha256 \
  stage_a_receipt_sha256 stage_a_postflight_sha256 stage_a_history_sha256 stage_a_checkpoint_release_sha256 \
  action_caption_sha256 visual_manifest_sha256 visual_release_sha256 \
  visual_evaluator_spec_sha256 postflight_sha256; do
  require_sha256 "${!name}" "${name}"
done
require_sha1 "${source_revision}" source_revision
[[ "${member_count}" =~ ^[1-9][0-9]*$ && "${origin_count}" =~ ^[1-9][0-9]*$ ]] || fail "source closure counts differ"
[[ "${heldout_row_index}" =~ ^([0-9]|1[0-5])$ && "${seed}" =~ ^[0-9]+$ ]] || fail "scientific scalar differs"
[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[1-9][0-9]*$ ]] || fail "Slurm job ID differs"

# Open all bootstrap trust roots before any wait, test, extraction, or model load.
exec {release_fd}<"${release_manifest}"
exec {archive_fd}<"${source_archive}"
exec {member_fd}<"${member_manifest}"
exec {origin_fd}<"${origin_manifest}"
exec {guard_fd}<"${rendezvous_guard}"
exec {receipt_fd}<"${submission_receipt}"
for pair in \
  "${release_fd}:${release_manifest_sha256}" \
  "${archive_fd}:${source_archive_sha256}" \
  "${member_fd}:${member_manifest_sha256}" \
  "${origin_fd}:${origin_manifest_sha256}" \
  "${guard_fd}:${rendezvous_guard_sha256}"; do
  fd_number="${pair%%:*}"; expected_sha="${pair#*:}"
  [[ "$(/usr/bin/sha256sum "/proc/$$/fd/${fd_number}" | /usr/bin/awk '{print $1}')" == "${expected_sha}" ]] || fail "retained bootstrap bytes differ"
done
[[ "$(identity "/proc/$$/fd/${receipt_fd}")" == "${submission_receipt_device}:${submission_receipt_inode}" ]] || fail "retained submission receipt identity differs"

[[ -f "$0" && "$(/usr/bin/sha256sum "$0" | /usr/bin/awk '{print $1}')" == "${launcher_sha256}" ]] || fail "spooled launcher bytes differ"
[[ -x "${python_bin}" && ! -L "${python_bin}" && "$(/usr/bin/sha256sum "${python_bin}" | /usr/bin/awk '{print $1}')" == "${python_sha256}" ]] || fail "Python bytes differ"
for name in checkpoint_manifest source_manifest stage_a_adapter stage_a_receipt \
  stage_a_postflight stage_a_history stage_a_checkpoint_release action_caption visual_manifest visual_release \
  visual_evaluator_spec postflight; do
  value="${!name}"; sha_name="${name}_sha256"; expected_sha="${!sha_name}"
  [[ -f "${value}" && ! -L "${value}" && "$(/usr/bin/stat -c '%h' -- "${value}")" == 1 && \
     "$(/usr/bin/sha256sum "${value}" | /usr/bin/awk '{print $1}')" == "${expected_sha}" ]] || fail "${name} bytes/identity differ"
done
for name in bernini_root veomni_root checkpoint visual_checkpoint; do
  value="${!name}"; [[ -d "${value}" && ! -L "${value}" ]] || fail "${name} directory differs"
done

output_parent="$(/usr/bin/dirname -- "${output}")"
[[ "$(identity "${output_parent}")" == "${output_parent_device}:${output_parent_inode}" && \
   "$(/usr/bin/stat -c '%u:%a' -- "${output_parent}")" == "$(/usr/bin/id -u):700" && \
   "$(identity "${slurm_log_dir}")" == "${slurm_log_dir_device}:${slurm_log_dir_inode}" && \
   "$(/usr/bin/stat -c '%u:%a' -- "${slurm_log_dir}")" == "$(/usr/bin/id -u):700" && \
   ! -e "${output}" && ! -L "${output}" ]] || fail "fresh output/log namespace differs"

# Wait only for the already-opened reservation inode to become terminal 0444.
receipt_ready=false
for _attempt in $(/usr/bin/seq 1 120); do
  if [[ "$(/usr/bin/stat -Lc '%d:%i:%a' -- "/proc/$$/fd/${receipt_fd}")" == \
        "${submission_receipt_device}:${submission_receipt_inode}:444" ]]; then
    receipt_ready=true; break
  fi
  /usr/bin/sleep 1
done
[[ "${receipt_ready}" == true && \
   "$(identity "${submission_receipt}")" == "${submission_receipt_device}:${submission_receipt_inode}" ]] || fail "submission receipt did not seal on its retained inode"
submission_receipt_sha256="$(/usr/bin/sha256sum "/proc/$$/fd/${receipt_fd}" | /usr/bin/awk '{print $1}')"
require_sha256 "${submission_receipt_sha256}" submission_receipt

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(/usr/bin/mktemp -d "${scratch_parent%/}/saic-anchor-diag-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/saic-anchor-diag-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    /usr/bin/chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    /usr/bin/rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

/usr/bin/mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/bootstrap" \
  "${task_scratch}/cache/miopen-user" "${task_scratch}/cache/miopen-custom" \
  "${task_scratch}/cache/torch-extensions" "${task_scratch}/cache/triton" \
  "${task_scratch}/cache/pycache" "${task_scratch}/tmp"
/usr/bin/cp -- "/proc/$$/fd/${archive_fd}" "${task_scratch}/bootstrap/source.tar"
/usr/bin/cp -- "/proc/$$/fd/${release_fd}" "${task_scratch}/bootstrap/release-manifest.json"
/usr/bin/cp -- "/proc/$$/fd/${member_fd}" "${task_scratch}/bootstrap/member-manifest.json"
/usr/bin/cp -- "/proc/$$/fd/${origin_fd}" "${task_scratch}/bootstrap/origin-manifest.json"
/usr/bin/cp -- "/proc/$$/fd/${guard_fd}" "${task_scratch}/bootstrap/rendezvous-guard.py"
/usr/bin/chmod 0444 -- "${task_scratch}/bootstrap/"*
[[ "$(/usr/bin/git get-tar-commit-id <"${task_scratch}/bootstrap/source.tar")" == "${source_revision}" ]] || fail "source archive revision differs"

# Validate every tar member and extract only regular files/directories.  The
# manifest must carry the six v2 release leaves, including all formerly
# untracked sources, before any project import is allowed.
"${python_bin}" -I -B - \
  "${task_scratch}/bootstrap/source.tar" \
  "${task_scratch}/bootstrap/member-manifest.json" \
  "${task_scratch}/source-tree" \
  "${source_archive_sha256}" "${member_manifest_digest}" "${member_count}" \
  "${launcher_sha256}" "${postflight_sha256}" <<'PY'
import hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath

archive, manifest_path, destination = map(Path, sys.argv[1:4])
archive_sha, expected_digest, expected_count = sys.argv[4:7]
launcher_sha, postflight_sha = sys.argv[7:9]
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
raw = manifest_path.read_bytes(); value = json.loads(raw.decode("ascii"))
unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
fields = {"schema_version","archive_sha256","member_count","members","members_digest","required_release_files","authority","receipt_digest"}
if (set(value) != fields or value.get("schema_version") != "saic-source-anchor-diagnostic-source-members-v1" or raw != canonical(value)+b"\n" or claimed != hashlib.sha256(canonical(unsigned)).hexdigest() or value.get("archive_sha256") != archive_sha or value.get("members_digest") != expected_digest or value.get("member_count") != int(expected_count)):
    raise SystemExit("source member manifest seal differs")
if hashlib.sha256(archive.read_bytes()).hexdigest() != archive_sha:
    raise SystemExit("retained source archive bytes differ")
rows=[]; payloads={}; seen=set()
with tarfile.open(archive, mode="r:*") as handle:
    for member in handle.getmembers():
        name=member.name
        pure=PurePosixPath(name)
        if (not name or pure.is_absolute() or ".." in pure.parts or name in seen or "\\" in name or not (member.isdir() or member.isfile())):
            raise SystemExit("source archive member safety/uniqueness differs")
        seen.add(name)
        if member.isdir():
            data=None; kind="directory"; digest=None; size=0
        else:
            stream=handle.extractfile(member)
            if stream is None: raise SystemExit("source archive file payload absent")
            data=stream.read(); kind="file"; digest=hashlib.sha256(data).hexdigest(); size=len(data); payloads[name]=data
        rows.append({"name":name,"type":kind,"mode":stat.S_IMODE(member.mode),"size":size,"sha256":digest})
if rows != value.get("members") or hashlib.sha256(canonical(rows)).hexdigest() != expected_digest or len(rows) != int(expected_count):
    raise SystemExit("source archive exact member closure differs")
required=value.get("required_release_files")
required_names={
 "methods/bernini_action_editing/infer_saic_source_anchor_checkpoint_diagnostic_v1.py":None,
 "methods/bernini_action_editing/scripts/auh_infer_saic_source_anchor_checkpoint_diagnostic_v1.sbatch":launcher_sha,
 "methods/bernini_action_editing/tools/submit_saic_source_anchor_checkpoint_diagnostic_v1.py":None,
 "methods/bernini_action_editing/tools/postflight_saic_source_anchor_checkpoint_diagnostic_v1.py":postflight_sha,
 "methods/bernini_action_editing/tests/test_infer_saic_source_anchor_checkpoint_diagnostic_v1.py":None,
 "methods/bernini_action_editing/tests/test_auh_infer_saic_source_anchor_checkpoint_diagnostic_v1_launcher.py":None,
}
if not isinstance(required, dict) or set(required) != set(required_names):
    raise SystemExit("source archive six-leaf release closure differs")
for name, pinned in required_names.items():
    observed=hashlib.sha256(payloads.get(name,b"")).hexdigest()
    if required.get(name) != observed or (pinned is not None and observed != pinned):
        raise SystemExit("source archive release leaf bytes differ")
for row in rows:
    target=destination / row["name"]
    if row["type"] == "directory": target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor=os.open(target, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        try:
            data=payloads[row["name"]]; offset=0
            while offset < len(data): offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally: os.close(descriptor)
        target.chmod(0o444)
PY

method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root differs"

# Isolated recursive import-origin closure.  The sealed origin manifest must
# equal every project module actually loaded by importing the diagnostic.
"${python_bin}" -I -B - "${method_root}" \
  "${task_scratch}/bootstrap/origin-manifest.json" \
  "${source_archive_sha256}" "${origin_manifest_digest}" "${origin_count}" <<'PY'
import hashlib, importlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); path=Path(sys.argv[2]); archive_sha=sys.argv[3]; expected_digest=sys.argv[4]; expected_count=int(sys.argv[5])
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
raw=path.read_bytes(); value=json.loads(raw.decode("ascii")); unsigned=dict(value); claimed=unsigned.pop("receipt_digest",None)
fields={"schema_version","source_archive_sha256","module_count","modules","modules_digest","authority","receipt_digest"}
if (set(value)!=fields or value.get("schema_version")!="saic-source-anchor-diagnostic-import-origins-v1" or raw!=canonical(value)+b"\n" or claimed!=hashlib.sha256(canonical(unsigned)).hexdigest() or value.get("source_archive_sha256")!=archive_sha or value.get("modules_digest")!=expected_digest or value.get("module_count")!=expected_count):
    raise SystemExit("source import-origin manifest seal differs")
sys.path.insert(0,str(root)); importlib.import_module("infer_saic_source_anchor_checkpoint_diagnostic_v1")
rows=[]
for name,module in sorted(sys.modules.items()):
    origin=getattr(module,"__file__",None)
    if not origin: continue
    candidate=Path(origin).resolve()
    try: relative=candidate.relative_to(root).as_posix()
    except ValueError: continue
    if candidate.suffix != ".py": continue
    rows.append({"module":name,"relative_path":relative,"sha256":hashlib.sha256(candidate.read_bytes()).hexdigest()})
if rows!=value.get("modules") or hashlib.sha256(canonical(rows)).hexdigest()!=expected_digest or len(rows)!=expected_count:
    raise SystemExit("recursive project import-origin closure differs")
required={"infer_saic_source_anchor_checkpoint_diagnostic_v1","infer_lora","infer_native_identity_generation_canary","infer_native_self_guided_action_field_canary","infer_source_kv_carrier_oracle","pair_v5_source_bound_preservation_evaluator_v1","saic_exact81_media_diagnostics_v1","saic_source_anchor_adapter_v1","saic_source_anchor_native_runtime_v1","score_pair_v5_source_bound_preservation_v1","train_saic_source_anchor_v1"}
if not required.issubset({row["module"] for row in rows}):
    raise SystemExit("required recursive project imports are absent")
PY

export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"

# Preserve the four-GPU Slurm allocation and expose it through exactly one ROCm
# visibility variable.  Never replace it with an assumed 0,1,2,3 mapping.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  gpu_visibility_source=ROCR_VISIBLE_DEVICES; gpu_visibility="${ROCR_VISIBLE_DEVICES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_visibility_source=CUDA_VISIBLE_DEVICES; gpu_visibility="${CUDA_VISIBLE_DEVICES}"
else
  fail "Slurm GPU visibility is absent"
fi
IFS=',' read -r -a gpu_tokens <<<"${gpu_visibility}"
[[ "${#gpu_tokens[@]}" == 4 ]] || fail "Slurm GPU visibility count differs"
declare -A seen_gpu=()
for token in "${gpu_tokens[@]}"; do
  [[ "${token}" =~ ^[0-9]+$ && -z "${seen_gpu[${token}]:-}" ]] || fail "Slurm GPU visibility token differs"
  seen_gpu["${token}"]=1
done
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES="${gpu_visibility}"

# The terminal submission receipt is the complete manifest/export authority.
"${python_bin}" -I -B - "/proc/$$/fd/${release_fd}" "/proc/$$/fd/${receipt_fd}" \
  "${release_manifest_sha256}" "${release_manifest_digest}" "${SLURM_JOB_ID}" \
  "${output}" "${gpu_visibility_source}" "${gpu_visibility}" <<'PY'
import hashlib,json,os,stat,sys
from pathlib import Path
release_path,receipt_path=map(Path,sys.argv[1:3]); release_sha,release_digest,job_id,output,gpu_source,gpu_value=sys.argv[3:9]
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def load(path,label):
 raw=path.read_bytes(); value=json.loads(raw.decode("ascii")); unsigned=dict(value); claimed=unsigned.pop("receipt_digest",None)
 if raw!=canonical(value)+b"\n" or claimed!=hashlib.sha256(canonical(unsigned)).hexdigest(): raise SystemExit(label+" seal differs")
 return value,raw
release,release_raw=load(release_path,"release manifest"); receipt,receipt_raw=load(receipt_path,"submission receipt")
if hashlib.sha256(release_raw).hexdigest()!=release_sha or release.get("receipt_digest")!=release_digest or release.get("schema_version")!="saic-source-anchor-diagnostic-release-v2": raise SystemExit("external release anchor differs")
exports=receipt.get("exports")
if not isinstance(exports,dict) or set(exports)!={name for name in os.environ if name.startswith("SAIC_ANCHOR_DIAG_")} or any(os.environ.get(name)!=value for name,value in exports.items()): raise SystemExit("complete Slurm export binding differs")
if (receipt.get("schema_version")!="saic-source-anchor-diagnostic-submission-v2" or receipt.get("status")!="submitted_single_attempt_no_retry" or receipt.get("submitted_job",{}).get("job_id")!=job_id or receipt.get("outputs",{}).get("diagnostic_output")!=output or receipt.get("inputs",{}).get("release_manifest",{}).get("sha256")!=release_sha or receipt.get("inputs",{}).get("resolved_release")!=release or exports.get("SAIC_ANCHOR_DIAG_RELEASE_MANIFEST_DIGEST")!=release_digest): raise SystemExit("submission input/job/output binding differs")
def exact_row(row,label,directory=False):
 if not isinstance(row,dict) or not isinstance(row.get("path"),str): raise SystemExit(label+" snapshot differs")
 path=Path(row["path"]); info=path.lstat()
 if path.resolve(strict=True)!=path or path==Path("/") or stat.S_ISLNK(info.st_mode) or (directory and not stat.S_ISDIR(info.st_mode)) or (not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1)) or row.get("device")!=info.st_dev or row.get("inode")!=info.st_ino or row.get("mode")!=f"{stat.S_IMODE(info.st_mode):04o}" or row.get("uid")!=info.st_uid: raise SystemExit(label+" current device/inode/mode differs")
 if not directory and (row.get("nlink")!=info.st_nlink or row.get("byte_size")!=info.st_size or hashlib.sha256(path.read_bytes()).hexdigest()!=row.get("sha256")): raise SystemExit(label+" current bytes differ")
for label,row in (("release root",release.get("release_root")),): exact_row(row,label,True)
for section in ("source_release","code","executables","base_model","scientific_input","stage_a_release","visual_release"):
 value=release.get(section)
 if not isinstance(value,dict): raise SystemExit(section+" release section differs")
 for key,row in value.items():
  if not isinstance(row,dict) or "path" not in row: continue
  exact_row(row,section+"/"+key,"sha256" not in row)
if gpu_source not in {"ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES"} or len(set(gpu_value.split(",")))!=4: raise SystemExit("GPU visibility evidence differs")
PY

# AUH-only contract tests and artifact preflight execute from the exact archive.
for test_file in test_infer_saic_source_anchor_checkpoint_diagnostic_v1.py \
  test_auh_infer_saic_source_anchor_checkpoint_diagnostic_v1_launcher.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

common_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}"
  --source-anchor-manifest "${source_manifest}"
  --expected-source-anchor-manifest-sha256 "${source_manifest_sha256}"
  --stage-a-adapter "${stage_a_adapter}"
  --expected-stage-a-adapter-sha256 "${stage_a_adapter_sha256}"
  --stage-a-receipt "${stage_a_receipt}"
  --expected-stage-a-receipt-sha256 "${stage_a_receipt_sha256}"
  --stage-a-formal-postflight "${stage_a_postflight}"
  --expected-stage-a-formal-postflight-sha256 "${stage_a_postflight_sha256}"
  --stage-a-history "${stage_a_history}"
  --expected-stage-a-history-sha256 "${stage_a_history_sha256}"
  --stage-a-checkpoint-release "${stage_a_checkpoint_release}"
  --expected-stage-a-checkpoint-release-sha256 "${stage_a_checkpoint_release_sha256}"
  --heldout-row-index "${heldout_row_index}"
  --action-caption-file "${action_caption}"
  --expected-action-caption-file-sha256 "${action_caption_sha256}"
  --visual-checkpoint "${visual_checkpoint}"
  --visual-checkpoint-content-manifest "${visual_manifest}"
  --expected-visual-checkpoint-content-manifest-sha256 "${visual_manifest_sha256}"
  --visual-release-manifest "${visual_release}"
  --expected-visual-release-manifest-sha256 "${visual_release_sha256}"
  --visual-evaluator-spec "${visual_evaluator_spec}"
  --expected-visual-evaluator-spec-sha256 "${visual_evaluator_spec_sha256}"
  --rendezvous-guard "${task_scratch}/bootstrap/rendezvous-guard.py"
  --expected-rendezvous-guard-sha256 "${rendezvous_guard_sha256}"
  --gpu-visibility-source "${gpu_visibility_source}"
  --expected-gpu-visibility "${gpu_visibility}"
  --output "${output}"
  --seed "${seed}"
  --num-inference-steps 40
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
  --method-source-release-manifest "${release_manifest}"
  --expected-method-source-release-manifest-sha256 "${release_manifest_sha256}"
  --method-source-member-manifest-sha256 "${member_manifest_sha256}"
  --method-source-member-manifest-digest "${member_manifest_digest}"
  --method-source-member-count "${member_count}"
  --method-source-origin-manifest-sha256 "${origin_manifest_sha256}"
  --method-source-origin-manifest-digest "${origin_manifest_digest}"
  --method-source-origin-count "${origin_count}"
  --ack-diagnostic-only-no-stage-b-authority
)
PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/infer_saic_source_anchor_checkpoint_diagnostic_v1.py" \
  "${common_args[@]}" \
  --rendezvous-evidence-root "${output}.rendezvous" \
  --expected-rendezvous-id "preflight-no-rendezvous" \
  --artifact-preflight-only

rendezvous_root="${output}.rendezvous"
[[ ! -e "${rendezvous_root}" && ! -L "${rendezvous_root}" ]] || fail "rendezvous evidence root is not fresh"
/usr/bin/mkdir -m 0700 -- "${rendezvous_root}"
rdzv_id="saic-anchor-${SLURM_JOB_ID}-${release_manifest_digest:0:16}"
[[ "${rdzv_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "rendezvous ID differs"

echo "[saic-anchor-diag-v2] start canary=true GPUs=4 topology=Ulysses4 cells=7 exact81=true exact40=true dynamic_rendezvous=true stage_b=false"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:0 --rdzv_id="${rdzv_id}" \
  "${method_root}/infer_saic_source_anchor_checkpoint_diagnostic_v1.py" \
  "${common_args[@]}" \
  --rendezvous-evidence-root "${rendezvous_root}" \
  --expected-rendezvous-id "${rdzv_id}" \
  --slurm-job-id "${SLURM_JOB_ID}"

[[ -d "${output}" && ! -L "${output}" && "$(/usr/bin/stat -c '%a' -- "${output}")" == 555 ]] || fail "diagnostic output publication differs"
[[ -f "${output}/receipt.json" && -f "${output}/decoded-diagnostics.json" && \
   "$(/usr/bin/find "${output}" -mindepth 1 -maxdepth 1 -type f | /usr/bin/wc -l)" == 19 ]] || fail "terminal artifact closure differs"
[[ "$(/usr/bin/find "${rendezvous_root}" -mindepth 1 -maxdepth 1 -type f | /usr/bin/wc -l)" == 5 && \
   -f "${rendezvous_root}/admission.json" ]] || fail "rendezvous evidence closure differs"
/usr/bin/chmod 0555 -- "${rendezvous_root}"
[[ "$(/usr/bin/sha256sum "${source_archive}" | /usr/bin/awk '{print $1}')" == "${source_archive_sha256}" && \
   "$(/usr/bin/sha256sum "${stage_a_adapter}" | /usr/bin/awk '{print $1}')" == "${stage_a_adapter_sha256}" && \
   "$(identity "${output_parent}")" == "${output_parent_device}:${output_parent_inode}" ]] || fail "terminal input/namespace closure changed"
bootstrap_names=(release_manifest source_archive member_manifest origin_manifest rendezvous_guard)
bootstrap_fds=("${release_fd}" "${archive_fd}" "${member_fd}" "${origin_fd}" "${guard_fd}")
bootstrap_shas=("${release_manifest_sha256}" "${source_archive_sha256}" "${member_manifest_sha256}" "${origin_manifest_sha256}" "${rendezvous_guard_sha256}")
for index in "${!bootstrap_names[@]}"; do
  name="${bootstrap_names[${index}]}"; fd_number="${bootstrap_fds[${index}]}"; expected_sha="${bootstrap_shas[${index}]}"; path="${!name}"
  [[ "$(identity "${path}")" == "$(identity "/proc/$$/fd/${fd_number}")" && \
     "$(/usr/bin/sha256sum "${path}" | /usr/bin/awk '{print $1}')" == "${expected_sha}" && \
     "$(/usr/bin/sha256sum "/proc/$$/fd/${fd_number}" | /usr/bin/awk '{print $1}')" == "${expected_sha}" ]] || fail "terminal retained bootstrap closure changed: ${name}"
done
[[ "$(identity "${submission_receipt}")" == "$(identity "/proc/$$/fd/${receipt_fd}")" && \
   "$(/usr/bin/sha256sum "${submission_receipt}" | /usr/bin/awk '{print $1}')" == "${submission_receipt_sha256}" && \
   "$(/usr/bin/sha256sum "/proc/$$/fd/${receipt_fd}" | /usr/bin/awk '{print $1}')" == "${submission_receipt_sha256}" ]] || fail "terminal retained submission receipt closure changed"

sentinel="SAIC_SOURCE_ANCHOR_CHECKPOINT_DIAGNOSTIC_V2_CANARY_OK job_id=${SLURM_JOB_ID} release_digest=${release_manifest_digest} output=${output}"
echo "${sentinel}"
