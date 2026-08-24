#!/usr/bin/env bash
#SBATCH --job-name=sail-sirm-2xsp4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=06:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# One allocation, two independent WORLD4 groups.  Each group computes one
# current-endpoint relational VJP and decodes its own base/+/- exact81 triplet.
# The final all8 directory (including master receipt) appears in one rename
# only after both children and the independent postflight have succeeded.

source_archive="${SIRM_SOURCE_ARCHIVE:?set SIRM_SOURCE_ARCHIVE}"
source_archive_sha256="${SIRM_SOURCE_ARCHIVE_SHA256:?set SIRM_SOURCE_ARCHIVE_SHA256}"
source_revision="${SIRM_SOURCE_REVISION:?set SIRM_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
specificity_receipt="${SIRM_SPECIFICITY_RECEIPT:?set SIRM_SPECIFICITY_RECEIPT}"
specificity_receipt_sha256="${SIRM_SPECIFICITY_RECEIPT_SHA256:?set SIRM_SPECIFICITY_RECEIPT_SHA256}"
output_root="${SIRM_OUTPUT_ROOT:?set SIRM_OUTPUT_ROOT}"
python_bin="${SIRM_PYTHON_BIN:?set SIRM_PYTHON_BIN}"

for group in DOG HUMAN; do
  for field in CANDIDATE_MANIFEST CANDIDATE_MANIFEST_SHA256 SOURCE_VIDEO SOURCE_VIDEO_SHA256 ACTION_CAPTION ACTION_CAPTION_SHA256 NOOP_CAPTION NOOP_CAPTION_SHA256 CLEAN_LATENT CLEAN_LATENT_SHA256 NATIVE_NOISE NATIVE_NOISE_SHA256 BASE_MP4 BASE_MP4_SHA256 BASE_RECEIPT BASE_RECEIPT_SHA256 TEACHER_RECEIPT TEACHER_RECEIPT_SHA256 TEACHER_RESIDUAL TEACHER_RESIDUAL_SHA256 TEACHER_RESIDUAL_TENSOR_SHA256 TEACHER_EPISODE_ID; do
    name="SIRM_${group}_${field}"
    [[ -n "${!name:-}" ]] || { echo "[sirm-dual4] ERROR: set ${name}" >&2; exit 2; }
  done
done

readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly pinned_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly pinned_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

fail() { echo "[sirm-dual4] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute non-root"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_id() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$ ]] || fail "$2 is not path-safe"; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() { [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"; }

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest specificity_receipt output_root python_bin; do
  require_absolute "${!name}" "$name"
done
require_sha256 "${source_archive_sha256}" source_archive_sha256
require_sha256 "${specificity_receipt_sha256}" specificity_receipt_sha256
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be 40-hex"
for group in DOG HUMAN; do
  for field in CANDIDATE_MANIFEST SOURCE_VIDEO ACTION_CAPTION NOOP_CAPTION CLEAN_LATENT NATIVE_NOISE BASE_MP4 BASE_RECEIPT TEACHER_RECEIPT TEACHER_RESIDUAL; do
    name="SIRM_${group}_${field}"
    require_absolute "${!name}" "$name"
  done
  for field in CANDIDATE_MANIFEST_SHA256 SOURCE_VIDEO_SHA256 ACTION_CAPTION_SHA256 NOOP_CAPTION_SHA256 CLEAN_LATENT_SHA256 NATIVE_NOISE_SHA256 BASE_MP4_SHA256 BASE_RECEIPT_SHA256 TEACHER_RECEIPT_SHA256 TEACHER_RESIDUAL_SHA256 TEACHER_RESIDUAL_TENSOR_SHA256; do
    name="SIRM_${group}_${field}"
    require_sha256 "${!name}" "$name"
  done
  name="SIRM_${group}_TEACHER_EPISODE_ID"
  require_id "${!name}" "$name"
done

source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
specificity_receipt="$(realpath -e -- "${specificity_receipt}")"
python_bin="$(realpath -e -- "${python_bin}")"
for name in source_archive checkpoint_manifest specificity_receipt; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "$name must be a plain file"
done
for name in bernini_root veomni_root checkpoint; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "$name must be a plain directory"
done
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python executable differs"
for group in DOG HUMAN; do
  for field in CANDIDATE_MANIFEST SOURCE_VIDEO ACTION_CAPTION NOOP_CAPTION CLEAN_LATENT NATIVE_NOISE BASE_MP4 BASE_RECEIPT TEACHER_RECEIPT TEACHER_RESIDUAL; do
    name="SIRM_${group}_${field}"
    declare "${name}=$(realpath -e -- "${!name}")"
    [[ -f "${!name}" && ! -L "${!name}" ]] || fail "$name must be a plain file"
  done
done

check_hash "${source_archive}" "${source_archive_sha256}" source_archive
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
check_hash "${checkpoint_manifest}" "${pinned_checkpoint_manifest_sha256}" checkpoint_manifest
check_hash "${specificity_receipt}" "${specificity_receipt_sha256}" specificity_receipt
specificity_binding="$("${python_bin}" -B - \
  "${specificity_receipt}" \
  "${SIRM_DOG_TEACHER_RECEIPT_SHA256}" \
  "${SIRM_DOG_TEACHER_RESIDUAL_SHA256}" \
  "${SIRM_DOG_TEACHER_RESIDUAL_TENSOR_SHA256}" \
  "${SIRM_DOG_TEACHER_EPISODE_ID}" \
  "${SIRM_HUMAN_TEACHER_RECEIPT_SHA256}" \
  "${SIRM_HUMAN_TEACHER_RESIDUAL_SHA256}" \
  "${SIRM_HUMAN_TEACHER_RESIDUAL_TENSOR_SHA256}" \
  "${SIRM_HUMAN_TEACHER_EPISODE_ID}" <<'PY'
from pathlib import Path
import hashlib
import json
import math
import sys

path=Path(sys.argv[1])
expected={
    'dog':{
        'receipt_file_sha256':sys.argv[2],
        'artifact_file_sha256':sys.argv[3],
        'tensor_sha256':sys.argv[4],
        'episode_id':sys.argv[5],
    },
    'human':{
        'receipt_file_sha256':sys.argv[6],
        'artifact_file_sha256':sys.argv[7],
        'tensor_sha256':sys.argv[8],
        'episode_id':sys.argv[9],
    },
}
core_roles=['same_video_reverse','same_video_phase_shuffle','semantic_generic_wrong_motion']
def pairs(rows):
    out={}
    for key,value in rows:
        if key in out: raise SystemExit('specificity receipt contains duplicate key')
        out[key]=value
    return out
row=json.loads(path.read_text(encoding='ascii'),object_pairs_hook=pairs,
               parse_constant=lambda token: (_ for _ in ()).throw(SystemExit('specificity receipt contains nonfinite JSON')))
declared=row.get('receipt_digest')
unsigned=dict(row); unsigned.pop('receipt_digest',None)
payload=json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
cells=row.get('cells')
if (row.get('schema_version')!='bernini-sail-relational-specificity-audit-v1'
    or declared!=hashlib.sha256(payload).hexdigest()
    or row.get('fixed_core_control_roles')!=core_roles
    or row.get('minimum_core_control_mismatch')!=.05
    or row.get('threshold_cli_exposed') is not False
    or row.get('seed_block_schedule_or_loss_selection_performed') is not False
    or row.get('training_performed') is not False
    or row.get('editor_update_authorized') is not False
    or row.get('all_families_core_specificity_passed') is not True
    or row.get('endpoint_vjp_submission_authorized') is not True
    or not isinstance(cells,list) or len(cells)!=2
    or sorted(cell.get('family') for cell in cells)!=['dog','human']):
    raise SystemExit('specificity receipt does not authorize the fixed dog/human endpoint gate')
for cell in cells:
    family=cell.get('family')
    mismatch=cell.get('core_control_mismatch')
    arms=cell.get('arms')
    positive=arms.get('positive') if isinstance(arms,dict) else None
    if (cell.get('core_specificity_passed') is not True
        or cell.get('core_control_mismatch_floor')!=.05
        or not isinstance(mismatch,dict) or set(mismatch)!=set(core_roles)
        or any(isinstance(value,bool) or not isinstance(value,(int,float))
               or not math.isfinite(float(value)) or float(value)<.05
               for value in mismatch.values())
        or not isinstance(positive,dict)
        or cell.get('episode_id')!=expected[family]['episode_id']
        or any(positive.get(key)!=expected[family][key]
               for key in ('receipt_file_sha256','artifact_file_sha256','tensor_sha256'))):
        raise SystemExit('specificity receipt core-control separation differs')
audit_sha=row.get('audit_source_file_sha256')
core_sha=row.get('relational_core_file_sha256')
if any(not isinstance(value,str) or len(value)!=64 for value in (audit_sha,core_sha)):
    raise SystemExit('specificity receipt source binding differs')
print(declared,audit_sha,core_sha)
PY
)"
read -r specificity_receipt_digest specificity_audit_source_sha256 specificity_core_source_sha256 <<<"${specificity_binding}"
require_sha256 "${specificity_receipt_digest}" specificity_receipt_digest
require_sha256 "${specificity_audit_source_sha256}" specificity_audit_source_sha256
require_sha256 "${specificity_core_source_sha256}" specificity_core_source_sha256
for group in DOG HUMAN; do
  for field in CANDIDATE_MANIFEST SOURCE_VIDEO ACTION_CAPTION NOOP_CAPTION CLEAN_LATENT NATIVE_NOISE BASE_MP4 BASE_RECEIPT TEACHER_RECEIPT TEACHER_RESIDUAL; do
    path_name="SIRM_${group}_${field}"
    digest_name="${path_name}_SHA256"
    check_hash "${!path_name}" "${!digest_name}" "$path_name"
  done
done

output_parent="$(dirname -- "${output_root}")"
output_name="$(basename -- "${output_root}")"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || fail "output parent differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/sirm-dual4-${SLURM_JOB_ID}.XXXXXX")"
staging_root="${output_parent}/.${output_name}.stage-${SLURM_JOB_ID}"
[[ ! -e "${staging_root}" && ! -L "${staging_root}" ]] || fail "staging root already exists"
mkdir -- "${staging_root}"
dog_pid=""
human_pid=""
published=false

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  if [[ "${published}" != true && -d "${staging_root}" && ! -L "${staging_root}" ]]; then
    case "${staging_root}" in "${output_parent}/.${output_name}.stage-${SLURM_JOB_ID}") rm -rf -- "${staging_root}" || cleanup_failed=1 ;; *) cleanup_failed=1 ;; esac
  fi
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    case "${task_scratch}" in "${scratch_parent%/}/sirm-dual4-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}" || cleanup_failed=1; rm -rf -- "${task_scratch}" || cleanup_failed=1 ;; *) cleanup_failed=1 ;; esac
  fi
  if [[ "${status}" == 0 && "${cleanup_failed}" == 0 && "${published}" == true ]]; then
    echo "[sirm-dual4] PASS all8=true topology=dog:0-3+human:4-7 dose_rms=.03 output=${output_root}"
  else
    [[ "${status}" != 0 ]] || status=2
    echo "[sirm-dual4] EXIT status=${status} published=${published} cleanup_verified=$([[ ${cleanup_failed} == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
check_hash "${archive_copy}" "${source_archive_sha256}" archive_copy

# Scan *all* members, reject links/devices/duplicates/path traversal, then
# create files with O_EXCL.  No shell tar extraction or archive-owned mode is used.
"${python_bin}" -B - "${archive_copy}" "${task_scratch}/source-tree" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import sys
import tarfile

archive = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve(strict=True)
required = {
    "methods/bernini_action_editing/run_self_imagined_relational_motion_sp4_v1.py",
    "methods/bernini_action_editing/self_imagined_relational_motion.py",
    "methods/bernini_action_editing/starch_live_vjp_bridge_v1.py",
    "methods/bernini_action_editing/run_starc_live_vjp_sp4_v1.py",
    "methods/bernini_action_editing/materialize_starc_core4_hidden_v1.py",
    "methods/bernini_action_editing/temporal_counterfactual_action_scorer_v1.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/audit_self_imagined_relational_specificity_v1.py",
    "methods/bernini_action_editing/scripts/auh_run_self_imagined_relational_motion_dual4_v1.sbatch",
    "methods/bernini_action_editing/tests/test_run_self_imagined_relational_motion_sp4_v1.py",
    "methods/bernini_action_editing/tests/test_auh_run_self_imagined_relational_motion_dual4_v1_launcher.py",
}
seen = set()
regular = {}
with tarfile.open(archive, "r:*") as handle:
    members = handle.getmembers()
    for member in members:
        pure = PurePosixPath(member.name)
        parts = tuple(part for part in pure.parts if part not in ("", "."))
        if pure.is_absolute() or not parts or ".." in parts:
            raise SystemExit("unsafe source archive path")
        normalized = PurePosixPath(*parts).as_posix()
        if normalized in seen:
            raise SystemExit("duplicate source archive member")
        seen.add(normalized)
        if not (member.isdir() or member.isfile()):
            raise SystemExit("source archive contains link/device/fifo member")
        destination = root.joinpath(*parts)
        if root not in destination.parents:
            raise SystemExit("source archive destination escapes root")
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True, mode=0o755)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        stream = handle.extractfile(member)
        if stream is None:
            raise SystemExit("plain member has no payload")
        digest = hashlib.sha256()
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "wb") as output:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        regular[normalized] = digest.hexdigest()
missing = sorted(required - regular.keys())
if missing:
    raise SystemExit(f"source archive lacks SIRM runtime closure: {missing}")
closure = {"schema_version": "sirm-source-closure-v1", "members": regular}
payload = json.dumps(closure, sort_keys=True, separators=(",", ":")).encode() + b"\n"
path = root / ".sirm-source-closure.json"
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as output:
    output.write(payload); output.flush(); os.fsync(output.fileno())
PY

source_tree="${task_scratch}/source-tree"
method_root="${source_tree}/methods/bernini_action_editing"
loader="${method_root}/run_self_imagined_relational_motion_sp4_v1.py"
launcher="${method_root}/scripts/auh_run_self_imagined_relational_motion_dual4_v1.sbatch"
specificity_audit_source="${method_root}/audit_self_imagined_relational_specificity_v1.py"
specificity_core_source="${method_root}/self_imagined_relational_motion.py"
closure_manifest="${source_tree}/.sirm-source-closure.json"
for path in "${loader}" "${launcher}" "${specificity_audit_source}" "${specificity_core_source}" "${closure_manifest}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "extracted runtime closure differs"
done
verify_source_tree_closure() {
  "${python_bin}" -B - "${source_tree}" "${closure_manifest}" <<'PY'
from pathlib import Path
import hashlib
import json
import stat
import sys

root=Path(sys.argv[1]).resolve(strict=True)
manifest=Path(sys.argv[2]).resolve(strict=True)
row=json.loads(manifest.read_text(encoding='ascii'))
members=row.get('members')
if row.get('schema_version')!='sirm-source-closure-v1' or not isinstance(members,dict):
    raise SystemExit('source closure manifest differs')
actual={}
for path in root.rglob('*'):
    if path == manifest:
        continue
    mode=path.lstat().st_mode
    if stat.S_ISLNK(mode) or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)):
        raise SystemExit('source closure contains a link or special entry')
    if stat.S_ISREG(mode):
        relative=path.relative_to(root).as_posix()
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):
                digest.update(chunk)
        actual[relative]=digest.hexdigest()
if actual != members:
    raise SystemExit('extracted source-tree file closure or content differs')
PY
}
verify_source_tree_closure
check_hash "${specificity_audit_source}" "${specificity_audit_source_sha256}" specificity_audit_source
check_hash "${specificity_core_source}" "${specificity_core_source_sha256}" specificity_core_source
# Slurm may execute a spool copy.  Bind bytes, never require path equality.
running_launcher="$(realpath -e -- "${BASH_SOURCE[0]}")"
[[ "$(hash_file "${running_launcher}")" == "$(hash_file "${launcher}")" ]] || fail "running launcher content differs from archive"
loader_sha256="$(hash_file "${loader}")"
closure_sha256="$(hash_file "${closure_manifest}")"
find "${source_tree}" -type f -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export PYTHONPATH="${method_root}:${source_tree}"

common_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --expected-checkpoint-tree-sha256 "${pinned_checkpoint_tree_sha256}"
  --expected-checkpoint-content-manifest-sha256 "${pinned_checkpoint_manifest_sha256}"
  --expected-bernini-commit "${pinned_bernini_commit}"
  --expected-veomni-commit "${pinned_veomni_commit}"
  --source-archive "${source_archive}"
  --expected-source-archive-sha256 "${source_archive_sha256}"
  --source-git-revision "${source_revision}"
  --expected-loader-source-sha256 "${loader_sha256}"
  --ack-mechanism-probe-only
  --ack-no-editor-parameter-or-update
  --ack-no-scientific-or-action-editing-claim
  --ack-live-query-has-no-source-condition
)

launch_group() {
  local family="$1" upper="$2" devices="$3" port="$4" output="$5" log="$6"
  local cache="${task_scratch}/groups/${family}"
  mkdir -- "${cache}" "${cache}/torch" "${cache}/miopen" "${cache}/triton" "${cache}/xdg" "${cache}/tmp"
  # Under ``set -u``, Bash expands every RHS in one ``local`` builtin before
  # installing any of the assignments.  Keep each dependent variable on a
  # later command so the Slurm runtime cannot observe an unbound base name.
  local candidate="SIRM_${upper}_CANDIDATE_MANIFEST"
  local candidate_sha="${candidate}_SHA256"
  local source="SIRM_${upper}_SOURCE_VIDEO"
  local source_sha="${source}_SHA256"
  local action="SIRM_${upper}_ACTION_CAPTION"
  local action_sha="${action}_SHA256"
  local noop="SIRM_${upper}_NOOP_CAPTION"
  local noop_sha="${noop}_SHA256"
  local clean="SIRM_${upper}_CLEAN_LATENT"
  local clean_sha="${clean}_SHA256"
  local noise="SIRM_${upper}_NATIVE_NOISE"
  local noise_sha="${noise}_SHA256"
  local base="SIRM_${upper}_BASE_MP4"
  local base_sha="${base}_SHA256"
  local base_receipt="SIRM_${upper}_BASE_RECEIPT"
  local base_receipt_sha="${base_receipt}_SHA256"
  local teacher_receipt="SIRM_${upper}_TEACHER_RECEIPT"
  local teacher_receipt_sha="${teacher_receipt}_SHA256"
  local teacher_residual="SIRM_${upper}_TEACHER_RESIDUAL"
  local teacher_residual_sha="${teacher_residual}_SHA256"
  local teacher_tensor_sha="SIRM_${upper}_TEACHER_RESIDUAL_TENSOR_SHA256"
  local teacher_episode="SIRM_${upper}_TEACHER_EPISODE_ID"
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="${devices}" \
    TORCH_EXTENSIONS_DIR="${cache}/torch" MIOPEN_USER_DB_PATH="${cache}/miopen" \
    TRITON_CACHE_DIR="${cache}/triton" XDG_CACHE_HOME="${cache}/xdg" TMPDIR="${cache}/tmp" \
    TORCHELASTIC_ERROR_FILE="${cache}/torch-elastic-error.json" \
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
      --master_addr=127.0.0.1 --master_port="${port}" \
      "${loader}" "${common_args[@]}" \
      --candidate-manifest "${!candidate}" --expected-candidate-manifest-sha256 "${!candidate_sha}" \
      --source-video "${!source}" --expected-source-video-sha256 "${!source_sha}" \
      --action-caption-file "${!action}" --expected-action-caption-file-sha256 "${!action_sha}" \
      --noop-caption-file "${!noop}" --expected-noop-caption-file-sha256 "${!noop_sha}" \
      --current-clean-latent "${!clean}" --expected-current-clean-latent-sha256 "${!clean_sha}" \
      --native-noise "${!noise}" --expected-native-noise-sha256 "${!noise_sha}" \
      --base-mp4 "${!base}" --expected-base-mp4-sha256 "${!base_sha}" \
      --base-receipt "${!base_receipt}" --expected-base-receipt-sha256 "${!base_receipt_sha}" \
      --teacher-receipt "${!teacher_receipt}" --expected-teacher-receipt-sha256 "${!teacher_receipt_sha}" \
      --teacher-residual "${!teacher_residual}" --expected-teacher-residual-sha256 "${!teacher_residual_sha}" \
      --expected-teacher-residual-tensor-sha256 "${!teacher_tensor_sha}" \
      --teacher-episode-id "${!teacher_episode}" --output-dir "${output}" \
      --published-output-dir "${output_root}/${family}" \
      >"${log}" 2>&1
}

echo "[sirm-dual4] allocation=8xMI210 topology=dog:0,1,2,3+human:4,5,6,7 exact81=true fixed_dose=.03 source_condition_in_live_query=false"
base_port=$((20000 + (SLURM_JOB_ID % 20000) * 2))
launch_group dog DOG 0,1,2,3 "${base_port}" "${staging_root}/dog" "${staging_root}/dog.log" & dog_pid=$!
launch_group human HUMAN 4,5,6,7 "$((base_port + 1))" "${staging_root}/human" "${staging_root}/human.log" & human_pid=$!
set +e
wait "${dog_pid}"; dog_status=$?
wait "${human_pid}"; human_status=$?
set -e
dog_pid=""; human_pid=""
if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
  tail -n 100 -- "${staging_root}/dog.log" >&2 || true
  tail -n 100 -- "${staging_root}/human.log" >&2 || true
  fail "WORLD4 child failed dog=${dog_status} human=${human_status}"
fi

# Re-authenticate every external input and the immutable archive after both GPU
# jobs.  A changed input can never be hidden by a successful child receipt.
check_hash "${source_archive}" "${source_archive_sha256}" source_archive_postflight
check_hash "${archive_copy}" "${source_archive_sha256}" archive_copy_postflight
check_hash "${checkpoint_manifest}" "${pinned_checkpoint_manifest_sha256}" checkpoint_manifest_postflight
check_hash "${specificity_receipt}" "${specificity_receipt_sha256}" specificity_receipt_postflight
for group in DOG HUMAN; do
  for field in CANDIDATE_MANIFEST SOURCE_VIDEO ACTION_CAPTION NOOP_CAPTION CLEAN_LATENT NATIVE_NOISE BASE_MP4 BASE_RECEIPT TEACHER_RECEIPT TEACHER_RESIDUAL; do
    path_name="SIRM_${group}_${field}"; digest_name="${path_name}_SHA256"
    check_hash "${!path_name}" "${!digest_name}" "${path_name}_postflight"
  done
done
[[ "$(hash_file "${running_launcher}")" == "$(hash_file "${launcher}")" ]] || fail "launcher changed during run"
[[ "$(hash_file "${loader}")" == "${loader_sha256}" ]] || fail "loader changed during run"
check_hash "${specificity_audit_source}" "${specificity_audit_source_sha256}" specificity_audit_source_postflight
check_hash "${specificity_core_source}" "${specificity_core_source_sha256}" specificity_core_source_postflight
[[ "$(hash_file "${closure_manifest}")" == "${closure_sha256}" ]] || fail "source closure changed during run"
verify_source_tree_closure

# Independent child postflight: exact paths, hashes, single tensor keys/shapes,
# fixed dose and explicit T2V-style query boundary.  Master is written only
# after this script returns success.
"${python_bin}" -B - "${staging_root}" "${source_archive_sha256}" "${source_revision}" "${closure_sha256}" "${loader_sha256}" "${output_root}" "${specificity_receipt}" "${specificity_receipt_sha256}" "${specificity_receipt_digest}" "${specificity_audit_source_sha256}" "${specificity_core_source_sha256}" "${checkpoint}" "${checkpoint_manifest}" "${pinned_checkpoint_tree_sha256}" "${pinned_checkpoint_manifest_sha256}" <<'PY'
from pathlib import Path
import hashlib
import json
import math
import os
import sys
from safetensors import safe_open
import starch_live_vjp_bridge_v1 as live_bridge
from tools import materialize_vae
import torch

root = Path(sys.argv[1]).resolve(strict=True)
archive_sha, revision, closure_sha, loader_sha = sys.argv[2:6]
published_root = Path(sys.argv[6])
specificity_path=Path(sys.argv[7])
specificity_sha=sys.argv[8]
specificity_digest=sys.argv[9]
specificity_audit_source_sha=sys.argv[10]
specificity_core_source_sha=sys.argv[11]
checkpoint_root=Path(sys.argv[12])
checkpoint_manifest=Path(sys.argv[13])
checkpoint_tree_sha=sys.argv[14]
checkpoint_manifest_sha=sys.argv[15]
if not published_root.is_absolute() or published_root.exists():
    raise SystemExit('published root must remain fresh during postflight')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
if sha(specificity_path)!=specificity_sha:
    raise SystemExit('specificity receipt changed before master publication')
def strict_json(path):
    def pairs(rows):
        out={}
        for k,v in rows:
            if k in out: raise SystemExit('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(SystemExit('nonfinite JSON')))
def tensor_digest(tensor):
    owned=tensor.detach().cpu().contiguous().clone()
    metadata={'dtype':str(owned.dtype),'shape':[int(v) for v in owned.shape]}
    header=json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'|'+bytes(owned.untyped_storage())).hexdigest()
children={}
child_checkpoint_bindings={}
for family in ('dog','human'):
    child=(root/family).resolve(strict=True)
    if child.parent != root: raise SystemExit('child path escapes staging root')
    receipt_path=child/'receipt.json'
    row=strict_json(receipt_path)
    receipt_digest=row.get('receipt_digest')
    unsigned_child=dict(row)
    unsigned_child.pop('receipt_digest',None)
    child_canonical=json.dumps(unsigned_child,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if receipt_digest!=hashlib.sha256(child_canonical).hexdigest():
        raise SystemExit(f'{family} child receipt seal differs')
    if (row.get('schema_version')!='bernini-sail-relational-motion-sp4-v1'
        or row.get('source_condition_in_live_query') is not False
        or row.get('current_endpoint_origin')!='authenticated_native_rv2v'
        or row.get('source_preservation_mechanism')!='endpoint_base_plus_fixed_projection_only'
        or row.get('current_query_is_rv2v_forward') is not False
        or row.get('scientific_claim_authorized') is not False
        or row.get('action_editing_success_claim_authorized') is not False):
        raise SystemExit(f'{family} receipt semantic boundary differs')
    child_checkpoint_bindings[family]=row.get('checkpoint_binding')
    intervention=row.get('intervention',{})
    if (float(intervention.get('dose_rms',-1)).hex()!=float(.03).hex()
        or intervention.get('dose_cli_or_search_exposed') is not False
        or intervention.get('projector_survival_floor_fixed_inside_core') is not True):
        raise SystemExit(f'{family} fixed intervention differs')
    scorer=row.get('scorer_contract',{})
    if (scorer.get('class')!='FrozenRelationalMotionScorer'
        or scorer.get('learned_head_or_checkpoint_consumed') is not False
        or scorer.get('signed_temporal_quotient') is not True
        or scorer.get('squared_gram_loss') is not True):
        raise SystemExit(f'{family} scorer contract differs')
    artifact_rows=[row['raw_cotangent_artifact'],row['projected_cotangent_artifact'],row['fixed_dose_delta_artifact']]
    artifact_rows += [row['latent_artifacts'][arm] for arm in ('base','plus','minus')]
    artifact_rows += [row['video_artifacts'][arm] for arm in ('base','plus','minus')]
    for artifact in artifact_rows:
        path=Path(artifact['path'])
        # Child receipt was authored against the final all8 path.  During the
        # all8 transaction only the common root prefix is still staging.
        expected=child/path.name
        if path != published_root/family/path.name or not expected.is_file() or expected.is_symlink():
            raise SystemExit(f'{family} artifact path/closure differs')
        if sha(expected)!=artifact['file_sha256']:
            raise SystemExit(f'{family} artifact hash differs')
    tensors={}
    for label, artifact in (
        ('raw',row['raw_cotangent_artifact']),
        ('projected',row['projected_cotangent_artifact']),
        ('delta',row['fixed_dose_delta_artifact']),
        *((arm,row['latent_artifacts'][arm]) for arm in ('base','plus','minus')),
    ):
        path=child/Path(artifact['path']).name
        with safe_open(str(path),framework='pt',device='cpu') as opened:
            if list(opened.keys()) != [artifact['tensor_key']]:
                raise SystemExit(f'{family}/{label} tensor key differs')
            tensor=opened.get_tensor(artifact['tensor_key']).contiguous()
        if (str(tensor.dtype)!='torch.float32'
            or list(tensor.shape)!=artifact['shape']
            or not bool(torch.isfinite(tensor).all().item())
            or tensor_digest(tensor)!=artifact['tensor_sha256']):
            raise SystemExit(f'{family}/{label} tensor value binding differs')
        tensors[label]=tensor
    if not (
        tuple(tensors['base'].shape[:3])==(1,16,21)
        and tuple(tensors['raw'].shape)==tuple(tensors['base'].shape)
        and tuple(tensors['projected'].shape)==tuple(tensors['base'].shape)
        and tuple(tensors['delta'].shape)==tuple(tensors['base'].shape)
        and tuple(tensors['plus'].shape)==tuple(tensors['base'].shape)
        and tuple(tensors['minus'].shape)==tuple(tensors['base'].shape)
    ):
        raise SystemExit(f'{family} exact81 latent/cotangent geometry differs')
    delta=tensors['delta']
    if (not torch.equal(tensors['plus'],(tensors['base']+delta).contiguous())
        or not torch.equal(tensors['minus'],(tensors['base']-delta).contiguous())):
        raise SystemExit(f'{family} plus/minus do not replay the sealed fixed-dose delta')
    dose=float(delta.to(torch.float64).square().mean().sqrt().item())
    if not math.isclose(dose,.03,rel_tol=1.0e-6,abs_tol=1.0e-8):
        raise SystemExit(f'{family} intervention dose differs')
    projected=tensors['projected'].to(torch.float64)
    delta64=delta.to(torch.float64)
    if (float(projected[:,:,0].abs().max().item())!=0.0
        or float(projected.sum(dim=2).abs().max().item())>3.0e-6):
        raise SystemExit(f'{family} nuisance-null projection constraints differ')
    height,width=projected.shape[-2:]
    x=torch.linspace(-1.0,1.0,width,dtype=torch.float64)
    y=torch.linspace(-1.0,1.0,height,dtype=torch.float64)
    basis=torch.stack((torch.ones(height,width,dtype=torch.float64),
                       x.unsqueeze(0).expand(height,width),
                       y.unsqueeze(1).expand(height,width))).reshape(3,-1)
    basis=basis/torch.linalg.vector_norm(basis,dim=1,keepdim=True)
    affine_dot=projected.reshape(-1,height*width)@basis.T
    if float(affine_dot.abs().max().item())>3.0e-6:
        raise SystemExit(f'{family} spatial-affine nuisance projection differs')
    raw64=tensors['raw'].to(torch.float64)
    raw_norm=float(torch.linalg.vector_norm(raw64).item())
    projected_norm=float(torch.linalg.vector_norm(projected).item())
    delta_norm=float(torch.linalg.vector_norm(delta64).item())
    if min(raw_norm,projected_norm,delta_norm)<=0.0:
        raise SystemExit(f'{family} cotangent/delta norm is zero')
    ascent=float(torch.dot(raw64.reshape(-1),projected.reshape(-1)).item())
    ascent/=raw_norm*projected_norm
    alignment=float(torch.dot(projected.reshape(-1),delta64.reshape(-1)).item())
    alignment/=projected_norm*delta_norm
    if (projected_norm/raw_norm < .10
        or ascent < .10 or alignment < .999999):
        raise SystemExit(f'{family} projector survival floor differs')
    for arm in ('base','plus','minus'):
        latent=child/f'{arm}.normalized-clean-latent.safetensors'
        with safe_open(str(latent),framework='pt',device='cpu') as opened:
            if list(opened.keys()) != ['normalized_clean_latent']:
                raise SystemExit(f'{family}/{arm} latent key differs')
            tensor=opened.get_tensor('normalized_clean_latent')
        if tuple(tensor.shape[:3])!=(1,16,21) or str(tensor.dtype)!='torch.float32':
            raise SystemExit(f'{family}/{arm} latent shape/dtype differs')
        video=child/f'{arm}.mp4'
        frames,fps,hw=materialize_vae._decode_exact_video(video)
        if int(frames.shape[0])!=81 or int(round(float(fps)))!=25:
            raise SystemExit(f'{family}/{arm} MP4 not exact81/fps25')
        metadata=row['video_artifacts'][arm]
        if [int(hw[0]),int(hw[1])] != [metadata['height'],metadata['width']]:
            raise SystemExit(f'{family}/{arm} MP4 geometry differs')
    children[family]={
        'receipt_path':str(published_root/family/'receipt.json'),
        'receipt_file_sha256':sha(receipt_path),
        'receipt_digest':receipt_digest,
        'candidate_id':row['candidate_id'],
    }
# Re-enumerate and rehash the entire pinned 23-file checkpoint closure only
# after both children have loaded the transformer/VAE and every child artifact
# has been reopened.  This is not a manifest-only check.
checkpoint_postflight=live_bridge.authenticate_frozen_bernini_checkpoint_content(
    checkpoint_root,
    checkpoint_manifest,
    expected_checkpoint_tree_sha256=checkpoint_tree_sha,
    expected_checkpoint_content_manifest_sha256=checkpoint_manifest_sha,
).receipt()
if any(binding != checkpoint_postflight for binding in child_checkpoint_bindings.values()):
    raise SystemExit('child/preflight and full postflight checkpoint identities differ')
logs={name:{'path':str(published_root/name),'file_sha256':sha(root/name)}
      for name in ('dog.log','human.log')}
unsigned={
    'schema_version':'bernini-sail-relational-motion-dual4-master-v1',
    'topology':{'allocated_gpus':8,'groups':{'dog':[0,1,2,3],'human':[4,5,6,7]},'world_size_each':4},
    'children':children,
    'logs':logs,
    'source_archive_file_sha256':archive_sha,
    'source_git_revision':revision,
    'source_closure_file_sha256':closure_sha,
    'runtime_loader_file_sha256':loader_sha,
    'specificity_gate':{
        'path':str(specificity_path),
        'file_sha256':specificity_sha,
        'receipt_digest':specificity_digest,
        'audit_source_file_sha256':specificity_audit_source_sha,
        'relational_core_file_sha256':specificity_core_source_sha,
        'endpoint_vjp_submission_authorized':True,
    },
    'checkpoint_content_postflight':checkpoint_postflight,
    'checkpoint_content_postflight_verified_after_all_child_artifacts':True,
    'fixed_dose_rms':.03,
    'source_condition_in_live_query':False,
    'postflight_complete':True,
    'all_six_mp4_exact81':True,
    'training_performed':False,
    'scientific_claim_authorized':False,
    'action_editing_success_claim_authorized':False,
}
canonical=lambda x: json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
master={**unsigned,'receipt_digest':hashlib.sha256(canonical(unsigned)).hexdigest()}
payload=canonical(master)+b'\n'
path=root/'master-receipt.json'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'wb') as out:
    out.write(payload); out.flush(); os.fsync(out.fileno())
if sha(path)!=hashlib.sha256(payload).hexdigest(): raise SystemExit('master write differs')
PY

[[ -f "${staging_root}/master-receipt.json" && ! -L "${staging_root}/master-receipt.json" ]] || fail "postflight did not create master"
check_hash "${archive_copy}" "${source_archive_sha256}" archive_copy_final
check_hash "${source_archive}" "${source_archive_sha256}" source_archive_final
check_hash "${specificity_receipt}" "${specificity_receipt_sha256}" specificity_receipt_final
check_hash "${checkpoint_manifest}" "${pinned_checkpoint_manifest_sha256}" checkpoint_manifest_final
for group in DOG HUMAN; do
  for field in CANDIDATE_MANIFEST SOURCE_VIDEO ACTION_CAPTION NOOP_CAPTION CLEAN_LATENT NATIVE_NOISE BASE_MP4 BASE_RECEIPT TEACHER_RECEIPT TEACHER_RESIDUAL; do
    path_name="SIRM_${group}_${field}"; digest_name="${path_name}_SHA256"
    check_hash "${!path_name}" "${!digest_name}" "${path_name}_final"
  done
done
[[ "$(hash_file "${running_launcher}")" == "$(hash_file "${launcher}")" ]] || fail "launcher changed before publication"
[[ "$(hash_file "${loader}")" == "${loader_sha256}" ]] || fail "loader changed before publication"
[[ "$(hash_file "${closure_manifest}")" == "${closure_sha256}" ]] || fail "source closure manifest changed before publication"
verify_source_tree_closure
"${python_bin}" -B - "${checkpoint}" "${checkpoint_manifest}" "${pinned_checkpoint_tree_sha256}" "${pinned_checkpoint_manifest_sha256}" <<'PY'
from pathlib import Path
import sys
import starch_live_vjp_bridge_v1 as live_bridge

binding=live_bridge.authenticate_frozen_bernini_checkpoint_content(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    expected_checkpoint_tree_sha256=sys.argv[3],
    expected_checkpoint_content_manifest_sha256=sys.argv[4],
).receipt()
if (binding.get('checkpoint_content_verified_file_count')!=23
    or binding.get('checkpoint_content_manifest_file_sha256')!=sys.argv[4]):
    raise SystemExit('final full checkpoint closure differs')
PY
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "archive revision changed"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "final output appeared before publish"
mv -- "${staging_root}" "${output_root}"
published=true
[[ -f "${output_root}/master-receipt.json" && -d "${output_root}/dog" && -d "${output_root}/human" ]] || fail "atomic publication differs"
