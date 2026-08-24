#!/usr/bin/env bash
#SBATCH --job-name=bern-gsac-v2
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
output_video="${BERNINI_ACTION_INFERENCE_OUTPUT:?set BERNINI_ACTION_INFERENCE_OUTPUT}"
arm="${BERNINI_GUIDED_SOURCE_ALIGNED_ARM:?set BERNINI_GUIDED_SOURCE_ALIGNED_ARM}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

source_video="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/288545b9c031491a/source.mp4"
instruction="Make the dog pick up the bone and hold it in its mouth."
expected_source_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
pinned_schedule_sha256="43cd53329945280dccea5c1a1aa3b5da05337a7f10cfec0ab5a727592ea77d25"
pinned_sigma_fp32_sha256="b0d8af444d64bd51d638973de86578ad6685681521acad453c7b422f2c628dcd"
receipt_path="${output_video}.receipt.json"
artifact_paths_validated=0
artifact_committed=0
artifact_ownership_captured=0

fail() {
  echo "[bernini-gsac-v2] ERROR: $*" >&2
  exit 2
}

case "${arm}" in
  C0)
    expected_calls=0; expected_draws=0; expected_candidates=0; candidate_mode="identity"
    ;;
  O0)
    expected_calls=80; expected_draws=0; expected_candidates=0; candidate_mode="official"
    ;;
  FIID1G)
    expected_calls=160; expected_draws=40; expected_candidates=40; candidate_mode="iid1"
    ;;
  FANC1G)
    expected_calls=160; expected_draws=40; expected_candidates=40; candidate_mode="anc1"
    ;;
  FAVG5G)
    expected_calls=208; expected_draws=52; expected_candidates=52; candidate_mode="avg5"
    ;;
  FSGA5G)
    expected_calls=208; expected_draws=52; expected_candidates=52; candidate_mode="sga5"
    ;;
  *) fail "arm must be exactly C0, O0, FIID1G, FANC1G, FAVG5G, or FSGA5G" ;;
esac

validate_checkpoint_content() {
  local manifest="$1"
  local actual_count
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest is not a plain file"
  [[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${checkpoint_content_manifest_sha256}" ]] || fail "checkpoint manifest digest mismatch"
  [[ "$(wc -l < "${manifest}")" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint manifest file count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains a non-cache symlink"
  actual_count="$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)"
  [[ "${actual_count}" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint non-cache file set differs"
  (cd "${checkpoint}" && sha256sum --strict --status -c "${manifest}") || fail "checkpoint content hash differs"
}

method_tree_digest() {
  "${python_bin}" - "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
if root.is_symlink():
    raise SystemExit("method root is a symlink")
root = root.resolve(strict=True)
rows = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise SystemExit(f"method source contains symlink: {relative}")
    if stat.S_ISDIR(mode):
        if path.name == "__pycache__":
            raise SystemExit("method source contains __pycache__")
        continue
    if not stat.S_ISREG(mode):
        raise SystemExit(f"method source contains non-regular entry: {relative}")
    if path.suffix in (".pyc", ".pyo"):
        raise SystemExit(f"method source contains bytecode: {relative}")
    if mode & 0o222:
        raise SystemExit(f"method source file is writable: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append({"path": relative, "sha256": digest.hexdigest()})
payload = json.dumps(
    rows,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

pinned_source_identity() {
  PYTHONPATH="${method_root}" "${python_bin}" -B - \
    "${bernini_root}" "${veomni_root}" "${bernini_commit}" "${veomni_commit}" <<'PY'
import sys
import infer_lora as legacy

bernini = sys.argv[1]
veomni = sys.argv[2]
expected_bernini = sys.argv[3]
expected_veomni = sys.argv[4]
try:
    root, _, bernini_revision, veomni_revision = legacy.trainer.validate_source_trees(
        bernini,
        veomni,
        expected_bernini_commit=expected_bernini,
        expected_veomni_commit=expected_veomni,
    )
    inference = legacy.validate_inference_source_files(root)
except (legacy.trainer.TrainingContractError, legacy.InferenceContractError) as error:
    raise SystemExit(str(error))
value = {
    "bernini_revision": bernini_revision,
    "veomni_revision": veomni_revision,
    "bernini_training_files": dict(legacy.trainer.BERNINI_PINNED_FILE_HASHES),
    "bernini_inference_files": inference,
}
print(legacy.object_sha256(value))
PY
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid durable method source archive"
[[ "${source_archive}" == /* ]] || fail "method source archive must use a durable absolute path"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable method archive must be read-only"
durable_archive_device="$(stat -c %d -- "${source_archive}")"
durable_archive_inode="$(stat -c %i -- "${source_archive}")"
durable_archive_size="$(stat -c %s -- "${source_archive}")"
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "invalid method source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "canonical dog source is invalid"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method revision"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute .mp4 path"
output_name="${output_video##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9._-]+\.mp4$ ]] || fail "output basename contains unsupported characters"
output_dir="${output_video%/*}"
mkdir -p -- "${output_dir}"
[[ -d "${output_dir}" && ! -L "${output_dir}" ]] || fail "output directory must be a non-symlink directory"

transaction_id="slurm-${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}"
output_temporary="${output_dir}/.${output_name%.mp4}.guided-v2-tmp-${transaction_id}.mp4"
receipt_name="${receipt_path##*/}"
receipt_temporary="${output_dir}/.${receipt_name}.guided-v2-tmp-${transaction_id}"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "refusing to overwrite receipt"
[[ ! -e "${output_temporary}" && ! -L "${output_temporary}" ]] || fail "stale job-private video temporary exists"
[[ ! -e "${receipt_temporary}" && ! -L "${receipt_temporary}" ]] || fail "stale job-private receipt temporary exists"
artifact_paths_validated=1

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "canonical source hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export BERNINI_OUTPUT_TRANSACTION_ID="${transaction_id}"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-gsac-v2-${SLURM_JOB_ID}.XXXXXX")"
ownership_record="${task_scratch}/artifact-ownership.json"

capture_artifact_ownership() {
  [[ -f "${output_video}" && ! -L "${output_video}" && -f "${receipt_path}" && ! -L "${receipt_path}" ]] || return 1
  "${python_bin}" -B - \
    --receipt "${receipt_path}" \
    --video "${output_video}" \
    --token "${transaction_id}" \
    --record "${ownership_record}" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

parser = argparse.ArgumentParser()
parser.add_argument("--receipt", required=True)
parser.add_argument("--video", required=True)
parser.add_argument("--token", required=True)
parser.add_argument("--record", required=True)
args = parser.parse_args()
receipt_path = Path(args.receipt)
video_path = Path(args.video)
record_path = Path(args.record)

def opened_identity(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("artifact fd is not regular")
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            payload.extend(block)
        observed = os.stat(path, follow_symlinks=False)
        if (
            int(observed.st_dev) != int(info.st_dev)
            or int(observed.st_ino) != int(info.st_ino)
            or int(observed.st_size) != int(info.st_size)
            or not stat.S_ISREG(observed.st_mode)
        ):
            raise RuntimeError("artifact path changed while ownership was captured")
        identity = {
            "path": str(path),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "size": int(info.st_size),
            "sha256": digest.hexdigest(),
        }
        return identity, bytes(payload)
    finally:
        os.close(descriptor)

try:
    receipt_identity, receipt_bytes = opened_identity(receipt_path)
    video_identity, _ = opened_identity(video_path)
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    candidate = dict(receipt)
    stored_digest = candidate.pop("receipt_digest")
    computed_digest = hashlib.sha256(
        json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    transaction = receipt["artifact_transaction"]
    transaction_output = transaction["output"]
    output = receipt["output"]
    owned = (
        stored_digest == computed_digest
        and transaction.get("token") == args.token
        and transaction.get("receipt_path") == str(receipt_path)
        and output.get("path") == str(video_path)
        and output.get("sha256") == video_identity["sha256"]
        and transaction_output == video_identity
    )
except (OSError, KeyError, RuntimeError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
    owned = False
if not owned:
    raise SystemExit(1)
record = {
    "transaction_token": args.token,
    "video": video_identity,
    "receipt": receipt_identity,
}
encoded = json.dumps(
    record,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8") + b"\n"
descriptor = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
PY
  [[ -f "${ownership_record}" && ! -L "${ownership_record}" ]] || return 1
  artifact_ownership_captured=1
}

unlink_captured_artifact() {
  local path="$1"
  local label="$2"
  "${python_bin}" -B - \
    --path "${path}" \
    --label "${label}" \
    --record "${ownership_record}" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--label", required=True, choices=("video", "receipt"))
parser.add_argument("--record", required=True)
args = parser.parse_args()
path = Path(args.path)
record_path = Path(args.record)
try:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = record[args.label]
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        observed = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    owned = (
        expected.get("path") == str(path)
        and stat.S_ISREG(info.st_mode)
        and stat.S_ISREG(observed.st_mode)
        and int(info.st_dev) == expected.get("device")
        and int(info.st_ino) == expected.get("inode")
        and int(info.st_size) == expected.get("size")
        and int(observed.st_dev) == int(info.st_dev)
        and int(observed.st_ino) == int(info.st_ino)
        and int(observed.st_size) == int(info.st_size)
        and digest.hexdigest() == expected.get("sha256")
    )
except (OSError, KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
    owned = False
if not owned:
    raise SystemExit(1)
os.unlink(path)
PY
}

cleanup() {
  local exit_status=$?
  local scratch_prefix="${scratch_parent%/}/bernini-gsac-v2-${SLURM_JOB_ID}."
  local cleanup_failed=0
  set +e
  if [[ "${artifact_paths_validated:-0}" == 1 ]]; then
    # These names contain this exact Slurm transaction token and are therefore
    # job-private.  Never glob and never remove another job's temporary path.
    rm -f -- "${output_temporary}" "${receipt_temporary}" || cleanup_failed=1
    if [[ "${artifact_committed:-0}" != 1 ]]; then
      if [[ "${artifact_ownership_captured:-0}" == 1 ]]; then
        unlink_captured_artifact "${receipt_path}" receipt || cleanup_failed=1
        unlink_captured_artifact "${output_video}" video || cleanup_failed=1
      elif [[ -e "${output_video}" || -L "${output_video}" || -e "${receipt_path}" || -L "${receipt_path}" ]]; then
        echo "[bernini-gsac-v2] refusing to delete unowned final artifact; manual audit required" >&2
        cleanup_failed=1
      fi
    fi
  fi
  if [[ -n "${task_scratch:-}" && "${task_scratch}" == "${scratch_prefix}"* && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" 2>/dev/null || true
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  if [[ -n "${task_scratch:-}" && -e "${task_scratch}" ]]; then
    cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 && "${artifact_committed:-0}" == 1 ]]; then
    echo "[bernini-gsac-v2] PASS cleanup_verified=true output_sha256=${output_sha_round2:-unknown} receipt_file_sha256=${receipt_sha_round2:-unknown}"
  else
    echo "[bernini-gsac-v2] EXIT status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  trap - EXIT TERM INT
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/input" \
  "${task_scratch}/pycache-prefix" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache-prefix"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

staged_source_video="${task_scratch}/input/source.mp4"
cp -- "${source_video}" "${staged_source_video}"
chmod a-w -- "${staged_source_video}"
[[ -f "${staged_source_video}" && ! -L "${staged_source_video}" ]] || fail "staged source is not plain"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "staged source digest differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "original source changed during staging"

archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod a-w -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "method revision is not a commit"
git -C "${source_repository}" archive --format=tar -o "${archive_from_revision}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive is not the declared Git revision"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  guided_source_aligned_controller.py \
  infer_guided_source_aligned_controller_oracle.py \
  source_aligned_controller.py \
  infer_source_aligned_controller_oracle.py \
  tri_branch_unipc.py \
  differential_sampler.py \
  infer_lora.py \
  train_lora.py \
  tools/materialize_vae.py \
  tests/test_guided_source_aligned_controller.py \
  tests/test_infer_guided_source_aligned_controller_oracle.py \
  tests/test_infer_source_aligned_controller_oracle.py \
  tests/test_tri_branch_unipc.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "archive lacks checkpoint manifest"
find "${method_root}" -type f -exec chmod a-w -- {} +
chmod a-w -- "${archive_from_revision}"
[[ -z "$(find "${method_root}" -type d -name __pycache__ -print -quit)" ]] || fail "private method tree contains pycache"
validate_checkpoint_content "${checkpoint_manifest}"
method_tree_digest_pre="$(method_tree_digest "${method_root}")"
pinned_source_identity_pre="$(pinned_source_identity)"
tri_branch_sha256="$(sha256sum "${method_root}/tri_branch_unipc.py" | awk '{print $1}')"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-gsac-v2] revision=${source_revision} archive_path=${source_archive} archive_sha256=${source_archive_sha256}"
echo "[bernini-gsac-v2] arm=${arm} candidate_mode=${candidate_mode} expected_candidates=${expected_candidates} expected_calls=${expected_calls} expected_draws=${expected_draws}"
echo "[bernini-gsac-v2] expected_frames=81 expected_fps=25 expected_steps=40 expected_seed=2027 expected_shift=5 expected_Ulysses=4 expected_APG=4.0/0.5/50.0/0.0 expected_tau=1.0"
echo "[bernini-gsac-v2] source=${source_video} sha256=${expected_source_sha256}"
echo "[bernini-gsac-v2] target=false mask=false flow=false pose=false track=false trajectory=false first_frame_anchor=false adapter=false"
echo "[bernini-gsac-v2] method_tree_digest_pre=${method_tree_digest_pre} tri_branch_unipc_sha256=${tri_branch_sha256} pinned_source_identity_pre=${pinned_source_identity_pre}"
echo "[bernini-gsac-v2] transaction=${transaction_id} output=${output_video}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_differential_sampler_contract.py \
  test_tri_branch_unipc.py \
  test_guided_source_aligned_controller.py \
  test_infer_lora_contract.py \
  test_infer_source_aligned_controller_oracle.py \
  test_infer_guided_source_aligned_controller_oracle.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_guided_source_aligned_controller_oracle.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --source-video "${staged_source_video}" \
  --original-source-path "${source_video}" \
  --expected-source-sha256 "${expected_source_sha256}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --arm "${arm}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive "${archive_copy}" \
  --durable-method-source-archive "${source_archive}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --method-source-tree-sha256 "${method_tree_digest_pre}"

[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "output video missing"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "output receipt missing"
capture_artifact_ownership || fail "could not prove this job owns both final artifacts"
output_sha_round1="$(sha256sum "${output_video}" | awk '{print $1}')"
receipt_sha_round1="$(sha256sum "${receipt_path}" | awk '{print $1}')"

audit_receipt() {
  PYTHONPATH="${method_root}" "${python_bin}" -B - \
    --receipt "${receipt_path}" \
    --arm "${arm}" \
    --candidate-mode "${candidate_mode}" \
    --expected-calls "${expected_calls}" \
    --expected-draws "${expected_draws}" \
    --expected-candidates "${expected_candidates}" \
    --revision "${source_revision}" \
    --scratch-archive "${archive_copy}" \
    --durable-archive "${source_archive}" \
    --archive-sha256 "${source_archive_sha256}" \
    --tree-sha256 "${method_tree_digest_pre}" \
    --tri-branch-sha256 "${tri_branch_sha256}" \
    --schedule-sha256 "${pinned_schedule_sha256}" \
    --sigma-fp32-sha256 "${pinned_sigma_fp32_sha256}" \
    --source "${source_video}" \
    --staged-source "${staged_source_video}" \
    --source-sha256 "${expected_source_sha256}" \
    --output "${output_video}" \
    --output-sha256 "${output_sha_round1}" \
    --transaction "${transaction_id}" <<'PY'
import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--receipt", required=True)
parser.add_argument("--arm", required=True)
parser.add_argument("--candidate-mode", required=True)
parser.add_argument("--expected-calls", required=True, type=int)
parser.add_argument("--expected-draws", required=True, type=int)
parser.add_argument("--expected-candidates", required=True, type=int)
parser.add_argument("--revision", required=True)
parser.add_argument("--scratch-archive", required=True)
parser.add_argument("--durable-archive", required=True)
parser.add_argument("--archive-sha256", required=True)
parser.add_argument("--tree-sha256", required=True)
parser.add_argument("--tri-branch-sha256", required=True)
parser.add_argument("--schedule-sha256", required=True)
parser.add_argument("--sigma-fp32-sha256", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--staged-source", required=True)
parser.add_argument("--source-sha256", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--output-sha256", required=True)
parser.add_argument("--transaction", required=True)
args = parser.parse_args()

import infer_guided_source_aligned_controller_oracle as oracle
import infer_lora as legacy

with open(args.receipt, encoding="utf-8") as handle:
    receipt = json.load(handle)

def first(mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None

method = receipt.get("method_provenance", {})
pre = method.get("pre", {})
post = method.get("post", {})
model = receipt.get("model_provenance", {})
checkpoint = model.get("checkpoint_content", {})
registry = receipt.get("arm_registry", {})
selected = registry.get("selected", {})
weights = receipt.get("weights", {})
optimization = receipt.get("optimization", {})
inputs = receipt.get("input", {})
preprocessing = receipt.get("preprocessing", {})
prompt = receipt.get("prompt_contract", {})
sampling = receipt.get("sampling", {})
config = sampling.get("controller_config", {})
contract = sampling.get("controller_contract", {})
schedule = sampling.get("schedule_identity", {})
if schedule is None:
    schedule = {}
runtime = sampling.get("runtime_execution_certificate", {})
source_consensus = runtime.get("source_latent_consensus", {})
consensus_matching_ranks = source_consensus.get("matching_ranks")
trace_validation = runtime.get("trace_validation", {})
trace = runtime.get("trace", {})
per_rank = runtime.get("per_rank", [])
transaction = receipt.get("artifact_transaction", {})
output = receipt.get("output", {})

expected_counts = []
if args.candidate_mode in ("iid1", "anc1"):
    expected_counts = [1] * 40
elif args.candidate_mode in ("avg5", "sga5"):
    expected_counts = [5, 5, 5] + [1] * 37

sigma_values = first(schedule, "sigmas", "sigma_values")
if sigma_values is None:
    sigma_values = trace.get("sigmas")
calls = first(trace_validation, "transformer_forward_calls", "model_forward_calls", "shared_step_calls")
if calls is None:
    calls = first(runtime, "transformer_forward_calls", "model_forward_calls", "shared_step_calls")
draws = first(trace_validation, "fresh_noise_draws", "keyed_fresh_noise_draws")
if draws is None:
    draws = first(runtime, "fresh_noise_draws", "keyed_fresh_noise_draws")
selected_calls = first(selected, "expected_transformer_forward_calls", "expected_model_forward_calls", "expected_shared_step_calls")
selected_draws = first(selected, "expected_fresh_noise_draws", "expected_keyed_fresh_noise_draws")
selected_candidates = selected.get("expected_candidate_evaluations")
tri_runtime = pre.get("runtime_source_sha256", {}).get("tri_branch_apg")
tri_archive = pre.get("archive_member_sha256", {}).get("tri_branch_apg")
apg = sampling.get("apg", {})
trace_apg = dict(trace.get("apg_parameters", ()))
expected_apg = {
    "guidance_scale": 4.0,
    "eta": 0.5,
    "norm_threshold": 50.0,
    "momentum": 0.0,
}
if args.candidate_mode in ("iid1", "anc1", "avg5", "sga5"):
    expected_branch_order = [
        "target_negative",
        "target_action",
        "source_negative",
        "source_noop",
    ]
    expected_branch_counts = [args.expected_candidates] * 4
elif args.candidate_mode == "official":
    expected_branch_order = ["negative", "action"]
    expected_branch_counts = [40, 40]
else:
    expected_branch_order = None
    expected_branch_counts = None
if args.arm == "C0":
    schedule_digest_valid = not schedule
else:
    schedule_payload = {
        "timesteps": schedule.get("timesteps"),
        "sigmas": schedule.get("sigmas"),
        "flow_shift": schedule.get("flow_shift"),
        "steps": schedule.get("steps"),
    }
    schedule_digest_valid = (
        schedule.get("digest") == args.schedule_sha256
        and schedule.get("digest") == oracle.v1.object_sha256(schedule_payload)
        and schedule.get("start_sigma_claimed_exact_one") is False
        and float(schedule.get("first_anc_retained_variance", 0.0)) > 0.0
        and schedule.get("first_anc_predecessor_policy")
        == "zero_initialized_per_dynaedit_pseudocode"
        and schedule.get("scheduler_sigma_dtype") == "torch.float32"
        and schedule.get("scheduler_sigma_device") == "cpu"
        and schedule.get("scheduler_sigma_direct_views") is True
        and schedule.get("scheduler_sigma_fp32_digest")
        == args.sigma_fp32_sha256
        and trace.get("schedule_digest") == schedule.get("digest")
        and trace.get("scheduler_sigma_fp32_digest")
        == args.sigma_fp32_sha256
    )
noise_bank_contract = oracle.guided.noise_bank_pairing_contract(seed=2027)
guided_mode = args.candidate_mode in ("iid1", "anc1", "avg5", "sga5")
expected_used_noise_digest = (
    oracle.guided.used_noise_key_digest(
        oracle.guided.GuidedSourceAlignedConfig(arm=args.arm)
    )
    if guided_mode
    else None
)

checks = (
    receipt.get("schema_version") == "bernini-r-1p3b-guided-source-aligned-controller-oracle-v2",
    receipt.get("official_dynaedit_reproduction_claimed") is False,
    method.get("pre_post_exact") is True,
    pre == post,
    pre.get("revision") == args.revision,
    pre.get("scratch_archive_path") == args.scratch_archive,
    pre.get("durable_archive_path") == args.durable_archive,
    method.get("durable_archive_path") == args.durable_archive,
    pre.get("archive_sha256") == args.archive_sha256,
    pre.get("runtime_tree", {}).get("tree_sha256") == args.tree_sha256,
    pre.get("runtime_tree", {}).get("all_plain_read_only") is True,
    pre.get("runtime_tree", {}).get("bytecode_absent") is True,
    tri_runtime == args.tri_branch_sha256,
    tri_archive == args.tri_branch_sha256,
    model.get("bernini_commit") == "2d2b4591ac053ec25c6371b01a5a6746679e5793",
    model.get("veomni_commit") == "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
    model.get("checkpoint_tree_sha256") == "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
    checkpoint.get("manifest_sha256_computed") == "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    checkpoint.get("every_file_sha256_verified") is True,
    registry.get("allowed_arms") == ["C0", "O0", "FIID1G", "FANC1G", "FAVG5G", "FSGA5G"],
    registry.get("arm_table_sha256") == oracle.ARM_TABLE_SHA256,
    registry.get("method_cli_controls") == ["arm"],
    registry.get("arbitrary_hyperparameter_cli_supported") is False,
    selected.get("arm") == args.arm,
    selected_calls == args.expected_calls,
    selected_draws == args.expected_draws,
    selected_candidates == args.expected_candidates,
    weights.get("base_checkpoint_loaded") is True,
    weights.get("base_frozen") is True,
    weights.get("adapter_loaded") is False,
    optimization == {"zero_training": True, "training_steps": 0, "optimizer_constructed": False, "backward_calls": 0},
    inputs.get("accepted_external_conditions") == ["source_video", "edit_instruction"],
    inputs.get("source_video_path") == args.source,
    inputs.get("staged_source_video_path") == args.staged_source,
    inputs.get("source_video_sha256") == args.source_sha256,
    inputs.get("target_video_argument") is False,
    inputs.get("target_video_accessed") is False,
    all(inputs.get(name) is False for name in ("mask", "optical_flow", "pose", "track", "trajectory", "swept_tube", "first_frame_anchor")),
    preprocessing.get("frame_count") == 81,
    math.isclose(float(preprocessing.get("fps", -1)), 25.0, rel_tol=0, abs_tol=1e-9),
    preprocessing.get("source_derived_bucket_hw") == [496, 480],
    prompt.get("action_prompt_uses_official_mv2v_training_path") is True,
    prompt.get("noop_is_internal_fixed_semantic_instruction") is True,
    prompt.get("negative_prompt_utf8_sha256")
    == hashlib.sha256(legacy.DEFAULT_NEGATIVE_PROMPT.encode("utf-8")).hexdigest(),
    prompt.get("negative_prompt_is_verbatim_pinned_bernini_default") is True,
    prompt.get("negative_prompt_cleaner_applied") is False,
    isinstance(prompt.get("negative_prompt_embeddings"), dict),
    isinstance(prompt.get("negative_prompt_embeddings", {}).get("content_sha256"), str)
    and len(prompt["negative_prompt_embeddings"]["content_sha256"]) == 64,
    sampling.get("num_frames") == 81,
    math.isclose(float(sampling.get("fps", -1)), 25.0, rel_tol=0, abs_tol=1e-9),
    sampling.get("num_inference_steps") == 40,
    sampling.get("seed") == 2027,
    math.isclose(float(sampling.get("flow_shift", -1)), 5.0, rel_tol=0, abs_tol=1e-12),
    sampling.get("ulysses_size") == 4,
    sampling.get("guidance_mode") == "v2v_apg",
    apg == expected_apg,
    (config is None)
    if args.candidate_mode in ("identity", "official")
    else config.get("arm") == args.arm
    and config.get("num_inference_steps") == 40
    and config.get("seed") == 2027
    and math.isclose(float(config.get("flow_shift", -1)), 5.0, rel_tol=0, abs_tol=0)
    and math.isclose(float(config.get("motion_scale", -1)), 1.0, rel_tol=0, abs_tol=0)
    and math.isclose(float(config.get("sga_temperature", -1)), 1.0, rel_tol=0, abs_tol=0)
    and math.isclose(float(config.get("anc_lock_sigma", -1)), 0.25, rel_tol=0, abs_tol=0)
    and math.isclose(float(config.get("apg_guidance_scale", -1)), 4.0, rel_tol=0, abs_tol=0)
    and math.isclose(float(config.get("apg_eta", -1)), 0.5, rel_tol=0, abs_tol=0)
    and math.isclose(float(config.get("apg_norm_threshold", -1)), 50.0, rel_tol=0, abs_tol=0)
    and math.isclose(float(config.get("apg_momentum", -1)), 0.0, rel_tol=0, abs_tol=0),
    contract.get("user_inputs") == ["source_video", "edit_instruction"],
    schedule_digest_valid,
    (sigma_values is None) if args.arm == "C0" else isinstance(sigma_values, list) and len(sigma_values) == 41,
    (sigma_values is None) if args.arm == "C0" else math.isclose(float(sigma_values[0]), 0.9999989867210388, rel_tol=0, abs_tol=2e-7),
    (sigma_values is None) if args.arm == "C0" else math.isclose(float(sigma_values[-1]), 0.0, rel_tol=0, abs_tol=1e-12),
    runtime.get("validated") is True,
    runtime.get("all_four_ranks_exact") is True,
    source_consensus.get("policy") == "pinned_three_of_four_then_broadcast",
    source_consensus.get("required_matching_ranks") == 3,
    source_consensus.get("pinned_content_sha256")
    == oracle.EXPECTED_SOURCE_LATENT_CONTENT_SHA256,
    source_consensus.get("pinned_raw_storage_sha256")
    == oracle.EXPECTED_SOURCE_LATENT_RAW_STORAGE_SHA256,
    isinstance(consensus_matching_ranks, list)
    and len(consensus_matching_ranks) >= 3,
    source_consensus.get("owner_rank") == min(consensus_matching_ranks)
    if isinstance(consensus_matching_ranks, list) and consensus_matching_ranks
    else False,
    source_consensus.get("all_rank_post_broadcast_exact") is True,
    isinstance(source_consensus.get("pre_broadcast_per_rank"), list)
    and len(source_consensus["pre_broadcast_per_rank"]) == 4,
    isinstance(source_consensus.get("post_broadcast_per_rank"), list)
    and len(source_consensus["post_broadcast_per_rank"]) == 4,
    trace_validation.get("validated") is True,
    calls == args.expected_calls,
    draws == args.expected_draws,
    (trace_validation.get("effective_candidate_counts") == expected_counts)
    if guided_mode
    else "effective_candidate_counts" not in trace_validation,
    (trace.get("branch_order") == expected_branch_order)
    if expected_branch_order is not None
    else "branch_order" not in trace,
    (trace.get("branch_counts") == expected_branch_counts)
    if expected_branch_counts is not None
    else "branch_counts" not in trace,
    (trace_apg == {"guidance_mode": "v2v_apg", **expected_apg})
    if guided_mode
    else not trace_apg,
    (trace.get("candidate_continuation") == "candidate_0")
    if guided_mode
    else "candidate_continuation" not in trace,
    (trace.get("weighted_noise_collapse_used") is False)
    if guided_mode
    else "weighted_noise_collapse_used" not in trace,
    (
        trace.get("target_branch_query_parity") is True
        and trace.get("source_branch_query_parity") is True
        and trace.get("raw_velocity_dtype") == "torch.bfloat16"
        and trace.get("guided_velocity_dtype") == "torch.float32"
        and trace.get("apg_clean_dtype") == "torch.float32"
        and trace.get("delta_dtype") == "torch.float32"
        and trace.get("edit_state_dtype") == "torch.float32"
        and trace.get("anc_initial_predecessor_policy")
        == "zero_initialized_per_dynaedit_pseudocode"
        and trace.get("full_noise_bank_digest")
        == noise_bank_contract["full_bank_digest"]
        and trace.get("candidate0_noise_bank_digest")
        == noise_bank_contract["candidate0_bank_digest"]
        and trace.get("used_noise_key_digest") == expected_used_noise_digest
        and all(
            isinstance(trace.get(name), str) and len(trace[name]) == 64
            for name in (
                "used_fresh_noise_content_digest",
                "candidate0_fresh_noise_content_digest",
            )
        )
        and all(
            isinstance(trace.get(name), list)
            and len(trace[name]) == 40
            and all(math.isfinite(float(value)) for value in trace[name])
            for name in ("sga_entropy", "sga_top1_margin")
        )
    )
    if guided_mode
    else all(
        name not in trace
        for name in (
            "target_branch_query_parity",
            "source_branch_query_parity",
            "raw_velocity_dtype",
            "guided_velocity_dtype",
            "apg_clean_dtype",
            "delta_dtype",
            "edit_state_dtype",
            "anc_initial_predecessor_policy",
            "used_fresh_noise_content_digest",
            "candidate0_fresh_noise_content_digest",
        )
    ),
    all(
        isinstance(trace.get(name), str) and len(trace[name]) == 64
        for name in ("used_noise_key_digest", "full_noise_bank_digest", "candidate0_noise_bank_digest")
    )
    if guided_mode
    else all(name not in trace for name in ("used_noise_key_digest", "full_noise_bank_digest", "candidate0_noise_bank_digest")),
    len(per_rank) == 4,
    sorted(row.get("rank") for row in per_rank) == [0, 1, 2, 3],
    all(row.get("arm") == args.arm for row in per_rank),
    transaction.get("token") == args.transaction,
    transaction.get("output", {}).get("device") == Path(args.output).stat().st_dev,
    transaction.get("output", {}).get("inode") == Path(args.output).stat().st_ino,
    transaction.get("output", {}).get("size") == Path(args.output).stat().st_size,
    transaction.get("output", {}).get("sha256") == args.output_sha256,
    transaction.get("receipt_path") == args.receipt,
    output.get("path") == args.output,
    output.get("sha256") == args.output_sha256,
    output.get("device") == Path(args.output).stat().st_dev,
    output.get("inode") == Path(args.output).stat().st_ino,
    output.get("size") == Path(args.output).stat().st_size,
    output.get("frame_count") == 81,
    output.get("fps") == 25.0,
    output.get("height") == 496,
    output.get("width") == 480,
    receipt.get("production_claim_forbidden") is True,
    receipt.get("scientific_claim_authorized") is False,
)
candidate = dict(receipt)
stored = candidate.pop("receipt_digest", None)
computed = hashlib.sha256(
    json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
checks += (stored == computed,)
raise SystemExit(0 if all(checks) else 1)
PY
}

audit_receipt || fail "receipt audit round 1 failed"
[[ "$(sha256sum "${output_video}" | awk '{print $1}')" == "${output_sha_round1}" ]] || fail "output changed after audit round 1"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt changed after audit round 1"
output_sha_round2="$(sha256sum "${output_video}" | awk '{print $1}')"
receipt_sha_round2="$(sha256sum "${receipt_path}" | awk '{print $1}')"
[[ "${output_sha_round2}" == "${output_sha_round1}" ]] || fail "output hash closure differs"
[[ "${receipt_sha_round2}" == "${receipt_sha_round1}" ]] || fail "receipt hash closure differs"
audit_receipt || fail "receipt audit round 2 failed"
[[ "$(sha256sum "${output_video}" | awk '{print $1}')" == "${output_sha_round2}" ]] || fail "output changed after audit round 2"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round2}" ]] || fail "receipt changed after audit round 2"

[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "canonical source changed"
[[ "$(sha256sum "${staged_source_video}" | awk '{print $1}')" == "${expected_source_sha256}" ]] || fail "staged source changed"
validate_checkpoint_content "${checkpoint_manifest}"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit changed"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source changed"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "durable method archive was replaced"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable method archive became writable"
[[ "$(stat -c %d -- "${source_archive}")" == "${durable_archive_device}" ]] || fail "durable method archive device changed"
[[ "$(stat -c %i -- "${source_archive}")" == "${durable_archive_inode}" ]] || fail "durable method archive inode changed"
[[ "$(stat -c %s -- "${source_archive}")" == "${durable_archive_size}" ]] || fail "durable method archive size changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable method archive changed"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive changed"
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "revision archive changed"
archive_after_revision="${task_scratch}/method-after-revision.tar"
git -C "${source_repository}" archive --format=tar -o "${archive_after_revision}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_after_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method revision/archive binding changed"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive changed"
method_tree_digest_post="$(method_tree_digest "${method_root}")"
[[ "${method_tree_digest_post}" == "${method_tree_digest_pre}" ]] || fail "executed method tree changed"
[[ "$(sha256sum "${method_root}/tri_branch_unipc.py" | awk '{print $1}')" == "${tri_branch_sha256}" ]] || fail "tri_branch_unipc changed"
pinned_source_identity_post="$(pinned_source_identity)"
[[ "${pinned_source_identity_post}" == "${pinned_source_identity_pre}" ]] || fail "Bernini/VeOmni source identity changed"
[[ -z "$(find "${task_scratch}/pycache-prefix" -mindepth 1 -print -quit)" ]] || fail "private pycache prefix is not empty"
[[ ! -e "${output_temporary}" && ! -L "${output_temporary}" ]] || fail "video temporary remains"
[[ ! -e "${receipt_temporary}" && ! -L "${receipt_temporary}" ]] || fail "receipt temporary remains"

echo "[bernini-gsac-v2] method_tree_digest_post=${method_tree_digest_post} pinned_source_identity_post=${pinned_source_identity_post}"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
artifact_committed=1
