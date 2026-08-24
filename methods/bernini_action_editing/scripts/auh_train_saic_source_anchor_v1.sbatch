#!/usr/bin/bash
#SBATCH --job-name=saic-anchor-f32-r1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-289,auh7-1b-gpu-318

# Formal32 Stage-A wrapper.  The submitter supplies an empty user environment.
# Stage 0 opens every file input with O_NOFOLLOW and re-execs this spooled
# wrapper while retaining those descriptors.  Stage 1 consumes only retained
# descriptors, extracts a private runtime tree, and launches one WORLD8 job.

set -Eeuo pipefail
umask 077
export PATH=/usr/bin:/bin
export LANG=C
export LC_ALL=C

readonly root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809
readonly stem=saic-source-anchor-formal32-retfd-20260813-r1
readonly release="${root}/releases/${stem}"
readonly inputs="${release}/inputs"
readonly expected_archive="${inputs}/videoedit-saic-20c2193-methods.tar"
readonly expected_trainer="${inputs}/train_saic_source_anchor_v1.py"
readonly expected_manifest="${inputs}/source-anchor-v1-2a52753.json"
readonly expected_checkpoint_manifest="${inputs}/bernini-r13-ff4c5d4-checkpoint.sha256"
readonly expected_full60_admission="${inputs}/formal-full60-admission.json"
readonly expected_release_manifest="${release}/release-manifest.json"
readonly expected_output_parent="${root}/runs/${stem}"
readonly expected_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly expected_bernini=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly expected_veomni=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly expected_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly archive_sha=3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b
readonly archive_revision=20c2193954e780e9654347754b1485f3492fbea5
readonly archive_member_manifest_sha=1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc
readonly archive_runtime_sha=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36
readonly runtime_origin_manifest_sha=2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba
readonly trainer_sha=28059c7a1ea5d641b35a04e929fa454531ee6df8444a01cceefb3f3a63622a1c
readonly manifest_sha=b56e9dca9085ba2f7b67acee04f29f7e45afd73a71b92172df5b19095fe33c4c
readonly manifest_digest=e3978e872c80c20b3bf8c974cd7324b62f54d58ca162e1760f548a02a2538d8d
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly python_version_sha=55ae85cf4bdb38743edbcd53ea68ff36511997ec6c21b1e83d8bebc939bf056b
readonly full60_admission_sha=__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_SHA256__
readonly full60_admission_digest=__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_DIGEST__
readonly full60_admission_schema=__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_SCHEMA__
readonly full60_admission_status=__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_STATUS__

fail() { echo "[saic-anchor-f32-r1] ERROR: $*" >&2; exit 2; }

[[ "${trainer_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "trainer pin is unresolved"
[[ "${full60_admission_sha}" =~ ^[0-9a-f]{64}$ && \
   "${full60_admission_digest}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "formal full60 terminal admission pins are unresolved"
[[ "${full60_admission_schema}" != __*__ && \
   "${full60_admission_status}" != __*__ ]] || \
  fail "formal full60 terminal admission semantics are unresolved"

archive="${SAIC_ANCHOR_ARCHIVE:?}"
trainer="${SAIC_ANCHOR_TRAINER:?}"
source_manifest="${SAIC_ANCHOR_SOURCE_MANIFEST:?}"
checkpoint_manifest="${SAIC_ANCHOR_CHECKPOINT_MANIFEST:?}"
full60_admission="${SAIC_ANCHOR_FULL60_ADMISSION:?}"
release_manifest="${SAIC_ANCHOR_RELEASE_MANIFEST:?}"
submission_receipt="${SAIC_ANCHOR_SUBMISSION_RECEIPT:?}"
output_parent="${SAIC_ANCHOR_OUTPUT_PARENT:?}"
release_manifest_sha="${SAIC_ANCHOR_RELEASE_MANIFEST_SHA256:?}"
release_manifest_digest="${SAIC_ANCHOR_RELEASE_MANIFEST_DIGEST:?}"
wrapper_sha="${SAIC_ANCHOR_WRAPPER_SHA256:?}"

[[ "${archive}" == "${expected_archive}" && \
   "${trainer}" == "${expected_trainer}" && \
   "${source_manifest}" == "${expected_manifest}" && \
   "${checkpoint_manifest}" == "${expected_checkpoint_manifest}" && \
   "${full60_admission}" == "${expected_full60_admission}" && \
   "${release_manifest}" == "${expected_release_manifest}" && \
   "${output_parent}" == "${expected_output_parent}" && \
   "${submission_receipt}" == "${expected_output_parent}/submission-receipt.json" ]] || \
  fail "exported path contract differs"
for value in release_manifest_sha release_manifest_digest wrapper_sha; do
  [[ "${!value}" =~ ^[0-9a-f]{64}$ ]] || fail "${value} differs"
done
[[ "${SLURM_JOB_ID:?}" =~ ^[1-9][0-9]*$ ]] || fail "Slurm job ID differs"
[[ "$(realpath -e -- "${expected_python}")" == "${expected_python}" && \
   -f "${expected_python}" && ! -L "${expected_python}" && \
   -x "${expected_python}" && \
   "$(stat -c '%h' -- "${expected_python}")" == 1 && \
   "$(sha256sum "${expected_python}" | awk '{print $1}')" == "${python_sha}" && \
   "$("${expected_python}" --version 2>&1 | sha256sum | awk '{print $1}')" == \
   "${python_version_sha}" ]] || fail "bootstrap Python identity/version differs"

if [[ "${SAIC_ANCHOR_REEXEC_STAGE+x}" != x ]]; then
  exec "${expected_python}" -B - \
    "$0" "${wrapper_sha}" "${archive}" "${archive_sha}" \
    "${trainer}" "${trainer_sha}" "${source_manifest}" "${manifest_sha}" \
    "${checkpoint_manifest}" "${checkpoint_manifest_sha}" \
    "${full60_admission}" "${full60_admission_sha}" \
    "${release_manifest}" "${release_manifest_sha}" \
    "${submission_receipt}" "${output_parent}" <<'PY'
import hashlib, json, os, resource, stat, sys
from pathlib import Path

args = sys.argv[1:]
spool = Path(args[0]); spool_sha = args[1]
pairs = [(Path(args[index]), args[index + 1]) for index in range(2, 14, 2)]
receipt = Path(args[14]); output = Path(args[15])
fds = []

def digest_fd(fd):
    h = hashlib.sha256(); os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk: break
        h.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return h.hexdigest()

def retain(path, expected, label, *, sealed=True):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW); fds.append(fd)
    before = os.fstat(fd); leaf = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode) or before.st_nlink != 1
            or leaf.st_nlink != 1 or (before.st_dev, before.st_ino)
            != (leaf.st_dev, leaf.st_ino) or (sealed and
            before.st_uid != os.getuid()) or (sealed and
            stat.S_IMODE(before.st_mode) != 0o444) or (not sealed and
            stat.S_IMODE(before.st_mode) & 0o022) or digest_fd(fd) != expected):
        raise SystemExit(label + " retained identity differs")
    after = os.fstat(fd); leaf_after = path.lstat()
    key = lambda value: (value.st_dev, value.st_ino, value.st_size,
                         value.st_mtime_ns, value.st_nlink,
                         stat.S_IMODE(value.st_mode))
    if key(before) != key(after) or key(after) != key(leaf_after):
        raise SystemExit(label + " changed during retained read")
    os.set_inheritable(fd, True)
    return fd

spool_fd = retain(spool, spool_sha, "spooled wrapper", sealed=False)
source_fds = [retain(path, expected, path.name) for path, expected in pairs]
manifest_fd = source_fds[2]
os.lseek(manifest_fd, 0, os.SEEK_SET)
manifest_blocks = []
while True:
    block = os.read(manifest_fd, 1024 * 1024)
    if not block:
        break
    manifest_blocks.append(block)
manifest_raw = b"".join(manifest_blocks)
os.lseek(manifest_fd, 0, os.SEEK_SET)

def no_duplicates(items):
    result = {}
    for key, value in items:
        if key in result:
            raise SystemExit("source manifest contains duplicate key")
        result[key] = value
    return result

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

try:
    manifest_value = json.loads(
        manifest_raw.decode("ascii"), object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-finite manifest value")),
    )
except (UnicodeError, ValueError) as error:
    raise SystemExit("source manifest strict decode differs") from error
root_fields = {
    "schema_version", "optimizer_authorized", "source_root",
    "selection_seed", "frame_count", "fps", "train_count",
    "holdout_count", "selected_bucket_counts", "eligible_bucket_counts",
    "strict_action_iids_excluded", "wrong_source_policy",
    "holdout_used_by_optimizer", "train_rows", "holdout_rows",
    "input_closure", "manifest_digest",
}
row_fields = {
    "schema_version", "split", "row_index", "dp_arm", "iid",
    "source_video_path", "source_video_sha256", "wrong_iid",
    "wrong_source_video_path", "wrong_source_video_sha256", "frame_count",
    "fps", "reported_fps", "bucket_hw", "scramble_seed", "row_digest",
}
if not isinstance(manifest_value, dict) or set(manifest_value) != root_fields:
    raise SystemExit("source manifest root fields differ")
if manifest_raw != canonical(manifest_value) + b"\n":
    raise SystemExit("source manifest canonical bytes differ")
unsigned_manifest = dict(manifest_value)
manifest_digest_value = unsigned_manifest.pop("manifest_digest", None)
if (manifest_value.get("schema_version")
        != "bernini-saic-source-anchor-manifest-v1"
        or manifest_value.get("optimizer_authorized") is not False
        or manifest_value.get("train_count") != 64
        or manifest_value.get("holdout_count") != 16
        or manifest_value.get("holdout_used_by_optimizer") is not False
        or manifest_digest_value
        != "e3978e872c80c20b3bf8c974cd7324b62f54d58ca162e1760f548a02a2538d8d"
        or hashlib.sha256(canonical(unsigned_manifest)).hexdigest()
        != manifest_digest_value):
    raise SystemExit("source manifest canonical seal differs")
source_pairs = {}
for split, expected_count in (("train", 64), ("holdout", 16)):
    rows = manifest_value.get(split + "_rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise SystemExit("source manifest row count differs")
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != row_fields:
            raise SystemExit("source manifest row fields differ")
        unsigned_row = dict(row); row_digest = unsigned_row.pop("row_digest", None)
        if (row.get("schema_version")
                != "bernini-saic-source-anchor-manifest-v1"
                or row.get("split") != split
                or row.get("row_index") != expected_index
                or not isinstance(row_digest, str)
                or hashlib.sha256(canonical(unsigned_row)).hexdigest()
                != row_digest):
            raise SystemExit("source manifest row seal differs")
        for path_key, sha_key in (
            ("source_video_path", "source_video_sha256"),
            ("wrong_source_video_path", "wrong_source_video_sha256"),
        ):
            path_value = row.get(path_key); sha_value = row.get(sha_key)
            if (not isinstance(path_value, str) or not path_value.startswith("/")
                    or "\x00" in path_value or not isinstance(sha_value, str)
                    or len(sha_value) != 64
                    or any(character not in "0123456789abcdef"
                           for character in sha_value)):
                raise SystemExit("source manifest path/SHA differs")
            previous = source_pairs.setdefault(path_value, sha_value)
            if previous != sha_value:
                raise SystemExit("source manifest path has multiple hashes")
if len(source_pairs) != 80:
    raise SystemExit("source manifest exact80 path closure differs")
soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft_limit < 512:
    raise SystemExit("RLIMIT_NOFILE is too small for exact80 retained sources")
source_video_fds = []
source_map_rows = []
for path_string, expected in sorted(source_pairs.items()):
    path = Path(path_string)
    if path.resolve(strict=True) != path:
        raise SystemExit("source video canonical path differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    fds.append(descriptor); source_video_fds.append(descriptor)
    before = os.fstat(descriptor); leaf = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode) or before.st_uid != os.getuid()
            or leaf.st_uid != os.getuid() or before.st_nlink != 1
            or leaf.st_nlink != 1 or stat.S_IMODE(before.st_mode) & 0o022
            or stat.S_IMODE(leaf.st_mode) & 0o022
            or not stat.S_IMODE(before.st_mode) & 0o400
            or not stat.S_IMODE(leaf.st_mode) & 0o400
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
            or digest_fd(descriptor) != expected):
        raise SystemExit("source video retained identity differs")
    after = os.fstat(descriptor); leaf_after = path.lstat()
    key = lambda value: (value.st_dev, value.st_ino, value.st_size,
                         value.st_mtime_ns, value.st_uid, value.st_nlink,
                         stat.S_IMODE(value.st_mode))
    if key(before) != key(after) or key(after) != key(leaf_after):
        raise SystemExit("source video changed during retained read")
    os.set_inheritable(descriptor, True)
    source_map_rows.append({
        "declared_path": path_string, "sha256": expected,
        "descriptor": descriptor, "device": before.st_dev,
        "inode": before.st_ino, "size": before.st_size,
        "mtime_ns": before.st_mtime_ns, "uid": before.st_uid,
        "mode": stat.S_IMODE(before.st_mode),
    })
receipt_fd = os.open(receipt, os.O_RDONLY | os.O_NOFOLLOW); fds.append(receipt_fd)
receipt_info = os.fstat(receipt_fd); receipt_leaf = receipt.lstat()
if (not stat.S_ISREG(receipt_info.st_mode) or stat.S_ISLNK(receipt_leaf.st_mode)
        or receipt_info.st_nlink != 1 or receipt_leaf.st_nlink != 1
        or receipt_info.st_uid != os.getuid()
        or stat.S_IMODE(receipt_info.st_mode) not in {0o600, 0o444}
        or (receipt_info.st_dev, receipt_info.st_ino)
        != (receipt_leaf.st_dev, receipt_leaf.st_ino)):
    raise SystemExit("submission receipt retained identity differs")
os.set_inheritable(receipt_fd, True)
output_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
fds.append(output_fd); output_info = os.fstat(output_fd); output_leaf = output.lstat()
if (not stat.S_ISDIR(output_info.st_mode) or stat.S_ISLNK(output_leaf.st_mode)
        or output_info.st_uid != os.getuid()
        or stat.S_IMODE(output_info.st_mode) != 0o700
        or (output_info.st_dev, output_info.st_ino)
        != (output_leaf.st_dev, output_leaf.st_ino)):
    raise SystemExit("output parent retained identity differs")
os.set_inheritable(output_fd, True)
if len(set([spool_fd, *source_fds, *source_video_fds,
            receipt_fd, output_fd])) != 89:
    raise SystemExit("retained descriptor closure differs")
source_map_core = {
    "schema_version": "saic-source-anchor-retained-source-fd-map-v1",
    "supervisor_pid": os.getpid(), "source_count": 80,
    "rows": source_map_rows,
    "source_descriptors_held_by_parent_supervisor": True,
    "workers_open_only_parent_proc_fd_paths": True,
}
source_map_value = {
    **source_map_core,
    "map_digest": hashlib.sha256(canonical(source_map_core)).hexdigest(),
}
source_map_raw = canonical(source_map_value)
env = {
    "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
    "SAIC_ANCHOR_REEXEC_STAGE": "retained-stage1",
    "SAIC_ANCHOR_SPOOL_FD": str(spool_fd),
    "SAIC_ANCHOR_ARCHIVE_FD": str(source_fds[0]),
    "SAIC_ANCHOR_TRAINER_FD": str(source_fds[1]),
    "SAIC_ANCHOR_SOURCE_MANIFEST_FD": str(source_fds[2]),
    "SAIC_ANCHOR_CHECKPOINT_MANIFEST_FD": str(source_fds[3]),
    "SAIC_ANCHOR_FULL60_ADMISSION_FD": str(source_fds[4]),
    "SAIC_ANCHOR_RELEASE_MANIFEST_FD": str(source_fds[5]),
    "SAIC_ANCHOR_SUBMISSION_RECEIPT_FD": str(receipt_fd),
    "SAIC_ANCHOR_OUTPUT_PARENT_FD": str(output_fd),
    "SAIC_ANCHOR_SOURCE_FD_MAP": source_map_raw.decode("ascii"),
    "SAIC_ANCHOR_SOURCE_FD_MAP_SHA256": hashlib.sha256(source_map_raw).hexdigest(),
    "SAIC_ANCHOR_SOURCE_SUPERVISOR_PID": str(os.getpid()),
}
for name in (
    "SAIC_ANCHOR_ARCHIVE", "SAIC_ANCHOR_TRAINER",
    "SAIC_ANCHOR_SOURCE_MANIFEST", "SAIC_ANCHOR_CHECKPOINT_MANIFEST",
    "SAIC_ANCHOR_FULL60_ADMISSION", "SAIC_ANCHOR_RELEASE_MANIFEST",
    "SAIC_ANCHOR_SUBMISSION_RECEIPT", "SAIC_ANCHOR_OUTPUT_PARENT",
    "SAIC_ANCHOR_RELEASE_MANIFEST_SHA256",
    "SAIC_ANCHOR_RELEASE_MANIFEST_DIGEST", "SAIC_ANCHOR_WRAPPER_SHA256",
):
    env[name] = os.environ[name]
for name in (
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_TMPDIR",
    "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
):
    if name in os.environ:
        env[name] = os.environ[name]
os.execve("/usr/bin/bash", ["/usr/bin/bash", f"/proc/self/fd/{spool_fd}"], env)
PY
fi

[[ "${SAIC_ANCHOR_SOURCE_SUPERVISOR_PID:?}" == "$$" && \
   "${SAIC_ANCHOR_SOURCE_FD_MAP_SHA256:?}" =~ ^[0-9a-f]{64}$ && \
   -n "${SAIC_ANCHOR_SOURCE_FD_MAP:?}" ]] || \
  fail "exact80 retained source supervisor contract differs"

for fd_name in SPOOL ARCHIVE TRAINER SOURCE_MANIFEST CHECKPOINT_MANIFEST \
  FULL60_ADMISSION RELEASE_MANIFEST SUBMISSION_RECEIPT OUTPUT_PARENT; do
  fd_var="SAIC_ANCHOR_${fd_name}_FD"
  [[ "${!fd_var}" =~ ^([3-9]|[1-9][0-9]+)$ ]] || fail "${fd_name} FD differs"
  [[ -e "/proc/$$/fd/${!fd_var}" ]] || fail "${fd_name} FD is absent"
done

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-anchor-f32-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-anchor-f32-${SLURM_JOB_ID}."*)
      chmod -R u+w -- "${task_scratch}" 2>/dev/null || true
      rm -rf -- "${task_scratch}" 2>/dev/null || status=2
      ;;
    *) status=2 ;;
  esac
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/input" \
  "${task_scratch}/cache/miopen-user" "${task_scratch}/cache/miopen-custom" \
  "${task_scratch}/cache/torch-extensions" "${task_scratch}/cache/triton" \
  "${task_scratch}/cache/pycache" "${task_scratch}/tmp"
archive_proof="${task_scratch}/input/archive-proof.json"
"${expected_python}" -I -B - \
  "/proc/$$/fd/${SAIC_ANCHOR_ARCHIVE_FD}" "${archive_sha}" \
  "${task_scratch}/source-tree" "${archive_member_manifest_sha}" \
  "${archive_runtime_sha}" "${archive_proof}" <<'PY'
import hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath

archive_path = sys.argv[1]; expected_archive_sha = sys.argv[2]
destination = Path(sys.argv[3]); expected_manifest_sha = sys.argv[4]
expected_runtime_sha = sys.argv[5]; proof_path = Path(sys.argv[6])
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
descriptor = os.open(archive_path, os.O_RDONLY)
try:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(chunk)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or digest.hexdigest() != expected_archive_sha):
        raise SystemExit("retained source archive identity differs")
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as source, \
            tarfile.open(fileobj=source, mode="r:") as bundle:
        if bundle.pax_headers.get("comment") \
                != "20c2193954e780e9654347754b1485f3492fbea5":
            raise SystemExit("source archive revision differs")
        members = bundle.getmembers()
        if len(members) != 864:
            raise SystemExit("source archive member count differs")
        names = set(); member_types = {}; rows = []
        for member in members:
            relative = PurePosixPath(member.name)
            if (not member.name or member.name == "." or "\\" in member.name
                    or "\x00" in member.name or relative.is_absolute()
                    or relative.as_posix() != member.name
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or not relative.parts or relative.parts[0] != "methods"
                    or member.name in names or member.issym() or member.islnk()
                    or member.isdev() or member.isfifo()
                    or not (member.isdir() or member.isfile())):
                raise SystemExit("source archive member differs")
            names.add(member.name)
            kind = "directory" if member.isdir() else "file"
            member_types[member.name] = kind
            row = {"name": member.name, "type": kind, "mode": member.mode,
                   "size": member.size}
            if member.isfile():
                handle = bundle.extractfile(member)
                if handle is None:
                    raise SystemExit("source archive member is unreadable")
                with handle:
                    raw = handle.read()
                if len(raw) != member.size:
                    raise SystemExit("source archive member size differs")
                row["sha256"] = hashlib.sha256(raw).hexdigest()
            rows.append(row)
        for name, kind in member_types.items():
            parts = PurePosixPath(name).parts
            if any(member_types.get("/".join(parts[:index])) != "directory"
                   for index in range(1, len(parts))):
                raise SystemExit("source archive ancestor differs")
            if kind == "file" and any(other.startswith(name + "/") for other in names):
                raise SystemExit("source archive file/directory conflict")
        rows_raw = canonical(rows)
        file_count = sum(row["type"] == "file" for row in rows)
        directory_count = sum(row["type"] == "directory" for row in rows)
        if (hashlib.sha256(rows_raw).hexdigest() != expected_manifest_sha
                or len(rows_raw) != 173053 or file_count != 853
                or directory_count != 11):
            raise SystemExit("source archive canonical manifest differs")
        for member, row in zip(members, rows):
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.relative_to(destination)
            if member.isdir():
                target.mkdir(mode=0o700, exist_ok=False)
                continue
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source_member = bundle.extractfile(member)
            if source_member is None:
                raise SystemExit("source archive member is unreadable")
            output = os.open(target, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                             os.O_NOFOLLOW, 0o400)
            try:
                member_digest = hashlib.sha256(); size = 0
                with source_member:
                    for chunk in iter(lambda: source_member.read(1024 * 1024), b""):
                        view = memoryview(chunk)
                        while view:
                            wrote = os.write(output, view)
                            if wrote <= 0:
                                raise SystemExit("source archive extraction stalled")
                            view = view[wrote:]
                        member_digest.update(chunk); size += len(chunk)
                os.fsync(output)
                if size != row["size"] or member_digest.hexdigest() != row["sha256"]:
                    raise SystemExit("extracted source archive bytes differ")
            finally:
                os.close(output)
        observed = {}
        for current, directories, files in os.walk(destination, followlinks=False):
            base = Path(current)
            for name, kind in ([(name, "directory") for name in directories]
                               + [(name, "file") for name in files]):
                leaf = base / name; info = leaf.lstat()
                relative = leaf.relative_to(destination).as_posix()
                required = 0o700 if kind == "directory" else 0o400
                if ((kind == "directory" and not stat.S_ISDIR(info.st_mode))
                        or (kind == "file" and not stat.S_ISREG(info.st_mode))
                        or stat.S_ISLNK(info.st_mode)
                        or stat.S_IMODE(info.st_mode) != required
                        or (kind == "file" and info.st_nlink != 1)):
                    raise SystemExit("extracted source tree inode differs")
                observed[relative] = kind
        if observed != member_types:
            raise SystemExit("extracted source tree entry set differs")
        actual_rows = []
        for row in rows:
            target = destination.joinpath(*PurePosixPath(row["name"]).parts)
            info = target.lstat()
            actual = {"name": row["name"], "type": row["type"],
                      "mode": stat.S_IMODE(info.st_mode), "size": info.st_size}
            if row["type"] == "file":
                actual["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual["size"] != row["size"] or actual["sha256"] != row["sha256"]:
                    raise SystemExit("actual extracted source tree bytes differ")
            actual_rows.append(actual)
        extracted_sha = hashlib.sha256(canonical(actual_rows)).hexdigest()
    os.lseek(descriptor, 0, os.SEEK_SET)
    final_digest = hashlib.sha256()
    for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        final_digest.update(chunk)
    after = os.fstat(descriptor)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or final_digest.hexdigest() != expected_archive_sha):
        raise SystemExit("retained source archive changed during extraction")
finally:
    os.close(descriptor)
runtime = destination / ("methods/bernini_action_editing/"
                         "generate_saic_pure_t2v_event_bank_topup_v2.py")
if hashlib.sha256(runtime.read_bytes()).hexdigest() != expected_runtime_sha:
    raise SystemExit("extracted archive runtime bytes differ")
core = {
    "schema_version": "saic-source-anchor-archive-extraction-binding-v1",
    "status": "exact864_archive_extracted_and_actual_tree_verified",
    "source_archive_sha256": expected_archive_sha,
    "archive_member_manifest_sha256": expected_manifest_sha,
    "extracted_tree_manifest_sha256": extracted_sha,
    "extracted_tree_manifest_source": "actual_lstat_after_extraction",
    "extracted_tree_entry_set_exact": True, "archive_member_count": 864,
    "archive_regular_file_count": 853, "archive_directory_count": 11,
    "source_archive_read_from_stage0_retained_fd": True,
}
value = {**core, "receipt_digest": hashlib.sha256(canonical(core)).hexdigest()}
payload = canonical(value) + b"\n"
output = os.open(proof_path, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW, 0o400)
try:
    os.write(output, payload); os.fsync(output); os.lseek(output, 0, os.SEEK_SET)
    if os.read(output, len(payload) + 1) != payload:
        raise SystemExit("archive proof same-FD reread differs")
    os.fchmod(output, 0o444)
finally:
    os.close(output)
PY
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "archive method root differs"
runtime_origin_proof="${task_scratch}/input/runtime-origin-proof.json"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "${expected_python}" -I -B - \
  "${method_root}/generate_saic_pure_t2v_event_bank_topup_v2.py" \
  "${method_root}" "${archive_runtime_sha}" "${archive_sha}" \
  "${archive_member_manifest_sha}" "${runtime_origin_manifest_sha}" \
  "${runtime_origin_proof}" <<'PY'
import hashlib, importlib.util, json, os, stat, sys
from pathlib import Path

runtime, method_root = map(Path, sys.argv[1:3])
runtime_sha, archive_sha, archive_manifest_sha, expected_origin_sha = sys.argv[3:7]
receipt_path = Path(sys.argv[7]); method_root = method_root.resolve(strict=True)
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
module_name = "generate_saic_pure_t2v_event_bank_topup_v2"
spec = importlib.util.spec_from_file_location(module_name, runtime)
if (runtime.resolve(strict=True).parent != method_root or spec is None
        or spec.loader is None or module_name in sys.modules):
    raise SystemExit("isolated runtime checker identity differs")
module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module
spec.loader.exec_module(module)
try:
    status = module.main(["--help"])
except SystemExit as error:
    status = error.code
if status not in (None, 0):
    raise SystemExit("isolated runtime help failed")
expected_modules = {
    "build_saic_reversible_source_set_v1", "generate_saic_pure_t2v_event_bank_v1",
    "infer_lora", "infer_native_identity_generation_canary",
    "infer_pair_v5_t2v_calibration_bank", "infer_source_kv_carrier_oracle",
    "infer_source_value_residual_oracle", "pair_v5_t2v_calibration_bank_spec",
    "saic_pure_t2v_event_bank_topup_v2", "saic_pure_t2v_event_bank_v1",
    "source_kv_replay", "source_kv_route_batches", "source_value_residual",
    "train_lora",
}
rows = []
for name, imported in sorted(sys.modules.items()):
    origin = getattr(imported, "__file__", None)
    if not isinstance(origin, str):
        continue
    try:
        path = Path(origin).resolve(strict=True)
        relative = path.relative_to(method_root).as_posix()
    except (FileNotFoundError, OSError, ValueError):
        continue
    info = path.lstat()
    if (name not in expected_modules or relative != name + ".py"
            or not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o400):
        raise SystemExit("runtime project import origin differs")
    rows.append({"module": name, "relative_path": relative,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
raw_rows = canonical(rows); origin_sha = hashlib.sha256(raw_rows).hexdigest()
if ({row["module"] for row in rows} != expected_modules or len(rows) != 14
        or len(raw_rows) != 2331 or origin_sha != expected_origin_sha
        or hashlib.sha256(runtime.read_bytes()).hexdigest() != runtime_sha):
    raise SystemExit("runtime recursive import closure differs")
core = {
    "schema_version": "saic-source-anchor-runtime-import-origin-closure-v1",
    "status": "isolated_runtime_help_exact14_origins_verified",
    "source_archive_sha256": archive_sha,
    "archive_member_manifest_sha256": archive_manifest_sha,
    "runtime_sha256": runtime_sha, "project_module_count": 14,
    "project_module_rows": rows, "project_module_manifest_sha256": origin_sha,
    "all_project_module_origins_from_extracted_archive": True,
    "isolated_python_no_environment_path": True,
}
value = {**core, "receipt_digest": hashlib.sha256(canonical(core)).hexdigest()}
payload = canonical(value) + b"\n"
descriptor = os.open(receipt_path, os.O_RDWR | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW, 0o400)
try:
    os.write(descriptor, payload); os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.read(descriptor, len(payload) + 1) != payload:
        raise SystemExit("runtime origin proof same-FD reread differs")
    os.fchmod(descriptor, 0o444)
finally:
    os.close(descriptor)
PY
cp -- "/proc/$$/fd/${SAIC_ANCHOR_TRAINER_FD}" \
  "${task_scratch}/input/train_saic_source_anchor_v1.py"
[[ -f "${method_root}/train_saic_source_anchor_v1.py" && \
   ! -L "${method_root}/train_saic_source_anchor_v1.py" ]] || \
  fail "archive trainer overlay target differs"
rm -f -- "${method_root}/train_saic_source_anchor_v1.py"
mv -- "${task_scratch}/input/train_saic_source_anchor_v1.py" \
  "${method_root}/train_saic_source_anchor_v1.py"
cp -- "/proc/$$/fd/${SAIC_ANCHOR_SOURCE_MANIFEST_FD}" \
  "${task_scratch}/input/source-anchor.json"
cp -- "/proc/$$/fd/${SAIC_ANCHOR_CHECKPOINT_MANIFEST_FD}" \
  "${task_scratch}/input/checkpoint.sha256"
chmod 0400 -- "${task_scratch}/input/source-anchor.json" \
  "${task_scratch}/input/checkpoint.sha256" \
  "${method_root}/train_saic_source_anchor_v1.py"
[[ "$(sha256sum "${method_root}/train_saic_source_anchor_v1.py" | awk '{print $1}')" == "${trainer_sha}" && \
   "$(sha256sum "${task_scratch}/input/source-anchor.json" | awk '{print $1}')" == "${manifest_sha}" && \
   "$(sha256sum "${task_scratch}/input/checkpoint.sha256" | awk '{print $1}')" == "${checkpoint_manifest_sha}" && \
   "$(git get-tar-commit-id <"/proc/$$/fd/${SAIC_ANCHOR_ARCHIVE_FD}")" == "${archive_revision}" ]] || \
  fail "retained input materialization differs"

mapfile -t archive_bindings < <("${expected_python}" -I -B - \
  "${archive_proof}" "${runtime_origin_proof}" \
  "${archive_sha}" "${archive_member_manifest_sha}" \
  "${runtime_origin_manifest_sha}" <<'PY'
import hashlib, json, sys
from pathlib import Path

canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
values = []
for path in map(Path, sys.argv[1:3]):
    raw = path.read_bytes(); value = json.loads(raw.decode("ascii"))
    unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
    if (raw != canonical(value) + b"\n"
            or hashlib.sha256(canonical(unsigned)).hexdigest() != claimed):
        raise SystemExit("archive/origin proof canonical seal differs")
    values.append(value)
archive, origin = values
if (archive.get("source_archive_sha256") != sys.argv[3]
        or archive.get("archive_member_manifest_sha256") != sys.argv[4]
        or archive.get("archive_member_count") != 864
        or archive.get("archive_regular_file_count") != 853
        or archive.get("archive_directory_count") != 11
        or archive.get("extracted_tree_manifest_source")
        != "actual_lstat_after_extraction"
        or archive.get("extracted_tree_entry_set_exact") is not True
        or origin.get("source_archive_sha256") != sys.argv[3]
        or origin.get("archive_member_manifest_sha256") != sys.argv[4]
        or origin.get("project_module_count") != 14
        or origin.get("project_module_manifest_sha256") != sys.argv[5]
        or origin.get("all_project_module_origins_from_extracted_archive") is not True):
    raise SystemExit("archive/origin proof semantics differ")
print(archive["extracted_tree_manifest_sha256"])
print(archive["receipt_digest"])
print(origin["receipt_digest"])
PY
)
[[ "${#archive_bindings[@]}" == 3 ]] || fail "archive proof binding count differs"
extracted_tree_manifest_sha="${archive_bindings[0]}"
archive_binding_digest="${archive_bindings[1]}"
runtime_origin_digest="${archive_bindings[2]}"
for value in extracted_tree_manifest_sha archive_binding_digest runtime_origin_digest; do
  [[ "${!value}" =~ ^[0-9a-f]{64}$ ]] || fail "${value} differs"
done

mapfile -t bindings < <("${expected_python}" -B - \
  "/proc/$$/fd/${SAIC_ANCHOR_RELEASE_MANIFEST_FD}" \
  "${release_manifest_sha}" "${release_manifest_digest}" \
  "/proc/$$/fd/${SAIC_ANCHOR_FULL60_ADMISSION_FD}" \
  "${full60_admission_sha}" "${full60_admission_digest}" \
  "${full60_admission_schema}" "${full60_admission_status}" \
  "/proc/$$/fd/${SAIC_ANCHOR_SUBMISSION_RECEIPT_FD}" \
  "${SLURM_JOB_ID}" "${wrapper_sha}" <<'PY'
import hashlib, json, os, stat, sys, time

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

def decode(path, expected_sha, expected_digest, label):
    raw = open(path, "rb", buffering=0).read()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SystemExit(label + " SHA differs")
    value = json.loads(raw.decode("ascii"))
    unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
    if (claimed != expected_digest or hashlib.sha256(canonical(unsigned)).hexdigest()
            != claimed or raw != canonical(value) + b"\n"):
        raise SystemExit(label + " canonical seal differs")
    return value

release = decode(sys.argv[1], sys.argv[2], sys.argv[3], "release manifest")
full60 = decode(sys.argv[4], sys.argv[5], sys.argv[6], "formal full60 admission")
full60_fields = {
    "schema_version", "status", "job_id", "job_success",
    "slurm_terminal_verified", "formal_admission",
    "formal_full60_result_claimed", "candidate_count", "attempt_count",
    "same_seed_proof_count", "master_receipt", "submission_receipt",
    "slurm_terminal_observation", "world8_closure",
    "failure_receipt_absent", "output_namespace_deep_closed", "authority",
    "receipt_digest",
}
master_fields = {
    "path", "sha256", "receipt_digest", "schema_version", "bank_id",
    "attempt_count", "row_count", "seed_cell_count", "branch_order",
    "master_deep_validated",
}
submission_fields = {
    "path", "sha256", "receipt_digest", "job_id",
    "request_candidate_count", "submission_deep_validated",
}
slurm_fields = {
    "executable_sha256", "stdout_sha256", "stderr_sha256",
    "exact_single_row", "state", "exit_code", "allocated_gpu_count",
    "alloc_tres", "node_list", "start", "end", "elapsed",
    "submit_line_sha256", "exact_submit_line",
}
world8_fields = {
    "retained_fd_world8_admission_path",
    "retained_fd_world8_admission_sha256",
    "retained_fd_world8_admission_digest",
    "compute_bash_probe_admission_path",
    "compute_bash_probe_admission_sha256",
    "compute_bash_probe_admission_digest",
    "archive_member_manifest_sha256", "archive_member_count",
    "archive_regular_file_count", "archive_directory_count",
    "runtime_origin_manifest_sha256", "runtime_origin_project_module_count",
    "underlying_world8_closure_deep_validated",
}
expected_authority = {
    "scientific": False, "action": False, "identity": False,
    "training": False, "checkpoint": False, "publication": False,
}
master = full60.get("master_receipt")
submission = full60.get("submission_receipt")
slurm = full60.get("slurm_terminal_observation")
world8 = full60.get("world8_closure")
sha = lambda value: isinstance(value, str) and len(value) == 64 and all(
    character in "0123456789abcdef" for character in value)
if (set(full60) != full60_fields
        or full60.get("schema_version") != sys.argv[7]
        or full60.get("schema_version")
        != "saic-t2v-topup-r6-formal-full60-terminal-admission-v1"
        or full60.get("status") != sys.argv[8]
        or full60.get("status") != "terminal_completed_formal_full60_admitted"
        or not isinstance(full60.get("job_id"), str)
        or not full60["job_id"].isdigit()
        or full60.get("job_success") is not True
        or full60.get("slurm_terminal_verified") is not True
        or full60.get("formal_admission") is not True
        or full60.get("formal_full60_result_claimed") is not True
        or full60.get("candidate_count") != 60
        or full60.get("attempt_count") != 60
        or full60.get("same_seed_proof_count") != 20
        or full60.get("failure_receipt_absent") is not True
        or full60.get("output_namespace_deep_closed") is not True
        or full60.get("authority") != expected_authority
        or not isinstance(master, dict) or set(master) != master_fields
        or master.get("schema_version")
        != "bernini-saic-pure-t2v-event-bank-topup-receipt-v2"
        or master.get("bank_id")
        != "saic-text-only-hard-negative-topup-exact81-v2"
        or master.get("attempt_count") != 60 or master.get("row_count") != 8
        or master.get("seed_cell_count") != 20
        or master.get("branch_order")
        != ["incomplete", "camera_only", "appearance_only"]
        or master.get("master_deep_validated") is not True
        or not all(sha(master.get(key)) for key in ("sha256", "receipt_digest"))
        or not isinstance(submission, dict) or set(submission) != submission_fields
        or submission.get("job_id") != full60.get("job_id")
        or submission.get("request_candidate_count") != 60
        or submission.get("submission_deep_validated") is not True
        or not all(sha(submission.get(key)) for key in ("sha256", "receipt_digest"))
        or not isinstance(slurm, dict) or set(slurm) != slurm_fields
        or slurm.get("exact_single_row") is not True
        or slurm.get("state") != "COMPLETED" or slurm.get("exit_code") != "0:0"
        or slurm.get("allocated_gpu_count") != 8
        or slurm.get("exact_submit_line") is not True
        or not all(sha(slurm.get(key)) for key in (
            "executable_sha256", "stdout_sha256", "stderr_sha256",
            "submit_line_sha256"))
        or not isinstance(world8, dict) or set(world8) != world8_fields
        or world8.get("archive_member_manifest_sha256")
        != "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
        or world8.get("archive_member_count") != 864
        or world8.get("archive_regular_file_count") != 853
        or world8.get("archive_directory_count") != 11
        or world8.get("runtime_origin_manifest_sha256")
        != "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
        or world8.get("runtime_origin_project_module_count") != 14
        or world8.get("underlying_world8_closure_deep_validated") is not True
        or not all(isinstance(world8.get(key), str)
                   and world8[key].startswith("/") for key in (
                       "retained_fd_world8_admission_path",
                       "compute_bash_probe_admission_path"))
        or not all(sha(world8.get(key)) for key in (
            "retained_fd_world8_admission_sha256",
            "retained_fd_world8_admission_digest",
            "compute_bash_probe_admission_sha256",
            "compute_bash_probe_admission_digest"))):
    raise SystemExit("formal full60 exact deep admission differs")
receipt_path = sys.argv[9]; job_id = sys.argv[10]; wrapper_sha = sys.argv[11]
receipt_fd = int(os.path.basename(receipt_path))
deadline = time.monotonic() + 120.0
while True:
    os.lseek(receipt_fd, 0, os.SEEK_SET)
    raw = os.read(receipt_fd, 1024 * 1024)
    try:
        receipt = json.loads(raw.decode("ascii"))
        unsigned = dict(receipt); claimed = unsigned.pop("receipt_digest")
        retained = os.fstat(receipt_fd)
        public = os.stat(os.environ["SAIC_ANCHOR_SUBMISSION_RECEIPT"],
                         follow_symlinks=False)
        if (raw == canonical(receipt) + b"\n"
                and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed
                and receipt.get("status") == "submitted"
                and receipt.get("submitted_job", {}).get("job_id") == job_id
                and receipt.get("inputs", {}).get("wrapper", {}).get("sha256")
                == wrapper_sha
                and (retained.st_dev, retained.st_ino)
                == (public.st_dev, public.st_ino)
                and retained.st_nlink == public.st_nlink == 1
                and stat.S_IMODE(retained.st_mode) == 0o444
                and stat.S_IMODE(public.st_mode) == 0o444):
            break
    except (ValueError, KeyError, UnicodeError):
        pass
    if time.monotonic() >= deadline:
        raise SystemExit("submission receipt did not become terminal")
    time.sleep(0.2)
print(receipt["receipt_digest"])
print(hashlib.sha256(raw).hexdigest())
print(release["receipt_digest"])
PY
)
[[ "${#bindings[@]}" == 3 ]] || fail "retained binding count differs"
submission_digest="${bindings[0]}"
submission_sha="${bindings[1]}"
observed_release_digest="${bindings[2]}"
[[ "${observed_release_digest}" == "${release_manifest_digest}" ]] || \
  fail "release digest differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"

run_output="${output_parent}/job-${SLURM_JOB_ID}"
[[ ! -e "${run_output}" && ! -L "${run_output}" ]] || fail "training output is not fresh"
echo "[saic-anchor-f32-r1] START WORLD8=DP2xSP4 exact81 formal32 lr=1e-5 seed=20260809 grad_accum=1"
PYTHONPATH="${method_root}" "${expected_python}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 \
  "${method_root}/train_saic_source_anchor_v1.py" \
  --bernini-root "${expected_bernini}" \
  --veomni-root "${expected_veomni}" \
  --checkpoint "${expected_checkpoint}" \
  --checkpoint-content-manifest "${task_scratch}/input/checkpoint.sha256" \
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha}" \
  --manifest "${task_scratch}/input/source-anchor.json" \
  --expected-manifest-sha256 "${manifest_sha}" \
  --output "${run_output}" --mode formal --max-updates 32 \
  --learning-rate 1e-5 --max-grad-norm 1 --gradient-accumulation-steps 1 \
  --wrong-source-margin 0.01 --ranking-weight 1 \
  --seed 20260809 --num-frames 81 \
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha}" \
  --method-source-revision "${archive_revision}" \
  --method-source-archive-sha256 "${archive_sha}" \
  --trainer-source-sha256 "${trainer_sha}" \
  --release-manifest-sha256 "${release_manifest_sha}" \
  --release-manifest-digest "${release_manifest_digest}" \
  --submission-receipt-sha256 "${submission_sha}" \
  --submission-receipt-digest "${submission_digest}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --python-executable "${expected_python}" \
  --python-executable-sha256 "${python_sha}" \
  --python-version-stdout-sha256 "${python_version_sha}" \
  --formal-full60-admission-sha256 "${full60_admission_sha}" \
  --formal-full60-admission-digest "${full60_admission_digest}" \
  --source-fd-map-sha256 "${SAIC_ANCHOR_SOURCE_FD_MAP_SHA256}" \
  --source-supervisor-pid "${SAIC_ANCHOR_SOURCE_SUPERVISOR_PID}" \
  --archive-member-manifest-sha256 "${archive_member_manifest_sha}" \
  --extracted-tree-manifest-sha256 "${extracted_tree_manifest_sha}" \
  --archive-binding-receipt-digest "${archive_binding_digest}" \
  --runtime-origin-manifest-sha256 "${runtime_origin_manifest_sha}" \
  --runtime-origin-receipt-digest "${runtime_origin_digest}" \
  --ack-source-anchor-only-no-action-claim

classification="$("${expected_python}" -B - "${run_output}" <<'PY'
import hashlib, json, pathlib, stat, sys
root = pathlib.Path(sys.argv[1]); receipt_path = root / "receipt.json"
raw = receipt_path.read_bytes(); receipt = json.loads(raw.decode("ascii"))
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
    ensure_ascii=True, allow_nan=False).encode("ascii")
unsigned = dict(receipt); claimed = unsigned.pop("receipt_digest", None)
if raw != canonical(receipt) + b"\n" or hashlib.sha256(canonical(unsigned)).hexdigest() != claimed:
    raise SystemExit("training receipt seal differs")
entries = {item.name for item in root.iterdir()}
status = receipt.get("status")
if status == "OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO":
    if entries != {"history.json", "receipt.json"}:
        raise SystemExit("NO_GO artifact closure differs")
elif status == "FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE":
    if entries != {"adapter.safetensors", "history.json", "receipt.json"}:
        raise SystemExit("PASS candidate artifact closure differs")
else:
    raise SystemExit("training scientific classification differs")
for item in root.iterdir():
    info = item.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit("private candidate artifact identity differs")
print(status)
PY
)"
echo "[saic-anchor-f32-r1] STRONG_AUDIT_OK classification=${classification} external_terminal_postflight_required=true"
