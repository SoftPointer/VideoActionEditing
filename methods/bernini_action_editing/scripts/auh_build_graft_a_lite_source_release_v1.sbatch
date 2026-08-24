#!/bin/bash -p
#SBATCH --job-name=graft-a-lite-c4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:20:00
#SBATCH --export=GRAFT_A_LITE_SOURCE_ARCHIVE,GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256,GRAFT_A_LITE_PYTHON_BIN,GRAFT_A_LITE_PYTHON_SHA256,GRAFT_A_LITE_LAUNCHER_SOURCE,GRAFT_A_LITE_LAUNCHER_SHA256,GRAFT_A_LITE_OUTPUT_STEM

# Slurm honors the shebang for a submitted script.  Direct callers must also
# use /bin/bash -p; fail before reading any task-controlled input otherwise.
case "$-" in
  *p*) ;;
  *) echo "[graft-a-lite-c4] ERROR: Bash privileged mode is required" >&2; exit 2 ;;
esac

set -Eeuo pipefail
umask 077

# The launcher has one Python invocation.  /usr/bin/env removes the submit
# shell environment, while -I -S prevents cwd/PYTHONPATH/user-site and site
# startup hooks from becoming code-loading inputs.  The supervisor below
# parses and compiles the exact builder member bytes already read from the
# hash-pinned archive; it never imports or executes a builder pathname.

source_archive="${GRAFT_A_LITE_SOURCE_ARCHIVE:?set one-file builder archive}"
source_archive_sha256="${GRAFT_A_LITE_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
python_bin="${GRAFT_A_LITE_PYTHON_BIN:?set absolute Python executable}"
python_sha256="${GRAFT_A_LITE_PYTHON_SHA256:?set Python executable SHA-256}"
launcher_source="${GRAFT_A_LITE_LAUNCHER_SOURCE:?set immutable launcher source}"
launcher_sha256="${GRAFT_A_LITE_LAUNCHER_SHA256:?set launcher SHA-256}"
output_stem="${GRAFT_A_LITE_OUTPUT_STEM:?set fresh logical output stem}"

readonly required_builder_commit=10d97980889138de28f95c72b08fd10cb3bbf6b9
readonly required_builder_git_blob_sha1=96078a71dd4be35ed1e137a1a96cb28e2b4ebe51
readonly required_builder_sha256=8419603d3edb33b965354869e7b090f0e3fe7c2f533d75ffb50e9afb73a6aabe
readonly required_builder_member=build_graft_a_lite_source_release_v1.py
readonly required_v16=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion128_v16_20260802T221943Z/prepare_full128_v16/candidates.jsonl
readonly required_v17=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/prepare_next1000_v17/candidates.jsonl
readonly required_v16_sha256=834e5a70e7c87683730ac644ce233b9343e4fc98eb3b3a45f55f93c8da94688d
readonly required_v17_sha256=24021e6a4c5d1758340f9e61df1a987383e1ad39063071526726e9658ccd1c10

fail() { echo "[graft-a-lite-c4] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

for item in source_archive python_bin launcher_source output_stem; do
  require_absolute "${!item}" "${item}"
done
for item in source_archive_sha256 python_sha256 launcher_sha256; do
  require_sha256 "${!item}" "${item}"
done
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
if [[ "${SLURM_TMPDIR+x}" == x ]]; then
  scratch_parent="${SLURM_TMPDIR}"
  scratch_parent_from_slurm_tmpdir=true
else
  scratch_parent=/tmp
  scratch_parent_from_slurm_tmpdir=false
fi
require_absolute "${scratch_parent}" "scratch parent"

# No inherited PYTHONPATH, PYTHONHOME, user site, sitecustomize, or arbitrary
# builder-control variables survive this boundary.  These are the complete
# process environment entries supplied to the single interpreter.
exec /usr/bin/env -i \
  PATH=/usr/bin:/bin LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "${python_bin}" -I -S -B - \
  "${python_bin}" "${python_sha256}" \
  "${launcher_source}" "${launcher_sha256}" "$0" \
  "${source_archive}" "${source_archive_sha256}" \
  "${required_builder_member}" "${required_builder_sha256}" \
  "${required_builder_git_blob_sha1}" "${required_builder_commit}" \
  "${required_v16}" "${required_v16_sha256}" \
  "${required_v17}" "${required_v17_sha256}" \
  "${output_stem}" "${scratch_parent}" "${scratch_parent_from_slurm_tmpdir}" \
  "${SLURM_JOB_ID}" \
  "${SLURM_JOB_NAME:-}" "${SLURM_JOB_NODELIST:-}" \
  "${SLURM_CLUSTER_NAME:-}" <<'PY'
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import tempfile
import types

(python_path, python_sha, launcher_path, launcher_sha, executed_launcher,
 archive_path, archive_sha, member_name, builder_sha, builder_blob,
 builder_commit, v16_path, v16_sha, v17_path, v17_sha, output_stem,
 scratch_parent, scratch_parent_from_slurm_tmpdir, job_id, job_name,
 node_list, cluster_name) = sys.argv[1:]

MANIFEST_SUFFIX = ".manifest.jsonl"
PRODUCER_RECEIPT_SUFFIX = ".receipt.json"
EXECUTION_RECEIPT_SUFFIX = ".execution.receipt.json"
NOOP = (
    "Keep every subject, action, timing, camera motion, framing, appearance, "
    "and background unchanged."
)
CANARY = (
    ("7b88a1ca1f804f41", "optimizer_train", True, False),
    ("a35b590961d24694", "optimizer_train", True, False),
    ("841b5e0080a1441d", "optimizer_confirmation", False, True),
    ("a66e6818e4144928", "optimizer_confirmation", False, True),
)
FFPROBE_PATH = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe"
FFPROBE_SHA = "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
FFPROBE_VERSION_SHA = "2271b81138bdaf07532b801ac7abd5b48d9e84dd66a6287a82fb44bc04c84f6b"
FFPROBE_VERSION_FIRST_LINE = (
    "ffprobe version 9.0 Copyright (c) 2007-2026 the FFmpeg developers"
)
FFPROBE_PIN_LABEL = "shared_portable_compute_verified_auh_ffprobe_v1"
FFPROBE_PROBE_KIND = "frozen_shared_portable_compute_verified_ffprobe_v2"
EXPECTED_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}


def die(message: str) -> None:
    raise SystemExit(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def object_digest(value: object) -> str:
    return digest(canonical(value))


def full_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def require_hex(value: str, length: int, label: str) -> None:
    if len(value) != length or any(ch not in "0123456789abcdef" for ch in value):
        die(f"{label} is not lowercase hex-{length}")


def read_plain(
    path_text: str,
    label: str,
    *,
    exact_realpath: bool = True,
    mode: int | None = None,
    executable: bool = False,
) -> tuple[bytes, dict[str, object]]:
    path = Path(path_text)
    if not path.is_absolute():
        die(f"{label} is not absolute")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        die(f"cannot resolve {label}: {error}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        die(f"{label} is not a plain file")
    if exact_realpath and resolved != path:
        die(f"{label} is not its exact realpath")
    if mode is not None and stat.S_IMODE(before.st_mode) != mode:
        die(f"{label} mode differs")
    if executable and not before.st_mode & 0o111:
        die(f"{label} is not executable")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if full_identity(opened) != full_identity(before):
            die(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
        leaf = path.lstat()
    finally:
        os.close(fd)
    if full_identity(opened) != full_identity(after) or full_identity(after) != full_identity(leaf):
        die(f"{label} changed while reading")
    raw = b"".join(chunks)
    return raw, {
        "path": str(path),
        "resolved_path": str(resolved),
        "sha256": digest(raw),
        "size_bytes": len(raw),
        "mode": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": after.st_dev,
        "inode": after.st_ino,
    }


def read_running_executable() -> tuple[bytes, dict[str, object]]:
    if sys.platform.startswith("linux") and Path("/proc/self/exe").exists():
        source = "/proc/self/exe"
        transport = "linux_proc_self_exe"
    else:
        source = str(Path(sys.executable).resolve(strict=True))
        transport = "sys_executable_path_fallback"
    fd = os.open(source, os.O_RDONLY)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            die("running Python executable is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if full_identity(metadata) != full_identity(after):
        die("running Python executable changed while hashing")
    raw = b"".join(chunks)
    return raw, {
        "transport": transport,
        "source": source,
        "sha256": digest(raw),
        "size_bytes": len(raw),
        "device": after.st_dev,
        "inode": after.st_ino,
    }


def parent_path_matches(parent: Path, wanted: tuple[int, int]) -> bool:
    try:
        metadata = parent.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and inode_identity(metadata) == wanted
    )


def open_output_parent(stem_text: str) -> tuple[Path, str, int, tuple[int, int]]:
    stem = Path(stem_text)
    if not stem.is_absolute() or not stem.name or stem.name in {".", ".."}:
        die("output stem is not an absolute leaf")
    if stem.name.endswith((MANIFEST_SUFFIX, PRODUCER_RECEIPT_SUFFIX, EXECUTION_RECEIPT_SUFFIX)):
        die("output stem includes an artifact suffix")
    parent = stem.parent
    try:
        before = parent.lstat()
        resolved = parent.resolve(strict=True)
    except OSError as error:
        die(f"output parent must already exist: {error}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode) or resolved != parent:
        die("output parent is not an exact plain directory")
    fd = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(fd)
    identity = inode_identity(opened)
    if identity != inode_identity(before) or not parent_path_matches(parent, identity):
        os.close(fd)
        die("output parent changed while pinning")
    for name in (
        stem.name,
        f"{stem.name}{MANIFEST_SUFFIX}",
        f"{stem.name}{PRODUCER_RECEIPT_SUFFIX}",
        f"{stem.name}{EXECUTION_RECEIPT_SUFFIX}",
    ):
        try:
            os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as error:
            os.close(fd)
            die(f"cannot preflight output {name}: {error}")
        os.close(fd)
        die(f"create-only output already exists: {name}")
    return parent, stem.name, fd, identity


def read_at(parent_fd: int, name: str, label: str, *, mode: int = 0o444) -> tuple[bytes, dict[str, object]]:
    if not name or "/" in name or name in {".", ".."}:
        die(f"unsafe {label} leaf")
    fd = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != mode:
            die(f"{label} type or mode differs")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if full_identity(opened) != full_identity(after):
        die(f"{label} changed while reading")
    raw = b"".join(chunks)
    return raw, {
        "leaf_name": name,
        "access": "retained_output_parent_fd_openat",
        "sha256": digest(raw),
        "size_bytes": len(raw),
        "mode": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": after.st_dev,
        "inode": after.st_ino,
    }


def create_at(parent_fd: int, name: str, raw: bytes) -> dict[str, object]:
    fd = os.open(
        name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                die("zero-byte execution receipt write")
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        metadata = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        reopened_chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            reopened_chunks.append(block)
        if b"".join(reopened_chunks) != raw:
            die("execution receipt same-fd reopen differs")
    finally:
        os.close(fd)
    os.fsync(parent_fd)
    return {
        "leaf_name": name,
        "access": "retained_output_parent_fd_openat_create_only",
        "sha256": digest(raw),
        "size_bytes": len(raw),
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def reject_constant(value: str) -> None:
    die(f"non-finite JSON constant: {value}")


def reject_duplicate_pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            die(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_canonical_line(raw: bytes, label: str) -> dict[str, object]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        die(f"{label} is not one newline-terminated JSON object")
    try:
        value = json.loads(
            raw[:-1].decode("ascii"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        die(f"{label} JSON differs: {error}")
    if not isinstance(value, dict) or canonical(value) + b"\n" != raw:
        die(f"{label} is not canonical")
    return value


def create_builder_backing(builder_raw: bytes) -> tuple[Path, int, tuple[int, int, int, int, int]]:
    scratch = Path(scratch_parent)
    if not scratch.is_absolute():
        die("scratch parent is not absolute")
    try:
        before = scratch.lstat()
        resolved = scratch.resolve(strict=True)
    except OSError as error:
        die(f"scratch parent differs: {error}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode) or resolved != scratch:
        die("scratch parent is not an exact plain directory")
    backing_dir = Path(tempfile.mkdtemp(prefix=f"graft-a-lite-c4-{job_id}.", dir=scratch))
    backing_path = backing_dir / member_name
    write_fd = os.open(
        backing_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(builder_raw)
        offset = 0
        while offset < len(view):
            written = os.write(write_fd, view[offset:])
            if written <= 0:
                die("zero-byte builder backing write")
            offset += written
        os.fchmod(write_fd, 0o444)
        os.fsync(write_fd)
    finally:
        os.close(write_fd)
    os.chmod(backing_dir, 0o555)
    retained_fd = os.open(backing_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    metadata = os.fstat(retained_fd)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        os.close(retained_fd)
        die("builder backing mode differs")
    return backing_path, retained_fd, full_identity(metadata)


def validate_backing(
    backing_path: Path,
    backing_fd: int,
    wanted_identity: tuple[int, int, int, int, int],
    builder_raw: bytes,
) -> None:
    opened = os.fstat(backing_fd)
    leaf = backing_path.lstat()
    if full_identity(opened) != wanted_identity or full_identity(leaf) != wanted_identity:
        die("builder backing identity changed")
    os.lseek(backing_fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(backing_fd, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    if b"".join(chunks) != builder_raw:
        die("builder backing bytes changed")


require_hex(python_sha, 64, "Python SHA-256")
require_hex(launcher_sha, 64, "launcher SHA-256")
require_hex(archive_sha, 64, "archive SHA-256")
require_hex(builder_sha, 64, "builder SHA-256")
require_hex(builder_blob, 40, "builder Git blob SHA-1")
require_hex(builder_commit, 40, "builder Git commit")
require_hex(v16_sha, 64, "v16 SHA-256")
require_hex(v17_sha, 64, "v17 SHA-256")
if not job_id.isdigit():
    die("Slurm job ID differs")
if scratch_parent_from_slurm_tmpdir not in {"true", "false"}:
    die("scratch-parent provenance differs")
if scratch_parent_from_slurm_tmpdir == "false":
    # /tmp is a symlink on some supported test hosts.  Resolve the fixed
    # fallback once, then validate and report the exact directory actually
    # used.  A caller-supplied SLURM_TMPDIR remains exact-realpath-only.
    scratch_parent = str(Path(scratch_parent).resolve(strict=True))
if dict(os.environ) != EXPECTED_ENVIRONMENT:
    die("supervisor environment differs from fixed allowlist")
if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.ignore_environment != 1
    or sys.flags.no_user_site != 1
):
    die("Python isolated invocation flags differ")

# The output directory descriptor is acquired before any archive parsing,
# builder compilation, or builder call, and remains open until the external
# receipt has been created and reopened.
output_parent, stem_name, parent_fd, parent_identity = open_output_parent(output_stem)
backing_fd = -1
try:
    python_raw, python_file = read_plain(python_path, "configured Python", executable=True)
    running_raw, running_file = read_running_executable()
    if digest(python_raw) != python_sha or digest(running_raw) != python_sha:
        die("running or configured Python executable SHA-256 differs")
    if Path(sys.executable).resolve(strict=True) != Path(python_path):
        die("running Python differs from configured exact realpath")

    launcher_raw, launcher_file = read_plain(
        launcher_path, "launcher source", mode=0o444
    )
    executed_raw, _ = read_plain(
        executed_launcher, "executed launcher", exact_realpath=False
    )
    if digest(launcher_raw) != launcher_sha or executed_raw != launcher_raw:
        die("executed launcher differs from pinned launcher source")

    archive_raw, archive_file = read_plain(
        archive_path, "source archive", mode=0o444
    )
    if digest(archive_raw) != archive_sha:
        die("source archive SHA-256 differs")
    v16_raw, v16_file = read_plain(v16_path, "v16 candidates")
    v17_raw, v17_file = read_plain(v17_path, "v17 candidates")
    if digest(v16_raw) != v16_sha:
        die("v16 candidates SHA-256 differs")
    if digest(v17_raw) != v17_sha:
        die("v17 candidates SHA-256 differs")

    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or ".." in member_path.parts or len(member_path.parts) != 1:
        die("builder archive member name is unsafe")
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:*") as handle:
        members = handle.getmembers()
        if len(members) != 1:
            die("source archive must contain exactly one member")
        member = members[0]
        if (
            member.name != member_name
            or not member.isfile()
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
        ):
            die("source archive member differs")
        stream = handle.extractfile(member)
        if stream is None:
            die("source archive member cannot be opened")
        builder_raw = stream.read()
    if digest(builder_raw) != builder_sha:
        die("builder SHA-256 differs")
    observed_blob = hashlib.sha1(
        b"blob " + str(len(builder_raw)).encode("ascii") + b"\0" + builder_raw
    ).hexdigest()
    if observed_blob != builder_blob:
        die("builder Git blob SHA-1 differs")

    backing_path, backing_fd, backing_identity = create_builder_backing(builder_raw)
    validate_backing(backing_path, backing_fd, backing_identity, builder_raw)

    # This compile input is the immutable bytes object parsed from archive_raw,
    # not backing_path.  __file__ exists solely because the committed builder
    # self-hashes it.  The backing file is observed and retained, not treated
    # as a formal Python-runtime or adversarial immutability boundary.
    module_name = "_graft_a_lite_archive_builder_v1"
    module = types.ModuleType(module_name)
    module.__file__ = str(backing_path)
    module.__package__ = None
    sys.modules[module_name] = module
    code = compile(builder_raw, str(backing_path), "exec", dont_inherit=True, optimize=0)
    exec(code, module.__dict__, module.__dict__)
    build_payload = module.__dict__.get("build_payload")
    publish_payload = module.__dict__.get("publish_payload")
    if not callable(build_payload) or not callable(publish_payload):
        die("builder API differs")
    validate_backing(backing_path, backing_fd, backing_identity, builder_raw)

    # Builder exceptions deliberately propagate.  Its create-only partial
    # artifacts and the tiny read-only backing directory are never removed or
    # rolled back by this launcher.
    payload = build_payload(
        v16_candidates=v16_path,
        v17_candidates=v17_path,
        mode="canary4",
        workers=4,
    )
    publish_payload(Path(output_stem), payload)
    validate_backing(backing_path, backing_fd, backing_identity, builder_raw)
    if dict(os.environ) != EXPECTED_ENVIRONMENT:
        die("supervisor environment changed during builder execution")

    # Revalidate every pathname observation after producer return.  In
    # particular, a replaced output-parent pathname fails before any external
    # execution receipt is created; reads below still use the retained fd.
    if not parent_path_matches(output_parent, parent_identity):
        die("output parent path identity changed after builder return")
    for path_text, wanted, label, mode in (
        (python_path, python_sha, "configured Python", None),
        (launcher_path, launcher_sha, "launcher source", 0o444),
        (archive_path, archive_sha, "source archive", 0o444),
        (v16_path, v16_sha, "v16 candidates", None),
        (v17_path, v17_sha, "v17 candidates", None),
    ):
        raw, _ = read_plain(path_text, label, mode=mode)
        if digest(raw) != wanted:
            die(f"{label} changed after builder return")
    running_after, _ = read_running_executable()
    if digest(running_after) != python_sha:
        die("running Python changed after builder return")

    manifest_name = f"{stem_name}{MANIFEST_SUFFIX}"
    producer_name = f"{stem_name}{PRODUCER_RECEIPT_SUFFIX}"
    execution_name = f"{stem_name}{EXECUTION_RECEIPT_SUFFIX}"
    manifest_raw, manifest_file = read_at(parent_fd, manifest_name, "manifest")
    producer_raw, producer_file = read_at(parent_fd, producer_name, "producer receipt")
    producer = parse_canonical_line(producer_raw, "producer receipt")
    producer_unsigned = dict(producer)
    producer_declared_digest = producer_unsigned.pop("receipt_digest", None)
    if producer_declared_digest != object_digest(producer_unsigned):
        die("producer receipt digest differs")
    if (
        producer.get("schema_version") != "bernini-graft-a-lite-source-noop-receipt-v1"
        or producer.get("status") != "complete"
        or producer.get("release_mode") != "canary4"
    ):
        die("producer receipt identity differs")
    artifact = producer.get("artifact", {})
    if artifact.get("manifest_rows") != 4 or artifact.get("manifest_sha256") != digest(manifest_raw):
        die("producer receipt does not bind canary4 manifest")
    expected_semantics = {
        "source_only": True,
        "same_clip_noop_only": True,
        "source_video_is_clean_noop_endpoint": True,
        "cross_clip_identity_authority": False,
        "action_authority": False,
        "quality_authority": False,
        "production_authority": False,
        "scientific_success_claimed": False,
        "canonical_noop_instruction": NOOP,
    }
    if producer.get("semantics") != expected_semantics:
        die("producer semantics authority boundary differs")
    research = producer.get("research_authorization_record", {})
    if any(
        research.get(key) is not False
        for key in (
            "data_governance_authority_claimed",
            "source_license_authority_claimed",
            "supersedes_upstream_release_or_receipt",
            "production_use_authorized",
        )
    ):
        die("producer receipt claims forbidden research authority")
    policy = producer.get("input_policy", {})
    if (
        any(
            policy.get(key) is not False
            for key in (
                "external_target_artifacts_opened",
                "wan_preview_opened",
                "generated_target_opened",
                "legacy_latent_or_receipt_opened",
                "anchor_image_opened",
            )
        )
        or policy.get("code_frozen_v16_v17_manifest_pins_matched") is not True
    ):
        die("producer input authority boundary differs")
    producer_inputs = producer.get("inputs")
    if not isinstance(producer_inputs, list) or len(producer_inputs) != 2:
        die("producer input manifest bindings differ")
    expected_input_bindings = {(str(Path(v16_path)), v16_sha), (str(Path(v17_path)), v17_sha)}
    observed_input_bindings = {
        (item.get("path"), item.get("file_sha256"))
        for item in producer_inputs
        if isinstance(item, dict)
    }
    if observed_input_bindings != expected_input_bindings:
        die("producer input manifest hashes differ")
    split = producer.get("split", {})
    if (
        split.get("optimizer_train_rows") != 2
        or split.get("optimizer_confirmation_rows") != 2
        or split.get("optimizer_confirmation_update_intended") is not False
        or split.get("optimizer_confirmation_update_authorized") is not False
        or split.get("optimizer_confirmation_actual_use_claimed") is not False
        or split.get("global_holdout") is not False
    ):
        die("producer split authority differs")
    media = producer.get("media_contract", {})
    if (
        media.get("probe_kind") != FFPROBE_PROBE_KIND
        or media.get("fresh_ffprobe") is not True
        or media.get("fresh_ffprobe_verified_rows") != 4
        or media.get("frame_count") != 81
        or media.get("fps_fraction") != "25/1"
        or media.get("temporal_padding_allowed") is not False
        or media.get("temporal_truncation_allowed") is not False
    ):
        die("producer media evidence differs")
    implementation = producer.get("implementation", {})
    if (
        implementation.get("sha256") != builder_sha
        or implementation.get("path") != str(backing_path)
        or implementation.get("media_probe_kind") != FFPROBE_PROBE_KIND
    ):
        die("producer builder implementation binding differs")
    probe = implementation.get("ffprobe_executable_observation", {})
    transport = probe.get("executable_transport")
    fixed_inode_transport = (
        transport in {"linux_proc_self_fd", "portable_dev_fd"}
        and probe.get("executable_fixed_inode_execution") is True
        and probe.get("absolute_path_fallback_pre_post_inode_sha") is False
    )
    absolute_path_transport = (
        transport == "absolute_realpath_pre_post_inode_sha_fallback"
        and probe.get("executable_fixed_inode_execution") is False
        and probe.get("absolute_path_fallback_pre_post_inode_sha") is True
    )
    if (
        probe.get("pin_label") != FFPROBE_PIN_LABEL
        or probe.get("configured_path") != FFPROBE_PATH
        or probe.get("resolved_path") != FFPROBE_PATH
        or probe.get("exact_realpath_matched") is not True
        or probe.get("path_lookup_used") is not False
        or probe.get("file_sha256_expected") != FFPROBE_SHA
        or probe.get("file_sha256_observed") != FFPROBE_SHA
        or probe.get("file_sha256_matched") is not True
        or probe.get("version_stdout_sha256_expected") != FFPROBE_VERSION_SHA
        or probe.get("version_stdout_sha256_observed") != FFPROBE_VERSION_SHA
        or probe.get("version_stdout_sha256_matched") is not True
        or probe.get("version_first_line_expected") != FFPROBE_VERSION_FIRST_LINE
        or probe.get("version_first_line_observed") != FFPROBE_VERSION_FIRST_LINE
        or probe.get("version_first_line_matched") is not True
        or probe.get("pre_and_post_version_identity_and_file_sha_revalidated") is not True
        or probe.get("caller_process_observation_only") is not True
        or probe.get("trusted_or_official_authority_claimed") is not False
        or not (fixed_inode_transport or absolute_path_transport)
    ):
        die("producer ffprobe pin evidence differs")
    publication = producer.get("publication", {})
    if (
        publication.get("publication_eligible") is not True
        or publication.get("create_only") is not True
        or publication.get("layout") != "flat_sibling_artifacts_v1"
        or publication.get("published_file_mode") != "0444"
        or publication.get("reader_must_require_producer_success") is not True
    ):
        die("producer publication contract differs")

    lines = manifest_raw.splitlines(keepends=True)
    if len(lines) != 4 or b"".join(lines) != manifest_raw:
        die("manifest row framing differs")
    rows: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []
    for index, (row_raw, expected) in enumerate(zip(lines, CANARY)):
        row = parse_canonical_line(row_raw, f"manifest row {index}")
        iid, wanted_split, update_authorized, confirmation_only = expected
        unsigned = dict(row)
        row_digest = unsigned.pop("row_digest", None)
        if row_digest != object_digest(unsigned):
            die(f"manifest row {index} digest differs")
        if (
            row.get("schema_version") != "bernini-graft-a-lite-source-noop-row-v1"
            or row.get("release_mode") != "canary4"
            or row.get("row_index") != index
            or row.get("iid") != iid
            or row.get("split") != wanted_split
            or row.get("optimizer_update_authorized") is not update_authorized
            or row.get("optimizer_confirmation_only") is not confirmation_only
            or row.get("prior_research_exposure") is not True
            or row.get("global_holdout") is not False
            or row.get("stable_identity_disjoint_split_claimed") is not False
            or row.get("same_clip_noop_only") is not True
            or row.get("source_video_is_clean_noop_endpoint") is not True
            or row.get("cross_clip_identity_authority") is not False
            or row.get("action_authority") is not False
            or row.get("quality_authority") is not False
            or row.get("production_authority") is not False
            or row.get("publication_eligible") is not True
        ):
            die(f"manifest row {index} authority differs")
        if (
            row.get("source_hash_and_probe_same_open_fd") is not True
            or row.get("source_sha256_recomputed_before_and_after_probe") is not True
            or row.get("source_pre_post_probe_sha256_matched") is not True
            or row.get("source_identity_includes_ctime_ns") is not True
            or row.get("source_path_inode_binding_revalidated") is not True
            or row.get("noop_instruction") != NOOP
        ):
            die(f"manifest row {index} source evidence differs")
        source_path = row.get("source_video_path")
        if not isinstance(source_path, str):
            die(f"manifest row {index} source path differs")
        source_raw, source_file = read_plain(source_path, f"source video {iid}")
        if (
            source_file["sha256"] != row.get("source_video_sha256")
            or source_file["size_bytes"] != row.get("source_file_size_bytes")
        ):
            die(f"source video {iid} hash or size differs after publication")
        source_file["iid"] = iid
        source_files.append(source_file)
        rows.append(row)

    validate_backing(backing_path, backing_fd, backing_identity, builder_raw)
    if not parent_path_matches(output_parent, parent_identity):
        die("output parent path identity changed before execution receipt")
    try:
        os.stat(execution_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        die("execution receipt already exists after builder return")

    core = {
        "schema_version": "bernini-graft-a-lite-source-noop-execution-receipt-v2",
        "status": "complete",
        "successful_return": True,
        "builder_successful_return": True,
        "builder_publication_reopened_and_verified": True,
        "python_runtime_closure_verified": False,
        "formal_runtime_authority": False,
        "observations_are_not_an_adversarial_runtime_boundary": True,
        "slurm": {
            "job_id": job_id,
            "job_name": job_name,
            "node_list": node_list,
            "cluster_name": cluster_name,
            "cpu_only_workload": True,
            "gpu_resource_requested_by_launcher": True,
            "gpu_resource_request": "gpu:mi210:1",
            "effective_submission_request_verified": False,
            "gpu_computation_used": False,
        },
        "runtime_observations": {
            "python_configured_file": python_file,
            "python_running_executable": running_file,
            "python_invocation_flags": ["-I", "-S", "-B"],
            "environment_replaced_with_fixed_allowlist": True,
            "environment_revalidated_after_builder_return": True,
            "python_dynamic_libraries_and_standard_library_closure_pinned": False,
            "launcher": launcher_file,
            "source_archive": archive_file,
            "configured_scratch_parent": {
                "path": scratch_parent,
                "slurm_tmpdir_was_present_in_launcher_environment": (
                    scratch_parent_from_slurm_tmpdir == "true"
                ),
                "selection_observed_before_environment_replacement": True,
                "passed_to_supervisor_via_argv": True,
            },
            "builder": {
                "git_commit_observed_pin": builder_commit,
                "git_blob_sha1_observed_and_matched": builder_blob,
                "sha256_observed_and_matched": builder_sha,
                "archive_member": member_name,
                "archive_exactly_one_plain_member": True,
                "compiled_from_exact_in_memory_archive_member_bytes": True,
                "executed_or_imported_from_builder_path": False,
                "live_repository_imported": False,
                "backing_file": {
                    "path": str(backing_path),
                    "sha256": builder_sha,
                    "mode": "0444",
                    "fd_retained_through_verification": True,
                    "left_in_configured_scratch_parent": True,
                    "configured_scratch_parent": scratch_parent,
                    "slurm_tmpdir_was_present_in_launcher_environment": (
                        scratch_parent_from_slurm_tmpdir == "true"
                    ),
                    "automatic_cleanup_performed": False,
                    "used_only_for_builder___file___self_hash": True,
                    "claimed_as_immutable_or_formal_boundary": False,
                },
            },
            "builder_ffprobe_pin_revalidated_from_producer_receipt": True,
            "builder_ffprobe_expected_contract": {
                "media_probe_kind": FFPROBE_PROBE_KIND,
                "pin_label": FFPROBE_PIN_LABEL,
                "configured_and_resolved_path": FFPROBE_PATH,
                "file_sha256": FFPROBE_SHA,
                "version_stdout_sha256": FFPROBE_VERSION_SHA,
                "version_first_line": FFPROBE_VERSION_FIRST_LINE,
                "shared_portable_compute_verified_label_is_provenance_not_runtime_authority": True,
            },
        },
        "inputs": {
            "v16_candidates": v16_file,
            "v17_candidates": v17_file,
            "selected_source_videos": source_files,
            "all_selected_source_sha256_recomputed_after_publication": True,
            "target_video_opened": False,
            "wan_preview_opened": False,
            "anchor_image_opened": False,
            "legacy_latent_or_receipt_opened": False,
        },
        "outputs": {
            "logical_output_stem": str(Path(output_stem)),
            "retained_output_parent": {
                "path": str(output_parent),
                "device": parent_identity[0],
                "inode": parent_identity[1],
                "fd_retained_from_preflight_through_final_reopen": True,
                "path_identity_revalidated_before_external_receipt": True,
            },
            "manifest": manifest_file,
            "producer_receipt": producer_file,
            "manifest_rows": 4,
            "release_mode": "canary4",
            "producer_receipt_digest": producer_declared_digest,
            "canonical_json_and_digests_verified": True,
        },
        "split_execution": {
            "optimizer_train_rows": 2,
            "optimizer_confirmation_rows": 2,
            "optimizer_update_performed": False,
            "optimizer_confirmation_update_performed": False,
            "optimizer_confirmation_update_authorized": False,
            "confirmation_rows_prior_research_exposure": True,
            "global_holdout_claimed": False,
        },
        "authority": {
            "action_authority": False,
            "identity_authority": False,
            "cross_clip_identity_authority": False,
            "quality_authority": False,
            "training_authority": False,
            "production_authority": False,
            "data_governance_authority": False,
            "data_license_authority": False,
            "scientific_success_claimed": False,
        },
        "failure_semantics": {
            "create_only": True,
            "automatic_cleanup_of_publication": False,
            "automatic_cleanup_of_builder_backing": False,
            "external_execution_receipt_is_last_commit_marker": True,
            "receipt_alone_proves_successful_process_return": False,
            "consumer_must_also_require_slurm_completed_exit_zero": True,
            "partial_builder_publication_is_preserved_but_not_success": True,
            "consumer_must_require_this_valid_execution_receipt": True,
        },
    }
    execution = {**core, "receipt_digest": object_digest(core)}
    execution_raw = canonical(execution) + b"\n"
    # This create-only, same-fd-verified, read-only file is the final durable
    # action.  There is deliberately no fallible semantic verification after
    # it that could leave a success-looking receipt for a failed supervisor.
    create_at(parent_fd, execution_name, execution_raw)
finally:
    if backing_fd >= 0:
        try:
            os.close(backing_fd)
        except OSError:
            pass
    try:
        os.close(parent_fd)
    except OSError:
        pass
PY
