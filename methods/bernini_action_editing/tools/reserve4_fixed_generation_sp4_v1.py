#!/usr/bin/env python3
"""Plan, run, and audit the sealed reserve4 PAIR-v5 media on one SP4 island.

This utility is deliberately upstream of ``Phi_v1``.  It renders complete
ten-branch cells from the two preregistered reserve4 specs, but generated RGB,
latents, and Gaussian tensors remain authoring evidence only.  Even a complete
generation audit does not authorize representation extraction until separate
blind full-81-frame reviews have been sealed, and it never makes generated
artifacts an editor input or target.

The normal fit-first sequence is::

  build-plan --split fit ...
  smoke-sp4 ...  # sealed first candidate, full native 40-step path
  run-sp4 --seed-slot seed1 --group-id sp4-a ...
  run-sp4 --seed-slot seed1 --group-id sp4-b ...
  run-sp4 --seed-slot seed2 --group-id sp4-a ...
  run-sp4 --seed-slot seed2 --group-id sp4-b ...
  audit ... --generation-root <each non-empty shard output>

Empty shards are absent from the sealed plan and therefore need not be run.
The smoke output and rank caches remain retained, nonreusable forensic state
when the terminal compute-child controller attests and seals the bound scratch
tree.  This release grants no physical or manual cleanup authority, and it
does not guarantee persistence after the Slurm step or node lifecycle.
Candidate generation is serial inside a shard so the 64-GiB holder never
silently becomes a two-replica data-parallel job.  Each torchrun rank uses a
private node-local cache root; NFS COMGR temporary storage is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import build_pair_v5_t2v_seed2_bank as seed2_builder  # noqa: E402


PLAN_SCHEMA = "bernini-reserve4-fixed-generation-sp4-plan-v1"
GAP_SCHEMA = "bernini-reserve4-fixed-generation-gap-receipt-v1"
SHARD_RECEIPT_SCHEMA = "bernini-reserve4-fixed-generation-shard-receipt-v1"
AUDIT_RECEIPT_SCHEMA = "bernini-reserve4-fixed-generation-audit-receipt-v1"
COMPILE_SMOKE_SCHEMA = "bernini-generic-action-fit40-compile-smoke-v10"
R10_COMPILE_SMOKE_SCHEMA = "bernini-generic-action-fit40-compile-smoke-v2"
R10_TENSOR_PARITY_AUTHORITY_SCHEMA = (
    "bernini-generic-action-fit40-r10-tensor-parity-authority-v2"
)
SMOKE_TENSOR_EVIDENCE_SCHEMA = (
    "bernini-generic-action-fit40-smoke-tensor-evidence-v2"
)
HOST_CGROUP_MEMORY_GATE_SCHEMA = (
    "bernini-generic-action-fit40-host-cgroup-sampled-memory-gate-v4"
)
HOST_CGROUP_MEMORY_MONITOR_START_SCHEMA = (
    "bernini-generic-action-fit40-host-cgroup-memory-monitor-start-v3"
)
HOST_MEMORY_LIMIT_GIB = 60
HOST_MEMORY_LIMIT_BYTES = HOST_MEMORY_LIMIT_GIB * 1024**3
HOST_MEMORY_SAFE_CEILING_GIB = 56
HOST_MEMORY_SAFE_CEILING_BYTES = HOST_MEMORY_SAFE_CEILING_GIB * 1024**3
HOST_MEMORY_SAMPLE_INTERVAL_NS = 10_000_000
HOST_MEMORY_MAX_SAMPLE_GAP_NS = 100_000_000
HOST_MEMORY_SAMPLE_STRUCT = struct.Struct(">QQQQQQQQ")
HOST_MEMORY_SAMPLE_ENCODING = (
    "big-endian-u64-sequence-wall_ns-monotonic_ns-current-max-oom-oom_kill-kind-v2"
)
HOST_MEMORY_MONITOR_STOP_TOKEN = b"stop\n"
HOST_MEMORY_MONITOR_STOP_TOKEN_SHA256 = hashlib.sha256(
    HOST_MEMORY_MONITOR_STOP_TOKEN
).hexdigest()
T2V_GPU_MEMORY_LIMIT_GIB = 52
T2V_GPU_MEMORY_LIMIT_BYTES = T2V_GPU_MEMORY_LIMIT_GIB * 1024**3
RANK_EXEC = METHOD_ROOT / "scripts/auh_generic_action_data_prep_rank_exec_v1.sh"
FROZEN_PYTHON_PATH = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
FROZEN_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
FROZEN_PYTHON_SIZE = 31_490_256
ROOT_BOOTSTRAP_PYTHON_PATH = Path("/usr/bin/python3.10")
ROOT_BOOTSTRAP_PYTHON_SHA256 = (
    "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
)
ROOT_BOOTSTRAP_PYTHON_SIZE = 5_937_800
FROZEN_SITE_PACKAGES = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
)
SAFE_RUNTIME_PATH = "/usr/bin:/bin"
_ISOLATED_TORCHRUN_BOOTSTRAP = r"""
import hashlib
import sys
from types import ModuleType

source = sys.argv[1]
origin = sys.argv[2]
expected_sha256 = sys.argv[3]
site_packages = sys.argv[4]
raw = source.encode("utf-8")
if hashlib.sha256(raw).hexdigest() != expected_sha256:
    raise SystemExit("verified torchrun source SHA differs")
if site_packages != "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages":
    raise SystemExit("verified torchrun site-packages path differs")
sys.path.append(site_packages)
sys.argv = [origin, *sys.argv[5:]]
module = ModuleType("__main__")
module.__file__ = origin
module.__package__ = "torch.distributed"
module.__loader__ = None
sys.modules["__main__"] = module
exec(compile(raw, origin, "exec", dont_inherit=True), module.__dict__)
""".strip()
_VERIFIED_RUNNER_BOOTSTRAP = r"""
import hashlib
import os
import stat
import sys

path, expected_sha256 = sys.argv[1:3]
if (
    not os.path.isabs(path)
    or len(expected_sha256) != 64
    or any(character not in "0123456789abcdef" for character in expected_sha256)
):
    raise SystemExit(70)
descriptor = os.open(
    path,
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
)

def fields(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_blocks,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )

def read_all():
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)

opened = os.fstat(descriptor)
first = read_all()
middle = os.fstat(descriptor)
os.lseek(descriptor, 0, os.SEEK_SET)
second = read_all()
closed = os.fstat(descriptor)
os.close(descriptor)
named = os.lstat(path)
if not (
    os.path.realpath(path) == path
    and stat.S_ISREG(opened.st_mode)
    and not stat.S_ISLNK(named.st_mode)
    and opened.st_nlink == 1
    and stat.S_IMODE(opened.st_mode) == 0o444
    and fields(opened) == fields(middle) == fields(closed) == fields(named)
    and first == second
    and len(first) == opened.st_size
    and hashlib.sha256(first).hexdigest() == expected_sha256
):
    raise SystemExit(70)
globals_value = {
    "__name__": "_box_exp_013_verified_runner_bootstrap",
    "__file__": path,
    "__package__": None,
    "__spec__": None,
    "__builtins__": __builtins__,
}
exec(compile(first, path, "exec", dont_inherit=True), globals_value)
raise SystemExit(globals_value["main"](sys.argv[3:]))
""".strip()
PREPROCESSING_TOOL_SHA256 = {
    "tools/build_renderer_dataset.py": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
    ),
    "tools/materialize_vae.py": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
    ),
}
SPEC_AUTHORITIES = {
    "seed1": "2861b1021531896d387b0dccb945b9fc2516bf01472982c0fe2f7c1377ca7bab",
    "seed2": "0578cd6c39cdb625e69cf04164ffce29487b81e852974619f0ee43325e49398e",
}
SEED_PREFIXES = {
    "seed1": "pair5-t2v-reserve4-v1-",
    "seed2": "pair5-t2v-reserve4-seed2-",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
EMPTY_FILE_SHA256 = hashlib.sha256(b"").hexdigest()
GAUSSIAN_SAFETENSORS_METADATA = {
    "coordinate": "bernini_native_target_latent_before_rearrange",
    "source": "observed_return_of_official_module_global_randn_tensor",
    "observer_only": "true",
    "external_initial_noise_injection": "false",
}
CLEAN_LATENT_SAFETENSORS_METADATA = {
    "coordinate": "bernini_normalized_clean_vae_latent",
    "frame_contract": "exact81_latent21",
    "artifact_role": "native_sampler_proposal",
    "source": "native_sampler_before_vae_decode",
}


class Reserve4GenerationError(RuntimeError):
    """Raised before a mutable, partial, or over-authorized run can pass."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Reserve4GenerationError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_plain_file_bytes(
    value: str | Path, label: str, expected_sha256: Optional[str] = None
) -> tuple[Path, bytes, str]:
    """Read one canonical single-link file twice through one no-follow fd."""

    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    if expected_sha256 is not None:
        _require(
            SHA256_RE.fullmatch(expected_sha256) is not None,
            f"{label} expected SHA-256 differs",
        )
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise Reserve4GenerationError(
            f"{label} cannot be opened without following links"
        ) from error

    def fields(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_blocks,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def read_all() -> bytes:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    try:
        opened = os.fstat(descriptor)
        first = read_all()
        middle = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = read_all()
        closed = os.fstat(descriptor)
    except OSError as error:
        raise Reserve4GenerationError(f"{label} stable read failed") from error
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise Reserve4GenerationError(f"{label} named identity differs") from error
    observed = hashlib.sha256(first).hexdigest()
    _require(
        resolved == path
        and stat.S_ISREG(opened.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and opened.st_nlink == 1
        and fields(opened)
        == fields(middle)
        == fields(closed)
        == fields(named)
        and first == second
        and len(first) == opened.st_size
        and (expected_sha256 is None or observed == expected_sha256),
        f"{label} stable identity/SHA differs",
    )
    return path, first, observed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Reserve4GenerationError(message)


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _plain_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Reserve4GenerationError(f"{label} is unavailable: {path}") from error
    _require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be a plain file",
    )
    return path.resolve(strict=True)


def _plain_dir(value: str | Path, label: str) -> Path:
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Reserve4GenerationError(f"{label} is unavailable: {path}") from error
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be a plain directory",
    )
    return path.resolve(strict=True)


def _load_json(
    value: str | Path, label: str, expected_sha256: Optional[str] = None
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require(
            SHA256_RE.fullmatch(expected_sha256) is not None
            and observed == expected_sha256,
            f"{label} SHA-256 differs",
        )
    try:
        result = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                Reserve4GenerationError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Reserve4GenerationError(f"{label} is not valid UTF-8 JSON") from error
    _require(type(result) is dict, f"{label} must be a JSON object")
    return result, path, observed


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    parent = _plain_dir(path.parent, "output parent")
    _require(
        path.is_absolute() and parent == path.parent,
        f"output parent is unavailable: {path}",
    )
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    raw = canonical_json_bytes(value) + b"\n"
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        os.fchmod(descriptor, 0o400)
        opened = os.fstat(descriptor)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            _require(written > 0, "published write made no progress")
            offset += written
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        _require(
            stat.S_ISREG(sealed.st_mode)
            and (sealed.st_dev, sealed.st_ino) == (opened.st_dev, opened.st_ino)
            and sealed.st_uid == os.geteuid()
            and sealed.st_gid == os.getegid()
            and stat.S_IMODE(sealed.st_mode) == 0o400
            and sealed.st_nlink == 1
            and sealed.st_size == len(raw),
            "published opened-file identity differs",
        )
        os.close(descriptor)
        descriptor = None
        _fsync_directory(parent)
        replay = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            replay_stat = os.fstat(replay)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(replay, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(replay)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    observed = hashlib.sha256(raw).hexdigest()
    _require(
        (replay_stat.st_dev, replay_stat.st_ino) == (opened.st_dev, opened.st_ino)
        and replay_stat.st_nlink == 1
        and stat.S_IMODE(replay_stat.st_mode) == 0o400
        and b"".join(chunks) == raw
        and hashlib.sha256(b"".join(chunks)).hexdigest() == observed
        and path.resolve(strict=True) == path,
        f"published bytes failed durable replay: {path}",
    )
    return observed


def _read_cgroup_ascii(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise Reserve4GenerationError(
            f"live host cgroup {label} is unavailable"
        ) from error
    _require(bool(value), f"live host cgroup {label} is empty")
    return value


def _read_cgroup_nonnegative_integer(path: Path, label: str) -> int:
    value = _read_cgroup_ascii(path, label)
    _require(value.isdecimal(), f"live host cgroup {label} differs")
    return int(value)


def _read_cgroup_memory_events(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in _read_cgroup_ascii(path, "memory.events").splitlines():
        fields = line.split()
        _require(
            len(fields) == 2
            and fields[0] not in result
            and fields[1].isdecimal(),
            "live host cgroup memory.events differs",
        )
        result[fields[0]] = int(fields[1])
    _require(
        "oom" in result and "oom_kill" in result,
        "live host cgroup memory.events lacks oom counters",
    )
    return {"oom": result["oom"], "oom_kill": result["oom_kill"]}


# The AUH Slurm child exposes current/max/events but no kernel-maintained peak
# counter, so host admission is based on a continuous fixed-cadence sampler.
def _process_start_ticks(pid: int) -> int:
    _require(type(pid) is int and pid > 1, "host monitor PID differs")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = raw.rfind(")")
        _require(close > 0, "host monitor process identity differs")
        fields = raw[close + 2 :].split()
        value = int(fields[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise Reserve4GenerationError(
            "host monitor process identity is unavailable"
        ) from error
    _require(value > 0, "host monitor process start identity differs")
    return value


def _process_identity_is_live(pid: int, start_ticks: int) -> bool:
    try:
        os.kill(pid, 0)
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = raw.rfind(")")
        if close <= 0:
            return False
        fields = raw[close + 2 :].split()
        # kill(pid, 0) and /proc both continue to succeed for an unreaped
        # background child.  A dead monitor must never be admitted merely
        # because it is still present as a zombie in the launcher's job table.
        return fields[0] not in {"Z", "X", "x"} and int(fields[19]) == start_ticks
    except (OSError, UnicodeError, ValueError, IndexError):
        return False


def _discover_live_cgroup_v2(
    *,
    slurm_job_id: str,
    slurm_step_id: str,
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    _require(
        slurm_job_id == "136141" and slurm_step_id.isdecimal(),
        "live cgroup discovery Slurm job/step binding differs",
    )
    try:
        lines = proc_cgroup_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise Reserve4GenerationError(
            "cannot read the live process cgroup"
        ) from error
    unified = [
        fields[2]
        for line in lines
        if len(fields := line.split(":", 2)) == 3
        and fields[0] == "0"
        and fields[1] == ""
    ]
    relative = unified[0] if len(unified) == 1 else ""
    components = relative.split("/")[1:] if relative.startswith("/") else []
    _require(
        bool(components)
        and all(component not in {"", ".", ".."} for component in components)
        and relative == "/" + "/".join(components),
        "one canonical cgroup-v2 leaf path is required",
    )
    _require(
        cgroup_root.is_absolute(), "cgroup-v2 mount path must be absolute"
    )
    try:
        root_metadata = cgroup_root.lstat()
        root = cgroup_root.resolve(strict=True)
    except OSError as error:
        raise Reserve4GenerationError("cgroup-v2 mount is unavailable") from error
    _require(
        stat.S_ISDIR(root_metadata.st_mode)
        and not stat.S_ISLNK(root_metadata.st_mode)
        and root == cgroup_root,
        "cgroup-v2 mount identity differs",
    )
    leaf = root
    for component in components:
        leaf = leaf / component
        try:
            metadata = leaf.lstat()
        except OSError as error:
            raise Reserve4GenerationError(
                "live process cgroup leaf hierarchy is unavailable"
            ) from error
        _require(
            stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode),
            "live process cgroup hierarchy contains a non-directory/symlink",
        )
    _require(
        leaf.resolve(strict=True) == leaf and root in leaf.parents,
        "live process cgroup escaped the cgroup-v2 mount",
    )

    def read_ancestor_maximum(path: Path, label: str) -> str:
        maximum_path = path / "memory.max"
        try:
            maximum_path.lstat()
        except FileNotFoundError:
            _require(
                path == root,
                f"{label} memory.max is absent outside the cgroup root",
            )
            # The cgroup-v2 root cannot itself be constrained and on AUH has
            # no memory.max file.  Only this exact mount-root absence is the
            # implicit unbounded value; every non-root ancestor must expose
            # an authenticated plain memory.max file.
            return "max"
        except OSError as error:
            raise Reserve4GenerationError(
                f"{label} memory.max is unavailable"
            ) from error
        return _read_cgroup_ascii(
            _plain_file(maximum_path, f"{label} cgroup memory.max"),
            f"{label} memory.max",
        )

    leaf_max_path = _plain_file(
        leaf / "memory.max", "live leaf cgroup memory.max"
    )
    _require(
        _read_cgroup_ascii(leaf_max_path, "leaf memory.max") == "max",
        "live PID leaf cgroup must inherit memory.max from a strict ancestor",
    )

    governing: Optional[Path] = None
    governing_max: Optional[int] = None
    candidate = leaf.parent
    while candidate == root or root in candidate.parents:
        maximum_text = read_ancestor_maximum(candidate, "candidate governing")
        if maximum_text != "max":
            _require(
                maximum_text.isdecimal(),
                "candidate governing cgroup memory.max differs",
            )
            governing = candidate
            governing_max = int(maximum_text)
            break
        if candidate == root:
            break
        candidate = candidate.parent
    _require(
        governing is not None and governing_max is not None,
        "no finite governing cgroup ancestor exists",
    )
    outer = governing.parent
    while outer == root or root in outer.parents:
        outer_text = read_ancestor_maximum(outer, "outer governing")
        if outer_text != "max":
            _require(
                outer_text.isdecimal()
                and int(outer_text) >= HOST_MEMORY_LIMIT_BYTES,
                "outer cgroup imposes a host memory limit below 60 GiB",
            )
        if outer == root:
            break
        outer = outer.parent
    governing_components = governing.relative_to(root).parts
    expected_suffix = (
        f"job_{slurm_job_id}", f"step_{slurm_step_id}", "user"
    )
    _require(
        len(governing_components) >= len(expected_suffix)
        and tuple(governing_components[-3:]) == expected_suffix
        and governing in leaf.parents
        and governing_max == HOST_MEMORY_LIMIT_BYTES,
        "nearest finite governing ancestor is not exact Slurm step-user 60 GiB",
    )
    leaf_metadata = leaf.lstat()
    governing_metadata = governing.lstat()
    _require(
        stat.S_ISDIR(leaf_metadata.st_mode)
        and not stat.S_ISLNK(leaf_metadata.st_mode)
        and stat.S_ISDIR(governing_metadata.st_mode)
        and not stat.S_ISLNK(governing_metadata.st_mode)
        and leaf.resolve(strict=True) == leaf
        and governing.resolve(strict=True) == governing
        and leaf_metadata.st_dev == governing_metadata.st_dev
        and governing_metadata.st_dev == root_metadata.st_dev
        and leaf_metadata.st_ino > 0
        and governing_metadata.st_ino > 0,
        "live cgroup leaf/governing identity changed during discovery",
    )
    governing_relative = "/" + "/".join(governing_components)
    files = {
        "memory_current": str(governing / "memory.current"),
        "memory_max": str(governing / "memory.max"),
        "memory_events": str(governing / "memory.events"),
    }
    for label, path in files.items():
        _plain_file(path, f"live cgroup {label}")
    return {
        "cgroup_version": 2,
        "leaf_cgroup": {
            "relative_path": relative,
            "path": str(leaf),
            "device": leaf_metadata.st_dev,
            "inode": leaf_metadata.st_ino,
            "memory_max": "max",
        },
        "governing_cgroup": {
            "relative_path": governing_relative,
            "path": str(governing),
            "device": governing_metadata.st_dev,
            "inode": governing_metadata.st_ino,
            "memory_max_bytes": governing_max,
            "slurm_job_id": slurm_job_id,
            "slurm_step_id": slurm_step_id,
            "scope": "user",
        },
        "measurement_files": files,
    }


def _sample_live_cgroup_memory(
    binding: Mapping[str, Any], *, sequence: int
) -> tuple[int, int, int, int, int, int, int]:
    files = binding["measurement_files"]
    sample = (
        sequence,
        time.time_ns(),
        time.monotonic_ns(),
        _read_cgroup_nonnegative_integer(
            Path(files["memory_current"]), "memory.current"
        ),
        _read_cgroup_nonnegative_integer(Path(files["memory_max"]), "memory.max"),
    )
    events = _read_cgroup_memory_events(Path(files["memory_events"]))
    _require(
        sample[4] == HOST_MEMORY_LIMIT_BYTES,
        "live Slurm child cgroup memory.max is not exact 60 GiB",
    )
    _require(
        sample[3] < HOST_MEMORY_SAFE_CEILING_BYTES,
        "live sampled memory.current reached the strict 56-GiB safe ceiling",
    )
    _require(
        events == {"oom": 0, "oom_kill": 0},
        "live Slurm child cgroup recorded OOM activity",
    )
    return (*sample, events["oom"], events["oom_kill"])


def _sample_row(values: Sequence[int]) -> dict[str, Any]:
    _require(len(values) in {7, 8}, "host memory sample width differs")
    kind = int(values[7]) if len(values) == 8 else 0
    _require(kind in {0, 1}, "host memory sample kind differs")
    sample_kind = "periodic" if kind == 0 else "stop_final"
    return {
        "sequence": int(values[0]),
        "wall_time_ns": int(values[1]),
        "monotonic_time_ns": int(values[2]),
        "memory_current_bytes": int(values[3]),
        "memory_max_bytes": int(values[4]),
        "memory_events": {"oom": int(values[5]), "oom_kill": int(values[6])},
        "sample_kind": sample_kind,
    }


def _canonical_relative_cgroup_parts(value: Any) -> tuple[str, ...]:
    _require(isinstance(value, str) and value.startswith("/"),
             "cgroup relative path differs")
    parts = tuple(value.split("/")[1:])
    _require(
        bool(parts)
        and all(part not in {"", ".", ".."} for part in parts)
        and value == "/" + "/".join(parts),
        "cgroup relative path is not canonical",
    )
    return parts


def _validate_cgroup_binding_descriptor(
    value: Mapping[str, Any],
) -> None:
    leaf = value.get("leaf_cgroup")
    governing = value.get("governing_cgroup")
    _require(
        value.get("cgroup_version") == 2
        and isinstance(leaf, Mapping)
        and set(leaf)
        == {"relative_path", "path", "device", "inode", "memory_max"}
        and isinstance(governing, Mapping)
        and set(governing)
        == {
            "relative_path", "path", "device", "inode",
            "memory_max_bytes", "slurm_job_id", "slurm_step_id", "scope",
        },
        "host memory cgroup leaf/governing descriptor differs",
    )
    leaf_parts = _canonical_relative_cgroup_parts(leaf["relative_path"])
    governing_parts = _canonical_relative_cgroup_parts(
        governing["relative_path"]
    )
    leaf_path = Path(str(leaf.get("path")))
    governing_path = Path(str(governing.get("path")))
    expected_suffix = (
        f"job_{value.get('slurm_job_id')}",
        f"step_{value.get('slurm_step_id')}",
        "user",
    )
    _require(
        leaf_path.is_absolute()
        and governing_path.is_absolute()
        and governing_path in leaf_path.parents
        and len(leaf_parts) > len(governing_parts)
        and leaf_parts[: len(governing_parts)] == governing_parts
        and tuple(governing_parts[-3:]) == expected_suffix
        and tuple(leaf_path.parts[-len(leaf_parts):]) == leaf_parts
        and tuple(governing_path.parts[-len(governing_parts):])
        == governing_parts
        and type(leaf.get("device")) is int
        and int(leaf["device"]) >= 0
        and type(leaf.get("inode")) is int
        and int(leaf["inode"]) > 0
        and type(governing.get("device")) is int
        and int(governing["device"]) == int(leaf["device"])
        and type(governing.get("inode")) is int
        and int(governing["inode"]) > 0
        and leaf.get("memory_max") == "max"
        and governing.get("memory_max_bytes") == HOST_MEMORY_LIMIT_BYTES
        and governing.get("slurm_job_id") == value.get("slurm_job_id")
        and governing.get("slurm_step_id") == value.get("slurm_step_id")
        and governing.get("scope") == "user"
        and value.get("measurement_files")
        == {
            "memory_current": str(governing_path / "memory.current"),
            "memory_max": str(governing_path / "memory.max"),
            "memory_events": str(governing_path / "memory.events"),
        },
        "host memory cgroup nearest finite Slurm step-user binding differs",
    )


def _validate_monitor_stop_capability(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "kind",
            "read_fd_at_start",
            "device",
            "inode",
            "uid",
            "gid",
            "mode",
            "link_count",
            "terminal_token_sha256",
            "named_filesystem_stop_token",
        }
        and value.get("kind") == "anonymous-pipe-read-end"
        and type(value.get("read_fd_at_start")) is int
        and int(value["read_fd_at_start"]) >= 3
        and type(value.get("device")) is int
        and int(value["device"]) >= 0
        and type(value.get("inode")) is int
        and int(value["inode"]) > 0
        and value.get("uid") == 2012
        and value.get("gid") == 2000
        and type(value.get("mode")) is int
        and (int(value["mode"]) & 0o007) == 0
        and type(value.get("link_count")) is int
        and int(value["link_count"]) >= 0
        and value.get("terminal_token_sha256")
        == HOST_MEMORY_MONITOR_STOP_TOKEN_SHA256
        and value.get("named_filesystem_stop_token") is False,
        "host memory monitor anonymous stop-capability binding differs",
    )
    return dict(value)


def validate_host_cgroup_memory_monitor_start(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Reserve4GenerationError(
            "host cgroup memory monitor start receipt is absent"
        )
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    expected_fields = {
        "schema_version", "monitor_pid", "monitor_proc_start_ticks",
        "supervisor_pid", "supervisor_proc_start_ticks", "slurm_job_id",
        "slurm_step_id", "monitor_started_before_compile_smoke_and_formal40",
        "cgroup_version", "leaf_cgroup", "governing_cgroup",
        "measurement_files", "sampling_source", "sample_journal",
        "stop_capability",
        "sample_interval_ns", "maximum_sample_gap_ns",
        "strict_host_memory_limit_gib", "strict_host_memory_limit_bytes",
        "host_memory_safe_ceiling_gib", "host_memory_safe_ceiling_bytes",
        "initial_sample", "receipt_digest",
    }
    _require(
        set(value) == expected_fields
        and value.get("schema_version")
        == HOST_CGROUP_MEMORY_MONITOR_START_SCHEMA
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and declared == object_sha256(unsigned),
        "host cgroup memory monitor start schema/digest differs",
    )
    _require(
        all(
            type(value.get(field)) is int and int(value[field]) > 1
            for field in ("monitor_pid", "supervisor_pid")
        )
        and all(
            type(value.get(field)) is int and int(value[field]) > 0
            for field in ("monitor_proc_start_ticks", "supervisor_proc_start_ticks")
        )
        and value.get("slurm_job_id") == "136141"
        and isinstance(value.get("slurm_step_id"), str)
        and str(value["slurm_step_id"]).isdecimal()
        and value.get("monitor_started_before_compile_smoke_and_formal40") is True,
        "host memory monitor Slurm/PID binding differs",
    )
    _validate_cgroup_binding_descriptor(value)
    _require(
        value.get("sampling_source")
        == "cgroup_v2_memory.current_fixed_10ms",
        "host memory monitor cgroup-v2 binding differs",
    )
    journal = value.get("sample_journal")
    stop_capability = value.get("stop_capability")
    _require(
        isinstance(journal, Mapping)
        and set(journal)
        == {"path", "device", "inode", "record_size", "record_encoding"}
        and Path(str(journal.get("path"))).is_absolute()
        and type(journal.get("device")) is int
        and int(journal["device"]) >= 0
        and type(journal.get("inode")) is int
        and int(journal["inode"]) > 0
        and journal.get("record_size") == HOST_MEMORY_SAMPLE_STRUCT.size
        and journal.get("record_encoding") == HOST_MEMORY_SAMPLE_ENCODING,
        "host memory monitor journal binding differs",
    )
    _validate_monitor_stop_capability(stop_capability)
    sample = value.get("initial_sample")
    _require(
        isinstance(sample, Mapping)
        and set(sample)
        == {"sequence", "wall_time_ns", "monotonic_time_ns",
            "memory_current_bytes", "memory_max_bytes", "memory_events",
            "sample_kind"}
        and sample.get("sequence") == 0
        and type(sample.get("wall_time_ns")) is int
        and int(sample["wall_time_ns"]) > 0
        and type(sample.get("monotonic_time_ns")) is int
        and int(sample["monotonic_time_ns"]) > 0
        and type(sample.get("memory_current_bytes")) is int
        and 0 <= int(sample["memory_current_bytes"]) < HOST_MEMORY_SAFE_CEILING_BYTES
        and sample.get("memory_max_bytes") == HOST_MEMORY_LIMIT_BYTES
        and sample.get("memory_events") == {"oom": 0, "oom_kill": 0}
        and sample.get("sample_kind") == "periodic"
        and value.get("sample_interval_ns") == HOST_MEMORY_SAMPLE_INTERVAL_NS
        and value.get("maximum_sample_gap_ns") == HOST_MEMORY_MAX_SAMPLE_GAP_NS
        and value.get("strict_host_memory_limit_gib") == HOST_MEMORY_LIMIT_GIB
        and value.get("strict_host_memory_limit_bytes") == HOST_MEMORY_LIMIT_BYTES
        and value.get("host_memory_safe_ceiling_gib")
        == HOST_MEMORY_SAFE_CEILING_GIB
        and value.get("host_memory_safe_ceiling_bytes")
        == HOST_MEMORY_SAFE_CEILING_BYTES,
        "host memory monitor initial sample/safety contract failed",
    )
    return dict(value)


def load_host_cgroup_memory_monitor_start(
    path: str | Path, expected_sha256: Optional[str] = None
) -> tuple[dict[str, Any], Path, str]:
    value, source, observed = _load_json(
        path, "host memory monitor start receipt", expected_sha256
    )
    _require(
        source.read_bytes() == canonical_json_bytes(value) + b"\n",
        "host memory monitor start receipt is not canonical JSON",
    )
    return validate_host_cgroup_memory_monitor_start(value), source, observed


def _assert_live_cgroup_binding_matches_start(
    start: Mapping[str, Any],
) -> dict[str, Any]:
    live = _discover_live_cgroup_v2(
        slurm_job_id=start["slurm_job_id"],
        slurm_step_id=start["slurm_step_id"],
    )
    expected = {
        "cgroup_version": start["cgroup_version"],
        "leaf_cgroup": start["leaf_cgroup"],
        "governing_cgroup": start["governing_cgroup"],
        "measurement_files": start["measurement_files"],
    }
    _require(
        live == expected,
        "live process leaf/governing cgroup binding drifted",
    )
    return live


def _journal_prefix(
    start: Mapping[str, Any], *, exact_terminal_size: bool
) -> tuple[bytes, list[tuple[int, ...]], os.stat_result, int]:
    journal_path = _plain_file(
        start["sample_journal"]["path"], "host memory sample journal"
    )
    metadata = journal_path.stat()
    _require(
        metadata.st_dev == start["sample_journal"]["device"]
        and metadata.st_ino == start["sample_journal"]["inode"]
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) in {0o600, 0o400},
        "host memory sample journal identity differs",
    )
    with journal_path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        fixed_size = opened.st_size
        observed_monotonic_ns = time.monotonic_ns()
        raw = handle.read(fixed_size)
        final_size = os.fstat(handle.fileno()).st_size
    _require(
        (opened.st_dev, opened.st_ino) == (metadata.st_dev, metadata.st_ino)
        and raw
        and len(raw) == fixed_size
        and fixed_size % HOST_MEMORY_SAMPLE_STRUCT.size == 0
        and final_size >= fixed_size,
        "host memory sample journal is truncated",
    )
    rows = [
        HOST_MEMORY_SAMPLE_STRUCT.unpack_from(raw, offset)
        for offset in range(0, len(raw), HOST_MEMORY_SAMPLE_STRUCT.size)
    ]
    if exact_terminal_size:
        _require(
            stat.S_IMODE(metadata.st_mode) == 0o400
            and stat.S_IMODE(opened.st_mode) == 0o400
            and final_size == fixed_size,
            "terminal host memory sample journal is not sealed read-only",
        )
    return raw, rows, metadata, observed_monotonic_ns


def _assert_fresh_live_journal_tail(
    start: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the newest two records without making a long prefix scan stale."""

    journal_path = _plain_file(
        start["sample_journal"]["path"], "host memory sample journal"
    )
    metadata = journal_path.stat()
    record_size = HOST_MEMORY_SAMPLE_STRUCT.size
    _require(
        metadata.st_dev == start["sample_journal"]["device"]
        and metadata.st_ino == start["sample_journal"]["inode"]
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o600,
        "live host memory sample journal identity differs",
    )
    with journal_path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        fixed_size = opened.st_size
        _require(
            fixed_size >= 2 * record_size and fixed_size % record_size == 0,
            "live host memory sample journal is truncated",
        )
        handle.seek(fixed_size - 2 * record_size)
        raw = handle.read(2 * record_size)
        observed_monotonic_ns = time.monotonic_ns()
    _require(
        (opened.st_dev, opened.st_ino)
        == (metadata.st_dev, metadata.st_ino)
        and opened.st_nlink == 1
        and stat.S_IMODE(opened.st_mode) == 0o600
        and len(raw) == 2 * record_size,
        "live host memory sample journal tail replay differs",
    )
    previous = _sample_row(HOST_MEMORY_SAMPLE_STRUCT.unpack_from(raw, 0))
    latest = _sample_row(
        HOST_MEMORY_SAMPLE_STRUCT.unpack_from(raw, record_size)
    )
    tail_age_ns = observed_monotonic_ns - int(latest["monotonic_time_ns"])
    _require(
        latest["sequence"] == previous["sequence"] + 1
        and previous["sample_kind"] == "periodic"
        and latest["sample_kind"] == "periodic"
        and 0
        < int(latest["monotonic_time_ns"])
        - int(previous["monotonic_time_ns"])
        <= HOST_MEMORY_MAX_SAMPLE_GAP_NS
        and 0 <= tail_age_ns <= HOST_MEMORY_MAX_SAMPLE_GAP_NS
        and int(latest["memory_current_bytes"])
        < HOST_MEMORY_SAFE_CEILING_BYTES
        and latest["memory_max_bytes"] == HOST_MEMORY_LIMIT_BYTES
        and latest["memory_events"] == {"oom": 0, "oom_kill": 0},
        "live host memory journal tail/cadence/safety gate failed",
    )
    return {
        "sequence": latest["sequence"],
        "monotonic_time_ns": latest["monotonic_time_ns"],
        "observed_monotonic_time_ns": observed_monotonic_ns,
        "observed_tail_age_ns": tail_age_ns,
    }


def _derive_host_cgroup_memory_gate(
    *,
    start_receipt: Mapping[str, Any],
    start_receipt_path: Path,
    start_receipt_sha256: str,
    raw_prefix: bytes,
    rows: Sequence[Sequence[int]],
    measurement_phase: str,
    formal_candidate_count_at_gate: int,
    live_tail_observed_monotonic_time_ns: Optional[int] = None,
) -> dict[str, Any]:
    phase_count = (measurement_phase, formal_candidate_count_at_gate)
    _require(
        phase_count
        in {
            ("compile_smoke_before_formal40", 0),
            ("terminal_after_formal40_before_slurm_child_exit", 40),
        },
        "host cgroup memory gate phase/formal count differs",
    )
    _require(len(rows) >= 2, "host memory monitor has too few samples")
    samples = [_sample_row(row) for row in rows]
    _require(
        samples[0] == start_receipt["initial_sample"],
        "host memory monitor first sample/start receipt differs",
    )
    _require(
        [sample["sequence"] for sample in samples] == list(range(len(samples))),
        "host memory sample sequence is not exact monotonic contiguous",
    )
    terminal = measurement_phase.startswith("terminal_")
    sample_kinds = [sample["sample_kind"] for sample in samples]
    _require(
        (
            terminal
            and sample_kinds[-1] == "stop_final"
            and all(kind == "periodic" for kind in sample_kinds[:-1])
        )
        or (
            not terminal
            and all(kind == "periodic" for kind in sample_kinds)
        ),
        "host memory stop/final sample marker differs",
    )
    wall = [int(sample["wall_time_ns"]) for sample in samples]
    monotonic = [int(sample["monotonic_time_ns"]) for sample in samples]
    _require(
        all(value > 0 for value in wall)
        and all(b > a for a, b in zip(monotonic, monotonic[1:])),
        "host memory sample monotonic timestamps are not strictly increasing",
    )
    gaps = [b - a for a, b in zip(monotonic, monotonic[1:])]
    maximum_gap = max(gaps)
    _require(
        maximum_gap <= HOST_MEMORY_MAX_SAMPLE_GAP_NS,
        "host memory monitor cadence gap exceeded 100 ms",
    )
    currents = [int(sample["memory_current_bytes"]) for sample in samples]
    _require(
        max(currents) < HOST_MEMORY_SAFE_CEILING_BYTES
        and all(
            sample["memory_max_bytes"] == HOST_MEMORY_LIMIT_BYTES
            for sample in samples
        )
        and all(
            sample["memory_events"] == {"oom": 0, "oom_kill": 0}
            for sample in samples
        ),
        "host memory sampled ceiling/max/OOM contract failed",
    )
    if terminal:
        _require(
            live_tail_observed_monotonic_time_ns is None,
            "terminal host gate must not carry a live-tail observation",
        )
        live_tail_observed = None
        live_tail_age = None
        live_tail_fresh = None
    else:
        live_tail_observed = live_tail_observed_monotonic_time_ns
        _require(
            type(live_tail_observed) is int,
            "nonterminal host gate requires an explicit live-tail observation",
        )
        live_tail_age = int(live_tail_observed) - monotonic[-1]
        _require(
            0 <= live_tail_age <= HOST_MEMORY_MAX_SAMPLE_GAP_NS,
            "host memory monitor live-tail age exceeded 100 ms",
        )
        live_tail_fresh = True
    unsigned = {
        "schema_version": HOST_CGROUP_MEMORY_GATE_SCHEMA,
        "measurement_phase": measurement_phase,
        "formal_candidate_count_at_gate": formal_candidate_count_at_gate,
        "monitor_start_receipt": {
            "path": str(start_receipt_path),
            "file_sha256": start_receipt_sha256,
            "receipt_digest": start_receipt["receipt_digest"],
        },
        "monitor_pid": start_receipt["monitor_pid"],
        "monitor_proc_start_ticks": start_receipt["monitor_proc_start_ticks"],
        "supervisor_pid": start_receipt["supervisor_pid"],
        "supervisor_proc_start_ticks": start_receipt["supervisor_proc_start_ticks"],
        "slurm_job_id": start_receipt["slurm_job_id"],
        "slurm_step_id": start_receipt["slurm_step_id"],
        "monitor_started_before_compile_smoke_and_formal40": True,
        "monitor_state_at_gate": (
            "clean_terminal_stop_after_formal40"
            if terminal
            else "alive_before_formal40"
        ),
        "monitor_exit_status": 0 if terminal else None,
        "monitor_identity_dead_at_gate": terminal,
        "terminal_gate_created_after_bound_supervisor_wait": terminal,
        "monitor_stop_capability": start_receipt["stop_capability"],
        "live_tail_observed_monotonic_time_ns": live_tail_observed,
        "observed_tail_age_ns": live_tail_age,
        "observed_tail_age_within_100_ms": live_tail_fresh,
        "cgroup_version": 2,
        "leaf_cgroup": start_receipt["leaf_cgroup"],
        "governing_cgroup": start_receipt["governing_cgroup"],
        "measurement_files": start_receipt["measurement_files"],
        "sampling_source": "cgroup_v2_memory.current_fixed_10ms",
        "sample_journal": {
            **start_receipt["sample_journal"],
            "prefix_byte_count": len(raw_prefix),
            "prefix_sha256": hashlib.sha256(raw_prefix).hexdigest(),
            "terminal_journal_sealed_read_only": terminal,
        },
        "sampling": {
            "requested_interval_ns": HOST_MEMORY_SAMPLE_INTERVAL_NS,
            "maximum_allowed_gap_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
            "sample_count": len(samples),
            "first_sequence": 0,
            "last_sequence": len(samples) - 1,
            "start_wall_time_ns": wall[0],
            "end_wall_time_ns": wall[-1],
            "start_monotonic_time_ns": monotonic[0],
            "end_monotonic_time_ns": monotonic[-1],
            "duration_ns": monotonic[-1] - monotonic[0],
            "observed_maximum_gap_ns": maximum_gap,
            "monotonic_timestamps_strictly_increasing": True,
            "wall_clock_timestamps_informational": True,
            "all_sample_gaps_within_100_ms": True,
        },
        "sampled_peak_memory_current_bytes": max(currents),
        "memory_current_bytes_at_gate": currents[-1],
        "host_memory_safe_ceiling_gib": HOST_MEMORY_SAFE_CEILING_GIB,
        "host_memory_safe_ceiling_bytes": HOST_MEMORY_SAFE_CEILING_BYTES,
        "sampled_peak_strictly_below_56_gib": True,
        "strict_host_memory_limit_gib": HOST_MEMORY_LIMIT_GIB,
        "strict_host_memory_limit_bytes": HOST_MEMORY_LIMIT_BYTES,
        "cgroup_memory_max_bytes": HOST_MEMORY_LIMIT_BYTES,
        "cgroup_memory_max_exactly_60_gib": True,
        "memory_events_at_start": {"oom": 0, "oom_kill": 0},
        "memory_events_at_gate": {"oom": 0, "oom_kill": 0},
        "all_samples_zero_oom_and_oom_kill": True,
        "monitor_alive_at_gate": not terminal,
        "monitor_clean_terminal_stop": terminal,
        "monitor_covered_compile_smoke_through_formal40": terminal,
        "terminal_stop_observed_before_final_sample": terminal,
        "terminal_stop_final_sample_sequence": (
            samples[-1]["sequence"] if terminal else None
        ),
        "terminal_stop_final_sample_monotonic_time_ns": (
            monotonic[-1] if terminal else None
        ),
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def validate_host_cgroup_memory_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one self-digested sampled-current gate structurally."""

    if not isinstance(value, Mapping):
        raise Reserve4GenerationError("host cgroup memory gate is absent")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    expected_fields = {
        "schema_version", "measurement_phase", "formal_candidate_count_at_gate",
        "monitor_start_receipt", "monitor_pid", "monitor_proc_start_ticks",
        "supervisor_pid", "supervisor_proc_start_ticks", "slurm_job_id",
        "slurm_step_id", "monitor_started_before_compile_smoke_and_formal40",
        "monitor_state_at_gate", "cgroup_version", "leaf_cgroup",
        "governing_cgroup", "measurement_files", "sampling_source",
        "sample_journal", "sampling", "sampled_peak_memory_current_bytes",
        "memory_current_bytes_at_gate", "host_memory_safe_ceiling_gib",
        "host_memory_safe_ceiling_bytes", "sampled_peak_strictly_below_56_gib",
        "strict_host_memory_limit_gib", "strict_host_memory_limit_bytes",
        "cgroup_memory_max_bytes", "cgroup_memory_max_exactly_60_gib",
        "memory_events_at_start", "memory_events_at_gate",
        "all_samples_zero_oom_and_oom_kill", "monitor_alive_at_gate",
        "monitor_clean_terminal_stop",
        "monitor_exit_status", "monitor_identity_dead_at_gate",
        "terminal_gate_created_after_bound_supervisor_wait",
        "monitor_stop_capability",
        "live_tail_observed_monotonic_time_ns", "observed_tail_age_ns",
        "observed_tail_age_within_100_ms",
        "monitor_covered_compile_smoke_through_formal40", "receipt_digest",
        "terminal_stop_observed_before_final_sample",
        "terminal_stop_final_sample_sequence",
        "terminal_stop_final_sample_monotonic_time_ns",
    }
    _require(
        set(value) == expected_fields
        and value.get("schema_version") == HOST_CGROUP_MEMORY_GATE_SCHEMA
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and declared == object_sha256(unsigned),
        "host cgroup sampled-memory gate schema/digest differs",
    )
    terminal = (
        value.get("measurement_phase")
        == "terminal_after_formal40_before_slurm_child_exit"
    )
    _require(
        (value.get("measurement_phase"), value.get("formal_candidate_count_at_gate"))
        in {
            ("compile_smoke_before_formal40", 0),
            ("terminal_after_formal40_before_slurm_child_exit", 40),
        }
        and value.get("monitor_state_at_gate")
        == ("clean_terminal_stop_after_formal40" if terminal else "alive_before_formal40")
        and value.get("monitor_alive_at_gate") is (not terminal)
        and value.get("monitor_clean_terminal_stop") is terminal
        and value.get("monitor_exit_status") == (0 if terminal else None)
        and value.get("monitor_identity_dead_at_gate") is terminal
        and value.get("terminal_gate_created_after_bound_supervisor_wait")
        is terminal
        and isinstance(value.get("monitor_stop_capability"), Mapping)
        and value["monitor_stop_capability"].get("kind")
        == "anonymous-pipe-read-end"
        and value["monitor_stop_capability"].get("named_filesystem_stop_token")
        is False
        and value.get("terminal_stop_observed_before_final_sample") is terminal
        and (
            (
                terminal
                and value.get("live_tail_observed_monotonic_time_ns") is None
                and value.get("observed_tail_age_ns") is None
                and value.get("observed_tail_age_within_100_ms") is None
            )
            or (
                not terminal
                and type(value.get("live_tail_observed_monotonic_time_ns"))
                is int
                and type(value.get("observed_tail_age_ns")) is int
                and 0 <= int(value["observed_tail_age_ns"])
                <= HOST_MEMORY_MAX_SAMPLE_GAP_NS
                and value.get("observed_tail_age_within_100_ms") is True
            )
        )
        and value.get("monitor_covered_compile_smoke_through_formal40") is terminal,
        "host cgroup sampled-memory phase/monitor state differs",
    )
    _validate_monitor_stop_capability(value.get("monitor_stop_capability"))
    _validate_cgroup_binding_descriptor(value)
    start_ref = value.get("monitor_start_receipt")
    journal = value.get("sample_journal")
    sampling = value.get("sampling")
    _require(
        isinstance(start_ref, Mapping)
        and set(start_ref) == {"path", "file_sha256", "receipt_digest"}
        and Path(str(start_ref.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(start_ref.get("file_sha256"))) is not None
        and SHA256_RE.fullmatch(str(start_ref.get("receipt_digest"))) is not None
        and isinstance(journal, Mapping)
        and set(journal)
        == {"path", "device", "inode", "record_size", "record_encoding",
            "prefix_byte_count", "prefix_sha256",
            "terminal_journal_sealed_read_only"}
        and Path(str(journal.get("path"))).is_absolute()
        and type(journal.get("device")) is int
        and type(journal.get("inode")) is int
        and journal.get("record_size") == HOST_MEMORY_SAMPLE_STRUCT.size
        and journal.get("record_encoding") == HOST_MEMORY_SAMPLE_ENCODING
        and type(journal.get("prefix_byte_count")) is int
        and int(journal["prefix_byte_count"]) >= 2 * HOST_MEMORY_SAMPLE_STRUCT.size
        and int(journal["prefix_byte_count"]) % HOST_MEMORY_SAMPLE_STRUCT.size == 0
        and SHA256_RE.fullmatch(str(journal.get("prefix_sha256"))) is not None
        and journal.get("terminal_journal_sealed_read_only") is terminal
        and isinstance(sampling, Mapping)
        and set(sampling)
        == {"requested_interval_ns", "maximum_allowed_gap_ns", "sample_count",
            "first_sequence", "last_sequence", "start_wall_time_ns",
            "end_wall_time_ns", "start_monotonic_time_ns",
            "end_monotonic_time_ns", "duration_ns", "observed_maximum_gap_ns",
            "monotonic_timestamps_strictly_increasing",
            "wall_clock_timestamps_informational",
            "all_sample_gaps_within_100_ms"}
        and sampling.get("requested_interval_ns") == HOST_MEMORY_SAMPLE_INTERVAL_NS
        and sampling.get("maximum_allowed_gap_ns") == HOST_MEMORY_MAX_SAMPLE_GAP_NS
        and type(sampling.get("sample_count")) is int
        and int(sampling["sample_count"]) >= 2
        and sampling.get("first_sequence") == 0
        and sampling.get("last_sequence") == int(sampling["sample_count"]) - 1
        and all(
            type(sampling.get(field)) is int
            for field in (
                "start_wall_time_ns",
                "end_wall_time_ns",
                "start_monotonic_time_ns",
                "end_monotonic_time_ns",
                "duration_ns",
            )
        )
        and sampling.get("monotonic_timestamps_strictly_increasing") is True
        and sampling.get("wall_clock_timestamps_informational") is True
        and sampling.get("all_sample_gaps_within_100_ms") is True
        and value.get("terminal_stop_final_sample_sequence")
        == (sampling.get("last_sequence") if terminal else None)
        and value.get("terminal_stop_final_sample_monotonic_time_ns")
        == (sampling.get("end_monotonic_time_ns") if terminal else None)
        and (
            terminal
            or int(value["live_tail_observed_monotonic_time_ns"])
            - int(sampling["end_monotonic_time_ns"])
            == int(value["observed_tail_age_ns"])
        )
        and type(sampling.get("observed_maximum_gap_ns")) is int
        and 0 < int(sampling["observed_maximum_gap_ns"])
        <= HOST_MEMORY_MAX_SAMPLE_GAP_NS,
        "host cgroup sampled-memory receipt/journal/cadence differs",
    )
    current = value.get("memory_current_bytes_at_gate")
    peak = value.get("sampled_peak_memory_current_bytes")
    _require(
        type(current) is int
        and 0 <= int(current) < HOST_MEMORY_SAFE_CEILING_BYTES
        and type(peak) is int
        and int(current) <= int(peak) < HOST_MEMORY_SAFE_CEILING_BYTES
        and value.get("host_memory_safe_ceiling_gib") == HOST_MEMORY_SAFE_CEILING_GIB
        and value.get("host_memory_safe_ceiling_bytes") == HOST_MEMORY_SAFE_CEILING_BYTES
        and value.get("sampled_peak_strictly_below_56_gib") is True
        and value.get("strict_host_memory_limit_gib") == HOST_MEMORY_LIMIT_GIB
        and value.get("strict_host_memory_limit_bytes") == HOST_MEMORY_LIMIT_BYTES
        and value.get("cgroup_memory_max_bytes") == HOST_MEMORY_LIMIT_BYTES
        and value.get("cgroup_memory_max_exactly_60_gib") is True
        and value.get("sampling_source")
        == "cgroup_v2_memory.current_fixed_10ms"
        and value.get("memory_events_at_start") == {"oom": 0, "oom_kill": 0}
        and value.get("memory_events_at_gate") == {"oom": 0, "oom_kill": 0}
        and value.get("all_samples_zero_oom_and_oom_kill") is True
        and value.get("monitor_started_before_compile_smoke_and_formal40") is True
        and value.get("slurm_job_id") == "136141"
        and isinstance(value.get("slurm_step_id"), str)
        and str(value["slurm_step_id"]).isdecimal(),
        "host cgroup sampled-current safety gate failed",
    )
    return dict(value)


def load_host_cgroup_memory_gate(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_phase: str,
    require_monitor_alive_now: bool,
    require_bound_cgroup_now: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    value, source, observed = _load_json(
        path, "host cgroup memory gate", expected_sha256
    )
    _require(
        source.read_bytes() == canonical_json_bytes(value) + b"\n",
        "host cgroup memory gate bytes are not canonical JSON",
    )
    validated = validate_host_cgroup_memory_gate(value)
    _require(
        validated["measurement_phase"] == expected_phase,
        "host memory gate phase differs",
    )
    start_ref = validated["monitor_start_receipt"]
    start, start_path, start_sha = load_host_cgroup_memory_monitor_start(
        start_ref["path"], start_ref["file_sha256"]
    )
    _require(
        start_path == Path(start_ref["path"])
        and start_sha == start_ref["file_sha256"]
        and start["receipt_digest"] == start_ref["receipt_digest"],
        "host memory monitor start receipt replay differs",
    )
    journal_path = _plain_file(
        start["sample_journal"]["path"], "host memory sample journal"
    )
    metadata = journal_path.stat()
    prefix_count = int(validated["sample_journal"]["prefix_byte_count"])
    _require(
        metadata.st_dev == validated["sample_journal"]["device"]
        and metadata.st_ino == validated["sample_journal"]["inode"]
        and metadata.st_nlink == 1
        and metadata.st_size >= prefix_count,
        "host memory journal identity/length differs",
    )
    with journal_path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        raw = handle.read(prefix_count)
        final_size = os.fstat(handle.fileno()).st_size
    _require(
        (opened.st_dev, opened.st_ino)
        == (metadata.st_dev, metadata.st_ino)
        and opened.st_nlink == 1
        and stat.S_IMODE(opened.st_mode) in {0o600, 0o400}
        and len(raw) == prefix_count
        and hashlib.sha256(raw).hexdigest()
        == validated["sample_journal"]["prefix_sha256"],
        "host memory journal prefix replay differs",
    )
    rows = [
        HOST_MEMORY_SAMPLE_STRUCT.unpack_from(raw, offset)
        for offset in range(0, len(raw), HOST_MEMORY_SAMPLE_STRUCT.size)
    ]
    expected = _derive_host_cgroup_memory_gate(
        start_receipt=start,
        start_receipt_path=start_path,
        start_receipt_sha256=start_sha,
        raw_prefix=raw,
        rows=rows,
        measurement_phase=validated["measurement_phase"],
        formal_candidate_count_at_gate=validated["formal_candidate_count_at_gate"],
        live_tail_observed_monotonic_time_ns=validated[
            "live_tail_observed_monotonic_time_ns"
        ],
    )
    _require(
        expected == validated,
        "host memory gate resigned/replayed fields differ",
    )
    terminal = expected_phase.startswith("terminal_")
    _require(
        not terminal
        or (
            final_size == prefix_count
            and stat.S_IMODE(metadata.st_mode) == 0o400
            and stat.S_IMODE(opened.st_mode) == 0o400
        ),
        "terminal host memory journal is not exact sealed closure",
    )
    if terminal:
        _require(
            not _process_identity_is_live(
                int(start["monitor_pid"]),
                int(start["monitor_proc_start_ticks"]),
            ),
            "terminal host memory gate was replayed while monitor is still live",
        )
    if require_monitor_alive_now:
        _require(
            not terminal
            and _process_identity_is_live(
                int(start["monitor_pid"]), int(start["monitor_proc_start_ticks"])
            )
            and _process_identity_is_live(
                int(start["supervisor_pid"]), int(start["supervisor_proc_start_ticks"])
            ),
            "host memory monitor/supervisor is not live at formal admission",
        )
        live_raw, live_rows, _, tail_observed = _journal_prefix(
            start, exact_terminal_size=False
        )
        _derive_host_cgroup_memory_gate(
            start_receipt=start,
            start_receipt_path=start_path,
            start_receipt_sha256=start_sha,
            raw_prefix=live_raw,
            rows=live_rows,
            measurement_phase="compile_smoke_before_formal40",
            formal_candidate_count_at_gate=0,
            live_tail_observed_monotonic_time_ns=tail_observed,
        )
        _assert_fresh_live_journal_tail(start)
        _require(
            _process_identity_is_live(
                int(start["monitor_pid"]),
                int(start["monitor_proc_start_ticks"]),
            )
            and _process_identity_is_live(
                int(start["supervisor_pid"]),
                int(start["supervisor_proc_start_ticks"]),
            ),
            "host memory monitor/supervisor died during formal admission replay",
        )
    if require_monitor_alive_now or require_bound_cgroup_now:
        _assert_live_cgroup_binding_matches_start(start)
    return validated, source, observed


def _host_memory_monitor_environment() -> tuple[Path, Path, int, int]:
    journal = os.environ.get("GADP_HOST_MEMORY_SAMPLE_JOURNAL")
    start = os.environ.get("GADP_HOST_MEMORY_MONITOR_START_RECEIPT")
    monitor_pid = os.environ.get("GADP_HOST_MEMORY_MONITOR_PID")
    supervisor_pid = os.environ.get("GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID")
    _require(
        bool(journal)
        and bool(start)
        and str(monitor_pid).isdecimal()
        and str(supervisor_pid).isdecimal(),
        "host memory monitor environment is absent",
    )
    return (
        Path(str(journal)), Path(str(start)), int(str(monitor_pid)),
        int(str(supervisor_pid)),
    )


def assert_live_host_cgroup_memory_monitor() -> dict[str, Any]:
    journal_path, start_path, monitor_pid, supervisor_pid = (
        _host_memory_monitor_environment()
    )
    start, _, start_sha = load_host_cgroup_memory_monitor_start(start_path)
    _require(
        Path(start["sample_journal"]["path"]) == journal_path
        and start["monitor_pid"] == monitor_pid
        and start["supervisor_pid"] == supervisor_pid
        and _process_identity_is_live(monitor_pid, start["monitor_proc_start_ticks"])
        and _process_identity_is_live(
            supervisor_pid, start["supervisor_proc_start_ticks"]
        ),
        "host memory monitor identity is not live",
    )
    raw, rows, _, tail_observed = _journal_prefix(
        start, exact_terminal_size=False
    )
    _derive_host_cgroup_memory_gate(
        start_receipt=start,
        start_receipt_path=start_path,
        start_receipt_sha256=start_sha,
        raw_prefix=raw,
        rows=rows,
        measurement_phase="compile_smoke_before_formal40",
        formal_candidate_count_at_gate=0,
        live_tail_observed_monotonic_time_ns=tail_observed,
    )
    _assert_fresh_live_journal_tail(start)
    _require(
        _process_identity_is_live(monitor_pid, start["monitor_proc_start_ticks"])
        and _process_identity_is_live(
            supervisor_pid, start["supervisor_proc_start_ticks"]
        ),
        "host memory monitor/supervisor died during live replay",
    )
    _assert_live_cgroup_binding_matches_start(start)
    return start


def write_host_cgroup_memory_gate_receipt(
    output: Path,
    *,
    measurement_phase: str,
    formal_candidate_count_at_gate: int,
    monitor_exit_status: Optional[int] = None,
) -> tuple[dict[str, Any], str]:
    _, start_path, monitor_pid, supervisor_pid = _host_memory_monitor_environment()
    start, _, start_sha = load_host_cgroup_memory_monitor_start(start_path)
    terminal = measurement_phase.startswith("terminal_")
    if terminal:
        _require(
            monitor_exit_status == 0
            and os.getppid() == supervisor_pid
            and not _process_identity_is_live(
                monitor_pid, start["monitor_proc_start_ticks"]
            )
            and _process_identity_is_live(
                supervisor_pid, start["supervisor_proc_start_ticks"]
            ),
            "terminal host gate requires bound-supervisor wait0 and dead monitor",
        )
    else:
        _require(
            monitor_exit_status is None
            and os.getpid() != monitor_pid
            and _process_identity_is_live(
                monitor_pid, start["monitor_proc_start_ticks"]
            )
            and _process_identity_is_live(
                supervisor_pid, start["supervisor_proc_start_ticks"]
            ),
            "host memory monitor died before compile-smoke admission",
        )
    _assert_live_cgroup_binding_matches_start(start)
    raw, rows, _, tail_observed = _journal_prefix(
        start, exact_terminal_size=terminal
    )
    value = _derive_host_cgroup_memory_gate(
        start_receipt=start,
        start_receipt_path=start_path,
        start_receipt_sha256=start_sha,
        raw_prefix=raw,
        rows=rows,
        measurement_phase=measurement_phase,
        formal_candidate_count_at_gate=formal_candidate_count_at_gate,
        live_tail_observed_monotonic_time_ns=(
            None if terminal else tail_observed
        ),
    )
    receipt_sha = _write_create_only(output, value)
    load_host_cgroup_memory_gate(
        output,
        receipt_sha,
        expected_phase=measurement_phase,
        require_monitor_alive_now=not terminal,
        require_bound_cgroup_now=True,
    )
    return value, receipt_sha


def run_host_cgroup_memory_monitor(
    *,
    sample_journal: Path,
    start_receipt_output: Path,
    stop_fd: int,
    supervisor_pid: int,
    slurm_job_id: str,
    slurm_step_id: str,
) -> int:
    """Sample cgroup-v2 memory.current every 10 ms through smoke and formal40."""

    _require(
        slurm_job_id == "136141" and slurm_step_id.isdecimal(),
        "host memory monitor is not bound to the retained 136141 Slurm child",
    )
    _require(
        os.getppid() == supervisor_pid,
        "host memory monitor supervisor is not its direct launcher parent",
    )
    monitor_pid = os.getpid()
    monitor_ticks = _process_start_ticks(monitor_pid)
    supervisor_ticks = _process_start_ticks(supervisor_pid)
    for path, label in (
        (sample_journal, "sample journal"),
        (start_receipt_output, "start receipt"),
    ):
        _require(
            path.is_absolute()
            and path.parent.is_dir()
            and not path.parent.is_symlink()
            and not path.exists()
            and not path.is_symlink(),
            f"host memory monitor {label} must be fresh",
        )
    _require(type(stop_fd) is int and stop_fd >= 3, "monitor stop fd differs")
    try:
        stop_metadata = os.fstat(stop_fd)
        os.set_blocking(stop_fd, False)
    except OSError as error:
        raise Reserve4GenerationError(
            "host memory monitor anonymous stop capability is unavailable"
        ) from error
    _require(
        stat.S_ISFIFO(stop_metadata.st_mode)
        and stop_metadata.st_uid == os.geteuid()
        and stop_metadata.st_gid == os.getegid()
        and (stat.S_IMODE(stop_metadata.st_mode) & 0o007) == 0,
        "host memory monitor anonymous stop capability differs",
    )
    binding = _discover_live_cgroup_v2(
        slurm_job_id=slurm_job_id,
        slurm_step_id=slurm_step_id,
    )
    descriptor = os.open(
        sample_journal,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
        0o600,
    )
    pending = bytearray()

    def append_sample(sample: Sequence[int], *, force: bool) -> None:
        pending.extend(HOST_MEMORY_SAMPLE_STRUCT.pack(*sample))
        if force or len(pending) >= HOST_MEMORY_SAMPLE_STRUCT.size:
            written = os.write(descriptor, pending)
            _require(
                written == len(pending),
                "host memory journal short write",
            )
            pending.clear()

    try:
        os.fchmod(descriptor, 0o600)
        first = (*_sample_live_cgroup_memory(binding, sequence=0), 0)
        append_sample(first, force=True)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        unsigned_start = {
            "schema_version": HOST_CGROUP_MEMORY_MONITOR_START_SCHEMA,
            "monitor_pid": monitor_pid,
            "monitor_proc_start_ticks": monitor_ticks,
            "supervisor_pid": supervisor_pid,
            "supervisor_proc_start_ticks": supervisor_ticks,
            "slurm_job_id": slurm_job_id,
            "slurm_step_id": slurm_step_id,
            "monitor_started_before_compile_smoke_and_formal40": True,
            **binding,
            "sampling_source": "cgroup_v2_memory.current_fixed_10ms",
            "sample_journal": {
                "path": str(sample_journal),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "record_size": HOST_MEMORY_SAMPLE_STRUCT.size,
                "record_encoding": HOST_MEMORY_SAMPLE_ENCODING,
            },
            "stop_capability": {
                "kind": "anonymous-pipe-read-end",
                "read_fd_at_start": stop_fd,
                "device": stop_metadata.st_dev,
                "inode": stop_metadata.st_ino,
                "uid": stop_metadata.st_uid,
                "gid": stop_metadata.st_gid,
                "mode": stat.S_IMODE(stop_metadata.st_mode),
                "link_count": stop_metadata.st_nlink,
                "terminal_token_sha256": (
                    HOST_MEMORY_MONITOR_STOP_TOKEN_SHA256
                ),
                "named_filesystem_stop_token": False,
            },
            "sample_interval_ns": HOST_MEMORY_SAMPLE_INTERVAL_NS,
            "maximum_sample_gap_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
            "strict_host_memory_limit_gib": HOST_MEMORY_LIMIT_GIB,
            "strict_host_memory_limit_bytes": HOST_MEMORY_LIMIT_BYTES,
            "host_memory_safe_ceiling_gib": HOST_MEMORY_SAFE_CEILING_GIB,
            "host_memory_safe_ceiling_bytes": HOST_MEMORY_SAFE_CEILING_BYTES,
            "initial_sample": _sample_row(first),
        }
        start_receipt = {
            **unsigned_start,
            "receipt_digest": object_sha256(unsigned_start),
        }
        _write_create_only(start_receipt_output, start_receipt)
        sequence = 1
        next_deadline = first[2] + HOST_MEMORY_SAMPLE_INTERVAL_NS
        stop_bytes = bytearray()
        stop_closed = False
        stop_poller = select.poll()
        stop_poller.register(
            stop_fd,
            select.POLLIN | select.POLLHUP | select.POLLERR | select.POLLNVAL,
        )
        while not stop_closed:
            _require(
                _process_identity_is_live(supervisor_pid, supervisor_ticks),
                "host memory monitor supervisor disappeared",
            )
            remaining = next_deadline - time.monotonic_ns()
            timeout_ms = max(0, (remaining + 999_999) // 1_000_000)
            events = stop_poller.poll(timeout_ms)
            for _, event_mask in events:
                _require(
                    not event_mask & (select.POLLERR | select.POLLNVAL),
                    "host memory monitor anonymous stop pipe failed",
                )
                while True:
                    try:
                        chunk = os.read(stop_fd, 64)
                    except BlockingIOError:
                        break
                    if not chunk:
                        stop_closed = True
                        break
                    stop_bytes.extend(chunk)
                    _require(
                        len(stop_bytes) <= len(HOST_MEMORY_MONITOR_STOP_TOKEN),
                        "host memory monitor stop capability token is oversized",
                    )
                if event_mask & select.POLLHUP:
                    stop_closed = True
            if stop_closed:
                break
            if time.monotonic_ns() < next_deadline:
                continue
            sample = (*_sample_live_cgroup_memory(binding, sequence=sequence), 0)
            _require(
                sample[2] - first[2]
                <= sequence * HOST_MEMORY_SAMPLE_INTERVAL_NS
                + HOST_MEMORY_MAX_SAMPLE_GAP_NS,
                "host memory monitor accumulated a scheduling lapse",
            )
            append_sample(sample, force=False)
            sequence += 1
            next_deadline += HOST_MEMORY_SAMPLE_INTERVAL_NS
        _require(
            bytes(stop_bytes) == HOST_MEMORY_MONITOR_STOP_TOKEN,
            "host memory monitor anonymous stop token differs",
        )
        final_sample = (*_sample_live_cgroup_memory(binding, sequence=sequence), 1)
        append_sample(final_sample, force=True)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.close(descriptor)
        descriptor = -1
        os.close(stop_fd)
        stop_fd = -1
        return 0
    except Exception:
        if descriptor >= 0:
            try:
                if pending:
                    os.write(descriptor, pending)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if stop_fd >= 0:
            os.close(stop_fd)
        if _process_identity_is_live(supervisor_pid, supervisor_ticks):
            try:
                os.kill(supervisor_pid, signal.SIGTERM)
            except OSError:
                pass
        raise


def _sealed_specs(
    seed1_path: str | Path, seed2_path: str | Path
) -> dict[str, tuple[dict[str, Any], Path, str]]:
    results: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for slot, path in (("seed1", seed1_path), ("seed2", seed2_path)):
        expected = SPEC_AUTHORITIES[slot]
        try:
            spec, observed = bank_contract.load_sealed_spec(path, expected)
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Reserve4GenerationError(str(error)) from error
        resolved = _plain_file(path, f"reserve4 {slot} spec")
        _require(observed == expected, f"reserve4 {slot} authority differs")
        results[slot] = (spec, resolved, observed)
    try:
        derived = seed2_builder.derive_seed2_spec(results["seed1"][0], "reserve4-v1")
    except seed2_builder.PairV5T2VSeed2Error as error:
        raise Reserve4GenerationError(str(error)) from error
    _require(
        derived == results["seed2"][0],
        "seed2 is not the registered seed-only derivation of seed1",
    )
    return results


def _cell_proof(tasks: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for task in tasks:
        key = (
            str(task["seed_slot"]),
            str(task["group_id"]),
            str(task["calibration_group_id"]),
            int(task["seed"]),
        )
        cells.setdefault(key, []).append(task)
    proofs: list[dict[str, Any]] = []
    for key, rows in cells.items():
        branches = [str(row["semantic_branch"]) for row in rows]
        _require(
            branches == list(bank_contract.MACE_BRANCH_ORDER),
            f"{label} cell {key!r} is not one complete ordered ten-branch cell",
        )
        _require(
            len({row["analysis_split"] for row in rows}) == 1,
            f"{label} cell split differs",
        )
        proofs.append(
            {
                "seed_slot": key[0],
                "group_id": key[1],
                "calibration_group_id": key[2],
                "seed": key[3],
                "analysis_split": rows[0]["analysis_split"],
                "candidate_ids": [row["candidate_id"] for row in rows],
                "branch_order": branches,
                "complete_ten_branch_cell": True,
            }
        )
    return proofs


def _tasks_from_candidate_plan(
    *, slot: str, spec_path: Path, spec_sha256: str, plan: Mapping[str, Any], split: str
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record in plan["candidate_records"]:
        envelope_path = _plain_file(record["path"], "candidate envelope")
        _require(
            file_sha256(envelope_path) == record["sha256"],
            "candidate plan/envelope SHA-256 differs",
        )
        try:
            envelope = bank_contract.load_candidate_envelope(
                envelope_path, spec_sha256
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Reserve4GenerationError(str(error)) from error
        candidate = envelope["candidate"]
        _require(
            candidate["candidate_id"] == record["candidate_id"],
            "candidate plan identity differs",
        )
        if candidate["analysis_split"] != split:
            continue
        _require(
            candidate["candidate_id"].startswith(SEED_PREFIXES[slot]),
            f"{slot} candidate prefix differs",
        )
        tasks.append(
            {
                "seed_slot": slot,
                "root_spec_path": str(spec_path),
                "root_spec_sha256": spec_sha256,
                "candidate_spec_path": str(envelope_path),
                "candidate_spec_sha256": record["sha256"],
                "group_id": envelope["group_id"],
                "visible_gpus": envelope["visible_gpus"],
                "ordinal": envelope["ordinal"],
                "candidate_id": candidate["candidate_id"],
                "analysis_split": candidate["analysis_split"],
                "calibration_group_id": candidate["calibration_group_id"],
                "semantic_branch": candidate["semantic_branch"],
                "seed": candidate["seed"],
            }
        )
    return tasks


def build_plan(
    *, seed1_spec: str | Path, seed2_spec: str | Path, split: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    _require(split in bank_contract.ANALYSIS_SPLITS, "split differs")
    output = Path(output_dir)
    _require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "plan output must be a fresh absolute directory with a plain parent",
    )
    authorities = _sealed_specs(seed1_spec, seed2_spec)
    output.mkdir(mode=0o700)
    tasks: list[dict[str, Any]] = []
    source_plans: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        _, spec_path, spec_sha = authorities[slot]
        candidate_plan_dir = output / f"{slot}-candidate-plan"
        try:
            candidate_plan = bank_contract.materialize_plan(
                spec_path=spec_path,
                expected_sha256=spec_sha,
                output_dir=candidate_plan_dir,
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise Reserve4GenerationError(str(error)) from error
        candidate_manifest_path = candidate_plan_dir / "manifest.json"
        source_plans.append(
            {
                "seed_slot": slot,
                "root_spec_path": str(spec_path),
                "root_spec_sha256": spec_sha,
                "candidate_plan_manifest_path": str(candidate_manifest_path),
                "candidate_plan_manifest_sha256": file_sha256(candidate_manifest_path),
                "candidate_plan_manifest_digest": candidate_plan["manifest_digest"],
            }
        )
        tasks.extend(
            _tasks_from_candidate_plan(
                slot=slot,
                spec_path=spec_path,
                spec_sha256=spec_sha,
                plan=candidate_plan,
                split=split,
            )
        )
    _require(len(tasks) == 40, f"{split} reserve4 scope must contain exactly 40 clips")
    cell_proofs = _cell_proof(tasks, "generation plan")
    _require(len(cell_proofs) == 4, f"{split} reserve4 scope must contain four seed cells")
    shards = []
    for slot in ("seed1", "seed2"):
        for group_id, visible_gpus in bank_contract.GROUP_LAYOUT:
            candidate_ids = [
                row["candidate_id"]
                for row in tasks
                if row["seed_slot"] == slot and row["group_id"] == group_id
            ]
            if candidate_ids:
                shards.append(
                    {
                        "shard_id": f"{slot}-{group_id}-{split}",
                        "seed_slot": slot,
                        "group_id": group_id,
                        "visible_gpus": visible_gpus,
                        "candidate_ids": candidate_ids,
                        "candidate_count": len(candidate_ids),
                    }
                )
    _require(
        sum(row["candidate_count"] for row in shards) == 40,
        "generation shard coverage differs",
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": f"reserve4-{split}-two-seed-complete10-v1",
        "analysis_split": split,
        "source_specs": source_plans,
        "generation_invocation_count": 40,
        "seed_cell_count": 4,
        "branch_order": list(bank_contract.MACE_BRANCH_ORDER),
        "tasks": tasks,
        "cell_proofs": cell_proofs,
        "shards": shards,
        "execution_contract": {
            "topology": "one_model_replica_world4_dp1_sp4",
            "candidate_order": "sealed_spec_group_then_ordinal",
            "candidates_serial_inside_shard": True,
            "fit_first_does_not_reclassify_confirmation": True,
            "generated_media_role": "representation_authoring_evidence_only",
            "generated_media_is_editor_input_or_target": False,
            "generated_latent_or_gaussian_is_editor_input_or_target": False,
            "visual_review_required_before_phi_v1_extraction": True,
            "optimizer_authorized": False,
        },
    }
    plan = {**plan, "plan_digest": object_sha256(plan)}
    plan_path = output / "reserve4-fixed-generation-plan-v1.json"
    plan_sha = _write_create_only(plan_path, plan)
    gap = {
        "schema_version": GAP_SCHEMA,
        "plan_path": str(plan_path),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": split,
        "expected_candidate_count": 40,
        "observed_candidate_count": 0,
        "missing_candidate_ids": [row["candidate_id"] for row in tasks],
        "complete_ten_branch_seed_cells": 0,
        "independent_full81_review_count": 0,
        "phi_v1_extraction_authorized": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    gap = {**gap, "receipt_digest": object_sha256(gap)}
    _write_create_only(output / "reserve4-generation-gap-before-run-v1.json", gap)
    return {**plan, "_path": str(plan_path), "_file_sha256": plan_sha}


def load_plan(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], Path, str]:
    plan, plan_path, observed = _load_json(path, "reserve4 generation plan", expected_sha256)
    _require(plan.get("schema_version") == PLAN_SCHEMA, "generation plan schema differs")
    declared = plan.get("plan_digest")
    _require(SHA256_RE.fullmatch(str(declared)) is not None, "plan digest differs")
    unsigned = dict(plan)
    del unsigned["plan_digest"]
    _require(object_sha256(unsigned) == declared, "plan digest differs")
    split = plan.get("analysis_split")
    _require(split in bank_contract.ANALYSIS_SPLITS, "plan split differs")
    source_specs = plan.get("source_specs")
    _require(type(source_specs) is list and len(source_specs) == 2, "plan source specs differ")
    refs = {row.get("seed_slot"): row for row in source_specs if type(row) is dict}
    _require(set(refs) == {"seed1", "seed2"}, "plan source slots differ")
    authorities = _sealed_specs(
        refs["seed1"]["root_spec_path"], refs["seed2"]["root_spec_path"]
    )
    expected_tasks: list[dict[str, Any]] = []
    for slot in ("seed1", "seed2"):
        _, spec_path, spec_sha = authorities[slot]
        ref = refs[slot]
        _require(
            ref.get("root_spec_path") == str(spec_path)
            and ref.get("root_spec_sha256") == spec_sha,
            "plan root spec binding differs",
        )
        candidate_manifest, manifest_path, manifest_sha = _load_json(
            ref["candidate_plan_manifest_path"], "candidate plan manifest",
            ref["candidate_plan_manifest_sha256"],
        )
        _require(
            manifest_sha == ref["candidate_plan_manifest_sha256"]
            and candidate_manifest.get("manifest_digest")
            == ref["candidate_plan_manifest_digest"],
            "candidate plan manifest binding differs",
        )
        expected_tasks.extend(
            _tasks_from_candidate_plan(
                slot=slot,
                spec_path=spec_path,
                spec_sha256=spec_sha,
                plan=candidate_manifest,
                split=split,
            )
        )
    _require(plan.get("tasks") == expected_tasks, "plan task bytes/order differ")
    _require(len(expected_tasks) == 40, "plan task count differs")
    expected_cells = _cell_proof(expected_tasks, "validated plan")
    _require(plan.get("cell_proofs") == expected_cells, "plan cell proof differs")
    _require(plan.get("seed_cell_count") == 4, "plan seed-cell count differs")
    _require(plan.get("generation_invocation_count") == 40, "plan invocation count differs")
    contract = plan.get("execution_contract", {})
    _require(
        contract.get("topology") == "one_model_replica_world4_dp1_sp4"
        and contract.get("candidates_serial_inside_shard") is True
        and contract.get("generated_media_is_editor_input_or_target") is False
        and contract.get("generated_latent_or_gaussian_is_editor_input_or_target") is False
        and contract.get("visual_review_required_before_phi_v1_extraction") is True
        and contract.get("optimizer_authorized") is False,
        "plan execution/authority contract differs",
    )
    return plan, plan_path, observed


def _expected_interpretation() -> dict[str, Any]:
    return {
        "calibration_evidence_only": True,
        "event_qualified_from_generation_receipt": False,
        "action_success_not_implied": True,
        "training_performed": False,
        "parameter_update_performed": False,
        "optimizer_authorized": False,
        "t2v_media_as_rv2v_policy_candidate_forbidden": True,
        "donor_or_pseudo_target_use_forbidden": True,
    }


def _validate_candidate_receipt(
    task: Mapping[str, Any], receipt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    import infer_pair_v5_t2v_calibration_bank as renderer

    try:
        envelope = bank_contract.load_candidate_envelope(
            task["candidate_spec_path"], task["root_spec_sha256"]
        )
        receipt = renderer._load_pair_receipt(receipt_path)  # type: ignore[attr-defined]
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise Reserve4GenerationError(str(error)) from error
    candidate = envelope["candidate"]
    expected_visible = ",".join(str(item) for item in task["visible_gpus"])
    _require(
        receipt["root_spec_raw_sha256"] == task["root_spec_sha256"]
        and receipt["candidate_envelope_sha256"] == task["candidate_spec_sha256"]
        and receipt["candidate"] == candidate
        and receipt["group_id"] == task["group_id"]
        and receipt["visible_gpus"] == task["visible_gpus"]
        and receipt["ordinal"] == task["ordinal"]
        and receipt["runtime_topology"]
        == {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": expected_visible,
        },
        f"candidate receipt/spec binding differs: {task['candidate_id']}",
    )
    _require(
        receipt["sampling_contract"] == bank_contract.SAMPLING_CONTRACT
        and receipt["semantic_input_closure"] == bank_contract.SEMANTIC_INPUT_CLOSURE
        and receipt["artifact_use_contract"] == bank_contract.ARTIFACT_USE_CONTRACT
        and receipt["split_contract"] == bank_contract.SPLIT_CONTRACT
        and receipt["interpretation"] == _expected_interpretation(),
        f"candidate receipt exceeds generation-only authority: {task['candidate_id']}",
    )
    native_path = _plain_file(receipt["native_receipt_path"], "native receipt")
    try:
        native_receipt = renderer._load_json(native_path, "receipt-bound native receipt")  # type: ignore[attr-defined]
        _require(
            file_sha256(native_path) == receipt["native_receipt_sha256"],
            "native receipt SHA-256 differs",
        )
        native_artifacts = renderer._verify_native_receipt(  # type: ignore[attr-defined]
            native_receipt, candidate
        )
        expected_artifacts = {
            "mp4": native_artifacts["mp4"],
            "predecode_clean_latent": native_artifacts["predecode_clean_latent"],
            "official_initial_gaussian": native_artifacts["official_initial_gaussian"],
        }
        _require(
            native_artifacts["native_receipt_digest"] == receipt["native_receipt_digest"]
            and receipt["artifacts"] == expected_artifacts,
            "candidate/native artifact binding differs",
        )
        for name, artifact in receipt["artifacts"].items():
            renderer._verify_file_artifact(  # type: ignore[attr-defined]
                artifact, f"{task['candidate_id']} {name}"
            )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise Reserve4GenerationError(str(error)) from error
    return receipt, candidate


def _gaussian_cell_proofs(
    validated: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for task, receipt in validated:
        key = (
            str(task["seed_slot"]),
            str(task["group_id"]),
            str(task["calibration_group_id"]),
            int(task["seed"]),
        )
        cells.setdefault(key, []).append((task, receipt))
    proofs: list[dict[str, Any]] = []
    identity_fields = (
        "raw_value_sha256", "content_sha256", "shape", "dtype", "stored_dtype",
        "generator_initial_seed",
    )
    for key, rows in cells.items():
        branches = [task["semantic_branch"] for task, _ in rows]
        _require(
            branches == list(bank_contract.MACE_BRANCH_ORDER),
            f"rendered cell {key!r} is incomplete or reordered",
        )
        gaussians = [receipt["artifacts"]["official_initial_gaussian"] for _, receipt in rows]
        identities = {
            object_sha256({field: artifact.get(field) for field in identity_fields})
            for artifact in gaussians
        }
        _require(len(identities) == 1, f"rendered cell {key!r} did not reuse one Gaussian")
        first = gaussians[0]
        proofs.append(
            {
                "seed_slot": key[0],
                "group_id": key[1],
                "calibration_group_id": key[2],
                "seed": key[3],
                "branch_order": branches,
                "official_gaussian_raw_value_sha256": first["raw_value_sha256"],
                "official_gaussian_content_sha256": first["content_sha256"],
                "all_ten_official_gaussian_tensor_values_byte_equal": True,
            }
        )
    return proofs


_ALLOWED_NODE_LOCAL_FILESYSTEMS = {"ext2/ext3", "xfs", "tmpfs"}
_UNSAFE_PYTHON_ENVIRONMENT = {
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONSAFEPATH",
    "PYTHONUSERBASE",
}
_UNSAFE_RANK_SHELL_ENVIRONMENT = {
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "BASHOPTS",
    "CDPATH",
    "GLOBIGNORE",
    "IFS",
}
_SMOKE_ARTIFACT_NAMES = (
    "mp4",
    "official_initial_gaussian",
    "predecode_clean_latent",
)


def _filesystem_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", "--", str(path)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Reserve4GenerationError(
            "cannot identify the COMGR scratch filesystem"
        ) from error
    value = result.stdout.strip()
    _require(value in _ALLOWED_NODE_LOCAL_FILESYSTEMS, "COMGR scratch is not node-local")
    return value


def _runtime_binding(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path, Path, str, str, Path]:
    python = _plain_file(args.python, "Python executable")
    _require(
        not any(name in os.environ for name in _UNSAFE_PYTHON_ENVIRONMENT),
        "caller-controlled Python environment presence is forbidden",
    )
    _require(
        python == FROZEN_PYTHON_PATH and os.access(python, os.X_OK),
        "Python executable identity/path differs",
    )
    _, _, python_observed = _stable_plain_file_bytes(
        python, "frozen Python executable", FROZEN_PYTHON_SHA256
    )
    python_metadata = python.lstat()
    _require(
        stat.S_IMODE(python_metadata.st_mode) == 0o755
        and python_metadata.st_uid == 2012
        and python_metadata.st_gid == 2000
        and python_metadata.st_nlink == 1
        and python_metadata.st_size == FROZEN_PYTHON_SIZE,
        "frozen Python executable metadata differs",
    )
    root_bootstrap, _, root_bootstrap_observed = _stable_plain_file_bytes(
        ROOT_BOOTSTRAP_PYTHON_PATH,
        "root-owned Python bootstrap",
        ROOT_BOOTSTRAP_PYTHON_SHA256,
    )
    root_bootstrap_metadata = root_bootstrap.lstat()
    _require(
        stat.S_IMODE(root_bootstrap_metadata.st_mode) == 0o755
        and root_bootstrap_metadata.st_uid == 0
        and root_bootstrap_metadata.st_gid == 0
        and root_bootstrap_metadata.st_nlink == 1
        and root_bootstrap_metadata.st_size == ROOT_BOOTSTRAP_PYTHON_SIZE,
        "root-owned Python bootstrap metadata differs",
    )
    bernini_root = _plain_dir(args.bernini_root, "Bernini root")
    veomni_root = _plain_dir(args.veomni_root, "VeOmni root")
    checkpoint = _plain_dir(args.checkpoint, "checkpoint")
    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, "checkpoint content manifest"
    )
    _require(
        REVISION_RE.fullmatch(args.method_source_revision) is not None
        and SHA256_RE.fullmatch(args.method_source_archive_sha256) is not None,
        "method source revision/archive SHA differs",
    )
    _require(1024 <= args.master_port <= 65535, "master port differs")
    worker = _plain_file(
        METHOD_ROOT / "infer_pair_v5_t2v_calibration_bank.py",
        "generation worker",
    )
    manifest_value = os.environ.get("F13_METHOD_MANIFEST")
    manifest_sha = os.environ.get("F13_METHOD_MANIFEST_SHA256")
    runner_value = os.environ.get("F13_VERIFIED_RUNNER_PATH")
    runner_sha = os.environ.get("F13_VERIFIED_RUNNER_SHA256")
    rank_exec_sha = os.environ.get("F13_RANK_WRAPPER_SHA256")
    _require(
        bool(manifest_value)
        and SHA256_RE.fullmatch(str(manifest_sha)) is not None
        and bool(runner_value)
        and SHA256_RE.fullmatch(str(runner_sha)) is not None
        and SHA256_RE.fullmatch(str(rank_exec_sha)) is not None,
        "verified release execution environment is absent",
    )
    manifest, _, manifest_observed = _stable_plain_file_bytes(
        str(manifest_value), "verified release manifest", str(manifest_sha)
    )
    runner, _, runner_observed = _stable_plain_file_bytes(
        str(runner_value), "verified release runner", str(runner_sha)
    )
    _require(
        runner
        == METHOD_ROOT
        / "tools/build_full30_action_arms_incomplete_repair_exact2_release_v1.py",
        "verified release runner path differs",
    )
    rank_exec, rank_exec_raw, rank_exec_observed = _stable_plain_file_bytes(
        RANK_EXEC, "per-rank cache wrapper", str(rank_exec_sha)
    )
    _require(os.access(rank_exec, os.X_OK), "per-rank cache wrapper is not executable")
    try:
        rank_exec_source = rank_exec_raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise Reserve4GenerationError(
            "per-rank cache wrapper is not ASCII shell source"
        ) from error
    site_packages = _plain_dir(FROZEN_SITE_PACKAGES, "frozen site-packages")
    torchrun_path, torchrun_raw, torchrun_sha = _stable_plain_file_bytes(
        site_packages / "torch/distributed/run.py",
        "isolated torchrun launcher",
    )
    try:
        torchrun_source = torchrun_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Reserve4GenerationError(
            "isolated torchrun launcher is not UTF-8 Python source"
        ) from error
    preprocessing: dict[str, str] = {}
    for relative, expected in PREPROCESSING_TOOL_SHA256.items():
        member = _plain_file(METHOD_ROOT / relative, f"release {relative}")
        observed = file_sha256(member)
        _require(observed == expected, f"release preprocessing identity differs: {relative}")
        preprocessing[relative] = observed
    scratch_value = os.environ.get("GADP_NODE_LOCAL_SCRATCH")
    expected_fstype = os.environ.get("GADP_NODE_LOCAL_SCRATCH_FSTYPE")
    _require(bool(scratch_value), "authenticated node-local scratch is absent")
    scratch = _plain_dir(str(scratch_value), "authenticated node-local scratch")
    observed_fstype = _filesystem_type(scratch)
    _require(
        expected_fstype == observed_fstype,
        "authenticated node-local scratch filesystem changed",
    )
    _require(
        os.environ.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED") == "1",
        "serialized host checkpoint-load requirement is absent",
    )
    _require(
        os.environ.get("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED") == "1",
        "T2V rank-GPU text-encoder residency requirement is absent",
    )
    monitor_start = assert_live_host_cgroup_memory_monitor()
    _, monitor_start_path, monitor_pid, monitor_supervisor_pid = (
        _host_memory_monitor_environment()
    )
    load_lock_value = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK")
    _require(bool(load_lock_value), "serialized host checkpoint-load lock is absent")
    load_lock = _plain_file(str(load_lock_value), "serialized host checkpoint-load lock")
    load_lock_metadata = load_lock.lstat()
    scratch_metadata = scratch.lstat()
    _require(
        load_lock.parent == scratch
        and load_lock_metadata.st_dev == scratch_metadata.st_dev
        and load_lock_metadata.st_uid == os.geteuid()
        and load_lock_metadata.st_gid == os.getegid()
        and load_lock_metadata.st_nlink == 1
        and stat.S_IMODE(load_lock_metadata.st_mode) == 0o400
        and load_lock_metadata.st_size == 0
        and file_sha256(load_lock) == EMPTY_FILE_SHA256,
        "serialized host checkpoint-load lock identity differs",
    )
    binding = {
        "method_root": str(METHOD_ROOT),
        "python": {"path": str(python), "sha256": python_observed},
        "bernini_root": str(bernini_root),
        "veomni_root": str(veomni_root),
        "checkpoint": str(checkpoint),
        "checkpoint_content_manifest": {
            "path": str(checkpoint_manifest),
            "sha256": file_sha256(checkpoint_manifest),
        },
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "generation_worker": {
            "path": str(worker),
            "sha256": file_sha256(worker),
        },
        "rank_cache_wrapper": {
            "path": str(rank_exec),
            "sha256": rank_exec_observed,
        },
        "verified_release_execution": {
            "isolated_python_flags": ["-I", "-S", "-B"],
            "held_fd_frozen_python_execution": {
                "root_owned_bootstrap": {
                    "path": str(root_bootstrap),
                    "sha256": root_bootstrap_observed,
                    "size_bytes": ROOT_BOOTSTRAP_PYTHON_SIZE,
                    "uid": 0,
                    "gid": 0,
                    "mode": "0755",
                    "link_count": 1,
                    "isolated_flags": ["-I", "-S", "-s", "-B"],
                },
                "frozen_target": {
                    "path": str(python),
                    "sha256": python_observed,
                    "size_bytes": FROZEN_PYTHON_SIZE,
                    "uid": 2012,
                    "gid": 2000,
                    "mode": "0755",
                    "link_count": 1,
                },
                "verified_runner_subcommand": "held-fd-exec-frozen-python",
                "execution_boundaries": [
                    "torchrun-coordinator",
                    "per-rank-worker",
                ],
                "target_opened_no_follow_and_double_sha256_verified": True,
                "target_execve_uses_retained_file_descriptor": True,
                "target_named_path_is_never_passed_to_execve": True,
                "candidate_process_anonymous_start_gate_before_target_exec": True,
            },
            "manifest": {
                "path": str(manifest),
                "sha256": manifest_observed,
            },
            "runner": {"path": str(runner), "sha256": runner_observed},
            "torchrun_runtime_snapshot": {
                "path": str(torchrun_path),
                "sha256": torchrun_sha,
                "site_packages": str(site_packages),
            },
            "torchrun_source_executed_from_stable_captured_in_memory_bytes": True,
            "torchrun_launcher_preauthorized_by_release": False,
            "torchrun_runtime_snapshot_only": True,
            "site_packages_dependency_bytes_preauthorized_by_release": False,
            "torchrun_automatic_site_initialization_disabled": True,
            "rank_wrapper_executed_from_verified_in_memory_bytes": True,
            "worker_and_release_imports_served_from_verified_in_memory_bytes": True,
            "caller_python_environment_presence_rejected": True,
            "rank_shell_startup_environment_scrubbed": True,
            "safe_runtime_path": SAFE_RUNTIME_PATH,
            "candidate_gpu_process_group_monitored_every_10ms": True,
            "per_candidate_post_exit_stable_step_cgroup_census_before_next_gpu": True,
        },
        "preprocessing_tools": preprocessing,
        "node_local_scratch": {
            "path": str(scratch),
            "filesystem_type": observed_fstype,
        },
        "serialized_host_checkpoint_load": {
            "required": True,
            "environment_variable": "NATIVE_V_AXIS_LOAD_LOCK",
            "path": str(load_lock),
            "sha256": EMPTY_FILE_SHA256,
            "mode": "0400",
            "device": load_lock_metadata.st_dev,
            "inode": load_lock_metadata.st_ino,
            "uid": load_lock_metadata.st_uid,
            "gid": load_lock_metadata.st_gid,
            "link_count": load_lock_metadata.st_nlink,
            "size_bytes": load_lock_metadata.st_size,
            "parent_is_authenticated_node_local_scratch": True,
            "lock_held_through_model_to_rank_gpu_and_malloc_trim": True,
        },
        "t2v_text_encoder_rank_gpu_residency": {
            "required": True,
            "environment_variable": "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
            "official_model_sample_preserved": True,
            "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
            "all_other_to_requests_delegated": True,
            "text_encoder_retired_only_with_renderer": True,
            "gpu_memory_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
            "gpu_memory_limit_bytes": T2V_GPU_MEMORY_LIMIT_BYTES,
            "per_rank_max_allocated_and_reserved_receipt_required": True,
            "all_rank_peak_reserved_strictly_below_limit_required": True,
        },
        "host_cgroup_sampled_memory_monitor": {
            "required": True,
            "start_receipt": {
                "path": str(monitor_start_path),
                "file_sha256": file_sha256(monitor_start_path),
                "receipt_digest": monitor_start["receipt_digest"],
            },
            "sample_journal": monitor_start["sample_journal"],
            "stop_capability": monitor_start["stop_capability"],
            "monitor_pid": monitor_pid,
            "monitor_proc_start_ticks": monitor_start["monitor_proc_start_ticks"],
            "supervisor_pid": monitor_supervisor_pid,
            "supervisor_proc_start_ticks": monitor_start[
                "supervisor_proc_start_ticks"
            ],
            "slurm_job_id": "136141",
            "slurm_step_id": monitor_start["slurm_step_id"],
            "memory_max_exact_gib": HOST_MEMORY_LIMIT_GIB,
            "sampled_current_safe_ceiling_gib": HOST_MEMORY_SAFE_CEILING_GIB,
            "sample_interval_ns": HOST_MEMORY_SAMPLE_INTERVAL_NS,
            "maximum_sample_gap_ns": HOST_MEMORY_MAX_SAMPLE_GAP_NS,
            "sampling_source": "cgroup_v2_memory.current_fixed_10ms",
            "zero_oom_and_oom_kill_required": True,
            "coverage": "before_compile_smoke_through_terminal_after_formal40",
        },
    }
    return (
        binding,
        python,
        worker,
        rank_exec,
        rank_exec_source,
        torchrun_source,
        scratch,
    )


def _candidate_command(
    args: argparse.Namespace,
    *,
    task: Mapping[str, Any],
    candidate_output: Path,
    python: Path,
    worker: Path,
    rank_exec: Path,
    rank_exec_source: str,
    torchrun_source: str,
    runtime: Mapping[str, Any],
) -> list[str]:
    torchrun = runtime["verified_release_execution"]["torchrun_runtime_snapshot"]
    runner = runtime["verified_release_execution"]["runner"]
    held_exec = runtime["verified_release_execution"][
        "held_fd_frozen_python_execution"
    ]
    root_bootstrap = held_exec["root_owned_bootstrap"]
    frozen_target = held_exec["frozen_target"]
    _require(
        str(python) == frozen_target["path"]
        and frozen_target["sha256"] == FROZEN_PYTHON_SHA256
        and root_bootstrap["path"] == str(ROOT_BOOTSTRAP_PYTHON_PATH)
        and root_bootstrap["sha256"] == ROOT_BOOTSTRAP_PYTHON_SHA256,
        "held-fd Python command authority differs",
    )
    return [
        str(ROOT_BOOTSTRAP_PYTHON_PATH),
        "-I",
        "-S",
        "-s",
        "-B",
        "-c",
        _VERIFIED_RUNNER_BOOTSTRAP,
        str(runner["path"]),
        str(runner["sha256"]),
        "held-fd-exec-frozen-python",
        "--start-gate-stdin",
        "--",
        "-I",
        "-S",
        "-B",
        "-c",
        _ISOLATED_TORCHRUN_BOOTSTRAP,
        torchrun_source,
        str(torchrun["path"]),
        str(torchrun["sha256"]),
        str(torchrun["site_packages"]),
        "--nproc_per_node=4",
        "--master_addr=127.0.0.1",
        f"--master_port={args.master_port}",
        "--no_python",
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-p",
        "-c",
        rank_exec_source,
        str(rank_exec),
        str(worker),
        "--candidate-spec",
        str(task["candidate_spec_path"]),
        "--expected-root-spec-sha256",
        str(task["root_spec_sha256"]),
        "--output-dir",
        str(candidate_output),
        "--bernini-root",
        args.bernini_root,
        "--veomni-root",
        args.veomni_root,
        "--checkpoint",
        args.checkpoint,
        "--checkpoint-content-manifest",
        args.checkpoint_content_manifest,
        "--method-source-revision",
        args.method_source_revision,
        "--method-source-archive-sha256",
        args.method_source_archive_sha256,
    ]


def _candidate_environment(
    *,
    expected_visible: str,
    python: Path,
    scratch: Path,
    cache_token: str,
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    _require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", cache_token) is not None,
        "rank cache token differs",
    )
    inherited = os.environ
    load_lock = inherited.get("NATIVE_V_AXIS_LOAD_LOCK")
    monitor_journal = inherited.get("GADP_HOST_MEMORY_SAMPLE_JOURNAL")
    monitor_start = inherited.get("GADP_HOST_MEMORY_MONITOR_START_RECEIPT")
    monitor_pid = inherited.get("GADP_HOST_MEMORY_MONITOR_PID")
    monitor_supervisor_pid = inherited.get(
        "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID"
    )
    _require(
        inherited.get("NATIVE_SERIALIZED_HOST_LOAD_REQUIRED") == "1"
        and bool(load_lock)
        and inherited.get("NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED") == "1",
        "serialized host/T5 residency environment is absent",
    )
    _require(
        bool(monitor_journal)
        and bool(monitor_start)
        and str(monitor_pid).isdecimal()
        and str(monitor_supervisor_pid).isdecimal(),
        "host memory monitor environment is absent",
    )
    _require(
        not any(name in inherited for name in _UNSAFE_RANK_SHELL_ENVIRONMENT)
        and not any(name.startswith("BASH_FUNC_") for name in inherited),
        "caller shell-startup environment presence is forbidden",
    )
    monitor = runtime["host_cgroup_sampled_memory_monitor"]
    scratch_runtime = runtime["node_local_scratch"]
    environment = {
            "PATH": SAFE_RUNTIME_PATH,
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MODELING_BACKEND": "hf",
            "ROCR_VISIBLE_DEVICES": expected_visible,
            "TMPDIR": str(scratch),
            "SLURM_JOB_ID": str(monitor["slurm_job_id"]),
            "SLURM_STEP_ID": str(monitor["slurm_step_id"]),
            "GADP_NODE_LOCAL_SCRATCH": str(scratch_runtime["path"]),
            "GADP_NODE_LOCAL_SCRATCH_FSTYPE": str(
                scratch_runtime["filesystem_type"]
            ),
            "GADP_RANK_CACHE_TOKEN": cache_token,
            "GADP_RANK_PYTHON_BIN": str(python),
            "GADP_METHOD_ROOT": str(METHOD_ROOT),
            "F13_METHOD_MANIFEST": str(
                runtime["verified_release_execution"]["manifest"]["path"]
            ),
            "F13_METHOD_MANIFEST_SHA256": str(
                runtime["verified_release_execution"]["manifest"]["sha256"]
            ),
            "F13_VERIFIED_RUNNER_PATH": str(
                runtime["verified_release_execution"]["runner"]["path"]
            ),
            "F13_VERIFIED_RUNNER_SHA256": str(
                runtime["verified_release_execution"]["runner"]["sha256"]
            ),
            "F13_RANK_WRAPPER_SHA256": str(
                runtime["rank_cache_wrapper"]["sha256"]
            ),
            "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
            "NATIVE_V_AXIS_LOAD_LOCK": str(load_lock),
            "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1",
            "GADP_HOST_MEMORY_SAMPLE_JOURNAL": str(monitor_journal),
            "GADP_HOST_MEMORY_MONITOR_START_RECEIPT": str(monitor_start),
            "GADP_HOST_MEMORY_MONITOR_PID": str(monitor_pid),
            "GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID": str(
                monitor_supervisor_pid
            ),
    }
    return environment


def _owned_candidate_process_observation(
    process: subprocess.Popen[Any], start_ticks: int
) -> str:
    """Return live/dead/unknown without authorizing a signal on ambiguity."""

    if process.poll() is not None:
        return "dead"
    try:
        raw = Path(f"/proc/{process.pid}/stat").read_text(encoding="ascii")
        close = raw.rfind(")")
        _require(close > 0, "candidate process identity format differs")
        fields = raw[close + 2 :].split()
        _require(
            len(fields) >= 20
            and fields[0]
            and fields[19].isdecimal(),
            "candidate process identity fields differ",
        )
        observed_ticks = int(fields[19])
        state = fields[0]
    except (OSError, UnicodeError, ValueError, IndexError, Reserve4GenerationError):
        return "unknown"
    if observed_ticks != start_ticks:
        return "unknown"
    if state in {"Z", "X", "x"}:
        return "dead"
    return "live"


def _candidate_cgroup_process_identity(pid: int) -> dict[str, Any]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        cgroup_raw = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise Reserve4GenerationError(
            "post-candidate cgroup process identity is unavailable"
        ) from error
    close = raw.rfind(")")
    fields = raw[close + 2 :].split() if close > 0 else []
    v2_rows = [line for line in cgroup_raw.splitlines() if line.startswith("0::")]
    _require(
        len(fields) >= 20
        and fields[1].isdecimal()
        and fields[2].isdecimal()
        and fields[3].isdecimal()
        and fields[19].isdecimal()
        and len(v2_rows) == 1,
        "post-candidate cgroup process identity format differs",
    )
    return {
        "pid": pid,
        "state": fields[0],
        "parent_pid": int(fields[1]),
        "process_group_id": int(fields[2]),
        "session_id": int(fields[3]),
        "start_ticks": int(fields[19]),
        "cgroup_v2_path": v2_rows[0][3:],
    }


def _stable_post_candidate_cgroup_census(
    monitor_start: Mapping[str, Any], *, candidate_process_group_id: int
) -> dict[str, Any]:
    """Prove no candidate/rank descendant remains before another GPU launch."""

    cgroup_path = str(monitor_start["leaf_cgroup"]["path"])
    _require(
        cgroup_path.startswith("/") and ".." not in Path(cgroup_path).parts,
        "post-candidate exact Slurm cgroup path differs",
    )
    membership_path = (
        Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "cgroup.procs"
    )

    def read_members() -> tuple[int, ...]:
        try:
            rows = membership_path.read_text(encoding="ascii").splitlines()
        except (OSError, UnicodeError) as error:
            raise Reserve4GenerationError(
                "post-candidate exact Slurm cgroup membership is unavailable"
            ) from error
        _require(
            all(row.isdecimal() and int(row) > 1 for row in rows)
            and len(rows) == len(set(rows)),
            "post-candidate exact Slurm cgroup membership format differs",
        )
        return tuple(sorted(int(row) for row in rows))

    before = read_members()
    first = {pid: _candidate_cgroup_process_identity(pid) for pid in before}
    middle = read_members()
    second = {pid: _candidate_cgroup_process_identity(pid) for pid in middle}
    after = read_members()
    current_pid = os.getpid()
    supervisor_pid = int(monitor_start["supervisor_pid"])
    monitor_pid = int(monitor_start["monitor_pid"])
    allowed = {current_pid, supervisor_pid, monitor_pid}
    _require(
        before == middle == after
        and first == second
        and set(first) == allowed
        and all(row["cgroup_v2_path"] == cgroup_path for row in first.values())
        and all(row["state"] not in {"Z", "X", "x"} for row in first.values())
        and first[current_pid]["parent_pid"] == supervisor_pid
        and first[monitor_pid]["parent_pid"] == supervisor_pid
        and first[monitor_pid]["start_ticks"]
        == monitor_start["monitor_proc_start_ticks"]
        and first[supervisor_pid]["start_ticks"]
        == monitor_start["supervisor_proc_start_ticks"]
        and all(
            row["process_group_id"] != candidate_process_group_id
            for row in first.values()
        ),
        "candidate descendants/process group remain in the exact Slurm step",
    )
    return {
        "cgroup_v2_path": cgroup_path,
        "cgroup_procs_path": str(membership_path),
        "candidate_process_group_id": candidate_process_group_id,
        "allowed_process_roles": {
            "resource_process": first[current_pid],
            "launcher_supervisor": first[supervisor_pid],
            "memory_monitor": first[monitor_pid],
        },
        "stable_membership_before": list(before),
        "stable_membership_middle": list(middle),
        "stable_membership_after": list(after),
        "identities_and_start_ticks_replayed_stably": True,
        "candidate_process_group_members_after_wait": [],
        "unexpected_same_cgroup_process_count": 0,
        "next_candidate_launch_authorized": True,
    }


def _validate_post_candidate_cgroup_census_shape(
    value: Any, monitor: Mapping[str, Any]
) -> dict[str, Any]:
    identity_fields = {
        "pid",
        "state",
        "parent_pid",
        "process_group_id",
        "session_id",
        "start_ticks",
        "cgroup_v2_path",
    }
    roles = value.get("allowed_process_roles") if isinstance(value, Mapping) else None
    _require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "cgroup_v2_path",
            "cgroup_procs_path",
            "candidate_process_group_id",
            "allowed_process_roles",
            "stable_membership_before",
            "stable_membership_middle",
            "stable_membership_after",
            "identities_and_start_ticks_replayed_stably",
            "candidate_process_group_members_after_wait",
            "unexpected_same_cgroup_process_count",
            "next_candidate_launch_authorized",
        }
        and isinstance(roles, Mapping)
        and set(roles)
        == {"resource_process", "launcher_supervisor", "memory_monitor"}
        and all(
            isinstance(row, Mapping)
            and set(row) == identity_fields
            and type(row.get("pid")) is int
            and int(row["pid"]) > 1
            and row.get("state") not in {"Z", "X", "x"}
            and type(row.get("parent_pid")) is int
            and type(row.get("process_group_id")) is int
            and type(row.get("session_id")) is int
            and type(row.get("start_ticks")) is int
            and int(row["start_ticks"]) > 0
            and row.get("cgroup_v2_path") == value.get("cgroup_v2_path")
            for row in roles.values()
        )
        and value.get("cgroup_v2_path") == monitor["leaf_cgroup"]["path"]
        and value.get("cgroup_procs_path")
        == str(
            Path("/sys/fs/cgroup")
            / str(value["cgroup_v2_path"]).lstrip("/")
            / "cgroup.procs"
        )
        and type(value.get("candidate_process_group_id")) is int
        and int(value["candidate_process_group_id"]) > 1
        and roles["resource_process"]["parent_pid"]
        == roles["launcher_supervisor"]["pid"]
        and roles["memory_monitor"]["parent_pid"]
        == roles["launcher_supervisor"]["pid"]
        and roles["launcher_supervisor"]["pid"] == monitor["supervisor_pid"]
        and roles["launcher_supervisor"]["start_ticks"]
        == monitor["supervisor_proc_start_ticks"]
        and roles["memory_monitor"]["pid"] == monitor["monitor_pid"]
        and roles["memory_monitor"]["start_ticks"]
        == monitor["monitor_proc_start_ticks"]
        and all(
            row["process_group_id"] != value["candidate_process_group_id"]
            for row in roles.values()
        )
        and value.get("stable_membership_before")
        == value.get("stable_membership_middle")
        == value.get("stable_membership_after")
        == sorted(row["pid"] for row in roles.values())
        and value.get("identities_and_start_ticks_replayed_stably") is True
        and value.get("candidate_process_group_members_after_wait") == []
        and value.get("unexpected_same_cgroup_process_count") == 0
        and value.get("next_candidate_launch_authorized") is True,
        "post-candidate exact Slurm cgroup census differs",
    )
    return dict(value)


def _terminate_owned_candidate_process_group(
    process: subprocess.Popen[Any], start_ticks: int
) -> None:
    """Boundedly stop one exact session leader; never signal on PID ambiguity."""

    if process.poll() is not None:
        return
    if (
        _owned_candidate_process_observation(process, start_ticks) != "live"
        or os.getpgid(process.pid) != process.pid
    ):
        raise Reserve4GenerationError(
            "candidate process identity became ambiguous before termination"
        )
    os.killpg(process.pid, signal.SIGTERM)
    for _ in range(50):
        try:
            process.wait(timeout=0.1)
            return
        except subprocess.TimeoutExpired:
            observation = _owned_candidate_process_observation(
                process, start_ticks
            )
            if observation == "live":
                continue
            if observation == "dead":
                process.wait(timeout=0.1)
                return
            raise Reserve4GenerationError(
                "candidate process identity became ambiguous after TERM"
            )
    if (
        _owned_candidate_process_observation(process, start_ticks) != "live"
        or os.getpgid(process.pid) != process.pid
    ):
        raise Reserve4GenerationError(
            "candidate process identity became ambiguous before KILL"
        )
    os.killpg(process.pid, signal.SIGKILL)
    for _ in range(50):
        try:
            process.wait(timeout=0.1)
            return
        except subprocess.TimeoutExpired:
            observation = _owned_candidate_process_observation(
                process, start_ticks
            )
            if observation == "live":
                continue
            if observation == "dead":
                process.wait(timeout=0.1)
                return
            raise Reserve4GenerationError(
                "candidate process identity became ambiguous after KILL"
            )
    raise Reserve4GenerationError("candidate process group teardown timed out")


def _run_candidate_under_live_monitor(
    command: Sequence[str], environment: Mapping[str, str]
) -> dict[str, Any]:
    """Run one GPU candidate with a gated exact PGID and 10-ms monitor watch."""

    monitor_start = assert_live_host_cgroup_memory_monitor()
    process = subprocess.Popen(
        list(command),
        env=dict(environment),
        stdin=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    _require(process.stdin is not None, "candidate anonymous start gate is absent")
    try:
        start_ticks = _process_start_ticks(process.pid)
        _require(
            os.getpgid(process.pid) == process.pid,
            "candidate process is not its exact session/process-group leader",
        )
    except Exception:
        process.stdin.close()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as error:
            raise Reserve4GenerationError(
                "candidate failed before exact PID capture and did not stop on gate EOF"
            ) from error
        raise
    _require(
        _owned_candidate_process_observation(process, start_ticks) == "live",
        "candidate process died before anonymous start-gate release",
    )
    monitor_pid = int(monitor_start["monitor_pid"])
    monitor_ticks = int(monitor_start["monitor_proc_start_ticks"])
    supervisor_pid = int(monitor_start["supervisor_pid"])
    supervisor_ticks = int(monitor_start["supervisor_proc_start_ticks"])
    try:
        _require(
            _process_identity_is_live(monitor_pid, monitor_ticks)
            and _process_identity_is_live(supervisor_pid, supervisor_ticks),
            "host memory monitor/supervisor died before candidate gate release",
        )
        _assert_fresh_live_journal_tail(monitor_start)
    except Exception:
        process.stdin.close()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _terminate_owned_candidate_process_group(process, start_ticks)
        raise
    try:
        process.stdin.write(b"go\n")
        process.stdin.flush()
        process.stdin.close()
    except (BrokenPipeError, OSError) as error:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _terminate_owned_candidate_process_group(process, start_ticks)
        raise Reserve4GenerationError(
            "candidate anonymous start-gate publication failed"
        ) from error
    while True:
        status = process.poll()
        if status is not None:
            assert_live_host_cgroup_memory_monitor()
            if status != 0:
                raise subprocess.CalledProcessError(status, list(command))
            return _stable_post_candidate_cgroup_census(
                monitor_start,
                candidate_process_group_id=process.pid,
            )
        monitor_live = _process_identity_is_live(monitor_pid, monitor_ticks)
        supervisor_live = _process_identity_is_live(
            supervisor_pid, supervisor_ticks
        )
        try:
            _require(
                monitor_live and supervisor_live,
                "host memory monitor/supervisor died during GPU candidate",
            )
            _assert_fresh_live_journal_tail(monitor_start)
        except Exception:
            _terminate_owned_candidate_process_group(process, start_ticks)
            raise
        time.sleep(HOST_MEMORY_SAMPLE_INTERVAL_NS / 1_000_000_000)


def _smoke_task_binding(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": task["candidate_id"],
        "seed_slot": task["seed_slot"],
        "group_id": task["group_id"],
        "visible_gpus": task["visible_gpus"],
        "analysis_split": task["analysis_split"],
        "ordinal": task["ordinal"],
        "candidate_spec_path": task["candidate_spec_path"],
        "candidate_spec_sha256": task["candidate_spec_sha256"],
        "root_spec_sha256": task["root_spec_sha256"],
    }


def _embedded_receipt_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    _require(
        isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared,
        f"{label} embedded digest differs",
    )
    return declared


def _validated_tensor_identity(
    value: Any, *, label: str, expected_label: str
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} tensor identity is absent")
    _require(
        set(value)
        == {
            "shape",
            "dtype",
            "numel",
            "byte_count",
            "content_sha256",
            "raw_storage_sha256",
            "finite",
            "label",
        },
        f"{label} tensor identity fields differ",
    )
    shape = value.get("shape")
    numel = value.get("numel")
    byte_count = value.get("byte_count")
    _require(
        isinstance(shape, list)
        and len(shape) == 5
        and all(type(item) is int and item > 0 for item in shape)
        and type(numel) is int
        and numel > 0
        and numel == math.prod(shape)
        and type(byte_count) is int
        and byte_count == numel * 4
        and value.get("dtype") == "torch.float32"
        and value.get("finite") is True
        and value.get("label") == expected_label
        and SHA256_RE.fullmatch(str(value.get("raw_storage_sha256"))) is not None
        and SHA256_RE.fullmatch(str(value.get("content_sha256"))) is not None,
        f"{label} tensor identity values differ",
    )
    return {
        "raw_storage_sha256": value["raw_storage_sha256"],
        "content_sha256": value["content_sha256"],
        "shape": list(shape),
        "dtype": value["dtype"],
        "numel": numel,
        "byte_count": byte_count,
    }


def _declared_native_smoke_tensor_evidence(
    native_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the bound receipt declarations used by historical r10.

    The historical r10 disposable containers were not retained.  Its authority
    therefore remains the uniquely bound canonical native-receipt line in the
    pinned generation log.  This helper is deliberately *not* the current-run
    path: r13 current evidence must use :func:`_native_smoke_tensor_evidence`,
    which physically reopens both safetensors files before returning.
    """

    digest = _embedded_receipt_digest(native_receipt, label="native smoke receipt")
    _require(
        native_receipt.get("schema_version")
        == "bernini-native-identity-generation-canary-v2"
        and native_receipt.get("method")
        == "frozen-bernini-native-identity-generation-canary"
        and native_receipt.get("arms") == ["t2v"],
        "native smoke receipt method/schema differs",
    )
    sampling_root = native_receipt.get("sampling")
    sampling = (
        sampling_root.get("t2v")
        if isinstance(sampling_root, Mapping)
        else None
    )
    _require(
        isinstance(sampling, Mapping)
        and sampling.get("num_frames") == 81
        and sampling.get("num_inference_steps") == 40
        and sampling.get("target_initialization")
        == "official_gen_wanx22_fresh_gaussian"
        and sampling.get("target_mixed_with_source_latent") is False
        and sampling.get("custom_sampler_or_scheduler") is False
        and sampling.get("ulysses_size") == 4
        and type(sampling.get("seed")) is int,
        "native smoke sampling authority differs",
    )
    noise_root = native_receipt.get("initial_noise_artifacts")
    gaussian = noise_root.get("t2v") if isinstance(noise_root, Mapping) else None
    _require(isinstance(gaussian, Mapping), "native Gaussian artifact is absent")
    all_rank_gaussian = gaussian.get("all_rank_identity")
    _require(
        isinstance(all_rank_gaussian, Mapping)
        and set(all_rank_gaussian) == {"all_rank_exact", "identity"}
        and all_rank_gaussian.get("all_rank_exact") is True,
        "native Gaussian all-rank identity differs",
    )
    gaussian_identity = _validated_tensor_identity(
        all_rank_gaussian["identity"],
        label="official initial Gaussian",
        expected_label="official_initial_gaussian_t2v",
    )
    _require(
        gaussian.get("raw_value_sha256")
        == gaussian_identity["raw_storage_sha256"]
        and gaussian.get("tensor_value_sha256")
        == gaussian_identity["raw_storage_sha256"]
        and gaussian.get("content_sha256")
        == gaussian_identity["content_sha256"]
        and gaussian.get("shape") == gaussian_identity["shape"]
        and gaussian.get("dtype") == gaussian_identity["dtype"]
        and gaussian.get("stored_dtype") == gaussian_identity["dtype"]
        and gaussian.get("numel") == gaussian_identity["numel"]
        and gaussian.get("byte_count") == gaussian_identity["byte_count"]
        and gaussian.get("generator_initial_seed") == sampling["seed"]
        and gaussian.get("tensor_key") == "official_initial_gaussian"
        and gaussian.get("coordinate")
        == GAUSSIAN_SAFETENSORS_METADATA["coordinate"]
        and gaussian.get("origin") == GAUSSIAN_SAFETENSORS_METADATA["source"]
        and gaussian.get("observer_only") is True
        and gaussian.get("captured_from_native_sampler") is True
        and gaussian.get("external_initial_noise_injection") is False
        and gaussian.get("source_or_target_derived") is False
        and gaussian.get("observer_changed_return_value") is False
        and gaussian.get("roundtrip_raw_value_exact") is True
        and SHA256_RE.fullmatch(str(gaussian.get("sha256"))) is not None,
        "native Gaussian tensor/provenance binding differs",
    )
    generated_root = native_receipt.get("generated_identities")
    generated = (
        generated_root.get("t2v")
        if isinstance(generated_root, Mapping)
        else None
    )
    _require(
        isinstance(generated, Mapping)
        and set(generated) == {"all_rank_exact", "identity"}
        and generated.get("all_rank_exact") is True,
        "native generated latent all-rank identity differs",
    )
    clean_identity = _validated_tensor_identity(
        generated["identity"],
        label="predecode clean latent",
        expected_label="generated_t2v",
    )
    outputs_root = native_receipt.get("outputs")
    output = outputs_root.get("t2v") if isinstance(outputs_root, Mapping) else None
    clean = (
        output.get("normalized_clean_latent")
        if isinstance(output, Mapping)
        else None
    )
    _require(
        isinstance(output, Mapping)
        and output.get("frame_count") == 81
        and output.get("fps") == 25
        and type(output.get("height")) is int
        and type(output.get("width")) is int
        and output["height"] > 0
        and output["width"] > 0
        and SHA256_RE.fullmatch(str(output.get("sha256"))) is not None
        and isinstance(clean, Mapping)
        and clean.get("shape") == clean_identity["shape"]
        and clean.get("stored_dtype") == clean_identity["dtype"]
        and clean.get("sampler_return_dtype") == clean_identity["dtype"]
        and clean.get("tensor_key") == "normalized_clean_latent"
        and clean.get("artifact_role") == "native_sampler_proposal"
        and clean.get("coordinate") == "bernini_normalized_clean_vae_latent"
        and clean.get("origin") == "native_sampler_before_vae_decode"
        and clean.get("native_sampler_before_vae_decode") is True
        and clean.get("source_video_vae_encode_before_any_decode") is False
        and clean.get("mp4_decode_reencode_used") is False
        and clean.get("roundtrip_byte_exact_fp32") is True
        and SHA256_RE.fullmatch(str(clean.get("sha256"))) is not None,
        "native generated identity is not bound to the predecode artifact",
    )
    return {
        "schema_version": SMOKE_TENSOR_EVIDENCE_SCHEMA,
        "native_receipt_digest": digest,
        "mp4": {
            "file_sha256": output["sha256"],
            "frame_count": output["frame_count"],
            "fps": output["fps"],
            "height": output["height"],
            "width": output["width"],
        },
        "official_initial_gaussian": {
            **gaussian_identity,
            "tensor_value_sha256": gaussian["tensor_value_sha256"],
            "generator_initial_seed": gaussian["generator_initial_seed"],
            "artifact_binding": {
                "container_file_sha256": gaussian["sha256"],
                "tensor_key": gaussian["tensor_key"],
                "raw_value_sha256": gaussian["raw_value_sha256"],
                "stored_dtype": gaussian["stored_dtype"],
                "roundtrip_raw_value_exact": gaussian[
                    "roundtrip_raw_value_exact"
                ],
            },
        },
        "predecode_clean_latent": {
            **clean_identity,
            "artifact_binding": {
                "container_file_sha256": clean["sha256"],
                "tensor_key": clean["tensor_key"],
                "shape": clean["shape"],
                "stored_dtype": clean["stored_dtype"],
                "sampler_return_dtype": clean["sampler_return_dtype"],
                "coordinate": clean["coordinate"],
                "artifact_role": clean["artifact_role"],
                "origin": clean["origin"],
                "native_sampler_before_vae_decode": clean[
                    "native_sampler_before_vae_decode"
                ],
                "mp4_decode_reencode_used": clean[
                    "mp4_decode_reencode_used"
                ],
                "roundtrip_byte_exact_fp32": clean[
                    "roundtrip_byte_exact_fp32"
                ],
            },
        },
        "physical_safetensors_reopen": {
            "performed": False,
            "reason": (
                "historical-r10-disposable-containers-not-retained;"
                "authority-is-bound-canonical-native-receipt-log-line"
            ),
            "artifacts": [],
        },
        "safetensors_container_sha256_cross_process_equivalence_required": False,
    }


def _physical_safetensor_tensor_evidence(
    artifact: Mapping[str, Any],
    *,
    artifact_name: str,
    expected_key: str,
    expected_metadata: Mapping[str, str],
    expected_label: str,
    safe_open_factory: Any = None,
    tensor_identity_fn: Any = None,
) -> dict[str, Any]:
    """Reopen one current container and derive identity from its real tensor.

    The container SHA binds this particular file for audit, but is not a
    cross-process tensor-equivalence key.  The tensor identity is computed by
    the same ``infer_source_value_residual_oracle.tensor_identity`` algorithm
    used by the native worker.
    """

    _require(isinstance(artifact, Mapping), f"{artifact_name} artifact is absent")
    declared_path = artifact.get("path")
    _require(isinstance(declared_path, str), f"{artifact_name} path is absent")
    path = _plain_file(declared_path, f"{artifact_name} safetensors")
    _require(
        str(path) == declared_path and path.suffix == ".safetensors",
        f"{artifact_name} safetensors path binding differs",
    )
    before = path.stat()
    _require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and before.st_size > 0,
        f"{artifact_name} safetensors file identity differs",
    )
    container_sha_before = file_sha256(path)
    _require(
        SHA256_RE.fullmatch(str(artifact.get("sha256"))) is not None
        and container_sha_before == artifact["sha256"],
        f"{artifact_name} safetensors whole-file SHA-256 differs",
    )
    if safe_open_factory is None:
        try:
            from safetensors import safe_open as safe_open_factory
        except ImportError as error:  # pragma: no cover - AUH runtime dependency
            raise Reserve4GenerationError(
                "physical safetensors reopen requires safetensors"
            ) from error
    if tensor_identity_fn is None:
        try:
            import infer_source_value_residual_oracle as value_audit
        except ImportError as error:  # pragma: no cover - release closure failure
            raise Reserve4GenerationError(
                "physical tensor identity implementation is unavailable"
            ) from error
        tensor_identity_fn = value_audit.tensor_identity
    try:
        with safe_open_factory(str(path), framework="pt", device="cpu") as opened:
            keys = list(opened.keys())
            _require(
                keys == [expected_key],
                f"{artifact_name} safetensors must contain exactly {expected_key!r}",
            )
            metadata = dict(opened.metadata() or {})
            _require(
                metadata == dict(expected_metadata),
                f"{artifact_name} safetensors metadata differs",
            )
            tensor = opened.get_tensor(expected_key)
            identity = tensor_identity_fn(tensor, label=expected_label)
    except Reserve4GenerationError:
        raise
    except Exception as error:
        raise Reserve4GenerationError(
            f"cannot physically reopen {artifact_name} safetensors"
        ) from error
    validated = _validated_tensor_identity(
        identity, label=f"physical {artifact_name}", expected_label=expected_label
    )
    after = path.stat()
    container_sha_after = file_sha256(path)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
    _require(
        all(getattr(before, field) == getattr(after, field) for field in stable_fields)
        and container_sha_after == container_sha_before,
        f"{artifact_name} safetensors changed during physical reopen",
    )
    return {
        "name": artifact_name,
        "path": str(path),
        "container_file_sha256": container_sha_after,
        "container_sha256_role": "current-run-file-audit-only",
        "tensor_key": expected_key,
        "exact_single_tensor_key": True,
        "metadata": metadata,
        "metadata_digest": object_sha256(metadata),
        "tensor_identity": {
            **validated,
            "finite": True,
            "label": expected_label,
        },
    }


def _native_smoke_tensor_evidence(
    native_receipt: Mapping[str, Any],
    *,
    safe_open_factory: Any = None,
    tensor_identity_fn: Any = None,
) -> dict[str, Any]:
    """Physically reopen current Gaussian/clean containers and recompute identity."""

    evidence = _declared_native_smoke_tensor_evidence(native_receipt)
    gaussian = native_receipt["initial_noise_artifacts"]["t2v"]
    clean = native_receipt["outputs"]["t2v"]["normalized_clean_latent"]
    reopened = [
        _physical_safetensor_tensor_evidence(
            gaussian,
            artifact_name="official_initial_gaussian",
            expected_key="official_initial_gaussian",
            expected_metadata=GAUSSIAN_SAFETENSORS_METADATA,
            expected_label="official_initial_gaussian_t2v",
            safe_open_factory=safe_open_factory,
            tensor_identity_fn=tensor_identity_fn,
        ),
        _physical_safetensor_tensor_evidence(
            clean,
            artifact_name="predecode_clean_latent",
            expected_key="normalized_clean_latent",
            expected_metadata=CLEAN_LATENT_SAFETENSORS_METADATA,
            expected_label="generated_t2v",
            safe_open_factory=safe_open_factory,
            tensor_identity_fn=tensor_identity_fn,
        ),
    ]
    for row, evidence_key in zip(
        reopened, ("official_initial_gaussian", "predecode_clean_latent")
    ):
        observed = {
            key: row["tensor_identity"][key]
            for key in (
                "raw_storage_sha256",
                "content_sha256",
                "shape",
                "dtype",
                "numel",
                "byte_count",
            )
        }
        expected = {
            key: evidence[evidence_key][key]
            for key in observed
        }
        _require(
            observed == expected,
            f"physical {row['name']} tensor differs from receipt/all-rank identity",
        )
    return {
        **evidence,
        "physical_safetensors_reopen": {
            "performed": True,
            "loader": "safetensors.safe_open",
            "framework": "pt",
            "device": "cpu",
            "container_sha256_role": "current-run-file-audit-only",
            "artifacts": reopened,
        },
    }


def _legacy_r10_artifact_identities(
    native_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replay the historical r10 receipt projection solely as provenance."""

    output = native_receipt["outputs"]["t2v"]
    artifacts = {
        "mp4": output,
        "official_initial_gaussian": native_receipt[
            "initial_noise_artifacts"
        ]["t2v"],
        "predecode_clean_latent": output["normalized_clean_latent"],
    }
    rows: list[dict[str, Any]] = []
    for name in _SMOKE_ARTIFACT_NAMES:
        artifact = artifacts[name]
        metadata = dict(artifact)
        path = metadata.pop("path", None)
        _require(
            isinstance(path, str)
            and Path(path).is_absolute()
            and SHA256_RE.fullmatch(str(artifact.get("sha256"))) is not None,
            f"historical r10 artifact binding differs: {name}",
        )
        rows.append(
            {
                "name": name,
                "file_sha256": artifact["sha256"],
                "metadata_digest": object_sha256(metadata),
            }
        )
    return rows


def _validate_physical_safetensors_reopen_evidence(
    evidence: Mapping[str, Any], *, require_performed: bool
) -> bool:
    physical = evidence.get("physical_safetensors_reopen")
    _require(
        isinstance(physical, Mapping),
        "smoke physical safetensors reopen evidence is absent",
    )
    if physical.get("performed") is False:
        _require(
            not require_performed
            and set(physical) == {"performed", "reason", "artifacts"}
            and physical.get("reason")
            == (
                "historical-r10-disposable-containers-not-retained;"
                "authority-is-bound-canonical-native-receipt-log-line"
            )
            and physical.get("artifacts") == [],
            "historical smoke physical-reopen declaration differs",
        )
        return False
    _require(
        physical.get("performed") is True
        and set(physical)
        == {
            "performed",
            "loader",
            "framework",
            "device",
            "container_sha256_role",
            "artifacts",
        }
        and physical.get("loader") == "safetensors.safe_open"
        and physical.get("framework") == "pt"
        and physical.get("device") == "cpu"
        and physical.get("container_sha256_role")
        == "current-run-file-audit-only"
        and isinstance(physical.get("artifacts"), list)
        and len(physical["artifacts"]) == 2,
        "current smoke physical-reopen declaration differs",
    )
    specifications = (
        (
            "official_initial_gaussian",
            "official_initial_gaussian",
            GAUSSIAN_SAFETENSORS_METADATA,
            "official_initial_gaussian_t2v",
        ),
        (
            "predecode_clean_latent",
            "normalized_clean_latent",
            CLEAN_LATENT_SAFETENSORS_METADATA,
            "generated_t2v",
        ),
    )
    for row, (name, key, metadata, label) in zip(
        physical["artifacts"], specifications
    ):
        _require(
            isinstance(row, Mapping)
            and set(row)
            == {
                "name",
                "path",
                "container_file_sha256",
                "container_sha256_role",
                "tensor_key",
                "exact_single_tensor_key",
                "metadata",
                "metadata_digest",
                "tensor_identity",
            }
            and row.get("name") == name
            and isinstance(row.get("path"), str)
            and Path(row["path"]).is_absolute()
            and SHA256_RE.fullmatch(
                str(row.get("container_file_sha256"))
            )
            is not None
            and row.get("container_sha256_role")
            == "current-run-file-audit-only"
            and row.get("tensor_key") == key
            and row.get("exact_single_tensor_key") is True
            and row.get("metadata") == dict(metadata)
            and row.get("metadata_digest") == object_sha256(metadata),
            f"physical-reopen artifact evidence differs: {name}",
        )
        identity = row.get("tensor_identity")
        _require(
            isinstance(identity, Mapping)
            and set(identity)
            == {
                "raw_storage_sha256",
                "content_sha256",
                "shape",
                "dtype",
                "numel",
                "byte_count",
                "finite",
                "label",
            }
            and identity.get("finite") is True
            and identity.get("label") == label,
            f"physical-reopen tensor identity declaration differs: {name}",
        )
        main_key = name
        main = evidence.get(main_key)
        binding = main.get("artifact_binding") if isinstance(main, Mapping) else None
        projected = {
            field: identity[field]
            for field in (
                "raw_storage_sha256",
                "content_sha256",
                "shape",
                "dtype",
                "numel",
                "byte_count",
            )
        }
        _require(
            isinstance(main, Mapping)
            and isinstance(binding, Mapping)
            and projected == {field: main.get(field) for field in projected}
            and binding.get("container_file_sha256")
            == row["container_file_sha256"]
            and binding.get("tensor_key") == row["tensor_key"],
            f"physical-reopen tensor is not bound to smoke evidence: {name}",
        )
    return True


def _r10_parity_projection(evidence: Mapping[str, Any]) -> dict[str, Any]:
    gaussian = evidence.get("official_initial_gaussian")
    clean = evidence.get("predecode_clean_latent")
    mp4 = evidence.get("mp4")
    _require(
        set(evidence)
        == {
            "schema_version",
            "native_receipt_digest",
            "mp4",
            "official_initial_gaussian",
            "predecode_clean_latent",
            "physical_safetensors_reopen",
            "safetensors_container_sha256_cross_process_equivalence_required",
        }
        and evidence.get("schema_version") == SMOKE_TENSOR_EVIDENCE_SCHEMA
        and evidence.get(
            "safetensors_container_sha256_cross_process_equivalence_required"
        )
        is False
        and SHA256_RE.fullmatch(str(evidence.get("native_receipt_digest")))
        is not None
        and isinstance(mp4, Mapping)
        and set(mp4) == {"file_sha256", "frame_count", "fps", "height", "width"}
        and SHA256_RE.fullmatch(str(mp4.get("file_sha256"))) is not None
        and isinstance(gaussian, Mapping)
        and isinstance(clean, Mapping),
        "smoke tensor evidence schema differs",
    )
    _validate_physical_safetensors_reopen_evidence(
        evidence, require_performed=False
    )
    gaussian_binding = gaussian.get("artifact_binding")
    clean_binding = clean.get("artifact_binding")
    _require(
        isinstance(gaussian_binding, Mapping)
        and set(gaussian_binding)
        == {
            "container_file_sha256",
            "tensor_key",
            "raw_value_sha256",
            "stored_dtype",
            "roundtrip_raw_value_exact",
        }
        and SHA256_RE.fullmatch(
            str(gaussian_binding.get("container_file_sha256"))
        )
        is not None
        and gaussian_binding.get("tensor_key") == "official_initial_gaussian"
        and gaussian_binding.get("raw_value_sha256")
        == gaussian.get("raw_storage_sha256")
        and gaussian_binding.get("stored_dtype") == gaussian.get("dtype")
        and gaussian_binding.get("roundtrip_raw_value_exact") is True
        and isinstance(clean_binding, Mapping)
        and set(clean_binding)
        == {
            "container_file_sha256",
            "tensor_key",
            "shape",
            "stored_dtype",
            "sampler_return_dtype",
            "coordinate",
            "artifact_role",
            "origin",
            "native_sampler_before_vae_decode",
            "mp4_decode_reencode_used",
            "roundtrip_byte_exact_fp32",
        }
        and SHA256_RE.fullmatch(
            str(clean_binding.get("container_file_sha256"))
        )
        is not None
        and clean_binding.get("tensor_key") == "normalized_clean_latent"
        and clean_binding.get("shape") == clean.get("shape")
        and clean_binding.get("stored_dtype") == clean.get("dtype")
        and clean_binding.get("sampler_return_dtype") == clean.get("dtype")
        and clean_binding.get("coordinate")
        == "bernini_normalized_clean_vae_latent"
        and clean_binding.get("artifact_role") == "native_sampler_proposal"
        and clean_binding.get("origin") == "native_sampler_before_vae_decode"
        and clean_binding.get("native_sampler_before_vae_decode") is True
        and clean_binding.get("mp4_decode_reencode_used") is False
        and clean_binding.get("roundtrip_byte_exact_fp32") is True,
        "smoke tensor artifact binding differs",
    )
    identity_fields = {
        "raw_storage_sha256",
        "content_sha256",
        "shape",
        "dtype",
        "numel",
        "byte_count",
    }
    _require(
        set(gaussian)
        == identity_fields
        | {"tensor_value_sha256", "generator_initial_seed", "artifact_binding"}
        and set(clean) == identity_fields | {"artifact_binding"}
        and gaussian.get("tensor_value_sha256")
        == gaussian.get("raw_storage_sha256")
        and type(gaussian.get("generator_initial_seed")) is int
        and all(
            SHA256_RE.fullmatch(str(row.get(field))) is not None
            for row in (gaussian, clean)
            for field in ("raw_storage_sha256", "content_sha256")
        )
        and all(
            isinstance(row.get("shape"), list)
            and len(row["shape"]) == 5
            and row.get("dtype") == "torch.float32"
            and type(row.get("numel")) is int
            and type(row.get("byte_count")) is int
            and row["byte_count"] == row["numel"] * 4
            for row in (gaussian, clean)
        ),
        "smoke tensor equivalence identity differs",
    )
    return {
        "mp4": dict(mp4),
        "official_initial_gaussian": {
            key: gaussian[key]
            for key in (
                "raw_storage_sha256",
                "tensor_value_sha256",
                "content_sha256",
                "shape",
                "dtype",
                "numel",
                "byte_count",
                "generator_initial_seed",
            )
        },
        "predecode_clean_latent": {
            key: clean[key]
            for key in (
                "raw_storage_sha256",
                "content_sha256",
                "shape",
                "dtype",
                "numel",
                "byte_count",
            )
        },
        "predecode_artifact_binding": {
            key: clean_binding[key]
            for key in (
                "tensor_key",
                "shape",
                "stored_dtype",
                "sampler_return_dtype",
                "coordinate",
                "artifact_role",
                "origin",
                "native_sampler_before_vae_decode",
                "mp4_decode_reencode_used",
                "roundtrip_byte_exact_fp32",
            )
        },
    }


def load_r10_tensor_parity_authority(
    compile_smoke_receipt_path: str | Path,
    expected_compile_smoke_receipt_sha256: str,
    generation_log_path: str | Path,
    expected_generation_log_sha256: str,
) -> dict[str, Any]:
    """Derive r10 tensor authority from its sealed receipt and bound log line."""

    receipt, receipt_path, receipt_sha = _load_json(
        compile_smoke_receipt_path,
        "historical r10 compile-smoke receipt",
        expected_compile_smoke_receipt_sha256,
    )
    _require(
        receipt_path.read_bytes() == canonical_json_bytes(receipt) + b"\n",
        "historical r10 compile-smoke receipt is not canonical JSON",
    )
    receipt_digest = _embedded_receipt_digest(
        receipt, label="historical r10 compile-smoke receipt"
    )
    candidate = receipt.get("candidate_evidence")
    _require(
        set(receipt)
        == {
            "schema_version",
            "plan",
            "smoke_task",
            "runtime",
            "candidate_evidence",
            "world_size",
            "full_native_sampling_steps",
            "formal_candidate_count_at_gate",
            "disposable_output_deleted",
            "compile_smoke_passed",
            "training_performed",
            "optimizer_authorized",
            "receipt_digest",
        }
        and receipt.get("schema_version") == R10_COMPILE_SMOKE_SCHEMA
        and receipt.get("world_size") == 4
        and receipt.get("full_native_sampling_steps") == 40
        and receipt.get("formal_candidate_count_at_gate") == 0
        and receipt.get("disposable_output_deleted") is True
        and receipt.get("compile_smoke_passed") is True
        and receipt.get("training_performed") is False
        and receipt.get("optimizer_authorized") is False
        and isinstance(candidate, Mapping)
        and set(candidate)
        == {
            "candidate_receipt_file_sha256",
            "candidate_receipt_digest",
            "native_receipt_file_sha256",
            "native_receipt_digest",
            "artifact_identities",
        }
        and all(
            SHA256_RE.fullmatch(str(candidate.get(field))) is not None
            for field in (
                "candidate_receipt_file_sha256",
                "candidate_receipt_digest",
                "native_receipt_file_sha256",
                "native_receipt_digest",
            )
        )
        and isinstance(candidate.get("artifact_identities"), list),
        "historical r10 compile-smoke authority differs",
    )
    log_path = _plain_file(generation_log_path, "historical r10 generation log")
    log_raw = log_path.read_bytes()
    log_sha = hashlib.sha256(log_raw).hexdigest()
    _require(
        SHA256_RE.fullmatch(expected_generation_log_sha256) is not None
        and log_sha == expected_generation_log_sha256,
        "historical r10 generation log SHA-256 differs",
    )
    matches: list[tuple[dict[str, Any], bytes]] = []
    for line in log_raw.splitlines():
        if not line.startswith(b"{"):
            continue
        try:
            value = json.loads(
                line.decode("ascii"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if (
            isinstance(value, dict)
            and canonical_json_bytes(value) == line
            and hashlib.sha256(line + b"\n").hexdigest()
            == candidate["native_receipt_file_sha256"]
        ):
            matches.append((value, line))
    _require(
        len(matches) == 1,
        "historical r10 log does not contain one uniquely bound native receipt",
    )
    native_receipt, native_line = matches[0]
    native_digest = _embedded_receipt_digest(
        native_receipt, label="historical r10 native receipt"
    )
    _require(
        native_digest == candidate["native_receipt_digest"]
        and hashlib.sha256(native_line + b"\n").hexdigest()
        == candidate["native_receipt_file_sha256"],
        "historical r10 native receipt file/digest binding differs",
    )
    historical_rows = _legacy_r10_artifact_identities(native_receipt)
    _require(
        historical_rows == [dict(row) for row in candidate["artifact_identities"]],
        "historical r10 artifact identities are not derived from the bound log line",
    )
    tensor_evidence = _declared_native_smoke_tensor_evidence(native_receipt)
    _r10_parity_projection(tensor_evidence)
    unsigned = {
        "schema_version": R10_TENSOR_PARITY_AUTHORITY_SCHEMA,
        "source_evidence": {
            "compile_smoke_receipt": {
                "path": str(receipt_path),
                "file_sha256": receipt_sha,
                "receipt_digest": receipt_digest,
            },
            "generation_log": {
                "path": str(log_path),
                "file_sha256": log_sha,
            },
            "native_receipt": {
                "file_sha256": candidate["native_receipt_file_sha256"],
                "receipt_digest": native_digest,
                "unique_canonical_json_line_in_generation_log": True,
            },
        },
        "tensor_evidence": tensor_evidence,
        "comparison_policy": {
            "mp4_whole_file_sha256_exact": True,
            "current_run_physical_safetensors_safe_open_required": True,
            "current_run_exact_single_tensor_key_required": True,
            "current_run_exact_safetensors_metadata_required": True,
            "official_initial_gaussian_tensor_identity_exact": True,
            "predecode_clean_latent_generated_identity_exact": True,
            "predecode_artifact_semantic_binding_exact": True,
            "safetensors_container_sha256_cross_process_equivalence_required": False,
        },
    }
    return {**unsigned, "authority_digest": object_sha256(unsigned)}


def _replay_r10_tensor_parity_authority(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(authority)
    declared = unsigned.pop("authority_digest", None)
    source = authority.get("source_evidence")
    _require(
        authority.get("schema_version") == R10_TENSOR_PARITY_AUTHORITY_SCHEMA
        and isinstance(declared, str)
        and SHA256_RE.fullmatch(declared) is not None
        and object_sha256(unsigned) == declared
        and isinstance(source, Mapping)
        and isinstance(source.get("compile_smoke_receipt"), Mapping)
        and isinstance(source.get("generation_log"), Mapping),
        "r10 tensor-parity authority schema/digest differs",
    )
    receipt = source["compile_smoke_receipt"]
    log = source["generation_log"]
    replayed = load_r10_tensor_parity_authority(
        receipt["path"], receipt["file_sha256"], log["path"], log["file_sha256"]
    )
    _require(
        replayed == dict(authority),
        "r10 tensor-parity authority does not replay from sealed evidence",
    )
    return replayed


def _validate_r10_smoke_tensor_parity(
    current: Mapping[str, Any], authority: Mapping[str, Any]
) -> None:
    _validate_physical_safetensors_reopen_evidence(
        current, require_performed=True
    )
    replayed = _replay_r10_tensor_parity_authority(authority)
    expected = _r10_parity_projection(replayed["tensor_evidence"])
    observed = _r10_parity_projection(current)
    _require(
        observed == expected,
        "r13 compile smoke differs from r10 MP4/tensor identity authority",
    )


def load_compile_smoke_receipt(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    import infer_native_identity_generation_canary as native

    value, source, observed = _load_json(
        path, "compile-smoke receipt", expected_sha256
    )
    _require(
        source.read_bytes() == canonical_json_bytes(value) + b"\n",
        "compile-smoke receipt bytes are not canonical JSON",
    )
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    plan = value.get("plan")
    task = value.get("smoke_task")
    runtime = value.get("runtime")
    evidence = value.get("candidate_evidence")
    r10_authority = value.get("r10_tensor_parity_authority")
    retention = value.get("smoke_root_retention")
    _require(
        set(value)
        == {
            "schema_version",
            "plan",
            "smoke_task",
            "runtime",
            "candidate_evidence",
            "r10_tensor_parity_authority",
            "host_cgroup_memory_gate",
            "world_size",
            "full_native_sampling_steps",
            "formal_candidate_count_at_gate",
            "per_rank_gpu_peak_memory_receipt_required",
            "gpu_peak_reserved_limit_gib",
            "all_rank_gpu_peak_reserved_strictly_below_limit",
            "mp4_whole_file_sha256_parity_required",
            "gaussian_and_clean_tensor_identity_parity_required",
            "physical_safetensors_safe_open_recomputation_required",
            "exact_single_tensor_key_and_metadata_required",
            "safetensors_container_sha256_cross_process_equivalence_required",
            "smoke_root_retention",
            "retained_terminal_contract",
            "retained_nonreusable",
            "manual_cleanup_not_authorized",
            "no_cleanup_authority",
            "retained_at_receipt",
            "future_availability_guaranteed",
            "future_content_immutability_guaranteed",
            "persistence_after_step_or_reboot_guaranteed",
            "compile_smoke_passed",
            "training_performed",
            "optimizer_authorized",
            "receipt_digest",
        }
        and value.get("schema_version") == COMPILE_SMOKE_SCHEMA
        and declared == object_sha256(unsigned)
        and value.get("world_size") == 4
        and value.get("full_native_sampling_steps") == 40
        and value.get("formal_candidate_count_at_gate") == 0
        and value.get("per_rank_gpu_peak_memory_receipt_required") is True
        and value.get("gpu_peak_reserved_limit_gib")
        == T2V_GPU_MEMORY_LIMIT_GIB
        and value.get("all_rank_gpu_peak_reserved_strictly_below_limit") is True
        and value.get("mp4_whole_file_sha256_parity_required") is True
        and value.get("gaussian_and_clean_tensor_identity_parity_required") is True
        and value.get("physical_safetensors_safe_open_recomputation_required")
        is True
        and value.get("exact_single_tensor_key_and_metadata_required") is True
        and value.get(
            "safetensors_container_sha256_cross_process_equivalence_required"
        )
        is False
        and isinstance(runtime, Mapping)
        and isinstance(retention, Mapping)
        and set(retention)
        == {
            "path",
            "basename",
            "parent_path",
            "device",
            "inode",
            "uid",
            "gid",
            "mode_octal",
            "link_count_at_receipt",
            "canonical_non_symlink",
            "original_inode_bound",
            "retained_at_receipt",
            "retained_nonreusable",
            "manual_cleanup_not_authorized",
            "no_cleanup_authority",
            "future_availability_guaranteed",
            "future_content_immutability_guaranteed",
            "persistence_after_step_or_reboot_guaranteed",
        }
        and Path(str(retention.get("path"))).is_absolute()
        and Path(str(retention.get("path"))).parent
        == Path(str(retention.get("parent_path")))
        and retention.get("parent_path")
        == runtime.get("node_local_scratch", {}).get("path")
        and retention.get("basename") == Path(str(retention.get("path"))).name
        and str(retention.get("basename", "")).startswith(
            "generic-action-compile-smoke."
        )
        and all(
            type(retention.get(field)) is int and retention[field] > 0
            for field in ("device", "inode", "link_count_at_receipt")
        )
        and retention.get("uid") == 2012
        and retention.get("gid") == 2000
        and retention.get("mode_octal") == "0700"
        and retention.get("canonical_non_symlink") is True
        and retention.get("original_inode_bound") is True
        and retention.get("retained_at_receipt") is True
        and retention.get("retained_nonreusable") is True
        and retention.get("manual_cleanup_not_authorized") is True
        and retention.get("no_cleanup_authority") is True
        and retention.get("future_availability_guaranteed") is False
        and retention.get("future_content_immutability_guaranteed") is False
        and retention.get("persistence_after_step_or_reboot_guaranteed") is False
        and value.get("retained_terminal_contract") is True
        and value.get("retained_nonreusable") is True
        and value.get("manual_cleanup_not_authorized") is True
        and value.get("no_cleanup_authority") is True
        and value.get("retained_at_receipt") is True
        and value.get("future_availability_guaranteed") is False
        and value.get("future_content_immutability_guaranteed") is False
        and value.get("persistence_after_step_or_reboot_guaranteed") is False
        and value.get("compile_smoke_passed") is True
        and value.get("training_performed") is False
        and value.get("optimizer_authorized") is False,
        "compile-smoke receipt schema/digest/authority differs",
    )
    _require(
        isinstance(plan, Mapping)
        and set(plan) == {"path", "file_sha256", "plan_digest"}
        and all(
            isinstance(plan.get(field), str)
            and SHA256_RE.fullmatch(str(plan.get(field))) is not None
            for field in ("file_sha256", "plan_digest")
        )
        and Path(str(plan.get("path"))).is_absolute(),
        "compile-smoke plan binding differs",
    )
    _require(
        isinstance(task, Mapping)
        and set(task)
        == {
            "candidate_id",
            "seed_slot",
            "group_id",
            "visible_gpus",
            "analysis_split",
            "ordinal",
            "candidate_spec_path",
            "candidate_spec_sha256",
            "root_spec_sha256",
        }
        and task.get("seed_slot") == "seed1"
        and task.get("group_id") == "sp4-a"
        and task.get("analysis_split") == "fit"
        and task.get("visible_gpus") == [0, 1, 2, 3]
        and type(task.get("ordinal")) is int
        and Path(str(task.get("candidate_spec_path"))).is_absolute()
        and all(
            isinstance(task.get(field), str)
            and SHA256_RE.fullmatch(str(task.get(field))) is not None
            for field in ("candidate_spec_sha256", "root_spec_sha256")
        ),
        "compile-smoke sealed first-task binding differs",
    )
    _require(
        isinstance(runtime, Mapping)
        and set(runtime)
        == {
            "method_root",
            "python",
            "bernini_root",
            "veomni_root",
            "checkpoint",
            "checkpoint_content_manifest",
            "method_source_revision",
            "method_source_archive_sha256",
            "generation_worker",
            "rank_cache_wrapper",
            "verified_release_execution",
            "preprocessing_tools",
            "node_local_scratch",
            "serialized_host_checkpoint_load",
            "t2v_text_encoder_rank_gpu_residency",
            "host_cgroup_sampled_memory_monitor",
        }
        and runtime.get("preprocessing_tools") == PREPROCESSING_TOOL_SHA256
        and isinstance(runtime.get("node_local_scratch"), Mapping)
        and set(runtime["node_local_scratch"]) == {"path", "filesystem_type"}
        and Path(str(runtime["node_local_scratch"].get("path"))).is_absolute()
        and runtime["node_local_scratch"].get("filesystem_type")
        in _ALLOWED_NODE_LOCAL_FILESYSTEMS
        and isinstance(runtime.get("serialized_host_checkpoint_load"), Mapping)
        and runtime["serialized_host_checkpoint_load"]
        == {
            "required": True,
            "environment_variable": "NATIVE_V_AXIS_LOAD_LOCK",
            "path": runtime["serialized_host_checkpoint_load"].get("path"),
            "sha256": EMPTY_FILE_SHA256,
            "mode": "0400",
            "device": runtime["serialized_host_checkpoint_load"].get("device"),
            "inode": runtime["serialized_host_checkpoint_load"].get("inode"),
            "uid": 2012,
            "gid": 2000,
            "link_count": 1,
            "size_bytes": 0,
            "parent_is_authenticated_node_local_scratch": True,
            "lock_held_through_model_to_rank_gpu_and_malloc_trim": True,
        }
        and Path(
            str(runtime["serialized_host_checkpoint_load"].get("path"))
        ).is_absolute()
        and Path(
            str(runtime["serialized_host_checkpoint_load"].get("path"))
        ).parent
        == Path(str(runtime["node_local_scratch"].get("path")))
        and all(
            type(runtime["serialized_host_checkpoint_load"].get(field)) is int
            and runtime["serialized_host_checkpoint_load"][field] > 0
            for field in ("device", "inode")
        )
        and runtime.get("t2v_text_encoder_rank_gpu_residency")
        == {
            "required": True,
            "environment_variable": "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
            "official_model_sample_preserved": True,
            "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
            "all_other_to_requests_delegated": True,
            "text_encoder_retired_only_with_renderer": True,
            "gpu_memory_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
            "gpu_memory_limit_bytes": T2V_GPU_MEMORY_LIMIT_BYTES,
            "per_rank_max_allocated_and_reserved_receipt_required": True,
            "all_rank_peak_reserved_strictly_below_limit_required": True,
        }
        and all(
            isinstance(runtime.get(field), str)
            and Path(str(runtime.get(field))).is_absolute()
            for field in (
                "method_root",
                "bernini_root",
                "veomni_root",
                "checkpoint",
            )
        )
        and REVISION_RE.fullmatch(str(runtime.get("method_source_revision")))
        is not None
        and SHA256_RE.fullmatch(
            str(runtime.get("method_source_archive_sha256"))
        )
        is not None,
        "compile-smoke runtime binding differs",
    )
    monitor = runtime.get("host_cgroup_sampled_memory_monitor")
    _require(
        isinstance(monitor, Mapping)
        and set(monitor)
        == {
            "required", "start_receipt", "sample_journal", "monitor_pid",
            "stop_capability",
            "monitor_proc_start_ticks", "supervisor_pid",
            "supervisor_proc_start_ticks", "slurm_job_id", "slurm_step_id",
            "memory_max_exact_gib", "sampled_current_safe_ceiling_gib",
            "sample_interval_ns", "maximum_sample_gap_ns",
            "sampling_source", "zero_oom_and_oom_kill_required",
            "coverage",
        }
        and monitor.get("required") is True
        and isinstance(monitor.get("stop_capability"), Mapping)
        and monitor.get("slurm_job_id") == "136141"
        and isinstance(monitor.get("slurm_step_id"), str)
        and str(monitor["slurm_step_id"]).isdecimal()
        and monitor.get("memory_max_exact_gib") == HOST_MEMORY_LIMIT_GIB
        and monitor.get("sampled_current_safe_ceiling_gib")
        == HOST_MEMORY_SAFE_CEILING_GIB
        and monitor.get("sample_interval_ns") == HOST_MEMORY_SAMPLE_INTERVAL_NS
        and monitor.get("maximum_sample_gap_ns") == HOST_MEMORY_MAX_SAMPLE_GAP_NS
        and monitor.get("sampling_source")
        == "cgroup_v2_memory.current_fixed_10ms"
        and monitor.get("zero_oom_and_oom_kill_required") is True
        and monitor.get("coverage")
        == "before_compile_smoke_through_terminal_after_formal40",
        "compile-smoke host memory monitor runtime binding differs",
    )
    _validate_monitor_stop_capability(monitor.get("stop_capability"))
    monitor_start_reference = monitor.get("start_receipt")
    monitor_journal = monitor.get("sample_journal")
    _require(
        isinstance(monitor_start_reference, Mapping)
        and set(monitor_start_reference)
        == {"path", "file_sha256", "receipt_digest"}
        and Path(str(monitor_start_reference.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(
            str(monitor_start_reference.get("file_sha256"))
        ) is not None
        and SHA256_RE.fullmatch(
            str(monitor_start_reference.get("receipt_digest"))
        ) is not None
        and isinstance(monitor_journal, Mapping)
        and set(monitor_journal)
        == {"path", "device", "inode", "record_size", "record_encoding"}
        and Path(str(monitor_journal.get("path"))).is_absolute()
        and type(monitor_journal.get("device")) is int
        and type(monitor_journal.get("inode")) is int
        and monitor_journal.get("record_size") == HOST_MEMORY_SAMPLE_STRUCT.size
        and monitor_journal.get("record_encoding") == HOST_MEMORY_SAMPLE_ENCODING
        and all(
            type(monitor.get(field)) is int and int(monitor[field]) > 0
            for field in (
                "monitor_pid",
                "monitor_proc_start_ticks",
                "supervisor_pid",
                "supervisor_proc_start_ticks",
            )
        ),
        "compile-smoke host memory monitor identity/journal binding differs",
    )
    for field in (
        "python",
        "checkpoint_content_manifest",
        "generation_worker",
        "rank_cache_wrapper",
    ):
        reference = runtime.get(field)
        _require(
            isinstance(reference, Mapping)
            and set(reference) == {"path", "sha256"}
            and Path(str(reference.get("path"))).is_absolute()
            and SHA256_RE.fullmatch(str(reference.get("sha256"))) is not None,
            f"compile-smoke runtime {field} binding differs",
        )
    verified_execution = runtime.get("verified_release_execution")
    torchrun_launcher = (
        verified_execution.get("torchrun_runtime_snapshot")
        if isinstance(verified_execution, Mapping)
        else None
    )
    _require(
        isinstance(verified_execution, Mapping)
        and set(verified_execution)
        == {
            "isolated_python_flags",
            "held_fd_frozen_python_execution",
            "manifest",
            "runner",
            "torchrun_runtime_snapshot",
            "torchrun_source_executed_from_stable_captured_in_memory_bytes",
            "torchrun_launcher_preauthorized_by_release",
            "torchrun_runtime_snapshot_only",
            "site_packages_dependency_bytes_preauthorized_by_release",
            "torchrun_automatic_site_initialization_disabled",
            "rank_wrapper_executed_from_verified_in_memory_bytes",
            "worker_and_release_imports_served_from_verified_in_memory_bytes",
            "caller_python_environment_presence_rejected",
            "rank_shell_startup_environment_scrubbed",
            "safe_runtime_path",
            "candidate_gpu_process_group_monitored_every_10ms",
            "per_candidate_post_exit_stable_step_cgroup_census_before_next_gpu",
        }
        and verified_execution.get("isolated_python_flags") == ["-I", "-S", "-B"]
        and verified_execution.get("held_fd_frozen_python_execution")
        == {
            "root_owned_bootstrap": {
                "path": str(ROOT_BOOTSTRAP_PYTHON_PATH),
                "sha256": ROOT_BOOTSTRAP_PYTHON_SHA256,
                "size_bytes": ROOT_BOOTSTRAP_PYTHON_SIZE,
                "uid": 0,
                "gid": 0,
                "mode": "0755",
                "link_count": 1,
                "isolated_flags": ["-I", "-S", "-s", "-B"],
            },
            "frozen_target": {
                "path": str(FROZEN_PYTHON_PATH),
                "sha256": FROZEN_PYTHON_SHA256,
                "size_bytes": FROZEN_PYTHON_SIZE,
                "uid": 2012,
                "gid": 2000,
                "mode": "0755",
                "link_count": 1,
            },
            "verified_runner_subcommand": "held-fd-exec-frozen-python",
            "execution_boundaries": [
                "torchrun-coordinator",
                "per-rank-worker",
            ],
            "target_opened_no_follow_and_double_sha256_verified": True,
            "target_execve_uses_retained_file_descriptor": True,
            "target_named_path_is_never_passed_to_execve": True,
            "candidate_process_anonymous_start_gate_before_target_exec": True,
        }
        and all(
            isinstance(verified_execution.get(name), Mapping)
            and set(verified_execution[name]) == {"path", "sha256"}
            and Path(str(verified_execution[name].get("path"))).is_absolute()
            and SHA256_RE.fullmatch(
                str(verified_execution[name].get("sha256"))
            )
            is not None
            for name in ("manifest", "runner")
        )
        and isinstance(torchrun_launcher, Mapping)
        and set(torchrun_launcher) == {"path", "sha256", "site_packages"}
        and Path(str(torchrun_launcher.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(str(torchrun_launcher.get("sha256"))) is not None
        and torchrun_launcher.get("site_packages") == str(FROZEN_SITE_PACKAGES)
        and verified_execution.get(
            "torchrun_source_executed_from_stable_captured_in_memory_bytes"
        )
        is True
        and verified_execution.get("torchrun_launcher_preauthorized_by_release")
        is False
        and verified_execution.get("torchrun_runtime_snapshot_only") is True
        and verified_execution.get(
            "site_packages_dependency_bytes_preauthorized_by_release"
        )
        is False
        and verified_execution.get(
            "torchrun_automatic_site_initialization_disabled"
        )
        is True
        and verified_execution.get(
            "rank_wrapper_executed_from_verified_in_memory_bytes"
        )
        is True
        and verified_execution.get(
            "worker_and_release_imports_served_from_verified_in_memory_bytes"
        )
        is True
        and verified_execution.get("caller_python_environment_presence_rejected")
        is True
        and verified_execution.get("rank_shell_startup_environment_scrubbed")
        is True
        and verified_execution.get("safe_runtime_path") == SAFE_RUNTIME_PATH
        and verified_execution.get(
            "candidate_gpu_process_group_monitored_every_10ms"
        )
        is True
        and verified_execution.get(
            "per_candidate_post_exit_stable_step_cgroup_census_before_next_gpu"
        )
        is True,
        "compile-smoke verified release execution binding differs",
    )
    _require(
        isinstance(evidence, Mapping)
        and set(evidence)
        == {
            "candidate_receipt_file_sha256",
            "candidate_receipt_digest",
            "native_receipt_file_sha256",
            "native_receipt_digest",
            "resource_lifecycle",
            "tensor_parity_evidence",
            "post_candidate_cgroup_census",
        }
        and all(
            SHA256_RE.fullmatch(str(evidence.get(field))) is not None
            for field in (
                "candidate_receipt_file_sha256",
                "candidate_receipt_digest",
                "native_receipt_file_sha256",
                "native_receipt_digest",
            )
        )
        and isinstance(evidence.get("tensor_parity_evidence"), Mapping),
        "compile-smoke candidate evidence differs",
    )
    _validate_post_candidate_cgroup_census_shape(
        evidence.get("post_candidate_cgroup_census"), monitor
    )
    _require(
        evidence["native_receipt_digest"]
        == evidence["tensor_parity_evidence"].get("native_receipt_digest"),
        "compile-smoke tensor evidence is not bound to its native receipt",
    )
    try:
        lifecycle = native.validate_t2v_resource_lifecycle(
            evidence.get("resource_lifecycle"), require_serialized_load=True
        )
    except native.NativeIdentityCanaryError as error:
        raise Reserve4GenerationError(
            "compile-smoke did not prove WORLD4 load completion and GPU "
            "resource-lifecycle closure before sampling"
        ) from error
    residency = lifecycle["world4_t2v_text_encoder_gpu_residency_gate"]
    _require(
        residency["all_rank_peak_reserved_within_52_gib"] is True
        and len(residency["rank_evidence"]) == 4
        and max(
            int(row["gpu_peak_reserved_bytes"])
            for row in residency["rank_evidence"]
        )
        < T2V_GPU_MEMORY_LIMIT_BYTES,
        "compile-smoke WORLD4 GPU peak-reserved gate failed",
    )
    host_gate_reference = value.get("host_cgroup_memory_gate")
    _require(
        isinstance(host_gate_reference, Mapping)
        and set(host_gate_reference)
        == {"path", "file_sha256", "receipt_digest"}
        and Path(str(host_gate_reference.get("path"))).is_absolute()
        and SHA256_RE.fullmatch(
            str(host_gate_reference.get("file_sha256"))
        ) is not None
        and SHA256_RE.fullmatch(
            str(host_gate_reference.get("receipt_digest"))
        ) is not None,
        "compile-smoke host memory gate reference differs",
    )
    host_gate, _, _ = load_host_cgroup_memory_gate(
        host_gate_reference["path"],
        host_gate_reference["file_sha256"],
        expected_phase="compile_smoke_before_formal40",
        require_monitor_alive_now=False,
    )
    _require(
        host_gate["receipt_digest"] == host_gate_reference["receipt_digest"]
        and host_gate["monitor_start_receipt"] == monitor_start_reference
        and {
            key: host_gate["sample_journal"][key]
            for key in ("path", "device", "inode", "record_size", "record_encoding")
        }
        == monitor_journal
        and host_gate["monitor_pid"] == monitor["monitor_pid"]
        and host_gate["monitor_proc_start_ticks"]
        == monitor["monitor_proc_start_ticks"]
        and host_gate["supervisor_pid"] == monitor["supervisor_pid"]
        and host_gate["supervisor_proc_start_ticks"]
        == monitor["supervisor_proc_start_ticks"]
        and host_gate["monitor_stop_capability"]
        == monitor["stop_capability"]
        and host_gate["slurm_job_id"] == monitor["slurm_job_id"]
        and host_gate["slurm_step_id"] == monitor["slurm_step_id"]
        and host_gate["cgroup_memory_max_exactly_60_gib"] is True
        and host_gate["sampled_peak_strictly_below_56_gib"] is True,
        "compile-smoke host memory gate/runtime replay differs",
    )
    _require(
        isinstance(r10_authority, Mapping),
        "compile-smoke r10 tensor-parity authority is absent",
    )
    _validate_r10_smoke_tensor_parity(
        evidence["tensor_parity_evidence"], r10_authority
    )
    return value, source, observed


def _load_compile_smoke_receipt_postretention_attested(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path, str]:
    """Replay sealed/shared smoke evidence without child-live or scratch access.

    This private entry has no caller-selectable bypass flag.  The common loader
    above already performs canonical schema/digest, shared journal-prefix,
    tensor, media, and rank evidence replay with its live-monitor and live-
    cgroup flags fixed false; it does not resolve the retained smoke root.
    """

    return load_compile_smoke_receipt(path, expected_sha256)


def _validate_compile_smoke_for_runtime(
    args: argparse.Namespace,
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_sha: str,
    runtime: Mapping[str, Any],
) -> Mapping[str, Any]:
    receipt, _, _ = load_compile_smoke_receipt(
        args.compile_smoke_receipt, args.expected_compile_smoke_receipt_sha256
    )
    expected_plan = {
        "path": str(plan_path),
        "file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
    }
    _require(receipt["plan"] == expected_plan, "compile-smoke plan replay differs")
    _require(
        receipt["smoke_task"] == _smoke_task_binding(plan["tasks"][0]),
        "compile-smoke first-task replay differs",
    )
    _require(receipt["runtime"] == runtime, "compile-smoke runtime replay differs")
    replay_retained_compile_smoke_root(receipt)
    host_reference = receipt["host_cgroup_memory_gate"]
    host_gate, _, _ = load_host_cgroup_memory_gate(
        host_reference["path"],
        host_reference["file_sha256"],
        expected_phase="compile_smoke_before_formal40",
        require_monitor_alive_now=True,
    )
    _require(
        host_gate["receipt_digest"] == host_reference["receipt_digest"],
        "compile-smoke host memory gate digest replay differs",
    )
    assert_live_host_cgroup_memory_monitor()
    return receipt


def _retained_smoke_root_identity(root: Path, scratch: Path) -> dict[str, Any]:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_scratch = scratch.resolve(strict=True)
        metadata = root.lstat()
    except OSError as error:
        raise Reserve4GenerationError("retained smoke root is unavailable") from error
    _require(
        root == resolved_root
        and not root.is_symlink()
        and root.is_dir()
        and resolved_root.parent == resolved_scratch
        and root.name.startswith("generic-action-compile-smoke."),
        "retained smoke root identity differs",
    )
    return {
        "path": str(root),
        "basename": root.name,
        "parent_path": str(scratch),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "link_count_at_receipt": metadata.st_nlink,
        "canonical_non_symlink": True,
        "original_inode_bound": True,
        "retained_at_receipt": True,
        "retained_nonreusable": True,
        "manual_cleanup_not_authorized": True,
        "no_cleanup_authority": True,
        "future_availability_guaranteed": False,
        "future_content_immutability_guaranteed": False,
        "persistence_after_step_or_reboot_guaranteed": False,
    }


def replay_retained_compile_smoke_root(receipt: Mapping[str, Any]) -> Path:
    retention = receipt["smoke_root_retention"]
    root = _plain_dir(retention["path"], "retained compile-smoke root")
    scratch = _plain_dir(
        receipt["runtime"]["node_local_scratch"]["path"],
        "compile-smoke node-local scratch",
    )
    observed = _retained_smoke_root_identity(root, scratch)
    _require(
        all(
            observed[field] == retention[field]
            for field in (
                "path",
                "basename",
                "parent_path",
                "device",
                "inode",
                "uid",
                "gid",
                "mode_octal",
                "canonical_non_symlink",
                "original_inode_bound",
                "retained_at_receipt",
                "retained_nonreusable",
                "manual_cleanup_not_authorized",
                "no_cleanup_authority",
                "future_availability_guaranteed",
                "future_content_immutability_guaranteed",
                "persistence_after_step_or_reboot_guaranteed",
            )
        )
        and observed["link_count_at_receipt"] >= retention["link_count_at_receipt"],
        "retained compile-smoke root was renamed/recreated or drifted",
    )
    return root


def run_compile_smoke_sp4(args: argparse.Namespace) -> int:
    import infer_pair_v5_t2v_calibration_bank as renderer

    plan, plan_path, plan_sha = load_plan(args.plan, args.expected_plan_sha256)
    task = plan["tasks"][0]
    _require(
        task["seed_slot"] == "seed1"
        and task["group_id"] == "sp4-a"
        and task["analysis_split"] == "fit"
        and task["visible_gpus"] == [0, 1, 2, 3],
        "sealed compile-smoke candidate is not first fit/SP4-A task",
    )
    expected_visible = "0,1,2,3"
    _require(
        os.environ.get("ROCR_VISIBLE_DEVICES") == expected_visible,
        "compile-smoke ROCR_VISIBLE_DEVICES differs",
    )
    (
        runtime,
        python,
        worker,
        rank_exec,
        rank_exec_source,
        torchrun_source,
        scratch,
    ) = _runtime_binding(args)
    r10_authority = load_r10_tensor_parity_authority(
        args.r10_compile_smoke_receipt,
        args.expected_r10_compile_smoke_receipt_sha256,
        args.r10_generation_log,
        args.expected_r10_generation_log_sha256,
    )
    receipt_output = Path(args.receipt_output)
    _require(
        receipt_output.is_absolute()
        and receipt_output.parent.is_dir()
        and not receipt_output.parent.is_symlink()
        and not receipt_output.exists()
        and not receipt_output.is_symlink(),
        "compile-smoke receipt output must be fresh",
    )
    smoke_root = Path(
        tempfile.mkdtemp(prefix="generic-action-compile-smoke.", dir=scratch)
    ).resolve(strict=True)
    candidate_output = smoke_root / str(task["candidate_id"])
    command = _candidate_command(
        args,
        task=task,
        candidate_output=candidate_output,
        python=python,
        worker=worker,
        rank_exec=rank_exec,
        rank_exec_source=rank_exec_source,
        torchrun_source=torchrun_source,
        runtime=runtime,
    )
    environment = _candidate_environment(
        expected_visible=expected_visible,
        python=python,
        scratch=scratch,
        cache_token="compile-smoke-" + object_sha256(_smoke_task_binding(task))[:20],
        runtime=runtime,
    )
    candidate_evidence: Optional[dict[str, Any]] = None
    try:
        candidate_census = _run_candidate_under_live_monitor(command, environment)
        candidate_receipt_path = (
            candidate_output / "pair-v5-t2v-calibration-receipt.json"
        )
        candidate_receipt, _ = _validate_candidate_receipt(
            task, candidate_receipt_path
        )
        native_receipt = renderer._load_json(  # type: ignore[attr-defined]
            _plain_file(candidate_receipt["native_receipt_path"], "native receipt"),
            "compile-smoke native receipt",
        )
        resource_lifecycle = renderer.native.validate_t2v_resource_lifecycle(  # type: ignore[attr-defined]
            native_receipt.get("resource_lifecycle"), require_serialized_load=True
        )
        tensor_parity_evidence = _native_smoke_tensor_evidence(native_receipt)
        _require(
            candidate_receipt["native_receipt_digest"]
            == tensor_parity_evidence["native_receipt_digest"],
            "compile-smoke tensor evidence is not bound to its native receipt",
        )
        _validate_r10_smoke_tensor_parity(
            tensor_parity_evidence, r10_authority
        )
        candidate_evidence = {
            "candidate_receipt_file_sha256": file_sha256(candidate_receipt_path),
            "candidate_receipt_digest": candidate_receipt["receipt_digest"],
            "native_receipt_file_sha256": candidate_receipt["native_receipt_sha256"],
            "native_receipt_digest": candidate_receipt["native_receipt_digest"],
            "resource_lifecycle": resource_lifecycle,
            "tensor_parity_evidence": tensor_parity_evidence,
            "post_candidate_cgroup_census": candidate_census,
        }
    except subprocess.CalledProcessError as error:
        raise Reserve4GenerationError(
            "full native40 compile smoke failed; formal40 remains forbidden"
        ) from error
    _require(candidate_evidence is not None, "compile-smoke evidence is absent")
    smoke_root_retention = _retained_smoke_root_identity(smoke_root, scratch)
    host_gate, host_gate_sha = write_host_cgroup_memory_gate_receipt(
        Path(args.host_memory_gate_output),
        measurement_phase="compile_smoke_before_formal40",
        formal_candidate_count_at_gate=0,
    )
    unsigned = {
        "schema_version": COMPILE_SMOKE_SCHEMA,
        "plan": {
            "path": str(plan_path),
            "file_sha256": plan_sha,
            "plan_digest": plan["plan_digest"],
        },
        "smoke_task": _smoke_task_binding(task),
        "runtime": runtime,
        "candidate_evidence": candidate_evidence,
        "r10_tensor_parity_authority": r10_authority,
        "host_cgroup_memory_gate": {
            "path": str(Path(args.host_memory_gate_output)),
            "file_sha256": host_gate_sha,
            "receipt_digest": host_gate["receipt_digest"],
        },
        "world_size": 4,
        "full_native_sampling_steps": 40,
        "formal_candidate_count_at_gate": 0,
        "per_rank_gpu_peak_memory_receipt_required": True,
        "gpu_peak_reserved_limit_gib": T2V_GPU_MEMORY_LIMIT_GIB,
        "all_rank_gpu_peak_reserved_strictly_below_limit": True,
        "mp4_whole_file_sha256_parity_required": True,
        "gaussian_and_clean_tensor_identity_parity_required": True,
        "physical_safetensors_safe_open_recomputation_required": True,
        "exact_single_tensor_key_and_metadata_required": True,
        "safetensors_container_sha256_cross_process_equivalence_required": False,
        "smoke_root_retention": smoke_root_retention,
        "retained_terminal_contract": True,
        "retained_nonreusable": True,
        "manual_cleanup_not_authorized": True,
        "no_cleanup_authority": True,
        "retained_at_receipt": True,
        "future_availability_guaranteed": False,
        "future_content_immutability_guaranteed": False,
        "persistence_after_step_or_reboot_guaranteed": False,
        "compile_smoke_passed": True,
        "training_performed": False,
        "optimizer_authorized": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    receipt_sha = _write_create_only(receipt_output, receipt)
    load_compile_smoke_receipt(receipt_output, receipt_sha)
    return 0


def run_sp4(args: argparse.Namespace) -> int:
    plan, plan_path, plan_sha = load_plan(args.plan, args.expected_plan_sha256)
    matching = [
        row for row in plan["tasks"]
        if row["seed_slot"] == args.seed_slot and row["group_id"] == args.group_id
    ]
    _require(matching, "requested shard is empty or absent from the sealed plan")
    expected_visible = ",".join(str(item) for item in matching[0]["visible_gpus"])
    _require(
        all(
            row["visible_gpus"] == matching[0]["visible_gpus"]
            and row["analysis_split"] == plan["analysis_split"]
            for row in matching
        ),
        "requested shard topology/split differs",
    )
    _require(
        os.environ.get("ROCR_VISIBLE_DEVICES") == expected_visible,
        f"ROCR_VISIBLE_DEVICES must equal sealed shard mapping {expected_visible}",
    )
    (
        runtime,
        python,
        worker,
        rank_exec,
        rank_exec_source,
        torchrun_source,
        scratch,
    ) = _runtime_binding(args)
    _validate_compile_smoke_for_runtime(
        args,
        plan=plan,
        plan_path=plan_path,
        plan_sha=plan_sha,
        runtime=runtime,
    )
    output = Path(args.output_dir)
    _require(
        output.is_absolute()
        and output != Path("/")
        and not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.parent.is_symlink(),
        "shard output must be a fresh absolute directory",
    )
    output.mkdir(mode=0o700)
    validated: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    receipt_rows: list[dict[str, Any]] = []
    for task in matching:
        assert_live_host_cgroup_memory_monitor()
        candidate_output = output / task["candidate_id"]
        _require(
            not candidate_output.exists() and not candidate_output.is_symlink(),
            f"refusing candidate output reuse: {task['candidate_id']}",
        )
        command = _candidate_command(
            args,
            task=task,
            candidate_output=candidate_output,
            python=python,
            worker=worker,
            rank_exec=rank_exec,
            rank_exec_source=rank_exec_source,
            torchrun_source=torchrun_source,
            runtime=runtime,
        )
        environment = _candidate_environment(
            expected_visible=expected_visible,
            python=python,
            scratch=scratch,
            cache_token=(
                "formal-"
                + str(task["seed_slot"])
                + "-"
                + str(task["group_id"])
                + "-"
                + str(task["ordinal"])
            ),
            runtime=runtime,
        )
        try:
            candidate_census = _run_candidate_under_live_monitor(
                command, environment
            )
        except subprocess.CalledProcessError as error:
            raise Reserve4GenerationError(
                f"generation failed for {task['candidate_id']}; partial shard is non-authoritative"
            ) from error
        assert_live_host_cgroup_memory_monitor()
        receipt_path = candidate_output / "pair-v5-t2v-calibration-receipt.json"
        receipt, _ = _validate_candidate_receipt(task, receipt_path)
        validated.append((task, receipt))
        receipt_rows.append(
            {
                "candidate_id": task["candidate_id"],
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path),
                "receipt_digest": receipt["receipt_digest"],
                "post_candidate_cgroup_census": candidate_census,
            }
        )
    assert_live_host_cgroup_memory_monitor()
    gaussian_proofs = _gaussian_cell_proofs(validated)
    _require(
        len(gaussian_proofs) * 10 == len(matching),
        "shard ended without complete ten-branch cells",
    )
    shard_receipt = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "plan_path": str(plan_path),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": plan["analysis_split"],
        "seed_slot": args.seed_slot,
        "group_id": args.group_id,
        "visible_gpus": matching[0]["visible_gpus"],
        "candidate_count": len(matching),
        "candidate_receipts": receipt_rows,
        "same_cell_gaussian_proofs": gaussian_proofs,
        "independent_full81_review_performed": False,
        "phi_v1_extraction_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    shard_receipt = {
        **shard_receipt, "receipt_digest": object_sha256(shard_receipt)
    }
    _write_create_only(output / "reserve4-generation-shard-receipt-v1.json", shard_receipt)
    return 0


def audit_plan(
    *, plan_path: str | Path, expected_plan_sha256: str,
    generation_roots: Sequence[str | Path], output: str | Path,
    gap_output: str | Path,
) -> dict[str, Any]:
    plan, resolved_plan, plan_sha = load_plan(plan_path, expected_plan_sha256)
    receipts: dict[str, Path] = {}
    for root_value in generation_roots:
        root = _plain_dir(root_value, "generation root")
        for path in root.rglob("pair-v5-t2v-calibration-receipt.json"):
            candidate_id = path.parent.name
            _require(candidate_id not in receipts, f"duplicate generation receipt: {candidate_id}")
            receipts[candidate_id] = path.resolve(strict=True)
    expected_ids = [row["candidate_id"] for row in plan["tasks"]]
    unexpected = sorted(set(receipts) - set(expected_ids))
    _require(not unexpected, f"generation roots contain candidates outside plan: {unexpected}")
    validated: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    receipt_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in plan["tasks"]:
        path = receipts.get(task["candidate_id"])
        if path is None:
            missing.append(task["candidate_id"])
            continue
        receipt, _ = _validate_candidate_receipt(task, path)
        validated.append((task, receipt))
        receipt_rows.append(
            {
                "candidate_id": task["candidate_id"],
                "path": str(path),
                "file_sha256": file_sha256(path),
                "receipt_digest": receipt["receipt_digest"],
            }
        )
    complete_cells = len(validated) // 10 if len(validated) % 10 == 0 else 0
    gap = {
        "schema_version": GAP_SCHEMA,
        "plan_path": str(resolved_plan),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": plan["analysis_split"],
        "expected_candidate_count": len(expected_ids),
        "observed_candidate_count": len(validated),
        "missing_candidate_ids": missing,
        "complete_ten_branch_seed_cells": complete_cells,
        "independent_full81_review_count": 0,
        "phi_v1_extraction_authorized": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    gap = {**gap, "receipt_digest": object_sha256(gap)}
    _write_create_only(Path(gap_output), gap)
    _require(not missing and len(validated) == 40, "reserve4 generation closure is incomplete; gap receipt written")
    gaussian_proofs = _gaussian_cell_proofs(validated)
    _require(len(gaussian_proofs) == 4, "reserve4 generation cell closure differs")
    audit = {
        "schema_version": AUDIT_RECEIPT_SCHEMA,
        "plan_path": str(resolved_plan),
        "plan_file_sha256": plan_sha,
        "plan_digest": plan["plan_digest"],
        "analysis_split": plan["analysis_split"],
        "candidate_count": 40,
        "seed_cell_count": 4,
        "candidate_receipts": receipt_rows,
        "same_cell_gaussian_proofs": gaussian_proofs,
        "generation_complete": True,
        "independent_full81_review_performed": False,
        "visual_review_required_before_phi_v1_extraction": True,
        "phi_v1_extraction_authorized": False,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_authorized": False,
        "generated_media_is_editor_input_or_target": False,
    }
    audit = {**audit, "receipt_digest": object_sha256(audit)}
    _write_create_only(Path(output), audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    monitor = commands.add_parser("host-memory-monitor")
    monitor.add_argument("--sample-journal", required=True)
    monitor.add_argument("--start-receipt-output", required=True)
    monitor.add_argument("--stop-fd", type=int, required=True)
    monitor.add_argument("--supervisor-pid", type=int, required=True)
    monitor.add_argument("--slurm-job-id", required=True)
    monitor.add_argument("--slurm-step-id", required=True)
    commands.add_parser("assert-host-memory-monitor-live")
    terminal = commands.add_parser("seal-terminal-host-memory-gate")
    terminal.add_argument("--output", required=True)
    terminal.add_argument("--monitor-exit-status", type=int, required=True)
    plan = commands.add_parser("build-plan")
    plan.add_argument("--seed1-spec", required=True)
    plan.add_argument("--seed2-spec", required=True)
    plan.add_argument("--split", choices=bank_contract.ANALYSIS_SPLITS, required=True)
    plan.add_argument("--output-dir", required=True)

    def add_runtime_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--plan", required=True)
        command.add_argument("--expected-plan-sha256", required=True)
        command.add_argument("--python", required=True)
        command.add_argument("--bernini-root", required=True)
        command.add_argument("--veomni-root", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--checkpoint-content-manifest", required=True)
        command.add_argument("--method-source-revision", required=True)
        command.add_argument("--method-source-archive-sha256", required=True)
        command.add_argument("--master-port", type=int, required=True)

    smoke = commands.add_parser("smoke-sp4")
    add_runtime_args(smoke)
    smoke.add_argument("--receipt-output", required=True)
    smoke.add_argument("--host-memory-gate-output", required=True)
    smoke.add_argument("--r10-compile-smoke-receipt", required=True)
    smoke.add_argument(
        "--expected-r10-compile-smoke-receipt-sha256", required=True
    )
    smoke.add_argument("--r10-generation-log", required=True)
    smoke.add_argument("--expected-r10-generation-log-sha256", required=True)
    run = commands.add_parser("run-sp4")
    add_runtime_args(run)
    run.add_argument("--seed-slot", choices=("seed1", "seed2"), required=True)
    run.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    run.add_argument("--compile-smoke-receipt", required=True)
    run.add_argument("--expected-compile-smoke-receipt-sha256", required=True)
    run.add_argument("--output-dir", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--plan", required=True)
    audit.add_argument("--expected-plan-sha256", required=True)
    audit.add_argument("--generation-root", action="append", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--gap-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "host-memory-monitor":
        return run_host_cgroup_memory_monitor(
            sample_journal=Path(args.sample_journal),
            start_receipt_output=Path(args.start_receipt_output),
            stop_fd=args.stop_fd,
            supervisor_pid=args.supervisor_pid,
            slurm_job_id=args.slurm_job_id,
            slurm_step_id=args.slurm_step_id,
        )
    if args.command == "seal-terminal-host-memory-gate":
        value, observed = write_host_cgroup_memory_gate_receipt(
            Path(args.output),
            measurement_phase="terminal_after_formal40_before_slurm_child_exit",
            formal_candidate_count_at_gate=40,
            monitor_exit_status=args.monitor_exit_status,
        )
        print(
            canonical_json_bytes(
                {
                    "path": str(Path(args.output)),
                    "file_sha256": observed,
                    "receipt_digest": value["receipt_digest"],
                }
            ).decode("ascii"),
            flush=True,
        )
        return 0
    if args.command == "assert-host-memory-monitor-live":
        value = assert_live_host_cgroup_memory_monitor()
        print(canonical_json_bytes(value).decode("ascii"), flush=True)
        return 0
    if args.command == "build-plan":
        value = build_plan(
            seed1_spec=args.seed1_spec,
            seed2_spec=args.seed2_spec,
            split=args.split,
            output_dir=args.output_dir,
        )
        print(
            canonical_json_bytes(
                {
                    "plan_path": value["_path"],
                    "plan_file_sha256": value["_file_sha256"],
                    "candidate_count": value["generation_invocation_count"],
                    "shards": value["shards"],
                }
            ).decode("ascii"),
            flush=True,
        )
        return 0
    if args.command == "run-sp4":
        return run_sp4(args)
    if args.command == "smoke-sp4":
        return run_compile_smoke_sp4(args)
    value = audit_plan(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        generation_roots=args.generation_root,
        output=args.output,
        gap_output=args.gap_output,
    )
    print(canonical_json_bytes(value).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
