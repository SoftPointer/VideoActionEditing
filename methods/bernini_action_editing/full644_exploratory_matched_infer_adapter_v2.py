#!/usr/bin/env python3
"""Eval-only named-output adapter for retained-FD ``infer_lora`` execution.

The frozen ``infer_lora.py`` correctly requires production media and receipt
publication through an inherited ``/proc/self/fd/<task-root-fd>`` directory.
The matched-evaluation plan, on the other hand, deliberately records durable
named output paths.  This adapter binds those two views without weakening
either contract:

* ``--output`` remains the durable named path recorded in the native receipt;
* the task publication-root FD must name that exact parent directory;
* only the two publication calls for that exact MP4/receipt pair are translated
  to the inherited proc-FD view;
* the original functions are restored on every exit path; and
* the named and proc-FD views are replayed as the same single-link inodes.

This module never creates model or adapter authority.  A frozen external
runner must establish the inherited-FD binding and model-consumption input
before invoking it on every torchrun rank.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Iterator, Mapping, Sequence

PUBLICATION_HANDOFF_ENV = "FULL644_MATCHED_PUBLICATION_HANDOFF_AUTHORITY"
PUBLICATION_HANDOFF_AUTHORITY_SCHEMA = (
    "full644-exploratory-matched-publication-handoff-authority-v1"
)
PUBLICATION_HANDOFF_PAYLOAD_SCHEMA = (
    "full644-exploratory-matched-publication-handoff-payload-v1"
)

INFER_LORA_SHA256 = (
    "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
)
MODEL_AUTHORITY_SHA256 = (
    "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf"
)
TRAIN_LORA_SHA256 = (
    "ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85"
)
SELF_GENERATED_PRESERVATION_SHA256 = (
    "11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c"
)
MATERIALIZE_VAE_SHA256 = (
    "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
)
BUILD_RENDERER_DATASET_SHA256 = (
    "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
)
SITE_PACKAGES_ENV = "FULL644_MATCHED_SITE_PACKAGES_ROOT"
FFMPEG_AUTHORITY_ENV = "FULL644_MATCHED_FFMPEG_EXEC_AUTHORITY"
FFMPEG_AUTHORITY_SCHEMA = (
    "full644-exploratory-matched-ffmpeg-exec-authority-v1"
)
_EXEC_IDENTITY_FIELDS = {
    "device",
    "inode",
    "uid",
    "gid",
    "mode",
    "nlink",
    "rdev",
    "size",
    "blocks",
    "mtime_ns",
    "ctime_ns",
}


class MatchedInferAdapterError(RuntimeError):
    """The named/proc-FD publication binding differs."""


def _read_exact_source(
    path_value: str | Path, expected_sha256: str, *, label: str
) -> tuple[Path, str]:
    path = Path(path_value).expanduser().resolve(strict=True)
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_sha256)
    ):
        raise MatchedInferAdapterError(f"{label} source SHA differs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or digest.hexdigest() != expected_sha256
        or identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or identity
        != (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_uid,
            named.st_gid,
            named.st_nlink,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        )
    ):
        raise MatchedInferAdapterError(f"{label} source identity differs")
    try:
        source = b"".join(chunks).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise MatchedInferAdapterError(f"{label} source is not UTF-8") from error
    return path, source


def _load_exact_source_module(
    name: str,
    path_value: str | Path,
    expected_sha256: str,
    *,
    require_absent: bool,
) -> Any:
    """Execute one pinned local module from source, ignoring every ``.pyc``."""

    path, source = _read_exact_source(path_value, expected_sha256, label=name)
    existing = sys.modules.get(name)
    if existing is not None:
        if require_absent:
            raise MatchedInferAdapterError(
                f"{name} was imported before source-only bootstrap"
            )
        origin = getattr(existing, "__file__", None)
        if origin is None or Path(origin).resolve(strict=True) != path:
            raise MatchedInferAdapterError(f"{name} existing origin differs")
        return existing
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, origin=str(path)
    )
    sys.modules[name] = module
    try:
        exec(
            compile(source, str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if module.__cached__ is not None or module.__file__ != str(path):
        raise MatchedInferAdapterError(f"{name} executed-source origin differs")
    return module


_METHOD_ROOT = Path(__file__).resolve(strict=True).parent
_MODEL_AUTHORITY_PATH = (
    _METHOD_ROOT / "action_preservation_decoded_eval_model_authority_v2.py"
)
model_authority = _load_exact_source_module(
    "action_preservation_decoded_eval_model_authority_v2",
    _MODEL_AUTHORITY_PATH,
    MODEL_AUTHORITY_SHA256,
    require_absent=(__name__ == "__main__"),
)


@dataclass(frozen=True)
class PublicationPaths:
    logical_output: Path
    logical_receipt: Path
    runtime_output: Path
    runtime_receipt: Path
    task_fd: int
    task_root: Path


@dataclass
class _PatchCalls:
    encoded_output: int = 0
    receipt: int = 0
    output_fd: int | None = None
    receipt_fd: int | None = None
    encoded_result: dict[str, Any] | None = None
    receipt_value: dict[str, Any] | None = None
    receipt_raw: bytes | None = None


_PATCH_ACTIVE = False
_FFMPEG_PATCH_ACTIVE = False


def load_bootstrap_sealed_authority_fds() -> dict[str, Any]:
    """Replay the descriptor table already sealed by the ``-c`` bootstrap.

    The isolated rank bootstrap validates exact numbers/dev/inode/mode/owner
    while the sanctioned descriptors are inheritable, restores CLOEXEC, and
    only then compiles this captured adapter source.
    """

    try:
        inherited = model_authority.load_inherited_fd_environment(
            verify_open_fds=True,
            expected_inheritable=False,
        )
        model_authority.validate_inherited_fd_binding(
            inherited,
            verify_open_fds=True,
            expected_inheritable=False,
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise MatchedInferAdapterError(str(error)) from error
    return inherited


def _exec_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "nlink": info.st_nlink,
        "rdev": info.st_rdev,
        "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _pread_exact(
    descriptor: int, size: int, *, allow_empty: bool = False
) -> bytes:
    if (
        type(size) is not int
        or type(allow_empty) is not bool
        or size < 0
        or (size == 0 and not allow_empty)
        or not hasattr(os, "pread")
    ):
        raise MatchedInferAdapterError("retained ffmpeg pread is unavailable")
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise MatchedInferAdapterError("retained ffmpeg read is incomplete")
    return raw


def _strict_pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise MatchedInferAdapterError(
                "retained ffmpeg authority has a duplicate key"
            )
        value[key] = item
    return value


def load_retained_ffmpeg_authority() -> dict[str, Any]:
    raw = os.environ.get(FFMPEG_AUTHORITY_ENV)
    if raw is None:
        raise MatchedInferAdapterError("retained ffmpeg authority is absent")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise MatchedInferAdapterError(
            "retained ffmpeg authority is not strict JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or model_authority.canonical_json_bytes(value).decode("utf-8") != raw
        or set(value) != {"schema_version", "row", "authority_digest"}
        or value.get("schema_version") != FFMPEG_AUTHORITY_SCHEMA
    ):
        raise MatchedInferAdapterError("retained ffmpeg authority differs")
    unsigned = dict(value)
    claimed = unsigned.pop("authority_digest", None)
    row = value.get("row")
    if (
        claimed != model_authority.object_sha256(unsigned)
        or not isinstance(row, Mapping)
        or set(row) != {"role", "fd", "source_path", "sha256", "identity"}
        or row.get("role") != "ffmpeg_executable"
        or type(row.get("fd")) is not int
        or row["fd"] < 3
        or type(row.get("source_path")) is not str
        or not Path(row["source_path"]).is_absolute()
        or os.path.normpath(row["source_path"]) != row["source_path"]
        or Path(row["source_path"]).is_symlink()
        or Path(row["source_path"]).resolve(strict=True)
        != Path(row["source_path"])
        or type(row.get("sha256")) is not str
        or len(row["sha256"]) != 64
        or any(item not in "0123456789abcdef" for item in row["sha256"])
        or not isinstance(row.get("identity"), Mapping)
        or set(row["identity"]) != _EXEC_IDENTITY_FIELDS
        or any(type(item) is not int for item in row["identity"].values())
    ):
        raise MatchedInferAdapterError("retained ffmpeg authority row differs")
    descriptor = row["fd"]
    path = Path(row["source_path"])
    try:
        before = os.fstat(descriptor)
        payload = _pread_exact(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        named = path.lstat()
        after = os.fstat(descriptor)
        inheritable = os.get_inheritable(descriptor)
    except OSError as error:
        raise MatchedInferAdapterError(
            "retained ffmpeg descriptor is unavailable"
        ) from error
    expected = dict(row["identity"])
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not before.st_mode & 0o111
        or _exec_identity(before) != expected
        or _exec_identity(middle) != expected
        or _exec_identity(named) != expected
        or _exec_identity(after) != expected
        or hashlib.sha256(payload).hexdigest() != row["sha256"]
        or inheritable
    ):
        raise MatchedInferAdapterError("retained ffmpeg FD replay differs")
    return {
        "schema_version": value["schema_version"],
        "row": {**dict(row), "identity": expected},
        "authority_digest": claimed,
    }


def load_empty_publication_handoff_authority() -> dict[str, Any]:
    raw = os.environ.get(PUBLICATION_HANDOFF_ENV)
    if raw is None:
        raise MatchedInferAdapterError("publication handoff authority is absent")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise MatchedInferAdapterError(
            "publication handoff authority is not strict JSON"
        ) from error
    unsigned = dict(value) if isinstance(value, dict) else {}
    claimed = unsigned.pop("authority_digest", None)
    identity = value.get("initial_identity") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or model_authority.canonical_json_bytes(value).decode("utf-8") != raw
        or set(value)
        != {
            "schema_version",
            "task_id",
            "fd",
            "initial_identity",
            "capacity",
            "authority_digest",
        }
        or value.get("schema_version") != PUBLICATION_HANDOFF_AUTHORITY_SCHEMA
        or claimed != model_authority.object_sha256(unsigned)
        or type(value.get("task_id")) is not str
        or not value["task_id"]
        or type(value.get("fd")) is not int
        or value["fd"] < 3
        or value.get("capacity") != 65536
        or not isinstance(identity, Mapping)
        or set(identity) != _EXEC_IDENTITY_FIELDS
        or any(type(item) is not int for item in identity.values())
    ):
        raise MatchedInferAdapterError("publication handoff authority differs")
    try:
        observed = os.fstat(value["fd"])
        inheritable = os.get_inheritable(value["fd"])
        seals = fcntl.fcntl(value["fd"], fcntl.F_GET_SEALS)
    except (OSError, AttributeError) as error:
        raise MatchedInferAdapterError(
            "publication handoff descriptor is unavailable"
        ) from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 0
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size != 0
        or _exec_identity(observed) != dict(identity)
        or inheritable
        or seals != 0
    ):
        raise MatchedInferAdapterError("empty publication handoff replay differs")
    return {**value, "initial_identity": dict(identity)}


def distributed_publication_contract() -> tuple[int, int]:
    """Return ``(rank, expected_local_publication_calls)`` for exact world4."""

    values: dict[str, int] = {}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        raw = os.environ.get(key)
        try:
            value = int(raw) if raw is not None else -1
        except ValueError as error:
            raise MatchedInferAdapterError(f"{key} differs") from error
        if raw is None or str(value) != raw:
            raise MatchedInferAdapterError(f"{key} differs")
        values[key] = value
    if (
        values["WORLD_SIZE"] != 4
        or values["LOCAL_WORLD_SIZE"] != 4
        or values["RANK"] not in range(4)
        or values["LOCAL_RANK"] != values["RANK"]
    ):
        raise MatchedInferAdapterError("rank publication world differs")
    rank = values["RANK"]
    return rank, (1 if rank == 0 else 0)


def configure_rank_cache() -> Path:
    base_value = os.environ.get("FULL644_MATCHED_RANK_CACHE_ROOT")
    rank_value = os.environ.get("LOCAL_RANK")
    if base_value is None or rank_value is None:
        raise MatchedInferAdapterError("rank-cache authority is absent")
    try:
        rank = int(rank_value)
    except ValueError as error:
        raise MatchedInferAdapterError("LOCAL_RANK differs") from error
    base = Path(base_value)
    if (
        rank not in range(4)
        or not base.is_absolute()
        or os.path.normpath(str(base)) != str(base)
        or not base.is_dir()
        or base.is_symlink()
    ):
        raise MatchedInferAdapterError("rank-cache root differs")
    root = base / f"rank-{rank}"
    try:
        root.mkdir(mode=0o700)
        for name in (
            "miopen-user",
            "miopen-custom",
            "xdg",
            "tmp",
            "triton",
            "inductor",
            "extensions",
            "pycache",
            "home",
            "hf",
            "torch",
        ):
            (root / name).mkdir(mode=0o700)
    except FileExistsError as error:
        raise MatchedInferAdapterError("rank-cache output is not fresh") from error
    environment = {
        "MIOPEN_USER_DB_PATH": root / "miopen-user",
        "MIOPEN_CUSTOM_CACHE_DIR": root / "miopen-custom",
        "XDG_CACHE_HOME": root / "xdg",
        "TMPDIR": root / "tmp",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
        "TRITON_CACHE_DIR": root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": root / "inductor",
        "TORCH_EXTENSIONS_DIR": root / "extensions",
        "PYTHONPYCACHEPREFIX": root / "pycache",
        "HOME": root / "home",
        "HF_HOME": root / "hf",
        "TORCH_HOME": root / "torch",
    }
    for key, value in environment.items():
        os.environ[key] = str(value)
    return root


def require_isolated_startup() -> None:
    if (
        sys.flags.no_site != 1
        or sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise MatchedInferAdapterError(
            "rank adapter requires -I -S -B before authority FD entry"
        )


def activate_fresh_rank_pycache(rank_root: Path) -> Path:
    cache = rank_root / "pycache"
    if not cache.is_dir() or cache.is_symlink() or any(cache.iterdir()):
        raise MatchedInferAdapterError("rank bytecode cache is not fresh")
    sys.pycache_prefix = str(cache)
    if sys.pycache_prefix != str(cache):
        raise MatchedInferAdapterError("rank bytecode cache activation differs")
    return cache


def activate_pinned_site_packages() -> Path:
    raw = os.environ.get(SITE_PACKAGES_ENV)
    if raw is None:
        raise MatchedInferAdapterError("site-packages authority is absent")
    root = Path(raw)
    if (
        not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or root.resolve(strict=True) != root
        or root.is_symlink()
        or not root.is_dir()
        or any(
            name == "torch" or name.startswith("torch.")
            for name in sys.modules
        )
    ):
        raise MatchedInferAdapterError("site-packages authority differs")
    if any(
        value and Path(value).resolve(strict=False) == root for value in sys.path
    ):
        raise MatchedInferAdapterError(
            "site-packages was active before rank FD sealing"
        )
    sys.path.append(str(root))
    if sys.path[-1] != str(root):
        raise MatchedInferAdapterError("site-packages activation differs")
    return root


def _method_root_path_entries() -> tuple[str, ...]:
    observed: list[str] = []
    for value in sys.path:
        if type(value) is not str:
            raise MatchedInferAdapterError("Python import path entry differs")
        try:
            resolved = Path(value or os.curdir).resolve(strict=False)
        except OSError as error:
            raise MatchedInferAdapterError(
                "Python import path entry is unavailable"
            ) from error
        if resolved == _METHOD_ROOT:
            observed.append(value)
    return tuple(observed)


def _validate_closed_local_import_paths() -> None:
    tools_package = sys.modules.get("tools")
    tools_spec = getattr(tools_package, "__spec__", None)
    if (
        _method_root_path_entries()
        or tools_package is None
        or tuple(getattr(tools_package, "__path__", ("missing",))) != ()
        or tools_spec is None
        or tuple(
            getattr(tools_spec, "submodule_search_locations", ("missing",))
        )
        != ()
    ):
        raise MatchedInferAdapterError("local lazy-import search path escaped")


def _create_tools_namespace(*, require_absent: bool) -> Any:
    existing = sys.modules.get("tools")
    if existing is not None:
        if require_absent:
            raise MatchedInferAdapterError(
                "tools was imported before source-only bootstrap"
            )
        return existing
    package = types.ModuleType("tools")
    package.__file__ = None
    package.__package__ = "tools"
    package.__loader__ = None
    package.__cached__ = None
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec(
        name="tools", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = []
    sys.modules["tools"] = package
    return package


def load_frozen_inference_sources(*, require_absent: bool) -> Any:
    """Preload every eval-reachable local module from pinned source bytes."""

    path_snapshot = list(sys.path)
    if require_absent and _method_root_path_entries():
        raise MatchedInferAdapterError(
            "method root was importable before source-only bootstrap"
        )
    try:
        trainer = _load_exact_source_module(
            "train_lora",
            _METHOD_ROOT / "train_lora.py",
            TRAIN_LORA_SHA256,
            require_absent=require_absent,
        )
        tools_package = _create_tools_namespace(require_absent=require_absent)
        raw_builder = _load_exact_source_module(
            "tools.build_renderer_dataset",
            _METHOD_ROOT / "tools/build_renderer_dataset.py",
            BUILD_RENDERER_DATASET_SHA256,
            require_absent=require_absent,
        )
        setattr(tools_package, "build_renderer_dataset", raw_builder)
        materializer = _load_exact_source_module(
            "tools.materialize_vae",
            _METHOD_ROOT / "tools/materialize_vae.py",
            MATERIALIZE_VAE_SHA256,
            require_absent=require_absent,
        )
        setattr(tools_package, "materialize_vae", materializer)
        _load_exact_source_module(
            "self_generated_action_preservation_v2",
            _METHOD_ROOT / "self_generated_action_preservation_v2.py",
            SELF_GENERATED_PRESERVATION_SHA256,
            require_absent=require_absent,
        )
        module = _load_exact_source_module(
            "infer_lora",
            _METHOD_ROOT / "infer_lora.py",
            INFER_LORA_SHA256,
            require_absent=require_absent,
        )
        if (
            module.__cached__ is not None
            or trainer.__cached__ is not None
            or Path(module.__file__).resolve(strict=True)
            != _METHOD_ROOT / "infer_lora.py"
        ):
            raise MatchedInferAdapterError(
                "frozen source-only module origin differs"
            )
        if require_absent and sys.path != [str(_METHOD_ROOT), *path_snapshot]:
            raise MatchedInferAdapterError(
                "frozen source import path mutation differs"
            )
    finally:
        if require_absent:
            sys.path[:] = path_snapshot
    if require_absent:
        _validate_closed_local_import_paths()
    return module


# On the executable rank path, replay/seal the inherited descriptor table and
# isolate the rank caches before importing the much larger frozen inference
# module.  Ordinary imports (including tests) remain side-effect free.
_EARLY_INBOUND_BINDING: dict[str, Any] | None = None
_EARLY_RANK_CACHE: Path | None = None
_EARLY_PYCACHE: Path | None = None
_EARLY_SITE_PACKAGES: Path | None = None
_EARLY_FFMPEG_AUTHORITY: dict[str, Any] | None = None
_EARLY_PUBLICATION_HANDOFF: dict[str, Any] | None = None
if __name__ == "__main__":
    require_isolated_startup()
    if "FULL644_MATCHED_PYTHON_EXECUTABLE_BINDING" in os.environ:
        raise MatchedInferAdapterError(
            "rank captured-source bootstrap did not consume its code FDs"
        )
    _EARLY_INBOUND_BINDING = load_bootstrap_sealed_authority_fds()
    _EARLY_FFMPEG_AUTHORITY = load_retained_ffmpeg_authority()
    _EARLY_PUBLICATION_HANDOFF = load_empty_publication_handoff_authority()
    if (
        _EARLY_PUBLICATION_HANDOFF["task_id"]
        != _EARLY_INBOUND_BINDING.get("task_id")
    ):
        raise MatchedInferAdapterError("publication handoff task differs")
    _EARLY_RANK_CACHE = configure_rank_cache()
    _EARLY_PYCACHE = activate_fresh_rank_pycache(_EARLY_RANK_CACHE)
    _EARLY_SITE_PACKAGES = activate_pinned_site_packages()

infer_lora = load_frozen_inference_sources(require_absent=(__name__ == "__main__"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mode != after.st_mode
        or before.st_nlink != after.st_nlink
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise MatchedInferAdapterError(f"source file changed while hashing: {path}")
    return digest.hexdigest()


def validate_frozen_origins() -> dict[str, str]:
    _validate_closed_local_import_paths()
    observed = {
        "infer_lora": _file_sha256(Path(infer_lora.__file__).resolve(strict=True)),
        "model_authority": _file_sha256(
            Path(model_authority.__file__).resolve(strict=True)
        ),
        "train_lora": _file_sha256(_METHOD_ROOT / "train_lora.py"),
        "self_generated_action_preservation_v2": _file_sha256(
            _METHOD_ROOT / "self_generated_action_preservation_v2.py"
        ),
        "tools.materialize_vae": _file_sha256(
            _METHOD_ROOT / "tools/materialize_vae.py"
        ),
        "tools.build_renderer_dataset": _file_sha256(
            _METHOD_ROOT / "tools/build_renderer_dataset.py"
        ),
    }
    expected = {
        "infer_lora": INFER_LORA_SHA256,
        "model_authority": MODEL_AUTHORITY_SHA256,
        "train_lora": TRAIN_LORA_SHA256,
        "self_generated_action_preservation_v2": (
            SELF_GENERATED_PRESERVATION_SHA256
        ),
        "tools.materialize_vae": MATERIALIZE_VAE_SHA256,
        "tools.build_renderer_dataset": BUILD_RENDERER_DATASET_SHA256,
    }
    if observed != expected:
        raise MatchedInferAdapterError("frozen inference origins differ")
    if (
        any(
            getattr(sys.modules.get(name), "__cached__", None) is not None
            for name in (
                "infer_lora",
                "train_lora",
                "self_generated_action_preservation_v2",
                "tools.materialize_vae",
                "tools.build_renderer_dataset",
                "action_preservation_decoded_eval_model_authority_v2",
            )
        )
    ):
        raise MatchedInferAdapterError("frozen inference used a bytecode origin")
    return observed


def _canonical_dependency_root(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=True) != path
    ):
        raise MatchedInferAdapterError(f"{label} dependency root differs")
    return path


def _module_origins_under(name: str, root: Path) -> list[str]:
    module = sys.modules.get(name)
    if module is None:
        raise MatchedInferAdapterError(f"required dependency was not imported: {name}")
    candidates: list[str] = []
    origin = getattr(module, "__file__", None)
    if isinstance(origin, str):
        candidates.append(origin)
    package_paths = getattr(module, "__path__", ())
    if package_paths is not None:
        candidates.extend(value for value in package_paths if isinstance(value, str))
    resolved: list[str] = []
    for value in candidates:
        try:
            path = Path(value).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise MatchedInferAdapterError(
                f"dependency origin is unavailable: {name}"
            ) from error
        if path != root and root not in path.parents:
            raise MatchedInferAdapterError(
                f"dependency escaped its pinned root: {name}"
            )
        resolved.append(str(path))
    if not resolved:
        raise MatchedInferAdapterError(f"dependency has no filesystem origin: {name}")
    return sorted(set(resolved))


_OS_GIT = Path("/usr/bin/git")
_GIT_ENV = {
    "GIT_ALLOW_PROTOCOL": "file",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
_BERNINI_TREE_SCOPES = (
    "bernini",
    "configs/bernini_renderer_wan21_1p3b",
)
_BERNINI_CONFIG_RELATIVE = (
    "configs/bernini_renderer_wan21_1p3b/config.json"
)
_VEOMNI_TREE_SCOPES = ("veomni",)
_VENDOR_IMPORT_PREFIXES = ("bernini", "veomni")


@dataclass(frozen=True)
class _CapturedVendorTree:
    label: str
    live_root: Path
    expected_commit: str
    scopes: tuple[str, ...]
    directories: tuple[str, ...]
    file_modes: Mapping[str, int]
    file_git_blobs: Mapping[str, str]
    file_sha256: Mapping[str, str]
    file_bytes: Mapping[str, bytes]
    closure_digest: str


def _stable_plain_file_bytes_at(
    parent_fd: int, name: str, *, label: str
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise MatchedInferAdapterError(f"{label} cannot be opened") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise MatchedInferAdapterError(f"{label} is not a regular file")
        raw = _pread_exact(descriptor, before.st_size, allow_empty=True)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(descriptor)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_rdev",
        "st_size",
        "st_blocks",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    identity = tuple(getattr(before, field) for field in fields)
    if (
        len(raw) != before.st_size
        or identity != tuple(getattr(after, field) for field in fields)
        or identity != tuple(getattr(named, field) for field in fields)
    ):
        raise MatchedInferAdapterError(f"{label} changed while captured")
    return raw, before


def _open_directory_at(parent_fd: int, name: str, *, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise MatchedInferAdapterError(f"{label} directory cannot be opened") from error
    info = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)
        != (named.st_dev, named.st_ino, named.st_mode, named.st_uid, named.st_gid)
    ):
        os.close(descriptor)
        raise MatchedInferAdapterError(f"{label} directory identity differs")
    return descriptor


def _open_relative_directory(root_fd: int, relative: str, *, label: str) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in Path(relative).parts:
            child = _open_directory_at(descriptor, part, label=label)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _entry_exists_at(root_fd: int, relative: str, *, label: str) -> bool:
    parts = Path(relative).parts
    if not parts:
        raise MatchedInferAdapterError(f"{label} relative entry differs")
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            try:
                child = _open_directory_at(parent, part, label=label)
            except MatchedInferAdapterError as error:
                try:
                    os.stat(part, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    return False
                raise error
            os.close(parent)
            parent = child
        try:
            os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(parent)


def _git_authority_from_open_fd(descriptor: int) -> dict[str, Any]:
    before = os.fstat(descriptor)
    raw = _pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    named = _OS_GIT.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_uid != 0
        or stat.S_IMODE(before.st_mode) & 0o022
        or not stat.S_IMODE(before.st_mode) & 0o111
        or _exec_identity(before) != _exec_identity(after)
        or _exec_identity(before) != _exec_identity(named)
    ):
        raise MatchedInferAdapterError("OS Git trust-root identity differs")
    row = {
        "path": str(_OS_GIT),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "identity": _exec_identity(before),
        "root_owned_non_group_other_writable": True,
    }
    row["authority_digest"] = model_authority.object_sha256(row)
    return row


def _open_git_executable_authority() -> tuple[int, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_OS_GIT, flags)
    except OSError as error:
        raise MatchedInferAdapterError("OS Git trust root is unavailable") from error
    try:
        authority = _git_authority_from_open_fd(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, authority


def _git_executable_authority() -> dict[str, Any]:
    descriptor, authority = _open_git_executable_authority()
    os.close(descriptor)
    return authority


def _run_git(
    root_fd: int,
    arguments: Sequence[str],
    *,
    label: str,
    root_path: Path | None = None,
) -> bytes:
    git_fd, git_authority = _open_git_executable_authority()
    proc_root = Path(f"/proc/self/fd/{root_fd}")
    proc_git = Path(f"/proc/self/fd/{git_fd}")
    if proc_root.exists() and proc_git.exists():
        root_argument = str(proc_root)
        executable = str(proc_git)
        pass_fds = (root_fd, git_fd)
    else:
        # Unit tests on Darwin have no procfs.  The directory identity is
        # replayed by the caller; production ranks are Linux/procfs-only.
        if root_path is None:
            os.close(git_fd)
            raise MatchedInferAdapterError(
                f"{label} named test root is absent without procfs"
            )
        root_argument = str(root_path)
        executable = git_authority["path"]
        pass_fds = (root_fd, git_fd)
    command = [
        git_authority["path"],
        "--no-replace-objects",
        "-C",
        root_argument,
        *arguments,
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                executable=executable,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                pass_fds=pass_fds,
                cwd="/",
                env=dict(_GIT_ENV),
            )
        except OSError as error:
            raise MatchedInferAdapterError(
                f"{label} Git audit could not execute"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace")[:400]
            raise MatchedInferAdapterError(
                f"{label} Git audit failed: {detail}"
            )
        if _git_authority_from_open_fd(git_fd) != git_authority:
            raise MatchedInferAdapterError(
                "retained OS Git trust root changed during audit"
            )
        return completed.stdout
    finally:
        os.close(git_fd)


def _parse_git_tree(raw: bytes, *, label: str) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    records = raw.split(b"\0")
    if not records or records[-1] != b"":
        raise MatchedInferAdapterError(f"{label} Git tree is not NUL terminated")
    for record in records[:-1]:
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode_raw, kind, blob_raw = metadata.split(b" ")
            path = path_raw.decode("utf-8", "strict")
            mode_text = mode_raw.decode("ascii", "strict")
            blob = blob_raw.decode("ascii", "strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise MatchedInferAdapterError(
                f"{label} Git tree record differs"
            ) from error
        parts = Path(path).parts
        if (
            kind != b"blob"
            or mode_text not in {"100644", "100755"}
            or len(blob) != 40
            or any(value not in "0123456789abcdef" for value in blob)
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in parts)
            or Path(path).as_posix() != path
            or path in result
        ):
            raise MatchedInferAdapterError(f"{label} Git tree closure differs")
        result[path] = (int(mode_text[-3:], 8), blob)
    if not result:
        raise MatchedInferAdapterError(f"{label} Git tree is empty")
    return result


def _expected_directories(
    files: Mapping[str, tuple[int, str]], scopes: Sequence[str]
) -> set[str]:
    directories = set(scopes)
    for relative in files:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _capture_git_vendor_tree(
    root_value: str | Path,
    *,
    expected_commit: str,
    scopes: Sequence[str],
    label: str,
) -> _CapturedVendorTree:
    root = _canonical_dependency_root(root_value, label=label)
    if (
        type(expected_commit) is not str
        or len(expected_commit) != 40
        or any(value not in "0123456789abcdef" for value in expected_commit)
        or not scopes
        or len(set(scopes)) != len(scopes)
    ):
        raise MatchedInferAdapterError(f"{label} source authority differs")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = os.open(root, flags)
    try:
        root_before = os.fstat(root_fd)
        git_directory_fd = _open_directory_at(root_fd, ".git", label=f"{label} .git")
        try:
            for forbidden in (
                "info/grafts",
                "objects/info/alternates",
                "refs/replace",
            ):
                if _entry_exists_at(
                    git_directory_fd,
                    forbidden,
                    label=f"{label} Git override metadata",
                ):
                    raise MatchedInferAdapterError(
                        f"{label} Git replacement/graft/alternate metadata is forbidden"
                    )
        finally:
            os.close(git_directory_fd)
        object_kind = _run_git(
            root_fd,
            ["cat-file", "-t", expected_commit],
            label=label,
            root_path=root,
        )
        if object_kind != b"commit\n":
            raise MatchedInferAdapterError(f"{label} pinned object is not a commit")
        tree_raw = _run_git(
            root_fd,
            [
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                expected_commit,
                "--",
                *scopes,
            ],
            label=label,
            root_path=root,
        )
        expected_files = _parse_git_tree(tree_raw, label=label)
        if any(
            not any(
                relative == scope or relative.startswith(scope + "/")
                for scope in scopes
            )
            for relative in expected_files
        ):
            raise MatchedInferAdapterError(f"{label} scoped Git tree escaped")
        expected_dirs = _expected_directories(expected_files, scopes)
        actual_dirs: set[str] = set()
        actual_files: set[str] = set()
        captured: dict[str, bytes] = {}
        file_sha256: dict[str, str] = {}
        file_modes: dict[str, int] = {}
        file_blobs: dict[str, str] = {}

        def scan(directory_fd: int, relative_directory: str) -> None:
            before = os.fstat(directory_fd)
            try:
                names = sorted(os.listdir(directory_fd))
            except OSError as error:
                raise MatchedInferAdapterError(
                    f"{label} directory cannot be enumerated"
                ) from error
            for name in names:
                if name in {"", ".", ".."} or "/" in name or "\0" in name:
                    raise MatchedInferAdapterError(
                        f"{label} source entry name differs"
                    )
                relative = f"{relative_directory}/{name}"
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as error:
                    raise MatchedInferAdapterError(
                        f"{label} source entry disappeared"
                    ) from error
                if stat.S_ISDIR(info.st_mode):
                    actual_dirs.add(relative)
                    child_fd = _open_directory_at(
                        directory_fd, name, label=f"{label} {relative}"
                    )
                    try:
                        scan(child_fd, relative)
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(info.st_mode):
                    actual_files.add(relative)
                    raw, stable = _stable_plain_file_bytes_at(
                        directory_fd, name, label=f"{label} {relative}"
                    )
                    captured[relative] = raw
                    file_sha256[relative] = hashlib.sha256(raw).hexdigest()
                    file_modes[relative] = stat.S_IMODE(stable.st_mode)
                    file_blobs[relative] = hashlib.sha1(
                        f"blob {len(raw)}\0".encode("ascii") + raw
                    ).hexdigest()
                else:
                    raise MatchedInferAdapterError(
                        f"{label} contains a non-regular source entry: {relative}"
                    )
            after = os.fstat(directory_fd)
            if _exec_identity(before) != _exec_identity(after):
                raise MatchedInferAdapterError(
                    f"{label} directory changed while captured"
                )

        for scope in scopes:
            scope_fd = _open_relative_directory(root_fd, scope, label=label)
            parent = Path(scope)
            while parent != Path("."):
                actual_dirs.add(parent.as_posix())
                parent = parent.parent
            try:
                scan(scope_fd, scope)
            finally:
                os.close(scope_fd)
        if actual_files != set(expected_files) or actual_dirs != expected_dirs:
            raise MatchedInferAdapterError(
                f"{label} exact physical tree closure differs: "
                f"extra_files={sorted(actual_files - set(expected_files))[:8]} "
                f"missing_files={sorted(set(expected_files) - actual_files)[:8]} "
                f"extra_dirs={sorted(actual_dirs - expected_dirs)[:8]} "
                f"missing_dirs={sorted(expected_dirs - actual_dirs)[:8]}"
            )
        for relative, (git_mode, git_blob) in expected_files.items():
            if (
                file_blobs.get(relative) != git_blob
                or (file_modes[relative] & 0o111) != (git_mode & 0o111)
            ):
                raise MatchedInferAdapterError(
                    f"{label} working-tree bytes/mode differ from commit: {relative}"
                )
        root_after = os.fstat(root_fd)
        named_after = root.lstat()
        if (
            _exec_identity(root_before) != _exec_identity(root_after)
            or _exec_identity(root_before) != _exec_identity(named_after)
        ):
            raise MatchedInferAdapterError(
                f"{label} root changed while captured"
            )
    finally:
        os.close(root_fd)
    rows = [
        {
            "path": relative,
            "mode": file_modes[relative],
            "git_blob_sha1": file_blobs[relative],
            "sha256": file_sha256[relative],
            "size": len(captured[relative]),
        }
        for relative in sorted(captured)
    ]
    closure = {
        "schema_version": "full644-exploratory-matched-vendor-tree-capture-v1",
        "label": label,
        "expected_commit": expected_commit,
        "scopes": list(scopes),
        "directories": sorted(expected_dirs),
        "files": rows,
        "git_executable_authority": _git_executable_authority(),
        "exact_physical_closure": True,
        "captured_before_import": True,
    }
    return _CapturedVendorTree(
        label=label,
        live_root=root,
        expected_commit=expected_commit,
        scopes=tuple(scopes),
        directories=tuple(sorted(expected_dirs)),
        file_modes=dict(file_modes),
        file_git_blobs=dict(file_blobs),
        file_sha256=dict(file_sha256),
        file_bytes=dict(captured),
        closure_digest=model_authority.object_sha256(closure),
    )


def _materialize_captured_vendor_tree(
    capture: _CapturedVendorTree,
    destination: Path,
) -> Path:
    if destination.exists() or destination.is_symlink():
        raise MatchedInferAdapterError(
            f"{capture.label} captured snapshot path is not fresh"
        )
    try:
        destination.mkdir(mode=0o700)
    except OSError as error:
        raise MatchedInferAdapterError(
            f"{capture.label} captured snapshot root cannot be created"
        ) from error
    root_fd = os.open(
        destination,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for relative in sorted(
            capture.directories,
            key=lambda value: (len(Path(value).parts), value),
        ):
            try:
                os.mkdir(relative, 0o700, dir_fd=root_fd)
            except OSError as error:
                raise MatchedInferAdapterError(
                    f"{capture.label} snapshot directory creation differs: {relative}"
                ) from error
        for relative in sorted(capture.file_bytes):
            raw = capture.file_bytes[relative]
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(relative, flags, 0o400, dir_fd=root_fd)
            except OSError as error:
                raise MatchedInferAdapterError(
                    f"{capture.label} snapshot file creation differs: {relative}"
                ) from error
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(descriptor, raw[offset:])
                    if written <= 0:
                        raise MatchedInferAdapterError(
                            f"{capture.label} snapshot write made no progress"
                        )
                    offset += written
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o400
                    or info.st_nlink != 1
                    or info.st_size != len(raw)
                ):
                    raise MatchedInferAdapterError(
                        f"{capture.label} snapshot file identity differs: {relative}"
                    )
            finally:
                os.close(descriptor)
        for relative in sorted(
            capture.directories,
            key=lambda value: (len(Path(value).parts), value),
            reverse=True,
        ):
            descriptor = _open_relative_directory(
                root_fd, relative, label=f"{capture.label} snapshot"
            )
            try:
                os.fchmod(descriptor, 0o500)
            finally:
                os.close(descriptor)
        os.fchmod(root_fd, 0o500)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    if (
        destination.is_symlink()
        or destination.resolve(strict=True) != destination
        or stat.S_IMODE(destination.lstat().st_mode) != 0o500
    ):
        raise MatchedInferAdapterError(
            f"{capture.label} snapshot root identity differs"
        )
    return destination


def _verify_captured_snapshot(
    capture: _CapturedVendorTree, snapshot_root: Path
) -> dict[str, Any]:
    root = _canonical_dependency_root(snapshot_root, label=f"{capture.label} snapshot")
    root_fd = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    observed_sha: dict[str, str] = {}
    try:
        root_before = os.fstat(root_fd)
        if stat.S_IMODE(root_before.st_mode) != 0o500:
            raise MatchedInferAdapterError(
                f"{capture.label} snapshot root mode differs"
            )

        def scan(directory_fd: int, relative_directory: str) -> None:
            before = os.fstat(directory_fd)
            if stat.S_IMODE(before.st_mode) != 0o500:
                raise MatchedInferAdapterError(
                    f"{capture.label} snapshot directory mode differs"
                )
            for name in sorted(os.listdir(directory_fd)):
                relative = (
                    name
                    if not relative_directory
                    else f"{relative_directory}/{name}"
                )
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    actual_dirs.add(relative)
                    child = _open_directory_at(
                        directory_fd,
                        name,
                        label=f"{capture.label} snapshot {relative}",
                    )
                    try:
                        scan(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    actual_files.add(relative)
                    raw, stable = _stable_plain_file_bytes_at(
                        directory_fd,
                        name,
                        label=f"{capture.label} snapshot {relative}",
                    )
                    if stat.S_IMODE(stable.st_mode) != 0o400 or stable.st_nlink != 1:
                        raise MatchedInferAdapterError(
                            f"{capture.label} snapshot file mode/link differs"
                        )
                    observed_sha[relative] = hashlib.sha256(raw).hexdigest()
                else:
                    raise MatchedInferAdapterError(
                        f"{capture.label} snapshot contains a non-regular entry"
                    )
            if _exec_identity(before) != _exec_identity(os.fstat(directory_fd)):
                raise MatchedInferAdapterError(
                    f"{capture.label} snapshot directory changed"
                )

        scan(root_fd, "")
        root_after = os.fstat(root_fd)
        named_after = root.lstat()
        if (
            _exec_identity(root_before) != _exec_identity(root_after)
            or _exec_identity(root_before) != _exec_identity(named_after)
        ):
            raise MatchedInferAdapterError(
                f"{capture.label} snapshot root changed"
            )
    finally:
        os.close(root_fd)
    if (
        actual_files != set(capture.file_bytes)
        or actual_dirs != set(capture.directories)
        or observed_sha != dict(capture.file_sha256)
    ):
        raise MatchedInferAdapterError(
            f"{capture.label} captured snapshot closure changed"
        )
    authority = {
        "schema_version": "full644-exploratory-matched-vendor-snapshot-v1",
        "label": capture.label,
        "live_root": str(capture.live_root),
        "snapshot_root": str(root),
        "expected_commit": capture.expected_commit,
        "scope": list(capture.scopes),
        "file_count": len(capture.file_bytes),
        "directory_count": len(capture.directories),
        "capture_closure_digest": capture.closure_digest,
        "snapshot_exact_physical_closure": True,
        "snapshot_files_mode_0400": True,
        "snapshot_directories_mode_0500": True,
    }
    authority["authority_digest"] = model_authority.object_sha256(authority)
    return authority


@dataclass(frozen=True)
class _CapturedVendorModule:
    fullname: str
    origin: Path
    raw: bytes
    sha256: str
    is_package: bool
    is_namespace: bool = False

    @property
    def package_directory(self) -> Path:
        if not self.is_package:
            raise MatchedInferAdapterError(
                f"captured vendor module is not a package: {self.fullname}"
            )
        return self.origin if self.is_namespace else self.origin.parent


def _native_pretrained_config_authority() -> tuple[type[Any], Any]:
    """Return the already-preloaded, pinned Transformers config authority."""

    module = sys.modules.get("transformers.configuration_utils")
    config_class = getattr(module, "PreTrainedConfig", None)
    compatibility_alias = getattr(module, "PretrainedConfig", None)
    local_method = (
        None
        if not isinstance(config_class, type)
        else config_class.__dict__.get("from_pretrained")
    )
    if (
        not isinstance(module, types.ModuleType)
        or getattr(module, "__name__", None)
        != "transformers.configuration_utils"
        or not isinstance(config_class, type)
        or config_class.__module__ != "transformers.configuration_utils"
        or config_class.__name__ != "PreTrainedConfig"
        or config_class.__qualname__ != "PreTrainedConfig"
        or compatibility_alias is not config_class
        or not isinstance(local_method, classmethod)
        or not callable(local_method.__func__)
        or getattr(local_method.__func__, "__module__", None)
        != "transformers.configuration_utils"
        or getattr(local_method.__func__, "__qualname__", None)
        != "PreTrainedConfig.from_pretrained"
    ):
        raise MatchedInferAdapterError(
            "native Transformers PretrainedConfig authority differs"
        )
    return config_class, local_method.__func__


class _SealedRendererConfigRedirect:
    """Redirect the one native config load to immutable captured bytes."""

    def __init__(
        self,
        *,
        descriptor: int,
        raw: bytes,
        logical_directory: Path,
        fd_path: str,
        seal_mask: int | None,
        test_only_unsealed_fd: bool,
    ) -> None:
        self.descriptor = descriptor
        self.raw = raw
        self.logical_directory = logical_directory
        self.fd_path = fd_path
        self.seal_mask = seal_mask
        self.test_only_unsealed_fd = test_only_unsealed_fd
        self.initial_identity = _exec_identity(os.fstat(descriptor))
        self.config_class: type[Any] | None = None
        self.original_local_present = False
        self.original_local: Any = None
        self.original_bound: Any = None
        self.redirected_function: Any = None
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._replay()

    def _replay(self) -> dict[str, Any]:
        if self.closed:
            raise MatchedInferAdapterError("renderer config memfd is closed")
        before = os.fstat(self.descriptor)
        observed = _pread_exact(self.descriptor, before.st_size)
        after = os.fstat(self.descriptor)
        if (
            observed != self.raw
            or _exec_identity(before) != self.initial_identity
            or _exec_identity(after) != self.initial_identity
            or os.get_inheritable(self.descriptor)
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
        ):
            raise MatchedInferAdapterError(
                "sealed renderer config FD replay differs"
            )
        if self.seal_mask is not None and (
            fcntl.fcntl(self.descriptor, fcntl.F_GET_SEALS) != self.seal_mask
        ):
            raise MatchedInferAdapterError(
                "sealed renderer config seal set differs"
            )
        return {
            "identity": dict(self.initial_identity),
            "sha256": hashlib.sha256(observed).hexdigest(),
            "size": len(observed),
            "seal_mask": self.seal_mask,
        }

    @staticmethod
    def _validate_kwargs(kwargs: Mapping[str, Any]) -> str:
        expected_keys = {
            "diff_dec_config_path",
            "ema_decay",
            "local_files_only",
            "max_sequence_length",
            "scratch",
            "shift",
            "skip_transformer_1",
            "skip_transformer_2",
            "switch_dit_boundary",
            "use_src_id_rotary_emb",
            "use_unipc",
            "wan22_base",
        }
        if set(kwargs) != expected_keys:
            raise MatchedInferAdapterError(
                "renderer config from_pretrained keyword set differs"
            )
        checkpoint = kwargs.get("wan22_base")
        if (
            type(checkpoint) is not str
            or not Path(checkpoint).is_absolute()
            or os.path.normpath(checkpoint) != checkpoint
            or kwargs.get("diff_dec_config_path") != checkpoint
            or kwargs.get("local_files_only") is not True
            or kwargs.get("skip_transformer_1") is not False
            or kwargs.get("skip_transformer_2") is not True
            or type(kwargs.get("switch_dit_boundary")) is not float
            or kwargs.get("switch_dit_boundary") != 0.0
            or type(kwargs.get("max_sequence_length")) is not int
            or kwargs.get("max_sequence_length") != 512
            or type(kwargs.get("shift")) is not float
            or kwargs.get("shift") != 5.0
            or kwargs.get("use_src_id_rotary_emb") is not True
            or kwargs.get("scratch") is not False
            or kwargs.get("ema_decay") is not None
            or kwargs.get("use_unipc") is not True
        ):
            raise MatchedInferAdapterError(
                "renderer config from_pretrained keyword values differ"
            )
        return checkpoint

    def install(self, module: types.ModuleType) -> None:
        if self.config_class is not None or self.redirected_function is not None:
            raise MatchedInferAdapterError(
                "renderer config redirect was installed more than once"
            )
        config_class = getattr(module, "BerniniRendererConfig", None)
        pretrained_config, native_from_pretrained = (
            _native_pretrained_config_authority()
        )
        if (
            not isinstance(config_class, type)
            or getattr(config_class, "__module__", None) != module.__name__
            or getattr(config_class, "__name__", None)
            != "BerniniRendererConfig"
            or getattr(config_class, "__qualname__", None)
            != "BerniniRendererConfig"
            or config_class.__bases__ != (pretrained_config,)
            or config_class.__mro__
            != (config_class, *pretrained_config.__mro__)
            or "from_pretrained" in config_class.__dict__
        ):
            raise MatchedInferAdapterError(
                "BerniniRendererConfig class origin differs"
            )
        original_bound = getattr(config_class, "from_pretrained", None)
        if (
            not callable(original_bound)
            or getattr(original_bound, "__self__", None) is not config_class
            or getattr(original_bound, "__func__", None)
            is not native_from_pretrained
        ):
            raise MatchedInferAdapterError(
                "BerniniRendererConfig.from_pretrained origin differs"
            )
        original_local_present = "from_pretrained" in config_class.__dict__
        original_local = config_class.__dict__.get("from_pretrained")
        redirect = self

        def redirected(
            inner_class: type[Any],
            pretrained_model_name_or_path: Any,
            *arguments: Any,
            **kwargs: Any,
        ) -> Any:
            if (
                inner_class is not config_class
                or type(pretrained_model_name_or_path) is not str
                or pretrained_model_name_or_path
                != str(redirect.logical_directory)
                or arguments
                or redirect.calls
            ):
                raise MatchedInferAdapterError(
                    "renderer config from_pretrained call differs"
                )
            checkpoint = redirect._validate_kwargs(kwargs)
            before = redirect._replay()
            result = original_bound(redirect.fd_path, **kwargs)
            after = redirect._replay()
            if after != before:
                raise MatchedInferAdapterError(
                    "renderer config memfd changed during native load"
                )
            redirect.calls.append(
                {
                    "logical_config_directory": str(
                        redirect.logical_directory
                    ),
                    "retained_fd_path": redirect.fd_path,
                    "checkpoint": checkpoint,
                    "kwargs_digest": model_authority.object_sha256(dict(kwargs)),
                    "native_class": (
                        "transformers.configuration_utils.PreTrainedConfig"
                    ),
                    "native_function_identity_verified": True,
                    "direct_native_mro_verified": True,
                    "native_bound_classmethod_called_once": True,
                }
            )
            return result

        config_class.from_pretrained = classmethod(redirected)
        installed = config_class.__dict__.get("from_pretrained")
        if not isinstance(installed, classmethod) or installed.__func__ is not redirected:
            if original_local_present:
                setattr(config_class, "from_pretrained", original_local)
            else:
                delattr(config_class, "from_pretrained")
            raise MatchedInferAdapterError(
                "renderer config redirect installation differs"
            )
        self.config_class = config_class
        self.original_local_present = original_local_present
        self.original_local = original_local
        self.original_bound = original_bound
        self.redirected_function = redirected

    def finalize_authority(self) -> dict[str, Any]:
        if (
            self.config_class is None
            or self.redirected_function is None
            or len(self.calls) != 1
            or not isinstance(
                self.config_class.__dict__.get("from_pretrained"), classmethod
            )
            or self.config_class.__dict__["from_pretrained"].__func__
            is not self.redirected_function
        ):
            raise MatchedInferAdapterError(
                "renderer config redirect lifecycle differs"
            )
        replay = self._replay()
        authority = {
            "schema_version": (
                "full644-exploratory-matched-renderer-config-memfd-v1"
            ),
            "logical_config_directory": str(self.logical_directory),
            "config_sha256": replay["sha256"],
            "config_size": replay["size"],
            "config_identity": replay["identity"],
            "seal_mask": replay["seal_mask"],
            "sealed_write_grow_shrink_and_seal": self.seal_mask is not None,
            "native_from_pretrained_call_count": 1,
            "native_from_pretrained_call": dict(self.calls[0]),
            "test_only_unsealed_fd": self.test_only_unsealed_fd,
        }
        authority["authority_digest"] = model_authority.object_sha256(authority)
        return authority

    def restore_and_close(self) -> None:
        if self.closed:
            raise MatchedInferAdapterError(
                "renderer config redirect was closed more than once"
            )
        failure: BaseException | None = None
        config_class = self.config_class
        try:
            if config_class is not None:
                installed = config_class.__dict__.get("from_pretrained")
                if (
                    not isinstance(installed, classmethod)
                    or installed.__func__ is not self.redirected_function
                ):
                    failure = MatchedInferAdapterError(
                        "renderer config redirect changed before restoration"
                    )
                if self.original_local_present:
                    setattr(config_class, "from_pretrained", self.original_local)
                else:
                    try:
                        delattr(config_class, "from_pretrained")
                    except AttributeError as error:
                        if failure is None:
                            failure = error
                restored = getattr(config_class, "from_pretrained", None)
                if (
                    not callable(restored)
                    or getattr(restored, "__func__", restored)
                    is not getattr(self.original_bound, "__func__", self.original_bound)
                    or getattr(restored, "__self__", None)
                    is not getattr(self.original_bound, "__self__", None)
                ) and failure is None:
                    failure = MatchedInferAdapterError(
                        "renderer config classmethod was not restored"
                    )
            self._replay()
        except BaseException as error:
            if failure is None:
                failure = error
        finally:
            os.close(self.descriptor)
            self.closed = True
        if failure is not None:
            raise failure


def _create_sealed_renderer_config_redirect(
    raw: bytes, logical_directory: Path
) -> _SealedRendererConfigRedirect:
    if not raw:
        raise MatchedInferAdapterError("captured renderer config is empty")
    descriptor: int
    seal_mask: int | None
    test_only_unsealed_fd = False
    if hasattr(os, "memfd_create") and Path("/proc/self/fd").is_dir():
        flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(
            os, "MFD_ALLOW_SEALING", 0
        )
        if not flags & getattr(os, "MFD_ALLOW_SEALING", 0):
            raise MatchedInferAdapterError(
                "renderer config memfd sealing is unavailable"
            )
        descriptor = os.memfd_create("full644-renderer-config", flags)
        fd_path = f"/proc/self/fd/{descriptor}"
        seal_mask = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
    else:
        if _EARLY_INBOUND_BINDING is not None or _EARLY_RANK_CACHE is None:
            raise MatchedInferAdapterError(
                "production renderer config requires Linux sealed memfd"
            )
        test_path = _EARLY_RANK_CACHE / "renderer-config-test-fd.json"
        descriptor = os.open(
            test_path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.unlink(test_path)
        fd_path = f"/dev/fd/{descriptor}"
        seal_mask = None
        test_only_unsealed_fd = True
    try:
        offset = 0
        while offset < len(raw):
            written = os.pwrite(descriptor, raw[offset:], offset)
            if written <= 0:
                raise MatchedInferAdapterError(
                    "renderer config memfd write made no progress"
                )
            offset += written
        os.ftruncate(descriptor, len(raw))
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.set_inheritable(descriptor, False)
        if seal_mask is not None:
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seal_mask)
        return _SealedRendererConfigRedirect(
            descriptor=descriptor,
            raw=raw,
            logical_directory=logical_directory,
            fd_path=fd_path,
            seal_mask=seal_mask,
            test_only_unsealed_fd=test_only_unsealed_fd,
        )
    except BaseException:
        os.close(descriptor)
        raise


class _CapturedVendorLoader(importlib.abc.Loader):
    def __init__(
        self,
        row: _CapturedVendorModule,
        resource_bytes: Mapping[str, bytes],
        config_redirect: _SealedRendererConfigRedirect,
    ) -> None:
        self.row = row
        self.resource_bytes = dict(resource_bytes)
        self.config_redirect = config_redirect

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        row = self.row
        module.__file__ = None if row.is_namespace else str(row.origin)
        module.__package__ = row.fullname if row.is_package else row.fullname.rpartition(".")[0]
        module.__cached__ = None
        if row.is_package:
            module.__path__ = [str(row.package_directory)]
        if not row.is_namespace:
            try:
                source = row.raw.decode("utf-8", "strict")
            except UnicodeDecodeError as error:
                raise MatchedInferAdapterError(
                    f"captured vendor source is not UTF-8: {row.fullname}"
                ) from error
            exec(
                compile(source, str(row.origin), "exec", dont_inherit=True),
                module.__dict__,
            )
        if row.fullname == "bernini.models.renderer":
            self.config_redirect.install(module)
        if module.__cached__ is not None:
            raise MatchedInferAdapterError(
                f"captured vendor module used bytecode: {row.fullname}"
            )

    def get_filename(self, fullname: str) -> str:
        if fullname != self.row.fullname:
            raise ImportError(fullname)
        return str(self.row.origin)

    def get_source(self, fullname: str) -> str:
        if fullname != self.row.fullname:
            raise ImportError(fullname)
        return self.row.raw.decode("utf-8", "strict")

    def is_package(self, fullname: str) -> bool:
        if fullname != self.row.fullname:
            raise ImportError(fullname)
        return self.row.is_package

    def get_data(self, path: str) -> bytes:
        if type(path) is not str:
            raise OSError("captured vendor resource path type differs")
        normalized = os.path.normpath(path)
        raw = self.resource_bytes.get(normalized)
        if raw is None:
            raise OSError(
                f"resource is outside the captured vendor closure: {path}"
            )
        return raw


class _CapturedVendorFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        modules: Mapping[str, _CapturedVendorModule],
        resource_bytes: Mapping[str, bytes],
        config_redirect: _SealedRendererConfigRedirect,
    ) -> None:
        self.modules = dict(modules)
        self.resource_bytes = dict(resource_bytes)
        self.config_redirect = config_redirect
        self.loaders: dict[str, _CapturedVendorLoader] = {}
        self.find_calls: list[str] = []

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if not any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in _VENDOR_IMPORT_PREFIXES
        ):
            return None
        if target is not None:
            raise MatchedInferAdapterError("captured vendor reload is forbidden")
        row = self.modules.get(fullname)
        if row is None:
            raise ModuleNotFoundError(
                f"{fullname!r} is outside the captured vendor source closure"
            )
        if fullname in self.loaders:
            raise MatchedInferAdapterError(
                f"captured vendor module was resolved more than once: {fullname}"
            )
        loader = _CapturedVendorLoader(
            row, self.resource_bytes, self.config_redirect
        )
        self.loaders[fullname] = loader
        self.find_calls.append(fullname)
        spec = importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=None if row.is_namespace else str(row.origin),
            is_package=row.is_package,
        )
        spec.cached = None
        if row.is_package:
            spec.submodule_search_locations = [str(row.package_directory)]
        return spec


def _captured_vendor_modules(
    captures: Sequence[tuple[_CapturedVendorTree, Path, str]],
) -> dict[str, _CapturedVendorModule]:
    modules: dict[str, _CapturedVendorModule] = {}
    namespace_origins: dict[str, Path] = {}
    for capture, snapshot, prefix in captures:
        for relative, raw in capture.file_bytes.items():
            if relative.endswith((".pyc", ".pyo", ".so", ".pyd", ".pth")):
                raise MatchedInferAdapterError(
                    f"{capture.label} commit contains a forbidden import node: {relative}"
                )
            path = Path(relative)
            if not path.parts or path.parts[0] != prefix:
                continue
            if path.name == "__init__.py":
                name_parts = path.parts[:-1]
                is_package = True
            elif path.suffix == ".py":
                name_parts = (*path.parts[:-1], path.stem)
                is_package = False
            else:
                continue
            fullname = ".".join(name_parts)
            if not fullname or fullname in modules:
                raise MatchedInferAdapterError(
                    f"{capture.label} module/package mapping is ambiguous: {fullname}"
                )
            modules[fullname] = _CapturedVendorModule(
                fullname=fullname,
                origin=snapshot / relative,
                raw=raw,
                sha256=capture.file_sha256[relative],
                is_package=is_package,
            )
        for relative in capture.directories:
            parts = Path(relative).parts
            if not parts or parts[0] != prefix:
                continue
            for length in range(1, len(parts) + 1):
                fullname = ".".join(parts[:length])
                namespace_origins.setdefault(
                    fullname, snapshot.joinpath(*parts[:length])
                )
    for fullname, origin in namespace_origins.items():
        if fullname not in modules:
            modules[fullname] = _CapturedVendorModule(
                fullname=fullname,
                origin=origin,
                raw=b"",
                sha256=hashlib.sha256(b"").hexdigest(),
                is_package=True,
                is_namespace=True,
            )
    for prefix in _VENDOR_IMPORT_PREFIXES:
        if prefix not in modules:
            raise MatchedInferAdapterError(
                f"captured vendor top-level package is absent: {prefix}"
            )
    return modules


def _preload_pinned_dependencies(site: Path) -> dict[str, Any]:
    """Import third-party roots before vendor activation and close Torch's JIT path."""

    path_before = list(sys.path)
    if (
        path_before.count(str(site)) != 1
        or path_before[-1] != str(site)
        or any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _VENDOR_IMPORT_PREFIXES
            for name in sys.modules
        )
    ):
        raise MatchedInferAdapterError(
            "third-party preload import scope is not fresh"
        )
    import torch
    import diffusers
    import peft
    import transformers
    from transformers.configuration_utils import (  # noqa: F401
        PretrainedConfig,
        PreTrainedConfig,
    )
    from diffusers.models import AutoencoderKLWan  # noqa: F401
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: F401

    _native_pretrained_config_authority()

    if sys.path[: len(path_before)] != path_before or len(sys.path) != len(path_before) + 1:
        raise MatchedInferAdapterError(
            "Torch JIT import path delta differs"
        )
    jit_path_text = sys.path[-1]
    if type(jit_path_text) is not str:
        raise MatchedInferAdapterError("Torch JIT import path type differs")
    jit_path = Path(jit_path_text)
    jit_module = sys.modules.get("torch.distributed.nn.jit.instantiator")
    temp_directory = getattr(jit_module, "_TEMP_DIR", None)
    expected_tmp_parent = Path(os.environ.get("TMPDIR", ""))
    try:
        canonical_jit_path = jit_path.resolve(strict=True)
        canonical_tmp_parent = expected_tmp_parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise MatchedInferAdapterError(
            "Torch JIT temporary directory is unavailable"
        ) from error
    if (
        not canonical_jit_path.is_dir()
        or jit_path.is_symlink()
        or canonical_jit_path.parent != canonical_tmp_parent
        or temp_directory is None
        or getattr(temp_directory, "name", None) != jit_path_text
        or getattr(jit_module, "INSTANTIATED_TEMPLATE_DIR_PATH", None)
        != jit_path_text
        or sys.path.count(jit_path_text) != 1
    ):
        raise MatchedInferAdapterError(
            "Torch JIT temporary-directory authority differs"
        )
    importer_cache_present = jit_path_text in sys.path_importer_cache
    sys.path.pop()
    sys.path_importer_cache.pop(jit_path_text, None)
    if sys.path != path_before or jit_path_text in sys.path:
        raise MatchedInferAdapterError("Torch JIT import path was not removed")
    origins = {
        name: _module_origins_under(name, site)
        for name in ("torch", "diffusers", "peft", "transformers")
    }
    authority = {
        "schema_version": (
            "full644-exploratory-matched-third-party-preload-authority-v1"
        ),
        "site_packages_root": str(site),
        "origins": origins,
        "autoencoder_kl_wan_preloaded": True,
        "wan_prompt_cleaner_preloaded": True,
        "native_pretrained_config_authority_preloaded": True,
        "torch_jit_instantiator_preloaded": True,
        "torch_jit_sys_path_delta_count": 1,
        "torch_jit_path": str(canonical_jit_path),
        "torch_jit_importer_cache_entry_removed": importer_cache_present,
        "sys_path_restored_before_vendor_activation": True,
    }
    authority["authority_digest"] = model_authority.object_sha256(authority)
    return authority


@contextmanager
def pinned_dependency_import_paths(site_root: str | Path) -> Iterator[dict[str, Any]]:
    """Bind vendor imports to captured commit bytes and restore every patch."""

    site = _canonical_dependency_root(site_root, label="site-packages")
    trainer = getattr(infer_lora, "trainer", None)
    original_validate = getattr(trainer, "validate_source_trees", None)
    original_activate = getattr(trainer, "activate_source_trees", None)
    if (
        trainer is None
        or not callable(original_validate)
        or not callable(original_activate)
    ):
        raise MatchedInferAdapterError("source-tree validation/activation origin differs")
    if _EARLY_RANK_CACHE is None:
        raise MatchedInferAdapterError("rank-private vendor snapshot root is absent")
    initial_path = list(sys.path)
    if initial_path.count(str(site)) != 1 or initial_path[-1] != str(site):
        raise MatchedInferAdapterError("pinned site-packages path order differs")
    preload_authority = _preload_pinned_dependencies(site)
    path_snapshot = list(sys.path)
    meta_path_snapshot = list(sys.meta_path)
    importer_cache_snapshot = dict(sys.path_importer_cache)
    if path_snapshot != initial_path:
        raise MatchedInferAdapterError(
            "third-party preload did not restore the import path"
        )
    preexisting_vendor_modules = {
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _VENDOR_IMPORT_PREFIXES
        )
    }
    if preexisting_vendor_modules:
        raise MatchedInferAdapterError(
            "vendor package was imported before captured-source activation"
        )

    validation_calls: list[dict[str, Any]] = []
    activation_calls: list[dict[str, Any]] = []
    active_path: list[str] | None = None
    captures: tuple[_CapturedVendorTree, _CapturedVendorTree] | None = None
    snapshots: tuple[Path, Path] | None = None
    modules: dict[str, _CapturedVendorModule] = {}
    finder: _CapturedVendorFinder | None = None
    config_redirect: _SealedRendererConfigRedirect | None = None
    authority: dict[str, Any] = {}

    def scoped_validate(
        bernini_value: str | Path,
        veomni_value: str | Path,
        *,
        expected_bernini_commit: str,
        expected_veomni_commit: str,
    ) -> tuple[Path, Path, str, str]:
        nonlocal captures, snapshots, modules, finder, config_redirect
        if (
            validation_calls
            or captures is not None
            or snapshots is not None
            or finder is not None
            or config_redirect is not None
            or sys.path != path_snapshot
            or sys.meta_path != meta_path_snapshot
        ):
            raise MatchedInferAdapterError("source-tree validation call differs")
        if (
            expected_bernini_commit
            != getattr(trainer, "BERNINI_OFFICIAL_COMMIT", None)
            or expected_veomni_commit
            != getattr(trainer, "VEOMNI_TESTED_COMMIT", None)
        ):
            raise MatchedInferAdapterError("vendor commit authority differs")
        bernini_capture = _capture_git_vendor_tree(
            bernini_value,
            expected_commit=expected_bernini_commit,
            scopes=_BERNINI_TREE_SCOPES,
            label="Bernini",
        )
        veomni_capture = _capture_git_vendor_tree(
            veomni_value,
            expected_commit=expected_veomni_commit,
            scopes=_VEOMNI_TREE_SCOPES,
            label="VeOmni",
        )
        if len({bernini_capture.live_root, veomni_capture.live_root, site}) != 3:
            raise MatchedInferAdapterError("dependency roots overlap")
        snapshot_parent = _EARLY_RANK_CACHE / "captured-vendor-source-v1"
        try:
            snapshot_parent.mkdir(mode=0o700)
        except OSError as error:
            raise MatchedInferAdapterError(
                "captured vendor snapshot parent is not fresh"
            ) from error
        bernini_snapshot = _materialize_captured_vendor_tree(
            bernini_capture, snapshot_parent / "bernini"
        )
        veomni_snapshot = _materialize_captured_vendor_tree(
            veomni_capture, snapshot_parent / "veomni"
        )
        os.chmod(snapshot_parent, 0o500)
        captures = (bernini_capture, veomni_capture)
        snapshots = (bernini_snapshot, veomni_snapshot)
        config_raw = bernini_capture.file_bytes.get(
            _BERNINI_CONFIG_RELATIVE
        )
        if config_raw is None:
            raise MatchedInferAdapterError(
                "captured Bernini renderer config is absent"
            )
        config_redirect = _create_sealed_renderer_config_redirect(
            config_raw,
            bernini_snapshot
            / "configs/bernini_renderer_wan21_1p3b",
        )
        modules = _captured_vendor_modules(
            (
                (bernini_capture, bernini_snapshot, "bernini"),
                (veomni_capture, veomni_snapshot, "veomni"),
            )
        )
        resource_bytes = {
            os.path.normpath(str(snapshot / relative)): raw
            for capture, snapshot in zip(captures, snapshots)
            for relative, raw in capture.file_bytes.items()
        }
        finder = _CapturedVendorFinder(
            modules, resource_bytes, config_redirect
        )
        validation_calls.append(
            {
                "live_bernini_root": str(bernini_capture.live_root),
                "live_veomni_root": str(veomni_capture.live_root),
                "bernini_snapshot_root": str(bernini_snapshot),
                "veomni_snapshot_root": str(veomni_snapshot),
                "bernini_capture_digest": bernini_capture.closure_digest,
                "veomni_capture_digest": veomni_capture.closure_digest,
            }
        )
        return (
            bernini_snapshot,
            veomni_snapshot,
            expected_bernini_commit,
            expected_veomni_commit,
        )

    def scoped_activate(
        bernini_value: str | Path, veomni_value: str | Path
    ) -> None:
        nonlocal active_path
        if (
            activation_calls
            or captures is None
            or snapshots is None
            or finder is None
            or sys.path != path_snapshot
            or sys.meta_path != meta_path_snapshot
        ):
            raise MatchedInferAdapterError("source-tree activation call differs")
        bernini = _canonical_dependency_root(bernini_value, label="Bernini snapshot")
        veomni = _canonical_dependency_root(veomni_value, label="VeOmni snapshot")
        if (bernini, veomni) != snapshots:
            raise MatchedInferAdapterError(
                "inference did not activate the captured vendor snapshots"
            )
        original_activate(bernini, veomni)
        active_path = [str(bernini), str(veomni), *path_snapshot]
        if sys.path != active_path:
            raise MatchedInferAdapterError(
                "original source-tree activation path delta differs"
            )
        sys.meta_path.insert(0, finder)
        if sys.meta_path != [finder, *meta_path_snapshot]:
            raise MatchedInferAdapterError(
                "captured vendor finder installation differs"
            )
        activation_calls.append(
            {
                "bernini_root": str(bernini),
                "veomni_root": str(veomni),
            }
        )

    trainer.validate_source_trees = scoped_validate
    trainer.activate_source_trees = scoped_activate
    try:
        yield authority
    finally:
        validate_changed = (
            getattr(trainer, "validate_source_trees", None) is not scoped_validate
        )
        activate_changed = (
            getattr(trainer, "activate_source_trees", None) is not scoped_activate
        )
        failure: BaseException | None = None
        try:
            if (
                validate_changed
                or activate_changed
                or len(validation_calls) != 1
                or len(activation_calls) != 1
                or captures is None
                or snapshots is None
                or finder is None
                or config_redirect is None
                or sys.path != active_path
                or sys.meta_path != [finder, *meta_path_snapshot]
            ):
                raise MatchedInferAdapterError(
                    "captured source-tree patch lifecycle differs"
                )
            loaded_vendor = {
                name: module
                for name, module in sys.modules.items()
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in _VENDOR_IMPORT_PREFIXES
                )
            }
            if not {"bernini", "veomni"}.issubset(loaded_vendor):
                raise MatchedInferAdapterError(
                    "required captured vendor packages were not imported"
                )
            loaded_rows: list[dict[str, Any]] = []
            for name, module in sorted(loaded_vendor.items()):
                row = modules.get(name)
                loader = finder.loaders.get(name)
                spec = getattr(module, "__spec__", None)
                if (
                    row is None
                    or loader is None
                    or loader.row is not row
                    or getattr(module, "__loader__", None) is not loader
                    or spec is None
                    or getattr(spec, "loader", None) is not loader
                    or getattr(module, "__cached__", None) is not None
                    or (
                        row.is_package
                        and tuple(getattr(module, "__path__", ()))
                        != (str(row.package_directory),)
                    )
                    or (
                        row.is_package
                        and tuple(
                            getattr(spec, "submodule_search_locations", ())
                        )
                        != (str(row.package_directory),)
                    )
                    or (
                        not row.is_namespace
                        and getattr(module, "__file__", None) != str(row.origin)
                    )
                ):
                    raise MatchedInferAdapterError(
                        f"captured vendor module origin differs: {name}"
                    )
                loaded_rows.append(
                    {
                        "module": name,
                        "origin": None if row.is_namespace else str(row.origin),
                        "sha256": row.sha256,
                        "is_package": row.is_package,
                        "is_namespace": row.is_namespace,
                    }
                )
            snapshot_authority = [
                _verify_captured_snapshot(capture, snapshot)
                for capture, snapshot in zip(captures, snapshots)
            ]
            live_replays = (
                _capture_git_vendor_tree(
                    captures[0].live_root,
                    expected_commit=captures[0].expected_commit,
                    scopes=captures[0].scopes,
                    label=captures[0].label,
                ),
                _capture_git_vendor_tree(
                    captures[1].live_root,
                    expected_commit=captures[1].expected_commit,
                    scopes=captures[1].scopes,
                    label=captures[1].label,
                ),
            )
            if tuple(row.closure_digest for row in live_replays) != tuple(
                row.closure_digest for row in captures
            ):
                raise MatchedInferAdapterError(
                    "live vendor tree changed after captured inference"
                )
            site_origins = {
                name: _module_origins_under(name, site)
                for name in ("torch", "diffusers", "peft", "transformers")
            }
            for name in ("safetensors", "imageio_ffmpeg", "decord"):
                if name in sys.modules:
                    site_origins[name] = _module_origins_under(name, site)
            config_authority = config_redirect.finalize_authority()
            authority.update(
                {
                    "schema_version": (
                        "full644-exploratory-matched-dependency-import-authority-v2"
                    ),
                    "site_packages_root": str(site),
                    **validation_calls[0],
                    **activation_calls[0],
                    "validation_call_count": 1,
                    "activation_call_count": 1,
                    "third_party_preload": preload_authority,
                    "site_package_origins": site_origins,
                    "vendor_snapshot_authorities": snapshot_authority,
                    "renderer_config_memfd_authority": config_authority,
                    "captured_vendor_module_count": len(modules),
                    "loaded_vendor_modules": loaded_rows,
                    "package_specific_finder_first": True,
                    "live_roots_never_used_for_import_or_config": True,
                    "source_snapshot_root_first_for_compatibility": True,
                    "activation_function_restored": True,
                    "validation_function_restored": True,
                    "meta_path_restored": True,
                    "sys_path_restored": True,
                    "path_importer_cache_restored": True,
                }
            )
            authority["authority_digest"] = model_authority.object_sha256(
                authority
            )
        except BaseException as error:
            failure = error
        finally:
            config_cleanup_failure: BaseException | None = None
            if config_redirect is not None:
                try:
                    config_redirect.restore_and_close()
                except BaseException as error:
                    config_cleanup_failure = error
            for name in sorted(
                (
                    name
                    for name in tuple(sys.modules)
                    if any(
                        name == prefix or name.startswith(prefix + ".")
                        for prefix in _VENDOR_IMPORT_PREFIXES
                    )
                    and name not in preexisting_vendor_modules
                ),
                key=lambda value: value.count("."),
                reverse=True,
            ):
                sys.modules.pop(name, None)
            if finder is not None:
                while finder in sys.meta_path:
                    sys.meta_path.remove(finder)
            trainer.validate_source_trees = original_validate
            trainer.activate_source_trees = original_activate
            sys.path[:] = path_snapshot
            sys.path_importer_cache.clear()
            sys.path_importer_cache.update(importer_cache_snapshot)
            if config_cleanup_failure is not None and failure is None:
                failure = config_cleanup_failure
        if (
            trainer.validate_source_trees is not original_validate
            or trainer.activate_source_trees is not original_activate
            or sys.path != path_snapshot
            or sys.meta_path != meta_path_snapshot
            or set(sys.path_importer_cache) != set(importer_cache_snapshot)
            or any(
                sys.path_importer_cache[key] is not value
                for key, value in importer_cache_snapshot.items()
            )
            or any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in _VENDOR_IMPORT_PREFIXES
                for name in sys.modules
                if name not in preexisting_vendor_modules
            )
        ):
            raise MatchedInferAdapterError(
                "captured source-tree state was not restored"
            )
        if failure is not None:
            raise failure


def _extract_exact_output(argv: Sequence[str]) -> Path:
    values: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--output":
            if index + 1 >= len(argv):
                raise MatchedInferAdapterError("--output has no value")
            values.append(argv[index + 1])
            index += 2
            continue
        if value.startswith("--output="):
            values.append(value.split("=", 1)[1])
        index += 1
    if len(values) != 1:
        raise MatchedInferAdapterError("exactly one --output is required")
    output = Path(values[0]).expanduser()
    if (
        not output.is_absolute()
        or os.path.normpath(str(output)) != str(output)
        or output.suffix.lower() != ".mp4"
        or output.name in {"", ".", ".."}
        or "\x00" in output.name
    ):
        raise MatchedInferAdapterError("logical output path differs")
    return output


def _same_directory_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(left.st_mode)
        and stat.S_ISDIR(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_uid == right.st_uid
        and left.st_gid == right.st_gid
        and left.st_rdev == right.st_rdev
        and left.st_nlink == right.st_nlink
    )


def resolve_publication_paths(
    logical_output: str | Path,
    inherited_binding: Mapping[str, Any],
) -> PublicationPaths:
    output = Path(logical_output).expanduser()
    if (
        not output.is_absolute()
        or os.path.normpath(str(output)) != str(output)
        or output.suffix.lower() != ".mp4"
    ):
        raise MatchedInferAdapterError("logical output is not a normalized MP4 path")
    try:
        binding = model_authority.validate_inherited_fd_binding(
            inherited_binding,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        task = model_authority.inherited_fd_row(
            binding, scope="task", role="publication_root"
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise MatchedInferAdapterError(str(error)) from error
    task_fd = task.get("fd")
    task_root_value = task.get("source_path")
    if type(task_fd) is not int or task_fd < 0 or not isinstance(task_root_value, str):
        raise MatchedInferAdapterError("task publication-root row differs")
    task_root = Path(task_root_value)
    if (
        not task_root.is_absolute()
        or os.path.normpath(str(task_root)) != str(task_root)
        or output.parent != task_root
    ):
        raise MatchedInferAdapterError(
            "logical output parent is not the inherited task publication root"
        )
    try:
        named = task_root.lstat()
        held = os.fstat(task_fd)
    except OSError as error:
        raise MatchedInferAdapterError(
            "task publication-root identity is unavailable"
        ) from error
    if task_root.is_symlink() or not _same_directory_identity(named, held):
        raise MatchedInferAdapterError("named and inherited task roots differ")
    logical_receipt = output.with_name(output.name + ".receipt.json")
    runtime_root = Path(f"/proc/self/fd/{task_fd}")
    paths = PublicationPaths(
        logical_output=output,
        logical_receipt=logical_receipt,
        runtime_output=runtime_root / output.name,
        runtime_receipt=runtime_root / logical_receipt.name,
        task_fd=task_fd,
        task_root=task_root,
    )
    for path in (paths.logical_output, paths.logical_receipt):
        if path.exists() or path.is_symlink():
            raise MatchedInferAdapterError(f"planned output is not fresh: {path}")
    return paths


def _inode_identity(path: Path) -> tuple[int, int, int, int, int, int, int]:
    info = path.lstat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
    )


def _open_publication_leaf(
    directory_fd: int, basename: str, *, expected_mode: int
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(basename, flags, dir_fd=directory_fd)
    try:
        os.set_inheritable(descriptor, False)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != expected_mode
            or info.st_size <= 0
        ):
            raise MatchedInferAdapterError(
                "captured publication leaf identity differs"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def replay_publication(
    paths: PublicationPaths, calls: _PatchCalls | None = None
) -> dict[str, Any]:
    if calls is not None and calls.output_fd is not None:
        if (
            calls.receipt_fd is None
            or calls.receipt_value is None
            or calls.receipt_raw is None
        ):
            raise MatchedInferAdapterError(
                "captured publication receipt closure differs"
            )
        output_info = os.fstat(calls.output_fd)
        receipt_info = os.fstat(calls.receipt_fd)
        output_identity_full = _exec_identity(output_info)
        receipt_identity_full = _exec_identity(receipt_info)
        output_raw = _pread_exact(calls.output_fd, output_info.st_size)
        receipt_raw = _pread_exact(calls.receipt_fd, receipt_info.st_size)
        output_value = calls.receipt_value.get("output", {})
        if (
            _exec_identity(paths.logical_output.lstat())
            != output_identity_full
            or _exec_identity(paths.runtime_output.lstat())
            != output_identity_full
            or _exec_identity(paths.logical_receipt.lstat())
            != receipt_identity_full
            or _exec_identity(paths.runtime_receipt.lstat())
            != receipt_identity_full
            or receipt_raw != calls.receipt_raw
            or hashlib.sha256(output_raw).hexdigest()
            != output_value.get("sha256")
            or len(output_raw) != output_value.get("size")
            or output_identity_full != output_value.get("publication_identity")
            or hashlib.sha256(receipt_raw).hexdigest()
            != hashlib.sha256(calls.receipt_raw).hexdigest()
            or os.get_inheritable(calls.output_fd)
            or os.get_inheritable(calls.receipt_fd)
        ):
            raise MatchedInferAdapterError(
                "captured publication/in-memory receipt differs"
            )
        return {
            "logical_output": str(paths.logical_output),
            "logical_receipt": str(paths.logical_receipt),
            "task_root": str(paths.task_root),
            "task_fd": paths.task_fd,
            "output_identity": output_identity_full,
            "receipt_identity": receipt_identity_full,
            "output_sha256": hashlib.sha256(output_raw).hexdigest(),
            "output_size": len(output_raw),
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_size": len(receipt_raw),
            "receipt_digest": calls.receipt_value.get("receipt_digest"),
            "bound_to_in_memory_receipt": True,
        }
    output_named = _inode_identity(paths.logical_output)
    output_runtime = _inode_identity(paths.runtime_output)
    receipt_named = _inode_identity(paths.logical_receipt)
    receipt_runtime = _inode_identity(paths.runtime_receipt)
    if (
        output_named != output_runtime
        or receipt_named != receipt_runtime
        or not stat.S_ISREG(output_named[2])
        or not stat.S_ISREG(receipt_named[2])
        or stat.S_IMODE(output_named[2]) != 0o444
        or stat.S_IMODE(receipt_named[2]) != 0o400
        or output_named[3] != 1
        or receipt_named[3] != 1
        or output_named[6] <= 0
        or receipt_named[6] <= 0
    ):
        raise MatchedInferAdapterError("named/proc-FD publication replay differs")
    raw = paths.logical_receipt.read_bytes()
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MatchedInferAdapterError("inference receipt is not UTF-8 JSON") from error
    if (
        not isinstance(receipt, dict)
        or receipt.get("output", {}).get("path") != str(paths.logical_output)
    ):
        raise MatchedInferAdapterError(
            "inference receipt did not retain the durable logical output path"
        )
    return {
        "logical_output": str(paths.logical_output),
        "logical_receipt": str(paths.logical_receipt),
        "task_root": str(paths.task_root),
        "task_fd": paths.task_fd,
        "output_identity": list(output_named),
        "receipt_identity": list(receipt_named),
        "output_sha256": _file_sha256(paths.logical_output),
        "receipt_sha256": _file_sha256(paths.logical_receipt),
        "bound_to_in_memory_receipt": False,
    }


def publish_publication_handoff(
    authority: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    verified = load_empty_publication_handoff_authority()
    if verified != dict(authority):
        raise MatchedInferAdapterError("publication handoff authority changed")
    if replay.get("bound_to_in_memory_receipt") is not True:
        raise MatchedInferAdapterError(
            "publication handoff lacks in-memory receipt binding"
        )
    payload: dict[str, Any] = {
        "schema_version": PUBLICATION_HANDOFF_PAYLOAD_SCHEMA,
        "task_id": verified["task_id"],
        "output_path": replay["logical_output"],
        "output_identity": replay["output_identity"],
        "output_sha256": replay["output_sha256"],
        "output_size": replay["output_size"],
        "receipt_path": replay["logical_receipt"],
        "receipt_identity": replay["receipt_identity"],
        "receipt_sha256": replay["receipt_sha256"],
        "receipt_size": replay["receipt_size"],
        "receipt_digest": replay["receipt_digest"],
    }
    payload["payload_digest"] = model_authority.object_sha256(payload)
    raw = model_authority.canonical_json_bytes(payload) + b"\n"
    descriptor = verified["fd"]
    if len(raw) > verified["capacity"]:
        raise MatchedInferAdapterError("publication handoff payload is too large")
    offset = 0
    while offset < len(raw):
        written = os.pwrite(descriptor, raw[offset:], offset)
        if written <= 0:
            raise MatchedInferAdapterError("publication handoff write is incomplete")
        offset += written
    os.ftruncate(descriptor, len(raw))
    os.fsync(descriptor)
    seal_mask = (
        fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_SEAL
    )
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seal_mask)
    before = os.fstat(descriptor)
    replay_raw = _pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    initial = verified["initial_identity"]
    immutable = ("device", "inode", "uid", "gid", "mode", "nlink", "rdev")
    if (
        replay_raw != raw
        or _exec_identity(before) != _exec_identity(after)
        or any(_exec_identity(before)[key] != initial[key] for key in immutable)
        or fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seal_mask
        or os.get_inheritable(descriptor)
    ):
        raise MatchedInferAdapterError("sealed publication handoff replay differs")
    return payload


@contextmanager
def translated_publication(
    paths: PublicationPaths,
    *,
    inference_module: Any = infer_lora,
) -> Iterator[_PatchCalls]:
    global _PATCH_ACTIVE
    if _PATCH_ACTIVE:
        raise MatchedInferAdapterError("publication translation is already active")
    original_encoded = inference_module._create_retained_encoded_output
    original_receipt = inference_module._atomic_write_json
    calls = _PatchCalls()

    def create_encoded(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if Path(path) != paths.logical_output or calls.encoded_output != 0:
            raise MatchedInferAdapterError("unexpected encoded-output publication call")
        calls.encoded_output += 1
        return original_encoded(paths.runtime_output, *args, **kwargs)

    def write_receipt(path: Path, value: Mapping[str, Any]) -> None:
        if Path(path) != paths.logical_receipt or calls.receipt != 0:
            raise MatchedInferAdapterError("unexpected receipt publication call")
        calls.receipt += 1
        if not isinstance(value, Mapping):
            raise MatchedInferAdapterError("in-memory receipt differs")
        receipt_raw = model_authority.canonical_json_bytes(value) + b"\n"
        try:
            receipt_value = json.loads(
                receipt_raw.decode("utf-8"),
                object_pairs_hook=_strict_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(token)
                ),
            )
        except (UnicodeError, ValueError, TypeError) as error:
            raise MatchedInferAdapterError(
                "in-memory receipt is not strict JSON"
            ) from error
        output_fd = _open_publication_leaf(
            paths.task_fd, paths.logical_output.name, expected_mode=0o444
        )
        try:
            output_info = os.fstat(output_fd)
            output_raw = _pread_exact(output_fd, output_info.st_size)
            output_value = receipt_value.get("output", {})
            if (
                not isinstance(output_value, Mapping)
                or output_value.get("path") != str(paths.logical_output)
                or output_value.get("sha256")
                != hashlib.sha256(output_raw).hexdigest()
                or output_value.get("size") != len(output_raw)
                or output_value.get("publication_identity")
                != _exec_identity(output_info)
            ):
                raise MatchedInferAdapterError(
                    "in-memory receipt/output publication differs"
                )
            original_receipt(paths.runtime_receipt, value)
            receipt_fd = _open_publication_leaf(
                paths.task_fd, paths.logical_receipt.name, expected_mode=0o400
            )
            actual_receipt = _pread_exact(
                receipt_fd, os.fstat(receipt_fd).st_size
            )
            if actual_receipt != receipt_raw:
                os.close(receipt_fd)
                raise MatchedInferAdapterError(
                    "published receipt bytes differ from in-memory receipt"
                )
        except BaseException:
            os.close(output_fd)
            raise
        calls.output_fd = output_fd
        calls.receipt_fd = receipt_fd
        calls.receipt_value = receipt_value
        calls.receipt_raw = receipt_raw

    _PATCH_ACTIVE = True
    inference_module._create_retained_encoded_output = create_encoded
    inference_module._atomic_write_json = write_receipt
    try:
        yield calls
    finally:
        hook_changed = (
            inference_module._create_retained_encoded_output is not create_encoded
            or inference_module._atomic_write_json is not write_receipt
        )
        inference_module._create_retained_encoded_output = original_encoded
        inference_module._atomic_write_json = original_receipt
        _PATCH_ACTIVE = False
        if (
            hook_changed
            or
            inference_module._create_retained_encoded_output is not original_encoded
            or inference_module._atomic_write_json is not original_receipt
        ):
            raise MatchedInferAdapterError("publication functions were not restored")


@contextmanager
def retained_ffmpeg_execution(
    authority: Mapping[str, Any], *, expected_calls: int
) -> Iterator[list[dict[str, Any]]]:
    """Force imageio's one encoder exec through the retained ffmpeg inode."""

    global _FFMPEG_PATCH_ACTIVE
    if _FFMPEG_PATCH_ACTIVE or expected_calls not in {0, 1}:
        raise MatchedInferAdapterError("retained ffmpeg patch state differs")
    verified = load_retained_ffmpeg_authority()
    if verified != dict(authority):
        raise MatchedInferAdapterError("retained ffmpeg authority changed")
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise MatchedInferAdapterError("imageio-ffmpeg is unavailable") from error
    write_frames = getattr(imageio_ffmpeg, "write_frames", None)
    owner = (
        sys.modules.get(getattr(write_frames, "__module__", ""))
        if callable(write_frames)
        else None
    )
    original_subprocess = getattr(owner, "subprocess", None)
    if (
        owner is None
        or original_subprocess is None
        or imageio_ffmpeg.get_ffmpeg_exe() != verified["row"]["source_path"]
    ):
        raise MatchedInferAdapterError("imageio-ffmpeg owner/executable differs")
    ffmpeg_fd = verified["row"]["fd"]
    calls: list[dict[str, Any]] = []

    class _RetainedSubprocess:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_subprocess, name)

        def Popen(self, *args: Any, **kwargs: Any) -> Any:
            if (
                calls
                or not args
                or not isinstance(args[0], (list, tuple))
                or not all(type(item) is str and item for item in args[0])
                or args[0][0] != verified["row"]["source_path"]
                or type(kwargs.get("pass_fds")) is not tuple
                or len(kwargs["pass_fds"]) != 1
                or type(kwargs["pass_fds"][0]) is not int
                or kwargs["pass_fds"][0] < 3
                or kwargs["pass_fds"][0] == ffmpeg_fd
                or kwargs.get("close_fds") is not True
                or kwargs.get("shell") not in (None, False)
                or "executable" in kwargs
                or "env" in kwargs
            ):
                raise MatchedInferAdapterError(
                    "imageio-ffmpeg retained launch differs"
                )
            output_fd = kwargs["pass_fds"][0]
            if os.get_inheritable(output_fd):
                raise MatchedInferAdapterError(
                    "imageio output descriptor remained inheritable"
                )
            load_retained_ffmpeg_authority()
            command = list(args[0])
            calls.append(
                {
                    "argv_digest": model_authority.object_sha256(command),
                    "output_fd": output_fd,
                    "ffmpeg_fd": ffmpeg_fd,
                }
            )
            kwargs["pass_fds"] = tuple(sorted((output_fd, ffmpeg_fd)))
            kwargs["executable"] = f"/proc/self/fd/{ffmpeg_fd}"
            kwargs["env"] = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
            try:
                return original_subprocess.Popen(
                    command,
                    *args[1:],
                    **kwargs,
                )
            finally:
                load_retained_ffmpeg_authority()
                if os.get_inheritable(output_fd):
                    raise MatchedInferAdapterError(
                        "imageio output descriptor leaked inheritable"
                    )

    scoped = _RetainedSubprocess()
    _FFMPEG_PATCH_ACTIVE = True
    owner.subprocess = scoped
    try:
        yield calls
    finally:
        hook_changed = getattr(owner, "subprocess", None) is not scoped
        owner.subprocess = original_subprocess
        _FFMPEG_PATCH_ACTIVE = False
        load_retained_ffmpeg_authority()
        if hook_changed or owner.subprocess is not original_subprocess:
            raise MatchedInferAdapterError(
                "imageio-ffmpeg subprocess origin was not restored"
            )
    if len(calls) != expected_calls:
        raise MatchedInferAdapterError("imageio-ffmpeg call count differs")


def run(
    argv: Sequence[str],
    *,
    ffmpeg_authority: Mapping[str, Any],
    publication_handoff: Mapping[str, Any],
    inference_main: Callable[[Sequence[str]], int] = infer_lora.main,
    binding_loader: Callable[..., Mapping[str, Any]] = (
        model_authority.load_inherited_fd_environment
    ),
    verify_origins: bool = True,
) -> dict[str, Any]:
    rank, expected_publications = distributed_publication_contract()
    origins_before = validate_frozen_origins() if verify_origins else {}
    logical_output = _extract_exact_output(argv)
    try:
        inherited = binding_loader(
            verify_open_fds=True,
            expected_inheritable=False,
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise MatchedInferAdapterError(str(error)) from error
    paths = resolve_publication_paths(logical_output, inherited)
    if publication_handoff.get("task_id") != inherited.get("task_id"):
        raise MatchedInferAdapterError("publication handoff task differs")
    origins_after: dict[str, str] = {}
    dependency_authority: dict[str, Any] = {}
    ffmpeg_calls: list[dict[str, Any]] = []
    calls = _PatchCalls()
    if verify_origins and _EARLY_SITE_PACKAGES is None:
        raise MatchedInferAdapterError(
            "pinned dependency site disappeared before inference"
        )
    dependency_scope = (
        pinned_dependency_import_paths(_EARLY_SITE_PACKAGES)
        if verify_origins
        else nullcontext(dependency_authority)
    )
    try:
        try:
            with retained_ffmpeg_execution(
                ffmpeg_authority,
                expected_calls=expected_publications,
            ) as ffmpeg_calls:
                with dependency_scope as dependency_authority:
                    with translated_publication(paths) as calls:
                        return_code = inference_main(list(argv))
        finally:
            origins_after = validate_frozen_origins() if verify_origins else {}
            if origins_after != origins_before:
                raise MatchedInferAdapterError(
                    "frozen inference origins changed during task"
                )
        if return_code != 0:
            raise MatchedInferAdapterError(
                f"infer_lora returned nonzero status: {return_code}"
            )
        if (
            calls.encoded_output != expected_publications
            or calls.receipt != expected_publications
        ):
            raise MatchedInferAdapterError("publication call count differs")
        replay = replay_publication(
            paths, calls if expected_publications == 1 else None
        )
        handoff_payload = (
            publish_publication_handoff(publication_handoff, replay)
            if rank == 0
            else None
        )
        return {
            "schema_version": "full644-exploratory-matched-infer-adapter-v2",
            "return_code": 0,
            "rank": rank,
            "world_size": 4,
            "rank0_decode_and_publication_only": True,
            "local_publication_call_count": expected_publications,
            "origins": origins_after,
            "dependency_import_authority": dependency_authority,
            "publication": replay,
            "publication_functions_restored": True,
            "publication_handoff_authority_digest": publication_handoff[
                "authority_digest"
            ],
            "publication_handoff_payload_digest": (
                None
                if handoff_payload is None
                else handoff_payload["payload_digest"]
            ),
            "rank0_publication_handoff_written_and_sealed": rank == 0,
            "ffmpeg_authority_digest": ffmpeg_authority["authority_digest"],
            "ffmpeg_exec_calls": ffmpeg_calls,
            "ffmpeg_subprocess_restored": True,
        }
    finally:
        closed: set[int] = set()
        for descriptor in (calls.output_fd, calls.receipt_fd):
            if type(descriptor) is int and descriptor not in closed:
                closed.add(descriptor)
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def main(argv: Sequence[str] | None = None) -> int:
    if _EARLY_INBOUND_BINDING is None:
        raise MatchedInferAdapterError(
            "rank adapter must enter through isolated executable bootstrap"
        )
    else:
        model_authority.validate_inherited_fd_binding(
            _EARLY_INBOUND_BINDING,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        if (
            _EARLY_RANK_CACHE is None
            or not _EARLY_RANK_CACHE.is_dir()
            or _EARLY_PYCACHE is None
            or sys.pycache_prefix != str(_EARLY_PYCACHE)
            or any(_EARLY_PYCACHE.iterdir())
            or _EARLY_SITE_PACKAGES is None
            or sys.path[-1] != str(_EARLY_SITE_PACKAGES)
        ):
            raise MatchedInferAdapterError("early rank cache disappeared")
    if _EARLY_FFMPEG_AUTHORITY is None:
        raise MatchedInferAdapterError("early retained ffmpeg authority disappeared")
    if _EARLY_PUBLICATION_HANDOFF is None:
        raise MatchedInferAdapterError(
            "early publication handoff authority disappeared"
        )
    result = run(
        sys.argv[1:] if argv is None else argv,
        ffmpeg_authority=_EARLY_FFMPEG_AUTHORITY,
        publication_handoff=_EARLY_PUBLICATION_HANDOFF,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
