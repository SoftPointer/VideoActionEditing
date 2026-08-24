#!/usr/bin/env python3
"""AUH r5f rank adapter with offset-independent inherited-FD reads.

The r5c adapter revalidated a capture receipt in a rank by dereferencing the
numeric ``private_parent.authority_fd`` recorded by the coordinator.  That FD
is deliberately *not* part of the rank allowlist: ranks inherit the model
namespace-root FD and the exact model-file FDs, not the coordinator's mutable
parent directory.  Once the rank has crossed ``exec``, that unowned number may
be closed or reused by an unrelated runtime object.

This eval-only adapter keeps the frozen r5c implementation, frozen
``infer_lora``, and every r5f namespace/reopen check unchanged.  Four ranks
inherit duplicates of the same open file descriptions, so a seek followed by
``read`` is not rank-local: another rank can move the shared offset between
the seek and read.  During one rank invocation this adapter additionally
replaces the authority's internal full-file reader with an offset-explicit
``pread`` loop that:

* structurally validates the signed capture receipt without dereferencing the
  unowned private-parent FD;
* validates the inherited allowlist and every live FD with the frozen code;
* walks the FD-view through the inherited namespace-root FD;
* checks every retained directory, symlink target, and resolved file identity;
* leaves the inherited open-file-description offset unchanged;
* restores the frozen validator, dependency context, and reader on all exit
  paths; and
* preserves a body exception when the dependency cleanup also reports a
  secondary lifecycle failure.

The coordinator remains responsible for the private parent's held-FD lifetime
checks.  No model, adapter, trainer, or inference source is modified here.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.machinery
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import types
from typing import Any, Iterator, Mapping


BASE_ADAPTER_SHA256 = (
    "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca"
)
MODEL_AUTHORITY_SHA256 = (
    "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf"
)
BASE_ADAPTER_MODULE = "_full644_exploratory_matched_infer_adapter_r5c_base"
_METHOD_ROOT = Path(__file__).resolve(strict=True).parent
_BASE_ADAPTER_PATH = _METHOD_ROOT / "full644_exploratory_matched_infer_adapter_v2.py"
_MODEL_AUTHORITY_PATH = (
    _METHOD_ROOT / "action_preservation_decoded_eval_model_authority_v2.py"
)


class MatchedInferAdapterR5FError(RuntimeError):
    """The r5f rank-only authority or patch lifecycle differs."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned_source(
    path: Path, expected_sha256: str, *, label: str
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_uid,
        before.st_gid,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_uid,
            after.st_gid,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or identity
        != (
            named.st_dev,
            named.st_ino,
            named.st_uid,
            named.st_gid,
            named.st_mode,
            named.st_nlink,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        )
        or _sha256_bytes(raw) != expected_sha256
    ):
        raise MatchedInferAdapterR5FError(f"frozen {label} source differs")
    try:
        return raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise MatchedInferAdapterR5FError(
            f"frozen {label} source is not UTF-8"
        ) from error


def _production_import_preflight() -> None:
    if __name__ != "__main__":
        return
    forbidden = {
        BASE_ADAPTER_MODULE,
        "action_preservation_decoded_eval_model_authority_v2",
        "infer_lora",
        "train_lora",
        "self_generated_action_preservation_v2",
        "tools",
        "tools.build_renderer_dataset",
        "tools.materialize_vae",
    }
    if forbidden.intersection(sys.modules):
        raise MatchedInferAdapterR5FError(
            "frozen inference source was imported before r5f bootstrap"
        )
    for value in sys.path:
        if type(value) is not str:
            raise MatchedInferAdapterR5FError("rank import path entry differs")
        if Path(value or os.curdir).resolve(strict=False) == _METHOD_ROOT:
            raise MatchedInferAdapterR5FError(
                "method root was importable before r5f bootstrap"
            )


def _load_early_model_authority() -> Any:
    name = "action_preservation_decoded_eval_model_authority_v2"
    source = _read_pinned_source(
        _MODEL_AUTHORITY_PATH,
        MODEL_AUTHORITY_SHA256,
        label="model authority",
    )
    if name in sys.modules:
        raise MatchedInferAdapterR5FError(
            "model authority was imported before early FD validation"
        )
    module = types.ModuleType(name)
    module.__file__ = str(_MODEL_AUTHORITY_PATH)
    module.__package__ = ""
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, origin=str(_MODEL_AUTHORITY_PATH)
    )
    sys.modules[name] = module
    try:
        exec(
            compile(
                source,
                str(_MODEL_AUTHORITY_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _configure_rank_cache_before_frozen_import() -> tuple[Path, Path]:
    base_value = os.environ.get("FULL644_MATCHED_RANK_CACHE_ROOT")
    rank_value = os.environ.get("LOCAL_RANK")
    if base_value is None or rank_value is None:
        raise MatchedInferAdapterR5FError("rank-cache authority is absent")
    try:
        rank = int(rank_value)
    except ValueError as error:
        raise MatchedInferAdapterR5FError("LOCAL_RANK differs") from error
    cache_parent = Path(base_value)
    if (
        rank not in range(4)
        or not cache_parent.is_absolute()
        or os.path.normpath(str(cache_parent)) != str(cache_parent)
        or not cache_parent.is_dir()
        or cache_parent.is_symlink()
    ):
        raise MatchedInferAdapterR5FError("rank-cache root differs")
    rank_root = cache_parent / f"rank-{rank}"
    names = (
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
    )
    try:
        rank_root.mkdir(mode=0o700)
        for name in names:
            (rank_root / name).mkdir(mode=0o700)
    except FileExistsError as error:
        raise MatchedInferAdapterR5FError(
            "rank-cache output is not fresh"
        ) from error
    environment = {
        "MIOPEN_USER_DB_PATH": rank_root / "miopen-user",
        "MIOPEN_CUSTOM_CACHE_DIR": rank_root / "miopen-custom",
        "XDG_CACHE_HOME": rank_root / "xdg",
        "TMPDIR": rank_root / "tmp",
        "TMP": rank_root / "tmp",
        "TEMP": rank_root / "tmp",
        "TRITON_CACHE_DIR": rank_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": rank_root / "inductor",
        "TORCH_EXTENSIONS_DIR": rank_root / "extensions",
        "PYTHONPYCACHEPREFIX": rank_root / "pycache",
        "HOME": rank_root / "home",
        "HF_HOME": rank_root / "hf",
        "TORCH_HOME": rank_root / "torch",
    }
    for key, value in environment.items():
        os.environ[key] = str(value)
    pycache = rank_root / "pycache"
    sys.pycache_prefix = str(pycache)
    if sys.pycache_prefix != str(pycache) or any(pycache.iterdir()):
        raise MatchedInferAdapterR5FError(
            "rank bytecode cache activation differs"
        )
    return rank_root, pycache


def _activate_site_packages_before_frozen_import() -> Path:
    raw = os.environ.get("FULL644_MATCHED_SITE_PACKAGES_ROOT")
    if raw is None:
        raise MatchedInferAdapterR5FError("site-packages authority is absent")
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
        or any(
            value and Path(value).resolve(strict=False) == root
            for value in sys.path
        )
    ):
        raise MatchedInferAdapterR5FError("site-packages authority differs")
    sys.path.append(str(root))
    if sys.path[-1] != str(root):
        raise MatchedInferAdapterR5FError(
            "site-packages activation differs"
        )
    return root


_EARLY_MODEL_AUTHORITY: Any | None = None
_EARLY_INBOUND_BINDING: dict[str, Any] | None = None
_EARLY_RANK_ROOT: Path | None = None
_EARLY_PYCACHE: Path | None = None
_EARLY_SITE_PACKAGES: Path | None = None


def _prepare_production_before_base_import() -> None:
    global _EARLY_MODEL_AUTHORITY
    global _EARLY_INBOUND_BINDING
    global _EARLY_RANK_ROOT
    global _EARLY_PYCACHE
    global _EARLY_SITE_PACKAGES
    if __name__ != "__main__":
        return
    if (
        sys.flags.no_site != 1
        or sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise MatchedInferAdapterR5FError(
            "rank adapter requires -I -S -B before authority FD entry"
        )
    if "FULL644_MATCHED_PYTHON_EXECUTABLE_BINDING" in os.environ:
        raise MatchedInferAdapterR5FError(
            "rank captured-source bootstrap did not consume its code FDs"
        )
    authority = _load_early_model_authority()
    try:
        inbound = authority.load_inherited_fd_environment(
            verify_open_fds=True,
            expected_inheritable=False,
        )
    except authority.ModelConsumptionAuthorityError as error:
        raise MatchedInferAdapterR5FError(str(error)) from error
    rank_root, pycache = _configure_rank_cache_before_frozen_import()
    site = _activate_site_packages_before_frozen_import()
    _EARLY_MODEL_AUTHORITY = authority
    _EARLY_INBOUND_BINDING = inbound
    _EARLY_RANK_ROOT = rank_root
    _EARLY_PYCACHE = pycache
    _EARLY_SITE_PACKAGES = site


def _load_base_adapter() -> Any:
    source = _read_pinned_source(
        _BASE_ADAPTER_PATH, BASE_ADAPTER_SHA256, label="r5c adapter"
    )
    existing = sys.modules.get(BASE_ADAPTER_MODULE)
    if existing is not None:
        return existing
    path_snapshot = list(sys.path)
    module = types.ModuleType(BASE_ADAPTER_MODULE)
    module.__file__ = str(_BASE_ADAPTER_PATH)
    module.__package__ = ""
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        BASE_ADAPTER_MODULE, loader=None, origin=str(_BASE_ADAPTER_PATH)
    )
    sys.modules[BASE_ADAPTER_MODULE] = module
    try:
        exec(
            compile(
                source,
                str(_BASE_ADAPTER_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(BASE_ADAPTER_MODULE, None)
        raise
    finally:
        if __name__ == "__main__":
            sys.path[:] = path_snapshot
    if module.__cached__ is not None or module.__file__ != str(_BASE_ADAPTER_PATH):
        raise MatchedInferAdapterR5FError("frozen r5c adapter origin differs")
    return module


_production_import_preflight()
_prepare_production_before_base_import()
base = _load_base_adapter()
model_authority = base.model_authority
MatchedInferAdapterError = base.MatchedInferAdapterError
if __name__ == "__main__" and model_authority is not _EARLY_MODEL_AUTHORITY:
    raise MatchedInferAdapterR5FError("early model authority origin differs")


def _capture_rows(
    capture: Mapping[str, Any], *, scope: str
) -> list[dict[str, Any]]:
    return [
        *model_authority._capture_fd_rows(capture, scope=scope),
        model_authority._capture_namespace_root_row(capture, scope=scope),
    ]


def _one_namespace_row(
    binding: Mapping[str, Any], *, scope: str
) -> dict[str, Any]:
    rows = [
        dict(item)
        for item in binding["fd_rows"]
        if item["scope"] == scope and item["role"] == "namespace_root"
    ]
    if len(rows) != 1:
        raise model_authority.ModelConsumptionAuthorityError(
            f"r5f inherited {scope} namespace-root closure differs"
        )
    return rows[0]


def _safe_relative(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise model_authority.ModelConsumptionAuthorityError(
            f"r5f {label} relative path differs"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\x00" in value:
        raise model_authority.ModelConsumptionAuthorityError(
            f"r5f {label} relative path differs"
        )
    normalized = pure.as_posix()
    if normalized in ("", "."):
        return "."
    if normalized != value:
        raise model_authority.ModelConsumptionAuthorityError(
            f"r5f {label} relative path differs"
        )
    return normalized


def _verify_capture_through_namespace_fd(
    capture: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    scope: str,
) -> None:
    """Replay one captured FD view without its non-inherited parent FD."""

    namespace = _one_namespace_row(binding, scope=scope)
    root_fd = namespace["fd"]
    expected_namespace = model_authority._capture_namespace_root_row(
        capture, scope=scope
    )
    if namespace != expected_namespace:
        raise model_authority.ModelConsumptionAuthorityError(
            f"r5f inherited {scope} namespace-root binding differs"
        )
    directory_rows = capture["view_directories"]
    expected_children: dict[str, set[str]] = {
        item["relative_path"]: set() for item in directory_rows
    }
    for item in directory_rows:
        relative = item["relative_path"]
        if relative == ".":
            continue
        pure = PurePosixPath(relative)
        parent = pure.parent.as_posix() or "."
        if parent not in expected_children:
            raise model_authority.ModelConsumptionAuthorityError(
                f"r5f captured {scope} view directory topology differs"
            )
        expected_children[parent].add(pure.name)
    for item in capture["files"]:
        pure = PurePosixPath(item["relative_path"])
        parent = pure.parent.as_posix() or "."
        if parent not in expected_children:
            raise model_authority.ModelConsumptionAuthorityError(
                f"r5f captured {scope} view leaf topology differs"
            )
        expected_children[parent].add(pure.name)
    for item in directory_rows:
        relative = _safe_relative(
            item.get("relative_path"), label=f"{scope} view directory"
        )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(relative, flags, dir_fd=root_fd)
            try:
                observed = model_authority._identity(os.fstat(descriptor))
                inheritable = os.get_inheritable(descriptor)
                children = set(os.listdir(descriptor))
            finally:
                os.close(descriptor)
        except OSError as error:
            raise model_authority.ModelConsumptionAuthorityError(
                f"r5f inherited {scope} view directory is unavailable"
            ) from error
        if (
            observed != item["identity"]
            or inheritable
            or children != expected_children[relative]
        ):
            raise model_authority.ModelConsumptionAuthorityError(
                f"r5f inherited {scope} view directory identity differs"
            )

    file_binding_by_relative = {
        item["relative_path"]: item
        for item in binding["fd_rows"]
        if item["scope"] == scope and item["role"] == "file"
    }
    for item in capture["files"]:
        relative = _safe_relative(
            item.get("relative_path"), label=f"{scope} view leaf"
        )
        bound = file_binding_by_relative.get(relative)
        if bound is None or bound != {
            "fd": item["authority_fd"],
            "scope": scope,
            "role": "file",
            "relative_path": relative,
            "source_path": item["path"],
            "identity": item["identity"],
        }:
            raise model_authority.ModelConsumptionAuthorityError(
                f"r5f inherited {scope} file binding differs"
            )
        try:
            leaf = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
            target = os.readlink(relative, dir_fd=root_fd)
            descriptor = os.open(
                relative,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                dir_fd=root_fd,
            )
            try:
                resolved = model_authority._identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)
        except OSError as error:
            raise model_authority.ModelConsumptionAuthorityError(
                f"r5f inherited {scope} FD-view leaf is unavailable"
            ) from error
        if (
            not stat.S_ISLNK(leaf.st_mode)
            or target != item["proc_fd_path"]
            or resolved != item["identity"]
        ):
            raise model_authority.ModelConsumptionAuthorityError(
                f"r5f inherited {scope} FD-view leaf identity differs"
            )


_FROZEN_VALIDATE_INHERITED = model_authority.validate_inherited_fd_binding
_FROZEN_READ_FD = model_authority._read_fd
_PREAD_CHUNK_SIZE = 16 * 1024 * 1024


def read_fd_with_pread_r5f(descriptor: int) -> bytes:
    """Read one current regular-file extent without changing its OFD offset."""

    size = os.fstat(descriptor).st_size
    offset = 0
    chunks: list[bytes] = []
    while offset < size:
        block = os.pread(
            descriptor,
            min(_PREAD_CHUNK_SIZE, size - offset),
            offset,
        )
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    return b"".join(chunks)


def validate_inherited_fd_binding_r5f(
    value: Any,
    *,
    model_capture: Mapping[str, Any] | None = None,
    adapter_capture: Mapping[str, Any] | None = None,
    task_publication_root: Mapping[str, Any] | None = None,
    verify_open_fds: bool,
    expected_inheritable: bool | None = None,
) -> dict[str, Any]:
    """Validate rank authority using only the descriptors ranks inherit."""

    model = (
        None
        if model_capture is None
        else model_authority.validate_model_capture_receipt(
            model_capture, verify_views=False
        )
    )
    adapter = (
        None
        if adapter_capture is None
        else model_authority.validate_adapter_capture_receipt(
            adapter_capture, verify_views=False
        )
    )
    if model is None and adapter is not None:
        raise model_authority.ModelConsumptionAuthorityError(
            "adapter capture supplied without model capture"
        )
    binding = _FROZEN_VALIDATE_INHERITED(
        value,
        model_capture=None,
        adapter_capture=None,
        task_publication_root=task_publication_root,
        verify_open_fds=verify_open_fds,
        expected_inheritable=expected_inheritable,
    )
    if model is None:
        return binding

    if (
        binding["model_capture_digest"] != model["capture_digest"]
        or binding["adapter_capture_digest"]
        != (None if adapter is None else adapter["capture_digest"])
        or (
            adapter is not None
            and adapter["task_id"] != binding["task_id"]
        )
    ):
        raise model_authority.ModelConsumptionAuthorityError(
            "r5f inherited capture digest/task binding differs"
        )
    inherited_numbers = {item["fd"] for item in binding["fd_rows"]}
    private_parent_numbers = {model["private_parent"]["authority_fd"]}
    if adapter is not None:
        private_parent_numbers.add(
            adapter["private_parent"]["authority_fd"]
        )
    if inherited_numbers.intersection(private_parent_numbers):
        raise model_authority.ModelConsumptionAuthorityError(
            "r5f private-parent FD was unexpectedly inherited"
        )

    expected_rows = [
        *_capture_rows(model, scope="model"),
        *([] if adapter is None else _capture_rows(adapter, scope="adapter")),
    ]
    observed_rows = [
        item for item in binding["fd_rows"] if item["scope"] != "task"
    ]
    if sorted(observed_rows, key=lambda item: item["fd"]) != sorted(
        expected_rows, key=lambda item: item["fd"]
    ):
        raise model_authority.ModelConsumptionAuthorityError(
            "r5f inherited FD/capture binding differs"
        )
    if verify_open_fds:
        _verify_capture_through_namespace_fd(model, binding, scope="model")
        if adapter is not None:
            _verify_capture_through_namespace_fd(
                adapter, binding, scope="adapter"
            )
    return binding


@contextmanager
def preserve_primary_context(manager: Any) -> Iterator[Any]:
    """Run a context manager without letting cleanup replace body failure."""

    entered = manager.__enter__()
    try:
        yield entered
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        try:
            suppressed = manager.__exit__(
                type(primary), primary, primary_traceback
            )
        except BaseException as cleanup:
            if cleanup is primary:
                raise
            if hasattr(primary, "add_note"):
                primary.add_note(
                    "secondary r5c dependency cleanup failure: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
            raise primary.with_traceback(primary_traceback) from cleanup
        if suppressed:
            return
        raise
    else:
        manager.__exit__(None, None, None)


_FROZEN_DEPENDENCY_CONTEXT = base.pinned_dependency_import_paths


def pinned_dependency_import_paths_r5f(site_root: str | Path) -> Any:
    return preserve_primary_context(_FROZEN_DEPENDENCY_CONTEXT(site_root))


@contextmanager
def patched_rank_validation() -> Iterator[None]:
    """Install the three r5f rank-only fixes and restore frozen globals."""

    if (
        model_authority.validate_inherited_fd_binding
        is not _FROZEN_VALIDATE_INHERITED
        or model_authority._read_fd is not _FROZEN_READ_FD
        or base.pinned_dependency_import_paths is not _FROZEN_DEPENDENCY_CONTEXT
        or base.infer_lora.model_authority is not model_authority
    ):
        raise MatchedInferAdapterR5FError("r5f patch origin differs")
    model_authority.validate_inherited_fd_binding = (
        validate_inherited_fd_binding_r5f
    )
    model_authority._read_fd = read_fd_with_pread_r5f
    base.pinned_dependency_import_paths = pinned_dependency_import_paths_r5f
    try:
        yield
    finally:
        changed = (
            model_authority.validate_inherited_fd_binding
            is not validate_inherited_fd_binding_r5f
            or model_authority._read_fd is not read_fd_with_pread_r5f
            or base.pinned_dependency_import_paths
            is not pinned_dependency_import_paths_r5f
        )
        model_authority.validate_inherited_fd_binding = (
            _FROZEN_VALIDATE_INHERITED
        )
        model_authority._read_fd = _FROZEN_READ_FD
        base.pinned_dependency_import_paths = _FROZEN_DEPENDENCY_CONTEXT
        if changed:
            raise MatchedInferAdapterR5FError("r5f patch lifecycle differs")


def _activate_production_rank() -> None:
    base.require_isolated_startup()
    if (
        _EARLY_INBOUND_BINDING is None
        or _EARLY_RANK_ROOT is None
        or _EARLY_PYCACHE is None
        or _EARLY_SITE_PACKAGES is None
        or model_authority is not _EARLY_MODEL_AUTHORITY
        or base.load_bootstrap_sealed_authority_fds()
        != _EARLY_INBOUND_BINDING
    ):
        raise MatchedInferAdapterR5FError(
            "early inherited authority replay differs"
        )
    base._EARLY_INBOUND_BINDING = _EARLY_INBOUND_BINDING
    base._EARLY_FFMPEG_AUTHORITY = base.load_retained_ffmpeg_authority()
    base._EARLY_PUBLICATION_HANDOFF = (
        base.load_empty_publication_handoff_authority()
    )
    if (
        base._EARLY_PUBLICATION_HANDOFF["task_id"]
        != base._EARLY_INBOUND_BINDING.get("task_id")
    ):
        raise MatchedInferAdapterR5FError("publication handoff task differs")
    base._EARLY_RANK_CACHE = _EARLY_RANK_ROOT
    base._EARLY_PYCACHE = _EARLY_PYCACHE
    base._EARLY_SITE_PACKAGES = _EARLY_SITE_PACKAGES
    if (
        not _EARLY_RANK_ROOT.is_dir()
        or _EARLY_RANK_ROOT.is_symlink()
        or not _EARLY_PYCACHE.is_dir()
        or _EARLY_PYCACHE.is_symlink()
        or any(_EARLY_PYCACHE.iterdir())
        or sys.pycache_prefix != str(_EARLY_PYCACHE)
        or sys.path[-1] != str(_EARLY_SITE_PACKAGES)
    ):
        raise MatchedInferAdapterR5FError("early rank isolation disappeared")
    base._validate_closed_local_import_paths()


def main(argv: list[str] | None = None) -> int:
    if __name__ == "__main__":
        _activate_production_rank()
    # Keep an inference/body failure primary even if our outer patch restore
    # independently detects lifecycle corruption.
    with preserve_primary_context(patched_rank_validation()):
        return base.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
