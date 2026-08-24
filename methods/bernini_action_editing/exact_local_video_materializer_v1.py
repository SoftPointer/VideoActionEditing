#!/usr/bin/env python3
"""Install the repository-local video materializer as a sealed namespace.

Several Bernini inference modules historically use the absolute lazy import
``from tools import materialize_vae``.  That name is not globally unique: an
unrelated dependency can populate ``sys.modules["tools"]`` before the first
video is decoded.  Merely prepending the method directory to ``sys.path`` does
not repair that case because Python consults ``sys.modules`` first.

This module transactionally replaces the ambiguous namespace with two exact,
source-only modules from the immutable method tree.  It verifies their source
bytes and file identities, executes neither bytecode nor a search-path import,
and gives the resulting ``tools`` package an empty search path.  Consequently
all later legacy lazy imports resolve to the same authenticated materializer.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import os
from pathlib import Path
import stat
import sys
import threading
import types
from typing import Any


METHOD_ROOT = Path(__file__).resolve(strict=True).parent
TOOLS_ROOT = METHOD_ROOT / "tools"
BUILD_RENDERER_DATASET_PATH = TOOLS_ROOT / "build_renderer_dataset.py"
MATERIALIZE_VAE_PATH = TOOLS_ROOT / "materialize_vae.py"
BUILD_RENDERER_DATASET_SHA256 = (
    "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
)
MATERIALIZE_VAE_SHA256 = (
    "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
)
SEALED_PACKAGE_MARKER = "bernini-exact-local-video-tools-v1"
_INSTALL_LOCK = threading.RLock()
_MISSING = object()


class ExactLocalVideoMaterializerError(RuntimeError):
    """The local video-tool source closure is absent, changed, or ambiguous."""


def _read_exact_source(
    path: Path, expected_sha256: str, *, label: str
) -> tuple[Path, str]:
    """Read one unchanged plain source file through a no-follow descriptor."""

    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_sha256)
    ):
        raise ExactLocalVideoMaterializerError(f"{label} source pin differs")
    try:
        requested = path.absolute()
        named_before = requested.lstat()
    except OSError as error:
        raise ExactLocalVideoMaterializerError(
            f"{label} source is unavailable: {requested}"
        ) from error
    if not stat.S_ISREG(named_before.st_mode) or stat.S_ISLNK(named_before.st_mode):
        raise ExactLocalVideoMaterializerError(
            f"{label} source is not a plain non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise ExactLocalVideoMaterializerError(
            f"{label} source cannot be opened without following links"
        ) from error
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened_before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named_after = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ExactLocalVideoMaterializerError(
            f"{label} source changed while it was read"
        ) from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    identities = [
        tuple(getattr(value, field) for field in identity_fields)
        for value in (named_before, opened_before, opened_after, named_after)
    ]
    if (
        len(set(identities)) != 1
        or not stat.S_ISREG(opened_before.st_mode)
        or resolved != requested
        or digest.hexdigest() != expected_sha256
    ):
        raise ExactLocalVideoMaterializerError(
            f"{label} source identity or SHA-256 differs"
        )
    try:
        source = b"".join(chunks).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise ExactLocalVideoMaterializerError(
            f"{label} source is not UTF-8"
        ) from error
    return resolved, source


def _execute_exact_module(
    name: str, path: Path, expected_sha256: str
) -> Any:
    resolved, source = _read_exact_source(path, expected_sha256, label=name)
    module = types.ModuleType(name)
    module.__file__ = str(resolved)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, origin=str(resolved)
    )
    sys.modules[name] = module
    try:
        exec(
            compile(source, str(resolved), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if (
        module.__file__ != str(resolved)
        or module.__cached__ is not None
        or module.__package__ != "tools"
    ):
        sys.modules.pop(name, None)
        raise ExactLocalVideoMaterializerError(
            f"{name} executed-source origin differs"
        )
    return module


def _new_sealed_tools_package() -> Any:
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
    package.__bernini_exact_local_video_tools__ = SEALED_PACKAGE_MARKER
    return package


def _validate_installed_package(package: Any) -> Any:
    builder = sys.modules.get("tools.build_renderer_dataset")
    materializer = sys.modules.get("tools.materialize_vae")
    package_spec = getattr(package, "__spec__", None)
    try:
        builder_path = Path(getattr(builder, "__file__", "")).resolve(strict=True)
        materializer_path = Path(
            getattr(materializer, "__file__", "")
        ).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise ExactLocalVideoMaterializerError(
            "installed local video-tool module origin is unavailable"
        ) from error
    if (
        getattr(package, "__bernini_exact_local_video_tools__", None)
        != SEALED_PACKAGE_MARKER
        or tuple(getattr(package, "__path__", ("missing",))) != ()
        or package_spec is None
        or tuple(
            getattr(package_spec, "submodule_search_locations", ("missing",))
        )
        != ()
        or builder is None
        or materializer is None
        or getattr(package, "build_renderer_dataset", None) is not builder
        or getattr(package, "materialize_vae", None) is not materializer
        or getattr(materializer, "raw_builder", None) is not builder
        or builder_path != BUILD_RENDERER_DATASET_PATH
        or materializer_path != MATERIALIZE_VAE_PATH
    ):
        raise ExactLocalVideoMaterializerError(
            "installed local video-tool module closure differs"
        )
    _read_exact_source(
        BUILD_RENDERER_DATASET_PATH,
        BUILD_RENDERER_DATASET_SHA256,
        label="tools.build_renderer_dataset",
    )
    _read_exact_source(
        MATERIALIZE_VAE_PATH,
        MATERIALIZE_VAE_SHA256,
        label="tools.materialize_vae",
    )
    return materializer


def install_exact_local_video_materializer() -> Any:
    """Install and return the exact local ``tools.materialize_vae`` module.

    An already-installed sealed package is revalidated.  Any unrelated
    ``tools`` package and submodules are displaced only after all source bytes
    have passed their pins; if execution fails, the previous namespace is
    restored transactionally.
    """

    with _INSTALL_LOCK:
        existing = sys.modules.get("tools")
        if (
            existing is not None
            and getattr(existing, "__bernini_exact_local_video_tools__", None)
            == SEALED_PACKAGE_MARKER
        ):
            return _validate_installed_package(existing)

        # Authenticate both files before altering the process import state.
        _read_exact_source(
            BUILD_RENDERER_DATASET_PATH,
            BUILD_RENDERER_DATASET_SHA256,
            label="tools.build_renderer_dataset",
        )
        _read_exact_source(
            MATERIALIZE_VAE_PATH,
            MATERIALIZE_VAE_SHA256,
            label="tools.materialize_vae",
        )
        displaced = {
            name: sys.modules.get(name, _MISSING)
            for name in tuple(sys.modules)
            if name == "tools" or name.startswith("tools.")
        }
        path_snapshot = list(sys.path)
        try:
            for name in displaced:
                sys.modules.pop(name, None)
            package = _new_sealed_tools_package()
            sys.modules["tools"] = package
            builder = _execute_exact_module(
                "tools.build_renderer_dataset",
                BUILD_RENDERER_DATASET_PATH,
                BUILD_RENDERER_DATASET_SHA256,
            )
            package.build_renderer_dataset = builder
            materializer = _execute_exact_module(
                "tools.materialize_vae",
                MATERIALIZE_VAE_PATH,
                MATERIALIZE_VAE_SHA256,
            )
            package.materialize_vae = materializer
            materializer = _validate_installed_package(package)
        except BaseException:
            for name in tuple(sys.modules):
                if name == "tools" or name.startswith("tools."):
                    sys.modules.pop(name, None)
            for name, value in displaced.items():
                if value is not _MISSING:
                    sys.modules[name] = value
            raise
        finally:
            # materialize_vae historically prepends METHOD_ROOT.  The caller
            # owns its import path, so exact loading must not mutate it.
            sys.path[:] = path_snapshot
        return materializer


__all__ = [
    "ExactLocalVideoMaterializerError",
    "install_exact_local_video_materializer",
]
