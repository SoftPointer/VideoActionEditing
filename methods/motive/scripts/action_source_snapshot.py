#!/usr/bin/env python3
"""Build and verify an immutable Motive/Lucy source snapshot.

The snapshot is authoritative even when the working tree contains untracked
files.  Git metadata is recorded for orientation, while SOURCE_FILES.jsonl and
its canonical digest bind the exact executable source bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SOURCE_ROOTS = ("lucy", "methods/motive")
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "runs",
    "experiments",
    "outputs",
}
EXCLUDED_FILE_NAMES = {".DS_Store", ".coverage"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
MANIFEST_NAME = "SOURCE_FILES.jsonl"
PROVENANCE_NAME = "SOURCE_PROVENANCE.json"
SNAPSHOT_SCHEMA = "motive-action-source-snapshot-v1"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.rstrip("\n")


def _included_files(repo_root: Path) -> Iterable[tuple[Path, Path]]:
    for root_name in SOURCE_ROOTS:
        source_root = repo_root / root_name
        if not source_root.is_dir():
            raise FileNotFoundError(f"missing source root: {source_root}")
        for directory, dir_names, file_names in os.walk(source_root):
            dir_names[:] = sorted(
                name for name in dir_names if name not in EXCLUDED_DIR_NAMES
            )
            directory_path = Path(directory)
            for file_name in sorted(file_names):
                if (
                    file_name in EXCLUDED_FILE_NAMES
                    or Path(file_name).suffix in EXCLUDED_SUFFIXES
                ):
                    continue
                source_path = directory_path / file_name
                relative_path = source_path.relative_to(repo_root)
                if source_path.is_symlink():
                    raise ValueError(
                        f"source snapshots do not admit symlinks: {relative_path}"
                    )
                if not source_path.is_file():
                    raise ValueError(
                        f"source snapshot entry is not a regular file: {relative_path}"
                    )
                yield source_path, relative_path


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _freeze_snapshot_tree(output_dir: Path) -> None:
    """Remove write bits after every snapshot byte has been committed."""

    for path in sorted(
        output_dir.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if path.is_symlink():
            raise ValueError(f"snapshot contains symlink: {path}")
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        else:
            raise ValueError(f"snapshot contains non-regular entry: {path}")
    output_dir.chmod(0o555)


def _make_tree_writable_for_cleanup(output_dir: Path) -> None:
    """Best-effort permission repair used only for failed, partial builds."""

    if not output_dir.exists() or output_dir.is_symlink():
        return
    for directory, dir_names, file_names in os.walk(output_dir):
        directory_path = Path(directory)
        try:
            directory_path.chmod(0o755)
        except OSError:
            pass
        for name in [*dir_names, *file_names]:
            path = directory_path / name
            if path.is_symlink():
                continue
            try:
                if path.is_dir():
                    path.chmod(0o755)
                elif path.is_file():
                    path.chmod(0o644)
            except OSError:
                pass


def build_snapshot(repo_root: Path, output_dir: Path) -> dict[str, object]:
    repo_root = repo_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse snapshot path: {output_dir}")
    output_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    try:
        for source_path, relative_path in _included_files(repo_root):
            target_path = output_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            mode = stat.S_IMODE(target_path.stat().st_mode) & ~0o222
            target_path.chmod(mode)
            rows.append(
                {
                    "mode": f"{mode:04o}",
                    "path": relative_path.as_posix(),
                    "sha256": _digest_file(target_path),
                    "size": target_path.stat().st_size,
                    "type": "file",
                }
            )
        rows.sort(key=lambda row: str(row["path"]))
        manifest_text = "".join(_canonical_json(row) + "\n" for row in rows)
        manifest_path = output_dir / MANIFEST_NAME
        _write_atomic(manifest_path, manifest_text)
        manifest_sha256 = _digest_file(manifest_path)
        tree_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        provenance = {
            "schema": SNAPSHOT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(repo_root),
            "source_roots": list(SOURCE_ROOTS),
            "source_file_count": len(rows),
            "source_tree_sha256": tree_sha256,
            "source_manifest_sha256": manifest_sha256,
            "git_base_commit": _git(repo_root, "rev-parse", "HEAD"),
            "git_status_short": _git(
                repo_root,
                "status",
                "--short",
                "--",
                *SOURCE_ROOTS,
            ).splitlines(),
        }
        _write_atomic(
            output_dir / PROVENANCE_NAME,
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        )
        _freeze_snapshot_tree(output_dir)
        verify_snapshot(output_dir, expected_tree_sha256=tree_sha256)
        return provenance
    except BaseException:
        _make_tree_writable_for_cleanup(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _load_manifest(snapshot: Path) -> tuple[list[dict[str, object]], str]:
    manifest_path = snapshot / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing snapshot manifest: {manifest_path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        rows.append(row)
    canonical_text = "".join(_canonical_json(row) + "\n" for row in rows)
    actual_text = manifest_path.read_text(encoding="utf-8")
    if actual_text != canonical_text:
        raise ValueError("SOURCE_FILES.jsonl is not canonical")
    return rows, canonical_text


def verify_snapshot(
    snapshot: Path,
    *,
    expected_tree_sha256: str | None = None,
) -> dict[str, object]:
    unresolved_snapshot = snapshot.expanduser()
    if unresolved_snapshot.is_symlink():
        raise ValueError("source snapshot root must not be a symlink")
    snapshot = unresolved_snapshot.resolve()
    rows, manifest_text = _load_manifest(snapshot)
    tree_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if expected_tree_sha256 and tree_sha256 != expected_tree_sha256:
        raise ValueError(
            "source tree digest mismatch: "
            f"expected={expected_tree_sha256} actual={tree_sha256}"
        )
    seen: set[str] = set()
    expected_directories: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        if (
            not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise ValueError(f"invalid or duplicate snapshot path: {relative!r}")
        seen.add(relative)
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        path = snapshot / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing/non-regular snapshot file: {relative}")
        size = path.stat().st_size
        digest = _digest_file(path)
        mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
        if size != int(row.get("size", -1)):
            raise ValueError(f"snapshot size mismatch: {relative}")
        if digest != row.get("sha256"):
            raise ValueError(f"snapshot SHA-256 mismatch: {relative}")
        if mode != row.get("mode"):
            raise ValueError(
                f"snapshot mode mismatch: {relative}; "
                f"expected={row.get('mode')} actual={mode}"
            )

    expected_files = seen | {MANIFEST_NAME, PROVENANCE_NAME}
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, dir_names, file_names in os.walk(
        snapshot,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        directory_mode = stat.S_IMODE(directory_path.stat().st_mode)
        if directory_mode & 0o222:
            relative_directory = directory_path.relative_to(snapshot)
            label = relative_directory.as_posix()
            raise ValueError(
                "snapshot directory remains writable: "
                f"{label if label != '.' else '<root>'}"
            )
        for name in [*dir_names, *file_names]:
            path = directory_path / name
            relative = path.relative_to(snapshot).as_posix()
            entry_stat = path.lstat()
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ValueError(f"snapshot contains symlink: {relative}")
            if stat.S_ISDIR(entry_stat.st_mode):
                actual_directories.add(relative)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ValueError(
                    f"snapshot contains non-regular entry: {relative}"
                )
            if stat.S_IMODE(entry_stat.st_mode) & 0o222:
                raise ValueError(
                    f"snapshot file remains writable: {relative}"
                )
            actual_files.add(relative)
    if actual_files != expected_files:
        extra = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise ValueError(
            "snapshot file closure mismatch: "
            f"extra={extra} missing={missing}"
        )
    if actual_directories != expected_directories:
        extra = sorted(actual_directories - expected_directories)
        missing = sorted(expected_directories - actual_directories)
        raise ValueError(
            "snapshot directory closure mismatch: "
            f"extra={extra} missing={missing}"
        )

    provenance_path = snapshot / PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("invalid source snapshot provenance schema")
    if provenance.get("source_tree_sha256") != tree_sha256:
        raise ValueError("provenance source_tree_sha256 mismatch")
    if provenance.get("source_file_count") != len(rows):
        raise ValueError("provenance source_file_count mismatch")
    if provenance.get("source_manifest_sha256") != _digest_file(
        snapshot / MANIFEST_NAME
    ):
        raise ValueError("provenance source_manifest_sha256 mismatch")
    return {
        "source_file_count": len(rows),
        "source_tree_sha256": tree_sha256,
        "source_manifest_sha256": provenance["source_manifest_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repo-root", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--snapshot", required=True, type=Path)
    verify.add_argument("--expected-tree-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build_snapshot(args.repo_root, args.output_dir)
    else:
        result = verify_snapshot(
            args.snapshot,
            expected_tree_sha256=args.expected_tree_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
