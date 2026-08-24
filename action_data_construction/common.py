"""Shared, dependency-free helpers for the MEV action dataset pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_PREFIX = "mev-action-edit"
PAIR_AUDIT_SCHEMA = f"{SCHEMA_PREFIX}-pair-audit-v5"
SAFE_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _replace_bytes(path, payload.encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        count = 0
        for row in rows:
            handle.write(canonical_bytes(row) + b"\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def publish_create_only(path: Path, value: Any) -> None:
    """Atomically publish JSON without permitting an overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_bytes(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def bool_field(value: Any) -> bool | None:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    if normalized in {"", "none", "null", "nan"}:
        return None
    raise ValueError(f"invalid boolean value: {value!r}")


def float_field(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    if not (parsed == parsed and abs(parsed) != float("inf")):
        raise ValueError(f"non-finite float: {value!r}")
    return parsed


def ensure_disjoint(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    source = source_root.expanduser().resolve(strict=True)
    output = output_root.expanduser().resolve(strict=False)
    if output == source or output in source.parents or source in output.parents:
        raise ValueError(f"output must not overlap read-only source: {source} vs {output}")
    return source, output


def split_for_uuid(uuid: str, salt: str = "mev-action-edit-split-v1") -> str:
    bucket = int(hashlib.sha256(f"{salt}\0{uuid}".encode()).hexdigest()[:8], 16) % 10000
    if bucket < 9000:
        return "train"
    if bucket < 9500:
        return "validation"
    return "test"


def source_inventory(source_root: Path, media_names: Iterable[str]) -> dict[str, Any]:
    """Hash names/stat metadata only; never writes or decodes the source tree."""

    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for name in sorted(set(media_names)):
        path = source_root / "videos" / name
        stat = path.stat()
        digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        count += 1
        total_bytes += stat.st_size
    return {
        "algorithm": "sha256(relative_name\\0size\\0mtime_ns\\n)",
        "media_count": count,
        "media_total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }
