"""Create an exact, receipt-bound retry manifest from Qwen-v16 errors.

The original candidate manifest is the only IID authority.  This helper first
requires every authoritative IID to have a valid Qwen-v16 terminal receipt,
then copies the *original JSONL bytes* for error rows into a new manifest.  It
never enumerates output directories to discover work and never rewrites an
existing artifact.

Publication is recoverable: ``--resume`` accepts an already-published file
only when its bytes are exactly the deterministic bytes that would be
published now.  A crash between the manifest and receipt publications can
therefore be resumed without overwriting either artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

try:  # Package execution: python -m motive.goku_full_motion_v16_retry_manifest
    from .goku_full_motion_qwen_v16 import (
        RECORD_SCHEMA,
        _iter_jsonl,
        _strict_read_object,
        _validate_terminal_receipt,
        object_sha256,
        validate_input_row,
    )
except ImportError:  # Standalone controller copy with frozen snapshot on PYTHONPATH.
    from motive.goku_full_motion_qwen_v16 import (  # type: ignore[no-redef]
        RECORD_SCHEMA,
        _iter_jsonl,
        _strict_read_object,
        _validate_terminal_receipt,
        object_sha256,
        validate_input_row,
    )


RETRY_RECEIPT_SCHEMA = "motive-goku-full-motion-v16-error-retry-manifest-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_KEYS = {
    "schema_version",
    "iid",
    "status",
    "input_digest",
    "input_row",
    "model",
    "runtime",
    "media_verification",
    "visual_input_digest",
    "source_stage",
    "target_stage",
    "source_census",
    "target_plan",
    "compiled_instruction",
    "error",
    "record_digest",
}


class GokuFullMotionV16RetryManifestError(RuntimeError):
    """The source terminal set or create-only publication is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: Path, *, context: str, allow_empty: bool = False) -> Path:
    if path.is_symlink() or not path.is_file():
        raise GokuFullMotionV16RetryManifestError(
            f"{context} is not a plain file: {path}"
        )
    if not allow_empty and path.stat().st_size == 0:
        raise GokuFullMotionV16RetryManifestError(
            f"{context} is empty: {path}"
        )
    return path.resolve(strict=True)


def _plain_directory(path: Path, *, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise GokuFullMotionV16RetryManifestError(
            f"{context} is not a plain directory: {path}"
        )
    return path.resolve(strict=True)


def _validate_result(
    *, row: Mapping[str, Any], qwen_root: Path, status: str
) -> dict[str, Any]:
    iid = str(row["iid"])
    result_path = qwen_root / "rows" / iid / "result.json"
    result = _strict_read_object(result_path)
    if set(result) != _RECORD_KEYS:
        raise GokuFullMotionV16RetryManifestError(
            f"Qwen result schema is open iid={iid}"
        )
    input_digest = object_sha256(row)
    if (
        result.get("schema_version") != RECORD_SCHEMA
        or result.get("iid") != iid
        or result.get("status") != status
        or result.get("input_digest") != input_digest
        or _canonical_bytes(result.get("input_row")) != _canonical_bytes(row)
    ):
        raise GokuFullMotionV16RetryManifestError(
            f"Qwen result identity/input binding differs iid={iid}"
        )
    record_digest = result.get("record_digest")
    bound = dict(result)
    bound["record_digest"] = None
    if (
        not isinstance(record_digest, str)
        or _SHA256_RE.fullmatch(record_digest) is None
        or record_digest != object_sha256(bound)
    ):
        raise GokuFullMotionV16RetryManifestError(
            f"Qwen result record digest differs iid={iid}"
        )
    if status == "error":
        error = result.get("error")
        if (
            not isinstance(error, Mapping)
            or not isinstance(error.get("type"), str)
            or not error["type"]
            or not isinstance(error.get("message"), str)
            or not error["message"]
        ):
            raise GokuFullMotionV16RetryManifestError(
                f"Qwen error result lacks error evidence iid={iid}"
            )
        passed = qwen_root / "passed" / f"{iid}.jsonl"
        if passed.exists() or passed.is_symlink():
            raise GokuFullMotionV16RetryManifestError(
                f"Qwen error unexpectedly has a passed fragment iid={iid}"
            )
    elif result.get("error") is not None:
        raise GokuFullMotionV16RetryManifestError(
            f"successful Qwen result contains error evidence iid={iid}"
        )
    return result


def _publish_create_only(path: Path, payload: bytes, *, resume: bool) -> None:
    if path.exists() or path.is_symlink():
        if (
            not resume
            or path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
        ):
            raise GokuFullMotionV16RetryManifestError(
                f"create-only target already exists or differs: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def derive_error_retry_manifest(
    *,
    input_manifest: Path,
    input_sha256: str,
    qwen_root: Path,
    output_manifest: Path,
    receipt_path: Path,
    expected_rows: int = 128,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate one closed terminal set and publish its exact error subset."""

    if _SHA256_RE.fullmatch(input_sha256) is None:
        raise GokuFullMotionV16RetryManifestError(
            "input_sha256 must be a lowercase SHA-256"
        )
    if type(expected_rows) is not int or expected_rows <= 0:
        raise GokuFullMotionV16RetryManifestError(
            "expected_rows must be a positive integer"
        )
    source = _plain_file(input_manifest, context="source candidate manifest")
    root = _plain_directory(qwen_root, context="source Qwen root")
    if _sha256_file(source) != input_sha256:
        raise GokuFullMotionV16RetryManifestError(
            "source candidate manifest SHA-256 differs"
        )
    for target, context in (
        (output_manifest, "retry manifest"),
        (receipt_path, "retry receipt"),
    ):
        if not target.is_absolute():
            raise GokuFullMotionV16RetryManifestError(
                f"{context} path is not absolute: {target}"
            )
        _plain_directory(target.parent, context=f"{context} parent")
    resolved_output = output_manifest.resolve(strict=False)
    resolved_receipt = receipt_path.resolve(strict=False)
    if resolved_output == resolved_receipt:
        raise GokuFullMotionV16RetryManifestError(
            "retry manifest and receipt paths must differ"
        )
    if source in {resolved_output, resolved_receipt}:
        raise GokuFullMotionV16RetryManifestError(
            "retry outputs must not alias the authoritative source manifest"
        )
    if root == resolved_output or root in resolved_output.parents:
        raise GokuFullMotionV16RetryManifestError(
            "retry manifest must not be published inside the source Qwen root"
        )
    if root == resolved_receipt or root in resolved_receipt.parents:
        raise GokuFullMotionV16RetryManifestError(
            "retry receipt must not be published inside the source Qwen root"
        )

    raw = source.read_bytes()
    rows = _iter_jsonl(source)
    raw_lines = raw.splitlines(keepends=True)
    if len(rows) != expected_rows or len(raw_lines) != expected_rows:
        raise GokuFullMotionV16RetryManifestError(
            f"source manifest row count differs: expected={expected_rows} "
            f"observed={len(rows)}"
        )

    error_lines: list[bytes] = []
    error_iids: list[str] = []
    terminal_evidence: list[dict[str, Any]] = []
    for row, raw_line in zip(rows, raw_lines, strict=True):
        validate_input_row(dict(row))
        iid = str(row["iid"])
        receipt_file = root / "terminal" / f"{iid}.receipt.json"
        receipt = _validate_terminal_receipt(
            receipt_file,
            output_root=root,
            iid=iid,
            input_digest=object_sha256(row),
        )
        _validate_result(row=row, qwen_root=root, status=str(receipt["status"]))
        terminal_evidence.append(
            {
                "iid": iid,
                "status": receipt["status"],
                "input_digest": receipt["input_digest"],
                "result_sha256": receipt["result_sha256"],
                "receipt_digest": receipt["receipt_digest"],
            }
        )
        if receipt["status"] == "error":
            error_iids.append(iid)
            error_lines.append(raw_line)

    output_payload = b"".join(error_lines)
    receipt: dict[str, Any] = {
        "schema_version": RETRY_RECEIPT_SCHEMA,
        "source_manifest": str(source),
        "source_manifest_sha256": input_sha256,
        "source_qwen_root": str(root),
        "expected_rows": expected_rows,
        "terminal_count": len(terminal_evidence),
        "error_count": len(error_iids),
        "error_iids": error_iids,
        "terminal_evidence": terminal_evidence,
        "retry_manifest": str(resolved_output),
        "retry_manifest_sha256": _sha256_bytes(output_payload),
        "receipt_digest": None,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    receipt_payload = _pretty_bytes(receipt)

    # Manifest first, receipt second.  --resume safely closes the only possible
    # interrupted state (an exact manifest with no receipt).
    _publish_create_only(resolved_output, output_payload, resume=resume)
    _publish_create_only(resolved_receipt, receipt_payload, resume=resume)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = derive_error_retry_manifest(
        input_manifest=args.input.expanduser(),
        input_sha256=args.input_sha256,
        qwen_root=args.qwen_root.expanduser(),
        output_manifest=args.output.expanduser(),
        receipt_path=args.receipt.expanduser(),
        expected_rows=args.expected_rows,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "terminal_count": receipt["terminal_count"],
                "error_count": receipt["error_count"],
                "error_iids": receipt["error_iids"],
                "retry_manifest": receipt["retry_manifest"],
                "retry_manifest_sha256": receipt["retry_manifest_sha256"],
                "receipt_digest": receipt["receipt_digest"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
