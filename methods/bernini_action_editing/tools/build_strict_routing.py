#!/usr/bin/env python3
"""Build the frozen strict-359 Bernini supervision-routing artifact.

The finalized 644-row VAE release contains a provenance-bound boolean named
``strict_selection_gates_all_true`` in every sample receipt.  This tool turns
that *existing* cohort decision into a complete ``bernini-cdf-routing-v1``
JSONL:

* strict rows are routed to ``motion_only``;
* non-strict rows are routed to ``reject``.

No media, mask, target-derived signal, or heuristic is inspected here.  The
builder fails closed unless the dataset summary, index, all sample receipts,
their hashes/digests, the 359/285 cohort counts, and all receipt/shard paths are
mutually bound.  Publication is create-only: the JSONL and its hash sidecar are
staged first and the receipt is linked last as the ready marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


SUMMARY_SCHEMA = "bernini-r-action-vae-dataset-summary-v2"
INDEX_ROW_SCHEMA = "bernini-r-action-vae-index-row-v2"
SAMPLE_RECEIPT_SCHEMA = "bernini-r-action-vae-sample-receipt-v2"
ROUTING_SCHEMA = "bernini-cdf-routing-v1"
ROUTING_RECEIPT_SCHEMA = "bernini-cdf-strict-routing-receipt-v1"
EXPECTED_INCLUSION_POLICY = "natural_release_all"
EXPECTED_ROWS = 644
EXPECTED_STRICT_ROWS = 359
EXPECTED_NON_STRICT_ROWS = 285
EXPECTED_FRAME_COUNT = 81
EXPECTED_FPS = 25.0
EXPECTED_LATENT_FRAME_COUNT = 21
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class StrictRoutingError(RuntimeError):
    """A source contract or create-only publication invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise StrictRoutingError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise StrictRoutingError(f"{context} is not a lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictRoutingError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise StrictRoutingError(f"non-finite JSON number: {value}")


def _decode_json_object(payload: bytes | str, *, context: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except StrictRoutingError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictRoutingError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise StrictRoutingError(f"{context} must contain one JSON object")
    return value


def _absolute_path(value: str | Path, *, context: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise StrictRoutingError(f"{context} must be an absolute path")
    return path


def _plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise StrictRoutingError(f"missing {context}: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise StrictRoutingError(f"{context} is not a plain file: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise StrictRoutingError(f"cannot resolve {context}: {path}: {error}") from error


def _plain_directory(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise StrictRoutingError(f"missing {context}: {path}") from error
    if not stat.S_ISDIR(mode) or path.is_symlink():
        raise StrictRoutingError(f"{context} is not a plain directory: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise StrictRoutingError(f"cannot resolve {context}: {path}: {error}") from error


def _read_plain_bytes(path: Path, *, context: str) -> bytes:
    resolved = _plain_file(path, context=context)
    try:
        before = resolved.stat()
        with resolved.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                before.st_dev != opened.st_dev
                or before.st_ino != opened.st_ino
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise StrictRoutingError(f"{context} changed while opening: {resolved}")
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise StrictRoutingError(f"cannot read {context}: {resolved}: {error}") from error
    if (
        opened.st_dev != after.st_dev
        or opened.st_ino != after.st_ino
        or opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
    ):
        raise StrictRoutingError(f"{context} changed while reading: {resolved}")
    return payload


def _path_from_field(value: Any, *, context: str, kind: str) -> Path:
    if type(value) is not str or not value:
        raise StrictRoutingError(f"{context} path must be non-empty text")
    path = _absolute_path(value, context=context)
    if kind == "file":
        return _plain_file(path, context=context)
    if kind == "directory":
        return _plain_directory(path, context=context)
    raise AssertionError(f"unknown path kind: {kind}")


def _validate_summary(summary: Mapping[str, Any]) -> None:
    candidate = dict(summary)
    declared_digest = candidate.pop("summary_digest", None)
    _required_sha256(declared_digest, context="dataset summary digest")
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise StrictRoutingError("dataset summary schema differs")
    if object_sha256(candidate) != declared_digest:
        raise StrictRoutingError("dataset summary digest differs")
    safety = {
        "complete": True,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    if any(summary.get(key) is not value for key, value in safety.items()):
        raise StrictRoutingError("dataset summary safety/completion state differs")
    if summary.get("experimental_inclusion_policy") != EXPECTED_INCLUSION_POLICY:
        raise StrictRoutingError("dataset summary inclusion policy differs")
    expected_counts = {
        "expected_sample_count": EXPECTED_ROWS,
        "materialized_sample_count": EXPECTED_ROWS,
        "missing_sample_count": 0,
        "raw_strict_selection_rows": EXPECTED_STRICT_ROWS,
        "raw_non_strict_selection_rows": EXPECTED_NON_STRICT_ROWS,
        "materialized_strict_selection_rows": EXPECTED_STRICT_ROWS,
        "materialized_non_strict_selection_rows": EXPECTED_NON_STRICT_ROWS,
        "frame_count": EXPECTED_FRAME_COUNT,
        "latent_frame_count": EXPECTED_LATENT_FRAME_COUNT,
    }
    if any(summary.get(key) != value for key, value in expected_counts.items()):
        raise StrictRoutingError("dataset summary 644/359/285 or temporal counts differ")
    if summary.get("missing_sample_ids") != []:
        raise StrictRoutingError("dataset summary must have no missing sample IDs")
    fps = summary.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != EXPECTED_FPS:
        raise StrictRoutingError("dataset summary FPS differs")
    bucket_counts = summary.get("bucket_counts")
    if (
        not isinstance(bucket_counts, dict)
        or not bucket_counts
        or any(type(count) is not int or count <= 0 for count in bucket_counts.values())
        or sum(bucket_counts.values()) != EXPECTED_ROWS
    ):
        raise StrictRoutingError("dataset summary bucket counts differ")
    _required_sha256(summary.get("index_sha256"), context="dataset index hash")
    _required_sha256(
        summary.get("vae_identity_digest"), context="dataset VAE identity digest"
    )


def _decode_index(payload: bytes) -> list[dict[str, Any]]:
    if not payload.endswith(b"\n"):
        raise StrictRoutingError("dataset index must end with one newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StrictRoutingError(f"dataset index is not UTF-8: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise StrictRoutingError(f"blank dataset index row at line {line_number}")
        rows.append(
            _decode_json_object(line, context=f"dataset index row {line_number}")
        )
    if len(rows) != EXPECTED_ROWS:
        raise StrictRoutingError(
            f"dataset index has {len(rows)} rows; expected {EXPECTED_ROWS}"
        )
    return rows


def _validate_sample_receipt(
    *,
    row: Mapping[str, Any],
    iid: str,
    receipt_payload: bytes,
    inclusion_policy: str,
) -> tuple[dict[str, Any], bool, str]:
    expected_file_sha = _required_sha256(
        row.get("sample_receipt_sha256"),
        context=f"sample receipt file hash for {iid}",
    )
    actual_file_sha = bytes_sha256(receipt_payload)
    if actual_file_sha != expected_file_sha:
        raise StrictRoutingError(f"sample receipt file hash differs: {iid}")
    receipt = _decode_json_object(receipt_payload, context=f"sample receipt {iid}")
    candidate = dict(receipt)
    declared_digest = candidate.pop("receipt_digest", None)
    _required_sha256(declared_digest, context=f"sample receipt digest for {iid}")
    if object_sha256(candidate) != declared_digest:
        raise StrictRoutingError(f"sample receipt digest differs: {iid}")
    if (
        receipt.get("schema_version") != SAMPLE_RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("iid") != iid
    ):
        raise StrictRoutingError(f"sample receipt schema or IID differs: {iid}")
    safety = {
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "shared_i0_exact": True,
    }
    if any(receipt.get(key) is not value for key, value in safety.items()):
        raise StrictRoutingError(f"sample receipt safety state differs: {iid}")
    if receipt.get("experimental_inclusion_policy") != inclusion_policy:
        raise StrictRoutingError(f"sample receipt inclusion policy differs: {iid}")
    if (
        receipt.get("frame_count") != EXPECTED_FRAME_COUNT
        or receipt.get("latent_frame_count") != EXPECTED_LATENT_FRAME_COUNT
        or isinstance(receipt.get("fps"), bool)
        or not isinstance(receipt.get("fps"), (int, float))
        or float(receipt["fps"]) != EXPECTED_FPS
    ):
        raise StrictRoutingError(f"sample receipt temporal contract differs: {iid}")
    strict = receipt.get("strict_selection_gates_all_true")
    if type(strict) is not bool:
        raise StrictRoutingError(f"sample receipt strict-selection flag differs: {iid}")
    gates_text = receipt.get("selection_gates_json")
    if type(gates_text) is not str:
        raise StrictRoutingError(f"sample receipt selection gates are missing: {iid}")
    gates = _decode_json_object(gates_text, context=f"selection gates for {iid}")
    if not gates or any(type(value) is not bool for value in gates.values()):
        raise StrictRoutingError(f"sample receipt selection gates differ: {iid}")
    if all(gates.values()) is not strict:
        raise StrictRoutingError(
            f"sample receipt strict-selection flag disagrees with gates: {iid}"
        )

    equality_fields = (
        "parquet_sha256",
        "materialized_row_digest",
        "bucket_hw",
        "posterior_parameters_shape",
    )
    if any(receipt.get(field) != row.get(field) for field in equality_fields):
        raise StrictRoutingError(f"sample receipt/index metadata differs: {iid}")
    _required_sha256(receipt.get("parquet_sha256"), context=f"shard hash for {iid}")
    _required_sha256(
        receipt.get("materialized_row_digest"),
        context=f"materialized row digest for {iid}",
    )
    return receipt, strict, declared_digest


def _review_router_digest(
    rows: Sequence[Mapping[str, Any]], *, default_tier: str
) -> str:
    serial = {
        str(row["iid"]): {
            "tier": row["tier"],
            "full_target_weight": row["full_target_weight"],
            "review": row["review"],
        }
        for row in rows
    }
    return object_sha256(
        {
            "schema": ROUTING_SCHEMA,
            "default_tier": default_tier,
            "routes": dict(sorted(serial.items())),
        }
    )


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_create_only(
    *, output: Path, routing_payload: bytes, receipt_payload: bytes, hash_payload: bytes
) -> None:
    receipt_path = Path(f"{output}.receipt.json")
    hash_path = Path(f"{output}.sha256")
    destinations = (output, hash_path, receipt_path)
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise StrictRoutingError(f"create-only output exists: {destination}")
    published: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging.", dir=output.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        staged = (
            (staging / output.name, output, routing_payload),
            (staging / hash_path.name, hash_path, hash_payload),
            # Receipt is the ready marker and is deliberately published last.
            (staging / receipt_path.name, receipt_path, receipt_payload),
        )
        try:
            for source, _destination, payload in staged:
                _write_fsynced(source, payload)
            for source, destination, _payload in staged:
                try:
                    os.link(source, destination)
                except FileExistsError as error:
                    raise StrictRoutingError(
                        f"create-only output appeared during publication: {destination}"
                    ) from error
                published.append(destination)
            try:
                directory_descriptor = os.open(output.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError as error:
                raise StrictRoutingError(
                    f"cannot fsync output directory: {output.parent}: {error}"
                ) from error
        except Exception:
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            raise


def build_strict_routing(
    *,
    dataset_summary: Path,
    expected_dataset_summary_sha256: str,
    output_jsonl: Path,
) -> dict[str, Any]:
    """Validate the complete frozen release and atomically emit its routes."""

    expected_summary_sha = _required_sha256(
        expected_dataset_summary_sha256, context="expected dataset summary hash"
    )
    summary_input = _absolute_path(dataset_summary, context="dataset summary")
    summary_path = _plain_file(summary_input, context="dataset summary")
    summary_payload = _read_plain_bytes(summary_path, context="dataset summary")
    summary_sha = bytes_sha256(summary_payload)
    if summary_sha != expected_summary_sha:
        raise StrictRoutingError(
            "dataset summary file hash differs from the caller-pinned hash"
        )
    summary = _decode_json_object(summary_payload, context="dataset summary")
    _validate_summary(summary)

    index_path = _path_from_field(
        summary.get("index_path"), context="dataset index", kind="file"
    )
    index_payload = _read_plain_bytes(index_path, context="dataset index")
    index_sha = bytes_sha256(index_payload)
    if index_sha != summary["index_sha256"]:
        raise StrictRoutingError("dataset index file hash differs from summary")
    index_rows = _decode_index(index_payload)

    shards_directory = _path_from_field(
        summary.get("shards_directory"), context="dataset shards directory", kind="directory"
    )
    if shards_directory.name != "shards":
        raise StrictRoutingError("dataset shards directory must end in /shards")
    materialized_root = shards_directory.parent
    receipts_directory = _plain_directory(
        materialized_root / "receipts", context="sample receipts directory"
    )

    seen_iids: set[str] = set()
    ordered_iids: list[str] = []
    routes: list[dict[str, Any]] = []
    receipt_bindings: list[dict[str, Any]] = []
    indexed_shards: set[Path] = set()
    indexed_receipts: set[Path] = set()
    strict_count = 0
    for line_number, row in enumerate(index_rows, 1):
        iid = row.get("iid")
        if (
            row.get("schema_version") != INDEX_ROW_SCHEMA
            or type(iid) is not str
            or not iid
            or "\x00" in iid
            or "/" in iid
            or iid in seen_iids
        ):
            raise StrictRoutingError(
                f"dataset index schema/IID differs at line {line_number}"
            )
        seen_iids.add(iid)
        ordered_iids.append(iid)
        if row.get("preview_only") is not True or row.get(
            "production_claim_forbidden"
        ) is not True:
            raise StrictRoutingError(f"dataset index safety state differs: {iid}")

        expected_shard = shards_directory / f"{iid}.parquet"
        shard_path = _path_from_field(
            row.get("parquet_path"), context=f"indexed shard {iid}", kind="file"
        )
        if shard_path != expected_shard:
            raise StrictRoutingError(f"indexed shard path is not dataset-bound: {iid}")
        indexed_shards.add(shard_path)
        _required_sha256(row.get("parquet_sha256"), context=f"shard hash for {iid}")
        _required_sha256(
            row.get("materialized_row_digest"),
            context=f"materialized row digest for {iid}",
        )

        expected_receipt = receipts_directory / f"{iid}.json"
        receipt_path = _path_from_field(
            row.get("sample_receipt_path"),
            context=f"sample receipt {iid}",
            kind="file",
        )
        if receipt_path != expected_receipt:
            raise StrictRoutingError(
                f"sample receipt path is not dataset-bound: {iid}"
            )
        indexed_receipts.add(receipt_path)
        receipt_payload = _read_plain_bytes(
            receipt_path, context=f"sample receipt {iid}"
        )
        receipt, strict, receipt_digest = _validate_sample_receipt(
            row=row,
            iid=iid,
            receipt_payload=receipt_payload,
            inclusion_policy=str(summary["experimental_inclusion_policy"]),
        )
        receipt_shard = _path_from_field(
            receipt.get("parquet_path"),
            context=f"sample receipt shard {iid}",
            kind="file",
        )
        if receipt_shard != shard_path:
            raise StrictRoutingError(f"sample receipt shard path differs: {iid}")

        strict_count += int(strict)
        tier = "motion_only" if strict else "reject"
        review = (
            "sample_receipt.strict_selection_gates_all_true=true"
            if strict
            else "sample_receipt.strict_selection_gates_all_true=false"
        )
        routes.append(
            {
                "schema_version": ROUTING_SCHEMA,
                "iid": iid,
                "tier": tier,
                "full_target_weight": 0.0,
                "review": review,
            }
        )
        receipt_bindings.append(
            {
                "iid": iid,
                "path": str(receipt_path),
                "file_sha256": bytes_sha256(receipt_payload),
                "receipt_digest": receipt_digest,
                "strict_selection_gates_all_true": strict,
            }
        )

    if ordered_iids != sorted(ordered_iids):
        raise StrictRoutingError("dataset index IIDs are not in canonical sorted order")
    actual_shards = {
        _plain_file(path, context="dataset shard membership")
        for path in shards_directory.glob("*.parquet")
    }
    if actual_shards != indexed_shards:
        raise StrictRoutingError("dataset shard/index membership differs")
    actual_receipts = {
        _plain_file(path, context="sample receipt membership")
        for path in receipts_directory.glob("*.json")
    }
    if actual_receipts != indexed_receipts:
        raise StrictRoutingError("sample receipt/index membership differs")
    non_strict_count = len(routes) - strict_count
    if (
        len(routes) != EXPECTED_ROWS
        or strict_count != EXPECTED_STRICT_ROWS
        or non_strict_count != EXPECTED_NON_STRICT_ROWS
    ):
        raise StrictRoutingError(
            "sample receipt cohort counts differ: "
            f"total={len(routes)} strict={strict_count} non_strict={non_strict_count}"
        )

    output_input = _absolute_path(output_jsonl, context="output JSONL")
    output_input.parent.mkdir(parents=True, exist_ok=True)
    output_parent = _plain_directory(output_input.parent, context="output directory")
    output = output_parent / output_input.name
    receipt_output = Path(f"{output}.receipt.json")
    hash_output = Path(f"{output}.sha256")
    if len({output, receipt_output, hash_output}) != 3:
        raise StrictRoutingError("output and sidecar paths must differ")
    for destination in (output, receipt_output, hash_output):
        if destination.exists() or destination.is_symlink():
            raise StrictRoutingError(f"create-only output exists: {destination}")

    routing_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in routes)
    routing_sha = bytes_sha256(routing_payload)
    source_receipts_digest = object_sha256(receipt_bindings)
    cohort_assignment_digest = object_sha256(
        [
            {
                "iid": binding["iid"],
                "strict_selection_gates_all_true": binding[
                    "strict_selection_gates_all_true"
                ],
            }
            for binding in receipt_bindings
        ]
    )
    result: dict[str, Any] = {
        "schema_version": ROUTING_RECEIPT_SCHEMA,
        "complete": True,
        "routing_schema_version": ROUTING_SCHEMA,
        "routing_policy": {
            "strict_selection_gates_all_true": "motion_only",
            "strict_selection_gates_not_all_true": "reject",
            "full_target_weight": 0.0,
            "required_unreviewed_tier": "reject",
        },
        "dataset_summary_path": str(summary_path),
        "dataset_summary_sha256": summary_sha,
        "dataset_summary_digest": summary["summary_digest"],
        "dataset_index_path": str(index_path),
        "dataset_index_sha256": index_sha,
        "dataset_shards_directory": str(shards_directory),
        "sample_receipts_directory": str(receipts_directory),
        "sample_receipts_binding_digest": source_receipts_digest,
        "cohort_assignment_digest": cohort_assignment_digest,
        "route_count": len(routes),
        "strict_motion_only_count": strict_count,
        "non_strict_reject_count": non_strict_count,
        "full_pair_count": 0,
        "routing_jsonl_path": str(output),
        "routing_jsonl_sha256": routing_sha,
        "routing_jsonl_bytes": len(routing_payload),
        "routing_jsonl_lines": len(routes),
        "routing_rows_digest": object_sha256(routes),
        "review_router_digest_required_default_reject": _review_router_digest(
            routes, default_tier="reject"
        ),
        "review_router_digest_default_motion_only": _review_router_digest(
            routes, default_tier="motion_only"
        ),
        "sha256_sidecar_path": str(hash_output),
        "receipt_path": str(receipt_output),
        "publication_contract": "create_only_staged_receipt_ready_marker_last",
    }
    result["receipt_digest"] = object_sha256(result)
    receipt_payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    receipt_sha = bytes_sha256(receipt_payload)
    hash_payload = (
        f"{routing_sha}  {output.name}\n"
        f"{receipt_sha}  {receipt_output.name}\n"
    ).encode("ascii")
    _publish_create_only(
        output=output,
        routing_payload=routing_payload,
        receipt_payload=receipt_payload,
        hash_payload=hash_payload,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument(
        "--expected-dataset-summary-sha256",
        required=True,
        help="caller-pinned SHA-256 of the finalized dataset summary",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_strict_routing(
            dataset_summary=args.dataset_summary,
            expected_dataset_summary_sha256=args.expected_dataset_summary_sha256,
            output_jsonl=args.output_jsonl,
        )
    except StrictRoutingError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
