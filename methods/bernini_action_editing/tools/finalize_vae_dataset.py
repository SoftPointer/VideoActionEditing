#!/usr/bin/env python3
"""Verify every Bernini VAE shard and publish a frozen training index."""

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


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_renderer_dataset as raw_builder  # noqa: E402
from tools import materialize_vae  # noqa: E402


SUMMARY_FORMAT = "bernini-r-action-vae-dataset-summary-v2"
INDEX_ROW_FORMAT = "bernini-r-action-vae-index-row-v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FinalizationError(RuntimeError):
    """Fail-closed dataset finalization error."""


def _plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise FinalizationError(f"missing {context}: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise FinalizationError(f"{context} is not a plain file: {path}")
    return path


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    _plain_file(path, context=context)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizationError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"{context} must contain one object")
    return value


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FinalizationError(f"create-only output exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_sample_receipt(receipt_path: Path, iid: str) -> dict[str, Any]:
    receipt = _load_json(receipt_path, context=f"sample receipt {iid}")
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    if (
        receipt.get("schema_version") != materialize_vae.SAMPLE_RECEIPT_FORMAT
        or receipt.get("complete") is not True
        or receipt.get("iid") != iid
        or materialize_vae.object_sha256(candidate) != digest
    ):
        raise FinalizationError(f"sample receipt contract differs: {iid}")
    if (
        receipt.get("frame_count") != materialize_vae.FRAME_COUNT
        or float(receipt.get("fps", -1)) != materialize_vae.FPS
        or receipt.get("latent_frame_count") != materialize_vae.LATENT_FRAME_COUNT
        or receipt.get("shared_i0_exact") is not True
        or receipt.get("preview_only") is not True
        or receipt.get("training_authorized") is not False
        or receipt.get("training_use_forbidden") is not True
        or receipt.get("production_claim_forbidden") is not True
    ):
        raise FinalizationError(f"sample safety/temporal contract differs: {iid}")
    if (
        receipt.get("experimental_inclusion_policy")
        not in {
            raw_builder.STRICT_INCLUSION_POLICY,
            raw_builder.NATURAL_RELEASE_INCLUSION_POLICY,
        }
        or type(receipt.get("strict_selection_gates_all_true")) is not bool
        or not isinstance(receipt.get("selection_gates_json"), str)
    ):
        raise FinalizationError(f"sample inclusion contract differs: {iid}")
    shard = _plain_file(
        Path(str(receipt.get("parquet_path"))).expanduser().resolve(strict=True),
        context=f"sample shard {iid}",
    )
    expected_sha = receipt.get("parquet_sha256")
    if not isinstance(expected_sha, str) or _SHA256_RE.fullmatch(expected_sha) is None:
        raise FinalizationError(f"invalid sample parquet hash: {iid}")
    if materialize_vae.file_sha256(shard) != expected_sha:
        raise FinalizationError(f"sample parquet hash mismatch: {iid}")
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(shard, columns=["iid", "schema_version"])
        persisted = table.to_pylist()
    except Exception as error:
        raise FinalizationError(f"cannot read sample shard {iid}: {error}") from error
    if (
        len(persisted) != 1
        or persisted[0].get("iid") != iid
        or persisted[0].get("schema_version") != materialize_vae.MATERIALIZED_ROW_FORMAT
    ):
        raise FinalizationError(f"sample shard identity differs: {iid}")
    return receipt


def finalize(
    *,
    raw_receipt_path: Path,
    raw_job_done_path: Path,
    materialized_root: Path,
    output_index: Path,
    output_summary: Path,
    allow_missing: bool = False,
) -> dict[str, Any]:
    raw = _load_json(raw_receipt_path.expanduser().resolve(strict=True), context="raw receipt")
    raw_candidate = dict(raw)
    raw_digest = raw_candidate.pop("receipt_digest", None)
    if (
        raw.get("schema_version") != raw_builder.RECEIPT_FORMAT
        or raw_builder.object_sha256(raw_candidate) != raw_digest
        or raw.get("preview_only") is not True
        or raw.get("training_authorized") is not False
        or raw.get("training_use_forbidden") is not True
        or raw.get("production_claim_forbidden") is not True
    ):
        raise FinalizationError("raw dataset receipt contract differs")
    try:
        materialize_vae._validate_raw_job_done(
            Path(str(raw.get("parquet_path"))),
            raw_receipt_path,
            raw_job_done_path,
        )
    except materialize_vae.VaeMaterializationError as error:
        raise FinalizationError(f"raw job-done validation failed: {error}") from error
    expected = raw.get("sample_ids")
    if not isinstance(expected, list) or not expected or any(type(x) is not str for x in expected):
        raise FinalizationError("raw receipt sample IDs differ")
    if len(set(expected)) != len(expected):
        raise FinalizationError("raw receipt contains duplicate sample IDs")
    raw_policy = raw.get("experimental_inclusion_policy")
    raw_strict_rows = raw.get("strict_selection_rows")
    raw_non_strict_rows = raw.get("non_strict_selection_rows")
    if (
        raw_policy
        not in {
            raw_builder.STRICT_INCLUSION_POLICY,
            raw_builder.NATURAL_RELEASE_INCLUSION_POLICY,
        }
        or type(raw_strict_rows) is not int
        or type(raw_non_strict_rows) is not int
        or raw_strict_rows < 0
        or raw_non_strict_rows < 0
        or raw_strict_rows + raw_non_strict_rows != len(expected)
    ):
        raise FinalizationError("raw receipt inclusion cohort differs")
    root = materialized_root.expanduser().resolve(strict=True)
    index_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    vae_identities: set[str] = set()
    bucket_counts: dict[str, int] = {}
    strict_materialized_rows = 0
    inclusion_policies: set[str] = set()
    for iid in sorted(expected):
        receipt_path = root / "receipts" / f"{iid}.json"
        if not receipt_path.exists():
            missing.append(iid)
            continue
        receipt = _verify_sample_receipt(receipt_path, iid)
        inclusion_policies.add(str(receipt["experimental_inclusion_policy"]))
        strict_materialized_rows += int(
            receipt["strict_selection_gates_all_true"]
        )
        vae_identity_digest = raw_builder.object_sha256(receipt["vae_identity"])
        vae_identities.add(vae_identity_digest)
        bucket_key = "x".join(str(x) for x in receipt["bucket_hw"])
        bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
        index_rows.append(
            {
                "schema_version": INDEX_ROW_FORMAT,
                "iid": iid,
                "parquet_path": receipt["parquet_path"],
                "parquet_sha256": receipt["parquet_sha256"],
                "materialized_row_digest": receipt["materialized_row_digest"],
                "bucket_hw": receipt["bucket_hw"],
                "posterior_parameters_shape": receipt["posterior_parameters_shape"],
                "sample_receipt_path": str(receipt_path),
                "sample_receipt_sha256": materialize_vae.file_sha256(receipt_path),
                "preview_only": True,
                "production_claim_forbidden": True,
            }
        )
    if missing and not allow_missing:
        raise FinalizationError(
            f"materialized dataset is incomplete: missing {len(missing)} rows; "
            f"first={missing[:8]}"
        )
    if not index_rows:
        raise FinalizationError("no materialized samples are available")
    if len(vae_identities) != 1:
        raise FinalizationError("sample receipts bind different VAE checkpoints")
    if inclusion_policies != {raw_policy}:
        raise FinalizationError("sample receipts bind a different inclusion policy")
    if not missing and (
        strict_materialized_rows != raw_strict_rows
        or len(index_rows) - strict_materialized_rows != raw_non_strict_rows
    ):
        raise FinalizationError("materialized cohort counts differ from raw receipt")
    index_payload = b"".join(
        raw_builder.canonical_json_bytes(row) + b"\n" for row in index_rows
    )
    index_sha = hashlib.sha256(index_payload).hexdigest()
    summary = {
        "schema_version": SUMMARY_FORMAT,
        "complete": len(missing) == 0,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "experimental_inclusion_policy": raw_policy,
        "raw_strict_selection_rows": raw_strict_rows,
        "raw_non_strict_selection_rows": raw_non_strict_rows,
        "materialized_strict_selection_rows": strict_materialized_rows,
        "materialized_non_strict_selection_rows": (
            len(index_rows) - strict_materialized_rows
        ),
        "raw_receipt_path": str(raw_receipt_path.expanduser().resolve(strict=True)),
        "raw_receipt_sha256": materialize_vae.file_sha256(raw_receipt_path),
        "raw_job_done_path": str(raw_job_done_path.expanduser().resolve(strict=True)),
        "raw_job_done_sha256": materialize_vae.file_sha256(raw_job_done_path),
        "expected_sample_count": len(expected),
        "materialized_sample_count": len(index_rows),
        "missing_sample_count": len(missing),
        "missing_sample_ids": missing,
        "frame_count": materialize_vae.FRAME_COUNT,
        "fps": materialize_vae.FPS,
        "latent_frame_count": materialize_vae.LATENT_FRAME_COUNT,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "vae_identity_digest": next(iter(vae_identities)),
        "shards_directory": str(root / "shards"),
        "index_path": str(output_index.expanduser().absolute()),
        "index_sha256": index_sha,
    }
    summary["summary_digest"] = raw_builder.object_sha256(summary)
    _atomic_create(output_index.expanduser().absolute(), index_payload)
    try:
        _atomic_create(
            output_summary.expanduser().absolute(),
            json.dumps(
                summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            ).encode("utf-8")
            + b"\n",
        )
    except Exception:
        output_index.expanduser().absolute().unlink(missing_ok=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--raw-job-done", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        finalize(
            raw_receipt_path=args.raw_receipt,
            raw_job_done_path=args.raw_job_done,
            materialized_root=args.materialized_root,
            output_index=args.output_index,
            output_summary=args.output_summary,
            allow_missing=args.allow_missing,
        )
    except FinalizationError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
