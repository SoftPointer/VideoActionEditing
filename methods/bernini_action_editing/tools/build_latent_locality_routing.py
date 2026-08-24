#!/usr/bin/env python3
"""Audit Q0 source/target latent residual locality for the frozen strict cohort.

This is deliberately an *offline data-curation audit*, not an inference
condition and not a training authorization.  It reads posterior parameters
already present in the finalized parquet release, takes the deterministic
posterior mode, applies the checkpoint's Wan latent normalization, and forms

    Q0(T - S)[:, t] = (T - S)[:, t] - (T - S)[:, 0].

The tool publishes per-IID metrics and a caller-specified threshold sweep.
Each candidate subset has a non-routing schema, a receipt, and hash sidecar.
No threshold is selected automatically; no mask, track, pose, box, or
target-derived inference input is produced.

Inputs are fail-closed and provenance bound: the caller pins both the dataset
summary and strict routing hashes; the complete index, sample receipts,
parquet shards, latent blobs, VAE identity/config, and routing receipt are
validated before the create-only audit directory is published.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_lora as legacy  # noqa: E402
from tools import build_strict_routing as strict  # noqa: E402


METRIC_SCHEMA = "bernini-q0-latent-locality-metric-v1"
CANDIDATE_ROW_SCHEMA = "bernini-q0-locality-candidate-iid-v1"
CANDIDATE_RECEIPT_SCHEMA = "bernini-q0-locality-candidate-receipt-v1"
SWEEP_ROW_SCHEMA = "bernini-q0-locality-threshold-sweep-row-v1"
AUDIT_RECEIPT_SCHEMA = "bernini-q0-locality-audit-receipt-v1"
MATERIALIZED_ROW_SCHEMA = "bernini-r-action-vae-row-v2"
Q0_DEFINITION = "q0[:,t,h,w]=(target-source)[:,t,h,w]-(target-source)[:,0,h,w]"
NUMERIC_PROGRAM = (
    "normalized_posterior_mode_float64_cpu_q0; "
    "cell_amplitude=sqrt(mean_channel(q0^2)); strict_greater_than_threshold"
)
EXPECTED_POSTERIOR_CHANNELS = 32
EXPECTED_LATENT_CHANNELS = 16
_ENERGY_LEVELS = (0.5, 0.8, 0.9)
_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)


class LatentLocalityError(RuntimeError):
    """A source, metric, or create-only publication contract failed."""


@dataclass(frozen=True)
class VaeStatistics:
    identity: Mapping[str, Any]
    identity_digest: str
    checkpoint_root: Path
    config_path: Path
    config_sha256: str
    means: tuple[float, ...]
    stds: tuple[float, ...]
    latent_channels: int


@dataclass(frozen=True)
class SourceRow:
    iid: str
    shard_path: Path
    shard_sha256: str
    sample_receipt_digest: str
    materialized_row_digest: str
    source_blob_sha256: str
    target_blob_sha256: str
    posterior_parameters_shape: tuple[int, ...]
    strict: bool


@dataclass(frozen=True)
class SourceBundle:
    summary_path: Path
    summary_sha256: str
    summary_digest: str
    index_path: Path
    index_sha256: str
    shards_directory: Path
    strict_routing_path: Path
    strict_routing_sha256: str
    strict_routing_rows_digest: str
    rows: tuple[SourceRow, ...]
    vae: VaeStatistics


def _raise_strict(error: Exception) -> LatentLocalityError:
    return LatentLocalityError(str(error))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return strict.canonical_json_bytes(value)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _object_sha256(value: Any) -> str:
    return _sha256(_canonical(value))


def _json_payload(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as error:
        raise LatentLocalityError(f"value is not finite JSON: {error}") from error


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(dict(row)) + b"\n" for row in rows)


def _required_sha256(value: Any, *, context: str) -> str:
    try:
        return strict._required_sha256(value, context=context)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _absolute_path(value: str | Path, *, context: str) -> Path:
    try:
        return strict._absolute_path(value, context=context)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _plain_file(path: Path, *, context: str) -> Path:
    try:
        return strict._plain_file(path, context=context)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _plain_directory(path: Path, *, context: str) -> Path:
    try:
        return strict._plain_directory(path, context=context)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _read_plain_bytes(path: Path, *, context: str) -> bytes:
    try:
        return strict._read_plain_bytes(path, context=context)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _path_from_field(value: Any, *, context: str, kind: str) -> Path:
    try:
        return strict._path_from_field(value, context=context, kind=kind)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _decode_json_object(payload: bytes | str, *, context: str) -> dict[str, Any]:
    try:
        return strict._decode_json_object(payload, context=context)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error


def _stable_float(value: float, *, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise LatentLocalityError(f"{context} is not finite")
    # A declared 12-significant-digit reporting grid avoids meaningless JSON
    # drift while selection itself remains integer-count based.
    return float(format(result, ".12g"))


def _validated_axis(
    values: Sequence[float],
    *,
    context: str,
    minimum: float,
    maximum: Optional[float],
) -> tuple[float, ...]:
    if not values:
        raise LatentLocalityError(f"{context} must contain at least one value")
    result: list[float] = []
    for ordinal, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LatentLocalityError(f"{context}[{ordinal}] is not numeric")
        number = float(value)
        if not math.isfinite(number) or number < minimum:
            raise LatentLocalityError(
                f"{context}[{ordinal}] must be finite and >= {minimum}"
            )
        if maximum is not None and number > maximum:
            raise LatentLocalityError(
                f"{context}[{ordinal}] must be <= {maximum}"
            )
        result.append(number)
    ordered = tuple(sorted(result))
    if len(set(ordered)) != len(ordered):
        raise LatentLocalityError(f"{context} contains duplicate values")
    return ordered


def _decode_jsonl(payload: bytes, *, context: str) -> list[dict[str, Any]]:
    if not payload.endswith(b"\n"):
        raise LatentLocalityError(f"{context} must end with one newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LatentLocalityError(f"{context} is not UTF-8: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise LatentLocalityError(
                f"blank {context} row at line {line_number}"
            )
        rows.append(
            _decode_json_object(line, context=f"{context} row {line_number}")
        )
    return rows


def _load_vae_statistics(
    identities: Sequence[Mapping[str, Any]], *, expected_digest: str
) -> VaeStatistics:
    if not identities:
        raise LatentLocalityError("sample receipts contain no VAE identity")
    first = dict(identities[0])
    if any(dict(identity) != first for identity in identities[1:]):
        raise LatentLocalityError("sample receipts bind different VAE identities")
    digest = _object_sha256(first)
    if digest != expected_digest:
        raise LatentLocalityError("dataset summary VAE identity digest differs")
    required_keys = {"checkpoint_root", "vae_config_sha256"}
    allowed_keys = required_keys | {"hf_revision"}
    if set(first) - allowed_keys or not required_keys <= set(first):
        raise LatentLocalityError("sample receipt VAE identity fields differ")
    checkpoint = _path_from_field(
        first.get("checkpoint_root"), context="VAE checkpoint root", kind="directory"
    )
    config_path = _plain_file(
        checkpoint / "vae" / "config.json", context="VAE config"
    )
    config_payload = _read_plain_bytes(config_path, context="VAE config")
    config_sha = _required_sha256(
        first.get("vae_config_sha256"), context="VAE config hash"
    )
    if _sha256(config_payload) != config_sha:
        raise LatentLocalityError("VAE config file hash differs from sample receipts")

    revision_path = checkpoint / ".hf_revision"
    if revision_path.exists() or revision_path.is_symlink():
        revision = _read_plain_bytes(revision_path, context="checkpoint revision")
        try:
            revision_text = revision.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise LatentLocalityError("checkpoint revision is not UTF-8") from error
        if first.get("hf_revision") != revision_text:
            raise LatentLocalityError("checkpoint revision differs from VAE identity")
    elif "hf_revision" in first:
        raise LatentLocalityError("VAE identity names a missing checkpoint revision")

    config = _decode_json_object(config_payload, context="VAE config")
    z_dim = config.get("z_dim")
    means = config.get("latents_mean")
    stds = config.get("latents_std")
    if (
        type(z_dim) is not int
        or z_dim != EXPECTED_LATENT_CHANNELS
        or not isinstance(means, list)
        or not isinstance(stds, list)
        or len(means) != z_dim
        or len(stds) != z_dim
    ):
        raise LatentLocalityError("unexpected Wan VAE latent statistics")
    parsed_means: list[float] = []
    parsed_stds: list[float] = []
    for channel, (mean, std) in enumerate(zip(means, stds)):
        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or isinstance(std, bool)
            or not isinstance(std, (int, float))
            or not math.isfinite(float(mean))
            or not math.isfinite(float(std))
            or float(std) <= 0.0
        ):
            raise LatentLocalityError(
                f"invalid VAE normalization statistics at channel {channel}"
            )
        parsed_means.append(float(mean))
        parsed_stds.append(float(std))
    return VaeStatistics(
        identity=first,
        identity_digest=digest,
        checkpoint_root=checkpoint,
        config_path=config_path,
        config_sha256=config_sha,
        means=tuple(parsed_means),
        stds=tuple(parsed_stds),
        latent_channels=z_dim,
    )


def _validate_routing_publication(
    *,
    routing_path: Path,
    routing_sha256: str,
    routing_rows: Sequence[Mapping[str, Any]],
    summary_sha256: str,
    index_sha256: str,
) -> None:
    receipt_path = _plain_file(
        Path(f"{routing_path}.receipt.json"), context="strict routing receipt"
    )
    receipt_payload = _read_plain_bytes(
        receipt_path, context="strict routing receipt"
    )
    receipt = _decode_json_object(receipt_payload, context="strict routing receipt")
    candidate = dict(receipt)
    declared_digest = candidate.pop("receipt_digest", None)
    _required_sha256(declared_digest, context="strict routing receipt digest")
    if _object_sha256(candidate) != declared_digest:
        raise LatentLocalityError("strict routing receipt digest differs")
    expected = {
        "schema_version": strict.ROUTING_RECEIPT_SCHEMA,
        "complete": True,
        "dataset_summary_sha256": summary_sha256,
        "dataset_index_sha256": index_sha256,
        "routing_jsonl_sha256": routing_sha256,
        "routing_jsonl_lines": strict.EXPECTED_ROWS,
        "strict_motion_only_count": strict.EXPECTED_STRICT_ROWS,
        "non_strict_reject_count": strict.EXPECTED_NON_STRICT_ROWS,
        "full_pair_count": 0,
        "routing_rows_digest": _object_sha256(list(routing_rows)),
        "publication_contract": "create_only_staged_receipt_ready_marker_last",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise LatentLocalityError("strict routing publication receipt differs")
    if receipt.get("routing_jsonl_path") != str(routing_path):
        raise LatentLocalityError("strict routing receipt path differs")

    sidecar_path = _plain_file(
        Path(f"{routing_path}.sha256"), context="strict routing hash sidecar"
    )
    sidecar_payload = _read_plain_bytes(
        sidecar_path, context="strict routing hash sidecar"
    )
    expected_sidecar = (
        f"{routing_sha256}  {routing_path.name}\n"
        f"{_sha256(receipt_payload)}  {receipt_path.name}\n"
    ).encode("ascii")
    if sidecar_payload != expected_sidecar:
        raise LatentLocalityError("strict routing hash sidecar differs")


def _load_source_bundle(
    *,
    dataset_summary: Path,
    expected_dataset_summary_sha256: str,
    preprocessed_parquet_dir: Path,
    strict_routing_jsonl: Path,
    expected_strict_routing_sha256: str,
) -> SourceBundle:
    expected_summary_sha = _required_sha256(
        expected_dataset_summary_sha256,
        context="expected dataset summary hash",
    )
    summary_path = _plain_file(
        _absolute_path(dataset_summary, context="dataset summary"),
        context="dataset summary",
    )
    summary_payload = _read_plain_bytes(summary_path, context="dataset summary")
    summary_sha = _sha256(summary_payload)
    if summary_sha != expected_summary_sha:
        raise LatentLocalityError(
            "dataset summary file hash differs from the caller-pinned hash"
        )
    summary = _decode_json_object(summary_payload, context="dataset summary")
    try:
        strict._validate_summary(summary)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error

    index_path = _path_from_field(
        summary.get("index_path"), context="dataset index", kind="file"
    )
    index_payload = _read_plain_bytes(index_path, context="dataset index")
    index_sha = _sha256(index_payload)
    if index_sha != summary.get("index_sha256"):
        raise LatentLocalityError("dataset index file hash differs from summary")
    try:
        index_rows = strict._decode_index(index_payload)
    except strict.StrictRoutingError as error:
        raise _raise_strict(error) from error

    declared_shards = _path_from_field(
        summary.get("shards_directory"),
        context="dataset shards directory",
        kind="directory",
    )
    supplied_shards = _plain_directory(
        _absolute_path(
            preprocessed_parquet_dir, context="preprocessed parquet directory"
        ),
        context="preprocessed parquet directory",
    )
    if supplied_shards != declared_shards or declared_shards.name != "shards":
        raise LatentLocalityError(
            "preprocessed parquet directory differs from finalized summary"
        )
    receipts_directory = _plain_directory(
        declared_shards.parent / "receipts", context="sample receipts directory"
    )

    source_rows: list[SourceRow] = []
    expected_routes: list[dict[str, Any]] = []
    identities: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    indexed_shards: set[Path] = set()
    indexed_receipts: set[Path] = set()
    for line_number, row in enumerate(index_rows, 1):
        iid = row.get("iid")
        if (
            row.get("schema_version") != strict.INDEX_ROW_SCHEMA
            or type(iid) is not str
            or not iid
            or "\x00" in iid
            or "/" in iid
            or iid in seen
        ):
            raise LatentLocalityError(
                f"dataset index schema/IID differs at line {line_number}"
            )
        seen.add(iid)
        if row.get("preview_only") is not True or row.get(
            "production_claim_forbidden"
        ) is not True:
            raise LatentLocalityError(f"dataset index safety state differs: {iid}")

        shard_path = _path_from_field(
            row.get("parquet_path"), context=f"indexed shard {iid}", kind="file"
        )
        if shard_path != declared_shards / f"{iid}.parquet":
            raise LatentLocalityError(f"indexed shard is not dataset-bound: {iid}")
        indexed_shards.add(shard_path)
        shard_sha = _required_sha256(
            row.get("parquet_sha256"), context=f"shard hash for {iid}"
        )

        receipt_path = _path_from_field(
            row.get("sample_receipt_path"),
            context=f"sample receipt {iid}",
            kind="file",
        )
        if receipt_path != receipts_directory / f"{iid}.json":
            raise LatentLocalityError(
                f"sample receipt path is not dataset-bound: {iid}"
            )
        indexed_receipts.add(receipt_path)
        receipt_payload = _read_plain_bytes(
            receipt_path, context=f"sample receipt {iid}"
        )
        try:
            receipt, is_strict, receipt_digest = strict._validate_sample_receipt(
                row=row,
                iid=iid,
                receipt_payload=receipt_payload,
                inclusion_policy=str(summary["experimental_inclusion_policy"]),
            )
        except strict.StrictRoutingError as error:
            raise _raise_strict(error) from error
        receipt_shard = _path_from_field(
            receipt.get("parquet_path"),
            context=f"sample receipt shard {iid}",
            kind="file",
        )
        if receipt_shard != shard_path:
            raise LatentLocalityError(f"sample receipt shard path differs: {iid}")

        shape_value = receipt.get("posterior_parameters_shape")
        if (
            not isinstance(shape_value, list)
            or len(shape_value) != 5
            or any(type(value) is not int or value <= 0 for value in shape_value)
            or shape_value[0] != 1
            or shape_value[1] != EXPECTED_POSTERIOR_CHANNELS
            or shape_value[2] != strict.EXPECTED_LATENT_FRAME_COUNT
        ):
            raise LatentLocalityError(
                f"sample posterior shape differs from [1,32,21,H,W]: {iid}"
            )
        source_blob_sha = _required_sha256(
            receipt.get("source_latent_blob_sha256"),
            context=f"source latent blob hash for {iid}",
        )
        target_blob_sha = _required_sha256(
            receipt.get("target_latent_blob_sha256"),
            context=f"target latent blob hash for {iid}",
        )
        materialized_digest = _required_sha256(
            receipt.get("materialized_row_digest"),
            context=f"materialized row digest for {iid}",
        )
        identity = receipt.get("vae_identity")
        if not isinstance(identity, dict):
            raise LatentLocalityError(f"sample VAE identity differs: {iid}")
        identities.append(identity)
        source_rows.append(
            SourceRow(
                iid=iid,
                shard_path=shard_path,
                shard_sha256=shard_sha,
                sample_receipt_digest=receipt_digest,
                materialized_row_digest=materialized_digest,
                source_blob_sha256=source_blob_sha,
                target_blob_sha256=target_blob_sha,
                posterior_parameters_shape=tuple(shape_value),
                strict=is_strict,
            )
        )
        expected_routes.append(
            {
                "schema_version": strict.ROUTING_SCHEMA,
                "iid": iid,
                "tier": "motion_only" if is_strict else "reject",
                "full_target_weight": 0.0,
                "review": (
                    "sample_receipt.strict_selection_gates_all_true=true"
                    if is_strict
                    else "sample_receipt.strict_selection_gates_all_true=false"
                ),
            }
        )

    if [row.iid for row in source_rows] != sorted(row.iid for row in source_rows):
        raise LatentLocalityError("dataset index IIDs are not canonically sorted")
    actual_shards = {
        _plain_file(path, context="dataset shard membership")
        for path in declared_shards.glob("*.parquet")
    }
    if actual_shards != indexed_shards:
        raise LatentLocalityError("dataset shard/index membership differs")
    actual_receipts = {
        _plain_file(path, context="sample receipt membership")
        for path in receipts_directory.glob("*.json")
    }
    if actual_receipts != indexed_receipts:
        raise LatentLocalityError("sample receipt/index membership differs")
    strict_count = sum(row.strict for row in source_rows)
    if (
        len(source_rows) != strict.EXPECTED_ROWS
        or strict_count != strict.EXPECTED_STRICT_ROWS
        or len(source_rows) - strict_count != strict.EXPECTED_NON_STRICT_ROWS
    ):
        raise LatentLocalityError("dataset strict cohort counts differ")

    vae = _load_vae_statistics(
        identities, expected_digest=str(summary["vae_identity_digest"])
    )

    expected_routing_sha = _required_sha256(
        expected_strict_routing_sha256, context="expected strict routing hash"
    )
    routing_path = _plain_file(
        _absolute_path(strict_routing_jsonl, context="strict routing JSONL"),
        context="strict routing JSONL",
    )
    routing_payload = _read_plain_bytes(routing_path, context="strict routing JSONL")
    routing_sha = _sha256(routing_payload)
    if routing_sha != expected_routing_sha:
        raise LatentLocalityError(
            "strict routing file hash differs from the caller-pinned hash"
        )
    routing_rows = _decode_jsonl(routing_payload, context="strict routing JSONL")
    if routing_rows != expected_routes:
        raise LatentLocalityError(
            "strict routing differs from finalized sample-receipt cohort"
        )
    _validate_routing_publication(
        routing_path=routing_path,
        routing_sha256=routing_sha,
        routing_rows=routing_rows,
        summary_sha256=summary_sha,
        index_sha256=index_sha,
    )
    return SourceBundle(
        summary_path=summary_path,
        summary_sha256=summary_sha,
        summary_digest=str(summary["summary_digest"]),
        index_path=index_path,
        index_sha256=index_sha,
        shards_directory=declared_shards,
        strict_routing_path=routing_path,
        strict_routing_sha256=routing_sha,
        strict_routing_rows_digest=_object_sha256(routing_rows),
        rows=tuple(source_rows),
        vae=vae,
    )


def _default_parquet_row_loader(payload: bytes) -> Mapping[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise LatentLocalityError("pyarrow is required to read parquet shards") from error
    try:
        table = pq.read_table(pa.BufferReader(payload))
    except Exception as error:
        raise LatentLocalityError(f"cannot decode parquet shard: {error}") from error
    if table.num_rows != 1:
        raise LatentLocalityError(
            f"each finalized parquet shard must contain one row, got {table.num_rows}"
        )
    rows = table.to_pylist()
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise LatentLocalityError("parquet shard did not decode to one object")
    return rows[0]


def _blob_bytes(value: Any, *, context: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise LatentLocalityError(f"{context} must be a serialized tensor blob")
    return bytes(value)


def _validate_materialized_row(
    row: Mapping[str, Any], *, binding: SourceRow, vae: VaeStatistics
) -> tuple[bytes, bytes]:
    if row.get("schema_version") != MATERIALIZED_ROW_SCHEMA:
        raise LatentLocalityError(f"materialized row schema differs: {binding.iid}")
    if row.get("iid") != binding.iid:
        raise LatentLocalityError(f"materialized row IID differs: {binding.iid}")
    blobs = row.get("video_vae_latents")
    if not isinstance(blobs, (list, tuple)) or len(blobs) != 2:
        raise LatentLocalityError(
            f"materialized row must contain source and target latents: {binding.iid}"
        )
    source_blob = _blob_bytes(blobs[0], context=f"source latent {binding.iid}")
    target_blob = _blob_bytes(blobs[1], context=f"target latent {binding.iid}")
    if _sha256(source_blob) != binding.source_blob_sha256:
        raise LatentLocalityError(f"source latent blob hash differs: {binding.iid}")
    if _sha256(target_blob) != binding.target_blob_sha256:
        raise LatentLocalityError(f"target latent blob hash differs: {binding.iid}")

    identity_text = row.get("bernini_vae_identity_json")
    if type(identity_text) is not str:
        raise LatentLocalityError(f"materialized VAE identity is missing: {binding.iid}")
    identity = _decode_json_object(
        identity_text, context=f"materialized VAE identity {binding.iid}"
    )
    if identity != dict(vae.identity):
        raise LatentLocalityError(f"materialized VAE identity differs: {binding.iid}")

    declared_digest = row.get("materialized_row_digest")
    if declared_digest != binding.materialized_row_digest:
        raise LatentLocalityError(
            f"materialized row/receipt digest differs: {binding.iid}"
        )
    digest_input = {
        key: value
        for key, value in row.items()
        if key not in ("video_vae_latents", "materialized_row_digest")
    }
    digest_input["video_vae_latents_sha256"] = [
        binding.source_blob_sha256,
        binding.target_blob_sha256,
    ]
    if _object_sha256(digest_input) != declared_digest:
        raise LatentLocalityError(f"materialized row digest differs: {binding.iid}")
    return source_blob, target_blob


def _default_mode_loader(
    blob: bytes, vae: VaeStatistics, expected_shape: tuple[int, ...]
) -> Any:
    try:
        import torch
    except ImportError as error:
        raise LatentLocalityError("torch is required to decode latent blobs") from error
    try:
        parameters = legacy._load_tensor_blob(blob)
    except Exception as error:
        raise LatentLocalityError(f"cannot load posterior parameters: {error}") from error
    if not isinstance(parameters, torch.Tensor):
        raise LatentLocalityError("latent loader did not return a torch tensor")
    shape = tuple(int(value) for value in parameters.shape)
    if shape != expected_shape:
        raise LatentLocalityError(
            f"posterior parameter shape {shape} differs from receipt {expected_shape}"
        )
    if parameters.dtype != torch.float32 or parameters.device.type != "cpu":
        raise LatentLocalityError("posterior parameters must be CPU float32")
    if parameters.requires_grad or not bool(torch.isfinite(parameters).all().item()):
        raise LatentLocalityError("posterior parameters are non-finite or require grad")
    # DiagonalGaussianDistribution.mode() is its first half (the posterior
    # mean).  Taking it explicitly avoids sampling and keeps this audit free of
    # diffusers runtime/version behavior while still reusing the training
    # tensor-blob loader above.
    mode = parameters[:, : vae.latent_channels].squeeze(0).double()
    mean = torch.tensor(vae.means, dtype=torch.float64).view(-1, 1, 1, 1)
    std = torch.tensor(vae.stds, dtype=torch.float64).view(-1, 1, 1, 1)
    normalized = (mode - mean) / std
    if not bool(torch.isfinite(normalized).all().item()):
        raise LatentLocalityError("normalized posterior mode is non-finite")
    return normalized.contiguous()


def _linear_quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(
        sorted_values[upper]
    ) * weight


def _metric_record(
    *,
    shape: tuple[int, int, int, int],
    raw_squared_sum: float,
    phase0_squared_sum: float,
    q0_squared_sum: float,
    nonboundary_q0_squared_sum: float,
    lag1_squared_sum: float,
    cell_energies: Sequence[float],
    amplitude_thresholds: Sequence[float],
) -> dict[str, Any]:
    channels, phases, height, width = shape
    all_values = channels * phases * height * width
    boundary_values = channels * height * width
    nonboundary_values = channels * (phases - 1) * height * width
    lag1_values = channels * (phases - 1) * height * width
    raw_rms = math.sqrt(raw_squared_sum / all_values)
    q0_rms = math.sqrt(q0_squared_sum / all_values)
    cell_amplitudes = sorted(
        math.sqrt(max(0.0, energy) / channels) for energy in cell_energies
    )
    cell_count = len(cell_amplitudes)
    if cell_count != (phases - 1) * height * width:
        raise LatentLocalityError("Q0 cell count differs from latent geometry")
    coverage: list[dict[str, Any]] = []
    for threshold in amplitude_thresholds:
        count = sum(amplitude > threshold for amplitude in cell_amplitudes)
        coverage.append(
            {
                "amplitude_threshold": threshold,
                "amplitude_threshold_hex": float(threshold).hex(),
                "nonboundary_cell_count_above": count,
                "nonboundary_cell_fraction_above": _stable_float(
                    count / cell_count, context="Q0 coverage"
                ),
            }
        )

    total_energy = math.fsum(cell_energies)
    descending_energy = sorted(cell_energies, reverse=True)
    energy_support: list[dict[str, Any]] = []
    for level in _ENERGY_LEVELS:
        if total_energy <= 0.0:
            support_count = 0
        else:
            target = total_energy * level
            cumulative = 0.0
            support_count = 0
            for value in descending_energy:
                cumulative += value
                support_count += 1
                if cumulative >= target:
                    break
        energy_support.append(
            {
                "energy_fraction": level,
                "minimum_nonboundary_cell_count": support_count,
                "minimum_nonboundary_cell_fraction": _stable_float(
                    support_count / cell_count, context="Q0 energy support"
                ),
            }
        )
    quantiles = {
        f"p{int(level * 100):02d}": _stable_float(
            _linear_quantile(cell_amplitudes, level), context="Q0 amplitude quantile"
        )
        for level in _QUANTILES
    }
    quantiles["max"] = _stable_float(
        cell_amplitudes[-1] if cell_amplitudes else 0.0,
        context="Q0 amplitude maximum",
    )
    return {
        "latent_mode_shape_cthw": list(shape),
        "q0_definition": Q0_DEFINITION,
        "numeric_program": NUMERIC_PROGRAM,
        "raw_delta_rms": _stable_float(raw_rms, context="raw delta RMS"),
        "phase0_delta_rms": _stable_float(
            math.sqrt(phase0_squared_sum / boundary_values),
            context="phase-0 delta RMS",
        ),
        "q0_residual_rms": _stable_float(q0_rms, context="Q0 RMS"),
        "q0_nonboundary_residual_rms": _stable_float(
            math.sqrt(nonboundary_q0_squared_sum / nonboundary_values),
            context="nonboundary Q0 RMS",
        ),
        "q0_to_raw_rms_ratio": (
            _stable_float(q0_rms / raw_rms, context="Q0/raw RMS ratio")
            if raw_rms > 0.0
            else None
        ),
        "q0_lag1_rms": _stable_float(
            math.sqrt(lag1_squared_sum / lag1_values),
            context="Q0 lag-1 RMS",
        ),
        "q0_boundary_max_abs": 0.0,
        "nonboundary_cell_count": cell_count,
        "q0_cell_amplitude_quantiles": quantiles,
        "q0_cell_energy_support": energy_support,
        "amplitude_coverage_sweep": coverage,
    }


def _nested_shape(value: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        raise LatentLocalityError(f"{context} has an empty dimension")
    child_shapes = [_nested_shape(child, context=context) for child in value]
    if any(shape != child_shapes[0] for shape in child_shapes[1:]):
        raise LatentLocalityError(f"{context} is ragged")
    return (len(value),) + child_shapes[0]


def _compute_q0_metrics_python(
    source: Sequence[Any], target: Sequence[Any], amplitude_thresholds: Sequence[float]
) -> dict[str, Any]:
    source_shape = _nested_shape(source, context="source posterior mode")
    target_shape = _nested_shape(target, context="target posterior mode")
    if source_shape != target_shape or len(source_shape) != 4:
        raise LatentLocalityError("source/target posterior mode shapes differ")
    channels, phases, height, width = source_shape
    if channels != EXPECTED_LATENT_CHANNELS or phases != strict.EXPECTED_LATENT_FRAME_COUNT:
        raise LatentLocalityError("posterior mode must be [16,21,H,W]")
    if height <= 0 or width <= 0:
        raise LatentLocalityError("posterior mode spatial shape is empty")

    raw_terms: list[float] = []
    phase0_terms: list[float] = []
    q0_terms: list[float] = []
    nonboundary_terms: list[float] = []
    lag1_terms: list[float] = []
    cell_energies = [0.0] * ((phases - 1) * height * width)
    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                source0 = source[channel][0][y][x]
                target0 = target[channel][0][y][x]
                if (
                    isinstance(source0, bool)
                    or not isinstance(source0, (int, float))
                    or isinstance(target0, bool)
                    or not isinstance(target0, (int, float))
                ):
                    raise LatentLocalityError("posterior mode contains a non-number")
                delta0 = float(target0) - float(source0)
                if not math.isfinite(delta0):
                    raise LatentLocalityError("posterior mode contains a non-finite value")
                phase0_terms.append(delta0 * delta0)
                raw_terms.append(delta0 * delta0)
                q0_terms.append(0.0)
                previous_q0 = 0.0
                for phase in range(1, phases):
                    source_value = source[channel][phase][y][x]
                    target_value = target[channel][phase][y][x]
                    if (
                        isinstance(source_value, bool)
                        or not isinstance(source_value, (int, float))
                        or isinstance(target_value, bool)
                        or not isinstance(target_value, (int, float))
                    ):
                        raise LatentLocalityError("posterior mode contains a non-number")
                    delta = float(target_value) - float(source_value)
                    q0 = delta - delta0
                    if not math.isfinite(delta) or not math.isfinite(q0):
                        raise LatentLocalityError(
                            "posterior mode contains a non-finite value"
                        )
                    raw_terms.append(delta * delta)
                    q0_terms.append(q0 * q0)
                    nonboundary_terms.append(q0 * q0)
                    lag = q0 - previous_q0
                    lag1_terms.append(lag * lag)
                    previous_q0 = q0
                    cell_index = ((phase - 1) * height + y) * width + x
                    cell_energies[cell_index] += q0 * q0
    return _metric_record(
        shape=(channels, phases, height, width),
        raw_squared_sum=math.fsum(raw_terms),
        phase0_squared_sum=math.fsum(phase0_terms),
        q0_squared_sum=math.fsum(q0_terms),
        nonboundary_q0_squared_sum=math.fsum(nonboundary_terms),
        lag1_squared_sum=math.fsum(lag1_terms),
        cell_energies=cell_energies,
        amplitude_thresholds=amplitude_thresholds,
    )


def _compute_q0_metrics_torch(
    source: Any, target: Any, amplitude_thresholds: Sequence[float]
) -> dict[str, Any]:
    import torch

    if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise LatentLocalityError("posterior modes must both be torch tensors")
    shape = tuple(int(value) for value in source.shape)
    if shape != tuple(int(value) for value in target.shape) or len(shape) != 4:
        raise LatentLocalityError("source/target posterior mode shapes differ")
    channels, phases, height, width = shape
    if channels != EXPECTED_LATENT_CHANNELS or phases != strict.EXPECTED_LATENT_FRAME_COUNT:
        raise LatentLocalityError("posterior mode must be [16,21,H,W]")
    if source.device.type != "cpu" or target.device.type != "cpu":
        raise LatentLocalityError("posterior modes must be on CPU")
    source = source.detach().double().contiguous()
    target = target.detach().double().contiguous()
    if not bool(torch.isfinite(source).all().item()) or not bool(
        torch.isfinite(target).all().item()
    ):
        raise LatentLocalityError("posterior mode contains a non-finite value")
    delta = target - source
    q0 = delta - delta[:, :1]
    nonboundary = q0[:, 1:]
    lag1 = q0[:, 1:] - q0[:, :-1]
    cell_energies_tensor = nonboundary.square().sum(dim=0).reshape(-1)
    cell_energies = [float(value) for value in cell_energies_tensor.tolist()]
    return _metric_record(
        shape=(channels, phases, height, width),
        raw_squared_sum=float(delta.square().sum().item()),
        phase0_squared_sum=float(delta[:, 0].square().sum().item()),
        q0_squared_sum=float(q0.square().sum().item()),
        nonboundary_q0_squared_sum=float(nonboundary.square().sum().item()),
        lag1_squared_sum=float(lag1.square().sum().item()),
        cell_energies=cell_energies,
        amplitude_thresholds=amplitude_thresholds,
    )


def compute_q0_metrics(
    source: Any, target: Any, *, amplitude_thresholds: Sequence[float]
) -> dict[str, Any]:
    """Compute the declared Q0 metric program on tensors or small nested lists."""

    thresholds = _validated_axis(
        amplitude_thresholds,
        context="amplitude thresholds",
        minimum=0.0,
        maximum=None,
    )
    if isinstance(source, (list, tuple)) and isinstance(target, (list, tuple)):
        return _compute_q0_metrics_python(source, target, thresholds)
    try:
        return _compute_q0_metrics_torch(source, target, thresholds)
    except ImportError as error:
        raise LatentLocalityError("torch is required for tensor metrics") from error


def _coverage_for_threshold(metric: Mapping[str, Any], threshold: float) -> float:
    rows = metric.get("amplitude_coverage_sweep")
    if not isinstance(rows, list):
        raise LatentLocalityError("metric coverage sweep is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("amplitude_threshold_hex") == float(threshold).hex()
    ]
    if len(matches) != 1:
        raise LatentLocalityError("metric threshold coverage is ambiguous")
    value = matches[0].get("nonboundary_cell_fraction_above")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LatentLocalityError("metric threshold coverage is not numeric")
    return float(value)


def _write_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory_create_only(
    *, output_dir: Path, payloads: Mapping[str, bytes], ready_marker: str
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise LatentLocalityError(f"create-only output exists: {output_dir}")
    parent = output_dir.parent
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging.", dir=parent
    ) as temporary_name:
        staging = Path(temporary_name)
        for relative, payload in payloads.items():
            _write_fsynced(staging / relative, payload)
        try:
            output_dir.mkdir(mode=0o755)
        except FileExistsError as error:
            raise LatentLocalityError(
                f"create-only output appeared during publication: {output_dir}"
            ) from error
        published: list[Path] = []
        try:
            relative_paths = sorted(path for path in payloads if path != ready_marker)
            relative_paths.append(ready_marker)
            for relative in relative_paths:
                source = staging / relative
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except FileExistsError as error:
                    raise LatentLocalityError(
                        f"create-only output appeared during publication: {destination}"
                    ) from error
                published.append(destination)
            for directory in (output_dir / "candidates", output_dir, parent):
                if not directory.exists():
                    continue
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except Exception:
            # The directory was atomically created by this process.  Remove
            # only its tracked files; refuse to delete any unexpected entrant.
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            candidate_dir = output_dir / "candidates"
            try:
                if candidate_dir.exists() and not any(candidate_dir.iterdir()):
                    candidate_dir.rmdir()
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
            except OSError:
                pass
            raise


def _audit_summary(
    metric_rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]
) -> dict[str, Any]:
    q0_values = sorted(float(row["q0_residual_rms"]) for row in metric_rows)
    coverage_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        values = sorted(_coverage_for_threshold(row, threshold) for row in metric_rows)
        coverage_rows.append(
            {
                "amplitude_threshold": threshold,
                "amplitude_threshold_hex": float(threshold).hex(),
                "mean_nonboundary_cell_fraction_above": _stable_float(
                    math.fsum(values) / len(values), context="mean Q0 coverage"
                ),
                "median_nonboundary_cell_fraction_above": _stable_float(
                    _linear_quantile(values, 0.5), context="median Q0 coverage"
                ),
                "minimum_nonboundary_cell_fraction_above": values[0],
                "maximum_nonboundary_cell_fraction_above": values[-1],
            }
        )
    return {
        "strict_iid_count": len(metric_rows),
        "q0_residual_rms_mean": _stable_float(
            math.fsum(q0_values) / len(q0_values), context="mean Q0 RMS"
        ),
        "q0_residual_rms_median": _stable_float(
            _linear_quantile(q0_values, 0.5), context="median Q0 RMS"
        ),
        "amplitude_coverage_summary": coverage_rows,
    }


def build_latent_locality_audit(
    *,
    dataset_summary: Path,
    expected_dataset_summary_sha256: str,
    preprocessed_parquet_dir: Path,
    strict_routing_jsonl: Path,
    expected_strict_routing_sha256: str,
    amplitude_thresholds: Sequence[float],
    coverage_caps: Sequence[float],
    output_dir: Path,
    _row_loader: Optional[Callable[[bytes], Mapping[str, Any]]] = None,
    _mode_loader: Optional[
        Callable[[bytes, VaeStatistics, tuple[int, ...]], Any]
    ] = None,
) -> dict[str, Any]:
    """Validate sources and publish an audit-only deterministic threshold sweep."""

    thresholds = _validated_axis(
        amplitude_thresholds,
        context="amplitude thresholds",
        minimum=0.0,
        maximum=None,
    )
    caps = _validated_axis(
        coverage_caps,
        context="coverage caps",
        minimum=0.0,
        maximum=1.0,
    )
    output_input = _absolute_path(output_dir, context="output audit directory")
    output_input.parent.mkdir(parents=True, exist_ok=True)
    output_parent = _plain_directory(output_input.parent, context="output parent")
    output = output_parent / output_input.name
    if output.exists() or output.is_symlink():
        raise LatentLocalityError(f"create-only output exists: {output}")

    source = _load_source_bundle(
        dataset_summary=dataset_summary,
        expected_dataset_summary_sha256=expected_dataset_summary_sha256,
        preprocessed_parquet_dir=preprocessed_parquet_dir,
        strict_routing_jsonl=strict_routing_jsonl,
        expected_strict_routing_sha256=expected_strict_routing_sha256,
    )
    row_loader = _row_loader or _default_parquet_row_loader
    mode_loader = _mode_loader or _default_mode_loader
    metric_rows: list[dict[str, Any]] = []
    for binding in source.rows:
        if not binding.strict:
            continue
        shard_payload = _read_plain_bytes(
            binding.shard_path, context=f"parquet shard {binding.iid}"
        )
        if _sha256(shard_payload) != binding.shard_sha256:
            raise LatentLocalityError(f"parquet shard hash differs: {binding.iid}")
        try:
            materialized_row = row_loader(shard_payload)
        except LatentLocalityError:
            raise
        except Exception as error:
            raise LatentLocalityError(
                f"cannot read parquet row {binding.iid}: {error}"
            ) from error
        if not isinstance(materialized_row, Mapping):
            raise LatentLocalityError(f"parquet row is not an object: {binding.iid}")
        source_blob, target_blob = _validate_materialized_row(
            materialized_row, binding=binding, vae=source.vae
        )
        try:
            source_mode = mode_loader(
                source_blob, source.vae, binding.posterior_parameters_shape
            )
            target_mode = mode_loader(
                target_blob, source.vae, binding.posterior_parameters_shape
            )
        except LatentLocalityError:
            raise
        except Exception as error:
            raise LatentLocalityError(
                f"cannot derive posterior mode for {binding.iid}: {error}"
            ) from error
        metrics = compute_q0_metrics(
            source_mode, target_mode, amplitude_thresholds=thresholds
        )
        metric_rows.append(
            {
                "schema_version": METRIC_SCHEMA,
                "iid": binding.iid,
                "audit_only": True,
                "target_used_only_for_offline_data_curation": True,
                "source_sample_receipt_digest": binding.sample_receipt_digest,
                "parquet_sha256": binding.shard_sha256,
                "source_latent_blob_sha256": binding.source_blob_sha256,
                "target_latent_blob_sha256": binding.target_blob_sha256,
                **metrics,
            }
        )
    if (
        len(metric_rows) != strict.EXPECTED_STRICT_ROWS
        or [row["iid"] for row in metric_rows]
        != sorted(str(row["iid"]) for row in metric_rows)
    ):
        raise LatentLocalityError("strict metric cohort count/order differs")

    payloads: dict[str, bytes] = {}
    metrics_relative = "metrics.jsonl"
    metrics_payload = _jsonl_payload(metric_rows)
    payloads[metrics_relative] = metrics_payload
    metrics_sha = _sha256(metrics_payload)

    candidate_bindings: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    candidate_ordinal = 0
    for threshold in thresholds:
        for cap in caps:
            candidate_id = f"candidate-{candidate_ordinal:04d}"
            candidate_ordinal += 1
            selected_metrics = [
                row
                for row in metric_rows
                if _coverage_for_threshold(row, threshold) <= cap
            ]
            candidate_rows = [
                {
                    "schema_version": CANDIDATE_ROW_SCHEMA,
                    "iid": row["iid"],
                    "source_metric_digest": _object_sha256(row),
                    "audit_only": True,
                }
                for row in selected_metrics
            ]
            candidate_relative = f"candidates/{candidate_id}.jsonl"
            candidate_payload = _jsonl_payload(candidate_rows)
            candidate_sha = _sha256(candidate_payload)
            payloads[candidate_relative] = candidate_payload
            selected_iids = [str(row["iid"]) for row in candidate_rows]
            candidate_receipt: dict[str, Any] = {
                "schema_version": CANDIDATE_RECEIPT_SCHEMA,
                "complete": True,
                "audit_only": True,
                "training_authorized": False,
                "automatic_training_authorization": False,
                "candidate_id": candidate_id,
                "candidate_schema_version": CANDIDATE_ROW_SCHEMA,
                "selection_rule": {
                    "metric": "nonboundary_cell_fraction_strictly_above_amplitude_threshold",
                    "comparison": "metric <= maximum_coverage",
                    "amplitude_threshold": threshold,
                    "amplitude_threshold_hex": float(threshold).hex(),
                    "maximum_coverage": cap,
                    "maximum_coverage_hex": float(cap).hex(),
                },
                "source_dataset_summary_sha256": source.summary_sha256,
                "source_strict_routing_sha256": source.strict_routing_sha256,
                "source_metrics_jsonl_sha256": metrics_sha,
                "strict_input_count": len(metric_rows),
                "selected_count": len(candidate_rows),
                "selected_iids_digest": _object_sha256(selected_iids),
                "candidate_jsonl_path": candidate_relative,
                "candidate_jsonl_sha256": candidate_sha,
                "candidate_jsonl_lines": len(candidate_rows),
                "not_a_routing_artifact": True,
                "selected_threshold_requires_human_review": True,
            }
            candidate_receipt["receipt_digest"] = _object_sha256(candidate_receipt)
            receipt_relative = f"candidates/{candidate_id}.receipt.json"
            receipt_payload = _json_payload(candidate_receipt)
            receipt_sha = _sha256(receipt_payload)
            payloads[receipt_relative] = receipt_payload
            sidecar_relative = f"candidates/{candidate_id}.sha256"
            sidecar_payload = (
                f"{candidate_sha}  {Path(candidate_relative).name}\n"
                f"{receipt_sha}  {Path(receipt_relative).name}\n"
            ).encode("ascii")
            payloads[sidecar_relative] = sidecar_payload
            binding = {
                "candidate_id": candidate_id,
                "amplitude_threshold": threshold,
                "amplitude_threshold_hex": float(threshold).hex(),
                "maximum_coverage": cap,
                "maximum_coverage_hex": float(cap).hex(),
                "selected_count": len(candidate_rows),
                "selected_iids_digest": candidate_receipt["selected_iids_digest"],
                "candidate_jsonl_path": candidate_relative,
                "candidate_jsonl_sha256": candidate_sha,
                "candidate_receipt_path": receipt_relative,
                "candidate_receipt_sha256": receipt_sha,
                "candidate_hash_sidecar_path": sidecar_relative,
                "candidate_hash_sidecar_sha256": _sha256(sidecar_payload),
            }
            candidate_bindings.append(binding)
            sweep_rows.append(
                {
                    "schema_version": SWEEP_ROW_SCHEMA,
                    "audit_only": True,
                    "strict_input_count": len(metric_rows),
                    "selected_fraction": _stable_float(
                        len(candidate_rows) / len(metric_rows),
                        context="candidate selected fraction",
                    ),
                    **binding,
                }
            )

    sweep_relative = "threshold_sweep.jsonl"
    sweep_payload = _jsonl_payload(sweep_rows)
    sweep_sha = _sha256(sweep_payload)
    payloads[sweep_relative] = sweep_payload

    receipt: dict[str, Any] = {
        "schema_version": AUDIT_RECEIPT_SCHEMA,
        "complete": True,
        "audit_only": True,
        "training_authorized": False,
        "automatic_training_authorization": False,
        "selected_candidate": None,
        "target_used_only_for_offline_data_curation": True,
        "inference_conditions_added": False,
        "spatial_mask_generated": False,
        "tracking_generated": False,
        "q0_definition": Q0_DEFINITION,
        "numeric_program": NUMERIC_PROGRAM,
        "source_dataset": {
            "summary_path": str(source.summary_path),
            "summary_sha256": source.summary_sha256,
            "summary_digest": source.summary_digest,
            "index_path": str(source.index_path),
            "index_sha256": source.index_sha256,
            "shards_directory": str(source.shards_directory),
        },
        "source_strict_routing": {
            "path": str(source.strict_routing_path),
            "sha256": source.strict_routing_sha256,
            "rows_digest": source.strict_routing_rows_digest,
            "strict_motion_only_count": len(metric_rows),
        },
        "vae_identity": dict(source.vae.identity),
        "vae_identity_digest": source.vae.identity_digest,
        "vae_config_path": str(source.vae.config_path),
        "vae_config_sha256": source.vae.config_sha256,
        "amplitude_thresholds": list(thresholds),
        "amplitude_thresholds_hex": [float(value).hex() for value in thresholds],
        "coverage_caps": list(caps),
        "coverage_caps_hex": [float(value).hex() for value in caps],
        "per_iid_metrics": {
            "path": metrics_relative,
            "sha256": metrics_sha,
            "lines": len(metric_rows),
            "rows_digest": _object_sha256(metric_rows),
        },
        "threshold_sweep": {
            "path": sweep_relative,
            "sha256": sweep_sha,
            "lines": len(sweep_rows),
            "rows_digest": _object_sha256(sweep_rows),
        },
        "candidate_subsets": candidate_bindings,
        "audit_summary": _audit_summary(metric_rows, thresholds),
        "limitations": [
            "Q0 locality is not an identity, appearance, sharpness, or temporal-consistency acceptance test.",
            "A moving identity/texture error may remain in Q0 and must be evaluated separately.",
            "Candidate subsets require downstream human/scientific review before any training use.",
            "Target latents are offline labels only and are unavailable at inference.",
        ],
        "publication_contract": (
            "create_only_directory; audit_receipt_json_is_ready_marker_published_last"
        ),
        "sha256_manifest_path": "SHA256SUMS",
    }
    receipt["receipt_digest"] = _object_sha256(receipt)
    ready_marker = "audit_receipt.json"
    receipt_payload = _json_payload(receipt)
    payloads[ready_marker] = receipt_payload
    manifest_lines = [
        f"{_sha256(payloads[path])}  {path}\n"
        for path in sorted(payloads)
    ]
    manifest_payload = "".join(manifest_lines).encode("ascii")
    payloads["SHA256SUMS"] = manifest_payload
    _publish_directory_create_only(
        output_dir=output, payloads=payloads, ready_marker=ready_marker
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--expected-dataset-summary-sha256", required=True)
    parser.add_argument("--preprocessed-parquet-dir", type=Path, required=True)
    parser.add_argument("--strict-routing-jsonl", type=Path, required=True)
    parser.add_argument("--expected-strict-routing-sha256", required=True)
    parser.add_argument(
        "--amplitude-thresholds",
        type=float,
        nargs="+",
        required=True,
        help="explicit normalized-Q0 cell-amplitude thresholds to audit",
    )
    parser.add_argument(
        "--coverage-caps",
        type=float,
        nargs="+",
        required=True,
        help="explicit maximum non-boundary cell fractions for the sweep",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = build_latent_locality_audit(
            dataset_summary=args.dataset_summary,
            expected_dataset_summary_sha256=args.expected_dataset_summary_sha256,
            preprocessed_parquet_dir=args.preprocessed_parquet_dir,
            strict_routing_jsonl=args.strict_routing_jsonl,
            expected_strict_routing_sha256=args.expected_strict_routing_sha256,
            amplitude_thresholds=args.amplitude_thresholds,
            coverage_caps=args.coverage_caps,
            output_dir=args.output_dir,
        )
    except LatentLocalityError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
