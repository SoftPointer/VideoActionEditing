#!/usr/bin/env python3
"""Build a provenance-bound Bernini-R raw parquet from Omni action previews.

The source manifest is deliberately *preview-only*.  This converter therefore
requires an explicit acknowledgement before it writes anything, preserves the
upstream authorization fields verbatim, and marks every output as unsuitable
for production or scientific claims.  It does not grant training authorization;
it records a user-directed experimental use despite the upstream prohibition.

The emitted ``inputs`` and ``videos`` columns follow Bernini's public renderer
training contract.  Each conversation is exactly: source video (no loss), edit
instruction (no loss), target video (loss).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STRICT_LOADER_PATH = (
    REPOSITORY_ROOT
    / "methods"
    / "omnivideo2_action_editing"
    / "tools"
    / "materialize_action_payloads.py"
)

ROW_FORMAT = "bernini-r-action-raw-row-v2"
RECEIPT_FORMAT = "bernini-r-action-raw-dataset-receipt-v2"
JOB_DONE_FORMAT = "bernini-r-action-raw-job-done-v2"
BERNINI_MESSAGE_TYPES = ("video", "text", "video_gen")
STRICT_INCLUSION_POLICY = "strict_single_actor"
NATURAL_RELEASE_INCLUSION_POLICY = "natural_release_all"
UPSTREAM_AUTHORIZATION_FIELDS = (
    "preview_only",
    "training_authorized",
    "training_use_forbidden",
    "production_eligible",
    "post_video_acceptance",
)


class RendererDatasetError(RuntimeError):
    """A fail-closed dataset conversion error."""


def _load_strict_preview_module() -> Any:
    """Load the existing strict validator from its exact repository path."""

    module_name = "_omnivideo2_strict_action_preview_materializer"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    if not STRICT_LOADER_PATH.is_file():
        raise RendererDatasetError(
            f"strict preview loader is missing: {STRICT_LOADER_PATH}"
        )
    spec = importlib.util.spec_from_file_location(module_name, STRICT_LOADER_PATH)
    if spec is None or spec.loader is None:
        raise RendererDatasetError(
            f"cannot import strict preview loader: {STRICT_LOADER_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect sys.modules while the source module is executing.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


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
        raise RendererDatasetError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RendererDatasetError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _plain_hashed_file(path_value: Any, hash_value: Any, *, context: str) -> Path:
    if type(path_value) is not str or not path_value:
        raise RendererDatasetError(f"{context} path must be non-empty text")
    if type(hash_value) is not str or not re.fullmatch(r"[0-9a-f]{64}", hash_value):
        raise RendererDatasetError(f"{context} hash must be a lowercase SHA-256")
    path = Path(path_value).expanduser()
    if not path.is_file() or path.is_symlink():
        raise RendererDatasetError(f"{context} is not a plain file: {path}")
    if file_sha256(path) != hash_value:
        raise RendererDatasetError(f"{context} hash mismatch: {path}")
    return path.resolve(strict=True)


def _natural_release_row_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise RendererDatasetError("natural release manifest must end with a newline")
    lines = payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise RendererDatasetError(
            "natural release manifest must contain non-blank JSONL rows"
        )
    index: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, RendererDatasetError) as error:
            raise RendererDatasetError(
                f"invalid natural release row {line_number}: {error}"
            ) from error
        if not isinstance(row, Mapping):
            raise RendererDatasetError(
                f"natural release row {line_number} must be an object"
            )
        iid = row.get("iid")
        if type(iid) is not str or not iid or iid in index:
            raise RendererDatasetError(
                f"invalid or duplicate natural release IID on line {line_number}"
            )
        if row.get("schema_version") != "motive-goku-natural-motion-dataset-row-v1":
            raise RendererDatasetError(
                f"natural release row schema differs for {iid}"
            )
        instruction = _required_text(
            row.get("natural_edit_instruction"),
            context=f"natural release instruction for {iid}",
        )
        instruction_sha = row.get("natural_edit_instruction_sha256")
        if (
            type(instruction_sha) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", instruction_sha)
            or hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            != instruction_sha
        ):
            raise RendererDatasetError(
                f"natural release instruction hash differs for {iid}"
            )
        semantic_audit = row.get("semantic_audit")
        diagnostics = (
            semantic_audit.get("model_reported_diagnostics")
            if isinstance(semantic_audit, Mapping)
            else None
        )
        if (
            row.get("label_status")
            != "structured_plan_semantic_audit_passed_video_audit_pending"
            or not isinstance(semantic_audit, Mapping)
            or semantic_audit.get("effective_verdict") != "pass"
            or not isinstance(diagnostics, Mapping)
            or diagnostics.get("confidence") != "high"
        ):
            raise RendererDatasetError(
                f"natural release semantic status differs for {iid}"
            )
        index[iid] = {
            "row": dict(row),
            "row_file_sha256": hashlib.sha256(
                canonical_json_bytes(row) + b"\n"
            ).hexdigest(),
        }
    return index


def _required_text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise RendererDatasetError(f"{context} must be non-empty text")
    return value


def _authorization_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy, never reinterpret, the fields validated by the upstream loader."""

    return {field: row[field] for field in UPSTREAM_AUTHORIZATION_FIELDS}


def _validate_natural_release_bindings(items: Sequence[Any]) -> dict[str, Any]:
    """Revalidate the release shared by an opt-in broad natural-label cohort."""

    identity: Optional[tuple[str, str, str, str]] = None
    for item in items:
        row = item.row
        iid = _required_text(row.get("iid"), context="IID")
        if row.get("instruction_source") != "natural":
            raise RendererDatasetError(
                f"broader natural-release inclusion has non-natural row: {iid}"
            )
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RendererDatasetError(f"missing preview provenance for {iid}")
        current = (
            _required_text(
                provenance.get("natural_release_summary_path"),
                context=f"natural release summary path for {iid}",
            ),
            _required_text(
                provenance.get("natural_release_summary_sha256"),
                context=f"natural release summary hash for {iid}",
            ),
            _required_text(
                provenance.get("natural_release_manifest_path"),
                context=f"natural release manifest path for {iid}",
            ),
            _required_text(
                provenance.get("natural_release_manifest_sha256"),
                context=f"natural release manifest hash for {iid}",
            ),
        )
        if identity is None:
            identity = current
        elif current != identity:
            raise RendererDatasetError("preview rows bind different natural releases")
    if identity is None:
        raise RendererDatasetError("natural-release cohort is empty")

    summary_path = _plain_hashed_file(
        identity[0], identity[1], context="natural release summary"
    )
    manifest_path = _plain_hashed_file(
        identity[2], identity[3], context="natural release manifest"
    )
    try:
        summary = json.loads(
            summary_path.read_bytes(), object_pairs_hook=_reject_duplicate_keys
        )
    except (json.JSONDecodeError, RendererDatasetError) as error:
        raise RendererDatasetError(f"invalid natural release summary: {error}") from error
    if not isinstance(summary, Mapping):
        raise RendererDatasetError("natural release summary must be an object")
    summary = dict(summary)
    declared_digest = summary.pop("summary_digest", None)
    if (
        summary.get("schema_version")
        != "motive-goku-natural-motion-verify-summary-v1"
        or declared_digest != object_sha256(summary)
    ):
        raise RendererDatasetError("natural release summary identity/digest differs")
    declared_manifest = Path(
        _required_text(
            summary.get("dataset_manifest_path"),
            context="natural release declared manifest path",
        )
    ).expanduser()
    if declared_manifest.resolve(strict=True) != manifest_path:
        raise RendererDatasetError("natural release summary manifest path differs")
    if summary.get("dataset_manifest_sha256") != identity[3]:
        raise RendererDatasetError("natural release summary manifest hash differs")
    index = _natural_release_row_index(manifest_path)
    expected = summary.get("expected_rows")
    terminal = summary.get("terminal_rows")
    ok_rows = summary.get("ok_rows")
    errors = summary.get("error_rows")
    if (
        type(expected) is not int
        or type(terminal) is not int
        or type(ok_rows) is not int
        or type(errors) is not int
        or expected != terminal
        or ok_rows + errors != terminal
        or ok_rows != len(index)
    ):
        raise RendererDatasetError("natural release count closure differs")

    selected_iids = {str(item.row["iid"]) for item in items}
    if selected_iids != set(index):
        missing = sorted(set(index) - selected_iids)
        extra = sorted(selected_iids - set(index))
        raise RendererDatasetError(
            "natural_release_all must exactly equal the published release: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    for item in items:
        iid = str(item.row["iid"])
        provenance = item.row["provenance"]
        declared_row_sha = provenance.get("natural_release_row_file_sha256")
        release_entry = index.get(iid)
        if (
            release_entry is None
            or declared_row_sha != release_entry["row_file_sha256"]
        ):
            raise RendererDatasetError(
                f"preview row is not bound to the natural release: {iid}"
            )
        release_row = release_entry["row"]
        if (
            item.row.get("edit_instruction")
            != release_row.get("natural_edit_instruction")
            or item.row.get("edit_instruction_sha256")
            != release_row.get("natural_edit_instruction_sha256")
        ):
            raise RendererDatasetError(
                f"preview instruction differs from natural release: {iid}"
            )
        qwen_path = _plain_hashed_file(
            provenance.get("qwen_passed_path"),
            provenance.get("qwen_passed_sha256"),
            context=f"Qwen passed row {iid}",
        )
        natural_result_path = _plain_hashed_file(
            provenance.get("natural_result_path"),
            provenance.get("natural_result_sha256"),
            context=f"natural result {iid}",
        )
        _plain_hashed_file(
            provenance.get("natural_receipt_path"),
            provenance.get("natural_receipt_sha256"),
            context=f"natural receipt {iid}",
        )
        natural_instruction_path = _plain_hashed_file(
            provenance.get("natural_instruction_path"),
            provenance.get("natural_instruction_file_sha256"),
            context=f"natural instruction {iid}",
        )
        release_qwen_path = _plain_hashed_file(
            release_row.get("source_passed_path"),
            release_row.get("source_passed_sha256"),
            context=f"natural release Qwen row {iid}",
        )
        release_result_path = _plain_hashed_file(
            release_row.get("result_path"),
            release_row.get("result_sha256"),
            context=f"natural release result {iid}",
        )
        if (
            release_qwen_path != qwen_path
            or release_row.get("source_passed_sha256")
            != provenance.get("qwen_passed_sha256")
            or release_result_path != natural_result_path
            or release_row.get("result_sha256")
            != provenance.get("natural_result_sha256")
            or natural_instruction_path.read_bytes()
            != (str(item.row["edit_instruction"]) + "\n").encode("utf-8")
        ):
            raise RendererDatasetError(
                f"natural release artifact binding differs: {iid}"
            )
    if file_sha256(summary_path) != identity[1] or file_sha256(manifest_path) != identity[3]:
        raise RendererDatasetError("natural release changed during conversion")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": identity[3],
        "summary_path": str(summary_path),
        "summary_sha256": identity[1],
        "expected_rows": expected,
        "release_rows": len(index),
        "error_rows": errors,
        "iid_set_sha256": hashlib.sha256(
            "".join(f"{iid}\n" for iid in sorted(index)).encode("utf-8")
        ).hexdigest(),
    }


def renderer_row(
    item: Any,
    *,
    preview_manifest: Path,
    manifest_sha256: str,
    inclusion_policy: str = STRICT_INCLUSION_POLICY,
) -> dict[str, Any]:
    """Convert one already validated preview item to Bernini's raw-row contract."""

    upstream = item.row
    iid = _required_text(upstream.get("iid"), context="IID")
    instruction = _required_text(
        upstream.get("edit_instruction"), context=f"instruction for {iid}"
    )
    authorization = _authorization_snapshot(upstream)
    selection_gates = upstream.get("selection_gates")
    if not isinstance(selection_gates, Mapping) or not selection_gates:
        raise RendererDatasetError(f"selection gates are missing for {iid}")
    if any(type(value) is not bool for value in selection_gates.values()):
        raise RendererDatasetError(f"selection gates are invalid for {iid}")

    messages = [
        {"type": "video", "has_loss": 0},
        {"type": "text", "text": instruction, "has_loss": 0},
        {"type": "video_gen", "has_loss": 1},
    ]
    videos = [
        {"video_path": str(item.source_video)},
        {"video_path": str(item.target_video)},
    ]
    row = {
        "schema_version": ROW_FORMAT,
        "inputs": canonical_json_bytes(messages).decode("utf-8"),
        "videos": videos,
        "iid": iid,
        "group_id": _required_text(
            upstream.get("group_id"), context=f"group_id for {iid}"
        ),
        "family": _required_text(
            upstream.get("family"), context=f"family for {iid}"
        ),
        "edit_instruction_sha256": upstream["edit_instruction_sha256"],
        "source_video_path": str(item.source_video),
        "source_video_declared_path": upstream["source_video_path"],
        "source_video_sha256": upstream["source_video_sha256"],
        "target_video_path": str(item.target_video),
        "target_video_declared_path": upstream["target_video_path"],
        "target_video_sha256": upstream["target_video_sha256"],
        "shared_i0_path": str(item.shared_i0.path),
        "shared_i0_sha256": item.shared_i0.sha256,
        "preview_manifest_path": str(preview_manifest),
        "preview_manifest_sha256": manifest_sha256,
        "preview_row_digest": item.preview_join_row_digest,
        "preview_row_file_sha256": item.preview_join_row_file_sha256,
        "experimental_inclusion_policy": inclusion_policy,
        "selection_gates_json": canonical_json_bytes(selection_gates).decode("utf-8"),
        "strict_selection_gates_all_true": all(selection_gates.values()),
        "upstream_authorization_json": canonical_json_bytes(authorization).decode(
            "utf-8"
        ),
        # These five values are copied verbatim.  In particular, this tool does
        # not flip training_authorized or training_use_forbidden.
        **authorization,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
    }
    row["renderer_row_digest"] = object_sha256(row)
    return row


def _build_renderer_rows_with_release(
    preview_manifest: Path,
    *,
    max_rows: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    allow_broader_natural_release: bool = False,
) -> tuple[list[dict[str, Any]], Path, str, Optional[dict[str, Any]]]:
    """Strictly validate a preview manifest and return deterministic rows."""

    preview_module = _load_strict_preview_module()
    if allow_broader_natural_release and (
        max_rows is not None or sample_ids is not None
    ):
        raise RendererDatasetError(
            "natural_release_all raw conversion must include the complete release"
        )
    try:
        manifest = preview_manifest.expanduser().resolve(strict=True)
        items = preview_module.load_preview_manifest(
            manifest,
            max_rows=max_rows,
            sample_ids=sample_ids,
            allow_failed_selection_gates=allow_broader_natural_release,
        )
        manifest_sha = preview_module.file_sha256(manifest)
    except Exception as error:
        if isinstance(error, RendererDatasetError):
            raise
        raise RendererDatasetError(f"preview manifest validation failed: {error}") from error
    release_binding = (
        _validate_natural_release_bindings(items)
        if allow_broader_natural_release
        else None
    )
    inclusion_policy = (
        NATURAL_RELEASE_INCLUSION_POLICY
        if allow_broader_natural_release
        else STRICT_INCLUSION_POLICY
    )
    rows = [
        renderer_row(
            item,
            preview_manifest=manifest,
            manifest_sha256=manifest_sha,
            inclusion_policy=inclusion_policy,
        )
        for item in items
    ]
    if preview_module.file_sha256(manifest) != manifest_sha:
        raise RendererDatasetError("preview manifest changed during conversion")
    return rows, manifest, manifest_sha, release_binding


def build_renderer_rows(
    preview_manifest: Path,
    *,
    max_rows: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    allow_broader_natural_release: bool = False,
) -> tuple[list[dict[str, Any]], Path, str]:
    """Public compatibility wrapper returning rows, manifest path, and hash."""

    rows, manifest, manifest_sha, _release = _build_renderer_rows_with_release(
        preview_manifest,
        max_rows=max_rows,
        sample_ids=sample_ids,
        allow_broader_natural_release=allow_broader_natural_release,
    )
    return rows, manifest, manifest_sha


def _arrow_schema(pa: Any) -> Any:
    video = pa.struct([pa.field("video_path", pa.string(), nullable=False)])
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("inputs", pa.string(), nullable=False),
            pa.field("videos", pa.list_(video), nullable=False),
            pa.field("iid", pa.string(), nullable=False),
            pa.field("group_id", pa.string(), nullable=False),
            pa.field("family", pa.string(), nullable=False),
            pa.field("edit_instruction_sha256", pa.string(), nullable=False),
            pa.field("source_video_path", pa.string(), nullable=False),
            pa.field("source_video_declared_path", pa.string(), nullable=False),
            pa.field("source_video_sha256", pa.string(), nullable=False),
            pa.field("target_video_path", pa.string(), nullable=False),
            pa.field("target_video_declared_path", pa.string(), nullable=False),
            pa.field("target_video_sha256", pa.string(), nullable=False),
            pa.field("shared_i0_path", pa.string(), nullable=False),
            pa.field("shared_i0_sha256", pa.string(), nullable=False),
            pa.field("preview_manifest_path", pa.string(), nullable=False),
            pa.field("preview_manifest_sha256", pa.string(), nullable=False),
            pa.field("preview_row_digest", pa.string(), nullable=False),
            pa.field("preview_row_file_sha256", pa.string(), nullable=False),
            pa.field("experimental_inclusion_policy", pa.string(), nullable=False),
            pa.field("selection_gates_json", pa.string(), nullable=False),
            pa.field("strict_selection_gates_all_true", pa.bool_(), nullable=False),
            pa.field("upstream_authorization_json", pa.string(), nullable=False),
            pa.field("preview_only", pa.bool_(), nullable=False),
            pa.field("training_authorized", pa.bool_(), nullable=False),
            pa.field("training_use_forbidden", pa.bool_(), nullable=False),
            pa.field("production_eligible", pa.bool_(), nullable=False),
            pa.field("post_video_acceptance", pa.string(), nullable=False),
            pa.field("experimental_training_acknowledged", pa.bool_(), nullable=False),
            pa.field("production_claim_forbidden", pa.bool_(), nullable=False),
            pa.field("renderer_row_digest", pa.string(), nullable=False),
        ]
    )


def write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write and immediately validate an official raw-parquet container."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RendererDatasetError(
            "writing Bernini raw parquet requires pyarrow"
        ) from error
    try:
        schema = _arrow_schema(pa)
        table = pa.Table.from_pylist(list(rows), schema=schema)
        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        )
        metadata = pq.read_metadata(path)
        persisted_schema = pq.read_schema(path)
    except Exception as error:
        raise RendererDatasetError(f"cannot write Bernini parquet: {error}") from error
    if metadata.num_rows != len(rows):
        raise RendererDatasetError(
            f"parquet row count differs: {metadata.num_rows} != {len(rows)}"
        )
    if persisted_schema.names != schema.names:
        raise RendererDatasetError("parquet schema fields differ after writing")


def _reserve_output(path: Path, *, context: str) -> None:
    if path.exists() or path.is_symlink():
        raise RendererDatasetError(f"create-only {context} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _publish_create_only(staging: Path, destination: Path) -> None:
    """Publish one staged plain file without overwriting a concurrent writer."""

    try:
        os.link(staging, destination)
    except FileExistsError as error:
        raise RendererDatasetError(
            f"create-only output appeared during publication: {destination}"
        ) from error


ParquetWriter = Callable[[Sequence[Mapping[str, Any]], Path], None]


def build_dataset(
    preview_manifest: Path,
    output_parquet: Path,
    *,
    receipt_path: Optional[Path] = None,
    acknowledge_preview_only: bool = False,
    acknowledge_broader_natural_release: bool = False,
    max_rows: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    parquet_writer: ParquetWriter = write_parquet,
) -> dict[str, Any]:
    """Build a create-only parquet and its experimental-use receipt."""

    if acknowledge_preview_only is not True:
        raise RendererDatasetError(
            "refusing preview-only data without --acknowledge-preview-only"
        )
    output = output_parquet.expanduser().absolute()
    receipt = (
        Path(f"{output}.receipt.json")
        if receipt_path is None
        else receipt_path.expanduser().absolute()
    )
    if output == receipt:
        raise RendererDatasetError("parquet and receipt paths must differ")
    _reserve_output(output, context="parquet")
    _reserve_output(receipt, context="receipt")

    rows, manifest, manifest_sha, release_binding = _build_renderer_rows_with_release(
        preview_manifest,
        max_rows=max_rows,
        sample_ids=sample_ids,
        allow_broader_natural_release=acknowledge_broader_natural_release,
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging.", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        staged_parquet = staging / output.name
        parquet_writer(rows, staged_parquet)
        if not staged_parquet.is_file() or staged_parquet.is_symlink():
            raise RendererDatasetError("parquet writer did not produce a plain file")
        parquet_sha = file_sha256(staged_parquet)
        row_digests = [str(row["renderer_row_digest"]) for row in rows]
        inclusion_policy = (
            NATURAL_RELEASE_INCLUSION_POLICY
            if acknowledge_broader_natural_release
            else STRICT_INCLUSION_POLICY
        )
        strict_selection_rows = sum(
            bool(row["strict_selection_gates_all_true"]) for row in rows
        )
        result = {
            "schema_version": RECEIPT_FORMAT,
            "complete": True,
            "experimental_training_acknowledged": True,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_eligible": False,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
            "source_preview_manifest": str(manifest),
            "source_preview_manifest_sha256": manifest_sha,
            "experimental_inclusion_policy": inclusion_policy,
            "broader_natural_release_inclusion_acknowledged": bool(
                acknowledge_broader_natural_release
            ),
            "sample_count": len(rows),
            "strict_selection_rows": strict_selection_rows,
            "non_strict_selection_rows": len(rows) - strict_selection_rows,
            "sample_ids": [str(row["iid"]) for row in rows],
            "renderer_row_digests_sha256": object_sha256(row_digests),
            "bernini_messages": list(BERNINI_MESSAGE_TYPES),
            "parquet_path": str(output),
            "parquet_sha256": parquet_sha,
        }
        if release_binding is not None:
            result["natural_release"] = release_binding
        result["receipt_digest"] = object_sha256(result)
        receipt_bytes = (
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        staged_receipt = staging / receipt.name
        staged_receipt.write_bytes(receipt_bytes)

        _publish_create_only(staged_parquet, output)
        try:
            _publish_create_only(staged_receipt, receipt)
        except Exception:
            # This invocation created ``output`` moments ago; remove only that
            # inode so a failed receipt publication cannot leave a false-ready
            # dataset artifact.
            output.unlink()
            raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-manifest", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument(
        "--acknowledge-preview-only",
        action="store_true",
        help=(
            "explicitly acknowledge that upstream forbids training use and "
            "that this output is experimental and claim-ineligible"
        ),
    )
    parser.add_argument(
        "--acknowledge-broader-natural-release",
        action="store_true",
        help=(
            "include every hash-bound semantic-pass row in the completed natural "
            "release even when a descriptive narrow-cohort gate is false"
        ),
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        build_dataset(
            args.preview_manifest,
            args.output_parquet,
            receipt_path=args.receipt_path,
            acknowledge_preview_only=args.acknowledge_preview_only,
            acknowledge_broader_natural_release=(
                args.acknowledge_broader_natural_release
            ),
            max_rows=args.max_rows,
            sample_ids=args.sample_ids,
        )
    except RendererDatasetError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
