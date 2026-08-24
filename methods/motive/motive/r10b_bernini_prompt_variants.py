"""Publish prompt-only manifests for the R10B Bernini controlled pilot.

The canonical manifest remains the immutable ``manifest.jsonl`` in the
finalized controlled-pilot commit.  This module publishes only the two
counterfactual prompt views needed by the retrieval audit:

``original.jsonl``
    Every pilot row copied verbatim except that ``prompt`` is replaced by
    ``original_prompt``.

``cross_family_shuffle.jsonl``
    Every pilot row copied verbatim except that ``prompt`` is replaced by
    ``cross_family_shuffle_prompt``.

The controlled-pilot validator from :mod:`r10b_bernini_retrieval_audit` is the
source of truth.  An unbalanced pilot is rejected before an output directory
is created.  This is manifest materialization only: video bytes are not
copied or decoded, and rendering, generation, optimization, and training are
never authorized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from . import r10b_bernini_retrieval_audit as retrieval
from .r10b_bernini_pilot_manifest import FINAL_MANIFEST_NAME
from .r10b_tangent_core import canonical_json, object_digest


SUMMARY_SCHEMA = "motive-r10b-bernini-prompt-variants-v1"
DONE_SCHEMA = "motive-r10b-bernini-prompt-variants-done-v1"
ORIGINAL_NAME = "original.jsonl"
CROSS_FAMILY_SHUFFLE_NAME = "cross_family_shuffle.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (
    ORIGINAL_NAME,
    CROSS_FAMILY_SHUFFLE_NAME,
    SUMMARY_NAME,
    DONE_NAME,
)

_VARIANTS = {
    ORIGINAL_NAME: ("original", "original_prompt"),
    CROSS_FAMILY_SHUFFLE_NAME: (
        "cross_family",
        "cross_family_shuffle_prompt",
    ),
}
_AUTHORIZATION = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}


class R10BBerniniPromptVariantError(ValueError):
    """A pilot, prompt-only transform, or immutable closure is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
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


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        canonical_json(dict(row)) + "\n" for row in rows
    ).encode("utf-8")


def _validated_pilot(
    pilot_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    try:
        rows, summary, binding = retrieval._pilot_commit(pilot_dir)
    except retrieval.R10BBerniniRetrievalAuditError as error:
        raise R10BBerniniPromptVariantError(
            f"controlled-pilot validation failed: {error}"
        ) from error
    if (
        summary.get("balanced_pilot_ready") is not True
        or binding.get("balanced_pilot_ready") is not True
    ):
        raise R10BBerniniPromptVariantError(
            "prompt variants require balanced_pilot_ready=true"
        )
    for index, row in enumerate(rows):
        canonical = row.get("canonical_prompt")
        if (
            not isinstance(canonical, str)
            or not canonical.strip()
            or row.get("prompt") != canonical
        ):
            raise R10BBerniniPromptVariantError(
                f"controlled-pilot row {index} is not canonical-prompt bound"
            )
        for _tag, field in _VARIANTS.values():
            prompt = row.get(field)
            if not isinstance(prompt, str) or not prompt.strip():
                raise R10BBerniniPromptVariantError(
                    f"controlled-pilot row {index} lacks {field}"
                )
    return rows, summary, binding


def _materialize_variant(
    pilot_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in pilot_rows:
        copied = copy.deepcopy(dict(row))
        copied["prompt"] = copied[field]
        output.append(copied)
    return output


def _assert_exact_prompt_transform(
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
    field: str,
    label: str,
) -> None:
    if len(variant_rows) != len(pilot_rows):
        raise R10BBerniniPromptVariantError(
            f"{label} row count differs from the controlled pilot"
        )
    for index, (pilot_row, observed_row) in enumerate(
        zip(pilot_rows, variant_rows, strict=True)
    ):
        expected = copy.deepcopy(dict(pilot_row))
        expected["prompt"] = expected[field]
        if dict(observed_row) != expected:
            raise R10BBerniniPromptVariantError(
                f"{label} row {index} is not the exact prompt-only transform"
            )


def _summary_payload(
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    pilot_binding: Mapping[str, Any],
    variant_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    pilot_path = Path(str(pilot_binding["path"]))
    iids = [str(row["iid"]) for row in pilot_rows]
    components = [str(row["component_id"]) for row in pilot_rows]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "experiment_role": "prompt_only_counterfactual_manifest_views",
        "rows": len(pilot_rows),
        "source_controlled_pilot": copy.deepcopy(dict(pilot_binding)),
        "canonical": {
            "copied": False,
            "manifest": str(pilot_path / FINAL_MANIFEST_NAME),
            "prompt_field": "canonical_prompt",
            "rows": len(pilot_rows),
            "sha256": pilot_binding["manifest_sha256"],
        },
        "variants": {
            name: {
                "prompt_field": field,
                "rows": len(pilot_rows),
                "sha256": _sha256_bytes(variant_bytes[name]),
            }
            for name, (_tag, field) in _VARIANTS.items()
        },
        "order_binding": {
            "order_preserved": True,
            "iid_order_sha256": object_digest(iids),
            "component_order_sha256": object_digest(components),
            "unique_iids": len(set(iids)),
            "unique_components": len(set(components)),
            "component_disjoint": len(set(components)) == len(components),
        },
        "transform_contract": {
            "copy_every_pilot_row": True,
            "copy_every_non_prompt_field_exactly": True,
            "replace_prompt_only": True,
            "row_reordering_allowed": False,
            "row_reuse_allowed": False,
            "component_reuse_allowed": False,
        },
        "canonical_manifest_copied": False,
        "video_bytes_copied": False,
        "videos_decoded": False,
        "rendering_performed": False,
        "generation_performed": False,
        "optimization_performed": False,
        "training_performed": False,
        "authorization": copy.deepcopy(_AUTHORIZATION),
    }


def _done_payload(
    *,
    rows: int,
    pilot_binding: Mapping[str, Any],
    original_bytes: bytes,
    cross_bytes: bytes,
    summary_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": DONE_SCHEMA,
        "rows": rows,
        "balanced_pilot_ready": True,
        "source_controlled_pilot_commit_digest": pilot_binding[
            "commit_digest"
        ],
        "canonical_manifest_sha256": pilot_binding["manifest_sha256"],
        "files": {
            ORIGINAL_NAME: _sha256_bytes(original_bytes),
            CROSS_FAMILY_SHUFFLE_NAME: _sha256_bytes(cross_bytes),
            SUMMARY_NAME: _sha256_bytes(summary_bytes),
        },
        "authorization": copy.deepcopy(_AUTHORIZATION),
    }


def _atomic_directory(
    output_dir: str | Path,
    files: Mapping[str, bytes],
) -> None:
    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        for name, payload in files.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_prompt_variants(
    *,
    pilot_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build the two immutable prompt-only manifest views."""

    pilot_rows, _pilot_summary, pilot_binding = _validated_pilot(pilot_dir)
    original_rows = _materialize_variant(
        pilot_rows,
        field="original_prompt",
    )
    cross_rows = _materialize_variant(
        pilot_rows,
        field="cross_family_shuffle_prompt",
    )
    _assert_exact_prompt_transform(
        pilot_rows=pilot_rows,
        variant_rows=original_rows,
        field="original_prompt",
        label=ORIGINAL_NAME,
    )
    _assert_exact_prompt_transform(
        pilot_rows=pilot_rows,
        variant_rows=cross_rows,
        field="cross_family_shuffle_prompt",
        label=CROSS_FAMILY_SHUFFLE_NAME,
    )
    original_bytes = _jsonl_bytes(original_rows)
    cross_bytes = _jsonl_bytes(cross_rows)
    variant_bytes = {
        ORIGINAL_NAME: original_bytes,
        CROSS_FAMILY_SHUFFLE_NAME: cross_bytes,
    }
    summary = _summary_payload(
        pilot_rows=pilot_rows,
        pilot_binding=pilot_binding,
        variant_bytes=variant_bytes,
    )
    summary_bytes = _pretty_bytes(summary)
    done = _done_payload(
        rows=len(pilot_rows),
        pilot_binding=pilot_binding,
        original_bytes=original_bytes,
        cross_bytes=cross_bytes,
        summary_bytes=summary_bytes,
    )
    _atomic_directory(
        output_dir,
        {
            ORIGINAL_NAME: original_bytes,
            CROSS_FAMILY_SHUFFLE_NAME: cross_bytes,
            SUMMARY_NAME: summary_bytes,
            DONE_NAME: _pretty_bytes(done),
        },
    )
    return validate_prompt_variants(
        variant_dir=output_dir,
        pilot_dir=pilot_dir,
    )


def _strict_jsonl(
    path: Path,
    *,
    field: str,
) -> tuple[list[dict[str, Any]], bytes]:
    try:
        return retrieval._read_jsonl(path, field=field)
    except retrieval.R10BBerniniRetrievalAuditError as error:
        raise R10BBerniniPromptVariantError(str(error)) from error


def _strict_json(
    path: Path,
    *,
    field: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        return retrieval._read_json_object(path, field=field)
    except retrieval.R10BBerniniRetrievalAuditError as error:
        raise R10BBerniniPromptVariantError(str(error)) from error


def validate_prompt_variants(
    *,
    variant_dir: str | Path,
    pilot_dir: str | Path,
) -> dict[str, Any]:
    """Validate exact prompt transforms and the immutable SHA closure."""

    pilot_rows, _pilot_summary, pilot_binding = _validated_pilot(pilot_dir)
    root = Path(variant_dir).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise R10BBerniniPromptVariantError(
            f"variant_dir must be one non-symlink directory: {root}"
        )
    observed = sorted(path.name for path in root.iterdir())
    if observed != sorted(OUTPUT_NAMES):
        raise R10BBerniniPromptVariantError(
            f"prompt-variant closure differs: {observed}"
        )

    original_rows, original_bytes = _strict_jsonl(
        root / ORIGINAL_NAME,
        field=ORIGINAL_NAME,
    )
    cross_rows, cross_bytes = _strict_jsonl(
        root / CROSS_FAMILY_SHUFFLE_NAME,
        field=CROSS_FAMILY_SHUFFLE_NAME,
    )
    summary, summary_bytes = _strict_json(
        root / SUMMARY_NAME,
        field=SUMMARY_NAME,
    )
    done, done_bytes = _strict_json(
        root / DONE_NAME,
        field=DONE_NAME,
    )
    if summary_bytes != _pretty_bytes(summary):
        raise R10BBerniniPromptVariantError(
            "summary.json is not canonical pretty JSON"
        )
    if done_bytes != _pretty_bytes(done):
        raise R10BBerniniPromptVariantError(
            "done.json is not canonical pretty JSON"
        )

    _assert_exact_prompt_transform(
        pilot_rows=pilot_rows,
        variant_rows=original_rows,
        field="original_prompt",
        label=ORIGINAL_NAME,
    )
    _assert_exact_prompt_transform(
        pilot_rows=pilot_rows,
        variant_rows=cross_rows,
        field="cross_family_shuffle_prompt",
        label=CROSS_FAMILY_SHUFFLE_NAME,
    )
    variant_bytes = {
        ORIGINAL_NAME: original_bytes,
        CROSS_FAMILY_SHUFFLE_NAME: cross_bytes,
    }
    expected_summary = _summary_payload(
        pilot_rows=pilot_rows,
        pilot_binding=pilot_binding,
        variant_bytes=variant_bytes,
    )
    if summary != expected_summary:
        raise R10BBerniniPromptVariantError(
            "summary.json differs from the exact pilot/variant binding"
        )
    expected_done = _done_payload(
        rows=len(pilot_rows),
        pilot_binding=pilot_binding,
        original_bytes=original_bytes,
        cross_bytes=cross_bytes,
        summary_bytes=summary_bytes,
    )
    if done != expected_done:
        raise R10BBerniniPromptVariantError(
            "done.json differs from the exact immutable closure"
        )

    files = {
        ORIGINAL_NAME: _sha256_bytes(original_bytes),
        CROSS_FAMILY_SHUFFLE_NAME: _sha256_bytes(cross_bytes),
        SUMMARY_NAME: _sha256_bytes(summary_bytes),
        DONE_NAME: _sha256_bytes(done_bytes),
    }
    return {
        "status": "VALID",
        "output_dir": str(root.resolve()),
        "rows": len(pilot_rows),
        "files": files,
        "commit_digest": object_digest(files),
        "canonical_manifest": summary["canonical"],
        "balanced_pilot_ready": True,
        "video_bytes_copied": False,
        "rendering_performed": False,
        "generation_performed": False,
        "training_performed": False,
        "authorization": copy.deepcopy(_AUTHORIZATION),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate Bernini controlled-pilot prompt views."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--pilot-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--pilot-dir", type=Path, required=True)
    validate.add_argument(
        "--variant-dir",
        "--output-dir",
        dest="variant_dir",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        result = build_prompt_variants(
            pilot_dir=args.pilot_dir,
            output_dir=args.output_dir,
        )
    else:
        result = validate_prompt_variants(
            variant_dir=args.variant_dir,
            pilot_dir=args.pilot_dir,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CROSS_FAMILY_SHUFFLE_NAME",
    "DONE_NAME",
    "ORIGINAL_NAME",
    "R10BBerniniPromptVariantError",
    "SUMMARY_NAME",
    "build_prompt_variants",
    "validate_prompt_variants",
]
