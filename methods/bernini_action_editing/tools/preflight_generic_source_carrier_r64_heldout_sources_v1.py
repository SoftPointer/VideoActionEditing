#!/usr/bin/env python3
"""No-model exact81 media preflight for all eight R64 held-out sources.

This program runs before ``torchrun``.  It authenticates the source-only-v3
manifest and raw source-only projection, then uses the exact release-local
``materialize_vae``/decord path to decode every frame of every held-out MP4.
It never imports Bernini, VeOmni, PyTorch, Transformers, or Diffusers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generic_source_carrier_r64_heldout_contract_v1 as contract  # noqa: E402

# Bind and authenticate the decoder before any generic ``tools`` namespace can
# resolve through an unrelated checkout or site package.
RELEASE_PREPROCESSING_TOOL_IDENTITIES = (
    contract.bind_release_preprocessing_tools(METHOD_ROOT)
)

import clean_source_visual_context_training_v1 as source_data  # noqa: E402


SCHEMA_VERSION = "bernini-generic-source-carrier-r64-heldout-source-preflight-v1"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_MODEL_PREFIXES = ("torch", "transformers", "diffusers", "bernini", "veomni")
_MODEL_MODULES = (
    "clean_source_visual_context_adapter_v1",
    "clean_source_visual_context_checkpoint_decode_runtime_v1",
    "infer_generic_source_carrier_r64_heldout_v1",
    "infer_native_identity_generation_canary",
    "infer_native_v_axis_exact81_probe_v1",
    "tri_branch_unipc",
)


class R64HeldoutSourcePreflightError(RuntimeError):
    """Raised before any model process may start."""


def fail(message: str) -> NoReturn:
    raise R64HeldoutSourcePreflightError(message)


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise R64HeldoutSourcePreflightError(f"{label} is unavailable") from error
    if resolved != path or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return path


def _fresh_output(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if (
        not path.is_absolute()
        or path == Path("/")
        or path.exists()
        or path.is_symlink()
        or _SAFE_NAME.fullmatch(path.name) is None
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        fail("output receipt must be one fresh safe absolute file")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=contract.SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument("--output-receipt", required=True)
    return parser


def _forbidden_model_modules() -> list[str]:
    observed = []
    for name in sys.modules:
        if name in _MODEL_MODULES or any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in _MODEL_PREFIXES
        ):
            observed.append(name)
    return sorted(observed)


def build_preflight_receipt(
    source_manifest_path: Path,
    *,
    expected_source_manifest_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate and fully decode all eight real held-out source MP4s."""

    source_manifest_path = _plain_file(
        source_manifest_path, label="source-only-v3 manifest"
    )
    if (
        expected_source_manifest_sha256 != contract.SOURCE_MANIFEST_SHA256
        or contract.file_sha256(source_manifest_path)
        != expected_source_manifest_sha256
    ):
        fail("source-only-v3 manifest bytes differ")
    try:
        manifest = source_data.load_source_only_split_manifest(
            source_manifest_path, verify_files=True
        )
    except Exception as error:
        raise R64HeldoutSourcePreflightError(
            "cannot authenticate source-only-v3 manifest"
        ) from error
    heldout = tuple(
        sorted(manifest.rows_for_split("heldout"), key=lambda row: row.iid)
    )
    if (
        manifest.manifest_digest != contract.SOURCE_MANIFEST_DIGEST
        or len(heldout) != contract.HELDOUT_ROWS
        or len({row.iid for row in heldout}) != contract.HELDOUT_ROWS
        or len({row.group_id for row in heldout}) != contract.HELDOUT_ROWS
        or len({row.source_video_sha256 for row in heldout})
        != contract.HELDOUT_ROWS
        or not all(row.heldout_action_canary_eligible for row in heldout)
    ):
        fail("held-out source-only-v3 row closure differs")

    raw_path = _plain_file(
        source_data.PINNED_RAW_PARQUET, label="pinned raw/full644"
    )
    if (
        source_data.PINNED_RAW_PARQUET_SHA256 != contract.RAW_PARQUET_SHA256
        or contract.file_sha256(raw_path) != contract.RAW_PARQUET_SHA256
    ):
        fail("pinned raw/full644 bytes differ")
    try:
        import pyarrow.parquet as pq

        raw_rows = pq.read_table(
            raw_path, columns=list(contract.RAW_SAFE_COLUMNS)
        ).to_pylist()
    except Exception as error:
        raise R64HeldoutSourcePreflightError(
            "cannot read raw source-only projection"
        ) from error
    raw_by_iid = {str(row.get("iid")): row for row in raw_rows}
    if len(raw_by_iid) != len(raw_rows):
        fail("raw source-only projection contains duplicate IID")

    rows = []
    for split_row in heldout:
        raw = raw_by_iid.get(split_row.iid)
        if (
            not isinstance(raw, Mapping)
            or raw.get("group_id") != split_row.group_id
            or raw.get("family") != split_row.action_family
            or raw.get("source_video_sha256") != split_row.source_video_sha256
            or raw.get("source_video_path")
            != raw.get("source_video_declared_path")
        ):
            fail(f"held-out raw/source-only identity differs: {split_row.iid}")
        source = _plain_file(
            raw["source_video_path"], label=f"{split_row.iid} raw source"
        )
        if contract.file_sha256(source) != split_row.source_video_sha256:
            fail(f"held-out source MP4 bytes differ: {split_row.iid}")
        media = contract.validate_exact81_media(source)
        rows.append(
            {
                "iid": split_row.iid,
                "group_id": split_row.group_id,
                "action_family_provenance_only": split_row.action_family,
                "source_video": str(source),
                "source_video_sha256": split_row.source_video_sha256,
                "decoder": media["decoder"],
                "decoder_backend": media["decoder_backend"],
                "decoder_source_sha256": media["decoder_source_sha256"],
                "all_frames_decoded": media["all_frames_decoded"],
                "frame_count": media["frame_count"],
                "fps": media["fps"],
                "height": media["height"],
                "width": media["width"],
                "channels": media["channels"],
                "dtype": media["dtype"],
            }
        )

    forbidden = _forbidden_model_modules()
    if forbidden:
        fail(f"no-model preflight imported model modules: {forbidden!r}")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "complete_action_result": False,
        "action_claim_forbidden": True,
        "model_imported": False,
        "sampling_started": False,
        "source_manifest": {
            "path": str(source_manifest_path),
            "file_sha256": expected_source_manifest_sha256,
            "manifest_digest": manifest.manifest_digest,
            "split": "heldout",
            "rows": contract.HELDOUT_ROWS,
            "row_order": "iid-lexicographic",
        },
        "raw_projection": {
            "path": str(raw_path),
            "file_sha256": contract.RAW_PARQUET_SHA256,
            "safe_columns_read": list(contract.RAW_SAFE_COLUMNS),
            "target_columns_read": False,
        },
        "decoder": {
            "implementation": (
                "release-tools.materialize_vae._decode_exact_video"
            ),
            "backend": "decord",
            "release_preprocessing_tool_identities": dict(
                RELEASE_PREPROCESSING_TOOL_IDENTITIES
            ),
            "external_binary_required": False,
            "all_frame_indices_requested": list(range(contract.FRAME_COUNT)),
            "full_frame_batch_decode_required": True,
            "frame_count": contract.FRAME_COUNT,
            "fps": contract.FPS,
            "positive_height_width_required": True,
            "rgb_uint8_required": True,
        },
        "rows": rows,
    }
    return {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    raw = contract.canonical_json_bytes(receipt) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = _fresh_output(args.output_receipt)
    receipt = build_preflight_receipt(
        Path(args.source_manifest),
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    _write_receipt(output, receipt)
    print(
        "R64_HELDOUT_SOURCE_MEDIA_PREFLIGHT_PASS "
        f"rows={contract.HELDOUT_ROWS} exact81=true fps={contract.FPS} "
        "all_frames_decoded=true model_imported=false external_binary=false "
        f"receipt={output} receipt_digest={receipt['receipt_digest']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

