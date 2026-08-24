#!/usr/bin/env python3
"""Create-only r5 calibration evidence from the authenticated r4 tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.numpy import load_file, save_file


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_owned_role_mask_calibration_v15b_r5 as calibration  # noqa: E402


INPUT_TENSOR_SHA256 = "1ac1d1643b71aae3275660d45035a73817c980a8c5dde952458acfb1c6bc94c2"
INPUT_RECEIPT_FILE_SHA256 = "1048ba5c86102311bbc57a2353ce1234ca45275428cada4a13c550050c669078"
INPUT_RECEIPT_CONTENT_SHA256 = "d71139299adeb0b68bfaac6a1296bdd4b5c5237b257bc6b938fe61105cde4497"


class R5MaterializationError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R5MaterializationError("cannot parse input receipt") from error
    if not isinstance(value, Mapping):
        raise R5MaterializationError("input receipt is not one object")
    return value


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    tensor_path = Path(args.tensors)
    receipt_path = Path(args.receipt)
    output = Path(args.output)
    output_tensors = Path(args.output_tensors)
    if output.exists() or output_tensors.exists():
        raise R5MaterializationError("output already exists")
    if _sha256_file(tensor_path) != INPUT_TENSOR_SHA256:
        raise R5MaterializationError("input tensor SHA differs")
    if _sha256_file(receipt_path) != INPUT_RECEIPT_FILE_SHA256:
        raise R5MaterializationError("input receipt file SHA differs")
    source_receipt = _read_json(receipt_path)
    if (
        source_receipt.get("receipt_sha256") != INPUT_RECEIPT_CONTENT_SHA256
        or source_receipt.get("route_authorized") is not False
        or source_receipt.get("training_authorized") is not False
        or source_receipt.get("decode_authorized") is not False
    ):
        raise R5MaterializationError("input observer authority differs")
    tensors = load_file(str(tensor_path))
    role_maps = np.stack(
        [tensors[f"block_{block:02d}_affinity"] for block in calibration.BLOCKS],
        axis=0,
    ).astype(np.float32, copy=False)
    result = calibration.calibrate_source_role_maps(
        role_maps,
        null_span_maps=None,
        null_registry_sha256=None,
    )
    if result.receipt["status"] != "strict_fail_null_token_bank_absent":
        raise R5MaterializationError("r5 unexpectedly passed strict calibration")
    payload = {
        "standardized_role_maps": result.standardized_role_maps,
        "exploratory_track_masks_u8": result.exploratory_track_masks.astype(np.uint8),
        "strict_block_masks_u8": result.strict_block_masks.astype(np.uint8),
        "strict_aggregate_masks_u8": result.strict_aggregate_masks.astype(np.uint8),
    }
    output_tensors.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        payload,
        str(output_tensors),
        metadata={
            "schema_version": calibration.SCHEMA_VERSION,
            "calibration_receipt_sha256": str(result.receipt["receipt_sha256"]),
        },
    )
    final_payload = {
        **dict(result.receipt),
        "input_probe_receipt_file_sha256": INPUT_RECEIPT_FILE_SHA256,
        "input_probe_receipt_sha256": INPUT_RECEIPT_CONTENT_SHA256,
        "input_diagnostic_tensor_file_sha256": INPUT_TENSOR_SHA256,
        "output_tensor_file_sha256": _sha256_file(output_tensors),
        "output_tensor_path": str(output_tensors.resolve(strict=True)),
    }
    # Bind the artifact envelope separately; the embedded calibration receipt
    # remains unchanged and independently checkable.
    final_payload["artifact_receipt_sha256"] = calibration.object_sha256(final_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(final_payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    except FileExistsError as error:
        raise R5MaterializationError("output receipt already exists") from error
    return final_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tensors", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (R5MaterializationError, calibration.V15BR5CalibrationError) as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
