#!/usr/bin/env python3
"""Strict source-only inference wrapper for the PDF-v2 diagnostic arm."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_delta_lora as base  # noqa: E402
import projected_differential_flow as pdf  # noqa: E402
import train_projected_delta_lora as pdf_train  # noqa: E402


INFERENCE_RECEIPT_SCHEMA = "bernini-r-1p3b-pdf-lora-inference-receipt-v2"
_base_build_parser = base.build_parser
_base_validate_cli = base.validate_cli
_base_validate_adapter = base.validate_training_adapter_contract


def build_parser() -> argparse.ArgumentParser:
    parser = _base_build_parser()
    parser.description = "Run strict source-only Bernini PDF-v2 diagnostic inference"
    parser.add_argument("--solver-substeps", type=int, default=2)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    _base_validate_cli(args)
    if args.sampling_mode != "differential":
        raise base.DeltaInferenceError("PDF-v2 requires projected differential sampling")
    try:
        pdf.set_default_substeps(args.solver_substeps)
    except pdf.ProjectedFlowContractError as error:
        raise base.DeltaInferenceError(str(error)) from error


def validate_training_adapter_contract(
    adapter_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_tree_sha256: str,
) -> dict[str, Any]:
    original_digest = base._validate_receipt_digest(receipt)
    if receipt.get("schema_version") != pdf_train.RECEIPT_SCHEMA:
        raise base.DeltaInferenceError("PDF-v2 training receipt schema differs")
    if receipt.get("method") != pdf_train.METHOD_NAME:
        raise base.DeltaInferenceError("PDF-v2 training method identity differs")
    immutable = receipt.get("immutable_contract")
    value = immutable.get("value") if isinstance(immutable, dict) else None
    supervision = receipt.get("supervision")
    if (
        not isinstance(value, dict)
        or value.get("train_inference_projection_identical") is not True
        or value.get("motion_representation")
        != "temporal-dc-zero-action-noop-velocity-v2"
        or value.get("full_target_framewise_loss") is not False
        or value.get("first_frame_anchor") is not False
        or not isinstance(supervision, dict)
        or supervision.get("train_inference_projection_identical") is not True
        or supervision.get("first_frame_anchor") is not False
        or supervision.get("full_target_framewise_loss_enabled") is not False
    ):
        raise base.DeltaInferenceError("PDF-v2 projected/source-preserving contract differs")

    # Reuse the exhaustive v1 adapter/commit/module checks on a digest-correct
    # compatibility view; retain the original v2 digest as the authority.
    compatible = dict(receipt)
    compatible["schema_version"] = base.delta_train.RECEIPT_SCHEMA
    compatible["method"] = base.delta_train.METHOD_NAME
    compatible.pop("receipt_digest", None)
    compatible["receipt_digest"] = base.legacy_train.object_sha256(compatible)
    result = _base_validate_adapter(
        adapter_config,
        compatible,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
    )
    result["receipt_digest"] = original_digest
    return result


def _install_v2_hooks() -> None:
    base.INFERENCE_RECEIPT_SCHEMA = INFERENCE_RECEIPT_SCHEMA
    base.differential_sampler = pdf
    base.build_parser = build_parser
    base.validate_cli = validate_cli
    base.validate_training_adapter_contract = validate_training_adapter_contract


def _augment_trace_receipt(output: str) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    trace = pdf.get_last_trace()
    if trace is None:
        return
    _, receipt_path = base.legacy_infer._resolve_output(output)
    receipt = base._read_json(receipt_path, label="PDF-v2 inference receipt")
    sampling = dict(receipt.get("sampling", {}))
    sampling["solver_substeps"] = pdf._DEFAULT_SUBSTEPS
    sampling["trace"] = {
        "identity_bypassed": trace.identity_bypassed,
        "sigmas": list(trace.sigmas),
        "projected_delta_rms": list(trace.delta_rms),
        "contribution_rms": list(trace.contribution_rms),
        "cumulative_update_rms": list(trace.cumulative_update_rms),
        "temporal_dc_rms_before_projection": list(
            trace.temporal_dc_rms_before_projection
        ),
        "interval_index": list(trace.interval_index),
        "substep_index": list(trace.substep_index),
    }
    receipt["sampling"] = sampling
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = base.legacy_train.object_sha256(receipt)
    base._atomic_write_json(receipt_path, receipt)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _install_v2_hooks()
    parsed = build_parser().parse_args(argv)
    validate_cli(parsed)
    result = base.main(argv)
    _augment_trace_receipt(parsed.output)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
