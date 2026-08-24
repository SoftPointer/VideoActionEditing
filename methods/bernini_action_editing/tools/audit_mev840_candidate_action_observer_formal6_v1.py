#!/usr/bin/env python3
"""Fail-closed preflight for the sealed MEV840 formal-six observer batch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


IDS = [
    "p0_s2027",
    "p0_s2028",
    "p1_s2027",
    "p1_s2028",
    "p2_s2027",
    "p2_s2028",
]
ARM_BY_PREFIX = {"p0": "p0a", "p1": "p1", "p2": "p2"}
FORMAL_SCHEMA = "mev840-native-rv2v-paired-prompt-matrix-formal-v1"


class FormalSixAuditError(RuntimeError):
    """The formal-six preflight contract failed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_exact(path_value: Any, sha_value: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        raise FormalSixAuditError(f"{label} authority is absent")
    path = Path(path_value).resolve(strict=True)
    if (
        len(sha_value) != 64
        or not path.is_file()
        or path.is_symlink()
        or file_sha256(path) != sha_value
    ):
        raise FormalSixAuditError(f"{label} bytes differ")
    return path


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FormalSixAuditError(f"cannot parse {path}") from error
    if not isinstance(value, dict):
        raise FormalSixAuditError(f"{path} must contain one object")
    return value


def load_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("sealed_formal6_batch_runner", path)
    if spec is None or spec.loader is None:
        raise FormalSixAuditError("cannot load batch runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(manifest_path: Path, runner_path: Path, runner_sha256: str) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FormalSixAuditError("manifest must be one regular file")
    runner_path = regular_exact(str(runner_path), runner_sha256, "batch runner")
    manifest = read_json(manifest_path)
    runner = load_runner(runner_path)
    runner._validate_manifest(manifest)

    interface = manifest.get("formal6_interface")
    candidates = manifest.get("candidates")
    if not isinstance(interface, dict) or not isinstance(candidates, list):
        raise FormalSixAuditError("formal-six interface is absent")
    if (
        interface.get("status") != "sealed_all_six_native_candidates"
        or interface.get("required_candidate_ids_in_order") != IDS
        or interface.get("candidate_complete_schema_required") != FORMAL_SCHEMA
        or interface.get("observer_scope") != "post_generation_action_measurement_only"
        or [row.get("candidate_id") for row in candidates] != IDS
    ):
        raise FormalSixAuditError("formal-six identity or status differs")
    gates = interface.get("external_gate_contract")
    if gates != {
        "appearance_quality_gate_external_required": True,
        "appearance_quality_gate_passed": None,
        "single_bottle_gate_external_required": True,
        "single_bottle_gate_passed": None,
        "single_bottle_gate_definition": "exactly one bottle remains the same source bottle throughout; no duplicate, replacement, or extra bottle",
        "selection_authorized": False,
    }:
        raise FormalSixAuditError("external gate contract differs")

    receipt_rows = interface.get("formal_seed_receipts")
    if not isinstance(receipt_rows, list) or [row.get("seed") for row in receipt_rows] != [2027, 2028]:
        raise FormalSixAuditError("formal receipt closure differs")
    receipts: dict[int, dict[str, Any]] = {}
    for row in receipt_rows:
        if (
            set(row)
            != {
                "seed",
                "path",
                "sha256",
                "schema_version",
                "mechanical_provenance_released",
                "scientific_claim_authorized",
            }
            or row.get("schema_version") != FORMAL_SCHEMA
            or row.get("mechanical_provenance_released") is not True
            or row.get("scientific_claim_authorized") is not False
        ):
            raise FormalSixAuditError("formal receipt authority differs")
        receipt_path = regular_exact(row["path"], row["sha256"], f"seed{row['seed']} receipt")
        receipt = read_json(receipt_path)
        if (
            receipt.get("schema_version") != FORMAL_SCHEMA
            or receipt.get("scientific_claim_authorized") is not False
            or receipt.get("production_claim_forbidden") is not True
            or receipt.get("interpretation", {}).get("training_performed") is not False
            or receipt.get("freeze_certificate", {}).get("trainable_parameter_elements") != 0
            or receipt.get("freeze_certificate", {}).get("trainable_parameter_tensors") != 0
            or receipt.get("freeze_certificate", {}).get("lora_module_count") != 0
        ):
            raise FormalSixAuditError("formal receipt semantics differ")
        receipts[int(row["seed"])] = receipt

    for candidate in candidates:
        identifier = candidate["candidate_id"]
        prefix, seed_text = identifier.split("_s", 1)
        output = receipts[int(seed_text)].get("outputs", {}).get(ARM_BY_PREFIX[prefix])
        if not isinstance(output, dict):
            raise FormalSixAuditError(f"{identifier} formal output is absent")
        if (
            output.get("path") != candidate.get("path")
            or output.get("sha256") != candidate.get("sha256")
            or [output.get(key) for key in ("frame_count", "fps", "width", "height")]
            != [81, 25, 656, 368]
        ):
            raise FormalSixAuditError(f"{identifier} receipt binding differs")
        regular_exact(candidate["path"], candidate["sha256"], identifier)

    output_root = Path(manifest["output_root"])
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise FormalSixAuditError("fresh absolute observer output is required")
    authority = manifest["authority"]
    if (
        authority.get("generator_process_reads_manifest") is not False
        or authority.get("generator_process_reads_target_action") is not False
        or authority.get("generator_process_reads_real_target_media") is not False
        or authority.get("observer_calls_generator") is not False
        or authority.get("training_authorized") is not False
        or authority.get("optimizer_updates") != 0
    ):
        raise FormalSixAuditError("observer isolation authority differs")

    return {
        "schema_version": "mev840-candidate-action-observer-formal6-preflight-v1",
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "runner_path": str(runner_path),
        "runner_sha256": runner_sha256,
        "candidate_ids": IDS,
        "candidate_count": 6,
        "formal_receipt_schema": FORMAL_SCHEMA,
        "formal_receipt_count": 2,
        "output_root": str(output_root),
        "output_root_fresh": True,
        "single_gpu_serial_observer": True,
        "generator_calls": 0,
        "optimizer_updates": 0,
        "appearance_quality_gate_passed": None,
        "single_bottle_gate_passed": None,
        "selection_authorized": False,
        "scientific_claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.manifest, args.runner, args.expected_runner_sha256),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
