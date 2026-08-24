#!/usr/bin/env python3
"""Fail-closed CPU specificity gate for one frozen SAIL teacher per family.

This audit does not search seeds, blocks, schedules, losses, or thresholds.  It
uses the already sealed positive residual as the teacher and asks whether three
pre-existing counterfactual controls (same-video reverse, same-video phase
shuffle, and semantic wrong motion) are separated by the fixed SIRM score.
Only a detached JSON receipt is produced; no model or editor parameter changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_starc_core4_hidden_v1 as materializer  # noqa: E402
import self_imagined_relational_motion as relational  # noqa: E402


SCHEMA_VERSION = "bernini-sail-relational-specificity-audit-v1"
ARM_SCHEMA_VERSION = "bernini-starc-core4-same-state-hidden-arm-v1"
TENSOR_KEY = "sketched_action_minus_noop_hidden_residual"
CORE_CONTROL_ROLES = (
    "same_video_reverse",
    "same_video_phase_shuffle",
    "semantic_generic_wrong_motion",
)
REPORT_ONLY_ROLES = (
    "same_video_freeze_first",
    "semantic_noop",
    "semantic_camera_only",
    "semantic_appearance_only",
)
ALL_ROLES = ("positive", *CORE_CONTROL_ROLES, *REPORT_ONLY_ROLES)
MINIMUM_CORE_CONTROL_MISMATCH = 0.05
_FAMILY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SIRMSpecificityAuditError(RuntimeError):
    """The frozen specificity receipt cannot be authenticated or separated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    """Match the sealed STARC FP32 tensor-value digest exactly."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise SIRMSpecificityAuditError("Torch runtime unavailable") from error
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise SIRMSpecificityAuditError("tensor hash requires a real tensor")
    owned = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in owned.shape],
        "dtype": str(owned.dtype),
        "layout": str(owned.layout),
    }
    raw = owned.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    return hashlib.sha256(_canonical_json_bytes(metadata) + b"\x00" + raw).hexdigest()


def _strict_json(path: Path) -> Mapping[str, Any]:
    def pairs(rows: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise SIRMSpecificityAuditError(
                    f"duplicate key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SIRMSpecificityAuditError(
                    f"non-finite JSON number {token} in {path}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SIRMSpecificityAuditError(f"invalid receipt {path}") from error
    if not isinstance(value, Mapping):
        raise SIRMSpecificityAuditError(f"receipt root is not an object: {path}")
    return value


def _plain_absolute_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path.is_symlink():
        raise SIRMSpecificityAuditError(f"{label} must be an absolute plain directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SIRMSpecificityAuditError(f"{label} is unavailable") from error
    if resolved != path or not path.is_dir():
        raise SIRMSpecificityAuditError(f"{label} must be canonical and plain")
    return path


def _parse_cell(value: str) -> tuple[str, Path]:
    family, separator, raw_path = value.partition("=")
    if not separator or _FAMILY_RE.fullmatch(family) is None:
        raise argparse.ArgumentTypeError("cell must be FAMILY=/absolute/cell-root")
    try:
        path = _plain_absolute_directory(raw_path, label=f"{family} cell root")
    except SIRMSpecificityAuditError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return family, path


def core_specificity_passes(
    mismatch_by_role: Mapping[str, float],
) -> bool:
    """Apply the fixed, non-CLI core-control separation threshold."""

    if set(CORE_CONTROL_ROLES) - set(mismatch_by_role):
        return False
    return all(
        isinstance(mismatch_by_role[role], (int, float))
        and not isinstance(mismatch_by_role[role], bool)
        and math.isfinite(float(mismatch_by_role[role]))
        and float(mismatch_by_role[role]) >= MINIMUM_CORE_CONTROL_MISMATCH
        for role in CORE_CONTROL_ROLES
    )


def _load_arm(cell_root: Path, role: str) -> tuple[Any, Mapping[str, Any], dict[str, str]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise SIRMSpecificityAuditError("Torch/safetensors runtime unavailable") from error

    arm_root = cell_root / role
    if arm_root.is_symlink() or not arm_root.is_dir():
        raise SIRMSpecificityAuditError(f"missing plain arm directory: {arm_root}")
    receipt_path = arm_root / "starc-block15-hidden-arm-receipt-v1.json"
    artifact_path = arm_root / "starc-block15-hidden-residual.safetensors"
    for path in (receipt_path, artifact_path):
        if path.is_symlink() or not path.is_file():
            raise SIRMSpecificityAuditError(f"missing plain arm artifact: {path}")
    receipt = _strict_json(receipt_path)
    checked = dict(receipt)
    declared_digest = checked.pop("receipt_digest", None)
    if (
        checked.get("schema_version") != ARM_SCHEMA_VERSION
        or not isinstance(declared_digest, str)
        or _object_sha256(checked) != declared_digest
    ):
        raise SIRMSpecificityAuditError(f"sealed receipt validation failed for {role}")
    checked["receipt_digest"] = declared_digest
    try:
        checked = materializer.validate_arm_receipt(checked, verify_artifact=True)
    except Exception as error:
        raise SIRMSpecificityAuditError(
            f"full sealed materializer contract failed for {role}"
        ) from error
    artifact = checked.get("artifact")
    if (
        checked.get("role") != role
        or not isinstance(artifact, Mapping)
        or Path(str(artifact.get("path"))) != artifact_path
        or artifact.get("file_sha256") != _file_sha256(artifact_path)
    ):
        raise SIRMSpecificityAuditError(f"receipt/path closure differs for {role}")
    with safe_open(str(artifact_path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [TENSOR_KEY]:
            raise SIRMSpecificityAuditError(f"tensor key closure differs for {role}")
        tensor = opened.get_tensor(TENSOR_KEY)
    if (
        artifact.get("tensor_key") != TENSOR_KEY
        or artifact.get("tensor_shape") != [1, 21, 16, 1536]
        or artifact.get("tensor_dtype") != "torch.float32"
        or artifact.get("detached_finite_fp32") is not True
        or tensor.dtype != torch.float32
        or tuple(int(item) for item in tensor.shape) != (1, 21, 16, 1536)
        or not bool(torch.isfinite(tensor).all().item())
        or _tensor_sha256(tensor) != artifact.get("tensor_sha256")
    ):
        raise SIRMSpecificityAuditError(f"tensor value closure differs for {role}")
    hashes = {
        "receipt_file_sha256": _file_sha256(receipt_path),
        "receipt_digest": str(checked["receipt_digest"]),
        "artifact_file_sha256": str(artifact["file_sha256"]),
        "tensor_sha256": str(artifact["tensor_sha256"]),
    }
    return tensor.contiguous(), checked, hashes


def audit_cell(family: str, cell_root: Path) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for role in ALL_ROLES:
        tensors[role], receipts[role], hashes[role] = _load_arm(cell_root, role)

    positive = receipts["positive"]
    if (
        positive.get("split") != "fit"
        or positive.get("label") != 1
        or positive.get("training_performed") is not False
        or positive.get("optimizer_authorized") is not False
    ):
        raise SIRMSpecificityAuditError(f"{family} positive authority differs")
    prompt_digest = positive["prompt_binding"]["prompt_pair_digest"]
    model_binding = positive["model_binding"]
    cell_identity_fields = (
        "group_id",
        "episode_id",
        "split",
        "action_family_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "seed",
    )
    query_contract_fields = (
        "native_schedule_index",
        "sigma",
        "native_timestep",
        "action_and_noop_share_exact_x_sigma_object",
        "action_and_noop_share_exact_noisy_latents_object",
        "action_and_noop_share_exact_rotary_object",
        "action_and_noop_share_exact_timestep_object",
        "shared_tensor_bytes_unchanged",
        "block0_input_and_attn1_exact_parity",
        "source_condition_consumed",
        "mask_flow_pose_track_or_trajectory_consumed",
        "event_labels_consumed",
    )
    hidden_geometry_fields = (
        "hook_coordinate",
        "ulysses_world",
        "latent_shape",
        "patch_positions",
        "patch_grid_height_width",
        "patch_flatten_order",
        "action_global_sketch_shape",
        "noop_global_sketch_shape",
        "residual_shape",
        "residual_dtype",
        "rank_local_sketch_then_all_reduce",
        "full_hidden_persisted",
    )
    positive_query = positive.get("same_state_query_binding")
    positive_hidden = positive.get("hidden_binding")
    if not isinstance(positive_query, Mapping) or not isinstance(
        positive_hidden, Mapping
    ):
        raise SIRMSpecificityAuditError(f"{family} positive query binding is absent")
    query_contract = {
        key: positive_query.get(key) for key in query_contract_fields
    }
    hidden_geometry = {
        key: positive_hidden.get(key) for key in hidden_geometry_fields
    }
    for role in ALL_ROLES[1:]:
        control = receipts[role]
        control_query = control.get("same_state_query_binding")
        control_hidden = control.get("hidden_binding")
        if (
            any(control.get(key) != positive.get(key) for key in cell_identity_fields)
            or control.get("prompt_binding") != positive.get("prompt_binding")
            or control.get("model_binding") != model_binding
            or control.get("spatial_sketch_binding")
            != positive.get("spatial_sketch_binding")
            or not isinstance(control_query, Mapping)
            or {
                key: control_query.get(key) for key in query_contract_fields
            }
            != query_contract
            or not isinstance(control_hidden, Mapping)
            or {
                key: control_hidden.get(key) for key in hidden_geometry_fields
            }
            != hidden_geometry
        ):
            raise SIRMSpecificityAuditError(
                f"{family}/{role} is not the same frozen query cell"
            )

    scorer = relational.FrozenRelationalMotionScorer(tensors["positive"])
    if list(scorer.parameters()) or scorer.training:
        scorer.eval()
    if list(scorer.parameters()) or scorer.training:
        raise SIRMSpecificityAuditError("parameter-free scorer closure differs")

    rows: dict[str, dict[str, Any]] = {}
    for role in ALL_ROLES:
        result = scorer(tensors[role], require_input_grad=False)
        components = dict(scorer.last_score_components or {})
        if not components or not math.isfinite(float(result.score.detach().item())):
            raise SIRMSpecificityAuditError(f"non-finite score for {family}/{role}")
        rows[role] = {**hashes[role], **components}
    positive_row = rows["positive"]
    if (
        abs(float(positive_row["score"])) > 1.0e-12
        or abs(float(positive_row["objective_mismatch"])) > 1.0e-12
        or abs(float(positive_row["meaningful_mismatch"])) > 1.0e-12
    ):
        raise SIRMSpecificityAuditError(f"{family} positive is not an exact fixed point")
    mismatch = {
        role: float(rows[role]["meaningful_mismatch"])
        for role in CORE_CONTROL_ROLES
    }
    return {
        "family": family,
        "cell_root": str(cell_root),
        "episode_id": positive["episode_id"],
        "prompt_pair_digest": prompt_digest,
        "cell_identity_binding": {
            key: positive[key] for key in cell_identity_fields
        },
        "same_state_query_contract": query_contract,
        "hidden_geometry_contract": hidden_geometry,
        "spatial_sketch_binding": positive["spatial_sketch_binding"],
        "all_controls_same_episode_prompt_model_sketch_geometry_and_query_contract": True,
        "core_control_mismatch_floor": MINIMUM_CORE_CONTROL_MISMATCH,
        "core_control_mismatch": mismatch,
        "core_specificity_passed": core_specificity_passes(mismatch),
        "arms": rows,
    }


def run(args: argparse.Namespace) -> int:
    if len(args.cell) != 2 or len({family for family, _ in args.cell}) != 2:
        raise SIRMSpecificityAuditError("exactly two distinct families are required")
    output = Path(args.output)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise SIRMSpecificityAuditError("output must be a fresh absolute plain-file path")
    cells = [audit_cell(family, cell_root) for family, cell_root in args.cell]
    all_passed = all(row["core_specificity_passed"] is True for row in cells)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "audit_source_file_sha256": _file_sha256(Path(__file__).resolve()),
        "relational_core_file_sha256": _file_sha256(
            Path(relational.__file__).resolve()
        ),
        "fixed_core_control_roles": list(CORE_CONTROL_ROLES),
        "fixed_report_only_roles": list(REPORT_ONLY_ROLES),
        "minimum_core_control_mismatch": MINIMUM_CORE_CONTROL_MISMATCH,
        "threshold_cli_exposed": False,
        "seed_block_schedule_or_loss_selection_performed": False,
        "training_performed": False,
        "editor_update_authorized": False,
        "cells": cells,
        "all_families_core_specificity_passed": all_passed,
        "endpoint_vjp_submission_authorized": all_passed,
    }
    sealed = {**unsigned, "receipt_digest": _object_sha256(unsigned)}
    payload = _canonical_json_bytes(sealed) + b"\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(sealed, sort_keys=True, indent=2))
    return 0 if all_passed else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        type=_parse_cell,
        metavar="FAMILY=/ABS/CELL_ROOT",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_ROLES",
    "CORE_CONTROL_ROLES",
    "MINIMUM_CORE_CONTROL_MISMATCH",
    "REPORT_ONLY_ROLES",
    "SCHEMA_VERSION",
    "SIRMSpecificityAuditError",
    "audit_cell",
    "build_parser",
    "core_specificity_passes",
    "main",
    "run",
]
