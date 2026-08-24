#!/usr/bin/env python3
"""PAIR-v7 Phase-A real Bernini single-cell gradient-geometry audit.

This executable computes, but never applies, one exact40-cell Action-LoRA
gradient geometry on WORLD8 arranged as DP2 x Ulysses-SP4:

* each DP arm obtains an authoritative pure-T2V CAGD gradient from its own
  event-qualified exact81 proposal trajectory;
* each arm obtains eight source-native exact81 feature-sketch VJPs from the
  exact deployed Bernini ``v2v_apg`` operator: four no-op identity rows and
  four camera-only-minus-noop rows;
* SP4 replicas are averaged first, then both DP arms exchange the unprojected
  gradients; one union identity span is projected exactly once;
* the shared result must remain a descent direction for each action family.

There is no optimizer, parameter update, adapter checkpoint, mask, flow, pose,
track, trajectory, cross-coordinate visual carrier, or action-success claim.
Phase A is deliberately one preregistered cell: native schedule index 33.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import tarfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import dclr_runtime_contract as t2v_runtime  # noqa: E402
import infer_lora as deployment_infer  # noqa: E402
import infer_native_identity_generation_canary as native_infer  # noqa: E402
import infer_source_kv_carrier_oracle as checkpoint_audit  # noqa: E402
import pair_v5_action_adapter as action_adapter  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v5_t2v_guidance_distill as cagd  # noqa: E402
import pair_v7_fit_only_geometry_authority as fit_authority  # noqa: E402
import pair_v7_dual_coordinate_nullspace_transport as nullspace  # noqa: E402
import source_self_native_ref_contrastive_v3 as native  # noqa: E402
import source_self_runtime as distributed_runtime  # noqa: E402
import train_lora as legacy  # noqa: E402


METHOD_NAME = "bernini-pair-v7-phase-a-real-gradient-geometry-audit"
RUN_RECEIPT_SCHEMA = "bernini-pair-v7-phase-a-geometry-audit-v3"
ACTION_GRADIENT_RECEIPT_SCHEMA = "bernini-pair-v7-phase-a-action-gradient-v1"
IDENTITY_VJP_RECEIPT_SCHEMA = "bernini-pair-v7-phase-a-identity-vjp-v2"
UNION_RECEIPT_SCHEMA = "bernini-pair-v7-phase-a-dp2-union-projection-v2"
GAUGE_RECEIPT_SCHEMA = "bernini-pair-v7-phase-a-fixed-a-b-only-gauge-v1"
SP4_VJP_BUNDLE_SCHEMA = "bernini-pair-v7-phase-a-sp4-vjp-bundle-v1"
WORLD_UNION_INPUT_SCHEMA = "bernini-pair-v7-phase-a-world-union-input-v1"
WORLD_UNION_AUTHORITY_SCHEMA = "bernini-pair-v7-phase-a-world-union-authority-v1"
WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
FRAME_COUNT = 81
FPS = 25.0
DEFAULT_SOURCE_NOISE_SEED = 20260808
FIXED_ACTION_LORA_INIT_SEED = 720260808
IDENTITY_SKETCHES_PER_FAMILY = 4
APG_GUIDANCE_MODE = "v2v_apg"
APG_GUIDANCE_SCALE = 4.0
APG_ETA = 0.5
APG_NORM_THRESHOLD = 50.0
APG_MOMENTUM = 0.0
DEPLOYMENT_FLOW_SHIFT = 5.0
VJP_RTOL = 2.0e-5
VJP_ATOL = 2.0e-5
FIRST_PHASE_A_SCHEDULE_INDEX = fit_authority.FIRST_SCHEDULE_INDEX

_NO_UPDATE_CLAIMS = {
    "global_population_go": False,
    "optimizer_authorized": False,
    "parameter_update_authorized": False,
    "action_success_claimed": False,
}

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RUNTIME_ARCHIVE_REQUIRED = frozenset(
    {
        "methods/bernini_action_editing/audit_pair_v7_phase_a_geometry.py",
        "methods/bernini_action_editing/pair_v7_dual_coordinate_nullspace_transport.py",
        "methods/bernini_action_editing/pair_v7_fit_only_geometry_authority.py",
        "methods/bernini_action_editing/pair_v5_t2v_guidance_distill.py",
        "methods/bernini_action_editing/pair_v5_action_adapter.py",
        "methods/bernini_action_editing/pair_v5_native_bridge.py",
        "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
        "methods/bernini_action_editing/mace_candidate_action_energy.py",
        "methods/bernini_action_editing/dclr_runtime_contract.py",
        "methods/bernini_action_editing/inference_sigma_strata.py",
        "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
        "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
        "methods/bernini_action_editing/source_self_native_target_adapter.py",
        "methods/bernini_action_editing/source_self_runtime.py",
        "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
        "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
        "methods/bernini_action_editing/infer_lora.py",
        "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
        "methods/bernini_action_editing/source_kv_replay.py",
        "methods/bernini_action_editing/source_kv_route_batches.py",
        "methods/bernini_action_editing/source_value_residual.py",
        "methods/bernini_action_editing/train_lora.py",
        "methods/bernini_action_editing/tests/test_pair_v5_action_adapter.py",
    }
)


class PairV7PhaseAError(RuntimeError):
    """The Phase-A audit cannot make an unambiguous read-only measurement."""


def canonical_json_bytes(value: Any) -> bytes:
    return nullspace.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return nullspace.object_sha256(value)


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise PairV7PhaseAError("receipt is already sealed")
    value = dict(unsigned)
    for field, expected in _NO_UPDATE_CLAIMS.items():
        if field in value and value[field] is not expected:
            raise PairV7PhaseAError(f"{field} must remain false")
        value[field] = expected
    return {**value, "receipt_digest": object_sha256(value)}


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1_RE if length == 40 else _SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PairV7PhaseAError(f"{label} must be lowercase SHA-{length}")
    return value


def _plain_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV7PhaseAError(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV7PhaseAError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _file_sha256(path: Path) -> str:
    return distributed_runtime.file_sha256(path)


@dataclass(frozen=True)
class PhaseAPreflight:
    action_manifest: Any
    selected_action_events: tuple[Any, Any]
    selected_action_specs: tuple[Any, Any]
    fit_geometry_authority: fit_authority.FitOnlyGeometryAuthority
    checkpoint_identity: Mapping[str, Any]
    runtime_archive_path: Path
    runtime_archive_sha256: str
    runtime_source_revision: str
    evidence_method_archive_path: Path
    evidence_method_archive_sha256: str
    evidence_method_source_revision: str


@dataclass(frozen=True)
class FixedBOnlyGauge:
    all_named: tuple[tuple[str, Any], ...]
    frozen_a_named: tuple[tuple[str, Any], ...]
    trainable_b_named: tuple[tuple[str, Any], ...]
    initial_full_state_sha256: str
    initial_a_state_sha256: str
    initial_b_state_sha256: str
    receipt: Mapping[str, Any]

    def full_state_mapping(self) -> dict[str, Any]:
        return {name: parameter for name, parameter in self.all_named}


@dataclass(frozen=True)
class PhaseASourceCoordinate:
    x_sigma: Any
    timestep: Any
    sigma: float
    schedule_index: int
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class DeploymentV2VPack:
    """The visual pack used by the comparison ``infer_lora.py`` path."""

    video: Any
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class PostAPGMeasurement:
    negative_raw: Any
    condition_raw: Any
    guided_velocity: Any


@dataclass(frozen=True)
class PhaseAUnionResult:
    transport: Any
    geometry_audit_passed: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class PhaseAWorldUnionResult:
    geometry_audit_passed: bool
    receipt: Mapping[str, Any]
    transport_receipt: Mapping[str, Any]
    authority_receipt: Mapping[str, Any]


def _validate_git_archive(
    path_value: str | Path,
    *,
    expected_sha256: str,
    expected_revision: str,
    label: str,
    required_members: Sequence[str] = (),
) -> Mapping[str, Any]:
    path = _plain_file(path_value, label=label)
    expected = _sha(expected_sha256, length=64, label=f"{label} SHA")
    revision = _sha(expected_revision, length=40, label=f"{label} revision")
    if _file_sha256(path) != expected:
        raise PairV7PhaseAError(f"{label} SHA-256 differs")
    seen: set[str] = set()
    try:
        with tarfile.open(path, "r:*") as handle:
            members = handle.getmembers()
            archive_revision = handle.pax_headers.get("comment")
            for member in members:
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise PairV7PhaseAError(f"{label} contains an unsafe member")
                seen.add(pure.as_posix().lstrip("./"))
    except (tarfile.TarError, OSError) as error:
        raise PairV7PhaseAError(f"{label} cannot be audited") from error
    if archive_revision != revision:
        raise PairV7PhaseAError(f"{label} commit identity differs")
    required = frozenset(required_members)
    missing = sorted(required - seen)
    if missing:
        raise PairV7PhaseAError(f"{label} lacks required closure: {missing}")
    value = {
        "path": str(path),
        "file_sha256": expected,
        "git_archive_revision": revision,
        "archive_role": label,
        "required_member_count": len(required),
        "required_member_closure_present": True,
    }
    return _seal(value)


def validate_runtime_archive(
    path_value: str | Path,
    *,
    expected_sha256: str,
    expected_revision: str,
) -> Mapping[str, Any]:
    return _validate_git_archive(
        path_value,
        expected_sha256=expected_sha256,
        expected_revision=expected_revision,
        label="runtime source archive",
        required_members=tuple(_RUNTIME_ARCHIVE_REQUIRED),
    )


def phase_a_schedule_policy(schedule_index: int) -> Mapping[str, Any]:
    if type(schedule_index) is not int or schedule_index != FIRST_PHASE_A_SCHEDULE_INDEX:
        raise PairV7PhaseAError(
            f"Phase-A schedule index must be {FIRST_PHASE_A_SCHEDULE_INDEX}"
        )
    gate_name, gate_weight = action_adapter.sigma_gate(schedule_index)
    if gate_name == "low_base_only" or gate_weight <= 0.0:
        raise PairV7PhaseAError("Phase-A cell must expose the action adapter")
    return {
        "schedule_index": schedule_index,
        "first_phase_a_schedule_index": FIRST_PHASE_A_SCHEDULE_INDEX,
        "is_preregistered_first_phase_a_cell": True,
        "single_fit_only_geometry_cell": True,
        "gate_name": gate_name,
        "gate_weight": float(gate_weight),
        "model_callbacks_authorized": True,
        "gradient_audit_authorized": True,
        "parameter_update_authorized": False,
    }


def source_carrier_extension_contract(mode: str) -> Mapping[str, Any]:
    if mode != "none":
        raise PairV7PhaseAError(
            "source-rich carrier is not validated; Phase-A exposes only mode=none"
        )
    value = {
        "mode": "none",
        "deployment_source_video_condition_used": True,
        "deployment_image_reference_count": 0,
        "extra_carrier_condition_rows": 0,
        "extra_carrier_tensor_accepted": False,
        "extension_point_reserved_for_new_schema": True,
    }
    return _seal(value)


def _named_gradient_sha256(mapping: Mapping[str, Any]) -> str:
    layout = nullspace.GradientLayout.from_named_gradients(mapping)
    flat = layout.flatten(mapping, label="gradient digest")
    return nullspace._tensor_sha256(flat.to(dtype=__import__("torch").float32))


def configure_fixed_a_b_only_gauge(handle: Any) -> FixedBOnlyGauge:
    """Freeze random synchronized LoRA A and expose only zero-init B gradients."""

    try:
        all_named = tuple(handle.trainable_named_parameters())
    except Exception as error:
        raise PairV7PhaseAError("cannot enumerate freshly installed Action-LoRA") from error
    a_named = tuple((name, parameter) for name, parameter in all_named if "action_lora_a.weight" in name)
    b_named = tuple((name, parameter) for name, parameter in all_named if "action_lora_b.weight" in name)
    if (
        not a_named
        or len(a_named) != len(b_named)
        or len(all_named) != len(a_named) + len(b_named)
    ):
        raise PairV7PhaseAError("Action-LoRA A/B parameter closure differs")
    for _, parameter in a_named:
        parameter.requires_grad_(False)
        parameter.grad = None
    for _, parameter in b_named:
        parameter.requires_grad_(True)
        parameter.grad = None
    observed = {
        id(parameter)
        for parameter in handle.transformer.parameters()
        if parameter.requires_grad
    }
    if observed != {id(parameter) for _, parameter in b_named}:
        raise PairV7PhaseAError("fixed-gauge trainability is not exactly B-only")
    import torch

    if any(bool(torch.count_nonzero(parameter.detach()).item()) for _, parameter in b_named):
        raise PairV7PhaseAError("B-only Phase-A requires exact zero-init B")
    full_mapping = {name: parameter for name, parameter in all_named}
    a_mapping = {name: parameter for name, parameter in a_named}
    b_mapping = {name: parameter for name, parameter in b_named}
    full_digest = nullspace.named_parameter_state_sha256(full_mapping)
    a_digest = nullspace.named_parameter_state_sha256(a_mapping)
    b_digest = nullspace.named_parameter_state_sha256(b_mapping)
    unsigned = {
        "schema_version": GAUGE_RECEIPT_SCHEMA,
        "gauge": "freeze_action_lora_A_train_zero_init_B_only",
        "reason": "remove_A_to_cA_B_to_B_over_c_euclidean_parameter_gauge_ambiguity",
        "a_parameter_count": len(a_named),
        "b_parameter_count": len(b_named),
        "a_requires_grad": False,
        "b_requires_grad": True,
        "b_exact_zero_at_measurement_state": True,
        "full_parameter_state_sha256": full_digest,
        "a_parameter_state_sha256": a_digest,
        "b_parameter_state_sha256": b_digest,
        "parameter_mutation_authorized": False,
    }
    return FixedBOnlyGauge(
        all_named=all_named,
        frozen_a_named=a_named,
        trainable_b_named=b_named,
        initial_full_state_sha256=full_digest,
        initial_a_state_sha256=a_digest,
        initial_b_state_sha256=b_digest,
        receipt=_seal(unsigned),
    )


def _clear_gauge_gradients(gauge: FixedBOnlyGauge) -> None:
    for _, parameter in gauge.all_named:
        parameter.grad = None


def _assert_fixed_gauge_state(
    gauge: FixedBOnlyGauge,
    expected_sha256: str,
    *,
    label: str,
) -> str:
    expected = _sha(expected_sha256, length=64, label=f"{label} expected parameter state")
    observed = nullspace.named_parameter_state_sha256(gauge.full_state_mapping())
    if observed != expected:
        raise PairV7PhaseAError(f"{label} Action-LoRA parameter state changed")
    return observed


def _validate_frozen_a_gradients(gauge: FixedBOnlyGauge) -> None:
    if any(parameter.grad is not None for _, parameter in gauge.frozen_a_named):
        raise PairV7PhaseAError("frozen LoRA A received a gradient")


def _average_b_gradients_over_sp4(
    gauge: FixedBOnlyGauge, parallel: Any, *, label: str
) -> Mapping[str, Any]:
    import torch
    import torch.distributed as dist

    ready = all(
        parameter.grad is not None
        and parameter.grad.dtype == torch.float32
        and bool(torch.isfinite(parameter.grad).all().item())
        for _, parameter in gauge.trainable_b_named
    )
    if not distributed_runtime.world_all_true(ready, group=parallel.world_group):
        raise PairV7PhaseAError(f"{label} has missing/non-finite B gradient")
    for _, parameter in gauge.trainable_b_named:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=parallel.sp_group)
        parameter.grad.div_(float(SP_SIZE))
    result = {
        name: parameter.grad.detach().float().contiguous().clone()
        for name, parameter in gauge.trainable_b_named
    }
    digest = _named_gradient_sha256(result)
    distributed_runtime.digest_consensus(
        digest,
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label=f"{label} SP4-averaged gradient",
    )
    return result


def _bundle_sp4_vjp_receipts(
    *,
    local_receipt: Mapping[str, Any],
    averaged_gradient: Mapping[str, Any],
    parallel: Any,
    label: str,
    common_fields: Sequence[str],
) -> Mapping[str, Any]:
    """Bind every rank-local VJP receipt to the one SP4-averaged gradient."""

    import torch.distributed as dist

    if not isinstance(local_receipt, Mapping):
        raise PairV7PhaseAError(f"{label} local VJP receipt differs")
    local = dict(local_receipt)
    declared = local.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(local) != declared:
        raise PairV7PhaseAError(f"{label} local VJP receipt seal differs")
    gradient_digest = _named_gradient_sha256(averaged_gradient)
    if (
        local_receipt.get("gradient_sha256") != gradient_digest
        or local_receipt.get("sp_rank") != parallel.contract.sp_rank
    ):
        raise PairV7PhaseAError(f"{label} local VJP gradient/rank binding differs")
    gathered: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(gathered, dict(local_receipt), group=parallel.sp_group)
    if [row.get("sp_rank") for row in gathered] != list(range(SP_SIZE)):
        raise PairV7PhaseAError(f"{label} SP4 VJP rank order differs")
    expected_fields = set(local_receipt)
    for rank, row in enumerate(gathered):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise PairV7PhaseAError(f"{label} SP4 VJP receipt[{rank}] closure differs")
        unsigned = dict(row)
        digest = unsigned.pop("receipt_digest", None)
        if not isinstance(digest, str) or object_sha256(unsigned) != digest:
            raise PairV7PhaseAError(f"{label} SP4 VJP receipt[{rank}] seal differs")
        for field in common_fields:
            if row.get(field) != local_receipt.get(field):
                raise PairV7PhaseAError(
                    f"{label} SP4 VJP receipt[{rank}] common field differs: {field}"
                )
        if row.get("gradient_sha256") != gradient_digest:
            raise PairV7PhaseAError(
                f"{label} SP4 VJP receipt[{rank}] averaged gradient differs"
            )
    bundle = _seal(
        {
            "schema_version": SP4_VJP_BUNDLE_SCHEMA,
            "label": label,
            "sp_size": SP_SIZE,
            "sp_rank_receipts": gathered,
            "sp_rank_receipt_digests": [
                row["receipt_digest"] for row in gathered
            ],
            "averaged_gradient_sha256": gradient_digest,
            "checkpoint_content_receipt_digest": local_receipt[
                "checkpoint_content_receipt_digest"
            ],
            "parameter_state_sha256": local_receipt["parameter_state_sha256"],
            "all_four_rank_local_vjps_bound": True,
            "sp4_arithmetic_average_bound": True,
            "parameter_mutation_performed": False,
        }
    )
    distributed_runtime.digest_consensus(
        bundle["receipt_digest"],
        group=parallel.sp_group,
        expected_count=SP_SIZE,
        label=f"{label} SP4 VJP bundle",
    )
    return bundle


def build_mask_free_feature_sketch(
    reference_field: Any,
    *,
    family: str,
    sketch_index: int,
    seed_digest: str,
) -> tuple[Any, Mapping[str, Any]]:
    """Build one deterministic structured exact81 sketch without a region mask."""

    import torch

    if family not in nullspace.REQUIRED_IDENTITY_FAMILIES:
        raise PairV7PhaseAError("feature-sketch family differs")
    if (
        isinstance(sketch_index, bool)
        or not isinstance(sketch_index, int)
        or not 0 <= sketch_index < IDENTITY_SKETCHES_PER_FAMILY
    ):
        raise PairV7PhaseAError("feature-sketch index differs")
    seed = bytes.fromhex(_sha(seed_digest, length=64, label="feature-sketch seed"))
    if (
        not isinstance(reference_field, torch.Tensor)
        or reference_field.ndim != 5
        or tuple(int(item) for item in reference_field.shape[:3]) != (1, 16, 21)
        or reference_field.device.type == "meta"
    ):
        raise PairV7PhaseAError("feature sketch requires exact81 field [1,16,21,H,W]")
    _, channels, phases, height, width = reference_field.shape
    c = torch.arange(channels, dtype=torch.float64, device="cpu")
    t = torch.arange(phases, dtype=torch.float64, device="cpu")
    y = torch.arange(height, dtype=torch.float64, device="cpu")
    x = torch.arange(width, dtype=torch.float64, device="cpu")
    channel = torch.where(
        ((c.to(torch.int64) + int(seed[sketch_index]) + sketch_index) % 2) == 0,
        torch.ones_like(c),
        -torch.ones_like(c),
    )
    if family == "deploy_noop_identity":
        # Four deterministic low-frequency/Rademacher views of the *same*
        # deployed no-op field.  These are output observations, never extra
        # image-reference inputs.
        temporal = torch.cos(
            math.pi * float(sketch_index) * (t + 0.5) / float(phases)
        )
        vertical = torch.cos(
            math.pi
            * float(1 + sketch_index // 2)
            * (y + 0.5)
            / float(height)
        )
        horizontal = torch.cos(
            math.pi
            * float(1 + sketch_index % 2)
            * (x + 0.5)
            / float(width)
        )
        construction = "deployment_V_noop_low_frequency_rademacher"
    else:
        temporal = t - float(phases - 1) / 2.0
        x_centered = (x - float(width - 1) / 2.0) / max(float(width), 1.0)
        y_centered = (y - float(height - 1) / 2.0) / max(float(height), 1.0)
        if sketch_index == 0:
            vertical = torch.ones_like(y)
            horizontal = x_centered
            construction = "deployment_V_camera_delta_pan_x_moment"
        elif sketch_index == 1:
            vertical = y_centered
            horizontal = torch.ones_like(x)
            construction = "deployment_V_camera_delta_pan_y_moment"
        elif sketch_index == 2:
            vertical = 1.0 + y_centered.square()
            horizontal = 1.0 + x_centered.square()
            construction = "deployment_V_camera_delta_radial_scale_moment"
        else:
            vertical = torch.ones_like(y)
            horizontal = torch.ones_like(x)
            construction = "deployment_V_camera_delta_temporal_drift_moment"
    sketch64 = (
        channel.reshape(1, channels, 1, 1, 1)
        * temporal.reshape(1, 1, phases, 1, 1)
        * vertical.reshape(1, 1, 1, height, 1)
        * horizontal.reshape(1, 1, 1, 1, width)
    )
    norm = torch.linalg.vector_norm(sketch64)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise PairV7PhaseAError("feature sketch is zero/non-finite")
    sketch = (sketch64 / norm).to(
        device=reference_field.device, dtype=torch.float32
    ).contiguous()
    sketch = (sketch / torch.linalg.vector_norm(sketch.float())).detach().contiguous()
    digest = distributed_runtime.tensor_sha256(sketch)
    receipt = _seal(
        {
            "family": family,
            "sketch_index": sketch_index,
            "seed_digest": seed_digest,
            "construction": construction,
            "shape": list(sketch.shape),
            "tensor_sha256": digest,
            "fp32_l2_norm": float(torch.linalg.vector_norm(sketch).item()),
            "spatial_region_mask_used": False,
            "flow_pose_track_or_trajectory_used": False,
            "source_pixels_or_latent_used_to_choose_weights": False,
        }
    )
    return sketch, receipt


def _gradient_mapping_mean(
    components: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    import torch

    if not isinstance(components, Sequence) or len(components) != DP_SIZE:
        raise PairV7PhaseAError("action aggregate requires exact DP2 components")
    layout = nullspace.GradientLayout.from_named_gradients(components[0])
    flats = [layout.flatten(item, label=f"action component[{index}]") for index, item in enumerate(components)]
    aggregate = torch.stack(flats, dim=0).mean(dim=0)
    return layout.unflatten(aggregate, label="DP2 action arithmetic mean")


def union_project_and_audit_action_families(
    *,
    action_gradient_by_family: Mapping[str, Mapping[str, Any]],
    action_metadata_by_family: Mapping[str, Mapping[str, Any]],
    identity_probes: Sequence[nullspace.IdentityGradientProbe],
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    fit_only_geometry_authority_digest: str,
    config: nullspace.TransportConfig = nullspace.TransportConfig(),
) -> PhaseAUnionResult:
    """Union both DP arms before one projection; never local-project-average."""

    import torch

    if (
        not isinstance(action_gradient_by_family, Mapping)
        or len(action_gradient_by_family) != DP_SIZE
        or set(action_metadata_by_family) != set(action_gradient_by_family)
    ):
        raise PairV7PhaseAError("union projection requires two action families and metadata")
    families = tuple(sorted(action_gradient_by_family))
    components = [action_gradient_by_family[family] for family in families]
    component_digests = tuple(_named_gradient_sha256(item) for item in components)
    metadata = [action_metadata_by_family[family] for family in families]
    for family, digest, row in zip(families, component_digests, metadata):
        if (
            row.get("action_family") != family
            or row.get("gradient_sha256") != digest
            or row.get("checkpoint_content_receipt_digest")
            != checkpoint_content_receipt_digest
            or row.get("parameter_state_sha256") != parameter_state_sha256
        ):
            raise PairV7PhaseAError("action component metadata/state binding differs")
    expected_probe_count = (
        DP_SIZE
        * len(nullspace.REQUIRED_IDENTITY_FAMILIES)
        * IDENTITY_SKETCHES_PER_FAMILY
    )
    if len(identity_probes) != expected_probe_count:
        raise PairV7PhaseAError(
            f"union requires exactly {expected_probe_count} identity rows"
        )
    coordinate_digests = sorted(
        {probe.source_coordinate_receipt_digest for probe in identity_probes}
    )
    if len(coordinate_digests) != DP_SIZE:
        raise PairV7PhaseAError("union identity rows require exact DP2 source coordinates")
    grouped_probes: dict[tuple[str, str], list[Any]] = {}
    for probe in identity_probes:
        grouped_probes.setdefault(
            (probe.source_coordinate_receipt_digest, probe.family), []
        ).append(probe)
    expected_group_keys = {
        (coordinate, family)
        for coordinate in coordinate_digests
        for family in nullspace.REQUIRED_IDENTITY_FAMILIES
    }
    if set(grouped_probes) != expected_group_keys:
        raise PairV7PhaseAError("identity source/family group closure differs")
    for group, rows in grouped_probes.items():
        if (
            len(rows) != IDENTITY_SKETCHES_PER_FAMILY
            or len({row.feature_sketch_sha256 for row in rows})
            != IDENTITY_SKETCHES_PER_FAMILY
        ):
            raise PairV7PhaseAError(f"identity K4 sketch closure differs for {group}")
    aggregate = _gradient_mapping_mean(components)
    provenance = nullspace.ActionGradientProvenance(
        candidate_ids=tuple(str(row["candidate_id"]) for row in metadata),
        action_families=families,
        event_digests=tuple(str(row["event_digest"]) for row in metadata),
        component_gradient_sha256=component_digests,
        gradient_computation_receipt_digests=tuple(
            str(row["gradient_computation_receipt_digest"]) for row in metadata
        ),
        fit_only_geometry_authority_digest=_sha(
            fit_only_geometry_authority_digest,
            length=64,
            label="fit-only geometry authority digest",
        ),
        aggregation="arithmetic_mean_dp2_after_sp4_fit_only_geometry_gradients",
    )
    transport = nullspace.project_action_gradient_to_identity_nullspace(
        action_gradient_by_parameter=aggregate,
        action_gradient_provenance=provenance,
        identity_probes=identity_probes,
        checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
        parameter_state_sha256=parameter_state_sha256,
        config=config,
    )
    layout = transport.layout
    safe = layout.flatten(transport.safe_gradient_by_parameter, label="union-safe gradient")
    safe_norm = float(torch.linalg.vector_norm(safe).item())
    thresholds = transport.receipt["thresholds"]
    family_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    group_rank_rows: list[dict[str, Any]] = []
    for (coordinate_digest, family), probes in sorted(grouped_probes.items()):
        normalized: list[Any] = []
        for probe in probes:
            flat = layout.flatten(
                probe.gradient_by_parameter,
                label=f"identity strength {probe.probe_id}",
            ).double()
            norm = torch.linalg.vector_norm(flat)
            if float(norm.item()) > config.minimum_identity_probe_norm:
                normalized.append(flat / norm)
        if normalized:
            matrix = torch.stack(normalized, dim=0)
            eigenvalues = torch.linalg.eigvalsh(matrix @ matrix.transpose(0, 1))
            lambda_max = max(float(eigenvalues[-1].item()), 0.0)
            rank_threshold = max(
                config.eigenvalue_absolute_tolerance,
                (config.singular_value_relative_tolerance**2) * lambda_max,
            )
            group_rank = int((eigenvalues > rank_threshold).sum().item())
        else:
            eigenvalues = torch.empty(0, dtype=torch.float64)
            rank_threshold = config.eigenvalue_absolute_tolerance
            group_rank = 0
        if group_rank < 3:
            failures.append(
                f"IDENTITY_SOURCE_FAMILY_EFFECTIVE_RANK_BELOW_3:{coordinate_digest}:{family}"
            )
        group_rank_rows.append(
            {
                "source_coordinate_receipt_digest": coordinate_digest,
                "family": family,
                "probe_count": len(probes),
                "effective_rank": group_rank,
                "minimum_required_rank": 3,
                "rank_threshold": rank_threshold,
                "gram_eigenvalues": [float(value) for value in eigenvalues.tolist()],
            }
        )
    global_identity_rank = int(transport.receipt["identity_effective_rank"])
    if global_identity_rank < 8:
        failures.append("IDENTITY_GLOBAL_EFFECTIVE_RANK_BELOW_8")
    for family in families:
        gradient = layout.flatten(
            action_gradient_by_family[family], label=f"action family {family}"
        )
        norm = float(torch.linalg.vector_norm(gradient).item())
        dot = float(torch.dot(gradient, safe).item())
        cosine = dot / (norm * safe_norm) if norm > 0.0 and safe_norm > 0.0 else None
        passed = (
            dot > float(thresholds["minimum_action_descent_gain"])
            and cosine is not None
            and cosine >= float(thresholds["minimum_action_descent_cosine"])
        )
        if not passed:
            failures.append(f"PER_FAMILY_ACTION_DESCENT_FAILED:{family}")
        family_rows.append(
            {
                "action_family": family,
                "component_gradient_sha256": _named_gradient_sha256(
                    action_gradient_by_family[family]
                ),
                "component_gradient_norm": norm,
                "dot_with_union_safe_gradient": dot,
                "descent_cosine": cosine,
                "passed": passed,
            }
        )
    if not transport.geometry_authorized:
        failures.append("UNION_IDENTITY_NULLSPACE_GEOMETRY_NO_GO")
    failures = sorted(set(failures))
    passed = not failures
    receipt = _seal(
        {
            "schema_version": UNION_RECEIPT_SCHEMA,
            "method_name": METHOD_NAME,
            "geometry_audit_passed": passed,
            "optimizer_authorized": False,
            "parameter_update_authorized": False,
            "parameter_mutation_performed": False,
            "failure_codes": failures,
            "topology": "DP2xUlysses-SP4",
            "sp4_average_before_dp_exchange": True,
            "unprojected_dp2_action_gradients_exchanged": True,
            "unprojected_dp2_identity_probe_union_exchanged": True,
            "projection_count_after_union": 1,
            "local_project_then_average": False,
            "action_aggregation": (
                "arithmetic_mean_dp2_after_sp4_fit_only_geometry_gradients"
            ),
            "fit_only_geometry_authority_digest": (
                fit_only_geometry_authority_digest
            ),
            "per_family_action_descent": family_rows,
            "identity_probe_union_count": len(identity_probes),
            "identity_required_family_count": len(nullspace.REQUIRED_IDENTITY_FAMILIES),
            "identity_sketches_per_source_family": IDENTITY_SKETCHES_PER_FAMILY,
            "identity_source_coordinate_count": len(coordinate_digests),
            "identity_source_family_rank_gate": group_rank_rows,
            "identity_global_effective_rank": global_identity_rank,
            "identity_minimum_global_effective_rank": 8,
            "transport_receipt_digest": transport.receipt["receipt_digest"],
            "transport_geometry_authorized": transport.geometry_authorized,
            "checkpoint_content_receipt_digest": checkpoint_content_receipt_digest,
            "parameter_state_sha256": parameter_state_sha256,
            "scientific_action_editing_success_claim": False,
        }
    )
    return PhaseAUnionResult(transport, passed, receipt)


def _sealed_receipt_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise PairV7PhaseAError(f"{label} is not a receipt")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or object_sha256(unsigned) != declared
        or any(value.get(field) is not expected for field, expected in _NO_UPDATE_CLAIMS.items())
    ):
        raise PairV7PhaseAError(f"{label} seal differs")
    return declared


def _world_union_input_receipt(
    *,
    action_gradient_by_family: Mapping[str, Mapping[str, Any]],
    action_metadata_by_family: Mapping[str, Mapping[str, Any]],
    identity_probes: Sequence[nullspace.IdentityGradientProbe],
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    fit_only_geometry_authority_digest: str,
    config: nullspace.TransportConfig,
) -> Mapping[str, Any]:
    if (
        not isinstance(action_gradient_by_family, Mapping)
        or set(action_gradient_by_family) != set(action_metadata_by_family)
        or not isinstance(identity_probes, Sequence)
        or any(
            not isinstance(probe, nullspace.IdentityGradientProbe)
            for probe in identity_probes
        )
    ):
        raise PairV7PhaseAError("WORLD8 union input closure differs")
    action_rows = []
    for family in sorted(action_gradient_by_family):
        metadata = action_metadata_by_family[family]
        if not isinstance(metadata, Mapping):
            raise PairV7PhaseAError("WORLD8 union action metadata differs")
        action_rows.append(
            {
                "action_family": family,
                "gradient_sha256": _named_gradient_sha256(
                    action_gradient_by_family[family]
                ),
                "metadata_sha256": object_sha256(metadata),
            }
        )
    probe_rows = []
    for probe in sorted(identity_probes, key=lambda row: (row.family, row.probe_id)):
        probe.validate_metadata()
        probe_rows.append(
            {
                "probe_id": probe.probe_id,
                "family": probe.family,
                "gradient_sha256": _named_gradient_sha256(
                    probe.gradient_by_parameter
                ),
                "feature_sketch_sha256": probe.feature_sketch_sha256,
                "source_coordinate_receipt_digest": (
                    probe.source_coordinate_receipt_digest
                ),
                "gradient_computation_receipt_digest": (
                    probe.gradient_computation_receipt_digest
                ),
                "checkpoint_content_receipt_digest": (
                    probe.checkpoint_content_receipt_digest
                ),
                "parameter_state_sha256": probe.parameter_state_sha256,
            }
        )
    return _seal(
        {
            "schema_version": WORLD_UNION_INPUT_SCHEMA,
            "topology": "WORLD8-DP2xUlysses-SP4",
            "action_rows": action_rows,
            "identity_rows": probe_rows,
            "checkpoint_content_receipt_digest": checkpoint_content_receipt_digest,
            "parameter_state_sha256": parameter_state_sha256,
            "fit_only_geometry_authority_digest": (
                fit_only_geometry_authority_digest
            ),
            "transport_config": {
                name: getattr(config, name) for name in config.__dataclass_fields__
            },
            "unprojected_gradient_bytes_bound": True,
            "world_input_digest_consensus_required_before_solver": True,
            "parameter_mutation_performed": False,
        }
    )


def _cpu_named_gradient_mapping(
    gradients: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    import torch

    expected_digest = _named_gradient_sha256(gradients)
    result: dict[str, Any] = {}
    for name, tensor in gradients.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise PairV7PhaseAError(f"{label} is not a closed FP32 gradient")
        result[name] = tensor.detach().to(device="cpu").contiguous().clone()
    if (
        not result
        or any(tensor.device.type != "cpu" for tensor in result.values())
        or _named_gradient_sha256(result) != expected_digest
    ):
        raise PairV7PhaseAError(f"{label} changed during CPU authority transfer")
    return result


def _cpu_identity_probe(
    probe: nullspace.IdentityGradientProbe,
) -> nullspace.IdentityGradientProbe:
    return nullspace.IdentityGradientProbe(
        probe_id=probe.probe_id,
        family=probe.family,
        gradient_by_parameter=_cpu_named_gradient_mapping(
            probe.gradient_by_parameter, label=f"identity probe {probe.probe_id}"
        ),
        feature_sketch_sha256=probe.feature_sketch_sha256,
        source_coordinate_receipt_digest=probe.source_coordinate_receipt_digest,
        gradient_computation_receipt_digest=(
            probe.gradient_computation_receipt_digest
        ),
        checkpoint_content_receipt_digest=probe.checkpoint_content_receipt_digest,
        parameter_state_sha256=probe.parameter_state_sha256,
    )


def _gather_world_status(
    local: Mapping[str, Any], *, parallel: Any, label: str
) -> Sequence[Mapping[str, Any]]:
    import torch.distributed as dist

    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, dict(local), group=parallel.world_group)
    if (
        any(not isinstance(row, Mapping) for row in gathered)
        or [row.get("rank") for row in gathered] != list(range(WORLD_SIZE))
    ):
        raise PairV7PhaseAError(f"{label} WORLD8 status closure differs")
    return gathered


def world_rank0_cpu_union_project_and_audit_action_families(
    *,
    action_gradient_by_family: Mapping[str, Mapping[str, Any]],
    action_metadata_by_family: Mapping[str, Mapping[str, Any]],
    identity_probes: Sequence[nullspace.IdentityGradientProbe],
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
    fit_only_geometry_authority_digest: str,
    parallel: Any,
    config: nullspace.TransportConfig = nullspace.TransportConfig(),
) -> PhaseAWorldUnionResult:
    """Solve replicated geometry once on rank-0 CPU and broadcast sealed facts.

    All WORLD8 ranks first bind the exact unprojected gradient inputs.  This
    avoids treating rank-local GPU reduction roundoff as scientific geometry
    while retaining byte-level consensus on the authoritative result.
    """

    import torch.distributed as dist

    input_receipt: Optional[Mapping[str, Any]] = None
    try:
        input_receipt = _world_union_input_receipt(
            action_gradient_by_family=action_gradient_by_family,
            action_metadata_by_family=action_metadata_by_family,
            identity_probes=identity_probes,
            checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
            parameter_state_sha256=parameter_state_sha256,
            fit_only_geometry_authority_digest=fit_only_geometry_authority_digest,
            config=config,
        )
        input_status = {
            "rank": parallel.contract.rank,
            "ok": True,
            "digest": input_receipt["receipt_digest"],
            "error_type": None,
        }
    except Exception as error:
        input_status = {
            "rank": parallel.contract.rank,
            "ok": False,
            "digest": None,
            "error_type": type(error).__name__,
        }
    input_rows = _gather_world_status(
        input_status, parallel=parallel, label="Phase-A union input"
    )
    failed_input_ranks = [row["rank"] for row in input_rows if row.get("ok") is not True]
    if failed_input_ranks:
        raise PairV7PhaseAError(
            "Phase-A union input validation failed on WORLD8 ranks: "
            + ",".join(str(rank) for rank in failed_input_ranks)
        )
    input_digests = {row.get("digest") for row in input_rows}
    if len(input_digests) != 1 or None in input_digests:
        raise PairV7PhaseAError(
            "Phase-A union unprojected input differs across WORLD8 ranks"
        )
    assert input_receipt is not None
    wire: list[Any] = [None]
    if parallel.contract.rank == 0:
        try:
            cpu_actions = {
                family: _cpu_named_gradient_mapping(
                    gradients, label=f"action family {family}"
                )
                for family, gradients in action_gradient_by_family.items()
            }
            cpu_probes = tuple(
                _cpu_identity_probe(probe) for probe in identity_probes
            )
            local = union_project_and_audit_action_families(
                action_gradient_by_family=cpu_actions,
                action_metadata_by_family=action_metadata_by_family,
                identity_probes=cpu_probes,
                checkpoint_content_receipt_digest=checkpoint_content_receipt_digest,
                parameter_state_sha256=parameter_state_sha256,
                fit_only_geometry_authority_digest=fit_only_geometry_authority_digest,
                config=config,
            )
            wire[0] = {
                "ok": True,
                "error_code": None,
                "error_type": None,
                "input_receipt_digest": input_receipt["receipt_digest"],
                "geometry_audit_passed": local.geometry_audit_passed,
                "union_projection_receipt": dict(local.receipt),
                "nullspace_transport_receipt": dict(local.transport.receipt),
            }
        except Exception as error:
            wire[0] = {
                "ok": False,
                "error_code": "ROOT_CPU_UNION_SOLVE_FAILED",
                "error_type": type(error).__name__,
            }
    dist.broadcast_object_list(wire, src=0, group=parallel.world_group)
    payload = wire[0]
    if not isinstance(payload, Mapping):
        raise PairV7PhaseAError("rank-0 CPU union broadcast closure differs")
    if payload.get("ok") is not True:
        raise PairV7PhaseAError(
            str(payload.get("error_code") or "ROOT_CPU_UNION_SOLVE_FAILED")
        )
    union_receipt: Any = None
    transport_receipt: Any = None
    union_digest: Optional[str] = None
    transport_digest: Optional[str] = None
    result_digest: Optional[str] = None
    try:
        if set(payload) != {
            "ok",
            "error_code",
            "error_type",
            "input_receipt_digest",
            "geometry_audit_passed",
            "union_projection_receipt",
            "nullspace_transport_receipt",
        }:
            raise PairV7PhaseAError("rank-0 CPU union success envelope differs")
        union_receipt = payload["union_projection_receipt"]
        transport_receipt = payload["nullspace_transport_receipt"]
        union_digest = _sealed_receipt_digest(
            union_receipt, label="WORLD8 union projection receipt"
        )
        transport_digest = _sealed_receipt_digest(
            transport_receipt, label="WORLD8 nullspace transport receipt"
        )
        if (
            payload.get("input_receipt_digest")
            != input_receipt["receipt_digest"]
            or union_receipt.get("transport_receipt_digest") != transport_digest
            or payload.get("geometry_audit_passed")
            is not union_receipt.get("geometry_audit_passed")
            or union_receipt.get("transport_geometry_authorized")
            is not transport_receipt.get("geometry_authorized")
        ):
            raise PairV7PhaseAError("rank-0 CPU union result binding differs")
        result_digest = object_sha256(
            {
                "input_receipt_digest": input_receipt["receipt_digest"],
                "geometry_audit_passed": payload["geometry_audit_passed"],
                "union_projection_receipt_digest": union_digest,
                "nullspace_transport_receipt_digest": transport_digest,
            }
        )
        result_status = {
            "rank": parallel.contract.rank,
            "ok": True,
            "digest": result_digest,
            "error_type": None,
        }
    except Exception as error:
        result_status = {
            "rank": parallel.contract.rank,
            "ok": False,
            "digest": None,
            "error_type": type(error).__name__,
        }
    result_rows = _gather_world_status(
        result_status, parallel=parallel, label="Phase-A union result"
    )
    failed_result_ranks = [
        row["rank"] for row in result_rows if row.get("ok") is not True
    ]
    if failed_result_ranks:
        raise PairV7PhaseAError(
            "Phase-A rank-0 CPU union result validation failed on WORLD8 ranks: "
            + ",".join(str(rank) for rank in failed_result_ranks)
        )
    result_digests = {row.get("digest") for row in result_rows}
    if len(result_digests) != 1 or None in result_digests:
        raise PairV7PhaseAError(
            "Phase-A rank-0 CPU union result differs across WORLD8 ranks"
        )
    assert (
        union_digest is not None
        and transport_digest is not None
        and result_digest is not None
        and isinstance(union_receipt, Mapping)
        and isinstance(transport_receipt, Mapping)
    )
    authority = _seal(
        {
            "schema_version": WORLD_UNION_AUTHORITY_SCHEMA,
            "topology": "WORLD8-DP2xUlysses-SP4",
            "world_size": WORLD_SIZE,
            "authoritative_world_rank": 0,
            "solver_execution_count": 1,
            "solver_device": "cpu",
            "solver_dtype": "FP64_geometry_from_exact_FP32_gradients",
            "replicated_gpu_solver_used": False,
            "world_input_digest_consensus": True,
            "input_receipt": input_receipt,
            "input_receipt_digest": input_receipt["receipt_digest"],
            "union_projection_receipt_digest": union_digest,
            "nullspace_transport_receipt_digest": transport_digest,
            "geometry_audit_passed": bool(payload["geometry_audit_passed"]),
            "result_broadcast_to_world": True,
            "world_result_digest": result_digest,
            "world_result_digest_consensus": True,
            "parameter_mutation_performed": False,
        }
    )
    authority_digest = _sealed_receipt_digest(
        authority, label="WORLD8 CPU union authority receipt"
    )
    distributed_runtime.digest_consensus(
        authority_digest,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="Phase-A WORLD8 CPU union authority",
    )
    return PhaseAWorldUnionResult(
        geometry_audit_passed=bool(authority["geometry_audit_passed"]),
        receipt=union_receipt,
        transport_receipt=transport_receipt,
        authority_receipt=authority,
    )


def _assert_world_receipt_field_consensus(
    unsigned: Mapping[str, Any], *, parallel: Any
) -> None:
    import torch.distributed as dist

    local = {name: object_sha256(value) for name, value in unsigned.items()}
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local, group=parallel.world_group)
    if any(not isinstance(row, Mapping) or set(row) != set(local) for row in gathered):
        raise PairV7PhaseAError("Phase-A final receipt field closure differs")
    divergent = sorted(
        name for name in local if len({row[name] for row in gathered}) != 1
    )
    if divergent:
        raise PairV7PhaseAError(
            "Phase-A final receipt fields differ across ranks: "
            + ",".join(divergent)
        )


def build_task_prompt_registry(
    raw_captions: Mapping[str, str], *, prompt_cleaner: Any
) -> tuple[Mapping[str, str], Mapping[str, str], Mapping[str, Any]]:
    captions = cagd.validate_prompt_bank(raw_captions)
    prefixes = (
        *tuple(native_infer.TASK_SYSTEM_PROMPTS.values()),
        deployment_infer.MV2V_SYSTEM_PROMPT,
    )
    if any(raw.startswith(prefix) for raw in captions.values() for prefix in prefixes):
        raise PairV7PhaseAError("raw source caption is already task-prefixed")
    t2v: dict[str, str] = {}
    deployment_v2v: dict[str, str] = {}
    for branch in cagd.BRANCH_ORDER:
        t2v[branch] = native_infer.build_task_prompt(
            "t2v", captions[branch], prompt_cleaner=prompt_cleaner
        )
        deployment_v2v[branch] = deployment_infer.build_training_prompt(
            captions[branch], prompt_cleaner=prompt_cleaner
        )
    unsigned = {
        "raw_caption_bank_sha256": object_sha256(captions),
        "t2v_task_prompt_bank_sha256": object_sha256(t2v),
        "deployment_v2v_task_prompt_bank_sha256": object_sha256(deployment_v2v),
        "deployment_v2v_system_prompt_sha256": hashlib.sha256(
            deployment_infer.MV2V_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "deployment_v2v_prompt_authority": "infer_lora.build_training_prompt",
        "task_prefix_applied_exactly_once": True,
    }
    return t2v, deployment_v2v, _seal(unsigned)


def build_source_coordinate(
    source_clean_latent: Any,
    source_native_epsilon: Any,
    *,
    schedule_index: int,
    sample_id: str,
) -> PhaseASourceCoordinate:
    import torch

    phase_a_schedule_policy(schedule_index)
    for label, value in (("source latent", source_clean_latent), ("source epsilon", source_native_epsilon)):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.ndim != 5
            or tuple(value.shape[:3]) != (1, 16, 21)
            or value.requires_grad
            or not bool(torch.isfinite(value).all().item())
        ):
            raise PairV7PhaseAError(f"{label} must be detached finite exact81 FP32")
    if source_clean_latent.shape != source_native_epsilon.shape:
        raise PairV7PhaseAError("source latent/epsilon geometry differs")
    states = native.build_multi_sigma_states(
        source_clean_latent,
        source_native_epsilon,
        indices=[schedule_index],
        device=source_clean_latent.device,
    )
    x_sigma = states.noisy[0].detach().float().contiguous()
    timestep = states.timesteps[0:1].detach().float().contiguous()
    sigma = float(states.sigmas[0].item())
    unsigned = {
        "sample_id": sample_id,
        "schedule_index": schedule_index,
        "source_clean_latent_sha256": distributed_runtime.tensor_sha256(source_clean_latent),
        "source_native_epsilon_sha256": distributed_runtime.tensor_sha256(source_native_epsilon),
        "x_sigma_sha256": distributed_runtime.tensor_sha256(x_sigma),
        "sigma_float64_hex": float(sigma).hex(),
        "timestep_float32_be_hex": struct.pack("!f", float(timestep.item())).hex(),
        "construction": "(1-sigma)*source_clean_latent+sigma*source_native_epsilon",
        "pure_t2v_official_gaussian_used": False,
    }
    return PhaseASourceCoordinate(
        x_sigma=x_sigma,
        timestep=timestep,
        sigma=sigma,
        schedule_index=schedule_index,
        receipt=_seal(unsigned),
    )


def _tokenize_positive(tokenizer: Any, text: str) -> tuple[Any, Any]:
    import torch

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids, mask = encoded.input_ids, encoded.attention_mask
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] != 1:
        raise PairV7PhaseAError("positive tokenization differs")
    if ids.shape[1] >= t2v_runtime.PINNED_TEXT_TOKENS:
        return ids[:, : t2v_runtime.PINNED_TEXT_TOKENS], mask[:, : t2v_runtime.PINNED_TEXT_TOKENS]
    padding = t2v_runtime.PINNED_TEXT_TOKENS - ids.shape[1]
    return (
        torch.cat((ids, ids.new_zeros((1, padding))), dim=1),
        torch.cat((mask, mask.new_zeros((1, padding))), dim=1),
    )


def _tokenize_negative(tokenizer: Any, text: str) -> tuple[Any, Any]:
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=t2v_runtime.PINNED_TEXT_TOKENS,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if tuple(encoded.input_ids.shape) != (1, t2v_runtime.PINNED_TEXT_TOKENS):
        raise PairV7PhaseAError("negative tokenization differs")
    return encoded.input_ids, encoded.attention_mask


def _broadcast_sp(value: Any, *, parallel: Any) -> None:
    import torch.distributed as dist

    source_rank = distributed_runtime.SP_GROUP_RANKS[parallel.contract.arm_index][0]
    dist.broadcast(value, src=source_rank, group=parallel.sp_group)


def _encode_source_video(
    path: Path,
    expected_sha256: str,
    *,
    vae: Any,
    device: Any,
    parallel: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import torch
    from bernini.pipeline import _vae_encode

    pixels, metadata, digest = checkpoint_audit.prepare_hashed_source_snapshot(path)
    if (
        digest != expected_sha256
        or metadata.get("frame_count") != FRAME_COUNT
        or float(metadata.get("fps")) != FPS
    ):
        raise PairV7PhaseAError("decoded source exact81/file binding differs")
    pixels = pixels.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        clean = _vae_encode(vae, pixels).float().detach().contiguous()
    _broadcast_sp(clean, parallel=parallel)
    if tuple(clean.shape[:3]) != (1, 16, 21):
        raise PairV7PhaseAError("encoded source geometry differs")
    receipt = _seal(
        {
            "video_path": str(path),
            "video_sha256": expected_sha256,
            "clean_latent_sha256": distributed_runtime.tensor_sha256(clean),
            # The comparison baseline in infer_lora.py passes only
            # multi_video_vae_latents=[source_latent].  Keep these explicit
            # empty fields so a future reintroduction of image references
            # cannot silently change the deployed visual condition.
            "deployment_visual_condition": "source_video_only_V",
            "image_reference_count": 0,
            "reference_indices": [],
            "reference_latent_sha256": [],
            "frame_count": FRAME_COUNT,
            "fps": FPS,
        }
    )
    return clean, receipt


def _encode_native_conditions(
    renderer: Any,
    tokenizer: Any,
    raw_captions: Mapping[str, str],
    *,
    device: Any,
    parallel: Any,
) -> tuple[Mapping[str, Any], Any, Mapping[str, str], Mapping[str, str], Mapping[str, Any]]:
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    t2v_prompts, deployment_v2v_prompts, prompt_receipt = build_task_prompt_registry(
        raw_captions, prompt_cleaner=prompt_clean
    )
    native_conditions: dict[str, Any] = {}
    for branch in cagd.BRANCH_ORDER:
        ids, mask = _tokenize_positive(tokenizer, deployment_v2v_prompts[branch])
        with torch.inference_mode():
            embedding = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
        _broadcast_sp(embedding, parallel=parallel)
        native_conditions[branch] = embedding
    ids, mask = _tokenize_negative(
        tokenizer, deployment_infer.DEFAULT_NEGATIVE_PROMPT
    )
    with torch.inference_mode():
        unconditional = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
    _broadcast_sp(unconditional, parallel=parallel)
    return (
        native_conditions,
        unconditional,
        t2v_prompts,
        deployment_v2v_prompts,
        prompt_receipt,
    )


def _disable_gradient_checkpointing(renderer: Any, transformer: Any) -> Mapping[str, Any]:
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    for owner in (renderer, transformer):
        if hasattr(owner, "gradient_checkpointing"):
            setattr(owner, "gradient_checkpointing", False)
    if bool(getattr(renderer, "is_gradient_checkpointing", False)) or bool(
        getattr(transformer, "gradient_checkpointing", False)
    ):
        raise PairV7PhaseAError("gradient checkpointing remains enabled")
    return _seal(
        {
            "disabled": True,
            "reason": "branch-local Action-LoRA route context must survive serial VJP",
        }
    )


def _encode_action_prompt_bank(
    *,
    renderer: Any,
    tokenizer: Any,
    prompt_by_branch: Mapping[str, str],
    device: Any,
    parallel: Any,
) -> Mapping[str, Any]:
    import torch

    prompts = cagd.validate_prompt_bank(prompt_by_branch)
    result: dict[str, Any] = {}
    for branch in cagd.BRANCH_ORDER:
        # The frozen proposal bank was authored by
        # ``infer_native_identity_generation_canary``, whose tokenization
        # authority is ``infer_lora``.  ``legacy`` in this module is
        # ``train_lora`` and deliberately does not expose this inference
        # helper; calling it only passed static tests and then failed after the
        # eight Bernini replicas had loaded.  Bind the replay to the same
        # inference implementation that authored the proposal embeddings.
        ids, mask = deployment_infer._tokenize_training_prompt(
            tokenizer, prompts[branch]
        )
        with torch.inference_mode():
            embedding = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
        _broadcast_sp(embedding, parallel=parallel)
        if (
            tuple(embedding.shape)
            != (1, t2v_runtime.PINNED_TEXT_TOKENS, t2v_runtime.PINNED_TEXT_DIM)
            or embedding.requires_grad
            or not bool(torch.isfinite(embedding).all().item())
        ):
            raise PairV7PhaseAError(f"action prompt embedding {branch} differs")
        result[branch] = embedding
    if len({cagd.tensor_sha256(value.float()) for value in result.values()}) != len(
        result
    ):
        raise PairV7PhaseAError("two action prompt embeddings alias exactly")
    return result


class NativeT2VGeometryCallback:
    """Bernini target-only callback for one fit-only same-state query."""

    def __init__(
        self,
        *,
        diffusion: Any,
        transformer: Any,
        action_handle: Any,
        condition_by_branch: Mapping[str, Any],
        prompt_by_branch: Mapping[str, str],
        sp_rank: int,
    ) -> None:
        import torch

        if set(condition_by_branch) != set(cagd.BRANCH_ORDER):
            raise PairV7PhaseAError("text embedding branch closure differs")
        self.diffusion = diffusion
        self.transformer = transformer
        self.action_handle = action_handle
        self.condition_by_branch = dict(condition_by_branch)
        self.prompt_by_branch = cagd.validate_prompt_bank(prompt_by_branch)
        self.sp_rank = sp_rank
        self._query_id: Optional[int] = None
        self._branch: Any = None
        self._video_shape: Optional[tuple[int, ...]] = None
        self._torch = torch

    def _patch(self, query: cagd.SameStateQuery) -> None:
        torch = self._torch
        if self._query_id is not None and self._query_id != id(query):
            self._query_id = None
            self._branch = None
            self._video_shape = None
        if self._query_id is not None:
            return
        dtype = getattr(self.transformer, "dtype", None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise PairV7PhaseAError("transformer dtype differs")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patched = self.transformer.patch_vae_latent(
                query.x_sigma.to(dtype=dtype), source_id=0
            )
        if not isinstance(patched, (tuple, list)) or len(patched) != 2:
            raise PairV7PhaseAError("native T2V patch result differs")
        self._branch = t2v_runtime.build_t2v_target_branch(
            patched[0], patched[1], target_source_id=0
        )
        self._video_shape = tuple(int(item) for item in query.x_sigma.shape)
        self._query_id = id(query)

    def __call__(self, request: cagd.DenoiseRequest) -> Any:
        torch = self._torch
        if not isinstance(request, cagd.DenoiseRequest):
            raise PairV7PhaseAError("native callback request type differs")
        request.query.assert_unchanged()
        if (
            request.branch not in cagd.BRANCH_ORDER
            or self.prompt_by_branch[request.branch] != request.prompt
        ):
            raise PairV7PhaseAError("native callback prompt binding differs")
        self._patch(request.query)
        branch = self._branch
        video_shape = self._video_shape
        if branch is None or video_shape is None:
            raise PairV7PhaseAError("native same-state packet is absent")
        route = action_adapter.PairV5ActionRoute(
            total_tokens=branch.total_token_count,
            condition_tokens=0,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            branch_name="none",
            sigma_schedule_index=request.query.schedule_index,
            enabled=request.adapter_enabled,
        )
        condition = self.condition_by_branch[request.branch]
        with self.action_handle.route(route), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            packed = self.diffusion.shared_step(
                model_id="transformer_1",
                noisy_latents=branch.noisy_latents,
                timesteps=request.query.timestep,
                cond_embeds=condition,
                rotary_embs=branch.rotary_embs,
                batch_vae_seqlen=list(branch.batch_vae_seqlen),
                batch_text_seqlen=[t2v_runtime.PINNED_TEXT_TOKENS],
            )
        if (
            not isinstance(packed, torch.Tensor)
            or tuple(int(item) for item in packed.shape)
            != (1, branch.total_token_count, t2v_runtime.PINNED_PATCH_DIM)
        ):
            raise PairV7PhaseAError("native T2V prediction geometry differs")
        spatial = native_bridge._unpack_spatial_velocity(
            packed[:, -branch.target_token_count :, :], video_shape=video_shape
        )
        request.query.assert_unchanged()
        return spatial


def build_fit_only_action_query(
    event_latent: Any,
    official_epsilon: Any,
    *,
    event_spec: fit_authority.FitOnlyEventSpec,
    manifest: fit_authority.FitOnlyManifest,
    schedule_index: int,
) -> tuple[cagd.SameStateQuery, Mapping[str, Any]]:
    """Build a CAGD query from sealed fit-only geometry inputs, with no GO gate."""

    import torch

    if schedule_index != FIRST_PHASE_A_SCHEDULE_INDEX:
        raise PairV7PhaseAError(
            f"first Phase-A geometry cell is fixed at {FIRST_PHASE_A_SCHEDULE_INDEX}"
        )
    if event_spec not in manifest.events:
        raise PairV7PhaseAError("fit-only event is absent from the sealed manifest")
    for label, value, expected_digest in (
        ("event latent", event_latent, event_spec.clean_latent_tensor_sha256),
        ("official Gaussian", official_epsilon, event_spec.official_gaussian_tensor_sha256),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(int(item) for item in value.shape) != event_spec.latent_shape
            or value.device.type == "meta"
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
            or cagd.tensor_sha256(value) != expected_digest
        ):
            raise PairV7PhaseAError(f"sealed fit-only {label} binding differs")
    if event_latent.shape != official_epsilon.shape:
        raise PairV7PhaseAError("fit-only latent/Gaussian geometry differs")
    gate_name, gate_weight = action_adapter.sigma_gate(schedule_index)
    if gate_name == "low_base_only" or gate_weight <= 0.0:
        raise PairV7PhaseAError("fixed first Phase-A cell cannot be base-only")
    sigma = torch.tensor(
        [native.NATIVE_UNIPC40_SIGMAS[schedule_index]],
        dtype=torch.float32,
        device=event_latent.device,
    )
    timestep = torch.tensor(
        [native.NATIVE_UNIPC40_TIMESTEPS[schedule_index]],
        dtype=torch.float32,
        device=event_latent.device,
    )
    sigma_view = sigma.reshape(1, 1, 1, 1, 1)
    x_sigma = (
        (1.0 - sigma_view) * event_latent + sigma_view * official_epsilon
    ).detach().contiguous()
    coordinate = {
        "authority_scope": "fit_only_read_only_gradient_geometry",
        "event_id": event_spec.event_id,
        "event_digest": event_spec.event_digest,
        "manifest_digest": manifest.manifest_digest,
        "clean_t2v_latent_tensor_sha256": event_spec.clean_latent_tensor_sha256,
        "official_gaussian_tensor_sha256": event_spec.official_gaussian_tensor_sha256,
        "x_sigma_tensor_sha256": cagd.tensor_sha256(x_sigma),
        "schedule_index": schedule_index,
        "sigma_float32_be_hex": struct.pack("!f", float(sigma.item())).hex(),
        "timestep_float32_be_hex": struct.pack("!f", float(timestep.item())).hex(),
        "construction": "(1-sigma)*fit_event_t2v_y0+sigma*its_own_official_epsilon",
    }
    query = cagd.SameStateQuery(
        sample_id=event_spec.event_id,
        x_sigma=x_sigma,
        sigma=sigma,
        timestep=timestep,
        schedule_index=schedule_index,
        gate_name=gate_name,
        gate_weight=float(gate_weight),
        coordinate_digest=object_sha256(coordinate),
        x_sigma_object_id=id(x_sigma),
        sigma_object_id=id(sigma),
        timestep_object_id=id(timestep),
        x_sigma_version=int(x_sigma._version),
        sigma_version=int(sigma._version),
        timestep_version=int(timestep._version),
    )
    receipt = _seal(
        {
            **coordinate,
            "schema_version": "bernini-pair-v7-fit-only-action-query-v1",
            "prompt_bank_sha256": event_spec.prompt_bank_sha256,
            "checkpoint_tree_sha256": manifest.checkpoint_tree_sha256,
            "geometry_measurement_authorized": True,
            "guidance_eligibility_consumed": False,
            "population_confirmation_consumed": False,
        }
    )
    return query, receipt


def build_fit_only_measurement_objective(
    packet: cagd.PredictionPacket,
) -> tuple[cagd.DistillObjective, Mapping[str, Any]]:
    """Rebuild the CAGD loss from numerical primitives, without optimizer GO."""

    import torch

    if not isinstance(packet, cagd.PredictionPacket):
        raise PairV7PhaseAError("measurement requires a same-state prediction packet")
    packet.query.assert_unchanged()
    if (
        packet.query.schedule_index != FIRST_PHASE_A_SCHEDULE_INDEX
        or packet.query.gate_name == "low_base_only"
        or packet.query.gate_weight <= 0.0
    ):
        raise PairV7PhaseAError("fit-only measurement coordinate differs")
    config = cagd.DistillConfig()
    config.validate()
    teacher = cagd.build_bounded_teacher(packet.base_by_branch, config=config)
    gated_teacher = teacher.vector * packet.query.gate_weight
    action_correction = (
        packet.student_by_branch["action"].float()
        - packet.base_by_branch["action"].float()
    )
    action_match = torch.nn.functional.mse_loss(action_correction, gated_teacher)
    parity = {
        branch: torch.nn.functional.mse_loss(
            packet.student_by_branch[branch].float(),
            packet.base_by_branch[branch].float(),
        )
        for branch in cagd.NEGATIVE_BRANCHES
    }
    negative_parity = torch.stack(tuple(parity.values())).mean()
    student_rms = (
        action_correction.float().square().mean().add(config.epsilon**2).sqrt()
        - config.epsilon
    )
    trust_cap = action_correction.new_tensor(
        max(
            teacher.bounded_rms
            * packet.query.gate_weight
            * config.student_teacher_rms_ratio,
            config.minimum_teacher_rms,
        )
    )
    trust_penalty = torch.relu(student_rms - trust_cap).square()
    loss = (
        action_match
        + config.negative_parity_weight * negative_parity
        + config.trust_penalty_weight * trust_penalty
    )
    components = {
        "loss": loss,
        "action_match_loss": action_match,
        "negative_parity_loss": negative_parity,
        "trust_penalty": trust_penalty,
    }
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 0
        or not bool(torch.isfinite(value).item())
        for value in components.values()
    ):
        raise PairV7PhaseAError("fit-only measurement objective scalar differs")
    if not loss.requires_grad or loss.grad_fn is None:
        raise PairV7PhaseAError("fit-only measurement loss is not graph-connected")
    receipt = _seal(
        {
            "schema_version": "bernini-pair-v7-fit-only-measurement-objective-v1",
            "authority_scope": "backward_and_vjp_measurement_only",
            "coordinate_digest": packet.query.coordinate_digest,
            "prompt_bank_sha256": packet.prompt_bank_digest,
            "schedule_index": packet.query.schedule_index,
            "sigma_gate": packet.query.gate_name,
            "sigma_gate_weight": packet.query.gate_weight,
            "leaf_vjp_mode": packet.leaf_vjp_mode,
            "branch_order": list(cagd.BRANCH_ORDER),
            "call_order": list(packet.call_order),
            "loss_value": float(loss.detach().item()),
            "action_match_loss_value": float(
                action_match.detach().item()
            ),
            "negative_parity_loss_value": float(
                negative_parity.detach().item()
            ),
            "trust_penalty_value": float(trust_penalty.detach().item()),
            "teacher_vector_sha256": cagd.tensor_sha256(teacher.vector),
            "teacher_raw_vector_sha256": cagd.tensor_sha256(
                teacher.raw_vector
            ),
            "loss_math": (
                "action_match_plus_negative_parity_plus_bounded_trust_penalty"
            ),
            "cagd_build_distill_objective_called": False,
            "optimizer_capable_receipt_constructed": False,
            "legacy_optimizer_authority_consumed": False,
            "backward_measurement_authorized": True,
            "vjp_replay_measurement_authorized": True,
        }
    )
    objective = cagd.DistillObjective(
        loss=loss,
        action_match_loss=action_match,
        negative_parity_loss=negative_parity,
        trust_penalty=trust_penalty,
        parity_by_branch=parity,
        teacher=teacher,
        receipt=receipt,
    )
    return objective, receipt


def _fresh_source_epsilon(
    shape: Sequence[int], *, seed: int, arm_index: int, device: Any
) -> Any:
    import torch

    material = f"{seed}\0pair-v7-phase-a-source-native\0{arm_index}".encode("ascii")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device).detach()


def _build_deployment_v2v_source_pack(
    transformer: Any,
    *,
    source_video: Any,
    noisy_target: Any,
) -> DeploymentV2VPack:
    """Build exactly the no-image V pack used by ``infer_lora.py``.

    The deployed call patches source id 1 followed by target id 0.  It does
    not construct or accept image-reference tokens.
    """

    import torch

    patch = getattr(transformer, "patch_vae_latent", None)
    if not callable(patch):
        raise PairV7PhaseAError("transformer lacks patch_vae_latent")
    if (
        not isinstance(source_video, torch.Tensor)
        or not isinstance(noisy_target, torch.Tensor)
        or source_video.shape != noisy_target.shape
        or tuple(int(value) for value in source_video.shape[:3]) != (1, 16, 21)
        or source_video.device != noisy_target.device
    ):
        raise PairV7PhaseAError("deployment V pack requires two exact81 source-shaped tensors")

    dtype = getattr(transformer, "dtype", source_video.dtype)
    patched: list[tuple[str, float, Any, Any]] = []
    for role, source_id, value in (
        ("source_video", 1.0, source_video),
        ("noisy_target", 0.0, noisy_target),
    ):
        result = patch(value.to(dtype=dtype), source_id=source_id)
        if not isinstance(result, tuple) or len(result) != 2:
            raise PairV7PhaseAError("patch_vae_latent output differs")
        latent, rotary = result
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 3
            or int(latent.shape[0]) != 1
            or not isinstance(rotary, torch.Tensor)
            or rotary.ndim < 3
        ):
            raise PairV7PhaseAError("deployment V patch geometry differs")
        patched.append((role, source_id, latent, rotary))

    source_tokens, target_tokens = patched[0][2], patched[1][2]
    source_rotary, target_rotary = patched[0][3], patched[1][3]
    condition_tokens = int(source_tokens.shape[1])
    target_count = int(target_tokens.shape[1])
    if condition_tokens != target_count:
        raise PairV7PhaseAError("deployment source/target token counts differ")
    latents = torch.cat((source_tokens, target_tokens), dim=1)
    rotary = torch.cat((source_rotary, target_rotary), dim=2)
    target_mask = torch.cat(
        (
            torch.zeros(condition_tokens, dtype=torch.bool, device=latents.device),
            torch.ones(target_count, dtype=torch.bool, device=latents.device),
        )
    )
    branch = native.NativeRV2VBranch(
        name="V",
        latents=latents,
        rotary=rotary,
        target_mask=target_mask,
        total_tokens=condition_tokens + target_count,
        condition_tokens=condition_tokens,
        source_ids=(1.0, 0.0),
        concat_order=native.BRANCH_CONCAT_ORDER["V"],
    )
    receipt = _seal(
        {
            "schema_version": "bernini-pair-v7-deployment-v-pack-v1",
            "deployment_entrypoint": "infer_lora.py:model.sample",
            "visual_condition": "source_video_only_V",
            "image_reference_count": 0,
            "patch_call_roles": ["source_video", "noisy_target"],
            "patch_call_source_ids": [1.0, 0.0],
            "concat_order": list(branch.concat_order),
            "condition_tokens": condition_tokens,
            "target_tokens": target_count,
            "total_tokens": branch.total_tokens,
        }
    )
    return DeploymentV2VPack(video=branch, receipt=receipt)


def _prepare_identity_deployment_protocol(
    diffusion: Any,
    coordinate: PhaseASourceCoordinate,
) -> tuple[Any, Mapping[str, Any]]:
    """Bind APG to scheduler cell 33 without calling buggy ``_apg_sigma``."""

    import torch

    if APG_MOMENTUM != 0.0:
        raise PairV7PhaseAError("isolated Phase-A requires exact zero APG momentum")
    scheduler = getattr(diffusion, "scheduler", None)
    set_timesteps = getattr(scheduler, "set_timesteps", None)
    if not callable(set_timesteps):
        raise PairV7PhaseAError("Bernini diffusion lacks its UniPC scheduler")
    scheduler_config = getattr(scheduler, "config", None)
    configured_flow_shift = (
        scheduler_config.get("flow_shift")
        if isinstance(scheduler_config, Mapping)
        else getattr(scheduler_config, "flow_shift", None)
    )
    if (
        not isinstance(configured_flow_shift, (int, float))
        or isinstance(configured_flow_shift, bool)
        or float(configured_flow_shift) != DEPLOYMENT_FLOW_SHIFT
    ):
        raise PairV7PhaseAError("identity scheduler is not deployment flow-shift 5")
    set_timesteps(40)
    sigmas = getattr(scheduler, "sigmas", None)
    timesteps = getattr(scheduler, "timesteps", None)
    if (
        not isinstance(sigmas, torch.Tensor)
        or sigmas.device.type != "cpu"
        or sigmas.dtype != torch.float32
        or sigmas.ndim != 1
        or int(sigmas.numel()) < 40
        or not isinstance(timesteps, torch.Tensor)
        or int(timesteps.numel()) != 40
    ):
        raise PairV7PhaseAError("official UniPC40 scheduler tensors differ")
    sigma = sigmas[coordinate.schedule_index]
    timestep = timesteps[coordinate.schedule_index]
    expected_sigma = torch.tensor(
        native.NATIVE_UNIPC40_SIGMAS[coordinate.schedule_index], dtype=torch.float32
    )
    expected_timestep = torch.tensor(
        native.NATIVE_UNIPC40_TIMESTEPS[coordinate.schedule_index], dtype=torch.float32
    )
    if (
        coordinate.schedule_index != FIRST_PHASE_A_SCHEDULE_INDEX
        or not torch.equal(sigma, expected_sigma)
        or not torch.equal(timestep.float().cpu(), expected_timestep)
        or struct.pack("!f", float(sigma.item()))
        != struct.pack("!f", float(coordinate.sigma))
    ):
        raise PairV7PhaseAError("official scheduler cell differs from sealed source coordinate")
    value = _seal(
        {
            "schema_version": "bernini-pair-v7-identity-deployment-protocol-v1",
            "authority": "VideoEdit_infer_lora_frozen_deployment_contract",
            "positive_prompt_authority": "infer_lora.build_training_prompt",
            "negative_prompt_authority": "infer_lora.DEFAULT_NEGATIVE_PROMPT",
            "guidance_mode": APG_GUIDANCE_MODE,
            "visual_condition": "source_video_only_V",
            "image_reference_count": 0,
            "forward_order_per_field": ["V_negative", "V_positive"],
            "forwarded_visual_branches": ["V"],
            "omega_txt": APG_GUIDANCE_SCALE,
            "eta": APG_ETA,
            "norm_threshold": APG_NORM_THRESHOLD,
            "momentum": APG_MOMENTUM,
            "flow_shift": DEPLOYMENT_FLOW_SHIFT,
            "num_inference_steps": 40,
            "schedule_index": coordinate.schedule_index,
            "timestep": int(expected_timestep.item()),
            "sigma_source": "scheduler.sigmas[33]_cpu_fp32",
            "sigma_float32_be_hex": struct.pack("!f", float(sigma.item())).hex(),
            "fresh_zero_momentum_history_equivalent": True,
            "old_diff_vjp_coefficient": 0.0,
            "sealed_source_noised_coordinate": True,
            "single_cell_local_field_geometry": True,
            "full_sampler_trajectory_equivalent": False,
            "vendor_apg_helper_used": True,
            "alignment_scope": "visual_pack_sampler_parameters_and_post_APG_operator",
            "action_lora_scope_is_method_specific_not_infer_lora_peft_scope": True,
        }
    )
    return sigma, value


class NativeFeatureVJPRuntime:
    """Deployed V-only post-APG measurements and exact serial VJPs."""

    def __init__(
        self,
        *,
        diffusion: Any,
        transformer: Any,
        action_handle: Any,
        correct_source: Any,
        coordinate: PhaseASourceCoordinate,
        condition_by_branch: Mapping[str, Any],
        unconditional: Any,
        sp_rank: int,
    ) -> None:
        self.diffusion = diffusion
        self.transformer = transformer
        self.action_handle = action_handle
        self.correct_source = correct_source
        self.coordinate = coordinate
        self.condition_by_branch = dict(condition_by_branch)
        self.unconditional = unconditional
        self.sp_rank = sp_rank
        self.pack: DeploymentV2VPack | None = None
        self.sigma, self.deployment_protocol = _prepare_identity_deployment_protocol(
            diffusion, coordinate
        )
        self.measurement_cache: dict[str, PostAPGMeasurement] = {}

    def _pack(self) -> Any:
        import torch

        if self.pack is None:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.pack = _build_deployment_v2v_source_pack(
                    self.transformer,
                    source_video=self.correct_source,
                    noisy_target=self.coordinate.x_sigma,
                )
        return self.pack

    def _forward_branch(self, branch: Any, text: Any) -> Any:
        route = action_adapter.PairV5ActionRoute(
            total_tokens=branch.total_tokens,
            condition_tokens=branch.condition_tokens,
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            branch_name=branch.name,
            sigma_schedule_index=self.coordinate.schedule_index,
            enabled=True,
        )
        with self.action_handle.route(route):
            return native.forward_native_target_branch(
                self.diffusion,
                branch,
                timestep=self.coordinate.timestep,
                cond_embeds=text,
            )

    def _post_apg_from_raw(self, negative_raw: Any, condition_raw: Any) -> Any:
        import torch
        import importlib

        if negative_raw.dtype != torch.bfloat16 or condition_raw.dtype != torch.bfloat16:
            raise PairV7PhaseAError("pinned Bernini APG raw velocities must be BF16")
        module_name = type(self.diffusion).__module__
        if module_name != "bernini.models.wan_diffusion":
            raise PairV7PhaseAError("APG helper owner is not pinned Bernini WanDiffusion")
        vendor = importlib.import_module(module_name)
        momentum = vendor.MomentumBuffer(APG_MOMENTUM)
        negative_clean = self.coordinate.x_sigma - self.sigma * negative_raw
        condition_clean = self.coordinate.x_sigma - self.sigma * condition_raw
        guided_clean = vendor.normalized_guidance(
            pred_cond=condition_clean,
            pred_uncond=negative_clean,
            guidance_scale=APG_GUIDANCE_SCALE,
            momentum_buffer=momentum,
            eta=APG_ETA,
            norm_threshold=APG_NORM_THRESHOLD,
        )
        if momentum.momentum != 0.0:
            raise PairV7PhaseAError("APG momentum history is not stateless")
        guided_velocity = (self.coordinate.x_sigma - guided_clean) / self.sigma
        if guided_velocity.dtype != torch.float32 or not bool(
            torch.isfinite(guided_velocity).all().item()
        ):
            raise PairV7PhaseAError("post-APG deployed velocity must be finite FP32")
        return guided_velocity

    def measure_post_apg(self, *, prompt_branch: str) -> Any:
        import torch

        if prompt_branch not in {"noop", "camera_only"}:
            raise PairV7PhaseAError("identity APG prompt branch differs")
        if prompt_branch not in self.measurement_cache:
            branch = self._pack().video
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                negative_packed = self._forward_branch(branch, self.unconditional)
                condition_packed = self._forward_branch(
                    branch, self.condition_by_branch[prompt_branch]
                )
            negative = native_bridge._unpack_spatial_velocity(
                negative_packed, video_shape=self.coordinate.x_sigma.shape
            ).detach().contiguous()
            condition = native_bridge._unpack_spatial_velocity(
                condition_packed, video_shape=self.coordinate.x_sigma.shape
            ).detach().contiguous()
            with torch.no_grad():
                guided = self._post_apg_from_raw(negative, condition).detach().contiguous()
            self.measurement_cache[prompt_branch] = PostAPGMeasurement(
                negative_raw=negative,
                condition_raw=condition,
                guided_velocity=guided,
            )
        return self.measurement_cache[prompt_branch].guided_velocity

    def replay_post_apg(
        self, *, prompt_branch: str, cotangent: Any, expected: Any
    ) -> float:
        import torch

        measured = self.measurement_cache.get(prompt_branch)
        if measured is None:
            self.measure_post_apg(prompt_branch=prompt_branch)
            measured = self.measurement_cache[prompt_branch]
        negative_leaf = measured.negative_raw.detach().requires_grad_(True)
        condition_leaf = measured.condition_raw.detach().requires_grad_(True)
        rebuilt = self._post_apg_from_raw(negative_leaf, condition_leaf)
        post_maximum = float((rebuilt.detach() - expected.float()).abs().max().item())
        post_scale = float(expected.float().abs().max().item())
        if post_maximum > VJP_ATOL + VJP_RTOL * post_scale:
            raise PairV7PhaseAError("post-APG leaf replay changed measured field")
        raw_cotangents = torch.autograd.grad(
            rebuilt,
            (negative_leaf, condition_leaf),
            grad_outputs=cotangent.to(rebuilt.dtype),
        )
        branch = self._pack().video
        raw_maxima: list[float] = []
        for label, text, raw_expected, raw_cotangent in (
            ("negative", self.unconditional, measured.negative_raw, raw_cotangents[0]),
            (
                "positive",
                self.condition_by_branch[prompt_branch],
                measured.condition_raw,
                raw_cotangents[1],
            ),
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                packed = self._forward_branch(branch, text)
                spatial = native_bridge._unpack_spatial_velocity(
                    packed, video_shape=self.coordinate.x_sigma.shape
                )
            maximum = float((spatial.detach() - raw_expected).abs().max().item())
            if not torch.equal(spatial.detach(), raw_expected):
                raise PairV7PhaseAError(f"deployed {label} raw replay changed")
            if not spatial.requires_grad or spatial.grad_fn is None:
                raise PairV7PhaseAError(
                    f"deployed {label} raw replay is detached from Action-LoRA"
                )
            torch.autograd.backward(
                spatial, grad_tensors=raw_cotangent.to(spatial.dtype)
            )
            raw_maxima.append(maximum)
        return max([post_maximum, *raw_maxima])


def _identity_term_spec(family: str) -> tuple[tuple[str, str, str, float], ...]:
    if family == "deploy_noop_identity":
        return (("post_apg", "V", "noop", 1.0),)
    if family == "deploy_camera_delta":
        return (
            ("post_apg", "V", "camera_only", 1.0),
            ("post_apg", "V", "noop", -1.0),
        )
    raise PairV7PhaseAError("identity family differs")


def extract_identity_probe_gradient(
    *,
    family: str,
    sketch_index: int,
    sample_id: str,
    runtime: NativeFeatureVJPRuntime,
    gauge: FixedBOnlyGauge,
    parallel: Any,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import torch

    state_before = _assert_fixed_gauge_state(
        gauge, parameter_state_sha256, label=f"identity {family} before VJP"
    )
    _clear_gauge_gradients(gauge)
    terms = _identity_term_spec(family)
    measured: list[Any] = []
    leaves: list[Any] = []
    for kind, visual_branch, prompt_branch, _coefficient in terms:
        if kind != "post_apg" or visual_branch != "V":
            raise PairV7PhaseAError("identity term is not deployed V-only post-APG")
        value = runtime.measure_post_apg(prompt_branch=prompt_branch)
        measured.append(value)
        leaves.append(value.detach().requires_grad_(True))
    seed_digest = object_sha256(
        {
            "method": METHOD_NAME,
            "sample_id": sample_id,
            "family": family,
            "sketch_index": sketch_index,
            "coordinate_receipt_digest": runtime.coordinate.receipt["receipt_digest"],
            "identity_deployment_protocol_digest": runtime.deployment_protocol[
                "receipt_digest"
            ],
            "parameter_state_sha256": parameter_state_sha256,
        }
    )
    sketch, sketch_receipt = build_mask_free_feature_sketch(
        measured[0],
        family=family,
        sketch_index=sketch_index,
        seed_digest=seed_digest,
    )
    scalar = torch.zeros((), dtype=torch.float32, device=sketch.device)
    for leaf, (_kind, _visual_branch, _prompt_branch, coefficient) in zip(leaves, terms):
        scalar = scalar + (leaf.float() * sketch).sum() * float(coefficient)
    if not bool(torch.isfinite(scalar).item()) or not scalar.requires_grad:
        raise PairV7PhaseAError("identity feature-sketch scalar differs")
    scalar.backward()
    replay_maxima: list[float] = []
    for leaf, expected, (kind, visual_branch, prompt_branch, _coefficient) in zip(
        leaves, measured, terms
    ):
        if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all().item()):
            raise PairV7PhaseAError("identity feature-sketch output cotangent is absent")
        if kind != "post_apg" or visual_branch != "V":
            raise PairV7PhaseAError("identity replay is not deployed V-only post-APG")
        maximum = runtime.replay_post_apg(
            prompt_branch=prompt_branch,
            cotangent=leaf.grad.detach(),
            expected=expected,
        )
        replay_maxima.append(maximum)
    _validate_frozen_a_gradients(gauge)
    gradient = _average_b_gradients_over_sp4(
        gauge, parallel, label=f"identity {family}"
    )
    state_after = _assert_fixed_gauge_state(
        gauge, parameter_state_sha256, label=f"identity {family} after VJP"
    )
    gradient_digest = _named_gradient_sha256(gradient)
    deployment_pack_receipt = dict(runtime._pack().receipt)
    positive_prompt_embedding_sha256 = {
        prompt_branch: distributed_runtime.tensor_sha256(
            runtime.condition_by_branch[prompt_branch].float()
        )
        for _kind, _visual_branch, prompt_branch, _coefficient in terms
    }
    unsigned = {
        "schema_version": IDENTITY_VJP_RECEIPT_SCHEMA,
        "sample_id": sample_id,
        "family": family,
        "sketch_index": sketch_index,
        "sp_rank": parallel.contract.sp_rank,
        "terms": [
            {
                "kind": kind,
                "visual_branch": visual_branch,
                "prompt_branch": prompt_branch,
                "coefficient": coefficient,
            }
            for kind, visual_branch, prompt_branch, coefficient in terms
        ],
        "identity_deployment_protocol_digest": runtime.deployment_protocol[
            "receipt_digest"
        ],
        "deployment_v_pack_receipt": deployment_pack_receipt,
        "deployment_v_pack_receipt_digest": deployment_pack_receipt[
            "receipt_digest"
        ],
        "negative_prompt_embedding_sha256": distributed_runtime.tensor_sha256(
            runtime.unconditional.float()
        ),
        "positive_prompt_embedding_sha256_by_branch": (
            positive_prompt_embedding_sha256
        ),
        "feature_sketch_receipt_digest": sketch_receipt["receipt_digest"],
        "feature_sketch_sha256": sketch_receipt["tensor_sha256"],
        "source_coordinate_receipt_digest": runtime.coordinate.receipt["receipt_digest"],
        "checkpoint_content_receipt_digest": checkpoint_content_receipt_digest,
        "parameter_state_sha256": parameter_state_sha256,
        "parameter_state_before_vjp_sha256": state_before,
        "parameter_state_after_vjp_sha256": state_after,
        "same_parameter_state_before_and_after_vjp": True,
        "gradient_sha256": gradient_digest,
        "vjp_replay_max_abs": max(replay_maxima),
        "sp4_averaged": True,
        "dp_averaged_before_union": False,
        "mask_flow_pose_track_or_trajectory_used": False,
        "parameter_mutation_performed": False,
    }
    rank_receipt = _seal(unsigned)
    sp4_bundle = _bundle_sp4_vjp_receipts(
        local_receipt=rank_receipt,
        averaged_gradient=gradient,
        parallel=parallel,
        label=f"identity:{sample_id}:{family}",
        common_fields=(
            "schema_version",
            "sample_id",
            "family",
            "sketch_index",
            "terms",
            "identity_deployment_protocol_digest",
            "deployment_v_pack_receipt",
            "deployment_v_pack_receipt_digest",
            "negative_prompt_embedding_sha256",
            "positive_prompt_embedding_sha256_by_branch",
            "feature_sketch_receipt_digest",
            "feature_sketch_sha256",
            "source_coordinate_receipt_digest",
            "checkpoint_content_receipt_digest",
            "parameter_state_sha256",
            "parameter_state_before_vjp_sha256",
            "parameter_state_after_vjp_sha256",
            "same_parameter_state_before_and_after_vjp",
            "gradient_sha256",
        ),
    )
    return gradient, {
        "rank_receipt": rank_receipt,
        "sp4_bundle": sp4_bundle,
        "sketch": sketch_receipt,
    }


def extract_action_gradient(
    *,
    runtime_event: Any,
    manifest: Any,
    diffusion: Any,
    transformer: Any,
    action_handle: Any,
    conditions: Mapping[str, Any],
    gauge: FixedBOnlyGauge,
    parallel: Any,
    sp_rank: int,
    schedule_index: int,
    device: Any,
    checkpoint_content_receipt_digest: str,
    parameter_state_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import torch

    state_before = _assert_fixed_gauge_state(
        gauge, parameter_state_sha256, label="pure-T2V action before VJP"
    )
    _clear_gauge_gradients(gauge)
    event_latent = runtime_event.event_latent_cpu.to(device=device).contiguous()
    official_epsilon = runtime_event.official_epsilon_cpu.to(device=device).contiguous()
    _broadcast_sp(event_latent, parallel=parallel)
    _broadcast_sp(official_epsilon, parallel=parallel)
    callback = NativeT2VGeometryCallback(
        diffusion=diffusion,
        transformer=transformer,
        action_handle=action_handle,
        condition_by_branch=conditions,
        prompt_by_branch=runtime_event.spec.prompt_by_branch,
        sp_rank=sp_rank,
    )
    query, query_receipt = build_fit_only_action_query(
        event_latent,
        official_epsilon,
        event_spec=runtime_event.spec,
        manifest=manifest,
        schedule_index=schedule_index,
    )
    packet = cagd.collect_same_state_predictions(
        query,
        prompt_by_branch=runtime_event.spec.prompt_by_branch,
        denoise_callback=callback,
        leaf_vjp_mode=True,
    )
    objective, measurement_objective_receipt = (
        build_fit_only_measurement_objective(packet)
    )
    objective.loss.backward()
    replay = cagd.replay_student_vjp(
        packet,
        runtime_event.spec.prompt_by_branch,
        callback,
        rtol=VJP_RTOL,
        atol=VJP_ATOL,
    )
    _validate_frozen_a_gradients(gauge)
    gradient = _average_b_gradients_over_sp4(gauge, parallel, label="pure-T2V CAGD action")
    state_after = _assert_fixed_gauge_state(
        gauge, parameter_state_sha256, label="pure-T2V action after VJP"
    )
    digest = _named_gradient_sha256(gradient)
    unsigned = {
        "schema_version": ACTION_GRADIENT_RECEIPT_SCHEMA,
        "candidate_id": runtime_event.spec.event_id,
        "action_family": runtime_event.spec.action_family,
        "sp_rank": sp_rank,
        "event_digest": runtime_event.spec.event_digest,
        "event_latent_sha256": cagd.tensor_sha256(event_latent),
        "official_proposal_gaussian_sha256": cagd.tensor_sha256(official_epsilon),
        "fit_only_action_query_receipt_digest": query_receipt["receipt_digest"],
        "fit_only_measurement_objective_receipt_digest": (
            measurement_objective_receipt["receipt_digest"]
        ),
        "legacy_authorized_objective_receipt_referenced": False,
        "lower_level_cagd_api": [
            "collect_same_state_predictions",
            "build_bounded_teacher",
            "replay_student_vjp",
        ],
        "guidance_eligibility_consumed": False,
        "population_confirmation_consumed": False,
        "gradient_sha256": digest,
        "vjp_replay_max_abs": max(replay.values()),
        "checkpoint_content_receipt_digest": checkpoint_content_receipt_digest,
        "parameter_state_sha256": parameter_state_sha256,
        "parameter_state_before_vjp_sha256": state_before,
        "parameter_state_after_vjp_sha256": state_after,
        "same_parameter_state_before_and_after_vjp": True,
        "sp4_averaged": True,
        "dp_averaged_before_union": False,
        "pure_t2v_visual_role": "action_arm_same_coordinate_reward_trajectory_only",
        "pure_t2v_visual_used_as_rv2v_target_noise_source_or_donor": False,
        "parameter_mutation_performed": False,
    }
    rank_receipt = _seal(unsigned)
    sp4_bundle = _bundle_sp4_vjp_receipts(
        local_receipt=rank_receipt,
        averaged_gradient=gradient,
        parallel=parallel,
        label=(
            f"action:{runtime_event.spec.event_id}:"
            f"{runtime_event.spec.action_family}"
        ),
        common_fields=(
            "schema_version",
            "candidate_id",
            "action_family",
            "event_digest",
            "event_latent_sha256",
            "official_proposal_gaussian_sha256",
            "fit_only_action_query_receipt_digest",
            "fit_only_measurement_objective_receipt_digest",
            "legacy_authorized_objective_receipt_referenced",
            "guidance_eligibility_consumed",
            "population_confirmation_consumed",
            "checkpoint_content_receipt_digest",
            "parameter_state_sha256",
            "parameter_state_before_vjp_sha256",
            "parameter_state_after_vjp_sha256",
            "same_parameter_state_before_and_after_vjp",
            "gradient_sha256",
        ),
    )
    return gradient, {"rank_receipt": rank_receipt, "sp4_bundle": sp4_bundle}


def _exchange_named_mapping_dp2(
    local: Mapping[str, Any], *, parallel: Any, label: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import torch
    import torch.distributed as dist

    layout = nullspace.GradientLayout.from_named_gradients(local)
    result = [dict(), dict()]
    for name in layout.names:
        tensor = local[name]
        gathered = [torch.empty_like(tensor), torch.empty_like(tensor)]
        dist.all_gather(gathered, tensor, group=parallel.dp_group)
        for arm in range(DP_SIZE):
            result[arm][name] = gathered[arm].detach().float().contiguous()
    for arm, mapping in enumerate(result):
        if set(mapping) != set(layout.names):
            raise PairV7PhaseAError(f"{label} DP arm {arm} mapping closure differs")
    return result[0], result[1]


def _exchange_metadata_dp2(local: Mapping[str, Any], *, parallel: Any, label: str) -> tuple[Any, Any]:
    import torch.distributed as dist

    gathered: list[Any] = [None, None]
    dist.all_gather_object(gathered, dict(local), group=parallel.dp_group)
    if [item.get("arm_index") for item in gathered] != [0, 1]:
        raise PairV7PhaseAError(f"{label} DP metadata order differs")
    return gathered[0], gathered[1]


def _select_source_receipts_by_arm(
    gathered_runtime: Sequence[Any],
    manifest: fit_authority.FitOnlyManifest,
) -> list[Mapping[str, Any]]:
    """Validate SP4 source-receipt consensus and retain its complete payload."""

    source_fields = {
        "video_path",
        "video_sha256",
        "clean_latent_sha256",
        "deployment_visual_condition",
        "image_reference_count",
        "reference_indices",
        "reference_latent_sha256",
        "frame_count",
        "fps",
        *_NO_UPDATE_CLAIMS,
        "receipt_digest",
    }
    if (
        not isinstance(gathered_runtime, Sequence)
        or len(gathered_runtime) != WORLD_SIZE
        or len(manifest.events) != DP_SIZE
    ):
        raise PairV7PhaseAError("WORLD8 source receipt closure differs")
    selected: list[Mapping[str, Any]] = []
    for arm_index, event in enumerate(manifest.events):
        rows = [
            row
            for row in gathered_runtime
            if isinstance(row, Mapping) and row.get("arm_index") == arm_index
        ]
        if (
            len(rows) != SP_SIZE
            or {row.get("sp_rank") for row in rows} != set(range(SP_SIZE))
            or {row.get("rank") for row in rows}
            != set(distributed_runtime.SP_GROUP_RANKS[arm_index])
        ):
            raise PairV7PhaseAError("SP4 source receipt rank closure differs")
        receipt = rows[0].get("source_receipt")
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != source_fields
            or any(row.get("source_receipt") != receipt for row in rows)
            or any(
                row.get("source_receipt_digest") != receipt.get("receipt_digest")
                for row in rows
            )
        ):
            raise PairV7PhaseAError("SP4 source receipt consensus differs")
        unsigned = dict(receipt)
        declared = _sha(
            unsigned.pop("receipt_digest"), length=64, label="source receipt"
        )
        references = receipt.get("reference_latent_sha256")
        video_path = Path(str(receipt.get("video_path")))
        if (
            object_sha256(unsigned) != declared
            or receipt.get("video_path") != str(event.source_video.path)
            or receipt.get("video_sha256") != event.source_video.sha256
            or not video_path.is_file()
            or video_path.is_symlink()
            or _file_sha256(video_path) != event.source_video.sha256
            or receipt.get("frame_count") != FRAME_COUNT
            or receipt.get("fps") != FPS
            or receipt.get("deployment_visual_condition") != "source_video_only_V"
            or receipt.get("image_reference_count") != 0
            or receipt.get("reference_indices") != []
            or not isinstance(references, list)
            or references != []
            or any(
                not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                for value in [receipt.get("clean_latent_sha256")]
            )
            or any(receipt.get(field) is not expected for field, expected in _NO_UPDATE_CLAIMS.items())
            or any(row.get("source_sample_id") != event.source_sample_id for row in rows)
            or any(row.get("source_event_digest") != event.event_digest for row in rows)
        ):
            raise PairV7PhaseAError("decoded source receipt binding differs")
        selected.append(
            {
                "arm_index": arm_index,
                "source_sample_id": event.source_sample_id,
                "source_event_digest": event.event_digest,
                "sp4_receipt_consensus": True,
                "source_receipt": dict(receipt),
            }
        )
    return selected


def _validate_fixed_gauge_consensus(
    gauge: FixedBOnlyGauge, *, parallel: Any
) -> FixedBOnlyGauge:
    """Require deterministic rank-local construction; never broadcast parameters."""

    full = nullspace.named_parameter_state_sha256(gauge.full_state_mapping())
    distributed_runtime.digest_consensus(
        full,
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="fixed Action-LoRA gauge state",
    )
    if full != gauge.initial_full_state_sha256:
        raise PairV7PhaseAError("device move changed fixed Action-LoRA state")
    # Rebuild the receipt after the device move and world-state consensus.
    unsigned = dict(gauge.receipt)
    unsigned.pop("receipt_digest")
    unsigned["full_parameter_state_sha256"] = full
    unsigned["a_parameter_state_sha256"] = nullspace.named_parameter_state_sha256(
        {name: parameter for name, parameter in gauge.frozen_a_named}
    )
    unsigned["b_parameter_state_sha256"] = nullspace.named_parameter_state_sha256(
        {name: parameter for name, parameter in gauge.trainable_b_named}
    )
    unsigned["deterministic_initialization_seed"] = FIXED_ACTION_LORA_INIT_SEED
    unsigned["parameter_broadcast_or_copy_performed"] = False
    unsigned["world8_parameter_state_consensus"] = True
    return FixedBOnlyGauge(
        all_named=gauge.all_named,
        frozen_a_named=gauge.frozen_a_named,
        trainable_b_named=gauge.trainable_b_named,
        initial_full_state_sha256=full,
        initial_a_state_sha256=unsigned["a_parameter_state_sha256"],
        initial_b_state_sha256=unsigned["b_parameter_state_sha256"],
        receipt=_seal(unsigned),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=checkpoint_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--action-event-manifest", required=True)
    parser.add_argument("--expected-action-event-manifest-sha256", required=True)
    parser.add_argument("--cagd-validator-evidence", required=True)
    parser.add_argument("--expected-cagd-validator-evidence-sha256", required=True)
    parser.add_argument("--scorer-group-receipt", action="append", required=True)
    parser.add_argument("--expected-scorer-group-receipt-sha256", action="append", required=True)
    parser.add_argument("--runtime-source-archive", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--evidence-method-source-archive", required=True)
    parser.add_argument("--evidence-method-source-archive-sha256", required=True)
    parser.add_argument("--evidence-method-source-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schedule-index", required=True, type=int)
    parser.add_argument("--source-noise-seed", type=int, default=DEFAULT_SOURCE_NOISE_SEED)
    parser.add_argument("--source-carrier-mode", default="none", choices=("none",))
    parser.add_argument("--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--ack-root-reviewed-phase-a-launch", action="store_true")
    parser.add_argument("--ack-no-parameter-mutation-no-success-claim", action="store_true")
    return parser


def preflight(args: argparse.Namespace) -> PhaseAPreflight:
    if args.ack_root_reviewed_phase_a_launch is not True:
        raise PairV7PhaseAError("root-reviewed Phase-A launch acknowledgement is required")
    if args.ack_no_parameter_mutation_no_success_claim is not True:
        raise PairV7PhaseAError("no-mutation/no-success-claim acknowledgement is required")
    phase_a_schedule_policy(args.schedule_index)
    if args.schedule_index != FIRST_PHASE_A_SCHEDULE_INDEX:
        raise PairV7PhaseAError(
            f"first Phase-A geometry audit is fixed at schedule index "
            f"{FIRST_PHASE_A_SCHEDULE_INDEX}"
        )
    source_carrier_extension_contract(args.source_carrier_mode)
    if type(args.source_noise_seed) is not int or not 0 <= args.source_noise_seed < 2**63:
        raise PairV7PhaseAError("source noise seed differs")
    for field in ("expected_bernini_commit", "expected_veomni_commit", "runtime_source_revision", "evidence_method_source_revision"):
        _sha(getattr(args, field), length=40, label=field)
    for field in (
        "expected_checkpoint_content_manifest_sha256",
        "expected_action_event_manifest_sha256",
        "expected_cagd_validator_evidence_sha256",
        "runtime_source_archive_sha256",
        "evidence_method_source_archive_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _sha(getattr(args, field), length=64, label=field)
    if (
        len(args.scorer_group_receipt) != 2
        or len(args.expected_scorer_group_receipt_sha256) != 2
    ):
        raise PairV7PhaseAError("exactly two scorer-group receipts are required")
    for digest in args.expected_scorer_group_receipt_sha256:
        _sha(digest, length=64, label="scorer-group receipt SHA")
    if (
        args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != checkpoint_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise PairV7PhaseAError("pinned Bernini checkpoint identity differs")
    archive_receipt = validate_runtime_archive(
        args.runtime_source_archive,
        expected_sha256=args.runtime_source_archive_sha256,
        expected_revision=args.runtime_source_revision,
    )
    evidence_archive_receipt = _validate_git_archive(
        args.evidence_method_source_archive,
        expected_sha256=args.evidence_method_source_archive_sha256,
        expected_revision=args.evidence_method_source_revision,
        label="evidence method source archive",
    )
    try:
        checkpoint_identity = checkpoint_audit.validate_checkpoint_content(
            Path(args.checkpoint),
            Path(args.checkpoint_content_manifest),
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
    except Exception as error:
        raise PairV7PhaseAError(f"checkpoint content audit failed: {error}") from error
    checkpoint_content_receipt_digest = object_sha256(checkpoint_identity)
    try:
        action_manifest, action_events, geometry_authority = (
            fit_authority.validate_fit_only_geometry_authority(
                manifest_path=args.action_event_manifest,
                expected_manifest_file_sha256=(
                    args.expected_action_event_manifest_sha256
                ),
                evidence_path=args.cagd_validator_evidence,
                expected_evidence_file_sha256=(
                    args.expected_cagd_validator_evidence_sha256
                ),
                expected_checkpoint_tree_sha256=(
                    args.expected_checkpoint_tree_sha256
                ),
                checkpoint_content_identity=checkpoint_identity,
                expected_checkpoint_content_receipt_digest=(
                    checkpoint_content_receipt_digest
                ),
                expected_action_adapter_schema_sha256=(
                    cagd.ACTION_ADAPTER_SCHEMA_SHA256
                ),
                cast_method_archive_path=args.evidence_method_source_archive,
                expected_cast_method_archive_sha256=(
                    args.evidence_method_source_archive_sha256
                ),
                expected_cast_method_revision=(
                    args.evidence_method_source_revision
                ),
                cast_group_receipt_paths=list(args.scorer_group_receipt),
                expected_cast_group_receipt_sha256=list(
                    args.expected_scorer_group_receipt_sha256
                ),
            )
        )
    except Exception as error:
        raise PairV7PhaseAError(
            f"sealed fit-only geometry authority failed: {error}"
        ) from error
    event_by_id = {event.spec.event_id: event for event in action_events}
    selected_events: list[Any] = []
    selected_specs: list[Any] = []
    try:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    except Exception as error:
        raise PairV7PhaseAError("official Wan prompt cleaner is unavailable") from error
    for action_spec in action_manifest.events:
        runtime_event = event_by_id.get(action_spec.fit_candidate_id)
        if runtime_event is None:
            raise PairV7PhaseAError("fit candidate is absent from action manifest")
        if (
            runtime_event.event_latent_cpu is None
            or runtime_event.official_epsilon_cpu is None
        ):
            raise PairV7PhaseAError("fit-only runtime tensors are unavailable")
        t2v_prompts, _deployment_v2v_prompts, _receipt = build_task_prompt_registry(
            action_spec.raw_caption_by_branch, prompt_cleaner=prompt_clean
        )
        if (
            runtime_event.spec.action_family != action_spec.action_family
            or runtime_event.spec.prompt_bank_sha256 != object_sha256(t2v_prompts)
        ):
            raise PairV7PhaseAError(
                "correct-source caption/action prompt binding differs"
            )
        selected_events.append(runtime_event)
        selected_specs.append(action_spec)
    return PhaseAPreflight(
        action_manifest=action_manifest,
        selected_action_events=(selected_events[0], selected_events[1]),
        selected_action_specs=(selected_specs[0], selected_specs[1]),
        fit_geometry_authority=geometry_authority,
        checkpoint_identity=checkpoint_identity,
        runtime_archive_path=Path(archive_receipt["path"]),
        runtime_archive_sha256=archive_receipt["file_sha256"],
        runtime_source_revision=archive_receipt["git_archive_revision"],
        evidence_method_archive_path=Path(evidence_archive_receipt["path"]),
        evidence_method_archive_sha256=evidence_archive_receipt["file_sha256"],
        evidence_method_source_revision=evidence_archive_receipt[
            "git_archive_revision"
        ],
    )


def _publish_receipt_only(stage: Path, output: Path, receipt: Mapping[str, Any]) -> None:
    distributed_runtime.atomic_json(stage / "receipt.json", receipt)
    entries = list(stage.iterdir())
    if len(entries) != 1 or entries[0].name != "receipt.json" or entries[0].is_symlink():
        raise PairV7PhaseAError("Phase-A output artifact closure differs")
    os.replace(stage, output)
    distributed_runtime.fsync_directory(output.parent)


def _source_tree_file_closure(root: Path) -> Mapping[str, Any]:
    """Hash the exact tracked tree, or the complete extracted source/config tree."""

    git_directory = root / ".git"
    tracked_modes: dict[str, str] = {}
    tracked_object_ids: dict[str, str] = {}
    if git_directory.is_dir():
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--stage", "-z"],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            relative_paths = []
            for raw_record in completed.stdout.split(b"\x00"):
                if not raw_record:
                    continue
                raw_metadata, raw_path = raw_record.split(b"\t", 1)
                mode_value, object_id, stage = raw_metadata.decode("ascii").split()
                relative = raw_path.decode("utf-8")
                pure = PurePosixPath(relative)
                if (
                    stage != "0"
                    or mode_value not in {"100644", "100755", "120000"}
                    or re.fullmatch(r"[0-9a-f]{40}", object_id) is None
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or pure.as_posix() != relative
                ):
                    raise ValueError("tracked source entry differs")
                relative_paths.append(relative)
                tracked_modes[relative] = mode_value
                tracked_object_ids[relative] = object_id
        except (
            OSError,
            subprocess.SubprocessError,
            UnicodeError,
            ValueError,
        ) as error:
            raise PairV7PhaseAError("source tree tracked-file closure failed") from error
        mode = "git_tracked_files"
    else:
        allowed_suffixes = {".py", ".json", ".yaml", ".yml", ".toml"}
        relative_paths = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in allowed_suffixes
        )
        mode = "extracted_source_and_config_files"
    if not relative_paths or len(set(relative_paths)) != len(relative_paths):
        raise PairV7PhaseAError("source tree file closure differs")
    entries: list[Mapping[str, str]] = []
    root_resolved = root.resolve(strict=True)
    tracked_symlink_count = 0
    for relative in sorted(relative_paths):
        path = root / relative
        if path.is_symlink():
            if mode != "git_tracked_files" or tracked_modes.get(relative) != "120000":
                raise PairV7PhaseAError(f"source tree symlink mode differs: {relative}")
            try:
                raw_link = os.readlink(os.fsencode(path))
                link_text = raw_link.decode("utf-8")
                link_pure = PurePosixPath(link_text)
                resolved_target = (path.parent / link_text).resolve(strict=True)
                target_relative = resolved_target.relative_to(root_resolved).as_posix()
            except (OSError, UnicodeError, ValueError) as error:
                raise PairV7PhaseAError(
                    f"source tree symlink target differs: {relative}"
                ) from error
            if (
                not link_text
                or link_pure.is_absolute()
                or ".." in link_pure.parts
                or link_pure.as_posix() != link_text
                or not resolved_target.is_file()
                or tracked_modes.get(target_relative) not in {"100644", "100755"}
            ):
                raise PairV7PhaseAError(
                    f"source tree symlink target differs: {relative}"
                )
            git_blob = hashlib.sha1(
                f"blob {len(raw_link)}\0".encode("ascii") + raw_link
            ).hexdigest()
            if git_blob != tracked_object_ids.get(relative):
                raise PairV7PhaseAError(
                    f"source tree symlink object differs: {relative}"
                )
            entries.append(
                {
                    "path": relative,
                    "kind": "tracked_relative_symlink",
                    "git_mode": "120000",
                    "git_blob_sha1": git_blob,
                    "link_text_sha256": hashlib.sha256(raw_link).hexdigest(),
                    "target_path": target_relative,
                    "target_sha256": _file_sha256(resolved_target),
                }
            )
            tracked_symlink_count += 1
            continue
        if not path.is_file() or tracked_modes.get(relative) == "120000":
            raise PairV7PhaseAError(f"source tree file differs: {relative}")
        entries.append(
            {
                "path": relative,
                "kind": "regular_file",
                "git_mode": tracked_modes.get(relative, "extracted_regular_file"),
                "sha256": _file_sha256(path),
            }
        )
    return {
        "closure_mode": mode,
        "file_count": len(entries),
        "regular_file_count": len(entries) - tracked_symlink_count,
        "tracked_relative_symlink_count": tracked_symlink_count,
        "tracked_relative_symlink_link_and_target_bytes_verified": True,
        "file_entries_digest": object_sha256(entries),
    }


def _validate_and_bind_source_trees(
    args: argparse.Namespace,
) -> tuple[Path, Path, Mapping[str, Any]]:
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
    except Exception as error:
        raise PairV7PhaseAError(str(error)) from error
    receipt = _seal(
        {
            "schema_version": "bernini-pair-v7-source-tree-binding-v2",
            "bernini": {
                "root": str(bernini_root),
                "revision": bernini_revision,
                **_source_tree_file_closure(bernini_root),
            },
            "veomni": {
                "root": str(veomni_root),
                "revision": veomni_revision,
                **_source_tree_file_closure(veomni_root),
            },
            "tracked_or_extracted_source_bytes_verified": True,
        }
    )
    return bernini_root, veomni_root, receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pre = preflight(args)
    bernini_root, veomni_root, source_tree_preflight_receipt = (
        _validate_and_bind_source_trees(args)
    )
    try:
        checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise PairV7PhaseAError(str(error)) from error
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_only": True,
                    "phase_a_runtime_authorized": True,
                    "parameter_update_authorized": False,
                    "topology": "WORLD8-DP2xSP4",
                    "schedule_policy": phase_a_schedule_policy(args.schedule_index),
                    "fit_candidates": [
                        event.spec.event_id for event in pre.selected_action_events
                    ],
                    "source_tree_preflight_receipt_digest": (
                        source_tree_preflight_receipt["receipt_digest"]
                    ),
                    "legacy_self_seal_trusted": False,
                    "scientific_action_editing_success_claim": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    contract = distributed_runtime.distributed_contract()
    device = distributed_runtime.initialise_distributed(contract)
    parallel = distributed_runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = distributed_runtime.prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    renderer_config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **deployment_infer.inference_renderer_config_overrides(checkpoint),
    )
    renderer_config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(renderer_config.to_dict(), checkpoint)
    if (
        float(renderer_config.shift) != DEPLOYMENT_FLOW_SHIFT
        or renderer_config.use_unipc is not True
        or float(deployment_infer.FLOW_SHIFT) != DEPLOYMENT_FLOW_SHIFT
    ):
        raise PairV7PhaseAError("renderer is not pinned deployment UniPC shift 5")
    renderer = BerniniRendererModel(renderer_config).requires_grad_(False).eval()
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV7PhaseAError("Phase-A requires the single Bernini-R 1.3B expert")
    _disable_gradient_checkpointing(renderer, transformer)
    if next(transformer.parameters()).device.type != "cpu":
        raise PairV7PhaseAError("fixed Action-LoRA must be constructed deterministically on CPU")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FIXED_ACTION_LORA_INIT_SEED)
        action_handle = action_adapter.install_pair_v5_action_adapter(transformer)
    gauge = configure_fixed_a_b_only_gauge(action_handle)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    source_spec = pre.action_manifest.events[contract.arm_index]
    source, source_receipt = _encode_source_video(
        source_spec.source_video.path,
        source_spec.source_video.sha256,
        vae=vae,
        device=device,
        parallel=parallel,
    )
    del vae
    torch.cuda.empty_cache()
    renderer.to(device).eval()
    gauge = _validate_fixed_gauge_consensus(gauge, parallel=parallel)
    parameter_state_digest = gauge.initial_full_state_sha256
    checkpoint_receipt_digest = object_sha256(pre.checkpoint_identity)

    (
        native_conditions,
        unconditional,
        rebuilt_t2v_prompts,
        _deployment_v2v_prompts,
        prompt_receipt,
    ) = (
        _encode_native_conditions(
            renderer,
            tokenizer,
            source_spec.raw_caption_by_branch,
            device=device,
            parallel=parallel,
        )
    )
    action_runtime_event = pre.selected_action_events[contract.arm_index]
    if action_runtime_event.spec.prompt_bank_sha256 != object_sha256(rebuilt_t2v_prompts):
        raise PairV7PhaseAError("runtime source/action prompt binding changed")
    action_conditions = _encode_action_prompt_bank(
        renderer=renderer,
        tokenizer=tokenizer,
        prompt_by_branch=action_runtime_event.spec.prompt_by_branch,
        device=device,
        parallel=parallel,
    )
    del tokenizer

    action_gradient, action_vjp_bundle = extract_action_gradient(
        runtime_event=action_runtime_event,
        manifest=pre.action_manifest,
        diffusion=diffusion,
        transformer=transformer,
        action_handle=action_handle,
        conditions=action_conditions,
        gauge=gauge,
        parallel=parallel,
        sp_rank=contract.sp_rank,
        schedule_index=args.schedule_index,
        device=device,
        checkpoint_content_receipt_digest=checkpoint_receipt_digest,
        parameter_state_sha256=parameter_state_digest,
    )
    _clear_gauge_gradients(gauge)
    torch.cuda.empty_cache()

    source_epsilon = _fresh_source_epsilon(
        source.shape,
        seed=args.source_noise_seed,
        arm_index=contract.arm_index,
        device=device,
    )
    _broadcast_sp(source_epsilon, parallel=parallel)
    source_coordinate = build_source_coordinate(
        source,
        source_epsilon,
        schedule_index=args.schedule_index,
        sample_id=source_spec.source_sample_id,
    )
    feature_runtime = NativeFeatureVJPRuntime(
        diffusion=diffusion,
        transformer=transformer,
        action_handle=action_handle,
        correct_source=source,
        coordinate=source_coordinate,
        condition_by_branch=native_conditions,
        unconditional=unconditional,
        sp_rank=contract.sp_rank,
    )
    local_identity: dict[
        tuple[str, int], tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    for family in nullspace.REQUIRED_IDENTITY_FAMILIES:
        for sketch_index in range(IDENTITY_SKETCHES_PER_FAMILY):
            local_identity[(family, sketch_index)] = extract_identity_probe_gradient(
                family=family,
                sketch_index=sketch_index,
                sample_id=source_spec.source_sample_id,
                runtime=feature_runtime,
                gauge=gauge,
                parallel=parallel,
                checkpoint_content_receipt_digest=checkpoint_receipt_digest,
                parameter_state_sha256=parameter_state_digest,
            )
            _clear_gauge_gradients(gauge)
            torch.cuda.empty_cache()

    action_pair = _exchange_named_mapping_dp2(
        action_gradient, parallel=parallel, label="action gradient"
    )
    action_rank_receipt = action_vjp_bundle["rank_receipt"]
    action_sp4_bundle = action_vjp_bundle["sp4_bundle"]
    local_action_meta = {
        "arm_index": contract.arm_index,
        "candidate_id": action_runtime_event.spec.event_id,
        "action_family": action_runtime_event.spec.action_family,
        "event_digest": action_runtime_event.spec.event_digest,
        "gradient_sha256": action_sp4_bundle["averaged_gradient_sha256"],
        "gradient_computation_receipt_digest": action_sp4_bundle[
            "receipt_digest"
        ],
        "gradient_computation_sp4_bundle": action_sp4_bundle,
        "checkpoint_content_receipt_digest": checkpoint_receipt_digest,
        "parameter_state_sha256": parameter_state_digest,
    }
    action_meta_pair = _exchange_metadata_dp2(
        local_action_meta, parallel=parallel, label="action gradient"
    )
    action_by_family = {
        action_meta_pair[arm]["action_family"]: action_pair[arm]
        for arm in range(DP_SIZE)
    }
    action_meta_by_family = {
        action_meta_pair[arm]["action_family"]: action_meta_pair[arm]
        for arm in range(DP_SIZE)
    }

    union_probes: list[nullspace.IdentityGradientProbe] = []
    identity_runtime_receipts: list[Mapping[str, Any]] = []
    for family in nullspace.REQUIRED_IDENTITY_FAMILIES:
        for sketch_index in range(IDENTITY_SKETCHES_PER_FAMILY):
            local_gradient, local_bundle = local_identity[(family, sketch_index)]
            gradient_pair = _exchange_named_mapping_dp2(
                local_gradient,
                parallel=parallel,
                label=f"identity {family} sketch {sketch_index}",
            )
            local_probe_receipt = local_bundle["rank_receipt"]
            local_probe_sp4_bundle = local_bundle["sp4_bundle"]
            local_probe_meta = {
                "arm_index": contract.arm_index,
                "probe_id": (
                    f"dp{contract.arm_index}.{source_spec.source_sample_id}."
                    f"{family}.k{sketch_index}"
                ),
                "family": family,
                "sketch_index": sketch_index,
                "gradient_sha256": local_probe_sp4_bundle[
                    "averaged_gradient_sha256"
                ],
                "feature_sketch_sha256": local_probe_receipt[
                    "feature_sketch_sha256"
                ],
                "source_coordinate_receipt_digest": local_probe_receipt[
                    "source_coordinate_receipt_digest"
                ],
                "identity_deployment_protocol_digest": local_probe_receipt[
                    "identity_deployment_protocol_digest"
                ],
                "deployment_v_pack_receipt_digest": local_probe_receipt[
                    "deployment_v_pack_receipt_digest"
                ],
                "negative_prompt_embedding_sha256": local_probe_receipt[
                    "negative_prompt_embedding_sha256"
                ],
                "positive_prompt_embedding_sha256_by_branch": local_probe_receipt[
                    "positive_prompt_embedding_sha256_by_branch"
                ],
                "identity_vjp_receipt_digest": local_probe_sp4_bundle[
                    "receipt_digest"
                ],
                "identity_vjp_sp4_bundle": local_probe_sp4_bundle,
                "checkpoint_content_receipt_digest": checkpoint_receipt_digest,
                "parameter_state_sha256": parameter_state_digest,
            }
            meta_pair = _exchange_metadata_dp2(
                local_probe_meta,
                parallel=parallel,
                label=f"identity {family} sketch {sketch_index}",
            )
            for arm in range(DP_SIZE):
                meta = meta_pair[arm]
                if _named_gradient_sha256(gradient_pair[arm]) != meta["gradient_sha256"]:
                    raise PairV7PhaseAError("exchanged identity gradient digest differs")
                union_probes.append(
                    nullspace.IdentityGradientProbe(
                        probe_id=meta["probe_id"],
                        family=meta["family"],
                        gradient_by_parameter=gradient_pair[arm],
                        feature_sketch_sha256=meta["feature_sketch_sha256"],
                        source_coordinate_receipt_digest=meta[
                            "source_coordinate_receipt_digest"
                        ],
                        gradient_computation_receipt_digest=meta[
                            "identity_vjp_receipt_digest"
                        ],
                        checkpoint_content_receipt_digest=meta[
                            "checkpoint_content_receipt_digest"
                        ],
                        parameter_state_sha256=meta["parameter_state_sha256"],
                    )
                )
                identity_runtime_receipts.append(meta)

    union = world_rank0_cpu_union_project_and_audit_action_families(
        action_gradient_by_family=action_by_family,
        action_metadata_by_family=action_meta_by_family,
        identity_probes=union_probes,
        checkpoint_content_receipt_digest=checkpoint_receipt_digest,
        parameter_state_sha256=parameter_state_digest,
        fit_only_geometry_authority_digest=(
            pre.fit_geometry_authority.authorization_digest
        ),
        parallel=parallel,
    )
    _clear_gauge_gradients(gauge)
    final_parameter_state = nullspace.named_parameter_state_sha256(
        gauge.full_state_mapping()
    )
    if final_parameter_state != parameter_state_digest:
        raise PairV7PhaseAError("Phase-A gradient audit mutated Action-LoRA parameters")
    final_checkpoint_identity = checkpoint_audit.validate_checkpoint_content(
        Path(args.checkpoint),
        Path(args.checkpoint_content_manifest),
        expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
    )
    if object_sha256(final_checkpoint_identity) != checkpoint_receipt_digest:
        raise PairV7PhaseAError("checkpoint content changed during Phase-A")
    pre.action_manifest.assert_unchanged()
    pre.fit_geometry_authority.assert_unchanged()
    if _file_sha256(pre.runtime_archive_path) != pre.runtime_archive_sha256:
        raise PairV7PhaseAError("runtime source archive changed during Phase-A")
    if (
        _file_sha256(pre.evidence_method_archive_path)
        != pre.evidence_method_archive_sha256
    ):
        raise PairV7PhaseAError("evidence method source archive changed during Phase-A")
    post_bernini_root, post_veomni_root, source_tree_postflight_receipt = (
        _validate_and_bind_source_trees(args)
    )
    if (
        post_bernini_root != bernini_root
        or post_veomni_root != veomni_root
        or source_tree_postflight_receipt != source_tree_preflight_receipt
    ):
        raise PairV7PhaseAError("Bernini/VeOmni source bytes changed during Phase-A")

    local_runtime = {
        "rank": contract.rank,
        "arm_index": contract.arm_index,
        "sp_rank": contract.sp_rank,
        "source_sample_id": source_spec.source_sample_id,
        "source_event_digest": source_spec.event_digest,
        "source_receipt_digest": source_receipt["receipt_digest"],
        "source_receipt": dict(source_receipt),
        "identity_deployment_protocol_digest": feature_runtime.deployment_protocol[
            "receipt_digest"
        ],
        "prompt_receipt_digest": prompt_receipt["receipt_digest"],
        "prompt_receipt": dict(prompt_receipt),
        "action_gradient_rank_receipt_digest": action_rank_receipt[
            "receipt_digest"
        ],
        "action_gradient_sp4_bundle_receipt_digest": action_sp4_bundle[
            "receipt_digest"
        ],
    }
    gathered_runtime: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_runtime, local_runtime, group=parallel.world_group)
    if any(
        not isinstance(row, Mapping)
        or row.get("identity_deployment_protocol_digest")
        != feature_runtime.deployment_protocol["receipt_digest"]
        for row in gathered_runtime
    ):
        raise PairV7PhaseAError("WORLD8 identity deployment protocol consensus differs")
    source_receipts_by_arm = _select_source_receipts_by_arm(
        gathered_runtime, pre.action_manifest
    )
    receipt_unsigned = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method_name": METHOD_NAME,
            "audit_complete": True,
            "geometry_audit_performed": True,
            "geometry_audit_passed": union.geometry_audit_passed,
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "candidate_delta_constructed": False,
            "parameter_add_called": False,
            "parameter_mutation_performed": False,
            "parameter_update_authorized": False,
            "scientific_action_editing_success_claim": False,
            "topology": "WORLD8-DP2xUlysses-SP4",
            "single_exact40_cell": True,
            "schedule_policy": phase_a_schedule_policy(args.schedule_index),
            "frame_count": FRAME_COUNT,
            "source_carrier": source_carrier_extension_contract(
                args.source_carrier_mode
            ),
            "identity_deployment_protocol": dict(
                feature_runtime.deployment_protocol
            ),
            "fixed_lora_gauge": gauge.receipt,
            "checkpoint": {
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_manifest_sha256": (
                    args.expected_checkpoint_content_manifest_sha256
                ),
                "content_receipt_digest": checkpoint_receipt_digest,
                "post_audit_unchanged": True,
            },
            "parameter_state_sha256": parameter_state_digest,
            "post_audit_parameter_state_sha256": final_parameter_state,
            "action_manifest": {
                "file_sha256": pre.action_manifest.raw_sha256,
                "manifest_digest": pre.action_manifest.manifest_digest,
                "candidate_ids": [row["candidate_id"] for row in action_meta_pair],
                "action_families": [row["action_family"] for row in action_meta_pair],
                "fit_only_geometry_authority_digest": (
                    pre.fit_geometry_authority.authorization_digest
                ),
                "fit_only_geometry_authority_validation_receipt": dict(
                    pre.fit_geometry_authority.validation_receipt
                ),
                "external_evidence_files_rehashed_post_audit": True,
                "global_population_authority_consumed": False,
                "cast_v4_receipts_consumed_as_fit_provenance_only": True,
                "population_scorer_optimizer_authority_consumed": False,
            },
            "correct_source_coordinates": {
                "manifest_file_sha256": pre.action_manifest.raw_sha256,
                "manifest_digest": pre.action_manifest.manifest_digest,
                "sample_ids": [
                    event.source_sample_id for event in pre.action_manifest.events
                ],
                "correct_source_media_bound_and_loaded": True,
                "source_frame_count": FRAME_COUNT,
                "source_fps": FPS,
                "deployment_visual_condition": "source_video_only_V",
                "image_reference_count": 0,
                "reference_indices": [],
                "decoded_source_receipts_by_arm": source_receipts_by_arm,
                "sp4_source_receipt_consensus_per_arm": True,
                "wrong_source_fields_present": False,
                "legacy_source_binding_manifest_consumed": False,
            },
            "gradient_information_flow": {
                "pure_t2v_action_arm_only": True,
                "pure_t2v_visual_used_as_rv2v_target_noise_source_or_donor": False,
                "source_native_epsilon_is_fresh_and_separate": True,
                "coordinate_coupling": "shared_fixed_gauge_action_lora_B_parameter_space_only",
                "mask_flow_pose_track_or_trajectory_used": False,
            },
            "dp_action_gradient_metadata": list(action_meta_pair),
            "identity_probe_metadata": identity_runtime_receipts,
            "world_union_solver_authority": union.authority_receipt,
            "union_projection_receipt": union.receipt,
            "nullspace_transport_receipt": union.transport_receipt,
            "rank_runtime_provenance": gathered_runtime,
            "runtime_source": {
                "revision": pre.runtime_source_revision,
                "archive_sha256": pre.runtime_archive_sha256,
                "post_audit_unchanged": True,
            },
            "evidence_method_source": {
                "revision": pre.evidence_method_source_revision,
                "archive_sha256": pre.evidence_method_archive_sha256,
                "post_audit_unchanged": True,
            },
            "model_source_trees": {
                "preflight_receipt": source_tree_preflight_receipt,
                "postflight_receipt": source_tree_postflight_receipt,
                "tracked_file_closure_rehashed_after_audit": True,
                "post_audit_unchanged": True,
            },
        }
    _assert_world_receipt_field_consensus(receipt_unsigned, parallel=parallel)
    receipt = _seal(receipt_unsigned)
    distributed_runtime.digest_consensus(
        receipt["receipt_digest"],
        group=parallel.world_group,
        expected_count=WORLD_SIZE,
        label="Phase-A final receipt",
    )
    if contract.rank == 0:
        _publish_receipt_only(stage, output, receipt)
        print(
            json.dumps(
                {
                    "audit_complete": True,
                    "geometry_audit_passed": union.geometry_audit_passed,
                    "parameter_mutation_performed": False,
                    "receipt": str(output / "receipt.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier(group=parallel.world_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FixedBOnlyGauge",
    "NativeFeatureVJPRuntime",
    "PairV7PhaseAError",
    "PhaseAUnionResult",
    "PhaseAWorldUnionResult",
    "build_mask_free_feature_sketch",
    "build_parser",
    "build_source_coordinate",
    "build_task_prompt_registry",
    "configure_fixed_a_b_only_gauge",
    "main",
    "phase_a_schedule_policy",
    "preflight",
    "source_carrier_extension_contract",
    "union_project_and_audit_action_families",
    "validate_runtime_archive",
    "world_rank0_cpu_union_project_and_audit_action_families",
]
