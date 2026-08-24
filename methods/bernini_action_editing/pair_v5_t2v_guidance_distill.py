#!/usr/bin/env python3
"""PAIR-v5 same-state T2V Counterfactual Action Guidance Distillation.

This is the fail-closed fallback used when native RV2V rollouts contain no
safe action-positive preference pair.  One independently EVENT-QUALIFIED,
pure-T2V clean latent ``y0`` is mixed with *its own stored official sampler
Gaussian* at one released exact40 coordinate.  Frozen Bernini and the
Action-LoRA student are queried under the complete action/hard-negative prompt
bank on that single ``y_sigma`` object.

The pure-T2V video is not an RV2V pseudo target.  No source video, RV2V video,
reference frame, proposal RGB, donor, mask, flow, pose, track, or trajectory is
accepted by this module.  Only an action-relative vector field is learned:

``teacher = frozen(action) - robust_median(frozen(hard negatives))``.

Stable nuisance directions induced by camera-only, appearance-only, and noop
counterfactuals are projected out.  The result is RMS bounded.  Action-LoRA is
trained to add that vector under the action prompt while every hard-negative
prompt is regularized to its byte-frozen base field.  Exact40 indices 38/39
are explicit base-only audit cells: no callback, loss, backward, or optimizer
update is authorized there.

The same Action-LoRA can subsequently be attached to Bernini's native RV2V
path because both paths modify the same ``attn2`` Q/O modules.  This module
does not itself receive or construct an RV2V sample, so cross-video residual
transport is impossible through its public API.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
import struct
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as functional

import mace_candidate_action_energy as mace
import pair_v5_action_adapter as action_adapter
import source_self_native_ref_contrastive_v3 as native_schedule


SCHEMA_VERSION = "bernini-pair-v5-same-state-t2v-action-guidance-v1"
ELIGIBILITY_SCHEMA = "bernini-pair-v5-t2v-guidance-eligibility-v1"
CELL_RECEIPT_SCHEMA = "bernini-pair-v5-t2v-guidance-cell-receipt-v1"
FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
BRANCH_ORDER = mace.BRANCH_ORDER
NEGATIVE_BRANCHES = mace.HARD_NEGATIVE_BRANCHES
NUISANCE_DIRECTION_ORDER = (
    "camera_only_minus_noop",
    "appearance_only_minus_noop",
    "noop_minus_robust_negative_center",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

_FORBIDDEN_PUBLIC_ARGUMENTS = frozenset(
    {
        "source",
        "source_video",
        "source_latent",
        "rv2v",
        "rv2v_video",
        "rv2v_latent",
        "target",
        "target_video",
        "target_latent",
        "proposal_rgb",
        "proposal_video",
        "donor",
        "mask",
        "flow",
        "pose",
        "track",
        "trajectory",
    }
)


class PairV5T2VGuidanceError(RuntimeError):
    """Raised before an ineligible or coordinate-ambiguous update is used."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5T2VGuidanceError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5T2VGuidanceError(f"{label} must be lowercase SHA-256")
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor metadata and exact contiguous storage bytes."""

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise PairV5T2VGuidanceError("tensor hash requires a real torch tensor")
    cpu = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "layout": str(cpu.layout),
    }
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


def action_adapter_schema_payload() -> Mapping[str, Any]:
    """Return the shape-independent Action-LoRA schema transferred to RV2V."""

    return {
        "schema_version": action_adapter.SCHEMA_VERSION,
        "block_indices": list(action_adapter.ACTION_BLOCK_INDICES),
        "projections": ["attn2.to_q", "attn2.to_out.0"],
        "rank": action_adapter.ACTION_LORA_RANK,
        "alpha": action_adapter.ACTION_LORA_ALPHA,
        "dropout": action_adapter.ACTION_LORA_DROPOUT,
        "high_sigma_indices": list(action_adapter.HIGH_SIGMA_INDICES),
        "mid_sigma_indices": list(action_adapter.MID_SIGMA_INDICES),
        "low_base_only_indices": list(action_adapter.LOW_SIGMA_INDICES),
        "target_rows_only": True,
        "t2v_native_branch": "none",
        "rv2v_transfer_branch_registry": list(action_adapter.NATIVE_BRANCHES),
    }


ACTION_ADAPTER_SCHEMA_SHA256 = object_sha256(action_adapter_schema_payload())


def validate_prompt_bank(value: Any) -> dict[str, str]:
    try:
        return mace.validate_prompt_closure(value)
    except mace.MACECandidateActionEnergyError as error:
        raise PairV5T2VGuidanceError(str(error)) from error


def prompt_bank_sha256(value: Any) -> str:
    return object_sha256(validate_prompt_bank(value))


def _exact81(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or int(value.shape[3]) % 2
        or int(value.shape[4]) % 2
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5T2VGuidanceError(
            f"{label} must be detached finite FP32 exact81 [1,16,21,H,W] with even H/W"
        )
    return value


@dataclass(frozen=True)
class GuidanceEligibility:
    sample_id: str
    action_family: str
    analysis_split: str
    latent_shape: tuple[int, ...]
    event_qualified: bool
    calibration_confirmation_passed: bool
    calibration_optimizer_authorized: bool
    clean_t2v_latent_tensor_sha256: str
    official_gaussian_tensor_sha256: str
    official_gaussian_artifact_sha256: str
    checkpoint_tree_sha256: str
    prompt_bank_sha256: str
    action_adapter_schema_sha256: str
    event_qualification_receipt_digest: str
    calibration_receipt_digest: str
    optimizer_authorized: bool
    receipt_digest: str

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": ELIGIBILITY_SCHEMA,
            "sample_id": self.sample_id,
            "action_family": self.action_family,
            "analysis_split": self.analysis_split,
            "frame_count": FRAME_COUNT,
            "latent_shape": list(self.latent_shape),
            "event_qualified": self.event_qualified,
            "calibration_confirmation_passed": self.calibration_confirmation_passed,
            "calibration_optimizer_authorized": self.calibration_optimizer_authorized,
            "clean_t2v_latent_tensor_sha256": self.clean_t2v_latent_tensor_sha256,
            "official_gaussian_tensor_sha256": self.official_gaussian_tensor_sha256,
            "official_gaussian_artifact_sha256": self.official_gaussian_artifact_sha256,
            "checkpoint_tree_sha256": self.checkpoint_tree_sha256,
            "prompt_bank_sha256": self.prompt_bank_sha256,
            "action_adapter_schema_sha256": self.action_adapter_schema_sha256,
            "event_qualification_receipt_digest": self.event_qualification_receipt_digest,
            "calibration_receipt_digest": self.calibration_receipt_digest,
            "pure_t2v_positive_role": "same_coordinate_frozen_field_query_only",
            "rv2v_target_input_noise_donor": False,
            "optimizer_authorized": self.optimizer_authorized,
        }

    def validate(
        self,
        *,
        event_latent: torch.Tensor,
        official_epsilon: torch.Tensor,
        prompt_by_branch: Mapping[str, str],
        checkpoint_tree_sha256: str,
    ) -> None:
        for name in ("sample_id", "action_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
                raise PairV5T2VGuidanceError(
                    f"eligibility {name} is not safe canonical text"
                )
        if self.analysis_split not in {"fit", "confirmation"}:
            raise PairV5T2VGuidanceError("eligibility analysis_split differs")
        for name in (
            "clean_t2v_latent_tensor_sha256",
            "official_gaussian_tensor_sha256",
            "official_gaussian_artifact_sha256",
            "checkpoint_tree_sha256",
            "prompt_bank_sha256",
            "action_adapter_schema_sha256",
            "event_qualification_receipt_digest",
            "calibration_receipt_digest",
            "receipt_digest",
        ):
            _sha256(getattr(self, name), label=name)
        if self.receipt_digest != object_sha256(self.payload()):
            raise PairV5T2VGuidanceError("eligibility receipt digest differs")
        if tuple(event_latent.shape) != self.latent_shape:
            raise PairV5T2VGuidanceError("event latent geometry differs from eligibility")
        if tensor_sha256(event_latent) != self.clean_t2v_latent_tensor_sha256:
            raise PairV5T2VGuidanceError("event latent differs from sealed pure-T2V positive")
        if tensor_sha256(official_epsilon) != self.official_gaussian_tensor_sha256:
            raise PairV5T2VGuidanceError("official Gaussian differs from its stored tensor hash")
        if prompt_bank_sha256(prompt_by_branch) != self.prompt_bank_sha256:
            raise PairV5T2VGuidanceError("hard-negative prompt bank differs from eligibility")
        if _sha256(checkpoint_tree_sha256, label="runtime checkpoint tree SHA-256") != self.checkpoint_tree_sha256:
            raise PairV5T2VGuidanceError("runtime checkpoint differs from eligibility")
        if self.action_adapter_schema_sha256 != ACTION_ADAPTER_SCHEMA_SHA256:
            raise PairV5T2VGuidanceError("Action-LoRA schema differs from eligibility")
        if (
            self.event_qualified is not True
            or self.calibration_confirmation_passed is not True
            or self.calibration_optimizer_authorized is not True
            or self.optimizer_authorized is not True
            or self.analysis_split != "fit"
        ):
            raise PairV5T2VGuidanceError(
                "T2V guidance is ineligible: only fit events with passing event/calibration gates may optimize"
            )


def seal_eligibility(
    *,
    sample_id: str,
    action_family: str,
    analysis_split: str,
    event_latent: torch.Tensor,
    official_epsilon: torch.Tensor,
    official_gaussian_artifact_sha256: str,
    checkpoint_tree_sha256: str,
    prompt_by_branch: Mapping[str, str],
    event_qualified: bool,
    calibration_confirmation_passed: bool,
    calibration_optimizer_authorized: bool,
    event_qualification_receipt_digest: str,
    calibration_receipt_digest: str,
) -> GuidanceEligibility:
    """Create a content-bound eligibility record; false gates remain false."""

    clean = _exact81(event_latent, label="pure-T2V event latent")
    noise = _exact81(official_epsilon, label="stored official Gaussian")
    if clean.shape != noise.shape or clean.device != noise.device:
        raise PairV5T2VGuidanceError("event latent and official Gaussian geometry/device differ")
    for label, value in (("sample_id", sample_id), ("action_family", action_family)):
        if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
            raise PairV5T2VGuidanceError(f"{label} is not safe canonical text")
    if analysis_split not in {"fit", "confirmation"}:
        raise PairV5T2VGuidanceError("analysis_split must be fit or confirmation")
    for label, value in (
        ("official_gaussian_artifact_sha256", official_gaussian_artifact_sha256),
        ("checkpoint_tree_sha256", checkpoint_tree_sha256),
        ("event_qualification_receipt_digest", event_qualification_receipt_digest),
        ("calibration_receipt_digest", calibration_receipt_digest),
    ):
        _sha256(value, label=label)
    gates = (
        type(event_qualified) is bool
        and type(calibration_confirmation_passed) is bool
        and type(calibration_optimizer_authorized) is bool
        and event_qualified
        and calibration_confirmation_passed
        and calibration_optimizer_authorized
        and analysis_split == "fit"
    )
    eligibility = GuidanceEligibility(
        sample_id=sample_id,
        action_family=action_family,
        analysis_split=analysis_split,
        latent_shape=tuple(int(item) for item in clean.shape),
        event_qualified=event_qualified,
        calibration_confirmation_passed=calibration_confirmation_passed,
        calibration_optimizer_authorized=calibration_optimizer_authorized,
        clean_t2v_latent_tensor_sha256=tensor_sha256(clean),
        official_gaussian_tensor_sha256=tensor_sha256(noise),
        official_gaussian_artifact_sha256=official_gaussian_artifact_sha256,
        checkpoint_tree_sha256=checkpoint_tree_sha256,
        prompt_bank_sha256=prompt_bank_sha256(prompt_by_branch),
        action_adapter_schema_sha256=ACTION_ADAPTER_SCHEMA_SHA256,
        event_qualification_receipt_digest=event_qualification_receipt_digest,
        calibration_receipt_digest=calibration_receipt_digest,
        optimizer_authorized=bool(gates),
        receipt_digest="",
    )
    return GuidanceEligibility(
        **{
            **eligibility.__dict__,
            "receipt_digest": object_sha256(eligibility.payload()),
        }
    )


@dataclass(frozen=True)
class SameStateQuery:
    """The only model coordinate used by one complete 20-forward cell."""

    sample_id: str
    x_sigma: torch.Tensor
    sigma: torch.Tensor
    timestep: torch.Tensor
    schedule_index: int
    gate_name: str
    gate_weight: float
    coordinate_digest: str
    x_sigma_object_id: int
    sigma_object_id: int
    timestep_object_id: int
    x_sigma_version: int
    sigma_version: int
    timestep_version: int

    def assert_unchanged(self) -> None:
        if (
            id(self.x_sigma) != self.x_sigma_object_id
            or id(self.sigma) != self.sigma_object_id
            or id(self.timestep) != self.timestep_object_id
            or int(self.x_sigma._version) != self.x_sigma_version
            or int(self.sigma._version) != self.sigma_version
            or int(self.timestep._version) != self.timestep_version
        ):
            raise PairV5T2VGuidanceError("same-state query object was replaced or mutated")


def build_same_state_query(
    event_latent: torch.Tensor,
    official_epsilon: torch.Tensor,
    *,
    schedule_index: int,
    eligibility: GuidanceEligibility,
    prompt_by_branch: Mapping[str, str],
    checkpoint_tree_sha256: str,
) -> SameStateQuery:
    """Construct ``y_sigma=(1-sigma)y0+sigma*eps`` exactly once."""

    clean = _exact81(event_latent, label="pure-T2V event latent")
    noise = _exact81(official_epsilon, label="stored official Gaussian")
    if clean.shape != noise.shape or clean.device != noise.device:
        raise PairV5T2VGuidanceError("event latent and official Gaussian geometry/device differ")
    if not isinstance(eligibility, GuidanceEligibility):
        raise PairV5T2VGuidanceError("eligibility must be loader-validated typed evidence")
    eligibility.validate(
        event_latent=clean,
        official_epsilon=noise,
        prompt_by_branch=prompt_by_branch,
        checkpoint_tree_sha256=checkpoint_tree_sha256,
    )
    try:
        gate_name, gate_weight = action_adapter.sigma_gate(schedule_index)
    except action_adapter.PairV5ActionAdapterError as error:
        raise PairV5T2VGuidanceError(str(error)) from error
    sigma = torch.tensor(
        [native_schedule.NATIVE_UNIPC40_SIGMAS[schedule_index]],
        dtype=torch.float32,
        device=clean.device,
    )
    timestep = torch.tensor(
        [native_schedule.NATIVE_UNIPC40_TIMESTEPS[schedule_index]],
        dtype=torch.float32,
        device=clean.device,
    )
    sigma_view = sigma.reshape(1, 1, 1, 1, 1)
    x_sigma = ((1.0 - sigma_view) * clean + sigma_view * noise).detach().contiguous()
    _exact81(x_sigma, label="same-state y_sigma")
    coordinate_value = {
        "sample_id": eligibility.sample_id,
        "clean_t2v_latent_tensor_sha256": eligibility.clean_t2v_latent_tensor_sha256,
        "official_gaussian_tensor_sha256": eligibility.official_gaussian_tensor_sha256,
        "x_sigma_tensor_sha256": tensor_sha256(x_sigma),
        "schedule_index": schedule_index,
        "sigma_float32_be_hex": struct.pack("!f", float(sigma.item())).hex(),
        "timestep_float32_be_hex": struct.pack("!f", float(timestep.item())).hex(),
        "construction": "(1-sigma)*event_t2v_y0+sigma*its_own_official_epsilon",
    }
    return SameStateQuery(
        sample_id=eligibility.sample_id,
        x_sigma=x_sigma,
        sigma=sigma,
        timestep=timestep,
        schedule_index=schedule_index,
        gate_name=gate_name,
        gate_weight=float(gate_weight),
        coordinate_digest=object_sha256(coordinate_value),
        x_sigma_object_id=id(x_sigma),
        sigma_object_id=id(sigma),
        timestep_object_id=id(timestep),
        x_sigma_version=int(x_sigma._version),
        sigma_version=int(sigma._version),
        timestep_version=int(timestep._version),
    )


@dataclass(frozen=True)
class DenoiseRequest:
    query: SameStateQuery
    branch: str
    prompt: str
    adapter_enabled: bool
    phase: str
    ordinal: int


@dataclass(frozen=True)
class DistillConfig:
    negative_parity_weight: float = 1.0
    trust_penalty_weight: float = 0.1
    teacher_absolute_rms_cap: float = 10.0
    teacher_reference_rms_ratio: float = 2.0
    student_teacher_rms_ratio: float = 1.25
    nuisance_min_rms: float = 1.0e-7
    nuisance_independence_ratio: float = 1.0e-4
    minimum_teacher_rms: float = 1.0e-8
    epsilon: float = 1.0e-8

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PairV5T2VGuidanceError(f"config {name} must be finite")
        for name in (
            "negative_parity_weight",
            "teacher_absolute_rms_cap",
            "teacher_reference_rms_ratio",
            "student_teacher_rms_ratio",
            "nuisance_min_rms",
            "nuisance_independence_ratio",
            "minimum_teacher_rms",
            "epsilon",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise PairV5T2VGuidanceError(f"config {name} must be positive")
        if self.trust_penalty_weight < 0.0:
            raise PairV5T2VGuidanceError("trust_penalty_weight must be nonnegative")
        if self.nuisance_independence_ratio >= 1.0:
            raise PairV5T2VGuidanceError("nuisance_independence_ratio must be below one")


def _prediction(
    value: Any,
    *,
    query: SameStateQuery,
    label: str,
    trainable: bool,
    allow_output_leaf: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != query.x_sigma.shape
        or value.device != query.x_sigma.device
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5T2VGuidanceError(f"{label} must be finite exact81 spatial velocity")
    if trainable and not value.requires_grad:
        raise PairV5T2VGuidanceError(f"{label} must carry an output cotangent")
    if trainable and allow_output_leaf:
        if not value.is_leaf or value.grad_fn is not None:
            raise PairV5T2VGuidanceError(f"{label} must be a measured output leaf")
    elif trainable and value.grad_fn is None:
        raise PairV5T2VGuidanceError(f"{label} must remain connected to Action-LoRA")
    if not trainable and (value.requires_grad or value.grad_fn is not None):
        raise PairV5T2VGuidanceError(f"{label} must be frozen/detached")
    return value


@dataclass(frozen=True)
class PredictionPacket:
    query: SameStateQuery
    base_by_branch: Mapping[str, torch.Tensor]
    student_by_branch: Mapping[str, torch.Tensor]
    prompt_bank_digest: str
    shared_query_object_all_forwards: bool
    call_order: tuple[str, ...]
    leaf_vjp_mode: bool


def collect_same_state_predictions(
    query: SameStateQuery,
    prompt_by_branch: Mapping[str, str],
    denoise_callback: Callable[[DenoiseRequest], torch.Tensor],
    *,
    leaf_vjp_mode: bool = False,
) -> PredictionPacket:
    """Run base/student branches serially on the exact same query object.

    In ``leaf_vjp_mode`` the measured student fields become output leaves;
    :func:`replay_student_vjp` later replays one branch graph at a time.  This
    is the memory-safe path for the real 1.3B model.
    """

    if not isinstance(query, SameStateQuery):
        raise PairV5T2VGuidanceError("query must be a SameStateQuery")
    query.assert_unchanged()
    if query.gate_name == "low_base_only":
        raise PairV5T2VGuidanceError("low-sigma cells forbid all distillation forwards")
    prompts = validate_prompt_bank(prompt_by_branch)
    if not callable(denoise_callback):
        raise PairV5T2VGuidanceError("denoise_callback must be callable")
    base: dict[str, torch.Tensor] = {}
    student: dict[str, torch.Tensor] = {}
    call_order: list[str] = []
    ordinal = 0
    for branch in BRANCH_ORDER:
        request = DenoiseRequest(query, branch, prompts[branch], False, "frozen_base", ordinal)
        with torch.no_grad():
            value = denoise_callback(request)
        query.assert_unchanged()
        base[branch] = _prediction(
            value.detach(), query=query, label=f"frozen base {branch}", trainable=False
        )
        call_order.append(f"base:{branch}")
        ordinal += 1
    for branch in BRANCH_ORDER:
        request = DenoiseRequest(
            query, branch, prompts[branch], True,
            "student_measurement_leaf" if leaf_vjp_mode else "student_graph",
            ordinal,
        )
        if leaf_vjp_mode:
            with torch.no_grad():
                measured = denoise_callback(request)
            value = measured.detach().requires_grad_(True)
        else:
            value = denoise_callback(request)
        query.assert_unchanged()
        student[branch] = _prediction(
            value,
            query=query,
            label=f"student {branch}",
            trainable=True,
            allow_output_leaf=leaf_vjp_mode,
        )
        call_order.append(f"student:{branch}")
        ordinal += 1
    return PredictionPacket(
        query=query,
        base_by_branch=base,
        student_by_branch=student,
        prompt_bank_digest=prompt_bank_sha256(prompts),
        shared_query_object_all_forwards=True,
        call_order=tuple(call_order),
        leaf_vjp_mode=leaf_vjp_mode,
    )


def _rms(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    # Keep the mathematical zero exact. ``epsilon`` is intentionally applied
    # only by callers at division/clamp sites; adding it under this square root
    # would turn an absent teacher into a seemingly non-zero action signal.
    del epsilon
    return value.float().square().mean().sqrt()


@dataclass(frozen=True)
class TeacherVector:
    vector: torch.Tensor
    robust_negative_center: torch.Tensor
    raw_vector: torch.Tensor
    accepted_nuisance_directions: tuple[str, ...]
    skipped_nuisance_directions: tuple[str, ...]
    projection_dot_after: Mapping[str, float]
    raw_rms: float
    projected_rms: float
    bounded_rms: float
    rms_cap: float
    bound_scale: float


def build_bounded_teacher(
    base_by_branch: Mapping[str, torch.Tensor],
    *,
    config: DistillConfig,
) -> TeacherVector:
    """Build a robust, nuisance-orthogonal, detached action vector."""

    config.validate()
    if not isinstance(base_by_branch, Mapping) or set(base_by_branch) != set(BRANCH_ORDER):
        raise PairV5T2VGuidanceError("base field branch closure differs")
    reference = base_by_branch[mace.ACTION_BRANCH]
    if not isinstance(reference, torch.Tensor):
        raise PairV5T2VGuidanceError("base action field is not a tensor")
    normalized: dict[str, torch.Tensor] = {}
    for branch in BRANCH_ORDER:
        value = base_by_branch[branch]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.device != reference.device
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise PairV5T2VGuidanceError(f"frozen base field {branch} differs")
        normalized[branch] = value.detach().float()
    negative_stack = torch.stack(
        [normalized[branch] for branch in NEGATIVE_BRANCHES], dim=0
    )
    robust_center = negative_stack.median(dim=0).values
    raw = normalized[mace.ACTION_BRANCH] - robust_center
    nuisance = {
        "camera_only_minus_noop": normalized["camera_only"] - normalized["noop"],
        "appearance_only_minus_noop": normalized["appearance_only"] - normalized["noop"],
        "noop_minus_robust_negative_center": normalized["noop"] - robust_center,
    }
    flat = raw.reshape(-1)
    basis: list[torch.Tensor] = []
    basis_names: list[str] = []
    skipped: list[str] = []
    for name in NUISANCE_DIRECTION_ORDER:
        direction = nuisance[name].reshape(-1)
        original_norm = torch.linalg.vector_norm(direction)
        if float(original_norm.item()) <= config.nuisance_min_rms * math.sqrt(direction.numel()):
            skipped.append(name)
            continue
        residual = direction
        for unit in basis:
            residual = residual - torch.dot(residual, unit) * unit
        residual_norm = torch.linalg.vector_norm(residual)
        if float((residual_norm / original_norm).item()) < config.nuisance_independence_ratio:
            skipped.append(name)
            continue
        basis.append(residual / residual_norm)
        basis_names.append(name)
    projected = flat
    for unit in basis:
        projected = projected - torch.dot(projected, unit) * unit
    projected_tensor = projected.reshape_as(raw)
    raw_rms_tensor = _rms(raw, config.epsilon)
    projected_rms_tensor = _rms(projected_tensor, config.epsilon)
    reference_rms = torch.maximum(
        _rms(normalized[mace.ACTION_BRANCH], config.epsilon),
        _rms(robust_center, config.epsilon),
    )
    cap_tensor = torch.minimum(
        reference_rms * config.teacher_reference_rms_ratio,
        reference_rms.new_tensor(config.teacher_absolute_rms_cap),
    )
    scale = torch.minimum(
        projected_rms_tensor.new_tensor(1.0),
        cap_tensor / projected_rms_tensor.clamp_min(config.epsilon),
    )
    bounded = (projected_tensor * scale).detach().contiguous()
    bounded_rms = float(_rms(bounded, config.epsilon).item())
    if not math.isfinite(bounded_rms) or bounded_rms <= config.minimum_teacher_rms:
        raise PairV5T2VGuidanceError(
            "event-qualified sample exposes no non-nuisance bounded action vector"
        )
    dots = {
        name: float(torch.dot(bounded.reshape(-1), unit).abs().item())
        for name, unit in zip(basis_names, basis)
    }
    return TeacherVector(
        vector=bounded,
        robust_negative_center=robust_center.detach().contiguous(),
        raw_vector=raw.detach().contiguous(),
        accepted_nuisance_directions=tuple(basis_names),
        skipped_nuisance_directions=tuple(skipped),
        projection_dot_after=dots,
        raw_rms=float(raw_rms_tensor.item()),
        projected_rms=float(projected_rms_tensor.item()),
        bounded_rms=bounded_rms,
        rms_cap=float(cap_tensor.item()),
        bound_scale=float(scale.item()),
    )


@dataclass(frozen=True)
class DistillObjective:
    loss: torch.Tensor
    action_match_loss: torch.Tensor
    negative_parity_loss: torch.Tensor
    trust_penalty: torch.Tensor
    parity_by_branch: Mapping[str, torch.Tensor]
    teacher: TeacherVector
    receipt: Mapping[str, Any]


def build_distill_objective(
    packet: PredictionPacket,
    *,
    config: Optional[DistillConfig] = None,
) -> DistillObjective:
    """Match the action vector and hold every hard negative at frozen base."""

    if not isinstance(packet, PredictionPacket):
        raise PairV5T2VGuidanceError("objective requires a same-state PredictionPacket")
    packet.query.assert_unchanged()
    if packet.query.gate_name == "low_base_only" or packet.query.gate_weight <= 0.0:
        raise PairV5T2VGuidanceError("low-sigma cells cannot construct an objective")
    cfg = DistillConfig() if config is None else config
    cfg.validate()
    teacher = build_bounded_teacher(packet.base_by_branch, config=cfg)
    # The Action-LoRA wrapper already scales its output by the exact40 gate.
    # Therefore the observable target correction must be scaled by the same
    # gate, while the loss itself remains unscaled.  Multiplying the loss by
    # the gate against an unscaled target would make a mid-sigma wrapper learn
    # an approximately 2x ungated residual.
    gated_teacher = teacher.vector * packet.query.gate_weight
    action_correction = (
        packet.student_by_branch[mace.ACTION_BRANCH].float()
        - packet.base_by_branch[mace.ACTION_BRANCH].float()
    )
    action_match = functional.mse_loss(action_correction, gated_teacher)
    parity = {
        branch: functional.mse_loss(
            packet.student_by_branch[branch].float(),
            packet.base_by_branch[branch].float(),
        )
        for branch in NEGATIVE_BRANCHES
    }
    negative_parity = torch.stack(tuple(parity.values())).mean()
    # ``sqrt(mean(x^2))`` has an undefined derivative at an exactly-zero
    # initial LoRA residual (PyTorch exposes it as NaN through the inactive
    # ReLU trust branch).  Keep teacher zero detection exact in ``_rms``, but
    # use a zero-preserving smooth norm for the trainable student path.
    student_rms = (
        action_correction.float().square().mean().add(cfg.epsilon**2).sqrt()
        - cfg.epsilon
    )
    trust_cap = action_correction.new_tensor(
        max(
            teacher.bounded_rms
            * packet.query.gate_weight
            * cfg.student_teacher_rms_ratio,
            cfg.minimum_teacher_rms,
        )
    )
    trust_penalty = torch.relu(student_rms - trust_cap).square()
    loss = (
        action_match
        + cfg.negative_parity_weight * negative_parity
        + cfg.trust_penalty_weight * trust_penalty
    )
    if (
        loss.dtype != torch.float32
        or loss.ndim != 0
        or not loss.requires_grad
        or loss.grad_fn is None
        or not bool(torch.isfinite(loss).item())
    ):
        raise PairV5T2VGuidanceError("distillation loss is not finite graph-connected FP32")
    value = {
        "schema_version": CELL_RECEIPT_SCHEMA,
        "coordinate_digest": packet.query.coordinate_digest,
        "sample_id": packet.query.sample_id,
        "schedule_index": packet.query.schedule_index,
        "sigma_gate": packet.query.gate_name,
        "sigma_gate_weight": packet.query.gate_weight,
        "same_query_object_all_twenty_forwards": packet.shared_query_object_all_forwards,
        "branch_order": list(BRANCH_ORDER),
        "call_order": list(packet.call_order),
        "prompt_bank_sha256": packet.prompt_bank_digest,
        "teacher": {
            "construction": "action_minus_coordinatewise_median_all_nine_hard_negatives",
            "accepted_nuisance_directions": list(teacher.accepted_nuisance_directions),
            "skipped_unstable_nuisance_directions": list(teacher.skipped_nuisance_directions),
            "nuisance_projection_dot_after": dict(teacher.projection_dot_after),
            "raw_rms": teacher.raw_rms,
            "projected_rms": teacher.projected_rms,
            "bounded_rms": teacher.bounded_rms,
            "observable_gated_target_rms": (
                teacher.bounded_rms * packet.query.gate_weight
            ),
            "rms_cap": teacher.rms_cap,
            "bound_scale": teacher.bound_scale,
            "detached": True,
        },
        "negative_base_parity_branches": list(NEGATIVE_BRANCHES),
        "gate_semantics": "output_amplitude_gate_target_scaled_loss_not_scaled",
        "leaf_vjp_mode": packet.leaf_vjp_mode,
        "pure_t2v_role": "same_coordinate_frozen_field_query_only",
        "rv2v_target_input_noise_donor": False,
        "cross_video_vector_transport": False,
        "optimizer_update_authorized": True,
        "semantic_action_editing_success_claimed": False,
    }
    receipt = {**value, "receipt_digest": object_sha256(value)}
    return DistillObjective(
        loss=loss,
        action_match_loss=action_match,
        negative_parity_loss=negative_parity,
        trust_penalty=trust_penalty,
        parity_by_branch=parity,
        teacher=teacher,
        receipt=receipt,
    )


def replay_student_vjp(
    packet: PredictionPacket,
    prompt_by_branch: Mapping[str, str],
    denoise_callback: Callable[[DenoiseRequest], torch.Tensor],
    *,
    rtol: float = 2.0e-5,
    atol: float = 2.0e-5,
) -> Mapping[str, float]:
    """Replay student branches one at a time after leaf-loss backward."""

    if not packet.leaf_vjp_mode:
        raise PairV5T2VGuidanceError("serial VJP replay requires leaf_vjp_mode")
    prompts = validate_prompt_bank(prompt_by_branch)
    if prompt_bank_sha256(prompts) != packet.prompt_bank_digest:
        raise PairV5T2VGuidanceError("VJP replay prompt bank differs")
    maxima: dict[str, float] = {}
    for ordinal, branch in enumerate(BRANCH_ORDER):
        leaf = packet.student_by_branch[branch]
        if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all().item()):
            raise PairV5T2VGuidanceError(f"student output cotangent absent for {branch}")
        request = DenoiseRequest(
            packet.query,
            branch,
            prompts[branch],
            True,
            "student_vjp_replay",
            2 * len(BRANCH_ORDER) + ordinal,
        )
        replay = _prediction(
            denoise_callback(request),
            query=packet.query,
            label=f"student replay {branch}",
            trainable=True,
        )
        packet.query.assert_unchanged()
        difference = float(
            (replay.detach().float() - leaf.detach().float()).abs().max().item()
        )
        if not torch.allclose(
            replay.detach().float(), leaf.detach().float(), rtol=rtol, atol=atol
        ):
            raise PairV5T2VGuidanceError(f"student VJP replay changed branch {branch}")
        replay.backward(leaf.grad.detach())
        maxima[branch] = difference
    return maxima


@dataclass(frozen=True)
class DistillCell:
    query: SameStateQuery
    packet: Optional[PredictionPacket]
    objective: Optional[DistillObjective]
    optimizer_authorized: bool
    zero_update: bool
    receipt: Mapping[str, Any]


def run_same_state_cell(
    event_latent: torch.Tensor,
    official_epsilon: torch.Tensor,
    *,
    schedule_index: int,
    eligibility: GuidanceEligibility,
    prompt_by_branch: Mapping[str, str],
    checkpoint_tree_sha256: str,
    denoise_callback: Callable[[DenoiseRequest], torch.Tensor],
    config: Optional[DistillConfig] = None,
    leaf_vjp_mode: bool = False,
) -> DistillCell:
    """Validate evidence, create one coordinate, and build one update/null cell."""

    query = build_same_state_query(
        event_latent,
        official_epsilon,
        schedule_index=schedule_index,
        eligibility=eligibility,
        prompt_by_branch=prompt_by_branch,
        checkpoint_tree_sha256=checkpoint_tree_sha256,
    )
    if query.gate_name == "low_base_only":
        value = {
            "schema_version": CELL_RECEIPT_SCHEMA,
            "sample_id": query.sample_id,
            "coordinate_digest": query.coordinate_digest,
            "eligibility_receipt_digest": eligibility.receipt_digest,
            "schedule_index": query.schedule_index,
            "sigma_gate": query.gate_name,
            "sigma_gate_weight": 0.0,
            "update_kind": "frozen_base_anchor_zero_update",
            "model_callback_called": False,
            "loss_constructed": False,
            "backward_called": False,
            "optimizer_step_authorized": False,
            "rv2v_target_input_noise_donor": False,
        }
        return DistillCell(
            query=query,
            packet=None,
            objective=None,
            optimizer_authorized=False,
            zero_update=True,
            receipt={**value, "receipt_digest": object_sha256(value)},
        )
    packet = collect_same_state_predictions(
        query,
        prompt_by_branch,
        denoise_callback,
        leaf_vjp_mode=leaf_vjp_mode,
    )
    objective = build_distill_objective(packet, config=config)
    value = {
        **dict(objective.receipt),
        "eligibility_receipt_digest": eligibility.receipt_digest,
        "official_gaussian_tensor_sha256": eligibility.official_gaussian_tensor_sha256,
        "checkpoint_tree_sha256": eligibility.checkpoint_tree_sha256,
        "action_adapter_schema_sha256": eligibility.action_adapter_schema_sha256,
        "event_qualified": eligibility.event_qualified,
        "action_family": eligibility.action_family,
        "analysis_split": eligibility.analysis_split,
        "calibration_confirmation_passed": eligibility.calibration_confirmation_passed,
    }
    value.pop("receipt_digest", None)
    return DistillCell(
        query=query,
        packet=packet,
        objective=objective,
        optimizer_authorized=True,
        zero_update=False,
        receipt={**value, "receipt_digest": object_sha256(value)},
    )


def contract_receipt() -> Mapping[str, Any]:
    public = (
        seal_eligibility,
        build_same_state_query,
        collect_same_state_predictions,
        build_bounded_teacher,
        build_distill_objective,
        replay_student_vjp,
        run_same_state_cell,
    )
    offending = {
        function.__name__: sorted(
            set(inspect.signature(function).parameters) & _FORBIDDEN_PUBLIC_ARGUMENTS
        )
        for function in public
        if set(inspect.signature(function).parameters) & _FORBIDDEN_PUBLIC_ARGUMENTS
    }
    if offending:
        raise PairV5T2VGuidanceError(f"public API exposes cross-video inputs: {offending}")
    value = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_channels": LATENT_CHANNELS,
        "latent_phases": LATENT_PHASES,
        "branch_order": list(BRANCH_ORDER),
        "nuisance_direction_order": list(NUISANCE_DIRECTION_ORDER),
        "same_coordinate": "one y_sigma object for frozen/student action+all negatives",
        "teacher": "bounded nuisance-projected action-minus-robust-negative field",
        "sigma_gate_semantics": "output_amplitude_and_teacher_target_scaled_loss_unscaled",
        "student_trust_norm": "zero_preserving_smooth_rms_finite_at_zero_lora",
        "negative_regularizer": "student hard-negative fields equal frozen base fields",
        "optimizer_analysis_split": "fit_only",
        "confirmation_split_role": "held_out_calibrator_and_transfer_gate_only",
        "action_adapter_schema_sha256": ACTION_ADAPTER_SCHEMA_SHA256,
        "exact40_schedule_sha256": action_adapter.sigma_strata.SCHEDULE_SHA256,
        "trainable_indices": list(
            action_adapter.HIGH_SIGMA_INDICES + action_adapter.MID_SIGMA_INDICES
        ),
        "low_base_only_indices": list(action_adapter.LOW_SIGMA_INDICES),
        "pure_t2v_video_is_rv2v_target_input_noise_or_donor": False,
        "cross_video_vector_transport": False,
        "public_api_forbidden_inputs_absent": True,
        "semantic_action_editing_success_claimed": False,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "ACTION_ADAPTER_SCHEMA_SHA256",
    "BRANCH_ORDER",
    "CELL_RECEIPT_SCHEMA",
    "DenoiseRequest",
    "DistillCell",
    "DistillConfig",
    "DistillObjective",
    "ELIGIBILITY_SCHEMA",
    "FRAME_COUNT",
    "GuidanceEligibility",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "NEGATIVE_BRANCHES",
    "NUISANCE_DIRECTION_ORDER",
    "PairV5T2VGuidanceError",
    "PredictionPacket",
    "SCHEMA_VERSION",
    "SameStateQuery",
    "TeacherVector",
    "action_adapter_schema_payload",
    "build_bounded_teacher",
    "build_distill_objective",
    "build_same_state_query",
    "collect_same_state_predictions",
    "contract_receipt",
    "object_sha256",
    "prompt_bank_sha256",
    "replay_student_vjp",
    "run_same_state_cell",
    "seal_eligibility",
    "tensor_sha256",
    "validate_prompt_bank",
]
