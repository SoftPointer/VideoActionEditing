#!/usr/bin/env python3
"""PAIR-v6 SCAID: source-coordinate action/identity decoupling.

SCAID estimates a frozen T2V action-relative velocity field on the *source
video's own* rectified-flow state

``x_sigma = (1 - sigma) * z_source + sigma * epsilon``.

It then removes prompt nuisance, temporal-DC, and frozen native
correct-source-vs-wrong-source binding directions before adding the bounded
residual to the detached native RV2V-4 base field.  Action-LoRA is optimized
directly in the native deployment task.  There is no T2V-to-RV2V parameter or
latent transfer.

Pure T2V generated videos are intentionally absent from every tensor API.
They may only authorize an action family through external, digest-bound fit
and confirmation calibration receipts.  Inference needs only the source
video/four native references, target prompt, frozen Bernini base, and LoRA.

This is an executable research primitive, not evidence of successful action
editing.  It accepts no target video, proposal, donor, mask, flow, pose,
track, or trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import struct
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as functional

import mace_candidate_action_energy as mace
import infer_native_identity_generation_canary as native_infer
import pair_v5_t2v_calibration_bank_spec as bank_spec
import pair_v5_action_adapter as action_adapter
import source_self_native_ref_contrastive_v3 as native_schedule
import train_pair_v5_t2v_guidance_distill as cagd_trainer
import validate_pair_v5_cagd_evidence_v3 as evidence_validator


SCHEMA_VERSION = "bernini-pair-v6-scaid-source-coordinate-v1"
AUTHORIZATION_SCHEMA = "bernini-pair-v6-scaid-authoritative-v3-gate-v1"
CELL_RECEIPT_SCHEMA = "bernini-pair-v6-scaid-cell-v1"
SURVIVAL_RECEIPT_SCHEMA = "bernini-pair-v6-scaid-residual-survival-v1"
FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
BRANCH_ORDER = mace.BRANCH_ORDER
NEGATIVE_BRANCHES = mace.HARD_NEGATIVE_BRANCHES
IDENTITY_BINDING_BRANCHES = (mace.ACTION_BRANCH, "noop")
IDENTITY_CONTROL_DIRECTION_ORDER = ("native_reference_dI",)
NUISANCE_DIRECTION_ORDER = (
    "camera_only_minus_noop",
    "appearance_only_minus_noop",
    "noop_minus_robust_negative_center",
)
NATIVE_GUIDANCE_COMPONENTS = (
    ("none_uncond", -0.25),
    ("V_uncond", -3.25),
    ("VI_uncond", 0.5),
    ("VI_cond", 4.0),
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_FORBIDDEN_PUBLIC_ARGUMENTS = frozenset(
    {
        "t2v_video",
        "t2v_latent",
        "generated_video",
        "generated_latent",
        "target",
        "target_video",
        "target_latent",
        "proposal",
        "proposal_video",
        "proposal_latent",
        "donor",
        "mask",
        "flow",
        "pose",
        "track",
        "trajectory",
    }
)


class PairV6SCAIDError(RuntimeError):
    """Raised before an ambiguous or unauthorised SCAID update."""


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
        raise PairV6SCAIDError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV6SCAIDError(f"{label} must be lowercase SHA-256")
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise PairV6SCAIDError("tensor hash requires a real torch tensor")
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
        raise PairV6SCAIDError(
            f"{label} must be detached finite FP32 exact81 [1,16,21,H,W] with even H/W"
        )
    return value


def _prompts(value: Any) -> dict[str, str]:
    try:
        return mace.validate_prompt_closure(value)
    except mace.MACECandidateActionEnergyError as error:
        raise PairV6SCAIDError(str(error)) from error


def build_task_prompt_banks(
    raw_caption_by_branch: Mapping[str, str],
    *,
    prompt_cleaner: Callable[[str], str],
) -> tuple[Mapping[str, str], Mapping[str, str], Mapping[str, Any]]:
    """Build the authoritative T2V/RV2V tasks once from one raw caption bank."""

    raw = _prompts(raw_caption_by_branch)
    prefixes = tuple(native_infer.TASK_SYSTEM_PROMPTS.values())
    if any(
        caption.startswith(prefix)
        for caption in raw.values()
        for prefix in prefixes
    ):
        raise PairV6SCAIDError(
            "raw caption is already task-prefixed; double-wrap forbidden"
        )
    t2v = {
        branch: native_infer.build_task_prompt(
            "t2v", raw[branch], prompt_cleaner=prompt_cleaner
        )
        for branch in BRANCH_ORDER
    }
    rv2v = {
        branch: native_infer.build_task_prompt(
            "rv2v", raw[branch], prompt_cleaner=prompt_cleaner
        )
        for branch in BRANCH_ORDER
    }
    value = {
        "raw_caption_bank_sha256": object_sha256(raw),
        "t2v_task_prompt_bank_sha256": object_sha256(t2v),
        "rv2v_task_prompt_bank_sha256": object_sha256(rv2v),
        "same_sealed_raw_caption_bank": True,
        "task_prefix_applied_exactly_once": True,
    }
    return t2v, rv2v, {**value, "digest": object_sha256(value)}


def _official_task_prompt_banks(
    raw_caption_by_branch: Mapping[str, str],
) -> tuple[Mapping[str, str], Mapping[str, str], Mapping[str, Any]]:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    return build_task_prompt_banks(
        raw_caption_by_branch, prompt_cleaner=prompt_clean
    )


def _registered_coordinate(
    source_clean_latent: torch.Tensor,
    official_epsilon: torch.Tensor,
    schedule_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, float]:
    source = _exact81(source_clean_latent, label="source clean latent")
    epsilon = _exact81(official_epsilon, label="official Gaussian")
    if epsilon.shape != source.shape or epsilon.device != source.device:
        raise PairV6SCAIDError("source clean latent and official Gaussian differ")
    try:
        gate_name, gate_weight = action_adapter.sigma_gate(schedule_index)
    except action_adapter.PairV5ActionAdapterError as error:
        raise PairV6SCAIDError(str(error)) from error
    sigma = torch.tensor(
        [native_schedule.NATIVE_UNIPC40_SIGMAS[schedule_index]],
        dtype=torch.float32,
        device=source.device,
    )
    timestep = torch.tensor(
        [native_schedule.NATIVE_UNIPC40_TIMESTEPS[schedule_index]],
        dtype=torch.float32,
        device=source.device,
    )
    shaped_sigma = sigma.reshape(1, 1, 1, 1, 1)
    x_sigma = ((1.0 - shaped_sigma) * source + shaped_sigma * epsilon).detach().contiguous()
    _exact81(x_sigma, label="source-coordinate x_sigma")
    return x_sigma, sigma, timestep, gate_name, float(gate_weight)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SCAIDAuthorization:
    """In-memory capability emitted only after authoritative v3 recomputation."""

    fit_candidate_id: str
    action_family: str
    prompt_bank_sha256: str
    checkpoint_tree_sha256: str
    evidence_path: Path
    evidence_file_sha256: str
    evidence_digest: str
    authorization_digest: str
    calibration_receipt_digest: str
    guidance_manifest_file_sha256: str
    source_bank_spec_file_sha256: str
    geometry_source_video_path: Path
    geometry_source_video_sha256: str
    def validate(
        self,
        *,
        prompt_by_branch: Mapping[str, str],
        checkpoint_tree_sha256: str,
    ) -> None:
        if (
            not self.evidence_path.is_absolute()
            or not self.evidence_path.is_file()
            or self.evidence_path.is_symlink()
            or _file_sha256(self.evidence_path) != self.evidence_file_sha256
        ):
            raise PairV6SCAIDError("authoritative evidence file changed")
        if self.checkpoint_tree_sha256 != _sha256(
            checkpoint_tree_sha256, label="checkpoint tree"
        ):
            raise PairV6SCAIDError("authoritative checkpoint tree differs")
        if self.prompt_bank_sha256 != object_sha256(_prompts(prompt_by_branch)):
            raise PairV6SCAIDError("prompt bank differs from authoritative fit event")
        try:
            refreshed = evidence_validator.validate_evidence(
                self.evidence_path,
                expected_evidence_sha256=self.evidence_file_sha256,
                checkpoint_tree_sha256=self.checkpoint_tree_sha256,
            )
        except Exception as error:
            raise PairV6SCAIDError(
                f"authoritative evidence revalidation failed: {error}"
            ) from error
        if (
            refreshed.get("authorization_digest") != self.authorization_digest
            or refreshed.get("optimizer_authorized") is not True
            or refreshed.get("recomputed_calibration_receipt_digest")
            != self.calibration_receipt_digest
        ):
            raise PairV6SCAIDError("authoritative authorization changed or became NO-GO")
        try:
            evidence = json.loads(self.evidence_path.read_text(encoding="ascii"))
            binding = evidence["guidance_manifest"]
            manifest = cagd_trainer.load_manifest(
                binding["path"], binding["file_sha256"]
            )
            spec_binding = evidence["source_bank_spec"]
            spec, observed_spec_sha = bank_spec.load_sealed_spec(
                spec_binding["path"], spec_binding["file_sha256"]
            )
        except Exception as error:
            raise PairV6SCAIDError(
                f"authoritative fit manifest revalidation failed: {error}"
            ) from error
        if (
            binding.get("file_sha256") != self.guidance_manifest_file_sha256
            or binding.get("file_sha256")
            != refreshed.get("guidance_manifest_file_sha256")
            or manifest.raw_sha256 != self.guidance_manifest_file_sha256
            or observed_spec_sha != self.source_bank_spec_file_sha256
            or spec_binding.get("file_sha256")
            != self.source_bank_spec_file_sha256
            or refreshed.get("source_bank_spec_sha256")
            != self.source_bank_spec_file_sha256
        ):
            raise PairV6SCAIDError("authoritative fit manifest binding changed")
        matches = [
            event
            for event in manifest.events
            if event.event_id == self.fit_candidate_id
        ]
        if len(matches) != 1 or matches[0].analysis_split != "fit":
            raise PairV6SCAIDError(
                "authorization does not name one authoritative fit event"
            )
        event = matches[0]
        source_anchor = _fit_source_anchor(
            spec,
            fit_candidate_id=self.fit_candidate_id,
            action_family=self.action_family,
        )
        prompt_digest = object_sha256(_prompts(prompt_by_branch))
        if (
            event.action_family != self.action_family
            or event.prompt_bank_sha256 != self.prompt_bank_sha256
            or event.prompt_bank_sha256 != prompt_digest
        ):
            raise PairV6SCAIDError(
                "authorization action family or prompt bank differs from fit event"
            )
        if (
            Path(source_anchor["geometry_source_video"]).resolve(strict=True)
            != self.geometry_source_video_path
            or source_anchor["geometry_source_video_sha256"]
            != self.geometry_source_video_sha256
            or _file_sha256(self.geometry_source_video_path)
            != self.geometry_source_video_sha256
        ):
            raise PairV6SCAIDError(
                "authorization geometry source anchor changed"
            )


def _fit_source_anchor(
    spec: Mapping[str, Any],
    *,
    fit_candidate_id: str,
    action_family: str,
) -> Mapping[str, Any]:
    candidates = [
        candidate
        for group in spec.get("groups", ())
        for candidate in group.get("candidates", ())
    ]
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_id") == fit_candidate_id
    ]
    if (
        len(matches) != 1
        or matches[0].get("analysis_split") != "fit"
        or matches[0].get("semantic_branch") != mace.ACTION_BRANCH
        or matches[0].get("action_family_id") != action_family
    ):
        raise PairV6SCAIDError(
            "fit candidate has no unique evidence-bound geometry source anchor"
        )
    anchor = matches[0]
    path = Path(anchor.get("geometry_source_video", ""))
    sha = anchor.get("geometry_source_video_sha256")
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or _sha256(sha, label="geometry source video SHA-256")
        != _file_sha256(path)
    ):
        raise PairV6SCAIDError("evidence-bound geometry source file differs")
    return anchor


def load_authoritative_v3_authorization(
    evidence_path: str | Path,
    *,
    expected_evidence_sha256: str,
    checkpoint_tree_sha256: str,
    fit_candidate_id: str,
) -> SCAIDAuthorization:
    """Recompute the complete v3 evidence graph and mint a process-local gate."""

    path = Path(evidence_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV6SCAIDError("CAGD v3 evidence must be an absolute plain file")
    expected_sha = _sha256(expected_evidence_sha256, label="evidence file SHA-256")
    if _file_sha256(path) != expected_sha:
        raise PairV6SCAIDError("CAGD v3 evidence file SHA-256 differs")
    try:
        authorization = evidence_validator.validate_evidence(
            path,
            expected_evidence_sha256=expected_sha,
            checkpoint_tree_sha256=_sha256(
                checkpoint_tree_sha256, label="checkpoint tree SHA-256"
            ),
        )
    except Exception as error:
        raise PairV6SCAIDError(
            f"authoritative v3 evidence recomputation failed: {error}"
        ) from error
    if (
        authorization.get("schema_version") != evidence_validator.AUTHORIZATION_SCHEMA
        or authorization.get("optimizer_authorized") is not True
        or authorization.get("legacy_eligibility_self_declaration_trusted") is not False
        or authorization.get("all_source_files_and_receipts_revalidated") is not True
        or authorization.get("calibration_recomputed_from_raw_global_scores") is not True
        or authorization.get("confirmation_event_count_for_optimizer") != 0
    ):
        raise PairV6SCAIDError("authoritative v3 output does not authorize an optimizer")
    try:
        evidence = json.loads(path.read_text(encoding="ascii"))
        binding = evidence["guidance_manifest"]
        manifest = cagd_trainer.load_manifest(binding["path"], binding["file_sha256"])
        spec_binding = evidence["source_bank_spec"]
        spec, observed_spec_sha = bank_spec.load_sealed_spec(
            spec_binding["path"], spec_binding["file_sha256"]
        )
    except Exception as error:
        raise PairV6SCAIDError(f"authoritative fit manifest reload failed: {error}") from error
    if (
        binding["file_sha256"] != authorization.get("guidance_manifest_file_sha256")
        or manifest.raw_sha256 != binding["file_sha256"]
        or len(manifest.events) != authorization.get("fit_event_count")
    ):
        raise PairV6SCAIDError("authoritative fit manifest binding differs")
    matches = [event for event in manifest.events if event.event_id == fit_candidate_id]
    if len(matches) != 1 or matches[0].analysis_split != "fit":
        raise PairV6SCAIDError("requested candidate is not one authoritative fit event")
    event = matches[0]
    source_anchor = _fit_source_anchor(
        spec,
        fit_candidate_id=event.event_id,
        action_family=event.action_family,
    )
    if (
        observed_spec_sha != spec_binding["file_sha256"]
        or authorization.get("source_bank_spec_sha256") != observed_spec_sha
    ):
        raise PairV6SCAIDError("authoritative source-bank spec binding differs")
    gate = SCAIDAuthorization(
        fit_candidate_id=event.event_id,
        action_family=event.action_family,
        prompt_bank_sha256=event.prompt_bank_sha256,
        checkpoint_tree_sha256=_sha256(checkpoint_tree_sha256, label="checkpoint tree"),
        evidence_path=path.resolve(strict=True),
        evidence_file_sha256=expected_sha,
        evidence_digest=_sha256(authorization["evidence_digest"], label="evidence digest"),
        authorization_digest=_sha256(
            authorization["authorization_digest"], label="authorization digest"
        ),
        calibration_receipt_digest=_sha256(
            authorization["recomputed_calibration_receipt_digest"],
            label="calibration receipt digest",
        ),
        guidance_manifest_file_sha256=manifest.raw_sha256,
        source_bank_spec_file_sha256=_sha256(
            observed_spec_sha, label="source-bank spec SHA-256"
        ),
        geometry_source_video_path=Path(
            source_anchor["geometry_source_video"]
        ).resolve(strict=True),
        geometry_source_video_sha256=_sha256(
            source_anchor["geometry_source_video_sha256"],
            label="geometry source video SHA-256",
        ),
    )
    return gate


@dataclass(frozen=True)
class SourceCoordinate:
    x_sigma: torch.Tensor
    sigma: torch.Tensor
    timestep: torch.Tensor
    schedule_index: int
    gate_name: str
    gate_weight: float
    coordinate_digest: str
    x_sigma_object_id: int
    x_sigma_version: int

    def assert_unchanged(self) -> None:
        if id(self.x_sigma) != self.x_sigma_object_id or int(self.x_sigma._version) != self.x_sigma_version:
            raise PairV6SCAIDError("source-coordinate state was replaced or modified in-place")


@dataclass(frozen=True)
class T2VFieldRequest:
    coordinate: SourceCoordinate
    branch: str
    prompt: str
    ordinal: int


@dataclass(frozen=True)
class NativeFieldRequest:
    coordinate: SourceCoordinate
    branch: str
    prompt: str
    source_role: str
    adapter_enabled: bool
    phase: str
    ordinal: int


@dataclass(frozen=True)
class SCAIDConfig:
    action_residual_weight: float = 0.35
    negative_parity_weight: float = 1.0
    action_base_trust_weight: float = 0.05
    absolute_residual_rms_cap: float = 4.0
    native_base_rms_ratio: float = 0.35
    direction_min_rms: float = 1.0e-7
    direction_independence_ratio: float = 1.0e-4
    minimum_residual_rms: float = 1.0e-8
    minimum_projected_survival_ratio: float = 0.01
    maximum_projection_cosine: float = 1.0e-4
    maximum_temporal_dc_relative_rms: float = 1.0e-5
    epsilon: float = 1.0e-8

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PairV6SCAIDError(f"config {name} must be finite")
        for name in (
            "action_residual_weight",
            "negative_parity_weight",
            "absolute_residual_rms_cap",
            "native_base_rms_ratio",
            "direction_min_rms",
            "direction_independence_ratio",
            "minimum_residual_rms",
            "minimum_projected_survival_ratio",
            "maximum_projection_cosine",
            "maximum_temporal_dc_relative_rms",
            "epsilon",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise PairV6SCAIDError(f"config {name} must be positive")
        if self.action_base_trust_weight < 0.0:
            raise PairV6SCAIDError("action_base_trust_weight must be nonnegative")
        if self.direction_independence_ratio >= 1.0:
            raise PairV6SCAIDError("direction_independence_ratio must be below one")
        if self.minimum_projected_survival_ratio > 1.0:
            raise PairV6SCAIDError("minimum_projected_survival_ratio must not exceed one")
        if self.maximum_projection_cosine >= 1.0:
            raise PairV6SCAIDError("maximum_projection_cosine must be below one")


def _field(
    value: Any,
    *,
    coordinate: SourceCoordinate,
    label: str,
    trainable: bool,
    allow_output_leaf: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != coordinate.x_sigma.shape
        or value.device != coordinate.x_sigma.device
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV6SCAIDError(f"{label} must be a finite exact81 velocity")
    if trainable and not value.requires_grad:
        raise PairV6SCAIDError(f"{label} must carry an output cotangent")
    if trainable and allow_output_leaf:
        if not value.is_leaf or value.grad_fn is not None:
            raise PairV6SCAIDError(f"{label} must be a measured output leaf")
    elif trainable and value.grad_fn is None:
        raise PairV6SCAIDError(f"{label} must remain connected to Action-LoRA")
    if not trainable and (value.requires_grad or value.grad_fn is not None):
        raise PairV6SCAIDError(f"{label} must be frozen/detached")
    return value


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _smooth_zero_rms(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Zero-preserving RMS with a finite derivative at an exact-zero tensor."""

    squared = value.float().square().mean()
    eps = squared.new_tensor(float(epsilon))
    return torch.sqrt(squared + eps.square()) - eps


def aggregate_native_guidance_components(
    components: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Aggregate native CFG once, in authoritative order and FP32 arithmetic."""

    expected_names = tuple(name for name, _ in NATIVE_GUIDANCE_COMPONENTS)
    if tuple(components) != expected_names:
        raise PairV6SCAIDError(
            "native CFG component mapping must preserve authoritative order"
        )
    reference = components[expected_names[0]]
    if (
        not isinstance(reference, torch.Tensor)
        or not reference.is_floating_point()
        or reference.device.type == "meta"
        or not bool(torch.isfinite(reference).all().item())
    ):
        raise PairV6SCAIDError("native CFG reference component differs")
    result = torch.zeros_like(reference, dtype=torch.float32)
    for name, coefficient in NATIVE_GUIDANCE_COMPONENTS:
        component = components[name]
        if (
            not isinstance(component, torch.Tensor)
            or component.shape != reference.shape
            or component.device != reference.device
            or not component.is_floating_point()
            or component.device.type == "meta"
            or not bool(torch.isfinite(component).all().item())
        ):
            raise PairV6SCAIDError(f"native CFG component {name} differs")
        # Do not rewrite this as staged CFG differences in BF16.  Measurement
        # and VJP replay must perform this exact ordered FP32 sum.
        result = result + component.float() * float(coefficient)
    if not bool(torch.isfinite(result).all().item()):
        raise PairV6SCAIDError("native CFG aggregate is non-finite")
    return result


def _temporal_center(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=2, keepdim=True)


@dataclass(frozen=True)
class SafeActionResidual:
    vector: torch.Tensor
    robust_negative_center: torch.Tensor
    raw_vector: torch.Tensor
    accepted_directions: tuple[str, ...]
    skipped_directions: tuple[str, ...]
    projection_dot_after: Mapping[str, float]
    projection_cosine_after: Mapping[str, float]
    temporal_dc_rms_after: float
    raw_rms: float
    projected_rms: float
    bounded_rms: float
    rms_cap: float
    bound_scale: float


def build_safe_action_residual(
    t2v_base_by_branch: Mapping[str, torch.Tensor],
    native_correct_by_branch: Mapping[str, torch.Tensor],
    native_wrong_source_by_branch: Mapping[str, torch.Tensor],
    native_identity_directions: Mapping[str, torch.Tensor],
    *,
    config: Optional[SCAIDConfig] = None,
) -> SafeActionResidual:
    """Create a detached action residual orthogonal to known identity/camera spans."""

    cfg = SCAIDConfig() if config is None else config
    cfg.validate()
    if set(t2v_base_by_branch) != set(BRANCH_ORDER):
        raise PairV6SCAIDError("frozen T2V branch closure differs")
    if set(native_correct_by_branch) != set(BRANCH_ORDER):
        raise PairV6SCAIDError("frozen native correct-source branch closure differs")
    if set(native_wrong_source_by_branch) != set(IDENTITY_BINDING_BRANCHES):
        raise PairV6SCAIDError("wrong-source identity span requires action and noop")
    if set(native_identity_directions) != set(IDENTITY_CONTROL_DIRECTION_ORDER):
        raise PairV6SCAIDError("native identity control requires the four-reference d_I direction")
    reference = t2v_base_by_branch[mace.ACTION_BRANCH]
    if (
        not isinstance(reference, torch.Tensor)
        or reference.ndim != 5
        or tuple(int(item) for item in reference.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(reference.shape[3]) <= 0
        or int(reference.shape[4]) <= 0
        or int(reference.shape[3]) % 2
        or int(reference.shape[4]) % 2
    ):
        raise PairV6SCAIDError("action fields must use exact81 [1,16,21,H,W]")
    values: dict[str, torch.Tensor] = {}
    for branch in BRANCH_ORDER:
        item = t2v_base_by_branch[branch]
        if (
            not isinstance(item, torch.Tensor)
            or item.shape != reference.shape
            or item.device != reference.device
            or not item.is_floating_point()
            or item.requires_grad
            or item.grad_fn is not None
            or not bool(torch.isfinite(item).all().item())
        ):
            raise PairV6SCAIDError(f"frozen T2V field {branch} differs")
        values[branch] = item.detach().float()
    native_correct: dict[str, torch.Tensor] = {}
    for branch in BRANCH_ORDER:
        item = native_correct_by_branch[branch]
        if (
            not isinstance(item, torch.Tensor)
            or item.shape != reference.shape
            or item.device != reference.device
            or item.requires_grad
            or item.grad_fn is not None
            or not item.is_floating_point()
            or not bool(torch.isfinite(item).all().item())
        ):
            raise PairV6SCAIDError(f"native correct-source field {branch} differs")
        native_correct[branch] = item.detach().float()
    native_wrong: dict[str, torch.Tensor] = {}
    for branch in IDENTITY_BINDING_BRANCHES:
        item = native_wrong_source_by_branch[branch]
        if (
            not isinstance(item, torch.Tensor)
            or item.shape != reference.shape
            or item.device != reference.device
            or item.requires_grad
            or item.grad_fn is not None
            or not item.is_floating_point()
            or not bool(torch.isfinite(item).all().item())
        ):
            raise PairV6SCAIDError(f"native wrong-source field {branch} differs")
        native_wrong[branch] = item.detach().float()
    identity_controls: dict[str, torch.Tensor] = {}
    for name in IDENTITY_CONTROL_DIRECTION_ORDER:
        item = native_identity_directions[name]
        if (
            not isinstance(item, torch.Tensor)
            or item.shape != reference.shape
            or item.device != reference.device
            or item.requires_grad
            or item.grad_fn is not None
            or not item.is_floating_point()
            or not bool(torch.isfinite(item).all().item())
        ):
            raise PairV6SCAIDError(f"native identity control {name} differs")
        identity_controls[name] = item.detach().float()

    negative_stack = torch.stack([values[name] for name in NEGATIVE_BRANCHES], dim=0)
    robust_center = negative_stack.median(dim=0).values
    raw = values[mace.ACTION_BRANCH] - robust_center
    nuisance = {
        "camera_only_minus_noop": values["camera_only"] - values["noop"],
        "appearance_only_minus_noop": values["appearance_only"] - values["noop"],
        "noop_minus_robust_negative_center": values["noop"] - robust_center,
        "identity_binding_action": native_correct[mace.ACTION_BRANCH] - native_wrong[mace.ACTION_BRANCH],
        "identity_binding_noop": native_correct["noop"] - native_wrong["noop"],
        "native_reference_dI": identity_controls["native_reference_dI"],
    }

    # Temporal centering is the exact orthogonal projection away from the
    # temporal-DC subspace.  All later basis vectors are centered as well, so
    # Gram-Schmidt projection cannot reintroduce a static/camera-layout term.
    projected = _temporal_center(raw).reshape(-1)
    basis: list[torch.Tensor] = []
    basis_names: list[str] = []
    skipped: list[str] = []
    for name in (
        *NUISANCE_DIRECTION_ORDER,
        *IDENTITY_CONTROL_DIRECTION_ORDER,
        "identity_binding_action",
        "identity_binding_noop",
    ):
        direction = _temporal_center(nuisance[name]).reshape(-1)
        original_norm = torch.linalg.vector_norm(direction)
        if float(original_norm.item()) <= cfg.direction_min_rms * math.sqrt(direction.numel()):
            skipped.append(name)
            continue
        residual = direction
        for unit in basis:
            residual = residual - torch.dot(residual, unit) * unit
        residual_norm = torch.linalg.vector_norm(residual)
        if float((residual_norm / original_norm).item()) < cfg.direction_independence_ratio:
            skipped.append(name)
            continue
        basis.append(residual / residual_norm)
        basis_names.append(name)
    for unit in basis:
        projected = projected - torch.dot(projected, unit) * unit
    projected_tensor = projected.reshape_as(raw)
    dc_after = projected_tensor.mean(dim=2, keepdim=True)

    reference_rms = torch.maximum(
        _rms(native_correct[mace.ACTION_BRANCH]), _rms(robust_center)
    )
    cap = torch.minimum(
        reference_rms * cfg.native_base_rms_ratio,
        reference_rms.new_tensor(cfg.absolute_residual_rms_cap),
    )
    projected_rms = _rms(projected_tensor)
    scale = torch.minimum(
        projected_rms.new_tensor(1.0),
        cap / projected_rms.clamp_min(cfg.epsilon),
    )
    bounded = (projected_tensor * scale).detach().contiguous()
    bounded_rms = float(_rms(bounded).item())
    if not math.isfinite(bounded_rms) or bounded_rms <= cfg.minimum_residual_rms:
        raise PairV6SCAIDError("no safe non-nuisance source-coordinate action residual")
    dots = {
        name: float(torch.dot(bounded.reshape(-1), unit).abs().item())
        for name, unit in zip(basis_names, basis)
    }
    bounded_norm = torch.linalg.vector_norm(bounded.reshape(-1)).clamp_min(cfg.epsilon)
    cosines = {name: value / float(bounded_norm.item()) for name, value in dots.items()}
    return SafeActionResidual(
        vector=bounded,
        robust_negative_center=robust_center.detach().contiguous(),
        raw_vector=raw.detach().contiguous(),
        accepted_directions=tuple(basis_names),
        skipped_directions=tuple(skipped),
        projection_dot_after=dots,
        projection_cosine_after=cosines,
        temporal_dc_rms_after=float(_rms(dc_after).item()),
        raw_rms=float(_rms(raw).item()),
        projected_rms=float(projected_rms.item()),
        bounded_rms=bounded_rms,
        rms_cap=float(cap.item()),
        bound_scale=float(scale.item()),
    )


@dataclass(frozen=True)
class ResidualSurvivalReceipt:
    optimizer_authorized: bool
    projected_survival_ratio: float
    maximum_projection_cosine: float
    temporal_dc_relative_rms: float
    receipt: Mapping[str, Any]


def build_residual_survival_receipt(
    safe: SafeActionResidual,
    *,
    config: Optional[SCAIDConfig] = None,
) -> ResidualSurvivalReceipt:
    """Fail closed unless projection leaves a finite, orthogonal, bounded signal."""

    if not isinstance(safe, SafeActionResidual):
        raise PairV6SCAIDError("residual survival audit requires SafeActionResidual")
    cfg = SCAIDConfig() if config is None else config
    cfg.validate()
    survival = safe.projected_rms / max(safe.raw_rms, cfg.epsilon)
    maximum_cosine = max(safe.projection_cosine_after.values(), default=0.0)
    dc_relative = safe.temporal_dc_rms_after / max(safe.bounded_rms, cfg.epsilon)
    finite = all(
        math.isfinite(value)
        for value in (
            safe.raw_rms,
            safe.projected_rms,
            safe.bounded_rms,
            safe.rms_cap,
            safe.bound_scale,
            survival,
            maximum_cosine,
            dc_relative,
        )
    )
    authorized = (
        finite
        and safe.raw_rms > 0.0
        and safe.projected_rms >= cfg.minimum_residual_rms
        and safe.bounded_rms >= cfg.minimum_residual_rms
        and safe.bounded_rms <= safe.rms_cap * (1.0 + 1.0e-5)
        and 0.0 < safe.bound_scale <= 1.0
        and survival >= cfg.minimum_projected_survival_ratio
        and maximum_cosine <= cfg.maximum_projection_cosine
        and dc_relative <= cfg.maximum_temporal_dc_relative_rms
        and "native_reference_dI" in safe.accepted_directions
        and any(
            name in safe.accepted_directions
            for name in ("identity_binding_action", "identity_binding_noop")
        )
    )
    value = {
        "schema_version": SURVIVAL_RECEIPT_SCHEMA,
        "raw_residual_tensor_sha256": tensor_sha256(safe.raw_vector),
        "safe_residual_tensor_sha256": tensor_sha256(safe.vector),
        "raw_rms": safe.raw_rms,
        "projected_rms": safe.projected_rms,
        "bounded_rms": safe.bounded_rms,
        "rms_cap": safe.rms_cap,
        "bound_scale": safe.bound_scale,
        "projected_survival_ratio": survival,
        "minimum_projected_survival_ratio": cfg.minimum_projected_survival_ratio,
        "projection_cosine_after": dict(safe.projection_cosine_after),
        "maximum_observed_projection_cosine": maximum_cosine,
        "maximum_allowed_projection_cosine": cfg.maximum_projection_cosine,
        "temporal_dc_relative_rms": dc_relative,
        "maximum_temporal_dc_relative_rms": cfg.maximum_temporal_dc_relative_rms,
        "all_values_finite": finite,
        "native_reference_dI_accepted": (
            "native_reference_dI" in safe.accepted_directions
        ),
        "identity_binding_action_or_noop_accepted": any(
            name in safe.accepted_directions
            for name in ("identity_binding_action", "identity_binding_noop")
        ),
        "optimizer_authorized": authorized,
    }
    receipt = {**value, "receipt_digest": object_sha256(value)}
    if not authorized:
        raise PairV6SCAIDError(
            "source-coordinate action residual failed survival/orthogonality gate"
        )
    return ResidualSurvivalReceipt(
        optimizer_authorized=True,
        projected_survival_ratio=survival,
        maximum_projection_cosine=maximum_cosine,
        temporal_dc_relative_rms=dc_relative,
        receipt=receipt,
    )


@dataclass(frozen=True)
class SCAIDObjective:
    loss: torch.Tensor
    action_match_loss: torch.Tensor
    negative_parity_loss: torch.Tensor
    action_base_trust_loss: torch.Tensor
    parity_by_branch: Mapping[str, torch.Tensor]
    composite_teacher: torch.Tensor
    safe_residual: SafeActionResidual
    survival: ResidualSurvivalReceipt
    native_student_by_branch: Mapping[str, torch.Tensor]
    leaf_vjp_mode: bool
    receipt: Mapping[str, Any]


def build_scaid_objective(
    coordinate: SourceCoordinate,
    t2v_base_by_branch: Mapping[str, torch.Tensor],
    native_correct_by_branch: Mapping[str, torch.Tensor],
    native_wrong_source_by_branch: Mapping[str, torch.Tensor],
    native_identity_directions: Mapping[str, torch.Tensor],
    native_student_by_branch: Mapping[str, torch.Tensor],
    *,
    config: Optional[SCAIDConfig] = None,
    leaf_vjp_mode: bool = False,
) -> SCAIDObjective:
    if coordinate.gate_name == "low_base_only" or coordinate.gate_weight <= 0.0:
        raise PairV6SCAIDError("low-sigma cells cannot construct an objective")
    cfg = SCAIDConfig() if config is None else config
    cfg.validate()
    if set(native_student_by_branch) != set(BRANCH_ORDER):
        raise PairV6SCAIDError("native student branch closure differs")
    safe = build_safe_action_residual(
        t2v_base_by_branch,
        native_correct_by_branch,
        native_wrong_source_by_branch,
        native_identity_directions,
        config=cfg,
    )
    survival = build_residual_survival_receipt(safe, config=cfg)
    base_action = native_correct_by_branch[mace.ACTION_BRANCH].detach().float()
    observable_residual = (
        safe.vector * cfg.action_residual_weight * coordinate.gate_weight
    )
    composite = (base_action + observable_residual).detach().contiguous()
    action_student = native_student_by_branch[mace.ACTION_BRANCH]
    _field(
        action_student,
        coordinate=coordinate,
        label="native student action",
        trainable=True,
        allow_output_leaf=leaf_vjp_mode,
    )
    action_match = functional.mse_loss(action_student.float(), composite)
    parity: dict[str, torch.Tensor] = {}
    for branch in NEGATIVE_BRANCHES:
        student = _field(
            native_student_by_branch[branch],
            coordinate=coordinate,
            label=f"native student {branch}",
            trainable=True,
            allow_output_leaf=leaf_vjp_mode,
        )
        parity[branch] = functional.mse_loss(
            student.float(), native_correct_by_branch[branch].detach().float()
        )
    negative_parity = torch.stack(tuple(parity.values())).mean()
    action_correction = action_student.float() - base_action
    trust_cap = safe.vector.new_tensor(
        safe.bounded_rms * cfg.action_residual_weight * coordinate.gate_weight
    )
    action_trust = torch.relu(
        _smooth_zero_rms(action_correction, cfg.epsilon) - trust_cap
    ).square()
    loss = (
        action_match
        + cfg.negative_parity_weight * negative_parity
        + cfg.action_base_trust_weight * action_trust
    )
    if (
        loss.dtype != torch.float32
        or loss.ndim != 0
        or not loss.requires_grad
        or loss.grad_fn is None
        or not bool(torch.isfinite(loss).item())
    ):
        raise PairV6SCAIDError("SCAID loss must be finite graph-connected FP32")
    value = {
        "schema_version": CELL_RECEIPT_SCHEMA,
        "coordinate_digest": coordinate.coordinate_digest,
        "schedule_index": coordinate.schedule_index,
        "sigma_gate": coordinate.gate_name,
        "sigma_gate_weight": coordinate.gate_weight,
        "teacher": {
            "construction": "detached_native_rv2v4_base_plus_lambda_safe_source_coordinate_t2v_action_residual",
            "action_residual_weight_hex": float(cfg.action_residual_weight).hex(),
            "robust_negative_center": "coordinatewise_median_all_nine_hard_negatives",
            "accepted_projection_directions": list(safe.accepted_directions),
            "skipped_projection_directions": list(safe.skipped_directions),
            "temporal_dc_rms_after": safe.temporal_dc_rms_after,
            "bounded_residual_rms": safe.bounded_rms,
            "rms_cap": safe.rms_cap,
            "detached": True,
            "dry_run_survival_receipt_digest": survival.receipt["receipt_digest"],
        },
        "negative_and_noop_base_parity_branches": list(NEGATIVE_BRANCHES),
        "student_task": "native_rv2v4_source_plus_refs_target_prompt",
        "native_sampler_decomposition": {
            "d_V": "epsilon_V_minus_epsilon_none_contains_old_motion_not_identity_guard",
            "d_I": "epsilon_VI_minus_epsilon_V_four_ref_identity_control_projected",
            "d_TI": "epsilon_VTI_minus_epsilon_VI_text_action_path",
        },
        "t2v_to_rv2v_parameter_transfer": False,
        "pure_t2v_generated_video_role": "calibration_receipt_only",
        "pure_t2v_visual_tensor_consumed": False,
        "mask_flow_pose_track_consumed": False,
        "inference_inputs": [
            "source_video_with_deterministically_derived_four_refs",
            "target_prompt",
            "action_lora",
        ],
        "optimizer_update_authorized": True,
        "leaf_vjp_mode": leaf_vjp_mode,
        "semantic_action_editing_success_claimed": False,
    }
    receipt = {**value, "receipt_digest": object_sha256(value)}
    return SCAIDObjective(
        loss=loss,
        action_match_loss=action_match,
        negative_parity_loss=negative_parity,
        action_base_trust_loss=action_trust,
        parity_by_branch=parity,
        composite_teacher=composite,
        safe_residual=safe,
        survival=survival,
        native_student_by_branch=dict(native_student_by_branch),
        leaf_vjp_mode=leaf_vjp_mode,
        receipt=receipt,
    )


@dataclass(frozen=True)
class SCAIDCell:
    coordinate: SourceCoordinate
    objective: Optional[SCAIDObjective]
    rv2v_prompt_by_branch: Mapping[str, str]
    optimizer_authorized: bool
    zero_update: bool
    receipt: Mapping[str, Any]


def run_scaid_cell(
    source_clean_latent: torch.Tensor,
    official_epsilon: torch.Tensor,
    *,
    schedule_index: int,
    authoritative_evidence_path: str | Path,
    expected_authoritative_evidence_sha256: str,
    fit_candidate_id: str,
    raw_caption_by_branch: Mapping[str, str],
    expected_raw_caption_bank_sha256: str,
    checkpoint_tree_sha256: str,
    frozen_t2v_callback: Callable[[T2VFieldRequest], torch.Tensor],
    native_callback: Callable[[NativeFieldRequest], torch.Tensor],
    config: Optional[SCAIDConfig] = None,
    leaf_vjp_mode: bool = False,
) -> SCAIDCell:
    """Run one source-coordinate cell; indices 38/39 invoke no callbacks."""

    # The optimizer-facing API never accepts a caller-constructed capability.
    # It recomputes the complete authoritative graph for this exact fit event.
    authorization = load_authoritative_v3_authorization(
        authoritative_evidence_path,
        expected_evidence_sha256=expected_authoritative_evidence_sha256,
        checkpoint_tree_sha256=checkpoint_tree_sha256,
        fit_candidate_id=fit_candidate_id,
    )
    raw_captions = _prompts(raw_caption_by_branch)
    raw_digest = object_sha256(raw_captions)
    if raw_digest != _sha256(
        expected_raw_caption_bank_sha256,
        label="raw caption bank SHA-256",
    ):
        raise PairV6SCAIDError("sealed raw caption bank digest differs")
    t2v_prompts, rv2v_prompts, prompt_receipt = _official_task_prompt_banks(
        raw_captions
    )
    if (
        authorization.checkpoint_tree_sha256 != checkpoint_tree_sha256
        or authorization.prompt_bank_sha256 != object_sha256(t2v_prompts)
    ):
        raise PairV6SCAIDError(
            "cell checkpoint or T2V prompt bank differs from recomputed fit event"
        )
    x_sigma, sigma, timestep, gate_name, gate_weight = _registered_coordinate(
        source_clean_latent, official_epsilon, schedule_index
    )
    coordinate_value = {
        "fit_candidate_id": authorization.fit_candidate_id,
        "source_clean_latent_sha256": tensor_sha256(source_clean_latent),
        "official_epsilon_sha256": tensor_sha256(official_epsilon),
        "schedule_index": schedule_index,
        "sigma_float32_be_hex": struct.pack("!f", float(sigma.item())).hex(),
        "timestep_float32_be_hex": struct.pack("!f", float(timestep.item())).hex(),
        "construction": "(1-sigma)*source_clean_latent+sigma*official_epsilon",
    }
    coordinate = SourceCoordinate(
        x_sigma=x_sigma,
        sigma=sigma,
        timestep=timestep,
        schedule_index=schedule_index,
        gate_name=gate_name,
        gate_weight=gate_weight,
        coordinate_digest=object_sha256(coordinate_value),
        x_sigma_object_id=id(x_sigma),
        x_sigma_version=int(x_sigma._version),
    )
    coordinate_receipt = {
        **coordinate_value,
        "coordinate_digest": coordinate.coordinate_digest,
        "x_sigma_tensor_sha256": tensor_sha256(x_sigma),
    }
    if gate_name == "low_base_only":
        value = {
            "schema_version": CELL_RECEIPT_SCHEMA,
            "coordinate_digest": coordinate.coordinate_digest,
            "schedule_index": schedule_index,
            "update_kind": "frozen_native_base_anchor_zero_update",
            "callbacks_invoked": 0,
            "task_prompt_construction_receipt": dict(prompt_receipt),
            "source_coordinate_receipt": coordinate_receipt,
            "low_sigma_native_identity_semantics": "action_residual_zero_frozen_dI_preserved",
            "loss_constructed": False,
            "backward_authorized": False,
            "optimizer_step_authorized": False,
        }
        return SCAIDCell(
            coordinate=coordinate,
            objective=None,
            rv2v_prompt_by_branch=MappingProxyType(dict(rv2v_prompts)),
            optimizer_authorized=False,
            zero_update=True,
            receipt={**value, "receipt_digest": object_sha256(value)},
        )
    if not callable(frozen_t2v_callback) or not callable(native_callback):
        raise PairV6SCAIDError("both field callbacks must be callable")

    t2v: dict[str, torch.Tensor] = {}
    correct: dict[str, torch.Tensor] = {}
    wrong: dict[str, torch.Tensor] = {}
    identity_directions: dict[str, torch.Tensor] = {}
    student: dict[str, torch.Tensor] = {}
    ordinal = 0
    for branch in BRANCH_ORDER:
        with torch.no_grad():
            observed = frozen_t2v_callback(
                T2VFieldRequest(coordinate, branch, t2v_prompts[branch], ordinal)
            )
        coordinate.assert_unchanged()
        t2v[branch] = _field(
            observed.detach(), coordinate=coordinate, label=f"frozen T2V {branch}", trainable=False
        )
        ordinal += 1
    with torch.no_grad():
        observed = native_callback(
            NativeFieldRequest(
                coordinate,
                mace.ACTION_BRANCH,
                rv2v_prompts[mace.ACTION_BRANCH],
                "correct",
                False,
                "frozen_native_reference_identity_control_dI",
                ordinal,
            )
        )
    coordinate.assert_unchanged()
    identity_directions["native_reference_dI"] = _field(
        observed.detach(),
        coordinate=coordinate,
        label="native four-reference d_I identity control",
        trainable=False,
    )
    ordinal += 1
    for branch in BRANCH_ORDER:
        with torch.no_grad():
            observed = native_callback(
                NativeFieldRequest(
                    coordinate, branch, rv2v_prompts[branch], "correct", False,
                    "frozen_native_base", ordinal,
                )
            )
        coordinate.assert_unchanged()
        correct[branch] = _field(
            observed.detach(), coordinate=coordinate, label=f"native correct {branch}", trainable=False
        )
        ordinal += 1
    for branch in IDENTITY_BINDING_BRANCHES:
        with torch.no_grad():
            observed = native_callback(
                NativeFieldRequest(
                    coordinate, branch, rv2v_prompts[branch], "wrong", False,
                    "frozen_identity_binding", ordinal,
                )
            )
        coordinate.assert_unchanged()
        wrong[branch] = _field(
            observed.detach(), coordinate=coordinate, label=f"native wrong {branch}", trainable=False
        )
        ordinal += 1
    for branch in BRANCH_ORDER:
        request = NativeFieldRequest(
            coordinate, branch, rv2v_prompts[branch], "correct", True,
            "native_student_measurement_leaf" if leaf_vjp_mode else "native_action_lora_student",
            ordinal,
        )
        if leaf_vjp_mode:
            with torch.no_grad():
                measured = native_callback(request)
            observed = measured.detach().requires_grad_(True)
        else:
            observed = native_callback(request)
        coordinate.assert_unchanged()
        student[branch] = _field(
            observed,
            coordinate=coordinate,
            label=f"native student {branch}",
            trainable=True,
            allow_output_leaf=leaf_vjp_mode,
        )
        ordinal += 1
    objective = build_scaid_objective(
        coordinate,
        t2v,
        correct,
        wrong,
        identity_directions,
        student,
        config=config,
        leaf_vjp_mode=leaf_vjp_mode,
    )
    value = {
        **objective.receipt,
        "authoritative_v3_authorization_digest": authorization.authorization_digest,
        "authoritative_evidence_file_sha256": authorization.evidence_file_sha256,
        "recomputed_calibration_receipt_digest": authorization.calibration_receipt_digest,
        "task_prompt_construction_receipt": dict(prompt_receipt),
        "source_coordinate_receipt": coordinate_receipt,
        "same_source_coordinate_request_wrapper_object_all_callbacks": True,
        "leaf_vjp_mode": leaf_vjp_mode,
        "call_counts": {
            "frozen_t2v": len(BRANCH_ORDER),
            "frozen_native_correct": len(BRANCH_ORDER),
            "frozen_native_reference_identity_control_dI": 1,
            "frozen_native_wrong_identity_span": len(IDENTITY_BINDING_BRANCHES),
            "native_student": len(BRANCH_ORDER),
        },
    }
    value.pop("receipt_digest", None)
    return SCAIDCell(
        coordinate=coordinate,
        objective=objective,
        rv2v_prompt_by_branch=MappingProxyType(dict(rv2v_prompts)),
        optimizer_authorized=True,
        zero_update=False,
        receipt={**value, "receipt_digest": object_sha256(value)},
    )


def replay_native_student_vjp(
    cell: SCAIDCell,
    native_callback: Callable[[NativeFieldRequest], torch.Tensor],
    *,
    rtol: float = 2.0e-5,
    atol: float = 2.0e-5,
) -> Mapping[str, float]:
    """Replay one native branch graph at a time from measured output cotangents."""

    if (
        not isinstance(cell, SCAIDCell)
        or cell.objective is None
        or not cell.optimizer_authorized
        or not cell.objective.leaf_vjp_mode
    ):
        raise PairV6SCAIDError("serial native VJP requires an authorized leaf-mode cell")
    prompts = _prompts(cell.rv2v_prompt_by_branch)
    if object_sha256(prompts) != cell.receipt.get(
        "task_prompt_construction_receipt", {}
    ).get("rv2v_task_prompt_bank_sha256"):
        raise PairV6SCAIDError("cell RV2V task prompt binding changed before replay")
    replay_component = getattr(native_callback, "replay_component", None)
    if not callable(native_callback) or not callable(replay_component):
        raise PairV6SCAIDError(
            "native callback must expose component-serial replay_component"
        )
    coordinate = cell.coordinate
    maxima: dict[str, float] = {}
    for ordinal, branch in enumerate(BRANCH_ORDER):
        leaf = cell.objective.native_student_by_branch[branch]
        cotangent = leaf.grad
        if cotangent is None:
            raise PairV6SCAIDError(
                "objective.backward() must populate every native output cotangent"
            )
        request = NativeFieldRequest(
            coordinate,
            branch,
            prompts[branch],
            "correct",
            True,
            "native_student_component_serial_vjp_replay",
            ordinal,
        )
        detached_components: dict[str, torch.Tensor] = {}
        for component_name, coefficient in NATIVE_GUIDANCE_COMPONENTS:
            replayed = replay_component(request, component_name)
            coordinate.assert_unchanged()
            replayed = _field(
                replayed,
                coordinate=coordinate,
                label=f"native replay {branch}/{component_name}",
                trainable=True,
            )
            detached_components[component_name] = replayed.detach().float()
            torch.autograd.backward(
                replayed,
                grad_tensors=cotangent.detach().to(replayed.dtype)
                * float(coefficient),
            )
            del replayed
        replayed_aggregate = aggregate_native_guidance_components(
            detached_components
        )
        difference = float(
            (replayed_aggregate - leaf.detach().float()).abs().max().item()
        )
        maxima[branch] = difference
        if not torch.allclose(
            replayed_aggregate, leaf.detach().float(), rtol=rtol, atol=atol
        ):
            raise PairV6SCAIDError(f"native VJP replay changed measured branch {branch}")
    return maxima


def contract_receipt() -> Mapping[str, Any]:
    public_functions = (
        load_authoritative_v3_authorization,
        build_safe_action_residual,
        build_residual_survival_receipt,
        build_scaid_objective,
        run_scaid_cell,
        replay_native_student_vjp,
    )
    public_names = {
        name
        for function in public_functions
        for name in inspect.signature(function).parameters
    }
    forbidden = sorted(public_names & _FORBIDDEN_PUBLIC_ARGUMENTS)
    value = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "source_coordinate": "(1-sigma)*z_source+sigma*official_epsilon",
        "action_field": "frozen_t2v_action_minus_robust_hard_negative_center_on_source_coordinate",
        "joint_projection": [
            *NUISANCE_DIRECTION_ORDER,
            "temporal_dc_subspace",
            "native_reference_dI_equals_epsilon_VI_minus_epsilon_V",
            "native_correct_minus_wrong_source_action_and_noop_span",
        ],
        "composite_teacher": "detached_native_rv2v4_base_plus_lambda_safe_action_residual",
        "student_training_path": "native_rv2v4_direct",
        "native_dV_policy": "not_used_as_identity_control_because_full_video_contains_old_motion",
        "low_sigma_policy": "zero_action_update_preserve_frozen_native_dI",
        "t2v_to_rv2v_parameter_transfer": False,
        "pure_t2v_generated_videos": "calibration_receipts_only",
        "low_sigma_zero_update_indices": list(action_adapter.LOW_SIGMA_INDICES),
        "fit_optimizer_confirmation_holdout": True,
        "pre_optimizer_residual_survival_gate": True,
        "authorization_boundary": "authoritative_v3_recompute_no_self_sealed_booleans_or_hashes",
        "task_prompt_boundary": "sealed_raw_caption_bank_internal_t2v_and_rv2v_rebuild",
        "native_cfg_aggregation": {
            "component_order": [name for name, _ in NATIVE_GUIDANCE_COMPONENTS],
            "coefficients": [value for _, value in NATIVE_GUIDANCE_COMPONENTS],
            "dtype": "float32",
            "shared_by_measurement_and_replay": True,
        },
        "correct_source_identity": "evidence_bound_fit_action_geometry_anchor",
        "public_api_forbidden_inputs": forbidden,
        "public_api_forbidden_inputs_absent": not forbidden,
        "mask_flow_pose_track_trajectory_consumed": False,
        "inference_inputs": [
            "source_video_with_deterministically_derived_four_refs",
            "target_prompt",
            "action_lora",
        ],
        "scientific_success_claimed": False,
    }
    return {**value, "digest": object_sha256(value)}
