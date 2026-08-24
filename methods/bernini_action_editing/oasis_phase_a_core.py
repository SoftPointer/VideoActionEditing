"""Fail-closed contracts for the OASIS Phase-A source-set-noise bank.

Phase A is a matched candidate-generation ablation.  It generates three
independent, full Bernini RV2V rollouts for the same source, edit instruction,
and native seed while changing only the initial-noise arm:

* the untouched native Gaussian (``rho=0``);
* source-set appearance noise at ``rho=0.05``; and
* source-set appearance noise at ``rho=0.10``.

This module deliberately contains no model loader, action scorer, endpoint
selector, optimizer, training loop, paired target, mask, flow, pose, track, or
trajectory interface.  A completed bank is exploratory evidence only.  It
cannot authorize an optimizer update or support an action-editing success
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-oasis-phase-a-source-set-noise-core-v1"
ROLLOUT_SCHEMA = "bernini-oasis-phase-a-source-set-noise-rollout-v2"
FRAME_COUNT = 81
FPS = 25.0
LATENT_CHANNELS = 16
LATENT_PHASES = 21
NATIVE_STEPS = 40
REFERENCE_INDICES = (0, 27, 53, 80)
FAMILY_ORDER = ("dog_sit_hold", "human_stand_hold")
SPLIT_ORDER = ("fit", "confirmation")
NOISE_ARM_ORDER = (
    "official_gaussian",
    "source_appearance_set_rho005",
    "source_appearance_set_rho010",
)
NOISE_RHO_BY_ARM = {
    "official_gaussian": 0.0,
    "source_appearance_set_rho005": 0.05,
    "source_appearance_set_rho010": 0.10,
}
NOISE_OPERATOR_CALLABLE = (
    "motion_null_appearance_noise.build_motion_null_appearance_noise"
)
FORBIDDEN_OPERATOR_INPUTS = frozenset(
    {
        "full_video_latent",
        "target",
        "paired_target",
        "action_proposal",
        "motion_reference",
        "mask",
        "flow",
        "pose",
        "track",
        "trajectory",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_ROLLOUT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "sample_id",
        "sample_digest",
        "source_video_sha256",
        "edit_instruction_sha256",
        "source_instruction_binding_digest",
        "source_conditioning_digest",
        "family",
        "analysis_split",
        "seed",
        "noise_arm",
        "source_carrier_rho",
        "carrier_seed",
        "source_frame_set_digest",
        "source_frame_order_consumed",
        "full_video_latent_consumed_by_carrier",
        "operator_receipt",
        "operator_receipt_digest",
        "operator_runtime_binding",
        "parent_official_gaussian_raw_value_sha256",
        "baseline_artifact",
        "sampler_initial_noise_artifact",
        "external_initial_noise_injection",
        "rho_zero_exact_native_object_forwarded",
        "active_noise_parent_matches_official_control",
        "native_sampling",
        "endpoint",
        "endpoint_candidate_only",
        "legacy_pair_v5_native_rollout_schema_compatible",
        "external_action_scorer_consumed",
        "action_source_scoring_performed",
        "endpoint_selection_performed",
        "optimizer_or_training_authorized",
        "training_performed",
        "scientific_action_editing_success_claim",
        "rollout_digest",
    }
)


class OASISPhaseAError(RuntimeError):
    """A source-set-noise candidate or its provenance is not closed."""


@dataclass(frozen=True)
class MatchedTripletAudit:
    sample_id: str
    sample_digest: str
    seed: int
    source_conditioning_digest: str
    source_frame_set_digest: str
    parent_official_gaussian_raw_value_sha256: str
    active_descriptor_sha256: str
    active_carrier_sha256: str
    candidate_ids: tuple[str, str, str]
    audit_digest: str


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
        raise OASISPhaseAError("value is not finite canonical ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OASISPhaseAError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise OASISPhaseAError(f"{label} is outside the closed identifier grammar")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OASISPhaseAError(f"{label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise OASISPhaseAError(f"{label} must be finite")
    return result


def carrier_seed_for(*, sample_digest: str, seed: int) -> int:
    """Domain-separate the appearance-carrier seed from the sampler seed."""

    digest = _sha(sample_digest, label="sample digest")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise OASISPhaseAError("rollout seed must lie in [0,2^63)")
    payload = f"oasis-source-set-carrier-v1:{digest}:{seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def candidate_id_for(*, sample_id: str, seed: int, noise_arm: str) -> str:
    sample = _safe_id(sample_id, label="sample ID")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise OASISPhaseAError("candidate seed must lie in [0,2^63)")
    if noise_arm not in NOISE_ARM_ORDER:
        raise OASISPhaseAError("candidate noise arm is outside the registry")
    return _safe_id(f"{sample}-s{seed}-{noise_arm}", label="candidate ID")


def _validate_artifact_shape(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        raise OASISPhaseAError(f"{label} artifact must be a mapping")
    _sha(value.get("file_sha256"), label=f"{label} file SHA")
    _sha(value.get("raw_value_sha256"), label=f"{label} raw-value SHA")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 5
        or any(type(item) is not int or item <= 0 for item in shape)
        or tuple(shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
    ):
        raise OASISPhaseAError(f"{label} is not an exact81 latent artifact")
    return tuple(shape)


def _validate_operator_receipt(
    value: Any,
    *,
    expected_digest: Any,
    rho: float,
    carrier_seed: int,
    expected_shape: tuple[int, ...],
) -> tuple[str | None, str | None]:
    if not isinstance(value, Mapping):
        raise OASISPhaseAError("motion-null operator receipt must be a mapping")
    digest = _sha(expected_digest, label="operator receipt digest")
    if object_sha256(value) != digest:
        raise OASISPhaseAError("motion-null operator receipt digest differs")
    forbidden = value.get("forbidden_api_inputs")
    if not isinstance(forbidden, list) or not FORBIDDEN_OPERATOR_INPUTS.issubset(
        set(forbidden)
    ):
        raise OASISPhaseAError("motion-null operator does not close forbidden inputs")
    if (
        value.get("ablation_only") is not True
        or value.get("scientific_claim_authorized") is not False
        or value.get("semantic_old_action_absence_claimed") is not False
    ):
        raise OASISPhaseAError("motion-null operator authority differs")
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise OASISPhaseAError("motion-null diagnostics are missing")
    if (
        _finite_float(diagnostics.get("rho"), label="operator rho") != rho
        or diagnostics.get("carrier_seed") != carrier_seed
        or diagnostics.get("gaussian_shape") != list(expected_shape)
        and diagnostics.get("gaussian_shape") != tuple(expected_shape)
        or diagnostics.get("independent_frame_count") != len(REFERENCE_INDICES)
        or diagnostics.get("source_temporal_indices_consumed") is not False
        or diagnostics.get("source_temporal_phase_consumed") is not False
        or diagnostics.get("source_spatial_phase_consumed") is not False
        or diagnostics.get("source_low_frequency_layout_consumed") is not False
        or diagnostics.get("carrier_strict_temporal_dc") is not True
        or diagnostics.get("numerical_audit_passed") is not True
    ):
        raise OASISPhaseAError("motion-null diagnostics differ")
    descriptor = diagnostics.get("descriptor_sha256")
    carrier = diagnostics.get("carrier_sha256")
    if rho == 0.0:
        if (
            descriptor is not None
            or carrier is not None
            or diagnostics.get("rho_zero_exact_object_alias") is not True
            or diagnostics.get("source_conditioned_non_gaussian") is not False
        ):
            raise OASISPhaseAError("rho0 operator is not the exact native control")
        return None, None
    if (
        _sha(descriptor, label="active appearance descriptor") != descriptor
        or _sha(carrier, label="active appearance carrier") != carrier
        or diagnostics.get("rho_zero_exact_object_alias") is not False
        or diagnostics.get("source_conditioned_non_gaussian") is not True
        or diagnostics.get("carrier_constructed") is not True
    ):
        raise OASISPhaseAError("active source-set carrier diagnostics differ")
    return descriptor, carrier


def _validate_native_sampling(value: Any, *, seed: int) -> None:
    if not isinstance(value, Mapping):
        raise OASISPhaseAError("native sampling receipt must be a mapping")
    required = {
        "num_frames": FRAME_COUNT,
        "num_inference_steps": NATIVE_STEPS,
        "guidance_mode": "rv2v",
        "seed": seed,
        "condition_mode": "rv2v4",
        "guidance_policy": "fixed_native_rv2v_no_ablation",
        "guidance_implementation_replaced": False,
        "sample_one_step_replaced": False,
        "scheduler_replaced": False,
        "exact81": True,
        "exact40": True,
    }
    if any(value.get(field) != expected for field, expected in required.items()):
        raise OASISPhaseAError("native exact81/exact40 sampling receipt differs")


def validate_matched_rollout_triplet(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_id: str,
    sample_digest: str,
    source_video_sha256: str,
    edit_instruction_sha256: str,
    source_conditioning_digest: str,
    source_frame_set_digest: str,
    family: str,
    analysis_split: str,
    seed: int,
) -> MatchedTripletAudit:
    """Validate one same-source/same-instruction/same-seed three-arm cell."""

    if isinstance(rows, (str, bytes)) or len(rows) != len(NOISE_ARM_ORDER):
        raise OASISPhaseAError("matched ablation requires exactly three rollout rows")
    sample = _safe_id(sample_id, label="sample ID")
    sample_sha = _sha(sample_digest, label="sample digest")
    source_sha = _sha(source_video_sha256, label="source video SHA")
    instruction_sha = _sha(edit_instruction_sha256, label="edit instruction SHA")
    source_instruction_sha = object_sha256(
        {
            "source_video_sha256": source_sha,
            "edit_instruction_sha256": instruction_sha,
        }
    )
    condition_sha = _sha(source_conditioning_digest, label="source conditioning digest")
    frame_set_sha = _sha(source_frame_set_digest, label="source frame-set digest")
    if family not in FAMILY_ORDER or analysis_split not in SPLIT_ORDER:
        raise OASISPhaseAError("rollout family/split is outside the registry")
    expected_carrier_seed = carrier_seed_for(sample_digest=sample_sha, seed=seed)

    by_arm: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise OASISPhaseAError("rollout row must be a mapping")
        if set(raw) != set(_ROLLOUT_FIELDS):
            raise OASISPhaseAError("rollout field closure differs")
        row = dict(raw)
        declared = _sha(row.pop("rollout_digest", None), label="rollout digest")
        if object_sha256(row) != declared:
            raise OASISPhaseAError("rollout digest differs")
        arm = raw.get("noise_arm")
        if arm not in NOISE_ARM_ORDER or arm in by_arm:
            raise OASISPhaseAError("rollout arm coverage is duplicate or unknown")
        by_arm[str(arm)] = raw
    if tuple(by_arm) != NOISE_ARM_ORDER:
        raise OASISPhaseAError("rollout rows must preserve registered arm order")

    parent_hashes: set[str] = set()
    descriptor_hashes: set[str] = set()
    carrier_hashes: set[str] = set()
    candidate_ids: list[str] = []
    latent_shape: tuple[int, ...] | None = None
    for arm in NOISE_ARM_ORDER:
        row = by_arm[arm]
        rho = NOISE_RHO_BY_ARM[arm]
        expected_candidate_id = candidate_id_for(
            sample_id=sample, seed=seed, noise_arm=arm
        )
        if any(
            row.get(field) != expected
            for field, expected in (
                ("candidate_id", expected_candidate_id),
                ("sample_id", sample),
                ("sample_digest", sample_sha),
                ("schema_version", ROLLOUT_SCHEMA),
                ("source_video_sha256", source_sha),
                ("edit_instruction_sha256", instruction_sha),
                ("source_instruction_binding_digest", source_instruction_sha),
                ("source_conditioning_digest", condition_sha),
                ("source_frame_set_digest", frame_set_sha),
                ("family", family),
                ("analysis_split", analysis_split),
                ("seed", seed),
                ("noise_arm", arm),
                ("source_carrier_rho", rho),
                ("carrier_seed", expected_carrier_seed),
                ("source_frame_order_consumed", False),
                ("full_video_latent_consumed_by_carrier", False),
                ("endpoint_candidate_only", True),
                ("legacy_pair_v5_native_rollout_schema_compatible", False),
                ("external_action_scorer_consumed", False),
                ("action_source_scoring_performed", False),
                ("endpoint_selection_performed", False),
                ("optimizer_or_training_authorized", False),
                ("training_performed", False),
                ("scientific_action_editing_success_claim", False),
            )
        ):
            raise OASISPhaseAError("matched rollout provenance/authority differs")
        baseline = row.get("baseline_artifact")
        injected = row.get("sampler_initial_noise_artifact")
        baseline_shape = _validate_artifact_shape(
            baseline, label=f"{arm} parent Gaussian"
        )
        injected_shape = _validate_artifact_shape(
            injected, label=f"{arm} sampler noise"
        )
        if baseline_shape != injected_shape or (
            latent_shape is not None and latent_shape != baseline_shape
        ):
            raise OASISPhaseAError("matched arm latent geometry differs")
        latent_shape = baseline_shape
        parent = _sha(
            row.get("parent_official_gaussian_raw_value_sha256"),
            label="official parent Gaussian",
        )
        if baseline.get("raw_value_sha256") != parent:
            raise OASISPhaseAError("parent Gaussian artifact binding differs")
        parent_hashes.add(parent)
        descriptor, carrier = _validate_operator_receipt(
            row.get("operator_receipt"),
            expected_digest=row.get("operator_receipt_digest"),
            rho=rho,
            carrier_seed=expected_carrier_seed,
            expected_shape=baseline_shape,
        )
        if descriptor is not None:
            descriptor_hashes.add(descriptor)
        if carrier is not None:
            carrier_hashes.add(carrier)
        _validate_native_sampling(row.get("native_sampling"), seed=seed)
        runtime_binding = row.get("operator_runtime_binding")
        if not (
            isinstance(runtime_binding, Mapping)
            and runtime_binding.get("callable") == NOISE_OPERATOR_CALLABLE
            and runtime_binding.get("integration_owner")
            == "infer_oasis_phase_a_noise_bank._sample_with_oasis_noise_arm"
            and runtime_binding.get("official_randn_called_first") is True
            and runtime_binding.get("inference_integration_executed") is True
            and runtime_binding.get("operator_self_registered_sampler_hook") is False
        ):
            raise OASISPhaseAError("noise-operator runtime binding differs")
        active = rho > 0.0
        if not active:
            if not (
                row.get("external_initial_noise_injection") is False
                and row.get("rho_zero_exact_native_object_forwarded") is True
                and baseline.get("raw_value_sha256")
                == injected.get("raw_value_sha256")
            ):
                raise OASISPhaseAError("rho0 is not the exact native control")
        elif not (
            row.get("external_initial_noise_injection") is True
            and row.get("rho_zero_exact_native_object_forwarded") is False
            and row.get("active_noise_parent_matches_official_control") is True
            and baseline.get("raw_value_sha256")
            != injected.get("raw_value_sha256")
        ):
            raise OASISPhaseAError("active source-set arm binding differs")
        endpoint = row.get("endpoint")
        if not isinstance(endpoint, Mapping) or any(
            endpoint.get(field) != expected
            for field, expected in (("frame_count", FRAME_COUNT), ("fps", FPS))
        ):
            raise OASISPhaseAError("endpoint exact81 metadata differs")
        normalized = endpoint.get("normalized_clean_latent")
        if (
            not isinstance(normalized, Mapping)
            or normalized.get("shape") != list(baseline_shape)
            or normalized.get("stored_dtype") != "torch.float32"
            or normalized.get("tensor_key") != "normalized_clean_latent"
            or normalized.get("native_sampler_before_vae_decode") is not True
            or normalized.get("mp4_decode_reencode_used") is not False
            or normalized.get("roundtrip_byte_exact_fp32") is not True
        ):
            raise OASISPhaseAError("endpoint clean-latent artifact is missing")
        _sha(normalized.get("sha256"), label="endpoint clean-latent file SHA")
        candidate_ids.append(expected_candidate_id)

    if len(parent_hashes) != 1:
        raise OASISPhaseAError("matched arms do not share one official Gaussian")
    if len(descriptor_hashes) != 1 or len(carrier_hashes) != 1:
        raise OASISPhaseAError(
            "rho=.05/.10 do not share one source descriptor and carrier"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample,
        "sample_digest": sample_sha,
        "seed": seed,
        "family": family,
        "analysis_split": analysis_split,
        "source_video_sha256": source_sha,
        "edit_instruction_sha256": instruction_sha,
        "source_instruction_binding_digest": source_instruction_sha,
        "source_conditioning_digest": condition_sha,
        "source_frame_set_digest": frame_set_sha,
        "parent_official_gaussian_raw_value_sha256": next(iter(parent_hashes)),
        "active_descriptor_sha256": next(iter(descriptor_hashes)),
        "active_carrier_sha256": next(iter(carrier_hashes)),
        "candidate_ids": candidate_ids,
        "same_source_instruction_seed_across_rho": True,
        "candidate_generation_only": True,
        "external_action_scorer_consumed": False,
        "optimizer_authorized": False,
        "scientific_action_editing_success_claim": False,
    }
    digest = object_sha256(payload)
    return MatchedTripletAudit(
        sample_id=sample,
        sample_digest=sample_sha,
        seed=seed,
        source_conditioning_digest=condition_sha,
        source_frame_set_digest=frame_set_sha,
        parent_official_gaussian_raw_value_sha256=next(iter(parent_hashes)),
        active_descriptor_sha256=next(iter(descriptor_hashes)),
        active_carrier_sha256=next(iter(carrier_hashes)),
        candidate_ids=tuple(candidate_ids),  # type: ignore[arg-type]
        audit_digest=digest,
    )


def static_contract() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "families": list(FAMILY_ORDER),
        "splits": list(SPLIT_ORDER),
        "noise_arm_order": list(NOISE_ARM_ORDER),
        "noise_rho_by_arm": dict(NOISE_RHO_BY_ARM),
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "native_steps": NATIVE_STEPS,
        "same_source_instruction_seed_across_rho_required": True,
        "candidate_generation_only": True,
        "paired_target_media_or_latent": False,
        "mask_flow_pose_track_or_trajectory": False,
        "external_action_scorer_dependency": False,
        "endpoint_selection_performed": False,
        "optimizer_authorized": False,
        "scientific_action_editing_success_claim": False,
    }
    return {**value, "receipt_digest": object_sha256(value)}


__all__ = [
    "FAMILY_ORDER",
    "FPS",
    "FRAME_COUNT",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "MatchedTripletAudit",
    "NATIVE_STEPS",
    "NOISE_ARM_ORDER",
    "NOISE_OPERATOR_CALLABLE",
    "NOISE_RHO_BY_ARM",
    "OASISPhaseAError",
    "REFERENCE_INDICES",
    "ROLLOUT_SCHEMA",
    "SCHEMA_VERSION",
    "SPLIT_ORDER",
    "candidate_id_for",
    "canonical_json_bytes",
    "carrier_seed_for",
    "object_sha256",
    "static_contract",
    "validate_matched_rollout_triplet",
]
