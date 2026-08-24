#!/usr/bin/env python3
"""Preregistered ABI for the v4 observer-only partial object graph.

This registry deliberately contains no renderer or training entry point.  It
defines the roles, sparse edge hypotheses, tensor cells and fixed admission
thresholds consumed by ``self_generated_partial_object_graph_observer_v4``.

An instruction may introduce a role query, but it cannot make the role
visually observed.  ``instruction_only`` and ``offscreen_effector`` roles are
therefore always unresolved in the public graph.  The implicit dustbin is not
an object slot and absorbs unsupported patches without a lower-bound quota.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence


METHOD = "bernini-self-generated-partial-object-graph-observer-v4"
REGISTRY_SCHEMA = "bernini-self-generated-partial-object-graph-registry-v4"
CAPTURE_SCHEMA = "bernini-self-generated-same-state-middle-cell-v4"
REDUCED_SCHEMA = "bernini-self-generated-partial-object-cell-v4"
RESULT_SCHEMA = "bernini-self-generated-partial-object-graph-v4"
RECEIPT_SCHEMA = "bernini-self-generated-partial-object-graph-receipt-v4"

DEFAULT_PHASES = 21
TEXT_TOKEN_COUNT = 512
BLOCKS = (6, 12, 18, 24)
SIGMA_BANDS = ("high", "mid", "mid_low")
ARMS = ("action", "noop", "reverse", "static")
MAX_ROLES = 8
APPEARANCE_COUNT = 3

EVIDENCE_MODES = (
    "visual_candidate",
    "instruction_only",
    "offscreen_effector",
)
SUPPORT_FRAME_ROLES = ("none", "start", "end")
ALLOWED_EDGE_TYPES = (
    "latent_affinity",
    "approaching",
    "receding",
    "near",
    "instruction_relation_unresolved",
)

# Every threshold that can affect admission is centralized here.  V4 does not
# tune these values after seeing a locked receipt.
ADMISSION_THRESHOLDS = MappingProxyType(
    {
        "simplex_atol": 2.0e-4,
        "simplex_rtol": 2.0e-4,
        "absolute_role_probability_min": 1.0e-4,
        "absolute_log_prior_enrichment_min": 1.0,
        "absolute_prior_equalized_probability_min": 0.30,
        "absolute_role_competitor_margin_min": 0.08,
        "duplicate_role_distribution_max_abs_max": 1.0e-6,
        "assignment_temperature": 0.35,
        "topk_fraction": 0.25,
        "dustbin_logit": 0.0,
        "role_capacity_fraction": 0.40,
        "partial_assignment_iterations": 12,
        "role_mass_fraction_min": 0.035,
        "role_peak_probability_min": 0.30,
        "role_competitor_margin_min": 0.08,
        "role_concentration_min": 0.10,
        "persistent_run_phases_min": 2,
        "centroid_jump_max": 0.80,
        "support_frame_phases_min": 2,
        "support_frame_scale_min": 0.08,
        "support_frame_huber_delta": 0.20,
        "support_frame_irls_iterations": 8,
        "support_frame_cosine_min": 0.80,
        "support_frame_inlier_fraction_min": 0.60,
        "shared_frame_endpoint_rms_max": 0.10,
        "shared_frame_direction_cosine_min": 0.95,
        "shared_frame_log_scale_abs_max": 0.15,
        "shared_frame_phase_fraction_min": 0.50,
        "cross_block_count_min": 2,
        "cross_sigma_count_min": 2,
        "critical_role_phase_fraction_min": 0.50,
        "action_delta_rms_min": 1.0e-3,
        "dynamic_over_null_ratio_min": 1.50,
        "null_transition_ratio_max": 0.60,
        "reverse_cycle_cosine_min": 0.95,
        "reverse_cycle_distance_max": 0.15,
        "reverse_endpoint_topology_rms_max": 0.15,
        "reverse_endpoint_topology_max_abs_max": 0.15,
        "appearance_cosine_min": 0.95,
        "appearance_distance_max": 0.15,
        "appearance_common_valid_fraction_min": 0.75,
        "change_point_delta_min": 0.05,
    }
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY_FRAGMENTS = (
    "target_type",
    "target_video",
    "target_rgb",
    "target_frame",
    "target_latent",
    "target_hidden",
    "target_q",
    "target_k",
    "target_v",
    "target_mask",
    "target_flow",
    "target_track",
    "teacher_video",
    "teacher_rgb",
    "real_target",
)
_FORBIDDEN_VALUE_TYPES = {
    "target",
    "real_target",
    "target_teacher_only",
    "target_video",
}


class PartialObjectGraphRegistryV4Error(ValueError):
    """A registry, provenance, or non-leakage invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PartialObjectGraphRegistryV4Error(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PartialObjectGraphRegistryV4Error(
            f"{label} must be a lowercase SHA256"
        )
    return value


def require_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PartialObjectGraphRegistryV4Error(
            f"{label} must be a canonical identifier"
        )
    return value


def assert_no_target_payload(value: Any, *, path: str = "metadata") -> None:
    """Reject real-target material or type tags at every public boundary."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PartialObjectGraphRegistryV4Error(
                    f"{path} contains a non-text key"
                )
            folded = key.casefold()
            if folded == "target" or any(
                fragment in folded for fragment in _FORBIDDEN_KEY_FRAGMENTS
            ):
                raise PartialObjectGraphRegistryV4Error(
                    f"forbidden target field at {path}.{key}"
                )
            assert_no_target_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            assert_no_target_payload(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and value.casefold() in _FORBIDDEN_VALUE_TYPES:
        raise PartialObjectGraphRegistryV4Error(
            f"forbidden target type at {path}"
        )


@dataclass(frozen=True)
class RoleSpecV4:
    role_id: str
    semantic_role: str
    evidence_mode: str = "visual_candidate"
    critical: bool = True
    support_frame_role: str = "none"

    def __post_init__(self) -> None:
        require_identifier(self.role_id, label="role_id")
        require_identifier(self.semantic_role, label="semantic_role")
        if self.evidence_mode not in EVIDENCE_MODES:
            raise PartialObjectGraphRegistryV4Error("role evidence_mode differs")
        if type(self.critical) is not bool:
            raise PartialObjectGraphRegistryV4Error("role critical flag differs")
        if self.support_frame_role not in SUPPORT_FRAME_ROLES:
            raise PartialObjectGraphRegistryV4Error(
                "support_frame_role differs"
            )
        if (
            self.support_frame_role != "none"
            and self.evidence_mode != "visual_candidate"
        ):
            raise PartialObjectGraphRegistryV4Error(
                "an unresolved role cannot define a support frame"
            )

    @property
    def can_be_observed(self) -> bool:
        return self.evidence_mode == "visual_candidate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "semantic_role": self.semantic_role,
            "evidence_mode": self.evidence_mode,
            "critical": self.critical,
            "support_frame_role": self.support_frame_role,
        }


@dataclass(frozen=True)
class EdgeSpecV4:
    source_role: str
    target_role: str
    relation_type: str = "latent_affinity"
    critical: bool = True

    def __post_init__(self) -> None:
        require_identifier(self.source_role, label="edge source role")
        require_identifier(self.target_role, label="edge target role")
        if self.source_role == self.target_role:
            raise PartialObjectGraphRegistryV4Error("self edge is not registered")
        if self.relation_type not in ALLOWED_EDGE_TYPES:
            raise PartialObjectGraphRegistryV4Error("edge relation type differs")
        if type(self.critical) is not bool:
            raise PartialObjectGraphRegistryV4Error("edge critical flag differs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_role": self.source_role,
            "target_role": self.target_role,
            "relation_type": self.relation_type,
            "critical": self.critical,
        }


@dataclass(frozen=True)
class ObserverRegistryV4:
    roles: tuple[RoleSpecV4, ...]
    edges: tuple[EdgeSpecV4, ...]
    phases: int = DEFAULT_PHASES
    appearance_count: int = APPEARANCE_COUNT
    requires_support_frame: bool = False
    thresholds: Mapping[str, float] = ADMISSION_THRESHOLDS

    def __post_init__(self) -> None:
        if not 2 <= len(self.roles) <= MAX_ROLES:
            raise PartialObjectGraphRegistryV4Error("role count must be in [2,8]")
        if not isinstance(self.phases, int) or isinstance(self.phases, bool) or self.phases < 3:
            raise PartialObjectGraphRegistryV4Error("phase count is invalid")
        if self.appearance_count != APPEARANCE_COUNT:
            raise PartialObjectGraphRegistryV4Error(
                "exactly three appearance controls are required"
            )
        if type(self.requires_support_frame) is not bool:
            raise PartialObjectGraphRegistryV4Error(
                "requires_support_frame flag differs"
            )
        role_ids = tuple(role.role_id for role in self.roles)
        if len(set(role_ids)) != len(role_ids):
            raise PartialObjectGraphRegistryV4Error("role IDs are not unique")
        if not self.edges:
            raise PartialObjectGraphRegistryV4Error("at least one edge is required")
        edge_ids = set()
        for edge in self.edges:
            if edge.source_role not in role_ids or edge.target_role not in role_ids:
                raise PartialObjectGraphRegistryV4Error(
                    "edge endpoint is absent from role registry"
                )
            identity = (edge.source_role, edge.target_role, edge.relation_type)
            if identity in edge_ids:
                raise PartialObjectGraphRegistryV4Error("duplicate edge")
            edge_ids.add(identity)
        starts = [role for role in self.roles if role.support_frame_role == "start"]
        ends = [role for role in self.roles if role.support_frame_role == "end"]
        if len(starts) > 1 or len(ends) > 1:
            raise PartialObjectGraphRegistryV4Error(
                "support frame endpoints are not unique"
            )
        if self.requires_support_frame and (len(starts), len(ends)) != (1, 1):
            raise PartialObjectGraphRegistryV4Error(
                "required support frame endpoints are missing"
            )
        if set(self.thresholds) != set(ADMISSION_THRESHOLDS):
            raise PartialObjectGraphRegistryV4Error(
                "admission threshold registry differs"
            )
        for key, expected in ADMISSION_THRESHOLDS.items():
            actual = self.thresholds[key]
            if isinstance(actual, bool) or not math.isfinite(float(actual)):
                raise PartialObjectGraphRegistryV4Error(
                    f"threshold {key} is not finite"
                )
            if float(actual) != float(expected):
                raise PartialObjectGraphRegistryV4Error(
                    f"threshold {key} was changed after preregistration"
                )

    @property
    def role_ids(self) -> tuple[str, ...]:
        return tuple(role.role_id for role in self.roles)

    @property
    def support_indices(self) -> tuple[int, int] | None:
        start = [
            index
            for index, role in enumerate(self.roles)
            if role.support_frame_role == "start"
        ]
        end = [
            index
            for index, role in enumerate(self.roles)
            if role.support_frame_role == "end"
        ]
        return (start[0], end[0]) if len(start) == len(end) == 1 else None

    @property
    def digest(self) -> str:
        return object_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA,
            "method": METHOD,
            "roles": [role.as_dict() for role in self.roles],
            "edges": [edge.as_dict() for edge in self.edges],
            "phases": self.phases,
            "appearance_count": self.appearance_count,
            "requires_support_frame": self.requires_support_frame,
            "blocks": list(BLOCKS),
            "sigma_bands": list(SIGMA_BANDS),
            "arms": list(ARMS),
            "thresholds": {key: self.thresholds[key] for key in sorted(self.thresholds)},
            "scientific_claim_authorized": False,
            "renderer_or_injection_authorized": False,
            "target_at_inference_authorized": False,
        }


def make_registry_v4(
    roles: Sequence[RoleSpecV4],
    edges: Sequence[EdgeSpecV4],
    *,
    phases: int = DEFAULT_PHASES,
    requires_support_frame: bool = False,
) -> ObserverRegistryV4:
    return ObserverRegistryV4(
        tuple(roles),
        tuple(edges),
        phases=phases,
        requires_support_frame=requires_support_frame,
    )


__all__ = [
    "ADMISSION_THRESHOLDS",
    "ALLOWED_EDGE_TYPES",
    "APPEARANCE_COUNT",
    "ARMS",
    "BLOCKS",
    "CAPTURE_SCHEMA",
    "DEFAULT_PHASES",
    "EVIDENCE_MODES",
    "EdgeSpecV4",
    "MAX_ROLES",
    "METHOD",
    "ObserverRegistryV4",
    "PartialObjectGraphRegistryV4Error",
    "RECEIPT_SCHEMA",
    "REDUCED_SCHEMA",
    "REGISTRY_SCHEMA",
    "RESULT_SCHEMA",
    "RoleSpecV4",
    "SIGMA_BANDS",
    "TEXT_TOKEN_COUNT",
    "assert_no_target_payload",
    "canonical_json_bytes",
    "make_registry_v4",
    "object_sha256",
    "require_identifier",
    "require_sha256",
]
