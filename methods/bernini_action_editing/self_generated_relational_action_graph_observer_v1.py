#!/usr/bin/env python3
"""Pure, streaming object-relation observer for self-generated Bernini states.

The observer consumes detached visual self-attention Q/K together with
cross-attention role responsibilities.  It reduces every capture immediately
to directed, relative object-pair trajectories and can then zero the owned raw
buffers.  It never decodes video, calls a renderer, changes a model output, or
uses a real target.

Every published relation comes from an explicit allowlisted ``EdgeSpec`` with
an applicability lifecycle.  There is no implicit all-role Cartesian graph:
required edges are evaluated only inside their declared phase window, while
not-applicable edges remain null and cannot contribute to admission or reward.

The scientific boundary is intentionally strict: a complete result is only a
mechanical observer receipt.  Real Bernini capture, cross-case admission and a
causal renderer experiment are separate gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

import torch


METHOD = "self-generated-relational-action-graph-observer-v1"
SCHEMA_VERSION = "bernini-self-generated-relational-action-graph-v1"
PHASES = 21
ARMS = ("action", "noop", "reverse", "static")
SIGMA_BANDS = ("high", "mid", "mid_low")
BLOCKS = (6, 12, 18, 24)
APPEARANCE_COUNT = 3
OWNERSHIP = (
    "source_owned",
    "self_generated_anchor_owned",
    "instruction_introduced",
)
EVIDENCE_MODES = ("observed_internal", "latent_unobserved")
EDGE_APPLICABILITIES = ("required", "not_applicable")
RELATION_TYPES = (
    "relative_motion",
    "approaching_or_receding",
    "supports",
    "grasped_or_attached",
    "releases",
    "part_of",
    "identity_guard",
    "noncontact_articulation",
)
SEMANTIC_ROLES = (
    "human_agent",
    "agent",
    "effector",
    "tool",
    "moving_object",
    "support_surface",
    "support",
    "patient",
    "distractor",
)
FEATURE_NAMES = (
    "signed_visual_qk",
    "relative_dx",
    "relative_dy",
    "relative_vx",
    "relative_vy",
    "relative_distance",
    "relative_overlap_proxy",
    "relative_covariance_xx",
    "relative_covariance_yy",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_EPS = 1.0e-8


class RelationalObserverError(ValueError):
    """A fail-closed observer contract was violated."""


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
        raise RelationalObserverError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise RelationalObserverError(f"{label} is not a canonical identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RelationalObserverError(f"{label} is not lowercase SHA-256")
    return value


def _finite_tensor(
    value: Any,
    *,
    label: str,
    ndim: int,
    nonnegative: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or value.device.type == "meta"
        or value.numel() == 0
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise RelationalObserverError(
            f"{label} must be one detached contiguous finite rank-{ndim} float tensor"
        )
    if nonnegative and bool((value < 0).any().item()):
        raise RelationalObserverError(f"{label} must be non-negative")
    return value


@dataclass(frozen=True)
class RoleSpec:
    role_id: str
    ownership: str
    semantic_role: str = "moving_object"
    evidence_mode: str = "observed_internal"
    first_reliable_phase: int = 0
    source_node_id: Optional[str] = None
    critical: bool = True

    def __post_init__(self) -> None:
        _identifier(self.role_id, "role_id")
        if self.ownership not in OWNERSHIP:
            raise RelationalObserverError("role ownership differs")
        if self.semantic_role not in SEMANTIC_ROLES:
            raise RelationalObserverError("semantic role differs")
        if self.evidence_mode not in EVIDENCE_MODES:
            raise RelationalObserverError("role evidence mode differs")
        if (
            isinstance(self.first_reliable_phase, bool)
            or not isinstance(self.first_reliable_phase, int)
            or not 0 <= self.first_reliable_phase < PHASES
        ):
            raise RelationalObserverError("first reliable phase differs")
        if not isinstance(self.critical, bool):
            raise RelationalObserverError("role critical flag differs")
        if self.source_node_id is not None:
            _identifier(self.source_node_id, "source_node_id")
        if self.ownership == "source_owned":
            if self.first_reliable_phase != 0:
                raise RelationalObserverError(
                    "source-owned role must be reliable from phase zero"
                )
        elif self.ownership == "self_generated_anchor_owned":
            if self.source_node_id is not None:
                raise RelationalObserverError(
                    "self-generated anchor role cannot inherit a source identity"
                )
        else:
            if self.first_reliable_phase <= 0 or self.source_node_id is not None:
                raise RelationalObserverError(
                    "instruction-introduced role cannot inherit a source identity"
                )
        if self.evidence_mode == "latent_unobserved" and self.source_node_id is not None:
            raise RelationalObserverError(
                "latent/offscreen role cannot claim a source mask identity"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "ownership": self.ownership,
            "semantic_role": self.semantic_role,
            "evidence_mode": self.evidence_mode,
            "first_reliable_phase": self.first_reliable_phase,
            "source_node_id": self.source_node_id,
            "critical": self.critical,
            "mask_identity_claimed": False,
            "physical_contact_truth_claimed": False,
        }


@dataclass(frozen=True)
class EdgeSpec:
    source_role: str
    target_role: str
    relation_type: str
    first_applicable_phase: int = 0
    last_applicable_phase: int = PHASES - 1
    applicability: str = "required"

    def __post_init__(self) -> None:
        _identifier(self.source_role, "edge source role")
        _identifier(self.target_role, "edge target role")
        if self.source_role == self.target_role:
            raise RelationalObserverError("typed edge cannot be a self-loop")
        if self.relation_type not in RELATION_TYPES:
            raise RelationalObserverError("typed edge relation differs")
        if self.applicability not in EDGE_APPLICABILITIES:
            raise RelationalObserverError("typed edge applicability differs")
        for label, value in (
            ("first applicable phase", self.first_applicable_phase),
            ("last applicable phase", self.last_applicable_phase),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < PHASES
            ):
                raise RelationalObserverError(f"edge {label} differs")
        if self.first_applicable_phase > self.last_applicable_phase:
            raise RelationalObserverError("typed edge applicability window is empty")

    @property
    def pair(self) -> tuple[str, str]:
        return (self.source_role, self.target_role)

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.source_role, self.target_role, self.relation_type)

    def receipt(self) -> dict[str, Any]:
        return {
            "source_role": self.source_role,
            "target_role": self.target_role,
            "relation_type": self.relation_type,
            "first_applicable_phase": self.first_applicable_phase,
            "last_applicable_phase": self.last_applicable_phase,
            "applicability": self.applicability,
            "contributes_to_reward": self.applicability == "required",
            # The type is a preregistered hypothesis selecting a graph edge;
            # generic Q/K geometry alone is not physical relation ground truth.
            "relation_type_is_preregistered_hypothesis": True,
            "typed_relation_truth_claimed": False,
            "physical_relation_truth_claimed": False,
        }


@dataclass(frozen=True)
class ObserverConfig:
    edge_specs: tuple[EdgeSpec, ...] = ()
    # Compatibility bridge for already written callers.  A non-empty tuple is
    # still an explicit registry; it maps to full-window relative-motion edges.
    critical_edges: tuple[tuple[str, str], ...] = ()
    appearance_count: int = APPEARANCE_COUNT
    action_norm_min: float = 1.0e-5
    reverse_retime_margin: float = 0.02
    static_transition_ratio_max: float = 0.60
    appearance_cosine_min: float = 0.95
    appearance_distance_max: float = 0.15
    role_mass_min: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.appearance_count != APPEARANCE_COUNT:
            raise RelationalObserverError("observer requires exactly three appearances")
        for name in (
            "action_norm_min",
            "reverse_retime_margin",
            "static_transition_ratio_max",
            "appearance_cosine_min",
            "appearance_distance_max",
            "role_mass_min",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise RelationalObserverError(f"{name} differs")
        if not 0 <= self.appearance_cosine_min <= 1:
            raise RelationalObserverError("appearance cosine threshold differs")
        if not isinstance(self.edge_specs, tuple) or not all(
            isinstance(item, EdgeSpec) for item in self.edge_specs
        ):
            raise RelationalObserverError("typed edge registry differs")
        if self.edge_specs and self.critical_edges:
            raise RelationalObserverError(
                "typed and legacy edge registries cannot both be supplied"
            )
        if len({item.identity for item in self.edge_specs}) != len(self.edge_specs):
            raise RelationalObserverError("typed edge registry contains duplicates")
        if not isinstance(self.critical_edges, tuple):
            raise RelationalObserverError("legacy critical edge registry differs")
        for edge in self.critical_edges:
            if not isinstance(edge, tuple) or len(edge) != 2 or edge[0] == edge[1]:
                raise RelationalObserverError("critical edge differs")
            _identifier(edge[0], "critical edge source")
            _identifier(edge[1], "critical edge target")
        if len(set(self.critical_edges)) != len(self.critical_edges):
            raise RelationalObserverError("critical edge list contains duplicates")
        if not self.edge_specs and not self.critical_edges:
            raise RelationalObserverError(
                "observer requires an explicit typed edge registry"
            )
        if self.edge_specs and not any(
            item.applicability == "required" for item in self.edge_specs
        ):
            raise RelationalObserverError(
                "typed edge registry requires at least one required edge"
            )

    def resolved_edge_specs(self) -> tuple[EdgeSpec, ...]:
        if self.edge_specs:
            return self.edge_specs
        return tuple(
            EdgeSpec(
                source_role=source,
                target_role=target,
                relation_type="relative_motion",
            )
            for source, target in self.critical_edges
        )

    @property
    def legacy_edge_registry(self) -> bool:
        return bool(self.critical_edges)


@dataclass(frozen=True)
class CaptureCell:
    appearance_id: str
    arm: str
    sigma_band: str
    block_index: int
    state_sha256: str
    prompt_sha256: str
    patch_height: int
    patch_width: int
    roles: tuple[str, ...]
    queries: torch.Tensor
    keys: torch.Tensor
    responsibilities: torch.Tensor

    def validate(self, registry: Sequence[RoleSpec]) -> None:
        _identifier(self.appearance_id, "appearance_id")
        if self.arm not in ARMS:
            raise RelationalObserverError("capture arm differs")
        if self.sigma_band not in SIGMA_BANDS:
            raise RelationalObserverError("capture sigma band differs")
        if self.block_index not in BLOCKS:
            raise RelationalObserverError("capture block differs")
        _sha256(self.state_sha256, "capture state")
        _sha256(self.prompt_sha256, "capture prompt")
        if (
            isinstance(self.patch_height, bool)
            or isinstance(self.patch_width, bool)
            or not isinstance(self.patch_height, int)
            or not isinstance(self.patch_width, int)
            or self.patch_height <= 0
            or self.patch_width <= 0
        ):
            raise RelationalObserverError("patch geometry differs")
        role_ids = tuple(item.role_id for item in registry)
        if self.roles != role_ids:
            raise RelationalObserverError("capture role order differs")
        query = _finite_tensor(self.queries, label="queries", ndim=3)
        key = _finite_tensor(self.keys, label="keys", ndim=3)
        responsibility = _finite_tensor(
            self.responsibilities,
            label="responsibilities",
            ndim=3,
            nonnegative=True,
        )
        patches = self.patch_height * self.patch_width
        if (
            tuple(query.shape[:2]) != (PHASES, patches)
            or tuple(key.shape) != tuple(query.shape)
            or tuple(responsibility.shape) != (PHASES, len(registry), patches)
        ):
            raise RelationalObserverError("capture tensor geometry differs")
        if query.dtype != key.dtype or query.device != key.device:
            raise RelationalObserverError("visual Q/K dtype or device differs")
        for index, role in enumerate(registry):
            row = responsibility[:, index]
            if role.evidence_mode == "latent_unobserved":
                if int(torch.count_nonzero(row).item()) != 0:
                    raise RelationalObserverError(
                        "latent/offscreen role cannot carry visual responsibility"
                    )
            elif role.first_reliable_phase > 0 and int(
                torch.count_nonzero(row[: role.first_reliable_phase]).item()
            ) != 0:
                raise RelationalObserverError(
                    "instruction-introduced role fabricated preappearance evidence"
                )


@dataclass(frozen=True)
class _ReducedCell:
    key: tuple[str, str, str, int]
    state_sha256: str
    prompt_sha256: str
    edge_ids: tuple[tuple[str, str], ...]
    values: torch.Tensor
    valid: torch.Tensor
    role_confidence: torch.Tensor
    digest: str


def _patch_coordinates(height: int, width: int) -> torch.Tensor:
    y = (torch.arange(height, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.double().reshape(-1)
    b = right.double().reshape(-1)
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denominator <= _EPS:
        return 0.0
    return float(torch.dot(a, b).item() / denominator)


def _normalized_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm((left - right).double()).item())
    denominator = max(
        float(torch.linalg.vector_norm(left.double()).item()),
        float(torch.linalg.vector_norm(right.double()).item()),
        _EPS,
    )
    return numerator / denominator


def _tensor_digest(value: torch.Tensor) -> str:
    owned = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = canonical_json_bytes(
        {"shape": list(map(int, owned.shape)), "dtype": "float32"}
    )
    return hashlib.sha256(header + owned.numpy().tobytes(order="C")).hexdigest()


def _reduce_cell(
    cell: CaptureCell,
    roles: Sequence[RoleSpec],
    config: ObserverConfig,
) -> _ReducedCell:
    q = cell.queries.detach().to(device="cpu", dtype=torch.float32).contiguous()
    k = cell.keys.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw_r = (
        cell.responsibilities.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    role_count = len(roles)
    coordinates = _patch_coordinates(cell.patch_height, cell.patch_width)
    mass = raw_r.sum(dim=-1)
    normalized = raw_r / mass.clamp_min(_EPS).unsqueeze(-1)
    observed_mode = torch.tensor(
        [item.evidence_mode == "observed_internal" for item in roles],
        dtype=torch.bool,
    ).reshape(1, role_count)
    phase_index = torch.arange(PHASES).reshape(PHASES, 1)
    first_phase = torch.tensor(
        [item.first_reliable_phase for item in roles], dtype=torch.long
    ).reshape(1, role_count)
    valid_role = observed_mode & (phase_index >= first_phase) & (
        mass >= float(config.role_mass_min)
    )

    slot_q = torch.einsum("trp,tpd->trd", normalized, q)
    slot_k = torch.einsum("trp,tpd->trd", normalized, k)
    slot_q = torch.nn.functional.normalize(slot_q, dim=-1, eps=_EPS)
    slot_k = torch.nn.functional.normalize(slot_k, dim=-1, eps=_EPS)
    centroids = torch.einsum("trp,pd->trd", normalized, coordinates)
    centered = coordinates.reshape(1, 1, -1, 2) - centroids.unsqueeze(2)
    covariance = torch.einsum(
        "trp,trpi,trpj->trij", normalized, centered, centered
    )
    spatial_entropy = -(
        normalized.clamp_min(_EPS) * normalized.clamp_min(_EPS).log()
    ).sum(dim=-1) / max(math.log(float(cell.patch_height * cell.patch_width)), 1.0)
    confidence = torch.where(
        valid_role,
        (1.0 - spatial_entropy).clamp(0.0, 1.0),
        torch.zeros_like(spatial_entropy),
    )

    # Build only the explicitly registered, reward-bearing graph.  Earlier
    # versions reduced the full observed-role Cartesian product and discarded
    # unregistered pairs later.  Besides wasting memory, that made the claimed
    # sparse graph boundary weaker than the actual computation.  Multiple
    # typed lifecycle hypotheses may share one underlying directed role pair;
    # the kinematic sequence is reduced once and evaluated separately inside
    # each EdgeSpec window during finalize().
    edges = tuple(
        dict.fromkeys(
            item.pair
            for item in config.resolved_edge_specs()
            if item.applicability == "required"
        )
    )
    edge_values = []
    edge_valid = []
    role_index = {item.role_id: index for index, item in enumerate(roles)}
    for source_id, target_id in edges:
        source = role_index[source_id]
        target = role_index[target_id]
        relative = centroids[:, target] - centroids[:, source]
        velocity = torch.cat(
            (torch.zeros((1, 2), dtype=torch.float32), relative[1:] - relative[:-1]),
            dim=0,
        )
        distance = torch.linalg.vector_norm(relative, dim=-1)
        cov_sum = covariance[:, source] + covariance[:, target]
        overlap = torch.exp(
            -0.5
            * (
                relative[:, 0].square() / cov_sum[:, 0, 0].clamp_min(1.0e-4)
                + relative[:, 1].square() / cov_sum[:, 1, 1].clamp_min(1.0e-4)
            )
        ).clamp(0.0, 1.0)
        directed_qk = (slot_q[:, source] * slot_k[:, target]).sum(dim=-1)
        values = torch.stack(
            (
                directed_qk,
                relative[:, 0],
                relative[:, 1],
                velocity[:, 0],
                velocity[:, 1],
                distance,
                overlap,
                covariance[:, target, 0, 0] - covariance[:, source, 0, 0],
                covariance[:, target, 1, 1] - covariance[:, source, 1, 1],
            ),
            dim=-1,
        )
        edge_values.append(values)
        edge_valid.append(valid_role[:, source] & valid_role[:, target])
    values_tensor = torch.stack(edge_values, dim=0).contiguous()
    valid_tensor = torch.stack(edge_valid, dim=0).contiguous()
    payload = {
        "key": [cell.appearance_id, cell.arm, cell.sigma_band, cell.block_index],
        "state_sha256": cell.state_sha256,
        "prompt_sha256": cell.prompt_sha256,
        "edges": [list(item) for item in edges],
        "values_sha256": _tensor_digest(values_tensor),
        "valid_sha256": hashlib.sha256(
            valid_tensor.to(torch.uint8).numpy().tobytes(order="C")
        ).hexdigest(),
        "role_confidence_sha256": _tensor_digest(confidence),
    }
    return _ReducedCell(
        key=(cell.appearance_id, cell.arm, cell.sigma_band, cell.block_index),
        state_sha256=cell.state_sha256,
        prompt_sha256=cell.prompt_sha256,
        edge_ids=edges,
        values=values_tensor,
        valid=valid_tensor,
        role_confidence=confidence.contiguous(),
        digest=object_sha256(payload),
    )


class StreamingRelationalObserver:
    """One-use stream: reduce each capture, optionally scrub its raw buffers."""

    def __init__(
        self,
        *,
        roles: Sequence[RoleSpec],
        config: Optional[ObserverConfig] = None,
    ) -> None:
        self.roles = tuple(roles)
        if not 2 <= len(self.roles) <= 8 or not all(
            isinstance(item, RoleSpec) for item in self.roles
        ):
            raise RelationalObserverError("role registry requires 2..8 RoleSpec rows")
        if len({item.role_id for item in self.roles}) != len(self.roles):
            raise RelationalObserverError("role registry contains duplicate IDs")
        self.config = config if config is not None else ObserverConfig()
        if not isinstance(self.config, ObserverConfig):
            raise RelationalObserverError("observer config differs")
        role_by_id = {item.role_id: item for item in self.roles}
        self.edge_specs = self.config.resolved_edge_specs()
        if not self.edge_specs or not any(
            item.applicability == "required" for item in self.edge_specs
        ):
            raise RelationalObserverError(
                "observer requires at least one explicit required edge"
            )
        for edge in self.edge_specs:
            if (
                edge.source_role not in role_by_id
                or edge.target_role not in role_by_id
            ):
                raise RelationalObserverError(
                    "typed edge references an unknown role"
                )
            if edge.applicability == "required":
                source = role_by_id[edge.source_role]
                target = role_by_id[edge.target_role]
                if (
                    source.evidence_mode != "observed_internal"
                    or target.evidence_mode != "observed_internal"
                ):
                    raise RelationalObserverError(
                        "required edge cannot depend on a latent/offscreen role"
                    )
                reliable_phase = max(
                    source.first_reliable_phase,
                    target.first_reliable_phase,
                )
                if edge.first_applicable_phase < reliable_phase:
                    raise RelationalObserverError(
                        "required edge starts before both roles are observable"
                    )
        self._cells: dict[tuple[str, str, str, int], _ReducedCell] = {}
        self._zeroized = 0
        self._finalized = False

    def add(self, cell: CaptureCell, *, zeroize: bool = True) -> None:
        if self._finalized:
            raise RelationalObserverError("observer stream is already finalized")
        if not isinstance(cell, CaptureCell):
            raise RelationalObserverError("stream received a non-CaptureCell")
        if not isinstance(zeroize, bool):
            raise RelationalObserverError("zeroize flag differs")
        tensors = (cell.queries, cell.keys, cell.responsibilities)
        try:
            cell.validate(self.roles)
            key = (cell.appearance_id, cell.arm, cell.sigma_band, cell.block_index)
            if key in self._cells:
                raise RelationalObserverError("duplicate capture cell")
            reduced = _reduce_cell(cell, self.roles, self.config)
            self._cells[key] = reduced
        finally:
            if zeroize:
                with torch.no_grad():
                    for tensor in tensors:
                        if isinstance(tensor, torch.Tensor) and tensor.device.type != "meta":
                            tensor.zero_()
                if all(
                    isinstance(tensor, torch.Tensor)
                    and int(torch.count_nonzero(tensor).item()) == 0
                    for tensor in tensors
                ):
                    self._zeroized += 1

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise RelationalObserverError("observer finalize is one-use")
        self._finalized = True
        expected_count = APPEARANCE_COUNT * len(ARMS) * len(SIGMA_BANDS) * len(BLOCKS)
        if len(self._cells) != expected_count or self._zeroized != expected_count:
            raise RelationalObserverError(
                f"observer requires exactly {expected_count} reduced and zeroized cells"
            )
        appearances = tuple(sorted({key[0] for key in self._cells}))
        if len(appearances) != APPEARANCE_COUNT:
            raise RelationalObserverError("appearance population differs")
        expected_keys = {
            (appearance, arm, sigma, block)
            for appearance in appearances
            for arm in ARMS
            for sigma in SIGMA_BANDS
            for block in BLOCKS
        }
        if set(self._cells) != expected_keys:
            raise RelationalObserverError("capture matrix is incomplete")
        first = next(iter(self._cells.values()))
        edges = first.edge_ids
        if not edges or any(item.edge_ids != edges for item in self._cells.values()):
            raise RelationalObserverError("reduced edge registry differs across cells")
        for appearance in appearances:
            for sigma in SIGMA_BANDS:
                state = {
                    self._cells[(appearance, arm, sigma, block)].state_sha256
                    for arm in ARMS
                    for block in BLOCKS
                }
                if len(state) != 1:
                    raise RelationalObserverError(
                        "four arms/blocks do not share one sealed noisy state"
                    )
                prompts = {
                    arm: {
                        self._cells[(appearance, arm, sigma, block)].prompt_sha256
                        for block in BLOCKS
                    }
                    for arm in ARMS
                }
                if any(len(value) != 1 for value in prompts.values()) or len(
                    {next(iter(value)) for value in prompts.values()}
                ) != len(ARMS):
                    raise RelationalObserverError(
                        "arm prompts must be stable across blocks and distinct across controls"
                    )

        edge_index = {edge: index for index, edge in enumerate(edges)}
        required_specs = tuple(
            item for item in self.edge_specs if item.applicability == "required"
        )
        not_applicable_specs = tuple(
            item for item in self.edge_specs if item.applicability == "not_applicable"
        )
        if not required_specs or any(
            item.pair not in edge_index for item in required_specs
        ):
            raise RelationalObserverError(
                "required observable edge registry is empty or invalid"
            )
        appearance_packets: dict[str, dict[tuple[str, str, str], Any]] = {}
        all_controls_pass = True
        any_uncertain = False
        for appearance in appearances:
            per_edge: dict[tuple[str, str, str], Any] = {}
            for edge_spec in self.edge_specs:
                edge_identity = edge_spec.identity
                edge_receipt = edge_spec.receipt()
                if edge_spec.applicability == "not_applicable":
                    per_edge[edge_identity] = {
                        **edge_receipt,
                        "edge_status": "not_applicable",
                        "phase_rows": [
                            {
                                "phase_index": phase,
                                "status": "not_applicable",
                                "relative_features": None,
                            }
                            for phase in range(PHASES)
                        ],
                        "control_metrics": None,
                        "control_gates": None,
                        "controls_passed": None,
                        "_action_tensor": None,
                        "_valid_tensor": None,
                    }
                    continue

                index = edge_index[edge_spec.pair]
                applicable = torch.zeros((PHASES,), dtype=torch.bool)
                applicable[
                    edge_spec.first_applicable_phase:
                    edge_spec.last_applicable_phase + 1
                ] = True
                quotient_rows = []
                reverse_rows = []
                static_rows = []
                valid_rows = []
                for sigma in SIGMA_BANDS:
                    for block in BLOCKS:
                        action = self._cells[(appearance, "action", sigma, block)]
                        noop = self._cells[(appearance, "noop", sigma, block)]
                        reverse = self._cells[(appearance, "reverse", sigma, block)]
                        static = self._cells[(appearance, "static", sigma, block)]
                        valid = (
                            action.valid[index]
                            & noop.valid[index]
                            & reverse.valid[index]
                            & static.valid[index]
                        )
                        action_q = action.values[index] - noop.values[index]
                        reverse_q = reverse.values[index] - noop.values[index]
                        static_q = static.values[index] - noop.values[index]
                        # Remove layout/appearance at the first jointly reliable
                        # phase inside this edge's explicit lifecycle window.
                        reliable_indices = torch.nonzero(
                            valid & applicable,
                            as_tuple=False,
                        ).flatten()
                        if int(reliable_indices.numel()) == 0:
                            anchor_phase = edge_spec.first_applicable_phase
                        else:
                            anchor_phase = int(reliable_indices[0].item())
                        quotient_rows.append(action_q - action_q[anchor_phase])
                        reverse_rows.append(reverse_q - reverse_q[anchor_phase])
                        static_rows.append(static_q - static_q[anchor_phase])
                        valid_rows.append(valid)
                stacked_valid = torch.stack(valid_rows, dim=0)
                aggregate_valid = stacked_valid.all(dim=0)
                action_value = torch.stack(quotient_rows, dim=0).median(dim=0).values
                reverse_value = torch.stack(reverse_rows, dim=0).median(dim=0).values
                static_value = torch.stack(static_rows, dim=0).median(dim=0).values
                first_phase = edge_spec.first_applicable_phase
                last_phase = edge_spec.last_applicable_phase
                window_valid = aggregate_valid[first_phase:last_phase + 1]
                if not bool(window_valid.all().item()):
                    any_uncertain = True
                action_window = action_value[first_phase:last_phase + 1]
                reverse_window = reverse_value[first_phase:last_phase + 1]
                static_window = static_value[first_phase:last_phase + 1]
                action_flat = action_window[window_valid].reshape(-1)
                reverse_flat = reverse_window[window_valid].reshape(-1)
                retime_valid = window_valid & torch.flip(window_valid, dims=(0,))
                retime_action_flat = action_window[retime_valid].reshape(-1)
                reversed_reverse = torch.flip(reverse_window, dims=(0,))[
                    retime_valid
                ].reshape(-1)
                action_norm = float(torch.linalg.vector_norm(action_flat.double()).item())
                reverse_same = _cosine(action_flat, reverse_flat)
                reverse_retimed = _cosine(retime_action_flat, reversed_reverse)
                action_transition = torch.diff(action_window, dim=0)[
                    window_valid[1:] & window_valid[:-1]
                ]
                static_transition = torch.diff(static_window, dim=0)[
                    window_valid[1:] & window_valid[:-1]
                ]
                transition_denominator = max(
                    float(torch.linalg.vector_norm(action_transition.double()).item()),
                    _EPS,
                )
                static_ratio = float(
                    torch.linalg.vector_norm(static_transition.double()).item()
                ) / transition_denominator
                gates = {
                    "action_nonzero": action_norm >= self.config.action_norm_min,
                    "reverse_retimes_order": reverse_retimed
                    >= reverse_same + self.config.reverse_retime_margin,
                    "static_lacks_transition": static_ratio
                    <= self.config.static_transition_ratio_max,
                    "all_cells_observed": bool(window_valid.all().item()),
                }
                controls_pass = all(gates.values())
                all_controls_pass = all_controls_pass and controls_pass
                phase_rows = []
                for phase in range(PHASES):
                    in_window = first_phase <= phase <= last_phase
                    phase_rows.append(
                        {
                            "phase_index": phase,
                            "status": (
                                "not_applicable"
                                if not in_window
                                else (
                                    "observed"
                                    if bool(aggregate_valid[phase])
                                    else "uncertain"
                                )
                            ),
                            "relative_features": (
                                {
                                    name: float(action_value[phase, feature].item())
                                    for feature, name in enumerate(FEATURE_NAMES)
                                }
                                if in_window and bool(aggregate_valid[phase])
                                else None
                            ),
                        }
                    )
                per_edge[edge_identity] = {
                    **edge_receipt,
                    "edge_status": (
                        "observed" if bool(window_valid.all().item()) else "uncertain"
                    ),
                    "phase_rows": phase_rows,
                    "control_metrics": {
                        "action_minus_noop_norm": action_norm,
                        "reverse_same_order_cosine": reverse_same,
                        "reverse_retimed_cosine": reverse_retimed,
                        "static_transition_ratio": static_ratio,
                    },
                    "control_gates": gates,
                    "controls_passed": controls_pass,
                    "_action_tensor": action_value,
                    "_valid_tensor": aggregate_valid,
                }
            appearance_packets[appearance] = per_edge

        consensus_rows = []
        consensus_pass = True
        for left_index in range(len(appearances)):
            for right_index in range(left_index + 1, len(appearances)):
                left_id = appearances[left_index]
                right_id = appearances[right_index]
                for edge_spec in required_specs:
                    edge_identity = edge_spec.identity
                    left = appearance_packets[left_id][edge_identity]
                    right = appearance_packets[right_id][edge_identity]
                    first_phase = edge_spec.first_applicable_phase
                    last_phase = edge_spec.last_applicable_phase
                    valid = (
                        left["_valid_tensor"][first_phase:last_phase + 1]
                        & right["_valid_tensor"][first_phase:last_phase + 1]
                    )
                    left_window = left["_action_tensor"][
                        first_phase:last_phase + 1
                    ]
                    right_window = right["_action_tensor"][
                        first_phase:last_phase + 1
                    ]
                    left_flat = left_window[valid].reshape(-1)
                    right_flat = right_window[valid].reshape(-1)
                    cosine = _cosine(left_flat, right_flat)
                    distance = _normalized_distance(left_flat, right_flat)
                    passed = (
                        bool(valid.all().item())
                        and cosine >= self.config.appearance_cosine_min
                        and distance <= self.config.appearance_distance_max
                    )
                    consensus_pass = consensus_pass and passed
                    consensus_rows.append(
                        {
                            "left": left_id,
                            "right": right_id,
                            "source_role": edge_spec.source_role,
                            "target_role": edge_spec.target_role,
                            "relation_type": edge_spec.relation_type,
                            "first_applicable_phase": first_phase,
                            "last_applicable_phase": last_phase,
                            "applicability": "required",
                            "contributes_to_reward": True,
                            "cosine": cosine,
                            "normalized_distance": distance,
                            "passed": passed,
                        }
                    )

        public_appearances = []
        for appearance in appearances:
            rows = []
            for edge_spec in self.edge_specs:
                value = dict(appearance_packets[appearance][edge_spec.identity])
                value.pop("_action_tensor")
                value.pop("_valid_tensor")
                rows.append(value)
            public_appearances.append({"appearance_id": appearance, "edges": rows})
        registry = [item.receipt() for item in self.roles]
        latent_roles = [
            item.role_id for item in self.roles
            if item.evidence_mode == "latent_unobserved"
        ]
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "status": (
                "MECHANICALLY_ADMITTED"
                if all_controls_pass and consensus_pass and not any_uncertain
                else "REJECTED"
            ),
            "capture_matrix": {
                "appearance_ids": list(appearances),
                "arms": list(ARMS),
                "sigma_bands": list(SIGMA_BANDS),
                "blocks": list(BLOCKS),
                "phase_count": PHASES,
                "capture_count": len(self._cells),
                "zeroized_capture_count": self._zeroized,
                "same_state_within_appearance_sigma": True,
            },
            "node_registry": registry,
            "latent_or_offscreen_roles": {
                "role_ids": latent_roles,
                "visual_mask_claimed": False,
                "identity_claimed": False,
                "physical_contact_truth_claimed": False,
                "status": "unresolved" if latent_roles else "not_applicable",
            },
            "edge_registry": [item.receipt() for item in self.edge_specs],
            "edge_registry_summary": {
                "edge_count": len(self.edge_specs),
                "required_edge_count": len(required_specs),
                "not_applicable_edge_count": len(not_applicable_specs),
                "legacy_edge_registry": self.config.legacy_edge_registry,
                "not_applicable_edges_contribute_to_reward": False,
                "default_cartesian_product_used": False,
            },
            # Legacy pair-only view retained for receipt readers.  Admission is
            # driven exclusively by the typed edge registry above.
            "critical_edges": [list(item.pair) for item in required_specs],
            "appearance_packets": public_appearances,
            "multiappearance_consensus": consensus_rows,
            "summary": {
                "all_control_gates_passed": all_controls_pass,
                "all_critical_edges_consistent_across_appearances": consensus_pass,
                "any_critical_phase_uncertain": any_uncertain,
                "required_edge_count": len(required_specs),
                "not_applicable_edge_count": len(not_applicable_specs),
                "not_applicable_edges_contributed_to_reward": False,
                "mechanical_admission_passed": (
                    all_controls_pass and consensus_pass and not any_uncertain
                ),
            },
            "published_representation": {
                "visual_qk_role_slots": False,
                "raw_q": False,
                "raw_k": False,
                "raw_h": False,
                "raw_v": False,
                "dense_role_responsibilities": False,
                "absolute_coordinates": False,
                "absolute_anchor_geometry": False,
                "relative_role_pair_sequences_only": True,
            },
            "base_frozen_required": True,
            "optimizer_updates": 0,
            "renderer_output_modified": False,
            "target_inputs_consumed": False,
            "teacher_video_decoded": False,
            "generator_injection_authorized": False,
            "gpu_launch_authorized": False,
            "scientific_claim_authorized": False,
            "stable_transferable_action_representation_claimed": False,
        }
        result["representation_digest"] = object_sha256(result)
        _validate_public_result(result)
        return result


def _validate_public_result(value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value)
    lowered = encoded.lower()
    forbidden = (
        b'"raw_q":true',
        b'"raw_k":true',
        b'"raw_h":true',
        b'"raw_v":true',
        b'"absolute_coordinates":true',
        b'"dense_role_responsibilities":true',
        b'"target_inputs_consumed":true',
        b'"scientific_claim_authorized":true',
    )
    if any(item in lowered for item in forbidden):
        raise RelationalObserverError("public representation crossed a forbidden boundary")


__all__ = [
    "APPEARANCE_COUNT",
    "ARMS",
    "BLOCKS",
    "CaptureCell",
    "EDGE_APPLICABILITIES",
    "EdgeSpec",
    "EVIDENCE_MODES",
    "FEATURE_NAMES",
    "METHOD",
    "ObserverConfig",
    "OWNERSHIP",
    "PHASES",
    "RelationalObserverError",
    "RELATION_TYPES",
    "RoleSpec",
    "SCHEMA_VERSION",
    "SIGMA_BANDS",
    "StreamingRelationalObserver",
    "canonical_json_bytes",
    "object_sha256",
]
