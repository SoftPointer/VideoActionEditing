#!/usr/bin/env python3
"""Observer-only same-state partial object graph, v4.

The reducer consumes four prompt-conditioned observations made on the exact
same noisy state (action/noop/reverse/static).  Per-role spatial scores are
turned into a *partial* competitive assignment with an explicit dustbin and
only upper capacity constraints: no role is forced to own a patch.  A role is
published as observed only after spatial, competitor-margin, persistence and
multi-block/multi-sigma evidence gates pass.

Raw post-RoPE Q/K and dense role scores are consumed and zeroized at the
streaming cell boundary.  The retained ABI contains only generic role-pair
signed relation deltas, normalized relative geometry, edge validity,
change-points and uncertainty.  It contains no renderer, route, optimizer,
decoder, final anchor video, target-at-inference path, persistent source
identity registry, or contact FSM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Sequence

import torch


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    from . import self_generated_partial_object_graph_registry_v4 as registry_v4
except ImportError:  # pragma: no cover - direct AUH script import.
    import self_generated_partial_object_graph_registry_v4 as registry_v4


METHOD = registry_v4.METHOD
CAPTURE_SCHEMA = registry_v4.CAPTURE_SCHEMA
REDUCED_SCHEMA = registry_v4.REDUCED_SCHEMA
RESULT_SCHEMA = registry_v4.RESULT_SCHEMA
RECEIPT_SCHEMA = registry_v4.RECEIPT_SCHEMA
ARMS = registry_v4.ARMS
BLOCKS = registry_v4.BLOCKS
SIGMA_BANDS = registry_v4.SIGMA_BANDS
_EPS = 1.0e-8


class PartialObjectGraphObserverV4Error(ValueError):
    """A tensor, provenance, control, or admission invariant failed."""


def _tensor_digest(value: torch.Tensor) -> str:
    owned = value.detach().to(device="cpu").contiguous()
    header = registry_v4.canonical_json_bytes(
        {"shape": list(map(int, owned.shape)), "dtype": str(owned.dtype)}
    )
    try:
        raw = owned.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    except Exception as error:
        raise PartialObjectGraphObserverV4Error(
            "tensor digest cannot be materialized"
        ) from error
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _finite_tensor(value: Any, *, label: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise PartialObjectGraphObserverV4Error(
            f"{label} must be a rank-{ndim} tensor"
        )
    if value.device.type == "meta" or not bool(torch.isfinite(value).all().item()):
        raise PartialObjectGraphObserverV4Error(f"{label} must be finite")
    if not value.dtype.is_floating_point:
        raise PartialObjectGraphObserverV4Error(f"{label} must be floating point")
    return value


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = float(
        torch.linalg.vector_norm(a).item()
        * torch.linalg.vector_norm(b).item()
    )
    if denominator <= _EPS:
        return 0.0
    return float(torch.dot(a, b).item() / denominator)


def _normalized_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = max(
        float(torch.linalg.vector_norm(a).item()),
        float(torch.linalg.vector_norm(b).item()),
        _EPS,
    )
    return float(torch.linalg.vector_norm(a - b).item()) / denominator


def _transition_energy(value: torch.Tensor, valid: torch.Tensor) -> float:
    if value.shape[0] < 2:
        return 0.0
    pair_valid = valid[1:] & valid[:-1]
    if not bool(pair_valid.any().item()):
        return 0.0
    delta = value[1:] - value[:-1]
    selected = delta[pair_valid]
    return float(torch.sqrt(torch.mean(selected.double().square())).item())


def _isotropic_patch_coordinates(height: int, width: int) -> torch.Tensor:
    """Patch centers in one isotropic unit, safe on non-square grids."""

    scale = float(max(height, width))
    y = (torch.arange(height, dtype=torch.float32) + 0.5 - height / 2.0) / scale
    x = (torch.arange(width, dtype=torch.float32) + 0.5 - width / 2.0) / scale
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _persistent_runs(mask: torch.Tensor, minimum: int) -> torch.Tensor:
    """Keep only observed runs of at least ``minimum``; never fill gaps."""

    if mask.ndim != 2:
        raise PartialObjectGraphObserverV4Error("persistence mask geometry differs")
    phases, roles = map(int, mask.shape)
    result = torch.zeros_like(mask, dtype=torch.bool)
    for role in range(roles):
        first = 0
        while first < phases:
            while first < phases and not bool(mask[first, role].item()):
                first += 1
            last = first
            while last < phases and bool(mask[last, role].item()):
                last += 1
            if last - first >= minimum:
                result[first:last, role] = True
            first = max(last, first + 1)
    return result


@dataclass
class MiddleObservationV4:
    """One block observation for one prompt arm on an authenticated state.

    Stored tensor geometry is ``Q/K=[T,P,H,D]`` and role scores ``[T,R,P]``.
    ``create`` also accepts a leading singleton batch dimension.  The caller
    retains ownership until ``reduce_same_state_cell_v4`` consumes and
    zeroizes all three tensors.
    """

    schema_version: str
    appearance_id: str
    arm: str
    sigma_band: str
    block_index: int
    state_sha256: str
    timestep_sha256: str
    rotary_sha256: str
    prompt_sha256: str
    role_order: tuple[str, ...]
    role_partition_sha256: str
    role_token_counts: tuple[int, ...]
    patch_height: int
    patch_width: int
    queries: torch.Tensor = field(repr=False)
    keys: torch.Tensor = field(repr=False)
    role_scores: torch.Tensor = field(repr=False)
    tensor_bundle_digest: str
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    consumed: bool = field(default=False, init=False)

    @classmethod
    def create(
        cls,
        *,
        appearance_id: str,
        arm: str,
        sigma_band: str,
        block_index: int,
        state_sha256: str,
        timestep_sha256: str,
        rotary_sha256: str,
        prompt_sha256: str,
        role_order: Sequence[str],
        role_partition_sha256: str,
        role_token_counts: Sequence[int],
        patch_height: int,
        patch_width: int,
        queries: torch.Tensor,
        keys: torch.Tensor,
        role_scores: torch.Tensor,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MiddleObservationV4":
        q = queries
        k = keys
        scores = role_scores
        if isinstance(q, torch.Tensor) and q.ndim == 5 and q.shape[0] == 1:
            q = q[0]
        if isinstance(k, torch.Tensor) and k.ndim == 5 and k.shape[0] == 1:
            k = k[0]
        if isinstance(scores, torch.Tensor) and scores.ndim == 4 and scores.shape[0] == 1:
            scores = scores[0]
        tensor_bundle_digest = registry_v4.object_sha256(
            {
                "queries_sha256": _tensor_digest(q),
                "keys_sha256": _tensor_digest(k),
                "role_scores_sha256": _tensor_digest(scores),
            }
        )
        return cls(
            CAPTURE_SCHEMA,
            appearance_id,
            arm,
            sigma_band,
            block_index,
            state_sha256,
            timestep_sha256,
            rotary_sha256,
            prompt_sha256,
            tuple(role_order),
            role_partition_sha256,
            tuple(role_token_counts),
            patch_height,
            patch_width,
            q,
            k,
            scores,
            tensor_bundle_digest,
            dict(metadata or {}),
        )

    def validate(self, graph_registry: registry_v4.ObserverRegistryV4) -> None:
        if self.consumed:
            raise PartialObjectGraphObserverV4Error("capture was already consumed")
        if self.schema_version != CAPTURE_SCHEMA:
            raise PartialObjectGraphObserverV4Error("capture schema differs")
        registry_v4.require_identifier(self.appearance_id, label="appearance_id")
        if self.arm not in ARMS:
            raise PartialObjectGraphObserverV4Error("capture arm differs")
        if self.sigma_band not in SIGMA_BANDS:
            raise PartialObjectGraphObserverV4Error("capture sigma band differs")
        if self.block_index not in BLOCKS:
            raise PartialObjectGraphObserverV4Error("capture block differs")
        for label, value in (
            ("state", self.state_sha256),
            ("timestep", self.timestep_sha256),
            ("rotary", self.rotary_sha256),
            ("prompt", self.prompt_sha256),
            ("role partition", self.role_partition_sha256),
        ):
            registry_v4.require_sha256(value, label=label)
        if self.role_order != graph_registry.role_ids:
            raise PartialObjectGraphObserverV4Error(
                "capture role order differs from graph registry"
            )
        if (
            len(self.role_token_counts) != len(self.role_order)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in self.role_token_counts
            )
            or sum(self.role_token_counts) != registry_v4.TEXT_TOKEN_COUNT
        ):
            raise PartialObjectGraphObserverV4Error(
                "capture token-prior counts differ"
            )
        if (
            not isinstance(self.patch_height, int)
            or isinstance(self.patch_height, bool)
            or not isinstance(self.patch_width, int)
            or isinstance(self.patch_width, bool)
            or self.patch_height < 1
            or self.patch_width < 1
        ):
            raise PartialObjectGraphObserverV4Error("patch geometry differs")
        q = _finite_tensor(self.queries, label="queries", ndim=4)
        k = _finite_tensor(self.keys, label="keys", ndim=4)
        scores = _finite_tensor(self.role_scores, label="role scores", ndim=3)
        patches = self.patch_height * self.patch_width
        if q.shape != k.shape or tuple(q.shape[:2]) != (
            graph_registry.phases,
            patches,
        ):
            raise PartialObjectGraphObserverV4Error("Q/K geometry differs")
        if q.shape[2] < 1 or q.shape[3] < 1:
            raise PartialObjectGraphObserverV4Error("Q/K head geometry differs")
        if tuple(scores.shape) != (
            graph_registry.phases,
            len(graph_registry.roles),
            patches,
        ):
            raise PartialObjectGraphObserverV4Error("role score geometry differs")
        if bool((scores < 0).any().item()):
            raise PartialObjectGraphObserverV4Error(
                "native role proxy must be nonnegative"
            )
        simplex_mass = scores.float().sum(dim=1)
        if not bool(
            torch.allclose(
                simplex_mass,
                torch.ones_like(simplex_mass),
                atol=float(graph_registry.thresholds["simplex_atol"]),
                rtol=float(graph_registry.thresholds["simplex_rtol"]),
            )
        ):
            raise PartialObjectGraphObserverV4Error(
                "native role proxy must sum to one over registered roles"
            )
        if q.device != k.device or q.device != scores.device:
            raise PartialObjectGraphObserverV4Error("capture tensors differ by device")
        registry_v4.require_sha256(
            self.tensor_bundle_digest, label="capture tensor bundle"
        )
        live_digest = registry_v4.object_sha256(
            {
                "queries_sha256": _tensor_digest(q),
                "keys_sha256": _tensor_digest(k),
                "role_scores_sha256": _tensor_digest(scores),
            }
        )
        if live_digest != self.tensor_bundle_digest:
            raise PartialObjectGraphObserverV4Error(
                "capture tensor content changed after registration"
            )
        registry_v4.assert_no_target_payload(self.metadata)

    def zeroize(self) -> None:
        with torch.inference_mode():
            self.queries.zero_()
            self.keys.zero_()
            self.role_scores.zero_()
        if any(
            int(torch.count_nonzero(value).item()) != 0
            for value in (self.queries, self.keys, self.role_scores)
        ):
            raise PartialObjectGraphObserverV4Error("raw capture did not zeroize")
        self.consumed = True


@dataclass
class ReducedArmObservationV4:
    """Compact result for one arm; safe to retain while later arms run."""

    schema_version: str
    appearance_id: str
    arm: str
    sigma_band: str
    block_index: int
    state_sha256: str
    timestep_sha256: str
    rotary_sha256: str
    prompt_sha256: str
    role_partition_sha256: str
    role_token_counts: tuple[int, ...]
    role_valid: torch.Tensor = field(repr=False)  # [T,R]
    role_confidence: torch.Tensor = field(repr=False)  # [T,R]
    edge_relation: torch.Tensor = field(repr=False)  # [T,E]
    edge_geometry: torch.Tensor = field(repr=False)  # [T,E,4]
    edge_valid: torch.Tensor = field(repr=False)  # [T,E]
    role_centroids: torch.Tensor = field(repr=False)  # ephemeral [T,R,2]
    absolute_eligible_mask: torch.Tensor = field(repr=False)  # ephemeral [T,R,P]
    absolute_margins: torch.Tensor = field(repr=False)  # [T,R]
    role_prior: torch.Tensor = field(repr=False)  # [R]
    support_frame_valid: bool
    raw_input_zeroized: bool
    digest: str
    zeroized: bool = field(default=False, init=False)

    def validate(self, graph_registry: registry_v4.ObserverRegistryV4) -> None:
        if self.schema_version != REDUCED_SCHEMA or self.zeroized:
            raise PartialObjectGraphObserverV4Error("reduced arm is unavailable")
        if self.arm not in ARMS or self.sigma_band not in SIGMA_BANDS:
            raise PartialObjectGraphObserverV4Error("reduced arm identity differs")
        if self.block_index not in BLOCKS:
            raise PartialObjectGraphObserverV4Error("reduced arm block differs")
        registry_v4.require_sha256(
            self.role_partition_sha256, label="reduced arm role partition"
        )
        if (
            len(self.role_token_counts) != len(graph_registry.roles)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in self.role_token_counts
            )
            or sum(self.role_token_counts) != registry_v4.TEXT_TOKEN_COUNT
        ):
            raise PartialObjectGraphObserverV4Error(
                "reduced arm token counts differ"
            )
        phases = graph_registry.phases
        roles = len(graph_registry.roles)
        edges = len(graph_registry.edges)
        expected = (
            ("role_valid", self.role_valid, (phases, roles)),
            ("role_confidence", self.role_confidence, (phases, roles)),
            ("edge_relation", self.edge_relation, (phases, edges)),
            ("edge_geometry", self.edge_geometry, (phases, edges, 4)),
            ("edge_valid", self.edge_valid, (phases, edges)),
            ("role_centroids", self.role_centroids, (phases, roles, 2)),
            ("absolute_margins", self.absolute_margins, (phases, roles)),
            ("role_prior", self.role_prior, (roles,)),
        )
        for label, value, shape in expected:
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise PartialObjectGraphObserverV4Error(
                    f"reduced arm {label} geometry differs"
                )
            if value.dtype != torch.bool and not bool(torch.isfinite(value).all().item()):
                raise PartialObjectGraphObserverV4Error(
                    f"reduced arm {label} is non-finite"
                )
        if (
            not isinstance(self.absolute_eligible_mask, torch.Tensor)
            or self.absolute_eligible_mask.dtype != torch.bool
            or tuple(self.absolute_eligible_mask.shape[:2]) != (phases, roles)
            or self.absolute_eligible_mask.ndim != 3
        ):
            raise PartialObjectGraphObserverV4Error(
                "reduced arm absolute eligibility geometry differs"
            )
        if not self.raw_input_zeroized:
            raise PartialObjectGraphObserverV4Error("one-arm raw input remains resident")
        payload = {
            "schema_version": REDUCED_SCHEMA,
            "appearance_id": self.appearance_id,
            "arm": self.arm,
            "sigma_band": self.sigma_band,
            "block_index": self.block_index,
            "state_sha256": self.state_sha256,
            "timestep_sha256": self.timestep_sha256,
            "rotary_sha256": self.rotary_sha256,
            "prompt_sha256": self.prompt_sha256,
            "role_partition_sha256": self.role_partition_sha256,
            "role_token_counts": list(self.role_token_counts),
            "registry_digest": graph_registry.digest,
            "role_valid_sha256": _tensor_digest(self.role_valid),
            "role_confidence_sha256": _tensor_digest(self.role_confidence),
            "edge_relation_sha256": _tensor_digest(self.edge_relation),
            "edge_geometry_sha256": _tensor_digest(self.edge_geometry),
            "edge_valid_sha256": _tensor_digest(self.edge_valid.to(torch.uint8)),
            "role_centroids_sha256": _tensor_digest(self.role_centroids),
            "absolute_eligible_mask_sha256": _tensor_digest(
                self.absolute_eligible_mask.to(torch.uint8)
            ),
            "absolute_margins_sha256": _tensor_digest(self.absolute_margins),
            "role_prior": self.role_prior.tolist(),
            "support_frame_valid": self.support_frame_valid,
            "raw_input_zeroized": self.raw_input_zeroized,
        }
        if registry_v4.object_sha256(payload) != self.digest:
            raise PartialObjectGraphObserverV4Error("reduced arm digest differs")

    def zeroize(self) -> None:
        with torch.inference_mode():
            for value in (
                self.role_valid,
                self.role_confidence,
                self.edge_relation,
                self.edge_geometry,
                self.edge_valid,
                self.role_centroids,
                self.absolute_eligible_mask,
                self.absolute_margins,
                self.role_prior,
            ):
                value.zero_()
        self.zeroized = True


@dataclass
class ReducedSameStateCellV4:
    schema_version: str
    appearance_id: str
    sigma_band: str
    block_index: int
    role_valid: torch.Tensor = field(repr=False)  # [A,T,R]
    role_confidence: torch.Tensor = field(repr=False)  # [A,T,R]
    edge_relation: torch.Tensor = field(repr=False)  # [A,T,E]
    edge_geometry: torch.Tensor = field(repr=False)  # [A,T,E,4]
    edge_valid: torch.Tensor = field(repr=False)  # [A,T,E]
    support_frame_valid: torch.Tensor = field(repr=False)  # [A]
    shared_frame_phase_valid: torch.Tensor = field(repr=False)  # [T]
    shared_frame_receipt: Mapping[str, Any]
    simplex_evidence_receipt: Mapping[str, Any]
    common_four_arm_edge_mask_digest: str
    state_sha256: str
    raw_inputs_zeroized: bool
    digest: str
    zeroized: bool = field(default=False, init=False)

    def validate(self, graph_registry: registry_v4.ObserverRegistryV4) -> None:
        if self.schema_version != REDUCED_SCHEMA or self.zeroized:
            raise PartialObjectGraphObserverV4Error("reduced cell is unavailable")
        arms = len(ARMS)
        phases = graph_registry.phases
        roles = len(graph_registry.roles)
        edges = len(graph_registry.edges)
        expected = (
            ("role_valid", self.role_valid, (arms, phases, roles)),
            ("role_confidence", self.role_confidence, (arms, phases, roles)),
            ("edge_relation", self.edge_relation, (arms, phases, edges)),
            ("edge_geometry", self.edge_geometry, (arms, phases, edges, 4)),
            ("edge_valid", self.edge_valid, (arms, phases, edges)),
            ("support_frame_valid", self.support_frame_valid, (arms,)),
            (
                "shared_frame_phase_valid",
                self.shared_frame_phase_valid,
                (phases,),
            ),
        )
        for label, value, shape in expected:
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise PartialObjectGraphObserverV4Error(
                    f"reduced {label} geometry differs"
                )
            if value.dtype != torch.bool and not bool(torch.isfinite(value).all().item()):
                raise PartialObjectGraphObserverV4Error(
                    f"reduced {label} is non-finite"
                )
        if not self.raw_inputs_zeroized:
            raise PartialObjectGraphObserverV4Error("raw inputs remain resident")
        registry_v4.require_sha256(
            self.common_four_arm_edge_mask_digest,
            label="four-arm common edge mask",
        )
        if (
            self.shared_frame_receipt.get("frame_sources") != ["noop", "static"]
            or self.shared_frame_receipt.get("action_or_reverse_defined_frame")
            is not False
            or self.simplex_evidence_receipt.get("native_simplex_validated")
            is not True
        ):
            raise PartialObjectGraphObserverV4Error(
                "shared-frame/simplex provenance differs"
            )
        digest_payload = {
            "schema_version": REDUCED_SCHEMA,
            "appearance_id": self.appearance_id,
            "sigma_band": self.sigma_band,
            "block_index": self.block_index,
            "state_sha256": self.state_sha256,
            "registry_digest": graph_registry.digest,
            "role_valid_sha256": _tensor_digest(self.role_valid),
            "role_confidence_sha256": _tensor_digest(self.role_confidence),
            "edge_relation_sha256": _tensor_digest(self.edge_relation),
            "edge_geometry_sha256": _tensor_digest(self.edge_geometry),
            "edge_valid_sha256": _tensor_digest(self.edge_valid.to(torch.uint8)),
            "support_frame_valid": self.support_frame_valid.tolist(),
            "shared_frame_phase_valid_sha256": _tensor_digest(
                self.shared_frame_phase_valid.to(torch.uint8)
            ),
            "shared_frame_receipt": dict(self.shared_frame_receipt),
            "simplex_evidence_receipt": dict(self.simplex_evidence_receipt),
            "common_four_arm_edge_mask_digest": (
                self.common_four_arm_edge_mask_digest
            ),
            "raw_inputs_zeroized": self.raw_inputs_zeroized,
        }
        if registry_v4.object_sha256(digest_payload) != self.digest:
            raise PartialObjectGraphObserverV4Error(
                "reduced cell content changed after sealing"
            )

    def zeroize(self) -> None:
        with torch.inference_mode():
            for value in (
                self.role_valid,
                self.role_confidence,
                self.edge_relation,
                self.edge_geometry,
                self.edge_valid,
                self.support_frame_valid,
                self.shared_frame_phase_valid,
            ):
                value.zero_()
        self.zeroized = True


def _partial_assign(
    raw_scores: torch.Tensor,
    roles: Sequence[registry_v4.RoleSpecV4],
    thresholds: Mapping[str, float],
    role_token_counts: Sequence[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Unbalanced role/patch competition with an unrestricted dustbin.

    Only an upper capacity is applied to visual roles.  There is deliberately
    no row lower bound, so unsupported instruction roles receive zero rather
    than a fabricated slot.
    """

    phases, role_count, patches = map(int, raw_scores.shape)
    scores = raw_scores.float()
    visual = torch.tensor(
        [role.can_be_observed for role in roles], dtype=torch.bool
    )
    counts = torch.tensor(role_token_counts, dtype=torch.float32)
    prior = counts / counts.sum().clamp_min(_EPS)
    enrichment_ratio = scores / prior.reshape(1, role_count, 1).clamp_min(_EPS)
    prior_equalized = enrichment_ratio / enrichment_ratio.sum(
        dim=1, keepdim=True
    ).clamp_min(_EPS)
    log_enrichment = enrichment_ratio.clamp_min(_EPS).log()

    # Absolute evidence is evaluated on the native simplex before any
    # per-role normalization.  A role gets no kernel entry unless it beats
    # its token prior, a fixed probability floor, every competing role and
    # the implicit dustbin.  This prevents z-score/top-k from manufacturing a
    # slot out of flat or arbitrarily weak noise.
    absolute_margin = torch.zeros_like(scores)
    eligible = torch.zeros_like(scores, dtype=torch.bool)
    for role_index in range(role_count):
        if not bool(visual[role_index].item()):
            continue
        competitors = [index for index in range(role_count) if index != role_index]
        competitor = prior_equalized[:, competitors].max(dim=1).values
        absolute_margin[:, role_index] = (
            prior_equalized[:, role_index] - competitor
        )
        eligible[:, role_index] = (
            (scores[:, role_index] >= float(thresholds["absolute_role_probability_min"]))
            & (
                prior_equalized[:, role_index]
                >= float(thresholds["absolute_prior_equalized_probability_min"])
            )
            & (
                log_enrichment[:, role_index]
                >= float(thresholds["absolute_log_prior_enrichment_min"])
            )
            & (
                absolute_margin[:, role_index]
                >= float(thresholds["absolute_role_competitor_margin_min"])
            )
        )

    # Identical role fields are ambiguous, even if both are spatially sharp.
    # Reject the whole affected phase instead of assigning duplicate objects.
    duplicate_phase = torch.zeros(phases, dtype=torch.bool)
    visual_indices = torch.nonzero(visual, as_tuple=False).flatten().tolist()
    for left_index, left in enumerate(visual_indices):
        for right in visual_indices[left_index + 1 :]:
            duplicate_phase |= (
                (scores[:, left] - scores[:, right]).abs().amax(dim=-1)
                <= float(thresholds["duplicate_role_distribution_max_abs_max"])
            )
    eligible[duplicate_phase] = False

    mean = log_enrichment.mean(dim=-1, keepdim=True)
    std = log_enrichment.std(dim=-1, unbiased=False, keepdim=True)
    standardized = torch.where(
        std > _EPS,
        (log_enrichment - mean) / std.clamp_min(_EPS),
        torch.zeros_like(log_enrichment),
    )
    if bool(visual.any().item()):
        common = standardized[:, visual].mean(dim=1, keepdim=True)
    else:  # Registry validation normally prevents a useful all-latent graph.
        common = torch.zeros((phases, 1, patches), dtype=torch.float32)
    contrast = standardized - common

    top_count = max(
        1,
        min(
            patches,
            int(math.ceil(patches * float(thresholds["topk_fraction"]))),
        ),
    )
    top_indices = torch.topk(contrast, top_count, dim=-1).indices
    keep = torch.zeros_like(contrast, dtype=torch.bool)
    keep.scatter_(-1, top_indices, True)
    keep &= eligible
    logits = contrast / float(thresholds["assignment_temperature"])
    role_kernel = torch.where(
        keep,
        torch.exp(logits.clamp(-40.0, 40.0)),
        torch.zeros_like(logits),
    )
    dustbin = torch.full(
        (phases, 1, patches),
        math.exp(float(thresholds["dustbin_logit"])),
        dtype=torch.float32,
    )
    capacity = patches * float(thresholds["role_capacity_fraction"])
    for _ in range(int(thresholds["partial_assignment_iterations"])):
        denominator = (role_kernel.sum(dim=1, keepdim=True) + dustbin).clamp_min(_EPS)
        probabilities = role_kernel / denominator
        mass = probabilities.sum(dim=-1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(mass),
            torch.full_like(mass, capacity) / mass.clamp_min(_EPS),
        )
        role_kernel = role_kernel * scale
    assignment = role_kernel / (
        role_kernel.sum(dim=1, keepdim=True) + dustbin
    ).clamp_min(_EPS)

    mass_fraction = assignment.sum(dim=-1) / float(patches)
    peak = assignment.max(dim=-1).values
    raw_spatial = torch.softmax(
        contrast / float(thresholds["assignment_temperature"]), dim=-1
    )
    entropy = -(
        raw_spatial.clamp_min(_EPS) * raw_spatial.clamp_min(_EPS).log()
    ).sum(dim=-1) / max(math.log(float(patches)), 1.0)
    concentration = (1.0 - entropy).clamp(0.0, 1.0)

    margins = torch.zeros((phases, role_count), dtype=torch.float32)
    for role_index in range(role_count):
        if not bool(visual[role_index].item()):
            continue
        spatial = assignment[:, role_index]
        normalized = spatial / spatial.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        margins[:, role_index] = (
            normalized * absolute_margin[:, role_index]
        ).sum(dim=-1)

    observed = (
        (mass_fraction >= float(thresholds["role_mass_fraction_min"]))
        & (peak >= float(thresholds["role_peak_probability_min"]))
        & (margins >= float(thresholds["role_competitor_margin_min"]))
        & (concentration >= float(thresholds["role_concentration_min"]))
        & visual.reshape(1, role_count)
    )
    ratios = torch.stack(
        (
            mass_fraction / float(thresholds["role_mass_fraction_min"]),
            peak / float(thresholds["role_peak_probability_min"]),
            margins.clamp_min(0.0) / float(thresholds["role_competitor_margin_min"]),
            concentration / float(thresholds["role_concentration_min"]),
        ),
        dim=-1,
    )
    confidence = ratios.min(dim=-1).values.clamp(0.0, 1.0)
    confidence = torch.where(observed, confidence, torch.zeros_like(confidence))
    return (
        assignment.contiguous(),
        observed.contiguous(),
        confidence.contiguous(),
        margins.contiguous(),
        eligible.contiguous(),
    )


def _robust_support_frame(
    centroids: torch.Tensor,
    role_valid: torch.Tensor,
    support_indices: tuple[int, int] | None,
    thresholds: Mapping[str, float],
) -> tuple[bool, torch.Tensor, float, torch.Tensor]:
    phases = int(centroids.shape[0])
    if support_indices is None:
        return False, torch.tensor((1.0, 0.0)), 1.0, torch.zeros(phases, dtype=torch.bool)
    start_index, end_index = support_indices
    eligible = role_valid[:, start_index] & role_valid[:, end_index]
    vectors = centroids[:, end_index] - centroids[:, start_index]
    scales = torch.linalg.vector_norm(vectors, dim=-1)
    eligible &= scales >= float(thresholds["support_frame_scale_min"])
    if int(eligible.sum().item()) < int(thresholds["support_frame_phases_min"]):
        return False, torch.tensor((1.0, 0.0)), 1.0, eligible
    selected = vectors[eligible]
    estimate = selected.median(dim=0).values
    delta = float(thresholds["support_frame_huber_delta"])
    for _ in range(int(thresholds["support_frame_irls_iterations"])):
        residual = torch.linalg.vector_norm(selected - estimate, dim=-1)
        weights = torch.where(
            residual <= delta,
            torch.ones_like(residual),
            torch.full_like(residual, delta) / residual.clamp_min(_EPS),
        )
        estimate = (selected * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(_EPS)
    frame_scale = float(torch.linalg.vector_norm(estimate).item())
    if frame_scale < float(thresholds["support_frame_scale_min"]):
        return False, torch.tensor((1.0, 0.0)), 1.0, eligible
    direction = estimate / frame_scale
    selected_unit = selected / torch.linalg.vector_norm(selected, dim=-1, keepdim=True).clamp_min(_EPS)
    cosine = (selected_unit * direction).sum(dim=-1)
    inliers = cosine >= float(thresholds["support_frame_cosine_min"])
    inlier_fraction = float(inliers.float().mean().item())
    if (
        int(inliers.sum().item()) < int(thresholds["support_frame_phases_min"])
        or inlier_fraction < float(thresholds["support_frame_inlier_fraction_min"])
    ):
        return False, torch.tensor((1.0, 0.0)), 1.0, eligible
    inlier_scales = torch.linalg.vector_norm(selected[inliers], dim=-1)
    robust_scale = float(inlier_scales.median().item())
    phase_inliers = torch.zeros(phases, dtype=torch.bool)
    eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
    phase_inliers[eligible_indices[inliers]] = True
    return True, direction.contiguous(), robust_scale, phase_inliers


def _pool_role_qk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    assignment: torch.Tensor,
    roles: Sequence[registry_v4.RoleSpecV4],
) -> torch.Tensor:
    mass = assignment.sum(dim=-1).clamp_min(_EPS)
    q_role = torch.einsum("trp,tphd->trhd", assignment, queries.float()) / mass[..., None, None]
    k_role = torch.einsum("trp,tphd->trhd", assignment, keys.float()) / mass[..., None, None]
    visual = torch.tensor([role.can_be_observed for role in roles], dtype=torch.bool)
    if bool(visual.any().item()):
        q_common = q_role[:, visual].mean(dim=1, keepdim=True)
        k_common = k_role[:, visual].mean(dim=1, keepdim=True)
        q_role = q_role - q_common
        k_role = k_role - k_common
    scale = math.sqrt(float(queries.shape[2] * queries.shape[3]))
    return torch.einsum("trhd,tshd->trs", q_role, k_role) / max(scale, _EPS)


def _reduce_arm(
    capture: MiddleObservationV4,
    graph_registry: registry_v4.ObserverRegistryV4,
) -> tuple[Any, ...]:
    scores = capture.role_scores.detach().to(device="cpu", dtype=torch.float32).contiguous()
    queries = capture.queries.detach().to(device="cpu", dtype=torch.float32).contiguous()
    keys = capture.keys.detach().to(device="cpu", dtype=torch.float32).contiguous()
    assignment, observed, confidence, margins, absolute_eligible = _partial_assign(
        scores,
        graph_registry.roles,
        graph_registry.thresholds,
        capture.role_token_counts,
    )
    coordinates = _isotropic_patch_coordinates(
        capture.patch_height, capture.patch_width
    )
    mass = assignment.sum(dim=-1)
    normalized = assignment / mass.clamp_min(_EPS).unsqueeze(-1)
    centroids = torch.einsum("trp,pd->trd", normalized, coordinates)

    persistent = _persistent_runs(
        observed,
        int(graph_registry.thresholds["persistent_run_phases_min"]),
    )
    for phase in range(1, graph_registry.phases):
        adjacent = persistent[phase] & persistent[phase - 1]
        jumps = torch.linalg.vector_norm(
            centroids[phase] - centroids[phase - 1], dim=-1
        )
        persistent[phase] &= (~adjacent) | (
            jumps <= float(graph_registry.thresholds["centroid_jump_max"])
        )
    persistent = _persistent_runs(
        persistent,
        int(graph_registry.thresholds["persistent_run_phases_min"]),
    )
    confidence = torch.where(persistent, confidence, torch.zeros_like(confidence))

    relation_matrix = _pool_role_qk(
        queries, keys, assignment, graph_registry.roles
    )
    role_index = {role.role_id: index for index, role in enumerate(graph_registry.roles)}
    edge_count = len(graph_registry.edges)
    edge_relation = torch.zeros((graph_registry.phases, edge_count), dtype=torch.float32)
    edge_geometry = torch.zeros((graph_registry.phases, edge_count, 4), dtype=torch.float32)
    edge_valid = torch.zeros((graph_registry.phases, edge_count), dtype=torch.bool)
    for edge_index, edge in enumerate(graph_registry.edges):
        source = role_index[edge.source_role]
        destination = role_index[edge.target_role]
        endpoints = persistent[:, source] & persistent[:, destination]
        if (
            not graph_registry.roles[source].can_be_observed
            or not graph_registry.roles[destination].can_be_observed
        ):
            endpoints.zero_()
        edge_valid[:, edge_index] = endpoints
        edge_relation[:, edge_index] = relation_matrix[:, source, destination]
        edge_relation[:, edge_index] = torch.where(
            endpoints, edge_relation[:, edge_index], torch.zeros_like(edge_relation[:, edge_index])
        )

    # Absolute centroids and eligibility are retained only until all compact
    # arms close the same-state cell.  The assembler builds one observer-
    # independent noop/static frame and then zeroizes these temporaries.
    centroid_evidence = centroids.detach().contiguous().clone()
    eligible_evidence = absolute_eligible.detach().contiguous().clone()
    margin_evidence = margins.detach().contiguous().clone()
    prior_evidence = (
        torch.tensor(capture.role_token_counts, dtype=torch.float32)
        / float(sum(capture.role_token_counts))
    ).contiguous()
    support_endpoint_evidence = False
    if graph_registry.support_indices is not None:
        start, end = graph_registry.support_indices
        support_endpoint_evidence = bool(
            (persistent[:, start] & persistent[:, end]).any().item()
        )

    # Clear every dense/absolute CPU temporary before the reduced values leave.
    temporaries = (
        scores,
        queries,
        keys,
        assignment,
        observed,
        margins,
        coordinates,
        mass,
        normalized,
        centroids,
        relation_matrix,
        absolute_eligible,
        margins,
    )
    with torch.inference_mode():
        for value in temporaries:
            value.zero_()
    return (
        persistent.contiguous(),
        confidence.contiguous(),
        edge_relation.contiguous(),
        edge_geometry.contiguous(),
        edge_valid.contiguous(),
        centroid_evidence,
        eligible_evidence,
        margin_evidence,
        prior_evidence,
        support_endpoint_evidence,
    )


def reduce_one_arm_v4(
    capture: MiddleObservationV4,
    *,
    graph_registry: registry_v4.ObserverRegistryV4,
) -> ReducedArmObservationV4:
    """Consume one raw arm immediately and return only compact evidence."""

    capture.validate(graph_registry)
    row: tuple[Any, ...] | None = None
    succeeded = False
    try:
        row = _reduce_arm(capture, graph_registry)
        capture.zeroize()
        payload = {
            "schema_version": REDUCED_SCHEMA,
            "appearance_id": capture.appearance_id,
            "arm": capture.arm,
            "sigma_band": capture.sigma_band,
            "block_index": capture.block_index,
            "state_sha256": capture.state_sha256,
            "timestep_sha256": capture.timestep_sha256,
            "rotary_sha256": capture.rotary_sha256,
            "prompt_sha256": capture.prompt_sha256,
            "role_partition_sha256": capture.role_partition_sha256,
            "role_token_counts": list(capture.role_token_counts),
            "registry_digest": graph_registry.digest,
            "role_valid_sha256": _tensor_digest(row[0]),
            "role_confidence_sha256": _tensor_digest(row[1]),
            "edge_relation_sha256": _tensor_digest(row[2]),
            "edge_geometry_sha256": _tensor_digest(row[3]),
            "edge_valid_sha256": _tensor_digest(row[4].to(torch.uint8)),
            "role_centroids_sha256": _tensor_digest(row[5]),
            "absolute_eligible_mask_sha256": _tensor_digest(
                row[6].to(torch.uint8)
            ),
            "absolute_margins_sha256": _tensor_digest(row[7]),
            "role_prior": row[8].tolist(),
            "support_frame_valid": row[9],
            "raw_input_zeroized": capture.consumed,
        }
        result = ReducedArmObservationV4(
            REDUCED_SCHEMA,
            capture.appearance_id,
            capture.arm,
            capture.sigma_band,
            capture.block_index,
            capture.state_sha256,
            capture.timestep_sha256,
            capture.rotary_sha256,
            capture.prompt_sha256,
            capture.role_partition_sha256,
            capture.role_token_counts,
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            capture.consumed,
            registry_v4.object_sha256(payload),
        )
        succeeded = True
        return result
    finally:
        if not capture.consumed:
            capture.zeroize()
        if not succeeded and row is not None:
            with torch.inference_mode():
                for value in row[:9]:
                    if isinstance(value, torch.Tensor):
                        value.zero_()


class SameStateCellAssemblerV4:
    """Collect compact arms without retaining four raw captures."""

    def __init__(self, graph_registry: registry_v4.ObserverRegistryV4) -> None:
        self.graph_registry = graph_registry
        self._arms: dict[str, ReducedArmObservationV4] = {}

    def add(self, reduced_arm: ReducedArmObservationV4) -> None:
        try:
            reduced_arm.validate(self.graph_registry)
            if reduced_arm.arm in self._arms:
                raise PartialObjectGraphObserverV4Error("duplicate compact arm")
            if self._arms:
                first = next(iter(self._arms.values()))
                for label in (
                    "appearance_id",
                    "sigma_band",
                    "block_index",
                    "state_sha256",
                    "timestep_sha256",
                    "rotary_sha256",
                ):
                    if getattr(first, label) != getattr(reduced_arm, label):
                        raise PartialObjectGraphObserverV4Error(
                            f"same-state compact arm {label} differs"
                        )
                if reduced_arm.prompt_sha256 in {
                    item.prompt_sha256 for item in self._arms.values()
                }:
                    raise PartialObjectGraphObserverV4Error(
                        "compact prompt arm is not independently authenticated"
                    )
            self._arms[reduced_arm.arm] = reduced_arm
        except BaseException:
            if not reduced_arm.zeroized:
                reduced_arm.zeroize()
            self.abort()
            raise

    def abort(self) -> None:
        """Destroy every compact arm still owned by this cell."""

        rows = tuple(self._arms.values())
        self._arms.clear()
        for row in rows:
            if not row.zeroized:
                row.zeroize()
            if any(
                int(torch.count_nonzero(value).item()) != 0
                for value in (
                    row.role_valid,
                    row.role_confidence,
                    row.edge_relation,
                    row.edge_geometry,
                    row.edge_valid,
                    row.role_centroids,
                    row.absolute_eligible_mask,
                    row.absolute_margins,
                    row.role_prior,
                )
            ):
                raise PartialObjectGraphObserverV4Error(
                    "compact arm failure scrub did not close"
                )

    def finalize(self) -> ReducedSameStateCellV4:
        try:
            return self._finalize_owned()
        finally:
            self.abort()

    def _finalize_owned(self) -> ReducedSameStateCellV4:
        if set(self._arms) != set(ARMS):
            raise PartialObjectGraphObserverV4Error(
                "compact assembler requires action/noop/reverse/static"
            )
        rows = [self._arms[arm] for arm in ARMS]
        for row in rows:
            row.validate(self.graph_registry)
        first = rows[0]
        role_valid = torch.stack([row.role_valid for row in rows], dim=0)
        role_confidence = torch.stack([row.role_confidence for row in rows], dim=0)
        edge_relation = torch.stack([row.edge_relation for row in rows], dim=0)
        centroids = torch.stack([row.role_centroids for row in rows], dim=0)

        phases = self.graph_registry.phases
        edges = len(self.graph_registry.edges)
        if self.graph_registry.support_indices is None:
            raise PartialObjectGraphObserverV4Error(
                "shared-frame cell lacks registered support endpoints"
            )
        start, end = self.graph_registry.support_indices
        noop_index, static_index = ARMS.index("noop"), ARMS.index("static")
        noop_start = centroids[noop_index, :, start]
        noop_end = centroids[noop_index, :, end]
        static_start = centroids[static_index, :, start]
        static_end = centroids[static_index, :, end]
        noop_vector = noop_end - noop_start
        static_vector = static_end - static_start
        noop_scale = torch.linalg.vector_norm(noop_vector, dim=-1)
        static_scale = torch.linalg.vector_norm(static_vector, dim=-1)
        endpoint_rms = torch.sqrt(
            (
                (noop_start - static_start).square().sum(dim=-1)
                + (noop_end - static_end).square().sum(dim=-1)
            )
            / 4.0
        )
        noop_unit = noop_vector / noop_scale.unsqueeze(-1).clamp_min(_EPS)
        static_unit = static_vector / static_scale.unsqueeze(-1).clamp_min(_EPS)
        direction_cosine = (noop_unit * static_unit).sum(dim=-1)
        log_scale_error = (
            noop_scale.clamp_min(_EPS).log()
            - static_scale.clamp_min(_EPS).log()
        ).abs()
        support_evidence = (
            role_valid[noop_index, :, start]
            & role_valid[noop_index, :, end]
            & role_valid[static_index, :, start]
            & role_valid[static_index, :, end]
        )
        shared_frame_phase_valid = (
            support_evidence
            & (noop_scale >= float(self.graph_registry.thresholds["support_frame_scale_min"]))
            & (static_scale >= float(self.graph_registry.thresholds["support_frame_scale_min"]))
            & (
                endpoint_rms
                <= float(self.graph_registry.thresholds["shared_frame_endpoint_rms_max"])
            )
            & (
                direction_cosine
                >= float(self.graph_registry.thresholds["shared_frame_direction_cosine_min"])
            )
            & (
                log_scale_error
                <= float(self.graph_registry.thresholds["shared_frame_log_scale_abs_max"])
            )
        )
        phase_fraction = float(shared_frame_phase_valid.float().mean().item())
        frame_admitted = phase_fraction >= float(
            self.graph_registry.thresholds["shared_frame_phase_fraction_min"]
        )
        if not frame_admitted:
            shared_frame_phase_valid.zero_()
        summed_direction = noop_unit + static_unit
        shared_direction = summed_direction / torch.linalg.vector_norm(
            summed_direction, dim=-1, keepdim=True
        ).clamp_min(_EPS)
        shared_perpendicular = torch.stack(
            (-shared_direction[:, 1], shared_direction[:, 0]), dim=-1
        )
        shared_scale = torch.sqrt((noop_scale * static_scale).clamp_min(_EPS))

        role_index = {
            role.role_id: index
            for index, role in enumerate(self.graph_registry.roles)
        }
        edge_valid = torch.zeros(
            (len(ARMS), phases, edges), dtype=torch.bool
        )
        edge_geometry = torch.zeros(
            (len(ARMS), phases, edges, 4), dtype=torch.float32
        )
        for edge_index, edge in enumerate(self.graph_registry.edges):
            source = role_index[edge.source_role]
            destination = role_index[edge.target_role]
            observable = (
                self.graph_registry.roles[source].can_be_observed
                and self.graph_registry.roles[destination].can_be_observed
            )
            common = shared_frame_phase_valid.clone()
            if observable:
                common &= (
                    role_valid[:, :, source].all(dim=0)
                    & role_valid[:, :, destination].all(dim=0)
                )
            else:
                common.zero_()
            edge_valid[:, :, edge_index] = common.reshape(1, phases).expand(
                len(ARMS), phases
            )
            for arm_index in range(len(ARMS)):
                relative = (
                    centroids[arm_index, :, destination]
                    - centroids[arm_index, :, source]
                )
                dx = (relative * shared_direction).sum(dim=-1) / shared_scale.clamp_min(_EPS)
                dy = (relative * shared_perpendicular).sum(dim=-1) / shared_scale.clamp_min(_EPS)
                distance = torch.linalg.vector_norm(relative, dim=-1) / shared_scale.clamp_min(_EPS)
                radial = torch.zeros_like(distance)
                pair_valid = common[1:] & common[:-1]
                radial[1:] = torch.where(
                    pair_valid,
                    distance[1:] - distance[:-1],
                    torch.zeros_like(distance[1:]),
                )
                geometry = torch.stack((dx, dy, distance, radial), dim=-1)
                edge_geometry[arm_index, :, edge_index] = torch.where(
                    common.unsqueeze(-1), geometry, torch.zeros_like(geometry)
                )
                edge_relation[arm_index, :, edge_index] = torch.where(
                    common,
                    edge_relation[arm_index, :, edge_index],
                    torch.zeros_like(edge_relation[arm_index, :, edge_index]),
                )

        support_frame_valid = torch.full(
            (len(ARMS),), frame_admitted, dtype=torch.bool
        )
        common_mask_digest = _tensor_digest(edge_valid.to(torch.uint8))
        selected = support_evidence
        shared_frame_receipt = {
            "frame_sources": ["noop", "static"],
            "action_or_reverse_defined_frame": False,
            "support_endpoint_roles": [
                self.graph_registry.roles[start].role_id,
                self.graph_registry.roles[end].role_id,
            ],
            "support_evidence_phase_count": int(selected.sum().item()),
            "shared_frame_valid_phase_count": int(
                shared_frame_phase_valid.sum().item()
            ),
            "shared_frame_valid_phase_fraction": phase_fraction,
            "frame_admitted": frame_admitted,
            "endpoint_rms_mean": (
                float(endpoint_rms[selected].mean().item())
                if bool(selected.any().item())
                else None
            ),
            "direction_cosine_min": (
                float(direction_cosine[selected].min().item())
                if bool(selected.any().item())
                else None
            ),
            "log_scale_error_max": (
                float(log_scale_error[selected].max().item())
                if bool(selected.any().item())
                else None
            ),
            "phase_failure_abstains_all_four_arms": True,
            "absolute_centroids_zeroized_at_cell_close": True,
        }
        simplex_evidence_receipt = {
            "native_simplex_validated": True,
            "simplex_axis": "registered_role",
            "prior_equalized": True,
            "prior_equalization_formula": "u=p/pi;q=u/sum_role(u)",
            "absolute_gate_channels": [
                "raw_probability_floor",
                "q_probability",
                "log_u_enrichment",
                "q_second_competitor_margin",
            ],
            "role_order": list(self.graph_registry.role_ids),
            "role_partition_sha256_by_arm": {
                row.arm: row.role_partition_sha256 for row in rows
            },
            "role_token_counts_by_arm": {
                row.arm: list(row.role_token_counts) for row in rows
            },
            "role_prior_by_arm": {
                row.arm: row.role_prior.tolist() for row in rows
            },
            "absolute_eligible_mask_sha256_by_arm": {
                row.arm: _tensor_digest(
                    row.absolute_eligible_mask.to(torch.uint8)
                )
                for row in rows
            },
            "absolute_margin_min_by_arm": {
                row.arm: float(row.absolute_margins.min().item())
                for row in rows
            },
            "dustbin_has_lower_quota": False,
            "role_has_lower_quota": False,
            "failed_absolute_evidence_has_exact_zero_kernel": True,
        }
        digest_payload = {
            "schema_version": REDUCED_SCHEMA,
            "appearance_id": first.appearance_id,
            "sigma_band": first.sigma_band,
            "block_index": first.block_index,
            "state_sha256": first.state_sha256,
            "registry_digest": self.graph_registry.digest,
            "role_valid_sha256": _tensor_digest(role_valid),
            "role_confidence_sha256": _tensor_digest(role_confidence),
            "edge_relation_sha256": _tensor_digest(edge_relation),
            "edge_geometry_sha256": _tensor_digest(edge_geometry),
            "edge_valid_sha256": _tensor_digest(edge_valid.to(torch.uint8)),
            "support_frame_valid": support_frame_valid.tolist(),
            "shared_frame_phase_valid_sha256": _tensor_digest(
                shared_frame_phase_valid.to(torch.uint8)
            ),
            "shared_frame_receipt": shared_frame_receipt,
            "simplex_evidence_receipt": simplex_evidence_receipt,
            "common_four_arm_edge_mask_digest": common_mask_digest,
            "raw_inputs_zeroized": all(row.raw_input_zeroized for row in rows),
        }
        result = ReducedSameStateCellV4(
            schema_version=REDUCED_SCHEMA,
            appearance_id=first.appearance_id,
            sigma_band=first.sigma_band,
            block_index=first.block_index,
            role_valid=role_valid,
            role_confidence=role_confidence,
            edge_relation=edge_relation,
            edge_geometry=edge_geometry,
            edge_valid=edge_valid,
            support_frame_valid=support_frame_valid,
            shared_frame_phase_valid=shared_frame_phase_valid,
            shared_frame_receipt=shared_frame_receipt,
            simplex_evidence_receipt=simplex_evidence_receipt,
            common_four_arm_edge_mask_digest=common_mask_digest,
            state_sha256=first.state_sha256,
            raw_inputs_zeroized=all(row.raw_input_zeroized for row in rows),
            digest=registry_v4.object_sha256(digest_payload),
        )
        for row in rows:
            row.zeroize()
        self._arms.clear()
        return result


def assemble_same_state_cell_v4(
    reduced_by_arm: Mapping[str, ReducedArmObservationV4],
    *,
    graph_registry: registry_v4.ObserverRegistryV4,
) -> ReducedSameStateCellV4:
    assembler = SameStateCellAssemblerV4(graph_registry)
    try:
        if set(reduced_by_arm) != set(ARMS):
            raise PartialObjectGraphObserverV4Error("compact arm map key differs")
        for arm in ARMS:
            if reduced_by_arm[arm].arm != arm:
                raise PartialObjectGraphObserverV4Error("compact arm map key differs")
            assembler.add(reduced_by_arm[arm])
        return assembler.finalize()
    finally:
        assembler.abort()
        for row in reduced_by_arm.values():
            if not row.zeroized:
                row.zeroize()


def reduce_same_state_cell_v4(
    captures_by_arm: Mapping[str, MiddleObservationV4],
    *,
    graph_registry: registry_v4.ObserverRegistryV4,
) -> ReducedSameStateCellV4:
    """Reduce four same-state arms and immediately destroy all raw tensors."""

    if set(captures_by_arm) != set(ARMS):
        raise PartialObjectGraphObserverV4Error(
            "same-state cell must contain action/noop/reverse/static"
        )
    captures = [captures_by_arm[arm] for arm in ARMS]
    for capture in captures:
        capture.validate(graph_registry)
    invariants = (
        "appearance_id",
        "sigma_band",
        "block_index",
        "state_sha256",
        "timestep_sha256",
        "rotary_sha256",
        "patch_height",
        "patch_width",
    )
    for label in invariants:
        if len({getattr(capture, label) for capture in captures}) != 1:
            raise PartialObjectGraphObserverV4Error(
                f"same-state four-arm {label} differs"
            )
    if len({capture.prompt_sha256 for capture in captures}) != len(ARMS):
        raise PartialObjectGraphObserverV4Error(
            "four prompt arms are not independently authenticated"
        )

    # Compatibility wrapper.  Native callers should instead call
    # ``reduce_one_arm_v4`` as each branch completes and feed the compact rows
    # to ``SameStateCellAssemblerV4``.
    assembler = SameStateCellAssemblerV4(graph_registry)
    try:
        for arm in ARMS:
            assembler.add(
                reduce_one_arm_v4(
                    captures_by_arm[arm], graph_registry=graph_registry
                )
            )
        return assembler.finalize()
    except BaseException:
        # Ownership begins only after all boundary/invariant checks above.
        # Once reduction starts, scrub pending compact rows and every raw arm,
        # including arms not yet reached by the streaming loop.
        assembler.abort()
        for capture in captures:
            if not capture.consumed:
                capture.zeroize()
        raise


@dataclass(frozen=True)
class PartialObjectGraphResultV4:
    admitted: bool
    registry_digest: str
    role_ids: tuple[str, ...]
    edge_rows: tuple[Mapping[str, Any], ...]
    signed_relation_delta: torch.Tensor = field(repr=False)  # [T,E]
    relative_geometry_delta: torch.Tensor = field(repr=False)  # [T,E,4]
    edge_valid: torch.Tensor = field(repr=False)  # [T,E]
    edge_uncertainty: torch.Tensor = field(repr=False)  # [T,E]
    change_points: tuple[tuple[int, ...], ...]
    diagnostics: Mapping[str, Any]
    receipt_digest: str

    def public_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "method": METHOD,
            "status": (
                "FOUR_ARM_OBSERVER_COMPONENT_ADMITTED"
                if self.admitted
                else "REJECTED_UNCERTAIN"
            ),
            "registry_digest": self.registry_digest,
            "role_ids": list(self.role_ids),
            "directed_edges": [dict(row) for row in self.edge_rows],
            "signed_relation_delta": self.signed_relation_delta.tolist(),
            "relative_geometry_channels": [
                "support_relative_dx",
                "support_relative_dy",
                "normalized_pair_distance",
                "normalized_radial_velocity",
            ],
            "relative_geometry_delta": self.relative_geometry_delta.tolist(),
            "edge_valid": self.edge_valid.tolist(),
            "edge_uncertainty": self.edge_uncertainty.tolist(),
            "edge_change_points": [list(row) for row in self.change_points],
            "diagnostics": dict(self.diagnostics),
            "scientific_claim_authorized": False,
            "stable_transferable_action_representation_claimed": False,
            "renderer_or_injection_authorized": False,
        }
        encoded = registry_v4.canonical_json_bytes(payload)
        forbidden_tokens = (
            b'"raw_q"',
            b'"raw_k"',
            b'"raw_v"',
            b'"hidden_state"',
            b'"dense_role_responsibilities"',
            b'"absolute_coordinates"',
            b'"target_video"',
            b'"target_rgb"',
            b'"target_latent"',
        )
        if any(token in encoded for token in forbidden_tokens):
            raise PartialObjectGraphObserverV4Error(
                "public payload contains a forbidden field"
            )
        return payload

    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD,
            "registry_digest": self.registry_digest,
            "component_four_arm_mechanical_admitted": self.admitted,
            "representation_admitted": False,
            "full_oceg_representation_admitted": False,
            "observer_only": True,
            "base_frozen_required": True,
            "same_state_four_arm_required": True,
            "partial_assignment_has_dustbin": True,
            "partial_assignment_has_role_lower_quota": False,
            "native_role_proxy_simplex_validated": True,
            "token_prior_correction_applied": True,
            "prior_equalized_probability_gate_applied": True,
            "absolute_evidence_precedes_spatial_normalization": True,
            "failed_absolute_evidence_has_exact_zero_kernel": True,
            "shared_frame_sources": ["noop", "static"],
            "action_or_reverse_defined_shared_frame": False,
            "four_arm_common_edge_domain_required": True,
            "competitor_margin_is_shuffled_prompt_control": False,
            "shuffled_prompt_control_observed": False,
            "shuffled_prompt_robustness_claimed": False,
            "instruction_only_or_offscreen_can_be_observed": False,
            "raw_qk_retained": False,
            "dense_role_scores_retained": False,
            "absolute_anchor_coordinates_retained": False,
            "target_inputs_consumed": False,
            "target_at_inference_authorized": False,
            "final_anchor_video_consumed": False,
            "persistent_source_identity_registry_present": False,
            "contact_fsm_present": False,
            "physical_contact_truth_claimed": False,
            "target_graph_pretraining_performed": False,
            "renderer_called": False,
            "route_or_injection_called": False,
            "optimizer_created": False,
            "parameter_updates": 0,
            "scientific_claim_authorized": False,
            "stable_transferable_action_representation_claimed": False,
            "result_public_payload_sha256": registry_v4.object_sha256(
                self.public_payload()
            ),
            "diagnostics": dict(self.diagnostics),
        }
        digest = registry_v4.object_sha256(payload)
        if digest != self.receipt_digest:
            raise PartialObjectGraphObserverV4Error("receipt digest differs")
        return {**payload, "receipt_digest": digest}


class PartialObjectGraphObserverV4:
    """Streaming accumulator for 3 appearances × 3 sigma × 4 blocks."""

    def __init__(self, graph_registry: registry_v4.ObserverRegistryV4) -> None:
        if not isinstance(graph_registry, registry_v4.ObserverRegistryV4):
            raise PartialObjectGraphObserverV4Error("registry type differs")
        self.graph_registry = graph_registry
        self._cells: MutableMapping[tuple[str, str, int], ReducedSameStateCellV4] = {}

    def add(self, cell: ReducedSameStateCellV4) -> None:
        try:
            cell.validate(self.graph_registry)
            key = (cell.appearance_id, cell.sigma_band, cell.block_index)
            if key in self._cells:
                raise PartialObjectGraphObserverV4Error("duplicate reduced cell")
            self._cells[key] = cell
        except BaseException:
            if not cell.zeroized:
                cell.zeroize()
            self.abort()
            raise

    def abort(self) -> None:
        """Scrub every reduced cell still owned by the stream."""

        cells = tuple(self._cells.values())
        self._cells.clear()
        for cell in cells:
            if not cell.zeroized:
                cell.zeroize()
            if any(
                int(torch.count_nonzero(value).item()) != 0
                for value in (
                    cell.role_valid,
                    cell.role_confidence,
                    cell.edge_relation,
                    cell.edge_geometry,
                    cell.edge_valid,
                    cell.support_frame_valid,
                    cell.shared_frame_phase_valid,
                )
            ):
                raise PartialObjectGraphObserverV4Error(
                    "reduced-cell failure scrub did not close"
                )

    def observe_same_state_cell(
        self, captures_by_arm: Mapping[str, MiddleObservationV4]
    ) -> None:
        self.add(
            reduce_same_state_cell_v4(
                captures_by_arm, graph_registry=self.graph_registry
            )
        )

    def _aggregate_appearance_arm(
        self,
        appearance_id: str,
        arm_index: int,
    ) -> dict[str, Any]:
        cells = [
            self._cells[(appearance_id, sigma, block)]
            for sigma in SIGMA_BANDS
            for block in BLOCKS
        ]
        role_valid = torch.stack([cell.role_valid[arm_index] for cell in cells], dim=0)
        role_confidence = torch.stack(
            [cell.role_confidence[arm_index] for cell in cells], dim=0
        )
        edge_relation = torch.stack(
            [cell.edge_relation[arm_index] for cell in cells], dim=0
        )
        edge_geometry = torch.stack(
            [cell.edge_geometry[arm_index] for cell in cells], dim=0
        )
        edge_valid = torch.stack([cell.edge_valid[arm_index] for cell in cells], dim=0)

        phases = self.graph_registry.phases
        role_count = len(self.graph_registry.roles)
        edge_count = len(self.graph_registry.edges)
        consensus_role = torch.zeros((phases, role_count), dtype=torch.bool)
        consensus_confidence = torch.zeros((phases, role_count), dtype=torch.float32)
        for phase in range(phases):
            for role in range(role_count):
                valid_indices = [
                    index
                    for index, cell in enumerate(cells)
                    if bool(role_valid[index, phase, role].item())
                ]
                blocks = {cells[index].block_index for index in valid_indices}
                sigmas = {cells[index].sigma_band for index in valid_indices}
                accepted = (
                    len(blocks) >= int(self.graph_registry.thresholds["cross_block_count_min"])
                    and len(sigmas) >= int(self.graph_registry.thresholds["cross_sigma_count_min"])
                )
                consensus_role[phase, role] = accepted
                if accepted:
                    consensus_confidence[phase, role] = role_confidence[
                        valid_indices, phase, role
                    ].median()

        role_index = {
            role.role_id: index for index, role in enumerate(self.graph_registry.roles)
        }
        consensus_edge = torch.zeros((phases, edge_count), dtype=torch.bool)
        relation = torch.zeros((phases, edge_count), dtype=torch.float32)
        geometry = torch.zeros((phases, edge_count, 4), dtype=torch.float32)
        uncertainty = torch.ones((phases, edge_count), dtype=torch.float32)
        for edge_index, edge in enumerate(self.graph_registry.edges):
            source = role_index[edge.source_role]
            destination = role_index[edge.target_role]
            for phase in range(phases):
                valid_indices = [
                    index
                    for index, cell in enumerate(cells)
                    if bool(edge_valid[index, phase, edge_index].item())
                ]
                blocks = {cells[index].block_index for index in valid_indices}
                sigmas = {cells[index].sigma_band for index in valid_indices}
                accepted = (
                    bool(consensus_role[phase, source].item())
                    and bool(consensus_role[phase, destination].item())
                    and len(blocks) >= int(self.graph_registry.thresholds["cross_block_count_min"])
                    and len(sigmas) >= int(self.graph_registry.thresholds["cross_sigma_count_min"])
                )
                consensus_edge[phase, edge_index] = accepted
                if accepted:
                    weights = torch.minimum(
                        role_confidence[valid_indices, phase, source],
                        role_confidence[valid_indices, phase, destination],
                    ).clamp_min(_EPS)
                    relation[phase, edge_index] = (
                        edge_relation[valid_indices, phase, edge_index] * weights
                    ).sum() / weights.sum()
                    geometry[phase, edge_index] = (
                        edge_geometry[valid_indices, phase, edge_index]
                        * weights.unsqueeze(-1)
                    ).sum(dim=0) / weights.sum()
                    uncertainty[phase, edge_index] = 1.0 - float(
                        torch.minimum(
                            consensus_confidence[phase, source],
                            consensus_confidence[phase, destination],
                        ).item()
                    )

        support_indices = [
            index
            for index, cell in enumerate(cells)
            if bool(cell.support_frame_valid[arm_index].item())
        ]
        support_blocks = {cells[index].block_index for index in support_indices}
        support_sigmas = {cells[index].sigma_band for index in support_indices}
        support_admitted = (
            len(support_blocks) >= int(self.graph_registry.thresholds["cross_block_count_min"])
            and len(support_sigmas) >= int(self.graph_registry.thresholds["cross_sigma_count_min"])
        )
        critical_role_rows = []
        critical_complete = True
        for role_index_value, role in enumerate(self.graph_registry.roles):
            if not role.critical:
                continue
            coverage = float(consensus_role[:, role_index_value].float().mean().item())
            passed = (
                role.can_be_observed
                and coverage
                >= float(self.graph_registry.thresholds["critical_role_phase_fraction_min"])
            )
            critical_complete = critical_complete and passed
            critical_role_rows.append(
                {
                    "role_id": role.role_id,
                    "evidence_mode": role.evidence_mode,
                    "observed_phase_fraction": coverage,
                    "passed": passed,
                }
            )
        evidence_complete = critical_complete and (
            support_admitted or not self.graph_registry.requires_support_frame
        )
        return {
            "relation": relation,
            "geometry": geometry,
            "edge_valid": consensus_edge,
            "uncertainty": uncertainty,
            "role_valid": consensus_role,
            "role_confidence": consensus_confidence,
            "support_frame_admitted": support_admitted,
            "critical_role_rows": critical_role_rows,
            "evidence_complete": evidence_complete,
        }

    def finalize(self) -> PartialObjectGraphResultV4:
        try:
            return self._finalize_owned()
        finally:
            self.abort()

    def _finalize_owned(self) -> PartialObjectGraphResultV4:
        appearances = sorted({key[0] for key in self._cells})
        if len(appearances) != self.graph_registry.appearance_count:
            raise PartialObjectGraphObserverV4Error(
                "exactly three appearance variants are required"
            )
        expected = {
            (appearance, sigma, block)
            for appearance in appearances
            for sigma in SIGMA_BANDS
            for block in BLOCKS
        }
        if set(self._cells) != expected:
            raise PartialObjectGraphObserverV4Error(
                "multi-block/multi-sigma capture matrix is incomplete"
            )

        aggregate: dict[str, dict[str, dict[str, Any]]] = {}
        appearance_diagnostics = []
        action_signatures: dict[str, torch.Tensor] = {}
        action_signature_validity: dict[str, torch.Tensor] = {}
        all_controls = True
        for appearance in appearances:
            aggregate[appearance] = {
                arm: self._aggregate_appearance_arm(appearance, index)
                for index, arm in enumerate(ARMS)
            }
            action = aggregate[appearance]["action"]
            noop = aggregate[appearance]["noop"]
            reverse = aggregate[appearance]["reverse"]
            static = aggregate[appearance]["static"]
            common_valid = (
                action["edge_valid"]
                & noop["edge_valid"]
                & reverse["edge_valid"]
                & static["edge_valid"]
            )
            action_relation_delta = action["relation"] - noop["relation"]
            reverse_relation_delta = reverse["relation"] - noop["relation"]
            static_relation_delta = static["relation"] - noop["relation"]
            action_geometry_delta = action["geometry"] - noop["geometry"]
            reverse_geometry_delta = reverse["geometry"] - noop["geometry"]
            static_geometry_delta = static["geometry"] - noop["geometry"]
            action_signature = torch.cat(
                (action_relation_delta.unsqueeze(-1), action_geometry_delta), dim=-1
            )
            noop_signature = torch.cat(
                (noop["relation"].unsqueeze(-1), noop["geometry"]), dim=-1
            )
            reverse_signature = torch.cat(
                (reverse_relation_delta.unsqueeze(-1), reverse_geometry_delta), dim=-1
            )
            static_signature = torch.cat(
                (static_relation_delta.unsqueeze(-1), static_geometry_delta), dim=-1
            )
            action_signature = torch.where(
                common_valid.unsqueeze(-1), action_signature, torch.zeros_like(action_signature)
            )
            noop_signature = torch.where(
                noop["edge_valid"].unsqueeze(-1),
                noop_signature,
                torch.zeros_like(noop_signature),
            )
            reverse_signature = torch.where(
                common_valid.unsqueeze(-1), reverse_signature, torch.zeros_like(reverse_signature)
            )
            static_signature = torch.where(
                common_valid.unsqueeze(-1), static_signature, torch.zeros_like(static_signature)
            )
            action_signatures[appearance] = action_signature
            action_signature_validity[appearance] = common_valid

            action_rms = float(torch.sqrt(torch.mean(action_signature.double().square())).item())
            action_energy = _transition_energy(action_signature, common_valid)
            reverse_aligned = torch.flip(reverse_signature, dims=(0,))
            reverse_valid = torch.flip(common_valid, dims=(0,)) & common_valid
            # The last channel is a backward finite difference.  Its sample
            # index shifts under time reversal, so reverse admission compares
            # the phase-state channels (relation, dx, dy, distance) rather than
            # rewarding or rejecting a discretization-boundary artifact.
            action_reverse_state = action_signature[..., :-1]
            reverse_aligned_state = reverse_aligned[..., :-1]
            reverse_cosine = _cosine(
                action_reverse_state[reverse_valid], reverse_aligned_state[reverse_valid]
            ) if bool(reverse_valid.any().item()) else 0.0
            reverse_distance = _normalized_distance(
                action_reverse_state[reverse_valid], reverse_aligned_state[reverse_valid]
            ) if bool(reverse_valid.any().item()) else 1.0e9
            critical_observable_edges = torch.tensor(
                [
                    edge.critical
                    and next(
                        role
                        for role in self.graph_registry.roles
                        if role.role_id == edge.source_role
                    ).can_be_observed
                    and next(
                        role
                        for role in self.graph_registry.roles
                        if role.role_id == edge.target_role
                    ).can_be_observed
                    for edge in self.graph_registry.edges
                ],
                dtype=torch.bool,
            )
            endpoint_valid = reverse_valid[[0, -1]]
            endpoint_complete = bool(critical_observable_edges.any().item()) and bool(
                endpoint_valid[:, critical_observable_edges].all().item()
            )
            if endpoint_complete:
                endpoint_delta = (
                    action_reverse_state[[0, -1]]
                    - reverse_aligned_state[[0, -1]]
                )[:, critical_observable_edges]
                reverse_endpoint_rms = float(
                    torch.sqrt(torch.mean(endpoint_delta.double().square())).item()
                )
                reverse_endpoint_max_abs = float(
                    endpoint_delta.abs().max().item()
                )
            else:
                reverse_endpoint_rms = 1.0e9
                reverse_endpoint_max_abs = 1.0e9
            noop_energy = _transition_energy(
                noop_signature, noop["edge_valid"]
            )
            static_energy = _transition_energy(static_signature, common_valid)
            null_energy = max(noop_energy, static_energy, _EPS)
            gates = {
                "critical_role_evidence_complete": all(
                    aggregate[appearance][arm]["evidence_complete"] for arm in ARMS
                ),
                "required_edges_have_consensus": bool(common_valid.any().item()),
                "action_nonzero_against_noop": action_rms
                >= float(self.graph_registry.thresholds["action_delta_rms_min"]),
                "dynamic_exceeds_null": action_energy / null_energy
                >= float(self.graph_registry.thresholds["dynamic_over_null_ratio_min"]),
                "noop_lacks_transition": noop_energy / max(action_energy, _EPS)
                <= float(self.graph_registry.thresholds["null_transition_ratio_max"]),
                "static_lacks_transition": static_energy / max(action_energy, _EPS)
                <= float(self.graph_registry.thresholds["null_transition_ratio_max"]),
                "reverse_retimes_order_cosine": reverse_cosine
                >= float(self.graph_registry.thresholds["reverse_cycle_cosine_min"]),
                "reverse_retimes_order_distance": reverse_distance
                <= float(self.graph_registry.thresholds["reverse_cycle_distance_max"]),
                "reverse_endpoint_topology_complete": endpoint_complete,
                "reverse_endpoint_topology_rms": reverse_endpoint_rms
                <= float(
                    self.graph_registry.thresholds[
                        "reverse_endpoint_topology_rms_max"
                    ]
                ),
                "reverse_endpoint_topology_max_abs": reverse_endpoint_max_abs
                <= float(
                    self.graph_registry.thresholds[
                        "reverse_endpoint_topology_max_abs_max"
                    ]
                ),
            }
            passed = all(gates.values())
            all_controls = all_controls and passed
            appearance_diagnostics.append(
                {
                    "appearance_id": appearance,
                    "gates": gates,
                    "controls_passed": passed,
                    "metrics": {
                        "action_delta_rms": action_rms,
                        "action_transition_energy": action_energy,
                        "noop_transition_energy": noop_energy,
                        "static_transition_energy": static_energy,
                        "dynamic_over_null_ratio": action_energy / null_energy,
                        "reverse_cycle_cosine": reverse_cosine,
                        "reverse_cycle_normalized_distance": reverse_distance,
                        "reverse_endpoint_topology_rms": reverse_endpoint_rms,
                        "reverse_endpoint_topology_max_abs": reverse_endpoint_max_abs,
                    },
                    "arms": {
                        arm: {
                            "support_frame_admitted": aggregate[appearance][arm]["support_frame_admitted"],
                            "evidence_complete": aggregate[appearance][arm]["evidence_complete"],
                            "critical_roles": aggregate[appearance][arm]["critical_role_rows"],
                        }
                        for arm in ARMS
                    },
                }
            )

        transfer_rows = []
        transfer_passed = True
        for left_index, left in enumerate(appearances):
            for right in appearances[left_index + 1 :]:
                left_valid = action_signature_validity[left]
                right_valid = action_signature_validity[right]
                observable_roles = {
                    role.role_id: role.can_be_observed
                    for role in self.graph_registry.roles
                }
                eligible_edges = torch.tensor(
                    [
                        edge.critical
                        and observable_roles[edge.source_role]
                        and observable_roles[edge.target_role]
                        for edge in self.graph_registry.edges
                    ],
                    dtype=torch.bool,
                )
                common_valid = (left_valid & right_valid)[:, eligible_edges]
                common_fraction = (
                    float(common_valid.sum().item())
                    / max(float(common_valid.numel()), 1.0)
                )
                if bool(common_valid.any().item()):
                    cosine = _cosine(
                        action_signatures[left][:, eligible_edges][common_valid],
                        action_signatures[right][:, eligible_edges][common_valid],
                    )
                    distance = _normalized_distance(
                        action_signatures[left][:, eligible_edges][common_valid],
                        action_signatures[right][:, eligible_edges][common_valid],
                    )
                else:
                    cosine = 0.0
                    distance = 1.0e9
                passed = (
                    common_fraction
                    >= float(
                        self.graph_registry.thresholds[
                            "appearance_common_valid_fraction_min"
                        ]
                    )
                    and cosine >= float(self.graph_registry.thresholds["appearance_cosine_min"])
                    and distance <= float(self.graph_registry.thresholds["appearance_distance_max"])
                )
                transfer_passed = transfer_passed and passed
                transfer_rows.append(
                    {
                        "left": left,
                        "right": right,
                        "cosine": cosine,
                        "normalized_distance": distance,
                        "common_valid_phase_edge_fraction": common_fraction,
                        "passed": passed,
                    }
                )
        admitted = all_controls and transfer_passed

        action_relations = []
        action_geometries = []
        action_validity = []
        action_uncertainty = []
        for appearance in appearances:
            action = aggregate[appearance]["action"]
            noop = aggregate[appearance]["noop"]
            reverse = aggregate[appearance]["reverse"]
            static = aggregate[appearance]["static"]
            valid = (
                action["edge_valid"]
                & noop["edge_valid"]
                & reverse["edge_valid"]
                & static["edge_valid"]
            )
            action_relations.append(action["relation"] - noop["relation"])
            action_geometries.append(action["geometry"] - noop["geometry"])
            action_validity.append(valid)
            action_uncertainty.append(
                torch.maximum(action["uncertainty"], noop["uncertainty"])
            )
        stacked_valid = torch.stack(action_validity, dim=0)
        public_valid = stacked_valid.all(dim=0)
        relation = torch.stack(action_relations, dim=0).mean(dim=0)
        geometry = torch.stack(action_geometries, dim=0).mean(dim=0)
        uncertainty = torch.stack(action_uncertainty, dim=0).max(dim=0).values
        relation = torch.where(public_valid, relation, torch.zeros_like(relation))
        geometry = torch.where(
            public_valid.unsqueeze(-1), geometry, torch.zeros_like(geometry)
        )
        uncertainty = torch.where(public_valid, uncertainty, torch.ones_like(uncertainty))

        change_points = []
        for edge in range(len(self.graph_registry.edges)):
            feature = torch.cat(
                (relation[:, edge : edge + 1], geometry[:, edge]), dim=-1
            )
            delta = torch.linalg.vector_norm(feature[1:] - feature[:-1], dim=-1)
            valid_pair = public_valid[1:, edge] & public_valid[:-1, edge]
            points = tuple(
                int(index + 1)
                for index in torch.nonzero(
                    valid_pair
                    & (
                        delta
                        >= float(self.graph_registry.thresholds["change_point_delta_min"])
                    ),
                    as_tuple=False,
                ).flatten().tolist()
            )
            change_points.append(points)

        edge_rows = []
        for edge in self.graph_registry.edges:
            source = next(
                role for role in self.graph_registry.roles if role.role_id == edge.source_role
            )
            destination = next(
                role for role in self.graph_registry.roles if role.role_id == edge.target_role
            )
            relation_type = edge.relation_type
            if not source.can_be_observed or not destination.can_be_observed:
                relation_type = "instruction_relation_unresolved"
            edge_rows.append(
                {
                    "source_role": edge.source_role,
                    "destination_role": edge.target_role,
                    "relation_type": relation_type,
                    "critical": edge.critical,
                    "physical_contact_truth_claimed": False,
                }
            )

        diagnostics: dict[str, Any] = {
            "appearance_controls": appearance_diagnostics,
            "multiappearance_consensus": transfer_rows,
            "all_control_gates_passed": all_controls,
            "all_appearance_consensus_gates_passed": transfer_passed,
            "full_capture_matrix_observed": True,
            "raw_capture_count": len(self._cells) * len(ARMS),
            "raw_capture_count_zeroized": len(self._cells) * len(ARMS),
            "reduced_cell_count": len(self._cells),
            "reduced_cells_zeroized_after_finalize": True,
            "uncertain_counts_as_failure": True,
            "native_role_proxy_simplex_validated": True,
            "token_prior_correction_applied": True,
            "prior_equalized_probability_gate_applied": True,
            "absolute_evidence_precedes_spatial_normalization": True,
            "failed_absolute_evidence_has_exact_zero_kernel": True,
            "shared_frame_sources": ["noop", "static"],
            "action_or_reverse_defined_shared_frame": False,
            "four_arm_common_edge_domain_required": True,
            "cell_evidence_receipts": [
                {
                    "appearance_id": cell.appearance_id,
                    "sigma_band": cell.sigma_band,
                    "block_index": cell.block_index,
                    "state_sha256": cell.state_sha256,
                    "shared_frame": dict(cell.shared_frame_receipt),
                    "simplex_evidence": dict(cell.simplex_evidence_receipt),
                    "common_four_arm_edge_mask_digest": (
                        cell.common_four_arm_edge_mask_digest
                    ),
                }
                for _key, cell in sorted(self._cells.items())
            ],
            "thresholds_preregistered": {
                key: self.graph_registry.thresholds[key]
                for key in sorted(self.graph_registry.thresholds)
            },
            "claim_boundary": (
                "observer component only; no persistent source identity registry, "
                "contact FSM, target-teacher pretraining result, renderer result, "
                "or stable transferable action representation claim"
            ),
        }
        for cell in self._cells.values():
            cell.zeroize()

        provisional_payload = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD,
            "registry_digest": self.graph_registry.digest,
            "component_four_arm_mechanical_admitted": admitted,
            "representation_admitted": False,
            "full_oceg_representation_admitted": False,
            "observer_only": True,
            "base_frozen_required": True,
            "same_state_four_arm_required": True,
            "partial_assignment_has_dustbin": True,
            "partial_assignment_has_role_lower_quota": False,
            "native_role_proxy_simplex_validated": True,
            "token_prior_correction_applied": True,
            "prior_equalized_probability_gate_applied": True,
            "absolute_evidence_precedes_spatial_normalization": True,
            "failed_absolute_evidence_has_exact_zero_kernel": True,
            "shared_frame_sources": ["noop", "static"],
            "action_or_reverse_defined_shared_frame": False,
            "four_arm_common_edge_domain_required": True,
            "competitor_margin_is_shuffled_prompt_control": False,
            "shuffled_prompt_control_observed": False,
            "shuffled_prompt_robustness_claimed": False,
            "instruction_only_or_offscreen_can_be_observed": False,
            "raw_qk_retained": False,
            "dense_role_scores_retained": False,
            "absolute_anchor_coordinates_retained": False,
            "target_inputs_consumed": False,
            "target_at_inference_authorized": False,
            "final_anchor_video_consumed": False,
            "persistent_source_identity_registry_present": False,
            "contact_fsm_present": False,
            "physical_contact_truth_claimed": False,
            "target_graph_pretraining_performed": False,
            "renderer_called": False,
            "route_or_injection_called": False,
            "optimizer_created": False,
            "parameter_updates": 0,
            "scientific_claim_authorized": False,
            "stable_transferable_action_representation_claimed": False,
            "result_public_payload_sha256": "PENDING",
            "diagnostics": diagnostics,
        }
        # Construct once to obtain the public payload digest, then seal receipt.
        draft = PartialObjectGraphResultV4(
            admitted,
            self.graph_registry.digest,
            self.graph_registry.role_ids,
            tuple(edge_rows),
            relation.contiguous(),
            geometry.contiguous(),
            public_valid.contiguous(),
            uncertainty.contiguous(),
            tuple(change_points),
            diagnostics,
            "0" * 64,
        )
        provisional_payload["result_public_payload_sha256"] = registry_v4.object_sha256(
            draft.public_payload()
        )
        receipt_digest = registry_v4.object_sha256(provisional_payload)
        return PartialObjectGraphResultV4(
            admitted,
            self.graph_registry.digest,
            self.graph_registry.role_ids,
            tuple(edge_rows),
            relation.contiguous(),
            geometry.contiguous(),
            public_valid.contiguous(),
            uncertainty.contiguous(),
            tuple(change_points),
            diagnostics,
            receipt_digest,
        )


def observe_same_state_bundle_v4(
    captures: Mapping[
        tuple[str, str, int], Mapping[str, MiddleObservationV4]
    ],
    *,
    graph_registry: registry_v4.ObserverRegistryV4,
) -> PartialObjectGraphResultV4:
    """Convenience CPU/toy entry point; native runtime should stream cells."""

    observer = PartialObjectGraphObserverV4(graph_registry)
    try:
        for key in sorted(captures):
            arms = captures[key]
            if any(
                (item.appearance_id, item.sigma_band, item.block_index) != key
                for item in arms.values()
            ):
                raise PartialObjectGraphObserverV4Error(
                    "bundle key differs from capture"
                )
            observer.observe_same_state_cell(arms)
        return observer.finalize()
    finally:
        observer.abort()


__all__ = [
    "MiddleObservationV4",
    "PartialObjectGraphObserverV4",
    "PartialObjectGraphObserverV4Error",
    "PartialObjectGraphResultV4",
    "ReducedArmObservationV4",
    "ReducedSameStateCellV4",
    "SameStateCellAssemblerV4",
    "assemble_same_state_cell_v4",
    "observe_same_state_bundle_v4",
    "reduce_one_arm_v4",
    "reduce_same_state_cell_v4",
]
