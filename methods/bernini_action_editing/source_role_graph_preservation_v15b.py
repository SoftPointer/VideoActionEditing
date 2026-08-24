"""Independent CPU reference seam for v15b source-property action editing.

The pure-T2V anchor can cross this seam only as a coordinate-free relation
graph.  Source appearance crosses through unordered, pre-RoPE role-local
position-scrubbed, fixed-instance phase-0 content slots.  Target ownership is
initialized once from the source proposals and may then move only through an
explicit one-to-one previous-phase transport reference; current target K
can retain a transported slot or make it unassigned, but cannot rename it.
Source motion is represented separately and is *removed*
from the signed edit graph; source background/support can be restored at the
same coordinates.  Editable-object hidden/K/V tensors are never copied at
the same coordinates after phase 0; the full phase-0 source frame is an
explicitly counted identity boundary condition shared by every causal arm.

This module does not import the v14/v15a controllers and does not patch or run
a video model.  It is deliberately a small CPU tensor/ABI reference.  In
particular, its r8 source material stores immutable canonical bytes and is
reopened at every builder/consumer boundary.  Those bytes prove internal
phase-0 hidden/pre-RoPE-K/V/mask consistency, not that caller-supplied video or
latent hashes came from a production source.  Its fixed position fixture and
label-map transport estimator are
self-contained synthetic witnesses: they authenticate their own material
replay, but they do *not* authenticate a production model, prove that position
was removed from source K, or establish native optical flow.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import itertools
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


METHOD = "bernini-source-role-graph-preservation-v15b"
LATENT_PHASES = 21
DENOISE_STEPS = 40
TRANSFORMER_BLOCKS = 22
CFG_BRANCHES = ("negative", "conditional")
EXPECTED_EXECUTION_CELLS = DENOISE_STEPS * TRANSFORMER_BLOCKS * len(CFG_BRANCHES)
GENERIC_ROLES = ("human_agent", "moving_object", "recipient")
SIGNED_ROLES = ("human_agent", "old_actor", "moving_object", "recipient")
DEFAULT_GRAPH_A_SLOT = "v0"
DEFAULT_GRAPH_B_SLOT = "v1"
DIAGNOSTIC_ANCHOR_SLOTS = ("v0", "v1", "v2", "v3")
STRICT_MIN_EDGE_COSINE = 0.95
STRICT_MAX_EDGE_DISTANCE = 0.15
DTW_MAX_PHASE_DISPLACEMENT = 6
DTW_MAX_SOURCE_ONLY_RUN = 4
DTW_MAX_CANONICAL_ONLY_RUN = 4
DTW_MAX_PATH_LENGTH = 31
STRICT_MIN_GRAPH_CONFIDENCE = 0.95
REQUIRED_EDGE_MIN_NORM = 0.5
REQUIRED_EDGE_MIN_QUERY_PHASES = 3
REQUIRED_EDGE_MIN_KEY_PHASES = 4
CORRIDOR_DILATION_RADIUS = 1
MIN_TRACK_AREA_PIXELS = 2
MAX_TRACK_AREA_FRACTION = 0.20
MAX_TRACK_AREA_RATIO = 4.0
MAX_TRACK_CENTROID_JUMP = 3.5
MAX_CORRIDOR_FRACTION = 0.70
MIN_BACKGROUND_FRACTION = 0.25
MIN_CONTENT_SLOTS_PER_ROLE = 2
TARGET_ROLE_MIN_COSINE = 0.80
TARGET_ROLE_NULL_MARGIN = 0.05
TARGET_ROLE_WINNER_MARGIN = 0.05
TARGET_ROLE_TEMPERATURE = 0.10
TARGET_TRACK_MAX_CENTROID_JUMP = 3.5
TARGET_TRACK_MAX_VACANCY_FRACTION = 0.50
TARGET_TRACK_MIN_AREA_PIXELS = 1
POSITION_PROJECTOR_TOLERANCE = 1.0e-5
POSITION_SVD_RELATIVE_TOLERANCE = 1.0e-5
POSITION_COUNTERFACTUAL_MIN_COUNT = 2
POSITION_CALIBRATION_BATCH = 1
POSITION_CALIBRATION_CHANNELS = 2
POSITION_CALIBRATION_HEIGHT = 7
POSITION_CALIBRATION_WIDTH = 7
POSITION_CALIBRATION_HEADS = 1
POSITION_CALIBRATION_HEAD_DIM = 8
POSITION_FIT_TRANSLATIONS_YX = ((-1, 0), (1, 0), (0, -1), (0, 1))
POSITION_HELDOUT_TRANSLATIONS_YX = ((-2, 0), (2, 0), (0, -2), (0, 2), (1, 1), (-1, -1))
MOTION_INTEGER_TOLERANCE = 1.0e-6
TARGET_TRACK_MIN_PHASE0_AREA_RATIO = 0.50
TARGET_TRACK_MAX_PHASE0_AREA_RATIO = 2.00
TARGET_TRACK_MAX_ASPECT_RATIO_CHANGE = 2.00
TARGET_TRACK_MAX_DIAMETER_RATIO = 1.75
TARGET_TRACK_MIN_COMPACTNESS_RATIO = 0.60
TARGET_TRACK_MAX_BOUNDARY_RATIO = 1.50
TARGET_TRACK_MIN_TRANSLATED_IOU = 0.50
TARGET_TRACK_MAX_TRANSLATED_HAUSDORFF = 1.00
TARGET_TRACK_MAX_CUMULATIVE_PATH_FACTOR = 2.0
TARGET_TRACK_STALE_MOTION_EPSILON = 0.25

# Target-action edges are registered, not inferred from donor energy.  Any
# anchor relation outside this list is structurally unreachable by target Q.
ACTION_ALLOWED_ADD_EDGES = {
    "pour": (
        ("human_agent", "human_agent"),
        ("moving_object", "moving_object"),
        ("recipient", "recipient"),
        ("human_agent", "moving_object"),
        ("moving_object", "recipient"),
    ),
}
ACTION_REQUIRED_ADD_EDGES = {
    "pour": (
        ("human_agent", "moving_object"),
        ("moving_object", "recipient"),
    ),
}
ACTION_ALLOWED_REMOVE_EDGES = {
    "pour": (
        ("human_agent", "old_actor"),
        ("old_actor", "moving_object"),
    ),
}
ACTION_REQUIRED_REMOVE_EDGES = ACTION_ALLOWED_REMOVE_EDGES

ANCHOR_GRAPH_SCHEMA = "bernini-anchor-relation-graph-only-v15b"
SOURCE_GRAPH_SCHEMA = "bernini-source-relation-graph-v15b"
SIGNED_GRAPH_SCHEMA = "bernini-signed-relation-edit-graph-v15b"
TRACE_SCHEMA = "bernini-role-contact-trace-v15b"
WARP_SCHEMA = "bernini-monotonic-event-warp-v15b"
BINDING_SCHEMA = "bernini-source-action-role-binding-v15b"
MASK_SCHEMA = "bernini-source-role-mask-set-v15b-r8"
TRACK_AUTHORITY_SCHEMA = "bernini-source-role-track-authority-v15b-r8"
POSITION_CALIBRATION_FIXTURE_SCHEMA = "bernini-fixed-position-calibration-fixture-v15b-r8"
POSITION_REFERENCE_SCHEMA = "bernini-position-counterfactual-reference-v15b-r8"
RAW_SOURCE_MATERIAL_SCHEMA = "bernini-source-phase0-raw-material-v15b-r8"
CANONICAL_EXTRACTION_CONFIG_SCHEMA = "bernini-source-phase0-canonical-extraction-v15b-r8"
SLOT_PROVENANCE_SCHEMA = "bernini-source-phase0-slot-provenance-v15b-r8"
MEMORY_SCHEMA = "bernini-source-role-content-memory-v15b-r8"
MEMORY_BUILDER_RECEIPT_SCHEMA = "bernini-source-role-content-builder-receipt-v15b-r8"
NATIVE_MOTION_REFERENCE_SCHEMA = "bernini-target-transport-reference-v15b-r8"
TARGET_TRANSPORT_SCHEMA = "bernini-target-transport-v15b-r8"
TARGET_ROLE_STATE_SCHEMA = "bernini-target-persistent-role-state-v15b-r8"
BACKGROUND_SCHEMA = "bernini-source-background-carrier-v15b-r8"
BLOCK_AUDIT_SCHEMA = "bernini-source-role-block-audit-v15b-r8"
FOUR_ARM_SCHEMA = "bernini-source-role-preservation-four-arm-contract-v15b-r8"

ARM_K0 = "A_K0_KEYED"
ARM_GRAPH_A_MEMORY = "B_GRAPH_A_SOURCE_CONTENT_MEMORY"
ARM_GRAPH_A_FULL = "C_GRAPH_A_MEMORY_BACKGROUND_RESTORE"
ARM_GRAPH_B_FULL = "D_GRAPH_B_MEMORY_BACKGROUND_RESTORE"
ARM_IDS = (ARM_K0, ARM_GRAPH_A_MEMORY, ARM_GRAPH_A_FULL, ARM_GRAPH_B_FULL)

ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SLOT_RE = re.compile(r"^v[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Exact and recursively scanned at all public mapping/receipt boundaries.
FORBIDDEN_ANCHOR_FIELDS = frozenset(
    {
        "anchor_video", "anchor_video_path", "anchor_rgb", "anchor_pixels",
        "anchor_value", "anchor_v", "anchor_key", "anchor_k",
        "anchor_hidden", "anchor_attention_output", "anchor_latent",
        "anchor_gaussian", "anchor_initial_gaussian", "donor_value",
        "donor_v", "donor_key", "donor_k", "donor_rgb", "donor_pixels",
        "donor_hidden", "donor_attention_output", "donor_latent",
        "donor_gaussian", "absolute_space", "absolute_spatial_coordinate",
        "spatial_index", "spatial_indices", "spatial_shape", "rgb_summary",
        "appearance_embedding", "material_embedding", "color_embedding",
    }
)


class V15BContractError(RuntimeError):
    """Raised instead of silently widening the v15b seam."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V15BContractError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise V15BContractError("tensor digest requires a material tensor")
    logical = value.detach().contiguous().cpu()
    header = canonical_json_bytes(
        {"dtype": str(logical.dtype), "shape": list(logical.shape)}
    )
    try:
        # Avoid a NumPy ABI dependency; this reference must be reopenable in
        # the same minimal CPU environment used by the validator.
        raw = bytes(logical.view(torch.uint8).reshape(-1).tolist())
    except Exception as error:
        raise V15BContractError("cannot materialize tensor digest") from error
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise V15BContractError(f"{label} must be a lowercase SHA-256")
    return value


def _role(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or ROLE_RE.fullmatch(value) is None:
        raise V15BContractError(f"{label} is not a canonical role ID")
    return value


def _slot(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SLOT_RE.fullmatch(value) is None:
        raise V15BContractError(f"{label} is not an opaque anchor slot")
    return value


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise V15BContractError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V15BContractError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise V15BContractError(f"{label} must be finite")
    return result


def _tensor(value: Any, *, label: str, ndim: int, floating: bool = True) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise V15BContractError(f"{label} must be rank {ndim}")
    if floating and not value.is_floating_point():
        raise V15BContractError(f"{label} must be floating point")
    if floating and not bool(torch.isfinite(value).all()):
        raise V15BContractError(f"{label} is non-finite")
    return value


def _cpu_fp32(value: torch.Tensor, *, label: str, ndim: int) -> torch.Tensor:
    value = _tensor(value, label=label, ndim=ndim)
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise V15BContractError(f"{label} must be CPU FP32")
    return value


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise V15BContractError("max-abs geometry differs")
    if left.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max())


def _masked_max_abs(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    if tuple(left.shape[:2]) != tuple(mask.shape) or tuple(right.shape) != tuple(left.shape):
        raise V15BContractError("masked max-abs geometry differs")
    expanded = mask.reshape(*mask.shape, *([1] * (left.ndim - 2))).expand_as(left)
    return _max_abs(left.masked_select(expanded), right.masked_select(expanded))


def _revalidate_material(value: Any, *, label: str) -> None:
    """Recompute current material authority at every consumer boundary.

    ``frozen=True`` protects dataclass attributes, not tensor storage.  Calling
    this helper after construction deliberately replays the object's complete
    invariant/digest check so an in-place mutation cannot ride a stale digest.
    Nested material classes invoke the same check for their children.
    """
    method = getattr(value, "__post_init__", None)
    if method is None or not callable(method):
        raise V15BContractError(f"{label} is not revalidatable material")
    method()


def _scan_forbidden_mapping(value: Mapping[str, Any], *, label: str) -> None:
    forbidden = {str(key).lower() for key in value}.intersection(FORBIDDEN_ANCHOR_FIELDS)
    if forbidden:
        raise V15BContractError(f"{label} contains forbidden anchor content: {sorted(forbidden)}")


def _project_relation_graph(graph: torch.Tensor) -> torch.Tensor:
    """Re-center key-time slices and keep phase-0 an exact identity boundary."""
    result = graph.float().clone()
    result -= result.mean(dim=3, keepdim=True)
    result[:, 0].zero_()
    return result


def _validate_relation_tensor(
    graph: torch.Tensor, *, roles: tuple[str, ...], label: str
) -> torch.Tensor:
    graph = _cpu_fp32(graph, label=label, ndim=5)
    expected = (LATENT_PHASES, len(roles), LATENT_PHASES, len(roles))
    if int(graph.shape[0]) < 1 or tuple(graph.shape[1:]) != expected:
        raise V15BContractError(f"{label} must be [H,21,{len(roles)},21,{len(roles)}]")
    if int(torch.count_nonzero(graph[:, 0])):
        raise V15BContractError(f"{label} phase-0 query slice must be zero")
    # Each query-role/key-role temporal operator is centered independently.
    if float(graph.sum(dim=3).abs().max()) > 1.0e-5:
        raise V15BContractError(f"{label} key-time slices must sum to zero")
    return graph


def _edge_norm_and_query_phase_coverage(
    graph: torch.Tensor, roles: tuple[str, ...], edge: tuple[str, str]
) -> tuple[float, int]:
    """Return a physical edge norm and number of nonzero query phases."""
    query_role, key_role = edge
    try:
        query_index = roles.index(query_role)
        key_index = roles.index(key_role)
    except ValueError as error:
        raise V15BContractError(f"edge {query_role}->{key_role} is outside vocabulary") from error
    logical = graph[:, :, query_index, :, key_index]
    norm = float(torch.linalg.vector_norm(logical.double()))
    coverage = int((logical.abs().amax(dim=(0, 2)) > 0).sum())
    return norm, coverage


def _edge_key_phase_coverage(
    graph: torch.Tensor, roles: tuple[str, ...], edge: tuple[str, str]
) -> int:
    query_role, key_role = edge
    logical = graph[
        :, :, roles.index(query_role), :, roles.index(key_role)
    ]
    return int((logical.abs().amax(dim=(0, 1)) > 0).sum())


def _require_registered_edges(
    *, graph: torch.Tensor, roles: tuple[str, ...], edges: Sequence[tuple[str, str]],
    label: str,
) -> None:
    for edge in edges:
        norm, coverage = _edge_norm_and_query_phase_coverage(graph, roles, edge)
        if norm < REQUIRED_EDGE_MIN_NORM:
            raise V15BContractError(f"{label} lacks required {edge[0]}->{edge[1]} norm")
        if coverage < REQUIRED_EDGE_MIN_QUERY_PHASES:
            raise V15BContractError(
                f"{label} required {edge[0]}->{edge[1]} has insufficient phase coverage"
            )
        if _edge_key_phase_coverage(graph, roles, edge) < REQUIRED_EDGE_MIN_KEY_PHASES:
            raise V15BContractError(
                f"{label} required {edge[0]}->{edge[1]} has insufficient key-phase coverage"
            )


@dataclass(frozen=True)
class AnchorRelationGraphV15B:
    """The entire anchor ABI: G[head,T,R,T,R], with no spatial/content axis."""

    schema_version: str
    action_id: str
    generic_roles: tuple[str, ...]
    graph: torch.Tensor
    confidence: float
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != ANCHOR_GRAPH_SCHEMA or self.generic_roles != GENERIC_ROLES:
            raise V15BContractError("anchor graph schema/role vocabulary differs")
        _role(self.action_id, label="anchor graph action")
        graph = _validate_relation_tensor(
            self.graph, roles=GENERIC_ROLES, label="anchor relation graph"
        )
        confidence = _finite(self.confidence, label="anchor graph confidence")
        if not 0.0 <= confidence <= 1.0:
            raise V15BContractError("anchor graph confidence is outside [0,1]")
        if confidence < STRICT_MIN_GRAPH_CONFIDENCE:
            raise V15BContractError("anchor graph confidence is below the registered gate")
        required = ACTION_REQUIRED_ADD_EDGES.get(self.action_id)
        if required is None:
            raise V15BContractError("anchor action has no required-edge registry")
        _require_registered_edges(
            graph=graph, roles=GENERIC_ROLES, edges=required,
            label="anchor relation graph",
        )
        if object_sha256(self._payload()) != _sha(self.digest, label="anchor graph digest"):
            raise V15BContractError("anchor graph digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "action_id": self.action_id,
            "generic_roles": self.generic_roles, "graph_sha256": tensor_sha256(self.graph),
            "confidence": self.confidence,
        }

    @classmethod
    def create(cls, *, action_id: str, graph: torch.Tensor, confidence: float = 1.0) -> "AnchorRelationGraphV15B":
        logical = _cpu_fp32(graph, label="anchor relation graph create", ndim=5).detach().clone()
        confidence = _finite(confidence, label="anchor graph confidence create")
        payload = {
            "schema_version": ANCHOR_GRAPH_SCHEMA, "action_id": action_id,
            "generic_roles": GENERIC_ROLES, "graph_sha256": tensor_sha256(logical),
            "confidence": confidence,
        }
        return cls(ANCHOR_GRAPH_SCHEMA, action_id, GENERIC_ROLES, logical,
                   confidence, object_sha256(payload))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnchorRelationGraphV15B":
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise V15BContractError("anchor graph mapping fields differ")
        _scan_forbidden_mapping(value, label="anchor graph mapping")
        row = dict(value)
        row["generic_roles"] = tuple(row["generic_roles"])
        return cls(**row)


@dataclass(frozen=True)
class SingleRoleTemporalGraphAblationV15B:
    """T x T ablation only; deliberately incompatible with the formal bank."""

    action_id: str
    role_id: str
    graph: torch.Tensor

    def __post_init__(self) -> None:
        _role(self.action_id, label="ablation action")
        _role(self.role_id, label="ablation role")
        graph = _cpu_fp32(self.graph, label="single-role ablation", ndim=3)
        if tuple(graph.shape[1:]) != (LATENT_PHASES, LATENT_PHASES):
            raise V15BContractError("single-role ablation must be [H,21,21]")


@dataclass(frozen=True)
class AnchorGraphBankV15B:
    action_id: str
    graph_set_id: str
    anchor_slot: str
    timing_trace_digest: str
    relation_graph: AnchorRelationGraphV15B
    digest: str

    def __post_init__(self) -> None:
        _role(self.action_id, label="graph bank action")
        _role(self.graph_set_id, label="graph set")
        _slot(self.anchor_slot, label="anchor slot")
        _sha(self.timing_trace_digest, label="graph timing trace")
        if not isinstance(self.relation_graph, AnchorRelationGraphV15B):
            raise V15BContractError("formal graph bank requires a relation graph")
        _revalidate_material(self.relation_graph, label="graph bank relation graph")
        if self.relation_graph.action_id != self.action_id:
            raise V15BContractError("graph bank action differs")
        payload = {
            "action_id": self.action_id, "graph_set_id": self.graph_set_id,
            "anchor_slot": self.anchor_slot, "timing_trace_digest": self.timing_trace_digest,
            "graph_digest": self.relation_graph.digest,
        }
        if object_sha256(payload) != _sha(self.digest, label="graph bank digest"):
            raise V15BContractError("graph bank digest differs")

    @classmethod
    def create(cls, *, action_id: str, graph_set_id: str, anchor_slot: str,
               timing_trace_digest: str,
               relation_graph: AnchorRelationGraphV15B) -> "AnchorGraphBankV15B":
        if not isinstance(relation_graph, AnchorRelationGraphV15B):
            raise V15BContractError("formal graph bank requires a relation graph")
        payload = {"action_id": action_id, "graph_set_id": graph_set_id,
                   "anchor_slot": anchor_slot, "timing_trace_digest": timing_trace_digest,
                   "graph_digest": relation_graph.digest}
        return cls(action_id, graph_set_id, anchor_slot, timing_trace_digest,
                   relation_graph, object_sha256(payload))


@dataclass(frozen=True)
class SourceRelationGraphV15B:
    """Source-only old interaction graph; it never carries appearance."""

    schema_version: str
    action_id: str
    roles: tuple[str, ...]
    timing_trace_digest: str
    graph: torch.Tensor
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_GRAPH_SCHEMA or self.roles != SIGNED_ROLES:
            raise V15BContractError("source relation graph schema/roles differ")
        _role(self.action_id, label="source graph action")
        _sha(self.timing_trace_digest, label="source graph timing trace")
        graph = _validate_relation_tensor(self.graph, roles=SIGNED_ROLES,
                                          label="source relation graph")
        required = ACTION_REQUIRED_REMOVE_EDGES.get(self.action_id)
        if required is None:
            raise V15BContractError("source action has no required-remove registry")
        _require_registered_edges(
            graph=graph, roles=SIGNED_ROLES, edges=required,
            label="source relation graph",
        )
        payload = {
            "schema_version": self.schema_version, "action_id": self.action_id,
            "roles": self.roles, "timing_trace_digest": self.timing_trace_digest,
            "graph_sha256": tensor_sha256(graph),
        }
        if object_sha256(payload) != _sha(self.digest, label="source graph digest"):
            raise V15BContractError("source graph digest differs")

    @classmethod
    def create(cls, *, action_id: str, timing_trace_digest: str,
               graph: torch.Tensor) -> "SourceRelationGraphV15B":
        logical = _cpu_fp32(
            graph, label="source relation graph create", ndim=5
        ).detach().clone()
        payload = {
            "schema_version": SOURCE_GRAPH_SCHEMA, "action_id": action_id,
            "roles": SIGNED_ROLES, "timing_trace_digest": timing_trace_digest,
            "graph_sha256": tensor_sha256(logical),
        }
        return cls(SOURCE_GRAPH_SCHEMA, action_id, SIGNED_ROLES,
                   timing_trace_digest, logical,
                   object_sha256(payload))


@dataclass(frozen=True)
class RoleContactTraceV15B:
    """Timing-only trace: generic-role activity plus contact energy."""

    schema_version: str
    anchor_slot: str
    asset_sha256: str
    extractor_code_sha256: str
    extractor_config_sha256: str
    channels: tuple[str, ...]
    energy: torch.Tensor
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA:
            raise V15BContractError("role/contact trace schema differs")
        _slot(self.anchor_slot, label="trace slot")
        _sha(self.asset_sha256, label="trace asset")
        _sha(self.extractor_code_sha256, label="trace extractor code")
        _sha(self.extractor_config_sha256, label="trace extractor config")
        expected_channels = GENERIC_ROLES + ("contact",)
        if self.channels != expected_channels:
            raise V15BContractError("trace channels must be role/contact-only")
        energy = _cpu_fp32(self.energy, label="role/contact energy", ndim=2)
        if tuple(energy.shape) != (LATENT_PHASES, len(expected_channels)):
            raise V15BContractError("role/contact trace geometry differs")
        if float(energy.min()) < 0.0:
            raise V15BContractError("role/contact energy must be nonnegative")
        payload = {
            "schema_version": self.schema_version,
            "anchor_slot": self.anchor_slot,
            "asset_sha256": self.asset_sha256,
            "extractor_code_sha256": self.extractor_code_sha256,
            "extractor_config_sha256": self.extractor_config_sha256,
            "channels": self.channels,
            "energy_sha256": tensor_sha256(energy),
        }
        if object_sha256(payload) != _sha(self.digest, label="trace digest"):
            raise V15BContractError("role/contact trace digest differs")

    @classmethod
    def create(
        cls, *, anchor_slot: str, asset_sha256: str,
        extractor_code_sha256: str, extractor_config_sha256: str,
        energy: torch.Tensor,
    ) -> "RoleContactTraceV15B":
        logical = _cpu_fp32(
            energy, label="role/contact trace create", ndim=2
        ).detach().clone()
        payload = {
            "schema_version": TRACE_SCHEMA,
            "anchor_slot": anchor_slot,
            "asset_sha256": asset_sha256,
            "extractor_code_sha256": extractor_code_sha256,
            "extractor_config_sha256": extractor_config_sha256,
            "channels": GENERIC_ROLES + ("contact",),
            "energy_sha256": tensor_sha256(logical),
        }
        return cls(
            TRACE_SCHEMA, anchor_slot, asset_sha256, extractor_code_sha256,
            extractor_config_sha256, GENERIC_ROLES + ("contact",), logical,
            object_sha256(payload),
        )


@dataclass(frozen=True)
class MonotonicEventWarpV15B:
    """Coordinate-free canonical timing warp computed only from role/contact energy."""

    schema_version: str
    source_slot: str
    canonical_slot: str
    source_trace_digest: str
    canonical_trace_digest: str
    path: tuple[tuple[int, int], ...]
    resample: torch.Tensor
    max_phase_displacement: int
    max_source_only_run: int
    max_canonical_only_run: int
    path_length: int
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != WARP_SCHEMA:
            raise V15BContractError("event warp schema differs")
        _slot(self.source_slot, label="warp source slot")
        _slot(self.canonical_slot, label="warp canonical slot")
        _sha(self.source_trace_digest, label="source trace digest")
        _sha(self.canonical_trace_digest, label="canonical trace digest")
        if not self.path or self.path[0] != (0, 0) or self.path[-1] != (
            LATENT_PHASES - 1, LATENT_PHASES - 1
        ):
            raise V15BContractError("DTW path must cover both endpoints")
        previous = (-1, -1)
        source_only_run = 0
        canonical_only_run = 0
        measured_source_only = 0
        measured_canonical_only = 0
        for source, canonical in self.path:
            if not (0 <= source < LATENT_PHASES and 0 <= canonical < LATENT_PHASES):
                raise V15BContractError("DTW path index is outside phase range")
            step = (source - previous[0], canonical - previous[1])
            if previous != (-1, -1) and step not in ((1, 0), (0, 1), (1, 1)):
                raise V15BContractError("DTW path is not monotonic/contiguous")
            if previous != (-1, -1):
                source_only_run = source_only_run + 1 if step == (1, 0) else 0
                canonical_only_run = canonical_only_run + 1 if step == (0, 1) else 0
                measured_source_only = max(measured_source_only, source_only_run)
                measured_canonical_only = max(measured_canonical_only, canonical_only_run)
            previous = (source, canonical)
        measured_displacement = max(abs(i - j) for i, j in self.path)
        if (
            self.max_phase_displacement != measured_displacement
            or self.max_source_only_run != measured_source_only
            or self.max_canonical_only_run != measured_canonical_only
            or self.path_length != len(self.path)
        ):
            raise V15BContractError("event warp gate metrics do not match its path")
        if measured_displacement > DTW_MAX_PHASE_DISPLACEMENT:
            raise V15BContractError("event warp exceeds registered phase displacement")
        if measured_source_only > DTW_MAX_SOURCE_ONLY_RUN:
            raise V15BContractError("event warp exceeds registered source-only run")
        if measured_canonical_only > DTW_MAX_CANONICAL_ONLY_RUN:
            raise V15BContractError("event warp exceeds registered canonical-only run")
        if len(self.path) > DTW_MAX_PATH_LENGTH:
            raise V15BContractError("event warp exceeds registered path length")
        resample = _cpu_fp32(self.resample, label="event resample", ndim=2)
        if tuple(resample.shape) != (LATENT_PHASES, LATENT_PHASES):
            raise V15BContractError("event resample must be [21,21]")
        if float(resample.min()) < 0.0 or float((resample.sum(1) - 1).abs().max()) > 1e-6:
            raise V15BContractError("event resample rows must be probabilities")
        if not (resample[0, 0] == 1 and resample[-1, -1] == 1):
            raise V15BContractError("event resample endpoints differ")
        expected_resample = torch.zeros_like(resample)
        for canonical_phase in range(LATENT_PHASES):
            indices = [source for source, canonical in self.path
                       if canonical == canonical_phase]
            if not indices:
                raise V15BContractError("DTW path omitted a canonical phase")
            expected_resample[canonical_phase, indices] = 1.0 / len(indices)
        expected_resample[0].zero_(); expected_resample[0, 0] = 1.0
        expected_resample[-1].zero_(); expected_resample[-1, -1] = 1.0
        if not torch.equal(resample, expected_resample):
            raise V15BContractError("event resample is not derived from its DTW path")
        payload = self._payload()
        if object_sha256(payload) != _sha(self.digest, label="event warp digest"):
            raise V15BContractError("event warp digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "source_slot": self.source_slot,
            "canonical_slot": self.canonical_slot,
            "source_trace_digest": self.source_trace_digest,
            "canonical_trace_digest": self.canonical_trace_digest,
            "path": [list(item) for item in self.path],
            "resample_sha256": tensor_sha256(self.resample),
            "max_phase_displacement": self.max_phase_displacement,
            "max_source_only_run": self.max_source_only_run,
            "max_canonical_only_run": self.max_canonical_only_run,
            "path_length": self.path_length,
        }


def _warp_gate_metrics(path: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    displacement = max(abs(i - j) for i, j in path)
    source_run = canonical_run = max_source = max_canonical = 0
    for (old_i, old_j), (new_i, new_j) in zip(path, path[1:]):
        step = (new_i - old_i, new_j - old_j)
        source_run = source_run + 1 if step == (1, 0) else 0
        canonical_run = canonical_run + 1 if step == (0, 1) else 0
        max_source = max(max_source, source_run)
        max_canonical = max(max_canonical, canonical_run)
    return displacement, max_source, max_canonical, len(path)


def compute_monotonic_event_warp_v15b(
    source: RoleContactTraceV15B, canonical: RoleContactTraceV15B
) -> MonotonicEventWarpV15B:
    """Classic monotonic DTW; cost sees only the four role/contact channels."""
    left = source.energy.double()
    right = canonical.energy.double()
    # Normalize channels to avoid one legal energy scale dominating timing.
    scale = torch.maximum(left.amax(0), right.amax(0)).clamp_min(1e-8)
    cost = torch.cdist(left / scale, right / scale, p=2)
    n = LATENT_PHASES
    dp = torch.full((n, n), float("inf"), dtype=torch.float64)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    dp[0, 0] = cost[0, 0]
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            candidates = []
            # Tie order favors diagonal, then advancing source, then canonical.
            if i and j:
                candidates.append((float(dp[i - 1, j - 1]), (i - 1, j - 1)))
            if i:
                candidates.append((float(dp[i - 1, j]), (i - 1, j)))
            if j:
                candidates.append((float(dp[i, j - 1]), (i, j - 1)))
            _, predecessor = min(candidates, key=lambda item: item[0])
            dp[i, j] = cost[i, j] + dp[predecessor]
            parent[(i, j)] = predecessor
    path = [(n - 1, n - 1)]
    while path[-1] != (0, 0):
        path.append(parent[path[-1]])
    path.reverse()
    resample = torch.zeros(n, n, dtype=torch.float32)  # canonical x source
    for canonical_phase in range(n):
        indices = [i for i, j in path if j == canonical_phase]
        if not indices:
            raise V15BContractError("DTW path omitted a canonical phase")
        resample[canonical_phase, indices] = 1.0 / len(indices)
    resample[0].zero_(); resample[0, 0] = 1.0
    resample[-1].zero_(); resample[-1, -1] = 1.0
    gate_metrics = _warp_gate_metrics(path)
    payload = {
        "schema_version": WARP_SCHEMA, "source_slot": source.anchor_slot,
        "canonical_slot": canonical.anchor_slot,
        "source_trace_digest": source.digest,
        "canonical_trace_digest": canonical.digest,
        "path": [list(item) for item in path],
        "resample_sha256": tensor_sha256(resample),
        "max_phase_displacement": gate_metrics[0],
        "max_source_only_run": gate_metrics[1],
        "max_canonical_only_run": gate_metrics[2],
        "path_length": gate_metrics[3],
    }
    return MonotonicEventWarpV15B(
        WARP_SCHEMA, source.anchor_slot, canonical.anchor_slot,
        source.digest, canonical.digest, tuple(path), resample,
        gate_metrics[0], gate_metrics[1], gate_metrics[2], gate_metrics[3],
        object_sha256(payload),
    )


def _warp_graph(graph: torch.Tensor, warp: MonotonicEventWarpV15B) -> torch.Tensor:
    warped = torch.einsum("ti,hirjs,uj->htrus", warp.resample, graph, warp.resample)
    return _project_relation_graph(warped)


def _graph_metrics(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    a = left.reshape(-1).double(); b = right.reshape(-1).double()
    an = float(torch.linalg.vector_norm(a)); bn = float(torch.linalg.vector_norm(b))
    if an == 0.0 and bn == 0.0:
        return 1.0, 0.0
    if an == 0.0 or bn == 0.0:
        return 0.0, float("inf")
    return float(torch.dot(a, b) / (an * bn)), float(torch.linalg.vector_norm(a / an - b / bn))


@dataclass(frozen=True)
class GraphEdgeMetricV15B:
    query_role: str
    key_role: str
    raw_cosine: float
    raw_normalized_frobenius_distance: float
    aligned_cosine: float
    aligned_normalized_frobenius_distance: float

    def __post_init__(self) -> None:
        _role(self.query_role, label="edge query role")
        _role(self.key_role, label="edge key role")
        for label, value in (
            ("edge raw cosine", self.raw_cosine),
            ("edge raw distance", self.raw_normalized_frobenius_distance),
            ("edge aligned cosine", self.aligned_cosine),
            ("edge aligned distance", self.aligned_normalized_frobenius_distance),
        ):
            _finite(value, label=label)

    @property
    def passed(self) -> bool:
        return (
            self.aligned_cosine >= STRICT_MIN_EDGE_COSINE
            and self.aligned_normalized_frobenius_distance
            <= STRICT_MAX_EDGE_DISTANCE
        )


@dataclass(frozen=True)
class GraphPairDiagnosticV15B:
    action_id: str
    slot_a: str
    slot_b: str
    graph_a_digest: str
    graph_b_digest: str
    warp_a_digest: str
    warp_b_digest: str
    warp_a_gate: tuple[int, int, int, int]
    warp_b_gate: tuple[int, int, int, int]
    raw_cosine: float
    raw_normalized_frobenius_distance: float
    aligned_cosine: float
    aligned_normalized_frobenius_distance: float
    edge_metrics: tuple[GraphEdgeMetricV15B, ...]
    min_aligned_cosine: float = STRICT_MIN_EDGE_COSINE
    max_aligned_distance: float = STRICT_MAX_EDGE_DISTANCE

    def __post_init__(self) -> None:
        _role(self.action_id, label="pair action")
        _slot(self.slot_a, label="pair slot A"); _slot(self.slot_b, label="pair slot B")
        for label, value in (("graph A", self.graph_a_digest), ("graph B", self.graph_b_digest),
                             ("warp A", self.warp_a_digest), ("warp B", self.warp_b_digest)):
            _sha(value, label=label)
        limits = (
            DTW_MAX_PHASE_DISPLACEMENT, DTW_MAX_SOURCE_ONLY_RUN,
            DTW_MAX_CANONICAL_ONLY_RUN, DTW_MAX_PATH_LENGTH,
        )
        for label, gate in (("warp A", self.warp_a_gate), ("warp B", self.warp_b_gate)):
            if (not isinstance(gate, tuple) or len(gate) != 4 or
                    any(isinstance(item, bool) or not isinstance(item, int)
                        for item in gate)):
                raise V15BContractError(f"{label} gate metrics differ")
            if any(value > limit for value, limit in zip(gate, limits)):
                raise V15BContractError(f"{label} exceeds registered timing gate")
        for label, value in (("raw cosine", self.raw_cosine),
                             ("raw distance", self.raw_normalized_frobenius_distance),
                             ("aligned cosine", self.aligned_cosine),
                             ("aligned distance", self.aligned_normalized_frobenius_distance)):
            _finite(value, label=label)
        allowed = ACTION_ALLOWED_ADD_EDGES.get(self.action_id)
        if allowed is None:
            raise V15BContractError("pair action has no registered critical edges")
        if tuple((item.query_role, item.key_role) for item in self.edge_metrics) != allowed:
            raise V15BContractError("pair edge diagnostics do not match action registry")
        if (self.min_aligned_cosine != STRICT_MIN_EDGE_COSINE or
                self.max_aligned_distance != STRICT_MAX_EDGE_DISTANCE):
            raise V15BContractError("graph swap thresholds cannot be relaxed")

    @property
    def passed(self) -> bool:
        # The global metric is diagnostic only; every critical edge must pass.
        return all(item.passed for item in self.edge_metrics)

    @property
    def digest(self) -> str:
        return object_sha256({
            "action_id": self.action_id, "slot_a": self.slot_a,
            "slot_b": self.slot_b, "graph_a_digest": self.graph_a_digest,
            "graph_b_digest": self.graph_b_digest,
            "warp_a_digest": self.warp_a_digest, "warp_b_digest": self.warp_b_digest,
            "warp_a_gate": self.warp_a_gate, "warp_b_gate": self.warp_b_gate,
            "raw_cosine": self.raw_cosine,
            "raw_normalized_frobenius_distance": self.raw_normalized_frobenius_distance,
            "aligned_cosine": self.aligned_cosine,
            "aligned_normalized_frobenius_distance": self.aligned_normalized_frobenius_distance,
            "edge_metrics": [
                {f.name: getattr(item, f.name) for f in fields(item)}
                for item in self.edge_metrics
            ],
            "min_aligned_cosine": self.min_aligned_cosine,
            "max_aligned_distance": self.max_aligned_distance,
        })


def compare_anchor_graphs_v15b(
    graph_a: AnchorGraphBankV15B, graph_b: AnchorGraphBankV15B,
    warp_a: MonotonicEventWarpV15B, warp_b: MonotonicEventWarpV15B,
) -> GraphPairDiagnosticV15B:
    if graph_a.action_id != graph_b.action_id:
        raise V15BContractError("pair graph action differs")
    if warp_a.source_slot != graph_a.anchor_slot or warp_b.source_slot != graph_b.anchor_slot:
        raise V15BContractError("pair graph/warp slot differs")
    if (warp_a.source_trace_digest != graph_a.timing_trace_digest or
            warp_b.source_trace_digest != graph_b.timing_trace_digest):
        raise V15BContractError("pair graph/warp timing-trace authority differs")
    if warp_a.canonical_slot != warp_b.canonical_slot:
        raise V15BContractError("pair graphs were not independently aligned to one canonical trace")
    raw_a = graph_a.relation_graph.graph; raw_b = graph_b.relation_graph.graph
    aligned_a = _warp_graph(raw_a, warp_a); aligned_b = _warp_graph(raw_b, warp_b)
    raw = _graph_metrics(raw_a, raw_b)
    aligned = _graph_metrics(aligned_a, aligned_b)
    edge_metrics = []
    allowed = ACTION_ALLOWED_ADD_EDGES.get(graph_a.action_id)
    if allowed is None:
        raise V15BContractError("pair action has no registered critical edges")
    for query_role, key_role in allowed:
        query_index = GENERIC_ROLES.index(query_role)
        key_index = GENERIC_ROLES.index(key_role)
        edge_raw = _graph_metrics(
            raw_a[:, :, query_index, :, key_index],
            raw_b[:, :, query_index, :, key_index],
        )
        edge_aligned = _graph_metrics(
            aligned_a[:, :, query_index, :, key_index],
            aligned_b[:, :, query_index, :, key_index],
        )
        edge_metrics.append(GraphEdgeMetricV15B(
            query_role, key_role, edge_raw[0], edge_raw[1],
            edge_aligned[0], edge_aligned[1],
        ))
    return GraphPairDiagnosticV15B(
        graph_a.action_id, graph_a.anchor_slot, graph_b.anchor_slot,
        graph_a.digest, graph_b.digest,
        warp_a.digest, warp_b.digest,
        (warp_a.max_phase_displacement, warp_a.max_source_only_run,
         warp_a.max_canonical_only_run, warp_a.path_length),
        (warp_b.max_phase_displacement, warp_b.max_source_only_run,
         warp_b.max_canonical_only_run, warp_b.path_length),
        raw[0], raw[1], aligned[0], aligned[1],
        tuple(edge_metrics), STRICT_MIN_EDGE_COSINE, STRICT_MAX_EDGE_DISTANCE,
    )


@dataclass(frozen=True)
class FourAnchorConsensusReportV15B:
    slots: tuple[str, ...]
    canonical_trace_digest: str
    extractor_code_sha256: str
    extractor_config_sha256: str
    asset_sha256_by_slot: tuple[tuple[str, str], ...]
    bank_digest_by_slot: tuple[tuple[str, str], ...]
    warp_digest_by_slot: tuple[tuple[str, str], ...]
    pairs: tuple[GraphPairDiagnosticV15B, ...]
    consensus_cosines: tuple[float, ...]
    consensus_distances: tuple[float, ...]
    min_aligned_cosine: float
    max_aligned_distance: float

    def __post_init__(self) -> None:
        if self.slots != DIAGNOSTIC_ANCHOR_SLOTS or len(self.pairs) != 6:
            raise V15BContractError("four-anchor diagnostic must contain v0-v3 and six pairs")
        _sha(self.canonical_trace_digest, label="canonical trace")
        _sha(self.extractor_code_sha256, label="trace extractor code")
        _sha(self.extractor_config_sha256, label="trace extractor config")
        expected_pairs = tuple(itertools.combinations(DIAGNOSTIC_ANCHOR_SLOTS, 2))
        actual_pairs = tuple((item.slot_a, item.slot_b) for item in self.pairs)
        if actual_pairs != expected_pairs:
            raise V15BContractError("four-anchor diagnostics are not the unique canonical six pairs")
        for pair in self.pairs:
            _revalidate_material(pair, label="four-anchor pair diagnostic")
        for label, authority in (
            ("asset", self.asset_sha256_by_slot),
            ("bank", self.bank_digest_by_slot),
            ("warp", self.warp_digest_by_slot),
        ):
            if tuple(slot for slot, _ in authority) != DIAGNOSTIC_ANCHOR_SLOTS:
                raise V15BContractError(f"four-anchor {label} authority slots differ")
            for _, digest in authority:
                _sha(digest, label=f"four-anchor {label} authority")
        if len({digest for _, digest in self.asset_sha256_by_slot}) != 4:
            raise V15BContractError("v0-v3 must bind four distinct physical assets")
        if len(self.consensus_cosines) != 4 or len(self.consensus_distances) != 4:
            raise V15BContractError("four-anchor consensus metric count differs")
        if (self.min_aligned_cosine != STRICT_MIN_EDGE_COSINE or
                self.max_aligned_distance != STRICT_MAX_EDGE_DISTANCE):
            raise V15BContractError("four-anchor thresholds cannot be relaxed")

    @property
    def robust_passed(self) -> bool:
        # Strict all-four gate; no threshold relaxation hides structural mismatch.
        return (all(pair.passed for pair in self.pairs) and
                all(c >= STRICT_MIN_EDGE_COSINE and d <= STRICT_MAX_EDGE_DISTANCE
                    for c, d in zip(self.consensus_cosines, self.consensus_distances)))

    @property
    def digest(self) -> str:
        return object_sha256({
            "slots": self.slots,
            "canonical_trace_digest": self.canonical_trace_digest,
            "extractor_code_sha256": self.extractor_code_sha256,
            "extractor_config_sha256": self.extractor_config_sha256,
            "asset_sha256_by_slot": self.asset_sha256_by_slot,
            "bank_digest_by_slot": self.bank_digest_by_slot,
            "warp_digest_by_slot": self.warp_digest_by_slot,
            "pair_digests": [p.digest for p in self.pairs],
            "consensus_cosines": self.consensus_cosines,
            "consensus_distances": self.consensus_distances,
            "min_aligned_cosine": self.min_aligned_cosine,
            "max_aligned_distance": self.max_aligned_distance,
        })


def diagnose_four_anchor_consensus_v15b(
    banks: Sequence[AnchorGraphBankV15B], warps: Sequence[MonotonicEventWarpV15B],
    traces: Sequence[RoleContactTraceV15B], canonical_trace: RoleContactTraceV15B,
) -> FourAnchorConsensusReportV15B:
    by_slot = {item.anchor_slot: item for item in banks}
    warp_by_slot = {item.source_slot: item for item in warps}
    trace_by_slot = {item.anchor_slot: item for item in traces}
    if (len(by_slot) != 4 or len(warp_by_slot) != 4 or len(trace_by_slot) != 4 or
            tuple(sorted(by_slot)) != DIAGNOSTIC_ANCHOR_SLOTS or
            set(warp_by_slot) != set(by_slot) or set(trace_by_slot) != set(by_slot)):
        raise V15BContractError("four-anchor banks/warps/traces must be uniquely v0-v3")
    if not isinstance(canonical_trace, RoleContactTraceV15B):
        raise V15BContractError("four-anchor diagnostic requires a canonical trace")
    extractor_authorities = {
        (item.extractor_code_sha256, item.extractor_config_sha256)
        for item in tuple(trace_by_slot.values()) + (canonical_trace,)
    }
    if len(extractor_authorities) != 1:
        raise V15BContractError("v0-v3/canonical traces do not share extractor authority")
    for slot in DIAGNOSTIC_ANCHOR_SLOTS:
        trace = trace_by_slot[slot]
        bank = by_slot[slot]
        warp = warp_by_slot[slot]
        if bank.timing_trace_digest != trace.digest:
            raise V15BContractError("graph bank is not bound to its physical trace")
        if (warp.source_trace_digest != trace.digest or
                warp.canonical_trace_digest != canonical_trace.digest or
                warp.canonical_slot != canonical_trace.anchor_slot):
            raise V15BContractError("warp trace/canonical authority differs")
    pairs = []
    for i, left_slot in enumerate(DIAGNOSTIC_ANCHOR_SLOTS):
        for right_slot in DIAGNOSTIC_ANCHOR_SLOTS[i + 1:]:
            pairs.append(compare_anchor_graphs_v15b(
                by_slot[left_slot], by_slot[right_slot], warp_by_slot[left_slot],
                warp_by_slot[right_slot],
            ))
    aligned = torch.stack([
        _warp_graph(by_slot[slot].relation_graph.graph, warp_by_slot[slot])
        for slot in DIAGNOSTIC_ANCHOR_SLOTS
    ])
    consensus = aligned.median(dim=0).values
    metrics = [_graph_metrics(item, consensus) for item in aligned]
    extractor_code, extractor_config = next(iter(extractor_authorities))
    return FourAnchorConsensusReportV15B(
        DIAGNOSTIC_ANCHOR_SLOTS, canonical_trace.digest, extractor_code,
        extractor_config,
        tuple((slot, trace_by_slot[slot].asset_sha256)
              for slot in DIAGNOSTIC_ANCHOR_SLOTS),
        tuple((slot, by_slot[slot].digest) for slot in DIAGNOSTIC_ANCHOR_SLOTS),
        tuple((slot, warp_by_slot[slot].digest) for slot in DIAGNOSTIC_ANCHOR_SLOTS),
        tuple(pairs), tuple(x[0] for x in metrics),
        tuple(x[1] for x in metrics), STRICT_MIN_EDGE_COSINE,
        STRICT_MAX_EDGE_DISTANCE,
    )


@dataclass(frozen=True)
class SourceActionRoleBindingV15B:
    schema_version: str
    action_id: str
    source_iid: str
    human_agent_source_role: str
    old_actor_source_role: str
    moving_object_source_role: str
    recipient_source_role: str
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != BINDING_SCHEMA:
            raise V15BContractError("source binding schema differs")
        _role(self.action_id, label="binding action")
        if not isinstance(self.source_iid, str) or not self.source_iid:
            raise V15BContractError("source iid is empty")
        if len(set(self.source_roles)) != 4:
            raise V15BContractError("source-bound action roles must be distinct")
        for role in self.source_roles:
            _role(role, label="source-bound role")
        if object_sha256(self._payload()) != _sha(self.digest, label="binding digest"):
            raise V15BContractError("source binding digest differs")

    @property
    def source_roles(self) -> tuple[str, ...]:
        return (self.human_agent_source_role, self.old_actor_source_role,
                self.moving_object_source_role, self.recipient_source_role)

    @property
    def signed_to_source(self) -> dict[str, str]:
        return dict(zip(SIGNED_ROLES, self.source_roles))

    def _payload(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "digest"}

    @classmethod
    def create(cls, *, action_id: str, source_iid: str, human_agent_source_role: str,
               old_actor_source_role: str, moving_object_source_role: str,
               recipient_source_role: str) -> "SourceActionRoleBindingV15B":
        payload = {
            "schema_version": BINDING_SCHEMA, "action_id": action_id,
            "source_iid": source_iid, "human_agent_source_role": human_agent_source_role,
            "old_actor_source_role": old_actor_source_role,
            "moving_object_source_role": moving_object_source_role,
            "recipient_source_role": recipient_source_role,
        }
        return cls(**payload, digest=object_sha256(payload))


@dataclass(frozen=True)
class SourceRoleTrackAuthorityV15B:
    """Source-only authority for four fully tracked, authenticated role masks."""

    schema_version: str
    source_video_sha256: str
    binding_digest: str
    temporal_phases: int
    height: int
    width: int
    role_track_receipt_sha256: tuple[tuple[str, str], ...]
    role_mask_sha256: tuple[tuple[str, str], ...]
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != TRACK_AUTHORITY_SCHEMA:
            raise V15BContractError("source track authority schema differs")
        _sha(self.source_video_sha256, label="source track video")
        _sha(self.binding_digest, label="source track binding")
        if self.temporal_phases != LATENT_PHASES:
            raise V15BContractError("source tracks must cover exactly 21 phases")
        _exact_int(self.height, label="source track height", minimum=1)
        _exact_int(self.width, label="source track width", minimum=1)
        receipt_roles = tuple(role for role, _ in self.role_track_receipt_sha256)
        mask_roles = tuple(role for role, _ in self.role_mask_sha256)
        if receipt_roles != mask_roles or len(receipt_roles) != 4 or len(set(receipt_roles)) != 4:
            raise V15BContractError("source track authority must bind four ordered roles")
        for role, digest in self.role_track_receipt_sha256:
            _role(role, label="source track role"); _sha(digest, label="source track receipt")
        for role, digest in self.role_mask_sha256:
            _role(role, label="source track mask role"); _sha(digest, label="source track mask")
        if len({digest for _, digest in self.role_track_receipt_sha256}) != 4:
            raise V15BContractError("four source roles must have distinct track receipts")
        if object_sha256(self._payload()) != _sha(self.digest, label="source track authority"):
            raise V15BContractError("source track authority digest differs")

    def _payload(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "digest"}


def _packed_mask_4d(mask: torch.Tensor, *, height: int, width: int, label: str) -> torch.Tensor:
    _tensor(mask, label=label, ndim=2, floating=False)
    if mask.dtype != torch.bool or mask.device.type != "cpu":
        raise V15BContractError(f"{label} must be a CPU bool tensor")
    if int(mask.shape[1]) != LATENT_PHASES * height * width:
        raise V15BContractError(f"{label} T/H/W geometry differs")
    return mask.reshape(int(mask.shape[0]), LATENT_PHASES, height, width)


def _binary_dilate_4d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask.clone()
    batch, phases, height, width = mask.shape
    flat = mask.reshape(batch * phases, 1, height, width).float()
    dilated = F.max_pool2d(flat, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return dilated.bool().reshape(batch, phases, height, width)


def _centroid(mask_2d: torch.Tensor) -> tuple[float, float]:
    coordinates = torch.nonzero(mask_2d, as_tuple=False)
    if not len(coordinates):
        raise V15BContractError("source track omitted a phase")
    count = int(coordinates.shape[0])
    return (
        int(coordinates[:, 0].sum()) / count,
        int(coordinates[:, 1].sum()) / count,
    )


def _centroid_pixel(mask_2d: torch.Tensor) -> tuple[int, int]:
    """Nearest deterministic grid point; gates always use the float centroid."""
    y, x = _centroid(mask_2d)
    return math.floor(y + 0.5), math.floor(x + 0.5)


def _component_count_4(mask_2d: torch.Tensor) -> int:
    coordinates = {tuple(int(x) for x in row) for row in torch.nonzero(mask_2d)}
    components = 0
    while coordinates:
        components += 1
        stack = [coordinates.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in coordinates:
                    coordinates.remove(neighbor)
                    stack.append(neighbor)
    return components


def _draw_bresenham(mask_2d: torch.Tensor, start: tuple[int, int], end: tuple[int, int]) -> None:
    y0, x0 = start; y1, x1 = end
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        mask_2d[y0, x0] = True
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy; x0 += sx
        if twice <= dx:
            error += dx; y0 += sy


def _source_vessel_tubes(
    role_masks_4d: Mapping[str, torch.Tensor], vessel_roles: Sequence[str]
) -> torch.Tensor:
    example = next(iter(role_masks_4d.values()))
    tubes = torch.zeros_like(example)
    for role in vessel_roles:
        track = role_masks_4d[role]
        tubes |= track
        for batch_index in range(int(track.shape[0])):
            for phase in range(1, LATENT_PHASES):
                _draw_bresenham(
                    tubes[batch_index, phase],
                    _centroid_pixel(track[batch_index, phase - 1]),
                    _centroid_pixel(track[batch_index, phase]),
                )
    return _binary_dilate_4d(tubes, CORRIDOR_DILATION_RADIUS)


def _source_transition_path(
    moving: torch.Tensor, recipient: torch.Tensor,
) -> torch.Tensor:
    path = torch.zeros_like(moving)
    for batch_index in range(int(moving.shape[0])):
        for phase in range(1, LATENT_PHASES):
            _draw_bresenham(
                path[batch_index, phase],
                _centroid_pixel(moving[batch_index, phase]),
                _centroid_pixel(recipient[batch_index, phase]),
            )
    return _binary_dilate_4d(path, CORRIDOR_DILATION_RADIUS)


@dataclass(frozen=True)
class SourceRoleMaskSetV15B:
    schema_version: str
    source_video_sha256: str
    binding_digest: str
    track_authority: SourceRoleTrackAuthorityV15B
    temporal_phases: int
    height: int
    width: int
    dilation_radius: int
    role_masks: Mapping[str, torch.Tensor]
    contact_mask: torch.Tensor
    vessel_tube_mask: torch.Tensor
    transition_path_mask: torch.Tensor
    editable_corridor_mask: torch.Tensor
    background_support_mask: torch.Tensor
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != MASK_SCHEMA:
            raise V15BContractError("source mask schema differs")
        _sha(self.source_video_sha256, label="mask source video")
        _sha(self.binding_digest, label="mask binding digest")
        if (not isinstance(self.track_authority, SourceRoleTrackAuthorityV15B) or
                self.track_authority.source_video_sha256 != self.source_video_sha256 or
                self.track_authority.binding_digest != self.binding_digest):
            raise V15BContractError("mask track authority differs")
        _revalidate_material(self.track_authority, label="mask track authority")
        if (self.temporal_phases, self.height, self.width) != (
            self.track_authority.temporal_phases,
            self.track_authority.height,
            self.track_authority.width,
        ) or self.dilation_radius != CORRIDOR_DILATION_RADIUS:
            raise V15BContractError("mask T/H/W or dilation contract differs")
        authority_roles = tuple(role for role, _ in self.track_authority.role_mask_sha256)
        if tuple(sorted(self.role_masks)) != tuple(sorted(authority_roles)):
            raise V15BContractError("mask roles differ from authenticated tracks")
        logical_4d: dict[str, torch.Tensor] = {}
        for role, mask in self.role_masks.items():
            _role(role, label="mask role")
            view = _packed_mask_4d(mask, height=self.height, width=self.width, label=f"mask {role}")
            if int(view.shape[0]) != 1 or not bool(view.any(dim=(2, 3)).all()):
                raise V15BContractError("each source role track must cover every phase")
            areas = [int(view[0, phase].sum()) for phase in range(LATENT_PHASES)]
            maximum_area = max(MIN_TRACK_AREA_PIXELS, int(self.height * self.width * MAX_TRACK_AREA_FRACTION))
            if min(areas) < MIN_TRACK_AREA_PIXELS or max(areas) > maximum_area:
                raise V15BContractError("source role track area is outside preregistered gates")
            if max(areas) / min(areas) > MAX_TRACK_AREA_RATIO:
                raise V15BContractError("source role track area ratio exceeds preregistered gate")
            centroids = []
            for phase in range(LATENT_PHASES):
                if _component_count_4(view[0, phase]) != 1:
                    raise V15BContractError("source role track must be 4-connected in every phase")
                centroids.append(_centroid(view[0, phase]))
            for left, right in zip(centroids, centroids[1:]):
                displacement = math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)
                if displacement > MAX_TRACK_CENTROID_JUMP:
                    raise V15BContractError("source role centroid jump exceeds preregistered gate")
            logical_4d[role] = view
        authority_masks = dict(self.track_authority.role_mask_sha256)
        if any(tensor_sha256(self.role_masks[role]) != authority_masks[role]
               for role in self.role_masks):
            raise V15BContractError("source role mask differs from track authority")
        vessel_roles = authority_roles[1:]
        for left, right in itertools.combinations(authority_roles, 2):
            if bool((logical_4d[left] & logical_4d[right]).any()):
                raise V15BContractError(
                    "human/#1/#2/#3 source instance tracks must be mutually exclusive"
                )
        shape = tuple(next(iter(self.role_masks.values())).shape)
        for label, mask in (
            ("contact", self.contact_mask), ("vessel tube", self.vessel_tube_mask),
            ("transition path", self.transition_path_mask),
            ("corridor", self.editable_corridor_mask),
        ):
            view = _packed_mask_4d(mask, height=self.height, width=self.width, label=label)
            if tuple(mask.shape) != shape:
                raise V15BContractError(f"{label} batch/token geometry differs")
            if bool(view[:, 0].any()):
                raise V15BContractError(f"{label} phase-0 must be exactly empty")
        background_view = _packed_mask_4d(
            self.background_support_mask, height=self.height, width=self.width,
            label="background/support",
        )
        if tuple(self.background_support_mask.shape) != shape or not bool(background_view[:, 0].all()):
            raise V15BContractError("background/support phase-0 must be exactly full")
        expected_tubes = _source_vessel_tubes(logical_4d, vessel_roles)
        expected_transition = _source_transition_path(
            logical_4d[authority_roles[2]], logical_4d[authority_roles[3]]
        )
        expected_tubes[:, 0].zero_(); expected_transition[:, 0].zero_()
        if not torch.equal(self.vessel_tube_mask.reshape_as(expected_tubes), expected_tubes):
            raise V15BContractError("vessel tube is not the deterministic source-only construction")
        if not torch.equal(self.transition_path_mask.reshape_as(expected_transition), expected_transition):
            raise V15BContractError("transition path is not source #2->#3 derived")
        contact_4d = self.contact_mask.reshape_as(expected_tubes)
        contact_4d = _binary_dilate_4d(contact_4d, self.dilation_radius)
        human = _binary_dilate_4d(logical_4d[authority_roles[0]], self.dilation_radius)
        expected_corridor = expected_tubes | expected_transition | contact_4d | human
        expected_corridor[:, 0].zero_()
        corridor_4d = self.editable_corridor_mask.reshape_as(expected_corridor)
        background_4d = self.background_support_mask.reshape_as(expected_corridor)
        if not torch.equal(corridor_4d, expected_corridor):
            raise V15BContractError("editable corridor is not source-only deterministic")
        if not torch.equal(background_4d, ~expected_corridor):
            raise V15BContractError("background/support must be exact corridor complement")
        if not bool(background_4d.any(dim=(2, 3)).all()):
            raise V15BContractError("background/support must be nonempty in every phase")
        spatial_area = self.height * self.width
        for phase in range(1, LATENT_PHASES):
            corridor_fraction = float(corridor_4d[0, phase].sum()) / spatial_area
            background_fraction = float(background_4d[0, phase].sum()) / spatial_area
            if corridor_fraction > MAX_CORRIDOR_FRACTION:
                raise V15BContractError("editable corridor exceeds preregistered per-phase area")
            if background_fraction < MIN_BACKGROUND_FRACTION:
                raise V15BContractError("background falls below preregistered per-phase area")
        if bool((background_4d & expected_transition).any()):
            raise V15BContractError("background restore would erase the new #2->#3 path")
        if object_sha256(self._payload()) != _sha(self.digest, label="mask digest"):
            raise V15BContractError("source mask digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_video_sha256": self.source_video_sha256,
            "binding_digest": self.binding_digest,
            "track_authority_digest": self.track_authority.digest,
            "temporal_phases": self.temporal_phases, "height": self.height,
            "width": self.width, "dilation_radius": self.dilation_radius,
            "roles": {k: tensor_sha256(v) for k, v in sorted(self.role_masks.items())},
            "contact": tensor_sha256(self.contact_mask),
            "vessel_tube": tensor_sha256(self.vessel_tube_mask),
            "transition_path": tensor_sha256(self.transition_path_mask),
            "corridor": tensor_sha256(self.editable_corridor_mask),
            "background": tensor_sha256(self.background_support_mask),
        }

    @classmethod
    def create(
        cls, *, source_video_sha256: str, binding: SourceActionRoleBindingV15B,
        role_masks: Mapping[str, torch.Tensor], contact_mask: torch.Tensor,
        height: int, width: int,
        role_track_receipt_sha256: Mapping[str, str],
    ) -> "SourceRoleMaskSetV15B":
        _revalidate_material(binding, label="mask builder binding")
        if set(role_masks) != set(binding.source_roles):
            raise V15BContractError("mask roles must exactly match all four source-bound roles")
        if set(role_track_receipt_sha256) != set(binding.source_roles):
            raise V15BContractError("track receipts must exactly authenticate four source roles")
        for role in binding.source_roles:
            mask = role_masks[role]
            if (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu" or
                    mask.dtype != torch.bool):
                raise V15BContractError("mask create requires exact CPU bool role masks")
        if (not isinstance(contact_mask, torch.Tensor) or
                contact_mask.device.type != "cpu" or contact_mask.dtype != torch.bool):
            raise V15BContractError("mask create requires an exact CPU bool contact mask")
        logical = {role: role_masks[role].detach().clone()
                   for role in binding.source_roles}
        contact = contact_mask.detach().clone()
        role_mask_authority = tuple((role, tensor_sha256(logical[role]))
                                    for role in binding.source_roles)
        track_payload = {
            "schema_version": TRACK_AUTHORITY_SCHEMA,
            "source_video_sha256": source_video_sha256,
            "binding_digest": binding.digest, "temporal_phases": LATENT_PHASES,
            "height": height, "width": width,
            "role_track_receipt_sha256": tuple(
                (role, role_track_receipt_sha256[role]) for role in binding.source_roles
            ),
            "role_mask_sha256": role_mask_authority,
        }
        authority = SourceRoleTrackAuthorityV15B(
            **track_payload, digest=object_sha256(track_payload)
        )
        logical_4d = {
            role: _packed_mask_4d(mask, height=height, width=width, label=f"mask {role}")
            for role, mask in logical.items()
        }
        vessel_roles = binding.source_roles[1:]
        tubes = _source_vessel_tubes(logical_4d, vessel_roles)
        transition = _source_transition_path(
            logical_4d[binding.moving_object_source_role],
            logical_4d[binding.recipient_source_role],
        )
        tubes[:, 0].zero_(); transition[:, 0].zero_()
        contact_4d = _packed_mask_4d(contact, height=height, width=width, label="contact")
        contact_4d[:, 0].zero_()
        human = _binary_dilate_4d(
            logical_4d[binding.human_agent_source_role], CORRIDOR_DILATION_RADIUS
        )
        corridor = tubes | transition | _binary_dilate_4d(
            contact_4d, CORRIDOR_DILATION_RADIUS
        ) | human
        corridor[:, 0].zero_()
        flat = lambda value: value.reshape(int(value.shape[0]), -1)
        payload = {
            "schema_version": MASK_SCHEMA, "source_video_sha256": source_video_sha256,
            "binding_digest": binding.digest, "track_authority_digest": authority.digest,
            "temporal_phases": LATENT_PHASES, "height": height, "width": width,
            "dilation_radius": CORRIDOR_DILATION_RADIUS,
            "roles": {k: tensor_sha256(v) for k, v in sorted(logical.items())},
            "contact": tensor_sha256(flat(contact_4d)),
            "vessel_tube": tensor_sha256(flat(tubes)),
            "transition_path": tensor_sha256(flat(transition)),
            "corridor": tensor_sha256(flat(corridor)),
            "background": tensor_sha256(flat(~corridor)),
        }
        return cls(
            MASK_SCHEMA, source_video_sha256, binding.digest, authority,
            LATENT_PHASES, height, width, CORRIDOR_DILATION_RADIUS, logical,
            flat(contact_4d), flat(tubes), flat(transition), flat(corridor),
            flat(~corridor), object_sha256(payload),
        )

def _position_projector(
    value: torch.Tensor, *, heads: int, head_dim: int, label: str,
) -> torch.Tensor:
    projector = _cpu_fp32(value, label=label, ndim=3)
    if tuple(projector.shape) != (heads, head_dim, head_dim):
        raise V15BContractError(f"{label} head geometry differs")
    transpose_error = float((projector - projector.transpose(-1, -2)).abs().max())
    idempotence_error = float((projector @ projector - projector).abs().max())
    if (transpose_error > POSITION_PROJECTOR_TOLERANCE or
            idempotence_error > POSITION_PROJECTOR_TOLERANCE):
        raise V15BContractError(f"{label} must be a symmetric idempotent projector")
    ranks = []
    for head in range(heads):
        rank = int(torch.linalg.matrix_rank(
            projector[head], tol=POSITION_PROJECTOR_TOLERANCE
        ))
        if rank < 1 or rank >= head_dim:
            raise V15BContractError(f"{label} rank must lie in [1, head_dim-1]")
        ranks.append(rank)
    return projector


def _scrub_position_subspace(
    key: torch.Tensor, projector: torch.Tensor,
) -> torch.Tensor:
    if (key.ndim != 4 or projector.ndim != 3 or
            tuple(key.shape[2:]) != tuple(projector.shape[:1]) +
            tuple(projector.shape[1:2])):
        raise V15BContractError("position scrub K/projector geometry differs")
    return key - torch.einsum("blhd,hde->blhe", key, projector)


def _fixed_position_encoder_checkpoint() -> torch.Tensor:
    """Exact coefficients of the synthetic calibration-only encoder."""
    checkpoint = torch.zeros(6, POSITION_CALIBRATION_HEAD_DIM, dtype=torch.float32)
    checkpoint[0, 0] = 1.0
    checkpoint[1, 1] = 1.0
    checkpoint[0, 2] = 0.5; checkpoint[1, 2] = 0.25
    checkpoint[0, 3] = -0.25; checkpoint[1, 3] = 0.5
    checkpoint[2, 4] = 1.0
    checkpoint[3, 5] = 0.125
    checkpoint[4, 6] = 1.0
    checkpoint[5, 7] = 1.0
    return checkpoint


def _fixed_position_base_latent() -> torch.Tensor:
    latent = torch.zeros(
        POSITION_CALIBRATION_BATCH, POSITION_CALIBRATION_CHANNELS,
        POSITION_CALIBRATION_HEIGHT, POSITION_CALIBRATION_WIDTH,
        dtype=torch.float32,
    )
    value = 1.0
    for y in range(2, 5):
        for x in range(2, 5):
            latent[0, 0, y, x] = value
            latent[0, 1, y, x] = float((int(value) * 3) % 7 + 1)
            value += 1.0
    return latent


def _translate_grid_zero_fill(value: torch.Tensor, translation_yx: tuple[int, int]) -> torch.Tensor:
    tensor = _cpu_fp32(value, label="fixed calibration input latent", ndim=4)
    if tuple(tensor.shape) != (
        POSITION_CALIBRATION_BATCH, POSITION_CALIBRATION_CHANNELS,
        POSITION_CALIBRATION_HEIGHT, POSITION_CALIBRATION_WIDTH,
    ):
        raise V15BContractError("fixed calibration latent geometry differs")
    if (not isinstance(translation_yx, tuple) or len(translation_yx) != 2 or
            any(isinstance(item, bool) or not isinstance(item, int)
                for item in translation_yx) or translation_yx == (0, 0)):
        raise V15BContractError("fixed calibration translation must be nonzero integer y/x")
    dy, dx = translation_yx
    output = torch.zeros_like(tensor)
    for y in range(POSITION_CALIBRATION_HEIGHT):
        for x in range(POSITION_CALIBRATION_WIDTH):
            target_y, target_x = y + dy, x + dx
            if (0 <= target_y < POSITION_CALIBRATION_HEIGHT and
                    0 <= target_x < POSITION_CALIBRATION_WIDTH):
                output[:, :, target_y, target_x] = tensor[:, :, y, x]
    return output


def _fixed_position_encoder(value: torch.Tensor) -> torch.Tensor:
    """Frozen synthetic map used only to exercise the reference ABI."""
    latent = _cpu_fp32(value, label="fixed calibration encoder input", ndim=4)
    if tuple(latent.shape) != (
        POSITION_CALIBRATION_BATCH, POSITION_CALIBRATION_CHANNELS,
        POSITION_CALIBRATION_HEIGHT, POSITION_CALIBRATION_WIDTH,
    ):
        raise V15BContractError("fixed calibration encoder geometry differs")
    content = latent.permute(0, 2, 3, 1)
    support = (content.abs().sum(-1, keepdim=True) > 0).float()
    bias = torch.ones_like(support)
    y = torch.arange(POSITION_CALIBRATION_HEIGHT, dtype=torch.float32).reshape(
        1, POSITION_CALIBRATION_HEIGHT, 1, 1
    ).expand(1, POSITION_CALIBRATION_HEIGHT, POSITION_CALIBRATION_WIDTH, 1)
    x = torch.arange(POSITION_CALIBRATION_WIDTH, dtype=torch.float32).reshape(
        1, 1, POSITION_CALIBRATION_WIDTH, 1
    ).expand(1, POSITION_CALIBRATION_HEIGHT, POSITION_CALIBRATION_WIDTH, 1)
    features = torch.cat((content, support, bias, y, x), dim=-1)
    encoded = features @ _fixed_position_encoder_checkpoint()
    return encoded.reshape(
        POSITION_CALIBRATION_BATCH,
        POSITION_CALIBRATION_HEIGHT * POSITION_CALIBRATION_WIDTH,
        POSITION_CALIBRATION_HEADS, POSITION_CALIBRATION_HEAD_DIM,
    )


def _translation_correspondence(
    base_latent: torch.Tensor, translation_yx: tuple[int, int],
) -> torch.Tensor:
    support = base_latent.abs().sum(1)[0] > 0
    dy, dx = translation_yx
    pairs = []
    for y, x in torch.nonzero(support, as_tuple=False).tolist():
        target_y, target_x = y + dy, x + dx
        if not (0 <= target_y < POSITION_CALIBRATION_HEIGHT and
                0 <= target_x < POSITION_CALIBRATION_WIDTH):
            raise V15BContractError("fixed calibration support exits translated fixture")
        pairs.append((
            y * POSITION_CALIBRATION_WIDTH + x,
            target_y * POSITION_CALIBRATION_WIDTH + target_x,
        ))
    return torch.tensor(pairs, dtype=torch.int64)


def _position_calibration_spec_material() -> tuple[str, str, torch.Tensor]:
    code = canonical_json_bytes({
        "algorithm": "fixed-linear-pre-rope-calibration-encoder",
        "features": ("content0", "content1", "support", "bias", "y", "x"),
        "translation": "integer-yx-zero-fill",
        "correspondence": "nonzero-base-support-forward-index",
        "abi_revision": 6,
    }).decode("utf-8")
    config = canonical_json_bytes({
        "batch": POSITION_CALIBRATION_BATCH,
        "channels": POSITION_CALIBRATION_CHANNELS,
        "height": POSITION_CALIBRATION_HEIGHT,
        "width": POSITION_CALIBRATION_WIDTH,
        "heads": POSITION_CALIBRATION_HEADS,
        "head_dim": POSITION_CALIBRATION_HEAD_DIM,
        "fit_translations_yx": POSITION_FIT_TRANSLATIONS_YX,
        "heldout_translations_yx": POSITION_HELDOUT_TRANSLATIONS_YX,
    }).decode("utf-8")
    return code, config, _fixed_position_encoder_checkpoint()


def _position_calibration_spec_digests() -> tuple[str, str, str]:
    code, config, checkpoint = _position_calibration_spec_material()
    return (
        hashlib.sha256(code.encode("utf-8")).hexdigest(),
        hashlib.sha256(config.encode("utf-8")).hexdigest(),
        tensor_sha256(checkpoint),
    )


def _fixed_position_fixture_material() -> dict[str, Any]:
    base = _fixed_position_base_latent()
    base_key = _fixed_position_encoder(base)
    code_material, config_material, checkpoint_material = (
        _position_calibration_spec_material()
    )
    code_sha, config_sha, checkpoint_sha = _position_calibration_spec_digests()
    fit_inputs = tuple(
        _translate_grid_zero_fill(base, translation)
        for translation in POSITION_FIT_TRANSLATIONS_YX
    )
    heldout_inputs = tuple(
        _translate_grid_zero_fill(base, translation)
        for translation in POSITION_HELDOUT_TRANSLATIONS_YX
    )
    fit_keys = tuple(_fixed_position_encoder(value) for value in fit_inputs)
    heldout_keys = tuple(_fixed_position_encoder(value) for value in heldout_inputs)
    fit_correspondence = tuple(
        _translation_correspondence(base, translation)
        for translation in POSITION_FIT_TRANSLATIONS_YX
    )
    heldout_correspondence = tuple(
        _translation_correspondence(base, translation)
        for translation in POSITION_HELDOUT_TRANSLATIONS_YX
    )
    transcript = object_sha256({
        "base_input_latent_sha256": tensor_sha256(base),
        "base_pre_rope_key_sha256": tensor_sha256(base_key),
        "frozen_model_code_sha256": code_sha,
        "frozen_model_config_sha256": config_sha,
        "frozen_model_checkpoint_sha256": checkpoint_sha,
        "fit": tuple({
            "translation_yx": translation,
            "translated_input_latent_sha256": tensor_sha256(latent),
            "translated_pre_rope_key_sha256": tensor_sha256(key),
            "correspondence_sha256": tensor_sha256(correspondence),
        } for translation, latent, key, correspondence in zip(
            POSITION_FIT_TRANSLATIONS_YX, fit_inputs, fit_keys, fit_correspondence
        )),
        "heldout": tuple({
            "translation_yx": translation,
            "translated_input_latent_sha256": tensor_sha256(latent),
            "translated_pre_rope_key_sha256": tensor_sha256(key),
            "correspondence_sha256": tensor_sha256(correspondence),
        } for translation, latent, key, correspondence in zip(
            POSITION_HELDOUT_TRANSLATIONS_YX, heldout_inputs, heldout_keys,
            heldout_correspondence,
        )),
    })
    return {
        "base_input_latent": base,
        "base_pre_rope_key": base_key,
        "fit_translated_input_latents": fit_inputs,
        "heldout_translated_input_latents": heldout_inputs,
        "fit_counterfactual_pre_rope_keys": fit_keys,
        "heldout_counterfactual_pre_rope_keys": heldout_keys,
        "fit_correspondence_token_indices": fit_correspondence,
        "heldout_correspondence_token_indices": heldout_correspondence,
        "frozen_model_code_material": code_material,
        "frozen_model_config_material": config_material,
        "frozen_model_checkpoint_material": checkpoint_material,
        "frozen_model_code_sha256": code_sha,
        "frozen_model_config_sha256": config_sha,
        "frozen_model_checkpoint_sha256": checkpoint_sha,
        "translation_transform_transcript_sha256": transcript,
    }


@dataclass(frozen=True)
class PositionCalibrationFixtureV15B:
    """One exact synthetic fixture; no source/video-phase K is accepted."""

    schema_version: str
    base_input_latent: torch.Tensor
    base_pre_rope_key: torch.Tensor
    fit_translations_yx: tuple[tuple[int, int], ...]
    heldout_translations_yx: tuple[tuple[int, int], ...]
    fit_translated_input_latents: tuple[torch.Tensor, ...]
    heldout_translated_input_latents: tuple[torch.Tensor, ...]
    fit_counterfactual_pre_rope_keys: tuple[torch.Tensor, ...]
    heldout_counterfactual_pre_rope_keys: tuple[torch.Tensor, ...]
    fit_correspondence_token_indices: tuple[torch.Tensor, ...]
    heldout_correspondence_token_indices: tuple[torch.Tensor, ...]
    frozen_model_code_material: str
    frozen_model_config_material: str
    frozen_model_checkpoint_material: torch.Tensor
    frozen_model_code_sha256: str
    frozen_model_config_sha256: str
    frozen_model_checkpoint_sha256: str
    translation_transform_transcript_sha256: str
    source_or_video_phase_key_accepted: bool
    material_bytes_reopenable: bool
    translation_correspondence_recomputed: bool
    externally_authenticated: bool
    position_removed_claimed: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != POSITION_CALIBRATION_FIXTURE_SCHEMA:
            raise V15BContractError("position calibration fixture schema differs")
        expected = _fixed_position_fixture_material()
        if (self.fit_translations_yx != POSITION_FIT_TRANSLATIONS_YX or
                self.heldout_translations_yx != POSITION_HELDOUT_TRANSLATIONS_YX):
            raise V15BContractError("position calibration translation split differs")
        tensor_fields = ("base_input_latent", "base_pre_rope_key")
        tuple_tensor_fields = (
            "fit_translated_input_latents", "heldout_translated_input_latents",
            "fit_counterfactual_pre_rope_keys", "heldout_counterfactual_pre_rope_keys",
            "fit_correspondence_token_indices", "heldout_correspondence_token_indices",
        )
        for name in tensor_fields:
            supplied = getattr(self, name)
            if (not isinstance(supplied, torch.Tensor) or
                    tensor_sha256(supplied) != tensor_sha256(expected[name]) or
                    not torch.equal(supplied, expected[name])):
                raise V15BContractError(f"fixed position calibration {name} differs")
        for name in tuple_tensor_fields:
            supplied = getattr(self, name)
            wanted = expected[name]
            if (not isinstance(supplied, tuple) or len(supplied) != len(wanted) or
                    any(not isinstance(left, torch.Tensor) or
                        tensor_sha256(left) != tensor_sha256(right) or
                        not torch.equal(left, right)
                        for left, right in zip(supplied, wanted))):
                raise V15BContractError(f"fixed position calibration {name} differs")
        for name in ("frozen_model_code_material", "frozen_model_config_material"):
            if not isinstance(getattr(self, name), str) or getattr(self, name) != expected[name]:
                raise V15BContractError(f"fixed position calibration {name} differs")
        if (not isinstance(self.frozen_model_checkpoint_material, torch.Tensor) or
                not torch.equal(
                    self.frozen_model_checkpoint_material,
                    expected["frozen_model_checkpoint_material"],
                )):
            raise V15BContractError(
                "fixed position calibration frozen_model_checkpoint_material differs"
            )
        for name in (
            "frozen_model_code_sha256", "frozen_model_config_sha256",
            "frozen_model_checkpoint_sha256", "translation_transform_transcript_sha256",
        ):
            if _sha(getattr(self, name), label=name) != expected[name]:
                raise V15BContractError(f"fixed position calibration {name} differs")
        if (self.source_or_video_phase_key_accepted is not False or
                self.material_bytes_reopenable is not True or
                self.translation_correspondence_recomputed is not True or
                self.externally_authenticated is not False or
                self.position_removed_claimed is not False or
                self.scientific_claim_authorized is not False or
                self.route_authorized is not False):
            raise V15BContractError("fixed calibration must remain synthetic/no-authority")
        if object_sha256(self._payload()) != _sha(self.digest, label="position fixture"):
            raise V15BContractError("position calibration fixture digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_input_latent_sha256": tensor_sha256(self.base_input_latent),
            "base_pre_rope_key_sha256": tensor_sha256(self.base_pre_rope_key),
            "fit_translations_yx": self.fit_translations_yx,
            "heldout_translations_yx": self.heldout_translations_yx,
            "fit_translated_input_latent_sha256": tuple(
                tensor_sha256(value) for value in self.fit_translated_input_latents
            ),
            "heldout_translated_input_latent_sha256": tuple(
                tensor_sha256(value) for value in self.heldout_translated_input_latents
            ),
            "fit_counterfactual_pre_rope_key_sha256": tuple(
                tensor_sha256(value) for value in self.fit_counterfactual_pre_rope_keys
            ),
            "heldout_counterfactual_pre_rope_key_sha256": tuple(
                tensor_sha256(value) for value in self.heldout_counterfactual_pre_rope_keys
            ),
            "fit_correspondence_sha256": tuple(
                tensor_sha256(value) for value in self.fit_correspondence_token_indices
            ),
            "heldout_correspondence_sha256": tuple(
                tensor_sha256(value) for value in self.heldout_correspondence_token_indices
            ),
            "frozen_model_code_material_sha256": hashlib.sha256(
                self.frozen_model_code_material.encode("utf-8")
            ).hexdigest(),
            "frozen_model_config_material_sha256": hashlib.sha256(
                self.frozen_model_config_material.encode("utf-8")
            ).hexdigest(),
            "frozen_model_checkpoint_material_sha256": tensor_sha256(
                self.frozen_model_checkpoint_material
            ),
            "frozen_model_code_sha256": self.frozen_model_code_sha256,
            "frozen_model_config_sha256": self.frozen_model_config_sha256,
            "frozen_model_checkpoint_sha256": self.frozen_model_checkpoint_sha256,
            "translation_transform_transcript_sha256": (
                self.translation_transform_transcript_sha256
            ),
            "source_or_video_phase_key_accepted": self.source_or_video_phase_key_accepted,
            "material_bytes_reopenable": self.material_bytes_reopenable,
            "translation_correspondence_recomputed": (
                self.translation_correspondence_recomputed
            ),
            "externally_authenticated": self.externally_authenticated,
            "position_removed_claimed": self.position_removed_claimed,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "route_authorized": self.route_authorized,
        }


def build_position_calibration_fixture_v15b() -> PositionCalibrationFixtureV15B:
    material = _fixed_position_fixture_material()
    payload = {
        "schema_version": POSITION_CALIBRATION_FIXTURE_SCHEMA,
        "base_input_latent_sha256": tensor_sha256(material["base_input_latent"]),
        "base_pre_rope_key_sha256": tensor_sha256(material["base_pre_rope_key"]),
        "fit_translations_yx": POSITION_FIT_TRANSLATIONS_YX,
        "heldout_translations_yx": POSITION_HELDOUT_TRANSLATIONS_YX,
        "fit_translated_input_latent_sha256": tuple(
            tensor_sha256(value) for value in material["fit_translated_input_latents"]
        ),
        "heldout_translated_input_latent_sha256": tuple(
            tensor_sha256(value) for value in material["heldout_translated_input_latents"]
        ),
        "fit_counterfactual_pre_rope_key_sha256": tuple(
            tensor_sha256(value) for value in material["fit_counterfactual_pre_rope_keys"]
        ),
        "heldout_counterfactual_pre_rope_key_sha256": tuple(
            tensor_sha256(value) for value in material["heldout_counterfactual_pre_rope_keys"]
        ),
        "fit_correspondence_sha256": tuple(
            tensor_sha256(value) for value in material["fit_correspondence_token_indices"]
        ),
        "heldout_correspondence_sha256": tuple(
            tensor_sha256(value) for value in material["heldout_correspondence_token_indices"]
        ),
        "frozen_model_code_material_sha256": hashlib.sha256(
            material["frozen_model_code_material"].encode("utf-8")
        ).hexdigest(),
        "frozen_model_config_material_sha256": hashlib.sha256(
            material["frozen_model_config_material"].encode("utf-8")
        ).hexdigest(),
        "frozen_model_checkpoint_material_sha256": tensor_sha256(
            material["frozen_model_checkpoint_material"]
        ),
        "frozen_model_code_sha256": material["frozen_model_code_sha256"],
        "frozen_model_config_sha256": material["frozen_model_config_sha256"],
        "frozen_model_checkpoint_sha256": material["frozen_model_checkpoint_sha256"],
        "translation_transform_transcript_sha256": (
            material["translation_transform_transcript_sha256"]
        ),
        "source_or_video_phase_key_accepted": False,
        "material_bytes_reopenable": True,
        "translation_correspondence_recomputed": True,
        "externally_authenticated": False,
        "position_removed_claimed": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
    }
    return PositionCalibrationFixtureV15B(
        POSITION_CALIBRATION_FIXTURE_SCHEMA,
        material["base_input_latent"], material["base_pre_rope_key"],
        POSITION_FIT_TRANSLATIONS_YX, POSITION_HELDOUT_TRANSLATIONS_YX,
        material["fit_translated_input_latents"],
        material["heldout_translated_input_latents"],
        material["fit_counterfactual_pre_rope_keys"],
        material["heldout_counterfactual_pre_rope_keys"],
        material["fit_correspondence_token_indices"],
        material["heldout_correspondence_token_indices"],
        material["frozen_model_code_material"],
        material["frozen_model_config_material"],
        material["frozen_model_checkpoint_material"],
        material["frozen_model_code_sha256"], material["frozen_model_config_sha256"],
        material["frozen_model_checkpoint_sha256"],
        material["translation_transform_transcript_sha256"],
        False, True, True, False, False, False, False, object_sha256(payload),
    )


def _axis_projector(matrix: torch.Tensor, *, label: str) -> tuple[torch.Tensor, int, float]:
    if matrix.ndim != 3 or int(matrix.shape[0]) < 1:
        raise V15BContractError(f"{label} calibration span is empty")
    heads, head_dim = int(matrix.shape[1]), int(matrix.shape[2])
    output = torch.zeros(heads, head_dim, head_dim, dtype=torch.float32)
    maximum_residual = 0.0
    ranks = []
    for head in range(heads):
        rows = matrix[:, head].double()
        _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
        maximum = float(singular.max()) if int(singular.numel()) else 0.0
        threshold = max(POSITION_PROJECTOR_TOLERANCE,
                        maximum * POSITION_SVD_RELATIVE_TOLERANCE)
        rank = int((singular > threshold).sum())
        if rank < 1 or rank >= head_dim:
            raise V15BContractError(f"{label} calibration rank must be proper/nonzero")
        basis = vh[:rank]
        output[head] = (basis.transpose(0, 1) @ basis).float()
        residual = rows - rows @ output[head].double()
        maximum_residual = max(maximum_residual, float(residual.abs().max()))
        ranks.append(rank)
    if len(set(ranks)) != 1:
        raise V15BContractError(f"{label} calibration rank differs by head")
    return output, ranks[0], maximum_residual


def _matched_translation_deltas(
    base_key: torch.Tensor, translated_keys: Sequence[torch.Tensor],
    correspondences: Sequence[torch.Tensor], translations: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_rows = []; x_rows = []; all_rows = []
    for translation, current, correspondence in zip(
            translations, translated_keys, correspondences):
        pairs = _tensor(correspondence, label="position correspondence", ndim=2,
                        floating=False)
        if (pairs.device.type != "cpu" or pairs.dtype != torch.int64 or
                int(pairs.shape[1]) != 2):
            raise V15BContractError("position correspondence must be CPU int64 [N,2]")
        source_index, target_index = pairs[:, 0], pairs[:, 1]
        delta = current[0, target_index] - base_key[0, source_index]
        all_rows.append(delta)
        if translation[0] != 0 and translation[1] == 0:
            y_rows.append(delta)
        if translation[1] != 0 and translation[0] == 0:
            x_rows.append(delta)
    if not y_rows or not x_rows:
        raise V15BContractError("position fit lacks independent pure-y/pure-x probes")
    return torch.cat(y_rows), torch.cat(x_rows), torch.cat(all_rows)


def _derive_position_projector(
    fixture: PositionCalibrationFixtureV15B,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], tuple[int, ...], float, float]:
    """Fit on pure-axis probes; validate only on disjoint held-out probes."""
    _revalidate_material(fixture, label="position projector fixture")
    base = fixture.base_pre_rope_key
    y_rows, x_rows, all_rows = _matched_translation_deltas(
        base, fixture.fit_counterfactual_pre_rope_keys,
        fixture.fit_correspondence_token_indices, fixture.fit_translations_yx,
    )
    y_projector, y_rank, y_residual = _axis_projector(y_rows, label="pure-y")
    x_projector, x_rank, x_residual = _axis_projector(x_rows, label="pure-x")
    cross = max(float((y_projector @ x_projector).abs().max()),
                float((x_projector @ y_projector).abs().max()))
    if cross > POSITION_PROJECTOR_TOLERANCE:
        raise V15BContractError("position y/x calibration spans are not independent")
    projector = y_projector + x_projector
    projector = _position_projector(
        projector, heads=POSITION_CALIBRATION_HEADS,
        head_dim=POSITION_CALIBRATION_HEAD_DIM,
        label="fixed-fixture position projector",
    )
    ranks = tuple(int(torch.linalg.matrix_rank(
        projector[head], tol=POSITION_PROJECTOR_TOLERANCE
    )) for head in range(POSITION_CALIBRATION_HEADS))
    fit_residual = max(
        y_residual, x_residual,
        float((all_rows - torch.einsum("nhd,hde->nhe", all_rows, projector)).abs().max()),
    )
    heldout_residual = 0.0
    scrubbed_base = _scrub_position_subspace(base, projector)
    for current, correspondence in zip(
            fixture.heldout_counterfactual_pre_rope_keys,
            fixture.heldout_correspondence_token_indices):
        scrubbed_current = _scrub_position_subspace(current, projector)
        source_index, target_index = correspondence[:, 0], correspondence[:, 1]
        # This is an independently re-forwarded, held-out correspondence check;
        # it is not the algebraic identity scrub(base + delta) == scrub(current).
        heldout_residual = max(heldout_residual, _max_abs(
            scrubbed_base[0, source_index], scrubbed_current[0, target_index]
        ))
    return (
        projector, tuple(y_rank for _ in range(POSITION_CALIBRATION_HEADS)),
        tuple(x_rank for _ in range(POSITION_CALIBRATION_HEADS)), ranks,
        fit_residual, heldout_residual,
    )


@dataclass(frozen=True)
class PositionCounterfactualReferenceV15B:
    """Synthetic fixed-fixture projector; never a source position-removal claim."""

    schema_version: str
    calibration_fixture: PositionCalibrationFixtureV15B
    projector: torch.Tensor
    fit_y_rank_by_head: tuple[int, ...]
    fit_x_rank_by_head: tuple[int, ...]
    rank_by_head: tuple[int, ...]
    fit_reconstruction_residual_max_abs: float
    heldout_correspondence_residual_max_abs: float
    fit_translation_count: int
    heldout_translation_count: int
    y_x_span_independent: bool
    material_digest_authenticated: bool
    externally_authenticated: bool
    position_removed_claimed: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != POSITION_REFERENCE_SCHEMA:
            raise V15BContractError("position counterfactual reference schema differs")
        if not isinstance(self.calibration_fixture, PositionCalibrationFixtureV15B):
            raise V15BContractError("position reference lacks fixed calibration fixture")
        _revalidate_material(self.calibration_fixture, label="position calibration fixture")
        derived = _derive_position_projector(self.calibration_fixture)
        supplied = _position_projector(
            self.projector, heads=POSITION_CALIBRATION_HEADS,
            head_dim=POSITION_CALIBRATION_HEAD_DIM,
            label="position reference projector",
        )
        if tensor_sha256(supplied) != tensor_sha256(derived[0]):
            raise V15BContractError("position projector is not fixed-fixture replay")
        if (self.fit_y_rank_by_head != derived[1] or
                self.fit_x_rank_by_head != derived[2] or self.rank_by_head != derived[3]):
            raise V15BContractError("position reference independent-axis ranks differ")
        for label, observed, expected in (
            ("fit reconstruction", self.fit_reconstruction_residual_max_abs, derived[4]),
            ("heldout correspondence", self.heldout_correspondence_residual_max_abs,
             derived[5]),
        ):
            if abs(_finite(observed, label=label) - expected) > POSITION_PROJECTOR_TOLERANCE:
                raise V15BContractError(f"position reference {label} replay differs")
            if expected > POSITION_PROJECTOR_TOLERANCE:
                raise V15BContractError(f"position reference {label} gate failed")
        if (self.fit_translation_count != len(POSITION_FIT_TRANSLATIONS_YX) or
                self.heldout_translation_count != len(POSITION_HELDOUT_TRANSLATIONS_YX)):
            raise V15BContractError("position reference train/heldout split count differs")
        if (self.y_x_span_independent is not True or
                self.material_digest_authenticated is not True or
                self.externally_authenticated is not False or
                self.position_removed_claimed is not False or
                self.scientific_claim_authorized is not False or
                self.route_authorized is not False):
            raise V15BContractError("position reference must remain synthetic/no-authority")
        if object_sha256(self._payload()) != _sha(self.digest, label="position reference"):
            raise V15BContractError("position counterfactual reference digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_fixture_digest": self.calibration_fixture.digest,
            "projector_sha256": tensor_sha256(self.projector),
            "fit_y_rank_by_head": self.fit_y_rank_by_head,
            "fit_x_rank_by_head": self.fit_x_rank_by_head,
            "rank_by_head": self.rank_by_head,
            "fit_reconstruction_residual_max_abs": self.fit_reconstruction_residual_max_abs,
            "heldout_correspondence_residual_max_abs": (
                self.heldout_correspondence_residual_max_abs
            ),
            "fit_translation_count": self.fit_translation_count,
            "heldout_translation_count": self.heldout_translation_count,
            "y_x_span_independent": self.y_x_span_independent,
            "material_digest_authenticated": self.material_digest_authenticated,
            "externally_authenticated": self.externally_authenticated,
            "position_removed_claimed": self.position_removed_claimed,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "route_authorized": self.route_authorized,
        }


def build_position_counterfactual_reference_v15b(
    *, calibration_fixture: Optional[PositionCalibrationFixtureV15B] = None,
) -> PositionCounterfactualReferenceV15B:
    fixture = (build_position_calibration_fixture_v15b()
               if calibration_fixture is None else calibration_fixture)
    if not isinstance(fixture, PositionCalibrationFixtureV15B):
        raise V15BContractError("position reference accepts only the sealed fixed fixture")
    _revalidate_material(fixture, label="position reference builder fixture")
    projector, y_rank, x_rank, ranks, fit_residual, heldout_residual = (
        _derive_position_projector(fixture)
    )
    payload = {
        "schema_version": POSITION_REFERENCE_SCHEMA,
        "calibration_fixture_digest": fixture.digest,
        "projector_sha256": tensor_sha256(projector),
        "fit_y_rank_by_head": y_rank, "fit_x_rank_by_head": x_rank,
        "rank_by_head": ranks,
        "fit_reconstruction_residual_max_abs": fit_residual,
        "heldout_correspondence_residual_max_abs": heldout_residual,
        "fit_translation_count": len(POSITION_FIT_TRANSLATIONS_YX),
        "heldout_translation_count": len(POSITION_HELDOUT_TRANSLATIONS_YX),
        "y_x_span_independent": True,
        "material_digest_authenticated": True,
        "externally_authenticated": False,
        "position_removed_claimed": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
    }
    return PositionCounterfactualReferenceV15B(
        POSITION_REFERENCE_SCHEMA, fixture, projector, y_rank, x_rank, ranks,
        fit_residual, heldout_residual, len(POSITION_FIT_TRANSLATIONS_YX),
        len(POSITION_HELDOUT_TRANSLATIONS_YX), True, True, False, False,
        False, False, object_sha256(payload),
    )


def _canonical_source_extraction_config_material() -> str:
    """Exact r8 phase-0 extraction algorithm, stored as reopenable text."""
    return canonical_json_bytes({
        "schema_version": CANONICAL_EXTRACTION_CONFIG_SCHEMA,
        "source_hidden_domain": "raw_pre_block_hidden",
        "source_key_domain": "raw_pre_rope",
        "source_value_domain": "raw_pre_attention_value",
        "source_phase": 0,
        "phase0_identity_record": "full_source_hidden_pre_rope_key_value",
        "mask_domain": "full_source_role_tracks_select_phase0",
        "position_scrub": "right_project_out_fixed_calibration_subspace",
        "slot_record": (
            "role", "batch_index", "phase0_spatial_token_index",
            "raw_key_sha256", "raw_value_sha256", "raw_role_mask_sha256",
        ),
        "slot_uuid": "sha256(raw_material_digest+slot_record)",
        "pair_sort": "paired_content_sha256_then_slot_uuid",
        "padding": "right_zero_to_max_role_count",
        "null_sort": "scrubbed_key_sha256_then_raw_index",
        "abi_revision": 8,
    }).decode("utf-8")


def _immutable_tensor_hex(value: torch.Tensor, *, label: str) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise V15BContractError(f"{label} must be a material CPU tensor")
    if value.dtype not in (torch.float32, torch.bool):
        raise V15BContractError(f"{label} has an unsupported immutable dtype")
    if value.dtype == torch.float32 and not bool(torch.isfinite(value).all()):
        raise V15BContractError(f"{label} is non-finite")
    return bytes(
        value.detach().contiguous().view(torch.uint8).reshape(-1).tolist()
    ).hex()


def _reopen_tensor_hex(
    value: str, *, shape: tuple[int, ...], dtype: torch.dtype, label: str,
) -> torch.Tensor:
    if (not isinstance(value, str) or len(value) % 2 or
            re.fullmatch(r"[0-9a-f]*", value) is None):
        raise V15BContractError(f"{label} is not canonical lowercase tensor bytes")
    if (not isinstance(shape, tuple) or not shape or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in shape
    )):
        raise V15BContractError(f"{label} shape is invalid")
    if dtype not in (torch.float32, torch.bool):
        raise V15BContractError(f"{label} reopen dtype is unsupported")
    item_size = 4 if dtype == torch.float32 else 1
    expected_bytes = math.prod(shape) * item_size
    if len(value) != expected_bytes * 2:
        raise V15BContractError(f"{label} byte length differs from geometry")
    raw = bytes.fromhex(value)
    # A new uint8 storage is allocated on every call.  The immutable string is
    # the authority; no caller-owned tensor storage crosses this boundary.
    byte_tensor = torch.tensor(tuple(raw), dtype=torch.uint8)
    result = byte_tensor.view(dtype).reshape(shape).clone()
    if dtype == torch.float32 and not bool(torch.isfinite(result).all()):
        raise V15BContractError(f"{label} reopened non-finite data")
    return result


def _raw_source_material_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": value["schema_version"],
        "source_video_sha256": value["source_video_sha256"],
        "source_latent_sha256": value["source_latent_sha256"],
        "binding_digest": value["binding_digest"],
        "mask_digest": value["mask_digest"],
        "track_authority_digest": value["track_authority_digest"],
        "step_index": value["step_index"], "block_index": value["block_index"],
        "branch": value["branch"], "temporal_phases": value["temporal_phases"],
        "batch_size": value["batch_size"], "height": value["height"],
        "width": value["width"], "heads": value["heads"],
        "head_dim": value["head_dim"], "hidden_width": value["hidden_width"],
        "raw_phase0_hidden_byte_sha256": hashlib.sha256(
            bytes.fromhex(value["raw_phase0_hidden_hex"])
        ).hexdigest(),
        "raw_phase0_pre_rope_key_byte_sha256": hashlib.sha256(
            bytes.fromhex(value["raw_phase0_pre_rope_key_hex"])
        ).hexdigest(),
        "raw_phase0_value_byte_sha256": hashlib.sha256(
            bytes.fromhex(value["raw_phase0_value_hex"])
        ).hexdigest(),
        "raw_phase0_hidden_sha256": value["raw_phase0_hidden_sha256"],
        "raw_phase0_pre_rope_key_sha256": value[
            "raw_phase0_pre_rope_key_sha256"
        ],
        "raw_phase0_value_sha256": value["raw_phase0_value_sha256"],
        "source_role_mask_byte_sha256_by_role": tuple(
            (role, hashlib.sha256(bytes.fromhex(raw)).hexdigest())
            for role, raw in value["source_role_mask_hex_by_role"]
        ),
        "source_role_mask_sha256_by_role": value[
            "source_role_mask_sha256_by_role"
        ],
        "canonical_extraction_config_material_sha256": hashlib.sha256(
            value["canonical_extraction_config_material"].encode("utf-8")
        ).hexdigest(),
        "canonical_extraction_config_sha256": value[
            "canonical_extraction_config_sha256"
        ],
        "authority_kind": value["authority_kind"],
        "immutable_byte_material": value["immutable_byte_material"],
        "material_reopenable": value["material_reopenable"],
        "externally_authenticated": value["externally_authenticated"],
        "scientific_claim_authorized": value["scientific_claim_authorized"],
        "route_authorized": value["route_authorized"],
    }


@dataclass(frozen=True)
class SourcePhase0RawMaterialV15B:
    """Immutable, reopenable caller-supplied raw source material for one cell.

    The video/latent SHA fields are caller assertions.  This object proves only
    that all CPU consumers used the same embedded raw bytes and masks.
    """

    schema_version: str
    source_video_sha256: str
    source_latent_sha256: str
    binding_digest: str
    mask_digest: str
    track_authority_digest: str
    step_index: int
    block_index: int
    branch: str
    temporal_phases: int
    batch_size: int
    height: int
    width: int
    heads: int
    head_dim: int
    hidden_width: int
    raw_phase0_hidden_hex: str
    raw_phase0_pre_rope_key_hex: str
    raw_phase0_value_hex: str
    raw_phase0_hidden_sha256: str
    raw_phase0_pre_rope_key_sha256: str
    raw_phase0_value_sha256: str
    source_role_mask_hex_by_role: tuple[tuple[str, str], ...]
    source_role_mask_sha256_by_role: tuple[tuple[str, str], ...]
    canonical_extraction_config_material: str
    canonical_extraction_config_sha256: str
    authority_kind: str
    immutable_byte_material: bool
    material_reopenable: bool
    externally_authenticated: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != RAW_SOURCE_MATERIAL_SCHEMA:
            raise V15BContractError("raw source material schema differs")
        for label, value in (
            ("source video", self.source_video_sha256),
            ("source latent", self.source_latent_sha256),
            ("binding", self.binding_digest), ("mask", self.mask_digest),
            ("track authority", self.track_authority_digest),
            ("raw phase0 hidden", self.raw_phase0_hidden_sha256),
            ("raw phase0 key", self.raw_phase0_pre_rope_key_sha256),
            ("raw phase0 value", self.raw_phase0_value_sha256),
            ("canonical extraction config", self.canonical_extraction_config_sha256),
        ):
            _sha(value, label=f"raw material {label}")
        _exact_int(self.step_index, label="raw material step")
        _exact_int(self.block_index, label="raw material block")
        if self.step_index >= DENOISE_STEPS or self.block_index >= TRANSFORMER_BLOCKS:
            raise V15BContractError("raw material step/block is outside execution geometry")
        if self.branch not in CFG_BRANCHES:
            raise V15BContractError("raw material branch differs")
        for label, value in (
            ("batch", self.batch_size), ("height", self.height),
            ("width", self.width), ("heads", self.heads),
            ("head dim", self.head_dim), ("hidden width", self.hidden_width),
        ):
            _exact_int(value, label=f"raw material {label}", minimum=1)
        if self.temporal_phases != LATENT_PHASES:
            raise V15BContractError("raw material must bind exactly 21 source phases")
        config = _canonical_source_extraction_config_material()
        if (self.canonical_extraction_config_material != config or
                hashlib.sha256(config.encode("utf-8")).hexdigest() !=
                self.canonical_extraction_config_sha256):
            raise V15BContractError("raw material canonical extraction config differs")
        roles = tuple(role for role, _ in self.source_role_mask_hex_by_role)
        sha_roles = tuple(role for role, _ in self.source_role_mask_sha256_by_role)
        if (roles != sha_roles or len(roles) != 4 or
                tuple(sorted(roles)) != roles or len(set(roles)) != 4):
            raise V15BContractError("raw material must carry four sorted source role masks")
        for role in roles:
            _role(role, label="raw material role")
        hidden, key, value, masks = self.reopen()
        if (tensor_sha256(hidden) != self.raw_phase0_hidden_sha256 or
                tensor_sha256(key) != self.raw_phase0_pre_rope_key_sha256 or
                tensor_sha256(value) != self.raw_phase0_value_sha256):
            raise V15BContractError(
                "raw phase0 H/K/V bytes differ from their tensor authority"
            )
        wanted_mask_sha = dict(self.source_role_mask_sha256_by_role)
        if any(tensor_sha256(masks[role]) != wanted_mask_sha[role] for role in roles):
            raise V15BContractError("raw source role-mask bytes differ from authority")
        stacked = torch.stack(tuple(masks[role] for role in roles), dim=1)
        if bool((stacked.sum(1) > 1).any()) or not bool(
            stacked.reshape(self.batch_size, 4, LATENT_PHASES, -1).any(-1).all()
        ):
            raise V15BContractError("raw source role masks overlap or omit a phase")
        if (self.authority_kind !=
                "caller_sha256_plus_embedded_immutable_cpu_tensor_bytes" or
                self.immutable_byte_material is not True or
                self.material_reopenable is not True or
                self.externally_authenticated is not False or
                self.scientific_claim_authorized is not False or
                self.route_authorized is not False):
            raise V15BContractError("raw source material exceeded CPU reference authority")
        if object_sha256(self._payload()) != _sha(self.digest, label="raw material digest"):
            raise V15BContractError("raw source material digest differs")

    @property
    def role_ids(self) -> tuple[str, ...]:
        return tuple(role for role, _ in self.source_role_mask_hex_by_role)

    def reopen(self) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        phase0_shape = (
            self.batch_size, self.height * self.width, self.heads, self.head_dim,
        )
        hidden = _reopen_tensor_hex(
            self.raw_phase0_hidden_hex,
            shape=(self.batch_size, self.height * self.width, self.hidden_width),
            dtype=torch.float32, label="raw phase0 hidden",
        )
        mask_shape = (
            self.batch_size, self.temporal_phases * self.height * self.width,
        )
        key = _reopen_tensor_hex(
            self.raw_phase0_pre_rope_key_hex, shape=phase0_shape,
            dtype=torch.float32, label="raw phase0 pre-RoPE K",
        )
        value = _reopen_tensor_hex(
            self.raw_phase0_value_hex, shape=phase0_shape,
            dtype=torch.float32, label="raw phase0 V",
        )
        masks = {
            role: _reopen_tensor_hex(
                raw, shape=mask_shape, dtype=torch.bool,
                label=f"raw source mask {role}",
            )
            for role, raw in self.source_role_mask_hex_by_role
        }
        return hidden, key, value, masks

    def _payload(self) -> dict[str, Any]:
        return _raw_source_material_payload({
            f.name: getattr(self, f.name) for f in fields(self)
            if f.name != "digest"
        })


def build_source_phase0_raw_material_v15b(
    *, source_video_sha256: str, source_latent_sha256: str,
    binding: SourceActionRoleBindingV15B, masks: SourceRoleMaskSetV15B,
    step_index: int, block_index: int, branch: str,
    source_hidden: torch.Tensor, source_pre_rope_key: torch.Tensor,
    source_value: torch.Tensor,
) -> SourcePhase0RawMaterialV15B:
    """Seal caller phase-0 H/K/V into immutable r8 bytes; never upgrades authority."""
    _revalidate_material(binding, label="raw material builder binding")
    _revalidate_material(masks, label="raw material builder masks")
    _sha(source_video_sha256, label="raw material builder source video")
    _sha(source_latent_sha256, label="raw material builder source latent")
    raw_hidden = _cpu_fp32(
        source_hidden, label="raw material source hidden", ndim=3
    )
    raw_key = _cpu_fp32(
        source_pre_rope_key, label="raw material source pre-RoPE key", ndim=4
    )
    raw_value = _cpu_fp32(
        source_value, label="raw material source value", ndim=4
    )
    if (tuple(raw_key.shape) != tuple(raw_value.shape) or
            tuple(raw_hidden.shape[:2]) != tuple(raw_key.shape[:2]) or
            tuple(raw_key.shape[:2]) != tuple(masks.editable_corridor_mask.shape) or
            masks.source_video_sha256 != source_video_sha256 or
            masks.binding_digest != binding.digest):
        raise V15BContractError("raw material source K/V/mask authority differs")
    _exact_int(step_index, label="raw material builder step")
    _exact_int(block_index, label="raw material builder block")
    if step_index >= DENOISE_STEPS or block_index >= TRANSFORMER_BLOCKS:
        raise V15BContractError("raw material builder cell is outside execution geometry")
    if branch not in CFG_BRANCHES:
        raise V15BContractError("raw material builder branch differs")
    spatial = masks.height * masks.width
    phase0_hidden = raw_hidden[:, :spatial].detach().clone()
    phase0_key = raw_key[:, :spatial].detach().clone()
    phase0_value = raw_value[:, :spatial].detach().clone()
    role_ids = tuple(sorted(binding.source_roles))
    role_masks = tuple(
        (role, masks.role_masks[role].detach().clone()) for role in role_ids
    )
    config = _canonical_source_extraction_config_material()
    values = {
        "schema_version": RAW_SOURCE_MATERIAL_SCHEMA,
        "source_video_sha256": source_video_sha256,
        "source_latent_sha256": source_latent_sha256,
        "binding_digest": binding.digest, "mask_digest": masks.digest,
        "track_authority_digest": masks.track_authority.digest,
        "step_index": step_index, "block_index": block_index, "branch": branch,
        "temporal_phases": LATENT_PHASES, "batch_size": int(raw_key.shape[0]),
        "height": masks.height, "width": masks.width,
        "heads": int(raw_key.shape[2]), "head_dim": int(raw_key.shape[3]),
        "hidden_width": int(raw_hidden.shape[2]),
        "raw_phase0_hidden_hex": _immutable_tensor_hex(
            phase0_hidden, label="raw phase0 hidden"
        ),
        "raw_phase0_pre_rope_key_hex": _immutable_tensor_hex(
            phase0_key, label="raw phase0 key"
        ),
        "raw_phase0_value_hex": _immutable_tensor_hex(
            phase0_value, label="raw phase0 value"
        ),
        "raw_phase0_hidden_sha256": tensor_sha256(phase0_hidden),
        "raw_phase0_pre_rope_key_sha256": tensor_sha256(phase0_key),
        "raw_phase0_value_sha256": tensor_sha256(phase0_value),
        "source_role_mask_hex_by_role": tuple(
            (role, _immutable_tensor_hex(mask, label=f"raw source mask {role}"))
            for role, mask in role_masks
        ),
        "source_role_mask_sha256_by_role": tuple(
            (role, tensor_sha256(mask)) for role, mask in role_masks
        ),
        "canonical_extraction_config_material": config,
        "canonical_extraction_config_sha256": hashlib.sha256(
            config.encode("utf-8")
        ).hexdigest(),
        "authority_kind": "caller_sha256_plus_embedded_immutable_cpu_tensor_bytes",
        "immutable_byte_material": True, "material_reopenable": True,
        "externally_authenticated": False,
        "scientific_claim_authorized": False, "route_authorized": False,
    }
    material = SourcePhase0RawMaterialV15B(
        **values, digest=object_sha256(_raw_source_material_payload(values))
    )
    _validate_raw_source_material_against_masks_v15b(material, masks, binding)
    return material


def _validate_raw_source_material_against_masks_v15b(
    raw_source_material: SourcePhase0RawMaterialV15B,
    masks: SourceRoleMaskSetV15B,
    binding: Optional[SourceActionRoleBindingV15B] = None,
) -> None:
    _revalidate_material(raw_source_material, label="raw/mask source material")
    _revalidate_material(masks, label="raw/mask source masks")
    if binding is not None:
        _revalidate_material(binding, label="raw/mask source binding")
    expected_binding_digest = masks.binding_digest if binding is None else binding.digest
    expected_roles = tuple(sorted(masks.role_masks)) if binding is None else tuple(
        sorted(binding.source_roles)
    )
    if (
        raw_source_material.source_video_sha256 != masks.source_video_sha256
        or raw_source_material.binding_digest != expected_binding_digest
        or raw_source_material.mask_digest != masks.digest
        or raw_source_material.track_authority_digest != masks.track_authority.digest
        or raw_source_material.role_ids != expected_roles
        or (raw_source_material.height, raw_source_material.width) !=
        (masks.height, masks.width)
    ):
        raise V15BContractError("raw source material/mask authority differs")
    _, _, _, reopened_masks = raw_source_material.reopen()
    for role in raw_source_material.role_ids:
        if not torch.equal(reopened_masks[role], masks.role_masks[role]):
            raise V15BContractError(
                f"raw source material role {role} mask differs from source authority"
            )


@dataclass(frozen=True)
class SourceContentBuilderReceiptV15B:
    schema_version: str
    source_video_sha256: str
    source_latent_sha256: str
    binding_digest: str
    mask_digest: str
    track_authority_digest: str
    raw_source_material_digest: str
    canonical_extraction_config_sha256: str
    step_index: int
    block_index: int
    branch: str
    source_shape: tuple[int, ...]
    source_dtype: str
    source_device: str
    source_pre_rope_key_sha256: str
    position_reference_digest: str
    source_position_projector_sha256: str
    source_scrubbed_pre_rope_key_sha256: str
    source_value_sha256: str
    role_token_counts: tuple[tuple[str, int], ...]
    null_token_count: int
    output_key_sha256: str
    output_value_sha256: str
    output_slot_valid_sha256: str
    output_null_key_sha256: str
    output_position_projector_sha256: str
    per_role_output_key_sha256: tuple[tuple[str, str], ...]
    per_role_output_value_sha256: tuple[tuple[str, str], ...]
    per_role_slot_uuid_sha256: tuple[tuple[str, str], ...]
    slot_provenance_digest: str
    permutation_probe_sha256: str
    permutation_invariant: bool
    basis_kind: str
    position_scrub_kind: str
    position_subspace_rank_by_head: tuple[int, ...]
    position_scrub_projection_residual_max_abs: float
    position_calibration_fixture_digest: str
    position_fit_translation_count: int
    position_heldout_translation_count: int
    position_fit_reconstruction_residual_max_abs: float
    position_heldout_correspondence_residual_max_abs: float
    position_material_digest_authenticated: bool
    raw_material_reopened: bool
    per_role_tensor_recomputed: bool
    slot_uuid_mask_provenance_verified: bool
    position_removed_claimed: bool
    scientific_claim_authorized: bool
    phase0_only: bool
    externally_authenticated: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_BUILDER_RECEIPT_SCHEMA:
            raise V15BContractError("source content builder receipt schema differs")
        for label, value in (
            ("source video", self.source_video_sha256), ("binding", self.binding_digest),
            ("source latent", self.source_latent_sha256),
            ("mask", self.mask_digest), ("track authority", self.track_authority_digest),
            ("raw source material", self.raw_source_material_digest),
            ("canonical extraction config", self.canonical_extraction_config_sha256),
            ("source key", self.source_pre_rope_key_sha256),
            ("position reference", self.position_reference_digest),
            ("source position projector", self.source_position_projector_sha256),
            ("source scrubbed key", self.source_scrubbed_pre_rope_key_sha256),
            ("source value", self.source_value_sha256), ("output key", self.output_key_sha256),
            ("output value", self.output_value_sha256),
            ("output slot valid", self.output_slot_valid_sha256),
            ("output null key", self.output_null_key_sha256),
            ("output position projector", self.output_position_projector_sha256),
            ("slot provenance", self.slot_provenance_digest),
            ("permutation probe", self.permutation_probe_sha256),
            ("position calibration fixture", self.position_calibration_fixture_digest),
        ):
            _sha(value, label=label)
        _exact_int(self.step_index, label="builder step")
        _exact_int(self.block_index, label="builder block")
        if self.step_index >= DENOISE_STEPS or self.block_index >= TRANSFORMER_BLOCKS:
            raise V15BContractError("builder step/block is outside exact execution geometry")
        if self.branch not in CFG_BRANCHES:
            raise V15BContractError("builder branch differs")
        if len(self.source_shape) != 4 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in self.source_shape
        ):
            raise V15BContractError("builder source tensor geometry differs")
        if self.source_dtype != "torch.float32" or self.source_device != "cpu":
            raise V15BContractError("CPU reference builder requires CPU FP32 source tensors")
        if (len(self.role_token_counts) != 4 or
                tuple(sorted(role for role, _ in self.role_token_counts)) !=
                tuple(role for role, _ in self.role_token_counts)):
            raise V15BContractError("builder role counts must be four sorted roles")
        for role, count in self.role_token_counts:
            _role(role, label="builder role")
            _exact_int(count, label="builder role tokens", minimum=MIN_CONTENT_SLOTS_PER_ROLE)
        expected_roles = tuple(role for role, _ in self.role_token_counts)
        for label, rows in (
            ("per-role output K", self.per_role_output_key_sha256),
            ("per-role output V", self.per_role_output_value_sha256),
            ("per-role slot UUID", self.per_role_slot_uuid_sha256),
        ):
            if tuple(role for role, _ in rows) != expected_roles:
                raise V15BContractError(f"builder {label} roles differ")
            for role, digest in rows:
                _role(role, label=f"builder {label} role")
                _sha(digest, label=f"builder {label} digest")
        _exact_int(self.null_token_count, label="builder null tokens", minimum=1)
        if (self.permutation_invariant is not True or
                self.basis_kind != "phase0_unordered_multislot_content_basis" or
                self.position_scrub_kind !=
                "fixed_synthetic_fixture_projection_reference_only" or
                self.phase0_only is not True):
            raise V15BContractError(
                "builder did not preserve the phase0 unordered reference-only basis"
            )
        if len(self.position_subspace_rank_by_head) != self.source_shape[2]:
            raise V15BContractError("builder position-subspace rank/head geometry differs")
        for rank in self.position_subspace_rank_by_head:
            _exact_int(rank, label="builder position-subspace rank", minimum=1)
            if rank >= self.source_shape[3]:
                raise V15BContractError("builder position-subspace rank is not proper")
        if _finite(
            self.position_scrub_projection_residual_max_abs,
            label="builder position scrub residual",
        ) > POSITION_PROJECTOR_TOLERANCE:
            raise V15BContractError("builder position scrub retained position-subspace energy")
        _exact_int(
            self.position_fit_translation_count,
            label="builder position fit translation count",
            minimum=len(POSITION_FIT_TRANSLATIONS_YX),
        )
        _exact_int(
            self.position_heldout_translation_count,
            label="builder position heldout translation count",
            minimum=len(POSITION_HELDOUT_TRANSLATIONS_YX),
        )
        if (self.position_fit_translation_count != len(POSITION_FIT_TRANSLATIONS_YX) or
                self.position_heldout_translation_count !=
                len(POSITION_HELDOUT_TRANSLATIONS_YX)):
            raise V15BContractError("builder position fit/heldout split differs")
        for label, value in (
            ("fit reconstruction", self.position_fit_reconstruction_residual_max_abs),
            ("heldout correspondence",
             self.position_heldout_correspondence_residual_max_abs),
        ):
            if _finite(value, label=f"builder {label} residual") > POSITION_PROJECTOR_TOLERANCE:
                raise V15BContractError(f"builder position {label} gate failed")
        if (self.position_material_digest_authenticated is not True or
                self.raw_material_reopened is not True or
                self.per_role_tensor_recomputed is not True or
                self.slot_uuid_mask_provenance_verified is not True or
                self.position_removed_claimed is not False or
                self.scientific_claim_authorized is not False):
            raise V15BContractError(
                "builder synthetic calibration cannot claim source position removal"
            )
        if self.externally_authenticated is not False or self.route_authorized is not False:
            raise V15BContractError("CPU memory receipt cannot claim external/route authority")
        if object_sha256(self._payload()) != _sha(self.digest, label="builder receipt"):
            raise V15BContractError("source content builder receipt digest differs")

    def _payload(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "digest"}


@dataclass(frozen=True)
class SourceRoleContentMemoryV15B:
    """Reference-only phase-0 unordered multi-slot source-property basis."""

    schema_version: str
    source_video_sha256: str
    binding_digest: str
    mask_digest: str
    step_index: int
    block_index: int
    branch: str
    role_ids: tuple[str, ...]
    key_content: torch.Tensor   # [R,M,H,D], pre-RoPE phase-0 slots
    value_content: torch.Tensor # [R,M,H,D], paired source-property slots
    slot_valid_mask: torch.Tensor # [R,M]
    null_key_content: torch.Tensor # [N,H,D], phase-0 non-role slots
    slot_provenance_by_role: tuple[
        tuple[str, tuple[tuple[str, int, int, str, str, str], ...]], ...
    ]
    raw_source_material: SourcePhase0RawMaterialV15B
    position_reference: PositionCounterfactualReferenceV15B
    builder_receipt: SourceContentBuilderReceiptV15B
    construction_authority: str
    raw_reextract_verified: bool
    slot_uuid_mask_provenance_verified: bool
    externally_authenticated: bool
    position_removed_claimed: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_SCHEMA:
            raise V15BContractError("source content-memory schema differs")
        _sha(self.source_video_sha256, label="source video")
        _sha(self.binding_digest, label="memory binding"); _sha(self.mask_digest, label="memory mask")
        _exact_int(self.step_index, label="memory step"); _exact_int(self.block_index, label="memory block")
        if self.branch not in CFG_BRANCHES:
            raise V15BContractError("memory branch differs")
        if (len(self.role_ids) != 4 or tuple(sorted(self.role_ids)) != self.role_ids or
                len(set(self.role_ids)) != 4):
            raise V15BContractError("memory must carry four sorted distinct source roles")
        for role in self.role_ids:
            _role(role, label="memory role")
        key = _cpu_fp32(self.key_content, label="source pre-RoPE K content", ndim=4)
        value = _cpu_fp32(self.value_content, label="source V content", ndim=4)
        valid = _tensor(self.slot_valid_mask, label="source content slot-valid mask", ndim=2, floating=False)
        null_key = _cpu_fp32(self.null_key_content, label="source null K content", ndim=3)
        if not isinstance(self.raw_source_material, SourcePhase0RawMaterialV15B):
            raise V15BContractError("source memory lacks immutable raw source material")
        _revalidate_material(self.raw_source_material, label="memory raw source material")
        raw = self.raw_source_material
        if (
            raw.source_video_sha256 != self.source_video_sha256
            or raw.binding_digest != self.binding_digest
            or raw.mask_digest != self.mask_digest
            or (raw.step_index, raw.block_index, raw.branch) !=
            (self.step_index, self.block_index, self.branch)
            or raw.role_ids != self.role_ids
        ):
            raise V15BContractError("source memory/raw material cell authority differs")
        if not isinstance(self.position_reference, PositionCounterfactualReferenceV15B):
            raise V15BContractError("source memory lacks a position counterfactual reference")
        _revalidate_material(self.position_reference, label="memory position reference")
        projector = _position_projector(
            self.position_reference.projector,
            heads=int(key.shape[2]), head_dim=int(key.shape[3]),
            label="source position-subspace projector",
        )
        if (tuple(key.shape) != tuple(value.shape) or int(key.shape[0]) != 4 or
                int(key.shape[1]) < MIN_CONTENT_SLOTS_PER_ROLE or
                tuple(valid.shape) != tuple(key.shape[:2]) or valid.dtype != torch.bool or
                valid.device.type != "cpu" or tuple(null_key.shape[1:]) != tuple(key.shape[2:])):
            raise V15BContractError("source content-memory multi-slot geometry differs")
        projected_role = torch.einsum("rmhd,hde->rmhe", key, projector)
        projected_null = torch.einsum("nhd,hde->nhe", null_key, projector)
        if (float(projected_role.abs().max()) > POSITION_PROJECTOR_TOLERANCE or
                float(projected_null.abs().max()) > POSITION_PROJECTOR_TOLERANCE):
            raise V15BContractError("source content K retained position-subspace energy")
        receipt = self.builder_receipt
        if not isinstance(receipt, SourceContentBuilderReceiptV15B):
            raise V15BContractError("source content memory lacks builder receipt")
        _revalidate_material(receipt, label="memory builder receipt")
        expected_binding = (
            receipt.source_video_sha256, receipt.binding_digest, receipt.mask_digest,
            receipt.step_index, receipt.block_index, receipt.branch,
        )
        actual_binding = (
            self.source_video_sha256, self.binding_digest, self.mask_digest,
            self.step_index, self.block_index, self.branch,
        )
        if actual_binding != expected_binding:
            raise V15BContractError("source content memory/builder binding differs")
        if tuple(role for role, _ in receipt.role_token_counts) != self.role_ids:
            raise V15BContractError("source content memory/builder roles differ")
        provenance_roles = tuple(role for role, _ in self.slot_provenance_by_role)
        if provenance_roles != self.role_ids:
            raise V15BContractError("source content slot provenance roles differ")
        for role_index, (_, count) in enumerate(receipt.role_token_counts):
            if int(valid[role_index].sum()) != count:
                raise V15BContractError("source content valid slots differ from builder counts")
            if not bool(valid[role_index, :count].all()) or bool(valid[role_index, count:].any()):
                raise V15BContractError("source content valid slots must be canonical packed prefixes")
            if count < int(key.shape[1]):
                if int(torch.count_nonzero(key[role_index, count:])) or int(
                    torch.count_nonzero(value[role_index, count:])
                ):
                    raise V15BContractError("source content padding must be exact zero")
            entries = self.slot_provenance_by_role[role_index][1]
            if len(entries) != count:
                raise V15BContractError("source content slot provenance count differs")
            for entry in entries:
                if (not isinstance(entry, tuple) or len(entry) != 6 or
                        any(isinstance(entry[index], bool) or
                            not isinstance(entry[index], int) or entry[index] < 0
                            for index in (1, 2))):
                    raise V15BContractError("source content slot provenance entry differs")
                for label, digest in (
                    ("UUID", entry[0]), ("role mask", entry[3]),
                    ("raw K", entry[4]), ("raw V", entry[5]),
                ):
                    _sha(digest, label=f"slot provenance {label}")
        if int(null_key.shape[0]) != receipt.null_token_count:
            raise V15BContractError("source null basis count differs")
        if (tensor_sha256(key) != receipt.output_key_sha256 or
                tensor_sha256(value) != receipt.output_value_sha256 or
                tensor_sha256(valid) != receipt.output_slot_valid_sha256 or
                tensor_sha256(null_key) != receipt.output_null_key_sha256 or
                tensor_sha256(projector) != receipt.output_position_projector_sha256):
            raise V15BContractError("source content memory differs from builder output")
        if (receipt.position_reference_digest != self.position_reference.digest or
                receipt.position_subspace_rank_by_head != self.position_reference.rank_by_head or
                receipt.position_calibration_fixture_digest !=
                self.position_reference.calibration_fixture.digest or
                receipt.position_fit_translation_count !=
                self.position_reference.fit_translation_count or
                receipt.position_heldout_translation_count !=
                self.position_reference.heldout_translation_count or
                receipt.position_fit_reconstruction_residual_max_abs !=
                self.position_reference.fit_reconstruction_residual_max_abs or
                receipt.position_heldout_correspondence_residual_max_abs !=
                self.position_reference.heldout_correspondence_residual_max_abs or
                receipt.position_removed_claimed is not False or
                receipt.scientific_claim_authorized is not False):
            raise V15BContractError("source memory/position reference replay differs")
        if (receipt.source_position_projector_sha256 != tensor_sha256(projector) or
                receipt.output_position_projector_sha256 != tensor_sha256(projector)):
            raise V15BContractError("source memory position input/projector binding differs")
        if (
            receipt.source_latent_sha256 != raw.source_latent_sha256
            or receipt.track_authority_digest != raw.track_authority_digest
            or receipt.raw_source_material_digest != raw.digest
            or receipt.canonical_extraction_config_sha256 !=
            raw.canonical_extraction_config_sha256
            or receipt.source_shape != (
                raw.batch_size, raw.height * raw.width, raw.heads, raw.head_dim
            )
            or receipt.source_pre_rope_key_sha256 !=
            raw.raw_phase0_pre_rope_key_sha256
            or receipt.source_value_sha256 != raw.raw_phase0_value_sha256
            or receipt.slot_provenance_digest !=
            object_sha256(self.slot_provenance_by_role)
        ):
            raise V15BContractError("source memory/raw receipt provenance differs")
        replay = _extract_source_role_content_from_raw_v15b(
            raw_source_material=raw, position_reference=self.position_reference,
        )
        if (
            receipt.source_scrubbed_pre_rope_key_sha256 !=
            replay["scrubbed_phase0_key_sha256"]
            or receipt.permutation_probe_sha256 !=
            replay["permutation_probe_sha256"]
            or receipt.permutation_invariant != replay["permutation_invariant"]
            or receipt.position_scrub_projection_residual_max_abs !=
            replay["scrub_residual"]
        ):
            raise V15BContractError(
                "source memory/raw extraction receipt replay differs"
            )
        for role_index, role in enumerate(self.role_ids):
            count = dict(receipt.role_token_counts)[role]
            if (not torch.equal(
                    key[role_index, :count], replay["key_content"][role_index, :count]
                ) or not torch.equal(
                    value[role_index, :count], replay["value_content"][role_index, :count]
                )):
                raise V15BContractError(
                    f"source memory role {role} K/V differs from raw re-extraction"
                )
        if (not torch.equal(valid, replay["slot_valid_mask"]) or
                not torch.equal(null_key, replay["null_key_content"]) or
                self.slot_provenance_by_role != replay["slot_provenance_by_role"]):
            raise V15BContractError("source memory slot/null provenance replay differs")
        if (
            receipt.per_role_output_key_sha256 != replay["per_role_key_sha256"]
            or receipt.per_role_output_value_sha256 != replay["per_role_value_sha256"]
            or receipt.per_role_slot_uuid_sha256 != replay["per_role_slot_uuid_sha256"]
            or receipt.raw_material_reopened is not True
            or receipt.per_role_tensor_recomputed is not True
            or receipt.slot_uuid_mask_provenance_verified is not True
        ):
            raise V15BContractError("source memory per-role raw replay receipt differs")
        if (self.construction_authority != "self_contained_cpu_reference_only" or
                self.raw_reextract_verified is not True or
                self.slot_uuid_mask_provenance_verified is not True or
                self.externally_authenticated is not False or
                self.position_removed_claimed is not False or
                self.scientific_claim_authorized is not False or
                self.route_authorized is not False):
            raise V15BContractError("source memory must remain unauthenticated reference-only")
        if object_sha256(self._payload()) != _sha(self.digest, label="memory digest"):
            raise V15BContractError("source content-memory digest differs")

    @property
    def pre_rope(self) -> bool:
        return True

    @property
    def coordinate_free(self) -> bool:
        # The fixed synthetic calibration is not an external model proof.
        return False

    @property
    def phase_indexed(self) -> bool:
        return False

    @property
    def spatial_indexed(self) -> bool:
        return False

    @property
    def unordered_slots(self) -> bool:
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_video_sha256": self.source_video_sha256,
            "binding_digest": self.binding_digest, "mask_digest": self.mask_digest,
            "step_index": self.step_index, "block_index": self.block_index,
            "branch": self.branch, "role_ids": self.role_ids,
            "key_sha256": tensor_sha256(self.key_content),
            "value_sha256": tensor_sha256(self.value_content),
            "slot_valid_sha256": tensor_sha256(self.slot_valid_mask),
            "null_key_sha256": tensor_sha256(self.null_key_content),
            "slot_provenance_digest": object_sha256(self.slot_provenance_by_role),
            "raw_source_material_digest": self.raw_source_material.digest,
            "position_projector_sha256": tensor_sha256(
                self.position_reference.projector
            ),
            "position_reference_digest": self.position_reference.digest,
            "builder_receipt_digest": self.builder_receipt.digest,
            "construction_authority": self.construction_authority,
            "raw_reextract_verified": self.raw_reextract_verified,
            "slot_uuid_mask_provenance_verified": (
                self.slot_uuid_mask_provenance_verified
            ),
            "externally_authenticated": self.externally_authenticated,
            "position_removed_claimed": self.position_removed_claimed,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "route_authorized": self.route_authorized,
        }

    @classmethod
    def create(cls, **_: Any) -> "SourceRoleContentMemoryV15B":
        raise V15BContractError(
            "self-reported memory metadata is forbidden; use build_source_role_content_memory_v15b"
        )


def _canonical_role_slot_records_v15b(
    *, raw_source_material: SourcePhase0RawMaterialV15B, role: str,
    coordinates: Sequence[tuple[int, int]], scrubbed_key: torch.Tensor,
    raw_key: torch.Tensor, raw_value: torch.Tensor, role_mask_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[str, int, int, str, str, str], ...]]:
    if len(coordinates) < MIN_CONTENT_SLOTS_PER_ROLE:
        raise V15BContractError("role content basis has insufficient phase-0 slots")
    records = []
    for batch_index, token_index in coordinates:
        key_slot = scrubbed_key[batch_index, token_index].detach().clone()
        value_slot = raw_value[batch_index, token_index].detach().clone()
        raw_key_sha = tensor_sha256(raw_key[batch_index, token_index])
        raw_value_sha = tensor_sha256(raw_value[batch_index, token_index])
        pair_sha = object_sha256({
            "scrubbed_key_sha256": tensor_sha256(key_slot),
            "raw_value_sha256": tensor_sha256(value_slot),
        })
        slot_uuid = object_sha256({
            "schema_version": SLOT_PROVENANCE_SCHEMA,
            "raw_source_material_digest": raw_source_material.digest,
            "canonical_extraction_config_sha256": (
                raw_source_material.canonical_extraction_config_sha256
            ),
            "role": role, "batch_index": batch_index,
            "phase0_spatial_token_index": token_index,
            "raw_role_mask_sha256": role_mask_sha256,
            "raw_key_sha256": raw_key_sha, "raw_value_sha256": raw_value_sha,
        })
        provenance = (
            slot_uuid, batch_index, token_index, role_mask_sha256,
            raw_key_sha, raw_value_sha,
        )
        records.append((pair_sha, slot_uuid, key_slot, value_slot, provenance))
    records.sort(key=lambda item: (item[0], item[1]))
    return (
        torch.stack(tuple(item[2] for item in records)),
        torch.stack(tuple(item[3] for item in records)),
        tuple(item[4] for item in records),
    )


def _canonical_null_key_slots_v15b(
    scrubbed_key: torch.Tensor, coordinates: Sequence[tuple[int, int]],
) -> torch.Tensor:
    if not coordinates:
        raise V15BContractError("null content basis is empty")
    records = []
    for batch_index, token_index in coordinates:
        slot = scrubbed_key[batch_index, token_index].detach().clone()
        records.append((tensor_sha256(slot), batch_index, token_index, slot))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    return torch.stack(tuple(item[3] for item in records))


def _extract_source_role_content_from_raw_v15b(
    *, raw_source_material: SourcePhase0RawMaterialV15B,
    position_reference: PositionCounterfactualReferenceV15B,
) -> dict[str, Any]:
    """Freshly reopen raw bytes and deterministically extract every role slot."""
    _revalidate_material(raw_source_material, label="source extraction raw material")
    _revalidate_material(position_reference, label="source extraction position reference")
    _raw_hidden, raw_key, raw_value, role_masks = raw_source_material.reopen()
    projector = _position_projector(
        position_reference.projector, heads=raw_source_material.heads,
        head_dim=raw_source_material.head_dim,
        label="source extraction position projector",
    )
    scrubbed_key = _scrub_position_subspace(raw_key, projector)
    scrub_residual = float(torch.einsum(
        "bshd,hde->bshe", scrubbed_key, projector
    ).abs().max())
    role_ids = raw_source_material.role_ids
    mask_sha = dict(raw_source_material.source_role_mask_sha256_by_role)
    spatial = raw_source_material.height * raw_source_material.width
    role_union = torch.zeros(raw_source_material.batch_size, spatial, dtype=torch.bool)
    role_keys = []; role_values = []; provenance_rows = []; counts = []
    reverse_keys = []; reverse_values = []; reverse_provenance = []
    for role in role_ids:
        phase0_mask = role_masks[role][:, :spatial]
        role_union |= phase0_mask
        coordinates = tuple(
            (int(row[0]), int(row[1]))
            for row in torch.nonzero(phase0_mask, as_tuple=False).tolist()
        )
        key_slots, value_slots, provenance = _canonical_role_slot_records_v15b(
            raw_source_material=raw_source_material, role=role,
            coordinates=coordinates, scrubbed_key=scrubbed_key,
            raw_key=raw_key, raw_value=raw_value,
            role_mask_sha256=mask_sha[role],
        )
        reverse_key, reverse_value, reverse_slots = _canonical_role_slot_records_v15b(
            raw_source_material=raw_source_material, role=role,
            coordinates=tuple(reversed(coordinates)), scrubbed_key=scrubbed_key,
            raw_key=raw_key, raw_value=raw_value,
            role_mask_sha256=mask_sha[role],
        )
        role_keys.append(key_slots); role_values.append(value_slots)
        provenance_rows.append((role, provenance)); counts.append((role, len(coordinates)))
        reverse_keys.append(reverse_key); reverse_values.append(reverse_value)
        reverse_provenance.append((role, reverse_slots))
    maximum_slots = max(count for _, count in counts)
    output_key = torch.zeros(
        4, maximum_slots, raw_source_material.heads,
        raw_source_material.head_dim, dtype=torch.float32,
    )
    output_value = torch.zeros_like(output_key)
    reverse_output_key = torch.zeros_like(output_key)
    reverse_output_value = torch.zeros_like(output_key)
    slot_valid = torch.zeros(4, maximum_slots, dtype=torch.bool)
    for index, (_, count) in enumerate(counts):
        output_key[index, :count] = role_keys[index]
        output_value[index, :count] = role_values[index]
        reverse_output_key[index, :count] = reverse_keys[index]
        reverse_output_value[index, :count] = reverse_values[index]
        slot_valid[index, :count] = True
    null_coordinates = tuple(
        (int(row[0]), int(row[1]))
        for row in torch.nonzero(~role_union, as_tuple=False).tolist()
    )
    null_key = _canonical_null_key_slots_v15b(scrubbed_key, null_coordinates)
    reverse_null_key = _canonical_null_key_slots_v15b(
        scrubbed_key, tuple(reversed(null_coordinates))
    )
    invariant = (
        torch.equal(output_key, reverse_output_key)
        and torch.equal(output_value, reverse_output_value)
        and torch.equal(null_key, reverse_null_key)
        and tuple(provenance_rows) == tuple(reverse_provenance)
    )
    probe = object_sha256({
        "raw_source_material_digest": raw_source_material.digest,
        "forward_key": tensor_sha256(output_key),
        "forward_value": tensor_sha256(output_value),
        "reverse_key": tensor_sha256(reverse_output_key),
        "reverse_value": tensor_sha256(reverse_output_value),
        "forward_null_key": tensor_sha256(null_key),
        "reverse_null_key": tensor_sha256(reverse_null_key),
        "forward_slot_provenance": object_sha256(tuple(provenance_rows)),
        "reverse_slot_provenance": object_sha256(tuple(reverse_provenance)),
        "position_projector": tensor_sha256(projector),
    })
    return {
        "role_ids": role_ids, "key_content": output_key,
        "value_content": output_value, "slot_valid_mask": slot_valid,
        "null_key_content": null_key,
        "slot_provenance_by_role": tuple(provenance_rows),
        "role_token_counts": tuple(counts),
        "null_token_count": int(null_key.shape[0]),
        "per_role_key_sha256": tuple(
            (role, tensor_sha256(role_keys[index]))
            for index, role in enumerate(role_ids)
        ),
        "per_role_value_sha256": tuple(
            (role, tensor_sha256(role_values[index]))
            for index, role in enumerate(role_ids)
        ),
        "per_role_slot_uuid_sha256": tuple(
            (role, object_sha256(tuple(entry[0] for entry in provenance)))
            for role, provenance in provenance_rows
        ),
        "scrubbed_phase0_key_sha256": tensor_sha256(scrubbed_key),
        "projector": projector, "scrub_residual": scrub_residual,
        "permutation_probe_sha256": probe,
        "permutation_invariant": invariant,
    }


def build_source_role_content_memory_v15b(
    *, source_video_sha256: str, source_latent_sha256: str,
    binding: SourceActionRoleBindingV15B,
    masks: SourceRoleMaskSetV15B, step_index: int, block_index: int,
    branch: str, source_hidden: torch.Tensor,
    source_pre_rope_key: torch.Tensor, source_value: torch.Tensor,
    position_reference: PositionCounterfactualReferenceV15B,
) -> SourceRoleContentMemoryV15B:
    """Build phase-0 slots with a synthetic projection, without a removal claim."""
    _revalidate_material(binding, label="memory builder binding")
    _revalidate_material(masks, label="memory builder masks")
    if not isinstance(position_reference, PositionCounterfactualReferenceV15B):
        raise V15BContractError(
            "memory builder requires the sealed fixed calibration reference"
        )
    _revalidate_material(position_reference, label="memory builder position reference")
    raw_material = build_source_phase0_raw_material_v15b(
        source_video_sha256=source_video_sha256,
        source_latent_sha256=source_latent_sha256,
        binding=binding, masks=masks, step_index=step_index,
        block_index=block_index, branch=branch,
        source_hidden=source_hidden,
        source_pre_rope_key=source_pre_rope_key, source_value=source_value,
    )
    extracted = _extract_source_role_content_from_raw_v15b(
        raw_source_material=raw_material, position_reference=position_reference,
    )
    projector = extracted["projector"]
    output_key = extracted["key_content"]
    output_value = extracted["value_content"]
    slot_valid = extracted["slot_valid_mask"]
    null_key = extracted["null_key_content"]
    counts = extracted["role_token_counts"]
    role_ids = extracted["role_ids"]
    receipt_payload = {
        "schema_version": MEMORY_BUILDER_RECEIPT_SCHEMA,
        "source_video_sha256": source_video_sha256,
        "source_latent_sha256": source_latent_sha256,
        "binding_digest": binding.digest, "mask_digest": masks.digest,
        "track_authority_digest": masks.track_authority.digest,
        "raw_source_material_digest": raw_material.digest,
        "canonical_extraction_config_sha256": (
            raw_material.canonical_extraction_config_sha256
        ),
        "step_index": step_index,
        "block_index": block_index, "branch": branch,
        "source_shape": (
            raw_material.batch_size, raw_material.height * raw_material.width,
            raw_material.heads, raw_material.head_dim,
        ),
        "source_dtype": "torch.float32", "source_device": "cpu",
        "source_pre_rope_key_sha256": raw_material.raw_phase0_pre_rope_key_sha256,
        "position_reference_digest": position_reference.digest,
        "source_position_projector_sha256": tensor_sha256(projector),
        "source_scrubbed_pre_rope_key_sha256": extracted[
            "scrubbed_phase0_key_sha256"
        ],
        "source_value_sha256": raw_material.raw_phase0_value_sha256,
        "role_token_counts": tuple(counts),
        "null_token_count": int(null_key.shape[0]),
        "output_key_sha256": tensor_sha256(output_key),
        "output_value_sha256": tensor_sha256(output_value),
        "output_slot_valid_sha256": tensor_sha256(slot_valid),
        "output_null_key_sha256": tensor_sha256(null_key),
        "output_position_projector_sha256": tensor_sha256(projector),
        "per_role_output_key_sha256": extracted["per_role_key_sha256"],
        "per_role_output_value_sha256": extracted["per_role_value_sha256"],
        "per_role_slot_uuid_sha256": extracted["per_role_slot_uuid_sha256"],
        "slot_provenance_digest": object_sha256(
            extracted["slot_provenance_by_role"]
        ),
        "permutation_probe_sha256": extracted["permutation_probe_sha256"],
        "permutation_invariant": extracted["permutation_invariant"],
        "basis_kind": "phase0_unordered_multislot_content_basis",
        "position_scrub_kind": "fixed_synthetic_fixture_projection_reference_only",
        "position_subspace_rank_by_head": tuple(
            int(torch.linalg.matrix_rank(
                projector[head], tol=POSITION_PROJECTOR_TOLERANCE
            )) for head in range(int(projector.shape[0]))
        ),
        "position_scrub_projection_residual_max_abs": extracted["scrub_residual"],
        "position_calibration_fixture_digest": (
            position_reference.calibration_fixture.digest
        ),
        "position_fit_translation_count": position_reference.fit_translation_count,
        "position_heldout_translation_count": position_reference.heldout_translation_count,
        "position_fit_reconstruction_residual_max_abs": (
            position_reference.fit_reconstruction_residual_max_abs
        ),
        "position_heldout_correspondence_residual_max_abs": (
            position_reference.heldout_correspondence_residual_max_abs
        ),
        "position_material_digest_authenticated": True,
        "raw_material_reopened": True,
        "per_role_tensor_recomputed": True,
        "slot_uuid_mask_provenance_verified": True,
        "position_removed_claimed": False,
        "scientific_claim_authorized": False,
        "phase0_only": True, "externally_authenticated": False,
        "route_authorized": False,
    }
    receipt = SourceContentBuilderReceiptV15B(
        **receipt_payload, digest=object_sha256(receipt_payload)
    )
    memory_payload = {
        "schema_version": MEMORY_SCHEMA, "source_video_sha256": source_video_sha256,
        "binding_digest": binding.digest, "mask_digest": masks.digest,
        "step_index": step_index, "block_index": block_index, "branch": branch,
        "role_ids": role_ids, "key_sha256": tensor_sha256(output_key),
        "value_sha256": tensor_sha256(output_value),
        "slot_valid_sha256": tensor_sha256(slot_valid),
        "null_key_sha256": tensor_sha256(null_key),
        "slot_provenance_digest": object_sha256(
            extracted["slot_provenance_by_role"]
        ),
        "raw_source_material_digest": raw_material.digest,
        "position_projector_sha256": tensor_sha256(projector),
        "position_reference_digest": position_reference.digest,
        "builder_receipt_digest": receipt.digest,
        "construction_authority": "self_contained_cpu_reference_only",
        "raw_reextract_verified": True,
        "slot_uuid_mask_provenance_verified": True,
        "externally_authenticated": False, "position_removed_claimed": False,
        "scientific_claim_authorized": False, "route_authorized": False,
    }
    return SourceRoleContentMemoryV15B(
        schema_version=MEMORY_SCHEMA, source_video_sha256=source_video_sha256,
        binding_digest=binding.digest, mask_digest=masks.digest,
        step_index=step_index, block_index=block_index, branch=branch,
        role_ids=role_ids, key_content=output_key, value_content=output_value,
        slot_valid_mask=slot_valid, null_key_content=null_key,
        slot_provenance_by_role=extracted["slot_provenance_by_role"],
        raw_source_material=raw_material, position_reference=position_reference,
        builder_receipt=receipt,
        construction_authority="self_contained_cpu_reference_only",
        raw_reextract_verified=True, slot_uuid_mask_provenance_verified=True,
        externally_authenticated=False, position_removed_claimed=False,
        scientific_claim_authorized=False, route_authorized=False,
        digest=object_sha256(memory_payload),
    )


def _normalize_role_candidate_masks(
    role_physical_candidate_masks: Mapping[str, torch.Tensor],
    *, height: int, width: int,
) -> tuple[tuple[str, ...], torch.Tensor]:
    if not isinstance(role_physical_candidate_masks, Mapping):
        raise V15BContractError("transport reference requires per-role candidate mapping")
    role_ids = tuple(sorted(role_physical_candidate_masks))
    if len(role_ids) != 4 or len(set(role_ids)) != 4:
        raise V15BContractError("transport reference requires four distinct role candidates")
    tensors = []
    for role in role_ids:
        _role(role, label="transport reference role")
        value = role_physical_candidate_masks[role]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
                value.dtype != torch.bool):
            raise V15BContractError("per-role transport candidates must be CPU bool")
        if tuple(value.shape) == (LATENT_PHASES, height, width):
            normalized = value.unsqueeze(0)
        elif tuple(value.shape) == (1, LATENT_PHASES, height, width):
            normalized = value
        elif tuple(value.shape) == (1, LATENT_PHASES, height * width):
            normalized = value.reshape(1, LATENT_PHASES, height, width)
        else:
            raise V15BContractError("per-role transport candidate geometry differs")
        tensors.append(normalized.reshape(1, LATENT_PHASES, height * width).clone())
    stacked = torch.stack(tensors, dim=1)
    if bool((stacked.sum(1) > 1).any()):
        raise V15BContractError("per-role transport candidates overlap")
    if bool((stacked.sum(-1) == 0).any()):
        raise V15BContractError("per-role transport candidate phase is empty")
    return role_ids, stacked


def _target_role_label_input(role_candidates: torch.Tensor) -> torch.Tensor:
    if (role_candidates.ndim != 4 or role_candidates.device.type != "cpu" or
            role_candidates.dtype != torch.bool):
        raise V15BContractError("target role-label input candidate geometry differs")
    batch, roles, phases, spatial = role_candidates.shape
    labels = torch.full((batch, phases, spatial), -1, dtype=torch.int64)
    for role_index in range(roles):
        labels[role_candidates[:, role_index]] = role_index
    return labels


def _transport_estimator_spec_material(
    *, role_ids: tuple[str, ...], height: int, width: int,
) -> tuple[str, str, str]:
    code = canonical_json_bytes({
        "algorithm": "role-label-rigid-integer-translation-estimator",
        "matching": "same-role-centroid-translation-exact-mask-replay",
        "forward_backward": "exact-bijection",
        "abi_revision": 6,
    }).decode("utf-8")
    config = canonical_json_bytes({
        "role_ids": role_ids, "height": height, "width": width,
        "temporal_phases": LATENT_PHASES,
        "maximum_step": TARGET_TRACK_MAX_CENTROID_JUMP,
    }).decode("utf-8")
    checkpoint = canonical_json_bytes({
        "kind": "no-learned-checkpoint",
        "estimator": "deterministic-self-contained-reference",
        "abi_revision": 6,
    }).decode("utf-8")
    return code, config, checkpoint


def _transport_estimator_spec_digests(
    *, role_ids: tuple[str, ...], height: int, width: int,
) -> tuple[str, str, str]:
    code, config, checkpoint = _transport_estimator_spec_material(
        role_ids=role_ids, height=height, width=width
    )
    return tuple(
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in (code, config, checkpoint)
    )


def _derive_transport_reference_maps(
    *, role_candidates: torch.Tensor, height: int, width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    candidates = _tensor(
        role_candidates, label="per-role transport candidates", ndim=4,
        floating=False,
    )
    if candidates.device.type != "cpu" or candidates.dtype != torch.bool:
        raise V15BContractError("per-role transport candidates must be CPU bool")
    batch, roles, phases, spatial = candidates.shape
    if (batch != 1 or roles != 4 or phases != LATENT_PHASES or
            spatial != height * width):
        raise V15BContractError("per-role transport candidate ABI differs")
    union = candidates.any(1)
    forward_flow = torch.zeros(
        batch, LATENT_PHASES - 1, spatial, 2, dtype=torch.float32
    )
    backward_flow = torch.zeros_like(forward_flow)
    forward = torch.full((batch, LATENT_PHASES - 1, spatial), -1, dtype=torch.int64)
    backward = torch.full_like(forward, -1)
    maximum_error = 0.0; closure_errors = 0
    for phase in range(LATENT_PHASES - 1):
        for role_index in range(roles):
            previous = candidates[0, role_index, phase].reshape(height, width)
            current = candidates[0, role_index, phase + 1].reshape(height, width)
            if int(previous.sum()) != int(current.sum()):
                raise V15BContractError("role transport candidate cardinality changed")
            previous_centroid = _centroid(previous)
            current_centroid = _centroid(current)
            dy_float = current_centroid[0] - previous_centroid[0]
            dx_float = current_centroid[1] - previous_centroid[1]
            dy, dx = int(round(dy_float)), int(round(dx_float))
            if (abs(dy_float - dy) > MOTION_INTEGER_TOLERANCE or
                    abs(dx_float - dx) > MOTION_INTEGER_TOLERANCE):
                raise V15BContractError("role transport is not an integer translation")
            if math.hypot(dy, dx) > TARGET_TRACK_MAX_CENTROID_JUMP:
                raise V15BContractError("role transport displacement exceeds continuity gate")
            if not torch.equal(_translated_mask(previous, dy, dx), current):
                raise V15BContractError("role transport shape/topology is not exact translation")
            for previous_token in torch.nonzero(previous.reshape(-1)).flatten().tolist():
                py, px = divmod(previous_token, width)
                current_token = (py + dy) * width + (px + dx)
                if (int(forward[0, phase, previous_token]) != -1 or
                        int(backward[0, phase, current_token]) != -1):
                    raise V15BContractError("role transport is not one-to-one")
                forward[0, phase, previous_token] = current_token
                backward[0, phase, current_token] = previous_token
                forward_flow[0, phase, previous_token] = torch.tensor((dy, dx))
                backward_flow[0, phase, current_token] = torch.tensor((-dy, -dx))
                maximum_error = max(maximum_error, float((
                    forward_flow[0, phase, previous_token] +
                    backward_flow[0, phase, current_token]
                ).abs().max()))
        valid_forward = forward[0, phase, union[0, phase]]
        valid_backward = backward[0, phase, union[0, phase + 1]]
        if (bool((valid_forward < 0).any()) or bool((valid_backward < 0).any()) or
                int(torch.unique(valid_forward).numel()) != int(valid_forward.numel()) or
                int(torch.unique(valid_backward).numel()) != int(valid_backward.numel())):
            closure_errors += 1
    if closure_errors or maximum_error > MOTION_INTEGER_TOLERANCE:
        raise V15BContractError("transport reference forward/backward closure failed")
    return forward_flow, backward_flow, forward, backward, maximum_error, closure_errors


@dataclass(frozen=True)
class TargetNativeMotionReferenceV15B:
    """Legacy name for a synthetic role-label transport reference, not native flow."""

    schema_version: str
    height: int
    width: int
    role_ids: tuple[str, ...]
    target_input_role_label_tensor: torch.Tensor
    role_physical_candidate_mask: torch.Tensor
    forward_displacement_yx: torch.Tensor
    backward_displacement_yx: torch.Tensor
    physical_candidate_mask: torch.Tensor
    forward_token_index: torch.Tensor
    backward_token_index: torch.Tensor
    forward_backward_error_max_abs: float
    physical_candidate_closure_error_count: int
    injective: bool
    estimator_code_material: str
    estimator_config_material: str
    estimator_checkpoint_material: str
    estimator_code_sha256: str
    estimator_config_sha256: str
    estimator_checkpoint_sha256: str
    estimator_output_transcript_sha256: str
    evidence_kind: str
    target_input_reopenable: bool
    estimator_material_reopenable: bool
    estimator_output_recomputed: bool
    material_digest_authenticated: bool
    externally_authenticated: bool
    native_flow_claimed: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_MOTION_REFERENCE_SCHEMA:
            raise V15BContractError("target transport reference schema differs")
        _exact_int(self.height, label="transport height", minimum=1)
        _exact_int(self.width, label="transport width", minimum=1)
        if (len(self.role_ids) != 4 or tuple(sorted(self.role_ids)) != self.role_ids or
                len(set(self.role_ids)) != 4):
            raise V15BContractError("transport reference roles differ")
        role_candidates = _tensor(
            self.role_physical_candidate_mask, label="role physical candidates",
            ndim=4, floating=False,
        )
        if (role_candidates.device.type != "cpu" or role_candidates.dtype != torch.bool or
                tuple(role_candidates.shape) !=
                (1, 4, LATENT_PHASES, self.height * self.width)):
            raise V15BContractError("role physical candidate tensor geometry differs")
        expected_input = _target_role_label_input(role_candidates)
        supplied_input = _tensor(
            self.target_input_role_label_tensor, label="target role-label input",
            ndim=3, floating=False,
        )
        if (supplied_input.device.type != "cpu" or supplied_input.dtype != torch.int64 or
                not torch.equal(supplied_input, expected_input)):
            raise V15BContractError("reopened target input differs from role candidates")
        union = role_candidates.any(1)
        supplied_union = _tensor(
            self.physical_candidate_mask, label="physical candidate union",
            ndim=3, floating=False,
        )
        if (supplied_union.device.type != "cpu" or supplied_union.dtype != torch.bool or
                not torch.equal(supplied_union, union)):
            raise V15BContractError("physical candidate union differs by role")
        derived = _derive_transport_reference_maps(
            role_candidates=role_candidates, height=self.height, width=self.width,
        )
        forward_flow, backward_flow, forward, backward, maximum_error, closure_errors = derived
        for label, supplied, expected in (
            ("forward displacement", self.forward_displacement_yx, forward_flow),
            ("backward displacement", self.backward_displacement_yx, backward_flow),
        ):
            value = _cpu_fp32(supplied, label=f"transport {label}", ndim=4)
            if not torch.equal(value, expected):
                raise V15BContractError(f"transport {label} is not estimator replay")
        for label, supplied, expected in (
            ("forward", self.forward_token_index, forward),
            ("backward", self.backward_token_index, backward),
        ):
            _tensor(supplied, label=f"transport {label} index", ndim=3,
                    floating=False)
            if (supplied.device.type != "cpu" or supplied.dtype != torch.int64 or
                    not torch.equal(supplied, expected)):
                raise V15BContractError(
                    f"transport {label} index is not deterministic replay"
                )
        if abs(_finite(
            self.forward_backward_error_max_abs,
            label="transport forward/backward error",
        ) - maximum_error) > MOTION_INTEGER_TOLERANCE:
            raise V15BContractError("transport consistency metric differs")
        _exact_int(
            self.physical_candidate_closure_error_count,
            label="transport physical closure errors",
        )
        code_material, config_material, checkpoint_material = (
            _transport_estimator_spec_material(
                role_ids=self.role_ids, height=self.height, width=self.width
            )
        )
        code_sha, config_sha, checkpoint_sha = _transport_estimator_spec_digests(
            role_ids=self.role_ids, height=self.height, width=self.width
        )
        for label, supplied, expected in (
            ("estimator code material", self.estimator_code_material, code_material),
            ("estimator config material", self.estimator_config_material, config_material),
            ("estimator checkpoint material", self.estimator_checkpoint_material,
             checkpoint_material),
        ):
            if not isinstance(supplied, str) or supplied != expected:
                raise V15BContractError(f"transport {label} differs")
        for label, supplied, expected in (
            ("estimator code", self.estimator_code_sha256, code_sha),
            ("estimator config", self.estimator_config_sha256, config_sha),
            ("estimator checkpoint", self.estimator_checkpoint_sha256, checkpoint_sha),
        ):
            if _sha(supplied, label=label) != expected:
                raise V15BContractError(f"transport {label} differs")
        transcript = object_sha256({
            "target_input_role_label_tensor_sha256": tensor_sha256(supplied_input),
            "role_physical_candidate_mask_sha256": tensor_sha256(role_candidates),
            "forward_displacement_yx_sha256": tensor_sha256(forward_flow),
            "backward_displacement_yx_sha256": tensor_sha256(backward_flow),
            "forward_token_index_sha256": tensor_sha256(forward),
            "backward_token_index_sha256": tensor_sha256(backward),
            "estimator_code_sha256": code_sha,
            "estimator_config_sha256": config_sha,
            "estimator_checkpoint_sha256": checkpoint_sha,
        })
        if _sha(self.estimator_output_transcript_sha256,
                label="estimator output transcript") != transcript:
            raise V15BContractError("transport estimator output transcript differs")
        if (self.physical_candidate_closure_error_count != closure_errors or
                closure_errors != 0 or self.injective is not True or
                self.evidence_kind != "self_contained_role_label_translation_reference" or
                self.target_input_reopenable is not True or
                self.estimator_material_reopenable is not True or
                self.estimator_output_recomputed is not True or
                self.material_digest_authenticated is not True or
                self.externally_authenticated is not False or
                self.native_flow_claimed is not False or
                self.scientific_claim_authorized is not False or
                self.route_authorized is not False):
            raise V15BContractError("transport reference must remain synthetic/no-authority")
        if object_sha256(self._payload()) != _sha(self.digest, label="transport reference"):
            raise V15BContractError("transport reference digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "height": self.height, "width": self.width,
            "role_ids": self.role_ids,
            "target_input_role_label_tensor_sha256": tensor_sha256(
                self.target_input_role_label_tensor
            ),
            "role_physical_candidate_mask_sha256": tensor_sha256(
                self.role_physical_candidate_mask
            ),
            "forward_displacement_yx_sha256": tensor_sha256(
                self.forward_displacement_yx
            ),
            "backward_displacement_yx_sha256": tensor_sha256(
                self.backward_displacement_yx
            ),
            "physical_candidate_mask_sha256": tensor_sha256(
                self.physical_candidate_mask
            ),
            "forward_token_index_sha256": tensor_sha256(self.forward_token_index),
            "backward_token_index_sha256": tensor_sha256(self.backward_token_index),
            "forward_backward_error_max_abs": self.forward_backward_error_max_abs,
            "physical_candidate_closure_error_count": (
                self.physical_candidate_closure_error_count
            ),
            "injective": self.injective,
            "estimator_code_material_sha256": hashlib.sha256(
                self.estimator_code_material.encode("utf-8")
            ).hexdigest(),
            "estimator_config_material_sha256": hashlib.sha256(
                self.estimator_config_material.encode("utf-8")
            ).hexdigest(),
            "estimator_checkpoint_material_sha256": hashlib.sha256(
                self.estimator_checkpoint_material.encode("utf-8")
            ).hexdigest(),
            "estimator_code_sha256": self.estimator_code_sha256,
            "estimator_config_sha256": self.estimator_config_sha256,
            "estimator_checkpoint_sha256": self.estimator_checkpoint_sha256,
            "estimator_output_transcript_sha256": self.estimator_output_transcript_sha256,
            "evidence_kind": self.evidence_kind,
            "target_input_reopenable": self.target_input_reopenable,
            "estimator_material_reopenable": self.estimator_material_reopenable,
            "estimator_output_recomputed": self.estimator_output_recomputed,
            "material_digest_authenticated": self.material_digest_authenticated,
            "externally_authenticated": self.externally_authenticated,
            "native_flow_claimed": self.native_flow_claimed,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "route_authorized": self.route_authorized,
        }


def build_target_native_motion_reference_v15b(
    *, role_physical_candidate_masks: Mapping[str, torch.Tensor],
    height: int, width: int,
) -> TargetNativeMotionReferenceV15B:
    _exact_int(height, label="transport reference height", minimum=1)
    _exact_int(width, label="transport reference width", minimum=1)
    role_ids, role_candidates = _normalize_role_candidate_masks(
        role_physical_candidate_masks, height=height, width=width,
    )
    target_input = _target_role_label_input(role_candidates)
    candidate = role_candidates.any(1)
    forward_flow, backward_flow, forward, backward, maximum_error, closure_errors = (
        _derive_transport_reference_maps(
            role_candidates=role_candidates, height=height, width=width,
        )
    )
    code_material, config_material, checkpoint_material = (
        _transport_estimator_spec_material(
            role_ids=role_ids, height=height, width=width
        )
    )
    code_sha, config_sha, checkpoint_sha = _transport_estimator_spec_digests(
        role_ids=role_ids, height=height, width=width
    )
    transcript = object_sha256({
        "target_input_role_label_tensor_sha256": tensor_sha256(target_input),
        "role_physical_candidate_mask_sha256": tensor_sha256(role_candidates),
        "forward_displacement_yx_sha256": tensor_sha256(forward_flow),
        "backward_displacement_yx_sha256": tensor_sha256(backward_flow),
        "forward_token_index_sha256": tensor_sha256(forward),
        "backward_token_index_sha256": tensor_sha256(backward),
        "estimator_code_sha256": code_sha,
        "estimator_config_sha256": config_sha,
        "estimator_checkpoint_sha256": checkpoint_sha,
    })
    payload = {
        "schema_version": NATIVE_MOTION_REFERENCE_SCHEMA,
        "height": height, "width": width,
        "role_ids": role_ids,
        "target_input_role_label_tensor_sha256": tensor_sha256(target_input),
        "role_physical_candidate_mask_sha256": tensor_sha256(role_candidates),
        "forward_displacement_yx_sha256": tensor_sha256(forward_flow),
        "backward_displacement_yx_sha256": tensor_sha256(backward_flow),
        "physical_candidate_mask_sha256": tensor_sha256(candidate),
        "forward_token_index_sha256": tensor_sha256(forward),
        "backward_token_index_sha256": tensor_sha256(backward),
        "forward_backward_error_max_abs": maximum_error,
        "physical_candidate_closure_error_count": closure_errors,
        "injective": True, "material_digest_authenticated": True,
        "estimator_code_material_sha256": hashlib.sha256(
            code_material.encode("utf-8")
        ).hexdigest(),
        "estimator_config_material_sha256": hashlib.sha256(
            config_material.encode("utf-8")
        ).hexdigest(),
        "estimator_checkpoint_material_sha256": hashlib.sha256(
            checkpoint_material.encode("utf-8")
        ).hexdigest(),
        "estimator_code_sha256": code_sha, "estimator_config_sha256": config_sha,
        "estimator_checkpoint_sha256": checkpoint_sha,
        "estimator_output_transcript_sha256": transcript,
        "evidence_kind": "self_contained_role_label_translation_reference",
        "target_input_reopenable": True,
        "estimator_material_reopenable": True,
        "estimator_output_recomputed": True,
        "externally_authenticated": False, "native_flow_claimed": False,
        "scientific_claim_authorized": False, "route_authorized": False,
    }
    return TargetNativeMotionReferenceV15B(
        NATIVE_MOTION_REFERENCE_SCHEMA, height, width, role_ids, target_input,
        role_candidates, forward_flow, backward_flow, candidate, forward, backward,
        maximum_error, closure_errors, True, code_material, config_material,
        checkpoint_material, code_sha, config_sha, checkpoint_sha,
        transcript, "self_contained_role_label_translation_reference", True, True,
        True, True, False, False, False, False, object_sha256(payload),
    )


@dataclass(frozen=True)
class TargetNativeTransportV15B:
    """Deterministic transport derived from the synthetic role-label reference."""

    schema_version: str
    source_video_sha256: str
    binding_digest: str
    mask_digest: str
    step_index: int
    block_index: int
    branch: str
    batch_size: int
    height: int
    width: int
    motion_reference: TargetNativeMotionReferenceV15B
    previous_token_index: torch.Tensor
    one_to_one: bool
    physical_candidate_closed: bool
    evidence_kind: str
    externally_authenticated: bool
    native_flow_claimed: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_TRANSPORT_SCHEMA:
            raise V15BContractError("target transport schema differs")
        for label, value in (
            ("source video", self.source_video_sha256),
            ("binding", self.binding_digest), ("mask", self.mask_digest),
        ):
            _sha(value, label=f"target transport {label}")
        _exact_int(self.step_index, label="target transport step")
        _exact_int(self.block_index, label="target transport block")
        if self.step_index >= DENOISE_STEPS or self.block_index >= TRANSFORMER_BLOCKS:
            raise V15BContractError("target transport execution cell differs")
        if self.branch not in CFG_BRANCHES:
            raise V15BContractError("target transport branch differs")
        for label, value in (
            ("batch", self.batch_size), ("height", self.height), ("width", self.width),
        ):
            _exact_int(value, label=f"target transport {label}", minimum=1)
        if not isinstance(self.motion_reference, TargetNativeMotionReferenceV15B):
            raise V15BContractError("transport lacks role-label material reference")
        _revalidate_material(self.motion_reference, label="transport motion reference")
        index = _tensor(
            self.previous_token_index, label="target transport index",
            ndim=3, floating=False,
        )
        if ((self.height, self.width) !=
                (self.motion_reference.height, self.motion_reference.width) or
                index.device.type != "cpu" or index.dtype != torch.int64 or
                not torch.equal(index, self.motion_reference.backward_token_index) or
                self.batch_size != int(index.shape[0])):
            raise V15BContractError("target transport is not estimator replay")
        if (self.one_to_one is not True or self.physical_candidate_closed is not True or
                self.evidence_kind != "self_contained_role_label_translation_reference" or
                self.externally_authenticated is not False or
                self.native_flow_claimed is not False or
                self.scientific_claim_authorized is not False or
                self.route_authorized is not False):
            raise V15BContractError("target transport must remain synthetic/no-authority")
        if object_sha256(self._payload()) != _sha(self.digest, label="target transport digest"):
            raise V15BContractError("target-native transport digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_video_sha256": self.source_video_sha256,
            "binding_digest": self.binding_digest, "mask_digest": self.mask_digest,
            "step_index": self.step_index, "block_index": self.block_index,
            "branch": self.branch, "batch_size": self.batch_size,
            "height": self.height, "width": self.width,
            "motion_reference_digest": self.motion_reference.digest,
            "previous_token_index_sha256": tensor_sha256(self.previous_token_index),
            "one_to_one": self.one_to_one,
            "physical_candidate_closed": self.physical_candidate_closed,
            "evidence_kind": self.evidence_kind,
            "externally_authenticated": self.externally_authenticated,
            "native_flow_claimed": self.native_flow_claimed,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "route_authorized": self.route_authorized,
        }

    @classmethod
    def create(cls, **_: Any) -> "TargetNativeTransportV15B":
        raise V15BContractError(
            "arbitrary transport indices are forbidden; use the sealed reference builder"
        )


def build_target_native_transport_v15b(
    *, source_video_sha256: str, binding: SourceActionRoleBindingV15B,
    masks: SourceRoleMaskSetV15B, step_index: int, block_index: int,
    branch: str, motion_reference: TargetNativeMotionReferenceV15B,
) -> TargetNativeTransportV15B:
    _revalidate_material(binding, label="transport builder binding")
    _revalidate_material(masks, label="transport builder masks")
    if not isinstance(motion_reference, TargetNativeMotionReferenceV15B):
        raise V15BContractError("transport builder requires synthetic role-label reference")
    _revalidate_material(motion_reference, label="transport builder motion reference")
    if (source_video_sha256 != masks.source_video_sha256 or
            binding.digest != masks.binding_digest or
            (motion_reference.height, motion_reference.width) !=
            (masks.height, masks.width)):
        raise V15BContractError("transport builder source/mask/motion authority differs")
    expected_role_ids = tuple(sorted(binding.source_roles))
    if motion_reference.role_ids != expected_role_ids:
        raise V15BContractError("transport reference role IDs differ from binding")
    spatial = masks.height * masks.width
    candidates = motion_reference.physical_candidate_mask
    role_union = torch.zeros_like(candidates[:, 0])
    for role_index, role in enumerate(expected_role_ids):
        source_phase0 = masks.role_masks[role][:, :spatial]
        role_union |= source_phase0
        if not torch.equal(
                motion_reference.role_physical_candidate_mask[:, role_index, 0],
                source_phase0):
            raise V15BContractError(
                "transport per-role candidate phase0 differs from source role mask"
            )
    if not torch.equal(candidates[:, 0], role_union):
        raise V15BContractError("native candidate phase0 is not exact source-role union")
    corridor = masks.editable_corridor_mask.reshape(
        int(candidates.shape[0]), LATENT_PHASES, spatial
    )
    if bool((candidates[:, 1:] & ~corridor[:, 1:]).any()):
        raise V15BContractError("transport physical candidates escape source-safe corridor")
    for role_index, role in enumerate(expected_role_ids):
        if bool((motion_reference.role_physical_candidate_mask[
                :, role_index, 1:] & ~corridor[:, 1:]).any()):
            raise V15BContractError(f"transport candidate {role} escapes corridor")
    index = motion_reference.backward_token_index.detach().clone()
    payload = {
        "schema_version": TARGET_TRANSPORT_SCHEMA,
        "source_video_sha256": source_video_sha256,
        "binding_digest": binding.digest, "mask_digest": masks.digest,
        "step_index": step_index, "block_index": block_index,
        "branch": branch, "batch_size": int(index.shape[0]),
        "height": masks.height, "width": masks.width,
        "motion_reference_digest": motion_reference.digest,
        "previous_token_index_sha256": tensor_sha256(index),
        "one_to_one": True, "physical_candidate_closed": True,
        "evidence_kind": "self_contained_role_label_translation_reference",
        "externally_authenticated": False, "native_flow_claimed": False,
        "scientific_claim_authorized": False, "route_authorized": False,
    }
    return TargetNativeTransportV15B(
        TARGET_TRANSPORT_SCHEMA, source_video_sha256, binding.digest, masks.digest,
        step_index, block_index, branch, int(index.shape[0]), masks.height,
        masks.width, motion_reference, index, True, True,
        "self_contained_role_label_translation_reference", False, False, False,
        False, object_sha256(payload),
    )


def _mask_centroid_float(mask_2d: torch.Tensor) -> tuple[float, float]:
    return _centroid(mask_2d)


def _background_hole_count_8(mask_2d: torch.Tensor) -> int:
    height, width = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    background = ~mask_2d
    remaining = {
        tuple(int(value) for value in row)
        for row in torch.nonzero(background, as_tuple=False)
    }
    holes = 0
    while remaining:
        seed = remaining.pop(); stack = [seed]; touches_border = False
        while stack:
            y, x = stack.pop()
            touches_border |= y in (0, height - 1) or x in (0, width - 1)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor = (y + dy, x + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor); stack.append(neighbor)
        if not touches_border:
            holes += 1
    return holes


def _boundary_edge_count_4(mask_2d: torch.Tensor) -> int:
    height, width = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    boundary = 0
    for y, x in torch.nonzero(mask_2d, as_tuple=False).tolist():
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= ny < height and 0 <= nx < width) or not bool(mask_2d[ny, nx]):
                boundary += 1
    return boundary


def _mask_geometry(mask_2d: torch.Tensor) -> dict[str, float]:
    coordinates = torch.nonzero(mask_2d, as_tuple=False).float()
    if not int(coordinates.shape[0]):
        raise V15BContractError("track geometry requires a nonempty mask")
    y_extent = float(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
    x_extent = float(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
    aspect = max(y_extent / x_extent, x_extent / y_extent)
    diameter = max(1.0, math.hypot(y_extent - 1.0, x_extent - 1.0))
    area = float(coordinates.shape[0])
    components = _component_count_4(mask_2d)
    holes = _background_hole_count_8(mask_2d)
    return {
        "area": area,
        "aspect": aspect,
        "diameter": diameter,
        "compactness": area / (y_extent * x_extent),
        "foreground_components": float(components),
        "holes": float(holes),
        "euler_characteristic": float(components - holes),
        "boundary_edges": float(_boundary_edge_count_4(mask_2d)),
    }


def _translated_mask(mask_2d: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    output = torch.zeros_like(mask_2d)
    for y, x in torch.nonzero(mask_2d, as_tuple=False).tolist():
        target_y, target_x = y + dy, x + dx
        if 0 <= target_y < int(mask_2d.shape[0]) and 0 <= target_x < int(mask_2d.shape[1]):
            output[target_y, target_x] = True
    return output


def _iou(left: torch.Tensor, right: torch.Tensor) -> float:
    union = int((left | right).sum())
    return float((left & right).sum()) / union if union else 1.0


def _hausdorff(left: torch.Tensor, right: torch.Tensor) -> float:
    left_points = torch.nonzero(left, as_tuple=False).float()
    right_points = torch.nonzero(right, as_tuple=False).float()
    if not int(left_points.shape[0]) or not int(right_points.shape[0]):
        return float("inf")
    distance = torch.cdist(left_points, right_points)
    return max(float(distance.amin(1).amax()), float(distance.amin(0).amax()))


def _connected_components_4(mask_2d: torch.Tensor) -> tuple[torch.Tensor, ...]:
    remaining = {tuple(int(value) for value in row) for row in torch.nonzero(mask_2d)}
    components = []
    while remaining:
        seed = remaining.pop(); stack = [seed]; points = [seed]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor); stack.append(neighbor); points.append(neighbor)
        component = torch.zeros_like(mask_2d)
        for y, x in points:
            component[y, x] = True
        components.append(component)
    return tuple(components)


def _track_gate_metrics(
    *, phase0: torch.Tensor, previous: torch.Tensor, candidate: torch.Tensor,
    final: torch.Tensor, corridor: torch.Tensor, role: str, phase: int,
) -> dict[str, Any]:
    if bool((candidate & ~corridor).any()) or bool((final & ~corridor).any()):
        raise V15BContractError(f"persistent slot {role} escaped source-safe corridor")
    if _component_count_4(candidate) != 1:
        raise V15BContractError(
            f"persistent slot {role} phase {phase} has dual-position candidate components"
        )
    if _component_count_4(final) != 1:
        raise V15BContractError(
            f"persistent slot {role} phase {phase} is not one 4-connected component"
        )
    phase0_geometry = _mask_geometry(phase0)
    previous_geometry = _mask_geometry(previous)
    candidate_geometry = _mask_geometry(candidate)
    final_geometry = _mask_geometry(final)
    previous_area = int(previous_geometry["area"])
    candidate_area = int(candidate_geometry["area"])
    final_area = int(final_geometry["area"])
    phase0_area = int(phase0_geometry["area"])
    if candidate_area != previous_area:
        raise V15BContractError(f"persistent slot {role} native candidate area changed")
    phase0_area_ratio = final_area / phase0_area
    if not (TARGET_TRACK_MIN_PHASE0_AREA_RATIO <= phase0_area_ratio <=
            TARGET_TRACK_MAX_PHASE0_AREA_RATIO):
        raise V15BContractError(f"persistent slot {role} phase0-relative area gate failed")
    vacancy = previous_area - final_area
    if (final_area < TARGET_TRACK_MIN_AREA_PIXELS or vacancy < 0 or
            vacancy / previous_area > TARGET_TRACK_MAX_VACANCY_FRACTION):
        raise V15BContractError(f"persistent slot {role} vacancy gate failed")
    for label, observed, reference, maximum in (
        ("aspect", final_geometry["aspect"], phase0_geometry["aspect"],
         TARGET_TRACK_MAX_ASPECT_RATIO_CHANGE),
        ("diameter", final_geometry["diameter"], phase0_geometry["diameter"],
         TARGET_TRACK_MAX_DIAMETER_RATIO),
    ):
        ratio = max(observed / reference, reference / observed)
        if ratio > maximum:
            raise V15BContractError(f"persistent slot {role} {label} gate failed")
    if (final_geometry["compactness"] /
            phase0_geometry["compactness"] < TARGET_TRACK_MIN_COMPACTNESS_RATIO):
        raise V15BContractError(f"persistent slot {role} compactness gate failed")
    for label in ("holes", "euler_characteristic"):
        if (candidate_geometry[label] != previous_geometry[label] or
                final_geometry[label] != phase0_geometry[label]):
            raise V15BContractError(
                f"persistent slot {role} topology gate failed ({label}); "
                "no authenticated occlusion evidence is present"
            )
    boundary_ratio = max(
        final_geometry["boundary_edges"] / phase0_geometry["boundary_edges"],
        phase0_geometry["boundary_edges"] / final_geometry["boundary_edges"],
    )
    if boundary_ratio > TARGET_TRACK_MAX_BOUNDARY_RATIO:
        raise V15BContractError(f"persistent slot {role} boundary gate failed")
    previous_centroid = _mask_centroid_float(previous)
    candidate_centroid = _mask_centroid_float(candidate)
    final_centroid = _mask_centroid_float(final)
    dy = math.floor(candidate_centroid[0] - previous_centroid[0] + 0.5)
    dx = math.floor(candidate_centroid[1] - previous_centroid[1] + 0.5)
    translated_previous = _translated_mask(previous, dy, dx)
    translated_iou = _iou(translated_previous, candidate)
    translated_hausdorff = _hausdorff(translated_previous, candidate)
    if (translated_iou < TARGET_TRACK_MIN_TRANSLATED_IOU or
            translated_hausdorff > TARGET_TRACK_MAX_TRANSLATED_HAUSDORFF):
        raise V15BContractError(f"persistent slot {role} transported shape gate failed")
    final_candidate_iou = _iou(final, candidate)
    final_candidate_hausdorff = _hausdorff(final, candidate)
    if (final_candidate_iou < TARGET_TRACK_MIN_TRANSLATED_IOU or
            final_candidate_hausdorff > TARGET_TRACK_MAX_TRANSLATED_HAUSDORFF):
        raise V15BContractError(f"persistent slot {role} K-retained shape gate failed")
    displacement = math.hypot(
        previous_centroid[0] - final_centroid[0],
        previous_centroid[1] - final_centroid[1],
    )
    if displacement > TARGET_TRACK_MAX_CENTROID_JUMP:
        raise V15BContractError(f"persistent slot {role} centroid jump exceeds 3.5")
    reference_displacement = math.hypot(
        previous_centroid[0] - candidate_centroid[0],
        previous_centroid[1] - candidate_centroid[1],
    )
    if (reference_displacement > TARGET_TRACK_STALE_MOTION_EPSILON and math.hypot(
            final_centroid[0] - candidate_centroid[0],
            final_centroid[1] - candidate_centroid[1],
    ) > TARGET_TRACK_MAX_TRANSLATED_HAUSDORFF):
        raise V15BContractError(f"persistent slot {role} is stale versus transport reference")
    entered = int((final & ~previous).sum())
    released = int((previous & ~final).sum())
    if entered and released < entered:
        raise V15BContractError(f"persistent slot {role} failed old-position release")
    return {
        "area": final_area, "centroid": final_centroid,
        "vacancy": vacancy, "released": released,
        "phase0_area_ratio": phase0_area_ratio,
        "aspect": final_geometry["aspect"],
        "diameter": final_geometry["diameter"],
        "compactness": final_geometry["compactness"],
        "holes": int(final_geometry["holes"]),
        "euler_characteristic": int(final_geometry["euler_characteristic"]),
        "boundary_edges": int(final_geometry["boundary_edges"]),
        "boundary_ratio_to_phase0": boundary_ratio,
        "occlusion_authenticated": False,
        "translated_iou": translated_iou,
        "translated_hausdorff": translated_hausdorff,
        "final_candidate_iou": final_candidate_iou,
        "final_candidate_hausdorff": final_candidate_hausdorff,
        "centroid_step": displacement,
        "reference_centroid_step": reference_displacement,
    }


def _track_gate(
    *, previous: torch.Tensor, candidate: torch.Tensor, final: torch.Tensor,
    corridor: torch.Tensor, role: str, phase: int,
    phase0: Optional[torch.Tensor] = None,
) -> tuple[int, tuple[float, float], int, int]:
    metrics = _track_gate_metrics(
        phase0=previous if phase0 is None else phase0,
        previous=previous, candidate=candidate, final=final,
        corridor=corridor, role=role, phase=phase,
    )
    return (
        metrics["area"], metrics["centroid"],
        metrics["vacancy"], metrics["released"],
    )


def _recompute_target_track_authority(
    *, confident: torch.Tensor, transported: torch.Tensor,
    affinity: torch.Tensor, null_affinity: torch.Tensor,
    corridor: torch.Tensor, role_physical_candidates: torch.Tensor,
    role_ids: tuple[str, ...], moving_object_role_id: str,
    height: int, width: int,
) -> dict[str, Any]:
    spatial = height * width
    corridor_4d = corridor.reshape(1, LATENT_PHASES, height, width)
    role_physical_5d = role_physical_candidates.reshape(
        1, len(role_ids), LATENT_PHASES, height, width
    )
    areas: dict[str, list[int]] = {}
    centroids: dict[str, list[tuple[float, float]]] = {}
    vacancies: dict[str, list[int]] = {}
    releases: dict[str, list[int]] = {}
    cumulative: dict[str, float] = {}
    reference_cumulative: dict[str, float] = {}
    geometry_payload: dict[str, list[dict[str, Any]]] = {}
    topology_by_role: dict[str, list[tuple[int, int, int, bool]]] = {}
    for role_index, role in enumerate(role_ids):
        phase0 = (confident[0, :spatial] == role_index).reshape(height, width)
        if _component_count_4(phase0) != 1:
            raise V15BContractError("persistent phase0 slot is not one component")
        areas[role] = [int(phase0.sum())]
        centroids[role] = [_mask_centroid_float(phase0)]
        vacancies[role] = [0]; releases[role] = [0]
        cumulative[role] = 0.0; reference_cumulative[role] = 0.0
        phase0_geometry = _mask_geometry(phase0)
        topology_by_role[role] = [(
            int(phase0_geometry["holes"]),
            int(phase0_geometry["euler_characteristic"]),
            int(phase0_geometry["boundary_edges"]), False,
        )]
        geometry_payload[role] = [{
            "phase": 0, "area": int(phase0_geometry["area"]),
            "centroid": centroids[role][0],
            "aspect": phase0_geometry["aspect"],
            "diameter": phase0_geometry["diameter"],
            "compactness": phase0_geometry["compactness"],
            "holes": int(phase0_geometry["holes"]),
            "euler_characteristic": int(phase0_geometry["euler_characteristic"]),
            "boundary_edges": int(phase0_geometry["boundary_edges"]),
            "boundary_ratio_to_phase0": 1.0,
            "occlusion_authenticated": False,
            "phase0_area_ratio": 1.0, "centroid_step": 0.0,
            "reference_centroid_step": 0.0,
        }]
        for phase in range(1, LATENT_PHASES):
            previous = (confident[
                0, (phase - 1) * spatial:phase * spatial
            ] == role_index).reshape(height, width)
            candidate = (transported[
                0, phase * spatial:(phase + 1) * spatial
            ] == role_index).reshape(height, width)
            final = (confident[
                0, phase * spatial:(phase + 1) * spatial
            ] == role_index).reshape(height, width)
            metrics = _track_gate_metrics(
                phase0=phase0, previous=previous, candidate=candidate, final=final,
                corridor=corridor_4d[0, phase], role=role, phase=phase,
            )
            areas[role].append(metrics["area"])
            centroids[role].append(metrics["centroid"])
            vacancies[role].append(metrics["vacancy"])
            releases[role].append(metrics["released"])
            cumulative[role] += metrics["centroid_step"]
            reference_cumulative[role] += metrics["reference_centroid_step"]
            geometry_payload[role].append({"phase": phase, **metrics})
            topology_by_role[role].append((
                metrics["holes"], metrics["euler_characteristic"],
                metrics["boundary_edges"], metrics["occlusion_authenticated"],
            ))
        maximum_path = TARGET_TRACK_MAX_CUMULATIVE_PATH_FACTOR * math.hypot(
            height - 1, width - 1
        )
        if (cumulative[role] > maximum_path or
                reference_cumulative[role] > maximum_path):
            raise V15BContractError(f"persistent slot {role} cumulative path gate failed")
    if (reference_cumulative[moving_object_role_id] <= TARGET_TRACK_STALE_MOTION_EPSILON or
            cumulative[moving_object_role_id] <= TARGET_TRACK_STALE_MOTION_EPSILON):
        raise V15BContractError(
            "moving-object track is completely stale versus transport reference"
        )
    ghost_components = 0
    for phase in range(1, LATENT_PHASES):
        current_slice = slice(phase * spatial, (phase + 1) * spatial)
        phase_scores = affinity[0, current_slice]
        phase_null = null_affinity[0, current_slice]
        for role_index, role in enumerate(role_ids):
            score = phase_scores[:, role_index]
            other = phase_scores.masked_fill(
                F.one_hot(
                    torch.full((spatial,), role_index, dtype=torch.int64),
                    num_classes=len(role_ids),
                ).bool(),
                -float("inf"),
            ).amax(-1)
            high = (
                corridor_4d[0, phase].reshape(-1)
                & (score >= TARGET_ROLE_MIN_COSINE)
                & ((score - phase_null) >= TARGET_ROLE_NULL_MARGIN)
                & ((score - other) >= TARGET_ROLE_WINNER_MARGIN)
            ).reshape(height, width)
            role_candidate = (transported[
                0, current_slice
            ] == role_index).reshape(height, width)
            role_physical = role_physical_5d[0, role_index, phase]
            if bool((role_candidate & ~role_physical).any()):
                raise V15BContractError(
                    f"transported candidate {role} occupies another role physical mask"
                )
            for component in _connected_components_4(high):
                if bool((component & ~role_candidate).any()):
                    ghost_components += 1
    if ghost_components:
        raise V15BContractError(
            "high-affinity role component lacks its role-specific transported support"
        )
    return {
        "role_area_by_phase": tuple((role, tuple(areas[role])) for role in role_ids),
        "role_centroid_by_phase": tuple(
            (role, tuple(centroids[role])) for role in role_ids
        ),
        "role_vacancy_by_phase": tuple(
            (role, tuple(vacancies[role])) for role in role_ids
        ),
        "role_old_position_release_by_phase": tuple(
            (role, tuple(releases[role])) for role in role_ids
        ),
        "role_cumulative_centroid_path": tuple(
            (role, cumulative[role]) for role in role_ids
        ),
        "role_reference_cumulative_centroid_path": tuple(
            (role, reference_cumulative[role]) for role in role_ids
        ),
        "track_geometry_digest": object_sha256(geometry_payload),
        "role_topology_by_phase": tuple(
            (role, tuple(topology_by_role[role])) for role in role_ids
        ),
        "ghost_candidate_component_count": ghost_components,
    }


@dataclass(frozen=True)
class TargetRoleStateV15B:
    """Persistent fixed-slot target ownership; K may retain/reject, never rename."""

    schema_version: str
    source_video_sha256: str
    binding_digest: str
    mask_digest: str
    memory_digest: str
    transport: TargetNativeTransportV15B
    step_index: int
    block_index: int
    branch: str
    height: int
    width: int
    role_ids: tuple[str, ...]
    moving_object_role_id: str
    native_target_key_sha256: str
    scrubbed_target_key: torch.Tensor
    position_projector_sha256: str
    maximum_corridor_mask: torch.Tensor
    phase0_slot_index: torch.Tensor
    transported_role_index: torch.Tensor
    role_affinity: torch.Tensor
    null_affinity: torch.Tensor
    role_weights: torch.Tensor
    confident_role_index: torch.Tensor
    assigned_token_count_by_phase: tuple[int, ...]
    unassigned_corridor_count_by_phase: tuple[int, ...]
    role_area_by_phase: tuple[tuple[str, tuple[int, ...]], ...]
    role_centroid_by_phase: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]
    role_vacancy_by_phase: tuple[tuple[str, tuple[int, ...]], ...]
    role_old_position_release_by_phase: tuple[tuple[str, tuple[int, ...]], ...]
    role_cumulative_centroid_path: tuple[tuple[str, float], ...]
    role_reference_cumulative_centroid_path: tuple[tuple[str, float], ...]
    track_geometry_digest: str
    role_topology_by_phase: tuple[
        tuple[str, tuple[tuple[int, int, int, bool], ...]], ...
    ]
    ghost_candidate_component_count: int
    physical_candidate_mask_sha256: str
    role_physical_candidate_mask_sha256: str
    position_reference_digest: str
    cross_role_rename_count: int
    corridor_escape_count: int
    dual_position_component_count: int
    min_role_cosine: float
    null_margin: float
    winner_margin: float
    assignment_kind: str
    phase0_source_masks_exact: bool
    externally_authenticated: bool
    position_removed_claimed: bool
    native_flow_claimed: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_ROLE_STATE_SCHEMA:
            raise V15BContractError("persistent target role-state schema differs")
        for label, value in (
            ("source video", self.source_video_sha256), ("binding", self.binding_digest),
            ("mask", self.mask_digest), ("memory", self.memory_digest),
            ("native target key", self.native_target_key_sha256),
            ("position projector", self.position_projector_sha256),
            ("track geometry", self.track_geometry_digest),
            ("physical candidate mask", self.physical_candidate_mask_sha256),
            ("role physical candidate mask", self.role_physical_candidate_mask_sha256),
            ("position reference", self.position_reference_digest),
        ):
            _sha(value, label=f"target role-state {label}")
        if (not isinstance(self.transport, TargetNativeTransportV15B) or
                self.transport.source_video_sha256 != self.source_video_sha256 or
                self.transport.binding_digest != self.binding_digest or
                self.transport.mask_digest != self.mask_digest):
            raise V15BContractError("persistent target role-state transport authority differs")
        _revalidate_material(self.transport, label="target role-state transport")
        if (self.step_index, self.block_index, self.branch) != (
            self.transport.step_index, self.transport.block_index, self.transport.branch
        ):
            raise V15BContractError("persistent role-state execution cell differs")
        if (self.height, self.width) != (self.transport.height, self.transport.width):
            raise V15BContractError("persistent role-state spatial geometry differs")
        if len(self.role_ids) != 4 or tuple(sorted(self.role_ids)) != self.role_ids:
            raise V15BContractError("persistent role-state roles must be four sorted fixed slots")
        if self.moving_object_role_id not in self.role_ids:
            raise V15BContractError("persistent moving-object role differs")
        scrubbed = _cpu_fp32(self.scrubbed_target_key, label="scrubbed target K", ndim=4)
        affinity = _cpu_fp32(self.role_affinity, label="target role affinity", ndim=3)
        null = _cpu_fp32(self.null_affinity, label="target null affinity", ndim=2)
        weights = _cpu_fp32(self.role_weights, label="target fixed-slot weights", ndim=3)
        corridor = _packed_mask_4d(
            self.maximum_corridor_mask, height=self.height, width=self.width,
            label="target maximum corridor",
        ).reshape(self.maximum_corridor_mask.shape)
        for label, tensor in (
            ("phase0 slot", self.phase0_slot_index),
            ("transported role", self.transported_role_index),
            ("confident role", self.confident_role_index),
        ):
            _tensor(tensor, label=f"target {label}", ndim=2, floating=False)
            if tensor.device.type != "cpu" or tensor.dtype != torch.int64:
                raise V15BContractError(f"target {label} must be CPU int64")
        confident = self.confident_role_index
        transported = self.transported_role_index
        batch, tokens = confident.shape
        if batch != 1:
            raise V15BContractError("persistent CPU role-state requires the one-source batch")
        spatial = self.height * self.width
        expected = (batch, tokens, len(self.role_ids))
        if (tokens != LATENT_PHASES * spatial or tuple(affinity.shape) != expected or
                tuple(weights.shape) != expected or tuple(null.shape) != (batch, tokens) or
                tuple(scrubbed.shape[:2]) != (batch, tokens) or
                tuple(transported.shape) != (batch, tokens) or
                tuple(self.phase0_slot_index.shape) != (batch, spatial) or
                tuple(self.maximum_corridor_mask.shape) != (batch, tokens)):
            raise V15BContractError("persistent target role-state tensor geometry differs")
        if not torch.equal(confident[:, :spatial], self.phase0_slot_index):
            raise V15BContractError("persistent role-state phase0 slot initialization differs")
        if not torch.equal(transported[:, :spatial], self.phase0_slot_index):
            raise V15BContractError("persistent role-state phase0 transport boundary differs")
        for phase in range(1, LATENT_PHASES):
            mapping = self.transport.previous_token_index[:, phase - 1]
            previous = confident[:, (phase - 1) * spatial:phase * spatial]
            inherited = torch.gather(previous, 1, mapping.clamp_min(0))
            inherited = inherited.masked_fill(mapping < 0, -1)
            current = transported[:, phase * spatial:(phase + 1) * spatial]
            if not torch.equal(current, inherited):
                raise V15BContractError("persistent slot identity was not propagated by transport")
        assigned = confident >= 0
        if bool((confident < -1).any()) or bool((confident >= len(self.role_ids)).any()):
            raise V15BContractError("persistent confident slot index is out of range")
        if bool((assigned & (confident != transported)).any()):
            raise V15BContractError("current target K renamed a persistent slot")
        expected_weights = F.one_hot(
            confident.clamp_min(0), num_classes=len(self.role_ids)
        ).float() * assigned[..., None]
        if not torch.equal(weights, expected_weights):
            raise V15BContractError("persistent slot weights are not exact one-hot/unassigned")
        inherited = transported >= 0
        inherited_index = transported.clamp_min(0)
        inherited_score = affinity.gather(-1, inherited_index[..., None]).squeeze(-1)
        competitor = affinity.masked_fill(
            F.one_hot(inherited_index, num_classes=len(self.role_ids)).bool(),
            -float("inf"),
        ).amax(-1)
        keep = (
            inherited & (inherited_score >= self.min_role_cosine)
            & ((inherited_score - null) >= self.null_margin)
            & ((inherited_score - competitor) >= self.winner_margin)
        )
        keep[:, :spatial] = inherited[:, :spatial]
        if not torch.equal(assigned, keep):
            raise V15BContractError("persistent slot confidence/unassigned decision differs")
        if bool((assigned[:, spatial:] & ~corridor[:, spatial:]).any()):
            raise V15BContractError("persistent editable support escaped maximum corridor")
        if (self.cross_role_rename_count != 0 or self.corridor_escape_count != 0 or
                self.dual_position_component_count != 0 or
                self.ghost_candidate_component_count != 0):
            raise V15BContractError("persistent slot invariant counter must be exactly zero")
        for count in (
            self.cross_role_rename_count, self.corridor_escape_count,
            self.dual_position_component_count, self.ghost_candidate_component_count,
        ):
            _exact_int(count, label="persistent invariant count")
        recomputed_assigned = tuple(
            int(assigned[:, p * spatial:(p + 1) * spatial].sum())
            for p in range(LATENT_PHASES)
        )
        recomputed_unassigned = tuple(
            int((corridor[:, p * spatial:(p + 1) * spatial] &
                 ~assigned[:, p * spatial:(p + 1) * spatial]).sum())
            for p in range(LATENT_PHASES)
        )
        if (recomputed_assigned != self.assigned_token_count_by_phase or
                recomputed_unassigned != self.unassigned_corridor_count_by_phase):
            raise V15BContractError("persistent role-state phase counts differ")
        physical = self.transport.motion_reference.physical_candidate_mask
        if tensor_sha256(physical) != self.physical_candidate_mask_sha256:
            raise V15BContractError("persistent physical candidate authority differs")
        role_physical = self.transport.motion_reference.role_physical_candidate_mask
        if tensor_sha256(role_physical) != self.role_physical_candidate_mask_sha256:
            raise V15BContractError("persistent per-role physical candidate authority differs")
        recomputed_track = _recompute_target_track_authority(
            confident=confident, transported=transported, affinity=affinity,
            null_affinity=null, corridor=corridor,
            role_physical_candidates=role_physical, role_ids=self.role_ids,
            moving_object_role_id=self.moving_object_role_id,
            height=self.height, width=self.width,
        )
        if any(getattr(self, name) != recomputed_track[name] for name in (
            "role_area_by_phase", "role_centroid_by_phase",
            "role_vacancy_by_phase", "role_old_position_release_by_phase",
            "role_cumulative_centroid_path",
            "role_reference_cumulative_centroid_path", "track_geometry_digest",
            "role_topology_by_phase",
            "ghost_candidate_component_count",
        )):
            raise V15BContractError("persistent role track statistics were not recomputed")
        if (self.min_role_cosine != TARGET_ROLE_MIN_COSINE or
                self.null_margin != TARGET_ROLE_NULL_MARGIN or
                self.winner_margin != TARGET_ROLE_WINNER_MARGIN or
                self.assignment_kind !=
                "previous_owner_synthetic_transport_reference_then_k_retain_or_unassign"):
            raise V15BContractError("persistent role-state assignment contract differs")
        if self.phase0_source_masks_exact is not True:
            raise V15BContractError("persistent role-state phase0 source-mask boundary differs")
        if (self.externally_authenticated is not False or
                self.position_removed_claimed is not False or
                self.native_flow_claimed is not False or
                self.scientific_claim_authorized is not False or
                self.route_authorized is not False):
            raise V15BContractError("persistent role-state must remain reference-only")
        if object_sha256(self._payload()) != _sha(self.digest, label="target role-state digest"):
            raise V15BContractError("persistent target role-state digest differs")

    @property
    def assigned_support_mask(self) -> torch.Tensor:
        return self.confident_role_index >= 0

    @property
    def confident_role_weights(self) -> torch.Tensor:
        return self.role_weights

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_video_sha256": self.source_video_sha256,
            "binding_digest": self.binding_digest, "mask_digest": self.mask_digest,
            "memory_digest": self.memory_digest, "transport_digest": self.transport.digest,
            "step_index": self.step_index, "block_index": self.block_index,
            "branch": self.branch, "height": self.height, "width": self.width,
            "role_ids": self.role_ids, "moving_object_role_id": self.moving_object_role_id,
            "native_target_key_sha256": self.native_target_key_sha256,
            "scrubbed_target_key_sha256": tensor_sha256(self.scrubbed_target_key),
            "position_projector_sha256": self.position_projector_sha256,
            "maximum_corridor_mask_sha256": tensor_sha256(self.maximum_corridor_mask),
            "phase0_slot_index_sha256": tensor_sha256(self.phase0_slot_index),
            "transported_role_index_sha256": tensor_sha256(self.transported_role_index),
            "role_affinity_sha256": tensor_sha256(self.role_affinity),
            "null_affinity_sha256": tensor_sha256(self.null_affinity),
            "role_weights_sha256": tensor_sha256(self.role_weights),
            "confident_role_index_sha256": tensor_sha256(self.confident_role_index),
            "assigned_token_count_by_phase": self.assigned_token_count_by_phase,
            "unassigned_corridor_count_by_phase": self.unassigned_corridor_count_by_phase,
            "role_area_by_phase": self.role_area_by_phase,
            "role_centroid_by_phase": self.role_centroid_by_phase,
            "role_vacancy_by_phase": self.role_vacancy_by_phase,
            "role_old_position_release_by_phase": self.role_old_position_release_by_phase,
            "role_cumulative_centroid_path": self.role_cumulative_centroid_path,
            "role_reference_cumulative_centroid_path": (
                self.role_reference_cumulative_centroid_path
            ),
            "track_geometry_digest": self.track_geometry_digest,
            "role_topology_by_phase": self.role_topology_by_phase,
            "ghost_candidate_component_count": self.ghost_candidate_component_count,
            "physical_candidate_mask_sha256": self.physical_candidate_mask_sha256,
            "role_physical_candidate_mask_sha256": (
                self.role_physical_candidate_mask_sha256
            ),
            "position_reference_digest": self.position_reference_digest,
            "cross_role_rename_count": self.cross_role_rename_count,
            "corridor_escape_count": self.corridor_escape_count,
            "dual_position_component_count": self.dual_position_component_count,
            "min_role_cosine": self.min_role_cosine, "null_margin": self.null_margin,
            "winner_margin": self.winner_margin, "assignment_kind": self.assignment_kind,
            "phase0_source_masks_exact": self.phase0_source_masks_exact,
            "externally_authenticated": self.externally_authenticated,
            "position_removed_claimed": self.position_removed_claimed,
            "native_flow_claimed": self.native_flow_claimed,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "route_authorized": self.route_authorized,
        }


def build_target_role_state_v15b(
    *, native_target_pre_rope_key: torch.Tensor,
    memory: SourceRoleContentMemoryV15B,
    masks: SourceRoleMaskSetV15B,
    binding: SourceActionRoleBindingV15B,
    target_native_transport: TargetNativeTransportV15B,
) -> TargetRoleStateV15B:
    _revalidate_material(binding, label="target role-state binding")
    _revalidate_material(masks, label="target role-state masks")
    _revalidate_material(memory, label="target role-state memory")
    _revalidate_material(target_native_transport, label="target role-state transport")
    _validate_raw_source_material_against_masks_v15b(
        memory.raw_source_material, masks, binding
    )
    raw_key = _cpu_fp32(
        native_target_pre_rope_key, label="native target pre-RoPE key", ndim=4
    )
    if (memory.source_video_sha256 != masks.source_video_sha256 or
            memory.binding_digest != binding.digest or memory.mask_digest != masks.digest or
            tuple(raw_key.shape[:2]) != tuple(masks.editable_corridor_mask.shape) or
            tuple(raw_key.shape[2:]) != tuple(memory.key_content.shape[2:])):
        raise V15BContractError("persistent role-state source/memory/mask/K authority differs")
    if (not isinstance(target_native_transport, TargetNativeTransportV15B) or
            target_native_transport.source_video_sha256 != memory.source_video_sha256 or
            target_native_transport.binding_digest != binding.digest or
            target_native_transport.mask_digest != masks.digest or
            target_native_transport.batch_size != int(raw_key.shape[0]) or
            (target_native_transport.step_index, target_native_transport.block_index,
             target_native_transport.branch) !=
            (memory.step_index, memory.block_index, memory.branch)):
        raise V15BContractError("persistent role-state target transport authority differs")
    key = _scrub_position_subspace(raw_key, memory.position_reference.projector)
    target_unit = key / torch.linalg.vector_norm(key, dim=-1, keepdim=True).clamp_min(1e-8)
    basis_unit = memory.key_content / torch.linalg.vector_norm(
        memory.key_content, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    similarity = torch.einsum("blhd,rmhd->blrmh", target_unit, basis_unit)
    similarity = similarity.reshape(
        int(key.shape[0]), int(key.shape[1]), len(memory.role_ids),
        int(memory.key_content.shape[1]), int(key.shape[2]),
    ).mean(-1)
    valid = memory.slot_valid_mask.reshape(1, 1, *memory.slot_valid_mask.shape)
    role_affinity = similarity.masked_fill(~valid, -float("inf")).amax(-1)
    null_unit = memory.null_key_content / torch.linalg.vector_norm(
        memory.null_key_content, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    null_affinity = torch.einsum(
        "blhd,nhd->blnh", target_unit, null_unit
    ).mean(-1).amax(-1)
    spatial = masks.height * masks.width
    batch = int(key.shape[0])
    phase0_roles = torch.stack(
        [masks.role_masks[role][:, :spatial] for role in memory.role_ids], dim=-1
    )
    if bool((phase0_roles.sum(-1) > 1).any()):
        raise V15BContractError("phase0 persistent source slots overlap")
    phase0_slot = torch.full((batch, spatial), -1, dtype=torch.int64)
    phase0_owned = phase0_roles.any(-1)
    phase0_slot[phase0_owned] = phase0_roles.long().argmax(-1)[phase0_owned]
    transported = torch.full((batch, LATENT_PHASES * spatial), -1, dtype=torch.int64)
    confident = transported.clone()
    transported[:, :spatial] = phase0_slot
    confident[:, :spatial] = phase0_slot
    role_areas = {role: [int(phase0_roles[..., index].sum())]
                  for index, role in enumerate(memory.role_ids)}
    role_centroids = {
        role: [_mask_centroid_float(
            phase0_roles[0, ..., index].reshape(masks.height, masks.width)
        )]
        for index, role in enumerate(memory.role_ids)
    }
    role_vacancies = {role: [0] for role in memory.role_ids}
    role_releases = {role: [0] for role in memory.role_ids}
    corridor_4d = masks.editable_corridor_mask.reshape(
        batch, LATENT_PHASES, masks.height, masks.width
    )
    for phase in range(1, LATENT_PHASES):
        mapping = target_native_transport.previous_token_index[:, phase - 1]
        previous_owner = confident[:, (phase - 1) * spatial:phase * spatial]
        inherited = torch.gather(previous_owner, 1, mapping.clamp_min(0))
        inherited = inherited.masked_fill(mapping < 0, -1)
        current_slice = slice(phase * spatial, (phase + 1) * spatial)
        transported[:, current_slice] = inherited
        inherited_index = inherited.clamp_min(0)
        phase_affinity = role_affinity[:, current_slice]
        inherited_score = phase_affinity.gather(
            -1, inherited_index[..., None]
        ).squeeze(-1)
        competitor = phase_affinity.masked_fill(
            F.one_hot(inherited_index, num_classes=len(memory.role_ids)).bool(),
            -float("inf"),
        ).amax(-1)
        keep = (
            (inherited >= 0)
            & (inherited_score >= TARGET_ROLE_MIN_COSINE)
            & ((inherited_score - null_affinity[:, current_slice]) >= TARGET_ROLE_NULL_MARGIN)
            & ((inherited_score - competitor) >= TARGET_ROLE_WINNER_MARGIN)
        )
        phase_confident = inherited.masked_fill(~keep, -1)
        confident[:, current_slice] = phase_confident
        for role_index, role in enumerate(memory.role_ids):
            previous_mask = (previous_owner[0] == role_index).reshape(
                masks.height, masks.width
            )
            candidate_mask = (inherited[0] == role_index).reshape(
                masks.height, masks.width
            )
            final_mask = (phase_confident[0] == role_index).reshape(
                masks.height, masks.width
            )
            area, centroid, vacancy, released = _track_gate(
                previous=previous_mask, candidate=candidate_mask, final=final_mask,
                corridor=corridor_4d[0, phase], role=role, phase=phase,
                phase0=phase0_roles[0, ..., role_index].reshape(
                    masks.height, masks.width
                ),
            )
            role_areas[role].append(area); role_centroids[role].append(centroid)
            role_vacancies[role].append(vacancy); role_releases[role].append(released)
    assigned = confident >= 0
    weights = F.one_hot(confident.clamp_min(0), num_classes=len(memory.role_ids)).float()
    weights *= assigned[..., None]
    diagnostic_corridor = masks.editable_corridor_mask.clone()
    diagnostic_corridor[:, :spatial] = False
    role_affinity = role_affinity.masked_fill(~diagnostic_corridor[..., None], 0.0)
    null_affinity = null_affinity.masked_fill(~diagnostic_corridor, 0.0)
    track_authority = _recompute_target_track_authority(
        confident=confident, transported=transported, affinity=role_affinity,
        null_affinity=null_affinity, corridor=masks.editable_corridor_mask,
        role_physical_candidates=(
            target_native_transport.motion_reference.role_physical_candidate_mask
        ),
        role_ids=memory.role_ids,
        moving_object_role_id=binding.moving_object_source_role,
        height=masks.height, width=masks.width,
    )
    # Phase0 confidence is a hard source boundary and therefore bypasses K gates.
    assigned_counts = tuple(
        int(assigned[:, p * spatial:(p + 1) * spatial].sum())
        for p in range(LATENT_PHASES)
    )
    unassigned_counts = tuple(
        int((masks.editable_corridor_mask[:, p * spatial:(p + 1) * spatial] &
             ~assigned[:, p * spatial:(p + 1) * spatial]).sum())
        for p in range(LATENT_PHASES)
    )
    constructor = {
        "schema_version": TARGET_ROLE_STATE_SCHEMA,
        "source_video_sha256": memory.source_video_sha256,
        "binding_digest": binding.digest, "mask_digest": masks.digest,
        "memory_digest": memory.digest, "transport": target_native_transport,
        "step_index": memory.step_index, "block_index": memory.block_index,
        "branch": memory.branch, "height": masks.height, "width": masks.width,
        "role_ids": memory.role_ids,
        "moving_object_role_id": binding.moving_object_source_role,
        "native_target_key_sha256": tensor_sha256(raw_key),
        "scrubbed_target_key": key,
        "position_projector_sha256": tensor_sha256(memory.position_reference.projector),
        "maximum_corridor_mask": masks.editable_corridor_mask.clone(),
        "phase0_slot_index": phase0_slot,
        "transported_role_index": transported,
        "role_affinity": role_affinity, "null_affinity": null_affinity,
        "role_weights": weights, "confident_role_index": confident,
        "assigned_token_count_by_phase": assigned_counts,
        "unassigned_corridor_count_by_phase": unassigned_counts,
        "role_area_by_phase": track_authority["role_area_by_phase"],
        "role_centroid_by_phase": track_authority["role_centroid_by_phase"],
        "role_vacancy_by_phase": track_authority["role_vacancy_by_phase"],
        "role_old_position_release_by_phase": (
            track_authority["role_old_position_release_by_phase"]
        ),
        "role_cumulative_centroid_path": (
            track_authority["role_cumulative_centroid_path"]
        ),
        "role_reference_cumulative_centroid_path": (
            track_authority["role_reference_cumulative_centroid_path"]
        ),
        "track_geometry_digest": track_authority["track_geometry_digest"],
        "role_topology_by_phase": track_authority["role_topology_by_phase"],
        "ghost_candidate_component_count": (
            track_authority["ghost_candidate_component_count"]
        ),
        "physical_candidate_mask_sha256": tensor_sha256(
            target_native_transport.motion_reference.physical_candidate_mask
        ),
        "role_physical_candidate_mask_sha256": tensor_sha256(
            target_native_transport.motion_reference.role_physical_candidate_mask
        ),
        "position_reference_digest": memory.position_reference.digest,
        "cross_role_rename_count": 0, "corridor_escape_count": 0,
        "dual_position_component_count": 0,
        "min_role_cosine": TARGET_ROLE_MIN_COSINE,
        "null_margin": TARGET_ROLE_NULL_MARGIN,
        "winner_margin": TARGET_ROLE_WINNER_MARGIN,
        "assignment_kind": (
            "previous_owner_synthetic_transport_reference_then_k_retain_or_unassign"
        ),
        "phase0_source_masks_exact": True,
        "externally_authenticated": False, "position_removed_claimed": False,
        "native_flow_claimed": False, "scientific_claim_authorized": False,
        "route_authorized": False,
    }
    payload = dict(constructor)
    payload.pop("transport")
    payload["transport_digest"] = target_native_transport.digest
    for name in (
        "scrubbed_target_key", "maximum_corridor_mask", "phase0_slot_index",
        "transported_role_index", "role_affinity", "null_affinity",
        "role_weights", "confident_role_index",
    ):
        payload[f"{name}_sha256"] = tensor_sha256(payload.pop(name))
    return TargetRoleStateV15B(**constructor, digest=object_sha256(payload))


@dataclass(frozen=True)
class SourceBackgroundCarrierV15B:
    """Raw-exact phase 0 plus caller-supplied post-phase background/support."""

    schema_version: str
    source_video_sha256: str
    binding_digest: str
    mask_digest: str
    step_index: int
    block_index: int
    branch: str
    hidden: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    masks: SourceRoleMaskSetV15B
    raw_source_material: SourcePhase0RawMaterialV15B
    phase0_authority_kind: str
    phase0_raw_hkv_exact: bool
    post_phase0_background_authority_kind: str
    post_phase0_background_caller_supplied: bool
    externally_authenticated: bool
    scientific_claim_authorized: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != BACKGROUND_SCHEMA:
            raise V15BContractError("background carrier schema differs")
        _sha(self.source_video_sha256, label="background source video")
        _sha(self.binding_digest, label="background binding"); _sha(self.mask_digest, label="background mask")
        _exact_int(self.step_index, label="background step"); _exact_int(self.block_index, label="background block")
        if self.step_index >= DENOISE_STEPS or self.block_index >= TRANSFORMER_BLOCKS:
            raise V15BContractError("background step/block is outside exact execution geometry")
        if self.branch not in CFG_BRANCHES:
            raise V15BContractError("background branch differs")
        hidden = _cpu_fp32(self.hidden, label="background hidden", ndim=3)
        key = _cpu_fp32(self.key, label="background key", ndim=4)
        value = _cpu_fp32(self.value, label="background value", ndim=4)
        if tuple(key.shape) != tuple(value.shape) or tuple(hidden.shape[:2]) != tuple(key.shape[:2]):
            raise V15BContractError("background H/K/V geometry differs")
        if hidden.dtype != key.dtype or key.dtype != value.dtype or hidden.device != key.device or key.device != value.device:
            raise V15BContractError("background H/K/V dtype/device differs")
        if (not isinstance(self.masks, SourceRoleMaskSetV15B) or
                self.masks.digest != self.mask_digest or
                self.masks.source_video_sha256 != self.source_video_sha256 or
                self.masks.binding_digest != self.binding_digest):
            raise V15BContractError("background mask authority differs")
        _revalidate_material(self.masks, label="background carrier masks")
        if not isinstance(self.raw_source_material, SourcePhase0RawMaterialV15B):
            raise V15BContractError("background carrier lacks raw source material")
        _validate_raw_source_material_against_masks_v15b(
            self.raw_source_material, self.masks
        )
        if (self.raw_source_material.step_index,
                self.raw_source_material.block_index,
                self.raw_source_material.branch) != (
                    self.step_index, self.block_index, self.branch
                ):
            raise V15BContractError("background/raw source execution cell differs")
        if tuple(self.masks.background_support_mask.shape) != tuple(hidden.shape[:2]):
            raise V15BContractError("background carrier token geometry differs")
        if (
            self.phase0_authority_kind !=
            "embedded_immutable_raw_source_material_hkv"
            or self.phase0_raw_hkv_exact is not True
            or self.post_phase0_background_authority_kind !=
            "caller_supplied_same_coordinate_source_background_support"
            or self.post_phase0_background_caller_supplied is not True
            or self.externally_authenticated is not False
            or self.scientific_claim_authorized is not False
            or self.route_authorized is not False
        ):
            raise V15BContractError(
                "background carrier exceeded its internal CPU authority"
            )
        self.reopen_phase0_raw_hkv()
        corridor = self.masks.editable_corridor_mask.clone()
        corridor[:, :self.masks.height * self.masks.width] = False
        for label, tensor in (("hidden", hidden), ("key", key), ("value", value)):
            zeros = torch.zeros_like(tensor)
            if _masked_max_abs(tensor, zeros, corridor) != 0.0:
                raise V15BContractError(
                    f"background carrier must scrub post-phase0 same-coordinate object {label}"
                )
        if object_sha256(self._payload()) != _sha(self.digest, label="background digest"):
            raise V15BContractError("background carrier digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "source_video_sha256": self.source_video_sha256,
            "binding_digest": self.binding_digest, "mask_digest": self.mask_digest,
            "step_index": self.step_index, "block_index": self.block_index, "branch": self.branch,
            "hidden_sha256": tensor_sha256(self.hidden), "key_sha256": tensor_sha256(self.key),
            "value_sha256": tensor_sha256(self.value),
            "raw_source_material_digest": self.raw_source_material.digest,
            "phase0_authority_kind": self.phase0_authority_kind,
            "phase0_raw_hkv_exact": self.phase0_raw_hkv_exact,
            "post_phase0_background_authority_kind": (
                self.post_phase0_background_authority_kind
            ),
            "post_phase0_background_caller_supplied": (
                self.post_phase0_background_caller_supplied
            ),
            "externally_authenticated": self.externally_authenticated,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "route_authorized": self.route_authorized,
        }

    def reopen_phase0_raw_hkv(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Freshly reopen and compare the only allowed full-source identity row."""
        raw_hidden, raw_key, raw_value, _ = self.raw_source_material.reopen()
        spatial = self.masks.height * self.masks.width
        if (
            tuple(raw_hidden.shape) != tuple(self.hidden[:, :spatial].shape)
            or tuple(raw_key.shape) != tuple(self.key[:, :spatial].shape)
            or tuple(raw_value.shape) != tuple(self.value[:, :spatial].shape)
            or not torch.equal(self.hidden[:, :spatial], raw_hidden)
            or not torch.equal(self.key[:, :spatial], raw_key)
            or not torch.equal(self.value[:, :spatial], raw_value)
        ):
            raise V15BContractError(
                "background carrier phase0 H/K/V differs from freshly reopened raw material"
            )
        return raw_hidden, raw_key, raw_value

    @classmethod
    def create(cls, *, source_video_sha256: str, binding: SourceActionRoleBindingV15B,
               masks: SourceRoleMaskSetV15B, step_index: int, block_index: int,
               branch: str, hidden: torch.Tensor, key: torch.Tensor,
               value: torch.Tensor,
               raw_source_material: SourcePhase0RawMaterialV15B,
               ) -> "SourceBackgroundCarrierV15B":
        _revalidate_material(binding, label="background builder binding")
        _revalidate_material(masks, label="background builder masks")
        logical_hidden = _cpu_fp32(
            hidden, label="background hidden create", ndim=3
        ).detach().clone()
        logical_key = _cpu_fp32(
            key, label="background key create", ndim=4
        ).detach().clone()
        logical_value = _cpu_fp32(
            value, label="background value create", ndim=4
        ).detach().clone()
        if not isinstance(raw_source_material, SourcePhase0RawMaterialV15B):
            raise V15BContractError("background builder lacks raw source material")
        _validate_raw_source_material_against_masks_v15b(
            raw_source_material, masks, binding
        )
        if (raw_source_material.step_index, raw_source_material.block_index,
                raw_source_material.branch) != (step_index, block_index, branch):
            raise V15BContractError("background/raw source execution cell differs")
        raw_hidden, raw_key, raw_value, _ = raw_source_material.reopen()
        spatial = masks.height * masks.width
        if (
            tuple(logical_hidden[:, :spatial].shape) != tuple(raw_hidden.shape)
            or tuple(logical_key[:, :spatial].shape) != tuple(raw_key.shape)
            or tuple(logical_value[:, :spatial].shape) != tuple(raw_value.shape)
            or not torch.equal(logical_hidden[:, :spatial], raw_hidden)
            or not torch.equal(logical_key[:, :spatial], raw_key)
            or not torch.equal(logical_value[:, :spatial], raw_value)
        ):
            raise V15BContractError(
                "background builder phase0 H/K/V differs from freshly reopened raw material"
            )
        payload = {
            "schema_version": BACKGROUND_SCHEMA,
            "source_video_sha256": source_video_sha256,
            "binding_digest": binding.digest, "mask_digest": masks.digest,
            "step_index": step_index, "block_index": block_index, "branch": branch,
            "hidden_sha256": tensor_sha256(logical_hidden),
            "key_sha256": tensor_sha256(logical_key),
            "value_sha256": tensor_sha256(logical_value),
            "raw_source_material_digest": raw_source_material.digest,
            "phase0_authority_kind": "embedded_immutable_raw_source_material_hkv",
            "phase0_raw_hkv_exact": True,
            "post_phase0_background_authority_kind": (
                "caller_supplied_same_coordinate_source_background_support"
            ),
            "post_phase0_background_caller_supplied": True,
            "externally_authenticated": False,
            "scientific_claim_authorized": False,
            "route_authorized": False,
        }
        return cls(
            schema_version=BACKGROUND_SCHEMA,
            source_video_sha256=source_video_sha256,
            binding_digest=binding.digest, mask_digest=masks.digest,
            step_index=step_index, block_index=block_index, branch=branch,
            hidden=logical_hidden, key=logical_key, value=logical_value,
            masks=masks, raw_source_material=raw_source_material,
            phase0_authority_kind=(
                "embedded_immutable_raw_source_material_hkv"
            ),
            phase0_raw_hkv_exact=True,
            post_phase0_background_authority_kind=(
                "caller_supplied_same_coordinate_source_background_support"
            ),
            post_phase0_background_caller_supplied=True,
            externally_authenticated=False,
            scientific_claim_authorized=False, route_authorized=False,
            digest=object_sha256(payload),
        )


@dataclass(frozen=True)
class SignedEditGraphV15B:
    schema_version: str
    action_id: str
    roles: tuple[str, ...]
    anchor_slot: str
    source_graph_digest: str
    anchor_graph_digest: str
    source_warp_digest: str
    anchor_warp_digest: str
    add_component: torch.Tensor
    remove_component: torch.Tensor
    graph: torch.Tensor
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != SIGNED_GRAPH_SCHEMA or self.roles != SIGNED_ROLES:
            raise V15BContractError("signed graph schema/roles differ")
        _role(self.action_id, label="signed graph action"); _slot(self.anchor_slot, label="signed graph slot")
        for label, value in (("source graph", self.source_graph_digest),
                             ("anchor graph", self.anchor_graph_digest),
                             ("source warp", self.source_warp_digest),
                             ("anchor warp", self.anchor_warp_digest)):
            _sha(value, label=label)
        add = _validate_relation_tensor(self.add_component, roles=SIGNED_ROLES, label="signed add component")
        remove = _validate_relation_tensor(self.remove_component, roles=SIGNED_ROLES, label="signed remove component")
        graph = _validate_relation_tensor(self.graph, roles=SIGNED_ROLES, label="signed edit graph")
        if not torch.equal(graph, add - remove):
            raise V15BContractError("signed graph must equal add minus remove exactly")
        allowed = ACTION_ALLOWED_ADD_EDGES.get(self.action_id)
        allowed_remove = ACTION_ALLOWED_REMOVE_EDGES.get(self.action_id)
        required_add = ACTION_REQUIRED_ADD_EDGES.get(self.action_id)
        required_remove = ACTION_REQUIRED_REMOVE_EDGES.get(self.action_id)
        if None in (allowed, allowed_remove, required_add, required_remove):
            raise V15BContractError("signed graph action registry is incomplete")
        _require_registered_edges(
            graph=add, roles=SIGNED_ROLES, edges=required_add,
            label="signed add component",
        )
        _require_registered_edges(
            graph=remove, roles=SIGNED_ROLES, edges=required_remove,
            label="signed remove component",
        )
        for query_role in SIGNED_ROLES:
            for key_role in SIGNED_ROLES:
                query_index = SIGNED_ROLES.index(query_role)
                key_index = SIGNED_ROLES.index(key_role)
                if (query_role, key_role) not in allowed and int(torch.count_nonzero(
                    add[:, :, query_index, :, key_index]
                )):
                    raise V15BContractError(
                        f"signed add contains disallowed {query_role}->{key_role} edge"
                    )
                if ((query_role, key_role) not in allowed_remove
                        and int(torch.count_nonzero(
                            remove[:, :, query_index, :, key_index]
                        ))):
                    raise V15BContractError(
                        "signed remove contains a non-old interaction edge"
                    )
        if object_sha256(self._payload()) != _sha(self.digest, label="signed graph digest"):
            raise V15BContractError("signed graph digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "action_id": self.action_id,
            "roles": self.roles, "anchor_slot": self.anchor_slot,
            "source_graph_digest": self.source_graph_digest,
            "anchor_graph_digest": self.anchor_graph_digest,
            "source_warp_digest": self.source_warp_digest,
            "anchor_warp_digest": self.anchor_warp_digest,
            "add_sha256": tensor_sha256(self.add_component),
            "remove_sha256": tensor_sha256(self.remove_component),
            "graph_sha256": tensor_sha256(self.graph),
        }

    @property
    def disallowed_add_edge_max_abs(self) -> float:
        allowed = ACTION_ALLOWED_ADD_EDGES[self.action_id]
        maximum = 0.0
        for query_role in SIGNED_ROLES:
            for key_role in SIGNED_ROLES:
                if (query_role, key_role) not in allowed:
                    edge = self.add_component[
                        :, :, SIGNED_ROLES.index(query_role), :,
                        SIGNED_ROLES.index(key_role)
                    ]
                    maximum = max(maximum, float(edge.abs().max()))
        return maximum

    @property
    def disallowed_remove_edge_max_abs(self) -> float:
        allowed = ACTION_ALLOWED_REMOVE_EDGES[self.action_id]
        maximum = 0.0
        for query_role in SIGNED_ROLES:
            for key_role in SIGNED_ROLES:
                if (query_role, key_role) not in allowed:
                    edge = self.remove_component[
                        :, :, SIGNED_ROLES.index(query_role), :,
                        SIGNED_ROLES.index(key_role)
                    ]
                    maximum = max(maximum, float(edge.abs().max()))
        return maximum


def build_signed_edit_graph_v15b(
    *, source_graph: SourceRelationGraphV15B, anchor_bank: AnchorGraphBankV15B,
    source_warp: MonotonicEventWarpV15B, anchor_warp: MonotonicEventWarpV15B,
) -> SignedEditGraphV15B:
    if source_graph.action_id != anchor_bank.action_id:
        raise V15BContractError("source/anchor action differs")
    if anchor_warp.source_slot != anchor_bank.anchor_slot:
        raise V15BContractError("anchor graph/warp slot differs")
    if (anchor_warp.source_trace_digest != anchor_bank.timing_trace_digest or
            source_warp.source_trace_digest != source_graph.timing_trace_digest):
        raise V15BContractError("source/anchor graph warp-trace authority differs")
    if source_warp.canonical_slot != anchor_warp.canonical_slot:
        raise V15BContractError("source/target graph canonical timing differs")
    source_aligned = _warp_graph(source_graph.graph, source_warp)
    anchor_aligned = _warp_graph(anchor_bank.relation_graph.graph, anchor_warp)
    heads = int(source_aligned.shape[0])
    if int(anchor_aligned.shape[0]) != heads:
        raise V15BContractError("source/anchor graph head count differs")
    add = torch.zeros(heads, LATENT_PHASES, len(SIGNED_ROLES), LATENT_PHASES,
                      len(SIGNED_ROLES), dtype=torch.float32)
    # Embed only action-critical edges; donor co-occurrence edges are exact zero.
    signed_index = {name: SIGNED_ROLES.index(name) for name in GENERIC_ROLES}
    allowed = ACTION_ALLOWED_ADD_EDGES.get(source_graph.action_id)
    if allowed is None:
        raise V15BContractError("source action has no registered add-edge contract")
    for query_role, key_role in allowed:
        add[:, :, signed_index[query_role], :, signed_index[key_role]] = (
            anchor_aligned[:, :, GENERIC_ROLES.index(query_role), :,
                           GENERIC_ROLES.index(key_role)]
        )
    remove = torch.zeros_like(add)
    allowed_remove = ACTION_ALLOWED_REMOVE_EDGES.get(source_graph.action_id)
    if allowed_remove is None:
        raise V15BContractError("source action has no registered remove-edge contract")
    for query_role, key_role in allowed_remove:
        remove[:, :, SIGNED_ROLES.index(query_role), :, SIGNED_ROLES.index(key_role)] = (
            source_aligned[:, :, SIGNED_ROLES.index(query_role), :,
                           SIGNED_ROLES.index(key_role)]
        )
    add = _project_relation_graph(add); remove = _project_relation_graph(remove)
    graph = add - remove
    payload = {
        "schema_version": SIGNED_GRAPH_SCHEMA, "action_id": source_graph.action_id,
        "roles": SIGNED_ROLES, "anchor_slot": anchor_bank.anchor_slot,
        "source_graph_digest": source_graph.digest,
        "anchor_graph_digest": anchor_bank.digest,
        "source_warp_digest": source_warp.digest,
        "anchor_warp_digest": anchor_warp.digest,
        "add_sha256": tensor_sha256(add), "remove_sha256": tensor_sha256(remove),
        "graph_sha256": tensor_sha256(graph),
    }
    return SignedEditGraphV15B(
        SIGNED_GRAPH_SCHEMA, source_graph.action_id, SIGNED_ROLES,
        anchor_bank.anchor_slot, source_graph.digest, anchor_bank.digest,
        source_warp.digest, anchor_warp.digest, add, remove, graph,
        object_sha256(payload),
    )


@dataclass(frozen=True)
class BlockAuditV15B:
    schema_version: str
    stage: str
    step_index: int
    block_index: int
    branch: str
    signed_graph_digest: Optional[str]
    mask_digest: str
    raw_source_material_digest: str
    source_latent_sha256: str
    canonical_extraction_config_sha256: str
    raw_source_material_reopened: bool
    memory_builder_receipt_digest: Optional[str]
    slot_provenance_digest: Optional[str]
    slot_uuid_mask_provenance_verified: bool
    target_write_ownership_sha256: Optional[str]
    target_write_ownership_verified: bool
    cross_role_zero_proof_sha256: Optional[str]
    target_role_state_digest: Optional[str]
    target_transport_digest: Optional[str]
    position_projector_sha256: Optional[str]
    scrubbed_target_key_sha256: Optional[str]
    persistent_support_sha256: Optional[str]
    target_role_assigned_token_count_by_phase: tuple[int, ...]
    target_role_unassigned_corridor_count_by_phase: tuple[int, ...]
    routed_roles: tuple[str, ...]
    role_memory_read_count: int
    route_strength: float
    memory_strength: float
    relation_operator: str
    target_key_sha256: Optional[str]
    tensor_batch_size: int
    tensor_temporal_phases: int
    tensor_height: int
    tensor_width: int
    tensor_heads: int
    tensor_head_dim: int
    tensor_hidden_width: int
    tensor_dtype: str
    tensor_device: str
    background_hidden_max_abs: Optional[float]
    background_key_max_abs: Optional[float]
    background_value_max_abs: Optional[float]
    route_delta_outside_corridor_max_abs: float
    memory_residual_outside_corridor_max_abs: float
    phase0_route_max_abs: float
    phase0_memory_max_abs: float
    disallowed_add_edge_max_abs: float
    disallowed_remove_edge_max_abs: float
    memory_hidden_mutation_max_abs: float
    memory_key_mutation_max_abs: float
    memory_convex_violation_max_abs: float
    cross_role_memory_write_max_abs: float
    target_cross_role_rename_count: int
    target_corridor_escape_count: int
    target_dual_position_component_count: int
    transition_background_overlap_count: int
    source_coordinate_target_write_count: int
    phase0_full_source_restore_call_count: int
    phase0_full_source_restore_token_count: int
    phase0_hidden_source_max_abs: float
    phase0_key_source_max_abs: float
    phase0_value_source_max_abs: float
    same_coordinate_object_kv_copy_count: int
    object_hidden_hard_restore_count: int
    phase_indexed_source_kv_access_count: int
    post_rope_source_kv_access_count: int
    anchor_forbidden_access_count: int
    input_hidden_sha256: str
    input_query_sha256: Optional[str]
    input_key_sha256: str
    input_value_sha256: str
    carrier_hidden_sha256: str
    carrier_key_sha256: str
    carrier_value_sha256: str
    output_hidden_sha256: str
    output_query_sha256: Optional[str]
    output_key_sha256: str
    output_value_sha256: str
    route_delta_sha256: Optional[str]
    appearance_residual_sha256: Optional[str]
    cell_tensor_abi_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != BLOCK_AUDIT_SCHEMA or self.stage not in ("pre", "post"):
            raise V15BContractError("block audit schema/stage differs")
        _exact_int(self.step_index, label="audit step"); _exact_int(self.block_index, label="audit block")
        if self.step_index >= DENOISE_STEPS or self.block_index >= TRANSFORMER_BLOCKS:
            raise V15BContractError("audit step/block is outside exact execution geometry")
        if self.branch not in CFG_BRANCHES:
            raise V15BContractError("audit branch differs")
        if self.signed_graph_digest is not None: _sha(self.signed_graph_digest, label="audit signed graph")
        _sha(self.mask_digest, label="audit mask")
        for label, value in (
            ("raw source material", self.raw_source_material_digest),
            ("source latent", self.source_latent_sha256),
            ("canonical extraction config", self.canonical_extraction_config_sha256),
        ):
            _sha(value, label=f"audit {label}")
        if self.raw_source_material_reopened is not True:
            raise V15BContractError("audit did not reopen immutable raw source material")
        if self.memory_builder_receipt_digest is not None:
            _sha(self.memory_builder_receipt_digest, label="audit memory builder")
        for label, value in (
            ("slot provenance", self.slot_provenance_digest),
            ("target write ownership", self.target_write_ownership_sha256),
            ("cross-role zero proof", self.cross_role_zero_proof_sha256),
        ):
            if value is not None:
                _sha(value, label=f"audit {label}")
        if self.target_role_state_digest is not None:
            _sha(self.target_role_state_digest, label="audit target role-state")
        for label, value in (
            ("target transport", self.target_transport_digest),
            ("position projector", self.position_projector_sha256),
            ("scrubbed target key", self.scrubbed_target_key_sha256),
            ("persistent support", self.persistent_support_sha256),
        ):
            if value is not None:
                _sha(value, label=f"audit {label}")
        for name in (
            "target_role_assigned_token_count_by_phase",
            "target_role_unassigned_corridor_count_by_phase",
        ):
            counts = getattr(self, name)
            if len(counts) != LATENT_PHASES:
                raise V15BContractError(f"audit {name} phase geometry differs")
            for count in counts:
                _exact_int(count, label=f"audit {name}")
        has_role_state = self.target_role_state_digest is not None
        if has_role_state != (self.memory_builder_receipt_digest is not None):
            raise V15BContractError("audit target role-state/memory activation differs")
        if has_role_state != all(value is not None for value in (
            self.target_transport_digest, self.position_projector_sha256,
            self.scrubbed_target_key_sha256, self.persistent_support_sha256,
        )):
            raise V15BContractError("audit persistent slot tensor authority activation differs")
        if has_role_state != all(value is not None for value in (
            self.slot_provenance_digest, self.target_write_ownership_sha256,
            self.cross_role_zero_proof_sha256,
        )):
            raise V15BContractError("audit cross-role provenance proof activation differs")
        if has_role_state != (
            self.slot_uuid_mask_provenance_verified is True
            and self.target_write_ownership_verified is True
        ):
            raise V15BContractError("audit cross-role zero lacks provenance/ownership proof")
        if not has_role_state and (
            self.slot_uuid_mask_provenance_verified is not False
            or self.target_write_ownership_verified is not False
        ):
            raise V15BContractError("inactive memory emitted cross-role proof")
        if not has_role_state and (
            any(self.target_role_assigned_token_count_by_phase)
            or any(self.target_role_unassigned_corridor_count_by_phase)
        ):
            raise V15BContractError("inactive target role-state emitted phase counts")
        if tuple(sorted(set(self.routed_roles))) != self.routed_roles:
            raise V15BContractError("audit routed roles must be sorted/unique")
        _exact_int(self.role_memory_read_count, label="memory read count")
        route_strength = _finite(self.route_strength, label="audit route strength")
        memory_strength = _finite(self.memory_strength, label="audit memory strength")
        if not 0 <= route_strength <= 1 or not 0 <= memory_strength <= 1:
            raise V15BContractError("audit route/memory strength is outside [0,1]")
        if self.relation_operator not in (
            "none", "position_scrubbed_target_key_persistent_role_pool_query_scatter"
        ):
            raise V15BContractError("audit relation operator differs")
        if self.target_key_sha256 is not None:
            _sha(self.target_key_sha256, label="audit target key")
        if (self.relation_operator == "none") != (self.target_key_sha256 is None):
            raise V15BContractError("audit target-key authority/operator activation differs")
        for name, value in (
            ("batch", self.tensor_batch_size),
            ("temporal phases", self.tensor_temporal_phases),
            ("height", self.tensor_height), ("width", self.tensor_width),
            ("heads", self.tensor_heads), ("head dim", self.tensor_head_dim),
            ("hidden width", self.tensor_hidden_width),
        ):
            _exact_int(value, label=f"audit tensor {name}", minimum=1)
        if self.tensor_temporal_phases != LATENT_PHASES:
            raise V15BContractError("audit tensor temporal geometry differs")
        if self.tensor_dtype != "torch.float32" or self.tensor_device != "cpu":
            raise V15BContractError("audit tensor dtype/device differs")
        for name in ("background_hidden_max_abs", "background_key_max_abs", "background_value_max_abs"):
            value = getattr(self, name)
            if value is not None and _finite(value, label=name) < 0: raise V15BContractError(f"{name} is negative")
        for name in ("route_delta_outside_corridor_max_abs",
                     "memory_residual_outside_corridor_max_abs",
                     "phase0_route_max_abs", "phase0_memory_max_abs",
                     "disallowed_add_edge_max_abs", "disallowed_remove_edge_max_abs",
                     "memory_hidden_mutation_max_abs", "memory_key_mutation_max_abs",
                     "memory_convex_violation_max_abs", "cross_role_memory_write_max_abs",
                     "phase0_hidden_source_max_abs", "phase0_key_source_max_abs",
                     "phase0_value_source_max_abs"):
            if _finite(getattr(self, name), label=name) < 0: raise V15BContractError(f"{name} is negative")
        for name in (
            "route_delta_outside_corridor_max_abs", "memory_residual_outside_corridor_max_abs",
            "phase0_route_max_abs", "phase0_memory_max_abs",
            "disallowed_add_edge_max_abs", "disallowed_remove_edge_max_abs",
            "memory_hidden_mutation_max_abs", "memory_key_mutation_max_abs",
            "memory_convex_violation_max_abs", "cross_role_memory_write_max_abs",
            "phase0_hidden_source_max_abs",
            "phase0_key_source_max_abs", "phase0_value_source_max_abs",
        ):
            if getattr(self, name) != 0.0:
                raise V15BContractError(f"{name} must be exactly zero")
        _exact_int(
            self.transition_background_overlap_count,
            label="transition/background overlap count",
        )
        if self.transition_background_overlap_count != 0:
            raise V15BContractError("background restore overlaps the transition corridor")
        _exact_int(
            self.phase0_full_source_restore_call_count,
            label="phase0 full-source restore calls",
        )
        _exact_int(
            self.phase0_full_source_restore_token_count,
            label="phase0 full-source restore tokens",
        )
        expected_phase0_tokens = self.tensor_batch_size * self.tensor_height * self.tensor_width
        if (self.phase0_full_source_restore_call_count != 1 or
                self.phase0_full_source_restore_token_count != expected_phase0_tokens):
            raise V15BContractError("phase0 full-source identity exception differs")
        for name in ("source_coordinate_target_write_count",
                     "target_cross_role_rename_count", "target_corridor_escape_count",
                     "target_dual_position_component_count",
                     "same_coordinate_object_kv_copy_count", "object_hidden_hard_restore_count",
                     "phase_indexed_source_kv_access_count", "post_rope_source_kv_access_count",
                     "anchor_forbidden_access_count"):
            value = getattr(self, name)
            _exact_int(value, label=f"audit {name}")
            if value != 0:
                raise V15BContractError(f"{name} must be exactly zero")
        for label, value in (
            ("input hidden", self.input_hidden_sha256),
            ("input key", self.input_key_sha256), ("input value", self.input_value_sha256),
            ("carrier hidden", self.carrier_hidden_sha256),
            ("carrier key", self.carrier_key_sha256),
            ("carrier value", self.carrier_value_sha256),
            ("output hidden", self.output_hidden_sha256),
            ("output key", self.output_key_sha256),
            ("output value", self.output_value_sha256),
            ("cell tensor ABI", self.cell_tensor_abi_digest),
        ):
            _sha(value, label=f"audit {label}")
        for label, value in (
            ("input query", self.input_query_sha256),
            ("output query", self.output_query_sha256),
            ("route delta", self.route_delta_sha256),
            ("appearance residual", self.appearance_residual_sha256),
        ):
            if value is not None:
                _sha(value, label=f"audit {label}")
        has_query_tensors = self.stage == "pre"
        if has_query_tensors != all(value is not None for value in (
            self.input_query_sha256, self.output_query_sha256,
            self.route_delta_sha256, self.appearance_residual_sha256,
        )):
            raise V15BContractError("audit pre/post tensor ABI activation differs")
        if object_sha256(self._tensor_abi_payload()) != self.cell_tensor_abi_digest:
            raise V15BContractError("audit cell tensor ABI digest differs")

    def _tensor_abi_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "stage": self.stage,
            "step_index": self.step_index, "block_index": self.block_index,
            "branch": self.branch, "mask_digest": self.mask_digest,
            "raw_source_material_digest": self.raw_source_material_digest,
            "source_latent_sha256": self.source_latent_sha256,
            "canonical_extraction_config_sha256": (
                self.canonical_extraction_config_sha256
            ),
            "raw_source_material_reopened": self.raw_source_material_reopened,
            "slot_provenance_digest": self.slot_provenance_digest,
            "slot_uuid_mask_provenance_verified": (
                self.slot_uuid_mask_provenance_verified
            ),
            "target_write_ownership_sha256": self.target_write_ownership_sha256,
            "target_write_ownership_verified": self.target_write_ownership_verified,
            "cross_role_zero_proof_sha256": self.cross_role_zero_proof_sha256,
            "target_role_state_digest": self.target_role_state_digest,
            "target_transport_digest": self.target_transport_digest,
            "position_projector_sha256": self.position_projector_sha256,
            "scrubbed_target_key_sha256": self.scrubbed_target_key_sha256,
            "persistent_support_sha256": self.persistent_support_sha256,
            "input_hidden_sha256": self.input_hidden_sha256,
            "input_query_sha256": self.input_query_sha256,
            "input_key_sha256": self.input_key_sha256,
            "input_value_sha256": self.input_value_sha256,
            "carrier_hidden_sha256": self.carrier_hidden_sha256,
            "carrier_key_sha256": self.carrier_key_sha256,
            "carrier_value_sha256": self.carrier_value_sha256,
            "output_hidden_sha256": self.output_hidden_sha256,
            "output_query_sha256": self.output_query_sha256,
            "output_key_sha256": self.output_key_sha256,
            "output_value_sha256": self.output_value_sha256,
            "route_delta_sha256": self.route_delta_sha256,
            "appearance_residual_sha256": self.appearance_residual_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class PreBlockStateV15B:
    hidden: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    route_delta: torch.Tensor
    appearance_residual: torch.Tensor
    target_role_state: Optional[TargetRoleStateV15B]
    audit: BlockAuditV15B

    def __post_init__(self) -> None:
        if not isinstance(self.audit, BlockAuditV15B) or self.audit.stage != "pre":
            raise V15BContractError("pre-block state lacks a pre audit")
        _revalidate_material(self.audit, label="pre-block audit")
        for observed, expected, label in (
            (tensor_sha256(self.hidden), self.audit.output_hidden_sha256, "hidden"),
            (tensor_sha256(self.query), self.audit.output_query_sha256, "query"),
            (tensor_sha256(self.key), self.audit.output_key_sha256, "key"),
            (tensor_sha256(self.value), self.audit.output_value_sha256, "value"),
            (tensor_sha256(self.route_delta), self.audit.route_delta_sha256, "route delta"),
            (tensor_sha256(self.appearance_residual),
             self.audit.appearance_residual_sha256, "appearance residual"),
        ):
            if observed != expected:
                raise V15BContractError(f"pre-block {label} differs from tensor ABI audit")
        if (self.target_role_state is None) != (self.audit.target_role_state_digest is None):
            raise V15BContractError("pre-block target role-state activation differs")
        if (self.target_role_state is not None and
                self.target_role_state.digest != self.audit.target_role_state_digest):
            raise V15BContractError("pre-block target role-state digest differs")
        if self.target_role_state is not None:
            _revalidate_material(self.target_role_state, label="pre-block target role-state")


@dataclass(frozen=True)
class PostBlockStateV15B:
    hidden: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    audit: BlockAuditV15B

    def __post_init__(self) -> None:
        if not isinstance(self.audit, BlockAuditV15B) or self.audit.stage != "post":
            raise V15BContractError("post-block state lacks a post audit")
        _revalidate_material(self.audit, label="post-block audit")
        for observed, expected, label in (
            (tensor_sha256(self.hidden), self.audit.output_hidden_sha256, "hidden"),
            (tensor_sha256(self.key), self.audit.output_key_sha256, "key"),
            (tensor_sha256(self.value), self.audit.output_value_sha256, "value"),
        ):
            if observed != expected:
                raise V15BContractError(f"post-block {label} differs from tensor ABI audit")


def _validate_target(hidden: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor, carrier: SourceBackgroundCarrierV15B) -> None:
    _cpu_fp32(hidden, label="target hidden", ndim=3)
    for label, tensor in (("query", query), ("key", key), ("value", value)):
        _cpu_fp32(tensor, label=f"target {label}", ndim=4)
    if tuple(query.shape) != tuple(key.shape) or tuple(key.shape) != tuple(value.shape):
        raise V15BContractError("target Q/K/V geometry differs")
    if tuple(hidden.shape[:2]) != tuple(query.shape[:2]):
        raise V15BContractError("target hidden/QKV token geometry differs")
    if tuple(hidden.shape) != tuple(carrier.hidden.shape) or tuple(key.shape) != tuple(carrier.key.shape):
        raise V15BContractError("target/background carrier geometry differs")
    masks = carrier.masks
    if int(query.shape[1]) != LATENT_PHASES * masks.height * masks.width:
        raise V15BContractError("target tensor T/H/W differs from mask authority")


def _relation_route_delta(target_key: torch.Tensor, signed_graph: SignedEditGraphV15B,
                          binding: SourceActionRoleBindingV15B,
                          target_roles: TargetRoleStateV15B,
                          masks: SourceRoleMaskSetV15B) -> torch.Tensor:
    """Read current target K by key role, then scatter the result to target Q roles."""
    _revalidate_material(signed_graph, label="relation signed graph")
    _revalidate_material(binding, label="relation binding")
    _revalidate_material(target_roles, label="relation target roles")
    _revalidate_material(masks, label="relation masks")
    batch, tokens, heads, width = target_key.shape
    if tokens % LATENT_PHASES or int(signed_graph.graph.shape[0]) != heads:
        raise V15BContractError("signed graph/target-K phase or head geometry differs")
    spatial = tokens // LATENT_PHASES
    if spatial != masks.height * masks.width:
        raise V15BContractError("relation target-K H/W differs from mask authority")
    if (target_roles.native_target_key_sha256 != tensor_sha256(target_key) or
            target_roles.binding_digest != binding.digest or
            target_roles.mask_digest != masks.digest):
        raise V15BContractError("relation target role-state/native-K authority differs")
    phase_k = target_roles.scrubbed_target_key.reshape(
        batch, LATENT_PHASES, spatial, heads, width
    )
    ownership = target_roles.confident_role_weights.reshape(
        batch, LATENT_PHASES, spatial, len(target_roles.role_ids)
    )
    role_state = torch.zeros(batch, LATENT_PHASES, len(SIGNED_ROLES), heads, width,
                             device=target_key.device, dtype=target_key.dtype)
    for role_index, signed_role in enumerate(SIGNED_ROLES):
        source_role = binding.signed_to_source[signed_role]
        weight = ownership[..., target_roles.role_ids.index(source_role)]
        denom = weight.sum(2).clamp_min(1e-8).reshape(batch, LATENT_PHASES, 1, 1)
        role_state[:, :, role_index] = (
            phase_k * weight[..., None, None]
        ).sum(2) / denom
    role_delta = torch.einsum(
        "htruk,bukhd->btrhd", signed_graph.graph.to(target_key.device), role_state.float()
    ).to(target_key.dtype)
    delta = torch.zeros_like(phase_k); weight = torch.zeros_like(phase_k[..., :1, :1])
    for role_index, signed_role in enumerate(SIGNED_ROLES):
        source_role = binding.signed_to_source[signed_role]
        role_weight = ownership[..., target_roles.role_ids.index(source_role)]
        delta += role_delta[:, :, role_index, None] * role_weight[..., None, None]
        weight += role_weight[..., None, None]
    delta = delta / weight.clamp_min(1)
    delta[:, 0].zero_()
    return delta.reshape_as(target_key)


def _role_content_read_components(
    native_target_key: torch.Tensor, memory: SourceRoleContentMemoryV15B,
    binding: SourceActionRoleBindingV15B, target_roles: TargetRoleStateV15B,
) -> torch.Tensor:
    _revalidate_material(memory, label="source content read memory")
    _revalidate_material(binding, label="source content read binding")
    _revalidate_material(target_roles, label="source content read target roles")
    if memory.binding_digest != binding.digest or set(memory.role_ids) != set(binding.source_roles):
        raise V15BContractError("source content memory/binding roles differ")
    if (memory.digest != target_roles.memory_digest or
            target_roles.native_target_key_sha256 != tensor_sha256(native_target_key)):
        raise V15BContractError("source content memory/target role-state authority differs")
    if tuple(memory.key_content.shape[2:]) != tuple(native_target_key.shape[2:]):
        raise V15BContractError("source content memory/target-K head geometry differs")
    components = torch.zeros(
        *native_target_key.shape[:2], len(target_roles.role_ids),
        *native_target_key.shape[2:], dtype=native_target_key.dtype,
        device=native_target_key.device,
    )
    scrubbed_target_key = target_roles.scrubbed_target_key
    for role in target_roles.role_ids:
        index = memory.role_ids.index(role)
        valid = memory.slot_valid_mask[index]
        key = memory.key_content[index, valid]
        value = memory.value_content[index, valid]
        score = torch.einsum("blhd,mhd->blhm", scrubbed_target_key, key) / math.sqrt(
            scrubbed_target_key.shape[-1]
        )
        source_read = torch.einsum(
            "blhm,mhd->blhd", score.float().softmax(-1).to(native_target_key.dtype), value
        )
        role_weight = target_roles.confident_role_weights[..., index]
        components[:, :, index] = source_read * role_weight[..., None, None]
    return components


def _role_content_read(native_target_key: torch.Tensor,
                       memory: SourceRoleContentMemoryV15B,
                       binding: SourceActionRoleBindingV15B,
                       target_roles: TargetRoleStateV15B) -> torch.Tensor:
    return _role_content_read_components(
        native_target_key, memory, binding, target_roles
    ).sum(2)


def apply_pre_block_v15b(
    *, target_hidden: torch.Tensor, target_query: torch.Tensor,
    target_key: torch.Tensor, target_value: torch.Tensor,
    carrier: SourceBackgroundCarrierV15B, binding: SourceActionRoleBindingV15B,
    signed_graph: Optional[SignedEditGraphV15B],
    content_memory: Optional[SourceRoleContentMemoryV15B],
    target_native_transport: Optional[TargetNativeTransportV15B],
    route_strength: float, memory_strength: float, restore_background: bool,
) -> PreBlockStateV15B:
    _revalidate_material(carrier, label="pre-block carrier")
    _revalidate_material(binding, label="pre-block binding")
    if signed_graph is not None:
        _revalidate_material(signed_graph, label="pre-block signed graph")
    if content_memory is not None:
        _revalidate_material(content_memory, label="pre-block content memory")
    if target_native_transport is not None:
        _revalidate_material(target_native_transport, label="pre-block transport")
    _validate_target(target_hidden, target_query, target_key, target_value, carrier)
    if carrier.binding_digest != binding.digest:
        raise V15BContractError("target binding/background authority differs")
    route_strength = _finite(route_strength, label="route strength")
    memory_strength = _finite(memory_strength, label="memory strength")
    if not 0 <= route_strength <= 1 or not 0 <= memory_strength <= 1:
        raise V15BContractError("route/memory strength must lie in [0,1]")
    if (signed_graph is None) != (route_strength == 0):
        raise V15BContractError("signed graph/route strength activation differs")
    if (content_memory is None) != (memory_strength == 0):
        raise V15BContractError("content memory/strength activation differs")
    if (content_memory is None) != (target_native_transport is None):
        raise V15BContractError("content memory requires one explicit target-native transport")
    if signed_graph is not None and content_memory is None:
        raise V15BContractError(
            "relation routing requires per-cell target roles from source content memory"
        )
    if signed_graph is not None and signed_graph.action_id != binding.action_id:
        raise V15BContractError("signed graph/binding action differs")
    if content_memory is not None:
        if (
            content_memory.source_video_sha256 != carrier.source_video_sha256
            or content_memory.binding_digest != binding.digest
            or content_memory.mask_digest != carrier.masks.digest
            or content_memory.step_index != carrier.step_index
            or content_memory.block_index != carrier.block_index
            or content_memory.branch != carrier.branch
            or content_memory.raw_source_material.digest !=
            carrier.raw_source_material.digest
        ):
            raise V15BContractError("content memory does not bind this source/mask/execution cell")
    target_role_state = None
    if content_memory is not None:
        target_role_state = build_target_role_state_v15b(
            native_target_pre_rope_key=target_key,
            memory=content_memory, masks=carrier.masks, binding=binding,
            target_native_transport=target_native_transport,
        )
    hidden = target_hidden.clone(); query = target_query.clone()
    key = target_key.clone(); value = target_value.clone()
    background = carrier.masks.background_support_mask
    spatial = carrier.masks.height * carrier.masks.width
    phase0 = torch.zeros_like(background)
    phase0[:, :spatial] = True
    raw_phase0_hidden, raw_phase0_key, raw_phase0_value = (
        carrier.reopen_phase0_raw_hkv()
    )
    # Explicit identity boundary condition, common to all A/B/K0 arms.  This
    # is the sole full-source object-coordinate copy exception in the CPU seam;
    # the bytes are freshly reopened rather than trusted from carrier tensors.
    hidden[:, :spatial] = raw_phase0_hidden
    key[:, :spatial] = raw_phase0_key
    value[:, :spatial] = raw_phase0_value
    post_phase0_background = background & ~phase0
    if restore_background:
        hidden[post_phase0_background] = carrier.hidden[post_phase0_background]
        key[post_phase0_background] = carrier.key[post_phase0_background]
        value[post_phase0_background] = carrier.value[post_phase0_background]
    route_delta = torch.zeros_like(query)
    if signed_graph is not None:
        if target_role_state is None:  # defensive, activation gate above is fail-closed
            raise V15BContractError("relation route lacks a target role-state")
        route_delta = _relation_route_delta(
            target_key, signed_graph, binding, target_role_state, carrier.masks
        )
        query += route_strength * route_delta
    appearance = torch.zeros_like(value)
    hidden_before_memory = hidden.clone(); key_before_memory = key.clone()
    value_before_memory = value.clone()
    expected_value_after_memory = value_before_memory.clone()
    cross_role_memory_write_max_abs = 0.0
    slot_provenance_digest = None
    slot_uuid_mask_provenance_verified = False
    target_write_ownership_sha256 = None
    target_write_ownership_verified = False
    cross_role_zero_proof_sha256 = None
    if content_memory is not None:
        if target_role_state is None:  # defensive, built from native target K above
            raise V15BContractError("source-property read lacks a target role-state")
        role_components = _role_content_read_components(
            target_key, content_memory, binding, target_role_state
        )
        source_read = role_components.sum(2)
        same_owner = target_role_state.confident_role_weights[
            ..., None, None
        ]
        cross_role_memory_write_max_abs = float(
            (role_components * (1.0 - same_owner)).abs().max()
        )
        slot_provenance_digest = object_sha256(
            content_memory.slot_provenance_by_role
        )
        slot_uuid_mask_provenance_verified = (
            content_memory.slot_uuid_mask_provenance_verified is True
            and slot_provenance_digest ==
            content_memory.builder_receipt.slot_provenance_digest
        )
        target_write_ownership_sha256 = tensor_sha256(
            target_role_state.confident_role_weights
        )
        target_write_ownership_verified = torch.equal(
            same_owner[..., 0, 0], target_role_state.confident_role_weights
        )
        cross_role_zero_proof_sha256 = object_sha256({
            "raw_source_material_digest": (
                content_memory.raw_source_material.digest
            ),
            "slot_provenance_digest": slot_provenance_digest,
            "target_write_ownership_sha256": target_write_ownership_sha256,
            "target_role_state_digest": target_role_state.digest,
            "role_component_sha256_by_role": tuple(
                (role, tensor_sha256(role_components[:, :, role_index]))
                for role_index, role in enumerate(content_memory.role_ids)
            ),
            "cross_role_memory_write_max_abs": cross_role_memory_write_max_abs,
        })
        target_support = target_role_state.assigned_support_mask.clone()
        target_support[:, :spatial] = False
        # Convex source-property blend: lambda=1 returns the source read;
        # lambda=0 is identity.  It cannot add source V on top of target V.
        appearance = (source_read - value) * target_support[..., None, None]
        value += memory_strength * appearance
        expected_value_after_memory = value_before_memory + memory_strength * appearance
    corridor = carrier.masks.editable_corridor_mask
    zeros_q = torch.zeros_like(query); zeros_v = torch.zeros_like(value)
    zero_phase_counts = (0,) * LATENT_PHASES
    tensor_abi_payload = {
        "schema_version": BLOCK_AUDIT_SCHEMA, "stage": "pre",
        "step_index": carrier.step_index, "block_index": carrier.block_index,
        "branch": carrier.branch, "mask_digest": carrier.masks.digest,
        "raw_source_material_digest": carrier.raw_source_material.digest,
        "source_latent_sha256": carrier.raw_source_material.source_latent_sha256,
        "canonical_extraction_config_sha256": (
            carrier.raw_source_material.canonical_extraction_config_sha256
        ),
        "raw_source_material_reopened": True,
        "slot_provenance_digest": slot_provenance_digest,
        "slot_uuid_mask_provenance_verified": (
            slot_uuid_mask_provenance_verified
        ),
        "target_write_ownership_sha256": target_write_ownership_sha256,
        "target_write_ownership_verified": target_write_ownership_verified,
        "cross_role_zero_proof_sha256": cross_role_zero_proof_sha256,
        "target_role_state_digest": (
            target_role_state.digest if target_role_state else None
        ),
        "target_transport_digest": (
            target_role_state.transport.digest if target_role_state else None
        ),
        "position_projector_sha256": (
            target_role_state.position_projector_sha256 if target_role_state else None
        ),
        "scrubbed_target_key_sha256": (
            tensor_sha256(target_role_state.scrubbed_target_key)
            if target_role_state else None
        ),
        "persistent_support_sha256": (
            tensor_sha256(target_role_state.assigned_support_mask)
            if target_role_state else None
        ),
        "input_hidden_sha256": tensor_sha256(target_hidden),
        "input_query_sha256": tensor_sha256(target_query),
        "input_key_sha256": tensor_sha256(target_key),
        "input_value_sha256": tensor_sha256(target_value),
        "carrier_hidden_sha256": tensor_sha256(carrier.hidden),
        "carrier_key_sha256": tensor_sha256(carrier.key),
        "carrier_value_sha256": tensor_sha256(carrier.value),
        "output_hidden_sha256": tensor_sha256(hidden),
        "output_query_sha256": tensor_sha256(query),
        "output_key_sha256": tensor_sha256(key),
        "output_value_sha256": tensor_sha256(value),
        "route_delta_sha256": tensor_sha256(route_delta),
        "appearance_residual_sha256": tensor_sha256(appearance),
    }
    audit = BlockAuditV15B(
        schema_version=BLOCK_AUDIT_SCHEMA, stage="pre",
        step_index=carrier.step_index, block_index=carrier.block_index,
        branch=carrier.branch,
        signed_graph_digest=signed_graph.digest if signed_graph else None,
        mask_digest=carrier.masks.digest,
        raw_source_material_digest=carrier.raw_source_material.digest,
        source_latent_sha256=carrier.raw_source_material.source_latent_sha256,
        canonical_extraction_config_sha256=(
            carrier.raw_source_material.canonical_extraction_config_sha256
        ),
        raw_source_material_reopened=True,
        memory_builder_receipt_digest=(
            content_memory.builder_receipt.digest if content_memory else None
        ),
        slot_provenance_digest=slot_provenance_digest,
        slot_uuid_mask_provenance_verified=slot_uuid_mask_provenance_verified,
        target_write_ownership_sha256=target_write_ownership_sha256,
        target_write_ownership_verified=target_write_ownership_verified,
        cross_role_zero_proof_sha256=cross_role_zero_proof_sha256,
        target_role_state_digest=(target_role_state.digest if target_role_state else None),
        target_transport_digest=(
            target_role_state.transport.digest if target_role_state else None
        ),
        position_projector_sha256=(
            target_role_state.position_projector_sha256 if target_role_state else None
        ),
        scrubbed_target_key_sha256=(
            tensor_sha256(target_role_state.scrubbed_target_key)
            if target_role_state else None
        ),
        persistent_support_sha256=(
            tensor_sha256(target_role_state.assigned_support_mask)
            if target_role_state else None
        ),
        target_role_assigned_token_count_by_phase=(
            target_role_state.assigned_token_count_by_phase
            if target_role_state else zero_phase_counts
        ),
        target_role_unassigned_corridor_count_by_phase=(
            target_role_state.unassigned_corridor_count_by_phase
            if target_role_state else zero_phase_counts
        ),
        routed_roles=tuple(sorted(SIGNED_ROLES)) if signed_graph else (),
        role_memory_read_count=len(binding.source_roles) if content_memory else 0,
        route_strength=route_strength, memory_strength=memory_strength,
        relation_operator=(
            "position_scrubbed_target_key_persistent_role_pool_query_scatter"
            if signed_graph else "none"
        ),
        target_key_sha256=tensor_sha256(target_key) if signed_graph else None,
        tensor_batch_size=int(target_hidden.shape[0]),
        tensor_temporal_phases=LATENT_PHASES,
        tensor_height=carrier.masks.height, tensor_width=carrier.masks.width,
        tensor_heads=int(target_key.shape[2]), tensor_head_dim=int(target_key.shape[3]),
        tensor_hidden_width=int(target_hidden.shape[2]),
        tensor_dtype=str(target_hidden.dtype), tensor_device=str(target_hidden.device),
        background_hidden_max_abs=(
            _masked_max_abs(hidden, carrier.hidden, background) if restore_background else None
        ),
        background_key_max_abs=(
            _masked_max_abs(key, carrier.key, background) if restore_background else None
        ),
        background_value_max_abs=(
            _masked_max_abs(value, carrier.value, background) if restore_background else None
        ),
        route_delta_outside_corridor_max_abs=_masked_max_abs(route_delta, zeros_q, ~corridor),
        memory_residual_outside_corridor_max_abs=_masked_max_abs(appearance, zeros_v, ~corridor),
        phase0_route_max_abs=_masked_max_abs(route_delta, zeros_q, phase0),
        phase0_memory_max_abs=_masked_max_abs(appearance, zeros_v, phase0),
        disallowed_add_edge_max_abs=(
            signed_graph.disallowed_add_edge_max_abs if signed_graph else 0.0
        ),
        disallowed_remove_edge_max_abs=(
            signed_graph.disallowed_remove_edge_max_abs if signed_graph else 0.0
        ),
        memory_hidden_mutation_max_abs=_max_abs(hidden, hidden_before_memory),
        memory_key_mutation_max_abs=_max_abs(key, key_before_memory),
        memory_convex_violation_max_abs=_max_abs(value, expected_value_after_memory),
        cross_role_memory_write_max_abs=cross_role_memory_write_max_abs,
        target_cross_role_rename_count=(
            target_role_state.cross_role_rename_count if target_role_state else 0
        ),
        target_corridor_escape_count=(
            target_role_state.corridor_escape_count if target_role_state else 0
        ),
        target_dual_position_component_count=(
            target_role_state.dual_position_component_count if target_role_state else 0
        ),
        transition_background_overlap_count=int((
            carrier.masks.transition_path_mask & background
        ).sum()),
        source_coordinate_target_write_count=0,
        phase0_full_source_restore_call_count=1,
        phase0_full_source_restore_token_count=int(phase0.sum()),
        phase0_hidden_source_max_abs=_max_abs(
            hidden[:, :spatial], raw_phase0_hidden
        ),
        phase0_key_source_max_abs=_max_abs(key[:, :spatial], raw_phase0_key),
        phase0_value_source_max_abs=_max_abs(
            value[:, :spatial], raw_phase0_value
        ),
        same_coordinate_object_kv_copy_count=0,
        object_hidden_hard_restore_count=0,
        phase_indexed_source_kv_access_count=0,
        post_rope_source_kv_access_count=0,
        anchor_forbidden_access_count=0,
        input_hidden_sha256=tensor_abi_payload["input_hidden_sha256"],
        input_query_sha256=tensor_abi_payload["input_query_sha256"],
        input_key_sha256=tensor_abi_payload["input_key_sha256"],
        input_value_sha256=tensor_abi_payload["input_value_sha256"],
        carrier_hidden_sha256=tensor_abi_payload["carrier_hidden_sha256"],
        carrier_key_sha256=tensor_abi_payload["carrier_key_sha256"],
        carrier_value_sha256=tensor_abi_payload["carrier_value_sha256"],
        output_hidden_sha256=tensor_abi_payload["output_hidden_sha256"],
        output_query_sha256=tensor_abi_payload["output_query_sha256"],
        output_key_sha256=tensor_abi_payload["output_key_sha256"],
        output_value_sha256=tensor_abi_payload["output_value_sha256"],
        route_delta_sha256=tensor_abi_payload["route_delta_sha256"],
        appearance_residual_sha256=tensor_abi_payload[
            "appearance_residual_sha256"
        ],
        cell_tensor_abi_digest=object_sha256(tensor_abi_payload),
    )
    return PreBlockStateV15B(
        hidden, query, key, value, route_delta, appearance, target_role_state, audit
    )


def apply_post_block_v15b(
    *, target_hidden: torch.Tensor, target_key: torch.Tensor, target_value: torch.Tensor,
    carrier: SourceBackgroundCarrierV15B, binding: SourceActionRoleBindingV15B,
    signed_graph_digest: Optional[str], restore_background: bool,
) -> PostBlockStateV15B:
    _revalidate_material(carrier, label="post-block carrier")
    _revalidate_material(binding, label="post-block binding")
    _cpu_fp32(target_hidden, label="post hidden", ndim=3)
    _cpu_fp32(target_key, label="post key", ndim=4)
    _cpu_fp32(target_value, label="post value", ndim=4)
    if tuple(target_hidden.shape) != tuple(carrier.hidden.shape) or tuple(target_key.shape) != tuple(carrier.key.shape) or tuple(target_value.shape) != tuple(carrier.value.shape):
        raise V15BContractError("post/background carrier geometry differs")
    if binding.digest != carrier.binding_digest:
        raise V15BContractError("post binding/background authority differs")
    if signed_graph_digest is not None: _sha(signed_graph_digest, label="post signed graph")
    hidden = target_hidden.clone(); key = target_key.clone(); value = target_value.clone()
    background = carrier.masks.background_support_mask
    spatial = carrier.masks.height * carrier.masks.width
    phase0 = torch.zeros_like(background)
    phase0[:, :spatial] = True
    raw_phase0_hidden, raw_phase0_key, raw_phase0_value = (
        carrier.reopen_phase0_raw_hkv()
    )
    hidden[:, :spatial] = raw_phase0_hidden
    key[:, :spatial] = raw_phase0_key
    value[:, :spatial] = raw_phase0_value
    post_phase0_background = background & ~phase0
    if restore_background:
        hidden[post_phase0_background] = carrier.hidden[post_phase0_background]
        key[post_phase0_background] = carrier.key[post_phase0_background]
        value[post_phase0_background] = carrier.value[post_phase0_background]
    zero_phase_counts = (0,) * LATENT_PHASES
    tensor_abi_payload = {
        "schema_version": BLOCK_AUDIT_SCHEMA, "stage": "post",
        "step_index": carrier.step_index, "block_index": carrier.block_index,
        "branch": carrier.branch, "mask_digest": carrier.masks.digest,
        "raw_source_material_digest": carrier.raw_source_material.digest,
        "source_latent_sha256": carrier.raw_source_material.source_latent_sha256,
        "canonical_extraction_config_sha256": (
            carrier.raw_source_material.canonical_extraction_config_sha256
        ),
        "raw_source_material_reopened": True,
        "slot_provenance_digest": None,
        "slot_uuid_mask_provenance_verified": False,
        "target_write_ownership_sha256": None,
        "target_write_ownership_verified": False,
        "cross_role_zero_proof_sha256": None,
        "target_role_state_digest": None, "target_transport_digest": None,
        "position_projector_sha256": None, "scrubbed_target_key_sha256": None,
        "persistent_support_sha256": None,
        "input_hidden_sha256": tensor_sha256(target_hidden),
        "input_query_sha256": None,
        "input_key_sha256": tensor_sha256(target_key),
        "input_value_sha256": tensor_sha256(target_value),
        "carrier_hidden_sha256": tensor_sha256(carrier.hidden),
        "carrier_key_sha256": tensor_sha256(carrier.key),
        "carrier_value_sha256": tensor_sha256(carrier.value),
        "output_hidden_sha256": tensor_sha256(hidden),
        "output_query_sha256": None,
        "output_key_sha256": tensor_sha256(key),
        "output_value_sha256": tensor_sha256(value),
        "route_delta_sha256": None, "appearance_residual_sha256": None,
    }
    audit = BlockAuditV15B(
        schema_version=BLOCK_AUDIT_SCHEMA, stage="post",
        step_index=carrier.step_index, block_index=carrier.block_index,
        branch=carrier.branch, signed_graph_digest=signed_graph_digest,
        mask_digest=carrier.masks.digest,
        raw_source_material_digest=carrier.raw_source_material.digest,
        source_latent_sha256=carrier.raw_source_material.source_latent_sha256,
        canonical_extraction_config_sha256=(
            carrier.raw_source_material.canonical_extraction_config_sha256
        ),
        raw_source_material_reopened=True,
        memory_builder_receipt_digest=None, slot_provenance_digest=None,
        slot_uuid_mask_provenance_verified=False,
        target_write_ownership_sha256=None,
        target_write_ownership_verified=False,
        cross_role_zero_proof_sha256=None,
        target_role_state_digest=None,
        target_transport_digest=None, position_projector_sha256=None,
        scrubbed_target_key_sha256=None, persistent_support_sha256=None,
        target_role_assigned_token_count_by_phase=zero_phase_counts,
        target_role_unassigned_corridor_count_by_phase=zero_phase_counts,
        routed_roles=(), role_memory_read_count=0,
        route_strength=0.0, memory_strength=0.0, relation_operator="none",
        target_key_sha256=None,
        tensor_batch_size=int(target_hidden.shape[0]),
        tensor_temporal_phases=LATENT_PHASES,
        tensor_height=carrier.masks.height, tensor_width=carrier.masks.width,
        tensor_heads=int(target_key.shape[2]), tensor_head_dim=int(target_key.shape[3]),
        tensor_hidden_width=int(target_hidden.shape[2]),
        tensor_dtype=str(target_hidden.dtype), tensor_device=str(target_hidden.device),
        background_hidden_max_abs=(
            _masked_max_abs(hidden, carrier.hidden, background) if restore_background else None
        ),
        background_key_max_abs=(
            _masked_max_abs(key, carrier.key, background) if restore_background else None
        ),
        background_value_max_abs=(
            _masked_max_abs(value, carrier.value, background) if restore_background else None
        ),
        route_delta_outside_corridor_max_abs=0.0,
        memory_residual_outside_corridor_max_abs=0.0,
        phase0_route_max_abs=0.0, phase0_memory_max_abs=0.0,
        disallowed_add_edge_max_abs=0.0, disallowed_remove_edge_max_abs=0.0,
        memory_hidden_mutation_max_abs=0.0, memory_key_mutation_max_abs=0.0,
        memory_convex_violation_max_abs=0.0,
        cross_role_memory_write_max_abs=0.0,
        target_cross_role_rename_count=0, target_corridor_escape_count=0,
        target_dual_position_component_count=0,
        transition_background_overlap_count=int((
            carrier.masks.transition_path_mask & background
        ).sum()),
        source_coordinate_target_write_count=0,
        phase0_full_source_restore_call_count=1,
        phase0_full_source_restore_token_count=int(phase0.sum()),
        phase0_hidden_source_max_abs=_max_abs(
            hidden[:, :spatial], raw_phase0_hidden
        ),
        phase0_key_source_max_abs=_max_abs(key[:, :spatial], raw_phase0_key),
        phase0_value_source_max_abs=_max_abs(
            value[:, :spatial], raw_phase0_value
        ),
        same_coordinate_object_kv_copy_count=0,
        object_hidden_hard_restore_count=0,
        phase_indexed_source_kv_access_count=0,
        post_rope_source_kv_access_count=0,
        anchor_forbidden_access_count=0,
        input_hidden_sha256=tensor_abi_payload["input_hidden_sha256"],
        input_query_sha256=None,
        input_key_sha256=tensor_abi_payload["input_key_sha256"],
        input_value_sha256=tensor_abi_payload["input_value_sha256"],
        carrier_hidden_sha256=tensor_abi_payload["carrier_hidden_sha256"],
        carrier_key_sha256=tensor_abi_payload["carrier_key_sha256"],
        carrier_value_sha256=tensor_abi_payload["carrier_value_sha256"],
        output_hidden_sha256=tensor_abi_payload["output_hidden_sha256"],
        output_query_sha256=None,
        output_key_sha256=tensor_abi_payload["output_key_sha256"],
        output_value_sha256=tensor_abi_payload["output_value_sha256"],
        route_delta_sha256=None, appearance_residual_sha256=None,
        cell_tensor_abi_digest=object_sha256(tensor_abi_payload),
    )
    return PostBlockStateV15B(hidden, key, value, audit)


@dataclass(frozen=True)
class ArmContractV15B:
    arm_id: str
    route_enabled: bool
    graph_slot: Optional[str]
    source_content_memory: bool
    restore_background_pre: bool
    restore_background_post: bool
    route_strength: float
    memory_strength: float
    initial_noise_mode: str

    def __post_init__(self) -> None:
        if self.arm_id not in ARM_IDS or self.initial_noise_mode != "keyed_only":
            raise V15BContractError("arm ID/noise mode differs")
        if self.graph_slot not in (None, "A", "B") or self.route_enabled != (self.graph_slot is not None):
            raise V15BContractError("arm route/graph slot differs")
        route_strength = _finite(self.route_strength, label="arm route strength")
        memory_strength = _finite(self.memory_strength, label="arm memory strength")
        if not 0 <= route_strength <= 1 or not 0 <= memory_strength <= 1:
            raise V15BContractError("arm route/memory strength is outside [0,1]")
        if self.route_enabled != (route_strength > 0):
            raise V15BContractError("arm route-strength activation differs")
        if self.source_content_memory != (memory_strength > 0):
            raise V15BContractError("arm memory-strength activation differs")


@dataclass(frozen=True)
class FourArmContractV15B:
    schema_version: str
    action_id: str
    source_video_sha256: str
    instruction_sha256: str
    binding_digest: str
    mask_digest: str
    track_authority_digest: str
    source_graph_digest: str
    graph_a_slot: str
    graph_b_slot: str
    signed_graph_a_digest: str
    signed_graph_b_digest: str
    aligned_swap_report_digest: str
    four_anchor_consensus_digest: str
    canonical_trace_digest: str
    trace_extractor_code_sha256: str
    trace_extractor_config_sha256: str
    anchor_asset_sha256_by_slot: tuple[tuple[str, str], ...]
    temporal_phases: int
    height: int
    width: int
    batch_size: int
    heads: int
    head_dim: int
    hidden_width: int
    tensor_dtype: str
    tensor_device: str
    denoise_steps: int
    transformer_blocks: int
    cfg_branches: tuple[str, ...]
    self_reported_model_checkpoint_sha256: str
    self_reported_model_code_sha256: str
    core_code_sha256: str
    validator_code_sha256: str
    runner_integration_present: bool
    route_authorized: bool
    arms: tuple[ArmContractV15B, ...]
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != FOUR_ARM_SCHEMA:
            raise V15BContractError("four-arm schema differs")
        _role(self.action_id, label="four-arm action")
        for label, value in (("source", self.source_video_sha256),
                             ("instruction", self.instruction_sha256),
                             ("binding", self.binding_digest),
                             ("mask", self.mask_digest),
                             ("track authority", self.track_authority_digest),
                             ("source graph", self.source_graph_digest),
                             ("signed A", self.signed_graph_a_digest),
                             ("signed B", self.signed_graph_b_digest),
                             ("swap report", self.aligned_swap_report_digest),
                             ("four-anchor consensus", self.four_anchor_consensus_digest),
                             ("canonical trace", self.canonical_trace_digest),
                             ("trace extractor code", self.trace_extractor_code_sha256),
                             ("trace extractor config", self.trace_extractor_config_sha256),
                             ("self-reported model checkpoint",
                              self.self_reported_model_checkpoint_sha256),
                             ("self-reported model code", self.self_reported_model_code_sha256),
                             ("core code", self.core_code_sha256),
                             ("validator code", self.validator_code_sha256)):
            _sha(value, label=label)
        if tuple(slot for slot, _ in self.anchor_asset_sha256_by_slot) != DIAGNOSTIC_ANCHOR_SLOTS:
            raise V15BContractError("contract v0-v3 asset authority slots differ")
        for _, digest in self.anchor_asset_sha256_by_slot:
            _sha(digest, label="contract anchor asset")
        if len({digest for _, digest in self.anchor_asset_sha256_by_slot}) != 4:
            raise V15BContractError("contract must bind four distinct anchor assets")
        for label, value in (
            ("height", self.height), ("width", self.width),
            ("batch", self.batch_size), ("heads", self.heads),
            ("head dim", self.head_dim), ("hidden width", self.hidden_width),
        ):
            _exact_int(value, label=f"contract {label}", minimum=1)
        if (self.temporal_phases != LATENT_PHASES or self.denoise_steps != DENOISE_STEPS or
                self.transformer_blocks != TRANSFORMER_BLOCKS or
                self.cfg_branches != CFG_BRANCHES):
            raise V15BContractError("contract execution geometry differs")
        if self.tensor_dtype != "torch.float32" or self.tensor_device != "cpu":
            raise V15BContractError("v15b independent reference contract must be CPU FP32")
        if self.runner_integration_present is not False or self.route_authorized is not False:
            raise V15BContractError(
                "CPU reference cannot authorize an unimplemented renderer/controller route"
            )
        if (self.graph_a_slot, self.graph_b_slot) != (DEFAULT_GRAPH_A_SLOT, DEFAULT_GRAPH_B_SLOT):
            raise V15BContractError("causal A/B must be appearance-counterfactual v0/v1")
        if tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise V15BContractError("four-arm order differs")
        for arm in self.arms:
            _revalidate_material(arm, label="four-arm arm contract")
        expected = (
            (False, None, False, False, False),
            (True, "A", True, False, False),
            (True, "A", True, True, True),
            (True, "B", True, True, True),
        )
        actual = tuple((a.route_enabled, a.graph_slot, a.source_content_memory,
                        a.restore_background_pre, a.restore_background_post) for a in self.arms)
        if actual != expected:
            raise V15BContractError("four-arm causal matrix differs")
        routed_strengths = {
            (arm.route_strength, arm.memory_strength)
            for arm in self.arms if arm.route_enabled
        }
        if len(routed_strengths) != 1:
            raise V15BContractError("routed causal arms must use one matched strength pair")
        if object_sha256(self._payload()) != _sha(self.digest, label="four-arm digest"):
            raise V15BContractError("four-arm digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "action_id": self.action_id,
            "source_video_sha256": self.source_video_sha256,
            "instruction_sha256": self.instruction_sha256,
            "binding_digest": self.binding_digest, "mask_digest": self.mask_digest,
            "track_authority_digest": self.track_authority_digest,
            "source_graph_digest": self.source_graph_digest,
            "graph_a_slot": self.graph_a_slot, "graph_b_slot": self.graph_b_slot,
            "signed_graph_a_digest": self.signed_graph_a_digest,
            "signed_graph_b_digest": self.signed_graph_b_digest,
            "aligned_swap_report_digest": self.aligned_swap_report_digest,
            "four_anchor_consensus_digest": self.four_anchor_consensus_digest,
            "canonical_trace_digest": self.canonical_trace_digest,
            "trace_extractor_code_sha256": self.trace_extractor_code_sha256,
            "trace_extractor_config_sha256": self.trace_extractor_config_sha256,
            "anchor_asset_sha256_by_slot": self.anchor_asset_sha256_by_slot,
            "temporal_phases": self.temporal_phases, "height": self.height,
            "width": self.width, "batch_size": self.batch_size,
            "heads": self.heads, "head_dim": self.head_dim,
            "hidden_width": self.hidden_width, "tensor_dtype": self.tensor_dtype,
            "tensor_device": self.tensor_device, "denoise_steps": self.denoise_steps,
            "transformer_blocks": self.transformer_blocks,
            "cfg_branches": self.cfg_branches,
            "self_reported_model_checkpoint_sha256": (
                self.self_reported_model_checkpoint_sha256
            ),
            "self_reported_model_code_sha256": self.self_reported_model_code_sha256,
            "core_code_sha256": self.core_code_sha256,
            "validator_code_sha256": self.validator_code_sha256,
            "runner_integration_present": self.runner_integration_present,
            "route_authorized": self.route_authorized,
            "arms": [{f.name: getattr(a, f.name) for f in fields(a)} for a in self.arms],
        }

    @classmethod
    def create(cls, *, action_id: str, source_video_sha256: str,
               instruction_sha256: str, binding_digest: str,
               mask_digest: str, track_authority_digest: str,
               source_graph_digest: str, graph_a_slot: str,
               graph_b_slot: str, signed_graph_a_digest: str,
               signed_graph_b_digest: str,
               aligned_swap_report_digest: str,
               four_anchor_consensus_digest: str,
               canonical_trace_digest: str,
               trace_extractor_code_sha256: str,
               trace_extractor_config_sha256: str,
               anchor_asset_sha256_by_slot: Sequence[tuple[str, str]],
               height: int, width: int, batch_size: int, heads: int,
               head_dim: int, hidden_width: int,
               self_reported_model_checkpoint_sha256: str,
               self_reported_model_code_sha256: str,
               core_code_sha256: str, validator_code_sha256: str,
               route_strength: float, memory_strength: float,
               ) -> "FourArmContractV15B":
        arms = (
            ArmContractV15B(ARM_K0, False, None, False, False, False, 0.0, 0.0, "keyed_only"),
            ArmContractV15B(ARM_GRAPH_A_MEMORY, True, "A", True, False, False,
                            route_strength, memory_strength, "keyed_only"),
            ArmContractV15B(ARM_GRAPH_A_FULL, True, "A", True, True, True,
                            route_strength, memory_strength, "keyed_only"),
            ArmContractV15B(ARM_GRAPH_B_FULL, True, "B", True, True, True,
                            route_strength, memory_strength, "keyed_only"),
        )
        payload = {
            "schema_version": FOUR_ARM_SCHEMA, "action_id": action_id,
            "source_video_sha256": source_video_sha256,
            "instruction_sha256": instruction_sha256,
            "binding_digest": binding_digest, "mask_digest": mask_digest,
            "track_authority_digest": track_authority_digest,
            "source_graph_digest": source_graph_digest,
            "graph_a_slot": graph_a_slot, "graph_b_slot": graph_b_slot,
            "signed_graph_a_digest": signed_graph_a_digest,
            "signed_graph_b_digest": signed_graph_b_digest,
            "aligned_swap_report_digest": aligned_swap_report_digest,
            "four_anchor_consensus_digest": four_anchor_consensus_digest,
            "canonical_trace_digest": canonical_trace_digest,
            "trace_extractor_code_sha256": trace_extractor_code_sha256,
            "trace_extractor_config_sha256": trace_extractor_config_sha256,
            "anchor_asset_sha256_by_slot": tuple(anchor_asset_sha256_by_slot),
            "temporal_phases": LATENT_PHASES, "height": height, "width": width,
            "batch_size": batch_size, "heads": heads, "head_dim": head_dim,
            "hidden_width": hidden_width, "tensor_dtype": "torch.float32",
            "tensor_device": "cpu", "denoise_steps": DENOISE_STEPS,
            "transformer_blocks": TRANSFORMER_BLOCKS, "cfg_branches": CFG_BRANCHES,
            "self_reported_model_checkpoint_sha256": (
                self_reported_model_checkpoint_sha256
            ),
            "self_reported_model_code_sha256": self_reported_model_code_sha256,
            "core_code_sha256": core_code_sha256,
            "validator_code_sha256": validator_code_sha256,
            "runner_integration_present": False, "route_authorized": False,
            "arms": [{f.name: getattr(a, f.name) for f in fields(a)} for a in arms],
        }
        constructor = dict(payload)
        constructor["arms"] = arms
        return cls(**constructor, digest=object_sha256(payload))


__all__ = [name for name in globals() if name.endswith("V15B") or name in {
    "METHOD", "LATENT_PHASES", "DENOISE_STEPS", "TRANSFORMER_BLOCKS",
    "CFG_BRANCHES", "EXPECTED_EXECUTION_CELLS", "GENERIC_ROLES", "SIGNED_ROLES",
    "DEFAULT_GRAPH_A_SLOT", "DEFAULT_GRAPH_B_SLOT", "DIAGNOSTIC_ANCHOR_SLOTS",
    "ACTION_ALLOWED_ADD_EDGES", "ACTION_REQUIRED_ADD_EDGES",
    "ACTION_ALLOWED_REMOVE_EDGES", "ACTION_REQUIRED_REMOVE_EDGES",
    "FORBIDDEN_ANCHOR_FIELDS", "ARM_IDS", "ARM_K0", "ARM_GRAPH_A_MEMORY",
    "ARM_GRAPH_A_FULL", "ARM_GRAPH_B_FULL", "apply_pre_block_v15b",
    "apply_post_block_v15b", "build_signed_edit_graph_v15b",
    "build_source_role_content_memory_v15b",
    "build_position_calibration_fixture_v15b",
    "build_position_counterfactual_reference_v15b",
    "build_target_native_motion_reference_v15b",
    "build_target_native_transport_v15b",
    "build_target_role_state_v15b",
    "compute_monotonic_event_warp_v15b", "compare_anchor_graphs_v15b",
    "diagnose_four_anchor_consensus_v15b", "object_sha256", "tensor_sha256",
}]
