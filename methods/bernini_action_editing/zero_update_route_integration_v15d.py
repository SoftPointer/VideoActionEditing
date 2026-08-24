"""Fail-closed CPU scaffold for the v15d no-anchor route seam.

This module is deliberately *not* a model runner.  It imports neither a
Bernini controller nor a trainer, owns no tensor runtime, and cannot enable a
route, decode, optimizer, training run, or scientific claim.  Its only job is
to make the intended future integration mechanically explicit:

* the native token grid is exactly ``21 x 37 x 25``;
* route selection is by 22 typed physical block addresses in the exact
  30-block inventory, never by a hook ordinal;
* one immutable instance-ownership ledger plans coordinates across all 30
  physical blocks and both CFG branches for every denoising step;
* the four preregistered A/B/C/D arms contain exact-zero planned write scales;
  this module does not emit or observe writes;
* the target ABI has no media/latent/Gaussian/path/file-descriptor inlet;
* target action data contains only #2 trajectory and #2 -> #3 contact/pour
  timing.  Color, material, shape, and liquid appearance (including the
  source-visible amber liquid) are excluded from prompt/graph payloads and
  remain source-authority-owned properties;
* source-future tracks are not an input to target motion;
* a vacated #2 initial site is owned by HOLE, never restored as source #2;
* source object features can come only from role-local property memory;
* a future background carrier is required to bind the same source static
  re-forward runtime and to scope only strict background/support, never
  objects, corridor, contact, or HOLE.

All provenance accepted here is caller-supplied and explicitly unauthenticated.
Consequently every *planned write scale* is zero and every
execution/training/science authorization is hard false.  Passing these checks
means only that an unexecuted synthetic plan survived local replay; it is not
evidence that SAM2, Bernini, a route, a write, or a GPU executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


METHOD = "bernini-zero-update-route-integration-v15d-r4"
DECISION = "UNEXECUTED_SYNTHETIC_PLAN_R4_PASS__RUNTIME_ROUTE_TRAINING_SCIENCE_NO_GO"
CONTRACT_SCHEMA = "bernini-zero-update-route-integration-contract-v15d-r4"
TARGET_ABI_SCHEMA = "bernini-no-anchor-target-abi-v15d-r4"
MOTION_SCHEMA = "bernini-target-native-motion-plan-v15d-r4"
PROPERTY_MEMORY_SCHEMA = "bernini-source-property-memory-boundary-v15d-r4"
BACKGROUND_CARRIER_SCHEMA = "bernini-source-static-background-carrier-v15d-r4"
LEDGER_SCHEMA = "bernini-single-instance-ledger-synthetic-plan-v15d-r4"
CELL_AUDIT_SCHEMA = "bernini-synthetic-zero-write-cell-plan-v15d-r4"
RUN_RECEIPT_SCHEMA = "bernini-unexecuted-synthetic-plan-receipt-v15d-r4"
BUNDLE_SCHEMA = "bernini-unexecuted-synthetic-plan-bundle-v15d-r4"

GRID = (21, 37, 25)
TEMPORAL_PHASES, HEIGHT, WIDTH = GRID
SPATIAL_CELLS = HEIGHT * WIDTH
TOKEN_CELLS = TEMPORAL_PHASES * SPATIAL_CELLS
DENOISE_STEPS = 40
CFG_BRANCHES = ("negative", "conditional")
BLOCK_NAMESPACE = "bernini_r_1p3b.transformer.blocks"
PHYSICAL_BLOCK_IDS = tuple(range(30))
ROUTE_PHYSICAL_BLOCK_IDS = (
    1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23,
    25, 26, 27, 29,
)

HUMAN_ROLE = "human_agent"
OLD_ACTOR_ROLE = "old_actor_1"
NEW_ACTOR_ROLE = "new_actor_2"
RECIPIENT_ROLE = "recipient_3"
HOLE_ROLE = "HOLE"
BACKGROUND_ROLE = "background"
FOREGROUND_ROLES = (
    HUMAN_ROLE, OLD_ACTOR_ROLE, NEW_ACTOR_ROLE, RECIPIENT_ROLE,
)
LEDGER_ROLES = FOREGROUND_ROLES + (HOLE_ROLE, BACKGROUND_ROLE)

ARM_A = "A_NATIVE_FROZEN_OBSERVERS_ONLY"
ARM_B = "B_LEDGER_PROPERTY_BACKGROUND_GRAPH0"
ARM_C = "C_SIGNED_GRAPH_V0"
ARM_D = "D_APPEARANCE_COUNTERFACTUAL_GRAPH_V1"
ARM_IDS = (ARM_A, ARM_B, ARM_C, ARM_D)

EDITABLE_ACTION_FACTORS = (
    "new_actor_2.target_native_trajectory",
    "new_actor_2_to_recipient_3.contact_timing",
    "new_actor_2_to_recipient_3.pour_timing",
)
SOURCE_OWNED_PROPERTY_FACTORS = (
    "human_agent.color", "human_agent.material", "human_agent.shape",
    "old_actor_1.color", "old_actor_1.material", "old_actor_1.shape",
    "new_actor_2.color", "new_actor_2.material", "new_actor_2.shape",
    "recipient_3.color", "recipient_3.material", "recipient_3.shape",
    "liquid.color", "liquid.material", "liquid.shape",
    "liquid.appearance",
)
CANONICAL_TARGET_ACTION_TEXT = (
    "#2 follows the registered target-native trajectory; #2 contacts #3 and "
    "performs the registered pour timing."
)
IDENTITY_LOCK_VALUE = "source-bound identity only; not a learnable appearance target"

WRITE_COMPONENTS = (
    "hidden_residual",
    "pre_rope_key_rewrite",
    "value_rewrite",
    "attention_output_rewrite",
    "source_property_restore",
    "source_background_restore",
    "hole_completion",
    "graph_route_v0",
    "graph_route_v1",
    "parameter_update",
)
ZERO_WRITE_SCALES = {name: 0.0 for name in WRITE_COMPONENTS}

AUTHORIZATION_FIELDS = (
    "external_source_authority_passed",
    "gpu_execution_authorized",
    "route_execution_authorized",
    "decode_authorized",
    "optimizer_authorized",
    "training_authorized",
    "scientific_claim_authorized",
    "nonzero_write_authorized",
)

SOURCE_AUTHORITY_STATE = "EXTERNAL_SOURCE_AUTHORITY_REQUIRED_NOT_PRESENT"
TARGET_MOTION_PROVENANCE = (
    "caller_supplied_target_native_transport_unverified_no_source_future_input"
)
PROPERTY_PROVENANCE = "caller_supplied_source_property_memory_unverified"
BACKGROUND_PROVENANCE = (
    "caller_supplied_same_source_static_reforward_runtime_unverified"
)
OBJECT_FEATURE_ROUTE = "source_property_memory_only"
BACKGROUND_FEATURE_ROUTE = "same_source_static_reforward_background_only"
TARGET_MOTION_ROUTE = "target_native_transport_only_no_source_future_tracks"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
MAX_INPUT_NESTING_DEPTH = 32
MAX_INPUT_COLLECTION_ITEMS = 100_000
MAX_INPUT_NODES = 1_000_000
MAX_INPUT_STRING_CHARS = 1_000_000

# These terms are not forbidden source properties.  They are forbidden only in
# the action text/graph because appearance must be supplied by source authority,
# not repeated as a target or graph objective.  In E00, for example, amber is
# visibly a source property; it still does not belong in the action payload.
APPEARANCE_ACTION_TERMS = frozenset(
    {
        "amber", "white", "black", "red", "green", "blue", "yellow",
        "orange", "purple", "pink", "brown", "gray", "grey", "silver",
        "gold", "golden", "clear", "transparent", "opaque", "glass",
        "ceramic", "porcelain", "metal", "metallic", "wood", "wooden",
        "plastic", "small", "large", "round", "square", "tall", "short",
        "liquid appearance", "liquid color", "colour", "color", "material",
        "texture", "shape",
    }
)

FORBIDDEN_TARGET_KEY_PARTS = (
    "anchor", "donor", "source_future_track", "future_source_track",
    "reference_video", "reference_media",
)
FORBIDDEN_TARGET_KEY_EXACT = frozenset(
    {
        "video", "rgb", "pixels", "latent", "gaussian", "initial_gaussian",
        "path", "file_path", "filepath", "fd", "file_descriptor",
        "file_handle", "handle", "stream", "bytes", "tensor",
    }
)

UNRESOLVED_DEPENDENCIES = (
    "externally authenticated v15c source authority for human/#1/#2/#3",
    "production proof that v15b property memory was extracted from that same source",
    "production proof of position removal for source property memory",
    "production target-role transport geometry/shape-preservation gate from v15b-r8",
    "runtime binding of the actual Bernini 30 physical block objects to these IDs",
    "externally authenticated target-native motion provenance independent of source-future tracks",
    "same-source static re-forward/runtime authentication for the background carrier",
    "overlap-aware human/vessel contact and corridor rasterization",
    "real per-cell route instrumentation for mask-exterior delta and HOLE/object-source checks",
    "human review of complete source tracks and appearance-counterfactual Graph-v1",
    "separate route/decode/training authorization; this scaffold can never grant it",
)


class V15DContractError(RuntimeError):
    """Raised whenever the scaffold would otherwise widen authority."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise V15DContractError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V15DContractError(f"canonical JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def canonical_json_loads_v15d(data: bytes) -> Any:
    """Parse canonical JSON bytes with duplicate-field and alias rejection."""

    if not isinstance(data, bytes):
        raise V15DContractError("fresh consumer requires immutable canonical bytes")
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_json_object_no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise V15DContractError("cannot parse canonical JSON bytes") from error
    _scan_no_anchor_value(
        value, label="canonical serialized material", _forbid_target_fields=False
    )
    if canonical_json_bytes(value) != data:
        raise V15DContractError("serialized evidence is not exact canonical JSON")
    # A second canonical round trip creates an ordinary fresh object graph and
    # prevents caller mappings/lists from aliasing the validator's material.
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise V15DContractError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if (
        type(value) is not int
        or value < minimum
        or value > maximum
    ):
        raise V15DContractError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _exact_int_value(value: Any, expected: int, *, label: str) -> int:
    if type(value) is not int or value != expected:
        raise V15DContractError(f"{label} must be exact integer {expected}")
    return value


def hard_false_authorizations_v15d() -> dict[str, bool]:
    """Return fresh material; no mutable mapping is an authorization root."""

    return {field: False for field in AUTHORIZATION_FIELDS}


def _validate_hard_false_authorizations(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise V15DContractError(f"{label} must be a mapping")
    _exact_keys(value, set(AUTHORIZATION_FIELDS), label=label)
    for field in AUTHORIZATION_FIELDS:
        if value[field] is not False:
            raise V15DContractError(f"{label}.{field} must be the bool False")


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise V15DContractError(f"{label} must be a mapping")
    try:
        keys = set(value)
    except (TypeError, ValueError) as error:
        raise V15DContractError(f"{label} fields cannot be enumerated") from error
    if keys != expected:
        missing = sorted(repr(item) for item in expected - keys)
        extra = sorted(repr(item) for item in keys - expected)
        raise V15DContractError(
            f"{label} fields differ; missing={missing}, extra={extra}"
        )


def _json_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise V15DContractError(f"{label} must be a JSON list")
    return value


def _strict_ascending_unique_int_json_list(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    """Validate serialized set material without normalizing caller order."""

    material = _json_list(value, label=label)
    result = tuple(
        _exact_int(
            item, label=f"{label}[{index}]", minimum=minimum, maximum=maximum
        )
        for index, item in enumerate(material)
    )
    if len(set(result)) != len(result):
        raise V15DContractError(
            f"{label} contains duplicate values; must be unique as submitted"
        )
    if result != tuple(sorted(result)):
        raise V15DContractError(
            f"{label} must be strictly ascending as submitted"
        )
    return result


def _finite_zero(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V15DContractError(f"{label} must be numeric exact zero")
    result = float(value)
    if not math.isfinite(result) or result != 0.0:
        raise V15DContractError(f"{label} must remain exact zero")
    return result


def _address_payload(physical_id: int) -> dict[str, Any]:
    return {
        "address_kind": "physical_transformer_block",
        "namespace": BLOCK_NAMESPACE,
        "physical_id": physical_id,
    }


def _arm_specs() -> list[dict[str, Any]]:
    return [
        {
            "arm_id": ARM_A,
            "graph_slot": None,
            "intended_components_after_future_authorization": [],
            "current_write_policy": "synthetic_plan_exact_zero_not_runtime_observation",
        },
        {
            "arm_id": ARM_B,
            "graph_slot": None,
            "intended_components_after_future_authorization": [
                "instance_ledger", "source_property_memory",
                "strict_source_background_restore", "hole_guard",
            ],
            "current_write_policy": "synthetic_plan_exact_zero_not_runtime_observation",
        },
        {
            "arm_id": ARM_C,
            "graph_slot": "v0",
            "intended_components_after_future_authorization": [
                "instance_ledger", "source_property_memory",
                "strict_source_background_restore", "hole_guard",
                "signed_action_graph_v0",
            ],
            "current_write_policy": "synthetic_plan_exact_zero_not_runtime_observation",
        },
        {
            "arm_id": ARM_D,
            "graph_slot": "v1",
            "intended_components_after_future_authorization": [
                "instance_ledger", "source_property_memory",
                "strict_source_background_restore", "hole_guard",
                "appearance_counterfactual_action_graph_v1",
            ],
            "current_write_policy": "synthetic_plan_exact_zero_not_runtime_observation",
        },
    ]


def expected_contract_v15d() -> dict[str, Any]:
    """Return the only accepted local v15d preregistration contract."""

    return {
        "schema_version": CONTRACT_SCHEMA,
        "method": METHOD,
        "decision": DECISION,
        "scope": "unexecuted_local_cpu_synthetic_plan_only",
        "runtime_execution_observed": False,
        "cell_audits_are_synthetic_plan": True,
        "fresh_bundle_validation_required": True,
        "standalone_receipt_authority": False,
        "grid": list(GRID),
        "denoise_steps": DENOISE_STEPS,
        "cfg_branches": list(CFG_BRANCHES),
        "physical_block_address_kind": "physical_transformer_block",
        "physical_block_namespace": BLOCK_NAMESPACE,
        "physical_block_ids": list(PHYSICAL_BLOCK_IDS),
        "route_physical_block_ids": list(ROUTE_PHYSICAL_BLOCK_IDS),
        "route_physical_block_count": len(ROUTE_PHYSICAL_BLOCK_IDS),
        "ordinal_block_addressing_forbidden": True,
        "ledger_roles": list(LEDGER_ROLES),
        "single_ledger_across_all_blocks_and_cfg": True,
        "action_contract": {
            "action_id": "pour",
            "actor_role": NEW_ACTOR_ROLE,
            "recipient_role": RECIPIENT_ROLE,
            "canonical_target_action_text": CANONICAL_TARGET_ACTION_TEXT,
            "editable_factors": list(EDITABLE_ACTION_FACTORS),
            "source_owned_nonlearnable_property_factors": list(
                SOURCE_OWNED_PROPERTY_FACTORS
            ),
            "identity_locks": {
                role: IDENTITY_LOCK_VALUE for role in FOREGROUND_ROLES
            },
            "appearance_values_in_prompt_or_graph_forbidden": True,
            "source_visible_amber_is_source_property_not_action_target": True,
        },
        "source_future_track_policy": (
            "not_an_input_to_target_motion_or_action_graph"
        ),
        "vacated_new_actor_2_initial_site_policy": "HOLE_no_source_object_restore",
        "object_feature_policy": OBJECT_FEATURE_ROUTE,
        "background_carrier_policy": {
            "provenance": "same_source_static_reforward_runtime",
            "allowed_scope": [BACKGROUND_ROLE, "strict_support"],
            "forbidden_scope": [
                HUMAN_ROLE, OLD_ACTOR_ROLE, NEW_ACTOR_ROLE, RECIPIENT_ROLE,
                "motion_corridor", "contact", HOLE_ROLE,
            ],
            "outside_allowed_mask_delta": "exact_zero_per_execution_cell",
        },
        "arms": _arm_specs(),
        "default_write_scales": dict(ZERO_WRITE_SCALES),
        "source_authority_state": SOURCE_AUTHORITY_STATE,
        "authorizations": hard_false_authorizations_v15d(),
        "unresolved_dependencies": list(UNRESOLVED_DEPENDENCIES),
    }


def validate_contract_v15d(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any preregistration drift, including a resealed widening."""

    expected = expected_contract_v15d()
    current_bytes = canonical_json_bytes(value)
    expected_bytes = canonical_json_bytes(expected)
    if current_bytes != expected_bytes:
        raise V15DContractError("v15d contract differs from exact preregistration")
    current = canonical_json_loads_v15d(current_bytes)
    if not isinstance(current, dict):
        raise V15DContractError("v15d contract root differs")
    return current


def load_contract_v15d(path: str | os.PathLike[str]) -> dict[str, Any]:
    path_obj = Path(path)
    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise V15DContractError("cannot read v15d contract") from error
    if not isinstance(value, Mapping):
        raise V15DContractError("v15d contract root must be a mapping")
    return validate_contract_v15d(value)


@dataclass(frozen=True)
class PhysicalBlockAddressV15D:
    address_kind: str
    namespace: str
    physical_id: int

    def __post_init__(self) -> None:
        if self.address_kind != "physical_transformer_block":
            raise V15DContractError("block address must explicitly be physical")
        if self.namespace != BLOCK_NAMESPACE:
            raise V15DContractError("physical block namespace differs")
        _exact_int(
            self.physical_id,
            label="physical block ID",
            minimum=0,
            maximum=len(PHYSICAL_BLOCK_IDS) - 1,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhysicalBlockAddressV15D":
        _exact_keys(
            value,
            {"address_kind", "namespace", "physical_id"},
            label="physical block address",
        )
        return cls(
            value["address_kind"], value["namespace"], value["physical_id"]
        )

    @classmethod
    def for_id(cls, physical_id: int) -> "PhysicalBlockAddressV15D":
        return cls.from_mapping(_address_payload(physical_id))

    def payload(self) -> dict[str, Any]:
        return _address_payload(self.physical_id)


def _scan_no_anchor_value(
    value: Any,
    *,
    label: str = "target ABI",
    _depth: int = 0,
    _active: set[int] | None = None,
    _node_count: list[int] | None = None,
    _forbid_target_fields: bool = True,
) -> None:
    """Recursively reject every media/path/FD-shaped inlet before parsing."""

    if _depth > MAX_INPUT_NESTING_DEPTH:
        raise V15DContractError(f"{label} exceeds maximum nesting depth")
    active = set() if _active is None else _active
    node_count = [0] if _node_count is None else _node_count
    node_count[0] += 1
    if node_count[0] > MAX_INPUT_NODES:
        raise V15DContractError("target ABI exceeds maximum material size")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise V15DContractError(f"{label} contains a cyclic mapping")
        if len(value) > MAX_INPUT_COLLECTION_ITEMS:
            raise V15DContractError(f"{label} mapping is too large")
        active.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise V15DContractError(f"{label} contains a non-string field")
                if len(key) > MAX_INPUT_STRING_CHARS:
                    raise V15DContractError(f"{label} contains an oversized field")
                normalized = key.casefold().replace("-", "_")
                if _forbid_target_fields and (
                    normalized in FORBIDDEN_TARGET_KEY_EXACT
                    or any(part in normalized for part in FORBIDDEN_TARGET_KEY_PARTS)
                    or normalized.endswith("_path")
                    or normalized.endswith("_fd")
                ):
                    raise V15DContractError(
                        f"{label} contains forbidden no-anchor field {key!r}"
                    )
                _scan_no_anchor_value(
                    child, label=f"{label}.{key}", _depth=_depth + 1,
                    _active=active, _node_count=node_count,
                    _forbid_target_fields=_forbid_target_fields,
                )
        finally:
            active.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise V15DContractError(f"{label} contains a cyclic sequence")
        if len(value) > MAX_INPUT_COLLECTION_ITEMS:
            raise V15DContractError(f"{label} sequence is too large")
        active.add(identity)
        try:
            for index, child in enumerate(value):
                _scan_no_anchor_value(
                    child, label=f"{label}[{index}]", _depth=_depth + 1,
                    _active=active, _node_count=node_count,
                    _forbid_target_fields=_forbid_target_fields,
                )
        finally:
            active.remove(identity)
        return
    if isinstance(value, (bytes, bytearray, memoryview, os.PathLike, io.IOBase)):
        raise V15DContractError(f"{label} contains media/path/FD-like material")
    if isinstance(value, str):
        if len(value) > MAX_INPUT_STRING_CHARS:
            raise V15DContractError(f"{label} string is too large")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise V15DContractError(f"{label} contains a non-finite number")
    if value is None or isinstance(value, (int, float, bool)):
        return
    raise V15DContractError(f"{label} contains unsupported material type")


def _appearance_terms_in_text(text: str) -> tuple[str, ...]:
    lowered = " ".join(text.casefold().replace("-", " ").split())
    found: list[str] = []
    for term in sorted(APPEARANCE_ACTION_TERMS):
        if " " in term:
            present = term in lowered
        else:
            present = re.search(rf"\b{re.escape(term)}\b", lowered) is not None
        if present:
            found.append(term)
    return tuple(found)


@dataclass(frozen=True)
class NoAnchorTargetABIV15D:
    schema_version: str
    case_id: str
    arm_id: str
    target_action: Mapping[str, Any]
    grid: tuple[int, int, int]
    denoise_steps: int
    cfg_branches: tuple[str, ...]
    physical_block_inventory: tuple[PhysicalBlockAddressV15D, ...]
    route_physical_allowlist: tuple[PhysicalBlockAddressV15D, ...]
    source_authority_state: str
    requested_write_scales: Mapping[str, float]
    authorizations: Mapping[str, bool]

    def __post_init__(self) -> None:
        if self.schema_version != TARGET_ABI_SCHEMA:
            raise V15DContractError("target ABI schema differs")
        if not isinstance(self.case_id, str) or CASE_ID_RE.fullmatch(self.case_id) is None:
            raise V15DContractError("target ABI case ID is not canonical")
        if self.arm_id not in ARM_IDS:
            raise V15DContractError("target ABI arm is not preregistered")
        if not isinstance(self.target_action, Mapping):
            raise V15DContractError("target action must be a mapping")
        expected_action = expected_contract_v15d()["action_contract"]
        if canonical_json_bytes(self.target_action) != canonical_json_bytes(expected_action):
            text = self.target_action.get("canonical_target_action_text", "")
            if isinstance(text, str) and _appearance_terms_in_text(text):
                raise V15DContractError(
                    "target action text contains source-owned appearance terms"
                )
            raise V15DContractError("target action differs from action-only contract")
        if (
            not isinstance(self.grid, tuple)
            or len(self.grid) != len(GRID)
        ):
            raise V15DContractError("target ABI native geometry/schedule differs")
        for axis, (current, expected) in enumerate(zip(self.grid, GRID)):
            _exact_int_value(current, expected, label=f"target ABI grid axis {axis}")
        _exact_int_value(
            self.denoise_steps, DENOISE_STEPS, label="target ABI denoise steps"
        )
        if not isinstance(self.cfg_branches, tuple) or self.cfg_branches != CFG_BRANCHES:
            raise V15DContractError("target ABI CFG branches differ")
        expected_inventory = tuple(
            PhysicalBlockAddressV15D.for_id(value) for value in PHYSICAL_BLOCK_IDS
        )
        expected_allowlist = tuple(
            PhysicalBlockAddressV15D.for_id(value)
            for value in ROUTE_PHYSICAL_BLOCK_IDS
        )
        if (
            not isinstance(self.physical_block_inventory, tuple)
            or self.physical_block_inventory != expected_inventory
        ):
            raise V15DContractError("target ABI physical inventory differs")
        if (
            not isinstance(self.route_physical_allowlist, tuple)
            or self.route_physical_allowlist != expected_allowlist
        ):
            raise V15DContractError("target ABI physical route allowlist differs")
        if self.source_authority_state != SOURCE_AUTHORITY_STATE:
            raise V15DContractError("target ABI cannot claim external source authority")
        if not isinstance(self.requested_write_scales, Mapping):
            raise V15DContractError("target ABI write scales must be a mapping")
        _validate_zero_write_scales(self.requested_write_scales)
        if (
            not isinstance(self.authorizations, Mapping)
        ):
            raise V15DContractError("target ABI authorizations must be a mapping")
        _validate_hard_false_authorizations(
            self.authorizations, label="target ABI authorizations"
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NoAnchorTargetABIV15D":
        _scan_no_anchor_value(value)
        expected_fields = {
            "schema_version", "case_id", "arm_id", "target_action", "grid",
            "denoise_steps", "cfg_branches", "physical_block_inventory",
            "route_physical_allowlist", "source_authority_state",
            "requested_write_scales", "authorizations",
        }
        _exact_keys(value, expected_fields, label="target ABI")
        inventory_raw = _json_list(
            value["physical_block_inventory"], label="physical inventory"
        )
        allowlist_raw = _json_list(
            value["route_physical_allowlist"], label="route allowlist"
        )
        inventory = tuple(
            PhysicalBlockAddressV15D.from_mapping(item)
            for item in inventory_raw
        )
        allowlist = tuple(
            PhysicalBlockAddressV15D.from_mapping(item)
            for item in allowlist_raw
        )
        _strict_ascending_unique_int_json_list(
            [item.physical_id for item in inventory],
            label="physical inventory IDs",
            minimum=0,
            maximum=len(PHYSICAL_BLOCK_IDS) - 1,
        )
        _strict_ascending_unique_int_json_list(
            [item.physical_id for item in allowlist],
            label="route allowlist IDs",
            minimum=0,
            maximum=len(PHYSICAL_BLOCK_IDS) - 1,
        )
        grid_raw = _json_list(value["grid"], label="target ABI grid")
        cfg_raw = _json_list(value["cfg_branches"], label="target ABI CFG branches")
        if not isinstance(value["requested_write_scales"], Mapping):
            raise V15DContractError("target ABI write scales must be a mapping")
        if not isinstance(value["authorizations"], Mapping):
            raise V15DContractError("target ABI authorizations must be a mapping")
        return cls(
            schema_version=value["schema_version"],
            case_id=value["case_id"],
            arm_id=value["arm_id"],
            target_action=value["target_action"],
            grid=tuple(grid_raw),
            denoise_steps=value["denoise_steps"],
            cfg_branches=tuple(cfg_raw),
            physical_block_inventory=inventory,
            route_physical_allowlist=allowlist,
            source_authority_state=value["source_authority_state"],
            requested_write_scales=dict(value["requested_write_scales"]),
            authorizations=dict(value["authorizations"]),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "target_action": json.loads(canonical_json_bytes(self.target_action)),
            "grid": list(self.grid),
            "denoise_steps": self.denoise_steps,
            "cfg_branches": list(self.cfg_branches),
            "physical_block_inventory": [
                item.payload() for item in self.physical_block_inventory
            ],
            "route_physical_allowlist": [
                item.payload() for item in self.route_physical_allowlist
            ],
            "source_authority_state": self.source_authority_state,
            "requested_write_scales": dict(self.requested_write_scales),
            "authorizations": dict(self.authorizations),
        }

    @property
    def digest(self) -> str:
        return object_sha256(self.payload())


def build_no_anchor_target_abi_v15d(
    *, case_id: str, arm_id: str
) -> NoAnchorTargetABIV15D:
    contract = expected_contract_v15d()
    value = {
        "schema_version": TARGET_ABI_SCHEMA,
        "case_id": case_id,
        "arm_id": arm_id,
        "target_action": contract["action_contract"],
        "grid": contract["grid"],
        "denoise_steps": contract["denoise_steps"],
        "cfg_branches": contract["cfg_branches"],
        "physical_block_inventory": [
            _address_payload(block_id) for block_id in PHYSICAL_BLOCK_IDS
        ],
        "route_physical_allowlist": [
            _address_payload(block_id) for block_id in ROUTE_PHYSICAL_BLOCK_IDS
        ],
        "source_authority_state": SOURCE_AUTHORITY_STATE,
        "requested_write_scales": dict(ZERO_WRITE_SCALES),
        "authorizations": hard_false_authorizations_v15d(),
    }
    return NoAnchorTargetABIV15D.from_mapping(value)


def _normalize_indices(values: Iterable[int], *, label: str) -> frozenset[int]:
    try:
        material = tuple(values)
    except TypeError as error:
        raise V15DContractError(f"{label} must be an iterable of token indices") from error
    result: set[int] = set()
    for index, value in enumerate(material):
        result.add(
            _exact_int(
                value,
                label=f"{label}[{index}]",
                minimum=0,
                maximum=TOKEN_CELLS - 1,
            )
        )
    if len(result) != len(material):
        raise V15DContractError(f"{label} contains duplicate token indices")
    return frozenset(result)


def _normalize_spatial_indices(
    values: Iterable[int], *, label: str
) -> frozenset[int]:
    try:
        material = tuple(values)
    except TypeError as error:
        raise V15DContractError(f"{label} must be an iterable of spatial indices") from error
    result: set[int] = set()
    for index, value in enumerate(material):
        result.add(
            _exact_int(
                value,
                label=f"{label}[{index}]",
                minimum=0,
                maximum=SPATIAL_CELLS - 1,
            )
        )
    if len(result) != len(material):
        raise V15DContractError(f"{label} contains duplicate spatial indices")
    return frozenset(result)


def _normalize_phases(values: Iterable[int], *, label: str) -> tuple[int, ...]:
    try:
        material = tuple(values)
    except TypeError as error:
        raise V15DContractError(f"{label} must be an iterable of phases") from error
    result = tuple(
        _exact_int(
            value,
            label=f"{label}[{index}]",
            minimum=0,
            maximum=TEMPORAL_PHASES - 1,
        )
        for index, value in enumerate(material)
    )
    if not result or result != tuple(sorted(set(result))):
        raise V15DContractError(f"{label} must be nonempty, sorted, and unique")
    return result


def _role_index_payload(role_indices: tuple[tuple[str, frozenset[int]], ...]) -> list[Any]:
    return [[role, sorted(indices)] for role, indices in role_indices]


@dataclass(frozen=True)
class TargetNativeMotionPlanV15D:
    """Caller-supplied target masks with no source-future-track ABI field."""

    schema_version: str
    grid: tuple[int, int, int]
    role_indices: tuple[tuple[str, frozenset[int]], ...]
    motion_corridor_indices: frozenset[int]
    contact_indices: frozenset[int]
    contact_phases: tuple[int, ...]
    pour_phases: tuple[int, ...]
    provenance_kind: str
    source_future_track_input_count: int
    externally_authenticated: bool
    digest: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != MOTION_SCHEMA
            or not isinstance(self.grid, tuple)
            or len(self.grid) != len(GRID)
        ):
            raise V15DContractError("target motion schema/grid differs")
        for axis, (current, expected) in enumerate(zip(self.grid, GRID)):
            _exact_int_value(current, expected, label=f"target motion grid axis {axis}")
        if (
            not isinstance(self.role_indices, tuple)
            or any(
                not isinstance(item, tuple) or len(item) != 2
                for item in self.role_indices
            )
        ):
            raise V15DContractError("target motion role bindings must be immutable pairs")
        if tuple(role for role, _ in self.role_indices) != FOREGROUND_ROLES:
            raise V15DContractError("target motion must bind four ordered roles")
        if self.provenance_kind != TARGET_MOTION_PROVENANCE:
            raise V15DContractError("target motion provenance differs")
        if (
            isinstance(self.source_future_track_input_count, bool)
            or not isinstance(self.source_future_track_input_count, int)
            or self.source_future_track_input_count != 0
        ):
            raise V15DContractError("source-future tracks cannot enter target motion")
        if self.externally_authenticated is not False:
            raise V15DContractError("local target motion cannot claim external authority")
        role_map = dict(self.role_indices)
        occupied: set[int] = set()
        for role in FOREGROUND_ROLES:
            indices = role_map[role]
            if not isinstance(indices, frozenset):
                raise V15DContractError("target role indices must be immutable")
            for token_index in indices:
                _exact_int(
                    token_index, label=f"target {role} token", minimum=0,
                    maximum=TOKEN_CELLS - 1,
                )
            if occupied.intersection(indices):
                raise V15DContractError("target foreground role masks overlap")
            occupied.update(indices)
            phase_counts = [0] * TEMPORAL_PHASES
            for token_index in indices:
                phase_counts[token_index // SPATIAL_CELLS] += 1
            if any(count == 0 for count in phase_counts):
                raise V15DContractError(
                    f"target role {role} must be present in every temporal phase"
                )
        for label, indices in (
            ("motion corridor", self.motion_corridor_indices),
            ("contact", self.contact_indices),
        ):
            if not isinstance(indices, frozenset):
                raise V15DContractError(f"{label} indices must be immutable")
            for index in indices:
                _exact_int(
                    index, label=f"{label} token", minimum=0,
                    maximum=TOKEN_CELLS - 1,
                )
        observed_contact_phases = tuple(sorted({
            index // SPATIAL_CELLS for index in self.contact_indices
        }))
        if not isinstance(self.contact_phases, tuple) or not isinstance(self.pour_phases, tuple):
            raise V15DContractError("contact/pour timing must be immutable tuples")
        if _normalize_phases(self.contact_phases, label="contact timing") != self.contact_phases:
            raise V15DContractError("contact timing differs")
        if _normalize_phases(self.pour_phases, label="pour timing") != self.pour_phases:
            raise V15DContractError("pour timing differs")
        if self.contact_phases != observed_contact_phases or not self.contact_phases:
            raise V15DContractError(
                "contact timing must equal the phases represented by contact indices"
            )
        if (
            not self.pour_phases
            or self.pour_phases != tuple(sorted(set(self.pour_phases)))
            or any(phase < 0 or phase >= TEMPORAL_PHASES for phase in self.pour_phases)
            or self.pour_phases[0] < self.contact_phases[0]
        ):
            raise V15DContractError(
                "pour timing must be nonempty/canonical and cannot precede first contact"
            )
        expected_digest = object_sha256(self._payload_without_digest())
        if self.digest != expected_digest:
            raise V15DContractError("target motion digest differs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grid": list(self.grid),
            "role_indices": _role_index_payload(self.role_indices),
            "motion_corridor_indices": sorted(self.motion_corridor_indices),
            "contact_indices": sorted(self.contact_indices),
            "contact_phases": list(self.contact_phases),
            "pour_phases": list(self.pour_phases),
            "provenance_kind": self.provenance_kind,
            "source_future_track_input_count": self.source_future_track_input_count,
            "externally_authenticated": self.externally_authenticated,
        }

    def payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "digest": self.digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TargetNativeMotionPlanV15D":
        fields = {
            "schema_version", "grid", "role_indices",
            "motion_corridor_indices", "contact_indices", "contact_phases",
            "pour_phases", "provenance_kind", "source_future_track_input_count",
            "externally_authenticated", "digest",
        }
        _exact_keys(value, fields, label="target motion serialized material")
        grid = tuple(_json_list(value["grid"], label="target motion grid"))
        role_rows = _json_list(
            value["role_indices"], label="target motion role indices"
        )
        if len(role_rows) != len(FOREGROUND_ROLES):
            raise V15DContractError("target motion serialized role count differs")
        ordered: list[tuple[str, frozenset[int]]] = []
        for index, row in enumerate(role_rows):
            row_list = _json_list(row, label=f"target motion role row {index}")
            if len(row_list) != 2 or not isinstance(row_list[0], str):
                raise V15DContractError("target motion serialized role row differs")
            indices = _strict_ascending_unique_int_json_list(
                row_list[1],
                label=f"target motion role {row_list[0]} indices",
                minimum=0,
                maximum=TOKEN_CELLS - 1,
            )
            ordered.append(
                (row_list[0], frozenset(indices))
            )
        return cls(
            value["schema_version"],
            grid,
            tuple(ordered),
            frozenset(_strict_ascending_unique_int_json_list(
                value["motion_corridor_indices"], label="motion corridor",
                minimum=0, maximum=TOKEN_CELLS - 1,
            )),
            frozenset(_strict_ascending_unique_int_json_list(
                value["contact_indices"], label="contact",
                minimum=0, maximum=TOKEN_CELLS - 1,
            )),
            tuple(_json_list(value["contact_phases"], label="contact phases")),
            tuple(_json_list(value["pour_phases"], label="pour phases")),
            value["provenance_kind"],
            value["source_future_track_input_count"],
            value["externally_authenticated"],
            value["digest"],
        )

    @classmethod
    def create(
        cls,
        *,
        role_indices: Mapping[str, Iterable[int]],
        motion_corridor_indices: Iterable[int],
        contact_indices: Iterable[int],
        pour_phases: Iterable[int],
    ) -> "TargetNativeMotionPlanV15D":
        if not isinstance(role_indices, Mapping) or set(role_indices) != set(FOREGROUND_ROLES):
            raise V15DContractError("target motion roles differ")
        ordered = tuple(
            (role, _normalize_indices(role_indices[role], label=f"target {role}"))
            for role in FOREGROUND_ROLES
        )
        corridor = _normalize_indices(motion_corridor_indices, label="motion corridor")
        contact = _normalize_indices(contact_indices, label="contact")
        contact_phases = tuple(sorted({
            index // SPATIAL_CELLS for index in contact
        }))
        pour = _normalize_phases(pour_phases, label="pour timing")
        payload = {
            "schema_version": MOTION_SCHEMA,
            "grid": list(GRID),
            "role_indices": _role_index_payload(ordered),
            "motion_corridor_indices": sorted(corridor),
            "contact_indices": sorted(contact),
            "contact_phases": list(contact_phases),
            "pour_phases": list(pour),
            "provenance_kind": TARGET_MOTION_PROVENANCE,
            "source_future_track_input_count": 0,
            "externally_authenticated": False,
        }
        return cls(
            MOTION_SCHEMA, GRID, ordered, corridor, contact, contact_phases, pour,
            TARGET_MOTION_PROVENANCE, 0, False, object_sha256(payload),
        )


@dataclass(frozen=True)
class SourcePropertyMemoryBoundaryV15D:
    schema_version: str
    source_instance_sha256: str
    source_runtime_receipt_sha256: str
    source_static_reforward_receipt_sha256: str
    role_slot_sha256: tuple[tuple[str, str], ...]
    liquid_property_sha256: str
    provenance_kind: str
    externally_authenticated: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != PROPERTY_MEMORY_SCHEMA:
            raise V15DContractError("property-memory schema differs")
        for label, value in (
            ("source instance", self.source_instance_sha256),
            ("source runtime receipt", self.source_runtime_receipt_sha256),
            ("source static re-forward receipt", self.source_static_reforward_receipt_sha256),
            ("source liquid property", self.liquid_property_sha256),
        ):
            _sha256(value, label=label)
        if (
            not isinstance(self.role_slot_sha256, tuple)
            or any(
                not isinstance(item, tuple) or len(item) != 2
                for item in self.role_slot_sha256
            )
        ):
            raise V15DContractError("property role slots must be immutable pairs")
        if tuple(role for role, _ in self.role_slot_sha256) != FOREGROUND_ROLES:
            raise V15DContractError("property memory must bind four source roles")
        for role, digest in self.role_slot_sha256:
            _sha256(digest, label=f"property slot {role}")
        if self.provenance_kind != PROPERTY_PROVENANCE:
            raise V15DContractError("property memory provenance differs")
        if self.externally_authenticated is not False or self.route_authorized is not False:
            raise V15DContractError("local property memory cannot grant authority")
        if self.digest != object_sha256(self._payload_without_digest()):
            raise V15DContractError("property-memory digest differs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_instance_sha256": self.source_instance_sha256,
            "source_runtime_receipt_sha256": self.source_runtime_receipt_sha256,
            "source_static_reforward_receipt_sha256": self.source_static_reforward_receipt_sha256,
            "role_slot_sha256": [list(item) for item in self.role_slot_sha256],
            "liquid_property_sha256": self.liquid_property_sha256,
            "provenance_kind": self.provenance_kind,
            "externally_authenticated": self.externally_authenticated,
            "route_authorized": self.route_authorized,
        }

    def payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "digest": self.digest}

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "SourcePropertyMemoryBoundaryV15D":
        fields = {
            "schema_version", "source_instance_sha256",
            "source_runtime_receipt_sha256",
            "source_static_reforward_receipt_sha256", "role_slot_sha256",
            "liquid_property_sha256", "provenance_kind",
            "externally_authenticated", "route_authorized", "digest",
        }
        _exact_keys(value, fields, label="property-memory serialized material")
        rows = _json_list(value["role_slot_sha256"], label="property role slots")
        ordered: list[tuple[str, str]] = []
        for index, row in enumerate(rows):
            pair = _json_list(row, label=f"property role slot {index}")
            if len(pair) != 2 or not isinstance(pair[0], str):
                raise V15DContractError("property serialized role slot differs")
            ordered.append((pair[0], pair[1]))
        return cls(
            value["schema_version"], value["source_instance_sha256"],
            value["source_runtime_receipt_sha256"],
            value["source_static_reforward_receipt_sha256"], tuple(ordered),
            value["liquid_property_sha256"], value["provenance_kind"],
            value["externally_authenticated"], value["route_authorized"],
            value["digest"],
        )

    @classmethod
    def create(
        cls,
        *,
        source_instance_sha256: str,
        source_runtime_receipt_sha256: str,
        source_static_reforward_receipt_sha256: str,
        role_slot_sha256: Mapping[str, str],
        liquid_property_sha256: str,
    ) -> "SourcePropertyMemoryBoundaryV15D":
        if not isinstance(role_slot_sha256, Mapping) or set(role_slot_sha256) != set(FOREGROUND_ROLES):
            raise V15DContractError("property role slots differ")
        ordered = tuple((role, role_slot_sha256[role]) for role in FOREGROUND_ROLES)
        payload = {
            "schema_version": PROPERTY_MEMORY_SCHEMA,
            "source_instance_sha256": source_instance_sha256,
            "source_runtime_receipt_sha256": source_runtime_receipt_sha256,
            "source_static_reforward_receipt_sha256": source_static_reforward_receipt_sha256,
            "role_slot_sha256": [list(item) for item in ordered],
            "liquid_property_sha256": liquid_property_sha256,
            "provenance_kind": PROPERTY_PROVENANCE,
            "externally_authenticated": False,
            "route_authorized": False,
        }
        return cls(
            PROPERTY_MEMORY_SCHEMA, source_instance_sha256,
            source_runtime_receipt_sha256, source_static_reforward_receipt_sha256,
            ordered, liquid_property_sha256, PROPERTY_PROVENANCE, False, False,
            object_sha256(payload),
        )


@dataclass(frozen=True)
class SourceStaticBackgroundCarrierV15D:
    schema_version: str
    source_instance_sha256: str
    source_runtime_receipt_sha256: str
    source_static_reforward_receipt_sha256: str
    strict_background_indices: frozenset[int]
    strict_support_indices: frozenset[int]
    provenance_kind: str
    externally_authenticated: bool
    route_authorized: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != BACKGROUND_CARRIER_SCHEMA:
            raise V15DContractError("background carrier schema differs")
        for label, value in (
            ("background source instance", self.source_instance_sha256),
            ("background runtime receipt", self.source_runtime_receipt_sha256),
            ("background static re-forward receipt", self.source_static_reforward_receipt_sha256),
        ):
            _sha256(value, label=label)
        if self.provenance_kind != BACKGROUND_PROVENANCE:
            raise V15DContractError("background carrier provenance differs")
        if self.externally_authenticated is not False or self.route_authorized is not False:
            raise V15DContractError("local background carrier cannot grant authority")
        for label, indices in (
            ("strict background", self.strict_background_indices),
            ("strict support", self.strict_support_indices),
        ):
            if not isinstance(indices, frozenset):
                raise V15DContractError(f"{label} carrier indices must be immutable")
            for index in indices:
                _exact_int(
                    index, label=f"{label} carrier token", minimum=0,
                    maximum=TOKEN_CELLS - 1,
                )
        if self.strict_background_indices.intersection(self.strict_support_indices):
            raise V15DContractError("background/support carrier scopes must be disjoint")
        if not self.allowed_indices:
            raise V15DContractError("background carrier must expose a nonempty strict scope")
        if self.digest != object_sha256(self._payload_without_digest()):
            raise V15DContractError("background carrier digest differs")

    @property
    def allowed_indices(self) -> frozenset[int]:
        return self.strict_background_indices.union(self.strict_support_indices)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_instance_sha256": self.source_instance_sha256,
            "source_runtime_receipt_sha256": self.source_runtime_receipt_sha256,
            "source_static_reforward_receipt_sha256": self.source_static_reforward_receipt_sha256,
            "strict_background_indices": sorted(self.strict_background_indices),
            "strict_support_indices": sorted(self.strict_support_indices),
            "provenance_kind": self.provenance_kind,
            "externally_authenticated": self.externally_authenticated,
            "route_authorized": self.route_authorized,
        }

    def payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "digest": self.digest}

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "SourceStaticBackgroundCarrierV15D":
        fields = {
            "schema_version", "source_instance_sha256",
            "source_runtime_receipt_sha256",
            "source_static_reforward_receipt_sha256",
            "strict_background_indices", "strict_support_indices",
            "provenance_kind", "externally_authenticated", "route_authorized",
            "digest",
        }
        _exact_keys(value, fields, label="background-carrier serialized material")
        return cls(
            value["schema_version"], value["source_instance_sha256"],
            value["source_runtime_receipt_sha256"],
            value["source_static_reforward_receipt_sha256"],
            frozenset(_strict_ascending_unique_int_json_list(
                value["strict_background_indices"], label="strict background",
                minimum=0, maximum=TOKEN_CELLS - 1,
            )),
            frozenset(_strict_ascending_unique_int_json_list(
                value["strict_support_indices"], label="strict support",
                minimum=0, maximum=TOKEN_CELLS - 1,
            )),
            value["provenance_kind"], value["externally_authenticated"],
            value["route_authorized"], value["digest"],
        )

    @classmethod
    def create(
        cls,
        *,
        source_instance_sha256: str,
        source_runtime_receipt_sha256: str,
        source_static_reforward_receipt_sha256: str,
        strict_background_indices: Iterable[int],
        strict_support_indices: Iterable[int],
    ) -> "SourceStaticBackgroundCarrierV15D":
        background = _normalize_indices(
            strict_background_indices, label="strict background carrier"
        )
        support = _normalize_indices(
            strict_support_indices, label="strict support carrier"
        )
        payload = {
            "schema_version": BACKGROUND_CARRIER_SCHEMA,
            "source_instance_sha256": source_instance_sha256,
            "source_runtime_receipt_sha256": source_runtime_receipt_sha256,
            "source_static_reforward_receipt_sha256": source_static_reforward_receipt_sha256,
            "strict_background_indices": sorted(background),
            "strict_support_indices": sorted(support),
            "provenance_kind": BACKGROUND_PROVENANCE,
            "externally_authenticated": False,
            "route_authorized": False,
        }
        return cls(
            BACKGROUND_CARRIER_SCHEMA, source_instance_sha256,
            source_runtime_receipt_sha256, source_static_reforward_receipt_sha256,
            background, support, BACKGROUND_PROVENANCE, False, False,
            object_sha256(payload),
        )


def _validate_zero_write_scales(value: Mapping[str, Any]) -> dict[str, float]:
    _exact_keys(value, set(WRITE_COMPONENTS), label="write scales")
    return {
        name: _finite_zero(value[name], label=f"write scale {name}")
        for name in WRITE_COMPONENTS
    }


def _temporalize_spatial(spatial_indices: frozenset[int]) -> frozenset[int]:
    return frozenset(
        phase * SPATIAL_CELLS + index
        for phase in range(TEMPORAL_PHASES)
        for index in spatial_indices
    )


@dataclass(frozen=True)
class CellAuditV15D:
    schema_version: str
    ledger_digest: str
    target_abi_digest: str
    target_motion_digest: str
    property_memory_digest: str
    background_carrier_digest: str
    step_index: int
    cfg_branch: str
    physical_block: PhysicalBlockAddressV15D
    route_allowlisted: bool
    arm_id: str
    outside_allowed_mask_delta_max_abs: float
    hole_source_new_actor_2_restore_count: int
    background_forbidden_scope_write_count: int
    source_object_direct_or_graph_appearance_write_count: int
    object_feature_route: str
    background_feature_route: str
    target_motion_route: str
    write_scales: tuple[tuple[str, float], ...]
    coordinate_nonce: str
    runtime_execution_observed: bool
    synthetic_plan: bool
    digest: str

    def payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ledger_digest": self.ledger_digest,
            "target_abi_digest": self.target_abi_digest,
            "target_motion_digest": self.target_motion_digest,
            "property_memory_digest": self.property_memory_digest,
            "background_carrier_digest": self.background_carrier_digest,
            "step_index": self.step_index,
            "cfg_branch": self.cfg_branch,
            "physical_block": self.physical_block.payload(),
            "route_allowlisted": self.route_allowlisted,
            "arm_id": self.arm_id,
            "outside_allowed_mask_delta_max_abs": self.outside_allowed_mask_delta_max_abs,
            "hole_source_new_actor_2_restore_count": self.hole_source_new_actor_2_restore_count,
            "background_forbidden_scope_write_count": self.background_forbidden_scope_write_count,
            "source_object_direct_or_graph_appearance_write_count": self.source_object_direct_or_graph_appearance_write_count,
            "object_feature_route": self.object_feature_route,
            "background_feature_route": self.background_feature_route,
            "target_motion_route": self.target_motion_route,
            "write_scales": [list(item) for item in self.write_scales],
            "coordinate_nonce": self.coordinate_nonce,
            "runtime_execution_observed": self.runtime_execution_observed,
            "synthetic_plan": self.synthetic_plan,
        }

    def payload(self) -> dict[str, Any]:
        return {**self.payload_without_digest(), "digest": self.digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CellAuditV15D":
        fields = {
            "schema_version", "ledger_digest", "target_abi_digest",
            "target_motion_digest", "property_memory_digest",
            "background_carrier_digest", "step_index", "cfg_branch",
            "physical_block", "route_allowlisted", "arm_id",
            "outside_allowed_mask_delta_max_abs",
            "hole_source_new_actor_2_restore_count",
            "background_forbidden_scope_write_count",
            "source_object_direct_or_graph_appearance_write_count",
            "object_feature_route", "background_feature_route",
            "target_motion_route", "write_scales", "coordinate_nonce",
            "runtime_execution_observed", "synthetic_plan", "digest",
        }
        _exact_keys(value, fields, label="synthetic cell serialized material")
        rows = _json_list(value["write_scales"], label="synthetic cell write scales")
        scales: list[tuple[str, float]] = []
        for index, row in enumerate(rows):
            pair = _json_list(row, label=f"synthetic cell write scale {index}")
            if len(pair) != 2 or not isinstance(pair[0], str):
                raise V15DContractError("synthetic cell write-scale row differs")
            scales.append((pair[0], pair[1]))
        return cls(
            value["schema_version"], value["ledger_digest"],
            value["target_abi_digest"], value["target_motion_digest"],
            value["property_memory_digest"], value["background_carrier_digest"],
            value["step_index"], value["cfg_branch"],
            PhysicalBlockAddressV15D.from_mapping(value["physical_block"]),
            value["route_allowlisted"], value["arm_id"],
            value["outside_allowed_mask_delta_max_abs"],
            value["hole_source_new_actor_2_restore_count"],
            value["background_forbidden_scope_write_count"],
            value["source_object_direct_or_graph_appearance_write_count"],
            value["object_feature_route"], value["background_feature_route"],
            value["target_motion_route"], tuple(scales),
            value["coordinate_nonce"], value["runtime_execution_observed"],
            value["synthetic_plan"], value["digest"],
        )


@dataclass(frozen=True)
class ZeroUpdateRunReceiptV15D:
    schema_version: str
    decision: str
    target_abi_digest: str
    target_motion_digest: str
    property_memory_digest: str
    background_carrier_digest: str
    source_instance_sha256: str
    source_runtime_receipt_sha256: str
    source_static_reforward_receipt_sha256: str
    ledger_digest: str
    ledger_creation_snapshot_sha256: str
    arm_id: str
    cell_count: int
    expected_cell_count: int
    selected_route_cell_count: int
    all_physical_blocks_seen_per_step_cfg: bool
    one_ledger_digest_across_all_cells: bool
    all_planned_cell_write_scales_exact_zero: bool
    all_planned_outside_allowed_mask_deltas_exact_zero: bool
    hole_never_restored_as_source_new_actor_2: bool
    source_objects_use_property_memory_only: bool
    background_scope_strict: bool
    source_future_tracks_used_for_target_motion: bool
    runtime_execution_observed: bool
    cell_audits_are_synthetic_plan: bool
    fresh_bundle_validation_required: bool
    standalone_receipt_authority: bool
    external_source_authority_passed: bool
    route_execution_authorized: bool
    training_authorized: bool
    scientific_claim_authorized: bool
    cell_audit_chain_sha256: str
    unresolved_dependencies: tuple[str, ...]
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != RUN_RECEIPT_SCHEMA:
            raise V15DContractError("run receipt schema differs")
        if self.decision != DECISION:
            raise V15DContractError("run receipt decision differs")
        for label, value in (
            ("run target ABI", self.target_abi_digest),
            ("run target motion", self.target_motion_digest),
            ("run property memory", self.property_memory_digest),
            ("run background carrier", self.background_carrier_digest),
            ("run source instance", self.source_instance_sha256),
            ("run source runtime", self.source_runtime_receipt_sha256),
            ("run source static re-forward", self.source_static_reforward_receipt_sha256),
            ("run ledger", self.ledger_digest),
            ("run ledger creation snapshot", self.ledger_creation_snapshot_sha256),
        ):
            _sha256(value, label=label)
        _sha256(self.cell_audit_chain_sha256, label="run cell audit chain")
        if self.arm_id not in ARM_IDS:
            raise V15DContractError("run receipt arm differs")
        expected_count = DENOISE_STEPS * len(CFG_BRANCHES) * len(PHYSICAL_BLOCK_IDS)
        expected_selected = (
            DENOISE_STEPS * len(CFG_BRANCHES) * len(ROUTE_PHYSICAL_BLOCK_IDS)
        )
        _exact_int_value(
            self.cell_count, expected_count, label="run receipt cell count"
        )
        _exact_int_value(
            self.expected_cell_count, expected_count,
            label="run receipt expected cell count",
        )
        _exact_int_value(
            self.selected_route_cell_count, expected_selected,
            label="run receipt selected route cell count",
        )
        for label, value in (
            ("all physical blocks", self.all_physical_blocks_seen_per_step_cfg),
            ("one ledger", self.one_ledger_digest_across_all_cells),
            ("zero planned writes", self.all_planned_cell_write_scales_exact_zero),
            (
                "zero planned outside-mask deltas",
                self.all_planned_outside_allowed_mask_deltas_exact_zero,
            ),
            ("HOLE guard", self.hole_never_restored_as_source_new_actor_2),
            ("source property-only object route", self.source_objects_use_property_memory_only),
            ("strict background", self.background_scope_strict),
        ):
            if value is not True:
                raise V15DContractError(f"run receipt {label} must be true")
        for label, value in (
            ("source-future target motion", self.source_future_tracks_used_for_target_motion),
            ("runtime execution observation", self.runtime_execution_observed),
            ("external source authority", self.external_source_authority_passed),
            ("route authorization", self.route_execution_authorized),
            ("training authorization", self.training_authorized),
            ("science authorization", self.scientific_claim_authorized),
        ):
            if value is not False:
                raise V15DContractError(f"run receipt {label} must be hard false")
        if self.cell_audits_are_synthetic_plan is not True:
            raise V15DContractError("run receipt cells must be labeled synthetic plan")
        if self.fresh_bundle_validation_required is not True:
            raise V15DContractError("run receipt must require fresh bundle validation")
        if self.standalone_receipt_authority is not False:
            raise V15DContractError("standalone synthetic receipt has no authority")
        if tuple(self.unresolved_dependencies) != UNRESOLVED_DEPENDENCIES:
            raise V15DContractError("run receipt unresolved dependencies differ")
        if self.digest != object_sha256(self.payload_without_digest()):
            raise V15DContractError("run receipt digest differs")

    def payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "target_abi_digest": self.target_abi_digest,
            "target_motion_digest": self.target_motion_digest,
            "property_memory_digest": self.property_memory_digest,
            "background_carrier_digest": self.background_carrier_digest,
            "source_instance_sha256": self.source_instance_sha256,
            "source_runtime_receipt_sha256": self.source_runtime_receipt_sha256,
            "source_static_reforward_receipt_sha256": self.source_static_reforward_receipt_sha256,
            "ledger_digest": self.ledger_digest,
            "ledger_creation_snapshot_sha256": self.ledger_creation_snapshot_sha256,
            "arm_id": self.arm_id,
            "cell_count": self.cell_count,
            "expected_cell_count": self.expected_cell_count,
            "selected_route_cell_count": self.selected_route_cell_count,
            "all_physical_blocks_seen_per_step_cfg": self.all_physical_blocks_seen_per_step_cfg,
            "one_ledger_digest_across_all_cells": self.one_ledger_digest_across_all_cells,
            "all_planned_cell_write_scales_exact_zero": self.all_planned_cell_write_scales_exact_zero,
            "all_planned_outside_allowed_mask_deltas_exact_zero": self.all_planned_outside_allowed_mask_deltas_exact_zero,
            "hole_never_restored_as_source_new_actor_2": self.hole_never_restored_as_source_new_actor_2,
            "source_objects_use_property_memory_only": self.source_objects_use_property_memory_only,
            "background_scope_strict": self.background_scope_strict,
            "source_future_tracks_used_for_target_motion": self.source_future_tracks_used_for_target_motion,
            "runtime_execution_observed": self.runtime_execution_observed,
            "cell_audits_are_synthetic_plan": self.cell_audits_are_synthetic_plan,
            "fresh_bundle_validation_required": self.fresh_bundle_validation_required,
            "standalone_receipt_authority": self.standalone_receipt_authority,
            "external_source_authority_passed": self.external_source_authority_passed,
            "route_execution_authorized": self.route_execution_authorized,
            "training_authorized": self.training_authorized,
            "scientific_claim_authorized": self.scientific_claim_authorized,
            "cell_audit_chain_sha256": self.cell_audit_chain_sha256,
            "unresolved_dependencies": list(self.unresolved_dependencies),
        }

    def payload(self) -> dict[str, Any]:
        return {**self.payload_without_digest(), "digest": self.digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ZeroUpdateRunReceiptV15D":
        expected = set(cls.__dataclass_fields__)
        _exact_keys(value, expected, label="synthetic run receipt")
        material = dict(value)
        material["unresolved_dependencies"] = tuple(
            _json_list(
                value["unresolved_dependencies"],
                label="run unresolved dependencies",
            )
        )
        return cls(**material)


class InstanceLedgerV15D:
    """One-use ledger spanning every physical block and both CFG branches."""

    def __init__(
        self,
        *,
        target_abi: NoAnchorTargetABIV15D,
        target_motion: TargetNativeMotionPlanV15D,
        source_initial_spatial_indices: Mapping[str, Iterable[int]],
        property_memory: SourcePropertyMemoryBoundaryV15D,
        background_carrier: SourceStaticBackgroundCarrierV15D,
    ) -> None:
        if not isinstance(target_abi, NoAnchorTargetABIV15D):
            raise V15DContractError("ledger requires the typed no-anchor target ABI")
        if not isinstance(target_motion, TargetNativeMotionPlanV15D):
            raise V15DContractError("ledger requires typed target-native motion")
        if not isinstance(property_memory, SourcePropertyMemoryBoundaryV15D):
            raise V15DContractError("ledger requires typed source property memory")
        if not isinstance(background_carrier, SourceStaticBackgroundCarrierV15D):
            raise V15DContractError("ledger requires typed source background carrier")
        # Freshly replay every typed boundary.  ``frozen=True`` prevents normal
        # mutation but is not treated as an authentication primitive.
        target_abi = NoAnchorTargetABIV15D.from_mapping(target_abi.payload())
        target_motion = TargetNativeMotionPlanV15D.from_mapping(
            target_motion.payload()
        )
        property_memory = SourcePropertyMemoryBoundaryV15D.from_mapping(
            property_memory.payload()
        )
        background_carrier = SourceStaticBackgroundCarrierV15D.from_mapping(
            background_carrier.payload()
        )
        if set(source_initial_spatial_indices) != set(FOREGROUND_ROLES):
            raise V15DContractError("source initial role set differs")
        initial = {
            role: _normalize_spatial_indices(
                source_initial_spatial_indices[role], label=f"source initial {role}"
            )
            for role in FOREGROUND_ROLES
        }
        if any(not indices for indices in initial.values()):
            raise V15DContractError("every source initial role must be nonempty")
        occupied_initial: set[int] = set()
        for role in FOREGROUND_ROLES:
            if occupied_initial.intersection(initial[role]):
                raise V15DContractError("source initial foreground roles overlap")
            occupied_initial.update(initial[role])

        target_roles = dict(target_motion.role_indices)
        target_new_phase0 = frozenset(
            index for index in target_roles[NEW_ACTOR_ROLE] if index < SPATIAL_CELLS
        )
        if target_new_phase0 != initial[NEW_ACTOR_ROLE]:
            raise V15DContractError("target #2 phase 0 must bind source #2 initial site")

        foreground_owner: dict[int, str] = {}
        for role, indices in target_motion.role_indices:
            for index in indices:
                if index in foreground_owner:
                    raise V15DContractError("target token received two foreground owners")
                foreground_owner[index] = role
        old_new_actor_sites = _temporalize_spatial(initial[NEW_ACTOR_ROLE])
        target_new_actor = target_roles[NEW_ACTOR_ROLE]
        released_candidates = frozenset(
            index for index in old_new_actor_sites
            if index >= SPATIAL_CELLS and index not in target_new_actor
        )
        if released_candidates.intersection(foreground_owner):
            raise V15DContractError(
                "a vacated #2 initial site must be HOLE, not another foreground owner"
            )
        released_holes = released_candidates
        if not released_holes:
            raise V15DContractError("E00 #2 trajectory must release its initial site as HOLE")
        background = frozenset(
            index for index in range(TOKEN_CELLS)
            if index not in foreground_owner and index not in released_holes
        )
        owners = dict(foreground_owner)
        owners.update((index, HOLE_ROLE) for index in released_holes)
        owners.update((index, BACKGROUND_ROLE) for index in background)
        if len(owners) != TOKEN_CELLS or set(owners) != set(range(TOKEN_CELLS)):
            raise V15DContractError("ledger does not own every native-grid token exactly once")

        if (
            background_carrier.source_instance_sha256
            != property_memory.source_instance_sha256
            or background_carrier.source_runtime_receipt_sha256
            != property_memory.source_runtime_receipt_sha256
            or background_carrier.source_static_reforward_receipt_sha256
            != property_memory.source_static_reforward_receipt_sha256
        ):
            raise V15DContractError(
                "background/property carriers must bind the same source static re-forward runtime"
            )
        allowed_background = background_carrier.allowed_indices
        forbidden_background = (
            frozenset(foreground_owner)
            | released_holes
            | target_motion.motion_corridor_indices
            | target_motion.contact_indices
        )
        if not allowed_background.issubset(background):
            raise V15DContractError("background carrier escaped strict background ownership")
        if allowed_background.intersection(forbidden_background):
            raise V15DContractError(
                "background carrier touches object/corridor/contact/HOLE"
            )

        ownership_counts = tuple(
            (role, sum(1 for owner in owners.values() if owner == role))
            for role in LEDGER_ROLES
        )
        source_initial_payload = [
            [role, sorted(initial[role])] for role in FOREGROUND_ROLES
        ]
        owner_assignment_payload = [
            [index, owners[index]] for index in range(TOKEN_CELLS)
        ]
        ledger_payload = {
            "schema_version": LEDGER_SCHEMA,
            "target_abi_digest": target_abi.digest,
            "target_motion_digest": target_motion.digest,
            "property_memory_digest": property_memory.digest,
            "background_carrier_digest": background_carrier.digest,
            "source_initial_sha256": object_sha256(source_initial_payload),
            "owner_assignment_sha256": object_sha256(owner_assignment_payload),
            "grid": list(GRID),
            "roles": list(LEDGER_ROLES),
            "ownership_counts": [list(item) for item in ownership_counts],
            "released_hole_indices": sorted(released_holes),
            "object_feature_route": OBJECT_FEATURE_ROUTE,
            "background_feature_route": BACKGROUND_FEATURE_ROUTE,
            "target_motion_route": TARGET_MOTION_ROUTE,
            "authorizations": hard_false_authorizations_v15d(),
        }
        self._target_abi = target_abi
        self._target_abi_digest = target_abi.digest
        self._target_arm_id = target_abi.arm_id
        self._target_motion = target_motion
        self._property_memory = property_memory
        self._background_carrier = background_carrier
        self._initial = tuple((role, initial[role]) for role in FOREGROUND_ROLES)
        self._owners = tuple(sorted(owners.items()))
        self._released_holes = released_holes
        self._background = background
        self._ownership_counts = ownership_counts
        self._digest = object_sha256(ledger_payload)
        self._target_abi_creation_bytes = canonical_json_bytes(target_abi.payload())
        self._target_motion_creation_bytes = canonical_json_bytes(target_motion.payload())
        self._property_memory_creation_bytes = canonical_json_bytes(
            property_memory.payload()
        )
        self._background_carrier_creation_bytes = canonical_json_bytes(
            background_carrier.payload()
        )
        self._source_initial_creation_bytes = canonical_json_bytes(
            source_initial_payload
        )
        self._ledger_creation_payload_bytes = canonical_json_bytes(ledger_payload)
        self._cells: dict[tuple[int, str, int], CellAuditV15D] = {}
        self._sealed = False

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def roles(self) -> tuple[str, ...]:
        return LEDGER_ROLES

    @property
    def released_hole_indices(self) -> frozenset[int]:
        return self._released_holes

    @property
    def background_indices(self) -> frozenset[int]:
        return self._background

    def owner_at(self, token_index: int) -> str:
        logical = _exact_int(
            token_index, label="ledger token index", minimum=0,
            maximum=TOKEN_CELLS - 1,
        )
        return dict(self._owners)[logical]

    def _expected_coordinate_nonce(
        self, *, step_index: int, cfg_branch: str,
        physical_block: PhysicalBlockAddressV15D,
    ) -> str:
        return object_sha256({
            "schema_version": "bernini-synthetic-cell-coordinate-nonce-v15d-r4",
            "ledger_digest": self._digest,
            "target_abi_digest": self._target_abi_digest,
            "target_motion_digest": self._target_motion.digest,
            "property_memory_digest": self._property_memory.digest,
            "background_carrier_digest": self._background_carrier.digest,
            "step_index": step_index,
            "cfg_branch": cfg_branch,
            "physical_block": physical_block.payload(),
        })

    def plan_cell(
        self,
        *,
        step_index: int,
        cfg_branch: str,
        physical_block: PhysicalBlockAddressV15D,
        requested_write_scales: Mapping[str, Any],
        outside_allowed_mask_delta_max_abs: float = 0.0,
        hole_source_new_actor_2_restore_count: int = 0,
        background_forbidden_scope_write_count: int = 0,
        source_object_direct_or_graph_appearance_write_count: int = 0,
        object_feature_route: str = OBJECT_FEATURE_ROUTE,
        background_feature_route: str = BACKGROUND_FEATURE_ROUTE,
        target_motion_route: str = TARGET_MOTION_ROUTE,
    ) -> CellAuditV15D:
        if self._sealed:
            raise V15DContractError("sealed instance ledger cannot accept another cell")
        step = _exact_int(
            step_index, label="denoise step", minimum=0, maximum=DENOISE_STEPS - 1
        )
        if cfg_branch not in CFG_BRANCHES:
            raise V15DContractError("CFG branch differs")
        if not isinstance(physical_block, PhysicalBlockAddressV15D):
            raise V15DContractError(
                "cell block must be a typed physical address, never an ordinal"
            )
        if physical_block.physical_id not in PHYSICAL_BLOCK_IDS:
            raise V15DContractError("cell physical block is outside inventory")
        write_scales = _validate_zero_write_scales(requested_write_scales)
        outside_delta = _finite_zero(
            outside_allowed_mask_delta_max_abs,
            label="outside allowed mask delta",
        )
        for label, value in (
            ("HOLE source #2 restore count", hole_source_new_actor_2_restore_count),
            ("background forbidden-scope write count", background_forbidden_scope_write_count),
            (
                "source object direct/graph appearance write count",
                source_object_direct_or_graph_appearance_write_count,
            ),
        ):
            if value != 0 or isinstance(value, bool) or not isinstance(value, int):
                raise V15DContractError(f"{label} must remain integer zero")
        if object_feature_route != OBJECT_FEATURE_ROUTE:
            raise V15DContractError("source object slots must use property memory only")
        if background_feature_route != BACKGROUND_FEATURE_ROUTE:
            raise V15DContractError("background feature route differs")
        if target_motion_route != TARGET_MOTION_ROUTE:
            raise V15DContractError("target motion cannot use source-future tracks")
        key = (step, cfg_branch, physical_block.physical_id)
        if key in self._cells:
            raise V15DContractError("duplicate physical block/step/CFG cell")
        route_allowlisted = physical_block.physical_id in ROUTE_PHYSICAL_BLOCK_IDS
        ordered_scales = tuple((name, write_scales[name]) for name in WRITE_COMPONENTS)
        partial = CellAuditV15D(
            schema_version=CELL_AUDIT_SCHEMA,
            ledger_digest=self._digest,
            target_abi_digest=self._target_abi_digest,
            target_motion_digest=self._target_motion.digest,
            property_memory_digest=self._property_memory.digest,
            background_carrier_digest=self._background_carrier.digest,
            step_index=step,
            cfg_branch=cfg_branch,
            physical_block=physical_block,
            route_allowlisted=route_allowlisted,
            arm_id=self._target_arm_id,
            outside_allowed_mask_delta_max_abs=outside_delta,
            hole_source_new_actor_2_restore_count=0,
            background_forbidden_scope_write_count=0,
            source_object_direct_or_graph_appearance_write_count=0,
            object_feature_route=OBJECT_FEATURE_ROUTE,
            background_feature_route=BACKGROUND_FEATURE_ROUTE,
            target_motion_route=TARGET_MOTION_ROUTE,
            write_scales=ordered_scales,
            coordinate_nonce=self._expected_coordinate_nonce(
                step_index=step, cfg_branch=cfg_branch,
                physical_block=physical_block,
            ),
            runtime_execution_observed=False,
            synthetic_plan=True,
            digest="",
        )
        cell = CellAuditV15D(
            **{
                **partial.__dict__,
                "digest": object_sha256(partial.payload_without_digest()),
            }
        )
        self._cells[key] = cell
        return cell

    def _revalidate_cell(
        self, key: tuple[int, str, int], cell: CellAuditV15D
    ) -> None:
        if not isinstance(cell, CellAuditV15D) or cell.schema_version != CELL_AUDIT_SCHEMA:
            raise V15DContractError("cell audit type/schema differs")
        if cell.ledger_digest != self._digest:
            raise V15DContractError("more than one ledger authority appears in cells")
        for label, current, expected in (
            ("target ABI", cell.target_abi_digest, self._target_abi_digest),
            ("target motion", cell.target_motion_digest, self._target_motion.digest),
            ("property memory", cell.property_memory_digest, self._property_memory.digest),
            (
                "background carrier", cell.background_carrier_digest,
                self._background_carrier.digest,
            ),
        ):
            if current != expected:
                raise V15DContractError(f"synthetic cell {label} binding differs")
        cell.physical_block.__post_init__()
        if key != (
            cell.step_index, cell.cfg_branch, cell.physical_block.physical_id
        ):
            raise V15DContractError("cell audit key/content differs")
        _exact_int(
            cell.step_index, label="cell denoise step", minimum=0,
            maximum=DENOISE_STEPS - 1,
        )
        if cell.cfg_branch not in CFG_BRANCHES:
            raise V15DContractError("cell CFG branch differs")
        expected_allowlisted = (
            cell.physical_block.physical_id in ROUTE_PHYSICAL_BLOCK_IDS
        )
        if cell.route_allowlisted is not expected_allowlisted:
            raise V15DContractError("cell route allowlist flag differs from physical ID")
        if cell.arm_id != self._target_arm_id:
            raise V15DContractError("cell arm differs from the single target ABI")
        _finite_zero(
            cell.outside_allowed_mask_delta_max_abs,
            label="cell outside-mask delta",
        )
        for label, value in (
            ("cell HOLE source #2 restore count", cell.hole_source_new_actor_2_restore_count),
            ("cell background forbidden-scope count", cell.background_forbidden_scope_write_count),
            (
                "cell direct/graph object appearance count",
                cell.source_object_direct_or_graph_appearance_write_count,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                raise V15DContractError(f"{label} must remain integer zero")
        if (
            cell.object_feature_route != OBJECT_FEATURE_ROUTE
            or cell.background_feature_route != BACKGROUND_FEATURE_ROUTE
            or cell.target_motion_route != TARGET_MOTION_ROUTE
        ):
            raise V15DContractError("cell feature/motion route differs")
        if tuple(name for name, _ in cell.write_scales) != WRITE_COMPONENTS:
            raise V15DContractError("cell write-scale order/components differ")
        _validate_zero_write_scales(dict(cell.write_scales))
        expected_nonce = self._expected_coordinate_nonce(
            step_index=cell.step_index,
            cfg_branch=cell.cfg_branch,
            physical_block=cell.physical_block,
        )
        if cell.coordinate_nonce != expected_nonce:
            raise V15DContractError("synthetic cell coordinate nonce differs")
        if cell.runtime_execution_observed is not False or cell.synthetic_plan is not True:
            raise V15DContractError("cell must remain an unexecuted synthetic plan")
        if cell.digest != object_sha256(cell.payload_without_digest()):
            raise V15DContractError("cell audit digest differs")

    def seal(self) -> ZeroUpdateRunReceiptV15D:
        if self._sealed:
            raise V15DContractError("instance ledger was already sealed")
        # Serialize current objects, compare them byte-for-byte with creation
        # snapshots, then rebuild an entirely fresh ledger.  Re-signing a
        # mutated frozen dataclass is therefore insufficient: its current
        # material must still equal the independently retained creation bytes,
        # and all ownership/cross-bind state is recomputed below.
        current_material = (
            ("target ABI", canonical_json_bytes(self._target_abi.payload()),
             self._target_abi_creation_bytes),
            ("target motion", canonical_json_bytes(self._target_motion.payload()),
             self._target_motion_creation_bytes),
            ("property memory", canonical_json_bytes(self._property_memory.payload()),
             self._property_memory_creation_bytes),
            ("background carrier", canonical_json_bytes(self._background_carrier.payload()),
             self._background_carrier_creation_bytes),
            (
                "source initial roles",
                canonical_json_bytes(
                    [[role, sorted(indices)] for role, indices in self._initial]
                ),
                self._source_initial_creation_bytes,
            ),
        )
        for label, current_bytes, creation_bytes in current_material:
            if current_bytes != creation_bytes:
                raise V15DContractError(f"{label} changed after ledger construction")

        current_abi = NoAnchorTargetABIV15D.from_mapping(
            canonical_json_loads_v15d(self._target_abi_creation_bytes)
        )
        if (
            current_abi.digest != self._target_abi_digest
            or current_abi.arm_id != self._target_arm_id
        ):
            raise V15DContractError(
                "cached target ABI digest/arm differs from fresh creation ABI"
            )
        current_motion = TargetNativeMotionPlanV15D.from_mapping(
            canonical_json_loads_v15d(self._target_motion_creation_bytes)
        )
        current_property = SourcePropertyMemoryBoundaryV15D.from_mapping(
            canonical_json_loads_v15d(self._property_memory_creation_bytes)
        )
        current_background = SourceStaticBackgroundCarrierV15D.from_mapping(
            canonical_json_loads_v15d(self._background_carrier_creation_bytes)
        )
        initial_rows = _json_list(
            canonical_json_loads_v15d(self._source_initial_creation_bytes),
            label="source initial serialized roles",
        )
        if len(initial_rows) != len(FOREGROUND_ROLES):
            raise V15DContractError("source initial serialized role count differs")
        fresh_initial: dict[str, list[int]] = {}
        for index, row in enumerate(initial_rows):
            pair = _json_list(row, label=f"source initial role row {index}")
            if len(pair) != 2 or pair[0] != FOREGROUND_ROLES[index]:
                raise V15DContractError("source initial serialized role order differs")
            fresh_initial[pair[0]] = list(
                _strict_ascending_unique_int_json_list(
                    pair[1], label=f"source initial {pair[0]} indices",
                    minimum=0, maximum=SPATIAL_CELLS - 1,
                )
            )
        recomputed = InstanceLedgerV15D(
            target_abi=current_abi,
            target_motion=current_motion,
            source_initial_spatial_indices=fresh_initial,
            property_memory=current_property,
            background_carrier=current_background,
        )
        if (
            recomputed._ledger_creation_payload_bytes
            != self._ledger_creation_payload_bytes
            or recomputed.digest != self._digest
            or recomputed._owners != self._owners
            or recomputed._released_holes != self._released_holes
            or recomputed._background != self._background
            or recomputed._ownership_counts != self._ownership_counts
        ):
            raise V15DContractError(
                "freshly recomputed ownership/cross-bind ledger differs from creation"
            )
        expected_keys = {
            (step, branch, physical_id)
            for step in range(DENOISE_STEPS)
            for branch in CFG_BRANCHES
            for physical_id in PHYSICAL_BLOCK_IDS
        }
        if set(self._cells) != expected_keys:
            missing = len(expected_keys - set(self._cells))
            extra = len(set(self._cells) - expected_keys)
            raise V15DContractError(
                f"ledger cells incomplete; missing={missing}, extra={extra}"
            )
        for key, cell in self._cells.items():
            self._revalidate_cell(key, cell)
        ordered_cells = tuple(self._cells[key] for key in sorted(self._cells))
        chain = hashlib.sha256()
        for cell in ordered_cells:
            chain.update(bytes.fromhex(cell.digest))
        expected_count = DENOISE_STEPS * len(CFG_BRANCHES) * len(PHYSICAL_BLOCK_IDS)
        selected_count = (
            DENOISE_STEPS * len(CFG_BRANCHES) * len(ROUTE_PHYSICAL_BLOCK_IDS)
        )
        receipt_payload = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "decision": DECISION,
            "target_abi_digest": self._target_abi_digest,
            "target_motion_digest": self._target_motion.digest,
            "property_memory_digest": self._property_memory.digest,
            "background_carrier_digest": self._background_carrier.digest,
            "source_instance_sha256": self._property_memory.source_instance_sha256,
            "source_runtime_receipt_sha256": (
                self._property_memory.source_runtime_receipt_sha256
            ),
            "source_static_reforward_receipt_sha256": (
                self._property_memory.source_static_reforward_receipt_sha256
            ),
            "ledger_digest": self._digest,
            "ledger_creation_snapshot_sha256": hashlib.sha256(
                self._ledger_creation_payload_bytes
            ).hexdigest(),
            "arm_id": self._target_arm_id,
            "cell_count": len(ordered_cells),
            "expected_cell_count": expected_count,
            "selected_route_cell_count": selected_count,
            "all_physical_blocks_seen_per_step_cfg": True,
            "one_ledger_digest_across_all_cells": len(
                {cell.ledger_digest for cell in ordered_cells}
            ) == 1,
            "all_planned_cell_write_scales_exact_zero": True,
            "all_planned_outside_allowed_mask_deltas_exact_zero": True,
            "hole_never_restored_as_source_new_actor_2": True,
            "source_objects_use_property_memory_only": True,
            "background_scope_strict": True,
            "source_future_tracks_used_for_target_motion": False,
            "runtime_execution_observed": False,
            "cell_audits_are_synthetic_plan": True,
            "fresh_bundle_validation_required": True,
            "standalone_receipt_authority": False,
            "external_source_authority_passed": False,
            "route_execution_authorized": False,
            "training_authorized": False,
            "scientific_claim_authorized": False,
            "cell_audit_chain_sha256": chain.hexdigest(),
            "unresolved_dependencies": list(UNRESOLVED_DEPENDENCIES),
        }
        constructor_payload = dict(receipt_payload)
        constructor_payload["unresolved_dependencies"] = UNRESOLVED_DEPENDENCIES
        constructor_payload["digest"] = object_sha256(receipt_payload)
        receipt = ZeroUpdateRunReceiptV15D(**constructor_payload)
        self._sealed = True
        return receipt


def _fresh_source_initial_from_json(value: Any) -> dict[str, list[int]]:
    rows = _json_list(value, label="bundle source initial roles")
    if len(rows) != len(FOREGROUND_ROLES):
        raise V15DContractError("bundle source initial role count differs")
    result: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        pair = _json_list(row, label=f"bundle source initial row {index}")
        if len(pair) != 2 or pair[0] != FOREGROUND_ROLES[index]:
            raise V15DContractError("bundle source initial role order differs")
        result[pair[0]] = list(
            _strict_ascending_unique_int_json_list(
                pair[1], label=f"bundle source initial {pair[0]}",
                minimum=0, maximum=SPATIAL_CELLS - 1,
            )
        )
    return result


def serialize_synthetic_plan_bundle_v15d(
    ledger: InstanceLedgerV15D,
    receipt: ZeroUpdateRunReceiptV15D,
) -> bytes:
    """Serialize a sealed, explicitly unexecuted plan and fresh-replay it."""

    if not isinstance(ledger, InstanceLedgerV15D) or ledger._sealed is not True:
        raise V15DContractError("bundle serialization requires a sealed synthetic ledger")
    if not isinstance(receipt, ZeroUpdateRunReceiptV15D):
        raise V15DContractError("bundle serialization requires a typed run receipt")
    receipt = ZeroUpdateRunReceiptV15D.from_mapping(receipt.payload())
    if receipt.ledger_digest != ledger.digest:
        raise V15DContractError("bundle receipt/ledger identity differs")
    ordered_cells = [
        ledger._cells[key].payload() for key in sorted(ledger._cells)
    ]
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "decision": DECISION,
        "contract_sha256": object_sha256(expected_contract_v15d()),
        "runtime_execution_observed": False,
        "cell_audits_are_synthetic_plan": True,
        "target_abi": canonical_json_loads_v15d(
            ledger._target_abi_creation_bytes
        ),
        "target_motion": canonical_json_loads_v15d(
            ledger._target_motion_creation_bytes
        ),
        "property_memory": canonical_json_loads_v15d(
            ledger._property_memory_creation_bytes
        ),
        "background_carrier": canonical_json_loads_v15d(
            ledger._background_carrier_creation_bytes
        ),
        "source_initial_spatial_indices": canonical_json_loads_v15d(
            ledger._source_initial_creation_bytes
        ),
        "ledger_creation_payload": canonical_json_loads_v15d(
            ledger._ledger_creation_payload_bytes
        ),
        "synthetic_cells": ordered_cells,
        "receipt": receipt.payload(),
    }
    serialized = canonical_json_bytes(bundle)
    replayed = fresh_deserialize_validate_synthetic_plan_v15d(serialized)
    if replayed.payload() != receipt.payload():
        raise V15DContractError("fresh serialized replay receipt differs")
    return serialized


def fresh_deserialize_validate_synthetic_plan_v15d(
    serialized: bytes,
) -> ZeroUpdateRunReceiptV15D:
    """Fresh canonical-byte consumer for the unexecuted r4 synthetic plan."""

    value = canonical_json_loads_v15d(serialized)
    if not isinstance(value, Mapping):
        raise V15DContractError("synthetic plan bundle root must be a mapping")
    fields = {
        "schema_version", "decision", "contract_sha256",
        "runtime_execution_observed", "cell_audits_are_synthetic_plan",
        "target_abi", "target_motion", "property_memory",
        "background_carrier", "source_initial_spatial_indices",
        "ledger_creation_payload", "synthetic_cells", "receipt",
    }
    _exact_keys(value, fields, label="synthetic plan bundle")
    if value["schema_version"] != BUNDLE_SCHEMA or value["decision"] != DECISION:
        raise V15DContractError("synthetic plan bundle schema/decision differs")
    if value["contract_sha256"] != object_sha256(expected_contract_v15d()):
        raise V15DContractError("synthetic plan bundle contract identity differs")
    if (
        value["runtime_execution_observed"] is not False
        or value["cell_audits_are_synthetic_plan"] is not True
    ):
        raise V15DContractError("bundle must remain an unexecuted synthetic plan")
    target_abi = NoAnchorTargetABIV15D.from_mapping(value["target_abi"])
    target_motion = TargetNativeMotionPlanV15D.from_mapping(value["target_motion"])
    property_memory = SourcePropertyMemoryBoundaryV15D.from_mapping(
        value["property_memory"]
    )
    background_carrier = SourceStaticBackgroundCarrierV15D.from_mapping(
        value["background_carrier"]
    )
    initial = _fresh_source_initial_from_json(
        value["source_initial_spatial_indices"]
    )
    ledger = InstanceLedgerV15D(
        target_abi=target_abi,
        target_motion=target_motion,
        source_initial_spatial_indices=initial,
        property_memory=property_memory,
        background_carrier=background_carrier,
    )
    ledger_payload = value["ledger_creation_payload"]
    if not isinstance(ledger_payload, Mapping):
        raise V15DContractError("bundle ledger creation payload must be a mapping")
    _exact_keys(
        ledger_payload,
        {
            "schema_version", "target_abi_digest", "target_motion_digest",
            "property_memory_digest", "background_carrier_digest",
            "source_initial_sha256", "owner_assignment_sha256", "grid",
            "roles", "ownership_counts", "released_hole_indices",
            "object_feature_route", "background_feature_route",
            "target_motion_route", "authorizations",
        },
        label="bundle ledger creation payload",
    )
    _strict_ascending_unique_int_json_list(
        ledger_payload["released_hole_indices"],
        label="bundle released HOLE indices",
        minimum=0,
        maximum=TOKEN_CELLS - 1,
    )
    if canonical_json_bytes(ledger_payload) != ledger._ledger_creation_payload_bytes:
        raise V15DContractError("bundle freshly recomputed ledger payload differs")
    cells = _json_list(value["synthetic_cells"], label="bundle synthetic cells")
    expected_count = DENOISE_STEPS * len(CFG_BRANCHES) * len(PHYSICAL_BLOCK_IDS)
    if len(cells) != expected_count:
        raise V15DContractError("bundle synthetic cell count differs")
    expected_coordinates = tuple(sorted(
        (step, branch, physical_id)
        for step in range(DENOISE_STEPS)
        for branch in CFG_BRANCHES
        for physical_id in PHYSICAL_BLOCK_IDS
    ))
    for index, (cell_value, expected_key) in enumerate(
        zip(cells, expected_coordinates)
    ):
        if not isinstance(cell_value, Mapping):
            raise V15DContractError(f"bundle synthetic cell {index} must be a mapping")
        cell = CellAuditV15D.from_mapping(cell_value)
        key = (cell.step_index, cell.cfg_branch, cell.physical_block.physical_id)
        if key != expected_key:
            raise V15DContractError(
                "bundle synthetic cells are not in unique canonical coordinate order"
            )
        if key in ledger._cells:
            raise V15DContractError("bundle contains duplicate synthetic coordinate")
        ledger._cells[key] = cell
    replayed = ledger.seal()
    submitted = ZeroUpdateRunReceiptV15D.from_mapping(value["receipt"])
    if submitted.payload() != replayed.payload():
        raise V15DContractError("bundle submitted receipt differs from fresh replay")
    return replayed


def preregistration_receipt_v15d() -> dict[str, Any]:
    """Small status receipt; no model/runtime action is performed."""

    contract = expected_contract_v15d()
    return {
        "schema_version": "bernini-v15d-unexecuted-synthetic-plan-status-r4",
        "decision": DECISION,
        "contract_sha256": object_sha256(contract),
        "grid": list(GRID),
        "physical_block_count": len(PHYSICAL_BLOCK_IDS),
        "route_physical_block_ids": list(ROUTE_PHYSICAL_BLOCK_IDS),
        "route_physical_block_count": len(ROUTE_PHYSICAL_BLOCK_IDS),
        "cfg_branches": list(CFG_BRANCHES),
        "arms": list(ARM_IDS),
        "default_all_planned_write_scales_exact_zero": True,
        "runtime_execution_observed": False,
        "cell_audits_are_synthetic_plan": True,
        "fresh_bundle_validation_required": True,
        "standalone_receipt_authority": False,
        "source_visible_amber_classification": (
            "source property; excluded from target action/graph payload"
        ),
        "unexecuted_local_cpu_synthetic_plan_only": True,
        "external_source_authority_passed": False,
        "gpu_execution_performed": False,
        "route_execution_performed": False,
        "decode_performed": False,
        "training_performed": False,
        "route_execution_authorized": False,
        "training_authorized": False,
        "scientific_claim_authorized": False,
        "unresolved_dependencies": list(UNRESOLVED_DEPENDENCIES),
    }


__all__ = [
    "ARM_A", "ARM_B", "ARM_C", "ARM_D", "ARM_IDS",
    "BACKGROUND_ROLE", "BLOCK_NAMESPACE", "CANONICAL_TARGET_ACTION_TEXT",
    "CFG_BRANCHES", "DECISION", "DENOISE_STEPS", "FOREGROUND_ROLES", "GRID",
    "AUTHORIZATION_FIELDS", "HOLE_ROLE", "HUMAN_ROLE", "LEDGER_ROLES",
    "NEW_ACTOR_ROLE", "OLD_ACTOR_ROLE", "PHYSICAL_BLOCK_IDS",
    "RECIPIENT_ROLE", "ROUTE_PHYSICAL_BLOCK_IDS", "SOURCE_AUTHORITY_STATE",
    "TOKEN_CELLS", "UNRESOLVED_DEPENDENCIES", "V15DContractError",
    "WRITE_COMPONENTS", "ZERO_WRITE_SCALES", "CellAuditV15D",
    "InstanceLedgerV15D", "NoAnchorTargetABIV15D", "PhysicalBlockAddressV15D",
    "SourcePropertyMemoryBoundaryV15D", "SourceStaticBackgroundCarrierV15D",
    "TargetNativeMotionPlanV15D", "ZeroUpdateRunReceiptV15D",
    "build_no_anchor_target_abi_v15d", "expected_contract_v15d",
    "canonical_json_loads_v15d", "fresh_deserialize_validate_synthetic_plan_v15d",
    "hard_false_authorizations_v15d",
    "load_contract_v15d", "object_sha256", "preregistration_receipt_v15d",
    "serialize_synthetic_plan_bundle_v15d", "validate_contract_v15d",
]
