#!/usr/bin/env python3
"""CAPER owner-to-editor segment preference cotangents.

This module is the narrow bridge between the externally audited pure-T2V
owners and the native Bernini RV2V graph.  The owner media never crosses this
bridge.  The only admitted owner value is the detached, normalized,
prompt-relative block-15 quotient produced by
``materialize_self_imagined_owner_core2_v1`` and authenticated by
``self_imagined_native_rv2v_hidden_vjp_v1``.

The historical global-cosine objective can hide one failed temporal event by
averaging it with an easier event.  CAPER therefore exposes four independent
rows per query seed: onset, ordered transition, terminal completion, and
terminal hold.  Each row has its own scalar and cotangent.  Rows are never
averaged, ranked, or selected; downstream QP code must keep all rows as hard
constraints.  A gate failure returns a sealed zero-update decision *before*
any differentiable leaf is constructed.

This file has no optimizer, model forward, parameter mutation, sampler, mask,
track, pose, optical flow, owner RGB/latent/noise/velocity, or target video
input.  Despite the compatibility name ``segment_dpo``, its production
objective is an action-specific hidden quotient, not raw target/rejected MSE.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import torch

import self_imagined_motion_cotangent_v1 as motion


SCHEMA_VERSION = "bernini-caper-pure-t2v-segment-dpo-v1"
AUTHORITY_SCHEMA_VERSION = "bernini-caper-owner-segment-authority-v1"
DECISION_SCHEMA_VERSION = "bernini-caper-owner-segment-zero-update-decision-v1"
COTANGENT_ROW_SCHEMA_VERSION = "bernini-caper-owner-segment-cotangent-row-v1"
COTANGENT_BUNDLE_SCHEMA_VERSION = "bernini-caper-owner-segment-cotangent-bundle-v1"
CANDIDATE_EVALUATION_SCHEMA_VERSION = (
    "bernini-caper-candidate-per-sample-segment-evaluation-v1"
)

CELL_IDS = ("dog", "human")
SEGMENT_ORDER = ("onset", "transition", "completion", "hold")
QUERY_SEEDS_BY_CELL = MappingProxyType(
    {
        "dog": (2026081502, 2026081503),
        "human": (2026081505, 2026081506),
    }
)
ACTION_FAMILY_BY_CELL = MappingProxyType(
    {
        "dog": "dog-stand-to-sit-facing-camera",
        "human": "human-one-knee-to-upright-stand",
    }
)

# ``temporal_motion_quotient`` concatenates 93 rows of width 2*1536:
# level21, lag1-21, lag2-21, lag4-21, boundary1, initial-hold4,
# terminal-hold4.  These selectors are frozen before any editor score is
# observed.  Overlap is deliberate: a completion must retain the late state,
# while the hold row independently prevents a transient terminal pose.
FEATURE_WIDTH = 2 * motion.HIDDEN_SIZE
FEATURE_ROW_COUNT = 4 * motion.LATENT_PHASES + 1 + 2 * motion.HOLD_PHASES
_LEVEL = 0
_LAG1 = 21
_LAG2 = 42
_LAG4 = 63
_BOUNDARY = 84
_INITIAL_HOLD = 85
_TERMINAL_HOLD = 89
SEGMENT_ROW_RANGES = MappingProxyType(
    {
        "onset": ((_INITIAL_HOLD, _INITIAL_HOLD + 4),),
        "transition": (
            (_LEVEL + 4, _LEVEL + 17),
            (_LAG1 + 4, _LAG1 + 17),
            (_LAG2 + 4, _LAG2 + 17),
            (_LAG4 + 4, _LAG4 + 17),
        ),
        "completion": (
            (_LEVEL + 17, _LEVEL + 21),
            (_BOUNDARY, _BOUNDARY + 1),
        ),
        "hold": ((_TERMINAL_HOLD, _TERMINAL_HOLD + 4),),
    }
)

FORBIDDEN_CHANNELS = (
    "owner_rgb",
    "owner_clean_latent",
    "owner_gaussian",
    "owner_velocity",
    "owner_text_condition",
    "owner_reference",
    "owner_target",
    "owner_donor",
    "generation_gaussian_as_training_epsilon",
    "mask",
    "track",
    "pose",
    "flow",
    "trajectory",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class CAPERPureT2VSegmentDPOError(RuntimeError):
    """An owner authority, temporal selector, tensor, or gate is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise CAPERPureT2VSegmentDPOError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CAPERPureT2VSegmentDPOError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CAPERPureT2VSegmentDPOError(f"{label} is not a safe identifier")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CAPERPureT2VSegmentDPOError(f"{label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise CAPERPureT2VSegmentDPOError(f"{label} must be finite")
    return result


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise CAPERPureT2VSegmentDPOError("receipt is already sealed")
    value = dict(unsigned)
    return {**value, "receipt_digest": object_sha256(value)}


def _tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not value.is_floating_point()
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise CAPERPureT2VSegmentDPOError(f"{label} is not a finite tensor")
    tensor = value.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def segment_selector_receipt() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "feature_owner": "self_imagined_motion_cotangent_v1.temporal_motion_quotient",
        "feature_rows": FEATURE_ROW_COUNT,
        "feature_width": FEATURE_WIDTH,
        "segment_order": list(SEGMENT_ORDER),
        "row_ranges": {
            name: [list(bounds) for bounds in SEGMENT_ROW_RANGES[name]]
            for name in SEGMENT_ORDER
        },
        "constraint_mode": "each_seed_x_segment_is_an_independent_hard_action_row",
        "mean_over_batch_before_worst_cell": False,
        "row_averaging": False,
        "seed_averaging": False,
        "seed_ranking_or_selection": False,
        "completion_metric": "action_specific_hidden_terminal_boundary",
        "hold_metric": "action_specific_hidden_terminal_hold_residual",
        "raw_target_rejected_latent_mse_is_semantic_evaluator": False,
    }
    return {**value, "digest": object_sha256(value)}


SELECTOR_DIGEST = str(segment_selector_receipt()["digest"])


@dataclass(frozen=True)
class ValidatedSegmentAuthority:
    cell_id: str
    query_seed: int
    action_family_id: str
    owner_packet_receipt_digest: str
    generation_root_receipt_digest: str
    generation_root_file_sha256: str
    generation_arm_receipt_digest: str
    generation_arm_file_sha256: str
    owner_quotient_root_receipt_digest: str
    owner_quotient_root_file_sha256: str
    owner_quotient_arm_receipt_digest: str
    owner_quotient_arm_file_sha256: str
    role_prompt_bank_sha256: str
    semantic_audit_sidecar_receipt_digest: str
    selector_digest: str
    minimum_score_by_segment: Mapping[str, float]
    receipt_digest: str


_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "cell_id",
        "query_seed",
        "action_family_id",
        "owner_packet_receipt_digest",
        "generation_mode",
        "generation_root_receipt_digest",
        "generation_root_file_sha256",
        "generation_arm_receipt_digest",
        "generation_arm_file_sha256",
        "owner_quotient_root_receipt_digest",
        "owner_quotient_root_file_sha256",
        "owner_quotient_arm_receipt_digest",
        "owner_quotient_arm_file_sha256",
        "role_prompt_bank_sha256",
        "semantic_audit_sidecar_receipt_digest",
        "selector_digest",
        "minimum_score_by_segment",
        "segment_semantic_gates",
        "source_relative_completion_contract",
        "forbidden_channels",
        "external_registration_required",
        "receipt_digest",
    }
)


def validate_segment_authority(
    value: Any,
    *,
    registered_receipt_digest: str,
    expected_owner_packet_receipt_digest: str,
) -> ValidatedSegmentAuthority:
    """Validate one externally registered, non-self-authorizing authority."""

    registered = _sha256(
        registered_receipt_digest, label="registered segment authority digest"
    )
    owner_digest = _sha256(
        expected_owner_packet_receipt_digest, label="owner packet receipt digest"
    )
    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
        raise CAPERPureT2VSegmentDPOError("segment authority field closure differs")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        value.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or declared != registered
        or object_sha256(unsigned) != declared
        or value.get("external_registration_required") is not True
    ):
        raise CAPERPureT2VSegmentDPOError(
            "segment authority seal/external registration differs"
        )
    cell = _safe_id(value.get("cell_id"), label="cell ID")
    seed = value.get("query_seed")
    family = _safe_id(value.get("action_family_id"), label="action family")
    if (
        cell not in CELL_IDS
        or type(seed) is not int
        or seed not in QUERY_SEEDS_BY_CELL[cell]
        or family != ACTION_FAMILY_BY_CELL[cell]
        or value.get("generation_mode") != "frozen_bernini_pure_t2v_exact81_exact40"
        or value.get("owner_packet_receipt_digest") != owner_digest
        or value.get("selector_digest") != SELECTOR_DIGEST
        or value.get("forbidden_channels") != list(FORBIDDEN_CHANNELS)
    ):
        raise CAPERPureT2VSegmentDPOError("segment authority identity differs")
    hashes = {}
    for name in (
        "generation_root_receipt_digest",
        "generation_root_file_sha256",
        "generation_arm_receipt_digest",
        "generation_arm_file_sha256",
        "owner_quotient_root_receipt_digest",
        "owner_quotient_root_file_sha256",
        "owner_quotient_arm_receipt_digest",
        "owner_quotient_arm_file_sha256",
        "role_prompt_bank_sha256",
        "semantic_audit_sidecar_receipt_digest",
    ):
        hashes[name] = _sha256(value.get(name), label=name)
    gates = value.get("segment_semantic_gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != set(SEGMENT_ORDER)
        or any(
            not isinstance(gates[name], Mapping)
            or gates[name].get("passed") is not True
            or gates[name].get("action_family_id") != family
            or gates[name].get("evaluator_frozen_before_editor_measurement") is not True
            for name in SEGMENT_ORDER
        )
    ):
        raise CAPERPureT2VSegmentDPOError("owner segment semantic gate differs")
    completion = value.get("source_relative_completion_contract")
    if (
        not isinstance(completion, Mapping)
        or completion.get("completion_metric")
        != "action_specific_source_relative_terminal_residual"
        or completion.get("hold_metric")
        != "action_specific_source_relative_terminal_hold"
        or completion.get("raw_absolute_latent_drift_used") is not False
        or completion.get("raw_target_rejected_mse_used") is not False
        or completion.get("frozen_evaluator_required_for_candidate") is not True
    ):
        raise CAPERPureT2VSegmentDPOError("completion evaluator contract differs")
    thresholds = value.get("minimum_score_by_segment")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(SEGMENT_ORDER):
        raise CAPERPureT2VSegmentDPOError("segment score threshold closure differs")
    parsed_thresholds = {
        name: _finite(thresholds[name], label=f"{name} minimum score")
        for name in SEGMENT_ORDER
    }
    if any(not -1.0 <= score <= 1.0 for score in parsed_thresholds.values()):
        raise CAPERPureT2VSegmentDPOError("segment score threshold lies outside [-1,1]")
    return ValidatedSegmentAuthority(
        cell_id=cell,
        query_seed=seed,
        action_family_id=family,
        owner_packet_receipt_digest=owner_digest,
        generation_root_receipt_digest=hashes["generation_root_receipt_digest"],
        generation_root_file_sha256=hashes["generation_root_file_sha256"],
        generation_arm_receipt_digest=hashes["generation_arm_receipt_digest"],
        generation_arm_file_sha256=hashes["generation_arm_file_sha256"],
        owner_quotient_root_receipt_digest=hashes[
            "owner_quotient_root_receipt_digest"
        ],
        owner_quotient_root_file_sha256=hashes[
            "owner_quotient_root_file_sha256"
        ],
        owner_quotient_arm_receipt_digest=hashes[
            "owner_quotient_arm_receipt_digest"
        ],
        owner_quotient_arm_file_sha256=hashes[
            "owner_quotient_arm_file_sha256"
        ],
        role_prompt_bank_sha256=hashes["role_prompt_bank_sha256"],
        semantic_audit_sidecar_receipt_digest=hashes[
            "semantic_audit_sidecar_receipt_digest"
        ],
        selector_digest=SELECTOR_DIGEST,
        minimum_score_by_segment=MappingProxyType(parsed_thresholds),
        receipt_digest=registered,
    )


def _validate_feature(value: Any, *, label: str, graph: bool) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.ndim != 2
        or tuple(value.shape) != (1, FEATURE_ROW_COUNT * FEATURE_WIDTH)
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
        or (graph and (not value.requires_grad or value.grad_fn is None))
        or (not graph and (value.requires_grad or value.grad_fn is not None))
    ):
        raise CAPERPureT2VSegmentDPOError(f"{label} feature geometry differs")
    return value


def _selector_indices(segment: str, *, device: torch.device) -> torch.Tensor:
    if segment not in SEGMENT_ORDER:
        raise CAPERPureT2VSegmentDPOError("segment selector differs")
    indices = []
    for begin, stop in SEGMENT_ROW_RANGES[segment]:
        if not 0 <= begin < stop <= FEATURE_ROW_COUNT:
            raise CAPERPureT2VSegmentDPOError("segment row range escaped feature")
        indices.extend(range(begin * FEATURE_WIDTH, stop * FEATURE_WIDTH))
    if not indices or len(set(indices)) != len(indices):
        raise CAPERPureT2VSegmentDPOError("segment selector is empty or repeats columns")
    return torch.tensor(indices, dtype=torch.int64, device=device)


def _segment_cosines(
    editor_feature: torch.Tensor,
    owner_unit_feature: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    if editor_feature.shape != owner_unit_feature.shape:
        raise CAPERPureT2VSegmentDPOError("editor/owner feature shapes differ")
    result = {}
    for segment in SEGMENT_ORDER:
        index = _selector_indices(segment, device=editor_feature.device)
        editor = editor_feature.index_select(1, index)
        owner = owner_unit_feature.index_select(1, index)
        editor_norm = torch.linalg.vector_norm(editor, dim=1)
        owner_norm = torch.linalg.vector_norm(owner, dim=1)
        if (
            float(editor_norm.min().detach().item())
            < motion.MotionQuotientConfig().minimum_feature_norm
            or float(owner_norm.min().detach().item())
            < motion.MotionQuotientConfig().minimum_feature_norm
        ):
            raise CAPERPureT2VSegmentDPOError(
                f"{segment} action-specific quotient is degenerate"
            )
        result[segment] = (
            (editor * owner).sum(dim=1) / (editor_norm * owner_norm)
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class SegmentZeroUpdateDecision:
    authorized: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class SegmentCotangentRow:
    cell_id: str
    query_seed: int
    sample_index: int
    segment: str
    score: float
    action_cotangent: torch.Tensor
    noop_cotangent: torch.Tensor
    authority_receipt_digest: str
    editor_measurement_digest: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class SegmentCotangentBundle:
    cell_id: str
    query_seed: int
    rows: tuple[SegmentCotangentRow, ...]
    receipt: Mapping[str, Any]


def _zero_decision(
    *,
    authority: ValidatedSegmentAuthority,
    scores: Optional[Mapping[str, float]],
    failures: Sequence[str],
) -> SegmentZeroUpdateDecision:
    failure_codes = sorted(set(str(item) for item in failures))
    unsigned = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "cell_id": authority.cell_id,
        "query_seed": authority.query_seed,
        "segment_authority_receipt_digest": authority.receipt_digest,
        "scores": None if scores is None else dict(scores),
        "failure_codes": failure_codes,
        "zero_update": True,
        "student_leaf_graph_constructed": False,
        "optimizer_constructed": False,
        "parameter_mutation_performed": False,
    }
    return SegmentZeroUpdateDecision(False, _seal(unsigned))


def build_segment_cotangents_from_authenticated_measurements(
    *,
    authority: ValidatedSegmentAuthority,
    owner_unit_feature: torch.Tensor,
    action_measurement: torch.Tensor,
    noop_measurement: torch.Tensor,
    owner_packet_receipt_digest: str,
) -> SegmentCotangentBundle | SegmentZeroUpdateDecision:
    """Build four independent cotangents, or fail before graph construction."""

    if not isinstance(authority, ValidatedSegmentAuthority):
        raise CAPERPureT2VSegmentDPOError("validated segment authority is required")
    owner_digest = _sha256(
        owner_packet_receipt_digest, label="owner packet receipt digest"
    )
    if owner_digest != authority.owner_packet_receipt_digest:
        raise CAPERPureT2VSegmentDPOError("owner packet/segment authority differs")
    for label, value in (
        ("action measurement", action_measurement),
        ("no-op measurement", noop_measurement),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.ndim != 4
            or tuple(value.shape[:2]) != (1, motion.LATENT_PHASES)
            or int(value.shape[-1]) != motion.HIDDEN_SIZE
            or value.device.type == "meta"
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise CAPERPureT2VSegmentDPOError(f"{label} geometry differs")
    if action_measurement.shape != noop_measurement.shape:
        raise CAPERPureT2VSegmentDPOError("action/no-op measurement shapes differ")
    owner = _validate_feature(
        owner_unit_feature, label="authenticated owner unit", graph=False
    ).to(device=action_measurement.device)

    # Gate on detached values first.  In particular, do not create action/noop
    # leaves and then return a zero loss.
    with torch.no_grad():
        detached_feature = motion.temporal_motion_quotient(
            (action_measurement - noop_measurement).float().contiguous(),
            require_input_grad=False,
        )
        detached_feature = _validate_feature(
            detached_feature, label="detached editor", graph=False
        )
        detached_scores = _segment_cosines(detached_feature, owner)
        score_values = {
            name: float(detached_scores[name].item()) for name in SEGMENT_ORDER
        }
    failures = [
        f"SEGMENT_SCORE_BELOW_FROZEN_THRESHOLD:{name}"
        for name in SEGMENT_ORDER
        if score_values[name] < authority.minimum_score_by_segment[name]
    ]
    if failures:
        return _zero_decision(
            authority=authority, scores=score_values, failures=failures
        )

    action_leaf = action_measurement.detach().clone().requires_grad_(True)
    noop_leaf = noop_measurement.detach().clone().requires_grad_(True)
    feature = motion.temporal_motion_quotient(
        (action_leaf - noop_leaf).float().contiguous(), require_input_grad=True
    )
    feature = _validate_feature(feature, label="graph editor", graph=True)
    graph_scores = _segment_cosines(feature, owner)
    rows = []
    measurement_digest = object_sha256(
        {
            "action": _tensor_sha256(action_measurement, label="action measurement"),
            "noop": _tensor_sha256(noop_measurement, label="noop measurement"),
        }
    )
    for ordinal, segment in enumerate(SEGMENT_ORDER):
        # Each constraint gets an independent VJP. retain_graph is used only
        # to evaluate the other preregistered rows; rows are never summed.
        action_q, noop_q = torch.autograd.grad(
            graph_scores[segment].reshape(()) / 4.0,
            (action_leaf, noop_leaf),
            retain_graph=ordinal + 1 < len(SEGMENT_ORDER),
            create_graph=False,
            allow_unused=False,
        )
        action_q = action_q.detach().float().contiguous()
        noop_q = noop_q.detach().float().contiguous()
        if (
            not torch.equal(action_q, -noop_q)
            or float(torch.linalg.vector_norm(action_q).item()) <= 0.0
            or not bool(torch.isfinite(action_q).all().item())
        ):
            raise CAPERPureT2VSegmentDPOError(
                f"{segment} cotangent is zero, asymmetric, or non-finite"
            )
        unsigned = {
            "schema_version": COTANGENT_ROW_SCHEMA_VERSION,
            "cell_id": authority.cell_id,
            "query_seed": authority.query_seed,
            "sample_index": 0,
            "segment": segment,
            "score": score_values[segment],
            "score_divisor": 4,
            "segment_authority_receipt_digest": authority.receipt_digest,
            "owner_packet_receipt_digest": owner_digest,
            "editor_measurement_digest": measurement_digest,
            "action_cotangent_sha256": _tensor_sha256(
                action_q, label=f"{segment} action cotangent"
            ),
            "noop_cotangent_sha256": _tensor_sha256(
                noop_q, label=f"{segment} no-op cotangent"
            ),
            "action_is_exact_negative_noop": True,
            "independent_hard_constraint": True,
            "mean_over_batch": False,
        }
        rows.append(
            SegmentCotangentRow(
                cell_id=authority.cell_id,
                query_seed=authority.query_seed,
                sample_index=0,
                segment=segment,
                score=score_values[segment],
                action_cotangent=action_q,
                noop_cotangent=noop_q,
                authority_receipt_digest=authority.receipt_digest,
                editor_measurement_digest=measurement_digest,
                receipt=_seal(unsigned),
            )
        )
    bundle_unsigned = {
        "schema_version": COTANGENT_BUNDLE_SCHEMA_VERSION,
        "cell_id": authority.cell_id,
        "query_seed": authority.query_seed,
        "segment_order": list(SEGMENT_ORDER),
        "row_receipt_digests": [row.receipt["receipt_digest"] for row in rows],
        "per_sample_constraint_count": len(rows),
        "rows_kept_independent": True,
        "worst_cell_or_explicit_multiconstraint": "explicit_multiconstraint",
        "row_averaging": False,
        "optimizer_constructed": False,
        "parameter_mutation_performed": False,
    }
    return SegmentCotangentBundle(
        cell_id=authority.cell_id,
        query_seed=authority.query_seed,
        rows=tuple(rows),
        receipt=_seal(bundle_unsigned),
    )


def candidate_per_sample_constraint_gate(
    value: Any,
    *,
    expected_candidate_delta_sha256: str,
    expected_registered_evaluator_digest: str,
) -> Mapping[str, Any]:
    """Re-evaluate every sample x segment cell after temporary direct-add.

    The evaluator is external and action-family specific.  This validator
    never derives semantics from raw latent distance.  A single failed cell
    makes the complete candidate a NO-GO.
    """

    candidate = _sha256(
        expected_candidate_delta_sha256, label="candidate delta SHA-256"
    )
    evaluator = _sha256(
        expected_registered_evaluator_digest,
        label="registered candidate evaluator digest",
    )
    if not isinstance(value, Mapping):
        raise CAPERPureT2VSegmentDPOError("candidate evaluation must be an object")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        value.get("schema_version") != CANDIDATE_EVALUATION_SCHEMA_VERSION
        or not isinstance(declared, str)
        or object_sha256(unsigned) != declared
        or value.get("candidate_delta_sha256") != candidate
        or value.get("registered_evaluator_digest") != evaluator
        or value.get("exact81") is not True
        or value.get("source_relative_terminal_residual") is not True
        or value.get("raw_target_rejected_mse_used") is not False
        or value.get("mask_track_pose_flow_used") is not False
    ):
        raise CAPERPureT2VSegmentDPOError("candidate evaluation binding differs")
    rows = value.get("cells")
    expected = {
        (cell, seed, segment)
        for cell in CELL_IDS
        for seed in QUERY_SEEDS_BY_CELL[cell]
        for segment in SEGMENT_ORDER
    }
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise CAPERPureT2VSegmentDPOError("candidate evaluation cell count differs")
    observed = set()
    failures = []
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CAPERPureT2VSegmentDPOError("candidate evaluation row differs")
        key = (row.get("cell_id"), row.get("query_seed"), row.get("segment"))
        if key not in expected or key in observed:
            raise CAPERPureT2VSegmentDPOError("candidate evaluation factorial differs")
        observed.add(key)
        margin = _finite(row.get("margin"), label=f"candidate margin {key}")
        threshold = _finite(row.get("minimum_margin"), label=f"candidate threshold {key}")
        passed = row.get("passed") is True and margin >= threshold
        if row.get("action_family_id") != ACTION_FAMILY_BY_CELL[key[0]]:
            raise CAPERPureT2VSegmentDPOError("candidate action family differs")
        if not passed:
            failures.append(f"CANDIDATE_CELL_FAILED:{key[0]}:{key[1]}:{key[2]}")
        normalized.append({**dict(row), "effective_passed": passed})
    if observed != expected:
        raise CAPERPureT2VSegmentDPOError("candidate evaluation cells are incomplete")
    worst = min(normalized, key=lambda row: float(row["margin"]) - float(row["minimum_margin"]))
    return _seal(
        {
            "schema_version": CANDIDATE_EVALUATION_SCHEMA_VERSION,
            "source_receipt_digest": declared,
            "candidate_delta_sha256": candidate,
            "registered_evaluator_digest": evaluator,
            "cell_count": len(normalized),
            "worst_cell": {
                "cell_id": worst["cell_id"],
                "query_seed": worst["query_seed"],
                "segment": worst["segment"],
                "slack": float(worst["margin"]) - float(worst["minimum_margin"]),
            },
            "failure_codes": sorted(failures),
            "canary_gate_passed": not failures,
            "per_sample_x_constraint_rechecked": True,
            "mean_over_batch_before_worst_cell": False,
            "optimizer_constructed": False,
            "retained_parameter_update": False,
        }
    )


def contract_receipt() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "cell_ids": list(CELL_IDS),
        "query_seeds_by_cell": {
            name: list(QUERY_SEEDS_BY_CELL[name]) for name in CELL_IDS
        },
        "segment_selector_digest": SELECTOR_DIGEST,
        "allowed_owner_channel": motion.ALLOWED_OWNER_TO_EDITOR_CHANNEL,
        "forbidden_owner_channels": list(motion.FORBIDDEN_OWNER_TO_EDITOR_CHANNELS),
        "forbidden_runtime_channels": list(FORBIDDEN_CHANNELS),
        "generation_gaussian_reused_as_training_epsilon": False,
        "semantic_objective": "action_specific_prompt_relative_hidden_quotient",
        "raw_target_rejected_mse_is_semantic_reward": False,
        "zero_update_builds_student_graph": False,
        "optimizer_constructed": False,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "ACTION_FAMILY_BY_CELL",
    "AUTHORITY_SCHEMA_VERSION",
    "CANDIDATE_EVALUATION_SCHEMA_VERSION",
    "CAPERPureT2VSegmentDPOError",
    "CELL_IDS",
    "FORBIDDEN_CHANNELS",
    "QUERY_SEEDS_BY_CELL",
    "SEGMENT_ORDER",
    "SELECTOR_DIGEST",
    "SegmentCotangentBundle",
    "SegmentCotangentRow",
    "SegmentZeroUpdateDecision",
    "ValidatedSegmentAuthority",
    "build_segment_cotangents_from_authenticated_measurements",
    "candidate_per_sample_constraint_gate",
    "canonical_json_bytes",
    "contract_receipt",
    "object_sha256",
    "segment_selector_receipt",
    "validate_segment_authority",
]
