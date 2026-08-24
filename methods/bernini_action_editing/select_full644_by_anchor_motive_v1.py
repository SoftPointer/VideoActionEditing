#!/usr/bin/env python3
"""Fail-closed, local-only MOTIVE-inspired selector for Bernini full644.

The selector consumes canonical, externally SHA-256-pinned *receipts*.  It never
loads anchor media, model weights, latents, datasets, or training code.  Anchor
positive/no-op projected-gradient deltas are attribution queries only.  Full644
chosen/no-op deltas are the only rows scored.  The sole output is a create-only,
read-only selection manifest; it is neither a dataset nor training authority.

The v4 serialized contract closes four important boundaries:

* every row is bound to a caller-pinned official exact-644 row-authority file;
* every anchor group is bound to a caller-pinned qualification/decision leaf,
  with one physical action parent plus all six exact veto branches;
* common randomness is bucket-aware: positive/no-op noise is identical per pair
  and identical within a geometry bucket, while different buckets may have
  different shapes and tensors;
* the public manifest validator requires either the original pinned inputs (and
  recomputes the scores) or an independently supplied expected manifest-file
  SHA-256.  A coherent rewrite of all serialized scores cannot self-authorize.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Optional, Sequence


ANCHOR_RECEIPT_SCHEMA = "bernini-motive-anchor-projected-gradient-receipt-v2"
ROW_RECEIPT_SCHEMA = "bernini-motive-full644-delta-projected-gradient-receipt-v2"
ROW_AUTHORITY_SCHEMA = "bernini-full644-exact-row-authority-manifest-v2"
ANCHOR_AUTHORITY_SCHEMA = "bernini-anchor-counterfactual-authority-v1"
ANCHOR_GROUP_QUALIFICATION_SCHEMA = (
    "bernini-anchor-counterfactual-group-qualification-v2"
)
ANCHOR_GROUP_DECISION_SCHEMA = "bernini-anchor-counterfactual-group-decision-v1"
ANCHOR_GROUP_LEAF_SCHEMA = "bernini-anchor-counterfactual-group-qualified-leaf-v2"
ANCHOR_BRANCH_AUTHORITY_SCHEMA = "bernini-anchor-gradient-branch-authority-v1"
ROW_BRANCH_AUTHORITY_SCHEMA = "bernini-full644-gradient-branch-authority-v1"
PAIR_RANDOMNESS_SCHEMA = "bernini-bucket-pair-common-randomness-v1"
GRADIENT_CLOSURE_SCHEMA = "bernini-motive-projected-gradient-common-closure-v2"
INPUT_PINS_SCHEMA = "bernini-motive-anchor-full644-selector-input-pins-v3"
SELECTION_MANIFEST_SCHEMA = "bernini-motive-full644-selection-manifest-v4"

METHOD = "motive-inspired-anchor-query-full644-offline-selection-v4"
EXPECTED_ROW_COUNT = 644
EXPECTED_FRAME_COUNT = 81
EXPECTED_LATENT_FRAME_COUNT = 21
FULL644_DATASET_SUMMARY_SHA256 = (
    "5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd"
)
FULL644_DATASET_INDEX_SHA256 = (
    "d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
)
FULL644_SOURCE_AUTHORITY_SHA256 = (
    "0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"
)

ACTION_QUERY_KIND = "action"
VETO_QUERY_KINDS = (
    "reverse",
    "incomplete",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
)
QUERY_KINDS = (ACTION_QUERY_KIND,) + VETO_QUERY_KINDS
MINIMUM_ACTION_QUERIES_PER_STRATUM = 2

COUNTERFACTUAL_AXIS = {
    "action": "requested_action",
    "reverse": "temporal_direction",
    "incomplete": "completion",
    "wrong_actor": "actor",
    "wrong_object": "object",
    "camera_only": "camera",
    "appearance_only": "appearance",
}
COUNTERFACTUAL_CONSTRUCTION = {
    "action": "self-generated-action-positive-v1",
    "reverse": "time-reversed-action-counterfactual-v1",
    "incomplete": "incomplete-action-counterfactual-v1",
    "wrong_actor": "wrong-actor-counterfactual-v1",
    "wrong_object": "wrong-object-counterfactual-v1",
    "camera_only": "camera-only-counterfactual-v1",
    "appearance_only": "appearance-only-counterfactual-v1",
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
_SERIALIZED_SCORE = re.compile(r"-?(?:0|1)\.[0-9]{12}\Z")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_VECTOR_DIMENSION = 65536
_MAX_ANCHOR_RECEIPTS = 4096
_SCORE_QUANTUM = Decimal("0.000000000001")

THRESHOLD_RULE = (
    "within-action-family-and-geometry-bucket-top-tail-rank-inclusive-ties-v2"
)
ACTION_VOTE_RULE = "positive-cosine-and-at-or-above-replayed-percentile-v2"
VETO_RULE = "any-positive-six-veto-query-at-or-above-replayed-percentile-v2"
VOTE_AGGREGATION = "per-query-within-family-and-bucket-percentile-vote-count-v2"
RANKING_RULE = (
    "vote-rate-desc-mean-action-score-desc-vote-count-desc-row-index-asc"
)
COMMON_RANDOMNESS_CONTRACT = (
    "same-noise-positive-noop-and-common-within-geometry-bucket-v2"
)

MATH_CONTRACT = {
    "anchor_delta": "projected-action-or-veto-minus-projected-noop",
    "row_delta": "projected-chosen-minus-projected-noop",
    "score": "cosine-anchor-delta-row-delta-then-round-half-even-12dp",
    "decimal_precision": 100,
    "score_serialization_decimal_places": 12,
    "threshold_rule": THRESHOLD_RULE,
    "action_vote_rule": ACTION_VOTE_RULE,
    "veto_rule": VETO_RULE,
    "ranking_tiebreak": RANKING_RULE,
}

QUERY_ONLY_CONTRACT = {
    "anchor_role": "attribution-query-only",
    "anchor_bytes_present_in_receipt": False,
    "anchor_bytes_loaded_by_selector": False,
    "optimizer_input_authorized": False,
    "training_target_authorized": False,
}

SAFETY_CONTRACT = {
    "anchor_is_training_target": False,
    "anchor_media_or_latent_bytes_in_manifest": False,
    "gpu_or_model_execution_performed": False,
    "manifest_is_training_dataset": False,
    "network_or_remote_execution_performed": False,
    "optimizer_authorized": False,
    "output_role": "offline-row-selection-only",
    "training_launch_authorized": False,
}


class MotiveSelectorError(RuntimeError):
    """An input, mathematical, or publication contract failed."""


@dataclass(frozen=True)
class AnchorFeature:
    query_id: str
    action_family: str
    bucket_id: str
    query_kind: str
    counterfactual_group_id: str
    parent_action_query_id: str
    action_semantics_sha256: str
    instruction_sha256: str
    shared_i0_frame_sha256: str
    content_id: str
    actor_id: str
    object_id: str
    scene_id: str
    appearance_id: str
    camera_id: str
    query_media_sha256: str
    noop_media_sha256: str
    parent_action_media_sha256: str
    query_media_authority_sha256: str
    noop_media_authority_sha256: str
    group_provenance_sha256: str
    branch_provenance_sha256: str
    mismatch_axis: str
    mismatch_authority_receipt_sha256: str
    counterfactual_construction: str
    delta: tuple[Decimal, ...]
    receipt_sha256: str
    motion_mask_sha256: str
    motion_mask_shape: tuple[int, int, int]
    noise_shape: tuple[int, int, int, int, int]
    common_noise_sha256: str
    bucket_noise_receipt_sha256: str
    anchor_authority_digest: str
    branch_authority_digest: str


@dataclass(frozen=True)
class RowFeature:
    row_index: int
    row_iid: str
    action_family: str
    bucket_id: str
    delta: tuple[Decimal, ...]
    receipt_sha256: str
    row_digest: str
    motion_mask_sha256: str
    motion_mask_shape: tuple[int, int, int]
    noise_shape: tuple[int, int, int, int, int]
    common_noise_sha256: str
    bucket_noise_receipt_sha256: str
    branch_authority_digest: str


@dataclass(frozen=True)
class AnchorGroupLeaf:
    counterfactual_group_id: str
    receipt_sha256: str
    leaf_digest: str
    parent_equivalence_sha256: str
    content_equivalence_sha256: str
    action_delta_equivalence_sha256: str
    delta_equivalence_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic, ASCII-only canonical JSON bytes (without newline)."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise MotiveSelectorError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise MotiveSelectorError(f"{label} must be a lowercase SHA-256")
    return value


def _authority_sha256(value: Any, *, label: str) -> str:
    digest = _sha256(value, label=label)
    if digest == "0" * 64:
        raise MotiveSelectorError(f"{label} cannot be an all-zero authority placeholder")
    return digest


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise MotiveSelectorError(f"{label} is not a canonical identifier")
    return value


def _integer(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    # bool is deliberately rejected even though bool subclasses int in Python.
    if type(value) is not int or not minimum <= value <= maximum:
        raise MotiveSelectorError(
            f"{label} must be a strict integer in [{minimum}, {maximum}]"
        )
    return value


def _exact_mapping(
    value: Any, fields: set[str] | frozenset[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise MotiveSelectorError(f"{label} field closure differs")
    return value


def _canonical_decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is not str or len(value) > 96 or _DECIMAL.fullmatch(value) is None:
        raise MotiveSelectorError(f"{label} must be a canonical decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise MotiveSelectorError(f"{label} is not a decimal") from error
    if not number.is_finite() or (number.is_zero() and value.startswith("-")):
        raise MotiveSelectorError(f"{label} must be finite and not negative zero")
    return number


def _score_string(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_HALF_EVEN
        quantized = value.quantize(_SCORE_QUANTUM)
    if quantized.is_zero():
        quantized = abs(quantized)
    return format(quantized, ".12f")


def _manifest_score(value: Any, *, label: str) -> Decimal:
    if type(value) is not str or _SERIALIZED_SCORE.fullmatch(value) is None:
        raise MotiveSelectorError(f"{label} is not a fixed 12-place score")
    score = Decimal(value)
    if (
        not score.is_finite()
        or not Decimal(-1) <= score <= Decimal(1)
        or (score.is_zero() and value.startswith("-"))
    ):
        raise MotiveSelectorError(f"{label} is outside canonical [-1, 1]")
    return score


def _vector(value: Any, *, dimension: int, label: str) -> tuple[Decimal, ...]:
    if not isinstance(value, list) or len(value) != dimension:
        raise MotiveSelectorError(f"{label} must contain exactly {dimension} values")
    return tuple(
        _canonical_decimal(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def cosine_similarity(
    left: Sequence[Decimal], right: Sequence[Decimal]
) -> Decimal:
    if len(left) != len(right) or len(left) < 2:
        raise MotiveSelectorError("cosine vectors have incompatible dimensions")
    with localcontext() as context:
        context.prec = 100
        dot = sum((a * b for a, b in zip(left, right)), Decimal(0))
        left_sq = sum((a * a for a in left), Decimal(0))
        right_sq = sum((b * b for b in right), Decimal(0))
        if left_sq <= 0 or right_sq <= 0:
            raise MotiveSelectorError("cosine vector has zero norm")
        result = dot / (left_sq.sqrt() * right_sq.sqrt())
        # Decimal roundoff at high precision may escape the mathematical range by
        # a few ulps for a vector compared with itself.
        tolerance = Decimal("1e-90")
        if result > 1 and result - 1 <= tolerance:
            result = Decimal(1)
        if result < -1 and -1 - result <= tolerance:
            result = Decimal(-1)
        if not Decimal(-1) <= result <= Decimal(1):
            raise MotiveSelectorError("cosine computation escaped [-1, 1]")
        return +result


def _validate_digest(
    value: Mapping[str, Any], *, digest_field: str, label: str
) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(digest_field, None)
    digest = _sha256(declared, label=f"{label} {digest_field}")
    if digest != object_sha256(unsigned):
        raise MotiveSelectorError(f"{label} digest differs")
    return digest


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise MotiveSelectorError(f"{label} must be a JSON boolean")
    return value


def _shape5(value: Any, *, label: str) -> tuple[int, int, int, int, int]:
    if not isinstance(value, list) or len(value) != 5:
        raise MotiveSelectorError(f"{label} must have five axes")
    shape = tuple(
        _integer(item, label=f"{label}[{index}]", minimum=1, maximum=65536)
        for index, item in enumerate(value)
    )
    if shape[:3] != (1, 16, EXPECTED_LATENT_FRAME_COUNT):
        raise MotiveSelectorError(f"{label} is not [1,16,21,H,W]")
    return shape  # type: ignore[return-value]


def _shape3(value: Any, *, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise MotiveSelectorError(f"{label} must have three axes")
    return tuple(
        _integer(item, label=f"{label}[{index}]", minimum=1, maximum=65536)
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def validate_gradient_closure(value: Any) -> tuple[str, int, str]:
    closure = _exact_mapping(
        value,
        {
            "schema_version",
            "base_model_checkpoint_tree_sha256",
            "parameter_scope_sha256",
            "frame_count",
            "latent_frame_count",
            "fixed_timestep",
            "projection",
            "motion_mask_producer",
            "gradient_producer",
            "common_randomness_producer",
            "common_randomness_contract",
        },
        label="gradient closure",
    )
    if closure["schema_version"] != GRADIENT_CLOSURE_SCHEMA:
        raise MotiveSelectorError("gradient closure schema differs")
    _sha256(
        closure["base_model_checkpoint_tree_sha256"],
        label="base model checkpoint tree",
    )
    _sha256(closure["parameter_scope_sha256"], label="parameter scope")
    if (
        _integer(closure["frame_count"], label="frame count", minimum=1, maximum=10000)
        != EXPECTED_FRAME_COUNT
        or _integer(
            closure["latent_frame_count"],
            label="latent frame count",
            minimum=1,
            maximum=10000,
        )
        != EXPECTED_LATENT_FRAME_COUNT
    ):
        raise MotiveSelectorError("gradient closure is not exact 81f/21 latent frames")
    timestep = _canonical_decimal(closure["fixed_timestep"], label="fixed timestep")
    if not Decimal(0) <= timestep <= Decimal(1000):
        raise MotiveSelectorError("fixed timestep is outside [0, 1000]")

    projection = _exact_mapping(
        closure["projection"],
        {
            "algorithm",
            "dimension",
            "state_sha256",
            "input_parameter_order_sha256",
            "normalization",
        },
        label="projection",
    )
    if (
        projection["algorithm"] != "fastfood-jl-v1"
        or projection["normalization"]
        != "none-before-delta-l2-only-at-selector-cosine-v1"
    ):
        raise MotiveSelectorError("projection algorithm/normalization differs")
    dimension = _integer(
        projection["dimension"],
        label="projection dimension",
        minimum=2,
        maximum=_MAX_VECTOR_DIMENSION,
    )
    _sha256(projection["state_sha256"], label="projection state")
    _sha256(
        projection["input_parameter_order_sha256"],
        label="projection parameter order",
    )

    mask_producer = _exact_mapping(
        closure["motion_mask_producer"],
        {
            "algorithm",
            "application",
            "normalization",
            "tracker_checkpoint_sha256",
            "implementation_sha256",
            "config_sha256",
        },
        label="motion mask producer",
    )
    if (
        mask_producer["algorithm"]
        != "alltracker-flow-magnitude-minmax-latent-loss-mask-v1"
        or mask_producer["application"]
        != "same-loss-space-mask-on-positive-and-noop-per-location-error-v1"
        or mask_producer["normalization"] != "per-video-minmax-epsilon-1e-6"
    ):
        raise MotiveSelectorError("motion mask producer semantics differ")
    for field in (
        "tracker_checkpoint_sha256",
        "implementation_sha256",
        "config_sha256",
    ):
        _sha256(mask_producer[field], label=f"motion mask producer {field}")

    producer = _exact_mapping(
        closure["gradient_producer"],
        {
            "source_archive_sha256",
            "entrypoint_sha256",
            "environment_sha256",
            "implementation_revision_sha256",
        },
        label="gradient producer",
    )
    for field, item in producer.items():
        _sha256(item, label=f"gradient producer {field}")

    randomness = _exact_mapping(
        closure["common_randomness_producer"],
        {
            "algorithm",
            "source_archive_sha256",
            "entrypoint_sha256",
            "environment_sha256",
            "implementation_revision_sha256",
            "config_sha256",
            "bucket_registry_sha256",
            "output_dtype",
        },
        label="common randomness producer",
    )
    if (
        randomness["algorithm"] != "bucket-aware-fixed-timestep-common-noise-v1"
        or randomness["output_dtype"] != "float32"
    ):
        raise MotiveSelectorError("common randomness producer semantics differ")
    for field in (
        "source_archive_sha256",
        "entrypoint_sha256",
        "environment_sha256",
        "implementation_revision_sha256",
        "config_sha256",
        "bucket_registry_sha256",
    ):
        _sha256(randomness[field], label=f"common randomness producer {field}")
    if closure["common_randomness_contract"] != COMMON_RANDOMNESS_CONTRACT:
        raise MotiveSelectorError("common randomness contract differs")
    return (
        object_sha256(closure),
        dimension,
        randomness["bucket_registry_sha256"],
    )


def _validate_pair_randomness(
    value: Any, *, expected_bucket_registry_sha256: str
) -> tuple[str, tuple[int, int, int, int, int], str, str, str]:
    pair = _exact_mapping(
        value,
        {
            "schema_version",
            "bucket_id",
            "bucket_registry_sha256",
            "shape",
            "dtype",
            "positive_noise_tensor_sha256",
            "noop_noise_tensor_sha256",
            "common_bucket_noise_tensor_sha256",
            "bucket_noise_receipt_sha256",
            "reused_for_positive_and_noop",
            "pair_randomness_digest",
        },
        label="pair randomness",
    )
    digest = _validate_digest(
        pair, digest_field="pair_randomness_digest", label="pair randomness"
    )
    if pair["schema_version"] != PAIR_RANDOMNESS_SCHEMA:
        raise MotiveSelectorError("pair randomness schema differs")
    bucket_id = _identifier(pair["bucket_id"], label="geometry bucket id")
    if (
        _sha256(pair["bucket_registry_sha256"], label="bucket registry")
        != expected_bucket_registry_sha256
    ):
        raise MotiveSelectorError("pair randomness bucket registry differs")
    shape = _shape5(pair["shape"], label="pair noise shape")
    if pair["dtype"] != "float32":
        raise MotiveSelectorError("pair noise dtype differs")
    positive = _sha256(
        pair["positive_noise_tensor_sha256"], label="positive noise tensor"
    )
    noop = _sha256(pair["noop_noise_tensor_sha256"], label="noop noise tensor")
    common = _sha256(
        pair["common_bucket_noise_tensor_sha256"],
        label="common bucket noise tensor",
    )
    if positive != noop or positive != common:
        raise MotiveSelectorError(
            "positive/noop pair noise must equal the common bucket noise"
        )
    if pair["reused_for_positive_and_noop"] is not True:
        raise MotiveSelectorError("pair noise reuse contract differs")
    bucket_receipt = _sha256(
        pair["bucket_noise_receipt_sha256"], label="bucket noise receipt"
    )
    return bucket_id, shape, common, bucket_receipt, digest


def _validate_motion_mask(
    value: Any, *, noise_shape: tuple[int, int, int, int, int]
) -> tuple[str, tuple[int, int, int]]:
    mask = _exact_mapping(
        value,
        {
            "tensor_sha256",
            "shape",
            "dtype",
            "minimum",
            "maximum",
            "nonzero_count",
            "applied_to_positive_and_noop",
        },
        label="motion mask",
    )
    tensor_sha = _sha256(mask["tensor_sha256"], label="motion mask tensor")
    shape = _shape3(mask["shape"], label="motion mask shape")
    if shape != (EXPECTED_LATENT_FRAME_COUNT, noise_shape[3], noise_shape[4]):
        raise MotiveSelectorError("motion mask does not match this pair noise shape")
    if (
        mask["dtype"] != "float32"
        or mask["minimum"] != "0"
        or mask["maximum"] != "1"
        or mask["applied_to_positive_and_noop"] is not True
    ):
        raise MotiveSelectorError("motion mask value/application contract differs")
    _integer(
        mask["nonzero_count"],
        label="motion mask nonzero count",
        minimum=1,
        maximum=shape[0] * shape[1] * shape[2],
    )
    return tensor_sha, shape


def _validate_projected_pair(
    value: Any, *, dimension: int, expected_positive_role: str
) -> tuple[tuple[Decimal, ...], str]:
    pair = _exact_mapping(
        value,
        {
            "dimension",
            "positive_role",
            "noop_role",
            "positive_projected_gradient",
            "noop_projected_gradient",
            "delta_projected_gradient",
            "delta_definition",
            "pair_digest",
        },
        label="projected gradient pair",
    )
    digest = _validate_digest(
        pair, digest_field="pair_digest", label="projected gradient pair"
    )
    if (
        _integer(
            pair["dimension"],
            label="projected pair dimension",
            minimum=2,
            maximum=_MAX_VECTOR_DIMENSION,
        )
        != dimension
        or pair["positive_role"] != expected_positive_role
        or pair["noop_role"] != "noop"
        or pair["delta_definition"]
        != "positive-minus-noop-linear-projection-before-l2-v1"
    ):
        raise MotiveSelectorError("projected gradient pair role/delta contract differs")
    positive = _vector(
        pair["positive_projected_gradient"],
        dimension=dimension,
        label="positive projected gradient",
    )
    noop = _vector(
        pair["noop_projected_gradient"],
        dimension=dimension,
        label="noop projected gradient",
    )
    delta = _vector(
        pair["delta_projected_gradient"],
        dimension=dimension,
        label="delta projected gradient",
    )
    with localcontext() as context:
        context.prec = 200
        for index, (positive_value, noop_value, delta_value) in enumerate(
            zip(positive, noop, delta)
        ):
            if positive_value - noop_value != delta_value:
                raise MotiveSelectorError(
                    f"declared projected delta differs at dimension {index}"
                )
    cosine_similarity(delta, delta)
    return delta, digest


ROW_AUTHORITY_FIELDS = frozenset(
    {
        "row_index",
        "row_iid",
        "action_family",
        "geometry_bucket_id",
        "source_media_sha256",
        "target_media_sha256",
        "instruction_sha256",
        "group_id",
        "actor_id",
        "object_id",
        "scene_id",
        "appearance_id",
        "endpoint_id",
        "shared_i0_frame_sha256",
        "source_authority_receipt_sha256",
        "target_authority_receipt_sha256",
        "instruction_authority_receipt_sha256",
        "shared_i0_authority_receipt_sha256",
        "endpoint_authority_receipt_sha256",
        "object_authority_receipt_sha256",
        "quality_receipt_sha256",
        "provenance_receipt_sha256",
        "same_i0_verified",
        "target_quality_qualified",
        "provenance_complete",
        "row_digest",
    }
)


def _validate_row_authority_entry(value: Any, *, expected_index: int) -> dict[str, Any]:
    row = dict(_exact_mapping(value, ROW_AUTHORITY_FIELDS, label="row authority entry"))
    if (
        _integer(
            row["row_index"],
            label="authority row index",
            minimum=0,
            maximum=EXPECTED_ROW_COUNT - 1,
        )
        != expected_index
    ):
        raise MotiveSelectorError("row authority indices are not exact contiguous order")
    _identifier(row["row_iid"], label="authority row IID")
    _identifier(row["action_family"], label="authority action family")
    _identifier(row["geometry_bucket_id"], label="authority geometry bucket")
    for field in (
        "source_media_sha256",
        "target_media_sha256",
        "instruction_sha256",
        "shared_i0_frame_sha256",
        "source_authority_receipt_sha256",
        "target_authority_receipt_sha256",
        "instruction_authority_receipt_sha256",
        "shared_i0_authority_receipt_sha256",
        "endpoint_authority_receipt_sha256",
        "object_authority_receipt_sha256",
        "quality_receipt_sha256",
        "provenance_receipt_sha256",
    ):
        _sha256(row[field], label=f"authority row {field}")
    for field in (
        "group_id",
        "actor_id",
        "object_id",
        "scene_id",
        "appearance_id",
        "endpoint_id",
    ):
        _identifier(row[field], label=f"authority row {field}")
    if row["source_media_sha256"] == row["target_media_sha256"]:
        raise MotiveSelectorError("authority source and target media alias")
    for field in (
        "same_i0_verified",
        "target_quality_qualified",
        "provenance_complete",
    ):
        if row[field] is not True:
            raise MotiveSelectorError(f"authority row {field} is not qualified")
    if row["source_authority_receipt_sha256"] == row[
        "target_authority_receipt_sha256"
    ]:
        raise MotiveSelectorError("authority source and target receipts alias")
    _validate_digest(row, digest_field="row_digest", label="row authority entry")
    return row


def _row_content_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical physical/semantic content committed by the official row list."""

    return {
        field: row[field]
        for field in sorted(
            ROW_AUTHORITY_FIELDS - {"row_index", "row_iid", "row_digest"}
        )
    }


def _row_physical_triplet(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_media_sha256": row["source_media_sha256"],
        "target_media_sha256": row["target_media_sha256"],
        "instruction_sha256": row["instruction_sha256"],
    }


def validate_row_authority_manifest(
    value: Any, *, expected_manifest_sha256: str
) -> tuple[dict[str, Any], ...]:
    expected_file_sha = _authority_sha256(
        expected_manifest_sha256,
        label="caller-pinned official row authority manifest",
    )
    if hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest() != expected_file_sha:
        raise MotiveSelectorError(
            "row authority differs from caller-pinned official file SHA-256"
        )
    manifest = _exact_mapping(
        value,
        {
            "schema_version",
            "authority_role",
            "dataset_summary_sha256",
            "dataset_index_sha256",
            "source_authority_receipt_sha256",
            "row_count",
            "rows",
            "exact_row_list_sha256",
            "row_content_root_sha256",
            "physical_triplet_root_sha256",
            "manifest_digest",
        },
        label="exact644 row authority manifest",
    )
    _validate_digest(
        manifest,
        digest_field="manifest_digest",
        label="exact644 row authority manifest",
    )
    if (
        manifest["schema_version"] != ROW_AUTHORITY_SCHEMA
        or manifest["authority_role"]
        != "exact-full644-row-membership-and-semantic-binding"
        or manifest["dataset_summary_sha256"] != FULL644_DATASET_SUMMARY_SHA256
        or manifest["dataset_index_sha256"] != FULL644_DATASET_INDEX_SHA256
        or manifest["source_authority_receipt_sha256"]
        != FULL644_SOURCE_AUTHORITY_SHA256
        or _integer(
            manifest["row_count"],
            label="authority row count",
            minimum=EXPECTED_ROW_COUNT,
            maximum=EXPECTED_ROW_COUNT,
        )
        != EXPECTED_ROW_COUNT
    ):
        raise MotiveSelectorError("exact644 row authority root binding differs")
    rows_value = manifest["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("row authority must contain exactly 644 rows")
    rows = tuple(
        _validate_row_authority_entry(item, expected_index=index)
        for index, item in enumerate(rows_value)
    )
    iids = [row["row_iid"] for row in rows]
    digests = [row["row_digest"] for row in rows]
    if len(set(iids)) != EXPECTED_ROW_COUNT or len(set(digests)) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("row authority IID/digest is not exact-644 unique")
    physical_triplets = [
        (
            row["source_media_sha256"],
            row["target_media_sha256"],
            row["instruction_sha256"],
        )
        for row in rows
    ]
    if len(set(physical_triplets)) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError(
            "row authority physical source/target/instruction triplet repeats"
        )
    content_digests = [object_sha256(_row_content_record(row)) for row in rows]
    if len(set(content_digests)) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("row authority physical/semantic content repeats")
    expected_row_list = object_sha256(digests)
    expected_content_root = object_sha256(
        [_row_content_record(row) for row in rows]
    )
    expected_triplet_root = object_sha256(
        [_row_physical_triplet(row) for row in rows]
    )
    if (
        manifest["exact_row_list_sha256"] != expected_row_list
        or manifest["row_content_root_sha256"] != expected_content_root
        or manifest["physical_triplet_root_sha256"] != expected_triplet_root
    ):
        raise MotiveSelectorError(
            "row authority exact row-list/content/triplet root does not replay"
        )
    if len({row["action_family"] for row in rows}) < 2:
        raise MotiveSelectorError("row authority must cover at least two action families")
    return rows


ANCHOR_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "counterfactual_group_id",
        "parent_action_query_id",
        "action_semantics_sha256",
        "instruction_sha256",
        "shared_i0_frame_sha256",
        "query_i0_frame_sha256",
        "noop_i0_frame_sha256",
        "same_i0_verified",
        "content_id",
        "actor_id",
        "object_id",
        "scene_id",
        "appearance_id",
        "camera_id",
        "query_media_sha256",
        "noop_media_sha256",
        "parent_action_media_sha256",
        "query_media_authority_sha256",
        "noop_media_authority_sha256",
        "counterfactual_group_provenance_sha256",
        "branch_provenance_sha256",
        "mismatch_axis",
        "mismatch_authority_receipt_sha256",
        "counterfactual_construction",
        "all_non_mismatch_axes_match_parent",
        "authority_digest",
    }
)


def _validate_anchor_authority(value: Any, *, query_id: str, kind: str) -> dict[str, Any]:
    authority = dict(
        _exact_mapping(value, ANCHOR_AUTHORITY_FIELDS, label="anchor authority")
    )
    digest = _validate_digest(
        authority, digest_field="authority_digest", label="anchor authority"
    )
    if authority["schema_version"] != ANCHOR_AUTHORITY_SCHEMA:
        raise MotiveSelectorError("anchor authority schema differs")
    for field in (
        "counterfactual_group_id",
        "parent_action_query_id",
        "content_id",
        "actor_id",
        "object_id",
        "scene_id",
        "appearance_id",
        "camera_id",
    ):
        _identifier(authority[field], label=f"anchor authority {field}")
    for field in (
        "action_semantics_sha256",
        "instruction_sha256",
        "shared_i0_frame_sha256",
        "query_i0_frame_sha256",
        "noop_i0_frame_sha256",
        "query_media_sha256",
        "noop_media_sha256",
        "parent_action_media_sha256",
        "query_media_authority_sha256",
        "noop_media_authority_sha256",
        "counterfactual_group_provenance_sha256",
        "branch_provenance_sha256",
        "mismatch_authority_receipt_sha256",
    ):
        _sha256(authority[field], label=f"anchor authority {field}")
    if (
        authority["same_i0_verified"] is not True
        or authority["all_non_mismatch_axes_match_parent"] is not True
        or authority["query_i0_frame_sha256"]
        != authority["shared_i0_frame_sha256"]
        or authority["noop_i0_frame_sha256"]
        != authority["shared_i0_frame_sha256"]
    ):
        raise MotiveSelectorError("anchor authority same-I0/non-mismatch contract differs")
    if authority["query_media_sha256"] == authority["noop_media_sha256"]:
        raise MotiveSelectorError("anchor query/noop media alias")
    if (
        authority["mismatch_axis"] != COUNTERFACTUAL_AXIS[kind]
        or authority["counterfactual_construction"]
        != COUNTERFACTUAL_CONSTRUCTION[kind]
    ):
        raise MotiveSelectorError(f"{kind} mismatch axis/provenance differs")
    if kind == ACTION_QUERY_KIND:
        if (
            authority["parent_action_query_id"] != query_id
            or authority["parent_action_media_sha256"]
            != authority["query_media_sha256"]
        ):
            raise MotiveSelectorError("action anchor is not its own physical parent")
    authority["authority_digest"] = digest
    return authority


def _validate_anchor_branch_authority(
    value: Any,
    *,
    authority: Mapping[str, Any],
    kind: str,
    motion_mask_sha256: str,
    pair_randomness_digest: str,
    projected_pair_digest: str,
) -> str:
    branch = _exact_mapping(
        value,
        {
            "schema_version",
            "anchor_authority_digest",
            "query_media_sha256",
            "noop_media_sha256",
            "query_media_authority_sha256",
            "noop_media_authority_sha256",
            "instruction_sha256",
            "shared_i0_frame_sha256",
            "positive_role",
            "positive_projected_gradient_artifact_sha256",
            "noop_projected_gradient_artifact_sha256",
            "motion_mask_artifact_sha256",
            "pair_randomness_digest",
            "projected_gradient_pair_digest",
            "gradient_producer_receipt_sha256",
            "branch_authority_digest",
        },
        label="anchor branch authority",
    )
    digest = _validate_digest(
        branch,
        digest_field="branch_authority_digest",
        label="anchor branch authority",
    )
    if branch["schema_version"] != ANCHOR_BRANCH_AUTHORITY_SCHEMA:
        raise MotiveSelectorError("anchor branch authority schema differs")
    exact = {
        "anchor_authority_digest": authority["authority_digest"],
        "query_media_sha256": authority["query_media_sha256"],
        "noop_media_sha256": authority["noop_media_sha256"],
        "query_media_authority_sha256": authority["query_media_authority_sha256"],
        "noop_media_authority_sha256": authority["noop_media_authority_sha256"],
        "instruction_sha256": authority["instruction_sha256"],
        "shared_i0_frame_sha256": authority["shared_i0_frame_sha256"],
        "positive_role": kind,
        "motion_mask_artifact_sha256": motion_mask_sha256,
        "pair_randomness_digest": pair_randomness_digest,
        "projected_gradient_pair_digest": projected_pair_digest,
    }
    for field, expected in exact.items():
        if branch[field] != expected:
            raise MotiveSelectorError(f"anchor branch does not exactly bind {field}")
    positive_artifact = _sha256(
        branch["positive_projected_gradient_artifact_sha256"],
        label="anchor positive projected-gradient artifact",
    )
    noop_artifact = _sha256(
        branch["noop_projected_gradient_artifact_sha256"],
        label="anchor noop projected-gradient artifact",
    )
    _sha256(
        branch["gradient_producer_receipt_sha256"],
        label="anchor gradient producer receipt",
    )
    if positive_artifact == noop_artifact:
        raise MotiveSelectorError("anchor positive/noop gradient artifacts alias")
    return digest


def _validate_row_branch_authority(
    value: Any,
    *,
    row_authority: Mapping[str, Any],
    motion_mask_sha256: str,
    pair_randomness_digest: str,
    projected_pair_digest: str,
) -> str:
    branch = _exact_mapping(
        value,
        {
            "schema_version",
            "row_digest",
            "source_media_sha256",
            "chosen_media_sha256",
            "noop_media_sha256",
            "instruction_sha256",
            "shared_i0_frame_sha256",
            "positive_projected_gradient_artifact_sha256",
            "noop_projected_gradient_artifact_sha256",
            "motion_mask_artifact_sha256",
            "pair_randomness_digest",
            "projected_gradient_pair_digest",
            "gradient_producer_receipt_sha256",
            "branch_authority_digest",
        },
        label="row branch authority",
    )
    digest = _validate_digest(
        branch,
        digest_field="branch_authority_digest",
        label="row branch authority",
    )
    if branch["schema_version"] != ROW_BRANCH_AUTHORITY_SCHEMA:
        raise MotiveSelectorError("row branch authority schema differs")
    exact = {
        "row_digest": row_authority["row_digest"],
        "source_media_sha256": row_authority["source_media_sha256"],
        "chosen_media_sha256": row_authority["target_media_sha256"],
        "noop_media_sha256": row_authority["source_media_sha256"],
        "instruction_sha256": row_authority["instruction_sha256"],
        "shared_i0_frame_sha256": row_authority["shared_i0_frame_sha256"],
        "motion_mask_artifact_sha256": motion_mask_sha256,
        "pair_randomness_digest": pair_randomness_digest,
        "projected_gradient_pair_digest": projected_pair_digest,
    }
    for field, expected in exact.items():
        if branch[field] != expected:
            raise MotiveSelectorError(f"row branch does not exactly bind {field}")
    positive_artifact = _sha256(
        branch["positive_projected_gradient_artifact_sha256"],
        label="row chosen projected-gradient artifact",
    )
    noop_artifact = _sha256(
        branch["noop_projected_gradient_artifact_sha256"],
        label="row noop projected-gradient artifact",
    )
    _sha256(
        branch["gradient_producer_receipt_sha256"],
        label="row gradient producer receipt",
    )
    if positive_artifact == noop_artifact:
        raise MotiveSelectorError("row chosen/noop gradient artifacts alias")
    return digest


def _receipt_file_binding(value: Mapping[str, Any], claimed_sha256: str, *, label: str) -> str:
    claimed = _sha256(claimed_sha256, label=f"{label} file")
    observed = hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()
    if observed != claimed:
        raise MotiveSelectorError(f"{label} value/file SHA-256 binding differs")
    return claimed


def validate_anchor_receipt(
    value: Any,
    *,
    receipt_sha256: str,
    expected_closure_sha256: str,
) -> AnchorFeature:
    receipt = _exact_mapping(
        value,
        {
            "schema_version",
            "anchor_query_id",
            "action_family",
            "query_kind",
            "gradient_closure",
            "gradient_closure_sha256",
            "pair_randomness",
            "motion_mask",
            "projected_gradient_pair",
            "anchor_authority",
            "branch_authority",
            "query_only_contract",
            "receipt_digest",
        },
        label="anchor receipt",
    )
    _validate_digest(receipt, digest_field="receipt_digest", label="anchor receipt")
    if receipt["schema_version"] != ANCHOR_RECEIPT_SCHEMA:
        raise MotiveSelectorError("anchor receipt schema differs")
    query_id = _identifier(receipt["anchor_query_id"], label="anchor query id")
    family = _identifier(receipt["action_family"], label="anchor action family")
    kind = receipt["query_kind"]
    if kind not in QUERY_KINDS:
        raise MotiveSelectorError("anchor query kind differs")
    closure_sha, dimension, registry_sha = validate_gradient_closure(
        receipt["gradient_closure"]
    )
    if (
        _sha256(receipt["gradient_closure_sha256"], label="anchor closure digest")
        != closure_sha
        or closure_sha != expected_closure_sha256
    ):
        raise MotiveSelectorError("anchor gradient closure differs from external pin")
    bucket_id, noise_shape, noise_sha, bucket_receipt, randomness_digest = (
        _validate_pair_randomness(
            receipt["pair_randomness"],
            expected_bucket_registry_sha256=registry_sha,
        )
    )
    mask_sha, mask_shape = _validate_motion_mask(
        receipt["motion_mask"], noise_shape=noise_shape
    )
    delta, pair_digest = _validate_projected_pair(
        receipt["projected_gradient_pair"],
        dimension=dimension,
        expected_positive_role=kind,
    )
    authority = _validate_anchor_authority(
        receipt["anchor_authority"], query_id=query_id, kind=kind
    )
    branch_digest = _validate_anchor_branch_authority(
        receipt["branch_authority"],
        authority=authority,
        kind=kind,
        motion_mask_sha256=mask_sha,
        pair_randomness_digest=randomness_digest,
        projected_pair_digest=pair_digest,
    )
    if receipt["query_only_contract"] != QUERY_ONLY_CONTRACT:
        raise MotiveSelectorError("anchor query-only safety contract differs")
    file_sha = _receipt_file_binding(receipt, receipt_sha256, label="anchor receipt")
    return AnchorFeature(
        query_id=query_id,
        action_family=family,
        bucket_id=bucket_id,
        query_kind=kind,
        counterfactual_group_id=authority["counterfactual_group_id"],
        parent_action_query_id=authority["parent_action_query_id"],
        action_semantics_sha256=authority["action_semantics_sha256"],
        instruction_sha256=authority["instruction_sha256"],
        shared_i0_frame_sha256=authority["shared_i0_frame_sha256"],
        content_id=authority["content_id"],
        actor_id=authority["actor_id"],
        object_id=authority["object_id"],
        scene_id=authority["scene_id"],
        appearance_id=authority["appearance_id"],
        camera_id=authority["camera_id"],
        query_media_sha256=authority["query_media_sha256"],
        noop_media_sha256=authority["noop_media_sha256"],
        parent_action_media_sha256=authority["parent_action_media_sha256"],
        query_media_authority_sha256=authority["query_media_authority_sha256"],
        noop_media_authority_sha256=authority["noop_media_authority_sha256"],
        group_provenance_sha256=authority[
            "counterfactual_group_provenance_sha256"
        ],
        branch_provenance_sha256=authority["branch_provenance_sha256"],
        mismatch_axis=authority["mismatch_axis"],
        mismatch_authority_receipt_sha256=authority[
            "mismatch_authority_receipt_sha256"
        ],
        counterfactual_construction=authority["counterfactual_construction"],
        delta=delta,
        receipt_sha256=file_sha,
        motion_mask_sha256=mask_sha,
        motion_mask_shape=mask_shape,
        noise_shape=noise_shape,
        common_noise_sha256=noise_sha,
        bucket_noise_receipt_sha256=bucket_receipt,
        anchor_authority_digest=authority["authority_digest"],
        branch_authority_digest=branch_digest,
    )


def validate_row_receipt(
    value: Any,
    *,
    receipt_sha256: str,
    expected_closure_sha256: str,
    expected_row_authority: Mapping[str, Any],
) -> RowFeature:
    receipt = _exact_mapping(
        value,
        {
            "schema_version",
            "row_index",
            "row_iid",
            "action_family",
            "row_authority",
            "gradient_closure",
            "gradient_closure_sha256",
            "pair_randomness",
            "motion_mask",
            "projected_gradient_pair",
            "branch_authority",
            "receipt_digest",
        },
        label="full644 row receipt",
    )
    _validate_digest(
        receipt, digest_field="receipt_digest", label="full644 row receipt"
    )
    if receipt["schema_version"] != ROW_RECEIPT_SCHEMA:
        raise MotiveSelectorError("full644 row receipt schema differs")
    row_index = _integer(
        receipt["row_index"],
        label="full644 row index",
        minimum=0,
        maximum=EXPECTED_ROW_COUNT - 1,
    )
    authority = _validate_row_authority_entry(
        receipt["row_authority"], expected_index=row_index
    )
    expected = dict(expected_row_authority)
    if authority != expected:
        raise MotiveSelectorError(
            "row receipt does not exactly match externally pinned row authority"
        )
    row_iid = _identifier(receipt["row_iid"], label="full644 row IID")
    family = _identifier(receipt["action_family"], label="full644 action family")
    if row_iid != authority["row_iid"] or family != authority["action_family"]:
        raise MotiveSelectorError("row receipt identity differs from row authority")
    closure_sha, dimension, registry_sha = validate_gradient_closure(
        receipt["gradient_closure"]
    )
    if (
        _sha256(receipt["gradient_closure_sha256"], label="row closure digest")
        != closure_sha
        or closure_sha != expected_closure_sha256
    ):
        raise MotiveSelectorError("row gradient closure differs from external pin")
    bucket_id, noise_shape, noise_sha, bucket_receipt, randomness_digest = (
        _validate_pair_randomness(
            receipt["pair_randomness"],
            expected_bucket_registry_sha256=registry_sha,
        )
    )
    if bucket_id != authority["geometry_bucket_id"]:
        raise MotiveSelectorError("row geometry bucket differs from row authority")
    mask_sha, mask_shape = _validate_motion_mask(
        receipt["motion_mask"], noise_shape=noise_shape
    )
    delta, pair_digest = _validate_projected_pair(
        receipt["projected_gradient_pair"],
        dimension=dimension,
        expected_positive_role="chosen",
    )
    branch_digest = _validate_row_branch_authority(
        receipt["branch_authority"],
        row_authority=authority,
        motion_mask_sha256=mask_sha,
        pair_randomness_digest=randomness_digest,
        projected_pair_digest=pair_digest,
    )
    file_sha = _receipt_file_binding(receipt, receipt_sha256, label="row receipt")
    return RowFeature(
        row_index=row_index,
        row_iid=row_iid,
        action_family=family,
        bucket_id=bucket_id,
        delta=delta,
        receipt_sha256=file_sha,
        row_digest=authority["row_digest"],
        motion_mask_sha256=mask_sha,
        motion_mask_shape=mask_shape,
        noise_shape=noise_shape,
        common_noise_sha256=noise_sha,
        bucket_noise_receipt_sha256=bucket_receipt,
        branch_authority_digest=branch_digest,
    )


def validate_selection_policy(value: Any) -> dict[str, Any]:
    policy = dict(
        _exact_mapping(
            value,
            {
                "action_percentile_basis_points",
                "veto_percentile_basis_points",
                "selection_budget",
                "minimum_action_votes",
                "threshold_rule",
                "action_vote_rule",
                "veto_rule",
                "vote_aggregation",
                "ranking_tiebreak",
                "veto_query_kinds",
            },
            label="selection policy",
        )
    )
    _integer(
        policy["action_percentile_basis_points"],
        label="action percentile basis points",
        minimum=0,
        maximum=9999,
    )
    _integer(
        policy["veto_percentile_basis_points"],
        label="veto percentile basis points",
        minimum=0,
        maximum=9999,
    )
    _integer(
        policy["selection_budget"],
        label="selection budget",
        minimum=1,
        maximum=EXPECTED_ROW_COUNT,
    )
    _integer(
        policy["minimum_action_votes"],
        label="minimum action votes",
        minimum=1,
        maximum=_MAX_ANCHOR_RECEIPTS,
    )
    if (
        policy["threshold_rule"] != THRESHOLD_RULE
        or policy["action_vote_rule"] != ACTION_VOTE_RULE
        or policy["veto_rule"] != VETO_RULE
        or policy["vote_aggregation"] != VOTE_AGGREGATION
        or policy["ranking_tiebreak"] != RANKING_RULE
        or policy["veto_query_kinds"] != list(VETO_QUERY_KINDS)
    ):
        raise MotiveSelectorError("selection policy semantic closure differs")
    return policy


def _validate_bucket_population(
    anchors: Sequence[AnchorFeature], rows: Sequence[RowFeature]
) -> None:
    buckets: dict[str, tuple[tuple[int, ...], str, str]] = {}
    shape_owner: dict[tuple[int, ...], str] = {}
    noise_owner: dict[str, str] = {}
    for item in list(anchors) + list(rows):
        closure = (
            tuple(item.noise_shape),
            item.common_noise_sha256,
            item.bucket_noise_receipt_sha256,
        )
        previous = buckets.setdefault(item.bucket_id, closure)
        if previous != closure:
            raise MotiveSelectorError(
                f"geometry bucket {item.bucket_id} lacks common-randomness closure"
            )
        old_shape_owner = shape_owner.setdefault(tuple(item.noise_shape), item.bucket_id)
        old_noise_owner = noise_owner.setdefault(item.common_noise_sha256, item.bucket_id)
        if old_shape_owner != item.bucket_id or old_noise_owner != item.bucket_id:
            raise MotiveSelectorError(
                "different geometry buckets alias a shape or common-noise tensor"
            )


def _validate_anchor_population(
    anchors: Sequence[AnchorFeature], *, check_delta_equivalence: bool = True
) -> None:
    if not anchors or len(anchors) > _MAX_ANCHOR_RECEIPTS:
        raise MotiveSelectorError("anchor population size is outside contract")
    query_ids = [anchor.query_id for anchor in anchors]
    receipt_shas = [anchor.receipt_sha256 for anchor in anchors]
    media_shas = [anchor.query_media_sha256 for anchor in anchors]
    media_authorities = [anchor.query_media_authority_sha256 for anchor in anchors]
    branch_provenance = [anchor.branch_provenance_sha256 for anchor in anchors]
    mismatch_receipts = [
        anchor.mismatch_authority_receipt_sha256 for anchor in anchors
    ]
    for values, label in (
        (query_ids, "query id"),
        (receipt_shas, "receipt"),
        (media_shas, "physical query media"),
        (media_authorities, "query media authority"),
        (branch_provenance, "branch provenance"),
        (mismatch_receipts, "counterfactual mismatch receipt"),
    ):
        if len(set(values)) != len(values):
            raise MotiveSelectorError(f"anchor {label} is not physically unique")

    # Content and object identities are group-owned across every branch, not
    # merely across action parents.  Repetition inside one seven-branch group is
    # expected; reuse by any branch of a different group is a Sybil collision.
    for field in ("content_id", "object_id"):
        owners: dict[str, str] = {}
        for anchor in anchors:
            previous = owners.setdefault(
                getattr(anchor, field), anchor.counterfactual_group_id
            )
            if previous != anchor.counterfactual_group_id:
                raise MotiveSelectorError(
                    f"anchor {field} is owned by more than one counterfactual group"
                )

    grouped: dict[tuple[str, str, str], list[AnchorFeature]] = {}
    for anchor in anchors:
        grouped.setdefault(
            (
                anchor.action_family,
                anchor.bucket_id,
                anchor.counterfactual_group_id,
            ),
            [],
        ).append(anchor)

    parent_equivalence_owners: dict[tuple[str, str, str], str] = {}
    content_equivalence_owners: dict[tuple[str, str, str], str] = {}
    action_delta_equivalence_owners: dict[tuple[str, str, str], str] = {}
    delta_equivalence_owners: dict[tuple[str, str, str], str] = {}
    action_triplet_owners: dict[tuple[str, str, str], str] = {}
    for group_key, group in grouped.items():
        by_kind: dict[str, AnchorFeature] = {}
        for anchor in group:
            if anchor.query_kind in by_kind:
                raise MotiveSelectorError(
                    f"counterfactual group {group_key} repeats a query kind"
                )
            by_kind[anchor.query_kind] = anchor
        if set(by_kind) != set(QUERY_KINDS):
            raise MotiveSelectorError(
                f"counterfactual group {group_key} lacks the exact six veto branches"
            )
        parent = by_kind[ACTION_QUERY_KIND]
        common_fields = (
            "action_family",
            "bucket_id",
            "counterfactual_group_id",
            "action_semantics_sha256",
            "instruction_sha256",
            "shared_i0_frame_sha256",
            "content_id",
            "noop_media_sha256",
            "noop_media_authority_sha256",
            "group_provenance_sha256",
            "noise_shape",
            "common_noise_sha256",
            "bucket_noise_receipt_sha256",
        )
        identity_fields = ("actor_id", "object_id", "scene_id", "appearance_id", "camera_id")
        for kind in QUERY_KINDS:
            branch = by_kind[kind]
            if any(getattr(branch, field) != getattr(parent, field) for field in common_fields):
                raise MotiveSelectorError(
                    f"{kind} counterfactual group common authority differs"
                )
            if branch.parent_action_query_id != parent.query_id:
                raise MotiveSelectorError(f"{kind} does not bind the exact action parent")
            if branch.parent_action_media_sha256 != parent.query_media_sha256:
                raise MotiveSelectorError(
                    f"{kind} does not bind the exact physical action parent"
                )
            expected_axis = COUNTERFACTUAL_AXIS[kind]
            expected_construction = COUNTERFACTUAL_CONSTRUCTION[kind]
            if (
                branch.mismatch_axis != expected_axis
                or branch.counterfactual_construction != expected_construction
            ):
                raise MotiveSelectorError(f"{kind} mismatch provenance differs")
            if kind != ACTION_QUERY_KIND and branch.query_media_sha256 == parent.query_media_sha256:
                raise MotiveSelectorError(f"{kind} aliases the physical action media")
            differences = {
                field
                for field in identity_fields
                if getattr(branch, field) != getattr(parent, field)
            }
            expected_differences = {
                "wrong_actor": {"actor_id"},
                "wrong_object": {"object_id"},
                "camera_only": {"camera_id"},
                "appearance_only": {"appearance_id"},
            }.get(kind, set())
            if differences != expected_differences:
                raise MotiveSelectorError(
                    f"{kind} has an inexact actor/object/scene/appearance/camera mismatch"
                )

        group_id = parent.counterfactual_group_id
        stratum = (parent.action_family, parent.bucket_id)
        parent_key = (
            parent.parent_action_media_sha256,
            parent.noop_media_sha256,
            parent.instruction_sha256,
        )
        content_key = (
            parent.content_id,
            parent.object_id,
            parent.shared_i0_frame_sha256,
        )
        delta_key = (
            object_sha256(
                {
                    kind: _scale_invariant_delta(by_kind[kind].delta)
                    for kind in QUERY_KINDS
                }
            )
            if check_delta_equivalence
            else None
        )
        action_delta_key = (
            object_sha256(_scale_invariant_delta(parent.delta))
            if check_delta_equivalence
            else None
        )
        action_triplet = (
            parent.noop_media_sha256,
            parent.query_media_sha256,
            parent.instruction_sha256,
        )
        equivalence_checks = [
            (parent_key, parent_equivalence_owners, "physical parent triplet"),
            (content_key, content_equivalence_owners, "content/object/I0"),
        ]
        if delta_key is not None:
            equivalence_checks.append(
                (
                    action_delta_key,
                    action_delta_equivalence_owners,
                    "action projected-delta direction",
                )
            )
            equivalence_checks.append(
                (delta_key, delta_equivalence_owners, "projected-delta direction")
            )
        for key, owners, label in equivalence_checks:
            scoped_key = (stratum[0], stratum[1], object_sha256(key))
            previous = owners.setdefault(scoped_key, group_id)
            if previous != group_id:
                raise MotiveSelectorError(
                    f"anchor group Sybil repeats {label} equivalence class"
                )
        previous_triplet = action_triplet_owners.setdefault(action_triplet, group_id)
        if previous_triplet != group_id:
            raise MotiveSelectorError(
                "anchor physical source/target/instruction triplet repeats"
            )

    all_actions = [
        anchor for anchor in anchors if anchor.query_kind == ACTION_QUERY_KIND
    ]
    for field in (
        "query_media_sha256",
        "content_id",
        "actor_id",
        "object_id",
        "scene_id",
        "appearance_id",
    ):
        values = [getattr(item, field) for item in all_actions]
        if len(set(values)) != len(values):
            raise MotiveSelectorError(
                f"action anchors are not globally cross-{field} unique"
            )

    strata: dict[tuple[str, str], list[AnchorFeature]] = {}
    for anchor in anchors:
        strata.setdefault((anchor.action_family, anchor.bucket_id), []).append(anchor)
    for key, stratum in strata.items():
        actions = [item for item in stratum if item.query_kind == ACTION_QUERY_KIND]
        if len(actions) < MINIMUM_ACTION_QUERIES_PER_STRATUM:
            raise MotiveSelectorError(f"anchor stratum {key} lacks two action parents")
        for field in (
            "query_media_sha256",
            "content_id",
            "actor_id",
            "object_id",
            "scene_id",
            "appearance_id",
        ):
            values = [getattr(item, field) for item in actions]
            if len(set(values)) != len(values):
                raise MotiveSelectorError(
                    f"action anchors in stratum {key} are not cross-{field} unique"
                )


def _scale_invariant_delta(delta: Sequence[Decimal]) -> list[list[int]]:
    """Represent a nonzero vector direction exactly, invariant to +ve scale."""

    fractions = [Fraction(value) for value in delta]
    pivot = next((abs(value) for value in fractions if value), None)
    if pivot is None:
        raise MotiveSelectorError("zero projected delta has no equivalence class")
    normalized = [value / pivot for value in fractions]
    return [[value.numerator, value.denominator] for value in normalized]


def _anchor_group_binding(
    group: Sequence[AnchorFeature],
) -> dict[str, Any]:
    by_kind = {anchor.query_kind: anchor for anchor in group}
    if set(by_kind) != set(QUERY_KINDS) or len(group) != len(QUERY_KINDS):
        raise MotiveSelectorError("anchor group cannot form an exact qualified leaf")
    parent = by_kind[ACTION_QUERY_KIND]
    parent_equivalence = object_sha256(
        {
            "action_family": parent.action_family,
            "bucket_id": parent.bucket_id,
            "parent_action_media_sha256": parent.parent_action_media_sha256,
            "noop_media_sha256": parent.noop_media_sha256,
            "instruction_sha256": parent.instruction_sha256,
            "shared_i0_frame_sha256": parent.shared_i0_frame_sha256,
            "action_semantics_sha256": parent.action_semantics_sha256,
        }
    )
    content_equivalence = object_sha256(
        {
            "content_id": parent.content_id,
            "actor_id": parent.actor_id,
            "object_id": parent.object_id,
            "scene_id": parent.scene_id,
            "appearance_id": parent.appearance_id,
            "shared_i0_frame_sha256": parent.shared_i0_frame_sha256,
        }
    )
    delta_equivalence = object_sha256(
        {
            kind: _scale_invariant_delta(by_kind[kind].delta)
            for kind in QUERY_KINDS
        }
    )
    action_delta_equivalence = object_sha256(
        _scale_invariant_delta(parent.delta)
    )
    members = [
        {
            "query_kind": kind,
            "query_id": by_kind[kind].query_id,
            "receipt_sha256": by_kind[kind].receipt_sha256,
            "anchor_authority_digest": by_kind[kind].anchor_authority_digest,
            "branch_authority_digest": by_kind[kind].branch_authority_digest,
            "delta_direction_sha256": object_sha256(
                _scale_invariant_delta(by_kind[kind].delta)
            ),
        }
        for kind in QUERY_KINDS
    ]
    return {
        "counterfactual_group_id": parent.counterfactual_group_id,
        "action_family": parent.action_family,
        "bucket_id": parent.bucket_id,
        "parent_action_query_id": parent.query_id,
        "parent_action_media_sha256": parent.query_media_sha256,
        "noop_media_sha256": parent.noop_media_sha256,
        "instruction_sha256": parent.instruction_sha256,
        "shared_i0_frame_sha256": parent.shared_i0_frame_sha256,
        "content_id": parent.content_id,
        "object_id": parent.object_id,
        "parent_equivalence_sha256": parent_equivalence,
        "content_equivalence_sha256": content_equivalence,
        "action_delta_equivalence_sha256": action_delta_equivalence,
        "delta_equivalence_sha256": delta_equivalence,
        "members": members,
    }


def validate_anchor_group_leaf_receipt(
    value: Any,
    *,
    receipt_sha256: str,
    expected_leaf_digest: str,
    anchors: Sequence[AnchorFeature],
) -> AnchorGroupLeaf:
    """Validate one externally pinned group qualification and admission decision."""

    leaf = _exact_mapping(
        value,
        {
            "schema_version",
            "counterfactual_group_id",
            "qualification_receipt",
            "decision_receipt",
            "leaf_digest",
        },
        label="anchor group qualified leaf",
    )
    leaf_digest = _validate_digest(
        leaf,
        digest_field="leaf_digest",
        label="anchor group qualified leaf",
    )
    if (
        leaf["schema_version"] != ANCHOR_GROUP_LEAF_SCHEMA
        or leaf_digest
        != _authority_sha256(
            expected_leaf_digest, label="expected anchor group leaf"
        )
    ):
        raise MotiveSelectorError(
            "anchor group leaf differs from caller-pinned qualification decision"
        )
    group_id = _identifier(
        leaf["counterfactual_group_id"], label="anchor group leaf id"
    )
    if not anchors or any(
        anchor.counterfactual_group_id != group_id for anchor in anchors
    ):
        raise MotiveSelectorError("anchor group leaf membership differs")
    binding = _anchor_group_binding(anchors)

    qualification = _exact_mapping(
        leaf["qualification_receipt"],
        {
            "schema_version",
            "counterfactual_group_id",
            "qualification_status",
            "qualifier_authority_sha256",
            "evidence_binding",
            "qualification_checks",
            "qualification_receipt_digest",
        },
        label="anchor group qualification receipt",
    )
    qualification_digest = _validate_digest(
        qualification,
        digest_field="qualification_receipt_digest",
        label="anchor group qualification receipt",
    )
    checks = _exact_mapping(
        qualification["qualification_checks"],
        {
            "same_i0_verified",
            "physical_parent_verified",
            "all_six_vetoes_verified",
            "non_mismatch_axes_verified",
            "provenance_complete",
            "quality_qualified",
            "selector_query_only",
        },
        label="anchor group qualification checks",
    )
    evidence_binding = _exact_mapping(
        qualification["evidence_binding"],
        set(binding),
        label="anchor group qualification evidence binding",
    )
    if (
        qualification["schema_version"] != ANCHOR_GROUP_QUALIFICATION_SCHEMA
        or qualification["counterfactual_group_id"] != group_id
        or qualification["qualification_status"] != "qualified"
        or dict(evidence_binding) != binding
        or any(value is not True for value in checks.values())
    ):
        raise MotiveSelectorError("anchor group qualification evidence differs")
    _authority_sha256(
        qualification["qualifier_authority_sha256"],
        label="anchor group qualifier authority",
    )

    decision = _exact_mapping(
        leaf["decision_receipt"],
        {
            "schema_version",
            "counterfactual_group_id",
            "qualification_receipt_digest",
            "decision",
            "decision_authority_sha256",
            "optimizer_authorized",
            "training_target_authorized",
            "decision_receipt_digest",
        },
        label="anchor group decision receipt",
    )
    _validate_digest(
        decision,
        digest_field="decision_receipt_digest",
        label="anchor group decision receipt",
    )
    if (
        decision["schema_version"] != ANCHOR_GROUP_DECISION_SCHEMA
        or decision["counterfactual_group_id"] != group_id
        or decision["qualification_receipt_digest"] != qualification_digest
        or decision["decision"] != "admit-selector-query-only"
        or decision["optimizer_authorized"] is not False
        or decision["training_target_authorized"] is not False
    ):
        raise MotiveSelectorError("anchor group qualification decision differs")
    _authority_sha256(
        decision["decision_authority_sha256"],
        label="anchor group decision authority",
    )
    file_sha = _receipt_file_binding(
        leaf, receipt_sha256, label="anchor group leaf receipt"
    )
    return AnchorGroupLeaf(
        counterfactual_group_id=group_id,
        receipt_sha256=file_sha,
        leaf_digest=leaf_digest,
        parent_equivalence_sha256=binding["parent_equivalence_sha256"],
        content_equivalence_sha256=binding["content_equivalence_sha256"],
        action_delta_equivalence_sha256=binding[
            "action_delta_equivalence_sha256"
        ],
        delta_equivalence_sha256=binding["delta_equivalence_sha256"],
    )


def _validate_anchor_group_leaf_population(
    anchors: Sequence[AnchorFeature],
    leaf_receipts: Sequence[tuple[Mapping[str, Any], str]],
    expected_leaf_digests: Mapping[str, str],
) -> tuple[AnchorGroupLeaf, ...]:
    grouped: dict[str, list[AnchorFeature]] = {}
    for anchor in anchors:
        grouped.setdefault(anchor.counterfactual_group_id, []).append(anchor)
    if type(expected_leaf_digests) is not dict:
        raise MotiveSelectorError("expected anchor group leaf map must be an exact dict")
    expected = {
        _identifier(key, label="expected anchor group id"): _authority_sha256(
            digest, label="expected anchor group leaf digest"
        )
        for key, digest in expected_leaf_digests.items()
    }
    if set(expected) != set(grouped) or len(leaf_receipts) != len(grouped):
        raise MotiveSelectorError(
            "anchor group qualification leaf coverage is not exact"
        )
    leaves: list[AnchorGroupLeaf] = []
    seen_groups: set[str] = set()
    for value, file_sha in leaf_receipts:
        if not isinstance(value, Mapping):
            raise MotiveSelectorError("anchor group leaf root must be an object")
        group_id = _identifier(
            value.get("counterfactual_group_id"), label="anchor group leaf id"
        )
        if group_id in seen_groups or group_id not in expected:
            raise MotiveSelectorError("anchor group qualification leaf repeats")
        seen_groups.add(group_id)
        leaves.append(
            validate_anchor_group_leaf_receipt(
                value,
                receipt_sha256=file_sha,
                expected_leaf_digest=expected[group_id],
                anchors=grouped[group_id],
            )
        )
    for field in ("receipt_sha256", "leaf_digest"):
        values = [getattr(leaf, field) for leaf in leaves]
        if len(set(values)) != len(values):
            raise MotiveSelectorError(
                f"anchor group qualified leaves repeat {field} equivalence"
            )
    return tuple(sorted(leaves, key=lambda leaf: leaf.counterfactual_group_id))


def _percentile_cutoff(scores: Sequence[Decimal], basis_points: int) -> Decimal:
    if not scores:
        raise MotiveSelectorError("cannot compute percentile over no scores")
    _integer(
        basis_points,
        label="percentile basis points",
        minimum=0,
        maximum=9999,
    )
    ordered = sorted(scores)
    if basis_points == 0:
        index = 0
    else:
        index = (basis_points * len(ordered) + 9999) // 10000 - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def _query_summary(anchor: AnchorFeature, cutoff: Decimal) -> dict[str, Any]:
    return {
        "query_id": anchor.query_id,
        "action_family": anchor.action_family,
        "bucket_id": anchor.bucket_id,
        "query_kind": anchor.query_kind,
        "counterfactual_group_id": anchor.counterfactual_group_id,
        "parent_action_query_id": anchor.parent_action_query_id,
        "action_semantics_sha256": anchor.action_semantics_sha256,
        "instruction_sha256": anchor.instruction_sha256,
        "shared_i0_frame_sha256": anchor.shared_i0_frame_sha256,
        "content_id": anchor.content_id,
        "actor_id": anchor.actor_id,
        "object_id": anchor.object_id,
        "scene_id": anchor.scene_id,
        "appearance_id": anchor.appearance_id,
        "camera_id": anchor.camera_id,
        "query_media_sha256": anchor.query_media_sha256,
        "noop_media_sha256": anchor.noop_media_sha256,
        "parent_action_media_sha256": anchor.parent_action_media_sha256,
        "query_media_authority_sha256": anchor.query_media_authority_sha256,
        "noop_media_authority_sha256": anchor.noop_media_authority_sha256,
        "counterfactual_group_provenance_sha256": anchor.group_provenance_sha256,
        "branch_provenance_sha256": anchor.branch_provenance_sha256,
        "mismatch_axis": anchor.mismatch_axis,
        "mismatch_authority_receipt_sha256": anchor.mismatch_authority_receipt_sha256,
        "counterfactual_construction": anchor.counterfactual_construction,
        "receipt_sha256": anchor.receipt_sha256,
        "motion_mask_sha256": anchor.motion_mask_sha256,
        "motion_mask_shape": list(anchor.motion_mask_shape),
        "noise_shape": list(anchor.noise_shape),
        "common_bucket_noise_tensor_sha256": anchor.common_noise_sha256,
        "bucket_noise_receipt_sha256": anchor.bucket_noise_receipt_sha256,
        "anchor_authority_digest": anchor.anchor_authority_digest,
        "branch_authority_digest": anchor.branch_authority_digest,
        "cutoff": _score_string(cutoff),
    }


def _score_population(
    anchors: Sequence[AnchorFeature],
    rows: Sequence[RowFeature],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_stratum: dict[tuple[str, str], list[RowFeature]] = {}
    for row in rows:
        rows_by_stratum.setdefault((row.action_family, row.bucket_id), []).append(row)
    anchors_by_stratum: dict[tuple[str, str], list[AnchorFeature]] = {}
    for anchor in anchors:
        anchors_by_stratum.setdefault(
            (anchor.action_family, anchor.bucket_id), []
        ).append(anchor)
    if set(rows_by_stratum) != set(anchors_by_stratum):
        raise MotiveSelectorError(
            "row and anchor action-family/geometry-bucket strata differ"
        )

    score_by_query_row: dict[tuple[str, int], Decimal] = {}
    cutoff_by_query: dict[str, Decimal] = {}
    for key in sorted(rows_by_stratum):
        stratum_rows = sorted(rows_by_stratum[key], key=lambda item: item.row_index)
        stratum_anchors = sorted(
            anchors_by_stratum[key], key=lambda item: item.query_id
        )
        for anchor in stratum_anchors:
            scores = []
            for row in stratum_rows:
                score = Decimal(_score_string(cosine_similarity(anchor.delta, row.delta)))
                score_by_query_row[(anchor.query_id, row.row_index)] = score
                scores.append(score)
            basis_points = (
                policy["action_percentile_basis_points"]
                if anchor.query_kind == ACTION_QUERY_KIND
                else policy["veto_percentile_basis_points"]
            )
            cutoff_by_query[anchor.query_id] = _percentile_cutoff(
                scores, basis_points
            )

    query_records = [
        _query_summary(anchor, cutoff_by_query[anchor.query_id])
        for anchor in sorted(
            anchors,
            key=lambda item: (
                item.action_family,
                item.bucket_id,
                item.query_kind,
                item.query_id,
            ),
        )
    ]

    row_records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.row_index):
        stratum_queries = sorted(
            anchors_by_stratum[(row.action_family, row.bucket_id)],
            key=lambda item: item.query_id,
        )
        action_scores: list[dict[str, Any]] = []
        veto_scores: list[dict[str, Any]] = []
        for anchor in stratum_queries:
            score = score_by_query_row[(anchor.query_id, row.row_index)]
            cutoff = cutoff_by_query[anchor.query_id]
            vote = score > 0 and score >= cutoff
            record = {
                "query_id": anchor.query_id,
                "score": _score_string(score),
                "cutoff": _score_string(cutoff),
                "vote": vote,
            }
            if anchor.query_kind == ACTION_QUERY_KIND:
                action_scores.append(record)
            else:
                veto_scores.append(record)
        action_vote_count = sum(1 for item in action_scores if item["vote"])
        action_query_count = len(action_scores)
        with localcontext() as context:
            context.prec = 100
            mean_action = sum(
                (Decimal(item["score"]) for item in action_scores), Decimal(0)
            ) / Decimal(action_query_count)
            vote_rate = Decimal(action_vote_count) / Decimal(action_query_count)
        veto_query_map = {anchor.query_id: anchor.query_kind for anchor in stratum_queries}
        veto_kinds = [
            kind
            for kind in VETO_QUERY_KINDS
            if any(
                item["vote"] and veto_query_map[item["query_id"]] == kind
                for item in veto_scores
            )
        ]
        eligible = (
            action_vote_count >= policy["minimum_action_votes"] and not veto_kinds
        )
        row_records.append(
            {
                "row_index": row.row_index,
                "row_iid": row.row_iid,
                "action_family": row.action_family,
                "bucket_id": row.bucket_id,
                "receipt_sha256": row.receipt_sha256,
                "row_digest": row.row_digest,
                "motion_mask_sha256": row.motion_mask_sha256,
                "motion_mask_shape": list(row.motion_mask_shape),
                "noise_shape": list(row.noise_shape),
                "common_bucket_noise_tensor_sha256": row.common_noise_sha256,
                "bucket_noise_receipt_sha256": row.bucket_noise_receipt_sha256,
                "branch_authority_digest": row.branch_authority_digest,
                "action_scores": action_scores,
                "veto_scores": veto_scores,
                "action_vote_count": action_vote_count,
                "action_query_count": action_query_count,
                "action_vote_rate": _score_string(vote_rate),
                "mean_action_score": _score_string(mean_action),
                "veto_kinds": veto_kinds,
                "eligible_before_budget": eligible,
                "rank": None,
            }
        )

    eligible_rows = [row for row in row_records if row["eligible_before_budget"]]
    eligible_rows.sort(
        key=lambda item: (
            -Decimal(item["action_vote_rate"]),
            -Decimal(item["mean_action_score"]),
            -item["action_vote_count"],
            item["row_index"],
        )
    )
    for rank, row in enumerate(eligible_rows, 1):
        row["rank"] = rank
    return query_records, row_records


def _selected_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rank": row["rank"],
        "row_index": row["row_index"],
        "row_iid": row["row_iid"],
        "action_family": row["action_family"],
        "bucket_id": row["bucket_id"],
        "row_digest": row["row_digest"],
        "receipt_sha256": row["receipt_sha256"],
        "action_vote_count": row["action_vote_count"],
        "action_query_count": row["action_query_count"],
        "action_vote_rate": row["action_vote_rate"],
        "mean_action_score": row["mean_action_score"],
    }


def build_selection_manifest(
    *,
    anchor_receipts: Sequence[tuple[Mapping[str, Any], str]],
    anchor_group_leaf_receipts: Sequence[tuple[Mapping[str, Any], str]],
    expected_anchor_group_leaf_digests: Mapping[str, str],
    row_receipts: Sequence[tuple[Mapping[str, Any], str]],
    row_authority_manifest: Mapping[str, Any],
    row_authority_manifest_sha256: str,
    expected_row_authority_manifest_sha256: str,
    selection_policy: Mapping[str, Any],
    expected_gradient_closure_sha256: str,
    input_pins_sha256: str,
) -> dict[str, Any]:
    policy = validate_selection_policy(selection_policy)
    expected_closure = _sha256(
        expected_gradient_closure_sha256, label="expected gradient closure"
    )
    pinset_sha = _sha256(input_pins_sha256, label="input pins")
    authority_file_sha = _sha256(
        row_authority_manifest_sha256, label="row authority manifest file"
    )
    expected_authority_file_sha = _authority_sha256(
        expected_row_authority_manifest_sha256,
        label="caller-pinned official row authority manifest file",
    )
    if (
        hashlib.sha256(canonical_json_bytes(row_authority_manifest) + b"\n").hexdigest()
        != authority_file_sha
        or authority_file_sha != expected_authority_file_sha
    ):
        raise MotiveSelectorError(
            "row authority differs from caller-pinned official file SHA-256"
        )
    authority_rows = validate_row_authority_manifest(
        row_authority_manifest,
        expected_manifest_sha256=expected_authority_file_sha,
    )
    authority_digest = row_authority_manifest["manifest_digest"]
    if len(row_receipts) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("row receipts are not exact full644")

    anchors = [
        validate_anchor_receipt(
            value,
            receipt_sha256=receipt_sha,
            expected_closure_sha256=expected_closure,
        )
        for value, receipt_sha in anchor_receipts
    ]
    rows: list[RowFeature] = []
    for value, receipt_sha in row_receipts:
        if not isinstance(value, Mapping):
            raise MotiveSelectorError("row receipt root must be an object")
        row_index = _integer(
            value.get("row_index"),
            label="prebound full644 row index",
            minimum=0,
            maximum=EXPECTED_ROW_COUNT - 1,
        )
        rows.append(
            validate_row_receipt(
                value,
                receipt_sha256=receipt_sha,
                expected_closure_sha256=expected_closure,
                expected_row_authority=authority_rows[row_index],
            )
        )
    rows.sort(key=lambda item: item.row_index)
    if [row.row_index for row in rows] != list(range(EXPECTED_ROW_COUNT)):
        raise MotiveSelectorError("row receipts do not cover exact indices 0..643")
    if len({row.row_iid for row in rows}) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("row receipt IIDs are not exact-644 unique")
    if len({row.receipt_sha256 for row in rows}) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("row receipt files are not exact-644 unique")
    _validate_anchor_population(anchors)
    group_leaves = _validate_anchor_group_leaf_population(
        anchors,
        anchor_group_leaf_receipts,
        expected_anchor_group_leaf_digests,
    )
    _validate_bucket_population(anchors, rows)

    queries, row_records = _score_population(anchors, rows, policy)
    eligible = sorted(
        (row for row in row_records if row["eligible_before_budget"]),
        key=lambda item: item["rank"],
    )
    selected = [
        _selected_record(row) for row in eligible[: policy["selection_budget"]]
    ]
    counts = {
        "anchor_queries": len(anchors),
        "qualified_anchor_groups": len(group_leaves),
        "action_queries": sum(
            1 for anchor in anchors if anchor.query_kind == ACTION_QUERY_KIND
        ),
        "veto_queries": sum(
            1 for anchor in anchors if anchor.query_kind in VETO_QUERY_KINDS
        ),
        "geometry_buckets": len({row.bucket_id for row in rows}),
        "full644_rows": len(rows),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
    }
    manifest: dict[str, Any] = {
        "schema_version": SELECTION_MANIFEST_SCHEMA,
        "method": METHOD,
        "input_binding": {
            "input_pins_sha256": pinset_sha,
            "expected_gradient_closure_sha256": expected_closure,
            "row_authority_manifest_sha256": authority_file_sha,
            "expected_official_row_authority_manifest_sha256": (
                expected_authority_file_sha
            ),
            "row_authority_manifest_digest": authority_digest,
            "row_authority_exact_row_list_sha256": row_authority_manifest[
                "exact_row_list_sha256"
            ],
            "row_authority_content_root_sha256": row_authority_manifest[
                "row_content_root_sha256"
            ],
            "row_authority_physical_triplet_root_sha256": row_authority_manifest[
                "physical_triplet_root_sha256"
            ],
            "full644_dataset_summary_sha256": FULL644_DATASET_SUMMARY_SHA256,
            "full644_dataset_index_sha256": FULL644_DATASET_INDEX_SHA256,
            "full644_source_authority_sha256": FULL644_SOURCE_AUTHORITY_SHA256,
            "anchor_receipt_sha256s": sorted(
                anchor.receipt_sha256 for anchor in anchors
            ),
            "anchor_group_qualified_leaves": [
                {
                    "counterfactual_group_id": leaf.counterfactual_group_id,
                    "receipt_sha256": leaf.receipt_sha256,
                    "leaf_digest": leaf.leaf_digest,
                    "parent_equivalence_sha256": leaf.parent_equivalence_sha256,
                    "content_equivalence_sha256": leaf.content_equivalence_sha256,
                    "action_delta_equivalence_sha256": (
                        leaf.action_delta_equivalence_sha256
                    ),
                    "delta_equivalence_sha256": leaf.delta_equivalence_sha256,
                }
                for leaf in group_leaves
            ],
            "row_receipt_sha256s": [row.receipt_sha256 for row in rows],
            "row_iids": [row.row_iid for row in rows],
            "row_digests": [row.row_digest for row in rows],
        },
        "math_contract": dict(MATH_CONTRACT),
        "safety_contract": dict(SAFETY_CONTRACT),
        "selection_policy": policy,
        "queries": queries,
        "rows": row_records,
        "selected_rows": selected,
        "counts": counts,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    _validate_selection_manifest_structure(manifest)
    return manifest


QUERY_SUMMARY_FIELDS = frozenset(
    {
        "query_id",
        "action_family",
        "bucket_id",
        "query_kind",
        "counterfactual_group_id",
        "parent_action_query_id",
        "action_semantics_sha256",
        "instruction_sha256",
        "shared_i0_frame_sha256",
        "content_id",
        "actor_id",
        "object_id",
        "scene_id",
        "appearance_id",
        "camera_id",
        "query_media_sha256",
        "noop_media_sha256",
        "parent_action_media_sha256",
        "query_media_authority_sha256",
        "noop_media_authority_sha256",
        "counterfactual_group_provenance_sha256",
        "branch_provenance_sha256",
        "mismatch_axis",
        "mismatch_authority_receipt_sha256",
        "counterfactual_construction",
        "receipt_sha256",
        "motion_mask_sha256",
        "motion_mask_shape",
        "noise_shape",
        "common_bucket_noise_tensor_sha256",
        "bucket_noise_receipt_sha256",
        "anchor_authority_digest",
        "branch_authority_digest",
        "cutoff",
    }
)

ROW_SUMMARY_FIELDS = frozenset(
    {
        "row_index",
        "row_iid",
        "action_family",
        "bucket_id",
        "receipt_sha256",
        "row_digest",
        "motion_mask_sha256",
        "motion_mask_shape",
        "noise_shape",
        "common_bucket_noise_tensor_sha256",
        "bucket_noise_receipt_sha256",
        "branch_authority_digest",
        "action_scores",
        "veto_scores",
        "action_vote_count",
        "action_query_count",
        "action_vote_rate",
        "mean_action_score",
        "veto_kinds",
        "eligible_before_budget",
        "rank",
    }
)


def _validate_query_summary(value: Any) -> dict[str, Any]:
    query = dict(_exact_mapping(value, QUERY_SUMMARY_FIELDS, label="query summary"))
    for field in (
        "query_id",
        "action_family",
        "bucket_id",
        "counterfactual_group_id",
        "parent_action_query_id",
        "content_id",
        "actor_id",
        "object_id",
        "scene_id",
        "appearance_id",
        "camera_id",
    ):
        _identifier(query[field], label=f"query summary {field}")
    if query["query_kind"] not in QUERY_KINDS:
        raise MotiveSelectorError("query summary kind differs")
    for field in (
        "action_semantics_sha256",
        "instruction_sha256",
        "shared_i0_frame_sha256",
        "query_media_sha256",
        "noop_media_sha256",
        "parent_action_media_sha256",
        "query_media_authority_sha256",
        "noop_media_authority_sha256",
        "counterfactual_group_provenance_sha256",
        "branch_provenance_sha256",
        "mismatch_authority_receipt_sha256",
        "receipt_sha256",
        "motion_mask_sha256",
        "common_bucket_noise_tensor_sha256",
        "bucket_noise_receipt_sha256",
        "anchor_authority_digest",
        "branch_authority_digest",
    ):
        _sha256(query[field], label=f"query summary {field}")
    mask_shape = _shape3(query["motion_mask_shape"], label="query mask shape")
    noise_shape = _shape5(query["noise_shape"], label="query noise shape")
    if mask_shape != (EXPECTED_LATENT_FRAME_COUNT, noise_shape[3], noise_shape[4]):
        raise MotiveSelectorError("query mask/noise geometry differs")
    if (
        query["mismatch_axis"] != COUNTERFACTUAL_AXIS[query["query_kind"]]
        or query["counterfactual_construction"]
        != COUNTERFACTUAL_CONSTRUCTION[query["query_kind"]]
    ):
        raise MotiveSelectorError("query summary mismatch provenance differs")
    _manifest_score(query["cutoff"], label="query cutoff")
    return query


def _manifest_anchor_feature(query: Mapping[str, Any]) -> AnchorFeature:
    # Delta is intentionally absent from the manifest.  The metadata population
    # validator does not inspect it; a nonzero placeholder avoids broadening the
    # serialized output into an optimizer-compatible representation target.
    return AnchorFeature(
        query_id=query["query_id"],
        action_family=query["action_family"],
        bucket_id=query["bucket_id"],
        query_kind=query["query_kind"],
        counterfactual_group_id=query["counterfactual_group_id"],
        parent_action_query_id=query["parent_action_query_id"],
        action_semantics_sha256=query["action_semantics_sha256"],
        instruction_sha256=query["instruction_sha256"],
        shared_i0_frame_sha256=query["shared_i0_frame_sha256"],
        content_id=query["content_id"],
        actor_id=query["actor_id"],
        object_id=query["object_id"],
        scene_id=query["scene_id"],
        appearance_id=query["appearance_id"],
        camera_id=query["camera_id"],
        query_media_sha256=query["query_media_sha256"],
        noop_media_sha256=query["noop_media_sha256"],
        parent_action_media_sha256=query["parent_action_media_sha256"],
        query_media_authority_sha256=query["query_media_authority_sha256"],
        noop_media_authority_sha256=query["noop_media_authority_sha256"],
        group_provenance_sha256=query["counterfactual_group_provenance_sha256"],
        branch_provenance_sha256=query["branch_provenance_sha256"],
        mismatch_axis=query["mismatch_axis"],
        mismatch_authority_receipt_sha256=query[
            "mismatch_authority_receipt_sha256"
        ],
        counterfactual_construction=query["counterfactual_construction"],
        delta=(Decimal(1), Decimal(0)),
        receipt_sha256=query["receipt_sha256"],
        motion_mask_sha256=query["motion_mask_sha256"],
        motion_mask_shape=tuple(query["motion_mask_shape"]),
        noise_shape=tuple(query["noise_shape"]),
        common_noise_sha256=query["common_bucket_noise_tensor_sha256"],
        bucket_noise_receipt_sha256=query["bucket_noise_receipt_sha256"],
        anchor_authority_digest=query["anchor_authority_digest"],
        branch_authority_digest=query["branch_authority_digest"],
    )


def _validate_score_entry(value: Any, *, label: str) -> dict[str, Any]:
    entry = dict(
        _exact_mapping(
            value, {"query_id", "score", "cutoff", "vote"}, label=label
        )
    )
    _identifier(entry["query_id"], label=f"{label} query id")
    _manifest_score(entry["score"], label=f"{label} score")
    _manifest_score(entry["cutoff"], label=f"{label} cutoff")
    _strict_bool(entry["vote"], label=f"{label} vote")
    return entry


def _expected_selected(row: Mapping[str, Any]) -> dict[str, Any]:
    return _selected_record(row)


def _validate_selection_manifest_structure(value: Any) -> None:
    manifest = _exact_mapping(
        value,
        {
            "schema_version",
            "method",
            "input_binding",
            "math_contract",
            "safety_contract",
            "selection_policy",
            "queries",
            "rows",
            "selected_rows",
            "counts",
            "manifest_digest",
        },
        label="selection manifest",
    )
    _validate_digest(
        manifest, digest_field="manifest_digest", label="selection manifest"
    )
    if (
        manifest["schema_version"] != SELECTION_MANIFEST_SCHEMA
        or manifest["method"] != METHOD
        or manifest["math_contract"] != MATH_CONTRACT
        or manifest["safety_contract"] != SAFETY_CONTRACT
    ):
        raise MotiveSelectorError("selection manifest root contract differs")
    policy = validate_selection_policy(manifest["selection_policy"])

    binding = _exact_mapping(
        manifest["input_binding"],
        {
            "input_pins_sha256",
            "expected_gradient_closure_sha256",
            "row_authority_manifest_sha256",
            "expected_official_row_authority_manifest_sha256",
            "row_authority_manifest_digest",
            "row_authority_exact_row_list_sha256",
            "row_authority_content_root_sha256",
            "row_authority_physical_triplet_root_sha256",
            "full644_dataset_summary_sha256",
            "full644_dataset_index_sha256",
            "full644_source_authority_sha256",
            "anchor_receipt_sha256s",
            "anchor_group_qualified_leaves",
            "row_receipt_sha256s",
            "row_iids",
            "row_digests",
        },
        label="manifest input binding",
    )
    for field in (
        "input_pins_sha256",
        "expected_gradient_closure_sha256",
        "row_authority_manifest_sha256",
        "expected_official_row_authority_manifest_sha256",
        "row_authority_manifest_digest",
        "row_authority_exact_row_list_sha256",
        "row_authority_content_root_sha256",
        "row_authority_physical_triplet_root_sha256",
    ):
        _sha256(binding[field], label=f"manifest input {field}")
    if (
        binding["row_authority_manifest_sha256"]
        != binding["expected_official_row_authority_manifest_sha256"]
    ):
        raise MotiveSelectorError(
            "manifest row authority is not the caller-pinned official file"
        )
    if (
        binding["full644_dataset_summary_sha256"] != FULL644_DATASET_SUMMARY_SHA256
        or binding["full644_dataset_index_sha256"] != FULL644_DATASET_INDEX_SHA256
        or binding["full644_source_authority_sha256"]
        != FULL644_SOURCE_AUTHORITY_SHA256
    ):
        raise MotiveSelectorError("manifest full644 root binding differs")

    queries_value = manifest["queries"]
    if not isinstance(queries_value, list) or not queries_value:
        raise MotiveSelectorError("manifest queries must be a nonempty list")
    queries = [_validate_query_summary(item) for item in queries_value]
    expected_query_order = sorted(
        queries,
        key=lambda item: (
            item["action_family"],
            item["bucket_id"],
            item["query_kind"],
            item["query_id"],
        ),
    )
    if queries != expected_query_order:
        raise MotiveSelectorError("manifest query ordering differs")
    if len({item["query_id"] for item in queries}) != len(queries):
        raise MotiveSelectorError("manifest query ids repeat")
    anchor_features = [_manifest_anchor_feature(item) for item in queries]
    _validate_anchor_population(anchor_features, check_delta_equivalence=False)
    query_by_id = {item["query_id"]: item for item in queries}

    anchor_receipt_shas = binding["anchor_receipt_sha256s"]
    if (
        not isinstance(anchor_receipt_shas, list)
        or anchor_receipt_shas != sorted(item["receipt_sha256"] for item in queries)
    ):
        raise MotiveSelectorError("manifest anchor receipt closure differs")
    leaf_summaries = binding["anchor_group_qualified_leaves"]
    if not isinstance(leaf_summaries, list):
        raise MotiveSelectorError("manifest anchor group leaf closure differs")
    parsed_leaf_summaries: list[dict[str, str]] = []
    for item in leaf_summaries:
        parsed = dict(
            _exact_mapping(
                item,
                {
                    "counterfactual_group_id",
                    "receipt_sha256",
                    "leaf_digest",
                    "parent_equivalence_sha256",
                    "content_equivalence_sha256",
                    "action_delta_equivalence_sha256",
                    "delta_equivalence_sha256",
                },
                label="manifest anchor group qualified leaf",
            )
        )
        _identifier(
            parsed["counterfactual_group_id"],
            label="manifest anchor group leaf id",
        )
        for field in (
            "receipt_sha256",
            "leaf_digest",
            "parent_equivalence_sha256",
            "content_equivalence_sha256",
            "action_delta_equivalence_sha256",
            "delta_equivalence_sha256",
        ):
            _sha256(parsed[field], label=f"manifest anchor group leaf {field}")
        parsed_leaf_summaries.append(parsed)
    expected_group_ids = sorted(
        {
            item["counterfactual_group_id"]
            for item in queries
            if item["query_kind"] == ACTION_QUERY_KIND
        }
    )
    if (
        parsed_leaf_summaries
        != sorted(
            parsed_leaf_summaries,
            key=lambda item: item["counterfactual_group_id"],
        )
        or [item["counterfactual_group_id"] for item in parsed_leaf_summaries]
        != expected_group_ids
        or len({item["receipt_sha256"] for item in parsed_leaf_summaries})
        != len(parsed_leaf_summaries)
        or len({item["leaf_digest"] for item in parsed_leaf_summaries})
        != len(parsed_leaf_summaries)
    ):
        raise MotiveSelectorError("manifest anchor group qualified leaf closure differs")

    rows_value = manifest["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("manifest rows are not exact full644")
    parsed_rows: list[dict[str, Any]] = []
    scores_by_query: dict[str, list[Decimal]] = {
        item["query_id"]: [] for item in queries
    }
    for expected_index, value_row in enumerate(rows_value):
        row = dict(
            _exact_mapping(value_row, ROW_SUMMARY_FIELDS, label="manifest row")
        )
        if (
            _integer(
                row["row_index"],
                label="manifest row index",
                minimum=0,
                maximum=EXPECTED_ROW_COUNT - 1,
            )
            != expected_index
        ):
            raise MotiveSelectorError("manifest rows are not in exact index order")
        _identifier(row["row_iid"], label="manifest row IID")
        _identifier(row["action_family"], label="manifest row action family")
        _identifier(row["bucket_id"], label="manifest row bucket")
        for field in (
            "receipt_sha256",
            "row_digest",
            "motion_mask_sha256",
            "common_bucket_noise_tensor_sha256",
            "bucket_noise_receipt_sha256",
            "branch_authority_digest",
        ):
            _sha256(row[field], label=f"manifest row {field}")
        mask_shape = _shape3(row["motion_mask_shape"], label="manifest row mask shape")
        noise_shape = _shape5(row["noise_shape"], label="manifest row noise shape")
        if mask_shape != (EXPECTED_LATENT_FRAME_COUNT, noise_shape[3], noise_shape[4]):
            raise MotiveSelectorError("manifest row mask/noise geometry differs")
        if not isinstance(row["action_scores"], list) or not isinstance(
            row["veto_scores"], list
        ):
            raise MotiveSelectorError("manifest row score lists differ")
        action_scores = [
            _validate_score_entry(item, label="manifest action score")
            for item in row["action_scores"]
        ]
        veto_scores = [
            _validate_score_entry(item, label="manifest veto score")
            for item in row["veto_scores"]
        ]
        expected_queries = sorted(
            (
                query
                for query in queries
                if query["action_family"] == row["action_family"]
                and query["bucket_id"] == row["bucket_id"]
            ),
            key=lambda item: item["query_id"],
        )
        expected_action_ids = [
            item["query_id"]
            for item in expected_queries
            if item["query_kind"] == ACTION_QUERY_KIND
        ]
        expected_veto_ids = [
            item["query_id"]
            for item in expected_queries
            if item["query_kind"] in VETO_QUERY_KINDS
        ]
        if [item["query_id"] for item in action_scores] != expected_action_ids:
            raise MotiveSelectorError("manifest row action-query closure differs")
        if [item["query_id"] for item in veto_scores] != expected_veto_ids:
            raise MotiveSelectorError("manifest row veto-query closure differs")
        for entry in action_scores + veto_scores:
            scores_by_query[entry["query_id"]].append(Decimal(entry["score"]))
        row["action_scores"] = action_scores
        row["veto_scores"] = veto_scores
        _integer(
            row["action_vote_count"],
            label="manifest action vote count",
            minimum=0,
            maximum=len(action_scores),
        )
        _integer(
            row["action_query_count"],
            label="manifest action query count",
            minimum=1,
            maximum=_MAX_ANCHOR_RECEIPTS,
        )
        _manifest_score(row["action_vote_rate"], label="manifest action vote rate")
        _manifest_score(row["mean_action_score"], label="manifest mean action score")
        if (
            not isinstance(row["veto_kinds"], list)
            or any(kind not in VETO_QUERY_KINDS for kind in row["veto_kinds"])
            or row["veto_kinds"]
            != [kind for kind in VETO_QUERY_KINDS if kind in row["veto_kinds"]]
        ):
            raise MotiveSelectorError("manifest row veto kind closure differs")
        _strict_bool(
            row["eligible_before_budget"],
            label="manifest row eligibility",
        )
        if row["rank"] is not None:
            _integer(
                row["rank"],
                label="manifest row rank",
                minimum=1,
                maximum=EXPECTED_ROW_COUNT,
            )
        parsed_rows.append(row)

    if len({row["row_iid"] for row in parsed_rows}) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("manifest row IIDs are not exact-644 unique")

    # Replay each query's percentile from the serialized 12-place scores.  This
    # intentionally does not trust either query.cutoff or the repeated row cutoff.
    cutoff_by_query: dict[str, Decimal] = {}
    for query in queries:
        query_id = query["query_id"]
        basis_points = (
            policy["action_percentile_basis_points"]
            if query["query_kind"] == ACTION_QUERY_KIND
            else policy["veto_percentile_basis_points"]
        )
        cutoff = _percentile_cutoff(scores_by_query[query_id], basis_points)
        cutoff_by_query[query_id] = cutoff
        if query["cutoff"] != _score_string(cutoff):
            raise MotiveSelectorError("manifest query cutoff does not replay")

    for row in parsed_rows:
        for entry in row["action_scores"] + row["veto_scores"]:
            score = Decimal(entry["score"])
            cutoff = cutoff_by_query[entry["query_id"]]
            expected_vote = score > 0 and score >= cutoff
            if entry["cutoff"] != _score_string(cutoff):
                raise MotiveSelectorError("manifest row cutoff does not replay")
            if entry["vote"] is not expected_vote:
                raise MotiveSelectorError("manifest row vote does not replay")

        action_count = len(row["action_scores"])
        action_votes = sum(1 for item in row["action_scores"] if item["vote"])
        with localcontext() as context:
            context.prec = 100
            mean = sum(
                (Decimal(item["score"]) for item in row["action_scores"]),
                Decimal(0),
            ) / Decimal(action_count)
            rate = Decimal(action_votes) / Decimal(action_count)
        veto_kinds = [
            kind
            for kind in VETO_QUERY_KINDS
            if any(
                item["vote"]
                and query_by_id[item["query_id"]]["query_kind"] == kind
                for item in row["veto_scores"]
            )
        ]
        eligible = action_votes >= policy["minimum_action_votes"] and not veto_kinds
        if (
            row["action_vote_count"] != action_votes
            or row["action_query_count"] != action_count
            or row["action_vote_rate"] != _score_string(rate)
            or row["mean_action_score"] != _score_string(mean)
            or row["veto_kinds"] != veto_kinds
            or row["eligible_before_budget"] is not eligible
        ):
            raise MotiveSelectorError("manifest row aggregate/eligibility does not replay")

    eligible_rows = [row for row in parsed_rows if row["eligible_before_budget"]]
    eligible_rows.sort(
        key=lambda item: (
            -Decimal(item["action_vote_rate"]),
            -Decimal(item["mean_action_score"]),
            -item["action_vote_count"],
            item["row_index"],
        )
    )
    expected_rank = {row["row_index"]: rank for rank, row in enumerate(eligible_rows, 1)}
    for row in parsed_rows:
        if row["rank"] != expected_rank.get(row["row_index"]):
            raise MotiveSelectorError("manifest row rank does not replay")

    selected_value = manifest["selected_rows"]
    if not isinstance(selected_value, list):
        raise MotiveSelectorError("manifest selected rows must be a list")
    expected_selected = [
        _expected_selected(row)
        for row in eligible_rows[: policy["selection_budget"]]
    ]
    if selected_value != expected_selected:
        raise MotiveSelectorError("manifest selected prefix does not replay")

    row_receipt_shas = binding["row_receipt_sha256s"]
    row_iids = binding["row_iids"]
    row_digests = binding["row_digests"]
    if (
        not isinstance(row_receipt_shas, list)
        or row_receipt_shas != [row["receipt_sha256"] for row in parsed_rows]
        or not isinstance(row_iids, list)
        or row_iids != [row["row_iid"] for row in parsed_rows]
        or not isinstance(row_digests, list)
        or row_digests != [row["row_digest"] for row in parsed_rows]
        or len(set(row_receipt_shas)) != EXPECTED_ROW_COUNT
        or len(set(row_iids)) != EXPECTED_ROW_COUNT
        or len(set(row_digests)) != EXPECTED_ROW_COUNT
    ):
        raise MotiveSelectorError("manifest exact644 receipt/row authority closure differs")

    # Replay bucket common-randomness closure using both query and row summaries.
    bucket_values: dict[str, tuple[tuple[int, ...], str, str]] = {}
    shape_owner: dict[tuple[int, ...], str] = {}
    noise_owner: dict[str, str] = {}
    for item in queries + parsed_rows:
        closure = (
            tuple(item["noise_shape"]),
            item["common_bucket_noise_tensor_sha256"],
            item["bucket_noise_receipt_sha256"],
        )
        bucket = item["bucket_id"]
        if bucket_values.setdefault(bucket, closure) != closure:
            raise MotiveSelectorError("manifest bucket common randomness differs")
        if shape_owner.setdefault(closure[0], bucket) != bucket:
            raise MotiveSelectorError("manifest buckets alias a noise shape")
        if noise_owner.setdefault(closure[1], bucket) != bucket:
            raise MotiveSelectorError("manifest buckets alias common noise")

    counts = _exact_mapping(
        manifest["counts"],
        {
            "anchor_queries",
            "qualified_anchor_groups",
            "action_queries",
            "veto_queries",
            "geometry_buckets",
            "full644_rows",
            "eligible_rows",
            "selected_rows",
        },
        label="manifest counts",
    )
    for field, maximum in (
        ("anchor_queries", _MAX_ANCHOR_RECEIPTS),
        ("qualified_anchor_groups", _MAX_ANCHOR_RECEIPTS // len(QUERY_KINDS)),
        ("action_queries", _MAX_ANCHOR_RECEIPTS),
        ("veto_queries", _MAX_ANCHOR_RECEIPTS),
        ("geometry_buckets", EXPECTED_ROW_COUNT),
        ("full644_rows", EXPECTED_ROW_COUNT),
        ("eligible_rows", EXPECTED_ROW_COUNT),
        ("selected_rows", EXPECTED_ROW_COUNT),
    ):
        _integer(counts[field], label=f"manifest count {field}", minimum=0, maximum=maximum)
    expected_counts = {
        "anchor_queries": len(queries),
        "qualified_anchor_groups": len(parsed_leaf_summaries),
        "action_queries": sum(
            query["query_kind"] == ACTION_QUERY_KIND for query in queries
        ),
        "veto_queries": sum(query["query_kind"] in VETO_QUERY_KINDS for query in queries),
        "geometry_buckets": len(bucket_values),
        "full644_rows": EXPECTED_ROW_COUNT,
        "eligible_rows": len(eligible_rows),
        "selected_rows": len(expected_selected),
    }
    if dict(counts) != expected_counts:
        raise MotiveSelectorError("manifest counts do not replay")


def validate_selection_manifest(
    value: Any,
    *,
    expected_manifest_sha256: Optional[str] = None,
    anchor_receipts: Optional[
        Sequence[tuple[Mapping[str, Any], str]]
    ] = None,
    anchor_group_leaf_receipts: Optional[
        Sequence[tuple[Mapping[str, Any], str]]
    ] = None,
    expected_anchor_group_leaf_digests: Optional[Mapping[str, str]] = None,
    row_receipts: Optional[Sequence[tuple[Mapping[str, Any], str]]] = None,
    row_authority_manifest: Optional[Mapping[str, Any]] = None,
    row_authority_manifest_sha256: Optional[str] = None,
    expected_row_authority_manifest_sha256: Optional[str] = None,
    selection_policy: Optional[Mapping[str, Any]] = None,
    expected_gradient_closure_sha256: Optional[str] = None,
    input_pins_sha256: Optional[str] = None,
) -> None:
    """Validate with an external trust root, never a self-signed manifest alone.

    Callers must choose exactly one mode: an independently obtained expected
    canonical manifest-file SHA-256, or the complete original pinned inputs.
    The latter rebuilds every projected-delta cosine, cutoff, vote, aggregate,
    rank and selected row before accepting the serialized value.
    """

    raw_values = (
        anchor_receipts,
        anchor_group_leaf_receipts,
        expected_anchor_group_leaf_digests,
        row_receipts,
        row_authority_manifest,
        row_authority_manifest_sha256,
        expected_row_authority_manifest_sha256,
        selection_policy,
        expected_gradient_closure_sha256,
        input_pins_sha256,
    )
    any_raw = any(item is not None for item in raw_values)
    all_raw = all(item is not None for item in raw_values)
    if expected_manifest_sha256 is not None:
        if any_raw:
            raise MotiveSelectorError(
                "manifest validator trust modes are mutually exclusive"
            )
        expected = _authority_sha256(
            expected_manifest_sha256,
            label="independently pinned selection manifest file",
        )
        observed = hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()
        if observed != expected:
            raise MotiveSelectorError(
                "selection manifest differs from independent expected file SHA-256"
            )
        _validate_selection_manifest_structure(value)
        return
    if not all_raw:
        raise MotiveSelectorError(
            "manifest validation requires original pinned inputs or independent expected SHA-256"
        )
    rebuilt = build_selection_manifest(
        anchor_receipts=anchor_receipts,  # type: ignore[arg-type]
        anchor_group_leaf_receipts=anchor_group_leaf_receipts,  # type: ignore[arg-type]
        expected_anchor_group_leaf_digests=(
            expected_anchor_group_leaf_digests  # type: ignore[arg-type]
        ),
        row_receipts=row_receipts,  # type: ignore[arg-type]
        row_authority_manifest=row_authority_manifest,  # type: ignore[arg-type]
        row_authority_manifest_sha256=(
            row_authority_manifest_sha256  # type: ignore[arg-type]
        ),
        expected_row_authority_manifest_sha256=(
            expected_row_authority_manifest_sha256  # type: ignore[arg-type]
        ),
        selection_policy=selection_policy,  # type: ignore[arg-type]
        expected_gradient_closure_sha256=(
            expected_gradient_closure_sha256  # type: ignore[arg-type]
        ),
        input_pins_sha256=input_pins_sha256,  # type: ignore[arg-type]
    )
    _validate_selection_manifest_structure(value)
    if canonical_json_bytes(value) != canonical_json_bytes(rebuilt):
        raise MotiveSelectorError(
            "selection manifest does not recompute from original pinned inputs"
        )


def _reject_float(_: str) -> None:
    raise MotiveSelectorError("JSON floating-point numbers are forbidden")


def _object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MotiveSelectorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise MotiveSelectorError(f"{label} is not ASCII JSON") from error
    try:
        value = json.loads(
            text,
            parse_float=_reject_float,
            parse_constant=_reject_float,
            object_pairs_hook=_object_pairs,
        )
    except MotiveSelectorError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise MotiveSelectorError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise MotiveSelectorError(f"{label} root must be an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise MotiveSelectorError(f"{label} is not canonical one-newline JSON")
    return value


def _canonical_plain_path(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise MotiveSelectorError(f"{label} path differs")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise MotiveSelectorError(f"{label} path must be canonical absolute")
    return path


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_pinned_json(
    path_value: Any, expected_sha256: Any, *, label: str
) -> tuple[dict[str, Any], Path, str, tuple[int, int]]:
    expected = _sha256(expected_sha256, label=f"{label} expected SHA-256")
    path = _canonical_plain_path(path_value, label=label)
    try:
        before = path.lstat()
    except OSError as error:
        raise MotiveSelectorError(f"cannot stat {label}: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON_BYTES:
        raise MotiveSelectorError(f"{label} is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MotiveSelectorError(f"cannot open {label}: {error}") from error
    try:
        fd_before = os.fstat(descriptor)
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > _MAX_JSON_BYTES:
                raise MotiveSelectorError(f"{label} exceeds read size limit")
            chunks.append(block)
        fd_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise MotiveSelectorError(f"cannot restat {label}: {error}") from error
    raw = b"".join(chunks)
    observed = hashlib.sha256(raw).hexdigest()
    identity = _stat_identity(before)
    if (
        identity != _stat_identity(fd_before)
        or identity != _stat_identity(fd_after)
        or identity != _stat_identity(after)
        or len(raw) != before.st_size
        or observed != expected
    ):
        raise MotiveSelectorError(f"{label} changed during read or SHA-256 differs")
    return (
        _decode_canonical_json(raw, label=label),
        path,
        observed,
        (before.st_dev, before.st_ino),
    )


def _validate_pin(value: Any, *, label: str) -> dict[str, str]:
    pin = _exact_mapping(value, {"path", "sha256"}, label=label)
    path = _canonical_plain_path(pin["path"], label=label)
    digest = _sha256(pin["sha256"], label=f"{label} SHA-256")
    return {"path": str(path), "sha256": digest}


def _validate_anchor_group_leaf_pin(value: Any, *, label: str) -> dict[str, str]:
    pin = _exact_mapping(
        value,
        {"counterfactual_group_id", "path", "sha256", "expected_leaf_digest"},
        label=label,
    )
    base = _validate_pin(
        {"path": pin["path"], "sha256": pin["sha256"]}, label=label
    )
    return {
        "counterfactual_group_id": _identifier(
            pin["counterfactual_group_id"], label=f"{label} group id"
        ),
        "path": base["path"],
        "sha256": base["sha256"],
        "expected_leaf_digest": _sha256(
            pin["expected_leaf_digest"], label=f"{label} expected leaf digest"
        ),
    }


def validate_input_pinset(
    value: Any, *, expected_row_authority_manifest_sha256: str
) -> dict[str, Any]:
    pinset = _exact_mapping(
        value,
        {
            "schema_version",
            "expected_gradient_closure_sha256",
            "expected_full644_dataset_summary_sha256",
            "expected_full644_dataset_index_sha256",
            "expected_full644_source_authority_sha256",
            "expected_official_row_authority_manifest_sha256",
            "row_authority_manifest",
            "selection_policy",
            "anchor_receipts",
            "anchor_group_leaf_receipts",
            "row_receipts",
            "pinset_digest",
        },
        label="input pinset",
    )
    _validate_digest(pinset, digest_field="pinset_digest", label="input pinset")
    if (
        pinset["schema_version"] != INPUT_PINS_SCHEMA
        or pinset["expected_full644_dataset_summary_sha256"]
        != FULL644_DATASET_SUMMARY_SHA256
        or pinset["expected_full644_dataset_index_sha256"]
        != FULL644_DATASET_INDEX_SHA256
        or pinset["expected_full644_source_authority_sha256"]
        != FULL644_SOURCE_AUTHORITY_SHA256
    ):
        raise MotiveSelectorError("input pinset full644 root binding differs")
    closure_sha = _sha256(
        pinset["expected_gradient_closure_sha256"],
        label="pinset expected gradient closure",
    )
    authority_pin = _validate_pin(
        pinset["row_authority_manifest"], label="row authority manifest pin"
    )
    caller_authority_sha = _authority_sha256(
        expected_row_authority_manifest_sha256,
        label="caller expected official row authority manifest",
    )
    embedded_authority_sha = _authority_sha256(
        pinset["expected_official_row_authority_manifest_sha256"],
        label="pinset expected official row authority manifest",
    )
    if (
        caller_authority_sha != embedded_authority_sha
        or authority_pin["sha256"] != caller_authority_sha
    ):
        raise MotiveSelectorError(
            "pinset row authority differs from caller-pinned official file SHA-256"
        )
    policy = validate_selection_policy(pinset["selection_policy"])
    if (
        not isinstance(pinset["anchor_receipts"], list)
        or not isinstance(pinset["anchor_group_leaf_receipts"], list)
        or not isinstance(pinset["row_receipts"], list)
    ):
        raise MotiveSelectorError("pinset receipt pins must be lists")
    if not 1 <= len(pinset["anchor_receipts"]) <= _MAX_ANCHOR_RECEIPTS:
        raise MotiveSelectorError("pinset anchor receipt count differs")
    if len(pinset["row_receipts"]) != EXPECTED_ROW_COUNT:
        raise MotiveSelectorError("pinset row receipt count is not exact644")
    anchor_pins = [
        _validate_pin(item, label=f"anchor receipt pin[{index}]")
        for index, item in enumerate(pinset["anchor_receipts"])
    ]
    group_leaf_pins = [
        _validate_anchor_group_leaf_pin(
            item, label=f"anchor group leaf pin[{index}]"
        )
        for index, item in enumerate(pinset["anchor_group_leaf_receipts"])
    ]
    if not group_leaf_pins or len(group_leaf_pins) * len(QUERY_KINDS) != len(
        anchor_pins
    ):
        raise MotiveSelectorError("pinset anchor group leaf count differs")
    row_pins = [
        _validate_pin(item, label=f"row receipt pin[{index}]")
        for index, item in enumerate(pinset["row_receipts"])
    ]
    if anchor_pins != sorted(anchor_pins, key=lambda item: item["path"]):
        raise MotiveSelectorError("anchor receipt pins are not path-sorted")
    if group_leaf_pins != sorted(
        group_leaf_pins, key=lambda item: item["counterfactual_group_id"]
    ):
        raise MotiveSelectorError("anchor group leaf pins are not group-sorted")
    if row_pins != sorted(row_pins, key=lambda item: item["path"]):
        raise MotiveSelectorError("row receipt pins are not path-sorted")
    group_ids = [item["counterfactual_group_id"] for item in group_leaf_pins]
    group_leaf_digests = [item["expected_leaf_digest"] for item in group_leaf_pins]
    if (
        len(set(group_ids)) != len(group_ids)
        or len(set(group_leaf_digests)) != len(group_leaf_digests)
    ):
        raise MotiveSelectorError("anchor group leaf pin identity repeats")
    all_paths = [authority_pin["path"]] + [
        item["path"] for item in anchor_pins + group_leaf_pins + row_pins
    ]
    if len(set(all_paths)) != len(all_paths):
        raise MotiveSelectorError("input pin paths repeat")
    return {
        "expected_gradient_closure_sha256": closure_sha,
        "expected_official_row_authority_manifest_sha256": caller_authority_sha,
        "row_authority_manifest": authority_pin,
        "selection_policy": policy,
        "anchor_receipts": anchor_pins,
        "anchor_group_leaf_receipts": group_leaf_pins,
        "row_receipts": row_pins,
    }


def _publish_create_only(path_value: str, value: Mapping[str, Any]) -> str:
    output = _canonical_plain_path(path_value, label="output")
    parent = output.parent
    try:
        parent_stat = parent.lstat()
    except OSError as error:
        raise MotiveSelectorError(f"cannot stat output directory: {error}") from error
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise MotiveSelectorError("output parent is not a directory")
    if output.exists() or output.is_symlink():
        raise MotiveSelectorError("output already exists; create-only publication refused")
    raw = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".staged", dir=parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output, follow_symlinks=False)
        linked = True
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise MotiveSelectorError(
            "output already exists; create-only publication refused"
        ) from error
    except OSError as error:
        if linked and output.exists():
            # Publication succeeded but a later durability check failed.  Never
            # overwrite or delete the newly published user-visible artifact.
            raise MotiveSelectorError(
                f"output published but durability verification failed: {error}"
            ) from error
        raise MotiveSelectorError(f"create-only publication failed: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    observed = output.read_bytes()
    mode = output.stat().st_mode & 0o777
    if observed != raw or mode != 0o444 or output.stat().st_nlink != 1:
        raise MotiveSelectorError("published manifest byte/mode/link verification failed")
    return hashlib.sha256(raw).hexdigest()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pins", required=True)
    parser.add_argument("--expected-input-pins-sha256", required=True)
    parser.add_argument("--expected-row-authority-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    pinset_value, _, pinset_sha, pinset_identity = _read_pinned_json(
        arguments.input_pins,
        arguments.expected_input_pins_sha256,
        label="input pinset",
    )
    pinset = validate_input_pinset(
        pinset_value,
        expected_row_authority_manifest_sha256=(
            arguments.expected_row_authority_manifest_sha256
        ),
    )
    authority_pin = pinset["row_authority_manifest"]
    authority_value, _, authority_sha, authority_identity = _read_pinned_json(
        authority_pin["path"],
        authority_pin["sha256"],
        label="row authority manifest",
    )
    validate_row_authority_manifest(
        authority_value,
        expected_manifest_sha256=pinset[
            "expected_official_row_authority_manifest_sha256"
        ],
    )

    observed_identities = {pinset_identity, authority_identity}
    anchors: list[tuple[Mapping[str, Any], str]] = []
    for index, pin in enumerate(pinset["anchor_receipts"]):
        value, _, digest, identity = _read_pinned_json(
            pin["path"], pin["sha256"], label=f"anchor receipt[{index}]"
        )
        if identity in observed_identities:
            raise MotiveSelectorError("input files alias the same inode")
        observed_identities.add(identity)
        anchors.append((value, digest))
    group_leaves: list[tuple[Mapping[str, Any], str]] = []
    expected_group_leaf_digests: dict[str, str] = {}
    for index, pin in enumerate(pinset["anchor_group_leaf_receipts"]):
        value, _, digest, identity = _read_pinned_json(
            pin["path"], pin["sha256"], label=f"anchor group leaf[{index}]"
        )
        if identity in observed_identities:
            raise MotiveSelectorError("input files alias the same inode")
        observed_identities.add(identity)
        group_leaves.append((value, digest))
        expected_group_leaf_digests[pin["counterfactual_group_id"]] = pin[
            "expected_leaf_digest"
        ]
    rows: list[tuple[Mapping[str, Any], str]] = []
    for index, pin in enumerate(pinset["row_receipts"]):
        value, _, digest, identity = _read_pinned_json(
            pin["path"], pin["sha256"], label=f"row receipt[{index}]"
        )
        if identity in observed_identities:
            raise MotiveSelectorError("input files alias the same inode")
        observed_identities.add(identity)
        rows.append((value, digest))

    manifest = build_selection_manifest(
        anchor_receipts=anchors,
        anchor_group_leaf_receipts=group_leaves,
        expected_anchor_group_leaf_digests=expected_group_leaf_digests,
        row_receipts=rows,
        row_authority_manifest=authority_value,
        row_authority_manifest_sha256=authority_sha,
        expected_row_authority_manifest_sha256=pinset[
            "expected_official_row_authority_manifest_sha256"
        ],
        selection_policy=pinset["selection_policy"],
        expected_gradient_closure_sha256=pinset[
            "expected_gradient_closure_sha256"
        ],
        input_pins_sha256=pinset_sha,
    )
    output_sha = _publish_create_only(arguments.output, manifest)
    print(
        json.dumps(
            {
                "output": arguments.output,
                "output_sha256": output_sha,
                "selected_rows": manifest["counts"]["selected_rows"],
                "training_authorized": False,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
