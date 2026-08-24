#!/usr/bin/env python3
"""Independent, local-only qualification ledger for action teacher codes.

This module is deliberately a review artifact, not a trainer.  It consumes an
externally pinned, content-disjoint D0 benchmark containing the *actual* FP32
``q`` payloads and the unchanged ``candidate_unqualified`` materialization
receipts emitted by :mod:`action_feature_teacher_v1`.  It recomputes every
pre-registered metric, and only then derives the exact external qualification
authority/leaf shapes consumed by :mod:`action_anchor_distillation_v1`.

Classification is a separate second stage.  It runs only after qualified
``q_y``/``q_anchor`` receipts exist, binds those exact receipt digests plus
both endpoints and materializations, and emits a downstream compatibility
receipt candidate.  The candidate still has to be independently pinned by a
frozen row/launch ledger before the distillation contract will consume it.

No function here launches training, touches a network or GPU, decodes video,
or claims that the repository's current teacher has passed a real benchmark.
Synthetic fixtures are accepted only by validation/diagnostic APIs when the
caller opts in explicitly; they can never reach either issuance API.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping, Sequence
import unicodedata
import zlib

if __package__:
    from . import action_anchor_distillation_v1 as distill
else:  # Direct import from methods/bernini_action_editing.
    import action_anchor_distillation_v1 as distill  # type: ignore[no-redef]


BENCHMARK_PAYLOAD_SCHEMA = "bernini-action-teacher-d0-benchmark-v1"
BENCHMARK_AUTHORITY_SCHEMA = (
    "bernini-action-teacher-d0-benchmark-authority-v1"
)
PROTOCOL_SCHEMA = "bernini-action-teacher-qualification-protocol-v1"
SPLIT_MANIFEST_SCHEMA = "bernini-action-teacher-d0-split-manifest-v1"
Q_TENSOR_PAYLOAD_SCHEMA = "bernini-action-plan-fp32-zlib-payload-v1"
CANDIDATE_ITEM_SCHEMA = "bernini-action-teacher-candidate-item-v1"
CASE_SCHEMA = "bernini-action-teacher-d0-case-v1"
CLASSIFICATION_REQUEST_SCHEMA = (
    "bernini-action-teacher-classification-request-v1"
)
METRICS_SCHEMA = "bernini-action-teacher-qualification-metrics-v1"
SYNTHETIC_DIAGNOSTIC_SCHEMA = (
    "bernini-action-teacher-synthetic-qualification-diagnostic-v1"
)
QUALIFICATION_BUNDLE_SCHEMA = (
    "bernini-action-teacher-qualification-bundle-v1"
)
CLASSIFICATION_AUTHORITY_SCHEMA = (
    "bernini-action-teacher-classification-authority-v1"
)
CLASSIFICATION_DECISION_SCHEMA = (
    "bernini-action-teacher-classification-decision-v1"
)
CLASSIFICATION_LEDGER_SCHEMA = (
    "bernini-action-teacher-classification-ledger-v1"
)

LOCAL_ONLY = True
TRAINING_AUTHORIZED = False
DECODED_VIDEO_SCIENTIFIC_GATE_IMPLEMENTED = False

CASE_COUNT = 32
ACTION_CLASS_COUNT = 8
CASES_PER_ACTION_CLASS = 4
PHASE_COUNT = distill.PHASE_COUNT
ACTION_WIDTH = distill.ACTION_WIDTH
Q_VECTOR_WIDTH = PHASE_COUNT * ACTION_WIDTH + ACTION_WIDTH

# The public classification vocabulary uses underscores.  The existing
# distillation ABI uses hyphens for two values, so the only conversion is
# explicit and closed below.
CLASSIFICATION_KINDS = (
    "compatible",
    "noop",
    "reverse",
    "incomplete",
    "wrong_actor",
    "wrong_object",
    "camera",
    "appearance",
)
HARD_NEGATIVE_KINDS = CLASSIFICATION_KINDS[1:]
CLASSIFICATION_VERDICTS = ("positive", "negative", "excluded")
ITEM_EVIDENCE_STATUSES = ("eligible", "unqualified")
Q_KINDS = ("q_y", "q_anchor")

_DISTILL_KIND = {
    "compatible": "compatible",
    "noop": "noop",
    "reverse": "reverse",
    "incomplete": "incomplete",
    "wrong_actor": "wrong-actor",
    "wrong_object": "wrong-object",
    "camera": "camera",
    "appearance": "appearance",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ZERO_SHA256 = "0" * 64
_MAX_JSON_BYTES = 256 * 1024 * 1024

_PROTOCOL_FIELDS = {
    "schema_version",
    "case_count",
    "action_class_count",
    "cases_per_action_class",
    "hard_negative_kinds",
    "action_similarity",
    "appearance_similarity",
    "hard_negative_auroc_min",
    "hard_negative_lower_bound_method",
    "hard_negative_lower_bound_strict_min",
    "clean_control_median_margin_min",
    "clean_control_pair_wins_min",
    "cross_content_recall_at_1_min",
    "appearance_correlation_abs_max",
    "effective_rank_definition",
    "effective_rank_min",
    "metric_rounding_significant_digits",
}
_SPLIT_FIELDS = {
    "schema_version",
    "d0_case_ids",
    "d0_content_ids",
    "d0_generator_ids",
    "d0_actor_scene_ids",
    "development_content_ids",
    "development_generator_ids",
    "development_actor_scene_ids",
    "content_disjoint_holdout",
    "generator_disjoint_holdout",
    "actor_scene_disjoint_holdout",
    "split_digest",
}
_TENSOR_FIELDS = {
    "schema_version",
    "canonical_dtype",
    "phase_shape",
    "global_shape",
    "phase_f32le_zlib_b64",
    "global_f32le_zlib_b64",
    "phase_raw_sha256",
    "global_raw_sha256",
    "payload_digest",
}
_ITEM_FIELDS = {
    "schema_version",
    "item_id",
    "q_kind",
    "materialization_role",
    "row_id",
    "source_sha256",
    "instruction_sha256",
    "endpoint_sha256",
    "semantics",
    "content_id",
    "generator_id",
    "actor_scene_id",
    "source_media_sha256",
    "endpoint_media_sha256",
    "media_provenance_sha256",
    "media_producer_sha256",
    "item_evidence_status",
    "materialization_receipt",
    "q_tensor",
    "candidate_receipt_digest",
}
_CASE_FIELDS = {
    "schema_version",
    "case_id",
    "split",
    "action_class",
    "query_item_id",
    "clean_item_id",
    "hard_negative_item_ids",
    "dino_query_embedding",
    "dino_clean_embedding",
}
_REQUEST_FIELDS = {
    "schema_version",
    "decision_id",
    "case_id",
    "q_y_item_id",
    "q_anchor_item_id",
    "candidate_kind",
    "candidate_status",
    "requested_verdict",
}
_PAYLOAD_FIELDS = {
    "schema_version",
    "benchmark_id",
    "synthetic_fixture",
    "protocol",
    "split_manifest",
    "q_items",
    "cases",
    "classification_requests",
    "payload_digest",
}
_BENCHMARK_AUTHORITY_FIELDS = {
    "schema_version",
    "benchmark_payload_sha256",
    "official_row_authority_sha256",
    "teacher_producer_sha256",
    "upstream_authority_manifest_sha256",
    "qualification_evaluator_sha256",
    "dino_model_sha256",
    "protocol_sha256",
    "split_manifest_sha256",
    "classification_request_manifest_sha256",
    "synthetic_fixture",
    "production_authority",
    "independent_evaluator",
    "content_disjoint_holdout",
    "no_training_authority",
    "authority_digest",
}
_METRIC_FIELDS = {
    "schema_version",
    "case_count",
    "action_class_count",
    "hard_negative_auroc",
    "hard_negative_leave_one_case_out_min_auroc",
    "clean_control_median_margin",
    "clean_control_pair_wins",
    "cross_content_recall_at_1",
    "appearance_action_similarity_abs_pearson_correlation",
    "effective_rank",
    "all_global_gates_pass",
    "metrics_digest",
}
_QUALIFICATION_ITEM_FIELDS = {
    "item_id",
    "q_kind",
    "candidate_receipt_digest",
    "qualification_receipt",
}
_QUALIFICATION_BUNDLE_FIELDS = {
    "schema_version",
    "benchmark_payload_sha256",
    "benchmark_authority_sha256",
    "metrics",
    "teacher_authority",
    "qualification_items",
    "independent_benchmark_overrides_candidate_unqualified",
    "scientific_scope",
    "decoded_video_gate_authorized",
    "training_authorized",
    "local_only",
    "bundle_digest",
}
_SYNTHETIC_DIAGNOSTIC_FIELDS = {
    "schema_version",
    "benchmark_payload_sha256",
    "benchmark_authority_sha256",
    "metrics",
    "synthetic_fixture",
    "distillation_authority_emitted",
    "qualification_leaves_emitted",
    "compatibility_receipts_emitted",
    "training_authorized",
    "local_only",
    "diagnostic_digest",
}
_CLASSIFICATION_AUTHORITY_FIELDS = {
    "schema_version",
    "benchmark_authority_sha256",
    "qualification_bundle_sha256",
    "teacher_authority_sha256",
    "classification_request_manifest_sha256",
    "classification_protocol_sha256",
    "classification_evaluator_sha256",
    "independent_evaluator",
    "decoded_video_gate_authorized",
    "training_authorized",
    "local_only",
    "authority_digest",
}
_RECEIPT_PAIR_FIELDS = {
    "decision_id",
    "q_y_receipt",
    "q_anchor_receipt",
}
_DECISION_FIELDS = {
    "schema_version",
    "decision_id",
    "row_id",
    "q_y_item_id",
    "q_anchor_item_id",
    "q_y_receipt_digest",
    "q_anchor_receipt_digest",
    "q_y_qualification_receipt_digest",
    "q_anchor_qualification_receipt_digest",
    "q_y_endpoint_sha256",
    "q_anchor_endpoint_sha256",
    "q_y_materialization_receipt_sha256",
    "q_anchor_materialization_receipt_sha256",
    "candidate_kind",
    "candidate_status",
    "verdict",
    "desired_semantics_sha256",
    "candidate_semantics_sha256",
    "axis_matches",
    "mismatch_axes",
    "classification_evaluator_sha256",
    "classification_protocol_sha256",
    "classification_authority_sha256",
    "compatibility_receipt_digest",
    "training_use",
    "contrastive_role",
    "training_authorized",
    "decision_digest",
}
_CLASSIFICATION_LEDGER_FIELDS = {
    "schema_version",
    "benchmark_authority_sha256",
    "qualification_bundle_sha256",
    "classification_authority",
    "decisions",
    "compatibility_receipts",
    "expected_external_pins_required_before_consumption",
    "training_authorized",
    "local_only",
    "ledger_digest",
}


class ActionTeacherQualificationError(RuntimeError):
    """Raised before unpinned or under-qualified evidence can be signed."""


def _fail(message: str) -> None:
    raise ActionTeacherQualificationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ActionTeacherQualificationError(
            f"value is not canonical finite JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _closed_dict(value: Any, fields: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact dict")
    if set(value) != fields:
        _fail(
            f"{label} field closure differs: "
            f"missing={sorted(fields - set(value))} "
            f"extra={sorted(set(value) - fields)}"
        )
    return value


def _exact_list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be an exact list")
    return value


def _sha256(value: Any, *, label: str, authority: bool = True) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    if authority and value == _ZERO_SHA256:
        _fail(f"{label} may not be an all-zero placeholder")
    return value


def _exact_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be an exact boolean")
    return value


def _exact_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        _fail(f"{label} must be an exact integer in range")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{label} must be an exact finite float")
    return value


def _text(value: Any, *, label: str, maximum_bytes: int = 1024) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > maximum_bytes
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        _fail(f"{label} must be non-empty canonical text")
    return value


def _unique_sha_list(value: Any, *, label: str) -> list[str]:
    values = _exact_list(value, label=label)
    normalized = [_sha256(item, label=f"{label}[{index}]") for index, item in enumerate(values)]
    if len(set(normalized)) != len(normalized):
        _fail(f"{label} contains duplicates")
    return normalized


def _metric_float(value: float) -> float:
    """Canonicalize calculated metrics to 12 significant decimal digits."""

    if not math.isfinite(value):
        _fail("calculated metric is non-finite")
    return float(format(value, ".12g"))


def qualification_protocol_v1() -> dict[str, Any]:
    """Return the exact pre-registered metric and threshold contract."""

    return {
        "schema_version": PROTOCOL_SCHEMA,
        "case_count": CASE_COUNT,
        "action_class_count": ACTION_CLASS_COUNT,
        "cases_per_action_class": CASES_PER_ACTION_CLASS,
        "hard_negative_kinds": list(HARD_NEGATIVE_KINDS),
        "action_similarity": "cosine(flatten(phase_21x256)||global_256)",
        "appearance_similarity": "cosine(pinned_dino_query,pinned_dino_clean)",
        "hard_negative_auroc_min": 0.80,
        "hard_negative_lower_bound_method": (
            "minimum-AUROC-over-32-deterministic-leave-one-case-out-folds"
        ),
        "hard_negative_lower_bound_strict_min": 0.65,
        "clean_control_median_margin_min": 0.10,
        "clean_control_pair_wins_min": 24,
        "cross_content_recall_at_1_min": 0.50,
        "appearance_correlation_abs_max": 0.20,
        "effective_rank_definition": (
            "exp(shannon-entropy(singular-values-of-centered-32xD-query-matrix))"
        ),
        "effective_rank_min": 8.0,
        "metric_rounding_significant_digits": 12,
    }


QUALIFICATION_PROTOCOL_SHA256 = object_sha256(qualification_protocol_v1())


def _classification_protocol_body() -> dict[str, Any]:
    return {
        "schema_version": "bernini-action-teacher-classification-policy-v1",
        "candidate_kinds": list(CLASSIFICATION_KINDS),
        "hard_negative_kinds": list(HARD_NEGATIVE_KINDS),
        "candidate_statuses": list(ITEM_EVIDENCE_STATUSES),
        "verdicts": list(CLASSIFICATION_VERDICTS),
        "compatible_route": "positive/contrastive-only",
        "hard_negative_route": "negative/contrastive-only",
        "unqualified_route": "excluded",
        "q_anchor_point_distillation_authorized": False,
        "downstream_kind_mapping": dict(_DISTILL_KIND),
        "downstream_qualification_verdicts": {
            "positive": "accept",
            "negative": "accept",
            "excluded": "abstain",
        },
        "training_authorized": False,
    }


CLASSIFICATION_PROTOCOL_SHA256 = object_sha256(_classification_protocol_body())


def _validate_protocol(value: Any) -> dict[str, Any]:
    protocol = _closed_dict(value, _PROTOCOL_FIELDS, label="qualification protocol")
    for name in (
        "case_count",
        "action_class_count",
        "cases_per_action_class",
        "clean_control_pair_wins_min",
        "metric_rounding_significant_digits",
    ):
        _exact_int(protocol[name], label=f"qualification protocol {name}", minimum=1)
    for name in (
        "hard_negative_auroc_min",
        "hard_negative_lower_bound_strict_min",
        "clean_control_median_margin_min",
        "cross_content_recall_at_1_min",
        "appearance_correlation_abs_max",
        "effective_rank_min",
    ):
        _finite_float(protocol[name], label=f"qualification protocol {name}")
    kinds = _exact_list(
        protocol["hard_negative_kinds"],
        label="qualification protocol hard-negative kinds",
    )
    if any(type(item) is not str for item in kinds):
        _fail("qualification protocol hard-negative kinds must be strings")
    for name in (
        "schema_version",
        "action_similarity",
        "appearance_similarity",
        "hard_negative_lower_bound_method",
        "effective_rank_definition",
    ):
        _text(protocol[name], label=f"qualification protocol {name}")
    # Exact-object equality rejects threshold, method, ordering and vocabulary drift.
    if canonical_json_bytes(protocol) != canonical_json_bytes(
        qualification_protocol_v1()
    ):
        _fail("qualification protocol differs from the pre-registered protocol")
    return json.loads(canonical_json_bytes(protocol).decode("ascii"))


def _decode_zlib_f32(
    encoded: Any, *, count: int, label: str
) -> tuple[bytes, list[float]]:
    if type(encoded) is not str or not encoded:
        _fail(f"{label} must be non-empty canonical base64")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ActionTeacherQualificationError(f"{label} base64 differs") from error
    if base64.b64encode(compressed).decode("ascii") != encoded:
        _fail(f"{label} base64 is not canonical")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(compressed, count * 4 + 1)
    except zlib.error as error:
        raise ActionTeacherQualificationError(f"{label} zlib differs") from error
    if (
        len(raw) != count * 4
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        _fail(f"{label} decompressed length/stream closure differs")
    try:
        trailing = decoder.flush()
    except zlib.error as error:
        raise ActionTeacherQualificationError(f"{label} zlib flush differs") from error
    if trailing:
        _fail(f"{label} has trailing decompressed bytes")
    # There is one canonical compression representation for a tensor payload.
    if zlib.compress(raw, level=9) != compressed:
        _fail(f"{label} compression is not canonical level-9 zlib")
    values = list(struct.unpack(f"<{count}f", raw))
    if any(not math.isfinite(item) for item in values):
        _fail(f"{label} contains a non-finite FP32 value")
    return raw, values


def encode_q_tensor_payload_v1(
    phase_values: Sequence[float], global_values: Sequence[float]
) -> dict[str, Any]:
    """Encode one actual 21x256 + 256 FP32 payload for a frozen fixture.

    This helper is deterministic and useful when an independent evaluator
    materializes its evidence file.  Values must be exact Python floats; bools
    and integers are intentionally refused.
    """

    if type(phase_values) not in (list, tuple) or type(global_values) not in (list, tuple):
        _fail("q tensor inputs must be exact lists or tuples")
    if len(phase_values) != PHASE_COUNT * ACTION_WIDTH or len(global_values) != ACTION_WIDTH:
        _fail("q tensor input shapes differ")
    for index, value in enumerate(tuple(phase_values) + tuple(global_values)):
        _finite_float(value, label=f"q tensor value[{index}]")
    phase_raw = struct.pack(f"<{len(phase_values)}f", *phase_values)
    global_raw = struct.pack(f"<{len(global_values)}f", *global_values)
    unsigned = {
        "schema_version": Q_TENSOR_PAYLOAD_SCHEMA,
        "canonical_dtype": "float32-little-endian",
        "phase_shape": [PHASE_COUNT, ACTION_WIDTH],
        "global_shape": [ACTION_WIDTH],
        "phase_f32le_zlib_b64": base64.b64encode(
            zlib.compress(phase_raw, level=9)
        ).decode("ascii"),
        "global_f32le_zlib_b64": base64.b64encode(
            zlib.compress(global_raw, level=9)
        ).decode("ascii"),
        "phase_raw_sha256": hashlib.sha256(phase_raw).hexdigest(),
        "global_raw_sha256": hashlib.sha256(global_raw).hexdigest(),
    }
    return {**unsigned, "payload_digest": object_sha256(unsigned)}


def _validate_q_tensor(value: Any) -> tuple[dict[str, Any], list[float]]:
    payload = _closed_dict(value, _TENSOR_FIELDS, label="q tensor payload")
    if (
        payload["schema_version"] != Q_TENSOR_PAYLOAD_SCHEMA
        or payload["canonical_dtype"] != "float32-little-endian"
        or type(payload["phase_shape"]) is not list
        or payload["phase_shape"] != [PHASE_COUNT, ACTION_WIDTH]
        or any(type(item) is not int for item in payload["phase_shape"])
        or type(payload["global_shape"]) is not list
        or payload["global_shape"] != [ACTION_WIDTH]
        or any(type(item) is not int for item in payload["global_shape"])
    ):
        _fail("q tensor schema/dtype/shape differs")
    phase_raw, phase = _decode_zlib_f32(
        payload["phase_f32le_zlib_b64"],
        count=PHASE_COUNT * ACTION_WIDTH,
        label="q phase payload",
    )
    global_raw, global_values = _decode_zlib_f32(
        payload["global_f32le_zlib_b64"],
        count=ACTION_WIDTH,
        label="q global payload",
    )
    if (
        _sha256(payload["phase_raw_sha256"], label="q phase raw SHA-256", authority=False)
        != hashlib.sha256(phase_raw).hexdigest()
        or _sha256(payload["global_raw_sha256"], label="q global raw SHA-256", authority=False)
        != hashlib.sha256(global_raw).hexdigest()
    ):
        _fail("q tensor raw digest differs")
    unsigned = dict(payload)
    declared = _sha256(unsigned.pop("payload_digest"), label="q tensor payload digest")
    if object_sha256(unsigned) != declared:
        _fail("q tensor payload digest differs")
    return json.loads(canonical_json_bytes(payload).decode("ascii")), phase + global_values


def _validate_item(value: Any) -> tuple[dict[str, Any], list[float]]:
    item = _closed_dict(value, _ITEM_FIELDS, label="candidate q item")
    if item["schema_version"] != CANDIDATE_ITEM_SCHEMA:
        _fail("candidate q item schema differs")
    for name in (
        "item_id",
        "row_id",
        "source_sha256",
        "instruction_sha256",
        "endpoint_sha256",
        "content_id",
        "generator_id",
        "actor_scene_id",
        "source_media_sha256",
        "endpoint_media_sha256",
        "media_provenance_sha256",
        "media_producer_sha256",
    ):
        _sha256(item[name], label=f"candidate item {name}")
    q_kind = item["q_kind"]
    if type(q_kind) is not str or q_kind not in Q_KINDS:
        _fail("candidate item q kind differs")
    expected_role = "target" if q_kind == "q_y" else "anchor"
    if item["materialization_role"] != expected_role:
        _fail("candidate item materialization role differs")
    if item["endpoint_sha256"] != item["endpoint_media_sha256"]:
        _fail("candidate endpoint/media binding differs")
    semantics = distill.validate_action_semantics(item["semantics"])
    status = item["item_evidence_status"]
    if type(status) is not str or status not in ITEM_EVIDENCE_STATUSES:
        _fail("candidate item evidence status differs")
    try:
        materialization = distill._validate_materialization_receipt(  # type: ignore[attr-defined]
            item["materialization_receipt"], q_kind=q_kind
        )
    except distill.ActionAnchorDistillationError as error:
        raise ActionTeacherQualificationError(
            f"candidate materialization differs: {error}"
        ) from error
    # Upstream receipts remain candidates.  Independent evidence below is the
    # only thing allowed to qualify them; the receipt is never rewritten.
    if (
        materialization["teacher_qualification_status"] != "candidate_unqualified"
        or materialization["point_distillation_authorized"] is not False
    ):
        _fail("candidate materialization attempted to self-qualify")
    tensor, vector = _validate_q_tensor(item["q_tensor"])
    if (
        tensor["phase_raw_sha256"] != materialization["phase_tokens_sha256"]
        or tensor["global_raw_sha256"] != materialization["global_token_sha256"]
    ):
        _fail("actual q tensor differs from candidate materialization")
    normalized_unsigned = {
        **{name: item[name] for name in _ITEM_FIELDS if name != "candidate_receipt_digest"},
        "semantics": semantics,
        "materialization_receipt": materialization,
        "q_tensor": tensor,
    }
    declared = _sha256(
        item["candidate_receipt_digest"], label="candidate item receipt digest"
    )
    if object_sha256(normalized_unsigned) != declared:
        _fail("candidate q item receipt digest differs")
    normalized = {**normalized_unsigned, "candidate_receipt_digest": declared}
    return json.loads(canonical_json_bytes(normalized).decode("ascii")), vector


def _semantic_relation(
    desired: Mapping[str, str], candidate: Mapping[str, str], kind: str
) -> tuple[dict[str, bool], list[str]]:
    matches = {axis: desired[axis] == candidate[axis] for axis in distill.SEMANTIC_AXES}
    mismatch = [axis for axis in distill.SEMANTIC_AXES if not matches[axis]]
    expected = {
        "compatible": [],
        "noop": ["action"],
        "reverse": ["direction"],
        "incomplete": ["outcome"],
        "wrong_actor": ["actor"],
        "wrong_object": ["object"],
        "camera": [],
        "appearance": [],
    }[kind]
    if mismatch != expected:
        _fail(f"{kind} candidate semantic mismatch axes differ")
    if kind == "noop" and candidate["action"] != "noop":
        _fail("noop candidate must use action=noop")
    if kind == "incomplete" and candidate["outcome"] != "incomplete":
        _fail("incomplete candidate must use outcome=incomplete")
    if kind == "noop" and desired["action"] == "noop":
        _fail("noop target cannot have a noop hard negative")
    return matches, mismatch


def _validate_case(
    value: Any,
    *,
    items: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case = _closed_dict(value, _CASE_FIELDS, label="D0 benchmark case")
    if case["schema_version"] != CASE_SCHEMA or case["split"] != "d0_holdout":
        _fail("D0 benchmark case schema/split differs")
    _sha256(case["case_id"], label="D0 case ID")
    _text(case["action_class"], label="D0 action class")
    query_id = _sha256(case["query_item_id"], label="D0 query item ID")
    clean_id = _sha256(case["clean_item_id"], label="D0 clean item ID")
    negatives = _closed_dict(
        case["hard_negative_item_ids"],
        set(HARD_NEGATIVE_KINDS),
        label="D0 hard-negative item IDs",
    )
    negative_ids = [
        _sha256(negatives[kind], label=f"D0 {kind} item ID")
        for kind in HARD_NEGATIVE_KINDS
    ]
    referenced = [query_id, clean_id] + negative_ids
    if len(set(referenced)) != len(referenced):
        _fail("D0 case contains duplicate item references")
    if any(item_id not in items for item_id in referenced):
        _fail("D0 case references an unknown candidate item")
    query = items[query_id]
    clean = items[clean_id]
    if query["q_kind"] != "q_y" or clean["q_kind"] != "q_anchor":
        _fail("D0 query/clean q roles differ")
    if query["item_evidence_status"] != "eligible":
        _fail("D0 q_y query must carry eligible evidence")
    if case["action_class"] != query["semantics"]["action"]:
        _fail("D0 action-class label differs from q_y action semantics")
    for name in ("row_id", "source_sha256", "instruction_sha256"):
        if query[name] != clean[name]:
            _fail(f"D0 compatible pair {name} binding differs")
    if query["endpoint_sha256"] == clean["endpoint_sha256"]:
        _fail("D0 compatible endpoint must differ from target endpoint")
    _semantic_relation(query["semantics"], clean["semantics"], "compatible")
    for kind, item_id in zip(HARD_NEGATIVE_KINDS, negative_ids):
        candidate = items[item_id]
        if candidate["q_kind"] != "q_anchor":
            _fail("D0 hard negative must be q_anchor")
        for name in ("row_id", "source_sha256", "instruction_sha256"):
            if query[name] != candidate[name]:
                _fail(f"D0 {kind} {name} binding differs")
        if query["endpoint_sha256"] == candidate["endpoint_sha256"]:
            _fail("D0 hard-negative endpoint equals target endpoint")
        _semantic_relation(query["semantics"], candidate["semantics"], kind)
    dino_query = _exact_list(
        case["dino_query_embedding"], label="D0 DINO query embedding"
    )
    dino_clean = _exact_list(
        case["dino_clean_embedding"], label="D0 DINO clean embedding"
    )
    if len(dino_query) < 2 or len(dino_query) != len(dino_clean):
        _fail("D0 DINO embedding geometry differs")
    for index, value_item in enumerate(dino_query + dino_clean):
        _finite_float(value_item, label=f"D0 DINO value[{index}]")
    if sum(item * item for item in dino_query) <= 0.0 or sum(
        item * item for item in dino_clean
    ) <= 0.0:
        _fail("D0 DINO embedding has zero norm")
    return json.loads(canonical_json_bytes(case).decode("ascii"))


def _validate_request(
    value: Any,
    *,
    cases: Mapping[str, Mapping[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    request = _closed_dict(
        value, _REQUEST_FIELDS, label="classification request"
    )
    if request["schema_version"] != CLASSIFICATION_REQUEST_SCHEMA:
        _fail("classification request schema differs")
    for name in ("decision_id", "case_id", "q_y_item_id", "q_anchor_item_id"):
        _sha256(request[name], label=f"classification request {name}")
    if request["case_id"] not in cases:
        _fail("classification request case is unknown")
    case = cases[request["case_id"]]
    if request["q_y_item_id"] != case["query_item_id"]:
        _fail("classification request q_y differs from its case query")
    kind = request["candidate_kind"]
    if type(kind) is not str or kind not in CLASSIFICATION_KINDS:
        _fail("classification request candidate kind differs")
    expected_anchor = (
        case["clean_item_id"]
        if kind == "compatible"
        else case["hard_negative_item_ids"][kind]
    )
    if request["q_anchor_item_id"] != expected_anchor:
        _fail("classification request anchor/kind binding differs")
    status = request["candidate_status"]
    verdict = request["requested_verdict"]
    if type(status) is not str or status not in ITEM_EVIDENCE_STATUSES:
        _fail("classification request candidate status differs")
    if type(verdict) is not str or verdict not in CLASSIFICATION_VERDICTS:
        _fail("classification request verdict differs")
    expected_verdict = (
        "excluded"
        if status == "unqualified"
        else "positive" if kind == "compatible" else "negative"
    )
    if verdict != expected_verdict:
        _fail("classification request verdict does not follow closed policy")
    q_y = items[request["q_y_item_id"]]
    q_anchor = items[request["q_anchor_item_id"]]
    if status != q_anchor["item_evidence_status"]:
        _fail("classification request/item evidence status differs")
    _semantic_relation(q_y["semantics"], q_anchor["semantics"], kind)
    return json.loads(canonical_json_bytes(request).decode("ascii"))


def _validate_split(
    value: Any,
    *,
    cases: Sequence[Mapping[str, Any]],
    referenced_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    split = _closed_dict(value, _SPLIT_FIELDS, label="D0 split manifest")
    if split["schema_version"] != SPLIT_MANIFEST_SCHEMA:
        _fail("D0 split schema differs")
    case_ids = _unique_sha_list(split["d0_case_ids"], label="D0 split case IDs")
    if case_ids != [case["case_id"] for case in cases]:
        _fail("D0 split case ordering/coverage differs")
    derived = {
        "d0_content_ids": sorted({item["content_id"] for item in referenced_items}),
        "d0_generator_ids": sorted({item["generator_id"] for item in referenced_items}),
        "d0_actor_scene_ids": sorted({item["actor_scene_id"] for item in referenced_items}),
    }
    for name, expected in derived.items():
        observed = _unique_sha_list(split[name], label=f"split {name}")
        if observed != expected:
            _fail(f"split {name} is not the exact derived D0 set")
    development_names = (
        "development_content_ids",
        "development_generator_ids",
        "development_actor_scene_ids",
    )
    for name in development_names:
        if not _unique_sha_list(split[name], label=f"split {name}"):
            _fail(f"split {name} must be non-empty")
    pairs = (
        ("d0_content_ids", "development_content_ids"),
        ("d0_generator_ids", "development_generator_ids"),
        ("d0_actor_scene_ids", "development_actor_scene_ids"),
    )
    for heldout, development in pairs:
        if set(split[heldout]) & set(split[development]):
            _fail(f"D0/development overlap exists for {heldout}")
    flags = (
        "content_disjoint_holdout",
        "generator_disjoint_holdout",
        "actor_scene_disjoint_holdout",
    )
    for name in flags:
        if _exact_bool(split[name], label=f"split {name}") is not True:
            _fail(f"split {name} must be true")
    unsigned = dict(split)
    declared = _sha256(unsigned.pop("split_digest"), label="split manifest digest")
    if object_sha256(unsigned) != declared:
        _fail("D0 split manifest digest differs")
    return json.loads(canonical_json_bytes(split).decode("ascii"))


def _request_manifest_sha256(requests: Sequence[Mapping[str, Any]]) -> str:
    return object_sha256(
        {
            "schema_version": "bernini-action-teacher-classification-request-manifest-v1",
            "requests": list(requests),
        }
    )


def validate_benchmark_v1(
    payload_value: Any,
    authority_value: Any,
    *,
    expected_payload_sha256: str,
    expected_authority_sha256: str,
    expected_official_row_authority_sha256: str,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Validate one externally pinned benchmark and all of its evidence.

    Expected digests are caller authority.  They are never inferred from the
    objects being validated.
    """

    expected_payload = _sha256(
        expected_payload_sha256, label="expected benchmark payload SHA-256"
    )
    expected_authority = _sha256(
        expected_authority_sha256, label="expected benchmark authority SHA-256"
    )
    expected_rows = _sha256(
        expected_official_row_authority_sha256,
        label="expected official row authority SHA-256",
    )
    payload = _closed_dict(payload_value, _PAYLOAD_FIELDS, label="benchmark payload")
    if payload["schema_version"] != BENCHMARK_PAYLOAD_SCHEMA:
        _fail("benchmark payload schema differs")
    _sha256(payload["benchmark_id"], label="benchmark ID")
    synthetic = _exact_bool(payload["synthetic_fixture"], label="synthetic fixture")
    if synthetic and not allow_synthetic_fixture:
        _fail("synthetic benchmark requires explicit caller opt-in")
    protocol = _validate_protocol(payload["protocol"])
    raw_items = _exact_list(payload["q_items"], label="benchmark q items")
    normalized_items: list[dict[str, Any]] = []
    vectors: dict[str, list[float]] = {}
    item_by_id: dict[str, dict[str, Any]] = {}
    seen_item_receipts: set[str] = set()
    seen_endpoints: set[str] = set()
    seen_materializations: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        item, vector = _validate_item(raw_item)
        item_id = item["item_id"]
        materialization_digest = item["materialization_receipt"]["receipt_sha256"]
        if (
            item_id in item_by_id
            or item["candidate_receipt_digest"] in seen_item_receipts
            or item["endpoint_sha256"] in seen_endpoints
            or materialization_digest in seen_materializations
        ):
            _fail(f"benchmark q item[{index}] duplicates an item/receipt/endpoint/materialization")
        item_by_id[item_id] = item
        vectors[item_id] = vector
        normalized_items.append(item)
        seen_item_receipts.add(item["candidate_receipt_digest"])
        seen_endpoints.add(item["endpoint_sha256"])
        seen_materializations.add(materialization_digest)
    raw_cases = _exact_list(payload["cases"], label="benchmark D0 cases")
    if len(raw_cases) != CASE_COUNT:
        _fail("benchmark must contain exactly 32 D0 cases")
    normalized_cases: list[dict[str, Any]] = []
    case_by_id: dict[str, dict[str, Any]] = {}
    all_case_item_ids: list[str] = []
    class_counts: dict[str, int] = {}
    for index, raw_case in enumerate(raw_cases):
        case = _validate_case(raw_case, items=item_by_id)
        if case["case_id"] in case_by_id:
            _fail(f"benchmark case[{index}] duplicates a case ID")
        case_by_id[case["case_id"]] = case
        normalized_cases.append(case)
        class_counts[case["action_class"]] = class_counts.get(case["action_class"], 0) + 1
        all_case_item_ids.extend(
            [case["query_item_id"], case["clean_item_id"]]
            + [case["hard_negative_item_ids"][kind] for kind in HARD_NEGATIVE_KINDS]
        )
    if (
        len(class_counts) != ACTION_CLASS_COUNT
        or any(count != CASES_PER_ACTION_CLASS for count in class_counts.values())
    ):
        _fail("D0 cases must cover exactly eight classes with four cases each")
    if len(set(all_case_item_ids)) != len(all_case_item_ids):
        _fail("D0 cases reuse a q item across cases/roles")
    if set(all_case_item_ids) != set(item_by_id):
        _fail("benchmark q-item coverage differs from D0 case evidence")
    split = _validate_split(
        payload["split_manifest"],
        cases=normalized_cases,
        referenced_items=[item_by_id[item_id] for item_id in all_case_item_ids],
    )
    raw_requests = _exact_list(
        payload["classification_requests"], label="classification requests"
    )
    normalized_requests: list[dict[str, Any]] = []
    seen_decisions: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw_request in enumerate(raw_requests):
        request = _validate_request(
            raw_request, cases=case_by_id, items=item_by_id
        )
        pair = (request["q_y_item_id"], request["q_anchor_item_id"])
        if request["decision_id"] in seen_decisions or pair in seen_pairs:
            _fail(f"classification request[{index}] duplicates a decision/pair")
        seen_decisions.add(request["decision_id"])
        seen_pairs.add(pair)
        normalized_requests.append(request)
    if not normalized_requests:
        _fail("benchmark must contain at least one classification request")
    normalized_unsigned = {
        "schema_version": BENCHMARK_PAYLOAD_SCHEMA,
        "benchmark_id": payload["benchmark_id"],
        "synthetic_fixture": synthetic,
        "protocol": protocol,
        "split_manifest": split,
        "q_items": normalized_items,
        "cases": normalized_cases,
        "classification_requests": normalized_requests,
    }
    declared_payload = _sha256(
        payload["payload_digest"], label="benchmark declared payload digest"
    )
    observed_payload = object_sha256(normalized_unsigned)
    if declared_payload != observed_payload or observed_payload != expected_payload:
        _fail("benchmark payload is not the externally pinned payload")
    normalized_payload = {**normalized_unsigned, "payload_digest": observed_payload}

    authority = _closed_dict(
        authority_value,
        _BENCHMARK_AUTHORITY_FIELDS,
        label="benchmark authority",
    )
    if authority["schema_version"] != BENCHMARK_AUTHORITY_SCHEMA:
        _fail("benchmark authority schema differs")
    for name in (
        "benchmark_payload_sha256",
        "official_row_authority_sha256",
        "teacher_producer_sha256",
        "upstream_authority_manifest_sha256",
        "qualification_evaluator_sha256",
        "dino_model_sha256",
        "protocol_sha256",
        "split_manifest_sha256",
        "classification_request_manifest_sha256",
        "authority_digest",
    ):
        _sha256(authority[name], label=f"benchmark authority {name}")
    if (
        authority["benchmark_payload_sha256"] != observed_payload
        or authority["official_row_authority_sha256"] != expected_rows
        or authority["protocol_sha256"] != QUALIFICATION_PROTOCOL_SHA256
        or authority["split_manifest_sha256"] != split["split_digest"]
        or authority["classification_request_manifest_sha256"]
        != _request_manifest_sha256(normalized_requests)
        or authority["synthetic_fixture"] is not synthetic
    ):
        _fail("benchmark authority binding differs")
    if authority["qualification_evaluator_sha256"] == authority["teacher_producer_sha256"]:
        _fail("benchmark evaluator is not independent from teacher producer")
    for name in (
        "synthetic_fixture",
        "production_authority",
        "independent_evaluator",
        "content_disjoint_holdout",
        "no_training_authority",
    ):
        _exact_bool(authority[name], label=f"benchmark authority {name}")
    if (
        authority["independent_evaluator"] is not True
        or authority["content_disjoint_holdout"] is not True
        or authority["no_training_authority"] is not True
        or authority["production_authority"] is not (not synthetic)
    ):
        _fail("benchmark authority safety gate differs")
    unsigned_authority = dict(authority)
    declared_authority = unsigned_authority.pop("authority_digest")
    observed_authority = object_sha256(unsigned_authority)
    if declared_authority != observed_authority or observed_authority != expected_authority:
        _fail("benchmark authority is not the externally pinned authority")
    return {
        "payload": json.loads(canonical_json_bytes(normalized_payload).decode("ascii")),
        "authority": json.loads(canonical_json_bytes(authority).decode("ascii")),
        "items_by_id": item_by_id,
        "vectors_by_id": vectors,
        "cases_by_id": case_by_id,
        "requests_by_id": {item["decision_id"]: item for item in normalized_requests},
    }


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        _fail("cosine vector geometry differs")
    left_norm = math.sqrt(_dot(left, left))
    right_norm = math.sqrt(_dot(right, right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        _fail("cosine vector has zero norm")
    return _dot(left, right) / (left_norm * right_norm)


def _auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        _fail("AUROC requires positive and negative scores")
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _median(values: Sequence[float]) -> float:
    if not values:
        _fail("median requires values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        _fail("Pearson correlation geometry differs")
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_delta = [item - left_mean for item in left]
    right_delta = [item - right_mean for item in right]
    denominator = math.sqrt(_dot(left_delta, left_delta) * _dot(right_delta, right_delta))
    if denominator <= 0.0:
        _fail("Pearson correlation is undefined for constant evidence")
    return _dot(left_delta, right_delta) / denominator


def _symmetric_jacobi_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Deterministic Jacobi eigenvalues for the at-most-32 square Gram matrix."""

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        _fail("effective-rank Gram matrix geometry differs")
    matrix_scale = max(abs(item) for row in matrix for item in row)
    if not math.isfinite(matrix_scale) or matrix_scale <= 0.0:
        _fail("effective-rank Gram matrix has zero or non-finite scale")
    # Normalize before applying an absolute floating-point stopping threshold.
    # Without this step, a perfectly valid low-rank matrix whose q vectors are
    # merely small in amplitude is mistaken for an already diagonal matrix.
    # Effective rank is scale invariant, so this normalization changes neither
    # the singular-value probabilities nor the intended gate.
    value = [[item / matrix_scale for item in row] for row in matrix]
    tolerance = 1e-13
    for _sweep in range(100 * size * size):
        p = 0
        q = 1 if size > 1 else 0
        largest = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                magnitude = abs(value[row][column])
                if magnitude > largest:
                    largest = magnitude
                    p, q = row, column
        if largest <= tolerance:
            break
        app = value[p][p]
        aqq = value[q][q]
        apq = value[p][q]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for index in range(size):
            if index in (p, q):
                continue
            aip = value[index][p]
            aiq = value[index][q]
            value[index][p] = value[p][index] = cosine * aip - sine * aiq
            value[index][q] = value[q][index] = sine * aip + cosine * aiq
        value[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        value[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        value[p][q] = value[q][p] = 0.0
    else:  # pragma: no cover - defensive numerical guard
        _fail("effective-rank Jacobi eigensolver did not converge")
    return [value[index][index] for index in range(size)]


def _effective_rank(vectors: Sequence[Sequence[float]]) -> float:
    if len(vectors) < 2 or any(len(vector) != len(vectors[0]) for vector in vectors):
        _fail("effective-rank input geometry differs")
    width = len(vectors[0])
    means = [math.fsum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]
    centered = [
        [value - means[index] for index, value in enumerate(vector)]
        for vector in vectors
    ]
    gram = [
        [_dot(left, right) for right in centered]
        for left in centered
    ]
    eigenvalues = _symmetric_jacobi_eigenvalues(gram)
    maximum = max(max(eigenvalues), 0.0)
    cutoff = maximum * 1e-12
    singular_values = [
        math.sqrt(max(value, 0.0))
        for value in eigenvalues
        if value > cutoff
    ]
    total = math.fsum(singular_values)
    if total <= 0.0:
        _fail("effective rank is undefined for zero-rank q evidence")
    probabilities = [value / total for value in singular_values]
    entropy = -math.fsum(probability * math.log(probability) for probability in probabilities)
    return math.exp(entropy)


def _enforce_metric_gates(metrics: Mapping[str, Any]) -> None:
    if metrics["hard_negative_auroc"] < 0.80:
        _fail("hard-negative AUROC is below 0.80")
    if metrics["hard_negative_leave_one_case_out_min_auroc"] <= 0.65:
        _fail("deterministic hard-negative lower bound is not greater than 0.65")
    if metrics["clean_control_median_margin"] < 0.10:
        _fail("clean-vs-control median margin is below 0.10")
    if metrics["clean_control_pair_wins"] < 24:
        _fail("clean-vs-control pair wins are below 24/32")
    if metrics["cross_content_recall_at_1"] < 0.50:
        _fail("cross-content R@1 is below 0.50")
    if metrics["appearance_action_similarity_abs_pearson_correlation"] > 0.20:
        _fail("action/appearance absolute correlation is above 0.20")
    if metrics["effective_rank"] < 8.0:
        _fail("q effective rank is below 8")


def recompute_qualification_metrics_v1(checked_benchmark: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all pre-registered metrics from validated case evidence."""

    if type(checked_benchmark) is not dict or set(checked_benchmark) != {
        "payload", "authority", "items_by_id", "vectors_by_id", "cases_by_id", "requests_by_id"
    }:
        _fail("checked benchmark shape differs")
    payload = checked_benchmark["payload"]
    items = checked_benchmark["items_by_id"]
    vectors = checked_benchmark["vectors_by_id"]
    cases = payload["cases"]
    positive_scores: list[float] = []
    negative_scores_by_case: list[list[float]] = []
    margins: list[float] = []
    action_clean_scores: list[float] = []
    appearance_clean_scores: list[float] = []
    query_vectors: list[list[float]] = []
    clean_vectors: list[list[float]] = []
    classes: list[str] = []
    for case in cases:
        query = vectors[case["query_item_id"]]
        clean = vectors[case["clean_item_id"]]
        query_vectors.append(query)
        clean_vectors.append(clean)
        classes.append(case["action_class"])
        clean_score = _cosine(query, clean)
        positive_scores.append(clean_score)
        action_clean_scores.append(clean_score)
        negative_scores = [
            _cosine(query, vectors[case["hard_negative_item_ids"][kind]])
            for kind in HARD_NEGATIVE_KINDS
        ]
        negative_scores_by_case.append(negative_scores)
        noop_score = negative_scores[HARD_NEGATIVE_KINDS.index("noop")]
        margins.append(clean_score - noop_score)
        appearance_clean_scores.append(
            _cosine(case["dino_query_embedding"], case["dino_clean_embedding"])
        )
    all_negative = [score for row in negative_scores_by_case for score in row]
    auroc = _auroc(positive_scores, all_negative)
    leave_one_out = []
    for excluded in range(CASE_COUNT):
        fold_positive = [score for index, score in enumerate(positive_scores) if index != excluded]
        fold_negative = [
            score
            for index, row in enumerate(negative_scores_by_case)
            if index != excluded
            for score in row
        ]
        leave_one_out.append(_auroc(fold_positive, fold_negative))
    wins = sum(1 for margin in margins if margin > 0.0)
    correct = 0
    for query_index, query in enumerate(query_vectors):
        scores = [_cosine(query, clean) for clean in clean_vectors]
        best_score = max(scores)
        best_indices = [index for index, score in enumerate(scores) if score == best_score]
        # All exact top ties must agree in class; otherwise R@1 is ambiguous and
        # counted as wrong rather than resolved by input ordering.
        if best_indices and all(classes[index] == classes[query_index] for index in best_indices):
            # Endpoint content is independently checked across all D0 q items;
            # repeat the local cross-content assertion at the retrieval point.
            query_content = items[cases[query_index]["query_item_id"]]["content_id"]
            if all(
                items[cases[index]["clean_item_id"]]["content_id"] != query_content
                for index in best_indices
            ):
                correct += 1
    unsigned = {
        "schema_version": METRICS_SCHEMA,
        "case_count": CASE_COUNT,
        "action_class_count": len(set(classes)),
        "hard_negative_auroc": _metric_float(auroc),
        "hard_negative_leave_one_case_out_min_auroc": _metric_float(min(leave_one_out)),
        "clean_control_median_margin": _metric_float(_median(margins)),
        "clean_control_pair_wins": wins,
        "cross_content_recall_at_1": _metric_float(correct / CASE_COUNT),
        "appearance_action_similarity_abs_pearson_correlation": _metric_float(
            abs(_pearson(action_clean_scores, appearance_clean_scores))
        ),
        "effective_rank": _metric_float(_effective_rank(query_vectors)),
        "all_global_gates_pass": True,
    }
    _enforce_metric_gates(unsigned)
    return {**unsigned, "metrics_digest": object_sha256(unsigned)}


def _build_teacher_authority(
    benchmark: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        benchmark["payload"]["synthetic_fixture"] is not False
        or benchmark["authority"]["production_authority"] is not True
    ):
        _fail("non-production benchmark cannot form a teacher authority")
    authority = benchmark["authority"]
    unsigned = {
        "schema_version": distill.TEACHER_QUALIFICATION_AUTHORITY_SCHEMA,
        "teacher_producer_sha256": authority["teacher_producer_sha256"],
        "upstream_authority_manifest_sha256": authority["upstream_authority_manifest_sha256"],
        "qualification_split_manifest_sha256": authority["split_manifest_sha256"],
        "qualification_protocol_sha256": authority["protocol_sha256"],
        "qualification_evaluator_sha256": authority["qualification_evaluator_sha256"],
        "qualification_metrics_sha256": metrics["metrics_digest"],
        "qualification_authority_sha256": authority["authority_digest"],
        "independent_evaluator": True,
        "content_disjoint_holdout": True,
    }
    return {**unsigned, "authority_digest": object_sha256(unsigned)}


def _build_qualification_receipt(
    item: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    materialization = item["materialization_receipt"]
    eligible = item["item_evidence_status"] == "eligible"
    if item["q_kind"] == "q_y" and not eligible:
        _fail("q_y cannot receive an unqualified item leaf")
    unsigned = {
        "schema_version": distill.TEACHER_QUALIFICATION_RECEIPT_SCHEMA,
        "materialization_receipt_sha256": materialization["receipt_sha256"],
        "materialization_role": materialization["role"],
        "phase_tokens_sha256": materialization["phase_tokens_sha256"],
        "global_token_sha256": materialization["global_token_sha256"],
        "row_id": item["row_id"],
        "source_sha256": item["source_sha256"],
        "instruction_sha256": item["instruction_sha256"],
        "endpoint_sha256": item["endpoint_sha256"],
        "semantics_sha256": distill.object_sha256(item["semantics"]),
        "teacher_producer_sha256": authority["teacher_producer_sha256"],
        "upstream_authority_manifest_sha256": authority["upstream_authority_manifest_sha256"],
        "qualification_split_manifest_sha256": authority["qualification_split_manifest_sha256"],
        "qualification_protocol_sha256": authority["qualification_protocol_sha256"],
        "qualification_evaluator_sha256": authority["qualification_evaluator_sha256"],
        "qualification_metrics_sha256": authority["qualification_metrics_sha256"],
        "qualification_authority_sha256": authority["qualification_authority_sha256"],
        "independent_evaluator": True,
        "content_disjoint_holdout": True,
        "qualification_status": "qualified" if eligible else "candidate_unqualified",
        "point_distillation_authorized": item["q_kind"] == "q_y" and eligible,
        "contrastive_authorized": eligible,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


def issue_teacher_qualification_v1(
    payload: Any,
    benchmark_authority: Any,
    *,
    expected_payload_sha256: str,
    expected_benchmark_authority_sha256: str,
    expected_official_row_authority_sha256: str,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Issue a deterministic review-only teacher qualification bundle."""

    checked = validate_benchmark_v1(
        payload,
        benchmark_authority,
        expected_payload_sha256=expected_payload_sha256,
        expected_authority_sha256=expected_benchmark_authority_sha256,
        expected_official_row_authority_sha256=expected_official_row_authority_sha256,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    if (
        checked["payload"]["synthetic_fixture"] is not False
        or checked["authority"]["production_authority"] is not True
    ):
        _fail(
            "synthetic evidence is diagnostic-only and may not emit "
            "distillation-compatible qualification authority or leaves"
        )
    metrics = recompute_qualification_metrics_v1(checked)
    teacher_authority = _build_teacher_authority(checked, metrics)
    qualification_items = []
    for item in checked["payload"]["q_items"]:
        qualification_items.append(
            {
                "item_id": item["item_id"],
                "q_kind": item["q_kind"],
                "candidate_receipt_digest": item["candidate_receipt_digest"],
                "qualification_receipt": _build_qualification_receipt(
                    item, authority=teacher_authority
                ),
            }
        )
    unsigned = {
        "schema_version": QUALIFICATION_BUNDLE_SCHEMA,
        "benchmark_payload_sha256": checked["payload"]["payload_digest"],
        "benchmark_authority_sha256": checked["authority"]["authority_digest"],
        "metrics": metrics,
        "teacher_authority": teacher_authority,
        "qualification_items": qualification_items,
        "independent_benchmark_overrides_candidate_unqualified": True,
        "scientific_scope": "representation-qualification-only-not-decoded-video",
        "decoded_video_gate_authorized": False,
        "training_authorized": False,
        "local_only": True,
    }
    bundle = {**unsigned, "bundle_digest": object_sha256(unsigned)}
    return json.loads(canonical_json_bytes(bundle).decode("ascii"))


def _validate_metrics_receipt(value: Any) -> dict[str, Any]:
    metrics = _closed_dict(value, _METRIC_FIELDS, label="qualification metrics")
    if metrics["schema_version"] != METRICS_SCHEMA:
        _fail("qualification metrics schema differs")
    _exact_int(metrics["case_count"], label="qualification metrics case count", minimum=1)
    _exact_int(
        metrics["action_class_count"],
        label="qualification metrics action class count",
        minimum=1,
    )
    _exact_int(
        metrics["clean_control_pair_wins"],
        label="qualification metrics clean/control wins",
        minimum=0,
    )
    for name in (
        "hard_negative_auroc",
        "hard_negative_leave_one_case_out_min_auroc",
        "clean_control_median_margin",
        "cross_content_recall_at_1",
        "appearance_action_similarity_abs_pearson_correlation",
        "effective_rank",
    ):
        _finite_float(metrics[name], label=f"qualification metrics {name}")
    if (
        metrics["case_count"] != CASE_COUNT
        or metrics["action_class_count"] != ACTION_CLASS_COUNT
        or metrics["clean_control_pair_wins"] > CASE_COUNT
        or _exact_bool(
            metrics["all_global_gates_pass"],
            label="qualification metrics global gate",
        )
        is not True
    ):
        _fail("qualification metrics domain differs")
    unsigned = dict(metrics)
    declared = _sha256(
        unsigned.pop("metrics_digest"), label="qualification metrics digest"
    )
    if object_sha256(unsigned) != declared:
        _fail("qualification metrics nested digest differs")
    _enforce_metric_gates(metrics)
    return json.loads(canonical_json_bytes(metrics).decode("ascii"))


def _validate_qualification_leaf_shape(
    value: Any, *, q_kind: str
) -> dict[str, Any]:
    fields = set(distill._QUALIFICATION_RECEIPT_FIELDS)  # type: ignore[attr-defined]
    leaf = _closed_dict(value, fields, label="teacher qualification leaf")
    if leaf["schema_version"] != distill.TEACHER_QUALIFICATION_RECEIPT_SCHEMA:
        _fail("teacher qualification leaf schema differs")
    expected_role = "target" if q_kind == "q_y" else "anchor"
    if leaf["materialization_role"] != expected_role:
        _fail("teacher qualification leaf role differs")
    for name in (
        "materialization_receipt_sha256",
        "phase_tokens_sha256",
        "global_token_sha256",
        "row_id",
        "source_sha256",
        "instruction_sha256",
        "endpoint_sha256",
        "semantics_sha256",
        "teacher_producer_sha256",
        "upstream_authority_manifest_sha256",
        "qualification_split_manifest_sha256",
        "qualification_protocol_sha256",
        "qualification_evaluator_sha256",
        "qualification_metrics_sha256",
        "qualification_authority_sha256",
        "receipt_digest",
    ):
        _sha256(leaf[name], label=f"teacher qualification leaf {name}")
    if leaf["qualification_evaluator_sha256"] == leaf["teacher_producer_sha256"]:
        _fail("teacher qualification leaf evaluator is not independent")
    for name in (
        "independent_evaluator",
        "content_disjoint_holdout",
        "point_distillation_authorized",
        "contrastive_authorized",
    ):
        _exact_bool(leaf[name], label=f"teacher qualification leaf {name}")
    if leaf["independent_evaluator"] is not True or leaf["content_disjoint_holdout"] is not True:
        _fail("teacher qualification leaf independence gate differs")
    status = leaf["qualification_status"]
    if type(status) is not str:
        _fail("teacher qualification leaf status must be an exact string")
    if q_kind == "q_y":
        if (
            status != "qualified"
            or leaf["point_distillation_authorized"] is not True
            or leaf["contrastive_authorized"] is not True
        ):
            _fail("q_y qualification leaf authorization differs")
    elif q_kind == "q_anchor":
        if status not in ("qualified", "candidate_unqualified", "rejected"):
            _fail("q_anchor qualification leaf status differs")
        if leaf["point_distillation_authorized"] is not False:
            _fail("q_anchor qualification leaf may not authorize point distillation")
        if leaf["contrastive_authorized"] is not (status == "qualified"):
            _fail("q_anchor qualification leaf contrastive route differs")
    else:
        _fail("qualification leaf q kind differs")
    unsigned = dict(leaf)
    declared = unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        _fail("teacher qualification leaf nested digest differs")
    return json.loads(canonical_json_bytes(leaf).decode("ascii"))


def _validate_qualification_bundle_layers(
    value: Any,
    *,
    expected_teacher_authority_sha256: str,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    bundle = _closed_dict(
        value, _QUALIFICATION_BUNDLE_FIELDS, label="qualification bundle"
    )
    if bundle["schema_version"] != QUALIFICATION_BUNDLE_SCHEMA:
        _fail("qualification bundle schema differs")
    for name in (
        "benchmark_payload_sha256",
        "benchmark_authority_sha256",
        "bundle_digest",
    ):
        _sha256(bundle[name], label=f"qualification bundle {name}")
    metrics = _validate_metrics_receipt(bundle["metrics"])
    expected_teacher = _sha256(
        expected_teacher_authority_sha256,
        label="expected teacher qualification authority SHA-256",
    )
    try:
        teacher_authority = distill._validate_teacher_authority(  # type: ignore[attr-defined]
            bundle["teacher_authority"], expected_sha256=expected_teacher
        )
    except distill.ActionAnchorDistillationError as error:
        raise ActionTeacherQualificationError(
            f"teacher authority is not distillation-compatible: {error}"
        ) from error
    raw_items = _exact_list(
        bundle["qualification_items"], label="qualification bundle items"
    )
    if len(raw_items) <= 0:
        _fail("qualification bundle must contain qualification items")
    items: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    seen_candidates: set[str] = set()
    seen_leaves: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        item = _closed_dict(
            raw_item,
            _QUALIFICATION_ITEM_FIELDS,
            label=f"qualification bundle item[{index}]",
        )
        item_id = _sha256(item["item_id"], label="qualification bundle item ID")
        candidate_digest = _sha256(
            item["candidate_receipt_digest"],
            label="qualification bundle candidate receipt digest",
        )
        q_kind = item["q_kind"]
        if type(q_kind) is not str or q_kind not in Q_KINDS:
            _fail("qualification bundle q kind differs")
        leaf = _validate_qualification_leaf_shape(
            item["qualification_receipt"], q_kind=q_kind
        )
        leaf_digest = leaf["receipt_digest"]
        if (
            item_id in seen_items
            or candidate_digest in seen_candidates
            or leaf_digest in seen_leaves
        ):
            _fail("qualification bundle duplicates an item/candidate/leaf")
        seen_items.add(item_id)
        seen_candidates.add(candidate_digest)
        seen_leaves.add(leaf_digest)
        items.append(
            {
                "item_id": item_id,
                "q_kind": q_kind,
                "candidate_receipt_digest": candidate_digest,
                "qualification_receipt": leaf,
            }
        )
    for name, expected in (
        ("independent_benchmark_overrides_candidate_unqualified", True),
        ("decoded_video_gate_authorized", False),
        ("training_authorized", False),
        ("local_only", True),
    ):
        if _exact_bool(bundle[name], label=f"qualification bundle {name}") is not expected:
            _fail(f"qualification bundle {name} differs")
    if bundle["scientific_scope"] != "representation-qualification-only-not-decoded-video":
        _fail("qualification bundle scientific scope differs")
    normalized_unsigned = {
        "schema_version": QUALIFICATION_BUNDLE_SCHEMA,
        "benchmark_payload_sha256": bundle["benchmark_payload_sha256"],
        "benchmark_authority_sha256": bundle["benchmark_authority_sha256"],
        "metrics": metrics,
        "teacher_authority": teacher_authority,
        "qualification_items": items,
        "independent_benchmark_overrides_candidate_unqualified": True,
        "scientific_scope": "representation-qualification-only-not-decoded-video",
        "decoded_video_gate_authorized": False,
        "training_authorized": False,
        "local_only": True,
    }
    declared_bundle = bundle["bundle_digest"]
    expected_bundle = _sha256(
        expected_bundle_sha256, label="expected qualification bundle SHA-256"
    )
    observed_bundle = object_sha256(normalized_unsigned)
    if declared_bundle != observed_bundle or observed_bundle != expected_bundle:
        _fail("qualification bundle nested digest/external pin differs")
    return {**normalized_unsigned, "bundle_digest": observed_bundle}


def evaluate_synthetic_diagnostic_v1(
    payload: Any,
    benchmark_authority: Any,
    *,
    expected_payload_sha256: str,
    expected_benchmark_authority_sha256: str,
    expected_official_row_authority_sha256: str,
) -> dict[str, Any]:
    """Evaluate a marked synthetic fixture without emitting trusted leaves.

    The returned schema is intentionally unrelated to the stable distillation
    authority and qualification schemas.  It contains neither a teacher
    authority nor any per-item receipt, so it cannot be fed to the distiller
    even when its own diagnostic digest is pinned.
    """

    checked = validate_benchmark_v1(
        payload,
        benchmark_authority,
        expected_payload_sha256=expected_payload_sha256,
        expected_authority_sha256=expected_benchmark_authority_sha256,
        expected_official_row_authority_sha256=expected_official_row_authority_sha256,
        allow_synthetic_fixture=True,
    )
    if (
        checked["payload"]["synthetic_fixture"] is not True
        or checked["authority"]["synthetic_fixture"] is not True
        or checked["authority"]["production_authority"] is not False
    ):
        _fail("synthetic diagnostic requires an exact non-production fixture authority")
    metrics = recompute_qualification_metrics_v1(checked)
    unsigned = {
        "schema_version": SYNTHETIC_DIAGNOSTIC_SCHEMA,
        "benchmark_payload_sha256": checked["payload"]["payload_digest"],
        "benchmark_authority_sha256": checked["authority"]["authority_digest"],
        "metrics": metrics,
        "synthetic_fixture": True,
        "distillation_authority_emitted": False,
        "qualification_leaves_emitted": False,
        "compatibility_receipts_emitted": False,
        "training_authorized": False,
        "local_only": True,
    }
    return {**unsigned, "diagnostic_digest": object_sha256(unsigned)}


def validate_teacher_qualification_bundle_v1(
    value: Any,
    *,
    payload: Any,
    benchmark_authority: Any,
    expected_payload_sha256: str,
    expected_benchmark_authority_sha256: str,
    expected_official_row_authority_sha256: str,
    expected_bundle_sha256: str,
    expected_teacher_authority_sha256: str,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    candidate = _validate_qualification_bundle_layers(
        value,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        expected_bundle_sha256=expected_bundle_sha256,
    )
    # Re-issuance from externally pinned evidence defeats a self-consistent
    # leaf/tree rewrite, even if an attacker can recompute all inner digests.
    expected = issue_teacher_qualification_v1(
        payload,
        benchmark_authority,
        expected_payload_sha256=expected_payload_sha256,
        expected_benchmark_authority_sha256=expected_benchmark_authority_sha256,
        expected_official_row_authority_sha256=expected_official_row_authority_sha256,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    if canonical_json_bytes(candidate) != canonical_json_bytes(expected):
        _fail("qualification bundle differs from pinned-evidence re-issuance")
    return json.loads(canonical_json_bytes(candidate).decode("ascii"))


def _qualification_by_item(bundle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for entry in bundle["qualification_items"]:
        if entry["item_id"] in result:
            _fail("qualification bundle contains duplicate item IDs")
        result[entry["item_id"]] = entry
    return result


def _validate_single_q_receipt(
    receipt: Any,
    *,
    expected_receipt_digest: str,
    q_kind: str,
    item: Mapping[str, Any],
    qualification_entry: Mapping[str, Any],
    teacher_authority_sha256: str,
) -> dict[str, Any]:
    expected_receipt = _sha256(
        expected_receipt_digest, label=f"expected {q_kind} receipt digest"
    )
    qualification_digest = qualification_entry["qualification_receipt"]["receipt_digest"]
    try:
        checked = distill.validate_q_receipt_v1(
            receipt,
            expected_teacher_authority_sha256=teacher_authority_sha256,
            expected_qualification_receipt_digests=[qualification_digest],
        )
    except distill.ActionAnchorDistillationError as error:
        raise ActionTeacherQualificationError(
            f"{q_kind} receipt is not distillation-compatible: {error}"
        ) from error
    if checked["receipt_digest"] != expected_receipt:
        _fail(f"{q_kind} receipt is not externally pinned")
    if checked["q_kind"] != q_kind or checked["layout"]["batch_size"] != 1:
        _fail("classification requires one-item teacher q receipts")
    receipt_item = checked["items"][0]
    for name in (
        "row_id",
        "source_sha256",
        "instruction_sha256",
        "endpoint_sha256",
        "semantics",
    ):
        if canonical_json_bytes(receipt_item[name]) != canonical_json_bytes(item[name]):
            _fail(f"{q_kind} receipt candidate-item {name} binding differs")
    evidence = receipt_item["teacher_evidence"]
    if (
        canonical_json_bytes(evidence["materialization_receipt"])
        != canonical_json_bytes(item["materialization_receipt"])
        or canonical_json_bytes(evidence["qualification_receipt"])
        != canonical_json_bytes(qualification_entry["qualification_receipt"])
    ):
        _fail(f"{q_kind} receipt evidence transplant detected")
    return checked


def _build_classification_authority(
    *,
    benchmark: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        benchmark["payload"]["synthetic_fixture"] is not False
        or benchmark["authority"]["production_authority"] is not True
    ):
        _fail("non-production benchmark cannot form a classification authority")
    unsigned = {
        "schema_version": CLASSIFICATION_AUTHORITY_SCHEMA,
        "benchmark_authority_sha256": benchmark["authority"]["authority_digest"],
        "qualification_bundle_sha256": bundle["bundle_digest"],
        "teacher_authority_sha256": bundle["teacher_authority"]["authority_digest"],
        "classification_request_manifest_sha256": benchmark["authority"]["classification_request_manifest_sha256"],
        "classification_protocol_sha256": CLASSIFICATION_PROTOCOL_SHA256,
        "classification_evaluator_sha256": benchmark["authority"]["qualification_evaluator_sha256"],
        "independent_evaluator": True,
        "decoded_video_gate_authorized": False,
        "training_authorized": False,
        "local_only": True,
    }
    return {**unsigned, "authority_digest": object_sha256(unsigned)}


def _build_decision(
    *,
    request: Mapping[str, Any],
    q_y: Mapping[str, Any],
    q_anchor: Mapping[str, Any],
    q_y_item: Mapping[str, Any],
    q_anchor_item: Mapping[str, Any],
    q_y_qualification: Mapping[str, Any],
    q_anchor_qualification: Mapping[str, Any],
    classification_authority: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    matches, mismatch = _semantic_relation(
        q_y_item["semantics"], q_anchor_item["semantics"], request["candidate_kind"]
    )
    verdict = request["requested_verdict"]
    training_use = "excluded" if verdict == "excluded" else "contrastive-only"
    contrastive_role = (
        "none"
        if verdict == "excluded"
        else "positive" if verdict == "positive" else "negative"
    )
    unsigned = {
        "schema_version": CLASSIFICATION_DECISION_SCHEMA,
        "decision_id": request["decision_id"],
        "row_id": q_y_item["row_id"],
        "q_y_item_id": q_y_item["item_id"],
        "q_anchor_item_id": q_anchor_item["item_id"],
        "q_y_receipt_digest": q_y["receipt_digest"],
        "q_anchor_receipt_digest": q_anchor["receipt_digest"],
        "q_y_qualification_receipt_digest": q_y_qualification["qualification_receipt"]["receipt_digest"],
        "q_anchor_qualification_receipt_digest": q_anchor_qualification["qualification_receipt"]["receipt_digest"],
        "q_y_endpoint_sha256": q_y_item["endpoint_sha256"],
        "q_anchor_endpoint_sha256": q_anchor_item["endpoint_sha256"],
        "q_y_materialization_receipt_sha256": q_y_item["materialization_receipt"]["receipt_sha256"],
        "q_anchor_materialization_receipt_sha256": q_anchor_item["materialization_receipt"]["receipt_sha256"],
        "candidate_kind": request["candidate_kind"],
        "candidate_status": request["candidate_status"],
        "verdict": verdict,
        "desired_semantics_sha256": distill.object_sha256(q_y_item["semantics"]),
        "candidate_semantics_sha256": distill.object_sha256(q_anchor_item["semantics"]),
        "axis_matches": matches,
        "mismatch_axes": mismatch,
        "classification_evaluator_sha256": classification_authority["classification_evaluator_sha256"],
        "classification_protocol_sha256": classification_authority["classification_protocol_sha256"],
        "classification_authority_sha256": classification_authority["authority_digest"],
        "compatibility_receipt_digest": compatibility["receipt_digest"],
        "training_use": training_use,
        "contrastive_role": contrastive_role,
        "training_authorized": False,
    }
    return {**unsigned, "decision_digest": object_sha256(unsigned)}


def _build_compatibility_candidate(
    *,
    request: Mapping[str, Any],
    q_y: Mapping[str, Any],
    q_anchor: Mapping[str, Any],
    classification_authority_sha256: str,
) -> dict[str, Any]:
    if request["requested_verdict"] == "excluded":
        candidate_kind = "unqualified"
        qualification_verdict = "abstain"
    else:
        candidate_kind = _DISTILL_KIND[request["candidate_kind"]]
        qualification_verdict = "accept"
    try:
        return distill._build_compatibility_without_recursive_validation(  # type: ignore[attr-defined]
            q_y=q_y,
            q_anchor=q_anchor,
            candidate_kinds=[candidate_kind],
            qualification_verdicts=[qualification_verdict],
            authority=classification_authority_sha256,
        )
    except distill.ActionAnchorDistillationError as error:
        raise ActionTeacherQualificationError(
            f"classification cannot form downstream compatibility receipt: {error}"
        ) from error


def issue_classification_ledger_v1(
    receipt_pairs: Sequence[Mapping[str, Any]],
    *,
    expected_q_y_receipt_digests: Sequence[str],
    expected_q_anchor_receipt_digests: Sequence[str],
    qualification_bundle: Any,
    payload: Any,
    benchmark_authority: Any,
    expected_payload_sha256: str,
    expected_benchmark_authority_sha256: str,
    expected_official_row_authority_sha256: str,
    expected_qualification_bundle_sha256: str,
    expected_teacher_authority_sha256: str,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Issue review-only classifications from externally pinned q receipts."""

    if type(receipt_pairs) not in (list, tuple):
        _fail("classification receipt pairs must be an exact list or tuple")
    if type(expected_q_y_receipt_digests) not in (list, tuple) or type(
        expected_q_anchor_receipt_digests
    ) not in (list, tuple):
        _fail("classification q receipt pins must be exact lists or tuples")
    if not (
        len(receipt_pairs)
        == len(expected_q_y_receipt_digests)
        == len(expected_q_anchor_receipt_digests)
    ):
        _fail("classification q receipt/pin coverage differs")
    checked_benchmark = validate_benchmark_v1(
        payload,
        benchmark_authority,
        expected_payload_sha256=expected_payload_sha256,
        expected_authority_sha256=expected_benchmark_authority_sha256,
        expected_official_row_authority_sha256=expected_official_row_authority_sha256,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    if (
        checked_benchmark["payload"]["synthetic_fixture"] is not False
        or checked_benchmark["authority"]["production_authority"] is not True
    ):
        _fail(
            "synthetic evidence may not emit a classification authority or "
            "downstream compatibility receipt"
        )
    checked_bundle = validate_teacher_qualification_bundle_v1(
        qualification_bundle,
        payload=payload,
        benchmark_authority=benchmark_authority,
        expected_payload_sha256=expected_payload_sha256,
        expected_benchmark_authority_sha256=expected_benchmark_authority_sha256,
        expected_official_row_authority_sha256=expected_official_row_authority_sha256,
        expected_bundle_sha256=expected_qualification_bundle_sha256,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    requests = checked_benchmark["requests_by_id"]
    items = checked_benchmark["items_by_id"]
    qualifications = _qualification_by_item(checked_bundle)
    if len(receipt_pairs) != len(requests):
        _fail("classification receipt pairs must exactly cover frozen requests")
    classification_authority = _build_classification_authority(
        benchmark=checked_benchmark, bundle=checked_bundle
    )
    seen_decisions: set[str] = set()
    decisions: list[dict[str, Any]] = []
    compatibility_receipts: list[dict[str, Any]] = []
    for index, raw_pair in enumerate(receipt_pairs):
        pair = _closed_dict(
            raw_pair, _RECEIPT_PAIR_FIELDS, label=f"classification receipt pair[{index}]"
        )
        decision_id = _sha256(pair["decision_id"], label="classification pair decision ID")
        if decision_id in seen_decisions or decision_id not in requests:
            _fail("classification pair duplicates or does not match a frozen request")
        seen_decisions.add(decision_id)
        request = requests[decision_id]
        q_y_item = items[request["q_y_item_id"]]
        q_anchor_item = items[request["q_anchor_item_id"]]
        q_y_qualification = qualifications[q_y_item["item_id"]]
        q_anchor_qualification = qualifications[q_anchor_item["item_id"]]
        q_y = _validate_single_q_receipt(
            pair["q_y_receipt"],
            expected_receipt_digest=expected_q_y_receipt_digests[index],
            q_kind="q_y",
            item=q_y_item,
            qualification_entry=q_y_qualification,
            teacher_authority_sha256=expected_teacher_authority_sha256,
        )
        q_anchor = _validate_single_q_receipt(
            pair["q_anchor_receipt"],
            expected_receipt_digest=expected_q_anchor_receipt_digests[index],
            q_kind="q_anchor",
            item=q_anchor_item,
            qualification_entry=q_anchor_qualification,
            teacher_authority_sha256=expected_teacher_authority_sha256,
        )
        if request["candidate_status"] == "eligible":
            receipt = q_anchor["items"][0]["teacher_evidence"]["qualification_receipt"]
            if (
                receipt["qualification_status"] != "qualified"
                or receipt["contrastive_authorized"] is not True
                or receipt["point_distillation_authorized"] is not False
            ):
                _fail("active q_anchor lacks contrastive-only qualification")
        compatibility = _build_compatibility_candidate(
            request=request,
            q_y=q_y,
            q_anchor=q_anchor,
            classification_authority_sha256=classification_authority["authority_digest"],
        )
        decision = _build_decision(
            request=request,
            q_y=q_y,
            q_anchor=q_anchor,
            q_y_item=q_y_item,
            q_anchor_item=q_anchor_item,
            q_y_qualification=q_y_qualification,
            q_anchor_qualification=q_anchor_qualification,
            classification_authority=classification_authority,
            compatibility=compatibility,
        )
        decisions.append(decision)
        compatibility_receipts.append(compatibility)
    if seen_decisions != set(requests):
        _fail("classification pair decision coverage differs")
    unsigned = {
        "schema_version": CLASSIFICATION_LEDGER_SCHEMA,
        "benchmark_authority_sha256": checked_benchmark["authority"]["authority_digest"],
        "qualification_bundle_sha256": checked_bundle["bundle_digest"],
        "classification_authority": classification_authority,
        "decisions": decisions,
        "compatibility_receipts": compatibility_receipts,
        "expected_external_pins_required_before_consumption": True,
        "training_authorized": False,
        "local_only": True,
    }
    return {**unsigned, "ledger_digest": object_sha256(unsigned)}


def _validate_classification_authority_layers(
    value: Any, *, expected_authority_sha256: str
) -> dict[str, Any]:
    authority = _closed_dict(
        value,
        _CLASSIFICATION_AUTHORITY_FIELDS,
        label="classification authority",
    )
    if authority["schema_version"] != CLASSIFICATION_AUTHORITY_SCHEMA:
        _fail("classification authority schema differs")
    for name in (
        "benchmark_authority_sha256",
        "qualification_bundle_sha256",
        "teacher_authority_sha256",
        "classification_request_manifest_sha256",
        "classification_protocol_sha256",
        "classification_evaluator_sha256",
        "authority_digest",
    ):
        _sha256(authority[name], label=f"classification authority {name}")
    if authority["classification_protocol_sha256"] != CLASSIFICATION_PROTOCOL_SHA256:
        _fail("classification authority protocol differs")
    for name, expected in (
        ("independent_evaluator", True),
        ("decoded_video_gate_authorized", False),
        ("training_authorized", False),
        ("local_only", True),
    ):
        if _exact_bool(authority[name], label=f"classification authority {name}") is not expected:
            _fail(f"classification authority {name} differs")
    unsigned = dict(authority)
    declared = unsigned.pop("authority_digest")
    expected_authority = _sha256(
        expected_authority_sha256,
        label="expected classification authority SHA-256",
    )
    observed = object_sha256(unsigned)
    if declared != observed or observed != expected_authority:
        _fail("classification authority nested digest/external pin differs")
    return json.loads(canonical_json_bytes(authority).decode("ascii"))


def _validate_classification_decision_layers(
    value: Any,
    *,
    classification_authority_sha256: str,
) -> dict[str, Any]:
    decision = _closed_dict(
        value, _DECISION_FIELDS, label="classification decision leaf"
    )
    if decision["schema_version"] != CLASSIFICATION_DECISION_SCHEMA:
        _fail("classification decision schema differs")
    for name in (
        "decision_id",
        "row_id",
        "q_y_item_id",
        "q_anchor_item_id",
        "q_y_receipt_digest",
        "q_anchor_receipt_digest",
        "q_y_qualification_receipt_digest",
        "q_anchor_qualification_receipt_digest",
        "q_y_endpoint_sha256",
        "q_anchor_endpoint_sha256",
        "q_y_materialization_receipt_sha256",
        "q_anchor_materialization_receipt_sha256",
        "desired_semantics_sha256",
        "candidate_semantics_sha256",
        "classification_evaluator_sha256",
        "classification_protocol_sha256",
        "classification_authority_sha256",
        "compatibility_receipt_digest",
        "decision_digest",
    ):
        _sha256(decision[name], label=f"classification decision {name}")
    if (
        decision["classification_protocol_sha256"] != CLASSIFICATION_PROTOCOL_SHA256
        or decision["classification_authority_sha256"]
        != classification_authority_sha256
    ):
        _fail("classification decision authority/protocol binding differs")
    kind = decision["candidate_kind"]
    status = decision["candidate_status"]
    verdict = decision["verdict"]
    if type(kind) is not str or kind not in CLASSIFICATION_KINDS:
        _fail("classification decision candidate kind differs")
    if type(status) is not str or status not in ITEM_EVIDENCE_STATUSES:
        _fail("classification decision candidate status differs")
    if type(verdict) is not str or verdict not in CLASSIFICATION_VERDICTS:
        _fail("classification decision verdict differs")
    expected_verdict = (
        "excluded"
        if status == "unqualified"
        else "positive" if kind == "compatible" else "negative"
    )
    if verdict != expected_verdict:
        _fail("classification decision verdict policy differs")
    matches = _closed_dict(
        decision["axis_matches"],
        set(distill.SEMANTIC_AXES),
        label="classification decision axis matches",
    )
    if any(type(matches[axis]) is not bool for axis in distill.SEMANTIC_AXES):
        _fail("classification decision axis matches must be exact booleans")
    mismatch = _exact_list(
        decision["mismatch_axes"], label="classification decision mismatch axes"
    )
    if any(type(axis) is not str for axis in mismatch) or len(set(mismatch)) != len(mismatch):
        _fail("classification decision mismatch axes differ")
    derived_mismatch = [axis for axis in distill.SEMANTIC_AXES if matches[axis] is False]
    expected_mismatch = {
        "compatible": [],
        "noop": ["action"],
        "reverse": ["direction"],
        "incomplete": ["outcome"],
        "wrong_actor": ["actor"],
        "wrong_object": ["object"],
        "camera": [],
        "appearance": [],
    }[kind]
    if mismatch != derived_mismatch or mismatch != expected_mismatch:
        _fail("classification decision semantic-axis route differs")
    expected_use = "excluded" if verdict == "excluded" else "contrastive-only"
    expected_role = (
        "none"
        if verdict == "excluded"
        else "positive" if verdict == "positive" else "negative"
    )
    if decision["training_use"] != expected_use or decision["contrastive_role"] != expected_role:
        _fail("classification decision training route differs")
    if _exact_bool(
        decision["training_authorized"],
        label="classification decision training authority",
    ) is not False:
        _fail("classification decision may not authorize training")
    unsigned = dict(decision)
    declared = unsigned.pop("decision_digest")
    if object_sha256(unsigned) != declared:
        _fail("classification decision nested digest differs")
    return json.loads(canonical_json_bytes(decision).decode("ascii"))


def _validate_compatibility_self_digest(value: Any) -> dict[str, Any]:
    fields = set(distill._COMPATIBILITY_FIELDS)  # type: ignore[attr-defined]
    receipt = _closed_dict(
        value, fields, label="classification compatibility receipt"
    )
    if receipt["schema_version"] != distill.COMPATIBILITY_RECEIPT_SCHEMA:
        _fail("classification compatibility receipt schema differs")
    for name in (
        "policy_sha256",
        "q_y_receipt_digest",
        "q_anchor_receipt_digest",
        "classification_authority_sha256",
        "receipt_digest",
    ):
        _sha256(receipt[name], label=f"classification compatibility {name}")
    if receipt["policy_sha256"] != distill.COMPATIBILITY_POLICY_SHA256:
        _fail("classification compatibility policy differs")
    _exact_list(receipt["items"], label="classification compatibility items")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        _fail("classification compatibility nested digest differs")
    return json.loads(canonical_json_bytes(receipt).decode("ascii"))


def _validate_classification_ledger_layers(
    value: Any,
    *,
    expected_classification_authority_sha256: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    ledger = _closed_dict(
        value, _CLASSIFICATION_LEDGER_FIELDS, label="classification ledger"
    )
    if ledger["schema_version"] != CLASSIFICATION_LEDGER_SCHEMA:
        _fail("classification ledger schema differs")
    for name in (
        "benchmark_authority_sha256",
        "qualification_bundle_sha256",
        "ledger_digest",
    ):
        _sha256(ledger[name], label=f"classification ledger {name}")
    authority = _validate_classification_authority_layers(
        ledger["classification_authority"],
        expected_authority_sha256=expected_classification_authority_sha256,
    )
    raw_decisions = _exact_list(
        ledger["decisions"], label="classification ledger decisions"
    )
    raw_compatibility = _exact_list(
        ledger["compatibility_receipts"],
        label="classification ledger compatibility receipts",
    )
    if len(raw_decisions) <= 0 or len(raw_decisions) != len(raw_compatibility):
        _fail("classification ledger decision/compatibility coverage differs")
    decisions: list[dict[str, Any]] = []
    compatibility: list[dict[str, Any]] = []
    seen_decisions: set[str] = set()
    seen_leaves: set[str] = set()
    seen_compatibility: set[str] = set()
    for index, (raw_decision, raw_receipt) in enumerate(
        zip(raw_decisions, raw_compatibility)
    ):
        decision = _validate_classification_decision_layers(
            raw_decision,
            classification_authority_sha256=authority["authority_digest"],
        )
        receipt = _validate_compatibility_self_digest(raw_receipt)
        if (
            decision["decision_id"] in seen_decisions
            or decision["decision_digest"] in seen_leaves
            or receipt["receipt_digest"] in seen_compatibility
        ):
            _fail("classification ledger duplicates a decision/leaf/compatibility receipt")
        if (
            decision["compatibility_receipt_digest"] != receipt["receipt_digest"]
            or receipt["classification_authority_sha256"]
            != authority["authority_digest"]
        ):
            _fail(f"classification ledger row[{index}] receipt binding differs")
        seen_decisions.add(decision["decision_id"])
        seen_leaves.add(decision["decision_digest"])
        seen_compatibility.add(receipt["receipt_digest"])
        decisions.append(decision)
        compatibility.append(receipt)
    for name, expected in (
        ("expected_external_pins_required_before_consumption", True),
        ("training_authorized", False),
        ("local_only", True),
    ):
        if _exact_bool(ledger[name], label=f"classification ledger {name}") is not expected:
            _fail(f"classification ledger {name} differs")
    normalized_unsigned = {
        "schema_version": CLASSIFICATION_LEDGER_SCHEMA,
        "benchmark_authority_sha256": ledger["benchmark_authority_sha256"],
        "qualification_bundle_sha256": ledger["qualification_bundle_sha256"],
        "classification_authority": authority,
        "decisions": decisions,
        "compatibility_receipts": compatibility,
        "expected_external_pins_required_before_consumption": True,
        "training_authorized": False,
        "local_only": True,
    }
    declared = ledger["ledger_digest"]
    expected_ledger = _sha256(
        expected_ledger_sha256, label="expected classification ledger SHA-256"
    )
    observed = object_sha256(normalized_unsigned)
    if declared != observed or observed != expected_ledger:
        _fail("classification ledger nested digest/external pin differs")
    return {**normalized_unsigned, "ledger_digest": observed}


def validate_classification_ledger_v1(
    value: Any,
    receipt_pairs: Sequence[Mapping[str, Any]],
    *,
    expected_q_y_receipt_digests: Sequence[str],
    expected_q_anchor_receipt_digests: Sequence[str],
    qualification_bundle: Any,
    payload: Any,
    benchmark_authority: Any,
    expected_payload_sha256: str,
    expected_benchmark_authority_sha256: str,
    expected_official_row_authority_sha256: str,
    expected_qualification_bundle_sha256: str,
    expected_teacher_authority_sha256: str,
    expected_classification_authority_sha256: str,
    expected_decision_leaf_digests: Sequence[str],
    expected_compatibility_receipt_digests: Sequence[str],
    expected_ledger_sha256: str,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    candidate = _validate_classification_ledger_layers(
        value,
        expected_classification_authority_sha256=
        expected_classification_authority_sha256,
        expected_ledger_sha256=expected_ledger_sha256,
    )
    expected = issue_classification_ledger_v1(
        receipt_pairs,
        expected_q_y_receipt_digests=expected_q_y_receipt_digests,
        expected_q_anchor_receipt_digests=expected_q_anchor_receipt_digests,
        qualification_bundle=qualification_bundle,
        payload=payload,
        benchmark_authority=benchmark_authority,
        expected_payload_sha256=expected_payload_sha256,
        expected_benchmark_authority_sha256=expected_benchmark_authority_sha256,
        expected_official_row_authority_sha256=expected_official_row_authority_sha256,
        expected_qualification_bundle_sha256=expected_qualification_bundle_sha256,
        expected_teacher_authority_sha256=expected_teacher_authority_sha256,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )
    if canonical_json_bytes(candidate) != canonical_json_bytes(expected):
        _fail("classification ledger differs from pinned-input re-issuance")
    if type(expected_decision_leaf_digests) not in (list, tuple) or type(
        expected_compatibility_receipt_digests
    ) not in (list, tuple):
        _fail("classification decision/compatibility pins must be lists or tuples")
    observed_decisions = [item["decision_digest"] for item in candidate["decisions"]]
    observed_compatibility = [item["receipt_digest"] for item in candidate["compatibility_receipts"]]
    pinned_decisions = [
        _sha256(item, label=f"expected decision leaf[{index}] digest")
        for index, item in enumerate(expected_decision_leaf_digests)
    ]
    pinned_compatibility = [
        _sha256(item, label=f"expected compatibility[{index}] digest")
        for index, item in enumerate(expected_compatibility_receipt_digests)
    ]
    if observed_decisions != pinned_decisions or observed_compatibility != pinned_compatibility:
        _fail("classification row-level external pins differ")
    for index, (raw_pair, compatibility) in enumerate(
        zip(receipt_pairs, candidate["compatibility_receipts"])
    ):
        pair = _closed_dict(
            raw_pair,
            _RECEIPT_PAIR_FIELDS,
            label=f"classification validation receipt pair[{index}]",
        )
        q_y_receipt = pair["q_y_receipt"]
        q_anchor_receipt = pair["q_anchor_receipt"]
        try:
            q_y_qualification_digest = q_y_receipt["items"][0][
                "teacher_evidence"
            ]["qualification_receipt"]["receipt_digest"]
            q_anchor_qualification_digest = q_anchor_receipt["items"][0][
                "teacher_evidence"
            ]["qualification_receipt"]["receipt_digest"]
        except (KeyError, IndexError, TypeError) as error:
            raise ActionTeacherQualificationError(
                "classification validation q receipt evidence shape differs"
            ) from error
        try:
            distill.validate_compatibility_receipt_v1(
                compatibility,
                q_y_receipt=q_y_receipt,
                q_anchor_receipt=q_anchor_receipt,
                expected_teacher_authority_sha256=
                expected_teacher_authority_sha256,
                expected_classification_authority_sha256=
                expected_classification_authority_sha256,
                expected_q_y_qualification_receipt_digests=[
                    q_y_qualification_digest
                ],
                expected_q_anchor_qualification_receipt_digests=[
                    q_anchor_qualification_digest
                ],
                expected_decision_receipt_digest=
                pinned_compatibility[index],
            )
        except distill.ActionAnchorDistillationError as error:
            raise ActionTeacherQualificationError(
                f"classification compatibility receipt[{index}] differs: {error}"
            ) from error
    return json.loads(canonical_json_bytes(candidate).decode("ascii"))


def _reject_json_constant(value: str) -> None:
    _fail(f"non-finite JSON constant is forbidden: {value}")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_canonical_json_bytes_v1(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_JSON_BYTES:
        _fail(f"{label} JSON bytes differ")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActionTeacherQualificationError(f"{label} JSON differs") from error
    if type(value) is not dict:
        _fail(f"{label} top-level JSON must be an exact object")
    if raw != canonical_json_bytes(value) + b"\n":
        _fail(f"{label} file is not canonical JSON plus one newline")
    return value


def _stable_pinned_bytes(path_value: os.PathLike[str] | str, *, expected_file_sha256: str, label: str) -> bytes:
    expected = _sha256(expected_file_sha256, label=f"expected {label} file SHA-256")
    path = Path(path_value)
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("stable file reads require O_NOFOLLOW")
    try:
        parent_before = path.parent.lstat()
    except OSError as error:
        raise ActionTeacherQualificationError(f"cannot inspect stable {label} parent") from error
    if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(parent_before.st_mode):
        _fail(f"stable {label} parent must be a plain directory")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActionTeacherQualificationError(f"cannot open stable {label}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0 or before.st_size > _MAX_JSON_BYTES:
            _fail(f"stable {label} must be one bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"stable {label} read made no progress")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"stable {label} grew during read")
        after = os.fstat(descriptor)
        named = path.lstat()
        parent_after = path.parent.lstat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        identity_named = (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns)
        parent_identity_before = (
            parent_before.st_dev,
            parent_before.st_ino,
            parent_before.st_mtime_ns,
        )
        parent_identity_after = (
            parent_after.st_dev,
            parent_after.st_ino,
            parent_after.st_mtime_ns,
        )
        if (
            identity_before != identity_after
            or identity_before != identity_named
            or parent_identity_before != parent_identity_after
            or not stat.S_ISREG(named.st_mode)
            or not stat.S_ISDIR(parent_after.st_mode)
            or stat.S_ISLNK(parent_after.st_mode)
        ):
            _fail(f"stable {label} identity changed during read")
        raw = b"".join(chunks)
        if hashlib.sha256(raw).hexdigest() != expected:
            _fail(f"stable {label} file is not externally pinned")
        return raw
    finally:
        os.close(descriptor)


def load_pinned_benchmark_files_v1(
    payload_path: os.PathLike[str] | str,
    authority_path: os.PathLike[str] | str,
    *,
    expected_payload_file_sha256: str,
    expected_authority_file_sha256: str,
    expected_payload_sha256: str,
    expected_authority_sha256: str,
    expected_official_row_authority_sha256: str,
    allow_synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """Stable-read and validate two independently pinned canonical JSON files."""

    payload_raw = _stable_pinned_bytes(
        payload_path,
        expected_file_sha256=expected_payload_file_sha256,
        label="benchmark payload",
    )
    authority_raw = _stable_pinned_bytes(
        authority_path,
        expected_file_sha256=expected_authority_file_sha256,
        label="benchmark authority",
    )
    payload = parse_canonical_json_bytes_v1(payload_raw, label="benchmark payload")
    authority = parse_canonical_json_bytes_v1(authority_raw, label="benchmark authority")
    return validate_benchmark_v1(
        payload,
        authority,
        expected_payload_sha256=expected_payload_sha256,
        expected_authority_sha256=expected_authority_sha256,
        expected_official_row_authority_sha256=expected_official_row_authority_sha256,
        allow_synthetic_fixture=allow_synthetic_fixture,
    )


def load_pinned_canonical_json_v1(
    path: os.PathLike[str] | str,
    *,
    expected_file_sha256: str,
    expected_object_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Stable-read any later-stage ledger/receipt with two external pins."""

    _text(label, label="pinned JSON label", maximum_bytes=128)
    expected_object = _sha256(
        expected_object_sha256, label=f"expected {label} object SHA-256"
    )
    raw = _stable_pinned_bytes(
        path, expected_file_sha256=expected_file_sha256, label=label
    )
    value = parse_canonical_json_bytes_v1(raw, label=label)
    if object_sha256(value) != expected_object:
        _fail(f"{label} object is not externally pinned")
    return value


def publish_create_only_json_v1(
    path_value: os.PathLike[str] | str,
    value: Any,
    *,
    expected_object_sha256: str,
) -> dict[str, Any]:
    """Publish one exact canonical JSON object without overwriting anything."""

    expected = _sha256(expected_object_sha256, label="expected publication object SHA-256")
    if type(value) is not dict or object_sha256(value) != expected:
        _fail("publication object is not the caller-pinned object")
    path = Path(path_value)
    parent_info = path.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        _fail("publication parent must be an existing plain directory")
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("create-only publication requires O_NOFOLLOW")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    raw = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as error:
        raise ActionTeacherQualificationError(
            f"refusing to overwrite create-only artifact: {path}"
        ) from error
    except OSError as error:
        raise ActionTeacherQualificationError("cannot create publication artifact") from error
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("create-only publication write made no progress")
            offset += written
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != len(raw):
            _fail("create-only publication file identity differs")
    finally:
        os.close(descriptor)
    reread = _stable_pinned_bytes(
        path,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        label="published artifact",
    )
    parsed = parse_canonical_json_bytes_v1(reread, label="published artifact")
    if canonical_json_bytes(parsed) != canonical_json_bytes(value):
        _fail("create-only publication reread differs")
    return {
        "path": str(path),
        "object_sha256": expected,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "created": True,
    }


def contract_receipt_v1() -> dict[str, Any]:
    unsigned = {
        "schema_version": "bernini-action-teacher-qualification-contract-v1",
        "benchmark_payload_schema": BENCHMARK_PAYLOAD_SCHEMA,
        "benchmark_authority_schema": BENCHMARK_AUTHORITY_SCHEMA,
        "qualification_authority_schema": distill.TEACHER_QUALIFICATION_AUTHORITY_SCHEMA,
        "qualification_receipt_schema": distill.TEACHER_QUALIFICATION_RECEIPT_SCHEMA,
        "classification_authority_schema": CLASSIFICATION_AUTHORITY_SCHEMA,
        "classification_decision_schema": CLASSIFICATION_DECISION_SCHEMA,
        "classification_ledger_schema": CLASSIFICATION_LEDGER_SCHEMA,
        "qualification_protocol_sha256": QUALIFICATION_PROTOCOL_SHA256,
        "classification_protocol_sha256": CLASSIFICATION_PROTOCOL_SHA256,
        "case_count": CASE_COUNT,
        "action_class_count": ACTION_CLASS_COUNT,
        "q_y_point_distillation_authorized_only_after_global_and_item_gates": True,
        "q_anchor_point_distillation_authorized": False,
        "candidate_materialization_is_never_mutated": True,
        "external_expected_pins_required": True,
        "synthetic_fixture_is_not_production_authority": True,
        "decoded_video_gate_authorized": False,
        "training_authorized": False,
        "local_only": True,
    }
    return {**unsigned, "receipt_digest": object_sha256(unsigned)}


__all__ = [
    "ACTION_CLASS_COUNT",
    "ACTION_WIDTH",
    "ActionTeacherQualificationError",
    "BENCHMARK_AUTHORITY_SCHEMA",
    "BENCHMARK_PAYLOAD_SCHEMA",
    "CANDIDATE_ITEM_SCHEMA",
    "CASE_COUNT",
    "CASE_SCHEMA",
    "CLASSIFICATION_AUTHORITY_SCHEMA",
    "CLASSIFICATION_DECISION_SCHEMA",
    "CLASSIFICATION_KINDS",
    "CLASSIFICATION_LEDGER_SCHEMA",
    "CLASSIFICATION_PROTOCOL_SHA256",
    "CLASSIFICATION_REQUEST_SCHEMA",
    "HARD_NEGATIVE_KINDS",
    "METRICS_SCHEMA",
    "PHASE_COUNT",
    "PROTOCOL_SCHEMA",
    "QUALIFICATION_BUNDLE_SCHEMA",
    "QUALIFICATION_PROTOCOL_SHA256",
    "Q_TENSOR_PAYLOAD_SCHEMA",
    "SPLIT_MANIFEST_SCHEMA",
    "SYNTHETIC_DIAGNOSTIC_SCHEMA",
    "canonical_json_bytes",
    "contract_receipt_v1",
    "encode_q_tensor_payload_v1",
    "evaluate_synthetic_diagnostic_v1",
    "issue_classification_ledger_v1",
    "issue_teacher_qualification_v1",
    "load_pinned_benchmark_files_v1",
    "load_pinned_canonical_json_v1",
    "object_sha256",
    "parse_canonical_json_bytes_v1",
    "publish_create_only_json_v1",
    "qualification_protocol_v1",
    "recompute_qualification_metrics_v1",
    "validate_benchmark_v1",
    "validate_classification_ledger_v1",
    "validate_teacher_qualification_bundle_v1",
]
