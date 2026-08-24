#!/usr/bin/env python3
"""Closed, fail-closed publication receipts for action-editing videos.

Visual media validity and action-editor serving authorization are intentionally
separate.  The latter additionally requires a pinned semantic action and
preservation verdict.  Every gate is bound to an authoritative exact81 decode
receipt and to the actual in-memory tensor bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Optional

import numpy as np

try:
    from .visual_validity_gate_v1 import (
        DEFAULT_THRESHOLDS,
        SCHEMA_VERSION as VISUAL_GATE_SCHEMA_VERSION,
        evaluate_visual_validity,
    )
except ImportError:  # pragma: no cover
    from visual_validity_gate_v1 import (  # type: ignore
        DEFAULT_THRESHOLDS,
        SCHEMA_VERSION as VISUAL_GATE_SCHEMA_VERSION,
        evaluate_visual_validity,
    )


SCHEMA_VERSION = "bernini-validated-inference-publication-v2"
DECODE_RECEIPT_SCHEMA = "bernini-exact81-authoritative-decode-receipt-v1"
SEMANTIC_VERDICT_SCHEMA = "bernini-action-preservation-semantic-verdict-v1"
FROZEN_QUALIFICATION_SCHEMA = "bernini-frozen-lkg-qualification-receipt-v1"
STRICT_REJECT = "strict-reject"
FROZEN_LAST_KNOWN_GOOD = "frozen-last-known-good"

_HEX = frozenset("0123456789abcdef")
_AUTHORITY_KEYS = frozenset({
    "authority_id", "authority_version", "implementation_sha256", "profile_sha256",
})
_CANDIDATE_KEYS = frozenset({
    "artifact_id", "role", "container_sha256", "decode_receipt_sha256",
    "semantic_verdict_receipt_sha256",
})
_FROZEN_KEYS = _CANDIDATE_KEYS | frozenset({"qualification_receipt_sha256"})
_DECODE_KEYS = frozenset({
    "schema_version", "artifact_id", "artifact_role", "container_sha256",
    "decoder_authority", "decoded_tensor", "receipt_sha256",
})
_TENSOR_KEYS = frozenset({"sha256", "dtype", "shape", "frame_count", "fps"})
_SEMANTIC_KEYS = frozenset({
    "schema_version", "artifact_id", "artifact_role", "container_sha256",
    "decoded_tensor_sha256", "action_instruction_sha256", "evaluation_authority",
    "full_trajectory_reviewed", "action_success", "preservation_success", "verdict",
    "receipt_sha256",
})
_QUALIFICATION_KEYS = frozenset({
    "schema_version", "artifact_id", "artifact_role", "container_sha256",
    "decoded_tensor_sha256", "qualification_authority", "full_trajectory_qualified",
    "qualified", "verdict", "receipt_sha256",
})


class PublicationInputError(ValueError):
    pass


def _closed(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise PublicationInputError(label + " must be a built-in dict")
    actual = frozenset(value)
    if actual != keys:
        raise PublicationInputError(
            "%s keys differ: missing=%s extra=%s"
            % (label, sorted(keys - actual), sorted(actual - keys))
        )
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise PublicationInputError(label + " must be a non-empty built-in string")
    return value


def _sha(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in _HEX for char in result):
        raise PublicationInputError(label + " must be lowercase SHA-256")
    return result


def _optional_sha(value: Any, label: str) -> Optional[str]:
    return None if value is None else _sha(value, label)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise PublicationInputError(label + " must be a built-in bool")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    return _canonical_digest(body)


def _check_receipt_digest(receipt: Mapping[str, Any], pin: str, label: str) -> None:
    embedded = _sha(receipt["receipt_sha256"], label + ".receipt_sha256")
    if embedded != _receipt_digest(receipt):
        raise PublicationInputError(label + " canonical receipt digest differs")
    if embedded != pin:
        raise PublicationInputError(label + " differs from external receipt pin")


def _authority(value: Any, label: str) -> dict[str, str]:
    item = _closed(value, _AUTHORITY_KEYS, label)
    return {
        "authority_id": _text(item["authority_id"], label + ".authority_id"),
        "authority_version": _text(item["authority_version"], label + ".authority_version"),
        "implementation_sha256": _sha(
            item["implementation_sha256"], label + ".implementation_sha256"
        ),
        "profile_sha256": _sha(item["profile_sha256"], label + ".profile_sha256"),
    }


def _trusted(embedded: Any, trusted: Any, label: str) -> dict[str, str]:
    first = _authority(embedded, label + ".embedded")
    second = _authority(trusted, label + ".trusted")
    if first != second:
        raise PublicationInputError(label + " differs from trusted authority pin")
    return first


def _artifact(value: Any, role: str, label: str) -> dict[str, Any]:
    fields = _FROZEN_KEYS if role == FROZEN_LAST_KNOWN_GOOD else _CANDIDATE_KEYS
    item = _closed(value, fields, label)
    if item["role"] != role:
        raise PublicationInputError(label + ".role differs")
    result = {
        "artifact_id": _text(item["artifact_id"], label + ".artifact_id"),
        "role": role,
        "container_sha256": _sha(item["container_sha256"], label + ".container_sha256"),
        "decode_receipt_sha256": _sha(
            item["decode_receipt_sha256"], label + ".decode_receipt_sha256"
        ),
        "semantic_verdict_receipt_sha256": _optional_sha(
            item["semantic_verdict_receipt_sha256"],
            label + ".semantic_verdict_receipt_sha256",
        ),
    }
    if role == FROZEN_LAST_KNOWN_GOOD:
        result["qualification_receipt_sha256"] = _sha(
            item["qualification_receipt_sha256"],
            label + ".qualification_receipt_sha256",
        )
    return result


def _decoded_binding(frames: Any) -> dict[str, Any]:
    if not isinstance(frames, np.ndarray) or frames.dtype.hasobject:
        raise PublicationInputError("frames must be a non-object numpy ndarray")
    shape = [int(x) for x in frames.shape]
    dtype = str(frames.dtype)
    header = json.dumps(
        {"dtype": dtype, "shape": shape}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(np.ascontiguousarray(frames)).cast("B"))
    return {
        "decoded_tensor_sha256": digest.hexdigest(),
        "dtype": dtype,
        "shape": shape,
        "frame_count": int(frames.shape[0]) if frames.ndim else 0,
    }


def _decode_receipt(
    value: Any,
    artifact: Mapping[str, Any],
    frames: Any,
    trusted_decoder: Any,
    label: str,
) -> dict[str, Any]:
    receipt = _closed(value, _DECODE_KEYS, label)
    if receipt["schema_version"] != DECODE_RECEIPT_SCHEMA:
        raise PublicationInputError(label + ".schema_version differs")
    joins = {
        "artifact_id": artifact["artifact_id"],
        "artifact_role": artifact["role"],
        "container_sha256": artifact["container_sha256"],
    }
    for key, expected in joins.items():
        if receipt[key] != expected:
            raise PublicationInputError(label + "." + key + " differs")
    _trusted(receipt["decoder_authority"], trusted_decoder, label + ".decoder_authority")
    tensor = _closed(receipt["decoded_tensor"], _TENSOR_KEYS, label + ".decoded_tensor")
    tensor_sha = _sha(tensor["sha256"], label + ".decoded_tensor.sha256")
    if type(tensor["shape"]) is not list or any(type(x) is not int or x <= 0 for x in tensor["shape"]):
        raise PublicationInputError(label + ".decoded_tensor.shape is invalid")
    if len(tensor["shape"]) != 4 or tensor["shape"][0] != 81 or tensor["shape"][-1] != 3:
        raise PublicationInputError(label + " must bind [81,H,W,3]")
    if type(tensor["frame_count"]) is not int or tensor["frame_count"] != 81:
        raise PublicationInputError(label + ".decoded_tensor.frame_count must be 81")
    dtype = _text(tensor["dtype"], label + ".decoded_tensor.dtype")
    fps_raw = tensor["fps"]
    if type(fps_raw) not in (int, float) or isinstance(fps_raw, bool):
        raise PublicationInputError(label + ".decoded_tensor.fps must be numeric")
    if not math.isfinite(float(fps_raw)) or float(fps_raw) <= 0:
        raise PublicationInputError(label + ".decoded_tensor.fps must be finite and positive")
    actual = _decoded_binding(frames)
    if (
        actual["decoded_tensor_sha256"] != tensor_sha
        or actual["shape"] != tensor["shape"]
        or actual["dtype"] != dtype
        or actual["frame_count"] != tensor["frame_count"]
    ):
        raise PublicationInputError(label + " does not bind the actual decoded frame tensor")
    _check_receipt_digest(receipt, artifact["decode_receipt_sha256"], label)
    return dict(receipt)


def _semantic_receipt(
    value: Optional[Any],
    trusted_authority: Optional[Any],
    artifact: Mapping[str, Any],
    decode: Mapping[str, Any],
    instruction_sha: str,
    label: str,
) -> Optional[dict[str, Any]]:
    pin = artifact["semantic_verdict_receipt_sha256"]
    if value is None:
        if pin is not None or trusted_authority is not None:
            raise PublicationInputError(label + " receipt/pin/authority must be supplied together")
        return None
    if pin is None or trusted_authority is None:
        raise PublicationInputError(label + " receipt/pin/authority must be supplied together")
    receipt = _closed(value, _SEMANTIC_KEYS, label)
    if receipt["schema_version"] != SEMANTIC_VERDICT_SCHEMA:
        raise PublicationInputError(label + ".schema_version differs")
    joins = {
        "artifact_id": artifact["artifact_id"],
        "artifact_role": artifact["role"],
        "container_sha256": artifact["container_sha256"],
        "decoded_tensor_sha256": decode["decoded_tensor"]["sha256"],
        "action_instruction_sha256": instruction_sha,
    }
    for key, expected in joins.items():
        if receipt[key] != expected:
            raise PublicationInputError(label + "." + key + " differs")
    _trusted(receipt["evaluation_authority"], trusted_authority, label + ".authority")
    full = _boolean(receipt["full_trajectory_reviewed"], label + ".full_trajectory_reviewed")
    action = _boolean(receipt["action_success"], label + ".action_success")
    preservation = _boolean(receipt["preservation_success"], label + ".preservation_success")
    if receipt["verdict"] not in ("PASS", "FAIL"):
        raise PublicationInputError(label + ".verdict must be PASS or FAIL")
    if (receipt["verdict"] == "PASS") is not (full and action and preservation):
        raise PublicationInputError(label + ".verdict is inconsistent with axes")
    _check_receipt_digest(receipt, pin, label)
    return dict(receipt)


def _qualification_receipt(
    value: Any,
    trusted_authority: Any,
    artifact: Mapping[str, Any],
    decode: Mapping[str, Any],
    decoder_authority: Any,
    label: str,
) -> dict[str, Any]:
    receipt = _closed(value, _QUALIFICATION_KEYS, label)
    if receipt["schema_version"] != FROZEN_QUALIFICATION_SCHEMA:
        raise PublicationInputError(label + ".schema_version differs")
    joins = {
        "artifact_id": artifact["artifact_id"],
        "artifact_role": artifact["role"],
        "container_sha256": artifact["container_sha256"],
        "decoded_tensor_sha256": decode["decoded_tensor"]["sha256"],
    }
    for key, expected in joins.items():
        if receipt[key] != expected:
            raise PublicationInputError(label + "." + key + " differs")
    qualification_authority = _trusted(
        receipt["qualification_authority"], trusted_authority, label + ".authority"
    )
    if qualification_authority["authority_id"] == _authority(
        decoder_authority, label + ".decoder_authority"
    )["authority_id"]:
        raise PublicationInputError(label + " authority is not independent of decoder")
    full = _boolean(receipt["full_trajectory_qualified"], label + ".full_trajectory_qualified")
    qualified = _boolean(receipt["qualified"], label + ".qualified")
    if receipt["verdict"] not in ("GO", "NO_GO"):
        raise PublicationInputError(label + ".verdict must be GO or NO_GO")
    if (receipt["verdict"] == "GO") is not (full and qualified):
        raise PublicationInputError(label + ".verdict is inconsistent with axes")
    _check_receipt_digest(receipt, artifact["qualification_receipt_sha256"], label)
    return dict(receipt)


def _gate(
    artifact: Mapping[str, Any],
    frames: Any,
    decode: Mapping[str, Any],
    semantic: Optional[Mapping[str, Any]],
    reference: Optional[Any],
    qualification: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    binding_before = _decoded_binding(frames)
    if binding_before["decoded_tensor_sha256"] != decode["decoded_tensor"]["sha256"]:
        raise PublicationInputError("decoded frames changed after decode-receipt validation")
    report = evaluate_visual_validity(
        frames, reference_frames=reference, thresholds=DEFAULT_THRESHOLDS
    )
    binding_after = _decoded_binding(frames)
    if binding_after != binding_before:
        raise PublicationInputError("decoded frames changed while the visual gate was running")
    return {
        "artifact": dict(artifact),
        "decode_receipt": dict(decode),
        "decoded_binding": binding_after,
        "visual_gate_report": report,
        "visual_gate_report_sha256": _canonical_digest(report),
        "semantic_verdict": None if semantic is None else dict(semantic),
        "qualification_receipt": None if qualification is None else dict(qualification),
    }


def _finalize(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    json.dumps(receipt, allow_nan=False, ensure_ascii=False, sort_keys=True)
    return receipt


def decide_serving_publication(
    candidate_frames: Any,
    candidate_artifact: Mapping[str, Any],
    *,
    candidate_decode_receipt: Mapping[str, Any],
    trusted_decoder_authority: Mapping[str, Any],
    action_instruction_sha256: str,
    candidate_semantic_verdict: Optional[Mapping[str, Any]] = None,
    trusted_candidate_semantic_authority: Optional[Mapping[str, Any]] = None,
    reference_frames: Optional[Any] = None,
    fallback_policy: str = STRICT_REJECT,
    frozen_frames: Optional[Any] = None,
    frozen_artifact: Optional[Mapping[str, Any]] = None,
    frozen_decode_receipt: Optional[Mapping[str, Any]] = None,
    frozen_qualification_receipt: Optional[Mapping[str, Any]] = None,
    trusted_frozen_qualification_authority: Optional[Mapping[str, Any]] = None,
    frozen_semantic_verdict: Optional[Mapping[str, Any]] = None,
    trusted_frozen_semantic_authority: Optional[Mapping[str, Any]] = None,
    frozen_reference_frames: Optional[Any] = None,
) -> dict[str, Any]:
    """Authorize only a byte-bound, visual, action, and preservation PASS."""

    instruction_sha = _sha(action_instruction_sha256, "action_instruction_sha256")
    decoder_authority = _authority(trusted_decoder_authority, "trusted_decoder_authority")
    candidate_pin = _artifact(candidate_artifact, "candidate", "candidate_artifact")
    candidate_decode = _decode_receipt(
        candidate_decode_receipt, candidate_pin, candidate_frames,
        decoder_authority, "candidate_decode_receipt",
    )
    candidate_semantic = _semantic_receipt(
        candidate_semantic_verdict, trusted_candidate_semantic_authority,
        candidate_pin, candidate_decode, instruction_sha, "candidate_semantic_verdict",
    )
    if fallback_policy not in (STRICT_REJECT, FROZEN_LAST_KNOWN_GOOD):
        raise PublicationInputError("unknown fallback policy")
    frozen_values = (
        frozen_frames, frozen_artifact, frozen_decode_receipt,
        frozen_qualification_receipt, trusted_frozen_qualification_authority,
        frozen_semantic_verdict, trusted_frozen_semantic_authority,
        frozen_reference_frames,
    )
    if fallback_policy == STRICT_REJECT and any(x is not None for x in frozen_values):
        raise PublicationInputError("strict-reject forbids frozen fallback inputs")

    fallback = None
    frozen_semantic = None
    frozen_qualification = None
    if fallback_policy == FROZEN_LAST_KNOWN_GOOD:
        required = frozen_values[:5]
        if any(x is None for x in required):
            raise PublicationInputError(
                "frozen fallback requires frames/artifact/decode receipt/qualification receipt/authority"
            )
        frozen_pin = _artifact(frozen_artifact, FROZEN_LAST_KNOWN_GOOD, "frozen_artifact")
        if (
            frozen_pin["artifact_id"] == candidate_pin["artifact_id"]
            or frozen_pin["container_sha256"] == candidate_pin["container_sha256"]
        ):
            raise PublicationInputError("candidate and fallback are not independent artifacts")
        frozen_decode = _decode_receipt(
            frozen_decode_receipt, frozen_pin, frozen_frames,
            decoder_authority, "frozen_decode_receipt",
        )
        frozen_qualification = _qualification_receipt(
            frozen_qualification_receipt, trusted_frozen_qualification_authority,
            frozen_pin, frozen_decode, decoder_authority, "frozen_qualification_receipt",
        )
        frozen_semantic = _semantic_receipt(
            frozen_semantic_verdict, trusted_frozen_semantic_authority,
            frozen_pin, frozen_decode, instruction_sha, "frozen_semantic_verdict",
        )
        fallback = _gate(
            frozen_pin, frozen_frames, frozen_decode, frozen_semantic,
            frozen_reference_frames, frozen_qualification,
        )

    candidate = _gate(
        candidate_pin, candidate_frames, candidate_decode,
        candidate_semantic, reference_frames,
    )
    candidate_visual = candidate["visual_gate_report"]["publishable"] is True
    candidate_semantic_pass = (
        candidate_semantic is not None and candidate_semantic["verdict"] == "PASS"
    )
    fallback_visual = (
        fallback is not None and fallback["visual_gate_report"]["publishable"] is True
    )
    fallback_qualified = (
        frozen_qualification is not None and frozen_qualification["verdict"] == "GO"
    )
    fallback_semantic_pass = (
        frozen_semantic is not None and frozen_semantic["verdict"] == "PASS"
    )

    selected = None
    if candidate_visual and candidate_semantic_pass:
        selected = candidate["artifact"]
        decision = "authorize_candidate_for_action_editor_serving"
    elif fallback_visual and fallback_qualified and fallback_semantic_pass:
        selected = fallback["artifact"]
        decision = "authorize_frozen_last_known_good_for_action_editor_serving"
    else:
        decision = "no_action_editor_serving_authorization"

    visual_artifact = None
    if candidate_visual:
        visual_artifact = candidate["artifact"]
    elif fallback_visual and fallback_qualified:
        visual_artifact = fallback["artifact"]

    blockers = []
    if selected is None:
        if not candidate_visual:
            blockers.append("candidate_visual_media_gate_failed")
        if candidate_semantic is None:
            blockers.append("candidate_semantic_verdict_missing")
        elif not candidate_semantic_pass:
            blockers.append("candidate_action_or_preservation_failed")
        if fallback is not None:
            if not fallback_visual:
                blockers.append("fallback_visual_media_gate_failed")
            if not fallback_qualified:
                blockers.append("fallback_not_independently_qualified")
            if frozen_semantic is None:
                blockers.append("fallback_semantic_verdict_missing")
            elif not fallback_semantic_pass:
                blockers.append("fallback_action_or_preservation_failed")

    return _finalize({
        "schema_version": SCHEMA_VERSION,
        "mode": "serving",
        "decision": decision,
        "visual_media_publishable": visual_artifact is not None,
        "visual_media_artifact": visual_artifact,
        "action_editor_serving_authorized": selected is not None,
        "selected_artifact": selected,
        "authorization_blockers": blockers,
        "action_instruction_sha256": instruction_sha,
        "fallback_policy": fallback_policy,
        "candidate": candidate,
        "fallback": fallback,
        "contract": {
            "visual_gate_schema_version": VISUAL_GATE_SCHEMA_VERSION,
            "decode_receipt_schema_version": DECODE_RECEIPT_SCHEMA,
            "semantic_verdict_schema_version": SEMANTIC_VERDICT_SCHEMA,
            "frozen_qualification_schema_version": FROZEN_QUALIFICATION_SCHEMA,
            "actual_decoded_tensor_digest_recomputed": True,
            "visual_media_gate_is_not_action_editor_success": True,
            "action_editor_requires_action_and_preservation_pass": True,
            "fallback_requires_independent_qualification": True,
            "scientific_review_must_not_substitute_fallback": True,
        },
    })


def build_scientific_review_receipt(
    candidate_frames: Any,
    candidate_artifact: Mapping[str, Any],
    *,
    candidate_decode_receipt: Mapping[str, Any],
    trusted_decoder_authority: Mapping[str, Any],
    reference_frames: Optional[Any] = None,
) -> dict[str, Any]:
    """Display the byte-bound candidate itself; never substitute a fallback."""

    decoder_authority = _authority(trusted_decoder_authority, "trusted_decoder_authority")
    artifact = _artifact(candidate_artifact, "candidate", "candidate_artifact")
    decode = _decode_receipt(
        candidate_decode_receipt, artifact, candidate_frames,
        decoder_authority, "candidate_decode_receipt",
    )
    candidate = _gate(artifact, candidate_frames, decode, None, reference_frames)
    visual = candidate["visual_gate_report"]["publishable"] is True
    return _finalize({
        "schema_version": SCHEMA_VERSION,
        "mode": "scientific-review",
        "decision": "display_candidate" if visual else "display_candidate_with_failure_label",
        "visual_media_publishable": visual,
        "action_editor_serving_authorized": False,
        "review_display_artifact": candidate["artifact"],
        "review_failure_label_required": not visual,
        "candidate": candidate,
        "fallback": None,
        "contract": {
            "visual_gate_schema_version": VISUAL_GATE_SCHEMA_VERSION,
            "decode_receipt_schema_version": DECODE_RECEIPT_SCHEMA,
            "actual_decoded_tensor_digest_recomputed": True,
            "candidate_evidence_must_remain_visible": True,
            "fallback_substitution_allowed": False,
            "visual_media_gate_is_not_action_editor_success": True,
        },
    })


__all__ = [
    "DECODE_RECEIPT_SCHEMA", "FROZEN_LAST_KNOWN_GOOD",
    "FROZEN_QUALIFICATION_SCHEMA", "PublicationInputError", "SCHEMA_VERSION",
    "SEMANTIC_VERDICT_SCHEMA", "STRICT_REJECT",
    "build_scientific_review_receipt", "decide_serving_publication",
]
