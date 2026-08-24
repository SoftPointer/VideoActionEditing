"""Source-anchored signed releases for full-motion Wan2.2 generation.

One release signs an ordered root generation manifest.  The release may then
authorize any *contiguous* eight-row JSONL slice whose rows are byte-for-byte
members of that signed order.  It does not authorize arbitrary subsets,
reordered rows, legacy action-anchor rows, or unsigned authorization booleans.

Every root row is revalidated against the full-motion contract and
deterministic instruction compiler before it is signed.  Verification repeats
that validation and binds the source video, exact-I0 lossless anchor, temporal
geometry, ``motion_spec``, compiled instruction, and sole executable
``edit_instruction``.  OpenSSH ``sshsig`` is used so the verifier has no
network or Python cryptography dependency.

The dedicated public key and fingerprint are frozen in source.  The private
key is intentionally absent from the repository and AUH; callers must never
source trust anchors from environment variables or a release envelope.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .goku_full_motion_contract import (
    MODEL_OUTPUT_CANONICALIZATION_POLICY,
    MODEL_OUTPUT_CANONICALIZATION_RECEIPT_SCHEMA,
    build_contract,
    object_sha256,
    validate_contract_binding,
    validate_coverage_critic,
    validate_source_census,
    validate_source_inventory_alignment,
    validate_target_plan,
)
from .goku_full_motion_instruction import (
    compile_full_motion_instruction,
    validate_compiled_instruction,
)


# The envelope, payload, request, signature namespace, and dispatch mode retain
# the established release-v3 transport boundary.  Its root manifest is now the
# stricter generation-v6 two-stage coverage-authority contract, so older rows fail
# closed before signing or verification.
RELEASE_SCHEMA = "motive-wan22-full-motion-signed-root-release-v3"
RELEASE_PAYLOAD_SCHEMA = "motive-wan22-full-motion-root-release-payload-v3"
RELEASE_REQUEST_SCHEMA = "motive-wan22-full-motion-release-request-v3"
GENERATION_MANIFEST_SCHEMA = "motive-goku-full-motion-generation-v6"
MOTION_SPEC_SCHEMA = "motive-goku-full-motion-generation-spec-v6"
TEMPORAL_GEOMETRY_SCHEMA = "motive-goku-full-motion-temporal-geometry-v1"
QWEN_EVIDENCE_SCHEMA = "motive-goku-full-motion-qwen-evidence-v6"
FINALIZATION_ROW_SCHEMA = "motive-goku-full-motion-finalization-row-v1"
FINALIZATION_POLICY = "full-motion-all-source-dynamics-v1"

SIGNATURE_NAMESPACE = "motive-wan22-full-motion-root-release-v3"
SIGNER_PRINCIPAL = "motive-wan22-full-motion-release"
SIGNER_KEY_FINGERPRINT = (
    "SHA256:A6zKKVBr6MSG29PO5J7A91aJYKcORNOkidofuI+jf6Y"
)
SIGNER_PUBLIC_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIKe6Q+9i1y9DZE5n6PZNXFJw/YQBEtojl3ClolirGDlO"
)

AUTHORIZATION_MODE = "sshsig_full_motion_root_contiguous8_release_v3"
CONTIGUOUS_SHARD_ROWS = 8
EXPECTED_FRAME_COUNT = 81
EXPECTED_FRAME_RATE = "25/1"
EXPECTED_TIMELINE_SECONDS = 3.2

PROMPT_POLICY = {
    "executable_field": "edit_instruction",
    "byte_exact_compiled_instruction_value": True,
    "motion_spec_prose_executable": False,
    "legacy_caption_or_prompt_executable": False,
}
TEMPORAL_POLICY = {
    "source_frame_count": EXPECTED_FRAME_COUNT,
    "target_frame_count": EXPECTED_FRAME_COUNT,
    "source_frame_rate": EXPECTED_FRAME_RATE,
    "target_frame_rate": EXPECTED_FRAME_RATE,
    "first_frame_is_exact_bound_anchor": True,
}

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")

_ROW_KEYS = {
    "schema_version",
    "iid",
    "group_id",
    "family",
    "source_video",
    "resolved_source_video",
    "anchor_image",
    "resolved_anchor_image",
    "source_video_sha256",
    "anchor_sha256",
    "selected_media_evidence",
    "selected_media_evidence_sha256",
    "strict_temporal_geometry",
    "edit_instruction",
    "edit_instruction_sha256",
    "motion_spec",
    "motion_spec_sha256",
    "qwen_evidence",
    "full_motion_finalization",
    "action_change_substantive",
    "manifest_role",
    "human_review_status",
    "generation_authorized",
    "production_eligible",
    "approval",
    "authorization_interface_available",
    "annotation_source",
    "human_reviewed",
}
_MOTION_SPEC_KEYS = {
    "schema_version",
    "change_region_proposals",
    "coverage_authority",
    "i0_grounding",
    "source_census",
    "secondary_source_census",
    "source_inventory_alignment",
    "coverage_authority_alignment",
    "target_plan",
    "compiled_instruction",
    "coverage_critic",
    "full_motion_contract",
    "qwen_result_digest",
    "qwen_provenance_digest",
}
_TEMPORAL_KEYS = {
    "schema_version",
    "source_frame_count",
    "source_frame_rate",
    "source_timeline_span_seconds",
    "target_frame_count",
    "target_frame_rate",
    "target_timeline_span_seconds",
    "requires_exact_frame_count_and_rate_match",
}
_QWEN_EVIDENCE_KEYS = {
    "schema_version",
    "record_schema_version",
    "input_digest",
    "result_digest",
    "provenance_digest",
    "config_digest",
    "run_config_digest",
    "implementation_digest",
    "visual_input_digest",
    "media_verification",
    "hard_gate",
    "change_region_proposals_digest",
    "coverage_authority_inventory_prompt_digest",
    "coverage_authority_inventory_visual_input_digest",
    "coverage_authority_inventory_digest",
    "coverage_authority_assignments_prompt_digest",
    "coverage_authority_assignments_visual_input_digest",
    "coverage_authority_assignments_digest",
    "coverage_authority_digest",
    "coverage_authority_alignment_digest",
    "i0_grounding_digest",
    "source_census_canonicalization",
    "source_census_canonicalization_digest",
    "source_census_digest",
    "secondary_source_census_canonicalization",
    "secondary_source_census_canonicalization_digest",
    "secondary_source_census_digest",
    "source_inventory_alignment_digest",
    "target_plan_canonicalization",
    "target_plan_canonicalization_digest",
    "target_plan_digest",
    "compiled_instruction_digest",
    "full_motion_contract_digest",
    "coverage_critic_digest",
    "shard_index",
    "num_shards",
    "receipt_digest",
    "receipt_sha256",
    "output_sha256",
    "model_path",
    "model_revision",
    "transformers_version",
    "qwen_record_payload",
}
_FINALIZATION_KEYS = {
    "schema_version",
    "policy_version",
    "candidate_rank",
    "review_rank",
    "selection_bucket",
    "dynamic_unit_count",
    "target_action_signatures",
    "family",
    "required_canary",
    "qwen_shard_index",
    "qwen_receipt_digest",
}


class Wan22FullMotionReleaseError(RuntimeError):
    """A full-motion release or requested shard violates its trust boundary."""


def _reject_constant(value: str) -> None:
    raise Wan22FullMotionReleaseError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Wan22FullMotionReleaseError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Wan22FullMotionReleaseError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, Wan22FullMotionReleaseError):
            raise
        raise Wan22FullMotionReleaseError(
            f"{context} is not strict JSON: {error}"
        ) from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Wan22FullMotionReleaseError(
            f"value is not finite canonical JSON: {error}"
        ) from error


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _ordered_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        b"".join(value.encode("utf-8") + b"\n" for value in values)
    ).hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    if set(value) != expected:
        raise Wan22FullMotionReleaseError(
            f"{context} keys differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Wan22FullMotionReleaseError(f"{context} must be an object")
    return value


def _string(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise Wan22FullMotionReleaseError(
            f"{context} must be one canonical non-empty string"
        )
    return value


def _sha(value: Any, *, context: str) -> str:
    text = _string(value, context=context)
    if _SHA_RE.fullmatch(text) is None:
        raise Wan22FullMotionReleaseError(
            f"{context} must be a lowercase SHA-256"
        )
    return text


def _validate_canonicalization_receipt(
    value: Any,
    *,
    artifact_kind: str,
    canonical: Mapping[str, Any],
    expected_context: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    """Independently close one projected Qwen v6 canonicalization receipt."""

    receipt = _mapping(value, context=context)
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "policy",
        "semantic_repair",
        "context",
        "raw_sha256",
        "canonical_sha256",
        "normalized_field_paths",
        "changed_field_paths",
        "receipt_sha256",
    }
    _exact_keys(receipt, expected_keys, context=context)
    if (
        receipt.get("schema_version")
        != MODEL_OUTPUT_CANONICALIZATION_RECEIPT_SCHEMA
        or receipt.get("artifact_kind") != artifact_kind
        or receipt.get("policy") != MODEL_OUTPUT_CANONICALIZATION_POLICY
        or receipt.get("semantic_repair") is not False
        or receipt.get("context") != dict(expected_context)
        or receipt.get("canonical_sha256") != object_sha256(canonical)
    ):
        raise Wan22FullMotionReleaseError(
            f"{context} canonical artifact binding differs"
        )
    _sha(receipt.get("raw_sha256"), context=f"{context}.raw_sha256")
    _sha(
        receipt.get("canonical_sha256"),
        context=f"{context}.canonical_sha256",
    )
    normalized = receipt.get("normalized_field_paths")
    changed = receipt.get("changed_field_paths")
    if (
        not isinstance(normalized, list)
        or not normalized
        or any(
            not isinstance(path, str) or not path or path != path.strip()
            for path in normalized
        )
        or len(set(normalized)) != len(normalized)
        or not isinstance(changed, list)
        or any(
            not isinstance(path, str) or not path or path != path.strip()
            for path in changed
        )
        or len(set(changed)) != len(changed)
        or not set(changed).issubset(normalized)
    ):
        raise Wan22FullMotionReleaseError(
            f"{context} normalized/changed path closure differs"
        )
    receipt_payload = dict(receipt)
    receipt_sha = receipt_payload.pop("receipt_sha256")
    if (
        _sha(receipt_sha, context=f"{context}.receipt_sha256")
        != object_sha256(receipt_payload)
    ):
        raise Wan22FullMotionReleaseError(f"{context} receipt SHA differs")
    return dict(receipt)


def _safe_iid(value: Any, *, context: str) -> str:
    iid = _string(value, context=context)
    if _IID_RE.fullmatch(iid) is None or iid in {".", ".."}:
        raise Wan22FullMotionReleaseError(f"{context} is not a safe IID")
    return iid


def _stable_regular_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise Wan22FullMotionReleaseError(f"{context} path must be absolute")
    if expanded.is_symlink() or not expanded.is_file():
        raise Wan22FullMotionReleaseError(
            f"{context} must be a non-symlink regular file: {expanded}"
        )
    return expanded.resolve(strict=True)


def _strict_json_file(path: Path, *, context: str) -> tuple[dict[str, Any], bytes, Path]:
    resolved = _stable_regular_file(path, context=context)
    raw = resolved.read_bytes()
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise Wan22FullMotionReleaseError(f"{context} must contain one object")
    return value, raw, resolved


def _strict_jsonl(
    path: Path, *, context: str
) -> tuple[list[dict[str, Any]], list[bytes], bytes, Path]:
    resolved = _stable_regular_file(path, context=context)
    raw = resolved.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise Wan22FullMotionReleaseError(
            f"{context} must be non-empty and newline-terminated"
        )
    rows: list[dict[str, Any]] = []
    lines: list[bytes] = []
    for line_number, bare in enumerate(raw.splitlines(), start=1):
        if not bare:
            raise Wan22FullMotionReleaseError(
                f"{context}:{line_number} is blank"
            )
        value = _parse_json(bare, context=f"{context}:{line_number}")
        if not isinstance(value, dict):
            raise Wan22FullMotionReleaseError(
                f"{context}:{line_number} is not an object"
            )
        canonical = _canonical_bytes(value)
        if canonical != bare:
            raise Wan22FullMotionReleaseError(
                f"{context}:{line_number} is not canonical JSON"
            )
        rows.append(value)
        lines.append(canonical + b"\n")
    return rows, lines, raw, resolved


def _implementation_binding() -> dict[str, str]:
    from . import goku_full_motion_contract as contract_module
    from . import goku_full_motion_finalize as finalize_module
    from . import goku_full_motion_instruction as instruction_module
    from . import goku_full_motion_qwen as qwen_module

    return {
        "release_module_sha256": _file_digest(Path(__file__).resolve(strict=True)),
        "contract_module_sha256": _file_digest(
            Path(contract_module.__file__).resolve(strict=True)
        ),
        "instruction_module_sha256": _file_digest(
            Path(instruction_module.__file__).resolve(strict=True)
        ),
        "finalize_module_sha256": _file_digest(
            Path(finalize_module.__file__).resolve(strict=True)
        ),
        "qwen_module_sha256": _file_digest(
            Path(qwen_module.__file__).resolve(strict=True)
        ),
    }


def _validate_temporal_geometry(value: Any) -> dict[str, Any]:
    geometry = _mapping(value, context="strict_temporal_geometry")
    _exact_keys(geometry, _TEMPORAL_KEYS, context="strict_temporal_geometry")
    expected = {
        "schema_version": TEMPORAL_GEOMETRY_SCHEMA,
        "source_frame_count": EXPECTED_FRAME_COUNT,
        "source_frame_rate": EXPECTED_FRAME_RATE,
        "source_timeline_span_seconds": EXPECTED_TIMELINE_SECONDS,
        "target_frame_count": EXPECTED_FRAME_COUNT,
        "target_frame_rate": EXPECTED_FRAME_RATE,
        "target_timeline_span_seconds": EXPECTED_TIMELINE_SECONDS,
        "requires_exact_frame_count_and_rate_match": True,
    }
    if dict(geometry) != expected:
        raise Wan22FullMotionReleaseError(
            "strict temporal geometry must be exactly 81 frames at 25/1"
        )
    return dict(geometry)


def _validate_qwen_evidence(
    value: Any,
    *,
    generation_row: Mapping[str, Any],
    motion_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    compiled: Mapping[str, Any],
    coverage: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _mapping(value, context="qwen_evidence")
    _exact_keys(evidence, _QWEN_EVIDENCE_KEYS, context="qwen_evidence")
    if evidence.get("schema_version") != QWEN_EVIDENCE_SCHEMA:
        raise Wan22FullMotionReleaseError("qwen_evidence schema differs")
    for field in (
        "input_digest",
        "result_digest",
        "provenance_digest",
        "config_digest",
        "run_config_digest",
        "implementation_digest",
        "visual_input_digest",
        "change_region_proposals_digest",
        "coverage_authority_inventory_prompt_digest",
        "coverage_authority_inventory_visual_input_digest",
        "coverage_authority_inventory_digest",
        "coverage_authority_assignments_prompt_digest",
        "coverage_authority_assignments_visual_input_digest",
        "coverage_authority_assignments_digest",
        "coverage_authority_digest",
        "coverage_authority_alignment_digest",
        "i0_grounding_digest",
        "source_census_canonicalization_digest",
        "source_census_digest",
        "secondary_source_census_canonicalization_digest",
        "secondary_source_census_digest",
        "source_inventory_alignment_digest",
        "target_plan_canonicalization_digest",
        "target_plan_digest",
        "compiled_instruction_digest",
        "full_motion_contract_digest",
        "coverage_critic_digest",
        "receipt_digest",
        "receipt_sha256",
        "output_sha256",
    ):
        _sha(evidence.get(field), context=f"qwen_evidence.{field}")
    from . import goku_full_motion_qwen as qwen_module
    from .goku_full_motion_qwen import (
        build_hard_gate,
        validate_change_region_proposals,
        validate_coverage_authority,
        validate_coverage_authority_alignment,
        validate_i0_grounding,
        validate_source_census_i0_binding,
    )

    try:
        i0_grounding = validate_i0_grounding(
            motion_spec.get("i0_grounding"), expected_iid=str(source["iid"])
        )
        source = validate_source_census_i0_binding(source, i0_grounding)
        secondary_source = validate_source_census(
            motion_spec.get("secondary_source_census")
        )
        secondary_source = validate_source_census_i0_binding(
            secondary_source, i0_grounding
        )
        inventory_alignment = validate_source_inventory_alignment(
            motion_spec.get("source_inventory_alignment"),
            primary=source,
            secondary=secondary_source,
        )
        change_region_proposals = validate_change_region_proposals(
            motion_spec.get("change_region_proposals"),
            expected_iid=str(source["iid"]),
        )
        coverage_authority = validate_coverage_authority(
            motion_spec.get("coverage_authority"),
            expected_iid=str(source["iid"]),
            change_region_proposals=change_region_proposals,
        )
        coverage_authority_alignment = validate_coverage_authority_alignment(
            motion_spec.get("coverage_authority_alignment"),
            coverage_authority=coverage_authority,
            change_region_proposals=change_region_proposals,
            i0_grounding=i0_grounding,
            primary=source,
            secondary=secondary_source,
            source_inventory_alignment=inventory_alignment,
        )
    except Exception as error:
        raise Wan22FullMotionReleaseError(
            f"Qwen v6 coverage-authority/exact-I0 closure differs: {error}"
        ) from error
    proposals_sha = object_sha256(change_region_proposals)
    authority_inventory_sha = object_sha256(
        coverage_authority["inventory"]
    )
    authority_assignments_sha = object_sha256(
        coverage_authority["assignments"]
    )
    authority_sha = object_sha256(coverage_authority)
    authority_alignment_sha = object_sha256(coverage_authority_alignment)
    i0_grounding_sha = object_sha256(i0_grounding)
    primary_sha = object_sha256(source)
    secondary_sha = object_sha256(secondary_source)
    alignment_sha = object_sha256(inventory_alignment)
    source_receipt = _validate_canonicalization_receipt(
        evidence.get("source_census_canonicalization"),
        artifact_kind="source_census",
        canonical=source,
        expected_context={"expected_iid": source["iid"]},
        context="qwen_evidence.source_census_canonicalization",
    )
    secondary_receipt = _validate_canonicalization_receipt(
        evidence.get("secondary_source_census_canonicalization"),
        artifact_kind="source_census",
        canonical=secondary_source,
        expected_context={"expected_iid": source["iid"]},
        context="qwen_evidence.secondary_source_census_canonicalization",
    )
    target_receipt = _validate_canonicalization_receipt(
        evidence.get("target_plan_canonicalization"),
        artifact_kind="target_plan",
        canonical=plan,
        expected_context={
            "iid": source["iid"],
            "source_census_sha256": primary_sha,
        },
        context="qwen_evidence.target_plan_canonicalization",
    )
    source_receipt_sha = object_sha256(source_receipt)
    secondary_receipt_sha = object_sha256(secondary_receipt)
    target_receipt_sha = object_sha256(target_receipt)
    expected_digests = {
        "change_region_proposals_digest": proposals_sha,
        "coverage_authority_inventory_digest": authority_inventory_sha,
        "coverage_authority_assignments_digest": authority_assignments_sha,
        "coverage_authority_digest": authority_sha,
        "coverage_authority_alignment_digest": authority_alignment_sha,
        "i0_grounding_digest": i0_grounding_sha,
        "source_census_canonicalization_digest": source_receipt_sha,
        "source_census_digest": primary_sha,
        "secondary_source_census_canonicalization_digest": (
            secondary_receipt_sha
        ),
        "secondary_source_census_digest": secondary_sha,
        "source_inventory_alignment_digest": alignment_sha,
        "target_plan_canonicalization_digest": target_receipt_sha,
        "target_plan_digest": object_sha256(plan),
        "compiled_instruction_digest": object_sha256(compiled),
        "full_motion_contract_digest": object_sha256(contract),
        "coverage_critic_digest": object_sha256(coverage),
    }
    for field, expected in expected_digests.items():
        if evidence.get(field) != expected:
            raise Wan22FullMotionReleaseError(
                f"qwen_evidence.{field} differs from motion_spec"
            )
    if (
        type(evidence.get("shard_index")) is not int
        or type(evidence.get("num_shards")) is not int
        or evidence["num_shards"] != 8
        or not 0 <= evidence["shard_index"] < evidence["num_shards"]
    ):
        raise Wan22FullMotionReleaseError("qwen_evidence shard binding differs")
    media_verification = evidence.get("media_verification")
    if not isinstance(media_verification, Mapping):
        raise Wan22FullMotionReleaseError(
            "qwen_evidence.media_verification must be an object"
        )
    try:
        expected_hard_gate = build_hard_gate(
            change_region_proposals=change_region_proposals,
            coverage_authority=coverage_authority,
            coverage_authority_alignment=coverage_authority_alignment,
            i0_grounding=i0_grounding,
            source_census=source,
            source_census_canonicalization=source_receipt,
            secondary_source_census=secondary_source,
            secondary_source_census_canonicalization=secondary_receipt,
            source_inventory_alignment=inventory_alignment,
            target_plan=plan,
            target_plan_canonicalization=target_receipt,
            compiled_instruction=compiled,
            coverage_critic=coverage,
        )
    except Exception as error:
        raise Wan22FullMotionReleaseError(
            f"Qwen v6 hard-gate reconstruction failed: {error}"
        ) from error
    if (
        evidence.get("hard_gate") != expected_hard_gate
        or expected_hard_gate.get("decision") != "pass"
        or expected_hard_gate.get("risk_codes") != []
    ):
        raise Wan22FullMotionReleaseError(
            "qwen v6 coverage-authority hard-gate binding is not pass"
        )

    # Rebuild the exact canonical success result from validated projections.
    # The complete projected record is retained so provenance can likewise be
    # recomputed; neither digest is accepted merely because two fields agree.
    expected_result_payload = {
        "change_region_proposals": change_region_proposals,
        "coverage_authority": coverage_authority,
        "i0_grounding": i0_grounding,
        "source_census": source,
        "source_census_canonicalization": source_receipt,
        "secondary_source_census": secondary_source,
        "secondary_source_census_canonicalization": secondary_receipt,
        "source_inventory_alignment": inventory_alignment,
        "coverage_authority_alignment": coverage_authority_alignment,
        "target_plan": plan,
        "target_plan_canonicalization": target_receipt,
        "compiled_instruction": compiled,
        "full_motion_contract": contract,
        "coverage_critic": coverage,
        "hard_gate": expected_hard_gate,
        "pipeline_stage": "coverage_critic",
        "pipeline_decision": "pass",
    }
    record_value = evidence.get("qwen_record_payload")
    if (
        not isinstance(record_value, Mapping)
        or set(record_value) != qwen_module._RECORD_KEYS
    ):
        raise Wan22FullMotionReleaseError(
            "qwen_record_payload is not the closed Qwen v6 record"
        )
    record = dict(record_value)
    scalar_bindings = {
        "schema_version": evidence.get("record_schema_version"),
        "iid": generation_row.get("iid"),
        "group_id": generation_row.get("group_id"),
        "family": generation_row.get("family"),
        "status": "ok",
        "error_type": None,
        "error": None,
        "input_digest": evidence.get("input_digest"),
        "config_digest": evidence.get("config_digest"),
        "run_config_digest": evidence.get("run_config_digest"),
        "implementation_digest": evidence.get("implementation_digest"),
        "model_path": evidence.get("model_path"),
        "model_revision": evidence.get("model_revision"),
        "transformers_version": evidence.get("transformers_version"),
        "shard_index": evidence.get("shard_index"),
        "num_shards": evidence.get("num_shards"),
        "resolved_src_video": generation_row.get("resolved_source_video"),
        "resolved_anchor_image": generation_row.get("resolved_anchor_image"),
        "visual_input_digest": evidence.get("visual_input_digest"),
        "failure_stage": None,
        "pipeline_stage": "coverage_critic",
        "pipeline_decision": "pass",
        "result_digest": evidence.get("result_digest"),
        "provenance_digest": evidence.get("provenance_digest"),
    }
    if any(
        record.get(field) != expected
        for field, expected in scalar_bindings.items()
    ):
        raise Wan22FullMotionReleaseError(
            "qwen_record_payload scalar provenance binding differs"
        )
    semantic_bindings = {
        "media_verification": media_verification,
        "change_region_proposals": change_region_proposals,
        "coverage_authority": coverage_authority,
        "i0_grounding": i0_grounding,
        "source_census": source,
        "source_census_canonicalization": source_receipt,
        "secondary_source_census": secondary_source,
        "secondary_source_census_canonicalization": secondary_receipt,
        "source_inventory_alignment": inventory_alignment,
        "coverage_authority_alignment": coverage_authority_alignment,
        "target_plan": plan,
        "target_plan_canonicalization": target_receipt,
        "compiled_instruction": compiled,
        "full_motion_contract": contract,
        "coverage_critic": coverage,
        "hard_gate": expected_hard_gate,
    }
    if any(
        record.get(field) != expected
        for field, expected in semantic_bindings.items()
    ):
        raise Wan22FullMotionReleaseError(
            "qwen_record_payload semantic artifact binding differs"
        )
    digest_fields = (
        "change_region_proposals_digest",
        "coverage_authority_inventory_prompt_digest",
        "coverage_authority_inventory_visual_input_digest",
        "coverage_authority_inventory_digest",
        "coverage_authority_assignments_prompt_digest",
        "coverage_authority_assignments_visual_input_digest",
        "coverage_authority_assignments_digest",
        "coverage_authority_digest",
        "coverage_authority_alignment_digest",
        "i0_grounding_digest",
        "source_census_canonicalization_digest",
        "source_census_digest",
        "secondary_source_census_canonicalization_digest",
        "secondary_source_census_digest",
        "source_inventory_alignment_digest",
        "target_plan_canonicalization_digest",
        "target_plan_digest",
        "compiled_instruction_digest",
        "full_motion_contract_digest",
        "coverage_critic_digest",
    )
    if any(
        record.get(field) != evidence.get(field) for field in digest_fields
    ):
        raise Wan22FullMotionReleaseError(
            "qwen_record_payload artifact digest binding differs"
        )
    try:
        inventory_raw = qwen_module.coverage_authority_validated_raw(
            record, stage="coverage_authority_inventory"
        )
        validated_inventory, inventory_validated_from = (
            qwen_module._validate_original_a0_output(
                stage="coverage_authority_inventory",
                original_raw=inventory_raw,
                validator=lambda value: (
                    qwen_module.validate_coverage_authority_inventory(
                        value, expected_iid=str(source["iid"])
                    )
                ),
                canonicalizer=lambda value: (
                    qwen_module.canonicalize_coverage_authority_inventory_model_output(
                        value, expected_iid=str(source["iid"])
                    )
                ),
            )
        )
        assignments_raw = qwen_module.coverage_authority_validated_raw(
            record, stage="coverage_authority_assignments"
        )
        validated_assignments, assignments_validated_from = (
            qwen_module._validate_original_a0_output(
                stage="coverage_authority_assignments",
                original_raw=assignments_raw,
                validator=lambda value: (
                    qwen_module.validate_coverage_authority_assignments(
                        value,
                        expected_iid=str(source["iid"]),
                        coverage_authority_inventory=validated_inventory,
                        change_region_proposals=change_region_proposals,
                    )
                ),
                canonicalizer=lambda value: (
                    qwen_module.canonicalize_coverage_authority_assignments_model_output(
                        value,
                        expected_iid=str(source["iid"]),
                        coverage_authority_inventory=validated_inventory,
                        change_region_proposals=change_region_proposals,
                    )
                ),
            )
        )
        rebuilt_authority = qwen_module.build_coverage_authority(
            coverage_authority_inventory=validated_inventory,
            coverage_authority_assignments=validated_assignments,
            change_region_proposals=change_region_proposals,
        )
    except Exception as error:
        raise Wan22FullMotionReleaseError(
            f"qwen_record_payload two-stage A0 raw closure differs: {error}"
        ) from error
    if (
        record.get("coverage_authority_inventory_validated_from")
        != inventory_validated_from
        or record.get("coverage_authority_assignments_validated_from")
        != assignments_validated_from
        or record.get("coverage_authority_inventory_digest")
        != object_sha256(validated_inventory)
        or record.get("coverage_authority_assignments_digest")
        != object_sha256(validated_assignments)
        or
        validated_inventory != coverage_authority["inventory"]
        or validated_assignments != coverage_authority["assignments"]
        or rebuilt_authority != coverage_authority
    ):
        raise Wan22FullMotionReleaseError(
            "qwen_record_payload two-stage A0 raw/object binding differs"
        )
    try:
        selected_target_raw = qwen_module.target_plan_validated_raw(
            record,
            source_census=source,
        )
        (
            _parsed_target_raw,
            validated_target_plan,
            validated_target_receipt,
        ) = qwen_module._canonicalize_target_plan_raw(
            selected_target_raw,
            stage="stored selected PASS_B target plan",
            source_census=source,
        )
    except Exception as error:
        raise Wan22FullMotionReleaseError(
            f"qwen_record_payload PASS_B selected raw closure differs: {error}"
        ) from error
    if (
        validated_target_plan != plan
        or validated_target_receipt
        != record.get("target_plan_canonicalization")
        or record.get("target_plan_digest")
        != object_sha256(validated_target_plan)
        or record.get("target_plan_canonicalization_digest")
        != object_sha256(validated_target_receipt)
    ):
        raise Wan22FullMotionReleaseError(
            "qwen_record_payload PASS_B selected raw/object binding differs"
        )
    if qwen_module.qwen_result_payload(record) != expected_result_payload:
        raise Wan22FullMotionReleaseError(
            "qwen_record_payload canonical result projection differs"
        )
    recomputed_result_digest = object_sha256(expected_result_payload)
    if (
        record.get("result_digest") != recomputed_result_digest
        or evidence.get("result_digest") != recomputed_result_digest
        or motion_spec.get("qwen_result_digest") != recomputed_result_digest
    ):
        raise Wan22FullMotionReleaseError(
            "Qwen result digest is not recomputed from the canonical payload"
        )
    recomputed_provenance_digest = qwen_module.qwen_provenance_digest(record)
    if (
        record.get("provenance_digest") != recomputed_provenance_digest
        or evidence.get("provenance_digest") != recomputed_provenance_digest
        or motion_spec.get("qwen_provenance_digest")
        != recomputed_provenance_digest
    ):
        raise Wan22FullMotionReleaseError(
            "Qwen provenance digest is not recomputed from the full record"
        )
    for field in (
        "record_schema_version",
        "model_path",
        "model_revision",
        "transformers_version",
    ):
        _string(evidence.get(field), context=f"qwen_evidence.{field}")
    if (
        evidence.get("record_schema_version")
        != "goku-full-motion-qwen-record-v6"
        or "Qwen3-VL-32B-Instruct" not in evidence["model_path"]
    ):
        raise Wan22FullMotionReleaseError(
            "Qwen evidence is not the frozen Qwen3-VL-32B lineage"
        )
    return dict(evidence)


def _validate_finalization(value: Any, *, family: str) -> dict[str, Any]:
    finalization = _mapping(value, context="full_motion_finalization")
    _exact_keys(finalization, _FINALIZATION_KEYS, context="full_motion_finalization")
    if (
        finalization.get("schema_version") != FINALIZATION_ROW_SCHEMA
        or finalization.get("policy_version") != FINALIZATION_POLICY
        or finalization.get("family") != family
        or finalization.get("selection_bucket")
        not in {"primary", "reserve", "review_only"}
    ):
        raise Wan22FullMotionReleaseError("full-motion finalization differs")
    for field in (
        "candidate_rank",
        "review_rank",
        "dynamic_unit_count",
    ):
        if type(finalization.get(field)) is not int or finalization[field] <= 0:
            raise Wan22FullMotionReleaseError(
                f"full_motion_finalization.{field} must be positive integer"
            )
    if (
        type(finalization.get("qwen_shard_index")) is not int
        or not 0 <= finalization["qwen_shard_index"] < 8
    ):
        raise Wan22FullMotionReleaseError(
            "full_motion_finalization.qwen_shard_index is outside 0..7"
        )
    if not 1 <= finalization["dynamic_unit_count"] <= 3:
        raise Wan22FullMotionReleaseError("dynamic_unit_count is outside 1..3")
    signatures = finalization.get("target_action_signatures")
    if (
        not isinstance(signatures, list)
        or len(signatures) != finalization["dynamic_unit_count"]
        or any(not isinstance(item, str) or not item for item in signatures)
    ):
        raise Wan22FullMotionReleaseError("target action signatures differ")
    if type(finalization.get("required_canary")) is not bool:
        raise Wan22FullMotionReleaseError("required_canary must be boolean")
    _sha(
        finalization.get("qwen_receipt_digest"),
        context="full_motion_finalization.qwen_receipt_digest",
    )
    return dict(finalization)


def _validate_actual_media(
    row: Mapping[str, Any], *, verify_media: bool
) -> dict[str, Any]:
    source = _stable_regular_file(
        Path(_string(row.get("resolved_source_video"), context="resolved source")),
        context="resolved source video",
    )
    anchor = _stable_regular_file(
        Path(_string(row.get("resolved_anchor_image"), context="resolved anchor")),
        context="resolved anchor image",
    )
    source_sha = _sha(row.get("source_video_sha256"), context="source video SHA")
    anchor_sha = _sha(row.get("anchor_sha256"), context="anchor SHA")
    if not verify_media:
        return {
            "source_path": str(source),
            "anchor_path": str(anchor),
            "source_video_sha256": source_sha,
            "anchor_sha256": anchor_sha,
        }
    if _file_digest(source) != source_sha or _file_digest(anchor) != anchor_sha:
        raise Wan22FullMotionReleaseError("bound media file SHA differs")
    try:
        import cv2
        import numpy as np
        from PIL import Image

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise Wan22FullMotionReleaseError("OpenCV cannot open source video")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, frame_zero_bgr = capture.read()
        capture.release()
        if not ok or frame_zero_bgr is None:
            raise Wan22FullMotionReleaseError(
                "cannot decode exact source frame zero"
            )
        if anchor.suffix.casefold() != ".png":
            raise Wan22FullMotionReleaseError(
                "exact-I0 anchor must be a lossless PNG"
            )
        with Image.open(anchor) as image:
            if image.format != "PNG":
                raise Wan22FullMotionReleaseError(
                    "exact-I0 anchor suffix/content is not PNG"
                )
            anchor_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        frame_zero_rgb = cv2.cvtColor(frame_zero_bgr, cv2.COLOR_BGR2RGB)
        if anchor_rgb.shape != frame_zero_rgb.shape or not np.array_equal(
            anchor_rgb, frame_zero_rgb
        ):
            raise Wan22FullMotionReleaseError(
                "anchor is not pixel-identical to decoded source frame zero"
            )
    except Wan22FullMotionReleaseError:
        raise
    except Exception as error:
        raise Wan22FullMotionReleaseError(
            f"cannot inspect source temporal geometry: {error}"
        ) from error
    if frame_count != EXPECTED_FRAME_COUNT or not math.isclose(
        fps, 25.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise Wan22FullMotionReleaseError(
            f"source media must be exactly 81 frames at 25 FPS, found "
            f"{frame_count} at {fps}"
        )
    return {
        "source_path": str(source),
        "anchor_path": str(anchor),
        "exact_i0": True,
        "lossless_png": True,
        "width": int(anchor_rgb.shape[1]),
        "height": int(anchor_rgb.shape[0]),
        "anchor_sha256": anchor_sha,
        "source_video_sha256": source_sha,
        "frame_zero_rgb_sha256": hashlib.sha256(
            anchor_rgb.tobytes(order="C")
        ).hexdigest(),
    }


def _validate_generation_row(
    value: Mapping[str, Any], *, verify_media: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .goku_full_motion_finalize import validate_generation_row

    row = _mapping(value, context="full-motion generation row")
    _exact_keys(row, _ROW_KEYS, context="full-motion generation row")
    try:
        finalized = validate_generation_row(row)
    except Exception as error:
        raise Wan22FullMotionReleaseError(
            f"generation row fails frozen finalizer validation: {error}"
        ) from error
    if finalized != dict(row):
        raise Wan22FullMotionReleaseError(
            "frozen finalizer changed the generation row"
        )
    if row.get("schema_version") != GENERATION_MANIFEST_SCHEMA:
        raise Wan22FullMotionReleaseError("generation row schema differs")
    iid = _safe_iid(row.get("iid"), context="generation iid")
    group_id = _string(row.get("group_id"), context="generation group_id")
    family = _string(row.get("family"), context="generation family")
    for field in (
        "source_video",
        "resolved_source_video",
        "anchor_image",
        "resolved_anchor_image",
        "edit_instruction",
    ):
        _string(row.get(field), context=f"generation {field}")
    for field in (
        "source_video_sha256",
        "anchor_sha256",
        "selected_media_evidence_sha256",
        "edit_instruction_sha256",
        "motion_spec_sha256",
    ):
        _sha(row.get(field), context=f"generation {field}")
    fixed = {
        "action_change_substantive": "yes",
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
        "authorization_interface_available": False,
        "annotation_source": "qwen3-vl-32b",
        "human_reviewed": False,
    }
    for field, expected in fixed.items():
        if row.get(field) != expected:
            raise Wan22FullMotionReleaseError(
                f"generation {field} must be exactly {expected!r}"
            )

    selected_media = _mapping(
        row.get("selected_media_evidence"), context="selected_media_evidence"
    )
    if object_sha256(selected_media) != row["selected_media_evidence_sha256"]:
        raise Wan22FullMotionReleaseError("selected media evidence SHA differs")
    selected_frames = selected_media.get("frame_count")
    selected_fps = selected_media.get("fps")
    if (
        selected_frames != EXPECTED_FRAME_COUNT
        or isinstance(selected_fps, bool)
        or not isinstance(selected_fps, (int, float))
        or not math.isfinite(float(selected_fps))
        or not math.isclose(float(selected_fps), 25.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise Wan22FullMotionReleaseError(
            "selected media evidence must bind exactly 81 frames at 25 FPS"
        )
    _validate_temporal_geometry(row.get("strict_temporal_geometry"))

    motion_spec = _mapping(row.get("motion_spec"), context="motion_spec")
    _exact_keys(motion_spec, _MOTION_SPEC_KEYS, context="motion_spec")
    if motion_spec.get("schema_version") != MOTION_SPEC_SCHEMA:
        raise Wan22FullMotionReleaseError("motion_spec schema differs")
    if object_sha256(motion_spec) != row["motion_spec_sha256"]:
        raise Wan22FullMotionReleaseError("motion_spec SHA differs")
    qwen_result_digest = _sha(
        motion_spec.get("qwen_result_digest"), context="qwen result digest"
    )
    qwen_provenance_digest = _sha(
        motion_spec.get("qwen_provenance_digest"),
        context="qwen provenance digest",
    )
    source = validate_source_census(motion_spec.get("source_census"))
    if source.get("iid") != iid:
        raise Wan22FullMotionReleaseError("source census IID differs")
    plan = validate_target_plan(motion_spec.get("target_plan"), source_census=source)
    compiled = validate_compiled_instruction(
        motion_spec.get("compiled_instruction"),
        source_census=source,
        target_plan=plan,
    )
    if compiled != compile_full_motion_instruction(source, plan):
        raise Wan22FullMotionReleaseError("compiled instruction is not deterministic")
    coverage = validate_coverage_critic(
        motion_spec.get("coverage_critic"),
        source_census=source,
        target_plan=plan,
        compiled_instruction=compiled,
    )
    contract = validate_contract_binding(
        motion_spec.get("full_motion_contract"),
        source_census=source,
        target_plan=plan,
    )
    if contract != build_contract(source_census=source, target_plan=plan):
        raise Wan22FullMotionReleaseError("full-motion contract binding differs")
    instruction = _string(
        row.get("edit_instruction"), context="generation edit_instruction"
    )
    if (
        instruction != compiled["edit_instruction"]
        or row.get("edit_instruction_sha256") != compiled["instruction_sha256"]
    ):
        raise Wan22FullMotionReleaseError(
            "edit_instruction differs from deterministic compiled instruction"
        )
    qwen_evidence = _validate_qwen_evidence(
        row.get("qwen_evidence"),
        generation_row=row,
        motion_spec=motion_spec,
        source=source,
        plan=plan,
        compiled=compiled,
        coverage=coverage,
        contract=contract,
    )
    qwen_media = qwen_evidence["media_verification"]
    if (
        qwen_media.get("exact_i0") is not True
        or qwen_media.get("source_video_sha256")
        != row["source_video_sha256"]
        or qwen_media.get("anchor_sha256") != row["anchor_sha256"]
    ):
        raise Wan22FullMotionReleaseError(
            "Qwen exact-I0 media evidence differs from generation media"
        )
    finalization = _validate_finalization(
        row.get("full_motion_finalization"), family=family
    )
    if (
        finalization["dynamic_unit_count"] != len(source["dynamic_units"])
        or finalization["qwen_shard_index"] != qwen_evidence["shard_index"]
        or finalization["qwen_receipt_digest"] != qwen_evidence["receipt_digest"]
    ):
        raise Wan22FullMotionReleaseError(
            "finalization differs from Qwen/full-motion unit evidence"
        )
    expected_signatures = [
        item["target_action_signature"] for item in plan["dynamic_unit_targets"]
    ]
    if finalization["target_action_signatures"] != expected_signatures:
        raise Wan22FullMotionReleaseError(
            "finalization target action signatures differ from target plan"
        )
    media = _validate_actual_media(row, verify_media=verify_media)
    closure = {
        "iid": iid,
        "group_id": group_id,
        "row_sha256": object_sha256(row),
        "source_video_sha256": row["source_video_sha256"],
        "anchor_sha256": row["anchor_sha256"],
        "selected_media_evidence_sha256": row[
            "selected_media_evidence_sha256"
        ],
        "motion_spec_sha256": row["motion_spec_sha256"],
        "compiled_instruction_sha256": object_sha256(compiled),
        "edit_instruction_sha256": row["edit_instruction_sha256"],
        "qwen_result_digest": qwen_result_digest,
        "qwen_provenance_digest": qwen_provenance_digest,
        "change_region_proposals_sha256": qwen_evidence[
            "change_region_proposals_digest"
        ],
        "coverage_authority_sha256": qwen_evidence[
            "coverage_authority_digest"
        ],
        "coverage_authority_inventory_sha256": qwen_evidence[
            "coverage_authority_inventory_digest"
        ],
        "coverage_authority_assignments_sha256": qwen_evidence[
            "coverage_authority_assignments_digest"
        ],
        "coverage_authority_alignment_sha256": qwen_evidence[
            "coverage_authority_alignment_digest"
        ],
        "primary_source_census_sha256": qwen_evidence[
            "source_census_digest"
        ],
        "source_census_canonicalization": dict(
            qwen_evidence["source_census_canonicalization"]
        ),
        "source_census_canonicalization_sha256": qwen_evidence[
            "source_census_canonicalization_digest"
        ],
        "secondary_source_census_sha256": qwen_evidence[
            "secondary_source_census_digest"
        ],
        "secondary_source_census_canonicalization": dict(
            qwen_evidence["secondary_source_census_canonicalization"]
        ),
        "secondary_source_census_canonicalization_sha256": qwen_evidence[
            "secondary_source_census_canonicalization_digest"
        ],
        "source_inventory_alignment_sha256": qwen_evidence[
            "source_inventory_alignment_digest"
        ],
        "target_plan_canonicalization": dict(
            qwen_evidence["target_plan_canonicalization"]
        ),
        "target_plan_canonicalization_sha256": qwen_evidence[
            "target_plan_canonicalization_digest"
        ],
        "qwen_hard_gate_sha256": object_sha256(qwen_evidence["hard_gate"]),
        "source_frame_count": EXPECTED_FRAME_COUNT,
        "source_frame_rate": EXPECTED_FRAME_RATE,
    }
    validated = dict(row)
    validated["_release_media_verification"] = media
    return validated, closure


def _validate_manifest(
    manifest_path: Path,
    *,
    expected_rows: int | None,
    verify_media: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, Path]:
    raw_rows, _lines, raw, resolved = _strict_jsonl(
        manifest_path, context="full-motion generation manifest"
    )
    if expected_rows is not None and len(raw_rows) != expected_rows:
        raise Wan22FullMotionReleaseError(
            f"manifest must contain exactly {expected_rows} rows"
        )
    rows: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    seen_iids: set[str] = set()
    seen_groups: set[str] = set()
    for raw_row in raw_rows:
        row, closure = _validate_generation_row(raw_row, verify_media=verify_media)
        if closure["iid"] in seen_iids or closure["group_id"] in seen_groups:
            raise Wan22FullMotionReleaseError(
                "manifest IID/group_id must be globally unique"
            )
        seen_iids.add(closure["iid"])
        seen_groups.add(closure["group_id"])
        rows.append(row)
        closures.append(closure)
    return rows, closures, raw, resolved


def _validate_timestamp(value: Any) -> str:
    timestamp = _string(value, context="issued_at_utc")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise Wan22FullMotionReleaseError("issued_at_utc is invalid") from error
    if parsed.tzinfo is None:
        raise Wan22FullMotionReleaseError("issued_at_utc must include timezone")
    return timestamp


def _require_signer_anchor() -> None:
    if (
        SIGNER_PUBLIC_KEY.startswith("REPLACE_WITH_")
        or SIGNER_KEY_FINGERPRINT.startswith("REPLACE_WITH_")
    ):
        raise Wan22FullMotionReleaseError(
            "dedicated full-motion release public key is not frozen"
        )
    if not SIGNER_PUBLIC_KEY.startswith("ssh-ed25519 "):
        raise Wan22FullMotionReleaseError(
            "full-motion signer must be a source-anchored Ed25519 key"
        )
    with tempfile.TemporaryDirectory(prefix="motive-full-motion-key-") as name:
        public_path = Path(name) / "release_key.pub"
        public_path.write_text(SIGNER_PUBLIC_KEY + "\n", encoding="utf-8")
        result = subprocess.run(
            ["ssh-keygen", "-lf", str(public_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if (
        result.returncode != 0
        or len(result.stdout.split()) < 2
        or result.stdout.split()[1] != SIGNER_KEY_FINGERPRINT
    ):
        raise Wan22FullMotionReleaseError(
            "source-anchored full-motion public key/fingerprint differ"
        )


def build_release_payload(
    *,
    root_manifest_path: str | Path,
    release_id: str,
    issued_at_utc: str,
    verify_media: bool = True,
) -> dict[str, Any]:
    """Validate and close an ordered root manifest before it is signed."""

    rows, closures, raw, _resolved = _validate_manifest(
        Path(root_manifest_path), expected_rows=None, verify_media=verify_media
    )
    if len(rows) < CONTIGUOUS_SHARD_ROWS:
        raise Wan22FullMotionReleaseError(
            "root manifest must contain at least eight rows"
        )
    _string(release_id, context="release_id")
    _validate_timestamp(issued_at_utc)
    return {
        "schema_version": RELEASE_PAYLOAD_SCHEMA,
        "release_id": release_id,
        "issued_at_utc": issued_at_utc,
        "purpose": "wan22_i2v_full_motion_root_contiguous8",
        "root_manifest": {
            "schema_version": GENERATION_MANIFEST_SCHEMA,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": len(rows),
            "ordered_iids_sha256": _ordered_digest(
                [closure["iid"] for closure in closures]
            ),
            "ordered_row_sha256": _ordered_digest(
                [closure["row_sha256"] for closure in closures]
            ),
            "authorized_submanifest_mode": "any_contiguous_rows_v1",
            "contiguous_shard_rows": CONTIGUOUS_SHARD_ROWS,
        },
        "row_authorizations": closures,
        "prompt_policy": dict(PROMPT_POLICY),
        "temporal_policy": dict(TEMPORAL_POLICY),
        "implementation": _implementation_binding(),
    }


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    output = path.expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o400)
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _sign_payload(
    *,
    signed: Mapping[str, Any],
    output_path: str | Path,
    signing_key: str | Path,
) -> dict[str, Any]:
    _require_signer_anchor()
    key = _stable_regular_file(Path(signing_key), context="release signing key")
    public = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(key)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if public.returncode != 0:
        raise Wan22FullMotionReleaseError("cannot read release signing key")
    derived = " ".join(public.stdout.decode("utf-8").strip().split()[:2])
    if derived != SIGNER_PUBLIC_KEY:
        raise Wan22FullMotionReleaseError(
            "signing key differs from source-anchored public key"
        )
    with tempfile.TemporaryDirectory(prefix="motive-full-motion-sign-") as name:
        message = Path(name) / "payload.json"
        message.write_bytes(_canonical_bytes(signed))
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                SIGNATURE_NAMESPACE,
                str(message),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        signature_path = Path(str(message) + ".sig")
        if result.returncode != 0 or not signature_path.is_file():
            raise Wan22FullMotionReleaseError(
                "release signing failed: "
                + result.stderr.decode("utf-8", errors="replace").strip()
            )
        armor = signature_path.read_bytes()
    envelope = {
        "schema_version": RELEASE_SCHEMA,
        "signed": dict(signed),
        "signature": {
            "format": "SSHSIG",
            "namespace": SIGNATURE_NAMESPACE,
            "principal": SIGNER_PRINCIPAL,
            "key_fingerprint": SIGNER_KEY_FINGERPRINT,
            "armored_signature_base64": base64.b64encode(armor).decode("ascii"),
        },
    }
    _atomic_new_json(Path(output_path), envelope)
    return envelope


def build_and_sign_release(
    *,
    root_manifest_path: str | Path,
    output_path: str | Path,
    signing_key: str | Path,
    release_id: str,
    issued_at_utc: str,
    verify_media: bool = True,
) -> dict[str, Any]:
    payload = build_release_payload(
        root_manifest_path=root_manifest_path,
        release_id=release_id,
        issued_at_utc=issued_at_utc,
        verify_media=verify_media,
    )
    return _sign_payload(
        signed=payload, output_path=output_path, signing_key=signing_key
    )


def prepare_release_request(
    *,
    root_manifest_path: str | Path,
    request_path: str | Path,
    release_id: str,
    issued_at_utc: str,
    challenge: str,
) -> dict[str, Any]:
    """Validate AUH media and publish a challenge-bound offline-sign request."""

    challenge_sha = _sha(challenge, context="release request challenge")
    signed = build_release_payload(
        root_manifest_path=root_manifest_path,
        release_id=release_id,
        issued_at_utc=issued_at_utc,
        verify_media=True,
    )
    request: dict[str, Any] = {
        "schema_version": RELEASE_REQUEST_SCHEMA,
        "challenge_sha256": challenge_sha,
        "builder": _implementation_binding(),
        "signed": signed,
    }
    request["request_digest"] = _object_digest(request)
    _atomic_new_json(Path(request_path), request)
    return request


def sign_prepared_request(
    *,
    request_path: str | Path,
    output_path: str | Path,
    signing_key: str | Path,
    expected_challenge: str,
) -> dict[str, Any]:
    """Sign only a closed request built by the byte-identical frozen source."""

    request, _raw, _resolved = _strict_json_file(
        Path(request_path), context="full-motion release request"
    )
    _exact_keys(
        request,
        {
            "schema_version",
            "challenge_sha256",
            "builder",
            "signed",
            "request_digest",
        },
        context="full-motion release request",
    )
    if request.get("schema_version") != RELEASE_REQUEST_SCHEMA:
        raise Wan22FullMotionReleaseError("release request schema differs")
    claimed_digest = _sha(
        request.get("request_digest"), context="release request digest"
    )
    unsigned = dict(request)
    del unsigned["request_digest"]
    if claimed_digest != _object_digest(unsigned):
        raise Wan22FullMotionReleaseError("release request digest differs")
    challenge = _sha(expected_challenge, context="expected release challenge")
    if request.get("challenge_sha256") != challenge:
        raise Wan22FullMotionReleaseError("release request challenge differs")
    if request.get("builder") != _implementation_binding():
        raise Wan22FullMotionReleaseError(
            "release request was not built by byte-identical frozen source"
        )
    signed = _mapping(request.get("signed"), context="release request payload")
    _validate_payload_shape(signed)
    return _sign_payload(
        signed=signed,
        output_path=output_path,
        signing_key=signing_key,
    )


def _verify_signature(
    signed: Mapping[str, Any], signature: Mapping[str, Any]
) -> None:
    _require_signer_anchor()
    _exact_keys(
        signature,
        {
            "format",
            "namespace",
            "principal",
            "key_fingerprint",
            "armored_signature_base64",
        },
        context="release signature",
    )
    expected = {
        "format": "SSHSIG",
        "namespace": SIGNATURE_NAMESPACE,
        "principal": SIGNER_PRINCIPAL,
        "key_fingerprint": SIGNER_KEY_FINGERPRINT,
    }
    for field, value in expected.items():
        if signature.get(field) != value:
            raise Wan22FullMotionReleaseError(
                f"release signature {field} differs"
            )
    try:
        armor = base64.b64decode(
            _string(
                signature.get("armored_signature_base64"),
                context="release signature bytes",
            ),
            validate=True,
        )
    except (TypeError, ValueError) as error:
        raise Wan22FullMotionReleaseError(
            "release signature is not strict base64"
        ) from error
    if not (
        armor.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
        and armor.endswith(b"-----END SSH SIGNATURE-----\n")
    ):
        raise Wan22FullMotionReleaseError("release signature armor differs")
    with tempfile.TemporaryDirectory(prefix="motive-full-motion-verify-") as name:
        root = Path(name)
        allowed = root / "allowed_signers"
        signature_path = root / "payload.sshsig"
        allowed.write_text(
            f"{SIGNER_PRINCIPAL} {SIGNER_PUBLIC_KEY}\n", encoding="utf-8"
        )
        signature_path.write_bytes(armor)
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed),
                "-I",
                SIGNER_PRINCIPAL,
                "-n",
                SIGNATURE_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=_canonical_bytes(signed),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        raise Wan22FullMotionReleaseError(
            "release SSH signature verification failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )


def _validate_payload_shape(signed: Mapping[str, Any]) -> None:
    _exact_keys(
        signed,
        {
            "schema_version",
            "release_id",
            "issued_at_utc",
            "purpose",
            "root_manifest",
            "row_authorizations",
            "prompt_policy",
            "temporal_policy",
            "implementation",
        },
        context="full-motion release payload",
    )
    if (
        signed.get("schema_version") != RELEASE_PAYLOAD_SCHEMA
        or signed.get("purpose")
        != "wan22_i2v_full_motion_root_contiguous8"
        or signed.get("prompt_policy") != PROMPT_POLICY
        or signed.get("temporal_policy") != TEMPORAL_POLICY
        or signed.get("implementation") != _implementation_binding()
    ):
        raise Wan22FullMotionReleaseError("full-motion release policy differs")
    _string(signed.get("release_id"), context="release_id")
    _validate_timestamp(signed.get("issued_at_utc"))
    scope = _mapping(signed.get("root_manifest"), context="root_manifest")
    _exact_keys(
        scope,
        {
            "schema_version",
            "sha256",
            "bytes",
            "rows",
            "ordered_iids_sha256",
            "ordered_row_sha256",
            "authorized_submanifest_mode",
            "contiguous_shard_rows",
        },
        context="root_manifest",
    )
    if (
        scope.get("schema_version") != GENERATION_MANIFEST_SCHEMA
        or type(scope.get("rows")) is not int
        or scope["rows"] < CONTIGUOUS_SHARD_ROWS
        or type(scope.get("bytes")) is not int
        or scope["bytes"] <= 0
        or scope.get("authorized_submanifest_mode")
        != "any_contiguous_rows_v1"
        or scope.get("contiguous_shard_rows") != CONTIGUOUS_SHARD_ROWS
    ):
        raise Wan22FullMotionReleaseError("root manifest scope differs")
    for field in ("sha256", "ordered_iids_sha256", "ordered_row_sha256"):
        _sha(scope.get(field), context=f"root_manifest.{field}")
    authorizations = signed.get("row_authorizations")
    if not isinstance(authorizations, list) or len(authorizations) != scope["rows"]:
        raise Wan22FullMotionReleaseError(
            "row authorizations differ from root row count"
        )
    expected_keys = {
        "iid",
        "group_id",
        "row_sha256",
        "source_video_sha256",
        "anchor_sha256",
        "selected_media_evidence_sha256",
        "motion_spec_sha256",
        "compiled_instruction_sha256",
        "edit_instruction_sha256",
        "qwen_result_digest",
        "qwen_provenance_digest",
        "change_region_proposals_sha256",
        "coverage_authority_sha256",
        "coverage_authority_inventory_sha256",
        "coverage_authority_assignments_sha256",
        "coverage_authority_alignment_sha256",
        "primary_source_census_sha256",
        "source_census_canonicalization",
        "source_census_canonicalization_sha256",
        "secondary_source_census_sha256",
        "secondary_source_census_canonicalization",
        "secondary_source_census_canonicalization_sha256",
        "source_inventory_alignment_sha256",
        "target_plan_canonicalization",
        "target_plan_canonicalization_sha256",
        "qwen_hard_gate_sha256",
        "source_frame_count",
        "source_frame_rate",
    }
    seen: set[str] = set()
    for index, item in enumerate(authorizations):
        closure = _mapping(item, context=f"row_authorizations[{index}]")
        _exact_keys(
            closure, expected_keys, context=f"row_authorizations[{index}]"
        )
        iid = _safe_iid(closure.get("iid"), context=f"authorization {index} iid")
        if iid in seen:
            raise Wan22FullMotionReleaseError("duplicate authorized IID")
        seen.add(iid)
        _string(closure.get("group_id"), context=f"authorization {index} group")
        receipt_fields = {
            "source_census_canonicalization",
            "secondary_source_census_canonicalization",
            "target_plan_canonicalization",
        }
        for field in expected_keys - {
            "iid",
            "group_id",
            "source_frame_count",
            "source_frame_rate",
            *receipt_fields,
        }:
            _sha(closure.get(field), context=f"authorization {index} {field}")
        for field in receipt_fields:
            receipt = _mapping(
                closure.get(field),
                context=f"authorization {index} {field}",
            )
            if object_sha256(receipt) != closure.get(f"{field}_sha256"):
                raise Wan22FullMotionReleaseError(
                    f"authorization {index} {field} digest differs"
                )
        if (
            closure.get("source_frame_count") != EXPECTED_FRAME_COUNT
            or closure.get("source_frame_rate") != EXPECTED_FRAME_RATE
        ):
            raise Wan22FullMotionReleaseError(
                "authorized row temporal geometry differs"
            )
    if (
        _ordered_digest([str(item["iid"]) for item in authorizations])
        != scope["ordered_iids_sha256"]
        or _ordered_digest([str(item["row_sha256"]) for item in authorizations])
        != scope["ordered_row_sha256"]
    ):
        raise Wan22FullMotionReleaseError("ordered root commitments differ")


def verify_signed_release(
    *,
    release_path: str | Path,
    manifest_path: str | Path,
    verify_media: bool = True,
) -> dict[str, Any]:
    """Authorize exactly one contiguous eight-row shard of the signed root."""

    envelope, _raw, resolved_release = _strict_json_file(
        Path(release_path), context="full-motion signed release"
    )
    _exact_keys(
        envelope,
        {"schema_version", "signed", "signature"},
        context="full-motion signed release",
    )
    if envelope.get("schema_version") != RELEASE_SCHEMA:
        raise Wan22FullMotionReleaseError("full-motion release schema differs")
    signed = _mapping(envelope.get("signed"), context="release signed payload")
    signature = _mapping(envelope.get("signature"), context="release signature")
    _verify_signature(signed, signature)
    _validate_payload_shape(signed)
    rows, closures, raw, resolved_manifest = _validate_manifest(
        Path(manifest_path),
        expected_rows=CONTIGUOUS_SHARD_ROWS,
        verify_media=verify_media,
    )
    authorizations = [dict(item) for item in signed["row_authorizations"]]
    by_iid = {
        str(item["iid"]): (index, item)
        for index, item in enumerate(authorizations)
    }
    indices: list[int] = []
    for closure in closures:
        entry = by_iid.get(closure["iid"])
        if entry is None or entry[1] != closure:
            raise Wan22FullMotionReleaseError(
                f"manifest row {closure['iid']} is outside signed root scope"
            )
        indices.append(entry[0])
    if indices != list(range(indices[0], indices[0] + CONTIGUOUS_SHARD_ROWS)):
        raise Wan22FullMotionReleaseError(
            "requested manifest is not one contiguous eight-row root shard"
        )
    payload_sha = _object_digest(signed)
    release_binding = {
        "path": str(resolved_release),
        "release_id": signed["release_id"],
        "payload_sha256": payload_sha,
        "signer_key_fingerprint": SIGNER_KEY_FINGERPRINT,
        "root_manifest_sha256": signed["root_manifest"]["sha256"],
        "root_manifest_rows": signed["root_manifest"]["rows"],
        "root_row_start_zero_based": indices[0],
        "root_row_stop_exclusive": indices[-1] + 1,
    }
    prepared: list[dict[str, Any]] = []
    for line_number, (row, closure) in enumerate(
        zip(rows, closures, strict=True), start=1
    ):
        item = dict(row)
        item.pop("_release_media_verification", None)
        item["_iid"] = closure["iid"]
        item["_line_number"] = line_number
        item["_row_digest"] = closure["row_sha256"]
        item["_authorization_mode"] = AUTHORIZATION_MODE
        item["_signed_release"] = dict(release_binding)
        # Compatibility-only display metadata for the legacy generated-target
        # manifest.  Neither field is executable, and both are derived solely
        # from the signed full-motion target plan.
        item["action_category"] = "full_motion"
        item["target_action_verb"] = "multi_entity_action_edit"
        prepared.append(item)
    return {
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_bytes": len(raw),
        "manifest_row_count": len(rows),
        "selected_rows": prepared,
        "selected_row_count": len(prepared),
        "release": release_binding,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root-manifest", required=True, type=Path)
    prepare.add_argument("--request", required=True, type=Path)
    prepare.add_argument("--release-id", required=True)
    prepare.add_argument("--issued-at-utc", required=True)
    prepare.add_argument("--challenge", required=True)
    sign = sub.add_parser("sign")
    sign.add_argument("--request", required=True, type=Path)
    sign.add_argument("--output", required=True, type=Path)
    sign.add_argument("--signing-key", required=True, type=Path)
    sign.add_argument("--expected-challenge", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root-manifest", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--signing-key", required=True, type=Path)
    build.add_argument("--release-id", required=True)
    build.add_argument("--issued-at-utc", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--release", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_release_request(
            root_manifest_path=args.root_manifest,
            request_path=args.request,
            release_id=args.release_id,
            issued_at_utc=args.issued_at_utc,
            challenge=args.challenge,
        )
    elif args.command == "sign":
        sign_prepared_request(
            request_path=args.request,
            output_path=args.output,
            signing_key=args.signing_key,
            expected_challenge=args.expected_challenge,
        )
    elif args.command == "build":
        build_and_sign_release(
            root_manifest_path=args.root_manifest,
            output_path=args.output,
            signing_key=args.signing_key,
            release_id=args.release_id,
            issued_at_utc=args.issued_at_utc,
        )
    else:
        verified = verify_signed_release(
            release_path=args.release, manifest_path=args.manifest
        )
        print(json.dumps(verified["release"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_MODE",
    "CONTIGUOUS_SHARD_ROWS",
    "GENERATION_MANIFEST_SCHEMA",
    "MOTION_SPEC_SCHEMA",
    "PROMPT_POLICY",
    "RELEASE_PAYLOAD_SCHEMA",
    "RELEASE_REQUEST_SCHEMA",
    "RELEASE_SCHEMA",
    "SIGNATURE_NAMESPACE",
    "SIGNER_KEY_FINGERPRINT",
    "SIGNER_PRINCIPAL",
    "SIGNER_PUBLIC_KEY",
    "TEMPORAL_GEOMETRY_SCHEMA",
    "TEMPORAL_POLICY",
    "Wan22FullMotionReleaseError",
    "build_and_sign_release",
    "build_release_payload",
    "prepare_release_request",
    "sign_prepared_request",
    "verify_signed_release",
]
