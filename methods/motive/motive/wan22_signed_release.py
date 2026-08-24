"""Cryptographically verify the only authorization accepted by Wan generation.

The Goku finalizer deliberately emits a pending, non-executable v9 manifest.
Changing its JSON booleans is not authorization.  The original v1 profile
admits exactly eight byte-bound rows only when one of two independently closed
evidence paths passes.  The isolated v2 profile admits exactly 512 rows only
after rerunning the scale selector and binding its complete closure:

* the original independent 16-row smoke acceptance result passes exactly and
  binds a Qwen3-VL-32B checkpoint, its complete model closure, and immutable
  Qwen/verifier source snapshot hashes; or
* the frozen exact-eight selector is rerun against a completed finalizer, its
  manifest and receipt are byte-identical to the requested selection, and the
  finalizer, receipt, and selector implementation hashes remain bound;
* every executable prompt is the byte-exact ``edit_instruction`` from the v9
  manifest (writer and edited-caption prose remain non-executable);
* every source video and first-frame anchor still hashes to the manifest; and
* the release payload has a valid OpenSSH signature from the source-anchored
  release key below.

The private key is intentionally absent from this repository and from AUH.
The verification dependency is the system ``ssh-keygen`` implementation; no
Python crypto package or network access is needed.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


RELEASE_SCHEMA = "motive-wan22-signed-generation-release-v1"
RELEASE_PAYLOAD_SCHEMA = "motive-wan22-generation-release-payload-v1"
RELEASE_REQUEST_SCHEMA = "motive-wan22-generation-release-request-v1"
RELEASE_SCHEMA_V2 = "motive-wan22-signed-generation-release-v2"
RELEASE_PAYLOAD_SCHEMA_V2 = "motive-wan22-generation-release-payload-v2"
RELEASE_REQUEST_SCHEMA_V2 = "motive-wan22-generation-release-request-v2"
RELEASE_PROFILE_EXACT8_V1 = "exact8-v1"
RELEASE_PROFILE_EXACT512_V2 = "exact512-v2"
GENERATION_MANIFEST_SCHEMA = "motive-goku-action-anchor-generation-v9"
SCALE512_GENERATION_MANIFEST_SCHEMA = (
    "motive-goku-action-anchor-generation-scale512-v1"
)
EXACT512_MANIFEST_SCOPE_SCHEMA = (
    "motive-wan22-exact512-generation-manifest-scope-v2"
)
ACCEPTANCE_RESULT_SCHEMA = "motive-goku-action-v16-acceptance-result-v1"
FINALIZER_SELECTION_EVIDENCE_SCHEMA = (
    "motive-wan22-finalizer-selection-evidence-v1"
)
FINALIZER_SELECTION_EVIDENCE_MODE = (
    "rerun_exact8_selector_byte_identical_v1"
)
EXACT8_SELECTION_RECEIPT_SCHEMA = (
    "motive-wan22-exact8-selection-receipt-v1"
)
EXACT8_SELECTION_POLICY = "lowest_finalizer_review_rank_exact8"
SCALE512_SELECTION_EVIDENCE_SCHEMA = (
    "motive-wan22-scale512-selection-evidence-v2"
)
SCALE512_SELECTION_EVIDENCE_MODE = (
    "rerun_exact512_selector_byte_identical_v2"
)
EXACT512_SELECTION_RECEIPT_SCHEMA = (
    "motive-wan22-exact512-selection-receipt-v1"
)
EXACT512_RANK_ONLY_POLICY = "lowest_finalizer_review_rank_exact512"
EXACT512_RETAIN_EXACT8_POLICY = (
    "retain_exact8_then_lowest_finalizer_review_rank_exact512"
)
TEMPORAL_GEOMETRY_SCHEMA = (
    "motive-goku-action-anchor-temporal-geometry-v1"
)
MEDIA_FILE_VERIFICATION_SCHEMA = (
    "motive-goku-action-anchor-media-file-verification-v1"
)

RELEASE_ROW_COUNT = 8
SCALE512_RELEASE_ROW_COUNT = 512
SCALE512_SHARD_ROW_COUNT = 8
SMOKE_ROW_COUNT = 16
FINALIZER_DONE_NAME = "done.json"
FINALIZER_SUMMARY_NAME = "summary.json"
EXACT8_MANIFEST_NAME = "generation_manifest.jsonl"
EXACT8_SELECTION_RECEIPT_NAME = "selection_receipt.json"
EXACT8_SELECTOR_NAME = "wan22_select_exact8.py"
EXACT512_SELECTOR_NAME = "wan22_select_exact512.py"
FINALIZER_IMPLEMENTATION_NAME = "goku_action_anchor_finalize.py"
SIGNATURE_NAMESPACE = "motive-wan22-generation-release-v1"
SIGNATURE_NAMESPACE_V2 = "motive-wan22-generation-release-v2"
SIGNER_PRINCIPAL = "motive-wan22-release"
# Filled only with the dedicated per-run release key. Never reuse a login,
# Git, or other identity key for this trust boundary.
SIGNER_KEY_FINGERPRINT = (
    "SHA256:a8Qhr5pGxAC1mEIMEV2yI1mUzzeT59j5SxA9KqprncY"
)
SIGNER_PUBLIC_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIN9viiNtE4g7w1XuJKl16FpEuI2LukJHB15bYAJq/35P"
)

AUTHORIZATION_MODE = "sshsig_qwen3_vl_32b_smoke_release_v1"
PROMPT_POLICY = {
    "executable_field": "edit_instruction",
    "byte_exact_manifest_value": True,
    "writer_proposals_executable": False,
    "edited_caption_provenance_executable": False,
}
TEMPORAL_POLICY = {
    "source_output_frame_count_equal": True,
    "source_output_fps_equal": True,
    "first_frame_is_exact_bound_anchor": True,
    "maximum_duration_delta_frames": 1,
}

_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
QWEN3_MODEL_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/"
    "MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct"
)


class Wan22ReleaseError(RuntimeError):
    """A release, its evidence, or its frozen manifest is invalid."""


def _reject_constant(value: str) -> None:
    raise Wan22ReleaseError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Wan22ReleaseError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Wan22ReleaseError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, Wan22ReleaseError):
            raise
        raise Wan22ReleaseError(f"{context} is not strict JSON: {error}") from error


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
        raise Wan22ReleaseError(f"value is not canonical JSON: {error}") from error


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise Wan22ReleaseError(
            f"{context} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Wan22ReleaseError(f"{context} must be one canonical non-empty string")
    return value


def _sha(value: Any, *, context: str) -> str:
    text = _string(value, context=context)
    if _SHA_RE.fullmatch(text) is None:
        raise Wan22ReleaseError(f"{context} must be a lowercase SHA-256")
    return text


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Wan22ReleaseError(f"{context} must be an object")
    return value


def _stable_read(path: Path, *, context: str) -> tuple[Path, bytes]:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise Wan22ReleaseError(f"{context} path must be absolute")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(expanded, flags)
    except OSError as error:
        raise Wan22ReleaseError(
            f"{context} is missing or not a non-symlink regular file: {expanded}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not os.path.isfile(expanded) or expanded.is_symlink():
            raise Wan22ReleaseError(f"{context} is not a regular non-symlink file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or len(raw) != after.st_size:
            raise Wan22ReleaseError(f"{context} changed while it was read")
    finally:
        os.close(descriptor)
    return expanded.resolve(strict=True), raw


def _load_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes, Path]:
    resolved, raw = _stable_read(path, context=context)
    value = _parse_json(raw, context=context)
    if not isinstance(value, dict):
        raise Wan22ReleaseError(f"{context} top level must be an object")
    return value, raw, resolved


def _load_jsonl(
    path: Path,
    *,
    context: str,
) -> tuple[list[dict[str, Any]], list[bytes], bytes, Path]:
    resolved, raw = _stable_read(path, context=context)
    if not raw or not raw.endswith(b"\n"):
        raise Wan22ReleaseError(f"{context} must be non-empty and newline-terminated")
    rows: list[dict[str, Any]] = []
    lines: list[bytes] = []
    for index, bare in enumerate(raw.splitlines(), start=1):
        if not bare:
            raise Wan22ReleaseError(f"{context}:{index} is blank")
        value = _parse_json(bare, context=f"{context}:{index}")
        if not isinstance(value, dict):
            raise Wan22ReleaseError(f"{context}:{index} is not an object")
        canonical = _canonical_bytes(value)
        if bare != canonical:
            raise Wan22ReleaseError(f"{context}:{index} is not canonical JSON")
        rows.append(value)
        lines.append(canonical + b"\n")
    return rows, lines, raw, resolved


def _ordered_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        b"".join(value.encode("utf-8") + b"\n" for value in values)
    ).hexdigest()


def _absolute_path_text(value: Any, *, context: str) -> str:
    text = _string(value, context=context)
    path = Path(text)
    if (
        not path.is_absolute()
        or path == Path("/")
        or os.path.normpath(text) != text
    ):
        raise Wan22ReleaseError(
            f"{context} must be a normalized non-root absolute path"
        )
    return text


def _validate_file_binding(value: Any, *, context: str) -> dict[str, Any]:
    binding = _mapping(value, context=context)
    _exact_keys(
        binding,
        {"path", "sha256", "bytes"},
        context=context,
    )
    path = _absolute_path_text(binding.get("path"), context=f"{context}.path")
    digest = _sha(binding.get("sha256"), context=f"{context}.sha256")
    size = binding.get("bytes")
    if type(size) is not int or size <= 0:
        raise Wan22ReleaseError(f"{context}.bytes must be a positive integer")
    return {"path": path, "sha256": digest, "bytes": size}


def _validate_signed_finalizer_selection_binding(value: Any) -> dict[str, Any]:
    context = "release finalizer selection binding"
    binding = _mapping(value, context=context)
    expected_keys = {
        "schema_version",
        "mode",
        "finalizer_dir",
        "done",
        "summary",
        "selection_receipt",
        "selector_implementation",
        "finalizer_implementation",
        "selected_manifest",
    }
    _exact_keys(binding, expected_keys, context=context)
    if binding.get("schema_version") != FINALIZER_SELECTION_EVIDENCE_SCHEMA:
        raise Wan22ReleaseError("release finalizer selection schema differs")
    if binding.get("mode") != FINALIZER_SELECTION_EVIDENCE_MODE:
        raise Wan22ReleaseError("release finalizer selection mode differs")
    finalizer_dir = _absolute_path_text(
        binding.get("finalizer_dir"),
        context="release finalizer selection finalizer_dir",
    )
    done = _validate_file_binding(binding.get("done"), context=f"{context}.done")
    summary = _validate_file_binding(
        binding.get("summary"),
        context=f"{context}.summary",
    )
    receipt = _validate_file_binding(
        binding.get("selection_receipt"),
        context=f"{context}.selection_receipt",
    )
    selector = _validate_file_binding(
        binding.get("selector_implementation"),
        context=f"{context}.selector_implementation",
    )
    finalizer = _validate_file_binding(
        binding.get("finalizer_implementation"),
        context=f"{context}.finalizer_implementation",
    )
    manifest_value = _mapping(
        binding.get("selected_manifest"),
        context=f"{context}.selected_manifest",
    )
    _exact_keys(
        manifest_value,
        {"path", "sha256", "bytes", "rows"},
        context=f"{context}.selected_manifest",
    )
    manifest = _validate_file_binding(
        {key: manifest_value[key] for key in ("path", "sha256", "bytes")},
        context=f"{context}.selected_manifest",
    )
    if manifest_value.get("rows") != RELEASE_ROW_COUNT:
        raise Wan22ReleaseError(
            "release finalizer selection manifest must contain exactly eight rows"
        )
    expected_done = str(Path(finalizer_dir) / FINALIZER_DONE_NAME)
    expected_summary = str(Path(finalizer_dir) / FINALIZER_SUMMARY_NAME)
    if done["path"] != expected_done or summary["path"] != expected_summary:
        raise Wan22ReleaseError(
            "release finalizer selection done/summary paths differ"
        )
    if Path(selector["path"]).name != EXACT8_SELECTOR_NAME:
        raise Wan22ReleaseError(
            "release finalizer selection selector filename differs"
        )
    expected_finalizer = str(
        Path(selector["path"]).with_name(FINALIZER_IMPLEMENTATION_NAME)
    )
    if finalizer["path"] != expected_finalizer:
        raise Wan22ReleaseError(
            "release finalizer selection finalizer implementation path differs"
        )
    return {
        "schema_version": FINALIZER_SELECTION_EVIDENCE_SCHEMA,
        "mode": FINALIZER_SELECTION_EVIDENCE_MODE,
        "finalizer_dir": finalizer_dir,
        "done": done,
        "summary": summary,
        "selection_receipt": receipt,
        "selector_implementation": selector,
        "finalizer_implementation": finalizer,
        "selected_manifest": {**manifest, "rows": RELEASE_ROW_COUNT},
    }


def _validate_signed_scale512_selection_binding(value: Any) -> dict[str, Any]:
    context = "release scale512 selection binding"
    binding = _mapping(value, context=context)
    expected_keys = {
        "schema_version",
        "mode",
        "finalizer_dir",
        "done",
        "summary",
        "selection_receipt",
        "selector_implementation",
        "shared_strict_io_implementation",
        "finalizer_implementation",
        "selected_manifest",
        "retained_exact8_manifest",
    }
    _exact_keys(binding, expected_keys, context=context)
    if binding.get("schema_version") != SCALE512_SELECTION_EVIDENCE_SCHEMA:
        raise Wan22ReleaseError("release scale512 selection schema differs")
    if binding.get("mode") != SCALE512_SELECTION_EVIDENCE_MODE:
        raise Wan22ReleaseError("release scale512 selection mode differs")
    finalizer_dir = _absolute_path_text(
        binding.get("finalizer_dir"),
        context=f"{context}.finalizer_dir",
    )
    done = _validate_file_binding(binding.get("done"), context=f"{context}.done")
    summary = _validate_file_binding(
        binding.get("summary"), context=f"{context}.summary"
    )
    receipt = _validate_file_binding(
        binding.get("selection_receipt"),
        context=f"{context}.selection_receipt",
    )
    selector = _validate_file_binding(
        binding.get("selector_implementation"),
        context=f"{context}.selector_implementation",
    )
    shared = _validate_file_binding(
        binding.get("shared_strict_io_implementation"),
        context=f"{context}.shared_strict_io_implementation",
    )
    finalizer = _validate_file_binding(
        binding.get("finalizer_implementation"),
        context=f"{context}.finalizer_implementation",
    )
    manifest_value = _mapping(
        binding.get("selected_manifest"),
        context=f"{context}.selected_manifest",
    )
    _exact_keys(
        manifest_value,
        {"path", "sha256", "bytes", "rows"},
        context=f"{context}.selected_manifest",
    )
    manifest = _validate_file_binding(
        {key: manifest_value[key] for key in ("path", "sha256", "bytes")},
        context=f"{context}.selected_manifest",
    )
    if manifest_value.get("rows") != SCALE512_RELEASE_ROW_COUNT:
        raise Wan22ReleaseError(
            "release scale512 selection manifest must contain exactly 512 rows"
        )
    retained_value = binding.get("retained_exact8_manifest")
    retained: dict[str, Any] | None
    if retained_value is None:
        retained = None
    else:
        retained_mapping = _mapping(
            retained_value,
            context=f"{context}.retained_exact8_manifest",
        )
        _exact_keys(
            retained_mapping,
            {"path", "sha256", "bytes", "rows"},
            context=f"{context}.retained_exact8_manifest",
        )
        retained_file = _validate_file_binding(
            {
                key: retained_mapping[key]
                for key in ("path", "sha256", "bytes")
            },
            context=f"{context}.retained_exact8_manifest",
        )
        if retained_mapping.get("rows") != RELEASE_ROW_COUNT:
            raise Wan22ReleaseError(
                "release retained exact8 manifest must contain exactly eight rows"
            )
        retained = {**retained_file, "rows": RELEASE_ROW_COUNT}
    if done["path"] != str(Path(finalizer_dir) / FINALIZER_DONE_NAME):
        raise Wan22ReleaseError("release scale512 done path differs")
    if summary["path"] != str(Path(finalizer_dir) / FINALIZER_SUMMARY_NAME):
        raise Wan22ReleaseError("release scale512 summary path differs")
    if Path(selector["path"]).name != EXACT512_SELECTOR_NAME:
        raise Wan22ReleaseError("release exact512 selector filename differs")
    expected_shared = str(Path(selector["path"]).with_name(EXACT8_SELECTOR_NAME))
    expected_finalizer = str(
        Path(selector["path"]).with_name(FINALIZER_IMPLEMENTATION_NAME)
    )
    if shared["path"] != expected_shared:
        raise Wan22ReleaseError("release shared strict selector path differs")
    if finalizer["path"] != expected_finalizer:
        raise Wan22ReleaseError("release scale512 finalizer source path differs")
    return {
        "schema_version": SCALE512_SELECTION_EVIDENCE_SCHEMA,
        "mode": SCALE512_SELECTION_EVIDENCE_MODE,
        "finalizer_dir": finalizer_dir,
        "done": done,
        "summary": summary,
        "selection_receipt": receipt,
        "selector_implementation": selector,
        "shared_strict_io_implementation": shared,
        "finalizer_implementation": finalizer,
        "selected_manifest": {
            **manifest,
            "rows": SCALE512_RELEASE_ROW_COUNT,
        },
        "retained_exact8_manifest": retained,
    }


def _release_evidence(
    signed: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if signed.get("schema_version") == RELEASE_PAYLOAD_SCHEMA_V2:
        if set(signed).intersection(
            {"smoke_acceptance", "finalizer_selection", "scale512_selection"}
        ) != {"scale512_selection"}:
            raise Wan22ReleaseError(
                "scale512 release must contain exactly its v2 selection evidence"
            )
        return (
            "scale512_selection",
            _validate_signed_scale512_selection_binding(
                signed.get("scale512_selection")
            ),
        )
    has_smoke = "smoke_acceptance" in signed
    has_finalizer = "finalizer_selection" in signed
    if has_smoke == has_finalizer:
        raise Wan22ReleaseError(
            "release must contain exactly one evidence mode"
        )
    if has_smoke:
        return (
            "smoke_acceptance",
            _validate_signed_acceptance_binding(signed.get("smoke_acceptance")),
        )
    return (
        "finalizer_selection",
        _validate_signed_finalizer_selection_binding(
            signed.get("finalizer_selection")
        ),
    )


def _require_dedicated_signer_anchor() -> None:
    if (
        not SIGNER_KEY_FINGERPRINT.startswith("SHA256:")
        or not SIGNER_PUBLIC_KEY.startswith(
            ("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")
        )
    ):
        raise Wan22ReleaseError(
            "dedicated Wan release signer anchor is not frozen"
        )


def _validate_pending_v9_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    verify_media: bool,
    allowed_schemas: frozenset[str] = frozenset({GENERATION_MANIFEST_SCHEMA}),
) -> dict[str, Any]:
    context = f"generation manifest row {line_number}"
    if row.get("schema_version") not in allowed_schemas:
        raise Wan22ReleaseError(
            f"{context} schema is outside the release profile"
        )
    iid = _string(row.get("iid"), context=f"{context}.iid")
    if _IID_RE.fullmatch(iid) is None:
        raise Wan22ReleaseError(f"{context}.iid is unsafe")
    _string(row.get("group_id"), context=f"{context}.group_id")
    instruction = _string(
        row.get("edit_instruction"),
        context=f"{context}.edit_instruction",
    )
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if row.get("edit_instruction_sha256") != instruction_sha:
        raise Wan22ReleaseError(f"{context} edit_instruction SHA differs")
    if row.get("source_instruction_provenance") != instruction:
        raise Wan22ReleaseError(f"{context} instruction provenance differs")
    if "absolute_target_prompt" in row or "writer_absolute_target_prompt" in row:
        raise Wan22ReleaseError(f"{context} contains a forbidden writer prompt")
    if row.get("source_edited_caption_provenance_role") != (
        "non_executable_provenance"
    ):
        raise Wan22ReleaseError(f"{context} edited caption is not non-executable")
    contract = _mapping(
        row.get("instruction_contract"),
        context=f"{context}.instruction_contract",
    )
    expected_contract = {
        "sole_candidate_instruction_field": "edit_instruction",
        "candidate_instruction_source": "frozen_selected_prompt",
        "writer_proposal_payload_included": False,
        "writer_proposals_executable": False,
        "requires_future_signed_release_verifier": True,
    }
    if dict(contract) != expected_contract:
        raise Wan22ReleaseError(f"{context} instruction contract differs")
    expected_pending = {
        "manifest_role": "review_proposal",
        "production_eligible": False,
        "human_review_status": "pending",
        "generation_authorized": False,
        "approval": None,
        "authorization_interface_available": False,
    }
    for field, expected in expected_pending.items():
        if row.get(field) != expected:
            raise Wan22ReleaseError(
                f"{context}.{field} is not the exact pending v9 value"
            )
    if row.get("action_change_substantive") != "yes":
        raise Wan22ReleaseError(f"{context} action change is not substantive")

    source_sha = _sha(
        row.get("source_video_sha256"),
        context=f"{context}.source_video_sha256",
    )
    anchor_sha = _sha(
        row.get("anchor_sha256"),
        context=f"{context}.anchor_sha256",
    )
    source_path = Path(
        _string(
            row.get("resolved_source_video"),
            context=f"{context}.resolved_source_video",
        )
    )
    anchor_path = Path(
        _string(
            row.get("resolved_anchor_image"),
            context=f"{context}.resolved_anchor_image",
        )
    )

    media = _mapping(
        row.get("selected_media_evidence"),
        context=f"{context}.selected_media_evidence",
    )
    if row.get("selected_media_evidence_sha256") != _object_digest(media):
        raise Wan22ReleaseError(f"{context} selected media digest differs")
    geometry = _mapping(
        row.get("strict_temporal_geometry"),
        context=f"{context}.strict_temporal_geometry",
    )
    if geometry.get("schema_version") != TEMPORAL_GEOMETRY_SCHEMA:
        raise Wan22ReleaseError(f"{context} temporal geometry schema differs")
    frame_count = geometry.get("source_frame_count")
    output_frames = geometry.get("required_output_frame_count")
    if (
        type(frame_count) is not int
        or frame_count <= 0
        or frame_count % 4 != 1
        or output_frames != frame_count
    ):
        raise Wan22ReleaseError(f"{context} frame-count contract differs")
    source_fps = geometry.get("source_fps")
    output_fps = geometry.get("required_output_fps")
    if (
        type(source_fps) not in (int, float)
        or not math.isfinite(float(source_fps))
        or float(source_fps) <= 0
        or float(output_fps) != float(source_fps)
    ):
        raise Wan22ReleaseError(f"{context} FPS contract differs")
    if geometry.get("maximum_duration_delta_frames") != 1:
        raise Wan22ReleaseError(f"{context} duration tolerance differs")
    requirements = _mapping(
        geometry.get("requirements"),
        context=f"{context}.strict_temporal_geometry.requirements",
    )
    if dict(requirements) != {
        "same_frame_count": True,
        "same_fps": True,
        "duration_absolute_delta_at_most_one_frame": True,
    }:
        raise Wan22ReleaseError(f"{context} temporal requirements differ")
    if (
        media.get("frame_count") != frame_count
        or float(media.get("fps")) != float(source_fps)
    ):
        raise Wan22ReleaseError(f"{context} media/geometry binding differs")

    verified = _mapping(
        row.get("finalizer_media_file_verification"),
        context=f"{context}.finalizer_media_file_verification",
    )
    if verified.get("schema_version") != MEDIA_FILE_VERIFICATION_SCHEMA:
        raise Wan22ReleaseError(f"{context} media verification schema differs")
    for label, path, digest in (
        ("source_video", source_path, source_sha),
        ("anchor_image", anchor_path, anchor_sha),
    ):
        item = _mapping(
            verified.get(label),
            context=f"{context}.finalizer_media_file_verification.{label}",
        )
        if (
            item.get("resolved_path") != str(path)
            or item.get("sha256") != digest
            or type(item.get("bytes")) is not int
            or item.get("bytes") <= 0
        ):
            raise Wan22ReleaseError(f"{context} {label} verification differs")
        if verify_media:
            resolved, raw = _stable_read(path, context=f"{context} {label}")
            if str(resolved) != str(path) or len(raw) != item["bytes"]:
                raise Wan22ReleaseError(f"{context} {label} path/size differs")
            if hashlib.sha256(raw).hexdigest() != digest:
                raise Wan22ReleaseError(f"{context} {label} SHA differs")
    expected_verification_digest = dict(verified)
    verification_digest = expected_verification_digest.pop(
        "verification_digest",
        None,
    )
    if verification_digest != _object_digest(expected_verification_digest):
        raise Wan22ReleaseError(f"{context} media verification digest differs")

    return {
        "iid": iid,
        "row_sha256": _object_digest(row),
        "edit_instruction_sha256": instruction_sha,
        "source_video_sha256": source_sha,
        "anchor_sha256": anchor_sha,
        "source_frame_count": frame_count,
        "source_fps": float(source_fps),
    }


def _validate_manifest(
    manifest_path: Path,
    *,
    expected_rows: int | None,
    verify_media: bool,
    allowed_schemas: frozenset[str] = frozenset({GENERATION_MANIFEST_SCHEMA}),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bytes, Path]:
    rows, _, raw, resolved = _load_jsonl(
        manifest_path,
        context="generation manifest",
    )
    if expected_rows is not None and len(rows) != expected_rows:
        raise Wan22ReleaseError(
            f"generation manifest must contain exactly {expected_rows} rows"
        )
    closures: list[dict[str, Any]] = []
    seen_iids: set[str] = set()
    seen_groups: set[str] = set()
    for index, row in enumerate(rows, start=1):
        closure = _validate_pending_v9_row(
            row,
            line_number=index,
            verify_media=verify_media,
            allowed_schemas=allowed_schemas,
        )
        iid = closure["iid"]
        group = str(row["group_id"])
        if iid in seen_iids or group in seen_groups:
            raise Wan22ReleaseError("generation manifest IID/group is duplicated")
        seen_iids.add(iid)
        seen_groups.add(group)
        closures.append(closure)
    return rows, closures, raw, resolved


def _acceptance_binding(path: Path) -> dict[str, Any]:
    value, raw, _ = _load_json(path, context="smoke acceptance result")
    if value.get("schema_version") != ACCEPTANCE_RESULT_SCHEMA:
        raise Wan22ReleaseError("smoke acceptance schema differs")
    expected_flags = {
        "passed": True,
        "full_123_authorized": True,
        "generation_authorized": False,
        "production_eligible": False,
        "wan_generation_authorized": False,
        "authorization_interface_available": False,
    }
    for field, expected in expected_flags.items():
        if value.get(field) is not expected:
            raise Wan22ReleaseError(f"smoke acceptance {field} differs")
    if value.get("failures") != []:
        raise Wan22ReleaseError("smoke acceptance contains failures")
    selected = _mapping(value.get("selected"), context="acceptance.selected")
    if selected.get("rows") != SMOKE_ROW_COUNT:
        raise Wan22ReleaseError("smoke acceptance is not the exact 16-row smoke")
    model = _mapping(value.get("model"), context="acceptance.model")
    model_path = _string(model.get("path"), context="acceptance.model.path")
    if model_path != QWEN3_MODEL_PATH:
        raise Wan22ReleaseError(
            "smoke acceptance model path is not the frozen Qwen3-VL-32B "
            "checkpoint"
        )
    closure = _mapping(
        value.get("model_closure"),
        context="acceptance.model_closure",
    )
    if closure.get("model_path") != model_path:
        raise Wan22ReleaseError("acceptance model closure path differs")
    source = _mapping(
        value.get("source_snapshot"),
        context="acceptance.source_snapshot",
    )
    implementations = _mapping(
        source.get("implementations"),
        context="acceptance.source_snapshot.implementations",
    )
    qwen_impl = _mapping(
        implementations.get("qwen"),
        context="acceptance Qwen implementation",
    )
    verifier_impl = _mapping(
        implementations.get("verifier"),
        context="acceptance verifier implementation",
    )
    binding = {
        "acceptance_result_sha256": hashlib.sha256(raw).hexdigest(),
        "acceptance_contract_sha256": _sha(
            _mapping(value.get("contract"), context="acceptance.contract").get(
                "sha256"
            ),
            context="acceptance contract SHA",
        ),
        "submission_contract_sha256": _sha(
            _mapping(
                value.get("submission_contract"),
                context="acceptance.submission_contract",
            ).get("sha256"),
            context="acceptance submission SHA",
        ),
        "completion_receipt_sha256": _sha(
            _mapping(
                value.get("completion_receipt"),
                context="acceptance.completion_receipt",
            ).get("sha256"),
            context="acceptance completion SHA",
        ),
        "selected_sha256": _sha(
            selected.get("sha256"),
            context="acceptance selected SHA",
        ),
        "selected_ordered_iids_sha256": _sha(
            selected.get("ordered_iids_sha256"),
            context="acceptance selected IID SHA",
        ),
        "model_path": model_path,
        "model_config_sha256": _sha(
            model.get("config_sha256"),
            context="acceptance model config SHA",
        ),
        "model_closure_manifest_sha256": _sha(
            closure.get("manifest_sha256"),
            context="acceptance model closure SHA",
        ),
        "model_closure_files_digest": _sha(
            closure.get("files_digest"),
            context="acceptance model closure file digest",
        ),
        "source_tree_sha256": _sha(
            source.get("tree_sha256"),
            context="acceptance source tree SHA",
        ),
        "qwen_implementation_sha256": _sha(
            qwen_impl.get("sha256"),
            context="acceptance Qwen implementation SHA",
        ),
        "verifier_implementation_sha256": _sha(
            verifier_impl.get("sha256"),
            context="acceptance verifier implementation SHA",
        ),
    }
    return binding


def _validate_signed_acceptance_binding(value: Any) -> dict[str, Any]:
    binding = _mapping(value, context="release smoke acceptance binding")
    expected_keys = {
        "acceptance_result_sha256",
        "acceptance_contract_sha256",
        "submission_contract_sha256",
        "completion_receipt_sha256",
        "selected_sha256",
        "selected_ordered_iids_sha256",
        "model_path",
        "model_config_sha256",
        "model_closure_manifest_sha256",
        "model_closure_files_digest",
        "source_tree_sha256",
        "qwen_implementation_sha256",
        "verifier_implementation_sha256",
    }
    _exact_keys(
        binding,
        expected_keys,
        context="release smoke acceptance binding",
    )
    if binding.get("model_path") != QWEN3_MODEL_PATH:
        raise Wan22ReleaseError(
            "release smoke binding is not the frozen Qwen3-VL-32B model"
        )
    for field in expected_keys - {"model_path"}:
        _sha(binding.get(field), context=f"release smoke binding {field}")
    return dict(binding)


def _validate_release_payload_shape_v2(value: Any) -> dict[str, Any]:
    signed = _mapping(value, context="scale512 release signed payload")
    _exact_keys(
        signed,
        {
            "schema_version",
            "profile",
            "release_id",
            "issued_at_utc",
            "purpose",
            "scale512_selection",
            "manifest",
            "row_authorizations",
            "prompt_policy",
            "temporal_policy",
        },
        context="scale512 release signed payload",
    )
    if (
        signed.get("schema_version") != RELEASE_PAYLOAD_SCHEMA_V2
        or signed.get("profile") != RELEASE_PROFILE_EXACT512_V2
        or signed.get("purpose") != "wan22_i2v_strict_motion_action_edit_512"
        or signed.get("prompt_policy") != PROMPT_POLICY
        or signed.get("temporal_policy") != TEMPORAL_POLICY
    ):
        raise Wan22ReleaseError("scale512 release payload policy differs")
    _string(signed.get("release_id"), context="release_id")
    issued = _string(signed.get("issued_at_utc"), context="issued_at_utc")
    try:
        issued_time = datetime.fromisoformat(issued.replace("Z", "+00:00"))
    except ValueError as error:
        raise Wan22ReleaseError("release issued_at_utc is invalid") from error
    if issued_time.tzinfo is None:
        raise Wan22ReleaseError("release issued_at_utc has no timezone")
    evidence_mode, evidence = _release_evidence(signed)
    if evidence_mode != "scale512_selection":
        raise Wan22ReleaseError("scale512 release evidence mode differs")
    retained_rows = (
        RELEASE_ROW_COUNT
        if evidence.get("retained_exact8_manifest") is not None
        else 0
    )
    manifest = _mapping(
        signed.get("manifest"), context="scale512 release manifest scope"
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "sha256",
            "bytes",
            "rows",
            "retained_exact8_rows",
            "scale512_rows",
            "ordered_iids_sha256",
            "ordered_row_sha256",
            "contiguous_shards_permitted",
            "contiguous_shard_rows",
        },
        context="scale512 release manifest scope",
    )
    if (
        manifest.get("schema_version") != EXACT512_MANIFEST_SCOPE_SCHEMA
        or manifest.get("rows") != SCALE512_RELEASE_ROW_COUNT
        or manifest.get("retained_exact8_rows") != retained_rows
        or manifest.get("scale512_rows")
        != SCALE512_RELEASE_ROW_COUNT - retained_rows
        or manifest.get("contiguous_shards_permitted") is not True
        or manifest.get("contiguous_shard_rows") != SCALE512_SHARD_ROW_COUNT
        or type(manifest.get("bytes")) is not int
        or manifest["bytes"] <= 0
    ):
        raise Wan22ReleaseError("scale512 release manifest scope differs")
    for field in ("sha256", "ordered_iids_sha256", "ordered_row_sha256"):
        _sha(manifest.get(field), context=f"scale512 release manifest {field}")
    rows = signed.get("row_authorizations")
    if not isinstance(rows, list) or len(rows) != SCALE512_RELEASE_ROW_COUNT:
        raise Wan22ReleaseError("scale512 release must authorize exactly 512 rows")
    seen: set[str] = set()
    for index, item in enumerate(rows):
        row = _mapping(item, context=f"scale512 row authorization {index}")
        _exact_keys(
            row,
            {
                "iid",
                "row_sha256",
                "edit_instruction_sha256",
                "source_video_sha256",
                "anchor_sha256",
                "source_frame_count",
                "source_fps",
            },
            context=f"scale512 row authorization {index}",
        )
        iid = _string(row.get("iid"), context=f"scale512 row {index} iid")
        if iid in seen or _IID_RE.fullmatch(iid) is None:
            raise Wan22ReleaseError("scale512 release row IID is unsafe or duplicated")
        seen.add(iid)
        for field in (
            "row_sha256",
            "edit_instruction_sha256",
            "source_video_sha256",
            "anchor_sha256",
        ):
            _sha(row.get(field), context=f"scale512 row {index} {field}")
        if (
            type(row.get("source_frame_count")) is not int
            or row["source_frame_count"] <= 0
            or row["source_frame_count"] % 4 != 1
            or type(row.get("source_fps")) not in (int, float)
            or not math.isfinite(float(row["source_fps"]))
            or float(row["source_fps"]) <= 0
        ):
            raise Wan22ReleaseError(
                f"scale512 row authorization {index} geometry is invalid"
            )
    return dict(signed)


def _validate_release_payload_shape(value: Any) -> dict[str, Any]:
    signed = _mapping(value, context="release signed payload")
    if signed.get("schema_version") == RELEASE_PAYLOAD_SCHEMA_V2:
        return _validate_release_payload_shape_v2(signed)
    common_keys = {
        "schema_version",
        "release_id",
        "issued_at_utc",
        "purpose",
        "manifest",
        "row_authorizations",
        "prompt_policy",
        "temporal_policy",
    }
    has_smoke = "smoke_acceptance" in signed
    has_finalizer = "finalizer_selection" in signed
    if has_smoke == has_finalizer:
        raise Wan22ReleaseError(
            "release must contain exactly one evidence mode"
        )
    _exact_keys(
        signed,
        common_keys
        | ({"smoke_acceptance"} if has_smoke else {"finalizer_selection"}),
        context="release signed payload",
    )
    if (
        signed.get("schema_version") != RELEASE_PAYLOAD_SCHEMA
        or signed.get("purpose")
        != "wan22_i2v_strict_motion_action_edit_8"
        or signed.get("prompt_policy") != PROMPT_POLICY
        or signed.get("temporal_policy") != TEMPORAL_POLICY
    ):
        raise Wan22ReleaseError("release payload policy differs")
    _string(signed.get("release_id"), context="release_id")
    issued = _string(signed.get("issued_at_utc"), context="issued_at_utc")
    try:
        issued_time = datetime.fromisoformat(issued.replace("Z", "+00:00"))
    except ValueError as error:
        raise Wan22ReleaseError("release issued_at_utc is invalid") from error
    if issued_time.tzinfo is None:
        raise Wan22ReleaseError("release issued_at_utc has no timezone")
    _release_evidence(signed)

    manifest = _mapping(signed.get("manifest"), context="release manifest scope")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "sha256",
            "bytes",
            "rows",
            "ordered_iids_sha256",
            "ordered_row_sha256",
            "contiguous_shards_permitted",
        },
        context="release manifest scope",
    )
    if (
        manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA
        or manifest.get("rows") != RELEASE_ROW_COUNT
        or manifest.get("contiguous_shards_permitted") is not True
        or type(manifest.get("bytes")) is not int
        or manifest["bytes"] <= 0
    ):
        raise Wan22ReleaseError("release manifest scope differs")
    for field in ("sha256", "ordered_iids_sha256", "ordered_row_sha256"):
        _sha(manifest.get(field), context=f"release manifest {field}")

    rows = signed.get("row_authorizations")
    if not isinstance(rows, list) or len(rows) != RELEASE_ROW_COUNT:
        raise Wan22ReleaseError("release must authorize exactly eight rows")
    seen: set[str] = set()
    for index, item in enumerate(rows):
        row = _mapping(item, context=f"release row authorization {index}")
        _exact_keys(
            row,
            {
                "iid",
                "row_sha256",
                "edit_instruction_sha256",
                "source_video_sha256",
                "anchor_sha256",
                "source_frame_count",
                "source_fps",
            },
            context=f"release row authorization {index}",
        )
        iid = _string(row.get("iid"), context=f"release row {index} iid")
        if iid in seen or _IID_RE.fullmatch(iid) is None:
            raise Wan22ReleaseError("release row IID is unsafe or duplicated")
        seen.add(iid)
        for field in (
            "row_sha256",
            "edit_instruction_sha256",
            "source_video_sha256",
            "anchor_sha256",
        ):
            _sha(row.get(field), context=f"release row {index} {field}")
        if (
            type(row.get("source_frame_count")) is not int
            or row["source_frame_count"] <= 0
            or row["source_frame_count"] % 4 != 1
            or type(row.get("source_fps")) not in (int, float)
            or not math.isfinite(float(row["source_fps"]))
            or float(row["source_fps"]) <= 0
        ):
            raise Wan22ReleaseError(
                f"release row authorization {index} geometry is invalid"
            )
    return dict(signed)


def _payload(
    *,
    manifest_path: Path,
    release_id: str,
    issued_at_utc: str,
    smoke_acceptance_path: Path | None = None,
    finalizer_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (smoke_acceptance_path is None) == (finalizer_selection is None):
        raise Wan22ReleaseError(
            "exactly one release evidence mode must be supplied"
        )
    rows, closures, raw, resolved_manifest = _validate_manifest(
        manifest_path,
        expected_rows=RELEASE_ROW_COUNT,
        verify_media=True,
    )
    _string(release_id, context="release_id")
    timestamp = _string(issued_at_utc, context="issued_at_utc")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise Wan22ReleaseError("issued_at_utc is invalid") from error
    if parsed.tzinfo is None:
        raise Wan22ReleaseError("issued_at_utc must include a timezone")
    evidence: dict[str, Any]
    if smoke_acceptance_path is not None:
        evidence = {
            "smoke_acceptance": _acceptance_binding(smoke_acceptance_path)
        }
    else:
        selection = _validate_signed_finalizer_selection_binding(
            finalizer_selection
        )
        selected_manifest = selection["selected_manifest"]
        if (
            selected_manifest["path"] != str(resolved_manifest)
            or selected_manifest["sha256"] != hashlib.sha256(raw).hexdigest()
            or selected_manifest["bytes"] != len(raw)
            or selected_manifest["rows"] != len(rows)
        ):
            raise Wan22ReleaseError(
                "finalizer selection evidence differs from requested manifest"
            )
        evidence = {"finalizer_selection": selection}
    return {
        "schema_version": RELEASE_PAYLOAD_SCHEMA,
        "release_id": release_id,
        "issued_at_utc": issued_at_utc,
        "purpose": "wan22_i2v_strict_motion_action_edit_8",
        **evidence,
        "manifest": {
            "schema_version": GENERATION_MANIFEST_SCHEMA,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": RELEASE_ROW_COUNT,
            "ordered_iids_sha256": _ordered_digest(
                [str(row["iid"]) for row in rows]
            ),
            "ordered_row_sha256": _ordered_digest(
                [item["row_sha256"] for item in closures]
            ),
            "contiguous_shards_permitted": True,
        },
        "row_authorizations": closures,
        "prompt_policy": dict(PROMPT_POLICY),
        "temporal_policy": dict(TEMPORAL_POLICY),
    }


def _payload_scale512(
    *,
    manifest_path: Path,
    release_id: str,
    issued_at_utc: str,
    scale512_selection: Mapping[str, Any],
) -> dict[str, Any]:
    selection = _validate_signed_scale512_selection_binding(
        scale512_selection
    )
    allowed_schemas = frozenset(
        {GENERATION_MANIFEST_SCHEMA, SCALE512_GENERATION_MANIFEST_SCHEMA}
    )
    rows, closures, raw, resolved_manifest = _validate_manifest(
        manifest_path,
        expected_rows=SCALE512_RELEASE_ROW_COUNT,
        verify_media=True,
        allowed_schemas=allowed_schemas,
    )
    _string(release_id, context="release_id")
    timestamp = _string(issued_at_utc, context="issued_at_utc")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise Wan22ReleaseError("issued_at_utc is invalid") from error
    if parsed.tzinfo is None:
        raise Wan22ReleaseError("issued_at_utc must include a timezone")
    selected_manifest = selection["selected_manifest"]
    if (
        selected_manifest["path"] != str(resolved_manifest)
        or selected_manifest["sha256"] != hashlib.sha256(raw).hexdigest()
        or selected_manifest["bytes"] != len(raw)
        or selected_manifest["rows"] != len(rows)
    ):
        raise Wan22ReleaseError(
            "scale512 selection evidence differs from requested manifest"
        )
    retained = selection["retained_exact8_manifest"]
    retained_rows = RELEASE_ROW_COUNT if retained is not None else 0
    for index, row in enumerate(rows):
        expected_schema = (
            GENERATION_MANIFEST_SCHEMA
            if index < retained_rows
            else SCALE512_GENERATION_MANIFEST_SCHEMA
        )
        if row.get("schema_version") != expected_schema:
            raise Wan22ReleaseError(
                "scale512 selected manifest row schema/order differs"
            )
    if retained is not None:
        retained_values, _, retained_raw, retained_path = _validate_manifest(
            Path(retained["path"]),
            expected_rows=RELEASE_ROW_COUNT,
            verify_media=True,
        )
        if (
            str(retained_path) != retained["path"]
            or hashlib.sha256(retained_raw).hexdigest() != retained["sha256"]
            or len(retained_raw) != retained["bytes"]
            or rows[:RELEASE_ROW_COUNT] != retained_values
            or not raw.startswith(retained_raw)
        ):
            raise Wan22ReleaseError(
                "scale512 retained exact8 manifest binding differs"
            )
    return {
        "schema_version": RELEASE_PAYLOAD_SCHEMA_V2,
        "profile": RELEASE_PROFILE_EXACT512_V2,
        "release_id": release_id,
        "issued_at_utc": issued_at_utc,
        "purpose": "wan22_i2v_strict_motion_action_edit_512",
        "scale512_selection": selection,
        "manifest": {
            "schema_version": EXACT512_MANIFEST_SCOPE_SCHEMA,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": SCALE512_RELEASE_ROW_COUNT,
            "retained_exact8_rows": retained_rows,
            "scale512_rows": SCALE512_RELEASE_ROW_COUNT - retained_rows,
            "ordered_iids_sha256": _ordered_digest(
                [str(row["iid"]) for row in rows]
            ),
            "ordered_row_sha256": _ordered_digest(
                [item["row_sha256"] for item in closures]
            ),
            "contiguous_shards_permitted": True,
            "contiguous_shard_rows": SCALE512_SHARD_ROW_COUNT,
        },
        "row_authorizations": closures,
        "prompt_policy": dict(PROMPT_POLICY),
        "temporal_policy": dict(TEMPORAL_POLICY),
    }


def _verify_signature(
    signed: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    namespace: str = SIGNATURE_NAMESPACE,
) -> None:
    _require_dedicated_signer_anchor()
    expected_signature_keys = {
        "format",
        "namespace",
        "principal",
        "key_fingerprint",
        "armored_signature_base64",
    }
    _exact_keys(signature, expected_signature_keys, context="release signature")
    expected = {
        "format": "SSHSIG",
        "namespace": namespace,
        "principal": SIGNER_PRINCIPAL,
        "key_fingerprint": SIGNER_KEY_FINGERPRINT,
    }
    for field, value in expected.items():
        if signature.get(field) != value:
            raise Wan22ReleaseError(f"release signature {field} differs")
    try:
        armor = base64.b64decode(
            _string(
                signature.get("armored_signature_base64"),
                context="release signature bytes",
            ),
            validate=True,
        )
    except (ValueError, TypeError) as error:
        raise Wan22ReleaseError("release signature is not strict base64") from error
    if not (
        armor.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
        and armor.endswith(b"-----END SSH SIGNATURE-----\n")
    ):
        raise Wan22ReleaseError("release signature armor differs")
    with tempfile.TemporaryDirectory(prefix="motive-wan22-verify-") as temporary:
        root = Path(temporary)
        allowed = root / "allowed_signers"
        signature_path = root / "release.sshsig"
        allowed.write_text(
            f"{SIGNER_PRINCIPAL} {SIGNER_PUBLIC_KEY}\n",
            encoding="utf-8",
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
                namespace,
                "-s",
                str(signature_path),
            ],
            input=_canonical_bytes(signed),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        raise Wan22ReleaseError(
            "release SSH signature verification failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )


def _verify_signed_release_v2(
    *,
    signed: Mapping[str, Any],
    resolved_release: Path,
    manifest_path: Path,
    require_exact_manifest: bool,
    verify_media: bool,
) -> dict[str, Any]:
    evidence_mode, evidence_binding = _release_evidence(signed)
    if evidence_mode != "scale512_selection":
        raise Wan22ReleaseError("scale512 release evidence mode differs")
    manifest_scope = _mapping(
        signed.get("manifest"), context="scale512 release manifest scope"
    )
    evidence_binding = _verify_scale512_selection_evidence(
        evidence_binding,
        manifest_scope=manifest_scope,
    )
    authorizations = signed.get("row_authorizations")
    if (
        not isinstance(authorizations, list)
        or len(authorizations) != SCALE512_RELEASE_ROW_COUNT
    ):
        raise Wan22ReleaseError("scale512 release must authorize exactly 512 rows")
    retained_rows = manifest_scope["retained_exact8_rows"]
    allowed_schemas = frozenset(
        {GENERATION_MANIFEST_SCHEMA, SCALE512_GENERATION_MANIFEST_SCHEMA}
    )
    rows, closures, raw, resolved_manifest = _validate_manifest(
        manifest_path,
        expected_rows=(
            SCALE512_RELEASE_ROW_COUNT if require_exact_manifest else None
        ),
        verify_media=verify_media,
        allowed_schemas=allowed_schemas,
    )
    if not require_exact_manifest and len(rows) != SCALE512_SHARD_ROW_COUNT:
        raise Wan22ReleaseError(
            "scale512 release shard must contain exactly eight rows"
        )
    by_iid = {
        str(item["iid"]): (index, dict(item))
        for index, item in enumerate(authorizations)
        if isinstance(item, Mapping)
    }
    indices: list[int] = []
    for closure in closures:
        entry = by_iid.get(closure["iid"])
        if entry is None or entry[1] != closure:
            raise Wan22ReleaseError(
                f"manifest row {closure['iid']} is outside scale512 release scope"
            )
        indices.append(entry[0])
    if not indices or indices != list(
        range(indices[0], indices[0] + len(indices))
    ):
        raise Wan22ReleaseError(
            "requested manifest is not one contiguous scale512 release shard"
        )
    for row, root_index in zip(rows, indices, strict=True):
        expected_schema = (
            GENERATION_MANIFEST_SCHEMA
            if root_index < retained_rows
            else SCALE512_GENERATION_MANIFEST_SCHEMA
        )
        if row.get("schema_version") != expected_schema:
            raise Wan22ReleaseError(
                "scale512 release shard row schema/order differs"
            )
    if require_exact_manifest:
        if (
            hashlib.sha256(raw).hexdigest() != manifest_scope["sha256"]
            or len(raw) != manifest_scope["bytes"]
            or indices != list(range(SCALE512_RELEASE_ROW_COUNT))
            or _ordered_digest([str(row["iid"]) for row in rows])
            != manifest_scope["ordered_iids_sha256"]
            or _ordered_digest([item["row_sha256"] for item in closures])
            != manifest_scope["ordered_row_sha256"]
        ):
            raise Wan22ReleaseError("exact scale512 manifest binding differs")
    prepared: list[dict[str, Any]] = []
    for line_number, (row, closure) in enumerate(
        zip(rows, closures, strict=True), start=1
    ):
        item = dict(row)
        item["_iid"] = closure["iid"]
        item["_line_number"] = line_number
        item["_row_digest"] = closure["row_sha256"]
        item["_authorization_mode"] = AUTHORIZATION_MODE
        item["_signed_release"] = {
            "path": str(resolved_release),
            "release_id": signed["release_id"],
            "payload_sha256": _object_digest(signed),
            "signer_key_fingerprint": SIGNER_KEY_FINGERPRINT,
        }
        prepared.append(item)
    release_result = {
        "path": str(resolved_release),
        "release_id": signed["release_id"],
        "profile": RELEASE_PROFILE_EXACT512_V2,
        "payload_sha256": _object_digest(signed),
        "signer_key_fingerprint": SIGNER_KEY_FINGERPRINT,
        "root_manifest_sha256": manifest_scope["sha256"],
        "root_manifest_rows": SCALE512_RELEASE_ROW_COUNT,
        "scale512_selection": evidence_binding,
    }
    return {
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_bytes": len(raw),
        "manifest_row_count": len(rows),
        "selected_rows": prepared,
        "selected_row_count": len(prepared),
        "release": release_result,
    }


def verify_signed_release(
    *,
    release_path: str | Path,
    manifest_path: str | Path,
    require_exact_manifest: bool,
    verify_media: bool = True,
) -> dict[str, Any]:
    """Verify the signed eight-row scope and return authorized manifest rows."""

    envelope, _, resolved_release = _load_json(
        Path(release_path),
        context="signed generation release",
    )
    _exact_keys(
        envelope,
        {"schema_version", "signed", "signature"},
        context="signed generation release",
    )
    envelope_schema = envelope.get("schema_version")
    if envelope_schema not in {RELEASE_SCHEMA, RELEASE_SCHEMA_V2}:
        raise Wan22ReleaseError("signed generation release schema differs")
    signed = _mapping(envelope.get("signed"), context="release signed payload")
    signature = _mapping(envelope.get("signature"), context="release signature")
    namespace = (
        SIGNATURE_NAMESPACE_V2
        if envelope_schema == RELEASE_SCHEMA_V2
        else SIGNATURE_NAMESPACE
    )
    _verify_signature(signed, signature, namespace=namespace)
    _validate_release_payload_shape(signed)
    if envelope_schema == RELEASE_SCHEMA_V2:
        if signed.get("schema_version") != RELEASE_PAYLOAD_SCHEMA_V2:
            raise Wan22ReleaseError("scale512 release payload schema differs")
        return _verify_signed_release_v2(
            signed=signed,
            resolved_release=resolved_release,
            manifest_path=Path(manifest_path),
            require_exact_manifest=require_exact_manifest,
            verify_media=verify_media,
        )
    evidence_mode, evidence_binding = _release_evidence(signed)
    if signed.get("schema_version") != RELEASE_PAYLOAD_SCHEMA:
        raise Wan22ReleaseError("release payload schema differs")
    if signed.get("purpose") != "wan22_i2v_strict_motion_action_edit_8":
        raise Wan22ReleaseError("release purpose differs")
    if signed.get("prompt_policy") != PROMPT_POLICY:
        raise Wan22ReleaseError("release prompt policy differs")
    if signed.get("temporal_policy") != TEMPORAL_POLICY:
        raise Wan22ReleaseError("release temporal policy differs")
    _string(signed.get("release_id"), context="release_id")
    issued = _string(signed.get("issued_at_utc"), context="issued_at_utc")
    try:
        issued_time = datetime.fromisoformat(issued.replace("Z", "+00:00"))
    except ValueError as error:
        raise Wan22ReleaseError("release issued_at_utc is invalid") from error
    if issued_time.tzinfo is None:
        raise Wan22ReleaseError("release issued_at_utc has no timezone")
    manifest_scope = _mapping(
        signed.get("manifest"),
        context="release manifest scope",
    )
    expected_manifest_scope_keys = {
        "schema_version",
        "sha256",
        "bytes",
        "rows",
        "ordered_iids_sha256",
        "ordered_row_sha256",
        "contiguous_shards_permitted",
    }
    _exact_keys(
        manifest_scope,
        expected_manifest_scope_keys,
        context="release manifest scope",
    )
    if (
        manifest_scope.get("schema_version") != GENERATION_MANIFEST_SCHEMA
        or manifest_scope.get("rows") != RELEASE_ROW_COUNT
        or manifest_scope.get("contiguous_shards_permitted") is not True
    ):
        raise Wan22ReleaseError("release manifest scope differs")
    _sha(manifest_scope.get("sha256"), context="release manifest SHA")
    _sha(
        manifest_scope.get("ordered_iids_sha256"),
        context="release ordered IID SHA",
    )
    _sha(
        manifest_scope.get("ordered_row_sha256"),
        context="release ordered row SHA",
    )
    if type(manifest_scope.get("bytes")) is not int or manifest_scope["bytes"] <= 0:
        raise Wan22ReleaseError("release manifest byte count is invalid")
    if evidence_mode == "finalizer_selection":
        evidence_binding = _verify_finalizer_selection_evidence(
            evidence_binding,
            manifest_scope=manifest_scope,
        )

    authorizations = signed.get("row_authorizations")
    if not isinstance(authorizations, list) or len(authorizations) != RELEASE_ROW_COUNT:
        raise Wan22ReleaseError("release must authorize exactly eight rows")
    for index, item in enumerate(authorizations):
        if not isinstance(item, Mapping):
            raise Wan22ReleaseError(f"release row authorization {index} is invalid")
        _exact_keys(
            item,
            {
                "iid",
                "row_sha256",
                "edit_instruction_sha256",
                "source_video_sha256",
                "anchor_sha256",
                "source_frame_count",
                "source_fps",
            },
            context=f"release row authorization {index}",
        )
        _string(item["iid"], context=f"release row {index} iid")
        for field in (
            "row_sha256",
            "edit_instruction_sha256",
            "source_video_sha256",
            "anchor_sha256",
        ):
            _sha(item[field], context=f"release row {index} {field}")

    rows, closures, raw, resolved_manifest = _validate_manifest(
        Path(manifest_path),
        expected_rows=RELEASE_ROW_COUNT if require_exact_manifest else None,
        verify_media=verify_media,
    )
    by_iid = {
        str(item["iid"]): (index, dict(item))
        for index, item in enumerate(authorizations)
    }
    indices: list[int] = []
    for closure in closures:
        entry = by_iid.get(closure["iid"])
        if entry is None or entry[1] != closure:
            raise Wan22ReleaseError(
                f"manifest row {closure['iid']} is outside signed release scope"
            )
        indices.append(entry[0])
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise Wan22ReleaseError("requested manifest is not one contiguous release shard")
    if require_exact_manifest:
        if (
            hashlib.sha256(raw).hexdigest() != manifest_scope["sha256"]
            or len(raw) != manifest_scope["bytes"]
            or indices != list(range(RELEASE_ROW_COUNT))
            or _ordered_digest([str(row["iid"]) for row in rows])
            != manifest_scope["ordered_iids_sha256"]
            or _ordered_digest([item["row_sha256"] for item in closures])
            != manifest_scope["ordered_row_sha256"]
        ):
            raise Wan22ReleaseError("exact generation manifest binding differs")

    prepared: list[dict[str, Any]] = []
    for line_number, (row, closure) in enumerate(
        zip(rows, closures, strict=True),
        start=1,
    ):
        item = dict(row)
        item["_iid"] = closure["iid"]
        item["_line_number"] = line_number
        item["_row_digest"] = closure["row_sha256"]
        item["_authorization_mode"] = AUTHORIZATION_MODE
        item["_signed_release"] = {
            "path": str(resolved_release),
            "release_id": signed["release_id"],
            "payload_sha256": _object_digest(signed),
            "signer_key_fingerprint": SIGNER_KEY_FINGERPRINT,
        }
        prepared.append(item)
    release_result = {
        "path": str(resolved_release),
        "release_id": signed["release_id"],
        "payload_sha256": _object_digest(signed),
        "signer_key_fingerprint": SIGNER_KEY_FINGERPRINT,
        "root_manifest_sha256": manifest_scope["sha256"],
        "root_manifest_rows": RELEASE_ROW_COUNT,
    }
    if evidence_mode == "smoke_acceptance":
        release_result["smoke_acceptance"] = evidence_binding
    else:
        release_result["finalizer_selection"] = evidence_binding
    return {
        "manifest_path": str(resolved_manifest),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_bytes": len(raw),
        "manifest_row_count": len(rows),
        "selected_rows": prepared,
        "selected_row_count": len(prepared),
        "release": release_result,
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
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
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


def _rerun_acceptance(smoke_acceptance_path: Path) -> dict[str, Any]:
    """Execute the frozen sibling verifier and require byte-identical success."""

    existing, existing_raw, _ = _load_json(
        smoke_acceptance_path,
        context="existing smoke acceptance result",
    )
    source = _mapping(
        existing.get("source_snapshot"),
        context="existing acceptance source snapshot",
    )
    implementations = _mapping(
        source.get("implementations"),
        context="existing acceptance source implementations",
    )
    verifier_record = _mapping(
        implementations.get("verifier"),
        context="existing acceptance verifier record",
    )
    verifier = Path(__file__).resolve().with_name(
        "goku_action_v13_acceptance.py"
    )
    verifier_resolved, verifier_raw = _stable_read(
        verifier,
        context="frozen sibling acceptance verifier",
    )
    if (
        verifier_record.get("path") != str(verifier_resolved)
        or verifier_record.get("sha256")
        != hashlib.sha256(verifier_raw).hexdigest()
    ):
        raise Wan22ReleaseError(
            "acceptance result is not bound to the executed sibling verifier"
        )
    path_bindings = {
        "--contract": _mapping(
            existing.get("contract"),
            context="acceptance contract",
        ).get("path"),
        "--smoke-gold": _mapping(
            existing.get("smoke_gold"),
            context="acceptance smoke gold",
        ).get("path"),
        "--selected": _mapping(
            existing.get("selected"),
            context="acceptance selected",
        ).get("path"),
        "--qwen-root": _mapping(
            existing.get("qwen"),
            context="acceptance qwen",
        ).get("root"),
        "--final-dir": _mapping(
            existing.get("final"),
            context="acceptance final",
        ).get("path"),
        "--source-snapshot": source.get("path"),
        "--submission-contract": _mapping(
            existing.get("submission_contract"),
            context="acceptance submission contract",
        ).get("path"),
        "--completion-receipt": _mapping(
            existing.get("completion_receipt"),
            context="acceptance completion receipt",
        ).get("path"),
    }
    command = [sys.executable, str(verifier_resolved)]
    for option, raw_path in path_bindings.items():
        path_text = _string(raw_path, context=f"acceptance rerun {option}")
        if not Path(path_text).is_absolute():
            raise Wan22ReleaseError(
                f"acceptance rerun {option} path is not absolute"
            )
        command.extend((option, path_text))
    with tempfile.TemporaryDirectory(
        prefix="motive-wan22-acceptance-rerun-"
    ) as temporary:
        rerun_path = Path(temporary) / "acceptance_result.json"
        command.extend(("--output", str(rerun_path)))
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or not rerun_path.is_file():
            raise Wan22ReleaseError(
                "independent acceptance verifier rerun failed: "
                + completed.stderr.decode("utf-8", errors="replace").strip()
            )
        rerun, rerun_raw, _ = _load_json(
            rerun_path,
            context="rerun smoke acceptance result",
        )
    if rerun_raw != existing_raw or rerun != existing:
        raise Wan22ReleaseError(
            "rerun acceptance result differs byte-for-byte from the "
            "candidate acceptance result"
        )
    # Re-apply the release-specific Qwen3 and closed success checks.
    _acceptance_binding(smoke_acceptance_path)
    return existing


def _binding_from_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _resolved_finalizer_directory(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or expanded == Path("/"):
        raise Wan22ReleaseError(
            "finalizer directory must be a non-root absolute path"
        )
    if expanded.is_symlink() or not expanded.is_dir():
        raise Wan22ReleaseError(
            "finalizer directory must be a non-symlink directory"
        )
    return expanded.resolve(strict=True)


def _validate_exact8_receipt(
    value: Mapping[str, Any],
    *,
    selected_manifest_raw: bytes,
    done_raw: bytes,
    summary_raw: bytes,
    finalizer_implementation_raw: bytes,
) -> None:
    context = "exact-eight selection receipt"
    _exact_keys(
        value,
        {"schema_version", "policy", "parent", "selection"},
        context=context,
    )
    if value.get("schema_version") != EXACT8_SELECTION_RECEIPT_SCHEMA:
        raise Wan22ReleaseError("exact-eight selection receipt schema differs")
    if value.get("policy") != EXACT8_SELECTION_POLICY:
        raise Wan22ReleaseError("exact-eight selection receipt policy differs")
    parent = _mapping(value.get("parent"), context=f"{context}.parent")
    _exact_keys(
        parent,
        {
            "done_sha256",
            "summary_sha256",
            "review_candidates_sha256",
            "generation_manifest_sha256",
            "finalizer_implementation_sha256",
        },
        context=f"{context}.parent",
    )
    if parent.get("done_sha256") != hashlib.sha256(done_raw).hexdigest():
        raise Wan22ReleaseError("selection receipt done binding differs")
    if parent.get("summary_sha256") != hashlib.sha256(summary_raw).hexdigest():
        raise Wan22ReleaseError("selection receipt summary binding differs")
    if parent.get("finalizer_implementation_sha256") != hashlib.sha256(
        finalizer_implementation_raw
    ).hexdigest():
        raise Wan22ReleaseError(
            "selection receipt finalizer implementation binding differs"
        )
    for field in ("review_candidates_sha256", "generation_manifest_sha256"):
        _sha(parent.get(field), context=f"{context}.parent.{field}")

    selection = _mapping(
        value.get("selection"),
        context=f"{context}.selection",
    )
    _exact_keys(
        selection,
        {
            "row_count",
            "ordered_iids",
            "ordered_review_ranks",
            "output_file",
            "output_sha256",
            "output_bytes",
        },
        context=f"{context}.selection",
    )
    if (
        selection.get("row_count") != RELEASE_ROW_COUNT
        or selection.get("output_file") != EXACT8_MANIFEST_NAME
        or selection.get("output_sha256")
        != hashlib.sha256(selected_manifest_raw).hexdigest()
        or selection.get("output_bytes") != len(selected_manifest_raw)
    ):
        raise Wan22ReleaseError("selection receipt manifest binding differs")
    ordered_iids = selection.get("ordered_iids")
    ordered_ranks = selection.get("ordered_review_ranks")
    if (
        not isinstance(ordered_iids, list)
        or len(ordered_iids) != RELEASE_ROW_COUNT
        or any(
            not isinstance(iid, str) or _IID_RE.fullmatch(iid) is None
            for iid in ordered_iids
        )
        or len(set(ordered_iids)) != RELEASE_ROW_COUNT
    ):
        raise Wan22ReleaseError("selection receipt ordered IIDs differ")
    if (
        not isinstance(ordered_ranks, list)
        or len(ordered_ranks) != RELEASE_ROW_COUNT
        or any(type(rank) is not int or rank <= 0 for rank in ordered_ranks)
        or len(set(ordered_ranks)) != RELEASE_ROW_COUNT
        or ordered_ranks != sorted(ordered_ranks)
    ):
        raise Wan22ReleaseError("selection receipt ordered ranks differ")


def _rerun_finalizer_selection(
    *,
    manifest_path: Path,
    finalizer_dir: Path,
    selection_receipt_path: Path,
) -> dict[str, Any]:
    """Rerun the sibling selector and bind byte-identical exact-eight output."""

    finalizer = _resolved_finalizer_directory(finalizer_dir)
    manifest_resolved, manifest_raw = _stable_read(
        manifest_path,
        context="requested exact-eight manifest",
    )
    receipt, receipt_raw, receipt_resolved = _load_json(
        selection_receipt_path,
        context="requested exact-eight selection receipt",
    )
    done_resolved, done_raw = _stable_read(
        finalizer / FINALIZER_DONE_NAME,
        context="finalizer done receipt",
    )
    summary_resolved, summary_raw = _stable_read(
        finalizer / FINALIZER_SUMMARY_NAME,
        context="finalizer summary",
    )
    selector_path = Path(__file__).resolve().with_name(EXACT8_SELECTOR_NAME)
    selector_resolved, selector_raw = _stable_read(
        selector_path,
        context="exact-eight selector implementation",
    )
    finalizer_implementation_path = selector_resolved.with_name(
        FINALIZER_IMPLEMENTATION_NAME
    )
    finalizer_implementation_resolved, finalizer_implementation_raw = (
        _stable_read(
            finalizer_implementation_path,
            context="finalizer implementation sibling",
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="motive-wan22-selector-rerun-"
    ) as temporary:
        # ``TemporaryDirectory`` may expose a symlinked spelling such as
        # ``/var`` while ``_stable_read`` returns the canonical ``/private/var``
        # spelling.  Canonicalize before invoking the selector so the escape
        # check compares equivalent paths on macOS as well as Linux.
        rerun_root = Path(temporary).resolve(strict=True) / "exact8"
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                str(selector_resolved),
                "--finalizer-dir",
                str(finalizer),
                "--output-dir",
                str(rerun_root),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            diagnostic = completed.stderr.decode(
                "utf-8", errors="replace"
            ).strip()
            raise Wan22ReleaseError(
                "exact-eight selector rerun failed: " + diagnostic
            )
        rerun_manifest_resolved, rerun_manifest_raw = _stable_read(
            rerun_root / EXACT8_MANIFEST_NAME,
            context="rerun exact-eight manifest",
        )
        rerun_receipt, rerun_receipt_raw, _ = _load_json(
            rerun_root / EXACT8_SELECTION_RECEIPT_NAME,
            context="rerun exact-eight selection receipt",
        )
        if rerun_manifest_resolved.parent != rerun_root.resolve(strict=True):
            raise Wan22ReleaseError("rerun exact-eight output escaped its root")

    if rerun_manifest_raw != manifest_raw:
        raise Wan22ReleaseError(
            "rerun exact-eight manifest is not byte-identical to the request"
        )
    if rerun_receipt_raw != receipt_raw or rerun_receipt != receipt:
        raise Wan22ReleaseError(
            "rerun exact-eight selection receipt is not byte-identical to the request"
        )
    _validate_exact8_receipt(
        receipt,
        selected_manifest_raw=manifest_raw,
        done_raw=done_raw,
        summary_raw=summary_raw,
        finalizer_implementation_raw=finalizer_implementation_raw,
    )
    binding = {
        "schema_version": FINALIZER_SELECTION_EVIDENCE_SCHEMA,
        "mode": FINALIZER_SELECTION_EVIDENCE_MODE,
        "finalizer_dir": str(finalizer),
        "done": _binding_from_bytes(done_resolved, done_raw),
        "summary": _binding_from_bytes(summary_resolved, summary_raw),
        "selection_receipt": _binding_from_bytes(
            receipt_resolved,
            receipt_raw,
        ),
        "selector_implementation": _binding_from_bytes(
            selector_resolved,
            selector_raw,
        ),
        "finalizer_implementation": _binding_from_bytes(
            finalizer_implementation_resolved,
            finalizer_implementation_raw,
        ),
        "selected_manifest": {
            **_binding_from_bytes(manifest_resolved, manifest_raw),
            "rows": RELEASE_ROW_COUNT,
        },
    }
    return _validate_signed_finalizer_selection_binding(binding)


def _verify_finalizer_selection_evidence(
    value: Any,
    *,
    manifest_scope: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _validate_signed_finalizer_selection_binding(value)
    rebuilt = _rerun_finalizer_selection(
        manifest_path=Path(binding["selected_manifest"]["path"]),
        finalizer_dir=Path(binding["finalizer_dir"]),
        selection_receipt_path=Path(binding["selection_receipt"]["path"]),
    )
    if rebuilt != binding:
        raise Wan22ReleaseError(
            "current finalizer selection closure differs from signed evidence"
        )
    selected = binding["selected_manifest"]
    if (
        selected["sha256"] != manifest_scope.get("sha256")
        or selected["bytes"] != manifest_scope.get("bytes")
        or selected["rows"] != manifest_scope.get("rows")
    ):
        raise Wan22ReleaseError(
            "finalizer selection evidence differs from signed manifest scope"
        )
    return binding


def _receipt_retained_exact8_source(
    receipt: Mapping[str, Any],
) -> dict[str, Any] | None:
    retention = _mapping(
        receipt.get("retention"), context="exact512 receipt retention"
    )
    source_value = retention.get("source")
    if source_value is None:
        return None
    source = _mapping(
        source_value, context="exact512 receipt retention source"
    )
    _exact_keys(
        source,
        {"path", "sha256", "bytes", "row_count"},
        context="exact512 receipt retention source",
    )
    path = _absolute_path_text(
        source.get("path"), context="exact512 retained manifest path"
    )
    digest = _sha(
        source.get("sha256"), context="exact512 retained manifest SHA"
    )
    size = source.get("bytes")
    if type(size) is not int or size <= 0:
        raise Wan22ReleaseError(
            "exact512 retained manifest bytes must be a positive integer"
        )
    if source.get("row_count") != RELEASE_ROW_COUNT:
        raise Wan22ReleaseError(
            "exact512 retained manifest must contain exactly eight rows"
        )
    return {
        "path": path,
        "sha256": digest,
        "bytes": size,
        "rows": RELEASE_ROW_COUNT,
    }


def _validate_exact512_receipt(
    value: Mapping[str, Any],
    *,
    selected_manifest_raw: bytes,
    done_raw: bytes,
    summary_raw: bytes,
    selector_raw: bytes,
    shared_selector_raw: bytes,
    finalizer_implementation_raw: bytes,
    retained_manifest_raw: bytes | None,
) -> dict[str, Any] | None:
    context = "exact512 selection receipt"
    _exact_keys(
        value,
        {
            "schema_version",
            "policy",
            "implementation",
            "parent",
            "retention",
            "selection",
        },
        context=context,
    )
    if value.get("schema_version") != EXACT512_SELECTION_RECEIPT_SCHEMA:
        raise Wan22ReleaseError("exact512 selection receipt schema differs")
    implementation = _mapping(
        value.get("implementation"), context=f"{context}.implementation"
    )
    _exact_keys(
        implementation,
        {"selector_sha256", "shared_strict_io_sha256"},
        context=f"{context}.implementation",
    )
    if implementation.get("selector_sha256") != hashlib.sha256(
        selector_raw
    ).hexdigest():
        raise Wan22ReleaseError("exact512 selector source binding differs")
    if implementation.get("shared_strict_io_sha256") != hashlib.sha256(
        shared_selector_raw
    ).hexdigest():
        raise Wan22ReleaseError("exact512 shared selector source binding differs")
    parent = _mapping(value.get("parent"), context=f"{context}.parent")
    _exact_keys(
        parent,
        {
            "done_sha256",
            "summary_sha256",
            "review_candidates_sha256",
            "proposed_512_sha256",
            "reserve_128_sha256",
            "generation_manifest_sha256",
            "finalizer_implementation_sha256",
        },
        context=f"{context}.parent",
    )
    if parent.get("done_sha256") != hashlib.sha256(done_raw).hexdigest():
        raise Wan22ReleaseError("exact512 receipt done binding differs")
    if parent.get("summary_sha256") != hashlib.sha256(summary_raw).hexdigest():
        raise Wan22ReleaseError("exact512 receipt summary binding differs")
    if parent.get("finalizer_implementation_sha256") != hashlib.sha256(
        finalizer_implementation_raw
    ).hexdigest():
        raise Wan22ReleaseError("exact512 receipt finalizer source differs")
    for field in (
        "review_candidates_sha256",
        "proposed_512_sha256",
        "reserve_128_sha256",
        "generation_manifest_sha256",
    ):
        _sha(parent.get(field), context=f"{context}.parent.{field}")

    retained = _receipt_retained_exact8_source(value)
    retention = _mapping(
        value.get("retention"), context=f"{context}.retention"
    )
    _exact_keys(
        retention,
        {"retained_row_count", "source", "ordered_iids", "ordered_group_ids"},
        context=f"{context}.retention",
    )
    retained_count = RELEASE_ROW_COUNT if retained is not None else 0
    if retention.get("retained_row_count") != retained_count:
        raise Wan22ReleaseError("exact512 receipt retained row count differs")
    if (retained_manifest_raw is None) != (retained is None):
        raise Wan22ReleaseError("exact512 retained manifest evidence differs")
    if retained is not None and retained_manifest_raw is not None:
        if (
            retained["sha256"]
            != hashlib.sha256(retained_manifest_raw).hexdigest()
            or retained["bytes"] != len(retained_manifest_raw)
        ):
            raise Wan22ReleaseError("exact512 retained manifest binding differs")

    selection = _mapping(
        value.get("selection"), context=f"{context}.selection"
    )
    _exact_keys(
        selection,
        {
            "row_count",
            "retained_row_count",
            "ranked_fill_row_count",
            "ordered_iids",
            "ordered_group_ids",
            "ordered_fill_iids",
            "ordered_fill_group_ids",
            "ordered_fill_review_ranks",
            "output_file",
            "output_sha256",
            "output_bytes",
        },
        context=f"{context}.selection",
    )
    fill_count = SCALE512_RELEASE_ROW_COUNT - retained_count
    expected_policy = (
        EXACT512_RETAIN_EXACT8_POLICY
        if retained is not None
        else EXACT512_RANK_ONLY_POLICY
    )
    if value.get("policy") != expected_policy:
        raise Wan22ReleaseError("exact512 selection receipt policy differs")
    if (
        selection.get("row_count") != SCALE512_RELEASE_ROW_COUNT
        or selection.get("retained_row_count") != retained_count
        or selection.get("ranked_fill_row_count") != fill_count
        or selection.get("output_file") != EXACT8_MANIFEST_NAME
        or selection.get("output_sha256")
        != hashlib.sha256(selected_manifest_raw).hexdigest()
        or selection.get("output_bytes") != len(selected_manifest_raw)
    ):
        raise Wan22ReleaseError("exact512 receipt manifest binding differs")

    ordered_iids = selection.get("ordered_iids")
    ordered_groups = selection.get("ordered_group_ids")
    fill_iids = selection.get("ordered_fill_iids")
    fill_groups = selection.get("ordered_fill_group_ids")
    fill_ranks = selection.get("ordered_fill_review_ranks")
    retention_iids = retention.get("ordered_iids")
    retention_groups = retention.get("ordered_group_ids")
    if (
        not isinstance(ordered_iids, list)
        or len(ordered_iids) != SCALE512_RELEASE_ROW_COUNT
        or any(
            not isinstance(iid, str) or _IID_RE.fullmatch(iid) is None
            for iid in ordered_iids
        )
        or len(set(ordered_iids)) != SCALE512_RELEASE_ROW_COUNT
    ):
        raise Wan22ReleaseError("exact512 receipt ordered IIDs differ")
    if (
        not isinstance(ordered_groups, list)
        or len(ordered_groups) != SCALE512_RELEASE_ROW_COUNT
        or any(
            not isinstance(group, str) or not group or group != group.strip()
            for group in ordered_groups
        )
        or len(set(ordered_groups)) != SCALE512_RELEASE_ROW_COUNT
    ):
        raise Wan22ReleaseError("exact512 receipt ordered groups differ")
    if (
        not isinstance(fill_iids, list)
        or fill_iids != ordered_iids[retained_count:]
        or not isinstance(fill_groups, list)
        or fill_groups != ordered_groups[retained_count:]
        or len(fill_iids) != fill_count
        or len(fill_groups) != fill_count
    ):
        raise Wan22ReleaseError("exact512 receipt ranked fill ordering differs")
    if (
        not isinstance(retention_iids, list)
        or retention_iids != ordered_iids[:retained_count]
        or not isinstance(retention_groups, list)
        or retention_groups != ordered_groups[:retained_count]
    ):
        raise Wan22ReleaseError("exact512 receipt retention ordering differs")
    if (
        not isinstance(fill_ranks, list)
        or len(fill_ranks) != fill_count
        or any(type(rank) is not int or rank <= 0 for rank in fill_ranks)
        or len(set(fill_ranks)) != fill_count
        or fill_ranks != sorted(fill_ranks)
    ):
        raise Wan22ReleaseError("exact512 receipt ranked fill ranks differ")
    return retained


def _rerun_scale512_selection(
    *,
    manifest_path: Path,
    finalizer_dir: Path,
    selection_receipt_path: Path,
) -> dict[str, Any]:
    finalizer = _resolved_finalizer_directory(finalizer_dir)
    manifest_resolved, manifest_raw = _stable_read(
        manifest_path, context="requested exact512 manifest"
    )
    receipt, receipt_raw, receipt_resolved = _load_json(
        selection_receipt_path,
        context="requested exact512 selection receipt",
    )
    retained = _receipt_retained_exact8_source(receipt)
    retained_raw: bytes | None = None
    retained_resolved: Path | None = None
    if retained is not None:
        retained_resolved, retained_raw = _stable_read(
            Path(retained["path"]), context="retained exact8 manifest"
        )
        if (
            str(retained_resolved) != retained["path"]
            or hashlib.sha256(retained_raw).hexdigest() != retained["sha256"]
            or len(retained_raw) != retained["bytes"]
        ):
            raise Wan22ReleaseError("retained exact8 manifest receipt differs")
    done_resolved, done_raw = _stable_read(
        finalizer / FINALIZER_DONE_NAME, context="scale512 finalizer done"
    )
    summary_resolved, summary_raw = _stable_read(
        finalizer / FINALIZER_SUMMARY_NAME,
        context="scale512 finalizer summary",
    )
    selector_path = Path(__file__).resolve().with_name(EXACT512_SELECTOR_NAME)
    selector_resolved, selector_raw = _stable_read(
        selector_path, context="exact512 selector implementation"
    )
    shared_resolved, shared_raw = _stable_read(
        selector_resolved.with_name(EXACT8_SELECTOR_NAME),
        context="shared strict selector implementation",
    )
    finalizer_source_resolved, finalizer_source_raw = _stable_read(
        selector_resolved.with_name(FINALIZER_IMPLEMENTATION_NAME),
        context="scale512 finalizer implementation",
    )
    with tempfile.TemporaryDirectory(
        prefix="motive-wan22-exact512-rerun-"
    ) as temporary:
        rerun_root = Path(temporary).resolve(strict=True) / "exact512"
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        package_root = str(selector_resolved.parent.parent)
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root
            if not existing_pythonpath
            else package_root + os.pathsep + existing_pythonpath
        )
        command = [
            sys.executable,
            "-m",
            "motive.wan22_select_exact512",
            "--finalizer-dir",
            str(finalizer),
            "--output-dir",
            str(rerun_root),
        ]
        if retained_resolved is not None:
            command.extend(
                ["--retain-exact8-manifest", str(retained_resolved)]
            )
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            diagnostic = completed.stderr.decode(
                "utf-8", errors="replace"
            ).strip()
            raise Wan22ReleaseError(
                "exact512 selector rerun failed: " + diagnostic
            )
        rerun_manifest_resolved, rerun_manifest_raw = _stable_read(
            rerun_root / EXACT8_MANIFEST_NAME,
            context="rerun exact512 manifest",
        )
        rerun_receipt, rerun_receipt_raw, _ = _load_json(
            rerun_root / EXACT8_SELECTION_RECEIPT_NAME,
            context="rerun exact512 selection receipt",
        )
        if rerun_manifest_resolved.parent != rerun_root.resolve(strict=True):
            raise Wan22ReleaseError("rerun exact512 output escaped its root")
    if rerun_manifest_raw != manifest_raw:
        raise Wan22ReleaseError(
            "rerun exact512 manifest is not byte-identical to the request"
        )
    if rerun_receipt_raw != receipt_raw or rerun_receipt != receipt:
        raise Wan22ReleaseError(
            "rerun exact512 selection receipt is not byte-identical to the request"
        )
    retained = _validate_exact512_receipt(
        receipt,
        selected_manifest_raw=manifest_raw,
        done_raw=done_raw,
        summary_raw=summary_raw,
        selector_raw=selector_raw,
        shared_selector_raw=shared_raw,
        finalizer_implementation_raw=finalizer_source_raw,
        retained_manifest_raw=retained_raw,
    )
    binding = {
        "schema_version": SCALE512_SELECTION_EVIDENCE_SCHEMA,
        "mode": SCALE512_SELECTION_EVIDENCE_MODE,
        "finalizer_dir": str(finalizer),
        "done": _binding_from_bytes(done_resolved, done_raw),
        "summary": _binding_from_bytes(summary_resolved, summary_raw),
        "selection_receipt": _binding_from_bytes(
            receipt_resolved, receipt_raw
        ),
        "selector_implementation": _binding_from_bytes(
            selector_resolved, selector_raw
        ),
        "shared_strict_io_implementation": _binding_from_bytes(
            shared_resolved, shared_raw
        ),
        "finalizer_implementation": _binding_from_bytes(
            finalizer_source_resolved, finalizer_source_raw
        ),
        "selected_manifest": {
            **_binding_from_bytes(manifest_resolved, manifest_raw),
            "rows": SCALE512_RELEASE_ROW_COUNT,
        },
        "retained_exact8_manifest": (
            None
            if retained is None or retained_resolved is None or retained_raw is None
            else {
                **_binding_from_bytes(retained_resolved, retained_raw),
                "rows": RELEASE_ROW_COUNT,
            }
        ),
    }
    return _validate_signed_scale512_selection_binding(binding)


def _verify_scale512_selection_evidence(
    value: Any,
    *,
    manifest_scope: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _validate_signed_scale512_selection_binding(value)
    rebuilt = _rerun_scale512_selection(
        manifest_path=Path(binding["selected_manifest"]["path"]),
        finalizer_dir=Path(binding["finalizer_dir"]),
        selection_receipt_path=Path(binding["selection_receipt"]["path"]),
    )
    if rebuilt != binding:
        raise Wan22ReleaseError(
            "current scale512 selection closure differs from signed evidence"
        )
    selected = binding["selected_manifest"]
    retained_rows = (
        RELEASE_ROW_COUNT
        if binding["retained_exact8_manifest"] is not None
        else 0
    )
    if (
        selected["sha256"] != manifest_scope.get("sha256")
        or selected["bytes"] != manifest_scope.get("bytes")
        or selected["rows"] != manifest_scope.get("rows")
        or retained_rows != manifest_scope.get("retained_exact8_rows")
    ):
        raise Wan22ReleaseError(
            "scale512 selection evidence differs from signed manifest scope"
        )
    return binding


def _resolve_evidence_inputs(
    *,
    manifest_path: Path,
    smoke_acceptance_path: Path | None,
    finalizer_dir: Path | None,
    selection_receipt_path: Path | None,
) -> tuple[str, dict[str, Any]]:
    smoke_mode = smoke_acceptance_path is not None
    finalizer_mode = (
        finalizer_dir is not None or selection_receipt_path is not None
    )
    if smoke_mode == finalizer_mode:
        raise Wan22ReleaseError(
            "supply exactly one evidence mode: --smoke-acceptance or "
            "--finalizer-dir with --selection-receipt"
        )
    if smoke_mode:
        assert smoke_acceptance_path is not None
        _rerun_acceptance(smoke_acceptance_path)
        return (
            "smoke_acceptance",
            {"smoke_acceptance": _acceptance_binding(smoke_acceptance_path)},
        )
    if finalizer_dir is None or selection_receipt_path is None:
        raise Wan22ReleaseError(
            "finalizer selection evidence requires both finalizer directory "
            "and selection receipt"
        )
    selection = _rerun_finalizer_selection(
        manifest_path=manifest_path,
        finalizer_dir=finalizer_dir,
        selection_receipt_path=selection_receipt_path,
    )
    return (
        "finalizer_selection",
        {"finalizer_selection": selection},
    )


def _resolve_scale512_evidence_inputs(
    *,
    manifest_path: Path,
    smoke_acceptance_path: Path | None,
    finalizer_dir: Path | None,
    selection_receipt_path: Path | None,
) -> dict[str, Any]:
    if smoke_acceptance_path is not None:
        raise Wan22ReleaseError(
            "exact512-v2 does not accept exact16 smoke evidence"
        )
    if finalizer_dir is None or selection_receipt_path is None:
        raise Wan22ReleaseError(
            "exact512-v2 requires finalizer directory and selection receipt"
        )
    return _rerun_scale512_selection(
        manifest_path=manifest_path,
        finalizer_dir=finalizer_dir,
        selection_receipt_path=selection_receipt_path,
    )


def _prepare_scale512_release_request(
    *,
    manifest_path: Path,
    request_path: Path,
    release_id: str,
    issued_at_utc: str,
    challenge_sha: str,
    smoke_acceptance_path: Path | None,
    finalizer_dir: Path | None,
    selection_receipt_path: Path | None,
) -> dict[str, Any]:
    evidence = _resolve_scale512_evidence_inputs(
        manifest_path=manifest_path,
        smoke_acceptance_path=smoke_acceptance_path,
        finalizer_dir=finalizer_dir,
        selection_receipt_path=selection_receipt_path,
    )
    signed = _payload_scale512(
        manifest_path=manifest_path,
        release_id=release_id,
        issued_at_utc=issued_at_utc,
        scale512_selection=evidence,
    )
    builder = {
        "release_module_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "selector_implementation_sha256": evidence[
            "selector_implementation"
        ]["sha256"],
        "shared_strict_io_implementation_sha256": evidence[
            "shared_strict_io_implementation"
        ]["sha256"],
        "finalizer_implementation_sha256": evidence[
            "finalizer_implementation"
        ]["sha256"],
    }
    request: dict[str, Any] = {
        "schema_version": RELEASE_REQUEST_SCHEMA_V2,
        "challenge_sha256": challenge_sha,
        "builder": builder,
        "signed": signed,
    }
    request["request_digest"] = _object_digest(request)
    _atomic_new_json(request_path, request)
    return request


def prepare_release_request(
    *,
    manifest_path: str | Path,
    request_path: str | Path,
    release_id: str,
    issued_at_utc: str,
    challenge: str,
    smoke_acceptance_path: str | Path | None = None,
    finalizer_dir: str | Path | None = None,
    selection_receipt_path: str | Path | None = None,
    profile: str = RELEASE_PROFILE_EXACT8_V1,
) -> dict[str, Any]:
    """Rerun one evidence path and publish a challenge-bound sign request."""

    challenge_sha = _sha(challenge, context="release request challenge")
    manifest = Path(manifest_path)
    smoke_path = (
        Path(smoke_acceptance_path)
        if smoke_acceptance_path is not None
        else None
    )
    finalizer_path = Path(finalizer_dir) if finalizer_dir is not None else None
    receipt_path = (
        Path(selection_receipt_path)
        if selection_receipt_path is not None
        else None
    )
    if profile == RELEASE_PROFILE_EXACT512_V2:
        return _prepare_scale512_release_request(
            manifest_path=manifest,
            request_path=Path(request_path),
            release_id=release_id,
            issued_at_utc=issued_at_utc,
            challenge_sha=challenge_sha,
            smoke_acceptance_path=smoke_path,
            finalizer_dir=finalizer_path,
            selection_receipt_path=receipt_path,
        )
    if profile != RELEASE_PROFILE_EXACT8_V1:
        raise Wan22ReleaseError(f"unsupported release profile: {profile!r}")
    evidence_mode, evidence = _resolve_evidence_inputs(
        manifest_path=manifest,
        smoke_acceptance_path=smoke_path,
        finalizer_dir=finalizer_path,
        selection_receipt_path=receipt_path,
    )
    signed = _payload(
        manifest_path=manifest,
        release_id=release_id,
        issued_at_utc=issued_at_utc,
        smoke_acceptance_path=smoke_path,
        finalizer_selection=evidence.get("finalizer_selection"),
    )
    release_module_sha = hashlib.sha256(
        Path(__file__).resolve().read_bytes()
    ).hexdigest()
    if evidence_mode == "smoke_acceptance":
        verifier_path = Path(__file__).resolve().with_name(
            "goku_action_v13_acceptance.py"
        )
        builder = {
            "release_module_sha256": release_module_sha,
            "acceptance_verifier_sha256": hashlib.sha256(
                verifier_path.read_bytes()
            ).hexdigest(),
        }
    else:
        selection = _validate_signed_finalizer_selection_binding(
            evidence["finalizer_selection"]
        )
        builder = {
            "release_module_sha256": release_module_sha,
            "selector_implementation_sha256": selection[
                "selector_implementation"
            ]["sha256"],
            "finalizer_implementation_sha256": selection[
                "finalizer_implementation"
            ]["sha256"],
        }
    request: dict[str, Any] = {
        "schema_version": RELEASE_REQUEST_SCHEMA,
        "challenge_sha256": challenge_sha,
        "builder": builder,
        "signed": signed,
    }
    request["request_digest"] = _object_digest(request)
    _atomic_new_json(Path(request_path), request)
    return request


def _sign_payload(
    *,
    signed: Mapping[str, Any],
    output_path: str | Path,
    signing_key: str | Path,
    envelope_schema: str = RELEASE_SCHEMA,
    signature_namespace: str = SIGNATURE_NAMESPACE,
) -> dict[str, Any]:
    _require_dedicated_signer_anchor()
    key = Path(signing_key).expanduser()
    public = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(key)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if public.returncode != 0:
        raise Wan22ReleaseError("cannot read the release signing key")
    derived_public = " ".join(
        public.stdout.decode("utf-8").strip().split()[:2]
    )
    if derived_public != SIGNER_PUBLIC_KEY:
        raise Wan22ReleaseError("signing key does not match source-anchored public key")
    with tempfile.TemporaryDirectory(prefix="motive-wan22-sign-") as temporary:
        message = Path(temporary) / "release-payload.json"
        message.write_bytes(_canonical_bytes(signed))
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                signature_namespace,
                str(message),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        signature_path = Path(str(message) + ".sig")
        if result.returncode != 0 or not signature_path.is_file():
            raise Wan22ReleaseError("release signing failed")
        armor = signature_path.read_bytes()
    envelope = {
        "schema_version": envelope_schema,
        "signed": signed,
        "signature": {
            "format": "SSHSIG",
            "namespace": signature_namespace,
            "principal": SIGNER_PRINCIPAL,
            "key_fingerprint": SIGNER_KEY_FINGERPRINT,
            "armored_signature_base64": base64.b64encode(armor).decode("ascii"),
        },
    }
    _atomic_new_json(Path(output_path), envelope)
    return envelope


def sign_prepared_request(
    *,
    request_path: str | Path,
    output_path: str | Path,
    signing_key: str | Path,
    expected_challenge: str,
) -> dict[str, Any]:
    """Sign only a fresh request produced by the byte-identical local source."""

    request, _, _ = _load_json(
        Path(request_path),
        context="prepared release request",
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
        context="prepared release request",
    )
    request_schema = request.get("schema_version")
    if request_schema not in {RELEASE_REQUEST_SCHEMA, RELEASE_REQUEST_SCHEMA_V2}:
        raise Wan22ReleaseError("release request schema differs")
    claimed_digest = _sha(
        request.get("request_digest"),
        context="release request digest",
    )
    unsigned_request = dict(request)
    del unsigned_request["request_digest"]
    if claimed_digest != _object_digest(unsigned_request):
        raise Wan22ReleaseError("release request digest differs")
    challenge = _sha(
        expected_challenge,
        context="expected release challenge",
    )
    if request.get("challenge_sha256") != challenge:
        raise Wan22ReleaseError("release request challenge differs")
    signed = _mapping(request.get("signed"), context="release request payload")
    _validate_release_payload_shape(signed)
    evidence_mode, evidence = _release_evidence(signed)
    builder = _mapping(request.get("builder"), context="release request builder")
    local_release_sha = hashlib.sha256(
        Path(__file__).resolve().read_bytes()
    ).hexdigest()
    if builder.get("release_module_sha256") != local_release_sha:
        raise Wan22ReleaseError("remote/local release module SHA differs")
    if request_schema == RELEASE_REQUEST_SCHEMA_V2:
        if (
            signed.get("schema_version") != RELEASE_PAYLOAD_SCHEMA_V2
            or evidence_mode != "scale512_selection"
        ):
            raise Wan22ReleaseError("scale512 request/payload profile differs")
        _exact_keys(
            builder,
            {
                "release_module_sha256",
                "selector_implementation_sha256",
                "shared_strict_io_implementation_sha256",
                "finalizer_implementation_sha256",
            },
            context="scale512 release request builder",
        )
        selector_path = Path(__file__).resolve().with_name(EXACT512_SELECTOR_NAME)
        shared_path = selector_path.with_name(EXACT8_SELECTOR_NAME)
        finalizer_path = selector_path.with_name(FINALIZER_IMPLEMENTATION_NAME)
        local_selector_sha = hashlib.sha256(selector_path.read_bytes()).hexdigest()
        local_shared_sha = hashlib.sha256(shared_path.read_bytes()).hexdigest()
        local_finalizer_sha = hashlib.sha256(finalizer_path.read_bytes()).hexdigest()
        if builder.get("selector_implementation_sha256") != local_selector_sha:
            raise Wan22ReleaseError("remote/local exact512 selector SHA differs")
        if (
            builder.get("shared_strict_io_implementation_sha256")
            != local_shared_sha
        ):
            raise Wan22ReleaseError("remote/local shared strict selector SHA differs")
        if builder.get("finalizer_implementation_sha256") != local_finalizer_sha:
            raise Wan22ReleaseError("remote/local finalizer implementation SHA differs")
        if (
            evidence["selector_implementation"]["sha256"]
            != local_selector_sha
            or evidence["shared_strict_io_implementation"]["sha256"]
            != local_shared_sha
            or evidence["finalizer_implementation"]["sha256"]
            != local_finalizer_sha
        ):
            raise Wan22ReleaseError(
                "scale512 request selection source binding differs"
            )
        return _sign_payload(
            signed=signed,
            output_path=output_path,
            signing_key=signing_key,
            envelope_schema=RELEASE_SCHEMA_V2,
            signature_namespace=SIGNATURE_NAMESPACE_V2,
        )
    if signed.get("schema_version") != RELEASE_PAYLOAD_SCHEMA:
        raise Wan22ReleaseError("exact8 request/payload profile differs")
    if evidence_mode == "smoke_acceptance":
        _exact_keys(
            builder,
            {"release_module_sha256", "acceptance_verifier_sha256"},
            context="release request builder",
        )
        local_verifier_sha = hashlib.sha256(
            Path(__file__).resolve()
            .with_name("goku_action_v13_acceptance.py")
            .read_bytes()
        ).hexdigest()
        if builder.get("acceptance_verifier_sha256") != local_verifier_sha:
            raise Wan22ReleaseError(
                "remote/local acceptance verifier SHA differs"
            )
        if evidence["verifier_implementation_sha256"] != local_verifier_sha:
            raise Wan22ReleaseError(
                "release request acceptance verifier binding differs"
            )
    else:
        _exact_keys(
            builder,
            {
                "release_module_sha256",
                "selector_implementation_sha256",
                "finalizer_implementation_sha256",
            },
            context="release request builder",
        )
        selector_path = Path(__file__).resolve().with_name(EXACT8_SELECTOR_NAME)
        finalizer_path = selector_path.with_name(FINALIZER_IMPLEMENTATION_NAME)
        local_selector_sha = hashlib.sha256(selector_path.read_bytes()).hexdigest()
        local_finalizer_sha = hashlib.sha256(finalizer_path.read_bytes()).hexdigest()
        if builder.get("selector_implementation_sha256") != local_selector_sha:
            raise Wan22ReleaseError("remote/local exact-eight selector SHA differs")
        if builder.get("finalizer_implementation_sha256") != local_finalizer_sha:
            raise Wan22ReleaseError("remote/local finalizer implementation SHA differs")
        if (
            evidence["selector_implementation"]["sha256"]
            != local_selector_sha
            or evidence["finalizer_implementation"]["sha256"]
            != local_finalizer_sha
        ):
            raise Wan22ReleaseError(
                "release request finalizer selection source binding differs"
            )
    return _sign_payload(
        signed=signed,
        output_path=output_path,
        signing_key=signing_key,
    )


def build_and_sign_release(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    signing_key: str | Path,
    release_id: str,
    issued_at_utc: str,
    smoke_acceptance_path: str | Path | None = None,
    finalizer_dir: str | Path | None = None,
    selection_receipt_path: str | Path | None = None,
    profile: str = RELEASE_PROFILE_EXACT8_V1,
) -> dict[str, Any]:
    """Single-host convenience path; rerun one evidence path before signing."""

    manifest = Path(manifest_path)
    smoke_path = (
        Path(smoke_acceptance_path)
        if smoke_acceptance_path is not None
        else None
    )
    finalizer_path = Path(finalizer_dir) if finalizer_dir is not None else None
    receipt_path = (
        Path(selection_receipt_path)
        if selection_receipt_path is not None
        else None
    )
    if profile == RELEASE_PROFILE_EXACT512_V2:
        evidence = _resolve_scale512_evidence_inputs(
            manifest_path=manifest,
            smoke_acceptance_path=smoke_path,
            finalizer_dir=finalizer_path,
            selection_receipt_path=receipt_path,
        )
        signed = _payload_scale512(
            manifest_path=manifest,
            release_id=release_id,
            issued_at_utc=issued_at_utc,
            scale512_selection=evidence,
        )
        return _sign_payload(
            signed=signed,
            output_path=output_path,
            signing_key=signing_key,
            envelope_schema=RELEASE_SCHEMA_V2,
            signature_namespace=SIGNATURE_NAMESPACE_V2,
        )
    if profile != RELEASE_PROFILE_EXACT8_V1:
        raise Wan22ReleaseError(f"unsupported release profile: {profile!r}")
    _evidence_mode, evidence = _resolve_evidence_inputs(
        manifest_path=manifest,
        smoke_acceptance_path=smoke_path,
        finalizer_dir=finalizer_path,
        selection_receipt_path=receipt_path,
    )
    signed = _payload(
        manifest_path=manifest,
        release_id=release_id,
        issued_at_utc=issued_at_utc,
        smoke_acceptance_path=smoke_path,
        finalizer_selection=evidence.get("finalizer_selection"),
    )
    return _sign_payload(
        signed=signed,
        output_path=output_path,
        signing_key=signing_key,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="rerun one evidence mode and prepare a challenge-bound request",
    )
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument(
        "--profile",
        choices=(RELEASE_PROFILE_EXACT8_V1, RELEASE_PROFILE_EXACT512_V2),
        default=RELEASE_PROFILE_EXACT8_V1,
    )
    evidence = prepare.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--smoke-acceptance", type=Path)
    evidence.add_argument("--finalizer-dir", type=Path)
    prepare.add_argument(
        "--selection-receipt",
        type=Path,
        help="required with --finalizer-dir and forbidden otherwise",
    )
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--release-id", required=True)
    prepare.add_argument("--challenge", required=True)
    prepare.add_argument(
        "--issued-at-utc",
        default=None,
        help="ISO-8601 UTC timestamp; defaults to the current UTC time",
    )
    build = subparsers.add_parser(
        "build-and-sign",
        help="rerun one evidence mode and sign on the same trusted host",
    )
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument(
        "--profile",
        choices=(RELEASE_PROFILE_EXACT8_V1, RELEASE_PROFILE_EXACT512_V2),
        default=RELEASE_PROFILE_EXACT8_V1,
    )
    build_evidence = build.add_mutually_exclusive_group(required=True)
    build_evidence.add_argument("--smoke-acceptance", type=Path)
    build_evidence.add_argument("--finalizer-dir", type=Path)
    build.add_argument("--selection-receipt", type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--release-id", required=True)
    build.add_argument("--signing-key", required=True, type=Path)
    build.add_argument(
        "--issued-at-utc",
        default=None,
        help="ISO-8601 UTC timestamp; defaults to the current UTC time",
    )
    sign = subparsers.add_parser(
        "sign-request",
        help="sign a prepared request locally with the source-anchored key",
    )
    sign.add_argument("--request", required=True, type=Path)
    sign.add_argument("--output", required=True, type=Path)
    sign.add_argument("--signing-key", required=True, type=Path)
    sign.add_argument("--challenge", required=True)
    verify = subparsers.add_parser(
        "verify",
        help="verify an exact8 v1 or exact512 v2 signed release",
    )
    verify.add_argument("--release", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            issued_at = args.issued_at_utc or datetime.now(
                timezone.utc
            ).isoformat()
            result = prepare_release_request(
                manifest_path=args.manifest,
                smoke_acceptance_path=args.smoke_acceptance,
                finalizer_dir=args.finalizer_dir,
                selection_receipt_path=args.selection_receipt,
                request_path=args.output,
                release_id=args.release_id,
                issued_at_utc=issued_at,
                challenge=args.challenge,
                profile=args.profile,
            )
            summary = {
                "status": "prepared",
                "release_id": result["signed"]["release_id"],
                "manifest_sha256": result["signed"]["manifest"]["sha256"],
                "rows": result["signed"]["manifest"]["rows"],
                "output": str(args.output.resolve()),
            }
        elif args.command == "build-and-sign":
            issued_at = args.issued_at_utc or datetime.now(
                timezone.utc
            ).isoformat()
            result = build_and_sign_release(
                manifest_path=args.manifest,
                smoke_acceptance_path=args.smoke_acceptance,
                finalizer_dir=args.finalizer_dir,
                selection_receipt_path=args.selection_receipt,
                output_path=args.output,
                signing_key=args.signing_key,
                release_id=args.release_id,
                issued_at_utc=issued_at,
                profile=args.profile,
            )
            summary = {
                "status": "signed",
                "release_id": result["signed"]["release_id"],
                "manifest_sha256": result["signed"]["manifest"]["sha256"],
                "rows": result["signed"]["manifest"]["rows"],
                "output": str(args.output.resolve()),
            }
        elif args.command == "sign-request":
            result = sign_prepared_request(
                request_path=args.request,
                output_path=args.output,
                signing_key=args.signing_key,
                expected_challenge=args.challenge,
            )
            summary = {
                "status": "signed",
                "release_id": result["signed"]["release_id"],
                "manifest_sha256": result["signed"]["manifest"]["sha256"],
                "rows": result["signed"]["manifest"]["rows"],
                "output": str(args.output.resolve()),
            }
        else:
            verified = verify_signed_release(
                release_path=args.release,
                manifest_path=args.manifest,
                require_exact_manifest=True,
                verify_media=True,
            )
            summary = {
                "status": "verified",
                "release_id": verified["release"]["release_id"],
                "manifest_sha256": verified["manifest_sha256"],
                "rows": verified["manifest_row_count"],
            }
    except (Wan22ReleaseError, FileNotFoundError, OSError) as error:
        print(f"[wan22-release] fatal {type(error).__name__}: {error}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
