"""Fail-closed manifest atomization for local action-editing research.

The source Goku/Wan data describes a *global* counterfactual: every dynamic
subject receives a new motion.  This module converts one accepted global row
into one row per non-interacting subject component.  It never claims that a
global target is a local RGB ground truth.  Source/target tubes in the emitted
rows are privileged supervision consumed by the latent PACT objective.
"""

from __future__ import annotations

import base64
import copy
from datetime import datetime
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ATOMIC_SCHEMA = "pact-atomic-action-row-v1"
TRACK_SCHEMA = "pact-actor-component-tracks-v1"
POSTGEN_RELEASE_SCHEMA = "pact-post-generation-signed-release-v1"
POSTGEN_RELEASE_PAYLOAD_SCHEMA = "pact-post-generation-release-payload-v1"
POSTGEN_RELEASE_PURPOSE = "pact-production-global-target-atomization"
POSTGEN_SIGNATURE_NAMESPACE = "pact-post-generation-global-manifest-v1"
POSTGEN_SIGNER_PRINCIPAL = "pact-post-generation-release"


class ManifestError(ValueError):
    """Raised when an input cannot safely become atomic supervision."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"value is not finite canonical JSON: {exc}") from exc


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ManifestError(
            f"{field} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _reject_constant(value: str) -> None:
    raise ManifestError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _parse_strict_json(raw: bytes, *, field: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{field} is not UTF-8") from exc
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{field} is not strict JSON: {exc}") from exc


def _stable_regular_file(path: os.PathLike[str] | str, *, field: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ManifestError(f"{field} must be a non-symlink regular file: {candidate}")
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"cannot resolve {field}: {candidate}") from exc


def _verified_file_sha256(
    path: os.PathLike[str] | str, declared: str, *, field: str
) -> str:
    resolved = _stable_regular_file(path, field=field)
    before = resolved.stat()
    actual = file_sha256(resolved)
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ManifestError(f"{field} changed while it was being hashed")
    if actual != declared:
        raise ManifestError(f"{field} digest differs")
    return actual


def _ordered_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        b"".join(value.encode("utf-8") + b"\n" for value in values)
    ).hexdigest()


def _strict_canonical_jsonl(
    path: os.PathLike[str] | str, *, field: str
) -> tuple[list[dict[str, Any]], bytes, Path]:
    resolved = _stable_regular_file(path, field=field)
    raw = resolved.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ManifestError(f"{field} must be non-empty and newline-terminated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ManifestError(f"{field}:{line_number} is blank")
        value = _parse_strict_json(line, field=f"{field}:{line_number}")
        if not isinstance(value, dict):
            raise ManifestError(f"{field}:{line_number} is not an object")
        if canonical_json_bytes(value) != line:
            raise ManifestError(f"{field}:{line_number} is not canonical JSON")
        rows.append(value)
    return rows, raw, resolved


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_id(value: Any, *, field: str) -> str:
    text = _nonempty_text(value, field=field)
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in text):
        raise ManifestError(f"{field} contains an unsafe character: {text!r}")
    return text


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


def get_nested(row: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ManifestError(f"missing manifest field: {dotted_key}")
        value = value[part]
    return value


def _is_parent_training_eligible(row: Mapping[str, Any]) -> bool:
    production_eligible = row.get("production_eligible")
    if type(production_eligible) is not bool:
        raise ManifestError("parent production_eligible must be bool")
    status = row.get("human_review_status")
    if not isinstance(status, str) or not status.strip():
        raise ManifestError("parent human_review_status must be non-empty")
    status_approved = status.casefold() in {"accepted", "approved", "passed"}
    human_approved = row.get("human_approved")
    if human_approved is not None:
        if type(human_approved) is not bool:
            raise ManifestError("parent human_approved must be bool when present")
        if human_approved != status_approved:
            raise ManifestError("parent human approval fields conflict")
    forbidden = False
    for key in ("production_use_forbidden", "training_use_forbidden"):
        value = row.get(key)
        if value is not None and type(value) is not bool:
            raise ManifestError(f"parent {key} must be bool when present")
        forbidden = forbidden or value is True
    if production_eligible and not status_approved:
        raise ManifestError("production-eligible parent lacks approved human review")
    return production_eligible and status_approved and not forbidden


def _validate_subject_tables(
    source_census: Mapping[str, Any], target_plan: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    raw_sources = source_census.get("dynamic_subjects")
    raw_targets = target_plan.get("dynamic_subject_targets")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ManifestError("source_census.dynamic_subjects must be non-empty")
    if not isinstance(raw_targets, list) or len(raw_targets) != len(raw_sources):
        raise ManifestError("target plan must cover every source subject exactly once")

    sources: dict[str, Mapping[str, Any]] = {}
    targets: dict[str, Mapping[str, Any]] = {}
    for item in raw_sources:
        if not isinstance(item, Mapping):
            raise ManifestError("source subject must be an object")
        subject_id = _safe_id(item.get("subject_id"), field="source subject_id")
        if subject_id in sources:
            raise ManifestError(f"duplicate source subject_id: {subject_id}")
        _nonempty_text(item.get("stable_reference"), field=f"{subject_id}.stable_reference")
        _nonempty_text(item.get("source_motion"), field=f"{subject_id}.source_motion")
        sources[subject_id] = item
    for item in raw_targets:
        if not isinstance(item, Mapping):
            raise ManifestError("target subject must be an object")
        subject_id = _safe_id(item.get("subject_id"), field="target subject_id")
        if subject_id in targets:
            raise ManifestError(f"duplicate target subject_id: {subject_id}")
        _nonempty_text(item.get("target_motion"), field=f"{subject_id}.target_motion")
        if item.get("substantive_change") is not True:
            raise ManifestError(f"{subject_id} target is not a substantive change")
        targets[subject_id] = item
    if list(sources) != list(targets):
        raise ManifestError("source and target subject IDs/order differ")
    return sources, targets


def _require_static_camera(
    source_census: Mapping[str, Any], target_plan: Mapping[str, Any]
) -> None:
    source_camera = source_census.get("camera")
    target_camera = target_plan.get("camera_target")
    if not isinstance(source_camera, Mapping) or not isinstance(target_camera, Mapping):
        raise ManifestError("source and target camera records are required")
    if source_camera.get("motion_class") != "locked_off":
        raise ManifestError("local atomization requires a locked-off source camera")
    if (
        target_camera.get("relation") != "preserve_static"
        or target_camera.get("motion_class") != "locked_off"
    ):
        raise ManifestError("local atomization requires a preserved locked-off target camera")


def validate_track_record(
    value: Mapping[str, Any], *, expected_iid: str | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError("track record must be an object")
    row = copy.deepcopy(dict(value))
    if row.get("schema_version") != TRACK_SCHEMA:
        raise ManifestError(f"track schema must be {TRACK_SCHEMA}")
    iid = _safe_id(row.get("iid"), field="track.iid")
    if expected_iid is not None and iid != expected_iid:
        raise ManifestError(f"track IID {iid!r} does not match parent {expected_iid!r}")
    row["component_id"] = _safe_id(row.get("component_id"), field="component_id")
    subject_ids = row.get("subject_ids")
    if not isinstance(subject_ids, list) or not subject_ids:
        raise ManifestError("track.subject_ids must be a non-empty list")
    row["subject_ids"] = [
        _safe_id(item, field="track.subject_ids[]") for item in subject_ids
    ]
    if len(set(row["subject_ids"])) != len(row["subject_ids"]):
        raise ManifestError("track.subject_ids contains duplicates")
    row["source_mask_path"] = _nonempty_text(
        row.get("source_mask_path"), field="source_mask_path"
    )
    row["target_mask_path"] = _nonempty_text(
        row.get("target_mask_path"), field="target_mask_path"
    )
    if row.get("interaction_safe") is not True:
        raise ManifestError("track component was not accepted as interaction-safe")
    if str(row.get("review_status", "")).casefold() not in {
        "accepted",
        "approved",
        "passed",
    }:
        raise ManifestError("track component lacks accepted review_status")
    confidence = row.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise ManifestError("track.confidence must be in [0, 1]")
    for key in ("source_mask_sha256", "target_mask_sha256"):
        _sha256(row.get(key), field=f"track.{key}")
    return row


def _compile_atomic_instruction(
    selected_ids: Sequence[str],
    sources: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
) -> str:
    changes = []
    for subject_id in selected_ids:
        reference = _nonempty_text(
            sources[subject_id]["stable_reference"], field=f"{subject_id}.stable_reference"
        ).rstrip(".!?;:")
        target_motion = _nonempty_text(
            targets[subject_id]["target_motion"], field=f"{subject_id}.target_motion"
        ).rstrip(".!?;:")
        changes.append(
            f"Replace the action of {reference} with this complete motion: {target_motion}"
        )
    return (
        "; ".join(changes)
        + "; preserve the actions, timing, and trajectories of every other subject; "
        "preserve every subject's identity and appearance and all scene content except "
        "changes physically required by the requested action; keep the camera locked off."
    )


def _compile_target_caption_contract(
    selected_ids: Sequence[str],
    sources: Mapping[str, Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
) -> str:
    selected = set(selected_ids)
    clauses = []
    for subject_id, source in sources.items():
        reference = _nonempty_text(
            source["stable_reference"], field=f"{subject_id}.stable_reference"
        ).rstrip(".!?;:")
        if subject_id in selected:
            motion = _nonempty_text(
                targets[subject_id]["target_motion"], field=f"{subject_id}.target_motion"
            ).rstrip(".!?;:")
        else:
            motion = _nonempty_text(
                source["source_motion"], field=f"{subject_id}.source_motion"
            ).rstrip(".!?;:")
        clauses.append(f"{reference}: {motion}")
    return (
        "A locked-off video in which "
        + "; ".join(clauses)
        + ". Every subject keeps the source identity and appearance, and the source scene "
        "remains unchanged outside the physically required action support."
    )


_VERIFIED_RELEASE_TOKEN = object()


@dataclass(frozen=True)
class VerifiedPostGenerationRelease:
    """Opaque result of SSHSIG verification for one exact global manifest."""

    release_id: str
    payload_sha256: str
    receipt_sha256: str
    signer_fingerprint: str
    global_manifest_sha256: str
    row_schema_version: str
    authorized_rows: tuple[tuple[str, str], ...]
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED_RELEASE_TOKEN:
            raise ManifestError(
                "VerifiedPostGenerationRelease can only be created by the verifier"
            )

    def authorize(self, parent: Mapping[str, Any]) -> None:
        iid = _safe_id(parent.get("iid"), field="parent.iid")
        expected = dict(self.authorized_rows).get(iid)
        if expected is None:
            raise ManifestError(f"parent {iid} is outside the verified release")
        if object_sha256(parent) != expected:
            raise ManifestError(f"parent {iid} differs from the verified release row")


def _validate_timestamp(value: Any) -> str:
    text = _nonempty_text(value, field="issued_at_utc")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError("issued_at_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise ManifestError("issued_at_utc must include a timezone")
    return text


def _release_field_contract(
    *,
    source_path_key: str,
    target_path_key: str,
    source_sha256_key: str,
    target_sha256_key: str,
) -> dict[str, str]:
    values = {
        "iid_key": "iid",
        "source_path_key": source_path_key,
        "target_path_key": target_path_key,
        "source_sha256_key": source_sha256_key,
        "target_sha256_key": target_sha256_key,
    }
    for key, value in values.items():
        _nonempty_text(value, field=f"field_contract.{key}")
    return values


def _release_row_closure(
    row: Mapping[str, Any],
    *,
    row_schema_version: str,
    field_contract: Mapping[str, str],
    verify_media: bool,
) -> dict[str, Any]:
    if row.get("schema_version") != row_schema_version:
        raise ManifestError(
            f"post-generation row schema must be {row_schema_version!r}"
        )
    iid = _safe_id(row.get("iid"), field="post-generation row iid")
    if not _is_parent_training_eligible(row):
        raise ManifestError(
            f"post-generation row {iid} is incomplete, preview, or ineligible"
        )
    source_path = _nonempty_text(
        get_nested(row, field_contract["source_path_key"]),
        field=field_contract["source_path_key"],
    )
    target_path = _nonempty_text(
        get_nested(row, field_contract["target_path_key"]),
        field=field_contract["target_path_key"],
    )
    source_sha = _sha256(
        get_nested(row, field_contract["source_sha256_key"]),
        field=field_contract["source_sha256_key"],
    )
    target_sha = _sha256(
        get_nested(row, field_contract["target_sha256_key"]),
        field=field_contract["target_sha256_key"],
    )
    if verify_media:
        _verified_file_sha256(source_path, source_sha, field=f"{iid} source video")
        _verified_file_sha256(target_path, target_sha, field=f"{iid} target video")
    return {
        "iid": iid,
        "row_sha256": object_sha256(row),
        "schema_version": row_schema_version,
        "production_eligible": True,
        "human_review_status": str(row["human_review_status"]).casefold(),
        "source_video_sha256": source_sha,
        "target_video_sha256": target_sha,
    }


def _release_manifest_closure(
    global_manifest_path: os.PathLike[str] | str,
    *,
    row_schema_version: str,
    field_contract: Mapping[str, str],
    verify_media: bool,
) -> tuple[list[dict[str, Any]], bytes, Path, list[dict[str, Any]], dict[str, Any]]:
    rows, raw, resolved = _strict_canonical_jsonl(
        global_manifest_path, field="post-generation global manifest"
    )
    expected_schema = _nonempty_text(
        row_schema_version, field="row_schema_version"
    )
    closures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        closure = _release_row_closure(
            row,
            row_schema_version=expected_schema,
            field_contract=field_contract,
            verify_media=verify_media,
        )
        if closure["iid"] in seen:
            raise ManifestError(
                f"post-generation global manifest has duplicate IID: {closure['iid']}"
            )
        seen.add(closure["iid"])
        closures.append(closure)
    scope = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows": len(rows),
        "row_schema_version": expected_schema,
        "unique_iids": True,
        "ordered_iids_sha256": _ordered_digest(
            [closure["iid"] for closure in closures]
        ),
        "ordered_row_sha256": _ordered_digest(
            [closure["row_sha256"] for closure in closures]
        ),
    }
    return rows, raw, resolved, closures, scope


def build_post_generation_release_payload(
    *,
    global_manifest_path: os.PathLike[str] | str,
    release_id: str,
    issued_at_utc: str,
    row_schema_version: str,
    source_path_key: str = "source_video_path",
    target_path_key: str = "target_video_path",
    source_sha256_key: str = "source_video_sha256",
    target_sha256_key: str = "target_video_sha256",
) -> dict[str, Any]:
    """Close a fully reviewed post-generation manifest before signing it."""

    field_contract = _release_field_contract(
        source_path_key=source_path_key,
        target_path_key=target_path_key,
        source_sha256_key=source_sha256_key,
        target_sha256_key=target_sha256_key,
    )
    _rows, _raw, _resolved, closures, scope = _release_manifest_closure(
        global_manifest_path,
        row_schema_version=row_schema_version,
        field_contract=field_contract,
        verify_media=True,
    )
    return {
        "schema_version": POSTGEN_RELEASE_PAYLOAD_SCHEMA,
        "release_id": _safe_id(release_id, field="release_id"),
        "issued_at_utc": _validate_timestamp(issued_at_utc),
        "purpose": POSTGEN_RELEASE_PURPOSE,
        "global_manifest": scope,
        "field_contract": field_contract,
        "eligibility": {
            "status": "complete",
            "complete": True,
            "production_eligible": True,
            "human_review_complete": True,
            "preview": False,
            "rejected_rows": 0,
            "media_sha256_verified": True,
        },
        "row_authorizations": closures,
    }


def _public_key_and_fingerprint(
    public_key_path: os.PathLike[str] | str,
) -> tuple[str, str, Path]:
    resolved = _stable_regular_file(public_key_path, field="release signer public key")
    parts = resolved.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519":
        raise ManifestError("release signer public key must be SSH Ed25519")
    public_key = " ".join(parts[:2])
    result = subprocess.run(
        ["ssh-keygen", "-lf", str(resolved), "-E", "sha256"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    tokens = result.stdout.split()
    if result.returncode != 0 or len(tokens) < 2 or not tokens[1].startswith("SHA256:"):
        raise ManifestError(
            "cannot determine release signer fingerprint: " + result.stderr.strip()
        )
    return public_key, tokens[1], resolved


def _write_create_only_json(path: os.PathLike[str] | str, value: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ManifestError(f"release receipt already exists: {output}")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o400)
        try:
            os.link(temporary_name, output)
        except FileExistsError as exc:
            raise ManifestError(f"release receipt already exists: {output}") from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return output.resolve(strict=True)


def sign_post_generation_release(
    *,
    global_manifest_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    signing_key_path: os.PathLike[str] | str,
    public_key_path: os.PathLike[str] | str,
    expected_signer_fingerprint: str,
    release_id: str,
    issued_at_utc: str,
    row_schema_version: str,
    source_path_key: str = "source_video_path",
    target_path_key: str = "target_video_path",
    source_sha256_key: str = "source_video_sha256",
    target_sha256_key: str = "target_video_sha256",
) -> dict[str, Any]:
    """Create one immutable SSHSIG receipt for final generated targets."""

    public_key, fingerprint, _public_path = _public_key_and_fingerprint(public_key_path)
    if fingerprint != _nonempty_text(
        expected_signer_fingerprint, field="expected_signer_fingerprint"
    ):
        raise ManifestError("release signer public-key fingerprint differs from expected")
    signing_key = _stable_regular_file(signing_key_path, field="release signing key")
    derived = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(signing_key)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if derived.returncode != 0 or " ".join(derived.stdout.strip().split()[:2]) != public_key:
        raise ManifestError("release signing key differs from the independent public key")
    signed = build_post_generation_release_payload(
        global_manifest_path=global_manifest_path,
        release_id=release_id,
        issued_at_utc=issued_at_utc,
        row_schema_version=row_schema_version,
        source_path_key=source_path_key,
        target_path_key=target_path_key,
        source_sha256_key=source_sha256_key,
        target_sha256_key=target_sha256_key,
    )
    with tempfile.TemporaryDirectory(prefix="pact-postgen-sign-") as directory:
        message = Path(directory) / "payload.json"
        message.write_bytes(canonical_json_bytes(signed))
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(signing_key),
                "-n",
                POSTGEN_SIGNATURE_NAMESPACE,
                str(message),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        signature_path = Path(str(message) + ".sig")
        if result.returncode != 0 or not signature_path.is_file():
            raise ManifestError(
                "post-generation release signing failed: "
                + result.stderr.decode("utf-8", errors="replace").strip()
            )
        armor = signature_path.read_bytes()
    envelope = {
        "schema_version": POSTGEN_RELEASE_SCHEMA,
        "signed": signed,
        "signature": {
            "format": "SSHSIG",
            "namespace": POSTGEN_SIGNATURE_NAMESPACE,
            "principal": POSTGEN_SIGNER_PRINCIPAL,
            "key_fingerprint": fingerprint,
            "armored_signature_base64": base64.b64encode(armor).decode("ascii"),
        },
    }
    _write_create_only_json(output_path, envelope)
    return envelope


def _verify_post_generation_signature(
    signed: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    public_key_path: os.PathLike[str] | str,
    expected_signer_fingerprint: str,
) -> str:
    public_key, fingerprint, _resolved = _public_key_and_fingerprint(public_key_path)
    if fingerprint != _nonempty_text(
        expected_signer_fingerprint, field="expected_signer_fingerprint"
    ):
        raise ManifestError("release signer public-key fingerprint differs from expected")
    _exact_keys(
        signature,
        {
            "format",
            "namespace",
            "principal",
            "key_fingerprint",
            "armored_signature_base64",
        },
        field="post-generation release signature",
    )
    expected = {
        "format": "SSHSIG",
        "namespace": POSTGEN_SIGNATURE_NAMESPACE,
        "principal": POSTGEN_SIGNER_PRINCIPAL,
        "key_fingerprint": fingerprint,
    }
    if any(signature.get(key) != value for key, value in expected.items()):
        raise ManifestError("post-generation release signature metadata differs")
    try:
        armor = base64.b64decode(
            _nonempty_text(
                signature.get("armored_signature_base64"),
                field="armored_signature_base64",
            ),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError("post-generation release signature is not strict base64") from exc
    if not (
        armor.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
        and armor.endswith(b"-----END SSH SIGNATURE-----\n")
    ):
        raise ManifestError("post-generation release signature armor differs")
    with tempfile.TemporaryDirectory(prefix="pact-postgen-verify-") as directory:
        root = Path(directory)
        allowed = root / "allowed_signers"
        signature_path = root / "payload.sshsig"
        allowed.write_text(
            f"{POSTGEN_SIGNER_PRINCIPAL} {public_key}\n", encoding="utf-8"
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
                POSTGEN_SIGNER_PRINCIPAL,
                "-n",
                POSTGEN_SIGNATURE_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=canonical_json_bytes(signed),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        raise ManifestError(
            "post-generation release SSH signature verification failed: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return fingerprint


def verify_post_generation_release(
    *,
    global_manifest_path: os.PathLike[str] | str,
    release_receipt_path: os.PathLike[str] | str,
    public_key_path: os.PathLike[str] | str,
    expected_signer_fingerprint: str,
    row_schema_version: str,
    source_path_key: str = "source_video_path",
    target_path_key: str = "target_video_path",
    source_sha256_key: str = "source_video_sha256",
    target_sha256_key: str = "target_video_sha256",
) -> VerifiedPostGenerationRelease:
    """Verify a final-target release against an independent SSH trust anchor."""

    receipt_path = _stable_regular_file(
        release_receipt_path, field="post-generation signed release receipt"
    )
    receipt_raw = receipt_path.read_bytes()
    envelope = _parse_strict_json(receipt_raw, field="post-generation release receipt")
    if not isinstance(envelope, Mapping):
        raise ManifestError("post-generation release receipt must contain one object")
    _exact_keys(
        envelope, {"schema_version", "signed", "signature"}, field="release receipt"
    )
    if envelope.get("schema_version") != POSTGEN_RELEASE_SCHEMA:
        raise ManifestError("post-generation release receipt schema differs")
    signed = envelope.get("signed")
    signature = envelope.get("signature")
    if not isinstance(signed, Mapping) or not isinstance(signature, Mapping):
        raise ManifestError("release signed payload and signature must be objects")
    fingerprint = _verify_post_generation_signature(
        signed,
        signature,
        public_key_path=public_key_path,
        expected_signer_fingerprint=expected_signer_fingerprint,
    )
    _exact_keys(
        signed,
        {
            "schema_version",
            "release_id",
            "issued_at_utc",
            "purpose",
            "global_manifest",
            "field_contract",
            "eligibility",
            "row_authorizations",
        },
        field="post-generation release payload",
    )
    if (
        signed.get("schema_version") != POSTGEN_RELEASE_PAYLOAD_SCHEMA
        or signed.get("purpose") != POSTGEN_RELEASE_PURPOSE
    ):
        raise ManifestError("post-generation release payload policy differs")
    release_id = _safe_id(signed.get("release_id"), field="release_id")
    _validate_timestamp(signed.get("issued_at_utc"))
    field_contract = signed.get("field_contract")
    if not isinstance(field_contract, Mapping):
        raise ManifestError("release field_contract must be an object")
    _exact_keys(
        field_contract,
        {
            "iid_key",
            "source_path_key",
            "target_path_key",
            "source_sha256_key",
            "target_sha256_key",
        },
        field="release field_contract",
    )
    if field_contract.get("iid_key") != "iid":
        raise ManifestError("release IID field contract differs")
    normalized_contract = _release_field_contract(
        source_path_key=field_contract.get("source_path_key"),
        target_path_key=field_contract.get("target_path_key"),
        source_sha256_key=field_contract.get("source_sha256_key"),
        target_sha256_key=field_contract.get("target_sha256_key"),
    )
    expected_contract = _release_field_contract(
        source_path_key=source_path_key,
        target_path_key=target_path_key,
        source_sha256_key=source_sha256_key,
        target_sha256_key=target_sha256_key,
    )
    if normalized_contract != expected_contract:
        raise ManifestError("release field contract differs from verifier configuration")
    _rows, _raw, _resolved, closures, scope = _release_manifest_closure(
        global_manifest_path,
        row_schema_version=row_schema_version,
        field_contract=normalized_contract,
        verify_media=True,
    )
    if signed.get("global_manifest") != scope:
        raise ManifestError(
            "post-generation global manifest bytes/order differ from signed release"
        )
    eligibility = signed.get("eligibility")
    expected_eligibility = {
        "status": "complete",
        "complete": True,
        "production_eligible": True,
        "human_review_complete": True,
        "preview": False,
        "rejected_rows": 0,
        "media_sha256_verified": True,
    }
    if eligibility != expected_eligibility:
        raise ManifestError(
            "post-generation release is missing complete production eligibility"
        )
    if signed.get("row_authorizations") != closures:
        raise ManifestError("post-generation release row authorizations differ")
    return VerifiedPostGenerationRelease(
        release_id=release_id,
        payload_sha256=object_sha256(signed),
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        signer_fingerprint=fingerprint,
        global_manifest_sha256=scope["sha256"],
        row_schema_version=row_schema_version,
        authorized_rows=tuple(
            (closure["iid"], closure["row_sha256"]) for closure in closures
        ),
        _token=_VERIFIED_RELEASE_TOKEN,
    )


@dataclass(frozen=True)
class AtomizeOptions:
    source_path_key: str = "source_video_path"
    target_path_key: str = "target_video_path"
    allow_preview: bool = False
    verify_mask_files: bool = False
    verify_media_files: bool = False
    minimum_track_confidence: float = 0.75
    source_sha256_key: str = "source_video_sha256"
    target_sha256_key: str = "target_video_sha256"
    verified_release: VerifiedPostGenerationRelease | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_path_key",
            "target_path_key",
            "source_sha256_key",
            "target_sha256_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestError(f"{name} must be a non-empty dotted key")
        if (
            type(self.allow_preview) is not bool
            or type(self.verify_mask_files) is not bool
            or type(self.verify_media_files) is not bool
        ):
            raise ManifestError("atomization flags must be bool")
        if self.verified_release is not None and not isinstance(
            self.verified_release, VerifiedPostGenerationRelease
        ):
            raise ManifestError("verified_release must come from the SSHSIG verifier")
        threshold = self.minimum_track_confidence
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not 0 <= float(threshold) <= 1
        ):
            raise ManifestError("minimum_track_confidence must be finite in [0, 1]")


def atomize_global_row(
    parent: Mapping[str, Any],
    track_records: Iterable[Mapping[str, Any]],
    *,
    options: AtomizeOptions | None = None,
) -> list[dict[str, Any]]:
    """Create actor/component-local weak-supervision rows from one global pair.

    Camera-changing rows are rejected because their target actor pixels cannot be
    consistently spliced into the source camera trajectory.  Preview parents can
    only produce explicitly training-forbidden preview atoms.
    """

    options = options or AtomizeOptions()
    iid = _safe_id(parent.get("iid"), field="parent.iid")
    parent_eligible = _is_parent_training_eligible(parent)
    if not parent_eligible and not options.allow_preview:
        raise ManifestError(f"parent {iid} is not an accepted production training row")
    if parent_eligible:
        if options.verified_release is None:
            raise ManifestError(
                f"parent {iid} lacks a verified post-generation signed release"
            )
        options.verified_release.authorize(parent)
        if not options.verify_media_files or not options.verify_mask_files:
            raise ManifestError(
                "production atomization requires media and mask SHA-256 verification"
            )

    source_census = parent.get("source_census")
    target_plan = parent.get("target_plan")
    if not isinstance(source_census, Mapping) or not isinstance(target_plan, Mapping):
        raise ManifestError("parent requires source_census and target_plan objects")
    _require_static_camera(source_census, target_plan)
    sources, targets = _validate_subject_tables(source_census, target_plan)
    source_path = _nonempty_text(
        get_nested(parent, options.source_path_key), field=options.source_path_key
    )
    target_path = _nonempty_text(
        get_nested(parent, options.target_path_key), field=options.target_path_key
    )
    source_video_sha256 = _sha256(
        get_nested(parent, options.source_sha256_key), field=options.source_sha256_key
    )
    target_video_sha256 = _sha256(
        get_nested(parent, options.target_sha256_key), field=options.target_sha256_key
    )
    if options.verify_media_files:
        _verified_file_sha256(
            source_path, source_video_sha256, field=f"{iid} source video"
        )
        _verified_file_sha256(
            target_path, target_video_sha256, field=f"{iid} target video"
        )

    validated_tracks: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    subject_components: dict[str, str] = {}
    for raw_track in track_records:
        track = validate_track_record(raw_track, expected_iid=iid)
        component_id = track["component_id"]
        if component_id in seen_components:
            raise ManifestError(f"duplicate component_id for {iid}: {component_id}")
        seen_components.add(component_id)
        if float(track["confidence"]) < options.minimum_track_confidence:
            raise ManifestError(
                f"track component {component_id} is below minimum confidence"
            )
        selected_ids = track["subject_ids"]
        unknown = sorted(set(selected_ids) - set(sources))
        if unknown:
            raise ManifestError(f"track component contains unknown subjects: {unknown}")
        for subject_id in selected_ids:
            prior_component = subject_components.get(subject_id)
            if prior_component is not None:
                raise ManifestError(
                    f"subject {subject_id} appears in multiple components: "
                    f"{prior_component}, {component_id}"
                )
            subject_components[subject_id] = component_id
        validated_tracks.append(track)

    uncovered = sorted(set(sources) - set(subject_components))
    if uncovered:
        raise ManifestError(
            f"component tracks do not cover every dynamic subject: {uncovered}"
        )

    atoms: list[dict[str, Any]] = []
    for track in validated_tracks:
        component_id = track["component_id"]
        selected_ids = track["subject_ids"]
        if options.verify_mask_files:
            for key in ("source_mask_path", "target_mask_path"):
                mask_path = Path(track[key])
                declared = track[key.replace("_path", "_sha256")]
                _verified_file_sha256(
                    mask_path,
                    declared,
                    field=f"{key} for {iid}/{component_id}",
                )

        instruction = _compile_atomic_instruction(selected_ids, sources, targets)
        target_caption = _compile_target_caption_contract(selected_ids, sources, targets)
        atom_id = _safe_id(f"{iid}__{component_id}", field="atom_id")
        selected_subjects = []
        for subject_id in selected_ids:
            selected_subjects.append(
                {
                    "subject_id": subject_id,
                    "stable_reference": sources[subject_id]["stable_reference"],
                    "i0_bbox_xyxy_1000": sources[subject_id].get("i0_bbox_xyxy_1000"),
                    "source_action_signature": sources[subject_id].get(
                        "source_action_signature"
                    ),
                    "source_motion": sources[subject_id]["source_motion"],
                    "target_action_signature": targets[subject_id].get(
                        "target_action_signature"
                    ),
                    "target_motion": targets[subject_id]["target_motion"],
                }
            )

        training_authorized = parent_eligible
        atom = {
            "schema_version": ATOMIC_SCHEMA,
            "atom_id": atom_id,
            "parent_iid": iid,
            "parent_row_sha256": object_sha256(parent),
            "component_id": component_id,
            "selected_subject_ids": list(selected_ids),
            "selected_subjects": selected_subjects,
            "source_video_path": source_path,
            "source_video_sha256": source_video_sha256,
            "global_counterfactual_target_video_path": target_path,
            "global_counterfactual_target_video_sha256": target_video_sha256,
            "source_component_mask_path": track["source_mask_path"],
            "source_component_mask_sha256": track["source_mask_sha256"],
            "target_component_mask_path": track["target_mask_path"],
            "target_component_mask_sha256": track["target_mask_sha256"],
            "track_record_sha256": object_sha256(track),
            "track_confidence": float(track["confidence"]),
            "edit_instruction": instruction,
            "edit_instruction_sha256": text_sha256(instruction),
            "target_caption_contract": target_caption,
            "target_caption_contract_sha256": text_sha256(target_caption),
            "camera_policy": "preserve_locked_off",
            "supervision_policy": "soft_shared_noise_latent_component_splice",
            "training_authorized": training_authorized,
            "training_use_forbidden": not training_authorized,
            "parent_preview_only": not parent_eligible,
            "post_generation_release": (
                {
                    "release_id": options.verified_release.release_id,
                    "payload_sha256": options.verified_release.payload_sha256,
                    "receipt_sha256": options.verified_release.receipt_sha256,
                    "signer_fingerprint": options.verified_release.signer_fingerprint,
                    "global_manifest_sha256": (
                        options.verified_release.global_manifest_sha256
                    ),
                    "row_schema_version": options.verified_release.row_schema_version,
                }
                if options.verified_release is not None and parent_eligible
                else None
            ),
        }
        atoms.append(validate_atomic_row(atom))
    return atoms


def validate_atomic_row(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError("atomic row must be an object")
    row = copy.deepcopy(dict(value))
    if row.get("schema_version") != ATOMIC_SCHEMA:
        raise ManifestError(f"atomic schema must be {ATOMIC_SCHEMA}")
    _safe_id(row.get("atom_id"), field="atom_id")
    _safe_id(row.get("parent_iid"), field="parent_iid")
    ids = row.get("selected_subject_ids")
    if not isinstance(ids, list) or not ids:
        raise ManifestError("selected_subject_ids must be non-empty")
    if len({_safe_id(item, field="selected_subject_ids[]") for item in ids}) != len(ids):
        raise ManifestError("selected_subject_ids contains duplicates")
    instruction = _nonempty_text(row.get("edit_instruction"), field="edit_instruction")
    caption = _nonempty_text(
        row.get("target_caption_contract"), field="target_caption_contract"
    )
    if row.get("edit_instruction_sha256") != text_sha256(instruction):
        raise ManifestError("edit instruction digest differs")
    if row.get("target_caption_contract_sha256") != text_sha256(caption):
        raise ManifestError("target caption contract digest differs")
    for key in (
        "source_video_path",
        "global_counterfactual_target_video_path",
        "source_component_mask_path",
        "target_component_mask_path",
    ):
        _nonempty_text(row.get(key), field=key)
    for key in (
        "parent_row_sha256",
        "track_record_sha256",
        "source_video_sha256",
        "global_counterfactual_target_video_sha256",
        "source_component_mask_sha256",
        "target_component_mask_sha256",
    ):
        _sha256(row.get(key), field=key)
    if row.get("camera_policy") != "preserve_locked_off":
        raise ManifestError("unsupported atomic camera policy")
    if row.get("supervision_policy") != "soft_shared_noise_latent_component_splice":
        raise ManifestError("unsupported supervision policy")
    authorized_raw = row.get("training_authorized")
    forbidden_raw = row.get("training_use_forbidden")
    preview_raw = row.get("parent_preview_only")
    if any(type(value) is not bool for value in (authorized_raw, forbidden_raw, preview_raw)):
        raise ManifestError("training authorization fields must be explicit bool values")
    if authorized_raw == forbidden_raw:
        raise ManifestError("training authorization flags are inconsistent")
    if preview_raw != forbidden_raw:
        raise ManifestError("preview and training-forbidden flags are inconsistent")
    release = row.get("post_generation_release")
    if authorized_raw:
        if not isinstance(release, Mapping):
            raise ManifestError("authorized atom lacks post-generation release binding")
        _exact_keys(
            release,
            {
                "release_id",
                "payload_sha256",
                "receipt_sha256",
                "signer_fingerprint",
                "global_manifest_sha256",
                "row_schema_version",
            },
            field="post_generation_release",
        )
        _safe_id(release.get("release_id"), field="release_id")
        for key in (
            "payload_sha256",
            "receipt_sha256",
            "global_manifest_sha256",
        ):
            _sha256(release.get(key), field=f"post_generation_release.{key}")
        _nonempty_text(
            release.get("signer_fingerprint"),
            field="post_generation_release.signer_fingerprint",
        )
        _nonempty_text(
            release.get("row_schema_version"),
            field="post_generation_release.row_schema_version",
        )
    elif release is not None:
        raise ManifestError("preview atom must not carry a production release binding")
    return row


def load_jsonl(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(value)
    return rows
