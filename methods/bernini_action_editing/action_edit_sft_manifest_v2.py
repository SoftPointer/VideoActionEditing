#!/usr/bin/env python3
"""Fail-closed 0817 action-edit SFT row, manifest, and sampler contract.

This module deliberately stops at data authority.  It does not import or call
the predictor, optimizer, or training runner.  A row is eligible only when it
is a complete ``row_tier=train`` source/instruction/target triplet.  Physical
byte identities and semantic identities are separate:

* source and target use an absolute-path, size, and SHA-256 file envelope;
* instruction text uses an exact UTF-8 size and SHA-256 envelope;
* instruction paraphrases, generation seeds, copies, transcodes, and repeated
  teacher candidates cannot manufacture additional semantic row identities;
* the sampler is the exact ordered member projection of the sealed manifest.

Source-container equivalence and instruction-paraphrase equivalence are not
inferred from self-reported rows.  Every public build/replay API requires an
externally pinned exact equivalence-authority digest and membership closure.
Formal candidates additionally require a caller-pinned, exact qualification-
receipt authority; the row receipts alone are never treated as self-authenticating.

``engineering_smoke`` exists solely to test the data path with an explicit
small minimum.  Its artifacts permanently say that they are neither D0 nor D2
evidence and carry no formal-training authority.  ``formal_d2`` always uses
the fixed 100,000 effective-row floor and accepted qualification receipts,
but this module still does not claim D2 or authorize training because the
global diversity, held-out, decode, and quality receipts live elsewhere.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import PurePosixPath
import re
import stat
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


ROW_SCHEMA = "bernini-action-edit-sft-row-v2"
INSTRUCTION_IDENTITY_SCHEMA = "bernini-action-edit-instruction-semantic-identity-v2"
SEMANTIC_EDIT_IDENTITY_SCHEMA = "bernini-action-edit-semantic-row-identity-v2"
ROW_IDENTITY_SCHEMA = "bernini-action-edit-sft-row-id-v2"
MANIFEST_SCHEMA = "bernini-action-edit-sft-train-manifest-v2"
SAMPLER_SCHEMA = "bernini-action-edit-sft-exact-sampler-v2"
QUALIFICATION_RECEIPT_SCHEMA = "bernini-action-edit-target-qualification-receipt-v2"
EQUIVALENCE_AUTHORITY_SCHEMA = "bernini-action-edit-equivalence-authority-v2"
QUALIFICATION_AUTHORITY_SCHEMA = "bernini-action-edit-qualification-authority-v2"

BUILD_MODE_FORMAL_D2 = "formal_d2"
BUILD_MODE_ENGINEERING_SMOKE = "engineering_smoke"
BUILD_MODES = (BUILD_MODE_FORMAL_D2, BUILD_MODE_ENGINEERING_SMOKE)

D0_MINIMUM_COUNT = 2_000
D2_MINIMUM_COUNT = 100_000
SOURCE_SEMANTIC_EDIT_CAP = 8
ACTOR_SCENE_ROW_CAP = 16

TRAINING_SUBSETS = (
    "general_edit",
    "action_motion",
    "interaction_contact",
    "noop_preservation",
    "long_horizon",
)
NONTRAIN_TIERS = ("calibration", "promotion", "locked_final")
ROW_TIERS = ("train",) + NONTRAIN_TIERS
CAMERA_CLASSES = ("static", "moving", "cut-free")
TARGET_PROVENANCE = (
    "real",
    "simulator",
    "licensed-dataset",
    "teacher-pseudo",
)
TARGET_SEMANTIC_TRUTH_CLASSES = (
    "real-counterfactual",
    "simulator-gt",
    "licensed-paired",
    "teacher-pseudo",
    "continuation",
    "noop",
)
QUALIFICATION_STATUSES = ("pending", "unqualified", "accepted")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}\Z")
_KNOWN_PLACEHOLDER_AUTHORITY_DIGESTS = {
    hashlib.sha256(token.encode("utf-8")).hexdigest()
    for token in (
        "human-review",
        "encoder",
        "qualified-q-y",
        "compatibility",
        "placeholder",
        "test",
        "fake",
        "todo",
    )
}


class ActionEditSFTManifestError(RuntimeError):
    """Raised before ambiguous or non-train data gains sampler authority."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the repository's closed, deterministic JSON representation."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ActionEditSFTManifestError(
            "value is not canonical JSON: {}".format(error)
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionEditSFTManifestError("{} must be an object".format(label))
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ActionEditSFTManifestError("{} must be an array".format(label))
    return value


def _closed(
    value: Any,
    required: Set[str],
    *,
    label: str,
    optional: Optional[Set[str]] = None,
) -> Mapping[str, Any]:
    row = _mapping(value, label=label)
    optional_fields = set() if optional is None else optional
    actual = set(row)
    missing = required - actual
    extra = actual - required - optional_fields
    if missing or extra:
        raise ActionEditSFTManifestError(
            "{} field closure differs: missing={} extra={}".format(
                label, sorted(missing), sorted(extra)
            )
        )
    return row


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ActionEditSFTManifestError(
            "{} must be non-empty text without boundary whitespace".format(label)
        )
    return value


def _identifier(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ActionEditSFTManifestError("{} is not a safe identifier".format(label))
    return text


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ActionEditSFTManifestError(
            "{} must be a lowercase SHA-256".format(label)
        )
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActionEditSFTManifestError(
            "{} must be a non-negative integer".format(label)
        )
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    result = _nonnegative_integer(value, label=label)
    if result == 0:
        raise ActionEditSFTManifestError("{} must be positive".format(label))
    return result


def _optional_sha256(value: Any, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _sha256(value, label=label)


def _authority_sha256(value: Any, *, label: str) -> str:
    digest = _sha256(value, label=label)
    # Catch conventional test/place-holder digests before they can be printed
    # as qualification evidence.  Cryptographic authority still comes from a
    # separately pinned manifest digest, not from this sanity check.
    if len(set(digest)) < 4 or digest in _KNOWN_PLACEHOLDER_AUTHORITY_DIGESTS:
        raise ActionEditSFTManifestError(
            "{} is a placeholder, not an authority digest".format(label)
        )
    return digest


def _optional_text(value: Any, *, label: str) -> Optional[str]:
    if value is None:
        return None
    return _text(value, label=label)


def _absolute_posix_path(value: Any, *, label: str) -> str:
    path = _text(value, label=label)
    if "\x00" in path:
        raise ActionEditSFTManifestError("{} contains NUL".format(label))
    pure = PurePosixPath(path)
    if not pure.is_absolute() or str(pure) != path or "." in pure.parts or ".." in pure.parts:
        raise ActionEditSFTManifestError(
            "{} must be a normalized absolute POSIX path".format(label)
        )
    return path


def _verify_regular_file_envelope(
    *,
    path: str,
    expected_size: int,
    expected_sha256: str,
    label: str,
    capture_bytes: bool = False,
) -> Optional[bytes]:
    """Hash a stable, plain regular file without following a symlink."""

    if os.path.realpath(path) != path:
        raise ActionEditSFTManifestError(
            "{} canonical path differs or traverses a symlink".format(label)
        )
    current = os.sep
    for component in PurePosixPath(path).parts[1:-1]:
        current = os.path.join(current, component)
        try:
            component_stat = os.lstat(current)
        except OSError as error:
            raise ActionEditSFTManifestError(
                "{} path component cannot be lstat'ed: {}".format(label, error)
            ) from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ActionEditSFTManifestError(
                "{} traverses an intermediate symlink".format(label)
            )
        if not stat.S_ISDIR(component_stat.st_mode):
            raise ActionEditSFTManifestError(
                "{} path component is not a directory".format(label)
            )
    try:
        path_before = os.lstat(path)
    except OSError as error:
        raise ActionEditSFTManifestError(
            "{} cannot be lstat'ed: {}".format(label, error)
        ) from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ActionEditSFTManifestError("{} is not a plain regular file".format(label))

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActionEditSFTManifestError(
            "{} cannot be opened without following links: {}".format(label, error)
        ) from error

    digest = hashlib.sha256()
    total = 0
    captured = bytearray() if capture_bytes else None
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise ActionEditSFTManifestError(
                "{} descriptor is not a regular file".format(label)
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
            if captured is not None:
                captured.extend(block)
        descriptor_after = os.fstat(descriptor)
    except OSError as error:
        raise ActionEditSFTManifestError(
            "{} could not be read completely: {}".format(label, error)
        ) from error
    finally:
        os.close(descriptor)

    try:
        path_after = os.lstat(path)
    except OSError as error:
        raise ActionEditSFTManifestError(
            "{} disappeared after hashing: {}".format(label, error)
        ) from error

    identity_before = (
        path_before.st_dev,
        path_before.st_ino,
        path_before.st_mode,
        path_before.st_size,
        path_before.st_mtime_ns,
        path_before.st_ctime_ns,
    )
    descriptor_identity_before = (
        descriptor_before.st_dev,
        descriptor_before.st_ino,
        descriptor_before.st_mode,
        descriptor_before.st_size,
        descriptor_before.st_mtime_ns,
        descriptor_before.st_ctime_ns,
    )
    descriptor_identity_after = (
        descriptor_after.st_dev,
        descriptor_after.st_ino,
        descriptor_after.st_mode,
        descriptor_after.st_size,
        descriptor_after.st_mtime_ns,
        descriptor_after.st_ctime_ns,
    )
    identity_after = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_mode,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if not (
        identity_before
        == descriptor_identity_before
        == descriptor_identity_after
        == identity_after
    ):
        raise ActionEditSFTManifestError("{} changed while it was hashed".format(label))
    if total != expected_size:
        raise ActionEditSFTManifestError(
            "{} size differs: expected={} actual={}".format(label, expected_size, total)
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ActionEditSFTManifestError(
            "{} SHA-256 differs: expected={} actual={}".format(
                label, expected_sha256, actual_sha256
            )
        )
    return None if captured is None else bytes(captured)


def _validate_file_envelope(
    value: Any, *, label: str, verify_files: bool
) -> Dict[str, Any]:
    row = _closed(
        value,
        {"path", "sha256", "size_bytes"},
        label="{} file envelope".format(label),
    )
    path = _absolute_posix_path(row["path"], label="{} path".format(label))
    sha = _sha256(row["sha256"], label="{} SHA-256".format(label))
    size = _positive_integer(row["size_bytes"], label="{} size".format(label))
    if verify_files:
        _verify_regular_file_envelope(
            path=path,
            expected_size=size,
            expected_sha256=sha,
            label=label,
        )
    return {"path": path, "sha256": sha, "size_bytes": size}


def instruction_semantic_identity(value: Any) -> Dict[str, Any]:
    """Return the seed/paraphrase-independent structured instruction identity."""

    row = _mapping(value, label="instruction")
    preserve = [
        _identifier(item, label="instruction preserve member")
        for item in _sequence(row.get("preserve"), label="instruction preserve")
    ]
    if not preserve or preserve != sorted(set(preserve)):
        raise ActionEditSFTManifestError(
            "instruction preserve must be a sorted, unique, non-empty list"
        )
    return {
        "schema_version": INSTRUCTION_IDENTITY_SCHEMA,
        "actor": _identifier(row.get("actor"), label="instruction actor"),
        "action": _identifier(row.get("action"), label="instruction action"),
        "object": _identifier(row.get("object"), label="instruction object"),
        "direction": _identifier(row.get("direction"), label="instruction direction"),
        "speed": _identifier(row.get("speed"), label="instruction speed"),
        "amplitude": _identifier(
            row.get("amplitude"), label="instruction amplitude"
        ),
        "onset": _identifier(row.get("onset"), label="instruction onset"),
        "outcome": _identifier(row.get("outcome"), label="instruction outcome"),
        "terminal_state": _identifier(
            row.get("terminal_state"), label="instruction terminal state"
        ),
        "preserve": preserve,
    }


def expected_instruction_semantic_id(value: Any) -> str:
    return object_sha256(instruction_semantic_identity(value))


def validate_equivalence_authority(
    value: Any, *, expected_authority_digest: str
) -> Dict[str, Any]:
    """Validate the externally frozen source/paraphrase equivalence closure."""

    row = _closed(
        value,
        {
            "schema_version",
            "exact_member_closure",
            "sources",
            "instructions",
            "authority_digest",
        },
        label="equivalence authority",
    )
    if row["schema_version"] != EQUIVALENCE_AUTHORITY_SCHEMA:
        raise ActionEditSFTManifestError("equivalence authority schema differs")
    if row["exact_member_closure"] is not True:
        raise ActionEditSFTManifestError("equivalence authority is not exact")
    declared_digest = _authority_sha256(
        row["authority_digest"], label="equivalence authority digest"
    )
    if _authority_sha256(
        expected_authority_digest, label="expected equivalence authority digest"
    ) != declared_digest:
        raise ActionEditSFTManifestError("equivalence authority differs from pinned digest")
    unsigned = dict(row)
    del unsigned["authority_digest"]
    if object_sha256(unsigned) != declared_digest:
        raise ActionEditSFTManifestError("equivalence authority self-digest differs")

    normalized_sources = []
    seen_canonical_ids: Set[str] = set()
    seen_source_ids: Set[str] = set()
    seen_source_shas: Set[str] = set()
    for index, raw_source in enumerate(
        _sequence(row["sources"], label="equivalence sources")
    ):
        source = _closed(
            raw_source,
            {
                "canonical_source_id",
                "source_ids",
                "upstream_group_id",
                "actor_scene_group_id",
                "file_sha256s",
            },
            label="equivalence source {}".format(index),
        )
        canonical_id = _identifier(
            source["canonical_source_id"], label="canonical source ID"
        )
        source_ids = [
            _identifier(item, label="equivalence source ID")
            for item in _sequence(source["source_ids"], label="equivalence source IDs")
        ]
        file_shas = [
            _sha256(item, label="equivalence source file SHA-256")
            for item in _sequence(
                source["file_sha256s"], label="equivalence source file SHA-256s"
            )
        ]
        if (
            canonical_id in seen_canonical_ids
            or not source_ids
            or source_ids != sorted(set(source_ids))
            or not file_shas
            or file_shas != sorted(set(file_shas))
            or seen_source_ids.intersection(source_ids)
            or seen_source_shas.intersection(file_shas)
        ):
            raise ActionEditSFTManifestError(
                "equivalence source entries are not a disjoint exact partition"
            )
        seen_canonical_ids.add(canonical_id)
        seen_source_ids.update(source_ids)
        seen_source_shas.update(file_shas)
        normalized_sources.append(
            {
                "canonical_source_id": canonical_id,
                "source_ids": source_ids,
                "upstream_group_id": _identifier(
                    source["upstream_group_id"], label="equivalence upstream group"
                ),
                "actor_scene_group_id": _identifier(
                    source["actor_scene_group_id"],
                    label="equivalence actor-scene group",
                ),
                "file_sha256s": file_shas,
            }
        )
    if [item["canonical_source_id"] for item in normalized_sources] != sorted(
        seen_canonical_ids
    ):
        raise ActionEditSFTManifestError(
            "equivalence sources must be sorted by canonical source ID"
        )

    normalized_instructions = []
    seen_semantic_ids: Set[str] = set()
    seen_text_shas: Set[str] = set()
    identity_fields = {
        "schema_version",
        "actor",
        "action",
        "object",
        "direction",
        "speed",
        "amplitude",
        "onset",
        "outcome",
        "terminal_state",
        "preserve",
    }
    for index, raw_instruction in enumerate(
        _sequence(row["instructions"], label="equivalence instructions")
    ):
        instruction = _closed(
            raw_instruction,
            {"semantic_id", "identity", "text_sha256s"},
            label="equivalence instruction {}".format(index),
        )
        identity_raw = _closed(
            instruction["identity"],
            identity_fields,
            label="equivalence instruction identity",
        )
        if identity_raw["schema_version"] != INSTRUCTION_IDENTITY_SCHEMA:
            raise ActionEditSFTManifestError(
                "equivalence instruction identity schema differs"
            )
        identity = instruction_semantic_identity(identity_raw)
        semantic_id = _sha256(
            instruction["semantic_id"], label="equivalence instruction semantic ID"
        )
        text_shas = [
            _sha256(item, label="equivalence instruction text SHA-256")
            for item in _sequence(
                instruction["text_sha256s"],
                label="equivalence instruction text SHA-256s",
            )
        ]
        if semantic_id != object_sha256(identity):
            raise ActionEditSFTManifestError(
                "equivalence instruction semantic ID differs from identity"
            )
        if (
            semantic_id in seen_semantic_ids
            or not text_shas
            or text_shas != sorted(set(text_shas))
            or seen_text_shas.intersection(text_shas)
        ):
            raise ActionEditSFTManifestError(
                "equivalence instruction entries are not a disjoint exact partition"
            )
        seen_semantic_ids.add(semantic_id)
        seen_text_shas.update(text_shas)
        normalized_instructions.append(
            {
                "semantic_id": semantic_id,
                "identity": identity,
                "text_sha256s": text_shas,
            }
        )
    if [item["semantic_id"] for item in normalized_instructions] != sorted(
        seen_semantic_ids
    ):
        raise ActionEditSFTManifestError(
            "equivalence instructions must be sorted by semantic ID"
        )
    normalized = {
        "schema_version": EQUIVALENCE_AUTHORITY_SCHEMA,
        "exact_member_closure": True,
        "sources": normalized_sources,
        "instructions": normalized_instructions,
        "authority_digest": declared_digest,
    }
    if normalized != row:
        raise ActionEditSFTManifestError(
            "equivalence authority is not in normalized closed form"
        )
    return normalized


def _validate_rows_against_equivalence_authority(
    rows: Sequence[Mapping[str, Any]], authority: Mapping[str, Any]
) -> None:
    sources = {
        item["canonical_source_id"]: item for item in authority["sources"]
    }
    instructions = {
        item["semantic_id"]: item for item in authority["instructions"]
    }
    for row in rows:
        canonical_id = row["source"]["canonical_source_id"]
        source_authority = sources.get(canonical_id)
        if source_authority is None:
            raise ActionEditSFTManifestError(
                "row canonical source is absent from frozen equivalence authority"
            )
        if (
            row["source"]["source_id"] not in source_authority["source_ids"]
            or row["source"]["sha256"] not in source_authority["file_sha256s"]
            or row["upstream_group_id"] != source_authority["upstream_group_id"]
            or row["actor_scene_group_id"]
            != source_authority["actor_scene_group_id"]
        ):
            raise ActionEditSFTManifestError(
                "row source is not exactly bound by frozen equivalence authority"
            )
        semantic_id = row["instruction"]["semantic_id"]
        instruction_authority = instructions.get(semantic_id)
        if instruction_authority is None:
            raise ActionEditSFTManifestError(
                "row instruction semantics are absent from frozen equivalence authority"
            )
        if (
            row["instruction"]["sha256"]
            not in instruction_authority["text_sha256s"]
            or instruction_semantic_identity(row["instruction"])
            != instruction_authority["identity"]
        ):
            raise ActionEditSFTManifestError(
                "row instruction is not exactly bound by frozen equivalence authority"
            )


def validate_qualification_authority(
    value: Any, *, expected_authority_digest: str
) -> Dict[str, Any]:
    row = _closed(
        value,
        {
            "schema_version",
            "exact_member_closure",
            "qualification_receipt_sha256s",
            "authority_digest",
        },
        label="qualification authority",
    )
    if row["schema_version"] != QUALIFICATION_AUTHORITY_SCHEMA:
        raise ActionEditSFTManifestError("qualification authority schema differs")
    if row["exact_member_closure"] is not True:
        raise ActionEditSFTManifestError("qualification authority is not exact")
    receipt_shas = [
        _authority_sha256(item, label="qualified receipt SHA-256")
        for item in _sequence(
            row["qualification_receipt_sha256s"],
            label="qualified receipt SHA-256s",
        )
    ]
    if not receipt_shas or receipt_shas != sorted(set(receipt_shas)):
        raise ActionEditSFTManifestError(
            "qualification authority receipts must be sorted and unique"
        )
    declared_digest = _authority_sha256(
        row["authority_digest"], label="qualification authority digest"
    )
    if _authority_sha256(
        expected_authority_digest,
        label="expected qualification authority digest",
    ) != declared_digest:
        raise ActionEditSFTManifestError(
            "qualification authority differs from pinned digest"
        )
    unsigned = dict(row)
    del unsigned["authority_digest"]
    if object_sha256(unsigned) != declared_digest:
        raise ActionEditSFTManifestError("qualification authority self-digest differs")
    normalized = {
        "schema_version": QUALIFICATION_AUTHORITY_SCHEMA,
        "exact_member_closure": True,
        "qualification_receipt_sha256s": receipt_shas,
        "authority_digest": declared_digest,
    }
    if normalized != row:
        raise ActionEditSFTManifestError(
            "qualification authority is not in normalized closed form"
        )
    return normalized


def _validate_mode_qualification_authority(
    rows: Sequence[Mapping[str, Any]],
    *,
    build_mode: str,
    qualification_authority: Any,
    expected_qualification_authority_digest: Optional[str],
) -> Optional[Dict[str, Any]]:
    if build_mode == BUILD_MODE_ENGINEERING_SMOKE:
        if (
            qualification_authority is not None
            or expected_qualification_authority_digest is not None
        ):
            raise ActionEditSFTManifestError(
                "engineering smoke must not attach qualification authority"
            )
        return None
    if qualification_authority is None or expected_qualification_authority_digest is None:
        raise ActionEditSFTManifestError(
            "formal D2 candidate requires pinned qualification authority"
        )
    sealed = validate_qualification_authority(
        qualification_authority,
        expected_authority_digest=expected_qualification_authority_digest,
    )
    expected_receipts = sorted(
        {row["target"]["qualification_receipt"]["sha256"] for row in rows}
    )
    if sealed["qualification_receipt_sha256s"] != expected_receipts:
        raise ActionEditSFTManifestError(
            "formal rows are not the exact qualified receipt authority closure"
        )
    return sealed


def _validate_source(value: Any, *, verify_files: bool) -> Dict[str, Any]:
    fields = {
        "path",
        "sha256",
        "size_bytes",
        "source_id",
        "canonical_source_id",
        "actor_ids",
        "scene_id",
        "camera_class",
        "initial_state",
    }
    row = _closed(value, fields, label="source")
    envelope = _validate_file_envelope(
        {key: row[key] for key in ("path", "sha256", "size_bytes")},
        label="source",
        verify_files=verify_files,
    )
    actor_ids = [
        _identifier(item, label="source actor ID")
        for item in _sequence(row["actor_ids"], label="source actor IDs")
    ]
    if not actor_ids or actor_ids != sorted(set(actor_ids)):
        raise ActionEditSFTManifestError(
            "source actor IDs must be sorted, unique, and non-empty"
        )
    camera_class = row["camera_class"]
    if camera_class not in CAMERA_CLASSES:
        raise ActionEditSFTManifestError("source camera class differs")
    return {
        **envelope,
        "source_id": _identifier(row["source_id"], label="source ID"),
        "canonical_source_id": _identifier(
            row["canonical_source_id"], label="canonical source ID"
        ),
        "actor_ids": actor_ids,
        "scene_id": _identifier(row["scene_id"], label="source scene ID"),
        "camera_class": camera_class,
        "initial_state": _text(row["initial_state"], label="source initial state"),
    }


def _validate_instruction(value: Any) -> Dict[str, Any]:
    fields = {
        "text",
        "sha256",
        "size_bytes",
        "encoding",
        "semantic_id",
        "template_family",
        "actor",
        "action",
        "object",
        "direction",
        "speed",
        "amplitude",
        "onset",
        "outcome",
        "terminal_state",
        "preserve",
    }
    row = _closed(value, fields, label="instruction")
    if row["encoding"] != "utf-8":
        raise ActionEditSFTManifestError("instruction encoding must be exactly utf-8")
    text = _text(row["text"], label="instruction text")
    text_bytes = text.encode("utf-8")
    declared_size = _positive_integer(
        row["size_bytes"], label="instruction UTF-8 size"
    )
    declared_sha = _sha256(row["sha256"], label="instruction SHA-256")
    if declared_size != len(text_bytes):
        raise ActionEditSFTManifestError("instruction UTF-8 size differs")
    if declared_sha != hashlib.sha256(text_bytes).hexdigest():
        raise ActionEditSFTManifestError("instruction UTF-8 SHA-256 differs")

    identity = instruction_semantic_identity(row)
    semantic_id = _sha256(row["semantic_id"], label="instruction semantic ID")
    if semantic_id != object_sha256(identity):
        raise ActionEditSFTManifestError(
            "instruction semantic ID is not derived from structured semantics"
        )
    return {
        "text": text,
        "sha256": declared_sha,
        "size_bytes": declared_size,
        "encoding": "utf-8",
        "semantic_id": semantic_id,
        "template_family": _identifier(
            row["template_family"], label="instruction template family"
        ),
        "actor": identity["actor"],
        "action": identity["action"],
        "object": identity["object"],
        "direction": identity["direction"],
        "speed": identity["speed"],
        "amplitude": identity["amplitude"],
        "onset": identity["onset"],
        "outcome": identity["outcome"],
        "terminal_state": identity["terminal_state"],
        "preserve": identity["preserve"],
    }


def _validate_target(value: Any, *, verify_files: bool) -> Dict[str, Any]:
    required = {
        "path",
        "sha256",
        "size_bytes",
        "provenance",
        "semantic_truth_class",
        "teacher_id",
        "qualification_status",
    }
    optional = {
        "qualification_receipt",
        "human_review",
        "human_review_receipt_sha256",
        "action_feature_encoder_sha256",
        "q_y_sha256",
        "compatibility_receipt_sha256",
    }
    row = _closed(value, required, optional=optional, label="target")
    envelope = _validate_file_envelope(
        {key: row[key] for key in ("path", "sha256", "size_bytes")},
        label="target",
        verify_files=verify_files,
    )
    provenance = row["provenance"]
    if provenance not in TARGET_PROVENANCE:
        raise ActionEditSFTManifestError("target provenance differs")
    truth_class = row["semantic_truth_class"]
    if truth_class not in TARGET_SEMANTIC_TRUTH_CLASSES:
        raise ActionEditSFTManifestError("target semantic truth class differs")
    teacher_id = _optional_text(row["teacher_id"], label="target teacher ID")
    if provenance == "teacher-pseudo" and teacher_id is None:
        raise ActionEditSFTManifestError(
            "teacher-pseudo target requires a teacher identity"
        )
    qualification_status = row["qualification_status"]
    if qualification_status not in QUALIFICATION_STATUSES:
        raise ActionEditSFTManifestError("target qualification status differs")
    human_review = row.get("human_review")
    authority_names = (
        "human_review_receipt_sha256",
        "action_feature_encoder_sha256",
        "q_y_sha256",
        "compatibility_receipt_sha256",
    )
    if qualification_status == "accepted":
        if human_review != "accepted":
            raise ActionEditSFTManifestError(
                "accepted target requires exact human_review=accepted"
            )
        authorities = {
            name: _authority_sha256(
                row.get(name), label="target {}".format(name)
            )
            for name in authority_names
        }
        qualification_receipt = _validate_file_envelope(
            row.get("qualification_receipt"),
            label="target qualification receipt",
            verify_files=verify_files,
        )
    else:
        if (
            human_review is not None
            or row.get("qualification_receipt") is not None
            or any(row.get(name) is not None for name in authority_names)
        ):
            raise ActionEditSFTManifestError(
                "pending/unqualified target must not carry qualification authority"
            )
        authorities = {name: None for name in authority_names}
        qualification_receipt = None
    return {
        **envelope,
        "provenance": provenance,
        "semantic_truth_class": truth_class,
        "teacher_id": teacher_id,
        "qualification_status": qualification_status,
        "qualification_receipt": qualification_receipt,
        "human_review": human_review,
        **authorities,
    }


def _validate_qualification_receipt(
    *,
    envelope: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    if envelope["size_bytes"] > 1024 * 1024:
        raise ActionEditSFTManifestError("qualification receipt is unexpectedly large")
    payload = _verify_regular_file_envelope(
        path=envelope["path"],
        expected_size=envelope["size_bytes"],
        expected_sha256=envelope["sha256"],
        label="target qualification receipt",
        capture_bytes=True,
    )
    assert payload is not None
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActionEditSFTManifestError(
            "qualification receipt is not canonical UTF-8 JSON"
        ) from error
    receipt = _closed(
        value,
        {
            "schema_version",
            "qualification_status",
            "row_id",
            "semantic_edit_id",
            "canonical_source_id",
            "source_sha256",
            "instruction_sha256",
            "instruction_semantic_id",
            "instruction_identity",
            "target_sha256",
            "action_family",
            "training_subset",
            "target_provenance",
            "target_semantic_truth_class",
            "target_teacher_id",
            "human_review",
            "human_review_receipt_sha256",
            "action_feature_encoder_sha256",
            "q_y_sha256",
            "compatibility_receipt_sha256",
            "receipt_digest",
        },
        label="qualification receipt",
    )
    declared_digest = _authority_sha256(
        receipt["receipt_digest"], label="qualification receipt digest"
    )
    unsigned = dict(receipt)
    del unsigned["receipt_digest"]
    if object_sha256(unsigned) != declared_digest:
        raise ActionEditSFTManifestError("qualification receipt self-digest differs")
    if canonical_json_bytes(dict(receipt)) != payload:
        raise ActionEditSFTManifestError("qualification receipt bytes are not canonical")
    source = row["source"]
    instruction = row["instruction"]
    target = row["target"]
    expected = {
        "schema_version": QUALIFICATION_RECEIPT_SCHEMA,
        "qualification_status": "accepted",
        "row_id": row["row_id"],
        "semantic_edit_id": row["semantic_edit_id"],
        "canonical_source_id": source["canonical_source_id"],
        "source_sha256": source["sha256"],
        "instruction_sha256": instruction["sha256"],
        "instruction_semantic_id": instruction["semantic_id"],
        "instruction_identity": instruction_semantic_identity(instruction),
        "target_sha256": target["sha256"],
        "action_family": row["action_family"],
        "training_subset": row["training_subset"],
        "target_provenance": target["provenance"],
        "target_semantic_truth_class": target["semantic_truth_class"],
        "target_teacher_id": target["teacher_id"],
        "human_review": target["human_review"],
        "human_review_receipt_sha256": target["human_review_receipt_sha256"],
        "action_feature_encoder_sha256": target[
            "action_feature_encoder_sha256"
        ],
        "q_y_sha256": target["q_y_sha256"],
        "compatibility_receipt_sha256": target[
            "compatibility_receipt_sha256"
        ],
    }
    if unsigned != expected:
        raise ActionEditSFTManifestError(
            "qualification receipt is not exactly bound to the full row semantics, endpoints, and authorities"
        )


def semantic_edit_identity(value: Any) -> Dict[str, Any]:
    """Return the effective identity; seeds, targets, and labels are absent.

    Classification labels are validated separately and never create another
    sample.  This prevents relabeling one source/instruction semantic pair as a
    different family, subset, group, or target class to inflate effective N.
    """

    row = _mapping(value, label="SFT row")
    source = _mapping(row.get("source"), label="source")
    instruction = _mapping(row.get("instruction"), label="instruction")
    return {
        "schema_version": SEMANTIC_EDIT_IDENTITY_SCHEMA,
        "canonical_source_id": _identifier(
            source.get("canonical_source_id"), label="canonical source ID"
        ),
        "instruction_semantic_id": _sha256(
            instruction.get("semantic_id"), label="instruction semantic ID"
        ),
    }


def expected_semantic_edit_id(value: Any) -> str:
    return object_sha256(semantic_edit_identity(value))


def expected_row_id(value: Any) -> str:
    semantic_edit_id = value.get("semantic_edit_id") if isinstance(value, Mapping) else None
    semantic_edit_id = _sha256(semantic_edit_id, label="semantic edit ID")
    return object_sha256(
        {
            "schema_version": ROW_IDENTITY_SCHEMA,
            "semantic_edit_id": semantic_edit_id,
        }
    )


def _validate_source_target_contract(row: Mapping[str, Any]) -> None:
    if row["source"]["sha256"] != row["target"]["sha256"]:
        return
    if not (
        row["training_subset"] == "noop_preservation"
        and row["target"]["semantic_truth_class"] == "noop"
        and row["instruction"]["action"] == "noop"
    ):
        raise ActionEditSFTManifestError(
            "source==target is allowed only for the explicit noop preservation contract"
        )


def _validate_action_anchors(
    value: Any,
    *,
    target_sha256: str,
    verify_files: bool,
) -> List[Dict[str, Any]]:
    anchors = []
    fields = {
        "path",
        "sha256",
        "size_bytes",
        "generation_seed",
        "role",
        "q_anchor_sha256",
        "compatibility_score",
        "compatibility_verdict",
        "training_use",
        "compatibility_receipt_sha256",
    }
    for index, raw_anchor in enumerate(
        _sequence(value, label="action anchors")
    ):
        anchor = _closed(
            raw_anchor, fields, label="action anchor {}".format(index)
        )
        envelope = _validate_file_envelope(
            {key: anchor[key] for key in ("path", "sha256", "size_bytes")},
            label="action anchor {}".format(index),
            verify_files=verify_files,
        )
        if envelope["sha256"] == target_sha256:
            raise ActionEditSFTManifestError(
                "action-reference-only anchor bytes must not be the supervised target"
            )
        seed = _nonnegative_integer(
            anchor["generation_seed"], label="action anchor generation seed"
        )
        if seed >= 2**63:
            raise ActionEditSFTManifestError("action anchor seed is outside uint63")
        if anchor["role"] != "action-reference-only":
            raise ActionEditSFTManifestError(
                "action anchor role must be action-reference-only"
            )
        score = anchor["compatibility_score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ActionEditSFTManifestError(
                "action anchor compatibility score must be finite in [0,1]"
            )
        if anchor["compatibility_verdict"] not in ("accept", "reject", "abstain"):
            raise ActionEditSFTManifestError(
                "action anchor compatibility verdict differs"
            )
        if anchor["training_use"] not in (
            "point-distill",
            "contrastive-only",
            "excluded",
        ):
            raise ActionEditSFTManifestError("action anchor training use differs")
        anchors.append(
            {
                **envelope,
                "generation_seed": seed,
                "role": "action-reference-only",
                "q_anchor_sha256": _authority_sha256(
                    anchor["q_anchor_sha256"], label="action anchor q SHA-256"
                ),
                "compatibility_score": float(score),
                "compatibility_verdict": anchor["compatibility_verdict"],
                "training_use": anchor["training_use"],
                "compatibility_receipt_sha256": _authority_sha256(
                    anchor["compatibility_receipt_sha256"],
                    label="action anchor compatibility receipt SHA-256",
                ),
            }
        )
    ordering = [(item["sha256"], item["path"]) for item in anchors]
    if ordering != sorted(set(ordering)):
        raise ActionEditSFTManifestError(
            "action anchors must be sorted and unique by byte identity/path"
        )
    return anchors


def validate_train_row(value: Any, *, verify_files: bool = True) -> Dict[str, Any]:
    """Validate and normalize one train-only action-edit SFT row."""

    required = {
        "schema_version",
        "row_id",
        "semantic_edit_id",
        "action_family",
        "upstream_group_id",
        "actor_scene_group_id",
        "source",
        "instruction",
        "target",
        "row_tier",
        "training_subset",
    }
    optional = {
        "calibration_kind",
        "evaluation_stratum",
        "action_anchors",
        "annotations",
        "generation_seed",
        "copy_of_row_id",
        "transcode_of_sha256",
    }
    row = _closed(value, required, optional=optional, label="SFT row")
    if row["schema_version"] != ROW_SCHEMA:
        raise ActionEditSFTManifestError("SFT row schema differs")
    tier = row["row_tier"]
    if tier not in ROW_TIERS:
        raise ActionEditSFTManifestError("SFT row tier differs")
    if tier != "train":
        raise ActionEditSFTManifestError(
            "{} row is forbidden from the train manifest and sampler".format(tier)
        )
    subset = row["training_subset"]
    if subset not in TRAINING_SUBSETS:
        raise ActionEditSFTManifestError("train row training subset differs")
    if row.get("calibration_kind") is not None:
        raise ActionEditSFTManifestError(
            "train row must not set calibration_kind"
        )
    if row.get("evaluation_stratum") is not None:
        raise ActionEditSFTManifestError(
            "train row must not set evaluation_stratum"
        )

    source = _validate_source(row["source"], verify_files=verify_files)
    instruction = _validate_instruction(row["instruction"])
    target = _validate_target(row["target"], verify_files=verify_files)
    action_family = _identifier(row["action_family"], label="action family")
    upstream_group_id = _identifier(
        row["upstream_group_id"], label="upstream group ID"
    )
    actor_scene_group_id = _identifier(
        row["actor_scene_group_id"], label="actor-scene group ID"
    )

    generation_seed = row.get("generation_seed")
    if generation_seed is not None:
        generation_seed = _nonnegative_integer(
            generation_seed, label="generation seed"
        )
        if generation_seed >= 2**63:
            raise ActionEditSFTManifestError("generation seed is outside uint63")
    copy_of_row_id = _optional_sha256(
        row.get("copy_of_row_id"), label="copy source row ID"
    )
    transcode_of_sha256 = _optional_sha256(
        row.get("transcode_of_sha256"), label="transcode source SHA-256"
    )

    action_anchors = _validate_action_anchors(
        row.get("action_anchors", []),
        target_sha256=target["sha256"],
        verify_files=verify_files,
    )
    annotations = dict(_mapping(row.get("annotations", {}), label="annotations"))
    # Reject non-JSON metadata and non-deterministic numbers even though
    # annotations do not participate in effective-N identity.
    canonical_json_bytes(annotations)

    normalized = {
        "schema_version": ROW_SCHEMA,
        "row_id": row["row_id"],
        "semantic_edit_id": row["semantic_edit_id"],
        "action_family": action_family,
        "upstream_group_id": upstream_group_id,
        "actor_scene_group_id": actor_scene_group_id,
        "source": source,
        "instruction": instruction,
        "target": target,
        "action_anchors": action_anchors,
        "annotations": annotations,
        "row_tier": "train",
        "training_subset": subset,
        "calibration_kind": None,
        "evaluation_stratum": None,
        "generation_seed": generation_seed,
        "copy_of_row_id": copy_of_row_id,
        "transcode_of_sha256": transcode_of_sha256,
    }
    semantic_edit_id = _sha256(
        normalized["semantic_edit_id"], label="semantic edit ID"
    )
    expected_semantic = expected_semantic_edit_id(normalized)
    if semantic_edit_id != expected_semantic:
        raise ActionEditSFTManifestError(
            "semantic edit ID is not derived from effective semantic identity"
        )
    normalized["semantic_edit_id"] = semantic_edit_id
    row_id = _sha256(normalized["row_id"], label="row ID")
    if row_id != expected_row_id(normalized):
        raise ActionEditSFTManifestError("row ID is not content-addressed")
    if copy_of_row_id is not None and copy_of_row_id != row_id:
        raise ActionEditSFTManifestError(
            "copy row must point to the same effective semantic row ID"
        )
    normalized["row_id"] = row_id
    _validate_source_target_contract(normalized)
    if target["qualification_status"] == "accepted" and verify_files:
        _validate_qualification_receipt(
            envelope=target["qualification_receipt"],
            row=normalized,
        )
    return normalized


def _candidate_rank(row: Mapping[str, Any]) -> Tuple[int, str]:
    derivative = int(
        row["copy_of_row_id"] is not None or row["transcode_of_sha256"] is not None
    )
    return derivative, object_sha256(row)


def _drop_duplicates_and_derivatives(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply endpoint then semantic deduplication in the preregistered order."""

    endpoint_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["source"]["sha256"],
            row["instruction"]["sha256"],
            row["target"]["sha256"],
        )
        endpoint_groups[key].append(row)

    endpoint_representatives: List[Dict[str, Any]] = []
    endpoint_dropped = 0
    for key in sorted(endpoint_groups):
        group = endpoint_groups[key]
        row_ids = {item["row_id"] for item in group}
        if len(row_ids) != 1:
            raise ActionEditSFTManifestError(
                "one endpoint-byte triple declares conflicting semantic identities"
            )
        endpoint_representatives.append(min(group, key=_candidate_rank))
        endpoint_dropped += len(group) - 1

    semantic_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in endpoint_representatives:
        semantic_groups[row["row_id"]].append(row)

    eligible: List[Dict[str, Any]] = []
    semantic_dropped = 0
    derivative_only_dropped = 0
    for row_id in sorted(semantic_groups):
        group = semantic_groups[row_id]
        originals = [
            item
            for item in group
            if item["copy_of_row_id"] is None
            and item["transcode_of_sha256"] is None
        ]
        if not originals:
            derivative_only_dropped += len(group)
            continue
        eligible.append(min(originals, key=lambda item: object_sha256(item)))
        semantic_dropped += len(group) - 1

    return eligible, {
        "endpoint_duplicate_rows_dropped": endpoint_dropped,
        "semantic_duplicate_rows_dropped": semantic_dropped,
        "derivative_only_rows_dropped": derivative_only_dropped,
    }


def _validate_source_authority_bindings(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject aliases that could evade source or actor-scene effective-N caps."""

    by_source_sha: Dict[str, str] = {}
    by_canonical_source: Dict[str, Tuple[str, str, str]] = {}
    by_source_id: Dict[str, str] = {}
    for row in rows:
        # Explicit copies/transcodes are diagnostics and are removed before
        # effective-N selection.  Their physical source bytes may legitimately
        # differ, so only would-be eligible originals establish this authority.
        if row["copy_of_row_id"] is not None or row["transcode_of_sha256"] is not None:
            continue
        source_sha = row["source"]["sha256"]
        source_id = row["source"]["source_id"]
        canonical_source_id = row["source"]["canonical_source_id"]
        binding = (
            source_id,
            row["upstream_group_id"],
            row["actor_scene_group_id"],
        )
        previous_canonical = by_source_sha.setdefault(source_sha, canonical_source_id)
        if previous_canonical != canonical_source_id:
            raise ActionEditSFTManifestError(
                "one source SHA declares conflicting canonical source identities"
            )
        previous_binding = by_canonical_source.setdefault(canonical_source_id, binding)
        if previous_binding != binding:
            raise ActionEditSFTManifestError(
                "one canonical source declares conflicting source/group identities"
            )
        previous_canonical_for_id = by_source_id.setdefault(
            source_id, canonical_source_id
        )
        if previous_canonical_for_id != canonical_source_id:
            raise ActionEditSFTManifestError(
                "one source ID declares multiple canonical source identities"
            )


def _validate_semantic_classification_bindings(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Keep descriptive labels from becoming alternate effective identities."""

    by_semantic_pair: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    by_exact_instruction: Dict[Tuple[str, str], str] = {}
    for row in rows:
        if row["copy_of_row_id"] is not None or row["transcode_of_sha256"] is not None:
            continue
        canonical_source_id = row["source"]["canonical_source_id"]
        semantic_id = row["instruction"]["semantic_id"]
        classification = (
            row["action_family"],
            row["training_subset"],
            row["target"]["semantic_truth_class"],
        )
        semantic_key = (canonical_source_id, semantic_id)
        previous_classification = by_semantic_pair.setdefault(
            semantic_key, classification
        )
        if previous_classification != classification:
            raise ActionEditSFTManifestError(
                "one effective source/instruction semantic pair declares conflicting classifications"
            )
        exact_key = (canonical_source_id, row["instruction"]["sha256"])
        previous_semantic_id = by_exact_instruction.setdefault(exact_key, semantic_id)
        if previous_semantic_id != semantic_id:
            raise ActionEditSFTManifestError(
                "one exact source/instruction byte pair declares conflicting semantics"
            )


def _apply_effective_caps(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    source_counts: Dict[str, int] = defaultdict(int)
    after_source: List[Dict[str, Any]] = []
    source_dropped = 0
    for row in sorted(rows, key=lambda item: item["row_id"]):
        source_key = row["source"]["canonical_source_id"]
        if source_counts[source_key] >= SOURCE_SEMANTIC_EDIT_CAP:
            source_dropped += 1
            continue
        source_counts[source_key] += 1
        after_source.append(row)

    actor_scene_counts: Dict[str, int] = defaultdict(int)
    effective: List[Dict[str, Any]] = []
    actor_scene_dropped = 0
    for row in after_source:
        group_id = row["actor_scene_group_id"]
        if actor_scene_counts[group_id] >= ACTOR_SCENE_ROW_CAP:
            actor_scene_dropped += 1
            continue
        actor_scene_counts[group_id] += 1
        effective.append(row)
    return effective, source_dropped, actor_scene_dropped


def _validate_effective_target_uniqueness(rows: Sequence[Mapping[str, Any]]) -> None:
    target_owners: Dict[str, str] = {}
    for row in rows:
        target_sha = row["target"]["sha256"]
        previous_row_id = target_owners.setdefault(target_sha, row["row_id"])
        if previous_row_id != row["row_id"]:
            raise ActionEditSFTManifestError(
                "one target byte identity is reused across distinct semantic rows"
            )


def _mode_contract(
    *, build_mode: str, engineering_smoke_minimum_count: Optional[int]
) -> Dict[str, Any]:
    if build_mode not in BUILD_MODES:
        raise ActionEditSFTManifestError("build mode differs")
    if build_mode == BUILD_MODE_ENGINEERING_SMOKE:
        if engineering_smoke_minimum_count is None:
            raise ActionEditSFTManifestError(
                "engineering smoke requires an explicit minimum count"
            )
        minimum = _positive_integer(
            engineering_smoke_minimum_count,
            label="engineering smoke minimum count",
        )
        return {
            "claim_scope": "engineering-smoke-only-not-d0-or-d2",
            "qualification_scope": "pending-or-unqualified-engineering-only",
            "minimum_effective_count": minimum,
            "engineering_smoke_only": True,
            "d0_claimed": False,
            "d2_claimed": False,
            "formal_training_authorized": False,
        }
    if engineering_smoke_minimum_count is not None:
        raise ActionEditSFTManifestError(
            "formal D2 mode must not accept an engineering minimum override"
        )
    return {
        "claim_scope": "d2-effective-count-candidate-not-training-authority",
        "qualification_scope": "accepted-canonical-receipt-bound",
        "minimum_effective_count": D2_MINIMUM_COUNT,
        "engineering_smoke_only": False,
        "d0_claimed": False,
        # This module does not possess the held-out split, decode/quality, or
        # global diversity receipts required to authorize the 0817 long run.
        "d2_claimed": False,
        "formal_training_authorized": False,
    }


def _validate_mode_qualification(
    rows: Sequence[Mapping[str, Any]], *, build_mode: str
) -> None:
    for row in rows:
        status = row["target"]["qualification_status"]
        if build_mode == BUILD_MODE_ENGINEERING_SMOKE:
            if status not in ("pending", "unqualified"):
                raise ActionEditSFTManifestError(
                    "engineering smoke accepts only pending/unqualified targets"
                )
        elif status != "accepted":
            raise ActionEditSFTManifestError(
                "formal D2 candidate requires qualification_status=accepted"
            )


def build_train_manifest(
    raw_rows: Sequence[Any],
    *,
    build_mode: str,
    equivalence_authority: Any,
    expected_equivalence_authority_digest: str,
    qualification_authority: Any = None,
    expected_qualification_authority_digest: Optional[str] = None,
    engineering_smoke_minimum_count: Optional[int] = None,
    verify_files: bool = True,
) -> Dict[str, Any]:
    """Build a deterministic train-only manifest from a raw row pool."""

    raw = list(_sequence(raw_rows, label="raw SFT rows"))
    mode = _mode_contract(
        build_mode=build_mode,
        engineering_smoke_minimum_count=engineering_smoke_minimum_count,
    )
    if verify_files is not True:
        raise ActionEditSFTManifestError(
            "manifest construction requires physical file verification"
        )
    sealed_equivalence_authority = validate_equivalence_authority(
        equivalence_authority,
        expected_authority_digest=expected_equivalence_authority_digest,
    )
    normalized = [
        validate_train_row(item, verify_files=verify_files) for item in raw
    ]
    _validate_rows_against_equivalence_authority(
        normalized, sealed_equivalence_authority
    )
    _validate_mode_qualification(normalized, build_mode=build_mode)
    _validate_source_authority_bindings(normalized)
    _validate_semantic_classification_bindings(normalized)
    eligible_before_caps, dedup = _drop_duplicates_and_derivatives(normalized)
    effective, source_dropped, actor_scene_dropped = _apply_effective_caps(
        eligible_before_caps
    )
    _validate_effective_target_uniqueness(effective)
    sealed_qualification_authority = _validate_mode_qualification_authority(
        effective,
        build_mode=build_mode,
        qualification_authority=qualification_authority,
        expected_qualification_authority_digest=expected_qualification_authority_digest,
    )
    effective_count = len(effective)
    minimum = mode["minimum_effective_count"]
    if effective_count < minimum:
        raise ActionEditSFTManifestError(
            "effective train rows are below the explicit minimum: actual={} minimum={}".format(
                effective_count, minimum
            )
        )

    deduplication = {
        **dedup,
        "source_cap_rows_dropped": source_dropped,
        "actor_scene_cap_rows_dropped": actor_scene_dropped,
    }
    rows_digest = object_sha256(
        [
            {"row_id": row["row_id"], "row_sha256": object_sha256(row)}
            for row in effective
        ]
    )
    unsigned = {
        "schema_version": MANIFEST_SCHEMA,
        "build_mode": build_mode,
        **mode,
        "row_tier": "train",
        "files_verified": verify_files is True,
        "equivalence_authority_digest": sealed_equivalence_authority[
            "authority_digest"
        ],
        "qualification_authority_digest": (
            None
            if sealed_qualification_authority is None
            else sealed_qualification_authority["authority_digest"]
        ),
        "raw_accounting_authoritative": False,
        "raw_row_count": len(normalized),
        "eligible_before_caps_count": len(eligible_before_caps),
        "engineering_effective_N": (
            effective_count
            if build_mode == BUILD_MODE_ENGINEERING_SMOKE
            else 0
        ),
        "D2_train_eligible_effective_N": (
            effective_count if build_mode == BUILD_MODE_FORMAL_D2 else 0
        ),
        "source_semantic_edit_cap": SOURCE_SEMANTIC_EDIT_CAP,
        "actor_scene_row_cap": ACTOR_SCENE_ROW_CAP,
        "deduplication": deduplication,
        "exact_member_closure": True,
        "rows": effective,
        "rows_digest": rows_digest,
    }
    return {**unsigned, "manifest_digest": object_sha256(unsigned)}


def _manifest_fields() -> Set[str]:
    return {
        "schema_version",
        "build_mode",
        "claim_scope",
        "qualification_scope",
        "minimum_effective_count",
        "engineering_smoke_only",
        "d0_claimed",
        "d2_claimed",
        "formal_training_authorized",
        "row_tier",
        "files_verified",
        "equivalence_authority_digest",
        "qualification_authority_digest",
        "raw_accounting_authoritative",
        "raw_row_count",
        "eligible_before_caps_count",
        "engineering_effective_N",
        "D2_train_eligible_effective_N",
        "source_semantic_edit_cap",
        "actor_scene_row_cap",
        "deduplication",
        "exact_member_closure",
        "rows",
        "rows_digest",
        "manifest_digest",
    }


def validate_train_manifest(
    value: Any,
    *,
    equivalence_authority: Any,
    expected_equivalence_authority_digest: str,
    expected_manifest_digest: str,
    qualification_authority: Any = None,
    expected_qualification_authority_digest: Optional[str] = None,
    verify_files: bool = True,
) -> Dict[str, Any]:
    """Replay the complete manifest schema, count, cap, and self-digest."""

    row = _closed(value, _manifest_fields(), label="SFT train manifest")
    if verify_files is not True:
        raise ActionEditSFTManifestError(
            "manifest replay requires physical file verification"
        )
    if row["schema_version"] != MANIFEST_SCHEMA:
        raise ActionEditSFTManifestError("SFT train manifest schema differs")
    declared_digest = _sha256(row["manifest_digest"], label="manifest digest")
    unsigned = dict(row)
    del unsigned["manifest_digest"]
    if object_sha256(unsigned) != declared_digest:
        raise ActionEditSFTManifestError("manifest self-digest differs")
    sealed_equivalence_authority = validate_equivalence_authority(
        equivalence_authority,
        expected_authority_digest=expected_equivalence_authority_digest,
    )
    if (
        _sha256(
            row["equivalence_authority_digest"],
            label="manifest equivalence authority digest",
        )
        != sealed_equivalence_authority["authority_digest"]
    ):
        raise ActionEditSFTManifestError(
            "manifest equivalence authority binding differs"
        )
    if _sha256(
        expected_manifest_digest, label="expected manifest digest"
    ) != declared_digest:
        raise ActionEditSFTManifestError("manifest differs from pinned digest")
    for boolean_key in (
        "engineering_smoke_only",
        "d0_claimed",
        "d2_claimed",
        "formal_training_authorized",
        "files_verified",
        "raw_accounting_authoritative",
        "exact_member_closure",
    ):
        if type(row[boolean_key]) is not bool:
            raise ActionEditSFTManifestError(
                "manifest {} must be boolean".format(boolean_key)
            )
    if row["files_verified"] is not True:
        raise ActionEditSFTManifestError(
            "manifest does not record successful physical file verification"
        )
    if row["raw_accounting_authoritative"] is not False:
        raise ActionEditSFTManifestError(
            "raw accounting is diagnostic until a raw-pool closure is attached"
        )
    if row["build_mode"] == BUILD_MODE_FORMAL_D2:
        if row["files_verified"] is not True or verify_files is not True:
            raise ActionEditSFTManifestError(
                "formal D2 manifest requires physical file verification"
            )

    declared_minimum = _positive_integer(
        row["minimum_effective_count"], label="manifest minimum effective count"
    )
    mode = _mode_contract(
        build_mode=row["build_mode"],
        engineering_smoke_minimum_count=(
            declared_minimum
            if row["build_mode"] == BUILD_MODE_ENGINEERING_SMOKE
            else None
        ),
    )
    for key, expected in mode.items():
        if row[key] != expected:
            raise ActionEditSFTManifestError(
                "manifest mode contract differs at {}".format(key)
            )
    if row["row_tier"] != "train" or row["exact_member_closure"] is not True:
        raise ActionEditSFTManifestError("manifest is not a train-only exact closure")
    source_cap = _positive_integer(
        row["source_semantic_edit_cap"], label="manifest source cap"
    )
    actor_scene_cap = _positive_integer(
        row["actor_scene_row_cap"], label="manifest actor-scene cap"
    )
    if source_cap != SOURCE_SEMANTIC_EDIT_CAP:
        raise ActionEditSFTManifestError("manifest source cap differs")
    if actor_scene_cap != ACTOR_SCENE_ROW_CAP:
        raise ActionEditSFTManifestError("manifest actor-scene cap differs")

    rows = []
    for index, item in enumerate(_sequence(row["rows"], label="manifest rows")):
        normalized = validate_train_row(item, verify_files=verify_files)
        if normalized != item:
            raise ActionEditSFTManifestError(
                "manifest row {} is not in normalized closed form".format(index)
            )
        if normalized["copy_of_row_id"] is not None or normalized[
            "transcode_of_sha256"
        ] is not None:
            raise ActionEditSFTManifestError(
                "derivative row leaked into effective manifest members"
            )
        rows.append(normalized)
    _validate_mode_qualification(rows, build_mode=row["build_mode"])
    sealed_qualification_authority = _validate_mode_qualification_authority(
        rows,
        build_mode=row["build_mode"],
        qualification_authority=qualification_authority,
        expected_qualification_authority_digest=expected_qualification_authority_digest,
    )
    expected_qualification_digest = (
        None
        if sealed_qualification_authority is None
        else sealed_qualification_authority["authority_digest"]
    )
    if row["qualification_authority_digest"] != expected_qualification_digest:
        raise ActionEditSFTManifestError(
            "manifest qualification authority binding differs"
        )
    _validate_rows_against_equivalence_authority(
        rows, sealed_equivalence_authority
    )
    row_ids = [item["row_id"] for item in rows]
    if row_ids != sorted(set(row_ids)):
        raise ActionEditSFTManifestError(
            "manifest rows must be sorted and unique by effective row ID"
        )
    _validate_effective_target_uniqueness(rows)
    effective_count = len(rows)
    engineering_effective_n = _nonnegative_integer(
        row["engineering_effective_N"], label="engineering effective N"
    )
    d2_effective_n = _nonnegative_integer(
        row["D2_train_eligible_effective_N"], label="D2 effective N"
    )
    expected_engineering_n = (
        effective_count
        if row["build_mode"] == BUILD_MODE_ENGINEERING_SMOKE
        else 0
    )
    expected_d2_n = (
        effective_count if row["build_mode"] == BUILD_MODE_FORMAL_D2 else 0
    )
    if (
        engineering_effective_n != expected_engineering_n
        or d2_effective_n != expected_d2_n
    ):
        raise ActionEditSFTManifestError("manifest mode-specific effective N differs")
    if effective_count < declared_minimum:
        raise ActionEditSFTManifestError("manifest is below its declared minimum")

    _validate_source_authority_bindings(rows)
    _validate_semantic_classification_bindings(rows)
    source_counts: Dict[str, int] = defaultdict(int)
    actor_scene_counts: Dict[str, int] = defaultdict(int)
    for item in rows:
        source_counts[item["source"]["canonical_source_id"]] += 1
        actor_scene_counts[item["actor_scene_group_id"]] += 1
    if any(count > SOURCE_SEMANTIC_EDIT_CAP for count in source_counts.values()):
        raise ActionEditSFTManifestError("manifest exceeds the per-source cap")
    if any(count > ACTOR_SCENE_ROW_CAP for count in actor_scene_counts.values()):
        raise ActionEditSFTManifestError("manifest exceeds the actor-scene cap")

    dedup = _closed(
        row["deduplication"],
        {
            "endpoint_duplicate_rows_dropped",
            "semantic_duplicate_rows_dropped",
            "derivative_only_rows_dropped",
            "source_cap_rows_dropped",
            "actor_scene_cap_rows_dropped",
        },
        label="manifest deduplication",
    )
    dropped = 0
    for key in dedup:
        dropped += _nonnegative_integer(dedup[key], label="deduplication {}".format(key))
    raw_count = _nonnegative_integer(row["raw_row_count"], label="raw row count")
    eligible_before_caps = _nonnegative_integer(
        row["eligible_before_caps_count"], label="eligible-before-caps count"
    )
    if raw_count != effective_count + dropped:
        raise ActionEditSFTManifestError("manifest raw/effective/drop accounting differs")
    if eligible_before_caps != (
        effective_count
        + dedup["source_cap_rows_dropped"]
        + dedup["actor_scene_cap_rows_dropped"]
    ):
        raise ActionEditSFTManifestError("manifest pre-cap accounting differs")

    expected_rows_digest = object_sha256(
        [
            {"row_id": item["row_id"], "row_sha256": object_sha256(item)}
            for item in rows
        ]
    )
    if _sha256(row["rows_digest"], label="rows digest") != expected_rows_digest:
        raise ActionEditSFTManifestError("manifest rows digest differs")
    return dict(row)


def build_exact_sampler(
    manifest: Any,
    *,
    equivalence_authority: Any,
    expected_equivalence_authority_digest: str,
    expected_manifest_digest: str,
    qualification_authority: Any = None,
    expected_qualification_authority_digest: Optional[str] = None,
    verify_files: bool = True,
) -> Dict[str, Any]:
    """Project the manifest into its only accepted ordered sampler closure."""

    sealed = validate_train_manifest(
        manifest,
        equivalence_authority=equivalence_authority,
        expected_equivalence_authority_digest=expected_equivalence_authority_digest,
        qualification_authority=qualification_authority,
        expected_qualification_authority_digest=expected_qualification_authority_digest,
        verify_files=verify_files,
        expected_manifest_digest=expected_manifest_digest,
    )
    return _sampler_from_sealed_manifest(sealed)


def _sampler_from_sealed_manifest(sealed: Mapping[str, Any]) -> Dict[str, Any]:
    members = [
        {
            "ordinal": index,
            "row_id": row["row_id"],
            "row_sha256": object_sha256(row),
        }
        for index, row in enumerate(sealed["rows"])
    ]
    unsigned = {
        "schema_version": SAMPLER_SCHEMA,
        "build_mode": sealed["build_mode"],
        "claim_scope": sealed["claim_scope"],
        "qualification_scope": sealed["qualification_scope"],
        "engineering_smoke_only": sealed["engineering_smoke_only"],
        "d0_claimed": sealed["d0_claimed"],
        "d2_claimed": sealed["d2_claimed"],
        "formal_training_authorized": sealed["formal_training_authorized"],
        "files_verified": sealed["files_verified"],
        "equivalence_authority_digest": sealed[
            "equivalence_authority_digest"
        ],
        "qualification_authority_digest": sealed[
            "qualification_authority_digest"
        ],
        "row_tier": "train",
        "manifest_digest": sealed["manifest_digest"],
        "member_count": len(members),
        "exact_member_closure": True,
        "members": members,
        "members_digest": object_sha256(members),
    }
    return {**unsigned, "sampler_digest": object_sha256(unsigned)}


def validate_exact_sampler(
    manifest: Any,
    sampler: Any,
    *,
    equivalence_authority: Any,
    expected_equivalence_authority_digest: str,
    expected_manifest_digest: str,
    expected_sampler_digest: str,
    qualification_authority: Any = None,
    expected_qualification_authority_digest: Optional[str] = None,
    verify_files: bool = True,
) -> Dict[str, Any]:
    """Require the sampler to equal, not merely be a subset of, the manifest."""

    sealed = validate_train_manifest(
        manifest,
        equivalence_authority=equivalence_authority,
        expected_equivalence_authority_digest=expected_equivalence_authority_digest,
        qualification_authority=qualification_authority,
        expected_qualification_authority_digest=expected_qualification_authority_digest,
        verify_files=verify_files,
        expected_manifest_digest=expected_manifest_digest,
    )
    expected = _sampler_from_sealed_manifest(sealed)
    row = _closed(sampler, set(expected), label="exact SFT sampler")
    if row["schema_version"] != SAMPLER_SCHEMA:
        raise ActionEditSFTManifestError("sampler schema differs")
    for key in (
        "engineering_smoke_only",
        "d0_claimed",
        "d2_claimed",
        "formal_training_authorized",
        "files_verified",
        "exact_member_closure",
    ):
        if type(row[key]) is not bool:
            raise ActionEditSFTManifestError(
                "sampler {} must be boolean".format(key)
            )
    member_count = _nonnegative_integer(
        row["member_count"], label="sampler member count"
    )
    members = []
    for index, raw_member in enumerate(
        _sequence(row["members"], label="sampler members")
    ):
        member = _closed(
            raw_member,
            {"ordinal", "row_id", "row_sha256"},
            label="sampler member {}".format(index),
        )
        ordinal = _nonnegative_integer(
            member["ordinal"], label="sampler member ordinal"
        )
        members.append(
            {
                "ordinal": ordinal,
                "row_id": _sha256(member["row_id"], label="sampler row ID"),
                "row_sha256": _sha256(
                    member["row_sha256"], label="sampler row SHA-256"
                ),
            }
        )
    if member_count != len(members):
        raise ActionEditSFTManifestError("sampler member count differs")
    if _sha256(row["members_digest"], label="sampler members digest") != object_sha256(
        members
    ):
        raise ActionEditSFTManifestError("sampler members digest differs")
    declared_sampler_digest = _sha256(
        row["sampler_digest"], label="sampler digest"
    )
    unsigned_sampler = dict(row)
    del unsigned_sampler["sampler_digest"]
    if object_sha256(unsigned_sampler) != declared_sampler_digest:
        raise ActionEditSFTManifestError("sampler self-digest differs")
    if _sha256(
        expected_sampler_digest, label="expected sampler digest"
    ) != declared_sampler_digest:
        raise ActionEditSFTManifestError("sampler differs from pinned digest")
    if dict(row) != expected:
        raise ActionEditSFTManifestError(
            "sampler is not the exact ordered train-manifest member closure"
        )
    return dict(row)


__all__ = [
    "ACTOR_SCENE_ROW_CAP",
    "ActionEditSFTManifestError",
    "BUILD_MODE_ENGINEERING_SMOKE",
    "BUILD_MODE_FORMAL_D2",
    "D0_MINIMUM_COUNT",
    "D2_MINIMUM_COUNT",
    "EQUIVALENCE_AUTHORITY_SCHEMA",
    "MANIFEST_SCHEMA",
    "QUALIFICATION_AUTHORITY_SCHEMA",
    "QUALIFICATION_RECEIPT_SCHEMA",
    "ROW_SCHEMA",
    "SAMPLER_SCHEMA",
    "SOURCE_SEMANTIC_EDIT_CAP",
    "TRAINING_SUBSETS",
    "build_exact_sampler",
    "build_train_manifest",
    "canonical_json_bytes",
    "expected_instruction_semantic_id",
    "expected_row_id",
    "expected_semantic_edit_id",
    "instruction_semantic_identity",
    "object_sha256",
    "semantic_edit_identity",
    "validate_exact_sampler",
    "validate_equivalence_authority",
    "validate_qualification_authority",
    "validate_train_manifest",
    "validate_train_row",
]
