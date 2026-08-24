#!/usr/bin/env python3
"""Create-only PAIR-v7 fit-only/no-update geometry evidence.

This module deliberately does not consume, reproduce, or reinterpret the
PAIR-v5 population calibration/confirmation decision.  It authenticates two
pre-existing fit examples (one per DP arm) solely as inputs to a read-only
gradient-geometry measurement at the first preregistered schedule cell.

The authority granted here is narrow: callbacks, a loss backward, and VJP
replay may be used to *measure* gradients.  It can never authorize an
optimizer, a parameter update, or an action-editing success claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import tarfile
from typing import Any, Mapping, Optional, Sequence


METHOD_NAME = "bernini-pair-v7-fit-only-no-update-geometry-authority"
EVENT_SCHEMA = "bernini-pair-v7-fit-only-geometry-event-v1"
MANIFEST_SCHEMA = "bernini-pair-v7-fit-only-geometry-manifest-v1"
EVIDENCE_SCHEMA = "bernini-pair-v7-fit-only-geometry-evidence-v1"
VALIDATION_SCHEMA = "bernini-pair-v7-fit-only-geometry-validation-v1"
CAST_V4_METHOD_REVISION = "d5e87caf8f58a63c2ff902386d6e19100f7fcf34"
CAST_V4_GROUP_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-group-v4"
CAST_V4_SCORE_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-score-v4"
CAST_V4_SCORE_FILENAME = "pair-v5-t2v-global-energy-score-v4.json"
CAST_V4_CANDIDATES_PER_GROUP = 20
CAST_V4_TOTAL_CANDIDATES = 40
LEGACY_V3_NO_GO_FILE_SHA256 = (
    "5b2d129b1cc1d14845edc731029bba4bfe8af041d09089d8923b7f082e1f5b52"
)
LEGACY_V3_OPTIMIZER_FALSE_JSON_PATH = "optimizer_authorized"
LEGACY_V3_CONFIRMATION_FALSE_JSON_PATH = "gates.confirmation_overall"
LEGACY_V3_NO_GO_RECEIPT_DIGEST = (
    "3ab868efe8e56d668ce9ef23bc952d5a30449dbdb2c83f92cb775c50dfcaab59"
)
SAIL_PRIOR_NO_SUCCESS_FILE_SHA256 = (
    "b132265a98d3830618eb18875153ab343f21118e41ea67eac57130e70b5d2973"
)
SAIL_PRIOR_NO_SUCCESS_RECEIPT_DIGEST = (
    "de5ff6380b669806c6cd7156e8655683bd918bf7083174b1ead48601c39d115c"
)
SAIL_CHILD_BINDINGS = {
    "dog": {
        "file_sha256": "791c53b13c1ebde5a1cadcd4d4016cbc1bb6e9f3304d90c9a486fdbf8883fd45",
        "receipt_digest": "8d1297159f636eee453e45dd8cdff17542208f0cfe490d8c551136bf25125fe3",
    },
    "human": {
        "file_sha256": "b2ed33bdf4652ca54be7d1f7eed70320315856b1e0179482d4c6bd08ced186d7",
        "receipt_digest": "a1e97c7539c5087e4f8e8f050d108de868c0fdaf821b42be413f0df9f1afba5a",
    },
}

FRAME_COUNT = 81
FPS = 25.0
REFERENCE_INDICES = (0, 27, 53, 80)
LATENT_CHANNELS = 16
LATENT_PHASES = 21
DP_SIZE = 2
FIRST_SCHEDULE_INDEX = 33
BRANCH_ORDER = (
    "action",
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_NO_UPDATE_CLAIMS = {
    "global_population_go": False,
    "optimizer_authorized": False,
    "parameter_update_authorized": False,
    "action_success_claimed": False,
}

_DRAFT_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "fit_candidate_id",
        "action_family",
        "prompt_by_branch",
        "source_sample_id",
        "source_video_path",
        "raw_caption_by_branch",
        "clean_latent_path",
        "clean_latent_tensor_key",
        "official_gaussian_path",
        "official_gaussian_tensor_key",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "fit_candidate_id",
        "action_family",
        "analysis_split",
        "prompt_by_branch",
        "prompt_bank_sha256",
        "source_sample_id",
        "source_video_path",
        "source_video_sha256",
        "source_frame_count",
        "source_fps",
        "source_reference_indices",
        "raw_caption_by_branch",
        "raw_caption_bank_sha256",
        "clean_latent_path",
        "clean_latent_file_sha256",
        "clean_latent_tensor_key",
        "clean_latent_tensor_sha256",
        "official_gaussian_path",
        "official_gaussian_file_sha256",
        "official_gaussian_tensor_key",
        "official_gaussian_tensor_sha256",
        "latent_shape",
        "frame_count",
        "pure_t2v_visual_role",
        "rv2v_target_input_noise_or_donor",
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
        "event_digest",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "authority_scope",
        "checkpoint_tree_sha256",
        "action_adapter_schema_sha256",
        "fit_event_count",
        "events",
        "schedule_indices",
        "first_schedule_index",
        "confirmation_population_consumed",
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
        "manifest_digest",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "method_name",
        "authority_scope",
        "geometry_measurement_authorized",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_digest",
        "checkpoint_tree_sha256",
        "checkpoint_content_receipt_digest",
        "checkpoint_content_identity_binding",
        "action_adapter_schema_sha256",
        "fit_event_count",
        "fit_event_ids",
        "fit_action_families",
        "fit_event_digests",
        "schedule_indices",
        "first_schedule_index",
        "create_only_authoring",
        "confirmation_population_consumed",
        "population_scorer_receipts_consumed",
        "population_scorer_receipts_role",
        "legacy_optimizer_authority_consumed",
        "cast_v4_method_archive",
        "cast_v4_root_spec",
        "cast_v4_groups",
        "selected_action_score_by_event",
        "negative_boundaries",
        "global_population_go",
        "optimizer_authorized",
        "parameter_update_authorized",
        "action_success_claimed",
        "evidence_digest",
    }
)

_CAST_GROUP_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "frozen_checkpoint_receipt_digest",
        "checkpoint_content_binding",
        "schedule_coordinate",
        "candidate_count",
        "candidate_receipt_digests",
        "primary_score_field",
        "phase_conjunctive_role",
        "input_closure",
        "training_performed",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "method_source_revision",
        "method_source_archive_sha256",
        "bernini_revision",
        "veomni_revision",
        "receipt_digest",
    }
)
_CAST_CHECKPOINT_BINDING_FIELDS = frozenset(
    {
        "all_loaded_parameters_frozen",
        "binding_digest",
        "every_file_sha256_verified",
        "freeze_certificate",
        "loaded_components",
        "manifest_sha256",
        "verified_entries_digest",
        "verified_file_count",
    }
)

_CAST_ARCHIVE_REQUIRED = frozenset(
    {
        "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
        "methods/bernini_action_editing/mace_candidate_action_energy.py",
        "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
        "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
        "methods/bernini_action_editing/pair_v5_native_bridge.py",
        "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
    }
)


class PairV7FitOnlyAuthorityError(RuntimeError):
    """Fit-only geometry inputs are ambiguous, mutable, or over-authorized."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV7FitOnlyAuthorityError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(unsigned: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    if digest_field in unsigned:
        raise PairV7FitOnlyAuthorityError("cannot seal an already sealed object")
    value = dict(unsigned)
    for field, expected in _NO_UPDATE_CLAIMS.items():
        if field in value and value[field] is not expected:
            raise PairV7FitOnlyAuthorityError(f"{field} must remain false")
        value[field] = expected
    return {**value, digest_field: object_sha256(value)}


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV7FitOnlyAuthorityError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise PairV7FitOnlyAuthorityError(f"{label} must be lowercase SHA-1")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV7FitOnlyAuthorityError(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV7FitOnlyAuthorityError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _fresh_absolute_output(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PairV7FitOnlyAuthorityError(f"{label} path must be text")
    path = Path(value)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise PairV7FitOnlyAuthorityError(f"{label} must be an absolute file path")
    parent = path.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise PairV7FitOnlyAuthorityError(f"{label} parent must be a plain directory")
    resolved = parent / path.name
    if resolved.exists() or resolved.is_symlink():
        raise PairV7FitOnlyAuthorityError(f"{label} is create-only and already exists")
    return resolved


def _write_create_only_json(path_value: Any, value: Mapping[str, Any]) -> Path:
    path = _fresh_absolute_output(path_value, label="output")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o640)
    try:
        payload = canonical_json_bytes(value) + b"\n"
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def _closed(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise PairV7FitOnlyAuthorityError(
            f"{label} field closure differs: missing={sorted(fields-actual)} "
            f"extra={sorted(actual-fields)}"
        )
    return value


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(
        pairs: Sequence[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        if b"NaN" in raw or b"Infinity" in raw or b"-Infinity" in raw:
            raise ValueError("non-finite JSON token")
        value = json.loads(
            raw.decode("ascii"), object_pairs_hook=reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PairV7FitOnlyAuthorityError(f"{label} is invalid ASCII JSON") from error
    if not isinstance(value, Mapping):
        raise PairV7FitOnlyAuthorityError(f"{label} root must be an object")
    return value


def _inspect_source_media(path: Path) -> Mapping[str, Any]:
    """Read the actual video stream geometry; never trust authored constants."""

    try:
        import av
    except ImportError as error:
        raise PairV7FitOnlyAuthorityError(
            "PyAV is required to inspect source media"
        ) from error
    try:
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise PairV7FitOnlyAuthorityError(
                    "source media video-stream closure differs"
                )
            stream = streams[0]
            fps_fraction = Fraction(str(stream.average_rate))
            frame_count = sum(1 for _ in container.decode(stream))
    except PairV7FitOnlyAuthorityError:
        raise
    except Exception as error:
        raise PairV7FitOnlyAuthorityError(
            "source media cannot be decoded with PyAV"
        ) from error
    if frame_count != FRAME_COUNT or fps_fraction != Fraction(25, 1):
        raise PairV7FitOnlyAuthorityError(
            "source media must decode as exact81 at exactly 25 fps"
        )
    return {"frame_count": frame_count, "fps": float(fps_fraction)}


def _validate_cast_method_archive(
    path_value: Any,
    *,
    expected_sha256: str,
    expected_revision: str,
) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label="CAST-v4 method archive")
    file_sha = _sha256(expected_sha256, label="CAST-v4 method archive")
    revision = _sha1(expected_revision, label="CAST-v4 method revision")
    if revision != CAST_V4_METHOD_REVISION:
        raise PairV7FitOnlyAuthorityError("CAST-v4 method revision differs from d5e87ca")
    if _file_sha256(path) != file_sha:
        raise PairV7FitOnlyAuthorityError("CAST-v4 method archive SHA-256 differs")
    seen: set[str] = set()
    try:
        with tarfile.open(path, "r:*") as handle:
            if handle.pax_headers.get("comment") != revision:
                raise PairV7FitOnlyAuthorityError(
                    "CAST-v4 archive commit identity differs"
                )
            for member in handle.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise PairV7FitOnlyAuthorityError(
                        "CAST-v4 archive contains an unsafe member"
                    )
                seen.add(pure.as_posix().lstrip("./"))
    except PairV7FitOnlyAuthorityError:
        raise
    except (tarfile.TarError, OSError) as error:
        raise PairV7FitOnlyAuthorityError("CAST-v4 archive cannot be audited") from error
    missing = sorted(_CAST_ARCHIVE_REQUIRED - seen)
    if missing:
        raise PairV7FitOnlyAuthorityError(
            f"CAST-v4 archive lacks method closure: {missing}"
        )
    return {
        "path": str(path),
        "file_sha256": file_sha,
        "git_archive_revision": revision,
        "required_member_count": len(_CAST_ARCHIVE_REQUIRED),
        "required_member_closure_present": True,
    }


def _validate_cast_root_spec(
    path_value: Any, *, expected_sha256: str
) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label="CAST-v4 root spec")
    file_sha = _sha256(expected_sha256, label="CAST-v4 root spec")
    if _file_sha256(path) != file_sha:
        raise PairV7FitOnlyAuthorityError("CAST-v4 root spec SHA-256 differs")
    try:
        import pair_v5_t2v_calibration_bank_spec as bank_spec

        _spec, observed = bank_spec.load_sealed_spec(path, file_sha)
    except ModuleNotFoundError:
        # Lightweight unit tests can validate the byte/digest closure without
        # importing the torch-backed scorer package.
        observed = file_sha
    except Exception as error:
        raise PairV7FitOnlyAuthorityError(
            f"CAST-v4 root spec semantic validation failed: {error}"
        ) from error
    if observed != file_sha:
        raise PairV7FitOnlyAuthorityError("CAST-v4 root spec digest differs")
    return {"path": str(path), "file_sha256": file_sha}


def _checkpoint_content_identity_binding(value: Any) -> Mapping[str, Any]:
    fields = {
        "manifest_path",
        "manifest_sha256_computed",
        "manifest_sha256_expected",
        "verified_file_count",
        "every_file_sha256_verified",
        "verified_entries_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PairV7FitOnlyAuthorityError("checkpoint content identity closure differs")
    manifest = _sha256(
        value.get("manifest_sha256_computed"), label="checkpoint manifest"
    )
    manifest_path = _plain_absolute_file(
        value.get("manifest_path"), label="checkpoint content manifest"
    )
    entries = _sha256(
        value.get("verified_entries_digest"), label="checkpoint entries"
    )
    if (
        value.get("manifest_sha256_expected") != manifest
        or _file_sha256(manifest_path) != manifest
        or type(value.get("verified_file_count")) is not int
        or value.get("verified_file_count") != 23
        or value.get("every_file_sha256_verified") is not True
    ):
        raise PairV7FitOnlyAuthorityError("checkpoint content identity differs")
    return {
        "manifest_sha256": manifest,
        "verified_file_count": 23,
        "verified_entries_digest": entries,
        "every_file_sha256_verified": True,
    }


def _embedded_receipt_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = _sha256(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if object_sha256(unsigned) != declared:
        raise PairV7FitOnlyAuthorityError(f"{label} embedded digest differs")
    return declared


def _validate_cast_score_receipt_file(path: Path) -> Mapping[str, Any]:
    row = _strict_json(path, label="CAST-v4 candidate score receipt")
    try:
        import score_pair_v5_t2v_energy_bank_v3 as scorer

        checked = scorer.validate_score_receipt(row)
    except ModuleNotFoundError:
        checked = dict(row)
        if checked.get("schema_version") != CAST_V4_SCORE_SCHEMA:
            raise PairV7FitOnlyAuthorityError("CAST-v4 score schema differs")
        _embedded_receipt_digest(checked, label="CAST-v4 score receipt")
    except Exception as error:
        raise PairV7FitOnlyAuthorityError(
            f"CAST-v4 candidate score validation failed: {error}"
        ) from error
    if (
        checked.get("schema_version") != CAST_V4_SCORE_SCHEMA
        or checked.get("scientific_action_editing_claim") is not False
    ):
        raise PairV7FitOnlyAuthorityError("CAST-v4 score no-success scope differs")
    return checked


def _discover_cast_candidate_receipts(
    group_path: Path,
    *,
    expected_digests: Sequence[str],
) -> list[Mapping[str, Any]]:
    expected = tuple(
        _sha256(value, label="CAST-v4 candidate receipt digest")
        for value in expected_digests
    )
    if len(expected) != CAST_V4_CANDIDATES_PER_GROUP or len(set(expected)) != len(
        expected
    ):
        raise PairV7FitOnlyAuthorityError("CAST-v4 group must bind 20 unique candidates")
    indexed: dict[str, Mapping[str, Any]] = {}
    pattern = f"*/{CAST_V4_SCORE_FILENAME}"
    for path in sorted(group_path.parent.glob(pattern)):
        if not path.is_file() or path.is_symlink() or path.parent.parent != group_path.parent:
            continue
        row = _validate_cast_score_receipt_file(path)
        digest = row.get("receipt_digest")
        if digest not in expected:
            continue
        if digest in indexed:
            raise PairV7FitOnlyAuthorityError("CAST-v4 candidate receipt digest repeats")
        indexed[digest] = {
            "candidate_id": row.get("candidate_id"),
            "path": str(path.resolve(strict=True)),
            "file_sha256": _file_sha256(path),
            "receipt_digest": digest,
            "analysis_split": row.get("analysis_split"),
            "action_family_id": row.get("action_family_id"),
            "semantic_branch": row.get("semantic_branch"),
            "root_spec_raw_sha256": row.get("root_spec_raw_sha256"),
            "frozen_checkpoint_receipt_digest": row.get(
                "frozen_checkpoint_receipt_digest"
            ),
            "checkpoint_content_binding": row.get("checkpoint_content_binding"),
            "geometry_source_video_sha256": row.get(
                "geometry_source_video_sha256"
            ),
            "full_t2v_caption_by_branch": row.get(
                "full_t2v_caption_by_branch"
            ),
            "clean_latent_tensor_sha256": row.get("clean_latent_tensor_sha256"),
            "official_gaussian_tensor_sha256": row.get(
                "official_gaussian_tensor_sha256"
            ),
            "prompt_by_branch": row.get("prompt_by_branch"),
            "candidate_shape": (
                row.get("frozen_t2v_packet_binding", {}).get("candidate_shape")
                if isinstance(row.get("frozen_t2v_packet_binding"), Mapping)
                else None
            ),
            "raw_global_action_energy_score": row.get(
                "raw_global_action_energy_score"
            ),
        }
    if set(indexed) != set(expected):
        raise PairV7FitOnlyAuthorityError(
            "CAST-v4 group candidate receipt file closure differs"
        )
    return [indexed[digest] for digest in expected]


def _validate_cast_group(
    path_value: Any,
    *,
    expected_file_sha256: str,
    root_spec_sha256: str,
    method_archive_sha256: str,
    method_revision: str,
    checkpoint_content_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label="CAST-v4 group receipt")
    file_sha = _sha256(expected_file_sha256, label="CAST-v4 group receipt")
    if _file_sha256(path) != file_sha:
        raise PairV7FitOnlyAuthorityError("CAST-v4 group receipt file SHA differs")
    row = _closed(
        _strict_json(path, label="CAST-v4 group receipt"),
        _CAST_GROUP_FIELDS,
        label="CAST-v4 group receipt",
    )
    digest = _embedded_receipt_digest(row, label="CAST-v4 group receipt")
    schedule = row.get("schedule_coordinate")
    phase_checkpoint = _checkpoint_content_identity_binding(
        checkpoint_content_identity
    )
    group_checkpoint = row.get("checkpoint_content_binding")
    if isinstance(group_checkpoint, Mapping):
        group_checkpoint = _closed(
            group_checkpoint,
            _CAST_CHECKPOINT_BINDING_FIELDS,
            label="CAST-v4 checkpoint content binding",
        )
    if (
        row.get("schema_version") != CAST_V4_GROUP_SCHEMA
        or row.get("root_spec_raw_sha256") != root_spec_sha256
        or row.get("method_source_archive_sha256") != method_archive_sha256
        or row.get("method_source_revision") != method_revision
        or not isinstance(group_checkpoint, Mapping)
        or any(
            group_checkpoint.get(field) != expected
            for field, expected in phase_checkpoint.items()
        )
        or group_checkpoint.get("all_loaded_parameters_frozen") is not True
        or row.get("candidate_count") != CAST_V4_CANDIDATES_PER_GROUP
        or row.get("primary_score_field") != "raw_global_action_energy_score"
        or row.get("phase_conjunctive_role")
        != "diagnostic_only_never_calibration_gate"
        or row.get("training_performed") is not False
        or row.get("optimizer_authorized") is not False
        or row.get("scientific_action_editing_claim") is not False
        or not isinstance(schedule, Mapping)
        or schedule.get("schedule_index") != FIRST_SCHEDULE_INDEX
    ):
        raise PairV7FitOnlyAuthorityError("CAST-v4 group semantic binding differs")
    candidates = _discover_cast_candidate_receipts(
        path, expected_digests=row.get("candidate_receipt_digests", ())
    )
    for candidate in candidates:
        if (
            candidate["root_spec_raw_sha256"] != root_spec_sha256
            or candidate["frozen_checkpoint_receipt_digest"]
            != row.get("frozen_checkpoint_receipt_digest")
            or candidate["checkpoint_content_binding"] != group_checkpoint
        ):
            raise PairV7FitOnlyAuthorityError(
                "CAST-v4 candidate root/checkpoint binding differs"
            )
    return {
        "path": str(path),
        "file_sha256": file_sha,
        "group_id": row.get("group_id"),
        "receipt_digest": digest,
        "frozen_checkpoint_receipt_digest": row.get(
            "frozen_checkpoint_receipt_digest"
        ),
        "checkpoint_content_binding": group_checkpoint,
        "candidate_count": len(candidates),
        "candidate_receipts": candidates,
    }


def _json_path(value: Mapping[str, Any], path: str, *, label: str) -> Any:
    if not isinstance(path, str) or not path or path.startswith(".") or path.endswith("."):
        raise PairV7FitOnlyAuthorityError(f"{label} JSON path differs")
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise PairV7FitOnlyAuthorityError(f"{label} lacks {path}")
        current = current[component]
    return current


def _validate_negative_boundary(
    *,
    boundary_id: str,
    path_value: Any,
    expected_file_sha256: str,
    expected_embedded_digest: str,
    required_booleans: Mapping[str, bool],
) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label=f"{boundary_id} boundary")
    file_sha = _sha256(expected_file_sha256, label=f"{boundary_id} boundary")
    if _file_sha256(path) != file_sha:
        raise PairV7FitOnlyAuthorityError(f"{boundary_id} boundary file SHA differs")
    row = _strict_json(path, label=f"{boundary_id} boundary")
    observed: dict[str, bool] = {}
    for json_path, expected in required_booleans.items():
        value = _json_path(row, json_path, label=f"{boundary_id} boundary")
        if type(value) is not bool or value is not expected:
            raise PairV7FitOnlyAuthorityError(
                f"{boundary_id} boundary {json_path} differs"
            )
        observed[json_path] = value
    embedded_digest = None
    for field in ("receipt_digest", "evidence_digest", "master_receipt_digest"):
        value = row.get(field)
        if isinstance(value, str) and _SHA256_RE.fullmatch(value):
            embedded_digest = value
            break
    expected_digest = _sha256(
        expected_embedded_digest, label=f"{boundary_id} embedded digest"
    )
    if embedded_digest is None or embedded_digest != expected_digest:
        raise PairV7FitOnlyAuthorityError(
            f"{boundary_id} boundary embedded digest differs"
        )
    return {
        "boundary_id": boundary_id,
        "path": str(path),
        "file_sha256": file_sha,
        "embedded_digest": embedded_digest,
        "required_boolean_observations": observed,
        "inherited_as_population_or_update_authority": False,
    }


def _validate_sail_prior_no_success(path_value: Any) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label="SAIL prior master receipt")
    if _file_sha256(path) != SAIL_PRIOR_NO_SUCCESS_FILE_SHA256:
        raise PairV7FitOnlyAuthorityError("SAIL prior master file SHA differs")
    row = _strict_json(path, label="SAIL prior master receipt")
    if row.get("receipt_digest") != SAIL_PRIOR_NO_SUCCESS_RECEIPT_DIGEST:
        raise PairV7FitOnlyAuthorityError("SAIL prior master receipt digest differs")
    _embedded_receipt_digest(row, label="SAIL prior master receipt")
    required_booleans = {
        "postflight_complete": True,
        "all_six_mp4_exact81": True,
        "scientific_claim_authorized": False,
        "action_editing_success_claim_authorized": False,
        "training_performed": False,
        "source_condition_in_live_query": False,
    }
    observed: dict[str, bool] = {}
    for json_path, expected in required_booleans.items():
        value = _json_path(row, json_path, label="SAIL prior master receipt")
        if type(value) is not bool or value is not expected:
            raise PairV7FitOnlyAuthorityError(
                f"SAIL prior master {json_path} differs"
            )
        observed[json_path] = value
    children = row.get("children")
    if not isinstance(children, Mapping) or set(children) != set(SAIL_CHILD_BINDINGS):
        raise PairV7FitOnlyAuthorityError("SAIL child receipt closure differs")
    child_receipts: list[Mapping[str, Any]] = []
    child_fields = {
        "candidate_id",
        "receipt_digest",
        "receipt_file_sha256",
        "receipt_path",
    }
    for role in ("dog", "human"):
        descriptor = children[role]
        expected = SAIL_CHILD_BINDINGS[role]
        if not isinstance(descriptor, Mapping) or set(descriptor) != child_fields:
            raise PairV7FitOnlyAuthorityError(f"SAIL {role} child descriptor differs")
        child_path = _plain_absolute_file(
            descriptor.get("receipt_path"), label=f"SAIL {role} child receipt"
        )
        child_file_sha = _sha256(
            descriptor.get("receipt_file_sha256"), label=f"SAIL {role} child"
        )
        child_digest = _sha256(
            descriptor.get("receipt_digest"), label=f"SAIL {role} child receipt"
        )
        if (
            child_file_sha != expected["file_sha256"]
            or child_digest != expected["receipt_digest"]
            or _file_sha256(child_path) != child_file_sha
        ):
            raise PairV7FitOnlyAuthorityError(f"SAIL {role} child bytes differ")
        child = _strict_json(child_path, label=f"SAIL {role} child receipt")
        if (
            child.get("receipt_digest") != child_digest
            or _embedded_receipt_digest(
                child, label=f"SAIL {role} child receipt"
            )
            != child_digest
            or child.get("candidate_id") != descriptor.get("candidate_id")
            or child.get("mechanism_probe_only") is not True
            or child.get("editor_parameter_or_update_authorized") is not False
            or child.get("scientific_claim_authorized") is not False
            or child.get("action_editing_success_claim_authorized") is not False
            or child.get("training_performed") is not False
            or child.get("source_condition_in_live_query") is not False
        ):
            raise PairV7FitOnlyAuthorityError(
                f"SAIL {role} child no-success scope differs"
            )
        live_vjp = child.get("live_vjp_proof")
        if (
            not isinstance(live_vjp, Mapping)
            or live_vjp.get("real_sp4_autograd_collective_observed") is not True
            or live_vjp.get("replica_consensus_observed") is not True
            or live_vjp.get("same_x_sigma_object_for_action_noop") is not True
        ):
            raise PairV7FitOnlyAuthorityError(
                f"SAIL {role} child measured-VJP proof differs"
            )
        child_receipts.append(
            {
                "role": role,
                "candidate_id": descriptor.get("candidate_id"),
                "path": str(child_path),
                "file_sha256": child_file_sha,
                "receipt_digest": child_digest,
                "mechanism_probe_only": True,
                "editor_parameter_or_update_authorized": False,
                "training_performed": False,
                "scientific_claim_authorized": False,
                "action_editing_success_claim_authorized": False,
                "source_condition_in_live_query": False,
                "measured_vjp_observations": {
                    "real_sp4_autograd_collective_observed": True,
                    "replica_consensus_observed": True,
                    "same_x_sigma_object_for_action_noop": True,
                },
            }
        )
    return {
        "boundary_id": "sail_prior_frozen_intervention_no_success",
        "path": str(path),
        "file_sha256": SAIL_PRIOR_NO_SUCCESS_FILE_SHA256,
        "embedded_digest": SAIL_PRIOR_NO_SUCCESS_RECEIPT_DIGEST,
        "required_boolean_observations": observed,
        "child_receipts": child_receipts,
        "inherited_as_population_or_update_authority": False,
    }


def validate_prompt_bank(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(BRANCH_ORDER):
        raise PairV7FitOnlyAuthorityError("prompt branch closure differs")
    result: dict[str, str] = {}
    seen: set[str] = set()
    for branch in BRANCH_ORDER:
        prompt = value[branch]
        if (
            not isinstance(prompt, str)
            or not prompt
            or prompt != prompt.strip()
            or "\x00" in prompt
        ):
            raise PairV7FitOnlyAuthorityError(
                f"prompt for {branch} must be canonical non-empty text"
            )
        try:
            prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise PairV7FitOnlyAuthorityError(
                f"prompt for {branch} is invalid UTF-8"
            ) from error
        if prompt in seen:
            raise PairV7FitOnlyAuthorityError("prompt branches must be distinct")
        seen.add(prompt)
        result[branch] = prompt
    return result


def _tensor_sha256(value: Any) -> str:
    """Hash a torch tensor identically to the lower-level CAGD helper."""

    try:
        import torch
    except ModuleNotFoundError as error:
        raise PairV7FitOnlyAuthorityError("torch is required to inspect tensors") from error
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise PairV7FitOnlyAuthorityError("tensor hash requires a materialized tensor")
    cpu = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "layout": str(cpu.layout),
    }
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorInspection:
    tensor_sha256: str
    shape: tuple[int, ...]


def _inspect_tensor_artifact(path: Path, key: str, *, label: str) -> TensorInspection:
    if not isinstance(key, str) or not key or "\x00" in key:
        raise PairV7FitOnlyAuthorityError(f"{label} tensor key differs")
    try:
        import torch
        from safetensors import safe_open
    except ModuleNotFoundError as error:
        raise PairV7FitOnlyAuthorityError(
            "torch and safetensors are required to inspect fit-only artifacts"
        ) from error
    try:
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            keys = tuple(opened.keys())
            if keys != (key,):
                raise PairV7FitOnlyAuthorityError(
                    f"{label} tensor-key closure differs"
                )
            tensor = opened.get_tensor(key).contiguous()
    except PairV7FitOnlyAuthorityError:
        raise
    except Exception as error:
        raise PairV7FitOnlyAuthorityError(f"{label} cannot be opened safely") from error
    if (
        tensor.dtype != torch.float32
        or tensor.ndim != 5
        or tuple(int(item) for item in tensor.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(tensor.shape[3]) <= 0
        or int(tensor.shape[4]) <= 0
        or int(tensor.shape[3]) % 2
        or int(tensor.shape[4]) % 2
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise PairV7FitOnlyAuthorityError(
            f"{label} must be detached finite FP32 exact81 [1,16,21,H,W]"
        )
    return TensorInspection(
        tensor_sha256=_tensor_sha256(tensor),
        shape=tuple(int(item) for item in tensor.shape),
    )


def _load_tensor_artifact(path: Path, key: str) -> Any:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError:
        return None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            return opened.get_tensor(key).float().contiguous()
    except Exception as error:
        raise PairV7FitOnlyAuthorityError(
            f"sealed tensor artifact cannot be loaded: {path}"
        ) from error


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    def assert_unchanged(self) -> None:
        if (
            not self.path.is_file()
            or self.path.is_symlink()
            or _file_sha256(self.path) != self.sha256
        ):
            raise PairV7FitOnlyAuthorityError(f"bound file changed: {self.path}")


@dataclass(frozen=True)
class FitOnlyEventSpec:
    event_id: str
    fit_candidate_id: str
    action_family: str
    prompt_by_branch: Mapping[str, str]
    prompt_bank_sha256: str
    source_sample_id: str
    source_video: FileBinding
    raw_caption_by_branch: Mapping[str, str]
    raw_caption_bank_sha256: str
    clean_latent: FileBinding
    clean_latent_tensor_key: str
    clean_latent_tensor_sha256: str
    official_gaussian: FileBinding
    official_gaussian_tensor_key: str
    official_gaussian_tensor_sha256: str
    latent_shape: tuple[int, ...]
    event_digest: str

    def assert_unchanged(self) -> None:
        self.source_video.assert_unchanged()
        self.clean_latent.assert_unchanged()
        self.official_gaussian.assert_unchanged()


@dataclass(frozen=True)
class FitOnlyEventRuntime:
    spec: FitOnlyEventSpec
    event_latent_cpu: Any
    official_epsilon_cpu: Any


@dataclass(frozen=True)
class FitOnlyManifest:
    path: Path
    raw_sha256: str
    checkpoint_tree_sha256: str
    action_adapter_schema_sha256: str
    events: tuple[FitOnlyEventSpec, FitOnlyEventSpec]
    manifest_digest: str

    def assert_unchanged(self) -> None:
        if _file_sha256(self.path) != self.raw_sha256:
            raise PairV7FitOnlyAuthorityError("fit-only manifest changed")
        for event in self.events:
            event.assert_unchanged()


@dataclass(frozen=True)
class FitOnlyGeometryAuthority:
    evidence_file: FileBinding
    evidence_digest: str
    checkpoint_content_receipt_digest: str
    external_evidence_files: tuple[FileBinding, ...]
    validation_receipt: Mapping[str, Any]

    @property
    def authorization_digest(self) -> str:
        """Compatibility name for geometry provenance, never optimizer GO."""

        return self.validation_receipt["receipt_digest"]

    def assert_unchanged(self) -> None:
        self.evidence_file.assert_unchanged()
        for binding in self.external_evidence_files:
            binding.assert_unchanged()


def _validate_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PairV7FitOnlyAuthorityError(f"{label} is unsafe")
    return value


def _event_from_draft(raw: Any, *, ordinal: int) -> dict[str, Any]:
    draft = _closed(raw, _DRAFT_EVENT_FIELDS, label=f"event draft[{ordinal}]")
    event_id = _validate_id(draft["event_id"], label="event ID")
    fit_candidate_id = _validate_id(
        draft["fit_candidate_id"], label="fit candidate ID"
    )
    if fit_candidate_id != event_id:
        raise PairV7FitOnlyAuthorityError(
            "event ID must equal the selected CAST-v4 fit candidate ID"
        )
    family = _validate_id(draft["action_family"], label="action family")
    source_sample_id = _validate_id(
        draft["source_sample_id"], label="source sample ID"
    )
    prompts = validate_prompt_bank(draft["prompt_by_branch"])
    raw_captions = validate_prompt_bank(draft["raw_caption_by_branch"])
    source = _plain_absolute_file(draft["source_video_path"], label="source video")
    source_media = _inspect_source_media(source)
    clean = _plain_absolute_file(draft["clean_latent_path"], label="clean latent")
    noise = _plain_absolute_file(
        draft["official_gaussian_path"], label="official Gaussian"
    )
    clean_inspection = _inspect_tensor_artifact(
        clean, draft["clean_latent_tensor_key"], label="clean latent"
    )
    noise_inspection = _inspect_tensor_artifact(
        noise, draft["official_gaussian_tensor_key"], label="official Gaussian"
    )
    if clean_inspection.shape != noise_inspection.shape:
        raise PairV7FitOnlyAuthorityError("clean latent/Gaussian geometry differs")
    unsigned = {
        "schema_version": EVENT_SCHEMA,
        "event_id": event_id,
        "fit_candidate_id": fit_candidate_id,
        "action_family": family,
        "analysis_split": "fit",
        "prompt_by_branch": prompts,
        "prompt_bank_sha256": object_sha256(prompts),
        "source_sample_id": source_sample_id,
        "source_video_path": str(source),
        "source_video_sha256": _file_sha256(source),
        "source_frame_count": source_media["frame_count"],
        "source_fps": source_media["fps"],
        "source_reference_indices": list(REFERENCE_INDICES),
        "raw_caption_by_branch": raw_captions,
        "raw_caption_bank_sha256": object_sha256(raw_captions),
        "clean_latent_path": str(clean),
        "clean_latent_file_sha256": _file_sha256(clean),
        "clean_latent_tensor_key": draft["clean_latent_tensor_key"],
        "clean_latent_tensor_sha256": clean_inspection.tensor_sha256,
        "official_gaussian_path": str(noise),
        "official_gaussian_file_sha256": _file_sha256(noise),
        "official_gaussian_tensor_key": draft["official_gaussian_tensor_key"],
        "official_gaussian_tensor_sha256": noise_inspection.tensor_sha256,
        "latent_shape": list(clean_inspection.shape),
        "frame_count": FRAME_COUNT,
        "pure_t2v_visual_role": "same_coordinate_frozen_field_query_only",
        "rv2v_target_input_noise_or_donor": False,
    }
    return _seal(unsigned, digest_field="event_digest")


def author_fit_only_manifest(
    *,
    output_path: str | Path,
    checkpoint_tree_sha256: str,
    action_adapter_schema_sha256: str,
    event_drafts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Author exactly one create-only DP2 fit manifest."""

    checkpoint = _sha256(checkpoint_tree_sha256, label="checkpoint tree")
    adapter = _sha256(action_adapter_schema_sha256, label="Action-LoRA schema")
    if not isinstance(event_drafts, Sequence) or len(event_drafts) != DP_SIZE:
        raise PairV7FitOnlyAuthorityError("manifest requires exactly two fit events")
    events = [_event_from_draft(raw, ordinal=i) for i, raw in enumerate(event_drafts)]
    if (
        len({row["event_id"] for row in events}) != DP_SIZE
        or len({row["action_family"] for row in events}) != DP_SIZE
        or len({row["prompt_bank_sha256"] for row in events}) != DP_SIZE
        or len({row["source_sample_id"] for row in events}) != DP_SIZE
        or len({row["source_video_sha256"] for row in events}) != DP_SIZE
    ):
        raise PairV7FitOnlyAuthorityError(
            "DP2 events require distinct fit IDs/families/prompts and correct sources"
        )
    unsigned = {
        "schema_version": MANIFEST_SCHEMA,
        "authority_scope": "fit_only_read_only_gradient_geometry",
        "checkpoint_tree_sha256": checkpoint,
        "action_adapter_schema_sha256": adapter,
        "fit_event_count": DP_SIZE,
        "events": events,
        "schedule_indices": [FIRST_SCHEDULE_INDEX],
        "first_schedule_index": FIRST_SCHEDULE_INDEX,
        "confirmation_population_consumed": False,
    }
    manifest = _seal(unsigned, digest_field="manifest_digest")
    _write_create_only_json(output_path, manifest)
    return manifest


def load_fit_only_manifest(
    path_value: str | Path,
    *,
    expected_file_sha256: str,
    expected_checkpoint_tree_sha256: str,
    expected_action_adapter_schema_sha256: str,
) -> tuple[FitOnlyManifest, tuple[FitOnlyEventRuntime, FitOnlyEventRuntime]]:
    path = _plain_absolute_file(path_value, label="fit-only manifest")
    expected_file = _sha256(expected_file_sha256, label="manifest file")
    if _file_sha256(path) != expected_file:
        raise PairV7FitOnlyAuthorityError("fit-only manifest file SHA-256 differs")
    root = _closed(
        _strict_json(path, label="fit-only manifest"),
        _MANIFEST_FIELDS,
        label="fit-only manifest",
    )
    unsigned_root = dict(root)
    declared = _sha256(
        unsigned_root.pop("manifest_digest"), label="manifest digest"
    )
    if object_sha256(unsigned_root) != declared:
        raise PairV7FitOnlyAuthorityError("fit-only manifest digest differs")
    for field, expected in _NO_UPDATE_CLAIMS.items():
        if root.get(field) is not expected:
            raise PairV7FitOnlyAuthorityError(f"manifest {field} must remain false")
    checkpoint = _sha256(
        expected_checkpoint_tree_sha256, label="expected checkpoint tree"
    )
    adapter = _sha256(
        expected_action_adapter_schema_sha256, label="expected Action-LoRA schema"
    )
    if (
        root.get("schema_version") != MANIFEST_SCHEMA
        or root.get("authority_scope") != "fit_only_read_only_gradient_geometry"
        or root.get("checkpoint_tree_sha256") != checkpoint
        or root.get("action_adapter_schema_sha256") != adapter
        or root.get("fit_event_count") != DP_SIZE
        or root.get("schedule_indices") != [FIRST_SCHEDULE_INDEX]
        or root.get("first_schedule_index") != FIRST_SCHEDULE_INDEX
        or root.get("confirmation_population_consumed") is not False
    ):
        raise PairV7FitOnlyAuthorityError("fit-only manifest authority contract differs")
    rows = root.get("events")
    if not isinstance(rows, list) or len(rows) != DP_SIZE:
        raise PairV7FitOnlyAuthorityError("fit-only manifest event closure differs")
    specs: list[FitOnlyEventSpec] = []
    runtimes: list[FitOnlyEventRuntime] = []
    for ordinal, raw in enumerate(rows):
        row = _closed(raw, _EVENT_FIELDS, label=f"fit event[{ordinal}]")
        unsigned_event = dict(row)
        event_digest = _sha256(
            unsigned_event.pop("event_digest"), label="event digest"
        )
        if object_sha256(unsigned_event) != event_digest:
            raise PairV7FitOnlyAuthorityError(f"fit event[{ordinal}] digest differs")
        for field, expected in _NO_UPDATE_CLAIMS.items():
            if row.get(field) is not expected:
                raise PairV7FitOnlyAuthorityError(
                    f"fit event[{ordinal}] {field} must remain false"
                )
        event_id = _validate_id(row.get("event_id"), label="event ID")
        fit_candidate_id = _validate_id(
            row.get("fit_candidate_id"), label="fit candidate ID"
        )
        family = _validate_id(row.get("action_family"), label="action family")
        source_sample_id = _validate_id(
            row.get("source_sample_id"), label="source sample ID"
        )
        prompts = validate_prompt_bank(row.get("prompt_by_branch"))
        raw_captions = validate_prompt_bank(row.get("raw_caption_by_branch"))
        prompt_digest = _sha256(row.get("prompt_bank_sha256"), label="prompt bank")
        raw_caption_digest = _sha256(
            row.get("raw_caption_bank_sha256"), label="raw caption bank"
        )
        if (
            object_sha256(prompts) != prompt_digest
            or object_sha256(raw_captions) != raw_caption_digest
        ):
            raise PairV7FitOnlyAuthorityError("fit event prompt/caption digest differs")
        shape = row.get("latent_shape")
        source_fps = row.get("source_fps")
        if (
            row.get("schema_version") != EVENT_SCHEMA
            or fit_candidate_id != event_id
            or row.get("analysis_split") != "fit"
            or row.get("frame_count") != FRAME_COUNT
            or row.get("pure_t2v_visual_role")
            != "same_coordinate_frozen_field_query_only"
            or row.get("rv2v_target_input_noise_or_donor") is not False
            or row.get("source_frame_count") != FRAME_COUNT
            or isinstance(source_fps, bool)
            or not isinstance(source_fps, (int, float))
            or float(source_fps) != FPS
            or row.get("source_reference_indices") != list(REFERENCE_INDICES)
            or not isinstance(shape, list)
            or len(shape) != 5
            or any(type(item) is not int for item in shape)
            or tuple(shape[:3]) != (1, LATENT_CHANNELS, LATENT_PHASES)
        ):
            raise PairV7FitOnlyAuthorityError("fit event geometry/scope differs")
        source = _plain_absolute_file(
            row.get("source_video_path"), label="correct source video"
        )
        source_media = _inspect_source_media(source)
        source_file_sha = _sha256(
            row.get("source_video_sha256"), label="correct source video"
        )
        if (
            _file_sha256(source) != source_file_sha
            or source_media["frame_count"] != row.get("source_frame_count")
            or source_media["fps"] != float(source_fps)
        ):
            raise PairV7FitOnlyAuthorityError("correct source video changed")
        clean = _plain_absolute_file(row.get("clean_latent_path"), label="clean latent")
        noise = _plain_absolute_file(
            row.get("official_gaussian_path"), label="official Gaussian"
        )
        clean_file_sha = _sha256(
            row.get("clean_latent_file_sha256"), label="clean latent file"
        )
        noise_file_sha = _sha256(
            row.get("official_gaussian_file_sha256"), label="Gaussian file"
        )
        if _file_sha256(clean) != clean_file_sha or _file_sha256(noise) != noise_file_sha:
            raise PairV7FitOnlyAuthorityError("fit event artifact file changed")
        clean_inspection = _inspect_tensor_artifact(
            clean, row.get("clean_latent_tensor_key"), label="clean latent"
        )
        noise_inspection = _inspect_tensor_artifact(
            noise, row.get("official_gaussian_tensor_key"), label="official Gaussian"
        )
        if (
            clean_inspection.shape != tuple(shape)
            or noise_inspection.shape != tuple(shape)
            or clean_inspection.tensor_sha256
            != _sha256(row.get("clean_latent_tensor_sha256"), label="clean tensor")
            or noise_inspection.tensor_sha256
            != _sha256(row.get("official_gaussian_tensor_sha256"), label="Gaussian tensor")
        ):
            raise PairV7FitOnlyAuthorityError("fit event tensor binding differs")
        spec = FitOnlyEventSpec(
            event_id=event_id,
            fit_candidate_id=fit_candidate_id,
            action_family=family,
            prompt_by_branch=prompts,
            prompt_bank_sha256=prompt_digest,
            source_sample_id=source_sample_id,
            source_video=FileBinding(source, source_file_sha),
            raw_caption_by_branch=raw_captions,
            raw_caption_bank_sha256=raw_caption_digest,
            clean_latent=FileBinding(clean, clean_file_sha),
            clean_latent_tensor_key=row["clean_latent_tensor_key"],
            clean_latent_tensor_sha256=row["clean_latent_tensor_sha256"],
            official_gaussian=FileBinding(noise, noise_file_sha),
            official_gaussian_tensor_key=row["official_gaussian_tensor_key"],
            official_gaussian_tensor_sha256=row["official_gaussian_tensor_sha256"],
            latent_shape=tuple(shape),
            event_digest=event_digest,
        )
        # Re-open here to return authenticated CPU tensors.  The file hash and
        # tensor digest were checked immediately above.
        clean_tensor = _load_tensor_artifact(clean, spec.clean_latent_tensor_key)
        noise_tensor = _load_tensor_artifact(noise, spec.official_gaussian_tensor_key)
        specs.append(spec)
        runtimes.append(FitOnlyEventRuntime(spec, clean_tensor, noise_tensor))
    if (
        len({item.event_id for item in specs}) != DP_SIZE
        or len({item.action_family for item in specs}) != DP_SIZE
        or len({item.prompt_bank_sha256 for item in specs}) != DP_SIZE
        or len({item.source_sample_id for item in specs}) != DP_SIZE
        or len({item.source_video.sha256 for item in specs}) != DP_SIZE
    ):
        raise PairV7FitOnlyAuthorityError(
            "DP2 events require distinct fit IDs/families/prompts and correct sources"
        )
    manifest = FitOnlyManifest(
        path=path,
        raw_sha256=expected_file,
        checkpoint_tree_sha256=checkpoint,
        action_adapter_schema_sha256=adapter,
        events=(specs[0], specs[1]),
        manifest_digest=declared,
    )
    return manifest, (runtimes[0], runtimes[1])


def _collect_cast_v4_bindings(
    *,
    manifest: FitOnlyManifest,
    checkpoint_content_identity: Mapping[str, Any],
    checkpoint_content_receipt_digest: str,
    cast_method_archive_path: str | Path,
    expected_cast_method_archive_sha256: str,
    expected_cast_method_revision: str,
    cast_root_spec_path: str | Path,
    expected_cast_root_spec_sha256: str,
    cast_group_receipt_paths: Sequence[str | Path],
    expected_cast_group_receipt_sha256: Sequence[str],
    legacy_v3_no_go_path: str | Path,
    sail_prior_no_success_path: str | Path,
) -> Mapping[str, Any]:
    checkpoint_content = _sha256(
        checkpoint_content_receipt_digest, label="checkpoint content receipt"
    )
    if object_sha256(checkpoint_content_identity) != checkpoint_content:
        raise PairV7FitOnlyAuthorityError(
            "checkpoint content identity differs from its receipt digest"
        )
    _checkpoint_content_identity_binding(checkpoint_content_identity)
    method = _validate_cast_method_archive(
        cast_method_archive_path,
        expected_sha256=expected_cast_method_archive_sha256,
        expected_revision=expected_cast_method_revision,
    )
    root_spec = _validate_cast_root_spec(
        cast_root_spec_path, expected_sha256=expected_cast_root_spec_sha256
    )
    if (
        not isinstance(cast_group_receipt_paths, Sequence)
        or not isinstance(expected_cast_group_receipt_sha256, Sequence)
        or len(cast_group_receipt_paths) != DP_SIZE
        or len(expected_cast_group_receipt_sha256) != DP_SIZE
    ):
        raise PairV7FitOnlyAuthorityError("CAST-v4 requires exactly A/B group receipts")
    groups = [
        _validate_cast_group(
            path,
            expected_file_sha256=file_sha,
            root_spec_sha256=root_spec["file_sha256"],
            method_archive_sha256=method["file_sha256"],
            method_revision=method["git_archive_revision"],
            checkpoint_content_identity=checkpoint_content_identity,
        )
        for path, file_sha in zip(
            cast_group_receipt_paths, expected_cast_group_receipt_sha256
        )
    ]
    if len({group["group_id"] for group in groups}) != DP_SIZE:
        raise PairV7FitOnlyAuthorityError("CAST-v4 A/B group IDs must be distinct")
    if len(
        {group["frozen_checkpoint_receipt_digest"] for group in groups}
    ) != 1:
        raise PairV7FitOnlyAuthorityError(
            "CAST-v4 A/B frozen scorer checkpoint digests differ"
        )
    candidates = [
        candidate
        for group in groups
        for candidate in group["candidate_receipts"]
    ]
    if (
        len(candidates) != CAST_V4_TOTAL_CANDIDATES
        or len({row["receipt_digest"] for row in candidates})
        != CAST_V4_TOTAL_CANDIDATES
        or len({row["candidate_id"] for row in candidates})
        != CAST_V4_TOTAL_CANDIDATES
    ):
        raise PairV7FitOnlyAuthorityError(
            "CAST-v4 A/B closure must contain 40 unique candidate receipts"
        )
    selected: dict[str, Mapping[str, Any]] = {}
    for event in manifest.events:
        matches = [row for row in candidates if row["candidate_id"] == event.event_id]
        if len(matches) != 1:
            raise PairV7FitOnlyAuthorityError(
                f"fit event {event.event_id} lacks one unique CAST-v4 score receipt"
            )
        row = matches[0]
        if (
            row["analysis_split"] != "fit"
            or row["semantic_branch"] != "action"
            or row["action_family_id"] != event.action_family
            or row["clean_latent_tensor_sha256"]
            != event.clean_latent_tensor_sha256
            or row["official_gaussian_tensor_sha256"]
            != event.official_gaussian_tensor_sha256
            or row["prompt_by_branch"] != dict(event.prompt_by_branch)
            or row["full_t2v_caption_by_branch"]
            != dict(event.raw_caption_by_branch)
            or row["geometry_source_video_sha256"] != event.source_video.sha256
            or row["candidate_shape"] != list(event.latent_shape)
        ):
            raise PairV7FitOnlyAuthorityError(
                f"fit event {event.event_id} CAST-v4 action-score binding differs"
            )
        selected[event.event_id] = {
            key: row[key]
            for key in (
                "candidate_id",
                "path",
                "file_sha256",
                "receipt_digest",
                "analysis_split",
                "action_family_id",
                "semantic_branch",
                "raw_global_action_energy_score",
            )
        }
    legacy_boundary = _validate_negative_boundary(
        boundary_id="d541801_v3_confirmation_no_optimizer_go",
        path_value=legacy_v3_no_go_path,
        expected_file_sha256=LEGACY_V3_NO_GO_FILE_SHA256,
        expected_embedded_digest=LEGACY_V3_NO_GO_RECEIPT_DIGEST,
        required_booleans={
            LEGACY_V3_OPTIMIZER_FALSE_JSON_PATH: False,
            LEGACY_V3_CONFIRMATION_FALSE_JSON_PATH: False,
        },
    )
    sail_boundary = _validate_sail_prior_no_success(sail_prior_no_success_path)
    return {
        "cast_v4_method_archive": method,
        "cast_v4_root_spec": root_spec,
        "cast_v4_groups": groups,
        "selected_action_score_by_event": selected,
        "negative_boundaries": [legacy_boundary, sail_boundary],
    }


def author_fit_only_evidence(
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    expected_manifest_file_sha256: str,
    checkpoint_tree_sha256: str,
    checkpoint_content_identity: Mapping[str, Any],
    checkpoint_content_receipt_digest: str,
    action_adapter_schema_sha256: str,
    cast_method_archive_path: str | Path,
    expected_cast_method_archive_sha256: str,
    expected_cast_method_revision: str,
    cast_root_spec_path: str | Path,
    expected_cast_root_spec_sha256: str,
    cast_group_receipt_paths: Sequence[str | Path],
    expected_cast_group_receipt_sha256: Sequence[str],
    legacy_v3_no_go_path: str | Path,
    sail_prior_no_success_path: str | Path,
) -> Mapping[str, Any]:
    manifest, _ = load_fit_only_manifest(
        manifest_path,
        expected_file_sha256=expected_manifest_file_sha256,
        expected_checkpoint_tree_sha256=checkpoint_tree_sha256,
        expected_action_adapter_schema_sha256=action_adapter_schema_sha256,
    )
    checkpoint_content = _sha256(
        checkpoint_content_receipt_digest, label="checkpoint content receipt"
    )
    cast_bindings = _collect_cast_v4_bindings(
        manifest=manifest,
        checkpoint_content_identity=checkpoint_content_identity,
        checkpoint_content_receipt_digest=checkpoint_content,
        cast_method_archive_path=cast_method_archive_path,
        expected_cast_method_archive_sha256=expected_cast_method_archive_sha256,
        expected_cast_method_revision=expected_cast_method_revision,
        cast_root_spec_path=cast_root_spec_path,
        expected_cast_root_spec_sha256=expected_cast_root_spec_sha256,
        cast_group_receipt_paths=cast_group_receipt_paths,
        expected_cast_group_receipt_sha256=expected_cast_group_receipt_sha256,
        legacy_v3_no_go_path=legacy_v3_no_go_path,
        sail_prior_no_success_path=sail_prior_no_success_path,
    )
    unsigned = {
        "schema_version": EVIDENCE_SCHEMA,
        "method_name": METHOD_NAME,
        "authority_scope": "fit_only_read_only_gradient_geometry",
        "geometry_measurement_authorized": True,
        "manifest_path": str(manifest.path),
        "manifest_file_sha256": manifest.raw_sha256,
        "manifest_digest": manifest.manifest_digest,
        "checkpoint_tree_sha256": manifest.checkpoint_tree_sha256,
        "checkpoint_content_receipt_digest": checkpoint_content,
        "checkpoint_content_identity_binding": (
            _checkpoint_content_identity_binding(checkpoint_content_identity)
        ),
        "action_adapter_schema_sha256": manifest.action_adapter_schema_sha256,
        "fit_event_count": DP_SIZE,
        "fit_event_ids": [event.event_id for event in manifest.events],
        "fit_action_families": [event.action_family for event in manifest.events],
        "fit_event_digests": [event.event_digest for event in manifest.events],
        "schedule_indices": [FIRST_SCHEDULE_INDEX],
        "first_schedule_index": FIRST_SCHEDULE_INDEX,
        "create_only_authoring": True,
        "confirmation_population_consumed": False,
        "population_scorer_receipts_consumed": True,
        "population_scorer_receipts_role": (
            "authenticate_selected_fit_action_scores_only_never_population_go"
        ),
        "legacy_optimizer_authority_consumed": False,
        **cast_bindings,
    }
    evidence = _seal(unsigned, digest_field="evidence_digest")
    _write_create_only_json(output_path, evidence)
    return evidence


def _external_evidence_file_bindings(
    cast_bindings: Mapping[str, Any],
    checkpoint_content_identity: Mapping[str, Any],
) -> tuple[FileBinding, ...]:
    """Materialize the complete preflight file closure for TOCTOU postflight."""

    descriptors: list[Mapping[str, Any]] = [
        {
            "path": checkpoint_content_identity.get("manifest_path"),
            "file_sha256": checkpoint_content_identity.get(
                "manifest_sha256_computed"
            ),
        },
        cast_bindings["cast_v4_method_archive"],
        cast_bindings["cast_v4_root_spec"],
        *cast_bindings["cast_v4_groups"],
        *cast_bindings["negative_boundaries"],
    ]
    for group in cast_bindings["cast_v4_groups"]:
        descriptors.extend(group["candidate_receipts"])
    for boundary in cast_bindings["negative_boundaries"]:
        child_receipts = boundary.get("child_receipts", ())
        if not isinstance(child_receipts, Sequence):
            raise PairV7FitOnlyAuthorityError(
                "negative-boundary child file closure differs"
            )
        descriptors.extend(child_receipts)
    bindings: list[FileBinding] = []
    seen: set[Path] = set()
    for ordinal, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise PairV7FitOnlyAuthorityError(
                f"external evidence descriptor[{ordinal}] differs"
            )
        path = _plain_absolute_file(
            descriptor.get("path"), label=f"external evidence[{ordinal}]"
        )
        digest = _sha256(
            descriptor.get("file_sha256"),
            label=f"external evidence[{ordinal}]",
        )
        if path in seen or _file_sha256(path) != digest:
            raise PairV7FitOnlyAuthorityError(
                "external evidence file closure differs"
            )
        seen.add(path)
        bindings.append(FileBinding(path, digest))
    return tuple(sorted(bindings, key=lambda item: str(item.path)))


def validate_fit_only_geometry_authority(
    *,
    manifest_path: str | Path,
    expected_manifest_file_sha256: str,
    evidence_path: str | Path,
    expected_evidence_file_sha256: str,
    expected_checkpoint_tree_sha256: str,
    checkpoint_content_identity: Mapping[str, Any],
    expected_checkpoint_content_receipt_digest: str,
    expected_action_adapter_schema_sha256: str,
    cast_method_archive_path: str | Path,
    expected_cast_method_archive_sha256: str,
    expected_cast_method_revision: str,
    cast_group_receipt_paths: Sequence[str | Path],
    expected_cast_group_receipt_sha256: Sequence[str],
) -> tuple[
    FitOnlyManifest,
    tuple[FitOnlyEventRuntime, FitOnlyEventRuntime],
    FitOnlyGeometryAuthority,
]:
    manifest, runtimes = load_fit_only_manifest(
        manifest_path,
        expected_file_sha256=expected_manifest_file_sha256,
        expected_checkpoint_tree_sha256=expected_checkpoint_tree_sha256,
        expected_action_adapter_schema_sha256=expected_action_adapter_schema_sha256,
    )
    evidence_path_resolved = _plain_absolute_file(evidence_path, label="fit-only evidence")
    evidence_file_sha = _sha256(
        expected_evidence_file_sha256, label="fit-only evidence file"
    )
    if _file_sha256(evidence_path_resolved) != evidence_file_sha:
        raise PairV7FitOnlyAuthorityError("fit-only evidence file SHA-256 differs")
    evidence = _closed(
        _strict_json(evidence_path_resolved, label="fit-only evidence"),
        _EVIDENCE_FIELDS,
        label="fit-only evidence",
    )
    unsigned = dict(evidence)
    evidence_digest = _sha256(
        unsigned.pop("evidence_digest"), label="evidence digest"
    )
    if object_sha256(unsigned) != evidence_digest:
        raise PairV7FitOnlyAuthorityError("fit-only evidence digest differs")
    for field, expected in _NO_UPDATE_CLAIMS.items():
        if evidence.get(field) is not expected:
            raise PairV7FitOnlyAuthorityError(f"evidence {field} must remain false")
    checkpoint_content = _sha256(
        expected_checkpoint_content_receipt_digest,
        label="expected checkpoint content receipt",
    )
    root_binding = evidence.get("cast_v4_root_spec")
    boundaries = evidence.get("negative_boundaries")
    if (
        not isinstance(root_binding, Mapping)
        or set(root_binding) != {"path", "file_sha256"}
        or not isinstance(boundaries, list)
        or len(boundaries) != 2
        or any(not isinstance(item, Mapping) for item in boundaries)
    ):
        raise PairV7FitOnlyAuthorityError(
            "fit-only CAST/root/negative-boundary descriptor closure differs"
        )
    boundary_by_id = {item.get("boundary_id"): item for item in boundaries}
    legacy_boundary = boundary_by_id.get(
        "d541801_v3_confirmation_no_optimizer_go"
    )
    sail_boundary = boundary_by_id.get(
        "sail_prior_frozen_intervention_no_success"
    )
    if not isinstance(legacy_boundary, Mapping) or not isinstance(
        sail_boundary, Mapping
    ):
        raise PairV7FitOnlyAuthorityError("required negative boundaries are absent")
    legacy_observations = legacy_boundary.get("required_boolean_observations")
    sail_observations = sail_boundary.get("required_boolean_observations")
    if (
        legacy_observations
        != {
            LEGACY_V3_OPTIMIZER_FALSE_JSON_PATH: False,
            LEGACY_V3_CONFIRMATION_FALSE_JSON_PATH: False,
        }
        or sail_observations
        != {
            "postflight_complete": True,
            "all_six_mp4_exact81": True,
            "scientific_claim_authorized": False,
            "action_editing_success_claim_authorized": False,
            "training_performed": False,
            "source_condition_in_live_query": False,
        }
    ):
        raise PairV7FitOnlyAuthorityError("negative-boundary observations differ")
    recomputed_cast_bindings = _collect_cast_v4_bindings(
        manifest=manifest,
        checkpoint_content_identity=checkpoint_content_identity,
        checkpoint_content_receipt_digest=checkpoint_content,
        cast_method_archive_path=cast_method_archive_path,
        expected_cast_method_archive_sha256=expected_cast_method_archive_sha256,
        expected_cast_method_revision=expected_cast_method_revision,
        cast_root_spec_path=root_binding["path"],
        expected_cast_root_spec_sha256=root_binding["file_sha256"],
        cast_group_receipt_paths=cast_group_receipt_paths,
        expected_cast_group_receipt_sha256=expected_cast_group_receipt_sha256,
        legacy_v3_no_go_path=legacy_boundary["path"],
        sail_prior_no_success_path=sail_boundary["path"],
    )
    if (
        evidence.get("schema_version") != EVIDENCE_SCHEMA
        or evidence.get("method_name") != METHOD_NAME
        or evidence.get("authority_scope") != "fit_only_read_only_gradient_geometry"
        or evidence.get("geometry_measurement_authorized") is not True
        or evidence.get("manifest_path") != str(manifest.path)
        or evidence.get("manifest_file_sha256") != manifest.raw_sha256
        or evidence.get("manifest_digest") != manifest.manifest_digest
        or evidence.get("checkpoint_tree_sha256") != manifest.checkpoint_tree_sha256
        or evidence.get("checkpoint_content_receipt_digest") != checkpoint_content
        or evidence.get("checkpoint_content_identity_binding")
        != _checkpoint_content_identity_binding(checkpoint_content_identity)
        or evidence.get("action_adapter_schema_sha256")
        != manifest.action_adapter_schema_sha256
        or evidence.get("fit_event_count") != DP_SIZE
        or evidence.get("fit_event_ids")
        != [event.event_id for event in manifest.events]
        or evidence.get("fit_action_families")
        != [event.action_family for event in manifest.events]
        or evidence.get("fit_event_digests")
        != [event.event_digest for event in manifest.events]
        or evidence.get("schedule_indices") != [FIRST_SCHEDULE_INDEX]
        or evidence.get("first_schedule_index") != FIRST_SCHEDULE_INDEX
        or evidence.get("create_only_authoring") is not True
        or evidence.get("confirmation_population_consumed") is not False
        or evidence.get("population_scorer_receipts_consumed") is not True
        or evidence.get("population_scorer_receipts_role")
        != "authenticate_selected_fit_action_scores_only_never_population_go"
        or evidence.get("legacy_optimizer_authority_consumed") is not False
        or evidence.get("cast_v4_method_archive")
        != recomputed_cast_bindings["cast_v4_method_archive"]
        or evidence.get("cast_v4_root_spec")
        != recomputed_cast_bindings["cast_v4_root_spec"]
        or evidence.get("cast_v4_groups")
        != recomputed_cast_bindings["cast_v4_groups"]
        or evidence.get("selected_action_score_by_event")
        != recomputed_cast_bindings["selected_action_score_by_event"]
        or evidence.get("negative_boundaries")
        != recomputed_cast_bindings["negative_boundaries"]
    ):
        raise PairV7FitOnlyAuthorityError("fit-only evidence binding/scope differs")
    external_evidence_files = _external_evidence_file_bindings(
        recomputed_cast_bindings, checkpoint_content_identity
    )
    validation = _seal(
        {
            "schema_version": VALIDATION_SCHEMA,
            "method_name": METHOD_NAME,
            "authority_scope": "fit_only_read_only_gradient_geometry",
            "geometry_measurement_authorized": True,
            "manifest_file_sha256": manifest.raw_sha256,
            "manifest_digest": manifest.manifest_digest,
            "evidence_file_sha256": evidence_file_sha,
            "evidence_digest": evidence_digest,
            "checkpoint_tree_sha256": manifest.checkpoint_tree_sha256,
            "checkpoint_content_receipt_digest": checkpoint_content,
            "action_adapter_schema_sha256": manifest.action_adapter_schema_sha256,
            "fit_event_count": DP_SIZE,
            "fit_event_ids": [event.event_id for event in manifest.events],
            "fit_action_families": [event.action_family for event in manifest.events],
            "schedule_indices": [FIRST_SCHEDULE_INDEX],
            "first_schedule_index": FIRST_SCHEDULE_INDEX,
            "confirmation_population_consumed": False,
            "population_scorer_receipts_consumed": True,
            "population_scorer_receipts_role": (
                "authenticate_selected_fit_action_scores_only_never_population_go"
            ),
            "legacy_optimizer_authority_consumed": False,
            "cast_v4_method_archive": recomputed_cast_bindings[
                "cast_v4_method_archive"
            ],
            "cast_v4_root_spec": recomputed_cast_bindings["cast_v4_root_spec"],
            "cast_v4_group_receipt_digests": [
                row["receipt_digest"]
                for row in recomputed_cast_bindings["cast_v4_groups"]
            ],
            "cast_v4_candidate_receipt_count": CAST_V4_TOTAL_CANDIDATES,
            "selected_action_score_by_event": recomputed_cast_bindings[
                "selected_action_score_by_event"
            ],
            "negative_boundaries": recomputed_cast_bindings[
                "negative_boundaries"
            ],
            "external_evidence_file_count": len(external_evidence_files),
            "all_external_evidence_files_bound_for_postflight": True,
        },
        digest_field="receipt_digest",
    )
    authority = FitOnlyGeometryAuthority(
        evidence_file=FileBinding(evidence_path_resolved, evidence_file_sha),
        evidence_digest=evidence_digest,
        checkpoint_content_receipt_digest=checkpoint_content,
        external_evidence_files=external_evidence_files,
        validation_receipt=validation,
    )
    return manifest, runtimes, authority


def _load_event_draft(path_value: str) -> Mapping[str, Any]:
    path = _plain_absolute_file(path_value, label="event draft")
    return _closed(
        _strict_json(path, label="event draft"),
        _DRAFT_EVENT_FIELDS,
        label="event draft",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("author-manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--checkpoint-tree-sha256", required=True)
    manifest.add_argument("--action-adapter-schema-sha256", required=True)
    manifest.add_argument("--event-draft", action="append", required=True)
    evidence = subparsers.add_parser("author-evidence")
    evidence.add_argument("--output", required=True)
    evidence.add_argument("--manifest", required=True)
    evidence.add_argument("--expected-manifest-file-sha256", required=True)
    evidence.add_argument("--checkpoint-tree-sha256", required=True)
    evidence.add_argument("--checkpoint-content-identity", required=True)
    evidence.add_argument("--checkpoint-content-receipt-digest", required=True)
    evidence.add_argument("--action-adapter-schema-sha256", required=True)
    evidence.add_argument("--cast-method-archive", required=True)
    evidence.add_argument("--expected-cast-method-archive-sha256", required=True)
    evidence.add_argument(
        "--expected-cast-method-revision", default=CAST_V4_METHOD_REVISION
    )
    evidence.add_argument("--cast-root-spec", required=True)
    evidence.add_argument("--expected-cast-root-spec-sha256", required=True)
    evidence.add_argument("--cast-group-receipt", action="append", required=True)
    evidence.add_argument(
        "--expected-cast-group-receipt-sha256", action="append", required=True
    )
    evidence.add_argument("--legacy-v3-no-go", required=True)
    evidence.add_argument("--sail-prior-no-success", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "author-manifest":
        value = author_fit_only_manifest(
            output_path=args.output,
            checkpoint_tree_sha256=args.checkpoint_tree_sha256,
            action_adapter_schema_sha256=args.action_adapter_schema_sha256,
            event_drafts=[_load_event_draft(path) for path in args.event_draft],
        )
        output = Path(args.output)
        print(
            json.dumps(
                {
                    "manifest": str(output),
                    "manifest_file_sha256": _file_sha256(output),
                    "manifest_digest": value["manifest_digest"],
                    **_NO_UPDATE_CLAIMS,
                },
                sort_keys=True,
            )
        )
        return 0
    value = author_fit_only_evidence(
        output_path=args.output,
        manifest_path=args.manifest,
        expected_manifest_file_sha256=args.expected_manifest_file_sha256,
        checkpoint_tree_sha256=args.checkpoint_tree_sha256,
        checkpoint_content_identity=_strict_json(
            _plain_absolute_file(
                args.checkpoint_content_identity,
                label="checkpoint content identity",
            ),
            label="checkpoint content identity",
        ),
        checkpoint_content_receipt_digest=args.checkpoint_content_receipt_digest,
        action_adapter_schema_sha256=args.action_adapter_schema_sha256,
        cast_method_archive_path=args.cast_method_archive,
        expected_cast_method_archive_sha256=(
            args.expected_cast_method_archive_sha256
        ),
        expected_cast_method_revision=args.expected_cast_method_revision,
        cast_root_spec_path=args.cast_root_spec,
        expected_cast_root_spec_sha256=args.expected_cast_root_spec_sha256,
        cast_group_receipt_paths=args.cast_group_receipt,
        expected_cast_group_receipt_sha256=(
            args.expected_cast_group_receipt_sha256
        ),
        legacy_v3_no_go_path=args.legacy_v3_no_go,
        sail_prior_no_success_path=args.sail_prior_no_success,
    )
    output = Path(args.output)
    print(
        json.dumps(
            {
                "evidence": str(output),
                "evidence_file_sha256": _file_sha256(output),
                "evidence_digest": value["evidence_digest"],
                **_NO_UPDATE_CLAIMS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRANCH_ORDER",
    "DP_SIZE",
    "EVIDENCE_SCHEMA",
    "EVENT_SCHEMA",
    "FIRST_SCHEDULE_INDEX",
    "FitOnlyEventRuntime",
    "FitOnlyEventSpec",
    "FitOnlyGeometryAuthority",
    "FitOnlyManifest",
    "MANIFEST_SCHEMA",
    "METHOD_NAME",
    "PairV7FitOnlyAuthorityError",
    "TensorInspection",
    "VALIDATION_SCHEMA",
    "author_fit_only_evidence",
    "author_fit_only_manifest",
    "canonical_json_bytes",
    "load_fit_only_manifest",
    "main",
    "object_sha256",
    "validate_fit_only_geometry_authority",
    "validate_prompt_bank",
]
