#!/usr/bin/env python3
"""Reduced, fail-closed authority and schedule for BOX-EXP-012.

This module reduces only the disposable canary population.  It delegates all
teacher, nuisance, same-state, wrong-control, materialization, and amplitude
numerics to the frozen full30 authority implementations.  It does not contain
an action loss, optimizer, model forward, checkpoint writer, or media author.

The checkpoint ABI remains the canonical 1,280-row schedule.  Only rows
``0..15`` are executable authority; the remaining rows are deterministic,
unauthorised serialization scaffolding and are rejected before any source,
condition, teacher, nuisance, or amplitude payload can be opened.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NoReturn, Sequence

try:
    import full30_action_amplitude_authority_v1 as amplitude_authority
    import full30_action_data_teacher_authority_v1 as data_authority
except ImportError:  # pragma: no cover - package import mode
    from . import full30_action_amplitude_authority_v1 as amplitude_authority
    from . import full30_action_data_teacher_authority_v1 as data_authority


SCHEMA_VERSION = "bernini-full30-action-mechanism-canary-authority-v2"
VALIDATION_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-validation-v1"
)
AMPLITUDE_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-amplitude-authority-v2"
)
AMPLITUDE_VALIDATION_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-amplitude-validation-v1"
)
PARENT_BINDING_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-parent-binding-v1"
)
SCHEDULE_AUTHORITY_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-schedule-authority-v1"
)
MATERIALIZER_PLAN_ADMISSION_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-materializer-plan-admission-v2"
)
MATERIALIZER_PLAN_BINDING_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-materializer-plan-binding-v1"
)
TEACHER_SEED_BINDING_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-teacher-seed-binding-v1"
)
TEACHER_SEED_CANDIDATE_BINDING_SCHEMA_VERSION = (
    "bernini-full30-action-mechanism-canary-teacher-seed-candidate-binding-v2"
)

PAIR_V5_CANDIDATE_ENVELOPE_SCHEMA_VERSION = (
    "pair-v5-frozen-bernini-t2v-calibration-candidate-v1"
)
PAIR_V5_NATIVE_RECEIPT_SCHEMA_VERSION = (
    "bernini-native-identity-generation-canary-v2"
)
PAIR_V5_NATIVE_METHOD = "frozen-bernini-native-identity-generation-canary"

EXPERIMENT_ID = "BOX-EXP-012"
POPULATION_PROFILE = "same_origin_two_seed_mechanism_only_v1"
BRANCHES = ("action", "incomplete")
SIGMA_INDICES = (4, 12, 20, 28, 35, 38)
TEACHER_CELLS = 2
SOURCES_PER_CELL = 4
SOURCE_UNITS = TEACHER_CELLS * SOURCES_PER_CELL
PAIR_ROWS = SOURCE_UNITS * len(BRANCHES)
REPRESENTATION_BUNDLES = TEACHER_CELLS * len(BRANCHES)
AMPLITUDE_BUNDLES = REPRESENTATION_BUNDLES
MAX_UPDATES = 2
GLOBAL_BATCH = 8
EXECUTABLE_FLAT_ROWS = MAX_UPDATES * GLOBAL_BATCH
CHECKPOINT_SOURCES = 64
CHECKPOINT_TEACHER_CELLS = 8
CHECKPOINT_UPDATES = 160
CHECKPOINT_FLAT_ROWS = CHECKPOINT_UPDATES * GLOBAL_BATCH
TAIL_SOURCES = CHECKPOINT_SOURCES - SOURCE_UNITS
TAIL_TEACHER_CELLS = CHECKPOINT_TEACHER_CELLS - TEACHER_CELLS

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TAIL_SOURCE_PREFIX = "unauthorised-canary-tail-source-"
_TAIL_CELL_PREFIX = "unauthorised-canary-tail-cell-"


class Full30MechanismCanaryAuthorityError(RuntimeError):
    """Raised before reduced or tail authority can be confused with formal."""


def fail(message: str) -> NoReturn:
    raise Full30MechanismCanaryAuthorityError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return data_authority.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return data_authority.object_sha256(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as error:
        raise Full30MechanismCanaryAuthorityError(
            f"{label} is unavailable: {error}"
        ) from error
    if resolved != path or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be one canonical plain file")
    return resolved


def _load_manifest(path: str | Path, expected_sha256: str, *, label: str) -> Mapping[str, Any]:
    source = _plain_file(path, label=label)
    expected = _sha(expected_sha256, label=f"{label} expected SHA-256")
    if data_authority.file_sha256(source) != expected:
        fail(f"{label} file SHA-256 differs")
    try:
        return data_authority._load_json(source, expected)
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"{label} is not strict canonical authority JSON: {error}"
        ) from error


def _sealed(value: Any, fields: Iterable[str], digest_field: str, *, label: str) -> Mapping[str, Any]:
    try:
        row = data_authority._closed(value, set(fields), label)
        data_authority._verify_seal(row, digest_field, label)
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(f"{label} differs: {error}") from error
    return row


_AUTHORITY_FIELDS = {
    "status",
    "experiment_id",
    "data_authority_complete",
    "teacher_authority_complete",
    "current_optimizer_pair_rows",
    "current_optimizer_teacher_bundles",
    "maximum_updates",
    "formal_authority",
    "mechanism_only",
    "tail_serialization_only",
    "scientific_success_claimed",
    "optimizer_authorized",
    "population_profile",
    "synthetic_target_bytes_read",
}
_TOP_FIELDS = {
    "schema_version",
    "population_profile",
    "materialization_run_receipt",
    "authority",
    "source_io_policy",
    "teacher_origins",
    "teacher_seed_bindings",
    "sources",
    "pairs",
    "representation_admissions",
    "authority_counts",
    "manifest_digest",
}
_EXPECTED_AUTHORITY = {
    "status": "optimizer_admitted_disposable_canary",
    "experiment_id": EXPERIMENT_ID,
    "data_authority_complete": True,
    "teacher_authority_complete": True,
    "current_optimizer_pair_rows": PAIR_ROWS,
    "current_optimizer_teacher_bundles": REPRESENTATION_BUNDLES,
    "maximum_updates": MAX_UPDATES,
    "formal_authority": False,
    "mechanism_only": True,
    "tail_serialization_only": True,
    "scientific_success_claimed": False,
    "optimizer_authorized": True,
    "population_profile": POPULATION_PROFILE,
    "synthetic_target_bytes_read": False,
}
_EXPECTED_COUNTS = {
    "teacher_cells": TEACHER_CELLS,
    "teacher_seed_bindings": TEACHER_CELLS,
    "distinct_teacher_generation_seeds": TEACHER_CELLS,
    "source_units": SOURCE_UNITS,
    "optimizer_pair_rows": PAIR_ROWS,
    "representation_bundles": REPRESENTATION_BUNDLES,
    "representation_anchor_evidence": REPRESENTATION_BUNDLES * 2,
    "representation_sigma_rows": REPRESENTATION_BUNDLES * len(SIGMA_INDICES),
    "amplitude_bundles": AMPLITUDE_BUNDLES,
    "maximum_updates": MAX_UPDATES,
}


def _validate_teacher_origins(value: Any) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or len(value) != TEACHER_CELLS:
        fail("canary teacher_origins must contain exactly two rows")
    rows: list[Mapping[str, Any]] = []
    cell_ids: set[str] = set()
    shared_origin_identity: tuple[str, ...] | None = None
    for ordinal, item in enumerate(value):
        label = f"teacher_origins[{ordinal}]"
        try:
            row = data_authority._closed(item, data_authority._TEACHER_FIELDS, label)
            data_authority._verify_seal(row, "origin_digest", label)
            data_authority._require(
                row["schema_version"] == data_authority.TEACHER_ORIGIN_SCHEMA,
                f"{label} schema differs",
            )
            data_authority._require(row["analysis_split"] == "fit", f"{label} is not fit")
            cell = data_authority._safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
            event = data_authority._safe_id(row["event_id"], f"{label}.event_id")
            iid = data_authority._iid(row["origin_iid"], f"{label}.origin_iid")
            sha = data_authority._sha(row["origin_source_sha256"], f"{label}.origin_source_sha256")
            data_authority._verify_file(row["origin_source_path"], sha, f"{label}.origin_source")
            group = data_authority._safe_id(row["origin_group_id"], f"{label}.origin_group_id")
            for field in ("actor_kind", "q0_id", "actor_id", "scene_id"):
                data_authority._safe_id(row[field], f"{label}.{field}")
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"{label} formal row validation failed: {error}"
            ) from error
        _require(cell not in cell_ids, "canary teacher cell is duplicated")
        origin_identity = (
            iid,
            str(row["origin_source_path"]),
            sha,
            group,
            event,
            str(row["actor_kind"]),
            str(row["q0_id"]),
            str(row["actor_id"]),
            str(row["scene_id"]),
        )
        if shared_origin_identity is None:
            shared_origin_identity = origin_identity
        else:
            _require(
                origin_identity == shared_origin_identity,
                "same_origin_two_seed_mechanism_only_v1 requires exact shared "
                "origin IID/path/SHA/group/event/actor/q0 identity",
            )
        cell_ids.add(cell)
        rows.append(row)
    _require(
        shared_origin_identity is not None and len(cell_ids) == TEACHER_CELLS,
        "same_origin_two_seed_mechanism_only_v1 origin closure differs",
    )
    return tuple(rows)


_TEACHER_SEED_BINDING_FIELDS = {
    "schema_version",
    "population_profile",
    "teacher_cell_id",
    "origin_iid",
    "generation_seed",
    "candidate_bindings",
    "binding_digest",
}
_TEACHER_SEED_CANDIDATE_FIELDS = {
    "schema_version",
    "authority_kind",
    "branch",
    "latent_authority_receipt_path",
    "latent_authority_receipt_file_sha256",
    "latent_authority_receipt_digest_field",
    "latent_authority_receipt_digest",
    "candidate_envelope_path",
    "candidate_envelope_file_sha256",
    "candidate_seed_json_pointer",
    "candidate_branch_json_pointer",
    "candidate_analysis_split_json_pointer",
    "candidate_id_json_pointer",
    "native_receipt_path",
    "native_receipt_file_sha256",
    "native_receipt_digest",
    "native_sampling_seed_json_pointer",
    "native_gaussian_seed_json_pointer",
    "native_gaussian_raw_sha256_json_pointer",
    "native_media_json_pointer",
    "native_predecode_latent_json_pointer",
    "materialization_record_id",
    "materialization_record_receipt_path",
    "materialization_record_receipt_file_sha256",
    "materialization_record_receipt_digest",
    "candidate_binding_digest",
}


_PAIR_V5_ENVELOPE_FIELDS = {
    "schema_version",
    "root_spec_raw_sha256",
    "group_id",
    "visible_gpus",
    "ordinal",
    "sampling_contract",
    "semantic_input_closure",
    "artifact_use_contract",
    "split_contract",
    "candidate",
}
_PAIR_V5_CANDIDATE_FIELDS = {
    "candidate_id",
    "analysis_split",
    "action_family_id",
    "calibration_group_id",
    "prompt_group_id",
    "action_family_group_id",
    "actor_group_id",
    "scene_group_id",
    "action_group_id",
    "geometry_source_video",
    "geometry_source_video_sha256",
    "geometry_contract",
    "semantic_branch",
    "full_t2v_caption",
    "full_t2v_caption_utf8_sha256",
    "caption_contract",
    "seed",
}
_PAIR_V5_POINTERS = {
    "candidate_seed_json_pointer": "/candidate/seed",
    "candidate_branch_json_pointer": "/candidate/semantic_branch",
    "candidate_analysis_split_json_pointer": "/candidate/analysis_split",
    "candidate_id_json_pointer": "/candidate/candidate_id",
    "native_sampling_seed_json_pointer": "/sampling/t2v/seed",
    "native_gaussian_seed_json_pointer": (
        "/initial_noise_artifacts/t2v/generator_initial_seed"
    ),
    "native_gaussian_raw_sha256_json_pointer": (
        "/initial_noise_artifacts/t2v/raw_value_sha256"
    ),
    "native_media_json_pointer": "/outputs/t2v",
    "native_predecode_latent_json_pointer": (
        "/outputs/t2v/normalized_clean_latent"
    ),
}


def _load_bound_json(
    path_value: Any,
    expected_sha256: Any,
    *,
    label: str,
    digest_field: str | None = None,
    expected_digest: Any = None,
) -> Mapping[str, Any]:
    try:
        _path, raw = data_authority._read_stable_plain_bytes(
            path_value,
            expected_sha256,
            label=label,
            exact_mode=None,
            maximum_bytes=8 * 1024 * 1024,
        )
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"{label} physical reopen failed: {error}"
        ) from error
    _require(
        raw.endswith(b"\n")
        and bool(raw[:-1])
        and not raw[:-1].endswith(b"\n"),
        f"{label} canonical newline differs",
    )
    try:
        receipt = json.loads(
            raw[:-1],
            object_pairs_hook=data_authority._reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Full30MechanismCanaryAuthorityError(
            f"{label} cannot be decoded"
        ) from error
    _require(type(receipt) is dict, f"{label} root differs")
    _require(
        canonical_json_bytes(receipt) == raw[:-1],
        f"{label} is not canonical JSON",
    )
    if digest_field is not None:
        _require(
            digest_field in receipt,
            f"{label} digest field differs",
        )
        try:
            data_authority._verify_seal(receipt, digest_field, label)
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"{label} seal differs: {error}"
            ) from error
        _require(
            receipt[digest_field] == expected_digest,
            f"{label} digest binding differs",
        )
    return receipt


def _load_seed_authority_receipt(
    candidate: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    """Reopen the materializer wrapper; it is never the seed truth."""

    return _load_bound_json(
        candidate["latent_authority_receipt_path"],
        candidate["latent_authority_receipt_file_sha256"],
        label=f"{label}.latent_authority_receipt",
        digest_field=str(candidate["latent_authority_receipt_digest_field"]),
        expected_digest=candidate["latent_authority_receipt_digest"],
    )


def _validate_pair_v5_seed_truth(
    candidate: Mapping[str, Any],
    *,
    branch: str,
    record: Mapping[str, Any],
    label: str,
) -> tuple[int, str, str, str, str]:
    """Read seed/media/latent truth from the original PAIR-v5 files.

    The materializer latent-authority JSON may bind the two physical file
    identities, but no integer stored in that wrapper is consulted here.
    """

    for field, expected in _PAIR_V5_POINTERS.items():
        _require(
            candidate[field] == expected,
            f"{label}.{field} is not the fixed PAIR-v5 pointer",
        )
    envelope = _load_bound_json(
        candidate["candidate_envelope_path"],
        candidate["candidate_envelope_file_sha256"],
        label=f"{label}.candidate_envelope",
    )
    try:
        data_authority._closed(
            envelope, _PAIR_V5_ENVELOPE_FIELDS, f"{label}.candidate_envelope"
        )
        candidate_row = data_authority._closed(
            envelope["candidate"],
            _PAIR_V5_CANDIDATE_FIELDS,
            f"{label}.candidate_envelope.candidate",
        )
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"{label} PAIR-v5 candidate closure differs: {error}"
        ) from error
    _require(
        envelope["schema_version"] == PAIR_V5_CANDIDATE_ENVELOPE_SCHEMA_VERSION,
        f"{label} PAIR-v5 candidate schema differs",
    )
    seed = data_authority._resolve_json_pointer(
        envelope,
        candidate["candidate_seed_json_pointer"],
        f"{label}.candidate_seed_json_pointer",
    )
    candidate_branch = data_authority._resolve_json_pointer(
        envelope,
        candidate["candidate_branch_json_pointer"],
        f"{label}.candidate_branch_json_pointer",
    )
    candidate_split = data_authority._resolve_json_pointer(
        envelope,
        candidate["candidate_analysis_split_json_pointer"],
        f"{label}.candidate_analysis_split_json_pointer",
    )
    candidate_id = data_authority._resolve_json_pointer(
        envelope,
        candidate["candidate_id_json_pointer"],
        f"{label}.candidate_id_json_pointer",
    )
    _require(
        type(seed) is int and 0 <= seed < 2**64,
        f"{label} candidate seed must be uint64",
    )
    _require(
        candidate_branch == branch and candidate_split == "fit",
        f"{label} candidate is not the fit {branch} branch",
    )
    _require(
        candidate_row["action_family_id"] == record["event_id"]
        and candidate_row["action_family_group_id"] == record["event_id"]
        and candidate_row["calibration_group_id"]
        == record["teacher_cell_id"]
        and candidate_row["actor_group_id"] == record["actor_id"]
        and candidate_row["scene_group_id"] == record["scene_id"]
        and candidate_row["action_group_id"] == record["q0_id"],
        f"{label} candidate cell/event/actor/scene/action binding differs",
    )
    data_authority._safe_id(candidate_id, f"{label}.candidate_id")
    _sha(
        candidate_row["geometry_source_video_sha256"],
        label=f"{label}.candidate.geometry_source_video_sha256",
    )
    _sha(
        candidate_row["full_t2v_caption_utf8_sha256"],
        label=f"{label}.candidate.full_t2v_caption_utf8_sha256",
    )

    native_receipt = _load_bound_json(
        candidate["native_receipt_path"],
        candidate["native_receipt_file_sha256"],
        label=f"{label}.native_receipt",
        digest_field="receipt_digest",
        expected_digest=candidate["native_receipt_digest"],
    )
    _require(
        native_receipt.get("schema_version")
        == PAIR_V5_NATIVE_RECEIPT_SCHEMA_VERSION
        and native_receipt.get("method") == PAIR_V5_NATIVE_METHOD
        and native_receipt.get("arms") == ["t2v"],
        f"{label} native PAIR-v5 receipt identity differs",
    )
    try:
        native_sampling_seed = data_authority._resolve_json_pointer(
            native_receipt,
            candidate["native_sampling_seed_json_pointer"],
            f"{label}.native_sampling_seed_json_pointer",
        )
        native_gaussian_seed = data_authority._resolve_json_pointer(
            native_receipt,
            candidate["native_gaussian_seed_json_pointer"],
            f"{label}.native_gaussian_seed_json_pointer",
        )
        gaussian_raw_sha = data_authority._resolve_json_pointer(
            native_receipt,
            candidate["native_gaussian_raw_sha256_json_pointer"],
            f"{label}.native_gaussian_raw_sha256_json_pointer",
        )
        native_media = data_authority._resolve_json_pointer(
            native_receipt,
            candidate["native_media_json_pointer"],
            f"{label}.native_media_json_pointer",
        )
        native_latent = data_authority._resolve_json_pointer(
            native_receipt,
            candidate["native_predecode_latent_json_pointer"],
            f"{label}.native_predecode_latent_json_pointer",
        )
        native_input = native_receipt["input"]
        native_gaussian = native_receipt["initial_noise_artifacts"]["t2v"]
        sampling_row = native_receipt["sampling"]["t2v"]
        checkpoint_row = native_receipt["checkpoint"]
    except (KeyError, TypeError) as error:
        raise Full30MechanismCanaryAuthorityError(
            f"{label} native PAIR-v5 receipt is incomplete"
        ) from error
    _require(
        native_sampling_seed == native_gaussian_seed == seed,
        f"{label} candidate/native/official-Gaussian seeds differ",
    )
    gaussian_raw_sha = _sha(
        gaussian_raw_sha, label=f"{label}.official_initial_gaussian.raw_value_sha256"
    )
    gaussian_content_sha = _sha(
        native_gaussian.get("content_sha256"),
        label=f"{label}.official_initial_gaussian.content_sha256",
    )
    _require(
        native_gaussian.get("captured_from_native_sampler") is True
        and native_gaussian.get("external_initial_noise_injection") is False
        and native_gaussian.get("source_or_target_derived") is False
        and native_gaussian.get("observer_changed_return_value") is False
        and native_gaussian.get("official_randn_tensor_call_count") == 1,
        f"{label} official initial Gaussian provenance differs",
    )
    try:
        sampling_without_seed = dict(sampling_row)
        sampling_without_seed.pop("seed")
        runtime_projection = {
            "schema_version": native_receipt["schema_version"],
            "method": native_receipt["method"],
            "method_source_revision": native_receipt["method_source_revision"],
            "method_source_archive_sha256": native_receipt[
                "method_source_archive_sha256"
            ],
            "bernini_commit": native_receipt["bernini_commit"],
            "veomni_commit": native_receipt["veomni_commit"],
            "bernini_inference_files": native_receipt[
                "bernini_inference_files"
            ],
            "checkpoint_tree_sha256": checkpoint_row["tree_sha256"],
            "runtime_versions": native_receipt["runtime_versions"],
            "freeze_certificate": native_receipt["freeze_certificate"],
            "sampling_without_seed": sampling_without_seed,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise Full30MechanismCanaryAuthorityError(
            f"{label} fixed native runtime identity is incomplete"
        ) from error
    _sha(
        runtime_projection["method_source_archive_sha256"],
        label=f"{label}.method_source_archive_sha256",
    )
    _sha(
        runtime_projection["checkpoint_tree_sha256"],
        label=f"{label}.checkpoint_tree_sha256",
    )
    runtime_identity_digest = object_sha256(runtime_projection)
    _require(
        isinstance(native_input, Mapping)
        and native_input.get("source_video_sha256")
        == candidate_row["geometry_source_video_sha256"]
        and native_input.get("action_prompt_utf8_sha256")
        == candidate_row["full_t2v_caption_utf8_sha256"],
        f"{label} candidate/native geometry or prompt binding differs",
    )
    expected_media = {
        "path": record["reviewed_media"]["path"],
        "sha256": record["reviewed_media"]["file_sha256"],
    }
    target = record["target_clean_latent"]
    expected_latent = {
        "path": target["path"],
        "sha256": target["file_sha256"],
        "tensor_key": target["tensor_key"],
        "raw_value_sha256": target["tensor_raw_sha256"],
        "shape": target["shape"],
        "stored_dtype": "torch.float32",
        "coordinate": "bernini_normalized_clean_vae_latent",
        "native_sampler_before_vae_decode": True,
        "mp4_decode_reencode_used": False,
    }
    _require(
        isinstance(native_media, Mapping)
        and {
            "path": native_media.get("path"),
            "sha256": native_media.get("sha256"),
        }
        == expected_media,
        f"{label} native reviewed media binding differs",
    )
    _require(
        native_latent == expected_latent,
        f"{label} native predecode latent binding differs",
    )
    return (
        seed,
        str(candidate_id),
        gaussian_raw_sha,
        gaussian_content_sha,
        runtime_identity_digest,
    )


def _validate_pair_v5_runtime_identity_closure_v1(
    digests: Iterable[str], *, label: str
) -> None:
    values = tuple(digests)
    _require(
        bool(values) and len(set(values)) == 1,
        f"{label} PAIR-v5 fixed runtime identities differ",
    )


def _validate_teacher_seed_bindings(
    value: Any,
    *,
    teachers: Sequence[Mapping[str, Any]],
    run_authority: Any,
) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or len(value) != TEACHER_CELLS:
        fail("teacher_seed_bindings must contain exactly two rows")
    teacher_by_cell = {
        str(teacher["teacher_cell_id"]): teacher for teacher in teachers
    }
    origin_receipts: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in run_authority.record_receipts.values():
        if (
            receipt["record_kind"] == "teacher_anchor"
            and receipt["evidence_role"] == "teacher_origin"
        ):
            origin_receipts[
                (str(receipt["teacher_cell_id"]), str(receipt["branch"]))
            ] = receipt
    expected_keys = {
        (cell, branch) for cell in teacher_by_cell for branch in BRANCHES
    }
    _require(
        set(origin_receipts) == expected_keys,
        "teacher seed origin materialization closure differs",
    )
    rows: list[Mapping[str, Any]] = []
    seen_cells: set[str] = set()
    seen_seeds: set[int] = set()
    seen_authority_receipts: set[tuple[str, str, str]] = set()
    seen_candidate_ids: set[str] = set()
    seen_candidate_files: set[tuple[str, str]] = set()
    seen_native_receipts: set[tuple[str, str, str]] = set()
    seen_cell_gaussian_raw_shas: set[str] = set()
    seen_cell_gaussian_content_shas: set[str] = set()
    seen_runtime_identity_digests: set[str] = set()
    for ordinal, item in enumerate(value):
        label = f"teacher_seed_bindings[{ordinal}]"
        row = _sealed(
            item,
            _TEACHER_SEED_BINDING_FIELDS,
            "binding_digest",
            label=label,
        )
        _require(
            row["schema_version"] == TEACHER_SEED_BINDING_SCHEMA_VERSION,
            f"{label} schema differs",
        )
        _require(
            row["population_profile"] == POPULATION_PROFILE,
            f"{label} population profile differs",
        )
        cell = str(row["teacher_cell_id"])
        _require(cell in teacher_by_cell, f"{label} teacher cell is unknown")
        _require(cell not in seen_cells, "teacher seed cell is duplicated")
        teacher = teacher_by_cell[cell]
        _require(
            row["origin_iid"] == teacher["origin_iid"],
            f"{label} origin IID differs",
        )
        seed = row["generation_seed"]
        _require(
            type(seed) is int and 0 <= seed < 2**64,
            f"{label} generation seed must be uint64",
        )
        candidates = row["candidate_bindings"]
        if type(candidates) is not list or len(candidates) != len(BRANCHES):
            fail(f"{label}.candidate_bindings must contain action/incomplete")
        observed_branches: list[str] = []
        cell_truth_seeds: set[int] = set()
        cell_gaussian_raw_shas: set[str] = set()
        cell_gaussian_content_shas: set[str] = set()
        for candidate_ordinal, candidate_value in enumerate(candidates):
            candidate_label = (
                f"{label}.candidate_bindings[{candidate_ordinal}]"
            )
            candidate = _sealed(
                candidate_value,
                _TEACHER_SEED_CANDIDATE_FIELDS,
                "candidate_binding_digest",
                label=candidate_label,
            )
            _require(
                candidate["schema_version"]
                == TEACHER_SEED_CANDIDATE_BINDING_SCHEMA_VERSION,
                f"{candidate_label} schema differs",
            )
            _require(
                candidate["authority_kind"]
                == "pair-v5-candidate-plus-native-receipt",
                f"{candidate_label} authority kind differs",
            )
            branch = str(candidate["branch"])
            observed_branches.append(branch)
            _require(branch in BRANCHES, f"{candidate_label} branch differs")
            materialization = origin_receipts[(cell, branch)]
            record = materialization["record_authority"]
            record_id = str(materialization["record_id"])
            reference = run_authority.record_refs[record_id]
            latent_authority = record["target_clean_latent_authority"]
            expected_candidate_binding = {
                "latent_authority_receipt_path": latent_authority["path"],
                "latent_authority_receipt_file_sha256": latent_authority[
                    "file_sha256"
                ],
                "latent_authority_receipt_digest_field": latent_authority[
                    "digest_field"
                ],
                "latent_authority_receipt_digest": latent_authority["digest"],
                "materialization_record_id": record_id,
                "materialization_record_receipt_path": reference["path"],
                "materialization_record_receipt_file_sha256": reference[
                    "file_sha256"
                ],
                "materialization_record_receipt_digest": reference[
                    "record_receipt_digest"
                ],
            }
            for field, expected in expected_candidate_binding.items():
                _require(
                    candidate[field] == expected,
                    f"{candidate_label}.{field} differs from materialization origin authority",
                )
            receipt_identity = (
                str(candidate["latent_authority_receipt_path"]),
                str(candidate["latent_authority_receipt_file_sha256"]),
                str(candidate["latent_authority_receipt_digest"]),
            )
            _require(
                receipt_identity not in seen_authority_receipts,
                "teacher seed authority receipt is reused",
            )
            authority_receipt = _load_seed_authority_receipt(
                candidate, label=candidate_label
            )
            try:
                wrapper_candidate = data_authority._resolve_json_pointer(
                    authority_receipt,
                    "/pair_v5_candidate",
                    f"{candidate_label}.pair_v5_candidate",
                )
                wrapper_native = data_authority._resolve_json_pointer(
                    authority_receipt,
                    "/native_receipt",
                    f"{candidate_label}.native_receipt",
                )
            except Exception as error:
                raise Full30MechanismCanaryAuthorityError(
                    f"{candidate_label} wrapper file binding failed: {error}"
                ) from error
            _require(
                wrapper_candidate
                == {
                    "path": candidate["candidate_envelope_path"],
                    "sha256": candidate["candidate_envelope_file_sha256"],
                }
                and wrapper_native
                == {
                    "path": candidate["native_receipt_path"],
                    "sha256": candidate["native_receipt_file_sha256"],
                    "receipt_digest": candidate["native_receipt_digest"],
                },
                f"{candidate_label} wrapper/direct file binding differs",
            )
            (
                truth_seed,
                candidate_id,
                gaussian_raw_sha,
                gaussian_content_sha,
                runtime_identity_digest,
            ) = _validate_pair_v5_seed_truth(
                    candidate,
                    branch=branch,
                    record=record,
                    label=candidate_label,
                )
            _require(
                truth_seed == seed,
                f"{candidate_label} declared generation seed differs from PAIR-v5 truth",
            )
            candidate_file_identity = (
                str(candidate["candidate_envelope_path"]),
                str(candidate["candidate_envelope_file_sha256"]),
            )
            native_identity = (
                str(candidate["native_receipt_path"]),
                str(candidate["native_receipt_file_sha256"]),
                str(candidate["native_receipt_digest"]),
            )
            _require(
                candidate_id not in seen_candidate_ids
                and candidate_file_identity not in seen_candidate_files
                and native_identity not in seen_native_receipts,
                "PAIR-v5 candidate/native seed authority is reused",
            )
            seen_authority_receipts.add(receipt_identity)
            seen_candidate_ids.add(candidate_id)
            seen_candidate_files.add(candidate_file_identity)
            seen_native_receipts.add(native_identity)
            cell_truth_seeds.add(truth_seed)
            cell_gaussian_raw_shas.add(gaussian_raw_sha)
            cell_gaussian_content_shas.add(gaussian_content_sha)
            seen_runtime_identity_digests.add(runtime_identity_digest)
        _require(
            tuple(observed_branches) == BRANCHES,
            f"{label} candidate branch order differs",
        )
        _require(
            cell_truth_seeds == {seed}
            and len(cell_gaussian_raw_shas) == 1
            and len(cell_gaussian_content_shas) == 1,
            f"{label} action/incomplete do not share one physical PAIR-v5 "
            "seed/raw/content Gaussian identity",
        )
        _require(seed not in seen_seeds, "teacher generation seed is reused")
        _require(
            not (cell_gaussian_raw_shas & seen_cell_gaussian_raw_shas),
            "teacher official initial Gaussian raw SHA-256 is reused across cells",
        )
        _require(
            not (
                cell_gaussian_content_shas
                & seen_cell_gaussian_content_shas
            ),
            "teacher official initial Gaussian content SHA-256 is reused across cells",
        )
        seen_cells.add(cell)
        seen_seeds.add(seed)
        seen_cell_gaussian_raw_shas.update(cell_gaussian_raw_shas)
        seen_cell_gaussian_content_shas.update(cell_gaussian_content_shas)
        rows.append(row)
    expected_order = tuple(str(row["teacher_cell_id"]) for row in teachers)
    observed_order = tuple(str(row["teacher_cell_id"]) for row in rows)
    _require(observed_order == expected_order, "teacher seed binding order differs")
    _require(
        len(seen_seeds) == TEACHER_CELLS
        and len(seen_cell_gaussian_raw_shas) == TEACHER_CELLS
        and len(seen_cell_gaussian_content_shas) == TEACHER_CELLS,
        "teacher seed/raw/content Gaussian identity count closure differs",
    )
    _validate_pair_v5_runtime_identity_closure_v1(
        seen_runtime_identity_digests,
        label="teacher",
    )
    return tuple(rows)


def _validate_sources(
    value: Any, teachers: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or len(value) != SOURCE_UNITS:
        fail("canary sources must contain exactly eight rows")
    rows: list[Mapping[str, Any]] = []
    seen_iids: set[str] = set()
    seen_video_shas: set[str] = set()
    seen_index0_shas: set[str] = set()
    seen_groups: set[str] = set()
    for ordinal, item in enumerate(value):
        label = f"sources[{ordinal}]"
        try:
            row = data_authority._closed(item, data_authority._SOURCE_FIELDS, label)
            data_authority._verify_seal(row, "source_digest", label)
            data_authority._require(
                row["schema_version"] == data_authority.SOURCE_SCHEMA,
                f"{label} schema differs",
            )
            data_authority._require(row["analysis_split"] == "fit", f"{label} is not fit")
            iid = data_authority._iid(row["source_iid"], f"{label}.source_iid")
            video_sha = data_authority._sha(row["source_video_sha256"], f"{label}.source_video_sha256")
            data_authority._verify_file(row["source_video_path"], video_sha, f"{label}.source_video")
            index0_sha = data_authority._sha(
                row["source_posterior_index0_sha256"],
                f"{label}.source_posterior_index0_sha256",
            )
            index0_path = data_authority._verify_file(
                row["source_posterior_index0_path"], index0_sha, f"{label}.source_posterior_index0"
            )
            data_authority._require(
                index0_path.name == f"{iid}.source-posterior-index0.pt",
                f"{label} index0 filename differs",
            )
            data_authority._text(row["source_posterior_tensor_key"], f"{label}.source_posterior_tensor_key")
            data_authority._require(row["posterior_index_decoded"] == 0, f"{label} posterior index differs")
            data_authority._require(row["physical_index0_only"] is True, f"{label} is not physical index0-only")
            for field in (
                "synthetic_target_index1_bytes_read",
                "synthetic_target_index1_decoded",
                "synthetic_target_index1_hashed",
            ):
                data_authority._require(row[field] is False, f"{label}.{field} is not false")
            group = data_authority._safe_id(row["source_group_id"], f"{label}.source_group_id")
            for field in (
                "actor_id",
                "scene_id",
                "event_id",
                "actor_kind",
                "q0_id",
                "source_motion_label",
            ):
                data_authority._safe_id(row[field], f"{label}.{field}")
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"{label} formal row/physical reopen failed: {error}"
            ) from error
        _require(iid not in seen_iids, "canary source IID is duplicated")
        _require(video_sha not in seen_video_shas, "canary source video SHA is duplicated")
        _require(index0_sha not in seen_index0_shas, "canary source index0 SHA is duplicated")
        _require(group not in seen_groups, "canary source group is duplicated")
        seen_iids.add(iid)
        seen_video_shas.add(video_sha)
        seen_index0_shas.add(index0_sha)
        seen_groups.add(group)
        rows.append(row)
    teacher_iids = {str(row["origin_iid"]) for row in teachers}
    teacher_shas = {str(row["origin_source_sha256"]) for row in teachers}
    teacher_groups = {str(row["origin_group_id"]) for row in teachers}
    _require(not (seen_iids & teacher_iids), "canary source overlaps teacher origin IID")
    _require(not (seen_video_shas & teacher_shas), "canary source overlaps teacher origin SHA")
    _require(not (seen_groups & teacher_groups), "canary source overlaps teacher origin group")
    return tuple(rows)


def _validate_pairs(
    value: Any,
    sources: Sequence[Mapping[str, Any]],
    teachers: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or len(value) != PAIR_ROWS:
        fail("canary pairs must contain exactly sixteen rows")
    source_by_iid = {str(row["source_iid"]): row for row in sources}
    teacher_by_cell = {str(row["teacher_cell_id"]): row for row in teachers}
    rows: list[Mapping[str, Any]] = []
    seen_pair_ids: set[str] = set()
    seen_review_ids: set[str] = set()
    branches_by_source: dict[str, set[str]] = defaultdict(set)
    teacher_by_source: dict[str, set[str]] = defaultdict(set)
    instruction_shas: dict[str, set[str]] = defaultdict(set)
    for ordinal, item in enumerate(value):
        label = f"pairs[{ordinal}]"
        try:
            row = data_authority._closed(item, data_authority._PAIR_FIELDS, label)
            data_authority._verify_seal(row, "pair_digest", label)
            data_authority._require(row["schema_version"] == data_authority.PAIR_SCHEMA, f"{label} schema differs")
            pair_id = data_authority._safe_id(row["pair_id"], f"{label}.pair_id")
            source_iid = data_authority._iid(row["source_iid"], f"{label}.source_iid")
            data_authority._require(source_iid in source_by_iid, f"{label} source is unknown")
            source = source_by_iid[source_iid]
            data_authority._require(row["analysis_split"] == source["analysis_split"] == "fit", f"{label} split differs")
            data_authority._require(row["source_video_sha256"] == source["source_video_sha256"], f"{label} source SHA differs")
            branch = row["branch"]
            data_authority._require(branch in BRANCHES, f"{label} branch differs")
            cell = data_authority._safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
            data_authority._require(cell in teacher_by_cell, f"{label} teacher is unknown")
            teacher = teacher_by_cell[cell]
            for field in ("event_id", "actor_kind", "q0_id"):
                data_authority._require(row[field] == source[field] == teacher[field], f"{label}.{field} compatibility differs")
            data_authority._require(row["source_motion_label"] == source["source_motion_label"], f"{label} source motion differs")
            data_authority._require(row["target_event_incompatible_with_source_motion"] is True, f"{label} target/source motion is compatible")
            data_authority._require(row["optimizer_admitted"] is True, f"{label} is not optimizer admitted")
            instruction = data_authority._text(row["instruction"], f"{label}.instruction")
            instruction_sha = data_authority._sha(row["instruction_utf8_sha256"], f"{label}.instruction_utf8_sha256")
            data_authority._require(
                hashlib.sha256(instruction.encode("utf-8")).hexdigest() == instruction_sha,
                f"{label} instruction SHA differs",
            )
            review = data_authority._validate_review(
                row["pre_admission_full81_review"],
                pair_id=pair_id,
                source=source,
                branch=str(branch),
                label=f"{label}.pre_admission_full81_review",
            )
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"{label} formal row/review validation failed: {error}"
            ) from error
        _require(pair_id not in seen_pair_ids, "canary pair ID is duplicated")
        review_id = str(review["review_id"])
        _require(review_id not in seen_review_ids, "canary pair full81 review is reused")
        _require(branch not in branches_by_source[source_iid], "canary source branch is duplicated")
        seen_pair_ids.add(pair_id)
        seen_review_ids.add(review_id)
        branches_by_source[source_iid].add(str(branch))
        teacher_by_source[source_iid].add(cell)
        instruction_shas[source_iid].add(instruction_sha)
        rows.append(row)
    _require(set(branches_by_source) == set(source_by_iid), "not every canary source has pair rows")
    _require(all(value == set(BRANCHES) for value in branches_by_source.values()), "each canary source needs action/incomplete")
    _require(all(len(value) == 1 for value in teacher_by_source.values()), "canary source binds multiple teachers")
    _require(all(len(value) == 2 for value in instruction_shas.values()), "canary source instructions alias")

    cell_order = tuple(str(row["teacher_cell_id"]) for row in teachers)
    source_order = tuple(str(row["source_iid"]) for row in sources)
    expected_cells = (cell_order[0],) * SOURCES_PER_CELL + (cell_order[1],) * SOURCES_PER_CELL
    observed_cells = tuple(next(iter(teacher_by_source[source])) for source in source_order)
    _require(observed_cells == expected_cells, "canary source order must be four sources per preregistered cell")
    expected_pair_order = tuple(
        (source, branch)
        for source in source_order
        for branch in BRANCHES
    )
    observed_pair_order = tuple((str(row["source_iid"]), str(row["branch"])) for row in rows)
    _require(observed_pair_order == expected_pair_order, "canary pair rows must follow source/action/incomplete order")
    for cell in cell_order:
        assigned = [source_by_iid[source] for source in source_order if cell in teacher_by_source[source]]
        _require(len(assigned) == SOURCES_PER_CELL, "canary teacher source capacity differs")
        for field in ("actor_id", "scene_id", "source_group_id"):
            _require(len({str(row[field]) for row in assigned}) == SOURCES_PER_CELL, f"canary {cell} {field} diversity differs")
    return tuple(rows)


def _validate_global_anchor_video_reuse_v1(value: Sequence[Any]) -> None:
    bindings: dict[str, tuple[str, str, str, str, str, str, str, str]] = {}
    for ordinal, item in enumerate(value):
        if not isinstance(item, Mapping):
            fail(f"representation_admissions[{ordinal}] is not a mapping")
        for field, role in (
            ("origin_evidence", "teacher_origin"),
            ("cross_anchor_evidence", "same_event_cross_anchor"),
        ):
            evidence = item.get(field)
            if not isinstance(evidence, Mapping):
                fail(f"representation_admissions[{ordinal}].{field} differs")
            video_sha = _sha(
                evidence.get("anchor_video_sha256"),
                label=f"representation_admissions[{ordinal}].{field}.anchor_video_sha256",
            )
            intrinsic = (
                role,
                str(evidence.get("anchor_iid")),
                str(evidence.get("anchor_split")),
                str(evidence.get("branch")),
                str(evidence.get("event_id")),
                str(evidence.get("actor_kind")),
                str(evidence.get("actor_id")),
                str(evidence.get("scene_id")),
            )
            prior = bindings.get(video_sha)
            if prior is None:
                bindings[video_sha] = intrinsic
            else:
                _require(
                    role == "same_event_cross_anchor" and prior == intrinsic,
                    "reused canary anchor MP4 intrinsic identity differs or is not cross-anchor reuse",
                )


def _validate_representations(
    value: Any,
    teachers: Sequence[Mapping[str, Any]],
    run_authority: Any,
) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or len(value) != REPRESENTATION_BUNDLES:
        fail("canary representation_admissions must contain exactly four rows")
    _validate_global_anchor_video_reuse_v1(value)
    teacher_by_cell = {str(row["teacher_cell_id"]): row for row in teachers}
    fragments_value = run_authority.receipt["representation_sigma_evidence_candidates"]
    if type(fragments_value) is not list:
        fail("canary materialization representation fragments differ")
    fragments: dict[tuple[str, str], Mapping[str, Any]] = {}
    for ordinal, item in enumerate(fragments_value):
        try:
            fragment = data_authority._closed(
                item,
                data_authority._MATERIALIZATION_REPRESENTATION_FRAGMENT_FIELDS,
                f"materialization representation fragments[{ordinal}]",
            )
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"canary materialization representation fragment differs: {error}"
            ) from error
        key = (str(fragment["teacher_cell_id"]), str(fragment["branch"]))
        _require(key not in fragments, "canary materialization representation fragment is duplicated")
        fragments[key] = fragment

    rows: list[Mapping[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_admission_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()
    seen_anchor_ids: set[str] = set()
    seen_review_ids: set[str] = set()
    seen_review_digests: set[str] = set()
    seen_sidecars: set[str] = set()
    seen_nuisance: set[str] = set()
    seen_materialization_record_ids: set[str] = set()
    seen_materialization_noise_seeds: set[int] = set()
    seen_materialization_noise_shas: set[str] = set()
    seen_materialization_state_digests: set[str] = set()
    seen_materialization_forward_digests: set[str] = set()
    seen_materialization_container_shas: set[str] = set()
    for ordinal, item in enumerate(value):
        label = f"representation_admissions[{ordinal}]"
        try:
            row = data_authority._closed(item, data_authority._REPRESENTATION_FIELDS, label)
            data_authority._verify_seal(row, "admission_digest", label)
            data_authority._require(row["schema_version"] == data_authority.REPRESENTATION_SCHEMA, f"{label} schema differs")
            admission_id = data_authority._safe_id(row["admission_id"], f"{label}.admission_id")
            cell = data_authority._safe_id(row["teacher_cell_id"], f"{label}.teacher_cell_id")
            data_authority._require(cell in teacher_by_cell, f"{label} teacher is unknown")
            origin = teacher_by_cell[cell]
            branch = row["branch"]
            data_authority._require(branch in BRANCHES, f"{label} branch differs")
            key = (cell, str(branch))
            data_authority._require(row["analysis_split"] == origin["analysis_split"] == "fit", f"{label} split differs")
            data_authority._require(row["event_id"] == origin["event_id"], f"{label} event differs")
            origin_evidence, origin_sidecar, origin_nuisance, origin_materialization = data_authority._validate_anchor_evidence(
                row["origin_evidence"],
                evidence_role="teacher_origin",
                origin=origin,
                branch=str(branch),
                run_authority=run_authority,
                label=f"{label}.origin_evidence",
            )
            cross_evidence, cross_sidecar, _cross_nuisance, cross_materialization = data_authority._validate_anchor_evidence(
                row["cross_anchor_evidence"],
                evidence_role="same_event_cross_anchor",
                origin=origin,
                branch=str(branch),
                run_authority=run_authority,
                label=f"{label}.cross_anchor_evidence",
            )
            data_authority._require(
                origin_evidence["anchor_split"]
                == cross_evidence["anchor_split"]
                == "fit"
                and origin_materialization["record_authority"][
                    "analysis_split"
                ]
                == cross_materialization["record_authority"][
                    "analysis_split"
                ]
                == "fit",
                f"{label} origin/cross/materialization population is not fit-only",
            )
            data_authority._require(
                origin_evidence["anchor_id"] != cross_evidence["anchor_id"],
                f"{label} cross anchor reuses origin",
            )
            data_authority._validate_sigma_evidence(
                row["sigma_evidence"],
                origin_anchor_id=str(origin_evidence["anchor_id"]),
                cross_anchor_id=str(cross_evidence["anchor_id"]),
                origin_sidecar=origin_sidecar,
                cross_sidecar=cross_sidecar,
                origin_nuisance=origin_nuisance,
                label=f"{label}.sigma_evidence",
            )
            data_authority._require(key in fragments, f"{label} materialization fragment is absent")
            expected_fragment = {
                "teacher_cell_id": cell,
                "branch": branch,
                "origin_record_id": origin_materialization["record_id"],
                "cross_anchor_record_id": cross_materialization["record_id"],
                "origin_evidence_digest": origin_materialization["candidate_authority_evidence"]["evidence_digest"],
                "cross_anchor_evidence_digest": cross_materialization["candidate_authority_evidence"]["evidence_digest"],
                "sigma_evidence": row["sigma_evidence"],
            }
            data_authority._require(fragments[key] == expected_fragment, f"{label} materialization fragment differs")
            data_authority._require(row["optimizer_admitted"] is True, f"{label} is not optimizer admitted")
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"{label} formal physical evidence validation failed: {error}"
            ) from error
        _require(key not in seen_keys, "canary representation cell/branch is duplicated")
        _require(admission_id not in seen_admission_ids, "canary representation admission ID is duplicated")
        seen_keys.add(key)
        seen_admission_ids.add(admission_id)
        for evidence_role, evidence in (
            ("teacher_origin", origin_evidence),
            ("same_event_cross_anchor", cross_evidence),
        ):
            evidence_id = str(evidence["evidence_id"])
            anchor_id = str(evidence["anchor_id"])
            review = evidence["pre_admission_blind_review"]
            review_id = str(review["review_id"])
            review_digest = str(review["review_digest"])
            sidecar_sha = str(evidence["psiout_sidecar_sha256"])
            nuisance_sha = str(evidence["nuisance_packet_sha256"])
            _require(evidence_id not in seen_evidence_ids, "canary representation evidence ID is reused")
            _require(anchor_id not in seen_anchor_ids, "canary representation anchor ID is reused")
            _require(review_id not in seen_review_ids, "canary representation review ID is reused")
            _require(review_digest not in seen_review_digests, "canary representation review digest is reused")
            _require(sidecar_sha not in seen_sidecars, "canary PsiOut sidecar is reused")
            _require(nuisance_sha not in seen_nuisance, "canary nuisance packet is reused")
            seen_evidence_ids.add(evidence_id)
            seen_anchor_ids.add(anchor_id)
            seen_review_ids.add(review_id)
            seen_review_digests.add(review_digest)
            seen_sidecars.add(sidecar_sha)
            seen_nuisance.add(nuisance_sha)
        for materialization in (origin_materialization, cross_materialization):
            record_id = str(materialization["record_id"])
            _require(record_id not in seen_materialization_record_ids, "canary teacher materialization record is reused")
            seen_materialization_record_ids.add(record_id)
        key_materializations = (origin_materialization, cross_materialization)
        key_noise_seeds = {
            int(materialization["noise_seed"])
            for materialization in key_materializations
        }
        key_noise_shas = {
            str(materialization["noise_raw_sha256"])
            for materialization in key_materializations
        }
        _require(
            len(key_noise_seeds) == 1,
            f"canary teacher seed differs inside {key!r}",
        )
        _require(
            not (key_noise_seeds & seen_materialization_noise_seeds),
            "canary teacher seed is reused across cell/branch keys",
        )
        _require(
            not (key_noise_shas & seen_materialization_noise_shas),
            "canary teacher noise bytes are reused across cell/branch keys",
        )
        seen_materialization_noise_seeds.update(key_noise_seeds)
        seen_materialization_noise_shas.update(key_noise_shas)
        key_state_digests = {
            str(state["state_digest"])
            for materialization in key_materializations
            for state in materialization["state_receipts"]
        }
        key_forward_digests = {
            str(forward["forward_digest"])
            for materialization in key_materializations
            for forward in materialization["forward_receipts"]
        }
        key_container_shas = {
            str(binding["file_sha256"])
            for materialization in key_materializations
            for binding in materialization["container_bindings"]
        }
        _require(
            not (key_state_digests & seen_materialization_state_digests),
            "canary materialization state is reused across cell/branch keys",
        )
        _require(
            not (key_forward_digests & seen_materialization_forward_digests),
            "canary materialization forward is reused across cell/branch keys",
        )
        _require(
            not (key_container_shas & seen_materialization_container_shas),
            "canary materialization container is reused across cell/branch keys",
        )
        seen_materialization_state_digests.update(key_state_digests)
        seen_materialization_forward_digests.update(key_forward_digests)
        seen_materialization_container_shas.update(key_container_shas)
        rows.append(row)

    expected_keys = {
        (str(teacher["teacher_cell_id"]), branch)
        for teacher in teachers
        for branch in BRANCHES
    }
    _require(seen_keys == expected_keys, "canary representation cell/branch closure differs")
    _require(set(fragments) == expected_keys, "canary materialization representation fragment closure differs")
    run_teacher_records = {
        record_id
        for record_id, receipt in run_authority.record_receipts.items()
        if receipt["record_kind"] == "teacher_anchor"
    }
    _require(seen_materialization_record_ids == run_teacher_records, "canary materialization teacher record closure differs")
    expected_order = tuple(
        (str(teacher["teacher_cell_id"]), branch)
        for teacher in teachers
        for branch in BRANCHES
    )
    observed_order = tuple((str(row["teacher_cell_id"]), str(row["branch"])) for row in rows)
    _require(observed_order == expected_order, "canary representation rows must follow teacher/action/incomplete order")
    _require(
        len(seen_materialization_noise_seeds) == REPRESENTATION_BUNDLES,
        "canary independent teacher noise authority closure differs",
    )
    return tuple(rows)


@dataclass(frozen=True)
class ValidatedMechanismCanaryDataAuthorityV1:
    manifest_file_sha256: str
    manifest_digest: str
    manifest: Mapping[str, Any]
    validation_receipt: Mapping[str, Any]


def load_data_authority_v1(
    *, manifest_path: str | Path, expected_manifest_sha256: str
) -> ValidatedMechanismCanaryDataAuthorityV1:
    manifest_sha = _sha(expected_manifest_sha256, label="canary data manifest expected SHA-256")
    manifest = _load_manifest(manifest_path, manifest_sha, label="canary data authority manifest")
    manifest = _sealed(manifest, _TOP_FIELDS, "manifest_digest", label="canary data authority manifest")
    _require(manifest["schema_version"] == SCHEMA_VERSION, "canary data authority schema differs")
    _require(
        manifest["population_profile"] == POPULATION_PROFILE,
        "canary population profile differs",
    )
    authority_row = data_authority._closed(manifest["authority"], _AUTHORITY_FIELDS, "canary authority")
    _require(authority_row == _EXPECTED_AUTHORITY, "canary execution authority differs")
    try:
        data_authority._validate_io_policy(manifest["source_io_policy"])
        run_authority = data_authority._load_materialization_run_v1(
            manifest["materialization_run_receipt"]
        )
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"canary source/materialization authority failed physical reopen: {error}"
        ) from error
    teachers = _validate_teacher_origins(manifest["teacher_origins"])
    seed_bindings = _validate_teacher_seed_bindings(
        manifest["teacher_seed_bindings"],
        teachers=teachers,
        run_authority=run_authority,
    )
    sources = _validate_sources(manifest["sources"], teachers)
    pairs = _validate_pairs(manifest["pairs"], sources, teachers)
    representations = _validate_representations(
        manifest["representation_admissions"], teachers, run_authority
    )
    _require(manifest["authority_counts"] == _EXPECTED_COUNTS, "canary authority count closure differs")
    generation_media_shas = [
        str(row["origin_evidence"]["anchor_video_sha256"])
        for row in representations
    ]
    _require(
        len(set(generation_media_shas)) == REPRESENTATION_BUNDLES,
        "canary teacher generation media is reused across cells/branches",
    )
    receipt_unsigned = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "manifest_file_sha256": manifest_sha,
        "manifest_digest": manifest["manifest_digest"],
        "materialization_run_digest": run_authority.receipt["run_digest"],
        "materialization_run_receipt_sha256": run_authority.binding["file_sha256"],
        "materialization_record_receipts": len(run_authority.record_receipts),
        "teacher_cells": len(teachers),
        "population_profile": POPULATION_PROFILE,
        "shared_origin_identities": len(
            {
                (
                    str(row["origin_iid"]),
                    str(row["origin_source_sha256"]),
                    str(row["origin_group_id"]),
                    str(row["event_id"]),
                    str(row["actor_kind"]),
                    str(row["q0_id"]),
                    str(row["actor_id"]),
                    str(row["scene_id"]),
                )
                for row in teachers
            }
        ),
        "same_origin_profile_verified": True,
        "teacher_seed_bindings": len(seed_bindings),
        "distinct_teacher_generation_seeds": len(
            {int(row["generation_seed"]) for row in seed_bindings}
        ),
        "distinct_teacher_gaussian_raw_sha256": TEACHER_CELLS,
        "distinct_teacher_gaussian_content_sha256": TEACHER_CELLS,
        "teacher_seed_physical_receipts_reopened": len(seed_bindings)
        * len(BRANCHES),
        "teacher_pair_v5_candidate_files_reopened": TEACHER_CELLS
        * len(BRANCHES),
        "teacher_native_receipts_reopened": TEACHER_CELLS * len(BRANCHES),
        "teacher_fixed_runtime_identity_digests": 1,
        "action_incomplete_candidate_roots_required_equal": False,
        "cross_experiment_action_incomplete_seed_binding_permitted": True,
        "materializer_wrapper_is_seed_truth": False,
        "source_units": len(sources),
        "optimizer_pair_rows": len(pairs),
        "representation_bundles": len(representations),
        "teacher_generation_media_outputs": len(generation_media_shas),
        "teacher_generation_media_outputs_unique": True,
        "per_cell_evidence_noise_state_forward_sidecar_container_unique": True,
        "maximum_updates": MAX_UPDATES,
        "physical_source_index0_reopened": True,
        "formal_private_materialization_validators_reused": True,
        "synthetic_target_index1_bytes_read": False,
        "synthetic_target_bytes_read": False,
        "confirmation_population_admitted": False,
        "generalization": False,
        "identity_generalization": False,
        "event_family_generalization": False,
        "formal_authority": False,
        "mechanism_only": True,
        "tail_serialization_only": True,
        "scientific_success_claimed": False,
        "optimizer_authorized": True,
    }
    receipt = {**receipt_unsigned, "validation_digest": object_sha256(receipt_unsigned)}
    return ValidatedMechanismCanaryDataAuthorityV1(
        manifest_file_sha256=manifest_sha,
        manifest_digest=str(manifest["manifest_digest"]),
        manifest=MappingProxyType(dict(manifest)),
        validation_receipt=MappingProxyType(receipt),
    )


_AMPLITUDE_PARENT_FIELDS = {
    "schema_version",
    "manifest_file_sha256",
    "manifest_digest",
    "validation_digest",
    "binding_digest",
}
_AMPLITUDE_AUTHORITY_FIELDS = {
    "status",
    "experiment_id",
    "calibration_complete",
    "current_optimizer_bundles",
    "current_calibrator_evidence",
    "current_frozen_fail_evidence",
    "maximum_updates",
    "formal_authority",
    "mechanism_only",
    "scientific_success_claimed",
    "optimizer_authorized",
    "population_profile",
    "synthetic_target_bytes_read",
}
_AMPLITUDE_TOP_FIELDS = {
    "schema_version",
    "population_profile",
    "parent_authority",
    "materialization_run_receipt",
    "materializer_plan_admission",
    "frozen_runtime_identity",
    "calibration_bundles",
    "authority_counts",
    "authority",
    "manifest_digest",
}
_MATERIALIZER_PLAN_BINDING_FIELDS = {
    "schema_version",
    "population_profile",
    "plan_path",
    "plan_file_sha256",
    "plan_id",
    "plan_digest",
    "run_plan_id",
    "run_plan_digest",
    "run_record_bridge_digest",
    "run_fragment_binding_digest",
    "schedule_run_seed",
    "admission_receipt_path",
    "admission_receipt_file_sha256",
    "admission_validation_digest",
    "authority_projection_digest",
    "parent_manifest_file_sha256",
    "parent_manifest_digest",
    "materialization_run_receipt_file_sha256",
    "materialization_run_digest",
    "binding_digest",
}
_EXPECTED_AMPLITUDE_COUNTS = {
    "optimizer_bundles": AMPLITUDE_BUNDLES,
    "calibrator_evidence": AMPLITUDE_BUNDLES
    * amplitude_authority.CALIBRATORS_PER_BUNDLE,
    "frozen_fail_evidence": AMPLITUDE_BUNDLES
    * amplitude_authority.FAIL_CONTROLS_PER_BUNDLE,
    "sigma_floor_rows": AMPLITUDE_BUNDLES * len(SIGMA_INDICES),
}
_EXPECTED_AMPLITUDE_AUTHORITY = {
    "status": "optimizer_admitted_disposable_canary",
    "experiment_id": EXPERIMENT_ID,
    "calibration_complete": True,
    "current_optimizer_bundles": AMPLITUDE_BUNDLES,
    "current_calibrator_evidence": _EXPECTED_AMPLITUDE_COUNTS[
        "calibrator_evidence"
    ],
    "current_frozen_fail_evidence": _EXPECTED_AMPLITUDE_COUNTS[
        "frozen_fail_evidence"
    ],
    "maximum_updates": MAX_UPDATES,
    "formal_authority": False,
    "mechanism_only": True,
    "scientific_success_claimed": False,
    "optimizer_authorized": True,
    "population_profile": POPULATION_PROFILE,
    "synthetic_target_bytes_read": False,
}


def _amplitude_authority_projection_digest(
    manifest: Mapping[str, Any],
) -> str:
    fields = _AMPLITUDE_TOP_FIELDS - {
        "materializer_plan_admission",
        "manifest_digest",
    }
    try:
        projection = {field: manifest[field] for field in fields}
    except KeyError as error:
        raise Full30MechanismCanaryAuthorityError(
            "canary amplitude authority projection is incomplete"
        ) from error
    return object_sha256(projection)


def _schedule_rows_from_parent_v1(
    parent: ValidatedMechanismCanaryDataAuthorityV1,
    *,
    learning_module: Any,
) -> tuple[Any, ...]:
    action_row_type = getattr(learning_module, "ActionPairRow", None)
    if not callable(action_row_type):
        fail("frozen learning ActionPairRow API differs")
    return tuple(
        action_row_type(
            row_id=row["pair_id"],
            source_id=row["source_iid"],
            branch=row["branch"],
            teacher_cell_id=row["teacher_cell_id"],
        )
        for row in parent.manifest["pairs"]
    )


def _official_materializer_plan_projection_v1(
    run_plan: Mapping[str, Any], *, materializer_module: Any
) -> Mapping[str, Any]:
    """Project only the frozen cross-condition ABI mismatch.

    The formal evidence validator retains three wrong-control conditions on a
    cross anchor, while the frozen official materializer ABI executes only
    branch/noop/camera/appearance for that record kind.  Every other byte of
    every record remains bound; the three legacy conditions are evidence-only.
    """

    seal = getattr(materializer_module, "seal_record", None)
    cross_roles = tuple(
        getattr(materializer_module, "CROSS_CONDITION_ROLES", ())
    )
    if not callable(seal) or cross_roles != (
        "branch",
        "noop",
        "camera_only",
        "appearance_only",
    ):
        fail("official materializer cross-condition projection API differs")
    try:
        projected = json.loads(canonical_json_bytes(run_plan))
        records = projected["records"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise Full30MechanismCanaryAuthorityError(
            "final materialization run plan cannot be projected"
        ) from error
    for ordinal, record in enumerate(records):
        if (
            record.get("record_kind") == "teacher_anchor"
            and record.get("evidence_role") == "same_event_cross_anchor"
        ):
            conditions = record.get("conditions")
            _require(
                type(conditions) is list
                and tuple(row.get("role") for row in conditions[:4])
                == cross_roles,
                f"final run cross record {ordinal} condition prefix differs",
            )
            unsigned = dict(record)
            unsigned.pop("record_digest", None)
            unsigned["conditions"] = conditions[:4]
            records[ordinal] = dict(seal(unsigned, "record_digest"))
    unsigned_plan = dict(projected)
    unsigned_plan.pop("plan_digest", None)
    unsigned_plan["records"] = records
    return MappingProxyType(dict(seal(unsigned_plan, "plan_digest")))


def _plan_run_bridge_digest_v1(
    official_plan: Mapping[str, Any],
    run_plan: Mapping[str, Any],
) -> str:
    official = official_plan.get("records")
    final = run_plan.get("records")
    if type(official) is not list or type(final) is not list or len(official) != len(final):
        fail("official/final materialization record bridge closure differs")
    rows = []
    for official_record, run_record in zip(official, final):
        _require(
            official_record.get("record_id") == run_record.get("record_id"),
            "official/final materialization record order differs",
        )
        rows.append(
            {
                "record_id": run_record["record_id"],
                "record_kind": run_record["record_kind"],
                "evidence_role": run_record["evidence_role"],
                "official_record_digest": official_record["record_digest"],
                "final_run_record_digest": run_record["record_digest"],
                "condition_roles": [
                    row["role"] for row in official_record["conditions"]
                ],
            }
        )
    return object_sha256(rows)


def _run_fragment_binding_digest_v1(run_receipt: Mapping[str, Any]) -> str:
    return object_sha256(
        {
            "representation_sigma_evidence_candidates": run_receipt[
                "representation_sigma_evidence_candidates"
            ],
            "amplitude_sigma_calibration_candidates": run_receipt[
                "amplitude_sigma_calibration_candidates"
            ],
        }
    )


def _validate_parent_binding(
    value: Any,
    *,
    parent: ValidatedMechanismCanaryDataAuthorityV1,
) -> Mapping[str, Any]:
    row = _sealed(
        value,
        _AMPLITUDE_PARENT_FIELDS,
        "binding_digest",
        label="canary amplitude parent authority",
    )
    expected = {
        "schema_version": PARENT_BINDING_SCHEMA_VERSION,
        "manifest_file_sha256": parent.manifest_file_sha256,
        "manifest_digest": parent.manifest_digest,
        "validation_digest": parent.validation_receipt["validation_digest"],
    }
    unsigned = dict(row)
    unsigned.pop("binding_digest")
    _require(unsigned == expected, "canary amplitude parent binding differs")
    return row


def _canary_amplitude_runtime_indexes(
    parent: ValidatedMechanismCanaryDataAuthorityV1,
) -> tuple[
    Mapping[tuple[str, str], Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    manifest = parent.manifest
    representations = {
        (str(row["teacher_cell_id"]), str(row["branch"])): row
        for row in manifest["representation_admissions"]
    }
    sources = {
        str(row["source_iid"]): row for row in manifest["sources"]
    }
    pairs: dict[str, Mapping[str, Any]] = {}
    for row in manifest["pairs"]:
        augmented = dict(row)
        augmented["source_posterior_index0_sha256"] = sources[
            str(row["source_iid"])
        ]["source_posterior_index0_sha256"]
        pairs[str(row["pair_id"])] = MappingProxyType(augmented)
    _require(
        len(representations) == REPRESENTATION_BUNDLES,
        "canary amplitude parent representation closure differs",
    )
    _require(len(pairs) == PAIR_ROWS, "canary amplitude parent pair closure differs")
    return MappingProxyType(representations), MappingProxyType(pairs)


def _validate_materializer_plan_binding_v1(
    value: Any,
    *,
    amplitude_manifest: Mapping[str, Any],
    parent: ValidatedMechanismCanaryDataAuthorityV1,
    run_authority: Any,
    materializer_module: Any,
    checkpoint_module: Any,
    learning_module: Any,
) -> Mapping[str, Any]:
    binding = _sealed(
        value,
        _MATERIALIZER_PLAN_BINDING_FIELDS,
        "binding_digest",
        label="canary materializer plan binding",
    )
    projection_digest = _amplitude_authority_projection_digest(
        amplitude_manifest
    )
    _require(
        binding["schema_version"] == MATERIALIZER_PLAN_BINDING_SCHEMA_VERSION
        and binding["population_profile"] == POPULATION_PROFILE
        and binding["authority_projection_digest"] == projection_digest
        and binding["parent_manifest_file_sha256"]
        == parent.manifest_file_sha256
        and binding["parent_manifest_digest"] == parent.manifest_digest
        and binding["materialization_run_receipt_file_sha256"]
        == run_authority.binding["file_sha256"]
        and binding["materialization_run_digest"]
        == run_authority.receipt["run_digest"]
        and binding["run_plan_id"] == run_authority.receipt["plan_id"]
        and binding["run_plan_digest"] == run_authority.receipt["plan_digest"],
        "canary materializer binding differs from final authority manifests/run",
    )
    run_seed = binding["schedule_run_seed"]
    _require(
        type(run_seed) is int and 0 <= run_seed < 2**64,
        "canary materializer schedule run seed must be uint64",
    )
    plan = _load_bound_json(
        binding["plan_path"],
        binding["plan_file_sha256"],
        label="canary exact official materializer plan",
        digest_field="plan_digest",
        expected_digest=binding["plan_digest"],
    )
    admission_receipt = _load_bound_json(
        binding["admission_receipt_path"],
        binding["admission_receipt_file_sha256"],
        label="canary materializer plan admission receipt",
        digest_field="validation_digest",
        expected_digest=binding["admission_validation_digest"],
    )
    _require(
        plan.get("plan_id") == binding["plan_id"]
        and plan.get("plan_digest") == binding["plan_digest"],
        "canary materializer plan identity differs",
    )
    schedule = build_checkpoint_scaffold_schedule_v1(
        _schedule_rows_from_parent_v1(parent, learning_module=learning_module),
        run_seed=run_seed,
        learning_module=learning_module,
        checkpoint_module=checkpoint_module,
    )
    admitted = admit_reduced_materialization_plan_v1(
        plan,
        schedule=schedule,
        materializer_module=materializer_module,
        checkpoint_module=checkpoint_module,
        authority_projection_digest=projection_digest,
        parent_manifest_file_sha256=parent.manifest_file_sha256,
        parent_manifest_digest=parent.manifest_digest,
    )
    _require(
        dict(admitted.validation_receipt) == dict(admission_receipt),
        "canary materializer admission receipt does not replay exactly",
    )
    expected_official_plan = _official_materializer_plan_projection_v1(
        run_authority.receipt["plan_authority"],
        materializer_module=materializer_module,
    )
    bridge_digest = _plan_run_bridge_digest_v1(
        admitted.plan, run_authority.receipt["plan_authority"]
    )
    fragment_digest = _run_fragment_binding_digest_v1(run_authority.receipt)
    _require(
        dict(admitted.plan) == dict(expected_official_plan)
        and binding["run_record_bridge_digest"] == bridge_digest
        and binding["run_fragment_binding_digest"] == fragment_digest,
        "canary official plan/run record-fragment bridge differs",
    )
    try:
        ordered_run_records = [
            run_authority.record_receipts[str(reference["record_id"])][
                "record_authority"
            ]
            for reference in run_authority.receipt["record_receipts"]
        ]
    except (KeyError, TypeError) as error:
        raise Full30MechanismCanaryAuthorityError(
            "canary materialization run record closure is incomplete"
        ) from error
    _require(
        ordered_run_records == run_authority.receipt["plan_authority"]["records"]
        and run_authority.receipt["record_count"]
        == admitted.validation_receipt["record_count"]
        and len(
            run_authority.receipt[
                "representation_sigma_evidence_candidates"
            ]
        )
        == REPRESENTATION_BUNDLES
        and len(
            run_authority.receipt[
                "amplitude_sigma_calibration_candidates"
            ]
        )
        == AMPLITUDE_BUNDLES,
        "canary plan/run records or materialization fragments differ",
    )
    return binding


def load_amplitude_authority_v1(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    parent_manifest_path: str | Path,
    expected_parent_manifest_sha256: str,
    materializer_module: Any,
    checkpoint_module: Any,
    learning_module: Any,
) -> amplitude_authority.ValidatedAmplitudeAuthorityV1:
    """Validate four amplitude bundles using the frozen formal validators."""

    parent = load_data_authority_v1(
        manifest_path=parent_manifest_path,
        expected_manifest_sha256=expected_parent_manifest_sha256,
    )
    manifest_sha = _sha(
        expected_manifest_sha256,
        label="canary amplitude manifest expected SHA-256",
    )
    manifest = _load_manifest(
        manifest_path,
        manifest_sha,
        label="canary amplitude authority manifest",
    )
    manifest = _sealed(
        manifest,
        _AMPLITUDE_TOP_FIELDS,
        "manifest_digest",
        label="canary amplitude authority manifest",
    )
    _require(
        manifest["schema_version"] == AMPLITUDE_SCHEMA_VERSION,
        "canary amplitude authority schema differs",
    )
    _require(
        manifest["population_profile"] == POPULATION_PROFILE,
        "canary amplitude population profile differs",
    )
    parent_binding = _validate_parent_binding(
        manifest["parent_authority"], parent=parent
    )
    try:
        run_authority = data_authority._load_materialization_run_v1(
            manifest["materialization_run_receipt"]
        )
        runtime = amplitude_authority._validate_runtime_identity(
            manifest["frozen_runtime_identity"]
        )
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"canary amplitude runtime/materialization physical reopen failed: {error}"
        ) from error
    _require(
        runtime == run_authority.receipt["runtime_identity"],
        "canary amplitude runtime differs from materialization run",
    )
    representations, pair_by_id = _canary_amplitude_runtime_indexes(parent)
    bundles = manifest["calibration_bundles"]
    if type(bundles) is not list or len(bundles) != AMPLITUDE_BUNDLES:
        fail("canary calibration_bundles must contain exactly four rows")

    fragments_value = run_authority.receipt[
        "amplitude_sigma_calibration_candidates"
    ]
    if type(fragments_value) is not list:
        fail("canary amplitude materialization fragments differ")
    fragments: dict[tuple[str, str], Mapping[str, Any]] = {}
    for ordinal, item in enumerate(fragments_value):
        try:
            fragment = data_authority._closed(
                item,
                data_authority._MATERIALIZATION_AMPLITUDE_FRAGMENT_FIELDS,
                f"canary amplitude materialization fragments[{ordinal}]",
            )
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"canary amplitude materialization fragment differs: {error}"
            ) from error
        key = (str(fragment["teacher_cell_id"]), str(fragment["branch"]))
        _require(key not in fragments, "canary amplitude materialization fragment is reused")
        fragments[key] = fragment

    floors: dict[tuple[str, str, int], Any] = {}
    seen_keys: set[tuple[str, str]] = set()
    observed_order: list[tuple[str, str]] = []
    seen_calibration_ids: set[str] = set()
    seen_identities: dict[str, set[str]] = {
        "evidence_ids": set(),
        "review_ids": set(),
        "pair_ids": set(),
        "output_shas": set(),
        "container_shas": set(),
        "calibrator_noise_shas": set(),
        "materialization_record_ids": set(),
    }
    expected_order = tuple(representations)
    for ordinal, item in enumerate(bundles):
        key = (
            str(item.get("teacher_cell_id")),
            str(item.get("branch")),
        ) if isinstance(item, Mapping) else ("", "")
        _require(key in fragments, f"canary calibration_bundles[{ordinal}] materialization fragment is absent")
        try:
            bundle, bundle_floors, identities = amplitude_authority._validate_bundle(
                item,
                representation_by_key=representations,
                pair_by_id=pair_by_id,
                run_authority=run_authority,
                materialization_fragment=fragments[key],
                label=f"canary calibration_bundles[{ordinal}]",
            )
        except Exception as error:
            raise Full30MechanismCanaryAuthorityError(
                f"canary calibration_bundles[{ordinal}] formal physical validation failed: {error}"
            ) from error
        _require(key not in seen_keys, "canary amplitude cell/branch is duplicated")
        calibration_id = str(bundle["calibration_id"])
        _require(calibration_id not in seen_calibration_ids, "canary amplitude calibration ID is duplicated")
        seen_keys.add(key)
        observed_order.append(key)
        seen_calibration_ids.add(calibration_id)
        for identity_kind, values in identities.items():
            observed = set(values)
            _require(
                not (observed & seen_identities[identity_kind]),
                f"canary amplitude {identity_kind} are reused across bundles",
            )
            seen_identities[identity_kind].update(observed)
        for sigma_index, floor in bundle_floors.items():
            floor_key = (*key, sigma_index)
            _require(floor_key not in floors, "canary amplitude floor is duplicated")
            floors[floor_key] = floor

    _require(tuple(observed_order) == expected_order, "canary amplitude bundle order differs")
    _require(seen_keys == set(representations), "canary amplitude cell/branch closure differs")
    _require(set(fragments) == set(representations), "canary amplitude materialization fragment closure differs")
    run_calibrator_records = {
        record_id
        for record_id, receipt in run_authority.record_receipts.items()
        if receipt["record_kind"] == "amplitude_calibrator"
    }
    _require(
        seen_identities["materialization_record_ids"] == run_calibrator_records,
        "canary amplitude materialization record closure differs",
    )
    _require(
        len(floors) == AMPLITUDE_BUNDLES * len(SIGMA_INDICES),
        "canary amplitude floor closure differs",
    )
    _require(
        manifest["authority_counts"] == _EXPECTED_AMPLITUDE_COUNTS,
        "canary amplitude authority counts differ",
    )
    authority_row = data_authority._closed(
        manifest["authority"],
        _AMPLITUDE_AUTHORITY_FIELDS,
        "canary amplitude authority",
    )
    _require(
        authority_row == _EXPECTED_AMPLITUDE_AUTHORITY,
        "canary amplitude execution authority differs",
    )
    materializer_binding = _validate_materializer_plan_binding_v1(
        manifest["materializer_plan_admission"],
        amplitude_manifest=manifest,
        parent=parent,
        run_authority=run_authority,
        materializer_module=materializer_module,
        checkpoint_module=checkpoint_module,
        learning_module=learning_module,
    )
    receipt_unsigned = {
        "schema_version": AMPLITUDE_VALIDATION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "manifest_file_sha256": manifest_sha,
        "manifest_digest": manifest["manifest_digest"],
        "parent_manifest_file_sha256": parent.manifest_file_sha256,
        "parent_manifest_digest": parent_binding["manifest_digest"],
        "parent_validation_digest": parent_binding["validation_digest"],
        "frozen_runtime_digest": runtime["runtime_digest"],
        "materialization_run_digest": run_authority.receipt["run_digest"],
        "materialization_run_receipt_sha256": run_authority.binding["file_sha256"],
        "materialization_record_receipts": len(run_authority.record_receipts),
        "materializer_plan_file_sha256": materializer_binding[
            "plan_file_sha256"
        ],
        "materializer_plan_digest": materializer_binding["plan_digest"],
        "materializer_admission_receipt_file_sha256": materializer_binding[
            "admission_receipt_file_sha256"
        ],
        "materializer_admission_validation_digest": materializer_binding[
            "admission_validation_digest"
        ],
        "materializer_authority_projection_digest": materializer_binding[
            "authority_projection_digest"
        ],
        "schedule_run_seed": materializer_binding["schedule_run_seed"],
        "secure_official_materializer_revalidated": True,
        "plan_run_records_fragments_exactly_bound": True,
        "optimizer_bundles": AMPLITUDE_BUNDLES,
        "calibrator_evidence": _EXPECTED_AMPLITUDE_COUNTS["calibrator_evidence"],
        "frozen_fail_evidence": _EXPECTED_AMPLITUDE_COUNTS["frozen_fail_evidence"],
        "sigma_floor_rows": len(floors),
        "formal_private_amplitude_validators_reused": True,
        "all_floors_greater_than_1e-6": True,
        "synthetic_target_index1_bytes_read": False,
        "synthetic_target_bytes_read": False,
        "population_profile": POPULATION_PROFILE,
        "generalization": False,
        "identity_generalization": False,
        "event_family_generalization": False,
        "formal_authority": False,
        "mechanism_only": True,
        "scientific_success_claimed": False,
        "optimizer_authorized": True,
    }
    receipt = {
        **receipt_unsigned,
        "validation_digest": object_sha256(receipt_unsigned),
    }
    return amplitude_authority.ValidatedAmplitudeAuthorityV1(
        manifest_file_sha256=manifest_sha,
        manifest_digest=str(manifest["manifest_digest"]),
        parent_manifest_file_sha256=parent.manifest_file_sha256,
        parent_manifest_digest=parent.manifest_digest,
        frozen_runtime_digest=str(runtime["runtime_digest"]),
        floors=MappingProxyType(floors),
        validation_receipt=MappingProxyType(receipt),
    )


@dataclass(frozen=True)
class ValidatedMechanismCanaryAuthorityV1:
    data: ValidatedMechanismCanaryDataAuthorityV1
    amplitude: amplitude_authority.ValidatedAmplitudeAuthorityV1


def load_mechanism_canary_authority_v1(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    amplitude_manifest_path: str | Path,
    expected_amplitude_manifest_sha256: str,
    materializer_module: Any,
    checkpoint_module: Any,
    learning_module: Any,
) -> ValidatedMechanismCanaryAuthorityV1:
    data = load_data_authority_v1(
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    amplitude = load_amplitude_authority_v1(
        manifest_path=amplitude_manifest_path,
        expected_manifest_sha256=expected_amplitude_manifest_sha256,
        parent_manifest_path=manifest_path,
        expected_parent_manifest_sha256=expected_manifest_sha256,
        materializer_module=materializer_module,
        checkpoint_module=checkpoint_module,
        learning_module=learning_module,
    )
    _require(
        amplitude.parent_manifest_file_sha256 == data.manifest_file_sha256,
        "canary composite amplitude parent differs",
    )
    return ValidatedMechanismCanaryAuthorityV1(data=data, amplitude=amplitude)


@dataclass(frozen=True)
class AdmittedMechanismCanaryMaterializationPlanV1:
    """An unchanged official plan plus reduced executable-coordinate receipt."""

    plan: Mapping[str, Any]
    validation_receipt: Mapping[str, Any]


def admit_reduced_materialization_plan_v1(
    plan: Any,
    *,
    schedule: Sequence[Any],
    materializer_module: Any,
    checkpoint_module: Any,
    authority_projection_digest: str,
    parent_manifest_file_sha256: str,
    parent_manifest_digest: str,
) -> AdmittedMechanismCanaryMaterializationPlanV1:
    """Admit an exact reduced plan without weakening the official materializer.

    The official provider ABI deliberately still computes its complete six-sigma
    evidence container for every admitted record.  This adapter binds only the
    evidence coordinates used by schedule rows 0..15 as executable.  Remaining
    sealed sigma evidence is non-executable and may never be resolved by the
    trainer.
    """

    validator = getattr(materializer_module, "validate_materialization_plan_v1", None)
    if not callable(validator):
        fail("official materializer plan-validator API differs")
    try:
        admitted_plan = validator(plan)
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"official reduced materialization plan validation failed: {error}"
        ) from error
    if not isinstance(admitted_plan, Mapping):
        fail("official materializer returned a non-mapping plan")
    if tuple(getattr(materializer_module, "SIGMA_INDICES", ())) != SIGMA_INDICES:
        fail("official materializer six-sigma ABI differs")
    authority_projection_digest = _sha(
        authority_projection_digest,
        label="canary final authority projection digest",
    )
    parent_manifest_file_sha256 = _sha(
        parent_manifest_file_sha256,
        label="canary parent manifest file SHA-256",
    )
    parent_manifest_digest = _sha(
        parent_manifest_digest,
        label="canary parent manifest digest",
    )

    records = admitted_plan.get("records")
    if type(records) is not list or len(records) != 16:
        fail("canary materialization plan must contain exactly sixteen records")
    teacher_records = [
        row for row in records if row.get("record_kind") == "teacher_anchor"
    ]
    amplitude_records = [
        row
        for row in records
        if row.get("record_kind") == "amplitude_calibrator"
    ]
    if len(teacher_records) != 8 or len(amplitude_records) != 8:
        fail("canary materialization plan must contain eight teacher and eight calibrator records")

    teacher_keys = {
        (str(row["teacher_cell_id"]), str(row["branch"]))
        for row in teacher_records
    }
    cells = tuple(
        sorted(
            {cell for cell, _branch in teacher_keys},
            key=lambda item: item.encode("utf-8"),
        )
    )
    if len(cells) != TEACHER_CELLS or teacher_keys != {
        (cell, branch) for cell in cells for branch in BRANCHES
    }:
        fail("canary materialization teacher cell/branch closure differs")
    for key in teacher_keys:
        rows = [
            row
            for row in teacher_records
            if (str(row["teacher_cell_id"]), str(row["branch"])) == key
        ]
        if Counter(str(row["evidence_role"]) for row in rows) != Counter(
            {"teacher_origin": 1, "same_event_cross_anchor": 1}
        ):
            fail(f"canary materialization teacher roles differ for {key!r}")
    origin_identities_by_cell: dict[str, set[tuple[str, ...]]] = (
        defaultdict(set)
    )
    for row in teacher_records:
        if row["evidence_role"] == "teacher_origin":
            origin_identities_by_cell[str(row["teacher_cell_id"])].add(
                (
                    str(row["anchor_iid"]),
                    str(row["analysis_split"]),
                    str(row["event_id"]),
                    str(row["actor_kind"]),
                    str(row["q0_id"]),
                    str(row["actor_id"]),
                    str(row["scene_id"]),
                )
            )
    if (
        set(origin_identities_by_cell) != set(cells)
        or any(
            len(identities) != 1
            for identities in origin_identities_by_cell.values()
        )
        or len(
            {
                next(iter(identities))
                for identities in origin_identities_by_cell.values()
            }
        )
        != 1
    ):
        fail(
            "same_origin_two_seed_mechanism_only_v1 materialization cells "
            "must share one exact origin identity"
        )
    origin_generation_media_shas = [
        str(row["reviewed_media"]["file_sha256"])
        for row in teacher_records
        if row["evidence_role"] == "teacher_origin"
    ]
    origin_generation_review_digests = [
        str(row["review"]["review_digest"])
        for row in teacher_records
        if row["evidence_role"] == "teacher_origin"
    ]
    _require(
        len(origin_generation_media_shas) == REPRESENTATION_BUNDLES
        and len(set(origin_generation_media_shas)) == REPRESENTATION_BUNDLES,
        "same-origin teacher generation media is reused across cells/branches",
    )
    _require(
        len(origin_generation_review_digests) == REPRESENTATION_BUNDLES
        and len(set(origin_generation_review_digests))
        == REPRESENTATION_BUNDLES,
        "same-origin teacher generation review is reused across cells/branches",
    )

    generation_seed_by_cell: dict[str, int] = {}
    generation_authority_receipts: set[tuple[str, str, str]] = set()
    generation_candidate_files: set[tuple[str, str]] = set()
    generation_native_receipts: set[tuple[str, str, str]] = set()
    generation_gaussian_raw_shas: set[str] = set()
    generation_gaussian_content_shas: set[str] = set()
    generation_runtime_identity_digests: set[str] = set()
    for cell in cells:
        cell_seed_values: set[int] = set()
        cell_candidate_ids: set[str] = set()
        cell_gaussian_raw_shas: set[str] = set()
        cell_gaussian_content_shas: set[str] = set()
        for branch in BRANCHES:
            record = next(
                row
                for row in teacher_records
                if row["teacher_cell_id"] == cell
                and row["branch"] == branch
                and row["evidence_role"] == "teacher_origin"
            )
            latent_authority = record["target_clean_latent_authority"]
            authority_candidate = {
                "latent_authority_receipt_path": latent_authority["path"],
                "latent_authority_receipt_file_sha256": latent_authority[
                    "file_sha256"
                ],
                "latent_authority_receipt_digest_field": latent_authority[
                    "digest_field"
                ],
                "latent_authority_receipt_digest": latent_authority["digest"],
            }
            authority_receipt = _load_seed_authority_receipt(
                authority_candidate,
                label=f"materialization generation seed {cell}/{branch}",
            )
            try:
                pair_v5_candidate = data_authority._resolve_json_pointer(
                    authority_receipt,
                    "/pair_v5_candidate",
                    f"materialization PAIR-v5 candidate {cell}/{branch}",
                )
                native_receipt = data_authority._resolve_json_pointer(
                    authority_receipt,
                    "/native_receipt",
                    f"materialization native receipt {cell}/{branch}",
                )
            except Exception as error:
                raise Full30MechanismCanaryAuthorityError(
                    f"materialization direct PAIR-v5 binding failed: {error}"
                ) from error
            _require(
                isinstance(pair_v5_candidate, Mapping)
                and set(pair_v5_candidate) == {"path", "sha256"}
                and isinstance(native_receipt, Mapping)
                and set(native_receipt) == {"path", "sha256", "receipt_digest"},
                "materialization direct PAIR-v5 file binding differs",
            )
            direct_candidate = {
                "candidate_envelope_path": pair_v5_candidate["path"],
                "candidate_envelope_file_sha256": pair_v5_candidate["sha256"],
                "native_receipt_path": native_receipt["path"],
                "native_receipt_file_sha256": native_receipt["sha256"],
                "native_receipt_digest": native_receipt["receipt_digest"],
                **_PAIR_V5_POINTERS,
            }
            (
                seed_value,
                candidate_id,
                gaussian_raw_sha,
                gaussian_content_sha,
                runtime_identity_digest,
            ) = _validate_pair_v5_seed_truth(
                    direct_candidate,
                    branch=branch,
                    record=record,
                    label=f"materialization generation truth {cell}/{branch}",
                )
            receipt_identity = (
                str(latent_authority["path"]),
                str(latent_authority["file_sha256"]),
                str(latent_authority["digest"]),
            )
            _require(
                receipt_identity not in generation_authority_receipts,
                "materialization generation authority receipt is reused",
            )
            candidate_identity = (
                str(pair_v5_candidate["path"]),
                str(pair_v5_candidate["sha256"]),
            )
            native_identity = (
                str(native_receipt["path"]),
                str(native_receipt["sha256"]),
                str(native_receipt["receipt_digest"]),
            )
            _require(
                candidate_identity not in generation_candidate_files
                and native_identity not in generation_native_receipts,
                "materialization PAIR-v5 candidate/native receipt is reused",
            )
            generation_authority_receipts.add(receipt_identity)
            generation_candidate_files.add(candidate_identity)
            generation_native_receipts.add(native_identity)
            cell_seed_values.add(seed_value)
            cell_candidate_ids.add(candidate_id)
            cell_gaussian_raw_shas.add(gaussian_raw_sha)
            cell_gaussian_content_shas.add(gaussian_content_sha)
            generation_runtime_identity_digests.add(runtime_identity_digest)
        _require(
            len(cell_seed_values) == 1
            and len(cell_candidate_ids) == len(BRANCHES)
            and len(cell_gaussian_raw_shas) == 1
            and len(cell_gaussian_content_shas) == 1,
            "same-event cell branches do not share one direct PAIR-v5 "
            "seed/raw/content Gaussian identity",
        )
        _require(
            not (cell_gaussian_raw_shas & generation_gaussian_raw_shas),
            "same-event teacher official Gaussian raw SHA-256 is reused across cells",
        )
        _require(
            not (
                cell_gaussian_content_shas
                & generation_gaussian_content_shas
            ),
            "same-event teacher official Gaussian content SHA-256 is reused across cells",
        )
        generation_gaussian_raw_shas.update(cell_gaussian_raw_shas)
        generation_gaussian_content_shas.update(cell_gaussian_content_shas)
        generation_seed_by_cell[cell] = next(iter(cell_seed_values))
    _require(
        len(set(generation_seed_by_cell.values())) == TEACHER_CELLS,
        "same-event teacher generation seed is reused across cells",
    )
    _require(
        len(generation_gaussian_raw_shas) == TEACHER_CELLS,
        "same-event teacher official Gaussian raw SHA-256 count closure differs",
    )
    _require(
        len(generation_gaussian_content_shas) == TEACHER_CELLS,
        "same-event teacher official Gaussian content SHA-256 count closure differs",
    )
    _validate_pair_v5_runtime_identity_closure_v1(
        generation_runtime_identity_digests,
        label="materialization",
    )

    teacher_records_by_key: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in teacher_records:
        teacher_records_by_key[
            (str(row["teacher_cell_id"]), str(row["branch"]))
        ].append(row)
    noise_seeds: set[int] = set()
    noise_shas: set[str] = set()
    for key, rows in teacher_records_by_key.items():
        key_seeds = {int(row["noise"]["seed"]) for row in rows}
        key_noise_shas = {
            str(row["noise"]["artifact"]["tensor_raw_sha256"])
            for row in rows
        }
        if len(key_seeds) != 1:
            fail(f"same-event teacher seed differs inside {key!r}")
        if key_seeds & noise_seeds:
            fail("same-event canary teacher seed is reused across cells/branches")
        if key_noise_shas & noise_shas:
            fail("same-event canary teacher noise bytes are reused across cells/branches")
        noise_seeds.update(key_seeds)
        noise_shas.update(key_noise_shas)

    amplitude_keys = {
        (str(row["teacher_cell_id"]), str(row["branch"]))
        for row in amplitude_records
    }
    if amplitude_keys != teacher_keys:
        fail("canary materialization calibrator cell/branch closure differs")
    for key in amplitude_keys:
        if sum(
            (
                str(row["teacher_cell_id"]),
                str(row["branch"]),
            )
            == key
            for row in amplitude_records
        ) != amplitude_authority.CALIBRATORS_PER_BUNDLE:
            fail(f"canary materialization calibrator count differs for {key!r}")
    amplitude_source_ids = tuple(
        str(row["source_iid"]) for row in amplitude_records
    )
    amplitude_pair_ids = tuple(str(row["pair_id"]) for row in amplitude_records)
    if (
        len(set(amplitude_source_ids)) != SOURCE_UNITS
        or len(set(amplitude_pair_ids)) != SOURCE_UNITS
    ):
        fail("canary materializer must physically reopen eight distinct real index0 sources")
    if any(row.get("analysis_split") != "fit" for row in records):
        fail("confirmation materialization records are forbidden in the canary")

    try:
        canonical = checkpoint_module.canonical_schedule_v2(schedule)
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"canary materializer schedule binding is not canonical_schedule_v2: {error}"
        ) from error
    if len(canonical) != CHECKPOINT_FLAT_ROWS:
        fail("canary materializer schedule binding length differs")
    executable = canonical[:EXECUTABLE_FLAT_ROWS]
    if [int(row["global_index"]) for row in executable] != list(
        range(EXECUTABLE_FLAT_ROWS)
    ):
        fail("canary materializer executable global-index prefix differs")
    if any(
        is_serialization_tail_source_v1(row["row"]["source_id"])
        for row in executable
    ):
        fail("canary materializer executable prefix contains serialization tail")
    executable_source_ids = {
        str(row["row"]["source_id"]) for row in executable
    }
    if executable_source_ids != set(amplitude_source_ids):
        fail("canary materializer source population differs from executable schedule")
    executable_pairs = {
        str(row["row"]["row_id"]): (
            str(row["row"]["source_id"]),
            str(row["row"]["branch"]),
            str(row["row"]["teacher_cell_id"]),
        )
        for row in executable
    }
    for row in amplitude_records:
        pair_id = str(row["pair_id"])
        expected = (
            str(row["source_iid"]),
            str(row["branch"]),
            str(row["teacher_cell_id"]),
        )
        if executable_pairs.get(pair_id) != expected:
            fail("canary materializer calibrator differs from executable pair authority")

    executable_coordinates = [
        {
            "global_index": int(row["global_index"]),
            "update": int(row["update"]),
            "source_iid": str(row["row"]["source_id"]),
            "pair_id": str(row["row"]["row_id"]),
            "teacher_cell_id": str(row["row"]["teacher_cell_id"]),
            "branch": str(row["row"]["branch"]),
            "sigma_index": int(row["sigma_index"]),
            "noise_seed": int(row["noise_seed"]),
        }
        for row in executable
    ]
    executable_evidence_keys = {
        (
            row["teacher_cell_id"],
            row["branch"],
            row["sigma_index"],
        )
        for row in executable_coordinates
    }
    materialized_evidence_keys = {
        (cell, branch, sigma_index)
        for cell, branch in teacher_keys
        for sigma_index in SIGMA_INDICES
    }
    if not executable_evidence_keys <= materialized_evidence_keys:
        fail("canary executable coordinate lacks official materializer evidence")
    sort_coordinate = lambda value: (
        value[0].encode("utf-8"),
        value[1].encode("utf-8"),
        value[2],
    )
    encoded_materialized = [
        {
            "teacher_cell_id": cell,
            "branch": branch,
            "sigma_index": sigma_index,
        }
        for cell, branch, sigma_index in sorted(
            materialized_evidence_keys, key=sort_coordinate
        )
    ]
    encoded_non_executable = [
        {
            "teacher_cell_id": cell,
            "branch": branch,
            "sigma_index": sigma_index,
        }
        for cell, branch, sigma_index in sorted(
            materialized_evidence_keys - executable_evidence_keys,
            key=sort_coordinate,
        )
    ]
    receipt_unsigned = {
        "schema_version": MATERIALIZER_PLAN_ADMISSION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "population_profile": POPULATION_PROFILE,
        "final_authority_projection_digest": authority_projection_digest,
        "parent_manifest_file_sha256": parent_manifest_file_sha256,
        "parent_manifest_digest": parent_manifest_digest,
        "plan_id": admitted_plan["plan_id"],
        "plan_digest": admitted_plan["plan_digest"],
        "record_count": len(records),
        "teacher_record_count": len(teacher_records),
        "amplitude_calibrator_record_count": len(amplitude_records),
        "teacher_cells": len(cells),
        "shared_origin_identities": len(
            {
                next(iter(identities))
                for identities in origin_identities_by_cell.values()
            }
        ),
        "teacher_generation_media_outputs": len(
            origin_generation_media_shas
        ),
        "teacher_generation_reviews": len(
            origin_generation_review_digests
        ),
        "independent_teacher_noise_authorities": len(noise_seeds),
        "distinct_teacher_generation_seeds": len(
            set(generation_seed_by_cell.values())
        ),
        "distinct_teacher_gaussian_raw_sha256": len(
            generation_gaussian_raw_shas
        ),
        "distinct_teacher_gaussian_content_sha256": len(
            generation_gaussian_content_shas
        ),
        "teacher_generation_authority_receipts_reopened": len(
            generation_authority_receipts
        ),
        "teacher_pair_v5_candidate_files_reopened": len(
            generation_candidate_files
        ),
        "teacher_native_receipts_reopened": len(generation_native_receipts),
        "teacher_fixed_runtime_identity_digests": len(
            generation_runtime_identity_digests
        ),
        "action_incomplete_candidate_roots_required_equal": False,
        "cross_experiment_action_incomplete_seed_binding_permitted": True,
        "materializer_wrapper_is_seed_truth": False,
        "physically_reopened_real_source_index0_records": len(
            set(amplitude_source_ids)
        ),
        "representation_bundles": len(teacher_keys),
        "amplitude_bundles": len(amplitude_keys),
        "official_materializer_plan_validator_reused": True,
        "official_materializer_six_sigma_abi_retained": True,
        "legacy_cross_wrong_control_conditions_executable": False,
        "official_cross_condition_roles": list(
            getattr(materializer_module, "CROSS_CONDITION_ROLES", ())
        ),
        "materialized_evidence_coordinates": encoded_materialized,
        "executable_training_coordinates": executable_coordinates,
        "executable_evidence_coordinate_count": len(executable_evidence_keys),
        "non_executable_evidence_coordinates": encoded_non_executable,
        "non_executable_evidence_trainer_read_authorized": False,
        "training_noise_materialized_by_official_materializer": False,
        "training_noise_authority": "frozen-learning-schedule-v1",
        "synthetic_target_index1_bytes_read": False,
        "synthetic_target_bytes_read": False,
        "confirmation_population_admitted": False,
        "generalization": False,
        "identity_generalization": False,
        "event_family_generalization": False,
        "formal_authority": False,
        "mechanism_only": True,
        "maximum_updates": MAX_UPDATES,
    }
    receipt = {
        **receipt_unsigned,
        "validation_digest": object_sha256(receipt_unsigned),
    }
    return AdmittedMechanismCanaryMaterializationPlanV1(
        # The frozen materializer intentionally requires an exact ``dict``
        # (rather than a generic Mapping) at its public entry point.
        plan=dict(admitted_plan),
        validation_receipt=MappingProxyType(receipt),
    )


def _row_coordinates(value: Any) -> tuple[str, str, str, str]:
    row = getattr(value, "row", value)
    try:
        coordinates = (
            row.row_id,
            row.source_id,
            row.branch,
            row.teacher_cell_id,
        )
    except AttributeError as error:
        raise Full30MechanismCanaryAuthorityError(
            "canary schedule input row type differs"
        ) from error
    if any(type(item) is not str or not item or "\x00" in item for item in coordinates):
        fail("canary schedule input row text differs")
    return coordinates


def _validate_executable_rows(
    rows: Iterable[Any],
) -> tuple[
    tuple[Any, ...],
    tuple[str, ...],
    tuple[str, ...],
    Mapping[str, Mapping[str, Any]],
]:
    values = tuple(rows)
    if len(values) != PAIR_ROWS:
        fail("canary executable schedule requires exactly sixteen pair rows")
    by_source: dict[str, dict[str, Any]] = {}
    row_contract: dict[str, Mapping[str, Any]] = {}
    cell_order: list[str] = []
    for value in values:
        row_id, source_id, branch, cell = _row_coordinates(value)
        if branch not in BRANCHES:
            fail("canary executable schedule branch differs")
        if row_id in row_contract:
            fail("canary executable schedule row ID is duplicated")
        source_rows = by_source.setdefault(source_id, {})
        if branch in source_rows:
            fail("canary executable source branch is duplicated")
        source_rows[branch] = value
        row_contract[row_id] = MappingProxyType(
            {
                "source_id": source_id,
                "branch": branch,
                "teacher_cell_id": cell,
            }
        )
        if cell not in cell_order:
            cell_order.append(cell)
    if len(by_source) != SOURCE_UNITS:
        fail("canary executable schedule requires eight unique sources")
    source_order = tuple(by_source)
    if any(set(branch_rows) != set(BRANCHES) for branch_rows in by_source.values()):
        fail("canary executable sources require matched action/incomplete rows")
    if len(cell_order) != TEACHER_CELLS:
        fail("canary executable schedule requires two teacher cells")
    expected_cells = (
        (cell_order[0],) * SOURCES_PER_CELL
        + (cell_order[1],) * SOURCES_PER_CELL
    )
    observed_cells = tuple(
        _row_coordinates(by_source[source][BRANCHES[0]])[3]
        for source in source_order
    )
    if observed_cells != expected_cells:
        fail("canary executable schedule source/cell order differs")
    for source in source_order:
        branches = by_source[source]
        if {
            _row_coordinates(branches[branch])[3] for branch in BRANCHES
        } != {_row_coordinates(branches[BRANCHES[0]])[3]}:
            fail("canary executable source branches bind different teachers")
    expected_order = tuple(
        (source, branch)
        for source in source_order
        for branch in BRANCHES
    )
    observed_order = tuple(
        (_row_coordinates(value)[1], _row_coordinates(value)[2])
        for value in values
    )
    if observed_order != expected_order:
        fail("canary executable rows must follow source/action/incomplete order")
    return (
        values,
        source_order,
        tuple(cell_order),
        MappingProxyType(row_contract),
    )


def _tail_source_id(ordinal: int) -> str:
    return f"{_TAIL_SOURCE_PREFIX}{ordinal:02d}"


def _tail_cell_id(ordinal: int) -> str:
    return f"{_TAIL_CELL_PREFIX}{ordinal:02d}"


def is_serialization_tail_source_v1(source_id: Any) -> bool:
    return type(source_id) is str and source_id.startswith(_TAIL_SOURCE_PREFIX)


def build_checkpoint_scaffold_schedule_v1(
    rows: Iterable[Any],
    *,
    run_seed: int,
    learning_module: Any,
    checkpoint_module: Any,
) -> tuple[Any, ...]:
    """Build a canonical checkpoint schedule with only a 16-row live prefix."""

    if type(run_seed) is not int or not 0 <= run_seed < 2**64:
        fail("canary run_seed must be an unsigned 64-bit integer")
    values, real_sources, real_cells, _row_contract = _validate_executable_rows(rows)
    action_row_type = getattr(learning_module, "ActionPairRow", None)
    scheduled_type = getattr(learning_module, "ScheduledActionPair", None)
    noise_seed = getattr(learning_module, "_noise_seed", None)
    if not callable(action_row_type) or not callable(scheduled_type) or not callable(noise_seed):
        fail("frozen learning schedule API differs")

    real_by_source: dict[str, dict[str, Any]] = defaultdict(dict)
    for value in values:
        _row_id, source_id, branch, _cell = _row_coordinates(value)
        real_by_source[source_id][branch] = value
    tail_sources = tuple(_tail_source_id(index) for index in range(TAIL_SOURCES))
    tail_cells = tuple(_tail_cell_id(index) for index in range(TAIL_TEACHER_CELLS))
    all_sources = (*real_sources, *tail_sources)
    _require(len(set(all_sources)) == CHECKPOINT_SOURCES, "canary scaffold source IDs collide")
    _require(not any(is_serialization_tail_source_v1(item) for item in real_sources), "canary real source aliases tail namespace")

    cell_by_source: dict[str, str] = {
        source: real_cells[index // SOURCES_PER_CELL]
        for index, source in enumerate(real_sources)
    }
    # Eight dummy sources fill the unused four formal-capacity slots in each
    # of the two real teacher cells; six tail cells then receive eight each.
    for index, source in enumerate(tail_sources[:8]):
        cell_by_source[source] = real_cells[index // SOURCES_PER_CELL]
    for index, source in enumerate(tail_sources[8:]):
        cell_by_source[source] = tail_cells[index // 8]
    _require(
        Counter(cell_by_source.values())
        == Counter({cell: 8 for cell in (*real_cells, *tail_cells)}),
        "canary scaffold teacher capacity differs",
    )

    by_source: dict[str, dict[str, Any]] = {}
    for source in all_sources:
        if source in real_by_source:
            by_source[source] = real_by_source[source]
            continue
        by_source[source] = {
            branch: action_row_type(
                row_id=f"serialization-tail-row:{source}:{branch}",
                source_id=source,
                branch=branch,
                teacher_cell_id=cell_by_source[source],
            )
            for branch in BRANCHES
        }

    result: list[Any] = []
    for epoch in range(10):
        for source_position, source in enumerate(all_sources):
            update = epoch * 16 + source_position // 4
            microbatch = source_position % 4
            sigma_index = SIGMA_INDICES[
                (epoch * CHECKPOINT_SOURCES + source_position)
                % len(SIGMA_INDICES)
            ]
            seed = noise_seed(run_seed, epoch, source, sigma_index)
            branch_order = (
                BRANCHES
                if (epoch + microbatch) % 2 == 0
                else tuple(reversed(BRANCHES))
            )
            for dp_rank, branch in enumerate(branch_order):
                global_index = update * GLOBAL_BATCH + microbatch * 2 + dp_rank
                result.append(
                    scheduled_type(
                        global_index=global_index,
                        epoch=epoch,
                        update=update,
                        microbatch=microbatch,
                        dp_rank=dp_rank,
                        sigma_index=sigma_index,
                        noise_seed=seed,
                        row=by_source[source][branch],
                    )
                )
    schedule = tuple(result)
    try:
        canonical = checkpoint_module.canonical_schedule_v2(schedule)
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"canary checkpoint scaffold is not canonical_schedule_v2: {error}"
        ) from error
    _require(len(canonical) == CHECKPOINT_FLAT_ROWS, "canary checkpoint scaffold length differs")
    executable = canonical[:EXECUTABLE_FLAT_ROWS]
    _require(
        not any(is_serialization_tail_source_v1(row["row"]["source_id"]) for row in executable),
        "canary executable prefix contains a tail source",
    )
    _require(
        all(is_serialization_tail_source_v1(row["row"]["source_id"]) for row in canonical[EXECUTABLE_FLAT_ROWS:GLOBAL_BATCH * 3]),
        "canary u3 boundary does not begin with unauthorized tail rows",
    )
    for update in range(MAX_UPDATES):
        group = executable[update * GLOBAL_BATCH : (update + 1) * GLOBAL_BATCH]
        _require(
            Counter(row["row"]["branch"] for row in group)
            == Counter({"action": 4, "incomplete": 4}),
            f"canary u{update + 1} branch closure differs",
        )
        _require(
            len({row["row"]["source_id"] for row in group}) == 4,
            f"canary u{update + 1} repeats a source",
        )
    _require(
        len({row["row"]["source_id"] for row in executable}) == SOURCE_UNITS,
        "canary u1/u2 source rows repeat",
    )
    return schedule


def schedule_authority_receipt_v1(
    schedule: Sequence[Any], *, checkpoint_module: Any
) -> Mapping[str, Any]:
    try:
        canonical = checkpoint_module.canonical_schedule_v2(schedule)
        full_sha, prefix_sha = checkpoint_module.schedule_digests_v2(
            canonical, MAX_UPDATES
        )
    except Exception as error:
        raise Full30MechanismCanaryAuthorityError(
            f"canary schedule receipt cannot canonicalize scaffold: {error}"
        ) from error
    executable = canonical[:EXECUTABLE_FLAT_ROWS]
    value = {
        "schema_version": SCHEDULE_AUTHORITY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "population_profile": POPULATION_PROFILE,
        "schedule_full_sha256": full_sha,
        "schedule_prefix_u2_sha256": prefix_sha,
        "checkpoint_flat_rows": len(canonical),
        "executable_flat_rows": EXECUTABLE_FLAT_ROWS,
        "tail_flat_rows": len(canonical) - EXECUTABLE_FLAT_ROWS,
        "executable_global_indices": list(range(EXECUTABLE_FLAT_ROWS)),
        "executable_source_ids": [
            row["row"]["source_id"] for row in executable[::2]
        ],
        "maximum_updates": MAX_UPDATES,
        "u3_authorized": False,
        "tail_serialization_only": True,
        "formal_authority": False,
        "mechanism_only": True,
        "generalization": False,
        "identity_generalization": False,
        "event_family_generalization": False,
        "synthetic_target_bytes_read": False,
        "synthetic_target_index1_bytes_read": False,
    }
    return MappingProxyType(
        {**value, "schedule_authority_digest": object_sha256(value)}
    )


def authorize_scheduled_row_v1(
    scheduled: Any,
    *,
    admitted_pairs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject the scaffold tail before any authority payload can be opened."""

    try:
        global_index = scheduled.global_index
        update = scheduled.update
        row_id, source_id, branch, teacher_cell_id = _row_coordinates(scheduled)
    except (AttributeError, TypeError) as error:
        raise Full30MechanismCanaryAuthorityError(
            "canary scheduled row is malformed"
        ) from error
    if (
        type(global_index) is not int
        or type(update) is not int
        or not 0 <= global_index < EXECUTABLE_FLAT_ROWS
        or not 0 <= update < MAX_UPDATES
        or is_serialization_tail_source_v1(source_id)
    ):
        fail(
            "canary schedule tail is serialization-only; u3/source/condition/teacher access is forbidden"
        )
    pair = admitted_pairs.get(row_id)
    if pair is None or (
        pair.get("source_iid"),
        pair.get("branch"),
        pair.get("teacher_cell_id"),
    ) != (source_id, branch, teacher_cell_id):
        fail("canary scheduled row differs from admitted reduced pair authority")


__all__ = [
    "AdmittedMechanismCanaryMaterializationPlanV1",
    "AMPLITUDE_BUNDLES",
    "AMPLITUDE_SCHEMA_VERSION",
    "EXECUTABLE_FLAT_ROWS",
    "EXPERIMENT_ID",
    "Full30MechanismCanaryAuthorityError",
    "MATERIALIZER_PLAN_ADMISSION_SCHEMA_VERSION",
    "MATERIALIZER_PLAN_BINDING_SCHEMA_VERSION",
    "MAX_UPDATES",
    "PAIR_ROWS",
    "POPULATION_PROFILE",
    "REPRESENTATION_BUNDLES",
    "SCHEMA_VERSION",
    "SOURCE_UNITS",
    "TEACHER_SEED_BINDING_SCHEMA_VERSION",
    "TEACHER_SEED_CANDIDATE_BINDING_SCHEMA_VERSION",
    "ValidatedMechanismCanaryAuthorityV1",
    "ValidatedMechanismCanaryDataAuthorityV1",
    "admit_reduced_materialization_plan_v1",
    "authorize_scheduled_row_v1",
    "build_checkpoint_scaffold_schedule_v1",
    "canonical_json_bytes",
    "is_serialization_tail_source_v1",
    "load_amplitude_authority_v1",
    "load_data_authority_v1",
    "load_mechanism_canary_authority_v1",
    "object_sha256",
    "schedule_authority_receipt_v1",
]
