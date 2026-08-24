#!/usr/bin/env python3
"""CPU-only verifier for the sanitized V10-A2 provisional P0 source list.

Passing this verifier proves only that the checked-in 64-row provisional
evidence is intact, sanitized, deterministically selected, and exactly
disjoint from the byte-pinned 16-row actual manifest by UUID, path, and media
SHA-256.  It cannot inspect the remote media, construct perceptual clusters,
qualify parent/part/interaction tracks, promote the list to an official P0
registry, authorize a GPU, or launch training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, NoReturn, Sequence


SCHEMA = "bernini-v10a2-p0-source-only-64-provisional-v1"
REGISTRY_ID = "v10a2_p0_source_only_64_provisional_v1"
ONLY_STATUS = "PROVISIONAL_64_INTEGRITY_ONLY_P0_AUTHORITY_NO"
RECEIPT_SCHEMA = "bernini-v10a2-p0-source-only-64-provisional-preflight-v1"
RECEIPT_STATUS = "PROVISIONAL_64_INTEGRITY_PASS_P0_AUTHORITY_NO"
V10A2_BLOCKER = (
    "PROVISIONAL_64_EXISTS_BUT_PERCEPTUAL_AND_OBSERVER_QUALIFICATION_MISSING"
)

METHOD_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = (
    METHOD_ROOT / "assets" / "v10a2_p0_source_only_64_provisional_v1.json"
)
DEFAULT_ACTUAL_MANIFEST_PATH = (
    METHOD_ROOT / "assets" / "target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json"
)

EXPECTED_REGISTRY_FILE_SHA256 = (
    "bc0779e5e75f9e1b9f9f49369d01d5e71d8c46765a6acea67b2f5354cf7aa5a6"
)
EXPECTED_REGISTRY_SELF_SHA256 = (
    "21cf6442089de3f1c2a96489c5eee60caf469394a720a25463bdb898dd3988d7"
)
EXPECTED_NO_TARGET_JSONL_SHA256 = (
    "592eb3a6f778316e1ac3caf8975c19816d93636b81a5ea706f6c9f8dd14ffdd7"
)
EXPECTED_MEV_JSON_SHA256 = (
    "2a6a06c38dda5a40e629ecdc8800e27495c214754e480344f267f939ad07fd19"
)
EXPECTED_REMOTE_PROPOSAL_SHA256 = (
    "9273614deedd7090962605a9d04f890bc90513456c3dcaf0b5c4df448b79c068"
)
EXPECTED_ACTUAL_MANIFEST_FILE_SHA256 = (
    "d43ff7f7c14b2c25bf949798fb71839f6f0e6325d8829784f2d9eef5a1516929"
)
EXPECTED_ACTUAL_MANIFEST_SELF_SHA256 = (
    "231da71f38bdd982a9276b02fb3351200b372563cfdded39fb7f7f6f7de93446"
)
EXPECTED_ACTUAL_SCHEMA = "bernini-target-factorized-soft-ot-graph-manifest-v5-r1b"
EXPECTED_ACTUAL_EXPERIMENT = "target_factorized_soft_ot_graph_teacher_pilot_v5_r1b"

SELECTION_SEED = b"v10a2-p0-provisional-source-only-v1"
REMOTE_VIDEO_ROOT = PurePosixPath(
    "/vast/users/guangyi.chen/dataset/MEV/MEV/videos"
)
STRATA = (
    "camera0_occlusion0",
    "camera0_occlusion1",
    "camera1_occlusion0",
    "camera1_occlusion1",
)
ELIGIBLE_COUNTS = {
    "camera0_occlusion0": 93,
    "camera0_occlusion1": 21,
    "camera1_occlusion0": 115,
    "camera1_occlusion1": 54,
}
KNOWN_UNRESOLVED = (
    "CROSS_UUID_MEDIA_ANCESTOR_PROVENANCE_MISSING",
    "TARGET_BLIND_PERCEPTUAL_CLUSTER_EXCLUSION_MISSING",
    "FROZEN_OBSERVER_PARENT_PART_INTERACTION_QUALIFICATION_MISSING",
    "OFFICIAL_SOURCE_ONLY_REGISTRAR_RELEASE_MISSING",
)
FORBIDDEN_KEY_FRAGMENTS = (
    "prompt",
    "caption",
    "instruction",
    "action_text",
    "interaction_family",
    "target_video_path",
    "target_media_sha256",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class V10A2P0ProvisionalPreflightError(RuntimeError):
    """Raised when provisional evidence differs or overclaims authority."""


def fail(message: str) -> NoReturn:
    raise V10A2P0ProvisionalPreflightError(message)


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
        raise V10A2P0ProvisionalPreflightError(
            "value is not canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    fail(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"JSON authority must be one regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except V10A2P0ProvisionalPreflightError:
        raise
    except Exception as error:
        raise V10A2P0ProvisionalPreflightError(
            f"cannot parse {path}: {error}"
        ) from error
    if type(value) is not dict:
        fail(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        fail(f"{label} must be an array")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _verify_self_hash(
    value: Mapping[str, Any], key: str, label: str
) -> str:
    expected = _sha256(value.get(key), f"{label}.{key}")
    payload = dict(value)
    payload.pop(key, None)
    actual = object_sha256(payload)
    if actual != expected:
        fail(f"{label} self hash differs")
    return actual


def _scan_forbidden_keys(value: Any, path: str = "registry") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if any(fragment in key.lower() for fragment in FORBIDDEN_KEY_FRAGMENTS):
                fail(f"sanitized registry contains forbidden text-bearing key: {path}.{key}")
            _scan_forbidden_keys(child, f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{path}[{index}]")


def _expected_rank(uuid: str) -> str:
    return hashlib.sha256(
        SELECTION_SEED + bytes([0]) + uuid.encode("ascii")
    ).hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be a finite number")
    return result


def _validate_actual_manifest(
    value: Mapping[str, Any], *, observed_file_sha256: str
) -> tuple[set[str], set[str], set[str]]:
    if observed_file_sha256 != EXPECTED_ACTUAL_MANIFEST_FILE_SHA256:
        fail("actual manifest bytes differ")
    if value.get("schema_version") != EXPECTED_ACTUAL_SCHEMA:
        fail("actual manifest schema differs")
    if value.get("experiment_id") != EXPECTED_ACTUAL_EXPERIMENT:
        fail("actual manifest experiment differs")
    digest = _verify_self_hash(value, "manifest_sha256", "actual manifest")
    if digest != EXPECTED_ACTUAL_MANIFEST_SELF_SHA256:
        fail("actual manifest self hash differs from the independent pin")
    pairs = _array(value.get("pairs"), "actual manifest.pairs")
    if len(pairs) != 16:
        fail("actual manifest must contain exactly 16 pairs")
    uuids: set[str] = set()
    paths: set[str] = set()
    media_sha256: set[str] = set()
    for ordinal, raw in enumerate(pairs):
        row = _mapping(raw, "actual manifest.pairs[]")
        if row.get("ordinal") != ordinal:
            fail("actual manifest ordinal order differs")
        uuid = row.get("uuid")
        if not isinstance(uuid, str) or UUID_RE.fullmatch(uuid) is None:
            fail("actual manifest UUID differs")
        if row.get("formal_sft_authorized") is not False:
            fail("actual manifest unexpectedly authorizes fitting")
        uuids.add(uuid)
        for role in ("source", "target"):
            path = row.get(f"{role}_video_path")
            media = _mapping(row.get(f"{role}_media"), f"actual {role}_media")
            if not isinstance(path, str):
                fail(f"actual {role} path differs")
            paths.add(path)
            media_sha256.add(_sha256(media.get("sha256"), f"actual {role} media"))
    if len(uuids) != 16 or len(paths) != 32 or len(media_sha256) != 32:
        fail("actual manifest UUID/path/media uniqueness differs")
    return uuids, paths, media_sha256


def validate_provisional_registry(
    value: Mapping[str, Any],
    *,
    observed_registry_file_sha256: str,
    actual_manifest: Mapping[str, Any],
    actual_manifest_file_sha256: str,
) -> Mapping[str, Any]:
    if observed_registry_file_sha256 != EXPECTED_REGISTRY_FILE_SHA256:
        fail("provisional registry bytes differ")
    if set(value) != {
        "schema_version",
        "registry_id",
        "status",
        "authority",
        "source_pins",
        "selection_contract",
        "actual_exclusion_audit",
        "known_unresolved",
        "v10a2_blocker",
        "rows",
        "provisional_registry_sha256",
    }:
        fail("provisional registry top-level schema differs")
    if value.get("schema_version") != SCHEMA or value.get("registry_id") != REGISTRY_ID:
        fail("provisional registry identity differs")
    if value.get("status") != ONLY_STATUS:
        fail("provisional registry cannot emit a ready or authority status")
    digest = _verify_self_hash(
        value, "provisional_registry_sha256", "provisional registry"
    )
    if digest != EXPECTED_REGISTRY_SELF_SHA256:
        fail("provisional registry is not the independently pinned revision")
    _scan_forbidden_keys(value)

    authority = _mapping(value.get("authority"), "authority")
    if authority != {
        "provisional_evidence_only": True,
        "official_source_only_registry": False,
        "perceptual_exclusion_complete": False,
        "frozen_observer_qualification_complete": False,
        "p0_slot_pretraining_authorized": False,
        "gpu_launch_authorized": False,
        "training_authorized": False,
        "parameter_updates_authorized": False,
        "can_be_promoted_by_rehash": False,
    }:
        fail("provisional authority must remain hard P0 NO")

    source_pins = _mapping(value.get("source_pins"), "source_pins")
    if source_pins != {
        "no_target_source_jsonl": {
            "path": "/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction/metadata_annotation_v2/no_target_sources_annotation_v2.jsonl",
            "file_sha256": EXPECTED_NO_TARGET_JSONL_SHA256,
            "schema_version": "mev-action-edit-no-target-annotation-v2",
            "row_count": 8015,
        },
        "mev_json": {
            "path": "/vast/users/guangyi.chen/dataset/MEV/MEV/annotations/mev.json",
            "file_sha256": EXPECTED_MEV_JSON_SHA256,
            "video_count": 8016,
        },
    }:
        fail("source JSONL/mev.json pins differ")

    selection = _mapping(value.get("selection_contract"), "selection_contract")
    if selection.get("seed_ascii") != SELECTION_SEED.decode("ascii"):
        fail("selection seed differs")
    if selection.get("rank_separator_hex") != "00" or selection.get("rank_form") != "sha256(seed_ascii_bytes || 0x00 || uuid_ascii_bytes)":
        fail("selection rank contract differs")
    if (
        selection.get("source_split") != "train"
        or selection.get("source_event_id") != 1
        or selection.get("target_must_be_null") is not True
        or selection.get("actual_uuid_exclusion_count") != 16
        or selection.get("eligible_count") != 283
        or selection.get("selected_count") != 64
        or selection.get("selection_order") != list(STRATA)
        or selection.get("tie_break") != ["selection_rank_sha256", "uuid"]
    ):
        fail("selection count/order/source-only contract differs")
    expected_eligibility = {
        "multi_person": True,
        "focus_object_present": True,
        "duration_seconds_min": 4.0,
        "duration_seconds_max": 10.0,
        "aesthetic_quality_min": 0.5,
        "imaging_quality_min": 0.6,
        "subject_consistency_min": 0.9,
        "background_consistency_min": 0.9,
        "motion_smoothness_min": 0.98,
        "temporal_flickering_min": 0.97,
        "appearance_change": False,
        "disappearance": False,
        "lighting_change": False,
        "camera_motion_must_be_boolean": True,
        "occlusion_must_be_boolean": True,
        "regular_non_symlink_under_pinned_video_root": True,
    }
    if selection.get("eligibility") != expected_eligibility:
        fail("selection eligibility thresholds differ")
    strata_rows = _array(selection.get("strata"), "selection_contract.strata")
    if strata_rows != [
        {
            "id": stratum,
            "eligible_count": ELIGIBLE_COUNTS[stratum],
            "selected_count": 16,
        }
        for stratum in STRATA
    ]:
        fail("eligible or selected stratum counts differ")

    actual_uuids, actual_paths, actual_media = _validate_actual_manifest(
        actual_manifest, observed_file_sha256=actual_manifest_file_sha256
    )
    audit = _mapping(value.get("actual_exclusion_audit"), "actual_exclusion_audit")
    if audit != {
        "manifest_relative_path": "target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json",
        "manifest_file_sha256": EXPECTED_ACTUAL_MANIFEST_FILE_SHA256,
        "manifest_self_sha256": EXPECTED_ACTUAL_MANIFEST_SELF_SHA256,
        "actual_pair_count": 16,
        "actual_media_count": 32,
        "candidate_uuid_unique_count": 64,
        "candidate_path_unique_count": 64,
        "candidate_media_sha256_unique_count": 64,
        "candidate_actual_uuid_overlap_count": 0,
        "candidate_actual_path_overlap_count": 0,
        "candidate_actual_media_sha256_overlap_count": 0,
        "remote_readonly_proposal_digest_sha256": EXPECTED_REMOTE_PROPOSAL_SHA256,
    }:
        fail("actual exclusion audit contract differs")
    if value.get("known_unresolved") != list(KNOWN_UNRESOLVED):
        fail("known unresolved qualification registry differs")
    if value.get("v10a2_blocker") != V10A2_BLOCKER:
        fail("V10-A2 blocker differs")

    rows = _array(value.get("rows"), "rows")
    if len(rows) != 64:
        fail("provisional registry must contain exactly 64 rows")
    uuids: list[str] = []
    paths: list[str] = []
    media_sha256: list[str] = []
    observed_strata: list[str] = []
    ranks_by_stratum: dict[str, list[tuple[str, str]]] = {
        stratum: [] for stratum in STRATA
    }
    expected_row_keys = {
        "ordinal",
        "uuid",
        "source_video_path",
        "size_bytes",
        "source_media_sha256",
        "stratum",
        "quality",
        "selection_rank_sha256",
    }
    expected_quality_keys = {
        "duration_seconds",
        "aesthetic_quality",
        "imaging_quality",
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "temporal_flickering",
        "multi_person",
        "focus_object_present",
        "appearance_change",
        "disappearance",
        "lighting_change",
        "camera_motion",
        "occlusion",
    }
    for ordinal, raw in enumerate(rows):
        row = _mapping(raw, f"rows[{ordinal}]")
        if set(row) != expected_row_keys or row.get("ordinal") != ordinal:
            fail(f"candidate row schema/ordinal differs: {ordinal}")
        uuid = row.get("uuid")
        if not isinstance(uuid, str) or UUID_RE.fullmatch(uuid) is None:
            fail(f"candidate UUID differs: {ordinal}")
        path_text = row.get("source_video_path")
        if not isinstance(path_text, str):
            fail(f"candidate path differs: {ordinal}")
        path = PurePosixPath(path_text)
        if path.parent != REMOTE_VIDEO_ROOT or path.name.startswith(uuid + "-") is False:
            fail(f"candidate path is outside the pinned UUID media root: {ordinal}")
        event1_name = re.compile(
            re.escape(uuid)
            + r"-\d{2}\.\d{2}\.\d{2}\.\d{3}"
            + r"-\d{2}\.\d{2}\.\d{2}\.\d{3}-seg0*1\.mp4"
        )
        if event1_name.fullmatch(path.name) is None or ".." in path.parts:
            fail(f"candidate path is not one event-1 MP4: {ordinal}")
        if type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0:
            fail(f"candidate size differs: {ordinal}")
        media_digest = _sha256(
            row.get("source_media_sha256"), f"rows[{ordinal}].source_media_sha256"
        )
        rank = _sha256(
            row.get("selection_rank_sha256"),
            f"rows[{ordinal}].selection_rank_sha256",
        )
        if rank != _expected_rank(uuid):
            fail(f"candidate selection rank differs: {ordinal}")
        stratum = row.get("stratum")
        if stratum not in STRATA:
            fail(f"candidate stratum differs: {ordinal}")
        quality = _mapping(row.get("quality"), f"rows[{ordinal}].quality")
        if set(quality) != expected_quality_keys:
            fail(f"candidate quality schema differs: {ordinal}")
        camera_motion = quality.get("camera_motion")
        occlusion = quality.get("occlusion")
        if type(camera_motion) is not bool or type(occlusion) is not bool:
            fail(f"candidate camera/occlusion is not boolean: {ordinal}")
        if stratum != f"camera{int(camera_motion)}_occlusion{int(occlusion)}":
            fail(f"candidate stratum/quality mismatch: {ordinal}")
        for key, expected in {
            "multi_person": True,
            "focus_object_present": True,
            "appearance_change": False,
            "disappearance": False,
            "lighting_change": False,
        }.items():
            if quality.get(key) is not expected:
                fail(f"candidate structural quality differs: {ordinal}/{key}")
        bounded = {
            "duration_seconds": (4.0, 10.0),
            "aesthetic_quality": (0.5, 1.0),
            "imaging_quality": (0.6, 1.0),
            "subject_consistency": (0.9, 1.0),
            "background_consistency": (0.9, 1.0),
            "motion_smoothness": (0.98, 1.0),
            "temporal_flickering": (0.97, 1.0),
        }
        for key, (minimum, maximum) in bounded.items():
            number = _number(quality.get(key), f"rows[{ordinal}].quality.{key}")
            if number < minimum or (maximum is not None and number > maximum):
                fail(f"candidate quality threshold fails: {ordinal}/{key}")
        uuids.append(uuid)
        paths.append(path_text)
        media_sha256.append(media_digest)
        observed_strata.append(stratum)
        ranks_by_stratum[stratum].append((rank, uuid))
    if len(set(uuids)) != 64:
        fail("candidate UUIDs are not unique")
    if len(set(paths)) != 64:
        fail("candidate paths are not unique")
    if len(set(media_sha256)) != 64:
        fail("candidate media SHA-256 values are not unique")
    if Counter(observed_strata) != Counter({stratum: 16 for stratum in STRATA}):
        fail("candidate selected strata are not exactly 4 x 16")
    expected_order = [stratum for stratum in STRATA for _ in range(16)]
    if observed_strata != expected_order:
        fail("candidate stratum block order differs")
    for stratum in STRATA:
        if ranks_by_stratum[stratum] != sorted(ranks_by_stratum[stratum]):
            fail(f"candidate deterministic rank order differs: {stratum}")
    if set(uuids) & actual_uuids:
        fail("candidate UUID leaks into the actual split")
    if set(paths) & actual_paths:
        fail("candidate path leaks into the actual split")
    if set(media_sha256) & actual_media:
        fail("candidate media SHA-256 leaks into the actual split")
    return value


def build_preflight_receipt(
    *,
    registry: Mapping[str, Any],
    registry_file_sha256: str,
    actual_manifest: Mapping[str, Any],
    actual_manifest_file_sha256: str,
) -> Mapping[str, Any]:
    validate_provisional_registry(
        registry,
        observed_registry_file_sha256=registry_file_sha256,
        actual_manifest=actual_manifest,
        actual_manifest_file_sha256=actual_manifest_file_sha256,
    )
    value = {
        "schema_version": RECEIPT_SCHEMA,
        "registry_id": REGISTRY_ID,
        "status": RECEIPT_STATUS,
        "integrity_verified": True,
        "sanitized_field_contract_verified": True,
        "candidate_count": 64,
        "eligible_count_recorded_not_recomputed_locally": 283,
        "selected_strata": {stratum: 16 for stratum in STRATA},
        "exact_uuid_path_media_overlap_with_actual": 0,
        "remote_media_bytes_revalidated_by_this_local_validator": False,
        "perceptual_exclusion_complete": False,
        "frozen_observer_qualification_complete": False,
        "official_source_only_registry": False,
        "p0_slot_pretraining_authorized": False,
        "gpu_launch_authorized": False,
        "training_authorized": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "v10a2_blocker": V10A2_BLOCKER,
        "known_unresolved": list(KNOWN_UNRESOLVED),
        "registry_file_sha256": registry_file_sha256,
        "registry_self_sha256": EXPECTED_REGISTRY_SELF_SHA256,
        "actual_manifest_file_sha256": actual_manifest_file_sha256,
    }
    return {**value, "receipt_sha256": object_sha256(value)}


def run_preflight(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    actual_manifest_path: Path = DEFAULT_ACTUAL_MANIFEST_PATH,
) -> Mapping[str, Any]:
    registry = load_json(registry_path)
    actual_manifest = load_json(actual_manifest_path)
    return build_preflight_receipt(
        registry=registry,
        registry_file_sha256=file_sha256(registry_path),
        actual_manifest=actual_manifest,
        actual_manifest_file_sha256=file_sha256(actual_manifest_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--actual-manifest", type=Path, default=DEFAULT_ACTUAL_MANIFEST_PATH
    )
    args = parser.parse_args(argv)
    receipt = run_preflight(args.registry, args.actual_manifest)
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ACTUAL_MANIFEST_PATH",
    "DEFAULT_REGISTRY_PATH",
    "EXPECTED_REGISTRY_FILE_SHA256",
    "EXPECTED_REGISTRY_SELF_SHA256",
    "KNOWN_UNRESOLVED",
    "ONLY_STATUS",
    "RECEIPT_STATUS",
    "STRATA",
    "V10A2P0ProvisionalPreflightError",
    "V10A2_BLOCKER",
    "build_preflight_receipt",
    "canonical_json_bytes",
    "file_sha256",
    "load_json",
    "main",
    "object_sha256",
    "run_preflight",
    "validate_provisional_registry",
]
