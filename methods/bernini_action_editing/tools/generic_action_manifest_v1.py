#!/usr/bin/env python3
"""Build and validate the two generic action-training manifests.

The builder is deliberately downstream of representation extraction and
independent full-video review.  It cannot create a quotient, infer q0/q1,
write a visual label, or make an unreviewed sidecar trainable.  Missing or
legacy evidence therefore produces a non-zero exit and no output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Iterable, Mapping, Optional, Sequence


REPRESENTATION_SCHEMA = "bernini-representation-train-manifest-v1"
PAIR_SCHEMA = "bernini-action-source-pair-manifest-v1"
EVIDENCE_SCHEMA = "bernini-phi-v1-reviewed-evidence-index-v1"
SIDECAR_SCHEMA = "bernini-phi-v1-representation-sidecar-receipt-v1"
REVIEW_SCHEMA = "bernini-independent-full81-action-review-v1"
Q0_SCHEMA = "bernini-generic-action-q0-source-authority-v1"

AUTHORING_SHA256 = "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
POPULATION_SHA256 = "71906510d162e6626338b5785fd1cf55b437de5ba77d9b9b122ad761694f8e62"
BRANCH_ORDER = ("action", "noop", "reverse", "incomplete")
REVIEW_BRANCHES = (
    "action", "noop", "incomplete", "reverse", "shuffle", "wrong_actor",
    "wrong_object", "camera_only", "appearance_only", "generic_wrong_motion",
)
NONNOOP = frozenset({"action", "reverse", "incomplete"})
OPERATOR_BRANCHES = frozenset({"action", "incomplete"})
SPLIT_ORDER = ("fit", "confirmation")
PHASE_ORDER = ("onset", "transition", "terminal", "hold")
ALLOWED_REVIEW_METHODS = frozenset(
    {
        "human_blind_video_review_v1",
        "fixed_vlm_blind_video_review_v1",
        "dual_consensus_blind_video_review_v1",
    }
)
P32_SEED = 2026081401
PHI_BLOCK = 22
PHI_SCHEDULE_INDEX = 29
PHASES = 21
CODE_WIDTH = 32
HIDDEN_WIDTH = 1536
RAW_CODE_BYTES = PHASES * CODE_WIDTH * 4
RAW_P32_BYTES = HIDDEN_WIDTH * CODE_WIDTH * 4
_SHA = re.compile(r"^[0-9a-f]{64}$")
_IID = re.compile(r"^[0-9a-f]{16}$")
_ROW_ID = re.compile(r"^gaav1:(fit|confirmation):[0-9a-f]{16}:s[0-9]+:(action|noop|reverse|incomplete)$")


class GenericActionManifestError(RuntimeError):
    """Raised before an ambiguous row or byte can enter training."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GenericActionManifestError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenericActionManifestError(message)


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise GenericActionManifestError(f"{label} must be lowercase SHA-256")
    return value


def _closed(value: Any, fields: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise GenericActionManifestError(
            f"{label} field closure differs: observed={observed!r}, expected={sorted(expected)!r}"
        )
    return value


def _plain_file(value: Any, label: str) -> Path:
    if type(value) is str:
        path = Path(value)
    elif isinstance(value, Path):
        path = value
    else:
        raise GenericActionManifestError(f"{label} path must be text or Path")
    if not path.is_absolute():
        raise GenericActionManifestError(f"{label} path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise GenericActionManifestError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GenericActionManifestError(f"{label} must be a plain non-symlink file")
    return path.resolve(strict=True)


def _load_json(path: str | Path, label: str, expected_sha256: Optional[str] = None) -> dict[str, Any]:
    source = _plain_file(path, label)
    raw = source.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != _sha(expected_sha256, label):
        raise GenericActionManifestError(f"{label} file SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise GenericActionManifestError(f"cannot decode {label}") from error
    if type(value) is not dict:
        raise GenericActionManifestError(f"{label} root must be an object")
    return value


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    unsigned = dict(value)
    return {**unsigned, field: object_sha256(unsigned)}


def _verify_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    declared = _sha(value.get(field), f"{label}.{field}")
    unsigned = dict(value)
    del unsigned[field]
    _require(object_sha256(unsigned) == declared, f"{label} digest differs")


def _utf8_binding(value: Any, label: str) -> Mapping[str, Any]:
    row = _closed(value, {"text", "utf8_sha256"}, label)
    text = row["text"]
    _require(type(text) is str and text.strip() == text and bool(text), f"{label}.text differs")
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _require(row["utf8_sha256"] == expected, f"{label} UTF-8 SHA-256 differs")
    return row


def validate_phase_labels(value: Any, label: str = "phase_labels") -> tuple[str, ...]:
    if type(value) is not list or len(value) != PHASES or any(type(item) is not str for item in value):
        raise GenericActionManifestError(f"{label} must contain exactly 21 strings")
    labels = tuple(value)
    _require(labels[0] == "onset", f"{label} phase zero must be onset")
    _require(set(labels) == set(PHASE_ORDER), f"{label} must contain all four registered phases")
    collapsed = tuple(item for index, item in enumerate(labels) if index == 0 or item != labels[index - 1])
    _require(collapsed == PHASE_ORDER, f"{label} must be contiguous onset->transition->terminal->hold")
    return labels


def validate_semantic_binding(value: Any, label: str = "semantic_binding") -> Mapping[str, Any]:
    row = _closed(value, {"q0_state", "q1_state", "owner", "object_contact"}, label)
    for field in row:
        _utf8_binding(row[field], f"{label}.{field}")
    return row


def _read_f32le(path: Path, expected_bytes: int, label: str) -> tuple[float, ...]:
    raw = path.read_bytes()
    _require(len(raw) == expected_bytes, f"{label} byte count differs")
    values = struct.unpack(f"<{expected_bytes // 4}f", raw)
    _require(all(math.isfinite(value) for value in values), f"{label} contains non-finite values")
    return values


def validate_code_tensor(binding: Any, *, is_noop: bool, label: str) -> Mapping[str, Any]:
    row = _closed(
        binding,
        {"path", "raw_sha256", "dtype", "byte_order", "shape", "normalization"},
        label,
    )
    _require(row["dtype"] == "float32", f"{label}.dtype differs")
    _require(row["byte_order"] == "little", f"{label}.byte_order differs")
    _require(row["shape"] == [PHASES, CODE_WIDTH], f"{label}.shape differs")
    expected_normalization = "exact_zero_not_normalized" if is_noop else "global_l2_unit"
    _require(row["normalization"] == expected_normalization, f"{label}.normalization differs")
    path = _plain_file(row["path"], label)
    _require(file_sha256(path) == _sha(row["raw_sha256"], label), f"{label} raw SHA-256 differs")
    raw = path.read_bytes()
    values = _read_f32le(path, RAW_CODE_BYTES, label)
    if is_noop:
        _require(raw == b"\x00" * RAW_CODE_BYTES, f"{label} noop bytes are not exact positive zero")
    else:
        _require(raw[: CODE_WIDTH * 4] == b"\x00" * (CODE_WIDTH * 4), f"{label} phase zero is not exact positive zero")
        norm = math.sqrt(sum(value * value for value in values))
        _require(abs(norm - 1.0) <= 5.0e-5, f"{label} global L2 norm differs: {norm}")
        for channel in range(CODE_WIDTH):
            mean = sum(values[phase * CODE_WIDTH + channel] for phase in range(1, PHASES)) / (PHASES - 1)
            _require(abs(mean) <= 2.0e-5, f"{label} temporal DC differs")
    return row


def validate_p32(path: Path, label: str = "P32") -> None:
    values = _read_f32le(path, RAW_P32_BYTES, label)
    for left in range(CODE_WIDTH):
        for right in range(left, CODE_WIDTH):
            product = sum(
                values[row * CODE_WIDTH + left] * values[row * CODE_WIDTH + right]
                for row in range(HIDDEN_WIDTH)
            )
            expected = 1.0 if left == right else 0.0
            _require(abs(product - expected) <= 2.0e-4, f"{label} columns are not orthonormal")


def validate_review_receipt(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    value = _load_json(path, "review receipt", expected_sha256)
    _closed(
        value,
        {
            "schema_version", "candidate_id", "branch", "media_sha256", "review_method",
            "entire_exact81_video_viewed", "frame_count", "fps",
            "reviewer_blinded_to_prompt_and_requested_branch",
            "sealed_before_phi_extraction", "quality_pass", "branch_semantics_pass",
            "phase_labels", "observations", "receipt_digest",
        },
        "review receipt",
    )
    _verify_seal(value, "receipt_digest", "review receipt")
    _require(value["schema_version"] == REVIEW_SCHEMA, "review schema differs")
    branch = value["branch"]
    _require(branch in REVIEW_BRANCHES, "review branch differs")
    _sha(value["media_sha256"], "review media")
    _require(value["review_method"] in ALLOWED_REVIEW_METHODS, "review method differs")
    for field in (
        "entire_exact81_video_viewed", "reviewer_blinded_to_prompt_and_requested_branch",
        "sealed_before_phi_extraction", "quality_pass", "branch_semantics_pass",
    ):
        _require(value[field] is True, f"review {field} is not true")
    _require(value["frame_count"] == 81 and value["fps"] == 25, "review media geometry differs")
    validate_phase_labels(value["phase_labels"], "review.phase_labels")
    observations = _closed(
        value["observations"],
        {"start_state_present", "transition_present", "requested_terminal_present", "terminal_hold_present", "full_target_event_present"},
        "review.observations",
    )
    expected = {
        "action": (True, True, True, True, True),
        "reverse": (True, True, True, True, True),
        "noop": (True, False, False, False, False),
        "incomplete": (True, True, False, False, False),
    }.get(branch, (True, False, False, False, False))
    observed = tuple(observations[field] for field in observations)
    canonical_fields = ("start_state_present", "transition_present", "requested_terminal_present", "terminal_hold_present", "full_target_event_present")
    observed = tuple(observations[field] for field in canonical_fields)
    _require(observed == expected, f"review branch observations differ for {branch}")
    return value


def validate_sidecar_receipt(
    path: str | Path,
    expected_sha256: str,
    *,
    require_admissible: bool = True,
    validated_p32: Optional[set[tuple[str, str]]] = None,
) -> Mapping[str, Any]:
    value = _load_json(path, "Phi_v1 sidecar receipt", expected_sha256)
    _closed(
        value,
        {
            "schema_version", "row_id", "candidate_id", "source_iid", "analysis_split", "seed", "branch",
            "phi_v1", "tensor", "nuisance_projection", "review_status",
            "generated_media_is_optimizer_input_or_target", "optimizer_authorized",
            "receipt_digest",
        },
        "Phi_v1 sidecar receipt",
    )
    _verify_seal(value, "receipt_digest", "Phi_v1 sidecar receipt")
    _require(value["schema_version"] == SIDECAR_SCHEMA, "sidecar schema differs")
    _require(type(value["row_id"]) is str and _ROW_ID.fullmatch(value["row_id"]), "sidecar row id differs")
    _require(type(value["source_iid"]) is str and _IID.fullmatch(value["source_iid"]), "sidecar IID differs")
    _require(value["analysis_split"] in SPLIT_ORDER and value["branch"] in BRANCH_ORDER, "sidecar split/branch differs")
    _require(type(value["seed"]) is int and value["seed"] > 0, "sidecar seed differs")
    phi = _closed(
        value["phi_v1"],
        {"hook", "block_index", "teacher_exact40_index", "sp_world", "sp_order", "append_padding_removed", "target_layout", "pooling", "phase0", "temporal_dc", "p32_seed", "p32_shape", "p32_raw_path", "p32_raw_sha256", "p32_generator_path", "p32_generator_source_sha256", "nuisance_order"},
        "sidecar.phi_v1",
    )
    _require(
        phi == {
            **phi,
            "hook": "transformer_1.blocks[22].output",
            "block_index": PHI_BLOCK,
            "teacher_exact40_index": PHI_SCHEDULE_INDEX,
            "sp_world": 4,
            "sp_order": "rank0_rank1_rank2_rank3_contiguous_global_target_indices",
            "append_padding_removed": True,
            "target_layout": "phase_major_21_then_patch_y_x",
            "pooling": "fixed_spatial_mean",
            "phase0": "exact_positive_zero",
            "temporal_dc": "phases_1_20_per_channel_mean_subtracted",
            "p32_seed": P32_SEED,
            "p32_shape": [HIDDEN_WIDTH, CODE_WIDTH],
            "nuisance_order": ["camera_only", "appearance_only_gram_schmidt_off_camera"],
        },
        "sidecar Phi_v1 contract differs",
    )
    p32_path = _plain_file(phi["p32_raw_path"], "sidecar P32")
    _require(p32_path.stat().st_size == RAW_P32_BYTES, "sidecar P32 byte count differs")
    p32_sha = _sha(phi["p32_raw_sha256"], "sidecar P32")
    _require(file_sha256(p32_path) == p32_sha, "sidecar P32 raw SHA-256 differs")
    p32_key = (str(p32_path), p32_sha)
    if validated_p32 is None or p32_key not in validated_p32:
        validate_p32(p32_path, "sidecar P32")
        if validated_p32 is not None:
            validated_p32.add(p32_key)
    generator_path = _plain_file(phi["p32_generator_path"], "sidecar P32 generator")
    _require(file_sha256(generator_path) == _sha(phi["p32_generator_source_sha256"], "sidecar P32 generator"), "sidecar P32 generator SHA-256 differs")
    is_noop = value["branch"] == "noop"
    validate_code_tensor(value["tensor"], is_noop=is_noop, label="sidecar.tensor")
    nuisance = _closed(
        value["nuisance_projection"],
        {"camera_raw_sha256", "appearance_raw_sha256", "camera_norm", "appearance_after_gs_norm", "pre_projection_norm", "post_projection_norm", "survival_cosine", "finite_non_degenerate"},
        "sidecar.nuisance_projection",
    )
    for field in ("camera_raw_sha256", "appearance_raw_sha256"):
        _sha(nuisance[field], f"sidecar nuisance {field}")
    for field in ("camera_norm", "appearance_after_gs_norm", "pre_projection_norm", "post_projection_norm", "survival_cosine"):
        _require(type(nuisance[field]) in (int, float) and math.isfinite(float(nuisance[field])), f"sidecar nuisance {field} differs")
    _require(nuisance["finite_non_degenerate"] is True, "sidecar nuisance projection degenerated")
    _require(value["generated_media_is_optimizer_input_or_target"] is False, "sidecar leaks generated media authority")
    _require(value["optimizer_authorized"] is False, "sidecar itself may not authorize optimizer")
    if require_admissible:
        _require(value["review_status"] == "PASS_SEALED_BEFORE_EXTRACTION", "sidecar review is not admissible")
    return value


def _expected_rows(authoring: Mapping[str, Any], population: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    _require(authoring.get("schema_version") == "pair-v5-pure-t2v-calibration-authoring-v1", "authoring schema differs")
    _require(authoring.get("expected_cell_count") == 8 and len(authoring.get("cells", [])) == 8, "authoring cell closure differs")
    profiles = {row["profile_id"]: row for row in population.get("inherited_bank_profiles", [])}
    inherited: dict[str, Mapping[str, Any]] = {}
    for family in population.get("action_families", []):
        for row in family.get("inherited_identity_scenes", []):
            inherited[row["source_iid"]] = row
    _require(set(inherited) == {row["iid"] for row in authoring["cells"]}, "population/authoring IID closure differs")
    cells = {row["iid"]: row for row in authoring["cells"]}
    expected: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        for iid in sorted(iid for iid, cell in cells.items() if cell["analysis_split"] == split):
            cell = cells[iid]
            population_row = inherited[iid]
            profile = profiles[population_row["source_bank_profile"]]
            seeds = population_row["seeds"]
            _require(len(seeds) == 2 and seeds[0] == cell["seed"], f"seed closure differs for {iid}")
            for seed_index, seed in enumerate(seeds):
                prefix = profile["seed1_candidate_prefix" if seed_index == 0 else "seed2_candidate_prefix"]
                for branch in BRANCH_ORDER:
                    expected.append(
                        {
                            "row_id": f"gaav1:{split}:{iid}:s{seed}:{branch}",
                            "candidate_id": f"{prefix}{iid}-{branch}",
                            "analysis_split": split,
                            "source_iid": iid,
                            "seed": seed,
                            "branch": branch,
                            "instruction_text": cell["branch_descriptions"][branch],
                            "source_path": cell["geometry_source_video"],
                        }
                    )
    _require(len(expected) == 64, "expected representation row closure differs")
    return tuple(expected)


def _validate_q0_authority(value: Mapping[str, Any], expected_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    _closed(
        value,
        {"schema_version", "authority_id", "authoring_registry", "read_only_sha256_audit", "rows", "generated_media_is_editor_input_or_target", "optimizer_authorized"},
        "q0 authority",
    )
    _require(value["schema_version"] == Q0_SCHEMA, "q0 authority schema differs")
    _require(value["generated_media_is_editor_input_or_target"] is False and value["optimizer_authorized"] is False, "q0 authority overclaims")
    rows = value["rows"]
    _require(type(rows) is list and len(rows) == 8, "q0 authority row count differs")
    by_iid: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _closed(row, {"iid", "analysis_split", "q0_source_video_path", "q0_source_video_sha256"}, "q0 row")
        _require(_IID.fullmatch(row["iid"]) is not None and row["iid"] not in by_iid, "q0 IID differs")
        _sha(row["q0_source_video_sha256"], "q0 video")
        _require(Path(row["q0_source_video_path"]).is_absolute(), "q0 path must be absolute")
        by_iid[row["iid"]] = row
    expected_iids = {row["source_iid"] for row in expected_rows}
    _require(set(by_iid) == expected_iids, "q0 IID closure differs")
    for row in expected_rows:
        source = by_iid[row["source_iid"]]
        _require(source["analysis_split"] == row["analysis_split"] and source["q0_source_video_path"] == row["source_path"], "q0 authoring binding differs")
    return by_iid


def build_manifests(
    *, authoring_path: str | Path, population_path: str | Path, evidence_index_path: str | Path,
    q0_authority_path: str | Path, representation_output: str | Path, pair_output: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authoring = _load_json(authoring_path, "authoring registry", AUTHORING_SHA256)
    population = _load_json(population_path, "population registry", POPULATION_SHA256)
    expected = _expected_rows(authoring, population)
    q0_raw = _load_json(q0_authority_path, "q0 authority")
    q0 = _validate_q0_authority(q0_raw, expected)
    evidence = _load_json(evidence_index_path, "reviewed evidence index")
    _closed(evidence, {"schema_version", "rows", "index_digest"}, "evidence index")
    _verify_seal(evidence, "index_digest", "evidence index")
    _require(evidence["schema_version"] == EVIDENCE_SCHEMA, "evidence index schema differs")
    rows = evidence["rows"]
    _require(type(rows) is list and len(rows) == 64, "evidence index must contain all 64 rows")
    observed_ids = [row.get("row_id") for row in rows]
    expected_ids = [row["row_id"] for row in expected]
    _require(observed_ids == expected_ids, "evidence row order/closure differs")
    representation_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    p32_sha: Optional[str] = None
    validated_p32: set[tuple[str, str]] = set()
    for registered, evidence_row in zip(expected, rows):
        _closed(
            evidence_row,
            {"row_id", "candidate_id", "instruction", "phase_labels", "semantic_binding", "sidecar_receipt", "review_receipt"},
            "evidence row",
        )
        _require(evidence_row["row_id"] == registered["row_id"] and evidence_row["candidate_id"] == registered["candidate_id"], "evidence row identity differs")
        instruction = _utf8_binding(evidence_row["instruction"], "evidence instruction")
        _require(instruction["text"] == registered["instruction_text"], "instruction differs from sealed authoring")
        phases = validate_phase_labels(evidence_row["phase_labels"])
        semantic = validate_semantic_binding(evidence_row["semantic_binding"])
        sidecar_ref = _closed(evidence_row["sidecar_receipt"], {"path", "file_sha256"}, "sidecar reference")
        review_ref = _closed(evidence_row["review_receipt"], {"path", "file_sha256"}, "review reference")
        sidecar = validate_sidecar_receipt(
            sidecar_ref["path"],
            sidecar_ref["file_sha256"],
            require_admissible=True,
            validated_p32=validated_p32,
        )
        review = validate_review_receipt(review_ref["path"], review_ref["file_sha256"])
        for field in ("row_id", "candidate_id"):
            _require(sidecar[field] == evidence_row[field], f"sidecar {field} differs")
        for field in ("branch",):
            _require(sidecar[field] == registered[field] == review[field], f"sidecar/review {field} differs")
        _require(review["candidate_id"] == registered["candidate_id"], "review candidate differs")
        _require(tuple(review["phase_labels"]) == phases, "review/evidence phase labels differ")
        _require(sidecar["source_iid"] == registered["source_iid"] and sidecar["seed"] == registered["seed"] and sidecar["analysis_split"] == registered["analysis_split"], "sidecar population binding differs")
        observed_p32 = sidecar["phi_v1"]["p32_raw_sha256"]
        p32_sha = observed_p32 if p32_sha is None else p32_sha
        _require(observed_p32 == p32_sha, "P32 bytes differ across rows")
        is_noop = registered["branch"] == "noop"
        representation_rows.append(
            {
                "row_id": registered["row_id"],
                "analysis_split": registered["analysis_split"],
                "source_iid": registered["source_iid"],
                "seed": registered["seed"],
                "branch": registered["branch"],
                "is_noop": is_noop,
                "instruction": dict(instruction),
                "phase_labels": list(phases),
                "semantic_binding": dict(semantic),
                "quotient_tensor": dict(sidecar["tensor"]),
                "sidecar_receipt": dict(sidecar_ref),
                "review_receipt": dict(review_ref),
                "planner_optimizer_eligible": registered["analysis_split"] == "fit" and registered["branch"] in NONNOOP,
            }
        )
        source = q0[registered["source_iid"]]
        reverse = registered["branch"] == "reverse"
        pair_rows.append(
            {
                "row_id": registered["row_id"],
                "representation_tensor_sha256": sidecar["tensor"]["raw_sha256"],
                "analysis_split": registered["analysis_split"],
                "source_iid": registered["source_iid"],
                "seed": registered["seed"],
                "branch": registered["branch"],
                "start_state": "q1" if reverse else "q0",
                "real_source_available": not reverse,
                "real_source_video_path": None if reverse else source["q0_source_video_path"],
                "real_source_video_sha256": None if reverse else source["q0_source_video_sha256"],
                "operator_optimizer_eligible": registered["analysis_split"] == "fit" and registered["branch"] in OPERATOR_BRANCHES,
            }
        )
    row_order_sha = object_sha256(expected_ids)
    representation = _seal(
        {
            "schema_version": REPRESENTATION_SCHEMA,
            "manifest_id": "generic-action-first8-phi-v1-r1",
            "authoring_registry_sha256": AUTHORING_SHA256,
            "population_registry_sha256": POPULATION_SHA256,
            "phi_v1_p32_raw_sha256": p32_sha,
            "row_order_sha256": row_order_sha,
            "counts": {"fit": 32, "confirmation": 32, "planner_optimizer": 24, "noop_audit": 16},
            "generated_media_is_optimizer_input_or_target": False,
            "rows": representation_rows,
        },
        "manifest_digest",
    )
    rep_output = Path(representation_output)
    pair_output_path = Path(pair_output)
    for output in (rep_output, pair_output_path):
        _require(output.is_absolute(), "output path must be absolute")
        _require(not output.exists() and not output.is_symlink(), f"refusing to overwrite {output}")
        _require(output.parent.is_dir(), f"output parent missing: {output.parent}")
    raw = canonical_json_bytes(representation) + b"\n"
    rep_output.write_bytes(raw)
    os.chmod(rep_output, 0o444)
    representation_file_sha = hashlib.sha256(raw).hexdigest()
    pairs = _seal(
        {
            "schema_version": PAIR_SCHEMA,
            "manifest_id": "generic-action-first8-source-pairs-r1",
            "representation_manifest_path": str(rep_output),
            "representation_manifest_file_sha256": representation_file_sha,
            "representation_manifest_digest": representation["manifest_digest"],
            "q0_authority_file_sha256": file_sha256(q0_authority_path),
            "row_order_sha256": row_order_sha,
            "counts": {"fit": 32, "confirmation": 32, "q0_available": 48, "q1_available": 0, "operator_optimizer": 16},
            "reverse_without_q1_is_operator_eligible": False,
            "generated_media_is_editor_source_or_target": False,
            "rows": pair_rows,
        },
        "manifest_digest",
    )
    pair_output_path.write_bytes(canonical_json_bytes(pairs) + b"\n")
    os.chmod(pair_output_path, 0o444)
    validate_manifest_pair(rep_output, pair_output_path)
    return representation, pairs


def validate_manifest_pair(
    representation_path: str | Path, pair_path: str | Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Replay the entire authority chain; never trust only counts or seals."""

    representation_file = _plain_file(representation_path, "representation manifest")
    pair_file = _plain_file(pair_path, "source-pair manifest")
    representation = _load_json(representation_file, "representation manifest")
    pairs = _load_json(pair_file, "source-pair manifest")
    _closed(
        representation,
        {
            "schema_version", "manifest_id", "authoring_registry_sha256",
            "population_registry_sha256", "phi_v1_p32_raw_sha256",
            "row_order_sha256", "counts",
            "generated_media_is_optimizer_input_or_target", "rows",
            "manifest_digest",
        },
        "representation manifest",
    )
    _closed(
        pairs,
        {
            "schema_version", "manifest_id", "representation_manifest_path",
            "representation_manifest_file_sha256",
            "representation_manifest_digest", "q0_authority_file_sha256",
            "row_order_sha256", "counts",
            "reverse_without_q1_is_operator_eligible",
            "generated_media_is_editor_source_or_target", "rows",
            "manifest_digest",
        },
        "source-pair manifest",
    )
    _verify_seal(representation, "manifest_digest", "representation manifest")
    _verify_seal(pairs, "manifest_digest", "source-pair manifest")
    _require(
        representation["schema_version"] == REPRESENTATION_SCHEMA
        and representation["manifest_id"] == "generic-action-first8-phi-v1-r1",
        "representation schema/identity differs",
    )
    _require(
        pairs["schema_version"] == PAIR_SCHEMA
        and pairs["manifest_id"] == "generic-action-first8-source-pairs-r1",
        "source-pair schema/identity differs",
    )
    _require(
        representation["authoring_registry_sha256"] == AUTHORING_SHA256
        and representation["population_registry_sha256"] == POPULATION_SHA256,
        "representation registry authority differs",
    )
    _require(
        representation["counts"]
        == {"fit": 32, "confirmation": 32, "planner_optimizer": 24, "noop_audit": 16}
        and pairs["counts"]
        == {"fit": 32, "confirmation": 32, "q0_available": 48, "q1_available": 0, "operator_optimizer": 16},
        "manifest declared counts differ",
    )
    _require(
        representation["generated_media_is_optimizer_input_or_target"] is False
        and pairs["generated_media_is_editor_source_or_target"] is False
        and pairs["reverse_without_q1_is_operator_eligible"] is False,
        "manifest generated-media/reverse authority differs",
    )
    _require(
        Path(pairs["representation_manifest_path"]) == representation_file
        and pairs["representation_manifest_file_sha256"]
        == file_sha256(representation_file)
        and pairs["representation_manifest_digest"]
        == representation["manifest_digest"],
        "pair representation file/digest binding differs",
    )

    method_root = Path(__file__).resolve().parents[1]
    authoring_path = method_root / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
    population_path = method_root / "assets/mosaic_event_population_compact6_topup20_v1.json"
    q0_path = method_root / "assets/action_source_q0_authority_first8_v1.json"
    authoring = _load_json(authoring_path, "authoring registry", AUTHORING_SHA256)
    population = _load_json(population_path, "population registry", POPULATION_SHA256)
    expected = _expected_rows(authoring, population)
    q0_raw = _load_json(q0_path, "q0 authority")
    q0 = _validate_q0_authority(q0_raw, expected)
    _require(
        pairs["q0_authority_file_sha256"] == file_sha256(q0_path),
        "pair q0 authority file binding differs",
    )
    expected_ids = [row["row_id"] for row in expected]
    expected_row_sha = object_sha256(expected_ids)
    _require(
        representation["row_order_sha256"]
        == pairs["row_order_sha256"]
        == expected_row_sha,
        "manifest row-order authority differs",
    )
    declared_p32_sha = _sha(
        representation["phi_v1_p32_raw_sha256"], "representation P32"
    )
    rep_rows = representation["rows"]
    pair_rows = pairs["rows"]
    _require(
        type(rep_rows) is list
        and type(pair_rows) is list
        and len(rep_rows) == len(pair_rows) == len(expected) == 64,
        "manifest row closure differs",
    )
    _require(
        [row.get("row_id") for row in rep_rows]
        == [row.get("row_id") for row in pair_rows]
        == expected_ids,
        "manifest row bytes/order differ from the pinned population",
    )
    validated_p32: set[tuple[str, str]] = set()
    for registered, rep, pair in zip(expected, rep_rows, pair_rows):
        _closed(
            rep,
            {
                "row_id", "analysis_split", "source_iid", "seed", "branch",
                "is_noop", "instruction", "phase_labels", "semantic_binding",
                "quotient_tensor", "sidecar_receipt", "review_receipt",
                "planner_optimizer_eligible",
            },
            "representation row",
        )
        _closed(
            pair,
            {
                "row_id", "representation_tensor_sha256", "analysis_split",
                "source_iid", "seed", "branch", "start_state",
                "real_source_available", "real_source_video_path",
                "real_source_video_sha256", "operator_optimizer_eligible",
            },
            "source-pair row",
        )
        for field in ("row_id", "analysis_split", "source_iid", "seed", "branch"):
            _require(
                rep[field] == pair[field] == registered[field],
                f"manifest {field} differs for {registered['row_id']}",
            )
        branch = registered["branch"]
        is_noop = branch == "noop"
        planner_eligible = (
            registered["analysis_split"] == "fit" and branch in NONNOOP
        )
        operator_eligible = (
            registered["analysis_split"] == "fit" and branch in OPERATOR_BRANCHES
        )
        _require(
            rep["is_noop"] is is_noop
            and rep["planner_optimizer_eligible"] is planner_eligible
            and pair["operator_optimizer_eligible"] is operator_eligible,
            f"optimizer eligibility differs for {registered['row_id']}",
        )
        instruction = _utf8_binding(rep["instruction"], "representation instruction")
        _require(
            instruction["text"] == registered["instruction_text"],
            f"instruction differs for {registered['row_id']}",
        )
        phases = validate_phase_labels(
            rep["phase_labels"], f"representation {registered['row_id']} phases"
        )
        validate_semantic_binding(
            rep["semantic_binding"],
            f"representation {registered['row_id']} semantic binding",
        )
        validate_code_tensor(
            rep["quotient_tensor"],
            is_noop=is_noop,
            label=f"representation {registered['row_id']}",
        )
        sidecar_ref = _closed(
            rep["sidecar_receipt"], {"path", "file_sha256"}, "sidecar reference"
        )
        review_ref = _closed(
            rep["review_receipt"], {"path", "file_sha256"}, "review reference"
        )
        sidecar = validate_sidecar_receipt(
            sidecar_ref["path"],
            sidecar_ref["file_sha256"],
            require_admissible=True,
            validated_p32=validated_p32,
        )
        review = validate_review_receipt(
            review_ref["path"], review_ref["file_sha256"]
        )
        _require(
            sidecar["row_id"] == registered["row_id"]
            and sidecar["candidate_id"] == registered["candidate_id"]
            and sidecar["source_iid"] == registered["source_iid"]
            and sidecar["analysis_split"] == registered["analysis_split"]
            and sidecar["seed"] == registered["seed"]
            and sidecar["branch"] == branch
            and sidecar["tensor"] == rep["quotient_tensor"],
            f"sidecar/representation binding differs for {registered['row_id']}",
        )
        _require(
            sidecar["phi_v1"]["p32_raw_sha256"] == declared_p32_sha,
            "manifest rows do not share one P32",
        )
        _require(
            review["candidate_id"] == registered["candidate_id"]
            and review["branch"] == branch
            and tuple(review["phase_labels"]) == phases,
            f"review/representation binding differs for {registered['row_id']}",
        )
        _require(
            pair["representation_tensor_sha256"]
            == rep["quotient_tensor"]["raw_sha256"],
            f"pair quotient binding differs for {registered['row_id']}",
        )
        source = q0[registered["source_iid"]]
        if branch == "reverse":
            _require(
                pair["start_state"] == "q1"
                and pair["real_source_available"] is False
                and pair["real_source_video_path"] is None
                and pair["real_source_video_sha256"] is None
                and pair["operator_optimizer_eligible"] is False,
                "reverse q1 fail-closed binding differs",
            )
        else:
            _require(
                pair["start_state"] == "q0"
                and pair["real_source_available"] is True
                and pair["real_source_video_path"]
                == source["q0_source_video_path"]
                and pair["real_source_video_sha256"]
                == source["q0_source_video_sha256"],
                f"q0 source authority differs for {registered['row_id']}",
            )
            _sha(pair["real_source_video_sha256"], "pair q0 video")
    _require(
        sum(row["planner_optimizer_eligible"] is True for row in rep_rows) == 24
        and sum(row["is_noop"] is True for row in rep_rows) == 16
        and sum(row["operator_optimizer_eligible"] is True for row in pair_rows) == 16
        and sum(row["real_source_available"] is True for row in pair_rows) == 48,
        "manifest observed counts differ",
    )
    return representation, pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--authoring", required=True)
    build.add_argument("--population", required=True)
    build.add_argument("--evidence-index", required=True)
    build.add_argument("--q0-authority", required=True)
    build.add_argument("--representation-output", required=True)
    build.add_argument("--pair-output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--representation", required=True)
    validate.add_argument("--pairs", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        build_manifests(
            authoring_path=args.authoring,
            population_path=args.population,
            evidence_index_path=args.evidence_index,
            q0_authority_path=args.q0_authority,
            representation_output=args.representation_output,
            pair_output=args.pair_output,
        )
    else:
        validate_manifest_pair(args.representation, args.pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
