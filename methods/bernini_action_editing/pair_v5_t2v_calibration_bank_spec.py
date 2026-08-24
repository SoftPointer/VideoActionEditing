#!/usr/bin/env python3
"""Closed, hash-bound specification for the PAIR-v5 pure-T2V bank.

The ``geometry_source_video`` is decoded only to select Bernini's exact-81
spatial bucket.  It is never encoded or supplied to the T2V transformer.  A
candidate contains exactly one preregistered MACE semantic branch and one
complete, standalone T2V caption.  Ten ordered candidates form one sealed
same-seed/same-geometry cell.  Fit and confirmation cells are actor-, scene-,
and action-instance-group disjoint while retaining common action families.
Generated media is calibration evidence; it is never a donor, pseudo-target,
student input, policy candidate, or noise source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v1"
SCHEMA_VERSION_V2 = "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2"
SUPPORTED_ROOT_SCHEMA_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_V2)
CANDIDATE_SCHEMA_VERSION = "pair-v5-frozen-bernini-t2v-calibration-candidate-v1"
PLAN_SCHEMA_VERSION = "pair-v5-frozen-bernini-t2v-calibration-plan-v1"
RECEIPT_SCHEMA_VERSION = "pair-v5-frozen-bernini-t2v-calibration-receipt-v1"
BANK_RECEIPT_SCHEMA_VERSION = "pair-v5-frozen-bernini-t2v-calibration-bank-receipt-v2"
AUTHORING_SCHEMA_VERSION = "pair-v5-pure-t2v-calibration-authoring-v1"
AUTHORING_SELECTION_SCHEMA_VERSION = "pair-v5-pure-t2v-calibration-authoring-selection-v1"

# This order is intentionally identical to mace_candidate_action_energy.py.
MACE_BRANCH_ORDER = (
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
ANALYSIS_SPLITS = ("fit", "confirmation")
BRANCH_ORDER = MACE_BRANCH_ORDER
SPLIT_GROUP_AXES = ("actor_group_id", "scene_group_id", "action_group_id")
GROUP_LAYOUT = (("sp4-a", [0, 1, 2, 3]), ("sp4-b", [4, 5, 6, 7]))
CAPTION_CONTRACT = "complete_standalone_t2v_generation_caption"
GEOMETRY_CONTRACT = "decode_exact81_for_bucket_shape_only_never_encode_or_condition"
TARGET_INITIALIZATION = "official_gen_wanx22_fresh_gaussian"
SAMPLING_CONTRACT = {
    "condition_mode": "t2v_apg",
    "num_frames": 81,
    "latent_frames": 21,
    "fps": 25,
    "num_inference_steps": 40,
    "target_initialization": TARGET_INITIALIZATION,
    "guidance": {"omega_txt": 4.0, "omega_vid": 1.25, "omega_img": 4.5},
    "same_cell_official_gaussian_reuse_by_seed": True,
    "semantic_branch_order": list(MACE_BRANCH_ORDER),
}
SPLIT_CONTRACT = {
    "split_names": list(ANALYSIS_SPLITS),
    "group_disjoint_axes": list(SPLIT_GROUP_AXES),
    "action_family_must_cover_both_splits": True,
    "assignment_sealed_before_rendering": True,
    "rendered_quality_used_for_assignment": False,
}
SEMANTIC_INPUT_CLOSURE = {
    "accepted_semantic_inputs": ["full_t2v_caption"],
    "geometry_source_role": "bucket_shape_only",
    "geometry_source_pixels_enter_transformer": False,
    "geometry_source_vae_latent_created": False,
    "source_content_conditioning": False,
    "source_frame_reference": False,
    "target_video": False,
    "proposal_media_as_condition": False,
    "proposal_media_as_student_input": False,
    "proposal_media_as_noise": False,
    "proposal_media_as_donor_or_pseudo_target": False,
    "mask": False,
    "flow": False,
    "pose": False,
    "track": False,
    "trajectory": False,
    "custom_initial_noise": False,
}
ARTIFACT_USE_CONTRACT = {
    "generated_mp4": "calibration_evidence_only",
    "predecode_clean_latent": "calibration_evidence_only",
    "official_initial_gaussian": "provenance_evidence_only",
    "training_donor": False,
    "pseudo_target": False,
    "student_input": False,
    "student_initial_noise": False,
}

_ROOT_FIELDS = {
    "schema_version",
    "sampling_contract",
    "semantic_input_closure",
    "artifact_use_contract",
    "split_contract",
    "groups",
}
_GROUP_FIELDS = {"group_id", "visible_gpus", "candidates"}
_CANDIDATE_FIELDS = {
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
_ENVELOPE_FIELDS = {
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
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


class PairT2VCalibrationSpecError(RuntimeError):
    """Raised before ambiguous or privileged calibration input is accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PairT2VCalibrationSpecError("value is not canonical finite JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> None:
    raise PairT2VCalibrationSpecError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairT2VCalibrationSpecError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _loads(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairT2VCalibrationSpecError(f"{label} is not valid UTF-8 JSON") from error


def _closed(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PairT2VCalibrationSpecError(
            f"{label} keys differ: expected={sorted(expected)!r}, actual={actual!r}"
        )
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PairT2VCalibrationSpecError(f"{label} is not path-safe")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairT2VCalibrationSpecError(f"{label} must be lowercase SHA-256")
    return value


def _absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise PairT2VCalibrationSpecError(f"{label} must be path text without NUL")
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise PairT2VCalibrationSpecError(f"{label} must be absolute and non-root")
    return str(path)


def _complete_caption(value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise PairT2VCalibrationSpecError("full_t2v_caption must be UTF-8 text")
    caption = value.strip()
    if not 64 <= len(caption.encode("utf-8")) <= 4096 or len(caption.split()) < 12:
        raise PairT2VCalibrationSpecError(
            "full_t2v_caption must be a complete standalone generation caption"
        )
    if "{" in caption or "}" in caption:
        raise PairT2VCalibrationSpecError("full_t2v_caption contains a placeholder")
    return caption


def validate_candidate(value: Any) -> dict[str, Any]:
    candidate = _closed(value, _CANDIDATE_FIELDS, "candidate")
    if candidate["analysis_split"] not in ANALYSIS_SPLITS:
        raise PairT2VCalibrationSpecError(
            f"analysis_split must be one of {ANALYSIS_SPLITS!r}"
        )
    if candidate["semantic_branch"] not in MACE_BRANCH_ORDER:
        raise PairT2VCalibrationSpecError("semantic_branch is outside exact MACE order")
    if candidate["geometry_contract"] != GEOMETRY_CONTRACT:
        raise PairT2VCalibrationSpecError("geometry contract differs")
    if candidate["caption_contract"] != CAPTION_CONTRACT:
        raise PairT2VCalibrationSpecError("caption contract differs")
    caption = _complete_caption(candidate["full_t2v_caption"])
    caption_digest = _sha256(
        candidate["full_t2v_caption_utf8_sha256"], "full_t2v_caption_utf8_sha256"
    )
    if sha256_bytes(caption.encode("utf-8")) != caption_digest:
        raise PairT2VCalibrationSpecError("full T2V caption SHA-256 differs")
    seed = candidate["seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise PairT2VCalibrationSpecError("seed must be an integer in [0,2^63)")
    result = {
        "candidate_id": _safe_id(candidate["candidate_id"], "candidate_id"),
        "analysis_split": candidate["analysis_split"],
        "action_family_id": _safe_id(candidate["action_family_id"], "action_family_id"),
        "calibration_group_id": _safe_id(
            candidate["calibration_group_id"], "calibration_group_id"
        ),
        "prompt_group_id": _safe_id(candidate["prompt_group_id"], "prompt_group_id"),
        "action_family_group_id": _safe_id(
            candidate["action_family_group_id"], "action_family_group_id"
        ),
        "actor_group_id": _safe_id(candidate["actor_group_id"], "actor_group_id"),
        "scene_group_id": _safe_id(candidate["scene_group_id"], "scene_group_id"),
        "action_group_id": _safe_id(candidate["action_group_id"], "action_group_id"),
        "geometry_source_video": _absolute_path(
            candidate["geometry_source_video"], "geometry_source_video"
        ),
        "geometry_source_video_sha256": _sha256(
            candidate["geometry_source_video_sha256"], "geometry_source_video_sha256"
        ),
        "geometry_contract": GEOMETRY_CONTRACT,
        "semantic_branch": candidate["semantic_branch"],
        "full_t2v_caption": caption,
        "full_t2v_caption_utf8_sha256": caption_digest,
        "caption_contract": CAPTION_CONTRACT,
        "seed": seed,
    }
    expected_prompt_group = (
        f"{result['actor_group_id']}--{result['scene_group_id']}"
    )
    if result["prompt_group_id"] != expected_prompt_group:
        raise PairT2VCalibrationSpecError(
            "prompt_group_id must be the sealed actor_group_id--scene_group_id composite"
        )
    if result["action_family_group_id"] != result["action_group_id"]:
        raise PairT2VCalibrationSpecError(
            "action_family_group_id must equal the sealed action_group_id"
        )
    return result


def validate_root_spec(value: Any) -> dict[str, Any]:
    root = _closed(value, _ROOT_FIELDS, "root spec")
    if root["schema_version"] not in SUPPORTED_ROOT_SCHEMA_VERSIONS:
        raise PairT2VCalibrationSpecError("root spec schema_version differs")
    root_schema_version = root["schema_version"]
    if root["sampling_contract"] != SAMPLING_CONTRACT:
        raise PairT2VCalibrationSpecError("sampling contract is not exact81/40 native T2V")
    if root["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE:
        raise PairT2VCalibrationSpecError("semantic input closure differs")
    if root["artifact_use_contract"] != ARTIFACT_USE_CONTRACT:
        raise PairT2VCalibrationSpecError("artifact use contract differs")
    if root["split_contract"] != SPLIT_CONTRACT:
        raise PairT2VCalibrationSpecError("fit/confirmation split contract differs")
    groups = root["groups"]
    if not isinstance(groups, list) or len(groups) != 2:
        raise PairT2VCalibrationSpecError("exactly two SP4 groups are required")
    candidate_ids: set[str] = set()
    matched_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    cell_execution_groups: dict[tuple[str, str, str], set[str]] = {}
    calibration_group_owners: dict[str, tuple[str, str]] = {}
    split_groups = {
        split: {axis: set() for axis in SPLIT_GROUP_AXES}
        for split in ANALYSIS_SPLITS
    }
    action_families_by_split = {split: set() for split in ANALYSIS_SPLITS}
    normalized_groups = []
    for raw_group, (expected_id, expected_gpus) in zip(groups, GROUP_LAYOUT):
        group = _closed(raw_group, _GROUP_FIELDS, "group")
        if group["group_id"] != expected_id or group["visible_gpus"] != expected_gpus:
            raise PairT2VCalibrationSpecError("groups must be sp4-a=0..3 and sp4-b=4..7")
        rows = group["candidates"]
        if not isinstance(rows, list) or not rows:
            raise PairT2VCalibrationSpecError("each SP4 group requires candidates")
        normalized_rows = []
        for raw_candidate in rows:
            candidate = validate_candidate(raw_candidate)
            if candidate["candidate_id"] in candidate_ids:
                raise PairT2VCalibrationSpecError("candidate_id values must be globally unique")
            candidate_ids.add(candidate["candidate_id"])
            cell_key = (
                candidate["analysis_split"],
                candidate["action_family_id"],
                candidate["calibration_group_id"],
            )
            owner = (
                candidate["analysis_split"], candidate["action_family_id"]
            )
            prior_owner = calibration_group_owners.setdefault(
                candidate["calibration_group_id"], owner
            )
            if prior_owner != owner:
                raise PairT2VCalibrationSpecError(
                    "calibration_group_id must identify exactly one split/action-family cell"
                )
            matched_groups.setdefault(cell_key, []).append(candidate)
            cell_execution_groups.setdefault(cell_key, set()).add(expected_id)
            for axis in SPLIT_GROUP_AXES:
                split_groups[candidate["analysis_split"]][axis].add(candidate[axis])
            action_families_by_split[candidate["analysis_split"]].add(
                candidate["action_family_id"]
            )
            normalized_rows.append(candidate)
        normalized_groups.append(
            {"group_id": expected_id, "visible_gpus": expected_gpus, "candidates": normalized_rows}
        )
    for group_key, rows in matched_groups.items():
        if len(cell_execution_groups[group_key]) != 1:
            raise PairT2VCalibrationSpecError(
                f"calibration group {group_key!r} spans multiple SP4 groups"
            )
        branches = [row["semantic_branch"] for row in rows]
        if branches != list(MACE_BRANCH_ORDER):
            raise PairT2VCalibrationSpecError(
                f"calibration group {group_key!r} must follow exact MACE branch order"
            )
        matched_coordinates = {
            (
                row["geometry_source_video"],
                row["geometry_source_video_sha256"],
                row["seed"],
            )
            for row in rows
        }
        if len(matched_coordinates) != 1:
            raise PairT2VCalibrationSpecError(
                f"calibration group {group_key!r} does not share geometry and seed"
            )
        if len({row["prompt_group_id"] for row in rows}) != 1:
            raise PairT2VCalibrationSpecError(
                f"calibration group {group_key!r} does not share prompt_group_id"
            )
        if len({row["action_family_group_id"] for row in rows}) != 1:
            raise PairT2VCalibrationSpecError(
                f"calibration group {group_key!r} does not share action_family_group_id"
            )
        for axis in SPLIT_GROUP_AXES:
            if len({row[axis] for row in rows}) != 1:
                raise PairT2VCalibrationSpecError(
                    f"calibration group {group_key!r} does not share {axis}"
                )
        if len({row["full_t2v_caption_utf8_sha256"] for row in rows}) != len(
            MACE_BRANCH_ORDER
        ):
            raise PairT2VCalibrationSpecError(
                f"calibration group {group_key!r} has colliding branch captions"
            )
    present_splits = {key[0] for key in matched_groups}
    if present_splits != set(ANALYSIS_SPLITS):
        raise PairT2VCalibrationSpecError("fit and confirmation both require sealed groups")
    if action_families_by_split["fit"] != action_families_by_split["confirmation"]:
        raise PairT2VCalibrationSpecError(
            "every action_family_id must have fit and confirmation cells"
        )
    for axis in SPLIT_GROUP_AXES:
        overlap = split_groups["fit"][axis] & split_groups["confirmation"][axis]
        if overlap:
            raise PairT2VCalibrationSpecError(
                f"fit/confirmation {axis} values overlap: {sorted(overlap)!r}"
            )
    return {
        "schema_version": root_schema_version,
        "sampling_contract": dict(SAMPLING_CONTRACT),
        "semantic_input_closure": dict(SEMANTIC_INPUT_CLOSURE),
        "artifact_use_contract": dict(ARTIFACT_USE_CONTRACT),
        "split_contract": dict(SPLIT_CONTRACT),
        "groups": normalized_groups,
    }


def _plain_file(path: str | Path, label: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise PairT2VCalibrationSpecError(f"{label} must be an absolute plain file")
    return value


def load_sealed_spec(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    expected = _sha256(expected_sha256, "expected root spec SHA-256")
    spec_path = _plain_file(path, "sealed root spec")
    raw = spec_path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected:
        raise PairT2VCalibrationSpecError("sealed root spec raw SHA-256 differs")
    return validate_root_spec(_loads(raw, label="sealed root spec")), actual


def materialize_plan(
    *, spec_path: str | Path, expected_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    spec, digest = load_sealed_spec(spec_path, expected_sha256)
    output = Path(output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise PairT2VCalibrationSpecError("plan output must be a fresh absolute non-root path")
    output.mkdir(parents=False, exist_ok=False)
    records = []
    for group in spec["groups"]:
        group_dir = output / group["group_id"]
        group_dir.mkdir()
        for ordinal, candidate in enumerate(group["candidates"]):
            envelope = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "root_spec_raw_sha256": digest,
                "group_id": group["group_id"],
                "visible_gpus": group["visible_gpus"],
                "ordinal": ordinal,
                "sampling_contract": SAMPLING_CONTRACT,
                "semantic_input_closure": SEMANTIC_INPUT_CLOSURE,
                "artifact_use_contract": ARTIFACT_USE_CONTRACT,
                "split_contract": SPLIT_CONTRACT,
                "candidate": candidate,
            }
            candidate_path = group_dir / f"{ordinal:04d}-{candidate['candidate_id']}.json"
            candidate_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
            os.chmod(candidate_path, 0o400)
            records.append(
                {
                    "group_id": group["group_id"],
                    "candidate_id": candidate["candidate_id"],
                    "path": str(candidate_path),
                    "sha256": sha256_bytes(candidate_path.read_bytes()),
                }
            )
    manifest = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "root_spec_raw_sha256": digest,
        "candidate_records": records,
    }
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest))
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    os.chmod(manifest_path, 0o400)
    return manifest


def load_candidate_envelope(
    path: str | Path, expected_root_sha256: str
) -> dict[str, Any]:
    expected = _sha256(expected_root_sha256, "expected root spec SHA-256")
    candidate_path = _plain_file(path, "candidate envelope")
    raw = candidate_path.read_bytes()
    envelope = _closed(
        _loads(raw, label="candidate envelope"), _ENVELOPE_FIELDS, "candidate envelope"
    )
    if envelope["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise PairT2VCalibrationSpecError("candidate envelope schema differs")
    if envelope["root_spec_raw_sha256"] != expected:
        raise PairT2VCalibrationSpecError("candidate envelope root binding differs")
    if envelope["sampling_contract"] != SAMPLING_CONTRACT:
        raise PairT2VCalibrationSpecError("candidate sampling contract differs")
    if envelope["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE:
        raise PairT2VCalibrationSpecError("candidate semantic closure differs")
    if envelope["artifact_use_contract"] != ARTIFACT_USE_CONTRACT:
        raise PairT2VCalibrationSpecError("candidate artifact-use contract differs")
    if envelope["split_contract"] != SPLIT_CONTRACT:
        raise PairT2VCalibrationSpecError("candidate split contract differs")
    layout = {group_id: gpus for group_id, gpus in GROUP_LAYOUT}
    if layout.get(envelope["group_id"]) != envelope["visible_gpus"]:
        raise PairT2VCalibrationSpecError("candidate SP4 binding differs")
    if type(envelope["ordinal"]) is not int or envelope["ordinal"] < 0:
        raise PairT2VCalibrationSpecError("candidate ordinal differs")
    return {
        **dict(envelope),
        "candidate": validate_candidate(envelope["candidate"]),
        "candidate_envelope_sha256": sha256_bytes(raw),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = materialize_plan(
        spec_path=args.spec, expected_sha256=args.expected_sha256, output_dir=args.output_dir
    )
    print(canonical_json_bytes(manifest).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SPLITS",
    "ARTIFACT_USE_CONTRACT",
    "AUTHORING_SCHEMA_VERSION",
    "AUTHORING_SELECTION_SCHEMA_VERSION",
    "BANK_RECEIPT_SCHEMA_VERSION",
    "BRANCH_ORDER",
    "CAPTION_CONTRACT",
    "CANDIDATE_SCHEMA_VERSION",
    "GEOMETRY_CONTRACT",
    "GROUP_LAYOUT",
    "MACE_BRANCH_ORDER",
    "PairT2VCalibrationSpecError",
    "RECEIPT_SCHEMA_VERSION",
    "SAMPLING_CONTRACT",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V2",
    "SEMANTIC_INPUT_CLOSURE",
    "SPLIT_CONTRACT",
    "SPLIT_GROUP_AXES",
    "SUPPORTED_ROOT_SCHEMA_VERSIONS",
    "TARGET_INITIALIZATION",
    "canonical_json_bytes",
    "load_candidate_envelope",
    "load_sealed_spec",
    "materialize_plan",
    "sha256_bytes",
    "validate_candidate",
    "validate_root_spec",
]
