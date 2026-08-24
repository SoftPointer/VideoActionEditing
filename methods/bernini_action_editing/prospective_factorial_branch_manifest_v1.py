#!/usr/bin/env python3
"""Author and validate the prospective source-conditioned seven-branch bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import prospective_factorial_source_registry_v1 as registry  # noqa: E402


DESCRIPTOR_SCHEMA = "bernini-factorial-branch-prompt-descriptors-v1"
MANIFEST_SCHEMA = "bernini-prospective-factorial-branch-manifest-v1"
BRANCHES = registry.BRANCHES
GENERATED_BRANCHES = tuple(branch for branch in BRANCHES if branch != "noop")
FAMILIES = ("dog-stand-to-sit", "human-one-knee-to-stand")
SPLITS = registry.SPLITS
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ID = re.compile(r"[0-9a-f]{16}\Z")


class FactorialBranchManifestError(RuntimeError):
    """Raised before a prompt or source can enter the generation bank."""


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
        raise FactorialBranchManifestError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FactorialBranchManifestError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FactorialBranchManifestError(f"cannot read {label}") from error
    if type(value) is not dict:
        raise FactorialBranchManifestError(f"{label} must contain one object")
    return value


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise FactorialBranchManifestError(f"{label} must be non-empty text")
    return value.strip()


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise FactorialBranchManifestError(f"{label} must be a lowercase SHA-256")
    return value


def _prompts(family: str, descriptor: Mapping[str, Any]) -> dict[str, str]:
    actor = _text(descriptor.get("target_actor"), label="target actor")
    preserve = _text(descriptor.get("preserve"), label="preservation description")
    appearance = _text(
        descriptor.get("appearance_change"), label="appearance-only change"
    )
    wrong_owner = _text(
        descriptor.get("wrong_owner"), label="wrong-owner instruction"
    )
    camera_preserve = preserve.replace(", framing, and locked camera", "")
    if family == "dog-stand-to-sit":
        action = "sit"
        pronoun = "it"
        forward = (
            f"Have {actor} smoothly bend both hind legs, lower its pelvis, complete "
            "a natural seated posture, and remain seated through the end. Preserve "
            f"{preserve}. Do not add, remove, or move any other actor or object."
        )
        noop = (
            f"No edit: keep {actor} in its original stable four-legged stand for "
            f"the entire clip and preserve {preserve}."
        )
        reverse = (
            f"Counterfactual reverse-action negative: have {actor} begin seated, "
            "rise onto all four legs, and remain standing through the end. Preserve "
            f"{preserve}."
        )
        incomplete = (
            f"Have {actor} begin bending both hind legs and lower its pelvis only "
            "halfway, then stop in a sustained half-crouch without ever reaching a "
            f"seated state. Preserve {preserve}."
        )
        initial_state = "stable_four_leg_stand"
        terminal_state = "stable_seated_posture"
    elif family == "human-one-knee-to-stand":
        action = "stand"
        pronoun = "them"
        forward = (
            f"Have {actor} transfer weight onto both feet, rise from the one-knee "
            "pose into a fully upright stand, and remain standing through the end. "
            f"Preserve {preserve}. Do not add, remove, or move any other actor or object."
        )
        noop = (
            f"No edit: keep {actor} in the original one-knee pose for the entire "
            f"clip and preserve {preserve}."
        )
        reverse = (
            f"Counterfactual reverse-action negative: have {actor} begin fully "
            "upright, lower one knee to the ground, and remain in a one-knee pose "
            f"through the end. Preserve {preserve}."
        )
        incomplete = (
            f"Have {actor} transfer weight toward both feet and lift the grounded "
            "knee only halfway, then stop in a sustained low crouch without ever "
            f"reaching an upright stand. Preserve {preserve}."
        )
        initial_state = "stable_one_knee_pose"
        terminal_state = "fully_upright_stand"
    else:
        raise FactorialBranchManifestError(f"unknown action family: {family}")
    return {
        "forward": forward,
        "noop": noop,
        "reverse": reverse,
        "incomplete": incomplete,
        "camera_only": (
            f"Counterfactual camera-only negative: keep {actor} in the original "
            f"{initial_state.replace('_', ' ')} for the entire clip and never let "
            f"{pronoun} perform the target {action} action, while only the camera performs "
            "a conspicuous smooth rightward orbit and mild push-in. Apart from that "
            f"registered camera motion, preserve {camera_preserve}."
        ),
        "appearance_only": (
            f"Counterfactual appearance-only negative: keep {actor} in the original "
            f"{initial_state.replace('_', ' ')} for the entire clip and never let "
            f"{pronoun} perform the target {action} action, while {appearance}. Preserve "
            "all other source identity cues, body geometry, scene elements, subject "
            "position, framing, and the locked camera."
        ),
        "wrong_actor_or_object": (
            f"Counterfactual wrong-owner negative: {wrong_owner}. The original "
            f"target must never perform the target {action} action. Preserve {preserve}."
        ),
        "_initial_state": initial_state,
        "_terminal_state": terminal_state,
    }


def _descriptor_map(value: Mapping[str, Any], registry_digest: str) -> dict[str, dict[str, Any]]:
    if value.get("schema_version") != DESCRIPTOR_SCHEMA:
        raise FactorialBranchManifestError("descriptor schema differs")
    if value.get("source_registry_digest") != registry_digest:
        raise FactorialBranchManifestError("descriptor registry digest differs")
    policy = value.get("prompt_policy")
    if not isinstance(policy, Mapping) or policy != {
        "atomic_forward_only": True,
        "explicit_target_owner": True,
        "explicit_terminal_hold": True,
        "explicit_preservation_contract": True,
        "noop_is_exact_source_copy": True,
        "reverse_is_counterfactual_and_may_contradict_source_initial_state": True,
        "wrong_owner_is_a_deliberate_actor_insertion_or_distractor_event": True,
        "old_synthetic_target_prompt_reused": False,
    }:
        raise FactorialBranchManifestError("prompt policy differs")
    rows = value.get("sources")
    if not isinstance(rows, list):
        raise FactorialBranchManifestError("descriptor sources differ")
    result: dict[str, dict[str, Any]] = {}
    fields = {
        "source_id",
        "target_actor",
        "preserve",
        "appearance_change",
        "wrong_owner",
    }
    for row in rows:
        if type(row) is not dict or set(row) != fields:
            raise FactorialBranchManifestError("descriptor field closure differs")
        source_id = row["source_id"]
        if type(source_id) is not str or _SOURCE_ID.fullmatch(source_id) is None:
            raise FactorialBranchManifestError("descriptor source identity differs")
        if source_id in result:
            raise FactorialBranchManifestError("duplicate descriptor source")
        for field in fields - {"source_id"}:
            _text(row[field], label=field)
        result[source_id] = dict(row)
    if list(result) != sorted(result):
        raise FactorialBranchManifestError("descriptor rows must be source-sorted")
    return result


def author(registry_path: str | Path, descriptor_path: str | Path) -> dict[str, Any]:
    registry_file = _plain_file(registry_path, label="source registry")
    descriptor_file = _plain_file(descriptor_path, label="prompt descriptors")
    source_spec = _read_object(registry_file, label="source registry")
    sealed = registry.seal_registry(source_spec)
    if (
        sealed.get("status")
        != "balanced_population_frozen_branch_generation_allowed"
        or sealed.get("branch_generation_allowed") is not True
        or sealed.get("optimizer_step_allowed") is not False
    ):
        raise FactorialBranchManifestError("source registry lacks generation authority")
    registry_digest = sealed["registry_digest"]
    descriptors = _descriptor_map(
        _read_object(descriptor_file, label="prompt descriptors"), registry_digest
    )
    accepted = [
        row
        for row in source_spec["sources"]
        if row["review_decision"] == "accepted_prospective"
    ]
    accepted.sort(key=lambda row: row["source_id"])
    if set(descriptors) != {row["source_id"] for row in accepted}:
        raise FactorialBranchManifestError("descriptor population differs from registry")
    entries: list[dict[str, Any]] = []
    source_states: dict[str, dict[str, str]] = {}
    for source in accepted:
        source_id = source["source_id"]
        prompts = _prompts(source["action_family"], descriptors[source_id])
        source_states[source_id] = {
            "initial_state": prompts.pop("_initial_state"),
            "terminal_state": prompts.pop("_terminal_state"),
        }
        for seed in source["registered_seeds"]:
            for branch in BRANCHES:
                instruction = prompts[branch]
                entries.append(
                    {
                        "entry_id": f"{source_id}-s{seed}-{branch}",
                        "source_id": source_id,
                        "action_family": source["action_family"],
                        "analysis_split": source["assigned_split"],
                        "source_video": source["source_media_path"],
                        "source_video_sha256": source["source_media_sha256"],
                        "seed": seed,
                        "branch": branch,
                        "executor": (
                            "exact_source_copy" if branch == "noop" else "frozen_bernini"
                        ),
                        "instruction": instruction,
                        "instruction_utf8_sha256": text_sha256(instruction),
                    }
                )
    entries.sort(key=lambda row: row["entry_id"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_id": "prospective-factorial-core18-two-seed-seven-branch-20260813-v1",
        "source_registry": {
            "file_sha256": file_sha256(registry_file),
            "registry_digest": registry_digest,
            "confirmation_digest": sealed["confirmation_registry_digest"],
        },
        "prompt_descriptors": {
            "file_sha256": file_sha256(descriptor_file),
            "object_digest": object_sha256(
                _read_object(descriptor_file, label="prompt descriptors")
            ),
        },
        "release_policy": {
            "fit_generation_allowed": True,
            "fit_review_allowed": True,
            "calibration_generation_allowed_after_fit_representation_freeze": True,
            "calibration_review_allowed_after_fit_representation_freeze": True,
            "confirmation_generation_allowed_after_policy_freeze": True,
            "confirmation_review_allowed_once": True,
            "optimizer_step_allowed": False,
        },
        "inference_contract": {
            "renderer": "infer_lora.py",
            "adapter": "frozen_base_no_adapter",
            "num_frames": 81,
            "fps": 25,
            "num_inference_steps": 40,
            "ulysses_size": 4,
            "same_source_and_seed_across_branches": True,
            "old_synthetic_target_accessed": False,
            "noop_executor": "exact_source_copy",
            "generated_branch_order": list(GENERATED_BRANCHES),
        },
        "branch_order": list(BRANCHES),
        "source_states": source_states,
        "entries": entries,
        "authority": {
            "branch_generation_authorized": True,
            "training_target_authorized": False,
            "optimizer_step_authorized": False,
            "method_success_claimed": False,
        },
    }
    validate_manifest(manifest)
    manifest["manifest_digest"] = object_sha256(manifest)
    return manifest


def validate_manifest(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = dict(value)
    declared = candidate.pop("manifest_digest", None)
    if declared is not None and (
        _SHA256.fullmatch(str(declared)) is None or object_sha256(candidate) != declared
    ):
        raise FactorialBranchManifestError("manifest digest differs")
    if candidate.get("schema_version") != MANIFEST_SCHEMA:
        raise FactorialBranchManifestError("manifest schema differs")
    if candidate.get("branch_order") != list(BRANCHES):
        raise FactorialBranchManifestError("branch order differs")
    authority = candidate.get("authority")
    if authority != {
        "branch_generation_authorized": True,
        "training_target_authorized": False,
        "optimizer_step_authorized": False,
        "method_success_claimed": False,
    }:
        raise FactorialBranchManifestError("authority differs")
    release = candidate.get("release_policy")
    if not isinstance(release, Mapping) or release.get("fit_generation_allowed") is not True:
        raise FactorialBranchManifestError("fit generation release differs")
    if release.get("optimizer_step_allowed") is not False:
        raise FactorialBranchManifestError("release policy authorized optimizer")
    contract = candidate.get("inference_contract")
    if not isinstance(contract, Mapping) or contract != {
        "renderer": "infer_lora.py",
        "adapter": "frozen_base_no_adapter",
        "num_frames": 81,
        "fps": 25,
        "num_inference_steps": 40,
        "ulysses_size": 4,
        "same_source_and_seed_across_branches": True,
        "old_synthetic_target_accessed": False,
        "noop_executor": "exact_source_copy",
        "generated_branch_order": list(GENERATED_BRANCHES),
    }:
        raise FactorialBranchManifestError("inference contract differs")
    rows = candidate.get("entries")
    if not isinstance(rows, list) or len(rows) != 18 * 2 * len(BRANCHES):
        raise FactorialBranchManifestError("manifest entry population differs")
    required = {
        "entry_id",
        "source_id",
        "action_family",
        "analysis_split",
        "source_video",
        "source_video_sha256",
        "seed",
        "branch",
        "executor",
        "instruction",
        "instruction_utf8_sha256",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise FactorialBranchManifestError("entry field closure differs")
        entry_id = _text(row["entry_id"], label="entry ID")
        source_id = row["source_id"]
        if (
            entry_id in seen
            or type(source_id) is not str
            or _SOURCE_ID.fullmatch(source_id) is None
            or row["action_family"] not in FAMILIES
            or row["analysis_split"] not in SPLITS
            or row["branch"] not in BRANCHES
            or type(row["seed"]) is not int
        ):
            raise FactorialBranchManifestError("entry identity differs")
        if row["entry_id"] != f"{source_id}-s{row['seed']}-{row['branch']}":
            raise FactorialBranchManifestError("entry ID binding differs")
        expected_executor = (
            "exact_source_copy" if row["branch"] == "noop" else "frozen_bernini"
        )
        instruction = _text(row["instruction"], label="instruction")
        if (
            row["executor"] != expected_executor
            or text_sha256(instruction) != row["instruction_utf8_sha256"]
        ):
            raise FactorialBranchManifestError("entry execution binding differs")
        _sha(row["source_video_sha256"], label="source media digest")
        path = Path(row["source_video"])
        if not path.is_absolute() or path == Path("/"):
            raise FactorialBranchManifestError("source path differs")
        seen.add(entry_id)
        normalized.append(dict(row))
        groups.setdefault((source_id, row["seed"]), []).append(dict(row))
    if [row["entry_id"] for row in normalized] != sorted(seen):
        raise FactorialBranchManifestError("entries must be entry-ID sorted")
    if len(groups) != 36:
        raise FactorialBranchManifestError("source/seed cell count differs")
    for group in groups.values():
        if [row["branch"] for row in sorted(group, key=lambda row: BRANCHES.index(row["branch"]))] != list(BRANCHES):
            raise FactorialBranchManifestError("factorial cell is incomplete")
        if len({row["source_video_sha256"] for row in group}) != 1:
            raise FactorialBranchManifestError("factorial source differs")
    return normalized


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise FactorialBranchManifestError("output must be a fresh absolute path")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    author_parser = sub.add_parser("author")
    author_parser.add_argument("--registry", required=True)
    author_parser.add_argument("--descriptors", required=True)
    author_parser.add_argument("--output", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    if args.command == "author":
        result = author(args.registry, args.descriptors)
        _write_create_only(Path(args.output), result)
        print(json.dumps({"output": args.output, "entries": len(result["entries"]), "manifest_digest": result["manifest_digest"]}, sort_keys=True))
        return 0
    manifest_path = _plain_file(args.manifest, label="branch manifest")
    manifest = _read_object(manifest_path, label="branch manifest")
    rows = validate_manifest(manifest)
    counts = {split: sum(row["analysis_split"] == split for row in rows) for split in SPLITS}
    print(json.dumps({"manifest_digest": manifest["manifest_digest"], "entries": len(rows), "split_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FactorialBranchManifestError as error:
        print(f"[factorial-branch-manifest] ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
