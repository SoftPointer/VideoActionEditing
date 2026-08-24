#!/usr/bin/env python3
"""Seal a prospective, leakage-safe source registry before branch generation.

This layer precedes ``factorial_margin_policy_v1``.  It records source review
decisions and split membership without generating targets or consuming scores.
Seven-branch generation is allowed only when every registered action family
has a complete and balanced fit/calibration/confirmation source population.

Opened development and confirmation media belong in ``excluded_sources``.
They can document provenance, but can never be silently recycled into a new
prospective population.  A partial registry is still useful: it emits exact
missing quotas and a fail-closed receipt with no generation or optimizer
authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SPEC_SCHEMA = "bernini-prospective-factorial-source-spec-v1"
REGISTRY_SCHEMA = "bernini-prospective-factorial-source-registry-v1"
SOURCE_SCHEMA = "bernini-prospective-factorial-source-v1"
EXCLUSION_SCHEMA = "bernini-factorial-source-exclusion-v1"

SPLITS = ("fit", "calibration", "confirmation")
BRANCHES = (
    "forward",
    "noop",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
    "wrong_actor_or_object",
)
SOURCE_DECISIONS = (
    "accepted_prospective",
    "reserve_unsealed",
    "rejected_source_quality",
    "rejected_typed_state",
)
EXCLUSION_REASONS = (
    "opened_development",
    "opened_confirmation",
    "media_duplicate",
    "typed_state_failure",
    "camera_or_scene_failure",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")


class ProspectiveRegistryError(RuntimeError):
    """Raised before ambiguous source population state is published."""


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
        raise ProspectiveRegistryError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProspectiveRegistryError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProspectiveRegistryError(f"{label} must be an array")
    return value


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    row = _mapping(value, label=label)
    if set(row) != fields:
        missing = sorted(fields - set(row))
        extra = sorted(set(row) - fields)
        raise ProspectiveRegistryError(
            f"{label} field closure differs: missing={missing} extra={extra}"
        )
    return row


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ProspectiveRegistryError(f"{label} is not a safe identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProspectiveRegistryError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectiveRegistryError(f"{label} must be non-empty text")
    return value.strip()


def _validate_seeds(value: Any, *, required: int, label: str) -> list[int]:
    seeds = list(_sequence(value, label=label))
    if len(seeds) != required or len(set(seeds)) != required:
        raise ProspectiveRegistryError(f"{label} must contain exactly {required} unique seeds")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
        for seed in seeds
    ):
        raise ProspectiveRegistryError(f"{label} contains an invalid seed")
    return seeds


def validate_spec(value: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "population_id",
        "created_utc",
        "minimum_sources_per_family_per_split",
        "required_seeds_per_source",
        "branch_order",
        "action_families",
        "evidence_bindings",
        "excluded_sources",
        "sources",
    }
    row = _closed(value, fields, label="source spec")
    if row["schema_version"] != SPEC_SCHEMA:
        raise ProspectiveRegistryError("source spec schema differs")
    population_id = _identifier(row["population_id"], label="population ID")
    created = row["created_utc"]
    if not isinstance(created, str) or not created.endswith("Z"):
        raise ProspectiveRegistryError("created_utc must be an explicit UTC string")
    minimum = row["minimum_sources_per_family_per_split"]
    required_seeds = row["required_seeds_per_source"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ProspectiveRegistryError("minimum source quota must be positive")
    if isinstance(required_seeds, bool) or not isinstance(required_seeds, int) or required_seeds < 2:
        raise ProspectiveRegistryError("at least two registered seeds are required")
    if tuple(_sequence(row["branch_order"], label="branch order")) != BRANCHES:
        raise ProspectiveRegistryError("seven-branch order differs")

    families = [
        _identifier(item, label="action family")
        for item in _sequence(row["action_families"], label="action families")
    ]
    if len(families) < 2 or families != sorted(set(families)):
        raise ProspectiveRegistryError("action families must be sorted, unique, and multi-family")

    bindings = []
    for index, raw in enumerate(_sequence(row["evidence_bindings"], label="evidence bindings")):
        binding = _closed(
            raw,
            {"evidence_id", "path", "sha256", "role"},
            label=f"evidence binding {index}",
        )
        bindings.append(
            {
                "evidence_id": _identifier(binding["evidence_id"], label="evidence ID"),
                "path": _text(binding["path"], label="evidence path"),
                "sha256": _sha(binding["sha256"], label="evidence SHA-256"),
                "role": _text(binding["role"], label="evidence role"),
            }
        )
    if not bindings or [item["evidence_id"] for item in bindings] != sorted(
        {item["evidence_id"] for item in bindings}
    ):
        raise ProspectiveRegistryError("evidence bindings must be sorted and unique")

    excluded = []
    excluded_ids: set[str] = set()
    excluded_hashes: set[str] = set()
    for index, raw in enumerate(_sequence(row["excluded_sources"], label="excluded sources")):
        item = _closed(
            raw,
            {"schema_version", "source_id", "source_media_sha256", "reason", "note"},
            label=f"excluded source {index}",
        )
        if item["schema_version"] != EXCLUSION_SCHEMA:
            raise ProspectiveRegistryError("exclusion schema differs")
        source_id = _identifier(item["source_id"], label="excluded source ID")
        media_sha = _sha(item["source_media_sha256"], label="excluded source SHA-256")
        if item["reason"] not in EXCLUSION_REASONS:
            raise ProspectiveRegistryError("exclusion reason differs")
        if source_id in excluded_ids or media_sha in excluded_hashes:
            raise ProspectiveRegistryError("excluded source ID or media is duplicated")
        excluded_ids.add(source_id)
        excluded_hashes.add(media_sha)
        excluded.append(
            {
                "schema_version": EXCLUSION_SCHEMA,
                "source_id": source_id,
                "source_media_sha256": media_sha,
                "reason": item["reason"],
                "note": _text(item["note"], label="exclusion note"),
            }
        )

    sources = []
    source_ids: set[str] = set()
    source_hashes: set[str] = set()
    source_fields = {
        "schema_version",
        "source_id",
        "source_media_path",
        "source_media_sha256",
        "action_family",
        "review_decision",
        "review_note",
        "assigned_split",
        "registered_seeds",
    }
    for index, raw in enumerate(_sequence(row["sources"], label="sources")):
        item = _closed(raw, source_fields, label=f"source {index}")
        if item["schema_version"] != SOURCE_SCHEMA:
            raise ProspectiveRegistryError("source schema differs")
        source_id = _identifier(item["source_id"], label="source ID")
        media_sha = _sha(item["source_media_sha256"], label="source media SHA-256")
        family = _identifier(item["action_family"], label="source action family")
        if family not in families:
            raise ProspectiveRegistryError("source action family is not registered")
        decision = item["review_decision"]
        if decision not in SOURCE_DECISIONS:
            raise ProspectiveRegistryError("source review decision differs")
        if source_id in source_ids or media_sha in source_hashes:
            raise ProspectiveRegistryError("source ID or media is duplicated")
        if source_id in excluded_ids or media_sha in excluded_hashes:
            raise ProspectiveRegistryError("excluded source leaked into prospective sources")
        source_ids.add(source_id)
        source_hashes.add(media_sha)
        assigned = item["assigned_split"]
        seeds_raw = item["registered_seeds"]
        if decision == "accepted_prospective":
            if assigned not in SPLITS:
                raise ProspectiveRegistryError("accepted source requires one prospective split")
            seeds = _validate_seeds(
                seeds_raw, required=required_seeds, label=f"source {source_id} seeds"
            )
        else:
            if assigned is not None or list(_sequence(seeds_raw, label="unaccepted source seeds")):
                raise ProspectiveRegistryError("unaccepted source cannot own split or seeds")
            seeds = []
        sources.append(
            {
                "schema_version": SOURCE_SCHEMA,
                "source_id": source_id,
                "source_media_path": _text(item["source_media_path"], label="source media path"),
                "source_media_sha256": media_sha,
                "action_family": family,
                "review_decision": decision,
                "review_note": _text(item["review_note"], label="source review note"),
                "assigned_split": assigned,
                "registered_seeds": seeds,
            }
        )

    if [item["source_id"] for item in sources] != sorted(source_ids):
        raise ProspectiveRegistryError("sources must be sorted by source ID")
    return {
        "schema_version": SPEC_SCHEMA,
        "population_id": population_id,
        "created_utc": created,
        "minimum_sources_per_family_per_split": minimum,
        "required_seeds_per_source": required_seeds,
        "branch_order": list(BRANCHES),
        "action_families": families,
        "evidence_bindings": bindings,
        "excluded_sources": excluded,
        "sources": sources,
    }


def seal_registry(value: Any) -> dict[str, Any]:
    spec = validate_spec(value)
    accepted = [
        item for item in spec["sources"] if item["review_decision"] == "accepted_prospective"
    ]
    counts = Counter((item["action_family"], item["assigned_split"]) for item in accepted)
    quota = spec["minimum_sources_per_family_per_split"]
    split_counts: dict[str, dict[str, int]] = {}
    missing: list[dict[str, Any]] = []
    for family in spec["action_families"]:
        split_counts[family] = {}
        for split in SPLITS:
            count = counts[(family, split)]
            split_counts[family][split] = count
            if count < quota:
                missing.append(
                    {
                        "action_family": family,
                        "split": split,
                        "required": quota,
                        "observed": count,
                        "missing": quota - count,
                    }
                )
    ready = not missing
    confirmation_rows = [item for item in accepted if item["assigned_split"] == "confirmation"]
    confirmation_identity = [
        {
            "source_id": item["source_id"],
            "source_media_sha256": item["source_media_sha256"],
            "action_family": item["action_family"],
            "registered_seeds": item["registered_seeds"],
        }
        for item in confirmation_rows
    ]
    body = {
        "schema_version": REGISTRY_SCHEMA,
        "status": (
            "balanced_population_frozen_branch_generation_allowed"
            if ready
            else "incomplete_population_zero_generation"
        ),
        "population_id": spec["population_id"],
        "spec_digest": object_sha256(spec),
        "branch_order": list(BRANCHES),
        "action_families": spec["action_families"],
        "minimum_sources_per_family_per_split": quota,
        "required_seeds_per_source": spec["required_seeds_per_source"],
        "split_counts": split_counts,
        "missing_quotas": missing,
        "accepted_sources": accepted,
        "excluded_sources": spec["excluded_sources"],
        "evidence_bindings": spec["evidence_bindings"],
        "confirmation_registry_digest": object_sha256(confirmation_identity),
        "branch_generation_allowed": ready,
        "optimizer_step_allowed": False,
        "scores_consumed": False,
        "method_success_claimed": False,
    }
    return {**body, "registry_digest": object_sha256(body)}


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProspectiveRegistryError("input must be an absolute plain file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProspectiveRegistryError(f"cannot read input: {error}") from error
    if not isinstance(value, dict):
        raise ProspectiveRegistryError("input must contain one object")
    return value


def _create_json(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ProspectiveRegistryError("output must be a fresh absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    registry = seal_registry(_read_object(args.spec))
    _create_json(args.output, registry)
    print(json.dumps({"status": registry["status"], "registry_digest": registry["registry_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
