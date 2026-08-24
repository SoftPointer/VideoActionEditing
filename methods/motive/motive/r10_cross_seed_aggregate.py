"""Immutable two-seed evidence aggregation for the R10A proxy search.

This module consumes exactly the two preregistered R10A fold-assignment
seeds.  Each input must first pass
``r10_dynamic_dino_representation_search.validate_published_search``; the
aggregator never opens media, recomputes a representation, renders a video,
or starts training.

The aggregate is deliberately not a promotion gate.  Stable strong
development evidence can only request a genuinely fresh content-disjoint
holdout.  Every representation, renderer, and editor-training permission
remains closed in both possible outcomes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from . import r7_artifact_permissions as artifact_permissions
from . import r10_dynamic_dino_representation_search as r10


AGGREGATE_SCHEMA = "motive-r10a-cross-seed-aggregate-v1"
DONE_SCHEMA = "motive-r10a-cross-seed-aggregate-done-v1"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (SUMMARY_NAME, DONE_NAME)
PAYLOAD_NAMES = (SUMMARY_NAME,)

REQUIRED_SEEDS = (260108837, 260108838)
STATUS_NEED_FRESH_HOLDOUT = "NEED_FRESH_HOLDOUT"
STATUS_CONTINUE_TO_R10B = "CONTINUE_TO_R10B"
STATUSES = (
    STATUS_NEED_FRESH_HOLDOUT,
    STATUS_CONTINUE_TO_R10B,
)

INPUT_BINDING_KEYS = {
    "candidate_manifest_dir",
    "candidate_manifest_done_sha256",
    "track_cache_final",
    "track_cache_done_sha256",
    "visual_features_final",
    "visual_features_done_sha256",
    "visual_candidates_manifest",
    "visual_candidates_sha256",
}
INPUT_BINDING_DIGEST_KEYS = {
    "candidate_manifest_done_sha256",
    "track_cache_done_sha256",
    "visual_features_done_sha256",
    "visual_candidates_sha256",
}
GATE_FIELDS = (
    "representation_gate_passed",
    "renderer_probe_authorized",
    "editor_training_authorized",
)
_EPS = 1e-12


class R10CrossSeedAggregateError(ValueError):
    """The R10A two-seed comparability or publication contract is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _object_digest(value: Any) -> str:
    return _digest_bytes(_canonical_json(value).encode("utf-8"))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R10CrossSeedAggregateError(f"{context} must be an object")
    return value


def _require_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise R10CrossSeedAggregateError(f"{context} must be boolean")
    return value


def _require_fraction(value: object, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise R10CrossSeedAggregateError(
            f"{context} must be one finite fraction"
        )
    return float(value)


def _require_positive_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise R10CrossSeedAggregateError(
            f"{context} must be one positive integer"
        )
    return value


def _require_nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R10CrossSeedAggregateError(
            f"{context} must be one nonnegative integer"
        )
    return value


def _implementation_source(
    summary: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, str], str]:
    implementation = _require_mapping(
        summary.get("implementation"),
        context=f"seed={seed} implementation",
    )
    files = _require_mapping(
        implementation.get("files"),
        context=f"seed={seed} implementation.files",
    )
    if not files:
        raise R10CrossSeedAggregateError(
            f"seed={seed} implementation.files is empty"
        )
    normalized_files: dict[str, str] = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not name or not _is_sha256(digest):
            raise R10CrossSeedAggregateError(
                f"seed={seed} implementation file digest differs"
            )
        normalized_files[name] = str(digest)
    bundle = implementation.get("bundle_sha256")
    if (
        not _is_sha256(bundle)
        or bundle != _object_digest(normalized_files)
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} implementation source digest differs"
        )
    return dict(sorted(normalized_files.items())), str(bundle)


def _source_tree_digest(
    summary: Mapping[str, Any],
    *,
    seed: int,
) -> str:
    source_snapshot = _require_mapping(
        summary.get("source_snapshot"),
        context=f"seed={seed} source_snapshot",
    )
    digest = source_snapshot.get("tree_sha256")
    if (
        not _is_sha256(digest)
        or source_snapshot.get(
            "exact_tree_verified_by_controller_before_search"
        )
        is not True
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} source snapshot tree digest differs"
        )
    return str(digest)


def _input_contract(
    summary: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    bindings = dict(
        _require_mapping(
            summary.get("input_bindings"),
            context=f"seed={seed} input_bindings",
        )
    )
    if set(bindings) != INPUT_BINDING_KEYS:
        raise R10CrossSeedAggregateError(
            f"seed={seed} input binding closure differs"
        )
    for key in INPUT_BINDING_DIGEST_KEYS:
        if not _is_sha256(bindings.get(key)):
            raise R10CrossSeedAggregateError(
                f"seed={seed} input binding digest differs: {key}"
            )
    for key in INPUT_BINDING_KEYS - INPUT_BINDING_DIGEST_KEYS:
        value = bindings.get(key)
        if not isinstance(value, str) or not value:
            raise R10CrossSeedAggregateError(
                f"seed={seed} input binding path differs: {key}"
            )

    coverage = dict(
        _require_mapping(
            summary.get("input_coverage"),
            context=f"seed={seed} input_coverage",
        )
    )
    common_rows = coverage.get("r7_common_rows")
    paired_rows = coverage.get("r10_paired_source_target_dino_rows")
    if (
        coverage.get("all_specs_share_exact_r10_cohort") is not True
        or isinstance(common_rows, bool)
        or not isinstance(common_rows, int)
        or common_rows < 1
        or isinstance(paired_rows, bool)
        or not isinstance(paired_rows, int)
        or not 1 <= paired_rows <= common_rows
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} cohort coverage contract differs"
        )
    observed = _require_fraction(
        coverage.get("r10_common_cohort_fraction_of_r7"),
        context=f"seed={seed} common cohort fraction",
    )
    minimum = _require_fraction(
        coverage.get("minimum_required_fraction"),
        context=f"seed={seed} minimum cohort fraction",
    )
    recomputed = paired_rows / common_rows
    if (
        not math.isclose(
            observed,
            recomputed,
            abs_tol=_EPS,
            rel_tol=0.0,
        )
        or not math.isclose(
            minimum,
            r10.MIN_COHORT_COVERAGE,
            abs_tol=_EPS,
            rel_tol=0.0,
        )
        or observed + _EPS < minimum
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} cohort fraction or sealed minimum differs"
        )
    binding = {
        "input_bindings": bindings,
        "input_coverage": coverage,
    }
    return bindings, coverage, _object_digest(binding)


def _champion_contract(
    summary: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], str, bool, bool]:
    champion = _require_mapping(
        summary.get("champion"),
        context=f"seed={seed} champion",
    )
    spec = dict(
        _require_mapping(
            champion.get("frozen_spec"),
            context=f"seed={seed} frozen_spec",
        )
    )
    spec_digest = spec.get("spec_digest")
    core = {key: value for key, value in spec.items() if key != "spec_digest"}
    if (
        not _is_sha256(spec_digest)
        or spec_digest != _object_digest(core)
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} frozen spec digest differs"
        )
    champion_signal = _require_bool(
        champion.get("single_seed_development_signal_passed"),
        context=f"seed={seed} champion development signal",
    )
    legacy_passed = _require_bool(
        champion.get("legacy_test_diagnostic_passed"),
        context=f"seed={seed} legacy diagnostic",
    )
    decision = _require_mapping(
        summary.get("decision"),
        context=f"seed={seed} decision",
    )
    decision_signal = _require_bool(
        decision.get("single_seed_development_signal_passed"),
        context=f"seed={seed} decision development signal",
    )
    if decision_signal is not champion_signal:
        raise R10CrossSeedAggregateError(
            f"seed={seed} development signal fields disagree"
        )
    if decision.get("legacy_test_diagnostic_passed") is not legacy_passed:
        raise R10CrossSeedAggregateError(
            f"seed={seed} legacy diagnostic fields disagree"
        )
    return spec, str(spec_digest), champion_signal, legacy_passed


def _nested_contract(
    summary: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    nested = _require_mapping(
        summary.get("nested_outer_model_selection"),
        context=f"seed={seed} nested outer selection",
    )
    records = nested.get("records")
    if not isinstance(records, list) or not records:
        raise R10CrossSeedAggregateError(
            f"seed={seed} nested outer records differ"
        )
    budget = _require_mapping(
        summary.get("budget"),
        context=f"seed={seed} budget",
    )
    repeats = _require_positive_int(
        budget.get("requested_repeats"),
        context=f"seed={seed} requested repeats",
    )
    folds = _require_positive_int(
        budget.get("requested_folds_per_repeat"),
        context=f"seed={seed} requested folds",
    )
    if (
        repeats != r10.DEFAULT_REPEATS
        or folds != r10.DEFAULT_FOLDS
        or len(records) != repeats * folds
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} preregistered nested-fold budget differs"
        )

    passed = 0
    all_evaluable_from_records = True
    selected_spec_digests: list[str] = []
    fold_ids: set[str] = set()
    for position, value in enumerate(records):
        record = _require_mapping(
            value,
            context=f"seed={seed} nested record {position}",
        )
        fold_id = record.get("outer_fold_id")
        if (
            not isinstance(fold_id, str)
            or not fold_id
            or fold_id in fold_ids
        ):
            raise R10CrossSeedAggregateError(
                f"seed={seed} nested outer fold IDs differ"
            )
        fold_ids.add(fold_id)
        outer_gate = _require_bool(
            record.get("outer_gate_passed"),
            context=f"seed={seed} {fold_id} outer gate",
        )
        outer_evaluable = _require_bool(
            record.get("outer_evaluable"),
            context=f"seed={seed} {fold_id} evaluability",
        )
        if record.get("outer_query_seen_by_inner_fit") is not False:
            raise R10CrossSeedAggregateError(
                f"seed={seed} {fold_id} inner/outer closure differs"
            )
        passed += int(outer_gate)
        all_evaluable_from_records &= outer_evaluable
        selected = record.get("selected_spec_digest")
        if selected is not None:
            if not _is_sha256(selected):
                raise R10CrossSeedAggregateError(
                    f"seed={seed} {fold_id} selected spec digest differs"
                )
            selected_spec_digests.append(str(selected))

    fraction = _require_fraction(
        nested.get("gate_pass_fraction"),
        context=f"seed={seed} nested outer gate fraction",
    )
    recomputed = passed / len(records)
    if not math.isclose(fraction, recomputed, abs_tol=1e-12, rel_tol=0.0):
        raise R10CrossSeedAggregateError(
            f"seed={seed} nested outer gate fraction is inconsistent"
        )
    all_requested_usable = _require_bool(
        nested.get("all_requested_folds_usable"),
        context=f"seed={seed} requested-fold usability",
    )
    budget_usable = budget.get("usable_search_folds")
    if (
        isinstance(budget_usable, bool)
        or not isinstance(budget_usable, int)
        or not 0 <= budget_usable <= len(records)
        or all_requested_usable
        is not (budget_usable == repeats * folds)
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} requested-fold usability fields disagree"
        )
    all_evaluable = _require_bool(
        nested.get("all_nested_outer_folds_evaluable"),
        context=f"seed={seed} nested evaluability",
    )
    if all_evaluable is not all_evaluable_from_records:
        raise R10CrossSeedAggregateError(
            f"seed={seed} nested evaluability fields disagree"
        )
    families = nested.get("stable_cross_fold_eligible_families")
    if (
        not isinstance(families, list)
        or any(not isinstance(value, str) or not value for value in families)
        or families != sorted(set(families))
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} stable family closure differs"
        )
    family_count = _require_nonnegative_int(
        nested.get("stable_family_count"),
        context=f"seed={seed} stable family count",
    )
    minimum_families = _require_positive_int(
        nested.get("minimum_stable_families"),
        context=f"seed={seed} minimum stable families",
    )
    if (
        family_count != len(families)
        or minimum_families != r10.MIN_STABLE_FAMILIES
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} stable family count or minimum differs"
        )
    selection = _require_mapping(
        summary.get("selection_protocol"),
        context=f"seed={seed} selection protocol",
    )
    threshold = _require_fraction(
        selection.get("minimum_development_fold_pass_fraction"),
        context=f"seed={seed} development fold threshold",
    )
    if not math.isclose(
        threshold,
        r10.MIN_DEVELOPMENT_FOLD_PASS_FRACTION,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} development threshold differs from R10A"
        )
    return {
        "requested_repeats": repeats,
        "requested_folds_per_repeat": folds,
        "records": len(records),
        "passed_records": passed,
        "gate_pass_fraction": fraction,
        "minimum_gate_pass_fraction": threshold,
        "all_requested_folds_usable": all_requested_usable,
        "all_nested_outer_folds_evaluable": all_evaluable,
        "stable_cross_fold_eligible_families": list(families),
        "stable_family_count": family_count,
        "minimum_stable_families": minimum_families,
        "selected_spec_digests": sorted(set(selected_spec_digests)),
    }


def _validate_seed_artifact(path: Path) -> dict[str, Any]:
    try:
        validated = r10.validate_published_search(path)
    except Exception as error:
        raise R10CrossSeedAggregateError(
            f"R10A seed artifact failed strict validation: {path}"
        ) from error
    root = Path(validated["root"])
    summary = _require_mapping(
        validated.get("summary"),
        context=f"{root} validated summary",
    )
    done = _require_mapping(
        validated.get("done"),
        context=f"{root} validated done",
    )
    seed_value = summary.get("seed")
    if (
        isinstance(seed_value, bool)
        or not isinstance(seed_value, int)
        or seed_value not in REQUIRED_SEEDS
    ):
        raise R10CrossSeedAggregateError(
            f"{root} is not one preregistered R10A seed"
        )
    seed = int(seed_value)
    bindings, coverage, cohort_binding = _input_contract(
        summary,
        seed=seed,
    )
    implementation_files, source_digest = _implementation_source(
        summary,
        seed=seed,
    )
    source_tree_sha256 = _source_tree_digest(summary, seed=seed)
    spec, spec_digest, signal, legacy_passed = _champion_contract(
        summary,
        seed=seed,
    )
    nested = _nested_contract(summary, seed=seed)
    folds_record = _require_mapping(
        _require_mapping(
            done.get("payload_files"),
            context=f"seed={seed} payload registry",
        ).get(r10.FOLDS_NAME),
        context=f"seed={seed} folds payload record",
    )
    folds_digest = folds_record.get("sha256")
    if not _is_sha256(folds_digest):
        raise R10CrossSeedAggregateError(
            f"seed={seed} folds payload digest differs"
        )
    transform_record = _require_mapping(
        _require_mapping(
            done.get("payload_files"),
            context=f"seed={seed} payload registry",
        ).get(r10.TRANSFORM_NAME),
        context=f"seed={seed} frozen transform payload record",
    )
    transform_digest = transform_record.get("sha256")
    if not _is_sha256(transform_digest):
        raise R10CrossSeedAggregateError(
            f"seed={seed} frozen transform payload digest differs"
        )
    frozen_transform = _require_mapping(
        summary.get("frozen_transform"),
        context=f"seed={seed} frozen transform summary",
    )
    transform_array_records = _require_mapping(
        frozen_transform.get("array_records"),
        context=f"seed={seed} frozen transform array records",
    )
    fold_protocol = _require_mapping(
        summary.get("fold_protocol"),
        context=f"seed={seed} fold protocol",
    )
    development_fold_assignment_sha256 = fold_protocol.get(
        "development_fold_assignment_sha256"
    )
    if (
        fold_protocol.get("seed_changes_group_fold_assignment") is not True
        or fold_protocol.get(
            "seed_is_stability_perturbation_not_independent_replication"
        )
        is not True
        or fold_protocol.get("legacy_test_excluded_from_selection") is not True
        or fold_protocol.get("legacy_test_is_fresh_promotion_holdout")
        is not False
        or not _is_sha256(development_fold_assignment_sha256)
    ):
        raise R10CrossSeedAggregateError(
            f"seed={seed} fold/freshness protocol differs"
        )
    return {
        "root": root,
        "seed": seed,
        "summary": summary,
        "done": done,
        "artifact_digest": str(done["artifact_digest"]),
        "done_sha256": _digest_file(root / r10.DONE_NAME),
        "summary_sha256": _digest_file(root / r10.SUMMARY_NAME),
        "folds_sha256": str(folds_digest),
        "development_fold_assignment_sha256":
            str(development_fold_assignment_sha256),
        "frozen_transform_sha256": str(transform_digest),
        "frozen_transform_array_records_sha256": _object_digest(
            dict(transform_array_records)
        ),
        "input_bindings": bindings,
        "input_coverage": coverage,
        "cohort_binding_sha256": cohort_binding,
        "implementation_files": implementation_files,
        "source_implementation_bundle_sha256": source_digest,
        "source_tree_sha256": source_tree_sha256,
        "champion_spec": spec,
        "champion_spec_digest": spec_digest,
        "single_seed_development_signal_passed": signal,
        "legacy_test_diagnostic_passed": legacy_passed,
        "nested": nested,
    }


def _input_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": record["seed"],
        "artifact_digest": record["artifact_digest"],
        "done_sha256": record["done_sha256"],
        "summary_sha256": record["summary_sha256"],
        "folds_sha256": record["folds_sha256"],
        "development_fold_assignment_sha256":
            record["development_fold_assignment_sha256"],
        "frozen_transform_sha256":
            record["frozen_transform_sha256"],
        "frozen_transform_array_records_sha256":
            record["frozen_transform_array_records_sha256"],
    }


def _aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    if len(records) != len(REQUIRED_SEEDS):
        raise R10CrossSeedAggregateError(
            "exactly two R10A seed artifacts are required"
        )
    by_seed: dict[int, Mapping[str, Any]] = {}
    roots: set[Path] = set()
    for record in records:
        seed = int(record["seed"])
        if seed in by_seed:
            raise R10CrossSeedAggregateError(
                f"duplicate R10A seed artifact: {seed}"
            )
        root = Path(record["root"])
        if root in roots:
            raise R10CrossSeedAggregateError(
                "R10A seed artifact directories must be distinct"
            )
        roots.add(root)
        by_seed[seed] = record
    if tuple(sorted(by_seed)) != REQUIRED_SEEDS:
        raise R10CrossSeedAggregateError(
            "R10A aggregate requires seeds 260108837 and 260108838"
        )
    ordered = [by_seed[seed] for seed in REQUIRED_SEEDS]

    first, second = ordered
    if first["input_bindings"] != second["input_bindings"]:
        raise R10CrossSeedAggregateError(
            "R10A seeds use different input bindings"
        )
    if first["input_coverage"] != second["input_coverage"]:
        raise R10CrossSeedAggregateError(
            "R10A seeds use different cohort coverage"
        )
    if (
        first["cohort_binding_sha256"]
        != second["cohort_binding_sha256"]
    ):
        raise R10CrossSeedAggregateError(
            "R10A seeds use different cohort bindings"
        )
    if (
        first["source_tree_sha256"] != second["source_tree_sha256"]
    ):
        raise R10CrossSeedAggregateError(
            "R10A seeds use different source snapshot tree digests"
        )
    if expected_source_tree_sha256 is not None:
        if not _is_sha256(expected_source_tree_sha256):
            raise R10CrossSeedAggregateError(
                "expected source tree digest is not SHA-256"
            )
        if first["source_tree_sha256"] != expected_source_tree_sha256:
            raise R10CrossSeedAggregateError(
                "R10A source tree differs from the external anchor"
            )
    if (
        first["source_implementation_bundle_sha256"]
        != second["source_implementation_bundle_sha256"]
        or first["implementation_files"] != second["implementation_files"]
    ):
        raise R10CrossSeedAggregateError(
            "R10A seeds use different implementation source digests"
        )

    champion_spec_stable = bool(
        first["champion_spec_digest"] == second["champion_spec_digest"]
        and first["champion_spec"] == second["champion_spec"]
    )
    stable_families_equal = bool(
        first["nested"]["stable_cross_fold_eligible_families"]
        == second["nested"]["stable_cross_fold_eligible_families"]
    )
    fold_assignments_distinct = bool(
        first["development_fold_assignment_sha256"]
        != second["development_fold_assignment_sha256"]
    )
    frozen_transforms_identical = bool(
        first["frozen_transform_sha256"]
        == second["frozen_transform_sha256"]
    )
    frozen_transform_array_records_identical = bool(
        first["frozen_transform_array_records_sha256"]
        == second["frozen_transform_array_records_sha256"]
    )

    per_seed: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    if not champion_spec_stable:
        reason_codes.append("champion_spec_not_stable_across_seeds")
    if not stable_families_equal:
        reason_codes.append("stable_family_sets_differ_across_seeds")
    if not fold_assignments_distinct:
        reason_codes.append("fold_assignments_not_distinct_across_seeds")

    for record in ordered:
        nested = record["nested"]
        nested_fraction_passed = bool(
            nested["gate_pass_fraction"] + _EPS
            >= nested["minimum_gate_pass_fraction"]
        )
        stable_family_support_passed = bool(
            nested["stable_family_count"]
            >= nested["minimum_stable_families"]
        )
        champion_reusable_identity = bool(
            record["champion_spec"].get("champion_eligible") is True
            and record["champion_spec"].get("head") == "identity"
        )
        seed_passed = bool(
            record["single_seed_development_signal_passed"]
            and nested_fraction_passed
            and nested["all_requested_folds_usable"]
            and nested["all_nested_outer_folds_evaluable"]
            and stable_family_support_passed
            and champion_reusable_identity
        )
        if not record["single_seed_development_signal_passed"]:
            reason_codes.append(
                f"seed_{record['seed']}_development_signal_failed"
            )
        if not nested_fraction_passed:
            reason_codes.append(
                f"seed_{record['seed']}_nested_outer_fraction_failed"
            )
        if not nested["all_requested_folds_usable"]:
            reason_codes.append(
                f"seed_{record['seed']}_requested_folds_not_all_usable"
            )
        if not nested["all_nested_outer_folds_evaluable"]:
            reason_codes.append(
                f"seed_{record['seed']}_nested_outer_not_all_evaluable"
            )
        if not stable_family_support_passed:
            reason_codes.append(
                f"seed_{record['seed']}_stable_family_support_failed"
            )
        if not champion_reusable_identity:
            reason_codes.append(
                f"seed_{record['seed']}_champion_not_reusable_identity"
            )
        per_seed.append(
            {
                **_input_identity(record),
                "champion_spec_digest": record["champion_spec_digest"],
                "single_seed_development_signal_passed":
                    record["single_seed_development_signal_passed"],
                "legacy_test_diagnostic_passed":
                    record["legacy_test_diagnostic_passed"],
                "nested_outer_gate_pass_fraction":
                    nested["gate_pass_fraction"],
                "minimum_nested_outer_gate_pass_fraction":
                    nested["minimum_gate_pass_fraction"],
                "nested_outer_gate_fraction_passed":
                    nested_fraction_passed,
                "all_requested_folds_usable":
                    nested["all_requested_folds_usable"],
                "all_nested_outer_folds_evaluable":
                    nested["all_nested_outer_folds_evaluable"],
                "stable_family_count": nested["stable_family_count"],
                "minimum_stable_families":
                    nested["minimum_stable_families"],
                "stable_cross_fold_eligible_families":
                    nested["stable_cross_fold_eligible_families"],
                "stable_family_support_passed":
                    stable_family_support_passed,
                "champion_reusable_identity":
                    champion_reusable_identity,
                "single_seed_strong_evidence_passed": seed_passed,
            }
        )

    strong = bool(
        champion_spec_stable
        and stable_families_equal
        and fold_assignments_distinct
        and all(
            row["single_seed_strong_evidence_passed"]
            for row in per_seed
        )
    )
    status = (
        STATUS_NEED_FRESH_HOLDOUT
        if strong
        else STATUS_CONTINUE_TO_R10B
    )
    if strong:
        next_experiment = (
            "freeze a genuinely unseen content-disjoint action cohort, "
            "then evaluate both sealed seed-specific transforms for the exact "
            "shared R10A structural spec without reselection"
        )
    else:
        next_experiment = (
            "R10B frozen-generator motion-weighted source-target "
            "delta-gradient representation"
        )

    input_records = {
        str(record["seed"]): _input_identity(record)
        for record in ordered
    }
    summary = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "artifact_kind": "immutable_final",
        "scope": (
            "R10A two end-to-end stability-seed development-evidence "
            "aggregation only; no media read, representation recomputation, "
            "rendering, or training"
        ),
        "seed_role": (
            "end-to-end stability perturbations affecting grouped folds, "
            "random "
            "projections, controls, and balanced-bank tie breaking; not "
            "pure fold-only seeds and not statistically independent "
            "replicates"
        ),
        "required_seeds": list(REQUIRED_SEEDS),
        "inputs": input_records,
        "input_comparability": {
            "strict_r10a_validation_passed_for_both": True,
            "input_artifact_closure_verified": True,
            "same_input_bindings": True,
            "same_cohort_coverage": True,
            "cohort_binding_sha256":
                first["cohort_binding_sha256"],
            "same_implementation_source_digest": True,
            "same_source_snapshot_tree_digest": True,
            "source_tree_sha256": first["source_tree_sha256"],
            "external_source_tree_anchor_sha256":
                expected_source_tree_sha256,
            "external_source_tree_anchor_verified":
                expected_source_tree_sha256 is not None,
            "source_implementation_bundle_sha256":
                first["source_implementation_bundle_sha256"],
            "contains_or_copies_video": False,
        },
        "cross_seed_evidence": {
            "per_seed": per_seed,
            "champion_spec_stable": champion_spec_stable,
            "shared_champion_spec_digest": (
                first["champion_spec_digest"]
                if champion_spec_stable
                else None
            ),
            "shared_champion_spec": (
                first["champion_spec"] if champion_spec_stable else None
            ),
            "stable_family_sets_equal": stable_families_equal,
            "shared_stable_cross_fold_eligible_families": (
                first["nested"][
                    "stable_cross_fold_eligible_families"
                ]
                if stable_families_equal
                else None
            ),
            "fold_assignments_distinct": fold_assignments_distinct,
            "legacy_test_diagnostics_are_non_gating": True,
            "frozen_transforms_identical":
                frozen_transforms_identical,
            "frozen_transform_array_records_identical":
                frozen_transform_array_records_identical,
            "frozen_transform_identity_is_not_a_stability_gate": True,
            "minimum_nested_outer_gate_pass_fraction":
                r10.MIN_DEVELOPMENT_FOLD_PASS_FRACTION,
            "minimum_observed_nested_outer_gate_pass_fraction": min(
                row["nested_outer_gate_pass_fraction"]
                for row in per_seed
            ),
            "mean_observed_nested_outer_gate_pass_fraction": (
                sum(
                    row["nested_outer_gate_pass_fraction"]
                    for row in per_seed
                )
                / len(per_seed)
            ),
            "cross_seed_development_signal_passed": strong,
            "failure_reason_codes": sorted(set(reason_codes)),
        },
        "decision": {
            "status": status,
            "next_experiment": next_experiment,
            "cross_seed_aggregation_passed": strong,
            "cross_seed_development_signal_passed": strong,
            "development_candidate_passed": False,
            "fresh_holdout_available": False,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        },
        "safety": {
            "fresh_holdout_is_mandatory_before_any_promotion": True,
            "legacy_test_is_not_a_fresh_promotion_holdout": True,
            "video_files_read": 0,
            "video_files_copied": 0,
            "renderer_calls": 0,
            "training_jobs_started": 0,
        },
        "formal_evidence": False,
        "training_authorized": False,
    }
    return summary


def aggregate_seed_artifacts(
    seed_artifact_dirs: Sequence[Path],
    *,
    expected_source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and aggregate exactly the two preregistered seed artifacts."""

    records = [
        _validate_seed_artifact(Path(path))
        for path in seed_artifact_dirs
    ]
    return _aggregate_records(
        records,
        expected_source_tree_sha256=expected_source_tree_sha256,
    )


def _publish(output_dir: Path, *, summary: Mapping[str, Any]) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    summary_bytes = _pretty_json_bytes(summary)
    summary_record = {
        "sha256": _digest_bytes(summary_bytes),
        "bytes": len(summary_bytes),
    }
    decision = _require_mapping(
        summary.get("decision"),
        context="aggregate decision",
    )
    if any(decision.get(field) is not False for field in GATE_FIELDS):
        raise R10CrossSeedAggregateError(
            "cross-seed publication attempted to open a protected gate"
        )
    done = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "payload_files": {SUMMARY_NAME: summary_record},
        "artifact_digest": _object_digest(
            {SUMMARY_NAME: summary_record}
        ),
        "input_artifact_digests": {
            seed: value["artifact_digest"]
            for seed, value in summary["inputs"].items()
        },
        "decision_status": decision["status"],
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
        "permission_contract": artifact_permissions.permission_contract(),
    }
    payloads = {
        SUMMARY_NAME: summary_bytes,
        DONE_NAME: _pretty_json_bytes(done),
    }
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    renamed = False
    try:
        for name in OUTPUT_NAMES:
            path = stage / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(payloads[name])
                    handle.flush()
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        artifact_permissions.seal_staging_tree(stage)
        artifact_permissions.assert_sealed_tree(stage)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(output_dir)
        root_was_temporarily_writable = False
        try:
            os.rename(stage, output_dir)
        except PermissionError:
            # macOS refuses to rename a 0555 directory even when its parent
            # is writable.  Keep every payload sealed, make only the private
            # root writable, and retain fail-closed validation plus cleanup
            # across the unavoidable rename/chmod interval.
            os.chmod(stage, 0o700)
            artifact_permissions.assert_sealed_tree(
                stage,
                allow_writable_root=True,
            )
            root_was_temporarily_writable = True
            os.rename(stage, output_dir)
        renamed = True
        if root_was_temporarily_writable:
            artifact_permissions.seal_published_root(output_dir)
        artifact_permissions.assert_sealed_tree(output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException as error:
        cleanup = output_dir if renamed else stage
        try:
            artifact_permissions.remove_staging_tree(cleanup)
            if renamed:
                _fsync_directory(output_dir.parent)
        except BaseException as cleanup_error:
            raise cleanup_error from error
        raise


def _validate_summary_semantics(
    summary: Mapping[str, Any],
    done: Mapping[str, Any],
) -> None:
    if (
        summary.get("schema_version") != AGGREGATE_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("artifact_kind") != "immutable_final"
        or done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed summary/done semantic contract differs"
        )
    if summary.get("required_seeds") != list(REQUIRED_SEEDS):
        raise R10CrossSeedAggregateError(
            "cross-seed required seed closure differs"
        )
    inputs = _require_mapping(
        summary.get("inputs"),
        context="cross-seed inputs",
    )
    if set(inputs) != {str(seed) for seed in REQUIRED_SEEDS}:
        raise R10CrossSeedAggregateError(
            "cross-seed input registry differs"
        )
    for seed in REQUIRED_SEEDS:
        record = _require_mapping(
            inputs[str(seed)],
            context=f"aggregate input seed={seed}",
        )
        if record.get("seed") != seed:
            raise R10CrossSeedAggregateError(
                f"aggregate input seed={seed} identity differs"
            )
        for field in (
            "artifact_digest",
            "done_sha256",
            "summary_sha256",
            "folds_sha256",
            "development_fold_assignment_sha256",
            "frozen_transform_sha256",
            "frozen_transform_array_records_sha256",
        ):
            if not _is_sha256(record.get(field)):
                raise R10CrossSeedAggregateError(
                    f"aggregate input seed={seed} {field} differs"
                )
    if done.get("input_artifact_digests") != {
        str(seed): inputs[str(seed)]["artifact_digest"]
        for seed in REQUIRED_SEEDS
    }:
        raise R10CrossSeedAggregateError(
            "cross-seed done input commitments differ"
        )

    comparability = _require_mapping(
        summary.get("input_comparability"),
        context="cross-seed input comparability",
    )
    for field in (
        "strict_r10a_validation_passed_for_both",
        "input_artifact_closure_verified",
        "same_input_bindings",
        "same_cohort_coverage",
        "same_implementation_source_digest",
        "same_source_snapshot_tree_digest",
    ):
        if comparability.get(field) is not True:
            raise R10CrossSeedAggregateError(
                f"cross-seed comparability field differs: {field}"
            )
    if (
        comparability.get("contains_or_copies_video") is not False
        or not _is_sha256(
            comparability.get("cohort_binding_sha256")
        )
        or not _is_sha256(
            comparability.get("source_tree_sha256")
        )
        or not _is_sha256(
            comparability.get(
                "source_implementation_bundle_sha256"
            )
        )
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed comparability commitment differs"
        )
    external_source_tree = comparability.get(
        "external_source_tree_anchor_sha256"
    )
    external_verified = _require_bool(
        comparability.get("external_source_tree_anchor_verified"),
        context="external source-tree anchor verification",
    )
    if (
        (external_source_tree is None and external_verified)
        or (
            external_source_tree is not None
            and (
                not _is_sha256(external_source_tree)
                or external_source_tree
                != comparability["source_tree_sha256"]
                or not external_verified
            )
        )
    ):
        raise R10CrossSeedAggregateError(
            "external source-tree anchor commitment differs"
        )

    evidence = _require_mapping(
        summary.get("cross_seed_evidence"),
        context="cross-seed evidence",
    )
    per_seed = evidence.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != 2:
        raise R10CrossSeedAggregateError(
            "cross-seed evidence rows differ"
        )
    row_by_seed: dict[int, Mapping[str, Any]] = {}
    for value in per_seed:
        row = _require_mapping(value, context="cross-seed evidence row")
        seed = row.get("seed")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in REQUIRED_SEEDS
            or seed in row_by_seed
        ):
            raise R10CrossSeedAggregateError(
                "cross-seed evidence seed closure differs"
            )
        if any(
            row.get(field) != inputs[str(seed)].get(field)
            for field in (
                "artifact_digest",
                "done_sha256",
                "summary_sha256",
                "folds_sha256",
                "development_fold_assignment_sha256",
                "frozen_transform_sha256",
                "frozen_transform_array_records_sha256",
            )
        ):
            raise R10CrossSeedAggregateError(
                f"seed={seed} evidence/input commitment differs"
            )
        if not _is_sha256(row.get("champion_spec_digest")):
            raise R10CrossSeedAggregateError(
                f"seed={seed} champion spec digest differs"
            )
        for field in (
            "single_seed_development_signal_passed",
            "legacy_test_diagnostic_passed",
            "nested_outer_gate_fraction_passed",
            "all_requested_folds_usable",
            "all_nested_outer_folds_evaluable",
            "stable_family_support_passed",
            "champion_reusable_identity",
            "single_seed_strong_evidence_passed",
        ):
            _require_bool(
                row.get(field),
                context=f"seed={seed} evidence {field}",
            )
        fraction = _require_fraction(
            row.get("nested_outer_gate_pass_fraction"),
            context=f"seed={seed} aggregate nested fraction",
        )
        threshold = _require_fraction(
            row.get("minimum_nested_outer_gate_pass_fraction"),
            context=f"seed={seed} aggregate nested threshold",
        )
        recomputed_fraction_pass = fraction + _EPS >= threshold
        if (
            row["nested_outer_gate_fraction_passed"]
            is not recomputed_fraction_pass
        ):
            raise R10CrossSeedAggregateError(
                f"seed={seed} aggregate nested decision differs"
            )
        family_count = _require_nonnegative_int(
            row.get("stable_family_count"),
            context=f"seed={seed} aggregate stable families",
        )
        minimum_family_count = _require_positive_int(
            row.get("minimum_stable_families"),
            context=f"seed={seed} aggregate minimum families",
        )
        recomputed_family_pass = family_count >= minimum_family_count
        if (
            row["stable_family_support_passed"]
            is not recomputed_family_pass
        ):
            raise R10CrossSeedAggregateError(
                f"seed={seed} aggregate family decision differs"
            )
        families = row.get("stable_cross_fold_eligible_families")
        if (
            not isinstance(families, list)
            or families != sorted(set(families))
            or any(
                not isinstance(family, str) or not family
                for family in families
            )
            or len(families) != family_count
        ):
            raise R10CrossSeedAggregateError(
                f"seed={seed} aggregate stable family closure differs"
            )
        recomputed_seed_pass = all(
            row[field] is True
            for field in (
                "single_seed_development_signal_passed",
                "nested_outer_gate_fraction_passed",
                "all_requested_folds_usable",
                "all_nested_outer_folds_evaluable",
                "stable_family_support_passed",
                "champion_reusable_identity",
            )
        )
        if (
            row["single_seed_strong_evidence_passed"]
            is not recomputed_seed_pass
        ):
            raise R10CrossSeedAggregateError(
                f"seed={seed} aggregate strong-evidence decision differs"
            )
        row_by_seed[seed] = row
    if tuple(sorted(row_by_seed)) != REQUIRED_SEEDS:
        raise R10CrossSeedAggregateError(
            "cross-seed evidence seed set differs"
        )
    if [row["seed"] for row in per_seed] != list(REQUIRED_SEEDS):
        raise R10CrossSeedAggregateError(
            "cross-seed evidence order differs"
        )

    declared_spec_stable = _require_bool(
        evidence.get("champion_spec_stable"),
        context="cross-seed champion stability",
    )
    declared_family_sets_equal = _require_bool(
        evidence.get("stable_family_sets_equal"),
        context="cross-seed family-set stability",
    )
    declared_folds_distinct = _require_bool(
        evidence.get("fold_assignments_distinct"),
        context="cross-seed fold distinction",
    )
    if evidence.get("legacy_test_diagnostics_are_non_gating") is not True:
        raise R10CrossSeedAggregateError(
            "legacy-test diagnostic gating contract differs"
        )
    spec_stable = bool(
        per_seed[0]["champion_spec_digest"]
        == per_seed[1]["champion_spec_digest"]
    )
    family_sets_equal = bool(
        per_seed[0]["stable_cross_fold_eligible_families"]
        == per_seed[1]["stable_cross_fold_eligible_families"]
    )
    folds_distinct = bool(
        per_seed[0]["development_fold_assignment_sha256"]
        != per_seed[1]["development_fold_assignment_sha256"]
    )
    frozen_transforms_identical = bool(
        per_seed[0]["frozen_transform_sha256"]
        == per_seed[1]["frozen_transform_sha256"]
    )
    frozen_array_records_identical = bool(
        per_seed[0]["frozen_transform_array_records_sha256"]
        == per_seed[1]["frozen_transform_array_records_sha256"]
    )
    if (
        declared_spec_stable is not spec_stable
        or declared_family_sets_equal is not family_sets_equal
        or declared_folds_distinct is not folds_distinct
        or evidence.get("frozen_transforms_identical")
        is not frozen_transforms_identical
        or evidence.get("frozen_transform_array_records_identical")
        is not frozen_array_records_identical
        or evidence.get(
            "frozen_transform_identity_is_not_a_stability_gate"
        )
        is not True
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed stability evidence is inconsistent"
        )
    shared_digest = evidence.get("shared_champion_spec_digest")
    shared_spec = evidence.get("shared_champion_spec")
    if spec_stable:
        shared_core = (
            {
                key: value
                for key, value in shared_spec.items()
                if key != "spec_digest"
            }
            if isinstance(shared_spec, Mapping)
            else {}
        )
        if (
            not _is_sha256(shared_digest)
            or not isinstance(shared_spec, Mapping)
            or shared_spec.get("spec_digest") != shared_digest
            or _object_digest(shared_core) != shared_digest
            or any(
                row["champion_spec_digest"] != shared_digest
                for row in per_seed
            )
        ):
            raise R10CrossSeedAggregateError(
                "cross-seed shared champion commitment differs"
            )
    elif shared_digest is not None or shared_spec is not None:
        raise R10CrossSeedAggregateError(
            "unstable champion cannot have one shared commitment"
        )
    shared_families = evidence.get(
        "shared_stable_cross_fold_eligible_families"
    )
    if family_sets_equal:
        if (
            shared_families
            != per_seed[0]["stable_cross_fold_eligible_families"]
        ):
            raise R10CrossSeedAggregateError(
                "cross-seed shared family commitment differs"
            )
    elif shared_families is not None:
        raise R10CrossSeedAggregateError(
            "different family sets cannot have one shared commitment"
        )

    aggregate_threshold = _require_fraction(
        evidence.get("minimum_nested_outer_gate_pass_fraction"),
        context="cross-seed aggregate threshold",
    )
    if (
        not math.isclose(
            aggregate_threshold,
            r10.MIN_DEVELOPMENT_FOLD_PASS_FRACTION,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or any(
            not math.isclose(
                row["minimum_nested_outer_gate_pass_fraction"],
                aggregate_threshold,
                abs_tol=1e-12,
                rel_tol=0.0,
            )
            for row in per_seed
        )
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed nested threshold commitment differs"
        )
    fractions = [
        float(row["nested_outer_gate_pass_fraction"])
        for row in per_seed
    ]
    minimum_observed = _require_fraction(
        evidence.get(
            "minimum_observed_nested_outer_gate_pass_fraction"
        ),
        context="cross-seed minimum observed nested fraction",
    )
    mean_observed = _require_fraction(
        evidence.get("mean_observed_nested_outer_gate_pass_fraction"),
        context="cross-seed mean observed nested fraction",
    )
    if (
        not math.isclose(
            minimum_observed,
            min(fractions),
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or not math.isclose(
            mean_observed,
            sum(fractions) / len(fractions),
            abs_tol=1e-12,
            rel_tol=0.0,
        )
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed nested aggregate statistics differ"
        )

    strong = bool(
        spec_stable
        and family_sets_equal
        and folds_distinct
        and all(
            row["single_seed_strong_evidence_passed"]
            for row in per_seed
        )
    )
    if evidence.get("cross_seed_development_signal_passed") is not strong:
        raise R10CrossSeedAggregateError(
            "cross-seed development evidence decision differs"
        )
    reason_codes = evidence.get("failure_reason_codes")
    expected_reason_codes: list[str] = []
    if not spec_stable:
        expected_reason_codes.append(
            "champion_spec_not_stable_across_seeds"
        )
    if not family_sets_equal:
        expected_reason_codes.append(
            "stable_family_sets_differ_across_seeds"
        )
    if not folds_distinct:
        expected_reason_codes.append(
            "fold_assignments_not_distinct_across_seeds"
        )
    for row in per_seed:
        seed = row["seed"]
        if not row["single_seed_development_signal_passed"]:
            expected_reason_codes.append(
                f"seed_{seed}_development_signal_failed"
            )
        if not row["nested_outer_gate_fraction_passed"]:
            expected_reason_codes.append(
                f"seed_{seed}_nested_outer_fraction_failed"
            )
        if not row["all_requested_folds_usable"]:
            expected_reason_codes.append(
                f"seed_{seed}_requested_folds_not_all_usable"
            )
        if not row["all_nested_outer_folds_evaluable"]:
            expected_reason_codes.append(
                f"seed_{seed}_nested_outer_not_all_evaluable"
            )
        if not row["stable_family_support_passed"]:
            expected_reason_codes.append(
                f"seed_{seed}_stable_family_support_failed"
            )
        if not row["champion_reusable_identity"]:
            expected_reason_codes.append(
                f"seed_{seed}_champion_not_reusable_identity"
            )
    expected_reason_codes = sorted(set(expected_reason_codes))
    if (
        not isinstance(reason_codes, list)
        or reason_codes != sorted(set(reason_codes))
        or any(not isinstance(value, str) or not value for value in reason_codes)
        or reason_codes != expected_reason_codes
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed failure-reason closure differs"
        )

    decision = _require_mapping(
        summary.get("decision"),
        context="cross-seed decision",
    )
    expected_status = (
        STATUS_NEED_FRESH_HOLDOUT
        if strong
        else STATUS_CONTINUE_TO_R10B
    )
    if (
        decision.get("status") != expected_status
        or done.get("decision_status") != expected_status
        or decision.get("cross_seed_aggregation_passed") is not strong
        or decision.get("cross_seed_development_signal_passed") is not strong
        or decision.get("development_candidate_passed") is not False
        or decision.get("fresh_holdout_available") is not False
        or any(decision.get(field) is not False for field in GATE_FIELDS)
        or any(done.get(field) is not False for field in GATE_FIELDS)
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed decision or gate closure differs"
        )
    safety = _require_mapping(
        summary.get("safety"),
        context="cross-seed safety",
    )
    if (
        safety.get("fresh_holdout_is_mandatory_before_any_promotion")
        is not True
        or safety.get("legacy_test_is_not_a_fresh_promotion_holdout")
        is not True
        or any(
            safety.get(field) != 0
            for field in (
                "video_files_read",
                "video_files_copied",
                "renderer_calls",
                "training_jobs_started",
            )
        )
        or summary.get("formal_evidence") is not False
        or summary.get("training_authorized") is not False
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed safety closure differs"
        )


def validate_published_aggregate(output_dir: Path) -> dict[str, Any]:
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    root = unresolved.resolve(strict=True)
    if {path.name for path in root.iterdir()} != set(OUTPUT_NAMES):
        raise R10CrossSeedAggregateError(
            "cross-seed published artifact closure differs"
        )
    artifact_permissions.assert_sealed_tree(root)
    payload_bytes = {
        name: (root / name).read_bytes() for name in OUTPUT_NAMES
    }
    try:
        summary = json.loads(payload_bytes[SUMMARY_NAME])
        done = json.loads(payload_bytes[DONE_NAME])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise R10CrossSeedAggregateError(
            "cross-seed summary/done is invalid JSON"
        ) from error
    summary = _require_mapping(summary, context="cross-seed summary")
    done = _require_mapping(done, context="cross-seed done")
    records = _require_mapping(
        done.get("payload_files"),
        context="cross-seed payload registry",
    )
    if set(records) != set(PAYLOAD_NAMES):
        raise R10CrossSeedAggregateError(
            "cross-seed payload registry differs"
        )
    record = _require_mapping(
        records[SUMMARY_NAME],
        context="cross-seed summary payload record",
    )
    if (
        record.get("sha256")
        != _digest_bytes(payload_bytes[SUMMARY_NAME])
        or record.get("bytes") != len(payload_bytes[SUMMARY_NAME])
        or done.get("artifact_digest") != _object_digest(dict(records))
    ):
        raise R10CrossSeedAggregateError(
            "cross-seed payload digest differs"
        )
    _validate_summary_semantics(summary, done)
    return {
        "root": str(root),
        "summary": dict(summary),
        "done": dict(done),
    }


def build_aggregate(
    *,
    seed_artifact_dirs: Sequence[Path],
    output_dir: Path,
    expected_source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Build once, or validate and reuse an existing identical final output."""

    records = [
        _validate_seed_artifact(Path(path))
        for path in seed_artifact_dirs
    ]
    summary = _aggregate_records(
        records,
        expected_source_tree_sha256=expected_source_tree_sha256,
    )
    resolved_output = output_dir.expanduser().resolve(strict=False)
    if resolved_output.exists() or resolved_output.is_symlink():
        validated = validate_published_aggregate(resolved_output)
        expected_inputs = {
            str(record["seed"]): _input_identity(record)
            for record in sorted(records, key=lambda value: value["seed"])
        }
        if validated["summary"].get("inputs") != expected_inputs:
            raise R10CrossSeedAggregateError(
                "existing aggregate is bound to different seed artifacts"
            )
        if _canonical_json(validated["summary"]) != _canonical_json(summary):
            raise R10CrossSeedAggregateError(
                "existing aggregate differs from the recomputed summary"
            )
        return {
            "created": False,
            "output_dir": validated["root"],
            "summary": validated["summary"],
            "validated": validated,
        }
    _publish(resolved_output, summary=summary)
    validated = validate_published_aggregate(resolved_output)
    return {
        "created": True,
        "output_dir": validated["root"],
        "summary": validated["summary"],
        "validated": validated,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate two immutable R10A seed artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument(
        "--seed-artifact-dir",
        action="append",
        required=True,
        type=Path,
        help=(
            "One sealed R10A seed artifact directory; pass exactly twice "
            "for seeds 260108837 and 260108838."
        ),
    )
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument(
        "--expected-source-tree-sha256",
        help=(
            "Optional external source-snapshot tree anchor; when supplied "
            "both sealed seed summaries must match it."
        ),
    )
    validate = commands.add_parser("validate")
    validate.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_aggregate(
            seed_artifact_dirs=args.seed_artifact_dir,
            output_dir=args.output_dir,
            expected_source_tree_sha256=(
                args.expected_source_tree_sha256
            ),
        )
        print(
            "[motive-r10a-cross-seed] "
            f"created={result['created']} "
            f"status={result['summary']['decision']['status']} "
            "representation_gate_passed=False "
            f"output={result['output_dir']}",
            flush=True,
        )
    else:
        result = validate_published_aggregate(args.output_dir)
        print(
            "[motive-r10a-cross-seed] "
            "validated=True "
            f"status={result['summary']['decision']['status']} "
            f"output={result['root']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
