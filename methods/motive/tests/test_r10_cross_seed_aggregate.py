from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from motive import r10_dynamic_dino_representation_search as r10
from motive import r10_cross_seed_aggregate as aggregate
from motive.r10_cross_seed_aggregate import (
    DONE_NAME,
    OUTPUT_NAMES,
    STATUS_CONTINUE_TO_R10B,
    STATUS_NEED_FRESH_HOLDOUT,
    SUMMARY_NAME,
    R10CrossSeedAggregateError,
    build_aggregate,
    validate_published_aggregate,
)


SOURCE_TREE_SHA256 = "4" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _transform_arrays(spec_digest: str) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray([r10.TRANSFORM_SCHEMA]),
        "spec_digest": np.asarray([spec_digest]),
        "raw_mean": np.zeros(2, dtype=np.float64),
        "raw_scale": np.ones(2, dtype=np.float64),
        "appearance_mean": np.zeros(1, dtype=np.float64),
        "appearance_scale": np.ones(1, dtype=np.float64),
        "content_ridge": np.zeros((1, 2), dtype=np.float64),
        "projection": np.eye(2, dtype=np.float64),
        "action_head": np.zeros((2, 0), dtype=np.float64),
        "action_families": np.asarray([], dtype="<U1"),
        "geometry_keep": np.asarray([0], dtype=np.int64),
        "raw_dimension": np.asarray([2], dtype=np.int64),
        "embedding_dimension": np.asarray([2], dtype=np.int64),
        "dino_mean": np.zeros(2, dtype=np.float64),
        "dino_basis": np.eye(2, dtype=np.float64),
        "appearance_dino_mean": np.zeros(2, dtype=np.float64),
        "appearance_dino_basis": np.eye(2, dtype=np.float64),
    }


def _seed_artifact(
    root: Path,
    *,
    seed: int,
    signal: bool = True,
    legacy_passed: bool = True,
    spec_name: str = "shared_dynamic_identity",
    binding_salt: str = "shared",
    stable_families: tuple[str, ...] = tuple(
        f"action-{index:02d}" for index in range(8)
    ),
    assignment_digest: str | None = None,
    requested_repeats: int = r10.DEFAULT_REPEATS,
    requested_folds: int = r10.DEFAULT_FOLDS,
    minimum_stable_families: int = r10.MIN_STABLE_FAMILIES,
    minimum_cohort_fraction: float = r10.MIN_COHORT_COVERAGE,
    coverage_fraction: float | None = None,
) -> Path:
    if assignment_digest is None:
        assignment_digest = hashlib.sha256(
            f"development-fold-assignment-{seed}".encode()
        ).hexdigest()
    spec_core = {
        "schema_version": "motive-r10a-dynamic-spec-v1",
        "name": spec_name,
        "raw_blocks": ["dino_edit_signed_dct"],
        "dino_channel_dim": 64,
        "standardize": True,
        "content_residual": True,
        "content_covariate":
            "source_and_target_pooled_dino_train_pca32",
        "projection": "jl",
        "projection_dim": 128,
        "head": "identity",
        "ridge": 0.0,
        "geometry_keep": 0,
        "similarity": "cosine",
        "selection_role": "reusable_representation_candidate",
        "champion_eligible": True,
    }
    spec = {
        **spec_core,
        "spec_digest": _object_digest(spec_core),
    }
    nested_records = [
        {
            "outer_fold_id": f"repeat_{repeat}_fold_{fold}",
            "repeat": repeat,
            "inner_fold_count": 2,
            "inner_folds": [],
            "outer_query_seen_by_inner_fit": False,
            "outer_evaluable": True,
            "selected_spec_digest": spec["spec_digest"],
            "selected_spec_name": spec_name,
            "outer_failure_codes": [],
            "outer_gate_passed": True,
            "outer_metrics": {},
        }
        for repeat in range(requested_repeats)
        for fold in range(requested_folds)
    ]
    fold_rows = [
        {
            "schema_version": r10.FOLD_SCHEMA,
            "fold_id": f"repeat_{repeat}_fold_{fold}",
            "seed": seed,
            "assignment_commitment": {
                name: hashlib.sha256(
                    (
                        f"{assignment_digest}:"
                        f"repeat_{repeat}_fold_{fold}:{name}"
                    ).encode()
                ).hexdigest()
                for name in (
                    "query_group_ids_sha256",
                    "query_iids_sha256",
                    "query_component_ids_sha256",
                )
            },
        }
        for repeat in range(requested_repeats)
        for fold in range(requested_folds)
    ]
    input_bindings = {
        "candidate_manifest_dir": "/sealed/candidate",
        "candidate_manifest_done_sha256":
            hashlib.sha256(
                f"candidate-{binding_salt}".encode()
            ).hexdigest(),
        "track_cache_final": "/sealed/track/final",
        "track_cache_done_sha256":
            hashlib.sha256(f"track-{binding_salt}".encode()).hexdigest(),
        "visual_features_final": "/sealed/visual/final",
        "visual_features_done_sha256":
            hashlib.sha256(f"visual-{binding_salt}".encode()).hexdigest(),
        "visual_candidates_manifest": "/sealed/candidates.jsonl",
        "visual_candidates_sha256":
            hashlib.sha256(
                f"visual-candidates-{binding_salt}".encode()
            ).hexdigest(),
    }
    coverage = {
        "r7_common_cohort": {
            "input_rows": 200,
            "modalities_share_exact_common_cohort": True,
        },
        "r7_common_rows": 180,
        "r10_paired_source_target_dino_rows": 176,
        "r10_common_cohort_fraction_of_r7": (
            176 / 180
            if coverage_fraction is None
            else coverage_fraction
        ),
        "r10_exclusion_reason_counts": {
            "source_dino_invalid": 4,
        },
        "all_specs_share_exact_r10_cohort": True,
        "minimum_required_fraction": minimum_cohort_fraction,
    }
    implementation_files = {
        "r10_dynamic_dino_representation_search.py": "1" * 64,
        "r7_artifact_permissions.py": "2" * 64,
        "r7_candidate_temporal_screen.py": "3" * 64,
    }
    arrays = _transform_arrays(spec["spec_digest"])
    summary = {
        "schema_version": r10.SEARCH_SCHEMA,
        "status": "complete",
        "seed": seed,
        "budget": {
            "requested_repeats": requested_repeats,
            "requested_folds_per_repeat": requested_folds,
            "realized_fold_rows": requested_repeats * requested_folds,
            "usable_search_folds": requested_repeats * requested_folds,
            "candidate_specs": 4,
            "champion_eligible_specs": 3,
            "closed_set_supervised_upper_bound_specs": 1,
        },
        "fold_protocol": {
            "seed_changes_group_fold_assignment": True,
            "development_fold_assignment_sha256":
                r10._fold_assignment_rows_digest(fold_rows),
            "assignment_digest_excludes_seed": True,
            "seed_is_stability_perturbation_not_independent_replication":
                True,
            "legacy_test_excluded_from_selection": True,
            "legacy_test_is_fresh_promotion_holdout": False,
        },
        "selection_protocol": {
            "minimum_development_fold_pass_fraction":
                r10.MIN_DEVELOPMENT_FOLD_PASS_FRACTION,
        },
        "nested_outer_model_selection": {
            "records": nested_records,
            "gate_pass_fraction": 1.0,
            "all_requested_folds_usable": True,
            "all_nested_outer_folds_evaluable": True,
            "stable_cross_fold_eligible_families":
                list(stable_families),
            "stable_family_count": len(stable_families),
            "minimum_stable_families": minimum_stable_families,
        },
        "champion": {
            "frozen_spec": spec,
            "single_seed_development_signal_passed": signal,
            "legacy_test_diagnostic_passed": legacy_passed,
        },
        "decision": {
            "single_seed_development_signal_passed": signal,
            "legacy_test_diagnostic_passed": legacy_passed,
            "cross_seed_aggregation_passed": False,
            "development_candidate_passed": False,
            "fresh_holdout_available": False,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        },
        "input_coverage": coverage,
        "input_bindings": input_bindings,
        "implementation": {
            "files": implementation_files,
            "bundle_sha256": _object_digest(implementation_files),
        },
        "source_snapshot": {
            "tree_sha256": SOURCE_TREE_SHA256,
            "exact_tree_verified_by_controller_before_search": True,
        },
        "frozen_transform": {
            "array_records": r10._array_records(arrays),
        },
    }
    output = root / f"seed_{seed}"
    r10._publish(
        output,
        trials=[
            {
                "schema_version": r10.TRIAL_SCHEMA,
                "trial_index": 0,
                "spec": spec,
            }
        ],
        folds=fold_rows,
        failures=[
            {
                "schema_version": r10.FAILURE_SCHEMA,
                "failure_code": "fresh_holdout_absent",
            }
        ],
        predictions=[
            {
                "schema_version":
                    "motive-r10a-champion-prediction-v1",
                "iid": f"example-{seed}",
            }
        ],
        summary=summary,
        transform_arrays=arrays,
    )
    r10.validate_published_search(output)
    return output


class CrossSeedAggregateTests(unittest.TestCase):
    def test_strong_evidence_only_requests_fresh_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(root, seed=260108837)
            second_seed = _seed_artifact(root, seed=260108838)
            final = root / "aggregate" / "final"
            result = build_aggregate(
                seed_artifact_dirs=[second_seed, first_seed],
                output_dir=final,
                expected_source_tree_sha256=SOURCE_TREE_SHA256,
            )
            self.assertTrue(result["created"])
            decision = result["summary"]["decision"]
            self.assertEqual(
                decision["status"],
                STATUS_NEED_FRESH_HOLDOUT,
            )
            self.assertTrue(
                decision["cross_seed_development_signal_passed"]
            )
            self.assertFalse(decision["development_candidate_passed"])
            self.assertFalse(decision["representation_gate_passed"])
            self.assertFalse(decision["renderer_probe_authorized"])
            self.assertFalse(decision["editor_training_authorized"])
            self.assertEqual(
                {path.name for path in final.iterdir()},
                set(OUTPUT_NAMES),
            )
            self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o555)
            for name in OUTPUT_NAMES:
                self.assertEqual(
                    stat.S_IMODE((final / name).stat().st_mode),
                    0o444,
                )
            validated = validate_published_aggregate(final)
            self.assertEqual(
                validated["summary"]["safety"]["video_files_read"],
                0,
            )

            repeated = build_aggregate(
                seed_artifact_dirs=[first_seed, second_seed],
                output_dir=final,
                expected_source_tree_sha256=SOURCE_TREE_SHA256,
            )
            self.assertFalse(repeated["created"])
            self.assertEqual(
                repeated["done"]["artifact_digest"]
                if "done" in repeated
                else repeated["validated"]["done"]["artifact_digest"],
                validated["done"]["artifact_digest"],
            )

    def test_legacy_diagnostic_is_reported_but_never_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(
                root,
                seed=260108837,
                legacy_passed=False,
            )
            second_seed = _seed_artifact(root, seed=260108838)
            result = build_aggregate(
                seed_artifact_dirs=[first_seed, second_seed],
                output_dir=root / "final",
            )
            evidence = result["summary"]["cross_seed_evidence"]
            self.assertTrue(
                evidence["legacy_test_diagnostics_are_non_gating"]
            )
            self.assertFalse(
                evidence["per_seed"][0][
                    "legacy_test_diagnostic_passed"
                ]
            )
            self.assertTrue(
                evidence["cross_seed_development_signal_passed"]
            )
            self.assertFalse(
                any(
                    "legacy_diagnostic" in code
                    for code in evidence["failure_reason_codes"]
                )
            )
            self.assertEqual(
                result["summary"]["decision"]["status"],
                STATUS_NEED_FRESH_HOLDOUT,
            )

    def test_identical_development_assignments_are_not_strong(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignment_digest = "a" * 64
            first_seed = _seed_artifact(
                root,
                seed=260108837,
                assignment_digest=assignment_digest,
            )
            second_seed = _seed_artifact(
                root,
                seed=260108838,
                assignment_digest=assignment_digest,
            )
            result = build_aggregate(
                seed_artifact_dirs=[first_seed, second_seed],
                output_dir=root / "final",
            )
            summary = result["summary"]
            evidence = summary["cross_seed_evidence"]
            self.assertNotEqual(
                summary["inputs"]["260108837"]["folds_sha256"],
                summary["inputs"]["260108838"]["folds_sha256"],
            )
            self.assertFalse(evidence["fold_assignments_distinct"])
            self.assertFalse(
                evidence["cross_seed_development_signal_passed"]
            )
            self.assertIn(
                "fold_assignments_not_distinct_across_seeds",
                evidence["failure_reason_codes"],
            )
            self.assertEqual(
                summary["decision"]["status"],
                STATUS_CONTINUE_TO_R10B,
            )

    def test_preregistered_budget_and_thresholds_are_locked(self) -> None:
        cases = (
            ("repeats", {"requested_repeats": 1}),
            ("folds", {"requested_folds": 2}),
            ("families", {"minimum_stable_families": 7}),
            ("cohort-minimum", {"minimum_cohort_fraction": 0.80}),
            ("cohort-ratio", {"coverage_fraction": 0.95}),
        )
        for name, overrides in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    first_seed = _seed_artifact(
                        root,
                        seed=260108837,
                        **overrides,
                    )
                    second_seed = _seed_artifact(
                        root,
                        seed=260108838,
                        **overrides,
                    )
                    final = root / "final"
                    with self.assertRaises(
                        R10CrossSeedAggregateError
                    ):
                        build_aggregate(
                            seed_artifact_dirs=[
                                first_seed,
                                second_seed,
                            ],
                            output_dir=final,
                        )
                    self.assertFalse(final.exists())

    def test_existing_output_must_match_recomputed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(root, seed=260108837)
            second_seed = _seed_artifact(root, seed=260108838)
            final = root / "final"
            build_aggregate(
                seed_artifact_dirs=[first_seed, second_seed],
                output_dir=final,
            )
            with self.assertRaisesRegex(
                R10CrossSeedAggregateError,
                "differs from the recomputed summary",
            ):
                build_aggregate(
                    seed_artifact_dirs=[first_seed, second_seed],
                    output_dir=final,
                    expected_source_tree_sha256=SOURCE_TREE_SHA256,
                )

    def test_failed_publish_removes_visible_and_staged_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(root, seed=260108837)
            second_seed = _seed_artifact(root, seed=260108838)
            parent = root / "aggregate"
            final = parent / "final"
            with mock.patch.object(
                aggregate,
                "_fsync_directory",
                side_effect=OSError("injected directory fsync failure"),
            ):
                with self.assertRaises(OSError):
                    build_aggregate(
                        seed_artifact_dirs=[first_seed, second_seed],
                        output_dir=final,
                    )
            self.assertFalse(final.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_validator_rejects_writable_or_incomplete_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(root, seed=260108837)
            second_seed = _seed_artifact(root, seed=260108838)
            final = root / "final"
            build_aggregate(
                seed_artifact_dirs=[first_seed, second_seed],
                output_dir=final,
            )
            final.chmod(0o700)
            try:
                with self.assertRaises(ValueError):
                    validate_published_aggregate(final)
            finally:
                final.chmod(0o555)

            incomplete = root / "incomplete"
            incomplete.mkdir()
            with self.assertRaisesRegex(
                R10CrossSeedAggregateError,
                "artifact closure differs",
            ):
                validate_published_aggregate(incomplete)

    def test_failed_single_seed_signal_continues_to_r10b(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(
                root,
                seed=260108837,
                signal=False,
            )
            second_seed = _seed_artifact(root, seed=260108838)
            final = root / "final"
            result = build_aggregate(
                seed_artifact_dirs=[first_seed, second_seed],
                output_dir=final,
            )
            self.assertEqual(
                result["summary"]["decision"]["status"],
                STATUS_CONTINUE_TO_R10B,
            )
            self.assertFalse(
                result["summary"]["decision"][
                    "cross_seed_aggregation_passed"
                ]
            )
            self.assertIn(
                "seed_260108837_development_signal_failed",
                result["summary"]["cross_seed_evidence"][
                    "failure_reason_codes"
                ],
            )
            self.assertFalse(
                result["summary"]["decision"][
                    "editor_training_authorized"
                ]
            )

    def test_unstable_champion_continues_to_r10b(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(root, seed=260108837)
            second_seed = _seed_artifact(
                root,
                seed=260108838,
                spec_name="different_dynamic_identity",
            )
            result = build_aggregate(
                seed_artifact_dirs=[first_seed, second_seed],
                output_dir=root / "final",
            )
            evidence = result["summary"]["cross_seed_evidence"]
            self.assertFalse(evidence["champion_spec_stable"])
            self.assertIsNone(evidence["shared_champion_spec"])
            self.assertEqual(
                result["summary"]["decision"]["status"],
                STATUS_CONTINUE_TO_R10B,
            )

    def test_mismatched_inputs_are_not_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(
                root,
                seed=260108837,
                binding_salt="first",
            )
            second_seed = _seed_artifact(
                root,
                seed=260108838,
                binding_salt="second",
            )
            with self.assertRaisesRegex(
                R10CrossSeedAggregateError,
                "different input bindings",
            ):
                build_aggregate(
                    seed_artifact_dirs=[first_seed, second_seed],
                    output_dir=root / "final",
                )
            self.assertFalse((root / "final").exists())

    def test_exact_preregistered_seed_set_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_seed = _seed_artifact(root, seed=260108837)
            with self.assertRaisesRegex(
                R10CrossSeedAggregateError,
                "duplicate",
            ):
                build_aggregate(
                    seed_artifact_dirs=[first_seed, first_seed],
                    output_dir=root / "final",
                )
            self.assertFalse((root / "final").exists())


if __name__ == "__main__":
    unittest.main()
