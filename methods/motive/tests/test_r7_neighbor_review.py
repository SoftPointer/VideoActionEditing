from __future__ import annotations

import copy
import hashlib
import inspect
import os
import stat
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

import motive.r7_neighbor_review as neighbor_review
from motive.r7_neighbor_audit_policy import (
    COHORT_DOUBLE_REVIEW_TARGETS,
    COHORT_PRIMARY_TARGETS,
    MERGED_REVIEW_SCHEMA,
    POPULATION_ROW_SCHEMA,
    policy_sha256,
)
from motive.r7_neighbor_review import (
    LABELS_DONE_NAME,
    LABELS_NAME,
    POPULATION_CONTEXT_NAME,
    POPULATION_DONE_NAME,
    POPULATION_NAME,
    REVIEW_BUNDLE_DONE_NAME,
    REVIEW_NAME,
    build_population_context,
    build_source_manifest,
    commit_population_manifest,
    commit_reviewer_labels,
    load_jsonl,
    merge_review_labels,
    population_context_sha256,
    prepare_reviewer_bundle,
    selection_sha256,
    statistical_unit_id,
    validate_population_manifest,
    validate_label_commit,
    validate_population_commit,
    validate_reviewer_bundle,
    validate_source_manifest,
    upstream_bindings_sha256,
)


PRIMARY_REVIEWER = "neighbor-reviewer-primary"
SECONDARY_REVIEWER = "neighbor-reviewer-secondary"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _risk_tier(row: dict[str, Any]) -> int:
    witness = row["witness"]
    if (
        witness["high_impact"]
        and witness["hard_edge"]
        and witness["msf_witness"]
    ):
        return 0
    if (
        witness["high_impact"]
        and not witness["hard_edge"]
        and (witness["cross_component"] or witness["cross_split"])
    ):
        return 1
    if witness["large_component_witness"]:
        return 2
    if witness["top_merge_witness"]:
        return 3
    return 4


def _make_population_fixture(
    root: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    media_root = root / "source-media"
    media_root.mkdir(parents=True)
    left_bytes = b"complete left video fixture bytes"
    right_bytes = b"complete right video fixture bytes"
    (media_root / "left.mp4").write_bytes(left_bytes)
    (media_root / "right.mp4").write_bytes(right_bytes)
    media = [
        {
            "relative_path": "left.mp4",
            "sha256": _sha_bytes(left_bytes),
            "size_bytes": len(left_bytes),
            "complete_video": True,
        },
        {
            "relative_path": "right.mp4",
            "sha256": _sha_bytes(right_bytes),
            "size_bytes": len(right_bytes),
            "complete_video": True,
        },
    ]
    bindings = {
        "indexed_graph_artifact_digest": _sha_text("indexed graph"),
        "dino_edges_artifact_digest": _sha_text("dino edges"),
        "sampling_population_sha256": _sha_text("upstream pair population"),
        "validated_quotient_artifact_digest": _sha_text(
            "validated quotient graph"
        ),
        "base_component_population_sha256": _sha_text(
            "base component population"
        ),
    }
    population: list[dict[str, Any]] = []
    counter = 0
    for cohort, sample_target in COHORT_PRIMARY_TARGETS.items():
        for local_index in range(sample_target + 5):
            pair = [
                f"base-{counter:05d}-a",
                f"base-{counter:05d}-b",
            ]
            unit_id = statistical_unit_id(pair)
            if cohort == "hard":
                score = 0.97
                top_neighbor = True
                priority = False
            elif cohort == "boundary":
                score = 0.94
                top_neighbor = True
                priority = False
            elif cohort == "below_floor":
                score = 0.91
                top_neighbor = True
                priority = False
            elif cohort == "far_negative":
                score = 0.10
                top_neighbor = False
                priority = False
            else:
                score = 0.97 if local_index % 2 == 0 else 0.90
                top_neighbor = local_index % 3 == 0
                priority = True
            hard_edge = score >= 0.96
            witness = {
                "high_impact": False,
                "hard_edge": hard_edge,
                "top_neighbor": top_neighbor,
                "msf_witness": False,
                "cross_component": True,
                "cross_split": local_index % 2 == 0,
                "priority": priority,
                "top_merge_witness": False,
                "large_component_witness": False,
            }
            if cohort == "component_risk":
                risk_kind = local_index % 5
                if risk_kind == 0:
                    witness.update(
                        {
                            "high_impact": True,
                            "hard_edge": True,
                            "msf_witness": True,
                        }
                    )
                    score = 0.97
                elif risk_kind == 1:
                    witness.update(
                        {
                            "high_impact": True,
                            "hard_edge": False,
                            "cross_component": True,
                        }
                    )
                    score = 0.90
                elif risk_kind == 2:
                    witness["large_component_witness"] = True
                elif risk_kind == 3:
                    witness["top_merge_witness"] = True
                witness["hard_edge"] = score >= 0.96
            population.append(
                {
                    "schema_version": POPULATION_ROW_SCHEMA,
                    "policy_sha256": policy_sha256(),
                    "thresholds_human_calibrated": False,
                    "statistical_unit_id": unit_id,
                    "base_component_pair": pair,
                    "source_bindings": dict(bindings),
                    "hidden_context": {
                        "score": score,
                        "score_bin": f"{cohort}-bin",
                        "threshold_relation": cohort,
                        "anchor_flags": [
                            local_index % 7 == 0,
                            local_index % 11 == 0,
                        ],
                        "qwen_evidence_sha256": [
                            _sha_text(f"qwen-left-{counter}"),
                            _sha_text(f"qwen-right-{counter}"),
                        ],
                        "iid_pair": [
                            f"iid-left-{counter}",
                            f"iid-right-{counter}",
                        ],
                        "provisional_split_pair": [
                            "train",
                            "validation" if local_index % 2 else "test",
                        ],
                    },
                    "witness": witness,
                    "media": copy.deepcopy(media),
                }
            )
            counter += 1
    # Input order is intentionally not canonical.  The digest and global
    # bottom-hash selection must not rely on producer ordering.
    population.reverse()
    context = build_population_context(population)
    return media_root, population, context


def _completed(
    templates: list[dict[str, Any]],
    *,
    verdict: str = "must_same_split",
    reason: str = "same_clip_or_transcode",
) -> list[dict[str, Any]]:
    labels = copy.deepcopy(templates)
    for row in labels:
        row["verdict"] = verdict
        row["reason_codes"] = [reason]
        row["notes"] = None
        row["completed_at_utc"] = "2026-07-28T01:02:03Z"
        row["review_attestation"] = {
            "video_1_reviewed_in_full": True,
            "video_2_reviewed_in_full": True,
            "independent_judgment": True,
            "other_reviewer_result_not_observed": True,
        }
    return labels


class R7NeighborReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        (
            cls.media_root,
            cls.population_rows,
            cls.population_context,
        ) = _make_population_fixture(cls.root)
        cls.population_sha256 = cls.population_context[
            "population_sha256"
        ]
        cls.population_context_sha256 = population_context_sha256(
            cls.population_context
        )
        cls.upstream_bindings_sha256 = upstream_bindings_sha256(
            cls.population_context["source_bindings"]
        )
        cls.population_root = cls.root / "population-commit"
        cls.population_done = commit_population_manifest(
            cls.population_rows,
            population_context=cls.population_context,
            expected_population_sha256=cls.population_sha256,
            expected_population_context_sha256=
                cls.population_context_sha256,
            expected_upstream_bindings_sha256=
                cls.upstream_bindings_sha256,
            output_directory=cls.population_root,
        )
        cls.population_commit_digest = cls.population_done[
            "artifact_digest"
        ]
        cls.source_rows = build_source_manifest(
            cls.population_root,
            expected_population_sha256=cls.population_sha256,
            expected_population_context_sha256=
                cls.population_context_sha256,
            expected_upstream_bindings_sha256=
                cls.upstream_bindings_sha256,
            expected_population_commit_digest=
                cls.population_commit_digest,
        )
        cls.primary_root = cls.root / "primary-reviewer-only"
        cls.secondary_root = cls.root / "secondary-reviewer-only"
        cls.primary_done = prepare_reviewer_bundle(
            cls.source_rows,
            population_commit_root=cls.population_root,
            expected_population_sha256=cls.population_sha256,
            expected_population_context_sha256=
                cls.population_context_sha256,
            expected_upstream_bindings_sha256=
                cls.upstream_bindings_sha256,
            expected_population_commit_digest=
                cls.population_commit_digest,
            source_media_root=cls.media_root,
            output_directory=cls.primary_root,
            reviewer_role="primary",
            reviewer_id=PRIMARY_REVIEWER,
        )
        cls.secondary_done = prepare_reviewer_bundle(
            cls.source_rows,
            population_commit_root=cls.population_root,
            expected_population_sha256=cls.population_sha256,
            expected_population_context_sha256=
                cls.population_context_sha256,
            expected_upstream_bindings_sha256=
                cls.upstream_bindings_sha256,
            expected_population_commit_digest=
                cls.population_commit_digest,
            source_media_root=cls.media_root,
            output_directory=cls.secondary_root,
            reviewer_role="secondary",
            reviewer_id=SECONDARY_REVIEWER,
        )
        cls.primary_templates = load_jsonl(cls.primary_root / REVIEW_NAME)
        cls.secondary_templates = load_jsonl(
            cls.secondary_root / REVIEW_NAME
        )
        cls.primary_labels = _completed(cls.primary_templates)
        cls.secondary_labels = _completed(
            cls.secondary_templates,
            verdict="independent_content",
            reason="unrelated",
        )
        cls.primary_label_root = cls.root / "primary-label-commit"
        cls.secondary_label_root = cls.root / "secondary-label-commit"
        cls.primary_label_done = commit_reviewer_labels(
            cls.primary_labels,
            cls.source_rows,
            population_commit_root=cls.population_root,
            expected_population_sha256=cls.population_sha256,
            expected_population_context_sha256=
                cls.population_context_sha256,
            expected_upstream_bindings_sha256=
                cls.upstream_bindings_sha256,
            expected_population_commit_digest=
                cls.population_commit_digest,
            source_media_root=cls.media_root,
            reviewer_bundle_root=cls.primary_root,
            reviewer_role="primary",
            reviewer_id=PRIMARY_REVIEWER,
            output_directory=cls.primary_label_root,
        )
        cls.secondary_label_done = commit_reviewer_labels(
            cls.secondary_labels,
            cls.source_rows,
            population_commit_root=cls.population_root,
            expected_population_sha256=cls.population_sha256,
            expected_population_context_sha256=
                cls.population_context_sha256,
            expected_upstream_bindings_sha256=
                cls.upstream_bindings_sha256,
            expected_population_commit_digest=
                cls.population_commit_digest,
            source_media_root=cls.media_root,
            reviewer_bundle_root=cls.secondary_root,
            reviewer_role="secondary",
            reviewer_id=SECONDARY_REVIEWER,
            output_directory=cls.secondary_label_root,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _commit_primary_labels(
        self,
        labels: list[dict[str, Any]],
        output_directory: Path,
    ) -> dict[str, Any]:
        return commit_reviewer_labels(
            labels,
            self.source_rows,
            population_commit_root=self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
            source_media_root=self.media_root,
            reviewer_bundle_root=self.primary_root,
            reviewer_role="primary",
            reviewer_id=PRIMARY_REVIEWER,
            output_directory=output_directory,
        )

    def test_complete_population_digest_and_global_selection_are_replayed(
        self,
    ) -> None:
        population = validate_population_manifest(
            self.population_rows,
            population_context=self.population_context,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
        )
        expected_counts = {
            cohort: target + 5
            for cohort, target in COHORT_PRIMARY_TARGETS.items()
        }
        self.assertEqual(
            population["context"]["cohort_population_counts"],
            expected_counts,
        )
        source = validate_source_manifest(
            self.source_rows,
            population_commit_root=self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
            media_root=self.media_root,
        )
        self.assertEqual(len(source), 800)
        for cohort, target in COHORT_PRIMARY_TARGETS.items():
            population_cohort = population["cohorts"][cohort]
            if cohort == "component_risk":
                expected = sorted(
                    population_cohort,
                    key=lambda row: (
                        _risk_tier(row),
                        selection_sha256(
                            cohort,
                            row["statistical_unit_id"],
                        ),
                        row["statistical_unit_id"],
                    ),
                )[:target]
            else:
                expected = sorted(
                    population_cohort,
                    key=lambda row: (
                        selection_sha256(
                            cohort,
                            row["statistical_unit_id"],
                        ),
                        row["statistical_unit_id"],
                    ),
                )[:target]
            sampled = [row for row in source if row["cohort"] == cohort]
            self.assertEqual(
                [row["statistical_unit_id"] for row in sampled],
                [row["statistical_unit_id"] for row in expected],
            )
            design = sampled[0]["sampling_design"]
            self.assertEqual(design["population_size_N_h"], target + 5)
            self.assertEqual(design["sample_size_n_h"], target)
            if cohort == "component_risk":
                self.assertEqual(
                    design["design"],
                    "nonprobability_purposive_priority",
                )
                self.assertIsNone(design["inclusion_probability_pi_h"])
                self.assertIsNone(design["design_weight"])
            else:
                self.assertEqual(design["design"], "SRSWOR")
                self.assertAlmostEqual(
                    design["inclusion_probability_pi_h"],
                    target / (target + 5),
                )
                self.assertAlmostEqual(
                    design["design_weight"],
                    (target + 5) / target,
                )

    def test_population_commit_has_external_anchors_and_exact_modes(
        self,
    ) -> None:
        committed = validate_population_commit(
            self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
        )
        self.assertEqual(
            committed["done"]["source_bindings"][
                "validated_quotient_artifact_digest"
            ],
            self.population_context["source_bindings"][
                "validated_quotient_artifact_digest"
            ],
        )
        self.assertFalse(
            committed["done"]["thresholds_human_calibrated"]
        )
        self.assertEqual(
            stat.S_IMODE(self.population_root.stat().st_mode),
            0o555,
        )
        self.assertEqual(
            {path.name for path in self.population_root.iterdir()},
            {
                POPULATION_NAME,
                POPULATION_CONTEXT_NAME,
                POPULATION_DONE_NAME,
            },
        )
        for path in self.population_root.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        with self.assertRaisesRegex(ValueError, "population-commit anchor"):
            validate_population_commit(
                self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=_sha_text(
                    "wrong population commit"
                ),
            )
        try:
            os.chmod(self.population_root, 0o777)
            with self.assertRaisesRegex(ValueError, "root mode"):
                validate_population_commit(
                    self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                )
        finally:
            os.chmod(self.population_root, 0o555)

    def test_population_context_membership_and_sample_tampering_fail(self) -> None:
        truncated = copy.deepcopy(self.population_rows[:-1])
        with self.assertRaisesRegex(ValueError, "external population anchor"):
            validate_population_manifest(
                truncated,
                population_context=self.population_context,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
            )

        tampered_population = copy.deepcopy(self.population_rows)
        tampered_population[0]["hidden_context"]["score_bin"] = "tampered"
        rebuilt_context = build_population_context(tampered_population)
        with self.assertRaisesRegex(ValueError, "external population-context"):
            validate_population_manifest(
                tampered_population,
                population_context=rebuilt_context,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
            )

        population = validate_population_manifest(
            self.population_rows,
            population_context=self.population_context,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
        )
        sampled_ids = {
            row["statistical_unit_id"] for row in self.source_rows
        }
        unselected = next(
            row
            for cohort_rows in population["cohorts"].values()
            for row in cohort_rows
            if row["statistical_unit_id"] not in sampled_ids
        )
        tampered_source = copy.deepcopy(self.source_rows)
        replacement = tampered_source[0]
        for field in (
            "statistical_unit_id",
            "base_component_pair",
            "source_bindings",
            "hidden_context",
            "witness",
            "media",
        ):
            replacement[field] = copy.deepcopy(unselected[field])
        replacement["population_row_sha256"] = _sha_text("invented")
        with self.assertRaisesRegex(ValueError, "globally selected"):
            validate_source_manifest(
                tampered_source,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                media_root=self.media_root,
            )

        nonexistent = copy.deepcopy(self.source_rows)
        nonexistent[0]["statistical_unit_id"] = _sha_text(
            "not in population"
        )
        with self.assertRaisesRegex(ValueError, "globally selected"):
            validate_source_manifest(
                nonexistent,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                media_root=self.media_root,
            )

    def test_rebuilt_internal_chain_cannot_replace_external_upstream_anchor(
        self,
    ) -> None:
        forged_rows = copy.deepcopy(self.population_rows)
        forged_digest = _sha_text("forged validated quotient receipt")
        for row in forged_rows:
            row["source_bindings"][
                "validated_quotient_artifact_digest"
            ] = forged_digest
        forged_context = build_population_context(forged_rows)
        forged_population_sha256 = forged_context["population_sha256"]
        forged_context_sha256 = population_context_sha256(forged_context)
        forged_upstream_sha256 = upstream_bindings_sha256(
            forged_context["source_bindings"]
        )
        self.assertNotEqual(
            forged_upstream_sha256,
            self.upstream_bindings_sha256,
        )

        forged_root = self.root / "forged-upstream-population-commit"
        forged_done = commit_population_manifest(
            forged_rows,
            population_context=forged_context,
            expected_population_sha256=forged_population_sha256,
            expected_population_context_sha256=forged_context_sha256,
            expected_upstream_bindings_sha256=forged_upstream_sha256,
            output_directory=forged_root,
        )
        with self.assertRaisesRegex(
            ValueError,
            "external upstream-bindings anchor",
        ):
            validate_population_commit(
                forged_root,
                expected_population_sha256=forged_population_sha256,
                expected_population_context_sha256=forged_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    forged_done["artifact_digest"],
            )

    def test_every_formal_stage_requires_external_upstream_anchor(self) -> None:
        formal_apis = (
            validate_population_manifest,
            commit_population_manifest,
            validate_population_commit,
            build_source_manifest,
            validate_source_manifest,
            prepare_reviewer_bundle,
            validate_reviewer_bundle,
            commit_reviewer_labels,
            validate_label_commit,
            merge_review_labels,
        )
        for api in formal_apis:
            parameter = inspect.signature(api).parameters.get(
                "expected_upstream_bindings_sha256"
            )
            self.assertIsNotNone(parameter, api.__name__)
            self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_cohort_precedence_duplicate_and_undersized_risk_fail_closed(
        self,
    ) -> None:
        risk_row = next(
            row
            for row in self.population_rows
            if row["hidden_context"]["threshold_relation"]
            == "component_risk"
            and row["hidden_context"]["score"] >= 0.96
        )
        validated = validate_population_manifest(
            self.population_rows,
            population_context=self.population_context,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
        )
        self.assertIn(
            risk_row["statistical_unit_id"],
            {
                row["statistical_unit_id"]
                for row in validated["cohorts"]["component_risk"]
            },
        )
        precedence_tamper = copy.deepcopy(self.population_rows)
        target = next(
            row
            for row in precedence_tamper
            if row["statistical_unit_id"] == risk_row["statistical_unit_id"]
        )
        target["hidden_context"]["threshold_relation"] = "hard"
        with self.assertRaisesRegex(ValueError, "cohort precedence"):
            build_population_context(precedence_tamper)

        duplicate = copy.deepcopy(self.population_rows)
        duplicate.append(copy.deepcopy(duplicate[0]))
        with self.assertRaisesRegex(ValueError, "repeats a statistical unit"):
            build_population_context(duplicate)

        undersized = [
            copy.deepcopy(row)
            for row in self.population_rows
            if (
                row["hidden_context"]["threshold_relation"]
                != "component_risk"
            )
        ]
        risk_population = [
            copy.deepcopy(row)
            for row in self.population_rows
            if row["hidden_context"]["threshold_relation"]
            == "component_risk"
        ]
        undersized.extend(risk_population[:79])
        undersized_context = build_population_context(undersized)
        with self.assertRaisesRegex(ValueError, "requires at least 80"):
            validate_population_manifest(
                undersized,
                population_context=undersized_context,
                expected_population_sha256=
                    undersized_context["population_sha256"],
                expected_population_context_sha256=
                    population_context_sha256(undersized_context),
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
            )

    def test_independent_bundles_have_exact_disjoint_closures(self) -> None:
        self.assertEqual(self.primary_done["review_count"], 800)
        self.assertEqual(self.secondary_done["review_count"], 160)
        self.assertFalse(self.primary_done["training_authorized"])
        self.assertFalse(self.secondary_done["training_authorized"])
        self.assertNotEqual(
            self.primary_done["assignment_set_digest"],
            self.secondary_done["assignment_set_digest"],
        )
        validate_reviewer_bundle(
            self.source_rows,
            population_commit_root=self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
            source_media_root=self.media_root,
            bundle_root=self.primary_root,
            reviewer_role="primary",
            reviewer_id=PRIMARY_REVIEWER,
        )
        validate_reviewer_bundle(
            self.source_rows,
            population_commit_root=self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
            source_media_root=self.media_root,
            bundle_root=self.secondary_root,
            reviewer_role="secondary",
            reviewer_id=SECONDARY_REVIEWER,
        )
        self.assertEqual(
            {
                path.name
                for path in self.primary_root.iterdir()
            },
            {"media", REVIEW_NAME, REVIEW_BUNDLE_DONE_NAME},
        )
        self.assertEqual(
            {
                path.name
                for path in self.secondary_root.iterdir()
            },
            {"media", REVIEW_NAME, REVIEW_BUNDLE_DONE_NAME},
        )
        signature = inspect.signature(prepare_reviewer_bundle)
        self.assertNotIn("primary_bundle", signature.parameters)
        self.assertNotIn("primary_labels", signature.parameters)
        self.assertFalse(
            hasattr(neighbor_review, "prepare_review_bundle"),
            "mixed distributable bundle API must not exist",
        )
        primary_ids = {
            row["review_instance_id"] for row in self.primary_templates
        }
        secondary_ids = {
            row["review_instance_id"] for row in self.secondary_templates
        }
        self.assertTrue(primary_ids.isdisjoint(secondary_ids))
        for done in (self.primary_done, self.secondary_done):
            self.assertFalse(done["thresholds_human_calibrated"])

    def test_bundle_root_and_directory_mode_tamper_is_rejected(self) -> None:
        try:
            os.chmod(self.primary_root, 0o777)
            with self.assertRaisesRegex(ValueError, "root mode"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(self.primary_root, 0o555)
        media_directory = self.primary_root / "media"
        try:
            os.chmod(media_directory, 0o777)
            with self.assertRaisesRegex(ValueError, "directory mode"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(media_directory, 0o555)

    def test_reviewer_templates_hide_all_upstream_semantics(self) -> None:
        hidden_keys = {
            "score",
            "score_bin",
            "cohort",
            "threshold_relation",
            "anchor",
            "anchor_flags",
            "qwen",
            "qwen_evidence_sha256",
            "iid",
            "iid_pair",
            "component",
            "base_component_pair",
            "provisional_split_pair",
            "annotator_slot",
            "primary_review",
            "primary_result",
        }

        def keys(value: Any) -> set[str]:
            if isinstance(value, dict):
                result = set(value)
                for item in value.values():
                    result.update(keys(item))
                return result
            if isinstance(value, list):
                result: set[str] = set()
                for item in value:
                    result.update(keys(item))
                return result
            return set()

        for root, rows in (
            (self.primary_root, self.primary_templates),
            (self.secondary_root, self.secondary_templates),
        ):
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o555)
            for directory in (root / "media").rglob("*"):
                if directory.is_dir():
                    self.assertEqual(
                        stat.S_IMODE(directory.stat().st_mode),
                        0o555,
                    )
            for row in rows:
                self.assertFalse(keys(row) & hidden_keys)
                self.assertIsNone(row["verdict"])
                self.assertEqual(row["reason_codes"], [])
                self.assertEqual(len(row["media"]), 2)
                for binding in row["media"]:
                    self.assertRegex(
                        binding["relative_path"],
                        r"^media/[0-9a-f]{64}/video_[12]\.[a-z0-9]+$",
                    )
                    path = root / binding["relative_path"]
                    self.assertTrue(path.is_file())
                    self.assertFalse(path.is_symlink())
                    self.assertEqual(
                        stat.S_IMODE(path.stat().st_mode),
                        0o444,
                    )

    def test_closure_rejects_extra_file_directory_and_symlink(self) -> None:
        extra = self.primary_root / "sidecar.txt"
        try:
            os.chmod(self.primary_root, 0o755)
            extra.write_text("not allowed", encoding="utf-8")
            os.chmod(self.primary_root, 0o555)
            with self.assertRaisesRegex(ValueError, "closure"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(self.primary_root, 0o755)
            extra.unlink(missing_ok=True)
            os.chmod(self.primary_root, 0o555)

        extra_directory = self.primary_root / "unexpected-empty-directory"
        try:
            os.chmod(self.primary_root, 0o755)
            extra_directory.mkdir()
            os.chmod(self.primary_root, 0o555)
            with self.assertRaisesRegex(ValueError, "closure"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(self.primary_root, 0o755)
            extra_directory.rmdir()
            os.chmod(self.primary_root, 0o555)

        symlink = self.primary_root / "forbidden-link.mp4"
        try:
            os.chmod(self.primary_root, 0o755)
            symlink.symlink_to(self.primary_root / REVIEW_NAME)
            os.chmod(self.primary_root, 0o555)
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(self.primary_root, 0o755)
            symlink.unlink(missing_ok=True)
            os.chmod(self.primary_root, 0o555)
        hardlink = self.primary_root / "forbidden-hardlink.jsonl"
        try:
            os.chmod(self.primary_root, 0o755)
            os.link(self.primary_root / REVIEW_NAME, hardlink)
            os.chmod(self.primary_root, 0o555)
            with self.assertRaisesRegex(ValueError, "hard-linked"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(self.primary_root, 0o755)
            hardlink.unlink(missing_ok=True)
            os.chmod(self.primary_root, 0o555)

    def test_bound_media_and_done_tampering_are_rejected(self) -> None:
        relative = self.primary_templates[0]["media"][0]["relative_path"]
        media_path = self.primary_root / relative
        original = media_path.read_bytes()
        try:
            os.chmod(media_path, 0o644)
            media_path.write_bytes(b"x" * len(original))
            os.chmod(media_path, 0o444)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(media_path, 0o644)
            media_path.write_bytes(original)
            os.chmod(media_path, 0o444)

        done_path = self.primary_root / REVIEW_BUNDLE_DONE_NAME
        original_done = done_path.read_bytes()
        try:
            os.chmod(done_path, 0o644)
            done_path.write_bytes(original_done + b"\n")
            os.chmod(done_path, 0o444)
            with self.assertRaisesRegex(ValueError, "canonically encoded"):
                validate_reviewer_bundle(
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                )
        finally:
            os.chmod(done_path, 0o644)
            done_path.write_bytes(original_done)
            os.chmod(done_path, 0o444)

    def test_verdict_reason_implications_are_frozen(self) -> None:
        must_labels = _completed(self.primary_templates)
        invalid_must = copy.deepcopy(must_labels)
        invalid_must[0]["reason_codes"] = ["same_action_only"]
        with self.assertRaisesRegex(ValueError, "must-same-split reasons"):
            self._commit_primary_labels(
                invalid_must,
                self.root / "invalid-must-label-commit",
            )
        independent = _completed(
            self.primary_templates,
            verdict="independent_content",
            reason="same_action_only",
        )
        self._commit_primary_labels(
            independent,
            self.root / "valid-independent-label-commit",
        )
        invalid_independent = copy.deepcopy(independent)
        invalid_independent[0]["reason_codes"] = [
            "same_generation_lineage"
        ]
        with self.assertRaisesRegex(ValueError, "independent-content reasons"):
            self._commit_primary_labels(
                invalid_independent,
                self.root / "invalid-independent-label-commit",
            )
        uncertain = _completed(
            self.primary_templates,
            verdict="uncertain",
            reason="same_subject_background_only",
        )
        self._commit_primary_labels(
            uncertain,
            self.root / "valid-uncertain-label-commit",
        )
        unreviewable = _completed(
            self.primary_templates,
            verdict="unreviewable",
            reason="media_failure",
        )
        unreviewable[0]["review_attestation"][
            "video_1_reviewed_in_full"
        ] = False
        # All rows must satisfy the media-failure attestation.
        for row in unreviewable:
            row["review_attestation"]["video_1_reviewed_in_full"] = False
        self._commit_primary_labels(
            unreviewable,
            self.root / "valid-unreviewable-label-commit",
        )

    def test_label_commits_are_independent_exact_closures(self) -> None:
        primary = validate_label_commit(
            self.primary_label_root,
            self.source_rows,
            population_commit_root=self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
            source_media_root=self.media_root,
            reviewer_bundle_root=self.primary_root,
            reviewer_role="primary",
            reviewer_id=PRIMARY_REVIEWER,
            expected_label_commit_digest=
                self.primary_label_done["artifact_digest"],
        )
        secondary = validate_label_commit(
            self.secondary_label_root,
            self.source_rows,
            population_commit_root=self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
            source_media_root=self.media_root,
            reviewer_bundle_root=self.secondary_root,
            reviewer_role="secondary",
            reviewer_id=SECONDARY_REVIEWER,
            expected_label_commit_digest=
                self.secondary_label_done["artifact_digest"],
        )
        self.assertEqual(len(primary["labels"]), 800)
        self.assertEqual(len(secondary["labels"]), 160)
        for root, done in (
            (self.primary_label_root, self.primary_label_done),
            (self.secondary_label_root, self.secondary_label_done),
        ):
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o555)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {LABELS_NAME, LABELS_DONE_NAME},
            )
            for path in root.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            self.assertFalse(done["thresholds_human_calibrated"])
            self.assertFalse(done["formal_report"])

    def test_label_commit_tamper_and_writable_root_are_rejected(self) -> None:
        try:
            os.chmod(self.primary_label_root, 0o777)
            with self.assertRaisesRegex(ValueError, "root mode"):
                validate_label_commit(
                    self.primary_label_root,
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    reviewer_bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                    expected_label_commit_digest=
                        self.primary_label_done["artifact_digest"],
                )
        finally:
            os.chmod(self.primary_label_root, 0o555)
        labels_path = self.primary_label_root / LABELS_NAME
        original = labels_path.read_bytes()
        try:
            os.chmod(labels_path, 0o644)
            labels_path.write_bytes(original + b"\n")
            os.chmod(labels_path, 0o444)
            with self.assertRaises(ValueError):
                validate_label_commit(
                    self.primary_label_root,
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    reviewer_bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                    expected_label_commit_digest=
                        self.primary_label_done["artifact_digest"],
                )
        finally:
            os.chmod(labels_path, 0o644)
            labels_path.write_bytes(original)
            os.chmod(labels_path, 0o444)
        sidecar = self.primary_label_root / "forbidden-sidecar.json"
        try:
            os.chmod(self.primary_label_root, 0o755)
            sidecar.write_text("{}", encoding="utf-8")
            os.chmod(sidecar, 0o444)
            os.chmod(self.primary_label_root, 0o555)
            with self.assertRaises(ValueError):
                validate_label_commit(
                    self.primary_label_root,
                    self.source_rows,
                    population_commit_root=self.population_root,
                    expected_population_sha256=self.population_sha256,
                    expected_population_context_sha256=
                        self.population_context_sha256,
                    expected_upstream_bindings_sha256=
                        self.upstream_bindings_sha256,
                    expected_population_commit_digest=
                        self.population_commit_digest,
                    source_media_root=self.media_root,
                    reviewer_bundle_root=self.primary_root,
                    reviewer_role="primary",
                    reviewer_id=PRIMARY_REVIEWER,
                    expected_label_commit_digest=
                        self.primary_label_done["artifact_digest"],
                )
        finally:
            os.chmod(self.primary_label_root, 0o755)
            sidecar.unlink(missing_ok=True)
            os.chmod(self.primary_label_root, 0o555)

    def test_merge_uses_two_roots_and_conserves_double_review(self) -> None:
        merged = merge_review_labels(
            self.source_rows,
            population_commit_root=self.population_root,
            expected_population_sha256=self.population_sha256,
            expected_population_context_sha256=
                self.population_context_sha256,
            expected_upstream_bindings_sha256=
                self.upstream_bindings_sha256,
            expected_population_commit_digest=
                self.population_commit_digest,
            source_media_root=self.media_root,
            primary_bundle_root=self.primary_root,
            secondary_bundle_root=self.secondary_root,
            primary_label_commit_root=self.primary_label_root,
            secondary_label_commit_root=self.secondary_label_root,
            expected_primary_label_commit_digest=
                self.primary_label_done["artifact_digest"],
            expected_secondary_label_commit_digest=
                self.secondary_label_done["artifact_digest"],
            primary_reviewer_id=PRIMARY_REVIEWER,
            secondary_reviewer_id=SECONDARY_REVIEWER,
        )
        self.assertEqual(len(merged), 800)
        double_counts = Counter(
            row["source"]["cohort"]
            for row in merged
            if row["secondary_review"] is not None
        )
        self.assertEqual(
            double_counts,
            Counter(COHORT_DOUBLE_REVIEW_TARGETS),
        )
        for row in merged:
            self.assertEqual(row["schema_version"], MERGED_REVIEW_SCHEMA)
            self.assertEqual(
                row["label_scope"],
                "split_threshold_audit_only",
            )
            self.assertFalse(row["thresholds_human_calibrated"])
            self.assertFalse(row["training_authorized"])
            self.assertFalse(row["direct_training_supervision_allowed"])

    def test_merge_accepts_only_committed_roots_and_external_digests(
        self,
    ) -> None:
        signature = inspect.signature(merge_review_labels)
        self.assertNotIn("primary_labels", signature.parameters)
        self.assertNotIn("secondary_labels", signature.parameters)
        with self.assertRaisesRegex(ValueError, "must differ"):
            merge_review_labels(
                self.source_rows,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                source_media_root=self.media_root,
                primary_bundle_root=self.primary_root,
                secondary_bundle_root=self.secondary_root,
                primary_label_commit_root=self.primary_label_root,
                secondary_label_commit_root=self.secondary_label_root,
                expected_primary_label_commit_digest=
                    self.primary_label_done["artifact_digest"],
                expected_secondary_label_commit_digest=
                    self.secondary_label_done["artifact_digest"],
                primary_reviewer_id=PRIMARY_REVIEWER,
                secondary_reviewer_id=PRIMARY_REVIEWER,
            )
        with self.assertRaisesRegex(ValueError, "external label-commit"):
            merge_review_labels(
                self.source_rows,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                source_media_root=self.media_root,
                primary_bundle_root=self.primary_root,
                secondary_bundle_root=self.secondary_root,
                primary_label_commit_root=self.primary_label_root,
                secondary_label_commit_root=self.secondary_label_root,
                expected_primary_label_commit_digest=_sha_text(
                    "wrong primary label commit"
                ),
                expected_secondary_label_commit_digest=
                    self.secondary_label_done["artifact_digest"],
                primary_reviewer_id=PRIMARY_REVIEWER,
                secondary_reviewer_id=SECONDARY_REVIEWER,
            )
        with self.assertRaises(ValueError):
            merge_review_labels(
                self.source_rows,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                source_media_root=self.media_root,
                primary_bundle_root=self.primary_root,
                secondary_bundle_root=self.secondary_root,
                primary_label_commit_root=self.secondary_label_root,
                secondary_label_commit_root=self.primary_label_root,
                expected_primary_label_commit_digest=
                    self.secondary_label_done["artifact_digest"],
                expected_secondary_label_commit_digest=
                    self.primary_label_done["artifact_digest"],
                primary_reviewer_id=PRIMARY_REVIEWER,
                secondary_reviewer_id=SECONDARY_REVIEWER,
            )

    def test_bundle_creation_is_create_only(self) -> None:
        with self.assertRaises(FileExistsError):
            prepare_reviewer_bundle(
                self.source_rows,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                source_media_root=self.media_root,
                output_directory=self.primary_root,
                reviewer_role="primary",
                reviewer_id=PRIMARY_REVIEWER,
            )
        with self.assertRaisesRegex(ValueError, "cannot be nested"):
            prepare_reviewer_bundle(
                self.source_rows,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                source_media_root=self.media_root,
                output_directory=self.primary_root / "nested-secondary",
                reviewer_role="secondary",
                reviewer_id=SECONDARY_REVIEWER,
            )
        with self.assertRaises(FileExistsError):
            commit_reviewer_labels(
                self.primary_labels,
                self.source_rows,
                population_commit_root=self.population_root,
                expected_population_sha256=self.population_sha256,
                expected_population_context_sha256=
                    self.population_context_sha256,
                expected_upstream_bindings_sha256=
                    self.upstream_bindings_sha256,
                expected_population_commit_digest=
                    self.population_commit_digest,
                source_media_root=self.media_root,
                reviewer_bundle_root=self.primary_root,
                reviewer_role="primary",
                reviewer_id=PRIMARY_REVIEWER,
                output_directory=self.primary_label_root,
            )


if __name__ == "__main__":
    unittest.main()
