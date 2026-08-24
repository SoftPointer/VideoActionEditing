from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import unittest
from math import comb
from pathlib import Path
from typing import Any
from unittest import mock

import motive.r7_human_audit_sample as human_audit_sample_module
from motive.human_review import merge as merge_human_review
from motive.r7_human_audit_report import (
    INDEPENDENT_REVIEWER_ATTESTATION_SCHEMA,
    KAPPA_BOOTSTRAP_DRAWS,
    KAPPA_BOOTSTRAP_SEED,
    Z_95,
    _build_gate,
    _double_review_report,
    _estimate_cohort,
    _hypergeometric_population_bound,
    build_human_audit_report,
)
from motive.r7_human_audit_policy import (
    HUMAN_AUDIT_POLICY,
    implementation_bundle_sha256,
    policy_payload,
    policy_sha256,
)
from motive.r7_human_audit_sample import (
    DONE_NAME,
    POSITIVES_NAME,
    PRIMARY_REVIEW_NAME,
    SAMPLED_MANIFEST_NAME,
    SECONDARY_MANIFEST_NAME,
    SECONDARY_REVIEW_NAME,
    SUMMARY_NAME,
    build_human_audit_sample,
)
from methods.motive.tests.test_r7_human_audit_sample import (
    PRIMARY_REVIEWER_ID,
    SECONDARY_REVIEWER_ID,
    _canonical,
    _rehash_source_artifacts,
    _write_formal_source,
    _write_source,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _merge(
    *,
    manifest: Path,
    labels: Path,
    output: Path,
) -> None:
    result = merge_human_review(
        argparse.Namespace(
            manifest=manifest,
            labels=labels,
            output=output,
            overwrite=False,
        )
    )
    if result != 0:
        raise AssertionError(f"human_review.merge returned {result}")


def _review_fixture(
    root: Path,
    *,
    diagnostic_unbound_media: bool = False,
) -> dict[str, Any]:
    source = (
        _write_source(root)
        if diagnostic_unbound_media
        else _write_formal_source(root)
    )
    source_done = json.loads(
        (
            source / human_audit_sample_module.SOURCE_DONE_NAME
        ).read_text(encoding="utf-8")
    )
    expected_implementation_digest = implementation_bundle_sha256()
    sample = root / "sample"
    sample_arguments: dict[str, Any] = {
        "source_dir": source,
        "output_dir": sample,
        "data_root": (
            None if diagnostic_unbound_media else root / "data"
        ),
        "primary_reviewer_id": PRIMARY_REVIEWER_ID,
        "secondary_reviewer_id": SECONDARY_REVIEWER_ID,
        "expected_implementation_bundle_digest":
            expected_implementation_digest,
        "expected_source_artifact_digest":
            source_done["artifact_digest"],
        "expected_source_input_sha256": source_done["input_sha256"],
        "diagnostic_unbound_media": diagnostic_unbound_media,
    }
    if diagnostic_unbound_media:
        sample_arguments.update(
            {
                "positive_sample": 6,
                "pseudo_negative_sample": 5,
                "review_sample": 4,
                "double_review_fraction": 0.6,
                "seed": 31,
            }
        )
    build_human_audit_sample(**sample_arguments)
    manifest_rows = {
        str(row["iid"]): row
        for row in _read_jsonl(sample / SAMPLED_MANIFEST_NAME)
    }
    primary_labels = root / "primary.labels.jsonl"
    primary_templates = _read_jsonl(sample / PRIMARY_REVIEW_NAME)
    for template in primary_templates:
        cohort = manifest_rows[str(template["iid"])][
            "r7_human_audit_sampling"
        ]["cohort"]
        template["verdict"] = (
            "valid_action"
            if cohort
            in {
                "pseudo_positive",
                "pseudo_positive_family_coverage",
            }
            else "static"
        )
        template["reviewer"] = "reviewer-primary"
    _write_jsonl(primary_labels, primary_templates)
    primary_merged = root / "primary.merged.jsonl"
    _merge(
        manifest=sample / SAMPLED_MANIFEST_NAME,
        labels=primary_labels,
        output=primary_merged,
    )

    secondary_labels = root / "secondary.labels.jsonl"
    secondary_templates = _read_jsonl(sample / SECONDARY_REVIEW_NAME)
    for index, template in enumerate(secondary_templates):
        primary = next(
            row
            for row in primary_templates
            if row["iid"] == template["iid"]
        )
        template["verdict"] = primary["verdict"]
        # Preserve a deterministic disagreement without adjudicating it.
        if index == 0:
            template["verdict"] = (
                "static"
                if primary["verdict"] == "valid_action"
                else "instruction_mismatch"
            )
        template["reviewer"] = "reviewer-secondary"
    _write_jsonl(secondary_labels, secondary_templates)
    secondary_merged = root / "secondary.merged.jsonl"
    _merge(
        manifest=sample / SECONDARY_MANIFEST_NAME,
        labels=secondary_labels,
        output=secondary_merged,
    )
    sample_done = json.loads(
        (sample / DONE_NAME).read_text(encoding="utf-8")
    )
    primary_labels_sha256 = hashlib.sha256(
        primary_labels.read_bytes()
    ).hexdigest()
    secondary_labels_sha256 = hashlib.sha256(
        secondary_labels.read_bytes()
    ).hexdigest()
    reviewer_attestation = root / "independent-review-process.json"
    reviewer_attestation.write_text(
        json.dumps(
            {
                "schema_version":
                    INDEPENDENT_REVIEWER_ATTESTATION_SCHEMA,
                "sample_artifact_digest":
                    sample_done["artifact_digest"],
                "assignment_set_digest":
                    sample_done["assignment_set_digest"],
                "primary_reviewer_id": PRIMARY_REVIEWER_ID,
                "secondary_reviewer_id": SECONDARY_REVIEWER_ID,
                "primary_labels_sha256": primary_labels_sha256,
                "secondary_labels_sha256": secondary_labels_sha256,
                "distinct_humans_attested": True,
                "secondary_blinded_to_primary_until_completion": True,
                "attestor_id": "audit-operations",
                "timestamp": "2026-07-28T12:00:00Z",
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source": source,
        "data_root": root / "data",
        "sample": sample,
        "expected_sample_artifact_digest": sample_done["artifact_digest"],
        "expected_implementation_bundle_digest":
            expected_implementation_digest,
        "expected_source_artifact_digest": source_done["artifact_digest"],
        "expected_source_input_sha256": source_done["input_sha256"],
        "primary_labels": primary_labels,
        "expected_primary_labels_sha256": primary_labels_sha256,
        "primary_merged": primary_merged,
        "secondary_labels": secondary_labels,
        "expected_secondary_labels_sha256": secondary_labels_sha256,
        "secondary_merged": secondary_merged,
        "independent_reviewer_attestation": reviewer_attestation,
        "expected_independent_reviewer_attestation_sha256":
            hashlib.sha256(
                reviewer_attestation.read_bytes()
            ).hexdigest(),
    }


def _build_fixture_report(
    fixture: dict[str, Any],
    *,
    output: Path,
    resume: bool = False,
) -> dict[str, Any]:
    return build_human_audit_report(
        sample_dir=fixture["sample"],
        expected_sample_artifact_digest=fixture[
            "expected_sample_artifact_digest"
        ],
        expected_implementation_bundle_digest=fixture[
            "expected_implementation_bundle_digest"
        ],
        expected_source_artifact_digest=fixture[
            "expected_source_artifact_digest"
        ],
        expected_source_input_sha256=fixture[
            "expected_source_input_sha256"
        ],
        primary_merged=fixture["primary_merged"],
        secondary_merged=fixture["secondary_merged"],
        primary_labels_path=fixture["primary_labels"],
        expected_primary_labels_sha256=fixture[
            "expected_primary_labels_sha256"
        ],
        secondary_labels_path=fixture["secondary_labels"],
        expected_secondary_labels_sha256=fixture[
            "expected_secondary_labels_sha256"
        ],
        independent_reviewer_attestation=fixture.get(
            "independent_reviewer_attestation"
        ),
        expected_independent_reviewer_attestation_sha256=fixture.get(
            "expected_independent_reviewer_attestation_sha256"
        ),
        output_path=output,
        resume=resume,
    )


class R7HumanAuditReportTests(unittest.TestCase):
    def test_biased_runtime_sampler_with_self_consistent_hashes_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def biased_rank(_seed: int, _scope: str, _value: str) -> str:
                # A compromised runtime can emit a fully self-consistent
                # sample while falsely claiming the on-disk sampler hash.
                return "0" * 64

            with mock.patch.object(
                human_audit_sample_module,
                "_stable_rank",
                side_effect=biased_rank,
            ):
                fixture = _review_fixture(
                    root,
                    diagnostic_unbound_media=True,
                )

            sampled = _read_jsonl(
                fixture["sample"] / SAMPLED_MANIFEST_NAME
            )
            sampled_iids = [str(row["iid"]) for row in sampled]
            self.assertEqual(sampled_iids, sorted(sampled_iids))
            summary = json.loads(
                (fixture["sample"] / SUMMARY_NAME).read_text(
                    encoding="utf-8"
                )
            )
            claimed_sampler_sha = hashlib.sha256(
                Path(
                    human_audit_sample_module.__file__
                ).resolve().read_bytes()
            ).hexdigest()
            self.assertEqual(
                summary["implementation_sha256"],
                claimed_sampler_sha,
            )
            self.assertEqual(
                summary["implementation_bundle"]["bundle_sha256"],
                implementation_bundle_sha256(),
            )

            with self.assertRaisesRegex(
                ValueError,
                "resume artifact differs",
            ):
                _build_fixture_report(
                    fixture,
                    output=root / "biased-sample-report.json",
                )

    def test_external_implementation_bundle_mismatch_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(
                root,
                diagnostic_unbound_media=True,
            )
            forged = dict(fixture)
            forged["expected_implementation_bundle_digest"] = "0" * 64
            with self.assertRaisesRegex(
                ValueError,
                "implementation bundle differs from the external",
            ):
                _build_fixture_report(
                    forged,
                    output=root / "wrong-implementation-bundle.json",
                )

    def test_changed_and_remerged_labels_miss_old_external_anchor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(
                root,
                diagnostic_unbound_media=True,
            )
            old_external_sha = fixture[
                "expected_primary_labels_sha256"
            ]
            rows = _read_jsonl(fixture["primary_labels"])
            rows[0]["notes"] = "changed after external attestation"
            _write_jsonl(fixture["primary_labels"], rows)
            self.assertNotEqual(
                hashlib.sha256(
                    fixture["primary_labels"].read_bytes()
                ).hexdigest(),
                old_external_sha,
            )
            result = merge_human_review(
                argparse.Namespace(
                    manifest=(
                        fixture["sample"] / SAMPLED_MANIFEST_NAME
                    ),
                    labels=fixture["primary_labels"],
                    output=fixture["primary_merged"],
                    overwrite=True,
                )
            )
            self.assertEqual(result, 0)

            with self.assertRaisesRegex(
                ValueError,
                "primary labels differ from the external expected digest",
            ):
                _build_fixture_report(
                    fixture,
                    output=root / "remerged-labels-report.json",
                )

    def test_missing_independent_reviewer_attestation_is_insufficient(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(
                root,
                diagnostic_unbound_media=True,
            )
            unasserted = dict(fixture)
            unasserted["independent_reviewer_attestation"] = None
            unasserted[
                "expected_independent_reviewer_attestation_sha256"
            ] = None
            report = _build_fixture_report(
                unasserted,
                output=root / "missing-attestation-report.json",
            )
            self.assertEqual(
                report["recommended_gate"]["status"],
                "INSUFFICIENT",
            )
            prerequisite = report["recommended_gate"][
                "prerequisite_checks"
            ]["external_independent_reviewer_attestation"]
            self.assertFalse(prerequisite["passed"])
            self.assertFalse(
                prerequisite["value"][
                    "external_process_attestation_verified"
                ]
            )
            self.assertFalse(
                prerequisite["value"]["independent_humans_attested"]
            )
            self.assertFalse(
                report["reviewer_evidence"][
                    "external_process_attestation_verified"
                ]
            )
            self.assertFalse(
                report["reviewer_evidence"][
                    "independent_humans_attested"
                ]
            )
            self.assertFalse(
                report["reviewer_evidence"][
                    "cryptographic_reviewer_identity_verified"
                ]
            )

    def test_reviewer_attestation_tamper_is_rejected(self) -> None:
        for attack in ("stale-external-hash", "rebound-wrong-binding"):
            temporary_context = tempfile.TemporaryDirectory()
            with self.subTest(attack=attack), temporary_context as temporary:
                root = Path(temporary)
                fixture = _review_fixture(
                    root,
                    diagnostic_unbound_media=True,
                )
                attestation_path = fixture[
                    "independent_reviewer_attestation"
                ]
                payload = json.loads(
                    attestation_path.read_text(encoding="utf-8")
                )
                if attack == "stale-external-hash":
                    payload["attestor_id"] = "post-attestation-mutator"
                    expected_message = (
                        "attestation differs from the external expected"
                    )
                else:
                    payload["primary_labels_sha256"] = "0" * 64
                    expected_message = (
                        "attestation binding differs: "
                        "primary_labels_sha256"
                    )
                attestation_path.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if attack == "rebound-wrong-binding":
                    fixture[
                        "expected_independent_reviewer_attestation_sha256"
                    ] = hashlib.sha256(
                        attestation_path.read_bytes()
                    ).hexdigest()
                with self.assertRaisesRegex(
                    ValueError,
                    expected_message,
                ):
                    _build_fixture_report(
                        fixture,
                        output=root / f"{attack}-report.json",
                    )

    def test_shared_policy_is_recursively_immutable_and_digest_bound(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            HUMAN_AUDIT_POLICY["population_gate"][
                "min_conclusive_per_probability_cohort"
            ] = 1
        self.assertEqual(
            hashlib.sha256(
                _canonical(policy_payload()).encode("utf-8")
            ).hexdigest(),
            policy_sha256(),
        )

    def test_stratified_weighting_fpc_variance_and_ci_match_by_hand(
        self,
    ) -> None:
        records: list[dict[str, object]] = []
        for outcome in ("positive", "negative"):
            records.append(
                {
                    "iid": f"a-{outcome}",
                    "cohort": "pseudo_positive",
                    "stratum_id": _sha("stratum-a"),
                    "population": 10,
                    "sample": 2,
                    "weight": 5.0,
                    "outcome": outcome,
                }
            )
        for index, outcome in enumerate(
            ("positive", "positive", "negative")
        ):
            records.append(
                {
                    "iid": f"b-{index}",
                    "cohort": "pseudo_positive",
                    "stratum_id": _sha("stratum-b"),
                    "population": 30,
                    "sample": 3,
                    "weight": 10.0,
                    "outcome": outcome,
                }
            )
        report = _estimate_cohort(records, cohort="pseudo_positive")
        lower = report["all_assigned_identification_bounds"]["lower"]
        expected_estimate = 0.25 * 0.5 + 0.75 * (2.0 / 3.0)
        expected_variance = (
            0.25**2 * (1.0 - 2.0 / 10.0) * 0.5 / 2.0
            + 0.75**2 * (1.0 - 3.0 / 30.0) * (1.0 / 3.0) / 3.0
        )
        self.assertAlmostEqual(lower["estimate"], expected_estimate)
        self.assertAlmostEqual(
            lower["finite_population_variance"],
            expected_variance,
        )
        self.assertAlmostEqual(
            lower["ci95"]["lower"],
            max(
                0.0,
                expected_estimate
                - Z_95 * math.sqrt(expected_variance),
            ),
        )
        self.assertAlmostEqual(
            report["conclusive_available_case"]["estimate"],
            expected_estimate,
        )
        self.assertEqual(
            sorted((row["n"], row["N"]) for row in report["strata"]),
            [(2, 10), (3, 30)],
        )

    def test_uncertain_and_missing_are_explicit_identification_bounds(
        self,
    ) -> None:
        records = [
            {
                "iid": f"iid-{index}",
                "cohort": "pseudo_negative",
                "stratum_id": _sha("single-stratum"),
                "population": 10,
                "sample": 4,
                "weight": 2.5,
                "outcome": outcome,
            }
            for index, outcome in enumerate(
                ("positive", "negative", "uncertain", "missing")
            )
        ]
        report = _estimate_cohort(records, cohort="pseudo_negative")
        bounds = report["all_assigned_identification_bounds"]
        self.assertEqual(report["conclusive_n"], 2)
        self.assertEqual(report["unresolved_n"], 2)
        self.assertAlmostEqual(report["unresolved_fraction"], 0.5)
        self.assertAlmostEqual(
            report["conclusive_available_case"]["estimate"],
            0.5,
        )
        self.assertAlmostEqual(bounds["lower"]["estimate"], 0.25)
        self.assertAlmostEqual(bounds["upper"]["estimate"], 0.75)
        self.assertEqual(
            bounds["point_identification_interval"],
            [0.25, 0.75],
        )

    def test_exact_finite_population_bounds_do_not_degenerate_for_sample(
        self,
    ) -> None:
        positive_records = [
            {
                "iid": f"positive-{index}",
                "cohort": "pseudo_positive",
                "stratum_id": _sha("positive-boundary"),
                "population": 100,
                "sample": 5,
                "weight": 20.0,
                "outcome": "positive",
            }
            for index in range(5)
        ]
        positive = _estimate_cohort(
            positive_records,
            cohort="pseudo_positive",
        )
        positive_bounds = positive["all_assigned_identification_bounds"]
        self.assertEqual(
            positive_bounds["lower"]["ci95"]["lower"],
            1.0,
        )
        self.assertEqual(
            positive_bounds["lower_completion_exact_lcb"][
                "bounded_population_successes"
            ],
            49,
        )
        self.assertAlmostEqual(
            positive_bounds["lower_completion_exact_lcb"]["bound"],
            0.49,
        )
        self.assertLess(
            positive_bounds["lower_completion_exact_lcb"]["bound"],
            1.0,
        )

        negative_records = [
            {
                "iid": f"negative-{index}",
                "cohort": "pseudo_negative",
                "stratum_id": _sha("negative-boundary"),
                "population": 100,
                "sample": 5,
                "weight": 20.0,
                "outcome": "negative",
            }
            for index in range(5)
        ]
        negative = _estimate_cohort(
            negative_records,
            cohort="pseudo_negative",
        )
        negative_bounds = negative["all_assigned_identification_bounds"]
        self.assertEqual(
            negative_bounds["upper"]["ci95"]["upper"],
            0.0,
        )
        self.assertEqual(
            negative_bounds["upper_completion_exact_ucb"][
                "bounded_population_successes"
            ],
            51,
        )
        self.assertAlmostEqual(
            negative_bounds["upper_completion_exact_ucb"]["bound"],
            0.51,
        )
        self.assertGreater(
            negative_bounds["upper_completion_exact_ucb"]["bound"],
            0.0,
        )

    def test_exact_finite_population_bounds_are_exact_for_census(
        self,
    ) -> None:
        positive = _estimate_cohort(
            [
                {
                    "iid": f"positive-census-{index}",
                    "cohort": "pseudo_positive",
                    "stratum_id": _sha("positive-census"),
                    "population": 5,
                    "sample": 5,
                    "weight": 1.0,
                    "outcome": "positive",
                }
                for index in range(5)
            ],
            cohort="pseudo_positive",
        )
        negative = _estimate_cohort(
            [
                {
                    "iid": f"negative-census-{index}",
                    "cohort": "pseudo_negative",
                    "stratum_id": _sha("negative-census"),
                    "population": 5,
                    "sample": 5,
                    "weight": 1.0,
                    "outcome": "negative",
                }
                for index in range(5)
            ],
            cohort="pseudo_negative",
        )
        self.assertEqual(
            positive["all_assigned_identification_bounds"][
                "lower_completion_exact_lcb"
            ]["bound"],
            1.0,
        )
        self.assertEqual(
            negative["all_assigned_identification_bounds"][
                "upper_completion_exact_ucb"
            ]["bound"],
            0.0,
        )

    def test_exact_hypergeometric_small_population_coverage(self) -> None:
        minimum_coverage = 1.0
        for population in range(2, 13):
            for sample in range(1, population + 1):
                for successes in range(population + 1):
                    coverage = 0.0
                    support_lower = max(
                        0,
                        sample - (population - successes),
                    )
                    support_upper = min(sample, successes)
                    for observed in range(
                        support_lower,
                        support_upper + 1,
                    ):
                        lower = _hypergeometric_population_bound(
                            population=population,
                            sample=sample,
                            observed=observed,
                            tail="lower",
                            tail_alpha=0.025,
                        )
                        upper = _hypergeometric_population_bound(
                            population=population,
                            sample=sample,
                            observed=observed,
                            tail="upper",
                            tail_alpha=0.025,
                        )
                        if lower <= successes <= upper:
                            coverage += (
                                comb(successes, observed)
                                * comb(
                                    population - successes,
                                    sample - observed,
                                )
                                / comb(population, sample)
                            )
                    minimum_coverage = min(minimum_coverage, coverage)
                    self.assertGreaterEqual(
                        coverage + 1e-12,
                        0.95,
                        msg=(
                            f"N={population}, n={sample}, "
                            f"K={successes}, coverage={coverage}"
                        ),
                    )
        self.assertLess(minimum_coverage, 1.0)

    def test_multistratum_unresolved_bounds_and_weighted_fraction(
        self,
    ) -> None:
        records: list[dict[str, object]] = []
        for index, outcome in enumerate(("positive", "missing")):
            records.append(
                {
                    "iid": f"small-{index}",
                    "cohort": "pseudo_positive",
                    "stratum_id": _sha("small-unresolved"),
                    "population": 10,
                    "sample": 2,
                    "weight": 5.0,
                    "outcome": outcome,
                }
            )
        for index, outcome in enumerate(
            ("positive", "negative", "uncertain")
        ):
            records.append(
                {
                    "iid": f"large-{index}",
                    "cohort": "pseudo_positive",
                    "stratum_id": _sha("large-unresolved"),
                    "population": 30,
                    "sample": 3,
                    "weight": 10.0,
                    "outcome": outcome,
                }
            )
        report = _estimate_cohort(records, cohort="pseudo_positive")
        bounds = report["all_assigned_identification_bounds"]
        self.assertAlmostEqual(report["raw_unresolved_fraction"], 0.4)
        self.assertAlmostEqual(report["unresolved_fraction"], 0.375)
        self.assertAlmostEqual(
            report["design_weighted_unresolved_population_fraction"],
            0.375,
        )
        self.assertEqual(
            bounds["point_identification_interval"],
            [0.375, 0.75],
        )
        self.assertLessEqual(
            bounds["lower_completion_exact_lcb"]["bound"],
            0.375,
        )
        self.assertGreaterEqual(
            bounds["upper_completion_exact_ucb"]["bound"],
            0.75,
        )
        self.assertEqual(
            bounds["lower_completion_exact_lcb"][
                "per_noncensus_stratum_tail_alpha"
            ],
            0.0125,
        )
        self.assertEqual(
            sorted(
                row["observed_completed_successes"]
                for row in bounds["lower_completion_exact_lcb"]["strata"]
            ),
            [1, 1],
        )
        self.assertEqual(
            sorted(
                row["observed_completed_successes"]
                for row in bounds["upper_completion_exact_ucb"]["strata"]
            ),
            [2, 2],
        )

    def test_weighted_unresolved_can_fail_when_raw_fraction_is_small(
        self,
    ) -> None:
        records: list[dict[str, object]] = [
            {
                "iid": "large-weight-positive",
                "cohort": "pseudo_positive",
                "stratum_id": _sha("large-weight"),
                "population": 30,
                "sample": 2,
                "weight": 15.0,
                "outcome": "positive",
            },
            {
                "iid": "large-weight-missing",
                "cohort": "pseudo_positive",
                "stratum_id": _sha("large-weight"),
                "population": 30,
                "sample": 2,
                "weight": 15.0,
                "outcome": "missing",
            },
        ]
        records.extend(
            {
                "iid": f"small-weight-{index}",
                "cohort": "pseudo_positive",
                "stratum_id": _sha("small-weight"),
                "population": 10,
                "sample": 8,
                "weight": 1.25,
                "outcome": "positive",
            }
            for index in range(8)
        )
        report = _estimate_cohort(records, cohort="pseudo_positive")
        self.assertAlmostEqual(report["raw_unresolved_fraction"], 0.1)
        self.assertAlmostEqual(report["unresolved_fraction"], 0.375)
        self.assertGreater(report["unresolved_fraction"], 0.20)

    def test_uncertain_double_reviews_cannot_create_perfect_gate(self) -> None:
        assigned = [f"double-{index:03d}" for index in range(105)]
        primary: dict[str, dict[str, str]] = {}
        secondary: dict[str, dict[str, str]] = {}
        for index, iid in enumerate(assigned):
            if index < 30:
                primary_verdict = "valid_action"
            elif index < 55:
                primary_verdict = "static"
            else:
                primary_verdict = (
                    "valid_action" if index % 2 else "static"
                )
            secondary_verdict = (
                primary_verdict if index < 55 else "uncertain"
            )
            primary[iid] = {
                "verdict": primary_verdict,
                "reviewer": "reviewer-primary",
            }
            secondary[iid] = {
                "verdict": secondary_verdict,
                "reviewer": "reviewer-secondary",
            }
        double = _double_review_report(
            assigned_iids=assigned,
            primary_reviews=primary,
            secondary_reviews=secondary,
            cohort_by_iid={
                iid: (
                    "pseudo_positive"
                    if index % 2
                    else "pseudo_negative"
                )
                for index, iid in enumerate(assigned)
            },
        )
        self.assertEqual(double["both_rated_n"], 105)
        self.assertEqual(double["both_conclusive_n"], 55)
        self.assertEqual(double["uncertain_pair_n"], 50)
        self.assertEqual(double["unresolved_pair_n"], 50)
        self.assertEqual(double["disagreement_count"], 50)
        self.assertAlmostEqual(
            double["exact_verdict"]["observed_agreement"],
            55 / 105,
        )
        self.assertEqual(
            double["conclusive_only_exact_verdict_diagnostic"]["value"],
            1.0,
        )
        self.assertLess(
            double["exact_verdict_raw_agreement_wilson_95"]["lower"],
            0.80,
        )
        bootstrap = double[
            "exact_verdict_cohen_kappa_bootstrap_95"
        ]
        self.assertEqual(bootstrap["seed"], KAPPA_BOOTSTRAP_SEED)
        self.assertEqual(bootstrap["draws"], KAPPA_BOOTSTRAP_DRAWS)
        repeated_double = _double_review_report(
            assigned_iids=assigned,
            primary_reviews=primary,
            secondary_reviews=secondary,
            cohort_by_iid={
                iid: (
                    "pseudo_positive"
                    if index % 2
                    else "pseudo_negative"
                )
                for index, iid in enumerate(assigned)
            },
        )
        self.assertEqual(
            repeated_double[
                "exact_verdict_cohen_kappa_bootstrap_95"
            ],
            bootstrap,
        )

        positive = _estimate_cohort(
            [
                {
                    "iid": f"gate-positive-{index}",
                    "cohort": "pseudo_positive",
                    "stratum_id": _sha("gate-positive"),
                    "population": 100,
                    "sample": 100,
                    "weight": 1.0,
                    "outcome": "positive",
                }
                for index in range(100)
            ],
            cohort="pseudo_positive",
        )
        negative = _estimate_cohort(
            [
                {
                    "iid": f"gate-negative-{index}",
                    "cohort": "pseudo_negative",
                    "stratum_id": _sha("gate-negative"),
                    "population": 100,
                    "sample": 100,
                    "weight": 1.0,
                    "outcome": "negative",
                }
                for index in range(100)
            ],
            cohort="pseudo_negative",
        )
        gate = _build_gate(
            {
                "pseudo_positive": positive,
                "pseudo_negative": negative,
            },
            double,
            implementation_bundle_sha256="0" * 64,
        )
        self.assertNotEqual(gate["status"], "PASS")
        self.assertFalse(
            gate["prerequisite_checks"][
                "double_unresolved_fraction"
            ]["passed"]
        )
        self.assertFalse(
            gate["evidence_checks"][
                "double_exact_verdict_raw_agreement_wilson_95_lcb"
            ]["passed"]
        )

    def test_formal_gate_can_pass_but_diagnostic_gate_never_can(self) -> None:
        positive = _estimate_cohort(
            [
                {
                    "iid": f"formal-positive-{index}",
                    "cohort": "pseudo_positive",
                    "stratum_id": _sha("formal-positive"),
                    "population": 100,
                    "sample": 100,
                    "weight": 1.0,
                    "outcome": "positive",
                }
                for index in range(100)
            ],
            cohort="pseudo_positive",
        )
        negative = _estimate_cohort(
            [
                {
                    "iid": f"formal-negative-{index}",
                    "cohort": "pseudo_negative",
                    "stratum_id": _sha("formal-negative"),
                    "population": 100,
                    "sample": 100,
                    "weight": 1.0,
                    "outcome": "negative",
                }
                for index in range(100)
            ],
            cohort="pseudo_negative",
        )
        assigned = [f"formal-pair-{index}" for index in range(100)]
        primary = {
            iid: {
                "verdict": (
                    "valid_action" if index < 50 else "static"
                ),
                "reviewer": "reviewer-primary",
            }
            for index, iid in enumerate(assigned)
        }
        secondary = {
            iid: {
                "verdict": primary[iid]["verdict"],
                "reviewer": "reviewer-secondary",
            }
            for iid in assigned
        }
        double = _double_review_report(
            assigned_iids=assigned,
            primary_reviews=primary,
            secondary_reviews=secondary,
            cohort_by_iid={
                iid: (
                    "pseudo_positive"
                    if index < 50
                    else "pseudo_negative"
                )
                for index, iid in enumerate(assigned)
            },
        )
        estimates = {
            "pseudo_positive": positive,
            "pseudo_negative": negative,
        }
        formal = _build_gate(
            estimates,
            double,
            implementation_bundle_sha256="1" * 64,
            formal_evidence_eligible=True,
            independent_reviewer_attestation={
                "external_process_attestation_verified": True,
                "independent_humans_attested": True,
            },
        )
        diagnostic = _build_gate(
            estimates,
            double,
            implementation_bundle_sha256="1" * 64,
            formal_evidence_eligible=False,
            independent_reviewer_attestation={
                "external_process_attestation_verified": True,
                "independent_humans_attested": True,
            },
        )
        self.assertEqual(formal["status"], "PASS")
        self.assertTrue(formal["next_stage_eligible"])
        self.assertEqual(diagnostic["status"], "INSUFFICIENT")
        self.assertFalse(diagnostic["next_stage_eligible"])
        self.assertFalse(
            diagnostic["prerequisite_checks"][
                "formal_media_and_source_evidence_bound"
            ]["passed"]
        )
        for gate in (formal, diagnostic):
            self.assertFalse(
                gate["direct_training_supervision_allowed"]
            )
            self.assertFalse(gate["training_authorized"])

    def test_end_to_end_excludes_purposive_and_keeps_disagreement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(root)
            output = root / "report.json"
            report = _build_fixture_report(
                fixture,
                output=output,
            )
            self.assertGreater(
                report[
                    "purposive_rows_excluded_from_population_inference"
                ]["rows"],
                0,
            )
            self.assertFalse(
                report[
                    "purposive_rows_excluded_from_population_inference"
                ]["population_inference_allowed"]
            )
            self.assertEqual(
                set(report["population_estimates"]),
                {"pseudo_positive", "pseudo_negative"},
            )
            reliability = report["inter_reviewer_reliability"]
            self.assertEqual(reliability["disagreement_count"], 1)
            self.assertLess(
                reliability["exact_verdict"]["observed_agreement"],
                1.0,
            )
            self.assertFalse(
                reliability["automatic_adjudication_performed"]
            )
            self.assertIsNone(reliability["adjudication_result"])
            self.assertEqual(
                report["recommended_gate"]["status"],
                "PASS",
            )
            self.assertTrue(
                report["recommended_gate"][
                    "external_process_attestation_verified"
                ]
            )
            self.assertTrue(
                report["recommended_gate"][
                    "independent_humans_attested"
                ]
            )
            self.assertFalse(
                report["recommended_gate"][
                    "cryptographic_reviewer_identity_verified"
                ]
            )
            self.assertFalse(
                report["recommended_gate"]["training_authorized"]
            )
            self.assertTrue(
                report["inputs"]["sample"]["external_anchor_verified"]
            )
            self.assertTrue(
                report["inputs"]["sample"]["media_bytes_bound"]
            )
            self.assertEqual(
                set(
                    entry["logical_name"]
                    for entry in report["implementation_bundle"]["files"]
                ),
                {
                    "report",
                    "sampler",
                    "policy",
                    "human_review",
                    "qwen_validator",
                    "source_manifest_validator",
                    "verdict_dependency",
                },
            )
            self.assertEqual(
                report["semantics"]["label_scope"],
                "rate_audit_only",
            )
            self.assertFalse(
                report["semantics"][
                    "direct_training_supervision_allowed"
                ]
            )

            original = output.read_bytes()
            resumed = _build_fixture_report(
                fixture,
                output=output,
                resume=True,
            )
            self.assertTrue(resumed["resume_verified"])
            self.assertEqual(output.read_bytes(), original)

    def test_external_anchor_and_live_media_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(root)
            forged = dict(fixture)
            forged["expected_sample_artifact_digest"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "external expected anchor"):
                _build_fixture_report(
                    forged,
                    output=root / "wrong-anchor.json",
                )

            sampled = _read_jsonl(
                fixture["sample"] / SAMPLED_MANIFEST_NAME
            )
            selected_src = (
                fixture["data_root"] / sampled[0]["src_video"]
            )
            selected_src.write_bytes(b"post-review-media-mutation")
            with self.assertRaisesRegex(
                ValueError,
                "media bytes or provenance differ",
            ):
                _build_fixture_report(
                    fixture,
                    output=root / "mutated-media.json",
                )

    def test_live_source_rehash_forgery_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(root)
            positives_path = fixture["source"] / POSITIVES_NAME
            positives = _read_jsonl(positives_path)
            positives[0]["prompt"] = "forged but internally rehashed prompt"
            positives_path.write_text(
                "".join(_canonical(row) + "\n" for row in positives),
                encoding="utf-8",
            )
            _rehash_source_artifacts(fixture["source"])
            with self.assertRaisesRegex(
                ValueError,
                "live source commit differs",
            ):
                _build_fixture_report(
                    fixture,
                    output=root / "forged-live-source.json",
                )

    def test_rehashed_sample_population_forgery_misses_external_anchor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(root)
            summary_path = fixture["sample"] / SUMMARY_NAME
            done_path = fixture["sample"] / DONE_NAME
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            done = json.loads(done_path.read_text(encoding="utf-8"))
            summary["source"]["populations"]["pseudo_positive"] += 1000
            summary_path.write_text(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            done["output_sha256"][SUMMARY_NAME] = hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
            done["artifact_digest"] = hashlib.sha256(
                _canonical(
                    {
                        name: str(digest)
                        for name, digest in sorted(
                            done["output_sha256"].items()
                        )
                    }
                ).encode("utf-8")
            ).hexdigest()
            done_path.write_text(
                json.dumps(
                    done,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "external expected anchor",
            ):
                _build_fixture_report(
                    fixture,
                    output=root / "rehashed-population-forgery.json",
                )

    def test_policy_and_sampler_implementation_tamper_fail(self) -> None:
        for tamper in ("policy", "implementation"):
            temporary_context = tempfile.TemporaryDirectory()
            with self.subTest(tamper=tamper), temporary_context as temporary:
                root = Path(temporary)
                fixture = _review_fixture(root)
                sample_summary_path = fixture["sample"] / SUMMARY_NAME
                sample_done_path = fixture["sample"] / DONE_NAME
                sample_summary = json.loads(
                    sample_summary_path.read_text(encoding="utf-8")
                )
                sample_done = json.loads(
                    sample_done_path.read_text(encoding="utf-8")
                )
                if tamper == "policy":
                    sample_summary["policy"]["population_gate"][
                        "min_conclusive_per_probability_cohort"
                    ] = 1
                    sample_done["policy"] = sample_summary["policy"]
                    message = "policy payload differs"
                else:
                    sample_summary["implementation_sha256"] = "0" * 64
                    sample_done["implementation_sha256"] = "0" * 64
                    message = "current sampler implementation"
                sample_summary_path.write_text(
                    json.dumps(
                        sample_summary,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                sample_done_path.write_text(
                    json.dumps(
                        sample_done,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    _build_fixture_report(
                        fixture,
                        output=root / f"{tamper}-tamper.json",
                    )

    def test_diagnostic_unbound_sample_can_never_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(
                root,
                diagnostic_unbound_media=True,
            )
            report = _build_fixture_report(
                fixture,
                output=root / "diagnostic-report.json",
            )
            self.assertFalse(
                report["inputs"]["sample"]["media_bytes_bound"]
            )
            self.assertEqual(
                report["recommended_gate"]["status"],
                "INSUFFICIENT",
            )
            self.assertFalse(
                report["recommended_gate"]["prerequisite_checks"][
                    "formal_media_and_source_evidence_bound"
                ]["passed"]
            )
            self.assertFalse(
                report["recommended_gate"]["next_stage_eligible"]
            )
            self.assertFalse(
                report["recommended_gate"][
                    "direct_training_supervision_allowed"
                ]
            )

    def test_sample_and_label_provenance_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(root)
            with (
                fixture["sample"] / PRIMARY_REVIEW_NAME
            ).open("ab") as handle:
                handle.write(b" \n")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                _build_fixture_report(
                    fixture,
                    output=root / "sample-tamper-report.json",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _review_fixture(root)
            rows = _read_jsonl(fixture["primary_merged"])
            rows[0]["human_review"]["label_source_sha256"] = "0" * 64
            _write_jsonl(fixture["primary_merged"], rows)
            merge_summary_path = fixture["primary_merged"].with_suffix(
                fixture["primary_merged"].suffix + ".summary.json"
            )
            merge_summary = json.loads(
                merge_summary_path.read_text(encoding="utf-8")
            )
            merge_summary["output_sha256"] = hashlib.sha256(
                fixture["primary_merged"].read_bytes()
            ).hexdigest()
            merge_summary_path.write_text(
                json.dumps(merge_summary, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "human label provenance differs",
            ):
                _build_fixture_report(
                    fixture,
                    output=root / "label-tamper-report.json",
                )


if __name__ == "__main__":
    unittest.main()
