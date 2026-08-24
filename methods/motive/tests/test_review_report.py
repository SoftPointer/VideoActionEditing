from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from motive.review_report import (
    BALANCED_SAMPLE_SCHEME,
    BALANCED_SAMPLE_VERSION,
    HUMAN_REVIEW_SCHEMA,
    REPORT_SCHEMA_VERSION,
    build_report,
    main,
    wilson_interval,
)


LABEL_SHA256 = "a" * 64


def _row(
    iid: str,
    *,
    decision: str,
    human_verdict: str,
    qwen_verdict: str = "valid_action",
    inclusion_probability: float | None = None,
) -> dict:
    row = {
        "iid": iid,
        "final_triage": {"decision": decision},
        "qwen_evidence": {
            "visual": {
                "status": "ok",
                "result": {"verdict": qwen_verdict},
            }
        },
        "human_review": {
            "schema_version": HUMAN_REVIEW_SCHEMA,
            "verdict": human_verdict,
            "reviewer": "reviewer-1",
            "label_source_sha256": LABEL_SHA256,
            "action_signature": "",
            "notes": "",
            "event_start_frame": None,
            "event_end_frame": None,
        },
    }
    if inclusion_probability is not None:
        fraction = Fraction(inclusion_probability).limit_denominator(10_000)
        row["sampling_provenance"] = {
            "scheme": BALANCED_SAMPLE_SCHEME,
            "version": BALANCED_SAMPLE_VERSION,
            "seed": 17,
            "stratum": f"test|{iid}",
            "stratum_population": fraction.denominator,
            "stratum_selected": fraction.numerator,
            "inclusion_probability": inclusion_probability,
            "inverse_probability_weight": 1.0 / inclusion_probability,
            "within_stratum_rank": 1,
        }
    return row


def _provenance(
    *,
    stratum: str = "walk|possible",
    population: int = 10,
    selected: int = 3,
    rank: int = 1,
    seed: int = 17,
) -> dict:
    probability = selected / population
    return {
        "scheme": BALANCED_SAMPLE_SCHEME,
        "version": BALANCED_SAMPLE_VERSION,
        "seed": seed,
        "stratum": stratum,
        "stratum_population": population,
        "stratum_selected": selected,
        "inclusion_probability": probability,
        "inverse_probability_weight": 1.0 / probability,
        "within_stratum_rank": rank,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class ReviewReportTests(unittest.TestCase):
    def test_instruction_mismatch_is_a_conclusive_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.jsonl"
            _write_jsonl(
                path,
                [
                    _row(
                        "mismatch",
                        decision="auto_reject",
                        human_verdict="instruction_mismatch",
                        qwen_verdict="instruction_mismatch",
                    )
                ],
            )
            report = build_report(path)
            self.assertEqual(
                report["human_outcomes"],
                {"positive": 0, "negative": 1, "uncertain": 0},
            )
            self.assertEqual(
                report["human_verdicts"],
                {"instruction_mismatch": 1},
            )

    def test_cross_tab_uncertain_and_ipw_are_kept_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.jsonl"
            rows = [
                _row(
                    "keep-positive",
                    decision="auto_keep",
                    human_verdict="valid_action",
                    qwen_verdict="valid_action",
                    inclusion_probability=0.5,
                ),
                _row(
                    "keep-negative",
                    decision="auto_keep",
                    human_verdict="static",
                    qwen_verdict="static",
                    inclusion_probability=0.25,
                ),
                _row(
                    "keep-uncertain",
                    decision="auto_keep",
                    human_verdict="uncertain",
                    qwen_verdict="uncertain",
                    inclusion_probability=0.5,
                ),
                _row(
                    "reject-miss",
                    decision="auto_reject",
                    human_verdict="valid_suppression",
                    qwen_verdict="static",
                    inclusion_probability=0.25,
                ),
                _row(
                    "reject-correct",
                    decision="auto_reject",
                    human_verdict="endpoint_only",
                    qwen_verdict="endpoint_only",
                    inclusion_probability=0.5,
                ),
            ]
            _write_jsonl(path, rows)

            report = build_report(path)

            self.assertEqual(
                report["schema_version"], REPORT_SCHEMA_VERSION
            )
            self.assertEqual(
                report["input_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["human_outcomes"],
                {"positive": 2, "negative": 2, "uncertain": 1},
            )
            keep = report["metrics"]["auto_keep_precision"]
            self.assertEqual(keep["numerator_human_positive"], 1)
            self.assertEqual(keep["denominator_conclusive"], 2)
            self.assertEqual(keep["human_uncertain_excluded"], 1)
            self.assertEqual(keep["uncertain_fraction"], 1.0 / 3.0)
            self.assertEqual(keep["estimate"], 0.5)
            self.assertEqual(keep["assessment"], "insufficient")
            reject = report["metrics"][
                "auto_reject_positive_contamination"
            ]
            self.assertEqual(reject["numerator_human_positive"], 1)
            self.assertEqual(reject["denominator_conclusive"], 2)
            self.assertEqual(reject["estimate"], 0.5)
            self.assertEqual(reject["assessment"], "insufficient")

            cells = {
                (
                    cell["final_triage_decision"],
                    cell["qwen_visual_verdict"],
                ): cell
                for cell in report["cross_tabulation"]["cells"]
            }
            self.assertEqual(
                cells[("auto_keep", "uncertain")]["human_outcomes"],
                {"positive": 0, "negative": 0, "uncertain": 1},
            )
            self.assertEqual(
                cells[("auto_reject", "static")]["human_verdicts"],
                {"valid_suppression": 1},
            )

            weighted = report["inverse_probability_weighting"]
            self.assertEqual(weighted["availability"], "complete")
            self.assertAlmostEqual(
                weighted["metrics"]["auto_keep_precision"][
                    "ratio_estimate"
                ],
                1.0 / 3.0,
            )
            self.assertAlmostEqual(
                weighted["metrics"][
                    "auto_reject_positive_contamination"
                ][
                    "ratio_estimate"
                ],
                2.0 / 3.0,
            )
            self.assertIsNone(weighted["confidence_intervals"])
            self.assertIn("unweighted", weighted["ci_note"])

    def test_gate_pass_and_insufficient_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.jsonl"
            rows = [
                *[
                    _row(
                        f"keep-{index}",
                        decision="auto_keep",
                        human_verdict="valid_action",
                    )
                    for index in range(100)
                ],
                *[
                    _row(
                        f"reject-{index}",
                        decision="auto_reject",
                        human_verdict="static",
                    )
                    for index in range(100)
                ],
            ]
            _write_jsonl(path, rows)
            report = build_report(path)
            self.assertEqual(
                report["metrics"]["auto_keep_precision"]["assessment"],
                "pass",
            )
            self.assertEqual(
                report["metrics"][
                    "auto_reject_positive_contamination"
                ]["assessment"],
                "pass",
            )
            self.assertFalse(report["scope"]["covers_rule_stage_rejects"])
            self.assertEqual(
                report["metrics"][
                    "rule_reject_positive_contamination"
                ]["assessment"],
                "insufficient",
            )

            only_uncertain = Path(directory) / "uncertain.jsonl"
            _write_jsonl(
                only_uncertain,
                [
                    _row(
                        "u",
                        decision="auto_keep",
                        human_verdict="uncertain",
                    )
                ],
            )
            uncertain_report = build_report(only_uncertain)
            keep = uncertain_report["metrics"]["auto_keep_precision"]
            self.assertEqual(keep["assessment"], "insufficient")
            self.assertEqual(keep["denominator_conclusive"], 0)
            self.assertIsNone(keep["estimate"])
            self.assertIsNone(keep["wilson_95_ci"])
            self.assertEqual(
                uncertain_report["metrics"][
                    "auto_reject_positive_contamination"
                ][
                    "assessment"
                ],
                "insufficient",
            )

            too_many_uncertain = Path(directory) / "too-many-uncertain.jsonl"
            _write_jsonl(
                too_many_uncertain,
                [
                    *[
                        _row(
                            f"c-{index}",
                            decision="auto_keep",
                            human_verdict="valid_action",
                        )
                        for index in range(100)
                    ],
                    *[
                        _row(
                            f"u-{index}",
                            decision="auto_keep",
                            human_verdict="uncertain",
                        )
                        for index in range(26)
                    ],
                ],
            )
            excessive = build_report(too_many_uncertain)["metrics"][
                "auto_keep_precision"
            ]
            self.assertEqual(excessive["denominator_conclusive"], 100)
            self.assertGreater(excessive["uncertain_fraction"], 0.20)
            self.assertEqual(excessive["assessment"], "insufficient")

    def test_rule_reject_audit_has_separate_contamination_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rule-reject-reviewed.jsonl"
            rows = []
            for index in range(100):
                row = _row(
                    f"rule-reject-{index}",
                    decision="unused",
                    human_verdict=(
                        "valid_action" if index < 2 else "endpoint_only"
                    ),
                )
                row.pop("final_triage")
                row["auto_rule"] = {"tier": "reject"}
                rows.append(row)
            _write_jsonl(path, rows)
            report = build_report(path)
            metric = report["metrics"][
                "rule_reject_positive_contamination"
            ]
            self.assertEqual(metric["denominator_conclusive"], 100)
            self.assertEqual(metric["numerator_human_positive"], 2)
            self.assertEqual(metric["estimate"], 0.02)
            self.assertEqual(metric["assessment"], "pass")
            self.assertTrue(report["scope"]["covers_rule_stage_rejects"])
            self.assertFalse(
                report["scope"][
                    "covers_downstream_feature_qwen_fusion"
                ]
            )
            self.assertIsNone(report["scope"]["false_negative_rate"])

    def test_partial_propensities_do_not_emit_population_ipw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.jsonl"
            _write_jsonl(
                path,
                [
                    _row(
                        "with-p",
                        decision="auto_keep",
                        human_verdict="valid_action",
                        inclusion_probability=0.5,
                    ),
                    _row(
                        "without-p",
                        decision="auto_keep",
                        human_verdict="static",
                    ),
                ],
            )
            weighting = build_report(path)[
                "inverse_probability_weighting"
            ]
            self.assertEqual(weighting["availability"], "incomplete")
            self.assertEqual(weighting["rows_with_inclusion_probability"], 1)
            self.assertNotIn("metrics", weighting)

    def test_valid_balanced_sample_subset_allows_population_ipw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.jsonl"
            positive = _row(
                "positive",
                decision="auto_keep",
                human_verdict="valid_action",
            )
            positive["sampling_provenance"] = _provenance(rank=1)
            negative = _row(
                "negative",
                decision="auto_keep",
                human_verdict="static",
            )
            # Only two of the three originally selected reviews are present.
            # Non-contiguous ranks are legal for a partially completed audit.
            negative["sampling_provenance"] = _provenance(rank=3)
            _write_jsonl(path, [positive, negative])

            weighting = build_report(path)[
                "inverse_probability_weighting"
            ]
            self.assertEqual(weighting["availability"], "complete")
            self.assertEqual(
                weighting["rows_with_valid_supported_provenance"], 2
            )
            self.assertAlmostEqual(
                weighting["ht_estimated_total"],
                2.0 * (10.0 / 3.0),
            )
            self.assertEqual(
                weighting["sampling_design"]["strata"]["walk|possible"],
                {"population": 10, "selected": 3, "reviewed": 2},
            )

    def test_unknown_missing_and_mixed_designs_never_emit_ipw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            unknown = _row(
                "unknown",
                decision="auto_keep",
                human_verdict="valid_action",
            )
            unknown["sampling_provenance"] = _provenance()
            unknown["sampling_provenance"]["scheme"] = "some-other-sampler"

            missing_scheme = _row(
                "missing-scheme",
                decision="auto_keep",
                human_verdict="static",
            )
            missing_scheme["sampling_provenance"] = _provenance()
            missing_scheme["sampling_provenance"].pop("scheme")

            supported = _row(
                "supported",
                decision="auto_keep",
                human_verdict="valid_action",
            )
            supported["sampling_provenance"] = _provenance(
                stratum="run|possible",
                rank=1,
            )

            cases = [
                ("unknown", [unknown], "unsupported"),
                ("missing-scheme", [missing_scheme], "incomplete"),
                ("mixed-scheme", [supported, unknown], "unsupported"),
                (
                    "partial-provenance",
                    [
                        supported,
                        _row(
                            "not-sampled",
                            decision="auto_keep",
                            human_verdict="static",
                        ),
                    ],
                    "incomplete",
                ),
            ]
            for name, rows, expected_availability in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.jsonl"
                    _write_jsonl(path, rows)
                    report = build_report(path)
                    weighting = report["inverse_probability_weighting"]
                    self.assertEqual(
                        weighting["availability"],
                        expected_availability,
                    )
                    self.assertNotIn("ht_estimated_total", weighting)
                    self.assertNotIn("metrics", weighting)
                    # The unweighted audited-sample report remains available.
                    self.assertIn("auto_keep_precision", report["metrics"])

    def test_claimed_balanced_sample_v1_is_strictly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                (
                    "missing-field",
                    lambda value: value.pop("seed"),
                    "supported sampling_provenance is missing",
                ),
                (
                    "bool-seed",
                    lambda value: value.__setitem__("seed", True),
                    "seed must be an integer",
                ),
                (
                    "blank-stratum",
                    lambda value: value.__setitem__("stratum", " "),
                    "stratum must be a canonical",
                ),
                (
                    "zero-population",
                    lambda value: value.__setitem__(
                        "stratum_population", 0
                    ),
                    "stratum_population must be a positive integer",
                ),
                (
                    "selected-over-population",
                    lambda value: value.__setitem__(
                        "stratum_selected", 11
                    ),
                    "stratum_selected must not exceed",
                ),
                (
                    "rank-over-selected",
                    lambda value: value.__setitem__(
                        "within_stratum_rank", 4
                    ),
                    "within_stratum_rank must not exceed",
                ),
                (
                    "probability-mismatch",
                    lambda value: value.__setitem__(
                        "inclusion_probability", 0.2
                    ),
                    "inclusion_probability must equal",
                ),
                (
                    "weight-mismatch",
                    lambda value: value.__setitem__(
                        "inverse_probability_weight", 2.0
                    ),
                    "inverse_probability_weight must equal",
                ),
            ]
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    row = _row(
                        name,
                        decision="auto_keep",
                        human_verdict="valid_action",
                    )
                    provenance = _provenance()
                    mutate(provenance)
                    row["sampling_provenance"] = provenance
                    path = root / f"{name}.jsonl"
                    _write_jsonl(path, [row])
                    with self.assertRaisesRegex(ValueError, message):
                        build_report(path)

    def test_stratum_configuration_and_observed_ranks_are_consistent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            first = _row(
                "first",
                decision="auto_keep",
                human_verdict="valid_action",
            )
            first["sampling_provenance"] = _provenance(rank=1)
            inconsistent = _row(
                "inconsistent",
                decision="auto_keep",
                human_verdict="static",
            )
            inconsistent["sampling_provenance"] = _provenance(
                population=20,
                selected=6,
                rank=2,
            )
            inconsistent_path = root / "inconsistent.jsonl"
            _write_jsonl(inconsistent_path, [first, inconsistent])
            with self.assertRaisesRegex(
                ValueError,
                "inconsistent sampling_provenance configuration",
            ):
                build_report(inconsistent_path)

            duplicate = _row(
                "duplicate",
                decision="auto_keep",
                human_verdict="static",
            )
            duplicate["sampling_provenance"] = _provenance(rank=1)
            duplicate_path = root / "duplicate-rank.jsonl"
            _write_jsonl(duplicate_path, [first, duplicate])
            with self.assertRaisesRegex(
                ValueError,
                "duplicates sampling_provenance within_stratum_rank",
            ):
                build_report(duplicate_path)

            other_seed = _row(
                "other-seed",
                decision="auto_keep",
                human_verdict="static",
            )
            other_seed["sampling_provenance"] = _provenance(
                stratum="run|possible",
                seed=23,
            )
            seed_path = root / "mixed-seed.jsonl"
            _write_jsonl(seed_path, [first, other_seed])
            with self.assertRaisesRegex(ValueError, "mixes seeds"):
                build_report(seed_path)

    def test_strict_human_review_and_probability_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                (
                    "bad-schema",
                    lambda row: row["human_review"].__setitem__(
                        "schema_version", "old"
                    ),
                    "unsupported human_review schema",
                ),
                (
                    "blank-reviewer",
                    lambda row: row["human_review"].__setitem__(
                        "reviewer", " "
                    ),
                    "reviewer must be a non-empty string",
                ),
                (
                    "bad-verdict",
                    lambda row: row["human_review"].__setitem__(
                        "verdict", "maybe"
                    ),
                    "invalid human_review verdict",
                ),
                (
                    "bad-sha",
                    lambda row: row["human_review"].__setitem__(
                        "label_source_sha256", "not-a-sha"
                    ),
                    "label_source_sha256 is invalid",
                ),
                (
                    "bad-probability",
                    lambda row: row.__setitem__(
                        "sampling_provenance",
                        {"inclusion_probability": 0.0},
                    ),
                    "must satisfy 0 < p <= 1",
                ),
            ]
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    row = _row(
                        name,
                        decision="auto_keep",
                        human_verdict="valid_action",
                    )
                    mutate(row)
                    path = root / f"{name}.jsonl"
                    _write_jsonl(path, [row])
                    with self.assertRaisesRegex(ValueError, message):
                        build_report(path)

    def test_cli_writes_atomically_and_refuses_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "reviewed.jsonl"
            output_path = root / "report.json"
            _write_jsonl(
                input_path,
                [
                    _row(
                        "one",
                        decision="auto_keep",
                        human_verdict="valid_action",
                    )
                ],
            )
            self.assertEqual(
                main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["rows"], 1)
            self.assertFalse(
                list(root.glob(f".{output_path.name}.*.tmp"))
            )
            with self.assertRaises(FileExistsError):
                main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )

    def test_wilson_interval_input_contract(self) -> None:
        self.assertIsNone(wilson_interval(0, 0))
        lower, upper = wilson_interval(50, 100) or (None, None)
        self.assertAlmostEqual(lower, 0.4038315303659956)
        self.assertAlmostEqual(upper, 0.5961684696340044)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
