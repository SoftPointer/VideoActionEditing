from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motive.r7_human_audit_sample import (
    DESIGN_VERSION,
    DONE_NAME,
    NEGATIVES_NAME,
    POSITIVES_NAME,
    PRIMARY_REVIEW_NAME,
    REVIEW_NAME,
    REVIEWER_ASSIGNMENTS_NAME,
    SAMPLED_MANIFEST_NAME,
    SAMPLING_LEDGER_NAME,
    SECONDARY_MANIFEST_NAME,
    SECONDARY_REVIEW_NAME,
    SOURCE_DONE_NAME,
    SOURCE_SUMMARY_NAME,
    SUMMARY_NAME,
    build_human_audit_sample,
    build_media_binding,
    validate_media_binding,
)
from motive.r7_human_audit_policy import implementation_bundle_sha256
from motive.r7_build_expansion_manifest import build_expansion_manifest
from methods.motive.tests.test_r7_build_expansion_manifest import (
    _row as _builder_input_row,
)

PRIMARY_REVIEWER_ID = "reviewer-primary"
SECONDARY_REVIEWER_ID = "reviewer-secondary"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _row(
    iid: str,
    *,
    bucket: str,
    family: str,
    verdict: str,
    confidence: str,
    source: str = "original",
    signature: str | None = None,
    negative_role: str | None = None,
) -> dict[str, object]:
    label: dict[str, object] = {
        "schema_version": "motive-r7-expansion-manifest-row-v2",
        "bucket": bucket,
        "primary_family": family,
        "verdict": verdict,
        "result_validated_from": source,
        "split_assigned": False,
        "human_label": False,
        "formal_evidence": False,
    }
    if signature is not None:
        label["action_signature"] = signature
    if negative_role is not None:
        label["negative_role"] = negative_role
        label["negative_type"] = verdict
    return {
        "schema_version": "motive-action-cascade-v1",
        "iid": iid,
        "input_digest": _sha(f"input:{iid}"),
        "prompt": f"perform {family}",
        "src_video": f"videos/{iid}/source.mp4",
        "tgt_video": f"videos/{iid}/edited.mp4",
        "qwen_evidence": {
            "visual": {
                "result": {
                    "verdict": verdict,
                    "confidence": confidence,
                    "action_signature": signature,
                }
            }
        },
        "auto_rule": {
            "tier": "high" if confidence == "high" else "possible",
        },
        "r7_expansion_manifest": label,
    }


def _write_source(root: Path) -> Path:
    def configured(
        row: dict[str, object],
        *,
        family: str,
        tier: str,
    ) -> dict[str, object]:
        row["r7_expansion_selection"]["primary_family"] = family
        row["auto_rule"]["action_families"] = [family]
        row["auto_rule"]["tier"] = tier
        row["auto_rule"]["score"] = 0.75 if tier == "high" else 0.65
        return row

    inputs = [
        configured(
            _builder_input_row(index, verdict="valid_action"),
            family=("wave" if index < 3 else "jump"),
            tier=("possible" if index == 2 else "high"),
        )
        for index in range(6)
    ]
    inputs.extend(
        [
            configured(
                _builder_input_row(
                    6,
                    verdict="static",
                    target_motion="none",
                ),
                family="wave",
                tier="high",
            ),
            configured(
                _builder_input_row(
                    7,
                    verdict="static",
                    target_motion="none",
                ),
                family="wave",
                tier="high",
            ),
            *[
                configured(
                    _builder_input_row(
                        index,
                        verdict="instruction_mismatch",
                        confidence="medium",
                    ),
                    family="jump",
                    tier="possible",
                )
                for index in range(8, 11)
            ],
            *[
                configured(
                    _builder_input_row(
                        index,
                        verdict="static",
                        target_motion="none",
                        result_source="original_sanitized",
                    ),
                    family=("wave" if index == 11 else "jump"),
                    tier="possible",
                )
                for index in range(11, 13)
            ],
        ]
    )
    inputs.extend(
        [
            configured(
                _builder_input_row(
                    13,
                    verdict="valid_action",
                    camera="medium",
                ),
                family="wave",
                tier="high",
            ),
            configured(
                _builder_input_row(
                    14,
                    verdict="static",
                    target_motion="none",
                    result_source="repair_1",
                ),
                family="jump",
                tier="high",
            ),
            configured(
                _builder_input_row(
                    15,
                    verdict="endpoint_only",
                    target_motion="none",
                    confidence="low",
                ),
                family="jump",
                tier="possible",
            ),
            configured(
                _builder_input_row(
                    16,
                    verdict="uncertain",
                    target_motion="none",
                    result_source="fallback_uncertain",
                    confidence="low",
                ),
                family="wave",
                tier="possible",
            ),
            configured(
                _builder_input_row(
                    17,
                    verdict="static",
                    target_motion="none",
                    confidence="low",
                ),
                family="wave",
                tier="possible",
            ),
            configured(
                _builder_input_row(
                    18,
                    verdict="instruction_mismatch",
                    confidence="low",
                ),
                family="jump",
                tier="possible",
            ),
            configured(
                _builder_input_row(
                    19,
                    verdict="static",
                    target_motion="none",
                    confidence="low",
                ),
                family="turn",
                tier="possible",
            ),
        ]
    )
    data_root = root / "data"
    for row in inputs:
        for field in ("src_video", "tgt_video"):
            media_path = data_root / str(row[field])
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(
                f"{row['iid']}:{field}:immutable-media".encode("utf-8")
            )
    fused = root / "fused.jsonl"
    fused.write_text(
        "".join(_canonical(row) + "\n" for row in inputs),
        encoding="utf-8",
    )
    source = root / "manifest"
    build_expansion_manifest(input_path=fused, output_dir=source)
    return source


def _write_formal_source(root: Path) -> Path:
    """Create a policy-sized source commit for formal report fixtures."""

    def configured(
        row: dict[str, object],
        *,
        family: str,
        tier: str,
    ) -> dict[str, object]:
        row["r7_expansion_selection"]["primary_family"] = family
        row["auto_rule"]["action_families"] = [family]
        row["auto_rule"]["tier"] = tier
        row["auto_rule"]["score"] = 0.75 if tier == "high" else 0.65
        return row

    families = ("wave", "jump", "turn")
    inputs = [
        configured(
            _builder_input_row(index, verdict="valid_action"),
            family=families[index % len(families)],
            tier="possible" if index % 7 == 0 else "high",
        )
        for index in range(240)
    ]
    inputs.extend(
        configured(
            _builder_input_row(
                1_000 + offset,
                verdict="instruction_mismatch",
                confidence="medium",
            ),
            family=families[offset % len(families)],
            tier="possible",
        )
        for offset in range(200)
    )
    inputs.extend(
        configured(
            _builder_input_row(
                2_000 + offset,
                verdict="uncertain",
                target_motion="none",
                result_source="fallback_uncertain",
                confidence="low",
            ),
            family=families[offset % len(families)],
            tier="possible",
        )
        for offset in range(80)
    )
    # Keep the source schema's explicit audit-only negative count non-zero;
    # these rows are committed but excluded from the probability population.
    inputs.append(
        configured(
            _builder_input_row(
                3_000,
                verdict="static",
                target_motion="none",
                result_source="original_sanitized",
            ),
            family="wave",
            tier="possible",
        )
    )
    data_root = root / "data"
    for row in inputs:
        for field in ("src_video", "tgt_video"):
            media_path = data_root / str(row[field])
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(
                f"{row['iid']}:{field}:immutable-media".encode("utf-8")
            )
    fused = root / "formal-fused.jsonl"
    fused.write_text(
        "".join(_canonical(row) + "\n" for row in inputs),
        encoding="utf-8",
    )
    source = root / "formal-manifest"
    build_expansion_manifest(input_path=fused, output_dir=source)
    return source


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _rehash_source_artifacts(source: Path) -> None:
    summary_path = source / SOURCE_SUMMARY_NAME
    done_path = source / SOURCE_DONE_NAME
    summary = json.loads(summary_path.read_text())
    done = json.loads(done_path.read_text())
    for name in (POSITIVES_NAME, NEGATIVES_NAME, REVIEW_NAME):
        digest = _file_digest(source / name)
        rows = len((source / name).read_text().splitlines())
        summary["outputs"][name]["sha256"] = digest
        summary["outputs"][name]["rows"] = rows
        done["output_sha256"][name] = digest
        done["output_rows"][name] = rows
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    done["output_sha256"][SOURCE_SUMMARY_NAME] = _file_digest(summary_path)
    bound = {
        name: done["output_sha256"][name]
        for name in (
            NEGATIVES_NAME,
            POSITIVES_NAME,
            REVIEW_NAME,
            SOURCE_SUMMARY_NAME,
        )
    }
    done["artifact_digest"] = _digest_bytes(_canonical(bound).encode())
    done_path.write_text(
        json.dumps(done, indent=2, sort_keys=True) + "\n"
    )


class R7HumanAuditSampleTests(unittest.TestCase):
    def test_formal_custom_design_parameters_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            cases = {
                "custom-seed": {"seed": 31},
                "custom-positive-target": {"positive_sample": 6},
                "custom-negative-target": {
                    "pseudo_negative_sample": 5
                },
                "custom-review-target": {"review_sample": 4},
                "custom-double-review-fraction": {
                    "double_review_fraction": 0.6
                },
            }
            for name, override in cases.items():
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError,
                    "formal sampling parameters must exactly equal",
                ):
                    build_human_audit_sample(
                        source_dir=source,
                        output_dir=root / name,
                        data_root=root / "data",
                        primary_reviewer_id=PRIMARY_REVIEWER_ID,
                        secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                        expected_implementation_bundle_digest=(
                            implementation_bundle_sha256()
                        ),
                        **override,
                    )
                self.assertFalse((root / name).exists())

    def test_formal_external_implementation_bundle_mismatch_rejects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            with self.assertRaisesRegex(
                ValueError,
                "implementation bundle differs from the external",
            ):
                build_human_audit_sample(
                    source_dir=source,
                    output_dir=root / "wrong-bundle",
                    data_root=root / "data",
                    primary_reviewer_id=PRIMARY_REVIEWER_ID,
                    secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                    expected_implementation_bundle_digest="0" * 64,
                )
            self.assertFalse((root / "wrong-bundle").exists())

    def test_formal_mode_requires_root_and_distinct_normalized_reviewers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            with self.assertRaisesRegex(
                ValueError,
                "formal sampling requires data_root",
            ):
                build_human_audit_sample(
                    source_dir=source,
                    output_dir=root / "missing-root",
                    data_root=None,
                    primary_reviewer_id=PRIMARY_REVIEWER_ID,
                    secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                    positive_sample=6,
                    pseudo_negative_sample=5,
                    review_sample=4,
                )
            with self.assertRaisesRegex(
                ValueError,
                "reviewer IDs must differ",
            ):
                build_human_audit_sample(
                    source_dir=source,
                    output_dir=root / "same-reviewer",
                    data_root=root / "data",
                    primary_reviewer_id="Reviewer-Primary",
                    secondary_reviewer_id=" reviewer-primary ",
                    positive_sample=6,
                    pseudo_negative_sample=5,
                    review_sample=4,
                    double_review_fraction=0.2,
                )

    def test_media_escape_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            positives_path = source / POSITIVES_NAME
            positives = _read_jsonl(positives_path)
            positives[0]["src_video"] = "../outside.mp4"
            positives_path.write_text(
                "".join(_canonical(row) + "\n" for row in positives),
                encoding="utf-8",
            )
            _rehash_source_artifacts(source)
            with self.assertRaisesRegex(
                ValueError,
                "parent traversal",
            ):
                build_media_binding(
                    positives[0],
                    data_root=root / "data",
                    diagnostic_unbound_media=False,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            positive = _read_jsonl(source / POSITIVES_NAME)[0]
            src_path = root / "data" / str(positive["src_video"])
            link_target = src_path.with_name("real-source.mp4")
            link_target.write_bytes(src_path.read_bytes())
            src_path.unlink()
            src_path.symlink_to(link_target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                build_media_binding(
                    positive,
                    data_root=root / "data",
                    diagnostic_unbound_media=False,
                )

    def test_post_sampling_media_mutation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            row = _read_jsonl(source / POSITIVES_NAME)[0]
            row["r7_media_binding"] = build_media_binding(
                row,
                data_root=root / "data",
                diagnostic_unbound_media=False,
            )
            media_path = root / "data" / str(row["tgt_video"])
            media_path.write_bytes(b"post-review mutation")
            with self.assertRaisesRegex(
                ValueError,
                "media bytes or provenance differ",
            ):
                validate_media_binding(
                    row,
                    expected_data_root=root / "data",
                    allow_diagnostic_unbound=False,
                )

    def test_diagnostic_unbound_mode_is_explicitly_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            output = root / "diagnostic-sample"
            summary = build_human_audit_sample(
                source_dir=source,
                output_dir=output,
                data_root=None,
                primary_reviewer_id=PRIMARY_REVIEWER_ID,
                secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                diagnostic_unbound_media=True,
                positive_sample=6,
                pseudo_negative_sample=5,
                review_sample=4,
            )
            done = json.loads((output / DONE_NAME).read_text())
            self.assertEqual(summary["media"]["mode"], "diagnostic_unbound")
            self.assertFalse(summary["media"]["media_bytes_bound"])
            self.assertFalse(
                summary["semantics"]["formal_gate_input_eligible"]
            )
            self.assertFalse(done["formal_gate_input_eligible"])
            self.assertFalse(done["training_authorized"])
            for row in _read_jsonl(output / SAMPLED_MANIFEST_NAME):
                self.assertFalse(
                    row["r7_media_binding"]["media_bytes_bound"]
                )
                self.assertFalse(
                    row["r7_human_audit_sampling"]["training_eligible"]
                )

    def test_probability_design_excludes_audit_only_and_resume_is_exact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            output = root / "sample"
            summary = build_human_audit_sample(
                source_dir=source,
                output_dir=output,
                data_root=None,
                primary_reviewer_id=PRIMARY_REVIEWER_ID,
                secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                diagnostic_unbound_media=True,
                positive_sample=6,
                pseudo_negative_sample=5,
                review_sample=5,
                double_review_fraction=0.2,
                seed=7,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    SAMPLED_MANIFEST_NAME,
                    SECONDARY_MANIFEST_NAME,
                    SAMPLING_LEDGER_NAME,
                    PRIMARY_REVIEW_NAME,
                    SECONDARY_REVIEW_NAME,
                    REVIEWER_ASSIGNMENTS_NAME,
                    SUMMARY_NAME,
                    DONE_NAME,
                },
            )
            self.assertEqual(summary["selected"]["total"], 16)
            self.assertEqual(
                summary["source"]["populations"]["audit_only_excluded"],
                2,
            )
            rows = _read_jsonl(output / SAMPLED_MANIFEST_NAME)
            self.assertEqual(len({row["iid"] for row in rows}), 16)
            self.assertFalse(
                {"iid-011", "iid-012"} & {row["iid"] for row in rows}
            )
            self.assertEqual(
                [row["r7_human_audit_sampling"]["sample_order"] for row in rows],
                list(range(1, 17)),
            )
            for row in rows:
                sampling = row["r7_human_audit_sampling"]
                self.assertFalse(sampling["split_assigned"])
                self.assertFalse(sampling["training_eligible"])
                if sampling["cohort"] == "priority_review":
                    self.assertIsNone(sampling["design_weight"])
                else:
                    self.assertGreater(sampling["design_weight"], 0.0)
                    self.assertGreater(
                        sampling["selection_probability"], 0.0
                    )
            for cohort, population in (
                ("pseudo_positive", 6),
                ("pseudo_negative", 5),
            ):
                estimated_population = sum(
                    row["r7_human_audit_sampling"]["design_weight"]
                    for row in rows
                    if row["r7_human_audit_sampling"]["cohort"] == cohort
                )
                self.assertAlmostEqual(estimated_population, population)

            primary = _read_jsonl(output / PRIMARY_REVIEW_NAME)
            secondary = _read_jsonl(output / SECONDARY_REVIEW_NAME)
            secondary_manifest = _read_jsonl(
                output / SECONDARY_MANIFEST_NAME
            )
            assignments = _read_jsonl(
                output / REVIEWER_ASSIGNMENTS_NAME
            )
            self.assertEqual(len(primary), 16)
            self.assertEqual(len(secondary), 3)
            self.assertEqual(
                {row["iid"] for row in secondary},
                {row["iid"] for row in secondary_manifest},
            )
            forbidden = {
                "qwen_evidence",
                "auto_rule",
                "r7_expansion_manifest",
                "r7_human_audit_sampling",
                "automation_hints",
                "cohort",
                "bucket",
            }
            self.assertTrue(
                all(not (forbidden & set(row)) for row in primary)
            )
            self.assertEqual(
                len({row["review_instance_id"] for row in assignments}),
                19,
            )
            self.assertEqual(
                sum(
                    row["annotator_slot"] == "secondary"
                    for row in assignments
                ),
                3,
            )

            before = {
                path.name: path.read_bytes() for path in output.iterdir()
            }
            resumed = build_human_audit_sample(
                source_dir=source,
                output_dir=output,
                data_root=None,
                primary_reviewer_id=PRIMARY_REVIEWER_ID,
                secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                diagnostic_unbound_media=True,
                positive_sample=6,
                pseudo_negative_sample=5,
                review_sample=5,
                double_review_fraction=0.2,
                seed=7,
                resume=True,
            )
            self.assertTrue(resumed["resume_verified"])
            self.assertEqual(
                before,
                {
                    path.name: path.read_bytes()
                    for path in output.iterdir()
                },
            )

    def test_review_is_explicitly_purposive_and_priority_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            output = root / "sample"
            summary = build_human_audit_sample(
                source_dir=source,
                output_dir=output,
                data_root=None,
                primary_reviewer_id=PRIMARY_REVIEWER_ID,
                secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                diagnostic_unbound_media=True,
                positive_sample=6,
                pseudo_negative_sample=5,
                review_sample=4,
                seed=9,
            )
            design = summary["designs"]["priority_review"]
            self.assertFalse(design["population_inference_allowed"])
            counts = {
                item["priority_category"]: item["sample"]
                for item in design["categories"]
            }
            self.assertEqual(counts["positive_quality_failure"], 1)
            self.assertEqual(counts["schema_repair"], 1)
            self.assertEqual(
                counts["original_endpoint_or_uncertain"], 1
            )
            self.assertEqual(counts["fallback_uncertain"], 1)
            self.assertEqual(
                counts["original_low_confidence_negative"], 0
            )

    def test_too_small_probability_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            with self.assertRaisesRegex(ValueError, "smaller than"):
                build_human_audit_sample(
                    source_dir=source,
                    output_dir=root / "sample",
                    data_root=None,
                    primary_reviewer_id=PRIMARY_REVIEWER_ID,
                    secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                    diagnostic_unbound_media=True,
                    positive_sample=5,
                    pseudo_negative_sample=5,
                    review_sample=4,
                )

    def test_source_tamper_and_extra_artifact_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            with (source / POSITIVES_NAME).open("ab") as handle:
                handle.write(b"{}\n")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                build_human_audit_sample(
                    source_dir=source,
                    output_dir=root / "sample-a",
                    data_root=None,
                    primary_reviewer_id=PRIMARY_REVIEWER_ID,
                    secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                    diagnostic_unbound_media=True,
                    positive_sample=6,
                    pseudo_negative_sample=5,
                    review_sample=4,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            positives = _read_jsonl(source / POSITIVES_NAME)
            positives[0]["r7_expansion_manifest"]["verdict"] = "static"
            (source / POSITIVES_NAME).write_text(
                "".join(_canonical(row) + "\n" for row in positives)
            )
            _rehash_source_artifacts(source)
            with self.assertRaisesRegex(
                ValueError,
                "recomputed verdict differs",
            ):
                build_human_audit_sample(
                    source_dir=source,
                    output_dir=root / "sample-semantic-tamper",
                    data_root=None,
                    primary_reviewer_id=PRIMARY_REVIEWER_ID,
                    secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                    diagnostic_unbound_media=True,
                    positive_sample=6,
                    pseudo_negative_sample=5,
                    review_sample=4,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            (source / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact set differs"):
                build_human_audit_sample(
                    source_dir=source,
                    output_dir=root / "sample-b",
                    data_root=None,
                    primary_reviewer_id=PRIMARY_REVIEWER_ID,
                    secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                    diagnostic_unbound_media=True,
                    positive_sample=6,
                    pseudo_negative_sample=5,
                    review_sample=4,
                )

    def test_output_contract_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root)
            output = root / "sample"
            build_human_audit_sample(
                source_dir=source,
                output_dir=output,
                data_root=None,
                primary_reviewer_id=PRIMARY_REVIEWER_ID,
                secondary_reviewer_id=SECONDARY_REVIEWER_ID,
                diagnostic_unbound_media=True,
                positive_sample=6,
                pseudo_negative_sample=5,
                review_sample=4,
            )
            done = json.loads((output / DONE_NAME).read_text())
            self.assertEqual(done["design_version"], DESIGN_VERSION)
            self.assertEqual(
                done["output_sha256"][SAMPLED_MANIFEST_NAME],
                _file_digest(output / SAMPLED_MANIFEST_NAME),
            )
            self.assertEqual(
                done["output_sha256"][SAMPLING_LEDGER_NAME],
                _file_digest(output / SAMPLING_LEDGER_NAME),
            )
            for name in (
                SECONDARY_MANIFEST_NAME,
                PRIMARY_REVIEW_NAME,
                SECONDARY_REVIEW_NAME,
                REVIEWER_ASSIGNMENTS_NAME,
            ):
                self.assertEqual(
                    done["output_sha256"][name],
                    _file_digest(output / name),
                )
            self.assertEqual(
                done["output_sha256"][SUMMARY_NAME],
                _file_digest(output / SUMMARY_NAME),
            )
            self.assertFalse(done["training_authorized"])

if __name__ == "__main__":
    unittest.main()
