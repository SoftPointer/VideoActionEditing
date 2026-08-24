from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motive.qwen_filter import _object_digest
from motive.r7_build_expansion_manifest import (
    DONE_NAME,
    DONE_SCHEMA,
    LEGACY_SPLIT_PROVENANCE,
    LEGACY_SPLIT_QUARANTINE_POLICY_VERSION,
    NEGATIVES_NAME,
    POSITIVES_NAME,
    REVIEW_NAME,
    SUMMARY_NAME,
    SUMMARY_SCHEMA,
    build_expansion_manifest,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _observation(
    *,
    target_motion: str = "clear",
    camera: str = "low",
    background: str = "low",
    artifact: str = "low",
    preservation: str = "acceptable",
    uncertainty: bool = False,
) -> dict[str, object]:
    target_action = (
        "person walks from left to right"
        if target_motion in {"clear", "weak"}
        else "no visible action"
    )
    return {
        "schema_version": "qwen-motion-observation-v2",
        "source_action": "no visible action",
        "target_action": target_action,
        "source_actor_motion": "none",
        "target_actor_motion": target_motion,
        "camera_dominance": camera,
        "background_dominance": background,
        "artifact_level": artifact,
        "preservation_quality": preservation,
        "temporal_evidence": [
            "Ordered target frames show the actor state over time."
        ],
        "uncertainty_codes": ["occlusion"] if uncertainty else [],
    }


def _result(
    verdict: str,
    *,
    confidence: str = "high",
    uncertainty: bool = False,
) -> dict[str, object]:
    if verdict == "valid_action":
        edit_effect = "started"
        signature = "walk right"
    elif verdict == "instruction_mismatch":
        edit_effect = "changed_action"
        signature = "unknown"
    elif verdict == "uncertain":
        edit_effect = "unclear"
        signature = "unknown"
    else:
        edit_effect = "none"
        signature = "unknown"
    return {
        "schema_version": "qwen-motion-judge-v4",
        "verdict": verdict,
        "edit_effect": edit_effect,
        "action_signature": signature,
        "reason_codes": [f"judge_{verdict}"],
        "uncertainty_codes": ["ambiguous"] if uncertainty else [],
        "confidence": confidence,
    }


def _row(
    index: int,
    *,
    verdict: str,
    target_motion: str = "clear",
    result_source: str = "original",
    observation_source: str = "original",
    camera: str = "low",
    confidence: str = "high",
    uncertainty: bool = False,
) -> dict[str, object]:
    iid = f"iid-{index:03d}"
    observation = _observation(
        target_motion=target_motion,
        camera=camera,
        uncertainty=uncertainty,
    )
    result = _result(
        verdict,
        confidence=confidence,
        uncertainty=uncertainty,
    )
    observation_digest = _object_digest(observation)
    result_digest = _object_digest(result)
    observation_repairs: list[dict[str, object]] = []
    alignment_repairs: list[dict[str, object]] = []
    if observation_source == "repair_1":
        observation_repairs = [
            {
                "attempt": 1,
                "status": "ok",
                "repair_generation_called": True,
            }
        ]
    if result_source == "repair_1":
        alignment_repairs = [
            {
                "attempt": 1,
                "status": "ok",
                "repair_generation_called": True,
                "authoritative_context_digest": observation_digest,
            }
        ]
    elif result_source == "original_sanitized":
        alignment_repairs = [
            {
                "attempt": 0,
                "status": "ok",
                "repair_generation_called": False,
                "authoritative_context_digest": observation_digest,
                "repair_sanitizations": [
                    {
                        "action": "downgrade_instruction_mismatch_to_static",
                    }
                ],
            }
        ]
    input_digest = _sha256(f"input-{iid}".encode())
    visual: dict[str, object] = {
        "iid": iid,
        "input_digest": input_digest,
        "mode": "visual",
        "status": "ok",
        "observation": observation,
        "observation_digest": observation_digest,
        "observation_repairs": observation_repairs,
        "observation_validated_from": observation_source,
        "alignment_repairs": alignment_repairs,
        "result": result,
        "result_digest": result_digest,
        "result_validated_from": result_source,
        "visual_input_digest": _sha256(f"visual-{iid}".encode()),
        "execution_shard_index": index % 8,
        "execution_shard_count": 8,
        "execution_manifest": f"/audit/shard-{index % 8:03d}.jsonl",
        "execution_manifest_sha256": _sha256(
            f"manifest-{index % 8}".encode()
        ),
        "run_config_digest": "a" * 64,
        "config_digest": _sha256(f"config-{index % 8}".encode()),
        "implementation_digest": "b" * 64,
        "model_revision": "c" * 40,
        "transformers_version": "4.51.3",
    }
    if observation_source == "fallback_uncertain":
        visual["observation_fallback"] = {
            "fallback_digest": observation_digest,
        }
    if result_source == "fallback_uncertain":
        visual["result_fallback"] = {
            "fallback_digest": result_digest,
            "authoritative_context_digest": observation_digest,
        }
    return {
        "iid": iid,
        "input_digest": input_digest,
        "prompt": "make the person move",
        "src_video": f"{index}/source.mp4",
        "tgt_video": f"{index}/edited.mp4",
        "auto_rule": {"action_families": ["walk"]},
        "r7_expansion_selection": {
            "schema_version": "motive-r7-expansion-selection-v1",
            "primary_family": "walk",
            "split_assigned": False,
        },
        "qwen_evidence": {"visual": visual},
    }


def _with_legacy_split(
    row: dict[str, object],
    split: str,
) -> dict[str, object]:
    row["split"] = split
    row["split_provenance"] = dict(LEGACY_SPLIT_PROVENANCE)
    return row


class R7BuildExpansionManifestTests(unittest.TestCase):
    def test_classifies_strict_positive_negative_and_review_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            rows = [
                _row(0, verdict="valid_action"),
                _row(1, verdict="instruction_mismatch"),
                _row(
                    2,
                    verdict="static",
                    target_motion="none",
                    result_source="original_sanitized",
                ),
                _row(
                    3,
                    verdict="valid_action",
                    result_source="repair_1",
                ),
                _row(
                    4,
                    verdict="valid_action",
                    camera="medium",
                ),
                _row(
                    5,
                    verdict="static",
                    target_motion="none",
                    confidence="low",
                ),
                _row(
                    6,
                    verdict="uncertain",
                    result_source="fallback_uncertain",
                ),
            ]
            _write_jsonl(input_path, rows)
            output = root / "manifest"
            returned = build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
            )

            positives = _read_jsonl(output / POSITIVES_NAME)
            negatives = _read_jsonl(output / NEGATIVES_NAME)
            review = _read_jsonl(output / REVIEW_NAME)
            self.assertEqual([row["iid"] for row in positives], ["iid-000"])
            self.assertEqual(
                [row["iid"] for row in negatives],
                ["iid-001", "iid-002"],
            )
            self.assertEqual(
                [row["iid"] for row in review],
                ["iid-003", "iid-004", "iid-005", "iid-006"],
            )
            positive_label = positives[0]["r7_expansion_manifest"]
            self.assertFalse(positive_label["human_label"])
            self.assertFalse(positive_label["formal_evidence"])
            self.assertFalse(positive_label["split_assigned"])
            self.assertEqual(
                negatives[0]["r7_expansion_manifest"]["negative_type"],
                "instruction_mismatch",
            )
            sanitized = negatives[1]["r7_expansion_manifest"]
            self.assertEqual(sanitized["negative_type"], "static")
            self.assertEqual(sanitized["negative_role"], "audit_only")
            self.assertEqual(
                review[0]["r7_expansion_manifest"][
                    "classification_reason"
                ],
                "result_validation_source:repair_1",
            )
            self.assertEqual(
                returned["bucket_counts"],
                {"negative": 2, "positive": 1, "review": 4},
            )

    def test_binds_input_implementation_and_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            _write_jsonl(
                input_path,
                [
                    _row(0, verdict="valid_action"),
                    _row(1, verdict="static", target_motion="none"),
                ],
            )
            output = root / "manifest"
            build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
            )
            summary = json.loads(
                (output / SUMMARY_NAME).read_text(encoding="utf-8")
            )
            done = json.loads(
                (output / DONE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema_version"], SUMMARY_SCHEMA)
            self.assertEqual(done["schema_version"], DONE_SCHEMA)
            self.assertEqual(
                summary["input"]["sha256"],
                _sha256(input_path.read_bytes()),
            )
            self.assertEqual(
                done["implementation_sha256"],
                summary["implementation"]["bundle_sha256"],
            )
            for name in (
                POSITIVES_NAME,
                NEGATIVES_NAME,
                REVIEW_NAME,
                SUMMARY_NAME,
            ):
                self.assertEqual(
                    done["output_sha256"][name],
                    _sha256((output / name).read_bytes()),
                )
            self.assertFalse(summary["semantics"]["split_assigned"])
            self.assertFalse(summary["semantics"]["human_labels_asserted"])
            self.assertFalse(summary["semantics"]["formal_evidence"])
            self.assertEqual(
                summary["legacy_split_quarantine"][
                    "rows_with_pair_removed"
                ],
                0,
            )
            self.assertEqual(
                summary["legacy_split_quarantine"][
                    "rows_with_no_legacy_pair_attested"
                ],
                2,
            )
            self.assertEqual(
                summary["legacy_split_quarantine"][
                    "input_rows_without_top_level_legacy_fields"
                ],
                2,
            )
            [positive] = _read_jsonl(output / POSITIVES_NAME)
            self.assertEqual(
                positive["r7_expansion_manifest"][
                    "legacy_split_quarantine"
                ],
                {
                    "canonical_sha256": None,
                    "quarantine_policy_version": (
                        LEGACY_SPLIT_QUARANTINE_POLICY_VERSION
                    ),
                    "removed": False,
                    "removed_by_builder": False,
                    "quarantine_stage": "none",
                    "selection_upstream_attestation": False,
                    "source_top_level_fields_removed": [],
                },
            )

    def test_original_sanitized_valid_action_can_never_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            _write_jsonl(
                input_path,
                [
                    _row(
                        0,
                        verdict="valid_action",
                        result_source="original_sanitized",
                    )
                ],
            )
            output = root / "manifest"
            build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
            )
            self.assertEqual(_read_jsonl(output / POSITIVES_NAME), [])
            [review] = _read_jsonl(output / REVIEW_NAME)
            self.assertEqual(
                review["r7_expansion_manifest"]["classification_reason"],
                "sanitized_nonnegative_verdict:valid_action",
            )

    def test_invalid_qwen_digest_fails_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            row = _row(0, verdict="valid_action")
            row["qwen_evidence"]["visual"]["result_digest"] = "0" * 64
            _write_jsonl(input_path, [row])
            output = root / "manifest"
            with self.assertRaisesRegex(ValueError, "result_digest mismatch"):
                build_expansion_manifest(
                    input_path=input_path,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_quarantines_canonical_legacy_split_pair_without_rebinding_qwen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            rows = [
                _with_legacy_split(
                    _row(0, verdict="valid_action"),
                    "train",
                ),
                _with_legacy_split(
                    _row(1, verdict="static", target_motion="none"),
                    "validation",
                ),
                _with_legacy_split(
                    _row(2, verdict="valid_action"),
                    "test",
                ),
                _with_legacy_split(
                    _row(
                        3,
                        verdict="valid_action",
                        result_source="repair_1",
                    ),
                    "train",
                ),
            ]
            original_input_digests = {
                str(row["iid"]): row["input_digest"] for row in rows
            }
            original_evidence_digests = {
                str(row["iid"]): _object_digest(row["qwen_evidence"])
                for row in rows
            }
            original_pair_digests = {
                str(row["iid"]): _object_digest(
                    {
                        "split": row["split"],
                        "split_provenance": row["split_provenance"],
                    }
                )
                for row in rows
            }
            _write_jsonl(input_path, rows)
            output = root / "manifest"
            build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
            )

            bucket_rows = {
                "positive": _read_jsonl(output / POSITIVES_NAME),
                "negative": _read_jsonl(output / NEGATIVES_NAME),
                "review": _read_jsonl(output / REVIEW_NAME),
            }
            self.assertTrue(all(bucket_rows.values()))
            output_rows = [
                row
                for rows_in_bucket in bucket_rows.values()
                for row in rows_in_bucket
            ]
            self.assertEqual(len(output_rows), 4)
            for row in output_rows:
                iid = str(row["iid"])
                self.assertNotIn("split", row)
                self.assertNotIn("split_provenance", row)
                self.assertEqual(
                    row["input_digest"], original_input_digests[iid]
                )
                self.assertEqual(
                    _object_digest(row["qwen_evidence"]),
                    original_evidence_digests[iid],
                )
                audit = row["r7_expansion_manifest"][
                    "legacy_split_quarantine"
                ]
                self.assertTrue(audit["removed"])
                self.assertTrue(audit["removed_by_builder"])
                self.assertEqual(
                    audit["quarantine_stage"], "builder_legacy"
                )
                self.assertEqual(
                    set(audit),
                    {
                        "canonical_sha256",
                        "quarantine_policy_version",
                        "quarantine_stage",
                        "removed",
                        "removed_by_builder",
                        "selection_upstream_attestation",
                        "source_top_level_fields_removed",
                    },
                )
                self.assertEqual(
                    audit["canonical_sha256"],
                    original_pair_digests[iid],
                )
                for forbidden_key in (
                    "legacy_split_value",
                    "legacy_split_provenance",
                    "legacy_split_provenance_sha256",
                    "legacy_split_pair_sha256",
                ):
                    self.assertNotIn(forbidden_key, audit)
                serialized_label = json.dumps(
                    row["r7_expansion_manifest"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for legacy_value in ("train", "validation", "test"):
                    self.assertNotIn(
                        json.dumps(legacy_value),
                        serialized_label,
                    )
                self.assertNotIn(
                    LEGACY_SPLIT_PROVENANCE["version"],
                    serialized_label,
                )
                self.assertNotIn(
                    str(LEGACY_SPLIT_PROVENANCE["seed"]),
                    serialized_label,
                )
                self.assertFalse(
                    row["r7_expansion_manifest"]["split_assigned"]
                )

            summary = json.loads(
                (output / SUMMARY_NAME).read_text(encoding="utf-8")
            )
            quarantine = summary["legacy_split_quarantine"]
            self.assertEqual(quarantine["rows_with_pair_removed"], 4)
            self.assertEqual(quarantine["rows_removed_by_builder"], 4)
            self.assertEqual(
                quarantine["rows_removed_by_selection_upstream"], 0
            )
            self.assertEqual(
                quarantine["rows_with_no_legacy_pair_attested"], 0
            )
            self.assertEqual(
                quarantine["input_rows_without_top_level_legacy_fields"], 0
            )
            self.assertEqual(
                quarantine["quarantine_stage_counts"],
                {"builder_legacy": 4},
            )
            self.assertEqual(
                quarantine["builder_legacy"]["split_value_counts"],
                {"test": 1, "train": 2, "validation": 1},
            )
            provenance_digest = _object_digest(LEGACY_SPLIT_PROVENANCE)
            self.assertEqual(
                quarantine["builder_legacy"][
                    "split_provenance_sha256_counts"
                ],
                {provenance_digest: 4},
            )
            expected_pair_counts = {
                _object_digest(
                    {
                        "split": split,
                        "split_provenance": LEGACY_SPLIT_PROVENANCE,
                    }
                ): count
                for split, count in {
                    "train": 2,
                    "validation": 1,
                    "test": 1,
                }.items()
            }
            self.assertEqual(
                quarantine["builder_legacy"][
                    "split_pair_sha256_counts"
                ],
                dict(sorted(expected_pair_counts.items())),
            )
            self.assertFalse(
                quarantine["output_rows_have_top_level_split"]
            )
            self.assertFalse(
                quarantine["qwen_input_digest_or_evidence_rewritten"]
            )

            verified = build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
                resume=True,
            )
            self.assertTrue(verified["resume_verified"])

    def test_rejects_partial_or_noncanonical_legacy_split_pair(self) -> None:
        cases: dict[str, dict[str, object]] = {}
        split_only = _row(0, verdict="valid_action")
        split_only["split"] = "train"
        cases["split_only"] = split_only
        provenance_only = _row(1, verdict="valid_action")
        provenance_only["split_provenance"] = dict(LEGACY_SPLIT_PROVENANCE)
        cases["provenance_only"] = provenance_only
        for name, split in (
            ("unknown_split", "dev"),
            ("whitespace_split", " train"),
        ):
            row = _row(len(cases), verdict="valid_action")
            row["split"] = split
            row["split_provenance"] = dict(LEGACY_SPLIT_PROVENANCE)
            cases[name] = row
        non_string_split = _row(len(cases), verdict="valid_action")
        non_string_split["split"] = True
        non_string_split["split_provenance"] = dict(
            LEGACY_SPLIT_PROVENANCE
        )
        cases["non_string_split"] = non_string_split
        for name, provenance in (
            (
                "extra_provenance_key",
                {**LEGACY_SPLIT_PROVENANCE, "source": "poison"},
            ),
            (
                "wrong_seed",
                {**LEGACY_SPLIT_PROVENANCE, "seed": 1},
            ),
            (
                "boolean_seed",
                {**LEGACY_SPLIT_PROVENANCE, "seed": True},
            ),
            (
                "wrong_version",
                {**LEGACY_SPLIT_PROVENANCE, "version": "future-v2"},
            ),
            ("non_object_provenance", ["caption-or-path-fallback-v1"]),
        ):
            row = _row(len(cases), verdict="valid_action")
            row["split"] = "train"
            row["split_provenance"] = provenance
            cases[name] = row

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_number, (name, row) in enumerate(cases.items()):
                with self.subTest(name=name):
                    input_path = root / f"{case_number:02d}-{name}.jsonl"
                    output = root / f"{case_number:02d}-{name}-output"
                    _write_jsonl(input_path, [row])
                    with self.assertRaisesRegex(
                        ValueError,
                        "legacy split",
                    ):
                        build_expansion_manifest(
                            input_path=input_path,
                            output_dir=output,
                        )
                    self.assertFalse(output.exists())

    def test_accepts_strict_selection_upstream_quarantine_attestation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_pair = {
                "split": "validation",
                "split_provenance": LEGACY_SPLIT_PROVENANCE,
            }
            removed_upstream = _row(0, verdict="valid_action")
            removed_upstream["r7_expansion_selection"][
                "legacy_split_quarantine"
            ] = {
                "present": True,
                "canonical_sha256": _object_digest(legacy_pair),
            }
            absent_upstream = _row(
                1,
                verdict="static",
                target_motion="none",
            )
            absent_upstream["r7_expansion_selection"][
                "legacy_split_quarantine"
            ] = {
                "present": False,
                "canonical_sha256": None,
            }
            input_path = root / "fused.jsonl"
            _write_jsonl(
                input_path,
                [removed_upstream, absent_upstream],
            )
            output = root / "manifest"
            build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
            )

            [positive] = _read_jsonl(output / POSITIVES_NAME)
            [negative] = _read_jsonl(output / NEGATIVES_NAME)
            upstream_audit = positive["r7_expansion_manifest"][
                "legacy_split_quarantine"
            ]
            self.assertEqual(
                upstream_audit,
                {
                    "canonical_sha256": _object_digest(legacy_pair),
                    "quarantine_policy_version": (
                        LEGACY_SPLIT_QUARANTINE_POLICY_VERSION
                    ),
                    "quarantine_stage": "selection_upstream",
                    "removed": True,
                    "removed_by_builder": False,
                    "selection_upstream_attestation": True,
                    "source_top_level_fields_removed": [
                        "split",
                        "split_provenance",
                    ],
                },
            )
            absent_audit = negative["r7_expansion_manifest"][
                "legacy_split_quarantine"
            ]
            self.assertEqual(absent_audit["quarantine_stage"], "none")
            self.assertFalse(absent_audit["removed"])
            self.assertTrue(
                absent_audit["selection_upstream_attestation"]
            )
            for row in (positive, negative):
                self.assertNotIn("split", row)
                self.assertNotIn("split_provenance", row)
                self.assertFalse(
                    row["r7_expansion_manifest"]["split_assigned"]
                )

            summary = json.loads(
                (output / SUMMARY_NAME).read_text(encoding="utf-8")
            )
            quarantine = summary["legacy_split_quarantine"]
            self.assertEqual(
                quarantine["quarantine_stage_counts"],
                {"none": 1, "selection_upstream": 1},
            )
            self.assertEqual(quarantine["rows_with_pair_removed"], 1)
            self.assertEqual(quarantine["rows_removed_by_builder"], 0)
            self.assertEqual(
                quarantine["rows_removed_by_selection_upstream"], 1
            )
            self.assertEqual(
                quarantine["input_rows_without_top_level_legacy_fields"], 2
            )
            self.assertEqual(
                quarantine["builder_legacy"]["split_value_counts"],
                {},
            )
            self.assertEqual(
                quarantine["selection_upstream"][
                    "canonical_sha256_counts"
                ],
                {_object_digest(legacy_pair): 1},
            )
            self.assertEqual(
                quarantine[
                    "all_removed_pair_canonical_sha256_counts"
                ],
                {_object_digest(legacy_pair): 1},
            )

    def test_rejects_malformed_or_conflicting_upstream_attestation(
        self,
    ) -> None:
        valid_digest = _object_digest(
            {
                "split": "train",
                "split_provenance": LEGACY_SPLIT_PROVENANCE,
            }
        )
        metadata_cases: dict[str, object] = {
            "missing_digest": {"present": True},
            "extra_key": {
                "present": True,
                "canonical_sha256": valid_digest,
                "split": "train",
            },
            "non_boolean_present": {
                "present": 1,
                "canonical_sha256": valid_digest,
            },
            "true_null_digest": {
                "present": True,
                "canonical_sha256": None,
            },
            "true_uppercase_digest": {
                "present": True,
                "canonical_sha256": valid_digest.upper(),
            },
            "false_non_null_digest": {
                "present": False,
                "canonical_sha256": valid_digest,
            },
            "non_object": ["present", True],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_number = 0
            for name, metadata in metadata_cases.items():
                with self.subTest(name=name):
                    row = _row(case_number, verdict="valid_action")
                    row["r7_expansion_selection"][
                        "legacy_split_quarantine"
                    ] = metadata
                    input_path = root / f"{case_number:02d}-{name}.jsonl"
                    output = root / f"{case_number:02d}-{name}-output"
                    _write_jsonl(input_path, [row])
                    with self.assertRaisesRegex(
                        ValueError,
                        "legacy split|legacy_split_quarantine",
                    ):
                        build_expansion_manifest(
                            input_path=input_path,
                            output_dir=output,
                        )
                    self.assertFalse(output.exists())
                    case_number += 1

            with self.subTest(name="upstream_and_top_level_conflict"):
                row = _with_legacy_split(
                    _row(case_number, verdict="valid_action"),
                    "train",
                )
                row["r7_expansion_selection"][
                    "legacy_split_quarantine"
                ] = {
                    "present": True,
                    "canonical_sha256": valid_digest,
                }
                input_path = root / "conflict.jsonl"
                output = root / "conflict-output"
                _write_jsonl(input_path, [row])
                with self.assertRaisesRegex(
                    ValueError,
                    "both selection-upstream",
                ):
                    build_expansion_manifest(
                        input_path=input_path,
                        output_dir=output,
                    )
                self.assertFalse(output.exists())

    def test_duplicate_iids_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_path = root / "duplicate.jsonl"
            row = _row(1, verdict="valid_action")
            _write_jsonl(duplicate_path, [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate input IID"):
                build_expansion_manifest(
                    input_path=duplicate_path,
                    output_dir=root / "duplicate-output",
                )

    def test_overwrite_refused_and_resume_is_verification_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "fused.jsonl"
            _write_jsonl(input_path, [_row(0, verdict="valid_action")])
            output = root / "manifest"
            build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
            )
            with self.assertRaises(FileExistsError):
                build_expansion_manifest(
                    input_path=input_path,
                    output_dir=output,
                )
            verified = build_expansion_manifest(
                input_path=input_path,
                output_dir=output,
                resume=True,
            )
            self.assertTrue(verified["resume_verified"])
            with self.assertRaises(FileNotFoundError):
                build_expansion_manifest(
                    input_path=input_path,
                    output_dir=root / "missing",
                    resume=True,
                )
            (output / REVIEW_NAME).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                build_expansion_manifest(
                    input_path=input_path,
                    output_dir=output,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
