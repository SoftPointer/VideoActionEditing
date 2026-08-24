#!/usr/bin/env python3

"""Tests for conjunctive joint G1 admission."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBJECT_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "score_g1_joint_action_repr_admission_v1.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "score_g1_joint_action_repr_admission_v1_test", SUBJECT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeEvaluator:
    @staticmethod
    def verify_evaluation_receipt(path: Path | str):
        return json.loads(Path(path).read_text(encoding="ascii"))


class JointAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original_loader = subject._evaluator_module
        subject._evaluator_module = lambda: _FakeEvaluator

    def tearDown(self) -> None:
        subject._evaluator_module = self.original_loader
        self.temporary.cleanup()

    def _scores(self, *, fail_modality: str | None = None) -> dict:
        result = {}
        for modality in subject.MODALITIES:
            branches = {}
            for index, branch in enumerate(subject.BRANCHES):
                base = 0.95 if branch == "correct" else 0.10 + index * 0.08
                branches[branch] = {axis: base for axis in subject.SCORE_AXES}
            if modality == fail_modality:
                branches["reverse"]["ordered_transitions"] = 0.99
            result[modality] = branches
        return result

    def _evaluation(
        self,
        *,
        case_id: str,
        family: str,
        anchor: str,
        fail_modality: str | None = None,
    ) -> tuple[Path, str]:
        receipt = {
            "case_id": case_id,
            "action_family": family,
            "subject_anchor_kind": anchor,
            "reference_anchor_kind": "target",
            "selfgen_uses_same_case_real_target_reference": anchor == "selfgen",
            "scoring_contract": {
                "weighted_compensation_forbidden": True,
                "total_energy_only": False,
            },
            "modality_scores": self._scores(fail_modality=fail_modality),
        }
        path = self.root / f"{case_id}-{anchor}.evaluation.json"
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="ascii")
        return path, _sha(path)

    def _manifest(
        self,
        *,
        admission_scope: str = "both",
        failure: tuple[str, str] | None = None,
        omit_last: bool = False,
    ) -> Path:
        cases = [
            {"case_id": "fit-case", "split": "fit", "action_family": "head_turn"},
            {"case_id": "held-case", "split": "heldout", "action_family": "object_place"},
        ]
        rows = []
        required_anchors = (
            subject.ANCHOR_KINDS
            if admission_scope == "both"
            else (admission_scope,)
        )
        for case in cases:
            for anchor in required_anchors:
                fail_modality = None
                if failure == (case["case_id"], anchor):
                    fail_modality = "middle"
                path, digest = self._evaluation(
                    case_id=case["case_id"],
                    family=case["action_family"],
                    anchor=anchor,
                    fail_modality=fail_modality,
                )
                rows.append(
                    {
                        **case,
                        "anchor_kind": anchor,
                        "evaluation_receipt_path": str(path),
                        "evaluation_receipt_sha256": digest,
                    }
                )
        if omit_last:
            rows.pop()
        manifest = {
            "schema_version": subject.EVIDENCE_SCHEMA_VERSION,
            "experiment_id": "g1-joint-test",
            "admission_scope": admission_scope,
            "expected_cases": cases,
            "evaluations": rows,
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="ascii")
        return path

    def test_all_controls_modalities_cases_and_anchors_must_pass(self) -> None:
        manifest = self._manifest()
        output = self.root / "admission.json"
        result = subject.score_and_publish(manifest, output)
        self.assertTrue(result["g1_target_passed"])
        self.assertTrue(result["g1_selfgen_passed"])
        self.assertTrue(result["g1_all_anchor_kinds_passed"])
        self.assertFalse(result["optimizer_creation_authorized_by_this_receipt"])
        self.assertFalse(result["weighted_or_scalar_compensation_used"])
        self.assertEqual(subject.verify_admission_receipt(output), result)

    def test_middle_failure_cannot_be_compensated_by_flow(self) -> None:
        result = subject.evaluate_manifest(
            self._manifest(failure=("held-case", "selfgen"))
        )
        self.assertTrue(result["g1_target_passed"])
        self.assertFalse(result["g1_selfgen_passed"])
        decision = result["anchor_decisions"]["selfgen"]
        self.assertTrue(decision["flow_g1_passed"])
        self.assertFalse(decision["middle_g1_passed"])
        self.assertFalse(decision["joint_g1_passed"])
        self.assertEqual(decision["failed_middle_case_ids"], ["held-case"])

    def test_target_and_selfgen_coverage_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(subject.G1JointAdmissionError, "sealed admission scope"):
            subject.evaluate_manifest(self._manifest(omit_last=True))

    def test_target_scope_closes_only_target_and_marks_selfgen_not_evaluated(self) -> None:
        manifest = self._manifest(admission_scope="target")
        output = self.root / "target-admission.json"
        result = subject.score_and_publish(
            manifest,
            output,
            admission_scope="target",
        )
        self.assertEqual(result["admission_scope"], "target")
        self.assertEqual(result["scope_required_anchor_kinds"], ["target"])
        self.assertTrue(result["g1_scope_passed"])
        self.assertTrue(result["g1_target_passed"])
        self.assertEqual(result["g1_target_status"], "passed")
        self.assertIsNone(result["g1_selfgen_passed"])
        self.assertEqual(result["g1_selfgen_status"], "not_evaluated")
        self.assertIsNone(result["g1_all_anchor_kinds_passed"])
        self.assertFalse(result["optimizer_creation_authorized_by_this_receipt"])
        self.assertEqual(subject.verify_admission_receipt(output), result)

    def test_selfgen_scope_requires_and_records_real_target_reference(self) -> None:
        result = subject.evaluate_manifest(
            self._manifest(admission_scope="selfgen"),
            admission_scope="selfgen",
        )
        self.assertEqual(result["g1_target_status"], "not_evaluated")
        self.assertEqual(result["g1_selfgen_status"], "passed")
        self.assertTrue(result["g1_scope_passed"])
        self.assertTrue(
            all(
                row["anchor_kind"] == "selfgen"
                for row in result["cohort_decisions"]
            )
        )

    def test_selfgen_scope_rejects_self_reference_claim(self) -> None:
        manifest = self._manifest(admission_scope="selfgen")
        document = json.loads(manifest.read_text(encoding="ascii"))
        row = document["evaluations"][0]
        score_path = Path(row["evaluation_receipt_path"])
        receipt = json.loads(score_path.read_text(encoding="ascii"))
        receipt["selfgen_uses_same_case_real_target_reference"] = False
        score_path.write_text(
            json.dumps(receipt, sort_keys=True) + "\n",
            encoding="ascii",
        )
        row["evaluation_receipt_sha256"] = _sha(score_path)
        manifest.write_text(
            json.dumps(document, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            subject.G1JointAdmissionError,
            "identity/contract",
        ):
            subject.evaluate_manifest(
                manifest,
                admission_scope="selfgen",
            )

    def test_cli_scope_must_match_manifest_seal_and_defaults_to_both(self) -> None:
        target_manifest = self._manifest(admission_scope="target")
        with self.assertRaisesRegex(
            subject.G1JointAdmissionError,
            "CLI admission scope differs",
        ):
            subject.evaluate_manifest(target_manifest)
        args = subject._parser().parse_args(
            [
                "score",
                "--evidence-manifest",
                str(target_manifest),
                "--output",
                str(self.root / "unused.json"),
            ]
        )
        self.assertEqual(args.admission_scope, "both")

    def test_scope_rejects_extra_anchor_rows_instead_of_ignoring_them(self) -> None:
        manifest = self._manifest()
        document = json.loads(manifest.read_text(encoding="ascii"))
        document["admission_scope"] = "target"
        manifest.write_text(
            json.dumps(document, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            subject.G1JointAdmissionError,
            "exactly cover",
        ):
            subject.evaluate_manifest(
                manifest,
                admission_scope="target",
            )

    def test_equal_or_negative_margin_fails_even_if_other_axes_are_high(self) -> None:
        manifest = self._manifest()
        document = json.loads(manifest.read_text(encoding="ascii"))
        row = document["evaluations"][0]
        score_path = Path(row["evaluation_receipt_path"])
        receipt = json.loads(score_path.read_text(encoding="ascii"))
        receipt["modality_scores"]["flow"]["temporal_shuffle"]["ordered_transitions"] = 0.95
        score_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="ascii")
        row["evaluation_receipt_sha256"] = _sha(score_path)
        manifest.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="ascii")
        result = subject.evaluate_manifest(manifest)
        first = next(
            item
            for item in result["cohort_decisions"]
            if item["case_id"] == "fit-case" and item["anchor_kind"] == "target"
        )
        comparison = first["modality_decisions"]["flow"]["comparisons"]["temporal_shuffle"]
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["axes"]["ordered_transitions"]["margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
