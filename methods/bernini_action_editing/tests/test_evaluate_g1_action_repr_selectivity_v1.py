#!/usr/bin/env python3

"""Tests for deterministic joint action-representation evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBJECT_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "evaluate_g1_action_repr_selectivity_v1.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "evaluate_g1_action_repr_selectivity_v1_test", SUBJECT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


subject = _load_subject()

try:
    import torch  # noqa: F401
    import safetensors  # noqa: F401
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


@unittest.skipUnless(torch is not None, "PyTorch/safetensors are unavailable")
class DeterministicJointEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        from methods.bernini_action_editing.tests.test_materialize_g1_flow_control_cohort_v1 import (
            G1FlowControlCohortTests,
        )
        from methods.bernini_action_editing.tests.test_materialize_g1_middle_control_cohort_v1 import (
            G1MiddleControlTests,
        )

        self.flow_fixture = G1FlowControlCohortTests(methodName="runTest")
        self.middle_fixture = G1MiddleControlTests(methodName="runTest")
        self.flow_fixture.setUp()
        self.middle_fixture.setUp()
        self.target_flow = self.flow_fixture._materialize("target-flow", "target")
        self.target_middle = self.middle_fixture._materialize("target-middle", "target")

        flow = self.flow_fixture
        flow.correct = flow._bundle("sg-correct", factor=0.96, offset=0.1, source="sg-source-a")
        flow.shuffle = flow._bundle("sg-shuffle", factor=0.7, offset=7.0, source="sg-source-a")
        flow.reverse = flow._bundle("sg-reverse", factor=-0.8, offset=2.0, source="sg-source-a")
        flow.wrong = flow._bundle("sg-wrong", factor=2.2, offset=-5.0, source="sg-source-b")
        self.selfgen_flow = flow._materialize("selfgen-flow", "selfgen")

        middle = self.middle_fixture
        middle.correct = middle._middle(
            "sg-correct", case_id="case-a", role="self_generated", instruction="turn", factor=0.96
        )
        middle.shuffle = middle._middle(
            "sg-shuffle",
            case_id="case-a",
            role="self_generated_temporal_shuffle",
            instruction="turn",
            factor=0.75,
            phase_shift=4,
        )
        middle.reverse = middle._middle(
            "sg-reverse",
            case_id="case-a",
            role="self_generated_reverse",
            instruction="turn",
            factor=-0.8,
            reverse=True,
        )
        middle.wrong = middle._middle(
            "sg-wrong",
            case_id="case-b",
            role="self_generated",
            instruction="lift",
            factor=2.2,
            phase_shift=6,
        )
        self.selfgen_middle = middle._materialize("selfgen-middle", "selfgen")

    def tearDown(self) -> None:
        self.middle_fixture.tearDown()
        self.flow_fixture.tearDown()

    @property
    def target_flow_receipt(self) -> Path:
        return self.flow_fixture.root / "target-flow" / "cohort_receipt.json"

    @property
    def target_middle_receipt(self) -> Path:
        return self.middle_fixture.root / "target-middle" / "cohort_receipt.json"

    @property
    def selfgen_flow_receipt(self) -> Path:
        return self.flow_fixture.root / "selfgen-flow" / "cohort_receipt.json"

    @property
    def selfgen_middle_receipt(self) -> Path:
        return self.middle_fixture.root / "selfgen-middle" / "cohort_receipt.json"

    def test_selfgen_is_scored_against_same_case_real_target_not_itself(self) -> None:
        receipt = subject.build_evaluation(
            target_flow_receipt=self.target_flow_receipt,
            target_middle_receipt=self.target_middle_receipt,
            subject_flow_receipt=self.selfgen_flow_receipt,
            subject_middle_receipt=self.selfgen_middle_receipt,
        )
        self.assertEqual(receipt["reference_anchor_kind"], "target")
        self.assertEqual(receipt["subject_anchor_kind"], "selfgen")
        self.assertTrue(receipt["selfgen_uses_same_case_real_target_reference"])
        self.assertFalse(receipt["scoring_contract"]["total_energy_only"])
        self.assertTrue(receipt["scoring_contract"]["weighted_compensation_forbidden"])
        for modality in subject.MODALITIES:
            scores = receipt["modality_scores"][modality]
            self.assertGreater(scores["correct"]["action_presence"], scores["zero_or_noop"]["action_presence"])
            self.assertGreater(scores["correct"]["completion"], scores["incomplete"]["completion"])
            self.assertGreater(
                scores["correct"]["action_identity"],
                scores["wrong_action_energy_matched"]["action_identity"],
            )

    def test_target_correct_self_comparison_is_exact_one_and_receipt_replays(self) -> None:
        output = self.flow_fixture.root / "target-evaluation.json"
        receipt = subject.evaluate_and_publish(
            target_flow_receipt=self.target_flow_receipt,
            target_middle_receipt=self.target_middle_receipt,
            subject_flow_receipt=self.target_flow_receipt,
            subject_middle_receipt=self.target_middle_receipt,
            output=output,
        )
        for modality in subject.MODALITIES:
            self.assertEqual(set(receipt["modality_scores"][modality]["correct"].values()), {1.0})
        self.assertEqual(subject.verify_evaluation_receipt(output), receipt)

    def test_selfgen_cannot_be_used_as_the_reference(self) -> None:
        with self.assertRaisesRegex(subject.G1DeterministicEvaluatorError, "reference must be real-target"):
            subject.build_evaluation(
                target_flow_receipt=self.selfgen_flow_receipt,
                target_middle_receipt=self.selfgen_middle_receipt,
                subject_flow_receipt=self.selfgen_flow_receipt,
                subject_middle_receipt=self.selfgen_middle_receipt,
            )


if __name__ == "__main__":
    unittest.main()
