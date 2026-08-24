from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import analyze_fitq_engineering_scan as analysis  # noqa: E402
import infer_fitq_official_runtime_scan as runtime  # noqa: E402


def valid_receipt() -> dict:
    value = {
        "schema_version": runtime.RECEIPT_SCHEMA,
        "training_authorized": False,
        "fitq_stage1_authorized": False,
        "scientific_claim_authorized": False,
        "optimizer_update": "null",
        "training": {
            "forward_only": True,
            "backward_performed": False,
            "optimizer_present": False,
            "checkpoint_saved": False,
            "model_weights_written": False,
        },
        "fitq_observation": {
            "statistics_artifact_count": 85,
            "context_count": 85,
            "tokenwise_localization_available": False,
            "fitq_go_authorized": False,
            "proposal_bank_status": "insufficient_bank",
            "statistics_artifacts": [{} for _ in range(85)],
        },
    }
    value["receipt_digest"] = analysis.object_sha256(value)
    return value


class FITQEngineeringAnalysisContractTests(unittest.TestCase):
    def test_receipt_validation_is_fail_closed_and_non_authorizing(self) -> None:
        receipt = valid_receipt()
        self.assertEqual(
            analysis.validate_runtime_receipt(receipt)["optimizer_update"], "null"
        )
        for field, value in (
            ("training_authorized", True),
            ("fitq_stage1_authorized", True),
            ("scientific_claim_authorized", True),
            ("optimizer_update", "step"),
        ):
            changed = valid_receipt()
            changed[field] = value
            changed["receipt_digest"] = analysis.object_sha256(
                {key: item for key, item in changed.items() if key != "receipt_digest"}
            )
            with self.subTest(field=field), self.assertRaises(
                analysis.FITQEngineeringAnalysisError
            ):
                analysis.validate_runtime_receipt(changed)

    def test_digest_and_artifact_mutation_are_detected(self) -> None:
        receipt = valid_receipt()
        receipt["fitq_observation"]["statistics_artifact_count"] = 84
        with self.assertRaisesRegex(
            analysis.FITQEngineeringAnalysisError, "digest differs"
        ):
            analysis.validate_runtime_receipt(receipt)

        receipt["receipt_digest"] = analysis.object_sha256(
            {key: item for key, item in receipt.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(
            analysis.FITQEngineeringAnalysisError, "scope differs"
        ):
            analysis.validate_runtime_receipt(receipt)

    def test_grid_and_scientific_limit_are_source_pinned(self) -> None:
        self.assertEqual(analysis.EXPECTED_SIGMAS, (0.8, 0.6, 0.35, 0.15))
        self.assertEqual(
            tuple(
                analysis._normalize_runtime_sigma(value)
                for value in analysis.EXPECTED_RUNTIME_FP32_SIGMAS
            ),
            analysis.EXPECTED_SIGMAS,
        )
        with self.assertRaisesRegex(
            analysis.FITQEngineeringAnalysisError, "exact registered FP32"
        ):
            analysis._normalize_runtime_sigma(0.8)
        self.assertEqual(analysis.EXPECTED_LAMBDAS, (1.0, 0.5, 0.0))
        self.assertEqual(len(analysis.EXPECTED_BRANCHES), 7)
        self.assertEqual(
            analysis.MIN_ACTION_TO_DUPLICATE_ENERGY_RATIO, 10.0
        )
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("backward", calls)
        self.assertNotIn("step", calls)
        self.assertIn('"optimizer_update": "null"', source)
        self.assertIn('"scientific_fitq_outcome": "not_evaluated_insufficient_bank"', source)


if __name__ == "__main__":
    unittest.main()
