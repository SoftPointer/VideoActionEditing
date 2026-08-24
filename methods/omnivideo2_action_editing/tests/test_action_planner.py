from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.planner import (  # noqa: E402
    TemporalMotionPlanPredictor,
    motion_plan_loss,
)


class TemporalMotionPlanPredictorTest(unittest.TestCase):
    def _model(self) -> TemporalMotionPlanPredictor:
        return TemporalMotionPlanPredictor(
            4,
            input_dim=2048,
            hidden_dim=32,
            depth=1,
            output_dim=2048,
            num_heads=8,
            mlp_ratio=2.0,
        )

    def test_source_only_forward_predicts_k_2048_tokens(self) -> None:
        model = self._model()
        source = torch.randn(2, 3, 2048, requires_grad=True)
        prediction = model(source)
        self.assertEqual(prediction.shape, (2, 4, 2048))
        self.assertEqual(
            list(inspect.signature(model.forward).parameters),
            ["source_vlm_context"],
        )
        prediction.square().mean().backward()
        self.assertIsNotNone(source.grad)
        self.assertGreater(float(source.grad.abs().sum()), 0.0)

    def test_variable_source_lengths_are_padded_internally(self) -> None:
        model = self._model().eval()
        first = torch.randn(2, 2048)
        second = torch.randn(5, 2048)
        with torch.no_grad():
            batch_prediction = model([first, second])
            first_prediction = model(first)
            second_prediction = model(second)
        self.assertEqual(batch_prediction.shape, (2, 4, 2048))
        self.assertTrue(
            torch.allclose(batch_prediction[0], first_prediction[0], atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(batch_prediction[1], second_prediction[0], atol=1e-5)
        )

    def test_target_tokens_are_detached_labels_used_only_by_loss(self) -> None:
        model = self._model()
        source = torch.randn(1, 3, 2048, requires_grad=True)
        target = torch.randn(1, 4, 2048, requires_grad=True)
        prediction = model(source)
        loss = motion_plan_loss(prediction, target)
        loss.backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNone(target.grad)
        self.assertGreater(float(source.grad.abs().sum()), 0.0)

    def test_dimensions_and_loss_shapes_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 2048"):
            TemporalMotionPlanPredictor(4, input_dim=1024, hidden_dim=32)
        with self.assertRaisesRegex(ValueError, "divisible"):
            TemporalMotionPlanPredictor(4, hidden_dim=30, num_heads=8)

        prediction = torch.randn(1, 4, 2048)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            motion_plan_loss(prediction, torch.randn(1, 3, 2048))
        with self.assertRaisesRegex(ValueError, "shape"):
            motion_plan_loss(prediction, torch.randn(4, 2048))


if __name__ == "__main__":
    unittest.main()
