from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from motive.validate_action_training import validate_action_training


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class ValidateActionTrainingTests(unittest.TestCase):
    def test_accepts_finite_curve_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            log.write_text(
                "step=1 loss=0.2 grad=0.5 gradl=0.4 gradr=0.3\n"
                "step=2 loss=0.1 grad=0.4 gradl=0.3 gradr=0.2\n",
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "step": 2,
                    "trainable_transformer": {"weight": torch.ones(2)},
                    "optimizer": {"state": {0: {"exp_avg": torch.zeros(2)}}},
                },
                checkpoint,
            )

            report = validate_action_training(
                log_path=log,
                checkpoint_path=checkpoint,
                expected_step=2,
                max_total_grad=1.0,
                max_transformer_grad=1.0,
                max_router_grad=1.0,
            )

        self.assertTrue(report["complete"])
        self.assertEqual(report["checkpoint_nonfinite_tensors"], [])

    def test_rejects_pathological_gradient_and_nonfinite_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            log.write_text(
                "step=2 loss=0.1 grad=1e9 gradl=1e9 gradr=0.1\n",
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "step": 2,
                    "trainable_transformer": {
                        "weight": torch.tensor([float("nan")])
                    },
                },
                checkpoint,
            )

            report = validate_action_training(
                log_path=log,
                checkpoint_path=checkpoint,
                expected_step=2,
                max_total_grad=100.0,
                max_transformer_grad=100.0,
                max_router_grad=100.0,
            )

        self.assertFalse(report["complete"])
        self.assertFalse(report["checks"]["grad_within_limit"])
        self.assertFalse(report["checks"]["checkpoint_tensors_finite"])

    def test_can_derive_group_norms_for_reused_plain_lora_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "train.log"
            log.write_text(
                "step=2 loss=0.1 grad=0.2\n",
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint.pt"
            torch.save({"step": 2, "weight": torch.ones(1)}, checkpoint)

            report = validate_action_training(
                log_path=log,
                checkpoint_path=checkpoint,
                expected_step=2,
                max_total_grad=1.0,
                max_transformer_grad=1.0,
                max_router_grad=1.0,
                allow_legacy_group_gradients=True,
            )

        self.assertTrue(report["complete"])
        self.assertTrue(report["legacy_group_gradients_derived"])
        self.assertEqual(report["gradient_maxima"]["gradl"], 0.2)
        self.assertEqual(report["gradient_maxima"]["gradr"], 0.0)


if __name__ == "__main__":
    unittest.main()
