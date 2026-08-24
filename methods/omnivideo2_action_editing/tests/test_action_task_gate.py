from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action.omni import (  # noqa: E402
    load_official_omnivideo2_1_3b,
    set_action_lora_gate,
)
from pact.lora import LoRALinear  # noqa: E402
from train_omnivideo2_action import MARPOmniTrainingModel  # noqa: E402


class _TinyAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = nn.Linear(3, 2, bias=False)
        self.projection = LoRALinear(
            base,
            rank=2,
            alpha=2.0,
            adapter_dtype=torch.float32,
        )
        with torch.no_grad():
            self.projection.lora_A.weight.fill_(0.25)
            self.projection.lora_B.weight.fill_(0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class ActionTaskGateTest(unittest.TestCase):
    def test_per_sample_zero_gate_is_exact_native_linear_output(self) -> None:
        model = _TinyAdapterModel()
        value = torch.ones(2, 4, 3)
        base = model.projection.base(value)
        self.assertEqual(
            set_action_lora_gate(model, torch.tensor([0.0, 1.0])), 1
        )
        output = model(value)
        self.assertTrue(torch.equal(output[0], base[0]))
        self.assertFalse(torch.equal(output[1], base[1]))
        output.sum().backward()
        self.assertIsNotNone(model.projection.lora_A.weight.grad)
        self.assertIsNotNone(model.projection.lora_B.weight.grad)

    def test_renderer_forward_has_no_target_motion_argument(self) -> None:
        parameters = set(inspect.signature(MARPOmniTrainingModel.forward).parameters)
        self.assertNotIn("target_motion_tokens", parameters)
        self.assertEqual(
            parameters,
            {
                "self",
                "x_t",
                "timestep",
                "text_context",
                "source_vlm_context",
                "source_latent",
                "sample_ids",
                "task_types",
            },
        )

    def test_uncompressed_source_policy_belongs_to_renderer_not_loader(self) -> None:
        loader_parameters = set(
            inspect.signature(load_official_omnivideo2_1_3b).parameters
        )
        renderer_parameters = set(
            inspect.signature(MARPOmniTrainingModel.__init__).parameters
        )
        self.assertNotIn("require_uncompressed_source", loader_parameters)
        self.assertIn("require_uncompressed_source", renderer_parameters)


if __name__ == "__main__":
    unittest.main()
