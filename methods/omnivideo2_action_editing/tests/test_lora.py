import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from pact.lora import (
    LoRALinear,
    expected_lora_module_count,
    inject_lora,
    load_lora_weights,
    lora_scope_target_regex,
    lora_state_dict,
    save_lora_weights,
)


class LinearLoRATests(unittest.TestCase):
    def _model(self) -> nn.Module:
        return nn.Sequential(
            OrderedDict(
                [
                    ("q_proj", nn.Linear(5, 7)),
                    ("activation", nn.SiLU()),
                    ("out", nn.Linear(7, 3)),
                ]
            )
        )

    def test_zero_initialized_injection_has_exact_forward_parity(self) -> None:
        torch.manual_seed(9)
        model = self._model()
        inputs = torch.randn(4, 5)
        before = model(inputs).detach().clone()
        names = inject_lora(model, r"q_proj$", rank=2, alpha=4)
        after = model(inputs).detach()
        self.assertEqual(names, ["q_proj"])
        self.assertTrue(torch.equal(before, after))
        self.assertIsInstance(model.q_proj, LoRALinear)
        self.assertIsInstance(model.out, nn.Linear)
        self.assertFalse(model.q_proj.base.weight.requires_grad)
        self.assertTrue(model.out.weight.requires_grad)

    def test_adapter_gradient_and_regex_failure(self) -> None:
        model = self._model()
        inject_lora(model, r"q_proj", rank=2)
        model(torch.randn(3, 5)).sum().backward()
        self.assertIsNone(model.q_proj.base.weight.grad)
        self.assertIsNotNone(model.q_proj.lora_B.weight.grad)
        self.assertGreater(float(model.q_proj.lora_B.weight.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            inject_lora(model, r"does_not_exist", rank=2)

    def test_adapter_dtype_can_keep_fp32_master_parameters(self) -> None:
        model = self._model().to(dtype=torch.bfloat16)
        inject_lora(
            model,
            r"q_proj$",
            rank=2,
            adapter_dtype=torch.float32,
        )
        self.assertEqual(model.q_proj.base.weight.dtype, torch.bfloat16)
        self.assertEqual(model.q_proj.lora_A.weight.dtype, torch.float32)
        self.assertEqual(model.q_proj.lora_B.weight.dtype, torch.float32)

    def test_module_list_targeting(self) -> None:
        model = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        names = inject_lora(model, r"^1$", rank=1)
        self.assertEqual(names, ["1"])
        self.assertIsInstance(model[1], LoRALinear)
        self.assertIsInstance(model[0], nn.Linear)

    def test_closed_wan_scope_presets_match_exact_modules(self) -> None:
        class Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                for name in ("q", "k", "v", "o"):
                    setattr(self, name, nn.Linear(4, 4))

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = Attention()
                self.cross_attn = Attention()
                self.ffn = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))

        class WrappedWan(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wan_model = nn.Module()
                self.wan_model.blocks = nn.ModuleList([Block(), Block()])

        expected = {"cross_qo": 4, "all_attn": 16, "diffsynth_full": 20}
        for scope, count in expected.items():
            with self.subTest(scope=scope):
                model = WrappedWan()
                names = inject_lora(
                    model,
                    lora_scope_target_regex(scope),
                    rank=1,
                )
                self.assertEqual(len(names), count)
                self.assertEqual(expected_lora_module_count(scope, 2), count)
        with self.assertRaises(ValueError):
            lora_scope_target_regex("custom")

    def test_adapter_only_state_and_save(self) -> None:
        model = self._model()
        inject_lora(model, r"(q_proj|out)$", rank=2)
        state = lora_state_dict(model)
        self.assertEqual(
            set(state),
            {
                "q_proj.lora_A.weight",
                "q_proj.lora_B.weight",
                "out.lora_A.weight",
                "out.lora_B.weight",
            },
        )
        self.assertFalse(any("base" in key for key in state))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapters.pt"
            self.assertEqual(
                save_lora_weights(
                    model, path, base_checkpoint_sha256="a" * 64
                ),
                path,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["format"], "pact-linear-lora-v1")
        self.assertEqual(set(payload["state_dict"]), set(state))
        self.assertEqual(payload["config"]["q_proj"]["rank"], 2)

    def test_save_load_round_trip_reproduces_adapter_output(self) -> None:
        torch.manual_seed(17)
        model = self._model()
        base_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        inject_lora(model, r"(q_proj|out)$", rank=2, alpha=4)
        with torch.no_grad():
            model.q_proj.lora_B.weight.normal_()
            model.out.lora_B.weight.normal_()
        inputs = torch.randn(3, 5)
        expected = model(inputs).detach()

        restored = self._model()
        restored.load_state_dict(base_state)
        inject_lora(restored, r"(q_proj|out)$", rank=2, alpha=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapters.pt"
            save_lora_weights(
                model, path, base_checkpoint_sha256="b" * 64
            )
            names = load_lora_weights(
                restored,
                path,
                expected_base_checkpoint_sha256="b" * 64,
            )
        self.assertEqual(names, ["q_proj", "out"])
        self.assertTrue(torch.equal(restored(inputs), expected))


if __name__ == "__main__":
    unittest.main()
