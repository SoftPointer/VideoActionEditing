from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from pact.checkpoint import AdapterCheckpointError, load_pact_adapter_bundle
from pact.lora import inject_lora, lora_scope_target_regex, lora_state_dict
from pact.router import PromptConditionedMaskRouter
from pact.training import validate_training_config


class _CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.o = nn.Linear(4, 4)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_attn = _CrossAttention()


class _Wan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 2
        self.blocks = nn.ModuleList([_Block()])


class _FakeUnified(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wan_model = _Wan()
        self.visual_context_adapter = nn.Linear(3, 3)
        self.vlm_norm = nn.LayerNorm(4)
        self.vlm_proj = nn.Linear(4, 4)


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


class PactAdapterCheckpointTest(unittest.TestCase):
    def _config(self) -> dict:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "pact_1_3b_cross_qo.json"
        )
        return validate_training_config(json.loads(path.read_text(encoding="utf-8")))

    def _bundle(self) -> tuple[
        dict,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        torch.manual_seed(23)
        config = self._config()
        model = _FakeUnified()
        base_state = _state(model)
        names = inject_lora(
            model,
            lora_scope_target_regex(config["lora"]["scope"]),
            rank=config["lora"]["rank"],
            alpha=config["lora"]["alpha"],
            dropout=config["lora"]["dropout"],
        )
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "lora_B"):
                    module.lora_B.weight.normal_()
            model.visual_context_adapter.weight.add_(0.5)
            model.vlm_proj.weight.sub_(0.25)
        router = PromptConditionedMaskRouter(
            in_channels=2,
            prompt_dim=2048,
            hidden_channels=config["router"]["hidden_channels"],
            depth=config["router"]["depth"],
        )
        model.eval()
        router.eval()
        query = torch.randn(2, 4)
        expected_query = model.wan_model.blocks[0].cross_attn.q(query).detach()
        video = torch.randn(1, 2, 2, 3, 3)
        prompt = torch.randn(1, 2048)
        expected_router = router(video, prompt).detach()
        payload = {
            "format": "pact-omnivideo2-adapters-v2",
            "step": 7,
            "config_sha256": "1" * 64,
            "validated_config": copy.deepcopy(config),
            "base_checkpoint_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
            "special_tokens_sha256": "4" * 64,
            "encoder_contract_sha256": "5" * 64,
            "lora_modules": names,
            "lora_state_dict": lora_state_dict(model),
            "router_state_dict": _state(router),
            "visual_context_adapter_state_dict": _state(
                model.visual_context_adapter
            ),
            "vlm_norm_state_dict": _state(model.vlm_norm),
            "vlm_proj_state_dict": _state(model.vlm_proj),
        }
        return (
            payload,
            base_state,
            query,
            expected_query,
            video,
            prompt,
            expected_router,
        )

    def test_strict_bundle_round_trip(self) -> None:
        (
            payload,
            base_state,
            query,
            expected_query,
            video,
            prompt,
            expected_router,
        ) = self._bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(payload, path)
            restored = _FakeUnified()
            restored.load_state_dict(base_state)
            loaded = load_pact_adapter_bundle(
                restored,
                path,
                expected_base_checkpoint_sha256="2" * 64,
                expected_manifest_sha256="3" * 64,
                expected_special_tokens_sha256="4" * 64,
                expected_encoder_contract_sha256="5" * 64,
            )
        restored.eval()
        loaded.router.eval()
        actual_query = restored.wan_model.blocks[0].cross_attn.q(query).detach()
        actual_router = loaded.router(video, prompt).detach()
        self.assertTrue(torch.equal(actual_query, expected_query))
        self.assertTrue(torch.equal(actual_router, expected_router))
        self.assertEqual(loaded.step, 7)
        self.assertEqual(loaded.lora_modules, tuple(payload["lora_modules"]))
        self.assertEqual(loaded.manifest_sha256, "3" * 64)
        self.assertEqual(loaded.encoder_contract_sha256, "5" * 64)

    def test_wrong_base_and_unknown_fields_fail_before_partial_restore(self) -> None:
        payload, base_state, *_ = self._bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(AdapterCheckpointError, "different base"):
                load_pact_adapter_bundle(
                    _FakeUnified(),
                    path,
                    expected_base_checkpoint_sha256="9" * 64,
                )
            payload["base_weights"] = base_state
            torch.save(payload, path)
            with self.assertRaisesRegex(AdapterCheckpointError, "unknown=.*base_weights"):
                load_pact_adapter_bundle(
                    _FakeUnified(),
                    path,
                    expected_base_checkpoint_sha256="2" * 64,
                )

    def test_bf16_clean_base_restores_conditioning_adapters_as_fp32(self) -> None:
        payload, _base_state, *_ = self._bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(payload, path)
            restored = _FakeUnified().to(dtype=torch.bfloat16)
            loaded = load_pact_adapter_bundle(
                restored,
                path,
                expected_base_checkpoint_sha256="2" * 64,
            )

        for attribute, state_field in (
            ("visual_context_adapter", "visual_context_adapter_state_dict"),
            ("vlm_norm", "vlm_norm_state_dict"),
            ("vlm_proj", "vlm_proj_state_dict"),
        ):
            module = getattr(restored, attribute)
            self.assertTrue(
                all(parameter.dtype == torch.float32 for parameter in module.parameters())
            )
            for key, expected in payload[state_field].items():
                self.assertTrue(torch.equal(module.state_dict()[key], expected))
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in loaded.router.parameters())
        )


if __name__ == "__main__":
    unittest.main()
