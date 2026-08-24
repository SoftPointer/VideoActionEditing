#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = METHOD_ROOT / "run_elal3_bernini_integration_probe_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    import run_elal3_bernini_integration_probe_v1 as runner

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    runner = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class ELAL3RealProbeStaticTests(unittest.TestCase):
    def test_runner_parses_and_contains_nontraining_real_model_contract(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for fragment in (
            "WanTransformer3DModel.from_pretrained(",
            'subfolder="transformer"',
            'variant="full"',
            "attention_width=64",
            '"source_prefix_immediate_pre_post_bit_exact_all30": True',
            '"all30_injections_have_finite_nonzero_gradient": True',
            '"optimizer_constructed": False',
            '"optimizer_steps": 0',
            '"training_authorized": False',
            '"synthetic_renderer_used": False',
            "os.O_EXCL",
        ):
            self.assertIn(fragment, source)


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required")
class ELAL3RealProbeFunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(11)

    def _fake_model(
        self,
        *,
        hidden_size: int = 16,
        block_count: int = 30,
        mixed_ingress_bfloat16: bool = False,
    ):
        testcase = self

        class DTypeOnlyPatchEmbedding(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.zeros(
                        (hidden_size, 16, 1, 2, 2), dtype=torch.bfloat16
                    )
                )
                self.bias = torch.nn.Parameter(
                    torch.zeros((hidden_size,), dtype=torch.bfloat16)
                )

        class FakeTextEmbedder(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.linear_1 = torch.nn.Linear(
                    4096, hidden_size, bias=True, dtype=dtype
                )

        class FakeConditionEmbedder(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.text_embedder = FakeTextEmbedder(dtype)

        class FakeRealBlock(torch.nn.Module):
            def __init__(self, index: int) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(
                    hidden_size, hidden_size, bias=False
                )
                with torch.no_grad():
                    self.projection.weight.zero_()
                    self.projection.weight.diagonal().fill_(
                        0.003 + index * 1.0e-5
                    )

            def forward(self, hidden_states, *_args, **_kwargs):
                # Global token mixing makes the test exercise the distinction
                # between immediate target-only writes and later source response.
                mixed = hidden_states.mean(dim=1, keepdim=True)
                return hidden_states + torch.tanh(
                    self.projection(hidden_states + mixed)
                )

        class FakeRealTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_size = hidden_size
                if mixed_ingress_bfloat16:
                    # Deliberately first: the regression must not infer ingress
                    # dtype from this retained-FP32 parameter.
                    self.retained_fp32_first = torch.nn.Parameter(
                        torch.tensor(1.0, dtype=torch.float32)
                    )
                    self.patch_embedding = DTypeOnlyPatchEmbedding()
                    ingress_dtype = torch.bfloat16
                else:
                    self.patch_embedding = torch.nn.Conv3d(
                        16,
                        hidden_size,
                        kernel_size=(1, 2, 2),
                        stride=(1, 2, 2),
                        bias=False,
                    )
                    ingress_dtype = torch.float32
                self.condition_embedder = FakeConditionEmbedder(
                    ingress_dtype
                )
                self.blocks = torch.nn.ModuleList(
                    FakeRealBlock(index) for index in range(block_count)
                )
                self.proj_out = torch.nn.Linear(hidden_size, 64, bias=False)
                self.gradient_checkpointing = False

            def patch_vae_latent(self, hidden_states, source_id=None):
                testcase.assertEqual(
                    hidden_states.dtype, self.patch_embedding.weight.dtype
                )
                if mixed_ingress_bfloat16:
                    batch, _channels, phases, height, width = hidden_states.shape
                    patch_seed = (
                        hidden_states.float()
                        .reshape(batch, 16, phases, height // 2, 2, width // 2, 2)
                        .mean(dim=(1, 4, 6))
                        .reshape(batch, phases * (height // 2) * (width // 2), 1)
                    )
                    tokens = patch_seed.expand(
                        batch, int(patch_seed.shape[1]), hidden_size
                    ).to(self.patch_embedding.weight.dtype).contiguous()
                else:
                    tokens = (
                        self.patch_embedding(hidden_states)
                        .flatten(2)
                        .transpose(1, 2)
                    )
                count = int(tokens.shape[1])
                angle = float(source_id)
                rotary = torch.polar(
                    torch.ones((1, 1, count, 64), device=tokens.device),
                    torch.full(
                        (1, 1, count, 64), angle, device=tokens.device
                    ),
                ).to(torch.complex128)
                return tokens, rotary

            def forward(
                self,
                hidden_states,
                timestep,
                *,
                encoder_hidden_states,
                rotary_emb,
                batch_image_vae_seqlen,
                text_features_length,
                return_dict=False,
            ):
                testcase.assertEqual(tuple(timestep.shape), (1,))
                testcase.assertEqual(timestep.dtype, torch.float32)
                testcase.assertEqual(batch_image_vae_seqlen, [42])
                testcase.assertEqual(text_features_length, [4])
                testcase.assertEqual(int(rotary_emb.shape[2]), 42)
                testcase.assertEqual(
                    tuple(encoder_hidden_states.shape), (1, 4, 4096)
                )
                testcase.assertEqual(
                    encoder_hidden_states.dtype,
                    self.condition_embedder.text_embedder.linear_1.weight.dtype,
                )
                testcase.assertEqual(
                    hidden_states.dtype, self.patch_embedding.weight.dtype
                )
                if mixed_ingress_bfloat16:
                    hidden_states = hidden_states.float()
                for block in self.blocks:
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        timestep,
                        rotary_emb,
                    )
                value = self.proj_out(hidden_states)
                return (value,) if not return_dict else value

        return FakeRealTransformer()

    def test_cpu_fake_executes_same_30_block_route_and_vjp_contract(self) -> None:
        model = self._fake_model()
        result = runner.run_loaded_transformer_probe(
            model,
            device=torch.device("cpu"),
            seed=20260817,
            hidden_size=16,
            test_only=True,
        )
        self.assertTrue(result["engineering_gate_pass"])
        self.assertEqual(result["registered_arm"], "full-w64")
        self.assertEqual(result["route"]["condition_tokens"], 21)
        self.assertEqual(result["route"]["target_tokens"], 21)
        audit = result["model_route_audit"]
        self.assertEqual(audit["hook_installation_count"], 30)
        self.assertEqual(audit["forward_hook_record_count"], 30)
        self.assertTrue(audit["source_prefix_immediate_pre_post_bit_exact_all30"])
        self.assertTrue(result["response"]["target_response_nonzero"])
        backward = result["backward"]
        self.assertTrue(backward["all30_injections_have_finite_nonzero_gradient"])
        self.assertEqual(len(backward["per_block"]), 30)
        self.assertTrue(
            all(row["has_nonzero_gradient"] for row in backward["per_block"])
        )
        scope = result["parameter_scope"]
        self.assertTrue(scope["frozen_base_parameter_version_counters_unchanged"])
        self.assertTrue(scope["elal3_parameter_bytes_unchanged"])
        self.assertFalse(hasattr(model, "elal3_c0_v1"))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_wrong_block_count_fails_before_installation(self) -> None:
        model = self._fake_model(block_count=29)
        with self.assertRaisesRegex(
            runner.ELAL3BerniniIntegrationProbeError, "30 distinct"
        ):
            runner.run_loaded_transformer_probe(
                model,
                device=torch.device("cpu"),
                seed=17,
                hidden_size=16,
                test_only=True,
            )

    def test_mixed_parameter_order_binds_bfloat16_ingress_weights(self) -> None:
        model = self._fake_model(mixed_ingress_bfloat16=True)
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertEqual(model.patch_embedding.weight.dtype, torch.bfloat16)
        self.assertEqual(
            model.condition_embedder.text_embedder.linear_1.weight.dtype,
            torch.bfloat16,
        )
        result = runner.run_loaded_transformer_probe(
            model,
            device=torch.device("cpu"),
            seed=20260818,
            hidden_size=16,
            test_only=True,
        )
        native = result["native_input"]
        contract = native["input_dtype_contract"]
        self.assertEqual(native["hidden_dtype"], "torch.bfloat16")
        self.assertEqual(native["text_dtype"], "torch.bfloat16")
        self.assertEqual(native["timestep_dtype"], "torch.float32")
        self.assertEqual(contract["first_model_parameter_dtype"], "torch.float32")
        self.assertEqual(
            contract["patch_embedding_weight_dtype"], "torch.bfloat16"
        )
        self.assertEqual(
            contract["text_linear_1_weight_dtype"], "torch.bfloat16"
        )
        self.assertTrue(result["engineering_gate_pass"])
        self.assertTrue(
            result["backward"]["all30_injections_have_finite_nonzero_gradient"]
        )

    def test_create_only_receipt_is_sealed_and_cannot_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "receipt.json"
            receipt = runner.seal(
                {
                    "schema_version": runner.RECEIPT_SCHEMA_VERSION,
                    "training_authorized": False,
                }
            )
            runner.write_create_only_json(output, receipt)
            parsed = json.loads(output.read_text(encoding="ascii"))
            digest = parsed.pop("receipt_digest")
            self.assertEqual(digest, runner.object_sha256(parsed))
            with self.assertRaisesRegex(
                runner.ELAL3BerniniIntegrationProbeError, "create-only"
            ):
                runner.write_create_only_json(output, receipt)

    def test_relative_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            runner.ELAL3BerniniIntegrationProbeError, "absolute"
        ):
            runner.resolve_create_only_output("receipt.json")

    def test_blocker_chain_keeps_underlying_runtime_cause(self) -> None:
        try:
            try:
                raise ValueError("native ABI detail")
            except ValueError as cause:
                raise runner.ELAL3BerniniIntegrationProbeError(
                    "official forward failed"
                ) from cause
        except runner.ELAL3BerniniIntegrationProbeError as error:
            rendered = runner.exception_chain(error)
        self.assertIn("official forward failed", rendered)
        self.assertIn("ValueError: native ABI detail", rendered)


if __name__ == "__main__":
    unittest.main()
