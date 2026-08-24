from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_self_imagined_phase_noise_canary as runtime  # noqa: E402


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class NativeSelfImaginedPhaseNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(runtime.__file__).resolve()
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_arm_registry_is_fixed_and_controls_are_matched(self) -> None:
        self.assertEqual(
            runtime.ARM_ORDER,
            (
                "matched-gaussian",
                "action-phi",
                "noop-phi",
                "reverse-phi",
                "action-phi-source-dc-rho02",
            ),
        )
        self.assertEqual(
            runtime.PROPOSAL_BRANCHES,
            ("full_action", "noop", "reverse_action"),
        )
        specs = {spec.arm_id: spec for spec in runtime.arm_plan()}
        self.assertIsNone(specs["matched-gaussian"].proposal_branch)
        self.assertEqual(specs["action-phi"].proposal_branch, "full_action")
        self.assertEqual(specs["noop-phi"].proposal_branch, "noop")
        self.assertEqual(specs["reverse-phi"].proposal_branch, "reverse_action")
        self.assertEqual(
            specs["action-phi-source-dc-rho02"].source_rho,
            runtime.SOURCE_DC_RHO,
        )
        self.assertEqual(runtime.SPATIAL_RADIUS, 3)
        self.assertEqual(runtime.GAMMA, 30.0)

    def test_native_condition_registry_is_exact_r2v5_or_rv2v4(self) -> None:
        r2v = runtime.condition_contract("r2v5")
        self.assertEqual(r2v["native_arm"], "r2v")
        self.assertEqual(r2v["source_reference_count"], 5)
        self.assertEqual(r2v["full_source_video_count"], 0)
        self.assertEqual(r2v["reference_indices"], [0, 20, 40, 60, 80])
        rv2v = runtime.condition_contract("rv2v4")
        self.assertEqual(rv2v["native_arm"], "rv2v")
        self.assertEqual(rv2v["source_reference_count"], 4)
        self.assertEqual(rv2v["full_source_video_count"], 1)
        self.assertEqual(rv2v["reference_indices"], [0, 27, 53, 80])
        with self.assertRaises(runtime.PhaseNoiseCanaryError):
            runtime.condition_contract("v2v")

    def test_cli_has_no_hidden_supervision_or_external_noise_inputs(self) -> None:
        parser = runtime._build_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--target", "--target-video", "--target-latent", "--mask",
            "--flow", "--pose", "--track", "--trajectory", "--donor-video",
            "--proposal-video", "--initial-noise", "--first-frame",
            "--optimizer", "--learning-rate", "--lora",
        ):
            self.assertNotIn(forbidden, options)
        for required in (
            "--factor-manifest", "--factor-bank-receipt", "--bank-output-root",
            "--execution-group", "--condition-mode", "--arms",
        ):
            self.assertIn(required, options)

    @unittest.skipUnless(TORCH_AVAILABLE, "local interpreter has no PyTorch; run on AUH vace")
    def test_injector_calls_native_rng_first_returns_injected_and_restores(self) -> None:
        import torch

        events: list[str] = []

        def canonical(shape, *, generator, device, dtype):
            events.append("native-rng")
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)

        module = SimpleNamespace(randn_tensor=canonical)
        source = torch.zeros(runtime.LATENT_SHAPE, dtype=torch.float32).contiguous()
        proposal = torch.ones(runtime.LATENT_SHAPE, dtype=torch.float32).contiguous()
        returned: dict[str, object] = {}

        def fake_operator(gaussian, action_reference, source_reference, **kwargs):
            events.append("cpu-fp64-operator")
            self.assertEqual(gaussian.device.type, "cpu")
            self.assertEqual(gaussian.dtype, torch.float64)
            self.assertEqual(action_reference.dtype, torch.float64)
            self.assertEqual(source_reference.dtype, torch.float64)
            self.assertEqual(kwargs["spatial_radius"], 3)
            self.assertEqual(kwargs["gamma"], 30.0)
            self.assertEqual(kwargs["source_rho"], 0.0)
            return SimpleNamespace(
                initial_noise=(gaussian + 1.0).contiguous(),
                receipt={"operator": "fake-phi", "scientific_claim_authorized": False},
            )

        def sample_fn():
            generator = torch.Generator(device="cpu").manual_seed(runtime.DEFAULT_SEED)
            returned["value"] = module.randn_tensor(
                runtime.LATENT_SHAPE,
                generator=generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            return "sample-result"

        with mock.patch.object(
            runtime.phase_noise, "build_factorized_phase_noise", side_effect=fake_operator
        ):
            result, capture = runtime._sample_with_phase_noise_injection(
                sample_fn=sample_fn,
                wan_diffusion_module=module,
                arm_spec=runtime.spec_for_arm("action-phi"),
                proposal_reference_cpu=proposal,
                source_normalized_latent_cpu=source,
                expected_shape=runtime.LATENT_SHAPE,
                expected_device=torch.device("cpu"),
                expected_seed=runtime.DEFAULT_SEED,
                canonical_randn_tensor=canonical,
            )
        self.assertEqual(events, ["native-rng", "cpu-fp64-operator"])
        self.assertEqual(result, "sample-result")
        self.assertIs(module.randn_tensor, canonical)
        self.assertTrue(capture.injection_performed)
        self.assertFalse(capture.original_return_object_forwarded)
        self.assertNotEqual(
            capture.baseline_raw_storage_sha256,
            capture.injected_raw_storage_sha256,
        )
        self.assertTrue(torch.equal(returned["value"], capture.injected_tensor))

    @unittest.skipUnless(TORCH_AVAILABLE, "local interpreter has no PyTorch; run on AUH vace")
    def test_matched_arm_forwards_exact_original_tensor_object(self) -> None:
        import torch

        baseline: dict[str, object] = {}

        def canonical(shape, *, generator, device, dtype):
            value = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            baseline["value"] = value
            return value

        module = SimpleNamespace(randn_tensor=canonical)
        source = torch.zeros(runtime.LATENT_SHAPE, dtype=torch.float32).contiguous()
        observed: dict[str, object] = {}

        def sample_fn():
            generator = torch.Generator(device="cpu").manual_seed(runtime.DEFAULT_SEED)
            observed["value"] = module.randn_tensor(
                runtime.LATENT_SHAPE, generator=generator,
                device=torch.device("cpu"), dtype=torch.float32,
            )
            return observed["value"]

        result, capture = runtime._sample_with_phase_noise_injection(
            sample_fn=sample_fn,
            wan_diffusion_module=module,
            arm_spec=runtime.spec_for_arm("matched-gaussian"),
            proposal_reference_cpu=None,
            source_normalized_latent_cpu=source,
            expected_shape=runtime.LATENT_SHAPE,
            expected_device=torch.device("cpu"),
            expected_seed=runtime.DEFAULT_SEED,
            canonical_randn_tensor=canonical,
        )
        self.assertIs(result, baseline["value"])
        self.assertIs(observed["value"], baseline["value"])
        self.assertIs(module.randn_tensor, canonical)
        self.assertFalse(capture.injection_performed)
        self.assertTrue(capture.original_return_object_forwarded)
        self.assertEqual(
            capture.baseline_raw_storage_sha256,
            capture.injected_raw_storage_sha256,
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "local interpreter has no PyTorch; run on AUH vace")
    def test_injector_restores_module_global_on_sampler_failure(self) -> None:
        import torch

        def canonical(shape, *, generator, device, dtype):
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)

        module = SimpleNamespace(randn_tensor=canonical)
        source = torch.zeros(runtime.LATENT_SHAPE, dtype=torch.float32).contiguous()

        def failing_sample():
            generator = torch.Generator(device="cpu").manual_seed(runtime.DEFAULT_SEED)
            module.randn_tensor(
                runtime.LATENT_SHAPE, generator=generator,
                device=torch.device("cpu"), dtype=torch.float32,
            )
            raise RuntimeError("synthetic sampler failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic sampler failure"):
            runtime._sample_with_phase_noise_injection(
                sample_fn=failing_sample,
                wan_diffusion_module=module,
                arm_spec=runtime.spec_for_arm("matched-gaussian"),
                proposal_reference_cpu=None,
                source_normalized_latent_cpu=source,
                expected_shape=runtime.LATENT_SHAPE,
                expected_device=torch.device("cpu"),
                expected_seed=runtime.DEFAULT_SEED,
                canonical_randn_tensor=canonical,
            )
        self.assertIs(module.randn_tensor, canonical)

    def test_runtime_only_loads_registered_predecode_tensor_paths(self) -> None:
        main_node = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        main_source = ast.get_source_segment(self.source, main_node)
        self.assertIn("load_registered_clean_donor", main_source)
        self.assertNotIn('row["video_path"]', main_source)
        self.assertNotIn("proposal_video_path", main_source)
        self.assertNotIn("_decode_exact_video", main_source)
        self.assertIn('"proposal_mp4_consumed": False', main_source)
        self.assertIn('"only_predecode_fp32_normalized_clean_latents_consumed": True', main_source)

    def test_output_transaction_and_claim_gates_are_explicit(self) -> None:
        for token in (
            "_output_staging_directory",
            "_commit_output_transaction",
            "_rebase_artifact_paths",
            'receipt["receipt_digest"] = donor_runtime.object_sha256(receipt)',
            '"wrapper_is_injector_not_observer": True',
            '"training_performed": False',
            '"scientific_claim_authorized": False',
            '"paired_edit_target": False',
        ):
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
