from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = METHOD_ROOT / "infer_oasis_phase_a_noise_bank.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch

    import infer_oasis_phase_a_noise_bank as runtime

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    runtime = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class OASISNoiseBankStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNTIME_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_wrapper_mutates_only_the_randn_tensor_module_global(self) -> None:
        module_attributes = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "setattr" or len(node.args) < 2:
                continue
            owner, attribute = node.args[:2]
            if isinstance(owner, ast.Name) and owner.id == "wan_diffusion_module":
                self.assertIsInstance(attribute, ast.Constant)
                module_attributes.append(attribute.value)
        self.assertEqual(module_attributes, ["randn_tensor", "randn_tensor"])
        self.assertNotIn('setattr(wan_diffusion_module, "sample_one_step"', self.source)
        self.assertNotIn('setattr(wan_diffusion_module, "guidance"', self.source)

    def test_guidance_is_fixed_native_and_explicitly_outside_this_ablation(self) -> None:
        for token in (
            'GUIDANCE_POLICY = "fixed_native_rv2v_no_ablation"',
            '"sample_one_step_replaced": False',
            '"native_cfg_or_apg_replaced": False',
            '"native_scheduler_replaced": False',
            '"future_guidance_ablation_composes_outside_this_runner": True',
        ):
            self.assertIn(token, self.source)

    def test_candidate_bank_is_source_only_and_non_authoritative(self) -> None:
        for token in (
            '"full_video_latent_consumed_by_carrier": False',
            '"target_mask_flow_pose_track_trajectory_consumed": False',
            '"proposal_media_latent_or_motion_donor_consumed": False',
            '"legacy_pair_v5_native_rollout_schema_compatible": False',
            '"external_action_scorer_consumed": False',
            '"action_source_scoring_performed": False',
            '"endpoint_selection_performed": False',
            '"optimizer_or_training_authorized": False',
            '"training_performed": False',
            '"scientific_action_editing_success_claim": False',
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("optimizer.step", self.source)
        self.assertNotIn("backward()", self.source)
        self.assertNotIn("mace_candidate_action_energy", self.source)
        self.assertNotIn("load_dedicated_scalar_calibration_evidence", self.source)

    def test_train_inference_operator_parity_is_fail_closed(self) -> None:
        self.assertIn(
            "NOISE_OPERATOR_CALLABLE = oasis_core.NOISE_OPERATOR_CALLABLE",
            self.source,
        )
        self.assertIn('"operator_runtime_binding": {', self.source)
        self.assertIn('"same_callable_required_for_any_future_training": True', self.source)
        self.assertIn('"alternate_training_noise_builder_authorized": False', self.source)
        self.assertIn('"training_integration_executed": False', self.source)


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is required for OASIS noise tests")
class OASISNoiseBankTensorTests(unittest.TestCase):
    shape = (1, 16, 21, 8, 8)
    seed = 20260808
    sample_digest = "a" * 64

    def setUp(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(7001)
        self.frames = tuple(
            torch.randn(
                (1, 16, 1, 8, 8),
                generator=generator,
                dtype=torch.float32,
            )
            .contiguous()
            .clone()
            for _ in range(4)
        )

    @staticmethod
    def canonical_randn_tensor(
        shape,
        *,
        generator=None,
        device=None,
        dtype=None,
        **_kwargs,
    ):
        return torch.randn(
            tuple(shape), generator=generator, device=device, dtype=dtype
        ).contiguous()

    def run_arm(self, arm: str, *, frames=None):
        module = SimpleNamespace(randn_tensor=self.canonical_randn_tensor)

        def sample():
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            return module.randn_tensor(
                self.shape,
                generator=generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        result, capture = runtime._sample_with_oasis_noise_arm(
            sample_fn=sample,
            wan_diffusion_module=module,
            noise_arm=arm,
            independent_frame_latents_cpu=self.frames if frames is None else frames,
            expected_shape=self.shape,
            expected_device=torch.device("cpu"),
            expected_seed=self.seed,
            carrier_seed=runtime.carrier_seed_for(
                sample_digest=self.sample_digest, seed=self.seed
            ),
            canonical_randn_tensor=self.canonical_randn_tensor,
        )
        self.assertIs(module.randn_tensor, self.canonical_randn_tensor)
        return result, capture

    def test_rho_zero_forwards_exact_original_return_and_active_arms_share_parent(self) -> None:
        native_result, native = self.run_arm("official_gaussian")
        active_result, active = self.run_arm("source_appearance_set_rho005")
        self.assertTrue(native.original_return_object_forwarded)
        self.assertFalse(native.external_initial_noise_injection)
        self.assertEqual(
            native.baseline_raw_value_sha256, native.injected_raw_value_sha256
        )
        self.assertTrue(torch.equal(native_result, native.injected_tensor))
        self.assertFalse(active.original_return_object_forwarded)
        self.assertTrue(active.external_initial_noise_injection)
        self.assertNotEqual(
            active.baseline_raw_value_sha256, active.injected_raw_value_sha256
        )
        self.assertEqual(
            active.baseline_raw_value_sha256, native.baseline_raw_value_sha256
        )
        self.assertTrue(torch.equal(active_result, active.injected_tensor))
        self.assertEqual(native.original_randn_call_count, 1)
        self.assertEqual(active.original_randn_call_count, 1)

    def test_frame_set_reversal_is_bit_exact_and_digest_invariant(self) -> None:
        first_result, first = self.run_arm("source_appearance_set_rho010")
        reversed_result, reversed_capture = self.run_arm(
            "source_appearance_set_rho010", frames=tuple(reversed(self.frames))
        )
        self.assertTrue(torch.equal(first_result, reversed_result))
        self.assertEqual(
            first.injected_raw_value_sha256,
            reversed_capture.injected_raw_value_sha256,
        )
        self.assertEqual(
            first.source_frame_set_digest,
            reversed_capture.source_frame_set_digest,
        )
        diagnostics = first.operator_receipt["diagnostics"]
        self.assertEqual(
            first.operator_receipt["schema_version"],
            "bernini-motion-null-appearance-noise-v2",
        )
        self.assertFalse(
            first.operator_receipt["operator_self_registers_sampler_hook"]
        )
        self.assertFalse(first.operator_receipt["operator_self_registers_launcher"])
        self.assertFalse(diagnostics["source_temporal_indices_consumed"])
        self.assertFalse(diagnostics["source_temporal_phase_consumed"])
        self.assertTrue(diagnostics["carrier_strict_temporal_dc"])

    def test_active_rho_arms_share_one_descriptor_and_carrier(self) -> None:
        _rho005_result, rho005 = self.run_arm("source_appearance_set_rho005")
        _rho010_result, rho010 = self.run_arm("source_appearance_set_rho010")
        self.assertEqual(
            rho005.operator_receipt["diagnostics"]["descriptor_sha256"],
            rho010.operator_receipt["diagnostics"]["descriptor_sha256"],
        )
        self.assertEqual(
            rho005.operator_receipt["diagnostics"]["carrier_sha256"],
            rho010.operator_receipt["diagnostics"]["carrier_sha256"],
        )

    def test_full_video_latents_cannot_masquerade_as_the_source_frame_set(self) -> None:
        full = torch.randn(self.shape, dtype=torch.float32).contiguous()
        with self.assertRaisesRegex(runtime.OASISNoiseBankError, "temporal latent"):
            runtime.source_frame_set_digest(tuple(full.clone() for _ in range(4)))

    def test_wrapper_restores_original_symbol_when_sampling_raises(self) -> None:
        module = SimpleNamespace(randn_tensor=self.canonical_randn_tensor)

        def failing_sample():
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            module.randn_tensor(
                self.shape,
                generator=generator,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            raise RuntimeError("sentinel")

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            runtime._sample_with_oasis_noise_arm(
                sample_fn=failing_sample,
                wan_diffusion_module=module,
                noise_arm="official_gaussian",
                independent_frame_latents_cpu=self.frames,
                expected_shape=self.shape,
                expected_device=torch.device("cpu"),
                expected_seed=self.seed,
                carrier_seed=7,
                canonical_randn_tensor=self.canonical_randn_tensor,
            )
        self.assertIs(module.randn_tensor, self.canonical_randn_tensor)


if __name__ == "__main__":
    unittest.main()
