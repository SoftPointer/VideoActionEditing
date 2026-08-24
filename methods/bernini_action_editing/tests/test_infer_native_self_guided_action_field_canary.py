#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_self_guided_action_field_canary as canary  # noqa: E402


class ActionFieldCanaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry_path = (
            METHOD_ROOT / "assets/self_guided_action_field_core2_v1.json"
        )
        cls.registry = json.loads(cls.registry_path.read_text(encoding="utf-8"))

    def test_registry_has_exact_two_predeclared_cells(self) -> None:
        dog = canary._registry_cell(self.registry, cell_id="dog")
        human = canary._registry_cell(self.registry, cell_id="human")
        self.assertEqual(dog["latent_shape"], [1, 16, 21, 60, 62])
        self.assertEqual(human["latent_shape"], [1, 16, 21, 64, 58])
        self.assertNotEqual(dog["seed"], human["seed"])
        self.assertEqual(
            self.registry["arm_scales"],
            {"native-rv2v": None, "action-field-075": 0.75, "action-field-150": 1.5},
        )

    def test_caption_bytes_are_sealed(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["cells"][0]["source_action_caption"] += " changed"
        with self.assertRaises(canary.ActionFieldCanaryError):
            canary._registry_cell(changed, cell_id="dog")

    def test_runtime_has_no_generated_owner_or_source_noise_input(self) -> None:
        source = Path(canary.__file__).read_text(encoding="utf-8")
        self.assertIn('"generated_owner_media": False', source)
        self.assertIn('"source_rich_initial_noise": False', source)
        self.assertIn('"mask_track_pose_flow": False', source)
        self.assertIn(
            "T2V-APG[target action] - T2V-APG[source action]", source
        )
        self.assertIn("same_target_only_negative_prompt", source)
        self.assertIn("target_latent_shape=latent_shape", source)
        self.assertIn("expected_condition_prefix_tokens", source)
        self.assertNotIn("owner_clean", source)
        self.assertNotIn("target_video_latent", source)

    def test_exact81_exact40_and_three_fixed_arms(self) -> None:
        self.assertEqual(canary.FRAME_COUNT, 81)
        self.assertEqual(canary.LATENT_PHASES, 21)
        self.assertEqual(canary.NUM_INFERENCE_STEPS, 40)
        self.assertEqual(
            canary.ARM_SCALES,
            (("native-rv2v", None), ("action-field-075", 0.75), ("action-field-150", 1.5)),
        )

    def test_vr2v_conditions_use_two_forward_v2v_apg_sampler(self) -> None:
        native_contract = canary.native.native_sampling_contract(
            "rv2v", steps=canary.NUM_INFERENCE_STEPS, seed=17
        )
        self.assertEqual(native_contract["guidance_mode"], "rv2v")
        contract = canary._action_field_sampling_contract(
            steps=canary.NUM_INFERENCE_STEPS, seed=17
        )
        self.assertEqual(contract["guidance_mode"], "v2v_apg")
        self.assertEqual(canary.NATIVE_GUIDANCE_MODE, "v2v_apg")
        self.assertEqual(contract["seed"], 17)
        self.assertEqual(contract["num_inference_steps"], canary.NUM_INFERENCE_STEPS)
        changed = copy.deepcopy(self.registry)
        changed["contract"]["native_guidance_mode"] = "rv2v"
        with self.assertRaises(canary.ActionFieldCanaryError):
            canary._registry_cell(changed, cell_id="dog")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
class StrongFreezeCertificateTests(unittest.TestCase):
    def test_hashes_parameter_and_buffer_values(self) -> None:
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
                self.register_buffer("counter", torch.tensor([2.0, 4.0]))
                self.register_buffer("scalar_counter", torch.tensor(7.0))

        model = Tiny().eval().requires_grad_(False)
        baseline = canary._strong_model_freeze_certificate(model)
        repeated = canary._strong_model_freeze_certificate(model)
        self.assertEqual(baseline, repeated)
        self.assertTrue(baseline["exact_parameter_and_buffer_bytes_hashed"])
        self.assertEqual(baseline["buffer_tensor_count"], 2)
        with torch.no_grad():
            model.counter[0].add_(1.0)
        changed_buffer = canary._strong_model_freeze_certificate(model)
        self.assertNotEqual(
            baseline["state_content_sha256"], changed_buffer["state_content_sha256"]
        )
        with torch.no_grad():
            model.counter[0].sub_(1.0)
            model.weight[0, 0].add_(1.0)
        changed_parameter = canary._strong_model_freeze_certificate(model)
        self.assertNotEqual(
            baseline["state_content_sha256"], changed_parameter["state_content_sha256"]
        )

    def test_trainable_or_adapter_state_is_rejected(self) -> None:
        import torch

        model = torch.nn.Linear(2, 2).eval()
        with self.assertRaises(canary.ActionFieldCanaryError):
            canary._strong_model_freeze_certificate(model)

        class Adapter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lora_probe = torch.nn.Identity()

        adapter = Adapter().eval().requires_grad_(False)
        with self.assertRaises(canary.ActionFieldCanaryError):
            canary._strong_model_freeze_certificate(adapter)


if __name__ == "__main__":
    unittest.main()
