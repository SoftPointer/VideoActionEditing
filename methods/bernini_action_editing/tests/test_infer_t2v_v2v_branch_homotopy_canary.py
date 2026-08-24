#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_t2v_v2v_branch_homotopy_canary as canary  # noqa: E402


class T2VV2VBranchHomotopyCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry_path = (
            METHOD_ROOT / "assets/t2v_v2v_branch_homotopy_core4_v1.json"
        )
        cls.registry = json.loads(cls.registry_path.read_text(encoding="utf-8"))

    def test_registry_is_sealed_exact_four_cell_two_wave_reuse(self) -> None:
        rows = [
            canary._registry_cell(self.registry, cell_id=cell)
            for cell in canary.CELL_ORDER
        ]
        self.assertEqual([row["cell_id"] for row in rows], list(canary.CELL_ORDER))
        self.assertEqual([row["seed"] for row in rows], [
            2026081601, 2026081602, 2026081301, 2026081302
        ])
        self.assertEqual(rows[0]["latent_shape"], [1, 16, 21, 60, 62])
        self.assertEqual(rows[1]["latent_shape"], [1, 16, 21, 64, 58])
        self.assertEqual(rows[2]["latent_shape"], [1, 16, 21, 60, 62])
        self.assertEqual(rows[3]["latent_shape"], [1, 16, 21, 68, 54])
        self.assertEqual(self.registry["arm_order"], list(canary.ARM_ORDER))
        self.assertTrue(
            self.registry["population_design"]["fit_and_confirmation_never_aggregated"]
        )
        self.assertFalse(
            self.registry["population_design"]["single_example_conclusion_authorized"]
        )

    def test_any_registry_or_reused_cell_change_fails_closed(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["contract"]["homotopy"]["high_sigma"] = 0.94
        with self.assertRaises(canary.T2VV2VBranchHomotopyCanaryError):
            canary._registry_cell(changed, cell_id=canary.CELL_ORDER[0])

        changed = copy.deepcopy(self.registry)
        changed["cells"][0]["target_action_caption"] += " changed"
        with self.assertRaises(canary.T2VV2VBranchHomotopyCanaryError):
            canary._registry_cell(changed, cell_id=canary.CELL_ORDER[0])

    def test_endpoint_conditions_are_source_absent_or_full_source_only(self) -> None:
        source = object()
        low = canary.conditions_for_arm(canary.ARM_ORDER[0], source_latent=source)
        high = canary.conditions_for_arm(canary.ARM_ORDER[1], source_latent=source)
        mixed = canary.conditions_for_arm(canary.ARM_ORDER[2], source_latent=source)
        self.assertIs(low["multi_video_vae_latents"][0], source)
        self.assertIs(mixed["multi_video_vae_latents"][0], source)
        self.assertIsNone(high["multi_video_vae_latents"])
        for value in (low, high, mixed):
            self.assertIsNone(value["image_vae_latents"])
            self.assertIsNone(value["multi_image_vae_latents"])
        with self.assertRaises(canary.T2VV2VBranchHomotopyCanaryError):
            canary.conditions_for_arm(canary.ARM_ORDER[0], source_latent=None)

    def test_mode_native_prompts_share_body_but_not_task_prefix(self) -> None:
        cleaner = lambda value: "<clean>" + value
        caption = "The dog bends its hind legs, sits, and holds."
        low = canary.build_mode_native_prompt(
            "source-mv2v", caption, prompt_cleaner=cleaner
        )
        high = canary.build_mode_native_prompt(
            "pure-t2v", caption, prompt_cleaner=cleaner
        )
        self.assertIn(caption, low)
        self.assertIn(caption, high)
        self.assertNotEqual(low, high)
        self.assertTrue(low.startswith("You are a helpful assistant for editing."))
        self.assertTrue(
            high.startswith("You are a helpful assistant specialized in text-to-video")
        )

    def test_native_sampling_contracts_are_exact81_exact40(self) -> None:
        for arm in canary.ARM_ORDER:
            value = canary.sampling_contract(arm, seed=17)
            self.assertEqual(value["guidance_mode"], canary.GUIDANCE_BY_ARM[arm])
            self.assertEqual(value["num_frames"], 81)
            self.assertEqual(value["num_inference_steps"], 40)
            self.assertEqual(value["flow_shift"], 5.0)
            self.assertEqual(value["omega_txt"], 4.0)
            self.assertEqual(value["eta"], 0.5)
            self.assertEqual(value["norm_threshold"], (50.0, 50.0))
            self.assertEqual(value["momentum"], 0.0)
        self.assertEqual(
            canary.sampling_contract(canary.ARM_ORDER[0], seed=17)["omega_img"],
            0.0,
        )
        self.assertEqual(
            canary.sampling_contract(canary.ARM_ORDER[1], seed=17)["omega_img"],
            4.5,
        )

    def test_exact40_trace_validator_locks_regions_and_counts(self) -> None:
        schedule = canary.branch_base.pinned_exact40_schedule_receipt()
        rows = []
        for index, (timestep, sigma) in enumerate(
            zip(schedule["timesteps"], canary.NATIVE_UNIPC40_SIGMAS)
        ):
            endpoint = (
                "high_pure_t2v_apg"
                if index < 9
                else "transition" if index < 26 else "low_source_v2v_apg"
            )
            rows.append({
                "step_index": index,
                "timestep": float(timestep),
                "sigma": sigma,
                "endpoint": endpoint,
                "high_pure_t2v_weight": 1.0 if index < 9 else 0.5 if index < 26 else 0.0,
                "transformer_forwards": 4,
                "low_source_v2v_forwards": 2,
                "high_pure_t2v_forwards": 2,
                "original_scheduler_calls": 1,
                "patch_call_count": 2,
                "patch_source_ids": [1.0, 0.0],
                "low_stock_apg_exact_parity": True,
                "vendor_apg_function": "bernini.models.wan_diffusion.normalized_guidance",
                "freeze_safe_no_grad_outputs": True,
            })
        trace = {
            "steps": 40,
            "transformer_forwards": 160,
            "low_source_v2v_forwards": 80,
            "high_pure_t2v_forwards": 80,
            "patch_vae_latent_calls": 80,
            "original_scheduler_calls": 40,
            "low_stock_apg_exact_parity_all_steps": True,
            "smoothstep_sigma_low": 0.75,
            "smoothstep_sigma_high": 0.95,
            "exact40_shift5_schedule_digest": canary.SCHEDULE_SHA256,
            "scheduler_mutation_surface": "model_output_argument_only",
            "runtime_source_identity_enforcement": "external_canary_required",
            "branch_apg": {
                "function": "bernini.models.wan_diffusion.normalized_guidance",
                "one_condition_per_branch": True,
                "omega_text": 4.0,
                "eta": 0.5,
                "norm_threshold": 50.0,
                "independent_momentum": 0.0,
            },
            "optimizer_created": False,
            "parameters_updated": False,
            "trace": rows,
        }
        self.assertEqual(
            canary.validate_homotopy_runtime_trace(trace)["digest"],
            canary.SCHEDULE_SHA256,
        )
        changed = copy.deepcopy(trace)
        changed["trace"][9]["endpoint"] = "high_pure_t2v_apg"
        with self.assertRaises(canary.T2VV2VBranchHomotopyCanaryError):
            canary.validate_homotopy_runtime_trace(changed)

    def test_runner_binds_real_vendor_hash_and_native_high_apg_operator(self) -> None:
        source = Path(canary.__file__).read_text(encoding="utf-8")
        self.assertIn("sampler_contract.validate_runtime_source_identity", source)
        self.assertIn("wan_diffusion_path=Path(wan_diffusion.__file__).resolve()", source)
        self.assertIn("observed_wan_diffusion_sha256=wan_source_sha", source)
        self.assertIn(
            "arm_source_latent = None if arm == ARM_ORDER[1] else source_latent",
            source,
        )
        self.assertIn('"pure_t2v_visual_conditions_all_none": True', source)
        self.assertIn(
            '"homotopy_high_same_state_target_only_t2v_apg_contract_certified": True',
            source,
        )
        self.assertIn(
            '"standalone_and_homotopy_trajectory_equality_claim_authorized": False',
            source,
        )
        self.assertIn('"training_performed": False', source)
        self.assertIn('"optimizer_created": False', source)
        self.assertIn('"parameter_update": False', source)
        self.assertNotIn("optimizer.step", source)


if __name__ == "__main__":
    unittest.main()
