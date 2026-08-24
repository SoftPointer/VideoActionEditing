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

import infer_native_branch_homotopy_canary as canary  # noqa: E402


class NativeBranchHomotopyCanaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry_path = (
            METHOD_ROOT / "assets/native_branch_homotopy_core4_v1.json"
        )
        cls.registry = json.loads(cls.registry_path.read_text(encoding="utf-8"))

    def test_registry_is_exact_four_cell_two_wave_predeclared_core(self) -> None:
        dog = canary._registry_cell(self.registry, cell_id="fit-dog-7b88")
        human = canary._registry_cell(self.registry, cell_id="fit-human-a35b")
        confirmation_dog = canary._registry_cell(
            self.registry, cell_id="confirmation-dog-841b"
        )
        confirmation_human = canary._registry_cell(
            self.registry, cell_id="confirmation-human-a66e"
        )
        self.assertEqual(dog["latent_shape"], [1, 16, 21, 60, 62])
        self.assertEqual(human["latent_shape"], [1, 16, 21, 64, 58])
        self.assertEqual(confirmation_dog["latent_shape"], [1, 16, 21, 60, 62])
        self.assertEqual(confirmation_human["latent_shape"], [1, 16, 21, 68, 54])
        self.assertNotEqual(dog["seed"], human["seed"])
        self.assertEqual(
            [row["cell_id"] for row in self.registry["cells"]],
            list(canary.CELL_ORDER),
        )
        self.assertEqual(self.registry["arm_order"], list(canary.ARM_ORDER))
        self.assertEqual(
            self.registry["schema_version"],
            "bernini-native-branch-homotopy-core4-v1",
        )
        population = self.registry["population_design"]
        self.assertEqual(population["wave_order"], list(canary.WAVE_ORDER))
        self.assertTrue(population["fit_and_confirmation_never_aggregated"])
        self.assertFalse(population["single_example_conclusion_authorized"])
        seeds = [row["seed"] for row in self.registry["cells"]]
        self.assertEqual(len(seeds), len(set(seeds)))
        for row in (confirmation_dog, confirmation_human):
            evidence = row["seed_collision_evidence"]
            self.assertEqual(evidence["status"], "unrendered_preregistered_seed")
            self.assertEqual(evidence["prior_auh_job_id"], 131492)
            self.assertEqual(evidence["prior_job_state"], "FAILED")
            self.assertEqual(evidence["prior_elapsed_seconds"], 9)
            self.assertEqual(evidence["prior_artifact_mp4_count"], 0)
            self.assertEqual(evidence["prior_artifact_latent_count"], 0)
            self.assertFalse(evidence["prior_candidate_rendered"])
            self.assertFalse(evidence["seed_reuse_collision_with_rendered_media"])

    def test_caption_and_scientific_contract_are_sealed(self) -> None:
        changed = copy.deepcopy(self.registry)
        changed["cells"][0]["target_action_caption"] += " changed"
        with self.assertRaises(canary.NativeBranchHomotopyCanaryError):
            canary._registry_cell(changed, cell_id="fit-dog-7b88")

        changed = copy.deepcopy(self.registry)
        changed["contract"]["homotopy"]["hard_switch"] = True
        with self.assertRaises(canary.NativeBranchHomotopyCanaryError):
            canary._registry_cell(changed, cell_id="fit-dog-7b88")

        changed = copy.deepcopy(self.registry)
        changed["contract"]["native_full_source_endpoint"][
            "stock_vr2v_deployment_parity"
        ] = True
        with self.assertRaises(canary.NativeBranchHomotopyCanaryError):
            canary._registry_cell(changed, cell_id="fit-human-a35b")

        changed = copy.deepcopy(self.registry)
        changed["cells"][3]["seed"] = changed["cells"][2]["seed"]
        with self.assertRaises(canary.NativeBranchHomotopyCanaryError):
            canary._registry_cell(changed, cell_id="confirmation-human-a66e")

    def test_endpoints_are_true_low_and_reference_only_high(self) -> None:
        contract = self.registry["contract"]
        low = contract["native_full_source_endpoint"]
        high = contract["r2v4_reference_only_endpoint"]
        self.assertEqual(low["guidance_mode"], "v2v_apg")
        self.assertEqual(low["full_source_video_count"], 1)
        self.assertEqual(low["independently_vae_encoded_source_reference_count"], 4)
        self.assertEqual(low["forward_order"], ["VI_negative", "VI_action"])
        self.assertFalse(low["stock_vr2v_deployment_parity"])
        self.assertEqual(high["guidance_mode"], "r2v_apg")
        self.assertEqual(high["full_source_video_count"], 0)
        self.assertEqual(
            high["forward_order"], ["none_negative", "I_negative", "I_action"]
        )
        self.assertEqual(canary.GUIDANCE_BY_ARM[canary.ARM_ORDER[1]], "r2v_apg")

    def test_mode_native_prompts_intentionally_differ_only_above_body(self) -> None:
        cleaner = lambda value: "<clean>" + value
        caption = "The dog sits."
        low = canary.build_mode_native_prompt(
            "low-vr2v", caption, prompt_cleaner=cleaner
        )
        high = canary.build_mode_native_prompt(
            "high-r2v4", caption, prompt_cleaner=cleaner
        )
        self.assertNotEqual(low, high)
        self.assertIn(caption, low)
        self.assertIn(caption, high)
        self.assertIn("image0, image1, image2, and image3", high)
        self.assertNotIn("image4", high)
        disclosure = self.registry["contract"]["prompt_homotopy_disclosure"]
        self.assertTrue(disclosure["task_prefix_and_visual_regime_change_together"])
        self.assertFalse(disclosure["shared_vr2v_positive_embedding_across_endpoints"])

    def test_sampling_contract_locks_apg_and_exact40(self) -> None:
        for arm, guidance in canary.GUIDANCE_BY_ARM.items():
            contract = canary.sampling_contract(arm, seed=17)
            self.assertEqual(contract["guidance_mode"], guidance)
            self.assertEqual(contract["num_frames"], 81)
            self.assertEqual(contract["num_inference_steps"], 40)
            self.assertEqual(contract["flow_shift"], 5.0)
            self.assertEqual(contract["omega_img"], 4.5)
            self.assertEqual(contract["omega_txt"], 4.0)
            self.assertEqual(contract["eta"], 0.5)
            self.assertEqual(contract["norm_threshold"], (50.0, 50.0))
            self.assertEqual(contract["momentum"], 0.0)

    def test_exact_shift5_schedule_digest_and_regions_are_presealed(self) -> None:
        self.assertEqual(
            canary.SCHEDULE_SHA256,
            "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2",
        )
        self.assertEqual(canary.HIGH_ENDPOINT_STEP_INDICES, tuple(range(15)))
        self.assertEqual(canary.TRANSITION_STEP_INDICES, tuple(range(15, 31)))
        self.assertEqual(canary.LOW_ENDPOINT_STEP_INDICES, tuple(range(31, 40)))
        regions = self.registry["contract"]["apg_and_scheduler"][
            "homotopy_regions"
        ]
        self.assertEqual(
            regions["high_r2v4_weight_one_step_indices"], list(range(15))
        )
        self.assertEqual(regions["strict_transition_step_indices"], list(range(15, 31)))
        self.assertEqual(
            regions["low_v2v_weight_one_step_indices"], list(range(31, 40))
        )

    def test_live_trace_validator_binds_schedule_and_region_endpoints(self) -> None:
        schedule = canary.pinned_exact40_schedule_receipt()
        rows = []
        for index, (timestep, sigma) in enumerate(
            zip(schedule["timesteps"], canary.NATIVE_UNIPC40_SIGMAS)
        ):
            endpoint = (
                "high_r2v4_apg"
                if index < 15
                else "transition" if index < 31 else "low_official_v2v_apg"
            )
            rows.append(
                {
                    "step_index": index,
                    "timestep": float(timestep),
                    "sigma": sigma,
                    "endpoint": endpoint,
                    "high_r2v4_weight": 1.0 if index < 15 else 0.5 if index < 31 else 0.0,
                    "transformer_forwards": 5,
                    "low_vi_forwards": 2,
                    "high_r2v4_forwards": 3,
                    "original_scheduler_calls": 1,
                    "patch_call_count": 10,
                    "low_official_apg_exact_parity": True,
                    "freeze_safe_no_grad_outputs": True,
                }
            )
        trace = {
            "steps": 40,
            "transformer_forwards": 200,
            "low_vi_forwards": 80,
            "high_r2v4_forwards": 120,
            "patch_vae_latent_calls": 400,
            "original_scheduler_calls": 40,
            "low_official_apg_exact_parity_all_steps": True,
            "smoothstep_sigma_low": 0.6,
            "smoothstep_sigma_high": 0.9,
            "trace": rows,
        }
        self.assertEqual(
            canary.validate_homotopy_runtime_trace(trace)["digest"],
            canary.SCHEDULE_SHA256,
        )
        changed = copy.deepcopy(trace)
        changed["trace"][15]["endpoint"] = "high_r2v4_apg"
        with self.assertRaises(canary.NativeBranchHomotopyCanaryError):
            canary.validate_homotopy_runtime_trace(changed)

    def test_runner_uses_runtime_patch_and_no_external_editing_signal(self) -> None:
        source = Path(canary.__file__).read_text(encoding="utf-8")
        self.assertIn("NativeBranchHomotopyRuntimePatch", source)
        self.assertIn("r2v_action_prompt_embeds=high_embeds", source)
        self.assertIn('"multi_video_vae_latents": (', source)
        self.assertIn("None if arm == ARM_ORDER[1]", source)
        self.assertIn('"target_video": False', source)
        self.assertIn('"custom_initial_noise": False', source)
        self.assertIn('"mask_track_pose_flow": False', source)
        self.assertNotIn("target_video_latent", source)
        self.assertNotIn("optimizer.step", source)


if __name__ == "__main__":
    unittest.main()
