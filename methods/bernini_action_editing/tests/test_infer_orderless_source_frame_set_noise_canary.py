from __future__ import annotations

import ast
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = METHOD_ROOT / "infer_orderless_source_frame_set_noise_canary.py"
OPERATOR_PATH = METHOD_ROOT / "orderless_source_frame_set_noise.py"
SPEC_PATH = METHOD_ROOT / "assets/orderless_source_frame_set_noise_core2_v1.json"
AUTHORING_PATH = (
    METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
)
RUNNER_SOURCE = RUNNER_PATH.read_text(encoding="utf-8")
RUNNER_TREE = ast.parse(RUNNER_SOURCE, filename=str(RUNNER_PATH))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
DIFFUSERS_AVAILABLE = importlib.util.find_spec("diffusers") is not None
_SPEC_FOR_MEDIA_CHECK = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
AUH_FIXED_MEDIA_AVAILABLE = all(
    Path(row[key]).is_file()
    for row in _SPEC_FOR_MEDIA_CHECK["cells"]
    for key in ("source_video", "wrong_source_video")
)


def _literal_assignment(name: str) -> object:
    for node in RUNNER_TREE.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = node.value
            return ast.literal_eval(value)
    raise AssertionError(f"missing literal assignment: {name}")


class OrderlessSourceFrameSetNoiseCanaryContractTests(unittest.TestCase):
    def test_exact81_five_arm_factorial_is_closed(self) -> None:
        self.assertEqual(_literal_assignment("FRAME_COUNT"), 81)
        self.assertEqual(_literal_assignment("LATENT_PHASES"), 21)
        self.assertEqual(_literal_assignment("NUM_INFERENCE_STEPS"), 40)
        self.assertEqual(_literal_assignment("WORLD_SIZE"), 4)
        self.assertEqual(_literal_assignment("SP_SIZE"), 4)
        self.assertEqual(_literal_assignment("REFERENCE_INDICES"), (0, 27, 53, 80))
        arms = (
            "official_gaussian",
            "correct_source_rho005",
            "wrong_source_rho005",
            "correct_source_rho010",
            "wrong_source_rho010",
        )
        self.assertEqual(_literal_assignment("ARM_ORDER"), arms)
        self.assertEqual(
            _literal_assignment("ARM_RHO"),
            {
                "official_gaussian": 0.0,
                "correct_source_rho005": 0.05,
                "wrong_source_rho005": 0.05,
                "correct_source_rho010": 0.10,
                "wrong_source_rho010": 0.10,
            },
        )
        self.assertEqual(
            _literal_assignment("ARM_CARRIER_ROLE"),
            {
                "official_gaussian": "correct",
                "correct_source_rho005": "correct",
                "wrong_source_rho005": "wrong",
                "correct_source_rho010": "correct",
                "wrong_source_rho010": "wrong",
            },
        )

    def test_core2_spec_preregisters_content_matched_wrong_sources(self) -> None:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            spec["schema_version"],
            "bernini-orderless-source-frame-set-noise-core2-spec-v1",
        )
        contract = spec["contract"]
        self.assertEqual(contract["frame_count"], 81)
        self.assertEqual(contract["latent_phases"], 21)
        self.assertEqual(contract["reference_indices"], [0, 27, 53, 80])
        self.assertEqual(
            contract["topology"], "two_isolated_world4_on_one_8gpu_node"
        )
        self.assertTrue(
            contract["same_source_action_seed_scheduler_guidance_across_five_arms"]
        )
        self.assertFalse(contract["full_source_video_latent_available_to_carrier"])
        self.assertFalse(contract["wrong_source_available_to_model_conditioning"])
        self.assertFalse(contract["target_video"])
        self.assertFalse(contract["mask_flow_pose_track_trajectory"])
        self.assertFalse(contract["trainer_or_critic_instantiated"])
        self.assertFalse(contract["training_harness_executed"])
        self.assertTrue(contract["validation_only_train_lora_module_imported"])
        self.assertFalse(
            contract["identity_orbit_or_identity_adapter_baseline_or_prior"]
        )
        self.assertEqual(
            contract["required_posthoc_gate_names"],
            [
                "same_rho_correct_vs_wrong_carrier_source_specificity",
                "old_motion_direction_and_order_leakage_nonincrease",
                "target_action_nonregression_vs_official_gaussian",
                "identity_blur_flicker_camera_quality_nonregression_vs_official_gaussian",
            ],
        )
        self.assertFalse(contract["posthoc_gates_executed_by_canary"])
        self.assertFalse(contract["best_arm_selection"])
        provenance = contract["selection_provenance"]
        self.assertEqual(
            provenance["file_sha256"],
            "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c",
        )
        self.assertEqual(
            hashlib.sha256(AUTHORING_PATH.read_bytes()).hexdigest(),
            provenance["file_sha256"],
        )

        cells = spec["cells"]
        self.assertEqual([row["cell_id"] for row in cells], ["dog", "human"])
        self.assertEqual({row["seed"] for row in cells}, {2026080907})
        expected_wrong = {
            "dog": (
                "7b88a1ca1f804f41",
                "841b5e0080a1441d",
                "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
                4,
                "6422e2e8b16b45eb8b58c51da6eee5f251cc1bc80a741bb1111d5fe394902bf1",
                5,
                "c5dcdafcc66149adafcb3b3e97cd7cab732fca2a5cc8f3c7ab9f79eb81438b7b",
            ),
            "human": (
                "a35b590961d24694",
                "a66e6818e4144928",
                "0fdc54d89250f355d2170a4d6f6aac0867abf592afb849668a8e2879a6617147",
                6,
                "10c81d1530025f70ed24e498a7fcd4d265aae4ca50230f5316bd8b4a632181c4",
                7,
                "05a5c301e7c9b066b76f03f642c3f659f3ff161e0a9f0ffdf4de31c1497a7442",
            ),
        }
        authoring = json.loads(AUTHORING_PATH.read_text(encoding="utf-8"))
        for row in cells:
            with self.subTest(cell=row["cell_id"]):
                (
                    source_iid,
                    iid,
                    digest,
                    source_index,
                    source_row_digest,
                    wrong_index,
                    wrong_row_digest,
                ) = expected_wrong[row["cell_id"]]
                self.assertEqual(row["actor_kind"], row["cell_id"])
                self.assertEqual(row["source_iid"], source_iid)
                self.assertEqual(row["wrong_source_iid"], iid)
                self.assertIn(f"/{iid}/samples/{iid}/source_video.mp4", row["wrong_source_video"])
                self.assertEqual(row["wrong_source_video_sha256"], digest)
                self.assertNotEqual(row["source_video"], row["wrong_source_video"])
                self.assertNotEqual(
                    row["source_video_sha256"], row["wrong_source_video_sha256"]
                )
                self.assertEqual(
                    hashlib.sha256(row["action_caption"].encode("utf-8")).hexdigest(),
                    row["action_caption_utf8_sha256"],
                )
                self.assertTrue(row["selected_before_generation"])
                self.assertEqual(row["source_authoring_row_index"], source_index)
                self.assertEqual(
                    row["source_authoring_row_sha256"], source_row_digest
                )
                self.assertEqual(row["wrong_source_authoring_row_index"], wrong_index)
                self.assertEqual(
                    row["wrong_source_authoring_row_sha256"], wrong_row_digest
                )
                for index, expected_iid, expected_digest in (
                    (source_index, source_iid, source_row_digest),
                    (wrong_index, iid, wrong_row_digest),
                ):
                    authoring_row = authoring["cells"][index]
                    self.assertEqual(authoring_row["iid"], expected_iid)
                    raw = json.dumps(
                        authoring_row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("ascii")
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_digest)
                match = row["wrong_source_content_match"]
                for key in (
                    "distinct_source_identity",
                    "same_broad_actor_class",
                    "same_actor_count",
                    "same_authoring_action_family_id",
                    "fixed_camera_in_both",
                    "selected_before_canary_generation",
                ):
                    self.assertTrue(match[key], key)
                self.assertFalse(match["same_exact_scene"])
                self.assertFalse(match["pure_identity_control"])
                self.assertEqual(
                    row["control_role"],
                    "action_family_matched_source_specificity_control_not_pure_identity_control",
                )
                if row["cell_id"] == "human":
                    self.assertFalse(match["same_initial_pose_family"])
                    self.assertFalse(match["same_native_geometry"])
                    self.assertTrue(match["direct_resize_geometry_confound"])
                    self.assertEqual(
                        row["geometry_pre_registration"],
                        {
                            "source_input_hw": [768, 704],
                            "source_native_bucket_hw": [512, 464],
                            "wrong_source_input_hw": [896, 704],
                            "wrong_source_native_bucket_hw": [544, 432],
                            "wrong_source_direct_resize_target_bucket_hw": [512, 464],
                            "wrong_source_direct_resize_anisotropic_aspect_distortion_fraction": 0.15340909090909083,
                            "differential_geometry_confound": True,
                        },
                    )
                else:
                    self.assertTrue(match["same_initial_pose_family"])
                    self.assertTrue(match["same_native_geometry"])
                    self.assertFalse(match["direct_resize_geometry_confound"])
                    self.assertEqual(
                        row["geometry_pre_registration"],
                        {
                            "source_input_hw": [704, 736],
                            "source_native_bucket_hw": [480, 496],
                            "wrong_source_input_hw": [704, 736],
                            "wrong_source_native_bucket_hw": [480, 496],
                            "wrong_source_direct_resize_target_bucket_hw": [480, 496],
                            "wrong_source_direct_resize_anisotropic_aspect_distortion_fraction": 0.011730205278592365,
                            "differential_geometry_confound": False,
                        },
                    )

    def test_wrong_source_is_carrier_only_and_each_reference_is_a_T1_call(self) -> None:
        self.assertIn(
            "correct_pixels[:, :, index : index + 1].contiguous()", RUNNER_SOURCE
        )
        self.assertIn(
            "wrong_pixels[:, :, index : index + 1].contiguous()", RUNNER_SOURCE
        )
        self.assertIn(
            '"multi_video_vae_latents": [full_source_latent]', RUNNER_SOURCE
        )
        self.assertIn(
            "correct_refs[index] for index in REFERENCE_INDICES", RUNNER_SOURCE
        )
        conditions_start = RUNNER_SOURCE.index("conditions = {")
        conditions_end = RUNNER_SOURCE.index("generated: dict", conditions_start)
        condition_block = RUNNER_SOURCE[conditions_start:conditions_end]
        self.assertNotIn("wrong_refs", condition_block)
        self.assertNotIn("wrong_frames_cpu", condition_block)
        self.assertIn(
            "wrong_source_used_as_model_condition\": False", RUNNER_SOURCE
        )
        self.assertIn("direct_raw_RGB_target_bucket_resize", RUNNER_SOURCE)
        self.assertIn('"double_resize_used": False', RUNNER_SOURCE)
        self.assertIn('"caller_selection_indices_consumed": True', RUNNER_SOURCE)
        self.assertIn('"operator_received_frame_indices": False', RUNNER_SOURCE)
        self.assertIn('"operator_set_sequence_order_consumed": False', RUNNER_SOURCE)
        self.assertIn('"unordered_pose_occupancy_leakage_possible": True', RUNNER_SOURCE)
        self.assertIn('"wrong_source_is_pure_identity_control": False', RUNNER_SOURCE)
        self.assertIn(
            '"direct_resize_anisotropic_aspect_distortion_fraction"', RUNNER_SOURCE
        )
        self.assertIn('"identity_orbit_decoded_role": "excluded_external_no_go"', RUNNER_SOURCE)
        self.assertIn('"identity_adapter_used_as_baseline": False', RUNNER_SOURCE)
        self.assertIn('"identity_adapter_used_as_prior": False', RUNNER_SOURCE)
        self.assertIn(
            '"same_rho_correct_vs_wrong_carrier_source_"', RUNNER_SOURCE
        )
        self.assertIn(
            '"old_motion_direction_and_order_leakage_"', RUNNER_SOURCE
        )

    def test_hook_changes_only_native_initial_noise_and_restores_in_finally(self) -> None:
        hook = next(
            node
            for node in RUNNER_TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_sample_with_source_set_noise_arm"
        )
        source = ast.get_source_segment(RUNNER_SOURCE, hook) or ""
        self.assertIn("official_native = original(*call_args, **call_kwargs)", source)
        self.assertIn(
            "source_set_noise.build_orderless_source_frame_set_noise(", source
        )
        self.assertLess(
            source.index("official_native = original(*call_args, **call_kwargs)"),
            source.index("build_orderless_source_frame_set_noise("),
        )
        self.assertIn("finally:", source)
        self.assertIn("setattr(wan_diffusion_module, \"randn_tensor\", original)", source)
        self.assertEqual(
            _literal_assignment("TEMPORARY_MUTATION_SURFACE"),
            ("bernini.models.wan_diffusion.randn_tensor",),
        )
        self.assertNotIn("sample_one_step =", source)
        self.assertNotIn("optimizer", source.lower())
        self.assertNotIn("backward", source.lower())
        self.assertIn(
            'prototype_artifact["raw_value_sha256"]\n'
            "                        != capture.source_set_prototype_raw_value_sha256",
            RUNNER_SOURCE,
        )
        self.assertIn(
            'carrier_artifact["raw_value_sha256"]\n'
            "                        != capture.temporal_dc_carrier_raw_value_sha256",
            RUNNER_SOURCE,
        )

    def test_operator_and_runner_make_chronology_limits_explicit(self) -> None:
        operator_source = OPERATOR_PATH.read_text(encoding="utf-8")
        self.assertIn("input_is_an_unordered_multiset", operator_source)
        self.assertIn('"caller_sequence_order_consumed": False', operator_source)
        self.assertIn('"source_frame_indices_consumed": False', operator_source)
        self.assertIn('"source_chronology_or_direction_representable": False', operator_source)
        self.assertIn('"static_pose_can_be_retained": True', operator_source)
        self.assertIn('"unordered_pose_occupancy_can_be_retained": True', operator_source)
        self.assertIn("source_frame_order_consumed\") is not False", RUNNER_SOURCE)
        self.assertIn("source_temporal_phase_consumed\") is not False", RUNNER_SOURCE)

    @unittest.skipUnless(
        TORCH_AVAILABLE and DIFFUSERS_AVAILABLE,
        "PyTorch and Diffusers are required for the dynamic hook test",
    )
    def test_upstream_authoring_rows_are_runtime_hash_bound(self) -> None:
        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        canary = importlib.import_module(
            "infer_orderless_source_frame_set_noise_canary"
        )
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        for cell in spec["cells"]:
            with self.subTest(cell=cell["cell_id"]):
                root, source, wrong, path, digest = canary.load_authoring_provenance(
                    AUTHORING_PATH,
                    expected_file_sha256=(
                        "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
                    ),
                    cell=cell,
                )
                self.assertEqual(root["bank_id"], "pair5-t2v-first8-v1")
                self.assertEqual(source["analysis_split"], "fit")
                self.assertEqual(wrong["analysis_split"], "confirmation")
                self.assertEqual(source["action_family_id"], wrong["action_family_id"])
                self.assertEqual(path, AUTHORING_PATH.resolve())
                self.assertEqual(digest, canary.AUTHORING_SPEC_SHA256)

    @unittest.skipUnless(
        TORCH_AVAILABLE and DIFFUSERS_AVAILABLE and AUH_FIXED_MEDIA_AVAILABLE,
        "AUH fixed media and vace dependencies are required",
    )
    def test_fixed_media_hashes_and_geometry_confound_match_preregistration(self) -> None:
        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        canary = importlib.import_module(
            "infer_orderless_source_frame_set_noise_canary"
        )
        spec_sha = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
        for expected_cell in _SPEC_FOR_MEDIA_CHECK["cells"]:
            cell_id = expected_cell["cell_id"]
            with self.subTest(cell=cell_id):
                _, cell, wrong, _, _ = canary.load_cell_spec(
                    SPEC_PATH,
                    expected_file_sha256=spec_sha,
                    cell_id=cell_id,
                )
                from tools import materialize_vae

                correct_frames, correct_fps, correct_hw = (
                    materialize_vae._decode_exact_video(Path(cell["source_video"]))
                )
                self.assertAlmostEqual(correct_fps, 25.0, places=3)
                target_hw = materialize_vae.source_aspect_bucket(
                    *correct_hw,
                    max_pixels=canary.native.legacy.MAX_PIXELS,
                    stride=canary.native.legacy.SPATIAL_STRIDE,
                )
                self.assertEqual(int(correct_frames.shape[0]), 81)
                del correct_frames
                gc.collect()
                wrong_tensor, wrong_metadata, _ = canary._prepare_source_snapshot_at_bucket(
                    Path(wrong["source_video"]), bucket_hw=target_hw
                )
                expected = cell["geometry_pre_registration"]
                self.assertEqual(list(correct_hw), expected["source_input_hw"])
                self.assertEqual(
                    list(target_hw),
                    expected["source_native_bucket_hw"],
                )
                self.assertEqual(wrong_metadata["source_input_hw"], expected["wrong_source_input_hw"])
                self.assertEqual(
                    wrong_metadata["source_native_bucket_hw"],
                    expected["wrong_source_native_bucket_hw"],
                )
                self.assertEqual(
                    wrong_metadata["target_cell_bucket_hw"],
                    expected["wrong_source_direct_resize_target_bucket_hw"],
                )
                self.assertAlmostEqual(
                    wrong_metadata[
                        "direct_resize_anisotropic_aspect_distortion_fraction"
                    ],
                    expected[
                        "wrong_source_direct_resize_anisotropic_aspect_distortion_fraction"
                    ],
                    places=12,
                )
                self.assertEqual(
                    wrong_metadata["native_bucket_matches_target_cell_bucket"],
                    not expected["differential_geometry_confound"],
                )
                self.assertEqual(int(wrong_tensor.shape[2]), 81)
                del wrong_tensor
                gc.collect()

    @unittest.skipUnless(
        TORCH_AVAILABLE and DIFFUSERS_AVAILABLE,
        "PyTorch and Diffusers are required for the dynamic hook test",
    )
    def test_dynamic_hook_has_one_parent_gaussian_and_orderless_controls(self) -> None:
        import torch

        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        canary = importlib.import_module(
            "infer_orderless_source_frame_set_noise_canary"
        )

        shape = (1, 16, 21, 4, 6)
        seed = 2026080907

        def canonical_randn_tensor(
            requested_shape: object,
            *,
            generator: torch.Generator,
            device: torch.device,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return torch.randn(
                tuple(requested_shape),
                generator=generator,
                device=device,
                dtype=dtype,
            ).contiguous()

        module = types.SimpleNamespace(randn_tensor=canonical_randn_tensor)
        correct_generator = torch.Generator(device="cpu").manual_seed(4103)
        wrong_generator = torch.Generator(device="cpu").manual_seed(9107)
        correct_frames = tuple(
            (
                torch.randn(
                    (1, 16, 1, 4, 6), generator=correct_generator, dtype=torch.float32
                )
                + 0.25
            ).contiguous()
            for _ in range(4)
        )
        wrong_frames = tuple(
            (
                torch.randn(
                    (1, 16, 1, 4, 6), generator=wrong_generator, dtype=torch.float32
                )
                - 0.35
            ).contiguous()
            for _ in range(4)
        )

        def run(arm: str, correct: tuple[torch.Tensor, ...] = correct_frames):
            def sample() -> torch.Tensor:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                return module.randn_tensor(
                    shape,
                    generator=generator,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

            result, capture = canary._sample_with_source_set_noise_arm(
                sample_fn=sample,
                wan_diffusion_module=module,
                arm=arm,
                correct_frame_latents_cpu=correct,
                wrong_frame_latents_cpu=wrong_frames,
                expected_shape=shape,
                expected_device=torch.device("cpu"),
                expected_seed=seed,
                canonical_randn_tensor=canonical_randn_tensor,
            )
            self.assertIs(module.randn_tensor, canonical_randn_tensor)
            self.assertEqual(
                capture.sampler_initial_noise_raw_value_sha256,
                canary._tensor_identity(result, label="returned")["raw_storage_sha256"],
            )
            return capture

        captures = {arm: run(arm) for arm in canary.ARM_ORDER}
        self.assertEqual(
            len(
                {
                    capture.official_gaussian_raw_value_sha256
                    for capture in captures.values()
                }
            ),
            1,
        )
        official = captures["official_gaussian"]
        self.assertTrue(official.original_return_object_forwarded)
        self.assertFalse(official.external_initial_noise_injection)
        self.assertEqual(
            official.official_gaussian_raw_value_sha256,
            official.sampler_initial_noise_raw_value_sha256,
        )
        for role in ("correct", "wrong"):
            low = captures[f"{role}_source_rho005"]
            high = captures[f"{role}_source_rho010"]
            self.assertEqual(
                low.source_frame_multiset_sha256, high.source_frame_multiset_sha256
            )
            self.assertEqual(
                low.source_set_prototype_sha256, high.source_set_prototype_sha256
            )
            self.assertEqual(low.temporal_dc_carrier_sha256, high.temporal_dc_carrier_sha256)
            self.assertEqual(
                low.source_set_prototype_raw_value_sha256,
                high.source_set_prototype_raw_value_sha256,
            )
            self.assertEqual(
                low.temporal_dc_carrier_raw_value_sha256,
                high.temporal_dc_carrier_raw_value_sha256,
            )
            self.assertNotEqual(
                low.sampler_initial_noise_raw_value_sha256,
                high.sampler_initial_noise_raw_value_sha256,
            )
        self.assertNotEqual(
            captures["correct_source_rho005"].temporal_dc_carrier_sha256,
            captures["wrong_source_rho005"].temporal_dc_carrier_sha256,
        )
        reversed_capture = run("correct_source_rho010", tuple(reversed(correct_frames)))
        self.assertEqual(
            reversed_capture.sampler_initial_noise_raw_value_sha256,
            captures["correct_source_rho010"].sampler_initial_noise_raw_value_sha256,
        )
        self.assertEqual(
            reversed_capture.operator_receipt_sha256,
            captures["correct_source_rho010"].operator_receipt_sha256,
        )


if __name__ == "__main__":
    unittest.main()
