#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_saic_source_state_flow_transport_v2 as runner  # noqa: E402


class FixedR2VVisualI0DiagnosticTest(unittest.TestCase):
    def test_minimal_arm_truth_table_is_exact(self) -> None:
        self.assertEqual(runner.ARM_NAMES, ("R00", "R11"))
        r00 = runner.arm_spec("R00")
        r11 = runner.arm_spec("R11")
        for spec in (r00, r11):
            self.assertEqual(spec.task_name, "r2v")
            self.assertEqual(spec.candidate_schedule, (1,) * 40)
            self.assertIs(spec.anc_enabled, False)
            self.assertEqual(spec.aggregation_mode, "uniform")
            self.assertIsNone(spec.temperature)
            self.assertIs(spec.anchor_latent_phase_zero, False)
            self.assertEqual(spec.expected_guided_queries, 80)
        self.assertEqual(r00.field_regime, "t2v_apg")
        self.assertEqual(r00.guidance_mode, "t2v_apg")
        self.assertEqual(r00.expected_raw_forwards, 160)
        self.assertIs(r00.uses_reference_frame0, False)
        self.assertEqual(r11.field_regime, "r2v_apg_source_i0")
        self.assertEqual(r11.guidance_mode, "r2v_apg")
        self.assertEqual(r11.expected_raw_forwards, 240)
        self.assertIs(r11.uses_reference_frame0, True)
        with self.assertRaises(runner.SAICInferenceError):
            runner.arm_spec("R10")

    def test_both_arms_use_identical_r2v_system_prompt(self) -> None:
        cleaner = lambda text: " " + text
        expected = runner.R2V_SYSTEM_PROMPT + " source body"
        for arm in runner.ARM_NAMES:
            self.assertEqual(
                runner.build_task_prompt(
                    runner.arm_spec(arm).task_name,
                    "source body",
                    prompt_cleaner=cleaner,
                ),
                expected,
            )

    def test_only_treatment_difference_is_whole_schedule_visual_i0(self) -> None:
        off = runner.guidance_contract(runner.arm_spec("R00"))
        on = runner.guidance_contract(runner.arm_spec("R11"))
        fixed = (
            "task_name",
            "guidance_scale",
            "apg_eta",
            "candidate_schedule",
            "candidate_continuation",
            "anc_enabled",
            "aggregation_mode",
            "temperature",
            "anchor_latent_phase_zero",
            "full_source_video_field_tokens",
            "spatial_arithmetic",
        )
        for key in fixed:
            self.assertEqual(off[key], on[key], key)
        self.assertEqual(off["visual_condition"], "none")
        self.assertEqual(
            on["visual_condition"],
            "independently_vae_encoded_source_rgb_frame0",
        )

    def test_keyed_noise_is_arm_independent_and_v1_compatible(self) -> None:
        import infer_saic_source_state_flow_transport_v1 as v1

        for seed in (2026082101, 2026082121):
            for step in (0, 17, 39):
                self.assertEqual(
                    runner.keyed_noise_seed(seed, step, 0),
                    v1.keyed_noise_seed(seed, step, 0),
                )
        self.assertEqual(runner.NOISE_DOMAIN, v1.NOISE_DOMAIN)

    def test_frame0_artifact_parser_accepts_exact_and_rejects_wrong_raw_hash(self) -> None:
        import torch
        from safetensors.torch import save

        tensor = torch.arange(1 * 16 * 1 * 2 * 3, dtype=torch.float32).reshape(
            1, 16, 1, 2, 3
        )
        payload = save(
            {runner.FRAME0_TENSOR_KEY: tensor},
            metadata=dict(runner.FRAME0_ARTIFACT_METADATA),
        )
        digest = runner.tensor_raw_sha256(tensor)
        restored, metadata = runner._parse_frame0_safetensors(
            payload,
            expected_shape=tensor.shape,
            expected_tensor_raw_sha256=digest,
        )
        self.assertTrue(torch.equal(restored, tensor))
        self.assertEqual(metadata, dict(runner.FRAME0_ARTIFACT_METADATA))
        with self.assertRaises(runner.SAICInferenceError):
            runner._parse_frame0_safetensors(
                payload,
                expected_shape=tensor.shape,
                expected_tensor_raw_sha256="0" * 64,
            )

    def test_frame0_receipt_parser_is_canonical_and_false_authority_is_hostile(self) -> None:
        receipt = {
            "schema_version": runner.FRAME0_RECEIPT_SCHEMA,
            "method": runner.FRAME0_MATERIALIZER_METHOD,
            "artifact": {},
            "sealed_inputs": {},
            "preprocessing": {},
            "model_closure": {},
            "encoding": {},
            "runtime": {},
            "authority": {"training": False, "selection": False},
        }
        receipt["receipt_digest"] = runner.legacy.object_sha256(receipt)
        payload = runner.legacy.canonical_json_bytes(receipt) + b"\n"
        restored, declared = runner._parse_frame0_receipt(payload)
        self.assertEqual(restored, receipt)
        self.assertEqual(declared, receipt["receipt_digest"])
        runner._require_false_authority(restored["authority"], label="test")
        poisoned = dict(restored["authority"])
        poisoned["selection"] = True
        with self.assertRaises(runner.SAICInferenceError):
            runner._require_false_authority(poisoned, label="test")

    def test_cli_and_runtime_closure_require_both_coordinates(self) -> None:
        parser = runner.build_parser()
        required = {action.dest for action in parser._actions if action.required}
        for name in (
            "source_clean_latent",
            "source_clean_latent_receipt",
            "reference_frame0_latent",
            "reference_frame0_latent_receipt",
            "expected_reference_frame0_tensor_raw_sha256",
        ):
            self.assertIn(name, required)
        self.assertIn(
            "infer_saic_source_state_flow_transport_v1.py",
            runner.RUNTIME_METHOD_FILES,
        )
        self.assertIn(
            "materialize_saic_frame0_latent_v1.py", runner.RUNTIME_METHOD_FILES
        )
        source = inspect.getsource(runner.main)
        self.assertNotIn("_vae_encode(", source)
        self.assertIn("load_sealed_frame0_coordinate", source)
        self.assertIn("revalidate_sealed_frame0_coordinate", source)


if __name__ == "__main__":
    unittest.main()
