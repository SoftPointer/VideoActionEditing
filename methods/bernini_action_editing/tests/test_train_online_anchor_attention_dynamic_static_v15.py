from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v15 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_online_anchor_attention_dynamic_static_v15 as method


WORKER = ROOT / "scripts/auh_train_online_anchor_dynamic_static_v15.sh"
LAUNCHER = ROOT / "scripts/auh_launch_online_anchor_dynamic_static_v15_job149363.sh"


class DynamicStaticV15Test(unittest.TestCase):
    def args(self, output: Path, *, profile: str = "dynamic_static"):
        return method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/Bernini-R-1.3B-Diffusers-ff4c5d4",
                "--pair-manifest", "/tmp/pairs.json",
                "--authoring", "/tmp/authoring.json",
                "--output", str(output),
                "--profile", profile,
                "--route-operator", method.ROUTE_OPERATOR,
                "--max-steps", "2",
                "--micro-records", "2",
                "--source-variant", "not_applicable",
                "--route-strength", "0.25",
                "--teacher-route-strength", "0.50",
                "--training-objective", method.OBJECTIVE,
                "--training-interface", "first_phase_caption_i2v",
                "--paired-target-fm-weight", "0",
                "--real-source-manifest", "/tmp/real-source.json",
                "--real-source-manifest-sha256", "8" * 64,
                "--teacher-delta-mode", "raw",
                "--routed-teacher-mode", "same_action_route_only",
                "--source-reconstruction-weight", "0.025",
                "--replay-combine-mode", method.REPLAY_COMBINE_MODE,
                "--source-reconstruction-prompt", "action",
                "--learning-rate", "1e-5",
                "--method-source-revision", "1" * 40,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )

    @staticmethod
    def spatial_to_patches(value):
        batch, channels, phases, height, width = map(int, value.shape)
        return (
            value.reshape(
                batch, channels, phases, 1, height // 2, 2, width // 2, 2
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(phases * (height // 2) * (width // 2), channels, 1, 2, 2)
            .contiguous()
        )

    def test_validation_accepts_only_fresh_exact_v15_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh-v15-s2"
            method.validate_args(self.args(output))
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(self.args(output, profile="action_noop"))

            wrong = self.args(output)
            wrong.routed_teacher_mode = "cross_caption_two_sided"
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(wrong)

            prior_adapter = self.args(output)
            prior_adapter.checkpoint = "/tmp/train_old/checkpoint-00000032"
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(prior_adapter)

    def test_dynamic_static_builder_enforces_state_not_caption_and_same_noise(self):
        dynamic = torch.arange(
            1 * 16 * 21 * 4 * 6, dtype=torch.float32
        ).reshape(1, 16, 21, 4, 6).div_(1000.0)
        clean_inputs = []
        prompts = []

        def transform(sample, seed):
            clean = sample["video_vae_latents"][-1].clone()
            clean_inputs.append(clean)
            prompts.append(sample["inputs"])
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(clean.shape, generator=generator)
            velocity = self.spatial_to_patches(noise - clean)
            return {
                "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.int64),
                "attention_mask": torch.ones((1, 3), dtype=torch.int64),
                "t5_input_lens": torch.tensor([[3]], dtype=torch.int64),
                "timesteps": torch.tensor([[500.0]], dtype=torch.bfloat16),
                "target_velocity": velocity,
            }

        donor = {"event_id": "event", "variant_id": "v1", "iid": "event-v1"}
        captions = {("event", "v1"): {"target": "A hand lifts the cup."}}
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        with mock.patch.object(
            method.base.pairs,
            "load_row_tensors",
            return_value=(dynamic, None, None, None, None),
        ), mock.patch.object(
            method.base, "_blob", side_effect=lambda clean, mean, std: clean.clone()
        ):
            action, contrast, shape = method.build_anchor_batches(
                target_row=donor,
                donor=donor,
                profile="dynamic_static",
                transform=transform,
                mean=None,
                std=None,
                seed=2026082301,
                captions=captions,
            )

        self.assertEqual(shape, (1, 16, 21, 4, 6))
        self.assertEqual(prompts[0], prompts[1])
        self.assertTrue(torch.equal(action["input_ids"], contrast["input_ids"]))
        self.assertTrue(torch.equal(clean_inputs[0][:, :, 0], clean_inputs[1][:, :, 0]))
        self.assertTrue(
            torch.equal(clean_inputs[1], clean_inputs[1][:, :, :1].expand_as(clean_inputs[1]))
        )
        self.assertFalse(torch.equal(clean_inputs[0][:, :, 1:], clean_inputs[1][:, :, 1:]))
        self.assertEqual(method._RUNTIME_AUDIT["batch_pair_count"], 1)
        self.assertLessEqual(
            method._RUNTIME_AUDIT["recovered_gaussian_fp32_max_abs_error"],
            method.NOISE_RECOVERY_ATOL,
        )

    def test_manifest_is_variant_major_and_event_interleaved(self):
        rows = [
            {
                "event_id": f"e{event:02d}",
                "variant_id": f"v{variant}",
                "iid": f"e{event:02d}-v{variant}",
            }
            for event in range(8)
            for variant in range(4)
        ]
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        with mock.patch.object(
            method, "_BASE_LOAD_MANIFEST", return_value=({"rows": 32}, rows)
        ):
            manifest, reordered = method.load_manifest_event_interleaved_v15(
                Path("/tmp/manifest.json")
            )
        self.assertEqual(manifest, {"rows": 32})
        self.assertEqual(
            [row["iid"] for row in reordered[:8]],
            [f"e{event:02d}-v0" for event in range(8)],
        )
        self.assertEqual(len({row["event_id"] for row in reordered[:8]}), 8)
        self.assertEqual(len({row["iid"] for row in reordered[:32]}), 32)
        self.assertEqual(
            method._RUNTIME_AUDIT["manifest_training_order"],
            "variant_major_event_interleaved_v15",
        )

    def test_builder_rejects_a_different_gaussian(self):
        dynamic = torch.arange(
            1 * 16 * 21 * 4 * 6, dtype=torch.float32
        ).reshape(1, 16, 21, 4, 6).div_(1000.0)
        calls = 0

        def transform(sample, seed):
            nonlocal calls
            clean = sample["video_vae_latents"][-1]
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(clean.shape, generator=generator)
            if calls == 1:
                noise = noise + 0.1
            calls += 1
            return {
                "input_ids": torch.tensor([[1]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1]], dtype=torch.int64),
                "t5_input_lens": torch.tensor([[1]], dtype=torch.int64),
                "timesteps": torch.tensor([[500.0]], dtype=torch.bfloat16),
                "target_velocity": self.spatial_to_patches(noise - clean),
            }

        donor = {"event_id": "event", "variant_id": "v1", "iid": "event-v1"}
        with mock.patch.object(
            method.base.pairs,
            "load_row_tensors",
            return_value=(dynamic, None, None, None, None),
        ), mock.patch.object(
            method.base, "_blob", side_effect=lambda clean, mean, std: clean.clone()
        ), self.assertRaises(method.base.OnlineAnchorTrainingError):
            method.build_anchor_batches(
                target_row=donor,
                donor=donor,
                profile="dynamic_static",
                transform=transform,
                mean=None,
                std=None,
                seed=7,
                captions={("event", "v1"): {"target": "Same caption."}},
            )

    def test_receipt_replaces_action_noop_claim_and_stays_non_scientific(self):
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        method._RUNTIME_AUDIT["batch_pair_count"] = 4
        method._RUNTIME_AUDIT["recovered_gaussian_fp32_max_abs_error"] = 1e-6
        method._RUNTIME_AUDIT["manifest_training_order"] = (
            "variant_major_event_interleaved_v15"
        )
        method._RUNTIME_AUDIT["manifest_ordered_iids"] = tuple(
            f"e{event:02d}-v{variant}"
            for variant in range(4)
            for event in range(8)
        )
        method._RUNTIME_AUDIT["target_iids"].update(("e00-v0", "e01-v0"))
        method._RUNTIME_AUDIT["target_events"].update(("e00", "e01"))
        method._RUNTIME_AUDIT["donor_iids"].update(("e00-v1", "e01-v1"))
        method._RUNTIME_AUDIT["donor_events"].update(("e00", "e01"))
        original = {
            "schema_version": method.base.QK_ONLY_RECEIPT_SCHEMA,
            "training_contract": {
                "profile": "action_noop",
                "anchor_qk_support_uses_phase0_relative_action_noop_contrast": True,
            },
        }
        with mock.patch.object(
            method, "_BASE_CHECKPOINT_RECEIPT", return_value=original
        ):
            receipt = method.checkpoint_receipt(args=object())

        contract = receipt["training_contract"]
        self.assertEqual(receipt["schema_version"], method.RECEIPT_SCHEMA)
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertEqual(contract["profile"], "dynamic_static")
        self.assertFalse(
            contract["anchor_qk_support_uses_phase0_relative_action_noop_contrast"]
        )
        self.assertTrue(
            contract[
                "anchor_qk_support_uses_phase0_relative_same_caption_dynamic_static_contrast"
            ]
        )
        self.assertFalse(
            contract["self_generated_rgb_or_latent_used_as_flow_matching_target"]
        )
        self.assertFalse(contract["scientific_claim_authorized"])
        self.assertEqual(contract["actual_distinct_target_iid_count"], 2)
        self.assertEqual(contract["actual_distinct_target_event_count"], 2)
        self.assertEqual(
            contract["training_manifest_order"],
            "variant_major_event_interleaved_v15",
        )

    def test_worker_and_launcher_are_hard_bound_and_fresh(self):
        worker = WORKER.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("expected_job=149363", worker)
        self.assertIn("expected_node=auh7-1b-gpu-312", worker)
        self.assertIn("job=149363", launcher)
        self.assertIn("node=auh7-1b-gpu-312", launcher)
        self.assertIn("run_stage 2\nrun_stage 8\nrun_stage 32", launcher)
        self.assertIn("train_online_anchor_attention_dynamic_static_v15.py", worker)
        self.assertIn("Bernini-R-1.3B-Diffusers-ff4c5d4", worker)
        self.assertIn("test ! -e \"$output\"", worker)
        self.assertNotIn("source-online-anchor-targetowned-qk-routed-teacher-v14r3", worker)
        self.assertNotIn("source-online-anchor-targetowned-qk-routed-teacher-v14r3", launcher)


if __name__ == "__main__":
    unittest.main()
