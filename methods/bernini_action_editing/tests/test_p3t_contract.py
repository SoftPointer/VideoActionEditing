import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import p3t
import train_p3t_lora as trainer
import infer_p3t_lora as inference


def _row(iid="dog-1"):
    instruction = "Make the dog pick up the bone."
    edit_instruction = "Make the dog pick up the bone while preserving its identity."
    plan = {
        "iid": iid,
        "dynamic_subject_targets": [{
            "subject_id": "subject_01",
            "target_action_signature": "lower_grip_raise",
            "target_motion": "the dog lowers its head, grips the bone, then raises its head",
        }],
        "camera_target": {"target_motion": "the camera remains locked off"},
    }
    census = {
        "iid": iid,
        "dynamic_subjects": [
            {
                "subject_id": "subject_01",
                "stable_reference": "the same dog beside the bone",
            }
        ],
    }
    row = {
        "iid": iid,
        "edit_instruction": edit_instruction,
        "edit_instruction_sha256": hashlib.sha256(
            edit_instruction.encode()
        ).hexdigest(),
        "generation_instruction": instruction,
        "generation_instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "target_plan": plan,
        "target_plan_sha256": p3t.object_sha256(plan),
        "source_census": census,
    }
    row["row_digest"] = p3t.object_sha256(row)
    return row


class P3TManifestTests(unittest.TestCase):
    def _manifest(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "preview.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        digest = p3t.file_sha256(path)
        return path, digest

    def test_hash_bound_exact_iid_and_phase_plan(self):
        path, digest = self._manifest([_row()])
        manifest = p3t.PreviewManifest(path, expected_sha256=digest)
        bound = manifest.require("dog-1")
        self.assertIn("preserving its identity", bound.edit_instruction)
        self.assertIn("PREPARE phases 00-03", bound.compiled_plan)
        self.assertIn("EXECUTE phases 04-15", bound.compiled_plan)
        self.assertIn("SETTLE phases 16-20", bound.compiled_plan)
        self.assertIn("the same dog beside the bone", bound.compiled_plan)
        self.assertIn("the camera remains locked off", bound.compiled_plan)
        with self.assertRaisesRegex(p3t.P3TContractError, "absent"):
            manifest.require("wrong-iid")

    def test_bad_hash_duplicate_and_plan_iid_fail_closed(self):
        path, digest = self._manifest([_row()])
        with self.assertRaisesRegex(p3t.P3TContractError, "SHA-256"):
            p3t.PreviewManifest(path, expected_sha256="0" * 64)
        path2, digest2 = self._manifest([_row(), _row()])
        with self.assertRaisesRegex(p3t.P3TContractError, "duplicate"):
            p3t.PreviewManifest(path2, expected_sha256=digest2)
        wrong = _row()
        wrong["target_plan"]["iid"] = "other"
        wrong["target_plan_sha256"] = p3t.object_sha256(wrong["target_plan"])
        wrong.pop("row_digest")
        wrong["row_digest"] = p3t.object_sha256(wrong)
        path3, digest3 = self._manifest([wrong])
        with self.assertRaisesRegex(p3t.P3TContractError, "IID differs"):
            p3t.PreviewManifest(path3, expected_sha256=digest3)

    def test_cli_defaults_cross_only_and_fixed_integration(self):
        parser = trainer.build_parser()
        args = parser.parse_args([
            "--bernini-root", "/b", "--veomni-root", "/v", "--checkpoint", "/c",
            "--preprocessed-parquet-dir", "/p", "--dataset-summary", "/s",
            "--preview-manifest", "/m", "--expected-preview-manifest-sha256", "a" * 64,
            "--output", "/o", "--method-source-revision", "b" * 40,
            "--method-source-archive-sha256", "c" * 64,
        ])
        self.assertEqual(args.lora_scope, "cross_q_out")
        self.assertEqual((args.integration_steps, args.integration_flow_shift), (40, 5.0))
        self.assertEqual(args.source_restoration_loss_weight, 0.0)
        trainer.validate_cli(args)

    def test_privileged_or_route_options_fail_closed(self):
        base = [
            "--bernini-root", "/b", "--veomni-root", "/v", "--checkpoint", "/c",
            "--preprocessed-parquet-dir", "/p", "--dataset-summary", "/s",
            "--preview-manifest", "/m", "--expected-preview-manifest-sha256", "a" * 64,
            "--output", "/o", "--method-source-revision", "b" * 40,
            "--method-source-archive-sha256", "c" * 64,
        ]
        for extra in (
            ["--routing-jsonl", "/routes"],
            ["--motion-objective", "raw_delta"],
            ["--anchor-loss-weight", "0.1"],
        ):
            with self.subTest(extra=extra), self.assertRaises(trainer.DeltaTrainingError):
                trainer.validate_cli(trainer.build_parser().parse_args(base + extra))

    def test_inference_defaults_standard_and_oracle_is_explicit(self):
        parser = inference.build_parser()
        common = [
            "--bernini-root", "/b", "--veomni-root", "/v", "--checkpoint", "/c",
            "--adapter-checkpoint", "/a", "--source-video", "/s.mp4",
            "--instruction", "make the dog pick up the bone", "--output", "/o.mp4",
            "--method-source-revision", "b" * 40,
            "--method-source-archive-sha256", "c" * 64,
        ]
        args = parser.parse_args(common)
        self.assertEqual(args.sampling_mode, "standard")
        inference.validate_cli(args)
        args = parser.parse_args(common + ["--compiled-plan-text", "oracle phases"])
        with self.assertRaisesRegex(inference.DeltaInferenceError, "supplied together"):
            inference.validate_cli(args)
        generic = p3t.compile_generic_phase_wrapper("pick up the bone")
        self.assertIn("EXECUTE phases 04-15", generic)


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
class P3TTensorTests(unittest.TestCase):
    def test_projection_decomposition_and_interval_terminal_emphasis(self):
        import torch
        field = torch.arange(42.0).reshape(1, 21, 2)
        projected = p3t.temporal_project(field)
        complement = p3t.temporal_complement(field)
        self.assertTrue(torch.allclose(projected + complement, field))
        self.assertTrue(torch.allclose(projected.reshape(1, 21, 1, 2).mean(1), torch.zeros(1, 1, 2), atol=1e-5))
        weights = p3t.interval_weight(torch.tensor([0.99, 0.01]))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_restoration_corruptions_are_deterministic_and_shape_safe(self):
        import torch
        clean = torch.ones(1, 2, 21, 8, 8)
        for kind in ("speed", "tube", "cube"):
            a = p3t.deterministic_source_corruption(clean, kind=kind, seed=7)
            b = p3t.deterministic_source_corruption(clean, kind=kind, seed=7)
            self.assertEqual(a.shape, clean.shape)
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(clean, torch.ones_like(clean)))


if __name__ == "__main__":
    unittest.main()
