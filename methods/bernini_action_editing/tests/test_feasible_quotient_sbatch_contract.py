#!/usr/bin/env python3
"""Static launch contracts for formal RS-FQT v8 train and inference jobs."""

from pathlib import Path
import re
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = METHOD_ROOT / "scripts"
TRAIN = SCRIPTS / "auh_train_feasible_quotient_v8_pilot.sbatch"
INFER = SCRIPTS / "auh_infer_trained_feasible_quotient_v8.sbatch"
MANIFEST = (
    METHOD_ROOT
    / "audits"
    / "bernini_r13_ff4c5d4_checkpoint.sha256"
)
MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)


class FeasibleQuotientSbatchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = TRAIN.read_text(encoding="utf-8")
        cls.infer = INFER.read_text(encoding="utf-8")

    def test_both_jobs_are_exact_four_gpu_and_revision_archived(self):
        for source in (self.train, self.infer):
            self.assertIn("#SBATCH --gres=gpu:mi210:4", source)
            self.assertIn("--nproc_per_node=4", source)
            self.assertIn("git -C \"${source_repository}\" archive", source)
            self.assertIn("archive bytes are not the declared git revision", source)
            self.assertIn("BERNINI_V8_SOURCE_ARCHIVE", source)
            self.assertIn("BERNINI_V8_SOURCE_REVISION", source)
            self.assertIn("BERNINI_V8_SOURCE_REPOSITORY", source)
            self.assertNotIn("BERNINI_V8_TRAIN_SOURCE_", source)

    def test_training_is_exact81_exact40_and_finalized_fail_closed(self):
        self.assertIn("--num-frames 81", self.train)
        self.assertIn("--max-steps 40", self.train)
        self.assertIn("--save-every 40", self.train)
        self.assertIn("--teacher-mode paired_displacement_only", self.train)
        self.assertIn("finalize_feasible_quotient_checkpoint.py", self.train)
        self.assertIn("post_save_strict_reload_complete", self.train)
        self.assertIn("accepted_sigma_schedule_indices", self.train)

    def test_inference_is_trained_v8_source_instruction_only(self):
        self.assertIn(
            'operator_mode="v8_reconstruction_section_feasible_quotient_transport"',
            self.infer,
        )
        self.assertIn("--source-video \"${source_video}\"", self.infer)
        self.assertIn("--instruction \"${instruction}\"", self.infer)
        self.assertIn("v8_projection_consistent_training_matched_main", self.infer)
        for forbidden_option in (
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--first-frame",
            "--target-video",
        ):
            self.assertNotIn(forbidden_option, self.infer)

    def test_checkpoint_contents_are_verified_not_just_declared(self):
        lines = MANIFEST.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 23)
        self.assertTrue(
            all(
                re.fullmatch(r"[0-9a-f]{64}  \./[^\n]+", line)
                for line in lines
            )
        )
        import hashlib

        self.assertEqual(
            hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            MANIFEST_SHA256,
        )
        for source in (self.train, self.infer):
            self.assertIn(MANIFEST_SHA256, source)
            self.assertIn("sha256sum --strict --status -c", source)
            self.assertIn("checkpoint non-cache file set differs", source)
        # Training verifies before torchrun and again immediately before the
        # post-save finalizer signs the adapter ready.
        self.assertEqual(
            self.train.count('validate_checkpoint_content "${checkpoint_manifest}"'),
            2,
        )


if __name__ == "__main__":
    unittest.main()
