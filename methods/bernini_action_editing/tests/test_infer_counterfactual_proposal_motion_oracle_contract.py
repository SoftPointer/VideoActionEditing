from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

try:
    import torch  # noqa: F401
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_counterfactual_proposal_motion_oracle as oracle


SOURCE_PATH = METHOD_ROOT / "infer_counterfactual_proposal_motion_oracle.py"


class CPMRFullVideoOracleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _args(self):
        return oracle.build_parser().parse_args(
            [
                "--bernini-root", "/b",
                "--veomni-root", "/v",
                "--checkpoint", "/c",
                "--checkpoint-content-manifest", "/m",
                "--source-video", "/s.mp4",
                "--original-source-path", "/s.mp4",
                "--instruction", oracle.EXPECTED_INSTRUCTION,
                "--output-dir", "/out/cpmr",
                "--method-source-revision", "a" * 40,
                "--method-source-archive-sha256", "b" * 64,
            ]
        )

    def test_frozen_81_frame_three_arm_contract(self):
        self.assertEqual(oracle.ARM_ORDER, ("B0", "Z0", "C10"))
        self.assertEqual(oracle.ARM_GATES, {"B0": None, "Z0": 0.0, "C10": 0.10})
        self.assertEqual(oracle.PROPOSAL_SEED, 2027)
        self.assertEqual(oracle.RENDER_SEED, 2028)
        self.assertIn("EXPECTED_FRAMES = 81", self.source)
        self.assertNotIn("EXPECTED_FRAMES = 41", self.source)
        self.assertIn("proposal_action_noop_same_seed", self.source)
        self.assertIn("render_arms_same_seed", self.source)

    def test_interface_has_no_privileged_cli(self):
        actions = [
            value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for value in (
                [node.args[0].value]
                if node.args and isinstance(node.args[0], ast.Constant)
                else []
            )
        ]
        joined = " ".join(actions)
        for forbidden in ("target", "mask", "flow", "pose", "track", "trajectory", "reference"):
            self.assertNotIn(forbidden, joined)

    def test_proposals_use_same_seed_and_render_is_noop_based(self):
        self.assertEqual(self.source.count("seed=PROPOSAL_SEED"), 2)
        self.assertEqual(self.source.count("seed=RENDER_SEED"), 3)
        self.assertGreaterEqual(self.source.count("input_ids=noop_ids"), 4)
        self.assertIn("build_motion_carrier(action_field, noop_field)", self.source)
        self.assertIn(
            "transformer.patch_embedding(latent.to(dtype=transformer_dtype))",
            self.source,
        )

    def test_gate_zero_parity_and_active_difference_fail_closed(self):
        self.assertIn("z0_byte_exact = _tensor_bytes_equal(b0, z0)", self.source)
        self.assertIn("c10_differs = not _tensor_bytes_equal(z0, c10)", self.source)
        self.assertIn("Z0 differs bytewise from B0", self.source)
        self.assertIn("C10 is byte-identical to Z0", self.source)

    def test_receipt_does_not_claim_quality_or_training(self):
        for fragment in (
            '"scientific_claim": False',
            '"video_quality_claim": False',
            '"training_claim": False',
            '"lora_claim": False',
            '"full_video_engineering_claim": True',
            '"source_instruction_only_inference": True',
        ):
            self.assertIn(fragment, self.source)

    def test_validate_cli_is_exact(self):
        args = self._args()
        oracle.validate_cli(args)
        args.num_inference_steps = 39
        with self.assertRaises(oracle.CPMRFullVideoOracleError):
            oracle.validate_cli(args)


if __name__ == "__main__":
    unittest.main()
