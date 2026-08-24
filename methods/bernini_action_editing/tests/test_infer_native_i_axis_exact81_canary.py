from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_i_axis_exact81_canary as runner
import native_i_axis_guidance as i_axis


ASSET = METHOD_ROOT / "assets/native_i_axis_exact81_core2_v1.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_trace(arm: str):
    actual_i_forward = arm in {"G-C", "G-W", "G-P", "G-S"}
    branches = ["none_uncond", "V_uncond", "VI_uncond", "VI_cond"]
    if arm in i_axis.GATED_ARMS:
        branches.append("I_uncond")
    steps = []
    for index in range(40):
        active = arm in i_axis.GATED_ARMS and index in i_axis.ACTIVE_STEP_INDICES
        native_hash = "a" * 64
        executed_hash = (
            "b" * 64
            if active and arm not in {"G-D"}
            else native_hash
        )
        calls = {
            name: (0 if name == "I_uncond" and arm == "G-D" else 1)
            for name in branches
        }
        steps.append(
            {
                "step_index": index,
                "gate_active": active,
                "branch_call_counts": calls,
                "branch_target_raw_sha256": {name: "c" * 64 for name in branches},
                "transformer_forward_count": 5 if actual_i_forward else 4,
                "original_scheduler_call_count": 1,
                "native_formula_exact_parity": True,
                "native_velocity_raw_sha256": native_hash,
                "executed_velocity_raw_sha256": executed_hash,
                "final_native_parity": index in (38, 39),
                "scheduler_received_original_model_output_object": (
                    not active or arm == "G-D"
                ),
                "i_axis_degenerate_alias_none": arm == "G-D",
                "target_tokens": 19_530,
            }
        )
    forwards = 200 if actual_i_forward else 160
    return {
        "step_count": 40,
        "expected_transformer_forwards": forwards,
        "observed_transformer_forwards": forwards,
        "steps": steps,
    }


class NativeIAxisRunnerContractTests(unittest.TestCase):
    def test_asset_and_two_presealed_seeds_load_exactly(self) -> None:
        digest = _file_sha256(ASSET)
        root, dog, path, observed = runner.load_cell_spec(
            ASSET.resolve(), expected_file_sha256=digest, cell_id="dog"
        )
        self.assertEqual(path, ASSET.resolve())
        self.assertEqual(observed, digest)
        self.assertEqual(root["contract"], runner._expected_spec_contract())
        self.assertEqual(dog["seeds"], [2026080825, 2026080925])
        _, human, _, _ = runner.load_cell_spec(
            ASSET.resolve(), expected_file_sha256=digest, cell_id="human"
        )
        self.assertEqual(human["seeds"], [2026080827, 2026080927])
        self.assertTrue(human["wrong_source_geometry_confound"])
        self.assertFalse(human["wrong_source_pure_identity_control"])

    def test_reference_controls_change_only_registered_list(self) -> None:
        correct = {index: object() for index in runner.ALL_CORRECT_REFERENCE_INDICES}
        wrong = {index: object() for index in runner.CORRECT_REFERENCE_INDICES}
        canonical = runner._references_for_arm("G-C", correct=correct, wrong=wrong)
        permuted = runner._references_for_arm("G-P", correct=correct, wrong=wrong)
        phase = runner._references_for_arm("G-S", correct=correct, wrong=wrong)
        wrong_selected = runner._references_for_arm("G-W", correct=correct, wrong=wrong)
        dropped = runner._references_for_arm("G-D", correct=correct, wrong=wrong)
        self.assertEqual(canonical, tuple(correct[index] for index in (0, 27, 53, 80)))
        self.assertEqual(permuted, (canonical[1], canonical[2], canonical[3], canonical[0]))
        self.assertEqual(sorted(map(id, canonical)), sorted(map(id, permuted)))
        self.assertEqual(phase, tuple(correct[index] for index in (10, 30, 50, 70)))
        self.assertEqual(wrong_selected, tuple(wrong[index] for index in (0, 27, 53, 80)))
        self.assertEqual(dropped, ())

    def test_exact40_receipt_gate_accepts_all_seven_arms(self) -> None:
        for arm in i_axis.ARM_ORDER:
            gate = runner.validate_exact40_trace(_valid_trace(arm), arm=arm)
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["step_count"], 40)
            self.assertEqual(gate["final_native_parity_indices"], [38, 39])
            self.assertEqual(len(gate["digest"]), 64)

    def test_exact40_receipt_gate_rejects_gate_digest_and_parity_drift(self) -> None:
        bad_gate = _valid_trace("G-C")
        bad_gate["steps"][32]["gate_active"] = True
        with self.assertRaises(runner.NativeIAxisCanaryError):
            runner.validate_exact40_trace(bad_gate, arm="G-C")
        bad_digest = _valid_trace("G-C")
        bad_digest["steps"][10]["branch_target_raw_sha256"]["I_uncond"] = "bad"
        with self.assertRaises(runner.NativeIAxisCanaryError):
            runner.validate_exact40_trace(bad_digest, arm="G-C")
        bad_final = _valid_trace("G-C")
        bad_final["steps"][38]["scheduler_received_original_model_output_object"] = False
        with self.assertRaises(runner.NativeIAxisCanaryError):
            runner.validate_exact40_trace(bad_final, arm="G-C")

    def test_cli_has_no_mask_pose_flow_or_optimizer_input(self) -> None:
        options = {
            option
            for action in runner.build_parser()._actions
            for option in action.option_strings
        }
        forbidden = {
            "--target-video", "--mask", "--pose", "--flow", "--track",
            "--trajectory", "--trainer", "--optimizer", "--adapter", "--lora",
        }
        self.assertTrue(forbidden.isdisjoint(options))

    def test_runtime_uses_atomic_directory_and_predecode_artifacts(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("prior._output_staging_directory(output_dir)", source)
        self.assertIn("prior._commit_output_transaction(staging=stage, final=output_dir)", source)
        self.assertIn("native._save_normalized_clean_latent_atomically", source)
        self.assertIn("native._save_outputs", source)
        self.assertIn("validate_exact40_trace(trace, arm=arm)", source)
        self.assertNotIn("normalized_guidance(", source)


if __name__ == "__main__":
    unittest.main()
