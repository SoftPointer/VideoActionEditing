from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_schedule_block_source_edge_localization_v2 as runner  # noqa: E402
import schedule_block_causal_policy_v1 as policy  # noqa: E402
import schedule_block_source_edge_ablation_v2 as edge  # noqa: E402


AUTHORING = METHOD_ROOT / "assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
SOURCE = METHOD_ROOT / "infer_schedule_block_source_edge_localization_v2.py"


class DecodedSourceEdgeRunnerTest(unittest.TestCase):
    def test_authoring_sha_and_dog_human_bindings_are_exact(self) -> None:
        observed = hashlib.sha256(AUTHORING.read_bytes()).hexdigest()
        self.assertEqual(observed, runner.AUTHORING_SHA256)
        for family, expected in runner.FAMILY_BINDINGS.items():
            _, correct, wrong, path, digest = runner.load_family_authority(
                AUTHORING,
                expected_sha256=runner.AUTHORING_SHA256,
                family=family,
            )
            self.assertEqual(path, AUTHORING.resolve())
            self.assertEqual(digest, observed)
            self.assertEqual(correct["iid"], expected["correct_iid"])
            self.assertEqual(wrong["iid"], expected["wrong_iid"])
            self.assertEqual(correct["action_family_id"], wrong["action_family_id"])

    def test_full_plan_has_104_outputs_per_family(self) -> None:
        schedules = policy.REGISTERED_SCHEDULE_INDICES
        bands = tuple(name for name, _ in policy.REGISTERED_BLOCK_BANDS)
        plan = runner.build_plan(schedules, bands)
        self.assertEqual(len(plan), 104)
        self.assertEqual(
            [row["key"] for row in plan[:6]],
            [f"native-correct-{name}" for name in edge.TEXT_BRANCHES],
        )
        self.assertEqual(plan[6]["key"], "native-wrong-owner-forward")
        self.assertEqual(plan[7]["key"], "parity-source-on-s16-early-forward")
        off = [row for row in plan if row["hook"] == "source-off"]
        self.assertEqual(len(off), 96)
        self.assertEqual(
            {(row["schedule_index"], row["band_name"], row["text_branch"]) for row in off},
            {
                (schedule, band, branch)
                for schedule in schedules
                for band in bands
                for branch in edge.TEXT_BRANCHES
            },
        )

    def test_shard_plan_retains_global_baselines_but_claims_only_requested_cells(self) -> None:
        plan = runner.build_plan((29, 35), ("early_middle", "late"))
        self.assertEqual(len(plan), 6 + 2 + 2 * 2 * 6)
        off = [row for row in plan if row["role"] == "source_edge_off_cell"]
        self.assertEqual({row["schedule_index"] for row in off}, {29, 35})
        self.assertEqual({row["band_name"] for row in off}, {"early_middle", "late"})

    def test_prompts_are_six_distinct_typed_controls(self) -> None:
        for family in runner.FAMILY_BINDINGS:
            _, correct, _, _, _ = runner.load_family_authority(
                AUTHORING,
                expected_sha256=runner.AUTHORING_SHA256,
                family=family,
            )
            captions = runner.branch_captions(correct)
            self.assertEqual(tuple(captions), edge.TEXT_BRANCHES)
            self.assertEqual(len(set(captions.values())), 6)
            self.assertIn(correct["scene_caption"], captions["forward"])
            self.assertIn(correct["camera_caption"], captions["forward"])

    def test_runtime_has_no_optimizer_reward_scalar_or_selection(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("backward(", source)
        self.assertNotIn("feature_scorer", source)
        self.assertNotIn("best_of", source)
        self.assertIn('"reward_computed": False', source)
        self.assertIn('"ranking_performed": False', source)
        self.assertIn('"selection_performed": False', source)
        self.assertIn('"optimizer_present": False', source)

    def test_runtime_receipt_binds_registered_policy_and_trace_digests(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"registered_schedule_block_policy": policy.default_policy().receipt()', source)
        self.assertIn('object_sha256(native_unsigned) != native_digest', source)
        self.assertIn('object_sha256(edge_unsigned) != edge_digest', source)

    def test_parse_axes_fail_closed_and_keep_registered_order(self) -> None:
        self.assertEqual(runner._parse_schedules("16,35"), (16, 35))
        self.assertEqual(runner._parse_bands("early,late"), ("early", "late"))
        for value in ("", "16,16", "35,16", "17"):
            with self.assertRaises(runner.DecodedSourceEdgeError):
                runner._parse_schedules(value)
        for value in ("", "early,early", "late,early", "middle"):
            with self.assertRaises(runner.DecodedSourceEdgeError):
                runner._parse_bands(value)


if __name__ == "__main__":
    unittest.main()
