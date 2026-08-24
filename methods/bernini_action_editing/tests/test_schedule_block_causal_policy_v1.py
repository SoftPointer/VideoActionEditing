from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import schedule_block_causal_policy_v1 as policy  # noqa: E402


class ScheduleBlockCausalPolicyTests(unittest.TestCase):
    def test_default_policy_is_exact_diagnostic_grid(self) -> None:
        value = policy.default_policy()
        receipt = value.receipt()
        self.assertEqual(value.cell_count, 16)
        self.assertEqual(receipt["schedule_indices"], [16, 29, 35, 38])
        self.assertEqual(
            receipt["block_bands"],
            {
                "early": list(range(0, 8)),
                "early_middle": list(range(8, 16)),
                "late_middle": list(range(16, 23)),
                "late": list(range(23, 30)),
            },
        )
        self.assertIs(receipt["low_sigma_action_gate_forced_zero"], False)
        self.assertIs(receipt["optimizer_authorized"], False)
        self.assertIs(receipt["parameter_update_authorized"], False)
        self.assertIs(receipt["runtime_integration_verified"], False)
        self.assertIs(receipt["decoded_intervention_executed"], False)
        self.assertEqual(
            receipt["registered_grid_sha256"], policy.REGISTERED_GRID_SHA256
        )
        self.assertEqual(
            [row["timestep_int64"] for row in receipt["schedule_cells"]],
            [882, 655, 418, 211],
        )
        self.assertEqual(
            [row["sigma_float32_be_hex"] for row in receipt["schedule_cells"]],
            ["3f61ed37", "3f27d446", "3ed6539a", "3e58b351"],
        )
        self.assertEqual(len(receipt["schedule_cells"]), 4)
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(policy.object_sha256(unsigned), digest)

    def test_schedule_and_block_axes_are_not_collapsed(self) -> None:
        receipt = policy.default_policy().receipt()
        self.assertIs(receipt["diffusion_time_is_not_block_depth"], True)
        self.assertIs(receipt["same_transformer_executes_every_schedule_step"], True)
        self.assertIs(receipt["block_motion_specialization_claimed"], False)

    def test_policy_rejects_ambiguous_or_trainable_mutation(self) -> None:
        mutations = (
            {"schedule_indices": (16, 16, 38)},
            {"schedule_indices": (38, 16)},
            {"schedule_indices": (40,)},
            {"schedule_indices": (0,)},
            {"schedule_indices": (16, 29, 35)},
            {
                "block_bands": (
                    ("early", tuple(range(10))),
                    ("late", tuple(range(20, 30))),
                )
            },
            {
                "block_bands": (
                    ("odd", tuple(range(1, 30, 2))),
                    ("even", tuple(range(0, 30, 2))),
                )
            },
            {"optimizer_authorized": True},
            {"parameter_update_authorized": True},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(policy.ScheduleBlockPolicyError):
                    policy.ScheduleBlockCausalPolicy(**mutation)


if __name__ == "__main__":
    unittest.main()
