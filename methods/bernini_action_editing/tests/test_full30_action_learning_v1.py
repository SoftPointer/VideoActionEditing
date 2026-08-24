#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

import full30_action_learning_v1 as core


def _rows() -> tuple[core.ActionPairRow, ...]:
    rows = []
    for source_index in range(64):
        teacher = f"teacher-{source_index // 8:02d}"
        source = f"source-{source_index:03d}"
        for branch in core.BRANCHES:
            rows.append(
                core.ActionPairRow(
                    row_id=f"{source}--{branch}",
                    source_id=source,
                    branch=branch,
                    teacher_cell_id=teacher,
                )
            )
    return tuple(rows)


class ScheduleTests(unittest.TestCase):
    def test_exact_formal_schedule(self) -> None:
        first = core.build_formal_schedule_v1(_rows(), run_seed=20260815)
        second = core.build_formal_schedule_v1(tuple(reversed(_rows())), run_seed=20260815)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1280)
        self.assertEqual({item.update for item in first}, set(range(160)))
        self.assertEqual(Counter(item.sigma_index for item in first), Counter({4: 214, 12: 214, 20: 214, 28: 214, 35: 212, 38: 212}))
        self.assertEqual(Counter(item.row.branch for item in first), Counter({"action": 640, "incomplete": 640}))
        self.assertTrue(all(sum(item.row.row_id == row.row_id for item in first) == 10 for row in _rows()))
        self.assertEqual(len({item.noise_seed for item in first}), 640)
        for update in range(160):
            group = first[update * 8 : (update + 1) * 8]
            self.assertEqual(
                [(item.microbatch, item.dp_rank) for item in group],
                [(micro, dp) for micro in range(4) for dp in range(2)],
            )
            self.assertEqual(Counter(item.row.branch for item in group), Counter({"action": 4, "incomplete": 4}))
            for microbatch in range(4):
                pair = [item for item in group if item.microbatch == microbatch]
                self.assertEqual(len({item.row.source_id for item in pair}), 1)
                self.assertEqual({item.row.branch for item in pair}, {"action", "incomplete"})
                self.assertEqual(len({item.sigma_index for item in pair}), 1)
                self.assertEqual(len({item.noise_seed for item in pair}), 1)

    def test_schedule_rejects_incomplete_pairing(self) -> None:
        rows = list(_rows())
        rows[-1] = core.ActionPairRow(
            row_id=rows[-1].row_id,
            source_id=rows[-1].source_id,
            branch="action",
            teacher_cell_id=rows[-1].teacher_cell_id,
        )
        with self.assertRaises(core.Full30ActionLearningError):
            core.build_formal_schedule_v1(rows, run_seed=1)

    def test_physical_branch_counts(self) -> None:
        self.assertEqual(core.physical_branch_evaluations_per_update("action-only"), 24)
        self.assertEqual(core.physical_branch_evaluations_per_update("action+retain"), 32)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable")
class TensorTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch

    def test_psiout_shape_causal_boundary_and_nuisance_projection(self) -> None:
        torch = self.torch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        delta = torch.randn((2, 16, 21, 6, 8), generator=generator, dtype=torch.float32).contiguous()
        camera_delta = torch.randn((2, 16, 21, 6, 8), generator=generator, dtype=torch.float32).contiguous()
        appearance_delta = torch.randn((2, 16, 21, 6, 8), generator=generator, dtype=torch.float32).contiguous()
        raw = core.psiout_raw_v1(delta)
        camera = core.psiout_raw_v1(camera_delta)
        appearance = core.psiout_raw_v1(appearance_delta)
        packet = core.build_nuisance_packet_v1(camera, appearance)
        projected = core.project_nuisances_v1(raw, packet)
        self.assertEqual(tuple(projected.shape), (2, 21, 32))
        self.assertTrue(torch.equal(projected[:, 0], torch.zeros_like(projected[:, 0])))
        flat = projected.reshape(2, -1)
        self.assertTrue(torch.allclose((flat * packet.camera_unit.reshape(2, -1)).sum(1), torch.zeros(2), atol=2e-5, rtol=0))
        self.assertTrue(torch.allclose((flat * packet.appearance_unit.reshape(2, -1)).sum(1), torch.zeros(2), atol=2e-5, rtol=0))

    def test_loss_is_per_record_and_penalizes_zero_amplitude(self) -> None:
        torch = self.torch
        teacher_raw = torch.zeros((2, 21, 32), dtype=torch.float32)
        teacher_raw[0, 1, 0] = 2.0
        teacher_raw[1, 2, 1] = -3.0
        teacher = core.teacher_unit_v1(teacher_raw.contiguous()).detach()
        floor = torch.tensor([2.0, 3.0], dtype=torch.float32)
        exact = core.paired_action_loss_v1(teacher * floor[:, None, None], teacher, floor)
        near_zero = core.paired_action_loss_v1(torch.zeros_like(teacher), teacher, floor)
        self.assertLess(float(exact.total.item()), 1e-6)
        self.assertGreater(float(near_zero.total.item()), 10.0)

        swapped = core.paired_action_loss_v1(
            torch.stack((teacher[1] * floor[0], teacher[0] * floor[1])),
            teacher,
            floor,
        )
        self.assertGreater(float(swapped.direction_mean.item()), 0.9)

    def test_action_first_update_projects_conflict_and_caps_noop(self) -> None:
        torch = self.torch
        action = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)
        noop = torch.tensor([-100.0, -100.0, 100.0, 30.0], dtype=torch.float32)
        moment = torch.zeros_like(action)
        result = core.action_first_update_v1(action, moment, noop_gradient=noop)
        self.assertLess(result.conflict_dot_before, 0.0)
        self.assertGreaterEqual(
            result.conflict_dot_after,
            -result.projection_rounding_tolerance,
        )
        self.assertGreater(result.actual_action_descent_dot, 0.0)
        self.assertLessEqual(result.noop_cap_factor, 1.0)
        self.assertTrue(torch.isfinite(result.descent_direction).all())

    def test_degenerate_nuisance_and_action_gradient_fail(self) -> None:
        torch = self.torch
        zero = torch.zeros((1, 21, 32), dtype=torch.float32)
        with self.assertRaises(core.Full30ActionLearningError):
            core.build_nuisance_packet_v1(zero, zero)
        with self.assertRaises(core.Full30ActionLearningError):
            core.action_first_update_v1(
                torch.zeros(4, dtype=torch.float32),
                torch.zeros(4, dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
