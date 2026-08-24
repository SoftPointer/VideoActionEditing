from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch

import t_qmosaic_trajectory_intervention_v1 as tq


def _schedule() -> tuple[tuple[int, float], ...]:
    return tuple(zip(tq.PINNED_TIMESTEPS, tq.PINNED_SIGMAS, strict=True))


class _FakeOfficialScheduler:
    """Deterministic scheduler whose output is a new state object each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def step(
        self, *, step_index: int, timestep: int, state: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append((step_index, timestep))
        return (state + float(step_index + 1) / 100.0).detach()


def _initial_state() -> torch.Tensor:
    # Include signed zero so byte-identity tests are stronger than value tests.
    return torch.tensor(
        [[[-0.0, 0.25], [-0.5, 1.0]]], dtype=torch.float32
    ).detach()


def _run_unfinalized_capture() -> tuple[
    tq.ActualTrajectoryCaptureV1,
    torch.Tensor,
    _FakeOfficialScheduler,
]:
    capture = tq.ActualTrajectoryCaptureV1()
    scheduler = _FakeOfficialScheduler()
    state = _initial_state()
    schedule = _schedule()
    for index, (timestep, sigma) in enumerate(schedule):
        capture.before_scheduler_step(
            step_index=index,
            timestep=timestep,
            sigma=sigma,
            state=state,
        )
        raw = scheduler.step(step_index=index, timestep=timestep, state=state)
        returned = capture.after_scheduler_step(
            step_index=index,
            next_state=raw,
        )
        assert returned is raw
        state = returned
    return capture, state, scheduler


def _run_capture() -> tuple[
    tq.CapturedActualTrajectoryV1,
    torch.Tensor,
    _FakeOfficialScheduler,
]:
    capture, state, scheduler = _run_unfinalized_capture()
    return capture.finalize(), state, scheduler


def _vjps(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    count = math.prod(shape)
    rows = []
    for position in range(3):
        row = torch.linspace(
            0.2 + position,
            1.2 + position,
            count,
            dtype=torch.float32,
        ).reshape(shape)
        if position == 1:
            row = torch.flip(row, dims=(-1,)).contiguous()
        rows.append(row.detach())
    return tuple(rows)


def _plan() -> tq.TQMosaicTrajectoryInterventionV1:
    capture, _terminal, _scheduler = _run_capture()
    return tq.build_trajectory_intervention_v1(
        capture=capture,
        state_vjps=_vjps(capture.shape),
    )


def _run_replay(
    plan: tq.TQMosaicTrajectoryInterventionV1,
    *,
    sign: int,
) -> tuple[
    tq.TQMosaicTrajectoryReplayV1,
    torch.Tensor,
    _FakeOfficialScheduler,
    list[tuple[int, torch.Tensor, torch.Tensor]],
]:
    replay = plan.new_replay(sign=sign)
    scheduler = _FakeOfficialScheduler()
    state = _initial_state().clone()
    observations: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for index, (timestep, sigma) in enumerate(_schedule()):
        replay.before_scheduler_step(
            step_index=index,
            timestep=timestep,
            sigma=sigma,
            state=state,
        )
        raw = scheduler.step(step_index=index, timestep=timestep, state=state)
        returned = replay.after_scheduler_step(
            step_index=index,
            next_state=raw,
        )
        observations.append((index, raw, returned))
        state = returned
    return replay, state, scheduler, observations


class ActualTrajectoryCaptureTests(unittest.TestCase):
    def test_exact40_capture_binds_real_pre_step_states_and_schedule(self) -> None:
        captured, terminal, scheduler = _run_capture()
        self.assertEqual(
            scheduler.calls,
            [(index, row[0]) for index, row in enumerate(_schedule())],
        )
        initial = _initial_state()
        for index, state in zip(
            tq.CAPTURE_PRE_STEP_INDICES, captured.states, strict=True
        ):
            expected = initial
            for prior_index in range(index):
                expected = (expected + float(prior_index + 1) / 100.0).detach()
            self.assertTrue(torch.equal(state, expected))
        receipt = captured.receipt()
        self.assertEqual(receipt["evidence_tier"], "ENGINEERING_ONLY")
        self.assertEqual(receipt["exact_scheduler_calls"], 40)
        self.assertEqual(receipt["schedule_sha256"], tq.PINNED_SCHEDULE_SHA256)
        self.assertEqual(
            [row["pre_step_index"] for row in receipt["captured_states"]],
            [20, 28, 33],
        )
        self.assertEqual(receipt["terminal_state_sha256"], tq._tensor_sha256(
            terminal, label="test terminal"
        ))
        self.assertFalse(receipt["semantic_success_assessed"])
        self.assertFalse(receipt["training_update_authorized"])

    def test_capture_rejects_missing_repeated_out_of_order_and_disconnected_state(self) -> None:
        timestep0, sigma0 = _schedule()[0]
        initial = _initial_state()

        capture = tq.ActualTrajectoryCaptureV1()
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "out-of-order"):
            capture.before_scheduler_step(
                step_index=1, timestep=timestep0, sigma=sigma0, state=initial
            )

        capture = tq.ActualTrajectoryCaptureV1()
        capture.before_scheduler_step(
            step_index=0, timestep=timestep0, sigma=sigma0, state=initial
        )
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "repeated pre-step"):
            capture.before_scheduler_step(
                step_index=0, timestep=timestep0, sigma=sigma0, state=initial
            )
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "expected 0"):
            capture.after_scheduler_step(step_index=1, next_state=initial.clone())

        capture = tq.ActualTrajectoryCaptureV1()
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "without a pre-step"):
            capture.after_scheduler_step(step_index=0, next_state=initial)

        capture = tq.ActualTrajectoryCaptureV1()
        capture.before_scheduler_step(
            step_index=0, timestep=timestep0, sigma=sigma0, state=initial
        )
        raw = initial + 0.01
        returned = capture.after_scheduler_step(step_index=0, next_state=raw)
        timestep1, sigma1 = _schedule()[1]
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "exact intercepted"):
            capture.before_scheduler_step(
                step_index=1,
                timestep=timestep1,
                sigma=sigma1,
                state=returned.clone(),
            )
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "has 1 scheduler"):
            capture.finalize()

    def test_wrong_pinned_anchor_metadata_is_rejected(self) -> None:
        capture = tq.ActualTrajectoryCaptureV1()
        scheduler = _FakeOfficialScheduler()
        state = _initial_state()
        for index, (timestep, sigma) in enumerate(_schedule()[:21]):
            if index == 20:
                timestep += 1
            if index == 20:
                with self.assertRaisesRegex(
                    tq.TQMosaicTrajectoryError, "timestep differs"
                ):
                    capture.before_scheduler_step(
                        step_index=index,
                        timestep=timestep,
                        sigma=sigma,
                        state=state,
                    )
                break
            capture.before_scheduler_step(
                step_index=index,
                timestep=timestep,
                sigma=sigma,
                state=state,
            )
            raw = scheduler.step(step_index=index, timestep=timestep, state=state)
            state = capture.after_scheduler_step(step_index=index, next_state=raw)

    def test_wrong_nonanchor_official_metadata_is_rejected(self) -> None:
        capture = tq.ActualTrajectoryCaptureV1()
        scheduler = _FakeOfficialScheduler()
        state = _initial_state()
        timestep0, sigma0 = _schedule()[0]
        capture.before_scheduler_step(
            step_index=0, timestep=timestep0, sigma=sigma0, state=state
        )
        raw = scheduler.step(step_index=0, timestep=timestep0, state=state)
        state = capture.after_scheduler_step(step_index=0, next_state=raw)
        timestep1, sigma1 = _schedule()[1]
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "timestep differs"):
            capture.before_scheduler_step(
                step_index=1,
                timestep=timestep1 - 1,
                sigma=sigma1,
                state=state,
            )

    def test_terminal_capture_mutation_before_finalize_fails_closed(self) -> None:
        capture, terminal, _scheduler = _run_unfinalized_capture()
        with torch.no_grad():
            terminal.data.add_(1.0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "bytes changed"):
            capture.finalize()

    def test_captured_schedule_and_scalar_control_are_immutable_or_sealed(self) -> None:
        captured, _terminal, _scheduler = _run_capture()
        with self.assertRaises(TypeError):
            captured._schedule[0]["timestep"] = 1
        captured._initial_state_sha256 = "0" * 64
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "control changed"):
            captured.receipt()


class TrajectoryInterventionTests(unittest.TestCase):
    def test_fixed_joint_relative_l2_dose_and_closed_inputs(self) -> None:
        plan = _plan()
        measured = []
        for base, delta, target in zip(
            plan.capture.states,
            plan.deltas,
            tq.FIXED_RELATIVE_L2_DOSES,
            strict=True,
        ):
            ratio = float(
                torch.linalg.vector_norm(delta.double())
                / torch.linalg.vector_norm(base.double())
            )
            measured.append(ratio)
            self.assertAlmostEqual(ratio, target, places=7)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in measured)),
            tq.TOTAL_RELATIVE_L2_DOSE,
            places=7,
        )
        self.assertEqual(
            tuple(round(value, 12) for value in tq.FIXED_RELATIVE_L2_DOSES),
            (0.006666666667, 0.006666666667, 0.003333333333),
        )
        signature = inspect.signature(tq.build_trajectory_intervention_v1)
        self.assertEqual(tuple(signature.parameters), ("capture", "state_vjps"))
        receipt = plan.receipt()
        for field in (
            "seed_input_authorized",
            "dose_input_authorized",
            "arm_selection_authorized",
            "mask_input_authorized",
            "track_input_authorized",
            "pose_input_authorized",
            "optical_flow_input_authorized",
            "callback_authority",
            "training_update_authorized",
        ):
            self.assertFalse(receipt[field])

    def test_zero_sign_is_exact_no_clone_no_arithmetic_replay(self) -> None:
        plan = _plan()
        replay, terminal, scheduler, observations = _run_replay(plan, sign=0)
        self.assertEqual(len(scheduler.calls), 40)
        self.assertTrue(all(returned is raw for _index, raw, returned in observations))
        base_terminal = _run_capture()[1]
        self.assertTrue(torch.equal(terminal, base_terminal))
        self.assertEqual(
            tq._tensor_sha256(terminal, label="zero terminal"),
            tq._tensor_sha256(base_terminal, label="base terminal"),
        )
        receipt = replay.finalize()
        self.assertTrue(
            receipt["zero_sign_all_scheduler_outputs_returned_by_identity"]
        )
        self.assertEqual(receipt["scheduler_outputs_returned_by_identity_count"], 40)
        self.assertTrue(
            all(
                row["returned_original_object"]
                and row["zero_sign_original_bytes"]
                for row in receipt["injection_observations"]
            )
        )

    def test_plus_and_minus_inject_only_after_19_27_32(self) -> None:
        plan = _plan()
        for sign in (-1, 1):
            replay, _terminal, scheduler, observations = _run_replay(plan, sign=sign)
            self.assertEqual(len(scheduler.calls), 40)
            for index, raw, returned in observations:
                if index in tq.INJECT_AFTER_STEP_INDICES:
                    position = tq.INJECT_AFTER_STEP_INDICES.index(index)
                    expected = raw + float(sign) * plan.deltas[position]
                    self.assertIsNot(returned, raw)
                    self.assertTrue(torch.equal(returned, expected))
                else:
                    self.assertIs(returned, raw)
            receipt = replay.finalize()
            self.assertEqual(
                [row["inject_after_step_index"] for row in receipt["injection_observations"]],
                [19, 27, 32],
            )
            self.assertEqual(
                [row["next_pre_step_index"] for row in receipt["injection_observations"]],
                [20, 28, 33],
            )
            self.assertEqual(
                receipt["scheduler_outputs_returned_by_identity_count"], 37
            )

    def test_replay_rejects_missing_repeated_out_of_order_and_schedule_change(self) -> None:
        plan = _plan()
        timestep0, sigma0 = _schedule()[0]
        initial = _initial_state().clone()

        replay = plan.new_replay(sign=0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "out-of-order"):
            replay.before_scheduler_step(
                step_index=1, timestep=timestep0, sigma=sigma0, state=initial
            )

        replay = plan.new_replay(sign=0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "schedule differs"):
            replay.before_scheduler_step(
                step_index=0,
                timestep=timestep0 - 1,
                sigma=sigma0,
                state=initial,
            )

        replay = plan.new_replay(sign=0)
        replay.before_scheduler_step(
            step_index=0, timestep=timestep0, sigma=sigma0, state=initial
        )
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "repeated replay"):
            replay.before_scheduler_step(
                step_index=0, timestep=timestep0, sigma=sigma0, state=initial
            )
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "before post-step"):
            replay.finalize()

        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "exactly -1"):
            plan.new_replay(sign=2)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "exactly -1"):
            plan.new_replay(sign=True)

    def test_terminal_replay_mutation_and_control_rebinding_fail_closed(self) -> None:
        plan = _plan()
        replay, terminal, _scheduler, _observations = _run_replay(plan, sign=0)
        with torch.no_grad():
            terminal.add_(1.0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "changed"):
            replay.finalize()

        replay = plan.new_replay(sign=0)
        replay._sign = 1
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "control changed"):
            replay.before_scheduler_step(
                step_index=0,
                timestep=tq.PINNED_TIMESTEPS[0],
                sigma=tq.PINNED_SIGMAS[0],
                state=_initial_state().clone(),
            )

        plan = _plan()
        replay, terminal, _scheduler, _observations = _run_replay(plan, sign=0)
        replay.finalize()
        with torch.no_grad():
            terminal.data.add_(1.0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "bytes changed"):
            replay.receipt()

    def test_base_state_vjp_and_delta_mutations_fail_closed(self) -> None:
        captured, _terminal, _scheduler = _run_capture()
        with torch.no_grad():
            captured.states[0].add_(1.0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "changed"):
            captured.receipt()

        plan = _plan()
        # .data deliberately tries to evade the ordinary version-counter seal;
        # the terminal byte rehash must still reject it.
        with torch.no_grad():
            plan.deltas[0].data.add_(1.0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "bytes changed"):
            plan.receipt()

        plan = _plan()
        with torch.no_grad():
            plan.state_vjps[1].mul_(0.0)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "changed"):
            plan.new_replay(sign=1)

    def test_state_vjps_must_be_exactly_three_detached_fp32_matched_tensors(self) -> None:
        captured, _terminal, _scheduler = _run_capture()
        rows = list(_vjps(captured.shape))
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "exactly three"):
            tq.build_trajectory_intervention_v1(
                capture=captured, state_vjps=rows[:2]
            )
        wrong_shape = rows.copy()
        wrong_shape[1] = torch.ones(2, dtype=torch.float32)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "shape differs"):
            tq.build_trajectory_intervention_v1(
                capture=captured, state_vjps=wrong_shape
            )
        wrong_dtype = rows.copy()
        wrong_dtype[1] = rows[1].double()
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "detached dense"):
            tq.build_trajectory_intervention_v1(
                capture=captured, state_vjps=wrong_dtype
            )
        graph_row = rows.copy()
        graph_row[1] = rows[1].clone().requires_grad_(True)
        with self.assertRaisesRegex(tq.TQMosaicTrajectoryError, "detached dense"):
            tq.build_trajectory_intervention_v1(
                capture=captured, state_vjps=graph_row
            )


if __name__ == "__main__":
    unittest.main()
