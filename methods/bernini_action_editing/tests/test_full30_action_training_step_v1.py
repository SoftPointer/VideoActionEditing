#!/usr/bin/env python3
"""CPU-hostile tests for the strict full30 single-update orchestrator."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:  # pragma: no cover - base host intentionally lacks torch
    torch = None

if torch is not None:
    import full30_action_learning_v1 as learning
    import full30_action_optimizer_v1 as optimizer_core
    import full30_action_runtime_v1 as runtime_core
    import full30_action_training_step_v1 as step_core


def _optimizer_payload_without_numpy(value):
    contiguous = value.detach().contiguous().cpu()
    expected = int(contiguous.numel()) * int(contiguous.element_size())
    raw = bytes(contiguous.untyped_storage())
    if len(raw) != expected:
        contiguous = contiguous.clone(memory_format=torch.contiguous_format)
        raw = bytes(contiguous.untyped_storage())
    if len(raw) != expected:
        raise AssertionError("test tensor storage byte count differs")
    return raw


@unittest.skipIf(torch is None, "PyTorch CPU is unavailable")
class Full30ActionTrainingStepV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rows = []
        for source_index in range(64):
            source_id = f"source-{source_index:02d}"
            cell = f"cell-{source_index // 8}"
            for branch in learning.BRANCHES:
                rows.append(
                    learning.ActionPairRow(
                        row_id=f"{source_id}-{branch}",
                        source_id=source_id,
                        branch=branch,
                        teacher_cell_id=cell,
                    )
                )
        cls.schedule = learning.build_formal_schedule_v1(rows, run_seed=17)

    def setUp(self) -> None:
        self.parameters = {
            "blocks.0.attn1.to_q.lora_A.default.weight": torch.nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32)
            ),
            "blocks.0.attn1.to_q.lora_B.default.weight": torch.nn.Parameter(
                torch.tensor([0.0], dtype=torch.float32)
            ),
            "patch.role_embedding": torch.nn.Parameter(
                torch.tensor([0.04], dtype=torch.float32)
            ),
            "patch.source_delta.weight": torch.nn.Parameter(
                torch.tensor([0.02], dtype=torch.float32)
            ),
            "patch.target_delta.weight": torch.nn.Parameter(
                torch.tensor([0.03], dtype=torch.float32)
            ),
        }
        self.optimizer = optimizer_core.Full30ActionFirstOptimizerV1(
            self.parameters, max_chunk_elements=2
        )
        camera = torch.zeros((1, 21, 32), dtype=torch.float32)
        appearance = torch.zeros_like(camera)
        teacher = torch.zeros_like(camera)
        camera.reshape(-1)[1] = 1.0
        appearance.reshape(-1)[2] = 1.0
        teacher.reshape(-1)[3] = 1.0
        self.nuisance = learning.build_nuisance_packet_v1(camera, appearance)
        self.teacher_unit = learning.teacher_unit_v1(teacher).detach().contiguous()
        self.collective_events: list[tuple[str, str, int]] = []
        self.consensus_events = 0
        self.scalar_events = 0

    @staticmethod
    def _basis() -> torch.Tensor:
        temporal = torch.arange(21, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
        channels = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1, 1)
        spatial = torch.tensor(
            [[-1.0, 0.25], [-0.5, 1.25]], dtype=torch.float32
        ).reshape(1, 1, 1, 2, 2)
        return (
            temporal / 21.0 + channels / 32.0 + spatial
        ).expand(1, 16, 21, 2, 2).contiguous()

    def _local_records(self, *, rank: int, update: int):
        dp_rank = rank // 4
        result = []
        for microbatch in range(4):
            scheduled = self.schedule[update * 8 + microbatch * 2 + dp_rank]
            branch_authority = hashlib.sha256(
                f"branch:{scheduled.row.row_id}".encode("ascii")
            ).hexdigest()
            noop_authority = hashlib.sha256(b"noop").hexdigest()
            runtime_record = runtime_core.Full30ActionRecordV1(
                row_id=scheduled.row.row_id,
                source_iid=scheduled.row.source_id,
                branch=scheduled.row.branch,
                source_patches=torch.zeros((21, 16, 1, 2, 2), dtype=torch.float32),
                noisy_target_patches=torch.ones(
                    (21, 16, 1, 2, 2), dtype=torch.float32
                ),
                rotary_embs=torch.zeros((1, 1, 42, 64), dtype=torch.complex64),
                timestep=torch.tensor([float(scheduled.sigma_index)]),
                spatial_shape=(1, 16, 21, 2, 2),
                branch_condition=runtime_core.ConditionBindingV1(
                    role="branch",
                    authority_sha256=branch_authority,
                    condition=object(),
                ),
                noop_condition=runtime_core.ConditionBindingV1(
                    role="noop",
                    authority_sha256=noop_authority,
                    condition=object(),
                ),
            )
            objective = step_core.seal_record_objective_authority_v1(
                row_id=scheduled.row.row_id,
                source_id=scheduled.row.source_id,
                branch=scheduled.row.branch,
                teacher_cell_id=scheduled.row.teacher_cell_id,
                sigma_index=scheduled.sigma_index,
                noise_seed=scheduled.noise_seed,
                teacher_unit=self.teacher_unit.detach().clone().contiguous(),
                minimum_amplitude=torch.tensor([0.05], dtype=torch.float32),
                minimum_amplitude_float32_le_sha256=hashlib.sha256(
                    struct.pack("<f", 0.05)
                ).hexdigest(),
                minimum_amplitude_bundle_digest="c" * 64,
                minimum_amplitude_calibration_id="calibration-cell-branch",
                nuisance_packet=learning.NuisancePacket(
                    camera_unit=self.nuisance.camera_unit.detach().clone().contiguous(),
                    appearance_unit=(
                        self.nuisance.appearance_unit.detach().clone().contiguous()
                    ),
                    camera_norm=self.nuisance.camera_norm.detach().clone().contiguous(),
                    appearance_norm=(
                        self.nuisance.appearance_norm.detach().clone().contiguous()
                    ),
                    appearance_residual_ratio=(
                        self.nuisance.appearance_residual_ratio.detach().clone().contiguous()
                    ),
                ),
                noop_target_velocity=(self._basis() * 2.0).detach().contiguous(),
                data_teacher_authority_manifest_sha256="a" * 64,
                amplitude_authority_manifest_sha256="b" * 64,
            )
            result.append(
                step_core.Full30LocalMicroRecordV1(
                    scheduled=scheduled,
                    runtime_record=runtime_record,
                    objective=objective,
                )
            )
        return tuple(result)

    class FakeRuntime:
        def __init__(self, owner, *, replay_mismatch=False, nonfinite=False, missing=None):
            self.owner = owner
            self.replay_mismatch = replay_mismatch
            self.nonfinite = nonfinite
            self.missing = missing
            self.events = []

        def _scalar(self):
            values = self.owner.parameters
            a = values["blocks.0.attn1.to_q.lora_A.default.weight"]
            b = values["blocks.0.attn1.to_q.lora_B.default.weight"]
            source = values["patch.source_delta.weight"]
            target = values["patch.target_delta.weight"]
            role = values["patch.role_embedding"]
            terms = {
                "a": a,
                "b": b,
                "source": source,
                "target": target,
                "role": role,
            }
            if self.missing is not None:
                terms[self.missing] = terms[self.missing].detach()
            return (
                terms["a"] * terms["b"]
                + terms["source"]
                + 1.1 * terms["target"]
                + 1.2 * terms["role"]
            )

        def _receipt(self, phase, record):
            binding_value = {
                "row_id": record.row_id,
                "source_iid": record.source_iid,
                "branch": record.branch,
            }
            if phase == "noop" and self.replay_mismatch:
                binding_value["tamper"] = True
            binding = runtime_core.object_sha256(binding_value)
            slots = (
                ["trainable_branch", "frozen_noop", "frozen_branch"]
                if phase == "action"
                else ["trainable_noop"]
            )
            value = {
                "schema_version": runtime_core.PHASE_RECEIPT_SCHEMA_VERSION,
                "runtime_schema_version": runtime_core.SCHEMA_VERSION,
                "phase": phase,
                "row_id": record.row_id,
                "source_iid": record.source_iid,
                "branch": record.branch,
                "input_binding_digest": binding,
                "phase_evaluation_plan": {
                    "global_batch": 8,
                    "evaluations_per_record": 3 if phase == "action" else 1,
                    "global_physical_evaluation_count": (
                        24 if phase == "action" else 8
                    ),
                    "slots": slots,
                },
            }
            return {**value, "receipt_digest": runtime_core.object_sha256(value)}

        def execute_action_phase(self, *, record):
            self.events.append(("action", record.row_id))
            basis = Full30ActionTrainingStepV1Tests._basis()
            trainable = (self._scalar() * basis).contiguous()
            if self.nonfinite:
                trainable = trainable.clone()
                trainable.reshape(-1)[0] = float("nan")
            return runtime_core.Full30ActionPhaseOutputsV1(
                trainable_branch_velocity=trainable,
                frozen_noop_velocity=torch.zeros_like(basis).detach().contiguous(),
                frozen_branch_velocity=(basis * 5.0).detach().contiguous(),
                receipt=self._receipt("action", record),
            )

        def execute_noop_phase(self, *, record):
            self.events.append(("noop", record.row_id))
            value = (self._scalar() * Full30ActionTrainingStepV1Tests._basis()).contiguous()
            return runtime_core.Full30NoopPhaseOutputsV1(
                trainable_noop_velocity=value,
                receipt=self._receipt("noop", record),
            )

    def _gradient_callback(self, *, mode=None):
        def callback(request):
            self.collective_events.append(
                (request.phase, request.scope, request.sequence_index)
            )
            if mode == "mixed" and request.sequence_index == 0:
                name = next(iter(request.gradients))
                request.gradients[name] = request.gradients[name].double()
            expected = dict(
                step_core.expected_gradient_collective_receipt_v1(request)
            )
            if mode == "wrong-order" and request.sequence_index == 0:
                expected["scope"] = "DP2"
            return expected

        return callback

    def _world_callback(self, *, mismatch=False):
        def callback(request):
            self.consensus_events += 1
            value = dict(step_core.expected_world_consensus_receipt_v1(request))
            if mismatch:
                value["all_equal"] = False
            return value

        return callback

    def _scalar_callback(self, *, fail_second=False):
        def callback(value):
            self.scalar_events += 1
            if self.scalar_events > 2:
                raise AssertionError("external scalar callback called after commit")
            if fail_second and self.scalar_events == 2:
                raise RuntimeError("injected second scalar collective failure")
            value.mul_(8.0)
            return None

        return callback

    def _run(
        self,
        *,
        arm="action+retain",
        rank=0,
        update=None,
        runtime=None,
        gradient_mode=None,
        consensus_mismatch=False,
        scalar_fail_second=False,
        records=None,
    ):
        if update is None:
            update = self.optimizer.update_count
        if runtime is None:
            runtime = self.FakeRuntime(self)
        if records is None:
            records = self._local_records(rank=rank, update=update)
        self.scalar_events = 0
        with mock.patch.object(
            optimizer_core,
            "_tensor_payload_bytes",
            side_effect=_optimizer_payload_without_numpy,
        ):
            result = step_core.execute_full30_action_training_step_v1(
                runtime=runtime,
                optimizer=self.optimizer,
                arm=arm,
                rank=rank,
                update_index=update,
                full_schedule=self.schedule,
                local_records=records,
                gradient_mean=self._gradient_callback(mode=gradient_mode),
                world_consensus=self._world_callback(
                    mismatch=consensus_mismatch
                ),
                optimizer_all_reduce_sum=self._scalar_callback(
                    fail_second=scalar_fail_second
                ),
                test_only_allow_small_capacity=True,
            )
        return result, runtime

    def test_main_u1_then_u2_strict_two_phase_roundtrip(self) -> None:
        first, runtime = self._run()
        self.assertEqual(self.optimizer.update_count, 1)
        self.assertEqual([event[0] for event in runtime.events], ["action"] * 4 + ["noop"] * 4)
        self.assertEqual(
            self.collective_events,
            [
                ("action", "SP4", 0),
                ("action", "DP2", 1),
                ("noop", "SP4", 2),
                ("noop", "DP2", 3),
            ],
        )
        self.assertEqual(first.receipt["runtime"]["formal_physical_evaluation_count"], 32)
        self.assertEqual(self.consensus_events, 1)
        self.assertEqual(self.scalar_events, 2)
        self.assertEqual(
            first.receipt["gradients"]["coverage_gate"]["gate_stage"],
            "u1-B-plus-typed",
        )
        self.assertEqual(len(first.receipt["records"]), 4)
        self.assertIn("teacher_unit_sha256", first.receipt["records"][0])
        step_core.canonical_receipt_bytes(first.receipt)

        self.collective_events.clear()
        second, _runtime = self._run(update=1)
        self.assertEqual(self.optimizer.update_count, 2)
        self.assertEqual(
            second.receipt["gradients"]["coverage_gate"]["gate_stage"],
            "u2+-A-B-plus-typed",
        )
        self.assertGreater(
            second.receipt["gradients"]["coverage_gate"]["lora_A_min_norm"],
            1.0e-12,
        )

    def test_action_only_has_no_noop_graph_or_gradient(self) -> None:
        result, runtime = self._run(arm="action-only")
        self.assertEqual([event[0] for event in runtime.events], ["action"] * 4)
        self.assertEqual(
            self.collective_events,
            [("action", "SP4", 0), ("action", "DP2", 1)],
        )
        self.assertEqual(result.receipt["runtime"]["formal_physical_evaluation_count"], 24)
        self.assertIsNone(result.receipt["gradients"]["noop_sha256"])
        self.assertEqual(result.optimizer_receipt["arm"], "action-only")

    def test_collective_order_and_mixed_dtype_fail_before_optimizer(self) -> None:
        for mode in ("wrong-order", "mixed"):
            with self.subTest(mode=mode):
                before = {
                    name: value.detach().clone() for name, value in self.parameters.items()
                }
                with self.assertRaises(step_core.Full30ActionTrainingStepError):
                    self._run(gradient_mode=mode)
                self.assertEqual(self.optimizer.update_count, 0)
                for name, value in self.parameters.items():
                    self.assertTrue(torch.equal(value, before[name]))
                self.collective_events.clear()

    def test_missing_zero_and_nonfinite_action_gradients_fail_closed(self) -> None:
        hostile = (
            self.FakeRuntime(self, missing="role"),
            self.FakeRuntime(self, nonfinite=True),
        )
        for runtime in hostile:
            with self.subTest(runtime=type(runtime).__name__):
                with self.assertRaises(step_core.Full30ActionTrainingStepError):
                    self._run(runtime=runtime)
                self.assertEqual(self.optimizer.update_count, 0)
        self.parameters[
            "blocks.0.attn1.to_q.lora_A.default.weight"
        ].data.zero_()
        with self.assertRaises(step_core.Full30ActionTrainingStepError):
            self._run()
        self.assertEqual(self.optimizer.update_count, 0)

    def test_noop_replay_and_world_consensus_mismatch_fail_closed(self) -> None:
        with self.assertRaises(step_core.Full30ActionTrainingStepError):
            self._run(runtime=self.FakeRuntime(self, replay_mismatch=True))
        self.assertEqual(self.optimizer.update_count, 0)
        with self.assertRaises(step_core.Full30ActionTrainingStepError):
            self._run(consensus_mismatch=True)
        self.assertEqual(self.optimizer.update_count, 0)

    def test_optimizer_second_collective_failure_rolls_back_exact_state(self) -> None:
        parameters_before = {
            name: value.detach().clone() for name, value in self.parameters.items()
        }
        moments_before = {
            name: self.optimizer.second_moment(name)
            for name in self.optimizer.canonical_parameter_names
        }
        with self.assertRaises(step_core.Full30ActionTrainingStepError):
            self._run(scalar_fail_second=True)
        self.assertEqual(self.optimizer.update_count, 0)
        for name, value in self.parameters.items():
            self.assertTrue(torch.equal(value, parameters_before[name]))
            self.assertTrue(
                torch.equal(self.optimizer.second_moment(name), moments_before[name])
            )

    def test_schedule_rank_tamper_and_objective_tamper_are_rejected(self) -> None:
        records = list(self._local_records(rank=0, update=0))
        records[0] = step_core.Full30LocalMicroRecordV1(
            scheduled=self.schedule[1],
            runtime_record=records[0].runtime_record,
            objective=records[0].objective,
        )
        with self.assertRaises(step_core.Full30ActionTrainingStepError):
            self._run(records=records)
        records = list(self._local_records(rank=0, update=0))
        records[0].objective.teacher_unit.reshape(-1)[3] = 0.5
        with self.assertRaises(step_core.Full30ActionTrainingStepError):
            self._run(records=records)
        records = list(self._local_records(rank=0, update=0))
        records[0] = replace(
            records[0],
            objective=replace(
                records[0].objective,
                minimum_amplitude_float32_le_sha256="d" * 64,
            ),
        )
        with self.assertRaises(step_core.Full30ActionTrainingStepError):
            self._run(records=records)

    def test_receipt_is_byte_deterministic_for_identical_update(self) -> None:
        first, _runtime = self._run(arm="action-only")
        first_bytes = step_core.canonical_receipt_bytes(first.receipt)
        first_optimizer = optimizer_core.canonical_receipt_bytes(
            first.optimizer_receipt
        )
        self.setUp()
        second, _runtime = self._run(arm="action-only")
        self.assertEqual(
            step_core.canonical_receipt_bytes(second.receipt), first_bytes
        )
        self.assertEqual(
            optimizer_core.canonical_receipt_bytes(second.optimizer_receipt),
            first_optimizer,
        )

    def test_postcommit_finalization_has_no_external_call_surface(self) -> None:
        source = inspect.getsource(
            step_core.execute_full30_action_training_step_v1
        )
        postcommit = source.split("optimizer_receipt = optimizer.step(", 1)[1]
        self.assertNotIn("gradient_mean(", postcommit)
        self.assertNotIn("world_consensus(", postcommit)
        self.assertNotIn("optimizer_all_reduce_sum(", postcommit)
        result, _runtime = self._run()
        self.assertEqual(self.optimizer.update_count, 1)
        self.assertEqual(self.consensus_events, 1)
        self.assertEqual(self.scalar_events, 2)
        step_core.canonical_receipt_bytes(result.receipt)
        optimizer_core.canonical_receipt_bytes(result.optimizer_receipt)


if __name__ == "__main__":
    unittest.main()
