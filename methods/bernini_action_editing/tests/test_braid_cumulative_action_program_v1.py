#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # local macOS workspace intentionally has no torch
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

if TORCH_AVAILABLE:
    import braid_cumulative_action_program_v1 as braid  # noqa: E402
else:
    braid = None  # type: ignore[assignment]


def _hidden(
    *, batch: int = 1, patches: int = 2, dtype: torch.dtype = None
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32
    value = torch.zeros(
        batch, braid.LATENT_PHASES, patches, braid.HIDDEN_SIZE, dtype=torch.float32
    )
    for patch in range(patches):
        value[:, :, patch] = 0.01 * float(patch + 1)
    return value.to(dtype=dtype)


def _action_plan(*, batch: int = 1) -> torch.Tensor:
    value = torch.zeros(batch, braid.PLAN_STAGES, braid.PLAN_WIDTH)
    for stage in range(braid.PLAN_STAGES):
        value[:, stage, stage] = float(stage + 1) / 4.0
    return value.contiguous()


def _plan_snapshot(
    *, role: str = "action", batch: int = 1, value: torch.Tensor = None
) -> braid.BraidPlanSnapshot:
    if value is None:
        value = (
            torch.zeros(batch, 4, 32, dtype=torch.float32)
            if role == "noop"
            else _action_plan(batch=batch)
        )
    return braid.BraidPlanSnapshot(value, role=role, origin_label="unit-plan")


def _evidence(
    *, batch: int = 1, failed: tuple[int, ...] = ()
) -> braid.BraidEvidenceSnapshot:
    correct = torch.ones(batch, 4, dtype=torch.float32)
    shuffled = torch.zeros(batch, 4, dtype=torch.float32)
    wrong = torch.zeros(batch, 4, dtype=torch.float32)
    noop = torch.zeros(batch, 4, dtype=torch.float32)
    for index in failed:
        wrong[index, 2] = 2.0
    return braid.BraidEvidenceSnapshot(
        correct,
        shuffled,
        wrong,
        noop,
        origin_label="unit-evidence",
    )


def _target(
    *, batch: int = 1, patches: int = 2, dtype: torch.dtype = None
) -> braid.BraidTargetSnapshot:
    return braid.BraidTargetSnapshot(
        _hidden(batch=batch, patches=patches, dtype=dtype),
        origin_label="unit-target",
    )


def _activate_decoder(
    module: braid.BraidCumulativeActionProgram, *, scale: float = 0.1
) -> None:
    with torch.no_grad():
        module.stage_decoder.zero_()
        decoder = module.hidden_encoder.T * float(scale)
        for stage in range(braid.PLAN_STAGES):
            module.stage_decoder[stage].copy_(decoder)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable in this environment")
class BraidCumulativeActionProgramTests(unittest.TestCase):
    def test_basis_is_start_anchored_cumulative_and_never_zero_dc(self) -> None:
        basis = braid.build_cumulative_smoothstep_basis()
        self.assertEqual(
            braid.STAGE_NAMES,
            ("onset", "transition", "completion", "terminal_hold"),
        )
        self.assertEqual(tuple(basis.shape), (21, 4))
        self.assertTrue(torch.equal(basis[0], torch.zeros(4)))
        self.assertTrue(torch.all(basis[1:] >= basis[:-1]))
        self.assertTrue(torch.equal(basis[-1], torch.ones(4)))
        self.assertTrue(torch.all(basis.sum(dim=0) > 0.0))
        self.assertFalse(
            torch.allclose(basis.mean(dim=0), torch.zeros(4), atol=0.0, rtol=0.0)
        )
        for stage, plateau_start in enumerate(braid.STAGE_PLATEAU_STARTS):
            with self.subTest(stage=stage):
                self.assertTrue(
                    torch.equal(
                        basis[plateau_start:, stage],
                        torch.ones(21 - plateau_start),
                    )
                )

    def test_cumulative_program_preserves_completed_stage_plateaus(self) -> None:
        module = braid.BraidCumulativeActionProgram()
        plan = torch.zeros(1, 4, 32)
        plan[:, 0, 0] = 1.0
        snapshot = _plan_snapshot(value=plan)
        program = module.cumulative_program(snapshot)
        plateau = braid.STAGE_PLATEAU_STARTS[0]
        expected = program[:, plateau : plateau + 1].expand_as(program[:, plateau:])
        self.assertTrue(torch.equal(program[:, plateau:], expected))
        self.assertEqual(int(torch.count_nonzero(program[:, 0]).item()), 0)
        self.assertGreater(int(torch.count_nonzero(program[:, -1]).item()), 0)

    def test_plan_has_no_layout_channel_and_is_spatially_broadcast(self) -> None:
        with self.assertRaises(braid.BraidError):
            braid.BraidPlanSnapshot(
                torch.zeros(1, 4, 2, 32),
                role="action",
                origin_label="bad-layout",
            )
        snapshot = _plan_snapshot()
        receipt = snapshot.receipt()
        self.assertEqual(receipt["shape"], [1, 4, 32])
        self.assertFalse(receipt["patch_axis_present"])
        self.assertFalse(receipt["owner_layout_channel_present"])
        self.assertEqual(receipt["forbidden_owner_channels_consumed_by_api"], [])

        module = braid.BraidCumulativeActionProgram()
        _activate_decoder(module)
        same_patch_hidden = torch.full((1, 21, 2, 1536), 0.01)
        target = braid.BraidTargetSnapshot(
            same_patch_hidden, origin_label="same-patches"
        )
        delta = module.adapter_delta(target, snapshot, _evidence(), sigma=0.8)
        self.assertTrue(torch.equal(delta[:, :, 0], delta[:, :, 1]))

        different = _target(patches=2)
        different_delta = module.adapter_delta(
            different, snapshot, _evidence(), sigma=0.8
        )
        self.assertFalse(
            torch.equal(different_delta[:, :, 0], different_delta[:, :, 1])
        )

    def test_plan_can_expand_only_through_current_hidden_projection(self) -> None:
        module = braid.BraidCumulativeActionProgram()
        _activate_decoder(module)
        zero_target = braid.BraidTargetSnapshot(
            torch.zeros(1, 21, 1, 1536), origin_label="zero-hidden"
        )
        plan = _plan_snapshot()
        zero_delta = module.adapter_delta(
            zero_target, plan, _evidence(), sigma=0.8
        )
        self.assertEqual(int(torch.count_nonzero(zero_delta).item()), 0)

        nonzero_delta = module.adapter_delta(
            _target(patches=1), plan, _evidence(), sigma=0.8
        )
        self.assertGreater(int(torch.count_nonzero(nonzero_delta).item()), 0)

    def test_zero_decoder_is_identity_and_receipt_is_structural_only(self) -> None:
        module = braid.BraidCumulativeActionProgram()
        target = _target()
        output = module(target, _plan_snapshot(), _evidence(), sigma=0.8)
        self.assertTrue(torch.equal(output, target.tensor_copy()))
        receipt = module.receipt()
        self.assertTrue(receipt["only_stage_decoder_trainable"])
        self.assertTrue(receipt["stage_decoder_zero_at_receipt_time"])
        self.assertFalse(receipt["temporal_centering_used"])
        self.assertFalse(receipt["zero_temporal_dc_required"])
        self.assertEqual(
            receipt["cumulative_basis_sha256"],
            braid._local_tensor_sha256(
                module.cumulative_basis, label="test cumulative basis"
            ),
        )
        self.assertFalse(receipt["scientific_authority"])
        self.assertNotIn("optimizer_constructed", receipt)
        self.assertNotIn("model_forward_performed", receipt)
        self.assertNotIn("training_performed", receipt)

    def test_canonical_noop_evidence_off_and_low_sigma_are_same_object(self) -> None:
        module = braid.BraidCumulativeActionProgram()
        _activate_decoder(module)
        target = _target()
        owned = object.__getattribute__(target, "_tensor")
        noop = _plan_snapshot(role="noop")
        noop_result = module(target, noop, _evidence(), sigma=0.8)
        self.assertIs(noop_result, owned)

        action = _plan_snapshot()
        evidence_off = _evidence(failed=(0,))
        off_result = module(target, action, evidence_off, sigma=0.8)
        self.assertIs(off_result, owned)
        low_result = module(target, action, _evidence(), sigma=0.1)
        self.assertIs(low_result, owned)

        with self.assertRaises(braid.BraidError):
            braid.BraidPlanSnapshot(
                torch.zeros(1, 4, 32), role="action", origin_label="false-action"
            )
        with self.assertRaises(braid.BraidError):
            braid.BraidPlanSnapshot(
                _action_plan(), role="noop", origin_label="false-noop"
            )

    def test_mixed_batch_isolates_failed_evidence_row(self) -> None:
        module = braid.BraidCumulativeActionProgram()
        _activate_decoder(module)
        target = _target(batch=2, patches=1)
        before = target.tensor_copy()
        result = module(
            target,
            _plan_snapshot(batch=2),
            _evidence(batch=2, failed=(1,)),
            sigma=0.8,
        )
        self.assertFalse(torch.equal(result[0], before[0]))
        self.assertTrue(torch.equal(result[1], before[1]))

    def test_segment_and_global_patch_token_norm_caps(self) -> None:
        segment_cap = 0.30
        global_cap = 0.45
        module = braid.BraidCumulativeActionProgram(
            max_segment_token_norm=segment_cap,
            max_global_token_norm=global_cap,
        )
        _activate_decoder(module, scale=1000.0)
        target = _target(patches=2)
        for stage in range(4):
            with self.subTest(stage=stage):
                plan = torch.zeros(1, 4, 32)
                plan[:, stage, stage] = 1000.0
                delta = module.adapter_delta(
                    target,
                    _plan_snapshot(value=plan),
                    _evidence(),
                    sigma=0.8,
                )
                token_norms = torch.linalg.vector_norm(delta.float(), dim=-1)
                self.assertTrue(torch.all(token_norms <= global_cap + 2.0e-5))
                self.assertTrue(torch.all(token_norms <= segment_cap + 2.0e-5))

        all_delta = module.adapter_delta(
            target, _plan_snapshot(), _evidence(), sigma=0.8
        )
        all_norms = torch.linalg.vector_norm(all_delta.float(), dim=-1)
        self.assertTrue(torch.all(all_norms <= global_cap + 2.0e-5))

    def test_shuffled_and_wrong_plans_are_active_diagnostic_programs(self) -> None:
        module = braid.BraidCumulativeActionProgram()
        _activate_decoder(module)
        action = _action_plan()
        shuffled = action[:, (2, 0, 3, 1), :].contiguous()
        wrong = (-action).contiguous()
        action_snapshot = _plan_snapshot(role="action", value=action)
        shuffled_snapshot = _plan_snapshot(role="shuffled", value=shuffled)
        wrong_snapshot = _plan_snapshot(role="wrong", value=wrong)

        action_program = module.cumulative_program(action_snapshot)
        shuffled_program = module.cumulative_program(shuffled_snapshot)
        wrong_program = module.cumulative_program(wrong_snapshot)
        self.assertFalse(torch.equal(action_program, shuffled_program))
        self.assertTrue(torch.equal(wrong_program, -action_program))

        target = _target(patches=1)
        evidence = _evidence()
        action_delta = module.adapter_delta(
            target, action_snapshot, evidence, sigma=0.8
        )
        shuffled_delta = module.adapter_delta(
            target, shuffled_snapshot, evidence, sigma=0.8
        )
        wrong_delta = module.adapter_delta(
            target, wrong_snapshot, evidence, sigma=0.8
        )
        self.assertGreater(int(torch.count_nonzero(shuffled_delta).item()), 0)
        self.assertFalse(torch.equal(action_delta, shuffled_delta))
        self.assertTrue(torch.equal(wrong_delta, -action_delta))

    def test_evidence_is_owned_fp64_per_stage_hard_conjunction(self) -> None:
        correct_leaf = torch.ones(1, 4, requires_grad=True)
        shuffled_leaf = torch.zeros(1, 4, requires_grad=True)
        wrong_leaf = torch.zeros(1, 4, requires_grad=True)
        noop_leaf = torch.zeros(1, 4, requires_grad=True)
        correct = correct_leaf * 1.0
        shuffled = shuffled_leaf * 1.0
        wrong = wrong_leaf * 1.0
        noop = noop_leaf * 1.0
        evidence = braid.BraidEvidenceSnapshot(
            correct,
            shuffled,
            wrong,
            noop,
            origin_label="graph-evidence",
        )
        before = evidence.margins
        self.assertEqual(before.dtype, torch.float64)
        self.assertFalse(before.requires_grad)
        with torch.no_grad():
            correct.add_(100.0)
            wrong.sub_(100.0)
        self.assertTrue(torch.equal(evidence.margins, before))
        self.assertEqual(evidence.eligible_copy().tolist(), [True])

        failed_correct = torch.ones(1, 4)
        failed_wrong = torch.zeros(1, 4)
        failed_wrong[0, 3] = 2.0
        failed = braid.BraidEvidenceSnapshot(
            failed_correct,
            torch.zeros(1, 4),
            failed_wrong,
            torch.zeros(1, 4),
            origin_label="one-stage-fail",
        )
        self.assertEqual(failed.eligible_copy().tolist(), [False])

    def test_owned_clone_live_sha_and_exact_type_reject_alias_and_subclass(self) -> None:
        hidden = _hidden()
        plan = _action_plan()
        target = braid.BraidTargetSnapshot(hidden, origin_label="owned-target")
        plan_snapshot = braid.BraidPlanSnapshot(
            plan, role="action", origin_label="owned-plan"
        )
        target_before = target.tensor_copy()
        plan_before = plan_snapshot.plan_copy()
        with torch.no_grad():
            hidden.add_(100.0)
            plan.mul_(10.0)
        self.assertTrue(torch.equal(target.tensor_copy(), target_before))
        self.assertTrue(torch.equal(plan_snapshot.plan_copy(), plan_before))

        internal = object.__getattribute__(target, "_tensor")
        with torch.no_grad():
            internal[0, 0, 0, 0].add_(1.0)
        with self.assertRaises(braid.BraidError):
            target.tensor_copy()

        class TensorSubclass(torch.Tensor):
            pass

        subclass_tensor = _hidden().as_subclass(TensorSubclass)
        with self.assertRaises(braid.BraidError):
            braid.BraidTargetSnapshot(subclass_tensor, origin_label="subclass")
        subclass_plan = _action_plan().as_subclass(TensorSubclass)
        with self.assertRaises(braid.BraidError):
            braid.BraidPlanSnapshot(
                subclass_plan, role="action", origin_label="subclass"
            )

        class SnapshotSubclass(braid.BraidTargetSnapshot):
            pass

        evil = object.__new__(SnapshotSubclass)
        with self.assertRaises(braid.BraidError):
            braid.BraidCumulativeActionProgram()(
                evil, _plan_snapshot(), _evidence(), sigma=0.8
            )

    def test_sigma_boundaries_mid_amplitude_and_fp16_bf16_dtype(self) -> None:
        self.assertEqual(braid.sigma_gate(0.0), ("low_exact_base", 0.0))
        self.assertEqual(braid.sigma_gate(0.249999), ("low_exact_base", 0.0))
        self.assertEqual(braid.sigma_gate(0.25), ("mid", 0.5))
        self.assertEqual(braid.sigma_gate(0.549999), ("mid", 0.5))
        self.assertEqual(braid.sigma_gate(0.55), ("high", 1.0))
        self.assertEqual(braid.sigma_gate(1.0), ("high", 1.0))

        module = braid.BraidCumulativeActionProgram(
            max_segment_token_norm=100.0, max_global_token_norm=100.0
        )
        _activate_decoder(module, scale=0.01)
        target = _target(patches=1)
        plan = _plan_snapshot()
        evidence = _evidence()
        high = module.adapter_delta(target, plan, evidence, sigma=0.55)
        mid = module.adapter_delta(target, plan, evidence, sigma=0.25)
        self.assertTrue(torch.equal(mid * 2.0, high))

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                typed_target = _target(patches=1, dtype=dtype)
                delta = module.adapter_delta(
                    typed_target, plan, evidence, sigma=0.8
                )
                output = module(typed_target, plan, evidence, sigma=0.8)
                self.assertEqual(delta.dtype, dtype)
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(
                    torch.equal(delta[:, 0], torch.zeros_like(delta[:, 0]))
                )

        for invalid in (True, -0.1, 1.01, float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(braid.BraidError):
                    braid.sigma_gate(invalid)

    def test_local_provenance_is_explicitly_non_authoritative(self) -> None:
        target_receipt = _target().receipt()
        plan_receipt = _plan_snapshot().receipt()
        evidence_receipt = _evidence().receipt()
        self.assertEqual(
            target_receipt["authority"], braid.LOCAL_RESEARCH_AUTHORITY
        )
        self.assertFalse(target_receipt["upstream_authentication_checked"])
        self.assertFalse(target_receipt["source_conditioning_semantics_verified"])
        self.assertFalse(plan_receipt["role_semantics_verified"])
        self.assertFalse(evidence_receipt["semantic_admission_authority"])
        self.assertFalse(evidence_receipt["update_authority"])


if __name__ == "__main__":
    unittest.main()
