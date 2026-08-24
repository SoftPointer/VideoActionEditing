from __future__ import annotations

import ast
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = METHOD_ROOT / "generic_source_anchored_action_v1.py"
CORE_SOURCE = CORE_PATH.read_text(encoding="utf-8")
CORE_TREE = ast.parse(CORE_SOURCE)
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import generic_source_anchored_action_v1 as core

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    core = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _TinyAttention(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)
            self.to_v = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )


    class _TinyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn2 = _TinyAttention()


    class _TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [_TinyBlock() for _ in range(core.TOTAL_BLOCKS_1P3B)]
            )


class GenericSourceAnchoredStaticContractTests(unittest.TestCase):
    def test_registered_topology_stages_and_counts_are_literal(self) -> None:
        for fragment in (
            'TOPOLOGY = "world4-dp1-sp4"',
            "WORLD_SIZE = 4",
            "SP_SIZE = 4",
            "DP_SIZE = 1",
            'STAGE_UPDATES = {"R": 64, "P": 24, "O": 16}',
            "CARRIER_BLOCK_INDICES = (8, 12, 16, 20)",
            "ACTION_BLOCK_INDICES = tuple(range(23))",
            "ACTION_OPERATOR_RANK = 8",
            "TRAIN_SIGMA_INDICES = (4, 12, 20, 28, 35, 38)",
            "R_SIGMA_COUNTS = (11, 11, 11, 11, 10, 10)",
            "O_SIGMA_COUNTS = (3, 3, 3, 3, 2, 2)",
        ):
            self.assertIn(fragment, CORE_SOURCE)

    def test_no_old_low_sigma_hard_off_or_action_family_route(self) -> None:
        for forbidden in (
            "LOW_SIGMA_EXACT_BASE_INDICES",
            "low_exact_base",
            "schedule_index < 38",
            "family_head",
            "dog_head",
            "human_head",
        ):
            self.assertNotIn(forbidden, CORE_SOURCE)
        self.assertIn('"all_exact40_operator_available": True', CORE_SOURCE)

    def test_planner_architecture_and_exact_count_are_preregistered(self) -> None:
        self.assertIn("PLANNER_PARAMETER_COUNT = 1_584_160", CORE_SOURCE)
        self.assertIn("ACTION_OPERATOR_PARAMETER_COUNT = 1_142_272", CORE_SOURCE)
        self.assertIn("CARRIER_PARAMETER_COUNT = 2_036_996", CORE_SOURCE)
        for fragment in (
            "self.text_projection = nn.Linear(TEXT_WIDTH, PLANNER_WIDTH",
            "self.cross_attention = nn.MultiheadAttention(",
            "self.attention_norm = nn.LayerNorm",
            "self.feedforward_norm = nn.LayerNorm",
            "nn.Linear(PLANNER_WIDTH, 512",
            "nn.Linear(512, PLANNER_WIDTH",
            "self.output = nn.Linear(PLANNER_WIDTH, PHASE_CODE_WIDTH",
        ):
            self.assertIn(fragment, CORE_SOURCE)

    def test_generated_optimizer_inputs_are_explicitly_forbidden(self) -> None:
        for field in (
            '"generated_rgb"',
            '"generated_video_path"',
            '"generated_latent"',
            '"target_velocity"',
            '"noise"',
            '"epsilon"',
            '"t2v_reference"',
            '"action_family_id"',
            '"pose"',
            '"flow"',
            '"track"',
            '"mask"',
        ):
            self.assertIn(field, CORE_SOURCE)

    def test_noop_and_phase_zero_are_structural_returns(self) -> None:
        planner = CORE_SOURCE[CORE_SOURCE.index("class NaturalLanguagePhasePlanner") :]
        operator = CORE_SOURCE[CORE_SOURCE.index("class CurrentHiddenPhaseResidual") :]
        self.assertLess(planner.index("if is_noop:"), planner.index("self.text_projection(instruction_tokens.float())"))
        self.assertLess(
            operator.index("if not route.enabled or route.is_noop:"),
            operator.index("self._delta("),
        )
        self.assertIn("selected = phases > 0", operator)
        self.assertIn('"protected_rows_bit_exact": protected_exact', operator)

    def test_nuisance_projection_uses_ordered_gram_schmidt(self) -> None:
        for fragment in (
            "def _gram_schmidt_nuisances(",
            "appearance_flat @ camera_flat",
            '"appearance nuisance direction is degenerate after Gram-Schmidt"',
            "camera_direction, appearance_direction = _gram_schmidt_nuisances(",
            "code = _project_off(code, camera_direction)",
            "code = _project_off(code, appearance_direction)",
        ):
            self.assertIn(fragment, CORE_SOURCE)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class GenericSourceAnchoredTensorTests(unittest.TestCase):
    def test_joint_composite_group_registry_survives_r_p_o_r_activation(self) -> None:
        class TinyTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = nn.Parameter(torch.zeros(1), requires_grad=False)

        class TinyCarrier:
            def __init__(self, transformer: nn.Module) -> None:
                self.components = nn.Linear(1, 1, bias=False)
                transformer.carrier_components = self.components

            def trainable_named_parameters(self):
                rows = tuple(self.components.named_parameters())
                if any(not parameter.requires_grad for _, parameter in rows):
                    raise RuntimeError("carrier accessor rejects frozen parameters")
                return rows

        class TinyOperator:
            def __init__(self, transformer: nn.Module) -> None:
                self.module = nn.Linear(1, 1, bias=False)
                transformer.operator_components = self.module

            def trainable_named_parameters(self):
                return tuple(self.module.named_parameters())

        class TinyPlanner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

            def trainable_named_parameters(self):
                return tuple(self.named_parameters())

        transformer = TinyTransformer()

        def install_carrier(value, **_kwargs):
            return TinyCarrier(value)

        def install_operator(value):
            return TinyOperator(value)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                core.carrier_core,
                "install_clean_source_visual_context_adapter_v1",
                side_effect=install_carrier,
            ))
            stack.enter_context(mock.patch.object(
                core, "install_action_operator_v1", side_effect=install_operator
            ))
            stack.enter_context(mock.patch.object(
                core, "NaturalLanguagePhasePlanner", TinyPlanner
            ))
            stack.enter_context(mock.patch.object(core, "CARRIER_PARAMETER_COUNT", 1))
            stack.enter_context(mock.patch.object(core, "PLANNER_PARAMETER_COUNT", 1))
            stack.enter_context(mock.patch.object(
                core, "ACTION_OPERATOR_PARAMETER_COUNT", 1
            ))
            handle = core.install_composite_v1(
                transformer,
                experiment="joint_source_anchored_v1",
                runtime_source_commit="a" * 40,
                model_revision="b" * 64,
                checkpoint_manifest_sha256="c" * 64,
            )
            controller = core.StageOptimizerController(handle)
            for stage in ("R", "P", "O", "R"):
                active = controller.activate(stage)
                groups = handle.named_parameter_groups()
                snapshot = core.frozen_inactive_snapshot(handle, stage)
                self.assertEqual(len(active), 1)
                self.assertEqual(
                    {name for name, parameter in groups[stage] if parameter.requires_grad},
                    {groups[stage][0][0]},
                )
                self.assertEqual(len(snapshot), 2)

    def test_small_operator_installer_has_exact_scope_and_restores(self) -> None:
        transformer = _TinyTransformer()
        transformer.requires_grad_(False)
        original_q = tuple(block.attn2.to_q for block in transformer.blocks)
        original_o = tuple(block.attn2.to_out[0] for block in transformer.blocks)
        handle = core.install_action_operator_v1(
            transformer, strict_production_shape=False
        )
        named = handle.trainable_named_parameters()
        self.assertEqual(len(named), 23 * 2 * 3)
        self.assertEqual(
            sum(parameter.numel() for _, parameter in named),
            23 * 2 * (8 * 8 + 32 * 8 + 8 * 8),
        )
        self.assertTrue(
            all(
                torch.count_nonzero(wrapper.output_up.weight) == 0
                for _, wrapper in handle.wrappers
            )
        )
        handle.restore()
        self.assertEqual(
            tuple(block.attn2.to_q for block in transformer.blocks), original_q
        )
        self.assertEqual(
            tuple(block.attn2.to_out[0] for block in transformer.blocks), original_o
        )

    def test_production_operator_parameter_arithmetic_is_exact(self) -> None:
        expected = 23 * 2 * (
            core.HIDDEN_SIZE_1P3B * core.ACTION_OPERATOR_RANK
            + core.PHASE_CODE_WIDTH * core.ACTION_OPERATOR_RANK
            + core.ACTION_OPERATOR_RANK * core.HIDDEN_SIZE_1P3B
        )
        self.assertEqual(expected, core.ACTION_OPERATOR_PARAMETER_COUNT)

    def test_planner_count_nonnoop_shape_and_noop_hard_bypass(self) -> None:
        planner = core.NaturalLanguagePhasePlanner()
        self.assertEqual(
            sum(parameter.numel() for parameter in planner.parameters()),
            core.PLANNER_PARAMETER_COUNT,
        )
        tokens = torch.randn(1, 5, core.TEXT_WIDTH)
        instruction = "Raise both hands and then place them on the hips."
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        result = planner(tokens, instruction=instruction, instruction_sha256=digest)
        self.assertEqual(tuple(result.shape), (1, 21, 32))
        with torch.no_grad():
            for parameter in planner.parameters():
                parameter.fill_(float("nan"))
        noop = planner(
            tokens,
            instruction=core.EXACT_NOOP_INSTRUCTION,
            instruction_sha256=core.EXACT_NOOP_INSTRUCTION_SHA256,
            is_noop=True,
        )
        self.assertTrue(torch.equal(noop, torch.zeros_like(noop)))

    def test_operator_keeps_source_phase0_noop_exact_and_runs_index38(self) -> None:
        base = nn.Linear(8, 8, bias=False)
        base.requires_grad_(False)
        wrapper = core.CurrentHiddenPhaseResidual(base, projection="to_q")
        with torch.no_grad():
            wrapper.state_down.weight.fill_(0.2)
            wrapper.phase_gate.weight.fill_(0.3)
            wrapper.output_up.weight.fill_(0.4)
        hidden = torch.randn(1, 3 + 21 * 2, 8)
        code = torch.ones(1, 21, 32)
        route = core.ActionRoute(45, 3, 0, 1, code, schedule_index=38)
        base_value = base(hidden)
        with core.activate_action_route(route):
            result = wrapper(hidden)
        action_audit = wrapper._last_runtime_audit
        self.assertTrue(torch.equal(result[:, :3], base_value[:, :3]))
        self.assertTrue(torch.equal(result[:, 3:5], base_value[:, 3:5]))
        self.assertFalse(torch.equal(result[:, 5:], base_value[:, 5:]))
        self.assertIs(action_audit["protected_rows_bit_exact"], True)
        self.assertEqual(action_audit["phase0_rows"], 2)

        zero = torch.zeros(1, 21, 32)
        noop = core.ActionRoute(45, 3, 0, 1, zero, schedule_index=39, is_noop=True)
        with torch.no_grad():
            wrapper.state_down.weight.fill_(float("nan"))
            wrapper.phase_gate.weight.fill_(float("nan"))
            wrapper.output_up.weight.fill_(float("nan"))
        with core.activate_action_route(noop):
            noop_result = wrapper(hidden)
        self.assertTrue(torch.equal(noop_result, base_value))
        self.assertIs(wrapper._last_runtime_audit["hard_bypass"], True)

        operator_off = core.ActionRoute(
            45, 3, 0, 1, code, schedule_index=38, enabled=False
        )
        with core.activate_action_route(operator_off):
            operator_off_result = wrapper(hidden)
        self.assertTrue(torch.equal(operator_off_result, base_value))
        self.assertIs(wrapper._last_runtime_audit["route_enabled"], False)
        self.assertIs(wrapper._last_runtime_audit["selected_delta_nonzero"], False)

    def test_o_update1_zero_safe_cosine_is_narrow_and_has_real_gradient(self) -> None:
        source = torch.randn(1, 21, 8)
        output_up = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(output_up.weight)
        prediction = output_up(source)
        teacher = torch.randn_like(prediction)
        teacher[:, 0, :] = 0.0
        teacher = teacher / teacher.norm()
        loss = core.zero_init_operator_cosine_quotient_loss(
            prediction, teacher.detach(), stage_update=1
        )
        loss.backward()
        self.assertGreater(float(output_up.weight.grad.norm()), 0.0)
        with self.assertRaises(core.GenericSourceAnchoredActionError):
            core.cosine_quotient_loss(prediction.detach(), teacher.detach())
        with self.assertRaises(core.GenericSourceAnchoredActionError):
            core.zero_init_operator_cosine_quotient_loss(
                prediction.detach(), teacher.detach(), stage_update=2
            )

    def test_o_zero_safe_cosine_still_rejects_zero_teacher_and_nonfinite(self) -> None:
        prediction = torch.randn(1, 21, 32)
        teacher = torch.randn_like(prediction)
        for bad_prediction, bad_teacher in (
            (prediction, torch.zeros_like(teacher)),
            (prediction.clone().fill_(float("nan")), teacher),
            (prediction, teacher.clone().fill_(float("nan"))),
        ):
            with self.assertRaises(core.GenericSourceAnchoredActionError):
                core.zero_init_operator_cosine_quotient_loss(
                    bad_prediction, bad_teacher, stage_update=1
                )

    def test_fixed_schedules_have_registered_balanced_counts(self) -> None:
        r = core.fixed_sigma_schedule("R")
        o = core.fixed_sigma_schedule("O")
        self.assertEqual(len(r), 64)
        self.assertEqual(len(o), 16)
        self.assertEqual(tuple(r.count(value) for value in core.TRAIN_SIGMA_INDICES), core.R_SIGMA_COUNTS)
        self.assertEqual(tuple(o.count(value) for value in core.TRAIN_SIGMA_INDICES), core.O_SIGMA_COUNTS)

    def test_payload_guard_rejects_every_privileged_tensor_route(self) -> None:
        for key in (
            "generated_rgb",
            "generated_latent",
            "target_velocity",
            "epsilon",
            "t2v_reference",
            "action_family_id",
            "pose",
            "flow",
            "track",
            "mask",
        ):
            with self.assertRaises(core.GenericSourceAnchoredActionError):
                core.assert_optimizer_payload_safe({key: "forbidden"})
        core.assert_optimizer_payload_safe(
            {
                "instruction": "an action",
                "instruction_sha256": "a" * 64,
                "quotient_path": "/sealed/representation.pt",
                "quotient_sha256": "b" * 64,
            }
        )

    def test_phi_phase0_and_temporal_dc_are_exactly_removed(self) -> None:
        delta = torch.randn(1, 7 + 21 * 2, core.HIDDEN_SIZE_1P3B)
        p32 = core.fixed_p32()
        code = core.phi_v1_from_global_hidden_delta(
            delta, condition_tokens=7, p32=p32
        )
        self.assertEqual(tuple(code.shape), (1, 21, 32))
        self.assertTrue(torch.equal(code[:, 0], torch.zeros_like(code[:, 0])))
        self.assertTrue(
            torch.allclose(
                code[:, 1:].mean(1),
                torch.zeros_like(code[:, 1:].mean(1)),
                atol=2.0e-6,
                rtol=0.0,
            )
        )

    def test_sp1_phi_matches_global_phi_and_preserves_autograd(self) -> None:
        delta = torch.randn(
            1, 7 + 21 * 2, core.HIDDEN_SIZE_1P3B, requires_grad=True
        )
        p32 = core.fixed_p32()
        camera = torch.randn(1, 21, 32)
        appearance = 0.3 * camera + torch.randn(1, 21, 32)
        route = core.ActionRoute(
            total_tokens=7 + 21 * 2,
            condition_tokens=7,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
            phase_code=torch.ones(1, 21, 32),
            schedule_index=29,
        )
        expected = core.phi_v1_from_global_hidden_delta(
            delta,
            condition_tokens=7,
            p32=p32,
            camera_nuisance=camera,
            appearance_nuisance=appearance,
        )
        observed = core.phi_v1_from_sp_hidden_delta(
            delta,
            route=route,
            p32=p32,
            camera_nuisance=camera,
            appearance_nuisance=appearance,
        )
        self.assertTrue(torch.allclose(observed, expected, atol=2.0e-6, rtol=0.0))
        observed.square().sum().backward()
        self.assertIsNotNone(delta.grad)
        self.assertGreater(float(delta.grad.norm()), 0.0)

    def test_checkpoint_context_replays_the_exact_action_route(self) -> None:
        route = core.ActionRoute(
            total_tokens=21,
            condition_tokens=0,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
            phase_code=torch.ones(1, 21, 32),
            schedule_index=12,
        )
        with core.activate_action_route(route):
            forward_context, recompute_context = (
                core.composite_checkpoint_route_context_fn()
            )
            with forward_context:
                self.assertIs(core.active_action_route(), route)
        self.assertIsNone(core.active_action_route())
        with recompute_context:
            self.assertIs(core.active_action_route(), route)
        self.assertIsNone(core.active_action_route())

    def test_phi_gram_schmidt_removes_both_nuisance_directions(self) -> None:
        delta = torch.randn(1, 5 + 21 * 2, core.HIDDEN_SIZE_1P3B)
        p32 = core.fixed_p32()
        camera = torch.randn(1, 21, 32)
        appearance = 0.7 * camera + torch.randn(1, 21, 32)
        code = core.phi_v1_from_global_hidden_delta(
            delta,
            condition_tokens=5,
            p32=p32,
            camera_nuisance=camera,
            appearance_nuisance=appearance,
        )
        camera_unit = camera.reshape(-1) / camera.norm()
        appearance_orthogonal = appearance.reshape(-1) - (
            appearance.reshape(-1) @ camera_unit
        ) * camera_unit
        appearance_unit = appearance_orthogonal / appearance_orthogonal.norm()
        self.assertAlmostEqual(float(code.reshape(-1) @ camera_unit), 0.0, places=4)
        self.assertAlmostEqual(float(code.reshape(-1) @ appearance_unit), 0.0, places=4)
        with self.assertRaises(core.GenericSourceAnchoredActionError):
            core.phi_v1_from_global_hidden_delta(
                delta,
                condition_tokens=5,
                p32=p32,
                camera_nuisance=camera,
                appearance_nuisance=2.0 * camera,
            )

    def test_contract_distinguishes_joint104_and_action40(self) -> None:
        joint = core.training_contract_receipt("joint_source_anchored_v1")
        action = core.training_contract_receipt("action_only_no_carrier_v1")
        self.assertEqual(joint["total_updates"], 104)
        self.assertEqual(action["total_updates"], 40)
        self.assertEqual(joint["stages"], ["R", "P", "O"])
        self.assertEqual(action["stages"], ["P", "O"])
        self.assertFalse(joint["rank_or_gpu_action_family_partition"])
        self.assertFalse(action["rank_or_gpu_action_family_partition"])


if __name__ == "__main__":
    unittest.main()
