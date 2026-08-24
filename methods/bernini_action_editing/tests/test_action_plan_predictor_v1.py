#!/usr/bin/env python3

from __future__ import annotations

import copy
from dataclasses import replace
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import action_plan_predictor_v1 as action_plan  # noqa: E402


try:
    import torch
except ModuleNotFoundError:  # local macOS workspace intentionally has no torch
    torch = None  # type: ignore[assignment]


class ActionPlanPredictorPureContractTests(unittest.TestCase):
    def test_formal_0817_architecture_and_exact_counts_are_frozen(self) -> None:
        config = action_plan.ActionPlanPredictorConfig()
        config.require_formal_0817()
        self.assertEqual(config.profile, action_plan.FORMAL_PROFILE)
        self.assertEqual(config.source_token_width, 1536)
        self.assertEqual(config.instruction_token_width, 4096)
        self.assertEqual(config.model_width, 512)
        self.assertEqual(config.attention_heads, 8)
        self.assertEqual(config.mlp_width, 2048)
        self.assertEqual(config.layer_count, 6)
        self.assertEqual(config.phase_count, 21)
        self.assertEqual(config.action_width, 256)
        self.assertEqual(config.actor_object_query_count, 2)
        self.assertEqual(
            action_plan.expected_predictor_parameter_count(config),
            22_083_072,
        )
        self.assertEqual(
            action_plan.expected_injection_parameter_count(hidden_width=1536),
            23_639_040,
        )
        self.assertEqual(
            action_plan.expected_conditioner_parameter_count(config),
            45_722_112,
        )

    def test_formal_profile_rejects_architecture_drift_and_test_profile_cannot_promote(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "formal-v1"
        ):
            action_plan.ActionPlanPredictorConfig(model_width=256).validate()
        test_config = _cpu_config()
        test_config.validate()
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "cannot be used"
        ):
            test_config.require_formal_0817()

    def test_product_predictor_signature_has_no_teacher_or_external_input(self) -> None:
        self.assertEqual(
            list(
                inspect.signature(action_plan.ActionPlanPredictorV1.forward).parameters
            ),
            ["self", "source_tokens", "instruction_tokens"],
        )
        forbidden = {
            "target",
            "anchor",
            "track",
            "pose",
            "contact",
            "tube",
            "mask",
            "annotation",
        }
        observed = set(
            inspect.signature(action_plan.ActionPlanPredictorV1.forward).parameters
        )
        self.assertTrue(forbidden.isdisjoint(observed))
        self.assertEqual(
            list(
                inspect.signature(
                    action_plan.ZeroInitTargetOnlyActionInjectionV1.forward
                ).parameters
            ),
            ["self", "target_hidden", "route", "block_index"],
        )


def _cpu_config(*, attention_heads: int = 4) -> action_plan.ActionPlanPredictorConfig:
    return action_plan.ActionPlanPredictorConfig(
        profile=action_plan.CPU_TEST_PROFILE,
        source_token_width=12,
        instruction_token_width=16,
        model_width=16,
        attention_heads=attention_heads,
        mlp_width=32,
        layer_count=2,
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable in this environment")
class ActionPlanPredictorTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(170817)
        self.config = _cpu_config()
        self.model = action_plan.ActionPlanConditionerV1(
            self.config, renderer_hidden_width=8
        )
        self.source = torch.randn(2, 3, 2, 2, self.config.source_token_width)
        self.instruction = torch.randn(
            2, 4, self.config.instruction_token_width
        )
        self.target = torch.linspace(
            -0.9,
            1.1,
            2 * action_plan.PHASE_COUNT * 3 * 8,
            dtype=torch.float32,
        ).reshape(2, action_plan.PHASE_COUNT, 3, 8)

    def _ownership(self, target=None):
        target = self.target if target is None else target
        target_tokens = 1
        for value in target.shape[1:-1]:
            target_tokens *= int(value)
        source_prefix_tokens = 11
        return action_plan.certify_closed_target_suffix_route(
            target,
            source_prefix_tokens=source_prefix_tokens,
            packed_total_tokens=source_prefix_tokens + target_tokens,
        )

    def _route(self, model=None, source=None, instruction=None, ownership=None):
        model = self.model if model is None else model
        source = self.source if source is None else source
        instruction = self.instruction if instruction is None else instruction
        ownership = self._ownership() if ownership is None else ownership
        return model.prepare_route(source, instruction, ownership)

    def _through_all_blocks(self):
        route = self._route()
        hidden = self.target
        for block_index in range(action_plan.TRANSFORMER_BLOCK_COUNT):
            hidden = self.model(hidden, route, block_index=block_index).target_hidden
        return action_plan.ConditionedTargetOutput(
            target_hidden=hidden,
            plan=route.plan,
        )

    def test_shapes_exact_inventory_and_state_dict_abi(self) -> None:
        route = self._route()
        output = self.model(self.target, route, block_index=0)
        self.assertEqual(
            tuple(output.plan.phase_tokens.shape),
            (2, 21, 256),
        )
        self.assertEqual(tuple(output.plan.global_token.shape), (2, 256))
        self.assertEqual(tuple(output.target_hidden.shape), tuple(self.target.shape))

        inventory = action_plan.exact_parameter_inventory(self.model)
        expected = action_plan.expected_conditioner_parameter_count(
            self.config, renderer_hidden_width=8
        )
        self.assertEqual(inventory["parameter_count"], expected)
        self.assertEqual(inventory["trainable_parameter_count"], expected)
        self.assertEqual(
            sum(item["numel"] for item in inventory["parameters"]), expected
        )
        self.assertTrue(
            all(item["requires_grad"] for item in inventory["parameters"])
        )

        abi = action_plan.exact_state_dict_abi(self.model)
        self.assertEqual(
            abi["schema_version"], action_plan.CONDITIONER_ABI_SCHEMA
        )
        self.assertRegex(abi["abi_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            [entry["name"] for entry in abi["entries"]],
            list(self.model.state_dict()),
        )
        self.assertIn("predictor._abi_config", self.model.state_dict())
        self.assertIn("injection._abi_config", self.model.state_dict())
        self.assertNotIn(
            "predictor.phase_position_encoding", self.model.state_dict()
        )
        self.assertIn(
            "predictor.phase_position_encoding_abi", self.model.state_dict()
        )
        self.assertIn(
            "predictor.source_position_encoding_abi", self.model.state_dict()
        )
        projection_weights = [
            name
            for name in self.model.state_dict()
            if name.startswith("injection.projections.")
            and name.endswith(".weight")
        ]
        self.assertEqual(
            projection_weights,
            [
                f"injection.projections.{index}.weight"
                for index in range(action_plan.TRANSFORMER_BLOCK_COUNT)
            ],
        )
        buffer_entries = {
            entry["name"]: entry
            for entry in abi["entries"]
            if entry["kind"] == "buffer"
        }
        for name in (
            "predictor._abi_config",
            "predictor.phase_position_encoding_abi",
            "predictor.source_position_encoding_abi",
            "injection._abi_config",
        ):
            self.assertRegex(
                buffer_entries[name]["semantic_value_sha256"],
                r"^[0-9a-f]{64}$",
            )

    def test_formal_conditioner_rejects_non_1536_renderer_width(self) -> None:
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "requires renderer hidden width 1536"
        ):
            action_plan.ActionPlanConditionerV1(
                action_plan.ActionPlanPredictorConfig(),
                renderer_hidden_width=8,
            )

    def test_deterministic_thw_position_signal_breaks_pooling_permutation(self) -> None:
        first = action_plan.deterministic_source_position_encoding(
            3, 2, 2, self.config.model_width, device=self.source.device
        )
        second = action_plan.deterministic_source_position_encoding(
            3, 2, 2, self.config.model_width, device=self.source.device
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (3, 2, 2, self.config.model_width))
        self.assertFalse(torch.equal(first[0, 0, 0], first[-1, -1, -1]))

        original = self.model.predictor(self.source, self.instruction)
        permuted = self.model.predictor(self.source.flip(1), self.instruction)
        self.assertFalse(
            torch.allclose(original.phase_tokens, permuted.phase_tokens)
        )
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "rank-5"
        ):
            self.model.predictor(
                self.source.flatten(1, 3), self.instruction
            )

    def test_bfloat16_inputs_use_internal_fp32_and_restore_output_dtype(self) -> None:
        source = self.source.to(torch.bfloat16)
        instruction = self.instruction.to(torch.bfloat16)
        target = self.target.to(torch.bfloat16)
        ownership = self._ownership(target)
        route = self._route(
            source=source,
            instruction=instruction,
            ownership=ownership,
        )
        reference = self.model.predictor(source.float(), instruction.float())
        self.assertEqual(route.plan.phase_tokens.dtype, torch.bfloat16)
        self.assertEqual(route.plan.global_token.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                route.plan.phase_tokens,
                reference.phase_tokens.to(torch.bfloat16),
            )
        )
        result = self.model(target, route, block_index=0)
        self.assertEqual(result.target_hidden.dtype, torch.bfloat16)
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in self.model.parameters())
        )

    def test_hot_thirty_block_route_reuses_one_finite_audit(self) -> None:
        route = self._route()
        hidden = self.target
        with mock.patch.object(
            action_plan.torch,
            "isfinite",
            wraps=action_plan.torch.isfinite,
        ) as finite:
            for block_index in range(action_plan.TRANSFORMER_BLOCK_COUNT):
                hidden = self.model(
                    hidden, route, block_index=block_index
                ).target_hidden
        self.assertEqual(finite.call_count, 0)

    def test_zero_projection_is_the_gate_without_second_zero_scalar(self) -> None:
        self.assertEqual(
            self.model.injection.gate_semantics,
            action_plan.ZERO_INIT_GATE_SEMANTICS,
        )
        self.assertFalse(
            any("gate" in name for name, _ in self.model.injection.named_parameters())
        )
        for projection in self.model.injection.projections:
            self.assertTrue(
                torch.equal(projection.weight, torch.zeros_like(projection.weight))
            )
            self.assertTrue(
                torch.equal(projection.bias, torch.zeros_like(projection.bias))
            )

    def test_zero_init_bootstrap_then_second_backward_reaches_predictor(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        first_route = self._route()
        first_hidden = self.target
        for block_index in range(action_plan.TRANSFORMER_BLOCK_COUNT):
            residual = self.model.injection.residual(
                first_route, block_index=block_index
            )
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
            first_hidden = self.model(
                first_hidden, first_route, block_index=block_index
            ).target_hidden
        first = action_plan.ConditionedTargetOutput(
            target_hidden=first_hidden,
            plan=first_route.plan,
        )
        self.assertTrue(torch.equal(first.target_hidden, self.target))

        coefficient = torch.linspace(
            0.1,
            1.3,
            self.target.numel(),
            dtype=self.target.dtype,
        ).reshape_as(self.target)
        first_loss = (first.target_hidden * coefficient).sum()
        first_loss.backward()

        projection_grads = [
            sum(
                float(parameter.grad.detach().abs().sum())
                for parameter in projection.parameters()
                if parameter.grad is not None
            )
            for projection in self.model.injection.projections
        ]
        predictor_bootstrap_grad = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in self.model.predictor.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(len(projection_grads), 30)
        self.assertTrue(all(value > 0.0 for value in projection_grads))
        self.assertEqual(predictor_bootstrap_grad, 0.0)

        optimizer.step()
        self.assertGreater(
            min(
                float(projection.weight.detach().abs().sum())
                for projection in self.model.injection.projections
            ),
            0.0,
        )
        optimizer.zero_grad(set_to_none=True)

        second = self._through_all_blocks()
        second_loss = (second.target_hidden * coefficient).sum()
        second_loss.backward()
        predictor_second_grad = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in self.model.predictor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(predictor_second_grad, 0.0)

    def test_target_ownership_not_shape_alone_and_tampering_fail_closed(self) -> None:
        plan = self.model.predictor(self.source, self.instruction)
        packed_source_and_target = torch.randn(2, 42, 8)
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "exactly 21 phases"
        ):
            action_plan.certify_closed_target_suffix_route(
                packed_source_and_target,
                source_prefix_tokens=11,
                packed_total_tokens=53,
            )
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "does not close"
        ):
            action_plan.certify_closed_target_suffix_route(
                self.target,
                source_prefix_tokens=11,
                packed_total_tokens=12,
            )
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "InjectionRoute"
        ):
            self.model.injection(self.target, plan, block_index=0)

        ownership = self._ownership()
        tampered = replace(ownership, target_only=False)
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "semantics"
        ):
            self.model.injection.bind_route(plan, tampered)

        nonfinite_phase = plan.phase_tokens.clone()
        nonfinite_phase[0, 0, 0] = float("nan")
        nonfinite_plan = action_plan.ActionPlanOutput(
            phase_tokens=nonfinite_phase,
            global_token=plan.global_token,
        )
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "finite audit"
        ):
            self.model.injection.bind_route(nonfinite_plan, ownership)

        route = self.model.injection.bind_route(plan, ownership)
        wrong_shape = self.target[:, :, :2]
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "certificate"
        ):
            self.model.injection(wrong_shape, route, block_index=0)

    def test_block_index_and_exact_block_count_fail_closed(self) -> None:
        route = self._route()
        expected_traversal = tuple(range(action_plan.TRANSFORMER_BLOCK_COUNT))
        self.assertEqual(
            self.model.injection.validate_block_traversal(expected_traversal),
            expected_traversal,
        )
        for invalid in (-1, 30, True, 1.0, "0"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    action_plan.ActionPlanPredictorError, "block_index"
                ):
                    self.model.injection(
                        self.target, route, block_index=invalid
                    )
        invalid_traversals = (
            (expected_traversal[:-1], "missing"),
            (expected_traversal[:-1] + (28,), "duplicate"),
            ((1, 0) + expected_traversal[2:], "order"),
        )
        for traversal, message in invalid_traversals:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    action_plan.ActionPlanPredictorError, message
                ):
                    self.model.injection.validate_block_traversal(traversal)
        with self.assertRaisesRegex(
            action_plan.ActionPlanPredictorError, "30 blocks"
        ):
            action_plan.ZeroInitTargetOnlyActionInjectionV1(
                hidden_width=8, block_count=29
            )

    def test_strict_reload_has_exact_forward_parity_and_rejects_semantic_abi_drift(
        self,
    ) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        trained = self._through_all_blocks()
        trained.target_hidden.square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        expected = self._through_all_blocks()
        state = copy.deepcopy(self.model.state_dict())
        restored = action_plan.ActionPlanConditionerV1(
            self.config, renderer_hidden_width=8
        )
        result = restored.load_state_dict(state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        restored_route = self._route(model=restored)
        restored_hidden = self.target
        for block_index in range(action_plan.TRANSFORMER_BLOCK_COUNT):
            restored_hidden = restored(
                restored_hidden,
                restored_route,
                block_index=block_index,
            ).target_hidden
        actual = action_plan.ConditionedTargetOutput(
            target_hidden=restored_hidden,
            plan=restored_route.plan,
        )
        torch.testing.assert_close(
            actual.plan.phase_tokens,
            expected.plan.phase_tokens,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual.plan.global_token,
            expected.plan.global_token,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual.target_hidden,
            expected.target_hidden,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            action_plan.exact_state_dict_abi(restored),
            action_plan.exact_state_dict_abi(self.model),
        )

        same_shapes_different_heads = action_plan.ActionPlanConditionerV1(
            _cpu_config(attention_heads=2), renderer_hidden_width=8
        )
        self.assertNotEqual(
            action_plan.exact_state_dict_abi(same_shapes_different_heads)[
                "abi_sha256"
            ],
            action_plan.exact_state_dict_abi(self.model)["abi_sha256"],
        )
        with self.assertRaisesRegex(RuntimeError, "semantic buffer value differs"):
            same_shapes_different_heads.load_state_dict(state, strict=True)

        missing_block = copy.deepcopy(state)
        del missing_block["injection.projections.29.weight"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            action_plan.ActionPlanConditionerV1(
                self.config, renderer_hidden_width=8
            ).load_state_dict(missing_block, strict=True)

        extra_block = copy.deepcopy(state)
        extra_block["injection.projections.30.weight"] = torch.zeros_like(
            state["injection.projections.29.weight"]
        )
        extra_block["injection.projections.30.bias"] = torch.zeros_like(
            state["injection.projections.29.bias"]
        )
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            action_plan.ActionPlanConditionerV1(
                self.config, renderer_hidden_width=8
            ).load_state_dict(extra_block, strict=True)

    def test_load_preflight_rejects_dtype_and_buffer_value_before_mutation(self) -> None:
        state = copy.deepcopy(self.model.state_dict())
        receiver = action_plan.ActionPlanConditionerV1(
            self.config, renderer_hidden_width=8
        )
        before = copy.deepcopy(receiver.state_dict())

        bf16_state = copy.deepcopy(state)
        bf16_state["predictor.phase_queries"] = bf16_state[
            "predictor.phase_queries"
        ].to(torch.bfloat16)
        with self.assertRaisesRegex(
            RuntimeError, "rejected before load/cast.*bfloat16.*float32"
        ):
            receiver.load_state_dict(bf16_state, strict=True)
        for name, value in receiver.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

        changed_buffer = copy.deepcopy(state)
        changed_buffer["predictor.phase_position_encoding_abi"] = changed_buffer[
            "predictor.phase_position_encoding_abi"
        ].clone()
        changed_buffer["predictor.phase_position_encoding_abi"][0].add_(1)
        with self.assertRaisesRegex(RuntimeError, "semantic buffer value differs"):
            receiver.load_state_dict(changed_buffer, strict=True)
        for name, value in receiver.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()
