from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("v16r2 online-anchor tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r2 as method


WORKER = ROOT / "scripts/auh_train_online_anchor_full644_dynamic_static_v16r2.sh"
CONTROLLER = ROOT / "scripts/auh_launch_online_anchor_full644_dynamic_static_v16r2.sh"


class Full644DynamicStaticV16R2Test(unittest.TestCase):
    def setUp(self) -> None:
        method._RUNTIME_AUDIT = method._empty_runtime_audit()
        method._ACTIVE_OPTIMIZER = None
        method._ACTIVE_MAX_GRAD_NORM = 10.0
        method.v16._RUNTIME_AUDIT = method.v16._empty_runtime_audit()

    def tearDown(self) -> None:
        method._ACTIVE_OPTIMIZER = None
        method._ACTIVE_MAX_GRAD_NORM = None

    @staticmethod
    def tensors():
        named = tuple(
            (
                f"adapter_{index:03d}",
                torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32)),
            )
            for index in range(method.EXPECTED_TRAINABLE_TENSOR_COUNT)
        )
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=1.0e-5,
            weight_decay=0.0,
        )
        return named, optimizer

    @staticmethod
    def build_action_ascent_candidate(named, optimizer):
        # Establish a negative first moment.  The current positive gradient
        # then produces a positive AdamW displacement (action ascent).
        for _ in range(20):
            for _, parameter in named:
                parameter.grad = torch.full_like(parameter, -1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)
        before = tuple(parameter.detach().clone() for _, parameter in named)
        actions = tuple(torch.ones_like(parameter) for _, parameter in named)
        replays = tuple(torch.full_like(parameter, 0.5) for _, parameter in named)
        for _, parameter in named:
            parameter.grad.copy_(torch.ones_like(parameter))
        optimizer.step()
        return before, actions, replays

    def test_unanimous_action_ascent_rolls_back_resets_and_reprobes(self):
        named, optimizer = self.tensors()
        before, actions, replays = self.build_action_ascent_candidate(
            named, optimizer
        )
        action_copies = tuple(action.clone() for action in actions)
        method._ACTIVE_OPTIMIZER = optimizer
        method.v16._RUNTIME_AUDIT["manifest_iids"] = tuple(
            f"iid-{index:03d}" for index in range(22)
        )

        # An independent empty-state, action-only AdamW step is the exact
        # reference semantics promised by the fallback.
        reference = tuple(
            torch.nn.Parameter(value.detach().clone()) for value in before
        )
        reference_optimizer = torch.optim.AdamW(
            reference, lr=1.0e-5, weight_decay=0.0
        )
        for parameter, action in zip(reference, actions):
            parameter.grad = action.detach().clone()
        torch.nn.utils.clip_grad_norm_(reference, 10.0)
        reference_optimizer.step()

        values = method.actual_optimizer_update_probe(
            named,
            before,
            actions,
            replays,
            replay_combine_mode=method.EXPECTED_REPLAY_MODE,
            step=22,
        )

        self.assertTrue(values["action_descent_passed"])
        self.assertTrue(values["v16r2_actual_action_descent_fallback_applied"])
        self.assertFalse(values["optimizer_semantics_observed_not_modified"])
        self.assertFalse(values["v16r2_action_descent_gate_relaxed"])
        self.assertEqual(method._RUNTIME_AUDIT["fallback_steps"], [22])
        self.assertEqual(
            method._RUNTIME_AUDIT["fallback_target_iids"], ["iid-021"]
        )
        self.assertEqual(len(optimizer.state), len(named))
        self.assertEqual(
            {
                int(float(optimizer.state[parameter]["step"].item()))
                for _, parameter in named
            },
            {1},
        )
        for (_, actual), expected in zip(named, reference):
            self.assertTrue(torch.equal(actual.detach(), expected.detach()))
        for action, original in zip(actions, action_copies):
            self.assertTrue(torch.equal(action, original))

    def test_normal_frozen_probe_pass_does_not_reset_optimizer(self):
        named, optimizer = self.tensors()
        before = tuple(parameter.detach().clone() for _, parameter in named)
        actions = tuple(torch.ones_like(parameter) for _, parameter in named)
        replays = tuple(torch.full_like(parameter, 0.5) for _, parameter in named)
        for (_, parameter), action in zip(named, actions):
            parameter.grad = action.detach().clone()
        optimizer.step()
        method._ACTIVE_OPTIMIZER = optimizer

        values = method.actual_optimizer_update_probe(
            named,
            before,
            actions,
            replays,
            replay_combine_mode=method.EXPECTED_REPLAY_MODE,
            step=1,
        )
        self.assertTrue(values["action_descent_passed"])
        self.assertFalse(values["v16r2_actual_action_descent_fallback_applied"])
        self.assertTrue(values["optimizer_semantics_observed_not_modified"])
        self.assertEqual(method._RUNTIME_AUDIT["fallback_count"], 0)

    def test_retry_is_once_and_a_second_failure_does_not_commit(self):
        named, optimizer = self.tensors()
        before, actions, replays = self.build_action_ascent_candidate(
            named, optimizer
        )
        method._ACTIVE_OPTIMIZER = optimizer
        method.v16._RUNTIME_AUDIT["manifest_iids"] = tuple(
            f"iid-{index:03d}" for index in range(22)
        )
        calls = 0

        def reject(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise method.base.OnlineAnchorTrainingError(
                method.ACTION_ASCENT_PREFIX + " synthetic"
            )

        with mock.patch.object(
            method, "_BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE", side_effect=reject
        ):
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.actual_optimizer_update_probe(
                    named,
                    before,
                    actions,
                    replays,
                    replay_combine_mode=method.EXPECTED_REPLAY_MODE,
                    step=22,
                )
        self.assertEqual(calls, 2)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(method._RUNTIME_AUDIT["fallback_count"], 0)
        for (_, parameter), value in zip(named, before):
            self.assertTrue(torch.equal(parameter.detach(), value))

    def test_non_target_probe_error_never_triggers_fallback(self):
        named, optimizer = self.tensors()
        before = tuple(parameter.detach().clone() for _, parameter in named)
        actions = tuple(torch.ones_like(parameter) for _, parameter in named)
        replays = tuple(torch.ones_like(parameter) for _, parameter in named)
        for _, parameter in named:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        method._ACTIVE_OPTIMIZER = optimizer

        def reject(*_args, **_kwargs):
            raise method.base.OnlineAnchorTrainingError(
                "actual optimizer update changed no LoRA parameter"
            )

        with mock.patch.object(
            method, "_BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE", side_effect=reject
        ):
            with self.assertRaisesRegex(
                method.base.OnlineAnchorTrainingError,
                "changed no LoRA parameter",
            ):
                method.actual_optimizer_update_probe(
                    named,
                    before,
                    actions,
                    replays,
                    replay_combine_mode=method.EXPECTED_REPLAY_MODE,
                    step=1,
                )
        self.assertEqual(method._RUNTIME_AUDIT["fallback_count"], 0)

    def test_optimizer_parameter_order_mismatch_fails_closed(self):
        named, optimizer = self.tensors()
        method._ACTIVE_OPTIMIZER = optimizer
        with self.assertRaises(method.base.OnlineAnchorTrainingError):
            method._validate_optimizer_closure(tuple(reversed(named)), optimizer)

    def test_mixed_rank_probe_category_fails_closed(self):
        def mixed(counts, op=None):
            del op
            counts.copy_(torch.tensor([3, 1, 0], dtype=counts.dtype))

        with mock.patch.object(torch.distributed, "is_available", return_value=True), \
             mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=4), \
             mock.patch.object(torch.distributed, "all_reduce", side_effect=mixed):
            with self.assertRaisesRegex(
                method.base.OnlineAnchorTrainingError,
                "differs across ranks",
            ):
                method._collective_category(
                    "pass", device=torch.device("cpu"), phase="synthetic"
                )

    def test_receipt_records_reset_not_optimizer_state_restore(self):
        event = {
            "step": 22,
            "target_iid": "iid-021",
            "failed_candidate_committed": False,
            "parameter_values_exactly_restored_before_retry": True,
            "optimizer_state_restored": False,
            "optimizer_state_reset": True,
            "committed_retry_reprobed_by_frozen_authority": True,
            "committed_retry": {"action_descent_passed": True},
        }
        method._RUNTIME_AUDIT.update(
            {
                "fallback_count": 1,
                "fallback_steps": [22],
                "fallback_target_iids": ["iid-021"],
                "fallback_geometry": [event],
                "optimizer_state_reset_count": 1,
            }
        )
        inherited = {
            "schema_version": method.v16.RECEIPT_SCHEMA,
            "global_step": 28,
            "training_contract": {"method": method.v16.METHOD},
        }
        with mock.patch.object(
            method, "_V16_CHECKPOINT_RECEIPT", return_value=inherited
        ):
            receipt = method.checkpoint_receipt(args=object())
        summary = receipt["v16r2_actual_action_descent_fallback_summary"]
        contract = receipt["training_contract"]
        self.assertEqual(receipt["schema_version"], method.RECEIPT_SCHEMA)
        self.assertFalse(summary["optimizer_state_restored"])
        self.assertTrue(summary["optimizer_state_reset_before_each_retry"])
        self.assertFalse(summary["optimizer_history_matches_uninterrupted_adamw"])
        self.assertFalse(contract["actual_action_descent_gate_relaxed"])
        self.assertEqual(contract["actual_action_descent_fallback_steps"], [22])

    def test_validation_adds_v16r2_namespace_and_clip_gate(self):
        good = SimpleNamespace(output="/tmp/fresh-v16r2-s644", max_grad_norm=10.0)
        with mock.patch.object(method, "_V16_VALIDATE_ARGS"):
            method.validate_args(good)
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(
                    SimpleNamespace(output="/tmp/fresh-v16-s644", max_grad_norm=10.0)
                )
            with self.assertRaises(method.base.OnlineAnchorTrainingError):
                method.validate_args(
                    SimpleNamespace(output="/tmp/fresh-v16r2-s644", max_grad_norm=9.0)
                )

    def test_main_restores_all_monkeypatches_on_failure(self):
        original_adamw = torch.optim.AdamW
        original_probe = method.base.actual_optimizer_update_probe
        original_validate = method.v16.validate_args
        original_receipt = method.v16.checkpoint_receipt

        def observe(_argv):
            self.assertIsNot(torch.optim.AdamW, original_adamw)
            self.assertIs(
                method.base.actual_optimizer_update_probe,
                method.actual_optimizer_update_probe,
            )
            self.assertIs(method.v16.validate_args, method.validate_args)
            self.assertIs(method.v16.checkpoint_receipt, method.checkpoint_receipt)
            raise RuntimeError("synthetic stop")

        with mock.patch.object(method.v16, "main", side_effect=observe):
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                method.main([])
        self.assertIs(torch.optim.AdamW, original_adamw)
        self.assertIs(method.base.actual_optimizer_update_probe, original_probe)
        self.assertIs(method.v16.validate_args, original_validate)
        self.assertIs(method.v16.checkpoint_receipt, original_receipt)
        self.assertIsNone(method._ACTIVE_OPTIMIZER)
        self.assertIsNone(method._ACTIVE_MAX_GRAD_NORM)

    def test_worker_and_controller_pin_v16r2_contract(self):
        worker = WORKER.read_text(encoding="utf-8")
        controller = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn(
            "train_online_anchor_attention_full644_dynamic_static_v16r2.py",
            worker,
        )
        self.assertIn("v16r2.method-source", worker)
        self.assertIn("$method_prefix.tar", worker)
        self.assertIn("run_exact644 644", controller)
        self.assertIn(method.RECEIPT_SCHEMA, controller)
        self.assertIn(method.METHOD, controller)
        self.assertIn("actual_action_descent_gate_relaxed == false", controller)
        self.assertIn("index(22) != null", controller)


if __name__ == "__main__":
    unittest.main()
