from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = METHOD_ROOT / "train_source_noised_carrier_strata_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_source_noised_carrier_strata_v1 as trainer


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class SourceNoisedCarrierStrataPureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = TRAINER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(TRAINER_PATH))

    def test_fixed_four_step_exact40_registry_is_bit_pinned(self) -> None:
        coordinates = trainer.validate_registered_schedule()
        self.assertEqual(
            tuple(item.schedule_index for item in coordinates),
            (16, 29, 35, 38),
        )
        self.assertEqual(
            tuple(item.timestep for item in coordinates),
            (882, 655, 418, 211),
        )
        self.assertEqual(
            tuple(item.sigma_float32_be_hex for item in coordinates),
            ("3f61ed37", "3f27d446", "3ed6539a", "3e58b351"),
        )
        self.assertEqual(
            trainer.EXPECTED_EXACT40_SCHEDULE_SHA256,
            "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03",
        )
        self.assertTrue(
            all(
                left.sigma > right.sigma
                for left, right in zip(coordinates, coordinates[1:])
            )
        )

    def test_custom_reordered_or_hostile_schedule_is_rejected(self) -> None:
        hostile_values = (
            [16, 29, 35, 38],
            (16, 29, 38, 35),
            (0, 1, 2, 3),
            (16, 29, 35),
            (),
            None,
        )
        for value in hostile_values:
            with self.subTest(value=value):
                with self.assertRaises(trainer.SourceNoisedCarrierTrainingError):
                    trainer.validate_registered_schedule(value)
        with mock.patch.object(trainer.exact40, "SCHEDULE_SHA256", "0" * 64):
            with self.assertRaises(trainer.SourceNoisedCarrierTrainingError):
                trainer.validate_registered_schedule()
        with mock.patch.object(
            trainer.exact40,
            "PINNED_TIMESTEPS",
            tuple(
                883 if index == 16 else value
                for index, value in enumerate(trainer.exact40.PINNED_TIMESTEPS)
            ),
        ):
            with self.assertRaises(trainer.SourceNoisedCarrierTrainingError):
                trainer.validate_registered_schedule()

    def test_step_lookup_rejects_wraparound_and_non_integer_indices(self) -> None:
        for hostile in (-1, 4, True, 1.0, "1", None):
            with self.subTest(hostile=hostile):
                with self.assertRaises(trainer.SourceNoisedCarrierTrainingError):
                    trainer.registered_coordinate(hostile)
        self.assertEqual(trainer.registered_coordinate(3).schedule_index, 38)

    def test_forward_noising_cannot_be_relabelled_inversion(self) -> None:
        trainer.validate_scientific_claim_contract()
        for kwargs in (
            {"inversion_claimed": True},
            {"exact_roundtrip_claimed": True},
            {"semantic_method_success_claimed": True},
            {"inversion_claimed": 0},
            {"exact_roundtrip_claimed": None},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(trainer.SourceNoisedCarrierTrainingError):
                    trainer.validate_scientific_claim_contract(**kwargs)

    def test_optimizer_gate_requires_exactly_two_complete_logical_arms(self) -> None:
        self.assertTrue(
            trainer.authorize_optimizer_step(
                completed_control_arms=(0, 1),
                completed_backward_arms=(0, 1),
            )
        )
        hostile_pairs = (
            ((0,), (0, 1)),
            ((0, 1), (0,)),
            ((1, 0), (0, 1)),
            ((0, 1), (1, 0)),
            ([0, 1], (0, 1)),
            ((0, 1), [0, 1]),
            ((0, 1, 2), (0, 1, 2)),
        )
        for controls, backwards in hostile_pairs:
            with self.subTest(controls=controls, backwards=backwards):
                with self.assertRaises(trainer.SourceNoisedCarrierTrainingError):
                    trainer.authorize_optimizer_step(
                        completed_control_arms=controls,
                        completed_backward_arms=backwards,
                    )

    def test_all_four_sigmas_are_real_updates_without_late_zero_gate(self) -> None:
        receipt = trainer.fixed_plan_receipt()
        self.assertEqual(receipt["optimizer_steps"], 4)
        self.assertEqual(receipt["registered_schedule_indices"], [16, 29, 35, 38])
        self.assertIs(receipt["optimizer_step_per_registered_sigma"], True)
        self.assertIs(receipt["all_registered_strata_optimizer_authorized"], True)
        self.assertIs(receipt["late_or_low_sigma_zero_update_gate_present"], False)
        self.assertIs(receipt["schedule_customization_authorized"], False)
        self.assertNotIn("sigma", trainer.authorize_optimizer_step.__code__.co_varnames)
        self.assertIn("for coordinate in validate_registered_schedule():", self.source)
        self.assertIn("optimizer.step()", self.source)
        self.assertIn('"optimizer_update_authorized_for_this_sigma": True', self.source)
        self.assertNotIn("if coordinate.sigma", self.source)
        self.assertNotIn("if sigma <", self.source)
        self.assertNotIn("if sigma <=", self.source)

    def test_same_epsilon_forward_carrier_and_clean_refs_are_explicit(self) -> None:
        for fragment in (
            "noisy_target = _shared_noise_state(clean, epsilon, coordinate.sigma)",
            "_shared_noise_state(value, epsilon, coordinate.sigma)",
            "velocity = (epsilon - clean).detach().contiguous()",
            '"same_epsilon_object_reused_during_target_and_donor_construction": True',
            '"target_formula_recomputed_and_equal"',
            '"donor_formula_recomputed_and_equal"',
            '"clean_source_references_routed": True',
            '"references_independently_encoded_from_source_rgb": True',
            '"reference_from_video_posterior_slice": False',
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("source_rich_conditional_base", self.source)
        self.assertNotIn("rho", self.source.lower())

    def test_timestep_is_per_condition_not_a_global_sigma1_constant(self) -> None:
        self.assertIn("coordinate: RegisteredCarrierCoordinate", self.source)
        self.assertIn(
            "timesteps=embedded.new_tensor([coordinate.timestep], dtype=torch.int64)",
            self.source,
        )
        self.assertNotIn("TIMESTEP = 1000", self.source)
        self.assertNotIn("SIGMA = 1.0", self.source)

    def test_both_arm_controls_precede_any_backward_in_each_step(self) -> None:
        step_loop = self.source.index(
            "for coordinate in validate_registered_schedule():"
        )
        control_loop = self.source.index(
            "for logical_arm in local_logical_arms:", step_loop
        )
        completed_controls = self.source.index(
            "if tuple(completed_control_arms)", control_loop
        )
        backward_loop = self.source.index(
            "for logical in prepared_arms:", completed_controls
        )
        backward = self.source.index(".backward()", backward_loop)
        optimizer = self.source.index("optimizer.step()", backward)
        self.assertLess(control_loop, completed_controls)
        self.assertLess(completed_controls, backward_loop)
        self.assertLess(backward_loop, backward)
        self.assertLess(backward, optimizer)
        self.assertIn("(loss * loss_scale).backward()", self.source)
        self.assertIn("if loss_scale != 0.5", self.source)

    def test_world4_only_and_no_semantic_success_claim(self) -> None:
        for fragment in (
            'choices=("world4-dp1-sp4",)',
            "topology != runtime.WORLD4_DP1_SP4",
            '"semantic_motion_preservation_claimed": False',
            '"natural_semantic_action_learned": False',
            '"action_editing_claim_authorized": False',
            '"method_success_claimed": False',
            '"scientific_claim_authorized": False',
        ):
            self.assertIn(fragment, self.source)

    def test_checkpoint_recomputation_uses_the_same_explicit_role_route(self) -> None:
        self.assertIn(
            '"context_fn": role.checkpoint_route_context_fn', self.source
        )
        self.assertNotIn(
            'gradient_checkpointing_kwargs={"use_reentrant": False}', self.source
        )
        self.assertIn(
            "absent.total_tokens >= main.layout.total_tokens", self.source
        )
        self.assertNotIn("absent.total_tokens >= main.total_tokens", self.source)

    def test_receipt_commits_to_checkpoint_route_context_replay(self) -> None:
        for fragment in (
            '"gradient_checkpointing_non_reentrant": True',
            '"source_self_role_repaint.checkpoint_route_context_fn"',
            '"checkpoint_recomputation_route_context_replayed": True',
        ):
            self.assertIn(fragment, self.source)

    def test_rank_zero_artifact_failure_is_propagated_to_every_rank(self) -> None:
        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        guarded_try = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Constant)
                and child.value == "adapter.safetensors"
                for child in ast.walk(node)
            )
        )
        call_names = {
            _dotted_name(child.func)
            for child in ast.walk(guarded_try)
            if isinstance(child, ast.Call)
        }
        for call_name in (
            "_atomic_adapter_safetensors",
            "runtime.atomic_torch_save",
            "runtime.atomic_json",
            "runtime.verify_staged_run_bundle",
            "runtime.fsync_directory",
        ):
            self.assertIn(call_name, call_names)
        self.assertEqual(len(guarded_try.handlers), 1)
        self.assertTrue(
            any(
                isinstance(child, ast.Name)
                and child.id == "rank_zero_publication_error"
                for child in ast.walk(guarded_try.handlers[0])
            )
        )

        publish_calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "publish_output_transaction"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "runtime"
        ]
        self.assertEqual(len(publish_calls), 1)
        keywords = {item.arg: item.value for item in publish_calls[0].keywords}
        self.assertIn("rank_zero_error", keywords)
        self.assertIsInstance(keywords["rank_zero_error"], ast.Name)
        self.assertEqual(keywords["rank_zero_error"].id, "rank_zero_publication_error")
        self.assertGreater(publish_calls[0].lineno, guarded_try.end_lineno)

    def test_fixed_plan_digest_is_stable(self) -> None:
        self.assertEqual(
            trainer.fixed_plan_receipt()["digest"],
            "2a1ca619eb333ef4a105ea1f548f6efb380c361f893f69ef36d5f330fb3641d4",
        )


if __name__ == "__main__":
    unittest.main()
