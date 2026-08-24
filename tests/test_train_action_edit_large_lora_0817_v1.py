#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextvars import Context
import functools
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPO_ROOT
    / "methods"
    / "bernini_action_editing"
    / "train_action_edit_large_lora_0817_v1.py"
)
PREDICTOR_PATH = RUNNER_PATH.with_name("action_plan_predictor_v1.py")
SPEC = importlib.util.spec_from_file_location(
    "train_action_edit_large_lora_0817_v1_test_subject", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def experimental_summary() -> dict[str, object]:
    return {
        "schema_version": "bernini-r-action-vae-dataset-summary-v2",
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "complete": True,
        "expected_sample_count": 644,
        "materialized_sample_count": 644,
        "missing_sample_count": 0,
        "raw_strict_selection_rows": 359,
        "raw_non_strict_selection_rows": 285,
        "materialized_strict_selection_rows": 359,
        "materialized_non_strict_selection_rows": 285,
        "frame_count": 81,
        "latent_frame_count": 21,
        "fps": 25.0,
    }


class ExperimentalAuthorityTests(unittest.TestCase):
    def test_accepts_only_explicit_nonformal_legacy_authority(self) -> None:
        receipt = runner.validate_experimental_authority(experimental_summary())
        self.assertEqual(receipt["authority"], "PRE_D0_ENGINEERING_ONLY")
        self.assertFalse(receipt["formal_training_authorized"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertIn("not_0817_qualified", receipt["target_quality_status"])

    def test_rejects_formal_or_scientific_authority_fabrication(self) -> None:
        for key in ("training_authorized", "scientific_claim_authorized"):
            value = experimental_summary()
            value[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(
                runner.PreD0EngineeringError, "authority differs"
            ):
                runner.validate_experimental_authority(value)

    def test_rejects_unacknowledged_experimental_training(self) -> None:
        value = experimental_summary()
        value["experimental_training_acknowledged"] = False
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "experimental_training_acknowledged"
        ):
            runner.validate_experimental_authority(value)


class StrictScheduleTests(unittest.TestCase):
    @staticmethod
    def _catalog(count: int = 18, strict: int = 16):
        rows = []
        for index in range(count):
            # Deliberately reverse IID order to prove canonical sorting.
            iid = f"iid-{count - index:04d}"
            rows.append(
                {
                    "iid": iid,
                    "row_index": index,
                    "strict_selection_gates_all_true": index < strict,
                    "parquet_name": f"{iid}.parquet",
                    "posterior_parameter_shape": [1, 32, 21, 8, 8],
                    "tokens_per_role": 21 * 4 * 4,
                }
            )
        return rows

    def test_mechanical_strict_only_and_two_step_dp2_schedule(self) -> None:
        strict = runner.validate_strict_catalog(
            self._catalog(), expected_rows=18, expected_strict=16
        )
        self.assertEqual(len(strict), 16)
        self.assertTrue(
            all(row["strict_selection_gates_all_true"] is True for row in strict)
        )
        self.assertEqual(
            [row["iid"] for row in strict],
            sorted(row["iid"] for row in strict),
        )
        selected = runner.strict_two_step_schedule(strict)
        observed = []
        for step in range(2):
            for microbatch in range(4):
                for arm in range(2):
                    observed.append(
                        runner.schedule_row(
                            selected,
                            optimizer_step_zero_based=step,
                            microbatch_index=microbatch,
                            dp_arm=arm,
                        )["iid"]
                    )
        self.assertEqual(observed, [row["iid"] for row in selected])
        self.assertEqual(len(set(observed)), 16)

    def test_later_two_row_global_strict_max_tier_is_cycled_to_16_records(self) -> None:
        rows = []
        for index in range(18):
            is_maximum = index >= 16
            iid = f"{'z-max' if is_maximum else 'a-small'}-{index:02d}"
            shape = [1, 32, 21, 10, 12] if is_maximum else [1, 32, 21, 8, 8]
            rows.append(
                {
                    "iid": iid,
                    "row_index": index,
                    "strict_selection_gates_all_true": True,
                    "parquet_name": f"{iid}.parquet",
                    "posterior_parameter_shape": shape,
                    "tokens_per_role": 21 * (shape[3] // 2) * (shape[4] // 2),
                }
            )
        strict = runner.validate_strict_catalog(
            rows, expected_rows=18, expected_strict=18
        )
        selected = runner.strict_two_step_schedule(strict)
        self.assertEqual(len(selected), 16)
        self.assertEqual(
            [row["iid"] for row in selected],
            [f"z-max-{16 + (index % 2):02d}" for index in range(16)],
        )
        self.assertEqual(len({row["iid"] for row in selected}), 2)
        self.assertTrue(all(row["tokens_per_role"] == 21 * 5 * 6 for row in selected))

    def test_strict_maximum_tier_requires_two_distinct_rows(self) -> None:
        rows = self._catalog(count=18, strict=18)
        rows[-1] = {
            **rows[-1],
            "posterior_parameter_shape": [1, 32, 21, 10, 12],
            "tokens_per_role": 21 * 5 * 6,
        }
        strict = runner.validate_strict_catalog(
            rows, expected_rows=18, expected_strict=18
        )
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "fewer than two distinct DP rows"
        ):
            runner.strict_two_step_schedule(strict)

    def test_catalog_rejects_nonstrict_count_or_shard_identity_drift(self) -> None:
        rows = self._catalog()
        rows[0] = {**rows[0], "parquet_name": "wrong.parquet"}
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "catalog schema differs"
        ):
            runner.validate_strict_catalog(
                rows, expected_rows=18, expected_strict=16
            )
        rows = self._catalog()
        rows[15] = {**rows[15], "strict_selection_gates_all_true": False}
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "strict row count differs"
        ):
            runner.validate_strict_catalog(
                rows, expected_rows=18, expected_strict=16
            )

    def test_only_two_updates_are_reachable(self) -> None:
        strict = runner.validate_strict_catalog(
            self._catalog(), expected_rows=18, expected_strict=16
        )
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "exactly two optimizer updates"
        ):
            runner.strict_two_step_schedule(strict, max_steps=3)


class SP4GlobalPhaseBridgeTests(unittest.TestCase):
    @staticmethod
    def _routes(spatial_tokens_per_phase: int):
        tokens = 21 * spatial_tokens_per_phase
        return [
            runner.ActionInjectionRoute(
                source_tokens=tokens,
                target_tokens=tokens,
                sequence_parallel_rank=rank,
                sequence_parallel_size=4,
                plan=object(),
                row_identity="row",
            )
            for rank in range(4)
        ]

    @staticmethod
    def _record_forward(route):
        recompute_contexts = []
        for block_index in range(runner.TRANSFORMER_BLOCKS):
            forward_context, recompute_context = (
                runner.action_route_checkpoint_context_fn()
            )
            with forward_context:
                binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                assert binding is not None
                route.record_block_call(
                    phase=binding.phase,
                    checkpoint_index=binding.checkpoint_index,
                    block_index=block_index,
                )
            recompute_contexts.append(recompute_context)
        route.validate_forward_traversal()
        return recompute_contexts

    @classmethod
    def _record_exact_forward_and_recompute(cls, route):
        with runner.activate_action_route(route):
            recompute_contexts = cls._record_forward(route)
            for block_index in reversed(range(runner.TRANSFORMER_BLOCKS)):
                with recompute_contexts[block_index]:
                    binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                    assert binding is not None
                    route.record_block_call(
                        phase=binding.phase,
                        checkpoint_index=binding.checkpoint_index,
                        block_index=block_index,
                    )
            route.validate_forward_and_recompute_traversal()
        if runner._ACTIVE_ACTION_ROUTE.get() is not None:
            raise AssertionError("action route leaked after exact replay")
        if not runner._ACTION_ROUTE_LEASE_OWNER.is_empty():
            raise AssertionError("action route lease leaked after exact replay")

    def test_divisible_geometry_splits_one_phase_across_target_ranks(self) -> None:
        routes = self._routes(spatial_tokens_per_phase=2)
        local = [route.local_phase_indices_tuple() for route in routes]
        self.assertEqual({len(value) for value in local}, {21})
        self.assertTrue(all(value == -1 for value in local[0] + local[1]))
        # Target phase 10 straddles the rank-2/rank-3 boundary.
        self.assertEqual(local[2][-1], 10)
        self.assertEqual(local[3][0], 10)
        flattened = [phase for shard in local for phase in shard if phase >= 0]
        self.assertEqual(
            {phase: flattened.count(phase) for phase in range(21)},
            {phase: 2 for phase in range(21)},
        )

    def test_nondivisible_geometry_covers_boundary_and_append_padding(self) -> None:
        routes = self._routes(spatial_tokens_per_phase=1)
        local = [route.local_phase_indices_tuple() for route in routes]
        self.assertEqual({len(value) for value in local}, {11})
        # Rank 1 crosses source->target; rank 3 owns two append-padding rows.
        self.assertEqual(local[1][:-1], (-1,) * 10)
        self.assertEqual(local[1][-1], 0)
        self.assertEqual(local[3][-2:], (-1, -1))
        flattened = [phase for shard in local for phase in shard if phase >= 0]
        self.assertEqual(flattened, list(range(21)))

    def test_exact_forward_and_checkpoint_recompute_traversal(self) -> None:
        route = self._routes(spatial_tokens_per_phase=1)[3]
        self._record_exact_forward_and_recompute(route)
        receipt = route.receipt()
        self.assertEqual(set(receipt["block_call_counts"]), {str(i) for i in range(30)})
        self.assertTrue(all(value == 2 for value in receipt["block_call_counts"].values()))
        self.assertEqual(receipt["checkpoint_context_captures"], 30)
        self.assertEqual(receipt["checkpoint_forward_contexts"], 30)
        self.assertEqual(receipt["checkpoint_recompute_contexts"], 30)
        self.assertFalse(receipt["source_or_padding_written"])

    def test_missing_or_third_hook_pass_fails_closed(self) -> None:
        route = self._routes(spatial_tokens_per_phase=1)[0]
        with runner.activate_action_route(route):
            recompute_contexts = self._record_forward(route)
            for block_index in reversed(range(1, runner.TRANSFORMER_BLOCKS)):
                with recompute_contexts[block_index]:
                    binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                    assert binding is not None
                    route.record_block_call(
                        phase=binding.phase,
                        checkpoint_index=binding.checkpoint_index,
                        block_index=block_index,
                    )
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "exactly one forward"
            ):
                route.validate_forward_and_recompute_traversal()

        route = self._routes(spatial_tokens_per_phase=1)[0]
        with runner.activate_action_route(route):
            recompute_contexts = self._record_forward(route)
            with recompute_contexts[0]:
                binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                assert binding is not None
                route.record_block_call(
                    phase=binding.phase,
                    checkpoint_index=binding.checkpoint_index,
                    block_index=0,
                )
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "recompute context was reused"
            ):
                with runner._replay_action_checkpoint_route(
                    route,
                    lease_serial=route._lease_serial,
                    phase=runner._ACTION_CHECKPOINT_RECOMPUTE,
                    checkpoint_index=0,
                ):
                    pass

    def test_fresh_context_rebinds_exact_route_and_stale_reuse_fails(self) -> None:
        route = self._routes(spatial_tokens_per_phase=1)[0]
        with runner.activate_action_route(route):
            forward_context, recompute_context = (
                runner.action_route_checkpoint_context_fn()
            )
            with forward_context:
                binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                assert binding is not None
                route.record_block_call(
                    phase=binding.phase,
                    checkpoint_index=binding.checkpoint_index,
                    block_index=0,
                )

            def fresh_recompute():
                self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
                with recompute_context:
                    self.assertIs(runner.active_action_route(), route)
                    binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                    assert binding is not None
                    route.record_block_call(
                        phase=binding.phase,
                        checkpoint_index=binding.checkpoint_index,
                        block_index=0,
                    )
                self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())

            Context().run(fresh_recompute)
        self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "stale or reused"
        ):
            with runner.activate_action_route(route):
                pass

        stale = self._routes(spatial_tokens_per_phase=1)[1]
        with runner.activate_action_route(stale):
            _, stale_recompute = runner.action_route_checkpoint_context_fn()
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "stale action route"
        ):
            Context().run(stale_recompute.__enter__)
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())

    def test_exception_cleanup_allows_next_identity_without_mixing(self) -> None:
        first = self._routes(spatial_tokens_per_phase=1)[0]
        with self.assertRaisesRegex(RuntimeError, "synthetic recompute failure"):
            with runner.activate_action_route(first):
                forward_context, recompute_context = (
                    runner.action_route_checkpoint_context_fn()
                )
                with forward_context:
                    binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
                    assert binding is not None
                    first.record_block_call(
                        phase=binding.phase,
                        checkpoint_index=binding.checkpoint_index,
                        block_index=0,
                    )

                def fail_in_fresh_context():
                    with recompute_context:
                        self.assertIs(runner.active_action_route(), first)
                        raise RuntimeError("synthetic recompute failure")

                Context().run(fail_in_fresh_context)
        self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
        self.assertIsNone(runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get())
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())

        second = self._routes(spatial_tokens_per_phase=1)[1]
        second.row_identity = "next-row"
        self._record_exact_forward_and_recompute(second)
        self.assertEqual(second.receipt()["row_identity"], "next-row")
        self.assertNotEqual(first.row_identity, second.row_identity)

    def test_different_route_in_fresh_context_fails_closed(self) -> None:
        first = self._routes(spatial_tokens_per_phase=1)[0]
        second = runner.ActionInjectionRoute(
            source_tokens=first.source_tokens,
            target_tokens=first.target_tokens,
            sequence_parallel_rank=first.sequence_parallel_rank,
            sequence_parallel_size=first.sequence_parallel_size,
            plan=first.plan,
            row_identity=first.row_identity,
        )
        self.assertIsNot(first, second)
        self.assertEqual(first, second)
        with runner.activate_action_route(first):
            _, stale_recompute = runner.action_route_checkpoint_context_fn()

            def enter_different_route():
                self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
                with runner.activate_action_route(second):
                    pass

            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "concurrent or nested"
            ):
                Context().run(enter_different_route)
            self.assertIs(runner.active_action_route(), first)
        self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())

        with runner.activate_action_route(second):

            def replay_stale_first():
                with stale_recompute:
                    pass

            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "stale action route"
            ):
                Context().run(replay_stale_first)
            self.assertIs(runner.active_action_route(), second)
        self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())

    def test_missing_context_fn_reproduces_fresh_context_route_failure(self) -> None:
        try:
            import torch
            from torch.utils.checkpoint import checkpoint
        except ImportError:
            self.skipTest("PyTorch unavailable")

        route = self._routes(spatial_tokens_per_phase=1)[0]
        value = torch.linspace(-0.5, 0.5, 17, dtype=torch.float32).requires_grad_()
        calls = []

        def checkpointed(item):
            active = runner.active_action_route()
            if active is None:
                runner.fail("Bernini block executed without authenticated action route")
            calls.append(active)
            return torch.sin(item)

        with runner.activate_action_route(route):
            result = checkpoint(checkpointed, value, use_reentrant=False)
            self.assertEqual(calls, [route])
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError,
                "without authenticated action route",
            ):
                Context().run(result.square().sum().backward)
        self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())

    def test_real_nonreentrant_checkpoint_recomputes_all_30_under_route(self) -> None:
        try:
            import torch
            from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
        except ImportError:
            self.skipTest("PyTorch unavailable")

        route = self._routes(spatial_tokens_per_phase=1)[2]
        value = torch.linspace(-0.5, 0.5, 17, dtype=torch.float32).requires_grad_()

        class TinyBlock(torch.nn.Module):
            def __init__(self, block_index):
                super().__init__()
                self.scale = 1.0 + block_index / 100.0

            def forward(self, item):
                # A nonlinear saved tensor forces non-reentrant recomputation.
                return torch.sin(item * self.scale)

        blocks = [TinyBlock(index) for index in range(runner.TRANSFORMER_BLOCKS)]

        def post_forward_hook(_module, _inputs, output, *, block_index):
            active = runner.active_action_route()
            self.assertIs(active, route)
            binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
            self.assertIsNotNone(binding)
            active.record_block_call(
                phase=binding.phase,
                checkpoint_index=binding.checkpoint_index,
                block_index=block_index,
            )
            return output

        for block_index, block in enumerate(blocks):
            block.register_forward_hook(
                functools.partial(post_forward_hook, block_index=block_index)
            )

        with runner.activate_action_route(route):
            with set_checkpoint_early_stop(False):
                result = value
                for block in blocks:
                    result = checkpoint(
                        block,
                        result,
                        use_reentrant=False,
                        context_fn=runner.action_route_checkpoint_context_fn,
                    )
                result.square().sum().backward()
            route.validate_forward_and_recompute_traversal()
        self.assertIsNotNone(value.grad)
        self.assertEqual(route.receipt()["checkpoint_recompute_contexts"], 30)
        self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())

    def test_default_checkpoint_early_stop_omits_hooks_and_is_rejected(self) -> None:
        try:
            import torch
            from torch.utils.checkpoint import checkpoint
        except ImportError:
            self.skipTest("PyTorch unavailable")

        route = self._routes(spatial_tokens_per_phase=1)[2]
        value = torch.linspace(-0.5, 0.5, 17, dtype=torch.float32).requires_grad_()

        class TinyBlock(torch.nn.Module):
            def __init__(self, block_index):
                super().__init__()
                self.scale = 1.0 + block_index / 100.0

            def forward(self, item):
                return torch.sin(item * self.scale)

        blocks = [TinyBlock(index) for index in range(runner.TRANSFORMER_BLOCKS)]

        def post_forward_hook(_module, _inputs, output, *, block_index):
            binding = runner._ACTIVE_ACTION_CHECKPOINT_BINDING.get()
            self.assertIsNotNone(binding)
            runner.active_action_route().record_block_call(
                phase=binding.phase,
                checkpoint_index=binding.checkpoint_index,
                block_index=block_index,
            )
            return output

        for block_index, block in enumerate(blocks):
            block.register_forward_hook(
                functools.partial(post_forward_hook, block_index=block_index)
            )

        with runner.activate_action_route(route):
            result = value
            for block in blocks:
                result = checkpoint(
                    block,
                    result,
                    use_reentrant=False,
                    context_fn=runner.action_route_checkpoint_context_fn,
                )
            result.square().sum().backward()
            self.assertEqual(len(route.forward_block_calls), 30)
            self.assertEqual(len(route.checkpoint_recompute_context_indices), 30)
            self.assertEqual(len(route.recompute_block_calls), 0)
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "exactly one forward"
            ):
                route.validate_forward_and_recompute_traversal()
        self.assertIsNone(runner._ACTIVE_ACTION_ROUTE.get())
        self.assertTrue(runner._ACTION_ROUTE_LEASE_OWNER.is_empty())


class CLIAndStaticClosureTests(unittest.TestCase):
    def _args(self, output: Path) -> argparse.Namespace:
        return argparse.Namespace(
            ack_pre_d0_engineering_only=True,
            ack_legacy_target_quality_unqualified=True,
            ack_no_d0_or_scientific_claim=True,
            ack_fresh_base_disposable=True,
            max_steps=2,
            learning_rate=runner.DEFAULT_LR,
            max_grad_norm=runner.DEFAULT_MAX_GRAD_NORM,
            seed=runner.DEFAULT_SEED,
            expected_bernini_commit=runner.BERNINI_COMMIT,
            expected_veomni_commit=runner.VEOMNI_COMMIT,
            expected_checkpoint_tree_sha256=runner.CHECKPOINT_TREE_SHA256,
            expected_checkpoint_content_manifest_sha256=(
                runner.CHECKPOINT_CONTENT_MANIFEST_SHA256
            ),
            workspace_base_revision="a" * 40,
            expected_runner_source_sha256="c" * 64,
            release_manifest="/tmp/frozen-pre-d0-release-manifest.json",
            expected_release_manifest_sha256="b" * 64,
            output=str(output),
        )

    def test_cli_requires_fresh_pre_d0_namespace_and_all_acknowledgements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run_pre_d0_engineering_2step"
            runner.validate_args(self._args(output))
            args = self._args(output)
            args.ack_no_d0_or_scientific_claim = False
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "acknowledgements"
            ):
                runner.validate_args(args)
            args = self._args(Path(directory) / "looks_formal")
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "PRE_D0_ENGINEERING"
            ):
                runner.validate_args(args)

    def test_static_world8_fresh_optimizer_and_full_state_contract(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        required = (
            'AUTHORITY = "PRE_D0_ENGINEERING_ONLY"',
            "WORLD_SIZE = 8",
            "SP_SIZE = 4",
            "DP_SIZE = 2",
            "MAX_STEPS = 2",
            "serialized_model_load()",
            "torch.optim.AdamW(",
            "rank0_full_trainable_state_roundtrip_reload_verified",
            "rank0_optimizer_roundtrip_reload_verified",
            "all8_rng_sampler_scheduler_state_bytes_persisted_verified",
            "conditioner.prepare_route(source, instruction_tokens, ownership)",
            "conditioner.injection.residual(",
            "import torch.utils.checkpoint as torch_checkpoint",
            "torch_checkpoint.set_checkpoint_early_stop(False)",
            '"context_fn": action_route_checkpoint_context_fn',
            "_replay_action_checkpoint_route(",
            "validate_action_route_checkpointing_installation(transformer)",
            "torch.where(target_selector, target_adapted, native)",
            'betas=(0.9, 0.95)',
            '"policy": "constant_lr_no_scheduler_object"',
            "validate_release_manifest(",
            "validate_imported_release_modules(",
            '"terminal_world8_consensus_precedes_receipt_publication": True',
            "materialize_training_text_embedding(text_embs)",
            'tokenized["t5_input_lens"]',
            'torch.count_nonzero(text_embs[:, actual_length:, :])',
            "source_and_padding_bit_exact_under_injection",
            '"formal_0817_manifest_consumed": False',
            '"scientific_claim_authorized": False',
            '"selection_scope": "global_maximum_within_mechanically_strict_rows"',
            '"selected_schedule_records": validation[',
            '"selected_unique_rows": validation["selected_unique_rows"]',
            '"selected_rows_repeated": validation["selected_rows_repeated"]',
            '"effective_scientific_sample_size_claimed": False',
            '"video_vae_latents"',
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, source)
        self.assertNotIn('add_argument("--resume"', source)
        self.assertNotIn("method_source_revision", source)
        self.assertIn('text_embs, tokenized["t5_input_lens"]', source)
        self.assertNotIn("canonical_instruction_tokens(text_embs, text_lens)", source)
        self.assertNotIn('"formal_training_started": True', source)
        self.assertNotIn("q_y", source)

    def test_live_checkpoint_partial_is_exact_and_identity_pinned(self) -> None:
        transformer = types.SimpleNamespace(
            gradient_checkpointing=True,
            _gradient_checkpointing_func=functools.partial(
                lambda: None,
                use_reentrant=False,
                context_fn=runner.action_route_checkpoint_context_fn,
            ),
        )
        receipt = runner.validate_action_route_checkpointing_installation(
            transformer
        )
        self.assertTrue(receipt["live_partial_identity_verified"])
        self.assertEqual(receipt["checkpoint_contexts_per_microbatch"], 30)

        def lookalike_context_fn():
            return runner.action_route_checkpoint_context_fn()

        transformer._gradient_checkpointing_func = functools.partial(
            lambda: None,
            use_reentrant=False,
            context_fn=lookalike_context_fn,
        )
        with self.assertRaisesRegex(
            runner.PreD0EngineeringError, "installation differs"
        ):
            runner.validate_action_route_checkpointing_installation(transformer)

    def test_exact_release_manifest_members_and_physical_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "methods" / "bernini_action_editing"
            root.mkdir(parents=True)
            rows = []
            for relative in sorted(runner.RELEASE_FILES_AND_MODES):
                path = root / relative
                payload = (relative + "\n").encode("ascii")
                path.write_bytes(payload)
                path.chmod(runner.RELEASE_FILES_AND_MODES[relative])
                rows.append(
                    {
                        "path": relative,
                        "mode": runner.RELEASE_FILES_AND_MODES[relative],
                        "size": len(payload),
                        "sha256": runner.hashlib.sha256(payload).hexdigest(),
                    }
                )
            manifest_value = {
                "schema_version": runner.RELEASE_MANIFEST_SCHEMA,
                "member_root": runner.RELEASE_MEMBER_ROOT,
                "files": rows,
            }
            manifest = base / "release-manifest.json"
            manifest.write_bytes(runner.canonical_json_bytes(manifest_value) + b"\n")
            receipt = runner.validate_release_manifest(
                manifest,
                expected_sha256=runner.file_sha256(manifest),
                method_root=root,
            )
            self.assertEqual(receipt["member_count"], len(rows))
            modules = {
                relative: argparse.Namespace(__file__=str(root / relative))
                for relative in runner.RELEASE_FILES_AND_MODES
            }
            imported = runner.validate_imported_release_modules(
                receipt, modules, method_root=root
            )
            self.assertEqual(imported["exact_imported_member_count"], len(rows))
            shadowed = dict(modules)
            shadowed[rows[0]["path"]] = argparse.Namespace(__file__=__file__)
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "identity differs"
            ):
                runner.validate_imported_release_modules(
                    receipt, shadowed, method_root=root
                )
            manifest_value["files"] = rows + [dict(rows[0])]
            manifest.write_bytes(runner.canonical_json_bytes(manifest_value) + b"\n")
            with self.assertRaisesRegex(
                runner.PreD0EngineeringError, "missing/extra"
            ):
                runner.validate_release_manifest(
                    manifest,
                    expected_sha256=runner.file_sha256(manifest),
                    method_root=root,
                )

    def test_final_predictor_source_pin_is_exact(self) -> None:
        self.assertEqual(
            runner.ACTION_PLAN_PREDICTOR_SOURCE_SHA256,
            "464cd500f0ba1edb6cbe6d4f07287bfff346ae0ba7968c0d7c7f3cc7cb667308",
        )
        self.assertEqual(
            runner.ACTION_PLAN_CONDITIONER_STATE_ABI_SHA256,
            "04c2fc8ff48fb8b027e912cd6c9c58cf19d4b554c84127fb6623268a9d1e398b",
        )
        self.assertEqual(
            runner.file_sha256(PREDICTOR_PATH),
            runner.ACTION_PLAN_PREDICTOR_SOURCE_SHA256,
        )


class TorchStateContractTests(unittest.TestCase):
    def test_fp32_adamw_two_moments_and_scalar_bit_comparison(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in this environment")
        first = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        second = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float32))
        named = (("first", first), ("second", second))
        optimizer = torch.optim.AdamW(
            (first, second),
            lr=runner.DEFAULT_LR,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        runner.validate_adamw_state_abi(optimizer.state_dict(), named, step=0)
        (first.square().sum() + second.square().sum()).backward()
        optimizer.step()
        state = runner._cpu_tree(optimizer.state_dict())
        runner.validate_adamw_state_abi(state, named, step=1)
        self.assertTrue(runner._state_tree_bits_equal(state, state))
        negative_zero = torch.tensor(-0.0, dtype=torch.float32)
        positive_zero = torch.tensor(0.0, dtype=torch.float32)
        self.assertFalse(runner._tensor_bits_equal(negative_zero, positive_zero))

    def test_inference_t5_tensor_is_cloned_before_trainable_backward(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable in this environment")
        with torch.inference_mode():
            hostile = torch.randn(1, 3, runner.INSTRUCTION_WIDTH)
        self.assertTrue(torch.is_inference(hostile))
        hostile_linear = torch.nn.Linear(runner.INSTRUCTION_WIDTH, 2)
        with self.assertRaises(RuntimeError):
            hostile_linear(hostile)
        safe = runner.materialize_training_text_embedding(hostile)
        actual = runner.canonical_instruction_tokens(
            safe, torch.tensor([[3]], dtype=torch.int64)
        )
        self.assertFalse(torch.is_inference(safe))
        self.assertFalse(torch.is_inference(actual))
        linear = torch.nn.Linear(runner.INSTRUCTION_WIDTH, 2)
        linear(actual).sum().backward()
        self.assertIsNotNone(linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
