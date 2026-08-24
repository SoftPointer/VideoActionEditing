from __future__ import annotations

from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import internal_temporal_quotient_observer as observer  # noqa: E402


try:
    import torch
except ImportError:  # pragma: no cover - lightweight local environment
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class ContextAndLayoutTests(unittest.TestCase):
    def _context(self, **changes):
        values = {
            "mode": "t2v",
            "branch": "action",
            "sigma": 0.75,
            "lambda_value": 1.0,
            "sp_rank": 0,
        }
        values.update(changes)
        return observer.FITQObserverContext(**values)

    def test_context_is_explicit_and_geometry_is_pinned(self) -> None:
        context = self._context()
        self.assertEqual(context.total_sequence_length, 19530)
        self.assertEqual(context.local_sequence_length, 4883)
        self.assertEqual(context.as_dict()["lambda"], 1.0)
        self.assertEqual(context.as_dict()["phase_geometry"], [21, 31, 30])
        self.assertEqual(context.branch_key, ("t2v", "action", 0.75, 1.0))

        invalid = (
            {"mode": "i2v"},
            {"branch": ""},
            {"branch": " action"},
            {"sigma": float("nan")},
            {"sigma": -0.1},
            {"lambda_value": -1.0},
            {"sp_rank": 4},
            {"sp_world": 8},
            {"global_target_tokens": 19529},
            {"phase_count": 20},
            {"patch_height": 30},
            {"patch_width": 31},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(
                observer.InternalTemporalQuotientObserverError
            ):
                self._context(**change)

    def test_t2v_rank3_excludes_exact_padding_and_recovers_phase_coverage(self) -> None:
        layout = observer.build_local_target_layout(self._context(sp_rank=3))
        self.assertEqual(layout.shard_start, 14649)
        self.assertEqual(layout.shard_stop, 19532)
        self.assertEqual(layout.valid_sequence_tokens, 4881)
        self.assertEqual(layout.source_tokens_excluded, 0)
        self.assertEqual(layout.target_tokens_selected, 4881)
        self.assertEqual(layout.padding_tokens_excluded, 2)
        counts = layout.expected_phase_token_count.tolist()
        self.assertEqual(counts[:15], [0] * 15)
        self.assertEqual(counts[15], 231)
        self.assertEqual(counts[16:], [930] * 5)
        self.assertEqual(
            int(layout.target_local_indices[-1].item()), 4880
        )  # local 4881/4882 are padding

    def test_mv2v_selects_only_target_tail_and_never_source(self) -> None:
        rank0 = observer.build_local_target_layout(
            self._context(mode="mv2v", sp_rank=0)
        )
        rank1 = observer.build_local_target_layout(
            self._context(mode="mv2v", sp_rank=1)
        )
        rank2 = observer.build_local_target_layout(
            self._context(mode="mv2v", sp_rank=2)
        )
        rank3 = observer.build_local_target_layout(
            self._context(mode="mv2v", sp_rank=3)
        )
        for layout in (rank0, rank1):
            self.assertEqual(layout.source_tokens_excluded, 9765)
            self.assertEqual(layout.target_tokens_selected, 0)
            self.assertEqual(int(layout.expected_phase_token_count.sum()), 0)
        for layout in (rank2, rank3):
            self.assertEqual(layout.source_tokens_excluded, 0)
            self.assertEqual(layout.target_tokens_selected, 9765)
            self.assertEqual(layout.padding_tokens_excluded, 0)
        self.assertEqual(rank2.expected_phase_token_count[:10].tolist(), [930] * 10)
        self.assertEqual(int(rank2.expected_phase_token_count[10]), 465)
        self.assertEqual(int(rank3.expected_phase_token_count[10]), 465)
        self.assertEqual(rank3.expected_phase_token_count[11:].tolist(), [930] * 10)
        total = sum(
            (
                layout.expected_phase_token_count
                for layout in (rank0, rank1, rank2, rank3)
            ),
            torch.zeros(21, dtype=torch.int64),
        )
        self.assertTrue(torch.equal(total, torch.full((21,), 930, dtype=torch.int64)))


if torch is not None:

    class _MockAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_out = torch.nn.ModuleList(
                [torch.nn.Identity(), torch.nn.Identity()]
            )


    class _MockBlock(torch.nn.Module):
        def __init__(self, *, duplicate_attn1: bool = False):
            super().__init__()
            self.attn1 = _MockAttention()
            self.attn2 = _MockAttention()
            self.duplicate_attn1 = duplicate_attn1

        def forward(self, hidden_states):
            hidden_states = self.attn1.to_out[0](hidden_states)
            if self.duplicate_attn1:
                hidden_states = self.attn1.to_out[0](hidden_states)
            hidden_states = self.attn2.to_out[0](hidden_states)
            return hidden_states


    class _MockTransformer(torch.nn.Module):
        def __init__(
            self,
            *,
            block_count: int = 30,
            config_heads: int = 12,
            duplicate_attn1_at: int | None = None,
        ):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [
                    _MockBlock(duplicate_attn1=index == duplicate_attn1_at)
                    for index in range(block_count)
                ]
            )
            self.proj_out = torch.nn.Identity()
            self.config = SimpleNamespace(
                num_layers=block_count,
                num_attention_heads=config_heads,
                attention_head_dim=128,
            )

        def patch_vae_latent(self, value=None):
            return value

        def forward(self, hidden_states):
            for block in self.blocks:
                hidden_states = block(hidden_states)
            return self.proj_out(hidden_states)


    class _Wrapper:
        def __init__(self, transformer):
            self.diff_dec = SimpleNamespace(transformer=transformer)


@unittest.skipIf(torch is None, "torch is unavailable")
class ResolverTests(unittest.TestCase):
    def test_resolves_exact_pinned_transformer_through_runtime_wrappers(self) -> None:
        transformer = _MockTransformer()
        wrapper = _Wrapper(transformer)
        self.assertIs(observer.resolve_pinned_wan_transformer(wrapper), transformer)
        self.assertIs(observer.resolve_pinned_30_block_transformer(wrapper), transformer)

    def test_rejects_wrong_depth_head_config_and_ambiguous_models(self) -> None:
        for transformer in (
            _MockTransformer(block_count=29),
            _MockTransformer(config_heads=16),
        ):
            with self.subTest(transformer=transformer), self.assertRaises(
                observer.InternalTemporalQuotientObserverError
            ):
                observer.resolve_pinned_wan_transformer(_Wrapper(transformer))

        class Ambiguous:
            def __init__(self):
                self.diff_dec = _MockTransformer()
                self.transformer = _MockTransformer()

        with self.assertRaises(observer.InternalTemporalQuotientObserverError):
            observer.resolve_pinned_wan_transformer(Ambiguous())


@unittest.skipIf(torch is None, "torch is unavailable")
class HookObserverTests(unittest.TestCase):
    @staticmethod
    def _context(*, branch="action", sp_rank=3, mode="t2v"):
        return observer.FITQObserverContext(
            mode=mode,
            branch=branch,
            sigma=0.5,
            lambda_value=1.25,
            sp_rank=sp_rank,
        )

    def test_all_official_sites_are_read_only_fp32_and_exactly_counted(self) -> None:
        transformer = _MockTransformer()
        base = torch.ones(1, 1, 1536, dtype=torch.float32, requires_grad=True)
        value = base.expand(1, 4883, 1536)
        baseline = transformer(value)
        fitted = observer.InternalTemporalQuotientObserver(
            transformer, capture_exact_block0=True
        )

        with fitted:
            with fitted.capture(self._context()) as session:
                observed_output = transformer(value)
            result = session.result
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(fitted.installed)
            self.assertFalse(fitted.active)
            self.assertEqual(fitted.trainable_parameters, ())
            self.assertFalse(isinstance(fitted, torch.nn.Module))
            self.assertIs(observed_output, value)
            self.assertTrue(torch.equal(observed_output, baseline))
            self.assertTrue(observed_output.requires_grad)
            self.assertEqual(result.call_sequence, observer.expected_site_order())
            self.assertEqual(len(result.sites), 121)
            self.assertEqual(set(result.call_counts.values()), {1})
            self.assertEqual(set(result.exact_fingerprints), set(observer.EXACT_PROBE_SITES))
            parity = observer.exact_block0_parity(result, result)
            self.assertEqual(parity["all"], True)

            first = result.sites["block.00.input"]
            self.assertEqual(first.sum.dtype, torch.float32)
            self.assertEqual(first.sumsq.dtype, torch.float32)
            self.assertEqual(first.count.dtype, torch.float32)
            self.assertFalse(first.sum.requires_grad)
            self.assertEqual(tuple(first.sum.shape), (21, 12, 128))
            expected = result.layout.expected_phase_token_count.float()
            self.assertTrue(torch.equal(first.count[:, 0].cpu(), expected))
            self.assertTrue(torch.equal(first.sum[:, 0, 0].cpu(), expected))
            self.assertTrue(torch.equal(first.sumsq[:, 0, 0].cpu(), expected))

            with self.assertRaisesRegex(
                observer.InternalTemporalQuotientObserverError, "duplicate"
            ):
                fitted.begin_forward(self._context())

        self.assertFalse(fitted.installed)
        # Hook-off is real: an unpinned mock shape now passes untouched.
        tiny = torch.zeros(1, 2, 1536)
        self.assertIs(transformer(tiny), tiny)

    def test_exact_fingerprint_is_value_based_across_independent_storage(self) -> None:
        transformer = _MockTransformer()
        fitted = observer.InternalTemporalQuotientObserver(
            transformer, capture_exact_block0=True
        )
        first = torch.zeros(1, 4883, 1536, dtype=torch.bfloat16)
        equal_clone = first.clone()
        one_bit_changed = first.clone()
        changed_bytes = one_bit_changed.view(torch.uint8).reshape(-1)
        changed_bytes[0] = changed_bytes[0] ^ 1

        with fitted:
            with fitted.capture(self._context(branch="fingerprint-a")) as session_a:
                transformer(first)
            with fitted.capture(self._context(branch="fingerprint-b")) as session_b:
                transformer(equal_clone)
            with fitted.capture(self._context(branch="fingerprint-c")) as session_c:
                transformer(one_bit_changed)

        assert session_a.result is not None
        assert session_b.result is not None
        assert session_c.result is not None
        self.assertIsNot(first, equal_clone)
        self.assertTrue(torch.equal(first, equal_clone))
        self.assertTrue(
            observer.exact_block0_parity(session_a.result, session_b.result)["all"]
        )
        changed_parity = observer.exact_block0_parity(
            session_a.result, session_c.result
        )
        self.assertFalse(changed_parity["block.00.input"])
        self.assertFalse(changed_parity["block.00.attn1"])
        self.assertFalse(changed_parity["all"])

    def test_official_inference_mode_tensor_has_no_version_counter_dependency(self) -> None:
        transformer = _MockTransformer()
        fitted = observer.InternalTemporalQuotientObserver(transformer).install()
        context = self._context(branch="inference-mode", sp_rank=0, mode="mv2v")
        fitted.begin_forward(context)
        with torch.inference_mode():
            # Created inside inference_mode, so reading ``tensor._version``
            # raises in PyTorch.  The observer must still remain read-only.
            value = torch.ones(1, 1, 1536).expand(1, 9765, 1536)
            output = transformer(value)
        result = fitted.finish_forward()
        fitted.remove()
        self.assertIs(output, value)
        self.assertEqual(result.layout.target_tokens_selected, 0)
        self.assertEqual(len(result.sites), 121)

    def test_inference_mode_inplace_change_recomputes_fingerprint_and_statistics(
        self,
    ) -> None:
        fitted = observer.InternalTemporalQuotientObserver(
            _MockTransformer(), capture_exact_block0=True
        )
        context = self._context(branch="inference-mode-mutation", mode="t2v")
        layout = observer.build_local_target_layout(context)

        with torch.inference_mode():
            value = torch.zeros(1, 4883, 1536, dtype=torch.float32)
            first_fingerprint = fitted._fingerprint_for_tensor(value)
            first_statistics = fitted._statistics_for_tensor(value, layout)
            self.assertEqual(int(torch.count_nonzero(value).item()), 0)

            value.add_(1.0)
            second_fingerprint = fitted._fingerprint_for_tensor(value)
            second_statistics = fitted._statistics_for_tensor(value, layout)
            self.assertTrue(bool((value == 1.0).all().item()))

        self.assertNotEqual(first_fingerprint, second_fingerprint)
        self.assertEqual(float(first_statistics.sum.abs().sum().item()), 0.0)
        self.assertGreater(float(second_statistics.sum.abs().sum().item()), 0.0)
        self.assertIsNot(first_statistics, second_statistics)
        self.assertNotIn(id(value), fitted._stat_cache)
        self.assertNotIn(id(value), fitted._fingerprint_cache)

    def test_installed_hook_without_context_fails_closed(self) -> None:
        transformer = _MockTransformer()
        fitted = observer.InternalTemporalQuotientObserver(transformer).install()
        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "without.*context"
        ):
            transformer(torch.zeros(1, 2, 1536))
        self.assertTrue(fitted.poisoned)
        fitted.remove()

    def test_shape_call_order_and_incomplete_coverage_fail_closed(self) -> None:
        transformer = _MockTransformer()
        fitted = observer.InternalTemporalQuotientObserver(transformer).install()
        fitted.begin_forward(self._context(branch="bad-shape"))
        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "shape must be"
        ):
            transformer(torch.zeros(1, 2, 1536))
        fitted.abort_forward()
        self.assertTrue(fitted.poisoned)
        fitted.remove()

        incomplete = observer.InternalTemporalQuotientObserver(
            _MockTransformer()
        ).install()
        incomplete.begin_forward(self._context(branch="incomplete"))
        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "missed.*reordered"
        ):
            incomplete.finish_forward()
        self.assertTrue(incomplete.poisoned)
        incomplete.remove()

        duplicate_model = _MockTransformer(duplicate_attn1_at=0)
        duplicate = observer.InternalTemporalQuotientObserver(
            duplicate_model
        ).install()
        duplicate.begin_forward(self._context(branch="duplicate-call"))
        value = torch.zeros(1, 1, 1536).expand(1, 4883, 1536)
        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "hook order/call"
        ):
            duplicate_model(value)
        duplicate.abort_forward()
        duplicate.remove()

    def test_output_tuple_is_rejected_instead_of_silently_unwrapped(self) -> None:
        class TupleBlock(_MockBlock):
            def forward(self, hidden_states):
                hidden_states = self.attn1.to_out[0](hidden_states)
                hidden_states = self.attn2.to_out[0](hidden_states)
                return (hidden_states,)

        transformer = _MockTransformer()
        transformer.blocks[0] = TupleBlock()
        fitted = observer.InternalTemporalQuotientObserver(transformer).install()
        fitted.begin_forward(self._context(branch="tuple-output"))
        value = torch.zeros(1, 1, 1536).expand(1, 4883, 1536)
        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "did not expose a tensor"
        ):
            transformer(value)
        fitted.abort_forward()
        fitted.remove()

    def test_mv2v_source_only_rank_accumulates_exact_zeros(self) -> None:
        transformer = _MockTransformer()
        fitted = observer.InternalTemporalQuotientObserver(transformer).install()
        context = self._context(branch="mv2v-source-rank", sp_rank=0, mode="mv2v")
        # Expanded storage keeps this exact official shape cheap in the mock.
        value = torch.ones(1, 1, 1536).expand(1, 9765, 1536)
        fitted.begin_forward(context)
        transformer(value)
        result = fitted.finish_forward()
        fitted.remove()
        self.assertEqual(result.layout.source_tokens_excluded, 9765)
        self.assertEqual(result.layout.target_tokens_selected, 0)
        for stats in result.sites.values():
            self.assertEqual(int(stats.count.sum().item()), 0)
            self.assertEqual(float(stats.sum.abs().sum().item()), 0.0)
            self.assertEqual(float(stats.sumsq.abs().sum().item()), 0.0)


@unittest.skipIf(torch is None, "torch is unavailable")
class OutsideHookReductionTests(unittest.TestCase):
    @staticmethod
    def _synthetic_local():
        context = observer.FITQObserverContext(
            mode="t2v",
            branch="reduce",
            sigma=0.25,
            lambda_value=0.5,
            sp_rank=0,
        )
        layout = observer.build_local_target_layout(context)
        shape = (21, 12, 128)
        counts = layout.expected_phase_token_count.float()[:, None].expand(21, 12)
        shared = observer.PhaseHeadSufficientStatistics(
            sum=torch.zeros(shape, dtype=torch.float32),
            sumsq=torch.zeros(shape, dtype=torch.float32),
            count=counts.clone(),
        )
        order = observer.expected_site_order()
        return observer.LocalFITQSufficientStatistics(
            context=context,
            layout=layout,
            sites=MappingProxyType({site: shared for site in order}),
            call_counts=MappingProxyType({site: 1 for site in order}),
            call_sequence=order,
            exact_fingerprints=MappingProxyType({}),
            globally_reduced=False,
        )

    def test_collective_is_explicitly_outside_hook_and_global_coverage_is_exact(self) -> None:
        class FakeDist:
            ReduceOp = SimpleNamespace(SUM="sum")

            def __init__(self):
                self.calls = []

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size(group=None):
                return 4

            def all_reduce(self, value, *, op, group):
                self.calls.append((tuple(value.shape), op, group))
                if len(self.calls) == 3:  # packed count collective
                    value.fill_(930.0)

        reducer = FakeDist()
        reduced = observer.all_reduce_local_sufficient_statistics(
            self._synthetic_local(), dist_module=reducer, group="ulysses"
        )
        self.assertTrue(reduced.globally_reduced)
        self.assertEqual(len(reducer.calls), 3)
        self.assertEqual(reducer.calls[0][0], (121, 21, 12, 128))
        self.assertEqual(reducer.calls[2][0], (121, 21, 12))
        self.assertTrue(
            torch.equal(
                reduced.sites["block.00.attn1"].count,
                torch.full((21, 12), 930.0),
            )
        )
        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "already globally"
        ):
            observer.all_reduce_local_sufficient_statistics(
                reduced, dist_module=reducer, group="ulysses"
            )

    def test_bad_world_or_incomplete_global_phase_coverage_fails_closed(self) -> None:
        class BadWorld:
            ReduceOp = SimpleNamespace(SUM="sum")

            @staticmethod
            def get_world_size(group=None):
                return 8

            @staticmethod
            def all_reduce(value, *, op, group):
                raise AssertionError("must fail before collective")

        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "differs.*SP world"
        ):
            observer.all_reduce_local_sufficient_statistics(
                self._synthetic_local(), dist_module=BadWorld()
            )

        class Incomplete:
            ReduceOp = SimpleNamespace(SUM="sum")

            @staticmethod
            def get_world_size(group=None):
                return 4

            @staticmethod
            def all_reduce(value, *, op, group):
                return None

        with self.assertRaisesRegex(
            observer.InternalTemporalQuotientObserverError, "global phase coverage"
        ):
            observer.all_reduce_local_sufficient_statistics(
                self._synthetic_local(), dist_module=Incomplete()
            )


if __name__ == "__main__":
    unittest.main()
