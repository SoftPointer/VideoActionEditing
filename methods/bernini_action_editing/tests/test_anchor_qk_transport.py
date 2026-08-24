from __future__ import annotations

import pathlib
import sys
import unittest
from dataclasses import fields
import inspect

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anchor_qk_transport as transport  # noqa: E402


class _Base:
    def __init__(self) -> None:
        self.delegations = 0

    def _project_qkv(self, _attn, hidden, *_args):
        value = hidden.reshape(1, hidden.shape[1], 1, hidden.shape[2])
        return value + 1, value + 2, value + 3

    def __call__(self, _attn, hidden, **_kwargs):
        self.delegations += 1
        return hidden + 9


class _Attn:
    def __init__(self) -> None:
        self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])


class _State:
    ulysses_enabled = False
    ulysses_rank = 0
    ulysses_size = 1


class AnchorQKTransportTest(unittest.TestCase):
    def test_qk_only_cache_abi_cannot_store_content_bearing_donor_fields(self):
        self.assertEqual(
            [item.name for item in fields(transport.AnchorQKOnlyEntry)],
            ["query", "key"],
        )
        self.assertEqual(
            list(inspect.signature(transport.AnchorQKCacheBank.capture_qk_only).parameters),
            ["self", "invocation", "block_index", "query", "key"],
        )
        bank = transport.AnchorQKCacheBank((0,))
        invocation = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2,
            replay_uses=2,
        )
        query = torch.zeros(1, transport.LATENT_PHASES * 2, 1, 3)
        with self.assertRaises(transport.AnchorQKTransportError):
            bank.capture(
                invocation=invocation,
                block_index=0,
                hidden_state=torch.zeros(1, transport.LATENT_PHASES * 2, 3),
                query=query,
                key=query,
                value=query,
                attention_output=query,
            )
        bank.capture_qk_only(
            invocation=invocation,
            block_index=0,
            query=query,
            key=query + 1,
        )
        entry = next(iter(bank._entries.values()))[0]
        self.assertIsInstance(entry, transport.AnchorQKOnlyEntry)
        for forbidden in (
            "value",
            "hidden_state",
            "attention_output",
            "rgb",
            "latent",
            "absolute_spatial_coordinate",
        ):
            self.assertFalse(hasattr(entry, forbidden))
        receipt = bank.receipt()
        self.assertEqual(receipt["qk_only_cached_fields"], ["query", "key"])
        self.assertEqual(receipt["qk_only_capture_count"], 1)

        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2,
            replay_scope=transport.FULL_SEQUENCE,
            replay_uses=2,
        )
        with self.assertRaises(transport.AnchorQKTransportError):
            bank.consume(
                invocation=replay,
                block_index=0,
                current_query=query,
                current_key=query,
                current_value=query,
                current_hidden_state=torch.zeros(
                    1, transport.LATENT_PHASES * 2, 3
                ),
            )
        for _ in range(2):
            consumed = bank.consume_qk_only(
                invocation=replay,
                block_index=0,
                current_query=query,
                current_key=query,
            )
            self.assertEqual([item.name for item in fields(consumed)], ["query", "key"])
        self.assertEqual(bank.receipt()["qk_only_replay_count"], 2)
        bank.assert_empty()

    def test_qk_only_kernel_is_noop_identity_and_donor_spatial_permutation_invariant(self):
        phases = transport.LATENT_PHASES
        spatial = 4
        heads = 1
        width = phases
        shape = (1, phases, spatial, heads, width)
        current_output = torch.randn(shape)
        current_value = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone()
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        identity = transport._qk_only_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_value),
            flat(noop_query),
            flat(noop_key),
            flat(noop_query),
            flat(noop_key),
            strength=1.0,
        ).reshape(shape)
        self.assertTrue(torch.equal(identity, current_output))

        action_query = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone() * 8.0
        action_key = action_query.clone()
        routed = transport._qk_only_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_value),
            flat(action_query),
            flat(action_key),
            flat(noop_query),
            flat(noop_key),
            strength=1.0,
        ).reshape(shape)
        self.assertTrue(torch.equal(routed[:, 0], current_output[:, 0]))
        self.assertGreater(float((routed[:, 1:] - current_output[:, 1:]).abs().sum()), 0.0)

        permutation = torch.tensor([2, 0, 3, 1])
        permuted = transport._qk_only_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_value),
            flat(action_query[:, :, permutation]),
            flat(action_key[:, :, permutation]),
            flat(noop_query[:, :, permutation]),
            flat(noop_key[:, :, permutation]),
            strength=1.0,
        ).reshape(shape)
        self.assertTrue(torch.allclose(permuted, routed, atol=1.0e-6))

        doubled_target_value = transport._qk_only_temporal_kernel_contrast_output(
            flat(current_output),
            flat(2.0 * current_value),
            flat(action_query),
            flat(action_key),
            flat(noop_query),
            flat(noop_key),
            strength=1.0,
        ).reshape(shape)
        self.assertTrue(
            torch.allclose(
                doubled_target_value - current_output,
                2.0 * (routed - current_output),
                atol=1.0e-5,
            )
        )

    def test_qk_only_kernel_and_support_reject_caption_dc_and_phase0_only_difference(self):
        phases = transport.LATENT_PHASES
        spatial = 3
        heads = 2
        width = 5
        shape = (1, phases, spatial, heads, width)
        generator = torch.Generator().manual_seed(20260820)
        action_query = torch.randn(shape, generator=generator)
        action_key = torch.randn(shape, generator=generator)
        noop_query = torch.randn(shape, generator=generator)
        noop_key = torch.randn(shape, generator=generator)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        kernel, support = transport._qk_only_anchor_temporal_kernel_components(
            flat(action_query), flat(action_key), flat(noop_query), flat(noop_key)
        )
        # A donor-caption/static offset can differ between action and noop, but
        # if it is time constant it must affect neither support nor the route.
        action_q_offset = torch.randn(
            (1, 1, spatial, heads, width), generator=generator
        )
        action_k_offset = torch.randn(
            (1, 1, spatial, heads, width), generator=generator
        )
        noop_q_offset = torch.randn(
            (1, 1, spatial, heads, width), generator=generator
        )
        noop_k_offset = torch.randn(
            (1, 1, spatial, heads, width), generator=generator
        )
        shifted_kernel, shifted_support = (
            transport._qk_only_anchor_temporal_kernel_components(
                flat(action_query + action_q_offset),
                flat(action_key + action_k_offset),
                flat(noop_query + noop_q_offset),
                flat(noop_key + noop_k_offset),
            )
        )
        self.assertTrue(torch.allclose(shifted_support, support, atol=1.0e-6))
        self.assertTrue(torch.allclose(shifted_kernel, kernel, atol=1.0e-6))

        phase0_action_query = torch.zeros(shape)
        phase0_action_key = torch.zeros(shape)
        phase0_action_query[:, 0] = torch.randn(
            (1, spatial, heads, width), generator=generator
        )
        phase0_action_key[:, 0] = torch.randn(
            (1, spatial, heads, width), generator=generator
        )
        zero = torch.zeros(shape)
        phase0_kernel, phase0_support = (
            transport._qk_only_anchor_temporal_kernel_components(
                flat(phase0_action_query),
                flat(phase0_action_key),
                flat(zero),
                flat(zero),
            )
        )
        self.assertTrue(torch.equal(phase0_support, torch.zeros_like(phase0_support)))
        self.assertTrue(torch.equal(phase0_kernel, torch.zeros_like(phase0_kernel)))

        current_output = torch.randn(shape, generator=generator)
        current_value = torch.randn(shape, generator=generator)
        phase0_routed = transport._qk_only_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_value),
            flat(phase0_action_query),
            flat(phase0_action_key),
            flat(zero),
            flat(zero),
            strength=1.0,
        )
        self.assertTrue(torch.equal(phase0_routed, flat(current_output)))

    def test_qk_only_target_gate_is_placed_only_by_target_qk_activity(self):
        phases = transport.LATENT_PHASES
        spatial = 10
        heads = 1
        width = phases
        shape = (1, phases, spatial, heads, width)
        current_output = torch.zeros(shape)
        for site in range(spatial):
            current_output[:, :, site] = (
                torch.arange(phases).reshape(1, phases, 1, 1)
                * float(site + 1)
            )
        current_value = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone()
        current_query = torch.zeros(shape)
        current_key = torch.zeros(shape)
        current_query[:, :, 3] = torch.arange(phases).reshape(1, phases, 1, 1)
        current_key[:, :, 3] = current_query[:, :, 3]
        action_query = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone() * 8.0
        action_key = action_query.clone()
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        routed = transport._qk_only_target_gated_hard_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_query),
            flat(current_key),
            flat(current_value),
            flat(action_query),
            flat(action_key),
            flat(noop_query),
            flat(noop_key),
            strength=1.0,
            target_keep_fraction=0.10,
        ).reshape(shape)
        changed = (
            (routed - current_output).abs().sum(dim=(1, 3, 4)).squeeze(0).ne(0)
        )
        self.assertTrue(torch.equal(routed[:, 0], current_output[:, 0]))
        self.assertEqual(int(changed.sum()), 1)
        self.assertTrue(bool(changed[3]))
        self.assertNotIn(
            "qk_only_correspondence_kernel25_attn_output",
            transport.TRANSPORTS,
        )
        for legacy_alias in (
            "qk_only_temporal_kernel_contrast_attn_output",
            "qk_only_target_gated_hard_kernel_top10_attn_output",
            "qk_only_target_gated_hard_kernel_top25_attn_output",
        ):
            self.assertNotIn(legacy_alias, transport.TRANSPORTS)

    def test_exact_forward_rms_uses_finite_zero_subgradient_only_at_zero(self):
        zero = torch.zeros(2, 3, 4, requires_grad=True)
        zero_rms = transport._exact_forward_zero_subgradient_rms(
            zero,
            dim=(1, 2),
            keepdim=True,
        )
        self.assertTrue(torch.equal(zero_rms, torch.zeros_like(zero_rms)))
        zero_rms.sum().backward()
        self.assertIsNotNone(zero.grad)
        self.assertTrue(torch.isfinite(zero.grad).all())
        self.assertTrue(torch.equal(zero.grad, torch.zeros_like(zero.grad)))

        generator = torch.Generator().manual_seed(20260823)
        reference = torch.randn(2, 3, 4, generator=generator, requires_grad=True)
        candidate = reference.detach().clone().requires_grad_(True)
        ordinary = reference.square().mean(dim=(1, 2), keepdim=True).sqrt()
        safe = transport._exact_forward_zero_subgradient_rms(
            candidate,
            dim=(1, 2),
            keepdim=True,
        )
        self.assertTrue(torch.equal(safe, ordinary))
        ordinary.sum().backward()
        safe.sum().backward()
        self.assertTrue(torch.equal(candidate.grad, reference.grad))

        nonfinite = torch.tensor([[[float("nan")]]])
        propagated = transport._exact_forward_zero_subgradient_rms(
            nonfinite,
            dim=(1, 2),
            keepdim=True,
        )
        self.assertTrue(torch.isnan(propagated).all())

    def test_qk_only_target_gate_zero_donor_route_backward_is_finite(self):
        # At scheduler timestep 1000 the dynamic/static donor inputs coincide.
        # Their captured Q/K tensors and temporal-kernel contrast are therefore
        # exact, making route RMS exactly zero.  The hard identity fallback must
        # have a finite zero subgradient instead of sqrt(0)'s 0*inf NaN.
        phases = transport.LATENT_PHASES
        spatial = 3
        heads = 1
        width = 5
        shape = (1, phases, spatial, heads, width)
        generator = torch.Generator().manual_seed(2026082301)
        current_output = torch.randn(
            shape, generator=generator, requires_grad=True
        )
        current_value = torch.randn(
            shape, generator=generator, requires_grad=True
        )
        current_query = torch.randn(shape, generator=generator)
        current_key = torch.randn(shape, generator=generator)
        donor_query = torch.randn(shape, generator=generator)
        donor_key = torch.randn(shape, generator=generator)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        routed = transport._qk_only_target_gated_hard_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_query),
            flat(current_key),
            flat(current_value),
            flat(donor_query),
            flat(donor_key),
            flat(donor_query.clone()),
            flat(donor_key.clone()),
            strength=0.25,
            target_keep_fraction=0.25,
        ).reshape(shape)
        self.assertTrue(torch.equal(routed, current_output))
        routed.float().square().mean().backward()
        self.assertIsNotNone(current_output.grad)
        self.assertIsNotNone(current_value.grad)
        self.assertTrue(torch.isfinite(current_output.grad).all())
        self.assertTrue(torch.isfinite(current_value.grad).all())
        self.assertTrue(
            torch.equal(current_value.grad, torch.zeros_like(current_value.grad))
        )

    def test_qk_only_target_gate_zero_current_temporal_backward_is_finite(self):
        phases = transport.LATENT_PHASES
        spatial = 3
        heads = 1
        width = phases
        shape = (1, phases, spatial, heads, width)
        generator = torch.Generator().manual_seed(2026082302)
        phase_zero = torch.randn(
            (1, 1, spatial, heads, width), generator=generator
        )
        current_output = phase_zero.expand(shape).clone().requires_grad_(True)
        current_value = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone().requires_grad_(True)
        current_query = torch.randn(shape, generator=generator)
        current_key = torch.randn(shape, generator=generator)
        action_query = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone() * 8.0
        action_key = action_query.clone()
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        routed = transport._qk_only_target_gated_hard_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_query),
            flat(current_key),
            flat(current_value),
            flat(action_query),
            flat(action_key),
            flat(noop_query),
            flat(noop_key),
            strength=0.25,
            target_keep_fraction=0.25,
        ).reshape(shape)
        self.assertTrue(torch.equal(routed, current_output))
        routed.float().square().mean().backward()
        self.assertIsNotNone(current_output.grad)
        self.assertIsNotNone(current_value.grad)
        self.assertTrue(torch.isfinite(current_output.grad).all())
        self.assertTrue(torch.isfinite(current_value.grad).all())

    def test_event01_dynamic_terminal_uses_anchor_object_actor_relation(self):
        centers = transport._event01_dynamic_target_centers(1)
        actor_xy, object_xy = centers[-1]
        anchor_actor = transport.EVENT01_ANCHOR_ACTOR_TRAJECTORY_XY[-1]
        anchor_object = transport.EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY[-1]
        self.assertEqual(
            object_xy[0] - actor_xy[0],
            anchor_object[0] - anchor_actor[0],
        )
        self.assertEqual(
            object_xy[1] - actor_xy[1],
            anchor_object[1] - anchor_actor[1],
        )
        self.assertEqual(object_xy, (7.0, 26.0))

        side_aligned = transport._event01_dynamic_target_centers(
            1,
            source_side_aligned=True,
        )
        aligned_actor, aligned_object = side_aligned[-1]
        source_dx = (
            transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[1][0]
            - transport.EVENT01_SOURCE_ACTOR_XY[0]
        )
        self.assertGreater((aligned_object[0] - aligned_actor[0]) * source_dx, 0.0)
        self.assertEqual(
            abs(aligned_object[0] - aligned_actor[0]),
            abs(anchor_object[0] - anchor_actor[0]),
        )
        self.assertEqual(
            aligned_object[1] - aligned_actor[1],
            anchor_object[1] - anchor_actor[1],
        )
        self.assertEqual(aligned_object, (18.0, 26.0))

    TOKENS = 4

    def _processor(self, bank, calls, *, block_index=0):
        def varlen(q, k, v, **_kwargs):
            calls.append((q.clone(), k.clone(), v.clone()))
            return q

        return transport.AnchorQKSelfAttnProcessor(
            _Base(),
            block_index=block_index,
            cache_bank=bank,
            varlen_attention_fn=varlen,
            get_parallel_state_fn=lambda: _State(),
            gather_heads_scatter_seq_fn=lambda value, **_kwargs: value,
        )

    @staticmethod
    def _call(processor, hidden):
        length = int(hidden.shape[1])
        return processor(
            _Attn(),
            hidden,
            rotary_emb=torch.ones(1),
            batch_image_vae_seqlen=[length],
            origin_hidden_states_seq_len=length,
            cu_seqlens_q_cache=torch.tensor([0, length], dtype=torch.int32),
            max_seqlen_q_cache=length,
        )

    def test_no_context_is_exact_base_delegation(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        hidden = torch.zeros(1, 4, 2)
        output = self._call(processor, hidden)
        self.assertTrue(torch.equal(output, hidden + 9))
        self.assertEqual(calls, [])
        self.assertEqual(processor.base_delegations, 1)

    def test_hard_qk_replaces_only_target_qk_and_keeps_current_value(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        anchor = torch.full((1, self.TOKENS, 2), 10.0)
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE, bank, step_index=3, candidate_index=2, rank=0, ulysses_size=1
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, anchor)

        current = torch.full((1, 2 * self.TOKENS, 2), 100.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY, bank, step_index=3, candidate_index=2, rank=0, ulysses_size=1
        )
        with transport.anchor_qk_invocation(replay):
            output = self._call(processor, current)

        replay_q, replay_k, replay_v = calls[-1]
        self.assertTrue(torch.equal(replay_q[: self.TOKENS], torch.full_like(replay_q[: self.TOKENS], 101)))
        self.assertTrue(torch.equal(replay_q[self.TOKENS :], torch.full_like(replay_q[self.TOKENS :], 11)))
        self.assertTrue(torch.equal(replay_k[: self.TOKENS], torch.full_like(replay_k[: self.TOKENS], 102)))
        self.assertTrue(torch.equal(replay_k[self.TOKENS :], torch.full_like(replay_k[self.TOKENS :], 12)))
        self.assertTrue(torch.equal(replay_v, torch.full_like(replay_v, 103)))
        self.assertTrue(torch.equal(output[:, : self.TOKENS], torch.full_like(output[:, : self.TOKENS], 101)))
        self.assertTrue(torch.equal(output[:, self.TOKENS :], torch.full_like(output[:, self.TOKENS :], 11)))
        bank.assert_empty()

    def test_hard_k_keeps_current_query(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE, bank, step_index=0, candidate_index=0, rank=0, ulysses_size=1
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, torch.zeros(1, self.TOKENS, 2))
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.HARD_K,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, torch.full((1, 2 * self.TOKENS, 2), 5.0))
        query, key, value = calls[-1]
        self.assertTrue(torch.equal(query, torch.full_like(query, 6)))
        self.assertTrue(torch.equal(key[self.TOKENS :], torch.full_like(key[self.TOKENS :], 2)))
        self.assertTrue(torch.equal(value, torch.full_like(value, 8)))

    def test_dual_route_early_block_copies_anchor_query_only(self):
        bank = transport.AnchorQKCacheBank((4,))
        calls = []
        processor = self._processor(bank, calls, block_index=4)
        route = transport.DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_ALL
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=route,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, torch.full((1, self.TOKENS, 2), 10.0))
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=route,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, torch.full((1, 2 * self.TOKENS, 2), 100.0))
        query, key, value = calls[-1]
        self.assertTrue(
            torch.equal(query[self.TOKENS :], torch.full_like(query[self.TOKENS :], 11))
        )
        self.assertTrue(torch.equal(key, torch.full_like(key, 102)))
        self.assertTrue(torch.equal(value, torch.full_like(value, 103)))
        bank.assert_empty()

    def test_dual_route_late_all_reprojects_source_at_target_positions(self):
        tokens = transport.LATENT_PHASES * 4
        bank = transport.AnchorQKCacheBank((18,))
        calls = []
        processor = self._processor(bank, calls, block_index=18)
        route = transport.DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_ALL
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=route,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, torch.full((1, tokens, 2), 10.0))
        self.assertEqual(bank.capture_count, 0)

        current = torch.cat(
            (
                torch.full((1, tokens, 2), 2.0),
                torch.full((1, tokens, 2), 100.0),
            ),
            dim=1,
        )
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=route,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, current)
        query, key, value = calls[-1]
        self.assertTrue(
            torch.equal(query[tokens:], torch.full_like(query[tokens:], 101))
        )
        self.assertTrue(torch.equal(key[tokens:], torch.full_like(key[tokens:], 4)))
        self.assertTrue(torch.equal(value[tokens:], torch.full_like(value[tokens:], 5)))
        self.assertEqual(bank.source_kv_late_all_replay_count, 1)
        bank.assert_empty()

    def test_dual_route_late_static75_keeps_only_dynamic_target_quarter(self):
        phases, spatial = transport.LATENT_PHASES, 4
        tokens = phases * spatial
        bank = transport.AnchorQKCacheBank((18,))
        calls = []
        processor = self._processor(bank, calls, block_index=18)
        route = transport.DUAL_HARD_Q_EARLY_SOURCE_KV_LATE_STATIC75
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=route,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, torch.zeros(1, tokens, 2))

        source = torch.ones(1, phases, spatial, 2)
        target = source.clone()
        target[:, 1:, 3] = 100.0
        current = torch.cat(
            (source.reshape(1, tokens, 2), target.reshape(1, tokens, 2)), dim=1
        )
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=route,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, current)
        _query, key, value = calls[-1]
        target_key = key[tokens:].reshape(phases, spatial, 1, 2)
        target_value = value[tokens:].reshape(phases, spatial, 1, 2)
        self.assertTrue(torch.equal(target_key[0], torch.full_like(target_key[0], 3)))
        self.assertTrue(
            torch.equal(target_value[0], torch.full_like(target_value[0], 4))
        )
        self.assertTrue(
            torch.equal(target_key[1:, :3], torch.full_like(target_key[1:, :3], 3))
        )
        self.assertTrue(
            torch.equal(target_value[1:, :3], torch.full_like(target_value[1:, :3], 4))
        )
        self.assertTrue(
            torch.equal(target_key[1:, 3], torch.full_like(target_key[1:, 3], 102))
        )
        self.assertTrue(
            torch.equal(target_value[1:, 3], torch.full_like(target_value[1:, 3], 103))
        )
        self.assertEqual(bank.source_kv_late_static75_replay_count, 1)
        bank.assert_empty()

    def test_step_candidate_mismatch_fails_closed(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE, bank, step_index=1, candidate_index=0, rank=0, ulysses_size=1
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, torch.zeros(1, self.TOKENS, 2))
        replay = transport.AnchorQKInvocation(
            transport.REPLAY, bank, step_index=2, candidate_index=0, rank=0, ulysses_size=1
        )
        with self.assertRaises(transport.AnchorQKTransportError):
            with transport.anchor_qk_invocation(replay):
                self._call(processor, torch.zeros(1, 2 * self.TOKENS, 2))

    def test_one_capture_can_condition_both_target_apg_branches(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=4,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            replay_uses=2,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, torch.zeros(1, self.TOKENS, 2))
        for _ in range(2):
            replay = transport.AnchorQKInvocation(
                transport.REPLAY,
                bank,
                step_index=4,
                candidate_index=0,
                rank=0,
                ulysses_size=1,
                replay_uses=2,
            )
            with transport.anchor_qk_invocation(replay):
                self._call(processor, torch.ones(1, 2 * self.TOKENS, 2))
        bank.assert_empty()
        self.assertEqual(bank.capture_count, 1)
        self.assertEqual(bank.replay_count, 2)

    def test_full_sequence_replay_conditions_source_free_t2v_target(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            replay_scope=transport.FULL_SEQUENCE,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, torch.full((1, self.TOKENS, 2), 7.0))
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            replay_scope=transport.FULL_SEQUENCE,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, torch.full((1, self.TOKENS, 2), 100.0))
        query, key, value = calls[-1]
        self.assertTrue(torch.equal(query, torch.full_like(query, 8)))
        self.assertTrue(torch.equal(key, torch.full_like(key, 9)))
        self.assertTrue(torch.equal(value, torch.full_like(value, 103)))
        bank.assert_empty()

    def test_sparse_temporal_residual_preserves_phase_zero_and_current_value(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        anchor = torch.zeros(1, tokens, 2)
        anchor[:, 4:] = 8.0
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_RESIDUAL_QK,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, anchor)
        current = torch.full((1, 2 * tokens, 2), 4.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_RESIDUAL_QK,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, current)
        query, key, value = calls[-1]
        target_q = query[tokens:].reshape(transport.LATENT_PHASES, 4, 1, 2)
        target_k = key[tokens:].reshape(transport.LATENT_PHASES, 4, 1, 2)
        self.assertTrue(torch.equal(target_q[0], torch.full_like(target_q[0], 5)))
        self.assertTrue(torch.equal(target_k[0], torch.full_like(target_k[0], 6)))
        self.assertGreater(int(torch.count_nonzero(target_q[1:] - 5)), 0)
        self.assertLessEqual(
            int(torch.count_nonzero((target_q[1:] - 5).sum(dim=(-1, -2)))),
            (transport.LATENT_PHASES - 1) * 1,
        )
        self.assertTrue(torch.equal(value, torch.full_like(value, 7)))

    def test_sparse_temporal_qkv_residual_moves_value_but_keeps_phase_zero(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        anchor = torch.zeros(1, tokens, 2)
        anchor[:, 4:] = 8.0
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_RESIDUAL_QKV,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, anchor)
        current = torch.full((1, 2 * tokens, 2), 4.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_RESIDUAL_QKV,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, current)
        _query, _key, value = calls[-1]
        target_v = value[tokens:].reshape(transport.LATENT_PHASES, 4, 1, 2)
        self.assertTrue(torch.equal(target_v[0], torch.full_like(target_v[0], 7)))
        self.assertGreater(int(torch.count_nonzero(target_v[1:] - 7)), 0)
        self.assertLessEqual(
            int(torch.count_nonzero((target_v[1:] - 7).sum(dim=(-1, -2)))),
            (transport.LATENT_PHASES - 1) * 1,
        )

    def test_sparse_temporal_attention_output_moves_aggregate_not_qkv(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        anchor = torch.zeros(1, tokens, 2)
        anchor[:, 4:] = 8.0
        capture = transport.AnchorQKInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_RESIDUAL_ATTN_OUTPUT,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(capture):
            self._call(processor, anchor)
        current = torch.full((1, 2 * tokens, 2), 4.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_RESIDUAL_ATTN_OUTPUT,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(replay):
            output = self._call(processor, current)

        query, key, value = calls[-1]
        self.assertTrue(torch.equal(query, torch.full_like(query, 5)))
        self.assertTrue(torch.equal(key, torch.full_like(key, 6)))
        self.assertTrue(torch.equal(value, torch.full_like(value, 7)))
        target = output[:, tokens:].reshape(1, transport.LATENT_PHASES, 4, 2)
        self.assertTrue(torch.equal(target[:, 0], torch.full_like(target[:, 0], 5)))
        self.assertGreater(int(torch.count_nonzero(target[:, 1:] - 5)), 0)
        self.assertLessEqual(
            int(torch.count_nonzero((target[:, 1:] - 5).sum(dim=-1))),
            (transport.LATENT_PHASES - 1) * 1,
        )
        bank.assert_empty()

    def test_correspondence_residual_aligns_permuted_anchor_tokens(self):
        phases, spatial, width = transport.LATENT_PHASES, 4, 4
        current = torch.full((1, phases * spatial, 1, width), 5.0)
        anchor = torch.zeros_like(current).reshape(1, phases, spatial, 1, width)
        for position in range(spatial):
            anchor[:, 1:, position] = float(position + 1)
        anchor = anchor.reshape_as(current)
        current_ref = torch.eye(spatial).reshape(1, 1, spatial, 1, width).repeat(
            1, phases, 1, 1, 1
        )
        anchor_ref = torch.flip(current_ref, dims=(2,))
        routed = transport._sparse_correspondence_temporal_residual(
            current,
            anchor,
            current_reference=current_ref.reshape_as(current),
            anchor_reference=anchor_ref.reshape_as(current),
            strength=0.25,
            anchor_stride=1,
        ).reshape(1, phases, spatial, 1, width)
        self.assertTrue(torch.equal(routed[:, 0], torch.full_like(routed[:, 0], 5)))
        changed = (routed[:, 1:] - 5).abs().sum(dim=(-1, -2)) > 0
        self.assertEqual(int(changed.sum()), phases - 1)
        self.assertTrue(changed[..., 0].all())
        self.assertFalse(changed[..., 1:].any())

    def test_attention_action_noop_contrast_is_additive_and_phase0_free(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        action = torch.zeros(1, tokens, 2)
        action.reshape(1, transport.LATENT_PHASES, 4, 2)[:, 1:, 2] = 8.0
        noop = torch.zeros_like(action)
        for hidden, slot in (
            (action, transport.ACTION_SLOT),
            (noop, transport.NOOP_SLOT),
        ):
            capture = transport.AnchorQKInvocation(
                transport.CAPTURE,
                bank,
                step_index=0,
                candidate_index=0,
                rank=0,
                ulysses_size=1,
                transport=transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
                transport_strength=0.25,
                slot=slot,
            )
            with transport.anchor_qk_invocation(capture):
                self._call(processor, hidden)
        current = torch.full((1, 2 * tokens, 2), 4.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_CONTRAST_ATTN_OUTPUT,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(replay):
            output = self._call(processor, current)
        target = output[:, tokens:].reshape(1, transport.LATENT_PHASES, 4, 2)
        self.assertTrue(torch.equal(target[:, 0], torch.full_like(target[:, 0], 5)))
        self.assertTrue(torch.allclose(target[:, 1:, 2], torch.full_like(target[:, 1:, 2], 7)))
        self.assertTrue(torch.equal(target[:, 1:, :2], torch.full_like(target[:, 1:, :2], 5)))
        self.assertTrue(torch.equal(target[:, 1:, 3:], torch.full_like(target[:, 1:, 3:], 5)))
        self.assertEqual(bank.capture_count, 2)
        self.assertEqual(bank.replay_count, 2)
        bank.assert_empty()

    def test_action_noop_observer_consumes_both_entries_and_is_exact_identity(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        for hidden, slot in (
            (torch.randn(1, tokens, 2), transport.ACTION_SLOT),
            (torch.randn(1, tokens, 2), transport.NOOP_SLOT),
        ):
            capture = transport.AnchorQKInvocation(
                transport.CAPTURE,
                bank,
                step_index=1,
                candidate_index=3,
                rank=0,
                ulysses_size=1,
                transport=transport.ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
                slot=slot,
                replay_uses=1,
            )
            with transport.anchor_qk_invocation(capture):
                self._call(processor, hidden)
        current = torch.randn(1, 2 * tokens, 2)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=1,
            candidate_index=3,
            rank=0,
            ulysses_size=1,
            transport=transport.ACTION_NOOP_OBSERVER_ATTN_OUTPUT,
        )
        with transport.anchor_qk_invocation(replay):
            output = self._call(processor, current)
        self.assertTrue(torch.equal(output, current + 1))
        self.assertEqual(bank.capture_count, 2)
        self.assertEqual(bank.replay_count, 2)
        bank.assert_empty()

    def test_qk_dynamic_static_contrast_changes_qk_but_not_value(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        dynamic = torch.zeros(1, tokens, 2)
        dynamic.reshape(1, transport.LATENT_PHASES, 4, 2)[:, 1:, 2] = 8.0
        static = torch.zeros_like(dynamic)
        for hidden, slot in (
            (dynamic, transport.ACTION_SLOT),
            (static, transport.NOOP_SLOT),
        ):
            capture = transport.AnchorQKInvocation(
                transport.CAPTURE,
                bank,
                step_index=0,
                candidate_index=0,
                rank=0,
                ulysses_size=1,
                transport=transport.TEMPORAL_CONTRAST_QK,
                transport_strength=0.25,
                slot=slot,
            )
            with transport.anchor_qk_invocation(capture):
                self._call(processor, hidden)
        current = torch.full((1, 2 * tokens, 2), 4.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.TEMPORAL_CONTRAST_QK,
            transport_strength=0.25,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, current)
        query, key, value = calls[-1]
        target_q = query[tokens:].reshape(transport.LATENT_PHASES, 4, 2)
        target_k = key[tokens:].reshape(transport.LATENT_PHASES, 4, 2)
        self.assertTrue(torch.equal(target_q[0], torch.full_like(target_q[0], 5)))
        self.assertTrue(torch.equal(target_k[0], torch.full_like(target_k[0], 6)))
        self.assertTrue(torch.allclose(target_q[1:, 2], torch.full_like(target_q[1:, 2], 7)))
        self.assertTrue(torch.allclose(target_k[1:, 2], torch.full_like(target_k[1:, 2], 8)))
        self.assertTrue(torch.equal(value, torch.full_like(value, 7)))
        self.assertEqual(bank.capture_count, 2)
        self.assertEqual(bank.replay_count, 2)
        bank.assert_empty()

    def test_correspondence_contrast_maps_action_and_hard_replaces_trajectory(self):
        phases, spatial, width = transport.LATENT_PHASES, 4, 4
        current = torch.full((1, phases * spatial, 1, width), 5.0)
        action = torch.zeros_like(current).reshape(1, phases, spatial, 1, width)
        action[:, 1:, 3] = 8.0
        action = action.reshape_as(current)
        noop = torch.zeros_like(current)
        current_ref = torch.eye(spatial).reshape(1, 1, spatial, 1, width).repeat(
            1, phases, 1, 1, 1
        )
        anchor_ref = torch.flip(current_ref, dims=(2,))
        soft = transport._sparse_correspondence_temporal_contrast(
            current,
            action,
            noop,
            current_reference=current_ref.reshape_as(current),
            anchor_reference=anchor_ref.reshape_as(current),
            strength=0.25,
            hard_replace=False,
            anchor_stride=1,
        ).reshape(1, phases, spatial, 1, width)
        hard = transport._sparse_correspondence_temporal_contrast(
            current,
            action,
            noop,
            current_reference=current_ref.reshape_as(current),
            anchor_reference=anchor_ref.reshape_as(current),
            strength=1.0,
            hard_replace=True,
            anchor_stride=1,
        ).reshape(1, phases, spatial, 1, width)
        mutual = transport._sparse_correspondence_temporal_contrast(
            current,
            action,
            noop,
            current_reference=current_ref.reshape_as(current),
            anchor_reference=anchor_ref.reshape_as(current),
            strength=0.5,
            hard_replace=False,
            mutual_gate=True,
        ).reshape(1, phases, spatial, 1, width)
        self.assertTrue(torch.equal(soft[:, 0], torch.full_like(soft[:, 0], 5)))
        self.assertTrue(torch.equal(hard[:, 0], torch.full_like(hard[:, 0], 5)))
        self.assertTrue(torch.allclose(soft[:, 1:, 0], torch.full_like(soft[:, 1:, 0], 7)))
        self.assertTrue(torch.allclose(hard[:, 1:, 0], torch.full_like(hard[:, 1:, 0], 13)))
        self.assertTrue(torch.equal(soft[:, 1:, 1:], torch.full_like(soft[:, 1:, 1:], 5)))
        self.assertTrue(torch.equal(hard[:, 1:, 1:], torch.full_like(hard[:, 1:, 1:], 5)))
        self.assertTrue(
            torch.allclose(mutual[:, 1:, 0], torch.full_like(mutual[:, 1:, 0], 9))
        )
        self.assertTrue(
            torch.equal(mutual[:, 1:, 1:], torch.full_like(mutual[:, 1:, 1:], 5))
        )

    def test_hard_phase_mean_contrast_replaces_temporal_mean_not_spatial_residual(self):
        phases, spatial, width = transport.LATENT_PHASES, 4, 2
        current_phase = torch.zeros(1, phases, spatial, 1, width)
        spatial_offsets = torch.tensor([-3.0, -1.0, 1.0, 3.0]).reshape(
            1, 1, spatial, 1, 1
        )
        current_phase += spatial_offsets
        current_phase += 100.0 * torch.arange(phases).reshape(1, phases, 1, 1, 1)
        action_phase = 10.0 * torch.arange(phases).reshape(1, phases, 1, 1, 1)
        action_phase = action_phase.expand_as(current_phase).clone()
        noop_phase = 2.0 * torch.arange(phases).reshape(1, phases, 1, 1, 1)
        noop_phase = noop_phase.expand_as(current_phase).clone()

        routed = transport._hard_phase_mean_temporal_contrast(
            current_phase.reshape(1, phases * spatial, 1, width),
            action_phase.reshape(1, phases * spatial, 1, width),
            noop_phase.reshape(1, phases * spatial, 1, width),
        ).reshape_as(current_phase)

        self.assertTrue(torch.equal(routed[:, 0], current_phase[:, 0]))
        expected_mean = 8.0 * torch.arange(phases, dtype=routed.dtype)
        self.assertTrue(
            torch.allclose(
                routed.mean(dim=(0, 2, 3, 4)),
                expected_mean,
            )
        )
        routed_residual = routed - routed.mean(dim=2, keepdim=True)
        current_residual = current_phase - current_phase.mean(dim=2, keepdim=True)
        self.assertTrue(torch.equal(routed_residual, current_residual))

    def test_hard_phase_mean_qk_uses_dynamic_static_anchor_and_keeps_target_value(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        dynamic = torch.zeros(1, tokens, 2)
        dynamic.reshape(1, transport.LATENT_PHASES, 4, 2)[:, 1:] = 8.0
        static = torch.zeros_like(dynamic)
        for hidden, slot in (
            (dynamic, transport.ACTION_SLOT),
            (static, transport.NOOP_SLOT),
        ):
            capture = transport.AnchorQKInvocation(
                transport.CAPTURE,
                bank,
                step_index=0,
                candidate_index=0,
                rank=0,
                ulysses_size=1,
                transport=transport.HARD_PHASE_MEAN_CONTRAST_QK,
                transport_strength=1.0,
                slot=slot,
            )
            with transport.anchor_qk_invocation(capture):
                self._call(processor, hidden)
        current = torch.full((1, 2 * tokens, 2), 4.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.HARD_PHASE_MEAN_CONTRAST_QK,
            transport_strength=1.0,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, current)
        query, key, value = calls[-1]
        target_q = query[tokens:].reshape(transport.LATENT_PHASES, 4, 2)
        target_k = key[tokens:].reshape(transport.LATENT_PHASES, 4, 2)
        self.assertTrue(torch.equal(target_q[0], torch.full_like(target_q[0], 5)))
        self.assertTrue(torch.equal(target_k[0], torch.full_like(target_k[0], 6)))
        self.assertTrue(torch.equal(target_q[1:], torch.full_like(target_q[1:], 13)))
        self.assertTrue(torch.equal(target_k[1:], torch.full_like(target_k[1:], 14)))
        self.assertTrue(torch.equal(value, torch.full_like(value, 7)))
        bank.assert_empty()

    def test_hard_prerope_phase_mean_routes_hidden_qk_and_keeps_original_value(self):
        bank = transport.AnchorQKCacheBank((0,))
        calls = []
        processor = self._processor(bank, calls)
        tokens = transport.LATENT_PHASES * 4
        dynamic = torch.zeros(1, tokens, 2)
        dynamic.reshape(1, transport.LATENT_PHASES, 4, 2)[:, 1:] = 8.0
        static = torch.zeros_like(dynamic)
        for hidden, slot in (
            (dynamic, transport.ACTION_SLOT),
            (static, transport.NOOP_SLOT),
        ):
            capture = transport.AnchorQKInvocation(
                transport.CAPTURE,
                bank,
                step_index=0,
                candidate_index=0,
                rank=0,
                ulysses_size=1,
                transport=transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
                transport_strength=1.0,
                slot=slot,
            )
            with transport.anchor_qk_invocation(capture):
                self._call(processor, hidden)
        current = torch.full((1, 2 * tokens, 2), 4.0)
        replay = transport.AnchorQKInvocation(
            transport.REPLAY,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport=transport.HARD_PREROPE_PHASE_MEAN_CONTRAST_QK,
            transport_strength=1.0,
        )
        with transport.anchor_qk_invocation(replay):
            self._call(processor, current)
        query, key, value = calls[-1]
        target_q = query[tokens:].reshape(transport.LATENT_PHASES, 4, 2)
        target_k = key[tokens:].reshape(transport.LATENT_PHASES, 4, 2)
        self.assertTrue(torch.equal(target_q[0], torch.full_like(target_q[0], 5)))
        self.assertTrue(torch.equal(target_k[0], torch.full_like(target_k[0], 6)))
        self.assertTrue(torch.equal(target_q[1:], torch.full_like(target_q[1:], 13)))
        self.assertTrue(torch.equal(target_k[1:], torch.full_like(target_k[1:], 14)))
        self.assertTrue(torch.equal(value, torch.full_like(value, 7)))
        bank.assert_empty()

    def test_temporal_kernel_contrast_uses_target_value_and_keeps_phase_zero(self):
        phases = transport.LATENT_PHASES
        spatial = 4
        heads = 1
        width = phases
        shape = (1, phases, spatial, heads, width)
        current_output = torch.full(shape, 5.0)
        current_value = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone()
        action_query = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone() * 8.0
        action_key = action_query.clone()
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)
        action_value = torch.zeros(shape)
        action_value[:, :, 2] = 1000.0
        noop_value = torch.zeros(shape)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        routed = transport._temporal_attention_kernel_contrast_output(
            flat(current_output),
            flat(current_value),
            flat(action_query),
            flat(action_key),
            flat(action_value),
            flat(noop_query),
            flat(noop_key),
            flat(noop_value),
            strength=1.0,
        ).reshape(shape)
        self.assertTrue(torch.equal(routed[:, 0], current_output[:, 0]))
        self.assertGreater(
            int(torch.count_nonzero(routed[:, 1:] - current_output[:, 1:])), 0
        )
        # A huge anchor V only selects the active anchor sites; it is never
        # copied.  The routed magnitude is determined by the target's own V.
        self.assertLess(float((routed - current_output).abs().max()), 2.0)

        constant_target = torch.full_like(current_value, 7.0)
        constant_routed = transport._temporal_attention_kernel_contrast_output(
            flat(current_output),
            flat(constant_target),
            flat(action_query),
            flat(action_key),
            flat(action_value),
            flat(noop_query),
            flat(noop_key),
            flat(noop_value),
            strength=1.0,
        ).reshape(shape)
        self.assertTrue(torch.allclose(constant_routed, current_output, atol=1e-5))

    def test_target_gated_hard_kernel_changes_only_most_active_target_sites(self):
        phases = transport.LATENT_PHASES
        spatial = 10
        heads = 1
        width = phases
        shape = (1, phases, spatial, heads, width)
        current_output = torch.zeros(shape)
        for site in range(spatial):
            current_output[:, :, site] = (
                torch.arange(phases).reshape(1, phases, 1, 1)
                * float(site + 1)
            )
        current_value = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone()
        action_query = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone() * 8.0
        action_key = action_query.clone()
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)
        action_value = torch.zeros(shape)
        action_value[:, :, 3] = 1000.0
        noop_value = torch.zeros(shape)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        routed = transport._target_gated_hard_temporal_kernel_contrast_output(
            flat(current_output),
            flat(current_value),
            flat(action_query),
            flat(action_key),
            flat(action_value),
            flat(noop_query),
            flat(noop_key),
            flat(noop_value),
            strength=1.0,
            target_keep_fraction=0.10,
        ).reshape(shape)
        self.assertTrue(torch.equal(routed[:, 0], current_output[:, 0]))
        changed = (
            (routed - current_output)
            .abs()
            .sum(dim=(1, 3, 4))
            .squeeze(0)
            .ne(0)
        )
        self.assertEqual(int(changed.sum()), 1)
        self.assertTrue(bool(changed[-1]))
        self.assertTrue(
            torch.equal(routed[:, :, :-1], current_output[:, :, :-1])
        )

    def test_correspondence_kernel_uses_local_qk_and_target_value_only(self):
        phases = transport.LATENT_PHASES
        spatial = 8
        heads = 1
        width = phases
        shape = (1, phases, spatial, heads, width)
        current_output = torch.zeros(shape)
        for site in range(spatial):
            current_output[:, :, site] = (
                torch.arange(phases).reshape(1, phases, 1, 1)
                * float(site + 1)
            )
        current_value = torch.eye(phases).reshape(
            1, phases, 1, 1, width
        ).expand(shape).clone()
        # Make the most-active target site's phase-zero feature correspond to
        # anchor site four.  Anchor V is used only to choose this index.
        current_value[:, 0, -1].zero_()
        current_value[:, 0, -1, 0, 4] = 1.0
        action_value = torch.zeros(shape)
        noop_value = torch.zeros(shape)
        action_value[:, 0, 4, 0, 4] = 1000.0
        noop_value[:, 0, 4, 0, 4] = 1000.0
        action_query = torch.zeros(shape)
        action_key = torch.zeros(shape)
        action_query[:, :, 4] = (
            torch.eye(phases).reshape(1, phases, 1, width) * 8.0
        )
        action_key[:, :, 4] = action_query[:, :, 4]
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)

        def flat(value):
            return value.reshape(1, phases * spatial, heads, width)

        routed = (
            transport._correspondence_gated_hard_temporal_kernel_contrast_output(
                flat(current_output),
                flat(current_value),
                flat(action_query),
                flat(action_key),
                flat(action_value),
                flat(noop_query),
                flat(noop_key),
                flat(noop_value),
                strength=1.0,
                target_keep_fraction=0.125,
                anchor_stride=1,
            ).reshape(shape)
        )
        self.assertTrue(torch.equal(routed[:, 0], current_output[:, 0]))
        changed = (
            (routed - current_output)
            .abs()
            .sum(dim=(1, 3, 4))
            .squeeze(0)
            .ne(0)
        )
        self.assertEqual(int(changed.sum()), 1)
        self.assertTrue(bool(changed[-1]))
        # The 1000-valued anchor feature only defines an argmax match.  The
        # output scale is bounded by the selected target trajectory RMS.
        self.assertLess(float(routed.abs().max()), 1000.0)

    def test_event01_role_graph_uses_source_values_and_proposal_local_support(self):
        torch.manual_seed(7)
        phases = transport.LATENT_PHASES
        spatial = transport.EVENT01_SPATIAL_HEIGHT * transport.EVENT01_SPATIAL_WIDTH
        shape = (1, phases * spatial, 1, 8)
        current_output = torch.randn(shape)
        current_value = torch.randn(shape)
        action_query = torch.randn(shape)
        action_key = torch.randn(shape)
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)

        routed_0 = transport._event01_role_graph_hard_attention_output(
            current_output,
            current_value,
            action_query,
            action_key,
            noop_query,
            noop_key,
            strength=1.0,
            proposal_index=0,
        ).reshape(1, phases, spatial, 1, 8)
        routed_4 = transport._event01_role_graph_hard_attention_output(
            current_output,
            current_value,
            action_query,
            action_key,
            noop_query,
            noop_key,
            strength=1.0,
            proposal_index=4,
        ).reshape(1, phases, spatial, 1, 8)
        original = current_output.reshape(1, phases, spatial, 1, 8)
        self.assertTrue(torch.equal(routed_0[:, 0], original[:, 0]))
        self.assertTrue(torch.equal(routed_4[:, 0], original[:, 0]))
        self.assertGreater(float((routed_0[:, 1:] - original[:, 1:]).abs().sum()), 0.0)
        self.assertGreater(float((routed_4[:, 1:] - original[:, 1:]).abs().sum()), 0.0)
        self.assertGreater(float((routed_0 - routed_4).abs().sum()), 0.0)

        yy, xx = torch.meshgrid(
            torch.arange(transport.EVENT01_SPATIAL_HEIGHT),
            torch.arange(transport.EVENT01_SPATIAL_WIDTH),
            indexing="ij",
        )
        support = torch.zeros_like(xx, dtype=torch.bool)
        for (center_x, center_y), (scale_x, scale_y) in (
            (transport.EVENT01_SOURCE_ACTOR_XY, (5.5, 8.5)),
            (transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[0], (2.0, 1.75)),
        ):
            support |= (
                ((xx - center_x) / scale_x).square()
                + ((yy - center_y) / scale_y).square()
                <= 1.0
            )
        outside = ~support.flatten()
        self.assertTrue(torch.equal(routed_0[:, :, outside], original[:, :, outside]))

    def test_event01_role_logit_bias_is_additive_local_and_noop_identity(self):
        torch.manual_seed(17)
        phases = transport.LATENT_PHASES
        spatial = transport.EVENT01_SPATIAL_HEIGHT * transport.EVENT01_SPATIAL_WIDTH
        shape = (1, phases * spatial, 1, 8)
        current_output = torch.randn(shape)
        current_query = torch.randn(shape)
        current_key = torch.randn(shape)
        current_value = torch.randn(shape)
        noop_query = torch.randn(shape)
        noop_key = torch.randn(shape)

        identity = transport._event01_role_graph_logit_bias_attention_output(
            current_output,
            current_query,
            current_key,
            current_value,
            noop_query,
            noop_key,
            noop_query,
            noop_key,
            strength=1.0,
            proposal_index=2,
        )
        self.assertTrue(torch.equal(identity, current_output))

        action_query = noop_query.clone()
        action_key = noop_key.clone()
        actor_site = (
            int(transport.EVENT01_ANCHOR_ACTOR_XY[1])
            * transport.EVENT01_SPATIAL_WIDTH
            + int(transport.EVENT01_ANCHOR_ACTOR_XY[0])
        )
        action_query.reshape(1, phases, spatial, 1, 8)[:, :, actor_site] *= 4.0
        action_key.reshape(1, phases, spatial, 1, 8)[:, :, actor_site] *= 4.0
        routed = transport._event01_role_graph_logit_bias_attention_output(
            current_output,
            current_query,
            current_key,
            current_value,
            action_query,
            action_key,
            noop_query,
            noop_key,
            strength=1.0,
            proposal_index=2,
        ).reshape(1, phases, spatial, 1, 8)
        original = current_output.reshape(1, phases, spatial, 1, 8)
        self.assertTrue(torch.equal(routed[:, 0], original[:, 0]))
        self.assertGreater(float((routed[:, 1:] - original[:, 1:]).abs().sum()), 0.0)

        yy, xx = torch.meshgrid(
            torch.arange(transport.EVENT01_SPATIAL_HEIGHT),
            torch.arange(transport.EVENT01_SPATIAL_WIDTH),
            indexing="ij",
        )
        support = torch.zeros_like(xx, dtype=torch.bool)
        for (center_x, center_y), (scale_x, scale_y) in (
            (transport.EVENT01_SOURCE_ACTOR_XY, (5.5, 8.5)),
            (transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[2], (2.0, 1.75)),
        ):
            support |= (
                ((xx - center_x) / scale_x).square()
                + ((yy - center_y) / scale_y).square()
                <= 1.0
            )
        self.assertTrue(
            torch.equal(routed[:, :, ~support.flatten()], original[:, :, ~support.flatten()])
        )

    def test_event01_dynamic_role_trajectory_and_source_object_carry(self):
        self.assertEqual(len(transport.EVENT01_ANCHOR_ACTOR_TRAJECTORY_XY), 21)
        self.assertEqual(len(transport.EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY), 21)
        self.assertEqual(len(transport.EVENT01_TARGET_ACTOR_TRAJECTORY_XY), 21)
        self.assertEqual(len(transport.EVENT01_TARGET_OBJECT_LIFT_PROGRESS), 21)
        self.assertNotEqual(
            transport.EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY[0],
            transport.EVENT01_ANCHOR_OBJECT_TRAJECTORY_XY[-1],
        )
        torch.manual_seed(23)
        phases = transport.LATENT_PHASES
        spatial = transport.EVENT01_SPATIAL_HEIGHT * transport.EVENT01_SPATIAL_WIDTH
        shape = (1, phases * spatial, 1, 8)
        current_output = torch.randn(shape)
        current_query = torch.randn(shape)
        current_key = torch.randn(shape)
        current_value = torch.randn(shape)
        action_query = torch.randn(shape)
        action_key = torch.randn(shape)
        noop_query = torch.zeros(shape)
        noop_key = torch.zeros(shape)

        dynamic = transport._event01_role_graph_logit_bias_attention_output(
            current_output,
            current_query,
            current_key,
            current_value,
            action_query,
            action_key,
            noop_query,
            noop_key,
            strength=1.0,
            proposal_index=0,
            dynamic_roles=True,
            source_object_carry=False,
        )
        carried = transport._event01_role_graph_logit_bias_attention_output(
            current_output,
            current_query,
            current_key,
            current_value,
            action_query,
            action_key,
            noop_query,
            noop_key,
            strength=1.0,
            proposal_index=0,
            dynamic_roles=True,
            source_object_carry=True,
        )
        self.assertTrue(
            torch.equal(dynamic.reshape(1, phases, spatial, 1, 8)[:, 0],
                        current_output.reshape(1, phases, spatial, 1, 8)[:, 0])
        )
        self.assertTrue(
            torch.equal(carried.reshape(1, phases, spatial, 1, 8)[:, 0],
                        current_output.reshape(1, phases, spatial, 1, 8)[:, 0])
        )
        self.assertGreater(float((dynamic - current_output).abs().sum()), 0.0)
        self.assertGreater(float((carried - dynamic).abs().sum()), 0.0)

        identity = transport._event01_role_graph_logit_bias_attention_output(
            current_output,
            current_query,
            current_key,
            current_value,
            noop_query,
            noop_key,
            noop_query,
            noop_key,
            strength=1.0,
            proposal_index=0,
            dynamic_roles=True,
            source_object_carry=True,
        )
        self.assertTrue(torch.equal(identity, current_output))

    def test_event01_dynamic_source_object_value_is_local_and_strength_linear(self):
        torch.manual_seed(29)
        phases = transport.LATENT_PHASES
        spatial = transport.EVENT01_SPATIAL_HEIGHT * transport.EVENT01_SPATIAL_WIDTH
        shape = (1, phases * spatial, 1, 8)
        current = torch.randn(shape)
        full = transport._event01_dynamic_source_object_value(
            current, strength=1.0, proposal_index=0
        )
        half = transport._event01_dynamic_source_object_value(
            current, strength=0.5, proposal_index=0
        )
        current_phase = current.reshape(1, phases, spatial, 1, 8)
        full_phase = full.reshape(1, phases, spatial, 1, 8)
        self.assertTrue(torch.equal(full_phase[:, 0], current_phase[:, 0]))
        self.assertGreater(float((full - current).abs().sum()), 0.0)
        self.assertTrue(torch.allclose(half, current + 0.5 * (full - current)))
        other = transport._event01_dynamic_source_object_value(
            current, strength=1.0, proposal_index=4
        )
        self.assertGreater(float((other - full).abs().sum()), 0.0)

    def test_event01_dynamic_source_object_output_is_local_and_strength_linear(self):
        torch.manual_seed(31)
        phases = transport.LATENT_PHASES
        spatial = transport.EVENT01_SPATIAL_HEIGHT * transport.EVENT01_SPATIAL_WIDTH
        shape = (1, phases * spatial, 2, 4)
        current = torch.randn(shape)
        full = transport._event01_dynamic_source_object_output(
            current, strength=1.0, proposal_index=2
        )
        quarter = transport._event01_dynamic_source_object_output(
            current, strength=0.25, proposal_index=2
        )
        current_phase = current.reshape(1, phases, spatial, 2, 4)
        full_phase = full.reshape(1, phases, spatial, 2, 4)
        self.assertTrue(torch.equal(full_phase[:, 0], current_phase[:, 0]))
        self.assertGreater(float((full - current).abs().sum()), 0.0)
        self.assertTrue(torch.allclose(quarter, current + 0.25 * (full - current)))

    def test_event01_source_patch_move_preserves_scale_and_vacates_origin(self):
        phases = transport.LATENT_PHASES
        height = transport.EVENT01_SPATIAL_HEIGHT
        width = transport.EVENT01_SPATIAL_WIDTH
        spatial = height * width
        current = torch.zeros((1, phases * spatial, 1, 1), dtype=torch.float32)
        phase = current.reshape(1, phases, spatial, 1, 1)
        proposal = 1
        source_x, source_y = transport.EVENT01_SOURCE_OBJECT_PROPOSALS_XY[proposal]
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        distance = (
            ((xx - source_x) / 1.0).square()
            + ((yy - source_y) / 0.75).square()
        )
        source_flat = torch.topk(
            distance.flatten(), k=4, largest=False, sorted=True
        ).indices
        phase[:, 0, source_flat, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # A distinct wrong object exists at the source site in the last phase;
        # the patch move must replace it with the local ring background.
        phase[:, -1, source_flat, 0, 0] = 9.0
        full = transport._event01_dynamic_source_patch_move(
            current, strength=1.0, proposal_index=proposal
        ).reshape_as(phase)
        half = transport._event01_dynamic_source_patch_move(
            current, strength=0.5, proposal_index=proposal
        )
        self.assertTrue(torch.equal(full[:, 0], phase[:, 0]))
        self.assertTrue(torch.equal(full[:, -1, source_flat], torch.zeros((1, 4, 1, 1))))

        final_x, final_y = transport._event01_dynamic_target_centers(proposal)[-1][1]
        source_yx = torch.stack(
            (
                torch.div(source_flat, width, rounding_mode="floor"),
                source_flat.remainder(width),
            ),
            dim=1,
        ).float()
        target_y = torch.floor(source_yx[:, 0] - source_y + final_y + 0.5).long()
        target_x = torch.floor(source_yx[:, 1] - source_x + final_x + 0.5).long()
        target_flat = target_y * width + target_x
        self.assertEqual(int(target_flat.unique().numel()), 4)
        self.assertTrue(
            torch.equal(
                full[:, -1, target_flat, 0, 0],
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            )
        )
        self.assertTrue(torch.allclose(half, current + 0.5 * (full.reshape_as(current) - current)))

        # The production paired-forward path must take the carrier from the
        # explicit source suffix, not from the target caption's phase zero.
        explicit_source = torch.zeros_like(current)
        explicit_source_phase = explicit_source.reshape_as(phase)
        explicit_source_phase[:, 0, source_flat, 0, 0] = torch.tensor(
            [11.0, 12.0, 13.0, 14.0]
        )
        explicit = transport._event01_dynamic_source_patch_move(
            current,
            strength=1.0,
            proposal_index=proposal,
            source_output=explicit_source,
        ).reshape_as(phase)
        self.assertTrue(torch.equal(explicit[:, 0], phase[:, 0]))
        self.assertTrue(
            torch.equal(
                explicit[:, -1, target_flat, 0, 0],
                torch.tensor([[11.0, 12.0, 13.0, 14.0]]),
            )
        )

        # Every registered source proposal must produce a valid four-token
        # translation; automatic early SGA is therefore still executable.
        for proposal_index in range(transport.EVENT01_ROLE_PROPOSALS):
            moved = transport._event01_dynamic_source_patch_move(
                current, strength=1.0, proposal_index=proposal_index
            )
            self.assertEqual(tuple(moved.shape), tuple(current.shape))


if __name__ == "__main__":
    unittest.main()
