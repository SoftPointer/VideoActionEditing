from __future__ import annotations

import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anchor_cross_attention_transport as transport  # noqa: E402


class _Base:
    def __call__(self, _attn, hidden, **_kwargs):
        return hidden.clone()


class AnchorCrossAttentionTransportTest(unittest.TestCase):
    def _processor(self, bank):
        return transport.AnchorCrossAttentionProcessor(
            _Base(), block_index=0, cache_bank=bank
        )

    @staticmethod
    def _call(processor, hidden):
        return processor(
            object(),
            hidden,
            encoder_hidden_states=torch.zeros(1, 2, 3),
            batch_image_vae_seqlen=[int(hidden.shape[1])],
        )

    def test_no_context_is_exact_official_output(self):
        bank = transport.AnchorCrossAttentionCache((0,))
        processor = self._processor(bank)
        hidden = torch.randn(1, 84, 3)
        output = self._call(processor, hidden)
        self.assertTrue(torch.equal(output, hidden))
        self.assertEqual(processor.base_delegations, 1)

    def test_action_minus_noop_changes_only_sparse_target_temporal_rows(self):
        bank = transport.AnchorCrossAttentionCache((0,))
        processor = self._processor(bank)
        tokens = transport.LATENT_PHASES * 4
        action = torch.zeros(1, tokens, 3)
        action[:, 4:] = 8.0
        noop = torch.zeros_like(action)
        for slot, hidden in (
            (transport.ACTION_SLOT, action),
            (transport.NOOP_SLOT, noop),
        ):
            invocation = transport.AnchorCrossAttentionInvocation(
                transport.CAPTURE,
                bank,
                step_index=2,
                candidate_index=1,
                rank=0,
                ulysses_size=1,
                transport_strength=0.25,
                slot=slot,
            )
            with transport.anchor_cross_attention_invocation(invocation):
                self._call(processor, hidden)

        current = torch.full((1, 2 * tokens, 3), 4.0)
        replay = transport.AnchorCrossAttentionInvocation(
            transport.REPLAY,
            bank,
            step_index=2,
            candidate_index=1,
            rank=0,
            ulysses_size=1,
            transport_strength=0.25,
        )
        with transport.anchor_cross_attention_invocation(replay):
            output = self._call(processor, current)
        source, target = output[:, :tokens], output[:, tokens:]
        phase = target.reshape(1, transport.LATENT_PHASES, 4, 3)
        self.assertTrue(torch.equal(source, current[:, :tokens]))
        self.assertTrue(torch.equal(phase[:, 0], current[:, tokens : tokens + 4]))
        changed = (phase[:, 1:] != 4).any(dim=-1)
        self.assertGreater(int(changed.sum()), 0)
        self.assertLessEqual(int(changed.sum()), transport.LATENT_PHASES - 1)
        bank.assert_empty()
        self.assertEqual(bank.capture_count, 2)
        self.assertEqual(bank.replay_count, 2)

    def test_step_mismatch_fails_closed(self):
        bank = transport.AnchorCrossAttentionCache((0,))
        processor = self._processor(bank)
        tokens = transport.LATENT_PHASES * 2
        capture = transport.AnchorCrossAttentionInvocation(
            transport.CAPTURE,
            bank,
            step_index=0,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport_strength=0.1,
        )
        with transport.anchor_cross_attention_invocation(capture):
            self._call(processor, torch.zeros(1, tokens, 3))
        replay = transport.AnchorCrossAttentionInvocation(
            transport.REPLAY,
            bank,
            step_index=1,
            candidate_index=0,
            rank=0,
            ulysses_size=1,
            transport_strength=0.1,
        )
        with self.assertRaises(transport.AnchorCrossAttentionError):
            with transport.anchor_cross_attention_invocation(replay):
                self._call(processor, torch.zeros(1, 2 * tokens, 3))


if __name__ == "__main__":
    unittest.main()
