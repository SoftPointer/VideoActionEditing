from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_kv_replay as replay
import source_value_residual as residual

try:
    import torch
except ImportError:  # pragma: no cover - lightweight local environment
    torch = None


def _reference_varlen(torch_module, events=None):
    def run(
        q,
        k,
        v,
        *,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    ):
        if events is not None:
            events.append(
                {
                    "q": int(q.shape[0]),
                    "k": int(k.shape[0]),
                    "v": int(v.shape[0]),
                    "cu_q": tuple(int(item) for item in cu_seqlens_q.tolist()),
                    "cu_k": tuple(int(item) for item in cu_seqlens_k.tolist()),
                    "max_q": int(max_seqlen_q),
                    "max_k": int(max_seqlen_k),
                    "causal": bool(causal),
                }
            )
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        out = torch_module.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, is_causal=causal
        )
        return out.squeeze(0).transpose(0, 1)

    return run


class _BaseProcessor:
    def __init__(self, torch_module, *, key_offset=0.0, value_offset=0.0):
        self.torch = torch_module
        self.key_offset = float(key_offset)
        self.value_offset = float(value_offset)
        self.project_calls = 0
        self.direct_calls = 0
        self.varlen = _reference_varlen(torch_module)

    def _project_qkv(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        origin_hidden_states_seq_len,
        is_cross_attn,
    ):
        del attn, encoder_hidden_states, rotary_emb, origin_hidden_states_seq_len
        if is_cross_attn:
            raise AssertionError("test base only supports self-attention")
        self.project_calls += 1
        query = hidden_states.unsqueeze(2)
        key = (hidden_states + self.key_offset).unsqueeze(2)
        value = (hidden_states + self.value_offset).unsqueeze(2)
        return query, key, value

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        batch_image_vae_seqlen=None,
        text_features_length=None,
        origin_hidden_states_seq_len=None,
        split_hidden_states_seq_len=None,
        cu_seqlens_q_cache=None,
        max_seqlen_q_cache=None,
        cu_seqlens_k_cross_cache=None,
        cu_seqlens_q_cross_cache=None,
        max_seqlen_k_cross_cache=None,
        max_seqlen_q_cross_cache=None,
    ):
        del (
            attention_mask,
            batch_image_vae_seqlen,
            text_features_length,
            split_hidden_states_seq_len,
            cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache,
        )
        self.direct_calls += 1
        q, k, v = self._project_qkv(
            attn,
            hidden_states,
            encoder_hidden_states,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        q = q.squeeze(0).contiguous()
        k = k.squeeze(0).contiguous()
        v = v.squeeze(0).contiguous()
        output = self.varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        output = output.unsqueeze(0).flatten(2, 3).contiguous().type_as(q)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


def _kwargs(torch_module, length):
    return {
        "rotary_emb": torch_module.ones(
            (1, length, 1, 1), dtype=torch_module.complex64
        ),
        "batch_image_vae_seqlen": [length],
        "cu_seqlens_q_cache": torch_module.tensor(
            [0, length], dtype=torch_module.int32
        ),
        "max_seqlen_q_cache": torch_module.tensor(
            length, dtype=torch_module.int32
        ),
        "origin_hidden_states_seq_len": length,
    }


def _invocation(bank, *, mode, branch):
    return replay.source_kv_replay_invocation(
        bank,
        mode=mode,
        branch_tag=branch,
        generation=4,
        step_index=9,
        timestep_token="step-9:float64-0x1.0p-2",
        rank=0,
        ulysses_size=1,
    )


class ContractTests(unittest.TestCase):
    def test_formal_smoke_pins_kv_heads_for_across_heads_rms_norm(self):
        smoke_path = METHOD_ROOT / "source_value_residual_ulysses_smoke.py"
        tree = ast.parse(smoke_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Attention"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        kv_heads = keywords.get("kv_heads")
        self.assertIsInstance(kv_heads, ast.Name)
        self.assertEqual(kv_heads.id, "HEADS")

    def test_main_contract_preserves_full_route(self):
        value = residual.source_value_residual_contract(
            selection="late", operator=residual.MAIN_OPERATOR, gate=0.25
        )
        self.assertEqual(value["block_indices"], list(range(20, 30)))
        self.assertTrue(value["operator_is_main"])
        self.assertTrue(value["main_route_invariants"]["current_full_pair_keys"])
        self.assertTrue(
            value["main_route_invariants"][
                "current_full_pair_softmax_normalization"
            ]
        )
        self.assertFalse(value["main_route_invariants"]["cached_key_used"])
        self.assertFalse(value["cross_block_source_stream_invariant_claimed"])
        self.assertEqual(
            value["carrier_runtime_dependency_schema"], replay.CORE_SCHEMA
        )
        self.assertEqual(len(value["contract_digest"]), 64)

    def test_gate_and_operator_fail_closed(self):
        for bad in (-0.1, 1.1, float("nan"), float("inf"), True, "x"):
            with self.assertRaises(residual.SourceValueResidualContractError):
                residual.validate_fixed_gate(bad)
        with self.assertRaises(residual.SourceValueResidualContractError):
            residual.source_value_residual_contract(
                selection="late", operator="hard_kv", gate=0.25
            )


@unittest.skipIf(torch is None, "torch unavailable")
class TensorCoreTests(unittest.TestCase):
    def _core(self, *, operator=residual.MAIN_OPERATOR, gate=0.25, events=None):
        bank = replay.SourceKVCacheBank((0,))
        base = _BaseProcessor(torch, key_offset=0.125, value_offset=0.25)
        state = SimpleNamespace(ulysses_enabled=False)
        processor = residual.SourceValueResidualSelfAttnProcessor(
            base,
            block_index=0,
            cache_bank=bank,
            operator=operator,
            gate=gate,
            varlen_attention_fn=_reference_varlen(torch, events),
            get_parallel_state_fn=lambda: state,
            gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
        )
        attn = SimpleNamespace(
            to_out=(torch.nn.Identity(), torch.nn.Identity())
        )
        return bank, base, processor, attn

    @staticmethod
    def _capture(bank, processor, attn, source):
        with _invocation(
            bank, mode=replay.CAPTURE_MODE, branch=replay.CAPTURE_BRANCH_TAG
        ):
            return processor(attn, source, **_kwargs(torch, source.shape[1]))

    @staticmethod
    def _replay(bank, processor, attn, source, target, branch="frozen_action"):
        pair = torch.cat((source, target), dim=1)
        with _invocation(bank, mode=replay.REPLAY_MODE, branch=branch):
            return processor(attn, pair, **_kwargs(torch, pair.shape[1]))

    def test_full_k_residual_equals_source_value_interpolation_for_target_only(self):
        gate = 0.5
        bank, base, processor, attn = self._core(gate=gate)
        carrier = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.8, -0.4]]], dtype=torch.float32
        )
        current_source = torch.tensor(
            [[[0.2, -0.1], [-0.3, 0.4], [0.1, 0.6]]], dtype=torch.float32
        )
        target = torch.tensor(
            [[[-0.2, 0.3], [0.7, -0.5], [0.4, 0.2]]], dtype=torch.float32
        )
        self._capture(bank, processor, attn, carrier)
        actual = self._replay(bank, processor, attn, current_source, target)

        pair = torch.cat((current_source, target), dim=1)
        q = pair.unsqueeze(2).squeeze(0)
        k = (pair + base.key_offset).unsqueeze(2).squeeze(0)
        v_current = (pair + base.value_offset).unsqueeze(2).squeeze(0)
        v_cached = (carrier + base.value_offset).unsqueeze(2).squeeze(0)
        v_interpolated = torch.cat(
            (
                v_current[:3] + gate * (v_cached - v_current[:3]),
                v_current[3:],
            ),
            dim=0,
        )
        full_cu = torch.tensor([0, 6], dtype=torch.int32)
        reference = _reference_varlen(torch)
        base_full = reference(
            q,
            k,
            v_current,
            cu_seqlens_q=full_cu,
            cu_seqlens_k=full_cu,
            max_seqlen_q=6,
            max_seqlen_k=6,
            causal=False,
        )
        target_interpolated = reference(
            q[3:],
            k,
            v_interpolated,
            cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=full_cu,
            max_seqlen_q=3,
            max_seqlen_k=6,
            causal=False,
        )
        expected = (
            torch.cat((base_full[:3], target_interpolated), dim=0)
            .unsqueeze(0)
            .flatten(2, 3)
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))
        self.assertEqual(processor.residual_varlen_calls, 1)
        self.assertEqual(bank.replay_lookups, 1)
        stats = processor.statistics()
        self.assertTrue(stats["metrics"]["all_finite"])
        self.assertTrue(
            stats["metrics"]["combined_attention_output_all_finite"]
        )
        self.assertTrue(stats["metrics"]["projected_output_all_finite"])
        self.assertEqual(stats["metrics"]["calls"], 1)

    def test_full_k_operator_uses_one_target_to_full_varlen_call(self):
        events = []
        bank, _, processor, attn = self._core(gate=0.25, events=events)
        source = torch.randn(1, 3, 2)
        target = torch.randn(1, 3, 2)
        self._capture(bank, processor, attn, source + 0.4)
        events.clear()
        self._replay(bank, processor, attn, source, target)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["q"], 6)
        self.assertEqual(events[0]["k"], 6)
        self.assertEqual(events[1]["q"], 3)
        self.assertEqual(events[1]["k"], 6)
        self.assertEqual(events[1]["v"], 6)
        self.assertEqual(events[1]["cu_q"], (0, 3))
        self.assertEqual(events[1]["cu_k"], (0, 6))

    def test_zero_gate_replay_delegates_exact_official_processor(self):
        bank, base, processor, attn = self._core(gate=0.0)
        carrier = torch.randn(1, 3, 2)
        current = torch.randn(1, 3, 2)
        target = torch.randn(1, 3, 2)
        self._capture(bank, processor, attn, carrier)
        pair = torch.cat((current, target), dim=1)
        expected = base(attn, pair, **_kwargs(torch, 6))
        direct_before = base.direct_calls
        actual = self._replay(bank, processor, attn, current, target)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(base.direct_calls, direct_before + 1)
        self.assertEqual(processor.zero_gate_delegations, 1)
        self.assertEqual(processor.residual_varlen_calls, 0)
        self.assertEqual(bank.replay_lookups, 0)

    def test_main_operator_ignores_cached_key(self):
        bank, _, processor, attn = self._core(gate=0.5)
        carrier = torch.randn(1, 3, 2)
        current = torch.randn(1, 3, 2)
        target = torch.randn(1, 3, 2)
        self._capture(bank, processor, attn, carrier)
        first = self._replay(bank, processor, attn, current, target)
        entry = bank._entries[0]
        bank._entries[0] = replace(entry, key=entry.key + 1000.0)
        second = self._replay(
            bank, processor, attn, current, target, branch="adapted_action"
        )
        self.assertTrue(torch.equal(first, second))

    def test_diagnostics_are_explicit_and_have_expected_call_counts(self):
        source = torch.randn(1, 3, 2)
        current = torch.randn(1, 3, 2)
        target = torch.randn(1, 3, 2)
        bank_s, _, proc_s, attn_s = self._core(
            operator=residual.SOURCE_NORMALIZED_DIAGNOSTIC, gate=0.1
        )
        self._capture(bank_s, proc_s, attn_s, source)
        self._replay(bank_s, proc_s, attn_s, current, target)
        self.assertEqual(proc_s.residual_varlen_calls, 1)

        bank_k, _, proc_k, attn_k = self._core(
            operator=residual.CACHED_KV_DIAGNOSTIC, gate=0.1
        )
        self._capture(bank_k, proc_k, attn_k, source)
        self._replay(bank_k, proc_k, attn_k, current, target)
        self.assertEqual(proc_k.residual_varlen_calls, 2)

    def test_ulysses_inverse_receives_full_gathered_pair(self):
        shapes = []
        bank = replay.SourceKVCacheBank((0,))
        base = _BaseProcessor(torch)
        state = SimpleNamespace(
            ulysses_enabled=True, ulysses_rank=0, ulysses_size=1
        )

        def inverse(value, **kwargs):
            shapes.append((tuple(value.shape), dict(kwargs)))
            return value

        processor = residual.SourceValueResidualSelfAttnProcessor(
            base,
            block_index=0,
            cache_bank=bank,
            operator=residual.MAIN_OPERATOR,
            gate=0.25,
            varlen_attention_fn=_reference_varlen(torch),
            get_parallel_state_fn=lambda: state,
            gather_heads_scatter_seq_fn=inverse,
        )
        attn = SimpleNamespace(
            to_out=(torch.nn.Identity(), torch.nn.Identity())
        )
        self._capture(bank, processor, attn, torch.randn(1, 3, 2))
        self._replay(
            bank,
            processor,
            attn,
            torch.randn(1, 3, 2),
            torch.randn(1, 3, 2),
        )
        self.assertEqual(shapes[0][0][1], 3)
        self.assertEqual(shapes[1][0][1], 6)
        self.assertEqual(shapes[1][1], {"head_dim": 2, "seq_dim": 1})


@unittest.skipIf(torch is None, "torch unavailable")
class InstallRestoreTests(unittest.TestCase):
    def test_patch_scope_and_exact_restore(self):
        originals = []
        blocks = []
        for _ in range(30):
            processor = _BaseProcessor(torch)
            originals.append(processor)
            attention = SimpleNamespace(processor=processor)

            def set_processor(value, owner=attention):
                owner.processor = value

            attention.set_processor = set_processor
            blocks.append(SimpleNamespace(attn1=attention))
        transformer = SimpleNamespace(
            blocks=blocks, patch_vae_latent=lambda value, source_id: value
        )

        def factory(original, index, bank):
            return residual.SourceValueResidualSelfAttnProcessor(
                original,
                block_index=index,
                cache_bank=bank,
                operator=residual.MAIN_OPERATOR,
                gate=0.25,
                varlen_attention_fn=_reference_varlen(torch),
                get_parallel_state_fn=lambda: SimpleNamespace(
                    ulysses_enabled=False
                ),
                gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
            )

        handle = residual.install_source_value_residual(
            transformer,
            selection="late",
            operator=residual.MAIN_OPERATOR,
            gate=0.25,
            processor_factory=factory,
        )
        self.assertEqual(handle.indices, tuple(range(20, 30)))
        for index in range(20):
            self.assertIs(blocks[index].attn1.processor, originals[index])
        for index in range(20, 30):
            self.assertIsInstance(
                blocks[index].attn1.processor,
                residual.SourceValueResidualSelfAttnProcessor,
            )
        handle.restore()
        self.assertTrue(handle.restored)
        for index in range(30):
            self.assertIs(blocks[index].attn1.processor, originals[index])

    def test_factory_cannot_swap_block_cache_or_base_identity(self):
        originals = []
        blocks = []
        for _ in range(30):
            processor = _BaseProcessor(torch)
            originals.append(processor)
            attention = SimpleNamespace(processor=processor)

            def set_processor(value, owner=attention):
                owner.processor = value

            attention.set_processor = set_processor
            blocks.append(SimpleNamespace(attn1=attention))
        transformer = SimpleNamespace(
            blocks=blocks, patch_vae_latent=lambda value, source_id: value
        )

        def wrong_factory(original, index, bank):
            return residual.SourceValueResidualSelfAttnProcessor(
                original,
                block_index=20 if index == 21 else index,
                cache_bank=bank,
                operator=residual.MAIN_OPERATOR,
                gate=0.25,
                varlen_attention_fn=_reference_varlen(torch),
                get_parallel_state_fn=lambda: SimpleNamespace(
                    ulysses_enabled=False
                ),
                gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
            )

        with self.assertRaises(residual.SourceValueResidualContractError):
            residual.install_source_value_residual(
                transformer,
                selection="late",
                operator=residual.MAIN_OPERATOR,
                gate=0.25,
                processor_factory=wrong_factory,
            )
        for index in range(30):
            self.assertIs(blocks[index].attn1.processor, originals[index])


if __name__ == "__main__":
    unittest.main()
