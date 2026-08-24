from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import object_memory_attn_processor_v1 as memory  # noqa: E402

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
        output = torch_module.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, is_causal=causal
        )
        return output.squeeze(0).transpose(0, 1)

    return run


def _apply_rotary(torch_module, tensor, rotary):
    as_complex = torch_module.view_as_complex(
        tensor.to(torch_module.float64).unflatten(3, (-1, 2)).contiguous()
    )
    return torch_module.view_as_real(as_complex * rotary).flatten(3, 4).type_as(tensor)


class _Attention:
    def __init__(self, torch_module):
        self.to_out = (torch_module.nn.Identity(), torch_module.nn.Identity())


class _BaseProcessor:
    def __init__(self, torch_module):
        self.torch = torch_module
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
        del attn, encoder_hidden_states, origin_hidden_states_seq_len
        if is_cross_attn:
            raise AssertionError("test base supports only self-attention")
        self.project_calls += 1
        query = hidden_states.unsqueeze(2)
        key = (hidden_states * 0.75 + 0.125).unsqueeze(2)
        value = (hidden_states * 1.25 - 0.25).unsqueeze(2)
        if rotary_emb is not None:
            query = _apply_rotary(self.torch, query, rotary_emb)
            key = _apply_rotary(self.torch, key, rotary_emb)
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
        query, key, value = self._project_qkv(
            attn,
            hidden_states,
            encoder_hidden_states,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        query = query.squeeze(0).contiguous()
        key = key.squeeze(0).contiguous()
        value = value.squeeze(0).contiguous()
        output = self.varlen(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        output = output.unsqueeze(0).flatten(2, 3).contiguous().type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


def _rotary(torch_module, length, width, *, phase_scale=1.0):
    angles = (
        torch_module.arange(length * (width // 2), dtype=torch_module.float64)
        .reshape(1, length, 1, width // 2)
        * (0.173 * phase_scale)
    )
    return torch_module.polar(torch_module.ones_like(angles), angles)


def _kwargs(torch_module, *, tokens, width=4, phase_scale=1.0):
    total = 2 * tokens
    return {
        "rotary_emb": _rotary(
            torch_module, total, width, phase_scale=phase_scale
        ),
        "batch_image_vae_seqlen": [total],
        "cu_seqlens_q_cache": torch_module.tensor(
            [0, total], dtype=torch_module.int32
        ),
        "max_seqlen_q_cache": torch_module.tensor(total, dtype=torch_module.int32),
        "origin_hidden_states_seq_len": total,
    }


def _masks(torch_module, tokens=5):
    source = {}
    target = {}
    for index, slot in enumerate(memory.SLOT_NAMES):
        source_mask = torch_module.zeros(tokens, dtype=torch_module.bool)
        target_mask = torch_module.zeros(tokens, dtype=torch_module.bool)
        source_mask[index] = True
        target_mask[index] = True
        source[slot] = source_mask
        target[slot] = target_mask
    return source, target


class ContractTests(unittest.TestCase):
    def test_contract_is_explicitly_non_production_and_pins_boundaries(self):
        value = memory.object_memory_contract(gate=0.25)
        self.assertFalse(value["production_ready"])
        self.assertEqual(value["block_indices"], [19, 24, 29])
        self.assertEqual(value["slots"], list(memory.SLOT_NAMES))
        self.assertIn(
            "native_world4_ulysses_forward_and_inverse_parity",
            value["unresolved_p0"],
        )
        self.assertIn(
            "gradient_checkpoint_context_rebinding", value["unresolved_p0"]
        )
        self.assertIn("external_wrong_object_donor_kv_for_swap", value["unresolved_p1"])
        self.assertEqual(len(value["contract_digest"]), 64)

    def test_gate_and_block_contract_fail_closed(self):
        for bad in (True, -0.1, 1.1, float("nan"), float("inf"), "bad"):
            with self.subTest(bad=bad), self.assertRaises(
                memory.ObjectMemoryContractError
            ):
                memory.validate_gate(bad)
        for blocks in ((), (1, 1), (True,), ("x",)):
            with self.subTest(blocks=blocks), self.assertRaises(
                memory.ObjectMemoryContractError
            ):
                memory.object_memory_contract(gate=0.1, block_indices=blocks)

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_invocation_modes_and_slot_keys_fail_closed(self):
        source, target = _masks(torch)
        with self.assertRaises(memory.ObjectMemoryContractError):
            with memory.object_memory_invocation(
                source_masks={"bone": source["bone"]},
                target_responsibility_masks=target,
                invocation_token="bad-keys",
            ):
                pass
        invalid_controls = (
            {"mode": "read", "drop_slots": ("bone",)},
            {"mode": "drop", "drop_slots": ()},
            {"mode": "swap", "swap_sources": {"bone": "bone"}},
            {"mode": "swap", "swap_sources": {"unknown": "bone"}},
        )
        for controls in invalid_controls:
            with self.subTest(controls=controls), self.assertRaises(
                memory.ObjectMemoryContractError
            ):
                with memory.object_memory_invocation(
                    source_masks=source,
                    target_responsibility_masks=target,
                    invocation_token="bad-controls",
                    **controls,
                ):
                    pass
        with memory.object_memory_invocation(
            source_masks=source,
            target_responsibility_masks=target,
            invocation_token="outer",
        ):
            with self.assertRaises(memory.ObjectMemoryContractError):
                with memory.object_memory_invocation(
                    source_masks=source,
                    target_responsibility_masks=target,
                    invocation_token="nested",
                ):
                    pass


@unittest.skipIf(torch is None, "torch unavailable")
class TensorCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.tokens = 5
        self.width = 4
        self.hidden = torch.randn(1, 2 * self.tokens, self.width)
        # Make the four source slots observably different for swap tests.
        self.hidden[0, :4] += torch.tensor(
            [[4.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0],
             [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 0.0, 7.0]]
        )
        self.attn = _Attention(torch)
        self.source_masks, self.target_masks = _masks(torch, self.tokens)
        self.state = SimpleNamespace(ulysses_enabled=False)

    def _processor(self, *, gate=0.5, events=None, base=None, state=None):
        if base is None:
            base = _BaseProcessor(torch)
        processor = memory.ObjectMemorySelfAttnProcessorV1(
            base,
            block_index=24,
            gate=gate,
            varlen_attention_fn=_reference_varlen(torch, events),
            get_parallel_state_fn=lambda: self.state if state is None else state,
            gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
        )
        return processor, base

    def _active(
        self,
        processor,
        *,
        kwargs=None,
        mode="read",
        drop_slots=(),
        swap_sources=None,
        source_masks=None,
        target_masks=None,
        rank=0,
        ulysses_size=1,
        token="test-call",
    ):
        if kwargs is None:
            kwargs = _kwargs(torch, tokens=self.tokens, width=self.width)
        if swap_sources is None:
            swap_sources = {}
        with memory.object_memory_invocation(
            source_masks=self.source_masks if source_masks is None else source_masks,
            target_responsibility_masks=(
                self.target_masks if target_masks is None else target_masks
            ),
            mode=mode,
            drop_slots=drop_slots,
            swap_sources=swap_sources,
            rank=rank,
            ulysses_size=ulysses_size,
            invocation_token=token,
        ):
            return processor(self.attn, self.hidden, **kwargs)

    def test_zero_gate_is_exact_one_call_delegate_without_project_or_invocation(self):
        class DelegateOnly:
            def __init__(self):
                self.calls = 0
                self.last = None

            def __call__(self, attn, hidden_states, **kwargs):
                self.calls += 1
                self.last = (attn, hidden_states, kwargs)
                return hidden_states.view(torch.uint8)

        base = DelegateOnly()
        processor = memory.ObjectMemorySelfAttnProcessorV1(
            base,
            block_index=0,
            gate=0.0,
            varlen_attention_fn=lambda *args, **kwargs: self.fail("not called"),
            get_parallel_state_fn=lambda: self.fail("not called"),
            gather_heads_scatter_seq_fn=lambda *args, **kwargs: self.fail("not called"),
        )
        # Deliberately invalid for the active contract: no masks, no RoPE, and
        # a cross-attention marker.  Zero gate must still be a pure delegate.
        marker = object()
        output = processor(
            self.attn,
            self.hidden,
            encoder_hidden_states=marker,
            attention_mask=marker,
        )
        self.assertEqual(base.calls, 1)
        self.assertIs(base.last[0], self.attn)
        self.assertIs(base.last[1], self.hidden)
        self.assertIs(base.last[2]["encoder_hidden_states"], marker)
        self.assertIs(base.last[2]["attention_mask"], marker)
        self.assertTrue(torch.equal(output, self.hidden.view(torch.uint8)))
        self.assertEqual(processor.zero_gate_delegations, 1)
        self.assertEqual(processor.active_calls, 0)

    def test_read_changes_only_responsible_target_rows_and_uses_all_slots(self):
        events = []
        processor, base = self._processor(events=events)
        kwargs = _kwargs(torch, tokens=self.tokens, width=self.width)
        official = base(self.attn, self.hidden, **kwargs)
        output = self._active(processor, kwargs=kwargs, token="read-all")
        self.assertTrue(torch.equal(output[:, : self.tokens], official[:, : self.tokens]))
        for index in range(4):
            self.assertFalse(
                torch.equal(
                    output[:, self.tokens + index],
                    official[:, self.tokens + index],
                )
            )
        # Target token 4 is outside every responsibility mask.
        self.assertTrue(
            torch.equal(output[:, -1], official[:, -1])
        )
        self.assertEqual(events[0]["q"], 2 * self.tokens)
        self.assertEqual(len(events), 1 + len(memory.SLOT_NAMES))
        stats = processor.statistics()
        self.assertEqual(stats["memory_varlen_calls"], 4)
        self.assertEqual(stats["mode_counts"]["read"], 1)
        self.assertEqual(stats["last_invocation_token"], "read-all")
        self.assertFalse(stats["production_ready"])

    def test_memory_delta_is_position_free_under_different_rotary_phases(self):
        kwargs_a = _kwargs(
            torch, tokens=self.tokens, width=self.width, phase_scale=0.5
        )
        kwargs_b = _kwargs(
            torch, tokens=self.tokens, width=self.width, phase_scale=2.25
        )

        processor_a, base_a = self._processor(gate=0.6)
        official_a = base_a(self.attn, self.hidden, **kwargs_a)
        output_a = self._active(processor_a, kwargs=kwargs_a, token="rope-a")

        processor_b, base_b = self._processor(gate=0.6)
        official_b = base_b(self.attn, self.hidden, **kwargs_b)
        output_b = self._active(processor_b, kwargs=kwargs_b, token="rope-b")

        delta_a = output_a - official_a
        delta_b = output_b - official_b
        self.assertTrue(torch.allclose(delta_a, delta_b, rtol=1e-5, atol=1e-6))
        self.assertFalse(torch.equal(official_a, official_b))

    def test_drop_and_swap_are_typed_structural_controls(self):
        kwargs = _kwargs(torch, tokens=self.tokens, width=self.width)

        read_processor, read_base = self._processor(gate=0.7)
        official = read_base(self.attn, self.hidden, **kwargs)
        read_output = self._active(read_processor, kwargs=kwargs, token="read")
        read_delta = read_output - official

        drop_processor, _ = self._processor(gate=0.7)
        drop_output = self._active(
            drop_processor,
            kwargs=kwargs,
            mode="drop",
            drop_slots=("bone",),
            token="drop-bone",
        )
        drop_delta = drop_output - official
        bone_row = self.tokens + memory.SLOT_NAMES.index("bone")
        self.assertTrue(torch.count_nonzero(read_delta[:, bone_row]).item() > 0)
        self.assertEqual(torch.count_nonzero(drop_delta[:, bone_row]).item(), 0)
        self.assertTrue(
            torch.allclose(
                drop_delta[:, self.tokens : bone_row],
                read_delta[:, self.tokens : bone_row],
            )
        )
        self.assertEqual(drop_processor.statistics()["drop_counts"]["bone"], 1)

        swap_processor, _ = self._processor(gate=0.7)
        swap_output = self._active(
            swap_processor,
            kwargs=kwargs,
            mode="swap",
            swap_sources={"bone": "dog_collar"},
            token="swap-bone-collar",
        )
        swap_delta = swap_output - official
        self.assertFalse(
            torch.allclose(swap_delta[:, bone_row], read_delta[:, bone_row])
        )
        swap_stats = swap_processor.statistics()
        self.assertEqual(swap_stats["swap_count"], 1)
        self.assertEqual(swap_stats["source_slot_use_counts"]["dog_collar"], 2)
        self.assertEqual(swap_stats["source_slot_use_counts"]["bone"], 0)

    def test_active_route_fails_closed_on_missing_context_and_world_mismatch(self):
        processor, _ = self._processor()
        kwargs = _kwargs(torch, tokens=self.tokens, width=self.width)
        with self.assertRaises(memory.ObjectMemoryContractError):
            processor(self.attn, self.hidden, **kwargs)
        with self.assertRaisesRegex(
            memory.ObjectMemoryContractError, "differs from runtime"
        ):
            self._active(
                processor,
                kwargs=kwargs,
                rank=0,
                ulysses_size=4,
                token="wrong-world",
            )

    def test_active_route_fails_closed_on_mask_dtype_shape_and_device(self):
        invalid_sources = []
        wrong_dtype = dict(self.source_masks)
        wrong_dtype["bone"] = wrong_dtype["bone"].to(torch.float32)
        invalid_sources.append(wrong_dtype)
        wrong_shape = dict(self.source_masks)
        wrong_shape["bone"] = torch.ones(self.tokens + 1, dtype=torch.bool)
        invalid_sources.append(wrong_shape)
        empty = dict(self.source_masks)
        empty["bone"] = torch.zeros(self.tokens, dtype=torch.bool)
        invalid_sources.append(empty)
        wrong_device = dict(self.source_masks)
        wrong_device["bone"] = torch.ones(
            self.tokens, dtype=torch.bool, device="meta"
        )
        invalid_sources.append(wrong_device)
        for index, source_masks in enumerate(invalid_sources):
            processor, _ = self._processor()
            with self.subTest(index=index), self.assertRaises(
                memory.ObjectMemoryContractError
            ):
                self._active(
                    processor,
                    source_masks=source_masks,
                    token=f"bad-mask-{index}",
                )

    def test_active_route_fails_closed_on_nonunit_rope_and_bad_qkv_dtype(self):
        processor, _ = self._processor()
        kwargs = _kwargs(torch, tokens=self.tokens, width=self.width)
        kwargs["rotary_emb"] = kwargs["rotary_emb"] * 1.01
        with self.assertRaisesRegex(
            memory.ObjectMemoryContractError, "unit-modulus"
        ):
            self._active(processor, kwargs=kwargs, token="bad-rope")

        class BadDtypeBase(_BaseProcessor):
            def _project_qkv(self, *args, **kwargs):
                query, key, value = super()._project_qkv(*args, **kwargs)
                return query, key.to(torch.float64), value

        processor, _ = self._processor(base=BadDtypeBase(torch))
        with self.assertRaisesRegex(
            memory.ObjectMemoryContractError, "dtypes differ"
        ):
            self._active(processor, token="bad-dtype")

    def test_ulysses_inverse_boundary_is_called_only_when_runtime_enables_it(self):
        inverse_calls = []
        state = SimpleNamespace(ulysses_enabled=True, ulysses_rank=0, ulysses_size=1)
        base = _BaseProcessor(torch)
        processor = memory.ObjectMemorySelfAttnProcessorV1(
            base,
            block_index=19,
            gate=0.2,
            varlen_attention_fn=_reference_varlen(torch),
            get_parallel_state_fn=lambda: state,
            gather_heads_scatter_seq_fn=lambda value, **kwargs: (
                inverse_calls.append(kwargs) or value
            ),
        )
        self._active(processor, token="mock-ulysses")
        self.assertEqual(inverse_calls, [{"head_dim": 2, "seq_dim": 1}])
        self.assertTrue(processor.statistics()["ulysses_observed"])


if __name__ == "__main__":
    unittest.main()
