from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import os
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_kv_replay as replay  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_main_and_ablation_block_scopes_are_explicit_for_1p3b(self) -> None:
        self.assertEqual(replay.resolve_block_indices(30, "all"), tuple(range(30)))
        self.assertEqual(replay.resolve_block_indices(30, "mid"), tuple(range(7, 23)))
        self.assertEqual(replay.resolve_block_indices(30, "late"), tuple(range(20, 30)))
        for invalid in ("early", "10-20", ""):
            with self.subTest(invalid=invalid), self.assertRaises(
                replay.SourceKVReplayContractError
            ):
                replay.resolve_block_indices(30, invalid)

    def test_contract_pins_post_rope_cache_and_pre_rope_shuffle_boundary(self) -> None:
        contract = replay.source_kv_replay_contract(selection="mid")
        self.assertEqual(contract["block_indices"], list(range(7, 23)))
        self.assertEqual(contract["scope_role"], "ablation")
        self.assertEqual(contract["pinned_mid_ablation_indices"], list(range(7, 23)))
        self.assertIn("post_rope", contract["captured_tensors"][0])
        self.assertEqual(contract["cache_autograd"], "detached_cloned")
        self.assertIs(contract["rotary_embedding"]["required"], True)
        self.assertIs(contract["rotary_embedding"]["none_allowed"], False)
        self.assertEqual(
            contract["gradient_checkpointing"]["context_fn"],
            "source_kv_replay_checkpoint_context_fn",
        )
        self.assertIs(
            contract["gradient_checkpointing"]["capture_checkpointing_allowed"],
            False,
        )
        self.assertIs(contract["ordinary_attention_fallback"], False)
        self.assertEqual(contract["capture_branch_tag"], "frozen_noop_carrier")
        self.assertEqual(
            contract["branch_tag_owner"],
            "outer_runner_explicit_no_processor_inference",
        )
        shuffle = contract["phase_shuffle"]
        self.assertIs(shuffle["implemented_by_this_module"], False)
        self.assertIs(shuffle["post_rope_cache_shuffle_allowed"], False)
        self.assertIn("pre_project_qkv_pre_rope", shuffle["required_stage_if_enabled"])
        self.assertEqual(len(contract["contract_digest"]), 64)
        main = replay.source_kv_replay_contract(selection="all")
        self.assertEqual(main["block_indices"], list(range(30)))
        self.assertEqual(main["scope_role"], "main_all_30")

    def test_capture_and_replay_layouts_are_disjoint_and_fail_closed(self) -> None:
        common = {
            "cu_seqlens_q_cache": [0, 3],
            "max_seqlen_q_cache": 3,
            "origin_hidden_states_seq_len": 3,
        }
        self.assertEqual(
            replay.validate_source_only_layout(
                gathered_sequence_length=3,
                batch_image_vae_seqlen=[3],
                **common,
            ),
            3,
        )
        with self.assertRaises(replay.SourceKVReplayContractError):
            replay.validate_equal_pair_layout(
                gathered_sequence_length=3,
                batch_image_vae_seqlen=[3],
                **common,
            )
        self.assertEqual(
            replay.validate_equal_pair_layout(
                gathered_sequence_length=6,
                batch_image_vae_seqlen=[6],
                cu_seqlens_q_cache=[0, 6],
                max_seqlen_q_cache=6,
                origin_hidden_states_seq_len=6,
            ),
            3,
        )
        invalid = (
            {"batch_image_vae_seqlen": [3, 3]},
            {"cu_seqlens_q_cache": [0, 3, 6]},
            {"max_seqlen_q_cache": 5},
            {"origin_hidden_states_seq_len": 3},
        )
        base = {
            "gathered_sequence_length": 6,
            "batch_image_vae_seqlen": [6],
            "cu_seqlens_q_cache": [0, 6],
            "max_seqlen_q_cache": 6,
            "origin_hidden_states_seq_len": 6,
        }
        for changes in invalid:
            values = dict(base)
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(
                replay.SourceKVReplayContractError
            ):
                replay.validate_equal_pair_layout(**values)

    def test_outer_branch_tags_are_mandatory(self) -> None:
        bank = replay.SourceKVCacheBank((0,))
        with self.assertRaises(replay.SourceKVReplayContractError):
            with replay.source_kv_replay_invocation(
                bank,
                mode="capture",
                branch_tag="adapted_action",
                generation=0,
                step_index=0,
                timestep_token="sigma:1.0",
                rank=0,
                ulysses_size=1,
            ):
                pass
        with self.assertRaises(replay.SourceKVReplayContractError):
            with replay.source_kv_replay_invocation(
                bank,
                mode="replay",
                branch_tag="action",
                generation=0,
                step_index=0,
                timestep_token="sigma:1.0",
                rank=0,
                ulysses_size=1,
            ):
                pass
        with self.assertRaises(replay.SourceKVReplayContractError):
            replay.current_source_kv_invocation()


class _BaseProcessor:
    """Official-boundary fake: output is already [1,S,H,D]."""

    def __init__(self, *, key_offset: float = 0.0, value_offset: float = 0.0):
        self.key_offset = key_offset
        self.value_offset = value_offset
        self.calls = []

    def _project_qkv(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        origin_hidden_states_seq_len,
        is_cross_attn,
    ):
        del attn
        self.calls.append(
            {
                "encoder_is_none": encoder_hidden_states is None,
                "origin": origin_hidden_states_seq_len,
                "is_cross": is_cross_attn,
                "length": int(hidden_states.shape[1]),
                "rotary_is_none": rotary_emb is None,
            }
        )
        query = hidden_states.unsqueeze(2)
        key = (hidden_states + self.key_offset).unsqueeze(2)
        value = (hidden_states + self.value_offset).unsqueeze(2)
        return query, key, value


def _reference_varlen(torch, events=None):
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
                (
                    "varlen",
                    tuple(int(value) for value in cu_seqlens_q.tolist()),
                    tuple(int(value) for value in cu_seqlens_k.tolist()),
                    int(max_seqlen_q),
                    int(max_seqlen_k),
                    causal,
                )
            )
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        v4 = v.transpose(0, 1).unsqueeze(0)
        output = torch.nn.functional.scaled_dot_product_attention(
            q4, k4, v4, is_causal=causal
        )
        return output.squeeze(0).transpose(0, 1)

    return run


def _common_kwargs(torch, length):
    return {
        "rotary_emb": torch.ones(
            (1, length, 1, 1), dtype=torch.complex64
        ),
        "batch_image_vae_seqlen": [length],
        "cu_seqlens_q_cache": torch.tensor([0, length], dtype=torch.int32),
        "max_seqlen_q_cache": torch.tensor(length, dtype=torch.int32),
        "origin_hidden_states_seq_len": length,
    }


def _invocation(bank, *, mode, branch, generation=7, step=3, token="sigma:0.750"):
    return replay.source_kv_replay_invocation(
        bank,
        mode=mode,
        branch_tag=branch,
        generation=generation,
        step_index=step,
        timestep_token=token,
        rank=0,
        ulysses_size=1,
    )


class TensorCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover - lightweight environments
            self.skipTest(f"torch unavailable: {error}")
        self.torch = torch

    def _core(self, *, key_offset=0.0, value_offset=0.0, state=None, inverse=None):
        torch = self.torch
        bank = replay.SourceKVCacheBank((0,))
        base = _BaseProcessor(key_offset=key_offset, value_offset=value_offset)
        if state is None:
            state = SimpleNamespace(ulysses_enabled=False)
        if inverse is None:
            inverse = lambda value, **kwargs: value
        processor = replay.SourceKVReplaySelfAttnProcessor(
            base,
            block_index=0,
            cache_bank=bank,
            varlen_attention_fn=_reference_varlen(torch),
            get_parallel_state_fn=lambda: state,
            gather_heads_scatter_seq_fn=inverse,
        )
        attention = SimpleNamespace(
            to_out=(torch.nn.Identity(), torch.nn.Identity())
        )
        return bank, base, processor, attention

    def _capture(self, bank, processor, attention, source):
        with _invocation(
            bank, mode="capture", branch=replay.CAPTURE_BRANCH_TAG
        ):
            return processor(
                attention, source, **_common_kwargs(self.torch, source.shape[1])
            )

    def _replay(self, bank, processor, attention, current_source, target, branch):
        hidden = self.torch.cat((current_source, target), dim=1)
        with _invocation(bank, mode="replay", branch=branch):
            return processor(
                attention, hidden, **_common_kwargs(self.torch, hidden.shape[1])
            )

    def test_cache_is_post_project_qkv_detached_and_storage_independent(self) -> None:
        torch = self.torch
        bank, base, processor, attention = self._core(
            key_offset=10.0, value_offset=20.0
        )
        source = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]], requires_grad=True
        )
        self._capture(bank, processor, attention, source)
        entry = bank.inspect_entry(0)
        self.assertTrue(torch.equal(entry.key.squeeze(2), source.detach() + 10.0))
        self.assertTrue(torch.equal(entry.value.squeeze(2), source.detach() + 20.0))
        self.assertFalse(entry.key.requires_grad)
        self.assertIsNone(entry.key.grad_fn)
        self.assertNotEqual(entry.key.data_ptr(), source.data_ptr())
        self.assertEqual(base.calls[0]["encoder_is_none"], True)
        self.assertEqual(base.calls[0]["is_cross"], False)
        self.assertEqual(base.calls[0]["rotary_is_none"], False)
        self.assertTrue(bank.complete)
        self.assertTrue(bank.receipt()["entries"][0]["detached"])
        self.assertEqual(
            bank.receipt()["entries"][0]["key_position_state"],
            "post_rope_verified_non_none_rotary",
        )

    def test_capture_stores_numeric_post_rope_key(self) -> None:
        torch = self.torch

        class NumericRoPEProcessor:
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
                self.is_cross_attn = is_cross_attn

                def apply_rope(value):
                    complex_value = torch.view_as_complex(
                        value.to(torch.float64).unflatten(3, (-1, 2))
                    )
                    return torch.view_as_real(
                        complex_value * rotary_emb
                    ).flatten(3, 4).type_as(value)

                projected = hidden_states.unsqueeze(2)
                return (
                    apply_rope(projected),
                    apply_rope(projected + 10.0),
                    projected + 20.0,
                )

        bank = replay.SourceKVCacheBank((0,))
        base = NumericRoPEProcessor()
        processor = replay.SourceKVReplaySelfAttnProcessor(
            base,
            block_index=0,
            cache_bank=bank,
            varlen_attention_fn=_reference_varlen(torch),
            get_parallel_state_fn=lambda: SimpleNamespace(ulysses_enabled=False),
            gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
        )
        attention = SimpleNamespace(
            to_out=(torch.nn.Identity(), torch.nn.Identity())
        )
        source = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [-2.0, 5.0]]], dtype=torch.float32
        )
        angles = torch.tensor([0.0, 0.5, -1.25], dtype=torch.float64)
        rotary = torch.polar(torch.ones_like(angles), angles).view(1, 3, 1, 1)
        kwargs = _common_kwargs(torch, 3)
        kwargs["rotary_emb"] = rotary
        with _invocation(
            bank, mode="capture", branch=replay.CAPTURE_BRANCH_TAG
        ):
            processor(attention, source, **kwargs)

        raw_key = (source + 10.0).unsqueeze(2)
        expected = torch.view_as_real(
            torch.view_as_complex(
                raw_key.to(torch.float64).unflatten(3, (-1, 2))
            )
            * rotary
        ).flatten(3, 4).type_as(raw_key)
        entry = bank.inspect_entry(0)
        self.assertTrue(torch.allclose(entry.key, expected))
        self.assertFalse(torch.allclose(entry.key, raw_key))
        self.assertFalse(base.is_cross_attn)
        stats = processor.statistics()
        self.assertEqual(stats["verified_post_rope_project_qkv_calls"], 1)
        self.assertEqual(stats["post_rope_phase_counts"]["eager"], 1)

    def test_source_only_carrier_is_invariant_to_paired_target_perturbation(self) -> None:
        torch = self.torch
        source = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]], dtype=torch.float32
        )
        target_a = torch.zeros_like(source)
        target_b = torch.tensor(
            [[[100.0, -80.0], [-90.0, 130.0], [75.0, 60.0]]],
            dtype=torch.float32,
        )
        pair_a = torch.cat((source, target_a), dim=1)
        pair_b = torch.cat((source, target_b), dim=1)

        bank_a, _, processor_a, attention_a = self._core()
        bank_b, _, processor_b, attention_b = self._core()
        # The capture API receives only the source slice.  There is no target
        # position, tensor, mask, or track through which pair_b's perturbation
        # could enter the carrier graph.
        self._capture(bank_a, processor_a, attention_a, pair_a[:, :3])
        self._capture(bank_b, processor_b, attention_b, pair_b[:, :3])
        entry_a = bank_a.inspect_entry(0)
        entry_b = bank_b.inspect_entry(0)
        self.assertTrue(torch.equal(entry_a.key, entry_b.key))
        self.assertTrue(torch.equal(entry_a.value, entry_b.value))

    def test_target_output_ignores_current_source_perturbation(self) -> None:
        torch = self.torch
        bank, _, processor, attention = self._core()
        carrier = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]], dtype=torch.float32
        )
        target = torch.tensor(
            [[[0.2, -0.1], [0.4, 0.3], [-0.2, 0.6]]], dtype=torch.float32
        )
        current_a = torch.zeros_like(carrier)
        current_b = torch.tensor(
            [[[100.0, -80.0], [-90.0, 130.0], [75.0, 60.0]]]
        )
        self._capture(bank, processor, attention, carrier)
        output_a = self._replay(
            bank, processor, attention, current_a, target, "frozen_action"
        )
        output_b = self._replay(
            bank, processor, attention, current_b, target, "adapted_action"
        )
        self.assertTrue(torch.equal(output_a[:, 3:], output_b[:, 3:]))
        self.assertFalse(torch.equal(output_a[:, :3], output_b[:, :3]))
        self.assertEqual(processor.capture_calls, 1)
        self.assertEqual(processor.replay_calls, 2)

    def test_target_output_changes_when_captured_source_kv_changes(self) -> None:
        torch = self.torch
        target = torch.tensor(
            [[[0.2, -0.1], [0.4, 0.3], [-0.2, 0.6]]], dtype=torch.float32
        )
        current_source = torch.zeros_like(target)
        carrier_a = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]], dtype=torch.float32
        )
        carrier_b = torch.tensor(
            [[[-2.0, 3.0], [4.0, -1.0], [2.5, 2.0]]], dtype=torch.float32
        )

        bank_a, _, processor_a, attention_a = self._core()
        bank_b, _, processor_b, attention_b = self._core()
        self._capture(bank_a, processor_a, attention_a, carrier_a)
        self._capture(bank_b, processor_b, attention_b, carrier_b)
        output_a = self._replay(
            bank_a, processor_a, attention_a, current_source, target, "adapted_action"
        )
        output_b = self._replay(
            bank_b, processor_b, attention_b, current_source, target, "adapted_action"
        )
        self.assertFalse(torch.allclose(output_a[:, 3:], output_b[:, 3:]))

    def test_backward_from_target_enters_only_current_target_route(self) -> None:
        torch = self.torch
        bank, _, processor, attention = self._core()
        carrier = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]], requires_grad=True
        )
        current_source = torch.tensor(
            [[[2.0, -1.0], [-3.0, 4.0], [1.5, 2.0]]], requires_grad=True
        )
        target = torch.tensor(
            [[[0.2, -0.1], [0.4, 0.3], [-0.2, 0.6]]], requires_grad=True
        )
        self._capture(bank, processor, attention, carrier)
        output = self._replay(
            bank, processor, attention, current_source, target, "adapted_action"
        )
        output[:, 3:].square().sum().backward()
        self.assertIsNone(carrier.grad)
        self.assertIsNotNone(current_source.grad)
        self.assertEqual(float(current_source.grad.abs().sum()), 0.0)
        self.assertIsNotNone(target.grad)
        self.assertGreater(float(target.grad.abs().sum()), 0.0)

    def test_non_reentrant_checkpoint_rebinds_two_combined_adapted_branches(self) -> None:
        torch = self.torch
        from torch.utils.checkpoint import checkpoint

        bank, _, processor, attention = self._core()
        carrier = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]], requires_grad=True
        )
        self._capture(bank, processor, attention, carrier)
        current_source = torch.tensor(
            [[[2.0, -1.0], [-3.0, 4.0], [1.5, 2.0]]], requires_grad=True
        )
        noop_target = torch.tensor(
            [[[0.2, -0.1], [0.4, 0.3], [-0.2, 0.6]]], requires_grad=True
        )
        action_target = torch.tensor(
            [[[-0.7, 0.9], [0.8, -0.4], [0.3, 1.1]]], requires_grad=True
        )
        kwargs = _common_kwargs(torch, 6)

        def run(branch, target):
            def checkpointed(source_value, target_value):
                return processor(
                    attention,
                    torch.cat((source_value, target_value), dim=1),
                    **kwargs,
                )

            with _invocation(bank, mode="replay", branch=branch):
                return checkpoint(
                    checkpointed,
                    current_source,
                    target,
                    use_reentrant=False,
                    context_fn=replay.source_kv_replay_checkpoint_context_fn,
                )

        noop_output = run("adapted_noop", noop_target)
        action_output = run("adapted_action", action_target)
        loss = noop_output[:, 3:].square().sum()
        loss = loss + 0.5 * action_output[:, 3:].square().sum()
        # Both outer branch contexts have ended.  Each checkpoint must restore
        # its own immutable invocation snapshot during the combined backward.
        with self.assertRaises(replay.SourceKVReplayContractError):
            replay.current_source_kv_invocation()
        loss.backward()

        self.assertIsNone(carrier.grad)
        self.assertIsNotNone(current_source.grad)
        self.assertEqual(float(current_source.grad.abs().sum()), 0.0)
        self.assertGreater(float(noop_target.grad.abs().sum()), 0.0)
        self.assertGreater(float(action_target.grad.abs().sum()), 0.0)
        receipt = bank.receipt()
        self.assertEqual(
            receipt["checkpoint_context_counts"],
            {"checkpoint_forward": 2, "checkpoint_recompute": 2},
        )
        for branch in ("adapted_noop", "adapted_action"):
            self.assertEqual(
                receipt["checkpoint_branch_counts"][branch],
                {"checkpoint_forward": 1, "checkpoint_recompute": 1},
            )
            self.assertEqual(
                receipt["replay_branch_phase_counts"][branch][
                    "checkpoint_forward"
                ],
                1,
            )
            self.assertEqual(
                receipt["replay_branch_phase_counts"][branch][
                    "checkpoint_recompute"
                ],
                1,
            )
        self.assertEqual(receipt["replay_phase_counts"]["checkpoint_forward"], 2)
        self.assertEqual(receipt["replay_phase_counts"]["checkpoint_recompute"], 2)
        with self.assertRaises(replay.SourceKVReplayContractError):
            replay.current_source_kv_invocation()

    def test_checkpoint_context_fails_closed_for_wrong_forward_and_stale_bank(self) -> None:
        torch = self.torch
        bank, _, processor, attention = self._core()
        source = torch.ones((1, 3, 2), dtype=torch.float32)
        self._capture(bank, processor, attention, source)

        with _invocation(bank, mode="replay", branch="adapted_action"):
            wrong_forward, _ = replay.source_kv_replay_checkpoint_context_fn()
        with self.assertRaises(replay.SourceKVReplayContractError):
            with wrong_forward:
                pass

        with _invocation(bank, mode="replay", branch="adapted_noop"):
            forward_context, recompute_context = (
                replay.source_kv_replay_checkpoint_context_fn()
            )
            with forward_context:
                pass
        bank.clear()
        with self.assertRaises(replay.SourceKVReplayContractError):
            with recompute_context:
                pass
        with self.assertRaises(replay.SourceKVReplayContractError):
            replay.current_source_kv_invocation()

        capture_bank = replay.SourceKVCacheBank((0,))
        with self.assertRaises(replay.SourceKVReplayContractError):
            with replay.source_kv_replay_invocation(
                capture_bank,
                mode="capture",
                branch_tag=replay.CAPTURE_BRANCH_TAG,
                generation=99,
                step_index=0,
                timestep_token="sigma:1.000",
                rank=0,
                ulysses_size=1,
            ):
                replay.source_kv_replay_checkpoint_context_fn()

    def test_stale_step_rank_shape_and_dtype_are_rejected(self) -> None:
        torch = self.torch
        bank, _, processor, attention = self._core()
        source = torch.ones((1, 3, 2), dtype=torch.float32)
        target = torch.zeros_like(source)
        self._capture(bank, processor, attention, source)

        with self.assertRaises(replay.SourceKVReplayContractError):
            with replay.source_kv_replay_invocation(
                bank,
                mode="replay",
                branch_tag="adapted_action",
                generation=7,
                step_index=4,
                timestep_token="sigma:0.500",
                rank=0,
                ulysses_size=1,
            ):
                pass

        with self.assertRaises(replay.SourceKVReplayContractError):
            self._replay(
                bank,
                processor,
                attention,
                torch.ones((1, 4, 2)),
                torch.zeros((1, 4, 2)),
                "adapted_action",
            )

        with self.assertRaises(replay.SourceKVReplayContractError):
            self._replay(
                bank,
                processor,
                attention,
                source.double(),
                target.double(),
                "adapted_action",
            )

        rank_bank = replay.SourceKVCacheBank((0,))
        rank_processor = replay.SourceKVReplaySelfAttnProcessor(
            _BaseProcessor(),
            block_index=0,
            cache_bank=rank_bank,
            varlen_attention_fn=_reference_varlen(torch),
            get_parallel_state_fn=lambda: SimpleNamespace(ulysses_enabled=False),
            gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
        )
        with self.assertRaises(replay.SourceKVReplayContractError):
            with replay.source_kv_replay_invocation(
                rank_bank,
                mode="capture",
                branch_tag=replay.CAPTURE_BRANCH_TAG,
                generation=8,
                step_index=0,
                timestep_token="sigma:1.000",
                rank=1,
                ulysses_size=2,
            ):
                rank_processor(attention, source, **_common_kwargs(torch, 3))

    def test_clear_retires_identity_and_prevents_silent_generation_reuse(self) -> None:
        torch = self.torch
        bank, _, processor, attention = self._core()
        source = torch.ones((1, 3, 2))
        self._capture(bank, processor, attention, source)
        bank.clear()
        with self.assertRaises(replay.SourceKVReplayContractError):
            with _invocation(
                bank, mode="capture", branch=replay.CAPTURE_BRANCH_TAG
            ):
                processor(attention, source, **_common_kwargs(torch, 3))

    def test_ulysses_uses_full_varlen_then_official_inverse(self) -> None:
        torch = self.torch
        events = []
        bank = replay.SourceKVCacheBank((0,))
        processor = replay.SourceKVReplaySelfAttnProcessor(
            _BaseProcessor(),
            block_index=0,
            cache_bank=bank,
            varlen_attention_fn=_reference_varlen(torch, events),
            get_parallel_state_fn=lambda: SimpleNamespace(
                ulysses_enabled=True, ulysses_rank=1, ulysses_size=2
            ),
            gather_heads_scatter_seq_fn=lambda value, **kwargs: (
                events.append(("inverse", kwargs["head_dim"], kwargs["seq_dim"]))
                or value
            ),
        )
        attention = SimpleNamespace(
            to_out=(torch.nn.Identity(), torch.nn.Identity())
        )
        source = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
        with replay.source_kv_replay_invocation(
            bank,
            mode="capture",
            branch_tag=replay.CAPTURE_BRANCH_TAG,
            generation=11,
            step_index=2,
            timestep_token="sigma:0.900",
            rank=1,
            ulysses_size=2,
        ):
            processor(attention, source, **_common_kwargs(torch, 3))
        with replay.source_kv_replay_invocation(
            bank,
            mode="replay",
            branch_tag="adapted_action",
            generation=11,
            step_index=2,
            timestep_token="sigma:0.900",
            rank=1,
            ulysses_size=2,
        ):
            processor(
                attention,
                torch.cat((torch.zeros_like(source), source), dim=1),
                **_common_kwargs(torch, 6),
            )
        self.assertEqual([event[0] for event in events], [
            "varlen", "inverse", "varlen", "inverse"
        ])
        self.assertEqual(events[0][1:5], ((0, 3), (0, 3), 3, 3))
        self.assertIs(events[0][5], False)
        self.assertEqual(events[2][1:5], ((0, 6), (0, 6), 6, 6))
        self.assertEqual(events[1], ("inverse", 2, 1))
        self.assertTrue(processor.saw_ulysses)

    def test_four_rank_ulysses_boundary_preserves_global_cache_and_local_output(self) -> None:
        torch = self.torch
        world_size = 4
        source_tokens = 6  # exercises capture padding: ceil(6 / 4) = 2
        pair_tokens = 12

        class GatheredHeadShardProcessor:
            def __init__(self, rank):
                self.rank = rank

            def _project_qkv(
                self,
                attn,
                hidden_states,
                encoder_hidden_states,
                rotary_emb,
                origin_hidden_states_seq_len,
                is_cross_attn,
            ):
                del attn, encoder_hidden_states
                self.assertion = (is_cross_attn, int(hidden_states.shape[1]))
                length = int(origin_hidden_states_seq_len)
                base = torch.arange(
                    length * 2,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                ).view(1, length, 1, 2)
                base = base + float(self.rank)

                def apply_rope(value):
                    complex_value = torch.view_as_complex(
                        value.to(torch.float64).unflatten(3, (-1, 2))
                    )
                    return torch.view_as_real(
                        complex_value * rotary_emb
                    ).flatten(3, 4).type_as(value)

                return apply_rope(base), apply_rope(base + 0.25), base + 0.5

        for rank in range(world_size):
            with self.subTest(rank=rank):
                bank = replay.SourceKVCacheBank((0,))

                def inverse(value, *, head_dim, seq_dim, rank=rank):
                    self.assertEqual((head_dim, seq_dim), (2, 1))
                    padding = (-int(value.shape[1])) % world_size
                    if padding:
                        value = torch.cat(
                            (
                                value,
                                value.new_zeros(
                                    (1, padding, value.shape[2], value.shape[3])
                                ),
                            ),
                            dim=1,
                        )
                    local = int(value.shape[1]) // world_size
                    shard = value[:, rank * local : (rank + 1) * local]
                    return shard.repeat(1, 1, world_size, 1)

                processor = replay.SourceKVReplaySelfAttnProcessor(
                    GatheredHeadShardProcessor(rank),
                    block_index=0,
                    cache_bank=bank,
                    varlen_attention_fn=_reference_varlen(torch),
                    get_parallel_state_fn=lambda rank=rank: SimpleNamespace(
                        ulysses_enabled=True,
                        ulysses_rank=rank,
                        ulysses_size=world_size,
                    ),
                    gather_heads_scatter_seq_fn=inverse,
                )
                attention = SimpleNamespace(
                    to_out=(torch.nn.Identity(), torch.nn.Identity())
                )
                capture_kwargs = _common_kwargs(torch, source_tokens)
                with replay.source_kv_replay_invocation(
                    bank,
                    mode="capture",
                    branch_tag=replay.CAPTURE_BRANCH_TAG,
                    generation=21,
                    step_index=4,
                    timestep_token="sigma:0.625",
                    rank=rank,
                    ulysses_size=world_size,
                ):
                    capture_output = processor(
                        attention,
                        torch.zeros((1, 2, 2), dtype=torch.bfloat16),
                        **capture_kwargs,
                    )
                entry = bank.inspect_entry(0)
                self.assertEqual(entry.key.shape, (1, source_tokens, 1, 2))
                self.assertEqual(entry.key.dtype, torch.bfloat16)
                self.assertEqual(capture_output.shape, (1, 2, 8))

                replay_kwargs = _common_kwargs(torch, pair_tokens)
                with replay.source_kv_replay_invocation(
                    bank,
                    mode="replay",
                    branch_tag="adapted_action",
                    generation=21,
                    step_index=4,
                    timestep_token="sigma:0.625",
                    rank=rank,
                    ulysses_size=world_size,
                ):
                    replay_output = processor(
                        attention,
                        torch.zeros((1, 3, 2), dtype=torch.bfloat16),
                        **replay_kwargs,
                    )
                self.assertEqual(replay_output.shape, (1, 3, 8))
                self.assertEqual(replay_output.dtype, torch.bfloat16)
                receipt = bank.receipt()
                self.assertEqual(receipt["identity"]["rank"], rank)
                self.assertEqual(receipt["identity"]["ulysses_size"], 4)

    def test_processor_rejects_cross_attention_mask_and_missing_context(self) -> None:
        torch = self.torch
        bank, _, processor, attention = self._core()
        source = torch.zeros((1, 3, 2))
        with self.assertRaises(replay.SourceKVReplayContractError):
            processor(attention, source, **_common_kwargs(torch, 3))
        with _invocation(
            bank, mode="capture", branch=replay.CAPTURE_BRANCH_TAG
        ):
            with self.assertRaises(replay.SourceKVReplayContractError):
                processor(
                    attention,
                    source,
                    encoder_hidden_states=source,
                    **_common_kwargs(torch, 3),
                )
            with self.assertRaises(replay.SourceKVReplayContractError):
                processor(
                    attention,
                    source,
                    attention_mask=torch.ones((3, 3)),
                    **_common_kwargs(torch, 3),
                )
            missing_rotary = _common_kwargs(torch, 3)
            missing_rotary["rotary_emb"] = None
            with self.assertRaises(replay.SourceKVReplayContractError):
                processor(attention, source, **missing_rotary)
            processor(attention, source, **_common_kwargs(torch, 3))


class PinnedBerniniBoundaryTests(unittest.TestCase):
    def test_pinned_official_processor_boundary_executes(self) -> None:
        root_value = os.environ.get("BERNINI_OFFICIAL_ROOT")
        veomni_value = os.environ.get("BERNINI_VEOMNI_ROOT")
        if not root_value or not veomni_value:
            self.skipTest("pinned Bernini/VeOmni source roots are not set")
        try:
            import torch
        except Exception as error:  # pragma: no cover
            self.skipTest(f"torch unavailable: {error}")

        root = Path(root_value).resolve()
        veomni_root = Path(veomni_value).resolve()
        transformer_source = root / "bernini/models/transformer_wan.py"
        if not transformer_source.is_file():
            self.fail(f"missing pinned Bernini source: {transformer_source}")
        digest = hashlib.sha256(transformer_source.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
        )
        inserted = [str(root), str(veomni_root)]
        sys.path[:0] = inserted
        try:
            from bernini.models.transformer_wan import (
                WanAttnProcessor2_0,
                WanTransformerBlock,
            )
        finally:
            for value in inserted:
                if value in sys.path:
                    sys.path.remove(value)

        block = WanTransformerBlock(dim=8, ffn_dim=16, num_heads=2)
        self.assertIsInstance(block.attn1.processor, WanAttnProcessor2_0)
        bank = replay.SourceKVCacheBank((0,))
        processor = replay.SourceKVReplaySelfAttnProcessor(
            block.attn1.processor,
            block_index=0,
            cache_bank=bank,
            # The pinned Bernini module selects installed FlashAttention at
            # import time, which cannot execute on this CPU boundary test.
            varlen_attention_fn=_reference_varlen(torch),
        )
        source = torch.randn((1, 4, 8), dtype=torch.float32)
        capture_kwargs = _common_kwargs(torch, 4)
        capture_kwargs["rotary_emb"] = torch.polar(
            torch.ones((1, 4, 1, 2), dtype=torch.float64),
            torch.linspace(0.0, 0.7, 4, dtype=torch.float64).view(1, 4, 1, 1),
        )
        with _invocation(
            bank, mode="capture", branch=replay.CAPTURE_BRANCH_TAG
        ):
            capture_output = processor(block.attn1, source, **capture_kwargs)
        self.assertEqual(capture_output.shape, source.shape)
        entry = bank.inspect_entry(0)
        self.assertEqual(entry.key.shape, (1, 4, 2, 4))
        self.assertFalse(entry.key.requires_grad)

        pair = torch.cat((torch.zeros_like(source), source), dim=1)
        replay_kwargs = _common_kwargs(torch, 8)
        replay_kwargs["rotary_emb"] = torch.polar(
            torch.ones((1, 8, 1, 2), dtype=torch.float64),
            torch.linspace(0.0, 1.4, 8, dtype=torch.float64).view(1, 8, 1, 1),
        )
        with _invocation(bank, mode="replay", branch="adapted_action"):
            replay_output = processor(block.attn1, pair, **replay_kwargs)
        self.assertEqual(replay_output.shape, pair.shape)
        self.assertEqual(processor.statistics()["verified_post_rope_project_qkv_calls"], 2)


class _MockAttention:
    def __init__(self, processor):
        self.processor = processor

    def set_processor(self, processor):
        self.processor = processor


class _MockBlock:
    def __init__(self, processor):
        self.attn1 = _MockAttention(processor)


class _MockTransformer:
    def __init__(self):
        self.blocks = [_MockBlock(_BaseProcessor()) for _ in range(30)]

    def patch_vae_latent(self):
        raise AssertionError("resolver must not call this method")


class InstallRestoreTests(unittest.TestCase):
    def test_wrapper_resolution_walks_every_candidate_without_skipping(self) -> None:
        transformer = _MockTransformer()
        empty_base = SimpleNamespace()
        decoder = SimpleNamespace(transformer=transformer)

        class Wrapper:
            diff_dec = decoder

            @staticmethod
            def get_base_model():
                return empty_base

        self.assertIs(replay.resolve_wan_transformer(Wrapper()), transformer)

    def test_patch_changes_only_selected_attn1_and_restores_exact_identity(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover
            self.skipTest(f"torch unavailable: {error}")
        renderer = SimpleNamespace(
            diff_dec=SimpleNamespace(transformer=_MockTransformer())
        )
        transformer = renderer.diff_dec.transformer
        originals = [block.attn1.processor for block in transformer.blocks]

        def factory(prior, index, bank):
            return replay.SourceKVReplaySelfAttnProcessor(
                prior,
                block_index=index,
                cache_bank=bank,
                varlen_attention_fn=_reference_varlen(torch),
                get_parallel_state_fn=lambda: SimpleNamespace(ulysses_enabled=False),
                gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
            )

        handle = replay.install_source_kv_replay(
            renderer, selection="mid", processor_factory=factory
        )
        self.assertEqual(handle.indices, tuple(range(7, 23)))
        self.assertEqual(handle.cache_bank.selected_block_indices, handle.indices)
        for index, block in enumerate(transformer.blocks):
            if 7 <= index < 23:
                self.assertIsInstance(
                    block.attn1.processor, replay.SourceKVReplaySelfAttnProcessor
                )
                self.assertIs(block.attn1.processor.cache_bank, handle.cache_bank)
            else:
                self.assertIs(block.attn1.processor, originals[index])
        receipt = handle.receipt()
        self.assertEqual(receipt["runtime"]["installed_block_count"], 16)
        self.assertIs(receipt["runtime"]["cache"]["complete"], False)
        handle.restore()
        handle.restore()
        self.assertTrue(handle.restored)
        for index, block in enumerate(transformer.blocks):
            self.assertIs(block.attn1.processor, originals[index])

    def test_restore_conflict_is_detected_before_any_partial_restore(self) -> None:
        try:
            import torch
        except Exception as error:  # pragma: no cover
            self.skipTest(f"torch unavailable: {error}")
        transformer = _MockTransformer()
        originals = [block.attn1.processor for block in transformer.blocks]

        def factory(prior, index, bank):
            return replay.SourceKVReplaySelfAttnProcessor(
                prior,
                block_index=index,
                cache_bank=bank,
                varlen_attention_fn=_reference_varlen(torch),
                get_parallel_state_fn=lambda: SimpleNamespace(ulysses_enabled=False),
                gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
            )

        handle = replay.install_source_kv_replay(
            transformer, selection="late", processor_factory=factory
        )
        replacement = object()
        transformer.blocks[29].attn1.processor = replacement
        with self.assertRaises(replay.SourceKVReplayContractError):
            handle.restore()
        for index in range(20, 29):
            self.assertIs(transformer.blocks[index].attn1.processor, handle.processors[index - 20])
        self.assertIs(transformer.blocks[29].attn1.processor, replacement)
        self.assertFalse(handle.restored)
        # Put back the installed object so the handle can restore cleanly.
        transformer.blocks[29].attn1.processor = handle.processors[-1]
        handle.restore()
        for index in range(30):
            self.assertIs(transformer.blocks[index].attn1.processor, originals[index])


if __name__ == "__main__":
    unittest.main()
