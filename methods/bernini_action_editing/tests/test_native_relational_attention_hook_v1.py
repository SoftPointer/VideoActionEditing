#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_self_generated_relational_graph_observer_v1 as native  # noqa: E402
import native_relational_attention_hook_v1 as hook  # noqa: E402


class FakeOfficialProcessor:
    """CPU protocol double for the pinned official processor ABI."""

    def __init__(self, *, rank: int, block: int, layout: hook.World4RankLayout) -> None:
        self.rank = rank
        self.block = block
        self.layout = layout
        self.project_calls = 0
        self.base_calls = 0
        self.last_output = None

    def _project_qkv(
        self,
        _attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        origin_hidden_states_seq_len,
        is_cross_attn,
    ):
        self.project_calls += 1
        if origin_hidden_states_seq_len != self.layout.global_tokens:
            raise AssertionError("origin length differs")
        if is_cross_attn:
            if encoder_hidden_states is None or rotary_emb is not None:
                raise AssertionError("cross-attention ABI differs")
            query = torch.arange(
                self.layout.padded_local_tokens * hook.TOTAL_HEADS * hook.HEAD_DIM,
                dtype=torch.float32,
            ).reshape(1, self.layout.padded_local_tokens, hook.TOTAL_HEADS, hook.HEAD_DIM)
            query = query.mul(1.0e-5).add(self.rank + self.block * 1.0e-3)
            text = int(encoder_hidden_states.shape[1])
            key = torch.arange(
                text * hook.TOTAL_HEADS * hook.HEAD_DIM,
                dtype=torch.float32,
            ).reshape(1, text, hook.TOTAL_HEADS, hook.HEAD_DIM)
            key = key.mul(2.0e-5).add(self.block * 1.0e-3)
            value = key.neg()
            return query, key, value
        if encoder_hidden_states is not None or rotary_emb is None:
            raise AssertionError("self-attention ABI differs")
        total = self.layout.global_tokens * hook.TOTAL_HEADS * hook.HEAD_DIM
        full = torch.arange(total, dtype=torch.float32).reshape(
            1, self.layout.global_tokens, hook.TOTAL_HEADS, hook.HEAD_DIM
        )
        start = self.rank * hook.LOCAL_ATTN1_HEADS
        stop = start + hook.LOCAL_ATTN1_HEADS
        query = full[:, :, start:stop].mul(1.0e-6).add(self.block * 1.0e-3)
        key = torch.flip(query, dims=(-1,))
        value = query.neg()
        return query, key, value

    def __call__(self, attn, hidden_states, **kwargs):
        self.base_calls += 1
        self._project_qkv(
            attn,
            hidden_states,
            kwargs["encoder_hidden_states"],
            kwargs["rotary_emb"],
            kwargs["origin_hidden_states_seq_len"],
            kwargs["encoder_hidden_states"] is not None,
        )
        self.last_output = hidden_states.square().add(0.125)
        return self.last_output


FakeOfficialProcessor.__name__ = hook.OFFICIAL_PROCESSOR_CLASS
FakeOfficialProcessor.__module__ = hook.OFFICIAL_PROCESSOR_MODULE


class FakeAttention:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.scale = hook.HEAD_DIM ** -0.5

    def set_processor(self, value) -> None:
        self.processor = value


def base_invocation() -> native.CaptureInvocation:
    return native.CaptureInvocation(
        "appearance_0",
        "action",
        native.SigmaCell("mid", 18, 0.55),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        1,
        2,
    )


def partition() -> hook.ExhaustiveTextRolePartition:
    return hook.ExhaustiveTextRolePartition(
        ("human_agent", "moving_object", "null_context"),
        (0, 1, 2, 2),
    )


def self_kwargs(layout: hook.World4RankLayout):
    return {
        "encoder_hidden_states": None,
        "attention_mask": None,
        "rotary_emb": torch.ones((1, layout.global_tokens, 1, hook.HEAD_DIM // 2)),
        "batch_image_vae_seqlen": [layout.global_tokens],
        "text_features_length": [partition().active_text_tokens],
        "origin_hidden_states_seq_len": layout.global_tokens,
        "split_hidden_states_seq_len": layout.padded_local_tokens,
        "cu_seqlens_q_cache": torch.tensor([0, layout.global_tokens], dtype=torch.int32),
        "max_seqlen_q_cache": torch.tensor(layout.global_tokens),
        "cu_seqlens_k_cross_cache": None,
        "cu_seqlens_q_cross_cache": None,
        "max_seqlen_k_cross_cache": None,
        "max_seqlen_q_cross_cache": None,
    }


def cross_kwargs(layout: hook.World4RankLayout):
    text = partition().active_text_tokens
    return {
        "encoder_hidden_states": torch.arange(text * 8, dtype=torch.float32).reshape(1, text, 8),
        "attention_mask": None,
        "rotary_emb": None,
        "batch_image_vae_seqlen": [layout.global_tokens],
        "text_features_length": [text],
        "origin_hidden_states_seq_len": layout.global_tokens,
        "split_hidden_states_seq_len": layout.padded_local_tokens,
        "cu_seqlens_q_cache": torch.tensor([0, layout.global_tokens], dtype=torch.int32),
        "max_seqlen_q_cache": torch.tensor(layout.global_tokens),
        "cu_seqlens_k_cross_cache": torch.tensor([0, text], dtype=torch.int32),
        "cu_seqlens_q_cross_cache": torch.tensor(
            [0, layout.valid_local_tokens], dtype=torch.int32
        ),
        "max_seqlen_k_cross_cache": torch.tensor(text),
        "max_seqlen_q_cross_cache": torch.tensor(layout.valid_local_tokens),
    }


def capture_world4():
    invocation = base_invocation()
    rank_bank = hook.InMemoryWorld4RankShardBank()
    shards = []
    identity_checks = []
    bit_exact_checks = []
    for rank in range(hook.WORLD_SIZE):
        layout = hook.World4RankLayout(rank, 1, 2)
        rank_invocation = hook.RankCaptureInvocation(invocation, layout, partition())
        hidden = torch.arange(
            layout.padded_local_tokens * 8, dtype=torch.float32
        ).reshape(1, layout.padded_local_tokens, 8)
        with rank_bank.observe(rank_invocation):
            for block in hook.BLOCKS:
                attn = SimpleNamespace(scale=hook.HEAD_DIM ** -0.5)

                baseline1 = FakeOfficialProcessor(rank=rank, block=block, layout=layout)
                expected1 = baseline1(attn, hidden, **self_kwargs(layout))
                observed_base1 = FakeOfficialProcessor(rank=rank, block=block, layout=layout)
                observer1 = hook.NativeAttn1PostRopeQKObserver(
                    observed_base1, block_index=block, rank_bank=rank_bank
                )
                observed1 = observer1(attn, hidden, **self_kwargs(layout))
                identity_checks.append(observed1 is observed_base1.last_output)
                bit_exact_checks.append(torch.equal(expected1, observed1))
                if observed_base1.project_calls != 1 or "_project_qkv" in observed_base1.__dict__:
                    raise AssertionError("attn1 projection interception did not restore")

                baseline2 = FakeOfficialProcessor(rank=rank, block=block, layout=layout)
                expected2 = baseline2(attn, hidden, **cross_kwargs(layout))
                observed_base2 = FakeOfficialProcessor(rank=rank, block=block, layout=layout)
                observer2 = hook.NativeAttn2DerivedRoleProxyObserver(
                    observed_base2, block_index=block, rank_bank=rank_bank
                )
                observed2 = observer2(attn, hidden, **cross_kwargs(layout))
                identity_checks.append(observed2 is observed_base2.last_output)
                bit_exact_checks.append(torch.equal(expected2, observed2))
                if observed_base2.project_calls != 1 or "_project_qkv" in observed_base2.__dict__:
                    raise AssertionError("attn2 projection interception did not restore")
        shards.extend(rank_bank.take_rank(rank_invocation))
    return invocation, rank_bank, tuple(shards), identity_checks, bit_exact_checks


class NativeRelationalAttentionHookTests(unittest.TestCase):
    def test_delegate_second_clone_fault_scrubs_first_clone_and_restores(self) -> None:
        layout = hook.World4RankLayout(0, 1, 2)
        processor = FakeOfficialProcessor(rank=0, block=6, layout=layout)
        hidden = torch.arange(
            layout.padded_local_tokens * 8, dtype=torch.float32
        ).reshape(1, layout.padded_local_tokens, 8)
        real_owned_clone = hook._owned_contiguous_clone
        allocated = []
        call_count = 0

        def clone_fault(value, ownership):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("second projection clone fault")
            result = real_owned_clone(value, ownership)
            allocated.append(result)
            return result

        with mock.patch.object(
            hook, "_owned_contiguous_clone", side_effect=clone_fault
        ):
            with self.assertRaisesRegex(RuntimeError, "second projection clone"):
                hook._delegate_with_same_call_qk(
                    processor,
                    SimpleNamespace(scale=hook.HEAD_DIM ** -0.5),
                    hidden,
                    self_kwargs(layout),
                )
        self.assertEqual(len(allocated), 1)
        self.assertEqual(int(torch.count_nonzero(allocated[0])), 0)
        self.assertNotIn("_project_qkv", processor.__dict__)

    def test_native_consume_incomplete_pop_scrubs_removed_raw(self) -> None:
        invocation = base_invocation()
        query = torch.ones((1, hook.PHASES, 2, 1, 1))
        key = torch.ones_like(query)
        proxy = torch.zeros((1, hook.PHASES, 3, 2))
        proxy[:, :, 0] = 1.0
        capture = native.NativeBlockCapture(
            native.CAPTURE_SCHEMA,
            invocation,
            hook.BLOCKS[0],
            query,
            key,
            proxy,
        )
        bank = native.InMemoryNativeCaptureBank()
        bank._captures[invocation.key] = {hook.BLOCKS[0]: capture}
        with self.assertRaises(native.NativeRelationalObserverError):
            bank.consume(invocation)
        self.assertNotIn(invocation.key, bank._captures)
        self.assertEqual(int(torch.count_nonzero(query)), 0)
        self.assertEqual(int(torch.count_nonzero(key)), 0)
        self.assertEqual(int(torch.count_nonzero(proxy)), 0)
        self.assertEqual(bank.zeroized_count, 1)

    def test_commit_second_and_third_cat_fault_scrub_prior_allocations(self) -> None:
        real_owned_cat = hook._owned_contiguous_cat
        for fault_at in (2, 3):
            with self.subTest(fault_at=fault_at):
                invocation, _bank, shards, _identity, _exact = capture_world4()
                allocated = []
                call_count = 0

                def allocation_fault(values, *, dim, ownership):
                    nonlocal call_count
                    call_count += 1
                    if call_count == fault_at:
                        raise RuntimeError(f"cat allocation {fault_at} fault")
                    value = real_owned_cat(
                        values, dim=dim, ownership=ownership
                    )
                    allocated.append(value)
                    return value

                with mock.patch.object(
                    hook,
                    "_owned_contiguous_cat",
                    side_effect=allocation_fault,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, f"cat allocation {fault_at} fault"
                    ):
                        hook.commit_world4_shards_to_native_bank(
                            native_bank=native.InMemoryNativeCaptureBank(),
                            invocation=invocation,
                            rank_shards=shards,
                        )
                self.assertEqual(len(allocated), fault_at - 1)
                self.assertTrue(
                    all(int(torch.count_nonzero(value)) == 0 for value in allocated)
                )

    def test_reconstruct_second_and_third_clone_fault_scrub_prior_clones(self) -> None:
        real_owned_clone = hook._owned_contiguous_clone
        for fault_at in (2, 3):
            with self.subTest(fault_at=fault_at):
                invocation, _bank, shards, _identity, _exact = capture_world4()
                selected = sorted(
                    (row for row in shards if row.block_index == hook.BLOCKS[0]),
                    key=lambda row: row.invocation.layout.rank,
                )
                payloads = [
                    row.collective_payload_and_zeroize() for row in selected
                ]
                qk = torch.stack(
                    [value[0] for value in payloads], dim=0
                ).contiguous()
                proxy = torch.stack(
                    [value[1] for value in payloads], dim=0
                ).contiguous()
                metadata = [dict(value[2]) for value in payloads]
                allocated = []
                call_count = 0

                def allocation_fault(value, ownership):
                    nonlocal call_count
                    call_count += 1
                    if call_count == fault_at:
                        raise RuntimeError(f"clone allocation {fault_at} fault")
                    result = real_owned_clone(value, ownership)
                    allocated.append(result)
                    return result

                with mock.patch.object(
                    hook,
                    "_owned_contiguous_clone",
                    side_effect=allocation_fault,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, f"clone allocation {fault_at} fault"
                    ):
                        hook.reconstruct_world4_block_from_collectives(
                            invocation=invocation,
                            role_partition=partition(),
                            block_index=hook.BLOCKS[0],
                            qk_rank_major=qk,
                            proxy_rank_major=proxy,
                            rank_metadata=metadata,
                        )
                self.assertEqual(len(allocated), fault_at - 1)
                self.assertTrue(
                    all(int(torch.count_nonzero(value)) == 0 for value in allocated)
                )
                self.assertEqual(int(torch.count_nonzero(qk)), 0)
                self.assertEqual(int(torch.count_nonzero(proxy)), 0)

    def test_collective_payload_metadata_fault_scrubs_source_and_payloads(self) -> None:
        _invocation, _bank, shards, _identity, _exact = capture_world4()
        shard = shards[0]
        with mock.patch.object(
            hook.World4BlockRankShard,
            "collective_metadata",
            side_effect=RuntimeError("metadata fault"),
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata fault"):
                shard.collective_payload_and_zeroize()
        self.assertEqual(int(torch.count_nonzero(shard.query)), 0)
        self.assertEqual(int(torch.count_nonzero(shard.key)), 0)
        self.assertEqual(
            int(
                torch.count_nonzero(
                    shard.derived_qk_role_responsibility_proxy
                )
            ),
            0,
        )

    def test_reconstruct_late_metadata_fault_scrubs_prior_rank_clones(self) -> None:
        invocation, _bank, shards, _identity, _exact = capture_world4()
        selected = sorted(
            (row for row in shards if row.block_index == hook.BLOCKS[0]),
            key=lambda row: row.invocation.layout.rank,
        )
        payloads = [row.collective_payload_and_zeroize() for row in selected]
        qk = torch.stack([value[0] for value in payloads], dim=0).contiguous()
        proxy = torch.stack([value[1] for value in payloads], dim=0).contiguous()
        metadata = [dict(value[2]) for value in payloads]
        metadata[2]["rank"] = 99
        real_shard_type = hook.World4BlockRankShard
        constructed = []

        def retain_constructed(*args, **kwargs):
            value = real_shard_type(*args, **kwargs)
            constructed.append(value)
            return value

        with mock.patch.object(
            hook, "World4BlockRankShard", side_effect=retain_constructed
        ):
            with self.assertRaises(hook.NativeRelationalAttentionHookError):
                hook.reconstruct_world4_block_from_collectives(
                    invocation=invocation,
                    role_partition=partition(),
                    block_index=hook.BLOCKS[0],
                    qk_rank_major=qk,
                    proxy_rank_major=proxy,
                    rank_metadata=metadata,
                )
        self.assertEqual(len(constructed), 2)
        for row in constructed:
            for tensor in (
                row.query,
                row.key,
                row.derived_qk_role_responsibility_proxy,
            ):
                self.assertEqual(int(torch.count_nonzero(tensor)), 0)
        self.assertEqual(int(torch.count_nonzero(qk)), 0)
        self.assertEqual(int(torch.count_nonzero(proxy)), 0)

    def test_take_rank_join_fault_zeroizes_every_popped_partial(self) -> None:
        invocation, _old_bank, shards, _identity, _exact = capture_world4()
        rank_invocation = next(
            row.invocation for row in shards if row.invocation.layout.rank == 0
        )
        selected = {
            row.block_index: row
            for row in shards
            if row.invocation.layout.rank == 0
        }
        bank = hook.InMemoryWorld4RankShardBank()
        partials = {}
        for block in hook.BLOCKS:
            row = selected[block]
            qk = hook.Attn1PostRopeQKRankShard(
                rank_invocation, block, row.query, row.key
            )
            role = hook.DerivedRoleProxyRankShard(
                rank_invocation,
                block,
                row.derived_qk_role_responsibility_proxy,
            )
            partials[block] = hook._PartialBlock(qk=qk, role=role)
        partials[hook.BLOCKS[-1]].role = None
        bank._rows[rank_invocation.key] = partials
        with self.assertRaises(hook.NativeRelationalAttentionHookError):
            bank.take_rank(rank_invocation)
        self.assertNotIn(rank_invocation.key, bank._rows)
        for partial in partials.values():
            for value in (partial.qk, partial.role):
                if value is not None:
                    tensors = (
                        (value.query, value.key)
                        if isinstance(value, hook.Attn1PostRopeQKRankShard)
                        else (value.proxy,)
                    )
                    self.assertTrue(
                        all(int(torch.count_nonzero(tensor)) == 0 for tensor in tensors)
                    )

    def test_commit_constructor_and_partial_bank_faults_scrub_all_owned_raw(self) -> None:
        invocation, _rank_bank, shards, _identity, _exact = capture_world4()
        real_capture_type = native.NativeBlockCapture
        constructed = []

        def constructor_fault(*args, **kwargs):
            if constructed:
                raise RuntimeError("constructor fault")
            value = real_capture_type(*args, **kwargs)
            constructed.append(value)
            return value

        with mock.patch.object(
            native, "NativeBlockCapture", side_effect=constructor_fault
        ):
            with self.assertRaisesRegex(RuntimeError, "constructor fault"):
                hook.commit_world4_shards_to_native_bank(
                    native_bank=native.InMemoryNativeCaptureBank(),
                    invocation=invocation,
                    rank_shards=shards,
                )
        self.assertEqual(len(constructed), 1)
        self.assertEqual(int(torch.count_nonzero(constructed[0].query)), 0)
        self.assertTrue(
            all(
                int(torch.count_nonzero(tensor)) == 0
                for row in shards
                for tensor in (
                    row.query,
                    row.key,
                    row.derived_qk_role_responsibility_proxy,
                )
            )
        )

    def test_first_native_capture_constructor_fault_scrubs_every_argument(self) -> None:
        invocation, _rank_bank, shards, _identity, _exact = capture_world4()
        constructor_arguments = []

        def constructor_fault(*args, **_kwargs):
            constructor_arguments.extend(args[3:6])
            raise RuntimeError("first constructor fault")

        with mock.patch.object(
            native, "NativeBlockCapture", side_effect=constructor_fault
        ):
            with self.assertRaisesRegex(RuntimeError, "first constructor fault"):
                hook.commit_world4_shards_to_native_bank(
                    native_bank=native.InMemoryNativeCaptureBank(),
                    invocation=invocation,
                    rank_shards=shards,
                )
        self.assertEqual(len(constructor_arguments), 3)
        self.assertTrue(
            all(
                int(torch.count_nonzero(value)) == 0
                for value in constructor_arguments
            )
        )

        invocation, _rank_bank, shards, _identity, _exact = capture_world4()
        native_bank = native.InMemoryNativeCaptureBank()
        passed = []
        real_capture = native_bank.capture

        def bank_fault(value):
            passed.append(value)
            real_capture(value)
            if len(passed) == 2:
                raise RuntimeError("bank fault")

        with mock.patch.object(native_bank, "capture", side_effect=bank_fault):
            with self.assertRaisesRegex(RuntimeError, "bank fault"):
                hook.commit_world4_shards_to_native_bank(
                    native_bank=native_bank,
                    invocation=invocation,
                    rank_shards=shards,
                )
        self.assertFalse(native_bank._captures)
        self.assertTrue(
            all(
                int(torch.count_nonzero(tensor)) == 0
                for value in passed
                for tensor in (
                    value.query,
                    value.key,
                    value.derived_qk_role_responsibility_proxy,
                )
            )
        )

    def test_zeroization_accepts_real_inference_tensors_outside_forward(self) -> None:
        with torch.inference_mode():
            value = torch.ones((2, 3), dtype=torch.float32).clone()
        self.assertTrue(value.is_inference())
        hook._zeroize_tensors((value,))
        self.assertEqual(int(torch.count_nonzero(value)), 0)

    def test_world4_same_object_bit_exact_commit_and_zeroize(self) -> None:
        invocation, rank_bank, shards, identity, exact = capture_world4()
        self.assertTrue(all(identity))
        self.assertTrue(all(exact))
        self.assertEqual(rank_bank.receipt()["implicit_collective_calls"], 0)
        before_nonzero = sum(
            int(torch.count_nonzero(tensor))
            for row in shards
            for tensor in (
                row.query,
                row.key,
                row.derived_qk_role_responsibility_proxy,
            )
        )
        self.assertGreater(before_nonzero, 0)

        native_bank = native.InMemoryNativeCaptureBank()
        receipt = hook.commit_world4_shards_to_native_bank(
            native_bank=native_bank,
            invocation=invocation,
            rank_shards=shards,
        )
        self.assertFalse(receipt["backend_attention_weights_observed"])
        self.assertEqual(receipt["responsibility_kind"], hook.RESPONSIBILITY_KIND)
        self.assertEqual(receipt["collective_calls_inside_attention_added"], 0)
        self.assertFalse(receipt["gpu_launch_authorized"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertTrue(
            all(
                int(torch.count_nonzero(tensor)) == 0
                for row in shards
                for tensor in (
                    row.query,
                    row.key,
                    row.derived_qk_role_responsibility_proxy,
                )
            )
        )

        captures = native_bank.consume(invocation)
        self.assertEqual(tuple(value.block_index for value in captures), hook.BLOCKS)
        for value in captures:
            self.assertEqual(tuple(value.query.shape), (1, 21, 2, 12, 128))
            self.assertEqual(
                tuple(value.derived_qk_role_responsibility_proxy.shape),
                (1, 21, 3, 2),
            )
            mass = value.derived_qk_role_responsibility_proxy.sum(dim=2)
            self.assertTrue(torch.allclose(mass, torch.ones_like(mass)))
        native_bank.zeroize(captures)

    def test_backend_weights_are_not_claimed_and_old_receipt_name_is_absent(self) -> None:
        self.assertFalse(hook.BACKEND_ATTENTION_WEIGHTS_OBSERVED)
        model = SimpleNamespace(
            blocks=[
                SimpleNamespace(
                    attn1=FakeAttention(
                        FakeOfficialProcessor(
                            rank=0, block=index, layout=hook.World4RankLayout(0, 1, 2)
                        )
                    ),
                    attn2=FakeAttention(
                        FakeOfficialProcessor(
                            rank=0, block=index, layout=hook.World4RankLayout(0, 1, 2)
                        )
                    ),
                )
                for index in range(30)
            ]
        )
        rank_bank = hook.InMemoryWorld4RankShardBank()
        originals = {
            index: (model.blocks[index].attn1.processor, model.blocks[index].attn2.processor)
            for index in hook.BLOCKS
        }
        handle = hook.install_native_relational_attention_hook(model, rank_bank=rank_bank)
        receipt_text = str(handle.receipt())
        self.assertNotIn("attn2_role_responsibility", receipt_text)
        self.assertIn(hook.RESPONSIBILITY_KIND, receipt_text)
        self.assertFalse(handle.receipt()["backend_attention_weights_observed"])
        handle.restore()
        for index in hook.BLOCKS:
            self.assertIs(model.blocks[index].attn1.processor, originals[index][0])
            self.assertIs(model.blocks[index].attn2.processor, originals[index][1])

    def test_missing_world4_rank_fails_closed_and_zeroizes_shards(self) -> None:
        invocation, _rank_bank, shards, _identity, _exact = capture_world4()
        missing = tuple(row for row in shards if row.invocation.layout.rank != 3)
        native_bank = native.InMemoryNativeCaptureBank()
        with self.assertRaises(hook.NativeRelationalAttentionHookError):
            hook.commit_world4_shards_to_native_bank(
                native_bank=native_bank,
                invocation=invocation,
                rank_shards=missing,
            )
        self.assertTrue(
            all(
                int(torch.count_nonzero(tensor)) == 0
                for row in missing
                for tensor in (
                    row.query,
                    row.key,
                    row.derived_qk_role_responsibility_proxy,
                )
            )
        )
        self.assertEqual(native_bank.receipt()["capture_count"], 0)

    def test_external_tensor_collective_abi_reconstructs_and_consumes_payload(self) -> None:
        invocation, _rank_bank, shards, _identity, _exact = capture_world4()
        selected = sorted(
            (row for row in shards if row.block_index == 6),
            key=lambda row: row.invocation.layout.rank,
        )
        payloads = [row.collective_payload_and_zeroize() for row in selected]
        qk = torch.stack([item[0] for item in payloads], dim=0).contiguous()
        proxy = torch.stack([item[1] for item in payloads], dim=0).contiguous()
        metadata = [item[2] for item in payloads]
        rebuilt = hook.reconstruct_world4_block_from_collectives(
            invocation=invocation,
            role_partition=partition(),
            block_index=6,
            qk_rank_major=qk,
            proxy_rank_major=proxy,
            rank_metadata=metadata,
        )
        self.assertEqual(tuple(row.invocation.layout.rank for row in rebuilt), (0, 1, 2, 3))
        self.assertEqual(int(torch.count_nonzero(qk)), 0)
        self.assertEqual(int(torch.count_nonzero(proxy)), 0)
        self.assertGreater(
            sum(int(torch.count_nonzero(row.query)) for row in rebuilt), 0
        )
        for row in rebuilt:
            row.zeroize()
        for row in shards:
            row.zeroize()

    def test_wrong_cross_rank_cache_fails_before_official_call(self) -> None:
        layout = hook.World4RankLayout(3, 1, 2)
        invocation = hook.RankCaptureInvocation(base_invocation(), layout, partition())
        rank_bank = hook.InMemoryWorld4RankShardBank()
        base = FakeOfficialProcessor(rank=3, block=6, layout=layout)
        observer = hook.NativeAttn2DerivedRoleProxyObserver(
            base, block_index=6, rank_bank=rank_bank
        )
        hidden = torch.zeros((1, layout.padded_local_tokens, 8))
        kwargs = dict(cross_kwargs(layout))
        kwargs["cu_seqlens_q_cross_cache"] = torch.tensor(
            [0, layout.padded_local_tokens], dtype=torch.int32
        )
        with self.assertRaises(hook.NativeRelationalAttentionHookError):
            with rank_bank.observe(invocation):
                observer(SimpleNamespace(scale=hook.HEAD_DIM ** -0.5), hidden, **kwargs)
        self.assertEqual(base.base_calls, 0)
        self.assertEqual(rank_bank.receipt()["resident_rank_invocations"], 0)

    def test_role_partition_must_be_exhaustive_and_every_role_supported(self) -> None:
        with self.assertRaises(hook.NativeRelationalAttentionHookError):
            hook.ExhaustiveTextRolePartition(("agent", "object", "null"), (0, 0, 2))
        with self.assertRaises(hook.NativeRelationalAttentionHookError):
            hook.ExhaustiveTextRolePartition(("agent", "object"), (0, 2))


if __name__ == "__main__":
    unittest.main()
