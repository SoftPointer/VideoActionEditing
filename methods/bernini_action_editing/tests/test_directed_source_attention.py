from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import directed_source_attention as directed  # noqa: E402


class LayoutContractTests(unittest.TestCase):
    def test_block_scopes_are_exact_thirds_for_1p3b(self) -> None:
        self.assertEqual(directed.resolve_block_indices(30, "all"), tuple(range(30)))
        self.assertEqual(directed.resolve_block_indices(30, "mid"), tuple(range(10, 20)))
        self.assertEqual(directed.resolve_block_indices(30, "late"), tuple(range(20, 30)))
        for selection in ("early", "10-20", ""):
            with self.subTest(selection=selection), self.assertRaises(
                directed.DirectedAttentionContractError
            ):
                directed.resolve_block_indices(30, selection)

    def test_contract_has_no_external_spatial_or_target_condition(self) -> None:
        contract = directed.oracle_contract(selection="all")
        self.assertEqual(contract["trained_parameters"], 0)
        self.assertEqual(contract["batch_size"], 1)
        self.assertIs(contract["equal_source_target_token_lengths"], True)
        self.assertEqual(contract["source_query_keys_values"], ["source"])
        self.assertEqual(contract["target_query_keys_values"], ["source", "target"])
        self.assertIs(contract["ordinary_attention_fallback"], False)
        self.assertEqual(contract["external_conditions"], ["source_video", "edit_instruction"])
        self.assertIn("mask", contract["forbidden_external_conditions"])
        self.assertIn("target_video", contract["forbidden_external_conditions"])
        self.assertEqual(len(contract["contract_digest"]), 64)

    def test_layout_requires_one_even_equal_pair(self) -> None:
        boundary = directed.validate_equal_pair_layout(
            gathered_sequence_length=12,
            batch_image_vae_seqlen=[12],
            cu_seqlens_q_cache=[0, 12],
            max_seqlen_q_cache=12,
            origin_hidden_states_seq_len=12,
        )
        self.assertEqual(boundary, 6)
        invalid = (
            {"gathered_sequence_length": 11},
            {"batch_image_vae_seqlen": [6, 6]},
            {"batch_image_vae_seqlen": [10]},
            {"cu_seqlens_q_cache": [0, 6, 12]},
            {"max_seqlen_q_cache": 11},
            {"origin_hidden_states_seq_len": 16},
        )
        base = {
            "gathered_sequence_length": 12,
            "batch_image_vae_seqlen": [12],
            "cu_seqlens_q_cache": [0, 12],
            "max_seqlen_q_cache": 12,
            "origin_hidden_states_seq_len": 12,
        }
        for change in invalid:
            values = dict(base)
            values.update(change)
            with self.subTest(change=change), self.assertRaises(
                directed.DirectedAttentionContractError
            ):
                directed.validate_equal_pair_layout(**values)


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
        self.blocks = [_MockBlock(object()) for _ in range(30)]

    def patch_vae_latent(self):
        raise AssertionError("not called")


class _MockDiffusion:
    def __init__(self):
        self.transformer = _MockTransformer()


class _MockRenderer:
    def __init__(self):
        self.diff_dec = _MockDiffusion()


class InstallRestoreTests(unittest.TestCase):
    def test_monkey_patch_changes_only_selected_attn1_and_restores_identity(self) -> None:
        renderer = _MockRenderer()
        transformer = renderer.diff_dec.transformer
        originals = [block.attn1.processor for block in transformer.blocks]

        class FakeBase:
            def _project_qkv(self):
                raise AssertionError("not called")

        # The production factory wraps each real prior processor.  This custom
        # factory lets the structure test use lightweight official-like mocks.
        def factory(_prior, index):
            return directed.DirectedSourceSelfAttnProcessor(
                FakeBase(), block_index=index
            )

        handle = directed.install_directed_source_attention(
            renderer, selection="mid", processor_factory=factory
        )
        self.assertEqual(handle.indices, tuple(range(10, 20)))
        for index, block in enumerate(transformer.blocks):
            if 10 <= index < 20:
                self.assertIsInstance(
                    block.attn1.processor, directed.DirectedSourceSelfAttnProcessor
                )
            else:
                self.assertIs(block.attn1.processor, originals[index])
        receipt = handle.receipt()
        self.assertEqual(receipt["runtime"]["installed_block_count"], 10)
        handle.restore()
        self.assertTrue(handle.restored)
        for index, block in enumerate(transformer.blocks):
            self.assertIs(block.attn1.processor, originals[index])


class TensorVisibilityTests(unittest.TestCase):
    def _torch(self):
        try:
            import torch
        except Exception as error:  # pragma: no cover - lightweight local environment
            self.skipTest(f"torch unavailable: {error}")
        return torch

    def test_source_output_is_invariant_to_arbitrary_target_perturbation(self) -> None:
        torch = self._torch()

        class BaseProcessor:
            def _project_qkv(
                self, attn, hidden_states, encoder_hidden_states, rotary_emb,
                origin_hidden_states_seq_len, is_cross_attn,
            ):
                self.projected_full_length = int(hidden_states.shape[1])
                return tuple(hidden_states.unsqueeze(2) for _ in range(3))

        class ParallelState:
            ulysses_enabled = False

        def reference_varlen(
            q, k, v, *, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, causal,
        ):
            self.assertFalse(causal)
            self.assertEqual(cu_seqlens_q.tolist()[0], 0)
            self.assertEqual(cu_seqlens_k.tolist()[0], 0)
            q4 = q.transpose(0, 1).unsqueeze(0)
            k4 = k.transpose(0, 1).unsqueeze(0)
            v4 = v.transpose(0, 1).unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4)
            return out.squeeze(0).transpose(0, 1)

        class Attention:
            to_out = (torch.nn.Identity(), torch.nn.Identity())

        base = BaseProcessor()
        processor = directed.DirectedSourceSelfAttnProcessor(
            base,
            block_index=0,
            varlen_attention_fn=reference_varlen,
            get_parallel_state_fn=lambda: ParallelState(),
            gather_heads_scatter_seq_fn=lambda value, **kwargs: value,
        )
        source = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]], dtype=torch.float32
        )
        target_a = torch.zeros_like(source)
        target_b = torch.tensor(
            [[[100.0, -70.0], [-80.0, 120.0], [60.0, 90.0]]],
            dtype=torch.float32,
        )

        def run(target):
            hidden = torch.cat((source, target), dim=1)
            return processor(
                Attention(),
                hidden,
                batch_image_vae_seqlen=[6],
                cu_seqlens_q_cache=torch.tensor([0, 6], dtype=torch.int32),
                max_seqlen_q_cache=torch.tensor(6),
            )

        output_a = run(target_a)
        output_b = run(target_b)
        self.assertEqual(base.projected_full_length, 6)
        self.assertTrue(torch.equal(output_a[:, :3], output_b[:, :3]))
        self.assertFalse(torch.equal(output_a[:, 3:], output_b[:, 3:]))
        self.assertEqual(processor.call_count, 2)
        self.assertEqual(processor.full_sequence_length, 6)

    def test_processor_rejects_cross_attention_mask_and_odd_sequence(self) -> None:
        torch = self._torch()

        class BaseProcessor:
            def _project_qkv(self, attn, hidden, *args):
                return tuple(hidden.unsqueeze(2) for _ in range(3))

        processor = directed.DirectedSourceSelfAttnProcessor(
            BaseProcessor(), block_index=0
        )
        hidden = torch.zeros((1, 5, 2))
        common = {
            "batch_image_vae_seqlen": [5],
            "cu_seqlens_q_cache": torch.tensor([0, 5], dtype=torch.int32),
            "max_seqlen_q_cache": torch.tensor(5),
        }
        with self.assertRaises(directed.DirectedAttentionContractError):
            processor(object(), hidden, **common)
        with self.assertRaises(directed.DirectedAttentionContractError):
            processor(
                object(),
                torch.zeros((1, 6, 2)),
                encoder_hidden_states=torch.zeros((1, 2, 2)),
                batch_image_vae_seqlen=[6],
                cu_seqlens_q_cache=torch.tensor([0, 6], dtype=torch.int32),
                max_seqlen_q_cache=torch.tensor(6),
            )
        with self.assertRaises(directed.DirectedAttentionContractError):
            processor(
                object(),
                torch.zeros((1, 6, 2)),
                attention_mask=torch.ones((6, 6)),
                batch_image_vae_seqlen=[6],
                cu_seqlens_q_cache=torch.tensor([0, 6], dtype=torch.int32),
                max_seqlen_q_cache=torch.tensor(6),
            )


if __name__ == "__main__":
    unittest.main()

