from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import schedule_block_source_edge_ablation_v2 as edge  # noqa: E402


class _Identity:
    def __call__(self, value):
        return value


class _Attention:
    def __init__(self) -> None:
        self.to_out = (_Identity(), _Identity())


class _BaseProcessor:
    def __init__(self, q=None, k=None, v=None) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.sentinel = object()
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.sentinel

    def _project_qkv(self, *args, **kwargs):
        return self.q, self.k, self.v


class _ParallelState:
    ulysses_enabled = False


def _invocation(
    *,
    mode: str = "source-off",
    schedule: int = 16,
    registered: int = 16,
    band: str = "early",
    branch: str = "V_uncond",
    total: int = 6,
    target: int = 2,
) -> edge.EdgeInvocation:
    return edge.EdgeInvocation(
        edge_mode=mode,
        schedule_index=schedule,
        registered_schedule_index=registered,
        band_name=band,
        branch_name=branch,
        total_tokens=total,
        target_tokens=target,
        ulysses_rank=0,
        ulysses_size=1,
    )


class SourceEdgeContractTest(unittest.TestCase):
    def test_grid_is_exact_registered_208_video_plan(self) -> None:
        value = edge.decoded_grid_contract()
        self.assertEqual(value["schedule_block_cell_count"], 16)
        self.assertEqual(value["outputs_per_family"], 104)
        self.assertEqual(value["total_decoded_outputs"], 208)
        self.assertFalse(value["scalar_score_or_reward"])
        self.assertFalse(value["automatic_ranking_or_selection"])

    def test_only_registered_coordinate_activates(self) -> None:
        active = _invocation()
        self.assertTrue(active.block_active(0))
        self.assertTrue(active.block_active(7))
        self.assertFalse(active.block_active(8))
        self.assertFalse(_invocation(schedule=17).block_active(0))
        self.assertFalse(_invocation(mode="source-on").block_active(0))
        self.assertFalse(
            _invocation(branch="none_uncond", total=2, target=2).block_active(0)
        )

    def test_source_on_delegates_exact_official_processor(self) -> None:
        base = _BaseProcessor()
        processor = edge.TargetQuerySourceEdgeProcessor(base, block_index=0)
        hidden = object()
        # A tiny shape-bearing shim is enough because the parity route delegates.
        hidden = mock.Mock(ndim=3)
        hidden.shape = (1, 3, 4)
        with edge.activate_edge(_invocation(mode="source-on")):
            result = processor(
                _Attention(),
                hidden,
                rotary_emb="rope",
                batch_image_vae_seqlen=[6],
                text_features_length=512,
                origin_hidden_states_seq_len=6,
                split_hidden_states_seq_len=6,
                cu_seqlens_q_cache=[0, 6],
                max_seqlen_q_cache=6,
            )
        self.assertIs(result, base.sentinel)
        self.assertEqual(len(base.calls), 1)
        self.assertEqual(processor.active_source_on_calls, 1)
        self.assertEqual(processor.active_edge_deletion_calls, 0)
        self.assertEqual(base.calls[0][1]["text_features_length"], 512)

    def test_active_path_keeps_native_source_rows_and_restricts_target_kv(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        q = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[-1.0, 0.0]],
              [[0.5, 0.5]], [[0.5, -0.5]]]],
            dtype=torch.float32,
        ).reshape(1, 6, 1, 2)
        k = torch.tensor(
            [[[[3.0, 0.0]], [[0.0, 3.0]], [[2.0, 2.0]], [[-2.0, 0.0]],
              [[1.0, 1.0]], [[1.0, -1.0]]]],
            dtype=torch.float32,
        ).reshape(1, 6, 1, 2)
        v = torch.arange(12, dtype=torch.float32).reshape(1, 6, 1, 2)
        base = _BaseProcessor(q, k, v)
        calls = []

        def varlen(query, key, value, **kwargs):
            calls.append((query.clone(), key.clone(), value.clone(), dict(kwargs)))
            logits = torch.einsum("qhd,khd->hqk", query, key)
            weights = torch.softmax(logits, dim=-1)
            return torch.einsum("hqk,khd->qhd", weights, value)

        processor = edge.TargetQuerySourceEdgeProcessor(
            base,
            block_index=0,
            varlen_attention_fn=varlen,
            get_parallel_state_fn=lambda: _ParallelState(),
            gather_heads_scatter_seq_fn=lambda value, **_: value,
        )
        hidden = torch.zeros((1, 6, 2), dtype=torch.float32)
        cu = torch.tensor([0, 6], dtype=torch.int32)
        with (
            mock.patch.object(edge.replay, "require_rotary_embedding"),
            mock.patch.object(edge.replay, "validate_projected_rotary_embedding"),
            edge.activate_edge(_invocation()),
            torch.inference_mode(),
        ):
            result = processor(
                _Attention(),
                hidden,
                rotary_emb=torch.zeros((1, 6, 1, 1)),
                batch_image_vae_seqlen=[6],
                origin_hidden_states_seq_len=6,
                cu_seqlens_q_cache=cu,
                max_seqlen_q_cache=6,
            )
        self.assertEqual(len(calls), 2)
        native = calls[0]
        restricted = calls[1]
        self.assertEqual(tuple(native[0].shape), (6, 1, 2))
        self.assertEqual(tuple(restricted[0].shape), (2, 1, 2))
        self.assertTrue(torch.equal(restricted[1], k.squeeze(0)[4:]))
        expected = torch.cat((
            varlen(q.squeeze(0), k.squeeze(0), v.squeeze(0))[:4],
            varlen(q.squeeze(0)[4:], k.squeeze(0)[4:], v.squeeze(0)[4:]),
        ), dim=0).reshape(1, 6, 2)
        self.assertTrue(torch.allclose(result, expected))
        self.assertEqual(processor.active_edge_deletion_calls, 1)
        self.assertEqual(processor.official_delegate_calls, 0)
        self.assertTrue(
            processor.last_active_geometry[
                "target_query_rows_from_target_KV_only_attention"
            ]
        )

    def test_inactive_schedule_and_unselected_block_delegate(self) -> None:
        for processor, invocation in (
            (
                edge.TargetQuerySourceEdgeProcessor(_BaseProcessor(), block_index=0),
                _invocation(schedule=17),
            ),
            (
                edge.TargetQuerySourceEdgeProcessor(_BaseProcessor(), block_index=8),
                _invocation(),
            ),
        ):
            hidden = mock.Mock(ndim=3)
            hidden.shape = (1, 3, 4)
            with edge.activate_edge(invocation):
                result = processor(
                    _Attention(), hidden, rotary_emb="rope",
                    batch_image_vae_seqlen=[6], cu_seqlens_q_cache=[0, 6],
                    max_seqlen_q_cache=6,
                )
            self.assertIs(result, processor.base_processor.sentinel)
            self.assertEqual(processor.official_delegate_calls, 1)
            self.assertEqual(processor.active_edge_deletion_calls, 0)

    def test_nested_or_missing_invocation_fails_closed(self) -> None:
        with self.assertRaises(edge.SourceEdgeAblationError):
            edge.current_edge_invocation()
        with edge.activate_edge(_invocation()):
            with self.assertRaises(edge.SourceEdgeAblationError):
                with edge.activate_edge(_invocation()):
                    pass


if __name__ == "__main__":
    unittest.main()
