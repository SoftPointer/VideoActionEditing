from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import anonymous_visual_projection_hook_v6 as hook


class AnonymousVisualProjectionHookV6Test(unittest.TestCase):
    def _identity(self, arm="action"):
        return hook.AnonymousCaptureIdentityV6(
            "appearance_0",
            arm,
            "high",
            18,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            37,
            25,
        )

    def test_fixed_projection_is_deterministic_and_orthogonal(self):
        left = hook.ProjectionAuthorityV6.create()
        right = hook.ProjectionAuthorityV6.create()
        left.validate()
        self.assertEqual(left.digest, right.digest)
        self.assertTrue(torch.equal(left.query, right.query))
        self.assertTrue(torch.equal(left.hidden, right.hidden))
        gram = left.query.double().T @ left.query.double()
        self.assertTrue(torch.allclose(gram, torch.eye(16, dtype=torch.float64), atol=2e-6, rtol=2e-6))

    def test_projection_zeroizes_owned_raw(self):
        raw = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).clone()
        projection = torch.eye(4, dtype=torch.float32)[:, :2].contiguous()
        result = hook.project_owned_raw_and_zeroize(raw, projection, label="unit")
        self.assertEqual(tuple(result.shape), (2, 3, 2))
        self.assertEqual(torch.count_nonzero(raw).item(), 0)
        self.assertGreater(torch.count_nonzero(result).item(), 0)

    def test_projection_failure_also_zeroizes_raw(self):
        raw = torch.ones((2, 4), dtype=torch.float32)
        with self.assertRaises(hook.AnonymousVisualProjectionHookV6Error):
            hook.project_owned_raw_and_zeroize(
                raw, torch.ones((3, 2), dtype=torch.float32), label="bad"
            )
        self.assertEqual(torch.count_nonzero(raw).item(), 0)

    def test_rank_layout_covers_global_sequence_once(self):
        layouts = [hook.World4VisualLayoutV6(rank) for rank in range(4)]
        covered = []
        for row in layouts:
            covered.extend(range(row.global_start, row.global_stop))
        self.assertEqual(covered, list(range(layouts[0].global_tokens)))

    def test_reconstruct_sums_query_and_concatenates_hidden(self):
        identity = self._identity()
        authority = hook.ProjectionAuthorityV6.create()
        layout = hook.World4VisualLayoutV6(0)
        query = torch.stack(
            [
                torch.full(
                    (1, layout.global_tokens, hook.QUERY_SKETCH_DIM),
                    float(rank + 1),
                )
                for rank in range(4)
            ],
            dim=0,
        ).contiguous()
        hidden = torch.stack(
            [
                torch.full(
                    (1, layout.padded_local_tokens, hook.HIDDEN_SKETCH_DIM),
                    float(rank + 1),
                )
                for rank in range(4)
            ],
            dim=0,
        ).contiguous()
        query_ref = query
        hidden_ref = hidden
        metadata = []
        for rank in range(4):
            rank_layout = hook.World4VisualLayoutV6(rank)
            metadata.append(
                {
                    "rank": rank,
                    "block_index": 6,
                    "identity_key": list(identity.key),
                    "projection_digest": authority.digest,
                    "valid_local_tokens": rank_layout.valid_local_tokens,
                }
            )
        capture = hook.reconstruct_projected_world4_block_v6(
            identity=identity,
            block_index=6,
            projection_digest=authority.digest,
            query_rank_major=query,
            hidden_rank_major=hidden,
            rank_metadata=metadata,
        )
        capture.validate()
        self.assertTrue(torch.all(capture.query_sketch == 10.0))
        flat_hidden = capture.hidden_sketch.reshape(1, -1, hook.HIDDEN_SKETCH_DIM)
        for rank in range(4):
            row = hook.World4VisualLayoutV6(rank)
            self.assertTrue(
                torch.all(flat_hidden[:, row.global_start : row.global_stop] == float(rank + 1))
            )
        self.assertEqual(torch.count_nonzero(query_ref).item(), 0)
        self.assertEqual(torch.count_nonzero(hidden_ref).item(), 0)
        capture.zeroize()

    def test_rank_bank_aborts_partial_projected_ownership(self):
        authority = hook.ProjectionAuthorityV6.create()
        identity = self._identity()
        invocation = hook.AnonymousRankInvocationV6(
            identity, hook.World4VisualLayoutV6(0), authority.digest
        )
        bank = hook.InMemoryProjectedRankBankV6()
        layout = invocation.layout
        query = torch.ones((1, layout.global_tokens, 16), dtype=torch.float32)
        hidden = torch.ones((1, layout.padded_local_tokens, 16), dtype=torch.float32)
        with self.assertRaises(hook.AnonymousVisualProjectionHookV6Error):
            with bank.observe(invocation):
                bank.capture(
                    hook.ProjectedVisualRankShardV6(invocation, 6, query, hidden)
                )
        self.assertEqual(torch.count_nonzero(query).item(), 0)
        self.assertEqual(torch.count_nonzero(hidden).item(), 0)
        self.assertEqual(bank.receipt()["resident_rank_invocation_count"], 0)

    def test_hook_contract_has_no_text_role_path(self):
        source = Path(hook.__file__).read_text(encoding="utf-8")
        self.assertNotIn("token_to_role", source)
        self.assertNotIn("role_scores", source)
        self.assertNotIn("attn2", source.lower())

    def _fake_processor_type(self, *, fail=False):
        class WanAttnProcessor2_0:
            calls = 0

            def _project_qkv(self, attn, hidden_states, **kwargs):
                tokens = int(kwargs["origin_hidden_states_seq_len"])
                query = torch.arange(
                    tokens * hook.LOCAL_HEADS * hook.HEAD_DIM,
                    dtype=torch.float32,
                ).reshape(1, tokens, hook.LOCAL_HEADS, hook.HEAD_DIM)
                # The fake applies its rotary marker before returning Q.  The
                # wrapper must see these post-marker values.
                query = query + float(kwargs["rotary_emb"].reshape(-1)[0])
                self.post_rope_query = query
                return query, torch.zeros_like(query), torch.zeros_like(query)

            def __call__(self, attn, hidden_states, **kwargs):
                type(self).calls += 1
                self._project_qkv(attn, hidden_states, **kwargs)
                if fail:
                    raise RuntimeError("injected official failure")
                self.output = hidden_states + 7.0
                return self.output

        WanAttnProcessor2_0.__module__ = "bernini.models.transformer_wan"
        return WanAttnProcessor2_0

    def _small_invocation(self, authority):
        identity = hook.AnonymousCaptureIdentityV6(
            "appearance_0", "action", "high", 18, "a" * 64, "b" * 64, "c" * 64, 2, 2
        )
        return hook.AnonymousRankInvocationV6(
            identity, hook.World4VisualLayoutV6(0, 2, 2), authority.digest
        )

    def _official_kwargs(self, layout):
        return {
            "encoder_hidden_states": None,
            "attention_mask": None,
            "rotary_emb": torch.tensor([3.0]),
            "batch_image_vae_seqlen": [layout.global_tokens],
            "text_features_length": None,
            "origin_hidden_states_seq_len": layout.global_tokens,
            "split_hidden_states_seq_len": layout.padded_local_tokens,
            "cu_seqlens_q_cache": torch.tensor([0, layout.global_tokens]),
            "max_seqlen_q_cache": layout.global_tokens,
            "cu_seqlens_k_cross_cache": None,
            "cu_seqlens_q_cross_cache": None,
            "max_seqlen_k_cross_cache": None,
            "max_seqlen_q_cross_cache": None,
        }

    def test_official_delegate_once_post_rope_hidden_and_same_output_object(self):
        with mock.patch.multiple(
            hook.registry, PHASES=2, PATCH_HEIGHT=2, PATCH_WIDTH=2, PATCHES=4
        ):
            authority = hook.ProjectionAuthorityV6.create()
            invocation = self._small_invocation(authority)
            bank = hook.InMemoryProjectedRankBankV6()
            processor_type = self._fake_processor_type()
            processor_type.calls = 0
            processor = processor_type()
            with mock.patch.object(
                hook.inspect,
                "getsourcefile",
                return_value=str(Path(__file__).resolve()),
            ), mock.patch.object(
                hook,
                "validate_official_transformer_source_file_v6",
                return_value=Path(__file__).resolve(),
            ):
                wrapper = hook.AnonymousAttn1ProjectionObserverV6(
                    processor,
                    block_index=6,
                    rank_bank=bank,
                    projection=authority,
                )
            hidden = torch.arange(
                invocation.layout.padded_local_tokens * hook.MODEL_WIDTH,
                dtype=torch.float32,
            ).reshape(1, invocation.layout.padded_local_tokens, hook.MODEL_WIDTH)
            with bank.observe(invocation):
                output = wrapper(torch.nn.Identity().eval(), hidden, **self._official_kwargs(invocation.layout))
                # Fill the other registered blocks through direct projected
                # shards; this test targets the block-6 official ABI.
                for block in (12, 18, 24):
                    bank.capture(
                        hook.ProjectedVisualRankShardV6(
                            invocation,
                            block,
                            torch.zeros((1, invocation.layout.global_tokens, 16)),
                            torch.zeros((1, invocation.layout.padded_local_tokens, 16)),
                        )
                    )
            self.assertIs(output, processor.output)
            self.assertTrue(torch.equal(output, hidden + 7.0))
            self.assertEqual(processor_type.calls, 1)
            shards = bank.take_rank(invocation)
            shard = shards[0]
            expected_q = processor.post_rope_query.reshape(
                1, invocation.layout.global_tokens, -1
            ).float() @ authority.query_rank_slice(0, torch.device("cpu"))
            expected_h = processor.output.float() @ authority.hidden_matrix(torch.device("cpu"))
            self.assertTrue(torch.allclose(shard.query_partial, expected_q))
            self.assertTrue(torch.allclose(shard.hidden_local, expected_h))
            for row in shards:
                row.zeroize()

    def test_official_exception_scrubs_observer_query_clone(self):
        with mock.patch.multiple(
            hook.registry, PHASES=2, PATCH_HEIGHT=2, PATCH_WIDTH=2, PATCHES=4
        ):
            authority = hook.ProjectionAuthorityV6.create()
            invocation = self._small_invocation(authority)
            bank = hook.InMemoryProjectedRankBankV6()
            processor = self._fake_processor_type(fail=True)()
            with mock.patch.object(
                hook.inspect,
                "getsourcefile",
                return_value=str(Path(__file__).resolve()),
            ), mock.patch.object(
                hook,
                "validate_official_transformer_source_file_v6",
                return_value=Path(__file__).resolve(),
            ):
                wrapper = hook.AnonymousAttn1ProjectionObserverV6(
                    processor, block_index=6, rank_bank=bank, projection=authority
                )
            hidden = torch.ones(
                (1, invocation.layout.padded_local_tokens, hook.MODEL_WIDTH),
                dtype=torch.float32,
            )
            scrubbed = []
            original_zeroize = hook._zeroize

            def recording_zeroize(values):
                rows = tuple(values)
                original_zeroize(rows)
                scrubbed.extend(
                    int(torch.count_nonzero(value).item())
                    for value in rows
                    if isinstance(value, torch.Tensor)
                )

            with mock.patch.object(hook, "_zeroize", side_effect=recording_zeroize):
                with self.assertRaises(RuntimeError):
                    with bank.observe(invocation):
                        wrapper(
                            torch.nn.Identity().eval(),
                            hidden,
                            **self._official_kwargs(invocation.layout),
                        )
            self.assertTrue(scrubbed)
            self.assertTrue(all(value == 0 for value in scrubbed))
            self.assertEqual(bank.receipt()["resident_rank_invocation_count"], 0)

    def test_install_and_restore_four_attn1_processors(self):
        processor_type = self._fake_processor_type()

        class Attention:
            def __init__(self):
                self.processor = processor_type()

            def set_processor(self, value):
                self.processor = value

        class Block:
            def __init__(self):
                self.attn1 = Attention()

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = [Block() for _ in range(30)]

        transformer = Transformer().eval()
        originals = [transformer.blocks[index].attn1.processor for index in hook.BLOCKS]
        with mock.patch.object(
            hook.inspect,
            "getsourcefile",
            return_value=str(Path(__file__).resolve()),
        ), mock.patch.object(
            hook,
            "validate_official_transformer_source_file_v6",
            return_value=Path(__file__).resolve(),
        ):
            handle = hook.install_anonymous_visual_projection_hook_v6(
                transformer, rank_bank=hook.InMemoryProjectedRankBankV6()
            )
        for index, wrapper in zip(hook.BLOCKS, handle.wrappers):
            self.assertIs(transformer.blocks[index].attn1.processor, wrapper)
        handle.restore()
        for index, original in zip(hook.BLOCKS, originals):
            self.assertIs(transformer.blocks[index].attn1.processor, original)

    def test_same_module_and_class_from_wrong_source_hash_is_rejected(self):
        processor = self._fake_processor_type()()
        with mock.patch.object(
            hook.inspect,
            "getsourcefile",
            return_value=str(Path(__file__).resolve()),
        ):
            with self.assertRaisesRegex(
                hook.AnonymousVisualProjectionHookV6Error,
                "source SHA-256 differs",
            ):
                hook.AnonymousAttn1ProjectionObserverV6(
                    processor,
                    block_index=6,
                    rank_bank=hook.InMemoryProjectedRankBankV6(),
                    projection=hook.ProjectionAuthorityV6.create(),
                )


if __name__ == "__main__":
    unittest.main()
