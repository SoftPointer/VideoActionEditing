from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import inspect
import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_same_state_noop_value_router as router


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SameStateRouterStaticContractTests(unittest.TestCase):
    def test_main_arm_is_paired_source_kv_not_value_only(self) -> None:
        receipt = router.same_state_noop_value_router_contract(
            selected_block_indices=(0, 1),
            arm=router.HYBRID_SOURCE_NOOP_KV,
            gate=1.0,
        )
        self.assertEqual(receipt["main_arm"], router.HYBRID_SOURCE_NOOP_KV)
        main = receipt["arms"][router.HYBRID_SOURCE_NOOP_KV]
        self.assertEqual(main["key"], "[K_noop_source,K_action_target]")
        self.assertEqual(main["value"], "[V_noop_source,V_action_target]")
        self.assertTrue(main["noop_source_key_value_from_same_projection"])
        self.assertTrue(main["action_target_key_kept_byte_exact"])
        self.assertTrue(main["action_target_value_kept_byte_exact"])
        self.assertTrue(
            receipt["action_branch_invariants"][
                "main_routes_paired_source_key_and_value"
            ]
        )

    def test_full_pair_noop_value_is_diagnostic_only(self) -> None:
        receipt = router.same_state_noop_value_router_contract(
            selected_block_indices=(0,),
            arm=router.FULL_PAIR_NOOP_VALUE_DIAGNOSTIC,
            gate=0.5,
        )
        diagnostic = receipt["arms"][
            router.FULL_PAIR_NOOP_VALUE_DIAGNOSTIC
        ]
        self.assertFalse(receipt["selected_arm_is_main"])
        self.assertEqual(diagnostic["key"], "K_action_full_pair")
        self.assertEqual(diagnostic["value"], "V_noop_full_pair")
        self.assertFalse(diagnostic["eligible_as_main_claim"])
        self.assertIn("action_suppression", diagnostic["role"])

    def test_receipt_disclaims_per_layer_same_state_and_names_failures(
        self,
    ) -> None:
        receipt = router.same_state_noop_value_router_contract(
            selected_block_indices=(0,),
            gate=0.75,
        )
        outer = receipt["outer_state"]
        self.assertTrue(
            outer["capture_and_action_share_exact_current_[source,target]_x_t"]
        )
        self.assertFalse(
            outer["same_outer_x_t_implies_same_per_layer_hidden"]
        )
        self.assertFalse(outer["per_layer_same_state_claimed"])
        self.assertEqual(
            outer["scientific_description"],
            "controlled_cross_branch_factorization_oracle",
        )
        differences = receipt["difference_from_failed_v9_v10"]
        self.assertIn("source_only", differences["V9"])
        self.assertIn("source_only_V_residual", differences["V10"])
        limitations = " ".join(receipt["fatal_limitations"])
        self.assertIn("off_manifold", limitations)
        self.assertIn("checkpoint", limitations)
        self.assertIn("suppress", limitations)

    def test_cache_lifecycle_ulysses_and_external_inputs_are_closed(self) -> None:
        receipt = router.same_state_noop_value_router_contract(
            selected_block_indices=(0, 2),
            gate=1.0,
        )
        lifecycle = receipt["cache_lifecycle"]
        self.assertFalse(lifecycle["cross_step_or_cross_trajectory_reuse"])
        self.assertFalse(lifecycle["equal_but_distinct_RoPE_tensor_allowed"])
        self.assertIn("block_index", lifecycle["identity"])
        self.assertEqual(receipt["ulysses"]["required_size"], 4)
        self.assertEqual(
            receipt["external_inputs_training_and_inference"],
            ["source_video", "edit_instruction"],
        )
        self.assertIn("target_video", receipt["forbidden_external_inputs"])
        self.assertIn("mask", receipt["forbidden_external_inputs"])
        self.assertFalse(receipt["optimizer_updates_authorized"])

    def test_zero_gate_contract_is_exact_delegate_without_cache(self) -> None:
        receipt = router.same_state_noop_value_router_contract(
            selected_block_indices=(0,),
            gate=0.0,
        )
        self.assertEqual(
            receipt["zero_gate"]["attn1"],
            "exact_untouched_official_processor_delegate",
        )
        self.assertEqual(
            receipt["zero_gate"]["block_boundary_hook"],
            "returns_None_without_tensor_write",
        )
        self.assertFalse(receipt["zero_gate"]["cache_required"])

    def test_module_imports_torch_lazily_and_api_has_no_oracle_input(self) -> None:
        source = Path(router.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        eager = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager.extend(
                    alias.name for alias in node.names if alias.name == "torch"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "torch":
                eager.append(node.module)
        self.assertEqual(eager, [])

        for function in (
            router.install_same_state_noop_value_router,
            router.same_state_noop_value_router,
            router.same_state_noop_value_invocation,
        ):
            names = set(inspect.signature(function).parameters)
            for forbidden in (
                "target_video",
                "proposal",
                "mask",
                "flow",
                "pose",
                "track",
            ):
                self.assertNotIn(forbidden, names)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class SameStateRouterTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

        class ParallelState:
            def __init__(self, rank: int, size: int = 4) -> None:
                self.ulysses_enabled = True
                self.ulysses_rank = rank
                self.ulysses_size = size

        class VarlenSpy:
            def __init__(self) -> None:
                self.calls = []

            def __call__(
                self,
                query,
                key,
                value,
                *,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal,
            ):
                self.calls.append(
                    {
                        "query": query.detach().clone(),
                        "key": key.detach().clone(),
                        "value": value.detach().clone(),
                        "cu_q": tuple(int(item) for item in cu_seqlens_q),
                        "cu_k": tuple(int(item) for item in cu_seqlens_k),
                        "max_q": int(max_seqlen_q),
                        "max_k": int(max_seqlen_k),
                        "causal": causal,
                    }
                )
                scores = torch.einsum("qhd,khd->hqk", query, key)
                scores = scores / math.sqrt(float(query.shape[-1]))
                weights = torch.softmax(scores, dim=-1)
                return torch.einsum("hqk,khd->qhd", weights, value)

        class FakeAttn:
            def __init__(self, processor) -> None:
                self.processor = processor
                self.to_out = [torch.nn.Identity(), torch.nn.Identity()]

            def set_processor(self, processor) -> None:
                self.processor = processor

        class BaseProcessor:
            def __init__(
                self,
                spy: VarlenSpy,
                inverse,
                *,
                gather_factor: int = 4,
            ) -> None:
                self.spy = spy
                self.inverse = inverse
                self.gather_factor = gather_factor
                self.delegate_calls = 0
                self.project_calls = 0
                self.project_records = []
                self.return_override = None

            def _project_qkv(
                self,
                attn,
                hidden_states,
                encoder_hidden_states,
                rotary_emb,
                origin_hidden_states_seq_len,
                is_cross,
            ):
                del (
                    attn,
                    encoder_hidden_states,
                    rotary_emb,
                    origin_hidden_states_seq_len,
                    is_cross,
                )
                self.project_calls += 1
                full = hidden_states.repeat(1, self.gather_factor, 1)
                query = full[..., 0:2].reshape(1, full.shape[1], 1, 2)
                key = full[..., 2:4].reshape(1, full.shape[1], 1, 2)
                value = full[..., 4:6].reshape(1, full.shape[1], 1, 2)
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()
                self.project_records.append(
                    {
                        "query": query.detach().clone(),
                        "key": key.detach().clone(),
                        "value": value.detach().clone(),
                    }
                )
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
                    encoder_hidden_states,
                    attention_mask,
                    batch_image_vae_seqlen,
                    text_features_length,
                    split_hidden_states_seq_len,
                    cu_seqlens_k_cross_cache,
                    cu_seqlens_q_cross_cache,
                    max_seqlen_k_cross_cache,
                    max_seqlen_q_cross_cache,
                )
                self.delegate_calls += 1
                if self.return_override is not None:
                    return self.return_override
                query, key, value = self._project_qkv(
                    attn,
                    hidden_states,
                    None,
                    rotary_emb,
                    origin_hidden_states_seq_len,
                    False,
                )
                output = self.spy(
                    query.squeeze(0),
                    key.squeeze(0),
                    value.squeeze(0),
                    cu_seqlens_q=cu_seqlens_q_cache,
                    cu_seqlens_k=cu_seqlens_q_cache,
                    max_seqlen_q=max_seqlen_q_cache,
                    max_seqlen_k=max_seqlen_q_cache,
                    causal=False,
                )
                output = self.inverse(
                    output.unsqueeze(0), head_dim=2, seq_dim=1
                )
                output = output.flatten(2, 3).contiguous().type_as(query)
                return attn.to_out[1](attn.to_out[0](output))

        class FakeBlock(torch.nn.Module):
            def __init__(self, base: BaseProcessor) -> None:
                super().__init__()
                self.attn1 = FakeAttn(base)

            def forward(self, hidden_states, rotary_emb, metadata):
                attention = self.attn1.processor(
                    self.attn1,
                    hidden_states,
                    rotary_emb=rotary_emb,
                    batch_image_vae_seqlen=metadata["lengths"],
                    origin_hidden_states_seq_len=metadata["origin"],
                    split_hidden_states_seq_len=metadata["split"],
                    cu_seqlens_q_cache=metadata["cu"],
                    max_seqlen_q_cache=metadata["max"],
                )
                return attention + 0.125 * hidden_states

        class FakeTransformer(torch.nn.Module):
            def __init__(self, bases) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [FakeBlock(base) for base in bases]
                )

            def forward(self, hidden_states, rotary_emb, metadata):
                output = hidden_states
                for block in self.blocks:
                    output = block(output, rotary_emb, metadata)
                return output

        cls.ParallelState = ParallelState
        cls.VarlenSpy = VarlenSpy
        cls.BaseProcessor = BaseProcessor
        cls.FakeAttn = FakeAttn
        cls.FakeBlock = FakeBlock
        cls.FakeTransformer = FakeTransformer

    def _identity(
        self,
        *,
        rank: int = 0,
        generation: int = 3,
        step: int = 7,
        total: int = 8,
    ) -> router.SameStateStepIdentity:
        return router.SameStateStepIdentity(
            generation=generation,
            step_index=step,
            timestep_token="sigma:0x1.4p-2",
            outer_trajectory_sha256="a" * 64,
            current_pair_state_sha256="b" * 64,
            rank=rank,
            ulysses_size=4,
            global_pair_tokens=total,
        )

    def _rope(self, total: int = 8):
        return self.torch.ones(
            (1, total, 1, 1), dtype=self.torch.complex64
        ).contiguous()

    def _hidden(self, identity, *, offset: float, requires_grad: bool = False):
        count = identity.local_sequence_tokens * 8
        value = self.torch.arange(count, dtype=self.torch.float32).reshape(
            1, identity.local_sequence_tokens, 8
        )
        value = (value + offset).contiguous()
        return value.requires_grad_(requires_grad)

    def _metadata(self, identity):
        return {
            "lengths": self.torch.tensor(
                [identity.global_pair_tokens], dtype=self.torch.int32
            ),
            "cu": self.torch.tensor(
                [0, identity.global_pair_tokens], dtype=self.torch.int32
            ),
            "max": identity.global_pair_tokens,
            "origin": identity.global_pair_tokens,
            "split": identity.local_sequence_tokens,
        }

    def _inverse(self, identity):
        def inverse(value, *, head_dim, seq_dim):
            self.assertEqual((head_dim, seq_dim), (2, 1))
            start = identity.local_sequence_start
            stop = start + identity.local_sequence_tokens
            local = value[:, start:stop]
            return local.repeat(1, 1, 4, 1).contiguous()

        return inverse

    def _model_bundle(
        self,
        *,
        identity=None,
        blocks: int = 1,
        arm: str = router.HYBRID_SOURCE_NOOP_KV,
        gate: float = 1.0,
        gather_factor: int = 4,
    ):
        identity = self._identity() if identity is None else identity
        spy = self.VarlenSpy()
        inverse = self._inverse(identity)
        bases = [
            self.BaseProcessor(spy, inverse, gather_factor=gather_factor)
            for _ in range(blocks)
        ]
        model = self.FakeTransformer(bases)
        state = self.ParallelState(identity.rank)
        handle = router.install_same_state_noop_value_router(
            model,
            selected_block_indices=tuple(range(blocks)),
            arm=arm,
            gate=gate,
            varlen_attention_fn=spy,
            get_parallel_state_fn=lambda: state,
            gather_heads_scatter_seq_fn=inverse,
        )
        return {
            "identity": identity,
            "spy": spy,
            "inverse": inverse,
            "bases": bases,
            "model": model,
            "state": state,
            "handle": handle,
            "bank": handle.cache_bank,
            "rope": self._rope(identity.global_pair_tokens),
            "metadata": self._metadata(identity),
        }

    def _capture(self, bundle, noop_hidden):
        with self.torch.no_grad(), router.same_state_noop_value_invocation(
            bundle["bank"],
            mode=router.CAPTURE_MODE,
            branch_tag=router.CAPTURE_BRANCH,
            arm=bundle["handle"].arm,
            gate=bundle["handle"].gate,
            identity=bundle["identity"],
        ):
            return bundle["model"](
                noop_hidden,
                bundle["rope"],
                bundle["metadata"],
            )

    def _action(self, bundle, action_hidden, *, rope=None, identity=None):
        active_identity = bundle["identity"] if identity is None else identity
        active_rope = bundle["rope"] if rope is None else rope
        with router.same_state_noop_value_invocation(
            bundle["bank"],
            mode=router.ACTION_MODE,
            branch_tag=router.ACTION_BRANCH,
            arm=bundle["handle"].arm,
            gate=bundle["handle"].gate,
            identity=active_identity,
        ):
            return bundle["model"](
                action_hidden,
                active_rope,
                bundle["metadata"],
            )

    def test_main_routes_paired_source_kv_and_preserves_target_suffix(self) -> None:
        bundle = self._model_bundle(blocks=1, gate=1.0)
        noop_hidden = self._hidden(bundle["identity"], offset=1.0)
        action_hidden = self._hidden(bundle["identity"], offset=101.0)
        noop_output = self._capture(bundle, noop_hidden)
        action_output = self._action(bundle, action_hidden)

        base = bundle["bases"][0]
        noop_projection, action_projection = base.project_records
        self.assertEqual(base.project_calls, 2)
        routed_call = bundle["spy"].calls[1]
        source = bundle["identity"].source_tokens
        self.assertTrue(
            self.torch.equal(
                routed_call["query"], action_projection["query"].squeeze(0)
            )
        )
        self.assertTrue(
            self.torch.equal(
                routed_call["key"][:source],
                noop_projection["key"].squeeze(0)[:source],
            )
        )
        self.assertTrue(
            self.torch.equal(
                routed_call["value"][:source],
                noop_projection["value"].squeeze(0)[:source],
            )
        )
        self.assertTrue(
            self.torch.equal(
                routed_call["key"][source:],
                action_projection["key"].squeeze(0)[source:],
            )
        )
        self.assertTrue(
            self.torch.equal(
                routed_call["value"][source:],
                action_projection["value"].squeeze(0)[source:],
            )
        )
        self.assertTrue(self.torch.equal(action_output, noop_output))
        self.assertTrue(bundle["bank"].complete)
        self.assertTrue(bundle["bank"].route_complete)
        captured = bundle["bank"].inspect_block(0)
        self.assertFalse(captured.noop_full_key.requires_grad)
        self.assertFalse(captured.noop_full_value.requires_grad)
        self.assertNotEqual(
            captured.noop_full_key.untyped_storage().data_ptr(),
            noop_projection["key"].untyped_storage().data_ptr(),
        )
        self.assertFalse(bundle["handle"].receipt()["optimizer_updates_authorized"])
        bundle["handle"].restore()

    def test_byte_exact_audit_distinguishes_signed_zero(self) -> None:
        positive = self.torch.tensor([0.0], dtype=self.torch.float32)
        negative = self.torch.tensor([-0.0], dtype=self.torch.float32)
        self.assertTrue(self.torch.equal(positive, negative))
        self.assertFalse(router._tensor_raw_bytes_equal(positive, negative))

    def test_action_gradient_survives_on_target_only_ulysses_rank(self) -> None:
        identity = self._identity(rank=2)
        bundle = self._model_bundle(identity=identity, gate=1.0)
        noop_hidden = self._hidden(identity, offset=2.0)
        self._capture(bundle, noop_hidden)
        action_hidden = self._hidden(
            identity,
            offset=51.0,
            requires_grad=True,
        )
        output = self._action(bundle, action_hidden)
        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(action_hidden.grad)
        self.assertTrue(self.torch.isfinite(action_hidden.grad).all())
        self.assertGreater(float(action_hidden.grad.abs().sum().item()), 0.0)
        boundary = bundle["handle"].boundary_routers[0]
        self.assertEqual(boundary.action_clamp_calls, 0)
        self.assertEqual(boundary.target_only_rank_delegations, 1)
        bundle["handle"].restore()

    def test_full_pair_noop_value_diagnostic_keeps_action_key(self) -> None:
        identity = self._identity(rank=2)
        bundle = self._model_bundle(
            identity=identity,
            arm=router.FULL_PAIR_NOOP_VALUE_DIAGNOSTIC,
            gate=1.0,
        )
        self._capture(bundle, self._hidden(identity, offset=3.0))
        self._action(bundle, self._hidden(identity, offset=103.0))
        base = bundle["bases"][0]
        noop_projection, action_projection = base.project_records
        routed = bundle["spy"].calls[1]
        self.assertTrue(
            self.torch.equal(
                routed["key"], action_projection["key"].squeeze(0)
            )
        )
        self.assertTrue(
            self.torch.equal(
                routed["value"], noop_projection["value"].squeeze(0)
            )
        )
        bundle["handle"].restore()

    def test_zero_gate_is_exact_official_delegate_and_boundary_noop(self) -> None:
        identity = self._identity(rank=0)
        spy = self.VarlenSpy()
        inverse = self._inverse(identity)
        base = self.BaseProcessor(spy, inverse)
        attn = self.FakeAttn(base)
        bank = router.SameStateNoopValueCacheBank((0,))
        processor = router.SameStateNoopValueSelfAttnProcessor(
            base,
            block_index=0,
            cache_bank=bank,
            arm=router.HYBRID_SOURCE_NOOP_KV,
            gate=0.0,
            varlen_attention_fn=spy,
            get_parallel_state_fn=lambda: self.ParallelState(0),
            gather_heads_scatter_seq_fn=inverse,
        )
        boundary = router.SameStateNoopSourceBoundaryRouter(
            block_index=0,
            cache_bank=bank,
            arm=router.HYBRID_SOURCE_NOOP_KV,
            gate=0.0,
        )
        sentinel = object()
        base.return_override = sentinel
        hidden = self._hidden(identity, offset=9.0)
        metadata = self._metadata(identity)
        with router.same_state_noop_value_invocation(
            bank,
            mode=router.ACTION_MODE,
            branch_tag=router.ACTION_BRANCH,
            arm=router.HYBRID_SOURCE_NOOP_KV,
            gate=0.0,
            identity=identity,
        ):
            result = processor(
                attn,
                hidden,
                rotary_emb=self._rope(),
                batch_image_vae_seqlen=metadata["lengths"],
                origin_hidden_states_seq_len=metadata["origin"],
                split_hidden_states_seq_len=metadata["split"],
                cu_seqlens_q_cache=metadata["cu"],
                max_seqlen_q_cache=metadata["max"],
            )
            hook_result = boundary(None, (), hidden)
        self.assertIs(result, sentinel)
        self.assertIsNone(hook_result)
        self.assertEqual(base.delegate_calls, 1)
        self.assertEqual(base.project_calls, 0)
        self.assertEqual(processor.zero_gate_delegations, 1)
        self.assertEqual(boundary.zero_gate_delegations, 1)
        self.assertFalse(bank.complete)

    def test_identity_rope_shape_and_source_only_capture_fail_closed(self) -> None:
        bundle = self._model_bundle(gate=1.0)
        self._capture(
            bundle,
            self._hidden(bundle["identity"], offset=1.0),
        )
        action = self._hidden(bundle["identity"], offset=31.0)
        variants = (
            replace(bundle["identity"], generation=4),
            replace(bundle["identity"], step_index=8),
            replace(bundle["identity"], outer_trajectory_sha256="c" * 64),
            replace(bundle["identity"], current_pair_state_sha256="d" * 64),
            replace(bundle["identity"], rank=1),
            replace(bundle["identity"], global_pair_tokens=16),
        )
        for identity in variants:
            with self.subTest(identity=identity):
                with self.assertRaisesRegex(
                    router.DMIQSameStateNoopValueRouterError,
                    "cross-step|cross-rank|cross-trajectory",
                ):
                    self._action(bundle, action, identity=identity)

        cloned_rope = bundle["rope"].clone().contiguous()
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "RoPE cache identity",
        ):
            self._action(bundle, action, rope=cloned_rope)
        self.assertTrue(bundle["bank"].poisoned)
        bundle["bank"].discard()
        bundle["handle"].restore()

        source_only = self._model_bundle(gate=1.0, gather_factor=2)
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "source-only caches are forbidden",
        ):
            self._capture(
                source_only,
                self._hidden(source_only["identity"], offset=1.0),
            )
        self.assertTrue(source_only["bank"].poisoned)
        source_only["bank"].discard()
        source_only["handle"].restore()

    def test_cache_is_one_use_then_explicitly_retired(self) -> None:
        bundle = self._model_bundle(gate=0.5)
        identity = bundle["identity"]
        self._capture(bundle, self._hidden(identity, offset=1.0))
        self._action(bundle, self._hidden(identity, offset=11.0))
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "already consumed",
        ):
            self._action(bundle, self._hidden(identity, offset=21.0))
        bundle["bank"].retire(identity)
        receipt = bundle["bank"].receipt()
        self.assertIsNone(receipt["identity"])
        self.assertEqual(receipt["retired_identity_count"], 1)
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "retired outer state",
        ):
            self._capture(bundle, self._hidden(identity, offset=1.0))

        next_identity = replace(identity, step_index=identity.step_index + 1)
        bundle["identity"] = next_identity
        bundle["metadata"] = self._metadata(next_identity)
        bundle["rope"] = self._rope(next_identity.global_pair_tokens)
        self._capture(bundle, self._hidden(next_identity, offset=2.0))
        self.assertTrue(bundle["bank"].complete)
        bundle["bank"].discard()
        bundle["handle"].restore()

    def test_incomplete_capture_poison_requires_discard(self) -> None:
        identity = self._identity()
        spy = self.VarlenSpy()
        inverse = self._inverse(identity)
        base = self.BaseProcessor(spy, inverse)
        attn = self.FakeAttn(base)
        bank = router.SameStateNoopValueCacheBank((0,))
        processor = router.SameStateNoopValueSelfAttnProcessor(
            base,
            block_index=0,
            cache_bank=bank,
            arm=router.HYBRID_SOURCE_NOOP_KV,
            gate=1.0,
            varlen_attention_fn=spy,
            get_parallel_state_fn=lambda: self.ParallelState(0),
            gather_heads_scatter_seq_fn=inverse,
        )
        metadata = self._metadata(identity)
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "missed a K/V pair or block boundary",
        ):
            with self.torch.no_grad(), router.same_state_noop_value_invocation(
                bank,
                mode=router.CAPTURE_MODE,
                branch_tag=router.CAPTURE_BRANCH,
                arm=router.HYBRID_SOURCE_NOOP_KV,
                gate=1.0,
                identity=identity,
            ):
                processor(
                    attn,
                    self._hidden(identity, offset=1.0),
                    rotary_emb=self._rope(),
                    batch_image_vae_seqlen=metadata["lengths"],
                    origin_hidden_states_seq_len=metadata["origin"],
                    split_hidden_states_seq_len=metadata["split"],
                    cu_seqlens_q_cache=metadata["cu"],
                    max_seqlen_q_cache=metadata["max"],
                )
        self.assertTrue(bank.poisoned)
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "poisoned",
        ):
            with router.same_state_noop_value_invocation(
                bank,
                mode=router.ACTION_MODE,
                branch_tag=router.ACTION_BRANCH,
                arm=router.HYBRID_SOURCE_NOOP_KV,
                gate=1.0,
                identity=identity,
            ):
                pass
        bank.discard()
        self.assertFalse(bank.poisoned)

    def test_ulysses4_and_block_identity_are_mandatory(self) -> None:
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "Ulysses=4",
        ):
            replace(self._identity(), ulysses_size=2).validate()
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "shard evenly",
        ):
            replace(self._identity(), global_pair_tokens=12).validate()
        bank = router.SameStateNoopValueCacheBank((0,))
        spy = self.VarlenSpy()
        identity = self._identity()
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "outside cache scope",
        ):
            router.SameStateNoopValueSelfAttnProcessor(
                self.BaseProcessor(spy, self._inverse(identity)),
                block_index=1,
                cache_bank=bank,
                arm=router.HYBRID_SOURCE_NOOP_KV,
                gate=1.0,
            )

    def test_install_restore_is_atomic_and_tamper_detected(self) -> None:
        bundle = self._model_bundle(blocks=2, gate=1.0)
        handle = bundle["handle"]
        for index, installed in zip(handle.indices, handle.processors):
            self.assertIs(
                bundle["model"].blocks[index].attn1.processor,
                installed,
            )
        originals = handle.original_processors
        handle.restore()
        for index, original in zip(handle.indices, originals):
            self.assertIs(
                bundle["model"].blocks[index].attn1.processor,
                original,
            )
            self.assertEqual(len(bundle["model"].blocks[index]._forward_hooks), 0)

        tampered = self._model_bundle(gate=1.0)
        installed = tampered["handle"].processors[0]
        tampered["model"].blocks[0].attn1.processor = object()
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "changed behind patch handle",
        ):
            tampered["handle"].restore()
        tampered["model"].blocks[0].attn1.processor = installed
        tampered["handle"].restore()

        hook_tampered = self._model_bundle(gate=1.0)
        extra_hook = hook_tampered["model"].blocks[0].register_forward_hook(
            lambda module, inputs, output: None
        )
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "boundary hook changed",
        ):
            hook_tampered["handle"].restore()
        extra_hook.remove()
        hook_tampered["handle"].restore()

        identity = self._identity()
        spy = self.VarlenSpy()
        inverse = self._inverse(identity)
        bases = [
            self.BaseProcessor(spy, inverse),
            self.BaseProcessor(spy, inverse),
        ]
        model = self.FakeTransformer(bases)
        model.blocks[1].attn1.processor = None
        with self.assertRaisesRegex(
            router.DMIQSameStateNoopValueRouterError,
            "lacks a processor",
        ):
            router.install_same_state_noop_value_router(
                model,
                selected_block_indices=(0, 1),
                gate=1.0,
                varlen_attention_fn=spy,
                get_parallel_state_fn=lambda: self.ParallelState(0),
                gather_heads_scatter_seq_fn=inverse,
            )
        self.assertIs(model.blocks[0].attn1.processor, bases[0])
        self.assertEqual(len(model.blocks[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
