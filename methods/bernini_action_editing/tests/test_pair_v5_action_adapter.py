from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import pair_v5_action_adapter as adapter

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    adapter = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _Attention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            # In a real CIO composition these may already be CIO wrappers.
            # Object identity is the contract needed by the action adapter.
            self.attn1 = _Attention(hidden)
            self.attn2 = _Attention(hidden)


    class _Transformer(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(16, hidden, kernel_size=(1, 2, 2))
            self.blocks = nn.ModuleList(
                [_Block(hidden) for _ in range(adapter.TOTAL_BLOCKS_1P3B)]
            )

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


def _route(
    *,
    sigma_index: int,
    rank: int = 0,
    size: int = 1,
    total: int = 13,
    condition: int = 5,
    branch_name: str = "VI",
) -> "adapter.PairV5ActionRoute":
    return adapter.PairV5ActionRoute(
        total_tokens=total,
        condition_tokens=condition,
        sequence_parallel_rank=rank,
        sequence_parallel_size=size,
        branch_name=branch_name,
        sigma_schedule_index=sigma_index,
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "AUH vace torch runtime is required")
class PairV5ActionRouteTests(unittest.TestCase):
    def test_exact40_gate_partition_is_preregistered(self) -> None:
        self.assertEqual(adapter.HIGH_SIGMA_INDICES, tuple(range(33)))
        self.assertEqual(adapter.MID_SIGMA_INDICES, tuple(range(33, 38)))
        self.assertEqual(adapter.LOW_SIGMA_INDICES, (38, 39))
        self.assertEqual(adapter.sigma_gate(0), ("high", 1.0))
        self.assertEqual(adapter.sigma_gate(33), ("mid", 0.5))
        self.assertEqual(adapter.sigma_gate(38), ("low_base_only", 0.0))
        with self.assertRaisesRegex(adapter.PairV5ActionAdapterError, "\[0,39\]"):
            adapter.sigma_gate(40)

    def test_sp4_selector_is_append_padded_target_suffix(self) -> None:
        selectors = [
            _route(sigma_index=0, rank=rank, size=4).local_target_selector(
                device=torch.device("cpu")
            )
            for rank in range(4)
        ]
        joined = torch.cat(selectors)
        self.assertEqual(joined.numel(), 16)
        self.assertTrue(
            torch.equal(joined[:13], torch.tensor([False] * 5 + [True] * 8))
        )
        self.assertFalse(bool(joined[13:].any()))

    def test_route_receipt_is_closed_and_digest_bound(self) -> None:
        receipt = dict(_route(sigma_index=33).receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, adapter._object_sha256(receipt))
        self.assertEqual(receipt["sigma_gate"], "mid")
        self.assertEqual(receipt["sigma_gate_weight"], 0.5)


@unittest.skipUnless(_TORCH_AVAILABLE, "AUH vace torch runtime is required")
class PairV5ActionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(71)
        self.model = _Transformer(hidden=8)
        self.model.requires_grad_(False)
        self.original_patch = self.model.patch_embedding
        self.original_attn1 = tuple(
            (
                block.attn1,
                block.attn1.to_q,
                block.attn1.to_out[0],
            )
            for block in self.model.blocks
        )
        self.original_q = tuple(block.attn2.to_q for block in self.model.blocks)
        self.original_o = tuple(block.attn2.to_out[0] for block in self.model.blocks)
        self.handle = adapter.install_pair_v5_action_adapter(self.model)

    def tearDown(self) -> None:
        if not self.handle.restored:
            self.handle.restore()

    def _make_nonzero(self, wrapper: "adapter.PairV5TargetRowActionLoRA") -> None:
        with torch.no_grad():
            wrapper.action_lora_a.weight.fill_(0.25)
            wrapper.action_lora_b.weight.fill_(0.5)

    def test_low_sigma_gate_is_direct_byte_exact_base_parity(self) -> None:
        q_wrapper = self.handle.q_wrappers[0][1]
        o_wrapper = self.handle.o_wrappers[0][1]
        self._make_nonzero(q_wrapper)
        self._make_nonzero(o_wrapper)
        hidden = torch.randn(1, 13, 8)
        with self.handle.route(_route(sigma_index=38)):
            actual_q = q_wrapper(hidden)
            actual_o = o_wrapper(hidden)
            delta_q = q_wrapper.adapter_delta(hidden)
        self.assertTrue(torch.equal(actual_q, self.original_q[0](hidden)))
        self.assertTrue(torch.equal(actual_o, self.original_o[0](hidden)))
        self.assertTrue(torch.equal(delta_q, torch.zeros_like(delta_q)))

    def test_source_rows_exact_and_target_delta_only_for_q_and_o(self) -> None:
        hidden = torch.ones(1, 13, 8)
        selector = _route(sigma_index=0).local_target_selector(
            device=torch.device("cpu")
        )
        for wrapper, original in (
            (self.handle.q_wrappers[0][1], self.original_q[0]),
            (self.handle.o_wrappers[0][1], self.original_o[0]),
        ):
            self._make_nonzero(wrapper)
            expected = original(hidden)
            with self.handle.route(_route(sigma_index=0)):
                changed = wrapper(hidden)
            self.assertTrue(torch.equal(changed[:, ~selector], expected[:, ~selector]))
            self.assertGreater(
                float((changed[:, selector] - expected[:, selector]).abs().sum()),
                0.0,
            )

    def test_sp4_source_and_padding_rows_are_exact_on_every_rank(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        original = self.original_q[0]
        self._make_nonzero(wrapper)
        for rank in range(4):
            route = _route(sigma_index=0, rank=rank, size=4)
            hidden = torch.ones(1, route.local_length, 8)
            selector = route.local_target_selector(device=torch.device("cpu"))
            expected = original(hidden)
            with self.handle.route(route):
                changed = wrapper(hidden)
            self.assertTrue(torch.equal(changed[:, ~selector], expected[:, ~selector]))
            if bool(selector.any().item()):
                self.assertGreater(
                    float((changed[:, selector] - expected[:, selector]).abs().sum()),
                    0.0,
                )

    def test_active_v_source_only_sp_shard_keeps_real_zero_b_graph(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        original = self.original_q[0]
        wrapper.action_lora_a.weight.requires_grad_(False)
        wrapper.action_lora_b.weight.requires_grad_(True)
        route = _route(
            sigma_index=33,
            rank=0,
            size=4,
            total=8,
            condition=4,
            branch_name="V",
        )
        selector = route.local_target_selector(device=torch.device("cpu"))
        self.assertFalse(bool(selector.any().item()))
        hidden = torch.ones(1, route.local_length, 8)
        expected = original(hidden)
        with self.handle.route(route):
            actual = wrapper(hidden)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.requires_grad)
        self.assertIsNotNone(actual.grad_fn)
        actual.sum().backward()
        self.assertIsNone(wrapper.action_lora_a.weight.grad)
        self.assertIsNotNone(wrapper.action_lora_b.weight.grad)
        self.assertEqual(
            int(torch.count_nonzero(wrapper.action_lora_b.weight.grad).item()), 0
        )
        self.assertTrue(all(parameter.grad is None for parameter in original.parameters()))

        wrapper.action_lora_b.weight.grad = None
        low_route = _route(
            sigma_index=38,
            rank=0,
            size=4,
            total=8,
            condition=4,
            branch_name="V",
        )
        with self.handle.route(low_route):
            low = wrapper(hidden)
        self.assertFalse(low.requires_grad)
        self.assertIsNone(low.grad_fn)
        self.assertIsNone(wrapper.action_lora_b.weight.grad)

    def test_mid_gate_is_exactly_half_the_high_action_delta(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, 13, 8)
        with self.handle.route(_route(sigma_index=0)):
            high = wrapper.adapter_delta(hidden)
        with self.handle.route(_route(sigma_index=33)):
            mid = wrapper.adapter_delta(hidden)
        self.assertTrue(torch.equal(mid * 2.0, high))

    def test_high_sigma_target_loss_reaches_only_action_lora(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, 13, 8)
        selector = _route(sigma_index=0).local_target_selector(
            device=torch.device("cpu")
        )
        with self.handle.route(_route(sigma_index=0)):
            wrapper(hidden)[:, selector].sum().backward()
        self.assertIsNotNone(wrapper.action_lora_a.weight.grad)
        self.assertIsNotNone(wrapper.action_lora_b.weight.grad)
        self.assertGreater(float(wrapper.action_lora_a.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(wrapper.action_lora_b.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in self.original_q[0].parameters())
        )

    def test_module_scope_base_freeze_and_receipt_closure(self) -> None:
        self.assertIs(self.model.patch_embedding, self.original_patch)
        self.assertTrue(self.handle.base_parameters_frozen())
        self.assertTrue(self.handle.self_attention_untouched())
        for index, block in enumerate(self.model.blocks):
            attn1, attn1_q, attn1_o = self.original_attn1[index]
            self.assertIs(block.attn1, attn1)
            self.assertIs(block.attn1.to_q, attn1_q)
            self.assertIs(block.attn1.to_out[0], attn1_o)
            if index in adapter.ACTION_BLOCK_INDICES:
                self.assertIsInstance(
                    block.attn2.to_q, adapter.PairV5TargetRowActionLoRA
                )
                self.assertIsInstance(
                    block.attn2.to_out[0], adapter.PairV5TargetRowActionLoRA
                )
            else:
                self.assertIs(block.attn2.to_q, self.original_q[index])
                self.assertIs(block.attn2.to_out[0], self.original_o[index])
        trainable = self.handle.trainable_named_parameters()
        self.assertEqual(len(trainable), 23 * 2 * 2)
        self.assertTrue(all("attn2" in name for name, _ in trainable))
        self.assertTrue(
            all("action_lora_a" in name or "action_lora_b" in name for name, _ in trainable)
        )
        self.assertTrue(
            all(tuple(parameter.shape) in {(8, 8)} for _, parameter in trainable)
        )
        receipt = dict(self.handle.receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, adapter._object_sha256(receipt))
        self.assertEqual(receipt["rank"], 8)
        self.assertEqual(receipt["alpha"], 8.0)
        self.assertEqual(receipt["dropout"], 0.0)
        self.assertTrue(receipt["self_attention_and_frozen_cio_untouched"])
        self.assertFalse(receipt["semantic_action_claim"])

    def test_trainable_state_save_load_is_closed_and_exact(self) -> None:
        state = dict(self.handle.state_dict_for_save())
        first_name = sorted(state)[0]
        before = dict(self.handle.state_dict_for_save())[first_name]
        state[first_name].add_(1.0)
        self.assertTrue(
            torch.equal(dict(self.handle.state_dict_for_save())[first_name], before)
        )
        state = dict(self.handle.state_dict_for_save())
        replacement = {
            name: torch.full_like(value, 0.125) for name, value in state.items()
        }
        load_receipt = dict(self.handle.load_trainable_state_dict(replacement))
        digest = load_receipt.pop("digest")
        self.assertEqual(digest, adapter._object_sha256(load_receipt))
        observed = dict(self.handle.state_dict_for_save())
        self.assertTrue(
            all(torch.equal(value, replacement[name]) for name, value in observed.items())
        )
        missing = dict(replacement)
        missing.pop(first_name)
        with self.assertRaisesRegex(adapter.PairV5ActionAdapterError, "key closure"):
            self.handle.load_trainable_state_dict(missing)
        bad_dtype = dict(replacement)
        bad_dtype[first_name] = bad_dtype[first_name].half()
        with self.assertRaisesRegex(adapter.PairV5ActionAdapterError, "CPU FP32"):
            self.handle.load_trainable_state_dict(bad_dtype)

    def test_restore_recovers_only_original_cross_attention_modules(self) -> None:
        self.handle.restore()
        for index, block in enumerate(self.model.blocks):
            self.assertIs(block.attn2.to_q, self.original_q[index])
            self.assertIs(block.attn2.to_out[0], self.original_o[index])
            self.assertIs(block.attn1, self.original_attn1[index][0])
        self.assertIs(self.model.patch_embedding, self.original_patch)

    def test_install_requires_frozen_base_and_exact_1p3b_structure(self) -> None:
        trainable = _Transformer()
        with self.assertRaisesRegex(adapter.PairV5ActionAdapterError, "freeze"):
            adapter.install_pair_v5_action_adapter(trainable)


if __name__ == "__main__":
    unittest.main()
