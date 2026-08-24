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

    import source_self_native_target_adapter as adapter

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
            self.attn1 = _Attention(hidden)


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


@unittest.skipUnless(_TORCH_AVAILABLE, "AUH vace torch runtime is required")
class NativeTargetRouteTests(unittest.TestCase):
    def test_sp4_selector_is_append_padded_target_suffix(self) -> None:
        selectors = []
        for rank in range(4):
            route = adapter.NativeTargetRoute(
                total_tokens=13,
                condition_tokens=5,
                sequence_parallel_rank=rank,
                sequence_parallel_size=4,
                branch_name="VI",
            )
            selectors.append(route.local_target_selector(device=torch.device("cpu")))
        joined = torch.cat(selectors)
        self.assertEqual(joined.numel(), 16)
        self.assertTrue(torch.equal(joined[:13], torch.tensor([False] * 5 + [True] * 8)))
        self.assertFalse(bool(joined[13:].any()))

    def test_target_row_delta_is_zero_on_conditions_and_padding(self) -> None:
        base = nn.Linear(4, 4, bias=False)
        base.requires_grad_(False)
        wrapper = adapter.NativeTargetRowLoRA(
            base, rank=2, alpha=2.0, projection="to_q"
        )
        nn.init.ones_(wrapper.lora_a.weight)
        nn.init.ones_(wrapper.lora_b.weight)
        hidden = torch.ones((1, 4, 4))
        route = adapter.NativeTargetRoute(13, 5, 3, 4, "VI")
        with adapter.activate_route(route):
            delta = wrapper.adapter_delta(hidden)
        selector = route.local_target_selector(device=torch.device("cpu"))
        self.assertTrue(torch.equal(delta[:, ~selector, :], torch.zeros_like(delta[:, ~selector, :])))
        self.assertGreater(float(delta[:, selector, :].abs().sum()), 0.0)

    def test_disabled_route_is_bitwise_base(self) -> None:
        base = nn.Linear(4, 4, bias=False)
        base.requires_grad_(False)
        wrapper = adapter.NativeTargetRowLoRA(
            base, rank=2, alpha=2.0, projection="to_out.0"
        )
        nn.init.ones_(wrapper.lora_b.weight)
        hidden = torch.randn((1, 3, 4))
        self.assertTrue(torch.equal(wrapper(hidden), base(hidden)))


@unittest.skipUnless(_TORCH_AVAILABLE, "AUH vace torch runtime is required")
class NativeTargetAdapterHandleTests(unittest.TestCase):
    def test_install_never_wraps_patch_and_only_qo_train(self) -> None:
        model = _Transformer()
        model.requires_grad_(False)
        patch = model.patch_embedding
        handle = adapter.install_native_target_adapter(model, rank=2, alpha=2.0)
        self.assertIs(model.patch_embedding, patch)
        self.assertTrue(handle.receipt()["patch_embedding_untouched"])
        self.assertTrue(handle.receipt()["base_parameters_frozen"])
        names = [name for name, _ in handle.trainable_named_parameters()]
        self.assertTrue(names)
        self.assertTrue(all("attn1.to_q" in name or "attn1.to_out.0" in name for name in names))
        self.assertFalse(any("patch" in name or "to_k" in name or "to_v" in name for name in names))
        state = handle.state_dict_for_save()
        self.assertEqual(set(state), set(names))
        handle.restore()
        self.assertIs(model.patch_embedding, patch)
        self.assertFalse(isinstance(model.blocks[0].attn1.to_q, adapter.NativeTargetRowLoRA))

    def test_install_requires_frozen_base_and_registered_scope(self) -> None:
        with self.assertRaisesRegex(adapter.NativeTargetAdapterError, "freeze"):
            adapter.install_native_target_adapter(_Transformer())
        model = _Transformer()
        model.requires_grad_(False)
        with self.assertRaisesRegex(adapter.NativeTargetAdapterError, "scope"):
            adapter.install_native_target_adapter(model, block_indices=(0, 1))

    def test_route_context_covers_backward(self) -> None:
        model = _Transformer(hidden=4)
        model.requires_grad_(False)
        handle = adapter.install_native_target_adapter(model, rank=2, alpha=2.0)
        route = adapter.NativeTargetRoute(8, 4, 0, 1, "VI")
        hidden = torch.randn((1, 8, 4))
        with handle.route(route):
            output = model.blocks[0].attn1.to_q(hidden)
            output.sum().backward()
        wrapper = model.blocks[0].attn1.to_q
        self.assertIsInstance(wrapper, adapter.NativeTargetRowLoRA)
        self.assertIsNotNone(wrapper.lora_a.weight.grad)
        self.assertIsNotNone(wrapper.lora_b.weight.grad)
        handle.restore()


if __name__ == "__main__":
    unittest.main()
