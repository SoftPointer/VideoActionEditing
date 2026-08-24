import unittest

import torch

from pact.guidance import (
    anchor_to_source_noisy,
    gate_keep_edit_deltas,
    source_noisy_anchor,
    spatially_gated_guidance,
)
from pact.router import PromptConditionedMaskRouter, bce_dice_loss, router_loss_components


class MaskRouterTests(unittest.TestCase):
    def test_router_shape_prompt_conditioning_and_gradient(self) -> None:
        torch.manual_seed(3)
        router = PromptConditionedMaskRouter(4, 6, hidden_channels=8, depth=1)
        features = torch.randn(2, 4, 3, 5, 7, requires_grad=True)
        prompts = torch.randn(2, 6, requires_grad=True)
        logits = router(features, prompts)
        self.assertEqual(logits.shape, (2, 1, 3, 5, 7))
        changed = router(features, prompts + 1.0)
        self.assertFalse(torch.equal(logits, changed))
        target = (torch.rand_like(logits) > 0.5).float()
        loss = bce_dice_loss(logits, target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(features.grad).all()))
        self.assertTrue(bool(torch.isfinite(prompts.grad).all()))
        self.assertGreater(float(router.prompt_film.weight.grad.abs().sum()), 0.0)

    def test_correct_logits_have_lower_bce_dice(self) -> None:
        target = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 1, 2)
        correct = torch.tensor([-8.0, 8.0]).reshape_as(target)
        wrong = -correct
        self.assertLess(
            float(bce_dice_loss(correct, target)),
            float(bce_dice_loss(wrong, target)),
        )
        components = router_loss_components(correct, target)
        self.assertEqual(set(components), {"total", "bce", "dice"})

    def test_router_rejects_prompt_shape_mismatch(self) -> None:
        router = PromptConditionedMaskRouter(2, 4)
        with self.assertRaises(ValueError):
            router(torch.randn(2, 2, 2, 2, 2), torch.randn(1, 4))


class GuidanceTests(unittest.TestCase):
    def test_keep_and_edit_deltas_are_spatially_gated(self) -> None:
        base = torch.zeros(1, 1, 1, 1, 2)
        keep_delta = torch.full_like(base, 2.0)
        edit_delta = torch.full_like(base, 4.0)
        mask = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 1, 2)
        guided = gate_keep_edit_deltas(base, keep_delta, edit_delta, mask)
        self.assertTrue(torch.equal(guided.flatten(), torch.tensor([2.0, 4.0])))

        unconditional = torch.ones_like(base)
        keep = torch.full_like(base, 3.0)
        edit = torch.full_like(base, 5.0)
        conditioned = spatially_gated_guidance(unconditional, keep, edit, mask)
        self.assertTrue(torch.equal(conditioned.flatten(), torch.tensor([3.0, 5.0])))

    def test_source_noisy_anchor_preserves_edit_and_anchors_outside(self) -> None:
        current = torch.full((1, 1, 1, 1, 2), 10.0)
        source_x_t = torch.full_like(current, 2.0)
        mask = torch.tensor([0.0, 1.0]).reshape_as(current)
        half = anchor_to_source_noisy(current, source_x_t, mask, strength=0.5)
        self.assertTrue(torch.equal(half.flatten(), torch.tensor([6.0, 10.0])))

        source_x0 = torch.zeros_like(current)
        noise = torch.full_like(current, 4.0)
        anchored = source_noisy_anchor(current, source_x0, noise, 0.25, mask)
        self.assertTrue(torch.equal(anchored.flatten(), torch.tensor([1.0, 10.0])))

    def test_soft_guidance_is_differentiable(self) -> None:
        base = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
        keep = torch.randn_like(base, requires_grad=True)
        edit = torch.randn_like(base, requires_grad=True)
        mask = torch.full((1, 1, 2, 2, 2), 0.3, requires_grad=True)
        output = gate_keep_edit_deltas(base, keep, edit, mask)
        output.square().mean().backward()
        for value in (base, keep, edit, mask):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))


if __name__ == "__main__":
    unittest.main()
