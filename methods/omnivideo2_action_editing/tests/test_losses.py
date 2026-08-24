import unittest

import torch

from pact.losses import (
    area_normalized_masked_loss,
    boundary_consistency_loss,
    edit_preserve_losses,
    outside_temporal_difference_loss,
    pact_reconstruction_losses,
)


class PactLossTests(unittest.TestCase):
    def test_edit_and_preserve_are_independently_area_normalized(self) -> None:
        prediction = torch.zeros(1, 2, 2, 2, 2)
        target = torch.full_like(prediction, 2.0)
        source = torch.full_like(prediction, 3.0)
        mask = torch.zeros(1, 1, 2, 2, 2)
        mask[..., 0] = 1.0
        losses = edit_preserve_losses(prediction, target, source, mask)
        self.assertEqual(float(losses["edit"]), 2.0)
        self.assertEqual(float(losses["preserve"]), 3.0)

    def test_preserve_loss_can_exclude_uncertain_boundary_ring(self) -> None:
        prediction = torch.zeros(1, 1, 1, 1, 4)
        source = torch.tensor([0.0, 10.0, 2.0, 2.0]).reshape_as(prediction)
        target = torch.zeros_like(prediction)
        edit_mask = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(
            1, 1, 1, 1, 4
        )
        ring = torch.tensor([0.0, 1.0, 0.0, 0.0]).reshape_as(edit_mask)
        losses = edit_preserve_losses(
            prediction,
            target,
            source,
            edit_mask,
            exclude_from_preserve=ring,
            loss_type="l1",
        )
        self.assertAlmostEqual(float(losses["preserve"]), 2.0, places=5)

    def test_mask_area_does_not_change_constant_error(self) -> None:
        prediction = torch.zeros(1, 1, 1, 3, 3)
        target = torch.ones_like(prediction)
        one = torch.zeros_like(prediction)
        one[..., 1, 1] = 1.0
        many = torch.ones_like(prediction)
        self.assertEqual(float(area_normalized_masked_loss(prediction, target, one)), 1.0)
        self.assertEqual(float(area_normalized_masked_loss(prediction, target, many)), 1.0)

    def test_boundary_matches_source_target_composite(self) -> None:
        source = torch.zeros(1, 1, 1, 7, 7)
        target = torch.full_like(source, 2.0)
        mask = torch.zeros(1, 1, 1, 7, 7)
        mask[..., 2:5, 2:5] = 1.0
        composite = source * (1.0 - mask) + target * mask
        self.assertEqual(
            float(boundary_consistency_loss(composite, target, source, mask)), 0.0
        )
        wrong = composite.clone()
        wrong[..., 1:6, 1:6] += 1.0
        self.assertGreater(
            float(boundary_consistency_loss(wrong, target, source, mask)), 0.0
        )

    def test_temporal_loss_ignores_selected_actor(self) -> None:
        source = torch.tensor([0.0, 1.0, 3.0]).reshape(1, 1, 3, 1, 1).expand(-1, -1, -1, 1, 2).clone()
        prediction = source.clone()
        prediction[..., 0] = torch.tensor([0.0, 100.0, -50.0]).reshape(1, 1, 3, 1)
        mask = torch.zeros(1, 1, 3, 1, 2)
        mask[..., 0] = 1.0
        loss = outside_temporal_difference_loss(prediction, source, mask)
        self.assertEqual(float(loss), 0.0)
        prediction[..., 1] += torch.tensor([0.0, 2.0, 6.0]).reshape(1, 1, 3, 1)
        self.assertGreater(
            float(outside_temporal_difference_loss(prediction, source, mask)), 0.0
        )

    def test_total_components_are_loggable_and_differentiable(self) -> None:
        prediction = torch.randn(2, 2, 3, 4, 4, requires_grad=True)
        target = torch.randn_like(prediction)
        source = torch.randn_like(prediction)
        mask = torch.zeros(2, 1, 3, 4, 4)
        mask[:, :, :, 1:3, 1:3] = 1.0
        losses = pact_reconstruction_losses(
            prediction,
            target,
            source,
            mask,
            weights={"boundary": 0.5, "temporal_outside": 2.0},
        )
        self.assertEqual(
            set(losses), {"total", "edit", "preserve", "boundary", "temporal_outside"}
        )
        self.assertTrue(all(value.ndim == 0 for value in losses.values()))
        losses["total"].backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_empty_mask_and_single_frame_return_differentiable_zero(self) -> None:
        prediction = torch.randn(1, 1, 1, 2, 2, requires_grad=True)
        target = torch.randn_like(prediction)
        empty = torch.zeros(1, 1, 1, 2, 2)
        first = area_normalized_masked_loss(prediction, target, empty)
        second = outside_temporal_difference_loss(prediction, target, empty)
        self.assertEqual(float(first), 0.0)
        self.assertEqual(float(second), 0.0)
        (first + second).backward()
        self.assertIsNotNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
