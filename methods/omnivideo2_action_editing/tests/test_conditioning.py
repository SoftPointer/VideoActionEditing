import unittest

import torch

from pact.conditioning import budget_source_latent, erase_source_motion


class SourceMotionErasureTests(unittest.TestCase):
    def test_zero_erasure_changes_only_selected_future_and_keeps_frame_zero(self) -> None:
        source = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
        mask = torch.zeros(1, 1, 3, 2, 2)
        mask[:, :, :, 0, 0] = 1.0
        output = erase_source_motion(source, mask, mode="zero")
        self.assertTrue(torch.equal(output[:, :, 0], source[:, :, 0]))
        self.assertTrue(torch.equal(output[:, :, 1:, 0, 0], torch.zeros(1, 2, 2)))
        self.assertTrue(torch.equal(output[:, :, 1:, 0, 1:], source[:, :, 1:, 0, 1:]))
        self.assertTrue(torch.equal(output[:, :, 1:, 1], source[:, :, 1:, 1]))

    def test_soft_mask_and_temporal_mean(self) -> None:
        source = torch.tensor([1.0, 5.0, 9.0]).reshape(1, 1, 3, 1, 1)
        soft = torch.full((1, 1, 3, 1, 1), 0.25)
        zeroed = erase_source_motion(source, soft, mode="zero")
        self.assertTrue(torch.equal(zeroed[:, :, 0], source[:, :, 0]))
        self.assertTrue(
            torch.allclose(zeroed.flatten(), torch.tensor([1.0, 3.75, 6.75]))
        )
        mean_erased = erase_source_motion(
            source, torch.ones_like(soft), mode="temporal_mean"
        )
        self.assertTrue(torch.equal(mean_erased.flatten(), torch.tensor([1.0, 5.0, 5.0])))

    def test_erasure_shape_validation_and_gradient(self) -> None:
        source = torch.randn(1, 2, 3, 2, 2, requires_grad=True)
        with self.assertRaises(ValueError):
            erase_source_motion(source, torch.ones(1, 1, 2, 2, 2))
        output = erase_source_motion(
            source, torch.full((1, 1, 3, 2, 2), 0.5), mode="temporal_mean"
        )
        output.square().mean().backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(bool(torch.isfinite(source.grad).all()))


class SourceLatentBudgetTests(unittest.TestCase):
    def test_81_video_frame_latent_fits_6144_context_after_time_pooling(self) -> None:
        source = torch.randn(1, 2, 21, 60, 104)
        output, metadata = budget_source_latent(
            source,
            max_context_len=6144,
            nonvisual_tokens=512,
            visual_patch_size=(1, 4, 4),
        )
        self.assertEqual(output.shape, (1, 2, 14, 60, 104))
        self.assertTrue(metadata.compressed)
        self.assertEqual(metadata.original_visual_tokens, 21 * 15 * 26)
        self.assertEqual(metadata.output_visual_tokens, 14 * 15 * 26)
        self.assertLessEqual(metadata.output_total_tokens, 6144)

    def test_under_budget_returns_bitwise_identical_object(self) -> None:
        source = torch.randn(1, 2, 3, 8, 8)
        output, metadata = budget_source_latent(
            source, max_context_len=128, nonvisual_tokens=4
        )
        self.assertIs(output, source)
        self.assertEqual(output.data_ptr(), source.data_ptr())
        self.assertTrue(torch.equal(output, source))
        self.assertFalse(metadata.compressed)

    def test_budget_fails_closed_when_one_frame_cannot_fit(self) -> None:
        source = torch.randn(1, 1, 2, 60, 104)
        with self.assertRaises(ValueError):
            budget_source_latent(source, max_context_len=389, nonvisual_tokens=0)
        with self.assertRaises(ValueError):
            budget_source_latent(source, max_context_len=500, nonvisual_tokens=200)

    def test_temporal_pooling_is_differentiable(self) -> None:
        source = torch.randn(1, 1, 6, 8, 8, requires_grad=True)
        output, _ = budget_source_latent(
            source, max_context_len=8, nonvisual_tokens=0
        )
        self.assertEqual(output.shape[2], 2)
        output.sum().backward()
        self.assertTrue(bool(torch.isfinite(source.grad).all()))

    def test_token_count_matches_no_padding_conv3d_floor_geometry(self) -> None:
        source = torch.randn(1, 1, 5, 9, 10)
        output, metadata = budget_source_latent(
            source,
            max_context_len=20,
            nonvisual_tokens=3,
            visual_patch_size=(2, 4, 4),
        )
        self.assertIs(output, source)
        self.assertEqual(metadata.original_visual_tokens, 2 * 2 * 2)
        with self.assertRaisesRegex(ValueError, "at least"):
            budget_source_latent(
                torch.randn(1, 1, 1, 3, 3),
                max_context_len=20,
                nonvisual_tokens=0,
                visual_patch_size=(2, 4, 4),
            )


if __name__ == "__main__":
    unittest.main()
