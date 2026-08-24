import unittest

import torch

from pact.masks import (
    boundary_ring,
    dilate_and_feather,
    dilate_mask,
    source_target_tube_union,
    validate_video_mask,
)


class VideoMaskTests(unittest.TestCase):
    def test_validation_preserves_tensor_and_rejects_bad_masks(self) -> None:
        mask = torch.rand(2, 1, 3, 4, 5, requires_grad=True)
        self.assertIs(validate_video_mask(mask, batch_size=2, frames=3), mask)
        with self.assertRaises(ValueError):
            validate_video_mask(torch.zeros(2, 3, 4, 5))
        with self.assertRaises(ValueError):
            validate_video_mask(torch.zeros(2, 2, 3, 4, 5))
        with self.assertRaises(TypeError):
            validate_video_mask(torch.zeros(2, 1, 3, 4, 5, dtype=torch.int64))
        bad_range = torch.zeros(1, 1, 1, 1, 2)
        bad_range[..., 0] = -0.1
        with self.assertRaises(ValueError):
            validate_video_mask(bad_range)
        nonfinite = torch.full((1, 1, 1, 1, 1), float("nan"))
        with self.assertRaises(ValueError):
            validate_video_mask(nonfinite)

    def test_source_target_union_supports_soft_tubes(self) -> None:
        source = torch.tensor([0.0, 0.3, 1.0]).reshape(1, 1, 1, 1, 3)
        target = torch.tensor([0.2, 0.8, 0.0]).reshape(1, 1, 1, 1, 3)
        expected = torch.tensor([0.2, 0.8, 1.0]).reshape_as(source)
        self.assertTrue(torch.equal(source_target_tube_union(source, target), expected))
        with self.assertRaises(ValueError):
            source_target_tube_union(source, target.expand(2, -1, -1, -1, -1))

    def test_dilation_feather_and_boundary_geometry(self) -> None:
        point = torch.zeros(1, 1, 1, 7, 7)
        point[..., 3, 3] = 1.0
        dilated = dilate_mask(point, (0, 1, 1))
        self.assertEqual(float(dilated.sum()), 9.0)
        feathered = dilate_and_feather(
            point, dilation_radius=0, feather_radius=(0, 1, 1)
        )
        self.assertEqual(float(feathered[..., 3, 3]), 1.0)
        self.assertAlmostEqual(float(feathered[..., 3, 2]), 1.0 / 9.0, places=6)
        self.assertEqual(float(feathered[..., 0, 0]), 0.0)

        square = torch.zeros_like(point)
        square[..., 2:5, 2:5] = 1.0
        self.assertEqual(float(boundary_ring(square, mode="inner").sum()), 8.0)
        self.assertEqual(float(boundary_ring(square, mode="outer").sum()), 16.0)
        self.assertEqual(float(boundary_ring(square, mode="both").sum()), 24.0)

    def test_morphology_keeps_gradient_path(self) -> None:
        mask = torch.rand(1, 1, 2, 4, 4, requires_grad=True)
        output = dilate_and_feather(
            mask, dilation_radius=(0, 1, 1), feather_radius=(0, 1, 1)
        )
        output.sum().backward()
        self.assertIsNotNone(mask.grad)
        self.assertTrue(bool(torch.isfinite(mask.grad).all()))


if __name__ == "__main__":
    unittest.main()
