from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "diagnose_saic_anchor_dinov2_temporal_representation_v1.py"
)
SPEC = importlib.util.spec_from_file_location("saic_temporal_rep", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SaicTemporalRepresentationTests(unittest.TestCase):
    def test_representation_geometry_and_unit_norm(self) -> None:
        rng = np.random.default_rng(7)
        global_feature = rng.normal(size=(17, 8)).astype(np.float32)
        dense_feature = rng.normal(size=(17, 4, 8)).astype(np.float32)
        output = MODULE.temporal_representations(
            global_feature, dense_feature
        )
        self.assertEqual(set(output), {
            "appearance_mean", "endpoint_arrow", "centered_trajectory",
            "velocity_trajectory", "speed_profile",
            "temporal_self_similarity", "dense_speed_profile",
            "dense_lag_profile",
        })
        self.assertEqual(output["appearance_mean"].shape, (8,))
        self.assertEqual(output["centered_trajectory"].shape, (136,))
        self.assertEqual(output["velocity_trajectory"].shape, (128,))
        for value in output.values():
            self.assertTrue(np.isfinite(value).all())
            self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0, places=5)

    def test_centering_is_invariant_to_static_offset(self) -> None:
        rng = np.random.default_rng(11)
        global_feature = rng.normal(size=(17, 8)).astype(np.float32)
        dense_feature = rng.normal(size=(17, 4, 8)).astype(np.float32)
        offset = rng.normal(size=(1, 8)).astype(np.float32)
        left = MODULE.temporal_representations(global_feature, dense_feature)
        right = MODULE.temporal_representations(
            global_feature + offset, dense_feature
        )
        np.testing.assert_allclose(
            left["centered_trajectory"], right["centered_trajectory"],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            left["velocity_trajectory"], right["velocity_trajectory"],
            atol=1.0e-6,
        )

    def test_endpoint_arrow_reverses_sign(self) -> None:
        rng = np.random.default_rng(13)
        global_feature = rng.normal(size=(17, 8)).astype(np.float32)
        dense_feature = rng.normal(size=(17, 4, 8)).astype(np.float32)
        forward = MODULE.temporal_representations(global_feature, dense_feature)
        reverse = MODULE.temporal_representations(
            global_feature[::-1].copy(), dense_feature[::-1].copy()
        )
        np.testing.assert_allclose(
            forward["endpoint_arrow"], -reverse["endpoint_arrow"],
            atol=1.0e-6,
        )

    def test_projection_score_uses_fit_span(self) -> None:
        basis = MODULE.orthonormal_basis([
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        ], rank=2)
        self.assertAlmostEqual(
            MODULE.projection_score(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32), basis
            ),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            MODULE.projection_score(
                np.asarray([0.0, 0.0, 1.0], dtype=np.float32), basis
            ),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
