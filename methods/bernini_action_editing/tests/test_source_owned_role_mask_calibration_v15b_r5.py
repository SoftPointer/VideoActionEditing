from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_owned_role_mask_calibration_v15b_r5 as r5  # noqa: E402


class V15BR5CalibrationTests(unittest.TestCase):
    def test_spatial_standardization_is_dc_and_positive_scale_invariant(self):
        rng = np.random.default_rng(7)
        value = rng.normal(size=(2, 3, 37, 25)).astype(np.float32)
        base = r5.spatial_median_mad_standardize(value)
        mutated = r5.spatial_median_mad_standardize(value * np.float32(3.25) + np.float32(19.0))
        np.testing.assert_allclose(base, mutated, atol=2e-5, rtol=2e-5)

    def test_constant_maps_never_force_nonempty(self):
        role_maps = np.zeros((5, 5, 21, 37, 25), dtype=np.float32)
        result = r5.calibrate_source_role_maps(
            role_maps, null_span_maps=None, null_registry_sha256=None
        )
        self.assertFalse(result.exploratory_track_masks.any())
        self.assertFalse(result.strict_block_masks.any())
        self.assertFalse(result.strict_aggregate_masks.any())
        self.assertEqual(result.receipt["status"], "strict_fail_null_token_bank_absent")
        self.assertFalse(result.receipt["track_policy"]["fixed_quota"])
        self.assertFalse(result.receipt["track_policy"]["forced_nonempty"])

    def test_missing_null_bank_cannot_emit_strict_mask_even_with_strong_track(self):
        role_maps = np.random.default_rng(17).normal(
            scale=0.05, size=(5, 5, 21, 37, 25)
        ).astype(np.float32)
        for phase in range(21):
            role_maps[1, 1, phase, 10, 10] = 20.0
            role_maps[1, 1, phase, 10, 11] = 8.0
        result = r5.calibrate_source_role_maps(
            role_maps, null_span_maps=None, null_registry_sha256=None
        )
        self.assertTrue(result.exploratory_track_masks.any())
        self.assertFalse(result.strict_aggregate_masks.any())
        self.assertFalse(result.receipt["mechanical_candidate_qualified"])
        self.assertFalse(result.receipt["legacy_averaged_null_consumed"])

    def test_vessel_competition_is_permutation_equivariant(self):
        rng = np.random.default_rng(11)
        maps = rng.normal(size=(5, 21, 37, 25)).astype(np.float32)
        winners = r5.vessel_standardized_winners(maps)
        # Swap old/new rows and their names; semantic winner labels must agree.
        perm = [0, 2, 1, 3, 4]
        names = tuple(r5.ROLE_NAMES[index] for index in perm)
        permuted = r5.vessel_standardized_winners(maps[perm], role_names=names)
        # Output offsets are canonical VESSEL_ROLES, independent of row order.
        np.testing.assert_array_equal(winners, permuted)

    def test_calibration_source_has_no_spatial_audit_input(self):
        source = inspect.getsource(r5.calibrate_source_role_maps).lower()
        self.assertNotIn("roi", source)
        self.assertNotIn("bounding", source)
        self.assertEqual(r5.ROLE_NAMES, ("agent", "old_actor", "new_actor", "recipient", "support"))

    def test_malformed_null_bank_fails_closed(self):
        role_maps = np.zeros((5, 5, 21, 37, 25), dtype=np.float32)
        malformed = np.zeros((5, 63, 21, 37, 25), dtype=np.float32)
        with self.assertRaisesRegex(r5.V15BR5CalibrationError, "null span maps"):
            r5.calibrate_source_role_maps(
                role_maps,
                null_span_maps=malformed,
                null_registry_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
