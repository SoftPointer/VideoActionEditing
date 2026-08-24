from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


try:
    import torch
except ModuleNotFoundError as error:  # lightweight local control environment
    raise unittest.SkipTest("real-source materializer tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import materialize_complex8_real_source_latents_v1 as materializer


class Complex8RealSourceMaterializerTest(unittest.TestCase):
    def test_raw_posterior_mode_is_normalized_exactly_once(self) -> None:
        mean = torch.tensor([1.0, -2.0]).reshape(2, 1, 1, 1)
        std = torch.tensor([2.0, 4.0]).reshape(2, 1, 1, 1)
        expected = torch.full((1, 2, 21, 1, 1), 3.0)
        raw_mode = mean.unsqueeze(0) + std.unsqueeze(0) * expected
        clean = materializer.normalize_posterior_mode(raw_mode, mean, std)
        self.assertTrue(torch.equal(clean, expected))

    def test_materializer_does_not_call_already_normalized_private_helper(self) -> None:
        source = inspect.getsource(materializer.main)
        self.assertNotIn("from bernini.pipeline import _vae_encode", source)
        self.assertIn("vae.encode(", source)
        self.assertIn("normalize_posterior_mode(raw_mode, mean, std)", source)

    def test_manifest_and_rows_record_one_normalization_application(self) -> None:
        source = inspect.getsource(materializer.main)
        self.assertEqual(source.count('"normalization_application_count": 1'), 2)
        # Retain the established trainer-facing field while making the audit
        # semantics explicit in both the manifest and every row.
        self.assertEqual(source.count('"normalization_count": 1'), 2)

    def test_invalid_statistics_fail_closed(self) -> None:
        raw_mode = torch.zeros((1, 2, 21, 1, 1))
        mean = torch.zeros((2, 1, 1, 1))
        bad_std = torch.tensor([1.0, 0.0]).reshape(2, 1, 1, 1)
        with self.assertRaises(materializer.MaterializationError):
            materializer.normalize_posterior_mode(raw_mode, mean, bad_std)


if __name__ == "__main__":
    unittest.main()
