#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import t2v_v2v_branch_homotopy_v1 as homotopy  # noqa: E402


class StaticContractTests(unittest.TestCase):
    def test_pinned_schedule_and_branch_semantics(self) -> None:
        self.assertEqual(homotopy.SIGMA_LOW, 0.75)
        self.assertEqual(homotopy.SIGMA_HIGH, 0.95)
        self.assertEqual(
            tuple(homotopy.T2VV2VBranchHomotopyStep.__dataclass_fields__),
            (
                "velocity",
                "sigma",
                "high_pure_t2v_weight",
                "low_source_v2v_weight",
                "endpoint",
            ),
        )

    def test_core_has_no_model_or_integrator_call(self) -> None:
        source = Path(homotopy.__file__).read_text(encoding="utf-8")
        self.assertNotIn("scheduler.step(", source)
        self.assertNotIn("shared_step(", source)
        self.assertNotIn("normalized_guidance", source)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is required",
)
class TensorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        torch.set_num_threads(1)
        cls.torch = torch

    def setUp(self) -> None:
        torch = self.torch
        self.target = torch.zeros((1, 3, 4), dtype=torch.bfloat16)
        self.high = torch.full((1, 3, 4), 2.0, dtype=torch.bfloat16)
        self.low = torch.zeros((1, 3, 4), dtype=torch.bfloat16)

    def test_low_endpoint_returns_stock_v2v_tensor_directly(self) -> None:
        result = homotopy.t2v_v2v_branch_homotopy_step(
            self.target,
            self.high,
            self.low,
            self.torch.tensor(0.75, dtype=self.torch.float32),
        )
        self.assertIs(result.velocity, self.low)
        self.assertEqual(result.endpoint, "low_source_v2v_apg")
        self.assertEqual(result.high_pure_t2v_weight, 0.0)
        self.assertTrue(result.trace_dict()["endpoint_exact"])

    def test_high_endpoint_returns_pure_t2v_tensor_directly(self) -> None:
        result = homotopy.t2v_v2v_branch_homotopy_step(
            self.target,
            self.high,
            self.low,
            self.torch.tensor(0.95, dtype=self.torch.float32),
        )
        self.assertIs(result.velocity, self.high)
        self.assertEqual(result.endpoint, "high_pure_t2v_apg")
        self.assertEqual(result.high_pure_t2v_weight, 1.0)

    def test_midpoint_is_fp32_lerp_cast_to_branch_dtype(self) -> None:
        torch = self.torch
        result = homotopy.t2v_v2v_branch_homotopy_step(
            self.target,
            self.high,
            self.low,
            torch.tensor(0.85, dtype=torch.float32),
        )
        self.assertEqual(result.endpoint, "transition")
        self.assertAlmostEqual(result.high_pure_t2v_weight, 0.5, places=6)
        self.assertEqual(result.velocity.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                result.velocity,
                torch.ones_like(result.velocity, dtype=torch.bfloat16),
            )
        )
        self.assertFalse(result.trace_dict()["endpoint_exact"])

    def test_exact40_live_sigmas_have_preregistered_regions(self) -> None:
        import source_self_native_ref_contrastive_v3 as schedule

        regions = []
        for value in schedule.NATIVE_UNIPC40_SIGMAS:
            weight = homotopy.smoothstep_pure_t2v_weight(
                self.torch.tensor(value, dtype=self.torch.float32)
            )
            numeric = float(weight.item())
            regions.append("high" if numeric == 1.0 else "low" if numeric == 0.0 else "transition")
        self.assertEqual(
            regions,
            ["high"] * 9 + ["transition"] * 17 + ["low"] * 14,
        )

    def test_tensor_wrapper_matches_full_step(self) -> None:
        sigma = self.torch.tensor(0.87, dtype=self.torch.float32)
        expected = homotopy.t2v_v2v_branch_homotopy_step(
            self.target, self.high, self.low, sigma
        ).velocity
        actual = homotopy.combine_t2v_v2v_apg_velocities(
            self.target, self.high, self.low, sigma
        )
        self.assertTrue(self.torch.equal(expected, actual))

    def test_invalid_geometry_dtype_sigma_and_momentum_fail(self) -> None:
        torch = self.torch
        with self.assertRaisesRegex(
            homotopy.T2VV2VBranchHomotopyError,
            "target packed-state shape",
        ):
            homotopy.t2v_v2v_branch_homotopy_step(
                self.target,
                self.high[:, :-1],
                self.low,
                torch.tensor(0.8, dtype=torch.float32),
            )
        with self.assertRaisesRegex(
            homotopy.T2VV2VBranchHomotopyError,
            "scheduler-bound dtype",
        ):
            homotopy.t2v_v2v_branch_homotopy_step(
                self.target,
                self.high.float(),
                self.low,
                torch.tensor(0.8, dtype=torch.float32),
            )
        for sigma in (
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(float("nan"), dtype=torch.float32),
            torch.tensor(0.8, dtype=torch.float64),
            torch.tensor([0.8], dtype=torch.float32),
        ):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(
                    homotopy.T2VV2VBranchHomotopyError,
                    "finite positive FP32 scalar",
                ):
                    homotopy.t2v_v2v_branch_homotopy_step(
                        self.target, self.high, self.low, sigma
                    )
        with self.assertRaisesRegex(
            homotopy.T2VV2VBranchHomotopyError,
            "pure-T2V APG momentum",
        ):
            homotopy.t2v_v2v_branch_homotopy_step(
                self.target,
                self.high,
                self.low,
                torch.tensor(0.8, dtype=torch.float32),
                high_pure_t2v_momentum=0.1,
            )

    def test_nonfinite_inactive_branch_is_still_rejected(self) -> None:
        high = self.high.clone()
        high[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(
            homotopy.T2VV2VBranchHomotopyError,
            "high pure-T2V APG velocity",
        ):
            homotopy.t2v_v2v_branch_homotopy_step(
                self.target,
                high,
                self.low,
                self.torch.tensor(0.70, dtype=self.torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
