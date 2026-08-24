#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import native_branch_homotopy_v1 as homotopy  # noqa: E402


class StaticContractTests(unittest.TestCase):
    def test_pinned_schedule_and_branch_semantics(self) -> None:
        self.assertEqual(homotopy.SIGMA_LOW, 0.60)
        self.assertEqual(homotopy.SIGMA_HIGH, 0.90)
        fields = homotopy.NativeBranchHomotopyStep.__dataclass_fields__
        self.assertEqual(
            tuple(fields),
            (
                "velocity",
                "sigma",
                "high_r2v4_weight",
                "low_official_v2v_apg_weight",
                "endpoint",
            ),
        )

    def test_core_has_no_integrator_call(self) -> None:
        source = Path(homotopy.__file__).read_text(encoding="utf-8")
        self.assertNotIn("scheduler.step(", source)
        self.assertNotIn("set_timesteps(", source)
        self.assertNotIn("normalized_guidance(", source)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "PyTorch is required",
)
class NativeBranchHomotopyTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def setUp(self) -> None:
        torch = self.torch
        self.target = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(
            1, 6, 4
        )
        self.high = torch.full((1, 6, 4), 3.0, dtype=torch.bfloat16)
        self.low = torch.full((1, 6, 4), -1.0, dtype=torch.bfloat16)

    def test_smoothstep_is_fp32_with_exact_endpoints_and_midpoint(self) -> None:
        torch = self.torch
        low = homotopy.smoothstep_high_branch_weight(
            torch.tensor(0.60, dtype=torch.float32)
        )
        high = homotopy.smoothstep_high_branch_weight(
            torch.tensor(0.90, dtype=torch.float32)
        )
        midpoint = homotopy.smoothstep_high_branch_weight(
            torch.tensor(0.75, dtype=torch.float32)
        )
        self.assertEqual(low.dtype, torch.float32)
        self.assertEqual(high.dtype, torch.float32)
        self.assertEqual(midpoint.dtype, torch.float32)
        self.assertEqual(float(low.item()), 0.0)
        self.assertEqual(float(high.item()), 1.0)
        self.assertAlmostEqual(float(midpoint.item()), 0.5, places=6)

    def test_low_endpoint_returns_official_v2v_tensor_directly(self) -> None:
        torch = self.torch
        result = homotopy.native_branch_homotopy_step(
            self.target,
            self.high,
            self.low,
            torch.tensor(0.25, dtype=torch.float32),
        )
        self.assertIs(result.velocity, self.low)
        self.assertEqual(result.endpoint, "low_official_v2v_apg")
        self.assertEqual(result.high_r2v4_weight, 0.0)
        self.assertEqual(result.low_official_v2v_apg_weight, 1.0)
        self.assertTrue(result.trace_dict()["endpoint_exact"])

    def test_high_endpoint_returns_r2v4_tensor_directly(self) -> None:
        torch = self.torch
        result = homotopy.native_branch_homotopy_step(
            self.target,
            self.high,
            self.low,
            torch.tensor(1.0, dtype=torch.float32),
        )
        self.assertIs(result.velocity, self.high)
        self.assertEqual(result.endpoint, "high_r2v4_apg")
        self.assertEqual(result.high_r2v4_weight, 1.0)
        self.assertEqual(result.low_official_v2v_apg_weight, 0.0)

    def test_transition_is_fp32_lerp_cast_to_common_branch_dtype(self) -> None:
        torch = self.torch
        result = homotopy.native_branch_homotopy_step(
            self.target,
            self.high,
            self.low,
            torch.tensor(0.75, dtype=torch.float32),
        )
        self.assertEqual(result.endpoint, "transition")
        self.assertEqual(result.velocity.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                result.velocity,
                torch.ones_like(result.velocity, dtype=torch.bfloat16),
            )
        )
        trace = result.trace_dict()
        self.assertEqual(
            trace["high_branch"],
            "references_only_r2v4_apg",
        )
        self.assertEqual(
            trace["low_branch"],
            "official_full_source_plus_four_refs_v2v_apg",
        )
        self.assertEqual(trace["interpolation_dtype"], "float32")
        self.assertFalse(trace["endpoint_exact"])

    def test_tensor_only_wrapper_matches_step(self) -> None:
        torch = self.torch
        sigma = torch.tensor(0.70, dtype=torch.float32)
        expected = homotopy.native_branch_homotopy_step(
            self.target,
            self.high,
            self.low,
            sigma,
        ).velocity
        actual = homotopy.combine_native_apg_velocities(
            self.target,
            self.high,
            self.low,
            sigma,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_shape_mismatch_is_rejected(self) -> None:
        torch = self.torch
        with self.assertRaisesRegex(
            homotopy.NativeBranchHomotopyError,
            "target packed-state shape",
        ):
            homotopy.native_branch_homotopy_step(
                self.target,
                self.high[:, :-1],
                self.low,
                torch.tensor(0.75, dtype=torch.float32),
            )

    def test_nonfinite_branch_is_rejected_even_at_inactive_endpoint(self) -> None:
        torch = self.torch
        high = self.high.clone()
        high[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(
            homotopy.NativeBranchHomotopyError,
            "high R2V-4 APG velocity",
        ):
            homotopy.native_branch_homotopy_step(
                self.target,
                high,
                self.low,
                torch.tensor(0.25, dtype=torch.float32),
            )

    def test_branch_dtype_mismatch_is_rejected(self) -> None:
        torch = self.torch
        with self.assertRaisesRegex(
            homotopy.NativeBranchHomotopyError,
            "share one scheduler-bound dtype",
        ):
            homotopy.native_branch_homotopy_step(
                self.target,
                self.high.float(),
                self.low,
                torch.tensor(0.75, dtype=torch.float32),
            )

    def test_sigma_must_be_positive_finite_fp32_scalar(self) -> None:
        torch = self.torch
        invalid = (
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(-0.1, dtype=torch.float32),
            torch.tensor(float("nan"), dtype=torch.float32),
            torch.tensor(0.75, dtype=torch.float64),
            torch.tensor([0.75], dtype=torch.float32),
        )
        for sigma in invalid:
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(
                    homotopy.NativeBranchHomotopyError,
                    "finite positive FP32 scalar",
                ):
                    homotopy.native_branch_homotopy_step(
                        self.target,
                        self.high,
                        self.low,
                        sigma,
                    )

    def test_both_apg_momenta_must_be_exactly_zero(self) -> None:
        torch = self.torch
        sigma = torch.tensor(0.75, dtype=torch.float32)
        with self.assertRaisesRegex(
            homotopy.NativeBranchHomotopyError,
            "high R2V-4 APG momentum",
        ):
            homotopy.native_branch_homotopy_step(
                self.target,
                self.high,
                self.low,
                sigma,
                high_r2v4_momentum=0.1,
            )
        with self.assertRaisesRegex(
            homotopy.NativeBranchHomotopyError,
            "low official v2v_apg momentum",
        ):
            homotopy.native_branch_homotopy_step(
                self.target,
                self.high,
                self.low,
                sigma,
                low_official_v2v_apg_momentum=0.1,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is required",
    )
    def test_device_mismatch_is_rejected_when_accelerator_exists(self) -> None:
        torch = self.torch
        if torch.cuda.is_available():
            other = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            other = torch.device("mps")
        else:
            self.skipTest("no second torch device")
        with self.assertRaisesRegex(
            homotopy.NativeBranchHomotopyError,
            "target packed-state device",
        ):
            homotopy.native_branch_homotopy_step(
                self.target,
                self.high.to(other),
                self.low,
                torch.tensor(0.75, dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
