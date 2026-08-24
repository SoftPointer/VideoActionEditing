#!/usr/bin/env python3

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock


try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - dependency-light hosts
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

if TORCH_AVAILABLE:
    import p_qmosaic_nuisance_null_projector_v1 as subject  # noqa: E402
else:
    subject = None  # type: ignore[assignment]


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
class PQMosaicProjectionTests(unittest.TestCase):
    @staticmethod
    def _affine_basis(height: int, width: int) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)
        x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        basis = torch.stack((torch.ones_like(xx), xx, yy), dim=0).reshape(3, -1)
        return basis / torch.linalg.vector_norm(basis, dim=1, keepdim=True)

    @classmethod
    def _independent_projection(cls, raw: torch.Tensor) -> torch.Tensor:
        raw64 = raw.to(torch.float64)
        active = raw64[:, :, 1:]
        active = active - active.mean(dim=2, keepdim=True)
        temporal = torch.cat((torch.zeros_like(raw64[:, :, :1]), active), dim=2)
        height, width = int(raw.shape[-2]), int(raw.shape[-1])
        basis = cls._affine_basis(height, width)
        flat = temporal.reshape(-1, height * width)
        return (flat - (flat @ basis.T) @ basis).reshape_as(raw64)

    def test_projection_reuses_old_kernel_and_independently_closes_all_nulls(self) -> None:
        generator = torch.Generator().manual_seed(9701)
        raw = torch.randn(1, 16, 21, 6, 7, generator=generator)
        raw_before = raw.clone()
        with mock.patch.object(
            subject._relational,  # noqa: SLF001 - proves the required reuse
            "project_source_safe_cotangent",
            wraps=subject._relational.project_source_safe_cotangent,
        ) as reused:
            result = subject.project_raw_clean_latent_vjp(raw)
        reused.assert_called_once_with(
            raw,
            minimum_survival_ratio=subject.MINIMUM_PROJECTION_SURVIVAL_RATIO,
        )
        self.assertTrue(torch.equal(raw, raw_before))
        self.assertEqual(tuple(result.tensor.shape), tuple(raw.shape))
        self.assertEqual(result.tensor.dtype, torch.float32)
        self.assertFalse(result.tensor.requires_grad)
        self.assertTrue(
            torch.equal(
                result.tensor[:, :, 0], torch.zeros_like(result.tensor[:, :, 0])
            )
        )
        self.assertLess(
            float(result.tensor.double()[:, :, 1:].sum(dim=2).abs().max()),
            3.0e-6,
        )
        affine = (
            result.tensor.double().reshape(-1, 6 * 7)
            @ self._affine_basis(6, 7).T
        )
        self.assertLess(float(affine.abs().max()), 3.0e-6)
        expected = self._independent_projection(raw)
        relative = float(
            torch.linalg.vector_norm(result.tensor.double() - expected)
            / torch.linalg.vector_norm(expected)
        )
        self.assertLess(relative, 1.0e-6)

        receipt = result.receipt()
        self.assertEqual(
            receipt["schema_version"], subject.PROJECTION_SCHEMA_VERSION
        )
        self.assertFalse(receipt["scientific_authority"])
        self.assertFalse(receipt["update"])
        self.assertFalse(receipt["parameter_update"])
        self.assertEqual(
            receipt["projector"]["fixed_nulls"],
            [
                "phase0",
                "active_phases_1_to_20_temporal_dc",
                "per_channel_phase_spatial_affine_1_x_y",
            ],
        )
        self.assertEqual(
            receipt["content_inputs"],
            {
                "mask": False,
                "track": False,
                "pose": False,
                "flow": False,
                "box": False,
                "content_derived_spatial_support": False,
            },
        )
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(
            digest,
            __import__("hashlib").sha256(
                json.dumps(
                    unsigned,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest(),
        )

    def test_removed_subspace_energy_is_mutually_exclusive_and_closes(self) -> None:
        generator = torch.Generator().manual_seed(9702)
        raw = torch.randn(1, 16, 21, 5, 6, generator=generator)
        result = subject.project_raw_clean_latent_vjp(raw)
        receipt = result.receipt()
        energies = receipt["removed_subspace_l2_energy"]
        self.assertGreater(energies["phase0"], 0.0)
        self.assertGreater(energies["active_temporal_dc_after_phase0"], 0.0)
        self.assertGreater(
            energies["spatial_affine_1_x_y_after_phase0_and_temporal_dc"], 0.0
        )
        self.assertTrue(
            math.isclose(
                energies["total_without_double_counting"],
                energies["phase0"]
                + energies["active_temporal_dc_after_phase0"]
                + energies[
                    "spatial_affine_1_x_y_after_phase0_and_temporal_dc"
                ],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
        )
        residuals = receipt["orthogonality_residuals"]
        self.assertLess(residuals["energy_closure_relative_error"], 2.0e-6)
        self.assertLess(
            residuals["removed_projected_dot_over_raw_energy"], 2.0e-6
        )
        self.assertLess(
            residuals["nuisance_component_pairwise_dot_over_raw_energy"],
            2.0e-6,
        )

    def test_project_then_normalize_keeps_the_raw_relative_l2_dose(self) -> None:
        generator = torch.Generator().manual_seed(9703)
        base = torch.randn(1, 16, 21, 5, 7, generator=generator)
        raw = torch.randn(1, 16, 21, 5, 7, generator=generator)
        result = subject.construct_projected_symmetric_latents(
            base_clean_latent=base,
            raw_clean_latent_vjp=raw,
        )

        projected = result.projection.tensor
        projected_norm = torch.linalg.vector_norm(projected)
        expected_direction = (projected / projected_norm).contiguous()
        raw_formula_scale = (
            torch.tensor(subject.RELATIVE_L2_DOSE, dtype=torch.float32)
            * torch.linalg.vector_norm(base)
        )
        expected_delta = (raw_formula_scale * expected_direction).contiguous()
        self.assertTrue(torch.equal(result.unit_direction, expected_direction))
        self.assertTrue(torch.equal(result.delta, expected_delta))
        self.assertTrue(torch.equal(result.plus, (base + expected_delta).contiguous()))
        self.assertTrue(torch.equal(result.minus, (base - expected_delta).contiguous()))
        self.assertTrue(
            math.isclose(
                float(torch.linalg.vector_norm(result.delta.double())
                      / torch.linalg.vector_norm(base.double())),
                subject.RELATIVE_L2_DOSE,
                rel_tol=2.0e-6,
                abs_tol=2.0e-8,
            )
        )
        raw_direction_delta = raw_formula_scale * (
            raw / torch.linalg.vector_norm(raw)
        )
        self.assertTrue(
            math.isclose(
                float(torch.linalg.vector_norm(result.delta.double())),
                float(torch.linalg.vector_norm(raw_direction_delta.double())),
                rel_tol=2.0e-6,
                abs_tol=2.0e-8,
            )
        )
        self.assertFalse(torch.equal(result.unit_direction, raw / raw.norm()))
        receipt = result.receipt()
        self.assertEqual(receipt["relative_l2_dose"], 0.01)
        self.assertTrue(receipt["projection_precedes_normalization"])
        self.assertTrue(receipt["same_relative_l2_dose_as_raw_qmosaic"])
        self.assertFalse(receipt["seed_selection"])
        self.assertFalse(receipt["dose_selection"])
        self.assertFalse(receipt["arm_selection"])
        self.assertFalse(receipt["scientific_authority"])
        self.assertFalse(receipt["update"])

    def test_public_signatures_have_no_seed_dose_or_support_selection(self) -> None:
        projection_parameters = inspect.signature(
            subject.project_raw_clean_latent_vjp
        ).parameters
        intervention_parameters = inspect.signature(
            subject.construct_projected_symmetric_latents
        ).parameters
        self.assertEqual(list(projection_parameters), ["raw_clean_latent_vjp"])
        self.assertEqual(
            list(intervention_parameters),
            ["base_clean_latent", "raw_clean_latent_vjp"],
        )
        forbidden = {"seed", "dose", "mask", "track", "pose", "flow", "box"}
        self.assertTrue(forbidden.isdisjoint(projection_parameters))
        self.assertTrue(forbidden.isdisjoint(intervention_parameters))
        self.assertEqual(subject.RELATIVE_L2_DOSE, 0.01)

    def test_registered_exact81_latent_geometries_pass_without_dose_change(self) -> None:
        for seed, shape in (
            (2026081502, (1, 16, 21, 60, 62)),
            (2026081505, (1, 16, 21, 64, 58)),
        ):
            with self.subTest(shape=shape):
                generator = torch.Generator().manual_seed(seed)
                base = torch.randn(*shape, generator=generator)
                raw = torch.randn(*shape, generator=generator)
                result = subject.construct_projected_symmetric_latents(
                    base_clean_latent=base,
                    raw_clean_latent_vjp=raw,
                )
                self.assertEqual(tuple(result.plus.shape), shape)
                self.assertTrue(
                    math.isclose(
                        result.observed_relative_l2_dose,
                        subject.RELATIVE_L2_DOSE,
                        rel_tol=5.0e-5,
                        abs_tol=5.0e-7,
                    )
                )
                self.assertLess(
                    result.projection.energy_closure_relative_error, 2.0e-6
                )

    def test_degenerate_nonfinite_and_geometry_mismatch_fail_closed(self) -> None:
        good = torch.randn(1, 16, 21, 4, 5)
        cases = [
            good.double(),
            torch.randn(2, 16, 21, 4, 5),
            torch.randn(1, 15, 21, 4, 5),
            torch.randn(1, 16, 20, 4, 5),
            torch.randn(1, 16, 21, 1, 5),
            good.clone().requires_grad_(True),
        ]
        nonfinite = good.clone()
        nonfinite[0, 0, 0, 0, 0] = float("nan")
        cases.append(nonfinite)
        for case in cases:
            with self.subTest(shape=tuple(case.shape), dtype=case.dtype):
                with self.assertRaises(subject.PQMosaicProjectionError):
                    subject.project_raw_clean_latent_vjp(case)

        fully_removed = torch.ones(1, 16, 21, 4, 5)
        with self.assertRaisesRegex(
            subject.PQMosaicProjectionError, "failed closed"
        ):
            subject.project_raw_clean_latent_vjp(fully_removed)

        with self.assertRaisesRegex(
            subject.PQMosaicProjectionError, "geometry/device differ"
        ):
            subject.construct_projected_symmetric_latents(
                base_clean_latent=torch.randn(1, 16, 21, 4, 6),
                raw_clean_latent_vjp=good,
            )

    def test_wrong_reused_projector_output_cannot_bypass_independent_formula(self) -> None:
        raw = torch.randn(1, 16, 21, 4, 5)
        with mock.patch.object(
            subject._relational,  # noqa: SLF001 - deliberate fault injection
            "project_source_safe_cotangent",
            return_value=raw.clone(),
        ):
            with self.assertRaisesRegex(
                subject.PQMosaicProjectionError, "independent fixed-null formula"
            ):
                subject.project_raw_clean_latent_vjp(raw)

        wrong_shape = torch.randn(1, 16, 21, 4, 6)
        with mock.patch.object(
            subject._relational,  # noqa: SLF001 - deliberate fault injection
            "project_source_safe_cotangent",
            return_value=wrong_shape,
        ):
            with self.assertRaisesRegex(
                subject.PQMosaicProjectionError, "geometry or device differs"
            ):
                subject.project_raw_clean_latent_vjp(raw)

    def test_plus_inplace_mutation_invalidates_the_construction_seal(self) -> None:
        generator = torch.Generator().manual_seed(9711)
        result = subject.construct_projected_symmetric_latents(
            base_clean_latent=torch.randn(
                1, 16, 21, 4, 5, generator=generator
            ),
            raw_clean_latent_vjp=torch.randn(
                1, 16, 21, 4, 5, generator=generator
            ),
        )
        before = result.receipt()
        bindings = before["tensor_bindings"]
        self.assertEqual(
            set(bindings),
            {
                "base_clean_latent",
                "unit_projected_direction",
                "projected_delta",
                "plus_clean_latent",
                "minus_clean_latent",
            },
        )
        self.assertEqual(len(bindings["plus_clean_latent"]["tensor_sha256"]), 64)
        self.assertEqual(bindings["plus_clean_latent"]["dtype"], "torch.float32")
        self.assertEqual(
            bindings["plus_clean_latent"]["shape"], [1, 16, 21, 4, 5]
        )
        result.plus.zero_()
        with self.assertRaisesRegex(
            subject.PQMosaicProjectionError,
            "plus clean latent changed after construction",
        ):
            result.receipt()

    def test_projected_tensor_inplace_mutation_invalidates_nested_and_outer_receipts(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(9712)
        result = subject.construct_projected_symmetric_latents(
            base_clean_latent=torch.randn(
                1, 16, 21, 4, 5, generator=generator
            ),
            raw_clean_latent_vjp=torch.randn(
                1, 16, 21, 4, 5, generator=generator
            ),
        )
        projection_before = result.projection.receipt()
        self.assertEqual(
            set(projection_before["tensor_bindings"]),
            {"raw_clean_latent_vjp", "projected_clean_latent_vjp"},
        )
        self.assertEqual(result.receipt(), result.receipt())
        result.projection.tensor.fill_(0.0)
        with self.assertRaisesRegex(
            subject.PQMosaicProjectionError,
            "projected clean-latent VJP changed after construction",
        ):
            result.projection.receipt()
        with self.assertRaisesRegex(
            subject.PQMosaicProjectionError,
            "projected clean-latent VJP changed after construction",
        ):
            result.receipt()

    def test_raw_vjp_mutation_is_not_hidden_by_the_owned_projection(self) -> None:
        raw = torch.randn(1, 16, 21, 4, 5)
        projection = subject.project_raw_clean_latent_vjp(raw)
        projection.receipt()
        raw.add_(1.0)
        with self.assertRaisesRegex(
            subject.PQMosaicProjectionError,
            "raw clean-latent VJP changed after construction",
        ):
            projection.receipt()


if __name__ == "__main__":
    unittest.main()
