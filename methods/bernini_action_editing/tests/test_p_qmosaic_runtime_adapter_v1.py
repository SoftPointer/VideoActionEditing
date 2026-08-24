#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import replace
import hashlib
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
    import p_qmosaic_runtime_adapter_v1 as subject  # noqa: E402
    import self_imagined_native_rv2v_hidden_vjp_v1 as qmosaic  # noqa: E402
else:
    subject = None  # type: ignore[assignment]
    qmosaic = None  # type: ignore[assignment]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sp4_clean_vjp(
    *,
    shape: tuple[int, int, int, int, int],
    query_seed: int,
    tensor_seed: int,
) -> object:
    generator = torch.Generator().manual_seed(tensor_seed)
    target = torch.randn(*shape, generator=generator, dtype=torch.float32)
    # Use non-identical pieces, but make the final SUM deterministic and rich
    # in temporal/spatial modes that survive the fixed nuisance quotient.
    pieces = (
        (0.10 * target).contiguous(),
        (0.20 * target).contiguous(),
        (0.30 * target).contiguous(),
        (0.40 * target).contiguous(),
    )
    rows = []
    for sp_rank, values in enumerate(pieces):
        row = qmosaic.RankLocalVJPRow(
            query_seed=query_seed,
            sp_rank=sp_rank,
            vjp_target="clean_latent",
            values=values,
            score_cotangent_receipt_digest=_digest(f"score-{sp_rank}"),
            editor_packet_receipt_digest=_digest(f"editor-{sp_rank}"),
            global_cotangent_identity_digest=_digest("global-cotangent"),
            value_sha256=qmosaic.tensor_sha256(
                values, label=f"fixture rank {sp_rank} clean VJP"
            ),
            value_norm=float(torch.linalg.vector_norm(values.double()).item()),
            replay_max_abs=float(values.abs().max().item()),
            parameter_state_sha256=_digest("parameter-state"),
        )
        rows.append(qmosaic._seal_rank_local_vjp_row(row))  # noqa: SLF001
    return qmosaic._sum_rank_local_vjp_rows_unsafe_for_test(rows)  # noqa: SLF001


def _base(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(
        *shape,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float32,
    ).detach()


def _build(cell_id: str = "dog") -> tuple[object, torch.Tensor, object]:
    shape = subject.REGISTERED_CELL_GEOMETRIES[cell_id]
    query_seed = subject.REGISTERED_QUERY_SEEDS[cell_id][0]
    row = _sp4_clean_vjp(
        shape=shape,
        query_seed=query_seed,
        tensor_seed=query_seed + 100,
    )
    base = _base(shape, query_seed + 200)
    result = subject.build_p_qmosaic_runtime_adaptation_v1(
        cell_id=cell_id,
        base_clean_latent=base,
        clean_vjp_row=row,
    )
    return result, base, row


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
class PQMosaicRuntimeAdapterTests(unittest.TestCase):
    def test_registered_dog_and_human_exact81_receipts_close(self) -> None:
        for cell_id in ("dog", "human"):
            with self.subTest(cell_id=cell_id):
                result, base, row = _build(cell_id)
                receipt = result.receipt()
                expected_shape = subject.REGISTERED_CELL_GEOMETRIES[cell_id]
                self.assertEqual(tuple(result.base.shape), expected_shape)
                self.assertEqual(tuple(result.plus.shape), expected_shape)
                self.assertEqual(tuple(result.minus.shape), expected_shape)
                self.assertEqual(receipt["evidence_tier"], "ENGINEERING_ONLY")
                self.assertEqual(
                    receipt["registered_cell"],
                    {
                        "cell_id": cell_id,
                        "query_seed_from_upstream_vjp": row.query_seed,
                        "query_seed_selected_by_adapter": False,
                        "frame_count": 81,
                        "latent_phases": 21,
                        "latent_shape": list(expected_shape),
                    },
                )
                self.assertEqual(receipt["relative_l2"]["dose"], 0.01)
                self.assertAlmostEqual(
                    receipt["relative_l2"]["observed"], 0.01, places=6
                )
                self.assertTrue(receipt["base_plus_minus_symmetry"]["passed"])
                self.assertTrue(
                    receipt["projection_occurs_after_real_clean_latent_vjp"]
                )
                self.assertTrue(
                    receipt["projection_occurs_before_normalization_and_dose"]
                )
                hashes = receipt["tensor_sha256"]
                self.assertEqual(
                    hashes["input_base_clean_latent"],
                    qmosaic.tensor_sha256(base, label="fixture base"),
                )
                self.assertEqual(
                    hashes["projector_owned_raw_clean_latent_vjp"],
                    row.receipt()["value_sha256"],
                )
                self.assertNotEqual(
                    hashes["projected_clean_latent_vjp"],
                    hashes["projector_owned_raw_clean_latent_vjp"],
                )
                self.assertFalse(receipt["semantic_success_assessed"])
                self.assertFalse(receipt["decode_publication_authorized"])
                self.assertFalse(receipt["lora_vjp_authorized"])
                self.assertFalse(receipt["training_update_authorized"])
                self.assertFalse(receipt["parameter_update"])
                self.assertFalse(receipt["scientific_authority"])
                self.assertTrue(all(
                    value is False for value in receipt["content_inputs"].values()
                ))
                unsigned = dict(receipt)
                digest = unsigned.pop("receipt_digest")
                self.assertEqual(
                    digest,
                    hashlib.sha256(
                        json.dumps(
                            unsigned,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                            allow_nan=False,
                        ).encode("ascii")
                    ).hexdigest(),
                )

    def test_adapter_reuses_frozen_projector_without_reimplementing_math(self) -> None:
        shape = subject.REGISTERED_CELL_GEOMETRIES["dog"]
        row = _sp4_clean_vjp(
            shape=shape,
            query_seed=subject.REGISTERED_QUERY_SEEDS["dog"][0],
            tensor_seed=9801,
        )
        base = _base(shape, 9802)
        with mock.patch.object(
            subject._projector,  # noqa: SLF001 - proves required reuse
            "construct_projected_symmetric_latents",
            wraps=subject._projector.construct_projected_symmetric_latents,
        ) as reused:
            result = subject.build_p_qmosaic_runtime_adaptation_v1(
                cell_id="dog",
                base_clean_latent=base,
                clean_vjp_row=row,
            )
        reused.assert_called_once()
        self.assertEqual(
            result.receipt()["projector_reused_without_math_copy"],
            (
                "p_qmosaic_nuisance_null_projector_v1."
                "construct_projected_symmetric_latents"
            ),
        )
        source = inspect.getsource(subject)
        for copied_formula_fragment in (
            "meshgrid(",
            ".mean(dim=2",
            "active_temporal_sum_max_abs",
            "spatial_affine_max_abs_coefficient",
        ):
            self.assertNotIn(copied_formula_fragment, source)

    def test_public_constructor_has_no_selection_or_content_support_inputs(
        self,
    ) -> None:
        parameters = inspect.signature(
            subject.build_p_qmosaic_runtime_adaptation_v1
        ).parameters
        self.assertEqual(
            tuple(parameters),
            ("cell_id", "base_clean_latent", "clean_vjp_row"),
        )
        forbidden = {
            "seed",
            "dose",
            "sign",
            "arm",
            "callback",
            "mask",
            "track",
            "pose",
            "flow",
            "box",
            "swept_tube",
        }
        self.assertTrue(forbidden.isdisjoint(parameters))
        self.assertEqual(subject.RELATIVE_L2_DOSE, 0.01)

    def test_wrong_cell_geometry_seed_and_unsealed_vjp_fail_closed(self) -> None:
        dog_shape = subject.REGISTERED_CELL_GEOMETRIES["dog"]
        dog_base = _base(dog_shape, 9810)
        good_row = _sp4_clean_vjp(
            shape=dog_shape,
            query_seed=subject.REGISTERED_QUERY_SEEDS["dog"][0],
            tensor_seed=9811,
        )
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError, "pre-registered"
        ):
            subject.build_p_qmosaic_runtime_adaptation_v1(
                cell_id="cat",
                base_clean_latent=dog_base,
                clean_vjp_row=good_row,
            )
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError, "registered detached FP32"
        ):
            subject.build_p_qmosaic_runtime_adaptation_v1(
                cell_id="dog",
                base_clean_latent=dog_base[..., :-1],
                clean_vjp_row=good_row,
            )
        wrong_seed = _sp4_clean_vjp(
            shape=dog_shape,
            query_seed=2026081599,
            tensor_seed=9812,
        )
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError, "selected exact81"
        ):
            subject.build_p_qmosaic_runtime_adaptation_v1(
                cell_id="dog",
                base_clean_latent=dog_base,
                clean_vjp_row=wrong_seed,
            )
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError, "sealed SP4SummedVJPRow"
        ):
            subject.build_p_qmosaic_runtime_adaptation_v1(
                cell_id="dog",
                base_clean_latent=dog_base,
                clean_vjp_row=object(),
            )

    def test_input_base_and_upstream_vjp_mutations_deny_receipt(self) -> None:
        result, base, _row = _build("dog")
        with torch.no_grad():
            base.data.add_(1.0)
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError, "input base clean latent changed"
        ):
            result.receipt()

        result, _base_input, row = _build("dog")
        with torch.no_grad():
            row.values.data.mul_(0.5)
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError, "upstream SP4.*not live"
        ):
            result.receipt()

    def test_projected_direction_delta_and_arm_mutations_deny_receipt(self) -> None:
        for role in (
            "projected_clean_latent_vjp",
            "unit_projected_direction",
            "delta",
            "plus",
            "minus",
        ):
            with self.subTest(role=role):
                result, _base_input, _row = _build("dog")
                target = getattr(result, role)
                with torch.no_grad():
                    target.data.reshape(-1)[0].add_(1.0)
                with self.assertRaisesRegex(
                    subject.PQMosaicRuntimeAdapterError,
                    "nested projected intervention changed",
                ):
                    result.receipt()

    def test_control_tamper_and_forged_dataclass_copy_deny_receipt(self) -> None:
        result, _base_input, _row = _build("dog")
        object.__setattr__(result, "query_seed", 2026081503)
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError,
            "upstream clean-latent VJP provenance changed",
        ):
            result.receipt()

        valid, _base_input, _row = _build("dog")
        forged = replace(valid)
        with self.assertRaisesRegex(
            subject.PQMosaicRuntimeAdapterError, "closed constructor"
        ):
            forged.receipt()


if __name__ == "__main__":
    unittest.main()
