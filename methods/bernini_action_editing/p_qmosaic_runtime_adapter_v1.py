#!/usr/bin/env python3
"""Closed runtime adapter for the P-Q-MOSAIC exact81 engineering canary.

The adapter is deliberately separate from the authenticated Q-MOSAIC
materializer, runner, and launcher.  It is intended to replace exactly one
future call site *after* the real WORLD4/SP4 clean-latent VJP has been summed
and *before* direction normalization and dose construction::

    SP4 SUM clean-latent VJP
        -> fixed nuisance-null projection
        -> normalization
        -> fixed relative-L2 dose 0.01
        -> base / plus / minus

The nuisance-null mathematics is not implemented here.  This module calls the
already frozen :mod:`p_qmosaic_nuisance_null_projector_v1` implementation,
then binds its live receipts to the authenticated ``SP4SummedVJPRow`` and the
two pre-registered exact81 cell geometries.

There is no mask, track, pose, optical flow, detector box, seed choice, dose
choice, arm choice, semantic evaluator, optimizer, or parameter-update path.
A valid receipt is ENGINEERING_ONLY evidence and cannot authorize a semantic
claim, decode publication, LoRA VJP, training, or an update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import torch

import p_qmosaic_nuisance_null_projector_v1 as _projector
import self_imagined_native_rv2v_hidden_vjp_v1 as _qmosaic


RUNTIME_ADAPTER_SCHEMA_VERSION = "bernini-p-qmosaic-runtime-adapter-v1"
EVIDENCE_TIER = "ENGINEERING_ONLY"
FRAME_COUNT = 81
LATENT_PHASES = 21
WORLD_SIZE = 4
SP_SIZE = 4
RELATIVE_L2_DOSE = _projector.RELATIVE_L2_DOSE

REGISTERED_CELL_GEOMETRIES = MappingProxyType(
    {
        "dog": (1, 16, 21, 60, 62),
        "human": (1, 16, 21, 64, 58),
    }
)
REGISTERED_QUERY_SEEDS = MappingProxyType(
    {
        "dog": (2026081502, 2026081503),
        "human": (2026081505, 2026081506),
    }
)

_EXPECTED_UPSTREAM_SCHEMA = _qmosaic.SP4_ROW_SCHEMA_VERSION
_CONSTRUCTION_TOKEN = object()


class PQMosaicRuntimeAdapterError(RuntimeError):
    """The closed P-Q-MOSAIC runtime binding failed before publication."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PQMosaicRuntimeAdapterError(
            "runtime receipt is not finite canonical ASCII JSON"
        ) from error


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise PQMosaicRuntimeAdapterError("runtime receipt is already sealed")
    plain = dict(unsigned)
    return {
        **plain,
        "receipt_digest": hashlib.sha256(_canonical_json_bytes(plain)).hexdigest(),
    }


def _registered_geometry(cell_id: Any) -> tuple[int, int, int, int, int]:
    if type(cell_id) is not str or cell_id not in REGISTERED_CELL_GEOMETRIES:
        raise PQMosaicRuntimeAdapterError(
            "cell must be one pre-registered exact81 dog/human cell"
        )
    return REGISTERED_CELL_GEOMETRIES[cell_id]


def _validate_base_input(
    value: Any,
    *,
    expected_shape: tuple[int, int, int, int, int],
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or tuple(map(int, value.shape)) != expected_shape
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PQMosaicRuntimeAdapterError(
            "base clean latent differs from the registered detached FP32 "
            "exact81 cell geometry"
        )
    norm = torch.linalg.vector_norm(value.to(dtype=torch.float64))
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise PQMosaicRuntimeAdapterError("base clean latent is zero or non-finite")
    return value


def _assert_upstream_clean_vjp(
    clean_vjp_row: Any,
    *,
    cell_id: str,
    expected_shape: tuple[int, int, int, int, int],
) -> tuple[Mapping[str, Any], torch.Tensor]:
    """Authenticate the live summed VJP capability and its registered cell."""

    if type(clean_vjp_row) is not _qmosaic.SP4SummedVJPRow:
        raise PQMosaicRuntimeAdapterError(
            "runtime adapter requires one sealed SP4SummedVJPRow"
        )
    try:
        # ``receipt()`` is the public terminal accessor and itself performs
        # the complete retained-rank/live-byte assertion exactly once.
        receipt = clean_vjp_row.receipt()
    except Exception as error:
        raise PQMosaicRuntimeAdapterError(
            "upstream SP4 clean-latent VJP is not live"
        ) from error
    values = clean_vjp_row.values
    expected_seeds = REGISTERED_QUERY_SEEDS[cell_id]
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != _EXPECTED_UPSTREAM_SCHEMA
        or receipt.get("vjp_target") != "clean_latent"
        or receipt.get("sp_size") != SP_SIZE
        or receipt.get("aggregation") != "SUM"
        or receipt.get("score_divided_by_sp4_before_rank_replay") is not True
        or receipt.get("divide_after_sum") is not False
        or receipt.get("normalization_count") != 1
        or receipt.get("optimizer_or_parameter_update") is not False
        or type(clean_vjp_row.query_seed) is not int
        or clean_vjp_row.query_seed not in expected_seeds
        or receipt.get("query_seed") != clean_vjp_row.query_seed
        or not isinstance(values, torch.Tensor)
        or values.layout != torch.strided
        or values.device.type == "meta"
        or values.dtype != torch.float32
        or tuple(map(int, values.shape)) != expected_shape
        or values.requires_grad
        or values.grad_fn is not None
        or not bool(torch.isfinite(values).all().item())
    ):
        raise PQMosaicRuntimeAdapterError(
            "upstream VJP does not bind the selected exact81 dog/human cell"
        )
    live_sha256 = _qmosaic.tensor_sha256(values, label="P-Q upstream clean VJP")
    live_norm = float(torch.linalg.vector_norm(values.double()).item())
    if (
        receipt.get("value_sha256") != live_sha256
        or not isinstance(receipt.get("value_norm"), (int, float))
        or not math.isfinite(float(receipt["value_norm"]))
        or not math.isclose(
            float(receipt["value_norm"]),
            live_norm,
            rel_tol=2.0e-6,
            abs_tol=1.0e-8,
        )
    ):
        raise PQMosaicRuntimeAdapterError("upstream VJP hash/norm binding differs")
    return receipt, values


@dataclass(frozen=True)
class PQMosaicRuntimeAdaptationV1:
    """Live, mutation-checked result of the fixed P-Q runtime insertion."""

    cell_id: str
    query_seed: int
    latent_shape: tuple[int, int, int, int, int]
    intervention: _projector.PQMosaicSymmetricLatents
    upstream_clean_vjp_receipt_digest: str
    upstream_clean_vjp_value_sha256: str
    input_base_tensor_sha256: str
    nested_intervention_receipt_digest: str
    _input_base: torch.Tensor = field(repr=False, compare=False)
    _clean_vjp_row: _qmosaic.SP4SummedVJPRow = field(repr=False, compare=False)
    _token: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def base(self) -> torch.Tensor:
        return self.intervention.base

    @property
    def plus(self) -> torch.Tensor:
        return self.intervention.plus

    @property
    def minus(self) -> torch.Tensor:
        return self.intervention.minus

    @property
    def projected_clean_latent_vjp(self) -> torch.Tensor:
        return self.intervention.projection.tensor

    @property
    def unit_projected_direction(self) -> torch.Tensor:
        return self.intervention.unit_direction

    @property
    def delta(self) -> torch.Tensor:
        return self.intervention.delta

    def receipt(self) -> dict[str, Any]:
        """Revalidate every live input/output before sealing a receipt."""

        if self._token is not _CONSTRUCTION_TOKEN:
            raise PQMosaicRuntimeAdapterError(
                "runtime adaptation was not built by the closed constructor"
            )
        expected_shape = _registered_geometry(self.cell_id)
        if (
            self.latent_shape != expected_shape
            or type(self.query_seed) is not int
            or self.query_seed not in REGISTERED_QUERY_SEEDS[self.cell_id]
        ):
            raise PQMosaicRuntimeAdapterError(
                "runtime adaptation scalar cell binding changed"
            )
        base_input = _validate_base_input(
            self._input_base, expected_shape=expected_shape
        )
        live_input_base_sha256 = _qmosaic.tensor_sha256(
            base_input, label="live P-Q input base clean latent"
        )
        if live_input_base_sha256 != self.input_base_tensor_sha256:
            raise PQMosaicRuntimeAdapterError(
                "input base clean latent changed after construction"
            )
        upstream, upstream_values = _assert_upstream_clean_vjp(
            self._clean_vjp_row,
            cell_id=self.cell_id,
            expected_shape=expected_shape,
        )
        if (
            upstream.get("digest") != self.upstream_clean_vjp_receipt_digest
            or upstream.get("value_sha256")
            != self.upstream_clean_vjp_value_sha256
            or self._clean_vjp_row.query_seed != self.query_seed
        ):
            raise PQMosaicRuntimeAdapterError(
                "upstream clean-latent VJP provenance changed"
            )
        try:
            intervention_receipt = self.intervention.receipt()
        except _projector.PQMosaicProjectionError as error:
            raise PQMosaicRuntimeAdapterError(
                "nested projected intervention changed after construction"
            ) from error
        if (
            intervention_receipt.get("schema_version")
            != _projector.INTERVENTION_SCHEMA_VERSION
            or intervention_receipt.get("receipt_digest")
            != self.nested_intervention_receipt_digest
            or intervention_receipt.get("relative_l2_dose") != RELATIVE_L2_DOSE
            or intervention_receipt.get("projection_precedes_normalization")
            is not True
            or intervention_receipt.get("latent_symmetry_passed") is not True
            or intervention_receipt.get("scientific_authority") is not False
            or intervention_receipt.get("update") is not False
            or intervention_receipt.get("parameter_update") is not False
        ):
            raise PQMosaicRuntimeAdapterError(
                "nested projected intervention receipt differs"
            )

        projection_receipt = intervention_receipt.get("projection_receipt")
        tensor_bindings = intervention_receipt.get("tensor_bindings")
        projection_bindings = (
            projection_receipt.get("tensor_bindings")
            if isinstance(projection_receipt, Mapping)
            else None
        )
        if not (
            isinstance(tensor_bindings, Mapping)
            and isinstance(projection_bindings, Mapping)
        ):
            raise PQMosaicRuntimeAdapterError(
                "nested projected tensor bindings are absent"
            )
        raw_binding = projection_bindings.get("raw_clean_latent_vjp")
        projected_binding = projection_bindings.get("projected_clean_latent_vjp")
        base_binding = tensor_bindings.get("base_clean_latent")
        direction_binding = tensor_bindings.get("unit_projected_direction")
        delta_binding = tensor_bindings.get("projected_delta")
        plus_binding = tensor_bindings.get("plus_clean_latent")
        minus_binding = tensor_bindings.get("minus_clean_latent")
        binding_rows = (
            raw_binding,
            projected_binding,
            base_binding,
            direction_binding,
            delta_binding,
            plus_binding,
            minus_binding,
        )
        if any(not isinstance(row, Mapping) for row in binding_rows):
            raise PQMosaicRuntimeAdapterError(
                "nested projected tensor binding closure differs"
            )
        assert isinstance(raw_binding, Mapping)
        assert isinstance(projected_binding, Mapping)
        assert isinstance(base_binding, Mapping)
        assert isinstance(direction_binding, Mapping)
        assert isinstance(delta_binding, Mapping)
        assert isinstance(plus_binding, Mapping)
        assert isinstance(minus_binding, Mapping)
        owned_raw_sha256 = raw_binding.get("tensor_sha256")
        if (
            owned_raw_sha256 != self.upstream_clean_vjp_value_sha256
            or _qmosaic.tensor_sha256(
                upstream_values, label="terminal P-Q upstream clean VJP"
            )
            != owned_raw_sha256
            or base_binding.get("tensor_sha256") != live_input_base_sha256
            or any(row.get("shape") != list(expected_shape) for row in binding_rows)
        ):
            raise PQMosaicRuntimeAdapterError(
                "upstream/base and projected tensor hash/geometry closure differs"
            )

        projected_sha256 = projected_binding.get("tensor_sha256")
        direction_sha256 = direction_binding.get("tensor_sha256")
        delta_sha256 = delta_binding.get("tensor_sha256")
        plus_sha256 = plus_binding.get("tensor_sha256")
        minus_sha256 = minus_binding.get("tensor_sha256")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in (
                owned_raw_sha256,
                projected_sha256,
                direction_sha256,
                delta_sha256,
                plus_sha256,
                minus_sha256,
            )
        ):
            raise PQMosaicRuntimeAdapterError(
                "runtime direction hashes are not closed SHA-256 values"
            )

        unsigned: dict[str, Any] = {
            "schema_version": RUNTIME_ADAPTER_SCHEMA_VERSION,
            "method": "P-Q-MOSAIC fixed nuisance-null clean-latent VJP adapter",
            "evidence_tier": EVIDENCE_TIER,
            "runtime_insertion_order": [
                "authenticated_WORLD4_SP4_SUM_clean_latent_VJP",
                "phase0_active_temporal_DC_spatial_affine_1_x_y_null_projection",
                "projected_VJP_FP32_L2_normalization",
                "fixed_relative_L2_dose_0.01",
                "symmetric_base_plus_minus_latents",
            ],
            "projection_occurs_after_real_clean_latent_vjp": True,
            "projection_occurs_before_normalization_and_dose": True,
            "projector_reused_without_math_copy": (
                "p_qmosaic_nuisance_null_projector_v1."
                "construct_projected_symmetric_latents"
            ),
            "registered_cell": {
                "cell_id": self.cell_id,
                "query_seed_from_upstream_vjp": self.query_seed,
                "query_seed_selected_by_adapter": False,
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
                "latent_shape": list(expected_shape),
            },
            "topology": {
                "world_size": WORLD_SIZE,
                "sp_size": SP_SIZE,
                "upstream_aggregation": "SUM",
            },
            "upstream_clean_latent_vjp": {
                "schema_version": upstream["schema_version"],
                "receipt_digest": self.upstream_clean_vjp_receipt_digest,
                "value_sha256": self.upstream_clean_vjp_value_sha256,
                "value_norm": float(upstream["value_norm"]),
                "vjp_target": "clean_latent",
                "detached": True,
                "parameter_update": False,
            },
            "tensor_sha256": {
                "input_base_clean_latent": live_input_base_sha256,
                "projector_owned_raw_clean_latent_vjp": owned_raw_sha256,
                "projected_clean_latent_vjp": projected_sha256,
                "unit_projected_direction": direction_sha256,
                "projected_delta": delta_sha256,
                "base_clean_latent": base_binding["tensor_sha256"],
                "plus_clean_latent": plus_sha256,
                "minus_clean_latent": minus_sha256,
            },
            "relative_l2": {
                "dose": RELATIVE_L2_DOSE,
                "observed": intervention_receipt["observed_relative_l2_dose"],
                "projection_precedes_normalization": True,
                "same_dose_as_raw_qmosaic": True,
                "dose_selected_by_adapter": False,
            },
            "base_plus_minus_symmetry": {
                "passed": True,
                "midpoint_max_abs_error": intervention_receipt[
                    "midpoint_max_abs_error"
                ],
                "delta_antisymmetry_max_abs_error": intervention_receipt[
                    "delta_antisymmetry_max_abs_error"
                ],
                "delta_norm_symmetry_absolute_error": intervention_receipt[
                    "delta_norm_symmetry_absolute_error"
                ],
                "symmetry_tolerance": intervention_receipt["symmetry_tolerance"],
                "arm_selected_by_adapter": False,
            },
            "nuisance_nulls": list(
                projection_receipt["projector"]["fixed_nulls"]
            ),
            "content_inputs": {
                "mask": False,
                "track": False,
                "pose": False,
                "optical_flow": False,
                "detector_box": False,
                "swept_tube": False,
                "content_derived_spatial_support": False,
            },
            "mutation_fail_close": {
                "upstream_sp4_vjp_revalidated": True,
                "input_base_rehashed": True,
                "projector_raw_and_projected_rehashed": True,
                "direction_delta_and_all_arms_rehashed": True,
                "nested_live_storage_and_object_seals_revalidated": True,
                "receipt_denied_after_mutation": True,
            },
            "nested_projection_receipt_digest": projection_receipt[
                "receipt_digest"
            ],
            "nested_intervention_receipt_digest": (
                self.nested_intervention_receipt_digest
            ),
            "semantic_success_assessed": False,
            "identity_or_camera_preservation_assessed": False,
            "decode_publication_authorized": False,
            "lora_vjp_authorized": False,
            "optimizer_created": False,
            "training_update_authorized": False,
            "parameter_update": False,
            "scientific_authority": False,
        }
        return _seal(unsigned)


def build_p_qmosaic_runtime_adaptation_v1(
    *,
    cell_id: str,
    base_clean_latent: torch.Tensor,
    clean_vjp_row: _qmosaic.SP4SummedVJPRow,
) -> PQMosaicRuntimeAdaptationV1:
    """Build the sole fixed P-Q exact81 runtime adaptation.

    ``clean_vjp_row`` must be the live sealed result returned by the current
    Q-MOSAIC WORLD4/SP4 SUM.  Its query seed is consumed as provenance, never
    selected here.  The public signature intentionally has no seed, dose,
    sign, arm, callback, mask, track, pose, flow, or box argument.
    """

    expected_shape = _registered_geometry(cell_id)
    base_input = _validate_base_input(
        base_clean_latent, expected_shape=expected_shape
    )
    upstream_receipt, raw_values = _assert_upstream_clean_vjp(
        clean_vjp_row,
        cell_id=cell_id,
        expected_shape=expected_shape,
    )
    input_base_sha256 = _qmosaic.tensor_sha256(
        base_input, label="P-Q input base clean latent"
    )
    raw_cpu = (
        raw_values.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .clone()
    )
    base_cpu = (
        base_input.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .clone()
    )
    if (
        _qmosaic.tensor_sha256(raw_cpu, label="P-Q owned raw clean VJP")
        != upstream_receipt["value_sha256"]
        or _qmosaic.tensor_sha256(base_cpu, label="P-Q owned base clean latent")
        != input_base_sha256
    ):
        raise PQMosaicRuntimeAdapterError(
            "CPU transfer changed clean-latent input bytes"
        )
    try:
        intervention = _projector.construct_projected_symmetric_latents(
            base_clean_latent=base_cpu,
            raw_clean_latent_vjp=raw_cpu,
        )
        intervention_receipt = intervention.receipt()
    except _projector.PQMosaicProjectionError as error:
        raise PQMosaicRuntimeAdapterError(
            "frozen nuisance-null projector failed closed"
        ) from error
    projection_receipt = intervention_receipt.get("projection_receipt")
    if not isinstance(projection_receipt, Mapping):
        raise PQMosaicRuntimeAdapterError("projection receipt is absent")
    projection_bindings = projection_receipt.get("tensor_bindings")
    intervention_bindings = intervention_receipt.get("tensor_bindings")
    if not (
        isinstance(projection_bindings, Mapping)
        and isinstance(intervention_bindings, Mapping)
        and projection_bindings["raw_clean_latent_vjp"]["tensor_sha256"]
        == upstream_receipt["value_sha256"]
        and intervention_bindings["base_clean_latent"]["tensor_sha256"]
        == input_base_sha256
    ):
        raise PQMosaicRuntimeAdapterError(
            "frozen projector did not preserve upstream/base byte identity"
        )
    result = PQMosaicRuntimeAdaptationV1(
        cell_id=cell_id,
        query_seed=clean_vjp_row.query_seed,
        latent_shape=expected_shape,
        intervention=intervention,
        upstream_clean_vjp_receipt_digest=upstream_receipt["digest"],
        upstream_clean_vjp_value_sha256=upstream_receipt["value_sha256"],
        input_base_tensor_sha256=input_base_sha256,
        nested_intervention_receipt_digest=intervention_receipt["receipt_digest"],
        _input_base=base_input,
        _clean_vjp_row=clean_vjp_row,
    )
    object.__setattr__(result, "_token", _CONSTRUCTION_TOKEN)
    result.receipt()
    return result


__all__ = [
    "EVIDENCE_TIER",
    "FRAME_COUNT",
    "LATENT_PHASES",
    "PQMosaicRuntimeAdaptationV1",
    "PQMosaicRuntimeAdapterError",
    "REGISTERED_CELL_GEOMETRIES",
    "REGISTERED_QUERY_SEEDS",
    "RELATIVE_L2_DOSE",
    "RUNTIME_ADAPTER_SCHEMA_VERSION",
    "SP_SIZE",
    "WORLD_SIZE",
    "build_p_qmosaic_runtime_adaptation_v1",
]
