#!/usr/bin/env python3
"""Fixed P-Q direction construction and portable evidence envelope.

This module is the only adapter between the authenticated Q-MOSAIC runner and
the frozen nuisance-null runtime adapter.  It owns no WORLD4 replay, decode,
semantic selection, optimizer, or update path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import p_qmosaic_runtime_adapter_v1 as adapter
import self_imagined_native_rv2v_hidden_vjp_v1 as qmosaic


METHOD_NAME = "bernini-p-qmosaic-nuisance-projected-editor-direction-sp4"
RUN_RECEIPT_SCHEMA = "bernini-p-qmosaic-editor-direction-sp4-run-v1"
POSTFLIGHT_SCHEMA = "bernini-p-qmosaic-editor-direction-postflight-v1"
EVIDENCE_SCHEMA = "bernini-p-qmosaic-direction-evidence-envelope-v1"
VARIANT_LOCK_SCHEMA = "bernini-qmosaic-direction-variant-lock-v1"
VARIANT_ID = "p_qmosaic_nuisance_null_projection_v1"
SEMANTIC_UNASSESSED = "UNASSESSED_NO_METHOD_OWNED_EVALUATOR"
FIXED_NULLS = (
    "phase0",
    "active_phases_1_to_20_temporal_dc",
    "per_channel_phase_spatial_affine_1_x_y",
)
FORMULA = (
    "q=P_null(raw_clean_vjp)/l2(P_null(raw_clean_vjp));"
    "scale=0.01*l2(base);plus=base+scale*q;minus=base-scale*q"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PQMosaicDirectionEnvelopeError(RuntimeError):
    """The fixed P-Q direction or its portable closure differed."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PQMosaicDirectionEnvelopeError(
            "P-Q evidence is not finite canonical ASCII JSON"
        ) from error


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    plain = dict(unsigned)
    if "receipt_digest" in plain:
        raise PQMosaicDirectionEnvelopeError("P-Q evidence is already sealed")
    return {
        **plain,
        "receipt_digest": hashlib.sha256(_canonical(plain)).hexdigest(),
    }


def _verify_seal(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise PQMosaicDirectionEnvelopeError(f"{label} is absent")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(_canonical(unsigned)).hexdigest() != digest
    ):
        raise PQMosaicDirectionEnvelopeError(f"{label} seal differs")
    return digest


def variant_lock() -> Mapping[str, Any]:
    return {
        "schema_version": VARIANT_LOCK_SCHEMA,
        "variant_id": VARIANT_ID,
        "fixed_by_versioned_entrypoint_before_submission": True,
        "cli_variant_argument_available": False,
        "seed_selection": False,
        "dose_selection": False,
        "sign_selection": False,
        "arm_selection": False,
        "callback_selection": False,
        "semantic_selection": False,
    }


_ADAPTER_FIELDS = frozenset(
    {
        "schema_version", "method", "evidence_tier", "runtime_insertion_order",
        "projection_occurs_after_real_clean_latent_vjp",
        "projection_occurs_before_normalization_and_dose",
        "projector_reused_without_math_copy", "registered_cell", "topology",
        "upstream_clean_latent_vjp", "tensor_sha256", "relative_l2",
        "base_plus_minus_symmetry", "nuisance_nulls", "content_inputs",
        "mutation_fail_close", "nested_projection_receipt_digest",
        "nested_intervention_receipt_digest", "semantic_success_assessed",
        "identity_or_camera_preservation_assessed", "decode_publication_authorized",
        "lora_vjp_authorized", "optimizer_created", "training_update_authorized",
        "parameter_update", "scientific_authority", "receipt_digest",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version", "variant_lock", "evidence_tier", "runtime_insertion",
        "formula", "relative_l2_dose", "fixed_nuisance_nulls", "cell_id",
        "query_seed", "tensor_sha256", "norms_and_symmetry",
        "adapter_receipt_recomputed_after_live_tensor_rehash",
        "runtime_adapter_receipt", "semantic_assessment", "decode_selection",
        "lora_vjp", "optimizer_created", "parameter_update",
        "scientific_authority", "receipt_digest",
    }
)
_HASH_KEYS = frozenset(
    {
        "input_base_clean_latent", "upstream_raw_clean_latent_vjp",
        "adapter_base_clean_latent", "projected_clean_latent_vjp",
        "unit_projected_direction", "projected_delta", "plus_clean_latent",
        "minus_clean_latent",
    }
)


def validate_envelope(
    value: Any,
    *,
    cell_id: str,
    query_seed: int,
    clean_vjp_receipt_digest: str,
    clean_vjp_value_sha256: str,
    base_tensor_sha256: str,
    plus_tensor_sha256: str,
    minus_tensor_sha256: str,
) -> Mapping[str, Any]:
    """Recompute seals and every portable adapter/live-hash cross-binding."""

    if not isinstance(value, Mapping) or set(value) != _ENVELOPE_FIELDS:
        raise PQMosaicDirectionEnvelopeError("P-Q envelope field closure differs")
    _verify_seal(value, label="P-Q direction envelope")
    runtime_receipt = value.get("runtime_adapter_receipt")
    if not isinstance(runtime_receipt, Mapping) or set(runtime_receipt) != _ADAPTER_FIELDS:
        raise PQMosaicDirectionEnvelopeError("P-Q adapter field closure differs")
    adapter_digest = _verify_seal(runtime_receipt, label="P-Q runtime adapter receipt")
    hashes = value.get("tensor_sha256")
    adapter_hashes = runtime_receipt.get("tensor_sha256")
    registered = runtime_receipt.get("registered_cell")
    upstream = runtime_receipt.get("upstream_clean_latent_vjp")
    relative = runtime_receipt.get("relative_l2")
    symmetry = runtime_receipt.get("base_plus_minus_symmetry")
    metrics = value.get("norms_and_symmetry")
    if not all(
        isinstance(row, Mapping)
        for row in (hashes, adapter_hashes, registered, upstream, relative, symmetry, metrics)
    ):
        raise PQMosaicDirectionEnvelopeError("P-Q nested mapping closure differs")
    assert isinstance(hashes, Mapping) and isinstance(adapter_hashes, Mapping)
    assert isinstance(registered, Mapping) and isinstance(upstream, Mapping)
    assert isinstance(relative, Mapping) and isinstance(symmetry, Mapping)
    assert isinstance(metrics, Mapping)
    expected_metrics = {
        "base_l2_norm", "clean_vjp_l2_norm", "projected_vjp_l2_norm",
        "direction_l2_norm", "absolute_dose_l2", "plus_delta_l2",
        "minus_delta_l2", "delta_norm_symmetry_absolute_error",
        "midpoint_max_abs_error", "delta_antisymmetry_max_abs_error",
        "symmetry_tolerance",
    }
    if set(hashes) != _HASH_KEYS or set(metrics) != expected_metrics:
        raise PQMosaicDirectionEnvelopeError("P-Q hash/metric closure differs")
    if any(
        not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None
        for item in (*hashes.values(), clean_vjp_receipt_digest, adapter_digest)
    ) or any(
        not isinstance(item, (int, float)) or not math.isfinite(float(item))
        for item in metrics.values()
    ):
        raise PQMosaicDirectionEnvelopeError("P-Q hash/metric value differs")
    expected_adapter_hashes = {
        "input_base_clean_latent": hashes["input_base_clean_latent"],
        "projector_owned_raw_clean_latent_vjp": hashes["upstream_raw_clean_latent_vjp"],
        "projected_clean_latent_vjp": hashes["projected_clean_latent_vjp"],
        "unit_projected_direction": hashes["unit_projected_direction"],
        "projected_delta": hashes["projected_delta"],
        "base_clean_latent": hashes["adapter_base_clean_latent"],
        "plus_clean_latent": hashes["plus_clean_latent"],
        "minus_clean_latent": hashes["minus_clean_latent"],
    }
    false_adapter = (
        "semantic_success_assessed", "identity_or_camera_preservation_assessed",
        "decode_publication_authorized", "lora_vjp_authorized", "optimizer_created",
        "training_update_authorized", "parameter_update", "scientific_authority",
    )
    tolerance = float(metrics.get("symmetry_tolerance", -1.0))
    if (
        value.get("schema_version") != EVIDENCE_SCHEMA
        or value.get("variant_lock") != variant_lock()
        or value.get("evidence_tier") != "ENGINEERING_ONLY"
        or value.get("runtime_insertion")
        != "after_WORLD4_SP4_SUM_clean_VJP_before_normalization_and_dose"
        or value.get("formula") != FORMULA
        or value.get("relative_l2_dose") != adapter.RELATIVE_L2_DOSE
        or value.get("fixed_nuisance_nulls") != list(FIXED_NULLS)
        or value.get("cell_id") != cell_id or value.get("query_seed") != query_seed
        or value.get("adapter_receipt_recomputed_after_live_tensor_rehash") is not True
        or value.get("semantic_assessment") != SEMANTIC_UNASSESSED
        or any(value.get(name) is not False for name in (
            "decode_selection", "lora_vjp", "optimizer_created",
            "parameter_update", "scientific_authority",
        ))
        or runtime_receipt.get("schema_version") != adapter.RUNTIME_ADAPTER_SCHEMA_VERSION
        or runtime_receipt.get("evidence_tier") != "ENGINEERING_ONLY"
        or runtime_receipt.get("nuisance_nulls") != list(FIXED_NULLS)
        or any(runtime_receipt.get(name) is not False for name in false_adapter)
        or registered.get("cell_id") != cell_id
        or registered.get("query_seed_from_upstream_vjp") != query_seed
        or registered.get("query_seed_selected_by_adapter") is not False
        or upstream.get("receipt_digest") != clean_vjp_receipt_digest
        or upstream.get("value_sha256") != clean_vjp_value_sha256
        or upstream.get("parameter_update") is not False
        or relative.get("dose") != adapter.RELATIVE_L2_DOSE
        or relative.get("dose_selected_by_adapter") is not False
        or symmetry.get("passed") is not True
        or symmetry.get("arm_selected_by_adapter") is not False
        or dict(adapter_hashes) != expected_adapter_hashes
        or hashes["input_base_clean_latent"] != base_tensor_sha256
        or hashes["adapter_base_clean_latent"] != base_tensor_sha256
        or hashes["upstream_raw_clean_latent_vjp"] != clean_vjp_value_sha256
        or hashes["plus_clean_latent"] != plus_tensor_sha256
        or hashes["minus_clean_latent"] != minus_tensor_sha256
        or float(metrics["base_l2_norm"]) <= 0.0
        or float(metrics["clean_vjp_l2_norm"]) <= 0.0
        or float(metrics["projected_vjp_l2_norm"]) <= 0.0
        or not math.isclose(float(metrics["direction_l2_norm"]), 1.0,
                            rel_tol=5.0e-5, abs_tol=5.0e-6)
        or tolerance < 0.0
        or float(metrics["midpoint_max_abs_error"]) > tolerance
        or float(metrics["delta_antisymmetry_max_abs_error"]) > 2.0 * tolerance
    ):
        raise PQMosaicDirectionEnvelopeError("P-Q provenance/authority closure differs")
    return dict(value)


def construct(
    *, cell_id: str, base_clean_latent: Any, clean_vjp_row: Any
) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    """Build the fixed projected arms and independently rehash every tensor."""

    try:
        result = adapter.build_p_qmosaic_runtime_adaptation_v1(
            cell_id=cell_id,
            base_clean_latent=base_clean_latent,
            clean_vjp_row=clean_vjp_row,
        )
        receipt_before = result.receipt()
        tensors = {
            "input_base_clean_latent": base_clean_latent,
            "upstream_raw_clean_latent_vjp": clean_vjp_row.values,
            "adapter_base_clean_latent": result.base,
            "projected_clean_latent_vjp": result.projected_clean_latent_vjp,
            "unit_projected_direction": result.unit_projected_direction,
            "projected_delta": result.delta,
            "plus_clean_latent": result.plus,
            "minus_clean_latent": result.minus,
        }
        hashes = {
            name: qmosaic.tensor_sha256(tensor, label=f"P-Q live {name}")
            for name, tensor in tensors.items()
        }
        receipt_after = result.receipt()
        intervention = result.intervention.receipt()
        upstream_receipt = clean_vjp_row.receipt()
    except Exception as error:
        raise PQMosaicDirectionEnvelopeError("P-Q live construction failed closed") from error
    if receipt_before != receipt_after:
        raise PQMosaicDirectionEnvelopeError("P-Q adapter changed across live rehash")
    envelope = _seal(
        {
            "schema_version": EVIDENCE_SCHEMA,
            "variant_lock": dict(variant_lock()),
            "evidence_tier": "ENGINEERING_ONLY",
            "runtime_insertion": "after_WORLD4_SP4_SUM_clean_VJP_before_normalization_and_dose",
            "formula": FORMULA,
            "relative_l2_dose": adapter.RELATIVE_L2_DOSE,
            "fixed_nuisance_nulls": list(FIXED_NULLS),
            "cell_id": cell_id,
            "query_seed": clean_vjp_row.query_seed,
            "tensor_sha256": hashes,
            "norms_and_symmetry": {
                "base_l2_norm": intervention["base_l2_norm"],
                "clean_vjp_l2_norm": receipt_after["upstream_clean_latent_vjp"]["value_norm"],
                "projected_vjp_l2_norm": intervention["projected_vjp_l2_norm_fp32"],
                "direction_l2_norm": intervention["direction_l2_norm"],
                "absolute_dose_l2": intervention["absolute_dose_l2"],
                "plus_delta_l2": intervention["plus_delta_l2"],
                "minus_delta_l2": intervention["minus_delta_l2"],
                "delta_norm_symmetry_absolute_error": intervention[
                    "delta_norm_symmetry_absolute_error"
                ],
                "midpoint_max_abs_error": intervention["midpoint_max_abs_error"],
                "delta_antisymmetry_max_abs_error": intervention[
                    "delta_antisymmetry_max_abs_error"
                ],
                "symmetry_tolerance": intervention["symmetry_tolerance"],
            },
            "adapter_receipt_recomputed_after_live_tensor_rehash": True,
            "runtime_adapter_receipt": receipt_after,
            "semantic_assessment": SEMANTIC_UNASSESSED,
            "decode_selection": False,
            "lora_vjp": False,
            "optimizer_created": False,
            "parameter_update": False,
            "scientific_authority": False,
        }
    )
    validate_envelope(
        envelope, cell_id=cell_id, query_seed=clean_vjp_row.query_seed,
        clean_vjp_receipt_digest=upstream_receipt["digest"],
        clean_vjp_value_sha256=upstream_receipt["value_sha256"],
        base_tensor_sha256=hashes["adapter_base_clean_latent"],
        plus_tensor_sha256=hashes["plus_clean_latent"],
        minus_tensor_sha256=hashes["minus_clean_latent"],
    )
    return result.base, result.plus, result.minus, envelope


__all__ = [
    "EVIDENCE_SCHEMA", "METHOD_NAME", "POSTFLIGHT_SCHEMA", "RUN_RECEIPT_SCHEMA",
    "PQMosaicDirectionEnvelopeError", "VARIANT_ID", "construct",
    "validate_envelope", "variant_lock",
]
