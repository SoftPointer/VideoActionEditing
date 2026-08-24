"""Pure-tensor dual-transport Gaussian-noise ablation for DMIQ-Edit.

This module tests a narrow hypothesis: a frozen source transport can correlate
otherwise standard Gaussian noise across latent phases, while an independently
sampled action transport supplies a second correlated marginal.  A detached
internal pre-noise gate mixes those two marginals without changing the
*analytic per-coordinate variance*.

For source/action fields ``xi`` and frozen orthogonal transports ``O``::

    eS[t] = sqrt(kS) O_S[t] xiS0 + sqrt(1-kS) xiS[t]
    eA[t] = sqrt(kA) O_A[t] xiA0 + sqrt(1-kA) xiA[t]
    e[t]  = sqrt(1-g[t]^2) eS[t] + g[t] eA[t]

The four Gaussian tensors are a caller-declared mutually independent standard
normal draw.  A single realization cannot prove normality or statistical
independence, so this implementation makes the mechanically verifiable part
explicit: every input is detached FP32, finite, contiguous, and backed by a
different storage allocation.  Orthogonality and the resulting conditional
variance coefficient are audited analytically in FP64.  The returned receipt
never calls the output native IID noise: shared base fields deliberately make
its joint temporal covariance non-IID.

The practical main ablation represents each ``O[t]`` as one exact permutation
of ``0..P-1`` plus elementwise signs in ``{-1,+1}``.  It applies the transport
with gather and multiplication, requiring ``O(21P)`` storage at the real
``P=62*60=3720`` latent grid.  Explicit ``[21,P,P]`` matrices and the offline
polar helper remain available only for small-``P`` numerical controls.

No RGB, VAE latent, target video, segmentation mask, flow, pose, track, or
trajectory is accepted.  This is an experimental ablation only; it is not a
training target, reward, or claim of successful action editing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence


LATENT_PHASES = 21
PRACTICAL_LATENT_HEIGHT = 62
PRACTICAL_LATENT_WIDTH = 60
PRACTICAL_LATENT_POSITIONS = PRACTICAL_LATENT_HEIGHT * PRACTICAL_LATENT_WIDTH
RECEIPT_SCHEMA = "dmiq-dual-transport-noise-ablation-v1"
DEFAULT_ORTHOGONALITY_TOLERANCE = 3.0e-4
DEFAULT_VARIANCE_TOLERANCE = 4.0e-4
DEFAULT_NON_IID_TOLERANCE = 1.0e-12


class DMIQDualTransportNoiseError(RuntimeError):
    """Raised before returning noise whose contract cannot be certified."""


@dataclass(frozen=True)
class DualTransportNoiseDiagnostics:
    shape: tuple[int, int, int, int]
    dtype: str
    device: str
    k_source: float
    k_action: float
    gate_min: float
    gate_max: float
    source_transport_left_orthogonality_max_abs_error: float
    source_transport_right_orthogonality_max_abs_error: float
    action_transport_left_orthogonality_max_abs_error: float
    action_transport_right_orthogonality_max_abs_error: float
    orthogonality_tolerance: float
    source_variance_coefficient_min: float
    source_variance_coefficient_max: float
    action_variance_coefficient_min: float
    action_variance_coefficient_max: float
    output_variance_coefficient_min: float
    output_variance_coefficient_max: float
    output_variance_coefficient_max_abs_error: float
    variance_tolerance: float
    analytic_per_coordinate_variance_passed: bool
    source_shared_carrier_effective_per_batch: tuple[bool, ...]
    action_shared_carrier_effective_per_batch: tuple[bool, ...]
    effective_carrier_rows_per_batch: tuple[int, ...]
    carrier_dimension_upper_bound_per_batch: tuple[int, ...]
    joint_covariance_non_iid_rank_certificate: bool
    joint_covariance_non_iid: bool
    non_iid_tolerance: float
    all_inputs_detached: bool
    all_floating_inputs_finite_fp32_contiguous: bool
    permutation_indices_contiguous_int64: bool | None
    all_input_storages_distinct: bool
    output_storage_distinct_from_inputs: bool
    caller_declared_standard_gaussian_fields: bool
    caller_declared_mutually_independent_gaussian_fields: bool
    statistical_independence_inferable_from_one_realization: bool
    conditional_mean_zero_under_input_contract: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DualTransportNoiseResult:
    """Constructed target noise and its deliberately conservative receipt."""

    initial_noise: Any
    diagnostics: DualTransportNoiseDiagnostics
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class PolarOrthogonalizationResult:
    """Offline FP64-CPU polar projection with an FP32 frozen output."""

    transport: Any
    receipt: Mapping[str, Any]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise DMIQDualTransportNoiseError(
            "DMIQ dual-transport noise requires PyTorch"
        ) from error
    return torch


def _validate_real_unit_interval(value: Real, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DMIQDualTransportNoiseError(f"{label} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise DMIQDualTransportNoiseError(
            f"{label} must be finite and lie in [0,1]"
        )
    return normalized


def _validate_positive_tolerance(value: Real, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DMIQDualTransportNoiseError(f"{label} must be a real scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise DMIQDualTransportNoiseError(
            f"{label} must be finite and lie strictly in (0,1)"
        )
    return normalized


def _validate_tensor_base(value: Any, *, label: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise DMIQDualTransportNoiseError(f"{label} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise DMIQDualTransportNoiseError(f"{label} must be torch.float32")
    if value.requires_grad or value.grad_fn is not None:
        raise DMIQDualTransportNoiseError(f"{label} must be detached/no-grad")
    if value.numel() <= 0:
        raise DMIQDualTransportNoiseError(f"{label} must be non-empty")
    if not value.is_contiguous():
        raise DMIQDualTransportNoiseError(f"{label} must be contiguous")
    if not bool(torch.isfinite(value).all().item()):
        raise DMIQDualTransportNoiseError(f"{label} must be finite")
    return value


def _storage_identity(value: Any) -> tuple[str, int]:
    """Return an allocation identity after non-empty/contiguous validation."""

    storage = value.untyped_storage()
    return (str(value.device), int(storage.data_ptr()))


def _validate_distinct_storages(named: Sequence[tuple[str, Any]]) -> None:
    identities: dict[tuple[str, int], str] = {}
    for label, value in named:
        identity = _storage_identity(value)
        previous = identities.get(identity)
        if previous is not None:
            raise DMIQDualTransportNoiseError(
                f"{label} aliases storage owned by {previous}"
            )
        identities[identity] = label


def _orthogonality_errors(transport: Any) -> tuple[float, float, Any]:
    """Return FP64 CPU left/right Gram errors and per-row squared norms."""

    torch = _torch()
    work = transport.detach().to(device="cpu", dtype=torch.float64).contiguous()
    width = int(work.shape[-1])
    identity = torch.eye(width, dtype=torch.float64, device="cpu").expand(
        LATENT_PHASES, width, width
    )
    left = torch.matmul(work, work.transpose(-1, -2))
    right = torch.matmul(work.transpose(-1, -2), work)
    left_error = float((left - identity).abs().max().item())
    right_error = float((right - identity).abs().max().item())
    row_norm_sq = work.square().sum(dim=-1)
    return left_error, right_error, row_norm_sq


def _validate_inputs(
    *,
    source_base: Any,
    source_innovations: Any,
    action_base: Any,
    action_innovations: Any,
    source_transport: Any,
    action_transport: Any,
    pre_noise_gate: Any,
    orthogonality_tolerance: float,
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    named = [
        ("source_base", source_base),
        ("source_innovations", source_innovations),
        ("action_base", action_base),
        ("action_innovations", action_innovations),
        ("source_transport", source_transport),
        ("action_transport", action_transport),
        ("pre_noise_gate", pre_noise_gate),
    ]
    for label, value in named:
        _validate_tensor_base(value, label=label)

    if source_base.ndim != 3:
        raise DMIQDualTransportNoiseError("source_base must be [B,C,P]")
    batch, channels, positions = (int(item) for item in source_base.shape)
    if batch <= 0 or channels <= 0 or positions <= 0:
        raise DMIQDualTransportNoiseError("B, C, and P must be positive")
    base_shape = (batch, channels, positions)
    innovation_shape = (batch, channels, LATENT_PHASES, positions)
    transport_shape = (LATENT_PHASES, positions, positions)
    gate_shape = (batch, 1, LATENT_PHASES, positions)
    expected_shapes = {
        "source_base": base_shape,
        "source_innovations": innovation_shape,
        "action_base": base_shape,
        "action_innovations": innovation_shape,
        "source_transport": transport_shape,
        "action_transport": transport_shape,
        "pre_noise_gate": gate_shape,
    }
    for label, value in named:
        if tuple(int(item) for item in value.shape) != expected_shapes[label]:
            raise DMIQDualTransportNoiseError(
                f"{label} must have shape {expected_shapes[label]}"
            )
        if value.device != source_base.device:
            raise DMIQDualTransportNoiseError(
                "all dual-transport inputs must share one device"
            )
    _validate_distinct_storages(named)

    torch = _torch()
    gate_min = float(pre_noise_gate.min().detach().cpu().item())
    gate_max = float(pre_noise_gate.max().detach().cpu().item())
    if gate_min < 0.0 or gate_max > 1.0:
        raise DMIQDualTransportNoiseError(
            "pre_noise_gate must lie elementwise in [0,1]"
        )

    source_left, source_right, source_row_norm_sq = _orthogonality_errors(
        source_transport
    )
    action_left, action_right, action_row_norm_sq = _orthogonality_errors(
        action_transport
    )
    if max(source_left, source_right) > orthogonality_tolerance:
        raise DMIQDualTransportNoiseError(
            "source_transport is not orthogonal within tolerance"
        )
    if max(action_left, action_right) > orthogonality_tolerance:
        raise DMIQDualTransportNoiseError(
            "action_transport is not orthogonal within tolerance"
        )
    return base_shape, {
        "gate_min": gate_min,
        "gate_max": gate_max,
        "source_left": source_left,
        "source_right": source_right,
        "action_left": action_left,
        "action_right": action_right,
        "source_row_norm_sq": source_row_norm_sq,
        "action_row_norm_sq": action_row_norm_sq,
        "torch": torch,
        "named": named,
    }


def _analytic_variance_and_non_iid_audit(
    *,
    pre_noise_gate: Any,
    source_row_norm_sq: Any,
    action_row_norm_sq: Any,
    k_source: float,
    k_action: float,
    variance_tolerance: float,
    non_iid_tolerance: float,
) -> dict[str, Any]:
    """Audit conditional variances and prove non-IID covariance by rank.

    For each batch, rows of the shared-carrier loading matrix live in at most
    ``P`` dimensions per active base field.  More nonzero rows than that upper
    bound cannot be mutually orthogonal, proving at least one nonzero
    cross-coordinate covariance without materializing a ``(21P)^2`` matrix.
    """

    torch = _torch()
    gate = pre_noise_gate.detach().to(device="cpu", dtype=torch.float64)
    gate_sq = gate.square()
    source_rows = source_row_norm_sq.view(1, 1, LATENT_PHASES, -1)
    action_rows = action_row_norm_sq.view(1, 1, LATENT_PHASES, -1)
    source_variance = k_source * source_rows + (1.0 - k_source)
    action_variance = k_action * action_rows + (1.0 - k_action)
    output_variance = (
        (1.0 - gate_sq) * source_variance + gate_sq * action_variance
    )
    variance_error = (output_variance - 1.0).abs()
    maximum_error = float(variance_error.max().item())
    if maximum_error > variance_tolerance:
        raise DMIQDualTransportNoiseError(
            "analytic per-coordinate variance coefficient differs from one"
        )

    # Squared norms of the source/action shared-base loading components.  The
    # innovation terms are coordinate-independent and cannot create temporal
    # off-diagonal covariance.
    source_loading = (1.0 - gate_sq) * k_source * source_rows
    action_loading = gate_sq * k_action * action_rows
    positions = int(gate.shape[-1])
    source_active = (source_loading > non_iid_tolerance).flatten(1).any(dim=1)
    action_active = (action_loading > non_iid_tolerance).flatten(1).any(dim=1)
    dimension_tensor = positions * (
        source_active.to(dtype=torch.int64)
        + action_active.to(dtype=torch.int64)
    )
    active_row_tensor = (
        (source_loading + action_loading > non_iid_tolerance)
        .flatten(1)
        .sum(dim=1)
    )
    rank_certificate = bool(
        (
            (dimension_tensor > 0)
            & (active_row_tensor > dimension_tensor)
        )
        .all()
        .item()
    )
    if not rank_certificate:
        raise DMIQDualTransportNoiseError(
            "joint non-IID covariance lacks the conservative carrier-rank "
            "certificate; increase an effective k or use a less degenerate gate"
        )

    return {
        "source_min": float(source_variance.min().item()),
        "source_max": float(source_variance.max().item()),
        "action_min": float(action_variance.min().item()),
        "action_max": float(action_variance.max().item()),
        "output_min": float(output_variance.min().item()),
        "output_max": float(output_variance.max().item()),
        "output_max_abs_error": maximum_error,
        "source_active": tuple(bool(item) for item in source_active.tolist()),
        "action_active": tuple(bool(item) for item in action_active.tolist()),
        "active_rows": tuple(int(item) for item in active_row_tensor.tolist()),
        "dimension_bounds": tuple(int(item) for item in dimension_tensor.tolist()),
        "rank_certificate": True,
    }


def _build_receipt(
    diagnostics: DualTransportNoiseDiagnostics,
    *,
    transport_representation: str = "explicit_dense_matrix",
) -> dict[str, Any]:
    if transport_representation not in (
        "explicit_dense_matrix",
        "implicit_signed_permutation",
    ):
        raise DMIQDualTransportNoiseError(
            "unsupported transport representation in receipt"
        )
    implicit = transport_representation == "implicit_signed_permutation"
    transport_inputs = (
        [
            "source_permutation_indices_int64",
            "source_elementwise_signs_fp32",
            "action_permutation_indices_int64",
            "action_elementwise_signs_fp32",
        ]
        if implicit
        else [
            "source_dense_orthogonal_transport",
            "action_dense_orthogonal_transport",
        ]
    )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "method": (
            "dmiq_dual_signed_permutation_noise"
            if implicit
            else "dmiq_dual_transport_noise"
        ),
        "transport_representation": transport_representation,
        "experimental_ablation": True,
        "ablation_only": True,
        "scientific_claim_authorized": False,
        "production_claim_authorized": False,
        "training_or_optimizer_step_performed": False,
        "pure_tensor_operator": True,
        "frozen_transport_operator": True,
        "inputs": {
            "accepted": [
                "source_base_gaussian",
                "source_innovation_gaussian",
                "action_base_gaussian",
                "action_innovation_gaussian",
                *transport_inputs,
                "detached_internal_pre_noise_gate",
                "k_source",
                "k_action",
            ],
            "four_standard_gaussian_fields": True,
            "mutual_statistical_independence_is_caller_contract": True,
            "distinct_storage_allocations_mechanically_verified": True,
            "single_realization_used_to_claim_independence": False,
            "gate_computed_before_noise_contract": True,
            "gate_noise_independence_is_caller_contract": True,
            "internal_gate_provenance_is_caller_contract": True,
            "transport_provenance_is_caller_contract": True,
            "provenance_inferable_from_tensor_values": False,
        },
        "condition_closure_limit": {
            "direct_media_or_control_tensor_arguments_accepted": False,
            "upstream_gate_provenance_mechanically_verified": False,
            "upstream_transport_provenance_mechanically_verified": False,
            "absence_of_target_mask_flow_pose_or_track_derivation_proved": False,
            "train_inference_builder_identity_proved": False,
            "numeric_operator_receipt_is_condition_closure_evidence": False,
            "runtime_provenance_binding_still_required": [
                "four_independent_rng_substream_receipts",
                "source_noop_receipt",
                "pass_a_action_plan_receipt",
                "checkpoint_and_prompt_digests",
                "gate_builder_revision_and_artifact_digest",
                "transport_builder_revision_and_artifact_digest",
            ],
        },
        "gate": {
            "kind": "detached_internal_pre_noise_counterfactual_sensitivity",
            "external_mask": False,
            "external_mask_parameter_accepted": False,
            "origin_mechanically_inferable_from_tensor_values": False,
            "train_inference_same_tensor_contract_is_caller_requirement": True,
            "train_inference_same_builder_identity_mechanically_verified": False,
            "range": [0.0, 1.0],
        },
        "transport": {
            "representation": transport_representation,
            "required_origin": "frozen_internal_source_and_action_correspondence",
            "origin_is_caller_contract": True,
            "origin_mechanically_inferable_from_tensor_values": False,
            "external_flow_pose_track_or_mask_transport_forbidden": True,
            "dense_transport_materialized": not implicit,
            "dense_validation_or_small_p_control_only": not implicit,
            "primary_practical_ablation": implicit,
            "storage_complexity": "O(21P)" if implicit else "O(21P^2)",
            "application": (
                "gather_then_elementwise_sign"
                if implicit
                else "dense_matrix_vector_product"
            ),
            "orthogonality_certification": (
                "exact_combinatorial_permutation_and_unit_sign_validation"
                if implicit
                else "fp64_left_and_right_gram_audit"
            ),
            "practical_latent_grid": {
                "height": PRACTICAL_LATENT_HEIGHT,
                "width": PRACTICAL_LATENT_WIDTH,
                "P": PRACTICAL_LATENT_POSITIONS,
            },
            "current_P": diagnostics.shape[-1],
            "current_P_must_equal_practical_P": False,
        },
        "distribution": {
            "conditional_mean_zero_under_declared_input_contract": True,
            "analytic_per_coordinate_variance_one": True,
            "variance_verified_from_realized_sample_statistics": False,
            "joint_covariance_non_iid": True,
            "joint_covariance_non_iid_rank_certified": True,
            "native_iid_gaussian_claim": False,
            "joint_density_equal_to_native_iid_gaussian_claim": False,
            "shared_source_base_configured": diagnostics.k_source > 0.0,
            "shared_action_base_configured": diagnostics.k_action > 0.0,
            "shared_source_base_effective_in_any_batch": any(
                diagnostics.source_shared_carrier_effective_per_batch
            ),
            "shared_action_base_effective_in_any_batch": any(
                diagnostics.action_shared_carrier_effective_per_batch
            ),
        },
        "formula": {
            "source": "sqrt(kS)*O_S[t]*xiS0+sqrt(1-kS)*xiS[t]",
            "action": "sqrt(kA)*O_A[t]*xiA0+sqrt(1-kA)*xiA[t]",
            "output": "sqrt(1-g[t]^2)*eS[t]+g[t]*eA[t]",
        },
        "diagnostics": diagnostics.as_dict(),
    }


def dmiq_dual_transport_noise(
    source_base: Any,
    source_innovations: Any,
    action_base: Any,
    action_innovations: Any,
    source_transport: Any,
    action_transport: Any,
    pre_noise_gate: Any,
    *,
    k_source: Real,
    k_action: Real,
    orthogonality_tolerance: Real = DEFAULT_ORTHOGONALITY_TOLERANCE,
    variance_tolerance: Real = DEFAULT_VARIANCE_TOLERANCE,
    non_iid_tolerance: Real = DEFAULT_NON_IID_TOLERANCE,
) -> DualTransportNoiseResult:
    """Construct one exact-21-phase source/action-rich noise ablation."""

    torch = _torch()
    k_source_value = _validate_real_unit_interval(k_source, label="k_source")
    k_action_value = _validate_real_unit_interval(k_action, label="k_action")
    orthogonality_tolerance_value = _validate_positive_tolerance(
        orthogonality_tolerance, label="orthogonality_tolerance"
    )
    variance_tolerance_value = _validate_positive_tolerance(
        variance_tolerance, label="variance_tolerance"
    )
    non_iid_tolerance_value = _validate_positive_tolerance(
        non_iid_tolerance, label="non_iid_tolerance"
    )
    base_shape, audit = _validate_inputs(
        source_base=source_base,
        source_innovations=source_innovations,
        action_base=action_base,
        action_innovations=action_innovations,
        source_transport=source_transport,
        action_transport=action_transport,
        pre_noise_gate=pre_noise_gate,
        orthogonality_tolerance=orthogonality_tolerance_value,
    )
    variance = _analytic_variance_and_non_iid_audit(
        pre_noise_gate=pre_noise_gate,
        source_row_norm_sq=audit["source_row_norm_sq"],
        action_row_norm_sq=audit["action_row_norm_sq"],
        k_source=k_source_value,
        k_action=k_action_value,
        variance_tolerance=variance_tolerance_value,
        non_iid_tolerance=non_iid_tolerance_value,
    )

    transported_source = torch.einsum(
        "tpq,bcq->bctp", source_transport, source_base
    )
    transported_action = torch.einsum(
        "tpq,bcq->bctp", action_transport, action_base
    )
    source_noise = (
        math.sqrt(k_source_value) * transported_source
        + math.sqrt(1.0 - k_source_value) * source_innovations
    )
    action_noise = (
        math.sqrt(k_action_value) * transported_action
        + math.sqrt(1.0 - k_action_value) * action_innovations
    )
    source_gate = (1.0 - pre_noise_gate.square()).clamp_min(0.0).sqrt()
    output = (source_gate * source_noise + pre_noise_gate * action_noise).contiguous()
    if (
        output.dtype != torch.float32
        or output.requires_grad
        or output.grad_fn is not None
        or tuple(int(item) for item in output.shape)
        != (base_shape[0], base_shape[1], LATENT_PHASES, base_shape[2])
        or not bool(torch.isfinite(output).all().item())
    ):
        raise DMIQDualTransportNoiseError(
            "dual-transport output violates its detached FP32 shape contract"
        )
    input_storage_ids = {_storage_identity(value) for _, value in audit["named"]}
    if _storage_identity(output) in input_storage_ids:
        raise DMIQDualTransportNoiseError("dual-transport output aliases an input")

    diagnostics = DualTransportNoiseDiagnostics(
        shape=tuple(int(item) for item in output.shape),
        dtype=str(output.dtype),
        device=str(output.device),
        k_source=k_source_value,
        k_action=k_action_value,
        gate_min=audit["gate_min"],
        gate_max=audit["gate_max"],
        source_transport_left_orthogonality_max_abs_error=audit["source_left"],
        source_transport_right_orthogonality_max_abs_error=audit["source_right"],
        action_transport_left_orthogonality_max_abs_error=audit["action_left"],
        action_transport_right_orthogonality_max_abs_error=audit["action_right"],
        orthogonality_tolerance=orthogonality_tolerance_value,
        source_variance_coefficient_min=variance["source_min"],
        source_variance_coefficient_max=variance["source_max"],
        action_variance_coefficient_min=variance["action_min"],
        action_variance_coefficient_max=variance["action_max"],
        output_variance_coefficient_min=variance["output_min"],
        output_variance_coefficient_max=variance["output_max"],
        output_variance_coefficient_max_abs_error=variance[
            "output_max_abs_error"
        ],
        variance_tolerance=variance_tolerance_value,
        analytic_per_coordinate_variance_passed=True,
        source_shared_carrier_effective_per_batch=variance["source_active"],
        action_shared_carrier_effective_per_batch=variance["action_active"],
        effective_carrier_rows_per_batch=variance["active_rows"],
        carrier_dimension_upper_bound_per_batch=variance["dimension_bounds"],
        joint_covariance_non_iid_rank_certificate=variance["rank_certificate"],
        joint_covariance_non_iid=True,
        non_iid_tolerance=non_iid_tolerance_value,
        all_inputs_detached=True,
        all_floating_inputs_finite_fp32_contiguous=True,
        permutation_indices_contiguous_int64=None,
        all_input_storages_distinct=True,
        output_storage_distinct_from_inputs=True,
        caller_declared_standard_gaussian_fields=True,
        caller_declared_mutually_independent_gaussian_fields=True,
        statistical_independence_inferable_from_one_realization=False,
        conditional_mean_zero_under_input_contract=True,
    )
    return DualTransportNoiseResult(
        initial_noise=output,
        diagnostics=diagnostics,
        receipt=_build_receipt(diagnostics),
    )


def _validate_permutation_index_tensor(value: Any, *, label: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise DMIQDualTransportNoiseError(f"{label} must be a torch.Tensor")
    if value.dtype != torch.int64:
        raise DMIQDualTransportNoiseError(f"{label} must be torch.int64")
    if value.requires_grad or value.grad_fn is not None:
        raise DMIQDualTransportNoiseError(f"{label} must be detached/no-grad")
    if value.numel() <= 0 or not value.is_contiguous():
        raise DMIQDualTransportNoiseError(
            f"{label} must be non-empty and contiguous"
        )
    return value


def _validate_signed_permutation_inputs(
    *,
    source_base: Any,
    source_innovations: Any,
    action_base: Any,
    action_innovations: Any,
    source_permutation: Any,
    source_signs: Any,
    action_permutation: Any,
    action_signs: Any,
    pre_noise_gate: Any,
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    floating_named = [
        ("source_base", source_base),
        ("source_innovations", source_innovations),
        ("action_base", action_base),
        ("action_innovations", action_innovations),
        ("source_signs", source_signs),
        ("action_signs", action_signs),
        ("pre_noise_gate", pre_noise_gate),
    ]
    for label, value in floating_named:
        _validate_tensor_base(value, label=label)
    _validate_permutation_index_tensor(
        source_permutation, label="source_permutation"
    )
    _validate_permutation_index_tensor(
        action_permutation, label="action_permutation"
    )

    if source_base.ndim != 3:
        raise DMIQDualTransportNoiseError("source_base must be [B,C,P]")
    batch, channels, positions = (int(item) for item in source_base.shape)
    if batch <= 0 or channels <= 0 or positions <= 0:
        raise DMIQDualTransportNoiseError("B, C, and P must be positive")
    base_shape = (batch, channels, positions)
    innovation_shape = (batch, channels, LATENT_PHASES, positions)
    permutation_shape = (LATENT_PHASES, positions)
    gate_shape = (batch, 1, LATENT_PHASES, positions)
    expected_shapes = {
        "source_base": base_shape,
        "source_innovations": innovation_shape,
        "action_base": base_shape,
        "action_innovations": innovation_shape,
        "source_permutation": permutation_shape,
        "source_signs": permutation_shape,
        "action_permutation": permutation_shape,
        "action_signs": permutation_shape,
        "pre_noise_gate": gate_shape,
    }
    all_named = [
        *floating_named[:4],
        ("source_permutation", source_permutation),
        ("source_signs", source_signs),
        ("action_permutation", action_permutation),
        ("action_signs", action_signs),
        ("pre_noise_gate", pre_noise_gate),
    ]
    for label, value in all_named:
        if tuple(int(item) for item in value.shape) != expected_shapes[label]:
            raise DMIQDualTransportNoiseError(
                f"{label} must have shape {expected_shapes[label]}"
            )
        if value.device != source_base.device:
            raise DMIQDualTransportNoiseError(
                "all signed-permutation inputs must share one device"
            )
    _validate_distinct_storages(all_named)

    torch = _torch()
    expected_permutation = torch.arange(
        positions, dtype=torch.int64, device=source_base.device
    ).view(1, positions).expand(LATENT_PHASES, positions)
    for label, permutation in (
        ("source_permutation", source_permutation),
        ("action_permutation", action_permutation),
    ):
        sorted_values = permutation.sort(dim=-1).values
        if not bool(torch.equal(sorted_values, expected_permutation)):
            raise DMIQDualTransportNoiseError(
                f"{label} must contain every index 0..P-1 exactly once per phase"
            )
    for label, signs in (
        ("source_signs", source_signs),
        ("action_signs", action_signs),
    ):
        if not bool(((signs == 1.0) | (signs == -1.0)).all().item()):
            raise DMIQDualTransportNoiseError(
                f"{label} must contain only exact elementwise -1 or +1"
            )

    gate_min = float(pre_noise_gate.min().detach().cpu().item())
    gate_max = float(pre_noise_gate.max().detach().cpu().item())
    if gate_min < 0.0 or gate_max > 1.0:
        raise DMIQDualTransportNoiseError(
            "pre_noise_gate must lie elementwise in [0,1]"
        )
    row_norm_sq = torch.ones(
        (LATENT_PHASES, positions), dtype=torch.float64, device="cpu"
    )
    return base_shape, {
        "gate_min": gate_min,
        "gate_max": gate_max,
        "source_row_norm_sq": row_norm_sq,
        "action_row_norm_sq": row_norm_sq.clone(),
        "named": all_named,
    }


def _apply_signed_permutation(
    base: Any,
    permutation: Any,
    signs: Any,
) -> Any:
    """Apply ``O[p,q]=sign[p] 1[q=permutation[p]]`` without dense ``O``."""

    batch, channels, positions = (int(item) for item in base.shape)
    expanded_base = base.unsqueeze(2).expand(
        batch, channels, LATENT_PHASES, positions
    )
    gather_index = permutation.view(1, 1, LATENT_PHASES, positions).expand(
        batch, channels, LATENT_PHASES, positions
    )
    gathered = _torch().gather(expanded_base, dim=-1, index=gather_index)
    return gathered * signs.view(1, 1, LATENT_PHASES, positions)


def dmiq_dual_signed_permutation_noise(
    source_base: Any,
    source_innovations: Any,
    action_base: Any,
    action_innovations: Any,
    source_permutation_indices: Any,
    source_signs: Any,
    action_permutation_indices: Any,
    action_signs: Any,
    pre_noise_gate: Any,
    *,
    k_source: Real,
    k_action: Real,
    variance_tolerance: Real = DEFAULT_VARIANCE_TOLERANCE,
    non_iid_tolerance: Real = DEFAULT_NON_IID_TOLERANCE,
) -> DualTransportNoiseResult:
    """Practical ``O(21P)`` dual-transport ablation for the 62x60 grid.

    ``P`` is deliberately not forced to 3720 so the operator can be unit-tested
    against an explicit dense signed-permutation matrix at small ``P``.  The
    scientific runtime target remains ``62 * 60 = 3720`` positions.
    """

    torch = _torch()
    k_source_value = _validate_real_unit_interval(k_source, label="k_source")
    k_action_value = _validate_real_unit_interval(k_action, label="k_action")
    variance_tolerance_value = _validate_positive_tolerance(
        variance_tolerance, label="variance_tolerance"
    )
    non_iid_tolerance_value = _validate_positive_tolerance(
        non_iid_tolerance, label="non_iid_tolerance"
    )
    base_shape, audit = _validate_signed_permutation_inputs(
        source_base=source_base,
        source_innovations=source_innovations,
        action_base=action_base,
        action_innovations=action_innovations,
        source_permutation=source_permutation_indices,
        source_signs=source_signs,
        action_permutation=action_permutation_indices,
        action_signs=action_signs,
        pre_noise_gate=pre_noise_gate,
    )
    variance = _analytic_variance_and_non_iid_audit(
        pre_noise_gate=pre_noise_gate,
        source_row_norm_sq=audit["source_row_norm_sq"],
        action_row_norm_sq=audit["action_row_norm_sq"],
        k_source=k_source_value,
        k_action=k_action_value,
        variance_tolerance=variance_tolerance_value,
        non_iid_tolerance=non_iid_tolerance_value,
    )

    transported_source = _apply_signed_permutation(
        source_base, source_permutation_indices, source_signs
    )
    transported_action = _apply_signed_permutation(
        action_base, action_permutation_indices, action_signs
    )
    source_noise = (
        math.sqrt(k_source_value) * transported_source
        + math.sqrt(1.0 - k_source_value) * source_innovations
    )
    action_noise = (
        math.sqrt(k_action_value) * transported_action
        + math.sqrt(1.0 - k_action_value) * action_innovations
    )
    source_gate = (1.0 - pre_noise_gate.square()).clamp_min(0.0).sqrt()
    output = (source_gate * source_noise + pre_noise_gate * action_noise).contiguous()
    expected_output_shape = (
        base_shape[0],
        base_shape[1],
        LATENT_PHASES,
        base_shape[2],
    )
    if (
        output.dtype != torch.float32
        or output.requires_grad
        or output.grad_fn is not None
        or tuple(int(item) for item in output.shape) != expected_output_shape
        or not bool(torch.isfinite(output).all().item())
    ):
        raise DMIQDualTransportNoiseError(
            "signed-permutation output violates its detached FP32 shape contract"
        )
    input_storage_ids = {_storage_identity(value) for _, value in audit["named"]}
    if _storage_identity(output) in input_storage_ids:
        raise DMIQDualTransportNoiseError(
            "signed-permutation output aliases an input"
        )

    diagnostics = DualTransportNoiseDiagnostics(
        shape=tuple(int(item) for item in output.shape),
        dtype=str(output.dtype),
        device=str(output.device),
        k_source=k_source_value,
        k_action=k_action_value,
        gate_min=audit["gate_min"],
        gate_max=audit["gate_max"],
        source_transport_left_orthogonality_max_abs_error=0.0,
        source_transport_right_orthogonality_max_abs_error=0.0,
        action_transport_left_orthogonality_max_abs_error=0.0,
        action_transport_right_orthogonality_max_abs_error=0.0,
        orthogonality_tolerance=0.0,
        source_variance_coefficient_min=variance["source_min"],
        source_variance_coefficient_max=variance["source_max"],
        action_variance_coefficient_min=variance["action_min"],
        action_variance_coefficient_max=variance["action_max"],
        output_variance_coefficient_min=variance["output_min"],
        output_variance_coefficient_max=variance["output_max"],
        output_variance_coefficient_max_abs_error=variance[
            "output_max_abs_error"
        ],
        variance_tolerance=variance_tolerance_value,
        analytic_per_coordinate_variance_passed=True,
        source_shared_carrier_effective_per_batch=variance["source_active"],
        action_shared_carrier_effective_per_batch=variance["action_active"],
        effective_carrier_rows_per_batch=variance["active_rows"],
        carrier_dimension_upper_bound_per_batch=variance["dimension_bounds"],
        joint_covariance_non_iid_rank_certificate=variance["rank_certificate"],
        joint_covariance_non_iid=True,
        non_iid_tolerance=non_iid_tolerance_value,
        all_inputs_detached=True,
        all_floating_inputs_finite_fp32_contiguous=True,
        permutation_indices_contiguous_int64=True,
        all_input_storages_distinct=True,
        output_storage_distinct_from_inputs=True,
        caller_declared_standard_gaussian_fields=True,
        caller_declared_mutually_independent_gaussian_fields=True,
        statistical_independence_inferable_from_one_realization=False,
        conditional_mean_zero_under_input_contract=True,
    )
    return DualTransportNoiseResult(
        initial_noise=output,
        diagnostics=diagnostics,
        receipt=_build_receipt(
            diagnostics,
            transport_representation="implicit_signed_permutation",
        ),
    )


def polar_orthogonalize_cpu(
    candidate: Any,
    *,
    singular_value_floor: Real = 1.0e-10,
    orthogonality_tolerance: Real = 3.0e-5,
) -> PolarOrthogonalizationResult:
    """Project ``[21,P,P]`` matrices to their polar factors offline.

    The SVD and polar product are always performed in FP64 on CPU.  The
    returned transport is a fresh detached contiguous FP32 CPU tensor suitable
    for freezing and later transfer to the sampler device.
    """

    torch = _torch()
    if not isinstance(candidate, torch.Tensor):
        raise DMIQDualTransportNoiseError("polar candidate must be a tensor")
    if candidate.dtype not in (torch.float32, torch.float64):
        raise DMIQDualTransportNoiseError("polar candidate must be FP32 or FP64")
    if candidate.device.type != "cpu":
        raise DMIQDualTransportNoiseError(
            "polar orthogonalization is an offline CPU-only operation"
        )
    if candidate.requires_grad or candidate.grad_fn is not None:
        raise DMIQDualTransportNoiseError("polar candidate must be detached")
    if (
        candidate.ndim != 3
        or int(candidate.shape[0]) != LATENT_PHASES
        or int(candidate.shape[1]) <= 0
        or int(candidate.shape[1]) != int(candidate.shape[2])
    ):
        raise DMIQDualTransportNoiseError(
            "polar candidate must be [21,P,P] with positive P"
        )
    if not candidate.is_contiguous() or not bool(torch.isfinite(candidate).all()):
        raise DMIQDualTransportNoiseError(
            "polar candidate must be finite and contiguous"
        )
    floor = _validate_positive_tolerance(
        singular_value_floor, label="singular_value_floor"
    )
    tolerance = _validate_positive_tolerance(
        orthogonality_tolerance, label="orthogonality_tolerance"
    )

    work = candidate.detach().to(dtype=torch.float64).contiguous()
    left, singular_values, right_h = torch.linalg.svd(
        work, full_matrices=False
    )
    minimum = float(singular_values.min().item())
    maximum = float(singular_values.max().item())
    if minimum <= floor:
        raise DMIQDualTransportNoiseError(
            "polar candidate is singular or below the singular-value floor"
        )
    polar64 = torch.matmul(left, right_h)
    transport = polar64.to(dtype=torch.float32).contiguous().detach()
    left_error, right_error, _ = _orthogonality_errors(transport)
    if max(left_error, right_error) > tolerance:
        raise DMIQDualTransportNoiseError(
            "FP32 polar transport exceeds orthogonality tolerance"
        )
    if _storage_identity(transport) == _storage_identity(candidate):
        raise DMIQDualTransportNoiseError("polar output aliases its candidate")
    receipt = {
        "schema_version": "dmiq-offline-polar-orthogonalization-v1",
        "offline_only": True,
        "computation_device": "cpu",
        "computation_dtype": "torch.float64",
        "output_device": "cpu",
        "output_dtype": "torch.float32",
        "input_shape": [int(item) for item in candidate.shape],
        "minimum_input_singular_value": minimum,
        "maximum_input_singular_value": maximum,
        "input_condition_number": maximum / minimum,
        "singular_value_floor": floor,
        "left_orthogonality_max_abs_error": left_error,
        "right_orthogonality_max_abs_error": right_error,
        "orthogonality_tolerance": tolerance,
        "output_detached": True,
        "output_storage_distinct_from_input": True,
    }
    return PolarOrthogonalizationResult(transport=transport, receipt=receipt)


dual_transport_noise = dmiq_dual_transport_noise


__all__ = [
    "DEFAULT_NON_IID_TOLERANCE",
    "DEFAULT_ORTHOGONALITY_TOLERANCE",
    "DEFAULT_VARIANCE_TOLERANCE",
    "DMIQDualTransportNoiseError",
    "DualTransportNoiseDiagnostics",
    "DualTransportNoiseResult",
    "LATENT_PHASES",
    "PRACTICAL_LATENT_HEIGHT",
    "PRACTICAL_LATENT_POSITIONS",
    "PRACTICAL_LATENT_WIDTH",
    "PolarOrthogonalizationResult",
    "RECEIPT_SCHEMA",
    "dmiq_dual_transport_noise",
    "dmiq_dual_signed_permutation_noise",
    "dual_transport_noise",
    "polar_orthogonalize_cpu",
]
