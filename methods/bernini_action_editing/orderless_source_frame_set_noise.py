"""Orderless source-frame-set initial-noise ablation for exact-81 Bernini.

This operator is a deliberately narrow alternative to
``motion_null_appearance_noise``.  The older operator discards source spatial
phase and low-frequency layout before synthesising a random texture.  That is
useful as an appearance-statistics control, but it is a weak identity carrier.
Here, a permutation-invariant robust prototype is built directly from
independently encoded ``T=1`` source-image latents:

* a coordinate-wise set median captures content shared across the frames;
* the set medoid retains one non-averaged, source-real spatial realisation; and
* their fixed 50/50 barycenter retains source low-frequency layout and spatial
  phase without assigning any member a temporal position.

The barycenter is repeated unchanged over all 21 exact-81 latent phases.  It
is orthogonalised against, and norm-matched to, the centred temporal-DC part
of Bernini's official realised Gaussian.  Only that Gaussian DC component is
rotated.  The scalar Gaussian mean and the entire Gaussian temporal residual
are preserved.  Thus the carrier cannot encode source chronology, direction,
speed, or phase order.  It can still encode a static pose and the unordered
occupancy distribution of source poses; consequently it does *not* prove that
old-action semantics have been removed.

The only registered arms are ``rho in {0, 0.05, 0.10}``.  ``rho == 0`` returns
the exact Gaussian tensor object.  Active arms are source-conditioned and
non-Gaussian.  This module is a pure, keyword-only tensor operator: it does not
self-register a sampler hook, trainer, launcher, reward, or optimizer.  Train
and inference must call this same function in the same unpacked clean-epsilon
coordinate: after canonical Gaussian creation and before either training-time
sigma interpolation or inference-time latent packing.  At native Bernini
inference that tensor is the return of the pinned module-global
``randn_tensor``.  The receipt authorises only an isolated factorial ablation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-orderless-source-frame-set-noise-v1"
METHOD = "orderless_source_frame_set_dc_transport"
EXACT_VIDEO_FRAMES = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
MIN_SOURCE_FRAMES = 2
MAX_SOURCE_FRAMES = EXACT_VIDEO_FRAMES
MIN_SPATIAL_EXTENT = 4
FACTORIAL_RHOS = (0.0, 0.05, 0.10)
PROTOTYPE_RULE = "half_coordinate_median_plus_half_set_medoid"


class OrderlessSourceFrameSetNoiseError(RuntimeError):
    """Raised before a non-auditable carrier or endpoint can be returned."""


@dataclass(frozen=True)
class OrderlessSourceFrameSetDiagnostics:
    """Serializable audit of one source-set transport call."""

    rho: float
    gaussian_shape: tuple[int, ...]
    source_frame_count: int
    dtype: str
    device: str
    rho_zero_exact_object_alias: bool
    source_conditioned_non_gaussian: bool
    source_storage_isolation_verified: bool
    source_encoder_invocation_proven_by_operator: bool
    source_frame_order_consumed: bool
    source_frame_indices_consumed: bool
    source_temporal_phase_consumed: bool
    source_spatial_phase_consumed: bool
    source_low_frequency_layout_consumed: bool
    source_pose_occupancy_may_be_retained: bool
    carrier_strict_temporal_dc: bool
    canonical_gaussian_sha256: str
    source_frame_multiset_sha256: str
    source_set_prototype_sha256: Optional[str]
    selected_medoid_value_sha256: Optional[str]
    transported_carrier_sha256: Optional[str]
    initial_noise_sha256: str
    gaussian_scalar_mean_max_abs_error: float
    gaussian_total_norm_max_relative_error: float
    gaussian_centered_dc_norm_max_relative_error: float
    gaussian_temporal_residual_max_abs_error: float
    carrier_temporal_dc_max_abs_error: float
    carrier_gaussian_dc_normalized_dot_max: float
    carrier_gaussian_dc_norm_max_relative_error: float
    transported_to_raw_source_carrier_normalized_dot_min: float
    numerical_audit_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderlessSourceFrameSetNoiseResult:
    """Initial endpoint plus inspectable carrier/prototype and strict receipt."""

    initial_noise: Any
    source_set_prototype: Optional[Any]
    temporal_dc_carrier: Optional[Any]
    diagnostics: OrderlessSourceFrameSetDiagnostics
    receipt: Mapping[str, Any]
    receipt_sha256: str


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH supplies torch
        raise OrderlessSourceFrameSetNoiseError(
            "orderless source-frame-set noise requires PyTorch"
        ) from error
    return torch


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
        raise OrderlessSourceFrameSetNoiseError(
            f"receipt is not canonical finite ASCII JSON: {error}"
        ) from error


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise OrderlessSourceFrameSetNoiseError(
            "tensor digest requires a materialized tensor"
        )
    tensor = value.detach().contiguous().cpu()
    metadata = _canonical_json_bytes(
        {"shape": [int(item) for item in tensor.shape], "dtype": str(tensor.dtype)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_rho(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OrderlessSourceFrameSetNoiseError("rho must be a real scalar")
    rho = float(value)
    if not math.isfinite(rho) or rho not in FACTORIAL_RHOS:
        raise OrderlessSourceFrameSetNoiseError(
            f"rho must be one registered factorial arm: {FACTORIAL_RHOS}"
        )
    return rho


def _is_standalone_storage(value: Any) -> bool:
    expected_bytes = int(value.numel()) * int(value.element_size())
    try:
        storage_bytes = int(value.untyped_storage().nbytes())
    except (AttributeError, RuntimeError):  # pragma: no cover - old torch
        storage_bytes = int(value.storage().nbytes())
    return (
        value._base is None
        and int(value.storage_offset()) == 0
        and storage_bytes == expected_bytes
    )


def _validate_gaussian(value: Any) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise OrderlessSourceFrameSetNoiseError(
            "canonical_gaussian must be a torch.Tensor"
        )
    if value.dtype not in (torch.float32, torch.float64):
        raise OrderlessSourceFrameSetNoiseError(
            "canonical_gaussian must be FP32 or FP64"
        )
    if (
        value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) < MIN_SPATIAL_EXTENT
        or int(value.shape[4]) < MIN_SPATIAL_EXTENT
        or not bool(torch.isfinite(value).all().item())
    ):
        raise OrderlessSourceFrameSetNoiseError(
            "canonical_gaussian must be detached contiguous finite "
            f"[1,{LATENT_CHANNELS},{LATENT_PHASES},H,W]"
        )
    return value


def _validate_source_frames(
    values: Sequence[Any], gaussian: Any
) -> tuple[Any, ...]:
    torch = _torch()
    if isinstance(values, torch.Tensor) or not isinstance(values, (tuple, list)):
        raise OrderlessSourceFrameSetNoiseError(
            "independent_frame_latents must be a tuple/list of standalone T=1 "
            "latents; a full-video tensor is forbidden"
        )
    frames = tuple(values)
    if not MIN_SOURCE_FRAMES <= len(frames) <= MAX_SOURCE_FRAMES:
        raise OrderlessSourceFrameSetNoiseError(
            f"independent_frame_latents must contain {MIN_SOURCE_FRAMES}.."
            f"{MAX_SOURCE_FRAMES} frames"
        )
    expected_shape = (
        1,
        LATENT_CHANNELS,
        1,
        int(gaussian.shape[3]),
        int(gaussian.shape[4]),
    )
    pointers: set[tuple[str, int]] = set()
    gaussian_pointer = (
        str(gaussian.device),
        int(gaussian.untyped_storage().data_ptr()),
    )
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, torch.Tensor):
            raise OrderlessSourceFrameSetNoiseError(
                f"source frame {ordinal} is not a tensor"
            )
        if (
            frame.dtype != gaussian.dtype
            or frame.device != gaussian.device
            or frame.layout != torch.strided
            or frame.requires_grad
            or frame.grad_fn is not None
            or not frame.is_contiguous()
            or tuple(int(item) for item in frame.shape) != expected_shape
            or not bool(torch.isfinite(frame).all().item())
            or not _is_standalone_storage(frame)
        ):
            raise OrderlessSourceFrameSetNoiseError(
                f"source frame {ordinal} must be a standalone detached finite "
                f"T=1 latent with shape {expected_shape}"
            )
        pointer = (str(frame.device), int(frame.untyped_storage().data_ptr()))
        if pointer == gaussian_pointer or pointer in pointers:
            raise OrderlessSourceFrameSetNoiseError(
                "source frames must not alias the Gaussian or each other"
            )
        pointers.add(pointer)
    return frames


def _frame_multiset_digest(frames: tuple[Any, ...]) -> tuple[str, tuple[str, ...]]:
    value_digests = tuple(sorted(_tensor_sha256(frame) for frame in frames))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "member_value_sha256_multiset": list(value_digests),
        "multiplicity_retained": True,
        "input_sequence_order_retained": False,
        "frame_indices_retained": False,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(), value_digests


def _source_set_prototype(
    frames: tuple[Any, ...], value_digests: tuple[str, ...]
) -> tuple[Any, str]:
    """Return a robust layout-preserving prototype and selected medoid hash.

    Frames are first put in content-hash order.  All subsequent reductions are
    therefore independent of caller order, including floating-point execution
    order and medoid tie breaking.
    """

    torch = _torch()
    rows = [
        (
            _tensor_sha256(frame),
            frame.detach().cpu().double()[0, :, 0].contiguous(),
        )
        for frame in frames
    ]
    # A key function is required because a legitimate multiset may contain
    # independently stored but byte-identical frames; tuple sorting would then
    # try to order the tensor payloads themselves.
    rows.sort(key=lambda row: row[0])
    if tuple(row[0] for row in rows) != value_digests:
        raise OrderlessSourceFrameSetNoiseError(
            "source multiset changed while constructing its prototype"
        )
    values = torch.stack([row[1] for row in rows], dim=0).contiguous()
    coordinate_sorted = torch.sort(values, dim=0).values
    count = int(coordinate_sorted.shape[0])
    if count % 2:
        coordinate_median = coordinate_sorted[count // 2]
    else:
        coordinate_median = 0.5 * (
            coordinate_sorted[count // 2 - 1] + coordinate_sorted[count // 2]
        )
    # A real set medoid minimizes aggregate distance to every member.  The
    # previous prototype used distance to the coordinate median, which is a
    # useful robust exemplar but is not the mathematical medoid named in the
    # contract.  The Gram formulation avoids materializing [N,N,C,H,W].
    flattened = values.flatten(1)
    squared_norms = flattened.square().sum(dim=1)
    pairwise_squared_l2 = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (flattened @ flattened.transpose(0, 1))
    ).clamp_min_(0.0)
    aggregate_distances = pairwise_squared_l2.sum(dim=1) / float(
        flattened.shape[1]
    )
    candidates = [
        (float(aggregate_distances[index].item()), rows[index][0], index)
        for index in range(count)
    ]
    _, medoid_digest, medoid_index = min(candidates)
    medoid = values[medoid_index]
    prototype = 0.5 * coordinate_median + 0.5 * medoid
    if (
        not bool(torch.isfinite(prototype).all().item())
        or float(prototype.square().sum().sqrt().item()) <= 1.0e-12
    ):
        raise OrderlessSourceFrameSetNoiseError(
            "source frame set produced a degenerate prototype"
        )
    return prototype.contiguous(), medoid_digest


def _flat_l2(value: Any) -> Any:
    return value.flatten(1).square().sum(dim=1).sqrt()


def _normalized_dot_max(left: Any, right: Any) -> float:
    torch = _torch()
    numerator = (left.flatten(1) * right.flatten(1)).sum(dim=1).abs()
    denominator = _flat_l2(left) * _flat_l2(right)
    ratio = torch.where(
        denominator > torch.finfo(left.dtype).tiny,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return float(ratio.max().item())


def _normalized_dot_min(left: Any, right: Any) -> float:
    torch = _torch()
    numerator = (left.flatten(1) * right.flatten(1)).sum(dim=1)
    denominator = _flat_l2(left) * _flat_l2(right)
    ratio = torch.where(
        denominator > torch.finfo(left.dtype).tiny,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return float(ratio.min().item())


def _relative_error_max(actual: Any, expected: Any) -> float:
    torch = _torch()
    denominator = torch.maximum(expected.abs(), torch.ones_like(expected))
    return float(((actual - expected).abs() / denominator).max().item())


def _transport_into_gaussian_dc(
    gaussian: Any,
    prototype: Any,
    *,
    rho: float,
) -> tuple[Any, Any, dict[str, float]]:
    torch = _torch()
    work = gaussian.detach().cpu().double()
    height, width = int(work.shape[-2]), int(work.shape[-1])
    raw_phase = prototype.reshape(1, LATENT_CHANNELS, 1, height, width)
    raw_phase = raw_phase - raw_phase.mean()
    raw_carrier = raw_phase.expand_as(work).contiguous()
    if float(_flat_l2(raw_carrier).min().item()) <= 1.0e-12:
        raise OrderlessSourceFrameSetNoiseError(
            "centred source-set prototype is degenerate"
        )

    gaussian_dc_phase = work.mean(dim=2, keepdim=True)
    scalar_mean = gaussian_dc_phase.mean(dim=(1, 2, 3, 4), keepdim=True)
    centered_dc = (gaussian_dc_phase - scalar_mean).expand_as(work)
    temporal_residual = work - gaussian_dc_phase.expand_as(work)
    dc_energy = centered_dc.flatten(1).square().sum(dim=1, keepdim=True)
    if bool((dc_energy <= 1.0e-20).any().item()):
        raise OrderlessSourceFrameSetNoiseError(
            "canonical Gaussian centred temporal-DC subspace is degenerate"
        )

    view = (int(work.shape[0]),) + (1,) * (work.ndim - 1)
    projection = (
        (raw_carrier.flatten(1) * centered_dc.flatten(1)).sum(dim=1, keepdim=True)
        / dc_energy
    )
    orthogonal = raw_carrier - projection.reshape(view) * centered_dc
    orthogonal = orthogonal - orthogonal.mean()
    projection_again = (
        (orthogonal.flatten(1) * centered_dc.flatten(1)).sum(dim=1, keepdim=True)
        / dc_energy
    )
    orthogonal = orthogonal - projection_again.reshape(view) * centered_dc
    orthogonal_norm = _flat_l2(orthogonal)
    centered_dc_norm = _flat_l2(centered_dc)
    if bool((orthogonal_norm <= 1.0e-12).any().item()):
        raise OrderlessSourceFrameSetNoiseError(
            "source-set carrier is degenerate or collinear with Gaussian DC"
        )
    transported = orthogonal * (centered_dc_norm / orthogonal_norm).reshape(view)
    mixed_dc = (
        math.sqrt(max(0.0, 1.0 - rho * rho)) * centered_dc
        + rho * transported
    )
    output64 = scalar_mean.expand_as(work) + temporal_residual + mixed_dc
    output = output64.to(dtype=gaussian.dtype, device=gaussian.device).contiguous()
    carrier = transported.to(dtype=gaussian.dtype, device=gaussian.device).contiguous()
    if (
        not bool(torch.isfinite(output).all().item())
        or output.requires_grad
        or output.grad_fn is not None
    ):
        raise OrderlessSourceFrameSetNoiseError(
            "source-set transport produced invalid initial noise"
        )

    realised = output.detach().cpu().double()
    realised_dc_phase = realised.mean(dim=2, keepdim=True)
    realised_scalar = realised_dc_phase.mean(dim=(1, 2, 3, 4), keepdim=True)
    realised_centered_dc = (realised_dc_phase - realised_scalar).expand_as(realised)
    realised_residual = realised - realised_dc_phase.expand_as(realised)
    carrier64 = carrier.detach().cpu().double()
    carrier_dc_error = float(
        (carrier64 - carrier64[:, :, :1].expand_as(carrier64)).abs().max().item()
    )
    metrics = {
        "scalar_mean_error": float((realised_scalar - scalar_mean).abs().max().item()),
        "total_norm_error": _relative_error_max(_flat_l2(realised), _flat_l2(work)),
        "dc_norm_error": _relative_error_max(
            _flat_l2(realised_centered_dc), _flat_l2(centered_dc)
        ),
        "residual_error": float((realised_residual - temporal_residual).abs().max().item()),
        "carrier_dc_error": carrier_dc_error,
        "carrier_dot": _normalized_dot_max(transported, centered_dc),
        "carrier_norm_error": _relative_error_max(
            _flat_l2(transported), _flat_l2(centered_dc)
        ),
        "source_alignment": _normalized_dot_min(transported, raw_carrier),
    }
    scale = max(1.0, float(work.abs().max().item()))
    if gaussian.dtype == torch.float64:
        absolute_tolerance, relative_tolerance = 2.0e-10, 2.0e-10
    else:
        absolute_tolerance, relative_tolerance = 8.0e-6, 6.0e-5
    if (
        metrics["scalar_mean_error"] > absolute_tolerance * scale
        or metrics["total_norm_error"] > relative_tolerance
        or metrics["dc_norm_error"] > relative_tolerance
        or metrics["residual_error"] > absolute_tolerance * scale
        or metrics["carrier_dc_error"] != 0.0
        or metrics["carrier_dot"] > 2.0e-10
        or metrics["carrier_norm_error"] > 2.0e-10
        or not 0.0 < metrics["source_alignment"] <= 1.0 + 2.0e-12
    ):
        raise OrderlessSourceFrameSetNoiseError(
            "source-frame-set noise failed its numerical invariants"
        )
    return output, carrier, metrics


def _receipt(diagnostics: OrderlessSourceFrameSetDiagnostics) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "ablation_only": True,
        "allowed_factorial_rhos": list(FACTORIAL_RHOS),
        "editor_optimizer_authorized": False,
        "editor_training_authorized": False,
        "critic_reward_authorized": False,
        "scientific_success_claim_authorized": False,
        "semantic_old_action_absence_claimed": False,
        "operator_self_registers_sampler_hook": False,
        "operator_self_registers_trainer": False,
        "operator_self_registers_launcher": False,
        "public_api_inputs": [
            "canonical_gaussian",
            "independent_frame_latents",
            "rho",
        ],
        "forbidden_api_inputs": [
            "full_source_video_latent",
            "ordered_source_trajectory",
            "source_frame_indices",
            "target",
            "paired_target",
            "action_proposal",
            "motion_reference",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
            "trainer_state",
            "inference_stage",
        ],
        "source_contract": {
            "standalone_T1_latents_required": True,
            "full_video_tensor_rejected": True,
            "shared_storage_views_rejected": True,
            "storage_isolation_verified": diagnostics.source_storage_isolation_verified,
            "independent_per_frame_VAE_invocations_required_externally": True,
            "independent_encoder_invocation_proven_by_operator": False,
            "input_is_an_unordered_multiset": True,
            "multiplicity_retained": True,
            "caller_sequence_order_consumed": False,
            "source_frame_indices_consumed": False,
            "operator_received_member_selection_indices": False,
            "external_member_selection_may_have_used_indices": True,
            "operator_proves_index_free_member_selection": False,
        },
        "prototype": {
            "rule": PROTOTYPE_RULE,
            "coordinatewise_set_median_weight": 0.5,
            "set_medoid_weight": 0.5,
            "set_medoid_objective": (
                "minimum_sum_pairwise_mean_squared_distance_to_all_multiset_members"
            ),
            "medoid_tie_break": "ascending_frame_tensor_sha256",
            "source_spatial_phase_retained": True,
            "source_low_frequency_layout_retained": True,
            "source_frame_temporal_positions_retained": False,
        },
        "carrier": {
            "strict_temporal_dc": diagnostics.carrier_strict_temporal_dc,
            "source_order_retained": False,
            "source_temporal_phase_retained": False,
            "source_chronology_or_direction_representable": False,
            "static_pose_can_be_retained": True,
            "unordered_pose_occupancy_can_be_retained": True,
        },
        "mix": {
            "rotated_subspace": "centered_official_gaussian_temporal_dc_only",
            "scalar_gaussian_mean_preserved": True,
            "official_gaussian_temporal_residual_preserved": True,
            "centered_gaussian_dc_norm_preserved": True,
            "total_realized_l2_norm_preserved": True,
            "rho_parameterization": (
                "sqrt(1-rho^2)*gaussian_dc+rho*orthogonal_source_set_carrier"
            ),
        },
        "distribution": {
            "rho_zero_exact_official_gaussian_object_alias": (
                diagnostics.rho_zero_exact_object_alias
            ),
            "rho_positive_source_conditioned_non_gaussian": (
                diagnostics.source_conditioned_non_gaussian
            ),
            "gaussianity_claimed_for_active_rho": False,
        },
        "train_inference_contract": {
            "same_callable_required": (
                "orderless_source_frame_set_noise."
                "build_orderless_source_frame_set_noise"
            ),
            "same_registered_rho_required": True,
            "same_tensor_coordinate_required": (
                "unpacked_[1,16,21,H,W]_clean_epsilon_after_canonical_gaussian_"
                "creation_before_sigma_interpolation_or_patch_packing"
            ),
            "inference_binding": (
                "return_of_pinned_bernini.models.wan_diffusion.randn_tensor_"
                "before_rearrange"
            ),
            "training_binding": (
                "canonical_epsilon_immediately_after_sampling_before_flow_"
                "sigma_interpolation_and_patch_packing"
            ),
            "inference_official_randn_tensor_must_be_called_first_with_unchanged_arguments": True,
            "rho_zero_must_forward_original_native_tensor_object": True,
            "active_rho_is_external_initial_noise_injection": True,
            "stage_dependent_operator_branch": False,
        },
        "required_external_factorial_controls": {
            "same_source_instruction_seed_scheduler_guidance": True,
            "arms": [
                {"name": "official_gaussian", "rho": 0.0},
                {"name": "orderless_source_set_rho005", "rho": 0.05},
                {"name": "orderless_source_set_rho010", "rho": 0.10},
            ],
            "wrong_source_counterfactual_required": True,
        },
        "required_external_gates": [
            "heldout_identity_retrieval_correct_source_beats_wrong_source",
            "old_action_direction_and_milestone_order_nonincrease",
            "target_action_noninferiority_to_rho0",
            "blur_flicker_camera_quality_noninferiority",
        ],
        "scientific_risks": [
            "active endpoint is not Gaussian and is off the native training distribution",
            "a static medoid pose can oppose the requested new action",
            "unordered pose occupancy can correlate with the old action label",
            "coordinatewise median can create a non-decodable hybrid latent",
            "source camera motion can blur or destabilize the set prototype",
            "storage isolation cannot prove independent per-frame VAE invocation provenance",
        ],
        "diagnostics": diagnostics.as_dict(),
    }


def build_orderless_source_frame_set_noise(
    *,
    canonical_gaussian: Any,
    independent_frame_latents: Sequence[Any],
    rho: Real,
) -> OrderlessSourceFrameSetNoiseResult:
    """Build one registered orderless source-set initial-noise endpoint."""

    gaussian = _validate_gaussian(canonical_gaussian)
    frames = _validate_source_frames(independent_frame_latents, gaussian)
    rho_value = _validate_rho(rho)
    gaussian_sha = _tensor_sha256(gaussian)
    multiset_sha, value_digests = _frame_multiset_digest(frames)

    if rho_value == 0.0:
        diagnostics = OrderlessSourceFrameSetDiagnostics(
            rho=0.0,
            gaussian_shape=tuple(int(item) for item in gaussian.shape),
            source_frame_count=len(frames),
            dtype=str(gaussian.dtype),
            device=str(gaussian.device),
            rho_zero_exact_object_alias=True,
            source_conditioned_non_gaussian=False,
            source_storage_isolation_verified=True,
            source_encoder_invocation_proven_by_operator=False,
            source_frame_order_consumed=False,
            source_frame_indices_consumed=False,
            source_temporal_phase_consumed=False,
            source_spatial_phase_consumed=False,
            source_low_frequency_layout_consumed=False,
            source_pose_occupancy_may_be_retained=False,
            carrier_strict_temporal_dc=True,
            canonical_gaussian_sha256=gaussian_sha,
            source_frame_multiset_sha256=multiset_sha,
            source_set_prototype_sha256=None,
            selected_medoid_value_sha256=None,
            transported_carrier_sha256=None,
            initial_noise_sha256=gaussian_sha,
            gaussian_scalar_mean_max_abs_error=0.0,
            gaussian_total_norm_max_relative_error=0.0,
            gaussian_centered_dc_norm_max_relative_error=0.0,
            gaussian_temporal_residual_max_abs_error=0.0,
            carrier_temporal_dc_max_abs_error=0.0,
            carrier_gaussian_dc_normalized_dot_max=0.0,
            carrier_gaussian_dc_norm_max_relative_error=0.0,
            transported_to_raw_source_carrier_normalized_dot_min=0.0,
            numerical_audit_passed=True,
        )
        receipt = _receipt(diagnostics)
        return OrderlessSourceFrameSetNoiseResult(
            initial_noise=gaussian,
            source_set_prototype=None,
            temporal_dc_carrier=None,
            diagnostics=diagnostics,
            receipt=receipt,
            receipt_sha256=hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest(),
        )

    prototype64, medoid_sha = _source_set_prototype(frames, value_digests)
    prototype = prototype64.reshape(
        1,
        LATENT_CHANNELS,
        1,
        int(gaussian.shape[-2]),
        int(gaussian.shape[-1]),
    ).to(dtype=gaussian.dtype, device=gaussian.device).contiguous()
    output, carrier, metrics = _transport_into_gaussian_dc(
        gaussian, prototype64, rho=rho_value
    )
    diagnostics = OrderlessSourceFrameSetDiagnostics(
        rho=rho_value,
        gaussian_shape=tuple(int(item) for item in gaussian.shape),
        source_frame_count=len(frames),
        dtype=str(gaussian.dtype),
        device=str(gaussian.device),
        rho_zero_exact_object_alias=False,
        source_conditioned_non_gaussian=True,
        source_storage_isolation_verified=True,
        source_encoder_invocation_proven_by_operator=False,
        source_frame_order_consumed=False,
        source_frame_indices_consumed=False,
        source_temporal_phase_consumed=False,
        source_spatial_phase_consumed=True,
        source_low_frequency_layout_consumed=True,
        source_pose_occupancy_may_be_retained=True,
        carrier_strict_temporal_dc=metrics["carrier_dc_error"] == 0.0,
        canonical_gaussian_sha256=gaussian_sha,
        source_frame_multiset_sha256=multiset_sha,
        source_set_prototype_sha256=_tensor_sha256(prototype),
        selected_medoid_value_sha256=medoid_sha,
        transported_carrier_sha256=_tensor_sha256(carrier),
        initial_noise_sha256=_tensor_sha256(output),
        gaussian_scalar_mean_max_abs_error=metrics["scalar_mean_error"],
        gaussian_total_norm_max_relative_error=metrics["total_norm_error"],
        gaussian_centered_dc_norm_max_relative_error=metrics["dc_norm_error"],
        gaussian_temporal_residual_max_abs_error=metrics["residual_error"],
        carrier_temporal_dc_max_abs_error=metrics["carrier_dc_error"],
        carrier_gaussian_dc_normalized_dot_max=metrics["carrier_dot"],
        carrier_gaussian_dc_norm_max_relative_error=metrics["carrier_norm_error"],
        transported_to_raw_source_carrier_normalized_dot_min=metrics[
            "source_alignment"
        ],
        numerical_audit_passed=True,
    )
    receipt = _receipt(diagnostics)
    return OrderlessSourceFrameSetNoiseResult(
        initial_noise=output,
        source_set_prototype=prototype,
        temporal_dc_carrier=carrier,
        diagnostics=diagnostics,
        receipt=receipt,
        receipt_sha256=hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest(),
    )


__all__ = [
    "EXACT_VIDEO_FRAMES",
    "FACTORIAL_RHOS",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "MAX_SOURCE_FRAMES",
    "METHOD",
    "MIN_SOURCE_FRAMES",
    "OrderlessSourceFrameSetDiagnostics",
    "OrderlessSourceFrameSetNoiseError",
    "OrderlessSourceFrameSetNoiseResult",
    "PROTOTYPE_RULE",
    "SCHEMA_VERSION",
    "build_orderless_source_frame_set_noise",
]
