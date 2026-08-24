#!/usr/bin/env python3
"""Counterfactual identity-orbit and motion-equivariance core for Bernini v4.

V4 replaces undefined ``reverse/wrong/off should have high error`` negatives
with exact positive supervision.  An orbit contains three videos with the
same motion, camera and scene but different appearances.  For every ordered
pair ``(m,n)``, the donor is member ``V_m``, the four native image references
come from ``V_n``, and the exact target is ``V_n``.  The full 3x3 Cartesian
product therefore teaches donor-identity invariance and reference-identity
selection without asking the optimizer to inflate an arbitrary negative.

A preregistered temporal transform is applied to donor and target together,
while references remain independently encoded from the untransformed target
orbit member.  Identity, reversal, and one endpoint-preserving monotonic warp
are included.  Every transformed target remains exact and auditable.

This module imports v3's native Bernini ``none/V/I/VI`` pack and pinned exact40
UniPC schedule rather than duplicating either contract.  Cross-scene wrong
references have no defined target and are accepted only by a held-out gate;
they are structurally absent from the training objective.

The optional source-rich base is conditioned on an orbit/source latent.  A
fixed temporal permutation is whitened, made orthogonal to the Gaussian and
norm matched before mixing.  ``rho == 0`` returns the original Gaussian
values exactly.  ``rho > 0`` is explicitly non-Gaussian.  Because rho varies
with sigma, the flow target includes ``sigma * dz/dsigma``; silently retaining
``epsilon-clean`` would train the wrong vector field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Optional, Sequence

import torch

import source_self_native_ref_contrastive_v3 as v3


SCHEMA_VERSION = "bernini-source-self-identity-orbit-v5"
ORBIT_MEMBER_NAMES = ("V0", "V1", "V2")
REFERENCE_INDICES = (0, 27, 53, 80)
TEMPORAL_TRANSFORMS = ("identity", "reverse", "monotonic_slow_fast")
TEMPORAL_INDEX_MAPS = {
    "identity": tuple(range(v3.LATENT_PHASES)),
    "reverse": tuple(reversed(range(v3.LATENT_PHASES))),
    # round_half_up(20 * (i/20)^2), explicitly materialized so a runtime or
    # language rounding change cannot alter the registered warp.
    "monotonic_slow_fast": (
        0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 16, 18, 20
    ),
}
PINNED_TEMPORAL_TRANSFORM_DIGEST = (
    "869efd0a0fe9565c27b591a9e64806b3c0d78de73d24c15347fb9bc4ee9b6c35"
)
DEFAULT_CARRIER_SEED = 20260808


class IdentityOrbitV4Error(RuntimeError):
    """Raised before an ambiguous orbit target, path, or loss is accepted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise IdentityOrbitV4Error(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def temporal_transform_receipt() -> Mapping[str, Any]:
    value = {
        "transforms": {
            name: list(TEMPORAL_INDEX_MAPS[name]) for name in TEMPORAL_TRANSFORMS
        },
        "latent_phases": v3.LATENT_PHASES,
        "reverse_is_bijective": True,
        "monotonic_warp_is_endpoint_preserving": True,
        "monotonic_warp_is_nondecreasing": True,
        "same_transform_applied_to_donor_and_target": True,
        "references_remain_from_untransformed_target_member": True,
    }
    return {**value, "digest": object_sha256(value)}


def _validate_temporal_registry() -> None:
    if set(TEMPORAL_INDEX_MAPS) != set(TEMPORAL_TRANSFORMS):
        raise IdentityOrbitV4Error("temporal transform registry differs")
    for name, indices in TEMPORAL_INDEX_MAPS.items():
        if (
            len(indices) != v3.LATENT_PHASES
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < v3.LATENT_PHASES
                for index in indices
            )
        ):
            raise IdentityOrbitV4Error(f"{name} temporal index map differs")
    warp = TEMPORAL_INDEX_MAPS["monotonic_slow_fast"]
    if warp[0] != 0 or warp[-1] != v3.LATENT_PHASES - 1:
        raise IdentityOrbitV4Error("monotonic warp lost endpoints")
    if any(left > right for left, right in zip(warp, warp[1:])):
        raise IdentityOrbitV4Error("registered monotonic warp is not nondecreasing")
    if temporal_transform_receipt()["digest"] != PINNED_TEMPORAL_TRANSFORM_DIGEST:
        raise IdentityOrbitV4Error("pinned temporal transform digest differs")


_validate_temporal_registry()


def apply_temporal_transform(value: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply one registered latent-phase transform on temporal dimension -3."""

    if transform not in TEMPORAL_INDEX_MAPS:
        raise IdentityOrbitV4Error("unknown temporal transform")
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 3
        or int(value.shape[-3]) != v3.LATENT_PHASES
    ):
        raise IdentityOrbitV4Error("temporal transform requires a 21-phase tensor")
    indices = torch.tensor(
        TEMPORAL_INDEX_MAPS[transform], dtype=torch.int64, device=value.device
    )
    return value.index_select(-3, indices).contiguous()


@dataclass(frozen=True)
class IdentityOrbitMember:
    name: str
    video_latent: torch.Tensor
    image_references: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]

    def __post_init__(self) -> None:
        if self.name not in ORBIT_MEMBER_NAMES:
            raise IdentityOrbitV4Error("orbit member name differs")
        value = self.video_latent
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 5
            or tuple(int(item) for item in value.shape[:3])
            != (1, v3.LATENT_CHANNELS, v3.LATENT_PHASES)
            or value.dtype != torch.float32
            or value.requires_grad
            or not bool(torch.isfinite(value).all().item())
        ):
            raise IdentityOrbitV4Error(
                "orbit video must be graph-free finite [1,16,21,H,W]"
            )
        if (
            not isinstance(self.image_references, tuple)
            or len(self.image_references) != v3.REFERENCE_COUNT
        ):
            raise IdentityOrbitV4Error("orbit member needs exactly four RV2V refs")
        for reference in self.image_references:
            if (
                not isinstance(reference, torch.Tensor)
                or reference.ndim != 5
                or tuple(int(item) for item in reference.shape[:3])
                != (1, v3.LATENT_CHANNELS, 1)
                or tuple(reference.shape[3:]) != tuple(value.shape[3:])
                or reference.dtype != torch.float32
                or reference.device != value.device
                or reference.requires_grad
                or not bool(torch.isfinite(reference).all().item())
            ):
                raise IdentityOrbitV4Error(
                    "orbit reference must be graph-free finite [1,16,1,H,W]"
                )


@dataclass(frozen=True)
class IdentityOrbit:
    members: tuple[IdentityOrbitMember, IdentityOrbitMember, IdentityOrbitMember]
    same_motion_attested: bool
    same_camera_attested: bool
    same_scene_attested: bool
    appearance_only_counterfactual_attested: bool
    independently_encoded_rgb_refs_attested: bool

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or len(self.members) != 3:
            raise IdentityOrbitV4Error("identity orbit must contain exactly V0/V1/V2")
        if tuple(member.name for member in self.members) != ORBIT_MEMBER_NAMES:
            raise IdentityOrbitV4Error("identity orbit order must be V0,V1,V2")
        if not all(
            (
                self.same_motion_attested,
                self.same_camera_attested,
                self.same_scene_attested,
                self.appearance_only_counterfactual_attested,
                self.independently_encoded_rgb_refs_attested,
            )
        ):
            raise IdentityOrbitV4Error("identity orbit scientific attestations are incomplete")
        reference = self.members[0].video_latent
        if any(
            member.video_latent.shape != reference.shape
            or member.video_latent.dtype != reference.dtype
            or member.video_latent.device != reference.device
            for member in self.members[1:]
        ):
            raise IdentityOrbitV4Error("identity orbit geometry/dtype/device differs")
        if any(
            torch.equal(self.members[left].video_latent, self.members[right].video_latent)
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise IdentityOrbitV4Error("appearance orbit members must not be byte-equal")

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "member_names": list(ORBIT_MEMBER_NAMES),
            "member_count": 3,
            "reference_count_per_member": v3.REFERENCE_COUNT,
            "reference_rgb_indices": list(REFERENCE_INDICES),
            "native_rv2v4_reference_contract_digest": (
                v3.native_rv2v4_reference_contract()["digest"]
            ),
            "cartesian_donor_ref_target_pairs": 9,
            "same_motion_attested": self.same_motion_attested,
            "same_camera_attested": self.same_camera_attested,
            "same_scene_attested": self.same_scene_attested,
            "appearance_only_counterfactual_attested": self.appearance_only_counterfactual_attested,
            "independently_encoded_rgb_refs_attested": self.independently_encoded_rgb_refs_attested,
            "wrong_cross_scene_in_training_objective": False,
            "native_pack_schema": v3.SCHEMA_VERSION,
            "native_schedule_digest": v3.native_unipc40_schedule_receipt()["digest"],
            "temporal_transform_digest": temporal_transform_receipt()["digest"],
        }
        return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True, order=True)
class OrbitCellKey:
    donor_index: int
    target_identity_index: int
    transform: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.donor_index, bool)
            or not isinstance(self.donor_index, int)
            or not 0 <= self.donor_index < 3
            or isinstance(self.target_identity_index, bool)
            or not isinstance(self.target_identity_index, int)
            or not 0 <= self.target_identity_index < 3
            or self.transform not in TEMPORAL_TRANSFORMS
        ):
            raise IdentityOrbitV4Error("orbit cell key differs")


@dataclass(frozen=True)
class OrbitCell:
    key: OrbitCellKey
    donor: torch.Tensor
    references: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    target: torch.Tensor


def build_identity_orbit_cells(
    orbit: IdentityOrbit,
    *,
    transforms: Sequence[str] = TEMPORAL_TRANSFORMS,
) -> tuple[OrbitCell, ...]:
    """Return all registered transform x 3 donor x 3 ref/target positives."""

    if not isinstance(orbit, IdentityOrbit):
        raise IdentityOrbitV4Error("cell construction requires IdentityOrbit")
    names = tuple(transforms)
    if not names or len(set(names)) != len(names) or any(
        name not in TEMPORAL_TRANSFORMS for name in names
    ):
        raise IdentityOrbitV4Error("cell transforms must be unique registered names")
    cells = tuple(
        OrbitCell(
            key=OrbitCellKey(donor_index, target_index, transform),
            donor=apply_temporal_transform(
                orbit.members[donor_index].video_latent, transform
            ),
            references=orbit.members[target_index].image_references,
            target=apply_temporal_transform(
                orbit.members[target_index].video_latent, transform
            ),
        )
        for transform in names
        for donor_index in range(3)
        for target_index in range(3)
    )
    if len(cells) != len(names) * 9 or len({cell.key for cell in cells}) != len(cells):
        raise IdentityOrbitV4Error("identity-orbit Cartesian product is incomplete")
    return cells


def carrier_temporal_permutation(
    *, seed: int = DEFAULT_CARRIER_SEED
) -> tuple[int, ...]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise IdentityOrbitV4Error("carrier seed must lie in [0,2^63)")
    material = f"{seed}\0source-rich-carrier\0{v3.LATENT_PHASES}".encode("ascii")
    derived = int.from_bytes(hashlib.sha256(material).digest(), "big")
    values = list(range(v3.LATENT_PHASES))
    random.Random(derived).shuffle(values)
    if values == list(range(v3.LATENT_PHASES)):
        values = values[1:] + values[:1]
    return tuple(values)


@dataclass(frozen=True)
class SourceRichRhoSchedule:
    max_rho: float = 0.35
    onset_sigma: float = 0.15
    saturation_sigma: float = 0.85

    def __post_init__(self) -> None:
        values = (self.max_rho, self.onset_sigma, self.saturation_sigma)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise IdentityOrbitV4Error("rho schedule values must be finite scalars")
        if not 0.0 <= self.max_rho < 1.0:
            raise IdentityOrbitV4Error("max_rho must lie in [0,1)")
        if not 0.0 <= self.onset_sigma < self.saturation_sigma <= 1.0:
            raise IdentityOrbitV4Error("rho onset/saturation interval differs")

    def rho_and_derivative(self, sigma: float) -> tuple[float, float]:
        if not math.isfinite(float(sigma)) or not 0.0 < float(sigma) <= 1.0:
            raise IdentityOrbitV4Error("rho query sigma must lie in (0,1]")
        value = float(sigma)
        if self.max_rho == 0.0 or value <= self.onset_sigma:
            return 0.0, 0.0
        if value >= self.saturation_sigma:
            return float(self.max_rho), 0.0
        width = self.saturation_sigma - self.onset_sigma
        u = (value - self.onset_sigma) / width
        smooth = 3.0 * u * u - 2.0 * u * u * u
        derivative = (6.0 * u - 6.0 * u * u) / width
        return float(self.max_rho * smooth), float(self.max_rho * derivative)

    def receipt(self) -> Mapping[str, Any]:
        # Build through ordered pairs and fail closed on duplicate field names.
        # A Python dict literal would silently retain only the last duplicate.
        fields = (
            ("schema_version", SCHEMA_VERSION),
            ("family", "registered_cubic_smoothstep"),
            ("max_rho_hex", float(self.max_rho).hex()),
            ("onset_sigma_hex", float(self.onset_sigma).hex()),
            ("saturation_sigma_hex", float(self.saturation_sigma).hex()),
            ("training_schedule_equals_inference_schedule", True),
            ("distribution_at_rho_zero", "strict_original_gaussian_values"),
            ("distribution_at_positive_rho", "source_conditioned_non_gaussian"),
        )
        if len({name for name, _ in fields}) != len(fields):
            raise IdentityOrbitV4Error("rho schedule receipt contains a duplicate key")
        value = dict(fields)
        return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class SourceCarrier:
    value: torch.Tensor
    temporal_permutation: tuple[int, ...]
    epsilon_inner_product: torch.Tensor
    carrier_norm: torch.Tensor
    epsilon_norm: torch.Tensor


def build_source_carrier(
    source_latent: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    seed: int = DEFAULT_CARRIER_SEED,
    whitening_epsilon: float = 1.0e-6,
) -> SourceCarrier:
    """Build a whitened Gaussian-orthogonal, norm-matched source carrier."""

    if (
        not isinstance(source_latent, torch.Tensor)
        or not isinstance(epsilon, torch.Tensor)
        or source_latent.shape != epsilon.shape
        or source_latent.ndim != 5
        or tuple(int(item) for item in source_latent.shape[:3])
        != (1, v3.LATENT_CHANNELS, v3.LATENT_PHASES)
        or source_latent.device != epsilon.device
        or source_latent.dtype != epsilon.dtype
        or source_latent.dtype != torch.float32
        or source_latent.requires_grad
        or epsilon.requires_grad
        or not bool(torch.isfinite(source_latent).all().item())
        or not bool(torch.isfinite(epsilon).all().item())
    ):
        raise IdentityOrbitV4Error(
            "carrier source/epsilon must be graph-free same-shape [1,16,21,H,W]"
        )
    if (
        not math.isfinite(float(whitening_epsilon))
        or float(whitening_epsilon) <= 0.0
    ):
        raise IdentityOrbitV4Error("whitening epsilon must be finite and positive")
    permutation = carrier_temporal_permutation(seed=seed)
    perm_index = torch.tensor(permutation, dtype=torch.int64, device=source_latent.device)
    work = source_latent.float().index_select(2, perm_index)
    # Per-channel whitening keeps appearance structure but removes DC/scale
    # shortcuts before the global Gram-Schmidt projection.
    reduce_dims = (2, 3, 4)
    work = work - work.mean(dim=reduce_dims, keepdim=True)
    rms = work.square().mean(dim=reduce_dims, keepdim=True).sqrt()
    if bool((rms <= float(whitening_epsilon)).any().item()):
        raise IdentityOrbitV4Error("source carrier has a degenerate whitened channel")
    work = work / rms.clamp_min(float(whitening_epsilon))

    carrier_flat = work.reshape(1, -1)
    carrier_flat = carrier_flat - carrier_flat.mean(dim=1, keepdim=True)
    epsilon_flat = epsilon.float().reshape(1, -1)
    epsilon_centered = epsilon_flat - epsilon_flat.mean(dim=1, keepdim=True)
    epsilon_centered_energy = epsilon_centered.square().sum(dim=1, keepdim=True)
    if bool((epsilon_centered_energy <= float(whitening_epsilon)).any().item()):
        raise IdentityOrbitV4Error("Gaussian base is degenerate")
    projection = (
        (carrier_flat * epsilon_centered).sum(dim=1, keepdim=True)
        / epsilon_centered_energy
    )
    carrier_flat = carrier_flat - projection * epsilon_centered
    carrier_flat = carrier_flat - carrier_flat.mean(dim=1, keepdim=True)
    carrier_norm = carrier_flat.square().sum(dim=1, keepdim=True).sqrt()
    epsilon_norm = epsilon_flat.square().sum(dim=1, keepdim=True).sqrt()
    if bool((carrier_norm <= float(whitening_epsilon)).any().item()) or bool(
        (epsilon_norm <= float(whitening_epsilon)).any().item()
    ):
        raise IdentityOrbitV4Error("orthogonal carrier or Gaussian norm is degenerate")
    carrier_flat = carrier_flat * (epsilon_norm / carrier_norm)
    value = carrier_flat.reshape_as(source_latent).to(source_latent.dtype).contiguous()
    # Recompute in FP64 for an auditable numerical residual after the cast.
    value64 = value.double().reshape(1, -1)
    epsilon64 = epsilon.double().reshape(1, -1)
    inner = (value64 * epsilon64).sum(dim=1)
    value_norm = value64.square().sum(dim=1).sqrt()
    epsilon_value_norm = epsilon64.square().sum(dim=1).sqrt()
    relative_inner = inner.abs() / (value_norm * epsilon_value_norm).clamp_min(1.0e-30)
    relative_norm_error = (value_norm - epsilon_value_norm).abs() / epsilon_value_norm.clamp_min(1.0e-30)
    # FP16/BF16 can lose more orthogonality; the actual training path is FP32.
    tolerance = 5.0e-5 if source_latent.dtype == torch.float32 else 5.0e-3
    if bool((relative_inner > tolerance).any().item()) or bool(
        (relative_norm_error > tolerance).any().item()
    ):
        raise IdentityOrbitV4Error("source carrier orthogonal/norm-match audit failed")
    return SourceCarrier(value, permutation, inner, value_norm, epsilon_value_norm)


def _mix_source_rich_noise(
    epsilon: torch.Tensor,
    carrier: torch.Tensor,
    *,
    rho: float,
    rho_derivative: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared train/inference implementation of z_sigma and dz/dsigma."""

    if rho == 0.0:
        # Preserve the Gaussian values exactly; even a nominal multiply by
        # one is avoided so this arm is a strict distributional control.
        return epsilon, torch.zeros_like(epsilon)
    scale = math.sqrt(1.0 - rho * rho)
    noise = scale * epsilon + rho * carrier
    if rho_derivative == 0.0:
        derivative = torch.zeros_like(epsilon)
    else:
        scale_derivative = -(rho * rho_derivative) / scale
        derivative = scale_derivative * epsilon + rho_derivative * carrier
    return noise.contiguous(), derivative.contiguous()


@dataclass(frozen=True)
class InferenceSourceRichNoise:
    schedule_index: int
    timestep: float
    sigma: float
    rho: float
    rho_derivative: float
    value: torch.Tensor
    derivative: torch.Tensor
    carrier: SourceCarrier
    receipt: Mapping[str, Any]


def build_inference_source_rich_noise(
    source_latent: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    schedule_index: int = 0,
    rho_schedule: SourceRichRhoSchedule = SourceRichRhoSchedule(),
    carrier_seed: int = DEFAULT_CARRIER_SEED,
) -> InferenceSourceRichNoise:
    """Build the inference base with the same registered train-time formula.

    Native sampling initializes from schedule index zero.  Other indices are
    exposed for train/inference contract tests and continuous-path audits; the
    carrier is not manually re-injected into the solver state at every step.
    The learned vector field uses the registered rho curve and its derivative.
    """

    if (
        isinstance(schedule_index, bool)
        or not isinstance(schedule_index, int)
        or not 0 <= schedule_index < 40
    ):
        raise IdentityOrbitV4Error("inference source-rich index must lie in [0,40)")
    if not isinstance(rho_schedule, SourceRichRhoSchedule):
        raise IdentityOrbitV4Error("inference source-rich schedule differs")
    carrier = build_source_carrier(source_latent, epsilon, seed=carrier_seed)
    sigma = float(v3.NATIVE_UNIPC40_SIGMAS[schedule_index])
    rho, rho_derivative = rho_schedule.rho_and_derivative(sigma)
    value, derivative = _mix_source_rich_noise(
        epsilon.float(),
        carrier.value.float(),
        rho=rho,
        rho_derivative=rho_derivative,
    )
    strict_gaussian = rho != 0.0 or torch.equal(value, epsilon.float())
    if not strict_gaussian:
        raise IdentityOrbitV4Error("rho0 inference base changed Gaussian values")
    receipt_value = {
        "schema_version": SCHEMA_VERSION,
        "role": "inference_source_rich_initial_noise"
        if schedule_index == 0
        else "inference_contract_audit_coordinate",
        "native_schedule_digest": v3.native_unipc40_schedule_receipt()["digest"],
        "schedule_index": schedule_index,
        "timestep": v3.NATIVE_UNIPC40_TIMESTEPS[schedule_index],
        "sigma_hex": sigma.hex(),
        "rho_hex": rho.hex(),
        "rho_derivative_hex": rho_derivative.hex(),
        "rho_schedule_digest": rho_schedule.receipt()["digest"],
        "carrier_seed": carrier_seed,
        "carrier_permutation": list(carrier.temporal_permutation),
        "rho_zero_strict_gaussian_verified": strict_gaussian,
        "rho_positive_non_gaussian_declared": rho > 0.0,
        "manual_per_step_reinjection": False,
    }
    receipt = {**receipt_value, "digest": object_sha256(receipt_value)}
    return InferenceSourceRichNoise(
        schedule_index,
        float(v3.NATIVE_UNIPC40_TIMESTEPS[schedule_index]),
        sigma,
        rho,
        rho_derivative,
        value,
        derivative,
        carrier,
        receipt,
    )


@dataclass(frozen=True)
class SourceRichStates:
    indices: tuple[int, ...]
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    rhos: torch.Tensor
    rho_derivatives: torch.Tensor
    noise_base: torch.Tensor
    noise_derivative: torch.Tensor
    noisy: torch.Tensor
    target_velocity: torch.Tensor
    clean_target: torch.Tensor
    weights: torch.Tensor
    carrier: SourceCarrier
    rho_schedule_receipt: Mapping[str, Any]
    rho_zero_strict_gaussian_verified: bool

    def receipt(self) -> Mapping[str, Any]:
        rho_values = [float(value) for value in self.rhos.detach().cpu().tolist()]
        value = {
            "schema_version": SCHEMA_VERSION,
            "native_schedule_digest": v3.native_unipc40_schedule_receipt()["digest"],
            "indices": list(self.indices),
            "timesteps": [int(value) for value in self.timesteps.detach().cpu().tolist()],
            "sigma_hex": [float(value).hex() for value in self.sigmas.detach().cpu().tolist()],
            "rho_hex": [value.hex() for value in rho_values],
            "rho_schedule_digest": self.rho_schedule_receipt["digest"],
            "carrier_temporal_permutation": list(self.carrier.temporal_permutation),
            "carrier_temporal_permutation_digest": object_sha256(
                list(self.carrier.temporal_permutation)
            ),
            "path": "x_sigma=(1-sigma)*target+sigma*z_sigma",
            "noise_base": "z_sigma=sqrt(1-rho^2)*epsilon+rho*carrier",
            "target_velocity": "dx/dsigma=z_sigma-target+sigma*dz_sigma/dsigma",
            "rho_zero_is_strict_gaussian": self.rho_zero_strict_gaussian_verified,
            "rho_positive_is_non_gaussian": any(rho > 0.0 for rho in rho_values),
            "training_inference_schedule_identical": True,
        }
        return {**value, "digest": object_sha256(value)}


def build_source_rich_states(
    clean_target: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    carrier_source: torch.Tensor,
    indices: Sequence[int],
    rho_schedule: SourceRichRhoSchedule = SourceRichRhoSchedule(),
    carrier_seed: int = DEFAULT_CARRIER_SEED,
) -> SourceRichStates:
    """Build exact flow states/velocities for the registered rho(sigma) path."""

    if (
        not isinstance(clean_target, torch.Tensor)
        or clean_target.shape != epsilon.shape
        or clean_target.shape != carrier_source.shape
        or clean_target.ndim != 5
        or tuple(int(item) for item in clean_target.shape[:3])
        != (1, v3.LATENT_CHANNELS, v3.LATENT_PHASES)
        or clean_target.dtype != epsilon.dtype
        or clean_target.dtype != carrier_source.dtype
        or clean_target.dtype != torch.float32
        or clean_target.device != epsilon.device
        or clean_target.device != carrier_source.device
        or clean_target.requires_grad
        or not bool(torch.isfinite(clean_target).all().item())
    ):
        raise IdentityOrbitV4Error("source-rich clean/carrier/epsilon contract differs")
    if not isinstance(rho_schedule, SourceRichRhoSchedule):
        raise IdentityOrbitV4Error("source-rich path requires SourceRichRhoSchedule")
    exact_indices = tuple(indices)
    if (
        not exact_indices
        or len(set(exact_indices)) != len(exact_indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 40
            for index in exact_indices
        )
    ):
        raise IdentityOrbitV4Error("source-rich indices must be unique exact40 indices")
    carrier = build_source_carrier(carrier_source, epsilon, seed=carrier_seed)
    work_target = clean_target.float()
    work_epsilon = epsilon.float()
    work_carrier = carrier.value.float()
    sigmas = torch.tensor(
        [v3.NATIVE_UNIPC40_SIGMAS[index] for index in exact_indices],
        dtype=torch.float64,
        device=clean_target.device,
    )
    timesteps = torch.tensor(
        [v3.NATIVE_UNIPC40_TIMESTEPS[index] for index in exact_indices],
        dtype=torch.float32,
        device=clean_target.device,
    )
    rho_pairs = [rho_schedule.rho_and_derivative(float(sigma)) for sigma in sigmas]
    rhos = torch.tensor(
        [pair[0] for pair in rho_pairs], dtype=torch.float64, device=clean_target.device
    )
    rho_derivatives = torch.tensor(
        [pair[1] for pair in rho_pairs], dtype=torch.float64, device=clean_target.device
    )
    noise_bases: list[torch.Tensor] = []
    noise_derivatives: list[torch.Tensor] = []
    for rho, derivative in rho_pairs:
        noise_base, noise_derivative = _mix_source_rich_noise(
            work_epsilon,
            work_carrier,
            rho=rho,
            rho_derivative=derivative,
        )
        noise_bases.append(noise_base)
        noise_derivatives.append(noise_derivative)
    noise_base = torch.stack(noise_bases, dim=0)
    noise_derivative = torch.stack(noise_derivatives, dim=0)
    sigma_shape = (len(exact_indices),) + (1,) * work_target.ndim
    sigma32 = sigmas.float().reshape(sigma_shape)
    target_s = work_target.unsqueeze(0)
    noisy = ((1.0 - sigma32) * target_s + sigma32 * noise_base).contiguous()
    target_velocity = (
        noise_base - target_s + sigma32 * noise_derivative
    ).contiguous()
    for position, rho in enumerate(rhos.tolist()):
        if rho == 0.0 and not torch.equal(noise_base[position], work_epsilon):
            raise IdentityOrbitV4Error("rho0 source-rich base changed Gaussian values")
    weights = torch.full(
        (len(exact_indices),),
        1.0 / float(len(exact_indices)),
        dtype=torch.float64,
        device=clean_target.device,
    )
    return SourceRichStates(
        exact_indices,
        sigmas,
        timesteps,
        rhos,
        rho_derivatives,
        noise_base,
        noise_derivative,
        noisy,
        target_velocity,
        work_target,
        weights,
        carrier,
        rho_schedule.receipt(),
        True,
    )


def build_orbit_cell_states(
    cell: OrbitCell,
    epsilon: torch.Tensor,
    *,
    indices: Sequence[int],
    rho_schedule: SourceRichRhoSchedule = SourceRichRhoSchedule(),
    carrier_seed: int = DEFAULT_CARRIER_SEED,
) -> SourceRichStates:
    """Bind one orbit positive to its exact target-derived carrier and path."""

    if not isinstance(cell, OrbitCell):
        raise IdentityOrbitV4Error("orbit state construction requires OrbitCell")
    return build_source_rich_states(
        cell.target,
        epsilon,
        carrier_source=cell.target,
        indices=indices,
        rho_schedule=rho_schedule,
        carrier_seed=carrier_seed,
    )


def pack_orbit_cell_at_sigma(
    transformer: Any,
    cell: OrbitCell,
    states: SourceRichStates,
    *,
    sigma_position: int,
) -> v3.NativeRV2VPack:
    """Pack one positive orbit cell through v3's exact native RV2V path."""

    if not isinstance(cell, OrbitCell) or not isinstance(states, SourceRichStates):
        raise IdentityOrbitV4Error("native orbit pack requires one cell and states")
    if (
        isinstance(sigma_position, bool)
        or not isinstance(sigma_position, int)
        or not 0 <= sigma_position < len(states.indices)
    ):
        raise IdentityOrbitV4Error("sigma position lies outside source-rich states")
    if not torch.equal(cell.target.float(), states.clean_target):
        raise IdentityOrbitV4Error("source-rich states bind a different exact orbit target")
    return v3.build_native_rv2v_pack(
        transformer,
        donor_video=cell.donor,
        image_references=cell.references,
        noisy_target=states.noisy[sigma_position].to(cell.target.dtype),
    )


@dataclass(frozen=True)
class OrbitObjectiveResult:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    donor_invariance_loss: torch.Tensor
    reference_selection_loss: torch.Tensor
    motion_equivariance_loss: torch.Tensor
    clean_prediction_by_cell: Mapping[OrbitCellKey, torch.Tensor]
    error_by_cell: Mapping[OrbitCellKey, torch.Tensor]


def _weighted_mse(
    left: torch.Tensor, right: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if left.shape != right.shape or int(left.shape[0]) != int(weights.numel()):
        raise IdentityOrbitV4Error("weighted MSE geometry differs")
    reduce_dims = tuple(range(1, left.ndim))
    per_sigma = (left.float() - right.float()).square().mean(dim=reduce_dims)
    normalized = weights.float() / weights.float().sum()
    return (per_sigma * normalized).sum()


def identity_orbit_objective(
    predictions: Mapping[OrbitCellKey, torch.Tensor],
    supervision: Mapping[OrbitCellKey, SourceRichStates],
    orbit: IdentityOrbit,
    *,
    reconstruction_weight: float = 1.0,
    donor_invariance_weight: float = 0.25,
    reference_selection_weight: float = 0.25,
    motion_equivariance_weight: float = 0.25,
    reference_margin: float = 0.02,
) -> OrbitObjectiveResult:
    """Train only exact-target orbit positives plus defined equivariances."""

    if not isinstance(orbit, IdentityOrbit) or set(predictions) != set(supervision):
        raise IdentityOrbitV4Error("prediction/supervision orbit membership differs")
    required = {
        OrbitCellKey(m, n, transform)
        for transform in TEMPORAL_TRANSFORMS
        for m in range(3)
        for n in range(3)
    }
    if set(predictions) != required:
        raise IdentityOrbitV4Error("objective requires the full registered 27-cell orbit")
    scalars = (
        reconstruction_weight,
        donor_invariance_weight,
        reference_selection_weight,
        motion_equivariance_weight,
        reference_margin,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in scalars
    ):
        raise IdentityOrbitV4Error("orbit objective weights/margin must be positive")

    clean_predictions: dict[OrbitCellKey, torch.Tensor] = {}
    errors: dict[OrbitCellKey, torch.Tensor] = {}
    canonical_indices: Optional[tuple[int, ...]] = None
    canonical_rho_schedule_digest: Optional[str] = None
    canonical_carrier_permutation: Optional[tuple[int, ...]] = None
    canonical_weights: Optional[torch.Tensor] = None
    for key in sorted(required):
        prediction = predictions[key]
        states = supervision[key]
        if (
            not isinstance(states, SourceRichStates)
            or not isinstance(prediction, torch.Tensor)
            or prediction.shape != states.target_velocity.shape
            or not prediction.requires_grad
        ):
            raise IdentityOrbitV4Error(
                "every orbit prediction must be graph-connected with exact velocity geometry"
            )
        expected_target = apply_temporal_transform(
            orbit.members[key.target_identity_index].video_latent.float(), key.transform
        )
        if not torch.equal(states.clean_target, expected_target):
            raise IdentityOrbitV4Error("orbit supervision binds a non-exact target")
        if canonical_indices is None:
            canonical_indices = states.indices
            canonical_rho_schedule_digest = str(states.rho_schedule_receipt.get("digest"))
            canonical_carrier_permutation = states.carrier.temporal_permutation
            canonical_weights = states.weights
        elif states.indices != canonical_indices:
            raise IdentityOrbitV4Error(
                "all orbit cells must share the same exact40 sigma coordinates"
            )
        elif (
            states.rho_schedule_receipt.get("digest") != canonical_rho_schedule_digest
            or states.carrier.temporal_permutation != canonical_carrier_permutation
            or canonical_weights is None
            or not torch.equal(states.weights, canonical_weights)
        ):
            raise IdentityOrbitV4Error(
                "orbit cells must share rho schedule, carrier permutation, and sigma weights"
            )
        errors[key] = _weighted_mse(
            prediction, states.target_velocity, states.weights
        )
        sigma_shape = (len(states.indices),) + (1,) * states.clean_target.ndim
        sigma = states.sigmas.float().reshape(sigma_shape)
        clean_predictions[key] = (
            states.noise_base
            + sigma * states.noise_derivative
            - prediction.float()
        )

    reconstruction = torch.stack(tuple(errors.values())).mean()

    invariance_terms: list[torch.Tensor] = []
    for transform in TEMPORAL_TRANSFORMS:
        for target_index in range(3):
            group = torch.stack(
                [
                    clean_predictions[OrbitCellKey(donor, target_index, transform)]
                    for donor in range(3)
                ],
                dim=0,
            )
            mean = group.mean(dim=0)
            states = supervision[OrbitCellKey(0, target_index, transform)]
            invariance_terms.extend(
                _weighted_mse(group[index], mean, states.weights)
                for index in range(3)
            )
    donor_invariance = torch.stack(invariance_terms).mean()

    selection_terms: list[torch.Tensor] = []
    for key in sorted(required):
        predicted_clean = clean_predictions[key]
        states = supervision[key]
        correct_target = apply_temporal_transform(
            orbit.members[key.target_identity_index].video_latent.float(), key.transform
        ).unsqueeze(0).expand_as(predicted_clean)
        correct_distance = _weighted_mse(
            predicted_clean, correct_target, states.weights
        )
        for alternative in range(3):
            if alternative == key.target_identity_index:
                continue
            alternative_target = apply_temporal_transform(
                orbit.members[alternative].video_latent.float(), key.transform
            ).unsqueeze(0).expand_as(predicted_clean)
            alternative_distance = _weighted_mse(
                predicted_clean, alternative_target, states.weights
            )
            selection_terms.append(
                torch.relu(
                    correct_distance.new_tensor(float(reference_margin))
                    + correct_distance
                    - alternative_distance
                )
            )
    reference_selection = torch.stack(selection_terms).mean()

    equivariance_terms: list[torch.Tensor] = []
    for transform in TEMPORAL_TRANSFORMS:
        if transform == "identity":
            continue
        for donor in range(3):
            for target_index in range(3):
                transformed_key = OrbitCellKey(donor, target_index, transform)
                identity_key = OrbitCellKey(donor, target_index, "identity")
                expected = apply_temporal_transform(
                    clean_predictions[identity_key], transform
                )
                equivariance_terms.append(
                    _weighted_mse(
                        clean_predictions[transformed_key],
                        expected,
                        supervision[transformed_key].weights,
                    )
                )
    motion_equivariance = torch.stack(equivariance_terms).mean()
    loss = (
        float(reconstruction_weight) * reconstruction
        + float(donor_invariance_weight) * donor_invariance
        + float(reference_selection_weight) * reference_selection
        + float(motion_equivariance_weight) * motion_equivariance
    )
    if not loss.requires_grad or not bool(torch.isfinite(loss).item()):
        raise IdentityOrbitV4Error("orbit objective is detached or non-finite")
    return OrbitObjectiveResult(
        loss,
        reconstruction,
        donor_invariance,
        reference_selection,
        motion_equivariance,
        clean_predictions,
        errors,
    )


MICROBATCH_STEP_TYPES = ("A", "B", "C")
MICROBATCH_FACTOR_TERMS = {
    "A": "donor_identity_invariance",
    "B": "reference_identity_selection",
    "C": "motion_equivariance",
}


def orbit_cell_key_id(key: OrbitCellKey) -> str:
    if not isinstance(key, OrbitCellKey):
        raise IdentityOrbitV4Error("orbit key id requires OrbitCellKey")
    return f"m{key.donor_index}-n{key.target_identity_index}-{key.transform}"


def _reconstruction_occurrence_count(key: OrbitCellKey) -> int:
    # Every cell appears once in A and once in B.  C adds one occurrence for a
    # non-identity transform and two for identity (paired with both non-id g).
    return 4 if key.transform == "identity" else 3


@dataclass(frozen=True)
class OrbitMicrobatch:
    ordinal: int
    step_type: str
    keys: tuple[OrbitCellKey, ...]
    factor_term: str
    reconstruction_cell_weights: tuple[float, ...]
    factor_cycle_weight: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or not 0 <= self.ordinal < 36
            or self.step_type not in MICROBATCH_STEP_TYPES
            or self.factor_term != MICROBATCH_FACTOR_TERMS[self.step_type]
            or not isinstance(self.keys, tuple)
            or len(set(self.keys)) != len(self.keys)
            or len(self.keys) != (2 if self.step_type == "C" else 3)
            or not isinstance(self.reconstruction_cell_weights, tuple)
            or len(self.reconstruction_cell_weights) != len(self.keys)
        ):
            raise IdentityOrbitV4Error("registered orbit microbatch structure differs")
        expected_weights = tuple(
            1.0 / float(_reconstruction_occurrence_count(key)) for key in self.keys
        )
        if self.reconstruction_cell_weights != expected_weights:
            raise IdentityOrbitV4Error("microbatch reconstruction de-duplication weights differ")
        expected_factor_weight = 1.0 / (18.0 if self.step_type == "C" else 9.0)
        if self.factor_cycle_weight != expected_factor_weight:
            raise IdentityOrbitV4Error("microbatch factor cycle weight differs")

        if self.step_type == "A":
            target_indices = {key.target_identity_index for key in self.keys}
            transforms = {key.transform for key in self.keys}
            donors = {key.donor_index for key in self.keys}
            if len(target_indices) != 1 or len(transforms) != 1 or donors != {0, 1, 2}:
                raise IdentityOrbitV4Error("A step must fix (n,g) and span m=0,1,2")
        elif self.step_type == "B":
            donors = {key.donor_index for key in self.keys}
            transforms = {key.transform for key in self.keys}
            targets = {key.target_identity_index for key in self.keys}
            if len(donors) != 1 or len(transforms) != 1 or targets != {0, 1, 2}:
                raise IdentityOrbitV4Error("B step must fix (m,g) and span n=0,1,2")
        else:
            identity, transformed = self.keys
            if (
                identity.transform != "identity"
                or transformed.transform == "identity"
                or identity.donor_index != transformed.donor_index
                or identity.target_identity_index != transformed.target_identity_index
            ):
                raise IdentityOrbitV4Error(
                    "C step must pair identity with one fixed nonidentity transform"
                )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "ordinal": self.ordinal,
            "step_type": self.step_type,
            "keys": [orbit_cell_key_id(key) for key in self.keys],
            "factor_term": self.factor_term,
            "reconstruction_cell_weight_hex": [
                float(value).hex() for value in self.reconstruction_cell_weights
            ],
            "factor_cycle_weight_hex": float(self.factor_cycle_weight).hex(),
            "gradient_checkpointing_enabled": False,
            "adapter_route_lifetime": "forward_only_no_backward_recompute",
        }
        return {**value, "digest": object_sha256(value)}


def registered_orbit_microbatch_cycle() -> tuple[OrbitMicrobatch, ...]:
    """Return the fixed 36-step A,C,B,C cycle; no data-dependent choice exists."""

    steps: list[OrbitMicrobatch] = []
    ordinal = 0
    nonidentity = ("reverse", "monotonic_slow_fast")
    for group_index in range(9):
        transform = TEMPORAL_TRANSFORMS[group_index // 3]
        varying_index = group_index % 3
        pair_m = group_index // 3
        pair_n = group_index % 3

        a_keys = tuple(
            OrbitCellKey(donor, varying_index, transform) for donor in range(3)
        )
        steps.append(
            OrbitMicrobatch(
                ordinal,
                "A",
                a_keys,
                MICROBATCH_FACTOR_TERMS["A"],
                tuple(1.0 / _reconstruction_occurrence_count(key) for key in a_keys),
                1.0 / 9.0,
            )
        )
        ordinal += 1

        reverse_keys = (
            OrbitCellKey(pair_m, pair_n, "identity"),
            OrbitCellKey(pair_m, pair_n, nonidentity[0]),
        )
        steps.append(
            OrbitMicrobatch(
                ordinal,
                "C",
                reverse_keys,
                MICROBATCH_FACTOR_TERMS["C"],
                tuple(
                    1.0 / _reconstruction_occurrence_count(key)
                    for key in reverse_keys
                ),
                1.0 / 18.0,
            )
        )
        ordinal += 1

        b_keys = tuple(
            OrbitCellKey(varying_index, target, transform) for target in range(3)
        )
        steps.append(
            OrbitMicrobatch(
                ordinal,
                "B",
                b_keys,
                MICROBATCH_FACTOR_TERMS["B"],
                tuple(1.0 / _reconstruction_occurrence_count(key) for key in b_keys),
                1.0 / 9.0,
            )
        )
        ordinal += 1

        warp_keys = (
            OrbitCellKey(pair_m, pair_n, "identity"),
            OrbitCellKey(pair_m, pair_n, nonidentity[1]),
        )
        steps.append(
            OrbitMicrobatch(
                ordinal,
                "C",
                warp_keys,
                MICROBATCH_FACTOR_TERMS["C"],
                tuple(
                    1.0 / _reconstruction_occurrence_count(key)
                    for key in warp_keys
                ),
                1.0 / 18.0,
            )
        )
        ordinal += 1

    cycle = tuple(steps)
    if len(cycle) != 36 or tuple(step.ordinal for step in cycle) != tuple(range(36)):
        raise IdentityOrbitV4Error("registered orbit microbatch cycle is incomplete")
    if tuple(step.step_type for step in cycle) != ("A", "C", "B", "C") * 9:
        raise IdentityOrbitV4Error("registered orbit step alternation differs")
    return cycle


def orbit_microbatch_cycle_receipt() -> Mapping[str, Any]:
    cycle = registered_orbit_microbatch_cycle()
    required = {
        OrbitCellKey(m, n, transform)
        for transform in TEMPORAL_TRANSFORMS
        for m in range(3)
        for n in range(3)
    }
    raw_counts = {key: 0 for key in required}
    weighted_counts = {key: 0.0 for key in required}
    a_groups: set[tuple[int, str]] = set()
    b_groups: set[tuple[int, str]] = set()
    c_pairs: set[tuple[int, int, str]] = set()
    for step in cycle:
        for key, weight in zip(step.keys, step.reconstruction_cell_weights):
            raw_counts[key] += 1
            weighted_counts[key] += weight
        if step.step_type == "A":
            a_groups.add((step.keys[0].target_identity_index, step.keys[0].transform))
        elif step.step_type == "B":
            b_groups.add((step.keys[0].donor_index, step.keys[0].transform))
        else:
            c_pairs.add(
                (
                    step.keys[0].donor_index,
                    step.keys[0].target_identity_index,
                    step.keys[1].transform,
                )
            )
    if set(raw_counts) != required or any(
        count != _reconstruction_occurrence_count(key)
        for key, count in raw_counts.items()
    ):
        raise IdentityOrbitV4Error("raw reconstruction cycle coverage differs")
    if any(abs(value - 1.0) > 1.0e-12 for value in weighted_counts.values()):
        raise IdentityOrbitV4Error("weighted reconstruction cycle coverage differs")
    if len(a_groups) != 9 or len(b_groups) != 9 or len(c_pairs) != 18:
        raise IdentityOrbitV4Error("factor-term cycle coverage differs")
    value = {
        "schema_version": SCHEMA_VERSION,
        "cycle_name": "registered_identity_orbit_A_C_B_C_v1",
        "step_count": len(cycle),
        "step_pattern": [step.step_type for step in cycle],
        "steps": [step.receipt() for step in cycle],
        "reconstruction_unique_cell_count": len(required),
        "raw_reconstruction_counts": {
            orbit_cell_key_id(key): raw_counts[key] for key in sorted(required)
        },
        "weighted_reconstruction_coverage_hex": {
            orbit_cell_key_id(key): float(weighted_counts[key]).hex()
            for key in sorted(required)
        },
        "weighted_reconstruction_each_cell_exactly_once": True,
        "donor_invariance_group_count": len(a_groups),
        "reference_selection_group_count": len(b_groups),
        "equivariance_pair_count": len(c_pairs),
        "dynamic_cell_selection_allowed": False,
        "gradient_checkpointing_enabled": False,
        "trainer_must_force_gradient_checkpointing_disabled": True,
        "adapter_route_lifetime": "forward_only_no_backward_recompute",
        "route_required_during_backward_recompute": False,
        "native_pack_schema": v3.SCHEMA_VERSION,
        "native_schedule_digest": v3.native_unipc40_schedule_receipt()["digest"],
    }
    return {**value, "digest": object_sha256(value)}


def validate_microbatch_runtime(*, gradient_checkpointing_enabled: bool) -> Mapping[str, Any]:
    if not isinstance(gradient_checkpointing_enabled, bool):
        raise IdentityOrbitV4Error("gradient checkpointing flag must be boolean")
    if gradient_checkpointing_enabled:
        raise IdentityOrbitV4Error(
            "v4 microbatch trainer must disable gradient checkpointing; route is forward-only"
        )
    value = {
        "gradient_checkpointing_enabled": False,
        "adapter_route_lifetime": "forward_only_no_backward_recompute",
        "registered_cycle_digest": orbit_microbatch_cycle_receipt()["digest"],
        "accepted": True,
    }
    return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class OrbitMicrobatchObjectiveResult:
    loss: torch.Tensor
    reconstruction_cycle_contribution: torch.Tensor
    factor_cycle_contribution: torch.Tensor
    raw_factor_value: torch.Tensor
    error_by_cell: Mapping[OrbitCellKey, torch.Tensor]
    clean_prediction_by_cell: Mapping[OrbitCellKey, torch.Tensor]


def identity_orbit_microbatch_objective(
    microbatch: OrbitMicrobatch,
    predictions: Mapping[OrbitCellKey, torch.Tensor],
    supervision: Mapping[OrbitCellKey, SourceRichStates],
    orbit: IdentityOrbit,
    *,
    reconstruction_weight: float = 1.0,
    donor_invariance_weight: float = 0.25,
    reference_selection_weight: float = 0.25,
    motion_equivariance_weight: float = 0.25,
    reference_margin: float = 0.02,
) -> OrbitMicrobatchObjectiveResult:
    """Compute one preregistered cycle contribution without a 27-cell graph."""

    if (
        not isinstance(microbatch, OrbitMicrobatch)
        or not isinstance(orbit, IdentityOrbit)
        or set(predictions) != set(microbatch.keys)
        or set(supervision) != set(microbatch.keys)
    ):
        raise IdentityOrbitV4Error("local objective membership differs from microbatch")
    scalars = (
        reconstruction_weight,
        donor_invariance_weight,
        reference_selection_weight,
        motion_equivariance_weight,
        reference_margin,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in scalars
    ):
        raise IdentityOrbitV4Error("local objective weights/margin must be positive")

    errors: dict[OrbitCellKey, torch.Tensor] = {}
    clean_predictions: dict[OrbitCellKey, torch.Tensor] = {}
    canonical_indices: Optional[tuple[int, ...]] = None
    canonical_schedule: Optional[str] = None
    canonical_permutation: Optional[tuple[int, ...]] = None
    canonical_weights: Optional[torch.Tensor] = None
    for key in microbatch.keys:
        states = supervision[key]
        prediction = predictions[key]
        expected_target = apply_temporal_transform(
            orbit.members[key.target_identity_index].video_latent.float(), key.transform
        )
        if (
            not isinstance(states, SourceRichStates)
            or not isinstance(prediction, torch.Tensor)
            or prediction.shape != states.target_velocity.shape
            or not prediction.requires_grad
            or not torch.equal(states.clean_target, expected_target)
        ):
            raise IdentityOrbitV4Error("local orbit prediction/supervision differs")
        if canonical_indices is None:
            canonical_indices = states.indices
            canonical_schedule = str(states.rho_schedule_receipt.get("digest"))
            canonical_permutation = states.carrier.temporal_permutation
            canonical_weights = states.weights
        elif (
            states.indices != canonical_indices
            or states.rho_schedule_receipt.get("digest") != canonical_schedule
            or states.carrier.temporal_permutation != canonical_permutation
            or canonical_weights is None
            or not torch.equal(states.weights, canonical_weights)
        ):
            raise IdentityOrbitV4Error("local microbatch path contracts differ")
        errors[key] = _weighted_mse(
            prediction, states.target_velocity, states.weights
        )
        sigma_shape = (len(states.indices),) + (1,) * states.clean_target.ndim
        sigma = states.sigmas.float().reshape(sigma_shape)
        clean_predictions[key] = (
            states.noise_base + sigma * states.noise_derivative - prediction.float()
        )

    reconstruction_contribution = sum(
        error * float(weight)
        for error, weight in zip(
            (errors[key] for key in microbatch.keys),
            microbatch.reconstruction_cell_weights,
        )
    ) / 27.0

    if microbatch.step_type == "A":
        group = torch.stack(
            [clean_predictions[key] for key in microbatch.keys], dim=0
        )
        mean = group.mean(dim=0)
        weights = supervision[microbatch.keys[0]].weights
        raw_factor = torch.stack(
            [_weighted_mse(group[index], mean, weights) for index in range(3)]
        ).mean()
        factor_weight = float(donor_invariance_weight)
    elif microbatch.step_type == "B":
        selection_terms: list[torch.Tensor] = []
        for key in microbatch.keys:
            predicted_clean = clean_predictions[key]
            weights = supervision[key].weights
            correct_target = apply_temporal_transform(
                orbit.members[key.target_identity_index].video_latent.float(), key.transform
            ).unsqueeze(0).expand_as(predicted_clean)
            correct_distance = _weighted_mse(
                predicted_clean, correct_target, weights
            )
            for alternative in range(3):
                if alternative == key.target_identity_index:
                    continue
                alternative_target = apply_temporal_transform(
                    orbit.members[alternative].video_latent.float(), key.transform
                ).unsqueeze(0).expand_as(predicted_clean)
                alternative_distance = _weighted_mse(
                    predicted_clean, alternative_target, weights
                )
                selection_terms.append(
                    torch.relu(
                        correct_distance.new_tensor(float(reference_margin))
                        + correct_distance
                        - alternative_distance
                    )
                )
        raw_factor = torch.stack(selection_terms).mean()
        factor_weight = float(reference_selection_weight)
    else:
        identity_key, transformed_key = microbatch.keys
        expected = apply_temporal_transform(
            clean_predictions[identity_key], transformed_key.transform
        )
        raw_factor = _weighted_mse(
            clean_predictions[transformed_key],
            expected,
            supervision[transformed_key].weights,
        )
        factor_weight = float(motion_equivariance_weight)

    factor_contribution = raw_factor * float(microbatch.factor_cycle_weight)
    loss = (
        float(reconstruction_weight) * reconstruction_contribution
        + factor_weight * factor_contribution
    )
    if not loss.requires_grad or not bool(torch.isfinite(loss).item()):
        raise IdentityOrbitV4Error("local orbit objective is detached or non-finite")
    return OrbitMicrobatchObjectiveResult(
        loss,
        reconstruction_contribution,
        factor_contribution,
        raw_factor,
        errors,
        clean_predictions,
    )


def heldout_wrong_scene_gate(
    *,
    correct_prediction_clean: torch.Tensor,
    wrong_scene_prediction_clean: torch.Tensor,
    exact_orbit_target: torch.Tensor,
    maximum_correct_error: float,
    minimum_wrong_scene_sensitivity: float,
) -> Mapping[str, Any]:
    """Evaluate a cross-scene intervention without assigning it a target."""

    if (
        not isinstance(correct_prediction_clean, torch.Tensor)
        or not isinstance(wrong_scene_prediction_clean, torch.Tensor)
        or not isinstance(exact_orbit_target, torch.Tensor)
        or correct_prediction_clean.shape != wrong_scene_prediction_clean.shape
        or correct_prediction_clean.shape != exact_orbit_target.shape
        or any(value.requires_grad for value in (correct_prediction_clean, wrong_scene_prediction_clean))
    ):
        raise IdentityOrbitV4Error("held-out wrong-scene gate requires detached equal geometry")
    for name, value in (
        ("maximum_correct_error", maximum_correct_error),
        ("minimum_wrong_scene_sensitivity", minimum_wrong_scene_sensitivity),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise IdentityOrbitV4Error(f"{name} must be finite and nonnegative")
    correct_error = float(
        (correct_prediction_clean.float() - exact_orbit_target.float())
        .square()
        .mean()
        .item()
    )
    sensitivity = float(
        (wrong_scene_prediction_clean.float() - correct_prediction_clean.float())
        .square()
        .mean()
        .sqrt()
        .item()
    )
    value = {
        "schema_version": SCHEMA_VERSION,
        "split": "heldout_wrong_cross_scene_only",
        "used_by_training_objective": False,
        "wrong_scene_target_defined": False,
        "wrong_scene_error_term_computed": False,
        "correct_orbit_target_error": correct_error,
        "wrong_scene_output_sensitivity": sensitivity,
        "maximum_correct_error": float(maximum_correct_error),
        "minimum_wrong_scene_sensitivity": float(minimum_wrong_scene_sensitivity),
        "correct_target_gate": correct_error <= float(maximum_correct_error),
        "wrong_scene_sensitivity_gate": sensitivity
        >= float(minimum_wrong_scene_sensitivity),
    }
    value["accepted"] = bool(
        value["correct_target_gate"] and value["wrong_scene_sensitivity_gate"]
    )
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "DEFAULT_CARRIER_SEED",
    "IdentityOrbit",
    "IdentityOrbitMember",
    "IdentityOrbitV4Error",
    "InferenceSourceRichNoise",
    "OrbitCell",
    "OrbitCellKey",
    "OrbitMicrobatch",
    "OrbitMicrobatchObjectiveResult",
    "OrbitObjectiveResult",
    "ORBIT_MEMBER_NAMES",
    "PINNED_TEMPORAL_TRANSFORM_DIGEST",
    "REFERENCE_INDICES",
    "SCHEMA_VERSION",
    "SourceCarrier",
    "SourceRichRhoSchedule",
    "SourceRichStates",
    "TEMPORAL_INDEX_MAPS",
    "TEMPORAL_TRANSFORMS",
    "apply_temporal_transform",
    "build_identity_orbit_cells",
    "build_inference_source_rich_noise",
    "build_orbit_cell_states",
    "build_source_carrier",
    "build_source_rich_states",
    "carrier_temporal_permutation",
    "heldout_wrong_scene_gate",
    "identity_orbit_microbatch_objective",
    "identity_orbit_objective",
    "MICROBATCH_FACTOR_TERMS",
    "MICROBATCH_STEP_TYPES",
    "orbit_cell_key_id",
    "orbit_microbatch_cycle_receipt",
    "pack_orbit_cell_at_sigma",
    "registered_orbit_microbatch_cycle",
    "temporal_transform_receipt",
    "validate_microbatch_runtime",
]
