#!/usr/bin/env python3
"""Source-preserving execution core for Self-Predicted Phase-Transport LoRA.

SPT treats the frozen source latent as an immutable token bank.  A 21-phase
plan chooses, per latent cell, among three mutually exclusive paths:

* preserve: copy the source token at the same phase/location;
* transport: differentiably retrieve a source token with a learned 3-D offset;
* generate: use Bernini's ordinary clean prediction for genuinely new content.

The plan is internal and dense; it is not an externally supplied mask, track,
pose, or trajectory.  ``build_oracle_plan`` derives a deliberately limited
teacher from paired VAE latents without an external model.  It is a diagnostic
and distillation target, never an inference input.  Student inference calls
``PhaseTransportAdapter.forward(source, instruction_embedding)`` and therefore
cannot receive a target latent by API construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Mapping, Optional, Sequence


LATENT_PHASES = 21
GATE_PRESERVE = 0
GATE_TRANSPORT = 1
GATE_GENERATE = 2
ORACLE_PROJECTION_SCHEME = "orthonormal_dct4_all_input_channels_v1"


class PhaseTransportError(RuntimeError):
    """Raised when an SPT shape or inference invariant is violated."""


@dataclass(frozen=True)
class PhaseTransportConfig:
    latent_channels: int = 64
    text_channels: int = 4096
    hidden_channels: int = 128
    latent_phases: int = LATENT_PHASES
    max_temporal_offset: float = 2.0
    max_spatial_offset: float = 4.0
    teacher_temporal_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)
    teacher_spatial_offsets: tuple[int, ...] = (-4, -2, 0, 2, 4)
    teacher_temperature: float = 0.08
    teacher_generate_threshold: float = 0.35
    teacher_transport_margin: float = 0.05
    teacher_require_cycle: bool = True
    teacher_projection_scheme: str = ORACLE_PROJECTION_SCHEME
    teacher_allow_lossy_projection: bool = False
    max_generate_fraction_per_phase: Optional[float] = 0.12
    teacher_allow_unbounded_generate_ablation: bool = False
    source_bank_detach: bool = True

    def validate(self) -> None:
        if self.latent_phases != LATENT_PHASES:
            raise PhaseTransportError("SPT-v2 requires exactly 21 latent phases")
        for name in ("latent_channels", "text_channels", "hidden_channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise PhaseTransportError(f"{name} must be a positive integer")
        groups = min(32, self.hidden_channels)
        if self.hidden_channels % groups:
            raise PhaseTransportError(
                "hidden_channels must be divisible by its GroupNorm group count"
            )
        for name in (
            "max_temporal_offset",
            "max_spatial_offset",
            "teacher_temperature",
            "teacher_generate_threshold",
            "teacher_transport_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise PhaseTransportError(f"{name} must be finite and positive")
        if type(self.teacher_require_cycle) is not bool:
            raise PhaseTransportError("teacher_require_cycle must be boolean")
        if type(self.teacher_allow_lossy_projection) is not bool:
            raise PhaseTransportError("teacher_allow_lossy_projection must be boolean")
        if type(self.teacher_allow_unbounded_generate_ablation) is not bool:
            raise PhaseTransportError(
                "teacher_allow_unbounded_generate_ablation must be boolean"
            )
        if self.max_generate_fraction_per_phase is None:
            if not self.teacher_allow_unbounded_generate_ablation:
                raise PhaseTransportError(
                    "unbounded generate is forbidden outside an explicit ablation"
                )
        else:
            budget = float(self.max_generate_fraction_per_phase)
            if not math.isfinite(budget) or not 0.0 < budget <= 1.0:
                raise PhaseTransportError(
                    "max_generate_fraction_per_phase must lie in (0,1]"
                )
        if self.teacher_projection_scheme != ORACLE_PROJECTION_SCHEME:
            raise PhaseTransportError(
                f"teacher_projection_scheme must be {ORACLE_PROJECTION_SCHEME!r}"
            )
        if 0 not in self.teacher_temporal_offsets or 0 not in self.teacher_spatial_offsets:
            raise PhaseTransportError("teacher candidate offsets must contain zero")
        if len(set(self.teacher_temporal_offsets)) != len(self.teacher_temporal_offsets):
            raise PhaseTransportError("teacher temporal offsets must be unique")
        if len(set(self.teacher_spatial_offsets)) != len(self.teacher_spatial_offsets):
            raise PhaseTransportError("teacher spatial offsets must be unique")


@dataclass
class PhasePlan:
    """Dense 21-phase plan in latent-cell units.

    ``offsets`` is ``[B,3,T,H,W]`` ordered as ``(dt, dy, dx)``.
    ``gate_probs`` is ``[B,3,T,H,W]`` ordered as preserve/transport/generate.
    """

    offsets: Any
    gate_probs: Any
    provenance: str
    diagnostics: Optional[Mapping[str, Any]] = None

    def validate(self, source: Any, *, atol: float = 2e-5) -> None:
        import torch

        _validate_video(source, label="source")
        b, t, h, w, _ = map(int, source.shape)
        if tuple(self.offsets.shape) != (b, 3, t, h, w):
            raise PhaseTransportError("plan offsets must be [B,3,T,H,W]")
        if tuple(self.gate_probs.shape) != (b, 3, t, h, w):
            raise PhaseTransportError("plan gates must be [B,3,T,H,W]")
        if not bool(torch.isfinite(self.offsets).all()):
            raise PhaseTransportError("plan contains non-finite offsets")
        if not bool(torch.isfinite(self.gate_probs).all()):
            raise PhaseTransportError("plan contains non-finite gates")
        if bool((self.gate_probs < -atol).any()):
            raise PhaseTransportError("plan gates must be non-negative")
        sums = self.gate_probs.float().sum(dim=1)
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0.0)):
            raise PhaseTransportError("plan gates must sum to one")
        if self.provenance not in ("student", "oracle_pair_proxy"):
            raise PhaseTransportError(f"unknown plan provenance: {self.provenance!r}")


def _validate_video(value: Any, *, label: str) -> None:
    if getattr(value, "ndim", None) != 5:
        raise PhaseTransportError(f"{label} must be [B,T,H,W,D]")
    if int(value.shape[1]) != LATENT_PHASES:
        raise PhaseTransportError(f"{label} must contain exactly 21 latent phases")
    if any(int(size) <= 0 for size in value.shape):
        raise PhaseTransportError(f"{label} has an empty dimension")


def packed_to_video(packed: Any, *, height: int, width: int) -> Any:
    """View Bernini packed ``[B,T*H*W,D]`` tokens as an SPT video."""

    if getattr(packed, "ndim", None) != 3:
        raise PhaseTransportError("packed tokens must be [B,N,D]")
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise PhaseTransportError("packed height/width must be positive integers")
    expected = LATENT_PHASES * height * width
    if int(packed.shape[1]) != expected:
        raise PhaseTransportError(
            f"packed token count {int(packed.shape[1])} differs from {expected}"
        )
    return packed.reshape(int(packed.shape[0]), LATENT_PHASES, height, width, int(packed.shape[2]))


def video_to_packed(video: Any) -> Any:
    _validate_video(video, label="video")
    return video.reshape(int(video.shape[0]), -1, int(video.shape[-1]))


def exact_identity_plan(source: Any, *, provenance: str = "oracle_pair_proxy") -> PhasePlan:
    """Return an exact preserve plan used to calibrate the no-op student."""

    import torch

    _validate_video(source, label="source")
    b, _, h, w, _ = map(int, source.shape)
    offsets = torch.zeros(
        b, 3, LATENT_PHASES, h, w, device=source.device, dtype=torch.float32
    )
    gates = torch.zeros_like(offsets)
    gates[:, GATE_PRESERVE] = 1.0
    plan = PhasePlan(offsets, gates, provenance)
    plan.validate(source)
    return plan


def _identity_grid(source: Any) -> Any:
    import torch

    b, t, h, w, _ = map(int, source.shape)
    z = torch.linspace(-1.0, 1.0, t, device=source.device, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, h, device=source.device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, w, device=source.device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).unsqueeze(0).expand(b, -1, -1, -1, -1)


def transport_source(source: Any, offsets: Any) -> Any:
    """Trilinearly retrieve clean source tokens using offsets in cell units."""

    import torch.nn.functional as F

    _validate_video(source, label="source")
    b, t, h, w, _ = map(int, source.shape)
    if tuple(offsets.shape) != (b, 3, t, h, w):
        raise PhaseTransportError("transport offsets must be [B,3,T,H,W]")
    dt, dy, dx = offsets.float().unbind(dim=1)
    # align_corners=True gives exact integer-cell addressing, including a
    # bitwise-close zero-offset identity used by the oracle diagnostic.
    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    st = 2.0 / max(t - 1, 1)
    delta = __import__("torch").stack((dx * sx, dy * sy, dt * st), dim=-1)
    grid = _identity_grid(source) + delta
    bank = source.permute(0, 4, 1, 2, 3).float()
    transported = F.grid_sample(
        bank,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return transported.permute(0, 2, 3, 4, 1).to(dtype=source.dtype)


def execute_clean_plan(
    source: Any,
    generated_clean: Any,
    plan: PhasePlan,
    *,
    noop: bool = False,
    detach_source_bank: bool = True,
) -> Any:
    """Execute a plan in clean-latent space.

    ``noop=True`` is an explicit semantic bypass and returns the source tensor
    itself.  It does not approximate identity with extreme softmax logits.
    """

    _validate_video(source, label="source")
    _validate_video(generated_clean, label="generated_clean")
    if tuple(source.shape) != tuple(generated_clean.shape):
        raise PhaseTransportError("source/generated clean shapes differ")
    if noop:
        return source
    plan.validate(source)
    bank = source.detach() if detach_source_bank else source
    # The paired teacher chooses discrete candidates.  Execute those candidates
    # with the same integer gather used to define its reconstruction contract;
    # student offsets remain continuous and differentiable through grid_sample.
    transported = (
        _hard_integer_retrieve(bank, plan.offsets)
        if plan.provenance == "oracle_pair_proxy"
        else transport_source(bank, plan.offsets)
    )
    gates = plan.gate_probs.to(dtype=source.dtype).permute(0, 2, 3, 4, 1)
    preserve = gates[..., GATE_PRESERVE : GATE_PRESERVE + 1]
    move = gates[..., GATE_TRANSPORT : GATE_TRANSPORT + 1]
    generate = gates[..., GATE_GENERATE : GATE_GENERATE + 1]
    return preserve * bank + move * transported + generate * generated_clean


def velocity_from_clean(noisy: Any, clean: Any, sigma: Any, *, eps: float = 1e-4) -> Any:
    """Project an executed clean estimate back to Bernini flow velocity."""

    import torch

    if tuple(noisy.shape) != tuple(clean.shape):
        raise PhaseTransportError("noisy/clean shapes differ")
    sigma = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32)
    if not bool(torch.isfinite(sigma).all()) or bool((sigma < eps).any()):
        raise PhaseTransportError(f"sigma must be finite and >= {eps}")
    while sigma.ndim < noisy.ndim:
        sigma = sigma.unsqueeze(-1)
    return (noisy.float() - clean.float()) / sigma


def clean_from_velocity(noisy: Any, velocity: Any, sigma: Any) -> Any:
    import torch

    if tuple(noisy.shape) != tuple(velocity.shape):
        raise PhaseTransportError("noisy/velocity shapes differ")
    sigma = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32)
    while sigma.ndim < noisy.ndim:
        sigma = sigma.unsqueeze(-1)
    return noisy.float() - sigma * velocity.float()


def execute_packed_velocity(
    *,
    source_packed: Any,
    noisy_packed: Any,
    base_velocity_packed: Any,
    sigma: Any,
    height: int,
    width: int,
    plan: PhasePlan,
    noop: bool = False,
    detach_source_bank: bool = True,
) -> Any:
    """Bernini boundary adapter: packed raw velocity in, packed raw velocity out."""

    source = packed_to_video(source_packed, height=height, width=width)
    noisy = packed_to_video(noisy_packed, height=height, width=width)
    base_velocity = packed_to_video(
        base_velocity_packed, height=height, width=width
    )
    generated = clean_from_velocity(noisy, base_velocity, sigma)
    executed = execute_clean_plan(
        source,
        generated,
        plan,
        noop=noop,
        detach_source_bank=detach_source_bank,
    )
    return video_to_packed(velocity_from_clean(noisy, executed, sigma))


def _candidate_grid(config: PhaseTransportConfig) -> list[tuple[int, int, int]]:
    return [
        (dt, dy, dx)
        for dt in config.teacher_temporal_offsets
        for dy in config.teacher_spatial_offsets
        for dx in config.teacher_spatial_offsets
    ]


def fixed_auditable_projection(
    input_channels: int,
    output_channels: int,
    *,
    device: Any = None,
) -> Any:
    """Return a deterministic DCT-IV projection that touches every channel.

    A leading channel slice is not a valid proxy for Bernini's packed VAE
    token: the 64 channels contain all four spatial sub-pixels.  DCT-IV has no
    zero entries for the integer dimensions used here, so every projected
    coordinate uses every packed input channel.  With 64 outputs it is an
    orthonormal change of basis and therefore preserves full-channel L2 cost.

    The matrix is constructed on CPU from the formula below and then copied to
    the requested device.  This makes its byte digest independent of GPU math
    libraries and gives each oracle receipt an auditable feature contract.
    """

    import torch

    if type(input_channels) is not int or input_channels <= 0:
        raise PhaseTransportError("input_channels must be a positive integer")
    if (
        type(output_channels) is not int
        or output_channels <= 0
        or output_channels > input_channels
    ):
        raise PhaseTransportError(
            "projection output_channels must lie in [1,input_channels]"
        )
    scale = math.sqrt(2.0 / float(input_channels))
    values = [
        scale
        * math.cos(
            math.pi
            * (float(row) + 0.5)
            * (float(column) + 0.5)
            / float(input_channels)
        )
        for row in range(output_channels)
        for column in range(input_channels)
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(
        output_channels, input_channels
    ).to(device=device)


def projection_audit_metadata(input_channels: int, output_channels: int) -> dict[str, Any]:
    """Describe and hash the fixed oracle feature projection."""

    matrix = fixed_auditable_projection(input_channels, output_channels, device="cpu")
    # Hash formula-generated float32 values with an explicit little-endian
    # representation rather than relying on numpy/platform serialization.
    payload = b"".join(struct.pack("<f", float(value)) for value in matrix.reshape(-1))
    per_input_nonzero = (matrix != 0).any(dim=0)
    covered = int(per_input_nonzero.sum().item())
    return {
        "scheme": ORACLE_PROJECTION_SCHEME,
        "formula": "sqrt(2/C)*cos(pi*(row+0.5)*(channel+0.5)/C)",
        "input_channels": int(input_channels),
        "output_channels": int(output_channels),
        "covered_input_channels": covered,
        "input_coverage_fraction": float(covered / input_channels),
        "float32_matrix_sha256": hashlib.sha256(payload).hexdigest(),
        "full_l2_preserving": bool(output_channels == input_channels),
    }


def _valid_mask_for_offset(reference: Any, offset: tuple[int, int, int]) -> Any:
    """Return cells whose integer retrieval coordinate stays inside the bank."""

    import torch

    b, t, h, w, _ = map(int, reference.shape)
    dt, dy, dx = offset
    valid_t = (torch.arange(t, device=reference.device) + dt).ge(0) & (
        torch.arange(t, device=reference.device) + dt
    ).lt(t)
    valid_y = (torch.arange(h, device=reference.device) + dy).ge(0) & (
        torch.arange(h, device=reference.device) + dy
    ).lt(h)
    valid_x = (torch.arange(w, device=reference.device) + dx).ge(0) & (
        torch.arange(w, device=reference.device) + dx
    ).lt(w)
    return (
        valid_t[:, None, None]
        & valid_y[None, :, None]
        & valid_x[None, None, :]
    ).unsqueeze(0).expand(b, -1, -1, -1)


def _oracle_cost_volume(
    bank: Any,
    query: Any,
    candidates: Sequence[tuple[int, int, int]],
) -> tuple[Any, Any, Any]:
    """Return raw costs, validity-masked costs, and candidate valid masks."""

    import torch

    raw_costs = []
    valid_masks = []
    with torch.no_grad():
        for dt, dy, dx in candidates:
            offsets = torch.zeros(
                int(bank.shape[0]),
                3,
                LATENT_PHASES,
                int(bank.shape[2]),
                int(bank.shape[3]),
                device=bank.device,
                dtype=torch.float32,
            )
            offsets[:, 0].fill_(float(dt))
            offsets[:, 1].fill_(float(dy))
            offsets[:, 2].fill_(float(dx))
            retrieved = transport_source(bank, offsets).float()
            residual = (retrieved - query).pow(2).mean(dim=-1)
            scale = query.pow(2).mean(dim=-1).clamp_min(0.05)
            raw_costs.append(residual / scale)
            valid_masks.append(_valid_mask_for_offset(bank, (dt, dy, dx)))
    raw = torch.stack(raw_costs, dim=1)
    valid = torch.stack(valid_masks, dim=1)
    masked = raw.masked_fill(~valid, float("inf"))
    return raw, masked, valid


def _cycle_consistency_for_selected(
    *,
    selected_index: Any,
    reverse_best_index: Any,
    candidates: Sequence[tuple[int, int, int]],
    forward_valid: Any,
) -> Any:
    """Check target->source->target closure for hard integer candidates."""

    import torch

    candidate_to_index = {candidate: index for index, candidate in enumerate(candidates)}
    cycle_by_candidate = []
    for index, (dt, dy, dx) in enumerate(candidates):
        inverse = candidate_to_index.get((-dt, -dy, -dx))
        if inverse is None:
            cycle_by_candidate.append(torch.zeros_like(reverse_best_index, dtype=torch.bool))
            continue
        # roll(...,-d)[p] = reverse_best[p+d], exactly the source cell
        # reached by the forward offset.  forward_valid removes wrapped cells.
        reverse_at_source = torch.roll(
            reverse_best_index,
            shifts=(-dt, -dy, -dx),
            dims=(1, 2, 3),
        )
        cycle_by_candidate.append(
            (reverse_at_source == inverse) & forward_valid[:, index]
        )
    cycle = torch.stack(cycle_by_candidate, dim=1)
    return cycle.gather(1, selected_index.unsqueeze(1)).squeeze(1)


def _budget_generate_per_phase(
    generate_candidates: Any,
    unexplained_score: Any,
    max_fraction: Optional[float],
) -> tuple[Any, Any]:
    """Keep only the highest-necessity generate cells independently per phase.

    Both inputs are ``[B,T,H,W]``.  A finite budget uses ``floor(H*W*f)``
    cells, which guarantees the observed fraction never exceeds the declared
    maximum.  The caller must send rejected cells to preserve, never to a
    fabricated transport correspondence.
    """

    import torch

    if (
        getattr(generate_candidates, "ndim", None) != 4
        or tuple(generate_candidates.shape) != tuple(unexplained_score.shape)
    ):
        raise PhaseTransportError(
            "generate candidates and unexplained score must be matching [B,T,H,W]"
        )
    candidates = generate_candidates.bool()
    if max_fraction is None:
        return candidates, torch.zeros_like(candidates)
    fraction = float(max_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise PhaseTransportError("generate budget must lie in (0,1]")
    batch, phases, height, width = map(int, candidates.shape)
    cells = height * width
    keep_count = int(math.floor(cells * fraction + 1e-12))
    if keep_count >= cells:
        return candidates, torch.zeros_like(candidates)
    if keep_count <= 0:
        return torch.zeros_like(candidates), candidates
    score = unexplained_score.float().reshape(batch, phases, cells)
    finite = torch.isfinite(score)
    if not bool(finite.all()):
        raise PhaseTransportError("generate unexplained score is non-finite")
    candidate_flat = candidates.reshape(batch, phases, cells)
    masked_score = score.masked_fill(~candidate_flat, float("-inf"))
    top_indices = masked_score.topk(keep_count, dim=-1, largest=True, sorted=False).indices
    selected = torch.zeros_like(candidate_flat)
    selected.scatter_(-1, top_indices, True)
    retained = (selected & candidate_flat).reshape_as(candidates)
    rejected = candidates & ~retained
    return retained, rejected


def _hard_integer_retrieve(source: Any, offsets: Any) -> Any:
    """Independently gather a hard integer plan for executor consistency QA."""

    import torch

    b, t, h, w, _ = map(int, source.shape)
    rounded = offsets.round().long()
    if not bool(torch.equal(rounded.float(), offsets.float())):
        raise PhaseTransportError("oracle offsets must be exact integer candidates")
    dt, dy, dx = rounded.unbind(dim=1)
    tt = torch.arange(t, device=source.device).view(1, t, 1, 1) + dt
    yy = torch.arange(h, device=source.device).view(1, 1, h, 1) + dy
    xx = torch.arange(w, device=source.device).view(1, 1, 1, w) + dx
    if bool((tt < 0).any() or (tt >= t).any()):
        raise PhaseTransportError("oracle temporal retrieval escaped the source bank")
    if bool((yy < 0).any() or (yy >= h).any()):
        raise PhaseTransportError("oracle vertical retrieval escaped the source bank")
    if bool((xx < 0).any() or (xx >= w).any()):
        raise PhaseTransportError("oracle horizontal retrieval escaped the source bank")
    batch = torch.arange(b, device=source.device).view(b, 1, 1, 1)
    return source[batch, tt, yy, xx]


def build_oracle_plan(
    source: Any,
    target: Any,
    config: PhaseTransportConfig,
    *,
    feature_channels: Optional[int] = None,
) -> PhasePlan:
    """Construct a conservative hard paired-latent correspondence teacher.

    This remains a falsifiable proxy, not ground-truth optical flow.  Every
    feature coordinate is a fixed projection of *all* packed VAE channels.
    Out-of-bank candidates are invalid, transport must beat zero displacement
    by an explicit margin, and (by default) its inverse match must close a
    target->source->target cycle.  The selected hard candidate is the exact
    offset later consumed by :func:`execute_clean_plan`.
    """

    import torch

    config.validate()
    _validate_video(source, label="source")
    _validate_video(target, label="target")
    if tuple(source.shape) != tuple(target.shape):
        raise PhaseTransportError("oracle source/target shapes differ")
    input_channels = int(source.shape[-1])
    channels = input_channels if feature_channels is None else feature_channels
    if type(channels) is not int or channels <= 0 or channels > input_channels:
        raise PhaseTransportError(
            "feature_channels must lie in [1,packed input channels]"
        )
    if channels != input_channels and not config.teacher_allow_lossy_projection:
        raise PhaseTransportError(
            "lossy oracle projections are forbidden on the main path; use all "
            "packed channels or explicitly enable diagnostic ablation"
        )
    projection = fixed_auditable_projection(
        input_channels, channels, device=source.device
    )
    source_f = torch.einsum(
        "bthwd,fd->bthwf", source.float().detach(), projection
    )
    target_f = torch.einsum(
        "bthwd,fd->bthwf", target.float().detach(), projection
    )
    candidates = _candidate_grid(config)
    with torch.no_grad():
        raw_cost, cost, valid = _oracle_cost_volume(source_f, target_f, candidates)
        _, reverse_cost, _ = _oracle_cost_volume(target_f, source_f, candidates)
        reverse_best_index = reverse_cost.argmin(dim=1)

        zero_index = candidates.index((0, 0, 0))
        zero_cost = cost[:, zero_index]
        nonzero_cost_volume = cost.clone()
        nonzero_cost_volume[:, zero_index] = float("inf")
        best_nonzero_cost, best_nonzero_index = nonzero_cost_volume.min(dim=1)
        candidate_tensor = torch.tensor(
            candidates, device=source.device, dtype=torch.float32
        )
        selected = candidate_tensor[best_nonzero_index]
        absolute_improvement = zero_cost - best_nonzero_cost
        # Express the margin relative to the zero-offset reconstruction error.
        # An absolute normalized-cost margin becomes phase/value dependent and
        # can reject an exact transported match merely because latent energy is
        # large.  Exact ties still yield zero rather than a false pass.
        relative_improvement = absolute_improvement / torch.maximum(
            zero_cost, best_nonzero_cost
        ).clamp_min(1e-6)
        margin_pass = relative_improvement >= config.teacher_transport_margin
        nonzero_explainable = best_nonzero_cost <= config.teacher_generate_threshold
        zero_explainable = zero_cost <= config.teacher_generate_threshold
        cycle_consistent = _cycle_consistency_for_selected(
            selected_index=best_nonzero_index,
            reverse_best_index=reverse_best_index,
            candidates=candidates,
            forward_valid=valid,
        )
        cycle_gate = cycle_consistent if config.teacher_require_cycle else torch.ones_like(
            cycle_consistent
        )
        transport_bool = nonzero_explainable & margin_pass & cycle_gate
        prebudget_preserve_bool = (~transport_bool) & zero_explainable
        prebudget_generate_bool = ~(transport_bool | prebudget_preserve_bool)
        # High unmatched coverage commonly signals synthetic-target appearance
        # drift, not genuine local innovation.  Retain only the most
        # unexplained cells per latent phase and conservatively copy source for
        # every rejected candidate.  Never invent a transport offset.
        unexplained_score = torch.minimum(zero_cost, best_nonzero_cost)
        generate_bool, budget_reject_bool = _budget_generate_per_phase(
            prebudget_generate_bool,
            unexplained_score,
            config.max_generate_fraction_per_phase,
        )
        preserve_bool = prebudget_preserve_bool | budget_reject_bool

        offsets = selected.permute(0, 4, 1, 2, 3).contiguous()
        offsets = offsets * transport_bool.unsqueeze(1)
        gates = torch.stack(
            (preserve_bool, transport_bool, generate_bool), dim=1
        ).float()

        # Measure how often the legacy border-clamped argmin would have picked
        # an invalid source coordinate.  This is the directly falsifiable
        # effect of the validity guard rather than a count of harmless invalid
        # candidates in the search volume.
        raw_best_index = raw_cost.argmin(dim=1)
        raw_best_valid = valid.gather(1, raw_best_index.unsqueeze(1)).squeeze(1)
        margin_reject = nonzero_explainable & ~margin_pass
        cycle_reject = nonzero_explainable & margin_pass & ~cycle_consistent

        # Independently gather the selected integer candidates and compare them
        # with the grid-sample executor.  A non-zero value reveals a teacher /
        # executor mismatch and invalidates the oracle diagnostic.
        direct_retrieved = _hard_integer_retrieve(source.float(), offsets)
        executor_retrieved = _hard_integer_retrieve(source.float(), offsets)
        executor_proxy_mse = (direct_retrieved - executor_retrieved).pow(2).mean()
        grid_sampler_numeric_mse = (
            direct_retrieved - transport_source(source.float(), offsets).float()
        ).pow(2).mean()
        projection_meta = projection_audit_metadata(input_channels, channels)
        diagnostics = {
            "preserve_fraction": float(preserve_bool.float().mean().item()),
            "transport_fraction": float(transport_bool.float().mean().item()),
            "generate_fraction": float(generate_bool.float().mean().item()),
            "prebudget_generate_fraction": float(
                prebudget_generate_bool.float().mean().item()
            ),
            "postbudget_generate_fraction": float(generate_bool.float().mean().item()),
            "budget_reject_fraction": float(
                budget_reject_bool.float().mean().item()
            ),
            "budget_reject_fraction_of_prebudget_generate": float(
                budget_reject_bool.float().sum().item()
                / max(prebudget_generate_bool.float().sum().item(), 1.0)
            ),
            "max_generate_fraction_per_phase": config.max_generate_fraction_per_phase,
            "observed_max_prebudget_generate_fraction_per_phase": float(
                prebudget_generate_bool.float().mean(dim=(-2, -1)).max().item()
            ),
            "observed_max_postbudget_generate_fraction_per_phase": float(
                generate_bool.float().mean(dim=(-2, -1)).max().item()
            ),
            "generate_budget_score": "min(zero_cost,best_nonzero_cost)",
            "generate_budget_selection": "per_batch_per_phase_topk_floor",
            "generate_budget_reject_fallback": "preserve",
            "valid_reject_fraction": float((~raw_best_valid).float().mean().item()),
            "invalid_candidate_fraction": float((~valid).float().mean().item()),
            "margin_reject_fraction": float(margin_reject.float().mean().item()),
            "cycle_reject_fraction": float(cycle_reject.float().mean().item())
            if config.teacher_require_cycle
            else 0.0,
            "cycle_inconsistent_fraction": float(
                (~cycle_consistent).float().mean().item()
            ),
            "hard_executor_candidate_mse": float(executor_proxy_mse.item()),
            "grid_sampler_integer_numeric_mse": float(
                grid_sampler_numeric_mse.item()
            ),
            "mean_zero_cost": float(zero_cost.mean().item()),
            "mean_best_nonzero_cost": float(best_nonzero_cost.mean().item()),
            "mean_absolute_zero_improvement": float(
                absolute_improvement.mean().item()
            ),
            "mean_relative_zero_improvement": float(
                relative_improvement.mean().item()
            ),
            "projection": projection_meta,
            "hard_candidate_assignment": True,
            "cycle_gate_enabled": bool(config.teacher_require_cycle),
        }
    plan = PhasePlan(
        offsets=offsets,
        gate_probs=gates,
        provenance="oracle_pair_proxy",
        diagnostics=diagnostics,
    )
    plan.validate(source)
    return plan


def make_proxy_target(source: Any, target: Any, oracle: PhasePlan) -> Any:
    """Use source retrieval wherever possible and target only for generation.

    This is the key protection against synthetic-target identity replacement:
    an explainable target cell is reconstructed from the clean source bank,
    while target appearance is exposed only where correspondence failed.
    """

    return execute_clean_plan(
        source,
        target,
        oracle,
        noop=False,
        detach_source_bank=True,
    ).detach()


def plan_distillation_loss(student: PhasePlan, teacher: PhasePlan, source: Any) -> dict[str, Any]:
    """Supervise student offsets only where the teacher chooses transport."""

    import torch

    student.validate(source)
    teacher.validate(source)
    transport_weight = teacher.gate_probs[:, GATE_TRANSPORT : GATE_TRANSPORT + 1]
    denominator = transport_weight.sum().clamp_min(1.0)
    offset = ((student.offsets.float() - teacher.offsets.float()).abs() * transport_weight).sum() / denominator
    gate = -(teacher.gate_probs.float() * student.gate_probs.float().clamp_min(1e-6).log()).sum(dim=1).mean()
    # Total variation regularizes plans without forcing moving boundaries to be
    # static.  Temporal and spatial axes are all explicit.
    tv_terms = []
    for axis in (2, 3, 4):
        left = student.offsets.narrow(axis, 1, int(student.offsets.shape[axis]) - 1)
        right = student.offsets.narrow(axis, 0, int(student.offsets.shape[axis]) - 1)
        tv_terms.append((left - right).abs().mean())
    smooth = torch.stack(tv_terms).mean()
    return {"offset": offset, "gate": gate, "smooth": smooth}


def _torch_module_base() -> Any:
    try:
        from torch import nn
    except ImportError as error:  # pragma: no cover - exercised on AUH
        raise PhaseTransportError("PhaseTransportAdapter requires PyTorch") from error
    return nn.Module


try:
    import torch
    from torch import nn

    class PhaseTransportAdapter(nn.Module):
        """Small source+instruction-only dense 21-phase planner.

        It operates on clean VAE patch tokens, not target latents or external
        spatial annotations.  Zero-initialized heads begin with an exact
        preserve-biased plan while still allowing the instruction branch to
        learn dense offsets and generation necessity.
        """

        def __init__(self, config: PhaseTransportConfig):
            super().__init__()
            config.validate()
            self.config = config
            hidden = config.hidden_channels
            self.source_in = nn.Conv3d(config.latent_channels, hidden, kernel_size=1)
            self.context = nn.Sequential(
                nn.Linear(config.text_channels, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * hidden),
            )
            self.body = nn.Sequential(
                nn.GroupNorm(min(32, hidden), hidden),
                nn.SiLU(),
                nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=1),
                nn.SiLU(),
            )
            self.offset_head = nn.Conv3d(hidden, 3, kernel_size=1)
            self.gate_head = nn.Conv3d(hidden, 3, kernel_size=1)
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)
            nn.init.zeros_(self.gate_head.weight)
            with torch.no_grad():
                self.gate_head.bias.copy_(torch.tensor((4.0, -2.0, -4.0)))

        def forward(self, source: Any, instruction_embedding: Any) -> PhasePlan:
            _validate_video(source, label="source")
            if int(source.shape[-1]) != self.config.latent_channels:
                raise PhaseTransportError("student source channel count differs")
            if getattr(instruction_embedding, "ndim", None) == 3:
                instruction_embedding = instruction_embedding.float().mean(dim=1)
            if (
                getattr(instruction_embedding, "ndim", None) != 2
                or int(instruction_embedding.shape[0]) != int(source.shape[0])
                or int(instruction_embedding.shape[1]) != self.config.text_channels
            ):
                raise PhaseTransportError("instruction embedding must be [B,text_channels]")
            bank = source.detach() if self.config.source_bank_detach else source
            hidden = self.source_in(bank.permute(0, 4, 1, 2, 3).float())
            scale, shift = self.context(instruction_embedding.float()).chunk(2, dim=-1)
            hidden = hidden * (1.0 + scale[..., None, None, None]) + shift[..., None, None, None]
            hidden = self.body(hidden)
            raw_offset = torch.tanh(self.offset_head(hidden))
            limits = raw_offset.new_tensor(
                (
                    self.config.max_temporal_offset,
                    self.config.max_spatial_offset,
                    self.config.max_spatial_offset,
                )
            ).view(1, 3, 1, 1, 1)
            offsets = raw_offset * limits
            gates = torch.softmax(self.gate_head(hidden), dim=1)
            plan = PhasePlan(offsets=offsets, gate_probs=gates, provenance="student")
            plan.validate(source)
            return plan

except ImportError:  # local contract-only environments intentionally omit torch

    class PhaseTransportAdapter:  # type: ignore[no-redef]
        def __init__(self, config: PhaseTransportConfig):
            config.validate()
            raise PhaseTransportError("PhaseTransportAdapter requires PyTorch")

        def forward(self, source: Any, instruction_embedding: Any) -> PhasePlan:
            raise PhaseTransportError("PhaseTransportAdapter requires PyTorch")
