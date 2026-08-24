"""Motion-masked losses and scalable projected gradient fingerprints.

The loss/masking code follows Motive's attribution semantics: masks reweight
the measurement loss only.  They must not be injected into forward noising or
silently reused as Lucy's edit-support mask.

The paper uses a Fastfood projection, but its implementation is not released at
the time of this prototype.  ``CountSketchProjector`` is a streaming,
low-memory Johnson-Lindenstrauss-style substitute.  It is intentionally named
and recorded as a different backend so experiments cannot be mistaken for an
exact paper reproduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for attribution. Install the optional "
            "dependency with `pip install -e '.[attribution]'`."
        ) from error
    return torch


def normalize_motion_magnitude_torch(
    magnitude: Any,
    *,
    mode: str = "robust",
    low_quantile: float = 0.05,
    high_quantile: float = 0.95,
    absolute_gate: float | None = None,
    eps: float = 1e-6,
) -> Any:
    """Normalize each sample independently with a zero-motion guard.

    Expected input is ``[B, T, H, W]`` or ``[T, H, W]``.  The first form is
    preferred because normalization is per clip, matching the paper.
    """

    torch = _torch()
    values = torch.as_tensor(magnitude)
    squeeze = values.ndim == 3
    if squeeze:
        values = values.unsqueeze(0)
    if values.ndim != 4:
        raise ValueError("magnitude must have shape [B,T,H,W] or [T,H,W]")
    flat = values.float().flatten(1)
    if mode == "motive":
        low = flat.amin(dim=1, keepdim=True)
        high = flat.amax(dim=1, keepdim=True)
    elif mode == "robust":
        low = torch.quantile(flat, low_quantile, dim=1, keepdim=True)
        high_values = []
        for row, row_low in zip(flat, low):
            active = row[row > row_low[0] + eps]
            high_values.append(
                torch.quantile(active, high_quantile)
                if active.numel()
                else row_low[0]
            )
        high = torch.stack(high_values).unsqueeze(1)
    else:
        raise ValueError(f"unsupported normalization mode: {mode}")
    span = high - low
    normalized = ((flat - low) / (span + eps)).clamp_(0.0, 1.0)
    normalized = torch.where(span > eps, normalized, torch.zeros_like(normalized))
    if absolute_gate is not None:
        p90 = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        normalized = torch.where(
            p90 >= absolute_gate,
            normalized,
            torch.zeros_like(normalized),
        )
    # Keep a floating mask even when callers accidentally pass an integer or
    # boolean magnitude tensor; casting normalized values back to those dtypes
    # would silently collapse the mask to {0, 1}.
    output_dtype = values.dtype if values.is_floating_point() else torch.float32
    normalized = normalized.reshape(values.shape).to(output_dtype)
    return normalized[0] if squeeze else normalized


def resize_motion_mask(
    mask: Any,
    *,
    target_frames: int,
    target_height: int,
    target_width: int,
) -> Any:
    """Spatially resize a ``[B,T,H,W]`` mask without temporal interpolation."""

    torch = _torch()
    functional = torch.nn.functional
    values = torch.as_tensor(mask)
    squeeze = values.ndim == 3
    if squeeze:
        values = values.unsqueeze(0)
    if values.ndim != 4:
        raise ValueError("mask must have shape [B,T,H,W] or [T,H,W]")
    if values.shape[1] != target_frames:
        raise ValueError(
            "temporal mask length differs from latent length; resample video "
            "and flow to a common physical fps before attribution"
        )
    batch, frames, height, width = values.shape
    flat = values.reshape(batch * frames, 1, height, width).float()
    resized = functional.interpolate(
        flat,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    result = resized.reshape(batch, frames, target_height, target_width)
    output_dtype = values.dtype if values.is_floating_point() else torch.float32
    result = result.to(output_dtype)
    return result[0] if squeeze else result


def pairwise_motion_to_frame_mask(mask: Any) -> Any:
    """Map ``F-1`` inter-frame motion maps onto an ``F``-frame time grid.

    Motion between frames ``t`` and ``t+1`` contributes to both endpoints.
    Interior frame saliency is the maximum of its incoming and outgoing maps,
    preserving short events instead of averaging them away.
    """

    torch = _torch()
    values = torch.as_tensor(mask)
    squeeze = values.ndim == 3
    if squeeze:
        values = values.unsqueeze(0)
    if values.ndim != 4 or values.shape[1] < 1:
        raise ValueError("pairwise mask must have shape [B,F-1,H,W] with F >= 2")
    if values.shape[1] == 1:
        result = torch.cat((values, values), dim=1)
    else:
        interior = torch.maximum(values[:, :-1], values[:, 1:])
        result = torch.cat((values[:, :1], interior, values[:, -1:]), dim=1)
    return result[0] if squeeze else result


def align_motion_mask_to_latents(
    mask: Any,
    *,
    target_frames: int,
    target_height: int,
    target_width: int,
    input_timing: str = "pairwise",
) -> Any:
    """Explicitly align pixel motion to a spatial/temporal latent grid.

    ``input_timing="pairwise"`` expects optical flow on an ``F-1`` grid and
    first maps it to video frames. Temporal downsampling then uses adaptive max
    pooling so short action events survive VAE temporal compression.
    """

    torch = _torch()
    functional = torch.nn.functional
    values = torch.as_tensor(mask)
    squeeze = values.ndim == 3
    if squeeze:
        values = values.unsqueeze(0)
    if values.ndim != 4:
        raise ValueError("mask must have shape [B,T,H,W] or [T,H,W]")
    if input_timing == "pairwise":
        values = pairwise_motion_to_frame_mask(values)
    elif input_timing != "frame":
        raise ValueError("input_timing must be 'pairwise' or 'frame'")

    batch, frames, height, width = values.shape
    # Max pooling is deliberate here: direct bilinear sampling can completely
    # miss a small hand/object when mapping a sparse mask to a coarse latent.
    spatial = functional.adaptive_max_pool2d(
        values.reshape(batch * frames, 1, height, width).float(),
        output_size=(target_height, target_width),
    ).reshape(batch, frames, target_height, target_width)
    if frames != target_frames:
        temporal = spatial.permute(0, 2, 3, 1).reshape(
            batch * target_height * target_width,
            1,
            frames,
        )
        temporal = functional.adaptive_max_pool1d(temporal, target_frames)
        spatial = temporal.reshape(
            batch,
            target_height,
            target_width,
            target_frames,
        ).permute(0, 3, 1, 2)
    output_dtype = values.dtype if values.is_floating_point() else torch.float32
    result = spatial.to(output_dtype)
    return result[0] if squeeze else result


def _broadcast_mask(mask: Any, prediction: Any) -> Any:
    """Broadcast ``[B,T,H,W]`` over common latent tensor layouts."""

    torch = _torch()
    values = torch.as_tensor(mask, device=prediction.device, dtype=prediction.dtype)
    if prediction.ndim != 5:
        raise ValueError("prediction must be a 5-D latent tensor")
    if values.ndim == 3:
        values = values.unsqueeze(0)
    if values.ndim != 4:
        raise ValueError("motion mask must have shape [B,T,H,W]")

    # Diffusion code commonly uses B,C,T,H,W.  If B,T,C,H,W is used, callers
    # should transpose explicitly; guessing when C == T is unsafe.
    if prediction.shape[0] != values.shape[0]:
        raise ValueError("prediction and mask batch sizes differ")
    if prediction.shape[2] != values.shape[1]:
        raise ValueError(
            "expected prediction layout [B,C,T,H,W] matching mask [B,T,H,W]"
        )
    if tuple(prediction.shape[-2:]) != tuple(values.shape[-2:]):
        values = resize_motion_mask(
            values,
            target_frames=prediction.shape[2],
            target_height=prediction.shape[3],
            target_width=prediction.shape[4],
        )
    return values.unsqueeze(1)


def motion_weighted_mse(
    prediction: Any,
    target: Any,
    motion_mask: Any,
    *,
    reduction: str = "paper",
    inverse_frame_scale: bool = False,
    eps: float = 1e-8,
) -> Any:
    """Motive-style per-location measurement loss.

    ``reduction="paper"`` computes the mean of masked squared errors.  The
    optional ``inverse_frame_scale`` exposes the paper's additional 1/F scalar,
    but defaults off: once the projected gradient is L2-normalized, multiplying
    a whole example gradient by 1/F cannot change cosine rankings.  Physical
    fps/length standardization should happen in preprocessing instead.
    """

    torch = _torch()
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    mask = _broadcast_mask(motion_mask, prediction)
    squared_error = (prediction - target).float().pow(2)
    weighted = squared_error * mask.float()
    if reduction == "paper":
        loss = weighted.mean()
    elif reduction == "active_mean":
        channel_count = prediction.shape[1]
        denominator = mask.float().sum() * channel_count
        loss = weighted.sum() / denominator.clamp_min(eps)
    else:
        raise ValueError("reduction must be 'paper' or 'active_mean'")
    if inverse_frame_scale:
        loss = loss / prediction.shape[2]
    return loss


def action_edit_measurement_loss(
    prediction: Any,
    target: Any,
    delta_motion_mask: Any,
    *,
    preservation_prediction: Any | None = None,
    preservation_target: Any | None = None,
    preservation_weight: float = 0.25,
    min_dynamic_fraction: float = 1e-4,
) -> Any:
    """Measurement loss for paired action editing attribution.

    The dynamic term asks which training pairs help the target motion change.
    The optional complement term protects non-edited regions.  This function is
    for attribution fingerprints; using it as the actual fine-tuning objective
    is a separate experiment.
    """

    torch = _torch()
    mask = torch.as_tensor(delta_motion_mask).float()
    dynamic_fraction = float(torch.mean(mask))
    if not math.isfinite(dynamic_fraction) or dynamic_fraction < min_dynamic_fraction:
        raise ValueError(
            "delta_motion_mask has insufficient dynamic support; exclude this "
            "pair from action attribution or treat it as preservation/no-op data"
        )
    dynamic_loss = motion_weighted_mse(
        prediction,
        target,
        delta_motion_mask,
        reduction="active_mean",
    )
    if preservation_prediction is None and preservation_target is None:
        return dynamic_loss
    if preservation_prediction is None or preservation_target is None:
        raise ValueError("both preservation tensors must be provided")
    preservation_loss = motion_weighted_mse(
        preservation_prediction,
        preservation_target,
        1.0 - mask,
        reduction="active_mean",
    )
    return dynamic_loss + preservation_weight * preservation_loss


@dataclass(frozen=True)
class CountSketchProjector:
    """Deterministic streaming projection with O(output_dim) extra memory."""

    output_dim: int = 512
    seed: int = 0
    chunk_size: int = 1_000_000
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def _chunks(self, tensor: Any) -> Iterator[Any]:
        flat = tensor.reshape(-1)
        for start in range(0, flat.numel(), self.chunk_size):
            yield flat[start : start + self.chunk_size]

    def _hash32(self, indices: Any, *, salt: int) -> Any:
        """Stateless 32-bit avalanche hash for global parameter indices.

        A simple linear-congruential hash becomes periodic modulo common output
        dimensions (notably 512), and changing the seed merely permutes buckets.
        Two rounds of the Thomas-Wang-style mixer avoid that failure while
        remaining chunk- and parameter-boundary invariant.
        """

        torch = _torch()
        mask = 0xFFFFFFFF

        def mix32(values: Any) -> Any:
            values = torch.bitwise_xor(
                values,
                torch.bitwise_right_shift(values, 16),
            )
            values = (values * 0x045D9F3B).bitwise_and(mask)
            values = torch.bitwise_xor(
                values,
                torch.bitwise_right_shift(values, 16),
            )
            values = (values * 0x045D9F3B).bitwise_and(mask)
            return torch.bitwise_xor(
                values,
                torch.bitwise_right_shift(values, 16),
            )

        seed32 = (int(self.seed) ^ int(salt)) & mask
        low = indices.bitwise_and(mask)
        high = torch.bitwise_right_shift(indices, 32).bitwise_and(mask)
        # Fold high bits through a separate avalanche so parameter vectors
        # larger than 2**32 do not repeat the first 2**32 assignments.
        folded_high = mix32(torch.bitwise_xor(high, 0xD1B54A35))
        values = torch.bitwise_xor(torch.bitwise_xor(low, folded_high), seed32)
        return mix32(values)

    def _project_slots(
        self,
        slots: Iterable[tuple[Any | None, int]],
        *,
        normalize: bool,
    ) -> Any:
        """Project tensors while preserving explicitly empty coordinate slots.

        A parameter can be unused by one conditioning arm and therefore have a
        ``None`` gradient.  Omitting that tensor would shift every later
        CountSketch coordinate and make source/target subtraction invalid.
        ``slots`` always advances by the declared flattened size, including
        when the tensor is absent.
        """

        torch = _torch()
        output = None
        global_offset = 0
        found_slot = False
        found_gradient = False
        for tensor, declared_size in slots:
            if (
                isinstance(declared_size, bool)
                or not isinstance(declared_size, int)
                or declared_size <= 0
            ):
                raise ValueError("each CountSketch slot size must be positive")
            found_slot = True
            if tensor is None:
                global_offset += declared_size
                continue
            if tensor.numel() != declared_size:
                raise ValueError(
                    "CountSketch tensor size does not match its declared slot"
                )
            found_gradient = True
            for chunk in self._chunks(tensor.detach()):
                values = chunk.float()
                if output is None:
                    output = torch.zeros(
                        self.output_dim,
                        device=values.device,
                        dtype=torch.float32,
                    )
                elif output.device != values.device:
                    raise ValueError("all gradients must be on the same device")
                local = torch.arange(
                    len(values),
                    device=values.device,
                    dtype=torch.int64,
                )
                indices = local + global_offset
                bucket_hash = self._hash32(indices, salt=0x9E3779B9)
                buckets = torch.remainder(bucket_hash, self.output_dim)
                sign_hash = self._hash32(indices, salt=0xA5A5A5A5)
                signs = (
                    torch.remainder(sign_hash, 2)
                    .float()
                    .mul_(2.0)
                    .sub_(1.0)
                )
                output.scatter_add_(0, buckets, values * signs)
                global_offset += len(values)
        if not found_slot:
            raise ValueError("no CountSketch coordinate slots were provided")
        if not found_gradient or output is None:
            raise ValueError("no tensor gradients were provided for projection")
        if not normalize:
            return output
        norm = torch.linalg.vector_norm(output)
        if float(norm) <= self.eps:
            return torch.zeros_like(output)
        return output / norm

    def project_slots(
        self,
        slots: Iterable[tuple[Any | None, int]],
        *,
        normalize: bool = True,
    ) -> Any:
        """Project explicit tensor/size slots in one stable global coordinate."""

        return self._project_slots(slots, normalize=normalize)

    def project_raw(self, tensors: Iterable[Any]) -> Any:
        """Project tensors without L2 normalization.

        Raw sketches are required for a quotient such as
        ``normalize(P g_target - P g_source)``.  Subtracting two already
        normalized fingerprints is a different representation.
        """

        return self._project_slots(
            ((tensor, int(tensor.numel())) for tensor in tensors if tensor is not None),
            normalize=False,
        )

    def project(self, tensors: Iterable[Any]) -> Any:
        return self._project_slots(
            ((tensor, int(tensor.numel())) for tensor in tensors if tensor is not None),
            normalize=True,
        )


def iter_parameter_gradients(
    model: Any,
    *,
    trainable_only: bool = True,
    name_contains: tuple[str, ...] | None = None,
) -> Iterator[Any]:
    """Yield gradients in stable named-parameter order."""

    matched = 0
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        if name_contains and not any(token in name for token in name_contains):
            continue
        if parameter.grad is None:
            continue
        matched += 1
        yield parameter.grad
    if matched == 0:
        raise ValueError("no matching parameter gradients were found")


def selected_parameter_manifest(
    model: Any,
    *,
    trainable_only: bool = True,
    name_contains: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Describe one ordered parameter subspace without reading weight values."""

    records = []
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        if name_contains and not any(token in name for token in name_contains):
            continue
        records.append(
            {
                "name": str(name),
                "shape": [int(value) for value in parameter.shape],
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
            }
        )
    if not records:
        raise ValueError("no model parameters matched the requested subspace")
    return {
        "ordered_parameters": records,
        "parameter_tensors": len(records),
        "parameter_count": int(sum(record["numel"] for record in records)),
    }


def project_parameter_gradients(
    model: Any,
    *,
    projector: CountSketchProjector | None = None,
    trainable_only: bool = True,
    name_contains: tuple[str, ...] | None = None,
    normalize: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Project a stable model-parameter coordinate system.

    Unlike ``iter_parameter_gradients``, this function retains a zero slot for
    every selected parameter whose gradient is ``None``.  This is essential
    when raw projected gradients from different conditioning cells are later
    subtracted.
    """

    projector = projector or CountSketchProjector()
    slots = []
    selected_names = []
    missing_names = []
    finite = True
    nonzero_tensors = 0
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        if name_contains and not any(token in name for token in name_contains):
            continue
        selected_names.append(str(name))
        gradient = parameter.grad
        if gradient is None:
            missing_names.append(str(name))
        else:
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            finite = finite and bool(_torch().isfinite(values.detach()).all())
            if bool(_torch().count_nonzero(values.detach())):
                nonzero_tensors += 1
        slots.append((gradient, int(parameter.numel())))
    if not slots:
        raise ValueError("no model parameters matched the requested subspace")
    projected = projector.project_slots(slots, normalize=normalize)
    diagnostics: dict[str, Any] = {
        "selected_parameter_tensors": len(slots),
        "selected_parameter_count": int(sum(size for _gradient, size in slots)),
        "gradient_tensors_present": len(slots) - len(missing_names),
        "gradient_tensors_nonzero": nonzero_tensors,
        "all_present_gradients_finite": bool(finite),
        "missing_gradient_names": missing_names,
        "selected_parameter_names": selected_names,
        "projection_normalized": bool(normalize),
        "raw_projection_l2": float(_torch().linalg.vector_norm(projected.detach())),
    }
    return projected, diagnostics


def normalize_projected_tangent(value: Any, *, eps: float = 1e-12) -> Any:
    """L2-normalize one projected tangent with an explicit zero guard."""

    torch = _torch()
    vector = torch.as_tensor(value).float()
    if vector.ndim != 1:
        raise ValueError("projected tangent must be one-dimensional")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("projected tangent contains non-finite values")
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= eps:
        return torch.zeros_like(vector)
    return vector / norm


def factorial_edit_tangents(
    cells: Mapping[str, Any],
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Build target, paired-delta, DID, and cross-cell tangent candidates.

    Required cells use a fixed source condition:

    ``tc`` = target video with edit instruction,
    ``sc`` = source video with edit instruction,
    ``t0`` = target video with no-op instruction, and
    ``s0`` = source video with no-op instruction.

    Every cell must be an *unnormalized* projected gradient.  Differences are
    formed first and normalized only once at the end.
    """

    required = {"tc", "sc", "t0", "s0"}
    missing = sorted(required - set(cells))
    extra = sorted(set(cells) - required)
    if missing or extra:
        raise ValueError(
            f"factorial tangent cells differ; missing={missing}, extra={extra}"
        )
    torch = _torch()
    vectors = {name: torch.as_tensor(cells[name]).float() for name in sorted(required)}
    shapes = {tuple(vector.shape) for vector in vectors.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError("all factorial tangent cells must share one vector shape")
    if not all(bool(torch.isfinite(vector).all()) for vector in vectors.values()):
        raise ValueError("factorial tangent cells contain non-finite values")
    raw = {
        "target_only": vectors["tc"],
        "paired_delta": vectors["tc"] - vectors["sc"],
        "factorial_did": (
            vectors["tc"]
            - vectors["sc"]
            - vectors["t0"]
            + vectors["s0"]
        ),
        "cross_cell_control": vectors["tc"] - vectors["s0"],
        "noop_target_delta": vectors["t0"] - vectors["s0"],
    }
    return {
        name: normalize_projected_tangent(vector, eps=eps)
        for name, vector in raw.items()
    }


def gradient_fingerprint(
    model: Any,
    loss: Any,
    *,
    projector: CountSketchProjector | None = None,
    trainable_only: bool = True,
    name_contains: tuple[str, ...] | None = None,
    retain_graph: bool = False,
) -> Any:
    """Backpropagate one deterministic measurement loss and project gradients.

    The caller is responsible for using the same checkpoint, parameter subset,
    VAE posterior realization, diffusion timestep, and Gaussian noise across
    every training/query example.
    """

    projector = projector or CountSketchProjector()
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    return projector.project(
        iter_parameter_gradients(
            model,
            trainable_only=trainable_only,
            name_contains=name_contains,
        )
    )
