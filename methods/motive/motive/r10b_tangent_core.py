"""Core contracts for the R10B frozen-generator tangent experiment.

R10A rejected the cheap six-frame DINO/track descriptor family.  R10B changes
the information source: it measures projected parameter gradients of a frozen
instruction-conditioned video generator.  Computing gradients here is an
attribution measurement only.  This module contains no optimizer, rendering,
checkpoint mutation, or editor-training path.

The experiment is *Motive-aligned but not a paper reproduction*:

* Motive uses a full generator parameter gradient and Fastfood.
* R10B first uses a preregistered late attention subspace and CountSketch.
* Motive represents one motion-weighted sample gradient.
* R10B additionally tests paired and factorial source/target quotients for
  action editing.  Those quotients are project extensions, not paper claims.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SMOKE_MANIFEST_SCHEMA = "motive-r10b-frozen-tangent-smoke-manifest-v1"
SMOKE_ROW_SCHEMA = "motive-r10b-frozen-tangent-smoke-row-v1"
PARAMETER_ROLES_SCHEMA = "motive-r10b-parameter-roles-v1"
TRACK_SALIENCY_SCHEMA = "motive-r10b-track-delta-saliency-v1"

DEFAULT_DISCOVERY_FAMILIES = (
    "sit_down",
    "lie_down",
    "jump",
    "walk",
    "wave",
    "pour",
    "fly",
    "stand_up",
)

_OPEN_RE = re.compile(r"\b(open|opening|uncover|unseal)\b", re.IGNORECASE)
_CLOSE_RE = re.compile(r"\b(close|closing|closed|shut|shutting|seal)\b", re.IGNORECASE)
_COMPOUND_RE = re.compile(
    r"(?:[,;]|\b(?:and then|then|before|after|followed by|while also|as well as)\b)",
    re.IGNORECASE,
)
_ATOMIC_PROMPT_PATTERNS = {
    "sit_down": re.compile(
        r"\b(?:sit(?:s|ting)? down|take(?:s| taking)? (?:a )?seat|lower(?:s|ing)? "
        r"(?:himself|herself|themselves) (?:onto|into))\b",
        re.IGNORECASE,
    ),
    "lie_down": re.compile(
        r"\b(?:lie(?:s| lying)? down|lay(?:s|ing)? down|recline(?:s|d|ing)?|"
        r"lower(?:s|ing)? (?:himself|herself|themselves) (?:onto|into) .*"
        r"(?:bed|ground|floor|sofa|couch))\b",
        re.IGNORECASE,
    ),
    "jump": re.compile(r"\b(?:jump|jumps|jumping|leap|leaps|leaping|hop|hops|hopping)\b", re.IGNORECASE),
    "walk": re.compile(r"\b(?:walk|walks|walking|stroll|strolls|strolling)\b", re.IGNORECASE),
    "wave": re.compile(r"\b(?:wave|waves|waving)\b", re.IGNORECASE),
    "pour": re.compile(r"\b(?:pour|pours|pouring|poured)\b", re.IGNORECASE),
    "fly": re.compile(r"\b(?:fly|flies|flying|take off|takes off|taking off)\b", re.IGNORECASE),
    "stand_up": re.compile(
        r"\b(?:stand(?:s|ing)? up|rise(?:s| rising)? to (?:his|her|their) feet|"
        r"get(?:s|ting)? up)\b",
        re.IGNORECASE,
    ),
    "open": _OPEN_RE,
    "close": _CLOSE_RE,
}
_LUCY_ATTENTION_RE = re.compile(
    r"(?:^|\.)blocks\.(?P<block>\d+)\."
    r"(?P<branch>attn1|attn2)\."
    r"(?P<projection>to_q|to_k|to_v|to_out\.0)\.weight$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class R10BTangentError(ValueError):
    """A frozen-tangent input, selection, or geometry contract is invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise R10BTangentError(
                    f"{path}:{line_number} is not valid JSON"
                ) from error
            if not isinstance(value, dict):
                raise R10BTangentError(
                    f"{path}:{line_number} must contain one JSON object"
                )
            rows.append(value)
    if not rows:
        raise R10BTangentError(f"{path} contains no rows")
    return rows


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise R10BTangentError(f"{field} must be one lowercase SHA-256")
    return value


def _require_relative_media_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise R10BTangentError(f"{field} must be one non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise R10BTangentError(
            f"{field} must be normalized and remain below data_root"
        )
    return value


def _require_data_root(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not os.path.isabs(value)
        or value.startswith("//")
        or value != os.path.normpath(value)
    ):
        raise R10BTangentError(
            f"{field} must be one normalized absolute directory path"
        )
    return value


def validate_smoke_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate the sealed row contract before an expensive GPU extraction."""

    if not rows:
        raise R10BTangentError("selected smoke manifest is empty")
    seen_iids = set()
    seen_components = set()
    for row_index, row in enumerate(rows):
        prefix = f"smoke row {row_index}"
        if not isinstance(row, Mapping):
            raise R10BTangentError(f"{prefix} must be one object")
        if row.get("schema_version") != SMOKE_ROW_SCHEMA:
            raise R10BTangentError(f"{prefix} schema differs")
        source_split = row.get("source_split")
        if (
            row.get("fresh") is not True
            or not isinstance(source_split, str)
            or not source_split
            or source_split == "test"
        ):
            raise R10BTangentError(f"{prefix} is not fresh non-test evidence")
        for field in (
            "formal_evidence",
            "training_authorized",
            "renderer_probe_authorized",
        ):
            if row.get(field) is not False:
                raise R10BTangentError(f"{prefix} false gate differs: {field}")
        iid = row.get("iid")
        component_id = row.get("component_id")
        family = row.get("family")
        if not isinstance(iid, str) or not iid or iid in seen_iids:
            raise R10BTangentError(f"{prefix} has a missing/duplicate iid")
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in seen_components
        ):
            raise R10BTangentError(
                f"{prefix} has a missing/duplicate component_id"
            )
        if not isinstance(family, str) or not family:
            raise R10BTangentError(f"{prefix} family is missing")
        seen_iids.add(iid)
        seen_components.add(component_id)
        _require_data_root(row.get("data_root"), f"{prefix} data_root")
        _require_relative_media_path(row.get("src_video"), f"{prefix} src_video")
        _require_relative_media_path(row.get("tgt_video"), f"{prefix} tgt_video")
        _require_sha256(
            row.get("src_video_sha256"),
            f"{prefix} src_video_sha256",
        )
        _require_sha256(
            row.get("tgt_video_sha256"),
            f"{prefix} tgt_video_sha256",
        )
        _require_sha256(
            row.get("candidate_input_digest"),
            f"{prefix} candidate_input_digest",
        )
        for field in ("track_input_index", "track_cache_index"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise R10BTangentError(
                    f"{prefix} {field} must be a nonnegative integer"
                )


def validate_track_cache_arrays(cache: Mapping[str, np.ndarray]) -> None:
    """Reject ambiguous row mappings and malformed paired-track geometry."""

    input_indices = np.asarray(cache["input_indices"])
    if (
        input_indices.ndim != 1
        or not np.issubdtype(input_indices.dtype, np.integer)
        or np.any(input_indices < 0)
        or len(np.unique(input_indices)) != len(input_indices)
    ):
        raise R10BTangentError(
            "track cache input_indices must be unique nonnegative integers"
        )
    rows = len(input_indices)
    source_tracks = np.asarray(cache["source_stabilized_tracks"])
    target_tracks = np.asarray(cache["target_stabilized_tracks"])
    source_visibility = np.asarray(cache["source_visibility"])
    target_visibility = np.asarray(cache["target_visibility"])
    if (
        source_tracks.shape != target_tracks.shape
        or source_tracks.ndim != 4
        or source_tracks.shape[0] != rows
        or source_tracks.shape[1] < 2
        or source_tracks.shape[2] < 1
        or source_tracks.shape[3] != 2
    ):
        raise R10BTangentError(
            "track cache source/target tracks must share shape [R,F,N,2]"
        )
    if (
        source_visibility.shape != source_tracks.shape[:3]
        or target_visibility.shape != source_tracks.shape[:3]
    ):
        raise R10BTangentError(
            "track cache visibility must match track geometry [R,F,N]"
        )
    if not all(
        np.isfinite(values).all()
        for values in (
            source_tracks,
            target_tracks,
            source_visibility,
            target_visibility,
        )
    ):
        raise R10BTangentError("track cache geometry contains non-finite values")
    for name in (
        "source_camera_valid",
        "target_camera_valid",
        "source_track_valid",
        "target_track_valid",
    ):
        values = np.asarray(cache[name])
        if values.shape != (rows,) or values.dtype != np.bool_:
            raise R10BTangentError(
                f"track cache {name} must be one boolean value per row"
            )


def split_directional_family(primary_family: str, prompt: str) -> str | None:
    """Split the conflated ``open_close`` pseudo-family when text permits."""

    family = str(primary_family).strip().lower()
    if family != "open_close":
        return family or None
    has_open = bool(_OPEN_RE.search(prompt))
    has_close = bool(_CLOSE_RE.search(prompt))
    if has_open == has_close:
        return None
    return "open" if has_open else "close"


def strict_atomic_family(primary_family: str, prompt: str) -> str | None:
    """Return a high-precision one-action key or reject a compound prompt."""

    family = split_directional_family(primary_family, prompt)
    if family is None or family not in _ATOMIC_PROMPT_PATTERNS:
        return None
    if _COMPOUND_RE.search(prompt):
        return None
    if not _ATOMIC_PROMPT_PATTERNS[family].search(prompt):
        return None
    # Reject prompts that explicitly contain another panel action.  This is
    # intentionally conservative: R10B gradients are too expensive to spend
    # on the broad, compound R7 pseudo taxonomy.
    other_hits = [
        other
        for other, pattern in _ATOMIC_PROMPT_PATTERNS.items()
        if other != family and pattern.search(prompt)
    ]
    if other_hits:
        return None
    return family


def _weighted_coordinate_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    order = np.argsort(values[valid], kind="mergesort")
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    threshold = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def track_delta_components(
    source_tracks: np.ndarray,
    target_tracks: np.ndarray,
    source_visibility: np.ndarray,
    target_visibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera-robust edit velocity, magnitude, and midpoint positions."""

    source = np.asarray(source_tracks, dtype=np.float64)
    target = np.asarray(target_tracks, dtype=np.float64)
    source_vis = np.asarray(source_visibility, dtype=np.float64)
    target_vis = np.asarray(target_visibility, dtype=np.float64)
    if (
        source.shape != target.shape
        or source.ndim != 3
        or source.shape[-1] != 2
        or source.shape[0] < 2
    ):
        raise R10BTangentError(
            "source/target tracks must share shape [F,N,2] with F >= 2"
        )
    if source_vis.shape != source.shape[:2] or target_vis.shape != source.shape[:2]:
        raise R10BTangentError("visibility must match track shape [F,N]")
    if not all(
        np.isfinite(value).all()
        for value in (source, target, source_vis, target_vis)
    ):
        raise R10BTangentError("track inputs contain non-finite values")

    visibility = np.minimum.reduce(
        (
            source_vis[:-1],
            source_vis[1:],
            target_vis[:-1],
            target_vis[1:],
        )
    ).clip(0.0, 1.0)
    source_velocity = np.diff(source, axis=0)
    target_velocity = np.diff(target, axis=0)
    edit_velocity = target_velocity - source_velocity

    # The tracks are already affine stabilized.  This second robust centering
    # rejects small residual global translations without erasing non-rigid
    # actor motion.
    for transition in range(edit_velocity.shape[0]):
        for coordinate in range(2):
            edit_velocity[transition, :, coordinate] -= _weighted_coordinate_median(
                edit_velocity[transition, :, coordinate],
                visibility[transition],
            )

    acceleration = np.zeros_like(edit_velocity)
    if edit_velocity.shape[0] > 1:
        acceleration[1:] = np.diff(edit_velocity, axis=0)
        acceleration[0] = acceleration[1]
    magnitude = (
        np.linalg.norm(edit_velocity, axis=-1)
        + 0.5 * np.linalg.norm(acceleration, axis=-1)
    ) * visibility
    # Keep every factorial cell in the same paired coordinate system.  Using
    # only target positions would shift the loss support between source and
    # target whenever the edit moves the actor, contaminating the quotient
    # with mask misalignment.
    midpoint = 0.25 * (
        source[:-1] + source[1:] + target[:-1] + target[1:]
    )
    return edit_velocity.astype(np.float32), magnitude.astype(np.float32), midpoint.astype(
        np.float32
    )


def track_delta_saliency(
    source_tracks: np.ndarray,
    target_tracks: np.ndarray,
    source_visibility: np.ndarray,
    target_visibility: np.ndarray,
    *,
    height: int = 32,
    width: int = 32,
    dilation_radius: int = 1,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize paired stabilized tracks into a soft pairwise motion mask."""

    if height <= 1 or width <= 1:
        raise R10BTangentError("saliency height/width must exceed one")
    if dilation_radius < 0:
        raise R10BTangentError("dilation_radius must be nonnegative")
    _velocity, magnitude, midpoint = track_delta_components(
        source_tracks,
        target_tracks,
        source_visibility,
        target_visibility,
    )
    frames, tracks = magnitude.shape
    canvas = np.zeros((frames, height, width), dtype=np.float32)
    weight_canvas = np.zeros_like(canvas)

    clipped_x = np.clip(midpoint[..., 0], 0.0, 1.0) * (width - 1)
    clipped_y = np.clip(midpoint[..., 1], 0.0, 1.0) * (height - 1)
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = clipped_x - x0
    wy = clipped_y - y0

    for transition in range(frames):
        values = magnitude[transition]
        for xs, ys, weights in (
            (x0[transition], y0[transition], (1.0 - wx[transition]) * (1.0 - wy[transition])),
            (x1[transition], y0[transition], wx[transition] * (1.0 - wy[transition])),
            (x0[transition], y1[transition], (1.0 - wx[transition]) * wy[transition]),
            (x1[transition], y1[transition], wx[transition] * wy[transition]),
        ):
            np.add.at(canvas[transition], (ys, xs), values * weights)
            np.add.at(weight_canvas[transition], (ys, xs), weights)
    canvas = np.divide(
        canvas,
        np.maximum(weight_canvas, eps),
        out=np.zeros_like(canvas),
        where=weight_canvas > eps,
    )

    if dilation_radius:
        padded = np.pad(
            canvas,
            ((0, 0), (dilation_radius, dilation_radius), (dilation_radius, dilation_radius)),
            mode="constant",
        )
        dilated = np.zeros_like(canvas)
        kernel = 2 * dilation_radius + 1
        for dy in range(kernel):
            for dx in range(kernel):
                dilated = np.maximum(
                    dilated,
                    padded[:, dy : dy + height, dx : dx + width],
                )
        canvas = dilated

    flat = canvas.reshape(-1)
    active = flat[flat > eps]
    if active.size:
        low = float(np.quantile(active, 0.05))
        high = float(np.quantile(active, 0.95))
    else:
        low = high = 0.0
    if high - low <= eps:
        normalized = (
            (canvas > eps).astype(np.float32)
            if high > eps
            else np.zeros_like(canvas)
        )
    else:
        normalized = np.clip((canvas - low) / (high - low + eps), 0.0, 1.0)
    diagnostics = {
        "schema_version": TRACK_SALIENCY_SCHEMA,
        "source_frames": int(source_tracks.shape[0]),
        "tracks": int(tracks),
        "pairwise_frames": int(frames),
        "raster_height": int(height),
        "raster_width": int(width),
        "dilation_radius": int(dilation_radius),
        "raw_mean": float(canvas.mean()),
        "raw_p90": float(np.quantile(canvas, 0.90)),
        "raw_p99": float(np.quantile(canvas, 0.99)),
        "normalized_mean": float(normalized.mean()),
        "normalized_active_fraction": float(np.mean(normalized > eps)),
        "track_delta_energy": float(np.mean(magnitude)),
        "track_delta_p90": float(np.quantile(magnitude, 0.90)),
    }
    return normalized.astype(np.float32), diagnostics


def resolve_lucy_attention_roles(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    block_index: int | None = None,
    include_cross_kv: bool = False,
) -> dict[str, Any]:
    """Resolve late self/cross-attention roles from actual parameter names."""

    matches = []
    sizes: dict[str, int] = {}
    for name, parameter in named_parameters:
        match = _LUCY_ATTENTION_RE.search(str(name))
        if match is None:
            continue
        record = {
            "name": str(name),
            "block": int(match.group("block")),
            "branch": match.group("branch"),
            "projection": match.group("projection"),
            "numel": int(parameter.numel()),
        }
        matches.append(record)
        sizes[str(name)] = int(parameter.numel())
    if not matches:
        raise R10BTangentError(
            "transformer exposes no recognized Wan/Lucy attention weights"
        )
    available_blocks = sorted({record["block"] for record in matches})
    selected_block = available_blocks[-1] if block_index is None else int(block_index)
    if selected_block not in available_blocks:
        raise R10BTangentError(
            f"requested block {selected_block} is not available: {available_blocks}"
        )

    self_names = sorted(
        record["name"]
        for record in matches
        if record["block"] == selected_block
        and record["branch"] == "attn1"
        and record["projection"] in {"to_q", "to_k", "to_v"}
    )
    cross_projections = {"to_q", "to_k", "to_v"} if include_cross_kv else {"to_q"}
    cross_names = sorted(
        record["name"]
        for record in matches
        if record["block"] == selected_block
        and record["branch"] == "attn2"
        and record["projection"] in cross_projections
    )
    if len(self_names) != 3:
        raise R10BTangentError(
            f"self-attention role requires q/k/v, found {self_names}"
        )
    expected_cross = 3 if include_cross_kv else 1
    if len(cross_names) != expected_cross:
        raise R10BTangentError(
            f"cross-attention role expected {expected_cross} weights, found {cross_names}"
        )
    overlap = set(self_names) & set(cross_names)
    if overlap:
        raise R10BTangentError(f"parameter roles overlap: {sorted(overlap)}")
    return {
        "schema_version": PARAMETER_ROLES_SCHEMA,
        "resolver": "wan_lucy_last_block_attention_v1",
        "available_blocks": available_blocks,
        "selected_block": selected_block,
        "roles": {
            "self_motion": {
                "names": self_names,
                "parameter_count": int(sum(sizes[name] for name in self_names)),
            },
            "cross_instruction": {
                "names": cross_names,
                "parameter_count": int(sum(sizes[name] for name in cross_names)),
            },
        },
    }


def set_only_parameters_trainable(
    model: Any,
    parameter_roles: Mapping[str, Any],
) -> None:
    selected = {
        name
        for role in parameter_roles["roles"].values()
        for name in role["names"]
    }
    observed = set()
    for name, parameter in model.named_parameters():
        is_selected = str(name) in selected
        parameter.requires_grad_(is_selected)
        if is_selected:
            observed.add(str(name))
    missing = sorted(selected - observed)
    if missing:
        raise R10BTangentError(f"selected parameters disappeared: {missing}")


def temporal_broadcast_noise(
    reference: Any,
    *,
    seed: int,
) -> Any:
    """Generate one spatial noise field and broadcast it over latent time."""

    import torch

    if reference.ndim != 5 or reference.shape[2] < 2:
        raise R10BTangentError("latent reference must have shape [B,C,T,H,W]")
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    base = torch.randn(
        (
            reference.shape[0],
            reference.shape[1],
            1,
            reference.shape[3],
            reference.shape[4],
        ),
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )
    return base.expand_as(reference)


def motion_x0_measurement_loss(
    prediction_velocity: Any,
    noisy_latents: Any,
    clean_latents: Any,
    pairwise_motion_mask: Any,
    *,
    sigma: float,
    level_weight: float = 0.25,
    temporal_weight: float = 0.75,
) -> tuple[Any, dict[str, float]]:
    """Motion-weighted x0 and temporal-difference loss for one factorial cell."""

    import torch

    from .attribution import align_motion_mask_to_latents, motion_weighted_mse

    if not 0.0 < float(sigma) < 1.0:
        raise R10BTangentError("sigma must lie strictly between zero and one")
    if (
        prediction_velocity.shape != noisy_latents.shape
        or clean_latents.shape != noisy_latents.shape
    ):
        raise R10BTangentError("prediction/noisy/clean latent shapes differ")
    if prediction_velocity.shape[2] < 2:
        raise R10BTangentError("motion loss needs at least two latent frames")
    if level_weight < 0 or temporal_weight <= 0 or level_weight + temporal_weight <= 0:
        raise R10BTangentError("invalid level/temporal loss weights")

    x0_hat = noisy_latents.float() - float(sigma) * prediction_velocity.float()
    clean = clean_latents.float()
    level_mask = align_motion_mask_to_latents(
        pairwise_motion_mask,
        target_frames=clean.shape[2],
        target_height=clean.shape[3],
        target_width=clean.shape[4],
        input_timing="pairwise",
    ).to(device=clean.device, dtype=clean.dtype)
    pair_mask = align_motion_mask_to_latents(
        pairwise_motion_mask,
        target_frames=clean.shape[2] - 1,
        target_height=clean.shape[3],
        target_width=clean.shape[4],
        input_timing="frame",
    ).to(device=clean.device, dtype=clean.dtype)
    level = motion_weighted_mse(
        x0_hat,
        clean,
        level_mask,
        reduction="active_mean",
    )
    temporal = motion_weighted_mse(
        torch.diff(x0_hat, dim=2),
        torch.diff(clean, dim=2),
        pair_mask,
        reduction="active_mean",
    )
    total_weight = float(level_weight + temporal_weight)
    loss = (
        float(level_weight) * level + float(temporal_weight) * temporal
    ) / total_weight
    return loss, {
        "level_loss": float(level.detach()),
        "temporal_loss": float(temporal.detach()),
        "combined_loss": float(loss.detach()),
        "level_weight": float(level_weight / total_weight),
        "temporal_weight": float(temporal_weight / total_weight),
    }


def _selection_energy(
    cache: Mapping[str, np.ndarray],
    cache_index: int,
) -> float:
    _velocity, magnitude, _midpoint = track_delta_components(
        cache["source_stabilized_tracks"][cache_index],
        cache["target_stabilized_tracks"][cache_index],
        cache["source_visibility"][cache_index],
        cache["target_visibility"][cache_index],
    )
    return float(np.quantile(magnitude, 0.90))


def build_smoke_manifest(
    *,
    candidate_manifest: str | Path,
    track_cache: str | Path,
    track_manifest: str | Path,
    families: Sequence[str] = DEFAULT_DISCOVERY_FAMILIES,
    per_family: int = 1,
    max_total: int | None = None,
    seed: int = 260108847,
) -> dict[str, Any]:
    """Select a deterministic high-motion, component-disjoint dev-only pilot."""

    if per_family <= 0:
        raise R10BTangentError("per_family must be positive")
    requested_families = tuple(str(value) for value in families)
    if not requested_families or len(set(requested_families)) != len(requested_families):
        raise R10BTangentError("families must be non-empty and unique")
    candidate_rows = read_jsonl(candidate_manifest)
    track_rows = read_jsonl(track_manifest)
    if len(candidate_rows) != len(track_rows):
        raise R10BTangentError(
            "candidate and track manifests must have identical row counts"
        )
    track_by_iid = {}
    for row in track_rows:
        iid = str(row.get("iid", ""))
        if not iid or iid in track_by_iid:
            raise R10BTangentError("track manifest has missing/duplicate iid")
        track_by_iid[iid] = row

    with np.load(track_cache, allow_pickle=False) as archive:
        required_arrays = {
            "input_indices",
            "source_stabilized_tracks",
            "target_stabilized_tracks",
            "source_visibility",
            "target_visibility",
            "source_camera_valid",
            "target_camera_valid",
            "source_track_valid",
            "target_track_valid",
        }
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise R10BTangentError(f"track cache is missing arrays: {missing}")
        cache = {name: np.asarray(archive[name]) for name in required_arrays}
    validate_track_cache_arrays(cache)

    input_to_cache = {
        int(input_index): cache_index
        for cache_index, input_index in enumerate(cache["input_indices"])
    }
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion_counts: Counter[str] = Counter()
    for row_index, row in enumerate(candidate_rows):
        iid = str(row.get("iid", ""))
        track_row = track_by_iid.get(iid)
        if track_row is None:
            exclusion_counts["missing_track_row"] += 1
            continue
        label = row.get("label")
        assignment = row.get("assignment")
        bindings = row.get("source_bindings", {}).get("media", {})
        if not isinstance(label, dict) or label.get("class") != "positive":
            exclusion_counts["not_positive"] += 1
            continue
        if not isinstance(assignment, dict) or not bool(assignment.get("fresh")):
            exclusion_counts["not_fresh"] += 1
            continue
        source_split = assignment.get("split")
        if not isinstance(source_split, str) or not source_split:
            exclusion_counts["split_missing"] += 1
            continue
        if source_split == "test":
            exclusion_counts["legacy_test"] += 1
            continue
        family = strict_atomic_family(
            str(label.get("primary_family", "")),
            str(row.get("prompt", "")),
        )
        if family not in requested_families:
            exclusion_counts["family_not_requested_or_ambiguous"] += 1
            continue
        input_index = int(track_row.get("input_index", -1))
        cache_index = input_to_cache.get(input_index)
        if cache_index is None:
            exclusion_counts["track_input_index_missing"] += 1
            continue
        valid = all(
            bool(cache[name][cache_index])
            for name in (
                "source_camera_valid",
                "target_camera_valid",
                "source_track_valid",
                "target_track_valid",
            )
        )
        if not valid or not bool(track_row.get("paired_camera_valid")):
            exclusion_counts["paired_track_or_camera_invalid"] += 1
            continue
        if not isinstance(bindings, dict):
            exclusion_counts["media_binding_missing"] += 1
            continue
        source_media = bindings.get("src_video")
        target_media = bindings.get("tgt_video")
        data_root = bindings.get("data_root")
        if (
            not isinstance(source_media, dict)
            or not isinstance(target_media, dict)
            or not isinstance(data_root, str)
        ):
            exclusion_counts["media_binding_missing"] += 1
            continue
        try:
            data_root = _require_data_root(data_root, f"{iid} data_root")
            source_relative_path = _require_relative_media_path(
                source_media.get("relative_path"),
                f"{iid} src_video.relative_path",
            )
            target_relative_path = _require_relative_media_path(
                target_media.get("relative_path"),
                f"{iid} tgt_video.relative_path",
            )
            source_sha256 = _require_sha256(
                source_media.get("sha256"),
                f"{iid} src_video.sha256",
            )
            target_sha256 = _require_sha256(
                target_media.get("sha256"),
                f"{iid} tgt_video.sha256",
            )
            candidate_input_digest = _require_sha256(
                row.get("input_digest"),
                f"{iid} input_digest",
            )
        except R10BTangentError:
            exclusion_counts["media_binding_invalid"] += 1
            continue
        component_id = str(assignment.get("component_id", ""))
        if not component_id:
            exclusion_counts["component_missing"] += 1
            continue
        energy = _selection_energy(cache, cache_index)
        tie = object_digest({"seed": int(seed), "iid": iid})
        candidates[family].append(
            {
                "schema_version": SMOKE_ROW_SCHEMA,
                "iid": iid,
                "family": family,
                "primary_family": str(label.get("primary_family")),
                "prompt": str(row.get("prompt", "")),
                "noop_prompt": "Keep the video unchanged.",
                "component_id": component_id,
                "source_split": source_split,
                "fresh": True,
                "candidate_row_index": int(row_index),
                "track_input_index": int(input_index),
                "track_cache_index": int(cache_index),
                "track_delta_p90": float(energy),
                "data_root": data_root,
                "src_video": source_relative_path,
                "tgt_video": target_relative_path,
                "src_video_sha256": source_sha256,
                "tgt_video_sha256": target_sha256,
                "candidate_input_digest": candidate_input_digest,
                "component_source": "r7_indexed_visual_component",
                "label_provenance": str(label.get("provenance_kind", "")),
                "human_label": bool(label.get("human_label", False)),
                "formal_evidence": False,
                "training_authorized": False,
                "renderer_probe_authorized": False,
                "_tie": tie,
            }
        )

    selected = []
    used_components = set()
    shortfalls = {}
    for family in requested_families:
        ranked = sorted(
            candidates.get(family, ()),
            key=lambda row: (-row["track_delta_p90"], row["_tie"]),
        )
        family_selected = []
        for row in ranked:
            if row["component_id"] in used_components:
                continue
            used_components.add(row["component_id"])
            row = dict(row)
            row.pop("_tie")
            row["within_family_rank"] = len(family_selected) + 1
            family_selected.append(row)
            if len(family_selected) == per_family:
                break
        selected.extend(family_selected)
        if len(family_selected) < per_family:
            shortfalls[family] = {
                "required": int(per_family),
                "selected": len(family_selected),
                "eligible_before_component_dedup": len(ranked),
            }
    if shortfalls:
        raise R10BTangentError(f"family support shortfall: {shortfalls}")
    selected.sort(key=lambda row: (requested_families.index(row["family"]), row["within_family_rank"]))
    if max_total is not None:
        if max_total <= 0:
            raise R10BTangentError("max_total must be positive")
        selected = selected[: int(max_total)]
    if not selected:
        raise R10BTangentError("smoke selection produced no rows")
    validate_smoke_rows(selected)

    rows_sha256 = hashlib.sha256(
        "".join(canonical_json(row) + "\n" for row in selected).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SMOKE_MANIFEST_SCHEMA,
        "experiment_role": "engineering_smoke_or_nonpromotional_pilot",
        "rows": selected,
        "summary": {
            "rows": len(selected),
            "families": sorted({row["family"] for row in selected}),
            "family_counts": dict(sorted(Counter(row["family"] for row in selected).items())),
            "unique_components": len({row["component_id"] for row in selected}),
            "all_fresh": all(row["fresh"] for row in selected),
            "legacy_test_rows": sum(row["source_split"] == "test" for row in selected),
            "human_label_rows": sum(row["human_label"] for row in selected),
            "formal_evidence": False,
            "renderer_probe_authorized": False,
            "training_authorized": False,
            "selection_seed": int(seed),
            "requested_families": list(requested_families),
            "per_family": int(per_family),
            "rows_sha256": rows_sha256,
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "inputs": {
            "candidate_manifest": str(Path(candidate_manifest).resolve()),
            "candidate_manifest_sha256": file_digest(candidate_manifest),
            "track_cache": str(Path(track_cache).resolve()),
            "track_cache_sha256": file_digest(track_cache),
            "track_manifest": str(Path(track_manifest).resolve()),
            "track_manifest_sha256": file_digest(track_manifest),
        },
    }


def write_smoke_manifest(payload: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = list(payload["rows"])
    with (output / "manifest.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    summary = {key: value for key, value in payload.items() if key != "rows"}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sealed R10B smoke manifest.")
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--track-cache", required=True, type=Path)
    parser.add_argument("--track-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_DISCOVERY_FAMILIES))
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--max-total", type=int)
    parser.add_argument("--seed", type=int, default=260108847)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_smoke_manifest(
        candidate_manifest=args.candidate_manifest,
        track_cache=args.track_cache,
        track_manifest=args.track_manifest,
        families=args.families,
        per_family=args.per_family,
        max_total=args.max_total,
        seed=args.seed,
    )
    write_smoke_manifest(payload, args.output_dir)
    print(
        canonical_json(
            {
                "output_dir": str(args.output_dir.resolve()),
                **payload["summary"],
            }
        )
    )


if __name__ == "__main__":
    main()
