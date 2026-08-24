"""Prompt-to-action representation distilled from motion geometry teachers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ACTION_REPR_SCHEMA = "motive-prompt-action-repr-v1"
PROMPT_HASH_VERSION = "lucy-blake2b-ngram-v1"


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for action representation") from error
    return torch


def _nn() -> Any:
    return _torch().nn


def prompt_hash_features(
    prompts: Sequence[str],
    *,
    feature_dim: int = 512,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Exact signed n-gram hash used by Lucy's LearnedComponentRouter."""

    torch = _torch()
    dtype = dtype or torch.float32
    features = torch.zeros(
        (len(prompts), feature_dim),
        device=device,
        dtype=dtype,
    )
    for row, prompt in enumerate(prompts):
        tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'_-]*", prompt.lower())
        grams: list[str] = []
        grams.extend(tokens)
        grams.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
        grams.extend(
            f"{a} {b} {c}"
            for a, b, c in zip(tokens, tokens[1:], tokens[2:])
        )
        if not grams:
            grams = ["<empty>"]
        scale = float(len(grams)) ** -0.5
        for gram in grams:
            digest = hashlib.blake2b(
                gram.encode("utf-8"),
                digest_size=8,
            ).digest()
            value = int.from_bytes(digest, "little", signed=False)
            index = value % feature_dim
            sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
            features[row, index] += sign * scale
        features[row] = features[row] / features[row].norm(p=2).clamp_min(1e-6)
    return features


class PromptActionEncoder(_nn().Module):
    """Architecture-compatible pretraining trunk for Lucy's prompt router."""

    def __init__(self, input_dim: int = 512, action_dim: int = 128) -> None:
        super().__init__()
        nn = _nn()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.input = nn.Linear(self.input_dim, self.action_dim)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(self.action_dim)

    def forward(self, features: Any, *, normalize: bool = True) -> Any:
        torch = _torch()
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [B,{self.input_dim}], got "
                f"{tuple(features.shape)}"
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("prompt features contain non-finite values")
        encoded = self.norm(self.activation(self.input(features)))
        return (
            torch.nn.functional.normalize(encoded.float(), dim=-1).to(encoded.dtype)
            if normalize
            else encoded
        )

    def encode_prompts(self, prompts: Sequence[str], *, normalize: bool = True) -> Any:
        parameter = next(self.parameters())
        features = prompt_hash_features(
            prompts,
            feature_dim=self.input_dim,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.forward(features, normalize=normalize)


@dataclass(frozen=True)
class TeacherTransform:
    raw_mean: list[float]
    raw_scale: list[float]
    pca_components: list[list[float]]
    pca_scale: list[float]
    output_dim: int
    camera_dims_excluded: int = 8
    raw_scale_floor: float = 0.0
    whitening_ridge: float = 0.0
    minimum_relative_variance: float = 0.0
    retained_variance_ratio: float = 1.0
    minimum_transform_energy: float = 1e-4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fit(
        cls,
        raw_features: np.ndarray,
        *,
        output_dim: int = 128,
        camera_dims_excluded: int = 8,
        eps: float = 1e-6,
        raw_scale_floor_fraction: float = 0.01,
        whitening_ridge_fraction: float = 0.05,
        minimum_relative_variance: float = 1e-4,
    ) -> "TeacherTransform":
        values = np.asarray(raw_features, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("raw teacher features require shape [N,D] with N >= 2")
        if values.shape[1] == 0:
            raise ValueError("raw teacher features must have at least one dimension")
        if not np.isfinite(values).all():
            raise ValueError("raw teacher features contain non-finite values")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if raw_scale_floor_fraction < 0.0:
            raise ValueError(
                "raw_scale_floor_fraction must be non-negative"
            )
        if whitening_ridge_fraction < 0.0:
            raise ValueError("whitening_ridge_fraction must be non-negative")
        if not 0.0 <= minimum_relative_variance < 1.0:
            raise ValueError(
                "minimum_relative_variance must be in [0,1)"
            )
        mean = np.mean(values, axis=0)
        raw_standard_deviation = np.std(values, axis=0)
        maximum_raw_standard_deviation = float(
            np.max(raw_standard_deviation)
        )
        if maximum_raw_standard_deviation <= eps:
            raise ValueError(
                "raw teacher features have no stable train-split variance"
            )
        raw_scale_floor = max(
            eps,
            raw_scale_floor_fraction
            * maximum_raw_standard_deviation,
        )
        scale = np.maximum(raw_standard_deviation, raw_scale_floor)
        standardized = (values - mean) / scale
        _, singular_values, right = np.linalg.svd(
            standardized,
            full_matrices=False,
        )
        component_variance = (
            singular_values**2 / max(len(values) - 1, 1)
        )
        maximum_variance = (
            float(component_variance[0])
            if len(component_variance)
            else 0.0
        )
        if maximum_variance <= eps**2:
            raise ValueError(
                "raw teacher features have no stable train-split variance"
            )
        variance_floor = maximum_variance * minimum_relative_variance
        stable_rank = int(
            np.count_nonzero(
                component_variance
                >= max(variance_floor, eps**2)
            )
        )
        rank = max(1, min(output_dim, right.shape[0], stable_rank))
        components = right[:rank]
        ridge = max(
            eps,
            whitening_ridge_fraction
            * np.sqrt(max(maximum_variance, 0.0)),
        )
        component_scale = np.sqrt(component_variance[:rank] + ridge**2)
        total_variance = float(np.sum(component_variance))
        retained_variance_ratio = (
            float(np.sum(component_variance[:rank]) / total_variance)
            if total_variance > eps**2
            else 0.0
        )
        return cls(
            raw_mean=mean.astype(np.float32).tolist(),
            raw_scale=scale.astype(np.float32).tolist(),
            pca_components=components.astype(np.float32).tolist(),
            pca_scale=component_scale.astype(np.float32).tolist(),
            output_dim=int(output_dim),
            camera_dims_excluded=int(camera_dims_excluded),
            raw_scale_floor=float(raw_scale_floor),
            whitening_ridge=float(ridge),
            minimum_relative_variance=float(minimum_relative_variance),
            retained_variance_ratio=retained_variance_ratio,
            minimum_transform_energy=1e-4,
        )

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_features, dtype=np.float32)
        mean = np.asarray(self.raw_mean, dtype=np.float32)
        scale = np.asarray(self.raw_scale, dtype=np.float32)
        components = np.asarray(self.pca_components, dtype=np.float32)
        pca_scale = np.asarray(self.pca_scale, dtype=np.float32)
        if (
            mean.ndim != 1
            or len(mean) == 0
            or scale.shape != mean.shape
            or components.ndim != 2
            or components.shape[1] != len(mean)
            or pca_scale.shape != (components.shape[0],)
            or components.shape[0] == 0
        ):
            raise ValueError("teacher transform arrays have inconsistent shapes")
        if not all(
            np.isfinite(item).all()
            for item in (mean, scale, components, pca_scale)
        ):
            raise ValueError("teacher transform contains non-finite values")
        if np.any(scale <= 0.0) or np.any(pca_scale <= 0.0):
            raise ValueError("teacher transform scales must be positive")
        if values.ndim != 2 or values.shape[-1] != len(mean):
            raise ValueError("teacher raw feature dimension mismatch")
        projected = ((values - mean) / scale) @ components.T
        projected = projected / pca_scale
        if projected.shape[1] < self.output_dim:
            projected = np.pad(
                projected,
                ((0, 0), (0, self.output_dim - projected.shape[1])),
            )
        norm = np.linalg.norm(projected, axis=1, keepdims=True)
        minimum_energy = max(float(self.minimum_transform_energy), 1e-8)
        return np.divide(
            projected,
            np.maximum(norm, minimum_energy),
            out=np.zeros_like(projected, dtype=np.float32),
            where=norm >= minimum_energy,
        ).astype(np.float32)


def build_raw_action_teacher(
    delta_descriptors: np.ndarray,
    scalar_features: np.ndarray,
    *,
    camera_dims: int = 8,
) -> np.ndarray:
    """Drop camera coordinates, renormalize actor delta, then append amplitudes."""

    descriptors = np.asarray(delta_descriptors, dtype=np.float32)
    scalars = np.asarray(scalar_features, dtype=np.float32)
    if descriptors.ndim != 2 or scalars.ndim != 2:
        raise ValueError("descriptors and scalar_features must be matrices")
    if len(descriptors) != len(scalars):
        raise ValueError("teacher descriptor/scalar length mismatch")
    if camera_dims <= 0 or descriptors.shape[1] <= camera_dims:
        raise ValueError("invalid camera_dims")
    actor = descriptors[:, :-camera_dims]
    norm = np.linalg.norm(actor, axis=1, keepdims=True)
    actor = np.divide(
        actor,
        np.maximum(norm, 1e-8),
        out=np.zeros_like(actor),
        where=norm > 1e-8,
    )
    return np.concatenate((actor, scalars), axis=1).astype(np.float32)


def _state_digest(state_dict: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def save_action_checkpoint(
    path: str | Path,
    *,
    model: PromptActionEncoder,
    teacher_transform: TeacherTransform,
    provenance: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    torch = _torch()
    state = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    metadata = {
        "schema_version": ACTION_REPR_SCHEMA,
        "prompt_hash_version": PROMPT_HASH_VERSION,
        "input_dim": model.input_dim,
        "action_dim": model.action_dim,
        "teacher_transform": teacher_transform.to_dict(),
        "provenance": provenance,
        "metrics": metrics,
        "state_digest": _state_digest(state),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sidecar = path.with_suffix(path.suffix + ".json")
    temporary_sidecar = sidecar.with_name(
        f".{sidecar.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save({"metadata": metadata, "state_dict": state}, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary_sidecar.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with temporary_sidecar.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
    finally:
        for incomplete in (temporary, temporary_sidecar):
            if incomplete.exists():
                incomplete.unlink()
    return metadata


def load_action_checkpoint(
    path: str | Path,
    *,
    expected_upstream_digest: str | None = None,
    map_location: str = "cpu",
) -> tuple[PromptActionEncoder, dict[str, Any]]:
    torch = _torch()
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    metadata = payload.get("metadata")
    state = payload.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("invalid action checkpoint")
    if metadata.get("schema_version") != ACTION_REPR_SCHEMA:
        raise ValueError("incompatible action checkpoint schema")
    if metadata.get("prompt_hash_version") != PROMPT_HASH_VERSION:
        raise ValueError("incompatible prompt hash version")
    if _state_digest(state) != metadata.get("state_digest"):
        raise ValueError("action checkpoint state digest mismatch")
    if expected_upstream_digest is not None:
        actual = metadata.get("provenance", {}).get("upstream_digest")
        if actual != expected_upstream_digest:
            raise ValueError(
                f"upstream digest mismatch: expected {expected_upstream_digest}, "
                f"got {actual}"
            )
    model = PromptActionEncoder(
        input_dim=int(metadata["input_dim"]),
        action_dim=int(metadata["action_dim"]),
    )
    model.load_state_dict(state, strict=True)
    return model, metadata
