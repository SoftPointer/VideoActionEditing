"""Fail-closed feature archives with compatibility provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "videoedit-motive-feature-v1"
GRADIENT_REQUIRED_PROVENANCE = {
    "checkpoint_digest",
    "parameter_digest",
    "objective",
    "timestep_or_sigma",
    "noise_digest",
    "vae_posterior_digest",
    "projection_backend",
    "projection_seed",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_feature_metadata(
    *,
    feature_kind: str,
    dimension: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build metadata whose digest defines whether two archives may be compared."""

    if not feature_kind:
        raise ValueError("feature_kind must be non-empty")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("provenance must be a non-empty dictionary")
    if feature_kind == "gradient_fingerprint":
        missing = sorted(GRADIENT_REQUIRED_PROVENANCE - set(provenance))
        if missing:
            raise ValueError(
                "gradient fingerprint provenance is missing: "
                + ", ".join(missing)
            )
    compatibility = {
        "schema_version": SCHEMA_VERSION,
        "feature_kind": feature_kind,
        "dimension": int(dimension),
        "provenance": provenance,
    }
    digest = hashlib.sha256(_canonical_json(compatibility).encode("utf-8")).hexdigest()
    return {**compatibility, "compatibility_digest": digest}


def save_feature_archive(
    path: str | Path,
    *,
    features: np.ndarray,
    ids: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    matrix = np.asarray(features, dtype=np.float32)
    identifiers = np.asarray(ids).astype(str)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("features must have shape [N,D]")
    if len(matrix) != len(identifiers):
        raise ValueError("feature/id length mismatch")
    if int(metadata.get("dimension", -1)) != matrix.shape[1]:
        raise ValueError("metadata dimension does not match features")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=matrix,
                ids=identifiers,
                metadata_json=np.asarray(_canonical_json(metadata)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_feature_archive(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "ids", "metadata_json"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"{path} is missing {missing}; legacy/unprovenanced archives "
                "are intentionally rejected"
            )
        features = np.asarray(archive["features"], dtype=np.float32)
        ids = np.asarray(archive["ids"]).astype(str)
        metadata_raw = np.asarray(archive["metadata_json"])
        if metadata_raw.ndim != 0:
            raise ValueError(f"{path} metadata_json must be a scalar string")
        metadata = json.loads(str(metadata_raw.item()))
    if features.ndim != 2 or len(features) != len(ids):
        raise ValueError(f"invalid feature/id arrays in {path}")
    provenance = metadata.get("provenance")
    rebuilt = build_feature_metadata(
        feature_kind=str(metadata.get("feature_kind", "")),
        dimension=int(metadata.get("dimension", -1)),
        provenance=provenance,
    )
    if rebuilt != metadata:
        raise ValueError(f"{path} metadata or compatibility digest is invalid")
    if features.shape[1] != metadata["dimension"]:
        raise ValueError(f"{path} feature dimension disagrees with metadata")
    return features, ids, metadata


def assert_archives_compatible(
    candidate_metadata: dict[str, Any],
    query_metadata: dict[str, Any],
) -> None:
    if (
        candidate_metadata.get("compatibility_digest")
        != query_metadata.get("compatibility_digest")
    ):
        raise ValueError(
            "candidate/query archives are incompatible: feature kind or "
            "checkpoint/parameter/loss/randomness/projection provenance differs"
        )
