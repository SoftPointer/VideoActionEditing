#!/usr/bin/env python3
"""Exact5 analytic linear frontier for the self-generated action anchor.

This is a burned-development diagnostic, not training and not a video editor.
For every frozen v2 outer fold, orthogonal projections are fitted only from
that fold's ``model_fit`` original anchor clips.  Only after fitting are the
corresponding OOF originals and their deterministic temporal variants passed
to the evaluator.  Family metadata is used only for the final evaluation
bootstrap.  There are no learned heads, losses, optimizers, whitening, source
subtraction, or derived fit rows.

The primary per-IID statistic for negative ``n`` is

    d(R(C(anchor)), R(C(n(anchor))))
      - d(R(C(anchor)), R(C(monotone_warp(anchor))))

where every distance is a sum of squared orthogonal-code differences divided
by 32*768.  Thus the identity teacher and every compressed candidate use the
same upstream geometry and normalization; this is not self-reconstruction.
Frame-PCA, clip-PCA and Tucker codes are compared only at identical actual
per-clip payloads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from methods.bernini_action_editing import semantic_anchor_action_sequence_vae_v2 as v2


authority = v2.authority
SCHEMA = "semantic-anchor-linear-frontier-exact5-receipt-v4-fast"
SEED = 20260819
TIME_STEPS = 32
FEATURE_DIM = 768
FULL_NUMEL = TIME_STEPS * FEATURE_DIM
OUTER_FOLDS = 5
PAYLOAD_BUDGETS = (32, 64, 128, 256, 384)
TUCKER_TIME_RANK = 4
NEGATIVES = ("reverse", "block_shuffle", "phase_swap")
TEACHER_RETENTION = 0.8
WARP_COORDINATES = (
    0.000000000, 1.241935484, 2.467741935, 3.677419355,
    4.870967742, 6.048387097, 7.209677419, 8.354838710,
    9.483870968, 10.596774194, 11.693548387, 12.774193548,
    13.838709677, 14.887096774, 15.919354839, 16.935483871,
    17.935483871, 18.919354839, 19.887096774, 20.838709677,
    21.774193548, 22.693548387, 23.596774194, 24.483870968,
    25.354838710, 26.209677419, 27.048387097, 27.870967742,
    28.677419355, 29.467741935, 30.241935484, 31.000000000,
)
PINNED_WARP_COORDINATES_SHA256 = (
    "e10f29d0b495c5e36297ed76306e795a3193aa2b4d2fc179518b8c658ad94009"
)
PHASE_BLOCK_PERMUTATION = (0, 1, 4, 5, 2, 3, 6, 7)
IDENTITY_BLOCK_PERMUTATION = tuple(range(8))
REVERSE_BLOCK_PERMUTATION = tuple(reversed(range(8)))
V2_OUTER_ASSIGNMENT_DIGEST = (
    "5ab9704f456768b440c966a53328de0c1a67836548f8f8ebd92e50d21846ab5f"
)
V2_FOLD_IID_DIGESTS = {
    0: "26b5cb90aea6140c8719ae48c2b98082a999d1ca79489ec5bdc70e6ce6745773",
    1: "18c7ad8a24f678ea93cc9d16365fcba0cb8d101667eed9542618240f3ed9c13f",
    2: "b1a85b86390bb773e23125f55f1a49152edf3c426de5ebe2e519aa421c3b430b",
    3: "b2abd43da040c878ac0620022e7fb4c5a8c967580dc6615ced7a6dec62404d3d",
    4: "473f906f5874ddc36227c77ccdc79ec80fa6fe55692f65adf12c049891e74fcf",
}


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    payload_budgets: tuple[int, ...] = PAYLOAD_BUDGETS
    tucker_time_rank: int = TUCKER_TIME_RANK
    block_count: int = 8
    teacher_retention: float = TEACHER_RETENTION
    bootstrap_draws: int = 10000
    bootstrap_alpha: float = 0.05

    def validate(self) -> None:
        if self != Config():
            raise ValueError("v4-fast analytic frontier configuration is immutable")
        for payload in self.payload_budgets:
            if payload % TIME_STEPS or payload % self.tucker_time_rank:
                raise ValueError("payload is not exactly realizable by all candidates")
            if payload > 384:
                raise ValueError("clip rank exceeds the preregistered fold-safe bound")
        if TIME_STEPS % self.block_count:
            raise ValueError("block count does not divide the sequence")


@dataclass(frozen=True)
class FrontierFit:
    frame_mean: torch.Tensor
    frame_basis: torch.Tensor
    clip_mean: torch.Tensor
    clip_basis: torch.Tensor
    temporal_basis: torch.Tensor
    content_basis: torch.Tensor
    fit_iid_digest: str
    fit_input_sha256: str
    diagnostics: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _object_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous().clone()
    digest = hashlib.sha256()
    digest.update(_canonical_json({
        "dtype": str(tensor.dtype), "shape": list(tensor.shape),
    }))
    digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    return authority.file_sha256(path)


def _binding() -> dict[str, str]:
    implementation = Path(__file__).resolve(strict=True)
    common = Path(v2.__file__).resolve(strict=True)
    feature_authority = Path(authority.__file__).resolve(strict=True)
    return {
        "implementation_path": str(implementation),
        "implementation_sha256": _file_sha(implementation),
        "v2_split_authority_path": str(common),
        "v2_split_authority_sha256": _file_sha(common),
        "exact1288_feature_authority_path": str(feature_authority),
        "exact1288_feature_authority_sha256": _file_sha(feature_authority),
    }


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("implementation or authority changed during execution")


def _write_json_create_only(path: Path, value: Any) -> str:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("output must be a fresh absolute JSON child")
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return hashlib.sha256(raw).hexdigest()


def canonical_action(value: torch.Tensor) -> torch.Tensor:
    """The sole upstream action canonicalization: temporal centering."""

    tensor = value.detach().to(dtype=torch.float32)
    if tuple(tensor.shape) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("ordered teacher sequence geometry differs")
    result = (tensor - tensor.mean(dim=0, keepdim=True)).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("canonical action is non-finite")
    return result


def _warp_coordinate_tensor() -> torch.Tensor:
    coordinates = torch.tensor(WARP_COORDINATES, dtype=torch.float32)
    if (
        tuple(coordinates.shape) != (TIME_STEPS,)
        or float(coordinates[0]) != 0.0
        or float(coordinates[-1]) != float(TIME_STEPS - 1)
        or not bool((coordinates[1:] > coordinates[:-1]).all())
        or bool(torch.equal(
            coordinates, torch.arange(TIME_STEPS, dtype=torch.float32)
        ))
        or _tensor_sha(coordinates) != PINNED_WARP_COORDINATES_SHA256
    ):
        raise RuntimeError("v4-A warp coordinate ABI differs")
    return coordinates


def _monotone_warp(value: torch.Tensor) -> torch.Tensor:
    sequence = canonical_action(value)
    positions = _warp_coordinate_tensor().to(device=sequence.device)
    lower = positions.floor().to(torch.long)
    upper = positions.ceil().to(torch.long)
    weight = (positions - lower.to(positions.dtype)).unsqueeze(1)
    warped = sequence.index_select(0, lower) * (1.0 - weight)
    warped = warped + sequence.index_select(0, upper) * weight
    return canonical_action(warped)


def _block_permutation(iid: str, seed: int, block_count: int) -> torch.Tensor:
    if type(iid) is not str or not iid or block_count != 8:
        raise ValueError("v4-A block-shuffle IID/count differs")
    ordered = sorted(
        range(8),
        key=lambda block: hashlib.sha256(
            f"v4a-block-shuffle:{seed}:{iid}:{block}".encode("utf-8")
        ).hexdigest(),
    )
    forbidden = {
        IDENTITY_BLOCK_PERMUTATION,
        REVERSE_BLOCK_PERMUTATION,
        PHASE_BLOCK_PERMUTATION,
    }
    candidate = tuple(ordered)
    if candidate in forbidden or all(
        candidate[index] < candidate[index + 1] for index in range(7)
    ):
        candidate = tuple(ordered[2:] + ordered[:2])
    if candidate in forbidden or set(candidate) != set(range(8)) or all(
        candidate[index] < candidate[index + 1] for index in range(7)
    ):
        candidate = (2, 0, 5, 1, 7, 3, 6, 4)
    if candidate in forbidden or set(candidate) != set(range(8)):
        raise RuntimeError("v4-A block-shuffle permutation ABI differs")
    return torch.tensor(candidate, dtype=torch.long)


def _expand_block_permutation(permutation: Sequence[int]) -> torch.Tensor:
    if tuple(sorted(permutation)) != tuple(range(8)):
        raise ValueError("v4-A block permutation differs")
    return torch.tensor([
        4 * block + offset for block in permutation for offset in range(4)
    ], dtype=torch.long)


def temporal_variants(
    raw_anchor: torch.Tensor, iid: str, config: Config
) -> dict[str, torch.Tensor]:
    """Create one order-preserving positive and three non-independent negatives."""

    if tuple(raw_anchor.shape) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("variant input geometry differs")
    device = raw_anchor.device
    query = canonical_action(raw_anchor)
    block_size = TIME_STEPS // config.block_count
    block_order = _block_permutation(iid, config.seed, config.block_count).to(device)
    blocks = query.reshape(config.block_count, block_size, FEATURE_DIM)
    return {
        "original": query,
        "monotone_warp": _monotone_warp(query),
        "reverse": canonical_action(query.flip(0)),
        "block_shuffle": canonical_action(
            blocks[block_order].reshape_as(query)
        ),
        "phase_swap": canonical_action(query.index_select(
            0, _expand_block_permutation(PHASE_BLOCK_PERMUTATION).to(device)
        )),
    }


def _transform_authority(iids: Sequence[str], config: Config) -> dict[str, Any]:
    if len(iids) != 644 or len(set(iids)) != 644:
        raise ValueError("transform authority requires exact644 IIDs")
    monotone_positions = _warp_coordinate_tensor().contiguous()
    phase_indices = _expand_block_permutation(PHASE_BLOCK_PERMUTATION)
    phase_block_map = list(PHASE_BLOCK_PERMUTATION)
    block_by_iid = {
        iid: _block_permutation(iid, config.seed, config.block_count).tolist()
        for iid in iids
    }
    if any(order == phase_block_map for order in block_by_iid.values()):
        raise ValueError("block shuffle collides with phase swap")
    block_tensor = torch.tensor(
        [block_by_iid[iid] for iid in iids], dtype=torch.long
    )
    return {
        "monotone_warp": {
            "canonical_name": "monotone_speed_warp",
            "internal_key_is_canonical_abi_alias": "monotone_warp",
            "definition": "temporally center, linearly interpolate at pinned continuous coordinates, then independently temporal center",
            "coordinates": [float(value) for value in monotone_positions],
            "coordinates_tensor_sha256": _tensor_sha(monotone_positions),
            "pinned_coordinates_tensor_sha256": PINNED_WARP_COORDINATES_SHA256,
            "strictly_monotone": bool(torch.all(
                monotone_positions[1:] > monotone_positions[:-1]
            )),
            "endpoint_preserving": bool(
                monotone_positions[0] == 0
                and monotone_positions[-1] == TIME_STEPS - 1
            ),
        },
        "reverse": {
            "frame_index_map": list(reversed(range(TIME_STEPS))),
            "frame_index_map_tensor_sha256": _tensor_sha(
                torch.arange(TIME_STEPS - 1, -1, -1, dtype=torch.long)
            ),
        },
        "phase_swap": {
            "definition": "temporally center, permute 8x4 blocks by pinned quarter-phase map, then independently temporal center",
            "frame_index_map": phase_indices.tolist(),
            "frame_index_map_tensor_sha256": _tensor_sha(phase_indices),
            "block_index_map_at_8x4": phase_block_map,
        },
        "block_shuffle": {
            "definition": "temporally center, order 8x4 blocks by sha256(v4a-block-shuffle:seed:iid:block), apply rotate-by-2/fixed fallback, then independently temporal center",
            "iid_order_digest": _object_sha(list(iids)),
            "per_iid_block_index_maps": block_by_iid,
            "per_iid_block_index_maps_digest": _object_sha(block_by_iid),
            "per_iid_block_index_tensor_sha256": _tensor_sha(block_tensor),
            "tensor_shape": list(block_tensor.shape),
            "excludes_identity_reverse_and_phase_swap_maps": True,
        },
    }


def _canonicalize_basis_sign(basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2 or basis.shape[1] == 0:
        raise ValueError("basis geometry differs")
    locations = basis.abs().argmax(dim=0)
    columns = torch.arange(basis.shape[1], device=basis.device)
    signs = torch.sign(basis[locations, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (basis * signs.unsqueeze(0)).contiguous()


def _orthogonality_error(basis: torch.Tensor) -> float:
    gram = basis.T @ basis
    identity = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    return float((gram - identity).abs().max().detach().cpu())


def _top_eigenbasis(covariance: torch.Tensor, rank: int) -> torch.Tensor:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance geometry differs")
    if not 1 <= rank <= covariance.shape[0]:
        raise ValueError("eigenbasis rank differs")
    _, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -rank:].flip(1).contiguous()
    basis = _canonicalize_basis_sign(basis)
    if _orthogonality_error(basis) > 3.0e-4:
        raise ValueError("eigenbasis is not orthogonal")
    return basis


def _fit_clip_basis(centered: torch.Tensor, rank: int) -> torch.Tensor:
    """Dual PCA followed by QR; no variance whitening is applied."""

    if centered.ndim != 2 or rank >= centered.shape[0]:
        raise ValueError("clip PCA needs rank below original fit count")
    gram = centered @ centered.T
    values, vectors = torch.linalg.eigh(gram)
    values = values[-rank:].flip(0)
    left = vectors[:, -rank:].flip(1)
    threshold = max(float(values.max().detach().cpu()) * 1.0e-10, 1.0e-12)
    if float(values.min().detach().cpu()) <= threshold:
        raise ValueError("clip PCA requested a numerically null direction")
    raw = (centered.T @ left) / values.sqrt().unsqueeze(0)
    basis = torch.linalg.qr(raw, mode="reduced").Q
    basis = _canonicalize_basis_sign(basis)
    if _orthogonality_error(basis) > 3.0e-4:
        raise ValueError("clip basis is not orthogonal")
    return basis


def _fit_frontier(
    fit_rows: Sequence[authority.PairRecord], config: Config, device: torch.device
) -> FrontierFit:
    """Fit only original anchors; source, family and derived variants are absent."""

    if len(fit_rows) <= max(config.payload_budgets):
        raise ValueError("model-fit original count is too small for clip frontier")
    fit_iids = [row.iid for row in fit_rows]
    if len(set(fit_iids)) != len(fit_iids):
        raise ValueError("model-fit IIDs are not unique")
    values = torch.stack([
        canonical_action(row.anchor_sequence) for row in fit_rows
    ]).to(device=device)
    if tuple(values.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("model-fit tensor geometry differs")

    max_frame_rank = max(config.payload_budgets) // TIME_STEPS
    max_clip_rank = max(config.payload_budgets)
    max_content_rank = max(config.payload_budgets) // config.tucker_time_rank

    tokens = values.reshape(-1, FEATURE_DIM)
    frame_mean = tokens.mean(dim=0, keepdim=True)
    token_centered = tokens - frame_mean
    feature_covariance = token_centered.T @ token_centered
    content_basis = _top_eigenbasis(feature_covariance, max_content_rank)
    frame_basis = content_basis[:, :max_frame_rank].contiguous()

    flat = values.flatten(1)
    clip_mean = flat.mean(dim=0, keepdim=True)
    clip_basis = _fit_clip_basis(flat - clip_mean, max_clip_rank)

    temporal_covariance = torch.einsum("ntd,nsd->ts", values, values)
    temporal_basis = _top_eigenbasis(
        temporal_covariance, config.tucker_time_rank
    )
    diagnostics = {
        "frame_basis_shape": list(frame_basis.shape),
        "frame_basis_sha256": _tensor_sha(frame_basis),
        "frame_basis_max_orthogonality_error": _orthogonality_error(frame_basis),
        "clip_basis_shape": list(clip_basis.shape),
        "clip_basis_sha256": _tensor_sha(clip_basis),
        "clip_basis_max_orthogonality_error": _orthogonality_error(clip_basis),
        "tucker_temporal_basis_shape": list(temporal_basis.shape),
        "tucker_temporal_basis_sha256": _tensor_sha(temporal_basis),
        "tucker_temporal_max_orthogonality_error": _orthogonality_error(temporal_basis),
        "tucker_content_basis_shape": list(content_basis.shape),
        "tucker_content_basis_sha256": _tensor_sha(content_basis),
        "tucker_content_max_orthogonality_error": _orthogonality_error(content_basis),
        "orthogonal_projection": True,
        "variance_whitening": False,
    }
    return FrontierFit(
        frame_mean=frame_mean.contiguous(),
        frame_basis=frame_basis,
        clip_mean=clip_mean.contiguous(),
        clip_basis=clip_basis,
        temporal_basis=temporal_basis,
        content_basis=content_basis,
        fit_iid_digest=_object_sha(fit_iids),
        fit_input_sha256=_tensor_sha(values),
        diagnostics=diagnostics,
    )


def candidate_specs(config: Config) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for payload in config.payload_budgets:
        candidates = (
            ("frame_pca", TIME_STEPS, payload // TIME_STEPS),
            ("clip_pca", 1, payload),
            ("tucker", config.tucker_time_rank, payload // config.tucker_time_rank),
        )
        for kind, temporal_rank, feature_or_clip_rank in candidates:
            name = (
                f"{kind}_b{payload:04d}_t{temporal_rank:02d}"
                f"_r{feature_or_clip_rank:03d}"
            )
            specs.append({
                "name": name,
                "kind": kind,
                "payload_numel": payload,
                "temporal_rank": temporal_rank,
                "feature_or_clip_rank": feature_or_clip_rank,
                "compression_ratio_vs_teacher": FULL_NUMEL / payload,
            })
    return specs


def _encode(
    value: torch.Tensor, spec: Mapping[str, Any], fitted: FrontierFit
) -> torch.Tensor:
    kind = spec["kind"]
    rank = int(spec["feature_or_clip_rank"])
    if kind == "frame_pca":
        result = ((value - fitted.frame_mean) @ fitted.frame_basis[:, :rank]).flatten()
    elif kind == "clip_pca":
        centered = value.flatten().unsqueeze(0) - fitted.clip_mean
        result = (centered @ fitted.clip_basis[:, :rank]).flatten()
    elif kind == "tucker":
        centered = value - fitted.frame_mean
        result = (
            fitted.temporal_basis.T @ centered @ fitted.content_basis[:, :rank]
        ).flatten()
    else:
        raise ValueError("candidate kind differs")
    if result.numel() != int(spec["payload_numel"]):
        raise ValueError("candidate actual code payload differs")
    return result


def normalized_squared_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("paired code geometry differs")
    value = (left - right).square().sum() / FULL_NUMEL
    if not bool(torch.isfinite(value)):
        raise ValueError("distance is non-finite")
    return float(value.detach().cpu())


def distance_margin(
    query: torch.Tensor, monotone: torch.Tensor, negative: torch.Tensor
) -> dict[str, float]:
    """Same-query, same-representation margin; never self-reconstruction."""

    positive_distance = normalized_squared_distance(query, monotone)
    negative_distance = normalized_squared_distance(query, negative)
    return {
        "monotone_distance": positive_distance,
        "negative_distance": negative_distance,
        "margin": negative_distance - positive_distance,
    }


def _evaluate_fold(
    oof_rows: Sequence[authority.PairRecord], fitted: FrontierFit,
    config: Config, device: torch.device,
) -> list[dict[str, Any]]:
    """Read OOF values only after the caller has completed projection fitting."""

    specs = candidate_specs(config)
    output: list[dict[str, Any]] = []
    for row in oof_rows:
        variants = temporal_variants(row.anchor_sequence.to(device), row.iid, config)
        teacher_query = variants["original"].flatten()
        teacher_positive = variants["monotone_warp"].flatten()
        teacher = {
            negative: distance_margin(
                teacher_query, teacher_positive, variants[negative].flatten()
            )
            for negative in NEGATIVES
        }
        candidate_rows: dict[str, Any] = {}
        for spec in specs:
            query = _encode(variants["original"], spec, fitted)
            positive = _encode(variants["monotone_warp"], spec, fitted)
            if query.numel() != spec["payload_numel"] or positive.shape != query.shape:
                raise ValueError("candidate actual code payload differs")
            candidate_rows[spec["name"]] = {
                negative: distance_margin(
                    query, positive, _encode(variants[negative], spec, fitted)
                )
                for negative in NEGATIVES
            }
        source_distance = normalized_squared_distance(
            teacher_query,
            canonical_action(
                canonical_action(row.source_sequence.to(device))
            ).flatten(),
        )
        output.append({
            "iid": row.iid,
            "family": row.family,
            "teacher": teacher,
            "candidates": candidate_rows,
            "paired_source_teacher_distance_diagnostic_only": source_distance,
        })
    return output


def _bootstrap_seed(config: Config, *parts: str) -> int:
    payload = ":".join((str(config.seed), *parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**63 - 1)


def _paired_bootstrap_lcbs(
    values: Sequence[float], families: Sequence[str], config: Config,
    seed_label: str,
) -> dict[str, Any]:
    if len(values) != 644 or len(families) != 644:
        raise ValueError("bootstrap requires exact644 paired originals")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("bootstrap values are non-finite")
    family_names = sorted(set(families))
    if len(family_names) != 28:
        raise ValueError("family bootstrap requires exact28 clusters")
    tensor = torch.tensor(values, dtype=torch.float64)

    clip_seed = _bootstrap_seed(config, seed_label, "clip")
    clip_generator = torch.Generator().manual_seed(clip_seed)
    clip_indices = torch.randint(
        len(tensor), (config.bootstrap_draws, len(tensor)), generator=clip_generator
    )
    clip_means = tensor[clip_indices].mean(dim=1)

    family_sums = torch.tensor([
        sum(float(value) for value, family in zip(values, families) if family == name)
        for name in family_names
    ], dtype=torch.float64)
    family_counts = torch.tensor([
        sum(family == name for family in families) for name in family_names
    ], dtype=torch.float64)
    family_seed = _bootstrap_seed(config, seed_label, "family")
    family_generator = torch.Generator().manual_seed(family_seed)
    family_indices = torch.randint(
        len(family_names),
        (config.bootstrap_draws, len(family_names)),
        generator=family_generator,
    )
    per_family_means = family_sums / family_counts
    family_bootstrap_means = per_family_means[family_indices].mean(dim=1)
    return {
        "paired_original_count": 644,
        "clip_micro_point_mean": float(tensor.mean()),
        "family_macro_point_mean": float(per_family_means.mean()),
        "clip_paired_bootstrap": {
            "draws": config.bootstrap_draws,
            "seed": clip_seed,
            "one_sided_alpha": config.bootstrap_alpha,
            "lcb": float(torch.quantile(clip_means, config.bootstrap_alpha)),
        },
        "family_cluster_paired_bootstrap": {
            "cluster_count": 28,
            "draws": config.bootstrap_draws,
            "seed": family_seed,
            "one_sided_alpha": config.bootstrap_alpha,
            "cluster_resampling": "compute_each_family_clip_mean_then_resample_28_family_means_with_equal_weight",
            "equal_family_weight": True,
            "lcb": float(torch.quantile(
                family_bootstrap_means, config.bootstrap_alpha
            )),
        },
    }


def _strictly_positive_both(value: Mapping[str, Any]) -> bool:
    return bool(
        value["clip_paired_bootstrap"]["lcb"] > 0.0
        and value["family_cluster_paired_bootstrap"]["lcb"] > 0.0
    )


def _negative_gate(
    teacher: Mapping[str, Any], candidate: Mapping[str, Any],
    retention: Mapping[str, Any],
) -> bool:
    return bool(
        _strictly_positive_both(teacher)
        and _strictly_positive_both(candidate)
        and _strictly_positive_both(retention)
    )


def _aggregate(
    rows: Sequence[Mapping[str, Any]], config: Config
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) != 644 or len({row["iid"] for row in rows}) != 644:
        raise ValueError("OOF aggregation is not exact644 once each")
    families = [str(row["family"]) for row in rows]
    specs = candidate_specs(config)
    teacher: dict[str, Any] = {}
    for negative in NEGATIVES:
        values = [float(row["teacher"][negative]["margin"]) for row in rows]
        stats = _paired_bootstrap_lcbs(
            values, families, config, f"teacher:{negative}:margin"
        )
        stats["both_lcbs_strictly_gt_zero"] = _strictly_positive_both(stats)
        teacher[negative] = stats

    candidates: dict[str, Any] = {}
    for spec in specs:
        name = spec["name"]
        negatives: dict[str, Any] = {}
        for negative in NEGATIVES:
            teacher_values = [
                float(row["teacher"][negative]["margin"]) for row in rows
            ]
            candidate_values = [
                float(row["candidates"][name][negative]["margin"]) for row in rows
            ]
            retention_values = [
                candidate - config.teacher_retention * reference
                for candidate, reference in zip(candidate_values, teacher_values)
            ]
            candidate_stats = _paired_bootstrap_lcbs(
                candidate_values, families, config, f"{name}:{negative}:margin"
            )
            retention_stats = _paired_bootstrap_lcbs(
                retention_values, families, config,
                f"{name}:{negative}:minus:{config.teacher_retention}:teacher",
            )
            candidate_stats["both_lcbs_strictly_gt_zero"] = (
                _strictly_positive_both(candidate_stats)
            )
            retention_stats["both_lcbs_strictly_gt_zero"] = (
                _strictly_positive_both(retention_stats)
            )
            negatives[negative] = {
                "teacher_margin": teacher[negative],
                "candidate_margin": candidate_stats,
                "candidate_minus_0p8_teacher_margin": retention_stats,
                "negative_gate": _negative_gate(
                    teacher[negative], candidate_stats, retention_stats
                ),
            }
        candidates[name] = {
            "spec": spec,
            "negative_results": negatives,
            "temporal_mechanics_gate": all(
                negatives[negative]["negative_gate"] for negative in NEGATIVES
            ),
        }

    frontier: dict[str, Any] = {}
    for payload in config.payload_budgets:
        names = [
            spec["name"] for spec in specs if spec["payload_numel"] == payload
        ]
        if len(names) != 3 or any(
            candidates[name]["spec"]["payload_numel"] != payload for name in names
        ):
            raise ValueError("same-payload frontier closure differs")
        frontier[str(payload)] = {
            "payload_numel": payload,
            "candidate_names": names,
            "exact_same_actual_code_numel": True,
            "qualified_candidates": [
                name for name in names if candidates[name]["temporal_mechanics_gate"]
            ],
            "oof_winner_selected": False,
            "candidates_listed_without_ranking": True,
            "across_negative_compensation_used": False,
        }
    return {
        "uncompressed_teacher": teacher,
        "candidates": candidates,
    }, frontier


def _compact_evidence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Embed exactly the paired fields needed to independently recompute gates."""

    output = []
    for row in rows:
        output.append({
            "iid": row["iid"],
            "family": row["family"],
            "outer_fold": row["outer_fold"],
            "teacher_margin_by_negative": {
                negative: float(row["teacher"][negative]["margin"])
                for negative in NEGATIVES
            },
            "candidate_margin_by_name_and_negative": {
                name: {
                    negative: float(values[negative]["margin"])
                    for negative in NEGATIVES
                }
                for name, values in row["candidates"].items()
            },
            "paired_source_teacher_distance_diagnostic_only": float(
                row["paired_source_teacher_distance_diagnostic_only"]
            ),
        })
    return output


def _run_fold(
    pairs: Sequence[authority.PairRecord], fold_index: int,
    config: Config, device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if device.type != "cpu":
        raise ValueError("v4-fast analytic frontier is CPU-only")
    groups, split = v2._split_fold(pairs, fold_index, config.seed)
    if (
        split["outer_assignment_digest"] != V2_OUTER_ASSIGNMENT_DIGEST
        or split["iid_digest"] != V2_FOLD_IID_DIGESTS[fold_index]
    ):
        raise ValueError("split is not bit-identical to frozen v2 exact5")
    fit_iids = [row.iid for row in groups["model_fit"]]
    oof_iids = [row.iid for row in groups["exploratory_oof"]]
    if (
        set(fit_iids) & set(oof_iids)
        or _object_sha(fit_iids) != split["model_fit_iid_digest"]
        or _object_sha(oof_iids) != split["exploratory_oof_iid_digest"]
    ):
        raise ValueError("model-fit/OOF IID closure differs")

    # This order is contractual: the OOF evaluator is not called until all
    # analytic projections for this fold have been fitted and validated.
    fitted = _fit_frontier(groups["model_fit"], config, device)
    if fitted.fit_iid_digest != split["model_fit_iid_digest"]:
        raise ValueError("projection fit IID digest differs")
    evaluation = _evaluate_fold(
        groups["exploratory_oof"], fitted, config, device
    )
    for row in evaluation:
        row["outer_fold"] = fold_index
    if [row["iid"] for row in evaluation] != oof_iids:
        raise ValueError("OOF evaluation IID order differs")
    fold_receipt = {
        "fold_index": fold_index,
        "frozen_v2_fold_iid_digest": split["iid_digest"],
        "frozen_v2_outer_assignment_digest": split["outer_assignment_digest"],
        "model_fit_original_count": len(fit_iids),
        "model_fit_iid_digest": _object_sha(fit_iids),
        "projection_fit_iid_digest": fitted.fit_iid_digest,
        "projection_fit_input_sha256": fitted.fit_input_sha256,
        "early_stop_validation_original_count": len(groups["early_stop_validation"]),
        "early_stop_validation_iid_digest": split[
            "early_stop_validation_iid_digest"
        ],
        "early_stop_validation_values_used": False,
        "oof_original_count": len(oof_iids),
        "oof_iid_digest": _object_sha(oof_iids),
        "oof_evaluation_iid_digest": split["exploratory_oof_iid_digest"],
        "fit_oof_iid_disjoint": True,
        "projection_fit_completed_before_oof_value_evaluation": True,
        "frozen_v2_partition_and_energy_metadata_are_not_projection_inputs": True,
        "oof_feature_values_read_by_projection_fit": False,
        "projection_fit_original_anchor_rows_only": True,
        "projection_fit_source_rows": 0,
        "projection_fit_derived_rows": 0,
        "projection_fit_family_or_transform_labels": False,
        "projection_fit_optimizer_steps": 0,
        "projection_diagnostics": fitted.diagnostics,
        "oof_evaluation_rows_sha256": _object_sha(evaluation),
    }
    return fold_receipt, evaluation


def run_exact5(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    config = Config()
    config.validate()
    device = torch.device("cpu")
    pairs, feature_receipt = authority.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    population = v2._exact644_population_authority(pairs)
    if len(pairs) != 644 or len({row.iid for row in pairs}) != 644:
        raise ValueError("feature authority is not exact644 paired originals")
    if len({row.family for row in pairs}) != 28:
        raise ValueError("feature authority is not exact28 families")
    exact_iids = [row.iid for row in pairs]
    transform_authority = _transform_authority(exact_iids, config)

    fold_receipts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for fold_index in range(OUTER_FOLDS):
        fold_receipt, rows = _run_fold(
            pairs, fold_index, config, device
        )
        fold_receipts.append(fold_receipt)
        all_rows.extend(rows)
    if sorted(row["iid"] for row in all_rows) != exact_iids:
        raise ValueError("exact5 OOF union is not exact644 once each")
    metrics, frontier = _aggregate(all_rows, config)
    compact_evidence = _compact_evidence(all_rows)
    source_values = [
        float(row["paired_source_teacher_distance_diagnostic_only"])
        for row in all_rows
    ]
    qualified = sorted(
        name for name, value in metrics["candidates"].items()
        if value["temporal_mechanics_gate"]
    )
    config_value = {
        "seed": config.seed,
        "payload_budgets": list(config.payload_budgets),
        "tucker_time_rank": config.tucker_time_rank,
        "block_count": config.block_count,
        "teacher_retention": config.teacher_retention,
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_alpha": config.bootstrap_alpha,
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "V4_FAST_EXACT5_LINEAR_FRONTIER_COMPLETE_BURNED_DEVELOPMENT",
        "implementation": run_binding,
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "device": "cpu",
        "cpu_analytic_only": True,
        "feature_authority": {
            "feature_root": str(Path(args.feature_root).resolve(strict=True)),
            "feature_receipt_sha256": args.expected_feature_receipt_sha256,
            "feature_receipt_digest": feature_receipt["receipt_digest"],
            "total_feature_records": 1288,
            "unique_original_base_clips": 644,
            "action_anchor_records": 644,
            "paired_source_diagnostic_records": 644,
            "family_count": 28,
            "derived_rows_in_sample_count": 0,
            "population_authority": population,
            "exact644_ordered_iid_digest": _object_sha(exact_iids),
        },
        "frozen_split": {
            "source": "semantic_anchor_action_sequence_vae_v2._split_fold",
            "outer_assignment_digest": V2_OUTER_ASSIGNMENT_DIGEST,
            "fold_iid_digests": V2_FOLD_IID_DIGESTS,
            "all_exact644_are_development": True,
            "fresh_scientific_confirmation_claimed": False,
            "iid_disjoint_only_not_actor_scene_generator_lineage_disjoint": True,
        },
        "metric_contract": {
            "canonicalization": "C(x)=x-mean_time(x), shape [32,768]",
            "teacher_mapping": "R_infinity(C(x))=flatten(C(x))",
            "candidate_mapping": "orthogonal code coordinates fitted on fold model_fit originals only",
            "formula": "d(R(C(anchor)),R(C(negative)))-d(R(C(anchor)),R(C(monotone_speed_warp(anchor))))",
            "same_query_and_same_mapping_within_each_margin": True,
            "self_reconstruction_metric_used": False,
            "distance": "sum_squared_code_difference/(32*768)",
            "negative_transforms": list(NEGATIVES),
            "positive_transform": "monotone_speed_warp with pinned continuous-coordinate ABI",
            "transform_authority": transform_authority,
            "derived_transforms_are_non_independent_and_not_samples": True,
            "derived_evaluations_per_original": 4,
            "original_sample_count": 644,
        },
        "projection_contract": {
            "families": ["frame_pca", "clip_pca", "tucker"],
            "analytic_no_optimizer": True,
            "orthogonal": True,
            "variance_whitening": False,
            "fit_original_anchor_only": True,
            "source_used_for_projection_or_temporal_compensation": False,
            "family_or_transform_labels_used_for_projection": False,
            "early_stop_validation_used": False,
            "oof_used_for_projection_or_selection": False,
            "payload_definition": "actual per-clip code tensor numel",
            "same_payload_comparisons_only": True,
            "cross_payload_ranking_performed": False,
            "oof_winner_selected": False,
            "across_negative_compensation_used": False,
            "uncompressed_teacher_payload_numel": FULL_NUMEL,
        },
        "oof_access_contract": {
            "exact1288_feature_authority_materialized_at_command_start": True,
            "frozen_v2_partition_computation_reads_exact644_anchor_energy_metadata": True,
            "separate_cryptographic_late_open_artifact_used": False,
            "same_process_fit_then_evaluate": True,
            "oof_tensors_supplied_to_projection_fit_function": False,
            "oof_transform_and_metric_evaluator_called_only_after_fold_fit_returns": True,
            "claim_is_no_oof_projection_fit_not_no_prior_feature_materialization": True,
        },
        "folds": fold_receipts,
        "oof_closure": {
            "unique_original_iids": 644,
            "each_original_evaluated_exactly_once": True,
            "oof_ordered_iid_digest": _object_sha([row["iid"] for row in all_rows]),
            "oof_sorted_iid_digest": _object_sha(sorted(row["iid"] for row in all_rows)),
            "per_iid_evaluation_sha256": _object_sha(all_rows),
            "embedded_paired_margin_evidence_count": len(compact_evidence),
            "embedded_paired_margin_evidence_sha256": _object_sha(compact_evidence),
            "embedded_paired_margin_evidence": compact_evidence,
            "embedded_fields_sufficient_to_recompute_all_bootstrap_gates": True,
        },
        "paired_source_diagnostic": {
            "single_column": "paired_source_teacher_distance_diagnostic_only",
            "definition": "d(identity(C(anchor)),identity(C(paired_source)))",
            "used_for_temporal_margin_or_candidate_fit": False,
            "count": 644,
            "mean": float(torch.tensor(source_values, dtype=torch.float64).mean()),
            "values_sha256": _object_sha(source_values),
        },
        "bootstrap_contract": {
            "clip_paired_resampling_unit": "original_IID",
            "family_cluster_paired_resampling_unit": "exact28_equal_weight_family_means",
            "family_point_estimate": "macro mean of 28 per-family clip means",
            "paired_within_IID": True,
            "strict_gate": "for every negative, teacher margin, candidate margin, and candidate-0.8*teacher margin both LCBs > 0",
        },
        "metrics": metrics,
        "same_payload_frontier": frontier,
        "qualified_temporal_mechanics_candidates": qualified,
        "qualification_scope": {
            "temporal_mechanics_qualified_candidates": qualified,
            "training_authorized": False,
            "action_representation_qualified": False,
            "identity_preservation_qualified": False,
            "vae_necessary": None,
            "generation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    output = Path(args.output)
    receipt_sha = _write_json_create_only(output, receipt)
    _assert_binding_unchanged(run_binding)
    return {
        "receipt": str(output.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "qualified_temporal_mechanics_candidates": qualified,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact5 no-optimizer orthogonal action-anchor frontier"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-exact5")
    run.add_argument("--feature-root", required=True)
    run.add_argument("--expected-feature-receipt-sha256", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=run_exact5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
