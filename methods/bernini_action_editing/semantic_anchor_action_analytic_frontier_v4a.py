#!/usr/bin/env python3
"""Exact644 analytic action-sequence compression frontier v4-A.

This is an independent, burned-development experiment.  It reuses the frozen
v3 exact-five-fold *assignment contract*, but it imports no v3 code and reads
no v3 artifact or OOF result.  Every representation is an original-fit-only,
unwhitened orthogonal projection of the literal target
``temporal_center(anchor ordered DINO)[32,768]``.

The primary test is temporal mechanics, not self-reconstruction.  A fixed
query payload is compared with one monotone speed-warp positive and three
anchor-derived hard negatives.  The uncompressed representation must pass
first; compressed representations must retain a preregistered fraction of its
signed distance margin.  Raw reconstruction MSE is secondary.  Source DINO is
used only for a separately reported source/no-op separation diagnostic and
cannot affect the temporal hard gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from methods.bernini_action_editing import semantic_anchor_action_sequence_vae_v2 as v2


authority = v2.authority
OUTER_FOLDS = 5
TIME_STEPS = 32
CHANNELS = 768
RAW_SCALAR_COUNT = TIME_STEPS * CHANNELS
FRAME_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)
TEMPORAL_RANKS = (2, 4, 8, 16)
TUCKER_CHANNEL_RANKS = (8, 16, 32, 64)
CLIP_RANKS = (16, 32, 64, 128, 256)
TEMPORAL_NEGATIVES = ("reverse", "block_shuffle", "phase_swap")
SOURCE_DIAGNOSTIC = "source_noop"
ALL_VIEWS = ("monotone_speed_warp", *TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)

# Explicit float literals are the transform ABI.  They are continuous sampling
# coordinates, not rounded integer indices.  Endpoints are fixed, all interior
# increments are strictly positive, and the map is not identity.
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

# Copied as contract constants from the sealed v3 preregistration.  The runtime
# verifies values recomputed by v2; it does not import or consume v3.
V3_FROZEN_OUTER_ASSIGNMENT_DIGEST = (
    "5ab9704f456768b440c966a53328de0c1a67836548f8f8ebd92e50d21846ab5f"
)
V3_FROZEN_FOLD_IID_DIGESTS = {
    0: "26b5cb90aea6140c8719ae48c2b98082a999d1ca79489ec5bdc70e6ce6745773",
    1: "18c7ad8a24f678ea93cc9d16365fcba0cb8d101667eed9542618240f3ed9c13f",
    2: "b1a85b86390bb773e23125f55f1a49152edf3c426de5ebe2e519aa421c3b430b",
    3: "b2abd43da040c878ac0620022e7fb4c5a8c967580dc6615ced7a6dec62404d3d",
    4: "473f906f5874ddc36227c77ccdc79ec80fa6fe55692f65adf12c049891e74fcf",
}

FIT_SCHEMA = "anchor-action-analytic-frontier-fit-bundle-v4a"
EVAL_SCHEMA = "anchor-action-analytic-frontier-eval-bundle-v4a"
PREPARE_SCHEMA = "anchor-action-analytic-frontier-prepare-receipt-v4a"
FOLD_SCHEMA = "anchor-action-analytic-frontier-fold-receipt-v4a"
AGGREGATE_SCHEMA = "anchor-action-analytic-frontier-aggregate-receipt-v4a"

DEVELOPMENT_FIELDS = {
    "prior_locked_partition_rows_burned": 96,
    "exact644_role": "BURNED_DEVELOPMENT_ONLY",
    "fresh_confirmation_requires_new_external_group_disjoint_data": True,
    "confirmation_evaluations_allowed_by_this_runtime": 0,
    "scientific_confirmation_claimed": False,
}

PREPARE_RECEIPT_KEYS = frozenset({
    "schema_version", "status", "unique_original_base_clips",
    "model_fit_original_rows", "early_stop_validation_original_rows",
    "exploratory_oof_original_rows", "anchor_derived_diagnostic_rows_this_fold",
    "observed_paired_source_diagnostic_rows_this_fold",
    "derived_or_diagnostic_rows_are_independent_samples",
    "derived_rows_consumed_by_fit", "labels_heads_losses_optimizers",
    "target", "source_usage",
    "family_labels_used_only_for_frozen_split_and_evaluation_bootstrap",
    "observed_oof_family_count", "family_or_transform_labels_consumed_by_fit",
    "v3_frozen_split_digest_contract_reused", "v3_runtime_imported",
    "v3_oof_result_consumed", "fold", "config", "config_sha256",
    "feature_receipt_sha256", "feature_receipt_digest", "exact644_iid_digest",
    "exact644_raw_target_sha256", "exact644_source_noop_sha256",
    "exact644_population_authority", "global_rms", "global_rms_sha256",
    "states_semantic_sha256", "candidate_manifest",
    "candidate_manifest_sha256", "payload_tiers", "transform_abi",
    "diagnostic_view_semantic_sha256", "fit_bundle",
    "exploratory_oof_bundle", "implementation", "receipt_digest",
}) | frozenset(DEVELOPMENT_FIELDS)

FOLD_RECEIPT_KEYS = frozenset({
    "schema_version", "status", "fold_index", "fold", "sample_accounting",
    "primary_margin_definition", "distance_definition",
    "same_query_payload_and_same_metric_for_teacher_and_candidates",
    "uncompressed_teacher_is_identity_flatten",
    "teacher_evaluated_before_compressed_qualification", "source_noop",
    "temporal_margin_bootstrap_by_negative", "teacher_hard_gate",
    "fold_candidate_gates_diagnostic_only", "per_iid_margins",
    "family_by_iid", "raw_reconstruction_metrics_secondary",
    "raw_mse_bootstrap_secondary", "raw_mse_used_for_primary_gate",
    "candidate_manifest", "candidate_manifest_sha256", "payload_tiers",
    "no_fold_winner_or_rank_selection", "config", "config_sha256",
    "feature_receipt_sha256", "feature_receipt_digest", "exact644_iid_digest",
    "exact644_raw_target_sha256", "exact644_source_noop_sha256",
    "exact644_population_authority", "prepare_receipt_sha256",
    "fit_bundle_sha256", "exploratory_oof_bundle_sha256",
    "states_semantic_sha256", "diagnostic_view_semantic_sha256",
    "transform_abi", "labels_heads_losses_optimizers_absent",
    "only_temporal_mechanics_tested", "action_representation_qualified",
    "source_identity_preservation_tested", "video_editing_tested",
    "prior_generation_qualified", "renderer_authorized",
    "inference_authorized", "vae_necessary", "implementation",
    "receipt_digest",
}) | frozenset(DEVELOPMENT_FIELDS)


@dataclass(frozen=True)
class Config:
    seed: int = 20260819
    margin_retention_floor: float = 0.80
    bootstrap_draws: int = 10000
    frame_ranks: tuple[int, ...] = FRAME_RANKS
    temporal_ranks: tuple[int, ...] = TEMPORAL_RANKS
    tucker_channel_ranks: tuple[int, ...] = TUCKER_CHANNEL_RANKS
    clip_ranks: tuple[int, ...] = CLIP_RANKS

    def validate(self) -> None:
        if self != Config():
            raise ValueError("v4-A analytic frontier is exact-preregistered and immutable")


def _object_sha(value: Any) -> str:
    return v2._object_sha(value)


def _tensor_sha(value: torch.Tensor) -> str:
    return v2._tensor_sha(value)


def _file_sha(path: Path) -> str:
    return v2._file_sha(path)


def _sha(value: Any, name: str) -> str:
    return v2._sha(value, name)


def _binding() -> dict[str, str]:
    implementation = Path(__file__).resolve(strict=True)
    common = Path(v2.__file__).resolve(strict=True)
    feature_authority = Path(authority.__file__).resolve(strict=True)
    return {
        "implementation_path": str(implementation),
        "implementation_sha256": _file_sha(implementation),
        "v2_split_common_path": str(common),
        "v2_split_common_sha256": _file_sha(common),
        "feature_authority_path": str(feature_authority),
        "feature_authority_sha256": _file_sha(feature_authority),
        "v3_runtime_imported": False,
        "v3_artifact_consumed": False,
    }


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("v4-A implementation/dependencies changed during command")


def _fresh_output(value: str) -> Path:
    return v2._fresh_output(value)


def _save_torch(path: Path, value: Any) -> str:
    return v2._save_torch_create_only(path, value)


def _write_json(path: Path, value: Any) -> str:
    return v2._write_json_create_only(path, value)


def _load_torch(path: Path, expected_sha: str, expected_size: int | None = None) -> Any:
    return v2._load_torch(path, expected_sha, expected_size)


def _load_receipt(path: Path, expected_sha: str, schema: str) -> dict[str, Any]:
    return v2._load_receipt(path, expected_sha, schema)


def anchor_action_target(pair: authority.PairRecord) -> torch.Tensor:
    """Literal anchor target; source cannot enter this function."""

    return v2.anchor_action_target(pair)


def _temporal_center(value: torch.Tensor) -> torch.Tensor:
    if value.ndim not in (2, 3) or value.shape[-2:] != (TIME_STEPS, CHANNELS):
        raise ValueError("v4-A sequence geometry differs")
    result = (value - value.mean(dim=-2, keepdim=True)).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("v4-A temporal-centered sequence is non-finite")
    return result


def source_noop_control(pair: authority.PairRecord) -> torch.Tensor:
    """Observed paired source, used only by the ineligible noop diagnostic."""

    source = pair.source_sequence.detach().to(dtype=torch.float32, device="cpu")
    if tuple(source.shape) != (TIME_STEPS, CHANNELS):
        raise ValueError("v4-A source/noop geometry differs")
    return _temporal_center(source)


def _warp_coordinate_tensor() -> torch.Tensor:
    coordinates = torch.tensor(WARP_COORDINATES, dtype=torch.float32)
    if (
        tuple(coordinates.shape) != (TIME_STEPS,)
        or float(coordinates[0]) != 0.0
        or float(coordinates[-1]) != float(TIME_STEPS - 1)
        or not bool((coordinates[1:] > coordinates[:-1]).all())
        or bool(torch.equal(coordinates, torch.arange(TIME_STEPS, dtype=torch.float32)))
        or _tensor_sha(coordinates) != PINNED_WARP_COORDINATES_SHA256
    ):
        raise RuntimeError("v4-A warp coordinate ABI differs")
    return coordinates


def monotone_speed_warp(value: torch.Tensor) -> torch.Tensor:
    """Endpoint-fixed strict continuous-coordinate linear interpolation."""

    sequence = _temporal_center(value)
    coordinates = _warp_coordinate_tensor().to(sequence.device)
    lower = coordinates.floor().to(torch.long)
    upper = coordinates.ceil().to(torch.long)
    weight = (coordinates - lower.to(coordinates.dtype)).reshape(
        *((1,) * (sequence.ndim - 2)), TIME_STEPS, 1
    )
    warped = sequence.index_select(-2, lower) * (1.0 - weight)
    warped = warped + sequence.index_select(-2, upper) * weight
    return _temporal_center(warped)


def _block_permutation(iid: str, seed: int) -> tuple[int, ...]:
    if type(iid) is not str or not iid:
        raise ValueError("v4-A block-shuffle IID differs")
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
    return candidate


def _expand_block_permutation(permutation: Sequence[int]) -> torch.Tensor:
    if tuple(sorted(permutation)) != tuple(range(8)):
        raise ValueError("v4-A block permutation differs")
    return torch.tensor(
        [4 * block + offset for block in permutation for offset in range(4)],
        dtype=torch.long,
    )


def _diagnostic_views(
    query: torch.Tensor,
    source_noop: torch.Tensor,
    iids: Sequence[str],
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    query = _temporal_center(query)
    source_noop = _temporal_center(source_noop)
    if query.ndim != 3 or tuple(source_noop.shape) != tuple(query.shape):
        raise ValueError("v4-A diagnostic-view batch geometry differs")
    if len(iids) != len(query) or len(set(iids)) != len(iids):
        raise ValueError("v4-A diagnostic-view IID closure differs")
    reverse = query.flip(1)
    phase_indices = _expand_block_permutation(PHASE_BLOCK_PERMUTATION)
    phase_swap = query.index_select(1, phase_indices)
    permutations = [_block_permutation(iid, seed) for iid in iids]
    shuffled = torch.stack([
        query[index].index_select(0, _expand_block_permutation(permutation))
        for index, permutation in enumerate(permutations)
    ])
    views = {
        "monotone_speed_warp": monotone_speed_warp(query),
        "reverse": _temporal_center(reverse),
        "block_shuffle": _temporal_center(shuffled),
        "phase_swap": _temporal_center(phase_swap),
        SOURCE_DIAGNOSTIC: _temporal_center(source_noop),
    }
    if set(views) != set(ALL_VIEWS):
        raise RuntimeError("v4-A diagnostic-view closure differs")
    for name, tensor in views.items():
        if tuple(tensor.shape) != tuple(query.shape):
            raise RuntimeError(f"v4-A {name} geometry differs")
        if float(tensor.mean(dim=1).abs().max()) > 3.0e-6:
            raise RuntimeError(f"v4-A {name} was not independently centered")
    abi = {
        "positive": "monotone_speed_warp",
        "temporal_hard_negatives": list(TEMPORAL_NEGATIVES),
        "source_diagnostic": SOURCE_DIAGNOSTIC,
        "source_diagnostic_eligible_for_temporal_mechanics_gate": False,
        "warp_coordinates": list(WARP_COORDINATES),
        "warp_coordinates_sha256": _tensor_sha(_warp_coordinate_tensor()),
        "warp_interpolation": "linear_continuous_coordinates",
        "warp_endpoints_fixed": True,
        "warp_coordinates_strictly_increasing": True,
        "reverse_index_map": list(reversed(range(TIME_STEPS))),
        "phase_block_permutation": list(PHASE_BLOCK_PERMUTATION),
        "block_shuffle_algorithm": "sha256(seed,iid,block)-ordered 8x4 blocks; deterministic fallback; intra-block order fixed",
        "block_shuffle_permutation_by_iid_sha256": _object_sha({
            iid: list(permutation) for iid, permutation in zip(iids, permutations)
        }),
        "every_view_temporal_centered_after_transform": True,
    }
    return views, abi


def _fit_orthogonal_states(fit: torch.Tensor, config: Config) -> dict[str, Any]:
    """Fit nested bases from original model-fit rows only; never derived rows."""

    if fit.ndim != 3 or tuple(fit.shape[1:]) != (TIME_STEPS, CHANNELS):
        raise ValueError("v4-A fit tensor geometry differs")
    if len(fit) < 2 or not bool(torch.isfinite(fit).all()):
        raise ValueError("v4-A fit tensor differs")
    frame = v2._fit_frame_pca(fit, max(config.frame_ranks))
    sequence_mean = fit.mean(dim=0, keepdim=True).contiguous()
    centered = fit - sequence_mean
    temporal_rows = centered.permute(0, 2, 1).reshape(-1, TIME_STEPS)
    temporal_covariance = temporal_rows.T @ temporal_rows / max(len(temporal_rows) - 1, 1)
    _, temporal_vectors = torch.linalg.eigh(temporal_covariance)
    temporal_basis = temporal_vectors[:, -max(config.temporal_ranks):].flip(1).contiguous()
    channel_rows = centered.reshape(-1, CHANNELS)
    channel_covariance = channel_rows.T @ channel_rows / max(len(channel_rows) - 1, 1)
    _, channel_vectors = torch.linalg.eigh(channel_covariance)
    channel_basis = channel_vectors[:, -max(config.tucker_channel_ranks):].flip(1).contiguous()
    clip_rank_limit = min(len(fit) - 1, fit[0].numel())
    available_clip_ranks = tuple(rank for rank in config.clip_ranks if rank <= clip_rank_limit)
    if not available_clip_ranks:
        raise ValueError("v4-A fit rank cannot support any preregistered clip PCA")
    clip = v2._fit_clip_pca(fit, max(available_clip_ranks))
    states = {
        "frame": frame,
        "tucker": {
            "mean": sequence_mean,
            "temporal_basis": temporal_basis,
            "channel_basis": channel_basis,
        },
        "clip": clip,
        "clip_fit_rank_limit": clip_rank_limit,
        "available_clip_ranks": available_clip_ranks,
        "omitted_clip_ranks": tuple(
            rank for rank in config.clip_ranks if rank > clip_rank_limit
        ),
    }
    _validate_states(states, len(fit), config)
    return states


def _orthonormal(value: torch.Tensor) -> bool:
    gram = value.T @ value
    return bool(torch.allclose(
        gram, torch.eye(value.shape[1], dtype=value.dtype, device=value.device),
        atol=8.0e-4, rtol=8.0e-4,
    ))


def _validate_states(states: Any, fit_count: int, config: Config) -> None:
    if type(states) is not dict or set(states) != {
        "frame", "tucker", "clip", "clip_fit_rank_limit",
        "available_clip_ranks", "omitted_clip_ranks",
    }:
        raise ValueError("v4-A state key closure differs")
    frame, tucker, clip = states["frame"], states["tucker"], states["clip"]
    if type(frame) is not dict or set(frame) != {"mean", "basis"}:
        raise ValueError("v4-A frame state differs")
    if type(clip) is not dict or set(clip) != {"mean", "basis"}:
        raise ValueError("v4-A clip state differs")
    if type(tucker) is not dict or set(tucker) != {
        "mean", "temporal_basis", "channel_basis"
    }:
        raise ValueError("v4-A Tucker state differs")
    expected_limit = min(fit_count - 1, RAW_SCALAR_COUNT)
    available = tuple(rank for rank in config.clip_ranks if rank <= expected_limit)
    omitted = tuple(rank for rank in config.clip_ranks if rank > expected_limit)
    if (
        states["clip_fit_rank_limit"] != expected_limit
        or tuple(states["available_clip_ranks"]) != available
        or tuple(states["omitted_clip_ranks"]) != omitted
        or tuple(frame["mean"].shape) != (1, CHANNELS)
        or tuple(frame["basis"].shape) != (CHANNELS, max(config.frame_ranks))
        or tuple(tucker["mean"].shape) != (1, TIME_STEPS, CHANNELS)
        or tuple(tucker["temporal_basis"].shape) != (TIME_STEPS, max(config.temporal_ranks))
        or tuple(tucker["channel_basis"].shape) != (CHANNELS, max(config.tucker_channel_ranks))
        or tuple(clip["mean"].shape) != (1, RAW_SCALAR_COUNT)
        or tuple(clip["basis"].shape) != (RAW_SCALAR_COUNT, max(available))
    ):
        raise ValueError("v4-A state geometry/rank authority differs")
    tensors = (
        frame["mean"], frame["basis"], tucker["mean"],
        tucker["temporal_basis"], tucker["channel_basis"],
        clip["mean"], clip["basis"],
    )
    if not all(type(value) is torch.Tensor and bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("v4-A state tensor differs")
    if not all(_orthonormal(value) for value in (
        frame["basis"], tucker["temporal_basis"],
        tucker["channel_basis"], clip["basis"],
    )):
        raise ValueError("v4-A projection basis is not orthonormal")


def _state_semantic_sha(states: Mapping[str, Any]) -> str:
    return _object_sha({
        "frame_mean": _tensor_sha(states["frame"]["mean"]),
        "frame_basis": _tensor_sha(states["frame"]["basis"]),
        "tucker_mean": _tensor_sha(states["tucker"]["mean"]),
        "tucker_temporal_basis": _tensor_sha(states["tucker"]["temporal_basis"]),
        "tucker_channel_basis": _tensor_sha(states["tucker"]["channel_basis"]),
        "clip_mean": _tensor_sha(states["clip"]["mean"]),
        "clip_basis": _tensor_sha(states["clip"]["basis"]),
        "clip_fit_rank_limit": states["clip_fit_rank_limit"],
        "available_clip_ranks": list(states["available_clip_ranks"]),
        "omitted_clip_ranks": list(states["omitted_clip_ranks"]),
    })


def _candidate_manifest(states: Mapping[str, Any], config: Config) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for rank in config.frame_ranks:
        name = f"frame_pca_r{rank:03d}"
        manifest[name] = {
            "kind": "frame_pca", "frame_channel_rank": rank,
            "payload_scalar_count": TIME_STEPS * rank,
            "payload_shape": [TIME_STEPS, rank], "unwhitened": True,
        }
    for temporal_rank in config.temporal_ranks:
        for channel_rank in config.tucker_channel_ranks:
            budget = temporal_rank * channel_rank
            name = f"tucker_t{temporal_rank:02d}_c{channel_rank:02d}_b{budget:04d}"
            manifest[name] = {
                "kind": "tucker", "temporal_rank": temporal_rank,
                "channel_rank": channel_rank,
                "payload_scalar_count": budget,
                "payload_shape": [temporal_rank, channel_rank],
                "budget_formula": "B=temporal_rank*channel_rank",
                "unwhitened": True,
            }
    for rank in states["available_clip_ranks"]:
        name = f"clip_pca_r{rank:03d}"
        manifest[name] = {
            "kind": "clip_pca", "clip_rank": rank,
            "payload_scalar_count": rank,
            "payload_shape": [rank], "unwhitened": True,
            "fit_rank_limited": True,
        }
    _validate_candidate_manifest(manifest, states, config)
    return manifest


def _validate_candidate_manifest(
    manifest: Any, states: Mapping[str, Any], config: Config
) -> None:
    if type(manifest) is not dict or len(manifest) != (
        len(config.frame_ranks)
        + len(config.temporal_ranks) * len(config.tucker_channel_ranks)
        + len(states["available_clip_ranks"])
    ):
        raise ValueError("v4-A candidate manifest closure differs")
    rebuilt: dict[str, dict[str, Any]] = {}
    # Rebuild without recursively invoking this validator.
    for rank in config.frame_ranks:
        rebuilt[f"frame_pca_r{rank:03d}"] = {
            "kind": "frame_pca", "frame_channel_rank": rank,
            "payload_scalar_count": TIME_STEPS * rank,
            "payload_shape": [TIME_STEPS, rank], "unwhitened": True,
        }
    for temporal_rank in config.temporal_ranks:
        for channel_rank in config.tucker_channel_ranks:
            budget = temporal_rank * channel_rank
            rebuilt[f"tucker_t{temporal_rank:02d}_c{channel_rank:02d}_b{budget:04d}"] = {
                "kind": "tucker", "temporal_rank": temporal_rank,
                "channel_rank": channel_rank,
                "payload_scalar_count": budget,
                "payload_shape": [temporal_rank, channel_rank],
                "budget_formula": "B=temporal_rank*channel_rank",
                "unwhitened": True,
            }
    for rank in states["available_clip_ranks"]:
        rebuilt[f"clip_pca_r{rank:03d}"] = {
            "kind": "clip_pca", "clip_rank": rank,
            "payload_scalar_count": rank,
            "payload_shape": [rank], "unwhitened": True,
            "fit_rank_limited": True,
        }
    if manifest != rebuilt:
        raise ValueError("v4-A candidate payload/rank manifest differs")


def _payload_tiers(manifest: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[str]] = {}
    for name, row in manifest.items():
        shape_count = math.prod(int(value) for value in row["payload_shape"])
        if shape_count != row["payload_scalar_count"]:
            raise ValueError("v4-A payload scalar count differs from code shape")
        grouped.setdefault(shape_count, []).append(name)
    result = {}
    for budget, names in sorted(grouped.items()):
        kinds = sorted({manifest[name]["kind"] for name in names})
        result[str(budget)] = {
            "payload_scalar_count": budget,
            "candidates": sorted(names),
            "structure_kinds": kinds,
            "cross_structure_comparison_allowed": len(kinds) >= 2,
            "cross_structure_status": (
                "EXACT_EQUAL_PAYLOAD_DIAGNOSTIC_ONLY"
                if len(kinds) >= 2 else "ABSTAIN_NO_EQUAL_PAYLOAD_COUNTERPART"
            ),
        }
    return result


def _encode_candidate(
    value: torch.Tensor,
    candidate: Mapping[str, Any],
    states: Mapping[str, Any],
) -> torch.Tensor:
    """Return the literal unwhitened payload; no per-PC scale is allowed."""

    value = _temporal_center(value)
    kind = candidate["kind"]
    if kind == "frame_pca":
        rank = candidate["frame_channel_rank"]
        basis = states["frame"]["basis"][:, :rank]
        code = (value - states["frame"]["mean"]) @ basis
    elif kind == "tucker":
        temporal_rank = candidate["temporal_rank"]
        channel_rank = candidate["channel_rank"]
        temporal = states["tucker"]["temporal_basis"][:, :temporal_rank]
        channel = states["tucker"]["channel_basis"][:, :channel_rank]
        centered = value - states["tucker"]["mean"]
        code = torch.einsum("ta,ntc,cb->nab", temporal, centered, channel)
    elif kind == "clip_pca":
        rank = candidate["clip_rank"]
        basis = states["clip"]["basis"][:, :rank]
        code = (value.flatten(1) - states["clip"]["mean"]) @ basis
    else:
        raise ValueError("v4-A candidate kind differs")
    actual_shape = list(code.shape[1:])
    if (
        actual_shape != candidate["payload_shape"]
        or code[0].numel() != candidate["payload_scalar_count"]
        or not bool(torch.isfinite(code).all())
    ):
        raise ValueError("v4-A encoded payload shape/count differs")
    return code.contiguous()


def _reconstruct_candidate(
    value: torch.Tensor,
    candidate: Mapping[str, Any],
    states: Mapping[str, Any],
) -> torch.Tensor:
    value = _temporal_center(value)
    kind = candidate["kind"]
    if kind == "frame_pca":
        rank = candidate["frame_channel_rank"]
        basis = states["frame"]["basis"][:, :rank]
        reconstruction = ((value - states["frame"]["mean"]) @ basis) @ basis.T
        reconstruction = reconstruction + states["frame"]["mean"]
    elif kind == "tucker":
        temporal_rank = candidate["temporal_rank"]
        channel_rank = candidate["channel_rank"]
        temporal = states["tucker"]["temporal_basis"][:, :temporal_rank]
        channel = states["tucker"]["channel_basis"][:, :channel_rank]
        centered = value - states["tucker"]["mean"]
        code = torch.einsum("ta,ntc,cb->nab", temporal, centered, channel)
        reconstruction = torch.einsum("ta,nab,cb->ntc", temporal, code, channel)
        reconstruction = reconstruction + states["tucker"]["mean"]
    elif kind == "clip_pca":
        rank = candidate["clip_rank"]
        basis = states["clip"]["basis"][:, :rank]
        flat = value.flatten(1)
        reconstruction = ((flat - states["clip"]["mean"]) @ basis) @ basis.T
        reconstruction = (reconstruction + states["clip"]["mean"]).reshape_as(value)
    else:
        raise ValueError("v4-A reconstruction candidate kind differs")
    return _temporal_center(reconstruction)


def _code_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Projected displacement energy in the teacher's original units."""

    if tuple(left.shape) != tuple(right.shape) or left.ndim < 2:
        raise ValueError("v4-A code distance geometry differs")
    distance = (left - right).flatten(1).square().sum(dim=1) / RAW_SCALAR_COUNT
    if not bool(torch.isfinite(distance).all()) or bool((distance < 0.0).any()):
        raise ValueError("v4-A code distance differs")
    return distance


def _teacher_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if tuple(left.shape) != tuple(right.shape) or left.shape[-2:] != (
        TIME_STEPS, CHANNELS
    ):
        raise ValueError("v4-A teacher distance geometry differs")
    return _code_distance(left.flatten(1), right.flatten(1))


def _margin_vectors(
    query: torch.Tensor,
    views: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """m=d(E(query),E(negative))-d(E(query),E(speed-warp positive))."""

    if set(views) != set(ALL_VIEWS):
        raise ValueError("v4-A margin view closure differs")
    # Canonicalize the query and every already-generated view once here, then
    # feed these exact same tensors to the teacher and every candidate.
    query = _temporal_center(query)
    views = {name: _temporal_center(views[name]) for name in ALL_VIEWS}
    positive = views["monotone_speed_warp"]
    teacher_positive_distance = _teacher_distance(query, positive)
    teacher = {
        negative: _teacher_distance(query, views[negative]) - teacher_positive_distance
        for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)
    }
    candidates: dict[str, dict[str, torch.Tensor]] = {}
    for name, specification in manifest.items():
        query_code = _encode_candidate(query, specification, states)
        positive_code = _encode_candidate(positive, specification, states)
        positive_distance = _code_distance(query_code, positive_code)
        candidates[name] = {}
        for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC):
            negative_code = _encode_candidate(views[negative], specification, states)
            candidates[name][negative] = (
                _code_distance(query_code, negative_code) - positive_distance
            )
    return teacher, candidates


def _resample_means(
    values: torch.Tensor,
    draws: int,
    seed: int,
) -> torch.Tensor:
    """Bootstrap row means using one shared draw matrix for all columns."""

    if values.ndim != 2 or not len(values) or draws <= 0:
        raise ValueError("v4-A bootstrap matrix differs")
    values = values.to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("v4-A bootstrap values are non-finite")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(values), (draws, len(values)), generator=generator)
    weights = torch.zeros((draws, len(values)), dtype=torch.float64)
    weights.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))
    return (weights @ values) / len(values)


def _family_means(
    values: torch.Tensor,
    iids: Sequence[str],
    family_by_iid: Mapping[str, str],
    expected_family_count: int,
) -> tuple[torch.Tensor, list[str]]:
    if (
        values.ndim != 2 or len(values) != len(iids)
        or len(set(iids)) != len(iids)
        or set(iids) != set(family_by_iid)
    ):
        raise ValueError("v4-A family bootstrap IID closure differs")
    families = sorted(set(family_by_iid.values()))
    if len(families) != expected_family_count:
        raise ValueError("v4-A family bootstrap requires exact family closure")
    index_by_iid = {iid: index for index, iid in enumerate(iids)}
    rows = []
    for family in families:
        indices = [
            index_by_iid[iid] for iid in iids if family_by_iid[iid] == family
        ]
        if not indices:
            raise ValueError("v4-A family bootstrap contains empty family")
        rows.append(values[indices].to(torch.float64).mean(dim=0))
    return torch.stack(rows), families


def _interval(samples: torch.Tensor) -> list[float]:
    if samples.ndim != 1 or not len(samples) or not bool(torch.isfinite(samples).all()):
        raise ValueError("v4-A bootstrap interval samples differ")
    return [
        float(torch.quantile(samples, 0.025)),
        float(torch.quantile(samples, 0.975)),
    ]


def _margin_bootstrap(
    teacher: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
    iids: Sequence[str],
    family_by_iid: Mapping[str, str],
    seed: int,
    config: Config,
    expected_family_count: int,
) -> dict[str, Any]:
    names = list(candidates)
    if (
        tuple(teacher.shape) != (len(iids),)
        or any(tuple(candidates[name].shape) != (len(iids),) for name in names)
    ):
        raise ValueError("v4-A margin bootstrap vector geometry differs")
    matrix = torch.stack([teacher, *(candidates[name] for name in names)], dim=1)
    clip_samples = _resample_means(matrix, config.bootstrap_draws, seed)
    family_matrix, families = _family_means(
        matrix, iids, family_by_iid, expected_family_count
    )
    family_samples = _resample_means(
        family_matrix, config.bootstrap_draws, seed + 1
    )

    def unit_summary(samples: torch.Tensor, point: torch.Tensor, unit: str) -> dict[str, Any]:
        return {
            "bootstrap_unit": unit,
            "bootstrap_seed": seed if unit == "iid" else seed + 1,
            "draws": config.bootstrap_draws,
            "iid_count": len(iids),
            "family_count": len(families) if unit == "family_cluster" else None,
            "mean_margin": float(point.mean()),
            "margin_95pct_ci": _interval(samples),
        }

    teacher_summary = {
        "clip_bootstrap": unit_summary(
            clip_samples[:, 0], matrix[:, 0], "iid"
        ),
        "family_cluster_bootstrap": unit_summary(
            family_samples[:, 0], family_matrix[:, 0], "family_cluster"
        ),
    }
    candidate_summary: dict[str, Any] = {}
    for column, name in enumerate(names, start=1):
        def candidate_unit(
            samples: torch.Tensor, points: torch.Tensor, unit: str
        ) -> dict[str, Any]:
            margin_samples = samples[:, column]
            retention_samples = margin_samples - (
                config.margin_retention_floor * samples[:, 0]
            )
            point_difference = points[:, column] - (
                config.margin_retention_floor * points[:, 0]
            )
            point_denominator = float(points[:, 0].mean())
            return {
                "bootstrap_unit": unit,
                "bootstrap_seed": seed if unit == "iid" else seed + 1,
                "draws": config.bootstrap_draws,
                "iid_count": len(iids),
                "family_count": len(families) if unit == "family_cluster" else None,
                "mean_margin": float(points[:, column].mean()),
                "margin_95pct_ci": _interval(margin_samples),
                "margin_retention_floor": config.margin_retention_floor,
                "mean_retention_difference": float(point_difference.mean()),
                "retention_difference_95pct_ci": _interval(retention_samples),
                "point_margin_retention_ratio": (
                    float(points[:, column].mean()) / point_denominator
                    if point_denominator > 0.0 else None
                ),
                "retention_bootstrap_is_direct_paired_difference": True,
            }

        candidate_summary[name] = {
            "clip_bootstrap": candidate_unit(clip_samples, matrix, "iid"),
            "family_cluster_bootstrap": candidate_unit(
                family_samples, family_matrix, "family_cluster"
            ),
        }
    return {
        "teacher": teacher_summary,
        "candidates": candidate_summary,
        "same_resample_draws_pair_teacher_and_every_candidate": True,
        "derived_views_are_not_bootstrap_units": True,
        "family_cluster_estimator": "equal mean of 28 resampled family means",
    }


def _teacher_gate(summary_by_negative: Mapping[str, Any]) -> dict[str, Any]:
    by_negative = {}
    for negative in TEMPORAL_NEGATIVES:
        evidence = summary_by_negative[negative]["teacher"]
        clip = evidence["clip_bootstrap"]["margin_95pct_ci"][0] > 0.0
        family = evidence["family_cluster_bootstrap"]["margin_95pct_ci"][0] > 0.0
        by_negative[negative] = {
            "clip_margin_lcb_strict_gt_zero": clip,
            "family_margin_lcb_strict_gt_zero": family,
            "hard_gate": bool(clip and family),
        }
    return {
        "by_temporal_negative": by_negative,
        "all_temporal_negatives_hard_gate": all(
            row["hard_gate"] for row in by_negative.values()
        ),
        "source_noop_excluded": True,
    }


def _candidate_gate(
    candidate: str,
    summary_by_negative: Mapping[str, Any],
    teacher_hard_gate: bool,
) -> dict[str, Any]:
    by_negative = {}
    for negative in TEMPORAL_NEGATIVES:
        evidence = summary_by_negative[negative]["candidates"][candidate]
        clip = evidence["clip_bootstrap"]
        family = evidence["family_cluster_bootstrap"]
        values = {
            "candidate_clip_margin_lcb_strict_gt_zero": (
                clip["margin_95pct_ci"][0] > 0.0
            ),
            "candidate_family_margin_lcb_strict_gt_zero": (
                family["margin_95pct_ci"][0] > 0.0
            ),
            "clip_paired_retention_difference_lcb_strict_gt_zero": (
                clip["retention_difference_95pct_ci"][0] > 0.0
            ),
            "family_paired_retention_difference_lcb_strict_gt_zero": (
                family["retention_difference_95pct_ci"][0] > 0.0
            ),
        }
        values["hard_gate"] = all(values.values())
        by_negative[negative] = values
    return {
        "teacher_hard_gate_prerequisite": teacher_hard_gate,
        "by_temporal_negative": by_negative,
        "all_temporal_negatives_hard_gate": bool(
            teacher_hard_gate
            and all(row["hard_gate"] for row in by_negative.values())
        ),
        "source_noop_excluded": True,
    }


def _family_ratio_secondary(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    family_by_iid: Mapping[str, str],
    seed: int,
    draws: int,
    expected_family_count: int,
) -> dict[str, Any]:
    left = {row["iid"]: float(row["raw_mse"]) for row in candidate}
    right = {row["iid"]: float(row["raw_mse"]) for row in baseline}
    if set(left) != set(right) or set(left) != set(family_by_iid):
        raise ValueError("v4-A secondary family-ratio IID closure differs")
    families = sorted(set(family_by_iid.values()))
    if len(families) != expected_family_count:
        raise ValueError("v4-A secondary family ratio family closure differs")
    family_values = torch.tensor([
        [
            sum(left[iid] for iid in left if family_by_iid[iid] == family)
            / sum(family_by_iid[iid] == family for iid in left),
            sum(right[iid] for iid in right if family_by_iid[iid] == family)
            / sum(family_by_iid[iid] == family for iid in right),
        ]
        for family in families
    ], dtype=torch.float64)
    samples = _resample_means(family_values, draws, seed)
    if bool((samples[:, 1] <= 0.0).any()) or bool((family_values[:, 1] <= 0.0).any()):
        raise ValueError("v4-A secondary family baseline is non-positive")
    ratios = samples[:, 0] / samples[:, 1]
    return {
        "bootstrap_unit": "family_cluster",
        "family_count": expected_family_count,
        "iid_count": len(left),
        "bootstrap_seed": seed,
        "draws": draws,
        "mean_ratio": float(family_values[:, 0].mean() / family_values[:, 1].mean()),
        "ratio_95pct_ci": _interval(ratios),
        "equal_family_mean_estimator": True,
    }


def _view_semantic_hashes(
    query: torch.Tensor, views: Mapping[str, torch.Tensor]
) -> dict[str, str]:
    result = {"query": _tensor_sha(query)}
    result.update({name: _tensor_sha(views[name]) for name in ALL_VIEWS})
    return result


def _make_config(args: argparse.Namespace) -> Config:
    config = Config(seed=args.seed)
    config.validate()
    return config


def _config_from_value(value: Any) -> Config:
    if type(value) is not dict or set(value) != set(asdict(Config())):
        raise ValueError("v4-A config key closure differs")
    converted = dict(value)
    for key in (
        "frame_ranks", "temporal_ranks", "tucker_channel_ranks", "clip_ranks"
    ):
        converted[key] = tuple(converted[key])
    config = Config(**converted)
    config.validate()
    return config


def _analytic_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4-A CUDA mode requires exactly one logical cuda:0")
    if torch.cuda.get_device_name(0) != "AMD Instinct MI210":
        raise RuntimeError("v4-A CUDA device must be AMD Instinct MI210")
    return torch.device("cuda:0")


def _states_to_cpu(states: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame": {
            key: value.detach().cpu().contiguous()
            for key, value in states["frame"].items()
        },
        "tucker": {
            key: value.detach().cpu().contiguous()
            for key, value in states["tucker"].items()
        },
        "clip": {
            key: value.detach().cpu().contiguous()
            for key, value in states["clip"].items()
        },
        "clip_fit_rank_limit": states["clip_fit_rank_limit"],
        "available_clip_ranks": tuple(states["available_clip_ranks"]),
        "omitted_clip_ranks": tuple(states["omitted_clip_ranks"]),
    }


def prepare_fold(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    config = _make_config(args)
    pairs, feature_receipt = authority.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    population = v2._exact644_population_authority(pairs)
    exact_iid_digest = _object_sha([row.iid for row in pairs])
    all_targets = torch.stack([anchor_action_target(row) for row in pairs])
    all_sources = torch.stack([source_noop_control(row) for row in pairs])
    raw_target_sha = _tensor_sha(all_targets)
    source_noop_sha = _tensor_sha(all_sources)
    groups, split = v2._split_fold(pairs, args.fold_index, config.seed)
    if (
        split["outer_assignment_digest"] != V3_FROZEN_OUTER_ASSIGNMENT_DIGEST
        or split["iid_digest"] != V3_FROZEN_FOLD_IID_DIGESTS[args.fold_index]
    ):
        raise ValueError("v4-A split is not bit-identical to frozen v3 exact5")
    rms = v2._global_rms(groups["model_fit"])
    fit = v2._tensor_rows(groups["model_fit"], rms)
    validation_iids = [row.iid for row in groups["early_stop_validation"]]
    oof_rows = groups["exploratory_oof"]
    query = v2._tensor_rows(oof_rows, rms)
    source_noop = (
        torch.stack([source_noop_control(row) for row in oof_rows]) / rms
    ).contiguous()
    views, transform_abi = _diagnostic_views(
        query["value"], source_noop, query["iids"], config.seed
    )
    view_hashes = _view_semantic_hashes(query["value"], views)
    device = _analytic_device(args.device)
    states = _states_to_cpu(_fit_orthogonal_states(fit["value"].to(device), config))
    state_sha = _state_semantic_sha(states)
    manifest = _candidate_manifest(states, config)
    tiers = _payload_tiers(manifest)
    family_by_iid = {row.iid: row.family for row in oof_rows}
    observed_fold_family_count = len(set(family_by_iid.values()))
    if observed_fold_family_count <= 0:
        raise ValueError("v4-A fold OOF family closure is empty")
    config_value = asdict(config)
    common = {
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "fold": split,
        "implementation": run_binding,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_iid_digest": exact_iid_digest,
        "exact644_raw_target_sha256": raw_target_sha,
        "exact644_source_noop_sha256": source_noop_sha,
        "exact644_population_authority": population,
        "global_rms": rms,
        "global_rms_sha256": _tensor_sha(rms),
        "global_rms_fit_originals_only": True,
        "candidate_manifest": manifest,
        "candidate_manifest_sha256": _object_sha(manifest),
        "payload_tiers": tiers,
        "transform_abi": transform_abi,
    }
    fit_bundle = {
        "schema_version": FIT_SCHEMA,
        **common,
        "model_fit_iids": fit["iids"],
        "early_stop_validation_iids": validation_iids,
        "states": states,
        "states_semantic_sha256": state_sha,
        "projection_fit_rows_are_original_model_fit_only": True,
        "derived_rows_consumed_by_fit": 0,
        "oof_or_validation_values_present": False,
    }
    eval_bundle = {
        "schema_version": EVAL_SCHEMA,
        **common,
        "exploratory_oof": query,
        "family_by_iid": family_by_iid,
        "diagnostic_views": views,
        "diagnostic_view_semantic_sha256": view_hashes,
        "model_fit_values_or_projection_states_present": False,
        "source_used_only_for_ineligible_noop_diagnostic": True,
    }
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    fit_path = output / "fit_bundle.pt"
    eval_path = output / "exploratory_oof_bundle.pt"
    fit_sha = _save_torch(fit_path, fit_bundle)
    eval_sha = _save_torch(eval_path, eval_bundle)
    receipt: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "status": "V4A_ANALYTIC_FRONTIER_FOLD_PREPARED_BURNED_DEVELOPMENT",
        **DEVELOPMENT_FIELDS,
        "unique_original_base_clips": 644,
        "model_fit_original_rows": len(fit["iids"]),
        "early_stop_validation_original_rows": len(validation_iids),
        "exploratory_oof_original_rows": len(query["iids"]),
        "anchor_derived_diagnostic_rows_this_fold": (
            len(query["iids"]) * (1 + len(TEMPORAL_NEGATIVES))
        ),
        "observed_paired_source_diagnostic_rows_this_fold": len(query["iids"]),
        "derived_or_diagnostic_rows_are_independent_samples": False,
        "derived_rows_consumed_by_fit": 0,
        "labels_heads_losses_optimizers": {
            "action_labels_present": False,
            "classification_head_present": False,
            "learned_loss_present": False,
            "optimizer_present": False,
        },
        "target": {
            "definition": "temporal_center(anchor ordered DINO)[32,768]",
            "source_subtracted": False,
            "full_768": True,
        },
        "source_usage": {
            "only_source_noop_separation_diagnostic": True,
            "eligible_for_temporal_mechanics_gate": False,
            "consumed_by_fit_or_projection": False,
        },
        "family_labels_used_only_for_frozen_split_and_evaluation_bootstrap": True,
        "observed_oof_family_count": observed_fold_family_count,
        "family_or_transform_labels_consumed_by_fit": False,
        "v3_frozen_split_digest_contract_reused": True,
        "v3_runtime_imported": False,
        "v3_oof_result_consumed": False,
        "fold": split,
        "config": config_value,
        "config_sha256": common["config_sha256"],
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_iid_digest": exact_iid_digest,
        "exact644_raw_target_sha256": raw_target_sha,
        "exact644_source_noop_sha256": source_noop_sha,
        "exact644_population_authority": population,
        "global_rms": float(rms),
        "global_rms_sha256": common["global_rms_sha256"],
        "states_semantic_sha256": state_sha,
        "candidate_manifest": manifest,
        "candidate_manifest_sha256": common["candidate_manifest_sha256"],
        "payload_tiers": tiers,
        "transform_abi": transform_abi,
        "diagnostic_view_semantic_sha256": view_hashes,
        "fit_bundle": {
            "path": str(fit_path.resolve()), "sha256": fit_sha,
            "size_bytes": fit_path.stat().st_size,
            "contains_oof_or_validation_values": False,
        },
        "exploratory_oof_bundle": {
            "path": str(eval_path.resolve()), "sha256": eval_sha,
            "size_bytes": eval_path.stat().st_size,
            "contains_model_fit_values_or_projection_states": False,
        },
        "implementation": run_binding,
    }
    if set(receipt) | {"receipt_digest"} != PREPARE_RECEIPT_KEYS:
        raise RuntimeError("v4-A internal prepare receipt key closure differs")
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "prepare_receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "fit_bundle_sha256": fit_sha,
        "exploratory_oof_bundle_sha256": eval_sha,
        "fold_counts": split["counts"],
        "candidate_count": len(manifest),
    }


_COMMON_BUNDLE_KEYS = {
    "config", "config_sha256", "fold", "implementation",
    "feature_receipt_sha256", "feature_receipt_digest",
    "exact644_iid_digest", "exact644_raw_target_sha256",
    "exact644_source_noop_sha256", "exact644_population_authority",
    "global_rms", "global_rms_sha256", "global_rms_fit_originals_only",
    "candidate_manifest", "candidate_manifest_sha256", "payload_tiers",
    "transform_abi",
}
_FIT_BUNDLE_KEYS = _COMMON_BUNDLE_KEYS | {
    "schema_version", "model_fit_iids", "early_stop_validation_iids",
    "states", "states_semantic_sha256",
    "projection_fit_rows_are_original_model_fit_only",
    "derived_rows_consumed_by_fit", "oof_or_validation_values_present",
}
_EVAL_BUNDLE_KEYS = _COMMON_BUNDLE_KEYS | {
    "schema_version", "exploratory_oof", "family_by_iid",
    "diagnostic_views", "diagnostic_view_semantic_sha256",
    "model_fit_values_or_projection_states_present",
    "source_used_only_for_ineligible_noop_diagnostic",
}


def _validate_common_bundle(
    bundle: Mapping[str, Any],
    prepare: Mapping[str, Any],
    run_binding: Mapping[str, Any],
    config: Config,
) -> None:
    if (
        bundle["config"] != asdict(config)
        or bundle["config_sha256"] != _object_sha(asdict(config))
        or bundle["implementation"] != run_binding
        or bundle["feature_receipt_sha256"] != prepare["feature_receipt_sha256"]
        or bundle["feature_receipt_digest"] != prepare["feature_receipt_digest"]
        or bundle["exact644_iid_digest"] != prepare["exact644_iid_digest"]
        or bundle["exact644_raw_target_sha256"] != prepare["exact644_raw_target_sha256"]
        or bundle["exact644_source_noop_sha256"] != prepare["exact644_source_noop_sha256"]
        or bundle["exact644_population_authority"] != prepare["exact644_population_authority"]
        or bundle["fold"] != prepare["fold"]
        or bundle["global_rms_sha256"] != prepare["global_rms_sha256"]
        or _tensor_sha(bundle["global_rms"]) != bundle["global_rms_sha256"]
        or bundle["global_rms_fit_originals_only"] is not True
        or bundle["candidate_manifest"] != prepare["candidate_manifest"]
        or bundle["candidate_manifest_sha256"] != _object_sha(bundle["candidate_manifest"])
        or bundle["candidate_manifest_sha256"] != prepare["candidate_manifest_sha256"]
        or bundle["payload_tiers"] != _payload_tiers(bundle["candidate_manifest"])
        or bundle["payload_tiers"] != prepare["payload_tiers"]
        or bundle["transform_abi"] != prepare["transform_abi"]
    ):
        raise ValueError("v4-A common bundle/prepare authority differs")


def _load_fold_bundles(
    args: argparse.Namespace,
    prepare: Mapping[str, Any],
    run_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Config]:
    fit_binding = prepare.get("fit_bundle")
    eval_binding = prepare.get("exploratory_oof_bundle")
    if (
        type(fit_binding) is not dict
        or type(eval_binding) is not dict
        or set(fit_binding) != {"path", "sha256", "size_bytes", "contains_oof_or_validation_values"}
        or set(eval_binding) != {
            "path", "sha256", "size_bytes",
            "contains_model_fit_values_or_projection_states",
        }
        or Path(args.fit_bundle).resolve() != Path(fit_binding["path"])
        or Path(args.eval_bundle).resolve() != Path(eval_binding["path"])
        or args.expected_fit_bundle_sha256 != fit_binding["sha256"]
        or args.expected_eval_bundle_sha256 != eval_binding["sha256"]
        or fit_binding["contains_oof_or_validation_values"] is not False
        or eval_binding["contains_model_fit_values_or_projection_states"] is not False
    ):
        raise ValueError("v4-A bundle path/SHA authority differs")
    fit = _load_torch(
        Path(args.fit_bundle), args.expected_fit_bundle_sha256,
        fit_binding["size_bytes"],
    )
    evaluation = _load_torch(
        Path(args.eval_bundle), args.expected_eval_bundle_sha256,
        eval_binding["size_bytes"],
    )
    if type(fit) is not dict or set(fit) != _FIT_BUNDLE_KEYS or fit.get("schema_version") != FIT_SCHEMA:
        raise ValueError("v4-A fit bundle exact-key/schema differs")
    if type(evaluation) is not dict or set(evaluation) != _EVAL_BUNDLE_KEYS or evaluation.get("schema_version") != EVAL_SCHEMA:
        raise ValueError("v4-A eval bundle exact-key/schema differs")
    config = _config_from_value(fit["config"])
    if _config_from_value(evaluation["config"]) != config:
        raise ValueError("v4-A fit/eval configs differ")
    _validate_common_bundle(fit, prepare, run_binding, config)
    _validate_common_bundle(evaluation, prepare, run_binding, config)
    if any(fit[key] != evaluation[key] for key in _COMMON_BUNDLE_KEYS if key != "global_rms"):
        raise ValueError("v4-A fit/eval common fields differ")
    if not torch.equal(fit["global_rms"], evaluation["global_rms"]):
        raise ValueError("v4-A fit/eval RMS differs")
    fit_count = fit["fold"]["counts"]["model_fit"]
    if (
        len(fit["model_fit_iids"]) != fit_count
        or len(fit["early_stop_validation_iids"])
        != fit["fold"]["counts"]["early_stop_validation"]
        or fit["projection_fit_rows_are_original_model_fit_only"] is not True
        or fit["derived_rows_consumed_by_fit"] != 0
        or fit["oof_or_validation_values_present"] is not False
    ):
        raise ValueError("v4-A fit-only row contract differs")
    _validate_states(fit["states"], fit_count, config)
    if (
        fit["states_semantic_sha256"] != _state_semantic_sha(fit["states"])
        or fit["states_semantic_sha256"] != prepare["states_semantic_sha256"]
    ):
        raise ValueError("v4-A projection state semantic SHA differs")
    _validate_candidate_manifest(fit["candidate_manifest"], fit["states"], config)
    return fit, evaluation, config


def _move_states(states: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved = _states_to_cpu(states)
    for family in ("frame", "tucker", "clip"):
        moved[family] = {
            key: value.to(device) for key, value in moved[family].items()
        }
    return moved


def evaluate_fold(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    prepare = _load_receipt(
        Path(args.prepare_receipt), args.expected_prepare_receipt_sha256,
        PREPARE_SCHEMA,
    )
    if (
        set(prepare) != PREPARE_RECEIPT_KEYS
        or prepare.get("implementation") != run_binding
        or args.fold_index != prepare.get("fold", {}).get("outer_fold")
        or prepare.get("v3_runtime_imported") is not False
        or prepare.get("v3_oof_result_consumed") is not False
        or prepare.get("derived_rows_consumed_by_fit") != 0
    ):
        raise ValueError("v4-A prepare authority/scope differs")
    fit, evaluation, config = _load_fold_bundles(args, prepare, run_binding)
    exploratory = evaluation["exploratory_oof"]
    if type(exploratory) is not dict or set(exploratory) != {"value", "iids"}:
        raise ValueError("v4-A OOF rows differ")
    iids = exploratory["iids"]
    query = exploratory["value"]
    family_by_iid = evaluation["family_by_iid"]
    views = evaluation["diagnostic_views"]
    if (
        type(views) is not dict or set(views) != set(ALL_VIEWS)
        or set(iids) != set(family_by_iid)
        or not 1 <= len(set(family_by_iid.values())) <= 28
        or evaluation["model_fit_values_or_projection_states_present"] is not False
        or evaluation["source_used_only_for_ineligible_noop_diagnostic"] is not True
        or _view_semantic_hashes(query, views)
        != evaluation["diagnostic_view_semantic_sha256"]
        or evaluation["diagnostic_view_semantic_sha256"]
        != prepare["diagnostic_view_semantic_sha256"]
    ):
        raise ValueError("v4-A OOF diagnostic-view authority differs")
    recomputed_views, recomputed_transform_abi = _diagnostic_views(
        query, views[SOURCE_DIAGNOSTIC], iids, config.seed
    )
    if (
        recomputed_transform_abi != evaluation["transform_abi"]
        or any(
            not torch.equal(recomputed_views[name], views[name])
            for name in ALL_VIEWS
        )
    ):
        raise ValueError("v4-A diagnostic transform ABI/tensors differ")
    closure = [*fit["model_fit_iids"], *fit["early_stop_validation_iids"], *iids]
    signed_groups = {
        "model_fit": fit["model_fit_iids"],
        "early_stop_validation": fit["early_stop_validation_iids"],
        "exploratory_oof": iids,
    }
    if (
        len(closure) != 644 or len(set(closure)) != 644
        or _object_sha(signed_groups) != fit["fold"]["iid_digest"]
        or fit["fold"]["outer_assignment_digest"]
        != V3_FROZEN_OUTER_ASSIGNMENT_DIGEST
        or fit["fold"]["iid_digest"]
        != V3_FROZEN_FOLD_IID_DIGESTS[args.fold_index]
    ):
        raise ValueError("v4-A exact644 fold closure differs")
    device = _analytic_device(args.device)
    states = _move_states(fit["states"], device)
    query_device = query.to(device)
    views_device = {name: value.to(device) for name, value in views.items()}
    teacher_device, candidates_device = _margin_vectors(
        query_device, views_device, fit["candidate_manifest"], states
    )
    teacher = {name: value.detach().cpu() for name, value in teacher_device.items()}
    candidate_margins = {
        candidate: {
            negative: value.detach().cpu()
            for negative, value in rows.items()
        }
        for candidate, rows in candidates_device.items()
    }
    summaries_by_negative = {}
    observed_family_count = len(set(family_by_iid.values()))
    for negative_index, negative in enumerate((*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)):
        summaries_by_negative[negative] = _margin_bootstrap(
            teacher[negative],
            {name: candidate_margins[name][negative] for name in fit["candidate_manifest"]},
            iids, family_by_iid,
            config.seed + 10000 + 100 * args.fold_index + 10 * negative_index,
            config, expected_family_count=observed_family_count,
        )
    teacher_gate = _teacher_gate(summaries_by_negative)
    fold_candidate_gates = {
        name: _candidate_gate(
            name, summaries_by_negative,
            teacher_gate["all_temporal_negatives_hard_gate"],
        )
        for name in fit["candidate_manifest"]
    }

    zero_metrics = v2._metric_rows(
        query, torch.zeros_like(query), iids, evaluation["global_rms"],
        evaluation["fold"]["fixed_energy_bin_edges_exact644"],
    )
    raw_metrics: dict[str, Any] = {"zero": zero_metrics}
    mse_secondary: dict[str, Any] = {}
    for candidate_index, (name, specification) in enumerate(
        fit["candidate_manifest"].items()
    ):
        reconstruction = _reconstruct_candidate(
            query_device, specification, states
        ).detach().cpu()
        metric = v2._metric_rows(
            query, reconstruction, iids, evaluation["global_rms"],
            evaluation["fold"]["fixed_energy_bin_edges_exact644"],
        )
        raw_metrics[name] = metric
        mse_secondary[name] = {
            "clip_bootstrap_vs_zero": v2._paired_ratio(
                metric["per_iid"], zero_metrics["per_iid"],
                config.seed + 13000 + 100 * args.fold_index + candidate_index,
                draws=config.bootstrap_draws,
            ),
            "family_cluster_bootstrap_vs_zero": _family_ratio_secondary(
                metric["per_iid"], zero_metrics["per_iid"], family_by_iid,
                config.seed + 14000 + 100 * args.fold_index + candidate_index,
                config.bootstrap_draws, observed_family_count,
            ),
            "eligible_for_primary_temporal_gate": False,
        }
    per_iid_margins = {
        "iids": list(iids),
        "teacher": {
            negative: [float(value) for value in teacher[negative]]
            for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)
        },
        "candidates": {
            name: {
                negative: [float(value) for value in candidate_margins[name][negative]]
                for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)
            }
            for name in fit["candidate_manifest"]
        },
    }
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    receipt: dict[str, Any] = {
        "schema_version": FOLD_SCHEMA,
        "status": "V4A_ANALYTIC_FRONTIER_OOF_FOLD_EVALUATED",
        **DEVELOPMENT_FIELDS,
        "fold_index": args.fold_index,
        "fold": fit["fold"],
        "sample_accounting": {
            "unique_exact644_development": 644,
            "oof_original_queries_this_fold": len(iids),
            "anchor_derived_diagnostic_rows_this_fold": (
                len(iids) * (1 + len(TEMPORAL_NEGATIVES))
            ),
            "observed_paired_source_diagnostic_rows_this_fold": len(iids),
            "derived_or_diagnostic_rows_are_independent_samples": False,
            "bootstrap_units": ["original_iid", "family_cluster"],
            "observed_oof_family_count": observed_family_count,
            "exact28_family_gate_deferred_to_aggregate": True,
        },
        "primary_margin_definition": (
            "d(E(query),E(negative))-d(E(query),E(monotone_speed_warp(query)))"
        ),
        "distance_definition": "sum((code_a-code_b)^2)/(32*768)",
        "same_query_payload_and_same_metric_for_teacher_and_candidates": True,
        "uncompressed_teacher_is_identity_flatten": True,
        "teacher_evaluated_before_compressed_qualification": True,
        "source_noop": {
            "status": "SEPARATION_DIAGNOSTIC_ONLY",
            "eligible_for_temporal_mechanics_gate": False,
            "observed_paired_authority_not_derived_sample": True,
            "bootstrap_summary": summaries_by_negative[SOURCE_DIAGNOSTIC],
        },
        "temporal_margin_bootstrap_by_negative": {
            negative: summaries_by_negative[negative]
            for negative in TEMPORAL_NEGATIVES
        },
        "teacher_hard_gate": teacher_gate,
        "fold_candidate_gates_diagnostic_only": fold_candidate_gates,
        "per_iid_margins": per_iid_margins,
        "family_by_iid": family_by_iid,
        "raw_reconstruction_metrics_secondary": raw_metrics,
        "raw_mse_bootstrap_secondary": mse_secondary,
        "raw_mse_used_for_primary_gate": False,
        "candidate_manifest": fit["candidate_manifest"],
        "candidate_manifest_sha256": fit["candidate_manifest_sha256"],
        "payload_tiers": fit["payload_tiers"],
        "no_fold_winner_or_rank_selection": True,
        "config": asdict(config),
        "config_sha256": _object_sha(asdict(config)),
        "feature_receipt_sha256": fit["feature_receipt_sha256"],
        "feature_receipt_digest": fit["feature_receipt_digest"],
        "exact644_iid_digest": fit["exact644_iid_digest"],
        "exact644_raw_target_sha256": fit["exact644_raw_target_sha256"],
        "exact644_source_noop_sha256": fit["exact644_source_noop_sha256"],
        "exact644_population_authority": fit["exact644_population_authority"],
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "fit_bundle_sha256": args.expected_fit_bundle_sha256,
        "exploratory_oof_bundle_sha256": args.expected_eval_bundle_sha256,
        "states_semantic_sha256": fit["states_semantic_sha256"],
        "diagnostic_view_semantic_sha256": evaluation["diagnostic_view_semantic_sha256"],
        "transform_abi": evaluation["transform_abi"],
        "labels_heads_losses_optimizers_absent": True,
        "only_temporal_mechanics_tested": True,
        "action_representation_qualified": False,
        "source_identity_preservation_tested": False,
        "video_editing_tested": False,
        "prior_generation_qualified": False,
        "renderer_authorized": False,
        "inference_authorized": False,
        "vae_necessary": None,
        "implementation": run_binding,
    }
    if set(receipt) | {"receipt_digest"} != FOLD_RECEIPT_KEYS:
        raise RuntimeError("v4-A internal fold receipt key closure differs")
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "teacher_fold_hard_gate": teacher_gate["all_temporal_negatives_hard_gate"],
    }


def _recompute_fold_margin_evidence(
    row: Mapping[str, Any], config: Config
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    per_iid = row.get("per_iid_margins")
    family_by_iid = row.get("family_by_iid")
    manifest = row.get("candidate_manifest")
    if (
        type(per_iid) is not dict
        or set(per_iid) != {"iids", "teacher", "candidates"}
        or type(family_by_iid) is not dict
        or type(manifest) is not dict
        or set(per_iid["teacher"])
        != set((*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC))
        or set(per_iid["candidates"]) != set(manifest)
    ):
        raise ValueError("v4-A fold per-IID margin closure differs")
    iids = per_iid["iids"]
    if (
        len(iids) != row["fold"]["counts"]["exploratory_oof"]
        or len(set(iids)) != len(iids)
        or set(iids) != set(family_by_iid)
    ):
        raise ValueError("v4-A fold margin IID/family closure differs")
    observed_family_count = len(set(family_by_iid.values()))
    summaries = {}
    for negative_index, negative in enumerate((*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)):
        teacher = torch.tensor(per_iid["teacher"][negative], dtype=torch.float32)
        candidates = {
            name: torch.tensor(per_iid["candidates"][name][negative], dtype=torch.float32)
            for name in manifest
        }
        summaries[negative] = _margin_bootstrap(
            teacher, candidates, iids, family_by_iid,
            config.seed + 10000 + 100 * row["fold_index"] + 10 * negative_index,
            config, observed_family_count,
        )
    teacher_gate = _teacher_gate(summaries)
    candidate_gates = {
        name: _candidate_gate(
            name, summaries, teacher_gate["all_temporal_negatives_hard_gate"]
        )
        for name in manifest
    }
    return summaries, teacher_gate, candidate_gates


def _aggregate_secondary_mse(
    combined_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
    family_by_iid: Mapping[str, str],
    config: Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = [name for name in combined_metrics if name != "zero"]
    iids = [row["iid"] for row in combined_metrics["zero"]]
    if len(iids) != 644 or len(set(iids)) != 644 or set(iids) != set(family_by_iid):
        raise ValueError("v4-A aggregate secondary MSE exact644 closure differs")
    matrices = []
    for name in ("zero", *names):
        rows = combined_metrics[name]
        if [item["iid"] for item in rows] != iids:
            raise ValueError("v4-A aggregate secondary MSE order differs")
        matrices.append(torch.tensor(
            [float(item["raw_mse"]) for item in rows], dtype=torch.float64
        ))
    matrix = torch.stack(matrices, dim=1)
    if bool((matrix[:, 0] <= 0.0).any()) or not bool(torch.isfinite(matrix).all()):
        raise ValueError("v4-A aggregate secondary MSE values differ")
    clip_samples = _resample_means(matrix, config.bootstrap_draws, config.seed + 23000)
    family_matrix, families = _family_means(matrix, iids, family_by_iid, 28)
    family_samples = _resample_means(
        family_matrix, config.bootstrap_draws, config.seed + 23001
    )
    aggregate_metrics = {
        name: v2._aggregate_metric_rows(combined_metrics[name])
        for name in combined_metrics
    }
    comparisons = {}
    for column, name in enumerate(names, start=1):
        clip_ratio = clip_samples[:, column] / clip_samples[:, 0]
        family_ratio = family_samples[:, column] / family_samples[:, 0]
        comparisons[name] = {
            "clip_bootstrap_vs_zero": {
                "bootstrap_unit": "iid", "iid_count": 644,
                "bootstrap_seed": config.seed + 23000,
                "draws": config.bootstrap_draws,
                "mean_ratio": float(matrix[:, column].mean() / matrix[:, 0].mean()),
                "ratio_95pct_ci": _interval(clip_ratio),
            },
            "family_cluster_bootstrap_vs_zero": {
                "bootstrap_unit": "family_cluster", "family_count": len(families),
                "iid_count": 644, "bootstrap_seed": config.seed + 23001,
                "draws": config.bootstrap_draws,
                "mean_ratio": float(
                    family_matrix[:, column].mean() / family_matrix[:, 0].mean()
                ),
                "ratio_95pct_ci": _interval(family_ratio),
                "equal_family_mean_estimator": True,
            },
            "eligible_for_primary_temporal_gate": False,
        }
    return aggregate_metrics, comparisons


def aggregate_oof(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    if (
        len(args.fold_receipt) != OUTER_FOLDS
        or len(args.expected_fold_receipt_sha256) != OUTER_FOLDS
    ):
        raise ValueError("v4-A aggregate requires exactly five fold receipts and SHAs")
    rows = [
        _load_receipt(Path(path), sha, FOLD_SCHEMA)
        for path, sha in zip(args.fold_receipt, args.expected_fold_receipt_sha256)
    ]
    by_fold = {row.get("fold_index"): row for row in rows}
    if set(by_fold) != set(range(OUTER_FOLDS)):
        raise ValueError("v4-A aggregate fold index closure differs")
    rows = [by_fold[index] for index in range(OUTER_FOLDS)]
    config = _config_from_value(rows[0].get("config"))
    common_keys = (
        "config", "config_sha256", "feature_receipt_sha256",
        "feature_receipt_digest", "exact644_iid_digest",
        "exact644_raw_target_sha256", "exact644_source_noop_sha256",
        "exact644_population_authority", "candidate_manifest",
        "candidate_manifest_sha256", "payload_tiers", "implementation",
    )
    for fold, row in enumerate(rows):
        if (
            set(row) != FOLD_RECEIPT_KEYS
            or row.get("implementation") != run_binding
            or row.get("fold", {}).get("outer_fold") != fold
            or row["fold"].get("outer_assignment_digest")
            != V3_FROZEN_OUTER_ASSIGNMENT_DIGEST
            or row["fold"].get("iid_digest") != V3_FROZEN_FOLD_IID_DIGESTS[fold]
            or row.get("no_fold_winner_or_rank_selection") is not True
            or row.get("raw_mse_used_for_primary_gate") is not False
            or row.get("labels_heads_losses_optimizers_absent") is not True
            or row.get("only_temporal_mechanics_tested") is not True
            or row.get("source_noop", {}).get("eligible_for_temporal_mechanics_gate") is not False
            or row.get("action_representation_qualified") is not False
            or row.get("renderer_authorized") is not False
            or row.get("inference_authorized") is not False
            or row.get("vae_necessary", "missing") is not None
            or any(row[key] != rows[0][key] for key in common_keys)
        ):
            raise ValueError("v4-A aggregate fold authority/scope differs")
        observed_family_count = len(set(row["family_by_iid"].values()))
        if row["sample_accounting"].get("observed_oof_family_count") != observed_family_count:
            raise ValueError("v4-A aggregate observed fold-family count differs")
        summaries, teacher_gate, candidate_gates = _recompute_fold_margin_evidence(
            row, config
        )
        if (
            summaries != {
                **row["temporal_margin_bootstrap_by_negative"],
                SOURCE_DIAGNOSTIC: row["source_noop"]["bootstrap_summary"],
            }
            or teacher_gate != row["teacher_hard_gate"]
            or candidate_gates != row["fold_candidate_gates_diagnostic_only"]
        ):
            raise ValueError("v4-A fold bootstrap/gate evidence was re-signed")
    manifest = rows[0]["candidate_manifest"]
    if (
        rows[0]["candidate_manifest_sha256"] != _object_sha(manifest)
        or rows[0]["payload_tiers"] != _payload_tiers(manifest)
    ):
        raise ValueError("v4-A aggregate candidate/payload authority differs")

    all_iids: list[str] = []
    family_by_iid: dict[str, str] = {}
    combined_teacher = {
        negative: [] for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)
    }
    combined_candidates = {
        name: {
            negative: [] for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)
        }
        for name in manifest
    }
    combined_metrics = {"zero": []} | {name: [] for name in manifest}
    for row in rows:
        fold_iids = row["per_iid_margins"]["iids"]
        if set(family_by_iid) & set(fold_iids):
            raise ValueError("v4-A aggregate fold IID overlap")
        all_iids.extend(fold_iids)
        family_by_iid.update(row["family_by_iid"])
        for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC):
            combined_teacher[negative].extend(
                row["per_iid_margins"]["teacher"][negative]
            )
        for name in manifest:
            for negative in (*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC):
                combined_candidates[name][negative].extend(
                    row["per_iid_margins"]["candidates"][name][negative]
                )
        for name in combined_metrics:
            metric = row["raw_reconstruction_metrics_secondary"][name]
            if [item["iid"] for item in metric["per_iid"]] != fold_iids:
                raise ValueError("v4-A aggregate metric/margin IID join differs")
            combined_metrics[name].extend(metric["per_iid"])
    if (
        len(all_iids) != 644 or len(set(all_iids)) != 644
        or _object_sha(sorted(all_iids)) != rows[0]["exact644_iid_digest"]
        or set(all_iids) != set(family_by_iid)
        or len(set(family_by_iid.values())) != 28
    ):
        raise ValueError("v4-A aggregate exact644/exact28 closure differs")

    aggregate_summaries = {}
    for negative_index, negative in enumerate((*TEMPORAL_NEGATIVES, SOURCE_DIAGNOSTIC)):
        aggregate_summaries[negative] = _margin_bootstrap(
            torch.tensor(combined_teacher[negative], dtype=torch.float32),
            {
                name: torch.tensor(combined_candidates[name][negative], dtype=torch.float32)
                for name in manifest
            },
            all_iids, family_by_iid,
            config.seed + 20000 + 10 * negative_index,
            config, expected_family_count=28,
        )
    teacher_gate = _teacher_gate(aggregate_summaries)
    candidate_gates = {
        name: _candidate_gate(
            name, aggregate_summaries,
            teacher_gate["all_temporal_negatives_hard_gate"],
        )
        for name in manifest
    }
    qualified = [
        name for name in manifest
        if candidate_gates[name]["all_temporal_negatives_hard_gate"]
    ]
    aggregate_metrics, mse_secondary = _aggregate_secondary_mse(
        combined_metrics, family_by_iid, config
    )
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    receipt: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "V4A_ANALYTIC_FRONTIER_EXACT5_OOF_AGGREGATED_BURNED644",
        **DEVELOPMENT_FIELDS,
        "oof_unique_original_iids": 644,
        "oof_each_original_iid_evaluated_once": True,
        "family_cluster_count": 28,
        "derived_or_diagnostic_rows_are_independent_samples": False,
        "bootstrap_units": ["original_iid", "family_cluster"],
        "primary_margin_definition": rows[0]["primary_margin_definition"],
        "distance_definition": rows[0]["distance_definition"],
        "teacher_temporal_margin_bootstrap_by_negative": {
            negative: aggregate_summaries[negative]["teacher"]
            for negative in TEMPORAL_NEGATIVES
        },
        "compressed_temporal_margin_bootstrap_by_negative": {
            negative: aggregate_summaries[negative]["candidates"]
            for negative in TEMPORAL_NEGATIVES
        },
        "teacher_hard_gate": teacher_gate,
        "candidate_gates": candidate_gates,
        "qualified_candidates_unranked": qualified,
        "qualified_candidate_count": len(qualified),
        "no_oof_winner_selected": True,
        "no_rank_or_factorization_selected": True,
        "qualified_list_is_not_downstream_selection_authorization": True,
        "source_noop_separation_diagnostic": {
            "eligible_for_temporal_mechanics_gate": False,
            "teacher": aggregate_summaries[SOURCE_DIAGNOSTIC]["teacher"],
            "candidates": aggregate_summaries[SOURCE_DIAGNOSTIC]["candidates"],
        },
        "raw_reconstruction_metrics_secondary": aggregate_metrics,
        "raw_mse_bootstrap_secondary": mse_secondary,
        "raw_mse_used_for_primary_gate": False,
        "candidate_manifest": manifest,
        "candidate_manifest_sha256": _object_sha(manifest),
        "payload_tiers": _payload_tiers(manifest),
        "equal_payload_cross_structure_comparisons_are_diagnostic_only": True,
        "config": asdict(config),
        "config_sha256": _object_sha(asdict(config)),
        "feature_receipt_sha256": rows[0]["feature_receipt_sha256"],
        "feature_receipt_digest": rows[0]["feature_receipt_digest"],
        "exact644_iid_digest": rows[0]["exact644_iid_digest"],
        "exact644_raw_target_sha256": rows[0]["exact644_raw_target_sha256"],
        "exact644_source_noop_sha256": rows[0]["exact644_source_noop_sha256"],
        "exact644_population_authority": rows[0]["exact644_population_authority"],
        "outer_assignment_digest": V3_FROZEN_OUTER_ASSIGNMENT_DIGEST,
        "fold_receipt_sha256": list(args.expected_fold_receipt_sha256),
        "fold_iid_digests": V3_FROZEN_FOLD_IID_DIGESTS,
        "labels_heads_losses_optimizers_absent": True,
        "only_temporal_mechanics_tested": True,
        "action_representation_qualified": False,
        "identity_disentanglement_qualified": False,
        "source_identity_preservation_tested": False,
        "video_editing_tested": False,
        "prior_generation_qualified": False,
        "renderer_authorized": False,
        "inference_authorized": False,
        "full644_refit_authorized": False,
        "vae_necessary": None,
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "teacher_hard_gate": teacher_gate["all_temporal_negatives_hard_gate"],
        "qualified_candidates_unranked": qualified,
    }


def _add_common_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--expected-feature-receipt-sha256", required=True)
    parser.add_argument("--fold-index", required=True, type=int, choices=range(OUTER_FOLDS))
    parser.add_argument("--seed", type=int, default=Config().seed)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-fold")
    _add_common_prepare_arguments(prepare)
    prepare.set_defaults(function=prepare_fold)

    evaluate = commands.add_parser("evaluate-fold")
    evaluate.add_argument("--prepare-receipt", required=True)
    evaluate.add_argument("--expected-prepare-receipt-sha256", required=True)
    evaluate.add_argument("--fit-bundle", required=True)
    evaluate.add_argument("--expected-fit-bundle-sha256", required=True)
    evaluate.add_argument("--eval-bundle", required=True)
    evaluate.add_argument("--expected-eval-bundle-sha256", required=True)
    evaluate.add_argument("--fold-index", required=True, type=int, choices=range(OUTER_FOLDS))
    evaluate.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(function=evaluate_fold)

    aggregate = commands.add_parser("aggregate-oof")
    aggregate.add_argument("--fold-receipt", action="append", required=True)
    aggregate.add_argument(
        "--expected-fold-receipt-sha256", action="append", required=True
    )
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(function=aggregate_oof)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.function(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
