#!/usr/bin/env python3
"""Exact5 analytic payload frontier over frozen ordered V-JEPA2 features.

The exact v4-A outer-fold IID assignment is consumed from its sealed receipt;
V-JEPA2 energy is never used to repartition the population.  For each fold,
orthogonal projections are fitted on original features from the other four
folds, then evaluated once on the held-out fold using five independently
materialized frozen-backbone views.  No OOF value selects a projection, rank,
factorization, winner, or downstream renderer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import torch

from methods.bernini_action_editing import extract_vjepa2_ordered_contextual_features_v4c as features
from methods.bernini_action_editing import semantic_anchor_linear_frontier_v4_fast as v4a


SCHEMA = "semantic-anchor-vjepa2-analytic-frontier-exact5-receipt-v4c"
STATUS = "V4C_VJEPA2_ANALYTIC_FRONTIER_COMPLETE_BURNED_DEVELOPMENT"
TIME_STEPS = 32
FEATURE_DIM = 1024
FULL_NUMEL = TIME_STEPS * FEATURE_DIM
OUTER_FOLDS = 5
PAYLOAD_BUDGETS = (32, 64, 128, 256, 384)
TUCKER_TIME_RANK = 4
NEGATIVES = ("reverse", "block_shuffle", "phase_swap")
POSITIVE = "monotone_warp"
SEED = 20260819
TEACHER_RETENTION = 0.80
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_ALPHA = 0.05
V4A_RECEIPT_FILE_SHA256 = "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2"
V4A_RECEIPT_DIGEST = "f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86"
V4A_EVIDENCE_SHA256 = "f1d34d9ade4e36200f5dbd0da277cf8cf1221482f66c76d42c168a984a0cf123"
V4A_IMPLEMENTATION_SHA256 = "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973"
EXTRACTOR_IMPLEMENTATION_SHA256 = "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc"
OUTER_ASSIGNMENT_DIGEST = "5ab9704f456768b440c966a53328de0c1a67836548f8f8ebd92e50d21846ab5f"
FOLD_IID_DIGESTS = {
    0: "26b5cb90aea6140c8719ae48c2b98082a999d1ca79489ec5bdc70e6ce6745773",
    1: "18c7ad8a24f678ea93cc9d16365fcba0cb8d101667eed9542618240f3ed9c13f",
    2: "b1a85b86390bb773e23125f55f1a49152edf3c426de5ebe2e519aa421c3b430b",
    3: "b2abd43da040c878ac0620022e7fb4c5a8c967580dc6615ced7a6dec62404d3d",
    4: "473f906f5874ddc36227c77ccdc79ec80fa6fe55692f65adf12c049891e74fcf",
}
OOF_COUNTS = (131, 127, 128, 129, 129)


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    payload_budgets: tuple[int, ...] = PAYLOAD_BUDGETS
    tucker_time_rank: int = TUCKER_TIME_RANK
    teacher_retention: float = TEACHER_RETENTION
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_alpha: float = BOOTSTRAP_ALPHA

    def validate(self) -> None:
        if self != Config():
            raise ValueError("v4-C analytic frontier configuration is immutable")
        if any(
            payload % TIME_STEPS or payload % self.tucker_time_rank or payload > 384
            for payload in self.payload_budgets
        ):
            raise ValueError("v4-C equal-payload frontier differs")


@dataclass(frozen=True)
class Record:
    iid: str
    family: str
    strict: bool
    views: Mapping[str, torch.Tensor]


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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _object_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return features.tensor_sha256(value)


def _file_sha(path: Path) -> str:
    return features.file_sha256(path)


def _binding() -> dict[str, Any]:
    paths = {
        "implementation": Path(__file__).resolve(strict=True),
        "extractor": Path(features.__file__).resolve(strict=True),
        "v4a_bootstrap_abi": Path(v4a.__file__).resolve(strict=True),
    }
    result = {
        name: {"path": str(path), "sha256": _file_sha(path), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    if (
        result["extractor"]["sha256"] != EXTRACTOR_IMPLEMENTATION_SHA256
        or result["v4a_bootstrap_abi"]["sha256"] != V4A_IMPLEMENTATION_SHA256
    ):
        raise RuntimeError("v4-C frozen dependency source differs")
    return result


def _load_json_sealed(path: Path, expected_sha: str) -> dict[str, Any]:
    expected_sha = features._sha(expected_sha, "expected sealed JSON SHA")
    if not path.is_absolute() or path.is_symlink() or str(path) != str(path.resolve(strict=True)):
        raise ValueError("sealed JSON path must be absolute/plain/canonical")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1 or before.st_size <= 0
    ):
        raise ValueError("sealed JSON must be 0444/nlink1 regular")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        raw = handle.read()
        handle.seek(0)
        raw_after = handle.read()
        closed = os.fstat(handle.fileno())
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mode,
        value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
    )
    if (
        raw != raw_after or hashlib.sha256(raw).hexdigest() != expected_sha
        or identity(before) != identity(opened) or identity(opened) != identity(closed)
        or identity(closed) != identity(path.lstat())
    ):
        raise ValueError("sealed JSON bytes/identity differ")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("sealed JSON root differs")
    return value


def load_frozen_v4a_split(path: Path, expected_sha: str) -> tuple[dict[str, int], dict[str, Any]]:
    if expected_sha != V4A_RECEIPT_FILE_SHA256:
        raise ValueError("v4-A receipt SHA is not the preregistered authority")
    receipt = _load_json_sealed(path, V4A_RECEIPT_FILE_SHA256)
    unsigned = dict(receipt)
    embedded_digest = unsigned.pop("receipt_digest", None)
    closure = receipt.get("oof_closure")
    evidence = closure.get("embedded_paired_margin_evidence") if type(closure) is dict else None
    folds = receipt.get("folds")
    if (
        receipt.get("schema_version") != v4a.SCHEMA
        or receipt.get("status") != "V4_FAST_EXACT5_LINEAR_FRONTIER_COMPLETE_BURNED_DEVELOPMENT"
        or embedded_digest != V4A_RECEIPT_DIGEST or _object_sha(unsigned) != embedded_digest
        or receipt.get("implementation", {}).get("implementation_sha256") != V4A_IMPLEMENTATION_SHA256
        or receipt.get("frozen_split", {}).get("outer_assignment_digest") != OUTER_ASSIGNMENT_DIGEST
        or receipt.get("frozen_split", {}).get("fold_iid_digests") != {
            str(key): value for key, value in FOLD_IID_DIGESTS.items()
        }
        or type(folds) is not list or len(folds) != 5
        or type(evidence) is not list or len(evidence) != 644
        or closure.get("embedded_paired_margin_evidence_count") != 644
        or closure.get("embedded_paired_margin_evidence_sha256") != V4A_EVIDENCE_SHA256
        or _object_sha(evidence) != V4A_EVIDENCE_SHA256
    ):
        raise ValueError("sealed v4-A exact5 authority differs")
    for fold, row in enumerate(folds):
        evidence_iids = [
            item.get("iid") for item in evidence if item.get("outer_fold") == fold
        ]
        if (
            row.get("fold_index") != fold
            or row.get("frozen_v2_outer_assignment_digest") != OUTER_ASSIGNMENT_DIGEST
            or row.get("frozen_v2_fold_iid_digest") != FOLD_IID_DIGESTS[fold]
            or row.get("oof_original_count") != OOF_COUNTS[fold]
            or len(evidence_iids) != OOF_COUNTS[fold]
            or row.get("oof_iid_digest") != _object_sha(evidence_iids)
        ):
            raise ValueError("sealed v4-A fold authority differs")
    assignment: dict[str, int] = {}
    for row in evidence:
        iid, fold = row.get("iid"), row.get("outer_fold")
        family = row.get("family")
        if (
            type(iid) is not str or not iid or iid in assignment
            or type(family) is not str or not family
            or type(fold) is not int or not 0 <= fold < 5
        ):
            raise ValueError("v4-A embedded IID/fold assignment differs")
        assignment[iid] = fold
    if (
        len(assignment) != 644
        or _object_sha(assignment) != OUTER_ASSIGNMENT_DIGEST
        or tuple(sum(value == fold for value in assignment.values()) for fold in range(5)) != OOF_COUNTS
        or len({row.get("family") for row in evidence}) != 28
    ):
        raise ValueError("v4-A embedded exact644 assignment closure differs")
    return assignment, receipt


def load_v4c_features(feature_root: Path, expected_receipt_sha: str) -> tuple[list[Record], dict[str, Any]]:
    if (
        not feature_root.is_absolute() or feature_root.is_symlink()
        or str(feature_root) != str(feature_root.resolve(strict=True))
    ):
        raise ValueError("v4-C feature root must be absolute/plain/canonical")
    root = feature_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("v4-C feature root differs")
    receipt_path = root / "feature_extraction_receipt.json"
    receipt = _load_json_sealed(receipt_path, expected_receipt_sha)
    unsigned = dict(receipt)
    embedded_digest = unsigned.pop("receipt_digest", None)
    model_and_source = receipt.get("model_and_source_closure")
    model_closure = model_and_source.get("model") if type(model_and_source) is dict else None
    module_closure = (
        model_and_source.get("transformers") if type(model_and_source) is dict else None
    )
    model_files = model_closure.get("files") if type(model_closure) is dict else None
    module_rows = module_closure.get("modules") if type(module_closure) is dict else None
    expected_sampling_abi = {
        "exact81_to_base64_formula": "floor(80*k/63), k=0..63",
        "exact81_to_base64_indices_sha256": features.BASE64_INDICES_SHA256,
        "warp64_formula": "coord[2*i+j]=2*float32(WARP32[i])+j",
        "warp64_coordinates_sha256": features.WARP64_COORDINATES_SHA256,
        "phase32_block_permutation": list(features.PHASE_BLOCK_PERMUTATION),
        "transform_axis": "pixel_values_videos temporal dim 1",
        "post_backbone_token_permutation_used": False,
    }
    if (
        receipt.get("schema_version") != features.RECEIPT_SCHEMA
        or receipt.get("status") != "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED"
        or receipt.get("formal_training_authorized") is not False
        or receipt.get("paired_ground_truth_claimed") is not False
        or receipt.get("burned_development_only") is not True
        or receipt.get("implementation", {}).get("sha256") != EXTRACTOR_IMPLEMENTATION_SHA256
        or embedded_digest != _object_sha(unsigned)
        or receipt.get("population") != {
            "unique_base_clips": 644, "action_anchor_records": 644,
            "source_records": 0, "total_feature_records": 644,
            "view_evaluations_per_anchor": 5,
            "derived_views_are_independent_samples": False,
            "family_count": 28, "strict_true": 359, "strict_false": 285,
        }
        or receipt.get("exact6_shards") is not True
        or receipt.get("each_anchor_processor_call_count") != 1
        or receipt.get("each_anchor_independent_backbone_forward_count") != 5
        or receipt.get("feature_geometry") != {
            "views": list(features.VIEW_NAMES),
            "stored_sequence_per_view": [32, 1024],
            "teacher": "V-JEPA2 ViT-L fpc64 256 frozen FP16 skip_predictor",
            "post_backbone_token_permutation_used": False,
        }
        or receipt.get("sampling_and_transform_abi") != expected_sampling_abi
        or type(model_and_source) is not dict
        or model_and_source.get("model_files_before_and_after_exact") is not True
        or model_and_source.get("transformers_modules_before_and_after_exact") is not True
        or type(model_closure) is not dict or type(model_files) is not list
        or model_closure.get("model_repo") != features.MODEL_REPO
        or model_closure.get("model_revision") != features.MODEL_REVISION
        or model_closure.get("root_mode") != 0o555
        or model_closure.get("exact_top_level_regular_file_count") != 3
        or len(model_files) != 3
        or model_closure.get("closure_sha256") != features.object_sha256(model_files)
        or {
            row.get("relative_path"): {
                "sha256": row.get("sha256"), "size_bytes": row.get("size_bytes"),
                "mode": row.get("mode"), "nlink": row.get("nlink"),
            }
            for row in model_files if type(row) is dict
        } != {
            name: {**expected, "mode": 0o444, "nlink": 1}
            for name, expected in features.MODEL_FILES.items()
        }
        or type(module_closure) is not dict or type(module_rows) is not list
        or module_closure.get("transformers_version") != features.TRANSFORMERS_VERSION
        or len(module_rows) != len(features.TRANSFORMERS_MODULES)
        or module_closure.get("closure_sha256") != features.object_sha256(module_rows)
        or {
            row.get("module"): row.get("sha256")
            for row in module_rows if type(row) is dict
        } != features.TRANSFORMERS_MODULES
        or features.SHA_RE.fullmatch(
            str(receipt.get("exact644_ordered_iid_digest"))
        ) is None
        or features.SHA_RE.fullmatch(
            str(receipt.get("exact644_record_semantic_sha256"))
        ) is None
        or receipt.get("action_representation_qualified") is not False
        or receipt.get("scientific_confirmation_claimed") is not False
        or receipt.get("identity_disentanglement_qualified") is not False
        or receipt.get("identity_preservation_qualified") is not False
        or receipt.get("prior_generation_qualified") is not False
        or receipt.get("generation_qualified") is not False
        or receipt.get("video_editing_qualified") is not False
        or receipt.get("full644_refit_authorized") is not False
        or receipt.get("renderer_authorized") is not False
        or receipt.get("inference_authorized") is not False
        or receipt.get("web_evaluation_authorized") is not False
        or receipt.get("vae_necessary") is not None
    ):
        raise ValueError("v4-C feature receipt authority differs")
    manifest = receipt.get("manifest")
    if (
        type(manifest) is not dict
        or manifest.get("sha256") != features.FEATURE_MANIFEST_SHA256
        or manifest.get("manifest_digest") != features.FEATURE_MANIFEST_DIGEST
        or manifest.get("source_manifest_sha256") != features.SOURCE_MANIFEST_FILE_SHA256
        or manifest.get("source_manifest_digest") != features.SOURCE_MANIFEST_DIGEST
    ):
        raise ValueError("v4-C feature manifest binding differs")
    anchors, _ = features.load_anchor_manifest(
        Path(manifest["path"]), manifest["sha256"]
    )
    shard_rows = receipt.get("shards")
    if type(shard_rows) is not list or len(shard_rows) != 6:
        raise ValueError("v4-C feature shard receipt closure differs")
    by_ordinal: dict[int, Mapping[str, Any]] = {}
    for expected_index, shard in enumerate(shard_rows):
        expected_ordinals = [
            ordinal for ordinal in range(644) if ordinal % 6 == expected_index
        ]
        if (
            type(shard) is not dict or shard.get("index") != expected_index
            or shard.get("mode") != 0o444 or shard.get("nlink") != 1
            or shard.get("single_fd_pre_post_sha256_exact") is not True
            or shard.get("record_count") != len(expected_ordinals)
        ):
            raise ValueError("v4-C feature shard placement differs")
        payload, actual = features._load_sealed_shard(
            Path(shard["path"]), shard["sha256"]
        )
        for key in (
            "path", "sha256", "size_bytes", "mode", "nlink",
            "semantic_sha256", "single_fd_pre_post_sha256_exact",
        ):
            if actual[key] != shard[key]:
                raise ValueError("v4-C feature shard binding differs")
        payload_records = payload.get("records")
        if (
            payload.get("schema_version") != features.FEATURE_SCHEMA
            or payload.get("status")
            != "V4C_VJEPA2_ORDERED_CONTEXTUAL_SHARD_COMPLETE_BURNED_DEVELOPMENT"
            or payload.get("authority") != "feature_mechanics_diagnostic_only"
            or payload.get("formal_training_authorized") is not False
            or payload.get("paired_ground_truth_claimed") is not False
            or payload.get("implementation") != receipt.get("implementation")
            or payload.get("manifest_sha256") != features.FEATURE_MANIFEST_SHA256
            or payload.get("manifest_digest") != features.FEATURE_MANIFEST_DIGEST
            or payload.get("source_manifest_sha256") != features.SOURCE_MANIFEST_FILE_SHA256
            or payload.get("source_manifest_digest") != features.SOURCE_MANIFEST_DIGEST
            or payload.get("shard_index") != expected_index or payload.get("num_shards") != 6
            or payload.get("global_anchor_ordinals")
            != expected_ordinals
            or type(payload_records) is not list
            or payload.get("record_count") != len(expected_ordinals)
            or len(payload_records) != len(expected_ordinals)
            or payload.get("processor_call_count") != payload.get("record_count")
            or type(payload.get("record_count")) is not int
            or payload.get("frozen_backbone_forward_count") != 5 * payload.get("record_count")
            or payload.get("one_processor_then_exact5_separate_forwards_per_anchor") is not True
            or payload.get("model_forward_batching_across_views") is not False
            or payload.get("model_repo") != features.MODEL_REPO
            or payload.get("model_revision") != features.MODEL_REVISION
            or payload.get("model_dtype") != "torch.float16"
            or payload.get("skip_predictor") is not True
            or payload.get("model_and_source_closure") != receipt.get("model_and_source_closure")
            or payload.get("sampling_and_transform_abi") != receipt.get("sampling_and_transform_abi")
        ):
            raise ValueError("v4-C feature shard payload authority differs")
        for row, ordinal in zip(payload_records, expected_ordinals):
            if (
                type(row) is not dict or row.get("ordinal") != ordinal
                or ordinal in by_ordinal
            ):
                raise ValueError("v4-C feature record ordinal differs")
            features._validate_record(row, anchors[ordinal])
            by_ordinal[ordinal] = row
    if set(by_ordinal) != set(range(644)):
        raise ValueError("v4-C feature shards are not exact644 once each")
    rows = [by_ordinal[index] for index in range(644)]
    semantic = _object_sha([
        {"iid": row["iid"], "ordinal": row["ordinal"], "view_sequence_sha256": {
            name: _tensor_sha(row["view_sequences"][name]) for name in features.VIEW_NAMES
        }} for row in rows
    ])
    if semantic != receipt.get("exact644_record_semantic_sha256"):
        raise ValueError("v4-C exact644 feature semantic digest differs")
    if _object_sha([row["iid"] for row in rows]) != receipt.get(
        "exact644_ordered_iid_digest"
    ):
        raise ValueError("v4-C exact644 ordered IID digest differs")
    records = [Record(
        iid=row["iid"], family=row["family"],
        strict=row["strict_selection_gates_all_true"], views=row["view_sequences"],
    ) for row in rows]
    return records, receipt


def canonical_action(value: torch.Tensor) -> torch.Tensor:
    if type(value) is not torch.Tensor or tuple(value.shape) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("ordered V-JEPA2 sequence geometry differs")
    tensor = value.detach().to(dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("ordered V-JEPA2 sequence canonicalization differs")
    result = (tensor - tensor.mean(dim=0, keepdim=True)).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("ordered V-JEPA2 sequence canonicalization differs")
    # Keep the frozen FP32 canonical tensor bit-identical, but validate its
    # residual mean at the numerical scale of a 32-term FP32 reduction.  Two
    # real V-JEPA sequences (coordinates around magnitude 20) legitimately
    # leave about 3.25e-6 after subtracting the rounded FP32 mean, just above a
    # fixed 3e-6 cutoff.  Accumulating this validation statistic in float64
    # avoids another reduction error without changing any frontier input.
    residual = float(result.to(torch.float64).mean(dim=0).abs().max())
    scale = max(1.0, float(tensor.abs().max()))
    roundoff_bound = (TIME_STEPS + 2) * torch.finfo(torch.float32).eps * scale
    if residual > roundoff_bound:
        raise ValueError("ordered V-JEPA2 sequence canonicalization differs")
    return result


def _fit_frontier(rows: Sequence[Record], config: Config) -> FrontierFit:
    if len(rows) not in {644 - count for count in OOF_COUNTS}:
        raise ValueError("outer-train exact-original count differs")
    iids = [row.iid for row in rows]
    if len(set(iids)) != len(iids):
        raise ValueError("projection fit IIDs differ")
    values = torch.stack([canonical_action(row.views["original"]) for row in rows])
    max_frame_rank = max(config.payload_budgets) // TIME_STEPS
    max_clip_rank = max(config.payload_budgets)
    max_content_rank = max(config.payload_budgets) // config.tucker_time_rank
    tokens = values.reshape(-1, FEATURE_DIM)
    frame_mean = tokens.mean(dim=0, keepdim=True)
    feature_covariance = (tokens - frame_mean).T @ (tokens - frame_mean)
    content_basis = v4a._top_eigenbasis(feature_covariance, max_content_rank)
    frame_basis = content_basis[:, :max_frame_rank].contiguous()
    flat = values.flatten(1)
    clip_mean = flat.mean(dim=0, keepdim=True)
    clip_basis = v4a._fit_clip_basis(flat - clip_mean, max_clip_rank)
    temporal_covariance = torch.einsum("ntd,nsd->ts", values, values)
    temporal_basis = v4a._top_eigenbasis(temporal_covariance, config.tucker_time_rank)
    diagnostics = {
        "frame_basis_shape": list(frame_basis.shape),
        "frame_basis_sha256": _tensor_sha(frame_basis),
        "frame_basis_max_orthogonality_error": v4a._orthogonality_error(frame_basis),
        "clip_basis_shape": list(clip_basis.shape),
        "clip_basis_sha256": _tensor_sha(clip_basis),
        "clip_basis_max_orthogonality_error": v4a._orthogonality_error(clip_basis),
        "tucker_temporal_basis_shape": list(temporal_basis.shape),
        "tucker_temporal_basis_sha256": _tensor_sha(temporal_basis),
        "tucker_temporal_max_orthogonality_error": v4a._orthogonality_error(temporal_basis),
        "tucker_content_basis_shape": list(content_basis.shape),
        "tucker_content_basis_sha256": _tensor_sha(content_basis),
        "tucker_content_max_orthogonality_error": v4a._orthogonality_error(content_basis),
        "orthogonal_projection": True,
        "variance_whitening": False,
    }
    return FrontierFit(
        frame_mean=frame_mean.contiguous(), frame_basis=frame_basis,
        clip_mean=clip_mean.contiguous(), clip_basis=clip_basis,
        temporal_basis=temporal_basis, content_basis=content_basis,
        fit_iid_digest=_object_sha(iids), fit_input_sha256=_tensor_sha(values),
        diagnostics=diagnostics,
    )


def candidate_specs(config: Config) -> list[dict[str, Any]]:
    specs = []
    for payload in config.payload_budgets:
        for kind, temporal_rank, rank in (
            ("frame_pca", TIME_STEPS, payload // TIME_STEPS),
            ("clip_pca", 1, payload),
            ("tucker", config.tucker_time_rank, payload // config.tucker_time_rank),
        ):
            specs.append({
                "name": f"{kind}_b{payload:04d}_t{temporal_rank:02d}_r{rank:03d}",
                "kind": kind, "payload_numel": payload,
                "temporal_rank": temporal_rank, "feature_or_clip_rank": rank,
                "compression_ratio_vs_teacher": FULL_NUMEL / payload,
            })
    if len(specs) != 15:
        raise RuntimeError("v4-C candidate closure differs")
    return specs


def _encode(value: torch.Tensor, spec: Mapping[str, Any], fitted: FrontierFit) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or tuple(value.shape) != (TIME_STEPS, FEATURE_DIM)
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("v4-C canonical code input differs")
    kind, rank = spec["kind"], int(spec["feature_or_clip_rank"])
    if kind == "frame_pca":
        code = ((value - fitted.frame_mean) @ fitted.frame_basis[:, :rank]).flatten()
    elif kind == "clip_pca":
        code = ((value.flatten().unsqueeze(0) - fitted.clip_mean) @ fitted.clip_basis[:, :rank]).flatten()
    elif kind == "tucker":
        code = (
            fitted.temporal_basis.T @ (value - fitted.frame_mean)
            @ fitted.content_basis[:, :rank]
        ).flatten()
    else:
        raise ValueError("v4-C candidate kind differs")
    if code.numel() != spec["payload_numel"] or not bool(torch.isfinite(code).all()):
        raise ValueError("v4-C actual code payload differs")
    return code


def normalized_squared_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("v4-C paired code geometry differs")
    distance = (left - right).square().sum() / FULL_NUMEL
    if not bool(torch.isfinite(distance)):
        raise ValueError("v4-C distance is non-finite")
    return float(distance)


def _margin(query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> dict[str, float]:
    positive_distance = normalized_squared_distance(query, positive)
    negative_distance = normalized_squared_distance(query, negative)
    return {
        "positive_distance": positive_distance,
        "negative_distance": negative_distance,
        "margin": negative_distance - positive_distance,
    }


def _evaluate_fold(rows: Sequence[Record], fitted: FrontierFit, config: Config) -> list[dict[str, Any]]:
    specs = candidate_specs(config)
    output = []
    for row in rows:
        if set(row.views) != set(features.VIEW_NAMES):
            raise ValueError("v4-C precomputed view closure differs")
        views = {name: canonical_action(row.views[name]) for name in features.VIEW_NAMES}
        teacher_query = views["original"].flatten()
        teacher_positive = views[POSITIVE].flatten()
        teacher = {
            negative: _margin(teacher_query, teacher_positive, views[negative].flatten())
            for negative in NEGATIVES
        }
        candidates = {}
        for spec in specs:
            query = _encode(views["original"], spec, fitted)
            positive = _encode(views[POSITIVE], spec, fitted)
            candidates[spec["name"]] = {
                negative: _margin(query, positive, _encode(views[negative], spec, fitted))
                for negative in NEGATIVES
            }
        output.append({"iid": row.iid, "family": row.family, "teacher": teacher, "candidates": candidates})
    return output


def _strictly_positive(value: Mapping[str, Any]) -> bool:
    return bool(
        value["clip_paired_bootstrap"]["lcb"] > 0.0
        and value["family_cluster_paired_bootstrap"]["lcb"] > 0.0
    )


def _bootstrap(values: Sequence[float], families: Sequence[str], config: Config, label: str) -> dict[str, Any]:
    result = v4a._paired_bootstrap_lcbs(values, families, config, label)
    result["both_lcbs_strictly_gt_zero"] = _strictly_positive(result)
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]], config: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) != 644 or len({row["iid"] for row in rows}) != 644:
        raise ValueError("v4-C OOF aggregation is not exact644 once each")
    families = [row["family"] for row in rows]
    if len(set(families)) != 28:
        raise ValueError("v4-C OOF family closure differs")
    teacher = {
        negative: _bootstrap(
            [row["teacher"][negative]["margin"] for row in rows], families,
            config, f"v4c:teacher:{negative}:margin",
        ) for negative in NEGATIVES
    }
    candidates = {}
    for spec in candidate_specs(config):
        name = spec["name"]
        negative_results = {}
        for negative in NEGATIVES:
            teacher_values = [row["teacher"][negative]["margin"] for row in rows]
            candidate_values = [row["candidates"][name][negative]["margin"] for row in rows]
            retention_values = [
                candidate - config.teacher_retention * reference
                for candidate, reference in zip(candidate_values, teacher_values)
            ]
            candidate_stats = _bootstrap(
                candidate_values, families, config, f"v4c:{name}:{negative}:margin"
            )
            retention_stats = _bootstrap(
                retention_values, families, config,
                f"v4c:{name}:{negative}:minus:{config.teacher_retention}:teacher",
            )
            negative_results[negative] = {
                "teacher_margin": teacher[negative],
                "candidate_margin": candidate_stats,
                "candidate_minus_0p8_teacher_margin": retention_stats,
                "negative_gate": bool(
                    _strictly_positive(teacher[negative])
                    and _strictly_positive(candidate_stats)
                    and _strictly_positive(retention_stats)
                ),
            }
        candidates[name] = {
            "spec": spec, "negative_results": negative_results,
            "temporal_mechanics_gate": all(
                negative_results[negative]["negative_gate"] for negative in NEGATIVES
            ),
        }
    frontier = {}
    for payload in config.payload_budgets:
        names = [
            spec["name"] for spec in candidate_specs(config)
            if spec["payload_numel"] == payload
        ]
        frontier[str(payload)] = {
            "payload_numel": payload, "candidate_names": names,
            "exact_same_actual_code_numel": True,
            "qualified_candidates": [
                name for name in names if candidates[name]["temporal_mechanics_gate"]
            ],
            "oof_winner_selected": False,
            "candidates_listed_without_ranking": True,
            "across_negative_compensation_used": False,
        }
    return {"uncompressed_teacher": teacher, "candidates": candidates}, frontier


def _run_fold(
    records: Sequence[Record], assignment: Mapping[str, int], fold: int, config: Config,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit_rows = [row for row in records if assignment[row.iid] != fold]
    oof_rows = [row for row in records if assignment[row.iid] == fold]
    if len(oof_rows) != OOF_COUNTS[fold] or len(fit_rows) != 644 - OOF_COUNTS[fold]:
        raise ValueError("v4-C frozen outer fold counts differ")
    fitted = _fit_frontier(fit_rows, config)
    evaluated = _evaluate_fold(oof_rows, fitted, config)
    for row in evaluated:
        row["outer_fold"] = fold
    if [row["iid"] for row in evaluated] != [row.iid for row in oof_rows]:
        raise RuntimeError("v4-C OOF evaluation order differs")
    return {
        "fold_index": fold,
        "frozen_v4a_fold_iid_digest": FOLD_IID_DIGESTS[fold],
        "frozen_v4a_outer_assignment_digest": OUTER_ASSIGNMENT_DIGEST,
        "outer_train_original_count": len(fit_rows),
        "outer_train_iid_digest": _object_sha([row.iid for row in fit_rows]),
        "projection_fit_iid_digest": fitted.fit_iid_digest,
        "projection_fit_input_sha256": fitted.fit_input_sha256,
        "oof_original_count": len(oof_rows),
        "oof_iid_digest": _object_sha([row.iid for row in oof_rows]),
        "fit_oof_iid_disjoint": not bool({row.iid for row in fit_rows} & {row.iid for row in oof_rows}),
        "projection_fit_completed_before_oof_metric_evaluation": True,
        "projection_fit_original_view_only": True,
        "projection_fit_derived_view_count": 0,
        "projection_fit_family_or_transform_labels": False,
        "projection_fit_optimizer_steps": 0,
        "oof_used_for_projection_rank_or_winner_selection": False,
        "projection_diagnostics": fitted.diagnostics,
        "oof_evaluation_rows_sha256": _object_sha(evaluated),
    }, evaluated


def _compact_evidence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep exactly the paired fields needed to recompute every OOF gate."""

    output: list[dict[str, Any]] = []
    expected_names = [spec["name"] for spec in candidate_specs(Config())]
    for row in rows:
        if list(row["candidates"]) != expected_names:
            raise ValueError("v4-C evidence candidate order differs")
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
                    negative: float(row["candidates"][name][negative]["margin"])
                    for negative in NEGATIVES
                }
                for name in expected_names
            },
        })
    return output


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("frontier receipt must be a fresh absolute JSON child")
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        written = os.fstat(handle.fileno())
    os.chmod(path, 0o444)
    before = path.lstat()
    expected_sha = hashlib.sha256(raw).hexdigest()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        raw_before = handle.read()
        handle.seek(0)
        raw_after = handle.read()
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mode,
        item.st_nlink, item.st_mtime_ns, item.st_ctime_ns,
    )
    if (
        path.is_symlink() or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
        or before.st_size != len(raw) or raw_before != raw or raw_after != raw
        or hashlib.sha256(raw_before).hexdigest() != expected_sha
        or hashlib.sha256(raw_after).hexdigest() != expected_sha
        or (written.st_dev, written.st_ino, written.st_size)
        != (before.st_dev, before.st_ino, before.st_size)
        or identity(before) != identity(opened) or identity(opened) != identity(closed)
        or identity(closed) != identity(after)
    ):
        raise RuntimeError("frontier receipt seal/readback differs")
    return {
        "path": str(path.resolve(strict=True)), "sha256": expected_sha,
        "size_bytes": before.st_size, "mode": 0o444, "nlink": 1,
        "single_fd_pre_post_sha256_exact": True,
    }


def _assert_input_files_unchanged(
    feature_receipt: Mapping[str, Any], v4a_path: Path,
) -> None:
    _load_json_sealed(v4a_path, V4A_RECEIPT_FILE_SHA256)
    feature_receipt_path = Path(feature_receipt["feature_receipt_path"])
    _load_json_sealed(
        feature_receipt_path, str(feature_receipt["feature_receipt_file_sha256"])
    )
    for shard in feature_receipt["shards"]:
        path = Path(shard["path"])
        if (
            not path.is_absolute() or path.is_symlink()
            or str(path) != str(path.resolve(strict=True))
        ):
            raise RuntimeError("v4-C feature input path changed during command")
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or before.st_size != shard["size_bytes"]
        ):
            raise RuntimeError("v4-C feature input changed during frontier command")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            closed = os.fstat(handle.fileno())
        after = path.lstat()
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mode,
            item.st_nlink, item.st_mtime_ns, item.st_ctime_ns,
        )
        if (
            digest.hexdigest() != shard["sha256"]
            or identity(before) != identity(opened)
            or identity(opened) != identity(closed)
            or identity(closed) != identity(after)
        ):
            raise RuntimeError("v4-C feature input changed during frontier command")


def _config_value(config: Config) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "payload_budgets": list(config.payload_budgets),
        "tucker_time_rank": config.tucker_time_rank,
        "teacher_retention": config.teacher_retention,
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_alpha": config.bootstrap_alpha,
    }


def run_exact5(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-C frontier torch version differs")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    v4a_path = Path(args.v4a_receipt)
    assignment, v4a_receipt = load_frozen_v4a_split(
        v4a_path, args.expected_v4a_receipt_sha256
    )
    records, feature_receipt = load_v4c_features(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    exact_iids = [row.iid for row in records]
    if (
        len(records) != 644 or len(set(exact_iids)) != 644
        or set(exact_iids) != set(assignment)
        or _object_sha(assignment) != OUTER_ASSIGNMENT_DIGEST
        or tuple(sum(assignment[iid] == fold for iid in exact_iids) for fold in range(5))
        != OOF_COUNTS
    ):
        raise ValueError("v4-C features do not match frozen v4-A exact5 IIDs")
    v4a_evidence = v4a_receipt["oof_closure"]["embedded_paired_margin_evidence"]
    v4a_family = {row["iid"]: row["family"] for row in v4a_evidence}
    if len(v4a_family) != 644 or any(v4a_family[row.iid] != row.family for row in records):
        raise ValueError("v4-C feature families differ from frozen v4-A authority")
    records_by_iid = {row.iid: row for row in records}
    split_order_records = [records_by_iid[row["iid"]] for row in v4a_evidence]

    fold_receipts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        fold_receipt, evaluated = _run_fold(
            split_order_records, assignment, fold, config
        )
        upstream = v4a_receipt["folds"][fold]
        if (
            fold_receipt["oof_iid_digest"] != upstream["oof_iid_digest"]
            or fold_receipt["oof_original_count"] != upstream["oof_original_count"]
        ):
            raise ValueError("v4-C OOF order differs from frozen v4-A fold")
        fold_receipt["frozen_v4a_oof_iid_digest"] = upstream["oof_iid_digest"]
        fold_receipts.append(fold_receipt)
        all_rows.extend(evaluated)
    if (
        len(all_rows) != 644 or len({row["iid"] for row in all_rows}) != 644
        or {row["iid"] for row in all_rows} != set(exact_iids)
        or tuple(sum(row["outer_fold"] == fold for row in all_rows) for fold in range(5))
        != OOF_COUNTS
    ):
        raise ValueError("v4-C exact5 OOF union is not exact644 once each")

    metrics, frontier = _aggregate(all_rows, config)
    evidence = _compact_evidence(all_rows)
    spec_order = [spec["name"] for spec in candidate_specs(config)]
    qualified = [
        name for name in spec_order
        if metrics["candidates"][name]["temporal_mechanics_gate"]
    ]
    config_value = _config_value(config)
    feature_receipt_path = (
        Path(args.feature_root).resolve(strict=True) / "feature_extraction_receipt.json"
    )
    # These two fields are used only by the end-of-command unchanged check and
    # are part of the signed receipt so the exact bytes remain auditable.
    feature_receipt = dict(feature_receipt)
    feature_receipt["feature_receipt_path"] = str(feature_receipt_path)
    feature_receipt["feature_receipt_file_sha256"] = features._sha(
        args.expected_feature_receipt_sha256, "feature receipt SHA"
    )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "authority": "burned_development_feature_mechanics_diagnostic_only",
        "formal_training_authorized": False,
        "paired_ground_truth_claimed": False,
        "burned_development_only": True,
        "implementation": run_binding,
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "device": "cpu",
        "cpu_analytic_no_optimizer": True,
        "runtime": {
            "torch": str(torch.__version__),
            "torch_hip": str(torch.version.hip),
            "manual_seed": config.seed,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "feature_authority": {
            "feature_root": str(Path(args.feature_root).resolve(strict=True)),
            "feature_receipt_path": str(feature_receipt_path),
            "feature_receipt_sha256": args.expected_feature_receipt_sha256,
            "feature_receipt_digest": feature_receipt["receipt_digest"],
            "extractor_implementation_sha256": EXTRACTOR_IMPLEMENTATION_SHA256,
            "unique_original_base_clips": 644,
            "action_anchor_records": 644,
            "stored_views_per_anchor": 5,
            "family_count": 28,
            "exact644_ordered_iid_digest": _object_sha(exact_iids),
            "exact81_to_base64_indices_sha256": features.BASE64_INDICES_SHA256,
            "warp64_coordinates_sha256": features.WARP64_COORDINATES_SHA256,
        },
        "frozen_split": {
            "source": "pinned v4-A receipt embedded IID-to-outer-fold evidence",
            "v4a_receipt_path": str(v4a_path.resolve(strict=True)),
            "v4a_receipt_file_sha256": V4A_RECEIPT_FILE_SHA256,
            "v4a_receipt_digest": V4A_RECEIPT_DIGEST,
            "v4a_embedded_evidence_sha256": V4A_EVIDENCE_SHA256,
            "outer_assignment_digest": OUTER_ASSIGNMENT_DIGEST,
            "fold_iid_digests": {str(key): value for key, value in FOLD_IID_DIGESTS.items()},
            "oof_counts": list(OOF_COUNTS),
            "split_recomputed_from_vjepa_feature_values": False,
            "vjepa_energy_used_for_split_or_rank": False,
            "v2_energy_stratified_split_function_called": False,
            "all_exact644_are_development": True,
            "fresh_scientific_confirmation_claimed": False,
            "iid_disjoint_only_not_actor_scene_generator_lineage_disjoint": True,
        },
        "metric_contract": {
            "canonicalization": (
                "C(x)=x-mean_time(x) in FP32, shape [32,1024]; residual mean "
                "max_abs(mean_time(float64(C_fp32(x)))) validated against "
                "34*FP32_epsilon*max(1,max_abs(x)); "
                "FP64 validation does not change the FP32 output"
            ),
            "teacher_mapping": "R_infinity(C(x))=flatten(C(x))",
            "candidate_mapping": "fit-only orthogonal code coordinates",
            "formula": (
                "d(R(C(original)),R(C(negative)))"
                "-d(R(C(original)),R(C(monotone_warp)))"
            ),
            "same_query_and_same_mapping_within_each_margin": True,
            "self_reconstruction_metric_used": False,
            "distance": "sum_squared_code_difference/(32*1024)",
            "negative_views": list(NEGATIVES),
            "positive_view": POSITIVE,
            "all_five_views_are_separate_frozen_backbone_forwards": True,
            "views_transformed_at_canonical_processor_input_temporal_dim": True,
            "post_backbone_token_permutation_used": False,
            "derived_views_are_non_independent_and_not_samples": True,
            "original_sample_count": 644,
        },
        "projection_contract": {
            "families": ["frame_pca", "clip_pca", "tucker"],
            "analytic_no_optimizer": True,
            "orthogonal": True,
            "variance_whitening": False,
            "fit_outer_train_original_view_only": True,
            "fit_derived_view_count": 0,
            "fit_family_or_transform_labels": False,
            "oof_used_for_projection_fit_rank_or_winner_selection": False,
            "payload_definition": "actual per-clip code tensor numel",
            "payload_budgets": list(PAYLOAD_BUDGETS),
            "same_payload_comparisons_only": True,
            "cross_payload_ranking_performed": False,
            "oof_winner_selected": False,
            "qualified_list_order": "preregistered candidate spec order, never metric rank",
            "across_negative_compensation_used": False,
            "uncompressed_teacher_payload_numel": FULL_NUMEL,
        },
        "oof_access_contract": {
            "feature_artifacts_materialized_before_command": True,
            "same_process_fit_then_evaluate": True,
            "oof_tensors_supplied_to_projection_fit_function": False,
            "oof_metric_evaluator_called_only_after_fold_fit_returns": True,
            "claim_is_no_oof_projection_fit_or_selection_not_late_file_open": True,
        },
        "folds": fold_receipts,
        "oof_closure": {
            "unique_original_iids": 644,
            "each_original_evaluated_exactly_once": True,
            "counts_by_fold": list(OOF_COUNTS),
            "oof_ordered_iid_digest": _object_sha([row["iid"] for row in all_rows]),
            "oof_sorted_iid_digest": _object_sha(sorted(row["iid"] for row in all_rows)),
            "per_iid_evaluation_sha256": _object_sha(all_rows),
            "embedded_paired_margin_evidence_count": len(evidence),
            "embedded_paired_margin_evidence_sha256": _object_sha(evidence),
            "embedded_paired_margin_evidence": evidence,
            "embedded_fields_sufficient_to_recompute_all_bootstrap_gates": True,
        },
        "bootstrap_contract": {
            "implementation": "semantic_anchor_linear_frontier_v4_fast._paired_bootstrap_lcbs",
            "clip_paired_resampling_unit": "original_IID",
            "family_cluster_paired_resampling_unit": "exact28_equal_weight_family_means",
            "family_point_estimate": "macro mean of 28 per-family clip means",
            "paired_within_IID": True,
            "strict_gate": (
                "for every negative, teacher margin, candidate margin, and "
                "candidate-0.8*teacher margin both LCBs > 0"
            ),
        },
        "metrics": metrics,
        "same_payload_frontier": frontier,
        "qualified_temporal_mechanics_candidates": qualified,
        "qualification_scope": {
            "temporal_mechanics_qualified_candidates": qualified,
            "candidates_listed_without_ranking_or_selection": True,
            "training_authorized": False,
            "action_representation_qualified": False,
            "scientific_confirmation_claimed": False,
            "identity_disentanglement_qualified": False,
            "identity_preservation_qualified": False,
            "prior_generation_qualified": False,
            "generation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "full644_refit_authorized": False,
            "vae_necessary": None,
            "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    if _binding() != run_binding:
        raise RuntimeError("v4-C implementation/source binding changed during command")
    _assert_input_files_unchanged(feature_receipt, v4a_path)
    output_binding = _write_json_create_only(Path(args.output), receipt)
    if _binding() != run_binding:
        raise RuntimeError("v4-C implementation/source binding changed after receipt write")
    _assert_input_files_unchanged(feature_receipt, v4a_path)
    return {
        "receipt": output_binding["path"],
        "receipt_sha256": output_binding["sha256"],
        "receipt_digest": receipt["receipt_digest"],
        "qualified_temporal_mechanics_candidates": qualified,
        "qualified_candidates_are_not_selected_winners": True,
        "output_binding": output_binding,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact5 no-optimizer ordered V-JEPA2 analytic frontier"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-exact5")
    run.add_argument("--feature-root", required=True)
    run.add_argument("--expected-feature-receipt-sha256", required=True)
    run.add_argument("--v4a-receipt", required=True)
    run.add_argument("--expected-v4a-receipt-sha256", required=True)
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
