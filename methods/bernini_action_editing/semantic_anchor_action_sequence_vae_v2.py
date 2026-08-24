#!/usr/bin/env python3
"""Anchor-only full-DINO sequence AE/beta-VAE development experiment.

All exact644 rows are development data.  Five deterministic family/energy
stratified outer folds provide exploratory OOF estimates; every outer-train is
split again into model-fit and early-stop validation rows.  No result from
this program is a fresh scientific-confirmation result.

The literal target is ``temporal_center(anchor ordered DINO)`` with shape
``[32,768]``.  There is no source subtraction and no PCA target.  A single
fit-only global RMS is the only target normalization and is exactly invertible.
PCA is used only as a separately fitted linear baseline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from methods.bernini_action_editing import semantic_action_cvae_canary_v1 as authority


TRAIN_SCHEMA = "anchor-action-sequence-vae-oof-train-bundle-v2"
EVAL_SCHEMA = "anchor-action-sequence-vae-oof-eval-bundle-v2"
PREPARE_SCHEMA = "anchor-action-sequence-vae-oof-prepare-receipt-v2"
ARM_CHECKPOINT_SCHEMA = "anchor-action-sequence-vae-oof-arm-checkpoint-v2"
ARM_RECEIPT_SCHEMA = "anchor-action-sequence-vae-oof-arm-receipt-v2"
COMPARE_SCHEMA = "anchor-action-sequence-vae-oof-compare-receipt-v2"
AGGREGATE_SCHEMA = "anchor-action-sequence-vae-oof-aggregate-receipt-v2"
REFIT_BUNDLE_SCHEMA = "anchor-action-sequence-vae-full644-refit-bundle-v2"
REFIT_PREPARE_SCHEMA = "anchor-action-sequence-vae-full644-refit-prepare-receipt-v2"
REFIT_CHECKPOINT_SCHEMA = "anchor-action-sequence-vae-full644-refit-checkpoint-v2"
REFIT_RECEIPT_SCHEMA = "anchor-action-sequence-vae-full644-refit-receipt-v2"
ARMS = ("deterministic_ae", "direct_beta_vae")
RAW_TARGET_DEFINITION = "temporal_center(anchor ordered DINO full768)"
MODEL_COORDINATE_DEFINITION = "raw_anchor_target / model_fit_only_global_RMS"
OUTER_FOLDS = 5
INNER_FOLDS = 5
EPS = 1.0e-8
EARLY_STOP_MIN_DELTA = 1.0e-8

TRAIN_BUNDLE_KEYS = {
    "schema_version", "config", "config_sha256", "fold",
    "feature_receipt_sha256", "feature_receipt_digest", "implementation",
    "exact644_iid_digest", "exact644_raw_target_sha256",
    "exact644_population_authority",
    "raw_target_definition", "model_coordinate_definition", "global_rms",
    "global_rms_sha256", "global_rms_fit_only", "pca_is_model_target",
    "model_fit", "early_stop_validation", "baselines", "baseline_sha256",
}
EVAL_BUNDLE_KEYS = {
    "schema_version", "config", "config_sha256", "fold",
    "feature_receipt_sha256", "feature_receipt_digest", "implementation",
    "exact644_iid_digest", "exact644_raw_target_sha256",
    "exact644_population_authority",
    "raw_target_definition", "model_coordinate_definition", "global_rms",
    "global_rms_sha256", "global_rms_fit_only", "pca_is_model_target",
    "exploratory_oof", "source_features_present",
}
REFIT_BUNDLE_KEYS = {
    "schema_version", "config", "config_sha256", "aggregate_receipt_sha256",
    "preregistered_steps_by_arm", "raw_target_definition",
    "model_coordinate_definition", "global_rms", "global_rms_sha256",
    "global_rms_fit_only", "pca_is_model_target", "full644_originals",
    "full644_model_coordinate_sha256",
    "full644_frame_pca_rank_l_hard_baseline",
    "full644_frame_pca_rank_l_sha256", "authorized_arms",
    "model_fit_unique_originals", "held_rows", "derived_rows",
    "feature_receipt_sha256", "feature_receipt_digest", "exact644_iid_digest",
    "exact644_raw_target_sha256", "development_energy_definition",
    "exact644_population_authority",
    "development_energy_bin_edges_raw", "implementation",
}
FOLD_KEYS = {
    "outer_fold", "outer_folds", "inner_folds", "algorithm", "counts",
    "iid_digest", "train_iid_digest", "model_fit_iid_digest",
    "early_stop_validation_iid_digest", "exploratory_oof_iid_digest",
    "all_exact644_are_development", "fresh_confirmation_claimed",
    "disjointness", "energy_definition", "energy_quantiles_exact644",
    "fixed_energy_bin_edges_exact644", "outer_fold_energy_bin_counts",
    "outer_assignment_digest",
}


@dataclass(frozen=True)
class Config:
    seed: int = 20260819
    frame_hidden_dim: int = 128
    latent_dim: int = 32
    max_steps: int = 2000
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    beta_kl: float = 0.02
    kl_warmup_steps: int = 500
    eval_interval: int = 20
    patience_evals: int = 10

    def validate(self) -> None:
        if not 16 <= self.frame_hidden_dim <= 512:
            raise ValueError("frame_hidden_dim must be in [16,512]")
        if not 1 <= self.latent_dim <= 128:
            raise ValueError("latent_dim must be in [1,128]")
        if self.max_steps <= 0 or self.batch_size <= 0:
            raise ValueError("max_steps and batch_size must be positive")
        if not 0.0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate differs")
        if not 0.0 < self.beta_kl <= 1.0:
            raise ValueError("beta_kl must be in (0,1]")
        if not 1 <= self.kl_warmup_steps <= self.max_steps:
            raise ValueError("kl_warmup_steps differs")
        if self.eval_interval <= 0 or self.patience_evals <= 0:
            raise ValueError("early-stop configuration differs")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _object_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    """Hash tensor semantics, not its torch.save container representation."""

    if type(value) is not torch.Tensor:
        raise TypeError("tensor digest requires an exact Tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical({
        "dtype": str(tensor.dtype), "shape": list(tensor.shape),
    }))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _pca_state_sha(value: Mapping[str, torch.Tensor]) -> str:
    if type(value) is not dict or set(value) != {"mean", "basis"}:
        raise ValueError("PCA state keys differ")
    return _object_sha({key: _tensor_sha(value[key]) for key in sorted(value)})


def _file_sha(path: Path) -> str:
    return authority.file_sha256(path)


def _binding() -> dict[str, str]:
    implementation = Path(__file__).resolve(strict=True)
    dependency = Path(authority.__file__).resolve(strict=True)
    return {
        "implementation_path": str(implementation),
        "implementation_sha256": _file_sha(implementation),
        "authority_path": str(dependency),
        "authority_sha256": _file_sha(dependency),
    }


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("implementation/dependency changed during this command")


def _exact644_population_authority(
    pairs: Sequence[authority.PairRecord],
) -> dict[str, int]:
    value = {
        "unique_original_base_clips": len(pairs),
        "family_count": len({row.family for row in pairs}),
        "strict_true": sum(row.strict is True for row in pairs),
        "strict_false": sum(row.strict is False for row in pairs),
        "derived_rows": 0,
    }
    if value != {
        "unique_original_base_clips": 644,
        "family_count": 28,
        "strict_true": 359,
        "strict_false": 285,
        "derived_rows": 0,
    }:
        raise ValueError("exact644 population authority differs")
    return value


def _sha(value: Any, name: str) -> str:
    return authority._sha(value, name)


def _save_torch_create_only(path: Path, value: Any) -> str:
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return _file_sha(path)


def _write_json_create_only(path: Path, value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return hashlib.sha256(raw).hexdigest()


def _fresh_output(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("output must be a fresh absolute child")
    path.mkdir(mode=0o700)
    return path


def anchor_action_target(pair: authority.PairRecord) -> torch.Tensor:
    """Literal anchor-only target; source content is never referenced."""

    anchor = pair.anchor_sequence.detach().to(dtype=torch.float32, device="cpu")
    target = (anchor - anchor.mean(dim=0, keepdim=True)).contiguous()
    if tuple(target.shape) != (32, 768) or not bool(torch.isfinite(target).all()):
        raise ValueError("anchor-only action target differs")
    if not bool(torch.allclose(target.mean(dim=0), torch.zeros(768), atol=2.0e-6)):
        raise ValueError("anchor-only target is not temporally centered")
    return target


def _energy(pair: authority.PairRecord) -> float:
    return float(anchor_action_target(pair).square().mean())


def stratified_fold_assignment(
    rows: Sequence[authority.PairRecord], seed: int, fold_count: int
) -> dict[str, int]:
    """Family-stratified energy-rank round robin with hashed family offsets."""

    by_family: dict[str, list[authority.PairRecord]] = defaultdict(list)
    for row in rows:
        by_family[row.family].append(row)
    assignment: dict[str, int] = {}
    for family, family_rows in sorted(by_family.items()):
        ordered = sorted(
            family_rows,
            key=lambda row: (
                _energy(row),
                hashlib.sha256(f"{seed}:{family}:{row.iid}".encode()).hexdigest(),
            ),
        )
        offset = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest()[:8], 16)
        offset %= fold_count
        for rank, row in enumerate(ordered):
            assignment[row.iid] = (rank + offset) % fold_count
    if len(assignment) != len(rows):
        raise ValueError("fold assignment is not exhaustive")
    return assignment


def _split_fold(
    pairs: Sequence[authority.PairRecord], outer_fold: int, seed: int
) -> tuple[dict[str, list[authority.PairRecord]], dict[str, Any]]:
    if len(pairs) != 644 or len({row.iid for row in pairs}) != 644:
        raise ValueError("OOF input must be exact644")
    if not 0 <= outer_fold < OUTER_FOLDS:
        raise ValueError("outer_fold differs")
    outer_assignment = stratified_fold_assignment(pairs, seed, OUTER_FOLDS)
    energy_by_iid = {row.iid: _energy(row) for row in pairs}
    ordered_energy = torch.tensor(
        sorted(energy_by_iid.values()), dtype=torch.float64
    )
    quantile_levels = (0.10, 0.25, 0.50, 0.75, 0.90)
    energy_quantiles = {
        f"q{int(level * 100):02d}": float(torch.quantile(ordered_energy, level))
        for level in quantile_levels
    }
    fixed_edges = [
        float(torch.quantile(ordered_energy, level))
        for level in (0.20, 0.40, 0.60, 0.80)
    ]
    def energy_bin(value: float) -> int:
        return sum(value > edge for edge in fixed_edges)
    fold_energy_bin_counts = {
        str(fold): [
            sum(
                outer_assignment[row.iid] == fold
                and energy_bin(energy_by_iid[row.iid]) == bin_index
                for row in pairs
            )
            for bin_index in range(5)
        ]
        for fold in range(OUTER_FOLDS)
    }
    exploratory = [row for row in pairs if outer_assignment[row.iid] == outer_fold]
    outer_train = [row for row in pairs if outer_assignment[row.iid] != outer_fold]
    inner_assignment = stratified_fold_assignment(
        outer_train, seed + 1000 + outer_fold, INNER_FOLDS
    )
    early_stop = [row for row in outer_train if inner_assignment[row.iid] == 0]
    model_fit = [row for row in outer_train if inner_assignment[row.iid] != 0]
    groups = {
        "model_fit": model_fit,
        "early_stop_validation": early_stop,
        "exploratory_oof": exploratory,
    }
    ids = {name: [row.iid for row in rows] for name, rows in groups.items()}
    closure = [iid for values in ids.values() for iid in values]
    if len(closure) != 644 or len(set(closure)) != 644:
        raise ValueError("OOF fold closure differs")
    return groups, {
        "outer_fold": outer_fold,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "algorithm": "per-family energy-rank round-robin with hashed offset",
        "counts": {name: len(rows) for name, rows in groups.items()},
        "iid_digest": _object_sha(ids),
        "train_iid_digest": _object_sha({
            "model_fit": ids["model_fit"],
            "early_stop_validation": ids["early_stop_validation"],
        }),
        "model_fit_iid_digest": _object_sha(ids["model_fit"]),
        "early_stop_validation_iid_digest": _object_sha(
            ids["early_stop_validation"]
        ),
        "exploratory_oof_iid_digest": _object_sha(ids["exploratory_oof"]),
        "all_exact644_are_development": True,
        "fresh_confirmation_claimed": False,
        "disjointness": "IID_ONLY_NOT_ACTOR_SCENE_GENERATOR_LINEAGE_DISJOINT",
        "energy_definition": "mean(square(temporal_center(anchor ordered DINO)))",
        "energy_quantiles_exact644": energy_quantiles,
        "fixed_energy_bin_edges_exact644": fixed_edges,
        "outer_fold_energy_bin_counts": fold_energy_bin_counts,
        "outer_assignment_digest": _object_sha(outer_assignment),
    }


def _global_rms(rows: Sequence[authority.PairRecord]) -> torch.Tensor:
    values = torch.stack([anchor_action_target(row) for row in rows])
    rms = values.square().mean().sqrt().reshape(1)
    if not bool(torch.isfinite(rms).all()) or float(rms) <= 1.0e-8:
        raise ValueError("fit-only global RMS differs")
    return rms.contiguous()


def _tensor_rows(
    rows: Sequence[authority.PairRecord], global_rms: torch.Tensor
) -> dict[str, Any]:
    raw = torch.stack([anchor_action_target(row) for row in rows])
    standardized = (raw / global_rms).contiguous()
    return {"value": standardized, "iids": [row.iid for row in rows]}


def _fit_frame_pca(
    fit: torch.Tensor, rank: int
) -> dict[str, torch.Tensor]:
    tokens = fit.reshape(-1, 768)
    mean = tokens.mean(dim=0, keepdim=True)
    centered = tokens - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -rank:].flip(1).contiguous()
    return {"mean": mean.contiguous(), "basis": basis}


def _fit_clip_pca(fit: torch.Tensor, rank: int) -> dict[str, torch.Tensor]:
    flat = fit.flatten(1)
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    values = eigenvalues[-rank:].flip(0).clamp_min(1.0e-10)
    left = eigenvectors[:, -rank:].flip(1)
    basis = (centered.T @ left) / values.sqrt().unsqueeze(0)
    basis = torch.linalg.qr(basis, mode="reduced").Q.contiguous()
    return {"mean": mean.contiguous(), "basis": basis}


def prepare_fold(args: argparse.Namespace) -> dict[str, Any]:
    binding = _binding()
    config = Config(
        seed=args.seed,
        frame_hidden_dim=args.frame_hidden_dim,
        latent_dim=args.latent_dim,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_kl=args.beta_kl,
        kl_warmup_steps=args.kl_warmup_steps,
        eval_interval=args.eval_interval,
        patience_evals=args.patience_evals,
    )
    config.validate()
    pairs, feature_receipt = authority.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    population_authority = _exact644_population_authority(pairs)
    exact644_iid_digest = _object_sha([row.iid for row in pairs])
    exact644_raw_target_sha = _tensor_sha(torch.stack([
        anchor_action_target(row) for row in pairs
    ]))
    groups, split = _split_fold(pairs, args.fold_index, config.seed)
    rms = _global_rms(groups["model_fit"])
    fit = _tensor_rows(groups["model_fit"], rms)
    validation = _tensor_rows(groups["early_stop_validation"], rms)
    exploratory = _tensor_rows(groups["exploratory_oof"], rms)
    frame_pca = _fit_frame_pca(fit["value"], config.latent_dim)
    clip_pca = _fit_clip_pca(fit["value"], config.latent_dim)
    rms_sha = _tensor_sha(rms)
    baseline_sha = {
        "frame_pca_rank_l": _pca_state_sha(frame_pca),
        "clip_pca_rank_l": _pca_state_sha(clip_pca),
    }
    config_value = asdict(config)
    _assert_binding_unchanged(binding)
    output = _fresh_output(args.output)
    common = {
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "fold": split,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_iid_digest": exact644_iid_digest,
        "exact644_raw_target_sha256": exact644_raw_target_sha,
        "exact644_population_authority": population_authority,
        "implementation": binding,
        "raw_target_definition": RAW_TARGET_DEFINITION,
        "model_coordinate_definition": MODEL_COORDINATE_DEFINITION,
        "global_rms": rms,
        "global_rms_sha256": rms_sha,
        "global_rms_fit_only": True,
        "pca_is_model_target": False,
    }
    train_bundle = {
        "schema_version": TRAIN_SCHEMA,
        **common,
        "model_fit": fit,
        "early_stop_validation": validation,
        "baselines": {"frame_pca_rank_l": frame_pca, "clip_pca_rank_l": clip_pca},
        "baseline_sha256": baseline_sha,
    }
    eval_bundle = {
        "schema_version": EVAL_SCHEMA,
        **common,
        "exploratory_oof": exploratory,
        "source_features_present": False,
    }
    train_path = output / "train_bundle.pt"
    eval_path = output / "exploratory_oof_bundle.pt"
    train_sha = _save_torch_create_only(train_path, train_bundle)
    eval_sha = _save_torch_create_only(eval_path, eval_bundle)
    receipt: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "status": "OOF_FOLD_PREPARED_ALL_DATA_DEVELOPMENT",
        "scientific_confirmation_claimed": False,
        "all_exact644_are_development": True,
        "unique_original_base_clips": 644,
        "derived_rows": 0,
        "exact644_iid_digest": exact644_iid_digest,
        "exact644_raw_target_sha256": exact644_raw_target_sha,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_population_authority": population_authority,
        "prior_locked_partition_rows_burned": 96,
        "exact644_role": "BURNED_DEVELOPMENT_ONLY",
        "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        "confirmation_evaluations_allowed_by_this_runtime": 0,
        "family_labels_used_only_for_stratified_split": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "direct_target_not_residual": True,
        "rgb_or_wan_reconstruction_performed": False,
        "source_identity_preservation_tested": False,
        "raw_target_definition": RAW_TARGET_DEFINITION,
        "model_coordinate_definition": MODEL_COORDINATE_DEFINITION,
        "source_subtracted": False,
        "source_features_used_by_target_or_model": False,
        "full_768_target": True,
        "pca_target_used": False,
        "normalization": "single model-fit-only global RMS; reversible multiply",
        "config": config_value,
        "config_sha256": common["config_sha256"],
        "fold": split,
        "global_rms": float(rms),
        "global_rms_sha256": rms_sha,
        "baseline_sha256": baseline_sha,
        "train_bundle": {
            "path": str(train_path.resolve()), "sha256": train_sha,
            "size_bytes": train_path.stat().st_size,
            "contains_exploratory_oof": False,
        },
        "exploratory_oof_bundle": {
            "path": str(eval_path.resolve()), "sha256": eval_sha,
            "size_bytes": eval_path.stat().st_size,
            "contains_model_fit_or_early_stop_values": False,
        },
        "implementation": binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "prepare_receipt.json"
    receipt_sha = _write_json_create_only(receipt_path, receipt)
    _assert_binding_unchanged(binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()), "receipt_sha256": receipt_sha,
        "train_bundle_sha256": train_sha, "exploratory_oof_bundle_sha256": eval_sha,
        "fold_counts": split["counts"],
    }


class SequenceCore(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden = config.frame_hidden_dim
        latent = config.latent_dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, 2 * latent, kernel_size=3, padding=1),
        )
        self.frame_decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.temporal_decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, 768, kernel_size=3, padding=1),
        )

    def encode_heads(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.frame_encoder(value).transpose(1, 2)
        encoded = self.temporal_encoder(hidden).transpose(1, 2)
        return encoded.chunk(2, dim=2)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.frame_decoder(latent).transpose(1, 2)
        reconstruction = self.temporal_decoder(hidden).transpose(1, 2)
        # The raw target is temporally centered by construction.  Enforcing the
        # same exact constraint prevents impossible DC energy at evaluation.
        return reconstruction - reconstruction.mean(dim=1, keepdim=True)


class DeterministicSequenceAE(SequenceCore):
    def forward(self, value: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        del sample
        first, second = self.encode_heads(value)
        latent = (first + second) / math.sqrt(2.0)
        return {"latent": latent, "reconstruction": self.decode(latent)}


class DirectSequenceBetaVAE(SequenceCore):
    def forward(self, value: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        mean, logvar = self.encode_heads(value)
        logvar = logvar.clamp(min=-12.0, max=8.0)
        latent = mean
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return {
            "latent": latent,
            "mean": mean,
            "logvar": logvar,
            "reconstruction": self.decode(latent),
        }


def _make_model(arm: str, config: Config) -> nn.Module:
    if arm == "deterministic_ae":
        return DeterministicSequenceAE(config)
    if arm == "direct_beta_vae":
        return DirectSequenceBetaVAE(config)
    raise ValueError("arm differs")


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def kl_element_mean(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL averaged over batch, time and latent coordinates; never a latent sum."""

    element = 0.5 * (mean.square() + logvar.exp() - logvar - 1.0)
    return element.mean()


def kl_weight(step: int, config: Config) -> float:
    return config.beta_kl * min(1.0, step / config.kl_warmup_steps)


def _loss(
    arm: str,
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    step: int,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = F.mse_loss(output["reconstruction"], target)
    kl = torch.zeros((), device=target.device)
    if arm == "direct_beta_vae":
        kl = kl_element_mean(output["mean"], output["logvar"])
    weight = kl_weight(step, config) if arm == "direct_beta_vae" else 0.0
    total = reconstruction + weight * kl
    return total, {
        "reconstruction": float(reconstruction.detach()),
        "kl_element_mean": float(kl.detach()),
        "effective_beta": weight,
        "total": float(total.detach()),
    }


@torch.no_grad()
def _validation_mse(model: nn.Module, value: torch.Tensor, device: torch.device) -> float:
    model.eval()
    target = value.to(device)
    return float(F.mse_loss(model(target, sample=False)["reconstruction"], target))


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def train_with_early_stop(
    arm: str,
    model: nn.Module,
    fit: torch.Tensor,
    validation: torch.Tensor,
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, float]], int, int, str]:
    model.to(device)
    fit = fit.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4
    )
    schedule = torch.Generator(device="cpu").manual_seed(config.seed + 200)
    schedule_digest = hashlib.sha256()
    history: list[dict[str, float]] = []
    best_mse = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_evals = 0
    for step in range(1, config.max_steps + 1):
        model.train()
        indices_cpu = torch.randint(
            len(fit), (min(config.batch_size, len(fit)),), generator=schedule
        )
        schedule_digest.update(indices_cpu.numpy().tobytes(order="C"))
        target = fit[indices_cpu.to(device)]
        output = model(target, sample=True)
        total, metrics = _loss(arm, output, target, step, config)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % config.eval_interval == 0 or step == config.max_steps:
            val_mse = _validation_mse(model, validation, device)
            history.append({"step": step, **metrics, "validation_mse": val_mse})
            if val_mse < best_mse - EARLY_STOP_MIN_DELTA:
                best_mse = val_mse
                best_step = step
                best_state = _cpu_state(model)
                stale_evals = 0
            else:
                stale_evals += 1
            if stale_evals >= config.patience_evals:
                break
    if best_state is None or best_step <= 0:
        raise RuntimeError("early stopping failed to select a checkpoint")
    model.load_state_dict(best_state, strict=True)
    return history, best_step, step, schedule_digest.hexdigest()


def _require_sealed(path: Path) -> None:
    stat = path.stat()
    if stat.st_nlink != 1 or (stat.st_mode & 0o777) != 0o444:
        raise ValueError("artifact must be mode0444/nlink1")


def _load_torch(path: Path, expected_sha: str, expected_size: int | None = None) -> Any:
    path = path.resolve(strict=True)
    _require_sealed(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError("artifact size differs")
    if _file_sha(path) != _sha(expected_sha, "artifact SHA"):
        raise ValueError("artifact SHA differs")
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_receipt(path: Path, expected_sha: str, schema: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    _require_sealed(path)
    if _file_sha(path) != _sha(expected_sha, "receipt SHA"):
        raise ValueError("receipt SHA differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != schema:
        raise ValueError("receipt schema differs")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if _object_sha(unsigned) != _sha(digest, "receipt digest"):
        raise ValueError("receipt self digest differs")
    return value


def _device(value: str) -> torch.device:
    if value != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("production arm requires exactly one logical cuda:0")
    if torch.cuda.get_device_name(0) != "AMD Instinct MI210":
        raise RuntimeError("GPU must be AMD Instinct MI210")
    return torch.device(value)


def _validate_sequence_rows(value: Mapping[str, Any], count: int, name: str) -> None:
    if type(value) is not dict or set(value) != {"value", "iids"}:
        raise ValueError(f"{name} keys differ")
    tensor = value.get("value")
    iids = value.get("iids")
    if (
        type(tensor) is not torch.Tensor
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != (count, 32, 768)
    ):
        raise ValueError(f"{name} shape differs")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} is non-finite")
    if not bool(torch.allclose(
        tensor.mean(dim=1), torch.zeros(count, 768), atol=3.0e-5, rtol=0.0
    )):
        raise ValueError(f"{name} is not temporally centered")
    if (
        type(iids) is not list
        or any(type(iid) is not str or not iid for iid in iids)
        or len(iids) != count
        or len(set(iids)) != count
    ):
        raise ValueError(f"{name} IID closure differs")


def _validate_fold(value: Any) -> None:
    if type(value) is not dict or set(value) != FOLD_KEYS:
        raise ValueError("fold keys differ")
    if value["outer_folds"] != OUTER_FOLDS or value["inner_folds"] != INNER_FOLDS:
        raise ValueError("fold cardinality differs")
    if type(value["outer_fold"]) is not int or not 0 <= value["outer_fold"] < OUTER_FOLDS:
        raise ValueError("outer fold differs")
    counts = value["counts"]
    expected_count_keys = {"model_fit", "early_stop_validation", "exploratory_oof"}
    if (
        type(counts) is not dict
        or set(counts) != expected_count_keys
        or any(type(count) is not int or count <= 0 for count in counts.values())
        or sum(counts.values()) != 644
    ):
        raise ValueError("fold counts differ")
    for key in (
        "iid_digest", "train_iid_digest", "model_fit_iid_digest",
        "early_stop_validation_iid_digest", "exploratory_oof_iid_digest",
        "outer_assignment_digest",
    ):
        _sha(value[key], key)
    if value["all_exact644_are_development"] is not True:
        raise ValueError("exact644 development status differs")
    if value["fresh_confirmation_claimed"] is not False:
        raise ValueError("fold cannot claim fresh confirmation")
    if value["disjointness"] != "IID_ONLY_NOT_ACTOR_SCENE_GENERATOR_LINEAGE_DISJOINT":
        raise ValueError("fold disjointness claim differs")
    if value["energy_definition"] != "mean(square(temporal_center(anchor ordered DINO)))":
        raise ValueError("energy definition differs")
    quantiles = value["energy_quantiles_exact644"]
    if type(quantiles) is not dict or set(quantiles) != {"q10", "q25", "q50", "q75", "q90"}:
        raise ValueError("energy quantile keys differ")
    quantile_values = list(quantiles.values())
    edges = value["fixed_energy_bin_edges_exact644"]
    if (
        any(type(item) is not float or not math.isfinite(item) or item < 0.0 for item in quantile_values)
        or quantile_values != sorted(quantile_values)
        or type(edges) is not list
        or len(edges) != 4
        or any(type(item) is not float or not math.isfinite(item) or item < 0.0 for item in edges)
        or edges != sorted(edges)
    ):
        raise ValueError("energy statistics differ")
    bins = value["outer_fold_energy_bin_counts"]
    if type(bins) is not dict or set(bins) != {str(index) for index in range(OUTER_FOLDS)}:
        raise ValueError("energy bin fold keys differ")
    if any(
        type(row) is not list or len(row) != 5
        or any(type(count) is not int or count < 0 for count in row)
        for row in bins.values()
    ):
        raise ValueError("energy bin counts differ")
    if sum(sum(row) for row in bins.values()) != 644:
        raise ValueError("energy bin population differs")
    if sum(bins[str(value["outer_fold"])]) != counts["exploratory_oof"]:
        raise ValueError("outer-fold energy count differs")


def _validate_common_bundle(
    bundle: Mapping[str, Any], prepare: Mapping[str, Any], expected_keys: set[str],
    expected_implementation: Mapping[str, str],
) -> Config:
    if type(bundle) is not dict or set(bundle) != expected_keys:
        raise ValueError("bundle exact-key allowlist differs")
    if type(bundle["config"]) is not dict or set(bundle["config"]) != set(asdict(Config())):
        raise ValueError("bundle config keys differ")
    config = Config(**bundle["config"])
    config.validate()
    if _object_sha(bundle["config"]) != _sha(bundle["config_sha256"], "config SHA"):
        raise ValueError("bundle config self digest differs")
    if bundle["config"] != prepare["config"] or bundle["config_sha256"] != prepare["config_sha256"]:
        raise ValueError("bundle/prepare config differs")
    _validate_fold(bundle["fold"])
    if bundle["fold"] != prepare["fold"]:
        raise ValueError("bundle/prepare fold differs")
    for key in (
        "feature_receipt_sha256", "feature_receipt_digest",
        "exact644_iid_digest", "exact644_raw_target_sha256",
    ):
        _sha(bundle[key], key)
        if bundle[key] != prepare[key]:
            raise ValueError(f"bundle/prepare {key} differs")
    if bundle["exact644_population_authority"] != {
        "unique_original_base_clips": 644,
        "family_count": 28,
        "strict_true": 359,
        "strict_false": 285,
        "derived_rows": 0,
    } or bundle["exact644_population_authority"] != prepare["exact644_population_authority"]:
        raise ValueError("bundle exact644 population authority differs")
    if (
        bundle["implementation"] != prepare["implementation"]
        or bundle["implementation"] != expected_implementation
    ):
        raise ValueError("bundle implementation pin differs")
    if bundle["raw_target_definition"] != RAW_TARGET_DEFINITION:
        raise ValueError("raw target definition differs")
    if bundle["model_coordinate_definition"] != MODEL_COORDINATE_DEFINITION:
        raise ValueError("model coordinate definition differs")
    if prepare["raw_target_definition"] != RAW_TARGET_DEFINITION or prepare["model_coordinate_definition"] != MODEL_COORDINATE_DEFINITION:
        raise ValueError("prepare target definition differs")
    rms = bundle["global_rms"]
    if (
        type(rms) is not torch.Tensor
        or rms.dtype != torch.float32
        or tuple(rms.shape) != (1,)
        or not bool(torch.isfinite(rms).all())
        or float(rms) <= EPS
        or _tensor_sha(rms) != _sha(bundle["global_rms_sha256"], "RMS SHA")
        or bundle["global_rms_sha256"] != prepare["global_rms_sha256"]
        or not math.isclose(float(rms), prepare["global_rms"], rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("global RMS authority differs")
    if bundle["global_rms_fit_only"] is not True or bundle["pca_is_model_target"] is not False:
        raise ValueError("target preprocessing contract differs")
    return config


def _validate_pca_state(
    state: Any, mean_shape: tuple[int, ...], basis_shape: tuple[int, ...], name: str
) -> None:
    if type(state) is not dict or set(state) != {"mean", "basis"}:
        raise ValueError(f"{name} state keys differ")
    mean, basis = state["mean"], state["basis"]
    if (
        type(mean) is not torch.Tensor or mean.dtype != torch.float32
        or tuple(mean.shape) != mean_shape or not bool(torch.isfinite(mean).all())
        or type(basis) is not torch.Tensor or basis.dtype != torch.float32
        or tuple(basis.shape) != basis_shape or not bool(torch.isfinite(basis).all())
    ):
        raise ValueError(f"{name} tensor geometry differs")
    identity = torch.eye(basis_shape[1], dtype=torch.float32)
    if not bool(torch.allclose(basis.T @ basis, identity, atol=2.0e-4, rtol=2.0e-4)):
        raise ValueError(f"{name} basis is not orthonormal")


def _load_train_bundle_against_prepare(
    args: argparse.Namespace, prepare: Mapping[str, Any],
    expected_implementation: Mapping[str, str],
) -> tuple[dict[str, Any], Config]:
    binding = prepare.get("train_bundle")
    if type(binding) is not dict or set(binding) != {
        "path", "sha256", "size_bytes", "contains_exploratory_oof"
    }:
        raise ValueError("prepare train-bundle binding differs")
    if binding["contains_exploratory_oof"] is not False:
        raise ValueError("train bundle cannot contain exploratory OOF values")
    path = Path(args.train_bundle).resolve(strict=True)
    if Path(binding["path"]).resolve(strict=True) != path:
        raise ValueError("train bundle path differs")
    expected_sha = _sha(args.expected_train_bundle_sha256, "train bundle SHA")
    if binding["sha256"] != expected_sha:
        raise ValueError("train bundle CLI/prepare SHA differs")
    if type(binding["size_bytes"]) is not int or binding["size_bytes"] <= 0:
        raise ValueError("train bundle size binding differs")
    bundle = _load_torch(path, expected_sha, binding["size_bytes"])
    if bundle.get("schema_version") != TRAIN_SCHEMA:
        raise ValueError("train schema differs")
    config = _validate_common_bundle(
        bundle, prepare, TRAIN_BUNDLE_KEYS, expected_implementation
    )
    counts = bundle["fold"]["counts"]
    _validate_sequence_rows(bundle["model_fit"], counts["model_fit"], "model_fit")
    _validate_sequence_rows(
        bundle["early_stop_validation"], counts["early_stop_validation"],
        "early_stop_validation",
    )
    fit_iids = bundle["model_fit"]["iids"]
    validation_iids = bundle["early_stop_validation"]["iids"]
    if set(fit_iids) & set(validation_iids):
        raise ValueError("model-fit/early-stop IIDs overlap")
    fold = bundle["fold"]
    if (
        _object_sha(fit_iids) != fold["model_fit_iid_digest"]
        or _object_sha(validation_iids) != fold["early_stop_validation_iid_digest"]
        or _object_sha({
            "model_fit": fit_iids,
            "early_stop_validation": validation_iids,
        }) != fold["train_iid_digest"]
    ):
        raise ValueError("train IID digest closure differs")
    fit_rms = bundle["model_fit"]["value"].square().mean().sqrt()
    if not bool(torch.isclose(fit_rms, torch.ones_like(fit_rms), atol=3.0e-6, rtol=3.0e-6)):
        raise ValueError("model-fit standardized RMS differs")
    baselines = bundle["baselines"]
    if type(baselines) is not dict or set(baselines) != {
        "frame_pca_rank_l", "clip_pca_rank_l"
    }:
        raise ValueError("baseline keys differ")
    _validate_pca_state(
        baselines["frame_pca_rank_l"], (1, 768), (768, config.latent_dim),
        "frame PCA",
    )
    _validate_pca_state(
        baselines["clip_pca_rank_l"], (1, 32 * 768),
        (32 * 768, config.latent_dim), "clip PCA",
    )
    expected_baseline_sha = {
        name: _pca_state_sha(state) for name, state in baselines.items()
    }
    if (
        type(bundle["baseline_sha256"]) is not dict
        or set(bundle["baseline_sha256"]) != set(baselines)
        or bundle["baseline_sha256"] != expected_baseline_sha
        or bundle["baseline_sha256"] != prepare["baseline_sha256"]
    ):
        raise ValueError("baseline tensor digest differs")
    return bundle, config


def _load_eval_bundle_against_prepare(
    args: argparse.Namespace, prepare: Mapping[str, Any],
    expected_implementation: Mapping[str, str],
) -> tuple[dict[str, Any], Config]:
    binding = prepare.get("exploratory_oof_bundle")
    if type(binding) is not dict or set(binding) != {
        "path", "sha256", "size_bytes", "contains_model_fit_or_early_stop_values"
    }:
        raise ValueError("prepare OOF-bundle binding differs")
    if binding["contains_model_fit_or_early_stop_values"] is not False:
        raise ValueError("OOF bundle cannot contain training values")
    path = Path(args.exploratory_oof_bundle).resolve(strict=True)
    if Path(binding["path"]).resolve(strict=True) != path:
        raise ValueError("OOF bundle path differs")
    expected_sha = _sha(
        args.expected_exploratory_oof_bundle_sha256, "OOF bundle SHA"
    )
    if binding["sha256"] != expected_sha:
        raise ValueError("OOF bundle CLI/prepare SHA differs")
    bundle = _load_torch(path, expected_sha, binding["size_bytes"])
    if bundle.get("schema_version") != EVAL_SCHEMA:
        raise ValueError("OOF bundle schema differs")
    config = _validate_common_bundle(
        bundle, prepare, EVAL_BUNDLE_KEYS, expected_implementation
    )
    if bundle["source_features_present"] is not False:
        raise ValueError("source features unexpectedly present")
    count = bundle["fold"]["counts"]["exploratory_oof"]
    _validate_sequence_rows(bundle["exploratory_oof"], count, "exploratory_oof")
    if _object_sha(bundle["exploratory_oof"]["iids"]) != bundle["fold"]["exploratory_oof_iid_digest"]:
        raise ValueError("OOF IID digest differs")
    return bundle, config


def train_fold_arm(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    prepare = _load_receipt(
        Path(args.prepare_receipt), args.expected_prepare_receipt_sha256, PREPARE_SCHEMA
    )
    bundle, config = _load_train_bundle_against_prepare(
        args, prepare, run_binding
    )
    if args.fold_index != bundle["fold"]["outer_fold"]:
        raise ValueError("train CLI fold index differs")
    if args.arm not in ARMS:
        raise ValueError("arm differs")
    with torch.random.fork_rng(devices=[]):
        ae_count = _parameter_count(DeterministicSequenceAE(config))
        vae_count = _parameter_count(DirectSequenceBetaVAE(config))
    if ae_count != vae_count:
        raise RuntimeError("arm parameter counts differ")
    device = _device(args.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = _make_model(args.arm, config)
    history, best_step, executed_steps, schedule_digest = train_with_early_stop(
        args.arm,
        model,
        bundle["model_fit"]["value"],
        bundle["early_stop_validation"]["value"],
        config,
        device,
    )
    best_validation_mse = _validation_mse(
        model, bundle["early_stop_validation"]["value"], device
    )
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    checkpoint = {
        "schema_version": ARM_CHECKPOINT_SCHEMA,
        "arm": args.arm,
        "config": asdict(config),
        "config_sha256": bundle["config_sha256"],
        "fold": bundle["fold"],
        "train_bundle_sha256": args.expected_train_bundle_sha256,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "best_step": best_step,
        "executed_steps": executed_steps,
        "model_state": _cpu_state(model),
        "implementation": run_binding,
    }
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_sha = _save_torch_create_only(checkpoint_path, checkpoint)
    receipt: dict[str, Any] = {
        "schema_version": ARM_RECEIPT_SCHEMA,
        "status": "OOF_ARM_EARLY_STOPPED_NOT_YET_EXPLORATORY_EVALUATED",
        "all_data_status": "BURNED_DEVELOPMENT_ONLY",
        "prior_locked_partition_rows_burned": 96,
        "exact644_role": "BURNED_DEVELOPMENT_ONLY",
        "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        "confirmation_evaluations_allowed_by_this_runtime": 0,
        "family_labels_used_only_for_stratified_split": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "direct_target_not_residual": True,
        "rgb_or_wan_reconstruction_performed": False,
        "source_identity_preservation_tested": False,
        "arm": args.arm,
        "fold": bundle["fold"],
        "target": {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
        },
        "config": asdict(config),
        "config_sha256": bundle["config_sha256"],
        "parameter_count": _parameter_count(model),
        "matched_parameter_count": True,
        "kl_reduction": "mean_over_batch_time_latent_elements",
        "kl_warmup": True,
        "best_step": best_step,
        "executed_steps": executed_steps,
        "best_validation_mse": best_validation_mse,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "training_history": history,
        "executed_minibatch_schedule_sha256": schedule_digest,
        "exploratory_oof_values_read_by_training": False,
        "train_bundle_sha256": args.expected_train_bundle_sha256,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json_create_only(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()), "receipt_sha256": receipt_sha,
        "checkpoint_sha256": checkpoint_sha, "best_step": best_step,
    }


def _reconstruct_frame_pca(value: torch.Tensor, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mean = state["mean"]
    basis = state["basis"]
    reconstruction = ((value - mean) @ basis) @ basis.T + mean
    return reconstruction - reconstruction.mean(dim=1, keepdim=True)


def _reconstruct_clip_pca(value: torch.Tensor, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    flat = value.flatten(1)
    mean = state["mean"]
    basis = state["basis"]
    reconstruction = ((flat - mean) @ basis) @ basis.T + mean
    reconstruction = reconstruction.reshape_as(value)
    return reconstruction - reconstruction.mean(dim=1, keepdim=True)


@torch.no_grad()
def _predict(model: nn.Module, value: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval().to(device)
    return model(value.to(device), sample=False)["reconstruction"].cpu()


def _metric_rows(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    iids: Sequence[str],
    global_rms: torch.Tensor,
    energy_edges_raw: Sequence[float],
) -> dict[str, Any]:
    if tuple(target.shape) != tuple(reconstruction.shape):
        raise ValueError("metric geometry differs")
    row_mse = (reconstruction - target).square().mean(dim=(1, 2))
    zero_mse = target.square().mean(dim=(1, 2))
    raw_row_mse = row_mse * global_rms.square()
    raw_zero_mse = zero_mse * global_rms.square()
    cosine = F.cosine_similarity(reconstruction.flatten(1), target.flatten(1), dim=1)
    target_delta = target[:, 1:] - target[:, :-1]
    reconstruction_delta = reconstruction[:, 1:] - reconstruction[:, :-1]
    delta_mse = (reconstruction_delta - target_delta).square().mean(dim=(1, 2))
    raw_delta_mse = delta_mse * global_rms.square()
    output_energy_raw = reconstruction.square().mean(dim=(1, 2)) * global_rms.square()
    raw_energy = zero_mse * global_rms.square()
    bins = torch.tensor(
        [sum(float(energy) > edge for edge in energy_edges_raw) for energy in raw_energy],
        dtype=torch.long,
    )
    by_energy = {}
    for index in range(5):
        mask = bins == index
        by_energy[str(index)] = {
            "count": int(mask.sum()),
            "mse": float(row_mse[mask].mean()) if bool(mask.any()) else None,
            "zero_mse": float(zero_mse[mask].mean()) if bool(mask.any()) else None,
            "raw_mse": float(raw_row_mse[mask].mean()) if bool(mask.any()) else None,
            "raw_target_energy": float(raw_zero_mse[mask].mean()) if bool(mask.any()) else None,
            "raw_output_energy": float(output_energy_raw[mask].mean()) if bool(mask.any()) else None,
            "raw_mse_ratio_vs_zero": (
                float(raw_row_mse[mask].mean() / raw_zero_mse[mask].mean().clamp_min(EPS))
                if bool(mask.any()) else None
            ),
            "cosine": float(cosine[mask].mean()) if bool(mask.any()) else None,
        }
    return {
        "count": len(target),
        "model_coordinate_mse": float(row_mse.mean()),
        "raw_anchor_feature_mse": float(row_mse.mean() * global_rms.square()),
        "zero_model_coordinate_mse": float(zero_mse.mean()),
        "normalized_mse_vs_zero": float(
            row_mse.mean() / zero_mse.mean().clamp_min(EPS)
        ),
        "cosine": float(cosine.mean()),
        "temporal_delta_mse": float(delta_mse.mean()),
        "temporal_mean_abs_max": float(reconstruction.mean(dim=1).abs().max()),
        "energy_strata": by_energy,
        "per_iid": [
            {
                "iid": iid,
                "mse": float(mse),
                "zero_mse": float(zero),
                "raw_mse": float(raw_mse),
                "raw_zero_mse": float(raw_zero),
                "raw_output_energy": float(output_energy),
                "cosine": float(cos),
                "temporal_delta_mse": float(delta),
                "raw_temporal_delta_mse": float(raw_delta),
                "energy_bin": int(bin_index),
            }
            for iid, mse, zero, raw_mse, raw_zero, output_energy, cos, delta, raw_delta, bin_index in zip(
                iids, row_mse, zero_mse, raw_row_mse, raw_zero_mse,
                output_energy_raw, cosine, delta_mse, raw_delta_mse, bins
            )
        ],
    }


def _paired_ratio(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    seed: int,
    draws: int = 10000,
    metric_key: str = "raw_mse",
) -> dict[str, Any]:
    if not candidate or not baseline:
        raise ValueError("paired comparator cannot be empty")
    candidate_iids = [row.get("iid") for row in candidate]
    baseline_iids = [row.get("iid") for row in baseline]
    if len(set(candidate_iids)) != len(candidate_iids) or len(set(baseline_iids)) != len(baseline_iids):
        raise ValueError("paired comparator contains duplicate IIDs")
    candidate_map = {row["iid"]: float(row[metric_key]) for row in candidate}
    baseline_map = {row["iid"]: float(row[metric_key]) for row in baseline}
    if set(candidate_map) != set(baseline_map):
        raise ValueError("paired comparator IID closure differs")
    iids = sorted(candidate_map)
    left = torch.tensor([candidate_map[iid] for iid in iids], dtype=torch.float64)
    right = torch.tensor([baseline_map[iid] for iid in iids], dtype=torch.float64)
    if (
        not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
        or bool((left < 0.0).any())
        or bool((right <= 0.0).any())
    ):
        raise ValueError("paired comparator values differ")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(iids), (draws, len(iids)), generator=generator)
    ratios = left[indices].mean(dim=1) / right[indices].mean(dim=1).clamp_min(EPS)
    deltas = left[indices].mean(dim=1) - right[indices].mean(dim=1)
    return {
        "iid_count": len(iids),
        "bootstrap_seed": seed,
        "draws": draws,
        "metric": metric_key,
        "mean_ratio": float(left.mean() / right.mean().clamp_min(EPS)),
        "ratio_95pct_ci": [
            float(torch.quantile(ratios, 0.025)),
            float(torch.quantile(ratios, 0.975)),
        ],
        "mean_delta": float((left - right).mean()),
        "delta_95pct_ci": [
            float(torch.quantile(deltas, 0.025)),
            float(torch.quantile(deltas, 0.975)),
        ],
    }


def _load_arm_model(
    arm: str,
    receipt_path: str,
    expected_receipt_sha: str,
    config: Config,
    fold: Mapping[str, Any],
    prepare_sha: str,
    train_sha: str,
    expected_implementation: Mapping[str, str],
) -> tuple[nn.Module, dict[str, Any]]:
    receipt = _load_receipt(Path(receipt_path), expected_receipt_sha, ARM_RECEIPT_SCHEMA)
    if receipt.get("arm") != arm or receipt.get("prepare_receipt_sha256") != prepare_sha:
        raise ValueError("arm receipt authority differs")
    if receipt.get("train_bundle_sha256") != train_sha:
        raise ValueError("arm train bundle authority differs")
    if receipt.get("config") != asdict(config) or receipt.get("config_sha256") != _object_sha(asdict(config)):
        raise ValueError("arm receipt config authority differs")
    if receipt.get("fold") != fold:
        raise ValueError("arm receipt fold authority differs")
    if receipt.get("implementation") != expected_implementation:
        raise ValueError("arm receipt implementation differs")
    checkpoint_binding = receipt["checkpoint"]
    if type(checkpoint_binding) is not dict or set(checkpoint_binding) != {
        "path", "sha256", "size_bytes"
    }:
        raise ValueError("arm checkpoint binding differs")
    checkpoint = _load_torch(
        Path(checkpoint_binding["path"]), checkpoint_binding["sha256"],
        checkpoint_binding["size_bytes"],
    )
    if type(checkpoint) is not dict or set(checkpoint) != {
        "schema_version", "arm", "config", "config_sha256", "fold",
        "train_bundle_sha256", "prepare_receipt_sha256", "best_step",
        "executed_steps", "model_state", "implementation",
    }:
        raise ValueError("arm checkpoint exact-key allowlist differs")
    if checkpoint.get("schema_version") != ARM_CHECKPOINT_SCHEMA:
        raise ValueError("arm checkpoint schema differs")
    if checkpoint.get("arm") != arm or checkpoint.get("best_step") != receipt["best_step"]:
        raise ValueError("arm checkpoint identity differs")
    if (
        checkpoint.get("executed_steps") != receipt.get("executed_steps")
        or checkpoint.get("config") != asdict(config)
        or checkpoint.get("config_sha256") != receipt["config_sha256"]
        or checkpoint.get("fold") != fold
        or checkpoint.get("prepare_receipt_sha256") != prepare_sha
        or checkpoint.get("train_bundle_sha256") != train_sha
        or checkpoint.get("implementation") != expected_implementation
    ):
        raise ValueError("arm checkpoint authority join differs")
    model = _make_model(arm, config)
    if receipt.get("parameter_count") != _parameter_count(model):
        raise ValueError("arm parameter-count receipt differs")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, receipt


def compare_fold(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    prepare = _load_receipt(
        Path(args.prepare_receipt), args.expected_prepare_receipt_sha256, PREPARE_SCHEMA
    )
    train_bundle, config = _load_train_bundle_against_prepare(
        args, prepare, run_binding
    )
    eval_bundle, eval_config = _load_eval_bundle_against_prepare(
        args, prepare, run_binding
    )
    if train_bundle["fold"] != eval_bundle["fold"]:
        raise ValueError("train/eval fold authority differs")
    if config != eval_config:
        raise ValueError("train/eval config authority differs")
    if args.fold_index != train_bundle["fold"]["outer_fold"]:
        raise ValueError("compare CLI fold index differs")
    exploratory = eval_bundle["exploratory_oof"]
    count = eval_bundle["fold"]["counts"]["exploratory_oof"]
    split_iids = {
        "model_fit": train_bundle["model_fit"]["iids"],
        "early_stop_validation": train_bundle["early_stop_validation"]["iids"],
        "exploratory_oof": exploratory["iids"],
    }
    combined_iids = [iid for values in split_iids.values() for iid in values]
    if (
        len(combined_iids) != 644
        or len(set(combined_iids)) != 644
        or _object_sha(split_iids) != train_bundle["fold"]["iid_digest"]
    ):
        raise ValueError("train/eval exact644 IID closure differs")
    device = _device(args.device)
    ae, ae_receipt = _load_arm_model(
        "deterministic_ae", args.ae_receipt, args.expected_ae_receipt_sha256,
        config, train_bundle["fold"], args.expected_prepare_receipt_sha256,
        args.expected_train_bundle_sha256,
        run_binding,
    )
    vae, vae_receipt = _load_arm_model(
        "direct_beta_vae", args.vae_receipt, args.expected_vae_receipt_sha256,
        config, train_bundle["fold"], args.expected_prepare_receipt_sha256,
        args.expected_train_bundle_sha256,
        run_binding,
    )
    target = exploratory["value"]
    ae_reconstruction = _predict(ae, target, device)
    vae_reconstruction = _predict(vae, target, device)
    frame_reconstruction = _reconstruct_frame_pca(
        target, train_bundle["baselines"]["frame_pca_rank_l"]
    )
    clip_reconstruction = _reconstruct_clip_pca(
        target, train_bundle["baselines"]["clip_pca_rank_l"]
    )
    zero_reconstruction = torch.zeros_like(target)
    rms = eval_bundle["global_rms"]
    energy_edges = eval_bundle["fold"]["fixed_energy_bin_edges_exact644"]
    reconstructions = {
        "deterministic_ae": ae_reconstruction,
        "direct_beta_vae": vae_reconstruction,
        "frame_pca_rank_l_hard_baseline": frame_reconstruction,
        "clip_pca_rank_l_diagnostic": clip_reconstruction,
        "zero_hard_baseline": zero_reconstruction,
    }
    metrics = {
        name: _metric_rows(target, value, exploratory["iids"], rms, energy_edges)
        for name, value in reconstructions.items()
    }
    comparisons = {}
    for arm_index, arm in enumerate(ARMS):
        comparisons[arm] = {
            "vs_zero": _paired_ratio(
                metrics[arm]["per_iid"], metrics["zero_hard_baseline"]["per_iid"],
                config.seed + 3000 + 10 * args.fold_index + arm_index,
            ),
            "vs_frame_pca_rank_l": _paired_ratio(
                metrics[arm]["per_iid"],
                metrics["frame_pca_rank_l_hard_baseline"]["per_iid"],
                config.seed + 4000 + 10 * args.fold_index + arm_index,
            ),
        }
    gates = {
        arm: {
            "beats_zero_ratio_ucb_lt_1": comparisons[arm]["vs_zero"]["ratio_95pct_ci"][1] < 1.0,
            "beats_frame_pca_ratio_ucb_lt_1": comparisons[arm]["vs_frame_pca_rank_l"]["ratio_95pct_ci"][1] < 1.0,
        }
        for arm in ARMS
    }
    for arm in ARMS:
        gates[arm]["absolute_and_linear_baseline_gate"] = bool(
            gates[arm]["beats_zero_ratio_ucb_lt_1"]
            and gates[arm]["beats_frame_pca_ratio_ucb_lt_1"]
        )
    gates["both_arms_absolute_and_linear_baseline_gate"] = bool(
        all(gates[arm]["absolute_and_linear_baseline_gate"] for arm in ARMS)
    )
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    receipt: dict[str, Any] = {
        "schema_version": COMPARE_SCHEMA,
        "status": "EXPLORATORY_OOF_FOLD_COMPARISON_COMPLETE",
        "fold_index": args.fold_index,
        "all_data_status": "BURNED_DEVELOPMENT_ONLY",
        "prior_locked_partition_rows_burned": 96,
        "exact644_role": "BURNED_DEVELOPMENT_ONLY",
        "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        "confirmation_evaluations_allowed_by_this_runtime": 0,
        "family_labels_used_only_for_stratified_split": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "direct_target_not_residual": True,
        "rgb_or_wan_reconstruction_performed": False,
        "source_identity_preservation_tested": False,
        "scientific_confirmation_claimed": False,
        "action_representation_qualified": False,
        "vae_necessary": None,
        "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION",
        "anchor_identity_may_remain": True,
        "target": {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
        },
        "sample_accounting": {
            "unique_exact644_development": 644,
            "exploratory_oof_original_rows_this_fold": count,
            "derived_rows": 0,
        },
        "feature_receipt_sha256": eval_bundle["feature_receipt_sha256"],
        "feature_receipt_digest": eval_bundle["feature_receipt_digest"],
        "exact644_iid_digest": eval_bundle["exact644_iid_digest"],
        "exact644_raw_target_sha256": eval_bundle["exact644_raw_target_sha256"],
        "exact644_population_authority": eval_bundle["exact644_population_authority"],
        "outer_assignment_digest": eval_bundle["fold"]["outer_assignment_digest"],
        "development_energy_definition": eval_bundle["fold"]["energy_definition"],
        "development_energy_bin_edges_raw": eval_bundle["fold"]["fixed_energy_bin_edges_exact644"],
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "train_bundle_sha256": args.expected_train_bundle_sha256,
        "exploratory_oof_bundle_sha256": args.expected_exploratory_oof_bundle_sha256,
        "baselines": {
            "primary_hard_linear": "time-shared frame PCA rank=latent_dim",
            "clip_pca_role": "diagnostic only; not capacity matched",
            "zero_hard": True,
        },
        "config": asdict(config),
        "config_sha256": _object_sha(asdict(config)),
        "fold": eval_bundle["fold"],
        "arm_best_steps": {
            "deterministic_ae": ae_receipt["best_step"],
            "direct_beta_vae": vae_receipt["best_step"],
        },
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "gates": gates,
        "full644_refit_authorized_by_single_fold": False,
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json_create_only(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {"receipt": str(receipt_path.resolve()), "receipt_sha256": receipt_sha, "gates": gates}


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("aggregate metric rows cannot be empty")
    required = {
        "iid", "mse", "zero_mse", "raw_mse", "raw_zero_mse", "cosine",
        "raw_output_energy", "temporal_delta_mse", "raw_temporal_delta_mse",
        "energy_bin",
    }
    if any(type(row) is not dict or set(row) != required for row in rows):
        raise ValueError("aggregate per-IID metric keys differ")
    for key in required - {"iid", "energy_bin"}:
        if any(not math.isfinite(float(row[key])) for row in rows):
            raise ValueError("aggregate metric contains non-finite values")
    by_energy: dict[str, Any] = {}
    for bin_index in range(5):
        selected = [row for row in rows if row["energy_bin"] == bin_index]
        by_energy[str(bin_index)] = {
            "count": len(selected),
            "raw_mse": (
                sum(float(row["raw_mse"]) for row in selected) / len(selected)
                if selected else None
            ),
            "raw_zero_mse": (
                sum(float(row["raw_zero_mse"]) for row in selected) / len(selected)
                if selected else None
            ),
            "raw_output_energy": (
                sum(float(row["raw_output_energy"]) for row in selected) / len(selected)
                if selected else None
            ),
            "raw_mse_ratio_vs_zero": (
                sum(float(row["raw_mse"]) for row in selected)
                / max(sum(float(row["raw_zero_mse"]) for row in selected), EPS)
                if selected else None
            ),
            "cosine": (
                sum(float(row["cosine"]) for row in selected) / len(selected)
                if selected else None
            ),
        }
    raw_mse = sum(float(row["raw_mse"]) for row in rows) / len(rows)
    raw_zero = sum(float(row["raw_zero_mse"]) for row in rows) / len(rows)
    return {
        "count": len(rows),
        "raw_anchor_feature_mse": raw_mse,
        "raw_zero_mse": raw_zero,
        "normalized_raw_mse_vs_zero": raw_mse / max(raw_zero, EPS),
        "cosine": sum(float(row["cosine"]) for row in rows) / len(rows),
        "raw_temporal_delta_mse": sum(
            float(row["raw_temporal_delta_mse"]) for row in rows
        ) / len(rows),
        "energy_strata": by_energy,
    }


def aggregate_oof(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    if len(args.fold_receipt) != OUTER_FOLDS or len(args.expected_fold_receipt_sha256) != OUTER_FOLDS:
        raise ValueError("aggregate requires exact5 fold receipts and SHAs")
    rows = [
        _load_receipt(Path(path), sha, COMPARE_SCHEMA)
        for path, sha in zip(args.fold_receipt, args.expected_fold_receipt_sha256)
    ]
    by_fold = {row["fold_index"]: row for row in rows}
    if set(by_fold) != set(range(OUTER_FOLDS)):
        raise ValueError("aggregate fold closure differs")
    config = rows[0]["config"]
    if any(row["config"] != config for row in rows):
        raise ValueError("aggregate configs differ")
    Config(**config).validate()
    if any(row.get("config_sha256") != _object_sha(config) for row in rows):
        raise ValueError("aggregate config digest differs")
    if any(row.get("implementation") != run_binding for row in rows):
        raise ValueError("aggregate fold implementation pin differs")
    if any(
        row.get("target") != {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
        }
        for row in rows
    ):
        raise ValueError("aggregate fold target contract differs")
    authority_fields = (
        "feature_receipt_sha256", "feature_receipt_digest",
        "exact644_iid_digest", "exact644_raw_target_sha256",
        "outer_assignment_digest", "development_energy_definition",
        "development_energy_bin_edges_raw", "exact644_population_authority",
    )
    for key in authority_fields:
        if any(row.get(key) != rows[0].get(key) for row in rows):
            raise ValueError(f"aggregate cross-fold {key} differs")
    for key in authority_fields[:4]:
        _sha(rows[0][key], key)
    if rows[0]["development_energy_definition"] != "mean(square(temporal_center(anchor ordered DINO)))":
        raise ValueError("aggregate energy definition differs")
    if rows[0]["exact644_population_authority"] != {
        "unique_original_base_clips": 644,
        "family_count": 28,
        "strict_true": 359,
        "strict_false": 285,
        "derived_rows": 0,
    }:
        raise ValueError("aggregate exact644 population authority differs")
    if any(
        row["fold_index"] != fold
        or row["fold"]["outer_fold"] != fold
        or row["outer_assignment_digest"] != row["fold"]["outer_assignment_digest"]
        for fold, row in by_fold.items()
    ):
        raise ValueError("aggregate fold-index authority differs")
    combined: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metric_names = {
        "deterministic_ae", "direct_beta_vae",
        "frame_pca_rank_l_hard_baseline", "clip_pca_rank_l_diagnostic",
        "zero_hard_baseline",
    }
    for fold in range(OUTER_FOLDS):
        row = by_fold[fold]
        if type(row.get("metrics")) is not dict or set(row["metrics"]) != metric_names:
            raise ValueError("aggregate metric names differ")
        expected_count = row["fold"]["counts"]["exploratory_oof"]
        reference_iids: list[str] | None = None
        for name, metric in row["metrics"].items():
            per_iid = metric.get("per_iid")
            if type(per_iid) is not list or len(per_iid) != expected_count:
                raise ValueError("aggregate fold metric count differs")
            metric_iids = [item.get("iid") for item in per_iid]
            if len(set(metric_iids)) != expected_count:
                raise ValueError("aggregate fold metric IID uniqueness differs")
            if reference_iids is None:
                reference_iids = metric_iids
            elif metric_iids != reference_iids:
                raise ValueError("aggregate fold metric IID order differs")
            combined[name].extend(metric["per_iid"])
    all_iids = [row["iid"] for row in combined["zero_hard_baseline"]]
    if len(all_iids) != 644 or len(set(all_iids)) != 644:
        raise ValueError("aggregate OOF IIDs must close exact644 once")
    if _object_sha(sorted(all_iids)) != rows[0]["exact644_iid_digest"]:
        raise ValueError("aggregate OOF/source exact644 IID authority differs")
    for name, metric_rows in combined.items():
        if [item["iid"] for item in metric_rows] != all_iids:
            raise ValueError(f"aggregate {name} IID closure differs")
    comparisons = {}
    for arm_index, arm in enumerate(ARMS):
        comparisons[arm] = {
            "vs_zero": _paired_ratio(
                combined[arm], combined["zero_hard_baseline"],
                config["seed"] + 5000 + arm_index,
            ),
            "vs_frame_pca_rank_l": _paired_ratio(
                combined[arm], combined["frame_pca_rank_l_hard_baseline"],
                config["seed"] + 6000 + arm_index,
            ),
        }
    gates: dict[str, Any] = {}
    for arm in ARMS:
        gates[arm] = {
            "beats_zero_ratio_ucb_lt_1": comparisons[arm]["vs_zero"]["ratio_95pct_ci"][1] < 1.0,
            "beats_frame_pca_ratio_ucb_lt_1": comparisons[arm]["vs_frame_pca_rank_l"]["ratio_95pct_ci"][1] < 1.0,
        }
        gates[arm]["absolute_and_linear_baseline_gate"] = bool(
            gates[arm]["beats_zero_ratio_ucb_lt_1"]
            and gates[arm]["beats_frame_pca_ratio_ucb_lt_1"]
        )
    gates["both_arms_absolute_and_linear_baseline_gate"] = bool(
        all(gates[arm]["absolute_and_linear_baseline_gate"] for arm in ARMS)
    )
    authorized_arms = [
        arm for arm in ARMS if gates[arm]["absolute_and_linear_baseline_gate"]
    ]
    selected_steps = {
        arm: int(torch.tensor(
            [by_fold[fold]["arm_best_steps"][arm] for fold in range(OUTER_FOLDS)],
            dtype=torch.float64,
        ).median())
        for arm in ARMS
    }
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    receipt: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "EXPLORATORY_5FOLD_OOF_AGGREGATED_ALL644_DEVELOPMENT",
        "all_exact644_are_burned_development": True,
        "prior_locked_partition_rows_burned": 96,
        "exact644_role": "BURNED_DEVELOPMENT_ONLY",
        "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        "confirmation_evaluations_allowed_by_this_runtime": 0,
        "scientific_confirmation_claimed": False,
        "family_labels_used_only_for_stratified_split": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "direct_target_not_residual": True,
        "rgb_or_wan_reconstruction_performed": False,
        "source_identity_preservation_tested": False,
        "oof_unique_iids": 644,
        "oof_each_iid_evaluated_once": True,
        "derived_rows": 0,
        "config": config,
        "config_sha256": _object_sha(config),
        "feature_receipt_sha256": rows[0]["feature_receipt_sha256"],
        "feature_receipt_digest": rows[0]["feature_receipt_digest"],
        "exact644_iid_digest": rows[0]["exact644_iid_digest"],
        "exact644_raw_target_sha256": rows[0]["exact644_raw_target_sha256"],
        "exact644_population_authority": rows[0]["exact644_population_authority"],
        "outer_assignment_digest": rows[0]["outer_assignment_digest"],
        "development_energy_definition": rows[0]["development_energy_definition"],
        "development_energy_bin_edges_raw": rows[0]["development_energy_bin_edges_raw"],
        "fold_receipt_sha256": args.expected_fold_receipt_sha256,
        "aggregate_metrics": {
            name: _aggregate_metric_rows(metric_rows)
            for name, metric_rows in combined.items()
        },
        "combined_paired_comparisons": comparisons,
        "gates": gates,
        "full644_refit_step_preregistration": {
            "strategy": "median of exact5 fold-specific early-stop best steps",
            "steps_by_arm": selected_steps,
            "frozen_before_refit": True,
        },
        "full644_refit_authorized_by_arm": {
            arm: gates[arm]["absolute_and_linear_baseline_gate"] for arm in ARMS
        },
        "full644_refit_authorized_arms": authorized_arms,
        "full644_refit_any_arm_authorized": bool(authorized_arms),
        "refit_timing": "after OOF gates and step freeze; before any future one-shot external confirmation",
        "action_representation_qualified": False,
        "vae_necessary": None,
        "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION",
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json_create_only(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()), "receipt_sha256": receipt_sha,
        "full644_refit_authorized_arms": authorized_arms,
        "steps_by_arm": selected_steps,
    }


def prepare_refit(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    aggregate = _load_receipt(
        Path(args.aggregate_receipt), args.expected_aggregate_receipt_sha256,
        AGGREGATE_SCHEMA,
    )
    authorized_arms = aggregate.get("full644_refit_authorized_arms")
    gates = aggregate.get("gates")
    expected_authorized_arms = (
        [arm for arm in ARMS if gates.get(arm, {}).get("absolute_and_linear_baseline_gate") is True]
        if type(gates) is dict else []
    )
    if (
        type(authorized_arms) is not list
        or not authorized_arms
        or authorized_arms != expected_authorized_arms
        or aggregate.get("full644_refit_authorized_by_arm")
        != {arm: arm in expected_authorized_arms for arm in ARMS}
        or aggregate.get("full644_refit_any_arm_authorized") is not bool(expected_authorized_arms)
        or any(arm not in ARMS for arm in authorized_arms)
        or len(set(authorized_arms)) != len(authorized_arms)
    ):
        raise ValueError("full644 refit is fail-closed because no arm passed OOF gates")
    if aggregate.get("implementation") != run_binding:
        raise ValueError("aggregate implementation pin differs")
    if args.expected_feature_receipt_sha256 != aggregate.get("feature_receipt_sha256"):
        raise ValueError("refit feature receipt differs from OOF authority")
    config = Config(**aggregate["config"])
    config.validate()
    pairs, feature_receipt = authority.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    if len(pairs) != 644:
        raise ValueError("refit requires exact644 original anchors")
    population_authority = _exact644_population_authority(pairs)
    exact644_iid_digest = _object_sha([row.iid for row in pairs])
    exact644_raw_target_sha = _tensor_sha(torch.stack([
        anchor_action_target(row) for row in pairs
    ]))
    if (
        feature_receipt["receipt_digest"] != aggregate["feature_receipt_digest"]
        or exact644_iid_digest != aggregate["exact644_iid_digest"]
        or exact644_raw_target_sha != aggregate["exact644_raw_target_sha256"]
        or population_authority != aggregate["exact644_population_authority"]
    ):
        raise ValueError("refit exact644 feature/target authority differs from OOF")
    rms = _global_rms(pairs)
    values = _tensor_rows(pairs, rms)
    frame_pca = _fit_frame_pca(values["value"], config.latent_dim)
    frame_pca_sha = _pca_state_sha(frame_pca)
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    bundle = {
        "schema_version": REFIT_BUNDLE_SCHEMA,
        "config": asdict(config),
        "config_sha256": _object_sha(asdict(config)),
        "aggregate_receipt_sha256": args.expected_aggregate_receipt_sha256,
        "preregistered_steps_by_arm": aggregate["full644_refit_step_preregistration"]["steps_by_arm"],
        "raw_target_definition": RAW_TARGET_DEFINITION,
        "model_coordinate_definition": MODEL_COORDINATE_DEFINITION,
        "global_rms": rms,
        "global_rms_sha256": _tensor_sha(rms),
        "global_rms_fit_only": True,
        "pca_is_model_target": False,
        "full644_originals": values,
        "full644_model_coordinate_sha256": _tensor_sha(values["value"]),
        "full644_frame_pca_rank_l_hard_baseline": frame_pca,
        "full644_frame_pca_rank_l_sha256": frame_pca_sha,
        "authorized_arms": authorized_arms,
        "model_fit_unique_originals": 644,
        "held_rows": 0,
        "derived_rows": 0,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_iid_digest": exact644_iid_digest,
        "exact644_raw_target_sha256": exact644_raw_target_sha,
        "exact644_population_authority": population_authority,
        "development_energy_definition": aggregate["development_energy_definition"],
        "development_energy_bin_edges_raw": aggregate["development_energy_bin_edges_raw"],
        "implementation": run_binding,
    }
    bundle_path = output / "refit_bundle.pt"
    bundle_sha = _save_torch_create_only(bundle_path, bundle)
    receipt: dict[str, Any] = {
        "schema_version": REFIT_PREPARE_SCHEMA,
        "status": "FULL644_REFIT_BUNDLE_PREPARED_NO_HELD_EVALUATION",
        "aggregate_receipt_sha256": args.expected_aggregate_receipt_sha256,
        "refit_bundle": {
            "path": str(bundle_path.resolve()), "sha256": bundle_sha,
            "size_bytes": bundle_path.stat().st_size,
        },
        "model_fit_unique_originals": 644,
        "held_rows": 0,
        "derived_rows": 0,
        "prior_locked_partition_rows_burned": 96,
        "exact644_role": "BURNED_DEVELOPMENT_ONLY",
        "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        "confirmation_evaluations_allowed_by_this_runtime": 0,
        "refit_timing": "after OOF gates and step freeze; before any future one-shot external confirmation",
        "family_labels_used_only_for_stratified_split": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "direct_target_not_residual": True,
        "rgb_or_wan_reconstruction_performed": False,
        "source_identity_preservation_tested": False,
        "global_rms": float(rms),
        "global_rms_sha256": _tensor_sha(rms),
        "full644_model_coordinate_sha256": _tensor_sha(values["value"]),
        "full644_frame_pca_rank_l_sha256": frame_pca_sha,
        "authorized_arms": authorized_arms,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_iid_digest": exact644_iid_digest,
        "exact644_raw_target_sha256": exact644_raw_target_sha,
        "exact644_population_authority": population_authority,
        "development_energy_definition": aggregate["development_energy_definition"],
        "development_energy_bin_edges_raw": aggregate["development_energy_bin_edges_raw"],
        "preregistered_steps_by_arm": bundle["preregistered_steps_by_arm"],
        "scientific_confirmation_claimed": False,
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "prepare_receipt.json"
    receipt_sha = _write_json_create_only(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {"receipt": str(receipt_path.resolve()), "receipt_sha256": receipt_sha, "bundle_sha256": bundle_sha}


def _load_refit_bundle_against_prepare(
    args: argparse.Namespace, prepare: Mapping[str, Any],
    expected_implementation: Mapping[str, str],
) -> tuple[dict[str, Any], Config]:
    binding = prepare.get("refit_bundle")
    if type(binding) is not dict or set(binding) != {"path", "sha256", "size_bytes"}:
        raise ValueError("refit bundle binding differs")
    path = Path(args.refit_bundle).resolve(strict=True)
    if Path(binding["path"]).resolve(strict=True) != path:
        raise ValueError("refit bundle path differs")
    expected_sha = _sha(args.expected_refit_bundle_sha256, "refit bundle SHA")
    if binding["sha256"] != expected_sha:
        raise ValueError("refit bundle CLI/prepare SHA differs")
    bundle = _load_torch(path, expected_sha, binding["size_bytes"])
    if type(bundle) is not dict or set(bundle) != REFIT_BUNDLE_KEYS:
        raise ValueError("refit bundle exact-key allowlist differs")
    if bundle["schema_version"] != REFIT_BUNDLE_SCHEMA:
        raise ValueError("refit bundle schema differs")
    if type(bundle["config"]) is not dict or set(bundle["config"]) != set(asdict(Config())):
        raise ValueError("refit config keys differ")
    config = Config(**bundle["config"])
    config.validate()
    if _object_sha(bundle["config"]) != _sha(bundle["config_sha256"], "refit config SHA"):
        raise ValueError("refit config digest differs")
    if bundle["aggregate_receipt_sha256"] != prepare["aggregate_receipt_sha256"]:
        raise ValueError("refit aggregate authority differs")
    if (
        bundle["implementation"] != prepare["implementation"]
        or bundle["implementation"] != expected_implementation
    ):
        raise ValueError("refit implementation pin differs")
    if (
        bundle["raw_target_definition"] != RAW_TARGET_DEFINITION
        or bundle["model_coordinate_definition"] != MODEL_COORDINATE_DEFINITION
        or bundle["global_rms_fit_only"] is not True
        or bundle["pca_is_model_target"] is not False
    ):
        raise ValueError("refit target contract differs")
    rms = bundle["global_rms"]
    if (
        type(rms) is not torch.Tensor or rms.dtype != torch.float32
        or tuple(rms.shape) != (1,) or not bool(torch.isfinite(rms).all())
        or float(rms) <= EPS or _tensor_sha(rms) != bundle["global_rms_sha256"]
        or bundle["global_rms_sha256"] != prepare["global_rms_sha256"]
        or not math.isclose(float(rms), prepare["global_rms"], rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("refit RMS authority differs")
    if (
        bundle["model_fit_unique_originals"] != 644
        or bundle["held_rows"] != 0
        or bundle["derived_rows"] != 0
    ):
        raise ValueError("refit sample accounting differs")
    _validate_sequence_rows(bundle["full644_originals"], 644, "full644_originals")
    values = bundle["full644_originals"]
    if not bool(torch.isclose(
        values["value"].square().mean().sqrt(), torch.tensor(1.0),
        atol=3.0e-6, rtol=3.0e-6,
    )):
        raise ValueError("refit standardized RMS differs")
    if _object_sha(values["iids"]) != bundle["exact644_iid_digest"]:
        raise ValueError("refit IID digest differs")
    if (
        _tensor_sha(values["value"]) != bundle["full644_model_coordinate_sha256"]
        or bundle["full644_model_coordinate_sha256"]
        != prepare["full644_model_coordinate_sha256"]
    ):
        raise ValueError("refit model-coordinate target digest differs")
    for key in (
        "feature_receipt_sha256", "feature_receipt_digest",
        "exact644_iid_digest", "exact644_raw_target_sha256",
    ):
        _sha(bundle[key], key)
        if bundle[key] != prepare[key]:
            raise ValueError(f"refit bundle/prepare {key} differs")
    if bundle["exact644_population_authority"] != {
        "unique_original_base_clips": 644,
        "family_count": 28,
        "strict_true": 359,
        "strict_false": 285,
        "derived_rows": 0,
    } or bundle["exact644_population_authority"] != prepare["exact644_population_authority"]:
        raise ValueError("refit population authority differs")
    pca = bundle["full644_frame_pca_rank_l_hard_baseline"]
    _validate_pca_state(pca, (1, 768), (768, config.latent_dim), "full644 frame PCA")
    if (
        _pca_state_sha(pca) != bundle["full644_frame_pca_rank_l_sha256"]
        or bundle["full644_frame_pca_rank_l_sha256"] != prepare["full644_frame_pca_rank_l_sha256"]
    ):
        raise ValueError("refit frame-PCA authority differs")
    authorized = bundle["authorized_arms"]
    if (
        type(authorized) is not list or not authorized
        or authorized != prepare["authorized_arms"]
        or any(arm not in ARMS for arm in authorized)
        or len(set(authorized)) != len(authorized)
    ):
        raise ValueError("refit authorized-arm closure differs")
    steps = bundle["preregistered_steps_by_arm"]
    if (
        type(steps) is not dict or set(steps) != set(ARMS)
        or steps != prepare["preregistered_steps_by_arm"]
        or any(type(step) is not int or not 1 <= step <= config.max_steps for step in steps.values())
    ):
        raise ValueError("refit preregistered-step closure differs")
    if (
        bundle["development_energy_definition"] != prepare["development_energy_definition"]
        or bundle["development_energy_bin_edges_raw"] != prepare["development_energy_bin_edges_raw"]
    ):
        raise ValueError("refit development energy authority differs")
    return bundle, config


def _train_fixed_steps(
    arm: str, model: nn.Module, value: torch.Tensor, steps: int,
    config: Config, device: torch.device,
) -> tuple[list[dict[str, float]], str]:
    model.to(device).train()
    value = value.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4)
    generator = torch.Generator().manual_seed(config.seed + 7000)
    schedule_digest = hashlib.sha256()
    history = []
    for step in range(1, steps + 1):
        indices_cpu = torch.randint(
            len(value), (min(config.batch_size, len(value)),), generator=generator
        )
        schedule_digest.update(indices_cpu.numpy().tobytes(order="C"))
        indices = indices_cpu.to(device)
        target = value[indices]
        output = model(target, sample=True)
        total, metrics = _loss(arm, output, target, step, config)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step == steps or step % max(steps // 10, 1) == 0:
            history.append({"step": step, **metrics})
    return history, schedule_digest.hexdigest()


def train_refit(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    prepare = _load_receipt(
        Path(args.prepare_receipt), args.expected_prepare_receipt_sha256,
        REFIT_PREPARE_SCHEMA,
    )
    bundle, config = _load_refit_bundle_against_prepare(
        args, prepare, run_binding
    )
    if args.arm not in ARMS:
        raise ValueError("refit arm differs")
    if args.arm not in bundle["authorized_arms"]:
        raise ValueError("this arm failed OOF gates and is not authorized for refit")
    steps = bundle["preregistered_steps_by_arm"][args.arm]
    if type(steps) is not int or not 1 <= steps <= config.max_steps:
        raise ValueError("preregistered refit step differs")
    device = _device(args.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = _make_model(args.arm, config)
    history, schedule_digest = _train_fixed_steps(
        args.arm, model, bundle["full644_originals"]["value"], steps, config, device
    )
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    checkpoint = {
        "schema_version": REFIT_CHECKPOINT_SCHEMA,
        "arm": args.arm,
        "config": asdict(config),
        "refit_bundle_sha256": args.expected_refit_bundle_sha256,
        "aggregate_receipt_sha256": bundle["aggregate_receipt_sha256"],
        "preregistered_steps": steps,
        "executed_steps": steps,
        "executed_minibatch_schedule_sha256": schedule_digest,
        "parameter_count": _parameter_count(model),
        "model_state": _cpu_state(model),
        "global_rms": bundle["global_rms"],
        "feature_receipt_sha256": bundle["feature_receipt_sha256"],
        "feature_receipt_digest": bundle["feature_receipt_digest"],
        "exact644_iid_digest": bundle["exact644_iid_digest"],
        "exact644_raw_target_sha256": bundle["exact644_raw_target_sha256"],
        "implementation": run_binding,
    }
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_sha = _save_torch_create_only(checkpoint_path, checkpoint)
    receipt: dict[str, Any] = {
        "schema_version": REFIT_RECEIPT_SCHEMA,
        "status": "FULL644_REFIT_COMPLETE_NO_HELD_EVALUATION",
        "arm": args.arm,
        "model_fit_unique_originals": 644,
        "held_rows": 0,
        "derived_rows": 0,
        "prior_locked_partition_rows_burned": 96,
        "exact644_role": "BURNED_DEVELOPMENT_ONLY",
        "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        "confirmation_evaluations_allowed_by_this_runtime": 0,
        "refit_timing": "after OOF gates and step freeze; before any future one-shot external confirmation",
        "family_labels_used_only_for_stratified_split": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "direct_target_not_residual": True,
        "rgb_or_wan_reconstruction_performed": False,
        "source_identity_preservation_tested": False,
        "preregistered_steps": steps,
        "executed_steps": steps,
        "early_stop_during_refit": False,
        "executed_minibatch_schedule_sha256": schedule_digest,
        "parameter_count": _parameter_count(model),
        "target": {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
        },
        "feature_receipt_sha256": bundle["feature_receipt_sha256"],
        "feature_receipt_digest": bundle["feature_receipt_digest"],
        "exact644_iid_digest": bundle["exact644_iid_digest"],
        "exact644_raw_target_sha256": bundle["exact644_raw_target_sha256"],
        "full644_frame_pca_rank_l_sha256": bundle["full644_frame_pca_rank_l_sha256"],
        "training_history": history,
        "refit_bundle_sha256": args.expected_refit_bundle_sha256,
        "aggregate_receipt_sha256": bundle["aggregate_receipt_sha256"],
        "scientific_confirmation_claimed": False,
        "action_representation_qualified": False,
        "vae_necessary": None,
        "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json_create_only(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {"receipt": str(receipt_path.resolve()), "receipt_sha256": receipt_sha, "checkpoint_sha256": checkpoint_sha}


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = Config()
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--frame-hidden-dim", type=int, default=defaults.frame_hidden_dim)
    parser.add_argument("--latent-dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--beta-kl", type=float, default=defaults.beta_kl)
    parser.add_argument("--kl-warmup-steps", type=int, default=defaults.kl_warmup_steps)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--patience-evals", type=int, default=defaults.patience_evals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anchor-only full-DINO sequence AE/beta-VAE exploratory OOF"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-fold")
    prepare.add_argument("--feature-root", required=True)
    prepare.add_argument("--expected-feature-receipt-sha256", required=True)
    prepare.add_argument("--fold-index", required=True, type=int, choices=range(OUTER_FOLDS))
    prepare.add_argument("--output", required=True)
    _add_config_arguments(prepare)
    prepare.set_defaults(handler=prepare_fold)

    train = subparsers.add_parser("train-fold")
    train.add_argument("--prepare-receipt", required=True)
    train.add_argument("--expected-prepare-receipt-sha256", required=True)
    train.add_argument("--train-bundle", required=True)
    train.add_argument("--expected-train-bundle-sha256", required=True)
    train.add_argument("--fold-index", required=True, type=int, choices=range(OUTER_FOLDS))
    train.add_argument("--arm", required=True, choices=ARMS)
    train.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    train.add_argument("--output", required=True)
    train.set_defaults(handler=train_fold_arm)

    compare = subparsers.add_parser("compare-fold")
    compare.add_argument("--prepare-receipt", required=True)
    compare.add_argument("--expected-prepare-receipt-sha256", required=True)
    compare.add_argument("--train-bundle", required=True)
    compare.add_argument("--expected-train-bundle-sha256", required=True)
    compare.add_argument("--exploratory-oof-bundle", required=True)
    compare.add_argument("--expected-exploratory-oof-bundle-sha256", required=True)
    compare.add_argument("--fold-index", required=True, type=int, choices=range(OUTER_FOLDS))
    compare.add_argument("--ae-receipt", required=True)
    compare.add_argument("--expected-ae-receipt-sha256", required=True)
    compare.add_argument("--vae-receipt", required=True)
    compare.add_argument("--expected-vae-receipt-sha256", required=True)
    compare.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=compare_fold)

    aggregate = subparsers.add_parser("aggregate-oof")
    aggregate.add_argument("--fold-receipt", required=True, nargs=OUTER_FOLDS)
    aggregate.add_argument(
        "--expected-fold-receipt-sha256", required=True, nargs=OUTER_FOLDS
    )
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=aggregate_oof)

    refit_prepare = subparsers.add_parser("prepare-refit")
    refit_prepare.add_argument("--aggregate-receipt", required=True)
    refit_prepare.add_argument("--expected-aggregate-receipt-sha256", required=True)
    refit_prepare.add_argument("--feature-root", required=True)
    refit_prepare.add_argument("--expected-feature-receipt-sha256", required=True)
    refit_prepare.add_argument("--output", required=True)
    refit_prepare.set_defaults(handler=prepare_refit)

    refit_train = subparsers.add_parser("train-refit")
    refit_train.add_argument("--prepare-receipt", required=True)
    refit_train.add_argument("--expected-prepare-receipt-sha256", required=True)
    refit_train.add_argument("--refit-bundle", required=True)
    refit_train.add_argument("--expected-refit-bundle-sha256", required=True)
    refit_train.add_argument("--arm", required=True, choices=ARMS)
    refit_train.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    refit_train.add_argument("--output", required=True)
    refit_train.set_defaults(handler=train_refit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
