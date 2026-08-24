#!/usr/bin/env python3
"""Exact644 original-only direct action-AE versus beta-VAE experiment.

``prepare`` seals the one shared IID split, fit-only PCA and tensors.  Two
independent ``train`` invocations then consume that exact bundle, allowing the
deterministic AE and direct beta-VAE arms to run on different one-GPU jobs.

The raw action definition is
``center(anchor ordered DINO) - center(source ordered DINO)``.  The model's
lossy target is its fit-only PCA projection, not the full 768-D quotient.  It
is neither RGB nor a Wan video-VAE latent.  The target is not yet a qualified structured
participant/role/phase/terminal action representation, and one execution per
IID cannot establish whether a stochastic VAE is necessary.
"""

from __future__ import annotations

import argparse
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

from methods.bernini_action_editing import semantic_action_cvae_canary_v1 as common


TRAIN_BUNDLE_SCHEMA = "semantic-action-direct-vae-exact644-train-bundle-v1"
HELD_BUNDLE_SCHEMA = "semantic-action-direct-vae-exact644-held-bundle-v1"
PREPARE_RECEIPT_SCHEMA = "semantic-action-direct-vae-exact644-prepare-receipt-v1"
ARM_RESULT_SCHEMA = "semantic-action-direct-vae-exact644-arm-receipt-v1"
FINAL_RESULT_SCHEMA = "semantic-action-direct-vae-exact644-final-receipt-v1"
CHECKPOINT_SCHEMA = "semantic-action-direct-vae-exact644-arm-checkpoint-v1"
ARMS = ("deterministic_ae", "direct_beta_vae")
RAW_ACTION_DEFINITION = "center(anchor ordered DINO) - center(source ordered DINO)"
MODEL_TARGET_DESCRIPTION = "flatten(standardize((raw_action - fit_mean) @ fit_only_PCA_basis))"
EXPECTED_COUNTS = {"fit": 452, "calibration": 96, "locked": 96}
EPS = 1.0e-8


@dataclass(frozen=True)
class Config:
    seed: int = 20260819
    pca_dim: int = 64
    latent_dim: int = 32
    hidden_dim: int = 256
    steps: int = 1200
    batch_size: int = 128
    learning_rate: float = 2.0e-3
    beta_kl: float = 0.02
    prior_samples: int = 8

    def validate(self) -> None:
        if not 1 <= self.pca_dim <= 256:
            raise ValueError("pca_dim must be in [1,256]")
        if not 1 <= self.latent_dim <= 256:
            raise ValueError("latent_dim must be in [1,256]")
        if not 16 <= self.hidden_dim <= 2048:
            raise ValueError("hidden_dim must be in [16,2048]")
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if not 0.0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate is outside the canary range")
        if not 0.0 < self.beta_kl <= 1.0:
            raise ValueError("beta_kl must be in (0,1]")
        if self.prior_samples < 2:
            raise ValueError("prior_samples must be at least two")


def _implementation_binding() -> dict[str, str]:
    implementation = Path(__file__).resolve(strict=True)
    dependency = Path(common.__file__).resolve(strict=True)
    return {
        "implementation_path": str(implementation),
        "implementation_sha256": common.file_sha256(implementation),
        "common_dependency_path": str(dependency),
        "common_dependency_sha256": common.file_sha256(dependency),
    }


def _config_from_args(args: argparse.Namespace) -> Config:
    config = Config(
        seed=args.seed,
        pca_dim=args.pca_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_kl=args.beta_kl,
        prior_samples=args.prior_samples,
    )
    config.validate()
    return config


def _fresh_output(path_string: str) -> Path:
    path = Path(path_string)
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("output must be a fresh absolute child of an existing directory")
    path.mkdir(mode=0o700)
    return path


def _preflight_fresh_output(path_string: str) -> Path:
    path = Path(path_string)
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("output must be a fresh absolute child of an existing directory")
    return path


def _save_torch_create_only(path: Path, value: Any) -> str:
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return common.file_sha256(path)


def _json_create_only(path: Path, value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return hashlib.sha256(raw).hexdigest()


def _projection_sha256(projection: common.Projection) -> str:
    digest = hashlib.sha256()
    for name, tensor in (
        ("mean", projection.mean),
        ("basis", projection.basis),
        ("scale", projection.scale),
    ):
        tensor = tensor.detach().cpu().contiguous()
        header = common.canonical_bytes(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _build_original_split(
    rows: Sequence[common.PairRecord],
    projection: common.Projection,
) -> dict[str, Any]:
    values = []
    iids = []
    for row in rows:
        # Exactly one optimizer/evaluation row per base clip: the original
        # action-anchor quotient.  No counterfactual enters this tensor.
        quotient = common.source_relative_quotient(row)
        values.append(projection.apply(quotient).flatten())
        iids.append(row.iid)
    return {
        "value": torch.stack(values).contiguous(),
        "iids": iids,
    }


def _build_held_raw_split(rows: Sequence[common.PairRecord]) -> dict[str, Any]:
    quotients = []
    iids = []
    for row in rows:
        quotients.append(common.source_relative_quotient(row))
        iids.append(row.iid)
    return {
        "raw_quotient": torch.stack(quotients).contiguous(),
        "iids": iids,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_from_args(args)
    binding = _implementation_binding()
    pairs, feature_receipt = common.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    splits, split_receipt = common.split_pairs(pairs, config.seed)
    if split_receipt["counts"] != EXPECTED_COUNTS:
        raise ValueError(f"exact644 split counts differ: {split_receipt['counts']}")
    if len({row.family for row in pairs}) != 28:
        raise ValueError("bundle requires exact28 families")
    projection = common.fit_projection(splits["fit"], config.pca_dim)
    fit_raw_frames = torch.cat(
        [common.source_relative_quotient(row) for row in splits["fit"]], dim=0
    )
    fit_centered = fit_raw_frames - projection.mean
    fit_pca_reconstruction = (fit_centered @ projection.basis) @ projection.basis.T
    retained_variance_ratio = float(
        fit_pca_reconstruction.square().sum()
        / fit_centered.square().sum().clamp_min(EPS)
    )
    fit_pca_ceiling_mse = float(
        (fit_pca_reconstruction - fit_centered).square().mean()
    )
    fit_originals = _build_original_split(splits["fit"], projection)
    if len(fit_originals["value"]) != 452:
        raise ValueError("optimizer tensor must contain exact452 original rows")

    output = _fresh_output(args.output)
    config_value = asdict(config)
    config_sha = common.object_sha256(config_value)
    population = {
        "unique_base_clips": 644,
        "optimizer_original_rows": 452,
        "calibration_original_rows": 96,
        "locked_original_rows": 96,
        "locked_derived_diagnostic_rows_when_finalized": 384,
        "counterfactuals_are_training_samples": False,
    }
    projection_basis_sha = hashlib.sha256(
        projection.basis.numpy().tobytes(order="C")
    ).hexdigest()
    projection_sha = _projection_sha256(projection)
    train_split_authority = {
        "seed": split_receipt["seed"],
        "counts": split_receipt["counts"],
        "split_digest": split_receipt["split_digest"],
        "scientific_split_status": split_receipt["scientific_split_status"],
        "source_identity_actor_scene_generator_disjoint_verified": False,
        "locked_scientific_use_authorized": False,
    }
    train_bundle = {
        "schema_version": TRAIN_BUNDLE_SCHEMA,
        "config": config_value,
        "config_sha256": config_sha,
        "implementation": binding,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "split": train_split_authority,
        "raw_action_definition": RAW_ACTION_DEFINITION,
        "model_target": MODEL_TARGET_DESCRIPTION,
        "projection_fit_only": True,
        "projection_basis_sha256": projection_basis_sha,
        "projection_sha256": projection_sha,
        "fit_originals": fit_originals,
        "population": population,
    }
    # Calibration and locked values are physically absent from train_bundle.
    # Only the single finalize process may open held_bundle and derive its
    # counterfactual diagnostics after both arm checkpoints are sealed.
    held_bundle = {
        "schema_version": HELD_BUNDLE_SCHEMA,
        "config_sha256": config_sha,
        "implementation": binding,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "split_digest": split_receipt["split_digest"],
        "raw_action_definition": RAW_ACTION_DEFINITION,
        "model_target": MODEL_TARGET_DESCRIPTION,
        "projection": {
            "fit_only": True,
            "mean": projection.mean,
            "basis": projection.basis,
            "scale": projection.scale,
            "basis_sha256": projection_basis_sha,
            "projection_sha256": projection_sha,
            "fit_retained_variance_ratio": retained_variance_ratio,
            "fit_raw_quotient_pca_ceiling_mse": fit_pca_ceiling_mse,
        },
        "calibration_originals": _build_held_raw_split(splits["calibration"]),
        "locked_originals": _build_held_raw_split(splits["locked"]),
        "fit_iids": list(fit_originals["iids"]),
    }
    train_bundle_path = output / "train_bundle.pt"
    held_bundle_path = output / "held_eval_bundle.pt"
    train_bundle_sha = _save_torch_create_only(train_bundle_path, train_bundle)
    held_bundle_sha = _save_torch_create_only(held_bundle_path, held_bundle)
    receipt = {
        "schema_version": PREPARE_RECEIPT_SCHEMA,
        "status": "SEALED_TRAIN_AND_HELD_BUNDLES_NOT_STRUCTURED_ACTION_TARGET",
        "formal_training_authorized": False,
        "video_model_updated": False,
        "paired_ground_truth_claimed": False,
        "structured_action_target_available": False,
        "raw_action_definition": RAW_ACTION_DEFINITION,
        "model_target": MODEL_TARGET_DESCRIPTION,
        "rgb_reconstructed": False,
        "wan_vae_latent_reconstructed": False,
        "source_identity_is_target": False,
        "config": config_value,
        "config_sha256": config_sha,
        "split": train_split_authority,
        "population": population,
        "projection_fit_only": True,
        "projection_basis_sha256": projection_basis_sha,
        "projection_sha256": projection_sha,
        "projection_fit_retained_variance_ratio": retained_variance_ratio,
        "projection_fit_raw_quotient_ceiling_mse": fit_pca_ceiling_mse,
        "train_bundle": {
            "path": str(train_bundle_path.resolve()),
            "sha256": train_bundle_sha,
            "size_bytes": train_bundle_path.stat().st_size,
            "contains_fit_originals_only": True,
            "contains_calibration_or_locked_values": False,
        },
        "held_eval_bundle": {
            "path": str(held_bundle_path.resolve()),
            "sha256": held_bundle_sha,
            "size_bytes": held_bundle_path.stat().st_size,
            "contains_derived_counterfactuals": False,
            "intended_reader_policy": "single_finalize_process_only",
        },
        "feature_receipt": {
            "sha256": args.expected_feature_receipt_sha256,
            "receipt_digest": feature_receipt["receipt_digest"],
        },
        "implementation": binding,
    }
    receipt["receipt_digest"] = common.object_sha256(receipt)
    receipt_path = output / "prepare_receipt.json"
    receipt_sha = _json_create_only(receipt_path, receipt)
    os.chmod(output, 0o555)
    return {
        "status": receipt["status"],
        "train_bundle": str(train_bundle_path.resolve()),
        "train_bundle_sha256": train_bundle_sha,
        "held_eval_bundle": str(held_bundle_path.resolve()),
        "held_eval_bundle_sha256": held_bundle_sha,
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "split_counts": split_receipt["counts"],
    }


class DeterministicActionAE(nn.Module):
    """Capacity-matched deterministic reconstruction baseline."""

    def __init__(self, input_dim: int, config: Config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim), nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 2 * config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim), nn.GELU(),
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, input_dim),
        )

    def forward(self, value: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        del sample
        first, second = self.encoder(value).chunk(2, dim=1)
        latent = (first + second) / (2.0 ** 0.5)
        return {
            "latent": latent,
            "reconstruction": self.decoder(latent),
        }


class DirectBetaVAE(nn.Module):
    """Directly encode and reconstruct x_action with a standard-normal prior."""

    def __init__(self, input_dim: int, config: Config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim), nn.GELU(),
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 2 * config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim), nn.GELU(),
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, input_dim),
        )
        self.latent_dim = config.latent_dim

    def forward(self, value: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        mean, logvar = self.encoder(value).chunk(2, dim=1)
        logvar = logvar.clamp(min=-12.0, max=8.0)
        latent = mean
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return {
            "latent": latent,
            "mean": mean,
            "logvar": logvar,
            "reconstruction": self.decoder(latent),
        }

    def sample_prior(self, count: int, device: torch.device) -> torch.Tensor:
        return self.decoder(torch.randn((count, self.latent_dim), device=device))


def standard_normal_kl(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).sum(dim=1)


def _loss(
    arm: str,
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = F.mse_loss(output["reconstruction"], target)
    kl = torch.zeros((), device=target.device)
    if arm == "direct_beta_vae":
        kl = standard_normal_kl(output["mean"], output["logvar"]).mean()
    total = reconstruction + config.beta_kl * kl
    return total, {
        "reconstruction": float(reconstruction.detach()),
        "kl": float(kl.detach()),
        "total": float(total.detach()),
    }


def train_model(
    arm: str,
    model: nn.Module,
    fit: Mapping[str, Any],
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, float]], str]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4
    )
    values = fit["value"].to(device)
    # A private CPU generator makes the row schedule identical across arms;
    # VAE posterior sampling cannot perturb it.
    schedule = torch.Generator(device="cpu").manual_seed(config.seed + 100)
    schedule_digest = hashlib.sha256()
    history = []
    for step in range(config.steps):
        indices_cpu = torch.randint(
            len(values), (min(config.batch_size, len(values)),), generator=schedule
        )
        schedule_digest.update(indices_cpu.numpy().tobytes(order="C"))
        indices = indices_cpu.to(device)
        value = values[indices]
        output = model(value, sample=True)
        total, metrics = _loss(arm, output, value, config)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(config.steps // 10, 1) == 0:
            history.append({"step": step + 1, **metrics})
    return history, schedule_digest.hexdigest()


def _require_sealed(path: Path) -> None:
    stat = path.stat()
    if stat.st_nlink != 1 or (stat.st_mode & 0o777) != 0o444:
        raise ValueError(f"artifact must be mode0444/nlink1: {path}")


def _load_json_receipt(path: Path, expected_sha256: str, schema: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    _require_sealed(path)
    if common.file_sha256(path) != common._sha(expected_sha256, "receipt SHA"):
        raise ValueError("receipt file SHA differs")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != schema:
        raise ValueError("receipt schema differs")
    digest = receipt.get("receipt_digest")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    if common.object_sha256(unsigned) != common._sha(digest, "receipt digest"):
        raise ValueError("receipt self-digest differs")
    return receipt


def _same_current_implementation(recorded: Mapping[str, Any]) -> None:
    current = _implementation_binding()
    for key in ("implementation_sha256", "common_dependency_sha256"):
        if recorded.get(key) != current[key]:
            raise ValueError(f"runtime implementation binding differs: {key}")


def _load_torch_bound(
    path: Path, expected_sha256: str, expected_size: int | None = None
) -> tuple[Any, str]:
    path = path.resolve(strict=True)
    _require_sealed(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError("sealed artifact size differs")
    actual = common.file_sha256(path)
    if actual != common._sha(expected_sha256, "artifact SHA"):
        raise ValueError("sealed artifact SHA differs")
    return torch.load(path, map_location="cpu", weights_only=False), actual


def _validate_train_bundle(bundle: Mapping[str, Any]) -> Config:
    if type(bundle) is not dict or bundle.get("schema_version") != TRAIN_BUNDLE_SCHEMA:
        raise ValueError("train bundle schema differs")
    expected_keys = {
        "schema_version", "config", "config_sha256", "implementation",
        "feature_receipt_sha256", "feature_receipt_digest", "split",
        "raw_action_definition", "model_target", "projection_fit_only",
        "projection_basis_sha256", "projection_sha256", "fit_originals",
        "population",
    }
    if set(bundle) != expected_keys:
        raise ValueError("train bundle top-level keys differ")
    config_value = bundle.get("config")
    if type(config_value) is not dict:
        raise ValueError("train bundle config differs")
    config = Config(**config_value)
    config.validate()
    if bundle.get("config_sha256") != common.object_sha256(config_value):
        raise ValueError("train bundle config digest differs")
    if bundle.get("raw_action_definition") != RAW_ACTION_DEFINITION:
        raise ValueError("train bundle raw action definition differs")
    if bundle.get("model_target") != MODEL_TARGET_DESCRIPTION:
        raise ValueError("train bundle model target differs")
    if bundle.get("projection_fit_only") is not True:
        raise ValueError("train bundle projection authority differs")
    common._sha(bundle.get("projection_basis_sha256"), "PCA basis SHA")
    common._sha(bundle.get("projection_sha256"), "full projection SHA")
    if bundle.get("population") != {
        "unique_base_clips": 644,
        "optimizer_original_rows": 452,
        "calibration_original_rows": 96,
        "locked_original_rows": 96,
        "locked_derived_diagnostic_rows_when_finalized": 384,
        "counterfactuals_are_training_samples": False,
    }:
        raise ValueError("train bundle population differs")
    if bundle.get("split", {}).get("counts") != EXPECTED_COUNTS:
        raise ValueError("train bundle split counts differ")
    fit = bundle.get("fit_originals")
    if type(fit) is not dict or set(fit) != {"value", "iids"}:
        raise ValueError("train bundle must contain only fit values and IIDs")
    expected_shape = (452, 32 * config.pca_dim)
    value = fit.get("value")
    if type(value) is not torch.Tensor or tuple(value.shape) != expected_shape:
        raise ValueError("fit tensor geometry differs")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("fit tensor contains non-finite values")
    iids = fit.get("iids")
    if type(iids) is not list or len(iids) != 452 or len(set(iids)) != 452:
        raise ValueError("fit IID population differs")
    forbidden = {"family", "transform", "source_mean", "locked", "calibration"}
    if forbidden.intersection(fit) or "original_splits" in bundle:
        raise ValueError("train bundle exposes forbidden labels or held values")
    _same_current_implementation(bundle.get("implementation", {}))
    return config


def _load_prepare_for_train(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Config, str]:
    receipt_path = Path(args.prepare_receipt).resolve(strict=True)
    receipt = _load_json_receipt(
        receipt_path, args.expected_prepare_receipt_sha256, PREPARE_RECEIPT_SCHEMA
    )
    if receipt.get("status") != "SEALED_TRAIN_AND_HELD_BUNDLES_NOT_STRUCTURED_ACTION_TARGET":
        raise ValueError("prepare receipt status differs")
    binding = receipt.get("train_bundle", {})
    bundle_path = Path(args.train_bundle).resolve(strict=True)
    if Path(binding.get("path", "")).resolve(strict=True) != bundle_path:
        raise ValueError("train bundle path differs from prepare receipt")
    if binding.get("sha256") != args.expected_train_bundle_sha256:
        raise ValueError("train bundle CLI SHA differs from prepare receipt")
    bundle, bundle_sha = _load_torch_bound(
        bundle_path, args.expected_train_bundle_sha256, binding.get("size_bytes")
    )
    config = _validate_train_bundle(bundle)
    if receipt.get("config_sha256") != bundle.get("config_sha256"):
        raise ValueError("prepare/train config binding differs")
    if receipt.get("split", {}).get("split_digest") != bundle.get("split", {}).get("split_digest"):
        raise ValueError("prepare/train split binding differs")
    if receipt.get("projection_basis_sha256") != bundle.get("projection_basis_sha256"):
        raise ValueError("prepare/train PCA binding differs")
    if receipt.get("projection_sha256") != bundle.get("projection_sha256"):
        raise ValueError("prepare/train full projection binding differs")
    _same_current_implementation(receipt.get("implementation", {}))
    return receipt, bundle, config, bundle_sha


def _device(value: str) -> torch.device:
    if value != "cuda:0":
        raise RuntimeError("production train/finalize device must be logical cuda:0")
    device = torch.device(value)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("experiment arm requires exactly one visible GPU")
    if torch.cuda.get_device_name(0) != "AMD Instinct MI210":
        raise RuntimeError("experiment GPU must be AMD Instinct MI210")
    return device


def _make_model(arm: str, input_dim: int, config: Config) -> nn.Module:
    if arm == "deterministic_ae":
        return DeterministicActionAE(input_dim, config)
    if arm == "direct_beta_vae":
        return DirectBetaVAE(input_dim, config)
    raise ValueError(f"unknown arm: {arm}")


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    value: torch.Tensor,
    arm: str,
    device: torch.device,
    raw_quotient: torch.Tensor | None = None,
    projection: common.Projection | None = None,
    iids: Sequence[str] | None = None,
) -> dict[str, Any]:
    model.eval().to(device)
    target = value.to(device)
    output = model(target, sample=False)
    reconstruction = output["reconstruction"]
    per_row_mse = (reconstruction - target).square().mean(dim=1)
    cosine = F.cosine_similarity(reconstruction, target, dim=1, eps=EPS)
    latent = output["mean"] if arm == "direct_beta_vae" else output["latent"]
    result: dict[str, Any] = {
        "count": len(target),
        "reconstruction_mse": float(per_row_mse.mean()),
        "reconstruction_cosine": float(cosine.mean()),
        "zero_baseline_mse": float(target.square().mean()),
        "normalized_mse_vs_zero": float(
            per_row_mse.mean() / target.square().mean().clamp_min(EPS)
        ),
        "r_squared_vs_zero": float(
            1.0 - per_row_mse.mean() / target.square().mean().clamp_min(EPS)
        ),
        "latent_effective_rank": int(
            (torch.linalg.svdvals(latent - latent.mean(dim=0)) > 1.0e-4).sum()
        ),
    }
    if arm == "direct_beta_vae":
        kl_by_dimension = 0.5 * (
            output["mean"].square() + output["logvar"].exp()
            - output["logvar"] - 1.0
        )
        kl = kl_by_dimension.sum(dim=1)
        result["mean_kl"] = float(kl.mean())
        result["mean_kl_per_dimension"] = [
            float(value) for value in kl_by_dimension.mean(dim=0)
        ]
        result["active_latent_units"] = int(
            (output["mean"].var(dim=0, unbiased=False) > 1.0e-3).sum()
        )
    if (raw_quotient is None) != (projection is None):
        raise ValueError("raw quotient and projection must be supplied together")
    if raw_quotient is not None and projection is not None:
        raw = raw_quotient.to(device)
        basis = projection.basis.to(device)
        scale = projection.scale.to(device)
        mean = projection.mean.to(device)
        projected_target = target.reshape(len(target), 32, -1)
        projected_reconstruction = reconstruction.reshape(len(target), 32, -1)
        ceiling = (projected_target * scale) @ basis.T + mean
        inverse_model = (projected_reconstruction * scale) @ basis.T + mean
        result["raw_quotient_pca_ceiling_mse"] = float(
            (ceiling - raw).square().mean()
        )
        result["raw_quotient_model_inverse_pca_mse"] = float(
            (inverse_model - raw).square().mean()
        )
    if iids is not None:
        if len(iids) != len(per_row_mse):
            raise ValueError("evaluation IID count differs")
        result["per_iid"] = [
            {
                "iid": iid,
                "reconstruction_mse": float(mse),
                "reconstruction_cosine": float(row_cosine),
            }
            for iid, mse, row_cosine in zip(iids, per_row_mse, cosine)
        ]
    return result


def train_arm(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    prepare_receipt, bundle, config, bundle_sha = _load_prepare_for_train(args)
    binding_before = _implementation_binding()
    device = _device(args.device)
    input_dim = 32 * config.pca_dim

    # Both classes have identical trainable parameter counts, widths, decoder,
    # bottleneck dimension, steps and minibatch schedule.  The deterministic
    # encoder combines two equal heads; the beta-VAE interprets them as mean
    # and log-variance.
    with torch.random.fork_rng(devices=[]):
        deterministic_count = _parameter_count(
            DeterministicActionAE(input_dim, config)
        )
        vae_count = _parameter_count(DirectBetaVAE(input_dim, config))
    if deterministic_count != vae_count:
        raise RuntimeError("arm parameter counts differ")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = _make_model(args.arm, input_dim, config)
    history, schedule_digest = train_model(
        args.arm, model, bundle["fit_originals"], config, device
    )
    fit_metrics = _evaluate(model, bundle["fit_originals"]["value"], args.arm, device)
    if _implementation_binding() != binding_before:
        raise RuntimeError("implementation changed during arm training")

    output = _fresh_output(args.output)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "arm": args.arm,
        "config": asdict(config),
        "config_sha256": bundle["config_sha256"],
        "train_bundle_sha256": bundle_sha,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "split_digest": bundle["split"]["split_digest"],
        "projection_basis_sha256": bundle["projection_basis_sha256"],
        "projection_sha256": bundle["projection_sha256"],
        "implementation": binding_before,
        "model_state": common._cpu_state_dict(model),
    }
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_sha = _save_torch_create_only(checkpoint_path, checkpoint)
    receipt: dict[str, Any] = {
        "schema_version": ARM_RESULT_SCHEMA,
        "status": "ARM_TRAINED_FIT_ONLY_NOT_EVALUATED_ON_HELD",
        "arm": args.arm,
        "formal_training_authorized": False,
        "video_model_updated": False,
        "paired_ground_truth_claimed": False,
        "structured_action_target_available": False,
        "raw_action_definition": RAW_ACTION_DEFINITION,
        "model_target": MODEL_TARGET_DESCRIPTION,
        "optimizer_population": {
            "unique_fit_base_clips": 452,
            "original_rows": 452,
            "derived_rows": 0,
        },
        "method": {
            "direct_target_reconstruction": True,
            "rgb_reconstructed": False,
            "wan_vae_latent_reconstructed": False,
            "family_labels_used_only_for_stratified_split_during_prepare": True,
            "family_or_transform_labels_consumed_by_model_or_optimizer": False,
            "source_identity_preservation_tested": False,
            "held_values_present_in_train_bundle": False,
            "held_values_read_by_training_code": False,
            "single_execution_per_iid": True,
        },
        "config": asdict(config),
        "config_sha256": bundle["config_sha256"],
        "parameter_count": _parameter_count(model),
        "parameter_count_exactly_matched_across_arms": True,
        "minibatch_schedule_sha256": schedule_digest,
        "train_bundle_sha256": bundle_sha,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "split_digest": bundle["split"]["split_digest"],
        "projection_basis_sha256": bundle["projection_basis_sha256"],
        "projection_sha256": bundle["projection_sha256"],
        "training_history": history,
        "fit_metrics_not_held": fit_metrics,
        "held_metrics": None,
        "gates": {
            "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION_PER_IID",
            "vae_necessary": None,
            "renderer_training_authorized": False,
        },
        "device": {
            "requested": str(device),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "hostname": os.uname().nodename,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "implementation": binding_before,
    }
    receipt["receipt_digest"] = common.object_sha256(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _json_create_only(receipt_path, receipt)
    os.chmod(output, 0o555)
    return {
        "status": receipt["status"],
        "arm": args.arm,
        "checkpoint_sha256": checkpoint_sha,
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "minibatch_schedule_sha256": schedule_digest,
    }


def _validate_held_bundle(
    bundle: Mapping[str, Any], prepare_receipt: Mapping[str, Any], config: Config
) -> common.Projection:
    if type(bundle) is not dict or bundle.get("schema_version") != HELD_BUNDLE_SCHEMA:
        raise ValueError("held bundle schema differs")
    expected_keys = {
        "schema_version", "config_sha256", "implementation",
        "feature_receipt_sha256", "split_digest", "raw_action_definition",
        "model_target", "projection", "calibration_originals",
        "locked_originals", "fit_iids",
    }
    if set(bundle) != expected_keys:
        raise ValueError("held bundle top-level keys differ")
    if bundle.get("config_sha256") != prepare_receipt.get("config_sha256"):
        raise ValueError("held bundle config binding differs")
    if bundle.get("split_digest") != prepare_receipt.get("split", {}).get("split_digest"):
        raise ValueError("held bundle split binding differs")
    if bundle.get("raw_action_definition") != RAW_ACTION_DEFINITION:
        raise ValueError("held bundle raw action definition differs")
    if bundle.get("model_target") != MODEL_TARGET_DESCRIPTION:
        raise ValueError("held bundle model target differs")
    _same_current_implementation(bundle.get("implementation", {}))
    projection_value = bundle.get("projection")
    if type(projection_value) is not dict or projection_value.get("fit_only") is not True:
        raise ValueError("held projection authority differs")
    if set(projection_value) != {
        "fit_only", "mean", "basis", "scale", "basis_sha256",
        "projection_sha256", "fit_retained_variance_ratio",
        "fit_raw_quotient_pca_ceiling_mse",
    }:
        raise ValueError("held projection keys differ")
    mean = projection_value.get("mean")
    basis = projection_value.get("basis")
    scale = projection_value.get("scale")
    if type(mean) is not torch.Tensor or tuple(mean.shape) != (1, 768):
        raise ValueError("held projection mean geometry differs")
    if type(basis) is not torch.Tensor or tuple(basis.shape) != (768, config.pca_dim):
        raise ValueError("held projection basis geometry differs")
    if type(scale) is not torch.Tensor or tuple(scale.shape) != (config.pca_dim,):
        raise ValueError("held projection scale geometry differs")
    if not all(bool(torch.isfinite(value).all()) for value in (mean, basis, scale)):
        raise ValueError("held projection contains non-finite values")
    basis_sha = hashlib.sha256(basis.numpy().tobytes(order="C")).hexdigest()
    if basis_sha != projection_value.get("basis_sha256"):
        raise ValueError("held projection basis digest differs")
    if basis_sha != prepare_receipt.get("projection_basis_sha256"):
        raise ValueError("held/prepare PCA binding differs")
    projection = common.Projection(mean=mean, basis=basis, scale=scale)
    full_projection_sha = _projection_sha256(projection)
    if full_projection_sha != projection_value.get("projection_sha256"):
        raise ValueError("held full projection digest differs")
    if full_projection_sha != prepare_receipt.get("projection_sha256"):
        raise ValueError("held/prepare full projection binding differs")
    retained = projection_value.get("fit_retained_variance_ratio")
    ceiling = projection_value.get("fit_raw_quotient_pca_ceiling_mse")
    if type(retained) is not float or not 0.0 <= retained <= 1.000001:
        raise ValueError("fit PCA retained variance differs")
    if type(ceiling) is not float or not math.isfinite(ceiling) or ceiling < 0.0:
        raise ValueError("fit PCA ceiling differs")
    fit_iids = bundle.get("fit_iids")
    if type(fit_iids) is not list or len(fit_iids) != 452 or len(set(fit_iids)) != 452:
        raise ValueError("held fit IID authority differs")
    seen: set[str] = set(fit_iids)
    split_ids = {"fit": fit_iids}
    for name, count in (("calibration_originals", 96), ("locked_originals", 96)):
        value = bundle.get(name)
        if type(value) is not dict or set(value) != {"raw_quotient", "iids"}:
            raise ValueError(f"{name} payload differs")
        raw = value.get("raw_quotient")
        iids = value.get("iids")
        if type(raw) is not torch.Tensor or tuple(raw.shape) != (count, 32, 768):
            raise ValueError(f"{name} geometry differs")
        if not bool(torch.isfinite(raw).all()):
            raise ValueError(f"{name} contains non-finite values")
        if type(iids) is not list or len(iids) != count or len(set(iids)) != count:
            raise ValueError(f"{name} IID population differs")
        if seen.intersection(iids):
            raise ValueError("fit/calibration/locked IIDs overlap")
        seen.update(iids)
        split_ids[name.removesuffix("_originals")] = iids
    if len(seen) != 644:
        raise ValueError("held split IID closure is not exact644")
    if common.object_sha256(split_ids) != bundle.get("split_digest"):
        raise ValueError("held split IID digest differs")
    return projection


def _project_raw_originals(
    raw_split: Mapping[str, Any], projection: common.Projection
) -> dict[str, Any]:
    values = [projection.apply(value).flatten() for value in raw_split["raw_quotient"]]
    return {"value": torch.stack(values).contiguous(), "iids": list(raw_split["iids"])}


def _derive_locked_diagnostics(
    raw_split: Mapping[str, Any], projection: common.Projection, seed: int
) -> dict[str, Any]:
    values = []
    names = []
    iids = []
    for quotient, iid in zip(raw_split["raw_quotient"], raw_split["iids"]):
        derived = common.counterfactuals(quotient, iid, seed)
        for name in common.COUNTERFACTUAL_TRANSFORMS:
            values.append(projection.apply(derived[name]).flatten())
            names.append(name)
            iids.append(iid)
    if len(values) != 384:
        raise ValueError("finalized locked diagnostics must contain exact384 rows")
    return {"value": torch.stack(values).contiguous(), "transform": names, "iids": iids}


def _load_arm_for_finalize(
    arm: str,
    receipt_path_string: str,
    expected_receipt_sha256: str,
    prepare_receipt_sha256: str,
    train_bundle_sha256: str,
    projection_sha256: str,
    config: Config,
    input_dim: int,
) -> tuple[nn.Module, dict[str, Any], str]:
    receipt = _load_json_receipt(
        Path(receipt_path_string), expected_receipt_sha256, ARM_RESULT_SCHEMA
    )
    if receipt.get("status") != "ARM_TRAINED_FIT_ONLY_NOT_EVALUATED_ON_HELD":
        raise ValueError("arm receipt status differs")
    if receipt.get("arm") != arm:
        raise ValueError("arm receipt identity differs")
    if receipt.get("config_sha256") != common.object_sha256(asdict(config)):
        raise ValueError("arm config binding differs")
    if receipt.get("prepare_receipt_sha256") != prepare_receipt_sha256:
        raise ValueError("arm prepare receipt binding differs")
    if receipt.get("train_bundle_sha256") != train_bundle_sha256:
        raise ValueError("arm train bundle binding differs")
    if receipt.get("projection_sha256") != projection_sha256:
        raise ValueError("arm full projection binding differs")
    if receipt.get("held_metrics") is not None:
        raise ValueError("arm receipt accessed held metrics before finalize")
    if receipt.get("method", {}).get("held_values_read_by_training_code") is not False:
        raise ValueError("arm held isolation claim differs")
    _same_current_implementation(receipt.get("implementation", {}))
    checkpoint_binding = receipt.get("checkpoint", {})
    checkpoint, checkpoint_sha = _load_torch_bound(
        Path(checkpoint_binding.get("path", "")),
        checkpoint_binding.get("sha256", ""),
        checkpoint_binding.get("size_bytes"),
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or checkpoint.get("arm") != arm:
        raise ValueError("arm checkpoint identity differs")
    for key, expected in (
        ("config_sha256", common.object_sha256(asdict(config))),
        ("train_bundle_sha256", train_bundle_sha256),
        ("prepare_receipt_sha256", prepare_receipt_sha256),
        ("projection_sha256", projection_sha256),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(f"arm checkpoint binding differs: {key}")
    _same_current_implementation(checkpoint.get("implementation", {}))
    model = _make_model(arm, input_dim, config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if _parameter_count(model) != receipt.get("parameter_count"):
        raise ValueError("arm parameter count differs")
    return model, receipt, checkpoint_sha


@torch.no_grad()
def _evaluate_diagnostics(
    model: nn.Module,
    data: Mapping[str, Any],
    arm: str,
    device: torch.device,
) -> dict[str, Any]:
    model.eval().to(device)
    value = data["value"].to(device)
    output = model(value, sample=False)
    reconstruction = output["reconstruction"]
    mse = (reconstruction - value).square().mean(dim=1)
    cosine = F.cosine_similarity(reconstruction, value, dim=1, eps=EPS)
    result = {"count": len(value), "by_transform": {}}
    for name in common.COUNTERFACTUAL_TRANSFORMS:
        mask = torch.tensor(
            [row == name for row in data["transform"]], dtype=torch.bool, device=device
        )
        result["by_transform"][name] = {
            "count": int(mask.sum()),
            "reconstruction_mse": float(mse[mask].mean()),
            "reconstruction_cosine": float(cosine[mask].mean()),
        }
    result["diagnostic_rows_used_for_optimization"] = 0
    return result


@torch.no_grad()
def _prior_diagnostics(
    model: DirectBetaVAE,
    target: torch.Tensor,
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    target = target.to(device)
    torch.manual_seed(config.seed + 500)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + 500)
    samples = torch.stack(
        [model.sample_prior(len(target), device) for _ in range(config.prior_samples)],
        dim=0,
    )
    errors = (samples - target.unsqueeze(0)).square().mean(dim=2)
    differences = []
    for left in range(config.prior_samples):
        for right in range(left + 1, config.prior_samples):
            differences.append((samples[left] - samples[right]).square().mean(dim=1))
    return {
        "prior_kind": "unconditional_standard_normal",
        "unconditional_random_pairing_best_of_k_target_mse": float(
            errors.min(dim=0).values.mean()
        ),
        "pairwise_diversity_mse": float(torch.stack(differences).mean()),
        "one_to_many_coverage_evaluated": False,
        "interpretation_limit": "random unconditional samples are not paired valid executions",
    }


def _paired_bootstrap(
    ae_per_iid: Sequence[Mapping[str, Any]],
    vae_per_iid: Sequence[Mapping[str, Any]],
    seed: int,
    draws: int = 10000,
) -> dict[str, Any]:
    ae_map = {row["iid"]: float(row["reconstruction_mse"]) for row in ae_per_iid}
    vae_map = {row["iid"]: float(row["reconstruction_mse"]) for row in vae_per_iid}
    if set(ae_map) != set(vae_map) or len(ae_map) != 96:
        raise ValueError("paired locked IID metrics do not form exact96 pairs")
    iids = sorted(ae_map)
    ae = torch.tensor([ae_map[iid] for iid in iids], dtype=torch.float64)
    vae = torch.tensor([vae_map[iid] for iid in iids], dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(len(iids), (draws, len(iids)), generator=generator)
    boot_ae = ae[indices].mean(dim=1)
    boot_vae = vae[indices].mean(dim=1)
    boot_delta = boot_vae - boot_ae
    boot_ratio = boot_vae / boot_ae.clamp_min(EPS)
    return {
        "iid_count": len(iids),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "delta_definition": "direct_beta_vae_mse_minus_deterministic_ae_mse",
        "mean_delta_mse": float((vae - ae).mean()),
        "delta_mse_95pct_ci": [
            float(torch.quantile(boot_delta, 0.025)),
            float(torch.quantile(boot_delta, 0.975)),
        ],
        "mean_mse_ratio": float(vae.mean() / ae.mean().clamp_min(EPS)),
        "mse_ratio_95pct_ci": [
            float(torch.quantile(boot_ratio, 0.025)),
            float(torch.quantile(boot_ratio, 0.975)),
        ],
        "paired_iid_order_sha256": common.object_sha256(iids),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    binding_before = _implementation_binding()
    prepare_receipt = _load_json_receipt(
        Path(args.prepare_receipt),
        args.expected_prepare_receipt_sha256,
        PREPARE_RECEIPT_SCHEMA,
    )
    _same_current_implementation(prepare_receipt.get("implementation", {}))
    config = Config(**prepare_receipt["config"])
    config.validate()
    held_binding = prepare_receipt.get("held_eval_bundle", {})
    held_path = Path(args.held_eval_bundle).resolve(strict=True)
    if Path(held_binding.get("path", "")).resolve(strict=True) != held_path:
        raise ValueError("held bundle path differs from prepare receipt")
    if held_binding.get("sha256") != args.expected_held_eval_bundle_sha256:
        raise ValueError("held bundle CLI SHA differs from prepare receipt")

    train_bundle_sha = prepare_receipt["train_bundle"]["sha256"]
    input_dim = 32 * config.pca_dim
    deterministic, deterministic_receipt, deterministic_checkpoint_sha = _load_arm_for_finalize(
        "deterministic_ae",
        args.deterministic_arm_receipt,
        args.expected_deterministic_arm_receipt_sha256,
        args.expected_prepare_receipt_sha256,
        train_bundle_sha,
        prepare_receipt["projection_sha256"],
        config,
        input_dim,
    )
    direct_vae, vae_receipt, vae_checkpoint_sha = _load_arm_for_finalize(
        "direct_beta_vae",
        args.direct_beta_vae_arm_receipt,
        args.expected_direct_beta_vae_arm_receipt_sha256,
        args.expected_prepare_receipt_sha256,
        train_bundle_sha,
        prepare_receipt["projection_sha256"],
        config,
        input_dim,
    )
    if deterministic_receipt["minibatch_schedule_sha256"] != vae_receipt["minibatch_schedule_sha256"]:
        raise ValueError("arm minibatch schedules differ")
    if deterministic_receipt["parameter_count"] != vae_receipt["parameter_count"]:
        raise ValueError("arm parameter counts differ")

    device = _device(args.device)
    _preflight_fresh_output(args.output)

    # Arm/checkpoint/schedule/device/output preflight is complete before this
    # invocation opens held values.  This is one load by this finalize process;
    # it is not an OS-level exclusion guarantee for other same-user processes.
    held_bundle, held_sha = _load_torch_bound(
        held_path,
        args.expected_held_eval_bundle_sha256,
        held_binding.get("size_bytes"),
    )
    projection = _validate_held_bundle(held_bundle, prepare_receipt, config)
    calibration = _project_raw_originals(
        held_bundle["calibration_originals"], projection
    )
    locked = _project_raw_originals(held_bundle["locked_originals"], projection)
    locked_diagnostics = _derive_locked_diagnostics(
        held_bundle["locked_originals"], projection, config.seed
    )

    metrics = {
        "calibration": {
            "deterministic_ae": _evaluate(
                deterministic, calibration["value"], "deterministic_ae", device,
                held_bundle["calibration_originals"]["raw_quotient"], projection,
                calibration["iids"],
            ),
            "direct_beta_vae": _evaluate(
                direct_vae, calibration["value"], "direct_beta_vae", device,
                held_bundle["calibration_originals"]["raw_quotient"], projection,
                calibration["iids"],
            ),
        },
        "locked": {
            "deterministic_ae": _evaluate(
                deterministic, locked["value"], "deterministic_ae", device,
                held_bundle["locked_originals"]["raw_quotient"], projection,
                locked["iids"],
            ),
            "direct_beta_vae": _evaluate(
                direct_vae, locked["value"], "direct_beta_vae", device,
                held_bundle["locked_originals"]["raw_quotient"], projection,
                locked["iids"],
            ),
        },
        "locked_derived_diagnostics": {
            "deterministic_ae": _evaluate_diagnostics(
                deterministic, locked_diagnostics, "deterministic_ae", device
            ),
            "direct_beta_vae": _evaluate_diagnostics(
                direct_vae, locked_diagnostics, "direct_beta_vae", device
            ),
        },
    }
    metrics["locked"]["direct_beta_vae"]["prior_diagnostics"] = _prior_diagnostics(
        direct_vae, locked["value"], config, device
    )
    locked_ae_mse = metrics["locked"]["deterministic_ae"]["reconstruction_mse"]
    locked_vae_mse = metrics["locked"]["direct_beta_vae"]["reconstruction_mse"]
    paired = _paired_bootstrap(
        metrics["locked"]["deterministic_ae"]["per_iid"],
        metrics["locked"]["direct_beta_vae"]["per_iid"],
        config.seed + 900,
    )
    ratio_ucb = paired["mse_ratio_95pct_ci"][1]
    locked_vae = metrics["locked"]["direct_beta_vae"]
    posterior_retention_gate = bool(ratio_ucb <= 1.02)
    posterior_noncollapse_gate = bool(
        locked_vae["mean_kl"] > 1.0e-4
        and locked_vae["active_latent_units"] > 0
    )
    comparison = {
        "locked_vae_to_ae_mse_ratio": locked_vae_mse / max(locked_ae_mse, EPS),
        "locked_vae_lower_mse": bool(locked_vae_mse < locked_ae_mse),
        "paired_bootstrap": paired,
        "interpretation_limit": "mechanics comparison only; necessity requires multiple valid executions per condition",
    }
    if _implementation_binding() != binding_before:
        raise RuntimeError("implementation changed during finalization")

    output = _fresh_output(args.output)
    receipt: dict[str, Any] = {
        "schema_version": FINAL_RESULT_SCHEMA,
        "status": "DIRECT_RECONSTRUCTION_COMPARISON_COMPLETE_NOT_ACTION_QUALIFIED",
        "formal_training_authorized": False,
        "video_model_updated": False,
        "video_edit_rendered": False,
        "paired_ground_truth_claimed": False,
        "structured_action_target_available": False,
        "participant_role_phase_terminal_qualified": False,
        "raw_action_definition": RAW_ACTION_DEFINITION,
        "model_target": MODEL_TARGET_DESCRIPTION,
        "method": {
            "optimizer_uses_original_action_anchors_only": True,
            "optimizer_base_clip_count": 452,
            "rgb_reconstructed": False,
            "wan_vae_latent_reconstructed": False,
            "family_labels_used_only_for_stratified_split_during_prepare": True,
            "family_or_transform_labels_consumed_by_model_or_optimizer": False,
            "source_identity_preservation_tested": False,
            "held_bundle_loaded_by_train_arms": False,
            "held_bundle_loaded_once_by_this_finalize_invocation": True,
            "global_os_level_single_reader_enforced": False,
            "locked_evaluation_phase_count_this_invocation": 1,
            "locked_original_rows": 96,
            "locked_derived_diagnostic_rows": 384,
            "locked_counterfactuals_derived_only_during_finalize": True,
            "locked_counterfactuals_are_samples": False,
            "locked_counterfactuals_are_independent_samples": False,
            "counterfactual_action_sensitivity_qualified": False,
            "counterfactual_interpretation_limit": "OOD self-reconstruction diagnostic only",
            "single_execution_per_iid": True,
        },
        "config": asdict(config),
        "config_sha256": common.object_sha256(asdict(config)),
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "train_bundle_sha256": train_bundle_sha,
        "held_eval_bundle_sha256": held_sha,
        "arm_receipts": {
            "deterministic_ae": args.expected_deterministic_arm_receipt_sha256,
            "direct_beta_vae": args.expected_direct_beta_vae_arm_receipt_sha256,
        },
        "arm_checkpoints": {
            "deterministic_ae": deterministic_checkpoint_sha,
            "direct_beta_vae": vae_checkpoint_sha,
        },
        "parameter_count_per_arm": deterministic_receipt["parameter_count"],
        "parameter_count_exactly_matched": True,
        "minibatch_schedule_sha256": deterministic_receipt["minibatch_schedule_sha256"],
        "metrics": metrics,
        "comparison": comparison,
        "gates": {
            "exact644_split_closure_verified": True,
            "optimizer_exact452_original_only_verified": True,
            "family_or_transform_labels_not_consumed_by_model_or_optimizer_verified": True,
            "same_parameter_count_verified": True,
            "same_minibatch_schedule_verified": True,
            "held_values_not_read_by_train_code_verified": True,
            "posterior_reconstruction_retention_ratio_ucb_threshold": 1.02,
            "posterior_reconstruction_retention_ratio_ucb": ratio_ucb,
            "posterior_reconstruction_retention_gate": posterior_retention_gate,
            "posterior_noncollapse_gate": posterior_noncollapse_gate,
            "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION_PER_IID",
            "vae_necessary": None,
            "structured_action_target_available": False,
            "renderer_training_authorized": False,
        },
        "device": {
            "requested": str(device),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "hostname": os.uname().nodename,
        },
        "implementation": binding_before,
    }
    receipt["receipt_digest"] = common.object_sha256(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _json_create_only(receipt_path, receipt)
    os.chmod(output, 0o555)
    return {
        "status": receipt["status"],
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "locked_vae_to_ae_mse_ratio": comparison["locked_vae_to_ae_mse_ratio"],
        "vae_necessity_status": receipt["gates"]["vae_necessity_status"],
        "vae_necessary": None,
    }


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--beta-kl", type=float, default=0.02)
    parser.add_argument("--prior-samples", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--feature-root", required=True)
    prepare_parser.add_argument("--expected-feature-receipt-sha256", required=True)
    prepare_parser.add_argument("--output", required=True)
    _add_config_arguments(prepare_parser)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--prepare-receipt", required=True)
    train_parser.add_argument("--expected-prepare-receipt-sha256", required=True)
    train_parser.add_argument("--train-bundle", required=True)
    train_parser.add_argument("--expected-train-bundle-sha256", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--device", default="cuda:0")

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--prepare-receipt", required=True)
    finalize_parser.add_argument("--expected-prepare-receipt-sha256", required=True)
    finalize_parser.add_argument("--held-eval-bundle", required=True)
    finalize_parser.add_argument("--expected-held-eval-bundle-sha256", required=True)
    finalize_parser.add_argument("--deterministic-arm-receipt", required=True)
    finalize_parser.add_argument("--expected-deterministic-arm-receipt-sha256", required=True)
    finalize_parser.add_argument("--direct-beta-vae-arm-receipt", required=True)
    finalize_parser.add_argument("--expected-direct-beta-vae-arm-receipt-sha256", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "train":
        result = train_arm(args)
    elif args.command == "finalize":
        result = finalize(args)
    else:  # pragma: no cover - argparse makes this unreachable
        raise RuntimeError("unknown command")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
