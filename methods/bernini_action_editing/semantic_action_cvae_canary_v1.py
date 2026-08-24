#!/usr/bin/env python3
"""Exact644 feature-level deterministic-AE versus conditional-VAE canary.

The diagnostic reconstructs only a source-relative, temporally ordered frozen
feature quotient.  It never reconstructs RGB, a Wan video-VAE latent, a
Bernini hidden state, or a paired edited target.  A successful run therefore
answers only whether this bottleneck has useful representation mechanics; it
does not authorize video-model training or a scientific action claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


FEATURE_SCHEMA = "semantic-moments-action-reward-features-v1"
RECEIPT_SCHEMA = "semantic-action-exact644-feature-extraction-receipt-v1"
RESULT_SCHEMA = "semantic-action-cvae-exact644-canary-receipt-v2"
SOURCE_MANIFEST_DIGEST = "96fe6188ad0f5ee72dcd89fbc018835f3f2995e45ff116f07449e863fa9b51d5"
DINO_WEIGHTS_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
ACTION_GROUP = "exact644_action_anchor"
SOURCE_GROUP = "exact644_source"
TRANSFORMS = ("original", "reverse", "shuffle", "zero", "tail_hold")
ORIGINAL_TRANSFORM = "original"
COUNTERFACTUAL_TRANSFORMS = TRANSFORMS[1:]
EPS = 1.0e-8


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_fp32(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if type(value) is not torch.Tensor or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


@dataclass(frozen=True)
class PairRecord:
    iid: str
    family: str
    instruction_sha256: str
    group_id: str
    strict: bool
    anchor_sequence: torch.Tensor
    source_sequence: torch.Tensor


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
    transform_weight: float = 0.30
    family_weight: float = 0.15
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
        if not 0.0 <= self.beta_kl <= 1.0:
            raise ValueError("beta_kl must be in [0,1]")
        if self.prior_samples < 2:
            raise ValueError("prior_samples must be at least two")


def _verify_receipt_digest(receipt: Mapping[str, Any]) -> None:
    expected = _sha(receipt.get("receipt_digest"), "receipt_digest")
    unsigned = dict(receipt)
    del unsigned["receipt_digest"]
    if object_sha256(unsigned) != expected:
        raise ValueError("feature extraction receipt self-digest differs")


def _join_exact644_records(records: Sequence[Mapping[str, Any]]) -> list[PairRecord]:
    """Validate the exact row/role closure before forming source-anchor pairs."""

    if len(records) != 1288:
        raise ValueError("loaded feature population is not exact1288")
    by_iid: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    item_ids: set[str] = set()
    for row in records:
        if type(row) is not dict:
            raise ValueError("feature record must be an object")
        item_id = row.get("item_id")
        if type(item_id) is not str or item_id in item_ids:
            raise ValueError("feature item IDs must be unique strings")
        item_ids.add(item_id)
        group = row.get("group")
        metadata = row.get("metadata")
        if type(metadata) is not dict:
            raise ValueError("feature metadata must be an object")
        if metadata.get("source_manifest_digest") != SOURCE_MANIFEST_DIGEST:
            raise ValueError("feature metadata source release differs")
        if metadata.get("paired_ground_truth_claimed") is not False:
            raise ValueError("feature metadata must not claim paired truth")
        iid = metadata.get("iid")
        role = metadata.get("role")
        if type(iid) is not str or role not in {"source", "action_anchor"}:
            raise ValueError("feature IID/role differs")
        expected_group = SOURCE_GROUP if role == "source" else ACTION_GROUP
        if group != expected_group:
            raise ValueError("feature role/group binding differs")
        if item_id != f"exact644:{iid}:{role}":
            raise ValueError("feature item ID does not bind IID/role")
        if metadata.get("base_video_id") != iid:
            raise ValueError("feature base_video_id does not bind IID")
        if type(metadata.get("family")) is not str or not metadata["family"]:
            raise ValueError("feature family must be a non-empty string")
        _sha(metadata.get("instruction_sha256"), "instruction SHA")
        _sha(metadata.get("group_id"), "group ID")
        if type(metadata.get("strict_selection_gates_all_true")) is not bool:
            raise ValueError("strict selection gate must be boolean")
        if role in by_iid[iid]:
            raise ValueError("duplicate IID role")
        by_iid[iid][role] = row

    if len(by_iid) != 644:
        raise ValueError("feature population must have exact644 IIDs")
    pairs = []
    for iid, roles in sorted(by_iid.items()):
        if set(roles) != {"source", "action_anchor"}:
            raise ValueError("every IID must contain source and action_anchor")
        source = roles["source"]
        anchor = roles["action_anchor"]
        sm = source["metadata"]
        am = anchor["metadata"]
        for key in (
            "iid",
            "base_video_id",
            "group_id",
            "family",
            "instruction_sha256",
            "strict_selection_gates_all_true",
            "source_manifest_digest",
            "paired_ground_truth_claimed",
        ):
            if sm.get(key) != am.get(key):
                raise ValueError(f"source/action metadata join differs: {key}")
        pairs.append(
            PairRecord(
                iid=iid,
                family=am["family"],
                instruction_sha256=_sha(am["instruction_sha256"], "instruction SHA"),
                group_id=_sha(am["group_id"], "group ID"),
                strict=am["strict_selection_gates_all_true"],
                anchor_sequence=_finite_fp32(
                    anchor["frame_sequence"], (32, 768), "anchor sequence"
                ),
                source_sequence=_finite_fp32(
                    source["frame_sequence"], (32, 768), "source sequence"
                ),
            )
        )
    return pairs


def load_exact644_pairs(
    feature_root: Path, expected_receipt_sha256: str
) -> tuple[list[PairRecord], dict[str, Any]]:
    """Load exact644 held feature pairs after verifying the sealed receipt."""

    feature_root = feature_root.resolve(strict=True)
    receipt_path = feature_root / "feature_extraction_receipt.json"
    if file_sha256(receipt_path) != _sha(
        expected_receipt_sha256, "expected feature receipt SHA"
    ):
        raise ValueError("feature extraction receipt file SHA differs")
    stat = receipt_path.stat()
    if stat.st_nlink != 1 or (stat.st_mode & 0o777) != 0o444:
        raise ValueError("feature extraction receipt must be mode0444/nlink1")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("feature extraction receipt schema differs")
    if receipt.get("status") != "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED":
        raise ValueError("feature extraction receipt status differs")
    if receipt.get("formal_training_authorized") is not False:
        raise ValueError("feature release must not authorize formal training")
    if receipt.get("paired_ground_truth_claimed") is not False:
        raise ValueError("feature release must not claim paired ground truth")
    _verify_receipt_digest(receipt)
    population = receipt.get("population")
    if population != {
        "unique_base_clips": 644,
        "action_anchor_records": 644,
        "source_records_for_nuisance_probe": 644,
        "total_feature_records": 1288,
        "counterfactual_rows": 0,
    }:
        raise ValueError("feature receipt population differs")
    if receipt.get("frozen_teacher") != {
        "kind": "DINOv2-base ordered per-frame descriptors",
        "weights_sha256": DINO_WEIGHTS_SHA256,
        "semantic_moments_role": "unordered auxiliary only",
    }:
        raise ValueError("feature receipt frozen teacher differs")

    shard_rows = receipt.get("shards")
    if type(shard_rows) is not list or len(shard_rows) != 8:
        raise ValueError("feature receipt must bind exact8 shards")
    records: list[Mapping[str, Any]] = []
    for index, binding in enumerate(shard_rows):
        if type(binding) is not dict or binding.get("index") != index:
            raise ValueError("feature shard receipt placement differs")
        path = (feature_root / "features" / f"features-shard-{index}.pt").resolve(
            strict=True
        )
        if Path(binding.get("path", "")).resolve(strict=True) != path:
            raise ValueError("feature shard path differs from receipt")
        if path.stat().st_nlink != 1 or (path.stat().st_mode & 0o777) != 0o444:
            raise ValueError("feature shards must be mode0444/nlink1")
        if path.stat().st_size != binding.get("size_bytes"):
            raise ValueError("feature shard size differs")
        if file_sha256(path) != _sha(binding.get("sha256"), "feature shard SHA"):
            raise ValueError("feature shard SHA differs")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != FEATURE_SCHEMA:
            raise ValueError("feature shard schema differs")
        if payload.get("shard_index") != index or payload.get("num_shards") != 8:
            raise ValueError("feature shard placement differs")
        if payload.get("record_count") != 161 or len(payload.get("records", [])) != 161:
            raise ValueError("feature shard count differs")
        records.extend(payload["records"])

    return _join_exact644_records(records), receipt


def split_pairs(
    pairs: Sequence[PairRecord], seed: int
) -> tuple[dict[str, list[PairRecord]], dict[str, Any]]:
    """Stable per-family IID split; counterfactuals are created only afterwards."""

    if len(pairs) != 644 or len({row.iid for row in pairs}) != 644:
        raise ValueError("split input must contain exact644 unique IIDs")
    by_family: dict[str, list[PairRecord]] = defaultdict(list)
    for row in pairs:
        by_family[row.family].append(row)
    if len(by_family) != 28:
        raise ValueError("split input must contain exact28 families")
    output = {"fit": [], "calibration": [], "locked": []}
    family_counts = {}
    abstain = []
    for family, rows in sorted(by_family.items()):
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{family}:{row.iid}".encode("utf-8")
            ).hexdigest(),
        )
        count = len(ordered)
        if count < 3:
            n_calibration = 0
            n_locked = 0
            abstain.append(family)
        else:
            n_calibration = max(1, int(round(count * 0.15)))
            n_locked = max(1, int(round(count * 0.15)))
            while count - n_calibration - n_locked < 1:
                if n_calibration >= n_locked and n_calibration > 1:
                    n_calibration -= 1
                elif n_locked > 1:
                    n_locked -= 1
                else:
                    raise ValueError("family cannot retain a fit row")
        n_fit = count - n_calibration - n_locked
        output["fit"].extend(ordered[:n_fit])
        output["calibration"].extend(ordered[n_fit : n_fit + n_calibration])
        output["locked"].extend(ordered[n_fit + n_calibration :])
        family_counts[family] = {
            "total": count,
            "fit": n_fit,
            "calibration": n_calibration,
            "locked": n_locked,
            "status": (
                "ABSTAIN_INSUFFICIENT_GROUPS" if count < 3 else "THREE_WAY_SPLIT"
            ),
        }
    all_ids = [row.iid for rows in output.values() for row in rows]
    if len(all_ids) != 644 or len(set(all_ids)) != 644:
        raise ValueError("split must be exhaustive and disjoint")
    return output, {
        "algorithm": "per-family sha256(seed:family:iid), approx70/15/15",
        "seed": seed,
        "scientific_split_status": "IID_DISJOINT_ONLY_NOT_CONTENT_DISJOINT",
        "source_identity_actor_scene_generator_disjoint_verified": False,
        "locked_scientific_use_authorized": False,
        "counts": {key: len(value) for key, value in output.items()},
        "family_counts": family_counts,
        "abstain_insufficient_groups": abstain,
        "split_digest": object_sha256(
            {key: [row.iid for row in value] for key, value in output.items()}
        ),
    }


def source_relative_quotient(pair: PairRecord) -> torch.Tensor:
    anchor = pair.anchor_sequence - pair.anchor_sequence.mean(dim=0, keepdim=True)
    source = pair.source_sequence - pair.source_sequence.mean(dim=0, keepdim=True)
    value = (anchor - source).contiguous()
    if tuple(value.shape) != (32, 768) or not bool(torch.isfinite(value).all()):
        raise ValueError("source-relative quotient is invalid")
    return value


def deterministic_permutation(iid: str, count: int, seed: int) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:shuffle:{iid}".encode()).hexdigest()
    generator = torch.Generator().manual_seed(int(digest[:16], 16))
    permutation = torch.randperm(count, generator=generator)
    natural = torch.arange(count)
    if torch.equal(permutation, natural) or torch.equal(permutation, natural.flip(0)):
        permutation = natural.roll(1)
    return permutation


def counterfactuals(value: torch.Tensor, iid: str, seed: int) -> dict[str, torch.Tensor]:
    if tuple(value.shape) != (32, 768):
        raise ValueError("counterfactual input geometry differs")
    half = value.shape[0] // 2
    return {
        "original": value,
        "reverse": value.flip(0),
        "shuffle": value[deterministic_permutation(iid, len(value), seed)],
        "zero": torch.zeros_like(value),
        "tail_hold": torch.cat(
            [value[:half], value[half - 1 : half].repeat(len(value) - half, 1)],
            dim=0,
        ),
    }


@dataclass(frozen=True)
class Projection:
    mean: torch.Tensor
    basis: torch.Tensor
    scale: torch.Tensor

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        result = ((value - self.mean) @ self.basis) / self.scale
        if not bool(torch.isfinite(result).all()):
            raise ValueError("projected quotient is non-finite")
        return result


def fit_projection(fit_rows: Sequence[PairRecord], dimension: int) -> Projection:
    values = torch.cat([source_relative_quotient(row) for row in fit_rows], dim=0)
    mean = values.mean(dim=0, keepdim=True)
    centered = values - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -dimension:].flip(1).contiguous()
    projected = centered @ basis
    scale = projected.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    if tuple(basis.shape) != (768, dimension):
        raise ValueError("projection basis geometry differs")
    return Projection(mean=mean.contiguous(), basis=basis, scale=scale.contiguous())


class DeterministicAE(nn.Module):
    def __init__(self, input_dim: int, family_count: int, config: Config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, input_dim),
        )
        self.transform_head = nn.Linear(config.latent_dim, len(TRANSFORMS))
        self.family_head = nn.Linear(config.latent_dim, family_count)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(value)
        return {
            "latent": latent,
            "reconstruction": self.decoder(latent),
            "transform_logits": self.transform_head(latent),
            "family_logits": self.family_head(latent),
        }


class ConditionalResidualVAE(nn.Module):
    """A stochastic residual attached to, and unable to update, a frozen q_det.

    This module is deliberately not a second stand-alone action autoencoder.
    It receives the frozen deterministic code and reconstruction, encodes only
    their remaining reconstruction residual, and adds a decoded correction to
    that deterministic reconstruction.  It has no family/transform heads.
    """

    def __init__(self, input_dim: int, config: Config):
        super().__init__()
        context_dim = config.latent_dim
        self.posterior = nn.Sequential(
            nn.Linear(input_dim + context_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 2 * config.latent_dim),
        )
        self.prior = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 2 * config.latent_dim),
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(2 * config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, input_dim),
        )
        # Start as the exact deterministic baseline.  Learning can only add a
        # residual correction; it cannot silently replace q_det at step zero.
        nn.init.zeros_(self.residual_decoder[-1].weight)
        nn.init.zeros_(self.residual_decoder[-1].bias)

    @staticmethod
    def _split(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = value.chunk(2, dim=1)
        return mean, logvar.clamp(min=-12.0, max=8.0)

    def forward(
        self,
        value: torch.Tensor,
        q_det: torch.Tensor,
        deterministic_reconstruction: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        # Detach at the module boundary as a second guard against accidentally
        # turning residual training into joint/stand-alone action-code training.
        q_det = q_det.detach()
        deterministic_reconstruction = deterministic_reconstruction.detach()
        residual_target = value - deterministic_reconstruction
        mean, logvar = self._split(
            self.posterior(torch.cat([residual_target, q_det], dim=1))
        )
        prior_mean, prior_logvar = self._split(self.prior(q_det))
        residual_sample = mean
        if sample:
            residual_sample = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        decoded_residual = self.residual_decoder(
            torch.cat([q_det, residual_sample], dim=1)
        )
        return {
            "q_det": q_det,
            "z_res_mean": mean,
            "z_res_sample": residual_sample,
            "mean": mean,
            "logvar": logvar,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "decoded_residual": decoded_residual,
            "reconstruction": deterministic_reconstruction + decoded_residual,
        }

    def sample_prior(
        self,
        q_det: torch.Tensor,
        deterministic_reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        q_det = q_det.detach()
        deterministic_reconstruction = deterministic_reconstruction.detach()
        mean, logvar = self._split(self.prior(q_det))
        z_res = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        decoded_residual = self.residual_decoder(
            torch.cat([q_det, z_res], dim=1)
        )
        return deterministic_reconstruction + decoded_residual


def diagonal_kl(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    variance_ratio = torch.exp(logvar - prior_logvar)
    squared = (mean - prior_mean).square() * torch.exp(-prior_logvar)
    return 0.5 * (prior_logvar - logvar + variance_ratio + squared - 1.0).sum(dim=1)


def build_tensor_split(
    rows: Sequence[PairRecord],
    projection: Projection,
    family_to_index: Mapping[str, int],
    seed: int,
) -> dict[str, torch.Tensor]:
    values = []
    families = []
    transforms = []
    iids = []
    source_means = []
    for row in rows:
        quotient = source_relative_quotient(row)
        for transform_index, name in enumerate(TRANSFORMS):
            values.append(projection.apply(counterfactuals(quotient, row.iid, seed)[name]).flatten())
            families.append(family_to_index[row.family])
            transforms.append(transform_index)
            iids.append(row.iid)
            source_means.append(row.source_sequence.mean(dim=0))
    if not values:
        return {
            "value": torch.empty((0, 32 * projection.basis.shape[1])),
            "family": torch.empty((0,), dtype=torch.long),
            "transform": torch.empty((0,), dtype=torch.long),
            "source_mean": torch.empty((0, 768)),
            "iids": [],
        }
    return {
        "value": torch.stack(values),
        "family": torch.tensor(families, dtype=torch.long),
        "transform": torch.tensor(transforms, dtype=torch.long),
        "source_mean": torch.stack(source_means),
        "iids": iids,
    }


def _deterministic_losses(
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    family: torch.Tensor,
    transform: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = F.mse_loss(output["reconstruction"], target)
    transform_loss = F.cross_entropy(output["transform_logits"], transform)
    family_loss = F.cross_entropy(output["family_logits"], family)
    total = (
        reconstruction
        + config.transform_weight * transform_loss
        + config.family_weight * family_loss
    )
    return total, {
        "reconstruction": float(reconstruction.detach()),
        "transform": float(transform_loss.detach()),
        "family": float(family_loss.detach()),
        "total": float(total.detach()),
    }


def _residual_vae_losses(
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = F.mse_loss(output["reconstruction"], target)
    kl = diagonal_kl(
        output["mean"],
        output["logvar"],
        output["prior_mean"],
        output["prior_logvar"],
    ).mean()
    total = reconstruction + config.beta_kl * kl
    return total, {
        "reconstruction": float(reconstruction.detach()),
        "kl": float(kl.detach()),
        "total": float(total.detach()),
    }


def train_deterministic_model(
    model: DeterministicAE,
    data: Mapping[str, Any],
    config: Config,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4
    )
    values = data["value"].to(device)
    families = data["family"].to(device)
    transforms = data["transform"].to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history = []
    for step in range(config.steps):
        indices = torch.randint(
            len(values), (min(config.batch_size, len(values)),), generator=generator
        ).to(device)
        value = values[indices]
        family = families[indices]
        transform = transforms[indices]
        output = model(value)
        total, metrics = _deterministic_losses(
            output, value, family, transform, config
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(config.steps // 10, 1) == 0:
            history.append({"step": step + 1, **metrics})
    return history


def freeze_deterministic_core(model: DeterministicAE) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def train_residual_vae(
    model: ConditionalResidualVAE,
    deterministic: DeterministicAE,
    data: Mapping[str, Any],
    config: Config,
    device: torch.device,
) -> list[dict[str, float]]:
    if any(parameter.requires_grad for parameter in deterministic.parameters()):
        raise ValueError("deterministic q_det core must be frozen before residual training")
    deterministic.to(device).eval()
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1.0e-4
    )
    values = data["value"].to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    history = []
    for step in range(config.steps):
        indices = torch.randint(
            len(values), (min(config.batch_size, len(values)),), generator=generator
        ).to(device)
        value = values[indices]
        with torch.no_grad():
            deterministic_output = deterministic(value)
        output = model(
            value,
            deterministic_output["latent"],
            deterministic_output["reconstruction"],
        )
        total, metrics = _residual_vae_losses(output, value, config)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(config.steps // 10, 1) == 0:
            history.append({"step": step + 1, **metrics})
    return history


def _cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=1, eps=EPS)


def _pairwise_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if len(left) < 3:
        return float("nan")
    left = F.normalize(left, dim=1, eps=EPS)
    right = F.normalize(right, dim=1, eps=EPS)
    indices = torch.triu_indices(len(left), len(left), offset=1, device=left.device)
    a = (left @ left.T)[indices[0], indices[1]]
    b = (right @ right.T)[indices[0], indices[1]]
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.square().sum().sqrt() * b.square().sum().sqrt()
    return float((a * b).sum() / denominator.clamp_min(EPS))


def _summarize_reconstruction(
    value: torch.Tensor,
    reconstruction: torch.Tensor,
    q_det: torch.Tensor,
    source_mean: torch.Tensor,
    family: torch.Tensor,
    transform: torch.Tensor,
    transform_logits: torch.Tensor,
    family_logits: torch.Tensor,
) -> dict[str, Any]:
    mse = (reconstruction - value).square().mean(dim=1)
    cosine = _cosine_rows(reconstruction, value)
    transform_prediction = transform_logits.argmax(dim=1)
    family_prediction = family_logits.argmax(dim=1)

    original = transform == 0
    by_transform = {}
    for index, name in enumerate(TRANSFORMS):
        mask = transform == index
        by_transform[name] = {
            "count": int(mask.sum()),
            "mse": float(mse[mask].mean()),
            "cosine": float(cosine[mask].mean()),
            "classification_accuracy": float(
                (transform_prediction[mask] == transform[mask]).float().mean()
            ),
        }
    return {
        "count": len(value),
        "base_clip_count": int(original.sum()),
        "reconstruction_mse": float(mse.mean()),
        "reconstruction_cosine": float(cosine.mean()),
        "transform_accuracy": float((transform_prediction == transform).float().mean()),
        "family_accuracy": float((family_prediction == family).float().mean()),
        "by_transform": by_transform,
        "q_det_effective_rank": int(
            (torch.linalg.svdvals(q_det - q_det.mean(dim=0)) > 1.0e-4).sum()
        ),
        "q_det_source_similarity_correlation_original": _pairwise_correlation(
            q_det[original], source_mean[original]
        ),
        "nuisance_leakage_status": "LEAKAGE_UNDETERMINED_UNIQUE_IID_ONLY",
    }


@torch.no_grad()
def evaluate_deterministic_model(
    model: DeterministicAE,
    data: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval().to(device)
    value = data["value"].to(device)
    family = data["family"].to(device)
    transform = data["transform"].to(device)
    source_mean = data["source_mean"].to(device)
    output = model(value)
    return _summarize_reconstruction(
        value,
        output["reconstruction"],
        output["latent"],
        source_mean,
        family,
        transform,
        output["transform_logits"],
        output["family_logits"],
    )


@torch.no_grad()
def evaluate_residual_vae(
    model: ConditionalResidualVAE,
    deterministic: DeterministicAE,
    data: Mapping[str, Any],
    config: Config,
    device: torch.device,
) -> dict[str, Any]:
    deterministic.eval().to(device)
    model.eval().to(device)
    value = data["value"].to(device)
    family = data["family"].to(device)
    transform = data["transform"].to(device)
    source_mean = data["source_mean"].to(device)
    deterministic_output = deterministic(value)
    output = model(
        value,
        deterministic_output["latent"],
        deterministic_output["reconstruction"],
        sample=False,
    )
    result = _summarize_reconstruction(
        value,
        output["reconstruction"],
        deterministic_output["latent"],
        source_mean,
        family,
        transform,
        deterministic_output["transform_logits"],
        deterministic_output["family_logits"],
    )
    result["action_label_metrics_source"] = "frozen_deterministic_q_det_heads"
    result["z_res_action_label_probe_trained"] = False
    original = transform == 0
    kl_per_row = diagonal_kl(
        output["mean"],
        output["logvar"],
        output["prior_mean"],
        output["prior_logvar"],
    )
    result["mean_kl"] = float(kl_per_row.mean())
    result["active_residual_units"] = int(
        (output["z_res_mean"][original].var(dim=0, unbiased=False) > 1.0e-3).sum()
    )
    result["z_res_mean_effective_rank"] = int(
        (
            torch.linalg.svdvals(
                output["z_res_mean"] - output["z_res_mean"].mean(dim=0)
            )
            > 1.0e-4
        ).sum()
    )
    result["z_res_source_similarity_correlation_original"] = _pairwise_correlation(
        output["z_res_mean"][original], source_mean[original]
    )
    original_q_det = deterministic_output["latent"][original]
    original_deterministic = deterministic_output["reconstruction"][original]
    original_target = value[original]
    prior_samples = []
    for _ in range(config.prior_samples):
        prior_samples.append(
            model.sample_prior(
                original_q_det, original_deterministic
            )
        )
    stack = torch.stack(prior_samples, dim=0)
    errors = (stack - original_target.unsqueeze(0)).square().mean(dim=2)
    result["prior_best_of_k_mse"] = float(errors.min(dim=0).values.mean())
    differences = []
    for left in range(config.prior_samples):
        for right in range(left + 1, config.prior_samples):
            differences.append((stack[left] - stack[right]).square().mean(dim=1))
    result["prior_pairwise_diversity_mse"] = float(torch.stack(differences).mean())
    result["prior_coverage_status"] = "UNDETERMINED_SINGLE_EXECUTION_PER_IID"
    return result


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(_cpu_state_dict(module).items()):
        header = canonical_bytes(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _write_json_create_only(path: Path, value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    os.chmod(path, 0o444)
    return hashlib.sha256(raw).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
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
    implementation_path = Path(__file__).resolve(strict=True)
    implementation_sha256 = file_sha256(implementation_path)
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        raise ValueError("output must be a fresh absolute child of an existing directory")
    pairs, feature_receipt = load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    splits, split_receipt = split_pairs(pairs, config.seed)
    families = sorted({row.family for row in pairs})
    family_to_index = {family: index for index, family in enumerate(families)}

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("canary requires exactly one visible GPU")
        if torch.cuda.get_device_name(0) != "AMD Instinct MI210":
            raise RuntimeError("canary GPU must be AMD Instinct MI210")

    projection = fit_projection(splits["fit"], config.pca_dim)
    tensors = {
        name: build_tensor_split(rows, projection, family_to_index, config.seed)
        for name, rows in splits.items()
    }
    output.mkdir(mode=0o700)
    input_dim = 32 * config.pca_dim
    deterministic = DeterministicAE(input_dim, len(families), config)
    ae_history = train_deterministic_model(
        deterministic, tensors["fit"], config, device
    )
    freeze_deterministic_core(deterministic)
    q_det_state_before_residual = _module_state_sha256(deterministic)
    residual_vae = ConditionalResidualVAE(input_dim, config)
    residual_history = train_residual_vae(
        residual_vae, deterministic, tensors["fit"], config, device
    )
    q_det_state_after_residual = _module_state_sha256(deterministic)
    if q_det_state_after_residual != q_det_state_before_residual:
        raise RuntimeError("frozen deterministic q_det changed during residual training")
    metrics = {
        split: {
            "deterministic_ae": evaluate_deterministic_model(
                deterministic, data, device
            ),
            "conditional_residual_vae": evaluate_residual_vae(
                residual_vae, deterministic, data, config, device
            ),
        }
        for split, data in tensors.items()
        if len(data["value"])
    }
    locked_ae = metrics["locked"]["deterministic_ae"]
    locked_residual = metrics["locked"]["conditional_residual_vae"]
    deterministic_mechanical_gate = bool(
        locked_ae["transform_accuracy"] >= 0.80
        and locked_ae["reconstruction_cosine"] >= 0.80
    )
    residual_mechanics_retention_gate = bool(
        deterministic_mechanical_gate
        and locked_residual["reconstruction_mse"]
        <= 1.02 * locked_ae["reconstruction_mse"]
        and locked_residual["active_residual_units"] > 0
        and locked_residual["mean_kl"] > 1.0e-4
    )
    vae_necessity_status = "UNDETERMINED_SINGLE_EXECUTION_PER_IID"

    if file_sha256(implementation_path) != implementation_sha256:
        raise RuntimeError("implementation changed during canary execution")

    checkpoint = {
        "schema_version": "semantic-action-cvae-exact644-canary-checkpoint-v2",
        "config": asdict(config),
        "implementation_sha256": implementation_sha256,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "families": families,
        "transforms": TRANSFORMS,
        "split_digest": split_receipt["split_digest"],
        "projection": {
            "mean": projection.mean,
            "basis": projection.basis,
            "scale": projection.scale,
        },
        "deterministic_ae": _cpu_state_dict(deterministic),
        "deterministic_q_det_state_sha256": q_det_state_after_residual,
        "conditional_residual_vae": _cpu_state_dict(residual_vae),
    }
    checkpoint_path = output / "checkpoint.pt"
    temporary = output / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    os.chmod(checkpoint_path, 0o444)

    receipt: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "STRUCTURED_TARGET_NOT_READY",
        "scope": "EXACT644_FEATURE_MECHANICS_CANARY",
        "formal_training_authorized": False,
        "video_model_updated": False,
        "paired_ground_truth_claimed": False,
        "scientific_action_representation_claimed": False,
        "input_population": {
            "unique_base_clips": 644,
            "action_anchor_feature_rows": 644,
            "source_nuisance_feature_rows": 644,
            "original_model_rows": 644,
            "counterfactual_rows_per_base": len(COUNTERFACTUAL_TRANSFORMS),
            "derived_counterfactual_model_rows": 644
            * len(COUNTERFACTUAL_TRANSFORMS),
            "total_model_rows": 644 * len(TRANSFORMS),
            "counterfactuals_are_independent_samples": False,
        },
        "feature_receipt": {
            "path": str((Path(args.feature_root) / "feature_extraction_receipt.json").resolve()),
            "sha256": args.expected_feature_receipt_sha256,
            "receipt_digest": feature_receipt["receipt_digest"],
        },
        "method": {
            "input": "center(anchor ordered DINO) - center(source ordered DINO)",
            "reconstruction_target": "source-relative ordered frozen feature quotient",
            "rgb_reconstructed": False,
            "wan_vae_latent_reconstructed": False,
            "anchor_identity_is_target": False,
            "semantic_moments_role": "unordered auxiliary only; not a sequence target",
            "deterministic_core_role": "frozen q_det after deterministic AE fitting",
            "deterministic_q_det_state_before_residual_sha256": q_det_state_before_residual,
            "deterministic_q_det_state_after_residual_sha256": q_det_state_after_residual,
            "deterministic_q_det_unchanged_during_residual_training": True,
            "conditional_vae_role": "stochastic correction on frozen q_det reconstruction only",
            "conditional_vae_is_standalone_action_autoencoder": False,
            "residual_family_or_transform_classification_loss": False,
            "family_or_transform_ground_truth_fed_to_residual_vae": False,
            "single_execution_per_iid": True,
        },
        "config": asdict(config),
        "config_sha256": object_sha256(asdict(config)),
        "device": {
            "requested": str(device),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        },
        "split": split_receipt,
        "projection": {
            "fit_only": True,
            "dimension": config.pca_dim,
            "basis_sha256": hashlib.sha256(
                projection.basis.numpy().tobytes(order="C")
            ).hexdigest(),
        },
        "training_history": {
            "deterministic_ae": ae_history,
            "conditional_residual_vae": residual_history,
        },
        "metrics": metrics,
        "gates": {
            "deterministic_mechanical_gate": deterministic_mechanical_gate,
            "conditional_residual_mechanics_retention_gate": residual_mechanics_retention_gate,
            "prior_coverage_verified": False,
            "vae_necessity_status": vae_necessity_status,
            "vae_necessary": None,
            "structured_action_target_available": False,
            "participant_role_phase_terminal_qualified": False,
            "renderer_training_authorized": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "implementation": {
            "path": str(implementation_path),
            "sha256": implementation_sha256,
        },
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json_create_only(receipt_path, receipt)
    result = {
        "receipt": str(receipt_path.resolve()),
        "sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "status": receipt["status"],
        "deterministic_mechanical_gate": deterministic_mechanical_gate,
        "conditional_residual_mechanics_retention_gate": residual_mechanics_retention_gate,
        "vae_necessity_status": vae_necessity_status,
        "vae_necessary": None,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--expected-feature-receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--beta-kl", type=float, default=0.02)
    parser.add_argument("--prior-samples", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
