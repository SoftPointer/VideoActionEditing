#!/usr/bin/env python3
"""PCA-initialized low-capacity anchor-only sequence AE/beta-VAE v3.

This is a new burned-development exact644 experiment.  It reuses the exact
v2 five-fold assignment and the literal full-DINO target
``temporal_center(anchor ordered DINO)``.  Source features, edited-video
targets, RGB/Wan reconstruction, action labels and residual *targets* are not
used.

Frame-PCA rank L is fitted on each fold's model-fit rows.  Its mean and basis
are frozen buffers inside both models, so the step-zero posterior-mean
reconstruction matches the linear PCA reconstruction within a pinned 3e-5
absolute tolerance.  Small zero-initialized nonlinear
encoder/decoder corrections can improve that mapping while checkpoint
selection can retain step zero.  Both arms execute the same full optimization
budget; an inner-fit validation split selects a checkpoint without terminating
training early.  PCA is an initialization and hard comparator, never the
reconstruction target.
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

from methods.bernini_action_editing import semantic_anchor_action_sequence_vae_v2 as v2


authority = v2.authority
RAW_TARGET_DEFINITION = v2.RAW_TARGET_DEFINITION
MODEL_COORDINATE_DEFINITION = v2.MODEL_COORDINATE_DEFINITION
OUTER_FOLDS = v2.OUTER_FOLDS
ARMS = v2.ARMS
EPS = v2.EPS
POSTERIOR_MC_SAMPLE_COUNT = 8
VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR = 1.0e-8
STEP0_MAX_ABS_TOLERANCE = 3.0e-5
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

TRAIN_SCHEMA = "anchor-action-pca-initialized-oof-train-bundle-v3"
EVAL_SCHEMA = "anchor-action-pca-initialized-oof-eval-bundle-v3"
PREPARE_SCHEMA = "anchor-action-pca-initialized-oof-prepare-receipt-v3"
ARM_CHECKPOINT_SCHEMA = "anchor-action-pca-initialized-oof-arm-checkpoint-v3"
ARM_RECEIPT_SCHEMA = "anchor-action-pca-initialized-oof-arm-receipt-v3"
COMPARE_SCHEMA = "anchor-action-pca-initialized-oof-compare-receipt-v3"
AGGREGATE_SCHEMA = "anchor-action-pca-initialized-oof-aggregate-receipt-v3"
REFIT_BUNDLE_SCHEMA = "anchor-action-pca-initialized-full644-refit-bundle-v3"
REFIT_PREPARE_SCHEMA = "anchor-action-pca-initialized-full644-refit-prepare-v3"
REFIT_CHECKPOINT_SCHEMA = "anchor-action-pca-initialized-full644-checkpoint-v3"
REFIT_RECEIPT_SCHEMA = "anchor-action-pca-initialized-full644-receipt-v3"

DEVELOPMENT_FIELDS = {
    "prior_locked_partition_rows_burned": 96,
    "exact644_role": "BURNED_DEVELOPMENT_ONLY",
    "fresh_confirmation_requires_new_external_group_disjoint_data": True,
    "confirmation_evaluations_allowed_by_this_runtime": 0,
    "scientific_confirmation_claimed": False,
}

COMMON_AGGREGATE_BOOLEAN_GATE_KEYS = frozenset({
    "clip_ratio_ucb_lt_1_vs_zero",
    "family_ratio_ucb_lt_1_vs_zero",
    "clip_ratio_ucb_lt_1_vs_frame_pca",
    "family_ratio_ucb_lt_1_vs_frame_pca",
    "all_five_folds_point_ratio_lt_1_vs_both",
    "all_five_energy_bins_point_ratio_le_1_vs_zero",
    "all_five_energy_bins_point_ratio_le_1_vs_frame_pca",
    "temporal_delta_clip_ucb_lt_1_vs_frame_pca",
    "temporal_delta_family_ucb_lt_1_vs_frame_pca",
    "cosine_clip_lcb_gt_0_vs_frame_pca",
    "cosine_family_lcb_gt_0_vs_frame_pca",
})
COMMON_AGGREGATE_NESTED_GATE_KEYS = frozenset({
    "fold_point_ratios", "energy_strata",
})
VAE_AGGREGATE_BOOLEAN_GATE_KEYS = frozenset({
    "all_folds_full_beta_eligible",
    "all_folds_finite_positive_residual_kl",
    "all_folds_have_residual_active_units",
    "posterior_vs_zero_residual_clip_ucb_lt_1",
    "posterior_vs_zero_residual_family_ucb_lt_1",
    "posterior_vs_shuffled_residual_clip_ucb_lt_1",
    "posterior_vs_shuffled_residual_family_ucb_lt_1",
    "normalized_sample_variance_clip_lcb_gt_floor",
    "normalized_sample_variance_family_lcb_gt_floor",
    "posterior_mc8_clip_ucb_lt_1_vs_zero",
    "posterior_mc8_family_ucb_lt_1_vs_zero",
    "posterior_mc8_clip_ucb_lt_1_vs_frame_pca",
    "posterior_mc8_family_ucb_lt_1_vs_frame_pca",
    "vae_vs_ae_clip_retention_ucb_le_1p02",
    "vae_vs_ae_family_retention_ucb_le_1p02",
})


def _expected_aggregate_gate_keys(arm: str) -> set[str]:
    if arm not in ARMS:
        raise ValueError("v3 aggregate arm differs")
    keys = set(COMMON_AGGREGATE_BOOLEAN_GATE_KEYS)
    keys.update(COMMON_AGGREGATE_NESTED_GATE_KEYS)
    keys.add("aggregate_hard_gate")
    if arm == "direct_beta_vae":
        keys.update(VAE_AGGREGATE_BOOLEAN_GATE_KEYS)
    return keys


@dataclass(frozen=True)
class Config:
    seed: int = 20260819
    latent_dim: int = 32
    correction_hidden_dim: int = 48
    max_steps: int = 1500
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    beta_kl: float = 5.0e-4
    kl_zero_steps: int = 100
    kl_warmup_steps: int = 500
    full_beta_plateau_steps: int = 200
    correction_penalty: float = 1.0e-4
    eval_interval: int = 25
    selection_relative_delta: float = 1.0e-4
    initial_logvar: float = -6.0

    def validate(self) -> None:
        if self != Config():
            raise ValueError("v3 hyperparameters are exact-preregistered and immutable")


def _object_sha(value: Any) -> str:
    return v2._object_sha(value)


def _tensor_sha(value: torch.Tensor) -> str:
    return v2._tensor_sha(value)


def _pca_state_sha(value: Mapping[str, torch.Tensor]) -> str:
    return v2._pca_state_sha(value)


def _sha(value: Any, name: str) -> str:
    return v2._sha(value, name)


def _file_sha(path: Path) -> str:
    return v2._file_sha(path)


def _binding() -> dict[str, str]:
    implementation = Path(__file__).resolve(strict=True)
    common = Path(v2.__file__).resolve(strict=True)
    feature_authority = Path(authority.__file__).resolve(strict=True)
    return {
        "implementation_path": str(implementation),
        "implementation_sha256": _file_sha(implementation),
        "v2_common_path": str(common),
        "v2_common_sha256": _file_sha(common),
        "feature_authority_path": str(feature_authority),
        "feature_authority_sha256": _file_sha(feature_authority),
    }


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("v3 implementation/dependencies changed during command")


def anchor_action_target(pair: authority.PairRecord) -> torch.Tensor:
    return v2.anchor_action_target(pair)


def _make_config(args: argparse.Namespace) -> Config:
    config = Config(
        seed=args.seed,
        latent_dim=args.latent_dim,
        correction_hidden_dim=args.correction_hidden_dim,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta_kl=args.beta_kl,
        kl_zero_steps=args.kl_zero_steps,
        kl_warmup_steps=args.kl_warmup_steps,
        full_beta_plateau_steps=args.full_beta_plateau_steps,
        correction_penalty=args.correction_penalty,
        eval_interval=args.eval_interval,
        selection_relative_delta=args.selection_relative_delta,
        initial_logvar=args.initial_logvar,
    )
    config.validate()
    return config


TRAIN_KEYS = {
    "schema_version", "config", "config_sha256", "fold", "implementation",
    "feature_receipt_sha256", "feature_receipt_digest", "exact644_iid_digest",
    "exact644_raw_target_sha256", "exact644_population_authority",
    "raw_target_definition", "model_coordinate_definition", "global_rms",
    "global_rms_sha256", "global_rms_fit_only", "pca_is_model_target",
    "model_fit", "early_stop_validation", "baselines", "baseline_sha256",
    "pca_initialization", "pca_initialization_sha256", "initialization_contract",
}
EVAL_KEYS = {
    "schema_version", "config", "config_sha256", "fold", "implementation",
    "feature_receipt_sha256", "feature_receipt_digest", "exact644_iid_digest",
    "exact644_raw_target_sha256", "exact644_population_authority",
    "raw_target_definition", "model_coordinate_definition", "global_rms",
    "global_rms_sha256", "global_rms_fit_only", "pca_is_model_target",
    "exploratory_oof", "exploratory_family_by_iid", "source_features_present",
}
REFIT_KEYS = {
    "schema_version", "config", "config_sha256", "implementation",
    "aggregate_receipt_sha256", "aggregate_receipt_digest",
    "feature_receipt_sha256", "feature_receipt_digest", "exact644_iid_digest",
    "exact644_raw_target_sha256", "exact644_population_authority",
    "raw_target_definition", "model_coordinate_definition", "global_rms",
    "global_rms_sha256", "global_rms_fit_only", "pca_is_model_target",
    "full644_originals", "full644_model_coordinate_sha256", "baselines",
    "baseline_sha256", "pca_initialization", "pca_initialization_sha256",
    "initialization_contract", "preregistered_steps_by_arm", "authorized_arms",
    "model_fit_unique_originals", "held_rows", "derived_rows",
    "development_energy_definition", "development_energy_bin_edges_raw",
}


def _save_torch(path: Path, value: Any) -> str:
    return v2._save_torch_create_only(path, value)


def _write_json(path: Path, value: Any) -> str:
    return v2._write_json_create_only(path, value)


def _fresh_output(value: str) -> Path:
    return v2._fresh_output(value)


def _load_receipt(path: Path, expected_sha: str, schema: str) -> dict[str, Any]:
    return v2._load_receipt(path, expected_sha, schema)


def _load_torch(path: Path, expected_sha: str, expected_size: int | None = None) -> Any:
    return v2._load_torch(path, expected_sha, expected_size)


def _fit_pca_initialization(
    fit: torch.Tensor, frame_pca: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Fit-only whitening coordinates whose mean decode is the PCA projector."""

    mean = frame_pca["mean"]
    basis = frame_pca["basis"]
    coefficients = (fit - mean) @ basis
    scale = coefficients.square().mean(dim=(0, 1)).sqrt().clamp_min(1.0e-4)
    state = {
        "mean": mean.detach().clone().contiguous(),
        "basis": basis.detach().clone().contiguous(),
        "latent_scale": scale.detach().clone().contiguous(),
    }
    _validate_pca_initialization(state, basis.shape[1])
    return state


def _validate_pca_initialization(value: Any, latent_dim: int) -> None:
    if type(value) is not dict or set(value) != {"mean", "basis", "latent_scale"}:
        raise ValueError("v3 PCA initialization keys differ")
    v2._validate_pca_state(
        {"mean": value["mean"], "basis": value["basis"]},
        (1, 768), (768, latent_dim), "v3 PCA initialization",
    )
    scale = value["latent_scale"]
    if (
        type(scale) is not torch.Tensor or scale.dtype != torch.float32
        or tuple(scale.shape) != (latent_dim,)
        or not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any())
    ):
        raise ValueError("v3 PCA latent scale differs")


def _pca_initialization_sha(value: Mapping[str, torch.Tensor]) -> str:
    _validate_pca_initialization(value, value["basis"].shape[1])
    return _object_sha({key: _tensor_sha(value[key]) for key in sorted(value)})


def _pca_subspace_equivalent(
    stored: Mapping[str, torch.Tensor], recomputed: Mapping[str, torch.Tensor]
) -> bool:
    """Compare PCA projectors, not sign/rotation-ambiguous basis bytes."""

    left_mean, right_mean = stored["mean"], recomputed["mean"]
    left_basis, right_basis = stored["basis"], recomputed["basis"]
    if (
        tuple(left_mean.shape) != tuple(right_mean.shape)
        or tuple(left_basis.shape) != tuple(right_basis.shape)
        or not torch.allclose(
            left_mean, right_mean, atol=2.0e-6, rtol=2.0e-6
        )
    ):
        return False
    singular_values = torch.linalg.svdvals(left_basis.T @ right_basis)
    return bool(torch.allclose(
        singular_values, torch.ones_like(singular_values),
        atol=5.0e-4, rtol=5.0e-4,
    ))


def prepare_fold(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    config = _make_config(args)
    pairs, feature_receipt = authority.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    population = v2._exact644_population_authority(pairs)
    exact_iid_digest = _object_sha([row.iid for row in pairs])
    raw_target_sha = _tensor_sha(torch.stack([
        anchor_action_target(row) for row in pairs
    ]))
    groups, split = v2._split_fold(pairs, args.fold_index, config.seed)
    if (
        split["outer_assignment_digest"] != V2_OUTER_ASSIGNMENT_DIGEST
        or split["iid_digest"] != V2_FOLD_IID_DIGESTS[args.fold_index]
    ):
        raise ValueError("v3 split is not bit-identical to frozen v2 OOF")
    rms = v2._global_rms(groups["model_fit"])
    fit = v2._tensor_rows(groups["model_fit"], rms)
    validation = v2._tensor_rows(groups["early_stop_validation"], rms)
    exploratory = v2._tensor_rows(groups["exploratory_oof"], rms)
    frame_pca = v2._fit_frame_pca(fit["value"], config.latent_dim)
    clip_pca = v2._fit_clip_pca(fit["value"], config.latent_dim)
    pca_initialization = _fit_pca_initialization(fit["value"], frame_pca)
    baseline_sha = {
        "frame_pca_rank_l": _pca_state_sha(frame_pca),
        "clip_pca_rank_l": _pca_state_sha(clip_pca),
    }
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
        "exact644_population_authority": population,
        "raw_target_definition": RAW_TARGET_DEFINITION,
        "model_coordinate_definition": MODEL_COORDINATE_DEFINITION,
        "global_rms": rms,
        "global_rms_sha256": _tensor_sha(rms),
        "global_rms_fit_only": True,
        "pca_is_model_target": False,
    }
    train_bundle = {
        "schema_version": TRAIN_SCHEMA,
        **common,
        "model_fit": fit,
        "early_stop_validation": validation,
        "baselines": {
            "frame_pca_rank_l": frame_pca,
            "clip_pca_rank_l": clip_pca,
        },
        "baseline_sha256": baseline_sha,
        "pca_initialization": pca_initialization,
        "pca_initialization_sha256": _pca_initialization_sha(pca_initialization),
        "initialization_contract": {
            "frozen_frame_pca_rank": config.latent_dim,
            "zero_initialized_nonlinear_heads": True,
            "step0_posterior_mean_reconstruction_matches_frame_pca_within_abs_3e_5": True,
            "raw_identity_skip": False,
            "frozen_pca_input_output_path": True,
            "decoder_nonlinear_output_orthogonal_to_pca": True,
            "learned_output_increment_entirely_pca_orthogonal": True,
            "residual_latent_is_internal_not_supervision_target": True,
            "posterior_observes_full_target_residual": False,
        },
    }
    eval_bundle = {
        "schema_version": EVAL_SCHEMA,
        **common,
        "exploratory_oof": exploratory,
        "exploratory_family_by_iid": {
            row.iid: row.family for row in groups["exploratory_oof"]
        },
        "source_features_present": False,
    }
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    train_path = output / "train_bundle.pt"
    eval_path = output / "exploratory_oof_bundle.pt"
    train_sha = _save_torch(train_path, train_bundle)
    eval_sha = _save_torch(eval_path, eval_bundle)
    receipt: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "status": "V3_OOF_FOLD_PREPARED_BURNED_DEVELOPMENT",
        **DEVELOPMENT_FIELDS,
        "unique_original_base_clips": 644,
        "derived_rows": 0,
        "exact644_population_authority": population,
        "exact644_iid_digest": exact_iid_digest,
        "exact644_raw_target_sha256": raw_target_sha,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "target": {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
            "direct_full768_reconstruction": True,
        },
        "family_labels_used_only_for_split_and_evaluation_statistics": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "rgb_or_wan_reconstruction_performed": False,
        "source_identity_preservation_tested": False,
        "config": config_value,
        "config_sha256": common["config_sha256"],
        "fold": split,
        "global_rms": float(rms),
        "global_rms_sha256": common["global_rms_sha256"],
        "baseline_sha256": baseline_sha,
        "pca_initialization_sha256": train_bundle["pca_initialization_sha256"],
        "initialization_contract": train_bundle["initialization_contract"],
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
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "prepare_receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "train_bundle_sha256": train_sha,
        "exploratory_oof_bundle_sha256": eval_sha,
        "fold_counts": split["counts"],
    }


class PCAInitializedSequenceCore(nn.Module):
    """Frozen PCA trunk plus low-capacity zero-initialized corrections."""

    def __init__(
        self, config: Config, pca_initialization: Mapping[str, torch.Tensor]
    ) -> None:
        super().__init__()
        _validate_pca_initialization(pca_initialization, config.latent_dim)
        self.config = config
        self.register_buffer(
            "pca_mean", pca_initialization["mean"].detach().clone()
        )
        self.register_buffer(
            "pca_basis", pca_initialization["basis"].detach().clone()
        )
        self.register_buffer(
            "latent_scale", pca_initialization["latent_scale"].detach().clone()
        )
        hidden = config.correction_hidden_dim
        latent = config.latent_dim
        self.temporal_depthwise = nn.Conv1d(
            latent, latent, kernel_size=5, padding=2, groups=latent,
            padding_mode="replicate",
        )
        self.temporal_pointwise = nn.Conv1d(latent, hidden, kernel_size=1)
        self.decoder_output = nn.Conv1d(hidden, 768, kernel_size=1)
        self.mean_output = nn.Conv1d(hidden, latent, kernel_size=1)
        nn.init.zeros_(self.decoder_output.weight)
        nn.init.zeros_(self.decoder_output.bias)
        nn.init.zeros_(self.mean_output.weight)
        nn.init.zeros_(self.mean_output.bias)

    def pca_encode(self, value: torch.Tensor) -> torch.Tensor:
        return ((value - self.pca_mean) @ self.pca_basis) / self.latent_scale

    def sequence_features(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal_depthwise(latent.transpose(1, 2))
        return F.silu(self.temporal_pointwise(hidden))

    def decode(
        self, base: torch.Tensor, residual_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The PCA trunk stays exact.  The learned latent parameterizes only
        # information missing from that trunk; the supervised target remains
        # the complete 768-D anchor feature sequence.
        linear = (base * self.latent_scale) @ self.pca_basis.T + self.pca_mean
        features = self.sequence_features(base + residual_latent)
        raw_correction = self.decoder_output(features).transpose(1, 2)
        # The complete learned output increment is confined to the fit-PCA
        # orthogonal complement, so it cannot erase the analytic baseline.
        correction = raw_correction - (
            (raw_correction @ self.pca_basis) @ self.pca_basis.T
        )
        reconstruction = linear + correction
        reconstruction = reconstruction - reconstruction.mean(dim=1, keepdim=True)
        correction = correction - correction.mean(dim=1, keepdim=True)
        return reconstruction, correction


class PCAInitializedDeterministicAE(PCAInitializedSequenceCore):
    def forward(self, value: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        del sample
        base = self.pca_encode(value)
        base_features = self.sequence_features(base)
        residual_latent = self.mean_output(base_features).transpose(1, 2)
        reconstruction, nonlinear = self.decode(base, residual_latent)
        return {
            "base_latent": base,
            "latent": residual_latent,
            "latent_delta": residual_latent,
            "nonlinear_output": nonlinear,
            "reconstruction": reconstruction,
        }


class PCAInitializedDirectBetaVAE(PCAInitializedSequenceCore):
    def __init__(
        self, config: Config, pca_initialization: Mapping[str, torch.Tensor]
    ) -> None:
        super().__init__(config, pca_initialization)
        self.logvar_output = nn.Conv1d(
            config.correction_hidden_dim, config.latent_dim, kernel_size=1
        )
        nn.init.zeros_(self.logvar_output.weight)
        nn.init.zeros_(self.logvar_output.bias)

    def forward(self, value: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        base = self.pca_encode(value)
        base_features = self.sequence_features(base)
        mean = self.mean_output(base_features).transpose(1, 2)
        residual_features = self.sequence_features(base + mean)
        logvar_delta = self.logvar_output(residual_features).transpose(1, 2)
        logvar = (self.config.initial_logvar + logvar_delta).clamp(-10.0, 4.0)
        residual_latent = mean
        if sample:
            residual_latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        reconstruction, nonlinear = self.decode(base, residual_latent)
        return {
            "base_latent": base,
            "latent": residual_latent,
            "latent_delta": mean,
            "mean": mean,
            "logvar": logvar,
            "nonlinear_output": nonlinear,
            "reconstruction": reconstruction,
        }


def _make_model(
    arm: str, config: Config, pca_initialization: Mapping[str, torch.Tensor]
) -> nn.Module:
    if arm == "deterministic_ae":
        return PCAInitializedDeterministicAE(config, pca_initialization)
    if arm == "direct_beta_vae":
        return PCAInitializedDirectBetaVAE(config, pca_initialization)
    raise ValueError("arm differs")


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def kl_element_mean(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).mean()


def kl_weight(step: int, config: Config) -> float:
    if step <= config.kl_zero_steps:
        return 0.0
    progress = (step - config.kl_zero_steps) / (
        config.kl_warmup_steps - config.kl_zero_steps
    )
    return config.beta_kl * min(1.0, progress)


def _loss(
    arm: str,
    output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    step: int,
    config: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = F.mse_loss(output["reconstruction"], target)
    correction = (
        output["latent_delta"].square().mean()
        + output["nonlinear_output"].square().mean()
    )
    kl = torch.zeros((), device=target.device)
    beta = 0.0
    if arm == "direct_beta_vae":
        kl = kl_element_mean(output["mean"], output["logvar"])
        beta = kl_weight(step, config)
    total = reconstruction + config.correction_penalty * correction + beta * kl
    return total, {
        "reconstruction": float(reconstruction.detach()),
        "correction_energy": float(correction.detach()),
        "kl_element_mean": float(kl.detach()),
        "effective_beta": beta,
        "total": float(total.detach()),
    }


@torch.no_grad()
def _validation_mse(model: nn.Module, value: torch.Tensor, device: torch.device) -> float:
    model.eval()
    target = value.to(device)
    reconstruction = model(target, sample=False)["reconstruction"]
    return float(F.mse_loss(reconstruction, target))


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


@torch.no_grad()
def _assert_step0_equals_pca(
    model: nn.Module,
    value: torch.Tensor,
    frame_pca: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float | bool]:
    model.eval().to(device)
    target = value.to(device)
    actual = model(target, sample=False)["reconstruction"].cpu()
    expected = v2._reconstruct_frame_pca(value, frame_pca)
    max_abs = float((actual - expected).abs().max())
    actual_mse = float(F.mse_loss(actual, value))
    expected_mse = float(F.mse_loss(expected, value))
    if max_abs >= STEP0_MAX_ABS_TOLERANCE or not math.isclose(
        actual_mse, expected_mse, rel_tol=1.0e-6, abs_tol=1.0e-8
    ):
        raise RuntimeError("step-zero model is not the frame-PCA comparator")
    return {
        "verified": True,
        "max_abs_difference": max_abs,
        "model_mse": actual_mse,
        "frame_pca_mse": expected_mse,
    }


def train_with_preregistered_selection(
    arm: str,
    model: nn.Module,
    fit: torch.Tensor,
    validation: torch.Tensor,
    frame_pca: Mapping[str, torch.Tensor],
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int, int, str, dict[str, Any]]:
    model.to(device)
    fit = fit.to(device)
    initialization = _assert_step0_equals_pca(
        model, validation, frame_pca, device
    )
    step0_mse = float(initialization["model_mse"])
    validation_zero_mse = float(validation.square().mean())
    min_delta = config.selection_relative_delta * validation_zero_mse
    eligibility_step = config.kl_warmup_steps + config.full_beta_plateau_steps
    if arm == "deterministic_ae":
        best_mse = step0_mse
        best_step = 0
        best_state: dict[str, torch.Tensor] | None = _cpu_state(model)
    else:
        # A beta-VAE receipt may never select the KL-free PCA initialization.
        best_mse = float("inf")
        best_step = -1
        best_state = None
    initial_kl = (
        0.5 * (
            math.exp(config.initial_logvar) - config.initial_logvar - 1.0
        )
        if arm == "direct_beta_vae" else 0.0
    )
    history: list[dict[str, Any]] = [{
        "step": 0,
        "reconstruction": step0_mse,
        "correction_energy": 0.0,
        "kl_element_mean": initial_kl,
        "effective_beta": 0.0,
        "total": step0_mse,
        "validation_mse": step0_mse,
        "checkpoint_eligible": arm == "deterministic_ae",
        "full_beta_plateau_steps_completed": 0,
    }]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 200)
    schedule_digest = hashlib.sha256()
    executed_steps = 0
    for step in range(1, config.max_steps + 1):
        executed_steps = step
        model.train()
        indices_cpu = torch.randint(
            len(fit), (min(config.batch_size, len(fit)),), generator=generator
        )
        schedule_digest.update(indices_cpu.numpy().tobytes(order="C"))
        target = fit[indices_cpu.to(device)]
        output = model(target, sample=True)
        total, metrics = _loss(arm, output, target, step, config)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % config.eval_interval == 0 or step == config.max_steps:
            validation_mse = _validation_mse(model, validation, device)
            eligible = arm == "deterministic_ae" or step >= eligibility_step
            history.append({
                "step": step,
                **metrics,
                "validation_mse": validation_mse,
                "checkpoint_eligible": eligible,
                "full_beta_plateau_steps_completed": max(
                    0, step - config.kl_warmup_steps
                ),
            })
            if eligible:
                if best_state is None or validation_mse < best_mse - min_delta:
                    best_mse = validation_mse
                    best_step = step
                    best_state = _cpu_state(model)
    if executed_steps != config.max_steps:
        raise RuntimeError("v3 arms must execute the same full optimization budget")
    if best_state is None or best_step < 0:
        raise RuntimeError("no full-beta eligible VAE checkpoint was produced")
    model.load_state_dict(best_state, strict=True)
    initialization.update({
        "validation_zero_mse": validation_zero_mse,
        "selection_absolute_min_delta": min_delta,
        "checkpoint_eligibility_step": eligibility_step,
        "selected_checkpoint_beats_step0": best_mse < step0_mse - min_delta,
        "selected_checkpoint_selection_eligible": True,
        "model_selection_scanned_full_max_steps": True,
        "early_termination_used": False,
        "selected_checkpoint_full_beta_eligible": (
            best_step >= eligibility_step if arm == "direct_beta_vae" else None
        ),
    })
    return (
        history, best_step, executed_steps, schedule_digest.hexdigest(),
        initialization,
    )


def _validate_common(
    bundle: Mapping[str, Any],
    prepare: Mapping[str, Any],
    expected_keys: set[str],
    expected_implementation: Mapping[str, str],
) -> Config:
    if type(bundle) is not dict or set(bundle) != expected_keys:
        raise ValueError("v3 bundle exact-key allowlist differs")
    config_value = bundle["config"]
    if type(config_value) is not dict or set(config_value) != set(asdict(Config())):
        raise ValueError("v3 config keys differ")
    config = Config(**config_value)
    config.validate()
    if _object_sha(config_value) != _sha(bundle["config_sha256"], "config SHA"):
        raise ValueError("v3 config self digest differs")
    if config_value != prepare["config"] or bundle["config_sha256"] != prepare["config_sha256"]:
        raise ValueError("v3 bundle/prepare config differs")
    v2._validate_fold(bundle["fold"])
    if bundle["fold"] != prepare["fold"]:
        raise ValueError("v3 bundle/prepare fold differs")
    fold_index = bundle["fold"]["outer_fold"]
    if (
        bundle["fold"]["outer_assignment_digest"] != V2_OUTER_ASSIGNMENT_DIGEST
        or bundle["fold"]["iid_digest"] != V2_FOLD_IID_DIGESTS[fold_index]
    ):
        raise ValueError("v3 bundle does not use frozen v2 split")
    for key in (
        "feature_receipt_sha256", "feature_receipt_digest",
        "exact644_iid_digest", "exact644_raw_target_sha256",
    ):
        _sha(bundle[key], key)
        if bundle[key] != prepare[key]:
            raise ValueError(f"v3 bundle/prepare {key} differs")
    if (
        bundle["exact644_population_authority"]
        != prepare["exact644_population_authority"]
        or bundle["exact644_population_authority"] != {
            "unique_original_base_clips": 644,
            "family_count": 28,
            "strict_true": 359,
            "strict_false": 285,
            "derived_rows": 0,
        }
    ):
        raise ValueError("v3 population authority differs")
    if (
        bundle["implementation"] != prepare["implementation"]
        or bundle["implementation"] != expected_implementation
    ):
        raise ValueError("v3 implementation authority differs")
    if (
        bundle["raw_target_definition"] != RAW_TARGET_DEFINITION
        or bundle["model_coordinate_definition"] != MODEL_COORDINATE_DEFINITION
        or bundle["global_rms_fit_only"] is not True
        or bundle["pca_is_model_target"] is not False
    ):
        raise ValueError("v3 target contract differs")
    rms = bundle["global_rms"]
    if (
        type(rms) is not torch.Tensor or rms.dtype != torch.float32
        or tuple(rms.shape) != (1,) or not bool(torch.isfinite(rms).all())
        or float(rms) <= EPS or _tensor_sha(rms) != bundle["global_rms_sha256"]
        or bundle["global_rms_sha256"] != prepare["global_rms_sha256"]
        or not math.isclose(float(rms), prepare["global_rms"], rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("v3 RMS authority differs")
    return config


def _load_train_bundle(
    args: argparse.Namespace,
    prepare: Mapping[str, Any],
    expected_implementation: Mapping[str, str],
) -> tuple[dict[str, Any], Config]:
    binding = prepare.get("train_bundle")
    if type(binding) is not dict or set(binding) != {
        "path", "sha256", "size_bytes", "contains_exploratory_oof"
    } or binding["contains_exploratory_oof"] is not False:
        raise ValueError("v3 train bundle binding differs")
    path = Path(args.train_bundle).resolve(strict=True)
    expected_sha = _sha(args.expected_train_bundle_sha256, "train bundle SHA")
    if Path(binding["path"]).resolve(strict=True) != path or binding["sha256"] != expected_sha:
        raise ValueError("v3 train path/SHA authority differs")
    bundle = _load_torch(path, expected_sha, binding["size_bytes"])
    if bundle.get("schema_version") != TRAIN_SCHEMA:
        raise ValueError("v3 train schema differs")
    config = _validate_common(bundle, prepare, TRAIN_KEYS, expected_implementation)
    counts = bundle["fold"]["counts"]
    v2._validate_sequence_rows(bundle["model_fit"], counts["model_fit"], "model_fit")
    v2._validate_sequence_rows(
        bundle["early_stop_validation"], counts["early_stop_validation"],
        "early_stop_validation",
    )
    fit_iids = bundle["model_fit"]["iids"]
    validation_iids = bundle["early_stop_validation"]["iids"]
    if set(fit_iids) & set(validation_iids):
        raise ValueError("v3 fit/validation IIDs overlap")
    fold = bundle["fold"]
    if (
        _object_sha(fit_iids) != fold["model_fit_iid_digest"]
        or _object_sha(validation_iids) != fold["early_stop_validation_iid_digest"]
        or _object_sha({
            "model_fit": fit_iids,
            "early_stop_validation": validation_iids,
        }) != fold["train_iid_digest"]
    ):
        raise ValueError("v3 train IID closure differs")
    if not bool(torch.isclose(
        bundle["model_fit"]["value"].square().mean().sqrt(),
        torch.tensor(1.0), atol=3.0e-6, rtol=3.0e-6,
    )):
        raise ValueError("v3 standardized fit RMS differs")
    baselines = bundle["baselines"]
    if type(baselines) is not dict or set(baselines) != {
        "frame_pca_rank_l", "clip_pca_rank_l"
    }:
        raise ValueError("v3 baseline keys differ")
    v2._validate_pca_state(
        baselines["frame_pca_rank_l"], (1, 768),
        (768, config.latent_dim), "v3 frame PCA",
    )
    v2._validate_pca_state(
        baselines["clip_pca_rank_l"], (1, 32 * 768),
        (32 * 768, config.latent_dim), "v3 clip PCA",
    )
    digests = {name: _pca_state_sha(state) for name, state in baselines.items()}
    recomputed_baselines = {
        "frame_pca_rank_l": v2._fit_frame_pca(
            bundle["model_fit"]["value"], config.latent_dim
        ),
        "clip_pca_rank_l": v2._fit_clip_pca(
            bundle["model_fit"]["value"], config.latent_dim
        ),
    }
    if (
        digests != bundle["baseline_sha256"]
        or digests != prepare["baseline_sha256"]
        or any(
            not _pca_subspace_equivalent(
                baselines[name], recomputed_baselines[name]
            )
            for name in baselines
        )
    ):
        raise ValueError("v3 baseline tensor digest differs")
    pca_initialization = bundle["pca_initialization"]
    _validate_pca_initialization(pca_initialization, config.latent_dim)
    initialization_sha = _pca_initialization_sha(pca_initialization)
    recomputed = _fit_pca_initialization(
        bundle["model_fit"]["value"], baselines["frame_pca_rank_l"]
    )
    if (
        initialization_sha != bundle["pca_initialization_sha256"]
        or initialization_sha != prepare["pca_initialization_sha256"]
        or any(
            not torch.allclose(
                pca_initialization[key], recomputed[key],
                atol=2.0e-6, rtol=2.0e-6,
            )
            for key in pca_initialization
        )
    ):
        raise ValueError("v3 fit-only PCA initialization authority differs")
    if bundle["initialization_contract"] != {
        "frozen_frame_pca_rank": config.latent_dim,
        "zero_initialized_nonlinear_heads": True,
        "step0_posterior_mean_reconstruction_matches_frame_pca_within_abs_3e_5": True,
        "raw_identity_skip": False,
        "frozen_pca_input_output_path": True,
        "decoder_nonlinear_output_orthogonal_to_pca": True,
        "learned_output_increment_entirely_pca_orthogonal": True,
        "residual_latent_is_internal_not_supervision_target": True,
        "posterior_observes_full_target_residual": False,
    } or bundle["initialization_contract"] != prepare["initialization_contract"]:
        raise ValueError("v3 initialization contract differs")
    return bundle, config


def _load_eval_bundle(
    args: argparse.Namespace,
    prepare: Mapping[str, Any],
    expected_implementation: Mapping[str, str],
) -> tuple[dict[str, Any], Config]:
    binding = prepare.get("exploratory_oof_bundle")
    if type(binding) is not dict or set(binding) != {
        "path", "sha256", "size_bytes", "contains_model_fit_or_early_stop_values"
    } or binding["contains_model_fit_or_early_stop_values"] is not False:
        raise ValueError("v3 OOF bundle binding differs")
    path = Path(args.exploratory_oof_bundle).resolve(strict=True)
    expected_sha = _sha(
        args.expected_exploratory_oof_bundle_sha256, "OOF bundle SHA"
    )
    if Path(binding["path"]).resolve(strict=True) != path or binding["sha256"] != expected_sha:
        raise ValueError("v3 OOF path/SHA authority differs")
    bundle = _load_torch(path, expected_sha, binding["size_bytes"])
    if bundle.get("schema_version") != EVAL_SCHEMA:
        raise ValueError("v3 OOF schema differs")
    config = _validate_common(bundle, prepare, EVAL_KEYS, expected_implementation)
    if bundle["source_features_present"] is not False:
        raise ValueError("v3 OOF bundle contains source features")
    count = bundle["fold"]["counts"]["exploratory_oof"]
    v2._validate_sequence_rows(bundle["exploratory_oof"], count, "exploratory_oof")
    family_by_iid = bundle["exploratory_family_by_iid"]
    if (
        type(family_by_iid) is not dict
        or set(family_by_iid) != set(bundle["exploratory_oof"]["iids"])
        or any(type(family) is not str or not family for family in family_by_iid.values())
    ):
        raise ValueError("v3 exploratory family bootstrap authority differs")
    if _object_sha(bundle["exploratory_oof"]["iids"]) != bundle["fold"]["exploratory_oof_iid_digest"]:
        raise ValueError("v3 OOF IID digest differs")
    return bundle, config


def train_fold_arm(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    prepare = _load_receipt(
        Path(args.prepare_receipt), args.expected_prepare_receipt_sha256,
        PREPARE_SCHEMA,
    )
    bundle, config = _load_train_bundle(args, prepare, run_binding)
    if args.fold_index != bundle["fold"]["outer_fold"]:
        raise ValueError("v3 train CLI fold differs")
    if args.arm not in ARMS:
        raise ValueError("v3 arm differs")
    pca_initialization = bundle["pca_initialization"]
    with torch.random.fork_rng(devices=[]):
        ae_count = _parameter_count(_make_model(
            "deterministic_ae", config, pca_initialization
        ))
        vae_count = _parameter_count(_make_model(
            "direct_beta_vae", config, pca_initialization
        ))
    expected_vae_extra = config.correction_hidden_dim * config.latent_dim + config.latent_dim
    if (
        ae_count > 50000 or vae_count > 50000
        or vae_count - ae_count != expected_vae_extra
    ):
        raise RuntimeError("v3 arm capacity contract differs")
    device = v2._device(args.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = _make_model(args.arm, config, pca_initialization)
    fit_initialization = _assert_step0_equals_pca(
        model, bundle["model_fit"]["value"],
        bundle["baselines"]["frame_pca_rank_l"], device,
    )
    history, best_step, executed_steps, schedule_digest, validation_initialization = (
        train_with_preregistered_selection(
            args.arm,
            model,
            bundle["model_fit"]["value"],
            bundle["early_stop_validation"]["value"],
            bundle["baselines"]["frame_pca_rank_l"],
            config,
            device,
        )
    )
    best_validation_mse = _validation_mse(
        model, bundle["early_stop_validation"]["value"], device
    )
    eligibility_step = config.kl_warmup_steps + config.full_beta_plateau_steps
    if args.arm == "direct_beta_vae" and best_step < eligibility_step:
        raise RuntimeError("v3 VAE selected a non-full-beta checkpoint")
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
        "pca_initialization_sha256": bundle["pca_initialization_sha256"],
        "best_step": best_step,
        "executed_steps": executed_steps,
        "executed_minibatch_schedule_sha256": schedule_digest,
        "model_state": _cpu_state(model),
        "implementation": run_binding,
    }
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_sha = _save_torch(checkpoint_path, checkpoint)
    receipt: dict[str, Any] = {
        "schema_version": ARM_RECEIPT_SCHEMA,
        "status": "V3_OOF_ARM_TRAINED_NOT_YET_OOF_EVALUATED",
        **DEVELOPMENT_FIELDS,
        "arm": args.arm,
        "primary_arm": args.arm == "deterministic_ae",
        "fold": bundle["fold"],
        "target": {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
            "direct_full768_reconstruction": True,
        },
        "architecture": {
            "fit_only_whitened_pca_initialization": True,
            "decoder_nonlinear_correction_projected_to_pca_orthogonal_complement": True,
            "learned_output_increment_entirely_pca_orthogonal": True,
            "raw_identity_skip": False,
            "frozen_pca_input_output_path": True,
            "posterior_observes_full_target_residual": False,
            "step0_posterior_mean_reconstruction_matches_frame_pca_within_abs_3e_5": True,
        },
        "config": asdict(config),
        "config_sha256": bundle["config_sha256"],
        "parameter_count": _parameter_count(model),
        "shared_reconstruction_backbone_matched": True,
        "exact_parameter_count_matched": False,
        "vae_posterior_variance_head_extra_parameters": expected_vae_extra,
        "v2_parameter_count_reference": 522048,
        "parameter_reduction_ratio_vs_v2": _parameter_count(model) / 522048,
        "fit_step0_pca_equivalence": fit_initialization,
        "validation_step0_pca_equivalence": validation_initialization,
        "best_step": best_step,
        "executed_steps": executed_steps,
        "best_validation_mse": best_validation_mse,
        "step0_fallback_allowed": args.arm == "deterministic_ae",
        "full_beta_checkpoint_required": args.arm == "direct_beta_vae",
        "full_beta_checkpoint_eligible_from_step": eligibility_step,
        "selected_checkpoint_selection_eligible": True,
        "selected_checkpoint_full_beta_eligible": (
            best_step >= eligibility_step
            if args.arm == "direct_beta_vae" else None
        ),
        "full_beta_exposure_steps_before_selected_checkpoint": max(
            0, best_step - config.kl_warmup_steps
        ),
        "kl_reduction": "mean_over_batch_time_latent_elements",
        "training_history": history,
        "executed_minibatch_schedule_sha256": schedule_digest,
        "exploratory_oof_tensor_rows_read_by_training": False,
        "outer_split_and_global_energy_summary_metadata_read": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "train_bundle_sha256": args.expected_train_bundle_sha256,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "pca_initialization_sha256": bundle["pca_initialization_sha256"],
        "checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "vae_necessary": None,
        "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION_PER_IID",
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
        "checkpoint_sha256": checkpoint_sha,
        "best_step": best_step,
    }


def _load_arm_model(
    arm: str,
    receipt_path: str,
    expected_receipt_sha: str,
    config: Config,
    fold: Mapping[str, Any],
    prepare_sha: str,
    train_sha: str,
    pca_initialization: Mapping[str, torch.Tensor],
    pca_initialization_sha: str,
    expected_implementation: Mapping[str, str],
) -> tuple[nn.Module, dict[str, Any]]:
    receipt = _load_receipt(
        Path(receipt_path), expected_receipt_sha, ARM_RECEIPT_SCHEMA
    )
    schedule_sha = _sha(
        receipt.get("executed_minibatch_schedule_sha256"),
        "v3 executed minibatch schedule SHA",
    )
    if (
        receipt.get("arm") != arm
        or receipt.get("config") != asdict(config)
        or receipt.get("config_sha256") != _object_sha(asdict(config))
        or receipt.get("fold") != fold
        or receipt.get("prepare_receipt_sha256") != prepare_sha
        or receipt.get("train_bundle_sha256") != train_sha
        or receipt.get("pca_initialization_sha256") != pca_initialization_sha
        or receipt.get("implementation") != expected_implementation
        or receipt.get("executed_steps") != config.max_steps
        or receipt.get("selected_checkpoint_selection_eligible") is not True
        or receipt.get("executed_minibatch_schedule_sha256") != schedule_sha
    ):
        raise ValueError("v3 arm receipt authority differs")
    if arm == "direct_beta_vae" and (
        receipt.get("full_beta_checkpoint_required") is not True
        or receipt.get("selected_checkpoint_full_beta_eligible") is not True
        or receipt.get("full_beta_exposure_steps_before_selected_checkpoint", -1)
        < config.full_beta_plateau_steps
    ):
        raise ValueError("v3 VAE receipt lacks full-beta exposure")
    checkpoint_binding = receipt.get("checkpoint")
    if type(checkpoint_binding) is not dict or set(checkpoint_binding) != {
        "path", "sha256", "size_bytes"
    }:
        raise ValueError("v3 checkpoint binding differs")
    checkpoint = _load_torch(
        Path(checkpoint_binding["path"]), checkpoint_binding["sha256"],
        checkpoint_binding["size_bytes"],
    )
    expected_checkpoint_keys = {
        "schema_version", "arm", "config", "config_sha256", "fold",
        "train_bundle_sha256", "prepare_receipt_sha256",
        "pca_initialization_sha256", "best_step", "executed_steps",
        "executed_minibatch_schedule_sha256",
        "model_state", "implementation",
    }
    if type(checkpoint) is not dict or set(checkpoint) != expected_checkpoint_keys:
        raise ValueError("v3 checkpoint exact-key allowlist differs")
    if (
        checkpoint["schema_version"] != ARM_CHECKPOINT_SCHEMA
        or checkpoint["arm"] != arm
        or checkpoint["config"] != asdict(config)
        or checkpoint["config_sha256"] != receipt["config_sha256"]
        or checkpoint["fold"] != fold
        or checkpoint["train_bundle_sha256"] != train_sha
        or checkpoint["prepare_receipt_sha256"] != prepare_sha
        or checkpoint["pca_initialization_sha256"] != pca_initialization_sha
        or checkpoint["best_step"] != receipt["best_step"]
        or checkpoint["executed_steps"] != receipt["executed_steps"]
        or checkpoint["executed_steps"] != config.max_steps
        or checkpoint["executed_minibatch_schedule_sha256"]
        != receipt["executed_minibatch_schedule_sha256"]
        or checkpoint["implementation"] != expected_implementation
    ):
        raise ValueError("v3 checkpoint authority join differs")
    model = _make_model(arm, config, pca_initialization)
    if receipt["parameter_count"] != _parameter_count(model):
        raise ValueError("v3 checkpoint parameter count differs")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, receipt


def _family_cluster_ratio(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    family_by_iid: Mapping[str, str],
    seed: int,
    metric_key: str = "raw_mse",
    draws: int = 10000,
    expected_family_count: int | None = None,
) -> dict[str, Any]:
    candidate_map = {row["iid"]: float(row[metric_key]) for row in candidate}
    baseline_map = {row["iid"]: float(row[metric_key]) for row in baseline}
    if (
        not candidate_map or set(candidate_map) != set(baseline_map)
        or set(candidate_map) != set(family_by_iid)
        or len(candidate_map) != len(candidate)
        or len(baseline_map) != len(baseline)
    ):
        raise ValueError("v3 family bootstrap IID closure differs")
    families = sorted(set(family_by_iid.values()))
    if expected_family_count is not None and len(families) != expected_family_count:
        raise ValueError("v3 family bootstrap family closure differs")
    candidate_sums = []
    baseline_sums = []
    counts = []
    for family in families:
        iids = [iid for iid in candidate_map if family_by_iid[iid] == family]
        candidate_sums.append(sum(candidate_map[iid] for iid in iids))
        baseline_sums.append(sum(baseline_map[iid] for iid in iids))
        counts.append(len(iids))
    left = torch.tensor(candidate_sums, dtype=torch.float64)
    right = torch.tensor(baseline_sums, dtype=torch.float64)
    cluster_counts = torch.tensor(counts, dtype=torch.float64)
    if (
        not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
        or bool((left < 0.0).any()) or bool((right <= 0.0).any())
    ):
        raise ValueError("v3 family bootstrap metric values differ")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(families), (draws, len(families)), generator=generator)
    sampled_left = left[indices].sum(dim=1)
    sampled_right = right[indices].sum(dim=1)
    sampled_counts = cluster_counts[indices].sum(dim=1)
    ratios = sampled_left / sampled_right.clamp_min(EPS)
    deltas = (sampled_left - sampled_right) / sampled_counts.clamp_min(1.0)
    return {
        "family_count": len(families),
        "iid_count": len(candidate_map),
        "bootstrap_unit": "family_cluster",
        "bootstrap_seed": seed,
        "draws": draws,
        "metric": metric_key,
        "mean_ratio": float(left.sum() / right.sum().clamp_min(EPS)),
        "ratio_95pct_ci": [
            float(torch.quantile(ratios, 0.025)),
            float(torch.quantile(ratios, 0.975)),
        ],
        "mean_delta": float((left.sum() - right.sum()) / cluster_counts.sum()),
        "delta_95pct_ci": [
            float(torch.quantile(deltas, 0.025)),
            float(torch.quantile(deltas, 0.975)),
        ],
    }


def _paired_improvement(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    metric_key: str,
    seed: int,
    draws: int = 10000,
) -> dict[str, Any]:
    left_map = {row["iid"]: float(row[metric_key]) for row in candidate}
    right_map = {row["iid"]: float(row[metric_key]) for row in baseline}
    if (
        not left_map or set(left_map) != set(right_map)
        or len(left_map) != len(candidate) or len(right_map) != len(baseline)
    ):
        raise ValueError("v3 paired improvement IID closure differs")
    iids = sorted(left_map)
    delta = torch.tensor(
        [left_map[iid] - right_map[iid] for iid in iids], dtype=torch.float64
    )
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("v3 paired improvement is non-finite")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(delta), (draws, len(delta)), generator=generator)
    sampled = delta[indices].mean(dim=1)
    return {
        "iid_count": len(iids),
        "bootstrap_unit": "iid",
        "bootstrap_seed": seed,
        "draws": draws,
        "metric": metric_key,
        "mean_improvement": float(delta.mean()),
        "improvement_95pct_ci": [
            float(torch.quantile(sampled, 0.025)),
            float(torch.quantile(sampled, 0.975)),
        ],
    }


def _family_cluster_improvement(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    family_by_iid: Mapping[str, str],
    metric_key: str,
    seed: int,
    draws: int = 10000,
    expected_family_count: int | None = None,
) -> dict[str, Any]:
    left_map = {row["iid"]: float(row[metric_key]) for row in candidate}
    right_map = {row["iid"]: float(row[metric_key]) for row in baseline}
    if set(left_map) != set(right_map) or set(left_map) != set(family_by_iid):
        raise ValueError("v3 family improvement IID closure differs")
    families = sorted(set(family_by_iid.values()))
    if expected_family_count is not None and len(families) != expected_family_count:
        raise ValueError("v3 family improvement family closure differs")
    delta_sums, counts = [], []
    for family in families:
        iids = [iid for iid in left_map if family_by_iid[iid] == family]
        delta_sums.append(sum(left_map[iid] - right_map[iid] for iid in iids))
        counts.append(len(iids))
    delta = torch.tensor(delta_sums, dtype=torch.float64)
    cluster_counts = torch.tensor(counts, dtype=torch.float64)
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("v3 family improvement is non-finite")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(families), (draws, len(families)), generator=generator)
    sampled = delta[indices].sum(dim=1) / cluster_counts[indices].sum(dim=1)
    return {
        "family_count": len(families),
        "iid_count": len(left_map),
        "bootstrap_unit": "family_cluster",
        "bootstrap_seed": seed,
        "draws": draws,
        "metric": metric_key,
        "mean_improvement": float(delta.sum() / cluster_counts.sum()),
        "improvement_95pct_ci": [
            float(torch.quantile(sampled, 0.025)),
            float(torch.quantile(sampled, 0.975)),
        ],
    }


@torch.no_grad()
def _predict(model: nn.Module, value: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval().to(device)
    return model(value.to(device), sample=False)["reconstruction"].cpu()


@torch.no_grad()
def _comparison_reconstruction(
    arm: str,
    model: nn.Module,
    best_step: int,
    value: torch.Tensor,
    frame_pca: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select the OOF metric tensor without a step-zero roundoff win."""
    actual = _predict(model, value, device)
    if arm != "deterministic_ae" or best_step != 0:
        return actual, {
            "analytic_frame_pca_alias_used": False,
            "checkpoint_output_max_abs_vs_analytic_frame_pca": None,
        }
    analytic = v2._reconstruct_frame_pca(value, frame_pca)
    max_abs = float((actual - analytic).abs().max())
    if not math.isfinite(max_abs) or max_abs >= STEP0_MAX_ABS_TOLERANCE:
        raise ValueError("v3 step-zero AE checkpoint differs from frame PCA")
    # A step-zero checkpoint contains no learned improvement.  Reuse the
    # analytic baseline tensor itself so device/reduction roundoff cannot
    # manufacture a point or bootstrap win over PCA.
    return analytic, {
        "analytic_frame_pca_alias_used": True,
        "checkpoint_output_max_abs_vs_analytic_frame_pca": max_abs,
    }


@torch.no_grad()
def _vae_mechanics(
    model: PCAInitializedDirectBetaVAE,
    target: torch.Tensor,
    iids: Sequence[str],
    global_rms: torch.Tensor,
    energy_edges_raw: Sequence[float],
    frame_pca: Mapping[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    if len(target) != len(iids) or len(set(iids)) != len(iids):
        raise ValueError("v3 VAE mechanics IID closure differs")
    model.eval().to(device)
    value = target.to(device)
    output = model(value, sample=False)
    base = output["base_latent"]
    mean = output["mean"]
    logvar = output["logvar"]
    kl_per_dim = 0.5 * (
        mean.square() + logvar.exp() - logvar - 1.0
    ).mean(dim=(0, 1))
    mean_variance = mean.reshape(-1, mean.shape[-1]).var(dim=0, unbiased=False)
    active = (kl_per_dim > 1.0e-4) & (mean_variance > 1.0e-3)
    if len(value) < 2:
        raise ValueError("v3 VAE mechanics needs at least two clips")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shuffle_offset = int(torch.randint(
        1, len(value), (1,), generator=generator
    ))
    permutation = (
        torch.arange(len(value), device=device) + shuffle_offset
    ) % len(value)
    shuffled_reconstruction = model.decode(base, mean[permutation])[0]
    zero_residual_reconstruction = model.decode(base, torch.zeros_like(mean))[0]
    analytic_pca_reconstruction = v2._reconstruct_frame_pca(
        target, frame_pca
    )
    posterior_reconstruction_cpu = output["reconstruction"].cpu()
    posterior_metric_rows = v2._metric_rows(
        target, posterior_reconstruction_cpu, iids, global_rms,
        energy_edges_raw,
    )["per_iid"]
    shuffled_metric_rows = v2._metric_rows(
        target, shuffled_reconstruction.cpu(), iids, global_rms,
        energy_edges_raw,
    )["per_iid"]
    zero_residual_metric_rows = v2._metric_rows(
        target, zero_residual_reconstruction.cpu(), iids, global_rms,
        energy_edges_raw,
    )["per_iid"]
    analytic_pca_metric_rows = v2._metric_rows(
        target, analytic_pca_reconstruction, iids, global_rms,
        energy_edges_raw,
    )["per_iid"]
    sample_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    posterior_samples = []
    residual_std = torch.exp(0.5 * logvar)
    for _ in range(POSTERIOR_MC_SAMPLE_COUNT):
        noise = torch.randn(
            mean.shape, generator=sample_generator, dtype=mean.dtype,
            device="cpu",
        ).to(device)
        posterior_samples.append(
            model.decode(base, mean + noise * residual_std)[0]
        )
    sample_stack = torch.stack(posterior_samples)
    sample_output_variance_by_iid = sample_stack.var(
        dim=0, unbiased=False
    ).mean(dim=(1, 2))
    sample_metrics = [
        v2._metric_rows(
            target, sample.cpu(), iids, global_rms, energy_edges_raw
        )["per_iid"]
        for sample in sample_stack
    ]
    mc_per_iid = []
    intervention_per_iid = []
    rms_squared = float(global_rms.square())
    if not math.isfinite(rms_squared) or rms_squared <= 0.0:
        raise ValueError("v3 VAE mechanics global RMS is invalid")
    for row_index, iid in enumerate(iids):
        mc_per_iid.append({
            "iid": iid,
            "raw_mse": sum(
                float(rows[row_index]["raw_mse"]) for rows in sample_metrics
            ) / len(sample_metrics),
            "raw_temporal_delta_mse": sum(
                float(rows[row_index]["raw_temporal_delta_mse"])
                for rows in sample_metrics
            ) / len(sample_metrics),
            "cosine": sum(
                float(rows[row_index]["cosine"]) for rows in sample_metrics
            ) / len(sample_metrics),
        })
        posterior_row = posterior_metric_rows[row_index]
        zero_row = zero_residual_metric_rows[row_index]
        shuffled_row = shuffled_metric_rows[row_index]
        analytic_pca_row = analytic_pca_metric_rows[row_index]
        zero_value = float(zero_row["raw_mse"])
        analytic_pca_residual = float(analytic_pca_row["raw_mse"])
        if not math.isfinite(analytic_pca_residual) or analytic_pca_residual <= 0.0:
            raise ValueError("v3 analytic PCA residual scale is invalid")
        sample_variance_model = float(sample_output_variance_by_iid[row_index])
        sample_variance_raw = sample_variance_model * rms_squared
        intervention_per_iid.append({
            "iid": iid,
            "posterior_raw_mse": float(posterior_row["raw_mse"]),
            "zero_residual_raw_mse": zero_value,
            "analytic_pca_residual_raw_mse": analytic_pca_residual,
            "shuffled_residual_raw_mse": float(shuffled_row["raw_mse"]),
            "normalized_sample_output_variance": float(
                sample_variance_raw / analytic_pca_residual
            ),
            "posterior_sample_output_variance_model_coordinate": sample_variance_model,
            "posterior_sample_output_variance_raw": sample_variance_raw,
        })
    kl_per_dim_serialized = [float(item) for item in kl_per_dim.cpu()]
    mean_variance_serialized = [float(item) for item in mean_variance.cpu()]
    sample_variance_serialized = [
        float(item["posterior_sample_output_variance_model_coordinate"])
        for item in intervention_per_iid
    ]
    posterior_mse_serialized = [
        float(item["mse"]) for item in posterior_metric_rows
    ]
    zero_mse_serialized = [
        float(item["mse"]) for item in zero_residual_metric_rows
    ]
    shuffled_mse_serialized = [
        float(item["mse"]) for item in shuffled_metric_rows
    ]
    posterior_raw_sum = sum(
        float(item["posterior_raw_mse"]) for item in intervention_per_iid
    )
    zero_raw_sum = sum(
        float(item["zero_residual_raw_mse"]) for item in intervention_per_iid
    )
    shuffled_raw_sum = sum(
        float(item["shuffled_residual_raw_mse"]) for item in intervention_per_iid
    )
    sample_output_variance = sum(sample_variance_serialized) / len(
        sample_variance_serialized
    )
    mechanics = {
        "residual_only_kl_element_mean": sum(kl_per_dim_serialized) / len(
            kl_per_dim_serialized
        ),
        "residual_only_kl_per_latent_dim": kl_per_dim_serialized,
        "residual_mean_variance_per_dim": mean_variance_serialized,
        "residual_active_unit_count": int(active.sum()),
        "residual_active_unit_thresholds": {
            "kl_per_dim_gt": 1.0e-4,
            "posterior_mean_variance_gt": 1.0e-3,
        },
        "residual_mean_energy": float(mean.square().mean()),
        "residual_shuffle_seed": seed,
        "residual_shuffle_offset": shuffle_offset,
        "residual_shuffle_fixed_point_count": 0,
        "residual_shuffle_is_cross_clip_not_prior_coverage": True,
        "posterior_mean_mse": sum(posterior_mse_serialized) / len(
            posterior_mse_serialized
        ),
        "zero_learned_residual_mse": sum(zero_mse_serialized) / len(
            zero_mse_serialized
        ),
        "cross_clip_shuffled_residual_mse": sum(shuffled_mse_serialized) / len(
            shuffled_mse_serialized
        ),
        "residual_mean_improves_over_zero_residual": bool(
            posterior_raw_sum < zero_raw_sum
        ),
        "residual_shuffle_increases_mse": bool(
            shuffled_raw_sum > posterior_raw_sum
        ),
        "residual_intervention_per_iid": intervention_per_iid,
        "posterior_mc_expected_per_iid": mc_per_iid,
        "posterior_sample_count": POSTERIOR_MC_SAMPLE_COUNT,
        "posterior_sample_output_variance": sample_output_variance,
        "stochastic_residual_changes_output": bool(
            sample_output_variance > 1.0e-10
        ),
        "full_pca_latent_excluded_from_kl_and_active_unit_metrics": True,
        "prior_coverage_tested": False,
    }
    return mechanics, posterior_reconstruction_cpu


def _energy_ratio_gates(
    metrics: Mapping[str, Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    values = {}
    for index in range(5):
        key = str(index)
        arm_row = metrics[arm]["energy_strata"][key]
        zero_row = metrics["zero_hard_baseline"]["energy_strata"][key]
        pca_row = metrics["frame_pca_rank_l_hard_baseline"]["energy_strata"][key]
        if arm_row["count"] <= 0 or zero_row["raw_mse"] <= 0 or pca_row["raw_mse"] <= 0:
            raise ValueError("v3 energy stratum is empty or degenerate")
        values[key] = {
            "count": arm_row["count"],
            "ratio_vs_zero": arm_row["raw_mse"] / zero_row["raw_mse"],
            "ratio_vs_frame_pca": arm_row["raw_mse"] / pca_row["raw_mse"],
        }
    return {
        "by_bin": values,
        "all_bins_point_ratio_vs_zero_le_1": all(
            row["ratio_vs_zero"] <= 1.0 for row in values.values()
        ),
        "all_bins_point_ratio_vs_frame_pca_le_1": all(
            row["ratio_vs_frame_pca"] <= 1.0 for row in values.values()
        ),
    }


def compare_fold(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    prepare = _load_receipt(
        Path(args.prepare_receipt), args.expected_prepare_receipt_sha256,
        PREPARE_SCHEMA,
    )
    train_bundle, config = _load_train_bundle(args, prepare, run_binding)
    eval_bundle, eval_config = _load_eval_bundle(args, prepare, run_binding)
    if config != eval_config or train_bundle["fold"] != eval_bundle["fold"]:
        raise ValueError("v3 train/OOF common authority differs")
    if args.fold_index != train_bundle["fold"]["outer_fold"]:
        raise ValueError("v3 compare CLI fold differs")
    exploratory = eval_bundle["exploratory_oof"]
    split_iids = {
        "model_fit": train_bundle["model_fit"]["iids"],
        "early_stop_validation": train_bundle["early_stop_validation"]["iids"],
        "exploratory_oof": exploratory["iids"],
    }
    closure = [iid for values in split_iids.values() for iid in values]
    if (
        len(closure) != 644 or len(set(closure)) != 644
        or _object_sha(split_iids) != train_bundle["fold"]["iid_digest"]
    ):
        raise ValueError("v3 train/OOF exact644 closure differs")
    pca_initialization = train_bundle["pca_initialization"]
    pca_initialization_sha = train_bundle["pca_initialization_sha256"]
    ae, ae_receipt = _load_arm_model(
        "deterministic_ae", args.ae_receipt,
        args.expected_ae_receipt_sha256, config, train_bundle["fold"],
        args.expected_prepare_receipt_sha256,
        args.expected_train_bundle_sha256, pca_initialization,
        pca_initialization_sha, run_binding,
    )
    vae, vae_receipt = _load_arm_model(
        "direct_beta_vae", args.vae_receipt,
        args.expected_vae_receipt_sha256, config, train_bundle["fold"],
        args.expected_prepare_receipt_sha256,
        args.expected_train_bundle_sha256, pca_initialization,
        pca_initialization_sha, run_binding,
    )
    if (
        ae_receipt["executed_steps"] != config.max_steps
        or vae_receipt["executed_steps"] != config.max_steps
        or ae_receipt["executed_minibatch_schedule_sha256"]
        != vae_receipt["executed_minibatch_schedule_sha256"]
    ):
        raise ValueError("v3 OOF arms do not share the full training budget")
    device = v2._device(args.device)
    target = exploratory["value"]
    rms = eval_bundle["global_rms"]
    edges = eval_bundle["fold"]["fixed_energy_bin_edges_exact644"]
    vae_mechanics, vae_reconstruction = _vae_mechanics(
        vae, target, exploratory["iids"], rms, edges,
        train_bundle["baselines"]["frame_pca_rank_l"], device,
        config.seed + 8000 + args.fold_index,
    )
    ae_reconstruction, ae_step0_metric_policy = _comparison_reconstruction(
        "deterministic_ae", ae, ae_receipt["best_step"], target,
        train_bundle["baselines"]["frame_pca_rank_l"], device,
    )
    frame_pca_reconstruction = (
        ae_reconstruction
        if ae_step0_metric_policy["analytic_frame_pca_alias_used"]
        else v2._reconstruct_frame_pca(
            target, train_bundle["baselines"]["frame_pca_rank_l"]
        )
    )
    reconstructions = {
        "deterministic_ae": ae_reconstruction,
        "direct_beta_vae": vae_reconstruction,
        "frame_pca_rank_l_hard_baseline": frame_pca_reconstruction,
        "clip_pca_rank_l_diagnostic": v2._reconstruct_clip_pca(
            target, train_bundle["baselines"]["clip_pca_rank_l"]
        ),
        "zero_hard_baseline": torch.zeros_like(target),
    }
    metrics = {
        name: v2._metric_rows(target, reconstruction, exploratory["iids"], rms, edges)
        for name, reconstruction in reconstructions.items()
    }
    family_by_iid = eval_bundle["exploratory_family_by_iid"]
    comparisons: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        comparisons[arm] = {}
        for baseline_index, (label, baseline) in enumerate((
            ("zero", "zero_hard_baseline"),
            ("frame_pca_rank_l", "frame_pca_rank_l_hard_baseline"),
        )):
            comparisons[arm][f"vs_{label}"] = {
                "clip_bootstrap": v2._paired_ratio(
                    metrics[arm]["per_iid"], metrics[baseline]["per_iid"],
                    config.seed + 3000 + 100 * args.fold_index
                    + 10 * arm_index + baseline_index,
                ),
                "family_cluster_bootstrap": _family_cluster_ratio(
                    metrics[arm]["per_iid"], metrics[baseline]["per_iid"],
                    family_by_iid,
                    config.seed + 4000 + 100 * args.fold_index
                    + 10 * arm_index + baseline_index,
                ),
            }
        comparisons[arm]["temporal_delta_vs_frame_pca"] = {
            "clip_bootstrap": v2._paired_ratio(
                metrics[arm]["per_iid"],
                metrics["frame_pca_rank_l_hard_baseline"]["per_iid"],
                config.seed + 5000 + 10 * args.fold_index + arm_index,
                metric_key="raw_temporal_delta_mse",
            ),
            "family_cluster_bootstrap": _family_cluster_ratio(
                metrics[arm]["per_iid"],
                metrics["frame_pca_rank_l_hard_baseline"]["per_iid"],
                family_by_iid,
                config.seed + 6000 + 10 * args.fold_index + arm_index,
                metric_key="raw_temporal_delta_mse",
            ),
        }
        comparisons[arm]["cosine_improvement_vs_frame_pca"] = {
            "clip_bootstrap": _paired_improvement(
                metrics[arm]["per_iid"],
                metrics["frame_pca_rank_l_hard_baseline"]["per_iid"],
                "cosine", config.seed + 6200 + 10 * args.fold_index + arm_index,
            ),
            "family_cluster_bootstrap": _family_cluster_improvement(
                metrics[arm]["per_iid"],
                metrics["frame_pca_rank_l_hard_baseline"]["per_iid"],
                family_by_iid, "cosine",
                config.seed + 6400 + 10 * args.fold_index + arm_index,
            ),
        }
        energy = _energy_ratio_gates(metrics, arm)
        gates[arm] = {
            "point_ratio_lt_1_vs_zero": (
                comparisons[arm]["vs_zero"]["clip_bootstrap"]["mean_ratio"] < 1.0
            ),
            "point_ratio_lt_1_vs_frame_pca": (
                comparisons[arm]["vs_frame_pca_rank_l"]["clip_bootstrap"]["mean_ratio"] < 1.0
            ),
            "temporal_delta_point_ratio_lt_1_vs_frame_pca": (
                comparisons[arm]["temporal_delta_vs_frame_pca"]["clip_bootstrap"]["mean_ratio"] < 1.0
            ),
            "cosine_point_improvement_gt_0_vs_frame_pca": (
                comparisons[arm]["cosine_improvement_vs_frame_pca"]["clip_bootstrap"]["mean_improvement"] > 0.0
            ),
            "energy_strata": energy,
            "fold_bootstrap_is_diagnostic_not_qualification": True,
        }
        gates[arm]["fold_point_direction_gate"] = bool(
            all(
                value is True for key, value in gates[arm].items()
                if key not in {
                    "energy_strata",
                    "fold_bootstrap_is_diagnostic_not_qualification",
                }
            )
        )
    vae_vs_ae = {
        "clip_bootstrap": v2._paired_ratio(
            metrics["direct_beta_vae"]["per_iid"],
            metrics["deterministic_ae"]["per_iid"],
            config.seed + 7000 + args.fold_index,
        ),
        "family_cluster_bootstrap": _family_cluster_ratio(
            metrics["direct_beta_vae"]["per_iid"],
            metrics["deterministic_ae"]["per_iid"], family_by_iid,
            config.seed + 7100 + args.fold_index,
        ),
    }
    gates["direct_beta_vae"]["retention_ucb_le_1p02_vs_ae_diagnostic"] = bool(
        vae_vs_ae["clip_bootstrap"]["ratio_95pct_ci"][1] <= 1.02
        and vae_vs_ae["family_cluster_bootstrap"]["ratio_95pct_ci"][1] <= 1.02
    )
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    receipt: dict[str, Any] = {
        "schema_version": COMPARE_SCHEMA,
        "status": "V3_EXPLORATORY_OOF_FOLD_COMPARISON_COMPLETE",
        **DEVELOPMENT_FIELDS,
        "fold_index": args.fold_index,
        "target": {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
            "direct_full768_reconstruction": True,
        },
        "sample_accounting": {
            "unique_exact644_development": 644,
            "exploratory_oof_original_rows_this_fold": len(target),
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
        "family_by_iid": family_by_iid,
        "family_labels_used_only_for_split_and_evaluation_statistics": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "train_bundle_sha256": args.expected_train_bundle_sha256,
        "exploratory_oof_bundle_sha256": args.expected_exploratory_oof_bundle_sha256,
        "pca_initialization_sha256": pca_initialization_sha,
        "baselines": {
            "primary_hard_linear": "fit-only time-shared frame PCA rank32",
            "zero_hard": True,
            "clip_pca_role": "diagnostic only",
            "deterministic_ae_step0_metric_policy": ae_step0_metric_policy,
        },
        "config": asdict(config),
        "config_sha256": _object_sha(asdict(config)),
        "fold": eval_bundle["fold"],
        "arm_best_steps": {
            "deterministic_ae": ae_receipt["best_step"],
            "direct_beta_vae": vae_receipt["best_step"],
        },
        "arm_receipt_sha256": {
            "deterministic_ae": args.expected_ae_receipt_sha256,
            "direct_beta_vae": args.expected_vae_receipt_sha256,
        },
        "arm_checkpoint_sha256": {
            "deterministic_ae": ae_receipt["checkpoint"]["sha256"],
            "direct_beta_vae": vae_receipt["checkpoint"]["sha256"],
        },
        "arm_executed_steps": {
            "deterministic_ae": ae_receipt["executed_steps"],
            "direct_beta_vae": vae_receipt["executed_steps"],
        },
        "shared_minibatch_schedule_sha256": ae_receipt[
            "executed_minibatch_schedule_sha256"
        ],
        "arm_full_beta_exposure": {
            "deterministic_ae": None,
            "direct_beta_vae": vae_receipt[
                "full_beta_exposure_steps_before_selected_checkpoint"
            ],
        },
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "vae_vs_ae_retention": vae_vs_ae,
        "vae_mechanics": vae_mechanics,
        "gates": gates,
        "action_representation_qualified": False,
        "source_identity_preservation_tested": False,
        "video_editing_tested": False,
        "prior_generation_qualified": False,
        "renderer_authorized": False,
        "inference_authorized": False,
        "vae_necessary": None,
        "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION_PER_IID",
        "full644_refit_authorized_by_single_fold": False,
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
        "gates": gates,
    }


def aggregate_oof(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    if (
        len(args.fold_receipt) != OUTER_FOLDS
        or len(args.expected_fold_receipt_sha256) != OUTER_FOLDS
    ):
        raise ValueError("v3 aggregate requires exact5 fold receipts")
    rows = [
        _load_receipt(Path(path), sha, COMPARE_SCHEMA)
        for path, sha in zip(
            args.fold_receipt, args.expected_fold_receipt_sha256
        )
    ]
    by_fold = {row["fold_index"]: row for row in rows}
    if set(by_fold) != set(range(OUTER_FOLDS)):
        raise ValueError("v3 aggregate fold closure differs")
    config = rows[0]["config"]
    Config(**config).validate()
    if any(
        row["config"] != config
        or row["config_sha256"] != _object_sha(config)
        or row["implementation"] != run_binding
        for row in rows
    ):
        raise ValueError("v3 aggregate config/implementation differs")
    authority_fields = (
        "feature_receipt_sha256", "feature_receipt_digest",
        "exact644_iid_digest", "exact644_raw_target_sha256",
        "exact644_population_authority", "outer_assignment_digest",
        "development_energy_definition", "development_energy_bin_edges_raw",
    )
    for key in authority_fields:
        if any(row[key] != rows[0][key] for row in rows):
            raise ValueError(f"v3 aggregate cross-fold {key} differs")
    if rows[0]["outer_assignment_digest"] != V2_OUTER_ASSIGNMENT_DIGEST:
        raise ValueError("v3 aggregate split differs from v2")
    if any(
        row["fold"]["iid_digest"] != V2_FOLD_IID_DIGESTS[fold]
        or row["fold"]["outer_fold"] != fold
        for fold, row in by_fold.items()
    ):
        raise ValueError("v3 aggregate per-fold v2 split differs")
    metric_names = {
        "deterministic_ae", "direct_beta_vae",
        "frame_pca_rank_l_hard_baseline", "clip_pca_rank_l_diagnostic",
        "zero_hard_baseline",
    }
    combined: dict[str, list[dict[str, Any]]] = {
        name: [] for name in metric_names
    }
    family_by_iid: dict[str, str] = {}
    for fold in range(OUTER_FOLDS):
        row = by_fold[fold]
        v2._validate_fold(row["fold"])
        if (
            {key: row.get(key) for key in DEVELOPMENT_FIELDS}
            != DEVELOPMENT_FIELDS
            or row.get("target") != {
                "raw": RAW_TARGET_DEFINITION,
                "model_coordinate": MODEL_COORDINATE_DEFINITION,
                "source_subtracted": False,
                "pca_target_used": False,
                "direct_full768_reconstruction": True,
            }
            or row.get("action_representation_qualified") is not False
            or row.get("prior_generation_qualified") is not False
            or row.get("renderer_authorized") is not False
            or row.get("inference_authorized") is not False
            or row.get("vae_necessary") is not None
            or row.get("development_energy_definition")
            != row["fold"]["energy_definition"]
            or row.get("development_energy_bin_edges_raw")
            != row["fold"]["fixed_energy_bin_edges_exact644"]
        ):
            raise ValueError("v3 aggregate fold fail-closed semantics differ")
        arm_steps = row.get("arm_executed_steps")
        if (
            type(arm_steps) is not dict
            or set(arm_steps) != set(ARMS)
            or any(step != config["max_steps"] for step in arm_steps.values())
            or row.get("shared_minibatch_schedule_sha256")
            != _sha(
                row.get("shared_minibatch_schedule_sha256"),
                "v3 shared minibatch schedule SHA",
            )
        ):
            raise ValueError("v3 aggregate fold training budget differs")
        step0_policy = row.get("baselines", {}).get(
            "deterministic_ae_step0_metric_policy"
        )
        ae_best_step = row.get("arm_best_steps", {}).get("deterministic_ae")
        expected_alias = ae_best_step == 0
        if (
            type(step0_policy) is not dict
            or set(step0_policy) != {
                "analytic_frame_pca_alias_used",
                "checkpoint_output_max_abs_vs_analytic_frame_pca",
            }
            or step0_policy["analytic_frame_pca_alias_used"] is not expected_alias
            or (
                expected_alias
                and (
                    type(step0_policy[
                        "checkpoint_output_max_abs_vs_analytic_frame_pca"
                    ]) is not float
                    or not math.isfinite(step0_policy[
                        "checkpoint_output_max_abs_vs_analytic_frame_pca"
                    ])
                    or not 0.0 <= step0_policy[
                        "checkpoint_output_max_abs_vs_analytic_frame_pca"
                    ] < STEP0_MAX_ABS_TOLERANCE
                )
            )
            or (
                not expected_alias
                and step0_policy[
                    "checkpoint_output_max_abs_vs_analytic_frame_pca"
                ] is not None
            )
        ):
            raise ValueError("v3 aggregate AE step-zero metric policy differs")
        if set(row["metrics"]) != metric_names:
            raise ValueError("v3 aggregate metric names differ")
        expected_count = row["fold"]["counts"]["exploratory_oof"]
        reference_iids: list[str] | None = None
        for name in sorted(metric_names):
            per_iid = row["metrics"][name]["per_iid"]
            if len(per_iid) != expected_count:
                raise ValueError("v3 aggregate metric count differs")
            iids = [item["iid"] for item in per_iid]
            if len(set(iids)) != expected_count:
                raise ValueError("v3 aggregate fold IIDs duplicate")
            if reference_iids is None:
                reference_iids = iids
            elif iids != reference_iids:
                raise ValueError("v3 aggregate metric IID order differs")
            combined[name].extend(per_iid)
        zero_rows = row["metrics"]["zero_hard_baseline"]["per_iid"]
        energy_edges = row["development_energy_bin_edges_raw"]
        if (
            type(energy_edges) is not list or len(energy_edges) != 4
            or any(type(edge) not in {float, int} or not math.isfinite(edge)
                   for edge in energy_edges)
            or any(
                int(item["energy_bin"])
                != sum(float(item["raw_zero_mse"]) > edge for edge in energy_edges)
                for item in zero_rows
            )
        ):
            raise ValueError("v3 aggregate energy-bin labels were re-signed")
        expected_bin_counts = row["fold"]["outer_fold_energy_bin_counts"][str(fold)]
        actual_bin_counts = [
            sum(int(item["energy_bin"]) == index for item in zero_rows)
            for index in range(5)
        ]
        if actual_bin_counts != expected_bin_counts:
            raise ValueError("v3 aggregate fold energy-bin authority differs")
        for zero_item in zero_rows:
            if not (
                math.isclose(
                    float(zero_item["mse"]), float(zero_item["zero_mse"]),
                    rel_tol=0.0, abs_tol=1.0e-12,
                )
                and math.isclose(
                    float(zero_item["raw_mse"]),
                    float(zero_item["raw_zero_mse"]),
                    rel_tol=0.0, abs_tol=1.0e-12,
                )
                and math.isclose(
                    float(zero_item["raw_output_energy"]), 0.0,
                    rel_tol=0.0, abs_tol=1.0e-12,
                )
            ):
                raise ValueError("v3 aggregate zero baseline was re-signed")
        zero_by_iid = {item["iid"]: item for item in zero_rows}
        for name in metric_names - {"zero_hard_baseline"}:
            for item in row["metrics"][name]["per_iid"]:
                authority_item = zero_by_iid[item["iid"]]
                if (
                    int(item["energy_bin"]) != int(authority_item["energy_bin"])
                    or not math.isclose(
                        float(item["zero_mse"]),
                        float(authority_item["zero_mse"]),
                        rel_tol=0.0, abs_tol=1.0e-12,
                    )
                    or not math.isclose(
                        float(item["raw_zero_mse"]),
                        float(authority_item["raw_zero_mse"]),
                        rel_tol=0.0, abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError("v3 aggregate target metric authority differs")
        fold_family = row["family_by_iid"]
        if (
            reference_iids is None
            or _object_sha(reference_iids)
            != row["fold"]["exploratory_oof_iid_digest"]
        ):
            raise ValueError("v3 aggregate fold OOF IID digest differs")
        if set(fold_family) != set(reference_iids or []):
            raise ValueError("v3 aggregate fold family closure differs")
        if set(family_by_iid) & set(fold_family):
            raise ValueError("v3 aggregate family IID overlap")
        family_by_iid.update(fold_family)
    all_iids = [row["iid"] for row in combined["zero_hard_baseline"]]
    if (
        len(all_iids) != 644 or len(set(all_iids)) != 644
        or _object_sha(sorted(all_iids)) != rows[0]["exact644_iid_digest"]
        or set(family_by_iid) != set(all_iids)
        or len(set(family_by_iid.values())) != 28
    ):
        raise ValueError("v3 aggregate exact644/family closure differs")
    for name in metric_names:
        if [row["iid"] for row in combined[name]] != all_iids:
            raise ValueError("v3 aggregate metric exact644 order differs")
    aggregate_metrics = {
        name: v2._aggregate_metric_rows(values)
        for name, values in combined.items()
    }
    comparisons: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        comparisons[arm] = {}
        for baseline_index, (label, baseline) in enumerate((
            ("zero", "zero_hard_baseline"),
            ("frame_pca_rank_l", "frame_pca_rank_l_hard_baseline"),
        )):
            comparisons[arm][f"vs_{label}"] = {
                "clip_bootstrap": v2._paired_ratio(
                    combined[arm], combined[baseline],
                    config["seed"] + 9000 + 10 * arm_index + baseline_index,
                ),
                "family_cluster_bootstrap": _family_cluster_ratio(
                    combined[arm], combined[baseline], family_by_iid,
                    config["seed"] + 9100 + 10 * arm_index + baseline_index,
                    expected_family_count=28,
                ),
            }
        comparisons[arm]["temporal_delta_vs_frame_pca"] = {
            "clip_bootstrap": v2._paired_ratio(
                combined[arm], combined["frame_pca_rank_l_hard_baseline"],
                config["seed"] + 9200 + arm_index,
                metric_key="raw_temporal_delta_mse",
            ),
            "family_cluster_bootstrap": _family_cluster_ratio(
                combined[arm], combined["frame_pca_rank_l_hard_baseline"],
                family_by_iid, config["seed"] + 9300 + arm_index,
                metric_key="raw_temporal_delta_mse", expected_family_count=28,
            ),
        }
        comparisons[arm]["cosine_improvement_vs_frame_pca"] = {
            "clip_bootstrap": _paired_improvement(
                combined[arm], combined["frame_pca_rank_l_hard_baseline"],
                "cosine", config["seed"] + 9400 + arm_index,
            ),
            "family_cluster_bootstrap": _family_cluster_improvement(
                combined[arm], combined["frame_pca_rank_l_hard_baseline"],
                family_by_iid, "cosine", config["seed"] + 9500 + arm_index,
                expected_family_count=28,
            ),
        }
        energy = _energy_ratio_gates(aggregate_metrics, arm)
        fold_point_values = {}
        for fold in range(OUTER_FOLDS):
            fold_metrics = by_fold[fold]["metrics"]
            arm_rows = fold_metrics[arm]["per_iid"]
            zero_rows = fold_metrics["zero_hard_baseline"]["per_iid"]
            pca_rows = fold_metrics[
                "frame_pca_rank_l_hard_baseline"
            ]["per_iid"]
            arm_sum = sum(float(item["raw_mse"]) for item in arm_rows)
            zero_sum = sum(float(item["raw_mse"]) for item in zero_rows)
            pca_sum = sum(float(item["raw_mse"]) for item in pca_rows)
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (arm_sum, zero_sum, pca_sum)
            ):
                raise ValueError("v3 aggregate fold point metrics differ")
            recomputed = {
                "vs_zero": arm_sum / zero_sum,
                "vs_frame_pca": arm_sum / pca_sum,
            }
            recorded = by_fold[fold]["paired_comparisons"][arm]
            if not (
                math.isclose(
                    recomputed["vs_zero"],
                    recorded["vs_zero"]["clip_bootstrap"]["mean_ratio"],
                    rel_tol=1.0e-12, abs_tol=1.0e-12,
                )
                and math.isclose(
                    recomputed["vs_frame_pca"],
                    recorded["vs_frame_pca_rank_l"]["clip_bootstrap"]["mean_ratio"],
                    rel_tol=1.0e-12, abs_tol=1.0e-12,
                )
            ):
                raise ValueError("v3 aggregate fold point receipt was re-signed")
            fold_point_values[str(fold)] = recomputed
        gates[arm] = {
            "clip_ratio_ucb_lt_1_vs_zero": comparisons[arm]["vs_zero"]["clip_bootstrap"]["ratio_95pct_ci"][1] < 1.0,
            "family_ratio_ucb_lt_1_vs_zero": comparisons[arm]["vs_zero"]["family_cluster_bootstrap"]["ratio_95pct_ci"][1] < 1.0,
            "clip_ratio_ucb_lt_1_vs_frame_pca": comparisons[arm]["vs_frame_pca_rank_l"]["clip_bootstrap"]["ratio_95pct_ci"][1] < 1.0,
            "family_ratio_ucb_lt_1_vs_frame_pca": comparisons[arm]["vs_frame_pca_rank_l"]["family_cluster_bootstrap"]["ratio_95pct_ci"][1] < 1.0,
            "all_five_folds_point_ratio_lt_1_vs_both": all(
                value["vs_zero"] < 1.0 and value["vs_frame_pca"] < 1.0
                for value in fold_point_values.values()
            ),
            "fold_point_ratios": fold_point_values,
            "all_five_energy_bins_point_ratio_le_1_vs_zero": energy["all_bins_point_ratio_vs_zero_le_1"],
            "all_five_energy_bins_point_ratio_le_1_vs_frame_pca": energy["all_bins_point_ratio_vs_frame_pca_le_1"],
            "energy_strata": energy["by_bin"],
            "temporal_delta_clip_ucb_lt_1_vs_frame_pca": comparisons[arm]["temporal_delta_vs_frame_pca"]["clip_bootstrap"]["ratio_95pct_ci"][1] < 1.0,
            "temporal_delta_family_ucb_lt_1_vs_frame_pca": comparisons[arm]["temporal_delta_vs_frame_pca"]["family_cluster_bootstrap"]["ratio_95pct_ci"][1] < 1.0,
            "cosine_clip_lcb_gt_0_vs_frame_pca": comparisons[arm]["cosine_improvement_vs_frame_pca"]["clip_bootstrap"]["improvement_95pct_ci"][0] > 0.0,
            "cosine_family_lcb_gt_0_vs_frame_pca": comparisons[arm]["cosine_improvement_vs_frame_pca"]["family_cluster_bootstrap"]["improvement_95pct_ci"][0] > 0.0,
        }
    expected_vae_mechanics_keys = {
        "residual_only_kl_element_mean",
        "residual_only_kl_per_latent_dim",
        "residual_mean_variance_per_dim",
        "residual_active_unit_count",
        "residual_active_unit_thresholds",
        "residual_mean_energy",
        "residual_shuffle_seed",
        "residual_shuffle_offset",
        "residual_shuffle_fixed_point_count",
        "residual_shuffle_is_cross_clip_not_prior_coverage",
        "posterior_mean_mse",
        "zero_learned_residual_mse",
        "cross_clip_shuffled_residual_mse",
        "residual_mean_improves_over_zero_residual",
        "residual_shuffle_increases_mse",
        "residual_intervention_per_iid",
        "posterior_mc_expected_per_iid",
        "posterior_sample_count",
        "posterior_sample_output_variance",
        "stochastic_residual_changes_output",
        "full_pca_latent_excluded_from_kl_and_active_unit_metrics",
        "prior_coverage_tested",
    }
    intervention_keys = {
        "iid", "posterior_raw_mse", "zero_residual_raw_mse",
        "analytic_pca_residual_raw_mse", "shuffled_residual_raw_mse",
        "normalized_sample_output_variance",
        "posterior_sample_output_variance_model_coordinate",
        "posterior_sample_output_variance_raw",
    }
    mc_keys = {"iid", "raw_mse", "raw_temporal_delta_mse", "cosine"}
    vae_interventions: list[dict[str, Any]] = []
    vae_mc_rows: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        mechanics = by_fold[fold].get("vae_mechanics")
        fold_iids = [
            item["iid"] for item in by_fold[fold]["metrics"][
                "zero_hard_baseline"
            ]["per_iid"]
        ]
        if (
            type(mechanics) is not dict
            or set(mechanics) != expected_vae_mechanics_keys
            or mechanics["posterior_sample_count"]
            != POSTERIOR_MC_SAMPLE_COUNT
            or mechanics[
                "full_pca_latent_excluded_from_kl_and_active_unit_metrics"
            ] is not True
            or mechanics["prior_coverage_tested"] is not False
            or mechanics["residual_shuffle_is_cross_clip_not_prior_coverage"]
            is not True
            or mechanics["residual_shuffle_seed"]
            != config["seed"] + 8000 + fold
            or type(mechanics["residual_shuffle_offset"]) is not int
            or not 1 <= mechanics["residual_shuffle_offset"] < len(fold_iids)
            or mechanics["residual_shuffle_fixed_point_count"] != 0
            or mechanics["residual_active_unit_thresholds"] != {
                "kl_per_dim_gt": 1.0e-4,
                "posterior_mean_variance_gt": 1.0e-3,
            }
        ):
            raise ValueError("v3 aggregate VAE mechanics contract differs")
        kl_by_dim = mechanics["residual_only_kl_per_latent_dim"]
        variance_by_dim = mechanics["residual_mean_variance_per_dim"]
        if (
            type(kl_by_dim) is not list
            or type(variance_by_dim) is not list
            or len(kl_by_dim) != config["latent_dim"]
            or len(variance_by_dim) != config["latent_dim"]
            or not all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in (*kl_by_dim, *variance_by_dim)
            )
            or not math.isclose(
                float(mechanics["residual_only_kl_element_mean"]),
                sum(float(value) for value in kl_by_dim) / len(kl_by_dim),
                rel_tol=1.0e-7, abs_tol=1.0e-9,
            )
            or mechanics["residual_active_unit_count"] != sum(
                float(kl) > 1.0e-4 and float(variance) > 1.0e-3
                for kl, variance in zip(kl_by_dim, variance_by_dim)
            )
        ):
            raise ValueError("v3 aggregate VAE residual statistics differ")
        interventions = mechanics["residual_intervention_per_iid"]
        mc_rows = mechanics["posterior_mc_expected_per_iid"]
        if (
            type(interventions) is not list
            or type(mc_rows) is not list
            or [item.get("iid") for item in interventions] != fold_iids
            or [item.get("iid") for item in mc_rows] != fold_iids
            or any(type(item) is not dict or set(item) != intervention_keys
                   for item in interventions)
            or any(type(item) is not dict or set(item) != mc_keys
                   for item in mc_rows)
        ):
            raise ValueError("v3 aggregate VAE per-IID mechanics differ")
        numeric_values = [
            float(item[key])
            for item in interventions
            for key in intervention_keys - {"iid"}
        ] + [
            float(item[key])
            for item in mc_rows
            for key in mc_keys - {"iid"}
        ]
        if (
            not all(math.isfinite(value) for value in numeric_values)
            or any(
                float(item[key]) < 0.0
                for item in interventions
                for key in intervention_keys - {"iid"}
            )
            or any(
                float(item[key]) < 0.0
                for item in mc_rows
                for key in {"raw_mse", "raw_temporal_delta_mse"}
            )
        ):
            raise ValueError("v3 aggregate VAE mechanics are non-finite")
        direct_rows = by_fold[fold]["metrics"]["direct_beta_vae"]["per_iid"]
        analytic_pca_rows = by_fold[fold]["metrics"][
            "frame_pca_rank_l_hard_baseline"
        ]["per_iid"]
        if any(
            not math.isclose(
                float(intervention["posterior_raw_mse"]),
                float(direct["raw_mse"]),
                rel_tol=1.0e-7, abs_tol=1.0e-10,
            )
            for intervention, direct in zip(interventions, direct_rows)
        ):
            raise ValueError("v3 aggregate VAE posterior metric join differs")
        if any(
            not math.isclose(
                float(intervention["analytic_pca_residual_raw_mse"]),
                float(analytic["raw_mse"]),
                rel_tol=1.0e-7, abs_tol=1.0e-10,
            )
            for intervention, analytic in zip(interventions, analytic_pca_rows)
        ):
            raise ValueError("v3 aggregate VAE PCA residual scale join differs")
        if any(
            float(item["analytic_pca_residual_raw_mse"]) <= 0.0
            or not math.isclose(
                float(item["normalized_sample_output_variance"]),
                float(item["posterior_sample_output_variance_raw"])
                / float(item["analytic_pca_residual_raw_mse"]),
                rel_tol=1.0e-6, abs_tol=1.0e-10,
            )
            for item in interventions
        ):
            raise ValueError("v3 aggregate normalized VAE variance differs")
        posterior_sum = sum(
            float(item["posterior_raw_mse"]) for item in interventions
        )
        zero_sum = sum(
            float(item["zero_residual_raw_mse"]) for item in interventions
        )
        shuffled_sum = sum(
            float(item["shuffled_residual_raw_mse"]) for item in interventions
        )
        sample_variance_mean = sum(
            float(item["posterior_sample_output_variance_model_coordinate"])
            for item in interventions
        ) / len(interventions)
        if (
            mechanics["residual_mean_improves_over_zero_residual"]
            is not (posterior_sum < zero_sum)
            or mechanics["residual_shuffle_increases_mse"]
            is not (shuffled_sum > posterior_sum)
            or not math.isclose(
                float(mechanics["posterior_sample_output_variance"]),
                sample_variance_mean, rel_tol=1.0e-7, abs_tol=1.0e-10,
            )
            or mechanics["stochastic_residual_changes_output"]
            is not (sample_variance_mean > 1.0e-10)
        ):
            raise ValueError("v3 aggregate VAE mechanics booleans were re-signed")
        vae_interventions.extend(interventions)
        vae_mc_rows.extend(mc_rows)
    if (
        [item["iid"] for item in vae_interventions] != all_iids
        or [item["iid"] for item in vae_mc_rows] != all_iids
    ):
        raise ValueError("v3 aggregate VAE exact644 mechanics order differs")
    posterior_residual_rows = [
        {"iid": item["iid"], "raw_mse": item["posterior_raw_mse"]}
        for item in vae_interventions
    ]
    zero_residual_rows = [
        {"iid": item["iid"], "raw_mse": item["zero_residual_raw_mse"]}
        for item in vae_interventions
    ]
    shuffled_residual_rows = [
        {"iid": item["iid"], "raw_mse": item["shuffled_residual_raw_mse"]}
        for item in vae_interventions
    ]
    sample_variance_rows = [
        {
            "iid": item["iid"],
            "normalized_sample_output_variance": item[
                "normalized_sample_output_variance"
            ],
        }
        for item in vae_interventions
    ]
    zero_variance_rows = [
        {"iid": item["iid"], "normalized_sample_output_variance": 0.0}
        for item in vae_interventions
    ]
    vae_residual_effects = {
        "posterior_vs_zero_residual": {
            "clip_bootstrap": v2._paired_ratio(
                posterior_residual_rows, zero_residual_rows,
                config["seed"] + 9750,
            ),
            "family_cluster_bootstrap": _family_cluster_ratio(
                posterior_residual_rows, zero_residual_rows, family_by_iid,
                config["seed"] + 9751, expected_family_count=28,
            ),
        },
        "posterior_vs_shuffled_residual": {
            "clip_bootstrap": v2._paired_ratio(
                posterior_residual_rows, shuffled_residual_rows,
                config["seed"] + 9752,
            ),
            "family_cluster_bootstrap": _family_cluster_ratio(
                posterior_residual_rows, shuffled_residual_rows, family_by_iid,
                config["seed"] + 9753, expected_family_count=28,
            ),
        },
        "normalized_posterior_sample_output_variance": {
            "clip_bootstrap": _paired_improvement(
                sample_variance_rows, zero_variance_rows,
                "normalized_sample_output_variance", config["seed"] + 9754,
            ),
            "family_cluster_bootstrap": _family_cluster_improvement(
                sample_variance_rows, zero_variance_rows, family_by_iid,
                "normalized_sample_output_variance", config["seed"] + 9755,
                expected_family_count=28,
            ),
        },
    }
    vae_mc_comparisons: dict[str, Any] = {}
    for index, (label, baseline) in enumerate((
        ("zero", combined["zero_hard_baseline"]),
        ("frame_pca_rank_l", combined["frame_pca_rank_l_hard_baseline"]),
    )):
        vae_mc_comparisons[f"vs_{label}"] = {
            "clip_bootstrap": v2._paired_ratio(
                vae_mc_rows, baseline, config["seed"] + 9760 + index,
            ),
            "family_cluster_bootstrap": _family_cluster_ratio(
                vae_mc_rows, baseline, family_by_iid,
                config["seed"] + 9770 + index, expected_family_count=28,
            ),
        }
    vae_vs_ae = {
        "clip_bootstrap": v2._paired_ratio(
            combined["direct_beta_vae"], combined["deterministic_ae"],
            config["seed"] + 9600,
        ),
        "family_cluster_bootstrap": _family_cluster_ratio(
            combined["direct_beta_vae"], combined["deterministic_ae"],
            family_by_iid, config["seed"] + 9700, expected_family_count=28,
        ),
    }
    vae_full_beta = all(
        by_fold[fold]["arm_best_steps"]["direct_beta_vae"]
        >= config["kl_warmup_steps"] + config["full_beta_plateau_steps"]
        and by_fold[fold]["arm_full_beta_exposure"]["direct_beta_vae"]
        >= config["full_beta_plateau_steps"]
        for fold in range(OUTER_FOLDS)
    )
    vae_mechanics_diagnostics = {
        "all_folds_full_beta_eligible": vae_full_beta,
        "all_folds_finite_positive_residual_kl": all(
            math.isfinite(by_fold[fold]["vae_mechanics"]["residual_only_kl_element_mean"])
            and by_fold[fold]["vae_mechanics"]["residual_only_kl_element_mean"] > 0.0
            for fold in range(OUTER_FOLDS)
        ),
        "all_folds_have_residual_active_units": all(
            by_fold[fold]["vae_mechanics"]["residual_active_unit_count"] > 0
            for fold in range(OUTER_FOLDS)
        ),
        "all_folds_residual_shuffle_increases_mse": all(
            by_fold[fold]["vae_mechanics"]["residual_shuffle_increases_mse"] is True
            for fold in range(OUTER_FOLDS)
        ),
        "all_folds_residual_mean_improves_over_zero_residual": all(
            by_fold[fold]["vae_mechanics"][
                "residual_mean_improves_over_zero_residual"
            ] is True
            for fold in range(OUTER_FOLDS)
        ),
        "all_folds_stochastic_residual_changes_output": all(
            by_fold[fold]["vae_mechanics"][
                "stochastic_residual_changes_output"
            ] is True
            for fold in range(OUTER_FOLDS)
        ),
    }
    vae_fold_mechanics_summary = {
        str(fold): {
            "selected_step": by_fold[fold]["arm_best_steps"][
                "direct_beta_vae"
            ],
            "full_beta_exposure_steps": by_fold[fold][
                "arm_full_beta_exposure"
            ]["direct_beta_vae"],
            "residual_only_kl_element_mean": by_fold[fold][
                "vae_mechanics"
            ]["residual_only_kl_element_mean"],
            "residual_active_unit_count": by_fold[fold]["vae_mechanics"][
                "residual_active_unit_count"
            ],
        }
        for fold in range(OUTER_FOLDS)
    }
    vae_mechanics_gates = {
        "all_folds_full_beta_eligible": vae_full_beta,
        "all_folds_finite_positive_residual_kl": vae_mechanics_diagnostics[
            "all_folds_finite_positive_residual_kl"
        ],
        "all_folds_have_residual_active_units": vae_mechanics_diagnostics[
            "all_folds_have_residual_active_units"
        ],
        "posterior_vs_zero_residual_clip_ucb_lt_1": (
            vae_residual_effects["posterior_vs_zero_residual"][
                "clip_bootstrap"
            ]["ratio_95pct_ci"][1] < 1.0
        ),
        "posterior_vs_zero_residual_family_ucb_lt_1": (
            vae_residual_effects["posterior_vs_zero_residual"][
                "family_cluster_bootstrap"
            ]["ratio_95pct_ci"][1] < 1.0
        ),
        "posterior_vs_shuffled_residual_clip_ucb_lt_1": (
            vae_residual_effects["posterior_vs_shuffled_residual"][
                "clip_bootstrap"
            ]["ratio_95pct_ci"][1] < 1.0
        ),
        "posterior_vs_shuffled_residual_family_ucb_lt_1": (
            vae_residual_effects["posterior_vs_shuffled_residual"][
                "family_cluster_bootstrap"
            ]["ratio_95pct_ci"][1] < 1.0
        ),
        "normalized_sample_variance_clip_lcb_gt_floor": (
            vae_residual_effects[
                "normalized_posterior_sample_output_variance"
            ]["clip_bootstrap"]["improvement_95pct_ci"][0]
            > VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR
        ),
        "normalized_sample_variance_family_lcb_gt_floor": (
            vae_residual_effects[
                "normalized_posterior_sample_output_variance"
            ]["family_cluster_bootstrap"]["improvement_95pct_ci"][0]
            > VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR
        ),
        "posterior_mc8_clip_ucb_lt_1_vs_zero": (
            vae_mc_comparisons["vs_zero"]["clip_bootstrap"]
            ["ratio_95pct_ci"][1] < 1.0
        ),
        "posterior_mc8_family_ucb_lt_1_vs_zero": (
            vae_mc_comparisons["vs_zero"]["family_cluster_bootstrap"]
            ["ratio_95pct_ci"][1] < 1.0
        ),
        "posterior_mc8_clip_ucb_lt_1_vs_frame_pca": (
            vae_mc_comparisons["vs_frame_pca_rank_l"]["clip_bootstrap"]
            ["ratio_95pct_ci"][1] < 1.0
        ),
        "posterior_mc8_family_ucb_lt_1_vs_frame_pca": (
            vae_mc_comparisons["vs_frame_pca_rank_l"]
            ["family_cluster_bootstrap"]["ratio_95pct_ci"][1] < 1.0
        ),
        "vae_vs_ae_clip_retention_ucb_le_1p02": (
            vae_vs_ae["clip_bootstrap"]["ratio_95pct_ci"][1] <= 1.02
        ),
        "vae_vs_ae_family_retention_ucb_le_1p02": (
            vae_vs_ae["family_cluster_bootstrap"]["ratio_95pct_ci"][1] <= 1.02
        ),
    }
    if set(vae_mechanics_gates) != set(VAE_AGGREGATE_BOOLEAN_GATE_KEYS):
        raise RuntimeError("v3 internal VAE aggregate gate schema differs")
    gates["direct_beta_vae"].update(vae_mechanics_gates)
    for arm in ARMS:
        expected_without_hard = _expected_aggregate_gate_keys(arm) - {
            "aggregate_hard_gate"
        }
        if set(gates[arm]) != expected_without_hard:
            raise RuntimeError("v3 internal aggregate gate schema differs")
        gates[arm]["aggregate_hard_gate"] = bool(all(
            value is True for key, value in gates[arm].items()
            if key not in {"fold_point_ratios", "energy_strata"}
        ))
        if set(gates[arm]) != _expected_aggregate_gate_keys(arm):
            raise RuntimeError("v3 internal aggregate hard gate schema differs")
    authorized_arms = [
        arm for arm in ARMS if gates[arm]["aggregate_hard_gate"]
    ]
    fold_point_ratios_by_arm = {
        arm: gates[arm]["fold_point_ratios"] for arm in ARMS
    }
    selected_steps = {
        arm: int(torch.tensor([
            by_fold[fold]["arm_best_steps"][arm]
            for fold in range(OUTER_FOLDS)
        ], dtype=torch.float64).median())
        for arm in ARMS
    }
    if "direct_beta_vae" in authorized_arms and selected_steps["direct_beta_vae"] < (
        config["kl_warmup_steps"] + config["full_beta_plateau_steps"]
    ):
        raise RuntimeError("v3 aggregate authorized a non-full-beta VAE refit")
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    receipt: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "V3_EXPLORATORY_5FOLD_OOF_AGGREGATED_BURNED644",
        **DEVELOPMENT_FIELDS,
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
        "arm_receipt_sha256_by_fold": {
            str(fold): by_fold[fold]["arm_receipt_sha256"]
            for fold in range(OUTER_FOLDS)
        },
        "arm_checkpoint_sha256_by_fold": {
            str(fold): by_fold[fold]["arm_checkpoint_sha256"]
            for fold in range(OUTER_FOLDS)
        },
        "family_labels_used_only_for_split_and_evaluation_statistics": True,
        "family_or_transform_labels_consumed_by_model_or_optimizer": False,
        "aggregate_metrics": aggregate_metrics,
        "paired_comparisons": comparisons,
        "fold_point_ratios_by_arm": fold_point_ratios_by_arm,
        "vae_vs_ae_retention": vae_vs_ae,
        "vae_residual_effects": vae_residual_effects,
        "vae_posterior_mc8_comparisons": vae_mc_comparisons,
        "vae_mechanics_diagnostics": vae_mechanics_diagnostics,
        "vae_fold_mechanics_summary": vae_fold_mechanics_summary,
        "vae_mechanics_gates": vae_mechanics_gates,
        "vae_normalized_sample_variance_lcb_floor": (
            VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR
        ),
        "gates": gates,
        "selected_steps_by_fold_by_arm": {
            str(fold): by_fold[fold]["arm_best_steps"]
            for fold in range(OUTER_FOLDS)
        },
        "full644_refit_step_preregistration": {
            "strategy": "median of exact5 selected fold steps",
            "steps_by_arm": selected_steps,
            "frozen_before_refit": True,
        },
        "full644_refit_authorized_by_arm": {
            arm: arm in authorized_arms for arm in ARMS
        },
        "full644_refit_authorized_arms": authorized_arms,
        "full644_refit_any_arm_authorized": bool(authorized_arms),
        "action_representation_qualified": False,
        "source_identity_preservation_tested": False,
        "video_editing_tested": False,
        "prior_generation_qualified": False,
        "renderer_authorized": False,
        "inference_authorized": False,
        "vae_necessary": None,
        "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION_PER_IID",
        "implementation": run_binding,
    }
    if _authorized_arms_from_aggregate(receipt) != authorized_arms:
        raise RuntimeError("v3 internal aggregate authorization differs")
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "full644_refit_authorized_arms": authorized_arms,
        "steps_by_arm": selected_steps,
    }


def _recompute_aggregate_boolean_gates(
    aggregate: Mapping[str, Any],
) -> dict[str, dict[str, bool]]:
    """Recompute every authorization boolean from stored metric evidence."""

    def ci_endpoint(value: Any, key: str, index: int) -> float:
        if type(value) is not dict:
            raise ValueError("v3 aggregate bootstrap evidence differs")
        interval = value.get(key)
        if type(interval) is not list or len(interval) != 2:
            raise ValueError("v3 aggregate bootstrap interval differs")
        endpoint = float(interval[index])
        if not math.isfinite(endpoint):
            raise ValueError("v3 aggregate bootstrap endpoint is non-finite")
        return endpoint

    try:
        config_value = aggregate["config"]
        config = Config(**config_value)
        config.validate()
        comparisons = aggregate["paired_comparisons"]
        metrics = aggregate["aggregate_metrics"]
        fold_ratios_by_arm = aggregate["fold_point_ratios_by_arm"]
        recomputed: dict[str, dict[str, bool]] = {}
        for arm in ARMS:
            arm_comparisons = comparisons[arm]
            fold_ratios = fold_ratios_by_arm[arm]
            if (
                type(fold_ratios) is not dict
                or set(fold_ratios)
                != {str(fold) for fold in range(OUTER_FOLDS)}
            ):
                raise ValueError("v3 aggregate fold-point evidence differs")
            for value in fold_ratios.values():
                if type(value) is not dict or set(value) != {
                    "vs_zero", "vs_frame_pca"
                }:
                    raise ValueError("v3 aggregate fold-point row differs")
                if not all(
                    math.isfinite(float(value[key]))
                    for key in ("vs_zero", "vs_frame_pca")
                ):
                    raise ValueError("v3 aggregate fold-point value differs")
            energy = _energy_ratio_gates(metrics, arm)
            recomputed[arm] = {
                "clip_ratio_ucb_lt_1_vs_zero": ci_endpoint(
                    arm_comparisons["vs_zero"]["clip_bootstrap"],
                    "ratio_95pct_ci", 1,
                ) < 1.0,
                "family_ratio_ucb_lt_1_vs_zero": ci_endpoint(
                    arm_comparisons["vs_zero"]["family_cluster_bootstrap"],
                    "ratio_95pct_ci", 1,
                ) < 1.0,
                "clip_ratio_ucb_lt_1_vs_frame_pca": ci_endpoint(
                    arm_comparisons["vs_frame_pca_rank_l"]["clip_bootstrap"],
                    "ratio_95pct_ci", 1,
                ) < 1.0,
                "family_ratio_ucb_lt_1_vs_frame_pca": ci_endpoint(
                    arm_comparisons["vs_frame_pca_rank_l"]
                    ["family_cluster_bootstrap"], "ratio_95pct_ci", 1,
                ) < 1.0,
                "all_five_folds_point_ratio_lt_1_vs_both": all(
                    float(value["vs_zero"]) < 1.0
                    and float(value["vs_frame_pca"]) < 1.0
                    for value in fold_ratios.values()
                ),
                "all_five_energy_bins_point_ratio_le_1_vs_zero": energy[
                    "all_bins_point_ratio_vs_zero_le_1"
                ],
                "all_five_energy_bins_point_ratio_le_1_vs_frame_pca": energy[
                    "all_bins_point_ratio_vs_frame_pca_le_1"
                ],
                "temporal_delta_clip_ucb_lt_1_vs_frame_pca": ci_endpoint(
                    arm_comparisons["temporal_delta_vs_frame_pca"]
                    ["clip_bootstrap"], "ratio_95pct_ci", 1,
                ) < 1.0,
                "temporal_delta_family_ucb_lt_1_vs_frame_pca": ci_endpoint(
                    arm_comparisons["temporal_delta_vs_frame_pca"]
                    ["family_cluster_bootstrap"], "ratio_95pct_ci", 1,
                ) < 1.0,
                "cosine_clip_lcb_gt_0_vs_frame_pca": ci_endpoint(
                    arm_comparisons["cosine_improvement_vs_frame_pca"]
                    ["clip_bootstrap"], "improvement_95pct_ci", 0,
                ) > 0.0,
                "cosine_family_lcb_gt_0_vs_frame_pca": ci_endpoint(
                    arm_comparisons["cosine_improvement_vs_frame_pca"]
                    ["family_cluster_bootstrap"], "improvement_95pct_ci", 0,
                ) > 0.0,
            }
            if set(recomputed[arm]) != set(COMMON_AGGREGATE_BOOLEAN_GATE_KEYS):
                raise RuntimeError("v3 recomputed common gate schema differs")

        summaries = aggregate["vae_fold_mechanics_summary"]
        if (
            type(summaries) is not dict
            or set(summaries) != {str(fold) for fold in range(OUTER_FOLDS)}
        ):
            raise ValueError("v3 aggregate VAE fold summary differs")
        expected_summary_keys = {
            "selected_step", "full_beta_exposure_steps",
            "residual_only_kl_element_mean", "residual_active_unit_count",
        }
        for summary in summaries.values():
            if type(summary) is not dict or set(summary) != expected_summary_keys:
                raise ValueError("v3 aggregate VAE fold summary row differs")
        effects = aggregate["vae_residual_effects"]
        mc = aggregate["vae_posterior_mc8_comparisons"]
        retention = aggregate["vae_vs_ae_retention"]
        variance_effect = effects[
            "normalized_posterior_sample_output_variance"
        ]
        vae = {
            "all_folds_full_beta_eligible": all(
                type(summary["selected_step"]) is int
                and summary["selected_step"]
                >= config.kl_warmup_steps + config.full_beta_plateau_steps
                and type(summary["full_beta_exposure_steps"]) is int
                and summary["full_beta_exposure_steps"]
                >= config.full_beta_plateau_steps
                for summary in summaries.values()
            ),
            "all_folds_finite_positive_residual_kl": all(
                math.isfinite(float(summary["residual_only_kl_element_mean"]))
                and float(summary["residual_only_kl_element_mean"]) > 0.0
                for summary in summaries.values()
            ),
            "all_folds_have_residual_active_units": all(
                type(summary["residual_active_unit_count"]) is int
                and summary["residual_active_unit_count"] > 0
                for summary in summaries.values()
            ),
            "posterior_vs_zero_residual_clip_ucb_lt_1": ci_endpoint(
                effects["posterior_vs_zero_residual"]["clip_bootstrap"],
                "ratio_95pct_ci", 1,
            ) < 1.0,
            "posterior_vs_zero_residual_family_ucb_lt_1": ci_endpoint(
                effects["posterior_vs_zero_residual"]
                ["family_cluster_bootstrap"], "ratio_95pct_ci", 1,
            ) < 1.0,
            "posterior_vs_shuffled_residual_clip_ucb_lt_1": ci_endpoint(
                effects["posterior_vs_shuffled_residual"]["clip_bootstrap"],
                "ratio_95pct_ci", 1,
            ) < 1.0,
            "posterior_vs_shuffled_residual_family_ucb_lt_1": ci_endpoint(
                effects["posterior_vs_shuffled_residual"]
                ["family_cluster_bootstrap"], "ratio_95pct_ci", 1,
            ) < 1.0,
            "normalized_sample_variance_clip_lcb_gt_floor": ci_endpoint(
                variance_effect["clip_bootstrap"],
                "improvement_95pct_ci", 0,
            ) > VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR,
            "normalized_sample_variance_family_lcb_gt_floor": ci_endpoint(
                variance_effect["family_cluster_bootstrap"],
                "improvement_95pct_ci", 0,
            ) > VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR,
            "posterior_mc8_clip_ucb_lt_1_vs_zero": ci_endpoint(
                mc["vs_zero"]["clip_bootstrap"], "ratio_95pct_ci", 1,
            ) < 1.0,
            "posterior_mc8_family_ucb_lt_1_vs_zero": ci_endpoint(
                mc["vs_zero"]["family_cluster_bootstrap"],
                "ratio_95pct_ci", 1,
            ) < 1.0,
            "posterior_mc8_clip_ucb_lt_1_vs_frame_pca": ci_endpoint(
                mc["vs_frame_pca_rank_l"]["clip_bootstrap"],
                "ratio_95pct_ci", 1,
            ) < 1.0,
            "posterior_mc8_family_ucb_lt_1_vs_frame_pca": ci_endpoint(
                mc["vs_frame_pca_rank_l"]["family_cluster_bootstrap"],
                "ratio_95pct_ci", 1,
            ) < 1.0,
            "vae_vs_ae_clip_retention_ucb_le_1p02": ci_endpoint(
                retention["clip_bootstrap"], "ratio_95pct_ci", 1,
            ) <= 1.02,
            "vae_vs_ae_family_retention_ucb_le_1p02": ci_endpoint(
                retention["family_cluster_bootstrap"],
                "ratio_95pct_ci", 1,
            ) <= 1.02,
        }
        if set(vae) != set(VAE_AGGREGATE_BOOLEAN_GATE_KEYS):
            raise RuntimeError("v3 recomputed VAE gate schema differs")
        recomputed["direct_beta_vae"].update(vae)
        return recomputed
    except (KeyError, IndexError, TypeError, OverflowError) as error:
        raise ValueError("v3 aggregate gate evidence differs") from error


def _authorized_arms_from_aggregate(aggregate: Mapping[str, Any]) -> list[str]:
    gates = aggregate.get("gates")
    if type(gates) is not dict or set(gates) != set(ARMS):
        raise ValueError("v3 aggregate gate closure differs")
    recomputed_evidence = _recompute_aggregate_boolean_gates(aggregate)
    expected = []
    for arm in ARMS:
        if (
            type(gates[arm]) is not dict
            or set(gates[arm]) != _expected_aggregate_gate_keys(arm)
            or type(gates[arm]["fold_point_ratios"]) is not dict
            or set(gates[arm]["fold_point_ratios"])
            != {str(fold) for fold in range(OUTER_FOLDS)}
            or type(gates[arm]["energy_strata"]) is not dict
            or set(gates[arm]["energy_strata"]) != {
                str(index) for index in range(5)
            }
        ):
            raise ValueError("v3 aggregate arm gate differs")
        boolean_keys = (
            set(COMMON_AGGREGATE_BOOLEAN_GATE_KEYS)
            | (set(VAE_AGGREGATE_BOOLEAN_GATE_KEYS)
               if arm == "direct_beta_vae" else set())
            | {"aggregate_hard_gate"}
        )
        if any(type(gates[arm][key]) is not bool for key in boolean_keys):
            raise ValueError("v3 aggregate gate type differs")
        for key, recomputed_value in recomputed_evidence[arm].items():
            if gates[arm][key] is not recomputed_value:
                raise ValueError("v3 aggregate subgate was re-signed")
        energy = _energy_ratio_gates(aggregate["aggregate_metrics"], arm)
        if (
            gates[arm]["fold_point_ratios"]
            != aggregate["fold_point_ratios_by_arm"][arm]
            or gates[arm]["energy_strata"] != energy["by_bin"]
        ):
            raise ValueError("v3 aggregate nested gate evidence differs")
        recomputed = bool(all(
            value is True for key, value in gates[arm].items()
            if key not in {
                "aggregate_hard_gate", "fold_point_ratios", "energy_strata"
            }
        ))
        if gates[arm].get("aggregate_hard_gate") is not recomputed:
            raise ValueError("v3 aggregate hard gate was re-signed")
        if recomputed:
            expected.append(arm)
    vae_gate_receipt = aggregate.get("vae_mechanics_gates")
    if (
        type(vae_gate_receipt) is not dict
        or set(vae_gate_receipt) != set(VAE_AGGREGATE_BOOLEAN_GATE_KEYS)
        or any(type(value) is not bool for value in vae_gate_receipt.values())
        or any(
            gates["direct_beta_vae"][key] is not value
            for key, value in vae_gate_receipt.items()
        )
        or aggregate.get("vae_normalized_sample_variance_lcb_floor")
        != VAE_NORMALIZED_SAMPLE_VARIANCE_LCB_FLOOR
    ):
        raise ValueError("v3 aggregate VAE gate receipt differs")
    config = Config(**aggregate.get("config", {}))
    config.validate()
    steps_by_fold = aggregate.get("selected_steps_by_fold_by_arm")
    if (
        type(steps_by_fold) is not dict
        or set(steps_by_fold) != {
            str(fold) for fold in range(OUTER_FOLDS)
        }
        or any(
            type(row) is not dict or set(row) != set(ARMS)
            for row in steps_by_fold.values()
        )
        or any(
            type(step) is not int or not 0 <= step <= config.max_steps
            for row in steps_by_fold.values() for step in row.values()
        )
        or any(
            row["direct_beta_vae"]
            < config.kl_warmup_steps + config.full_beta_plateau_steps
            for row in steps_by_fold.values()
        )
        or any(
            aggregate["vae_fold_mechanics_summary"][str(fold)][
                "selected_step"
            ] != steps_by_fold[str(fold)]["direct_beta_vae"]
            for fold in range(OUTER_FOLDS)
        )
    ):
        raise ValueError("v3 aggregate selected-step evidence differs")
    recomputed_steps = {
        arm: sorted(row[arm] for row in steps_by_fold.values())[
            OUTER_FOLDS // 2
        ]
        for arm in ARMS
    }
    step_preregistration = aggregate.get(
        "full644_refit_step_preregistration"
    )
    if (
        type(step_preregistration) is not dict
        or set(step_preregistration) != {
            "strategy", "steps_by_arm", "frozen_before_refit"
        }
        or step_preregistration["strategy"]
        != "median of exact5 selected fold steps"
        or step_preregistration["steps_by_arm"] != recomputed_steps
        or step_preregistration["frozen_before_refit"] is not True
    ):
        raise ValueError("v3 aggregate refit steps were re-signed")
    if (
        aggregate.get("full644_refit_authorized_arms") != expected
        or aggregate.get("full644_refit_authorized_by_arm")
        != {arm: arm in expected for arm in ARMS}
        or aggregate.get("full644_refit_any_arm_authorized") is not bool(expected)
    ):
        raise ValueError("v3 aggregate refit authorization was re-signed")
    return expected


def prepare_refit(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    aggregate = _load_receipt(
        Path(args.aggregate_receipt), args.expected_aggregate_receipt_sha256,
        AGGREGATE_SCHEMA,
    )
    authorized_arms = _authorized_arms_from_aggregate(aggregate)
    if not authorized_arms:
        raise ValueError("v3 full644 refit is fail-closed because no arm passed OOF")
    if (
        aggregate.get("implementation") != run_binding
        or {key: aggregate.get(key) for key in DEVELOPMENT_FIELDS}
        != DEVELOPMENT_FIELDS
        or aggregate.get("action_representation_qualified") is not False
        or aggregate.get("prior_generation_qualified") is not False
        or aggregate.get("renderer_authorized") is not False
        or aggregate.get("inference_authorized") is not False
        or aggregate.get("vae_necessary") is not None
    ):
        raise ValueError("v3 aggregate implementation/scope differs")
    if args.expected_feature_receipt_sha256 != aggregate.get(
        "feature_receipt_sha256"
    ):
        raise ValueError("v3 refit feature authority differs")
    config = Config(**aggregate["config"])
    config.validate()
    if aggregate.get("config_sha256") != _object_sha(aggregate["config"]):
        raise ValueError("v3 aggregate config digest differs")
    pairs, feature_receipt = authority.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    if len(pairs) != 644:
        raise ValueError("v3 refit requires exact644 originals")
    population = v2._exact644_population_authority(pairs)
    exact_iid_digest = _object_sha([row.iid for row in pairs])
    raw_target_sha = _tensor_sha(torch.stack([
        anchor_action_target(row) for row in pairs
    ]))
    if (
        feature_receipt["receipt_digest"] != aggregate["feature_receipt_digest"]
        or exact_iid_digest != aggregate["exact644_iid_digest"]
        or raw_target_sha != aggregate["exact644_raw_target_sha256"]
        or population != aggregate["exact644_population_authority"]
    ):
        raise ValueError("v3 refit exact644 authority differs from OOF")
    rms = v2._global_rms(pairs)
    values = v2._tensor_rows(pairs, rms)
    frame_pca = v2._fit_frame_pca(values["value"], config.latent_dim)
    clip_pca = v2._fit_clip_pca(values["value"], config.latent_dim)
    pca_initialization = _fit_pca_initialization(values["value"], frame_pca)
    baselines = {
        "frame_pca_rank_l": frame_pca,
        "clip_pca_rank_l": clip_pca,
    }
    baseline_sha = {
        name: _pca_state_sha(state) for name, state in baselines.items()
    }
    step_preregistration = aggregate.get("full644_refit_step_preregistration")
    if (
        type(step_preregistration) is not dict
        or set(step_preregistration) != {"strategy", "steps_by_arm", "frozen_before_refit"}
        or step_preregistration["strategy"]
        != "median of exact5 selected fold steps"
        or step_preregistration["frozen_before_refit"] is not True
    ):
        raise ValueError("v3 refit step preregistration authority differs")
    steps = step_preregistration["steps_by_arm"]
    if type(steps) is not dict or set(steps) != set(ARMS):
        raise ValueError("v3 refit step preregistration differs")
    for arm, step in steps.items():
        if type(step) is not int or not 0 <= step <= config.max_steps:
            raise ValueError("v3 refit step differs")
        if arm == "direct_beta_vae" and arm in authorized_arms and step < (
            config.kl_warmup_steps + config.full_beta_plateau_steps
        ):
            raise ValueError("v3 authorized VAE refit is not full-beta")
    initialization_contract = {
        "frozen_frame_pca_rank": config.latent_dim,
        "zero_initialized_nonlinear_heads": True,
        "step0_posterior_mean_reconstruction_matches_frame_pca_within_abs_3e_5": True,
        "raw_identity_skip": False,
        "frozen_pca_input_output_path": True,
        "decoder_nonlinear_output_orthogonal_to_pca": True,
        "learned_output_increment_entirely_pca_orthogonal": True,
        "residual_latent_is_internal_not_supervision_target": True,
        "posterior_observes_full_target_residual": False,
    }
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    bundle = {
        "schema_version": REFIT_BUNDLE_SCHEMA,
        "config": asdict(config),
        "config_sha256": _object_sha(asdict(config)),
        "implementation": run_binding,
        "aggregate_receipt_sha256": args.expected_aggregate_receipt_sha256,
        "aggregate_receipt_digest": aggregate["receipt_digest"],
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_iid_digest": exact_iid_digest,
        "exact644_raw_target_sha256": raw_target_sha,
        "exact644_population_authority": population,
        "raw_target_definition": RAW_TARGET_DEFINITION,
        "model_coordinate_definition": MODEL_COORDINATE_DEFINITION,
        "global_rms": rms,
        "global_rms_sha256": _tensor_sha(rms),
        "global_rms_fit_only": True,
        "pca_is_model_target": False,
        "full644_originals": values,
        "full644_model_coordinate_sha256": _tensor_sha(values["value"]),
        "baselines": baselines,
        "baseline_sha256": baseline_sha,
        "pca_initialization": pca_initialization,
        "pca_initialization_sha256": _pca_initialization_sha(pca_initialization),
        "initialization_contract": initialization_contract,
        "preregistered_steps_by_arm": steps,
        "authorized_arms": authorized_arms,
        "model_fit_unique_originals": 644,
        "held_rows": 0,
        "derived_rows": 0,
        "development_energy_definition": aggregate[
            "development_energy_definition"
        ],
        "development_energy_bin_edges_raw": aggregate[
            "development_energy_bin_edges_raw"
        ],
    }
    if set(bundle) != REFIT_KEYS:
        raise RuntimeError("v3 internal refit bundle keys differ")
    bundle_path = output / "refit_bundle.pt"
    bundle_sha = _save_torch(bundle_path, bundle)
    receipt: dict[str, Any] = {
        "schema_version": REFIT_PREPARE_SCHEMA,
        "status": "V3_FULL644_REFIT_PREPARED_BURNED_DEVELOPMENT_NO_EVAL",
        **DEVELOPMENT_FIELDS,
        "aggregate_receipt": {
            "path": str(Path(args.aggregate_receipt).resolve(strict=True)),
            "sha256": args.expected_aggregate_receipt_sha256,
            "receipt_digest": aggregate["receipt_digest"],
        },
        "refit_bundle": {
            "path": str(bundle_path.resolve()), "sha256": bundle_sha,
            "size_bytes": bundle_path.stat().st_size,
        },
        "model_fit_unique_originals": 644,
        "held_rows": 0,
        "derived_rows": 0,
        "authorized_arms": authorized_arms,
        "preregistered_steps_by_arm": steps,
        "feature_receipt_sha256": args.expected_feature_receipt_sha256,
        "feature_receipt_digest": feature_receipt["receipt_digest"],
        "exact644_iid_digest": exact_iid_digest,
        "exact644_raw_target_sha256": raw_target_sha,
        "exact644_population_authority": population,
        "global_rms": float(rms),
        "global_rms_sha256": _tensor_sha(rms),
        "full644_model_coordinate_sha256": _tensor_sha(values["value"]),
        "baseline_sha256": baseline_sha,
        "pca_initialization_sha256": bundle["pca_initialization_sha256"],
        "initialization_contract": initialization_contract,
        "development_energy_definition": bundle[
            "development_energy_definition"
        ],
        "development_energy_bin_edges_raw": bundle[
            "development_energy_bin_edges_raw"
        ],
        "action_representation_qualified": False,
        "prior_generation_qualified": False,
        "renderer_authorized": False,
        "inference_authorized": False,
        "vae_necessary": None,
        "implementation": run_binding,
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    receipt_path = output / "prepare_receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    _assert_binding_unchanged(run_binding)
    os.chmod(output, 0o555)
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_sha,
        "bundle_sha256": bundle_sha,
    }


def _load_refit_bundle(
    args: argparse.Namespace,
    prepare: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    expected_implementation: Mapping[str, str],
) -> tuple[dict[str, Any], Config]:
    binding = prepare.get("refit_bundle")
    if type(binding) is not dict or set(binding) != {
        "path", "sha256", "size_bytes"
    }:
        raise ValueError("v3 refit bundle binding differs")
    path = Path(args.refit_bundle).resolve(strict=True)
    expected_sha = _sha(args.expected_refit_bundle_sha256, "refit bundle SHA")
    if (
        Path(binding["path"]).resolve(strict=True) != path
        or binding["sha256"] != expected_sha
    ):
        raise ValueError("v3 refit bundle path/SHA differs")
    bundle = _load_torch(path, expected_sha, binding["size_bytes"])
    if type(bundle) is not dict or set(bundle) != REFIT_KEYS:
        raise ValueError("v3 refit bundle exact-key allowlist differs")
    if bundle["schema_version"] != REFIT_BUNDLE_SCHEMA:
        raise ValueError("v3 refit bundle schema differs")
    config_value = bundle["config"]
    if type(config_value) is not dict or set(config_value) != set(asdict(Config())):
        raise ValueError("v3 refit config keys differ")
    config = Config(**config_value)
    config.validate()
    if (
        bundle["config_sha256"] != _object_sha(config_value)
        or bundle["implementation"] != expected_implementation
        or bundle["implementation"] != prepare["implementation"]
        or aggregate["implementation"] != expected_implementation
    ):
        raise ValueError("v3 refit config/implementation differs")
    aggregate_binding = prepare.get("aggregate_receipt")
    if type(aggregate_binding) is not dict or set(aggregate_binding) != {
        "path", "sha256", "receipt_digest"
    }:
        raise ValueError("v3 refit aggregate binding differs")
    if (
        Path(aggregate_binding["path"]).resolve(strict=True)
        != Path(args.aggregate_receipt).resolve(strict=True)
        or aggregate_binding["sha256"] != args.expected_aggregate_receipt_sha256
        or aggregate_binding["receipt_digest"] != aggregate["receipt_digest"]
        or bundle["aggregate_receipt_sha256"]
        != args.expected_aggregate_receipt_sha256
        or bundle["aggregate_receipt_digest"] != aggregate["receipt_digest"]
    ):
        raise ValueError("v3 refit aggregate authority differs")
    authorized = _authorized_arms_from_aggregate(aggregate)
    if (
        not authorized
        or bundle["authorized_arms"] != authorized
        or prepare["authorized_arms"] != authorized
    ):
        raise ValueError("v3 refit authorized arms differ")
    if (
        bundle["raw_target_definition"] != RAW_TARGET_DEFINITION
        or bundle["model_coordinate_definition"] != MODEL_COORDINATE_DEFINITION
        or bundle["global_rms_fit_only"] is not True
        or bundle["pca_is_model_target"] is not False
        or bundle["model_fit_unique_originals"] != 644
        or bundle["held_rows"] != 0
        or bundle["derived_rows"] != 0
    ):
        raise ValueError("v3 refit target/sample contract differs")
    rms = bundle["global_rms"]
    if (
        type(rms) is not torch.Tensor or rms.dtype != torch.float32
        or tuple(rms.shape) != (1,) or not bool(torch.isfinite(rms).all())
        or float(rms) <= EPS or _tensor_sha(rms) != bundle["global_rms_sha256"]
        or bundle["global_rms_sha256"] != prepare["global_rms_sha256"]
        or not math.isclose(
            float(rms), float(prepare["global_rms"]),
            rel_tol=0.0, abs_tol=1.0e-12,
        )
    ):
        raise ValueError("v3 refit RMS authority differs")
    values = bundle["full644_originals"]
    v2._validate_sequence_rows(values, 644, "v3 full644 originals")
    if (
        _object_sha(values["iids"]) != bundle["exact644_iid_digest"]
        or _tensor_sha(values["value"])
        != bundle["full644_model_coordinate_sha256"]
        or bundle["full644_model_coordinate_sha256"]
        != prepare["full644_model_coordinate_sha256"]
        or not bool(torch.isclose(
            values["value"].square().mean().sqrt(), torch.tensor(1.0),
            atol=3.0e-6, rtol=3.0e-6,
        ))
    ):
        raise ValueError("v3 refit full644 tensor authority differs")
    for key in (
        "feature_receipt_sha256", "feature_receipt_digest",
        "exact644_iid_digest", "exact644_raw_target_sha256",
    ):
        _sha(bundle[key], key)
        if bundle[key] != prepare[key] or bundle[key] != aggregate[key]:
            raise ValueError(f"v3 refit {key} differs")
    expected_population = {
        "unique_original_base_clips": 644,
        "family_count": 28,
        "strict_true": 359,
        "strict_false": 285,
        "derived_rows": 0,
    }
    if (
        bundle["exact644_population_authority"] != expected_population
        or prepare["exact644_population_authority"] != expected_population
        or aggregate["exact644_population_authority"] != expected_population
    ):
        raise ValueError("v3 refit population authority differs")
    baselines = bundle["baselines"]
    if type(baselines) is not dict or set(baselines) != {
        "frame_pca_rank_l", "clip_pca_rank_l"
    }:
        raise ValueError("v3 refit baseline keys differ")
    v2._validate_pca_state(
        baselines["frame_pca_rank_l"], (1, 768),
        (768, config.latent_dim), "v3 refit frame PCA",
    )
    v2._validate_pca_state(
        baselines["clip_pca_rank_l"], (1, 32 * 768),
        (32 * 768, config.latent_dim), "v3 refit clip PCA",
    )
    baseline_sha = {
        name: _pca_state_sha(state) for name, state in baselines.items()
    }
    recomputed_baselines = {
        "frame_pca_rank_l": v2._fit_frame_pca(
            values["value"], config.latent_dim
        ),
        "clip_pca_rank_l": v2._fit_clip_pca(
            values["value"], config.latent_dim
        ),
    }
    if (
        baseline_sha != bundle["baseline_sha256"]
        or baseline_sha != prepare["baseline_sha256"]
        or any(
            not _pca_subspace_equivalent(
                baselines[name], recomputed_baselines[name]
            )
            for name in baselines
        )
    ):
        raise ValueError("v3 refit baseline digest differs")
    pca_initialization = bundle["pca_initialization"]
    _validate_pca_initialization(pca_initialization, config.latent_dim)
    pca_sha = _pca_initialization_sha(pca_initialization)
    recomputed = _fit_pca_initialization(
        values["value"], baselines["frame_pca_rank_l"]
    )
    if (
        pca_sha != bundle["pca_initialization_sha256"]
        or pca_sha != prepare["pca_initialization_sha256"]
        or any(
            not torch.allclose(
                pca_initialization[key], recomputed[key],
                atol=2.0e-6, rtol=2.0e-6,
            )
            for key in pca_initialization
        )
        or bundle["initialization_contract"] != prepare["initialization_contract"]
    ):
        raise ValueError("v3 refit PCA initialization authority differs")
    steps = bundle["preregistered_steps_by_arm"]
    aggregate_steps = aggregate[
        "full644_refit_step_preregistration"
    ]["steps_by_arm"]
    if (
        type(steps) is not dict or set(steps) != set(ARMS)
        or steps != aggregate_steps or steps != prepare["preregistered_steps_by_arm"]
    ):
        raise ValueError("v3 refit preregistered steps differ")
    for arm, step in steps.items():
        if type(step) is not int or not 0 <= step <= config.max_steps:
            raise ValueError("v3 refit step range differs")
        if arm == "direct_beta_vae" and arm in authorized and step < (
            config.kl_warmup_steps + config.full_beta_plateau_steps
        ):
            raise ValueError("v3 refit VAE step is not full-beta")
    if (
        bundle["development_energy_definition"]
        != prepare["development_energy_definition"]
        or bundle["development_energy_definition"]
        != aggregate["development_energy_definition"]
        or bundle["development_energy_bin_edges_raw"]
        != prepare["development_energy_bin_edges_raw"]
        or bundle["development_energy_bin_edges_raw"]
        != aggregate["development_energy_bin_edges_raw"]
    ):
        raise ValueError("v3 refit energy authority differs")
    return bundle, config


def _train_fixed_steps(
    arm: str,
    model: nn.Module,
    value: torch.Tensor,
    steps: int,
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, float]], str]:
    if type(steps) is not int or not 0 <= steps <= config.max_steps:
        raise ValueError("v3 fixed-step count differs")
    model.to(device)
    value = value.to(device)
    history: list[dict[str, float]] = []
    schedule_digest = hashlib.sha256()
    if steps == 0:
        return history, schedule_digest.hexdigest()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 7000)
    for step in range(1, steps + 1):
        indices_cpu = torch.randint(
            len(value), (min(config.batch_size, len(value)),),
            generator=generator,
        )
        schedule_digest.update(indices_cpu.numpy().tobytes(order="C"))
        target = value[indices_cpu.to(device)]
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
    aggregate = _load_receipt(
        Path(args.aggregate_receipt), args.expected_aggregate_receipt_sha256,
        AGGREGATE_SCHEMA,
    )
    bundle, config = _load_refit_bundle(
        args, prepare, aggregate, run_binding
    )
    if args.arm not in ARMS or args.arm not in bundle["authorized_arms"]:
        raise ValueError("v3 refit arm was not authorized by exact5 OOF")
    steps = bundle["preregistered_steps_by_arm"][args.arm]
    if args.arm == "direct_beta_vae" and steps < (
        config.kl_warmup_steps + config.full_beta_plateau_steps
    ):
        raise ValueError("v3 VAE refit lacks full-beta exposure")
    device = v2._device(args.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    pca_initialization = bundle["pca_initialization"]
    model = _make_model(args.arm, config, pca_initialization)
    step0 = _assert_step0_equals_pca(
        model, bundle["full644_originals"]["value"],
        bundle["baselines"]["frame_pca_rank_l"], device,
    )
    history, schedule_digest = _train_fixed_steps(
        args.arm, model, bundle["full644_originals"]["value"],
        steps, config, device,
    )
    _assert_binding_unchanged(run_binding)
    output = _fresh_output(args.output)
    checkpoint = {
        "schema_version": REFIT_CHECKPOINT_SCHEMA,
        "arm": args.arm,
        "config": asdict(config),
        "config_sha256": bundle["config_sha256"],
        "refit_bundle_sha256": args.expected_refit_bundle_sha256,
        "aggregate_receipt_sha256": args.expected_aggregate_receipt_sha256,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "preregistered_steps": steps,
        "executed_steps": steps,
        "executed_minibatch_schedule_sha256": schedule_digest,
        "pca_initialization_sha256": bundle["pca_initialization_sha256"],
        "model_state": _cpu_state(model),
        "implementation": run_binding,
    }
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_sha = _save_torch(checkpoint_path, checkpoint)
    receipt: dict[str, Any] = {
        "schema_version": REFIT_RECEIPT_SCHEMA,
        "status": "V3_FULL644_REFIT_COMPLETE_BURNED_DEVELOPMENT_NO_EVAL",
        **DEVELOPMENT_FIELDS,
        "arm": args.arm,
        "arm_semantics": (
            "PCA-conditioned residual beta-VAE with frozen PCA trunk"
            if args.arm == "direct_beta_vae"
            else "PCA-conditioned deterministic residual decoder with frozen PCA trunk"
        ),
        "model_fit_unique_originals": 644,
        "held_rows": 0,
        "derived_rows": 0,
        "preregistered_steps": steps,
        "executed_steps": steps,
        "fixed_step_refit_without_early_stop": True,
        "executed_minibatch_schedule_sha256": schedule_digest,
        "step0_pca_equivalence": step0,
        "full_beta_exposure_steps": max(0, steps - config.kl_warmup_steps),
        "target": {
            "raw": RAW_TARGET_DEFINITION,
            "model_coordinate": MODEL_COORDINATE_DEFINITION,
            "source_subtracted": False,
            "pca_target_used": False,
            "direct_full768_reconstruction": True,
        },
        "training_history": history,
        "refit_bundle_sha256": args.expected_refit_bundle_sha256,
        "aggregate_receipt_sha256": args.expected_aggregate_receipt_sha256,
        "prepare_receipt_sha256": args.expected_prepare_receipt_sha256,
        "feature_receipt_sha256": bundle["feature_receipt_sha256"],
        "feature_receipt_digest": bundle["feature_receipt_digest"],
        "exact644_iid_digest": bundle["exact644_iid_digest"],
        "exact644_raw_target_sha256": bundle["exact644_raw_target_sha256"],
        "pca_initialization_sha256": bundle["pca_initialization_sha256"],
        "action_representation_qualified": False,
        "source_identity_preservation_tested": False,
        "video_editing_tested": False,
        "prior_generation_qualified": False,
        "renderer_authorized": False,
        "inference_authorized": False,
        "vae_necessary": None,
        "vae_necessity_status": "UNDETERMINED_SINGLE_EXECUTION_PER_IID",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
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
        "checkpoint_sha256": checkpoint_sha,
    }


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = Config()
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--latent-dim", type=int, default=defaults.latent_dim)
    parser.add_argument(
        "--correction-hidden-dim", type=int,
        default=defaults.correction_hidden_dim,
    )
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate
    )
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--beta-kl", type=float, default=defaults.beta_kl)
    parser.add_argument(
        "--kl-zero-steps", type=int, default=defaults.kl_zero_steps
    )
    parser.add_argument(
        "--kl-warmup-steps", type=int, default=defaults.kl_warmup_steps
    )
    parser.add_argument(
        "--full-beta-plateau-steps", type=int,
        default=defaults.full_beta_plateau_steps,
    )
    parser.add_argument(
        "--correction-penalty", type=float,
        default=defaults.correction_penalty,
    )
    parser.add_argument(
        "--eval-interval", type=int, default=defaults.eval_interval
    )
    parser.add_argument(
        "--selection-relative-delta", type=float,
        default=defaults.selection_relative_delta,
    )
    parser.add_argument(
        "--initial-logvar", type=float, default=defaults.initial_logvar
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PCA-safe anchor-only residual AE/beta-VAE burned-development OOF"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-fold")
    prepare.add_argument("--feature-root", required=True)
    prepare.add_argument("--expected-feature-receipt-sha256", required=True)
    prepare.add_argument(
        "--fold-index", required=True, type=int, choices=range(OUTER_FOLDS)
    )
    prepare.add_argument("--output", required=True)
    _add_config_arguments(prepare)
    prepare.set_defaults(handler=prepare_fold)

    train = subparsers.add_parser("train-fold")
    train.add_argument("--prepare-receipt", required=True)
    train.add_argument("--expected-prepare-receipt-sha256", required=True)
    train.add_argument("--train-bundle", required=True)
    train.add_argument("--expected-train-bundle-sha256", required=True)
    train.add_argument(
        "--fold-index", required=True, type=int, choices=range(OUTER_FOLDS)
    )
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
    compare.add_argument(
        "--expected-exploratory-oof-bundle-sha256", required=True
    )
    compare.add_argument(
        "--fold-index", required=True, type=int, choices=range(OUTER_FOLDS)
    )
    compare.add_argument("--ae-receipt", required=True)
    compare.add_argument("--expected-ae-receipt-sha256", required=True)
    compare.add_argument("--vae-receipt", required=True)
    compare.add_argument("--expected-vae-receipt-sha256", required=True)
    compare.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=compare_fold)

    aggregate = subparsers.add_parser("aggregate-oof")
    aggregate.add_argument(
        "--fold-receipt", required=True, nargs=OUTER_FOLDS
    )
    aggregate.add_argument(
        "--expected-fold-receipt-sha256", required=True, nargs=OUTER_FOLDS
    )
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=aggregate_oof)

    refit_prepare = subparsers.add_parser("prepare-refit")
    refit_prepare.add_argument("--aggregate-receipt", required=True)
    refit_prepare.add_argument(
        "--expected-aggregate-receipt-sha256", required=True
    )
    refit_prepare.add_argument("--feature-root", required=True)
    refit_prepare.add_argument(
        "--expected-feature-receipt-sha256", required=True
    )
    refit_prepare.add_argument("--output", required=True)
    refit_prepare.set_defaults(handler=prepare_refit)

    refit_train = subparsers.add_parser("train-refit")
    refit_train.add_argument("--prepare-receipt", required=True)
    refit_train.add_argument(
        "--expected-prepare-receipt-sha256", required=True
    )
    refit_train.add_argument("--aggregate-receipt", required=True)
    refit_train.add_argument(
        "--expected-aggregate-receipt-sha256", required=True
    )
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
