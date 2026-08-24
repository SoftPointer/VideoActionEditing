#!/usr/bin/env python3
"""Burned-development exact5 nonlinear temporal codec over frozen V-JEPA2.

The target is only the temporally centered ordered contextual V-JEPA2 sequence
``C(view)`` with shape ``[32,1024]``.  Every fold starts at a fit-only fixed
Tucker-B384 map.  A zero-initialized encoder residual may change its sole
``[4,96]`` code and a zero-initialized decoder residual may reconstruct from
that code; the decoder has no access to the input and there are no skips.

Training reads model-fit originals only.  It never reads any of the four
derived evaluation views.  Training runs its full fixed budget.  A checkpoint
is chosen from fixed steps, including exact analytic step 0, by inner-
validation original reconstruction only.  OOF model values are first computed
after selection; no OOF tensor enters the optimizer or checkpoint selection.

This remains a development codec diagnostic.  Even a passing decoded-space
gate cannot qualify the latent metric (its gauge is not fixed), an action
representation, generation, rendering, inference, or video editing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
import torch.nn.functional as F

from methods.bernini_action_editing import semantic_anchor_vjepa2_analytic_frontier_v4c as v4c


v4a = v4c.v4a
features = v4c.features
SCHEMA = "semantic-anchor-vjepa2-nonlinear-temporal-codec-exact5-receipt-v4d"
STATUS = "V4D_VJEPA2_EXACT5_NONLINEAR_TEMPORAL_CODEC_COMPLETE_BURNED_DEVELOPMENT"
SEED = 20260819
TIME_STEPS = 32
FEATURE_DIM = 1024
FULL_NUMEL = TIME_STEPS * FEATURE_DIM
OUTER_FOLDS = 5
INNER_FOLDS = 5
CODE_TIME = 4
CODE_CHANNELS = 96
CODE_NUMEL = CODE_TIME * CODE_CHANNELS
MAX_TRAINABLE_PARAMETERS = 150000
EXACT_TRAINABLE_PARAMETERS = 143360
BASELINE_NAME = "tucker_b0384_t04_r096"
INNER_SPLIT_NAMESPACE = "v4d-vjepa2-inner-family-sha256-round-robin-v1"
V4A_RECEIPT_FILE_SHA256 = "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2"
V4A_RECEIPT_SELFDIGEST = "f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86"
V4A_IMPLEMENTATION_SHA256 = "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973"
V4C_IMPLEMENTATION_SHA256 = "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef"
EXTRACTOR_IMPLEMENTATION_SHA256 = "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc"
V4C_FEATURE_RECEIPT_SHA256 = "895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a"
V4C_FRONTIER_RECEIPT_SHA256 = "8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9"
V4C_FRONTIER_RECEIPT_DIGEST = "376a98dc74e30ab80a277c8866028677d56ba894073d195612a0edb0bbd74f17"
FROZEN_OOF_COUNTS = (131, 127, 128, 129, 129)
FROZEN_INNER_SPLITS: tuple[dict[str, Any], ...] = (
    {
        "outer_fold": 0,
        "counts": {"model_fit": 400, "inner_validation": 113,
                   "exploratory_oof": 131},
        "outer_train_family_count": 28,
        "model_fit_family_count": 28,
        "inner_validation_family_count": 27,
        "singleton_families": ["exit"],
        "inner_assignment_digest":
            "b2e3143700ecf10ff54416395267b7cc3c90f33c7acedd402ede8062f374635a",
        "model_fit_iid_digest":
            "40ce2072cdbc2cded22bd99cb916897011b066fba48d7ebc389bef8efb67dd18",
        "inner_validation_iid_digest":
            "b98b67342049c45898c055546ee9f49bde70c6169e49db7255e5d5f0d03c02aa",
        "partition_iid_digest":
            "4826c85125572c150284d2bfa593848e380e2d907152701b3df9a60148f927d7",
    },
    {
        "outer_fold": 1,
        "counts": {"model_fit": 402, "inner_validation": 115,
                   "exploratory_oof": 127},
        "outer_train_family_count": 28,
        "model_fit_family_count": 28,
        "inner_validation_family_count": 26,
        "singleton_families": ["climb", "exit"],
        "inner_assignment_digest":
            "d9431d202451a3ce99d3b7be67806918dc5ed812259c0b06a4deab0ebf7f2a6f",
        "model_fit_iid_digest":
            "4e45aac2efcb9a2327586860661b1c26be2cd643766307e768c11ba948ba2cd0",
        "inner_validation_iid_digest":
            "e306b191378bf5eff9b3734080c3be036000c03921c3c03ce127ae5733b5e873",
        "partition_iid_digest":
            "ee126c414fdb2e95d4b56f8876e590fbdfbd54bf31a7646a6c88fd3f7f9c8bf4",
    },
    {
        "outer_fold": 2,
        "counts": {"model_fit": 401, "inner_validation": 115,
                   "exploratory_oof": 128},
        "outer_train_family_count": 28,
        "model_fit_family_count": 28,
        "inner_validation_family_count": 26,
        "singleton_families": ["climb", "exit"],
        "inner_assignment_digest":
            "10302330a6a4feddece521d12b3e86efac92bb8e5eb5151b631113aeda069f5c",
        "model_fit_iid_digest":
            "8321d846bb3f98405251580f1088cd6071f3820fcdf0bbe9ac858dc1a2aa7b78",
        "inner_validation_iid_digest":
            "110db33e61e02cccf95424937fa718ab21dee688934b345425fda2bf7fcc5102",
        "partition_iid_digest":
            "608f19e7e05db3202513a23ad5be27c7d78a4a5d9f828abda7da7c9dca77a65a",
    },
    {
        "outer_fold": 3,
        "counts": {"model_fit": 403, "inner_validation": 112,
                   "exploratory_oof": 129},
        "outer_train_family_count": 27,
        "model_fit_family_count": 27,
        "inner_validation_family_count": 27,
        "singleton_families": [],
        "inner_assignment_digest":
            "9f5ee3fce90bca584af36b761ed7a9a2d975d8c10270b1ee18adc3b70b42692f",
        "model_fit_iid_digest":
            "90da16b4d97de006e16fc522d77061b79f68aefaea3599a6a2a4a28659988353",
        "inner_validation_iid_digest":
            "5517ded71818723d464e99ca380bc6cfb34a34f576d2d564235a1d7f3a5c10e4",
        "partition_iid_digest":
            "4235825a4031a247d2bfbba0552596bf7587142be324f2081be77f78bc269997",
    },
    {
        "outer_fold": 4,
        "counts": {"model_fit": 403, "inner_validation": 112,
                   "exploratory_oof": 129},
        "outer_train_family_count": 28,
        "model_fit_family_count": 28,
        "inner_validation_family_count": 27,
        "singleton_families": ["exit"],
        "inner_assignment_digest":
            "6b2502ef34eaf4bd81e1abcda313accb13306adb1b1f97f38aea84e97bf1760a",
        "model_fit_iid_digest":
            "bcd16de76199767e77a889d44b504b96db1703b980facf08b26b50b09730283c",
        "inner_validation_iid_digest":
            "45f7f48fd625941494f88860acc2ed9b81c7035f00950e8304070f7faf919a32",
        "partition_iid_digest":
            "7c7e93a17afec29032b8b7e6948184a43796bb1e8fe680c8591266f3cedab9e8",
    },
)
NEGATIVES = v4c.NEGATIVES
EVAL_VIEWS = tuple(features.VIEW_NAMES)


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    max_steps: int = 1200
    batch_size: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-6
    checkpoint_steps: tuple[int, ...] = (0, 300, 600, 900, 1200)
    bootstrap_draws: int = 10000
    bootstrap_alpha: float = 0.05
    teacher_retention: float = 0.8
    recon_ratio_limit: float = 1.05

    def validate(self) -> None:
        if self != Config():
            raise ValueError("v4-D configuration is immutable")
        if self.checkpoint_steps[0] != 0 or self.checkpoint_steps[-1] != self.max_steps:
            raise ValueError("fixed checkpoint schedule does not span full budget")
        if CODE_NUMEL != 384:
            raise ValueError("actual code payload differs")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _object_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous().clone()
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
    digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    return features.file_sha256(path)


def _binding() -> dict[str, str]:
    paths = {
        "implementation": Path(__file__).resolve(strict=True),
        "v4c_implementation": Path(v4c.__file__).resolve(strict=True),
        "extractor_implementation": Path(features.__file__).resolve(strict=True),
        "v4a_implementation": Path(v4a.__file__).resolve(strict=True),
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        result[f"{name}_path"] = str(path)
        result[f"{name}_sha256"] = _file_sha(path)
    if (
        result["v4c_implementation_sha256"] != V4C_IMPLEMENTATION_SHA256
        or result["extractor_implementation_sha256"]
        != EXTRACTOR_IMPLEMENTATION_SHA256
        or result["v4a_implementation_sha256"] != V4A_IMPLEMENTATION_SHA256
    ):
        raise RuntimeError("v4-D frozen implementation dependency differs")
    return result


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("implementation or authority changed during execution")


def _write_json_create_only(path: Path, value: Any) -> str:
    return v4c._write_json_create_only(path, value)["sha256"]


@dataclass(frozen=True)
class TuckerFit:
    frame_mean: torch.Tensor
    temporal_basis: torch.Tensor
    content_basis: torch.Tensor
    fit_iid_digest: str
    fit_input_sha256: str
    diagnostics: Mapping[str, Any]


def _fit_tucker_b384(rows: Sequence[v4c.Record]) -> TuckerFit:
    """Fit the fixed step-0 comparator from model-fit originals only."""

    iids = [row.iid for row in rows]
    if not rows or len(set(iids)) != len(iids):
        raise ValueError("fit-only Tucker IID closure differs")
    values = torch.stack([
        v4c.canonical_action(row.views["original"]) for row in rows
    ]).to(dtype=torch.float32, device="cpu")
    if tuple(values.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("fit-only Tucker tensor geometry differs")
    tokens = values.reshape(-1, FEATURE_DIM)
    frame_mean = tokens.mean(dim=0, keepdim=True)
    feature_covariance = (tokens - frame_mean).T @ (tokens - frame_mean)
    content_basis = v4a._top_eigenbasis(feature_covariance, CODE_CHANNELS)
    temporal_covariance = torch.einsum("ntd,nsd->ts", values, values)
    temporal_basis = v4a._top_eigenbasis(temporal_covariance, CODE_TIME)
    if (
        tuple(frame_mean.shape) != (1, FEATURE_DIM)
        or tuple(temporal_basis.shape) != (TIME_STEPS, CODE_TIME)
        or tuple(content_basis.shape) != (FEATURE_DIM, CODE_CHANNELS)
        or not bool(torch.isfinite(frame_mean).all())
        or not bool(torch.isfinite(temporal_basis).all())
        or not bool(torch.isfinite(content_basis).all())
    ):
        raise RuntimeError("fit-only Tucker basis closure differs")
    diagnostics = {
        "fit_original_count": len(rows),
        "fit_original_only": True,
        "fit_derived_view_count": 0,
        "fit_family_or_transform_labels": False,
        "frame_mean_shape": list(frame_mean.shape),
        "frame_mean_sha256": _tensor_sha(frame_mean),
        "temporal_basis_shape": list(temporal_basis.shape),
        "temporal_basis_sha256": _tensor_sha(temporal_basis),
        "temporal_basis_max_orthogonality_error": v4a._orthogonality_error(
            temporal_basis
        ),
        "content_basis_shape": list(content_basis.shape),
        "content_basis_sha256": _tensor_sha(content_basis),
        "content_basis_max_orthogonality_error": v4a._orthogonality_error(
            content_basis
        ),
        "temporal_rank": CODE_TIME,
        "content_rank": CODE_CHANNELS,
        "actual_code_numel": CODE_NUMEL,
        "variance_whitening": False,
    }
    return TuckerFit(
        frame_mean=frame_mean.contiguous(),
        temporal_basis=temporal_basis.contiguous(),
        content_basis=content_basis.contiguous(),
        fit_iid_digest=_object_sha(iids),
        fit_input_sha256=_tensor_sha(values),
        diagnostics=diagnostics,
    )


def _inner_hash(
    *, seed: int, outer_fold: int, family: str, iid: str,
) -> str:
    preimage = _canonical_json([
        INNER_SPLIT_NAMESPACE, seed, outer_fold, family, iid,
    ])
    return hashlib.sha256(preimage).hexdigest()


def _split_fold(
    records: Sequence[v4c.Record], outer_assignment: Mapping[str, int],
    outer_fold: int, config: Config,
) -> tuple[dict[str, list[v4c.Record]], dict[str, Any]]:
    """Freeze inner fold 0 without reading any V-JEPA tensor value."""

    if (
        len(records) != 644
        or len({row.iid for row in records}) != 644
        or set(outer_assignment) != {row.iid for row in records}
        or not 0 <= outer_fold < OUTER_FOLDS
    ):
        raise ValueError("v4-D outer population closure differs")
    exploratory_oof = [
        row for row in records if outer_assignment[row.iid] == outer_fold
    ]
    outer_train = [
        row for row in records if outer_assignment[row.iid] != outer_fold
    ]
    by_family: dict[str, list[v4c.Record]] = {}
    for row in outer_train:
        by_family.setdefault(row.family, []).append(row)
    if not by_family or len(by_family) > 28:
        raise ValueError("v4-D outer-train family closure differs")
    inner_assignment: dict[str, int] = {}
    singleton_families: list[str] = []
    for family in sorted(by_family):
        ordered = sorted(
            by_family[family],
            key=lambda row: (
                _inner_hash(
                    seed=config.seed, outer_fold=outer_fold,
                    family=family, iid=row.iid,
                ),
                row.iid,
            ),
        )
        if len(ordered) == 1:
            # Never remove the only outer-train member of a family from model
            # fit.  The override is metadata-only and fixed before any tensor
            # value is read.
            inner_assignment[ordered[0].iid] = 1
            singleton_families.append(family)
        else:
            for rank, row in enumerate(ordered):
                inner_assignment[row.iid] = rank % INNER_FOLDS
    inner_validation = [
        row for row in outer_train if inner_assignment[row.iid] == 0
    ]
    model_fit = [
        row for row in outer_train if inner_assignment[row.iid] != 0
    ]
    groups = {
        "model_fit": model_fit,
        "inner_validation": inner_validation,
        "exploratory_oof": exploratory_oof,
    }
    ids = {name: [row.iid for row in group] for name, group in groups.items()}
    closure = [iid for group in ids.values() for iid in group]
    if (
        len(exploratory_oof) != FROZEN_OOF_COUNTS[outer_fold]
        or len(closure) != 644
        or len(set(closure)) != 644
        or not model_fit
        or not inner_validation
        or len({row.family for row in model_fit}) != len(by_family)
        or not {row.family for row in inner_validation}.issubset(by_family)
    ):
        raise ValueError("v4-D model-fit/inner/OOF split closure differs")
    split = {
        "outer_fold": outer_fold,
        "outer_assignment_digest": _object_sha(dict(outer_assignment)),
        "outer_oof_count": len(exploratory_oof),
        "outer_oof_iid_digest": _object_sha(ids["exploratory_oof"]),
        "inner_algorithm": (
            "within each family, sort by SHA256(canonical JSON ASCII array "
            "[namespace,seed,outer_fold,family,iid]); rank modulo 5; "
            "inner fold 0 is validation; singleton families are forced to "
            "inner fold 1/model-fit"
        ),
        "inner_namespace": INNER_SPLIT_NAMESPACE,
        "inner_seed": config.seed,
        "inner_fold_count": INNER_FOLDS,
        "inner_validation_fold": 0,
        "inner_assignment_digest": _object_sha(inner_assignment),
        "outer_train_family_count": len(by_family),
        "singleton_family_count": len(singleton_families),
        "singleton_families": singleton_families,
        "singleton_forced_model_fit": True,
        "model_fit_family_count": len({row.family for row in model_fit}),
        "inner_validation_family_count": len({row.family for row in inner_validation}),
        "counts": {name: len(group) for name, group in groups.items()},
        "model_fit_iid_digest": _object_sha(ids["model_fit"]),
        "inner_validation_iid_digest": _object_sha(ids["inner_validation"]),
        "partition_iid_digest": _object_sha(ids),
        "split_tensor_values_read": False,
        "split_vjepa_energy_used": False,
        "split_strict_label_used": False,
        "split_family_metadata_used": True,
    }
    if split["outer_assignment_digest"] == v4c.OUTER_ASSIGNMENT_DIGEST:
        literal = {
            key: split[key] for key in FROZEN_INNER_SPLITS[outer_fold]
        }
        if literal != FROZEN_INNER_SPLITS[outer_fold]:
            raise ValueError("v4-D frozen real exact644 inner split differs")
        split["frozen_real_exact644_literal_match"] = True
    else:
        split["frozen_real_exact644_literal_match"] = False
    return groups, split


class TuckerInitializedVJepaTemporalCodec(nn.Module):
    """One 384-scalar bottleneck; decoder input is exactly and only ``z``."""

    def __init__(self, fitted: TuckerFit, fit_only_rms: torch.Tensor) -> None:
        super().__init__()
        if (tuple(fit_only_rms.shape) != (1,)
                or not bool(torch.isfinite(fit_only_rms).all())
                or float(fit_only_rms) <= 0.0):
            raise ValueError("fit-only global RMS geometry differs")
        if (tuple(fitted.frame_mean.shape) != (1, FEATURE_DIM)
                or tuple(fitted.temporal_basis.shape) != (TIME_STEPS, CODE_TIME)
                or fitted.content_basis.ndim != 2
                or fitted.content_basis.shape[0] != FEATURE_DIM
                or fitted.content_basis.shape[1] < CODE_CHANNELS):
            raise ValueError("pinned Tucker basis geometry differs")
        self.register_buffer("fit_only_rms", fit_only_rms.detach().reshape(1))
        self.register_buffer("frame_mean", fitted.frame_mean.detach().reshape(1, FEATURE_DIM))
        self.register_buffer("temporal_basis", fitted.temporal_basis.detach())
        self.register_buffer("content_basis", fitted.content_basis[:, :CODE_CHANNELS].detach())
        self.encoder_delta = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, 40, 1), nn.GELU(),
            nn.Conv1d(40, 56, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(56, CODE_CHANNELS, 4, stride=4, padding=0),
        )
        self.decoder_residual = nn.Sequential(
            nn.ConvTranspose1d(CODE_CHANNELS, 56, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(56, 40, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(40, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(32, FEATURE_DIM, 1),
        )
        nn.init.zeros_(self.encoder_delta[-1].weight)
        nn.init.zeros_(self.encoder_delta[-1].bias)
        nn.init.zeros_(self.decoder_residual[-1].weight)
        nn.init.zeros_(self.decoder_residual[-1].bias)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != EXACT_TRAINABLE_PARAMETERS or count >= MAX_TRAINABLE_PARAMETERS:
            raise RuntimeError("v4-D codec parameter closure differs")

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        if (
            value.ndim != 3 or value.shape[0] == 0
            or tuple(value.shape[1:]) != (TIME_STEPS, FEATURE_DIM)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("encoder input geometry differs")
        # Every caller must supply the upstream C(view); do not apply C twice,
        # because step 0 must be the exact fit-only fixed Tucker encoder.
        if float(value.detach().mean(dim=1).abs().max().cpu()) > 1.0e-5:
            raise ValueError("encoder input is not upstream-temporally-centered C(view)")
        centered = value - self.frame_mean
        analytic = torch.einsum("tk,btd,dc->bkc", self.temporal_basis,
                                centered, self.content_basis)
        delta = self.encoder_delta((value / self.fit_only_rms).transpose(1, 2))
        delta = delta.transpose(1, 2) * self.fit_only_rms
        code = analytic + delta
        if (
            tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS)
            or code[0].numel() != 384 or code.dtype != torch.float32
            or not code.is_contiguous() or not bool(torch.isfinite(code).all())
        ):
            raise RuntimeError("actual code is not [4,96]=384")
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if (
            code.ndim != 3 or code.shape[0] == 0
            or tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS)
            or code.dtype != torch.float32 or not code.is_contiguous()
            or not bool(torch.isfinite(code).all())
        ):
            raise ValueError("decoder input must be the sole [4,96] code")
        analytic = self.frame_mean + torch.einsum(
            "tk,bkc,dc->btd", self.temporal_basis, code, self.content_basis
        )
        residual = self.decoder_residual(
            (code / self.fit_only_rms).transpose(1, 2)
        ).transpose(1, 2)
        if tuple(residual.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
            raise RuntimeError("decoder residual geometry differs")
        output = analytic + residual * self.fit_only_rms
        result = (output - output.mean(dim=1, keepdim=True)).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("decoder output is non-finite")
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(value))


TuckerInitializedTemporalConvAE = TuckerInitializedVJepaTemporalCodec


def _fit_only_global_rms(rows: Sequence[v4c.Record], device: torch.device) -> torch.Tensor:
    values = torch.stack([v4c.canonical_action(row.views["original"]) for row in rows])
    rms = values.square().mean().sqrt().reshape(1).to(device)
    if not bool(torch.isfinite(rms).all()) or float(rms) <= 1.0e-8:
        raise ValueError("fit-only global RMS differs")
    return rms


def _canonical_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or tuple(value.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("canonical batch geometry differs")
    return value - value.mean(dim=1, keepdim=True)


def _raw_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-2:] != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("raw reconstruction geometry differs")
    return (prediction - target).square().mean()


def _fixed_training_loss(
    original_prediction: torch.Tensor, original_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Only model-fit original reconstruction mechanics are optimized."""

    raw = F.smooth_l1_loss(original_prediction, original_target, beta=0.1)
    deltas: dict[int, torch.Tensor] = {}
    for stride in (1, 2, 4):
        deltas[stride] = F.smooth_l1_loss(
            original_prediction[:, stride:] - original_prediction[:, :-stride],
            original_target[:, stride:] - original_target[:, :-stride], beta=0.1,
        )
    terminal = F.smooth_l1_loss(
        original_prediction[:, -1] - original_prediction[:, 0],
        original_target[:, -1] - original_target[:, 0], beta=0.1,
    )
    total = raw + 0.20 * sum(deltas.values()) + 0.20 * terminal
    values = {
        "raw_feature": float(raw.detach().cpu()),
        "signed_delta_stride1": float(deltas[1].detach().cpu()),
        "signed_delta_stride2": float(deltas[2].detach().cpu()),
        "signed_delta_stride4": float(deltas[4].detach().cpu()),
        "terminal_displacement": float(terminal.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, values


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("checkpoint contains non-finite state")
    return state


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return _object_sha({name: _tensor_sha(state[name]) for name in sorted(state)})


@torch.no_grad()
def _validation_original_mse(
    model: TuckerInitializedTemporalConvAE, values: torch.Tensor, batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(values), batch_size):
        target = values[start:start + batch_size]
        prediction = model(target)
        per_row = (prediction - target).square().mean(dim=(1, 2))
        total += float(per_row.double().sum().cpu())
        count += len(target)
    if count != len(values) or count == 0:
        raise ValueError("inner-validation closure differs")
    result = total / count
    if not math.isfinite(result):
        raise RuntimeError("inner-validation score is non-finite")
    return result


@torch.no_grad()
def _step0_equivalence(
    model: TuckerInitializedTemporalConvAE, values: torch.Tensor,
    fitted: TuckerFit, batch_size: int,
) -> dict[str, Any]:
    max_abs = 0.0
    squared_sum = 0.0
    numel = 0
    model_outputs: list[torch.Tensor] = []
    reference_outputs: list[torch.Tensor] = []
    model_codes: list[torch.Tensor] = []
    reference_codes: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        actual_code = model.encode(batch)
        reference_code = _analytic_tucker_encode(batch, fitted)
        if not torch.equal(actual_code, reference_code):
            raise RuntimeError("step-0 codec code is not exact Tucker-B384")
        actual = model.decode(actual_code)
        reference = _analytic_tucker_decode_from_code(reference_code, fitted)
        difference = actual - reference
        max_abs = max(max_abs, float(difference.abs().max().cpu()))
        squared_sum += float(difference.double().square().sum().cpu())
        numel += difference.numel()
        model_outputs.append(actual.cpu())
        reference_outputs.append(reference.cpu())
        model_codes.append(actual_code.cpu())
        reference_codes.append(reference_code.cpu())
    mse = squared_sum / numel
    actual_all = torch.cat(model_outputs)
    reference_all = torch.cat(reference_outputs)
    actual_code_all = torch.cat(model_codes)
    reference_code_all = torch.cat(reference_codes)
    bit_exact = torch.equal(actual_all, reference_all)
    code_bit_exact = torch.equal(actual_code_all, reference_code_all)
    if not bit_exact or not code_bit_exact or max_abs != 0.0 or mse != 0.0:
        raise RuntimeError("step-0 ConvAE is not the analytic Tucker comparator")
    return {
        "original_count": len(values),
        "input_sha256": _tensor_sha(values),
        "model_output_sha256": _tensor_sha(actual_all),
        "analytic_output_sha256": _tensor_sha(reference_all),
        "bit_exact": bit_exact,
        "model_code_sha256": _tensor_sha(actual_code_all),
        "analytic_code_sha256": _tensor_sha(reference_code_all),
        "code_bit_exact": code_bit_exact,
        "max_abs_difference": max_abs,
        "mean_squared_difference": mse,
        "required_max_abs_difference": 0.0,
        "required_mean_squared_difference": 0.0,
        "code_shape": [CODE_TIME, CODE_CHANNELS],
        "actual_code_numel": CODE_NUMEL,
    }


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _train_fold_model(
    model_fit: Sequence[v4c.Record],
    inner_validation: Sequence[v4c.Record],
    fitted: TuckerFit, config: Config, fold_index: int,
    device: torch.device,
) -> tuple[TuckerInitializedTemporalConvAE, int, dict[str, Any]]:
    """Run every fixed step; select only by held inner original reconstruction."""

    seed = config.seed + 10000 + fold_index
    _seed_everything(seed, device)
    fit_original = torch.stack([
        v4c.canonical_action(row.views["original"]) for row in model_fit
    ]).to(device)
    validation_original = torch.stack([
        v4c.canonical_action(row.views["original"]) for row in inner_validation
    ]).to(device)
    rms = _fit_only_global_rms(model_fit, device)
    model = TuckerInitializedTemporalConvAE(fitted, rms).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (
        trainable_parameters != EXACT_TRAINABLE_PARAMETERS
        or trainable_parameters >= MAX_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("trainable parameter gate differs")
    fit_step0 = _step0_equivalence(model, fit_original, fitted, config.batch_size)
    validation_step0 = _step0_equivalence(
        model, validation_original, fitted, config.batch_size
    )

    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {0: _state_to_cpu(model)}
    checkpoint_scores: dict[int, float] = {
        0: _validation_original_mse(model, validation_original, config.batch_size)
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    minibatch_schedule = torch.randint(
        len(fit_original), (config.max_steps, config.batch_size),
        generator=batch_generator,
    )
    last_components: dict[str, float] | None = None
    model.train()
    for step in range(1, config.max_steps + 1):
        indices = minibatch_schedule[step - 1].to(device)
        original_target = fit_original.index_select(0, indices)
        original_prediction = model(original_target)
        loss, last_components = _fixed_training_loss(original_prediction, original_target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("training loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError("training gradient is non-finite")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise RuntimeError("trained parameter is non-finite")
        if step in config.checkpoint_steps:
            checkpoint_states[step] = _state_to_cpu(model)
            checkpoint_scores[step] = _validation_original_mse(
                model, validation_original, config.batch_size
            )
            model.train()
    if set(checkpoint_states) != set(config.checkpoint_steps) or last_components is None:
        raise RuntimeError("full fixed training/checkpoint budget did not close")
    selected_step = min(config.checkpoint_steps, key=lambda step: (checkpoint_scores[step], step))
    model.load_state_dict(checkpoint_states[selected_step], strict=True)
    model.to(device).eval()
    audit = {
        "fold_seed": seed,
        "full_budget_steps_executed": config.max_steps,
        "early_stopped": False,
        "checkpoint_steps": list(config.checkpoint_steps),
        "inner_validation_original_mse_by_step": {
            str(step): checkpoint_scores[step] for step in config.checkpoint_steps
        },
        "selection_rule": "minimum inner-validation original raw MSE; ties choose smaller fixed step",
        "selected_step": selected_step,
        "selected_state_sha256": _state_sha(checkpoint_states[selected_step]),
        "step0_state_sha256": _state_sha(checkpoint_states[0]),
        "final_step_state_sha256": _state_sha(checkpoint_states[config.max_steps]),
        "last_training_loss_components": last_components,
        "trainable_parameter_count": trainable_parameters,
        "trainable_parameter_limit_exclusive": MAX_TRAINABLE_PARAMETERS,
        "fit_only_global_rms": float(rms.detach().cpu()),
        "fit_only_global_rms_sha256": _tensor_sha(rms),
        "step0_model_fit_equivalence": fit_step0,
        "step0_inner_validation_equivalence": validation_step0,
        "zero_initialized_encoder_and_decoder_residual_final_layers": True,
        "minibatch_schedule_shape": list(minibatch_schedule.shape),
        "minibatch_schedule_sha256": _tensor_sha(minibatch_schedule),
        "minibatch_schedule_definition": "all fixed model-fit row indices in executed step order",
        "model_fit_original_count": len(model_fit),
        "model_fit_ordered_iids": [row.iid for row in model_fit],
        "model_fit_iid_digest": _object_sha([row.iid for row in model_fit]),
        "model_fit_original_tensor_sha256": _tensor_sha(fit_original),
        "model_fit_derived_training_view_rows": 0,
        "derived_training_views_per_original": 0,
        "derived_training_view_independent_sample_count": 0,
        "inner_validation_original_count": len(inner_validation),
        "inner_validation_iid_digest": _object_sha([row.iid for row in inner_validation]),
        "inner_validation_derived_views_used": 0,
        "oof_tensors_supplied_to_optimizer_or_checkpoint_selection": False,
        "source_rows_used": 0,
        "negative_rows_used": 0,
        "evaluation_positive_rows_used": 0,
        "family_or_transform_labels_entered_loss_or_model_input": False,
        "family_metadata_used_for_inner_split": True,
        "transform_metadata_used_for_inner_split": False,
    }
    return model, selected_step, audit


def _analytic_tucker_encode(value: torch.Tensor, fitted: TuckerFit) -> torch.Tensor:
    batched = value.unsqueeze(0) if value.ndim == 2 else value
    frame_mean = fitted.frame_mean.to(batched.device)
    temporal_basis = fitted.temporal_basis.to(batched.device)
    content_basis = fitted.content_basis[:, :CODE_CHANNELS].to(batched.device)
    centered = batched - frame_mean
    code = torch.einsum("tk,btd,dc->bkc", temporal_basis, centered, content_basis)
    if tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS):
        raise RuntimeError("analytic Tucker-B384 code geometry differs")
    return code[0] if value.ndim == 2 else code


def _analytic_tucker_decode_from_code(
    code: torch.Tensor, fitted: TuckerFit,
) -> torch.Tensor:
    unbatched = code.ndim == 2
    batched = code.unsqueeze(0) if unbatched else code
    if tuple(batched.shape[1:]) != (CODE_TIME, CODE_CHANNELS):
        raise ValueError("analytic Tucker-B384 decoder code geometry differs")
    frame_mean = fitted.frame_mean.to(batched.device)
    temporal_basis = fitted.temporal_basis.to(batched.device)
    content_basis = fitted.content_basis[:, :CODE_CHANNELS].to(batched.device)
    decoded = frame_mean + torch.einsum(
        "tk,bkc,dc->btd", temporal_basis, batched, content_basis
    )
    decoded = _canonical_batch(decoded)
    return decoded[0] if unbatched else decoded


def _analytic_tucker_decode(value: torch.Tensor, fitted: TuckerFit) -> torch.Tensor:
    """Decode the fixed B384 Tucker code and apply output C in raw coordinates."""

    return _analytic_tucker_decode_from_code(
        _analytic_tucker_encode(value, fitted), fitted
    )


@torch.no_grad()
def _model_decode_batches(
    model: TuckerInitializedTemporalConvAE, values: torch.Tensor, batch_size: int,
) -> torch.Tensor:
    output = []
    model.eval()
    for start in range(0, len(values), batch_size):
        output.append(model(values[start:start + batch_size]))
    result = torch.cat(output, dim=0)
    if result.shape != values.shape or not bool(torch.isfinite(result).all()):
        raise RuntimeError("decoded candidate output differs")
    return result


def _evaluate_fold(
    oof_rows: Sequence[v4c.Record],
    model: TuckerInitializedTemporalConvAE, selected_step: int,
    fitted: TuckerFit, config: Config, device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate only the five independently materialized backbone views."""

    if not oof_rows or any(set(row.views) != set(EVAL_VIEWS) for row in oof_rows):
        raise ValueError("v4-D OOF five-view closure differs")
    stacked_cpu = {
        name: torch.stack([v4c.canonical_action(row.views[name]) for row in oof_rows])
        for name in EVAL_VIEWS
    }
    stacked = {name: values.to(device) for name, values in stacked_cpu.items()}
    baseline = {
        name: _analytic_tucker_decode(values, fitted) for name, values in stacked.items()
    }
    if selected_step == 0:
        # Exact alias is contractual: no floating-point discrepancy may create
        # a fictitious step-0 improvement over the analytic baseline.
        candidate = baseline
        step0_alias_used = True
    else:
        candidate = {
            name: _model_decode_batches(model, values, config.batch_size)
            for name, values in stacked.items()
        }
        step0_alias_used = False
    output: list[dict[str, Any]] = []
    for index, row in enumerate(oof_rows):
        teacher_margin = {
            negative: v4c._margin(
                stacked_cpu["original"][index].flatten(),
                stacked_cpu["monotone_warp"][index].flatten(),
                stacked_cpu[negative][index].flatten(),
            )
            for negative in NEGATIVES
        }
        baseline_margin = {
            negative: v4c._margin(
                baseline["original"][index].flatten(),
                baseline["monotone_warp"][index].flatten(),
                baseline[negative][index].flatten(),
            )
            for negative in NEGATIVES
        }
        candidate_margin = {
            negative: v4c._margin(
                candidate["original"][index].flatten(),
                candidate["monotone_warp"][index].flatten(),
                candidate[negative][index].flatten(),
            )
            for negative in NEGATIVES
        }
        reconstruction = {
            view: {
                "candidate_raw_mse": float(_raw_mse(
                    candidate[view][index], stacked[view][index]
                ).detach().cpu()),
                "tucker_b384_raw_mse": float(_raw_mse(
                    baseline[view][index], stacked[view][index]
                ).detach().cpu()),
            }
            for view in EVAL_VIEWS
        }
        finite_values = [
            value for view in reconstruction.values() for value in view.values()
        ] + [
            values["margin"]
            for table in (teacher_margin, baseline_margin, candidate_margin)
            for values in table.values()
        ]
        if any(not math.isfinite(float(value)) for value in finite_values):
            raise RuntimeError("OOF evidence contains a non-finite value")
        output.append({
            "iid": row.iid,
            "family": row.family,
            "teacher_margin_by_negative": {
                name: teacher_margin[name]["margin"] for name in NEGATIVES
            },
            "tucker_b384_margin_by_negative": {
                name: baseline_margin[name]["margin"] for name in NEGATIVES
            },
            "candidate_margin_by_negative": {
                name: candidate_margin[name]["margin"] for name in NEGATIVES
            },
            "raw_reconstruction_by_view": reconstruction,
            "selected_step0_exact_tucker_alias_used": step0_alias_used,
        })
    return output


def _paired_lcb(
    values: Sequence[float], families: Sequence[str], config: Config, label: str,
) -> dict[str, Any]:
    result = v4a._paired_bootstrap_lcbs(values, families, config, f"v4d:{label}")
    result["both_lcbs_strictly_gt_zero"] = v4a._strictly_positive_both(result)
    return result


def _paired_ratio_ucb(
    candidate_errors: Sequence[float], baseline_errors: Sequence[float],
    families: Sequence[str], config: Config, label: str,
) -> dict[str, Any]:
    """Paired ratio-of-means bootstrap; never mean of per-IID ratios."""

    if len(candidate_errors) != 644 or len(baseline_errors) != 644 or len(families) != 644:
        raise ValueError("paired ratio bootstrap requires exact644")
    if (any(not math.isfinite(float(value)) or float(value) < 0.0 for value in candidate_errors)
            or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in baseline_errors)):
        raise ValueError("paired ratio errors are non-finite or denominator is non-positive")
    family_names = sorted(set(families))
    if len(family_names) != 28:
        raise ValueError("paired ratio family closure is not exact28")
    candidate = torch.tensor(candidate_errors, dtype=torch.float64)
    baseline = torch.tensor(baseline_errors, dtype=torch.float64)

    clip_seed = v4a._bootstrap_seed(config, "v4d", label, "ratio", "clip")
    clip_generator = torch.Generator().manual_seed(clip_seed)
    clip_indices = torch.randint(
        644, (config.bootstrap_draws, 644), generator=clip_generator
    )
    clip_candidate = candidate[clip_indices].mean(dim=1)
    clip_baseline = baseline[clip_indices].mean(dim=1)
    if not bool((clip_baseline > 0.0).all()):
        raise ValueError("clip bootstrap ratio denominator is non-positive")
    clip_ratios = clip_candidate / clip_baseline

    candidate_family = torch.tensor([
        sum(float(value) for value, family in zip(candidate_errors, families) if family == name)
        / sum(family == name for family in families)
        for name in family_names
    ], dtype=torch.float64)
    baseline_family = torch.tensor([
        sum(float(value) for value, family in zip(baseline_errors, families) if family == name)
        / sum(family == name for family in families)
        for name in family_names
    ], dtype=torch.float64)
    if not bool((baseline_family > 0.0).all()):
        raise ValueError("family ratio denominator is non-positive")
    family_seed = v4a._bootstrap_seed(config, "v4d", label, "ratio", "family")
    family_generator = torch.Generator().manual_seed(family_seed)
    family_indices = torch.randint(
        28, (config.bootstrap_draws, 28), generator=family_generator
    )
    family_candidate_draw = candidate_family[family_indices].mean(dim=1)
    family_baseline_draw = baseline_family[family_indices].mean(dim=1)
    if not bool((family_baseline_draw > 0.0).all()):
        raise ValueError("family-bootstrap ratio denominator is non-positive")
    family_ratios = family_candidate_draw / family_baseline_draw
    quantile = 1.0 - config.bootstrap_alpha
    clip_point = float(candidate.mean() / baseline.mean())
    family_point = float(candidate_family.mean() / baseline_family.mean())
    clip_ucb = float(torch.quantile(clip_ratios, quantile))
    family_ucb = float(torch.quantile(family_ratios, quantile))
    result = {
        "paired_original_count": 644,
        "ratio_estimand": "ratio_of_paired_resampled_mean_raw_MSEs_not_mean_of_per_IID_ratios",
        "clip_micro_point_ratio": clip_point,
        "family_macro_point_ratio": family_point,
        "clip_paired_bootstrap": {
            "draws": config.bootstrap_draws,
            "seed": clip_seed,
            "one_sided_alpha": config.bootstrap_alpha,
            "ucb": clip_ucb,
        },
        "family_cluster_paired_bootstrap": {
            "cluster_count": 28,
            "draws": config.bootstrap_draws,
            "seed": family_seed,
            "one_sided_alpha": config.bootstrap_alpha,
            "equal_family_weight": True,
            "ucb": family_ucb,
        },
        "limit": config.recon_ratio_limit,
        "both_ucbs_le_1p05": bool(
            clip_ucb <= config.recon_ratio_limit
            and family_ucb <= config.recon_ratio_limit
        ),
    }
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]], config: Config) -> dict[str, Any]:
    if len(rows) != 644 or len({row["iid"] for row in rows}) != 644:
        raise ValueError("OOF aggregation is not exact644 once each")
    families = [str(row["family"]) for row in rows]
    fold_counts = tuple(
        sum(int(row["outer_fold"]) == fold for row in rows)
        for fold in range(OUTER_FOLDS)
    )
    if fold_counts != FROZEN_OOF_COUNTS:
        raise ValueError("frozen exact5 OOF counts differ")
    fidelity: dict[str, Any] = {}
    for view in EVAL_VIEWS:
        candidate_errors = [
            float(row["raw_reconstruction_by_view"][view]["candidate_raw_mse"])
            for row in rows
        ]
        baseline_errors = [
            float(row["raw_reconstruction_by_view"][view]["tucker_b384_raw_mse"])
            for row in rows
        ]
        fidelity[view] = _paired_ratio_ucb(
            candidate_errors, baseline_errors, families, config, f"recon:{view}"
        )
        per_fold_point_ratio: dict[str, float] = {}
        for fold in range(OUTER_FOLDS):
            indices = [index for index, row in enumerate(rows) if int(row["outer_fold"]) == fold]
            denominator = sum(baseline_errors[index] for index in indices) / len(indices)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("per-fold fidelity denominator is non-positive")
            ratio = (sum(candidate_errors[index] for index in indices) / len(indices)) / denominator
            if not math.isfinite(ratio):
                raise ValueError("per-fold fidelity ratio is non-finite")
            per_fold_point_ratio[str(fold)] = ratio
        fidelity[view]["per_fold_ratio_of_mean_raw_mses"] = per_fold_point_ratio
        fidelity[view]["all_five_fold_point_ratios_le_1p05"] = all(
            value <= config.recon_ratio_limit for value in per_fold_point_ratio.values()
        )
    five_view_fidelity_gate = all(
        fidelity[view]["both_ucbs_le_1p05"]
        and fidelity[view]["all_five_fold_point_ratios_le_1p05"]
        for view in EVAL_VIEWS
    )

    def add_per_fold_gate(
        statistics: Mapping[str, Any], values: Sequence[float], label: str,
    ) -> dict[str, Any]:
        means = {
            str(fold): sum(
                float(value) for value, row in zip(values, rows)
                if int(row["outer_fold"]) == fold
            ) / fold_counts[fold]
            for fold in range(OUTER_FOLDS)
        }
        if any(not math.isfinite(value) for value in means.values()):
            raise ValueError(f"per-fold {label} point mean is non-finite")
        result = dict(statistics)
        result["per_fold_point_mean"] = means
        result["all_five_fold_point_means_strictly_gt_zero"] = all(
            value > 0.0 for value in means.values()
        )
        return result

    negative_results: dict[str, Any] = {}
    for negative in NEGATIVES:
        teacher_values = [float(row["teacher_margin_by_negative"][negative]) for row in rows]
        baseline_values = [
            float(row["tucker_b384_margin_by_negative"][negative]) for row in rows
        ]
        candidate_values = [
            float(row["candidate_margin_by_negative"][negative]) for row in rows
        ]
        retention_values = [
            candidate - config.teacher_retention * teacher
            for candidate, teacher in zip(candidate_values, teacher_values)
        ]
        improvement_values = [
            candidate - baseline
            for candidate, baseline in zip(candidate_values, baseline_values)
        ]
        teacher = add_per_fold_gate(
            _paired_lcb(teacher_values, families, config, f"teacher:{negative}"),
            teacher_values, f"teacher:{negative}",
        )
        baseline = _paired_lcb(baseline_values, families, config, f"tucker:{negative}")
        candidate = add_per_fold_gate(
            _paired_lcb(candidate_values, families, config, f"candidate:{negative}"),
            candidate_values, f"candidate:{negative}",
        )
        retention = add_per_fold_gate(
            _paired_lcb(
                retention_values, families, config,
                f"candidate-minus-0p8-teacher:{negative}",
            ),
            retention_values, f"candidate-minus-0p8-teacher:{negative}",
        )
        improvement = add_per_fold_gate(
            _paired_lcb(
                improvement_values, families, config,
                f"candidate-minus-tucker:{negative}",
            ),
            improvement_values, f"candidate-minus-tucker:{negative}",
        )
        gate = bool(
            teacher["both_lcbs_strictly_gt_zero"]
            and teacher["all_five_fold_point_means_strictly_gt_zero"]
            and candidate["both_lcbs_strictly_gt_zero"]
            and candidate["all_five_fold_point_means_strictly_gt_zero"]
            and retention["both_lcbs_strictly_gt_zero"]
            and retention["all_five_fold_point_means_strictly_gt_zero"]
            and improvement["both_lcbs_strictly_gt_zero"]
            and improvement["all_five_fold_point_means_strictly_gt_zero"]
        )
        negative_results[negative] = {
            "teacher_margin": teacher,
            "fixed_tucker_b384_margin": baseline,
            "candidate_margin": candidate,
            "candidate_minus_0p8_teacher_margin": retention,
            "candidate_minus_fixed_tucker_b384_margin": improvement,
            "all_four_quantities_pass_dual_bootstrap_and_every_fold": gate,
            "decoded_negative_gate": gate,
        }
    all_negative_gates = all(
        negative_results[name]["decoded_negative_gate"] for name in NEGATIVES
    )
    development_gate = bool(five_view_fidelity_gate and all_negative_gates)
    return {
        "five_view_raw_reconstruction_ratio_vs_fixed_tucker_b384": fidelity,
        "five_view_fidelity_gate": five_view_fidelity_gate,
        "negative_results": negative_results,
        "all_three_decoded_negative_gates": all_negative_gates,
        "decoded_temporal_codec_development_gate": development_gate,
        "latent_metric_qualified": False,
        "latent_gauge_fixed": False,
        "frozen_oof_counts_by_fold": list(fold_counts),
    }


def _load_v4c_frontier_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    if expected_sha256 != V4C_FRONTIER_RECEIPT_SHA256:
        raise ValueError("v4-C frontier receipt SHA is not the frozen authority")
    value = v4c._load_json_sealed(path, V4C_FRONTIER_RECEIPT_SHA256)
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    closure = value.get("oof_closure")
    evidence = (
        closure.get("embedded_paired_margin_evidence")
        if type(closure) is dict else None
    )
    if (
        value.get("schema_version") != v4c.SCHEMA
        or value.get("status") != v4c.STATUS
        or digest != V4C_FRONTIER_RECEIPT_DIGEST
        or _object_sha(unsigned) != digest
        or value.get("implementation", {}).get("implementation", {}).get("sha256")
        != V4C_IMPLEMENTATION_SHA256
        or value.get("feature_authority", {}).get("feature_receipt_sha256")
        != V4C_FEATURE_RECEIPT_SHA256
        or value.get("frozen_split", {}).get("outer_assignment_digest")
        != v4c.OUTER_ASSIGNMENT_DIGEST
        or value.get("qualified_temporal_mechanics_candidates") != []
        or type(evidence) is not list or len(evidence) != 644
        or closure.get("embedded_paired_margin_evidence_count") != 644
        or closure.get("embedded_paired_margin_evidence_sha256")
        != _object_sha(evidence)
    ):
        raise ValueError("sealed v4-C frontier authority differs")
    return value


def _verify_v4c_embedded_teacher_evidence(
    rows: Sequence[Mapping[str, Any]], v4c_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    upstream_rows = v4c_receipt["oof_closure"]["embedded_paired_margin_evidence"]
    upstream = {row["iid"]: row for row in upstream_rows}
    if len(upstream) != 644 or set(upstream) != {row["iid"] for row in rows}:
        raise ValueError("v4-C/v4-D exact644 IID evidence closure differs")
    max_teacher = 0.0
    for row in rows:
        reference = upstream[row["iid"]]
        if (reference["family"] != row["family"]
                or int(reference["outer_fold"]) != int(row["outer_fold"])):
            raise ValueError("v4-C/v4-D family or fold authority differs")
        for negative in NEGATIVES:
            teacher_difference = abs(
                float(row["teacher_margin_by_negative"][negative])
                - float(reference["teacher_margin_by_negative"][negative])
            )
            max_teacher = max(max_teacher, teacher_difference)
    if max_teacher > 1.0e-12:
        raise ValueError("v4-C/v4-D teacher evidence differs")
    return {
        "exact644_iids_matched": True,
        "exact28_families_matched": len({row["family"] for row in rows}) == 28,
        "outer_fold_matched_per_iid": True,
        "max_abs_teacher_margin_difference_vs_v4c_receipt": max_teacher,
        "teacher_reference_tolerance": 1.0e-12,
    }


def _run_fold(
    records: Sequence[v4c.Record], outer_assignment: Mapping[str, int],
    v4a_receipt: Mapping[str, Any], fold_index: int, config: Config,
    device: torch.device, checkpoint_path: Path,
    run_binding: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups, split = _split_fold(records, outer_assignment, fold_index, config)
    upstream_fold = v4a_receipt["folds"][fold_index]
    if (
        split["outer_assignment_digest"] != v4c.OUTER_ASSIGNMENT_DIGEST
        or len(groups["exploratory_oof"]) != FROZEN_OOF_COUNTS[fold_index]
        or split["outer_oof_iid_digest"] != upstream_fold["oof_iid_digest"]
    ):
        raise ValueError("fold is not the frozen v2/v4-A exact5 split")
    fit_iids = [row.iid for row in groups["model_fit"]]
    validation_iids = [row.iid for row in groups["inner_validation"]]
    oof_iids = [row.iid for row in groups["exploratory_oof"]]
    if (
        set(fit_iids) & set(validation_iids)
        or set(fit_iids) & set(oof_iids)
        or set(validation_iids) & set(oof_iids)
        or _object_sha(fit_iids) != split["model_fit_iid_digest"]
        or _object_sha(validation_iids) != split["inner_validation_iid_digest"]
        or _object_sha(oof_iids) != split["outer_oof_iid_digest"]
    ):
        raise ValueError("model-fit/inner-validation/OOF fold closure differs")

    fitted = _fit_tucker_b384(groups["model_fit"])
    if fitted.fit_iid_digest != split["model_fit_iid_digest"]:
        raise RuntimeError("fit-only Tucker/model-fit IID join differs")
    model, selected_step, training_audit = _train_fold_model(
        groups["model_fit"], groups["inner_validation"], fitted,
        config, fold_index, device,
    )
    checkpoint = _save_selected_checkpoint_create_only(
        checkpoint_path, model, fitted, selected_step, training_audit,
        config, fold_index, run_binding, groups["inner_validation"], device,
    )
    evaluation = _evaluate_fold(
        groups["exploratory_oof"], model, selected_step, fitted, config, device
    )
    for row in evaluation:
        row["outer_fold"] = fold_index
    if [row["iid"] for row in evaluation] != oof_iids:
        raise ValueError("OOF evaluation order differs")
    fold_receipt = {
        "fold_index": fold_index,
        "frozen_v4a_fold_iid_digest": v4c.FOLD_IID_DIGESTS[fold_index],
        "frozen_v4a_outer_assignment_digest": split["outer_assignment_digest"],
        "frozen_v4a_oof_iid_digest": upstream_fold["oof_iid_digest"],
        "inner_split": split,
        "model_fit_original_count": len(fit_iids),
        "model_fit_ordered_iids": fit_iids,
        "model_fit_iid_digest": _object_sha(fit_iids),
        "inner_validation_original_count": len(validation_iids),
        "inner_validation_iid_digest": _object_sha(validation_iids),
        "oof_original_count": len(oof_iids),
        "oof_iid_digest": _object_sha(oof_iids),
        "partition_pairwise_disjoint": True,
        "fixed_tucker_b384_fit_input_sha256": fitted.fit_input_sha256,
        "fixed_tucker_b384_fit_iid_digest": fitted.fit_iid_digest,
        "fixed_tucker_b384_diagnostics": fitted.diagnostics,
        "training": training_audit,
        "selected_checkpoint_artifact": checkpoint,
        "selected_checkpoint_completed_before_oof_transform_or_model_evaluation": True,
        "oof_used_for_training_checkpoint_or_hyperparameter_selection": False,
        "oof_evaluation_sha256": _object_sha(evaluation),
    }
    return fold_receipt, evaluation


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" or name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        device = torch.device(name)
        _ = torch.empty(1, device=device)
        return device
    raise ValueError("device must be cpu, cuda, or cuda:N")


def _config_value(config: Config) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "checkpoint_steps": list(config.checkpoint_steps),
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_alpha": config.bootstrap_alpha,
        "teacher_retention": config.teacher_retention,
        "recon_ratio_limit": config.recon_ratio_limit,
        "code_shape": [CODE_TIME, CODE_CHANNELS],
        "actual_code_numel": CODE_NUMEL,
    }


def _checkpoint_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_size, value.st_mode,
        value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
    )


def _load_selected_checkpoint_sealed(
    path: Path, expected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    """Load and semantically replay one sealed checkpoint through one FD."""

    if (
        not path.is_absolute() or path.is_symlink()
        or str(path) != str(path.resolve(strict=True))
    ):
        raise ValueError("checkpoint path must be absolute/canonical/non-symlink")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1
    ):
        raise RuntimeError("checkpoint pre-open seal differs")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("checkpoint O_NOFOLLOW is unavailable")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if _checkpoint_stat_identity(opened) != _checkpoint_stat_identity(before):
                raise RuntimeError("checkpoint path/open FD identity differs")
            digest_before = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_before.update(chunk)
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
            loaded = os.fstat(handle.fileno())
            handle.seek(0)
            digest_after = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_after.update(chunk)
            closed = os.fstat(handle.fileno())
    except BaseException:
        # fdopen owns and closes descriptor on every path.
        raise
    after = path.lstat()
    identities = tuple(
        _checkpoint_stat_identity(value)
        for value in (before, opened, loaded, closed, after)
    )
    file_sha = digest_before.hexdigest()
    physical_identity = {
        "device": before.st_dev,
        "inode": before.st_ino,
        "size_bytes": before.st_size,
    }
    if (
        path.is_symlink() or len(set(identities)) != 1
        or digest_after.hexdigest() != file_sha
        or before.st_size <= 0
    ):
        raise RuntimeError("checkpoint single-FD pre/post identity or SHA differs")
    if (
        expected.get("file_sha256") is not None
        and expected.get("file_sha256") != file_sha
    ):
        raise RuntimeError("checkpoint expected file SHA differs")
    if (
        expected.get("size_bytes") is not None
        and int(expected["size_bytes"]) != before.st_size
    ):
        raise RuntimeError("checkpoint expected size differs")
    if (
        expected.get("physical_identity") is not None
        and expected.get("physical_identity") != physical_identity
    ):
        raise RuntimeError("checkpoint expected physical identity differs")
    if type(payload) is not dict or set(payload) != {"metadata", "state_dict"}:
        raise RuntimeError("checkpoint safe payload envelope differs")
    metadata = payload["metadata"]
    state = payload["state_dict"]
    if type(metadata) is not dict or type(state) is not dict or not state:
        raise RuntimeError("checkpoint metadata/state envelope differs")
    metadata_unsigned = dict(metadata)
    metadata_digest = metadata_unsigned.pop("metadata_digest", None)
    model_fit_iids = metadata.get("model_fit_ordered_iids")
    implementation = metadata.get("implementation")
    if (
        metadata.get("schema_version")
        != "semantic-anchor-vjepa2-nonlinear-temporal-codec-selected-fold-checkpoint-v4d"
        or metadata_digest != _object_sha(metadata_unsigned)
        or metadata.get("outer_fold") != expected.get("outer_fold")
        or metadata.get("selected_step") != expected.get("selected_step")
        or metadata.get("model_state_sha256")
        != expected.get("model_state_sha256")
        or metadata_digest != expected.get("metadata_digest")
        or metadata.get("full_budget_steps_executed") != Config().max_steps
        or metadata.get("checkpoint_schedule") != list(Config().checkpoint_steps)
        or metadata.get("config") != _config_value(Config())
        or metadata.get("config_sha256") != _object_sha(_config_value(Config()))
        or type(implementation) is not dict
        or implementation.get("implementation_sha256")
        != expected.get("implementation_sha256")
        or metadata.get("refit_artifact") is not False
        or metadata.get("inference_authorized") is not False
        or type(model_fit_iids) is not list or not model_fit_iids
        or len(model_fit_iids) != metadata.get("model_fit_original_count")
        or len(set(model_fit_iids)) != len(model_fit_iids)
        or _object_sha(model_fit_iids) != metadata.get("model_fit_iid_digest")
    ):
        raise RuntimeError("checkpoint semantic metadata replay differs")
    if (
        any(type(name) is not str or type(value) is not torch.Tensor
            for name, value in state.items())
        or any(not bool(value.isfinite().all()) for value in state.values())
    ):
        raise RuntimeError("checkpoint state tensor closure differs")
    state_sha = _state_sha(state)
    if state_sha != metadata["model_state_sha256"]:
        raise RuntimeError("checkpoint semantic state replay differs")
    binding = {
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha,
        "size_bytes": before.st_size,
        "mode_octal": "0444",
        "nlink": before.st_nlink,
        "physical_identity": physical_identity,
        "single_fd_pre_post_sha256_exact": True,
        "semantic_metadata_state_replay_verified": True,
    }
    return metadata, state, binding


def _save_selected_checkpoint_create_only(
    path: Path, model: TuckerInitializedTemporalConvAE,
    fitted: TuckerFit, selected_step: int,
    training_audit: Mapping[str, Any], config: Config, fold_index: int,
    run_binding: Mapping[str, str], validation_rows: Sequence[v4c.Record],
    device: torch.device,
) -> dict[str, Any]:
    if (
        not path.is_absolute() or not path.parent.is_dir()
        or path.exists() or path.is_symlink()
    ):
        raise ValueError("checkpoint must be a fresh absolute child")
    state = _state_to_cpu(model)
    state_sha = _state_sha(state)
    if (
        selected_step != int(training_audit["selected_step"])
        or state_sha != training_audit["selected_state_sha256"]
    ):
        raise RuntimeError("physical checkpoint does not join selected step/state")
    metadata: dict[str, Any] = {
        "schema_version": "semantic-anchor-vjepa2-nonlinear-temporal-codec-selected-fold-checkpoint-v4d",
        "outer_fold": fold_index,
        "selected_step": selected_step,
        "full_budget_steps_executed": config.max_steps,
        "checkpoint_schedule": list(config.checkpoint_steps),
        "minibatch_schedule_sha256": training_audit["minibatch_schedule_sha256"],
        "model_state_sha256": state_sha,
        "selected_training_audit_state_join_verified": True,
        "config": _config_value(config),
        "config_sha256": _object_sha(_config_value(config)),
        "implementation": dict(run_binding),
        "fixed_comparator_name": BASELINE_NAME,
        "basis": {
            "frame_mean_sha256": _tensor_sha(fitted.frame_mean),
            "temporal_basis_sha256": _tensor_sha(fitted.temporal_basis),
            "content_basis_first96_sha256": _tensor_sha(
                fitted.content_basis[:, :CODE_CHANNELS]
            ),
            "fit_only_global_rms_sha256": training_audit["fit_only_global_rms_sha256"],
            "fixed_tucker_fit_input_sha256": fitted.fit_input_sha256,
        },
        "model_fit_original_count": training_audit["model_fit_original_count"],
        "model_fit_ordered_iids": training_audit["model_fit_ordered_iids"],
        "model_fit_iid_digest": training_audit["model_fit_iid_digest"],
        "inner_validation_iid_digest": training_audit["inner_validation_iid_digest"],
        "artifact_scope": "selected burned-development fold codec checkpoint; not refit or authorized inference",
        "refit_artifact": False,
        "inference_authorized": False,
        "cross_environment_bit_exact_weights_claimed": False,
    }
    metadata["metadata_digest"] = _object_sha(metadata)
    payload = {"metadata": metadata, "state_dict": state}
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
        written = os.fstat(handle.fileno())
    os.chmod(path, 0o444)
    expectation = {
        "outer_fold": fold_index,
        "selected_step": selected_step,
        "model_state_sha256": state_sha,
        "metadata_digest": metadata["metadata_digest"],
        "implementation_sha256": run_binding["implementation_sha256"],
    }
    loaded_metadata, loaded_state, binding = _load_selected_checkpoint_sealed(
        path, expectation
    )
    if (
        loaded_metadata != metadata or _state_sha(loaded_state) != state_sha
        or binding["physical_identity"] != {
            "device": written.st_dev,
            "inode": written.st_ino,
            "size_bytes": written.st_size,
        }
    ):
        raise RuntimeError("fresh checkpoint reload metadata/state differs")
    reloaded = TuckerInitializedTemporalConvAE(
        fitted, model.fit_only_rms.detach().cpu()
    )
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.to(device).eval()
    probe = torch.stack([
        v4c.canonical_action(row.views["original"])
        for row in validation_rows[:min(config.batch_size, len(validation_rows))]
    ]).to(device)
    with torch.no_grad():
        expected = model(probe)
        actual = reloaded(probe)
    if not torch.equal(expected, actual):
        raise RuntimeError("fresh checkpoint strict reload output is not bit-exact")
    # The caller continues with this exact object, so make the sealed file the
    # last state authority before any OOF value is evaluated.
    model.load_state_dict(loaded_state, strict=True)
    model.to(device).eval()
    return {
        **binding,
        "outer_fold": fold_index,
        "selected_step": selected_step,
        "model_state_sha256": state_sha,
        "implementation_sha256": run_binding["implementation_sha256"],
        "selected_training_audit_state_join_verified": True,
        "metadata_digest": metadata["metadata_digest"],
        "fresh_reload_strict_state_verified": True,
        "fresh_reload_output_bit_exact": True,
        "caller_model_reloaded_from_sealed_artifact_before_oof": True,
    }


def _verify_checkpoint_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> None:
    if len(artifacts) != OUTER_FOLDS:
        raise RuntimeError("selected checkpoint artifact count differs")
    for fold, artifact in enumerate(artifacts):
        path = Path(str(artifact["path"]))
        if (
            int(artifact["outer_fold"]) != fold
        ):
            raise RuntimeError("sealed selected checkpoint artifact changed")
        metadata, state, binding = _load_selected_checkpoint_sealed(path, artifact)
        if (
            binding["file_sha256"] != artifact["file_sha256"]
            or binding["size_bytes"] != artifact["size_bytes"]
            or metadata["model_state_sha256"] != artifact["model_state_sha256"]
            or _state_sha(state) != artifact["model_state_sha256"]
        ):
            raise RuntimeError("sealed selected checkpoint replay changed")


def run_exact5(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-D torch runtime differs")
    torch.set_num_threads(1)
    device = _resolve_device(args.device)
    output = Path(args.output)
    if (
        not output.is_absolute() or not output.parent.is_dir()
        or output.exists() or output.is_symlink()
    ):
        raise ValueError("output must be a fresh absolute JSON child")
    checkpoint_paths = [
        output.with_name(f"{output.stem}.selected_fold{fold}.pt")
        for fold in range(OUTER_FOLDS)
    ]
    if len(set(checkpoint_paths)) != OUTER_FOLDS or any(
        path.exists() or path.is_symlink() or path.parent != output.parent
        for path in checkpoint_paths
    ):
        raise ValueError("all five selected checkpoint paths must be fresh siblings")
    if args.expected_feature_receipt_sha256 != V4C_FEATURE_RECEIPT_SHA256:
        raise ValueError("v4-C feature receipt SHA is not frozen")
    v4a_receipt_path = Path(args.v4a_receipt)
    outer_assignment, v4a_receipt = v4c.load_frozen_v4a_split(
        v4a_receipt_path, args.expected_v4a_receipt_sha256
    )
    v4c_frontier_path = Path(args.v4c_frontier_receipt)
    v4c_frontier_receipt = _load_v4c_frontier_receipt(
        v4c_frontier_path, args.expected_v4c_frontier_receipt_sha256
    )
    records, feature_receipt = v4c.load_v4c_features(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    records_by_iid = {row.iid: row for row in records}
    split_evidence = v4a_receipt["oof_closure"]["embedded_paired_margin_evidence"]
    split_order_records = [records_by_iid[row["iid"]] for row in split_evidence]
    exact_iids = [row.iid for row in split_order_records]
    if (
        len(records) != 644 or len(records_by_iid) != 644
        or len(set(exact_iids)) != 644 or set(exact_iids) != set(outer_assignment)
        or len({row.family for row in split_order_records}) != 28
        or _object_sha(outer_assignment) != v4c.OUTER_ASSIGNMENT_DIGEST
        or any(row.family != evidence["family"]
               for row, evidence in zip(split_order_records, split_evidence))
        or v4c_frontier_receipt["feature_authority"]["feature_receipt_sha256"]
            != args.expected_feature_receipt_sha256
    ):
        raise ValueError("v4-D feature/split/frontier population differs")
    feature_root = Path(args.feature_root).resolve(strict=True)
    feature_receipt_bound = dict(feature_receipt)
    feature_receipt_bound["feature_receipt_path"] = str(
        feature_root / "feature_extraction_receipt.json"
    )
    feature_receipt_bound["feature_receipt_file_sha256"] = (
        args.expected_feature_receipt_sha256
    )

    fold_receipts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for fold_index in range(OUTER_FOLDS):
        fold_receipt, rows = _run_fold(
            split_order_records, outer_assignment, v4a_receipt, fold_index,
            config, device, checkpoint_paths[fold_index], run_binding,
        )
        fold_receipts.append(fold_receipt)
        all_rows.extend(rows)
    if (
        len(all_rows) != 644 or len({row["iid"] for row in all_rows}) != 644
        or {row["iid"] for row in all_rows} != set(exact_iids)
        or tuple(sum(int(row["outer_fold"]) == fold for row in all_rows)
                 for fold in range(OUTER_FOLDS)) != FROZEN_OOF_COUNTS
    ):
        raise ValueError("exact5 OOF union is not exact644 once each")
    upstream_match = _verify_v4c_embedded_teacher_evidence(
        all_rows, v4c_frontier_receipt
    )
    metrics = _aggregate(all_rows, config)
    selected_steps = [
        int(fold["training"]["selected_step"]) for fold in fold_receipts
    ]
    config_value = _config_value(config)
    checkpoint_artifacts = [
        fold["selected_checkpoint_artifact"] for fold in fold_receipts
    ]
    if (
        len(checkpoint_artifacts) != OUTER_FOLDS
        or [artifact["outer_fold"] for artifact in checkpoint_artifacts]
            != list(range(OUTER_FOLDS))
        or [artifact["selected_step"] for artifact in checkpoint_artifacts]
            != selected_steps
        or any(
            artifact["model_state_sha256"]
            != fold_receipts[index]["training"]["selected_state_sha256"]
            for index, artifact in enumerate(checkpoint_artifacts)
        )
    ):
        raise RuntimeError("five selected checkpoint artifacts do not join folds")
    _verify_checkpoint_artifacts(checkpoint_artifacts)
    v4c._assert_input_files_unchanged(feature_receipt_bound, v4a_receipt_path)
    _load_v4c_frontier_receipt(
        v4c_frontier_path, args.expected_v4c_frontier_receipt_sha256
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "authority": "burned_development_decoded_temporal_codec_diagnostic_only",
        "implementation": run_binding,
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "device": str(device),
        "deterministic_algorithms_required": True,
        "runtime": {
            "torch": str(torch.__version__),
            "torch_hip": str(torch.version.hip),
            "torch_num_threads": torch.get_num_threads(),
            "full_precision_fp32_training": True,
            "autocast_used": False,
            "distributed_training_used": False,
        },
        "feature_authority": {
            "feature_root": str(feature_root),
            "feature_receipt_path": str(
                (feature_root / "feature_extraction_receipt.json").resolve(strict=True)
            ),
            "feature_receipt_sha256": args.expected_feature_receipt_sha256,
            "feature_receipt_digest": feature_receipt["receipt_digest"],
            "extractor_implementation_sha256": EXTRACTOR_IMPLEMENTATION_SHA256,
            "unique_original_iids": 644,
            "family_count": 28,
            "stored_views_per_original": len(EVAL_VIEWS),
            "stored_views": list(EVAL_VIEWS),
            "all_five_views_are_separate_frozen_backbone_forwards": True,
            "post_backbone_token_transform_used_by_v4d": False,
            "exact644_ordered_iid_digest": _object_sha(exact_iids),
        },
        "upstream_authorities": {
            "v4a_receipt_path": str(v4a_receipt_path.resolve(strict=True)),
            "v4a_receipt_file_sha256": V4A_RECEIPT_FILE_SHA256,
            "v4a_receipt_self_digest": V4A_RECEIPT_SELFDIGEST,
            "v4c_frontier_receipt_path": str(v4c_frontier_path.resolve(strict=True)),
            "v4c_frontier_receipt_file_sha256": V4C_FRONTIER_RECEIPT_SHA256,
            "v4c_frontier_receipt_self_digest": V4C_FRONTIER_RECEIPT_DIGEST,
            "v4c_embedded_teacher_evidence_match": upstream_match,
        },
        "fixed_comparator_authority": {
            "fixed_comparator_name": BASELINE_NAME,
            "v4c_oof_was_burned_before_v4d": True,
            "clip_pca_b384_was_descriptively_higher_in_v4c": True,
            "clip_pca_used_to_select_tucker_rank_or_mapping": False,
            "rank_and_mapping_frozen_before_v4d_oof": True,
            "fold_basis_fit_model_fit_original_only": True,
            "inner_validation_used_for_basis_fit": False,
            "oof_used_for_basis_fit": False,
            "derived_views_used_for_basis_fit": False,
            "called_best_or_winner": False,
            "same_payload_384_scalars_only": True,
            "parameter_or_flop_fairness_claimed": False,
        },
        "frozen_split": {
            "outer_source": "pinned v4-A embedded exact5 IID assignment",
            "outer_assignment_digest": v4c.OUTER_ASSIGNMENT_DIGEST,
            "fold_iid_digests": {
                str(key): value for key, value in v4c.FOLD_IID_DIGESTS.items()
            },
            "oof_counts_by_fold": list(FROZEN_OOF_COUNTS),
            "inner_source": INNER_SPLIT_NAMESPACE,
            "inner_singleton_rule": "force inner fold 1/model-fit",
            "inner_literal_pins": list(FROZEN_INNER_SPLITS),
            "inner_split_reads_vjepa_values_or_energy": False,
            "inner_split_by_outer_fold": [fold["inner_split"] for fold in fold_receipts],
            "all_exact644_are_development": True,
            "fresh_scientific_confirmation_claimed": False,
            "iid_disjoint_only_not_actor_scene_generator_lineage_disjoint": True,
        },
        "model_contract": {
            "input": "C(view) FP32 [32,1024]",
            "fit_only_normalization": "single global RMS from model-fit originals",
            "code_shape": [CODE_TIME, CODE_CHANNELS],
            "actual_code_numel": CODE_NUMEL,
            "decoder_input": "sole [4,96] code",
            "raw_input_skip_or_side_channel": False,
            "decoder_fold_global_buffers_only": [
                "frame_mean", "temporal_basis", "content_basis", "fit_only_rms"
            ],
            "step0": "exact fold-fit fixed Tucker-B384 encoder/decoder",
            "step0_code_and_decode_bit_exact": True,
            "encoder_residual_final_layer_zero_initialized": True,
            "decoder_residual_final_layer_zero_initialized": True,
            "decoder_output_temporally_centered": True,
            "latent_scale_or_rotation_gauge_fixed": False,
            "latent_distance_used_for_gate_or_report": False,
            "exact_trainable_parameter_count": EXACT_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_exclusive": MAX_TRAINABLE_PARAMETERS,
        },
        "training_contract": {
            "model_fit_original_only": True,
            "derived_training_view_count": 0,
            "post_backbone_training_warp_used": False,
            "loss_terms": ["raw_feature", "signed_delta_stride1", "signed_delta_stride2",
                           "signed_delta_stride4", "terminal_displacement"],
            "negative_views_used_for_training": 0,
            "evaluation_positive_views_used_for_training": 0,
            "source_rows_used_for_training": 0,
            "family_or_transform_labels_enter_loss_or_model_input": False,
            "family_metadata_used_for_inner_split": True,
            "transform_metadata_used_for_inner_split": False,
            "fixed_full_budget_no_early_stop": True,
            "inner_validation_checkpoint_selection_original_reconstruction_only": True,
            "oof_selection": False,
            "selected_steps_by_fold": selected_steps,
        },
        "oof_access_contract": {
            "sealed_feature_artifacts_materialized_before_command": True,
            "outer_and_inner_splits_frozen_before_model_fit": True,
            "oof_tensors_supplied_to_optimizer_or_checkpoint_selection": False,
            "oof_model_outputs_computed_before_selection": False,
            "claim_is_no_oof_model_value_use_not_no_prior_materialization": True,
        },
        "evaluation_contract": {
            "primary_mapping": "decoded C(D(E(C(view)))) in [32,1024] coordinates",
            "views": list(EVAL_VIEWS),
            "views_read_from_independent_frozen_backbone_outputs": True,
            "post_token_view_construction": False,
            "fixed_comparator": BASELINE_NAME,
            "fidelity_gate": "each view: paired clip/family ratio-of-means UCB<=1.05 and every fold point ratio<=1.05",
            "distance": "sum_squared_decoded_difference/(32*1024)",
            "negative_gate": "each negative: teacher, candidate, candidate-0.8*teacher, candidate-Tucker each has clip/family LCB>0 and all five fold point means>0",
            "aggregate_cross_fold_compensation_sufficient": False,
            "across_negative_compensation_used": False,
            "selected_step0_aliases_exact_analytic_tucker": True,
            "latent_metric_diagnostic_or_gate": False,
        },
        "folds": fold_receipts,
        "selected_fold_checkpoint_artifacts": {
            "count": len(checkpoint_artifacts),
            "all_create_only_mode0444_nlink1": all(
                artifact["mode_octal"] == "0444" and artifact["nlink"] == 1
                for artifact in checkpoint_artifacts
            ),
            "fold_selected_step_join_verified": True,
            "artifacts_manifest_sha256": _object_sha(checkpoint_artifacts),
            "artifacts_reverified_immediately_before_receipt_write": True,
            "artifacts_reverified_after_receipt_write_by_command_before_success_return": True,
            "partial_run_policy": "any existing sibling fails closed; no resume or reuse; retry requires a new output stem; orphan checkpoints are not a completed result",
            "artifacts": checkpoint_artifacts,
        },
        "oof_closure": {
            "unique_original_iids": 644,
            "each_original_evaluated_exactly_once": True,
            "oof_counts_by_fold": list(FROZEN_OOF_COUNTS),
            "ordered_iid_digest": _object_sha([row["iid"] for row in all_rows]),
            "sorted_iid_digest": _object_sha(sorted(row["iid"] for row in all_rows)),
            "embedded_per_iid_evidence_count": len(all_rows),
            "embedded_per_iid_evidence_sha256": _object_sha(all_rows),
            "embedded_per_iid_evidence": all_rows,
            "evidence_sufficient_to_recompute_all_gates": True,
        },
        "metrics": metrics,
        "qualification_scope": {
            "temporal_codec_development_gate": metrics[
                "decoded_temporal_codec_development_gate"
            ],
            "latent_metric_qualified": False,
            "action_representation_qualified": False,
            "scientific_confirmation_claimed": False,
            "identity_disentanglement_qualified": False,
            "identity_preservation_qualified": False,
            "vae_necessary": None,
            "generation_qualified": False,
            "prior_qualified": False,
            "prior_generation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "training_authorized": False,
            "full644_refit_authorized": False,
            "video_model_training_performed": False,
            "postselection_all644_refit_authorized_or_performed": False,
        },
        "descriptive_scope": {
            "fold_local_model_fit_performed": True,
            "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    v4c._assert_input_files_unchanged(feature_receipt_bound, v4a_receipt_path)
    _load_v4c_frontier_receipt(
        v4c_frontier_path, args.expected_v4c_frontier_receipt_sha256
    )
    receipt_sha = _write_json_create_only(output, receipt)
    _verify_checkpoint_artifacts(checkpoint_artifacts)
    _assert_binding_unchanged(run_binding)
    v4c._assert_input_files_unchanged(feature_receipt_bound, v4a_receipt_path)
    _load_v4c_frontier_receipt(
        v4c_frontier_path, args.expected_v4c_frontier_receipt_sha256
    )
    return {
        "receipt": str(output.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "decoded_temporal_codec_development_gate": metrics[
            "decoded_temporal_codec_development_gate"
        ],
        "latent_metric_qualified": False,
        "selected_checkpoint_artifacts_reverified_after_receipt_write": True,
        "feature_receipt_six_shards_v4a_v4c_reverified_after_receipt_write": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact5 Tucker-initialized V-JEPA2 nonlinear temporal codec"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-exact5")
    run.add_argument("--feature-root", required=True)
    run.add_argument("--expected-feature-receipt-sha256", required=True)
    run.add_argument("--v4a-receipt", required=True)
    run.add_argument("--expected-v4a-receipt-sha256", required=True)
    run.add_argument("--v4c-frontier-receipt", required=True)
    run.add_argument("--expected-v4c-frontier-receipt-sha256", required=True)
    run.add_argument("--device", default="cuda")
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
