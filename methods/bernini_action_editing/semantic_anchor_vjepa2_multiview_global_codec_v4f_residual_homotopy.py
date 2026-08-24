#!/usr/bin/env python3
"""Future-use v4-F residual-homotopy codec over frozen V-JEPA2 sequences.

Each outer fold fits a clip-PCA B384 comparator using only model-fit originals.
The trainable codec starts bit-exactly at that comparator and communicates only
through a sole ``[12,32]`` code.  Twelve learned queries cross-attend all 32
input frames for an encoder delta; 32 learned time queries cross-attend only
the code for a decoder residual.  Both output projections are zero initialized.

Training is exactly the frozen v4-E 1200-step, rho=1 objective over all five
model-fit views, but step 1200 is fixed rather than selected.  Its preselection
checkpoint is sealed before any inner tensor is materialized.  Only then may
the five known-exposed inner views choose the first PASS in the preregistered
ascending residual-homotopy grid.  This is an ordering rule, not a monotonicity
or minimum-distortion claim.  Rho zero is an exact clip-PCA
comparator and can never be selected.  A fold with no passing rho terminates as
INNER_NO_GO without reading any OOF tensor.  A passing selected checkpoint is
separately sealed and strongly reloaded before OOF materialization.

This is a burned development diagnostic for the *known exposed transforms*.
It cannot qualify unseen hostile transforms, a latent metric, action or
identity representations, generation, rendering, inference, web evaluation,
video editing, or full-644 refitting.
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
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
import torch.nn.functional as F

from methods.bernini_action_editing import semantic_anchor_vjepa2_analytic_frontier_v4c as v4c
from methods.bernini_action_editing import semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d as v4d


v4a = v4c.v4a
features = v4c.features
SCHEMA = "semantic-anchor-vjepa2-residual-homotopy-exact5-receipt-v4f"
FOLD_SCHEMA = "semantic-anchor-vjepa2-residual-homotopy-fold-receipt-v4f"
CHECKPOINT_SCHEMA = "semantic-anchor-vjepa2-residual-homotopy-fold-checkpoint-v4f"
STATUS = "V4F_RESIDUAL_HOMOTOPY_KNOWN_EXPOSED_DEVELOPMENT"
INNER_NO_GO_STATUS = "V4F_INNER_NO_GO_OOF_UNREAD"
RELEASE_SEALED = True
V4E_BURNED_IMPLEMENTATION_SHA256 = (
    "4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a"
)
V4E_BURNED_FOLD_RECEIPT_SHA256 = (
    "76d5aaf4667ac7a99f26788faa3f205c360479836200f5abc4715d3a9afd7cee",
    "bd30bf71b509154847bba3a7a474a9e2ecfd38c13b04687970be6adec82b0d67",
    "10b5c8d2271353baf94633eeda9e359ef765271234d7d69486506fd32abdc25f",
    "c1639cee6151e7ad28adaecd27186c3dc70fe241cdb19c3cf905541648dc8d0d",
    "9dbd57b84b8e3498315536c8aff6d19123add300d1cb12cba134e073215cf33d",
)
SEED = 20260819
TIME_STEPS = 32
FEATURE_DIM = 1024
FULL_NUMEL = TIME_STEPS * FEATURE_DIM
OUTER_FOLDS = 5
INNER_FOLDS = 5
CODE_TIME = 12
CODE_CHANNELS = 32
CODE_NUMEL = CODE_TIME * CODE_CHANNELS
MAX_TRAINABLE_PARAMETERS = 150000
EXACT_TRAINABLE_PARAMETERS = 79040
BASELINE_NAME = "clip_pca_b0384_t01_r384"
RHO_GRID = (1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0,
            1.0 / 4.0, 1.0 / 2.0, 1.0)
RHO_COMPARATOR = 0.0
TRAINING_RHO = 1.0
FIXED_SELECTED_STEP = 1200
INNER_SPLIT_NAMESPACE = "v4d-vjepa2-inner-family-sha256-round-robin-v1"
V4A_RECEIPT_FILE_SHA256 = "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2"
V4A_RECEIPT_SELFDIGEST = "f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86"
V4A_IMPLEMENTATION_SHA256 = "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973"
V4C_IMPLEMENTATION_SHA256 = "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef"
EXTRACTOR_IMPLEMENTATION_SHA256 = "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc"
V4C_FEATURE_RECEIPT_SHA256 = "895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a"
V4C_FRONTIER_RECEIPT_SHA256 = "8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9"
V4C_FRONTIER_RECEIPT_DIGEST = "376a98dc74e30ab80a277c8866028677d56ba894073d195612a0edb0bbd74f17"
V4D_IMPLEMENTATION_SHA256 = "20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc"
V4D_RECEIPT_SHA256 = "53910bcb71ce02a193bd47e44c3a97de0ee24f431576db64a763637447720b6f"
V4D_RECEIPT_DIGEST = "45d2ae7c45f1db8ccee9b14ba8a7543cfd1ff0d311128472ae116d6befa92f9c"
V4D_RECEIPT_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/semantic_anchor_vjepa2_nonlinear_codec_v4d_20260820/"
    "runs/exact5_20934925_v2/receipt.json"
)
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
    checkpoint_steps: tuple[int, ...] = (0, 1200)
    bootstrap_draws: int = 10000
    bootstrap_alpha: float = 0.05
    teacher_retention: float = 0.8
    recon_ratio_limit: float = 1.05
    geometry_weight: float = 0.25

    def validate(self) -> None:
        if self != Config():
            raise ValueError("v4-F configuration is immutable")
        if self.checkpoint_steps[0] != 0 or self.checkpoint_steps[-1] != self.max_steps:
            raise ValueError("fixed checkpoint schedule does not span full budget")
        if CODE_NUMEL != 384 or (CODE_TIME, CODE_CHANNELS) != (12, 32):
            raise ValueError("actual code payload differs")
        if (
            self.max_steps != FIXED_SELECTED_STEP
            or len(RHO_GRID) != 7
            or RHO_GRID != (
                1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0,
                1.0 / 4.0, 1.0 / 2.0, 1.0,
            )
            or len(set(RHO_GRID)) != len(RHO_GRID)
            or any(type(rho) is not float for rho in RHO_GRID)
            or any(
                not left < right
                for left, right in zip(RHO_GRID, RHO_GRID[1:])
            )
            or not all(0.0 < rho <= 1.0 for rho in RHO_GRID)
            or any(math.frexp(rho)[0] != 0.5 for rho in RHO_GRID)
            or any(
                float(torch.tensor(rho, dtype=torch.float32).item()) != rho
                for rho in RHO_GRID
            )
            or RHO_COMPARATOR in RHO_GRID
            or TRAINING_RHO != 1.0
        ):
            raise ValueError("v4-F fixed-step residual-homotopy contract differs")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _object_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _reject_duplicate_json_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous().clone()
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
    digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    return features.file_sha256(path)


def _require_release_sealed() -> None:
    """Keep execution NO-GO until a detached release flips the build flag.

    Runtime must not reverse-pin its controller or release manifest: those
    detached authorities pin this runtime/tests/tree in one direction and put
    their own final hashes into the run seal, avoiding an impossible SHA cycle.
    """

    if RELEASE_SEALED is not True:
        raise RuntimeError(
            "UNSEALED v4-F residual-homotopy candidate: detached release not sealed"
        )


def _binding() -> dict[str, str]:
    v4e_path = Path(__file__).resolve(strict=True).with_name(
        "semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py"
    )
    paths = {
        "implementation": Path(__file__).resolve(strict=True),
        "v4c_implementation": Path(v4c.__file__).resolve(strict=True),
        "extractor_implementation": Path(features.__file__).resolve(strict=True),
        "v4a_implementation": Path(v4a.__file__).resolve(strict=True),
        "v4d_implementation": Path(v4d.__file__).resolve(strict=True),
        "v4e_burned_implementation": v4e_path.resolve(strict=True),
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
        or result["v4d_implementation_sha256"] != V4D_IMPLEMENTATION_SHA256
        or result["v4e_burned_implementation_sha256"]
        != V4E_BURNED_IMPLEMENTATION_SHA256
    ):
        raise RuntimeError("v4-F frozen implementation dependency differs")
    return result


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("implementation or authority changed during execution")


def _write_json_create_only(path: Path, value: Any) -> str:
    return v4c._write_json_create_only(path, value)["sha256"]


@dataclass(frozen=True)
class ClipPCAFit:
    clip_mean: torch.Tensor
    clip_basis: torch.Tensor
    fit_iid_digest: str
    fit_input_sha256: str
    diagnostics: Mapping[str, Any]


def _fit_clip_pca_b384(rows: Sequence[v4c.Record]) -> ClipPCAFit:
    """Fit the fixed step-0 clip-PCA comparator from model-fit originals only."""

    iids = [row.iid for row in rows]
    if not rows or len(set(iids)) != len(iids):
        raise ValueError("fit-only clip-PCA IID closure differs")
    values = torch.stack([
        v4c.canonical_action(row.views["original"]) for row in rows
    ]).to(dtype=torch.float32, device="cpu")
    if tuple(values.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("fit-only clip-PCA tensor geometry differs")
    flat = values.flatten(1)
    clip_mean = flat.mean(dim=0, keepdim=True)
    clip_basis = v4a._fit_clip_basis(flat - clip_mean, CODE_NUMEL)
    if (
        tuple(clip_mean.shape) != (1, FULL_NUMEL)
        or tuple(clip_basis.shape) != (FULL_NUMEL, CODE_NUMEL)
        or not bool(torch.isfinite(clip_mean).all())
        or not bool(torch.isfinite(clip_basis).all())
    ):
        raise RuntimeError("fit-only clip-PCA basis closure differs")
    diagnostics = {
        "fit_original_count": len(rows),
        "fit_original_only": True,
        "fit_derived_view_count": 0,
        "fit_family_or_transform_labels": False,
        "clip_mean_shape": list(clip_mean.shape),
        "clip_mean_sha256": _tensor_sha(clip_mean),
        "clip_basis_shape": list(clip_basis.shape),
        "clip_basis_sha256": _tensor_sha(clip_basis),
        "clip_basis_max_orthogonality_error": v4a._orthogonality_error(clip_basis),
        "clip_rank": CODE_NUMEL,
        "actual_code_numel": CODE_NUMEL,
        "orthogonal_projection": True,
        "variance_whitening": False,
    }
    return ClipPCAFit(
        clip_mean=clip_mean.contiguous(),
        clip_basis=clip_basis.contiguous(),
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
        raise ValueError("v4-F outer population closure differs")
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
        raise ValueError("v4-F outer-train family closure differs")
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
        raise ValueError("v4-F model-fit/inner/OOF split closure differs")
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
            raise ValueError("v4-F frozen real exact644 inner split differs")
        split["frozen_real_exact644_literal_match"] = True
    else:
        split["frozen_real_exact644_literal_match"] = False
    return groups, split


class CrossAttention32(nn.Module):
    """Pinned single-head 32-channel cross-attention without hidden defaults."""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(CODE_CHANNELS, CODE_CHANNELS)
        self.key = nn.Linear(CODE_CHANNELS, CODE_CHANNELS)
        self.value = nn.Linear(CODE_CHANNELS, CODE_CHANNELS)
        self.output = nn.Linear(CODE_CHANNELS, CODE_CHANNELS)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if (
            query.ndim != 3 or context.ndim != 3 or query.shape[0] == 0
            or query.shape[0] != context.shape[0]
            or query.shape[-1] != CODE_CHANNELS
            or context.shape[-1] != CODE_CHANNELS
        ):
            raise ValueError("cross-attention geometry differs")
        logits = torch.matmul(
            self.query(query), self.key(context).transpose(1, 2)
        ) / math.sqrt(CODE_CHANNELS)
        weights = torch.softmax(logits, dim=-1)
        result = self.output(torch.matmul(weights, self.value(context)))
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("cross-attention output is non-finite")
        return result


class ClipPCAInitializedVJepaGlobalCodec(nn.Module):
    """Sole [12,32] code with global encoder and sole-code decoder attention."""

    def __init__(self, fitted: ClipPCAFit, fit_only_rms: torch.Tensor) -> None:
        super().__init__()
        if (tuple(fit_only_rms.shape) != (1,)
                or not bool(torch.isfinite(fit_only_rms).all())
                or float(fit_only_rms) <= 0.0):
            raise ValueError("fit-only global RMS geometry differs")
        if (tuple(fitted.clip_mean.shape) != (1, FULL_NUMEL)
                or tuple(fitted.clip_basis.shape) != (FULL_NUMEL, CODE_NUMEL)):
            raise ValueError("pinned clip-PCA basis geometry differs")
        self.register_buffer("fit_only_rms", fit_only_rms.detach().reshape(1))
        self.register_buffer("clip_mean", fitted.clip_mean.detach())
        self.register_buffer("clip_basis", fitted.clip_basis.detach())
        self.register_buffer(
            "residual_gate_rho", torch.tensor([TRAINING_RHO], dtype=torch.float32)
        )
        self.input_projection = nn.Linear(FEATURE_DIM, CODE_CHANNELS)
        self.input_position = nn.Parameter(torch.zeros(TIME_STEPS, CODE_CHANNELS))
        self.code_queries = nn.Parameter(torch.zeros(CODE_TIME, CODE_CHANNELS))
        self.encoder_attention = CrossAttention32()
        self.encoder_norm = nn.LayerNorm(CODE_CHANNELS)
        self.encoder_delta = nn.Linear(CODE_CHANNELS, CODE_CHANNELS)
        self.time_queries = nn.Parameter(torch.zeros(TIME_STEPS, CODE_CHANNELS))
        self.code_position = nn.Parameter(torch.zeros(CODE_TIME, CODE_CHANNELS))
        self.decoder_attention = CrossAttention32()
        self.decoder_norm = nn.LayerNorm(CODE_CHANNELS)
        self.decoder_output = nn.Linear(CODE_CHANNELS, FEATURE_DIM)
        nn.init.normal_(self.input_position, std=0.02)
        nn.init.normal_(self.code_queries, std=0.02)
        nn.init.normal_(self.time_queries, std=0.02)
        nn.init.normal_(self.code_position, std=0.02)
        nn.init.zeros_(self.encoder_delta.weight)
        nn.init.zeros_(self.encoder_delta.bias)
        nn.init.zeros_(self.decoder_output.weight)
        nn.init.zeros_(self.decoder_output.bias)
        count = sum(parameter.numel() for parameter in self.parameters())
        if count != EXACT_TRAINABLE_PARAMETERS or count >= MAX_TRAINABLE_PARAMETERS:
            raise RuntimeError("v4-F codec parameter closure differs")

    def set_residual_gate_rho(self, rho: float) -> None:
        if (
            type(rho) is not float
            or rho not in (RHO_COMPARATOR, *RHO_GRID)
            or not math.isfinite(rho)
        ):
            raise ValueError("rho must be one exact preregistered FP32 power of two")
        fp32 = torch.tensor([rho], dtype=torch.float32)
        if float(fp32.item()) != rho:
            raise ValueError("rho is not exactly representable as FP32")
        with torch.no_grad():
            self.residual_gate_rho.copy_(fp32.to(self.residual_gate_rho.device))

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        if (
            value.ndim != 3 or value.shape[0] == 0
            or tuple(value.shape[1:]) != (TIME_STEPS, FEATURE_DIM)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("encoder input geometry differs")
        # Every caller must supply the upstream C(view); do not apply C twice,
        # because step 0 must be the exact fit-only fixed clip-PCA encoder.
        if float(value.detach().mean(dim=1).abs().max().cpu()) > 1.0e-5:
            raise ValueError("encoder input is not upstream-temporally-centered C(view)")
        analytic = (
            (value.flatten(1) - self.clip_mean) @ self.clip_basis
        ).reshape(-1, CODE_TIME, CODE_CHANNELS)
        tokens = self.input_projection(value / self.fit_only_rms)
        tokens = tokens + self.input_position.unsqueeze(0)
        queries = self.code_queries.unsqueeze(0).expand(value.shape[0], -1, -1)
        attended = self.encoder_attention(queries, tokens)
        delta = (
            self.encoder_delta(self.encoder_norm(attended))
            * self.fit_only_rms * self.residual_gate_rho
        )
        code = analytic + delta
        if (
            tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS)
            or code[0].numel() != 384 or code.dtype != torch.float32
            or not code.is_contiguous() or not bool(torch.isfinite(code).all())
        ):
            raise RuntimeError("actual code is not [12,32]=384")
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if (
            code.ndim != 3 or code.shape[0] == 0
            or tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS)
            or code.dtype != torch.float32 or not code.is_contiguous()
            or not bool(torch.isfinite(code).all())
        ):
            raise ValueError("decoder input must be the sole [12,32] code")
        analytic = (
            self.clip_mean + code.flatten(1) @ self.clip_basis.T
        ).reshape(-1, TIME_STEPS, FEATURE_DIM)
        queries = self.time_queries.unsqueeze(0).expand(code.shape[0], -1, -1)
        context = code / self.fit_only_rms + self.code_position.unsqueeze(0)
        attended = self.decoder_attention(queries, context)
        residual = self.decoder_output(self.decoder_norm(attended))
        if tuple(residual.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
            raise RuntimeError("decoder residual geometry differs")
        output = (
            analytic
            + residual * self.fit_only_rms * self.residual_gate_rho
        )
        result = (output - output.mean(dim=1, keepdim=True)).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("decoder output is non-finite")
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(value))


VJepa2GlobalCodec = ClipPCAInitializedVJepaGlobalCodec


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


def _single_view_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The frozen v4D reconstruction objective, applied view-symmetrically."""

    if (
        prediction.shape != target.shape or prediction.ndim != 4
        or tuple(prediction.shape[-2:]) != (TIME_STEPS, FEATURE_DIM)
        or prediction.shape[0] == 0 or prediction.shape[1] != len(EVAL_VIEWS)
    ):
        raise ValueError("five-view reconstruction-loss geometry differs")
    def equal_view_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        elementwise = F.smooth_l1_loss(
            left, right, beta=0.1, reduction="none"
        )
        if elementwise.ndim == 4:
            per_view = elementwise.mean(dim=(0, 2, 3))
        elif elementwise.ndim == 3:
            per_view = elementwise.mean(dim=(0, 2))
        else:
            raise RuntimeError("equal-view loss rank differs")
        # Sorting makes the floating-point reduction order independent of a
        # shared permutation of the five view slots.
        return torch.sort(per_view).values.mean()

    raw = equal_view_loss(prediction, target)
    deltas: dict[int, torch.Tensor] = {}
    for stride in (1, 2, 4):
        deltas[stride] = equal_view_loss(
            prediction[:, :, stride:] - prediction[:, :, :-stride],
            target[:, :, stride:] - target[:, :, :-stride],
        )
    terminal = equal_view_loss(
        prediction[:, :, -1] - prediction[:, :, 0],
        target[:, :, -1] - target[:, :, 0],
    )
    total = raw + 0.20 * sum(deltas.values()) + 0.20 * terminal
    return total, {
        "raw_feature": raw,
        "signed_delta_stride1": deltas[1],
        "signed_delta_stride2": deltas[2],
        "signed_delta_stride4": deltas[4],
        "terminal_displacement": terminal,
    }


def _multiview_training_loss(
    prediction: torch.Tensor, target: torch.Tensor, geometry_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equal five-view reconstruction plus all ten unordered teacher distances.

    The computation is exchangeable along the view axis.  It receives no view
    names or semantic roles, and therefore a shared permutation of prediction
    and target leaves every scalar bit-identical on CPU.
    """

    if geometry_weight != 0.25:
        raise ValueError("geometry weight is immutable")
    reconstruction, reconstruction_terms = _single_view_reconstruction_loss(
        prediction, target
    )
    teacher_distances: list[torch.Tensor] = []
    candidate_distances: list[torch.Tensor] = []
    for left in range(len(EVAL_VIEWS)):
        for right in range(left + 1, len(EVAL_VIEWS)):
            teacher_distances.append((
                target[:, left] - target[:, right]
            ).square().mean(dim=(1, 2)))
            candidate_distances.append((
                prediction[:, left] - prediction[:, right]
            ).square().mean(dim=(1, 2)))
    if len(teacher_distances) != 10 or len(candidate_distances) != 10:
        raise RuntimeError("five-view unordered-pair closure differs")
    teacher_geometry = torch.stack(teacher_distances, dim=1)
    candidate_geometry = torch.stack(candidate_distances, dim=1)
    # Canonicalize pair reduction order without using view identities.  The
    # teacher ordering also preserves candidate/teacher pair correspondence.
    teacher_geometry, pair_order = torch.sort(teacher_geometry, dim=1)
    candidate_geometry = torch.gather(candidate_geometry, 1, pair_order)
    per_iid_scale = teacher_geometry.detach().mean(dim=1, keepdim=True) + 1.0e-8
    normalized_error = (candidate_geometry - teacher_geometry) / per_iid_scale
    per_pair_geometry = F.smooth_l1_loss(
        normalized_error, torch.zeros_like(normalized_error), beta=0.1,
        reduction="none",
    )
    geometry = torch.sort(per_pair_geometry, dim=1).values.mean()
    total = reconstruction + geometry_weight * geometry
    if not bool(torch.isfinite(total)):
        raise RuntimeError("five-view loss is non-finite")
    values = {
        name: float(value.detach().cpu())
        for name, value in reconstruction_terms.items()
    }
    values.update({
        "equal_view_reconstruction": float(reconstruction.detach().cpu()),
        "all_ten_unordered_pair_teacher_geometry": float(geometry.detach().cpu()),
        "geometry_per_iid_teacher_mean_stopgrad_scale_min": float(
            per_iid_scale.detach().min().cpu()
        ),
        "geometry_weight": geometry_weight,
        "total": float(total.detach().cpu()),
    })
    return total, values


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise RuntimeError("checkpoint contains non-finite state")
    return state


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return _object_sha({name: _tensor_sha(state[name]) for name in sorted(state)})


def _base_state_sha(state: Mapping[str, torch.Tensor]) -> str:
    """Hash trained weights while deliberately excluding the deployment rho.

    Rho is a separately bound, exact FP32 deployment buffer.  Excluding only
    that one buffer lets the inner scan prove that no trained parameter or
    fitted comparator buffer changed while the seven candidates were scored.
    """

    if set(state).isdisjoint({"residual_gate_rho"}):
        raise ValueError("residual-gate buffer is absent from codec state")
    return _object_sha({
        name: _tensor_sha(state[name])
        for name in sorted(state) if name != "residual_gate_rho"
    })


@torch.no_grad()
def _step0_equivalence(
    model: VJepa2GlobalCodec, values: torch.Tensor,
    fitted: ClipPCAFit, batch_size: int,
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
        reference_code = _analytic_clip_pca_encode(batch, fitted)
        if not torch.equal(actual_code, reference_code):
            raise RuntimeError("step-0 codec code is not exact clip-PCA-B384")
        actual = model.decode(actual_code)
        reference = _analytic_clip_pca_decode_from_code(reference_code, fitted)
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
        raise RuntimeError("step-0 global codec is not the analytic clip-PCA comparator")
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
    inner_validation_iids: Sequence[str],
    fitted: ClipPCAFit, config: Config, fold_index: int,
    device: torch.device,
) -> tuple[VJepa2GlobalCodec, int, dict[str, Any]]:
    """Run the frozen v4-E objective at rho=1 and fix step 1200 a priori.

    ``inner_validation_iids`` is metadata only.  No inner tensor is accepted by
    this function, which makes the preselection checkpoint independent of the
    later known-transform-exposed rho selection.
    """

    seed = config.seed + 10000 + fold_index
    _seed_everything(seed, device)
    fit_views = torch.stack([
        torch.stack([v4c.canonical_action(row.views[name]) for name in EVAL_VIEWS])
        for row in model_fit
    ]).to(device)
    if tuple(fit_views.shape[1:]) != (
        len(EVAL_VIEWS), TIME_STEPS, FEATURE_DIM
    ):
        raise ValueError("model-fit exposed-five-view tensor geometry differs")
    original_index = EVAL_VIEWS.index("original")
    fit_original = fit_views[:, original_index]
    rms = _fit_only_global_rms(model_fit, device)
    model = VJepa2GlobalCodec(fitted, rms).to(device)
    model.set_residual_gate_rho(TRAINING_RHO)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (
        trainable_parameters != EXACT_TRAINABLE_PARAMETERS
        or trainable_parameters >= MAX_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("trainable parameter gate differs")
    fit_step0 = _step0_equivalence(
        model, fit_views.flatten(0, 1), fitted, config.batch_size
    )

    checkpoint_states: dict[int, dict[str, torch.Tensor]] = {0: _state_to_cpu(model)}
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
        target = fit_views.index_select(0, indices)
        flat_target = target.flatten(0, 1)
        prediction = model(flat_target).reshape_as(target)
        loss, last_components = _multiview_training_loss(
            prediction, target, config.geometry_weight
        )
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
            model.train()
    if set(checkpoint_states) != set(config.checkpoint_steps) or last_components is None:
        raise RuntimeError("full fixed training/checkpoint budget did not close")
    selected_step = FIXED_SELECTED_STEP
    model.load_state_dict(checkpoint_states[selected_step], strict=True)
    model.to(device).eval()
    audit = {
        "fold_seed": seed,
        "full_budget_steps_executed": config.max_steps,
        "early_stopped": False,
        "checkpoint_steps": list(config.checkpoint_steps),
        "inner_validation_tensor_count_before_preselection_checkpoint_seal": 0,
        "inner_validation_metric_computed_before_preselection_checkpoint_seal": False,
        "selection_rule": "fixed preregistered final step 1200; no checkpoint winner selection",
        "checkpoint_winner_selection_performed": False,
        "selected_step": selected_step,
        "selected_state_sha256": _state_sha(checkpoint_states[selected_step]),
        "selected_base_state_sha256": _base_state_sha(checkpoint_states[selected_step]),
        "step0_state_sha256": _state_sha(checkpoint_states[0]),
        "final_step_state_sha256": _state_sha(checkpoint_states[config.max_steps]),
        "final_step_base_state_sha256": _base_state_sha(
            checkpoint_states[config.max_steps]
        ),
        "last_training_loss_components": last_components,
        "trainable_parameter_count": trainable_parameters,
        "trainable_parameter_limit_exclusive": MAX_TRAINABLE_PARAMETERS,
        "fit_only_global_rms": float(rms.detach().cpu()),
        "fit_only_global_rms_sha256": _tensor_sha(rms),
        "step0_model_fit_all_five_views_equivalence": fit_step0,
        "training_residual_gate_rho": TRAINING_RHO,
        "training_residual_gate_rho_fp32_exact": True,
        "zero_initialized_encoder_and_decoder_residual_final_layers": True,
        "minibatch_schedule_shape": list(minibatch_schedule.shape),
        "minibatch_schedule_sha256": _tensor_sha(minibatch_schedule),
        "minibatch_schedule_definition": "all fixed model-fit row indices in executed step order",
        "model_fit_original_count": len(model_fit),
        "model_fit_ordered_iids": [row.iid for row in model_fit],
        "model_fit_iid_digest": _object_sha([row.iid for row in model_fit]),
        "model_fit_original_tensor_sha256": _tensor_sha(fit_original),
        "model_fit_all_five_views_tensor_sha256": _tensor_sha(fit_views),
        "model_fit_exposed_training_view_rows": len(model_fit) * len(EVAL_VIEWS),
        "exposed_training_views_per_original": len(EVAL_VIEWS),
        "model_fit_five_view_tensors_used_for_gradient_and_model_input": True,
        "model_fit_transform_role_family_teacher_or_pca_metadata_used_for_gradient_or_model_input": False,
        "derived_training_view_tensor_count": len(model_fit) * 4,
        "independent_sample_count_increment_from_derived_views": 0,
        "model_fit_iid_cluster_count": len(model_fit),
        "all_five_views_equal_reconstruction_weight": True,
        "all_ten_unordered_view_pairs_equal_geometry_weight": True,
        "view_axis_permutation_invariant_loss": True,
        "inner_validation_original_count": len(inner_validation_iids),
        "inner_validation_ordered_iids": list(inner_validation_iids),
        "inner_validation_iid_digest": _object_sha(list(inner_validation_iids)),
        "inner_validation_five_view_tensor_count_used_during_training": 0,
        "inner_validation_derived_view_tensor_count_used_during_training": 0,
        "inner_validation_any_view_used_during_gradient_or_preselection_checkpoint_selection": False,
        "inner_transform_role_family_teacher_or_pca_metadata_used_during_training": False,
        "oof_tensors_supplied_to_optimizer_or_checkpoint_selection": False,
        "known_negative_view_tensor_rows_used": len(model_fit) * 3,
        "known_positive_derived_view_tensor_rows_used": len(model_fit),
        "view_name_or_positive_negative_role_labels_used": 0,
        "family_or_transform_labels_entered_loss_or_model_input": False,
        "family_metadata_used_for_inner_split": True,
        "transform_metadata_used_for_inner_split": False,
    }
    return model, selected_step, audit


def _analytic_clip_pca_encode(value: torch.Tensor, fitted: ClipPCAFit) -> torch.Tensor:
    batched = value.unsqueeze(0) if value.ndim == 2 else value
    if batched.ndim != 3 or tuple(batched.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("analytic clip-PCA encoder input geometry differs")
    clip_mean = fitted.clip_mean.to(batched.device)
    clip_basis = fitted.clip_basis.to(batched.device)
    code = ((batched.flatten(1) - clip_mean) @ clip_basis).reshape(
        -1, CODE_TIME, CODE_CHANNELS
    )
    if tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS):
        raise RuntimeError("analytic clip-PCA-B384 code geometry differs")
    return code[0] if value.ndim == 2 else code


def _analytic_clip_pca_decode_from_code(
    code: torch.Tensor, fitted: ClipPCAFit,
) -> torch.Tensor:
    unbatched = code.ndim == 2
    batched = code.unsqueeze(0) if unbatched else code
    if tuple(batched.shape[1:]) != (CODE_TIME, CODE_CHANNELS):
        raise ValueError("analytic clip-PCA-B384 decoder code geometry differs")
    clip_mean = fitted.clip_mean.to(batched.device)
    clip_basis = fitted.clip_basis.to(batched.device)
    decoded = (clip_mean + batched.flatten(1) @ clip_basis.T).reshape(
        -1, TIME_STEPS, FEATURE_DIM
    )
    decoded = _canonical_batch(decoded)
    return decoded[0] if unbatched else decoded


def _analytic_clip_pca_decode(value: torch.Tensor, fitted: ClipPCAFit) -> torch.Tensor:
    """Decode fixed clip-PCA-B384 and apply output C in raw coordinates."""

    return _analytic_clip_pca_decode_from_code(
        _analytic_clip_pca_encode(value, fitted), fitted
    )


@torch.no_grad()
def _model_decode_batches(
    model: VJepa2GlobalCodec, values: torch.Tensor, batch_size: int,
) -> torch.Tensor:
    output = []
    model.eval()
    for start in range(0, len(values), batch_size):
        output.append(model(values[start:start + batch_size]))
    result = torch.cat(output, dim=0)
    if result.shape != values.shape or not bool(torch.isfinite(result).all()):
        raise RuntimeError("decoded candidate output differs")
    return result


def _evaluate_rows_at_rho(
    rows: Sequence[v4c.Record], model: VJepa2GlobalCodec, rho: float,
    fitted: ClipPCAFit, config: Config, device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate one already-materialized fold-local population at fixed rho."""

    if rho not in (RHO_COMPARATOR, *RHO_GRID):
        raise ValueError("evaluation rho is not preregistered")
    if not rows or any(set(row.views) != set(EVAL_VIEWS) for row in rows):
        raise ValueError("v4-F evaluation five-view closure differs")
    stacked_cpu = {
        name: torch.stack([v4c.canonical_action(row.views[name]) for row in rows])
        for name in EVAL_VIEWS
    }
    stacked = {name: values.to(device) for name, values in stacked_cpu.items()}
    baseline = {
        name: _analytic_clip_pca_decode(values, fitted) for name, values in stacked.items()
    }
    model.set_residual_gate_rho(float(rho))
    if rho == RHO_COMPARATOR:
        # Exact alias is contractual: no floating-point discrepancy may create
        # a fictitious rho-zero improvement over the analytic baseline.
        candidate = baseline
        comparator_alias_used = True
    else:
        candidate = {
            name: _model_decode_batches(model, values, config.batch_size)
            for name, values in stacked.items()
        }
        comparator_alias_used = False
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
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
                "clip_pca_b384_raw_mse": float(_raw_mse(
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
            "clip_pca_b384_margin_by_negative": {
                name: baseline_margin[name]["margin"] for name in NEGATIVES
            },
            "candidate_margin_by_negative": {
                name: candidate_margin[name]["margin"] for name in NEGATIVES
            },
            "raw_reconstruction_by_view": reconstruction,
            "residual_gate_rho": rho,
            "rho0_exact_clip_pca_alias_used": comparator_alias_used,
        })
    return output


def _evaluate_fold(
    oof_rows: Sequence[v4c.Record], model: VJepa2GlobalCodec,
    selected_rho: float, fitted: ClipPCAFit, config: Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    return _evaluate_rows_at_rho(
        oof_rows, model, selected_rho, fitted, config, device
    )


def _fixed_inner_reference_projection(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project the teacher and fixed-PCA fields that must not vary with rho."""

    return [{
        "iid": row.get("iid"),
        "family": row.get("family"),
        "teacher_margin_by_negative": row.get("teacher_margin_by_negative"),
        "clip_pca_b384_margin_by_negative": row.get(
            "clip_pca_b384_margin_by_negative"
        ),
        "clip_pca_b384_raw_mse_by_view": {
            view: row.get("raw_reconstruction_by_view", {}).get(view, {}).get(
                "clip_pca_b384_raw_mse"
            )
            for view in EVAL_VIEWS
        },
    } for row in evidence]


def _paired_lcb(
    values: Sequence[float], families: Sequence[str], config: Config, label: str,
    *, namespace: str = "v4e",
) -> dict[str, Any]:
    if not values or len(values) != len(families):
        raise ValueError("paired bootstrap population differs")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("paired bootstrap values are non-finite")
    family_names = sorted(set(families))
    if not family_names:
        raise ValueError("paired bootstrap has no family clusters")
    tensor = torch.tensor(values, dtype=torch.float64)
    clip_seed = v4a._bootstrap_seed(config, f"{namespace}:{label}", "clip")
    clip_generator = torch.Generator().manual_seed(clip_seed)
    clip_indices = torch.randint(
        len(tensor), (config.bootstrap_draws, len(tensor)), generator=clip_generator
    )
    family_means = torch.tensor([
        sum(float(value) for value, family in zip(values, families) if family == name)
        / sum(family == name for family in families)
        for name in family_names
    ], dtype=torch.float64)
    family_seed = v4a._bootstrap_seed(config, f"{namespace}:{label}", "family")
    family_generator = torch.Generator().manual_seed(family_seed)
    family_indices = torch.randint(
        len(family_names), (config.bootstrap_draws, len(family_names)),
        generator=family_generator,
    )
    result = {
        "paired_original_count": len(values),
        "clip_micro_point_mean": float(tensor.mean()),
        "family_macro_point_mean": float(family_means.mean()),
        "clip_paired_bootstrap": {
            "draws": config.bootstrap_draws, "seed": clip_seed,
            "one_sided_alpha": config.bootstrap_alpha,
            "lcb": float(torch.quantile(
                tensor[clip_indices].mean(dim=1), config.bootstrap_alpha
            )),
        },
        "family_cluster_paired_bootstrap": {
            "cluster_count": len(family_names), "draws": config.bootstrap_draws,
            "seed": family_seed, "one_sided_alpha": config.bootstrap_alpha,
            "cluster_resampling": (
                "compute_each_family_clip_mean_then_resample_family_means_"
                "with_equal_weight"
            ),
            "equal_family_weight": True,
            "lcb": float(torch.quantile(
                family_means[family_indices].mean(dim=1), config.bootstrap_alpha
            )),
        },
    }
    result["both_lcbs_strictly_gt_zero"] = bool(
        result["clip_paired_bootstrap"]["lcb"] > 0.0
        and result["family_cluster_paired_bootstrap"]["lcb"] > 0.0
    )
    return result


def _paired_ratio_ucb(
    candidate_errors: Sequence[float], baseline_errors: Sequence[float],
    families: Sequence[str], config: Config, label: str, *,
    namespace: str = "v4e",
) -> dict[str, Any]:
    """Paired ratio-of-means bootstrap; never mean of per-IID ratios."""

    count = len(candidate_errors)
    if count == 0 or len(baseline_errors) != count or len(families) != count:
        raise ValueError("paired ratio bootstrap population differs")
    if (any(not math.isfinite(float(value)) or float(value) < 0.0 for value in candidate_errors)
            or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in baseline_errors)):
        raise ValueError("paired ratio errors are non-finite or denominator is non-positive")
    family_names = sorted(set(families))
    if not family_names:
        raise ValueError("paired ratio family closure is empty")
    candidate = torch.tensor(candidate_errors, dtype=torch.float64)
    baseline = torch.tensor(baseline_errors, dtype=torch.float64)

    clip_seed = v4a._bootstrap_seed(
        config, namespace, label, "ratio", "clip"
    )
    clip_generator = torch.Generator().manual_seed(clip_seed)
    clip_indices = torch.randint(
        count, (config.bootstrap_draws, count), generator=clip_generator
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
    family_seed = v4a._bootstrap_seed(
        config, namespace, label, "ratio", "family"
    )
    family_generator = torch.Generator().manual_seed(family_seed)
    family_indices = torch.randint(
        len(family_names), (config.bootstrap_draws, len(family_names)),
        generator=family_generator
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
        "paired_original_count": count,
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
            "cluster_count": len(family_names),
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


def _positive_point_and_lcb_gate(statistics: Mapping[str, Any]) -> bool:
    return bool(
        statistics["clip_micro_point_mean"] > 0.0
        and statistics["family_macro_point_mean"] > 0.0
        and statistics["both_lcbs_strictly_gt_zero"]
    )


def _inner_candidate_gate(
    evidence: Sequence[Mapping[str, Any]], config: Config, *,
    fold_index: int, rho_ordinal: int,
) -> dict[str, Any]:
    """Apply the complete preregistered gate to one fold-local rho candidate."""

    if (
        not evidence or len({str(row["iid"]) for row in evidence}) != len(evidence)
        or not 0 <= fold_index < OUTER_FOLDS
        or not 0 <= rho_ordinal < len(RHO_GRID)
    ):
        raise ValueError("inner candidate population or ordinal differs")
    families = [str(row["family"]) for row in evidence]
    prefix = f"inner:fold{fold_index}:rho{rho_ordinal:02d}"
    fidelity: dict[str, Any] = {}
    for view in EVAL_VIEWS:
        candidate_errors = [
            float(row["raw_reconstruction_by_view"][view]["candidate_raw_mse"])
            for row in evidence
        ]
        baseline_errors = [
            float(row["raw_reconstruction_by_view"][view]["clip_pca_b384_raw_mse"])
            for row in evidence
        ]
        statistics = _paired_ratio_ucb(
            candidate_errors, baseline_errors, families, config,
            f"{prefix}:recon:{view}", namespace="v4f",
        )
        statistics["both_point_ratios_le_1p05"] = bool(
            statistics["clip_micro_point_ratio"] <= config.recon_ratio_limit
            and statistics["family_macro_point_ratio"] <= config.recon_ratio_limit
        )
        statistics["inner_view_gate"] = bool(
            statistics["both_ucbs_le_1p05"]
            and statistics["both_point_ratios_le_1p05"]
        )
        fidelity[view] = statistics
    fidelity_gate = all(fidelity[view]["inner_view_gate"] for view in EVAL_VIEWS)

    negatives: dict[str, Any] = {}
    for negative in NEGATIVES:
        teacher_values = [
            float(row["teacher_margin_by_negative"][negative]) for row in evidence
        ]
        baseline_values = [
            float(row["clip_pca_b384_margin_by_negative"][negative])
            for row in evidence
        ]
        candidate_values = [
            float(row["candidate_margin_by_negative"][negative])
            for row in evidence
        ]
        retention_values = [
            candidate - config.teacher_retention * teacher
            for candidate, teacher in zip(candidate_values, teacher_values)
        ]
        improvement_values = [
            candidate - baseline
            for candidate, baseline in zip(candidate_values, baseline_values)
        ]
        teacher = _paired_lcb(
            teacher_values, families, config,
            f"inner:fold{fold_index}:teacher-fixed:{negative}",
            namespace="v4f",
        )
        candidate = _paired_lcb(
            candidate_values, families, config, f"{prefix}:candidate:{negative}",
            namespace="v4f",
        )
        retention = _paired_lcb(
            retention_values, families, config,
            f"{prefix}:candidate-minus-0p8-teacher:{negative}", namespace="v4f",
        )
        improvement = _paired_lcb(
            improvement_values, families, config,
            f"{prefix}:candidate-minus-clip-pca:{negative}", namespace="v4f",
        )
        gate = bool(
            _positive_point_and_lcb_gate(teacher)
            and _positive_point_and_lcb_gate(candidate)
            and _positive_point_and_lcb_gate(retention)
            and _positive_point_and_lcb_gate(improvement)
        )
        negatives[negative] = {
            "teacher_fixed_gate_included": True,
            "teacher_margin": teacher,
            "candidate_margin": candidate,
            "candidate_minus_0p8_teacher_margin": retention,
            "candidate_minus_fixed_clip_pca_b384_margin": improvement,
            "all_four_clip_and_family_point_means_and_lcbs_strictly_gt_zero": gate,
            "inner_negative_gate": gate,
        }
    negative_gate = all(negatives[name]["inner_negative_gate"] for name in NEGATIVES)
    return {
        "population_count": len(evidence),
        "family_cluster_count": len(set(families)),
        "five_view_raw_reconstruction_ratio_vs_fixed_clip_pca_b384": fidelity,
        "five_view_fidelity_gate": fidelity_gate,
        "negative_results": negatives,
        "all_three_negative_full_gates": negative_gate,
        "complete_candidate_dependent_inner_gate": bool(fidelity_gate and negative_gate),
        "teacher_fixed_gate_included": True,
        "aggregate_or_cross_negative_compensation_allowed": False,
    }


def _bootstrap_seed_ledger(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if type(value) is dict:
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            if key == "seed":
                output.append({"path": child, "seed": value[key]})
            else:
                output.extend(_bootstrap_seed_ledger(value[key], child))
    elif type(value) is list:
        for index, item in enumerate(value):
            output.extend(_bootstrap_seed_ledger(item, f"{prefix}[{index}]"))
    return output


@torch.no_grad()
def _select_fold_local_rho(
    inner_rows: Sequence[v4c.Record], model: VJepa2GlobalCodec,
    fitted: ClipPCAFit, config: Config, fold_index: int, device: torch.device,
) -> tuple[float | None, dict[str, Any]]:
    """Evaluate all exact-seven candidates and choose first passing by ordinal."""

    initial_state = _state_to_cpu(model)
    initial_base_sha = _base_state_sha(initial_state)
    comparator = _evaluate_rows_at_rho(
        inner_rows, model, RHO_COMPARATOR, fitted, config, device
    )
    if any(
        not row["rho0_exact_clip_pca_alias_used"]
        or any(
            row["raw_reconstruction_by_view"][view]["candidate_raw_mse"]
            != row["raw_reconstruction_by_view"][view]["clip_pca_b384_raw_mse"]
            for view in EVAL_VIEWS
        )
        for row in comparator
    ):
        raise RuntimeError("rho-zero comparator is not exact clip-PCA-B384")
    fixed_reference = _fixed_inner_reference_projection(comparator)
    fixed_reference_sha = _object_sha(fixed_reference)
    candidates: list[dict[str, Any]] = []
    selected_rho: float | None = None
    for ordinal, rho in enumerate(RHO_GRID):
        evidence = _evaluate_rows_at_rho(
            inner_rows, model, float(rho), fitted, config, device
        )
        if _object_sha(_fixed_inner_reference_projection(evidence)) != fixed_reference_sha:
            raise RuntimeError("teacher or fixed-PCA inner reference changed across rho")
        gate = _inner_candidate_gate(
            evidence, config, fold_index=fold_index, rho_ordinal=ordinal
        )
        if _base_state_sha(_state_to_cpu(model)) != initial_base_sha:
            raise RuntimeError("trained base state changed during rho scan")
        passed = bool(gate["complete_candidate_dependent_inner_gate"])
        candidates.append({
            "rho_ordinal": ordinal,
            "rho": rho,
            "rho_fp32_exact_power_of_two": True,
            "inner_evidence_count": len(evidence),
            "inner_evidence_sha256": _object_sha(evidence),
            "inner_evidence": evidence,
            "fixed_teacher_and_clip_pca_reference_sha256": fixed_reference_sha,
            "gate": gate,
            "bootstrap_seed_ledger": _bootstrap_seed_ledger(gate),
            "pass": passed,
        })
        if selected_rho is None and passed:
            selected_rho = rho
    final_base_sha = _base_state_sha(_state_to_cpu(model))
    if final_base_sha != initial_base_sha or len(candidates) != len(RHO_GRID):
        raise RuntimeError("exact-seven rho scan changed base state or did not close")
    if selected_rho is not None:
        model.set_residual_gate_rho(float(selected_rho))
    return selected_rho, {
        "selection_scope": "fold_local_inner_only",
        "outer_fold": fold_index,
        "rho_candidate_count": len(RHO_GRID),
        "single_candidate": False,
        "rho_grid_preregistered_ascending": list(RHO_GRID),
        "rho0_comparator": RHO_COMPARATOR,
        "rho0_selectable": False,
        "rho0_exact_clip_pca_b384_alias": True,
        "rho0_inner_evidence_count": len(comparator),
        "rho0_inner_evidence_sha256": _object_sha(comparator),
        "rho0_inner_evidence": comparator,
        "fixed_teacher_and_clip_pca_reference_sha256": fixed_reference_sha,
        "selection_rule": "evaluate all exact7 then choose first PASS in preregistered ascending order",
        "monotonic_metric_behavior_assumed": False,
        "smallest_rho_minimizes_distortion_claimed": False,
        "candidates": candidates,
        "exact7_candidate_ledger_complete": True,
        "selected_rho": selected_rho,
        "inner_pass": selected_rho is not None,
        "no_pass_action": "INNER_NO_GO and OOF semantic tensor read count exact0",
        "base_state_sha256_before_scan": initial_base_sha,
        "base_state_sha256_after_scan": final_base_sha,
        "base_state_unchanged_across_all_rho_candidates": True,
        "inner_five_view_tensors_used_for_hyperparameter_selection": True,
        "inner_five_view_tensors_used_for_gradient_or_model_input": False,
        "transform_role_and_family_metadata_used_for_hyperparameter_selection": True,
        "transform_role_and_family_metadata_used_for_gradient": False,
        "transform_role_and_family_metadata_used_for_model_input": False,
        "teacher_and_fixed_pca_metadata_used_for_hyperparameter_selection": True,
        "teacher_and_fixed_pca_metadata_used_for_gradient_or_model_input": False,
        "cross_fold_inner_metric_aggregation_or_global_rho_selection": False,
    }


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
            float(row["raw_reconstruction_by_view"][view]["clip_pca_b384_raw_mse"])
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
            float(row["clip_pca_b384_margin_by_negative"][negative]) for row in rows
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
        baseline = _paired_lcb(baseline_values, families, config, f"clip-pca:{negative}")
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
                f"candidate-minus-clip-pca:{negative}",
            ),
            improvement_values, f"candidate-minus-clip-pca:{negative}",
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
            "fixed_clip_pca_b384_margin": baseline,
            "candidate_margin": candidate,
            "candidate_minus_0p8_teacher_margin": retention,
            "candidate_minus_fixed_clip_pca_b384_margin": improvement,
            "all_four_quantities_pass_dual_bootstrap_and_every_fold": gate,
            "decoded_negative_gate": gate,
        }
    all_negative_gates = all(
        negative_results[name]["decoded_negative_gate"] for name in NEGATIVES
    )
    development_gate = bool(five_view_fidelity_gate and all_negative_gates)
    return {
        "five_view_raw_reconstruction_ratio_vs_fixed_clip_pca_b384": fidelity,
        "five_view_fidelity_gate": five_view_fidelity_gate,
        "negative_results": negative_results,
        "all_three_decoded_negative_gates": all_negative_gates,
        "exposed_five_view_codec_development_gate": development_gate,
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


def _load_v4d_burned_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Bind the failed v4D diagnostic as motivation, never as train data."""

    if expected_sha256 != V4D_RECEIPT_SHA256:
        raise ValueError("v4-D receipt SHA is not the frozen burned authority")
    if str(path) != V4D_RECEIPT_PATH:
        raise ValueError("v4-D receipt path is not the frozen burned authority")
    value = v4c._load_json_sealed(path, V4D_RECEIPT_SHA256)
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if (
        value.get("schema_version") != v4d.SCHEMA
        or value.get("status") != v4d.STATUS
        or digest != V4D_RECEIPT_DIGEST
        or _object_sha(unsigned) != digest
        or value.get("implementation", {}).get("implementation_sha256")
            != V4D_IMPLEMENTATION_SHA256
        or value.get("metrics", {}).get(
            "decoded_temporal_codec_development_gate"
        ) is not False
        or value.get("qualification_scope", {}).get("inference_authorized") is not False
        or value.get("qualification_scope", {}).get("video_editing_qualified") is not False
    ):
        raise ValueError("sealed v4-D burned authority differs")
    return value


def _verify_v4c_embedded_teacher_evidence(
    rows: Sequence[Mapping[str, Any]], v4c_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    upstream_rows = v4c_receipt["oof_closure"]["embedded_paired_margin_evidence"]
    upstream = {row["iid"]: row for row in upstream_rows}
    if len(upstream) != 644 or set(upstream) != {row["iid"] for row in rows}:
        raise ValueError("v4-C/v4-F exact644 IID evidence closure differs")
    max_teacher = 0.0
    for row in rows:
        reference = upstream[row["iid"]]
        if (reference["family"] != row["family"]
                or int(reference["outer_fold"]) != int(row["outer_fold"])):
            raise ValueError("v4-C/v4-F family or fold authority differs")
        for negative in NEGATIVES:
            teacher_difference = abs(
                float(row["teacher_margin_by_negative"][negative])
                - float(reference["teacher_margin_by_negative"][negative])
            )
            max_teacher = max(max_teacher, teacher_difference)
    if max_teacher > 1.0e-12:
        raise ValueError("v4-C/v4-F teacher evidence differs")
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
    device: torch.device, preselection_path: Path, checkpoint_path: Path,
    run_binding: Mapping[str, str], feature_index: Mapping[str, Any],
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

    # Stage 1 is intentionally model-fit only.  Neither inner nor OOF tensor
    # storage may be read before the fixed-step preselection checkpoint seals.
    stage1_request = {iid: EVAL_VIEWS for iid in fit_iids}
    stage1_rows, stage1_audit = _selective_materialize_feature_rows(
        feature_index, stage1_request, stage="stage1_model_fit_only"
    )
    model_fit_rows = [stage1_rows[iid] for iid in fit_iids]
    expected_stage1_count = len(fit_iids) * len(EVAL_VIEWS)
    if (
        stage1_audit["semantic_tensor_materialized_count"] != expected_stage1_count
        or any(
            stage1_audit["semantic_tensor_materialized_count_by_view"][view]
                != len(fit_iids)
            for view in EVAL_VIEWS
        )
    ):
        raise RuntimeError("stage1 selective tensor read closure differs")

    fitted = _fit_clip_pca_b384(model_fit_rows)
    if fitted.fit_iid_digest != split["model_fit_iid_digest"]:
        raise RuntimeError("fit-only clip-PCA/model-fit IID join differs")
    model, selected_step, training_audit = _train_fold_model(
        model_fit_rows, validation_iids, fitted,
        config, fold_index, device,
    )
    if selected_step != FIXED_SELECTED_STEP:
        raise RuntimeError("preselection training did not fix step 1200")
    preselection = _save_checkpoint_create_only(
        preselection_path, model, fitted, training_audit, config, fold_index,
        run_binding, model_fit_rows, device,
        checkpoint_role="preselection_fixed_step1200",
        deployment_rho=TRAINING_RHO, preselection_artifact=None,
    )
    if preselection["preselection_base_state_sha256"] != training_audit[
        "final_step_base_state_sha256"
    ]:
        raise RuntimeError("preselection checkpoint/trained base join differs")

    # Stage 2 is the first materialization of any inner tensor, and contains
    # all five views because transform-role/family metadata enters the honest
    # fold-local hyperparameter gate (never the gradient or model input).
    stage2_rows, stage2_audit = _selective_materialize_feature_rows(
        feature_index, {iid: EVAL_VIEWS for iid in validation_iids},
        stage="stage2_post_preselection_seal_inner_five_views",
    )
    if (
        stage2_audit["semantic_tensor_materialized_count"]
            != len(validation_iids) * len(EVAL_VIEWS)
        or any(
            stage2_audit["semantic_tensor_materialized_count_by_view"][view]
                != len(validation_iids) for view in EVAL_VIEWS
        )
    ):
        raise RuntimeError("stage2 selective inner tensor read closure differs")
    inner_rows = [stage2_rows[iid] for iid in validation_iids]
    selected_rho, rho_selection = _select_fold_local_rho(
        inner_rows, model, fitted, config, fold_index, device
    )

    checkpoint: dict[str, Any] | None = None
    checkpoint_pair_join: dict[str, Any] | None = None
    evaluation: list[dict[str, Any]] = []
    if selected_rho is None:
        stage3_audit = {
            "stage": "stage3_inner_no_go_oof_unread",
            "requested_iid_cluster_count": 0,
            "semantic_tensor_materialized_count": 0,
            "semantic_tensor_materialized_count_by_view": {
                view: 0 for view in EVAL_VIEWS
            },
            "requested_iid_view_map_sha256": _object_sha({}),
            "unrequested_tensor_storage_materialized_count": 0,
            "oof_semantic_tensor_read_count_exact0": True,
        }
        fold_status = INNER_NO_GO_STATUS
    else:
        # Freeze rho in its own selected artifact and strong-reload it before
        # the first OOF semantic tensor is requested.
        checkpoint = _save_checkpoint_create_only(
            checkpoint_path, model, fitted, training_audit, config, fold_index,
            run_binding, inner_rows, device,
            checkpoint_role="selected_fold_local_rho",
            deployment_rho=float(selected_rho),
            preselection_artifact=preselection,
        )
        if (
            checkpoint["preselection_base_state_sha256"]
                != preselection["preselection_base_state_sha256"]
            or checkpoint["preselection_checkpoint_file_sha256"]
                != preselection["file_sha256"]
        ):
            raise RuntimeError("selected checkpoint/preselection strong join differs")
        checkpoint_pair_join = _verify_distinct_checkpoint_pair(
            preselection, checkpoint, float(selected_rho)
        )
        stage3_rows, stage3_audit = _selective_materialize_feature_rows(
            feature_index, {iid: EVAL_VIEWS for iid in oof_iids},
            stage="stage3_post_selected_seal_oof",
        )
        if (
            stage3_audit["semantic_tensor_materialized_count"]
                != len(oof_iids) * len(EVAL_VIEWS)
            or any(
                stage3_audit["semantic_tensor_materialized_count_by_view"][view]
                    != len(oof_iids) for view in EVAL_VIEWS
            )
        ):
            raise RuntimeError("stage3 selective OOF tensor read closure differs")
        oof_rows = [stage3_rows[iid] for iid in oof_iids]
        evaluation = _evaluate_fold(
            oof_rows, model, float(selected_rho), fitted, config, device
        )
        for row in evaluation:
            row["outer_fold"] = fold_index
        if [row["iid"] for row in evaluation] != oof_iids:
            raise ValueError("OOF evaluation order differs")
        fold_status = STATUS
    fold_receipt = {
        "fold_status": fold_status,
        "fold_index": fold_index,
        "frozen_v4a_fold_iid_digest": v4c.FOLD_IID_DIGESTS[fold_index],
        "frozen_v4a_outer_assignment_digest": split["outer_assignment_digest"],
        "frozen_v4a_oof_iid_digest": upstream_fold["oof_iid_digest"],
        "inner_split": split,
        "model_fit_original_count": len(fit_iids),
        "model_fit_ordered_iids": fit_iids,
        "model_fit_iid_digest": _object_sha(fit_iids),
        "inner_validation_original_count": len(validation_iids),
        "inner_validation_ordered_iids": validation_iids,
        "inner_validation_iid_digest": _object_sha(validation_iids),
        "oof_original_count": len(oof_iids),
        "oof_ordered_iids": oof_iids,
        "oof_iid_digest": _object_sha(oof_iids),
        "partition_pairwise_disjoint": True,
        "selective_feature_materialization": {
            "stage1_before_checkpoint_seal": stage1_audit,
            "stage2_only_after_preselection_checkpoint_strong_seal_reload": stage2_audit,
            "stage3_only_after_selected_checkpoint_strong_seal_reload_or_no_go": stage3_audit,
            "stage1_model_fit_iid_cluster_count": len(fit_iids),
            "stage1_inner_validation_iid_cluster_count": 0,
            "stage1_original_tensor_count": len(fit_iids),
            "stage1_each_derived_view_tensor_count": len(fit_iids),
            "stage1_oof_semantic_tensor_count": 0,
            "stage1_inner_any_view_semantic_tensor_count": 0,
            "stage2_inner_iid_cluster_count": len(validation_iids),
            "stage2_each_view_tensor_count": len(validation_iids),
            "stage2_model_fit_or_oof_semantic_tensor_count": 0,
            "stage3_oof_iid_cluster_count": len(oof_iids) if selected_rho is not None else 0,
            "stage3_each_view_tensor_count": len(oof_iids) if selected_rho is not None else 0,
            "stage3_model_fit_or_inner_semantic_tensor_count": 0,
            "oof_first_semantic_materialization_after_selected_checkpoint_seal": (
                selected_rho is not None
            ),
            "inner_no_go_oof_semantic_tensor_read_count_exact0": (
                selected_rho is None
            ),
        },
        "fixed_clip_pca_b384_fit_input_sha256": fitted.fit_input_sha256,
        "fixed_clip_pca_b384_fit_iid_digest": fitted.fit_iid_digest,
        "fixed_clip_pca_b384_diagnostics": fitted.diagnostics,
        "training": training_audit,
        "preselection_checkpoint_artifact": preselection,
        "rho_selection": rho_selection,
        "rho_candidate_count": len(RHO_GRID),
        "single_candidate": False,
        "selected_rho": selected_rho,
        "selected_checkpoint_artifact": checkpoint,
        "preselection_selected_checkpoint_pair_join": checkpoint_pair_join,
        "selected_checkpoint_completed_before_oof_transform_or_model_evaluation": (
            selected_rho is not None
        ),
        "oof_used_for_training_checkpoint_or_hyperparameter_selection": False,
        "oof_semantic_tensor_materialized_count": len(evaluation) * len(EVAL_VIEWS),
        "oof_semantic_tensor_read_count_exact0_on_inner_no_go": selected_rho is None,
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
        "geometry_weight": config.geometry_weight,
        "code_shape": [CODE_TIME, CODE_CHANNELS],
        "actual_code_numel": CODE_NUMEL,
        "training_rho": TRAINING_RHO,
        "rho_grid": list(RHO_GRID),
        "rho0_comparator": RHO_COMPARATOR,
        "fixed_preselection_step": FIXED_SELECTED_STEP,
        "rho_candidate_count": len(RHO_GRID),
        "single_candidate": False,
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
            preparse_sha = digest_before.hexdigest()
            preparse_physical = {
                "device": before.st_dev, "inode": before.st_ino,
                "size_bytes": before.st_size,
            }
            expected_file_sha = expected.get("file_sha256")
            trusted_fresh_write_branch = expected_file_sha is None
            if (
                (expected_file_sha is not None and expected_file_sha != preparse_sha)
                or (
                    expected.get("size_bytes") is not None
                    and int(expected["size_bytes"]) != before.st_size
                )
                or (
                    expected.get("physical_identity") is not None
                    and expected["physical_identity"] != preparse_physical
                )
            ):
                raise RuntimeError("checkpoint expected binding differs before torch parse")
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
        metadata.get("schema_version") != CHECKPOINT_SCHEMA
        or metadata_digest != _object_sha(metadata_unsigned)
        or metadata.get("outer_fold") != expected.get("outer_fold")
        or metadata.get("selected_step") != expected.get("selected_step")
        or metadata.get("checkpoint_role") != expected.get("checkpoint_role")
        or metadata.get("deployment_rho") != expected.get("deployment_rho")
        or metadata.get("deployment_rho_fp32_exact_power_of_two") is not True
        or metadata.get("preselection_base_state_sha256")
            != expected.get("preselection_base_state_sha256")
        or metadata.get("preselection_checkpoint_binding_sha256")
            != expected.get("preselection_checkpoint_binding_sha256")
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
    base_state_sha = _base_state_sha(state)
    preselection_binding = metadata.get("preselection_checkpoint_binding")
    expected_preselection_sha = expected.get("preselection_checkpoint_file_sha256")
    if (
        state_sha != metadata["model_state_sha256"]
        or base_state_sha != metadata["preselection_base_state_sha256"]
        or metadata.get("preselection_checkpoint_binding_sha256") != (
            _object_sha(preselection_binding)
            if preselection_binding is not None else None
        )
        or (
            metadata["checkpoint_role"] == "preselection_fixed_step1200"
            and (
                metadata["deployment_rho"] != TRAINING_RHO
                or preselection_binding is not None
            )
        )
        or (
            metadata["checkpoint_role"] == "selected_fold_local_rho"
            and (
                metadata["deployment_rho"] not in RHO_GRID
                or type(preselection_binding) is not dict
                or preselection_binding.get("preselection_base_state_sha256")
                    != base_state_sha
            )
        )
        or (
            expected_preselection_sha is not None
            and (
                type(preselection_binding) is not dict
                or preselection_binding.get("file_sha256") != expected_preselection_sha
            )
        )
    ):
        raise RuntimeError("checkpoint semantic state replay differs")
    basis = metadata.get("basis")
    required_buffers = {
        "clip_mean", "clip_basis", "fit_only_rms", "residual_gate_rho"
    }
    if (
        type(basis) is not dict or not required_buffers.issubset(state)
        or basis.get("clip_mean_sha256") != _tensor_sha(state["clip_mean"])
        or basis.get("clip_basis_sha256") != _tensor_sha(state["clip_basis"])
        or basis.get("fit_only_global_rms_sha256")
            != _tensor_sha(state["fit_only_rms"])
        or tuple(state["clip_mean"].shape) != (1, FULL_NUMEL)
        or tuple(state["clip_basis"].shape) != (FULL_NUMEL, CODE_NUMEL)
        or tuple(state["fit_only_rms"].shape) != (1,)
        or tuple(state["residual_gate_rho"].shape) != (1,)
        or any(state[name].dtype != torch.float32 for name in required_buffers)
        or float(state["residual_gate_rho"].item())
            != float(metadata.get("deployment_rho", float("nan")))
        or float(state["residual_gate_rho"].item())
            not in (RHO_COMPARATOR, *RHO_GRID)
    ):
        raise RuntimeError("checkpoint basis metadata/state join differs")
    template_fit = ClipPCAFit(
        clip_mean=state["clip_mean"], clip_basis=state["clip_basis"],
        fit_iid_digest=str(metadata["model_fit_iid_digest"]),
        fit_input_sha256=str(basis.get("fixed_clip_pca_fit_input_sha256")),
        diagnostics={},
    )
    template = VJepa2GlobalCodec(template_fit, state["fit_only_rms"])
    template_state = template.state_dict()
    if (
        set(template_state) != set(state)
        or any(
            template_state[name].shape != state[name].shape
            or template_state[name].dtype != state[name].dtype
            for name in template_state
        )
        or sum(parameter.numel() for parameter in template.parameters())
            != EXACT_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("checkpoint exact model schema closure differs")
    template.load_state_dict(state, strict=True)
    binding = {
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha,
        "size_bytes": before.st_size,
        "mode_octal": "0444",
        "nlink": before.st_nlink,
        "physical_identity": physical_identity,
        "single_fd_pre_post_sha256_exact": True,
        "semantic_metadata_state_replay_verified": True,
        "expected_file_binding_checked_before_torch_parse": not trusted_fresh_write_branch,
        "trusted_fresh_write_branch_before_first_external_file_sha": trusted_fresh_write_branch,
        "basis_metadata_state_hash_join_verified": True,
        "preselection_base_state_sha256": base_state_sha,
        "deployment_rho": metadata["deployment_rho"],
        "checkpoint_role": metadata["checkpoint_role"],
        "model_schema_reconstructed_and_strict_loaded": True,
        "model_forward_executed_by_loader": False,
    }
    return metadata, state, binding


def _save_checkpoint_create_only(
    path: Path, model: VJepa2GlobalCodec, fitted: ClipPCAFit,
    training_audit: Mapping[str, Any], config: Config, fold_index: int,
    run_binding: Mapping[str, str], probe_rows: Sequence[v4c.Record],
    device: torch.device, *, checkpoint_role: str, deployment_rho: float,
    preselection_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (
        not path.is_absolute() or not path.parent.is_dir()
        or path.exists() or path.is_symlink()
    ):
        raise ValueError("checkpoint must be a fresh absolute child")
    state = _state_to_cpu(model)
    state_sha = _state_sha(state)
    base_state_sha = _base_state_sha(state)
    if (
        checkpoint_role not in {
            "preselection_fixed_step1200", "selected_fold_local_rho"
        }
        or deployment_rho not in (RHO_COMPARATOR, *RHO_GRID)
        or float(state["residual_gate_rho"].item()) != deployment_rho
        or int(training_audit["selected_step"]) != FIXED_SELECTED_STEP
        or base_state_sha != training_audit["final_step_base_state_sha256"]
        or (checkpoint_role == "preselection_fixed_step1200"
            and (
                deployment_rho != TRAINING_RHO
                or preselection_artifact is not None
                or state_sha != training_audit["selected_state_sha256"]
            ))
        or (checkpoint_role == "selected_fold_local_rho"
            and (deployment_rho not in RHO_GRID or type(preselection_artifact) is not dict))
    ):
        raise RuntimeError("physical checkpoint does not join fixed trained base/rho role")
    preselection_binding = None
    if preselection_artifact is not None:
        if (
            preselection_artifact.get("checkpoint_role")
                != "preselection_fixed_step1200"
            or preselection_artifact.get("deployment_rho") != TRAINING_RHO
            or preselection_artifact.get("preselection_base_state_sha256")
                != base_state_sha
        ):
            raise RuntimeError("selected checkpoint does not join preselection authority")
        preselection_binding = {
            key: preselection_artifact[key] for key in (
                "path", "file_sha256", "size_bytes", "metadata_digest",
                "model_state_sha256", "preselection_base_state_sha256",
            )
        }
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "outer_fold": fold_index,
        "checkpoint_role": checkpoint_role,
        "selected_step": FIXED_SELECTED_STEP,
        "deployment_rho": deployment_rho,
        "deployment_rho_fp32_exact_power_of_two": deployment_rho in RHO_GRID,
        "full_budget_steps_executed": config.max_steps,
        "checkpoint_schedule": list(config.checkpoint_steps),
        "minibatch_schedule_sha256": training_audit["minibatch_schedule_sha256"],
        "model_state_sha256": state_sha,
        "preselection_base_state_sha256": base_state_sha,
        "preselection_checkpoint_binding": preselection_binding,
        "preselection_checkpoint_binding_sha256": (
            _object_sha(preselection_binding)
            if preselection_binding is not None else None
        ),
        "selected_training_audit_state_join_verified": True,
        "config": _config_value(config),
        "config_sha256": _object_sha(_config_value(config)),
        "implementation": dict(run_binding),
        "fixed_comparator_name": BASELINE_NAME,
        "basis": {
            "clip_mean_sha256": _tensor_sha(fitted.clip_mean),
            "clip_basis_sha256": _tensor_sha(fitted.clip_basis),
            "fit_only_global_rms_sha256": training_audit["fit_only_global_rms_sha256"],
            "fixed_clip_pca_fit_input_sha256": fitted.fit_input_sha256,
        },
        "model_fit_original_count": training_audit["model_fit_original_count"],
        "model_fit_ordered_iids": training_audit["model_fit_ordered_iids"],
        "model_fit_iid_digest": training_audit["model_fit_iid_digest"],
        "inner_validation_iid_digest": training_audit["inner_validation_iid_digest"],
        "artifact_scope": (
            "burned-development fold codec checkpoint; fixed trained base plus "
            "separately bound fold-local deployment rho; not refit or authorized inference"
        ),
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
        "checkpoint_role": checkpoint_role,
        "selected_step": FIXED_SELECTED_STEP,
        "deployment_rho": deployment_rho,
        "preselection_base_state_sha256": base_state_sha,
        "preselection_checkpoint_file_sha256": (
            preselection_artifact.get("file_sha256")
            if preselection_artifact is not None else None
        ),
        "preselection_checkpoint_binding_sha256": (
            _object_sha(preselection_binding)
            if preselection_binding is not None else None
        ),
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
    reloaded = VJepa2GlobalCodec(
        fitted, model.fit_only_rms.detach().cpu()
    )
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.to(device).eval()
    if not probe_rows:
        raise RuntimeError("checkpoint strict-reload probe rows are empty")
    probe = torch.stack([
        v4c.canonical_action(row.views["original"])
        for row in probe_rows[:min(config.batch_size, len(probe_rows))]
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
        "checkpoint_role": checkpoint_role,
        "selected_step": FIXED_SELECTED_STEP,
        "deployment_rho": deployment_rho,
        "preselection_base_state_sha256": base_state_sha,
        "preselection_checkpoint_file_sha256": (
            preselection_artifact.get("file_sha256")
            if preselection_artifact is not None else None
        ),
        "preselection_checkpoint_binding_sha256": (
            _object_sha(preselection_binding)
            if preselection_binding is not None else None
        ),
        "model_state_sha256": state_sha,
        "implementation_sha256": run_binding["implementation_sha256"],
        "selected_training_audit_state_join_verified": True,
        "metadata_digest": metadata["metadata_digest"],
        "fresh_reload_strict_state_verified": True,
        "fresh_reload_output_bit_exact": True,
        "caller_model_reloaded_from_sealed_artifact_before_next_stage": True,
    }


def _verify_checkpoint_artifacts(
    artifacts: Sequence[Mapping[str, Any]], *, expected_role: str,
) -> None:
    if len(artifacts) != OUTER_FOLDS:
        raise RuntimeError("selected checkpoint artifact count differs")
    for fold, artifact in enumerate(artifacts):
        path = Path(str(artifact["path"]))
        if (
            int(artifact["outer_fold"]) != fold
            or artifact.get("checkpoint_role") != expected_role
            or (
                expected_role == "preselection_fixed_step1200"
                and artifact.get("deployment_rho") != TRAINING_RHO
            )
            or (
                expected_role == "selected_fold_local_rho"
                and artifact.get("deployment_rho") not in RHO_GRID
            )
        ):
            raise RuntimeError("sealed selected checkpoint artifact changed")
        metadata, state, binding = _load_selected_checkpoint_sealed(path, artifact)
        if (
            binding["file_sha256"] != artifact["file_sha256"]
            or binding["size_bytes"] != artifact["size_bytes"]
            or metadata["model_state_sha256"] != artifact["model_state_sha256"]
            or _state_sha(state) != artifact["model_state_sha256"]
            or _base_state_sha(state)
                != artifact["preselection_base_state_sha256"]
        ):
            raise RuntimeError("sealed selected checkpoint replay changed")


def _verify_distinct_checkpoint_pair(
    preselection: Mapping[str, Any], selected: Mapping[str, Any],
    selected_rho: float,
) -> dict[str, Any]:
    """Hard-join two independently created files around one trained base."""

    if type(preselection) is not dict or type(selected) is not dict:
        raise RuntimeError("preselection/selected checkpoint envelopes differ")
    pre_physical = preselection.get("physical_identity")
    selected_physical = selected.get("physical_identity")
    if (
        type(pre_physical) is not dict or type(selected_physical) is not dict
        or preselection.get("checkpoint_role") != "preselection_fixed_step1200"
        or selected.get("checkpoint_role") != "selected_fold_local_rho"
        or preselection.get("deployment_rho") != TRAINING_RHO
        or selected.get("deployment_rho") != selected_rho
        or selected_rho not in RHO_GRID
        or preselection.get("path") == selected.get("path")
        or (
            pre_physical.get("device"), pre_physical.get("inode")
        ) == (
            selected_physical.get("device"), selected_physical.get("inode")
        )
        or preselection.get("preselection_base_state_sha256")
            != selected.get("preselection_base_state_sha256")
        or selected.get("preselection_checkpoint_file_sha256")
            != preselection.get("file_sha256")
        or any(
            artifact.get("semantic_metadata_state_replay_verified") is not True
            or artifact.get("fresh_reload_strict_state_verified") is not True
            or artifact.get("fresh_reload_output_bit_exact") is not True
            or artifact.get(
                "caller_model_reloaded_from_sealed_artifact_before_next_stage"
            ) is not True
            for artifact in (preselection, selected)
        )
    ):
        raise RuntimeError("preselection/selected checkpoint pair join differs")
    return {
        "preselection_path": preselection["path"],
        "selected_path": selected["path"],
        "preselection_device_inode": [
            pre_physical["device"], pre_physical["inode"]
        ],
        "selected_device_inode": [
            selected_physical["device"], selected_physical["inode"]
        ],
        "distinct_device_inode_pair": True,
        "same_preselection_base_state_sha256": True,
        "preselection_base_state_sha256": preselection[
            "preselection_base_state_sha256"
        ],
        "selected_deployment_rho": selected_rho,
        "selected_rho_strict_reload_verified": True,
        "both_checkpoint_files_strongly_and_strictly_reloaded": True,
    }


def _load_feature_metadata_authority(
    feature_root: Path, expected_receipt_sha256: str,
) -> tuple[list[v4c.Record], dict[str, Any], dict[str, Any]]:
    """Load sealed receipt/manifest metadata without materializing feature tensors."""

    if expected_receipt_sha256 != V4C_FEATURE_RECEIPT_SHA256:
        raise ValueError("feature receipt SHA is not frozen")
    if (
        not feature_root.is_absolute() or feature_root.is_symlink()
        or str(feature_root) != str(feature_root.resolve(strict=True))
    ):
        raise ValueError("feature root must be absolute/plain/canonical")
    receipt_path = feature_root / "feature_extraction_receipt.json"
    receipt = v4c._load_json_sealed(receipt_path, expected_receipt_sha256)
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest", None)
    manifest = receipt.get("manifest")
    shard_rows = receipt.get("shards")
    if (
        receipt.get("schema_version") != features.RECEIPT_SCHEMA
        or receipt.get("status") != "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED"
        or digest != _object_sha(unsigned)
        or receipt.get("implementation", {}).get("sha256")
            != EXTRACTOR_IMPLEMENTATION_SHA256
        or receipt.get("exact6_shards") is not True
        or receipt.get("population", {}).get("unique_base_clips") != 644
        or receipt.get("population", {}).get("family_count") != 28
        or receipt.get("feature_geometry", {}).get("views") != list(EVAL_VIEWS)
        or receipt.get("feature_geometry", {}).get("stored_sequence_per_view")
            != [TIME_STEPS, FEATURE_DIM]
        or receipt.get("formal_training_authorized") is not False
        or receipt.get("burned_development_only") is not True
        or type(manifest) is not dict
        or manifest.get("sha256") != features.FEATURE_MANIFEST_SHA256
        or manifest.get("manifest_digest") != features.FEATURE_MANIFEST_DIGEST
        or type(shard_rows) is not list or len(shard_rows) != 6
    ):
        raise ValueError("feature metadata authority differs")
    anchors, _ = features.load_anchor_manifest(
        Path(manifest["path"]), manifest["sha256"]
    )
    if (
        len(anchors) != 644 or len({row.iid for row in anchors}) != 644
        or _object_sha([row.iid for row in anchors])
            != receipt.get("exact644_ordered_iid_digest")
    ):
        raise ValueError("feature anchor metadata population differs")
    for index, shard in enumerate(shard_rows):
        expected_ordinals = [ordinal for ordinal in range(644) if ordinal % 6 == index]
        path = Path(str(shard.get("path"))) if type(shard) is dict else Path("")
        if (
            type(shard) is not dict or shard.get("index") != index
            or shard.get("record_count") != len(expected_ordinals)
            or shard.get("mode") != 0o444 or shard.get("nlink") != 1
            or not path.is_absolute() or path.is_symlink()
            or str(path) != str(path.resolve(strict=True))
        ):
            raise ValueError("feature shard metadata placement differs")
        value_stat = path.lstat()
        if (
            not stat.S_ISREG(value_stat.st_mode)
            or stat.S_IMODE(value_stat.st_mode) != 0o444
            or value_stat.st_nlink != 1
            or value_stat.st_size != shard.get("size_bytes")
        ):
            raise ValueError("feature shard metadata seal differs")
    metadata = [v4c.Record(
        iid=row.iid, family=row.family, strict=row.strict, views={}
    ) for row in anchors]
    bound = dict(receipt)
    bound["feature_receipt_path"] = str(receipt_path)
    bound["feature_receipt_file_sha256"] = expected_receipt_sha256
    index = {
        "anchors_by_iid": {row.iid: row for row in anchors},
        "ordinal_by_iid": {row.iid: row.ordinal for row in anchors},
        "receipt": receipt,
    }
    return metadata, bound, index


def _validate_fake_record_metadata(
    record: Mapping[str, Any], anchor: Any,
) -> dict[str, torch.Tensor]:
    sequences = record.get("view_sequences")
    view_receipts = record.get("view_receipts")
    if (
        record.get("ordinal") != anchor.ordinal or record.get("iid") != anchor.iid
        or record.get("family") != anchor.family
        or record.get("strict_selection_gates_all_true") is not anchor.strict
        or record.get("role") != "action_anchor"
        or record.get("view_order") != list(EVAL_VIEWS)
        or type(sequences) is not dict or set(sequences) != set(EVAL_VIEWS)
        or type(view_receipts) is not dict or set(view_receipts) != set(EVAL_VIEWS)
    ):
        raise ValueError("fake feature record metadata differs")
    output: dict[str, torch.Tensor] = {}
    for view in EVAL_VIEWS:
        tensor = sequences[view]
        storage = tensor.untyped_storage() if isinstance(tensor, torch.Tensor) else None
        offset = getattr(storage, "_checkpoint_offset", None)
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != (TIME_STEPS, FEATURE_DIM)
            or tensor.dtype != torch.float32 or not tensor.is_contiguous()
            or tuple(tensor.stride()) != (FEATURE_DIM, 1)
            or tensor.storage_offset() != 0 or tensor.numel() != FULL_NUMEL
            or tensor.element_size() != 4 or type(offset) is not int or offset < 0
            or features.SHA_RE.fullmatch(str(
                view_receipts[view].get("ordered_contextual_sequence_sha256")
            )) is None
        ):
            raise ValueError("fake feature tensor metadata differs")
        output[view] = tensor
    return output


def _selective_materialize_feature_rows(
    feature_index: Mapping[str, Any],
    requested_views_by_iid: Mapping[str, Sequence[str]], *, stage: str,
) -> tuple[dict[str, v4c.Record], dict[str, Any]]:
    """Materialize only named IID/view storages from sealed Torch ZIP offsets."""

    if stage not in {
        "stage1_model_fit_only",
        "stage2_post_preselection_seal_inner_five_views",
        "stage3_post_selected_seal_oof",
    }:
        raise ValueError("selective feature stage differs")
    if sys.byteorder != "little" or not requested_views_by_iid:
        raise RuntimeError("selective feature byte-order/request closure differs")
    anchors = feature_index["anchors_by_iid"]
    ordinals = feature_index["ordinal_by_iid"]
    receipt = feature_index["receipt"]
    normalized: dict[str, tuple[str, ...]] = {}
    for iid, requested in requested_views_by_iid.items():
        views = tuple(requested)
        if (
            iid not in anchors or not views or len(set(views)) != len(views)
            or not set(views).issubset(EVAL_VIEWS)
        ):
            raise ValueError("selective IID/view request differs")
        normalized[iid] = views
    requested_by_shard: dict[int, list[str]] = {index: [] for index in range(6)}
    for iid in normalized:
        requested_by_shard[ordinals[iid] % 6].append(iid)
    output: dict[str, v4c.Record] = {}
    materialized_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(receipt["shards"]):
        path = Path(shard["path"])
        before = path.lstat()
        if (
            path.is_symlink() or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or before.st_size != shard["size_bytes"] or before.st_size <= 0
        ):
            raise RuntimeError("selective shard pre-open seal differs")
        if not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("selective shard O_NOFOLLOW is unavailable")
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            digest_before = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_before.update(chunk)
            if (
                digest_before.hexdigest() != shard["sha256"]
                or _checkpoint_stat_identity(before)
                    != _checkpoint_stat_identity(opened)
            ):
                raise RuntimeError(
                    "selective shard failed pinned SHA/identity before metadata parse"
                )
            handle.seek(0)
            try:
                from torch._subclasses.fake_tensor import FakeTensorMode
            except ImportError as error:
                raise RuntimeError("FakeTensorMode is required for selective loading") from error
            with FakeTensorMode():
                payload = torch.load(handle, map_location="cpu", weights_only=True)
            after_fake = os.fstat(handle.fileno())
            payload_records = payload.get("records") if type(payload) is dict else None
            expected_ordinals = [
                ordinal for ordinal in range(644) if ordinal % 6 == shard_index
            ]
            if (
                type(payload) is not dict
                or payload.get("schema_version") != features.FEATURE_SCHEMA
                or payload.get("shard_index") != shard_index
                or payload.get("num_shards") != 6
                or payload.get("global_anchor_ordinals") != expected_ordinals
                or type(payload_records) is not list
                or len(payload_records) != len(expected_ordinals)
            ):
                raise ValueError("fake shard payload metadata differs")
            fake_by_iid: dict[str, tuple[Mapping[str, Any], dict[str, torch.Tensor]]] = {}
            all_intervals: list[tuple[int, int, str, str]] = []
            for record, ordinal in zip(payload_records, expected_ordinals):
                anchor = anchors.get(record.get("iid")) if type(record) is dict else None
                if anchor is None or anchor.ordinal != ordinal:
                    raise ValueError("fake shard record ordinal differs")
                fake_views = _validate_fake_record_metadata(record, anchor)
                fake_by_iid[anchor.iid] = (record, fake_views)
                for view, tensor in fake_views.items():
                    start = int(tensor.untyped_storage()._checkpoint_offset)
                    end = start + FULL_NUMEL * 4
                    all_intervals.append((start, end, anchor.iid, view))
                    offset_rows.append({
                        "shard": shard_index, "iid": anchor.iid, "view": view,
                        "offset": start, "payload_bytes": FULL_NUMEL * 4,
                    })
            expected_shard_iids = {
                iid for iid, ordinal in ordinals.items()
                if ordinal % 6 == shard_index
            }
            if set(fake_by_iid) != expected_shard_iids:
                raise ValueError("fake shard exact IID set differs")
            ordered_intervals = sorted(all_intervals)
            if (
                len(ordered_intervals) != len(expected_ordinals) * len(EVAL_VIEWS)
                or any(left[0] == right[0] or left[1] > right[0]
                       for left, right in zip(ordered_intervals, ordered_intervals[1:]))
                or any(start < 0 or end > before.st_size
                       for start, end, _, _ in ordered_intervals)
            ):
                raise RuntimeError("fake tensor storage offsets alias or exceed shard")
            for iid in requested_by_shard[shard_index]:
                record, fake_views = fake_by_iid[iid]
                selected: dict[str, torch.Tensor] = {}
                for view in normalized[iid]:
                    fake = fake_views[view]
                    offset = int(fake.untyped_storage()._checkpoint_offset)
                    raw = os.pread(handle.fileno(), FULL_NUMEL * 4, offset)
                    if len(raw) != FULL_NUMEL * 4:
                        raise RuntimeError("selective tensor pread was short")
                    value = torch.frombuffer(
                        bytearray(raw), dtype=torch.float32
                    ).reshape(TIME_STEPS, FEATURE_DIM).clone()
                    expected_sha = record["view_receipts"][view][
                        "ordered_contextual_sequence_sha256"
                    ]
                    actual_sha = _tensor_sha(value)
                    if actual_sha != expected_sha or not bool(torch.isfinite(value).all()):
                        raise RuntimeError("selectively materialized tensor digest differs")
                    selected[view] = value
                    materialized_rows.append({
                        "iid": iid, "view": view, "tensor_sha256": actual_sha,
                        "shard": shard_index, "offset": offset,
                    })
                output[iid] = v4c.Record(
                    iid=iid, family=anchors[iid].family,
                    strict=anchors[iid].strict, views=selected,
                )
            handle.seek(0)
            digest_after = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_after.update(chunk)
            closed = os.fstat(handle.fileno())
        after = path.lstat()
        identities = {
            _checkpoint_stat_identity(value)
            for value in (before, opened, after_fake, closed, after)
        }
        if (
            digest_before.hexdigest() != shard["sha256"]
            or digest_after.hexdigest() != shard["sha256"]
            or len(identities) != 1
        ):
            raise RuntimeError("selective shard single-FD seal changed")
    if set(output) != set(normalized):
        raise RuntimeError("selective requested IID closure differs")
    materialized_rows.sort(key=lambda row: (ordinals[row["iid"]], EVAL_VIEWS.index(row["view"])))
    offset_rows.sort(key=lambda row: (row["shard"], row["offset"]))
    count_by_view = {
        view: sum(row["view"] == view for row in materialized_rows)
        for view in EVAL_VIEWS
    }
    audit = {
        "stage": stage,
        "requested_iid_cluster_count": len(normalized),
        "semantic_tensor_materialized_count": len(materialized_rows),
        "semantic_tensor_materialized_count_by_view": count_by_view,
        "semantic_tensor_materialization_ledger_sha256": _object_sha(materialized_rows),
        "requested_iid_view_map_sha256": _object_sha({
            iid: list(normalized[iid]) for iid in sorted(normalized)
        }),
        "all_fake_tensor_offset_map_sha256": _object_sha(offset_rows),
        "all_fake_tensor_offsets_unique_nonoverlapping_in_file": True,
        "full_shard_raw_bytes_cryptographic_scanned": True,
        "cryptographic_raw_byte_scan_counted_as_semantic_tensor_materialization": False,
        "unrequested_tensor_storage_materialized_count": 0,
        "fake_metadata_load_materialized_tensor_storage": False,
        "single_fd_pre_post_shard_sha_and_identity_exact": True,
    }
    return output, audit


def _prepare_authorities(args: argparse.Namespace) -> dict[str, Any]:
    """Load every frozen authority and recover the one exact644 row order."""

    if args.expected_feature_receipt_sha256 != V4C_FEATURE_RECEIPT_SHA256:
        raise ValueError("v4-C feature receipt SHA is not frozen")
    v4a_path = Path(args.v4a_receipt)
    outer_assignment, v4a_receipt = v4c.load_frozen_v4a_split(
        v4a_path, args.expected_v4a_receipt_sha256
    )
    v4c_path = Path(args.v4c_frontier_receipt)
    v4c_receipt = _load_v4c_frontier_receipt(
        v4c_path, args.expected_v4c_frontier_receipt_sha256
    )
    v4d_path = Path(args.v4d_receipt)
    v4d_receipt = _load_v4d_burned_receipt(
        v4d_path, args.expected_v4d_receipt_sha256
    )
    feature_root = Path(args.feature_root).resolve(strict=True)
    records, feature_bound, feature_index = _load_feature_metadata_authority(
        feature_root, args.expected_feature_receipt_sha256
    )
    feature_receipt = feature_index["receipt"]
    records_by_iid = {row.iid: row for row in records}
    split_evidence = v4a_receipt["oof_closure"]["embedded_paired_margin_evidence"]
    ordered = [records_by_iid[row["iid"]] for row in split_evidence]
    exact_iids = [row.iid for row in ordered]
    if (
        len(records) != 644 or len(records_by_iid) != 644
        or len(set(exact_iids)) != 644 or set(exact_iids) != set(outer_assignment)
        or len({row.family for row in ordered}) != 28
        or _object_sha(outer_assignment) != v4c.OUTER_ASSIGNMENT_DIGEST
        or any(row.family != evidence["family"]
               for row, evidence in zip(ordered, split_evidence))
        or v4c_receipt["feature_authority"]["feature_receipt_sha256"]
            != args.expected_feature_receipt_sha256
    ):
        raise ValueError("v4-F feature/split/frontier population differs")
    return {
        "outer_assignment": outer_assignment,
        "v4a_receipt": v4a_receipt,
        "v4a_path": v4a_path,
        "v4c_receipt": v4c_receipt,
        "v4c_path": v4c_path,
        "v4d_receipt": v4d_receipt,
        "v4d_path": v4d_path,
        "feature_root": feature_root,
        "feature_receipt": feature_receipt,
        "feature_bound": feature_bound,
        "feature_index": feature_index,
        "ordered_records": ordered,
        "exact_iids": exact_iids,
    }


def _reverify_authorities(authority: Mapping[str, Any], args: argparse.Namespace) -> None:
    v4c._assert_input_files_unchanged(
        authority["feature_bound"], authority["v4a_path"]
    )
    _load_v4c_frontier_receipt(
        authority["v4c_path"], args.expected_v4c_frontier_receipt_sha256
    )
    _load_v4d_burned_receipt(
        authority["v4d_path"], args.expected_v4d_receipt_sha256
    )


def _resolve_fold_root(
    path_string: str, *, fresh: bool,
) -> tuple[Path, Path, Path, Path]:
    root = Path(path_string)
    if (
        not root.is_absolute() or root.is_symlink()
        or not root.is_dir() or str(root) != str(root.resolve(strict=True))
    ):
        raise ValueError("fold root must be an existing absolute canonical directory")
    fold_path = root / "fold.json"
    preselection_path = root / "preselection.pt"
    checkpoint_path = root / "selected.pt"
    if fresh and (
        fold_path.exists() or fold_path.is_symlink()
        or preselection_path.exists() or preselection_path.is_symlink()
        or checkpoint_path.exists() or checkpoint_path.is_symlink()
    ):
        raise ValueError("fold root output artifacts must all be fresh")
    return root, fold_path, preselection_path, checkpoint_path


def run_train_fold(args: argparse.Namespace) -> dict[str, Any]:
    """Train one fold and execute its nested, fail-closed rho procedure."""

    _require_release_sealed()
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-F torch runtime differs")
    if type(args.fold_index) is not int or not 0 <= args.fold_index < OUTER_FOLDS:
        raise ValueError("fold index must be in [0,4]")
    torch.set_num_threads(1)
    device = _resolve_device(args.device)
    fold_root, fold_path, preselection_path, checkpoint_path = _resolve_fold_root(
        args.fold_root, fresh=True
    )
    authority = _prepare_authorities(args)
    fold, evidence = _run_fold(
        authority["ordered_records"], authority["outer_assignment"],
        authority["v4a_receipt"], args.fold_index, config, device,
        preselection_path, checkpoint_path, run_binding, authority["feature_index"],
    )
    inner_pass = fold["rho_selection"]["inner_pass"]
    if inner_pass:
        if (
            len(evidence) != FROZEN_OOF_COUNTS[args.fold_index]
            or any(int(row["outer_fold"]) != args.fold_index for row in evidence)
            or _object_sha([row["iid"] for row in evidence]) != fold["oof_iid_digest"]
        ):
            raise RuntimeError("single-fold OOF evidence closure differs")
    elif (
        evidence != []
        or fold["fold_status"] != INNER_NO_GO_STATUS
        or fold["selected_checkpoint_artifact"] is not None
        or checkpoint_path.exists()
        or fold["oof_semantic_tensor_materialized_count"] != 0
    ):
        raise RuntimeError("INNER_NO_GO did not preserve exact-zero OOF read closure")
    config_value = _config_value(config)
    receipt: dict[str, Any] = {
        "schema_version": FOLD_SCHEMA,
        "status": STATUS if inner_pass else INNER_NO_GO_STATUS,
        "authority": "burned_exposed_known_transform_development_fold_only",
        "implementation": run_binding,
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "fold_root": str(fold_root),
        "runtime": {
            "torch": str(torch.__version__), "torch_hip": str(torch.version.hip),
            "device": str(device), "full_precision_fp32_training": True,
            "autocast_used": False, "distributed_training_used": False,
        },
        "feature_authority": {
            "feature_root": str(authority["feature_root"]),
            "feature_receipt_sha256": V4C_FEATURE_RECEIPT_SHA256,
            "feature_receipt_digest": authority["feature_receipt"]["receipt_digest"],
        },
        "upstream_authorities": {
            "v4a_receipt_path": str(authority["v4a_path"].resolve(strict=True)),
            "v4a_receipt_sha256": V4A_RECEIPT_FILE_SHA256,
            "v4c_frontier_receipt_path": str(authority["v4c_path"].resolve(strict=True)),
            "v4c_frontier_receipt_sha256": V4C_FRONTIER_RECEIPT_SHA256,
            "v4d_burned_receipt_path": V4D_RECEIPT_PATH,
            "v4d_burned_receipt_sha256": V4D_RECEIPT_SHA256,
            "v4d_burned_receipt_digest": V4D_RECEIPT_DIGEST,
            "v4d_burned_development_gate": False,
            "v4e_burned_implementation_sha256": V4E_BURNED_IMPLEMENTATION_SHA256,
            "v4e_burned_fold_receipt_sha256":
                V4E_BURNED_FOLD_RECEIPT_SHA256[args.fold_index],
            "v4e_burned_oof_informed_residual_homotopy_choice": True,
            "v4e_oof_used_to_select_fold_local_rho": False,
        },
        "fold": fold,
        "oof_evidence_count": len(evidence),
        "oof_evidence_sha256": _object_sha(evidence),
        "oof_evidence": evidence,
        "nested_rho_selection_contract": {
            "one_predeclared_fold_local_selection_algorithm": True,
            "rho_candidate_count": len(RHO_GRID),
            "single_candidate": False,
            "rho0_exact_comparator_only_not_selectable": True,
            "model_fit_five_view_tensors_used_for_gradient_and_model_input": True,
            "inner_five_view_tensors_used_for_hyperparameter_selection": True,
            "inner_five_view_tensors_used_for_gradient_or_model_input": False,
            "transform_role_and_family_metadata_used_for_hyperparameter_selection": True,
            "transform_role_and_family_metadata_used_for_gradient": False,
            "transform_role_and_family_metadata_used_for_model_input": False,
            "teacher_and_fixed_pca_metadata_used_for_hyperparameter_selection": True,
            "teacher_and_fixed_pca_metadata_used_for_gradient_or_model_input": False,
            "cross_fold_inner_aggregation_or_global_rho_selection": False,
        },
        "known_transform_families_exposed_during_model_fit": True,
        "unseen_hostile_transform_gate_evaluated": False,
        "qualification_scope": {
            "exposed_five_view_codec_development_gate": None,
            "inner_fold_local_gate_passed": inner_pass,
            "aggregate_gate_evaluated": False,
            "latent_metric_qualified": False,
            "action_representation_qualified": False,
            "identity_disentanglement_qualified": False,
            "identity_preservation_qualified": False,
            "vae_necessary": None,
            "prior_qualified": False,
            "prior_generation_qualified": False,
            "generation_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "full644_refit_authorized": False,
        },
        "descriptive_scope": {"fold_local_development_training_performed": True},
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    _reverify_authorities(authority, args)
    receipt_sha = _write_json_create_only(fold_path, receipt)
    _load_selected_checkpoint_sealed(
        preselection_path, fold["preselection_checkpoint_artifact"]
    )
    if inner_pass:
        _load_selected_checkpoint_sealed(
            checkpoint_path, fold["selected_checkpoint_artifact"]
        )
    reloaded_receipt, reloaded_binding = _load_fold_receipt_sealed(
        str(fold_root), run_binding
    )
    if (
        reloaded_receipt != receipt
        or reloaded_binding["file_sha256"] != receipt_sha
        or reloaded_binding["receipt_digest"] != receipt["receipt_digest"]
    ):
        raise RuntimeError("fresh fold receipt strong self-read differs")
    _assert_binding_unchanged(run_binding)
    _reverify_authorities(authority, args)
    return {
        "fold": args.fold_index,
        "fold_receipt": str(fold_path.resolve(strict=True)),
        "fold_receipt_sha256": receipt_sha,
        "fold_receipt_digest": receipt["receipt_digest"],
        "status": receipt["status"],
        "preselection_checkpoint": str(preselection_path.resolve(strict=True)),
        "selected_checkpoint": (
            str(checkpoint_path.resolve(strict=True)) if inner_pass else None
        ),
        "selected_step": fold["training"]["selected_step"],
        "selected_rho": fold["selected_rho"],
        "oof_semantic_tensor_materialized_count": fold[
            "oof_semantic_tensor_materialized_count"
        ],
    }


def _load_fold_receipt_sealed(
    fold_root_string: str, expected_implementation: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one fixed-name fold receipt through one no-follow descriptor."""

    root, path, preselection_path, checkpoint_path = _resolve_fold_root(
        fold_root_string, fresh=False
    )
    if path.is_symlink() or not path.exists() or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("sealed fold receipt path differs")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1 or before.st_size <= 0
    ):
        raise ValueError("sealed fold receipt must be 0444/nlink1 regular")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        raw = handle.read()
        handle.seek(0)
        raw_after = handle.read()
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    identities = tuple(
        _checkpoint_stat_identity(value)
        for value in (before, opened, closed, after)
    )
    file_sha = hashlib.sha256(raw).hexdigest()
    if raw != raw_after or len(set(identities)) != 1 or path.is_symlink():
        raise RuntimeError("fold receipt single-FD bytes or identity changed")
    value = json.loads(
        raw, object_pairs_hook=_reject_duplicate_json_pairs,
        parse_constant=_reject_nonfinite_json,
    )
    if type(value) is not dict:
        raise ValueError("fold receipt JSON root differs")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    fold = value.get("fold")
    if (
        type(fold) is not dict or type(fold.get("fold_index")) is not int
        or not 0 <= fold["fold_index"] < OUTER_FOLDS
    ):
        raise ValueError("sealed fold receipt fold envelope differs")
    evidence = value.get("oof_evidence")
    artifact = fold.get("selected_checkpoint_artifact") if type(fold) is dict else None
    preselection = (
        fold.get("preselection_checkpoint_artifact") if type(fold) is dict else None
    )
    training = fold.get("training") if type(fold) is dict else None
    fit_iids = fold.get("model_fit_ordered_iids") if type(fold) is dict else None
    oof_iids = fold.get("oof_ordered_iids") if type(fold) is dict else None
    inner_iids = fold.get("inner_validation_ordered_iids")
    receipt_status = value.get("status")
    inner_pass = (
        fold.get("rho_selection", {}).get("inner_pass")
        if type(fold) is dict else None
    )
    evidence_count_expected = (
        FROZEN_OOF_COUNTS[fold["fold_index"]]
        if inner_pass is True else 0
    )
    selected_artifact_valid = bool(
        type(fold) is dict and type(training) is dict and (
            (
                inner_pass is True
                and type(artifact) is dict
                and type(preselection) is dict
                and checkpoint_path.exists()
                and artifact.get("path") == str(checkpoint_path.resolve(strict=True))
                and artifact.get("outer_fold") == fold["fold_index"]
                and artifact.get("selected_step") == FIXED_SELECTED_STEP
                and artifact.get("deployment_rho") == fold.get("selected_rho")
                and artifact.get("checkpoint_role") == "selected_fold_local_rho"
                and artifact.get("preselection_base_state_sha256")
                    == preselection.get("preselection_base_state_sha256")
                and artifact.get("preselection_checkpoint_file_sha256")
                    == preselection.get("file_sha256")
                and artifact.get("implementation_sha256")
                    == expected_implementation.get("implementation_sha256")
            )
            or (
                inner_pass is False and artifact is None
                and fold.get("selected_rho") is None
                and not checkpoint_path.exists()
                and not checkpoint_path.is_symlink()
            )
        )
    )
    if (
        value.get("schema_version") != FOLD_SCHEMA
        or receipt_status not in {STATUS, INNER_NO_GO_STATUS}
        or (receipt_status == STATUS) is not (inner_pass is True)
        or fold.get("fold_status") != receipt_status
        or digest != _object_sha(unsigned)
        or value.get("implementation") != dict(expected_implementation)
        or value.get("config") != _config_value(Config())
        or value.get("config_sha256") != _object_sha(_config_value(Config()))
        or value.get("fold_root") != str(root)
        or value.get("feature_authority", {}).get("feature_receipt_sha256")
            != V4C_FEATURE_RECEIPT_SHA256
        or value.get("upstream_authorities", {}).get("v4a_receipt_sha256")
            != V4A_RECEIPT_FILE_SHA256
        or value.get("upstream_authorities", {}).get("v4c_frontier_receipt_sha256")
            != V4C_FRONTIER_RECEIPT_SHA256
        or value.get("upstream_authorities", {}).get("v4d_burned_receipt_path")
            != V4D_RECEIPT_PATH
        or value.get("upstream_authorities", {}).get("v4d_burned_receipt_sha256")
            != V4D_RECEIPT_SHA256
        or value.get("upstream_authorities", {}).get("v4d_burned_receipt_digest")
            != V4D_RECEIPT_DIGEST
        or value.get("upstream_authorities", {}).get(
            "v4e_burned_implementation_sha256"
        ) != V4E_BURNED_IMPLEMENTATION_SHA256
        or value.get("upstream_authorities", {}).get(
            "v4e_burned_fold_receipt_sha256"
        ) != V4E_BURNED_FOLD_RECEIPT_SHA256[fold["fold_index"]]
        or value.get("upstream_authorities", {}).get(
            "v4e_burned_oof_informed_residual_homotopy_choice"
        ) is not True
        or value.get("upstream_authorities", {}).get(
            "v4e_oof_used_to_select_fold_local_rho"
        ) is not False
        or type(training) is not dict
        or type(fit_iids) is not list or len(fit_iids) != fold.get("model_fit_original_count")
        or len(set(fit_iids)) != len(fit_iids)
        or _object_sha(fit_iids) != fold.get("model_fit_iid_digest")
        or type(inner_iids) is not list
        or len(inner_iids) != fold.get("inner_validation_original_count")
        or len(set(inner_iids)) != len(inner_iids)
        or _object_sha(inner_iids) != fold.get("inner_validation_iid_digest")
        or training.get("inner_validation_ordered_iids") != inner_iids
        or training.get("inner_validation_iid_digest")
            != fold.get("inner_validation_iid_digest")
        or type(oof_iids) is not list or len(oof_iids) != fold.get("oof_original_count")
        or len(set(oof_iids)) != len(oof_iids)
        or _object_sha(oof_iids) != fold.get("oof_iid_digest")
        or type(evidence) is not list
        or len(evidence) != evidence_count_expected
        or value.get("oof_evidence_count") != len(evidence)
        or value.get("oof_evidence_sha256") != _object_sha(evidence)
        or any(row.get("outer_fold") != fold["fold_index"] for row in evidence)
        or (
            inner_pass is True and [row.get("iid") for row in evidence] != oof_iids
        )
        or (
            inner_pass is True and any(
                row.get("residual_gate_rho") != fold.get("selected_rho")
                or row.get("rho0_exact_clip_pca_alias_used") is not False
                for row in evidence
            )
        )
        or (inner_pass is False and evidence != [])
        or type(preselection) is not dict or not preselection_path.exists()
        or preselection.get("path") != str(preselection_path.resolve(strict=True))
        or preselection.get("outer_fold") != fold["fold_index"]
        or preselection.get("selected_step") != FIXED_SELECTED_STEP
        or preselection.get("checkpoint_role") != "preselection_fixed_step1200"
        or preselection.get("deployment_rho") != TRAINING_RHO
        or preselection.get("model_state_sha256")
            != training.get("selected_state_sha256")
        or preselection.get("preselection_base_state_sha256")
            != training.get("final_step_base_state_sha256")
        or preselection.get("implementation_sha256")
            != expected_implementation.get("implementation_sha256")
        or not selected_artifact_valid
        or fold.get("rho_candidate_count") != len(RHO_GRID)
        or fold.get("single_candidate") is not False
        or fold.get("rho_selection", {}).get("rho_candidate_count") != len(RHO_GRID)
        or fold.get("rho_selection", {}).get("single_candidate") is not False
        or fold.get("rho_selection", {}).get("selected_rho")
            != fold.get("selected_rho")
        or training.get("selected_step") != FIXED_SELECTED_STEP
        or value.get("qualification_scope", {}).get("inference_authorized") is not False
        or value.get("qualification_scope", {}).get("full644_refit_authorized") is not False
    ):
        raise ValueError("sealed fold receipt semantic replay differs")
    _verify_fold_selective_materialization_ledger(fold)
    binding = {
        "fold_root": str(root), "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha, "receipt_digest": digest,
        "size_bytes": before.st_size, "mode_octal": "0444",
        "nlink": before.st_nlink,
        "physical_identity": {
            "device": before.st_dev, "inode": before.st_ino,
            "size_bytes": before.st_size,
        },
        "single_fd_pre_post_bytes_and_identity_exact": True,
    }
    return value, binding


def _verify_rho_selection_ledger(fold: Mapping[str, Any]) -> None:
    selection = fold.get("rho_selection")
    candidates = selection.get("candidates") if type(selection) is dict else None
    inner_iids = fold.get("inner_validation_ordered_iids")
    comparator = (
        selection.get("rho0_inner_evidence") if type(selection) is dict else None
    )
    if (
        type(candidates) is not list or len(candidates) != len(RHO_GRID)
        or type(inner_iids) is not list
        or type(comparator) is not list
    ):
        raise ValueError("rho selection candidate ledger differs")
    if (
        len(comparator) != len(inner_iids)
        or [row.get("iid") for row in comparator] != inner_iids
        or selection.get("rho0_inner_evidence_count") != len(comparator)
        or selection.get("rho0_inner_evidence_sha256") != _object_sha(comparator)
        or any(
            row.get("residual_gate_rho") != RHO_COMPARATOR
            or row.get("rho0_exact_clip_pca_alias_used") is not True
            or any(
                row.get("raw_reconstruction_by_view", {}).get(view, {}).get(
                    "candidate_raw_mse"
                )
                != row.get("raw_reconstruction_by_view", {}).get(view, {}).get(
                    "clip_pca_b384_raw_mse"
                )
                for view in EVAL_VIEWS
            )
            for row in comparator
        )
    ):
        raise ValueError("rho-zero exact comparator evidence replay differs")
    fixed_reference_sha = _object_sha(
        _fixed_inner_reference_projection(comparator)
    )
    if selection.get(
        "fixed_teacher_and_clip_pca_reference_sha256"
    ) != fixed_reference_sha:
        raise ValueError("rho-zero fixed teacher/PCA reference digest differs")
    config = Config()
    config.validate()
    for ordinal, candidate in enumerate(candidates):
        evidence = candidate.get("inner_evidence")
        if (
            type(evidence) is not list or len(evidence) != len(inner_iids)
            or [row.get("iid") for row in evidence] != inner_iids
            or candidate.get("inner_evidence_count") != len(evidence)
            or candidate.get("inner_evidence_sha256") != _object_sha(evidence)
            or any(
                row.get("residual_gate_rho") != RHO_GRID[ordinal]
                or row.get("rho0_exact_clip_pca_alias_used") is not False
                for row in evidence
            )
            or candidate.get("fixed_teacher_and_clip_pca_reference_sha256")
                != fixed_reference_sha
            or _object_sha(_fixed_inner_reference_projection(evidence))
                != fixed_reference_sha
        ):
            raise ValueError("rho candidate embedded inner evidence differs")
        recomputed_gate = _inner_candidate_gate(
            evidence, config, fold_index=int(fold["fold_index"]),
            rho_ordinal=ordinal,
        )
        if (
            candidate.get("gate") != recomputed_gate
            or type(candidate.get("pass")) is not bool
            or candidate["pass"]
                is not recomputed_gate["complete_candidate_dependent_inner_gate"]
            or candidate.get("bootstrap_seed_ledger")
                != _bootstrap_seed_ledger(recomputed_gate)
            or len(candidate.get("bootstrap_seed_ledger", [])) != 34
        ):
            raise ValueError("rho candidate gate/bootstrap replay differs")
    pass_rhos = [
        row.get("rho") for row in candidates if row.get("pass") is True
    ]
    expected_selected = pass_rhos[0] if pass_rhos else None
    if (
        selection.get("selection_scope") != "fold_local_inner_only"
        or selection.get("outer_fold") != fold.get("fold_index")
        or selection.get("rho_candidate_count") != len(RHO_GRID)
        or selection.get("single_candidate") is not False
        or selection.get("rho_grid_preregistered_ascending") != list(RHO_GRID)
        or selection.get("rho0_comparator") != RHO_COMPARATOR
        or selection.get("rho0_selectable") is not False
        or selection.get("exact7_candidate_ledger_complete") is not True
        or len(candidates) != len(RHO_GRID)
        or [row.get("rho_ordinal") for row in candidates]
            != list(range(len(RHO_GRID)))
        or [row.get("rho") for row in candidates] != list(RHO_GRID)
        or selection.get("selected_rho") != expected_selected
        or fold.get("selected_rho") != expected_selected
        or selection.get("inner_pass") is not (expected_selected is not None)
        or selection.get("base_state_sha256_before_scan")
            != selection.get("base_state_sha256_after_scan")
        or selection.get("base_state_sha256_before_scan")
            != fold.get("preselection_checkpoint_artifact", {}).get(
                "preselection_base_state_sha256"
            )
        or selection.get("base_state_unchanged_across_all_rho_candidates") is not True
        or selection.get("inner_five_view_tensors_used_for_hyperparameter_selection")
            is not True
        or selection.get("inner_five_view_tensors_used_for_gradient_or_model_input")
            is not False
        or selection.get(
            "transform_role_and_family_metadata_used_for_hyperparameter_selection"
        ) is not True
        or selection.get("transform_role_and_family_metadata_used_for_gradient")
            is not False
        or selection.get("transform_role_and_family_metadata_used_for_model_input")
            is not False
        or selection.get(
            "teacher_and_fixed_pca_metadata_used_for_hyperparameter_selection"
        ) is not True
        or selection.get(
            "teacher_and_fixed_pca_metadata_used_for_gradient_or_model_input"
        ) is not False
        or selection.get(
            "cross_fold_inner_metric_aggregation_or_global_rho_selection"
        ) is not False
        or selection.get("monotonic_metric_behavior_assumed") is not False
        or selection.get("smallest_rho_minimizes_distortion_claimed") is not False
    ):
        raise ValueError("fold-local exact7 rho-selection replay differs")


def _verify_fold_selective_materialization_ledger(fold: Mapping[str, Any]) -> None:
    """Recompute the three-stage request ledgers from sealed IID authorities."""

    fit_iids = fold.get("model_fit_ordered_iids")
    inner_split = fold.get("inner_split")
    training = fold.get("training")
    inner_iids = fold.get("inner_validation_ordered_iids")
    oof_iids = fold.get("oof_ordered_iids")
    ledger = fold.get("selective_feature_materialization")
    if (
        type(fit_iids) is not list or type(inner_iids) is not list
        or type(oof_iids) is not list or type(inner_split) is not dict
        or type(ledger) is not dict
    ):
        raise ValueError("fold selective ledger ordered-IID authority differs")
    inner_pass = fold.get("rho_selection", {}).get("inner_pass")
    if type(inner_pass) is not bool:
        raise ValueError("fold inner gate status differs")
    stage1_request = {iid: list(EVAL_VIEWS) for iid in fit_iids}
    stage2_request = {iid: list(EVAL_VIEWS) for iid in inner_iids}
    stage3_request = (
        {iid: list(EVAL_VIEWS) for iid in oof_iids} if inner_pass else {}
    )
    stage1 = ledger.get("stage1_before_checkpoint_seal")
    stage2 = ledger.get("stage2_only_after_preselection_checkpoint_strong_seal_reload")
    stage3 = ledger.get(
        "stage3_only_after_selected_checkpoint_strong_seal_reload_or_no_go"
    )
    preselection_artifact = fold.get("preselection_checkpoint_artifact")
    selected_artifact = fold.get("selected_checkpoint_artifact")
    checkpoint_pair_join = fold.get(
        "preselection_selected_checkpoint_pair_join"
    )
    expected_checkpoint_pair_join = None
    if inner_pass and type(preselection_artifact) is dict and type(selected_artifact) is dict:
        expected_checkpoint_pair_join = _verify_distinct_checkpoint_pair(
            preselection_artifact, selected_artifact, float(fold["selected_rho"])
        )
    expected_stage1_counts = {view: len(fit_iids) for view in EVAL_VIEWS}
    expected_stage2_counts = {view: len(inner_iids) for view in EVAL_VIEWS}
    expected_stage3_counts = {
        view: len(oof_iids) if inner_pass else 0 for view in EVAL_VIEWS
    }
    if (
        len(set(fit_iids + inner_iids + oof_iids))
            != len(fit_iids) + len(inner_iids) + len(oof_iids)
        or len(fit_iids) + len(inner_iids) + len(oof_iids) != 644
        or type(training) is not dict
        or type(preselection_artifact) is not dict
        or preselection_artifact.get(
            "caller_model_reloaded_from_sealed_artifact_before_next_stage"
        ) is not True
        or (
            inner_pass and (
                type(selected_artifact) is not dict
                or selected_artifact.get(
                    "caller_model_reloaded_from_sealed_artifact_before_next_stage"
                ) is not True
            )
        )
        or (not inner_pass and selected_artifact is not None)
        or checkpoint_pair_join != expected_checkpoint_pair_join
        or training.get("inner_validation_ordered_iids") != inner_iids
        or type(stage1) is not dict or type(stage2) is not dict or type(stage3) is not dict
        or stage1.get("stage") != "stage1_model_fit_only"
        or stage2.get("stage")
            != "stage2_post_preselection_seal_inner_five_views"
        or stage3.get("stage") != (
            "stage3_post_selected_seal_oof"
            if inner_pass else "stage3_inner_no_go_oof_unread"
        )
        or stage1.get("requested_iid_cluster_count") != len(fit_iids)
        or stage2.get("requested_iid_cluster_count") != len(inner_iids)
        or stage3.get("requested_iid_cluster_count")
            != (len(oof_iids) if inner_pass else 0)
        or stage1.get("requested_iid_view_map_sha256")
            != _object_sha({iid: stage1_request[iid] for iid in sorted(stage1_request)})
        or stage2.get("requested_iid_view_map_sha256")
            != _object_sha({iid: stage2_request[iid] for iid in sorted(stage2_request)})
        or stage3.get("requested_iid_view_map_sha256")
            != _object_sha({iid: stage3_request[iid] for iid in sorted(stage3_request)})
        or stage1.get("semantic_tensor_materialized_count_by_view")
            != expected_stage1_counts
        or stage2.get("semantic_tensor_materialized_count_by_view")
            != expected_stage2_counts
        or stage3.get("semantic_tensor_materialized_count_by_view")
            != expected_stage3_counts
        or stage1.get("semantic_tensor_materialized_count")
            != sum(expected_stage1_counts.values())
        or stage2.get("semantic_tensor_materialized_count")
            != sum(expected_stage2_counts.values())
        or stage3.get("semantic_tensor_materialized_count")
            != sum(expected_stage3_counts.values())
        or stage1.get("unrequested_tensor_storage_materialized_count") != 0
        or stage2.get("unrequested_tensor_storage_materialized_count") != 0
        or stage3.get("unrequested_tensor_storage_materialized_count") != 0
        or (
            not inner_pass
            and stage3.get("oof_semantic_tensor_read_count_exact0") is not True
        )
        or ledger.get("stage1_oof_semantic_tensor_count") != 0
        or ledger.get("stage1_inner_any_view_semantic_tensor_count") != 0
        or ledger.get("stage2_model_fit_or_oof_semantic_tensor_count") != 0
        or ledger.get("stage3_model_fit_or_inner_semantic_tensor_count") != 0
        or ledger.get("inner_no_go_oof_semantic_tensor_read_count_exact0")
            is not (not inner_pass)
        or ledger.get("oof_first_semantic_materialization_after_selected_checkpoint_seal")
            is not inner_pass
        or fold.get("oof_semantic_tensor_materialized_count")
            != sum(expected_stage3_counts.values())
    ):
        raise ValueError("fold selective materialization ledger replay differs")
    _verify_rho_selection_ledger(fold)


def _verify_fold_split_against_authority(
    fold: Mapping[str, Any], records: Sequence[v4c.Record],
    outer_assignment: Mapping[str, int], config: Config,
) -> None:
    """Recompute model-fit/inner/OOF membership without trusting fold JSON."""

    fold_index = fold.get("fold_index")
    if type(fold_index) is not int or not 0 <= fold_index < OUTER_FOLDS:
        raise ValueError("fold index differs during independent split replay")
    groups, split = _split_fold(records, outer_assignment, fold_index, config)
    fit_iids = [row.iid for row in groups["model_fit"]]
    inner_iids = [row.iid for row in groups["inner_validation"]]
    inner_iid_family = [
        {"iid": row.iid, "family": row.family}
        for row in groups["inner_validation"]
    ]
    oof_iids = [row.iid for row in groups["exploratory_oof"]]
    training = fold.get("training")
    rho_selection = fold.get("rho_selection")
    embedded_inner_populations = []
    if type(rho_selection) is dict:
        embedded_inner_populations.append(rho_selection.get("rho0_inner_evidence"))
        if type(rho_selection.get("candidates")) is list:
            embedded_inner_populations.extend(
                candidate.get("inner_evidence")
                if type(candidate) is dict else None
                for candidate in rho_selection["candidates"]
            )
    if (
        type(training) is not dict
        or fold.get("inner_split") != split
        or fold.get("model_fit_ordered_iids") != fit_iids
        or fold.get("model_fit_original_count") != len(fit_iids)
        or fold.get("model_fit_iid_digest") != _object_sha(fit_iids)
        or training.get("model_fit_ordered_iids") != fit_iids
        or training.get("model_fit_iid_digest") != _object_sha(fit_iids)
        or training.get("inner_validation_ordered_iids") != inner_iids
        or training.get("inner_validation_original_count") != len(inner_iids)
        or training.get("inner_validation_iid_digest") != _object_sha(inner_iids)
        or fold.get("inner_validation_original_count") != len(inner_iids)
        or fold.get("inner_validation_ordered_iids") != inner_iids
        or fold.get("inner_validation_iid_digest") != _object_sha(inner_iids)
        or fold.get("oof_ordered_iids") != oof_iids
        or fold.get("oof_original_count") != len(oof_iids)
        or fold.get("oof_iid_digest") != _object_sha(oof_iids)
        or fold.get("rho_selection", {}).get("outer_fold") != fold_index
        or fold.get("rho_selection", {}).get(
            "cross_fold_inner_metric_aggregation_or_global_rho_selection"
        ) is not False
        or len(embedded_inner_populations) != len(RHO_GRID) + 1
        or any(
            type(population) is not list
            or [
                {"iid": row.get("iid"), "family": row.get("family")}
                for row in population
            ] != inner_iid_family
            for population in embedded_inner_populations
        )
    ):
        raise ValueError("fold split differs from independently replayed authority")


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """CPU-only aggregation of five already sealed fold artifacts."""

    _require_release_sealed()
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-F torch runtime differs")
    torch.set_num_threads(1)
    output = Path(args.output)
    if (
        not output.is_absolute() or not output.parent.is_dir()
        or output.exists() or output.is_symlink()
    ):
        raise ValueError("aggregate output must be a fresh absolute JSON child")
    if len(args.fold_root) != OUTER_FOLDS or len(set(args.fold_root)) != OUTER_FOLDS:
        raise ValueError("aggregate requires exactly five distinct --fold-root values")
    authority = _prepare_authorities(args)
    loaded = [
        _load_fold_receipt_sealed(root, run_binding) for root in args.fold_root
    ]
    loaded.sort(key=lambda item: item[0]["fold"]["fold_index"])
    receipts = [item[0] for item in loaded]
    bindings = [item[1] for item in loaded]
    if [receipt["fold"]["fold_index"] for receipt in receipts] != list(range(OUTER_FOLDS)):
        raise ValueError("sealed fold receipts are not folds 0 through 4 exactly once")
    if any(
        receipt.get("status") != STATUS
        or receipt.get("fold", {}).get("rho_selection", {}).get("inner_pass") is not True
        for receipt in receipts
    ):
        raise RuntimeError(
            "aggregate fail-closed: all five fold-local inner gates must PASS; "
            "INNER_NO_GO folds have zero OOF evidence and cannot be aggregated"
        )
    fold_rows = [receipt["fold"] for receipt in receipts]
    for fold in fold_rows:
        _verify_fold_split_against_authority(
            fold, authority["ordered_records"], authority["outer_assignment"], config
        )
        _verify_fold_selective_materialization_ledger(fold)
    checkpoint_artifacts = [
        fold["selected_checkpoint_artifact"] for fold in fold_rows
    ]
    preselection_artifacts = [
        fold["preselection_checkpoint_artifact"] for fold in fold_rows
    ]
    _verify_checkpoint_artifacts(
        preselection_artifacts, expected_role="preselection_fixed_step1200"
    )
    _verify_checkpoint_artifacts(
        checkpoint_artifacts, expected_role="selected_fold_local_rho"
    )
    if any(
        selected["preselection_base_state_sha256"]
            != preselection["preselection_base_state_sha256"]
        or selected["preselection_checkpoint_file_sha256"]
            != preselection["file_sha256"]
        for selected, preselection in zip(
            checkpoint_artifacts, preselection_artifacts
        )
    ):
        raise RuntimeError("aggregate selected/preselection checkpoint join differs")
    checkpoint_pair_joins = [
        _verify_distinct_checkpoint_pair(
            preselection, selected, float(fold["selected_rho"])
        )
        for fold, preselection, selected in zip(
            fold_rows, preselection_artifacts, checkpoint_artifacts
        )
    ]
    if checkpoint_pair_joins != [
        fold["preselection_selected_checkpoint_pair_join"] for fold in fold_rows
    ]:
        raise RuntimeError("aggregate checkpoint pair receipt replay differs")
    evidence = [row for receipt in receipts for row in receipt["oof_evidence"]]
    exact_iids = authority["exact_iids"]
    if (
        len(evidence) != 644 or len({row["iid"] for row in evidence}) != 644
        or {row["iid"] for row in evidence} != set(exact_iids)
        or tuple(sum(int(row["outer_fold"]) == fold for row in evidence)
                 for fold in range(OUTER_FOLDS)) != FROZEN_OOF_COUNTS
    ):
        raise ValueError("aggregate OOF union is not exact644 once each")
    upstream_match = _verify_v4c_embedded_teacher_evidence(
        evidence, authority["v4c_receipt"]
    )
    metrics = _aggregate(evidence, config)
    config_value = _config_value(config)
    selected_steps = [int(fold["training"]["selected_step"]) for fold in fold_rows]
    selected_rhos = [float(fold["selected_rho"]) for fold in fold_rows]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "authority": "burned_exposed_known_transform_development_only",
        "implementation": run_binding,
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "runtime": {
            "torch": str(torch.__version__), "torch_hip": str(torch.version.hip),
            "aggregate_device": "cpu",
            "model_schema_reconstructed_and_strict_loaded": True,
            "model_forward_executed": False,
            "model_trained_or_recomputed": False,
        },
        "feature_authority": {
            "feature_root": str(authority["feature_root"]),
            "feature_receipt_sha256": V4C_FEATURE_RECEIPT_SHA256,
            "feature_receipt_digest": authority["feature_receipt"]["receipt_digest"],
            "unique_original_iids": 644, "family_count": 28,
            "stored_views": list(EVAL_VIEWS),
            "all_five_views_are_separate_frozen_backbone_forwards": True,
        },
        "upstream_authorities": {
            "v4a_receipt_path": str(authority["v4a_path"].resolve(strict=True)),
            "v4a_receipt_file_sha256": V4A_RECEIPT_FILE_SHA256,
            "v4a_receipt_self_digest": V4A_RECEIPT_SELFDIGEST,
            "v4c_frontier_receipt_path": str(authority["v4c_path"].resolve(strict=True)),
            "v4c_frontier_receipt_file_sha256": V4C_FRONTIER_RECEIPT_SHA256,
            "v4c_frontier_receipt_self_digest": V4C_FRONTIER_RECEIPT_DIGEST,
            "v4c_embedded_teacher_evidence_match": upstream_match,
            "v4d_burned_receipt_path": V4D_RECEIPT_PATH,
            "v4d_burned_receipt_file_sha256": V4D_RECEIPT_SHA256,
            "v4d_burned_receipt_self_digest": V4D_RECEIPT_DIGEST,
            "v4d_burned_development_gate": False,
            "v4e_burned_implementation_sha256": V4E_BURNED_IMPLEMENTATION_SHA256,
            "v4e_burned_fold_receipt_sha256": list(
                V4E_BURNED_FOLD_RECEIPT_SHA256
            ),
            "v4e_burned_oof_informed_residual_homotopy_choice": True,
            "v4e_oof_used_to_select_fold_local_rho": False,
        },
        "fixed_comparator_authority": {
            "fixed_comparator_name": BASELINE_NAME,
            "v4c_burned_oof_informed_clip_pca_b384_choice": True,
            "v4e_oof_used_to_select_comparator": False,
            "v4e_burned_oof_informed_v4f_homotopy_choice": True,
            "one_predeclared_fold_local_selection_algorithm": True,
            "rho_candidate_count": len(RHO_GRID),
            "single_candidate": False,
            "fold_basis_fit_model_fit_original_only": True,
            "inner_validation_or_oof_used_for_basis_fit": False,
            "same_payload_384_scalars_only": True,
            "called_best_or_winner": False,
            "parameter_or_flop_fairness_claimed": False,
        },
        "frozen_split": {
            "outer_assignment_digest": v4c.OUTER_ASSIGNMENT_DIGEST,
            "fold_iid_digests": {str(k): v for k, v in v4c.FOLD_IID_DIGESTS.items()},
            "oof_counts_by_fold": list(FROZEN_OOF_COUNTS),
            "inner_source": INNER_SPLIT_NAMESPACE,
            "inner_literal_pins": list(FROZEN_INNER_SPLITS),
            "all_exact644_are_burned_development": True,
        },
        "model_contract": {
            "input": "C(view) FP32 [32,1024]", "code_shape": [12, 32],
            "actual_code_numel": 384, "decoder_input": "sole [12,32] code",
            "raw_input_skip_or_side_channel": False,
            "step0": "exact fold-fit fixed clip-PCA-B384 encoder/decoder",
            "encoder": "12 learned queries cross-attend all 32 projected frames",
            "decoder": "32 learned time queries cross-attend sole code",
            "exact_trainable_parameter_count": EXACT_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_exclusive": MAX_TRAINABLE_PARAMETERS,
            "latent_scale_or_rotation_gauge_fixed": False,
            "rho_fp32_power_of_two_buffer": True,
            "rho_scales_encoder_delta_and_decoder_residual": True,
            "raw_output_or_raw_input_blend": False,
        },
        "training_contract": {
            "all_five_known_views_exposed_for_each_model_fit_iid": True,
            "model_fit_five_view_tensors_used_for_gradient_and_model_input": True,
            "all_five_view_reconstruction_terms_equal_weight": True,
            "geometry_all_ten_unordered_pairs": True,
            "geometry_per_iid_scale": "stopgrad(mean_10_teacher_distance)+1e-8",
            "geometry_weight": config.geometry_weight,
            "view_axis_permutation_invariant": True,
            "view_name_positive_negative_role_family_action_or_strict_labels_enter_loss_or_model": False,
            "fixed_full_budget_no_early_stop": True,
            "fixed_preselection_step": FIXED_SELECTED_STEP,
            "checkpoint_winner_selection_performed": False,
            "training_rho": TRAINING_RHO,
            "inner_five_views_materialized_only_after_preselection_strong_seal": True,
            "inner_five_view_tensors_used_for_hyperparameter_selection": True,
            "inner_five_view_tensors_used_for_gradient_or_model_input": False,
            "rho_grid": list(RHO_GRID),
            "rho0_exact_comparator_only_not_selectable": True,
            "rho_candidate_count": len(RHO_GRID),
            "single_candidate": False,
            "transform_role_and_family_metadata_used_for_hyperparameter_selection": True,
            "transform_role_and_family_metadata_used_for_gradient": False,
            "transform_role_and_family_metadata_used_for_model_input": False,
            "teacher_and_fixed_pca_metadata_used_for_hyperparameter_selection": True,
            "teacher_and_fixed_pca_metadata_used_for_gradient_or_model_input": False,
            "each_outer_fold_selected_rho_independently": True,
            "cross_fold_inner_aggregation_or_global_rho_selection": False,
            "selection_rule": "evaluate exact7; choose first PASS in preregistered ascending order",
            "monotonic_metric_behavior_assumed": False,
            "smallest_rho_minimizes_distortion_claimed": False,
            "oof_selection": False,
            "selected_steps_by_fold": selected_steps,
            "selected_rhos_by_fold": selected_rhos,
        },
        "evaluation_contract": {
            "known_exposed_transform_families_only": True,
            "unseen_hostile_transform_gate_evaluated": False,
            "views": list(EVAL_VIEWS), "fixed_comparator": BASELINE_NAME,
            "final_oof_thresholds_unchanged_from_v4e": True,
            "final_oof_bootstrap_seed_namespace": "v4e exact namespace",
            "inner_rho_bootstrap_seed_namespace": "v4f fold-and-rho-local namespace",
            "fidelity_gate": "each view dual ratio UCB<=1.05 and every fold ratio<=1.05",
            "negative_gate": "each negative teacher/candidate/retention/improvement dual LCB>0 and every fold>0",
            "aggregate_cross_fold_or_across_negative_compensation_sufficient": False,
        },
        "fold_receipt_artifacts": {
            "count": len(bindings), "bindings": bindings,
            "all_single_fd_mode0444_nlink1": all(
                item["mode_octal"] == "0444" and item["nlink"] == 1
                for item in bindings
            ),
        },
        "folds": fold_rows,
        "selected_fold_checkpoint_artifacts": {
            "count": len(checkpoint_artifacts), "artifacts": checkpoint_artifacts,
            "artifacts_manifest_sha256": _object_sha(checkpoint_artifacts),
            "all_reverified_by_cpu_aggregate": True,
        },
        "preselection_fold_checkpoint_artifacts": {
            "count": len(preselection_artifacts),
            "artifacts": preselection_artifacts,
            "artifacts_manifest_sha256": _object_sha(preselection_artifacts),
            "all_reverified_by_cpu_aggregate": True,
            "all_selected_artifacts_join_same_fold_preselection_base_state": True,
            "distinct_checkpoint_pair_joins": checkpoint_pair_joins,
        },
        "oof_closure": {
            "unique_original_iids": 644, "each_original_evaluated_exactly_once": True,
            "oof_counts_by_fold": list(FROZEN_OOF_COUNTS),
            "embedded_per_iid_evidence_count": len(evidence),
            "embedded_per_iid_evidence_sha256": _object_sha(evidence),
            "embedded_per_iid_evidence": evidence,
            "evidence_sufficient_to_recompute_all_gates": True,
        },
        "metrics": metrics,
        "qualification_scope": {
            "exposed_five_view_codec_development_gate": metrics[
                "exposed_five_view_codec_development_gate"
            ],
            "unseen_hostile_transform_gate": False,
            "unseen_hostile_transform_gate_evaluated": False,
            "latent_metric_qualified": False, "action_representation_qualified": False,
            "identity_disentanglement_qualified": False,
            "identity_preservation_qualified": False, "vae_necessary": None,
            "generation_qualified": False, "prior_qualified": False,
            "prior_generation_qualified": False,
            "renderer_qualified": False, "video_editing_qualified": False,
            "inference_authorized": False, "web_evaluation_authorized": False,
            "full644_refit_authorized": False, "video_model_training_performed": False,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    _reverify_authorities(authority, args)
    binding_by_root = {item["fold_root"]: item for item in bindings}
    for original_root in args.fold_root:
        _, replayed = _load_fold_receipt_sealed(original_root, run_binding)
        if replayed != binding_by_root[replayed["fold_root"]]:
            raise RuntimeError("fold receipt changed before aggregate write")
    _verify_checkpoint_artifacts(
        preselection_artifacts, expected_role="preselection_fixed_step1200"
    )
    _verify_checkpoint_artifacts(
        checkpoint_artifacts, expected_role="selected_fold_local_rho"
    )
    receipt_sha = _write_json_create_only(output, receipt)
    reloaded_final = v4c._load_json_sealed(output, receipt_sha)
    reloaded_unsigned = dict(reloaded_final)
    reloaded_digest = reloaded_unsigned.pop("receipt_digest", None)
    if (
        reloaded_final != receipt
        or reloaded_final.get("schema_version") != SCHEMA
        or reloaded_final.get("status") != STATUS
        or reloaded_digest != receipt["receipt_digest"]
        or _object_sha(reloaded_unsigned) != reloaded_digest
    ):
        raise RuntimeError("fresh aggregate receipt strong self-read differs")
    _assert_binding_unchanged(run_binding)
    _reverify_authorities(authority, args)
    for original_root in args.fold_root:
        _, replayed = _load_fold_receipt_sealed(original_root, run_binding)
        if replayed != binding_by_root[replayed["fold_root"]]:
            raise RuntimeError("fold receipt changed after aggregate write")
    _verify_checkpoint_artifacts(
        preselection_artifacts, expected_role="preselection_fixed_step1200"
    )
    _verify_checkpoint_artifacts(
        checkpoint_artifacts, expected_role="selected_fold_local_rho"
    )
    return {
        "receipt": str(output.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "exposed_five_view_codec_development_gate": metrics[
            "exposed_five_view_codec_development_gate"
        ],
        "inference_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    _require_release_sealed()
    parser = argparse.ArgumentParser(
        description="NO-GO until sealed: v4-F fold-local residual homotopy"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_authorities(command: argparse.ArgumentParser) -> None:
        command.add_argument("--feature-root", required=True)
        command.add_argument("--expected-feature-receipt-sha256", required=True)
        command.add_argument("--v4a-receipt", required=True)
        command.add_argument("--expected-v4a-receipt-sha256", required=True)
        command.add_argument("--v4c-frontier-receipt", required=True)
        command.add_argument("--expected-v4c-frontier-receipt-sha256", required=True)
        command.add_argument("--v4d-receipt", required=True)
        command.add_argument("--expected-v4d-receipt-sha256", required=True)

    train = subparsers.add_parser("train-fold")
    add_authorities(train)
    train.add_argument("--fold-index", type=int, required=True)
    train.add_argument("--fold-root", required=True)
    train.add_argument("--device", default="cuda")
    train.set_defaults(handler=run_train_fold)

    aggregate = subparsers.add_parser("aggregate")
    add_authorities(aggregate)
    aggregate.add_argument("--fold-root", action="append", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=run_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_release_sealed()
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
