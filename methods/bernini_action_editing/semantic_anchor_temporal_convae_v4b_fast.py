#!/usr/bin/env python3
"""Burned-development exact5 Tucker-initialized temporal ConvAE canary.

The target is only the temporally centered action-anchor DINO sequence
``C(anchor)`` with shape ``[32,768]``.  Every fold starts at the fixed v4-A
Tucker-B384 map.  A zero-initialized encoder residual may change its sole
``[4,96]`` code and a zero-initialized decoder residual may reconstruct from
that code; the decoder has no access to the input and there are no skips.

Training reads model-fit originals and a separately pinned training-only
monotone warp.  It never constructs the v4-A evaluation positive or any of
the three evaluation negatives.  Training runs its full fixed budget.  A
checkpoint is chosen from fixed steps, including exact analytic step 0, by
inner-validation reconstruction only.  OOF tensors are first passed to
temporal transforms/model evaluation only after selection; no OOF tensor or
model value enters the optimizer or checkpoint selection.

This remains a development codec diagnostic.  Even a passing decoded-space
gate cannot qualify the latent metric (its gauge is not fixed), an action
representation, generation, rendering, inference, or video editing.
"""

from __future__ import annotations

import argparse
import copy
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

from methods.bernini_action_editing import semantic_anchor_linear_frontier_v4_fast as v4a


authority = v4a.authority
v2 = v4a.v2
SCHEMA = "semantic-anchor-temporal-convae-exact5-receipt-v4b-fast"
SEED = 20260819
TIME_STEPS = 32
FEATURE_DIM = 768
FULL_NUMEL = TIME_STEPS * FEATURE_DIM
OUTER_FOLDS = 5
CODE_TIME = 4
CODE_CHANNELS = 96
CODE_NUMEL = CODE_TIME * CODE_CHANNELS
MAX_TRAINABLE_PARAMETERS = 150000
BASELINE_NAME = "tucker_b0384_t04_r096"
V4A_RECEIPT_FILE_SHA256 = "568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2"
V4A_RECEIPT_SELFDIGEST = "f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86"
V4A_IMPLEMENTATION_SHA256 = "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973"
V2_SPLIT_SHA256 = "46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca"
FEATURE_AUTHORITY_SHA256 = "74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233"
FROZEN_OOF_COUNTS = (131, 127, 128, 129, 129)
NEGATIVES = v4a.NEGATIVES
EVAL_VIEWS = ("original", "monotone_warp", *NEGATIVES)

# This ABI is deliberately different from v4-A's e10f... evaluation positive.
TRAIN_WARP_COORDINATES = (
    0.000000000, 0.662272334, 1.439428806, 2.266795874,
    3.128555059, 4.016825676, 4.926812172, 5.855262756,
    6.799819469, 7.758686543, 8.730449677, 9.713962555,
    10.708276749, 11.712595940, 12.726236343, 13.748610497,
    14.779201508, 15.817556381, 16.863269806, 17.915983200,
    18.975366592, 20.041130066, 21.113002777, 22.190738678,
    23.274116516, 24.362924576, 25.456972122, 26.556081772,
    27.660089493, 28.768840790, 29.882188797, 31.000000000,
)
TRAIN_WARP_COORDINATES_SHA256 = "e08c6bb31a0767eaed9f81dd9330f06d8fa7db3453e8afc036e2b3fc6b24c137"


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
            raise ValueError("v4-B fast configuration is immutable")
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
    return authority.file_sha256(path)


def _binding() -> dict[str, str]:
    paths = {
        "implementation": Path(__file__).resolve(strict=True),
        "v4a_implementation": Path(v4a.__file__).resolve(strict=True),
        "v2_split_authority": Path(v2.__file__).resolve(strict=True),
        "feature_authority": Path(authority.__file__).resolve(strict=True),
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        result[f"{name}_path"] = str(path)
        result[f"{name}_sha256"] = _file_sha(path)
    if result["v4a_implementation_sha256"] != V4A_IMPLEMENTATION_SHA256:
        raise RuntimeError("v4-A implementation pin differs")
    if result["v2_split_authority_sha256"] != V2_SPLIT_SHA256:
        raise RuntimeError("v2 split pin differs")
    if result["feature_authority_sha256"] != FEATURE_AUTHORITY_SHA256:
        raise RuntimeError("feature authority pin differs")
    return result


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("implementation or authority changed during execution")


def _write_json_create_only(path: Path, value: Any) -> str:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("output must be a fresh absolute JSON child")
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                     allow_nan=False).encode("ascii") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    expected_sha = hashlib.sha256(raw).hexdigest()
    path_stat = path.stat()
    readback = path.read_bytes()
    if (
        stat.S_IMODE(path_stat.st_mode) != 0o444
        or path_stat.st_nlink != 1
        or path_stat.st_size != len(raw)
        or hashlib.sha256(readback).hexdigest() != expected_sha
    ):
        raise RuntimeError("fresh JSON receipt seal/readback differs")
    return expected_sha


def _train_warp_coordinate_tensor() -> torch.Tensor:
    coordinates = torch.tensor(TRAIN_WARP_COORDINATES, dtype=torch.float32)
    if (tuple(coordinates.shape) != (TIME_STEPS,)
            or float(coordinates[0]) != 0.0
            or float(coordinates[-1]) != 31.0
            or not bool((coordinates[1:] > coordinates[:-1]).all())
            or _tensor_sha(coordinates) != TRAIN_WARP_COORDINATES_SHA256
            or TRAIN_WARP_COORDINATES_SHA256 == v4a.PINNED_WARP_COORDINATES_SHA256
            or torch.equal(coordinates, v4a._warp_coordinate_tensor())):
        raise RuntimeError("training-only warp ABI differs or collides with evaluation")
    return coordinates


def training_only_monotone_warp(value: torch.Tensor) -> torch.Tensor:
    sequence = v4a.canonical_action(value)
    positions = _train_warp_coordinate_tensor().to(sequence.device)
    lower, upper = positions.floor().long(), positions.ceil().long()
    weight = (positions - lower.to(positions.dtype)).unsqueeze(1)
    warped = sequence.index_select(0, lower) * (1.0 - weight)
    warped = warped + sequence.index_select(0, upper) * weight
    return v4a.canonical_action(warped)


class TuckerInitializedTemporalConvAE(nn.Module):
    """One 384-scalar bottleneck; decoder input is exactly and only ``z``."""

    def __init__(self, fitted: v4a.FrontierFit, fit_only_rms: torch.Tensor) -> None:
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
        if count >= MAX_TRAINABLE_PARAMETERS:
            raise RuntimeError("ConvAE is not the preregistered low-capacity model")

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or tuple(value.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
            raise ValueError("encoder input geometry differs")
        # Every caller must supply the upstream C(view); do not apply C twice,
        # because step 0 must be the exact frozen v4-A Tucker encoder.
        if float(value.detach().mean(dim=1).abs().max().cpu()) > 1.0e-5:
            raise ValueError("encoder input is not upstream-temporally-centered C(view)")
        centered = value - self.frame_mean
        analytic = torch.einsum("tk,btd,dc->bkc", self.temporal_basis,
                                centered, self.content_basis)
        delta = self.encoder_delta((value / self.fit_only_rms).transpose(1, 2))
        delta = delta.transpose(1, 2) * self.fit_only_rms
        code = analytic + delta
        if tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS) or code[0].numel() != 384:
            raise RuntimeError("actual code is not [4,96]=384")
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if code.ndim != 3 or tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS):
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
        return output - output.mean(dim=1, keepdim=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(value))


def _fit_only_global_rms(rows: Sequence[authority.PairRecord], device: torch.device) -> torch.Tensor:
    values = torch.stack([v4a.canonical_action(row.anchor_sequence) for row in rows])
    rms = values.square().mean().sqrt().reshape(1).to(device)
    if not bool(torch.isfinite(rms).all()) or float(rms) <= 1.0e-8:
        raise ValueError("fit-only global RMS differs")
    return rms


def _canonical_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or tuple(value.shape[1:]) != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("canonical batch geometry differs")
    return value - value.mean(dim=1, keepdim=True)


def _training_warp_batch(value: torch.Tensor) -> torch.Tensor:
    value = _canonical_batch(value)
    positions = _train_warp_coordinate_tensor().to(value.device)
    lower, upper = positions.floor().long(), positions.ceil().long()
    weight = (positions - lower.to(positions.dtype)).reshape(1, TIME_STEPS, 1)
    output = value.index_select(1, lower) * (1.0 - weight)
    output = output + value.index_select(1, upper) * weight
    return _canonical_batch(output)


def _raw_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-2:] != (TIME_STEPS, FEATURE_DIM):
        raise ValueError("raw reconstruction geometry differs")
    return (prediction - target).square().mean()


def _fixed_training_loss(
    original_prediction: torch.Tensor, original_target: torch.Tensor,
    warp_prediction: torch.Tensor, warp_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Only original/fit-derived-warp reconstruction mechanics are optimized."""

    pairs = ((original_prediction, original_target), (warp_prediction, warp_target))
    raw = sum(F.smooth_l1_loss(left, right, beta=0.1) for left, right in pairs) / 2.0
    deltas: dict[int, torch.Tensor] = {}
    for stride in (1, 2, 4):
        deltas[stride] = sum(
            F.smooth_l1_loss(
                left[:, stride:] - left[:, :-stride],
                right[:, stride:] - right[:, :-stride], beta=0.1,
            )
            for left, right in pairs
        ) / 2.0
    terminal = sum(
        F.smooth_l1_loss(left[:, -1] - left[:, 0], right[:, -1] - right[:, 0], beta=0.1)
        for left, right in pairs
    ) / 2.0
    # Equivariance under the independently pinned training warp; no evaluation
    # positive or negative is reachable from this function.
    consistency = F.smooth_l1_loss(
        _training_warp_batch(original_prediction), warp_prediction, beta=0.1
    )
    total = raw + 0.20 * sum(deltas.values()) + 0.20 * terminal + 0.20 * consistency
    values = {
        "raw_feature": float(raw.detach().cpu()),
        "signed_delta_stride1": float(deltas[1].detach().cpu()),
        "signed_delta_stride2": float(deltas[2].detach().cpu()),
        "signed_delta_stride4": float(deltas[4].detach().cpu()),
        "terminal_displacement": float(terminal.detach().cpu()),
        "training_warp_consistency": float(consistency.detach().cpu()),
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
    fitted: v4a.FrontierFit, batch_size: int,
) -> dict[str, Any]:
    max_abs = 0.0
    squared_sum = 0.0
    numel = 0
    model_outputs: list[torch.Tensor] = []
    reference_outputs: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        actual = model(batch)
        reference = _analytic_tucker_decode(batch, fitted)
        difference = actual - reference
        max_abs = max(max_abs, float(difference.abs().max().cpu()))
        squared_sum += float(difference.double().square().sum().cpu())
        numel += difference.numel()
        model_outputs.append(actual.cpu())
        reference_outputs.append(reference.cpu())
    mse = squared_sum / numel
    actual_all = torch.cat(model_outputs)
    reference_all = torch.cat(reference_outputs)
    bit_exact = torch.equal(actual_all, reference_all)
    if not bit_exact or max_abs != 0.0 or mse != 0.0:
        raise RuntimeError("step-0 ConvAE is not the analytic Tucker comparator")
    return {
        "original_count": len(values),
        "input_sha256": _tensor_sha(values),
        "model_output_sha256": _tensor_sha(actual_all),
        "analytic_output_sha256": _tensor_sha(reference_all),
        "bit_exact": bit_exact,
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
    model_fit: Sequence[authority.PairRecord],
    inner_validation: Sequence[authority.PairRecord],
    fitted: v4a.FrontierFit, config: Config, fold_index: int,
    device: torch.device,
) -> tuple[TuckerInitializedTemporalConvAE, int, dict[str, Any]]:
    """Run every fixed step; select only by held inner original reconstruction."""

    seed = config.seed + 10000 + fold_index
    _seed_everything(seed, device)
    fit_original = torch.stack([
        v4a.canonical_action(row.anchor_sequence) for row in model_fit
    ]).to(device)
    validation_original = torch.stack([
        v4a.canonical_action(row.anchor_sequence) for row in inner_validation
    ]).to(device)
    # This is the only derived training view.  v4-A eval transformations are
    # intentionally not called anywhere in fitting.
    fit_warp = _training_warp_batch(fit_original).detach()
    rms = _fit_only_global_rms(model_fit, device)
    model = TuckerInitializedTemporalConvAE(fitted, rms).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters >= MAX_TRAINABLE_PARAMETERS:
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
        warp_target = fit_warp.index_select(0, indices)
        original_prediction = model(original_target)
        warp_prediction = model(warp_target)
        loss, last_components = _fixed_training_loss(
            original_prediction, original_target, warp_prediction, warp_target
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
        "model_fit_iid_digest": _object_sha([row.iid for row in model_fit]),
        "model_fit_original_tensor_sha256": _tensor_sha(fit_original),
        "model_fit_training_warp_tensor_sha256": _tensor_sha(fit_warp),
        "model_fit_derived_training_warp_rows": len(model_fit),
        "derived_training_warps_per_original": 1,
        "derived_training_warp_independent_sample_count": 0,
        "inner_validation_original_count": len(inner_validation),
        "inner_validation_iid_digest": _object_sha([row.iid for row in inner_validation]),
        "inner_validation_derived_views_used": 0,
        "oof_tensors_supplied_to_optimizer_or_checkpoint_selection": False,
        "source_rows_used": 0,
        "negative_rows_used": 0,
        "evaluation_positive_rows_used": 0,
        "family_or_transform_labels_used": False,
    }
    return model, selected_step, audit


def _analytic_tucker_decode(value: torch.Tensor, fitted: v4a.FrontierFit) -> torch.Tensor:
    """Decode the fixed B384 Tucker code and apply output C in raw coordinates."""

    batched = value.unsqueeze(0) if value.ndim == 2 else value
    frame_mean = fitted.frame_mean.to(batched.device)
    temporal_basis = fitted.temporal_basis.to(batched.device)
    content_basis = fitted.content_basis[:, :CODE_CHANNELS].to(batched.device)
    centered = batched - frame_mean
    code = torch.einsum("tk,btd,dc->bkc", temporal_basis, centered, content_basis)
    decoded = frame_mean + torch.einsum(
        "tk,bkc,dc->btd", temporal_basis, code, content_basis
    )
    decoded = _canonical_batch(decoded)
    return decoded[0] if value.ndim == 2 else decoded


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
    oof_rows: Sequence[authority.PairRecord],
    model: TuckerInitializedTemporalConvAE, selected_step: int,
    fitted: v4a.FrontierFit, config: Config, device: torch.device,
) -> list[dict[str, Any]]:
    """OOF is first passed to the temporal-transform/model evaluator here."""

    by_view: dict[str, list[torch.Tensor]] = {name: [] for name in EVAL_VIEWS}
    for row in oof_rows:
        # This is evaluation-only.  No fitting function calls temporal_variants.
        variants = v4a.temporal_variants(row.anchor_sequence, row.iid, v4a.Config())
        for name in EVAL_VIEWS:
            by_view[name].append(variants[name])
    stacked_cpu = {name: torch.stack(values) for name, values in by_view.items()}
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
            negative: v4a.distance_margin(
                stacked_cpu["original"][index].flatten(),
                stacked_cpu["monotone_warp"][index].flatten(),
                stacked_cpu[negative][index].flatten(),
            )
            for negative in NEGATIVES
        }
        tucker_spec = next(
            spec for spec in v4a.candidate_specs(v4a.Config())
            if spec["name"] == BASELINE_NAME
        )
        tucker_codes = {
            name: v4a._encode(stacked_cpu[name][index], tucker_spec, fitted)
            for name in EVAL_VIEWS
        }
        tucker_code_margin = {
            negative: v4a.distance_margin(
                tucker_codes["original"], tucker_codes["monotone_warp"],
                tucker_codes[negative],
            )
            for negative in NEGATIVES
        }
        baseline_margin = {
            negative: v4a.distance_margin(
                baseline["original"][index].flatten(),
                baseline["monotone_warp"][index].flatten(),
                baseline[negative][index].flatten(),
            )
            for negative in NEGATIVES
        }
        candidate_margin = {
            negative: v4a.distance_margin(
                candidate["original"][index].flatten(),
                candidate["monotone_warp"][index].flatten(),
                candidate[negative][index].flatten(),
            )
            for negative in NEGATIVES
        }
        decoded_vs_code_differences = {
            negative: abs(
                baseline_margin[negative]["margin"]
                - tucker_code_margin[negative]["margin"]
            )
            for negative in NEGATIVES
        }
        if max(decoded_vs_code_differences.values()) > 1.0e-7:
            raise RuntimeError("decoded Tucker margin is not the fixed v4-A code comparator")
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
            "v4a_tucker_b384_code_margin_by_negative": {
                name: tucker_code_margin[name]["margin"] for name in NEGATIVES
            },
            "decoded_tucker_vs_v4a_code_margin_abs_diff_by_negative": decoded_vs_code_differences,
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
    result = v4a._paired_bootstrap_lcbs(values, families, config, f"v4b:{label}")
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

    clip_seed = v4a._bootstrap_seed(config, "v4b", label, "ratio", "clip")
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
    family_seed = v4a._bootstrap_seed(config, "v4b", label, "ratio", "family")
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
        teacher = _paired_lcb(teacher_values, families, config, f"teacher:{negative}")
        baseline = _paired_lcb(baseline_values, families, config, f"tucker:{negative}")
        candidate = _paired_lcb(candidate_values, families, config, f"candidate:{negative}")
        retention = _paired_lcb(
            retention_values, families, config, f"candidate-minus-0p8-teacher:{negative}"
        )
        improvement = _paired_lcb(
            improvement_values, families, config, f"candidate-minus-tucker:{negative}"
        )
        per_fold_improvement = {
            str(fold): sum(
                value for value, row in zip(improvement_values, rows)
                if int(row["outer_fold"]) == fold
            ) / fold_counts[fold]
            for fold in range(OUTER_FOLDS)
        }
        if any(not math.isfinite(value) for value in per_fold_improvement.values()):
            raise ValueError("per-fold margin improvement is non-finite")
        all_folds_improve = all(value > 0.0 for value in per_fold_improvement.values())
        gate = bool(
            teacher["both_lcbs_strictly_gt_zero"]
            and candidate["both_lcbs_strictly_gt_zero"]
            and retention["both_lcbs_strictly_gt_zero"]
            and improvement["both_lcbs_strictly_gt_zero"]
            and all_folds_improve
        )
        negative_results[negative] = {
            "teacher_margin": teacher,
            "fixed_tucker_b384_margin": baseline,
            "candidate_margin": candidate,
            "candidate_minus_0p8_teacher_margin": retention,
            "candidate_minus_fixed_tucker_b384_margin": improvement,
            "per_fold_candidate_minus_fixed_tucker_b384_point_mean": per_fold_improvement,
            "all_five_fold_point_improvements_strictly_gt_zero": all_folds_improve,
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


def _load_v4a_receipt(path: Path, expected_feature_receipt_sha256: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    path_stat = path.stat()
    if (not path.is_file() or _file_sha(path) != V4A_RECEIPT_FILE_SHA256
            or stat.S_IMODE(path_stat.st_mode) != 0o444 or path_stat.st_nlink != 1):
        raise ValueError("v4-A receipt file SHA pin differs")
    value = json.loads(path.read_text(encoding="ascii"))
    if type(value) is not dict:
        raise ValueError("v4-A receipt is not an object")
    digest = value.get("receipt_digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    if digest != V4A_RECEIPT_SELFDIGEST or _object_sha(unsigned) != digest:
        raise ValueError("v4-A receipt self-digest differs")
    if (
        value.get("schema_version") != v4a.SCHEMA
        or value.get("implementation", {}).get("implementation_sha256")
            != V4A_IMPLEMENTATION_SHA256
        or value.get("feature_authority", {}).get("feature_receipt_sha256")
            != expected_feature_receipt_sha256
        or value.get("qualified_temporal_mechanics_candidates") != []
        or value.get("same_payload_frontier", {}).get("384", {}).get("candidate_names", []).count(BASELINE_NAME) != 1
        or value.get("oof_closure", {}).get("embedded_paired_margin_evidence_count") != 644
    ):
        raise ValueError("v4-A receipt semantic authority differs")
    return value


def _verify_v4a_embedded_evidence(
    rows: Sequence[Mapping[str, Any]], v4a_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    upstream_rows = v4a_receipt["oof_closure"]["embedded_paired_margin_evidence"]
    upstream = {row["iid"]: row for row in upstream_rows}
    if len(upstream) != 644 or set(upstream) != {row["iid"] for row in rows}:
        raise ValueError("v4-A/v4-B exact644 IID evidence closure differs")
    max_teacher = 0.0
    max_tucker_code = 0.0
    max_decoded_code = 0.0
    for row in rows:
        reference = upstream[row["iid"]]
        if (reference["family"] != row["family"]
                or int(reference["outer_fold"]) != int(row["outer_fold"])):
            raise ValueError("v4-A/v4-B family or fold authority differs")
        for negative in NEGATIVES:
            teacher_difference = abs(
                float(row["teacher_margin_by_negative"][negative])
                - float(reference["teacher_margin_by_negative"][negative])
            )
            tucker_difference = abs(
                float(row["v4a_tucker_b384_code_margin_by_negative"][negative])
                - float(reference["candidate_margin_by_name_and_negative"]
                        [BASELINE_NAME][negative])
            )
            decoded_difference = float(
                row["decoded_tucker_vs_v4a_code_margin_abs_diff_by_negative"][negative]
            )
            max_teacher = max(max_teacher, teacher_difference)
            max_tucker_code = max(max_tucker_code, tucker_difference)
            max_decoded_code = max(max_decoded_code, decoded_difference)
    if max_teacher > 1.0e-12 or max_tucker_code > 1.0e-12 or max_decoded_code > 1.0e-7:
        raise ValueError("v4-A embedded teacher/Tucker evidence does not bind v4-B")
    return {
        "exact644_iids_matched": True,
        "exact28_families_matched": len({row["family"] for row in rows}) == 28,
        "outer_fold_matched_per_iid": True,
        "max_abs_teacher_margin_difference_vs_v4a_receipt": max_teacher,
        "max_abs_tucker_code_margin_difference_vs_v4a_receipt": max_tucker_code,
        "max_abs_decoded_tucker_vs_v4a_code_margin_difference": max_decoded_code,
        "teacher_and_fixed_tucker_reference_tolerance": 1.0e-12,
        "decoded_vs_code_orthogonal_equivalence_tolerance": 1.0e-7,
    }


def _run_fold(
    pairs: Sequence[authority.PairRecord], fold_index: int, config: Config,
    device: torch.device, checkpoint_path: Path,
    run_binding: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups, split = v2._split_fold(pairs, fold_index, config.seed)
    if (
        split["outer_assignment_digest"] != v4a.V2_OUTER_ASSIGNMENT_DIGEST
        or split["iid_digest"] != v4a.V2_FOLD_IID_DIGESTS[fold_index]
        or len(groups["exploratory_oof"]) != FROZEN_OOF_COUNTS[fold_index]
    ):
        raise ValueError("fold is not the frozen v2/v4-A exact5 split")
    fit_iids = [row.iid for row in groups["model_fit"]]
    validation_iids = [row.iid for row in groups["early_stop_validation"]]
    oof_iids = [row.iid for row in groups["exploratory_oof"]]
    if (
        set(fit_iids) & set(validation_iids)
        or set(fit_iids) & set(oof_iids)
        or set(validation_iids) & set(oof_iids)
        or _object_sha(fit_iids) != split["model_fit_iid_digest"]
        or _object_sha(validation_iids) != split["early_stop_validation_iid_digest"]
        or _object_sha(oof_iids) != split["exploratory_oof_iid_digest"]
    ):
        raise ValueError("model-fit/inner-validation/OOF fold closure differs")

    # The prior-fixed comparator is refitted on model-fit originals exactly as
    # v4-A did; it is not an OOF-selected winner.
    fitted = v4a._fit_frontier(groups["model_fit"], v4a.Config(), torch.device("cpu"))
    model, selected_step, training_audit = _train_fold_model(
        groups["model_fit"], groups["early_stop_validation"], fitted,
        config, fold_index, device,
    )
    checkpoint = _save_selected_checkpoint_create_only(
        checkpoint_path, model, fitted, selected_step, training_audit,
        config, fold_index, run_binding, groups["early_stop_validation"], device,
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
        "frozen_v2_fold_iid_digest": split["iid_digest"],
        "frozen_v2_outer_assignment_digest": split["outer_assignment_digest"],
        "model_fit_original_count": len(fit_iids),
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


def _save_selected_checkpoint_create_only(
    path: Path, model: TuckerInitializedTemporalConvAE,
    fitted: v4a.FrontierFit, selected_step: int,
    training_audit: Mapping[str, Any], config: Config, fold_index: int,
    run_binding: Mapping[str, str], validation_rows: Sequence[authority.PairRecord],
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError("checkpoint must be a fresh absolute child")
    state = _state_to_cpu(model)
    state_sha = _state_sha(state)
    if (
        selected_step != int(training_audit["selected_step"])
        or state_sha != training_audit["selected_state_sha256"]
    ):
        raise RuntimeError("physical checkpoint does not join selected step/state")
    metadata: dict[str, Any] = {
        "schema_version": "semantic-anchor-temporal-convae-selected-fold-checkpoint-v4b-fast",
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
    os.chmod(path, 0o444)
    file_sha = _file_sha(path)
    file_stat = path.stat()
    if stat.S_IMODE(file_stat.st_mode) != 0o444 or file_stat.st_nlink != 1:
        raise RuntimeError("checkpoint seal differs")

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    loaded_metadata = loaded.get("metadata")
    loaded_state = loaded.get("state_dict")
    if (loaded_metadata != metadata or type(loaded_state) is not dict
            or _state_sha(loaded_state) != state_sha):
        raise RuntimeError("fresh checkpoint reload metadata/state differs")
    reloaded = TuckerInitializedTemporalConvAE(
        fitted, model.fit_only_rms.detach().cpu()
    )
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.to(device).eval()
    probe = torch.stack([
        v4a.canonical_action(row.anchor_sequence)
        for row in validation_rows[:min(config.batch_size, len(validation_rows))]
    ]).to(device)
    with torch.no_grad():
        expected = model(probe)
        actual = reloaded(probe)
    if not torch.equal(expected, actual):
        raise RuntimeError("fresh checkpoint strict reload output is not bit-exact")
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha,
        "size_bytes": file_stat.st_size,
        "mode_octal": "0444",
        "nlink": file_stat.st_nlink,
        "outer_fold": fold_index,
        "selected_step": selected_step,
        "model_state_sha256": state_sha,
        "selected_training_audit_state_join_verified": True,
        "metadata_digest": metadata["metadata_digest"],
        "fresh_reload_strict_state_verified": True,
        "fresh_reload_output_bit_exact": True,
    }


def _verify_checkpoint_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> None:
    if len(artifacts) != OUTER_FOLDS:
        raise RuntimeError("selected checkpoint artifact count differs")
    for fold, artifact in enumerate(artifacts):
        path = Path(str(artifact["path"]))
        path_stat = path.stat()
        if (
            int(artifact["outer_fold"]) != fold
            or not path.is_file()
            or path_stat.st_size != int(artifact["size_bytes"])
            or stat.S_IMODE(path_stat.st_mode) != 0o444
            or path_stat.st_nlink != 1
            or _file_sha(path) != artifact["file_sha256"]
        ):
            raise RuntimeError("sealed selected checkpoint artifact changed")


def run_exact5(args: argparse.Namespace) -> dict[str, Any]:
    run_binding = _binding()
    config = Config()
    config.validate()
    device = _resolve_device(args.device)
    output = Path(args.output)
    if not output.is_absolute() or not output.parent.is_dir() or output.exists():
        raise ValueError("output must be a fresh absolute JSON child")
    checkpoint_paths = [
        output.with_name(f"{output.stem}.selected_fold{fold}.pt")
        for fold in range(OUTER_FOLDS)
    ]
    if len(set(checkpoint_paths)) != OUTER_FOLDS or any(
        path.exists() or path.parent != output.parent for path in checkpoint_paths
    ):
        raise ValueError("all five selected checkpoint paths must be fresh siblings")
    v4a_receipt_path = Path(args.v4a_receipt)
    upstream_receipt = _load_v4a_receipt(
        v4a_receipt_path, args.expected_feature_receipt_sha256
    )
    pairs, feature_receipt = authority.load_exact644_pairs(
        Path(args.feature_root), args.expected_feature_receipt_sha256
    )
    population = v2._exact644_population_authority(pairs)
    exact_iids = [row.iid for row in pairs]
    if (
        len(pairs) != 644 or len(set(exact_iids)) != 644
        or len({row.family for row in pairs}) != 28
        or _object_sha(exact_iids)
            != upstream_receipt["feature_authority"]["exact644_ordered_iid_digest"]
    ):
        raise ValueError("feature authority/v4-A exact644 population differs")

    fold_receipts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for fold_index in range(OUTER_FOLDS):
        fold_receipt, rows = _run_fold(
            pairs, fold_index, config, device, checkpoint_paths[fold_index],
            run_binding,
        )
        fold_receipts.append(fold_receipt)
        all_rows.extend(rows)
    if sorted(row["iid"] for row in all_rows) != exact_iids:
        raise ValueError("exact5 OOF union is not exact644 once each")
    upstream_match = _verify_v4a_embedded_evidence(all_rows, upstream_receipt)
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
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "V4B_FAST_EXACT5_TEMPORAL_CONVAE_COMPLETE_BURNED_DEVELOPMENT",
        "implementation": run_binding,
        "config": config_value,
        "config_sha256": _object_sha(config_value),
        "device": str(device),
        "deterministic_algorithms_required": True,
        "feature_authority": {
            "feature_root": str(Path(args.feature_root).resolve(strict=True)),
            "feature_receipt_sha256": args.expected_feature_receipt_sha256,
            "feature_receipt_digest": feature_receipt["receipt_digest"],
            "population_authority": population,
            "exact644_ordered_iid_digest": _object_sha(exact_iids),
            "source_tensor_materialized_by_upstream_loader_but_never_read_here": True,
        },
        "v4a_prior_fixed_comparator_authority": {
            "receipt_path": str(v4a_receipt_path.resolve(strict=True)),
            "receipt_file_sha256": V4A_RECEIPT_FILE_SHA256,
            "receipt_self_digest": V4A_RECEIPT_SELFDIGEST,
            "fixed_comparator_name": BASELINE_NAME,
            "chosen_before_v4b_oof": True,
            "called_best_or_winner": False,
            "same_payload_384_scalars_only": True,
            "parameter_or_flop_fairness_claimed": False,
            "embedded_evidence_match": upstream_match,
        },
        "frozen_split": {
            "source": "semantic_anchor_action_sequence_vae_v2._split_fold",
            "outer_assignment_digest": v4a.V2_OUTER_ASSIGNMENT_DIGEST,
            "fold_iid_digests": v4a.V2_FOLD_IID_DIGESTS,
            "oof_counts_by_fold": list(FROZEN_OOF_COUNTS),
            "all_exact644_are_development": True,
            "fresh_scientific_confirmation_claimed": False,
            "iid_disjoint_only_not_actor_scene_generator_lineage_disjoint": True,
        },
        "model_contract": {
            "input": "C(anchor/view) [32,768]",
            "fit_only_normalization": "single global RMS from model-fit originals",
            "code_shape": [CODE_TIME, CODE_CHANNELS],
            "actual_code_numel": CODE_NUMEL,
            "decoder_input": "sole [4,96] code",
            "raw_input_skip_or_side_channel": False,
            "step0": "exact prior-fixed v4-A Tucker-B384 encoder/decoder",
            "encoder_residual_final_layer_zero_initialized": True,
            "decoder_residual_final_layer_zero_initialized": True,
            "decoder_output_temporally_centered": True,
            "latent_scale_or_rotation_gauge_fixed": False,
            "latent_distance_used_for_gate_or_report": False,
            "trainable_parameter_limit_exclusive": MAX_TRAINABLE_PARAMETERS,
        },
        "training_contract": {
            "model_fit_original_anchor_only": True,
            "derived_training_view": "one independently pinned monotone warp per model-fit original",
            "training_warp_coordinates": list(TRAIN_WARP_COORDINATES),
            "training_warp_coordinates_tensor_sha256": TRAIN_WARP_COORDINATES_SHA256,
            "v4a_evaluation_positive_coordinates_tensor_sha256": v4a.PINNED_WARP_COORDINATES_SHA256,
            "training_and_evaluation_positive_warp_abis_disjoint": True,
            "loss_terms": ["raw_feature", "signed_delta_stride1", "signed_delta_stride2",
                           "signed_delta_stride4", "terminal_displacement",
                           "training_warp_consistency"],
            "negative_views_used_for_training": 0,
            "v4a_evaluation_positive_views_used_for_training": 0,
            "source_rows_used_for_training": 0,
            "family_or_transform_labels_used_for_training": False,
            "fixed_full_budget_no_early_stop": True,
            "inner_validation_checkpoint_selection_original_reconstruction_only": True,
            "oof_selection": False,
            "selected_steps_by_fold": selected_steps,
        },
        "oof_access_contract": {
            "exact1288_feature_authority_materialized_at_command_start": True,
            "frozen_v2_split_computation_reads_exact644_anchor_energy_metadata": True,
            "v4a_upstream_oof_evidence_preloaded_for_later_binding": True,
            "oof_tensors_supplied_to_optimizer_or_checkpoint_selection": False,
            "oof_temporal_transforms_or_model_outputs_computed_before_selection": False,
            "claim_is_no_oof_model_value_use_not_no_prior_materialization": True,
        },
        "evaluation_contract": {
            "primary_mapping": "decoded C(D(E(C(view)))) in original [32,768] coordinates",
            "views": list(EVAL_VIEWS),
            "fixed_comparator": BASELINE_NAME,
            "fidelity_gate": "each view: paired clip/family ratio-of-means UCB<=1.05 and every fold point ratio<=1.05",
            "negative_gate": "each negative: candidate margin, candidate-0.8*teacher, candidate-Tucker both LCB>0; every fold candidate-Tucker point mean>0",
            "aggregate_cross_fold_compensation_sufficient": False,
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
            "identity_preservation_qualified": False,
            "vae_necessary": None,
            "generation_qualified": False,
            "prior_qualified": False,
            "renderer_qualified": False,
            "video_editing_qualified": False,
            "inference_authorized": False,
            "web_evaluation_authorized": False,
            "video_model_training_performed": False,
            "fold_local_model_fit_performed": True,
            "postselection_all644_refit_authorized_or_performed": False,
            "fresh_confirmation_requires_new_external_group_disjoint_data": True,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    if _file_sha(v4a_receipt_path.resolve(strict=True)) != V4A_RECEIPT_FILE_SHA256:
        raise RuntimeError("v4-A receipt changed during execution")
    receipt_sha = _write_json_create_only(output, receipt)
    _verify_checkpoint_artifacts(checkpoint_artifacts)
    _assert_binding_unchanged(run_binding)
    return {
        "receipt": str(output.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "decoded_temporal_codec_development_gate": metrics[
            "decoded_temporal_codec_development_gate"
        ],
        "latent_metric_qualified": False,
        "selected_checkpoint_artifacts_reverified_after_receipt_write": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact5 Tucker-initialized action-anchor temporal ConvAE canary"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-exact5")
    run.add_argument("--feature-root", required=True)
    run.add_argument("--expected-feature-receipt-sha256", required=True)
    run.add_argument("--v4a-receipt", required=True)
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
