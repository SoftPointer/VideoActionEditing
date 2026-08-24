#!/usr/bin/env python3
"""Future-use v4-G role-directed teacher-margin codec.

This is a burned, known-transform-exposed development diagnostic.  It keeps
the v4-F fold-fit clip-PCA-B384 initialization, sole ``[12,32]`` code, 79,040
trainable parameters, exact outer/inner splits, five-view reconstruction,
optimizer, batch size, fixed 1,200-step budget, and unchanged inner/final gates.

The sole scientific change was preregistered before the v4-F result: replace
the exchangeable ten-pair geometry training term with three role-directed
decoded teacher-margin SmoothL1 terms.  ``original`` is the query,
``monotone_warp`` the positive, and the three named hostile transforms are
negatives.  Each IID is scaled by stopgrad(mean of all ten teacher pair
distances)+1e-8.  Weight and SmoothL1 beta are fixed at 0.25 and 0.1.

There is one fixed candidate only: step 1200 at residual scale one.  A
``train-fold`` process writes only two independently sealed checkpoints and an
    inner receipt; it cannot read OOF tensors.  A separate controller-only
    ``verify-inner-barrier`` process recomputes model-fit provenance and all five
    inner checkpoint forwards into one create-only barrier receipt.  An
    ``evaluate-fold`` process accepts only that controller-pinned barrier and
    independently repeats it before the first OOF tensor request.  One failed
    inner gate therefore leaves all-fold OOF reads exact zero and forbids
    aggregation.

No result here can qualify unseen transforms, latent/action/identity
representations, generation, rendering, inference, web evaluation, video
editing, or full-644 refitting.
"""

from __future__ import annotations

import argparse
import ctypes
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

from methods.bernini_action_editing import (
    semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy
    as frozen,
)


v4c = frozen.v4c
v4a = frozen.v4a
features = frozen.features

SCHEMA = "semantic-anchor-vjepa2-role-directed-teacher-margin-exact5-receipt-v4g"
INNER_SCHEMA = "semantic-anchor-vjepa2-role-directed-teacher-margin-inner-receipt-v4g"
FOLD_SCHEMA = "semantic-anchor-vjepa2-role-directed-teacher-margin-fold-receipt-v4g"
CHECKPOINT_SCHEMA = "semantic-anchor-vjepa2-role-directed-teacher-margin-checkpoint-v4g"
BARRIER_SCHEMA = (
    "semantic-anchor-vjepa2-role-directed-teacher-margin-global-inner-barrier-v4g"
)
STATUS = "V4G_ROLE_DIRECTED_TEACHER_MARGIN_KNOWN_EXPOSED_DEVELOPMENT"
INNER_PASS_STATUS = "V4G_FIXED1200_INNER_PASS_OOF_UNREAD"
INNER_NO_GO_STATUS = "V4G_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD"
BARRIER_PASS_STATUS = "V4G_EXACT5_INNER_BARRIER_PASS_OOF_UNREAD"

# A detached manifest/controller must one-way pin runtime/tests/tree.  Runtime
# never reverse-pins those authorities, avoiding a SHA cycle.
RELEASE_SEALED = True

V4F_RUNTIME_DEPENDENCY_SHA256 = (
    "97cd77e64a4dfaf3036e6c50a5b85060fd616f87371e5d967e69db1170466d74"
)
SEED = frozen.SEED
TIME_STEPS = frozen.TIME_STEPS
FEATURE_DIM = frozen.FEATURE_DIM
FULL_NUMEL = frozen.FULL_NUMEL
OUTER_FOLDS = frozen.OUTER_FOLDS
CODE_TIME = frozen.CODE_TIME
CODE_CHANNELS = frozen.CODE_CHANNELS
CODE_NUMEL = frozen.CODE_NUMEL
MAX_TRAINABLE_PARAMETERS = frozen.MAX_TRAINABLE_PARAMETERS
EXACT_TRAINABLE_PARAMETERS = frozen.EXACT_TRAINABLE_PARAMETERS
BASELINE_NAME = frozen.BASELINE_NAME
FIXED_SELECTED_STEP = 1200
FIXED_RESIDUAL_SCALE = 1.0
FIXED_CANDIDATE_COUNT = 1
SCIENTIFIC_DESIGN_PREREGISTERED_BEFORE_V4F_RESULT = True
FROZEN_OOF_COUNTS = frozen.FROZEN_OOF_COUNTS
FROZEN_INNER_SPLITS = frozen.FROZEN_INNER_SPLITS
INNER_SPLIT_NAMESPACE = frozen.INNER_SPLIT_NAMESPACE
NEGATIVES = frozen.NEGATIVES
EVAL_VIEWS = frozen.EVAL_VIEWS
ClipPCAFit = frozen.ClipPCAFit


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
    teacher_margin_weight: float = 0.25
    teacher_margin_beta: float = 0.1

    def validate(self) -> None:
        if self != Config():
            raise ValueError("v4-G configuration is immutable")
        if (
            self.max_steps != FIXED_SELECTED_STEP
            or self.checkpoint_steps != (0, FIXED_SELECTED_STEP)
            or FIXED_RESIDUAL_SCALE != 1.0
            or FIXED_CANDIDATE_COUNT != 1
            or SCIENTIFIC_DESIGN_PREREGISTERED_BEFORE_V4F_RESULT is not True
            or (CODE_TIME, CODE_CHANNELS, CODE_NUMEL) != (12, 32, 384)
            or EXACT_TRAINABLE_PARAMETERS != 79040
            or self.teacher_margin_weight != 0.25
            or self.teacher_margin_beta != 0.1
        ):
            raise ValueError("v4-G fixed-candidate teacher-margin contract differs")


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
    # ``bytes(UntypedStorage)`` iterates byte-by-byte in Python on some Torch
    # builds.  The clone is contiguous, CPU-owned, offset zero, so this reads
    # the identical byte interval without changing the digest definition.
    digest.update(ctypes.string_at(
        tensor.data_ptr(), tensor.numel() * tensor.element_size()
    ))
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    return features.file_sha256(path)


def _require_release_sealed() -> None:
    """Fail before parsing arguments, resolving devices, or touching outputs."""

    if RELEASE_SEALED is not True:
        raise RuntimeError(
            "UNSEALED v4-G teacher-margin candidate: detached release not sealed"
        )


def _binding() -> dict[str, str]:
    dependency = Path(frozen.__file__).resolve(strict=True)
    dependency_sha = _file_sha(dependency)
    if dependency_sha != V4F_RUNTIME_DEPENDENCY_SHA256:
        raise RuntimeError("frozen final v4-F runtime dependency differs")
    upstream = frozen._binding()
    return {
        "implementation_path": str(Path(__file__).resolve(strict=True)),
        "implementation_sha256": _file_sha(Path(__file__).resolve(strict=True)),
        "frozen_v4f_runtime_path": str(dependency),
        "frozen_v4f_runtime_sha256": dependency_sha,
        "v4c_implementation_sha256": upstream["v4c_implementation_sha256"],
        "extractor_implementation_sha256": upstream[
            "extractor_implementation_sha256"
        ],
        "v4a_implementation_sha256": upstream["v4a_implementation_sha256"],
        "v4d_implementation_sha256": upstream["v4d_implementation_sha256"],
        "v4e_burned_implementation_sha256": upstream[
            "v4e_burned_implementation_sha256"
        ],
    }


def _assert_binding_unchanged(expected: Mapping[str, str]) -> None:
    if _binding() != expected:
        raise RuntimeError("implementation or frozen authority changed")


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
        "teacher_margin_weight": config.teacher_margin_weight,
        "teacher_margin_beta": config.teacher_margin_beta,
        "code_shape": [CODE_TIME, CODE_CHANNELS],
        "actual_code_numel": CODE_NUMEL,
        "fixed_step": FIXED_SELECTED_STEP,
        "fixed_residual_scale": FIXED_RESIDUAL_SCALE,
        "candidate_count": FIXED_CANDIDATE_COUNT,
        "single_candidate": True,
        "hyperparameter_selection_performed": False,
        "scientific_design_preregistered_before_v4f_result": True,
    }


def _qualification_scope(
    exposed_five_view_codec_development_gate: bool | None,
) -> dict[str, Any]:
    """Complete non-generalizing v4-F qualification/authorization surface."""

    return {
        "exposed_five_view_codec_development_gate": (
            exposed_five_view_codec_development_gate
        ),
        "unseen_hostile_transform_gate": False,
        "unseen_hostile_transform_gate_evaluated": False,
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
        "video_model_training_performed": False,
    }


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


def _runtime_fingerprint(device: torch.device) -> dict[str, Any]:
    """Bind numerical runtime and device class, not a replaceable device index."""

    result: dict[str, Any] = {
        "torch": str(torch.__version__),
        "torch_hip": str(torch.version.hip),
        "device_type": device.type,
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "autocast_used": False,
        "full_precision_fp32": True,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["device_class"] = {
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
            "gcn_arch_name": str(getattr(properties, "gcnArchName", "")),
        }
    else:
        result["device_class"] = {"name": "cpu"}
    return result


class CrossAttention32(nn.Module):
    """Pinned single-head 32-channel cross-attention."""

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
        result = self.output(torch.matmul(
            torch.softmax(logits, dim=-1), self.value(context)
        ))
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("cross-attention output is non-finite")
        return result


class ClipPCAInitializedVJepaGlobalCodec(nn.Module):
    """Sole [12,32] code; the only residual scale is literal one."""

    def __init__(self, fitted: ClipPCAFit, fit_only_rms: torch.Tensor) -> None:
        super().__init__()
        if (
            tuple(fit_only_rms.shape) != (1,)
            or not bool(torch.isfinite(fit_only_rms).all())
            or float(fit_only_rms) <= 0.0
        ):
            raise ValueError("fit-only global RMS geometry differs")
        if (
            tuple(fitted.clip_mean.shape) != (1, FULL_NUMEL)
            or tuple(fitted.clip_basis.shape) != (FULL_NUMEL, CODE_NUMEL)
        ):
            raise ValueError("pinned clip-PCA basis geometry differs")
        self.register_buffer("fit_only_rms", fit_only_rms.detach().reshape(1))
        self.register_buffer("clip_mean", fitted.clip_mean.detach())
        self.register_buffer("clip_basis", fitted.clip_basis.detach())
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
            raise RuntimeError("v4-G codec parameter closure differs")

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        if (
            value.ndim != 3 or value.shape[0] == 0
            or tuple(value.shape[1:]) != (TIME_STEPS, FEATURE_DIM)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("encoder input geometry differs")
        if float(value.detach().mean(dim=1).abs().max().cpu()) > 1.0e-5:
            raise ValueError("encoder input is not upstream-centered C(view)")
        analytic = (
            (value.flatten(1) - self.clip_mean) @ self.clip_basis
        ).reshape(-1, CODE_TIME, CODE_CHANNELS)
        tokens = self.input_projection(value / self.fit_only_rms)
        tokens = tokens + self.input_position.unsqueeze(0)
        queries = self.code_queries.unsqueeze(0).expand(value.shape[0], -1, -1)
        attended = self.encoder_attention(queries, tokens)
        delta = self.encoder_delta(self.encoder_norm(attended)) * self.fit_only_rms
        code = (analytic + delta * FIXED_RESIDUAL_SCALE).contiguous()
        if (
            tuple(code.shape[1:]) != (CODE_TIME, CODE_CHANNELS)
            or code[0].numel() != CODE_NUMEL or code.dtype != torch.float32
            or not bool(torch.isfinite(code).all())
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
        output = analytic + residual * self.fit_only_rms * FIXED_RESIDUAL_SCALE
        result = (output - output.mean(dim=1, keepdim=True)).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("decoder output is non-finite")
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(value))


VJepa2GlobalCodec = ClipPCAInitializedVJepaGlobalCodec


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    if any(not bool(value.isfinite().all()) for value in state.values()):
        raise RuntimeError("checkpoint contains non-finite state")
    return state


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    return _object_sha({name: _tensor_sha(state[name]) for name in sorted(state)})


def _single_view_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if (
        prediction.shape != target.shape or prediction.ndim != 4
        or tuple(prediction.shape[-2:]) != (TIME_STEPS, FEATURE_DIM)
        or prediction.shape[0] == 0 or prediction.shape[1] != len(EVAL_VIEWS)
    ):
        raise ValueError("five-view reconstruction geometry differs")

    def equal_view_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        elementwise = F.smooth_l1_loss(left, right, beta=0.1, reduction="none")
        if elementwise.ndim == 4:
            per_view = elementwise.mean(dim=(0, 2, 3))
        elif elementwise.ndim == 3:
            per_view = elementwise.mean(dim=(0, 2))
        else:
            raise RuntimeError("equal-view loss rank differs")
        return torch.sort(per_view).values.mean()

    raw = equal_view_loss(prediction, target)
    deltas = {
        stride: equal_view_loss(
            prediction[:, :, stride:] - prediction[:, :, :-stride],
            target[:, :, stride:] - target[:, :, :-stride],
        )
        for stride in (1, 2, 4)
    }
    terminal = equal_view_loss(
        prediction[:, :, -1] - prediction[:, :, 0],
        target[:, :, -1] - target[:, :, 0],
    )
    return raw + 0.20 * sum(deltas.values()) + 0.20 * terminal, {
        "raw_feature": raw,
        "signed_delta_stride1": deltas[1],
        "signed_delta_stride2": deltas[2],
        "signed_delta_stride4": deltas[4],
        "terminal_displacement": terminal,
    }


def _multiview_training_loss(
    prediction: torch.Tensor, target: torch.Tensor,
    teacher_margin_weight: float = 0.25, teacher_margin_beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Five-view reconstruction plus exact-three role-directed margins."""

    if teacher_margin_weight != 0.25 or teacher_margin_beta != 0.1:
        raise ValueError("teacher-margin loss constants are immutable")
    reconstruction, terms = _single_view_reconstruction_loss(prediction, target)
    teacher_distances = []
    for left in range(len(EVAL_VIEWS)):
        for right in range(left + 1, len(EVAL_VIEWS)):
            teacher_distances.append(
                (target[:, left] - target[:, right]).square().mean(dim=(1, 2))
            )
    if len(teacher_distances) != 10:
        raise RuntimeError("mean-ten teacher-distance scale closure differs")
    per_iid_scale = (
        torch.stack(teacher_distances, dim=1).detach().mean(dim=1, keepdim=True)
        + 1.0e-8
    )
    role_index = {name: EVAL_VIEWS.index(name) for name in EVAL_VIEWS}
    query_index = role_index["original"]
    positive_index = role_index["monotone_warp"]

    def decoded_margin(value: torch.Tensor, negative: str) -> torch.Tensor:
        query = value[:, query_index]
        positive = value[:, positive_index]
        hostile = value[:, role_index[negative]]
        return (
            (query - hostile).square().mean(dim=(1, 2))
            - (query - positive).square().mean(dim=(1, 2))
        )

    teacher_margins = torch.stack([
        decoded_margin(target, negative) for negative in NEGATIVES
    ], dim=1)
    candidate_margins = torch.stack([
        decoded_margin(prediction, negative) for negative in NEGATIVES
    ], dim=1)
    if tuple(teacher_margins.shape) != (prediction.shape[0], 3):
        raise RuntimeError("exact-three decoded teacher-margin closure differs")
    normalized_error = (candidate_margins - teacher_margins) / per_iid_scale
    margin_loss = F.smooth_l1_loss(
        normalized_error, torch.zeros_like(normalized_error),
        beta=teacher_margin_beta, reduction="mean",
    )
    total = reconstruction + teacher_margin_weight * margin_loss
    if not bool(torch.isfinite(total)):
        raise RuntimeError("training loss is non-finite")
    values = {name: float(value.detach().cpu()) for name, value in terms.items()}
    values.update({
        "equal_view_reconstruction": float(reconstruction.detach().cpu()),
        "exact_three_role_directed_decoded_teacher_margin_smooth_l1": float(
            margin_loss.detach().cpu()
        ),
        "teacher_distance_scale_all_ten_pairs_stopgrad_min": float(
            per_iid_scale.detach().min().cpu()
        ),
        "teacher_margin_weight": teacher_margin_weight,
        "teacher_margin_beta": teacher_margin_beta,
        "transform_roles_used_for_gradient_loss": True,
        "total": float(total.detach().cpu()),
    })
    return total, values


def _train_fold_model(
    model_fit: Sequence[v4c.Record], inner_validation_iids: Sequence[str],
    fitted: ClipPCAFit, config: Config, fold_index: int, device: torch.device,
) -> tuple[VJepa2GlobalCodec, dict[str, Any]]:
    """Fit the sole step-1200 candidate; inner IIDs are metadata only."""

    seed = config.seed + 10000 + fold_index
    _seed_everything(seed, device)
    fit_views = torch.stack([
        torch.stack([v4c.canonical_action(row.views[name]) for name in EVAL_VIEWS])
        for row in model_fit
    ]).to(device)
    if tuple(fit_views.shape[1:]) != (
        len(EVAL_VIEWS), TIME_STEPS, FEATURE_DIM
    ):
        raise ValueError("model-fit exposed-five-view geometry differs")
    fit_original = fit_views[:, EVAL_VIEWS.index("original")]
    rms = frozen._fit_only_global_rms(model_fit, device)
    model = VJepa2GlobalCodec(fitted, rms).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXACT_TRAINABLE_PARAMETERS:
        raise RuntimeError("exact trainable parameter closure differs")
    step0 = frozen._step0_equivalence(
        model, fit_views.flatten(0, 1), fitted, config.batch_size
    )
    step0_state = _state_to_cpu(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    minibatches = torch.randint(
        len(model_fit), (config.max_steps, config.batch_size), generator=generator
    )
    last_components: dict[str, float] | None = None
    model.train()
    for step in range(1, config.max_steps + 1):
        indices = minibatches[step - 1].to(device)
        target = fit_views.index_select(0, indices)
        prediction = model(target.flatten(0, 1)).reshape_as(target)
        loss, last_components = _multiview_training_loss(
            prediction, target, config.teacher_margin_weight,
            config.teacher_margin_beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError("training gradient is non-finite")
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("trained parameter is non-finite")
    if last_components is None:
        raise RuntimeError("full training budget did not execute")
    model.eval()
    final_state = _state_to_cpu(model)
    audit = {
        "fold_seed": seed,
        "runtime_fingerprint": _runtime_fingerprint(device),
        "full_budget_steps_executed": config.max_steps,
        "early_stopped": False,
        "fixed_step": FIXED_SELECTED_STEP,
        "checkpoint_winner_selection_performed": False,
        "hyperparameter_selection_performed": False,
        "step0_state_sha256": _state_sha(step0_state),
        "final_step_state_sha256": _state_sha(final_state),
        "selected_state_sha256": _state_sha(final_state),
        "last_training_loss_components": last_components,
        "trainable_parameter_count": parameter_count,
        "fit_only_global_rms": float(rms.detach().cpu()),
        "fit_only_global_rms_sha256": _tensor_sha(rms),
        "step0_model_fit_all_five_views_equivalence": step0,
        "minibatch_schedule_shape": list(minibatches.shape),
        "minibatch_schedule_sha256": _tensor_sha(minibatches),
        "model_fit_original_count": len(model_fit),
        "model_fit_ordered_iids": [row.iid for row in model_fit],
        "model_fit_iid_digest": _object_sha([row.iid for row in model_fit]),
        "fixed_clip_pca_fit_input_sha256": fitted.fit_input_sha256,
        "model_fit_original_tensor_sha256": _tensor_sha(fit_original),
        "model_fit_all_five_views_tensor_sha256": _tensor_sha(fit_views),
        "model_fit_five_view_tensors_used_for_gradient_and_model_input": True,
        "model_fit_transform_roles_used_for_gradient_loss": True,
        "model_fit_transform_roles_used_for_model_input": False,
        "model_fit_family_metadata_used_for_gradient_or_model_input": False,
        "exact_three_role_directed_decoded_teacher_margins": True,
        "teacher_margin_scale_mean_all_ten_teacher_distances_plus_1e_minus_8": True,
        "inner_validation_original_count": len(inner_validation_iids),
        "inner_validation_ordered_iids": list(inner_validation_iids),
        "inner_validation_iid_digest": _object_sha(list(inner_validation_iids)),
        "inner_validation_five_view_tensor_count_used_during_training": 0,
        "inner_validation_any_view_used_during_gradient_or_checkpoint_selection": False,
        "oof_tensors_supplied_to_optimizer_checkpoint_or_inner_gate": False,
        "family_labels_entered_loss_or_model_input": False,
        "family_metadata_used_for_split_and_inner_final_gate_bootstrap": True,
        "family_metadata_used_for_gradient_model_input_or_loss": False,
    }
    return model, audit


@torch.no_grad()
def _model_decode_batches(
    model: VJepa2GlobalCodec, values: torch.Tensor, batch_size: int,
) -> torch.Tensor:
    model.eval()
    output = [
        model(values[start:start + batch_size])
        for start in range(0, len(values), batch_size)
    ]
    result = torch.cat(output)
    if result.shape != values.shape or not bool(torch.isfinite(result).all()):
        raise RuntimeError("decoded candidate output differs")
    return result


def _raw_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("raw reconstruction geometry differs")
    return (prediction - target).square().mean()


@torch.no_grad()
def _evaluate_rows_fixed(
    rows: Sequence[v4c.Record], model: VJepa2GlobalCodec,
    fitted: ClipPCAFit, config: Config, device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate the sole fixed candidate on an already materialized population."""

    if not rows or any(set(row.views) != set(EVAL_VIEWS) for row in rows):
        raise ValueError("fixed-candidate five-view population differs")
    stacked_cpu = {
        view: torch.stack([v4c.canonical_action(row.views[view]) for row in rows])
        for view in EVAL_VIEWS
    }
    stacked = {view: value.to(device) for view, value in stacked_cpu.items()}
    baseline = {
        view: frozen._analytic_clip_pca_decode(value, fitted)
        for view, value in stacked.items()
    }
    candidate = {
        view: _model_decode_batches(model, value, config.batch_size)
        for view, value in stacked.items()
    }
    output = []
    for index, row in enumerate(rows):
        teacher_margin = {
            negative: v4c._margin(
                stacked_cpu["original"][index].flatten(),
                stacked_cpu["monotone_warp"][index].flatten(),
                stacked_cpu[negative][index].flatten(),
            )["margin"]
            for negative in NEGATIVES
        }
        baseline_margin = {
            negative: v4c._margin(
                baseline["original"][index].flatten(),
                baseline["monotone_warp"][index].flatten(),
                baseline[negative][index].flatten(),
            )["margin"]
            for negative in NEGATIVES
        }
        candidate_margin = {
            negative: v4c._margin(
                candidate["original"][index].flatten(),
                candidate["monotone_warp"][index].flatten(),
                candidate[negative][index].flatten(),
            )["margin"]
            for negative in NEGATIVES
        }
        reconstruction = {
            view: {
                "candidate_raw_mse": float(_raw_mse(
                    candidate[view][index], stacked[view][index]
                ).cpu()),
                "clip_pca_b384_raw_mse": float(_raw_mse(
                    baseline[view][index], stacked[view][index]
                ).cpu()),
            }
            for view in EVAL_VIEWS
        }
        finite = [
            value for table in (teacher_margin, baseline_margin, candidate_margin)
            for value in table.values()
        ] + [value for table in reconstruction.values() for value in table.values()]
        if any(not math.isfinite(float(value)) for value in finite):
            raise RuntimeError("fixed-candidate evidence is non-finite")
        output.append({
            "iid": row.iid,
            "family": row.family,
            "teacher_margin_by_negative": teacher_margin,
            "clip_pca_b384_margin_by_negative": baseline_margin,
            "candidate_margin_by_negative": candidate_margin,
            "raw_reconstruction_by_view": reconstruction,
            "fixed_step": FIXED_SELECTED_STEP,
            "fixed_residual_scale": FIXED_RESIDUAL_SCALE,
            "single_fixed_candidate": True,
        })
    return output


def _positive_point_and_lcb_gate(statistics: Mapping[str, Any]) -> bool:
    return bool(
        statistics["clip_micro_point_mean"] > 0.0
        and statistics["family_macro_point_mean"] > 0.0
        and statistics["both_lcbs_strictly_gt_zero"]
    )


def _inner_candidate_gate(
    evidence: Sequence[Mapping[str, Any]], config: Config, *, fold_index: int,
) -> dict[str, Any]:
    """Apply the unchanged full v4-F gate to exactly one fixed candidate."""

    if (
        not evidence or len({str(row["iid"]) for row in evidence}) != len(evidence)
        or not 0 <= fold_index < OUTER_FOLDS
    ):
        raise ValueError("inner fixed-candidate population differs")
    families = [str(row["family"]) for row in evidence]
    prefix = f"inner:fold{fold_index}:fixed1200"
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
        statistics = frozen._paired_ratio_ucb(
            candidate_errors, baseline_errors, families, config,
            f"{prefix}:recon:{view}", namespace="v4g",
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
        teacher = frozen._paired_lcb(
            teacher_values, families, config,
            f"inner:fold{fold_index}:teacher-fixed:{negative}", namespace="v4g",
        )
        candidate = frozen._paired_lcb(
            candidate_values, families, config,
            f"{prefix}:candidate:{negative}", namespace="v4g",
        )
        retention = frozen._paired_lcb(
            retention_values, families, config,
            f"{prefix}:candidate-minus-0p8-teacher:{negative}", namespace="v4g",
        )
        improvement = frozen._paired_lcb(
            improvement_values, families, config,
            f"{prefix}:candidate-minus-clip-pca:{negative}", namespace="v4g",
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
        "complete_candidate_dependent_inner_gate": bool(
            fidelity_gate and negative_gate
        ),
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
def _evaluate_fixed_inner_candidate(
    inner_rows: Sequence[v4c.Record], model: VJepa2GlobalCodec,
    fitted: ClipPCAFit, config: Config, fold_index: int, device: torch.device,
) -> dict[str, Any]:
    state_before = _state_sha(_state_to_cpu(model))
    evidence = _evaluate_rows_fixed(inner_rows, model, fitted, config, device)
    gate = _inner_candidate_gate(evidence, config, fold_index=fold_index)
    state_after = _state_sha(_state_to_cpu(model))
    if state_before != state_after:
        raise RuntimeError("model state changed during exact-one inner evaluation")
    passed = bool(gate["complete_candidate_dependent_inner_gate"])
    return {
        "validation_scope": "fold_local_inner_exact_one_fixed_candidate",
        "outer_fold": fold_index,
        "candidate_count": FIXED_CANDIDATE_COUNT,
        "single_candidate": True,
        "candidate_name": "fixed1200_residual_scale1",
        "fixed_step": FIXED_SELECTED_STEP,
        "fixed_residual_scale": FIXED_RESIDUAL_SCALE,
        "hyperparameter_selection_performed": False,
        "inner_evidence_count": len(evidence),
        "inner_evidence_sha256": _object_sha(evidence),
        "inner_evidence": evidence,
        "gate": gate,
        "bootstrap_seed_ledger": _bootstrap_seed_ledger(gate),
        "pass": passed,
        "inner_pass": passed,
        "no_pass_action": "global INNER_NO_GO; all five folds OOF read exact0",
        "model_state_sha256_before_inner": state_before,
        "model_state_sha256_after_inner": state_after,
        "model_state_unchanged_during_inner_evaluation": True,
        "inner_five_view_tensors_used_for_gate_evaluation": True,
        "inner_five_view_tensors_used_for_gradient_or_model_input": False,
        "transform_roles_used_for_gate_evaluation": True,
        "transform_roles_used_for_gradient_loss_during_training": True,
        "transform_roles_used_for_model_input": False,
        "family_metadata_used_for_gate_and_bootstrap": True,
        "family_metadata_used_for_gradient_model_input_or_loss": False,
        "teacher_and_fixed_pca_metadata_used_for_gate_evaluation": True,
        "teacher_and_fixed_pca_metadata_used_for_gradient_or_model_input": False,
        "cross_fold_inner_metric_aggregation_or_selection": False,
    }


def _checkpoint_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_size, value.st_mode,
        value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
    )


def _load_checkpoint_sealed(
    path: Path, expected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    """Hash a mode-0444/nlink-1 checkpoint before safe Torch parsing."""

    required_expectations = {
        "outer_fold", "checkpoint_role", "model_state_sha256",
        "preselection_checkpoint_file_sha256",
        "preselection_checkpoint_binding",
        "preselection_checkpoint_binding_sha256", "metadata_digest",
        "implementation_sha256", "model_fit_original_count",
        "model_fit_ordered_iids", "model_fit_iid_digest",
        "inner_validation_iid_digest", "fixed_clip_pca_fit_input_sha256",
        "minibatch_schedule_sha256", "runtime_fingerprint",
    }
    if not required_expectations.issubset(expected):
        raise RuntimeError("checkpoint authoritative expectation is incomplete")
    if (
        not path.is_absolute() or path.is_symlink()
        or str(path) != str(path.resolve(strict=True))
    ):
        raise ValueError("checkpoint path must be absolute/canonical/non-symlink")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise RuntimeError("checkpoint pre-open seal differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        digest_before = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_before.update(chunk)
        file_sha = digest_before.hexdigest()
        physical = {
            "device": before.st_dev, "inode": before.st_ino,
            "size_bytes": before.st_size,
        }
        if (
            _checkpoint_stat_identity(opened) != _checkpoint_stat_identity(before)
            or expected.get("file_sha256") not in (None, file_sha)
            or expected.get("size_bytes") not in (None, before.st_size)
            or expected.get("physical_identity") not in (None, physical)
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
    after = path.lstat()
    if (
        len({_checkpoint_stat_identity(value) for value in (
            before, opened, loaded, closed, after
        )}) != 1
        or digest_after.hexdigest() != file_sha or before.st_size <= 0
    ):
        raise RuntimeError("checkpoint single-FD identity/SHA replay differs")
    if type(payload) is not dict or set(payload) != {"metadata", "state_dict"}:
        raise RuntimeError("checkpoint safe envelope differs")
    metadata, state = payload["metadata"], payload["state_dict"]
    if type(metadata) is not dict or type(state) is not dict or not state:
        raise RuntimeError("checkpoint metadata/state envelope differs")
    unsigned = dict(metadata)
    metadata_digest = unsigned.pop("metadata_digest", None)
    implementation = metadata.get("implementation")
    model_fit_iids = metadata.get("model_fit_ordered_iids")
    if (
        metadata.get("schema_version") != CHECKPOINT_SCHEMA
        or metadata_digest != _object_sha(unsigned)
        or metadata.get("outer_fold") != expected.get("outer_fold")
        or metadata.get("checkpoint_role") != expected.get("checkpoint_role")
        or metadata.get("fixed_step") != FIXED_SELECTED_STEP
        or metadata.get("fixed_residual_scale") != FIXED_RESIDUAL_SCALE
        or metadata.get("candidate_count") != FIXED_CANDIDATE_COUNT
        or metadata.get("single_candidate") is not True
        or metadata.get("hyperparameter_selection_performed") is not False
        or metadata.get("full_budget_steps_executed") != Config().max_steps
        or metadata.get("checkpoint_schedule") != list(Config().checkpoint_steps)
        or metadata.get("config_sha256") != _object_sha(_config_value(Config()))
        or metadata.get("model_state_sha256") != expected.get("model_state_sha256")
        or metadata.get("minibatch_schedule_sha256")
            != expected.get("minibatch_schedule_sha256")
        or metadata.get("runtime_fingerprint")
            != expected.get("runtime_fingerprint")
        or metadata.get("preselection_checkpoint_binding_sha256")
            != expected.get("preselection_checkpoint_binding_sha256")
        or metadata.get("preselection_checkpoint_binding")
            != expected.get("preselection_checkpoint_binding")
        or metadata_digest != expected.get("metadata_digest")
        or metadata.get("config") != _config_value(Config())
        or type(implementation) is not dict
        or implementation.get("implementation_sha256")
            != expected.get("implementation_sha256")
        or metadata.get("refit_artifact") is not False
        or metadata.get("inference_authorized") is not False
        or type(model_fit_iids) is not list or not model_fit_iids
        or metadata.get("model_fit_original_count")
            != expected.get("model_fit_original_count")
        or model_fit_iids != expected.get("model_fit_ordered_iids")
        or metadata.get("model_fit_iid_digest")
            != expected.get("model_fit_iid_digest")
        or metadata.get("inner_validation_iid_digest")
            != expected.get("inner_validation_iid_digest")
        or len(set(model_fit_iids)) != len(model_fit_iids)
        or len(model_fit_iids) != metadata.get("model_fit_original_count")
        or _object_sha(model_fit_iids) != metadata.get("model_fit_iid_digest")
    ):
        raise RuntimeError("checkpoint semantic metadata replay differs")
    if any(
        type(name) is not str or type(value) is not torch.Tensor
        or not bool(value.isfinite().all()) for name, value in state.items()
    ):
        raise RuntimeError("checkpoint tensor closure differs")
    state_sha = _state_sha(state)
    preselection_binding = metadata.get("preselection_checkpoint_binding")
    if (
        state_sha != metadata["model_state_sha256"]
        or metadata.get("preselection_checkpoint_binding_sha256") != (
            _object_sha(preselection_binding) if preselection_binding is not None
            else None
        )
        or (
            metadata["checkpoint_role"] == "preselection_fixed_step1200"
            and preselection_binding is not None
        )
        or (
            metadata["checkpoint_role"] == "fixed1200_candidate"
            and (
                type(preselection_binding) is not dict
                or preselection_binding.get("model_state_sha256") != state_sha
                or preselection_binding.get("file_sha256")
                    != expected.get("preselection_checkpoint_file_sha256")
            )
        )
    ):
        raise RuntimeError("checkpoint state/preselection join differs")
    required_buffers = {"clip_mean", "clip_basis", "fit_only_rms"}
    basis = metadata.get("basis")
    if (
        type(basis) is not dict or not required_buffers.issubset(state)
        or basis.get("clip_mean_sha256") != _tensor_sha(state["clip_mean"])
        or basis.get("clip_basis_sha256") != _tensor_sha(state["clip_basis"])
        or basis.get("fit_only_global_rms_sha256")
            != _tensor_sha(state["fit_only_rms"])
        or basis.get("fixed_clip_pca_fit_input_sha256")
            != expected.get("fixed_clip_pca_fit_input_sha256")
        or tuple(state["clip_mean"].shape) != (1, FULL_NUMEL)
        or tuple(state["clip_basis"].shape) != (FULL_NUMEL, CODE_NUMEL)
        or tuple(state["fit_only_rms"].shape) != (1,)
    ):
        raise RuntimeError("checkpoint basis metadata/state join differs")
    fitted = ClipPCAFit(
        clip_mean=state["clip_mean"], clip_basis=state["clip_basis"],
        fit_iid_digest=str(metadata["model_fit_iid_digest"]),
        fit_input_sha256=str(basis["fixed_clip_pca_fit_input_sha256"]),
        diagnostics={},
    )
    template = VJepa2GlobalCodec(fitted, state["fit_only_rms"])
    if set(template.state_dict()) != set(state):
        raise RuntimeError("checkpoint exact model schema differs")
    template.load_state_dict(state, strict=True)
    binding = {
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha,
        "size_bytes": before.st_size,
        "mode_octal": "0444",
        "nlink": before.st_nlink,
        "physical_identity": physical,
        "single_fd_pre_post_sha256_exact": True,
        "semantic_metadata_state_replay_verified": True,
        "checkpoint_role": metadata["checkpoint_role"],
        "outer_fold": metadata["outer_fold"],
        "metadata_digest": metadata_digest,
        "implementation_sha256": implementation["implementation_sha256"],
        "model_state_sha256": state_sha,
        "model_fit_original_count": metadata["model_fit_original_count"],
        "model_fit_ordered_iids": metadata["model_fit_ordered_iids"],
        "model_fit_iid_digest": metadata["model_fit_iid_digest"],
        "inner_validation_iid_digest": metadata[
            "inner_validation_iid_digest"
        ],
        "fixed_clip_pca_fit_input_sha256": basis[
            "fixed_clip_pca_fit_input_sha256"
        ],
        "minibatch_schedule_sha256": metadata["minibatch_schedule_sha256"],
        "runtime_fingerprint": metadata["runtime_fingerprint"],
        "model_schema_reconstructed_and_strict_loaded": True,
    }
    return metadata, state, binding


def _save_checkpoint_create_only(
    path: Path, model: VJepa2GlobalCodec, fitted: ClipPCAFit,
    training: Mapping[str, Any], config: Config, fold_index: int,
    run_binding: Mapping[str, str], probe_rows: Sequence[v4c.Record],
    device: torch.device, *, checkpoint_role: str,
    preselection_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if (
        not path.is_absolute() or not path.parent.is_dir()
        or path.exists() or path.is_symlink()
    ):
        raise ValueError("checkpoint must be a fresh absolute child")
    state = _state_to_cpu(model)
    state_sha = _state_sha(state)
    if (
        checkpoint_role not in {
            "preselection_fixed_step1200", "fixed1200_candidate",
        }
        or training.get("fixed_step") != FIXED_SELECTED_STEP
        or training.get("final_step_state_sha256") != state_sha
        or (
            checkpoint_role == "preselection_fixed_step1200"
            and preselection_artifact is not None
        )
        or (
            checkpoint_role == "fixed1200_candidate"
            and type(preselection_artifact) is not dict
        )
    ):
        raise RuntimeError("checkpoint does not join the sole trained state/role")
    preselection_binding = None
    if preselection_artifact is not None:
        if (
            preselection_artifact.get("checkpoint_role")
                != "preselection_fixed_step1200"
            or preselection_artifact.get("model_state_sha256") != state_sha
        ):
            raise RuntimeError("fixed checkpoint does not join preselection")
        preselection_binding = {
            key: preselection_artifact[key] for key in (
                "path", "file_sha256", "size_bytes", "mode_octal", "nlink",
                "metadata_digest", "model_state_sha256", "physical_identity",
            )
        }
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "outer_fold": fold_index,
        "checkpoint_role": checkpoint_role,
        "fixed_step": FIXED_SELECTED_STEP,
        "fixed_residual_scale": FIXED_RESIDUAL_SCALE,
        "candidate_count": FIXED_CANDIDATE_COUNT,
        "single_candidate": True,
        "hyperparameter_selection_performed": False,
        "full_budget_steps_executed": config.max_steps,
        "checkpoint_schedule": list(config.checkpoint_steps),
        "minibatch_schedule_sha256": training["minibatch_schedule_sha256"],
        "runtime_fingerprint": training["runtime_fingerprint"],
        "model_state_sha256": state_sha,
        "preselection_checkpoint_binding": preselection_binding,
        "preselection_checkpoint_binding_sha256": (
            _object_sha(preselection_binding)
            if preselection_binding is not None else None
        ),
        "config": _config_value(config),
        "config_sha256": _object_sha(_config_value(config)),
        "implementation": dict(run_binding),
        "fixed_comparator_name": BASELINE_NAME,
        "basis": {
            "clip_mean_sha256": _tensor_sha(fitted.clip_mean),
            "clip_basis_sha256": _tensor_sha(fitted.clip_basis),
            "fit_only_global_rms_sha256": training["fit_only_global_rms_sha256"],
            "fixed_clip_pca_fit_input_sha256": fitted.fit_input_sha256,
        },
        "model_fit_original_count": training["model_fit_original_count"],
        "model_fit_ordered_iids": training["model_fit_ordered_iids"],
        "model_fit_iid_digest": training["model_fit_iid_digest"],
        "inner_validation_iid_digest": training["inner_validation_iid_digest"],
        "artifact_scope": (
            "burned-development fixed step1200 scale1 fold codec; not refit "
            "and not authorized inference"
        ),
        "refit_artifact": False,
        "inference_authorized": False,
        "cross_environment_bit_exact_weights_claimed": False,
    }
    metadata["metadata_digest"] = _object_sha(metadata)
    with path.open("xb") as handle:
        torch.save({"metadata": metadata, "state_dict": state}, handle)
        handle.flush()
        os.fsync(handle.fileno())
        written = os.fstat(handle.fileno())
    os.chmod(path, 0o444)
    expectation = {
        "outer_fold": fold_index,
        "checkpoint_role": checkpoint_role,
        "model_state_sha256": state_sha,
        "preselection_checkpoint_file_sha256": (
            preselection_artifact.get("file_sha256")
            if preselection_artifact is not None else None
        ),
        "preselection_checkpoint_binding": preselection_binding,
        "preselection_checkpoint_binding_sha256": (
            _object_sha(preselection_binding)
            if preselection_binding is not None else None
        ),
        "metadata_digest": metadata["metadata_digest"],
        "implementation_sha256": run_binding["implementation_sha256"],
        "model_fit_original_count": training["model_fit_original_count"],
        "model_fit_ordered_iids": training["model_fit_ordered_iids"],
        "model_fit_iid_digest": training["model_fit_iid_digest"],
        "inner_validation_iid_digest": training[
            "inner_validation_iid_digest"
        ],
        "fixed_clip_pca_fit_input_sha256": training[
            "fixed_clip_pca_fit_input_sha256"
        ],
        "minibatch_schedule_sha256": training["minibatch_schedule_sha256"],
        "runtime_fingerprint": training["runtime_fingerprint"],
    }
    loaded_metadata, loaded_state, binding = _load_checkpoint_sealed(
        path, expectation
    )
    if (
        loaded_metadata != metadata or _state_sha(loaded_state) != state_sha
        or binding["physical_identity"] != {
            "device": written.st_dev, "inode": written.st_ino,
            "size_bytes": written.st_size,
        }
    ):
        raise RuntimeError("fresh checkpoint strong reload differs")
    reloaded = VJepa2GlobalCodec(fitted, state["fit_only_rms"])
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.to(device).eval()
    if not probe_rows:
        raise RuntimeError("checkpoint reload probe population is empty")
    probe = torch.stack([
        v4c.canonical_action(row.views["original"])
        for row in probe_rows[:min(config.batch_size, len(probe_rows))]
    ]).to(device)
    with torch.no_grad():
        expected_output = model(probe)
        actual_output = reloaded(probe)
    if not torch.equal(expected_output, actual_output):
        raise RuntimeError("fresh checkpoint strict reload output differs")
    model.load_state_dict(loaded_state, strict=True)
    model.to(device).eval()
    return {
        **binding,
        "outer_fold": fold_index,
        "checkpoint_role": checkpoint_role,
        "fixed_step": FIXED_SELECTED_STEP,
        "fixed_residual_scale": FIXED_RESIDUAL_SCALE,
        "model_state_sha256": state_sha,
        "preselection_checkpoint_file_sha256": (
            preselection_artifact.get("file_sha256")
            if preselection_artifact is not None else None
        ),
        "preselection_checkpoint_binding": preselection_binding,
        "preselection_checkpoint_binding_sha256": (
            _object_sha(preselection_binding)
            if preselection_binding is not None else None
        ),
        "implementation_sha256": run_binding["implementation_sha256"],
        "metadata_digest": metadata["metadata_digest"],
        "model_fit_original_count": training["model_fit_original_count"],
        "model_fit_ordered_iids": training["model_fit_ordered_iids"],
        "model_fit_iid_digest": training["model_fit_iid_digest"],
        "inner_validation_iid_digest": training[
            "inner_validation_iid_digest"
        ],
        "fixed_clip_pca_fit_input_sha256": training[
            "fixed_clip_pca_fit_input_sha256"
        ],
        "minibatch_schedule_sha256": training["minibatch_schedule_sha256"],
        "runtime_fingerprint": training["runtime_fingerprint"],
        "fresh_reload_strict_state_verified": True,
        "fresh_reload_output_bit_exact": True,
        "caller_model_reloaded_from_sealed_artifact_before_next_stage": True,
    }


def _verify_distinct_checkpoint_pair(
    preselection: Mapping[str, Any], fixed: Mapping[str, Any],
) -> dict[str, Any]:
    pre_physical = preselection.get("physical_identity")
    fixed_physical = fixed.get("physical_identity")
    provenance_fields = (
        "outer_fold", "model_state_sha256", "model_fit_original_count",
        "model_fit_ordered_iids", "model_fit_iid_digest",
        "inner_validation_iid_digest", "fixed_clip_pca_fit_input_sha256",
        "minibatch_schedule_sha256", "runtime_fingerprint",
        "implementation_sha256",
    )
    if (
        type(pre_physical) is not dict or type(fixed_physical) is not dict
        or preselection.get("checkpoint_role") != "preselection_fixed_step1200"
        or fixed.get("checkpoint_role") != "fixed1200_candidate"
        or preselection.get("path") == fixed.get("path")
        or (pre_physical.get("device"), pre_physical.get("inode"))
            == (fixed_physical.get("device"), fixed_physical.get("inode"))
        or preselection.get("model_state_sha256") != fixed.get("model_state_sha256")
        or fixed.get("preselection_checkpoint_file_sha256")
            != preselection.get("file_sha256")
        or fixed.get("preselection_checkpoint_binding") != {
            key: preselection[key] for key in (
                "path", "file_sha256", "size_bytes", "mode_octal", "nlink",
                "metadata_digest", "model_state_sha256", "physical_identity",
            )
        }
        or fixed.get("preselection_checkpoint_binding_sha256")
            != _object_sha(fixed.get("preselection_checkpoint_binding"))
        or any(preselection.get(key) != fixed.get(key) for key in provenance_fields)
        or any(
            artifact.get("semantic_metadata_state_replay_verified") is not True
            or artifact.get("fresh_reload_strict_state_verified") is not True
            or artifact.get("fresh_reload_output_bit_exact") is not True
            or artifact.get(
                "caller_model_reloaded_from_sealed_artifact_before_next_stage"
            ) is not True
            for artifact in (preselection, fixed)
        )
    ):
        raise RuntimeError("preselection/fixed1200 checkpoint pair differs")
    return {
        "preselection_path": preselection["path"],
        "fixed1200_path": fixed["path"],
        "preselection_device_inode": [
            pre_physical["device"], pre_physical["inode"],
        ],
        "fixed1200_device_inode": [
            fixed_physical["device"], fixed_physical["inode"],
        ],
        "distinct_device_inode_pair": True,
        "same_model_state_sha256": True,
        "model_state_sha256": fixed["model_state_sha256"],
        "both_checkpoint_files_strongly_and_strictly_reloaded": True,
    }


def _write_json_create_only(path: Path, value: Any) -> str:
    return v4c._write_json_create_only(path, value)["sha256"]


def _resolve_fold_root(
    value: str, *, fresh_train: bool = False,
) -> tuple[Path, Path, Path, Path]:
    root = Path(value)
    if (
        not root.is_absolute() or root.is_symlink() or not root.is_dir()
        or str(root) != str(root.resolve(strict=True))
    ):
        raise ValueError("fold root must be an existing absolute canonical directory")
    preselection = root / "preselection.pt"
    fixed = root / "fixed1200.pt"
    inner = root / "inner.json"
    fold = root / "fold.json"
    if fresh_train and any(
        path.exists() or path.is_symlink()
        for path in (preselection, fixed, inner, fold)
    ):
        raise ValueError("train-fold outputs must all be fresh")
    return root, preselection, fixed, inner


def _verify_fixed_candidate_ledger(
    receipt: Mapping[str, Any], config: Config,
) -> None:
    candidate = receipt.get("fixed_candidate")
    evidence = candidate.get("inner_evidence") if type(candidate) is dict else None
    gate = candidate.get("gate") if type(candidate) is dict else None
    if (
        type(candidate) is not dict or type(evidence) is not list or not evidence
        or candidate.get("candidate_count") != FIXED_CANDIDATE_COUNT
        or candidate.get("single_candidate") is not True
        or candidate.get("fixed_step") != FIXED_SELECTED_STEP
        or candidate.get("fixed_residual_scale") != FIXED_RESIDUAL_SCALE
        or candidate.get("hyperparameter_selection_performed") is not False
        or candidate.get("inner_evidence_count") != len(evidence)
        or candidate.get("inner_evidence_sha256") != _object_sha(evidence)
        or [row.get("iid") for row in evidence]
            != receipt.get("inner_validation_ordered_iids")
        or any(
            row.get("fixed_step") != FIXED_SELECTED_STEP
            or row.get("fixed_residual_scale") != FIXED_RESIDUAL_SCALE
            or row.get("single_fixed_candidate") is not True
            for row in evidence
        )
    ):
        raise RuntimeError("fixed exact-one candidate evidence ledger differs")
    recomputed_gate = _inner_candidate_gate(
        evidence, config, fold_index=int(receipt["fold_index"])
    )
    passed = bool(recomputed_gate["complete_candidate_dependent_inner_gate"])
    if (
        gate != recomputed_gate
        or candidate.get("bootstrap_seed_ledger")
            != _bootstrap_seed_ledger(recomputed_gate)
        or candidate.get("pass") is not passed
        or candidate.get("inner_pass") is not passed
        or receipt.get("inner_pass") is not passed
        or candidate.get("model_state_sha256_before_inner")
            != candidate.get("model_state_sha256_after_inner")
        or candidate.get("model_state_sha256_before_inner")
            != receipt.get("fixed1200_checkpoint_artifact", {}).get(
                "model_state_sha256"
            )
    ):
        raise RuntimeError("fixed exact-one candidate gate/bootstrap replay differs")


def _checkpoint_expectations_from_inner_receipt(
    receipt: Mapping[str, Any],
    *, expected_runtime_fingerprint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build strong checkpoint expectations from receipt-level provenance.

    The artifact dictionaries are transport receipts, never the source of fold
    identity or fit-population truth.  Those values must join the fold receipt,
    its training ledger, and (during independent replay) frozen authority.
    """

    fold_index = receipt.get("fold_index")
    training = receipt.get("training")
    preselection = receipt.get("preselection_checkpoint_artifact")
    fixed = receipt.get("fixed1200_checkpoint_artifact")
    candidate = receipt.get("fixed_candidate")
    fit_iids = receipt.get("model_fit_ordered_iids")
    inner_iids = receipt.get("inner_validation_ordered_iids")
    receipt_runtime = receipt.get("runtime_fingerprint")
    if (
        type(fold_index) is not int or not 0 <= fold_index < OUTER_FOLDS
        or type(training) is not dict
        or type(preselection) is not dict or type(fixed) is not dict
        or type(candidate) is not dict
        or type(fit_iids) is not list or not fit_iids
        or type(inner_iids) is not list or not inner_iids
        or len(set(fit_iids)) != len(fit_iids)
        or len(set(inner_iids)) != len(inner_iids)
        or type(receipt_runtime) is not dict
    ):
        raise RuntimeError("inner receipt checkpoint provenance envelope differs")
    fit_digest = _object_sha(fit_iids)
    inner_digest = _object_sha(inner_iids)
    pca_input_sha = receipt.get("fixed_clip_pca_b384_fit_input_sha256")
    state_sha = training.get("final_step_state_sha256")
    schedule_sha = training.get("minibatch_schedule_sha256")
    common_expected = {
        "outer_fold": fold_index,
        "model_state_sha256": state_sha,
        "implementation_sha256": receipt.get("implementation", {}).get(
            "implementation_sha256"
        ),
        "model_fit_original_count": len(fit_iids),
        "model_fit_ordered_iids": fit_iids,
        "model_fit_iid_digest": fit_digest,
        "inner_validation_iid_digest": inner_digest,
        "fixed_clip_pca_fit_input_sha256": pca_input_sha,
        "minibatch_schedule_sha256": schedule_sha,
        "runtime_fingerprint": receipt_runtime,
    }
    provenance_fields = tuple(common_expected)
    if (
        receipt.get("model_fit_original_count") != len(fit_iids)
        or receipt.get("model_fit_iid_digest") != fit_digest
        or receipt.get("inner_validation_original_count") != len(inner_iids)
        or receipt.get("inner_validation_iid_digest") != inner_digest
        or receipt.get("fixed_clip_pca_b384_fit_iid_digest") != fit_digest
        or training.get("model_fit_original_count") != len(fit_iids)
        or training.get("model_fit_ordered_iids") != fit_iids
        or training.get("model_fit_iid_digest") != fit_digest
        or training.get("inner_validation_original_count") != len(inner_iids)
        or training.get("inner_validation_ordered_iids") != inner_iids
        or training.get("inner_validation_iid_digest") != inner_digest
        or training.get("fixed_clip_pca_fit_input_sha256") != pca_input_sha
        or training.get("runtime_fingerprint") != receipt_runtime
        or training.get("fixed_step") != FIXED_SELECTED_STEP
        or training.get("full_budget_steps_executed") != Config().max_steps
        or training.get("minibatch_schedule_shape")
            != [Config().max_steps, Config().batch_size]
        or training.get("selected_state_sha256") != state_sha
        or candidate.get("model_state_sha256_before_inner") != state_sha
        or candidate.get("model_state_sha256_after_inner") != state_sha
        or any(artifact.get(key) != value for key, value in common_expected.items()
               for artifact in (preselection, fixed))
        or preselection.get("checkpoint_role")
            != "preselection_fixed_step1200"
        or fixed.get("checkpoint_role") != "fixed1200_candidate"
        or fixed.get("preselection_checkpoint_file_sha256")
            != preselection.get("file_sha256")
        or fixed.get("preselection_checkpoint_binding") != {
            key: preselection.get(key) for key in (
                "path", "file_sha256", "size_bytes", "mode_octal", "nlink",
                "metadata_digest", "model_state_sha256", "physical_identity",
            )
        }
        or fixed.get("preselection_checkpoint_binding_sha256")
            != _object_sha(fixed.get("preselection_checkpoint_binding"))
        or preselection.get("preselection_checkpoint_binding") is not None
        or preselection.get("preselection_checkpoint_file_sha256") is not None
        or preselection.get("preselection_checkpoint_binding_sha256") is not None
        or any(artifact.get(key) is None for artifact in (preselection, fixed)
               for key in provenance_fields)
    ):
        raise RuntimeError("inner receipt training/checkpoint provenance join differs")
    if (
        expected_runtime_fingerprint is not None
        and receipt_runtime != dict(expected_runtime_fingerprint)
    ):
        raise RuntimeError(
            "checkpoint training/evaluate runtime and device class differ"
        )

    preselection_expected = {
        **preselection,
        **common_expected,
        "checkpoint_role": "preselection_fixed_step1200",
        "preselection_checkpoint_file_sha256": None,
        "preselection_checkpoint_binding": None,
        "preselection_checkpoint_binding_sha256": None,
    }
    fixed_expected = {
        **fixed,
        **common_expected,
        "checkpoint_role": "fixed1200_candidate",
        "preselection_checkpoint_file_sha256": preselection["file_sha256"],
        "preselection_checkpoint_binding": fixed.get(
            "preselection_checkpoint_binding"
        ),
        "preselection_checkpoint_binding_sha256": fixed.get(
            "preselection_checkpoint_binding_sha256"
        ),
    }
    return preselection_expected, fixed_expected


def _load_inner_receipt_sealed(
    fold_root: str, expected_sha256: str,
    expected_binding: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root, preselection_path, fixed_path, inner_path = _resolve_fold_root(fold_root)
    file_sha = _file_sha(inner_path)
    if file_sha != expected_sha256:
        raise RuntimeError("inner receipt expected SHA differs")
    value = v4c._load_json_sealed(inner_path, file_sha)
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    fold_index = value.get("fold_index")
    fixed_candidate = value.get("fixed_candidate")
    preselection = value.get("preselection_checkpoint_artifact")
    fixed = value.get("fixed1200_checkpoint_artifact")
    if (
        value.get("schema_version") != INNER_SCHEMA
        or digest != _object_sha(unsigned)
        or type(fold_index) is not int or not 0 <= fold_index < OUTER_FOLDS
        or value.get("implementation") != expected_binding
        or value.get("config") != _config_value(Config())
        or value.get("config_sha256") != _object_sha(_config_value(Config()))
        or value.get("candidate_count") != FIXED_CANDIDATE_COUNT
        or value.get("single_candidate") is not True
        or value.get("hyperparameter_selection_performed") is not False
        or value.get("oof_semantic_tensor_materialized_count") != 0
        or value.get("oof_semantic_tensor_read_count_exact0") is not True
        or type(fixed_candidate) is not dict
        or fixed_candidate.get("candidate_count") != FIXED_CANDIDATE_COUNT
        or fixed_candidate.get("single_candidate") is not True
        or fixed_candidate.get("hyperparameter_selection_performed") is not False
        or type(value.get("inner_pass")) is not bool
        or fixed_candidate.get("inner_pass") is not value.get("inner_pass")
        or value.get("status") != (
            INNER_PASS_STATUS if value.get("inner_pass") is True
            else INNER_NO_GO_STATUS
        )
        or value.get("qualification_scope") != {
            **_qualification_scope(None),
            "inner_fold_local_gate_passed": value.get("inner_pass"),
            "aggregate_gate_evaluated": False,
        }
        or type(preselection) is not dict or type(fixed) is not dict
        or preselection.get("path") != str(preselection_path.resolve(strict=True))
        or fixed.get("path") != str(fixed_path.resolve(strict=True))
    ):
        raise RuntimeError("sealed inner receipt semantic replay differs")
    _verify_fixed_candidate_ledger(value, Config())
    preselection_expected, fixed_expected = (
        _checkpoint_expectations_from_inner_receipt(value)
    )
    preselection_metadata, _, preselection_binding = _load_checkpoint_sealed(
        preselection_path, preselection_expected
    )
    actual_preselection_subset = {
        key: preselection_binding[key] for key in (
            "path", "file_sha256", "size_bytes", "mode_octal", "nlink",
            "metadata_digest", "model_state_sha256", "physical_identity",
        )
    }
    if (
        fixed_expected.get("preselection_checkpoint_binding")
            != actual_preselection_subset
    ):
        raise RuntimeError(
            "fixed checkpoint embedded preselection full binding differs"
        )
    fixed_metadata, _, fixed_binding = _load_checkpoint_sealed(
        fixed_path, fixed_expected
    )
    pair = _verify_distinct_checkpoint_pair(preselection, fixed)
    if pair != value.get("preselection_fixed1200_checkpoint_pair_join"):
        raise RuntimeError("inner receipt checkpoint pair replay differs")
    return value, {
        "fold_root": str(root),
        "path": str(inner_path.resolve(strict=True)),
        "file_sha256": file_sha,
        "receipt_digest": digest,
        "mode_octal": "0444",
        "nlink": inner_path.lstat().st_nlink,
        "preselection_checkpoint_metadata_digest": preselection_metadata[
            "metadata_digest"
        ],
        "fixed1200_checkpoint_metadata_digest": fixed_metadata[
            "metadata_digest"
        ],
        "checkpoint_provenance_binding_sha256": _object_sha({
            "preselection": preselection_binding,
            "fixed1200": fixed_binding,
        }),
    }


def _load_all_inner_receipts_or_fail_before_oof(
    fold_roots: Sequence[str], expected_sha256: Sequence[str],
    run_binding: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """The global barrier.  This function has no feature-tensor loader call."""

    if (
        len(fold_roots) != OUTER_FOLDS
        or len(expected_sha256) != OUTER_FOLDS
        or len(set(fold_roots)) != OUTER_FOLDS
        or len(set(expected_sha256)) != OUTER_FOLDS
    ):
        raise ValueError("global inner barrier requires exactly five fold roots")
    loaded = [
        _load_inner_receipt_sealed(root, expected, run_binding)
        for root, expected in zip(fold_roots, expected_sha256)
    ]
    loaded.sort(key=lambda item: item[0]["fold_index"])
    receipts = [item[0] for item in loaded]
    bindings = [item[1] for item in loaded]
    if [row["fold_index"] for row in receipts] != list(range(OUTER_FOLDS)):
        raise RuntimeError("global inner barrier lacks exact folds 0..4")
    if any(row.get("inner_pass") is not True for row in receipts):
        raise RuntimeError(
            "GLOBAL_INNER_NO_GO: at least one fixed candidate failed; all-fold "
            "OOF tensor reads must remain exact0"
        )
    barrier_value = [{
        "fold_index": row["fold_index"],
        "inner_receipt_file_sha256": binding["file_sha256"],
        "inner_receipt_digest": binding["receipt_digest"],
        "fixed_checkpoint_file_sha256": row[
            "fixed1200_checkpoint_artifact"
        ]["file_sha256"],
        "inner_pass": row["inner_pass"],
    } for row, binding in zip(receipts, bindings)]
    return receipts, bindings, _object_sha(barrier_value)


def _verify_inner_receipt_against_authority(
    receipt: Mapping[str, Any], authority: Mapping[str, Any], config: Config,
) -> None:
    """Replay one split and IID/family population without reading tensors."""

    _checkpoint_expectations_from_inner_receipt(receipt)
    fold_index = int(receipt["fold_index"])
    groups, split = frozen._split_fold(
        authority["ordered_records"], authority["outer_assignment"],
        fold_index, config,
    )
    fit_iids = [row.iid for row in groups["model_fit"]]
    inner_iids = [row.iid for row in groups["inner_validation"]]
    oof_iids = [row.iid for row in groups["exploratory_oof"]]
    inner_iid_family = [
        {"iid": row.iid, "family": row.family}
        for row in groups["inner_validation"]
    ]
    evidence = receipt.get("fixed_candidate", {}).get("inner_evidence")
    if (
        receipt.get("inner_split") != split
        or receipt.get("model_fit_ordered_iids") != fit_iids
        or receipt.get("model_fit_original_count") != len(fit_iids)
        or receipt.get("inner_validation_ordered_iids") != inner_iids
        or receipt.get("inner_validation_original_count") != len(inner_iids)
        or receipt.get("oof_ordered_iids") != oof_iids
        or receipt.get("model_fit_iid_digest") != _object_sha(fit_iids)
        or receipt.get("inner_validation_iid_digest") != _object_sha(inner_iids)
        or receipt.get("oof_iid_digest") != _object_sha(oof_iids)
        or receipt.get("fixed_clip_pca_b384_fit_iid_digest")
            != _object_sha(fit_iids)
        or type(evidence) is not list
        or [
            {"iid": row.get("iid"), "family": row.get("family")}
            for row in evidence
        ] != inner_iid_family
    ):
        raise RuntimeError("inner receipt differs from frozen split authority")


def _recompute_model_fit_provenance_from_authority(
    authority: Mapping[str, Any], groups: Mapping[str, Sequence[v4c.Record]],
    receipt: Mapping[str, Any], config: Config, fold_index: int,
    device: torch.device,
) -> tuple[ClipPCAFit, torch.Tensor, dict[str, Any]]:
    """Recompute every non-optimizer model-fit fact from frozen tensors.

    This deliberately does not repeat the 1,200 optimizer updates.  It reads the
    authority model-fit five views, recomputes the original-only PCA/RMS, and
    regenerates the preregistered CPU minibatch schedule.
    """

    model_fit = list(groups["model_fit"])
    fit_iids = [row.iid for row in model_fit]
    oof_iids = {row.iid for row in groups["exploratory_oof"]}
    if not model_fit or set(fit_iids) & oof_iids:
        raise RuntimeError("authority model-fit/OOF provenance population differs")
    model_fit_map, materialization = frozen._selective_materialize_feature_rows(
        authority["feature_index"], {iid: EVAL_VIEWS for iid in fit_iids},
        stage="stage1_model_fit_only",
    )
    if (
        materialization["semantic_tensor_materialized_count"]
            != len(fit_iids) * len(EVAL_VIEWS)
        or any(iid in oof_iids for iid in model_fit_map)
        or [iid for iid in fit_iids if iid not in model_fit_map]
        or any(
            materialization["semantic_tensor_materialized_count_by_view"][view]
                != len(fit_iids) for view in EVAL_VIEWS
        )
    ):
        raise RuntimeError("authority model-fit selective materialization differs")
    rows = [model_fit_map[iid] for iid in fit_iids]
    all_five = torch.stack([
        torch.stack([v4c.canonical_action(row.views[view]) for view in EVAL_VIEWS])
        for row in rows
    ])
    originals = all_five[:, EVAL_VIEWS.index("original")]
    fitted = frozen._fit_clip_pca_b384(rows)
    rms = frozen._fit_only_global_rms(rows, device)
    fold_seed = config.seed + 10000 + fold_index
    generator = torch.Generator(device="cpu").manual_seed(fold_seed + 1)
    minibatches = torch.randint(
        len(rows), (config.max_steps, config.batch_size), generator=generator,
    )
    training = receipt["training"]
    fixed_artifact = receipt["fixed1200_checkpoint_artifact"]
    if (
        [row.iid for row in rows] != receipt["model_fit_ordered_iids"]
        or fitted.fit_iid_digest != _object_sha(fit_iids)
        or fitted.fit_input_sha256
            != receipt["fixed_clip_pca_b384_fit_input_sha256"]
        or fitted.diagnostics != receipt["fixed_clip_pca_b384_diagnostics"]
        or training.get("fold_seed") != fold_seed
        or training.get("model_fit_original_tensor_sha256")
            != _tensor_sha(originals)
        or training.get("model_fit_all_five_views_tensor_sha256")
            != _tensor_sha(all_five)
        or training.get("fit_only_global_rms_sha256") != _tensor_sha(rms)
        or training.get("fit_only_global_rms") != float(rms.detach().cpu())
        or training.get("minibatch_schedule_sha256") != _tensor_sha(minibatches)
        or fixed_artifact.get("minibatch_schedule_sha256")
            != _tensor_sha(minibatches)
    ):
        raise RuntimeError("authority model-fit PCA/RMS/schedule receipt join differs")
    ledger = {
        "fold_index": fold_index,
        "model_fit_original_count": len(rows),
        "model_fit_ordered_iids": fit_iids,
        "model_fit_iid_digest": _object_sha(fit_iids),
        "model_fit_original_tensor_sha256": _tensor_sha(originals),
        "model_fit_all_five_views_tensor_sha256": _tensor_sha(all_five),
        "clip_pca_fit_input_sha256": fitted.fit_input_sha256,
        "clip_pca_fit_iid_digest": fitted.fit_iid_digest,
        "clip_pca_clip_mean_sha256": _tensor_sha(fitted.clip_mean),
        "clip_pca_clip_basis_sha256": _tensor_sha(fitted.clip_basis),
        "clip_pca_diagnostics_sha256": _object_sha(fitted.diagnostics),
        "fit_only_global_rms": float(rms.detach().cpu()),
        "fit_only_global_rms_sha256": _tensor_sha(rms),
        "fold_seed": fold_seed,
        "minibatch_generator_seed": fold_seed + 1,
        "minibatch_schedule_shape": list(minibatches.shape),
        "minibatch_schedule_sha256": _tensor_sha(minibatches),
        "authority_model_fit_all_five_views_re_materialized": True,
        "pca_and_rms_recomputed_from_original_only": True,
        "minibatch_schedule_regenerated_without_training": True,
        "optimizer_steps_reexecuted": 0,
        "oof_semantic_tensor_read_count": 0,
        "materialization_audit_sha256": _object_sha(materialization),
    }
    return fitted, rms, ledger


def _independently_replay_all_inner_gates_before_oof(
    receipts: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]], authority: Mapping[str, Any],
    config: Config, device: torch.device,
) -> tuple[list[dict[str, Any]], str]:
    """Re-forward all five fixed checkpoints on authority inner tensors.

    Receipt SHA values are transport bindings, not scientific evidence.  This
    independent replay loads every fixed checkpoint, materializes only that
    fold's authority-derived inner five views, recomputes candidate evidence,
    the complete gate, and every bootstrap seed/result, then exact-compares the
    full candidate ledger.  There is intentionally no OOF loader call here.
    """

    if (
        len(receipts) != OUTER_FOLDS or len(bindings) != OUTER_FOLDS
        or [row.get("fold_index") for row in receipts]
            != list(range(OUTER_FOLDS))
    ):
        raise RuntimeError("independent global inner replay population differs")
    replay_ledger: list[dict[str, Any]] = []
    replay_runtime = _runtime_fingerprint(device)
    for fold_index, (receipt, receipt_binding) in enumerate(
        zip(receipts, bindings)
    ):
        _verify_inner_receipt_against_authority(receipt, authority, config)
        groups, split = frozen._split_fold(
            authority["ordered_records"], authority["outer_assignment"],
            fold_index, config,
        )
        authoritative_fitted, authoritative_rms, model_fit_replay = (
            _recompute_model_fit_provenance_from_authority(
                authority, groups, receipt, config, fold_index, device,
            )
        )
        root = Path(str(receipt_binding["fold_root"]))
        checkpoint_artifact = receipt["fixed1200_checkpoint_artifact"]
        _, checkpoint_expected = _checkpoint_expectations_from_inner_receipt(
            receipt, expected_runtime_fingerprint=replay_runtime,
        )
        metadata, state, checkpoint_binding = _load_checkpoint_sealed(
            root / "fixed1200.pt", checkpoint_expected
        )
        training = receipt["training"]
        pair = _verify_distinct_checkpoint_pair(
            receipt["preselection_checkpoint_artifact"], checkpoint_artifact,
        )
        if (
            pair != receipt["preselection_fixed1200_checkpoint_pair_join"]
            or not torch.equal(state["clip_mean"], authoritative_fitted.clip_mean)
            or not torch.equal(state["clip_basis"], authoritative_fitted.clip_basis)
            or not torch.equal(
                state["fit_only_rms"].to(device), authoritative_rms
            )
            or metadata["basis"]["clip_mean_sha256"]
                != _tensor_sha(authoritative_fitted.clip_mean)
            or metadata["basis"]["clip_basis_sha256"]
                != _tensor_sha(authoritative_fitted.clip_basis)
            or metadata["basis"]["fit_only_global_rms_sha256"]
                != _tensor_sha(authoritative_rms)
            or metadata["basis"]["fixed_clip_pca_fit_input_sha256"]
                != authoritative_fitted.fit_input_sha256
            or receipt["fixed_clip_pca_b384_diagnostics"]
                != authoritative_fitted.diagnostics
            or training.get("trainable_parameter_count")
                != EXACT_TRAINABLE_PARAMETERS
            or training.get("full_budget_steps_executed") != config.max_steps
            or training.get("early_stopped") is not False
            or training.get("checkpoint_winner_selection_performed") is not False
            or training.get("hyperparameter_selection_performed") is not False
            or metadata["minibatch_schedule_sha256"]
                != model_fit_replay["minibatch_schedule_sha256"]
        ):
            raise RuntimeError(
                "authority PCA/RMS/schedule/checkpoint-state replay differs"
            )
        fitted = authoritative_fitted
        model = VJepa2GlobalCodec(fitted, authoritative_rms)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        inner_iids = [row.iid for row in groups["inner_validation"]]
        oof_iids = {row.iid for row in groups["exploratory_oof"]}
        if (
            split != receipt["inner_split"]
            or set(inner_iids) & oof_iids
            or inner_iids != receipt["inner_validation_ordered_iids"]
        ):
            raise RuntimeError("independent inner replay split differs")
        inner_map, materialization = frozen._selective_materialize_feature_rows(
            authority["feature_index"],
            {iid: EVAL_VIEWS for iid in inner_iids},
            stage="stage2_post_preselection_seal_inner_five_views",
        )
        if (
            materialization["semantic_tensor_materialized_count"]
                != len(inner_iids) * len(EVAL_VIEWS)
            or any(iid in oof_iids for iid in inner_map)
        ):
            raise RuntimeError("independent replay read a non-inner tensor")
        replay = _evaluate_fixed_inner_candidate(
            [inner_map[iid] for iid in inner_iids], model, fitted, config,
            fold_index, device,
        )
        if (
            replay != receipt.get("fixed_candidate")
            or replay.get("inner_pass") is not True
            or replay.get("pass") is not True
            or replay.get("model_state_sha256_before_inner")
                != checkpoint_artifact.get("model_state_sha256")
        ):
            raise RuntimeError(
                "independent checkpoint-forward inner evidence/gate replay differs"
            )
        inner_replay_binding = {
            "fold_index": fold_index,
            "fixed_candidate_ledger_sha256": _object_sha(replay),
            "inner_iid_digest": _object_sha(inner_iids),
            "inner_evidence_sha256": replay["inner_evidence_sha256"],
            "complete_gate_sha256": _object_sha(replay["gate"]),
            "bootstrap_seed_ledger_sha256": _object_sha(
                replay["bootstrap_seed_ledger"]
            ),
            "fixed1200_checkpoint_file_sha256": checkpoint_binding[
                "file_sha256"
            ],
            "fixed1200_model_state_sha256": checkpoint_binding[
                "model_state_sha256"
            ],
        }
        replay_ledger.append({
            "fold_index": fold_index,
            "inner_receipt_file_sha256": receipt_binding["file_sha256"],
            "fixed1200_checkpoint_file_sha256": checkpoint_binding[
                "file_sha256"
            ],
            "fixed1200_model_state_sha256": checkpoint_binding[
                "model_state_sha256"
            ],
            "checkpoint_outer_fold": metadata["outer_fold"],
            "checkpoint_outer_fold_authority_join": (
                metadata["outer_fold"] == fold_index
            ),
            "checkpoint_model_fit_ordered_iids_authority_join": (
                metadata["model_fit_ordered_iids"]
                == [row.iid for row in groups["model_fit"]]
            ),
            "checkpoint_model_fit_count_and_digest_authority_join": (
                metadata["model_fit_original_count"] == len(groups["model_fit"])
                and metadata["model_fit_iid_digest"]
                == _object_sha([row.iid for row in groups["model_fit"]])
            ),
            "checkpoint_inner_iid_digest_authority_join": (
                metadata["inner_validation_iid_digest"]
                == _object_sha(inner_iids)
            ),
            "checkpoint_pca_fit_input_receipt_training_join": (
                metadata["basis"]["fixed_clip_pca_fit_input_sha256"]
                == receipt["fixed_clip_pca_b384_fit_input_sha256"]
                == receipt["training"]["fixed_clip_pca_fit_input_sha256"]
            ),
            "checkpoint_minibatch_schedule_receipt_training_join": (
                metadata["minibatch_schedule_sha256"]
                == receipt["training"]["minibatch_schedule_sha256"]
            ),
            "checkpoint_state_receipt_training_inner_join": (
                metadata["model_state_sha256"]
                == receipt["training"]["final_step_state_sha256"]
                == replay["model_state_sha256_before_inner"]
                == replay["model_state_sha256_after_inner"]
            ),
            "training_evaluate_runtime_fingerprint_exact_match": (
                metadata["runtime_fingerprint"] == replay_runtime
            ),
            "runtime_fingerprint": replay_runtime,
            "model_fit_provenance_replay": model_fit_replay,
            "model_fit_provenance_replay_sha256": _object_sha(
                model_fit_replay
            ),
            "inner_replay_binding": inner_replay_binding,
            "inner_replay_sha256": _object_sha(inner_replay_binding),
            "checkpoint_clip_mean_equals_authority_recomputed_pca": True,
            "checkpoint_clip_basis_equals_authority_recomputed_pca": True,
            "checkpoint_fit_only_rms_equals_authority_recomputed_rms": True,
            "checkpoint_schema_and_exact79040_strict_loaded": True,
            "preselection_fixed1200_full_binding_reverified": True,
            "preselection_fixed1200_pair_join_sha256": _object_sha(pair),
            "full1200_optimizer_trajectory_reexecuted": False,
            "full1200_causal_weights_trusted_only_to_controller_pinned_sealed_training_execution": True,
            "duplicate_training_performed": False,
            "inner_iid_digest": _object_sha(inner_iids),
            "inner_evidence_sha256": replay["inner_evidence_sha256"],
            "complete_gate_sha256": _object_sha(replay["gate"]),
            "bootstrap_seed_ledger_sha256": _object_sha(
                replay["bootstrap_seed_ledger"]
            ),
            "authority_inner_five_views_re_materialized": True,
            "checkpoint_forward_reexecuted": True,
            "full_candidate_ledger_exact_match": True,
            "inner_pass": True,
            "oof_semantic_tensor_read_count": 0,
            "materialization_audit_sha256": _object_sha(materialization),
        })
    if len(replay_ledger) != OUTER_FOLDS:
        raise RuntimeError("independent exact-five inner replay did not close")
    return replay_ledger, _object_sha(replay_ledger)


def _barrier_authority_binding(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_root": str(authority["feature_root"]),
        "feature_receipt_sha256": frozen.V4C_FEATURE_RECEIPT_SHA256,
        "v4a_receipt_path": str(Path(authority["v4a_path"]).resolve(strict=True)),
        "v4a_receipt_file_sha256": _file_sha(Path(authority["v4a_path"])),
        "v4c_frontier_receipt_path": str(
            Path(authority["v4c_path"]).resolve(strict=True)
        ),
        "v4c_frontier_receipt_file_sha256": _file_sha(
            Path(authority["v4c_path"])
        ),
        "v4d_receipt_path": str(Path(authority["v4d_path"]).resolve(strict=True)),
        "v4d_receipt_file_sha256": _file_sha(Path(authority["v4d_path"])),
        "outer_assignment_digest": v4c.OUTER_ASSIGNMENT_DIGEST,
        "exact644_ordered_iid_digest": _object_sha(authority["exact_iids"]),
        "frozen_inner_split_literals_sha256": _object_sha(
            list(FROZEN_INNER_SPLITS)
        ),
        "frozen_v4f_runtime_sha256": V4F_RUNTIME_DEPENDENCY_SHA256,
    }


def _barrier_members(
    receipts: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        len(receipts) != OUTER_FOLDS or len(bindings) != OUTER_FOLDS
        or len(replay) != OUTER_FOLDS
        or [row.get("fold_index") for row in receipts]
            != list(range(OUTER_FOLDS))
        or [row.get("fold_index") for row in replay]
            != list(range(OUTER_FOLDS))
    ):
        raise RuntimeError("global barrier member population differs")
    return [{
        "fold_index": fold_index,
        "fold_root": binding["fold_root"],
        "inner_receipt_binding": dict(binding),
        "preselection_checkpoint_artifact": receipt[
            "preselection_checkpoint_artifact"
        ],
        "fixed1200_checkpoint_artifact": receipt[
            "fixed1200_checkpoint_artifact"
        ],
        "preselection_fixed1200_checkpoint_pair_join": receipt[
            "preselection_fixed1200_checkpoint_pair_join"
        ],
        "fixed_candidate_ledger_sha256": _object_sha(
            receipt["fixed_candidate"]
        ),
        "independent_model_fit_provenance_replay_sha256": replay_row[
            "model_fit_provenance_replay_sha256"
        ],
        "independent_inner_replay_sha256": replay_row[
            "inner_replay_sha256"
        ],
        "inner_pass": receipt["inner_pass"],
        "oof_semantic_tensor_read_count_exact0": receipt[
            "oof_semantic_tensor_read_count_exact0"
        ],
    } for fold_index, (receipt, binding, replay_row) in enumerate(
        zip(receipts, bindings, replay)
    )]


def _barrier_replay_semantically_complete(
    members: Any, replay: Any, runtime_fingerprint: Any,
) -> bool:
    """Reject a self-consistent transport hash with incomplete replay facts."""

    if (
        type(members) is not list or type(replay) is not list
        or len(members) != OUTER_FOLDS or len(replay) != OUTER_FOLDS
        or type(runtime_fingerprint) is not dict
        or [row.get("fold_index") for row in members]
            != list(range(OUTER_FOLDS))
        or [row.get("fold_index") for row in replay]
            != list(range(OUTER_FOLDS))
    ):
        return False
    required_true = (
        "checkpoint_outer_fold_authority_join",
        "checkpoint_model_fit_ordered_iids_authority_join",
        "checkpoint_model_fit_count_and_digest_authority_join",
        "checkpoint_inner_iid_digest_authority_join",
        "checkpoint_pca_fit_input_receipt_training_join",
        "checkpoint_minibatch_schedule_receipt_training_join",
        "checkpoint_state_receipt_training_inner_join",
        "training_evaluate_runtime_fingerprint_exact_match",
        "checkpoint_clip_mean_equals_authority_recomputed_pca",
        "checkpoint_clip_basis_equals_authority_recomputed_pca",
        "checkpoint_fit_only_rms_equals_authority_recomputed_rms",
        "checkpoint_schema_and_exact79040_strict_loaded",
        "preselection_fixed1200_full_binding_reverified",
        "full1200_causal_weights_trusted_only_to_controller_pinned_sealed_training_execution",
        "authority_inner_five_views_re_materialized",
        "checkpoint_forward_reexecuted",
        "full_candidate_ledger_exact_match",
        "inner_pass",
    )
    for fold_index, (member, row) in enumerate(zip(members, replay)):
        provenance = row.get("model_fit_provenance_replay")
        inner_replay = row.get("inner_replay_binding")
        fixed = member.get("fixed1200_checkpoint_artifact")
        inner_binding = member.get("inner_receipt_binding")
        if (
            type(provenance) is not dict or type(inner_replay) is not dict
            or type(fixed) is not dict or type(inner_binding) is not dict
            or any(row.get(key) is not True for key in required_true)
            or row.get("full1200_optimizer_trajectory_reexecuted") is not False
            or row.get("duplicate_training_performed") is not False
            or row.get("oof_semantic_tensor_read_count") != 0
            or row.get("runtime_fingerprint") != runtime_fingerprint
            or row.get("inner_receipt_file_sha256")
                != inner_binding.get("file_sha256")
            or row.get("fixed1200_checkpoint_file_sha256")
                != fixed.get("file_sha256")
            or row.get("fixed1200_model_state_sha256")
                != fixed.get("model_state_sha256")
            or row.get("preselection_fixed1200_pair_join_sha256")
                != _object_sha(
                    member.get("preselection_fixed1200_checkpoint_pair_join")
                )
            or row.get("model_fit_provenance_replay_sha256")
                != _object_sha(provenance)
            or member.get(
                "independent_model_fit_provenance_replay_sha256"
            ) != row.get("model_fit_provenance_replay_sha256")
            or row.get("inner_replay_sha256") != _object_sha(inner_replay)
            or member.get("independent_inner_replay_sha256")
                != row.get("inner_replay_sha256")
            or member.get("fixed_candidate_ledger_sha256")
                != inner_replay.get("fixed_candidate_ledger_sha256")
            or inner_replay != {
                "fold_index": fold_index,
                "fixed_candidate_ledger_sha256": inner_replay.get(
                    "fixed_candidate_ledger_sha256"
                ),
                "inner_iid_digest": row.get("inner_iid_digest"),
                "inner_evidence_sha256": row.get("inner_evidence_sha256"),
                "complete_gate_sha256": row.get("complete_gate_sha256"),
                "bootstrap_seed_ledger_sha256": row.get(
                    "bootstrap_seed_ledger_sha256"
                ),
                "fixed1200_checkpoint_file_sha256": row.get(
                    "fixed1200_checkpoint_file_sha256"
                ),
                "fixed1200_model_state_sha256": row.get(
                    "fixed1200_model_state_sha256"
                ),
            }
            or provenance.get("fold_index") != fold_index
            or provenance.get("model_fit_original_count")
                != fixed.get("model_fit_original_count")
            or provenance.get("model_fit_ordered_iids")
                != fixed.get("model_fit_ordered_iids")
            or provenance.get("model_fit_iid_digest")
                != fixed.get("model_fit_iid_digest")
            or provenance.get("clip_pca_fit_input_sha256")
                != fixed.get("fixed_clip_pca_fit_input_sha256")
            or provenance.get("minibatch_schedule_sha256")
                != fixed.get("minibatch_schedule_sha256")
            or provenance.get("fold_seed")
                != Config().seed + 10000 + fold_index
            or provenance.get("minibatch_generator_seed")
                != Config().seed + 10001 + fold_index
            or provenance.get("minibatch_schedule_shape")
                != [Config().max_steps, Config().batch_size]
            or provenance.get(
                "authority_model_fit_all_five_views_re_materialized"
            ) is not True
            or provenance.get("pca_and_rms_recomputed_from_original_only")
                is not True
            or provenance.get("minibatch_schedule_regenerated_without_training")
                is not True
            or provenance.get("optimizer_steps_reexecuted") != 0
            or provenance.get("oof_semantic_tensor_read_count") != 0
        ):
            return False
    return True


def _resolve_barrier_path(value: str, *, fresh: bool) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or path.name != "barrier.json":
        raise ValueError("global barrier path must be absolute and non-symlink")
    if fresh:
        if (
            not path.parent.is_dir()
            or str(path.parent) != str(path.parent.resolve(strict=True))
            or path.exists()
        ):
            raise ValueError("global barrier output must be a fresh canonical child")
    elif (
        not path.is_file() or str(path) != str(path.resolve(strict=True))
        or stat.S_IMODE(path.lstat().st_mode) != 0o444
        or path.lstat().st_nlink != 1
    ):
        raise ValueError("global barrier receipt seal/path differs")
    return path


def _load_barrier_receipt_sealed(
    path_string: str, expected_sha256: str, run_binding: Mapping[str, str],
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _resolve_barrier_path(path_string, fresh=False)
    file_sha = _file_sha(path)
    if file_sha != expected_sha256:
        raise RuntimeError("controller-bound global barrier expected SHA differs")
    value = v4c._load_json_sealed(path, file_sha)
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    members = value.get("members")
    replay = value.get("independent_replay_ledger")
    trust = value.get("causal_training_trust_boundary")
    if (
        value.get("schema_version") != BARRIER_SCHEMA
        or value.get("status") != BARRIER_PASS_STATUS
        or digest != _object_sha(unsigned)
        or value.get("implementation") != run_binding
        or value.get("config") != _config_value(Config())
        or value.get("config_sha256") != _object_sha(_config_value(Config()))
        or value.get("authority_binding") != _barrier_authority_binding(authority)
        or value.get("replay_seed") != Config().seed + 30000
        or type(value.get("runtime_fingerprint")) is not dict
        or type(members) is not list or len(members) != OUTER_FOLDS
        or [member.get("fold_index") for member in members]
            != list(range(OUTER_FOLDS))
        or len({member.get("fold_root") for member in members}) != OUTER_FOLDS
        or any(
            member.get("inner_pass") is not True
            or member.get("oof_semantic_tensor_read_count_exact0") is not True
            for member in members
        )
        or value.get("members_sha256") != _object_sha(members)
        or type(replay) is not list or len(replay) != OUTER_FOLDS
        or value.get("independent_replay_sha256") != _object_sha(replay)
        or not _barrier_replay_semantically_complete(
            members, replay, value.get("runtime_fingerprint")
        )
        or any(
            row.get("inner_pass") is not True
            or row.get("oof_semantic_tensor_read_count") != 0
            or row.get("checkpoint_outer_fold_authority_join") is not True
            or row.get("checkpoint_model_fit_ordered_iids_authority_join")
                is not True
            or row.get("checkpoint_inner_iid_digest_authority_join") is not True
            or row.get("checkpoint_clip_basis_equals_authority_recomputed_pca")
                is not True
            or row.get("checkpoint_fit_only_rms_equals_authority_recomputed_rms")
                is not True
            for row in replay
        )
        or value.get("oof_semantic_tensor_read_count") != 0
        or value.get("oof_semantic_tensor_read_count_exact0") is not True
        or value.get("all_five_exact_one_full_gates_pass") is not True
        or value.get(
            "all_five_authority_model_fit_provenances_recomputed"
        ) is not True
        or value.get(
            "all_five_authority_inner_checkpoint_forwards_reexecuted"
        ) is not True
        or value.get(
            "evaluate_fold_accepts_only_this_barrier_path_and_controller_expected_sha"
        ) is not True
        or value.get("arbitrary_evaluate_fold_child_roots_or_inner_shas_accepted")
            is not False
        or value.get("qualification_scope") != {
            **_qualification_scope(None),
            "all_five_inner_fixed_candidate_gates_passed": True,
            "aggregate_gate_evaluated": False,
        }
        or trust != {
            "full1200_optimizer_trajectory_reexecuted_by_verifier": False,
            "duplicate_training_performed": False,
            "causal_weights_trusted_only_to_official_controller_pinned_runtime_execution": True,
            "barrier_expected_sha_must_be_supplied_by_detached_controller_not_untrusted_caller": True,
            "inference_or_refit_authorized": False,
        }
    ):
        raise RuntimeError("sealed controller-bound global barrier replay differs")
    return value, {
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha,
        "receipt_digest": digest,
        "mode_octal": "0444",
        "nlink": path.lstat().st_nlink,
        "controller_expected_sha_exact": True,
    }


def _train_inner_fold(
    authority: Mapping[str, Any], fold_index: int, config: Config,
    device: torch.device, preselection_path: Path, fixed_path: Path,
    run_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Materialize model-fit/inner only and return an OOF-empty receipt."""

    groups, split = frozen._split_fold(
        authority["ordered_records"], authority["outer_assignment"],
        fold_index, config,
    )
    upstream_fold = authority["v4a_receipt"]["folds"][fold_index]
    fit_iids = [row.iid for row in groups["model_fit"]]
    inner_iids = [row.iid for row in groups["inner_validation"]]
    oof_iids = [row.iid for row in groups["exploratory_oof"]]
    if (
        len(oof_iids) != FROZEN_OOF_COUNTS[fold_index]
        or split["outer_oof_iid_digest"] != upstream_fold["oof_iid_digest"]
        or set(fit_iids) & set(inner_iids)
        or set(fit_iids) & set(oof_iids)
        or set(inner_iids) & set(oof_iids)
    ):
        raise RuntimeError("frozen model-fit/inner/OOF partition differs")

    # Only model-fit tensors can be requested before checkpoint creation.
    stage1_request = {iid: EVAL_VIEWS for iid in fit_iids}
    stage1_rows, stage1_audit = frozen._selective_materialize_feature_rows(
        authority["feature_index"], stage1_request,
        stage="stage1_model_fit_only",
    )
    if stage1_audit["semantic_tensor_materialized_count"] != (
        len(fit_iids) * len(EVAL_VIEWS)
    ):
        raise RuntimeError("stage1 tensor count differs")
    model_fit_rows = [stage1_rows[iid] for iid in fit_iids]
    fitted = frozen._fit_clip_pca_b384(model_fit_rows)
    if fitted.fit_iid_digest != split["model_fit_iid_digest"]:
        raise RuntimeError("fold-fit PCA/IID join differs")
    model, training = _train_fold_model(
        model_fit_rows, inner_iids, fitted, config, fold_index, device
    )
    preselection = _save_checkpoint_create_only(
        preselection_path, model, fitted, training, config, fold_index,
        run_binding, model_fit_rows, device,
        checkpoint_role="preselection_fixed_step1200",
        preselection_artifact=None,
    )
    fixed = _save_checkpoint_create_only(
        fixed_path, model, fitted, training, config, fold_index,
        run_binding, model_fit_rows, device,
        checkpoint_role="fixed1200_candidate",
        preselection_artifact=preselection,
    )
    pair = _verify_distinct_checkpoint_pair(preselection, fixed)

    # Inner tensors appear only after both fixed-state artifacts strongly seal.
    stage2_rows, stage2_audit = frozen._selective_materialize_feature_rows(
        authority["feature_index"], {iid: EVAL_VIEWS for iid in inner_iids},
        stage="stage2_post_preselection_seal_inner_five_views",
    )
    if stage2_audit["semantic_tensor_materialized_count"] != (
        len(inner_iids) * len(EVAL_VIEWS)
    ):
        raise RuntimeError("stage2 tensor count differs")
    inner_rows = [stage2_rows[iid] for iid in inner_iids]
    fixed_candidate = _evaluate_fixed_inner_candidate(
        inner_rows, model, fitted, config, fold_index, device
    )
    inner_pass = bool(fixed_candidate["inner_pass"])
    receipt: dict[str, Any] = {
        "schema_version": INNER_SCHEMA,
        "status": INNER_PASS_STATUS if inner_pass else INNER_NO_GO_STATUS,
        "authority": "burned_exposed_known_transform_development_only",
        "implementation": dict(run_binding),
        "config": _config_value(config),
        "config_sha256": _object_sha(_config_value(config)),
        "fold_index": fold_index,
        "frozen_v4a_fold_iid_digest": v4c.FOLD_IID_DIGESTS[fold_index],
        "frozen_v4a_outer_assignment_digest": split[
            "outer_assignment_digest"
        ],
        "frozen_v4a_oof_iid_digest": upstream_fold["oof_iid_digest"],
        "inner_split": split,
        "model_fit_original_count": len(fit_iids),
        "model_fit_ordered_iids": fit_iids,
        "model_fit_iid_digest": _object_sha(fit_iids),
        "inner_validation_original_count": len(inner_iids),
        "inner_validation_ordered_iids": inner_iids,
        "inner_validation_iid_digest": _object_sha(inner_iids),
        "oof_original_count": len(oof_iids),
        "oof_ordered_iids": oof_iids,
        "oof_iid_digest": _object_sha(oof_iids),
        "partition_pairwise_disjoint": True,
        "fixed_clip_pca_b384_fit_input_sha256": fitted.fit_input_sha256,
        "fixed_clip_pca_b384_fit_iid_digest": fitted.fit_iid_digest,
        "fixed_clip_pca_b384_diagnostics": fitted.diagnostics,
        "runtime_fingerprint": training["runtime_fingerprint"],
        "training": training,
        "preselection_checkpoint_artifact": preselection,
        "fixed1200_checkpoint_artifact": fixed,
        "preselection_fixed1200_checkpoint_pair_join": pair,
        "fixed_candidate": fixed_candidate,
        "candidate_count": FIXED_CANDIDATE_COUNT,
        "single_candidate": True,
        "hyperparameter_selection_performed": False,
        "inner_pass": inner_pass,
        "selective_feature_materialization_before_global_barrier": {
            "stage1_model_fit_only": stage1_audit,
            "stage2_post_both_checkpoint_seals_inner_only": stage2_audit,
            "stage1_oof_semantic_tensor_count": 0,
            "stage2_oof_semantic_tensor_count": 0,
        },
        "oof_used_for_training_checkpoint_or_inner_gate": False,
        "oof_semantic_tensor_materialized_count": 0,
        "oof_semantic_tensor_read_count_exact0": True,
        "global_barrier_required_before_any_fold_oof": True,
        "scientific_design_preregistered_before_v4f_result": True,
        "v4f_fold_results_used_to_choose_any_v4g_parameter": False,
        "transform_roles_used_for_gradient_loss": True,
        "family_metadata_used_for_split_and_inner_gate_bootstrap": True,
        "family_metadata_used_for_gradient_model_input_or_loss": False,
        "qualification_scope": {
            **_qualification_scope(None),
            "inner_fold_local_gate_passed": inner_pass,
            "aggregate_gate_evaluated": False,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    return receipt


def run_train_fold(args: argparse.Namespace) -> dict[str, Any]:
    """Train one exact fold; this entry point has no OOF materialization path."""

    _require_release_sealed()
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-G Torch runtime differs")
    torch.set_num_threads(1)
    if type(args.fold_index) is not int or not 0 <= args.fold_index < OUTER_FOLDS:
        raise ValueError("fold index differs")
    _, preselection_path, fixed_path, inner_path = _resolve_fold_root(
        args.fold_root, fresh_train=True
    )
    device = _resolve_device(args.device)
    _seed_everything(config.seed + 10000 + args.fold_index, device)
    authority = frozen._prepare_authorities(args)
    receipt = _train_inner_fold(
        authority, args.fold_index, config, device, preselection_path,
        fixed_path, run_binding,
    )
    _assert_binding_unchanged(run_binding)
    frozen._reverify_authorities(authority, args)
    receipt_sha = _write_json_create_only(inner_path, receipt)
    replay, binding = _load_inner_receipt_sealed(
        args.fold_root, receipt_sha, run_binding
    )
    if replay != receipt or binding["file_sha256"] != receipt_sha:
        raise RuntimeError("fresh inner receipt strong replay differs")
    return {
        "inner_receipt": str(inner_path.resolve(strict=True)),
        "inner_receipt_sha256": receipt_sha,
        "inner_receipt_digest": receipt["receipt_digest"],
        "fold_index": args.fold_index,
        "inner_pass": receipt["inner_pass"],
        "oof_semantic_tensor_read_count": 0,
        "inference_authorized": False,
    }


def run_verify_inner_barrier(args: argparse.Namespace) -> dict[str, Any]:
    """Create the sole exact-five barrier artifact; never materialize OOF."""

    _require_release_sealed()
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-G Torch runtime differs")
    torch.set_num_threads(1)
    device = _resolve_device(args.device)
    replay_seed = config.seed + 30000
    _seed_everything(replay_seed, device)
    runtime_fingerprint = _runtime_fingerprint(device)
    output = _resolve_barrier_path(args.barrier_output, fresh=True)
    authority = frozen._prepare_authorities(args)
    receipts, bindings, transport_sha = (
        _load_all_inner_receipts_or_fail_before_oof(
            args.fold_root, args.expected_inner_receipt_sha256, run_binding,
        )
    )
    replay, replay_sha = _independently_replay_all_inner_gates_before_oof(
        receipts, bindings, authority, config, device,
    )
    members = _barrier_members(receipts, bindings, replay)
    receipt: dict[str, Any] = {
        "schema_version": BARRIER_SCHEMA,
        "status": BARRIER_PASS_STATUS,
        "authority": "burned_exposed_known_transform_development_only",
        "implementation": dict(run_binding),
        "config": _config_value(config),
        "config_sha256": _object_sha(_config_value(config)),
        "authority_binding": _barrier_authority_binding(authority),
        "runtime_fingerprint": runtime_fingerprint,
        "replay_seed": replay_seed,
        "members": members,
        "members_sha256": _object_sha(members),
        "inner_transport_barrier_sha256": transport_sha,
        "independent_replay_ledger": replay,
        "independent_replay_sha256": replay_sha,
        "all_five_exact_one_full_gates_pass": True,
        "all_five_authority_model_fit_provenances_recomputed": True,
        "all_five_authority_inner_checkpoint_forwards_reexecuted": True,
        "oof_semantic_tensor_read_count": 0,
        "oof_semantic_tensor_read_count_exact0": True,
        "evaluate_fold_accepts_only_this_barrier_path_and_controller_expected_sha": True,
        "arbitrary_evaluate_fold_child_roots_or_inner_shas_accepted": False,
        "causal_training_trust_boundary": {
            "full1200_optimizer_trajectory_reexecuted_by_verifier": False,
            "duplicate_training_performed": False,
            "causal_weights_trusted_only_to_official_controller_pinned_runtime_execution": True,
            "barrier_expected_sha_must_be_supplied_by_detached_controller_not_untrusted_caller": True,
            "inference_or_refit_authorized": False,
        },
        "qualification_scope": {
            **_qualification_scope(None),
            "all_five_inner_fixed_candidate_gates_passed": True,
            "aggregate_gate_evaluated": False,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    frozen._reverify_authorities(authority, args)
    receipt_sha = _write_json_create_only(output, receipt)
    replayed, barrier_binding = _load_barrier_receipt_sealed(
        str(output), receipt_sha, run_binding, authority,
    )
    if replayed != receipt:
        raise RuntimeError("fresh global barrier strong replay differs")
    return {
        "barrier_receipt": str(output.resolve(strict=True)),
        "barrier_receipt_sha256": receipt_sha,
        "barrier_receipt_digest": receipt["receipt_digest"],
        "barrier_binding": barrier_binding,
        "all_five_inner_pass": True,
        "oof_semantic_tensor_read_count": 0,
        "inference_authorized": False,
    }


def run_evaluate_fold(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one OOF fold only after the all-five inner PASS barrier."""

    _require_release_sealed()
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-G Torch runtime differs")
    torch.set_num_threads(1)
    if (
        type(args.fold_index) is not int
        or not 0 <= args.fold_index < OUTER_FOLDS
    ):
        raise ValueError("fold index differs")
    device = _resolve_device(args.device)
    replay_seed = config.seed + 30000
    _seed_everything(replay_seed, device)
    replay_runtime = _runtime_fingerprint(device)
    authority = frozen._prepare_authorities(args)

    # GLOBAL BARRIER: the detached controller pins one create-only exact-five
    # barrier.  evaluate-fold accepts no caller-supplied child roots or SHAs.
    barrier_receipt, barrier_binding = _load_barrier_receipt_sealed(
        args.barrier_receipt, args.expected_barrier_receipt_sha256,
        run_binding, authority,
    )
    barrier_members = barrier_receipt["members"]
    member_roots = [member["fold_root"] for member in barrier_members]
    member_inner_shas = [
        member["inner_receipt_binding"]["file_sha256"]
        for member in barrier_members
    ]
    inner_receipts, inner_bindings, barrier_sha = (
        _load_all_inner_receipts_or_fail_before_oof(
            member_roots, member_inner_shas, run_binding
        )
    )
    if (
        _barrier_members(
            inner_receipts, inner_bindings,
            barrier_receipt["independent_replay_ledger"],
        ) != barrier_members
        or barrier_sha != barrier_receipt["inner_transport_barrier_sha256"]
    ):
        raise RuntimeError("controller barrier member/checkpoint binding changed")
    independent_replay, independent_replay_sha = (
        _independently_replay_all_inner_gates_before_oof(
            inner_receipts, inner_bindings, authority, config, device
        )
    )
    if (
        independent_replay
            != barrier_receipt["independent_replay_ledger"]
        or independent_replay_sha
            != barrier_receipt["independent_replay_sha256"]
    ):
        raise RuntimeError("evaluate replay differs from controller barrier replay")
    target = inner_receipts[args.fold_index]
    target_root = Path(inner_bindings[args.fold_index]["fold_root"])
    fold_path = target_root / "fold.json"
    if fold_path.exists() or fold_path.is_symlink():
        raise ValueError("fold evaluation output must be fresh")
    fixed_artifact = target["fixed1200_checkpoint_artifact"]
    _, fixed_expected = _checkpoint_expectations_from_inner_receipt(
        target, expected_runtime_fingerprint=replay_runtime,
    )
    metadata, state, fixed_binding = _load_checkpoint_sealed(
        target_root / "fixed1200.pt", fixed_expected
    )
    if fixed_binding["file_sha256"] != fixed_artifact["file_sha256"]:
        raise RuntimeError("fixed checkpoint changed after global barrier")
    fitted = ClipPCAFit(
        clip_mean=state["clip_mean"], clip_basis=state["clip_basis"],
        fit_iid_digest=str(metadata["model_fit_iid_digest"]),
        fit_input_sha256=str(metadata["basis"]["fixed_clip_pca_fit_input_sha256"]),
        diagnostics=target["fixed_clip_pca_b384_diagnostics"],
    )
    model = VJepa2GlobalCodec(fitted, state["fit_only_rms"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    groups, split = frozen._split_fold(
        authority["ordered_records"], authority["outer_assignment"],
        args.fold_index, config,
    )
    oof_iids = [row.iid for row in groups["exploratory_oof"]]
    if (
        oof_iids != target["oof_ordered_iids"]
        or split != target["inner_split"]
    ):
        raise RuntimeError("target OOF split differs after global barrier")

    # This is the first and only OOF semantic tensor request in this process.
    oof_map, oof_audit = frozen._selective_materialize_feature_rows(
        authority["feature_index"], {iid: EVAL_VIEWS for iid in oof_iids},
        stage="stage3_post_selected_seal_oof",
    )
    if oof_audit["semantic_tensor_materialized_count"] != (
        len(oof_iids) * len(EVAL_VIEWS)
    ):
        raise RuntimeError("OOF selective materialization count differs")
    evidence = _evaluate_rows_fixed(
        [oof_map[iid] for iid in oof_iids], model, fitted, config, device
    )
    for row in evidence:
        row["outer_fold"] = args.fold_index
    if [row["iid"] for row in evidence] != oof_iids:
        raise RuntimeError("OOF evidence order differs")
    receipt: dict[str, Any] = {
        "schema_version": FOLD_SCHEMA,
        "status": STATUS,
        "authority": "burned_exposed_known_transform_development_only",
        "implementation": dict(run_binding),
        "config": _config_value(config),
        "config_sha256": _object_sha(_config_value(config)),
        "fold_index": args.fold_index,
        "evaluate_replay_seed": replay_seed,
        "runtime_fingerprint": replay_runtime,
        "global_inner_barrier": {
            "controller_barrier_receipt_binding": barrier_binding,
            "controller_barrier_receipt_digest": barrier_receipt[
                "receipt_digest"
            ],
            "exact_five_inner_receipts_verified": True,
            "all_five_exact_one_full_gates_pass": True,
            "barrier_sha256": barrier_sha,
            "inner_receipt_bindings": inner_bindings,
            "all_five_checkpoints_independently_forward_replayed": True,
            "all_five_authority_inner_populations_re_materialized": True,
            "all_five_full_candidate_ledgers_exact_match": True,
            "independent_replay_sha256": independent_replay_sha,
            "independent_replay_ledger": independent_replay,
            "evaluate_replay_seed": replay_seed,
            "runtime_fingerprint": replay_runtime,
            "barrier_completed_before_any_oof_tensor_request": True,
        },
        "inner_receipt_file_sha256": inner_bindings[
            args.fold_index
        ]["file_sha256"],
        "controller_barrier_receipt_file_sha256": barrier_binding[
            "file_sha256"
        ],
        "fixed1200_checkpoint_artifact": fixed_artifact,
        "fixed1200_evaluate_checkpoint_binding": fixed_binding,
        "inner_split": split,
        "oof_original_count": len(oof_iids),
        "oof_ordered_iids": oof_iids,
        "oof_iid_digest": _object_sha(oof_iids),
        "oof_selective_materialization": oof_audit,
        "oof_evidence_count": len(evidence),
        "oof_evidence_sha256": _object_sha(evidence),
        "oof_evidence": evidence,
        "oof_used_for_training_checkpoint_inner_gate_or_selection": False,
        "family_metadata_used_for_final_gate_and_bootstrap": True,
        "family_metadata_used_for_gradient_model_input_or_loss": False,
        "qualification_scope": {
            **_qualification_scope(None),
            "inner_fold_local_gate_passed": target["inner_pass"],
            "aggregate_gate_evaluated": False,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    frozen._reverify_authorities(authority, args)
    receipt_sha = _write_json_create_only(fold_path, receipt)
    return {
        "fold_receipt": str(fold_path.resolve(strict=True)),
        "fold_receipt_sha256": receipt_sha,
        "fold_receipt_digest": receipt["receipt_digest"],
        "fold_index": args.fold_index,
        "oof_original_count": len(evidence),
        "inference_authorized": False,
    }


def _load_fold_receipt_sealed(
    fold_root: str, expected_sha256: str,
    run_binding: Mapping[str, str], barrier_sha: str,
    barrier_receipt_binding: Mapping[str, Any],
    barrier_receipt_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(fold_root).resolve(strict=True)
    path = root / "fold.json"
    file_sha = _file_sha(path)
    if file_sha != expected_sha256:
        raise RuntimeError("evaluated-fold receipt expected SHA differs")
    value = v4c._load_json_sealed(path, file_sha)
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    fold_index = value.get("fold_index")
    evidence = value.get("oof_evidence")
    barrier = value.get("global_inner_barrier")
    fixed_artifact = value.get("fixed1200_checkpoint_artifact")
    fixed_binding = value.get("fixed1200_evaluate_checkpoint_binding")
    independent_replay = (
        barrier.get("independent_replay_ledger")
        if type(barrier) is dict else None
    )
    if (
        value.get("schema_version") != FOLD_SCHEMA
        or value.get("status") != STATUS
        or digest != _object_sha(unsigned)
        or value.get("implementation") != run_binding
        or value.get("config") != _config_value(Config())
        or value.get("config_sha256") != _object_sha(_config_value(Config()))
        or type(fold_index) is not int or not 0 <= fold_index < OUTER_FOLDS
        or type(barrier) is not dict
        or barrier.get("exact_five_inner_receipts_verified") is not True
        or barrier.get("all_five_exact_one_full_gates_pass") is not True
        or barrier.get("all_five_checkpoints_independently_forward_replayed")
            is not True
        or barrier.get("all_five_authority_inner_populations_re_materialized")
            is not True
        or barrier.get("all_five_full_candidate_ledgers_exact_match") is not True
        or type(independent_replay) is not list
        or len(independent_replay) != OUTER_FOLDS
        or barrier.get("independent_replay_sha256")
            != _object_sha(independent_replay)
        or barrier.get("evaluate_replay_seed") != Config().seed + 30000
        or barrier.get("runtime_fingerprint")
            != value.get("runtime_fingerprint")
        or type(value.get("runtime_fingerprint")) is not dict
        or barrier.get("barrier_completed_before_any_oof_tensor_request") is not True
        or barrier.get("barrier_sha256") != barrier_sha
        or value.get("controller_barrier_receipt_file_sha256")
            != barrier_receipt_binding.get("file_sha256")
        or barrier.get("controller_barrier_receipt_binding")
            != barrier_receipt_binding
        or barrier.get("controller_barrier_receipt_digest")
            != barrier_receipt_digest
        or type(evidence) is not list
        or type(fixed_artifact) is not dict or type(fixed_binding) is not dict
        or fixed_binding.get("file_sha256")
            != fixed_artifact.get("file_sha256")
        or fixed_binding.get("model_state_sha256")
            != fixed_artifact.get("model_state_sha256")
        or fixed_binding.get("outer_fold") != fold_index
        or len(evidence) != value.get("oof_original_count")
        or len(evidence) != value.get("oof_evidence_count")
        or _object_sha(evidence) != value.get("oof_evidence_sha256")
        or [row.get("iid") for row in evidence] != value.get("oof_ordered_iids")
        or any(
            row.get("outer_fold") != fold_index
            or row.get("fixed_step") != FIXED_SELECTED_STEP
            or row.get("fixed_residual_scale") != FIXED_RESIDUAL_SCALE
            or row.get("single_fixed_candidate") is not True
            for row in evidence
        )
        or value.get("oof_used_for_training_checkpoint_inner_gate_or_selection")
            is not False
        or value.get("qualification_scope", {}).get("inference_authorized")
            is not False
        or value.get("qualification_scope") != {
            **_qualification_scope(None),
            "inner_fold_local_gate_passed": True,
            "aggregate_gate_evaluated": False,
        }
    ):
        raise RuntimeError("sealed evaluated-fold receipt replay differs")
    return value, {
        "fold_root": str(root),
        "path": str(path.resolve(strict=True)),
        "file_sha256": file_sha,
        "receipt_digest": digest,
        "mode_octal": "0444",
        "nlink": path.lstat().st_nlink,
    }


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Aggregate five sealed OOF folds after independently replaying the barrier."""

    _require_release_sealed()
    run_binding = _binding()
    config = Config()
    config.validate()
    if str(torch.__version__) != "2.7.1+rocm6.3":
        raise RuntimeError("v4-G Torch runtime differs")
    torch.set_num_threads(1)
    output = Path(args.output)
    if (
        not output.is_absolute() or not output.parent.is_dir()
        or output.exists() or output.is_symlink()
    ):
        raise ValueError("aggregate output must be a fresh absolute JSON child")
    authority = frozen._prepare_authorities(args)
    barrier_receipt, barrier_binding = _load_barrier_receipt_sealed(
        args.barrier_receipt, args.expected_barrier_receipt_sha256,
        run_binding, authority,
    )
    barrier_members = barrier_receipt["members"]
    member_roots = [member["fold_root"] for member in barrier_members]
    member_inner_shas = [
        member["inner_receipt_binding"]["file_sha256"]
        for member in barrier_members
    ]
    inner_receipts, inner_bindings, barrier_sha = (
        _load_all_inner_receipts_or_fail_before_oof(
            member_roots, member_inner_shas, run_binding
        )
    )
    if (
        _barrier_members(
            inner_receipts, inner_bindings,
            barrier_receipt["independent_replay_ledger"],
        ) != barrier_members
        or barrier_sha != barrier_receipt["inner_transport_barrier_sha256"]
    ):
        raise RuntimeError("aggregate controller barrier membership differs")
    for receipt in inner_receipts:
        _verify_inner_receipt_against_authority(receipt, authority, config)
    if (
        len(args.expected_fold_receipt_sha256) != OUTER_FOLDS
        or len(set(args.expected_fold_receipt_sha256)) != OUTER_FOLDS
    ):
        raise ValueError("aggregate requires exactly five expected fold receipt SHAs")
    loaded = [
        _load_fold_receipt_sealed(
            root, expected, run_binding, barrier_sha,
            barrier_binding, barrier_receipt["receipt_digest"],
        )
        for root, expected in zip(
            member_roots, args.expected_fold_receipt_sha256
        )
    ]
    loaded.sort(key=lambda item: item[0]["fold_index"])
    folds = [item[0] for item in loaded]
    fold_bindings = [item[1] for item in loaded]
    if [row["fold_index"] for row in folds] != list(range(OUTER_FOLDS)):
        raise RuntimeError("aggregate lacks evaluated folds 0..4 exactly once")
    for fold, inner, inner_binding in zip(
        folds, inner_receipts, inner_bindings
    ):
        if (
            fold.get("inner_receipt_file_sha256")
                != inner_binding["file_sha256"]
            or fold.get("fixed1200_checkpoint_artifact")
                != inner.get("fixed1200_checkpoint_artifact")
            or fold.get("inner_split") != inner.get("inner_split")
            or fold.get("global_inner_barrier", {}).get(
                "inner_receipt_bindings"
            ) != inner_bindings
            or fold.get("runtime_fingerprint")
                != inner.get("runtime_fingerprint")
        ):
            raise RuntimeError("evaluated fold/global inner barrier join differs")
    replay_ledgers = [
        fold["global_inner_barrier"]["independent_replay_ledger"]
        for fold in folds
    ]
    replay_shas = [
        fold["global_inner_barrier"]["independent_replay_sha256"]
        for fold in folds
    ]
    if (
        len(set(replay_shas)) != 1
        or any(ledger != replay_ledgers[0] for ledger in replay_ledgers[1:])
        or replay_ledgers[0] != barrier_receipt["independent_replay_ledger"]
        or replay_shas[0] != barrier_receipt["independent_replay_sha256"]
    ):
        raise RuntimeError("per-evaluator independent global replay differs")
    evidence = [row for fold in folds for row in fold["oof_evidence"]]
    if (
        len(evidence) != 644 or len({row["iid"] for row in evidence}) != 644
        or {row["iid"] for row in evidence} != set(authority["exact_iids"])
        or tuple(
            sum(int(row["outer_fold"]) == fold for row in evidence)
            for fold in range(OUTER_FOLDS)
        ) != FROZEN_OOF_COUNTS
    ):
        raise RuntimeError("aggregate OOF union is not exact644 once each")
    upstream_match = frozen._verify_v4c_embedded_teacher_evidence(
        evidence, authority["v4c_receipt"]
    )
    metrics = frozen._aggregate(evidence, config)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "authority": "burned_exposed_known_transform_development_only",
        "implementation": dict(run_binding),
        "config": _config_value(config),
        "config_sha256": _object_sha(_config_value(config)),
        "controller_barrier_receipt_binding": barrier_binding,
        "controller_barrier_receipt_digest": barrier_receipt[
            "receipt_digest"
        ],
        "runtime": {
            "torch": str(torch.__version__),
            "torch_hip": str(torch.version.hip),
            "aggregate_device": "cpu",
            "model_trained_or_recomputed": False,
            "model_forward_executed": False,
        },
        "feature_authority": {
            "feature_root": str(authority["feature_root"]),
            "feature_receipt_sha256": frozen.V4C_FEATURE_RECEIPT_SHA256,
            "unique_original_iids": 644,
            "family_count": 28,
            "stored_views": list(EVAL_VIEWS),
        },
        "upstream_authorities": {
            "v4a_receipt_file_sha256": frozen.V4A_RECEIPT_FILE_SHA256,
            "v4c_frontier_receipt_file_sha256": frozen.V4C_FRONTIER_RECEIPT_SHA256,
            "v4d_burned_receipt_file_sha256": frozen.V4D_RECEIPT_SHA256,
            "frozen_final_v4f_runtime_sha256": V4F_RUNTIME_DEPENDENCY_SHA256,
            "v4c_embedded_teacher_evidence_match": upstream_match,
            "v4g_scientific_design_preregistered_before_v4f_result": True,
            "v4f_fold_results_used_to_choose_any_v4g_parameter": False,
            "frozen_v4f_dependency_used_only_for_preexisting_audited_primitives": True,
            "v4f_candidate_selection_function_invoked": False,
        },
        "frozen_split": {
            "outer_assignment_digest": v4c.OUTER_ASSIGNMENT_DIGEST,
            "oof_counts_by_fold": list(FROZEN_OOF_COUNTS),
            "inner_source": INNER_SPLIT_NAMESPACE,
            "inner_literal_pins": list(FROZEN_INNER_SPLITS),
            "all_exact644_are_burned_development": True,
        },
        "model_contract": {
            "input": "C(view) FP32 [32,1024]",
            "code_shape": [CODE_TIME, CODE_CHANNELS],
            "actual_code_numel": CODE_NUMEL,
            "decoder_input": "sole [12,32] code",
            "raw_input_skip_or_side_channel": False,
            "step0": "exact fold-fit fixed clip-PCA-B384 encoder/decoder",
            "exact_trainable_parameter_count": EXACT_TRAINABLE_PARAMETERS,
            "fixed_residual_scale": FIXED_RESIDUAL_SCALE,
            "deployment_scale_or_candidate_selection": False,
        },
        "training_contract": {
            "all_five_known_views_exposed_for_each_model_fit_iid": True,
            "all_five_view_reconstruction_terms_equal_weight": True,
            "exchangeable_ten_pair_geometry_removed": True,
            "exact_three_role_directed_decoded_teacher_margins": True,
            "teacher_margin_roles": {
                "query": "original", "positive": "monotone_warp",
                "negatives": list(NEGATIVES),
            },
            "teacher_margin_scale": (
                "stopgrad(mean_all_10_teacher_pair_distances)+1e-8"
            ),
            "teacher_margin_weight": config.teacher_margin_weight,
            "teacher_margin_smooth_l1_beta": config.teacher_margin_beta,
            "transform_roles_used_for_gradient_loss": True,
            "transform_roles_used_for_model_input": False,
            "family_metadata_used_for_split_and_inner_final_gate_bootstrap": True,
            "family_metadata_used_for_gradient_model_input_or_loss": False,
            "fixed_full_budget_no_early_stop": True,
            "fixed_step": FIXED_SELECTED_STEP,
            "fixed_residual_scale": FIXED_RESIDUAL_SCALE,
            "candidate_count": FIXED_CANDIDATE_COUNT,
            "single_candidate": True,
            "hyperparameter_selection_performed": False,
            "oof_selection": False,
        },
        "global_inner_barrier": {
            "exact_five_inner_receipts_verified": True,
            "all_five_exact_one_full_gates_pass": True,
            "barrier_sha256": barrier_sha,
            "inner_receipt_bindings": inner_bindings,
            "all_five_evaluators_independently_replayed_all_five_inner_gates": True,
            "per_evaluator_independent_replay_sha256": replay_shas,
            "common_independent_replay_sha256": replay_shas[0],
            "common_independent_replay_ledger": replay_ledgers[0],
            "every_train_fold_oof_read_count_exact0": all(
                row["oof_semantic_tensor_read_count_exact0"] is True
                for row in inner_receipts
            ),
            "any_inner_failure_forbids_every_fold_evaluate_command": True,
            "cross_fold_selection_performed": False,
            "causal_training_trust_boundary": barrier_receipt[
                "causal_training_trust_boundary"
            ],
        },
        "evaluation_contract": {
            "known_exposed_transform_families_only": True,
            "unseen_hostile_transform_gate_evaluated": False,
            "final_oof_thresholds_and_seed_namespace_exactly_reused_from_v4f": True,
            "family_metadata_used_for_final_gate_and_bootstrap": True,
            "family_metadata_used_for_gradient_model_input_or_loss": False,
            "fidelity_gate": (
                "each view dual ratio UCB<=1.05 and every fold ratio<=1.05"
            ),
            "negative_gate": (
                "each negative teacher/candidate/retention/improvement dual "
                "LCB>0 and every fold>0"
            ),
        },
        "inner_receipts": {
            "count": len(inner_bindings),
            "bindings": inner_bindings,
        },
        "fold_receipts": {
            "count": len(fold_bindings),
            "bindings": fold_bindings,
        },
        "folds": folds,
        "oof_closure": {
            "unique_original_iids": 644,
            "each_original_evaluated_exactly_once": True,
            "oof_counts_by_fold": list(FROZEN_OOF_COUNTS),
            "embedded_per_iid_evidence_count": len(evidence),
            "embedded_per_iid_evidence_sha256": _object_sha(evidence),
            "embedded_per_iid_evidence": evidence,
            "evidence_sufficient_to_recompute_all_gates": True,
        },
        "metrics": metrics,
        "qualification_scope": {
            **_qualification_scope(metrics[
                "exposed_five_view_codec_development_gate"
            ]),
            "all_five_inner_fixed_candidate_gates_passed": True,
            "aggregate_gate_evaluated": True,
        },
    }
    receipt["receipt_digest"] = _object_sha(receipt)
    _assert_binding_unchanged(run_binding)
    frozen._reverify_authorities(authority, args)
    receipt_sha = _write_json_create_only(output, receipt)
    replay = v4c._load_json_sealed(output, receipt_sha)
    if replay != receipt:
        raise RuntimeError("fresh aggregate receipt replay differs")
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
        description="NO-GO until sealed: v4-G role-directed teacher margins"
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

    barrier = subparsers.add_parser("verify-inner-barrier")
    add_authorities(barrier)
    barrier.add_argument("--fold-root", action="append", required=True)
    barrier.add_argument(
        "--expected-inner-receipt-sha256", action="append", required=True
    )
    barrier.add_argument("--barrier-output", required=True)
    barrier.add_argument("--device", default="cuda")
    barrier.set_defaults(handler=run_verify_inner_barrier)

    evaluate = subparsers.add_parser("evaluate-fold")
    add_authorities(evaluate)
    evaluate.add_argument("--fold-index", type=int, required=True)
    evaluate.add_argument("--barrier-receipt", required=True)
    evaluate.add_argument("--expected-barrier-receipt-sha256", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.set_defaults(handler=run_evaluate_fold)

    aggregate = subparsers.add_parser("aggregate")
    add_authorities(aggregate)
    aggregate.add_argument("--barrier-receipt", required=True)
    aggregate.add_argument("--expected-barrier-receipt-sha256", required=True)
    aggregate.add_argument(
        "--expected-fold-receipt-sha256", action="append", required=True
    )
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
