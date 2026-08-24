"""Closed experiment configuration for mask-free OmniVideo2 action tuning."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_contract import (
    OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS,
    OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID,
)


ACTION_CONFIG_FORMAT = "marp-omnivideo2-action-training-v2"
VLM_DIM = 2048
MOTION_TOKEN_DIM = 2048
WAN_VAE_CHANNELS = 16
WAN_VAE_TEMPORAL_STRIDE = 4
WAN_VAE_SPATIAL_STRIDE = 8
ACTION_TASK_TYPES = (
    "action_edit",
    "identity_reconstruction",
    "native_replay",
    "native_isolation_probe",
)
LORA_SCOPES = ("diffsynth_full", "all_attn", "cross_qo")
CONTEXT_PADDING_MODES = ("fixed_budget", "batch_exact")
TEMPORAL_MODES = (
    "full_81_25fps",
    "smoke_41_12p5fps",
    "synthetic_fixture",
)
SPATIAL_PROFILES = ("full_480p", "motion_384p", "synthetic_fixture")


class ActionConfigError(ValueError):
    """Raised when a configuration is incomplete, ambiguous, or unsafe."""


def _closed_mapping(
    value: Any, *, name: str, fields: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionConfigError(f"{name} must be an object")
    result = dict(value)
    actual = set(result)
    if actual != fields:
        raise ActionConfigError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return result


def _integer(
    value: Any, *, name: str, minimum: int = 0, allow_zero: bool = True
) -> int:
    if type(value) is not int:
        raise ActionConfigError(f"{name} must be an integer")
    lower = minimum if allow_zero else max(1, minimum)
    if value < lower:
        raise ActionConfigError(f"{name} must be at least {lower}")
    return value


def _number(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ActionConfigError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ActionConfigError(f"{name} must be finite")
    too_large = maximum is not None and (
        result > maximum if maximum_inclusive else result >= maximum
    )
    if result < minimum or too_large:
        suffix = (
            f" and {'at most' if maximum_inclusive else 'smaller than'} {maximum}"
            if maximum is not None
            else ""
        )
        raise ActionConfigError(f"{name} must be at least {minimum}{suffix}")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ActionConfigError(f"{name} must be bool")
    return value


def _triplet(value: Any, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ActionConfigError(f"{name} must be a length-3 integer list")
    return tuple(
        _integer(item, name=f"{name}[{index}]", minimum=1, allow_zero=False)
        for index, item in enumerate(value)
    )


def _betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ActionConfigError("optimizer.betas must be a length-2 number list")
    first = _number(
        value[0],
        name="optimizer.betas[0]",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    second = _number(
        value[1],
        name="optimizer.betas[1]",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    return first, second


def _task_types(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ActionConfigError("training.allowed_task_types must be a non-empty list")
    if any(type(item) is not str for item in value):
        raise ActionConfigError(
            "training.allowed_task_types must contain only strings"
        )
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ActionConfigError(
            "training.allowed_task_types cannot contain duplicates"
        )
    unknown = sorted(set(result) - set(ACTION_TASK_TYPES))
    if unknown:
        raise ActionConfigError(f"unknown training task types: {unknown}")
    return result


@dataclass(frozen=True)
class ModelConfig:
    max_context_len: int
    checkpoint_contract_id: str
    context_padding_mode: str
    expected_special_token_rows: int
    visual_patch_size: tuple[int, int, int]
    wan_patch_size: tuple[int, int, int]
    require_special_tokens: bool
    require_uncompressed_source: bool
    gradient_checkpointing: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelConfig":
        item = _closed_mapping(
            value,
            name="model",
            fields={
                "max_context_len",
                "checkpoint_contract_id",
                "context_padding_mode",
                "expected_special_token_rows",
                "visual_patch_size",
                "wan_patch_size",
                "require_special_tokens",
                "require_uncompressed_source",
                "gradient_checkpointing",
            },
        )
        padding_mode = item["context_padding_mode"]
        if type(padding_mode) is not str or padding_mode not in CONTEXT_PADDING_MODES:
            raise ActionConfigError(
                "model.context_padding_mode must be one of "
                f"{list(CONTEXT_PADDING_MODES)}"
            )
        checkpoint_contract_id = item["checkpoint_contract_id"]
        if (
            type(checkpoint_contract_id) is not str
            or checkpoint_contract_id != OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID
        ):
            raise ActionConfigError(
                "model.checkpoint_contract_id must equal the pinned "
                f"OmniVideo2-1.3B contract {OMNIVIDEO2_1_3B_CHECKPOINT_CONTRACT_ID!r}"
            )
        return cls(
            max_context_len=_integer(
                item["max_context_len"],
                name="model.max_context_len",
                minimum=1,
                allow_zero=False,
            ),
            checkpoint_contract_id=checkpoint_contract_id,
            context_padding_mode=padding_mode,
            expected_special_token_rows=_integer(
                item["expected_special_token_rows"],
                name="model.expected_special_token_rows",
                minimum=0,
            ),
            visual_patch_size=_triplet(
                item["visual_patch_size"], name="model.visual_patch_size"
            ),
            wan_patch_size=_triplet(
                item["wan_patch_size"], name="model.wan_patch_size"
            ),
            require_special_tokens=_boolean(
                item["require_special_tokens"],
                name="model.require_special_tokens",
            ),
            require_uncompressed_source=_boolean(
                item["require_uncompressed_source"],
                name="model.require_uncompressed_source",
            ),
            gradient_checkpointing=_boolean(
                item["gradient_checkpointing"],
                name="model.gradient_checkpointing",
            ),
        )


@dataclass(frozen=True)
class DataConfig:
    """Exact decoded-video geometry expected by one training run.

    Payloads store VAE latents rather than decoded frames.  Keeping the video
    geometry in the closed config makes 81-frame training auditable and lets
    the loader derive the one permissible latent shape without trusting a
    filename or a free-form provenance note.
    """

    video_num_frames: int
    video_fps: float
    video_height: int
    video_width: int
    temporal_mode: str
    spatial_profile: str
    allow_transpose: bool
    smoke_only: bool
    require_materialization_metadata: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataConfig":
        item = _closed_mapping(
            value,
            name="data",
            fields={
                "video_num_frames",
                "video_fps",
                "video_height",
                "video_width",
                "temporal_mode",
                "spatial_profile",
                "allow_transpose",
                "smoke_only",
                "require_materialization_metadata",
            },
        )
        frames = _integer(
            item["video_num_frames"],
            name="data.video_num_frames",
            minimum=1,
            allow_zero=False,
        )
        if frames % WAN_VAE_TEMPORAL_STRIDE != 1:
            raise ActionConfigError(
                "data.video_num_frames must be 4n+1 for the Wan VAE"
            )
        height = _integer(
            item["video_height"],
            name="data.video_height",
            minimum=32,
            allow_zero=False,
        )
        width = _integer(
            item["video_width"],
            name="data.video_width",
            minimum=32,
            allow_zero=False,
        )
        # VAE /8 followed by the official Visual Context Adapter /4 must be
        # exact.  Silent convolution-border token loss is forbidden.
        for name, size in (("height", height), ("width", width)):
            if size % (WAN_VAE_SPATIAL_STRIDE * 4):
                raise ActionConfigError(
                    f"data.video_{name} must be divisible by 32"
                )
        fps = _number(item["video_fps"], name="data.video_fps", minimum=1e-12)
        smoke_only = _boolean(item["smoke_only"], name="data.smoke_only")
        allow_transpose = _boolean(
            item["allow_transpose"], name="data.allow_transpose"
        )
        require_metadata = _boolean(
            item["require_materialization_metadata"],
            name="data.require_materialization_metadata",
        )
        temporal_mode = item["temporal_mode"]
        spatial_profile = item["spatial_profile"]
        if type(temporal_mode) is not str or temporal_mode not in TEMPORAL_MODES:
            raise ActionConfigError(
                f"data.temporal_mode must be one of {list(TEMPORAL_MODES)}"
            )
        if type(spatial_profile) is not str or spatial_profile not in SPATIAL_PROFILES:
            raise ActionConfigError(
                f"data.spatial_profile must be one of {list(SPATIAL_PROFILES)}"
            )
        expected_temporal = {
            "full_81_25fps": (81, 25.0, False),
            "smoke_41_12p5fps": (41, 12.5, True),
            "synthetic_fixture": (5, 1.0, True),
        }[temporal_mode]
        if (frames, fps, smoke_only) != expected_temporal:
            raise ActionConfigError(
                "data temporal_mode disagrees with frames/fps/smoke_only: "
                f"mode={temporal_mode!r} requires {expected_temporal}"
            )
        expected_spatial = {
            "full_480p": (480, 832),
            "motion_384p": (384, 640),
            "synthetic_fixture": (64, 64),
        }[spatial_profile]
        if (height, width) != expected_spatial:
            raise ActionConfigError(
                "data spatial_profile disagrees with height/width: "
                f"profile={spatial_profile!r} requires {expected_spatial}"
            )
        is_synthetic = temporal_mode == spatial_profile == "synthetic_fixture"
        if (temporal_mode == "synthetic_fixture") != (
            spatial_profile == "synthetic_fixture"
        ):
            raise ActionConfigError("synthetic temporal/spatial profiles must be paired")
        if is_synthetic and require_metadata:
            raise ActionConfigError(
                "synthetic_fixture cannot require real materialization metadata"
            )
        if not is_synthetic and not require_metadata:
            raise ActionConfigError(
                "real temporal/spatial profiles require materialization metadata"
            )
        return cls(
            video_num_frames=frames,
            video_fps=fps,
            video_height=height,
            video_width=width,
            temporal_mode=temporal_mode,
            spatial_profile=spatial_profile,
            allow_transpose=allow_transpose,
            smoke_only=smoke_only,
            require_materialization_metadata=require_metadata,
        )

    @property
    def expected_latent_shape(self) -> tuple[int, int, int, int]:
        return (
            WAN_VAE_CHANNELS,
            (self.video_num_frames - 1) // WAN_VAE_TEMPORAL_STRIDE + 1,
            self.video_height // WAN_VAE_SPATIAL_STRIDE,
            self.video_width // WAN_VAE_SPATIAL_STRIDE,
        )

    @property
    def expected_latent_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        primary = self.expected_latent_shape
        if not self.allow_transpose or self.video_height == self.video_width:
            return (primary,)
        return (primary, (primary[0], primary[1], primary[3], primary[2]))

    @property
    def expected_raw_num_frames(self) -> int:
        return 5 if self.temporal_mode == "synthetic_fixture" else 81

    @property
    def expected_raw_fps(self) -> float:
        return 1.0 if self.temporal_mode == "synthetic_fixture" else 25.0


@dataclass(frozen=True)
class FlowConfig:
    shift: float
    num_train_timesteps: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlowConfig":
        item = _closed_mapping(
            value,
            name="flow",
            fields={"shift", "num_train_timesteps"},
        )
        return cls(
            shift=_number(item["shift"], name="flow.shift", minimum=1e-12),
            num_train_timesteps=_integer(
                item["num_train_timesteps"],
                name="flow.num_train_timesteps",
                minimum=2,
                allow_zero=False,
            ),
        )


@dataclass(frozen=True)
class LoraConfig:
    scope: str
    rank: int
    alpha: float
    dropout: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LoraConfig":
        item = _closed_mapping(
            value,
            name="lora",
            fields={"scope", "rank", "alpha", "dropout"},
        )
        scope = item["scope"]
        if type(scope) is not str or scope not in LORA_SCOPES:
            raise ActionConfigError(f"lora.scope must be one of {list(LORA_SCOPES)}")
        return cls(
            scope=scope,
            rank=_integer(
                item["rank"], name="lora.rank", minimum=1, allow_zero=False
            ),
            alpha=_number(item["alpha"], name="lora.alpha", minimum=1e-12),
            dropout=_number(
                item["dropout"],
                name="lora.dropout",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
        )


@dataclass(frozen=True)
class PlannerConfig:
    num_tokens: int
    input_dim: int
    hidden_dim: int
    depth: int
    weight: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlannerConfig":
        item = _closed_mapping(
            value,
            name="planner",
            fields={"num_tokens", "input_dim", "hidden_dim", "depth", "weight"},
        )
        input_dim = _integer(
            item["input_dim"],
            name="planner.input_dim",
            minimum=1,
            allow_zero=False,
        )
        if input_dim != VLM_DIM:
            raise ActionConfigError(f"planner.input_dim must be exactly {VLM_DIM}")
        hidden_dim = _integer(
            item["hidden_dim"],
            name="planner.hidden_dim",
            minimum=8,
            allow_zero=False,
        )
        if hidden_dim % 8:
            raise ActionConfigError("planner.hidden_dim must be divisible by 8")
        return cls(
            num_tokens=_integer(
                item["num_tokens"],
                name="planner.num_tokens",
                minimum=1,
                allow_zero=False,
            ),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=_integer(
                item["depth"],
                name="planner.depth",
                minimum=1,
                allow_zero=False,
            ),
            weight=_number(item["weight"], name="planner.weight", minimum=0.0),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OptimizerConfig":
        item = _closed_mapping(
            value,
            name="optimizer",
            fields={
                "learning_rate",
                "betas",
                "weight_decay",
                "eps",
            },
        )
        return cls(
            learning_rate=_number(
                item["learning_rate"],
                name="optimizer.learning_rate",
                minimum=1e-16,
            ),
            betas=_betas(item["betas"]),
            weight_decay=_number(
                item["weight_decay"],
                name="optimizer.weight_decay",
                minimum=0.0,
            ),
            eps=_number(item["eps"], name="optimizer.eps", minimum=1e-16),
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    num_workers: int
    mixed_precision: str
    log_every: int
    save_every: int
    allow_preview: bool
    allowed_task_types: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        item = _closed_mapping(
            value,
            name="training",
            fields={
                "batch_size",
                "gradient_accumulation_steps",
                "max_steps",
                "num_workers",
                "mixed_precision",
                "log_every",
                "save_every",
                "allow_preview",
                "allowed_task_types",
            },
        )
        positive = {
            key: _integer(
                item[key], name=f"training.{key}", minimum=1, allow_zero=False
            )
            for key in (
                "batch_size",
                "gradient_accumulation_steps",
                "log_every",
                "save_every",
            )
        }
        mixed_precision = item["mixed_precision"]
        if type(mixed_precision) is not str or mixed_precision not in {
            "no",
            "fp16",
            "bf16",
        }:
            raise ActionConfigError(
                "training.mixed_precision must be 'no', 'fp16', or 'bf16'"
            )
        return cls(
            batch_size=positive["batch_size"],
            gradient_accumulation_steps=positive[
                "gradient_accumulation_steps"
            ],
            max_steps=_integer(
                item["max_steps"], name="training.max_steps", minimum=0
            ),
            num_workers=_integer(
                item["num_workers"], name="training.num_workers", minimum=0
            ),
            mixed_precision=mixed_precision,
            log_every=positive["log_every"],
            save_every=positive["save_every"],
            allow_preview=_boolean(
                item["allow_preview"], name="training.allow_preview"
            ),
            allowed_task_types=_task_types(item["allowed_task_types"]),
        )


@dataclass(frozen=True)
class ActionConfig:
    """Validated configuration with closed top-level and nested schemas."""

    format: str
    seed: int
    data: DataConfig
    model: ModelConfig
    flow: FlowConfig
    lora: LoraConfig
    planner: PlannerConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActionConfig":
        item = _closed_mapping(
            value,
            name="action config",
            fields={
                "format",
                "seed",
                "data",
                "model",
                "flow",
                "lora",
                "planner",
                "optimizer",
                "training",
            },
        )
        if type(item["format"]) is not str or item["format"] != ACTION_CONFIG_FORMAT:
            raise ActionConfigError(f"format must be {ACTION_CONFIG_FORMAT!r}")
        data = DataConfig.from_mapping(item["data"])
        model = ModelConfig.from_mapping(item["model"])
        training = TrainingConfig.from_mapping(item["training"])
        if not model.require_uncompressed_source:
            raise ActionConfigError(
                "model.require_uncompressed_source must be true; source truncation "
                "is forbidden"
            )
        if not model.require_special_tokens:
            raise ActionConfigError(
                "model.require_special_tokens must be true for the pinned "
                "OmniVideo2-1.3B checkpoint contract"
            )
        if (
            model.expected_special_token_rows
            != OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS
        ):
            raise ActionConfigError(
                "model.expected_special_token_rows must equal the pinned official "
                "OmniVideo2-1.3B checkpoint row count "
                f"({OMNIVIDEO2_1_3B_ACTIVE_SPECIAL_TOKEN_ROWS}) when special "
                "tokens are required"
            )
        if (
            model.context_padding_mode == "batch_exact"
            and training.batch_size != 1
        ):
            raise ActionConfigError(
                "training.batch_size must be 1 for the batch_exact padding ablation"
            )
        latent_t, latent_h, latent_w = data.expected_latent_shape[1:]
        visual_patch = model.visual_patch_size
        if any(
            size % patch
            for size, patch in zip(
                (latent_t, latent_h, latent_w), visual_patch
            )
        ):
            raise ActionConfigError(
                "configured latent grid is not divisible by model.visual_patch_size"
            )
        visual_tokens = math.prod(
            size // patch
            for size, patch in zip(
                (latent_t, latent_h, latent_w), visual_patch
            )
        )
        if visual_tokens >= model.max_context_len:
            raise ActionConfigError(
                "model.max_context_len cannot hold the configured source visual "
                f"tokens: visual={visual_tokens}, budget={model.max_context_len}"
            )
        return cls(
            format=ACTION_CONFIG_FORMAT,
            seed=_integer(item["seed"], name="seed", minimum=0),
            data=data,
            model=model,
            flow=FlowConfig.from_mapping(item["flow"]),
            lora=LoraConfig.from_mapping(item["lora"]),
            planner=PlannerConfig.from_mapping(item["planner"]),
            optimizer=OptimizerConfig.from_mapping(item["optimizer"]),
            training=training,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model"]["visual_patch_size"] = list(self.model.visual_patch_size)
        value["model"]["wan_patch_size"] = list(self.model.wan_patch_size)
        value["optimizer"]["betas"] = list(self.optimizer.betas)
        value["training"]["allowed_task_types"] = list(
            self.training.allowed_task_types
        )
        return value


def validate_action_config(value: Mapping[str, Any] | ActionConfig) -> ActionConfig:
    if isinstance(value, ActionConfig):
        return ActionConfig.from_mapping(value.to_dict())
    return ActionConfig.from_mapping(value)


def load_action_config(path: str | Path) -> ActionConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActionConfigError(
            f"cannot load action config at {config_path}: {error}"
        ) from error
    return ActionConfig.from_mapping(value)


__all__ = [
    "ACTION_CONFIG_FORMAT",
    "ACTION_TASK_TYPES",
    "CONTEXT_PADDING_MODES",
    "SPATIAL_PROFILES",
    "TEMPORAL_MODES",
    "LORA_SCOPES",
    "MOTION_TOKEN_DIM",
    "VLM_DIM",
    "ActionConfig",
    "ActionConfigError",
    "DataConfig",
    "FlowConfig",
    "LoraConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PlannerConfig",
    "TrainingConfig",
    "load_action_config",
    "validate_action_config",
]
