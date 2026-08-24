#!/usr/bin/env python3
"""Reversible PAIR-v7 Action-LoRA routing for stock Bernini V-only sampling.

This module owns no denoising algorithm.  It wraps one official Bernini
``v2v_apg`` call only long enough to place the already-installed PAIR-v5
Action-LoRA in an authenticated :class:`PairV5ActionRoute` around each stock
``shared_step`` forward.  The official negative/action forward pair and the
official UniPC ``scheduler.step`` are otherwise called once with their exact
original arguments.

The accepted deployment surface is deliberately narrow: one source-video
latent, no image/reference/mask/flow/pose/track branch, exact81 latent time,
the captured shift-5 exact40 schedule, one Bernini-R 1.3B transformer, and an
explicit DP2 x SP4 rank coordinate.  Schedule indices 0..32 use the registered
weight 1, indices 33..37 weight .5, and indices 38..39 enter the registered
low-sigma route whose existing Action-LoRA implementation directly returns
the frozen base projection.

This is an inference routing primitive, not a trainer and not evidence of a
semantic action-editing improvement.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import inspect
import math
import re
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import inference_sigma_strata as sigma_strata


SCHEMA_VERSION = "bernini-pair-v7-vonly-exact81-route-runtime-v1"
EXPECTED_FRAMES = 81
EXPECTED_LATENT_PHASES = 21
EXPECTED_STEPS = 40
EXPECTED_FLOW_SHIFT = 5.0
EXPECTED_GUIDANCE_MODE = "v2v_apg"
EXPECTED_MODEL_ID = "transformer_1"
EXPECTED_HIDDEN_DIM = 1536
EXPECTED_TEXT_TOKENS = 512
EXPECTED_TEXT_DIM = 4096
EXPECTED_DP_SIZE = 2
EXPECTED_SP_SIZE = 4
EXPECTED_WORLD_SIZE = EXPECTED_DP_SIZE * EXPECTED_SP_SIZE
FORWARDS_PER_CELL = 2
FORWARD_ORDER = ("negative", "action")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_FORBIDDEN_VISUAL_TOKENS = (
    "mask",
    "swept_tube",
    "pose",
    "track",
    "trajectory",
    "optical_flow",
    "flow_field",
    "reference_image",
    "reference_video",
    "target_video",
    "first_frame",
    "image_vae_latents",
    "multi_image_vae_latents",
)


class PairV7VOnlyRouteRuntimeError(RuntimeError):
    """Raised before an unauthenticated forward or UniPC integration."""


def _load_action_adapter_module() -> Any:
    """Load the existing Action-LoRA lazily so model-free tests can import us."""

    try:
        return importlib.import_module("pair_v5_action_adapter")
    except Exception as error:  # pragma: no cover - exercised on model hosts
        raise PairV7VOnlyRouteRuntimeError(
            "PAIR-v5 Action-LoRA runtime is unavailable"
        ) from error


def _resolve_diffusion_core(renderer_or_diffusion: Any) -> Any:
    """Resolve the GEN_Wanx22-like object without accepting an ambiguous leaf."""

    queue = [renderer_or_diffusion]
    seen: set[int] = set()
    matches: list[Any] = []
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if all(
            callable(getattr(candidate, name, None))
            for name in ("sample", "shared_step")
        ) and callable(getattr(getattr(candidate, "scheduler", None), "step", None)):
            matches.append(candidate)
        nested = getattr(candidate, "diff_dec", None)
        if nested is not None:
            queue.append(nested)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None:
                queue.append(nested)
    unique = {id(value): value for value in matches}
    if len(unique) != 1:
        raise PairV7VOnlyRouteRuntimeError(
            "expected exactly one GEN_Wanx22-compatible diffusion core"
        )
    return next(iter(unique.values()))


def _flatten_bound_arguments(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise PairV7VOnlyRouteRuntimeError(
            "call does not match the pinned Bernini signature"
        ) from error
    values = dict(bound.arguments)
    for parameter in signature.parameters.values():
        if parameter.kind is not parameter.VAR_KEYWORD:
            continue
        extra = values.pop(parameter.name, {})
        if not isinstance(extra, Mapping):
            raise PairV7VOnlyRouteRuntimeError("variadic keyword payload differs")
        for name, value in extra.items():
            if name in values:
                raise PairV7VOnlyRouteRuntimeError(
                    f"duplicate variadic keyword {name}"
                )
            values[name] = value
    return values


def _extract_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    if len(args) > index and name in kwargs:
        raise PairV7VOnlyRouteRuntimeError(f"call received duplicate {name}")
    if len(args) > index:
        return args[index]
    if name in kwargs:
        return kwargs[name]
    raise PairV7VOnlyRouteRuntimeError(f"call is missing {name}")


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.shape)
    except Exception as error:
        raise PairV7VOnlyRouteRuntimeError(
            f"{label} must expose an integer tensor shape"
        ) from error
    if any(item < 0 for item in shape):
        raise PairV7VOnlyRouteRuntimeError(f"{label} has a negative dimension")
    return shape


def _finite_tensor(value: Any) -> bool:
    """Validate finiteness for torch tensors or the explicit test double API."""

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        try:
            return bool(torch.isfinite(value).all().item())
        except Exception:
            return False
    checker = getattr(value, "isfinite_all", None)
    if callable(checker):
        try:
            return checker() is True
        except Exception:
            return False
    return False


def _detached_tensor(value: Any, *, label: str) -> None:
    if (
        not hasattr(value, "shape")
        or not hasattr(value, "dtype")
        or not hasattr(value, "device")
        or bool(getattr(value, "requires_grad", False))
        or getattr(value, "grad_fn", None) is not None
        or not _finite_tensor(value)
    ):
        raise PairV7VOnlyRouteRuntimeError(
            f"{label} must be a detached finite tensor"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PairV7VOnlyRouteRuntimeError(f"{label} must be lowercase SHA-256")
    return value


def _tensor_sha256(value: Any, *, label: str) -> str:
    """Digest one materialized tensor with the project-wide canonical codec.

    The explicit hook is only the model-free tensor-double surface used by this
    module's tests.  Real tensors always flow through ``source_self_runtime``.
    """

    test_double = getattr(value, "pair_v7_tensor_sha256", None)
    if callable(test_double):
        try:
            return _require_sha256(test_double(), label=f"{label} digest")
        except PairV7VOnlyRouteRuntimeError:
            raise
        except Exception as error:
            raise PairV7VOnlyRouteRuntimeError(
                f"cannot digest {label} test tensor"
            ) from error
    try:
        runtime = importlib.import_module("source_self_runtime")
        digest = runtime.tensor_sha256(value)
    except Exception as error:
        raise PairV7VOnlyRouteRuntimeError(
            f"cannot canonically digest {label}"
        ) from error
    return _require_sha256(digest, label=f"{label} digest")


def _object_sha256(value: Mapping[str, Any]) -> str:
    try:
        runtime = importlib.import_module("source_self_runtime")
        digest = runtime.object_sha256(value)
    except Exception as error:
        raise PairV7VOnlyRouteRuntimeError(
            "cannot canonically digest route receipt"
        ) from error
    return _require_sha256(digest, label="route object digest")


def _scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise PairV7VOnlyRouteRuntimeError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        result = float(candidate)
    except PairV7VOnlyRouteRuntimeError:
        raise
    except Exception as error:
        raise PairV7VOnlyRouteRuntimeError(f"{label} must be scalar") from error
    if not math.isfinite(result):
        raise PairV7VOnlyRouteRuntimeError(f"{label} must be finite")
    return result


def _exact_int_scalar(value: Any, *, label: str) -> int:
    numeric = _scalar(value, label=label)
    integer = int(numeric)
    if numeric != float(integer):
        raise PairV7VOnlyRouteRuntimeError(f"{label} must be an exact integer")
    return integer


def _metadata_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise PairV7VOnlyRouteRuntimeError(
            f"{label} must be positive integer list/tuple metadata"
        )
    return tuple(value)


def _same_object(left: Any, right: Any, *, label: str) -> None:
    if left is not right:
        raise PairV7VOnlyRouteRuntimeError(
            f"{label} must be the exact same official sampler object"
        )


def _normalized_threshold(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if not value or any(
            _scalar(item, label="norm_threshold") != 50.0 for item in value
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "norm_threshold must contain only 50"
            )
        return 50.0
    observed = _scalar(value, label="norm_threshold")
    if observed != 50.0:
        raise PairV7VOnlyRouteRuntimeError("norm_threshold must equal 50")
    return observed


def _explicit_non_none_visual_inputs(values: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in values.items():
        lowered = str(name).lower()
        extra_image_or_video = (
            ("image" in lowered or "video" in lowered)
            and lowered != "multi_video_vae_latents"
        )
        if value is not None and (
            extra_image_or_video
            or any(token in lowered for token in _FORBIDDEN_VISUAL_TOKENS)
        ):
            result.append(str(name))
    return sorted(result)


def _selector_values(selector: Any) -> tuple[bool, ...]:
    try:
        candidate = selector.detach() if hasattr(selector, "detach") else selector
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "tolist"):
            candidate = candidate.tolist()
        values = tuple(candidate)
    except Exception as error:
        raise PairV7VOnlyRouteRuntimeError(
            "Action-LoRA local target selector is unreadable"
        ) from error
    if any(type(item) is not bool for item in values):
        raise PairV7VOnlyRouteRuntimeError(
            "Action-LoRA local target selector must be boolean"
        )
    return values


@dataclass(frozen=True)
class PairV7DPSPRouteMetadata:
    """Explicit DP-major rank coordinate used by PAIR-v7 DP2 x SP4."""

    data_parallel_rank: int
    data_parallel_size: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    global_rank: int
    world_size: int

    def validate(self) -> None:
        values = (
            self.data_parallel_rank,
            self.data_parallel_size,
            self.sequence_parallel_rank,
            self.sequence_parallel_size,
            self.global_rank,
            self.world_size,
        )
        if any(type(value) is not int for value in values):
            raise PairV7VOnlyRouteRuntimeError(
                "DP/SP route metadata must contain exact integers"
            )
        if (
            self.data_parallel_size != EXPECTED_DP_SIZE
            or self.sequence_parallel_size != EXPECTED_SP_SIZE
            or self.world_size != EXPECTED_WORLD_SIZE
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "PAIR-v7 deployment requires DP2 x SP4 WORLD8"
            )
        if not 0 <= self.data_parallel_rank < self.data_parallel_size:
            raise PairV7VOnlyRouteRuntimeError("DP rank lies outside DP2")
        if not 0 <= self.sequence_parallel_rank < self.sequence_parallel_size:
            raise PairV7VOnlyRouteRuntimeError("SP rank lies outside SP4")
        if not 0 <= self.global_rank < self.world_size:
            raise PairV7VOnlyRouteRuntimeError("global rank lies outside WORLD8")
        expected_global = (
            self.data_parallel_rank * self.sequence_parallel_size
            + self.sequence_parallel_rank
        )
        if self.global_rank != expected_global:
            raise PairV7VOnlyRouteRuntimeError(
                "global rank differs from DP-major DP2 x SP4 coordinate"
            )

    def as_dict(self) -> dict[str, int]:
        self.validate()
        return {
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "global_rank": self.global_rank,
            "world_size": self.world_size,
        }


def _validate_live_distributed_route(
    parallel: PairV7DPSPRouteMetadata,
) -> Mapping[str, Any]:
    """Collectively prove that caller metadata is the live WORLD8 topology."""

    try:
        dist = importlib.import_module("torch.distributed")
        if not dist.is_available() or not dist.is_initialized():
            raise PairV7VOnlyRouteRuntimeError(
                "PAIR-v7 route requires initialized WORLD8 collectives"
            )
        world = int(dist.get_world_size())
        rank = int(dist.get_rank())
        if world != EXPECTED_WORLD_SIZE or rank != parallel.global_rank:
            raise PairV7VOnlyRouteRuntimeError(
                "live WORLD rank differs from registered DP2 x SP4 coordinate"
            )
        gathered: list[Any] = [None] * EXPECTED_WORLD_SIZE
        dist.all_gather_object(gathered, parallel.as_dict())
    except PairV7VOnlyRouteRuntimeError:
        raise
    except Exception as error:
        raise PairV7VOnlyRouteRuntimeError(
            "cannot establish live WORLD8 route consensus"
        ) from error
    expected = {
        (
            dp,
            sp,
            dp * EXPECTED_SP_SIZE + sp,
        )
        for dp in range(EXPECTED_DP_SIZE)
        for sp in range(EXPECTED_SP_SIZE)
    }
    observed = set()
    for row in gathered:
        if not isinstance(row, Mapping):
            raise PairV7VOnlyRouteRuntimeError(
                "WORLD8 route consensus payload differs"
            )
        metadata = PairV7DPSPRouteMetadata(
            data_parallel_rank=row.get("data_parallel_rank"),
            data_parallel_size=row.get("data_parallel_size"),
            sequence_parallel_rank=row.get("sequence_parallel_rank"),
            sequence_parallel_size=row.get("sequence_parallel_size"),
            global_rank=row.get("global_rank"),
            world_size=row.get("world_size"),
        )
        metadata.validate()
        observed.add(
            (
                metadata.data_parallel_rank,
                metadata.sequence_parallel_rank,
                metadata.global_rank,
            )
        )
    if observed != expected or len(gathered) != len(observed):
        raise PairV7VOnlyRouteRuntimeError(
            "WORLD8 does not contain each DP2 x SP4 coordinate exactly once"
        )
    return {
        "world_size": world,
        "global_rank": rank,
        "all_gather_object_consensus": True,
        "dp_major_coordinate_set_complete": True,
    }


@dataclass(frozen=True)
class PairV7VOnlyExact81RouteConfig:
    """Pinned exact81 geometry plus mandatory distributed route ownership."""

    target_latent_shape: tuple[int, int, int, int, int]
    parallel: PairV7DPSPRouteMetadata
    expected_seed: int
    expected_source_latent_sha256: str
    expected_action_prompt_sha256: str
    expected_negative_prompt_sha256: str
    expected_hidden_dim: int = EXPECTED_HIDDEN_DIM
    expected_text_tokens: int = EXPECTED_TEXT_TOKENS
    expected_text_dim: int = EXPECTED_TEXT_DIM

    @property
    def target_tokens(self) -> int:
        _, _, phases, height, width = self.target_latent_shape
        return int(phases * (height // 2) * (width // 2))

    @property
    def total_v_tokens(self) -> int:
        return 2 * self.target_tokens

    def validate(self) -> None:
        shape = tuple(self.target_latent_shape)
        if (
            len(shape) != 5
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[0] != 1
            or shape[1] != 16
            or shape[2] != EXPECTED_LATENT_PHASES
            or shape[3] % 2
            or shape[4] % 2
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "target latent must be exact81 Bernini [1,16,21,even,even]"
            )
        if (
            self.expected_hidden_dim != EXPECTED_HIDDEN_DIM
            or self.expected_text_tokens != EXPECTED_TEXT_TOKENS
            or self.expected_text_dim != EXPECTED_TEXT_DIM
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "runtime is pinned to Bernini-R 1.3B hidden/text geometry"
            )
        if type(self.expected_seed) is not int or self.expected_seed < 0:
            raise PairV7VOnlyRouteRuntimeError(
                "expected_seed must be one explicit non-negative integer"
            )
        _require_sha256(
            self.expected_source_latent_sha256,
            label="expected source latent digest",
        )
        _require_sha256(
            self.expected_action_prompt_sha256,
            label="expected action prompt digest",
        )
        _require_sha256(
            self.expected_negative_prompt_sha256,
            label="expected negative prompt digest",
        )
        if not isinstance(self.parallel, PairV7DPSPRouteMetadata):
            raise PairV7VOnlyRouteRuntimeError(
                "explicit PairV7DPSPRouteMetadata is required"
            )
        self.parallel.validate()


@dataclass(frozen=True)
class _ForwardObservation:
    role: str
    schedule_index: int
    noisy_latents: Any
    timesteps: Any
    rotary_embs: Any
    batch_vae_seqlen: tuple[int, ...]


@dataclass
class _ActiveSample:
    action_prompt: Any
    negative_prompt: Any
    completed_cells: int = 0
    pending: list[_ForwardObservation] = field(default_factory=list)


class PairV7VOnlyExact81RoutePatch:
    """Reversible wrapper around one official source-video-only sample."""

    _MARKER = "_bernini_pair_v7_vonly_exact81_route_runtime_v1"

    def __init__(
        self,
        renderer_or_diffusion: Any,
        *,
        action_handle: Any,
        config: PairV7VOnlyExact81RouteConfig,
    ) -> None:
        config.validate()
        action_module = _load_action_adapter_module()
        handle_type = getattr(action_module, "PairV5ActionAdapterHandle", None)
        if not isinstance(handle_type, type) or not isinstance(action_handle, handle_type):
            raise PairV7VOnlyRouteRuntimeError(
                "action_handle must be an existing PairV5ActionAdapterHandle"
            )
        diffusion = _resolve_diffusion_core(renderer_or_diffusion)
        transformer = getattr(diffusion, "transformer", None)
        scheduler = getattr(diffusion, "scheduler", None)
        if (
            getattr(action_handle, "transformer", None) is not transformer
            or bool(getattr(action_handle, "restored", True))
            or not callable(getattr(action_handle, "route", None))
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "Action-LoRA handle is not live on the sampled transformer"
            )
        if getattr(diffusion, "use_unipc", None) is not True:
            raise PairV7VOnlyRouteRuntimeError("runtime requires native UniPC")
        if getattr(diffusion, "transformer_2", None) is not None:
            raise PairV7VOnlyRouteRuntimeError(
                "runtime supports only Bernini-R 1.3B transformer_1"
            )
        if _scalar(
            getattr(diffusion, "switch_dit_boundary", None),
            label="switch_dit_boundary",
        ) != 0.0:
            raise PairV7VOnlyRouteRuntimeError(
                "Bernini-R 1.3B route requires switch_dit_boundary=0"
            )
        if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
            getattr(transformer, "is_gradient_checkpointing", False)
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "gradient checkpointing is forbidden for branch-local routes"
            )
        transformer_config = getattr(transformer, "config", None)

        def config_value(name: str) -> Any:
            if isinstance(transformer_config, Mapping):
                return transformer_config.get(name)
            return getattr(transformer_config, name, None)

        heads = config_value("num_attention_heads")
        head_dim = config_value("attention_head_dim")
        if (
            type(heads) is not int
            or type(head_dim) is not int
            or heads * head_dim != config.expected_hidden_dim
            or config_value("in_channels") != 16
            or config_value("out_channels") != 16
            or tuple(config_value("patch_size") or ()) != (1, 2, 2)
            or config_value("text_dim") != config.expected_text_dim
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "transformer is not the pinned Bernini-R 1.3B geometry"
            )
        originals = {
            "sample": getattr(diffusion, "sample", None),
            "shared_step": getattr(diffusion, "shared_step", None),
            "scheduler.step": getattr(scheduler, "step", None),
        }
        if any(not callable(value) for value in originals.values()):
            raise PairV7VOnlyRouteRuntimeError(
                "pinned Bernini sampler call surface differs"
            )
        for owner, name in (
            (diffusion, "sample"),
            (diffusion, "shared_step"),
            (scheduler, "step"),
        ):
            try:
                if name in vars(owner):
                    raise PairV7VOnlyRouteRuntimeError(
                        f"refusing stacked instance override on {name}"
                    )
            except TypeError as error:
                raise PairV7VOnlyRouteRuntimeError(
                    f"cannot inspect {name} owner"
                ) from error
        if any(getattr(value, self._MARKER, None) is not None for value in originals.values()):
            raise PairV7VOnlyRouteRuntimeError("PAIR-v7 route wrapper is already installed")
        active_route = getattr(action_module, "active_route", None)
        if not callable(active_route) or active_route() is not None:
            raise PairV7VOnlyRouteRuntimeError(
                "Action-LoRA has a pre-existing active route"
            )

        self.diffusion = diffusion
        self.transformer = transformer
        self.scheduler = scheduler
        self.action_handle = action_handle
        self.action_module = action_module
        self.config = config
        self.distributed_audit = dict(_validate_live_distributed_route(config.parallel))
        self.original_sample = originals["sample"]
        self.original_shared_step = originals["shared_step"]
        self.original_scheduler_step = originals["scheduler.step"]
        self._patches: list[tuple[Any, str, bool, Any, Any]] = []
        self._active: Optional[_ActiveSample] = None
        self.installed = False
        self.restored = False
        self.finalized = False
        self.schedule_audit: Optional[Mapping[str, Any]] = None
        self.original_sample_invocations = 0
        self.successful_sample_calls = 0
        self.original_shared_step_calls = 0
        self.original_scheduler_step_calls = 0
        self.trace: list[dict[str, Any]] = []
        self.input_binding: Optional[dict[str, Any]] = None
        self.sample_contract: Optional[dict[str, Any]] = None
        self.output_latent_sha256: Optional[str] = None

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        try:
            instance = vars(owner)
        except TypeError as error:
            raise PairV7VOnlyRouteRuntimeError(
                f"cannot reversibly patch {name} owner"
            ) from error
        had_instance = name in instance
        previous = instance.get(name)
        resolved_before = getattr(owner, name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous, resolved_before))

    def install(self) -> None:
        if self.installed or self.restored or self.finalized:
            raise PairV7VOnlyRouteRuntimeError("PAIR-v7 route patch lifecycle differs")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, scheduler_wrapper):
            setattr(wrapper, self._MARKER, self)
        try:
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self._restore_patches(require_wrapper_identity=False)
            raise
        self.installed = True

    def _restore_patches(self, *, require_wrapper_identity: bool) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous, resolved_before = self._patches.pop()
            try:
                current = getattr(owner, name, None)
                if require_wrapper_identity and getattr(
                    current, self._MARKER, None
                ) is not self:
                    errors.append(
                        PairV7VOnlyRouteRuntimeError(
                            f"{name} changed during PAIR-v7 route patch"
                        )
                    )
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
                if getattr(owner, name, None) != resolved_before:
                    errors.append(
                        PairV7VOnlyRouteRuntimeError(f"{name} restoration failed")
                    )
            except Exception as error:
                errors.append(error)
        self._active = None
        if errors:
            raise PairV7VOnlyRouteRuntimeError(
                f"failed to restore {len(errors)} PAIR-v7 route wrapper(s)"
            ) from errors[0]

    def restore(self) -> None:
        if not self.installed or self.restored:
            raise PairV7VOnlyRouteRuntimeError("PAIR-v7 route patch restore differs")
        try:
            self._restore_patches(require_wrapper_identity=True)
        finally:
            self.installed = False
            self.restored = not self._patches

    def _validate_prompt(self, value: Any, *, label: str) -> None:
        _detached_tensor(value, label=label)
        if _shape(value, label=label) != (
            1,
            self.config.expected_text_tokens,
            self.config.expected_text_dim,
        ):
            raise PairV7VOnlyRouteRuntimeError(f"{label} prompt geometry differs")

    def _validate_sample_contract(self, values: Mapping[str, Any]) -> _ActiveSample:
        if (
            values.get("guidance_mode") != EXPECTED_GUIDANCE_MODE
            or values.get("num_frames") != EXPECTED_FRAMES
            or values.get("num_inference_steps") != EXPECTED_STEPS
            or _scalar(values.get("flow_shift"), label="flow_shift")
            != EXPECTED_FLOW_SHIFT
            or _scalar(values.get("omega_txt"), label="omega_txt") != 4.0
            or _scalar(values.get("omega_vid"), label="omega_vid") != 1.25
            or _scalar(values.get("omega_scale"), label="omega_scale") != 0.8
            or _scalar(values.get("eta"), label="eta") != 0.5
            or _scalar(values.get("momentum"), label="momentum") != 0.0
            or values.get("prompt_embeds_t2") is not None
            or values.get("uncond_embeds_t2") is not None
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "official exact81/exact40 v2v_apg sample contract differs"
            )
        _, _, _, latent_height, latent_width = self.config.target_latent_shape
        if (
            type(values.get("seed")) is not int
            or values.get("seed") != self.config.expected_seed
            or type(values.get("width")) is not int
            or values.get("width") != latent_width * 8
            or type(values.get("height")) is not int
            or values.get("height") != latent_height * 8
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "seed or decoded spatial geometry differs from registration"
            )
        if "omega_img" in values and _scalar(
            values.get("omega_img"), label="omega_img"
        ) != 0.0:
            raise PairV7VOnlyRouteRuntimeError(
                "source-video-only V sampling requires omega_img=0"
            )
        _normalized_threshold(values.get("norm_threshold"))
        forbidden = _explicit_non_none_visual_inputs(values)
        if forbidden:
            raise PairV7VOnlyRouteRuntimeError(
                "image/reference/mask/extra visual inputs are forbidden: "
                + ",".join(forbidden)
            )
        videos = values.get("multi_video_vae_latents")
        if not isinstance(videos, (list, tuple)) or len(videos) != 1:
            raise PairV7VOnlyRouteRuntimeError(
                "V-only sampling requires exactly one source-video latent"
            )
        source = videos[0]
        _detached_tensor(source, label="source-video latent")
        if _shape(source, label="source-video latent") != tuple(
            self.config.target_latent_shape
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "source-video latent is not the exact81 target geometry"
            )
        action = values.get("prompt_embeds")
        negative = values.get("uncond_prompt_embeds")
        self._validate_prompt(action, label="action")
        self._validate_prompt(negative, label="negative")
        if action is negative:
            raise PairV7VOnlyRouteRuntimeError(
                "action and negative prompt objects must be distinct"
            )
        binding = {
            "seed": values["seed"],
            "width": values["width"],
            "height": values["height"],
            "source_latent_sha256": _tensor_sha256(
                source, label="source-video latent"
            ),
            "action_prompt_sha256": _tensor_sha256(action, label="action prompt"),
            "negative_prompt_sha256": _tensor_sha256(
                negative, label="negative prompt"
            ),
        }
        expected_binding = {
            "seed": self.config.expected_seed,
            "width": latent_width * 8,
            "height": latent_height * 8,
            "source_latent_sha256": self.config.expected_source_latent_sha256,
            "action_prompt_sha256": self.config.expected_action_prompt_sha256,
            "negative_prompt_sha256": self.config.expected_negative_prompt_sha256,
        }
        if binding != expected_binding:
            raise PairV7VOnlyRouteRuntimeError(
                "source/action/negative tensor binding differs from registration"
            )
        self.input_binding = binding
        self.sample_contract = {
            "guidance_mode": EXPECTED_GUIDANCE_MODE,
            "num_frames": EXPECTED_FRAMES,
            "num_inference_steps": EXPECTED_STEPS,
            "flow_shift": EXPECTED_FLOW_SHIFT,
            "omega_vid": 1.25,
            "omega_img": 0.0,
            "omega_txt": 4.0,
            "omega_scale": 0.8,
            "eta": 0.5,
            "norm_threshold": 50.0,
            "momentum": 0.0,
            "prompt_embeds_t2": None,
            "uncond_embeds_t2": None,
            "source_video_latent_count": 1,
            **binding,
        }
        return _ActiveSample(action_prompt=action, negative_prompt=negative)

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if (
            self._active is not None
            or self.original_sample_invocations != 0
            or self.action_module.active_route() is not None
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "route patch permits exactly one non-nested official sample"
            )
        if (
            self.diffusion.scheduler is not self.scheduler
            or self.action_handle.transformer is not self.transformer
            or bool(getattr(self.action_handle, "restored", True))
        ):
            raise PairV7VOnlyRouteRuntimeError("sample ownership changed")
        values = _flatten_bound_arguments(self.original_sample, args, kwargs)
        state = self._validate_sample_contract(values)
        self._active = state
        self.original_sample_invocations += 1
        try:
            result = self.original_sample(*args, **kwargs)
            if (
                state.completed_cells != EXPECTED_STEPS
                or state.pending
                or self.original_shared_step_calls
                != EXPECTED_STEPS * FORWARDS_PER_CELL
                or self.original_scheduler_step_calls != EXPECTED_STEPS
                or len(self.trace) != EXPECTED_STEPS
            ):
                raise PairV7VOnlyRouteRuntimeError(
                    "official sample returned without exact40 two-forward closure"
                )
            _detached_tensor(result, label="official sample result")
            if _shape(result, label="official sample result") != tuple(
                self.config.target_latent_shape
            ):
                raise PairV7VOnlyRouteRuntimeError(
                    "official sample result is not exact81 latent geometry"
                )
            if str(getattr(result, "dtype", "")) != "torch.float32":
                raise PairV7VOnlyRouteRuntimeError(
                    "official exact81 sample result must be torch.float32"
                )
            self.output_latent_sha256 = _tensor_sha256(
                result, label="official exact81 output latent"
            )
            self.successful_sample_calls += 1
            return result
        finally:
            self._active = None

    def _validate_live_schedule(self) -> None:
        if self.schedule_audit is not None:
            return
        try:
            audit = sigma_strata.audit_runtime_unipc_schedule(
                self.scheduler, initialize=False
            )
        except sigma_strata.InferenceSigmaStrataError as error:
            raise PairV7VOnlyRouteRuntimeError(
                f"live exact40 shift-5 schedule differs: {error}"
            ) from error
        if audit.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256:
            raise PairV7VOnlyRouteRuntimeError("exact40 schedule digest differs")
        self.schedule_audit = dict(audit)

    def _schedule_index(self, timestep: Any, *, expected: int) -> int:
        observed = _exact_int_scalar(timestep, label="model timestep")
        matches = [
            index
            for index, registered in enumerate(sigma_strata.PINNED_TIMESTEPS)
            if registered == observed
        ]
        if len(matches) != 1:
            raise PairV7VOnlyRouteRuntimeError(
                "model timestep is outside the registered exact40 schedule"
            )
        index = matches[0]
        if index != expected:
            raise PairV7VOnlyRouteRuntimeError(
                f"exact40 call order differs: expected index {expected}, got {index}"
            )
        return index

    def _validate_shared_call(
        self,
        values: Mapping[str, Any],
        *,
        state: _ActiveSample,
        role: str,
        schedule_index: int,
    ) -> _ForwardObservation:
        if values.get("model_id") != EXPECTED_MODEL_ID:
            raise PairV7VOnlyRouteRuntimeError("shared_step model route differs")
        prompt = state.negative_prompt if role == "negative" else state.action_prompt
        _same_object(values.get("cond_embeds"), prompt, label=f"{role} prompt")
        noisy = values.get("noisy_latents")
        timestep = values.get("timesteps")
        rotary = values.get("rotary_embs")
        _detached_tensor(noisy, label=f"{role} V pack")
        _detached_tensor(timestep, label=f"{role} timestep")
        _detached_tensor(rotary, label=f"{role} rotary")
        if _shape(noisy, label=f"{role} V pack") != (
            1,
            self.config.total_v_tokens,
            self.config.expected_hidden_dim,
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "shared_step is not the exact source-prefix + target-suffix V pack"
            )
        if _shape(timestep, label=f"{role} timestep") != (1,):
            raise PairV7VOnlyRouteRuntimeError(
                "shared_step timestep must be one expanded scalar"
            )
        self._schedule_index(timestep, expected=schedule_index)
        rotary_shape = _shape(rotary, label=f"{role} rotary")
        if (
            len(rotary_shape) != 4
            or rotary_shape[0] != 1
            or rotary_shape[1] != 1
            or rotary_shape[2] != self.config.total_v_tokens
            or rotary_shape[3] <= 0
        ):
            raise PairV7VOnlyRouteRuntimeError("V-only rotary geometry differs")
        vae_length = _metadata_tuple(
            values.get("batch_vae_seqlen"), label=f"{role} batch_vae_seqlen"
        )
        if vae_length != (self.config.total_v_tokens,):
            raise PairV7VOnlyRouteRuntimeError(
                "V-only batch_vae_seqlen differs from the two-video pack"
            )
        if _metadata_tuple(
            values.get("batch_text_seqlen"), label=f"{role} batch_text_seqlen"
        ) != (self.config.expected_text_tokens,):
            raise PairV7VOnlyRouteRuntimeError("shared_step text length differs")
        if state.pending:
            first = state.pending[0]
            for label, left, right in (
                ("noisy_latents", first.noisy_latents, noisy),
                ("timesteps", first.timesteps, timestep),
                ("rotary_embs", first.rotary_embs, rotary),
            ):
                _same_object(left, right, label=f"negative/action {label}")
            if first.batch_vae_seqlen != vae_length:
                raise PairV7VOnlyRouteRuntimeError(
                    "negative/action V-pack metadata differ"
                )
        return _ForwardObservation(
            role=role,
            schedule_index=schedule_index,
            noisy_latents=noisy,
            timesteps=timestep,
            rotary_embs=rotary,
            batch_vae_seqlen=vae_length,
        )

    def _build_route(self, schedule_index: int, *, device: Any) -> tuple[Any, str, float]:
        try:
            gate_name, gate_weight = self.action_module.sigma_gate(schedule_index)
            route = self.action_module.PairV5ActionRoute(
                total_tokens=self.config.total_v_tokens,
                condition_tokens=self.config.target_tokens,
                sequence_parallel_rank=(
                    self.config.parallel.sequence_parallel_rank
                ),
                sequence_parallel_size=(
                    self.config.parallel.sequence_parallel_size
                ),
                branch_name="V",
                sigma_schedule_index=schedule_index,
                enabled=True,
            )
        except Exception as error:
            raise PairV7VOnlyRouteRuntimeError(
                f"cannot build registered Action-LoRA route: {error}"
            ) from error
        if (
            getattr(route, "branch_name", None) != "V"
            or getattr(route, "total_tokens", None) != self.config.total_v_tokens
            or getattr(route, "condition_tokens", None) != self.config.target_tokens
            or getattr(route, "sigma_schedule_index", None) != schedule_index
            or getattr(route, "gate_name", None) != gate_name
            or float(getattr(route, "gate_weight", math.nan)) != float(gate_weight)
            or bool(getattr(route, "adapter_active", False))
            is not (schedule_index < 38)
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "constructed Action-LoRA route differs from registration"
            )
        try:
            selector = _selector_values(route.local_target_selector(device=device))
        except PairV7VOnlyRouteRuntimeError:
            raise
        except Exception as error:
            raise PairV7VOnlyRouteRuntimeError(
                "cannot evaluate Action-LoRA local selector"
            ) from error
        local_length = math.ceil(
            self.config.total_v_tokens
            / self.config.parallel.sequence_parallel_size
        )
        global_selector = (
            [False] * self.config.target_tokens
            + [True] * self.config.target_tokens
        )
        global_selector.extend(
            [False]
            * (local_length * self.config.parallel.sequence_parallel_size
               - self.config.total_v_tokens)
        )
        start = self.config.parallel.sequence_parallel_rank * local_length
        expected_selector = tuple(global_selector[start : start + local_length])
        if selector != expected_selector:
            raise PairV7VOnlyRouteRuntimeError(
                "Action-LoRA selector is not the V-pack target suffix on this SP rank"
            )
        return route, str(gate_name), float(gate_weight)

    def _validate_shared_result(self, result: Any) -> None:
        _detached_tensor(result, label="official shared_step result")
        shape = _shape(result, label="official shared_step result")
        if (
            len(shape) != 3
            or shape[0] != 1
            or shape[1] != self.config.total_v_tokens
            or shape[2] != 64
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "official shared_step result does not preserve the V pack"
            )

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise PairV7VOnlyRouteRuntimeError(
                "shared_step ran outside the authenticated sample"
            )
        self._validate_live_schedule()
        if len(state.pending) >= FORWARDS_PER_CELL:
            raise PairV7VOnlyRouteRuntimeError(
                "more than two official forwards occurred before scheduler.step"
            )
        schedule_index = state.completed_cells
        role = FORWARD_ORDER[len(state.pending)]
        values = _flatten_bound_arguments(self.original_shared_step, args, kwargs)
        observation = self._validate_shared_call(
            values,
            state=state,
            role=role,
            schedule_index=schedule_index,
        )
        route, _, _ = self._build_route(
            schedule_index, device=values["noisy_latents"].device
        )
        if self.action_module.active_route() is not None:
            raise PairV7VOnlyRouteRuntimeError(
                "unexpected nested Action-LoRA route before official forward"
            )
        with self.action_handle.route(route):
            if self.action_module.active_route() is not route:
                raise PairV7VOnlyRouteRuntimeError(
                    "Action-LoRA handle did not activate the exact route object"
                )
            result = self.original_shared_step(*args, **kwargs)
        if self.action_module.active_route() is not None:
            raise PairV7VOnlyRouteRuntimeError(
                "Action-LoRA route leaked after official forward"
            )
        self._validate_shared_result(result)
        self.original_shared_step_calls += 1
        state.pending.append(observation)
        return result

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise PairV7VOnlyRouteRuntimeError(
                "scheduler.step ran outside the authenticated sample"
            )
        self._validate_live_schedule()
        if (
            len(state.pending) != FORWARDS_PER_CELL
            or tuple(row.role for row in state.pending) != FORWARD_ORDER
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "scheduler.step arrived before one official negative/action pair"
            )
        index = state.completed_cells
        timestep = _extract_argument(args, kwargs, index=1, name="timestep")
        self._schedule_index(timestep, expected=index)
        shared_timestep = state.pending[0].timesteps
        if _exact_int_scalar(shared_timestep, label="shared timestep") != _exact_int_scalar(
            timestep, label="scheduler timestep"
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "shared_step and scheduler.step timesteps differ"
            )
        live_index = getattr(self.scheduler, "step_index", None)
        if live_index is None:
            live_index = getattr(self.scheduler, "_step_index", None)
        if live_index is not None and _exact_int_scalar(
            live_index, label="scheduler step_index"
        ) != index:
            raise PairV7VOnlyRouteRuntimeError(
                "scheduler live index differs from exact40 call order"
            )
        result = self.original_scheduler_step(*args, **kwargs)
        self.original_scheduler_step_calls += 1
        gate_name, gate_weight = self.action_module.sigma_gate(index)
        self.trace.append(
            {
                "schema_version": SCHEMA_VERSION,
                "schedule_index": index,
                "timestep": sigma_strata.PINNED_TIMESTEPS[index],
                "sigma_float32_be_hex": (
                    sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
                ),
                "forward_order": list(FORWARD_ORDER),
                "official_shared_step_calls": FORWARDS_PER_CELL,
                "original_scheduler_step_calls": 1,
                "branch_name": "V",
                "condition_tokens": self.config.target_tokens,
                "target_tokens": self.config.target_tokens,
                "total_tokens": self.config.total_v_tokens,
                "gate_name": str(gate_name),
                "gate_weight": float(gate_weight),
                "adapter_active": index < 38,
                "low_sigma_direct_base": index >= 38,
                "parallel": self.config.parallel.as_dict(),
                "official_arguments_mutated": False,
            }
        )
        state.pending.clear()
        state.completed_cells += 1
        return result

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise PairV7VOnlyRouteRuntimeError("PAIR-v7 route finalize lifecycle differs")
        if (
            self.schedule_audit is None
            or self.original_sample_invocations != 1
            or self.successful_sample_calls != 1
            or self.original_shared_step_calls != EXPECTED_STEPS * FORWARDS_PER_CELL
            or self.original_scheduler_step_calls != EXPECTED_STEPS
            or len(self.trace) != EXPECTED_STEPS
            or [row["schedule_index"] for row in self.trace]
            != list(range(EXPECTED_STEPS))
            or self.input_binding is None
            or self.sample_contract is None
            or self.output_latent_sha256 is None
        ):
            raise PairV7VOnlyRouteRuntimeError(
                "PAIR-v7 exact40 route call-count certificate differs"
            )
        expected_gates = [
            self.action_module.sigma_gate(index) for index in range(EXPECTED_STEPS)
        ]
        if [
            (row["gate_name"], row["gate_weight"]) for row in self.trace
        ] != [(str(name), float(weight)) for name, weight in expected_gates]:
            raise PairV7VOnlyRouteRuntimeError(
                "PAIR-v7 exact40 registered gate trace differs"
            )
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "official_sample_calls": 1,
            "exact81": True,
            "exact40": True,
            "source_video_only_V_pack": True,
            "official_shared_step_calls": self.original_shared_step_calls,
            "official_shared_steps_per_cell": FORWARDS_PER_CELL,
            "official_scheduler_step_calls": self.original_scheduler_step_calls,
            "official_sampler_arguments_mutated": False,
            "route_branch": "V",
            "parallel": self.config.parallel.as_dict(),
            "target_latent_shape": list(self.config.target_latent_shape),
            "registered_inputs": dict(self.input_binding),
            "registered_sample_contract": dict(self.sample_contract),
            "registered_sample_contract_sha256": _object_sha256(
                self.sample_contract
            ),
            "official_output_latent_sha256": self.output_latent_sha256,
            "switch_dit_boundary": 0.0,
            "live_distributed_route_audit": dict(self.distributed_audit),
            "condition_tokens": self.config.target_tokens,
            "target_tokens": self.config.target_tokens,
            "total_tokens": self.config.total_v_tokens,
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "active_schedule_indices": list(range(38)),
            "direct_base_schedule_indices": [38, 39],
            "registered_gate_weights": [row["gate_weight"] for row in self.trace],
            "image_reference_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "extra_visual_branch_consumed": False,
            "optimizer_created": False,
            "parameters_updated": False,
            "semantic_action_editing_claim": False,
            "trace": list(self.trace),
        }
        self.finalized = True
        return {**unsigned, "receipt_digest": _object_sha256(unsigned)}


@contextmanager
def pair_v7_vonly_exact81_route_hook(
    renderer_or_diffusion: Any,
    *,
    action_handle: Any,
    config: PairV7VOnlyExact81RouteConfig,
) -> Iterator[PairV7VOnlyExact81RoutePatch]:
    """Install the route wrappers and always restore them in ``finally``."""

    patch = PairV7VOnlyExact81RoutePatch(
        renderer_or_diffusion,
        action_handle=action_handle,
        config=config,
    )
    patch.install()
    try:
        yield patch
    finally:
        patch.restore()


__all__ = [
    "EXPECTED_DP_SIZE",
    "EXPECTED_FLOW_SHIFT",
    "EXPECTED_FRAMES",
    "EXPECTED_GUIDANCE_MODE",
    "EXPECTED_LATENT_PHASES",
    "EXPECTED_SP_SIZE",
    "EXPECTED_STEPS",
    "EXPECTED_WORLD_SIZE",
    "FORWARD_ORDER",
    "FORWARDS_PER_CELL",
    "PairV7DPSPRouteMetadata",
    "PairV7VOnlyExact81RouteConfig",
    "PairV7VOnlyExact81RoutePatch",
    "PairV7VOnlyRouteRuntimeError",
    "SCHEMA_VERSION",
    "pair_v7_vonly_exact81_route_hook",
]
