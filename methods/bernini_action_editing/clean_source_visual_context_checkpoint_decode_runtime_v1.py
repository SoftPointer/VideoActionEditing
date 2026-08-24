#!/usr/bin/env python3
"""Branch-aware inference routing for clean-source visual-context checkpoints.

The Stage-B training adapter is attached to frozen Bernini transformer blocks.
Those block hooks fail closed unless every transformer forward carries an
authenticated :class:`VisualContextRoute`.  A native Bernini RV2V denoising
step does *not* have one static packed layout: it executes target-only, V,
VI-unconditional and VI-conditional forwards with different global sequence
lengths.  Therefore wrapping ``diffusion.sample`` in one route is invalid.

This module installs a reversible instance wrapper on ``shared_step`` and
binds a fresh route around each of the four native forwards.  It also provides
the live official-Gaussian capture needed by the registered
``same_noise_forward_noised_source`` memory arm.  It neither changes native
guidance nor patches the scheduler, and contains no optimizer, reward, scorer,
ranking or selection path.

The module is a runtime primitive, not a decoded-quality claim.  A full decode
runner must still load the pinned renderer, VAE, checkpoint and fixed sentinel
registry and must validate the trace returned here before publishing media.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import clean_source_visual_context_adapter_v1 as visual


SCHEMA_VERSION = "bernini-clean-source-visual-context-decode-route-v1"
FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
NATIVE_BRANCH_ORDER = (
    "none_uncond",
    "V_uncond",
    "VI_uncond",
    "VI_cond",
)
SOURCE_CONTROL_ARMS = (
    "correct",
    "carrier-off",
    "wrong-owner",
    "order-permutation",
)
MEMORY_TRANSFORMS = (
    "identity",
    "reverse-phase-order-20-to-0",
)


class CleanSourceVisualContextDecodeError(RuntimeError):
    """Raised before an unaudited route reaches a transformer forward."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CleanSourceVisualContextDecodeError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CleanSourceVisualContextDecodeError(f"{label} must be lowercase SHA-256")
    return value


def _scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise CleanSourceVisualContextDecodeError(f"{label} is not scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        result = float(candidate)
    except CleanSourceVisualContextDecodeError:
        raise
    except Exception as error:
        raise CleanSourceVisualContextDecodeError(f"{label} is not scalar") from error
    if not math.isfinite(result):
        raise CleanSourceVisualContextDecodeError(f"{label} is non-finite")
    return result


def _bind_call(function: Callable[..., Any], args: Sequence[Any], kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        bound = inspect.signature(function).bind(*args, **dict(kwargs))
    except (TypeError, ValueError) as error:
        raise CleanSourceVisualContextDecodeError(
            "pinned shared_step signature or invocation differs"
        ) from error
    bound.apply_defaults()
    return dict(bound.arguments)


def _tensor_sha256(value: Any, *, label: str) -> str:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH supplies torch
        raise CleanSourceVisualContextDecodeError("tensor hashing requires torch") from error
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise CleanSourceVisualContextDecodeError(f"{label} must be a tensor")
    raw = value.detach().contiguous().view(torch.uint8).cpu().reshape(-1)
    try:
        payload = raw.numpy().tobytes()
    except RuntimeError as error:
        if "Numpy is not available" not in str(error):
            raise
        payload = bytes(raw.tolist())
    return hashlib.sha256(payload).hexdigest()


def reverse_latent_phase_order(clean_source_latent: Any) -> Any:
    """Return the registered order control: exact phase reversal 20..0."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise CleanSourceVisualContextDecodeError("phase permutation requires torch") from error
    if (
        not isinstance(clean_source_latent, torch.Tensor)
        or clean_source_latent.ndim != 5
        or tuple(int(value) for value in clean_source_latent.shape[:3])
        != (1, 16, LATENT_PHASES)
        or clean_source_latent.dtype != torch.float32
        or clean_source_latent.requires_grad
        or not clean_source_latent.is_contiguous()
        or not bool(torch.isfinite(clean_source_latent).all().item())
    ):
        raise CleanSourceVisualContextDecodeError(
            "order control requires detached contiguous FP32 [1,16,21,H,W]"
        )
    result = clean_source_latent.flip((2,)).contiguous()
    if result.data_ptr() == clean_source_latent.data_ptr() or tuple(result.shape) != tuple(
        clean_source_latent.shape
    ):
        raise CleanSourceVisualContextDecodeError("phase reversal did not create one exact peer")
    return result


class VisualMemoryProvider:
    """Build one checkpoint memory, with optional live-noise conditioning."""

    def __init__(
        self,
        *,
        handle: visual.CleanSourceVisualContextHandle,
        source_latent: Any,
        source_video_sha256: str,
        memory_input_kind: str,
        scheduler: Any,
        memory_transform: str = "identity",
        tensor_sha256: Callable[..., str] = _tensor_sha256,
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise CleanSourceVisualContextDecodeError("memory provider requires torch") from error
        if memory_input_kind not in visual.MEMORY_INPUT_KINDS:
            raise CleanSourceVisualContextDecodeError("memory input kind differs")
        if memory_transform not in MEMORY_TRANSFORMS:
            raise CleanSourceVisualContextDecodeError("memory transform differs")
        if (
            not isinstance(source_latent, torch.Tensor)
            or source_latent.ndim != 5
            or tuple(int(value) for value in source_latent.shape[:3])
            != (1, 16, LATENT_PHASES)
            or source_latent.dtype != torch.float32
            or source_latent.requires_grad
            or not source_latent.is_contiguous()
            or not bool(torch.isfinite(source_latent).all().item())
        ):
            raise CleanSourceVisualContextDecodeError(
                "memory source must be detached contiguous finite FP32 exact81 latent"
            )
        self.handle = handle
        self.source_latent = source_latent
        self.source_video_sha256 = _sha256(
            source_video_sha256, label="memory source video SHA-256"
        )
        self.memory_input_kind = memory_input_kind
        self.memory_transform = memory_transform
        self.scheduler = scheduler
        self.tensor_sha256 = tensor_sha256
        self._official_epsilon: Optional[Any] = None
        self._cache_key: Optional[tuple[float, float]] = None
        self._cache: Optional[visual.CleanSourceVisualMemory] = None
        self._build_count = 0

    def bind_official_initial_gaussian(self, epsilon: Any) -> None:
        """Bind the exact tensor returned by Bernini's official randn call."""

        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise CleanSourceVisualContextDecodeError("noise binding requires torch") from error
        if self._official_epsilon is not None:
            raise CleanSourceVisualContextDecodeError("official Gaussian was bound twice")
        if (
            not isinstance(epsilon, torch.Tensor)
            or epsilon is self.source_latent
            or tuple(epsilon.shape) != tuple(self.source_latent.shape)
            or epsilon.device != self.source_latent.device
            or epsilon.dtype != torch.float32
            or epsilon.requires_grad
            or not epsilon.is_contiguous()
            or not bool(torch.isfinite(epsilon).all().item())
        ):
            raise CleanSourceVisualContextDecodeError(
                "official Gaussian differs from exact source latent geometry"
            )
        # Keep the exact live tensor.  No clone or RNG replay is accepted.
        self._official_epsilon = epsilon

    @property
    def official_initial_gaussian_sha256(self) -> Optional[str]:
        if self._official_epsilon is None:
            return None
        return self.tensor_sha256(self._official_epsilon, label="official initial Gaussian")

    @property
    def build_count(self) -> int:
        return self._build_count

    def _resolve_sigma(self, timestep: Any) -> float:
        # Keep this dependency lazy so CPU unit tests can supply a tiny pinned
        # scheduler double without importing the Bernini inference stack.
        if hasattr(self.scheduler, "resolve_sigma"):
            sigma = self.scheduler.resolve_sigma(timestep)
            return _scalar(sigma, label="resolved sigma")
        try:
            import tri_branch_unipc as sampler_contract

            _, _, sigma = sampler_contract._resolve_sigma(self.scheduler, timestep)
        except Exception as error:
            raise CleanSourceVisualContextDecodeError(
                "cannot resolve the native UniPC sigma for visual memory"
            ) from error
        return _scalar(sigma, label="resolved sigma")

    def __call__(self, timestep: Any) -> visual.CleanSourceVisualMemory:
        timestep_float = _scalar(timestep, label="shared_step timestep")
        if self.memory_input_kind == "clean_source":
            sigma = -1.0
            memory_input = self.source_latent
            # Clean source memory is independent of the denoising coordinate.
            # Reusing its exact tensor object is both cheaper and a useful
            # trace invariant.  The noised arm below remains step-specific.
            cache_timestep = -1.0
        else:
            if self._official_epsilon is None:
                raise CleanSourceVisualContextDecodeError(
                    "same-noise memory requested before the official Gaussian was observed"
                )
            sigma = self._resolve_sigma(timestep)
            if not 0.0 <= sigma <= 1.0:
                raise CleanSourceVisualContextDecodeError("native sigma lies outside [0,1]")
            memory_input = (
                (1.0 - sigma) * self.source_latent + sigma * self._official_epsilon
            ).float().contiguous()
            cache_timestep = timestep_float
        key = (cache_timestep, sigma)
        if self._cache_key == key:
            if self._cache is None:
                raise CleanSourceVisualContextDecodeError("memory cache state differs")
            return self._cache
        digest = self.tensor_sha256(memory_input, label="visual memory input latent")
        memory = self.handle.build_memory(
            memory_input,
            source_video_sha256=self.source_video_sha256,
            memory_input_latent_sha256=digest,
            input_kind=self.memory_input_kind,
        )
        self._build_count += 1
        self._cache_key = key
        self._cache = memory
        return memory


@dataclass(frozen=True)
class _ObservedCall:
    branch: str
    step_index: int
    timestep: float
    total_tokens: int
    condition_tokens: int
    target_tokens: int
    route_enabled: bool
    memory_source_video_sha256: Optional[str]
    memory_construction_digest: Optional[str]

    def receipt(self) -> Mapping[str, Any]:
        return {
            "branch": self.branch,
            "step_index": self.step_index,
            "timestep": self.timestep,
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "route_enabled": self.route_enabled,
            "memory_source_video_sha256": self.memory_source_video_sha256,
            "memory_construction_digest": self.memory_construction_digest,
        }


class BranchAwareVisualContextRouteHook:
    """Bind one authenticated visual route around each native shared_step."""

    def __init__(
        self,
        diffusion: Any,
        *,
        handle: visual.CleanSourceVisualContextHandle,
        target_tokens: int,
        sequence_parallel_rank: int,
        sequence_parallel_size: int,
        source_control_arm: str,
        target_source_video_sha256: str,
        memory_provider: Optional[VisualMemoryProvider],
        expected_steps: int = NUM_INFERENCE_STEPS,
    ) -> None:
        if source_control_arm not in SOURCE_CONTROL_ARMS:
            raise CleanSourceVisualContextDecodeError("source-control arm differs")
        if expected_steps != NUM_INFERENCE_STEPS:
            raise CleanSourceVisualContextDecodeError("decode route is fixed to exact40")
        if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens <= 0:
            raise CleanSourceVisualContextDecodeError("target_tokens must be positive")
        if sequence_parallel_size != 4 or not 0 <= sequence_parallel_rank < 4:
            raise CleanSourceVisualContextDecodeError("decode route requires WORLD4/SP4")
        if source_control_arm == "carrier-off":
            if memory_provider is not None:
                raise CleanSourceVisualContextDecodeError("carrier-off cannot carry memory")
        elif not isinstance(memory_provider, VisualMemoryProvider):
            raise CleanSourceVisualContextDecodeError("enabled source-control arm requires memory")
        target_source_sha = _sha256(
            target_source_video_sha256, label="target source video SHA-256"
        )
        if source_control_arm == "correct":
            if (
                memory_provider is None
                or memory_provider.source_video_sha256 != target_source_sha
                or memory_provider.memory_transform != "identity"
            ):
                raise CleanSourceVisualContextDecodeError(
                    "correct carrier must use the target owner's unpermuted source"
                )
        elif source_control_arm == "wrong-owner":
            if (
                memory_provider is None
                or memory_provider.source_video_sha256 == target_source_sha
                or memory_provider.memory_transform != "identity"
            ):
                raise CleanSourceVisualContextDecodeError(
                    "wrong-owner carrier must use a distinct unpermuted owner"
                )
        elif source_control_arm == "order-permutation":
            if (
                memory_provider is None
                or memory_provider.source_video_sha256 != target_source_sha
                or memory_provider.memory_transform != "reverse-phase-order-20-to-0"
            ):
                raise CleanSourceVisualContextDecodeError(
                    "order control must reverse phases of the target owner's source"
                )
        shared = getattr(diffusion, "shared_step", None)
        if not callable(shared) or "shared_step" in vars(diffusion):
            raise CleanSourceVisualContextDecodeError(
                "shared_step must be the unwrapped pinned class method"
            )
        self.diffusion = diffusion
        self.handle = handle
        self.target_tokens = target_tokens
        self.sequence_parallel_rank = sequence_parallel_rank
        self.sequence_parallel_size = sequence_parallel_size
        self.source_control_arm = source_control_arm
        self.target_source_video_sha256 = target_source_sha
        self.memory_provider = memory_provider
        self.expected_steps = expected_steps
        self._original_shared = shared
        self._installed = False
        self._active = False
        self._calls: list[_ObservedCall] = []
        self.sample_calls = 0
        self.trace: Mapping[str, Any] = {}

    def install(self) -> None:
        if self._installed or "shared_step" in vars(self.diffusion):
            raise CleanSourceVisualContextDecodeError("decode route hook is already installed")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared(*args, **kwargs)

        setattr(wrapper, "_clean_source_visual_context_decode_route_v1", self)
        setattr(self.diffusion, "shared_step", wrapper)
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            raise CleanSourceVisualContextDecodeError("decode route hook is not installed")
        current = vars(self.diffusion).get("shared_step")
        if getattr(current, "_clean_source_visual_context_decode_route_v1", None) is not self:
            raise CleanSourceVisualContextDecodeError("shared_step wrapper changed while installed")
        delattr(self.diffusion, "shared_step")
        self._installed = False
        self._active = False

    @contextmanager
    def sample(self) -> Iterator[None]:
        if not self._installed or self._active or self.sample_calls:
            raise CleanSourceVisualContextDecodeError(
                "route hook permits exactly one non-nested native sample"
            )
        self._active = True
        self._calls.clear()
        try:
            yield
            expected = self.expected_steps * len(NATIVE_BRANCH_ORDER)
            if len(self._calls) != expected:
                raise CleanSourceVisualContextDecodeError(
                    f"native sample made {len(self._calls)} shared calls, expected {expected}"
                )
            for step_index in range(self.expected_steps):
                group = self._calls[
                    step_index * len(NATIVE_BRANCH_ORDER) :
                    (step_index + 1) * len(NATIVE_BRANCH_ORDER)
                ]
                if (
                    tuple(item.branch for item in group) != NATIVE_BRANCH_ORDER
                    or any(item.step_index != step_index for item in group)
                    or len({item.timestep for item in group}) != 1
                    or group[0].condition_tokens != 0
                    or any(item.condition_tokens <= 0 for item in group[1:])
                ):
                    raise CleanSourceVisualContextDecodeError(
                        f"native branch/layout closure differs at step {step_index}"
                    )
            expected_memory_builds = (
                0
                if self.memory_provider is None
                else (
                    1
                    if self.memory_provider.memory_input_kind == "clean_source"
                    else self.expected_steps
                )
            )
            if (
                self.memory_provider is not None
                and self.memory_provider.build_count != expected_memory_builds
            ):
                raise CleanSourceVisualContextDecodeError(
                    "visual memory was not built at the registered persistence cadence"
                )
            unsigned = {
                "schema_version": SCHEMA_VERSION,
                "source_control_arm": self.source_control_arm,
                "target_source_video_sha256": self.target_source_video_sha256,
                "memory_source_video_sha256": (
                    None
                    if self.memory_provider is None
                    else self.memory_provider.source_video_sha256
                ),
                "memory_transform": (
                    None
                    if self.memory_provider is None
                    else self.memory_provider.memory_transform
                ),
                "memory_input_kind": (
                    None
                    if self.memory_provider is None
                    else self.memory_provider.memory_input_kind
                ),
                "exact40": True,
                "step_count": self.expected_steps,
                "shared_step_call_count": expected,
                "native_branch_order": list(NATIVE_BRANCH_ORDER),
                "target_tokens": self.target_tokens,
                "sequence_parallel_size": self.sequence_parallel_size,
                "sequence_parallel_rank": self.sequence_parallel_rank,
                "shared_step_only_wrapped": True,
                "native_guidance_changed": False,
                "scheduler_changed": False,
                "optimizer_present": False,
                "memory_build_count": expected_memory_builds,
                "calls": [item.receipt() for item in self._calls],
            }
            self.trace = {**unsigned, "trace_digest": object_sha256(unsigned)}
            self.sample_calls = 1
        finally:
            self._active = False

    def _wrapped_shared(self, *args: Any, **kwargs: Any) -> Any:
        if not self._active:
            raise CleanSourceVisualContextDecodeError(
                "shared_step ran outside one authenticated sample"
            )
        values = _bind_call(self._original_shared, args, kwargs)
        call_index = len(self._calls)
        step_index, branch_index = divmod(call_index, len(NATIVE_BRANCH_ORDER))
        if step_index >= self.expected_steps:
            raise CleanSourceVisualContextDecodeError("native sample exceeded exact40")
        branch = NATIVE_BRANCH_ORDER[branch_index]
        lengths = values.get("batch_vae_seqlen")
        if (
            values.get("model_id") != "transformer_1"
            or not isinstance(lengths, (list, tuple))
            or len(lengths) != 1
            or isinstance(lengths[0], bool)
        ):
            raise CleanSourceVisualContextDecodeError("shared_step branch binding differs")
        total_tokens = int(lengths[0])
        if total_tokens < self.target_tokens:
            raise CleanSourceVisualContextDecodeError(
                "packed sequence is shorter than the exact target suffix"
            )
        condition_tokens = total_tokens - self.target_tokens
        if (branch == "none_uncond") is not (condition_tokens == 0):
            raise CleanSourceVisualContextDecodeError(
                "target-only/native visual branch order differs"
            )
        timestep = values.get("timesteps")
        timestep_float = _scalar(timestep, label="shared_step timestep")
        if self.source_control_arm == "carrier-off":
            memory = None
            enabled = False
        else:
            assert self.memory_provider is not None
            memory = self.memory_provider(timestep)
            enabled = True
        route = visual.VisualContextRoute(
            total_tokens=total_tokens,
            condition_tokens=condition_tokens,
            sequence_parallel_rank=self.sequence_parallel_rank,
            sequence_parallel_size=self.sequence_parallel_size,
            memory=memory,
            enabled=enabled,
        )
        with self.handle.route(route):
            result = self._original_shared(*args, **kwargs)
        self._calls.append(
            _ObservedCall(
                branch=branch,
                step_index=step_index,
                timestep=timestep_float,
                total_tokens=total_tokens,
                condition_tokens=condition_tokens,
                target_tokens=self.target_tokens,
                route_enabled=enabled,
                memory_source_video_sha256=(
                    None if memory is None else memory.source_video_sha256
                ),
                memory_construction_digest=(
                    None if memory is None else memory.construction_digest
                ),
            )
        )
        return result


@contextmanager
def observe_official_initial_gaussian(
    wan_diffusion_module: Any,
    *,
    expected_shape: Sequence[int],
    expected_device: Any,
    expected_seed: int,
    on_tensor: Callable[[Any], None],
) -> Iterator[Mapping[str, Any]]:
    """Observe the one official Bernini Gaussian and expose its live tensor.

    The original callable sees the exact original arguments and its return
    object is forwarded unchanged.  ``on_tensor`` runs synchronously before
    the first transformer forward, enabling same-noise visual memory without
    RNG replay or a custom initial-noise injection.
    """

    try:
        import torch
        from diffusers.utils.torch_utils import randn_tensor as canonical
    except ImportError as error:  # pragma: no cover
        raise CleanSourceVisualContextDecodeError(
            "official Gaussian observation requires torch and diffusers"
        ) from error
    if not callable(on_tensor):
        raise CleanSourceVisualContextDecodeError("Gaussian callback must be callable")
    expected = tuple(int(value) for value in expected_shape)
    if (
        expected[:3] != (1, 16, LATENT_PHASES)
        or len(expected) != 5
        or type(expected_seed) is not int
        or not 0 <= expected_seed < 2**63
    ):
        raise CleanSourceVisualContextDecodeError("official Gaussian contract differs")
    original = getattr(wan_diffusion_module, "randn_tensor", None)
    if original is not canonical:
        raise CleanSourceVisualContextDecodeError(
            "wan_diffusion.randn_tensor is already wrapped or differs"
        )
    record: dict[str, Any] = {"call_count": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if record["call_count"]:
            raise CleanSourceVisualContextDecodeError("official randn_tensor ran twice")
        shape_value = args[0] if args else kwargs.get("shape")
        try:
            shape = tuple(int(value) for value in shape_value)
        except Exception as error:
            raise CleanSourceVisualContextDecodeError("randn shape differs") from error
        generator = kwargs.get("generator")
        result = original(*args, **kwargs)
        if (
            shape != expected
            or not isinstance(generator, torch.Generator)
            or str(generator.device) != "cpu"
            or int(generator.initial_seed()) != expected_seed
            or not isinstance(result, torch.Tensor)
            or tuple(result.shape) != expected
            or result.device != torch.device(expected_device)
            or result.dtype != torch.float32
            or result.requires_grad
            or not result.is_contiguous()
        ):
            raise CleanSourceVisualContextDecodeError(
                "official initial Gaussian identity/geometry differs"
            )
        on_tensor(result)
        record.update(
            {
                "call_count": 1,
                "seed": expected_seed,
                "shape": list(expected),
                "dtype": str(result.dtype),
                "device": str(result.device),
                "raw_sha256": _tensor_sha256(result, label="official initial Gaussian"),
                "same_live_tensor_forwarded": True,
            }
        )
        return result

    setattr(wan_diffusion_module, "randn_tensor", wrapper)
    try:
        yield record
    finally:
        unchanged = getattr(wan_diffusion_module, "randn_tensor", None) is wrapper
        setattr(wan_diffusion_module, "randn_tensor", original)
        if not unchanged or getattr(wan_diffusion_module, "randn_tensor", None) is not original:
            raise CleanSourceVisualContextDecodeError(
                "official randn_tensor wrapper restoration failed"
            )
    if record.get("call_count") != 1:
        raise CleanSourceVisualContextDecodeError(
            "native sample did not make exactly one official randn_tensor call"
        )


__all__ = [
    "BranchAwareVisualContextRouteHook",
    "CleanSourceVisualContextDecodeError",
    "FRAME_COUNT",
    "LATENT_PHASES",
    "NATIVE_BRANCH_ORDER",
    "NUM_INFERENCE_STEPS",
    "SCHEMA_VERSION",
    "SOURCE_CONTROL_ARMS",
    "MEMORY_TRANSFORMS",
    "VisualMemoryProvider",
    "canonical_json_bytes",
    "object_sha256",
    "observe_official_initial_gaussian",
    "reverse_latent_phase_order",
]
