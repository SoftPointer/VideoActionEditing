"""Read-only internal observer for the frozen Bernini FITQ scan.

The official Bernini-R 1.3B transformer has no temporal-only block.  This
module therefore observes the joint spatiotemporal stream at stable official
``nn.Module`` boundaries without replacing an attention processor:

* every block's self-attention ``attn1.to_out[0]`` input;
* every block's text cross-attention ``attn2.to_out[0]`` input;
* every block input and output; and
* the final ``proj_out`` input.

Hooks never return a tensor, run a collective, or retain an autograd edge.
They only form additive FP32 phase/head sufficient statistics on the local
Ulysses shard.  Distributed reduction is deliberately a separate function
that must be called after the official forward has completed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Optional, Sequence
import weakref


try:  # Keep contract/source tests importable in lightweight environments.
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]


EXPECTED_BLOCK_COUNT = 30
EXPECTED_HEAD_COUNT = 12
EXPECTED_HEAD_DIM = 128
EXPECTED_HIDDEN_DIM = EXPECTED_HEAD_COUNT * EXPECTED_HEAD_DIM
EXPECTED_SP_WORLD = 4

EXPECTED_PHASE_COUNT = 21
EXPECTED_PATCH_HEIGHT = 31
EXPECTED_PATCH_WIDTH = 30
EXPECTED_TOKENS_PER_PHASE = EXPECTED_PATCH_HEIGHT * EXPECTED_PATCH_WIDTH
EXPECTED_GLOBAL_TARGET_TOKENS = EXPECTED_PHASE_COUNT * EXPECTED_TOKENS_PER_PHASE

VALID_MODES = ("t2v", "mv2v")
EXACT_PROBE_SITES = ("block.00.input", "block.00.attn1")


class InternalTemporalQuotientObserverError(RuntimeError):
    """Raised when a pinned FITQ observer contract is violated."""


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on deployment environment
        raise InternalTemporalQuotientObserverError("FITQ observer requires torch")
    return torch


def _tensor_version(tensor: Any) -> int:
    """Return a cache/mutation token without rejecting inference tensors.

    PyTorch inference tensors deliberately do not expose ``_version``.  The
    pinned Bernini scan runs under ``torch.inference_mode()``.  Such tensors can
    still be mutated *inside* that context, so ``-1`` explicitly means that no
    safe cache token is available and value-derived statistics must be
    recomputed on every observation.
    """

    try:
        return int(tensor._version)
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        return -1


def _finite_real(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise InternalTemporalQuotientObserverError(f"{name} must be a finite real")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise InternalTemporalQuotientObserverError(
            f"{name} must be a finite real"
        ) from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise InternalTemporalQuotientObserverError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class FITQObserverContext:
    """Explicit identity of one frozen official-model forward.

    ``lambda_value`` is serialized under the literal key ``lambda``.  Python
    cannot use ``lambda`` as a field/keyword name.
    """

    mode: str
    branch: str
    sigma: float
    lambda_value: float
    sp_rank: int
    sp_world: int = EXPECTED_SP_WORLD
    global_target_tokens: int = EXPECTED_GLOBAL_TARGET_TOKENS
    phase_count: int = EXPECTED_PHASE_COUNT
    patch_height: int = EXPECTED_PATCH_HEIGHT
    patch_width: int = EXPECTED_PATCH_WIDTH

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise InternalTemporalQuotientObserverError(
                f"mode must be one of {VALID_MODES}, got {self.mode!r}"
            )
        if not isinstance(self.branch, str) or not self.branch.strip():
            raise InternalTemporalQuotientObserverError(
                "branch must be a non-empty explicit label"
            )
        if self.branch != self.branch.strip():
            raise InternalTemporalQuotientObserverError(
                "branch must not contain leading/trailing whitespace"
            )
        object.__setattr__(self, "sigma", _finite_real("sigma", self.sigma, nonnegative=True))
        object.__setattr__(
            self,
            "lambda_value",
            _finite_real("lambda", self.lambda_value, nonnegative=True),
        )
        if isinstance(self.sp_rank, bool) or not isinstance(self.sp_rank, int):
            raise InternalTemporalQuotientObserverError("sp_rank must be an integer")
        if self.sp_world != EXPECTED_SP_WORLD:
            raise InternalTemporalQuotientObserverError(
                f"pinned FITQ scan requires SP world {EXPECTED_SP_WORLD}, got {self.sp_world}"
            )
        if not 0 <= self.sp_rank < self.sp_world:
            raise InternalTemporalQuotientObserverError(
                f"sp_rank {self.sp_rank} is outside [0, {self.sp_world})"
            )
        pinned = (
            self.global_target_tokens == EXPECTED_GLOBAL_TARGET_TOKENS
            and self.phase_count == EXPECTED_PHASE_COUNT
            and self.patch_height == EXPECTED_PATCH_HEIGHT
            and self.patch_width == EXPECTED_PATCH_WIDTH
            and self.global_target_tokens
            == self.phase_count * self.patch_height * self.patch_width
        )
        if not pinned:
            raise InternalTemporalQuotientObserverError(
                "FITQ geometry must be pinned to N=19530 and 21x31x30"
            )

    @property
    def total_sequence_length(self) -> int:
        return self.global_target_tokens * (2 if self.mode == "mv2v" else 1)

    @property
    def local_sequence_length(self) -> int:
        return math.ceil(self.total_sequence_length / self.sp_world)

    @property
    def target_global_offset(self) -> int:
        return self.global_target_tokens if self.mode == "mv2v" else 0

    @property
    def branch_key(self) -> tuple[str, str, float, float]:
        """Uniquely identify a branch at one sigma/lambda scan cell."""

        return (self.mode, self.branch, self.sigma, self.lambda_value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "branch": self.branch,
            "sigma": self.sigma,
            "lambda": self.lambda_value,
            "sp_rank": self.sp_rank,
            "sp_world": self.sp_world,
            "global_target_tokens": self.global_target_tokens,
            "phase_geometry": [
                self.phase_count,
                self.patch_height,
                self.patch_width,
            ],
            "total_sequence_length": self.total_sequence_length,
            "local_sequence_length": self.local_sequence_length,
            "target_global_offset": self.target_global_offset,
        }


@dataclass(frozen=True)
class LocalTargetLayout:
    """Pinned mapping from one Ulysses local shard to target phases."""

    context: FITQObserverContext
    shard_start: int
    shard_stop: int
    valid_sequence_tokens: int
    source_tokens_excluded: int
    target_tokens_selected: int
    padding_tokens_excluded: int
    target_local_indices: Any = field(repr=False)
    target_phase_indices: Any = field(repr=False)
    expected_phase_token_count: Any = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_start": self.shard_start,
            "shard_stop": self.shard_stop,
            "valid_sequence_tokens": self.valid_sequence_tokens,
            "source_tokens_excluded": self.source_tokens_excluded,
            "target_tokens_selected": self.target_tokens_selected,
            "padding_tokens_excluded": self.padding_tokens_excluded,
            "phase_token_count": [
                int(value) for value in self.expected_phase_token_count.tolist()
            ],
        }


def build_local_target_layout(context: FITQObserverContext) -> LocalTargetLayout:
    """Apply ``global = rank * ceil(L/4) + local`` and select target only."""

    torch_module = _require_torch()
    local_length = context.local_sequence_length
    shard_start = context.sp_rank * local_length
    shard_stop = shard_start + local_length
    local_indices = torch_module.arange(local_length, dtype=torch_module.int64)
    global_indices = local_indices + shard_start
    valid = global_indices < context.total_sequence_length
    target_start = context.target_global_offset
    target_stop = target_start + context.global_target_tokens
    target = valid & (global_indices >= target_start) & (global_indices < target_stop)
    source = valid & (global_indices < target_start)
    selected_local = local_indices[target]
    selected_target = global_indices[target] - target_start
    phases = torch_module.div(
        selected_target,
        EXPECTED_TOKENS_PER_PHASE,
        rounding_mode="floor",
    )
    phase_count = torch_module.bincount(
        phases, minlength=EXPECTED_PHASE_COUNT
    ).to(dtype=torch_module.int64)
    valid_count = int(valid.sum().item())
    source_count = int(source.sum().item())
    target_count = int(target.sum().item())
    padding_count = local_length - valid_count
    if source_count + target_count != valid_count:
        raise InternalTemporalQuotientObserverError(
            "local sequence contains tokens outside pinned source/target layout"
        )
    if int(phase_count.sum().item()) != target_count:
        raise InternalTemporalQuotientObserverError(
            "target phase coverage does not equal selected target tokens"
        )
    if phases.numel() and (
        int(phases.min().item()) < 0
        or int(phases.max().item()) >= EXPECTED_PHASE_COUNT
    ):
        raise InternalTemporalQuotientObserverError(
            "target phase index escaped pinned 21-phase geometry"
        )
    return LocalTargetLayout(
        context=context,
        shard_start=shard_start,
        shard_stop=shard_stop,
        valid_sequence_tokens=valid_count,
        source_tokens_excluded=source_count,
        target_tokens_selected=target_count,
        padding_tokens_excluded=padding_count,
        target_local_indices=selected_local,
        target_phase_indices=phases,
        expected_phase_token_count=phase_count,
    )


@dataclass(frozen=True)
class PhaseHeadSufficientStatistics:
    """Additive FP32 statistics with shapes [phase, head, head_dim]/[phase, head]."""

    sum: Any
    sumsq: Any
    count: Any

    def clone(self) -> "PhaseHeadSufficientStatistics":
        return PhaseHeadSufficientStatistics(
            sum=self.sum.clone(),
            sumsq=self.sumsq.clone(),
            count=self.count.clone(),
        )


@dataclass(frozen=True)
class TensorExactFingerprint:
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    sha256: str


@dataclass(frozen=True)
class LocalFITQSufficientStatistics:
    """Complete rank-local result returned only after a verified forward."""

    context: FITQObserverContext
    layout: LocalTargetLayout
    sites: Mapping[str, PhaseHeadSufficientStatistics]
    call_counts: Mapping[str, int]
    call_sequence: tuple[str, ...]
    exact_fingerprints: Mapping[str, TensorExactFingerprint]
    globally_reduced: bool = False

    def require_complete(self) -> None:
        expected = expected_site_order()
        if self.call_sequence != expected:
            raise InternalTemporalQuotientObserverError(
                "local result does not contain the exact official hook sequence"
            )
        if tuple(self.sites) != expected or any(
            self.call_counts.get(site) != 1 for site in expected
        ):
            raise InternalTemporalQuotientObserverError(
                "local result has missing, duplicate, or reordered hook sites"
            )


def expected_site_order() -> tuple[str, ...]:
    order: list[str] = []
    for index in range(EXPECTED_BLOCK_COUNT):
        prefix = f"block.{index:02d}"
        order.extend(
            (
                f"{prefix}.input",
                f"{prefix}.attn1",
                f"{prefix}.attn2",
                f"{prefix}.output",
            )
        )
    order.append("proj_out.input")
    return tuple(order)


def _config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _validate_pinned_transformer(transformer: Any) -> None:
    blocks = getattr(transformer, "blocks", None)
    if blocks is None or len(blocks) != EXPECTED_BLOCK_COUNT:
        got = None if blocks is None else len(blocks)
        raise InternalTemporalQuotientObserverError(
            f"Bernini-R 1.3B must have {EXPECTED_BLOCK_COUNT} blocks, got {got}"
        )
    if not callable(getattr(transformer, "patch_vae_latent", None)):
        raise InternalTemporalQuotientObserverError(
            "candidate is not the official Wan transformer: patch_vae_latent missing"
        )
    proj_out = getattr(transformer, "proj_out", None)
    if proj_out is None or not callable(
        getattr(proj_out, "register_forward_pre_hook", None)
    ):
        raise InternalTemporalQuotientObserverError(
            "pinned transformer lacks hookable proj_out"
        )
    config = getattr(transformer, "config", None)
    for name, expected in (
        ("num_layers", EXPECTED_BLOCK_COUNT),
        ("num_attention_heads", EXPECTED_HEAD_COUNT),
        ("attention_head_dim", EXPECTED_HEAD_DIM),
    ):
        value = _config_value(config, name)
        if value is not None and int(value) != expected:
            raise InternalTemporalQuotientObserverError(
                f"pinned transformer config {name} must be {expected}, got {value}"
            )
    for index, block in enumerate(blocks):
        if not callable(getattr(block, "register_forward_pre_hook", None)) or not callable(
            getattr(block, "register_forward_hook", None)
        ):
            raise InternalTemporalQuotientObserverError(
                f"block {index} is not hookable"
            )
        for attention_name in ("attn1", "attn2"):
            attention = getattr(block, attention_name, None)
            to_out = getattr(attention, "to_out", None)
            try:
                projection = to_out[0]
            except (TypeError, IndexError, KeyError):
                projection = None
            if projection is None or not callable(
                getattr(projection, "register_forward_pre_hook", None)
            ):
                raise InternalTemporalQuotientObserverError(
                    f"block {index} {attention_name}.to_out[0] is not hookable"
                )
            in_features = getattr(projection, "in_features", None)
            if in_features is not None and int(in_features) != EXPECTED_HIDDEN_DIM:
                raise InternalTemporalQuotientObserverError(
                    f"block {index} {attention_name} input width must be "
                    f"{EXPECTED_HIDDEN_DIM}, got {in_features}"
                )


def resolve_pinned_wan_transformer(model: Any) -> Any:
    """Resolve exactly one official pinned 30-block transformer through wrappers."""

    queue = [model]
    seen: set[int] = set()
    matches: list[Any] = []
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None and callable(getattr(candidate, "patch_vae_latent", None)):
            _validate_pinned_transformer(candidate)
            matches.append(candidate)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                nested = get_base_model()
            except Exception:
                nested = None
            if nested is not None and nested is not candidate:
                queue.append(nested)
        for name in ("diff_dec", "transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    unique = {id(candidate): candidate for candidate in matches}
    if len(unique) != 1:
        qualifier = "could not resolve" if not unique else "resolved multiple"
        raise InternalTemporalQuotientObserverError(
            f"{qualifier} pinned 30-block Bernini-R Wan transformer"
        )
    return next(iter(unique.values()))


# Explicit alias used by code that emphasizes the block-count pin.
resolve_pinned_30_block_transformer = resolve_pinned_wan_transformer


@dataclass
class FITQCaptureSession:
    """Result holder for :meth:`InternalTemporalQuotientObserver.capture`."""

    context: FITQObserverContext
    result: Optional[LocalFITQSufficientStatistics] = None


class InternalTemporalQuotientObserver:
    """Strict read-only hook owner for one pinned official transformer."""

    def __init__(self, model: Any, *, capture_exact_block0: bool = False) -> None:
        self.transformer = resolve_pinned_wan_transformer(model)
        self.capture_exact_block0 = bool(capture_exact_block0)
        self._handles: list[Any] = []
        self._installed = False
        self._poisoned: Optional[str] = None
        self._active_context: Optional[FITQObserverContext] = None
        self._layout: Optional[LocalTargetLayout] = None
        self._seen_branch_keys: set[tuple[str, str, float, float]] = set()
        self._sites: dict[str, PhaseHeadSufficientStatistics] = {}
        self._call_counts: dict[str, int] = {}
        self._call_sequence: list[str] = []
        self._fingerprints: dict[str, TensorExactFingerprint] = {}
        self._stat_cache: dict[
            int, tuple[weakref.ReferenceType[Any], int, PhaseHeadSufficientStatistics]
        ] = {}
        self._fingerprint_cache: dict[
            int, tuple[weakref.ReferenceType[Any], int, TensorExactFingerprint]
        ] = {}

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def active(self) -> bool:
        return self._active_context is not None

    @property
    def poisoned(self) -> bool:
        return self._poisoned is not None

    @property
    def trainable_parameters(self) -> tuple[Any, ...]:
        """The observer is deliberately not an ``nn.Module`` or optimizer source."""

        return ()

    def install(self) -> "InternalTemporalQuotientObserver":
        if self._installed:
            raise InternalTemporalQuotientObserverError(
                "observer hooks are already installed"
            )
        if self.active:
            raise InternalTemporalQuotientObserverError(
                "cannot install hooks during an active forward"
            )
        handles: list[Any] = []
        try:
            for index, block in enumerate(self.transformer.blocks):
                prefix = f"block.{index:02d}"
                handles.append(
                    block.register_forward_pre_hook(
                        self._make_pre_hook(f"{prefix}.input")
                    )
                )
                handles.append(
                    block.attn1.to_out[0].register_forward_pre_hook(
                        self._make_pre_hook(f"{prefix}.attn1")
                    )
                )
                handles.append(
                    block.attn2.to_out[0].register_forward_pre_hook(
                        self._make_pre_hook(f"{prefix}.attn2")
                    )
                )
                handles.append(
                    block.register_forward_hook(
                        self._make_output_hook(f"{prefix}.output")
                    )
                )
            handles.append(
                self.transformer.proj_out.register_forward_pre_hook(
                    self._make_pre_hook("proj_out.input")
                )
            )
        except Exception:
            for handle in reversed(handles):
                handle.remove()
            raise
        self._handles = handles
        self._installed = True
        return self

    def remove(self) -> None:
        if not self._installed:
            raise InternalTemporalQuotientObserverError(
                "observer hooks are not installed"
            )
        if self.active:
            raise InternalTemporalQuotientObserverError(
                "cannot remove observer hooks during an active forward"
            )
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._installed = False

    def __enter__(self) -> "InternalTemporalQuotientObserver":
        return self.install()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.active:
            self.abort_forward()
        if self.installed:
            self.remove()

    def _contract_failure(self, message: str) -> InternalTemporalQuotientObserverError:
        self._poisoned = message
        return InternalTemporalQuotientObserverError(message)

    def begin_forward(self, context: FITQObserverContext) -> None:
        if not isinstance(context, FITQObserverContext):
            raise InternalTemporalQuotientObserverError(
                "begin_forward requires an explicit FITQObserverContext"
            )
        if not self._installed:
            raise InternalTemporalQuotientObserverError(
                "observer hooks must be installed before begin_forward"
            )
        if self._poisoned is not None:
            raise InternalTemporalQuotientObserverError(
                f"observer is fail-closed after: {self._poisoned}"
            )
        if self.active:
            raise InternalTemporalQuotientObserverError(
                "another observer forward is already active"
            )
        if context.branch_key in self._seen_branch_keys:
            raise InternalTemporalQuotientObserverError(
                "duplicate mode/branch/sigma/lambda observer cell"
            )
        self._seen_branch_keys.add(context.branch_key)
        self._active_context = context
        self._layout = build_local_target_layout(context)
        self._sites = {}
        self._call_counts = {}
        self._call_sequence = []
        self._fingerprints = {}
        self._stat_cache = {}
        self._fingerprint_cache = {}

    def abort_forward(self) -> None:
        if not self.active:
            raise InternalTemporalQuotientObserverError(
                "no active observer forward to abort"
            )
        self._active_context = None
        self._layout = None
        self._sites = {}
        self._call_counts = {}
        self._call_sequence = []
        self._fingerprints = {}
        self._stat_cache = {}
        self._fingerprint_cache = {}

    @contextmanager
    def capture(self, context: FITQObserverContext) -> Iterator[FITQCaptureSession]:
        session = FITQCaptureSession(context=context)
        self.begin_forward(context)
        try:
            yield session
        except BaseException:
            if self.active:
                self.abort_forward()
            raise
        else:
            session.result = self.finish_forward()

    def _make_pre_hook(self, site: str) -> Any:
        def hook(module: Any, inputs: Sequence[Any]) -> None:
            del module
            if not isinstance(inputs, tuple) or not inputs:
                raise self._contract_failure(f"{site} pre-hook lacks positional tensor input")
            self._observe(site, inputs[0])
            return None

        return hook

    def _make_output_hook(self, site: str) -> Any:
        def hook(module: Any, inputs: Sequence[Any], output: Any) -> None:
            del module, inputs
            self._observe(site, output)
            return None

        return hook

    def _observe(self, site: str, tensor: Any) -> None:
        torch_module = _require_torch()
        context = self._active_context
        layout = self._layout
        if not self._installed or context is None or layout is None:
            raise self._contract_failure(
                f"{site} fired without an explicit active observer context"
            )
        expected = expected_site_order()
        ordinal = len(self._call_sequence)
        if ordinal >= len(expected) or site != expected[ordinal]:
            wanted = None if ordinal >= len(expected) else expected[ordinal]
            raise self._contract_failure(
                f"unexpected hook order/call: got {site}, expected {wanted}"
            )
        if self._call_counts.get(site, 0) != 0:
            raise self._contract_failure(f"duplicate hook call for {site}")
        if not isinstance(tensor, torch_module.Tensor):
            raise self._contract_failure(f"{site} did not expose a tensor")
        wanted_shape = (1, context.local_sequence_length, EXPECTED_HIDDEN_DIM)
        if tuple(tensor.shape) != wanted_shape:
            raise self._contract_failure(
                f"{site} shape must be {wanted_shape}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise self._contract_failure(f"{site} tensor must be floating point")
        version_before = _tensor_version(tensor)
        stats = self._statistics_for_tensor(tensor, layout)
        if _tensor_version(tensor) != version_before:
            raise self._contract_failure(f"observer mutated tensor at {site}")
        self._sites[site] = stats
        self._call_counts[site] = 1
        self._call_sequence.append(site)
        if self.capture_exact_block0 and site in EXACT_PROBE_SITES:
            self._fingerprints[site] = self._fingerprint_for_tensor(tensor)

    def _statistics_for_tensor(
        self, tensor: Any, layout: LocalTargetLayout
    ) -> PhaseHeadSufficientStatistics:
        torch_module = _require_torch()
        tensor_id = id(tensor)
        version = _tensor_version(tensor)
        cacheable = version >= 0
        if cacheable:
            cached = self._stat_cache.get(tensor_id)
            if cached is not None and cached[0]() is tensor and cached[1] == version:
                return cached[2]
        with torch_module.no_grad():
            device = tensor.device
            local_indices = layout.target_local_indices.to(device=device)
            phase_indices = layout.target_phase_indices.to(device=device)
            values = tensor.detach()[0].reshape(
                tensor.shape[1], EXPECTED_HEAD_COUNT, EXPECTED_HEAD_DIM
            )
            values = values.index_select(0, local_indices).to(dtype=torch_module.float32)
            if values.numel() and not bool(torch_module.isfinite(values).all().item()):
                raise self._contract_failure("observer encountered non-finite activation")
            sums = torch_module.zeros(
                EXPECTED_PHASE_COUNT,
                EXPECTED_HEAD_COUNT,
                EXPECTED_HEAD_DIM,
                dtype=torch_module.float32,
                device=device,
            )
            sumsqs = torch_module.zeros_like(sums)
            if values.shape[0]:
                sums.index_add_(0, phase_indices, values)
                sumsqs.index_add_(0, phase_indices, values.square())
            phase_count = layout.expected_phase_token_count.to(
                device=device, dtype=torch_module.float32
            )
            counts = phase_count[:, None].expand(
                EXPECTED_PHASE_COUNT, EXPECTED_HEAD_COUNT
            ).clone()
            stats = PhaseHeadSufficientStatistics(
                sum=sums.detach(), sumsq=sumsqs.detach(), count=counts.detach()
            )
        if cacheable:
            self._stat_cache[tensor_id] = (weakref.ref(tensor), version, stats)
        return stats

    def _fingerprint_for_tensor(self, tensor: Any) -> TensorExactFingerprint:
        torch_module = _require_torch()
        tensor_id = id(tensor)
        version = _tensor_version(tensor)
        cacheable = version >= 0
        if cacheable:
            cached = self._fingerprint_cache.get(tensor_id)
            if cached is not None and cached[0]() is tensor and cached[1] == version:
                return cached[2]
        with torch_module.no_grad():
            # Hash the actual tensor value bytes, not a ``torch.save`` archive.
            # The legacy serializer embeds storage identities, so two
            # independently allocated but byte-equal tensors need not produce
            # the same archive bytes.  Viewing a private contiguous CPU clone
            # as uint8 preserves BF16 and NaN payload bits while excluding any
            # larger parent storage.
            detached = tensor.detach().to(device="cpu").contiguous().clone()
            expected_nbytes = int(tensor.numel()) * int(tensor.element_size())
            if detached.storage_offset() != 0 or (
                int(detached.numel()) * int(detached.element_size())
                != expected_nbytes
            ):
                raise self._contract_failure(
                    "exact block0 probe could not isolate tensor storage bytes"
                )
            raw_value_bytes = (
                detached.view(torch_module.uint8)
                .reshape(-1)
                .numpy()
                .tobytes(order="C")
            )
            if len(raw_value_bytes) != expected_nbytes:
                raise self._contract_failure(
                    "exact block0 probe byte count does not match tensor storage"
                )
        digest = hashlib.sha256()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(raw_value_bytes)
        fingerprint = TensorExactFingerprint(
            shape=tuple(int(value) for value in tensor.shape),
            dtype=str(tensor.dtype),
            nbytes=expected_nbytes,
            sha256=digest.hexdigest(),
        )
        if cacheable:
            self._fingerprint_cache[tensor_id] = (
                weakref.ref(tensor),
                version,
                fingerprint,
            )
        return fingerprint

    def finish_forward(self) -> LocalFITQSufficientStatistics:
        if not self.active or self._active_context is None or self._layout is None:
            raise InternalTemporalQuotientObserverError(
                "no active observer forward to finish"
            )
        context = self._active_context
        layout = self._layout
        expected = expected_site_order()
        try:
            if tuple(self._call_sequence) != expected:
                raise self._contract_failure(
                    "official forward missed, duplicated, or reordered observer hooks"
                )
            if tuple(self._sites) != expected or any(
                self._call_counts.get(site) != 1 for site in expected
            ):
                raise self._contract_failure(
                    "observer site/call-count coverage is incomplete"
                )
            expected_counts = layout.expected_phase_token_count.to(
                dtype=_require_torch().float32
            )[:, None].expand(EXPECTED_PHASE_COUNT, EXPECTED_HEAD_COUNT)
            for site in expected:
                stats = self._sites[site]
                if stats.sum.dtype != _require_torch().float32 or stats.sumsq.dtype != _require_torch().float32 or stats.count.dtype != _require_torch().float32:
                    raise self._contract_failure(f"{site} statistics are not FP32")
                if stats.sum.requires_grad or stats.sumsq.requires_grad or stats.count.requires_grad:
                    raise self._contract_failure(f"{site} statistics retained autograd")
                if tuple(stats.sum.shape) != (
                    EXPECTED_PHASE_COUNT,
                    EXPECTED_HEAD_COUNT,
                    EXPECTED_HEAD_DIM,
                ) or tuple(stats.sumsq.shape) != tuple(stats.sum.shape):
                    raise self._contract_failure(f"{site} phase/head statistic shape mismatch")
                if tuple(stats.count.shape) != (
                    EXPECTED_PHASE_COUNT,
                    EXPECTED_HEAD_COUNT,
                ):
                    raise self._contract_failure(f"{site} phase/head count shape mismatch")
                if not _require_torch().equal(stats.count.cpu(), expected_counts):
                    raise self._contract_failure(f"{site} local phase coverage mismatch")
                if not bool(_require_torch().isfinite(stats.sum).all().item()) or not bool(
                    _require_torch().isfinite(stats.sumsq).all().item()
                ):
                    raise self._contract_failure(f"{site} sufficient statistics are non-finite")
            fingerprints = dict(self._fingerprints)
            if self.capture_exact_block0 and tuple(fingerprints) != EXACT_PROBE_SITES:
                raise self._contract_failure("block0 exact probes are incomplete")
            result = LocalFITQSufficientStatistics(
                context=context,
                layout=layout,
                sites=MappingProxyType(dict(self._sites)),
                call_counts=MappingProxyType(dict(self._call_counts)),
                call_sequence=tuple(self._call_sequence),
                exact_fingerprints=MappingProxyType(fingerprints),
                globally_reduced=False,
            )
            result.require_complete()
            return result
        finally:
            self._active_context = None
            self._layout = None
            self._sites = {}
            self._call_counts = {}
            self._call_sequence = []
            self._fingerprints = {}
            self._stat_cache = {}
            self._fingerprint_cache = {}


def exact_block0_parity(
    left: LocalFITQSufficientStatistics,
    right: LocalFITQSufficientStatistics,
) -> dict[str, bool]:
    """Compare block-0 input/attn1 probes byte-exactly across two forwards."""

    left.require_complete()
    right.require_complete()
    left_state = (
        left.context.mode,
        left.context.sigma,
        left.context.lambda_value,
        left.context.sp_rank,
        left.context.sp_world,
        left.context.global_target_tokens,
        left.context.phase_count,
        left.context.patch_height,
        left.context.patch_width,
    )
    right_state = (
        right.context.mode,
        right.context.sigma,
        right.context.lambda_value,
        right.context.sp_rank,
        right.context.sp_world,
        right.context.global_target_tokens,
        right.context.phase_count,
        right.context.patch_height,
        right.context.patch_width,
    )
    if left_state != right_state:
        raise InternalTemporalQuotientObserverError(
            "exact block0 parity requires same mode/sigma/lambda/SP/geometry state"
        )
    missing = [
        site
        for site in EXACT_PROBE_SITES
        if site not in left.exact_fingerprints or site not in right.exact_fingerprints
    ]
    if missing:
        raise InternalTemporalQuotientObserverError(
            f"exact block0 fingerprints missing for {missing}"
        )
    result = {
        site: left.exact_fingerprints[site] == right.exact_fingerprints[site]
        for site in EXACT_PROBE_SITES
    }
    result["all"] = all(result.values())
    return result


def all_reduce_local_sufficient_statistics(
    local: LocalFITQSufficientStatistics,
    *,
    dist_module: Any = None,
    group: Any = None,
) -> LocalFITQSufficientStatistics:
    """SUM rank-local additive statistics outside all observer hooks.

    Three packed collectives are used (sum, sumsq, count).  Global phase
    coverage is accepted only when every one of the 21 phases has exactly
    ``31*30`` target tokens for every head and every site.
    """

    torch_module = _require_torch()
    local.require_complete()
    if local.globally_reduced:
        raise InternalTemporalQuotientObserverError(
            "sufficient statistics were already globally reduced"
        )
    dist = torch_module.distributed if dist_module is None else dist_module
    is_available = getattr(dist, "is_available", None)
    if callable(is_available) and not bool(is_available()):
        raise InternalTemporalQuotientObserverError("torch.distributed is unavailable")
    is_initialized = getattr(dist, "is_initialized", None)
    if callable(is_initialized) and not bool(is_initialized()):
        raise InternalTemporalQuotientObserverError("torch.distributed is not initialized")
    get_world_size = getattr(dist, "get_world_size", None)
    if not callable(get_world_size):
        raise InternalTemporalQuotientObserverError(
            "distributed reducer lacks get_world_size"
        )
    try:
        world = int(get_world_size(group=group))
    except TypeError:
        world = int(get_world_size(group))
    if world != local.context.sp_world:
        raise InternalTemporalQuotientObserverError(
            f"all-reduce world {world} differs from context SP world {local.context.sp_world}"
        )
    all_reduce = getattr(dist, "all_reduce", None)
    reduce_op = getattr(getattr(dist, "ReduceOp", None), "SUM", None)
    if not callable(all_reduce) or reduce_op is None:
        raise InternalTemporalQuotientObserverError(
            "distributed reducer lacks all_reduce SUM"
        )
    order = expected_site_order()
    packed_sum = torch_module.stack([local.sites[site].sum for site in order]).clone()
    packed_sumsq = torch_module.stack(
        [local.sites[site].sumsq for site in order]
    ).clone()
    packed_count = torch_module.stack(
        [local.sites[site].count for site in order]
    ).clone()
    for value in (packed_sum, packed_sumsq, packed_count):
        all_reduce(value, op=reduce_op, group=group)
    expected_count = torch_module.full_like(
        packed_count, float(EXPECTED_TOKENS_PER_PHASE)
    )
    if not torch_module.equal(packed_count, expected_count):
        raise InternalTemporalQuotientObserverError(
            "global phase coverage failed: source/padding exclusion or SP reduction is wrong"
        )
    if not bool(torch_module.isfinite(packed_sum).all().item()) or not bool(
        torch_module.isfinite(packed_sumsq).all().item()
    ):
        raise InternalTemporalQuotientObserverError(
            "globally reduced sufficient statistics are non-finite"
        )
    sites = {
        site: PhaseHeadSufficientStatistics(
            sum=packed_sum[index].detach(),
            sumsq=packed_sumsq[index].detach(),
            count=packed_count[index].detach(),
        )
        for index, site in enumerate(order)
    }
    result = LocalFITQSufficientStatistics(
        context=local.context,
        layout=local.layout,
        sites=MappingProxyType(sites),
        call_counts=local.call_counts,
        call_sequence=local.call_sequence,
        exact_fingerprints=local.exact_fingerprints,
        globally_reduced=True,
    )
    result.require_complete()
    return result


def install_internal_temporal_quotient_observer(
    model: Any, *, capture_exact_block0: bool = False
) -> InternalTemporalQuotientObserver:
    observer = InternalTemporalQuotientObserver(
        model, capture_exact_block0=capture_exact_block0
    )
    observer.install()
    return observer


__all__ = [
    "EXACT_PROBE_SITES",
    "EXPECTED_BLOCK_COUNT",
    "EXPECTED_GLOBAL_TARGET_TOKENS",
    "EXPECTED_HEAD_COUNT",
    "EXPECTED_HEAD_DIM",
    "EXPECTED_HIDDEN_DIM",
    "EXPECTED_PATCH_HEIGHT",
    "EXPECTED_PATCH_WIDTH",
    "EXPECTED_PHASE_COUNT",
    "EXPECTED_SP_WORLD",
    "EXPECTED_TOKENS_PER_PHASE",
    "FITQCaptureSession",
    "FITQObserverContext",
    "InternalTemporalQuotientObserver",
    "InternalTemporalQuotientObserverError",
    "LocalFITQSufficientStatistics",
    "LocalTargetLayout",
    "PhaseHeadSufficientStatistics",
    "TensorExactFingerprint",
    "all_reduce_local_sufficient_statistics",
    "build_local_target_layout",
    "exact_block0_parity",
    "expected_site_order",
    "install_internal_temporal_quotient_observer",
    "resolve_pinned_30_block_transformer",
    "resolve_pinned_wan_transformer",
]
