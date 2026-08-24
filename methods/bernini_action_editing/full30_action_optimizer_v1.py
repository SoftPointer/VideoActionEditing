#!/usr/bin/env python3
"""Transactional action-first optimizer core for full-30 Bernini training.

The runner is responsible for synchronizing the FP32 action/no-op gradients
before calling this module.  This core then provides the parts which must not
depend on a framework optimizer:

* canonical UTF-8 parameter-name order and bounded global buckets;
* one set of FP64 global projection, no-op cap, and clip coefficients;
* the registered no-momentum diagonal preconditioner; and
* an actual-FP32-displacement gate with byte-exact parameter rollback.

No CUDA or distributed process is started here.  A WORLD8 runner supplies an
``all_reduce_sum`` callback; replicated scalars are always divided by eight
before they are used.  The callback form also makes the collective semantics
hostile-testable on CPU.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Callable, Optional

try:
    import torch
except ImportError:  # Keep receipt/bucket helpers importable on non-torch hosts.
    torch = None  # type: ignore[assignment]


SCHEMA_VERSION = "bernini-full30-action-first-optimizer-v1"
RECEIPT_SCHEMA_VERSION = "bernini-full30-action-first-update-receipt-v1"
STATE_SCHEMA_VERSION = "bernini-full30-action-first-optimizer-state-v1"

ACTION_LEARNING_RATE = 1.0e-4
ACTION_BETA2 = 0.999
NUMERIC_EPSILON = 1.0e-8
GLOBAL_UPDATE_CLIP = 1.0
MIN_ACTION_GRAD_NORM = 1.0e-12
PROJECTION_ROUNDING_RELATIVE_TOLERANCE = 1.0e-6
MAX_FP32_BUCKET_ELEMENTS = 16_777_216
REGISTERED_WORLD_SIZE = 8

_STAT_G2 = 0
_STAT_PA2 = 1
_STAT_PN2 = 2
_STAT_G_DOT_PN = 3
_STAT_PA_DOT_PN = 4
_STAT_PA_DOT_G = 5
_STAT_COUNT = 6

_VALIDATION_G_DOT_Q = 0
_VALIDATION_Q2 = 1
_VALIDATION_ACTUAL_DOT = 2
_VALIDATION_DISPLACEMENT2 = 3
_VALIDATION_COUNT = 4


class Full30ActionOptimizerError(RuntimeError):
    """Raised when the registered optimizer contract cannot be satisfied."""


@dataclass(frozen=True)
class TensorChunkV1:
    """A closed-over parameter name and flat half-open element interval."""

    parameter_name: str
    parameter_offset: int
    element_count: int

    @property
    def parameter_stop(self) -> int:
        return self.parameter_offset + self.element_count

    def receipt(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_offset": self.parameter_offset,
            "parameter_stop": self.parameter_stop,
            "element_count": self.element_count,
        }


@dataclass(frozen=True)
class ParameterBucketV1:
    bucket_index: int
    global_offset: int
    element_count: int
    chunks: tuple[TensorChunkV1, ...]

    @property
    def global_stop(self) -> int:
        return self.global_offset + self.element_count

    def receipt(self) -> dict[str, Any]:
        return {
            "bucket_index": self.bucket_index,
            "global_offset": self.global_offset,
            "global_stop": self.global_stop,
            "element_count": self.element_count,
            "chunks": [chunk.receipt() for chunk in self.chunks],
        }


AllReduceSum = Callable[[Any], Optional[Any]]


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict, deterministic ASCII JSON bytes without a newline."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Full30ActionOptimizerError("value is not canonical-JSON encodable") from error
    return encoded


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Validate a sealed step receipt and return its canonical bytes."""

    if not isinstance(receipt, Mapping):
        raise Full30ActionOptimizerError("receipt must be a mapping")
    value = dict(receipt)
    digest = value.pop("receipt_digest", None)
    if type(digest) is not str or digest != object_sha256(value):
        raise Full30ActionOptimizerError("receipt digest differs")
    return canonical_json_bytes(dict(receipt))


def _seal_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in value:
        raise Full30ActionOptimizerError("unsigned receipt already has a digest")
    sealed = dict(value)
    sealed["receipt_digest"] = object_sha256(value)
    canonical_receipt_bytes(sealed)
    return sealed


def _canonical_name(name: Any) -> tuple[str, bytes]:
    if type(name) is not str or not name or "\x00" in name:
        raise Full30ActionOptimizerError("parameter names must be nonempty NUL-free text")
    try:
        encoded = name.encode("utf-8")
    except UnicodeEncodeError as error:
        raise Full30ActionOptimizerError("parameter name is not valid UTF-8 text") from error
    return name, encoded


def _canonical_numels(
    named_numels: Mapping[str, int] | Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    raw = tuple(named_numels.items()) if isinstance(named_numels, Mapping) else tuple(named_numels)
    seen: set[str] = set()
    checked: list[tuple[str, bytes, int]] = []
    for item in raw:
        if not isinstance(item, tuple) or len(item) != 2:
            raise Full30ActionOptimizerError("named element counts must contain name/count pairs")
        name, count = item
        text, encoded = _canonical_name(name)
        if text in seen:
            raise Full30ActionOptimizerError(f"duplicate parameter name: {text}")
        if type(count) is not int or count <= 0:
            raise Full30ActionOptimizerError(f"parameter {text} must contain at least one element")
        seen.add(text)
        checked.append((text, encoded, count))
    if not checked:
        raise Full30ActionOptimizerError("at least one named parameter is required")
    checked.sort(key=lambda item: item[1])
    return tuple((name, count) for name, _, count in checked)


def build_canonical_bucket_plan_v1(
    named_numels: Mapping[str, int] | Iterable[tuple[str, int]],
    *,
    max_chunk_elements: int = MAX_FP32_BUCKET_ELEMENTS,
) -> tuple[ParameterBucketV1, ...]:
    """Pack the canonical flat parameter stream into bounded buckets.

    A tensor may cross any number of bucket boundaries.  Every executable
    closure therefore carries both its parameter name and its own flat offset;
    no loop-variable lambda capture is involved.
    """

    if (
        type(max_chunk_elements) is not int
        or max_chunk_elements <= 0
        or max_chunk_elements > MAX_FP32_BUCKET_ELEMENTS
    ):
        raise Full30ActionOptimizerError(
            f"max_chunk_elements must be in [1,{MAX_FP32_BUCKET_ELEMENTS}]"
        )
    canonical = _canonical_numels(named_numels)
    buckets: list[ParameterBucketV1] = []
    chunks: list[TensorChunkV1] = []
    bucket_elements = 0
    bucket_global_offset = 0
    emitted = 0

    def close_bucket() -> None:
        nonlocal chunks, bucket_elements, bucket_global_offset
        if not chunks or bucket_elements <= 0:
            raise Full30ActionOptimizerError("internal empty bucket")
        buckets.append(
            ParameterBucketV1(
                bucket_index=len(buckets),
                global_offset=bucket_global_offset,
                element_count=bucket_elements,
                chunks=tuple(chunks),
            )
        )
        bucket_global_offset += bucket_elements
        chunks = []
        bucket_elements = 0

    for name, count in canonical:
        parameter_offset = 0
        while parameter_offset < count:
            available = max_chunk_elements - bucket_elements
            take = min(available, count - parameter_offset)
            chunks.append(
                TensorChunkV1(
                    parameter_name=name,
                    parameter_offset=parameter_offset,
                    element_count=take,
                )
            )
            parameter_offset += take
            bucket_elements += take
            emitted += take
            if bucket_elements == max_chunk_elements:
                close_bucket()
    if chunks:
        close_bucket()
    total = sum(count for _, count in canonical)
    if emitted != total or sum(bucket.element_count for bucket in buckets) != total:
        raise Full30ActionOptimizerError("bucket plan does not close the parameter stream")
    if any(
        bucket.element_count > max_chunk_elements
        or any(chunk.element_count > max_chunk_elements for chunk in bucket.chunks)
        for bucket in buckets
    ):
        raise Full30ActionOptimizerError("bucket plan exceeds the registered element cap")
    return tuple(buckets)


def bucket_plan_receipt_v1(
    plan: Iterable[ParameterBucketV1], *, max_chunk_elements: int
) -> dict[str, Any]:
    buckets = tuple(plan)
    payload = {
        "ordering": "parameter-name-utf8-byte-ascending-then-flat-offset-ascending",
        "registered_max_elements": MAX_FP32_BUCKET_ELEMENTS,
        "configured_max_elements": max_chunk_elements,
        "bucket_count": len(buckets),
        "element_count": sum(bucket.element_count for bucket in buckets),
        "buckets": [bucket.receipt() for bucket in buckets],
    }
    return {**payload, "bucket_order_digest": object_sha256(payload)}


def _named_parameters(
    named_parameters: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
) -> tuple[tuple[str, torch.Tensor], ...]:
    raw = (
        tuple(named_parameters.items())
        if isinstance(named_parameters, Mapping)
        else tuple(named_parameters)
    )
    names_and_counts: list[tuple[str, int]] = []
    by_name: dict[str, torch.Tensor] = {}
    storages: set[tuple[str, int]] = set()
    device: Optional[torch.device] = None
    for item in raw:
        if not isinstance(item, tuple) or len(item) != 2:
            raise Full30ActionOptimizerError("named_parameters must contain name/tensor pairs")
        name, parameter = item
        text, _ = _canonical_name(name)
        if text in by_name:
            raise Full30ActionOptimizerError(f"duplicate parameter name: {text}")
        if (
            not isinstance(parameter, torch.Tensor)
            or parameter.layout != torch.strided
            or parameter.dtype != torch.float32
            or not parameter.is_contiguous()
            or parameter.numel() <= 0
        ):
            raise Full30ActionOptimizerError(
                f"parameter {text} must be a nonempty contiguous strided FP32 tensor"
            )
        if device is None:
            device = parameter.device
        elif parameter.device != device:
            raise Full30ActionOptimizerError("all parameters must be on one device")
        storage_key = (str(parameter.device), int(parameter.untyped_storage().data_ptr()))
        if storage_key in storages:
            raise Full30ActionOptimizerError("parameter storage aliasing is not supported")
        storages.add(storage_key)
        by_name[text] = parameter
        names_and_counts.append((text, int(parameter.numel())))
    canonical = _canonical_numels(names_and_counts)
    return tuple((name, by_name[name]) for name, _ in canonical)


def _flat_chunk(tensors: Mapping[str, torch.Tensor], chunk: TensorChunkV1) -> torch.Tensor:
    value = tensors[chunk.parameter_name].view(-1)
    return value.narrow(0, chunk.parameter_offset, chunk.element_count)


def _tensor_payload_bytes(value: torch.Tensor) -> bytes:
    contiguous = value.detach().contiguous().cpu()
    return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def _named_tensor_digest(
    domain: bytes,
    canonical_names: Iterable[str],
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256(domain)
    for name in canonical_names:
        value = tensors[name]
        encoded = name.encode("utf-8")
        raw = _tensor_payload_bytes(value)
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">I", value.ndim))
        for size in value.shape:
            digest.update(struct.pack(">Q", int(size)))
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _receipt_float(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise Full30ActionOptimizerError(f"{label} is non-finite")
    return 0.0 if number == 0.0 else number


def _nonnegative_square(value: float, *, scale: float, label: str) -> float:
    if not math.isfinite(value):
        raise Full30ActionOptimizerError(f"{label} is non-finite")
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(1.0, abs(scale))
    if value < -tolerance:
        raise Full30ActionOptimizerError(f"{label} is negative beyond FP64 roundoff")
    return max(0.0, value)


class Full30ActionFirstOptimizerV1:
    """Registered full-vector action-first optimizer with transactional state."""

    def __init__(
        self,
        named_parameters: Mapping[str, torch.Tensor]
        | Iterable[tuple[str, torch.Tensor]],
        *,
        max_chunk_elements: int = MAX_FP32_BUCKET_ELEMENTS,
    ) -> None:
        if torch is None:  # pragma: no cover - depends on the execution host
            raise Full30ActionOptimizerError("PyTorch is required for optimizer execution")
        canonical = _named_parameters(named_parameters)
        self._parameters = {name: parameter for name, parameter in canonical}
        self._names = tuple(name for name, _ in canonical)
        self._device = canonical[0][1].device
        self._max_chunk_elements = max_chunk_elements
        self._plan = build_canonical_bucket_plan_v1(
            ((name, int(parameter.numel())) for name, parameter in canonical),
            max_chunk_elements=max_chunk_elements,
        )
        self._plan_receipt = bucket_plan_receipt_v1(
            self._plan, max_chunk_elements=max_chunk_elements
        )
        self._second_moments = {
            name: torch.zeros_like(parameter, memory_format=torch.preserve_format)
            for name, parameter in canonical
        }
        self._update_count = 0
        self._assert_parameters_and_state_finite()

    @property
    def canonical_parameter_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def bucket_plan(self) -> tuple[ParameterBucketV1, ...]:
        return self._plan

    @property
    def update_count(self) -> int:
        return self._update_count

    def second_moment(self, name: str) -> torch.Tensor:
        if name not in self._second_moments:
            raise Full30ActionOptimizerError(f"unknown parameter: {name}")
        return self._second_moments[name].detach().clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "update_count": self._update_count,
            "canonical_parameter_names": list(self._names),
            "second_moments": {
                name: self._second_moments[name].detach().clone() for name in self._names
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise Full30ActionOptimizerError("optimizer state must be a mapping")
        update_count = state.get("update_count")
        raw_moments = state.get("second_moments")
        if (
            state.get("schema_version") != STATE_SCHEMA_VERSION
            or state.get("canonical_parameter_names") != list(self._names)
            or type(update_count) is not int
            or update_count < 0
            or not isinstance(raw_moments, Mapping)
            or set(raw_moments) != set(self._names)
        ):
            raise Full30ActionOptimizerError("optimizer state contract differs")
        candidates: dict[str, torch.Tensor] = {}
        for name in self._names:
            value = raw_moments[name]
            reference = self._parameters[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or value.shape != reference.shape
                or value.device != reference.device
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
                or bool((value < 0).any().item())
            ):
                raise Full30ActionOptimizerError(f"second moment for {name} differs")
            candidates[name] = value.detach().clone()
        self._second_moments = candidates
        self._update_count = update_count

    def _assert_parameters_and_state_finite(self) -> None:
        for name in self._names:
            parameter = self._parameters[name]
            moment = self._second_moments[name]
            if not bool(torch.isfinite(parameter).all().item()):
                raise Full30ActionOptimizerError(f"parameter {name} is non-finite")
            if (
                not bool(torch.isfinite(moment).all().item())
                or bool((moment < 0).any().item())
            ):
                raise Full30ActionOptimizerError(f"second moment {name} is invalid")

    def _gradients(
        self,
        gradients: Mapping[str, torch.Tensor],
        *,
        label: str,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(gradients, Mapping) or set(gradients) != set(self._names):
            raise Full30ActionOptimizerError(f"{label} gradient names differ")
        checked: dict[str, torch.Tensor] = {}
        parameter_storage = {
            (str(value.device), int(value.untyped_storage().data_ptr()))
            for value in self._parameters.values()
        }
        for name in self._names:
            value = gradients[name]
            reference = self._parameters[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.dtype != torch.float32
                or value.shape != reference.shape
                or value.device != reference.device
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise Full30ActionOptimizerError(
                    f"{label} gradient {name} must be matching finite contiguous FP32"
                )
            storage_key = (str(value.device), int(value.untyped_storage().data_ptr()))
            if storage_key in parameter_storage:
                raise Full30ActionOptimizerError(
                    f"{label} gradient {name} aliases parameter storage"
                )
            checked[name] = value.detach()
        return checked

    def _world_mean(
        self,
        local: torch.Tensor,
        *,
        world_size: int,
        all_reduce_sum: Optional[AllReduceSum],
        label: str,
    ) -> tuple[torch.Tensor, bool]:
        if local.dtype != torch.float64 or local.ndim != 1 or local.device != self._device:
            raise Full30ActionOptimizerError("internal scalar vector contract differs")
        if world_size == 1:
            if all_reduce_sum is not None:
                raise Full30ActionOptimizerError(
                    "all_reduce_sum must be absent for the local CPU path"
                )
            return local, True
        if world_size != REGISTERED_WORLD_SIZE or all_reduce_sum is None:
            raise Full30ActionOptimizerError("only local or registered WORLD8 reduction is valid")
        work = local.clone()
        try:
            returned = all_reduce_sum(work)
        except Exception as error:
            raise Full30ActionOptimizerError(f"{label} WORLD8 all-reduce failed") from error
        reduced = work if returned is None else returned
        if (
            not isinstance(reduced, torch.Tensor)
            or reduced.dtype != torch.float64
            or reduced.shape != local.shape
            or reduced.device != local.device
            or not bool(torch.isfinite(reduced).all().item())
        ):
            raise Full30ActionOptimizerError(f"{label} WORLD8 sum differs")
        mean = reduced / float(world_size)
        delta = float(torch.max(torch.abs(mean - local)).item())
        scale = max(1.0, float(torch.max(torch.abs(local)).item()))
        consensus = delta <= 32.0 * torch.finfo(torch.float64).eps * scale
        if not consensus:
            raise Full30ActionOptimizerError(
                f"{label} WORLD8 replicated-scalar consensus differs"
            )
        return mean, consensus

    def _preconditioned(
        self,
        name: str,
        chunk: TensorChunkV1,
        action: Mapping[str, torch.Tensor],
        noop: Optional[Mapping[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        g = _flat_chunk(action, chunk)
        old_moment = _flat_chunk(self._second_moments, chunk)
        new_moment = (
            ACTION_BETA2 * old_moment
            + (1.0 - ACTION_BETA2) * g.square()
        ).float()
        denominator = new_moment.sqrt() + NUMERIC_EPSILON
        action_direction = (g / denominator).float()
        noop_direction = None
        if noop is not None:
            noop_direction = (_flat_chunk(noop, chunk) / denominator).float()
        values = (new_moment, denominator, action_direction)
        if any(not bool(torch.isfinite(value).all().item()) for value in values) or (
            noop_direction is not None
            and not bool(torch.isfinite(noop_direction).all().item())
        ):
            raise Full30ActionOptimizerError(
                f"preconditioned direction for {name} is non-finite"
            )
        return g, action_direction, noop_direction, new_moment

    def step(
        self,
        action_gradients: Mapping[str, torch.Tensor],
        *,
        noop_gradients: Optional[Mapping[str, torch.Tensor]] = None,
        world_size: int = 1,
        all_reduce_sum: Optional[AllReduceSum] = None,
    ) -> dict[str, Any]:
        """Apply one all-parameter transaction and return a sealed receipt."""

        self._assert_parameters_and_state_finite()
        action = self._gradients(action_gradients, label="action")
        noop = (
            None
            if noop_gradients is None
            else self._gradients(noop_gradients, label="noop")
        )
        arm = "action-only" if noop is None else "action+retain"
        before_parameter_digest = _named_tensor_digest(
            b"full30-action-parameters-v1\x00", self._names, self._parameters
        )
        before_moment_digest = _named_tensor_digest(
            b"full30-action-second-moment-v1\x00", self._names, self._second_moments
        )

        # Pass 1: canonical bucket FP64 statistics.  No coefficient is computed
        # inside a bucket, so bucket boundaries cannot alter the update.
        local_stats = torch.zeros(_STAT_COUNT, dtype=torch.float64, device=self._device)
        with torch.no_grad():
            for bucket in self._plan:
                bucket_stats = torch.zeros_like(local_stats)
                for chunk in bucket.chunks:
                    g, pa, pn, _ = self._preconditioned(
                        chunk.parameter_name, chunk, action, noop
                    )
                    gd = g.double()
                    pad = pa.double()
                    bucket_stats[_STAT_G2] += torch.sum(gd * gd)
                    bucket_stats[_STAT_PA2] += torch.sum(pad * pad)
                    if pn is not None:
                        pnd = pn.double()
                        bucket_stats[_STAT_PN2] += torch.sum(pnd * pnd)
                        bucket_stats[_STAT_G_DOT_PN] += torch.sum(gd * pnd)
                        bucket_stats[_STAT_PA_DOT_PN] += torch.sum(pad * pnd)
                        bucket_stats[_STAT_PA_DOT_G] += torch.sum(pad * gd)
                local_stats += bucket_stats
        global_stats, first_consensus = self._world_mean(
            local_stats,
            world_size=world_size,
            all_reduce_sum=all_reduce_sum,
            label="coefficient-pass",
        )
        stats = [float(value.item()) for value in global_stats]
        if not all(math.isfinite(value) for value in stats):
            raise Full30ActionOptimizerError("global coefficient statistic is non-finite")
        g2 = _nonnegative_square(stats[_STAT_G2], scale=stats[_STAT_G2], label="action norm square")
        action_norm = math.sqrt(g2)
        if action_norm <= MIN_ACTION_GRAD_NORM:
            raise Full30ActionOptimizerError("action gradient is degenerate")
        pa2 = _nonnegative_square(stats[_STAT_PA2], scale=stats[_STAT_PA2], label="action direction norm square")
        action_direction_norm = math.sqrt(pa2)

        conflict_before: Optional[float] = None
        projection_coefficient: Optional[float] = None
        projected_noop2 = 0.0
        projected_action_noop_dot = 0.0
        noop_norm_before: Optional[float] = None
        projected_noop_norm: Optional[float] = None
        cap_factor: Optional[float] = None
        projection_applied = False
        if noop is not None:
            pn2 = _nonnegative_square(stats[_STAT_PN2], scale=stats[_STAT_PN2], label="noop direction norm square")
            noop_norm_before = math.sqrt(pn2)
            conflict_before = stats[_STAT_G_DOT_PN]
            projection_applied = conflict_before < 0.0
            projection_coefficient = conflict_before / g2 if projection_applied else 0.0
            projected_raw = (
                pn2
                - 2.0 * projection_coefficient * conflict_before
                + projection_coefficient * projection_coefficient * g2
            )
            projected_scale = max(
                pn2,
                abs(2.0 * projection_coefficient * conflict_before),
                abs(projection_coefficient * projection_coefficient * g2),
            )
            projected_noop2 = _nonnegative_square(
                projected_raw,
                scale=projected_scale,
                label="projected noop norm square",
            )
            projected_noop_norm = math.sqrt(projected_noop2)
            projected_action_noop_dot = (
                stats[_STAT_PA_DOT_PN]
                - projection_coefficient * stats[_STAT_PA_DOT_G]
            )
            cap_factor = min(
                1.0,
                action_direction_norm / (projected_noop_norm + NUMERIC_EPSILON),
            )

        if noop is None:
            preclip2_raw = pa2
            preclip_scale = pa2
        else:
            assert cap_factor is not None
            preclip2_raw = (
                pa2
                + cap_factor * cap_factor * projected_noop2
                + 2.0 * cap_factor * projected_action_noop_dot
            )
            preclip_scale = max(
                pa2,
                cap_factor * cap_factor * projected_noop2,
                abs(2.0 * cap_factor * projected_action_noop_dot),
            )
        preclip2 = _nonnegative_square(
            preclip2_raw, scale=preclip_scale, label="combined direction norm square"
        )
        preclip_norm = math.sqrt(preclip2)
        clip_factor = min(
            1.0, GLOBAL_UPDATE_CLIP / (preclip_norm + NUMERIC_EPSILON)
        )
        for label, value in (
            ("projection coefficient", projection_coefficient),
            ("noop cap factor", cap_factor),
            ("global clip factor", clip_factor),
        ):
            if value is not None and not math.isfinite(value):
                raise Full30ActionOptimizerError(f"{label} is non-finite")

        # Pass 2: recompute each bounded chunk, apply the one global set of
        # coefficients in FP32, and measure the displacement which really
        # landed in FP32 parameters.  State is staged until every gate passes.
        snapshots = {
            name: self._parameters[name].detach().clone() for name in self._names
        }
        candidate_moments = {
            name: torch.empty_like(self._second_moments[name]) for name in self._names
        }
        local_validation = torch.zeros(
            _VALIDATION_COUNT, dtype=torch.float64, device=self._device
        )
        mutated = False
        try:
            with torch.no_grad():
                for bucket in self._plan:
                    bucket_validation = torch.zeros_like(local_validation)
                    for chunk in bucket.chunks:
                        g, pa, pn, new_moment = self._preconditioned(
                            chunk.parameter_name, chunk, action, noop
                        )
                        _flat_chunk(candidate_moments, chunk).copy_(new_moment)
                        if pn is None:
                            direction = (pa * clip_factor).float()
                        else:
                            assert projection_coefficient is not None
                            assert cap_factor is not None
                            if projection_applied:
                                projected_noop = (
                                    pn.double()
                                    - projection_coefficient * g.double()
                                ).float()
                            else:
                                projected_noop = pn.clone()
                            qd = projected_noop.double()
                            gd = g.double()
                            bucket_validation[_VALIDATION_G_DOT_Q] += torch.sum(gd * qd)
                            bucket_validation[_VALIDATION_Q2] += torch.sum(qd * qd)
                            direction = (
                                (pa + (projected_noop * cap_factor).float()).float()
                                * clip_factor
                            ).float()
                        if not bool(torch.isfinite(direction).all().item()):
                            raise Full30ActionOptimizerError("final FP32 direction is non-finite")
                        parameter_chunk = _flat_chunk(self._parameters, chunk)
                        before_chunk = _flat_chunk(snapshots, chunk)
                        mutated = True
                        parameter_chunk.add_(direction, alpha=-ACTION_LEARNING_RATE)
                        if not bool(torch.isfinite(parameter_chunk).all().item()):
                            raise Full30ActionOptimizerError("candidate FP32 parameter is non-finite")
                        displacement = (before_chunk - parameter_chunk).float()
                        gd = g.double()
                        dd = displacement.double()
                        bucket_validation[_VALIDATION_ACTUAL_DOT] += torch.sum(gd * dd)
                        bucket_validation[_VALIDATION_DISPLACEMENT2] += torch.sum(dd * dd)
                    local_validation += bucket_validation

            global_validation, second_consensus = self._world_mean(
                local_validation,
                world_size=world_size,
                all_reduce_sum=all_reduce_sum,
                label="transaction-validation-pass",
            )
            validation = [float(value.item()) for value in global_validation]
            if not all(math.isfinite(value) for value in validation):
                raise Full30ActionOptimizerError("global transaction validation is non-finite")
            actual_projected_noop2 = _nonnegative_square(
                validation[_VALIDATION_Q2],
                scale=validation[_VALIDATION_Q2],
                label="actual projected noop norm square",
            )
            actual_projected_noop_norm = math.sqrt(actual_projected_noop2)
            projection_residual = validation[_VALIDATION_G_DOT_Q] if noop is not None else None
            projection_tolerance = (
                PROJECTION_ROUNDING_RELATIVE_TOLERANCE
                * max(1.0, action_norm * actual_projected_noop_norm)
                if noop is not None
                else None
            )
            if (
                projection_applied
                and projection_residual is not None
                and projection_tolerance is not None
                and projection_residual < -projection_tolerance
            ):
                raise Full30ActionOptimizerError(
                    "FP32 noop projection still opposes the action gradient"
                )
            actual_action_descent_dot = validation[_VALIDATION_ACTUAL_DOT]
            if (
                not math.isfinite(actual_action_descent_dot)
                or actual_action_descent_dot <= 0.0
            ):
                raise Full30ActionOptimizerError(
                    "actual FP32 action displacement dot is not strictly positive"
                )
            displacement2 = _nonnegative_square(
                validation[_VALIDATION_DISPLACEMENT2],
                scale=validation[_VALIDATION_DISPLACEMENT2],
                label="actual displacement norm square",
            )
            actual_displacement_norm = math.sqrt(displacement2)
            after_parameter_digest = _named_tensor_digest(
                b"full30-action-parameters-v1\x00", self._names, self._parameters
            )
            after_moment_digest = _named_tensor_digest(
                b"full30-action-second-moment-v1\x00", self._names, candidate_moments
            )
            unsigned = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "status": "committed",
                "arm": arm,
                "update_count_before": self._update_count,
                "update_count_after": self._update_count + 1,
                "canonical_parameter_names": list(self._names),
                "parameter_tensor_count": len(self._names),
                "parameter_element_count": self._plan_receipt["element_count"],
                "bucket_plan": self._plan_receipt,
                "algorithm": {
                    "pass_count": 2,
                    "coefficient_scope": "one-global-set-after-all-canonical-buckets",
                    "projection_scope": "global-once",
                    "noop_cap_scope": "global-once",
                    "clip_scope": "global-once",
                    "actual_displacement_dtype": "float32",
                    "transactional_parameter_rollback": True,
                    "state_committed_only_after_gate": True,
                },
                "hyperparameters": {
                    "learning_rate": ACTION_LEARNING_RATE,
                    "beta2": ACTION_BETA2,
                    "epsilon": NUMERIC_EPSILON,
                    "global_clip": GLOBAL_UPDATE_CLIP,
                    "initial_second_moment": "zeros-float32",
                    "momentum": False,
                    "weight_decay": False,
                    "bias_correction": False,
                },
                "world_reduction": {
                    "world_size": world_size,
                    "registered_world_size": REGISTERED_WORLD_SIZE,
                    "replicated_scalar_policy": (
                        "local-identity"
                        if world_size == 1
                        else "all-reduce-sum-divided-by-world-size"
                    ),
                    "uses_mean_not_sum": True,
                    "coefficient_pass_consensus": first_consensus,
                    "validation_pass_consensus": second_consensus,
                    "all_reduce_call_count": 0 if world_size == 1 else 2,
                },
                "statistics": {
                    "action_gradient_norm": _receipt_float(action_norm, label="action norm"),
                    "action_preconditioned_norm": _receipt_float(
                        action_direction_norm, label="action direction norm"
                    ),
                    "noop_preconditioned_norm_before_projection": (
                        None
                        if noop_norm_before is None
                        else _receipt_float(noop_norm_before, label="noop norm")
                    ),
                    "conflict_dot_before_projection": (
                        None
                        if conflict_before is None
                        else _receipt_float(conflict_before, label="conflict dot")
                    ),
                    "projection_applied": projection_applied,
                    "global_projection_coefficient": (
                        None
                        if projection_coefficient is None
                        else _receipt_float(
                            projection_coefficient, label="projection coefficient"
                        )
                    ),
                    "projected_noop_norm_for_coefficients": (
                        None
                        if projected_noop_norm is None
                        else _receipt_float(projected_noop_norm, label="projected noop norm")
                    ),
                    "fp32_projection_residual": (
                        None
                        if projection_residual is None
                        else _receipt_float(projection_residual, label="projection residual")
                    ),
                    "fp32_projection_tolerance": (
                        None
                        if projection_tolerance is None
                        else _receipt_float(projection_tolerance, label="projection tolerance")
                    ),
                    "noop_cap_factor": (
                        None
                        if cap_factor is None
                        else _receipt_float(cap_factor, label="noop cap factor")
                    ),
                    "pre_global_clip_norm": _receipt_float(
                        preclip_norm, label="preclip norm"
                    ),
                    "global_clip_factor": _receipt_float(
                        clip_factor, label="clip factor"
                    ),
                    "actual_action_descent_dot": _receipt_float(
                        actual_action_descent_dot, label="actual action descent dot"
                    ),
                    "actual_parameter_displacement_norm": _receipt_float(
                        actual_displacement_norm, label="actual displacement norm"
                    ),
                },
                "digests": {
                    "parameters_before": before_parameter_digest,
                    "parameters_after": after_parameter_digest,
                    "second_moment_before": before_moment_digest,
                    "second_moment_after": after_moment_digest,
                },
                "gate": {
                    "formula": "g_action_dot_theta_before_minus_theta_after",
                    "comparison": "strictly-greater-than-zero",
                    "passed": True,
                },
            }
            receipt = _seal_receipt(unsigned)
            self._second_moments = candidate_moments
            self._update_count += 1
            return receipt
        except Exception as error:
            if mutated:
                with torch.no_grad():
                    for name in self._names:
                        self._parameters[name].copy_(snapshots[name])
            if isinstance(error, Full30ActionOptimizerError):
                raise
            raise Full30ActionOptimizerError("optimizer transaction failed and rolled back") from error


# Short alias for runner code which does not need to repeat "First".
Full30ActionOptimizerV1 = Full30ActionFirstOptimizerV1


__all__ = [
    "ACTION_BETA2",
    "ACTION_LEARNING_RATE",
    "Full30ActionFirstOptimizerV1",
    "Full30ActionOptimizerError",
    "Full30ActionOptimizerV1",
    "GLOBAL_UPDATE_CLIP",
    "MAX_FP32_BUCKET_ELEMENTS",
    "NUMERIC_EPSILON",
    "ParameterBucketV1",
    "REGISTERED_WORLD_SIZE",
    "TensorChunkV1",
    "bucket_plan_receipt_v1",
    "build_canonical_bucket_plan_v1",
    "canonical_json_bytes",
    "canonical_receipt_bytes",
    "object_sha256",
]
