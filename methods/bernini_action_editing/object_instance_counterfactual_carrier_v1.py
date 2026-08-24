#!/usr/bin/env python3
"""Fail-closed pure-tensor carrier for one counterfactual patient instance.

The core transports only the source-patient residual relative to a clean
bone-removed origin::

    z_cf = z0 + T(mask * (zs - z0))

``T`` is not a learned or interpolating operator.  It is the exact per-phase
token bijection supplied by :class:`PhaseCorrespondence`.  Before lift every
pair must be the identity.  During lift and hold, source-only origin tokens
remain byte-identical to ``z0``.  Every non-target token also remains
byte-identical to ``z0``.  The transported target residual is copied without
dtype conversion and is checked byte-for-byte against its source residual.

This file deliberately contains no renderer, scheduler, model forward,
distributed collective, file I/O, or visual-success assertion.  Tensor
digests bind caller-provided authorities; they do not authenticate the
upstream video, mask, or correspondence author.  Arbitrary caller storage
backing also cannot be authenticated through PyTorch's public API, so caller
values are byte-pinned and copied into module-minted private snapshots before
any carrier arithmetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Sequence


SCHEMA_VERSION = "bernini-object-instance-counterfactual-carrier-v1"
TRACE_SCHEMA_VERSION = (
    "bernini-object-instance-counterfactual-carrier-trace-v1"
)
SOURCE_ROLE = "exact_source_patient_zs"
ORIGIN_ROLE = "clean_bone_removed_origin_z0"
PACKED_CHANNELS = 64
PHASE_REGIMES = ("pre_lift", "lift", "hold")
_PHASE_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ObjectInstanceCounterfactualCarrierError(RuntimeError):
    """Raised before an ambiguous or mutated carrier can be returned."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production has torch
        raise ObjectInstanceCounterfactualCarrierError(
            "counterfactual carrier requires torch"
        ) from error
    return torch


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ObjectInstanceCounterfactualCarrierError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def tensor_core_contract() -> dict[str, Any]:
    """Return the immutable scientific and numerical boundary of this core."""

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "pure_tensor_counterfactual_patient_carrier",
        "equation": "z_cf=z0+T(mask*(zs-z0))",
        "packed_layout": "B,N,64",
        "source_role": SOURCE_ROLE,
        "origin_role": ORIGIN_ROLE,
        "transport": "exact_one_to_one_token_index_copy_no_interpolation",
        "pre_lift_transport": "identity",
        "pre_lift_output_identity": (
            "formula_observation_only_not_required_byte_equal_to_zs"
        ),
        "lift_hold_source_target_overlap": "forbidden",
        "lift_hold_origin": "byte_exact_z0_on_all_source_tokens",
        "target_residual": "byte_exact_masked_zs_minus_z0_addend",
        "target_value": "z0_target_plus_one_native_dtype_residual_add",
        "complement": "byte_exact_z0",
        "phase_partition": "target_disjoint_union_complement_equals_phase",
        "global_phase_partition": "all_phases_disjoint_union_all_packed_tokens",
        "single_target_occupancy": True,
        "source_aux_content_equality_allowed": False,
        "caller_backing_independence_authenticated": False,
        "caller_inputs_copied_to_private_snapshots": True,
        "working_snapshot_storage_alias_allowed": False,
        "nonfinite_allowed": False,
        "renderer_integration": False,
        "model_integration": False,
        "visual_success_claimed": False,
        "upstream_provenance_authenticated": False,
        "authority_boundary": (
            "caller_pinned_exact_values_copied_to_private_snapshots_and_plan_digest"
        ),
    }


@dataclass(frozen=True)
class PhaseCorrespondence:
    """One phase's exact source-to-target patient-token bijection.

    ``phase_tokens`` owns every packed token in this phase.  The target side
    of ``correspondence`` and ``target_complement`` must be its exact disjoint
    partition.  Source and target indices must each be unique.  All index
    containers are exact tuples so a mutable list cannot silently change the
    plan after review.
    """

    phase_index: int
    phase_id: str
    regime: str
    phase_tokens: tuple[int, ...]
    correspondence: tuple[tuple[int, int], ...]
    target_complement: tuple[int, ...]


@dataclass(frozen=True)
class TensorBytePin:
    """Tensor-free exact-byte identity for one dense tensor."""

    shape: tuple[int, ...]
    dtype: str
    device_type: str
    byte_count: int
    raw_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device_type": self.device_type,
            "byte_count": self.byte_count,
            "raw_sha256": self.raw_sha256,
        }


@dataclass(frozen=True)
class PatientCarrierAuthority:
    """Tensor-free seal replayed immediately before carrier construction."""

    schema_version: str
    source_role: str
    origin_role: str
    source: TensorBytePin
    bone_removed_origin: TensorBytePin
    source_mask: TensorBytePin
    phase_count: int
    packed_token_count: int
    selected_source_token_count: int
    phase_plan_sha256: str
    authority_digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_role": self.source_role,
            "origin_role": self.origin_role,
            "source": self.source.as_dict(),
            "bone_removed_origin": self.bone_removed_origin.as_dict(),
            "source_mask": self.source_mask.as_dict(),
            "phase_count": self.phase_count,
            "packed_token_count": self.packed_token_count,
            "selected_source_token_count": self.selected_source_token_count,
            "phase_plan_sha256": self.phase_plan_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.payload(), "authority_digest": self.authority_digest}


@dataclass(frozen=True)
class PhaseCarrierTrace:
    """Tensor-free construction checks for one phase."""

    phase_index: int
    phase_id: str
    regime: str
    phase_token_count: int
    correspondence_count: int
    target_token_count: int
    complement_token_count: int
    origin_token_count: int
    correspondence_sha256: str
    partition_sha256: str
    pre_lift_identity: bool
    pre_lift_output_byte_equal_source: bool | None
    target_residual_byte_exact: bool
    complement_byte_exact_z0: bool
    origin_byte_exact_z0: bool
    single_target_occupancy: bool


@dataclass(frozen=True)
class CounterfactualCarrierTrace:
    """Serializable audit receipt with no tensor or storage references."""

    schema_version: str
    authority_digest: str
    packed_shape: tuple[int, ...]
    dtype: str
    device_type: str
    selected_source_token_count: int
    source_raw_sha256: str
    bone_removed_origin_raw_sha256: str
    source_mask_raw_sha256: str
    phase_plan_sha256: str
    patient_residual_raw_sha256: str
    transported_residual_raw_sha256: str
    counterfactual_raw_sha256: str
    phases: tuple[PhaseCarrierTrace, ...]
    source_is_not_aux: bool
    caller_inputs_copied_to_private_snapshots: bool
    caller_backing_independence_authenticated: bool
    working_snapshot_storages_pairwise_distinct: bool
    output_storages_fresh_and_pairwise_distinct: bool
    working_snapshots_unmutated: bool
    all_target_residuals_byte_exact: bool
    all_complements_byte_exact_z0: bool
    all_lift_hold_origins_byte_exact_z0: bool
    single_target_occupancy: bool
    renderer_integration: bool
    visual_success_claimed: bool
    trace_digest: str

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("trace_digest")
        value["packed_shape"] = list(self.packed_shape)
        return value

    def as_dict(self) -> dict[str, Any]:
        return {**self.payload(), "trace_digest": self.trace_digest}


@dataclass(frozen=True)
class PatientCarrierResult:
    """Counterfactual tensor, exact transported addend, and tensor-free trace."""

    counterfactual: Any
    transported_residual: Any
    trace: CounterfactualCarrierTrace

    def audit_receipt(self) -> dict[str, Any]:
        return self.trace.as_dict()


@dataclass(frozen=True)
class _ValidatedPhase:
    phase: PhaseCorrespondence
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    origin: tuple[int, ...]
    correspondence_sha256: str
    partition_sha256: str


def _raw_tensor_bytes(value: Any, *, label: str) -> bytes:
    torch = _torch()
    try:
        snapshot = value.detach().contiguous().cpu()
        raw = snapshot.view(torch.uint8).numpy().tobytes(order="C")
    except Exception as error:
        raise ObjectInstanceCounterfactualCarrierError(
            f"cannot expose exact bytes for {label}"
        ) from error
    expected = int(snapshot.numel()) * int(snapshot.element_size())
    if len(raw) != expected:
        raise ObjectInstanceCounterfactualCarrierError(
            f"{label} raw byte length differs"
        )
    return raw


def tensor_raw_sha256(value: Any) -> str:
    """Return SHA-256 of the tensor's exact contiguous bytes, without casting."""

    return hashlib.sha256(_raw_tensor_bytes(value, label="tensor")).hexdigest()


def _tensor_pin(value: Any, *, label: str) -> TensorBytePin:
    raw = _raw_tensor_bytes(value, label=label)
    return TensorBytePin(
        shape=tuple(int(item) for item in value.shape),
        dtype=str(value.dtype),
        device_type=value.device.type,
        byte_count=len(raw),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _validate_pin(pin: Any, *, label: str) -> None:
    if type(pin) is not TensorBytePin:
        raise ObjectInstanceCounterfactualCarrierError(
            f"{label} must be an exact TensorBytePin"
        )
    if (
        type(pin.shape) is not tuple
        or not pin.shape
        or any(type(item) is not int or item <= 0 for item in pin.shape)
        or type(pin.dtype) is not str
        or not pin.dtype.startswith("torch.")
        or type(pin.device_type) is not str
        or pin.device_type not in {"cpu", "cuda"}
        or type(pin.byte_count) is not int
        or pin.byte_count <= 0
        or type(pin.raw_sha256) is not str
        or _SHA256.fullmatch(pin.raw_sha256) is None
    ):
        raise ObjectInstanceCounterfactualCarrierError(f"{label} pin differs")


def _storage_identity_and_interval(
    value: Any,
) -> tuple[tuple[str, int | None, int], tuple[str, int | None, int, int]]:
    """Return allocation identity and the tensor's occupied byte interval.

    Allocation identity alone is insufficient: two ``torch.frombuffer``
    tensors can expose overlapping slices of one external buffer through two
    distinct Storage objects.  All admitted tensors are contiguous, so the
    half-open ``[data_ptr, data_ptr + numel*element_size)`` interval is exact.
    """

    try:
        storage = value.untyped_storage()
    except AttributeError:  # pragma: no cover - old torch fallback
        storage = value.storage()
    device_type = value.device.type
    device_index = value.device.index
    allocation_pointer = int(storage.data_ptr())
    start = int(value.data_ptr())
    byte_count = int(value.numel()) * int(value.element_size())
    end = start + byte_count
    return (
        (device_type, device_index, allocation_pointer),
        (device_type, device_index, start, end),
    )


def _require_distinct_storages(
    values: Sequence[tuple[str, Any]], *, label: str
) -> None:
    owners: dict[tuple[str, int | None, int], str] = {}
    intervals: list[tuple[str, int | None, int, int, str]] = []
    for name, value in values:
        key, interval = _storage_identity_and_interval(value)
        device_type, device_index, start, end = interval
        if key[2] == 0 or start == 0 or end <= start:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} {name} exposes a null storage pointer"
            )
        if key in owners:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} storage alias is forbidden: {owners[key]} and {name}"
            )
        for (
            previous_device_type,
            previous_device_index,
            previous_start,
            previous_end,
            previous_name,
        ) in intervals:
            if (
                device_type == previous_device_type
                and device_index == previous_device_index
                and start < previous_end
                and previous_start < end
            ):
                raise ObjectInstanceCounterfactualCarrierError(
                    f"{label} storage byte intervals overlap: "
                    f"{previous_name} and {name}"
                )
        owners[key] = name
        intervals.append((device_type, device_index, start, end, name))


def _validate_packed_inputs(
    source_packed: Any, bone_removed_packed: Any, source_mask: Any
) -> tuple[int, int]:
    torch = _torch()
    float_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    for label, value in (
        ("source_packed", source_packed),
        ("bone_removed_packed", bone_removed_packed),
    ):
        if (
            type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.device.type not in {"cpu", "cuda"}
            or value.dtype not in float_dtypes
            or value.ndim != 3
            or int(value.shape[0]) <= 0
            or int(value.shape[1]) <= 0
            or int(value.shape[2]) != PACKED_CHANNELS
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} must be detached contiguous finite floating [B,N,64]"
            )
    if (
        tuple(source_packed.shape) != tuple(bone_removed_packed.shape)
        or source_packed.dtype != bone_removed_packed.dtype
        or source_packed.device != bone_removed_packed.device
    ):
        raise ObjectInstanceCounterfactualCarrierError(
            "source and bone-removed origin tensor ABI differs"
        )
    token_count = int(source_packed.shape[1])
    if (
        type(source_mask) is not torch.Tensor
        or source_mask.layout != torch.strided
        or source_mask.device != source_packed.device
        or source_mask.dtype != torch.bool
        or source_mask.ndim != 1
        or tuple(source_mask.shape) != (token_count,)
        or not source_mask.is_contiguous()
        or source_mask.requires_grad
        or source_mask.grad_fn is not None
    ):
        raise ObjectInstanceCounterfactualCarrierError(
            "source_mask must be detached contiguous bool [N] on the packed device"
        )
    if _tensor_pin(source_packed, label="source_packed").raw_sha256 == _tensor_pin(
        bone_removed_packed, label="bone_removed_packed"
    ).raw_sha256:
        raise ObjectInstanceCounterfactualCarrierError(
            "source patient equals the bone-removed auxiliary origin"
        )
    return int(source_packed.shape[0]), token_count


def _exact_index_tuple(value: Any, *, label: str, upper: int) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise ObjectInstanceCounterfactualCarrierError(
            f"{label} must be one nonempty exact integer tuple"
        )
    if tuple(sorted(set(value))) != value:
        raise ObjectInstanceCounterfactualCarrierError(
            f"{label} must be sorted and unique"
        )
    if any(item < 0 or item >= upper for item in value):
        raise ObjectInstanceCounterfactualCarrierError(
            f"{label} contains an out-of-range token"
        )
    return value


def _validate_phases(
    phases: Any, *, token_count: int
) -> tuple[tuple[_ValidatedPhase, ...], dict[str, Any]]:
    if type(phases) is not tuple or not phases:
        raise ObjectInstanceCounterfactualCarrierError(
            "phases must be one nonempty exact tuple"
        )
    validated: list[_ValidatedPhase] = []
    phase_ids: set[str] = set()
    globally_owned: set[int] = set()
    regimes_seen: set[str] = set()
    previous_regime_rank = -1
    plan_rows: list[dict[str, Any]] = []
    for expected_index, phase in enumerate(phases):
        label = f"phase[{expected_index}]"
        if type(phase) is not PhaseCorrespondence:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} must be an exact PhaseCorrespondence"
            )
        if phase.phase_index != expected_index or type(phase.phase_index) is not int:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} index ordering differs"
            )
        if (
            type(phase.phase_id) is not str
            or _PHASE_ID.fullmatch(phase.phase_id) is None
            or phase.phase_id in phase_ids
        ):
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} phase_id is invalid or duplicated"
            )
        phase_ids.add(phase.phase_id)
        if phase.regime not in PHASE_REGIMES or type(phase.regime) is not str:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} regime differs"
            )
        regime_rank = PHASE_REGIMES.index(phase.regime)
        if regime_rank < previous_regime_rank:
            raise ObjectInstanceCounterfactualCarrierError(
                "phase regimes are not monotone pre_lift/lift/hold"
            )
        previous_regime_rank = regime_rank
        regimes_seen.add(phase.regime)
        phase_tokens = _exact_index_tuple(
            phase.phase_tokens, label=f"{label}.phase_tokens", upper=token_count
        )
        complement = _exact_index_tuple(
            phase.target_complement,
            label=f"{label}.target_complement",
            upper=token_count,
        )
        if type(phase.correspondence) is not tuple or not phase.correspondence:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label}.correspondence must be one nonempty exact tuple"
            )
        pairs: list[tuple[int, int]] = []
        for pair_index, pair in enumerate(phase.correspondence):
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(item) is not int for item in pair)
            ):
                raise ObjectInstanceCounterfactualCarrierError(
                    f"{label}.correspondence[{pair_index}] differs"
                )
            source_index, target_index = pair
            if not 0 <= source_index < token_count or not 0 <= target_index < token_count:
                raise ObjectInstanceCounterfactualCarrierError(
                    f"{label} correspondence escapes packed tokens"
                )
            pairs.append(pair)
        if tuple(sorted(pairs)) != tuple(pairs):
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} correspondence must be canonical source order"
            )
        sources = tuple(pair[0] for pair in pairs)
        targets = tuple(pair[1] for pair in pairs)
        if (
            len(set(pairs)) != len(pairs)
            or len(set(sources)) != len(sources)
            or len(set(targets)) != len(targets)
        ):
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} correspondence is duplicate or non-bijective"
            )
        phase_set = set(phase_tokens)
        source_set = set(sources)
        target_set = set(targets)
        complement_set = set(complement)
        if not source_set.issubset(phase_set) or not target_set.issubset(phase_set):
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} correspondence escapes its phase"
            )
        if target_set & complement_set:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} target/complement partition overlaps"
            )
        if target_set | complement_set != phase_set:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} target/complement partition has a gap or foreign token"
            )
        if phase.regime == "pre_lift" and any(
            source != target for source, target in pairs
        ):
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} pre_lift transport is not identity"
            )
        if phase.regime in {"lift", "hold"} and source_set & target_set:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{label} lift/hold source and target supports overlap"
            )
        origin = tuple(sorted(source_set)) if phase.regime in {"lift", "hold"} else ()
        overlap = globally_owned & phase_set
        if overlap:
            raise ObjectInstanceCounterfactualCarrierError(
                "global phase partition overlaps"
            )
        globally_owned.update(phase_set)
        correspondence_value = [[source, target] for source, target in pairs]
        partition_value = {
            "phase_tokens": list(phase_tokens),
            "targets": sorted(target_set),
            "target_complement": list(complement),
        }
        correspondence_sha256 = _object_sha256(correspondence_value)
        partition_sha256 = _object_sha256(partition_value)
        row = {
            "phase_index": phase.phase_index,
            "phase_id": phase.phase_id,
            "regime": phase.regime,
            "phase_tokens": list(phase_tokens),
            "correspondence": correspondence_value,
            "target_complement": list(complement),
        }
        plan_rows.append(row)
        validated.append(
            _ValidatedPhase(
                phase=phase,
                sources=sources,
                targets=targets,
                origin=origin,
                correspondence_sha256=correspondence_sha256,
                partition_sha256=partition_sha256,
            )
        )
    expected_global = set(range(token_count))
    if globally_owned != expected_global:
        raise ObjectInstanceCounterfactualCarrierError(
            "global phase partition has a gap or foreign token"
        )
    if regimes_seen != set(PHASE_REGIMES):
        raise ObjectInstanceCounterfactualCarrierError(
            "phase plan must contain pre_lift, lift, and hold"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "packed_token_count": token_count,
        "phases": plan_rows,
    }
    return tuple(validated), payload


def _true_mask_indices(source_mask: Any) -> tuple[int, ...]:
    torch = _torch()
    return tuple(
        int(item)
        for item in torch.nonzero(source_mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def _validate_mask_against_phases(
    source_mask: Any, validated: Sequence[_ValidatedPhase]
) -> tuple[int, ...]:
    expected = tuple(sorted({item for row in validated for item in row.sources}))
    observed = _true_mask_indices(source_mask)
    if observed != expected:
        raise ObjectInstanceCounterfactualCarrierError(
            "source mask differs from exact phase correspondence domains"
        )
    if not expected:
        raise ObjectInstanceCounterfactualCarrierError(
            "source patient mask is empty"
        )
    return expected


def _selected_residual_nonzero(
    source_packed: Any, bone_removed_packed: Any, selected: Sequence[int]
) -> None:
    torch = _torch()
    index = torch.tensor(selected, dtype=torch.int64, device=source_packed.device)
    residual = source_packed.index_select(1, index) - bone_removed_packed.index_select(
        1, index
    )
    if not bool(torch.isfinite(residual).all().item()):
        raise ObjectInstanceCounterfactualCarrierError(
            "selected patient residual is non-finite"
        )
    if int(torch.count_nonzero(residual).item()) <= 0:
        raise ObjectInstanceCounterfactualCarrierError(
            "selected source patient contains no residual relative to z0"
        )


def _authority_from_current(
    *,
    source_packed: Any,
    bone_removed_packed: Any,
    source_mask: Any,
    phase_count: int,
    token_count: int,
    selected_count: int,
    phase_plan_sha256: str,
) -> PatientCarrierAuthority:
    provisional = PatientCarrierAuthority(
        schema_version=SCHEMA_VERSION,
        source_role=SOURCE_ROLE,
        origin_role=ORIGIN_ROLE,
        source=_tensor_pin(source_packed, label="source_packed"),
        bone_removed_origin=_tensor_pin(
            bone_removed_packed, label="bone_removed_packed"
        ),
        source_mask=_tensor_pin(source_mask, label="source_mask"),
        phase_count=phase_count,
        packed_token_count=token_count,
        selected_source_token_count=selected_count,
        phase_plan_sha256=phase_plan_sha256,
        authority_digest="",
    )
    digest = _object_sha256(provisional.payload())
    return PatientCarrierAuthority(
        **{**provisional.__dict__, "authority_digest": digest}
    )


def _validate_authority(authority: Any) -> None:
    if type(authority) is not PatientCarrierAuthority:
        raise ObjectInstanceCounterfactualCarrierError(
            "authority must be an exact PatientCarrierAuthority"
        )
    _validate_pin(authority.source, label="authority.source")
    _validate_pin(
        authority.bone_removed_origin, label="authority.bone_removed_origin"
    )
    _validate_pin(authority.source_mask, label="authority.source_mask")
    if (
        type(authority.schema_version) is not str
        or authority.schema_version != SCHEMA_VERSION
        or type(authority.source_role) is not str
        or authority.source_role != SOURCE_ROLE
        or type(authority.origin_role) is not str
        or authority.origin_role != ORIGIN_ROLE
        or type(authority.phase_count) is not int
        or authority.phase_count <= 0
        or type(authority.packed_token_count) is not int
        or authority.packed_token_count <= 0
        or type(authority.selected_source_token_count) is not int
        or authority.selected_source_token_count <= 0
        or authority.selected_source_token_count > authority.packed_token_count
        or type(authority.phase_plan_sha256) is not str
        or _SHA256.fullmatch(authority.phase_plan_sha256) is None
        or type(authority.authority_digest) is not str
        or _SHA256.fullmatch(authority.authority_digest) is None
        or _object_sha256(authority.payload()) != authority.authority_digest
    ):
        raise ObjectInstanceCounterfactualCarrierError(
            "patient carrier authority fields or digest differ"
        )


def seal_patient_carrier_authority(
    *,
    source_packed: Any,
    bone_removed_packed: Any,
    source_mask: Any,
    phases: tuple[PhaseCorrespondence, ...],
    source_role: str = SOURCE_ROLE,
    origin_role: str = ORIGIN_ROLE,
) -> PatientCarrierAuthority:
    """Validate and seal exact caller inputs into a tensor-free authority.

    The returned seal should be retained independently and replayed by
    :func:`build_counterfactual_patient_carrier`.  Supplying a changed source,
    origin, mask, or phase plan then fails before any result is returned.
    """

    if source_role != SOURCE_ROLE or type(source_role) is not str:
        raise ObjectInstanceCounterfactualCarrierError(
            "source role is not the exact source-patient role"
        )
    if origin_role != ORIGIN_ROLE or type(origin_role) is not str:
        raise ObjectInstanceCounterfactualCarrierError(
            "origin role is not the clean bone-removed role"
        )
    (
        source_snapshot,
        bone_removed_snapshot,
        mask_snapshot,
    ), _ = _snapshot_packed_inputs(
        source_packed=source_packed,
        bone_removed_packed=bone_removed_packed,
        source_mask=source_mask,
    )
    _, token_count = _validate_packed_inputs(
        source_snapshot, bone_removed_snapshot, mask_snapshot
    )
    validated, plan_payload = _validate_phases(phases, token_count=token_count)
    selected = _validate_mask_against_phases(mask_snapshot, validated)
    _selected_residual_nonzero(
        source_snapshot, bone_removed_snapshot, selected
    )
    return _authority_from_current(
        source_packed=source_snapshot,
        bone_removed_packed=bone_removed_snapshot,
        source_mask=mask_snapshot,
        phase_count=len(validated),
        token_count=token_count,
        selected_count=len(selected),
        phase_plan_sha256=_object_sha256(plan_payload),
    )


def _byte_equal(left: Any, right: Any) -> bool:
    torch = _torch()
    if (
        tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _index_tensor(indices: Sequence[int], *, device: Any) -> Any:
    torch = _torch()
    return torch.tensor(tuple(indices), dtype=torch.int64, device=device)


def _tensor_version_or_none(value: Any) -> int | None:
    """Return the mutation counter when torch exposes one.

    Tensors allocated inside ``torch.inference_mode`` intentionally have no
    version counter.  Exact pre/post byte pins remain mandatory in that case.
    """

    try:
        return int(value._version)
    except RuntimeError:
        return None


def _snapshot_packed_inputs(
    *,
    source_packed: Any,
    bone_removed_packed: Any,
    source_mask: Any,
) -> tuple[tuple[Any, Any, Any], tuple[TensorBytePin, ...]]:
    """Copy caller values into module-minted private working snapshots.

    Python/PyTorch exposes no non-mutating proof that arbitrary external
    tensor backing is private.  Separate ``mmap`` objects can map overlapping
    file bytes at unrelated virtual addresses.  The core therefore treats
    caller tensors strictly as byte-valued inputs: it pins them before and
    after cloning, then consumes only three independently allocated clones.
    It does not claim that the caller's original backing stores were
    independent.
    """

    torch = _torch()
    _validate_packed_inputs(source_packed, bone_removed_packed, source_mask)
    caller_values = (
        ("source_packed", source_packed),
        ("bone_removed_packed", bone_removed_packed),
        ("source_mask", source_mask),
    )
    caller_versions_before = tuple(
        _tensor_version_or_none(value) for _, value in caller_values
    )
    caller_pins_before = tuple(
        _tensor_pin(value, label=name) for name, value in caller_values
    )
    snapshots = tuple(
        value.detach().clone(memory_format=torch.contiguous_format)
        for _, value in caller_values
    )
    caller_versions_after = tuple(
        _tensor_version_or_none(value) for _, value in caller_values
    )
    caller_pins_after = tuple(
        _tensor_pin(value, label=name) for name, value in caller_values
    )
    if (
        caller_versions_after != caller_versions_before
        or caller_pins_after != caller_pins_before
    ):
        raise ObjectInstanceCounterfactualCarrierError(
            "one or more caller inputs changed while private snapshots were made"
        )

    source_snapshot, bone_removed_snapshot, mask_snapshot = snapshots
    _validate_packed_inputs(
        source_snapshot, bone_removed_snapshot, mask_snapshot
    )
    snapshot_values = (
        ("source_snapshot", source_snapshot),
        ("bone_removed_snapshot", bone_removed_snapshot),
        ("mask_snapshot", mask_snapshot),
    )
    _require_distinct_storages(snapshot_values, label="working snapshot")
    snapshot_pins = tuple(
        _tensor_pin(value, label=name) for name, value in snapshot_values
    )
    if snapshot_pins != caller_pins_before:
        raise ObjectInstanceCounterfactualCarrierError(
            "private working snapshots differ from caller input bytes"
        )
    return (
        (source_snapshot, bone_removed_snapshot, mask_snapshot),
        snapshot_pins,
    )


def build_counterfactual_patient_carrier(
    *,
    source_packed: Any,
    bone_removed_packed: Any,
    source_mask: Any,
    phases: tuple[PhaseCorrespondence, ...],
    authority: PatientCarrierAuthority,
) -> PatientCarrierResult:
    """Construct ``z0 + T(mask * (zs-z0))`` under an exact replayed seal."""

    torch = _torch()
    _validate_authority(authority)
    (
        source_packed,
        bone_removed_packed,
        source_mask,
    ), pins_before = _snapshot_packed_inputs(
        source_packed=source_packed,
        bone_removed_packed=bone_removed_packed,
        source_mask=source_mask,
    )
    _, token_count = _validate_packed_inputs(
        source_packed, bone_removed_packed, source_mask
    )
    validated, plan_payload = _validate_phases(phases, token_count=token_count)
    selected = _validate_mask_against_phases(source_mask, validated)
    _selected_residual_nonzero(source_packed, bone_removed_packed, selected)
    plan_sha256 = _object_sha256(plan_payload)
    observed_authority = _authority_from_current(
        source_packed=source_packed,
        bone_removed_packed=bone_removed_packed,
        source_mask=source_mask,
        phase_count=len(validated),
        token_count=token_count,
        selected_count=len(selected),
        phase_plan_sha256=plan_sha256,
    )
    if observed_authority != authority:
        raise ObjectInstanceCounterfactualCarrierError(
            "runtime source/origin/mask/phase inputs differ from sealed authority"
        )

    working_values = (
        ("source_snapshot", source_packed),
        ("bone_removed_snapshot", bone_removed_packed),
        ("mask_snapshot", source_mask),
    )
    versions_before = tuple(
        _tensor_version_or_none(value) for _, value in working_values
    )
    if pins_before != (
        authority.source,
        authority.bone_removed_origin,
        authority.source_mask,
    ):
        raise ObjectInstanceCounterfactualCarrierError(
            "carrier input changed after authority replay"
        )

    patient_residual = source_packed - bone_removed_packed
    broadcast_mask = source_mask.reshape(1, token_count, 1).to(
        dtype=source_packed.dtype
    )
    masked_residual = (patient_residual * broadcast_mask).contiguous()
    selected_index = _index_tensor(selected, device=source_packed.device)
    if not _byte_equal(
        masked_residual.index_select(1, selected_index),
        patient_residual.index_select(1, selected_index),
    ):
        raise ObjectInstanceCounterfactualCarrierError(
            "mask changed selected source residual bytes"
        )

    transported = torch.zeros_like(bone_removed_packed).contiguous()
    counterfactual = bone_removed_packed.detach().clone().contiguous()
    phase_traces: list[PhaseCarrierTrace] = []
    for row in validated:
        source_index = _index_tensor(row.sources, device=source_packed.device)
        target_index = _index_tensor(row.targets, device=source_packed.device)
        complement_index = _index_tensor(
            row.phase.target_complement, device=source_packed.device
        )
        source_residual = masked_residual.index_select(1, source_index).contiguous()
        transported.index_copy_(1, target_index, source_residual)
        target_origin = bone_removed_packed.index_select(1, target_index)
        target_value = (target_origin + source_residual).contiguous()
        counterfactual.index_copy_(1, target_index, target_value)

        target_exact = _byte_equal(
            transported.index_select(1, target_index), source_residual
        )
        complement_exact = _byte_equal(
            counterfactual.index_select(1, complement_index),
            bone_removed_packed.index_select(1, complement_index),
        )
        if row.origin:
            origin_index = _index_tensor(
                row.origin, device=source_packed.device
            )
            origin_exact = _byte_equal(
                counterfactual.index_select(1, origin_index),
                bone_removed_packed.index_select(1, origin_index),
            )
        else:
            origin_exact = True
        pre_lift_identity = row.phase.regime != "pre_lift" or all(
            source == target
            for source, target in row.phase.correspondence
        )
        pre_lift_output_byte_equal_source = (
            _byte_equal(
                counterfactual.index_select(1, target_index),
                source_packed.index_select(1, source_index),
            )
            if row.phase.regime == "pre_lift"
            else None
        )
        if not target_exact:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{row.phase.phase_id} target residual is not byte exact"
            )
        if not complement_exact:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{row.phase.phase_id} complement differs from z0"
            )
        if row.phase.regime in {"lift", "hold"} and not origin_exact:
            raise ObjectInstanceCounterfactualCarrierError(
                f"{row.phase.phase_id} source origin differs from z0"
            )
        phase_traces.append(
            PhaseCarrierTrace(
                phase_index=row.phase.phase_index,
                phase_id=row.phase.phase_id,
                regime=row.phase.regime,
                phase_token_count=len(row.phase.phase_tokens),
                correspondence_count=len(row.sources),
                target_token_count=len(row.targets),
                complement_token_count=len(row.phase.target_complement),
                origin_token_count=len(row.origin),
                correspondence_sha256=row.correspondence_sha256,
                partition_sha256=row.partition_sha256,
                pre_lift_identity=pre_lift_identity,
                pre_lift_output_byte_equal_source=(
                    pre_lift_output_byte_equal_source
                ),
                target_residual_byte_exact=target_exact,
                complement_byte_exact_z0=complement_exact,
                origin_byte_exact_z0=origin_exact,
                single_target_occupancy=True,
            )
        )

    if not bool(torch.isfinite(patient_residual).all().item()):
        raise ObjectInstanceCounterfactualCarrierError(
            "patient residual contains non-finite values"
        )
    if not bool(torch.isfinite(counterfactual).all().item()):
        raise ObjectInstanceCounterfactualCarrierError(
            "counterfactual overflowed to a non-finite value"
        )
    _require_distinct_storages(
        (
            *working_values,
            ("transported_residual", transported),
            ("counterfactual", counterfactual),
        ),
        label="input/output",
    )
    versions_after = tuple(
        _tensor_version_or_none(value) for _, value in working_values
    )
    pins_after = tuple(
        _tensor_pin(value, label=name) for name, value in working_values
    )
    if versions_after != versions_before or pins_after != pins_before:
        raise ObjectInstanceCounterfactualCarrierError(
            "one or more private working snapshots mutated during construction"
        )

    trace_without_digest = CounterfactualCarrierTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        authority_digest=authority.authority_digest,
        packed_shape=tuple(int(item) for item in source_packed.shape),
        dtype=str(source_packed.dtype),
        device_type=source_packed.device.type,
        selected_source_token_count=len(selected),
        source_raw_sha256=pins_before[0].raw_sha256,
        bone_removed_origin_raw_sha256=pins_before[1].raw_sha256,
        source_mask_raw_sha256=pins_before[2].raw_sha256,
        phase_plan_sha256=plan_sha256,
        patient_residual_raw_sha256=tensor_raw_sha256(patient_residual),
        transported_residual_raw_sha256=tensor_raw_sha256(transported),
        counterfactual_raw_sha256=tensor_raw_sha256(counterfactual),
        phases=tuple(phase_traces),
        source_is_not_aux=True,
        caller_inputs_copied_to_private_snapshots=True,
        caller_backing_independence_authenticated=False,
        working_snapshot_storages_pairwise_distinct=True,
        output_storages_fresh_and_pairwise_distinct=True,
        working_snapshots_unmutated=True,
        all_target_residuals_byte_exact=all(
            item.target_residual_byte_exact for item in phase_traces
        ),
        all_complements_byte_exact_z0=all(
            item.complement_byte_exact_z0 for item in phase_traces
        ),
        all_lift_hold_origins_byte_exact_z0=all(
            item.regime == "pre_lift" or item.origin_byte_exact_z0
            for item in phase_traces
        ),
        single_target_occupancy=True,
        renderer_integration=False,
        visual_success_claimed=False,
        trace_digest="",
    )
    trace_digest = _object_sha256(trace_without_digest.payload())
    trace = CounterfactualCarrierTrace(
        **{**trace_without_digest.__dict__, "trace_digest": trace_digest}
    )
    return PatientCarrierResult(
        counterfactual=counterfactual,
        transported_residual=transported,
        trace=trace,
    )


__all__ = [
    "ORIGIN_ROLE",
    "PACKED_CHANNELS",
    "PHASE_REGIMES",
    "SCHEMA_VERSION",
    "SOURCE_ROLE",
    "TRACE_SCHEMA_VERSION",
    "CounterfactualCarrierTrace",
    "ObjectInstanceCounterfactualCarrierError",
    "PatientCarrierAuthority",
    "PatientCarrierResult",
    "PhaseCarrierTrace",
    "PhaseCorrespondence",
    "TensorBytePin",
    "build_counterfactual_patient_carrier",
    "seal_patient_carrier_authority",
    "tensor_core_contract",
    "tensor_raw_sha256",
]
