#!/usr/bin/env python3
"""Fail-closed joint lifecycle for the two SAIC Bernini adapters.

The source-anchor and temporal-action handles intentionally certify an
*exclusive* transformer-wide trainable set.  Consequently, installing one
handle and then naively installing the other is impossible: the second
installer sees the first adapter's trainable parameters.  Merely freezing the
first adapter before installing the second is also insufficient because the
first legacy handle can no longer emit its own receipt in that gauge.

This module resolves that contract conflict without weakening either adapter.
It records the source handle's valid receipt while it exclusively owns the
trainable gauge, loads and binds its closed state, freezes it, and only then
installs the motion handle and records its valid exclusive receipt.  The joint
certificate thereafter owns the composition:

* A Stage-B birth gauge may mark only registered motion parameters as
  ``requires_grad``, but this lifecycle object authorizes neither optimizer
  access nor any parameter update.
* Inference freezes both adapters.
* The live transformer parameter/module IDs must be the strict union of the
  original vendor tree and the two registered adapter additions.
* Vendor, source, and motion parameter bytes are re-hashed on every audit
  against a process-local registry-issued birth seal; supported public fields
  cannot re-sign an altered state.
* Restoration is motion first, source second, and must recreate the exact
  original vendor named-module and named-parameter identities.

The threat model is deliberately narrow: this is fail-closed lifecycle integrity
against supported public-API misuse and ordinary tensor/module alias mutation in
one cooperative Python process.  It is not a sandbox or a security boundary
against arbitrary same-process reflection, private-function calls, closure-cell
inspection, monkeypatching, or native memory modification.

Every audit intentionally re-hashes all logical transformer parameter bytes on
CPU.  Its cost is ``O(total parameter bytes)`` per process, includes device-to-
CPU synchronization for accelerator weights, and is suitable only for sparse
lifecycle boundaries—not an optimizer step or denoising-loop inner path.

This v1 object deliberately manages non-authoritative adapter lifecycle only.  The native
sampler wrappers need the diffusion/renderer object (not just its transformer)
and have their own inner/outer install order.  They must be composed by the
next runtime seam after this certificate is established; no runtime success is
claimed here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
from threading import RLock
from typing import Any, Iterator, Mapping, Optional
from weakref import WeakKeyDictionary, ref

import torch
from torch import nn

if __package__:
    from . import saic_source_anchor_adapter_v1 as source_anchor
    from . import saic_temporal_action_operator_v2 as temporal_action
else:  # Direct import from methods/bernini_action_editing.
    import saic_source_anchor_adapter_v1 as source_anchor
    import saic_temporal_action_operator_v2 as temporal_action


SCHEMA_VERSION = "bernini-saic-joint-composition-v1"
CLASSIFICATION = (
    "adapter_joint_lifecycle/process_local_public_api_integrity/"
    "no_parameter_update_native_runtime_or_training_authority"
)
STAGE_B_TRAIN = "stage_b_train"
INFERENCE = "inference"
ALLOWED_MODES = frozenset({STAGE_B_TRAIN, INFERENCE})
NATIVE_RUNTIME_STATUS = "pending_exact_diffusion_runtime_composition"

_CONSTRUCTION_MINT = object()
_RUNTIME_LEASE_MINT = object()
_TRUST_MINT = object()
_RESTORE_LIVE = "source_and_motion_active"
_RESTORE_MOTION_REMOVED = "motion_removed_source_active"
_RESTORE_CHILDREN_REMOVED = "source_and_motion_removed_vendor_unverified"
_RESTORE_COMPLETE = "complete_vendor_restored"
_RESTORE_AMBIGUOUS = "ambiguous_registered_slots"


class SAICJointCompositionError(RuntimeError):
    """Raised before a structurally or numerically ambiguous joint state is used."""


class SAICJointCompositionRestoreError(SAICJointCompositionError):
    """A resumable restore attempt failed or completed with recorded errors."""

    def __init__(
        self,
        message: str,
        *,
        receipt: Mapping[str, Any],
        root_cause: BaseException,
    ) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)
        self.root_cause = root_cause


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SAICJointCompositionError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sealed_copy(receipt: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping):
        raise SAICJointCompositionError(f"{label} receipt must be a mapping")
    try:
        copied = json.loads(_canonical_json(dict(receipt)).decode("ascii"))
    except Exception as error:
        raise SAICJointCompositionError(f"{label} receipt is not canonical") from error
    digest = copied.pop("digest", None)
    if type(digest) is not str or digest != _object_sha256(copied):
        raise SAICJointCompositionError(f"{label} receipt seal differs")
    copied["digest"] = digest
    return copied


def _named_parameter_rows(module: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    try:
        rows = tuple(module.named_parameters(remove_duplicate=False))
    except TypeError as error:  # pragma: no cover - pinned torch supports this.
        raise SAICJointCompositionError(
            "torch named_parameters(remove_duplicate=False) is required"
        ) from error
    if not rows or len({name for name, _ in rows}) != len(rows):
        raise SAICJointCompositionError("transformer parameter names are not closed")
    aliases: dict[int, list[str]] = {}
    for name, parameter in rows:
        if type(name) is not str or not isinstance(parameter, nn.Parameter):
            raise SAICJointCompositionError("transformer parameter row differs")
        aliases.setdefault(id(parameter), []).append(name)
    repeated = [names for names in aliases.values() if len(names) != 1]
    if repeated:
        raise SAICJointCompositionError(
            f"transformer parameter alias detected: {repeated[0][:2]}"
        )
    return rows


def _named_module_rows(module: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    try:
        rows = tuple(module.named_modules(remove_duplicate=False))
    except TypeError as error:  # pragma: no cover - pinned torch supports this.
        raise SAICJointCompositionError(
            "torch named_modules(remove_duplicate=False) is required"
        ) from error
    if not rows or len({name for name, _ in rows}) != len(rows):
        raise SAICJointCompositionError("transformer module names are not closed")
    aliases: dict[int, list[str]] = {}
    for name, child in rows:
        if type(name) is not str or not isinstance(child, nn.Module):
            raise SAICJointCompositionError("transformer module row differs")
        aliases.setdefault(id(child), []).append(name)
    repeated = [names for names in aliases.values() if len(names) != 1]
    if repeated:
        raise SAICJointCompositionError(
            f"transformer module alias detected: {repeated[0][:2]}"
        )
    return rows


def _storage_pointer(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - older torch compatibility.
        return int(value.storage().data_ptr())


def _parameter_binding(parameter: nn.Parameter) -> tuple[Any, ...]:
    return (
        id(parameter),
        _storage_pointer(parameter),
        int(parameter.storage_offset()),
        tuple(map(int, parameter.shape)),
        tuple(map(int, parameter.stride())),
        str(parameter.dtype),
        str(parameter.device),
        str(parameter.layout),
        int(getattr(parameter, "_version", 0)),
    )


def _parameter_identity_rows(
    rows: tuple[tuple[str, nn.Parameter], ...],
) -> tuple[tuple[str, int], ...]:
    return tuple((name, id(parameter)) for name, parameter in rows)


def _module_identity_rows(
    rows: tuple[tuple[str, nn.Module], ...],
) -> tuple[tuple[str, int], ...]:
    return tuple((name, id(module)) for name, module in rows)


def _parameter_binding_map(
    rows: tuple[tuple[str, nn.Parameter], ...],
) -> Mapping[str, tuple[Any, ...]]:
    return {name: _parameter_binding(parameter) for name, parameter in rows}


def _state_sha256(
    named_parameters: tuple[tuple[str, nn.Parameter], ...], *, label: str
) -> str:
    if not named_parameters:
        raise SAICJointCompositionError(f"{label} state is empty")
    if len({name for name, _ in named_parameters}) != len(named_parameters):
        raise SAICJointCompositionError(f"{label} state names alias")
    if len({id(parameter) for _, parameter in named_parameters}) != len(
        named_parameters
    ):
        raise SAICJointCompositionError(f"{label} state parameter alias detected")
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda row: row[0]):
        if (
            type(name) is not str
            or name.encode("ascii", "strict").decode("ascii") != name
            or parameter.dtype != torch.float32
            or parameter.layout != torch.strided
            or not bool(torch.isfinite(parameter.detach()).all().item())
        ):
            raise SAICJointCompositionError(
                f"{label} parameter {name!r} must be finite strided FP32"
            )
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json(list(map(int, value.shape))))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parameter_content_sha256(
    named_parameters: tuple[tuple[str, nn.Parameter], ...], *, label: str
) -> str:
    """Hash logical tensor bytes independently of PyTorch version counters."""

    if not named_parameters:
        raise SAICJointCompositionError(f"{label} parameter content is empty")
    digest = hashlib.sha256()
    digest.update(label.encode("ascii", "strict"))
    for name, parameter in sorted(named_parameters, key=lambda row: row[0]):
        if (
            type(name) is not str
            or name.encode("ascii", "strict").decode("ascii") != name
            or not isinstance(parameter, nn.Parameter)
            or parameter.layout != torch.strided
        ):
            raise SAICJointCompositionError(
                f"{label} parameter content row differs"
            )
        value = parameter.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(map(int, value.shape))))
        digest.update(raw)
    return digest.hexdigest()


def _validate_closed_cpu_state(
    state: Mapping[str, torch.Tensor],
    expected: tuple[tuple[str, nn.Parameter], ...],
    *,
    label: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise SAICJointCompositionError(f"{label} state must be a mapping")
    expected_map = dict(expected)
    if set(state) != set(expected_map):
        missing = sorted(set(expected_map) - set(state))
        unexpected = sorted(set(state) - set(expected_map))
        raise SAICJointCompositionError(
            f"{label} state key closure differs: "
            f"missing={missing[:2]} unexpected={unexpected[:2]}"
        )
    normalized: dict[str, torch.Tensor] = {}
    for name, parameter in expected:
        value = state[name]
        if (
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.requires_grad
            or value.grad_fn is not None
            or not value.is_contiguous()
            or tuple(value.shape) != tuple(parameter.shape)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise SAICJointCompositionError(
                f"{label} state {name} must be exact-shape finite contiguous CPU FP32"
            )
        normalized[name] = value
    return normalized


def _load_closed_state(
    state: Mapping[str, torch.Tensor],
    expected: tuple[tuple[str, nn.Parameter], ...],
    *,
    label: str,
) -> Mapping[str, Any]:
    normalized = _validate_closed_cpu_state(state, expected, label=label)
    before = {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in expected
    }
    try:
        with torch.no_grad():
            for name, parameter in expected:
                parameter.copy_(normalized[name].to(device=parameter.device))
        state_digest = _state_sha256(expected, label=label)
    except Exception as error:
        with torch.no_grad():
            for name, parameter in expected:
                parameter.copy_(before[name].to(device=parameter.device))
        raise SAICJointCompositionError(f"failed to load {label} state") from error
    value = {
        "schema_version": SCHEMA_VERSION,
        "adapter": label,
        "closed_exact_key_set": True,
        "state_key_count": len(expected),
        "state_key_sha256": _object_sha256(sorted(normalized)),
        "state_tensor_sha256": state_digest,
    }
    return {**value, "digest": _object_sha256(value)}


def _source_named_parameters(
    handle: source_anchor.SAICSourceAnchorHandle,
) -> tuple[tuple[str, nn.Parameter], ...]:
    rows: list[tuple[str, nn.Parameter]] = []
    for index, wrapper in handle.q_wrappers:
        rows.extend(
            (
                (f"blocks.{index}.attn1.to_q.state_down.weight", wrapper.state_down.weight),
                (f"blocks.{index}.attn1.to_q.output_up.weight", wrapper.output_up.weight),
            )
        )
    for index, wrapper in handle.o_wrappers:
        rows.extend(
            (
                (
                    f"blocks.{index}.attn1.to_out.0.state_down.weight",
                    wrapper.state_down.weight,
                ),
                (
                    f"blocks.{index}.attn1.to_out.0.output_up.weight",
                    wrapper.output_up.weight,
                ),
            )
        )
    return tuple(rows)


def _motion_named_parameters(
    handle: temporal_action.SAICTemporalActionOperatorHandle,
) -> tuple[tuple[str, nn.Parameter], ...]:
    rows: list[tuple[str, nn.Parameter]] = []
    for index, wrapper in handle.q_wrappers:
        rows.extend(
            (
                (f"blocks.{index}.attn2.to_q.state_down.weight", wrapper.state_down.weight),
                (f"blocks.{index}.attn2.to_q.phase_gate.weight", wrapper.phase_gate.weight),
                (f"blocks.{index}.attn2.to_q.output_up.weight", wrapper.output_up.weight),
            )
        )
    for index, wrapper in handle.o_wrappers:
        rows.extend(
            (
                (
                    f"blocks.{index}.attn2.to_out.0.state_down.weight",
                    wrapper.state_down.weight,
                ),
                (
                    f"blocks.{index}.attn2.to_out.0.phase_gate.weight",
                    wrapper.phase_gate.weight,
                ),
                (
                    f"blocks.{index}.attn2.to_out.0.output_up.weight",
                    wrapper.output_up.weight,
                ),
            )
        )
    return tuple(rows)


def _registered_added_module_ids(
    wrappers: tuple[tuple[int, nn.Module], ...],
    *,
    preexisting_ids: set[int],
) -> set[int]:
    """Return only modules introduced by exact registered wrapper subtrees."""

    observed: set[int] = set()
    for _, wrapper in wrappers:
        for _, child in _named_module_rows(wrapper):
            observed.add(id(child))
    return observed - preexisting_ids


def _binding_items(mapping: Mapping[Any, tuple[Any, ...]]) -> tuple[Any, ...]:
    try:
        return tuple(sorted((key, tuple(value)) for key, value in mapping.items()))
    except (AttributeError, TypeError, ValueError) as error:
        raise SAICJointCompositionError("parameter binding mapping differs") from error


def _adapter_slot_identity_rows(
    *, source_handle: Any, motion_handle: Any
) -> tuple[tuple[str, int, str, int, int], ...]:
    rows: list[tuple[str, int, str, int, int]] = []
    for family, handle in (("source", source_handle), ("motion", motion_handle)):
        try:
            original_q = dict(handle.original_q)
            original_o = dict(handle.original_o)
            rows.extend(
                (family, index, "q", id(wrapper), id(original_q[index]))
                for index, wrapper in handle.q_wrappers
            )
            rows.extend(
                (family, index, "o", id(wrapper), id(original_o[index]))
                for index, wrapper in handle.o_wrappers
            )
        except (AttributeError, KeyError, TypeError) as error:
            raise SAICJointCompositionError(
                "registered adapter slot identity rows differ"
            ) from error
    return tuple(rows)


@dataclass(frozen=True)
class _HandleBirthSeal:
    handle_id: int
    transformer_id: int
    source_handle_id: int
    motion_handle_id: int
    birth_mode: str
    vendor_parameter_identity_rows: tuple[tuple[str, int], ...]
    vendor_module_identity_rows: tuple[tuple[str, int], ...]
    after_source_parameter_identity_rows: tuple[tuple[str, int], ...]
    after_source_module_identity_rows: tuple[tuple[str, int], ...]
    active_parameter_identity_rows: tuple[tuple[str, int], ...]
    active_module_identity_rows: tuple[tuple[str, int], ...]
    source_parameter_identity_rows: tuple[tuple[str, int], ...]
    motion_parameter_identity_rows: tuple[tuple[str, int], ...]
    allowed_parameter_ids: tuple[int, ...]
    allowed_module_ids: tuple[int, ...]
    vendor_binding_items: tuple[Any, ...]
    source_binding_items: tuple[Any, ...]
    motion_binding_items: tuple[Any, ...]
    adapter_slot_identity_rows: tuple[tuple[str, int, str, int, int], ...]
    vendor_content_sha256: str
    source_content_sha256: str
    motion_content_sha256: str
    source_state_sha256: str
    motion_state_sha256: str
    source_install_receipt_digest: str
    motion_install_receipt_digest: str
    source_load_receipt_present: bool
    source_load_receipt_digest: Optional[str]
    motion_load_receipt_present: bool
    motion_load_receipt_digest: Optional[str]
    digest: str
    construction_token: Any = field(repr=False)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "private_joint_handle_birth_seal",
            "handle_id": self.handle_id,
            "transformer_id": self.transformer_id,
            "source_handle_id": self.source_handle_id,
            "motion_handle_id": self.motion_handle_id,
            "birth_mode": self.birth_mode,
            "vendor_parameter_identity_rows": self.vendor_parameter_identity_rows,
            "vendor_module_identity_rows": self.vendor_module_identity_rows,
            "after_source_parameter_identity_rows": (
                self.after_source_parameter_identity_rows
            ),
            "after_source_module_identity_rows": self.after_source_module_identity_rows,
            "active_parameter_identity_rows": self.active_parameter_identity_rows,
            "active_module_identity_rows": self.active_module_identity_rows,
            "source_parameter_identity_rows": self.source_parameter_identity_rows,
            "motion_parameter_identity_rows": self.motion_parameter_identity_rows,
            "allowed_parameter_ids": self.allowed_parameter_ids,
            "allowed_module_ids": self.allowed_module_ids,
            "vendor_binding_items": self.vendor_binding_items,
            "source_binding_items": self.source_binding_items,
            "motion_binding_items": self.motion_binding_items,
            "adapter_slot_identity_rows": self.adapter_slot_identity_rows,
            "vendor_content_sha256": self.vendor_content_sha256,
            "source_content_sha256": self.source_content_sha256,
            "motion_content_sha256": self.motion_content_sha256,
            "source_state_sha256": self.source_state_sha256,
            "motion_state_sha256": self.motion_state_sha256,
            "source_install_receipt_digest": self.source_install_receipt_digest,
            "motion_install_receipt_digest": self.motion_install_receipt_digest,
            "source_load_receipt_present": self.source_load_receipt_present,
            "source_load_receipt_digest": self.source_load_receipt_digest,
            "motion_load_receipt_present": self.motion_load_receipt_present,
            "motion_load_receipt_digest": self.motion_load_receipt_digest,
        }

    def validate(self) -> None:
        if (
            self.construction_token is not _TRUST_MINT
            or self.digest != _object_sha256(self._payload())
        ):
            raise SAICJointCompositionError("private handle birth seal changed")


@dataclass
class _HandleTrustState:
    birth: _HandleBirthSeal
    lease_generation: int = 0
    live_lease: Optional["_LeaseBirthSeal"] = None
    live_lease_ref: Any = field(default=None, repr=False)
    restored: bool = False


@dataclass(frozen=True)
class _LeaseBirthSeal:
    handle_id: int
    transformer_id: int
    lease_id: int
    acquisition_id: int
    generation: int
    pre_lease_audit_digest: str
    leased_audit_digest: str
    acquisition_digest: str
    digest: str
    construction_token: Any = field(repr=False)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "private_registry_lease_birth",
            "handle_id": self.handle_id,
            "transformer_id": self.transformer_id,
            "lease_id": self.lease_id,
            "acquisition_id": self.acquisition_id,
            "generation": self.generation,
            "pre_lease_audit_digest": self.pre_lease_audit_digest,
            "leased_audit_digest": self.leased_audit_digest,
            "acquisition_digest": self.acquisition_digest,
        }

    def validate(self) -> None:
        if (
            self.construction_token is not _TRUST_MINT
            or self.digest != _object_sha256(self._payload())
        ):
            raise SAICJointCompositionError("private lease birth seal changed")


@dataclass
class _LeaseTrustState:
    birth: _LeaseBirthSeal
    handle_ref: Any = field(repr=False)
    released: bool = False


def _make_private_trust_registry():
    handles: WeakKeyDictionary[Any, _HandleTrustState] = WeakKeyDictionary()
    leases: WeakKeyDictionary[Any, _LeaseTrustState] = WeakKeyDictionary()
    lock = RLock()

    def register_handle(handle: Any, state: _HandleTrustState) -> None:
        if handle in handles:
            raise SAICJointCompositionError("joint handle is already registered")
        handles[handle] = state

    def handle_state(handle: Any) -> _HandleTrustState:
        try:
            state = handles[handle]
        except (KeyError, TypeError) as error:
            raise SAICJointCompositionError(
                "joint handle was not issued by the private registry"
            ) from error
        state.birth.validate()
        if state.birth.handle_id != id(handle):
            raise SAICJointCompositionError("joint handle registry identity changed")
        return state

    def register_lease(lease: Any, state: _LeaseTrustState) -> None:
        if lease in leases:
            raise SAICJointCompositionError("runtime lease is already registered")
        leases[lease] = state

    def lease_state(lease: Any) -> _LeaseTrustState:
        try:
            state = leases[lease]
        except (KeyError, TypeError) as error:
            raise SAICJointCompositionError(
                "runtime lease was not issued by the private registry"
            ) from error
        state.birth.validate()
        return state

    def unregister_lease(lease: Any) -> None:
        try:
            del leases[lease]
        except (KeyError, TypeError):
            pass

    return lock, register_handle, handle_state, register_lease, lease_state, unregister_lease


(
    _TRUST_LOCK,
    _register_handle_trust,
    _handle_trust,
    _register_lease_trust,
    _lease_trust,
    _unregister_lease_trust,
) = _make_private_trust_registry()
del _make_private_trust_registry


@dataclass(frozen=True)
class _RuntimeLeaseAcquisition:
    handle_id: int
    transformer_id: int
    generation: int
    pre_lease_audit_digest: str
    leased_audit_digest: str
    digest: str
    construction_token: Any = field(repr=False)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "immutable_runtime_lease_acquisition",
            "handle_id": self.handle_id,
            "transformer_id": self.transformer_id,
            "generation": self.generation,
            "pre_lease_audit_digest": self.pre_lease_audit_digest,
            "leased_audit_digest": self.leased_audit_digest,
        }

    def validate(self, *, expected_digest: str) -> None:
        if (
            self.construction_token is not _RUNTIME_LEASE_MINT
            or type(expected_digest) is not str
            or self.digest != expected_digest
            or self.digest != _object_sha256(self._payload())
        ):
            raise SAICJointCompositionError(
                "runtime lease acquisition provenance changed"
            )


def _mint_runtime_lease_acquisition(
    *,
    handle: "SAICJointCompositionHandle",
    generation: int,
    pre_lease_audit_digest: str,
    leased_audit_digest: str,
) -> _RuntimeLeaseAcquisition:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": "immutable_runtime_lease_acquisition",
        "handle_id": id(handle),
        "transformer_id": id(handle.transformer),
        "generation": generation,
        "pre_lease_audit_digest": pre_lease_audit_digest,
        "leased_audit_digest": leased_audit_digest,
    }
    return _RuntimeLeaseAcquisition(
        handle_id=id(handle),
        transformer_id=id(handle.transformer),
        generation=generation,
        pre_lease_audit_digest=pre_lease_audit_digest,
        leased_audit_digest=leased_audit_digest,
        digest=_object_sha256(payload),
        construction_token=_RUNTIME_LEASE_MINT,
    )


def _validated_handle_receipts(
    handle: Any,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Optional[Mapping[str, Any]],
    Optional[Mapping[str, Any]],
]:
    """Validate the exact stored receipt envelopes against the live bound state."""

    source_install = _validate_source_receipt(handle.source_install_receipt)
    motion_install = _validate_motion_receipt(handle.motion_install_receipt)
    source_load = (
        None
        if handle.source_load_receipt is None
        else _validate_source_load_receipt(
            handle.source_load_receipt,
            expected=tuple(handle.source_parameters),
        )
    )
    motion_load = (
        None
        if handle.motion_load_receipt is None
        else _validate_motion_load_receipt(
            handle.motion_load_receipt,
            expected=tuple(handle.motion_parameters),
        )
    )
    return source_install, motion_install, source_load, motion_load


def _mint_handle_birth_seal(handle: Any) -> _HandleBirthSeal:
    vendor_rows = tuple(handle.vendor_parameter_rows)
    source_rows = tuple(handle.source_parameters)
    motion_rows = tuple(handle.motion_parameters)
    source_state = _state_sha256(source_rows, label="source_anchor")
    motion_state = _state_sha256(motion_rows, label="motion_operator")
    source_install, motion_install, source_load, motion_load = (
        _validated_handle_receipts(handle)
    )
    if source_load is not None and source_load["state_tensor_sha256"] != source_state:
        raise SAICJointCompositionError(
            "source load receipt does not bind the installed source state"
        )
    if motion_load is not None and motion_load["state_tensor_sha256"] != motion_state:
        raise SAICJointCompositionError(
            "motion load receipt does not bind the installed motion state"
        )
    unsealed = _HandleBirthSeal(
        handle_id=id(handle),
        transformer_id=id(handle.transformer),
        source_handle_id=id(handle.source_handle),
        motion_handle_id=id(handle.motion_handle),
        birth_mode=handle.mode,
        vendor_parameter_identity_rows=_parameter_identity_rows(vendor_rows),
        vendor_module_identity_rows=_module_identity_rows(
            tuple(handle.vendor_module_rows)
        ),
        after_source_parameter_identity_rows=tuple(
            handle.after_source_parameter_identity_rows
        ),
        after_source_module_identity_rows=tuple(
            handle.after_source_module_identity_rows
        ),
        active_parameter_identity_rows=_parameter_identity_rows(
            _named_parameter_rows(handle.transformer)
        ),
        active_module_identity_rows=_module_identity_rows(
            _named_module_rows(handle.transformer)
        ),
        source_parameter_identity_rows=_parameter_identity_rows(source_rows),
        motion_parameter_identity_rows=_parameter_identity_rows(motion_rows),
        allowed_parameter_ids=tuple(
            sorted(id(parameter) for _, parameter in _named_parameter_rows(handle.transformer))
        ),
        allowed_module_ids=tuple(
            sorted(id(module) for _, module in _named_module_rows(handle.transformer))
        ),
        vendor_binding_items=_binding_items(
            {
                id(parameter): _parameter_binding(parameter)
                for _, parameter in vendor_rows
            }
        ),
        source_binding_items=_binding_items(_parameter_binding_map(source_rows)),
        motion_binding_items=_binding_items(_parameter_binding_map(motion_rows)),
        adapter_slot_identity_rows=_adapter_slot_identity_rows(
            source_handle=handle.source_handle,
            motion_handle=handle.motion_handle,
        ),
        vendor_content_sha256=_parameter_content_sha256(
            vendor_rows, label="vendor_birth"
        ),
        source_content_sha256=_parameter_content_sha256(
            source_rows, label="source_birth"
        ),
        motion_content_sha256=_parameter_content_sha256(
            motion_rows, label="motion_birth"
        ),
        source_state_sha256=source_state,
        motion_state_sha256=motion_state,
        source_install_receipt_digest=source_install["digest"],
        motion_install_receipt_digest=motion_install["digest"],
        source_load_receipt_present=source_load is not None,
        source_load_receipt_digest=(
            None if source_load is None else source_load["digest"]
        ),
        motion_load_receipt_present=motion_load is not None,
        motion_load_receipt_digest=(
            None if motion_load is None else motion_load["digest"]
        ),
        digest="",
        construction_token=_TRUST_MINT,
    )
    sealed = replace(unsealed, digest=_object_sha256(unsealed._payload()))
    sealed.validate()
    return sealed


def _mint_lease_birth_seal(
    *,
    handle: Any,
    lease: Any,
    acquisition: _RuntimeLeaseAcquisition,
) -> _LeaseBirthSeal:
    unsealed = _LeaseBirthSeal(
        handle_id=id(handle),
        transformer_id=id(handle.transformer),
        lease_id=id(lease),
        acquisition_id=id(acquisition),
        generation=acquisition.generation,
        pre_lease_audit_digest=acquisition.pre_lease_audit_digest,
        leased_audit_digest=acquisition.leased_audit_digest,
        acquisition_digest=acquisition.digest,
        digest="",
        construction_token=_TRUST_MINT,
    )
    sealed = replace(unsealed, digest=_object_sha256(unsealed._payload()))
    sealed.validate()
    return sealed


def _assert_handle_birth_view(handle: Any, state: _HandleTrustState) -> None:
    birth = state.birth
    source_install, motion_install, source_load, motion_load = (
        _validated_handle_receipts(handle)
    )
    try:
        matches = (
            birth.handle_id == id(handle)
            and birth.transformer_id == id(handle.transformer)
            and birth.source_handle_id == id(handle.source_handle)
            and birth.motion_handle_id == id(handle.motion_handle)
            and handle.mode == birth.birth_mode
            and _parameter_identity_rows(tuple(handle.vendor_parameter_rows))
            == birth.vendor_parameter_identity_rows
            and _module_identity_rows(tuple(handle.vendor_module_rows))
            == birth.vendor_module_identity_rows
            and tuple(handle.after_source_parameter_identity_rows)
            == birth.after_source_parameter_identity_rows
            and tuple(handle.after_source_module_identity_rows)
            == birth.after_source_module_identity_rows
            and tuple(handle.active_parameter_identity_rows)
            == birth.active_parameter_identity_rows
            and tuple(handle.active_module_identity_rows)
            == birth.active_module_identity_rows
            and _parameter_identity_rows(tuple(handle.source_parameters))
            == birth.source_parameter_identity_rows
            and _parameter_identity_rows(tuple(handle.motion_parameters))
            == birth.motion_parameter_identity_rows
            and tuple(sorted(handle.allowed_parameter_ids))
            == birth.allowed_parameter_ids
            and tuple(sorted(handle.allowed_module_ids)) == birth.allowed_module_ids
            and _binding_items(handle.vendor_parameter_bindings)
            == birth.vendor_binding_items
            and _binding_items(handle.source_parameter_bindings)
            == birth.source_binding_items
            and _binding_items(handle.motion_parameter_bindings)
            == birth.motion_binding_items
            and _adapter_slot_identity_rows(
                source_handle=handle.source_handle,
                motion_handle=handle.motion_handle,
            )
            == birth.adapter_slot_identity_rows
            and handle.source_state_sha256 == birth.source_state_sha256
            and handle.motion_state_sha256 == birth.motion_state_sha256
            and source_install["digest"] == birth.source_install_receipt_digest
            and motion_install["digest"] == birth.motion_install_receipt_digest
            and (source_load is not None) == birth.source_load_receipt_present
            and (
                None if source_load is None else source_load["digest"]
            )
            == birth.source_load_receipt_digest
            and (motion_load is not None) == birth.motion_load_receipt_present
            and (
                None if motion_load is None else motion_load["digest"]
            )
            == birth.motion_load_receipt_digest
            and bool(handle.restored) == state.restored
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise SAICJointCompositionError("public joint handle view differs") from error
    if not matches:
        raise SAICJointCompositionError(
            "public joint handle fields differ from private birth seal"
        )


@dataclass(eq=False)
class SAICJointCompositionHandle:
    """Own the only legal composed gauge of the two legacy adapter handles."""

    transformer: nn.Module
    mode: str
    source_handle: source_anchor.SAICSourceAnchorHandle
    motion_handle: temporal_action.SAICTemporalActionOperatorHandle
    source_install_receipt: Mapping[str, Any]
    motion_install_receipt: Mapping[str, Any]
    source_load_receipt: Optional[Mapping[str, Any]]
    motion_load_receipt: Optional[Mapping[str, Any]]
    source_parameters: tuple[tuple[str, nn.Parameter], ...] = field(repr=False)
    motion_parameters: tuple[tuple[str, nn.Parameter], ...] = field(repr=False)
    vendor_parameter_rows: tuple[tuple[str, nn.Parameter], ...] = field(repr=False)
    vendor_module_rows: tuple[tuple[str, nn.Module], ...] = field(repr=False)
    vendor_parameter_bindings: Mapping[int, tuple[Any, ...]] = field(repr=False)
    after_source_parameter_identity_rows: tuple[tuple[str, int], ...] = field(
        repr=False
    )
    after_source_module_identity_rows: tuple[tuple[str, int], ...] = field(
        repr=False
    )
    active_parameter_identity_rows: tuple[tuple[str, int], ...] = field(repr=False)
    active_module_identity_rows: tuple[tuple[str, int], ...] = field(repr=False)
    source_parameter_bindings: Mapping[str, tuple[Any, ...]] = field(repr=False)
    motion_parameter_bindings: Mapping[str, tuple[Any, ...]] = field(repr=False)
    allowed_parameter_ids: frozenset[int] = field(repr=False)
    allowed_module_ids: frozenset[int] = field(repr=False)
    source_state_sha256: str
    motion_state_sha256: str
    construction_token: Any = field(repr=False)
    restored: bool = False

    def __post_init__(self) -> None:
        if self.construction_token is not _CONSTRUCTION_MINT:
            raise SAICJointCompositionError(
                "joint composition handles must come from the registered installer"
            )

    def _assert_registered_slots(self) -> None:
        if self.source_handle.transformer is not self.transformer:
            raise SAICJointCompositionError("source handle transformer binding changed")
        if self.motion_handle.transformer is not self.transformer:
            raise SAICJointCompositionError("motion handle transformer binding changed")
        if self.source_handle.restored or self.motion_handle.restored:
            raise SAICJointCompositionError("a child adapter was restored out of order")
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != source_anchor.TOTAL_BLOCKS_1P3B:
            raise SAICJointCompositionError("Bernini block closure changed")
        for index, wrapper in self.source_handle.q_wrappers:
            if blocks[index].attn1.to_q is not wrapper:
                raise SAICJointCompositionError("source query wrapper identity changed")
        for index, wrapper in self.source_handle.o_wrappers:
            if blocks[index].attn1.to_out[0] is not wrapper:
                raise SAICJointCompositionError("source output wrapper identity changed")
        for index, wrapper in self.motion_handle.q_wrappers:
            if blocks[index].attn2.to_q is not wrapper:
                raise SAICJointCompositionError("motion query wrapper identity changed")
        for index, wrapper in self.motion_handle.o_wrappers:
            if blocks[index].attn2.to_out[0] is not wrapper:
                raise SAICJointCompositionError("motion output wrapper identity changed")
        live_source = tuple(
            (name, id(parameter))
            for name, parameter in _source_named_parameters(self.source_handle)
        )
        registered_source = tuple(
            (name, id(parameter)) for name, parameter in self.source_parameters
        )
        if live_source != registered_source:
            raise SAICJointCompositionError("source registered parameter binding changed")
        live_motion = tuple(
            (name, id(parameter))
            for name, parameter in _motion_named_parameters(self.motion_handle)
        )
        registered_motion = tuple(
            (name, id(parameter)) for name, parameter in self.motion_parameters
        )
        if live_motion != registered_motion:
            raise SAICJointCompositionError("motion registered parameter binding changed")

    def _audit_structure_and_gauge(
        self,
        *,
        check_motion_state: bool,
        runtime_lease_active_override: Optional[bool] = None,
        validate_runtime_lease: bool = True,
    ) -> Mapping[str, Any]:
        del check_motion_state  # Updates are deliberately never authorized here.
        with _TRUST_LOCK:
            state = _handle_trust(self)
            birth = state.birth
            _assert_handle_birth_view(self, state)
            if state.restored:
                raise SAICJointCompositionError("joint composition was restored")
            lease_active = state.live_lease is not None
            if validate_runtime_lease and lease_active:
                state.live_lease.validate()
                lease = (
                    None
                    if state.live_lease_ref is None
                    else state.live_lease_ref()
                )
                if (
                    lease is None
                    or state.live_lease.handle_id != id(self)
                    or state.live_lease.transformer_id != id(self.transformer)
                    or state.live_lease.generation != state.lease_generation
                ):
                    raise SAICJointCompositionError(
                        "private runtime lease registry binding changed"
                    )
                lease._registry_states()
            receipt_lease_active = (
                lease_active
                if runtime_lease_active_override is None
                else runtime_lease_active_override
            )

            current_parameters = _named_parameter_rows(self.transformer)
            if (
                _parameter_identity_rows(current_parameters)
                != birth.active_parameter_identity_rows
            ):
                raise SAICJointCompositionError(
                    "active parameter name-to-object-ID binding changed"
                )
            current_parameter_ids = frozenset(
                id(parameter) for _, parameter in current_parameters
            )
            allowed_parameter_ids = frozenset(birth.allowed_parameter_ids)
            if current_parameter_ids != allowed_parameter_ids:
                raise SAICJointCompositionError(
                    "transformer parameter ID union differs from private birth seal"
                )
            current_modules = _named_module_rows(self.transformer)
            if _module_identity_rows(current_modules) != birth.active_module_identity_rows:
                raise SAICJointCompositionError(
                    "active module name-to-object-ID binding changed"
                )
            if frozenset(id(module) for _, module in current_modules) != frozenset(
                birth.allowed_module_ids
            ):
                raise SAICJointCompositionError(
                    "transformer module ID union differs from private birth seal"
                )
            self._assert_registered_slots()

            vendor_bindings = dict(birth.vendor_binding_items)
            source_bindings = dict(birth.source_binding_items)
            motion_bindings = dict(birth.motion_binding_items)
            if any(
                _parameter_binding(parameter) != vendor_bindings.get(id(parameter))
                for _, parameter in self.vendor_parameter_rows
            ):
                raise SAICJointCompositionError("vendor parameter binding changed")
            if any(
                _parameter_binding(parameter) != source_bindings.get(name)
                for name, parameter in self.source_parameters
            ):
                raise SAICJointCompositionError("source parameter binding changed")
            if any(
                _parameter_binding(parameter) != motion_bindings.get(name)
                for name, parameter in self.motion_parameters
            ):
                raise SAICJointCompositionError("motion parameter binding changed")

            vendor_content = _parameter_content_sha256(
                self.vendor_parameter_rows, label="vendor_birth"
            )
            source_content = _parameter_content_sha256(
                self.source_parameters, label="source_birth"
            )
            motion_content = _parameter_content_sha256(
                self.motion_parameters, label="motion_birth"
            )
            if vendor_content != birth.vendor_content_sha256:
                raise SAICJointCompositionError("vendor parameter bytes changed")
            if source_content != birth.source_content_sha256:
                raise SAICJointCompositionError("source parameter bytes changed")
            if motion_content != birth.motion_content_sha256:
                raise SAICJointCompositionError("motion parameter bytes changed")
            source_digest = _state_sha256(
                self.source_parameters, label="source_anchor"
            )
            motion_digest = _state_sha256(
                self.motion_parameters, label="motion_operator"
            )
            if source_digest != birth.source_state_sha256:
                raise SAICJointCompositionError("bound source-anchor state changed")
            if motion_digest != birth.motion_state_sha256:
                raise SAICJointCompositionError("bound motion-operator state changed")

            vendor_ids = {
                parameter_id for _, parameter_id in birth.vendor_parameter_identity_rows
            }
            source_ids = {
                parameter_id for _, parameter_id in birth.source_parameter_identity_rows
            }
            motion_ids = {
                parameter_id for _, parameter_id in birth.motion_parameter_identity_rows
            }
            if (
                not source_ids
                or not motion_ids
                or vendor_ids & source_ids
                or vendor_ids & motion_ids
                or source_ids & motion_ids
                or frozenset(vendor_ids | source_ids | motion_ids)
                != allowed_parameter_ids
            ):
                raise SAICJointCompositionError(
                    "private joint parameter partition is not disjoint"
                )
            observed_trainable = {
                id(parameter)
                for _, parameter in current_parameters
                if parameter.requires_grad
            }
            expected_trainable = (
                motion_ids if birth.birth_mode == STAGE_B_TRAIN else set()
            )
            if observed_trainable != expected_trainable:
                raise SAICJointCompositionError(
                    "joint trainable gauge differs from immutable birth mode"
                )

            value = {
                "schema_version": SCHEMA_VERSION,
                "mode": birth.birth_mode,
                "immutable_birth_mode": True,
                "process_local_registry_handle_identity": True,
                "arbitrary_same_process_reflection_resistance_claim": False,
                "strict_parameter_id_union": True,
                "strict_module_id_union": True,
                "exact_active_parameter_name_object_binding": True,
                "exact_active_module_name_object_binding": True,
                "per_audit_parameter_byte_hashes": True,
                "per_audit_hash_cost": "O(total_transformer_parameter_bytes)",
                "per_audit_device_to_cpu_synchronization": True,
                "parameter_alias_free": True,
                "adapter_parameter_sets_disjoint": True,
                "vendor_parameter_count": len(vendor_ids),
                "source_parameter_count": len(source_ids),
                "motion_parameter_count": len(motion_ids),
                "trainable_parameter_count": len(observed_trainable),
                "only_motion_trainable": birth.birth_mode == STAGE_B_TRAIN,
                "all_parameters_frozen": birth.birth_mode == INFERENCE,
                "scoped_motion_parameter_update_authorized": False,
                "parameter_update_authorized": False,
                "optimizer_parameter_access_authorized": False,
                "end_to_end_training_authorized": False,
                "gradient_state_rollback_authorized": False,
                "optimizer_or_scaler_state_rollback_authorized": False,
                "rng_state_rollback_authorized": False,
                "vendor_content_sha256": vendor_content,
                "source_content_sha256": source_content,
                "motion_content_sha256": motion_content,
                "source_state_sha256": source_digest,
                "motion_state_sha256": motion_digest,
                "motion_state_generation": 0,
                "source_parameter_binding_sha256": _object_sha256(
                    birth.source_binding_items
                ),
                "motion_parameter_binding_sha256": _object_sha256(
                    birth.motion_binding_items
                ),
                "active_parameter_identity_sha256": _object_sha256(
                    birth.active_parameter_identity_rows
                ),
                "active_module_identity_sha256": _object_sha256(
                    birth.active_module_identity_rows
                ),
                "runtime_lease_active": receipt_lease_active,
                "runtime_lease_generation": state.lease_generation,
                "runtime_lease_process_local_registry_bound": receipt_lease_active,
                "runtime_parameter_id_partition_sha256": _object_sha256(
                    {
                        "vendor": sorted(vendor_ids),
                        "source": sorted(source_ids),
                        "motion": sorted(motion_ids),
                    }
                ),
            }
            return {**value, "digest": _object_sha256(value)}

    def audit(self) -> Mapping[str, Any]:
        """Verify the live composition under the documented process-local scope."""

        return self._audit_structure_and_gauge(check_motion_state=True)

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Reject optimizer access: this object is lifecycle-only."""

        self.audit()
        raise SAICJointCompositionError(
            "joint lifecycle does not authorize optimizer parameter access"
        )

    @contextmanager
    def motion_update(self) -> Iterator[tuple[tuple[str, nn.Parameter], ...]]:
        """Reject state updates until a native runtime boundary owns authority."""

        self.audit()
        raise SAICJointCompositionError(
            "joint lifecycle does not authorize parameter updates"
        )
        if False:  # pragma: no cover - keeps the contextmanager generator shape.
            yield self.motion_parameters

    def acquire_runtime_lease(self) -> "SAICJointCompositionRuntimeLease":
        """Lease the immutable inference composition to one native runtime.

        The lease is intentionally opaque and exclusive.  While it is live,
        :meth:`restore` rejects before invoking either child adapter restore.
        """

        with _TRUST_LOCK:
            state = _handle_trust(self)
            pre_audit = self.audit()
            if state.birth.birth_mode != INFERENCE:
                raise SAICJointCompositionError(
                    "runtime leases require the immutable inference birth mode"
                )
            if state.live_lease is not None:
                raise SAICJointCompositionError(
                    "a composition runtime lease is active"
                )
            previous_generation = state.lease_generation
            state.lease_generation = previous_generation + 1
            lease: Optional[SAICJointCompositionRuntimeLease] = None
            try:
                prospective = self._audit_structure_and_gauge(
                    check_motion_state=True,
                    runtime_lease_active_override=True,
                    validate_runtime_lease=False,
                )
                acquisition = _mint_runtime_lease_acquisition(
                    handle=self,
                    generation=state.lease_generation,
                    pre_lease_audit_digest=pre_audit["digest"],
                    leased_audit_digest=prospective["digest"],
                )
                lease = SAICJointCompositionRuntimeLease(
                    _handle=self,
                    _acquisition=acquisition,
                    _construction_token=_RUNTIME_LEASE_MINT,
                )
                lease_birth = _mint_lease_birth_seal(
                    handle=self, lease=lease, acquisition=acquisition
                )
                _register_lease_trust(
                    lease,
                    _LeaseTrustState(birth=lease_birth, handle_ref=ref(self)),
                )
                state.live_lease = lease_birth
                state.live_lease_ref = ref(lease)
                leased_audit = self.audit()
                if leased_audit["digest"] != prospective["digest"]:
                    raise SAICJointCompositionError(
                        "runtime lease prospective/live audit digest differs"
                    )
            except BaseException:
                state.live_lease = None
                state.live_lease_ref = None
                state.lease_generation = previous_generation
                if lease is not None:
                    _unregister_lease_trust(lease)
                raise
            return lease

    @contextmanager
    def runtime_lease(self) -> Iterator["SAICJointCompositionRuntimeLease"]:
        lease = self.acquire_runtime_lease()
        try:
            yield lease
        except BaseException as body_error:
            with _TRUST_LOCK:
                registry_released = _lease_trust(lease).released
            if not registry_released:
                try:
                    lease.release()
                except BaseException as cleanup_error:
                    try:
                        setattr(
                            body_error,
                            "saic_runtime_lease_release_error",
                            cleanup_error,
                        )
                    except Exception:
                        pass
            raise
        else:
            with _TRUST_LOCK:
                registry_released = _lease_trust(lease).released
            if not registry_released:
                lease.release()

    def receipt(self) -> Mapping[str, Any]:
        with _TRUST_LOCK:
            audit = self.audit()
            state = _handle_trust(self)
            birth = state.birth
            lease_birth = state.live_lease
            source_install, motion_install, source_load, motion_load = (
                _validated_handle_receipts(self)
            )
            if (
                source_install["digest"] != birth.source_install_receipt_digest
                or motion_install["digest"] != birth.motion_install_receipt_digest
                or (source_load is not None) != birth.source_load_receipt_present
                or (
                    None if source_load is None else source_load["digest"]
                )
                != birth.source_load_receipt_digest
                or (motion_load is not None) != birth.motion_load_receipt_present
                or (
                    None if motion_load is None else motion_load["digest"]
                )
                != birth.motion_load_receipt_digest
            ):
                raise SAICJointCompositionError(
                    "stored receipts differ from process-local birth seal"
                )
        value = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "mode": birth.birth_mode,
            "immutable_birth_mode": True,
            "process_local_registry_integrity_root": True,
            "arbitrary_same_process_reflection_resistance_claim": False,
            "lifecycle_order": [
                "vendor_frozen",
                "source_install_load_receipt",
                "source_freeze",
                "motion_install_load_receipt",
                "joint_gauge",
            ],
            "restore_order": ["motion", "source", "verify_vendor_identity"],
            "source_install_receipt_digest": birth.source_install_receipt_digest,
            "motion_install_receipt_digest": birth.motion_install_receipt_digest,
            "source_load_receipt_digest": (
                None
                if source_load is None
                else birth.source_load_receipt_digest
            ),
            "source_load_receipt_present": birth.source_load_receipt_present,
            "motion_load_receipt_digest": (
                None
                if motion_load is None
                else birth.motion_load_receipt_digest
            ),
            "motion_load_receipt_present": birth.motion_load_receipt_present,
            "source_state_sha256": birth.source_state_sha256,
            "motion_state_sha256": birth.motion_state_sha256,
            "motion_state_generation": 0,
            "strict_vendor_plus_registered_adapter_parameter_union": True,
            "per_audit_hash_cost": "O(total_transformer_parameter_bytes)",
            "audit_intended_for_sparse_lifecycle_boundaries_only": True,
            "unknown_parameter_or_alias_rejected": True,
            "source_immutable_after_load": True,
            "motion_transition_requires_joint_context": False,
            "motion_update_rollback_scope": "none_non_authoritative_lifecycle",
            "gradient_state_rollback_authorized": False,
            "optimizer_or_scaler_state_rollback_authorized": False,
            "rng_state_rollback_authorized": False,
            "scoped_motion_parameter_update_authorized": False,
            "parameter_update_authorized": False,
            "optimizer_parameter_access_authorized": False,
            "end_to_end_training_authorized": False,
            "runtime_lease_active": lease_birth is not None,
            "runtime_lease_generation": state.lease_generation,
            "runtime_lease_registry_digest": (
                None if lease_birth is None else lease_birth.digest
            ),
            "native_runtime_lifecycle_managed": False,
            "native_runtime_status": NATIVE_RUNTIME_STATUS,
            "next_runtime_seam": (
                "bind the exact diffusion object; install source-anchor native "
                "wrapper inner, then online-motion native wrapper outer, and "
                "restore those wrappers before this adapter handle"
            ),
            "audit_receipt_digest": audit["digest"],
            "optimizer_created": False,
            "training_authorized": False,
            "training_authorized_legacy_field_semantics": (
                "end_to_end_training_workflow_only"
            ),
            "semantic_action_success_claim": False,
        }
        return {**value, "digest": _object_sha256(value)}

    @staticmethod
    def _slot_family_state(
        *,
        blocks: tuple[nn.Module, ...],
        family: str,
        birth: _HandleBirthSeal,
    ) -> str:
        """Classify one adapter family from exact registered live slots."""

        if len(blocks) != source_anchor.TOTAL_BLOCKS_1P3B:
            return "ambiguous"
        active: list[bool] = []
        removed: list[bool] = []
        try:
            attention_name = "attn1" if family == "source" else "attn2"
            rows = [
                row for row in birth.adapter_slot_identity_rows if row[0] == family
            ]
            for _, index, projection, wrapper_id, original_id in rows:
                attention = getattr(blocks[index], attention_name)
                slot = (
                    attention.to_q
                    if projection == "q"
                    else attention.to_out[0]
                )
                active.append(id(slot) == wrapper_id)
                removed.append(id(slot) == original_id)
        except (AttributeError, IndexError, KeyError, TypeError):
            return "ambiguous"
        if active and all(active):
            return "active"
        if removed and all(removed):
            return "removed"
        return "ambiguous"

    def _restore_phase(self) -> str:
        """Derive the resumable phase from exact slots, never a cached flag."""

        state = _handle_trust(self)
        _assert_handle_birth_view(self, state)
        birth = state.birth
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        source = self._slot_family_state(
            blocks=blocks,
            family="source",
            birth=birth,
        )
        motion = self._slot_family_state(
            blocks=blocks,
            family="motion",
            birth=birth,
        )
        if motion == "removed":
            self.motion_handle.restored = True
        elif self.motion_handle.restored:
            return _RESTORE_AMBIGUOUS
        if source == "removed":
            self.source_handle.restored = True
        elif self.source_handle.restored:
            return _RESTORE_AMBIGUOUS
        return {
            ("active", "active"): _RESTORE_LIVE,
            ("active", "removed"): _RESTORE_MOTION_REMOVED,
            ("removed", "removed"): _RESTORE_CHILDREN_REMOVED,
        }.get((source, motion), _RESTORE_AMBIGUOUS)

    def _audit_restore_phase(self, phase: str) -> Mapping[str, Any]:
        """Verify either legal post-child tree under the process-local scope."""

        state = _handle_trust(self)
        birth = state.birth
        if phase not in {_RESTORE_MOTION_REMOVED, _RESTORE_CHILDREN_REMOVED}:
            raise SAICJointCompositionError("restore audit phase differs")
        if self._restore_phase() != phase:
            raise SAICJointCompositionError("registered slots differ for restore phase")
        parameter_rows = _named_parameter_rows(self.transformer)
        module_rows = _named_module_rows(self.transformer)
        if phase == _RESTORE_MOTION_REMOVED:
            expected_parameters = birth.after_source_parameter_identity_rows
            expected_modules = birth.after_source_module_identity_rows
            bound_rows = self.vendor_parameter_rows + self.source_parameters
        else:
            expected_parameters = birth.vendor_parameter_identity_rows
            expected_modules = birth.vendor_module_identity_rows
            bound_rows = self.vendor_parameter_rows
        if _parameter_identity_rows(parameter_rows) != expected_parameters:
            raise SAICJointCompositionError(
                "named-parameter identities differ for restore phase"
            )
        if _module_identity_rows(module_rows) != expected_modules:
            raise SAICJointCompositionError(
                "named-module identities differ for restore phase"
            )
        expected_bindings = {
            **{
                name: dict(birth.vendor_binding_items)[id(parameter)]
                for name, parameter in self.vendor_parameter_rows
            },
            **(
                dict(birth.source_binding_items)
                if phase == _RESTORE_MOTION_REMOVED
                else {}
            ),
        }
        if any(
            _parameter_binding(parameter) != expected_bindings.get(name)
            for name, parameter in bound_rows
        ):
            raise SAICJointCompositionError(
                "parameter binding/state differs for restore phase"
            )
        if any(parameter.requires_grad for _, parameter in parameter_rows):
            raise SAICJointCompositionError("restore phase gauge is not frozen")
        vendor_content = _parameter_content_sha256(
            self.vendor_parameter_rows, label="vendor_birth"
        )
        if vendor_content != birth.vendor_content_sha256:
            raise SAICJointCompositionError(
                "vendor parameter bytes changed during restoration"
            )
        source_digest: Optional[str] = None
        source_content: Optional[str] = None
        if phase == _RESTORE_MOTION_REMOVED:
            source_content = _parameter_content_sha256(
                self.source_parameters, label="source_birth"
            )
            if source_content != birth.source_content_sha256:
                raise SAICJointCompositionError(
                    "source parameter bytes changed during restoration"
                )
            source_digest = _state_sha256(
                self.source_parameters, label="source_anchor_restore"
            )
            if source_digest != birth.source_state_sha256:
                raise SAICJointCompositionError(
                    "source state digest changed during restoration"
                )
        value = {
            "schema_version": SCHEMA_VERSION,
            "restore_phase": phase,
            "exact_parameter_identity": True,
            "exact_module_identity": True,
            "parameters_frozen": True,
            "vendor_content_sha256": vendor_content,
            "source_content_sha256": source_content,
            "source_state_sha256": source_digest,
        }
        return {**value, "digest": _object_sha256(value)}

    @staticmethod
    def _restore_error_row(*, stage: str, error: BaseException) -> Mapping[str, str]:
        return {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
        }

    def _finish_restore_attempt(
        self,
        *,
        initial_state: str,
        pre_restore_audit_digest: Optional[str],
        attempted_stages: list[str],
        errors: list[tuple[str, BaseException]],
        vendor_audit: Optional[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        final_state = self._restore_phase()
        vendor_verified = vendor_audit is not None
        if vendor_verified:
            state = _handle_trust(self)
            state.restored = True
            self.restored = True
            final_state = _RESTORE_COMPLETE
        error_rows = [
            self._restore_error_row(stage=stage, error=error)
            for stage, error in errors
        ]
        value = {
            "schema_version": SCHEMA_VERSION,
            "classification": "resumable_motion_then_source_restore_attempt",
            "initial_restore_state": initial_state,
            "final_restore_state": final_state,
            "restore_order": ["motion", "source", "verify_vendor_identity"],
            "attempted_stages": attempted_stages,
            "pre_restore_audit_digest": pre_restore_audit_digest,
            "vendor_audit_digest": (
                None if vendor_audit is None else vendor_audit["digest"]
            ),
            "motion_child_restored": bool(self.motion_handle.restored),
            "source_child_restored": bool(self.source_handle.restored),
            "original_vendor_parameter_ids_restored": vendor_verified,
            "original_vendor_module_ids_restored": vendor_verified,
            "vendor_parameters_frozen": vendor_verified,
            "vendor_verified": vendor_verified,
            "complete": vendor_verified,
            "retryable": not vendor_verified and final_state != _RESTORE_AMBIGUOUS,
            "errors": error_rows,
            "root_cause": None if not error_rows else error_rows[0],
        }
        receipt = {**value, "digest": _object_sha256(value)}
        if errors:
            root_cause = errors[0][1]
            raise SAICJointCompositionRestoreError(
                "joint composition restore recorded one or more failures; "
                f"final_state={final_state}",
                receipt=receipt,
                root_cause=root_cause,
            ) from root_cause
        return receipt

    def restore(self) -> Mapping[str, Any]:
        with _TRUST_LOCK:
            return self._restore_locked()

    def _restore_locked(self) -> Mapping[str, Any]:
        """Best-effort, resumable motion-then-source restoration.

        Each retry observes exact registered slots and resumes from that phase;
        it never invokes a child twice after that child's complete slot mutation.
        Every failure carries a sealed receipt containing the original cause and
        the cleanup state reached by this attempt.
        """

        state = _handle_trust(self)
        _assert_handle_birth_view(self, state)
        if state.live_lease is not None:
            raise SAICJointCompositionError(
                "cannot restore composition while a runtime lease is active"
            )
        if state.restored:
            raise SAICJointCompositionError("joint composition was restored")
        attempted_stages: list[str] = []
        errors: list[tuple[str, BaseException]] = []
        vendor_audit: Optional[Mapping[str, Any]] = None
        pre_restore_audit_digest: Optional[str] = None
        initial_state = self._restore_phase()

        if temporal_action.active_route() is not None:
            errors.append(
                (
                    "preflight_routes",
                    SAICJointCompositionError("motion route is active during restore"),
                )
            )
        if source_anchor.active_route() is not None:
            errors.append(
                (
                    "preflight_routes",
                    SAICJointCompositionError("source route is active during restore"),
                )
            )
        if initial_state == _RESTORE_AMBIGUOUS:
            errors.append(
                (
                    "observe_restore_state",
                    SAICJointCompositionError(
                        "registered adapter slots are partially restored or out of order"
                    ),
                )
            )
        if errors:
            return self._finish_restore_attempt(
                initial_state=initial_state,
                pre_restore_audit_digest=pre_restore_audit_digest,
                attempted_stages=attempted_stages,
                errors=errors,
                vendor_audit=vendor_audit,
            )

        if initial_state == _RESTORE_LIVE:
            attempted_stages.append("audit_live_composition")
            try:
                pre_restore_audit_digest = self.audit()["digest"]
            except BaseException as error:
                errors.append(("audit_live_composition", error))
                return self._finish_restore_attempt(
                    initial_state=initial_state,
                    pre_restore_audit_digest=pre_restore_audit_digest,
                    attempted_stages=attempted_stages,
                    errors=errors,
                    vendor_audit=vendor_audit,
                )

            attempted_stages.append("restore_motion")
            try:
                self.motion_handle.restore()
            except BaseException as error:
                errors.append(("restore_motion", error))
            state = self._restore_phase()
            if state != _RESTORE_MOTION_REMOVED:
                if not any(stage == "restore_motion" for stage, _ in errors):
                    errors.append(
                        (
                            "restore_motion",
                            SAICJointCompositionError(
                                "motion restore did not reach the exact motion-removed phase"
                            ),
                        )
                    )
                return self._finish_restore_attempt(
                    initial_state=initial_state,
                    pre_restore_audit_digest=pre_restore_audit_digest,
                    attempted_stages=attempted_stages,
                    errors=errors,
                    vendor_audit=vendor_audit,
                )

        state = self._restore_phase()
        if state == _RESTORE_MOTION_REMOVED:
            attempted_stages.append("audit_after_motion")
            try:
                self._audit_restore_phase(_RESTORE_MOTION_REMOVED)
            except BaseException as error:
                # Once all motion slots are exactly original, source cleanup is
                # still safe and is attempted even if the wider union was dirty.
                errors.append(("audit_after_motion", error))

            attempted_stages.append("restore_source")
            try:
                self.source_handle.restore()
            except BaseException as error:
                errors.append(("restore_source", error))
            state = self._restore_phase()
            if state != _RESTORE_CHILDREN_REMOVED and not any(
                stage == "restore_source" for stage, _ in errors
            ):
                errors.append(
                    (
                        "restore_source",
                        SAICJointCompositionError(
                            "source restore did not reach the exact children-removed phase"
                        ),
                    )
                )

        state = self._restore_phase()
        if state == _RESTORE_CHILDREN_REMOVED:
            attempted_stages.append("verify_vendor_identity")
            try:
                vendor_audit = self._audit_restore_phase(
                    _RESTORE_CHILDREN_REMOVED
                )
            except BaseException as error:
                errors.append(("verify_vendor_identity", error))

        return self._finish_restore_attempt(
            initial_state=initial_state,
            pre_restore_audit_digest=pre_restore_audit_digest,
            attempted_stages=attempted_stages,
            errors=errors,
            vendor_audit=vendor_audit,
        )

    def __enter__(self) -> "SAICJointCompositionHandle":
        self.audit()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, traceback
        with _TRUST_LOCK:
            registry_restored = _handle_trust(self).restored
        if not registry_restored:
            try:
                self.restore()
            except BaseException as restore_error:
                if exc is None:
                    raise
                try:
                    setattr(exc, "saic_joint_restore_error", restore_error)
                except Exception:
                    pass


@dataclass(eq=False)
class SAICJointCompositionRuntimeLease:
    """Opaque exclusive lease held by a future joint native runtime."""

    _handle: SAICJointCompositionHandle = field(repr=False)
    _acquisition: _RuntimeLeaseAcquisition = field(repr=False)
    _construction_token: Any = field(repr=False)
    released: bool = False

    def __post_init__(self) -> None:
        if self._construction_token is not _RUNTIME_LEASE_MINT:
            raise SAICJointCompositionError(
                "runtime leases must come from the joint composition handle"
            )

    @property
    def generation(self) -> int:
        with _TRUST_LOCK:
            return _lease_trust(self).birth.generation

    def _registry_states(
        self,
    ) -> tuple[_LeaseTrustState, _HandleTrustState, _LeaseBirthSeal]:
        lease_state = _lease_trust(self)
        handle = lease_state.handle_ref()
        if handle is None or handle is not self._handle:
            raise SAICJointCompositionError(
                "runtime lease cross-handle binding changed"
            )
        handle_state = _handle_trust(handle)
        birth = lease_state.birth
        acquisition = self._acquisition
        if (
            birth.lease_id != id(self)
            or birth.handle_id != id(handle)
            or birth.transformer_id != id(handle.transformer)
            or birth.acquisition_id != id(acquisition)
            or type(acquisition) is not _RuntimeLeaseAcquisition
            or acquisition.handle_id != birth.handle_id
            or acquisition.transformer_id != birth.transformer_id
            or acquisition.generation != birth.generation
            or acquisition.pre_lease_audit_digest
            != birth.pre_lease_audit_digest
            or acquisition.leased_audit_digest != birth.leased_audit_digest
            or acquisition.digest != birth.acquisition_digest
            or handle_state.live_lease is not birth
            or handle_state.lease_generation != birth.generation
            or bool(self.released) != lease_state.released
        ):
            raise SAICJointCompositionError(
                "runtime lease differs from private registry birth"
            )
        acquisition.validate(expected_digest=birth.acquisition_digest)
        return lease_state, handle_state, birth

    def audit(self) -> Mapping[str, Any]:
        with _TRUST_LOCK:
            lease_state, handle_state, birth = self._registry_states()
            if lease_state.released or handle_state.restored:
                raise SAICJointCompositionError(
                    "composition runtime lease was released"
                )
            joint_audit = self._handle.audit()
            if joint_audit.get("runtime_lease_active") is not True:
                raise SAICJointCompositionError(
                    "composition audit does not contain the live runtime lease"
                )
            if joint_audit["digest"] != birth.leased_audit_digest:
                raise SAICJointCompositionError(
                    "runtime lease live audit differs from private registry birth"
                )
            value = {
                "schema_version": SCHEMA_VERSION,
                "classification": "private_registry_exclusive_inference_lease",
                "generation": birth.generation,
                "lease_registry_digest": birth.digest,
                "acquisition_digest": birth.acquisition_digest,
                "pre_lease_audit_digest": birth.pre_lease_audit_digest,
                "leased_audit_digest": joint_audit["digest"],
                "composition_restore_blocked_before_mutation": True,
                "parameter_update_authorized": False,
                "optimizer_parameter_access_authorized": False,
                "end_to_end_training_authorized": False,
            }
            return {**value, "digest": _object_sha256(value)}

    def release(self) -> Mapping[str, Any]:
        with _TRUST_LOCK:
            receipt = self.audit()
            lease_state, handle_state, birth = self._registry_states()
            prospective = self._handle._audit_structure_and_gauge(
                check_motion_state=True,
                runtime_lease_active_override=False,
                validate_runtime_lease=False,
            )
            lease_reference = handle_state.live_lease_ref
            handle_state.live_lease = None
            handle_state.live_lease_ref = None
            try:
                post_audit = self._handle.audit()
                if post_audit["digest"] != prospective["digest"]:
                    raise SAICJointCompositionError(
                        "runtime lease post-release audit digest differs"
                    )
            except BaseException:
                handle_state.live_lease = birth
                handle_state.live_lease_ref = lease_reference
                raise
            lease_state.released = True
            self.released = True
            value = {
                "schema_version": SCHEMA_VERSION,
                "generation": birth.generation,
                "lease_registry_digest": birth.digest,
                "lease_audit_digest": receipt["digest"],
                "post_release_audit_digest": post_audit["digest"],
                "released": True,
            }
            return {**value, "digest": _object_sha256(value)}


_SOURCE_INSTALL_RECEIPT_KEYS = frozenset(
    {
        "accepted_timestep_representations",
        "active_sigma_indices",
        "appearance_preservation_success_claim",
        "base_parameters_frozen",
        "bias",
        "blocks",
        "classification",
        "digest",
        "exact40_schedule_sha256",
        "fp32_trainable",
        "full_source_native_branches",
        "gradient_checkpointing_supported",
        "mask_pose_flow_track_trajectory_consumed",
        "only_registered_self_attention_qo_replaced",
        "operator",
        "optimizer_created",
        "output_up_zero_initialized_at_install",
        "patch_embedding_untouched",
        "projections",
        "prompt_role_agnostic_action_and_noop",
        "proposal_or_target_video_consumed",
        "rank",
        "route_accepts_caller_rank_size_index_or_mask",
        "route_binds_live_parallel_native_mask_and_actual_scheduler_sigma",
        "schema_version",
        "semantic_action_success_claim",
        "source_reference_padding_rows_exact_base",
        "target_suffix_is_native_pack_structure_not_object_mask",
        "trainable",
        "trainable_state_closed",
        "trainable_state_key_sha256",
        "training_authorized",
    }
)
_MOTION_INSTALL_RECEIPT_KEYS = frozenset(
    {
        "action_id_consumed",
        "active_sigma_indices",
        "base_parameters_frozen",
        "blocks",
        "classification",
        "digest",
        "fp32_bias_free_trainable",
        "full_source_native_branches",
        "low_sigma_indices_exact_base",
        "mask_pose_flow_track_or_trajectory_consumed",
        "only_registered_cross_attention_qo_replaced",
        "operator",
        "optimizer_created",
        "output_up_zero_initialized_at_install",
        "patch_embedding_untouched",
        "phase_code_dimension",
        "phase_count_exact81",
        "projections",
        "rank",
        "route_accepts_mask_rank_size_index_sigma_phase_or_code",
        "route_binds_current_state_live_sp_native_mask_actual_sigma",
        "schema_version",
        "semantic_action_success_claim",
        "source_reference_padding_rows_exact_base",
        "t2v_media_or_proposal_consumed",
        "target_phase_map_from_native_suffix",
        "trainable_key_sha256",
        "training_and_inference_route_identical",
        "training_authorized",
    }
)
_SOURCE_LOAD_RECEIPT_KEYS = frozenset(
    {
        "closed_exact_key_set",
        "digest",
        "schema_version",
        "state_key_count",
        "state_key_sha256",
        "state_tensor_sha256",
    }
)
_MOTION_LOAD_RECEIPT_KEYS = frozenset(
    set(_SOURCE_LOAD_RECEIPT_KEYS) | {"adapter"}
)


def _validate_source_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    sealed = _sealed_copy(receipt, label="source install")
    required = {
        "schema_version": source_anchor.SCHEMA_VERSION,
        "only_registered_self_attention_qo_replaced": True,
        "base_parameters_frozen": True,
        "patch_embedding_untouched": True,
        "trainable_state_closed": True,
        "optimizer_created": False,
        "training_authorized": False,
    }
    if set(sealed) != _SOURCE_INSTALL_RECEIPT_KEYS or any(
        sealed.get(name) != expected for name, expected in required.items()
    ):
        raise SAICJointCompositionError("source install receipt contract differs")
    return sealed


def _validate_motion_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    sealed = _sealed_copy(receipt, label="motion install")
    required = {
        "schema_version": temporal_action.SCHEMA_VERSION,
        "only_registered_cross_attention_qo_replaced": True,
        "base_parameters_frozen": True,
        "patch_embedding_untouched": True,
        "optimizer_created": False,
        "training_authorized": False,
    }
    if set(sealed) != _MOTION_INSTALL_RECEIPT_KEYS or any(
        sealed.get(name) != expected for name, expected in required.items()
    ):
        raise SAICJointCompositionError("motion install receipt contract differs")
    return sealed


def _validate_source_load_receipt(
    receipt: Mapping[str, Any],
    *,
    expected: tuple[tuple[str, nn.Parameter], ...],
) -> Mapping[str, Any]:
    sealed = _sealed_copy(receipt, label="source load")
    expected_names = sorted(name for name, _ in expected)
    required = {
        "schema_version": source_anchor.SCHEMA_VERSION,
        "closed_exact_key_set": True,
        "state_key_count": len(expected),
        "state_key_sha256": _object_sha256(expected_names),
    }
    state_digest = sealed.get("state_tensor_sha256")
    if (
        set(sealed) != _SOURCE_LOAD_RECEIPT_KEYS
        or any(sealed.get(name) != value for name, value in required.items())
        or type(state_digest) is not str
        or len(state_digest) != 64
        or any(character not in "0123456789abcdef" for character in state_digest)
    ):
        raise SAICJointCompositionError("source load receipt contract differs")
    return sealed


def _validate_motion_load_receipt(
    receipt: Mapping[str, Any],
    *,
    expected: tuple[tuple[str, nn.Parameter], ...],
) -> Mapping[str, Any]:
    sealed = _sealed_copy(receipt, label="motion load")
    expected_names = sorted(name for name, _ in expected)
    required = {
        "schema_version": SCHEMA_VERSION,
        "adapter": "motion_operator",
        "closed_exact_key_set": True,
        "state_key_count": len(expected),
        "state_key_sha256": _object_sha256(expected_names),
    }
    state_digest = sealed.get("state_tensor_sha256")
    if (
        set(sealed) != _MOTION_LOAD_RECEIPT_KEYS
        or any(sealed.get(name) != value for name, value in required.items())
        or type(state_digest) is not str
        or len(state_digest) != 64
        or any(character not in "0123456789abcdef" for character in state_digest)
    ):
        raise SAICJointCompositionError("motion load receipt contract differs")
    return sealed


def install_saic_joint_composition(
    transformer: nn.Module,
    *,
    mode: str,
    source_state: Optional[Mapping[str, torch.Tensor]] = None,
    motion_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> SAICJointCompositionHandle:
    """Install, optionally load, and certify both adapters in legal order."""

    if not isinstance(transformer, nn.Module):
        raise SAICJointCompositionError("transformer must be nn.Module")
    if type(mode) is not str or mode not in ALLOWED_MODES:
        raise SAICJointCompositionError(
            f"mode must be one of {sorted(ALLOWED_MODES)}"
        )
    vendor_parameter_rows = _named_parameter_rows(transformer)
    vendor_module_rows = _named_module_rows(transformer)
    if any(parameter.requires_grad for _, parameter in vendor_parameter_rows):
        raise SAICJointCompositionError(
            "freeze the complete vendor transformer before joint installation"
        )
    vendor_parameter_bindings = {
        id(parameter): _parameter_binding(parameter)
        for _, parameter in vendor_parameter_rows
    }
    vendor_parameter_ids = {id(parameter) for _, parameter in vendor_parameter_rows}
    vendor_module_ids = {id(module) for _, module in vendor_module_rows}

    source_handle: Optional[source_anchor.SAICSourceAnchorHandle] = None
    motion_handle: Optional[temporal_action.SAICTemporalActionOperatorHandle] = None
    try:
        source_handle = source_anchor.install_saic_source_anchor_adapter(transformer)
        source_parameters = source_handle.trainable_named_parameters()
        if source_state is None:
            source_load_receipt = None
        else:
            # Use the adapter's own strict source-state loader, then seal its
            # receipt before changing the legacy handle's gauge.
            source_load_receipt = _sealed_copy(
                source_handle.load_trainable_state_dict(source_state),
                label="source load",
            )
        source_install_receipt = _validate_source_receipt(source_handle.receipt())
        source_state_sha256 = _state_sha256(
            source_parameters, label="source_anchor"
        )

        source_parameter_ids = {id(parameter) for _, parameter in source_parameters}
        after_source_parameters = _named_parameter_rows(transformer)
        if {id(parameter) for _, parameter in after_source_parameters} != (
            vendor_parameter_ids | source_parameter_ids
        ):
            raise SAICJointCompositionError(
                "source installation did not form vendor-plus-source parameter union"
            )
        after_source_modules = _named_module_rows(transformer)
        after_source_module_ids = {id(module) for _, module in after_source_modules}
        registered_source_module_ids = _registered_added_module_ids(
            source_handle.q_wrappers + source_handle.o_wrappers,
            preexisting_ids=vendor_module_ids,
        )
        if after_source_module_ids != (
            vendor_module_ids | registered_source_module_ids
        ):
            raise SAICJointCompositionError(
                "source installation module union is not vendor plus registered source"
            )
        for _, parameter in source_parameters:
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for _, parameter in after_source_parameters):
            raise SAICJointCompositionError("source adapter freeze failed")

        motion_handle = temporal_action.install_saic_temporal_action_operator(transformer)
        motion_parameters = motion_handle.trainable_named_parameters()
        if motion_state is None:
            motion_load_receipt = None
        else:
            motion_load_receipt = _sealed_copy(
                _load_closed_state(
                    motion_state, motion_parameters, label="motion_operator"
                ),
                label="motion load",
            )
        motion_install_receipt = _validate_motion_receipt(motion_handle.receipt())
        motion_state_sha256 = _state_sha256(
            motion_parameters, label="motion_operator"
        )
        if mode == INFERENCE:
            for _, parameter in motion_parameters:
                parameter.requires_grad_(False)

        source_parameter_ids = {id(parameter) for _, parameter in source_parameters}
        motion_parameter_ids = {id(parameter) for _, parameter in motion_parameters}
        if (
            vendor_parameter_ids & source_parameter_ids
            or vendor_parameter_ids & motion_parameter_ids
            or source_parameter_ids & motion_parameter_ids
        ):
            raise SAICJointCompositionError("adapter/vendor parameter alias detected")
        allowed_parameter_ids = frozenset(
            vendor_parameter_ids | source_parameter_ids | motion_parameter_ids
        )
        current_parameters = _named_parameter_rows(transformer)
        if {id(parameter) for _, parameter in current_parameters} != allowed_parameter_ids:
            raise SAICJointCompositionError(
                "joint transformer parameter ID union differs at installation"
            )
        current_modules = _named_module_rows(transformer)
        current_module_ids = {id(module) for _, module in current_modules}
        registered_motion_module_ids = _registered_added_module_ids(
            motion_handle.q_wrappers + motion_handle.o_wrappers,
            preexisting_ids=after_source_module_ids,
        )
        allowed_module_ids = frozenset(
            vendor_module_ids
            | registered_source_module_ids
            | registered_motion_module_ids
        )
        if current_module_ids != allowed_module_ids:
            raise SAICJointCompositionError(
                "joint module union is not vendor plus two registered adapters"
            )

        handle = SAICJointCompositionHandle(
            transformer=transformer,
            mode=mode,
            source_handle=source_handle,
            motion_handle=motion_handle,
            source_install_receipt=source_install_receipt,
            motion_install_receipt=motion_install_receipt,
            source_load_receipt=source_load_receipt,
            motion_load_receipt=motion_load_receipt,
            source_parameters=source_parameters,
            motion_parameters=motion_parameters,
            vendor_parameter_rows=vendor_parameter_rows,
            vendor_module_rows=vendor_module_rows,
            vendor_parameter_bindings=vendor_parameter_bindings,
            after_source_parameter_identity_rows=_parameter_identity_rows(
                after_source_parameters
            ),
            after_source_module_identity_rows=_module_identity_rows(
                after_source_modules
            ),
            active_parameter_identity_rows=_parameter_identity_rows(
                current_parameters
            ),
            active_module_identity_rows=_module_identity_rows(current_modules),
            source_parameter_bindings=_parameter_binding_map(source_parameters),
            motion_parameter_bindings=_parameter_binding_map(motion_parameters),
            allowed_parameter_ids=allowed_parameter_ids,
            allowed_module_ids=allowed_module_ids,
            source_state_sha256=source_state_sha256,
            motion_state_sha256=motion_state_sha256,
            construction_token=_CONSTRUCTION_MINT,
        )
        with _TRUST_LOCK:
            birth = _mint_handle_birth_seal(handle)
            _register_handle_trust(handle, _HandleTrustState(birth=birth))
        handle.audit()
        return handle
    except Exception as error:
        cleanup_errors: list[str] = []
        if motion_handle is not None and not motion_handle.restored:
            try:
                motion_handle.restore()
            except Exception as cleanup_error:  # pragma: no cover - catastrophic drift.
                cleanup_errors.append(f"motion={cleanup_error}")
        if source_handle is not None and not source_handle.restored:
            try:
                source_handle.restore()
            except Exception as cleanup_error:  # pragma: no cover - catastrophic drift.
                cleanup_errors.append(f"source={cleanup_error}")
        suffix = "" if not cleanup_errors else f"; cleanup failed: {cleanup_errors}"
        if isinstance(error, SAICJointCompositionError):
            raise SAICJointCompositionError(f"{error}{suffix}") from error
        raise SAICJointCompositionError(
            f"joint composition installation failed: {error}{suffix}"
        ) from error


__all__ = [
    "ALLOWED_MODES",
    "CLASSIFICATION",
    "INFERENCE",
    "NATIVE_RUNTIME_STATUS",
    "SAICJointCompositionError",
    "SAICJointCompositionHandle",
    "SAICJointCompositionRestoreError",
    "SAICJointCompositionRuntimeLease",
    "SCHEMA_VERSION",
    "STAGE_B_TRAIN",
    "install_saic_joint_composition",
]
