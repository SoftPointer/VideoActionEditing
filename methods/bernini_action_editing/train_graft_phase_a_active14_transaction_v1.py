#!/usr/bin/env python3
"""Fail-closed in-memory optimizer transaction over Phase-A indices 26..39.

This core is deliberately narrower than a training program.  It accepts a
completed, sealed Field14 qualification and a *same-process* Field14 receipt,
then performs exactly one post-bootstrap AdamW update at each active UniPC
coordinate in official order.  The AdamW moments are explicitly fresh after
the preceding short/Field14 replay; dependency jobs never provide weights or
optimizer state.

The transaction snapshots every trainable tensor before its first update.  A
bad native cell, collective, parameter/base invariant, route receipt, or
downstream continuation restores that snapshot, clears gradients and optimizer
state, and raises with a sealed failure receipt.  A successful transaction is
committed only in process memory.  There is no checkpoint or publication API,
and all semantic/scientific authority remains false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

import graft_phase_a_field14_exact40_v1 as field14
import graft_phase_a_native_training_closure_v2 as native_v2
import train_graft_phase_a_a_lite_short_v1 as short_trainer


SCHEMA_VERSION = "bernini-graft-phase-a-active14-transaction-v1"
FAILURE_SCHEMA_VERSION = "bernini-graft-phase-a-active14-transaction-failure-v1"
PLAN_SCHEMA_VERSION = "bernini-graft-phase-a-active14-cell-plan-v1"
UPSTREAM_SCHEMA_VERSION = "bernini-graft-phase-a-field14-world8-parent-v2"
LOCAL_FIELD14_SCHEMA_VERSION = field14.SCHEMA_VERSION

WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
ACTIVE_INDICES = tuple(range(26, 40))
TRAINING_REGIME = "post_bootstrap"
OPTIMIZER_LEARNING_RATE = 1.0e-3
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_EPS = 1.0e-8
OPTIMIZER_WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0

AUTHORITY_FIELDS = tuple(short_trainer.AUTHORITY_FIELDS)


class Active14TransactionError(RuntimeError):
    """A transaction rejected its inputs or rolled its trainables back."""

    def __init__(self, message: str, *, failure_receipt: Any = None) -> None:
        super().__init__(message)
        self.failure_receipt = failure_receipt


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise Active14TransactionError("canonical mapping key differs")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise Active14TransactionError("value is not canonical JSON data")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise Active14TransactionError("value is not finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Active14TransactionError(f"{label} must be lowercase SHA256")
    return value


def seal_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = dict(value)
    if "digest" in plain:
        raise Active14TransactionError("mapping is already sealed")
    plain["digest"] = object_sha256(plain)
    return MappingProxyType(plain)


def validate_sealed_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Active14TransactionError(f"{label} must be a mapping")
    plain = dict(value)
    digest = plain.pop("digest", None)
    if type(digest) is not str or digest != object_sha256(plain):
        raise Active14TransactionError(f"{label} digest differs")
    plain["digest"] = digest
    return plain


def _false_authority() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_FIELDS}


def assert_no_authority(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "authority":
                if (
                    not isinstance(item, Mapping)
                    or any(flag is not False for flag in item.values())
                ):
                    raise Active14TransactionError("authority mapping differs")
            elif (
                key in AUTHORITY_FIELDS
                or key.endswith("_authorized")
                or "authority" in key
            ) and item is not False:
                raise Active14TransactionError(
                    f"authority-shaped field is not false: {key}"
                )
            if key in {
                "checkpoint_written",
                "checkpoint_payload_returned",
                "publication_performed",
            } and item is not False:
                raise Active14TransactionError("artifact publication differs")
            assert_no_authority(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_no_authority(item)


def validate_upstream_field14_parent(
    value: Any,
    *,
    expected_job_id: str,
) -> Mapping[str, Any]:
    """Validate the create-only Field14 parent receipt used as queue evidence."""

    if not isinstance(value, Mapping):
        raise Active14TransactionError("upstream Field14 parent is not a mapping")
    receipt = dict(value)
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_digest", None)
    expected_keys = {
        "schema_version",
        "status",
        "complete",
        "pass",
        "job_id",
        "runner_result_digest",
        "runtime",
        "validated",
        "runner_result",
        "checkpoint_written",
        "checkpoint_payload_returned",
        "publication_performed",
        "authority",
        "receipt_publication",
        "receipt_digest",
    }
    if (
        set(receipt) != expected_keys
        or claimed != object_sha256(unsigned)
        or receipt.get("schema_version") != UPSTREAM_SCHEMA_VERSION
        or receipt.get("status") != "completed_diagnostic_no_checkpoint"
        or receipt.get("complete") is not True
        or receipt.get("pass") is not True
        or receipt.get("job_id") != expected_job_id
        or receipt.get("checkpoint_written") is not False
        or receipt.get("checkpoint_payload_returned") is not False
        or receipt.get("publication_performed") is not False
    ):
        raise Active14TransactionError("upstream Field14 parent contract differs")
    _require_sha256(receipt.get("runner_result_digest"), label="runner result digest")
    validated = receipt.get("validated")
    required_true = {
        "same_process_short_then_exact40",
        "exact40_official_order",
        "inactive_0_25_zero_gate_preinstall_parity_producer_validated",
        "active_26_39_finite_nonzero_gate_producer_validated",
        "all_eight_full_field14_receipts_deeply_validated",
        "both_sp4_per_index_field_hash_and_metric_consensus_recomputed",
        "rank_log_root_identity_retained_from_before_torchrun",
        "rank_stdout_opened_relative_to_retained_root_with_O_NOFOLLOW",
        "output_root_identity_retained_from_creation",
        "checkpoint_content_full_rehash_pre_and_post",
        "per_index_hash_then_release",
    }
    if (
        not isinstance(validated, Mapping)
        or any(validated.get(name) is not True for name in required_true)
        or validated.get("cross_index_compensation_or_selection") is not False
        or validated.get("semantic_metrics_authoritative") is not False
    ):
        raise Active14TransactionError("upstream Field14 validation differs")
    if receipt.get("receipt_publication") != {
        "create_only_O_EXCL": True,
        "provisional_mode": "0600",
        "final_success_mode": "0444",
        "mode_0444_is_terminal_success_transition": True,
        "canonical_ascii_json_newline": True,
        "output_root_identity_retained_from_creation": True,
    }:
        raise Active14TransactionError("upstream Field14 publication differs")
    runner = validate_sealed_mapping(
        receipt.get("runner_result"), label="upstream Field14 runner result"
    )
    if (
        runner["digest"] != receipt["runner_result_digest"]
        or runner.get("status")
        != "completed_in_memory_short_then_exact40_no_checkpoint"
        or runner.get("checkpoint_written") is not False
        or runner.get("publication_performed") is not False
    ):
        raise Active14TransactionError("upstream Field14 runner result differs")
    world8 = validate_sealed_mapping(
        runner.get("world8"), label="upstream Field14 WORLD8 result"
    )
    if (
        world8.get("all_eight_exact40_completed") is not True
        or world8.get("both_sp4_arms_exact_field_hash_and_metric_consensus") is not True
        or world8.get("all_eight_trainable_bytes_unchanged_during_sweep") is not True
        or world8.get("all_eight_base_bytes_unchanged_entire_process") is not True
    ):
        raise Active14TransactionError("upstream Field14 WORLD8 closure differs")
    assert_no_authority(receipt)
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-active14-upstream-qualification-v1",
            "field14_job_id": expected_job_id,
            "field14_parent_receipt_digest": claimed,
            "field14_runner_result_digest": runner["digest"],
            "field14_world8_result_digest": world8["digest"],
            "slurm_afterok_is_queue_gate_only": True,
            "weights_inherited_from_dependency_job": False,
            "optimizer_state_inherited_from_dependency_job": False,
            "checkpoint_consumed_from_dependency_job": False,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def validate_local_field14_receipt(value: Any) -> Mapping[str, Any]:
    """Admit the Field14 sweep rerun immediately before this transaction."""

    receipt = validate_sealed_mapping(value, label="same-process Field14 receipt")
    if (
        receipt.get("schema_version") != LOCAL_FIELD14_SCHEMA_VERSION
        or receipt.get("status")
        != "completed_in_memory_exact40_no_grad_no_checkpoint"
        or receipt.get("schedule_indices") != list(range(40))
        or receipt.get("inactive_indices") != list(range(26))
        or receipt.get("active_indices") != list(ACTIVE_INDICES)
        or receipt.get("exact40_official_order") is not True
        or receipt.get("ambient_torch_no_grad") is not True
        or receipt.get("one_index_admitted_hashed_and_released_before_next")
        is not True
        or receipt.get("cross_index_tensor_retention") is not False
        or receipt.get("cross_index_compensation_used") is not False
        or receipt.get("cross_index_selection_used") is not False
        or receipt.get("checkpoint_written") is not False
        or receipt.get("publication_performed") is not False
        or len(receipt.get("rows", ())) != 40
    ):
        raise Active14TransactionError("same-process Field14 receipt differs")
    for index, row in enumerate(receipt["rows"]):
        if (
            not isinstance(row, Mapping)
            or row.get("schedule_index") != index
            or row.get("all_field_tensor_objects_released") is not True
        ):
            raise Active14TransactionError("same-process Field14 row differs")
    assert_no_authority(receipt)
    return receipt


@dataclass(frozen=True, init=False)
class Active14CellPlan:
    update_number: int
    schedule_index: int
    training_regime: str
    dp_arm: int
    row_iid: str
    row_source_sha256: str
    plan_digest: str
    _token: object = field(repr=False, compare=False)

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "Active14CellPlan":
        raise Active14TransactionError("active14 plans are transaction-minted")

    @classmethod
    def _mint(cls, **values: Any) -> "Active14CellPlan":
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "_token", _PLAN_TOKEN)
        return instance


@dataclass(frozen=True)
class Active14TransactionResult:
    receipt: Mapping[str, Any]
    active14_commit_receipt: Mapping[str, Any]
    preparation_receipt: Mapping[str, Any]
    finalize_receipt: Mapping[str, Any]
    checkpoint_payload: None = None
    publication_payload: None = None


_PLAN_TOKEN = object()


class AuthenticatedActive14Services:
    """Opaque callbacks for native cells and route provenance."""

    __slots__ = (
        "_make_update_cell",
        "_after_update",
        "_assert_schedule_unchanged",
        "_test_only",
        "_receipt",
        "_token",
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise Active14TransactionError("active14 services require authentication")

    @classmethod
    def _mint(
        cls,
        *,
        token: object,
        make_update_cell: Callable[[Active14CellPlan], Any],
        after_update: Callable[[Active14CellPlan, Mapping[str, Any]], Mapping[str, Any]],
        assert_schedule_unchanged: Callable[[], None],
        test_only: bool,
        receipt: Mapping[str, Any],
    ) -> "AuthenticatedActive14Services":
        if token is not _SERVICES_TOKEN:
            raise Active14TransactionError("active14 service mint differs")
        if not all(
            callable(value)
            for value in (make_update_cell, after_update, assert_schedule_unchanged)
        ):
            raise Active14TransactionError("active14 service callback differs")
        instance = object.__new__(cls)
        instance._make_update_cell = make_update_cell
        instance._after_update = after_update
        instance._assert_schedule_unchanged = assert_schedule_unchanged
        instance._test_only = test_only
        instance._receipt = receipt
        instance._token = _SERVICES_TOKEN
        instance.assert_live()
        return instance

    @property
    def test_only(self) -> bool:
        return self._test_only

    def assert_live(self) -> None:
        if (
            self._token is not _SERVICES_TOKEN
            or type(self._test_only) is not bool
            or not all(
                callable(value)
                for value in (
                    self._make_update_cell,
                    self._after_update,
                    self._assert_schedule_unchanged,
                )
            )
        ):
            raise Active14TransactionError("active14 services changed")
        validate_sealed_mapping(self._receipt, label="active14 services")

    def make_update_cell(self, plan: Active14CellPlan) -> Any:
        self.assert_live()
        return self._make_update_cell(plan)

    def after_update(
        self, plan: Active14CellPlan, update_receipt: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.assert_live()
        return self._after_update(plan, update_receipt)

    def assert_schedule_unchanged(self) -> None:
        self.assert_live()
        self._assert_schedule_unchanged()

    def receipt(self) -> Mapping[str, Any]:
        self.assert_live()
        return self._receipt


_SERVICES_TOKEN = object()


def authenticate_official_services(
    *,
    make_update_cell: Callable[[Active14CellPlan], Any],
    after_update: Callable[[Active14CellPlan, Mapping[str, Any]], Mapping[str, Any]],
    assert_schedule_unchanged: Callable[[], None],
) -> AuthenticatedActive14Services:
    return AuthenticatedActive14Services._mint(
        token=_SERVICES_TOKEN,
        make_update_cell=make_update_cell,
        after_update=after_update,
        assert_schedule_unchanged=assert_schedule_unchanged,
        test_only=False,
        receipt=seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-active14-services-v1",
                "kind": "official-native-v2-world8-dp2sp4",
                "test_only": False,
                "active_indices": list(ACTIVE_INDICES),
                "checkpoint_written": False,
                "publication_performed": False,
                **_false_authority(),
            }
        ),
    )


def authenticate_cpu_test_services(
    *,
    test_name: str,
    make_update_cell: Callable[[Active14CellPlan], Any],
    after_update: Callable[[Active14CellPlan, Mapping[str, Any]], Mapping[str, Any]],
    assert_schedule_unchanged: Callable[[], None] = lambda: None,
) -> AuthenticatedActive14Services:
    if not isinstance(test_name, str) or not test_name.startswith("cpu_fake_"):
        raise Active14TransactionError("CPU test service name differs")
    return AuthenticatedActive14Services._mint(
        token=_SERVICES_TOKEN,
        make_update_cell=make_update_cell,
        after_update=after_update,
        assert_schedule_unchanged=assert_schedule_unchanged,
        test_only=True,
        receipt=seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-active14-services-v1",
                "kind": test_name,
                "test_only": True,
                "active_indices": list(ACTIVE_INDICES),
                "checkpoint_written": False,
                "publication_performed": False,
                **_false_authority(),
            }
        ),
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    # Reuse the already pinned chunked ctypes reader: it supports scalars,
    # BF16, CUDA/ROCm tensors, and does not depend on NumPy's ABI.
    return short_trainer._tensor_bytes_sha256(value)  # noqa: SLF001


def _registry_digest(named: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    return object_sha256(
        [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "tensor_sha256": _tensor_sha256(parameter),
            }
            for name, parameter in named
        ]
    )


def _trainable_registry(
    bindings: native_v2.AuthenticatedNativeBindings,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    rows = tuple(bindings.named_trainable_parameters)
    if (
        not rows
        or len({name for name, _ in rows}) != len(rows)
        or len({id(parameter) for _, parameter in rows}) != len(rows)
        or any(
            type(name) is not str
            or type(parameter) is not torch.nn.Parameter
            or not parameter.requires_grad
            or not parameter.is_floating_point()
            for name, parameter in rows
        )
    ):
        raise Active14TransactionError("active14 trainable registry differs")
    return rows


def _frozen_registry(
    bindings: native_v2.AuthenticatedNativeBindings,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    trainable_ids = {id(parameter) for _, parameter in bindings.named_trainable_parameters}
    seen: set[int] = set()
    rows: list[tuple[str, torch.nn.Parameter]] = []
    owners = (
        ("diffusion", bindings.diffusion),
        ("transformer", bindings.transformer),
        *bindings.external_trainable_owner_modules,
    )
    for owner_name, module in owners:
        for name, parameter in module.named_parameters(recurse=True):
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            if identity in trainable_ids:
                continue
            if parameter.requires_grad or parameter.grad is not None:
                raise Active14TransactionError("frozen base is trainable or has grad")
            rows.append((f"{owner_name}.{name}", parameter))
    if not rows:
        raise Active14TransactionError("active14 frozen registry is empty")
    return tuple(rows)


def _category(name: str) -> str:
    if name.startswith("atlas_encoder."):
        return "atlas_encoder"
    for projection in ("query", "key", "value", "output"):
        if name.endswith(f".identity_rebinder.{projection}.weight"):
            return f"{projection}_projection"
    raise Active14TransactionError(f"unknown trainable category: {name}")


def _synchronize_gradients(
    *,
    named: Sequence[tuple[str, torch.nn.Parameter]],
    backend: short_trainer.AuthenticatedDP2SP4Backend,
    update_number: int,
) -> Mapping[str, Any]:
    category_rows: dict[str, list[torch.nn.Parameter]] = {
        name: [] for name in native_v2.GRADIENT_CATEGORIES
    }
    materialized = []
    for name, parameter in named:
        category_rows[_category(name)].append(parameter)
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
            materialized.append(name)
        if (
            parameter.grad.shape != parameter.shape
            or parameter.grad.dtype != parameter.dtype
            or parameter.grad.device != parameter.device
            or not bool(torch.isfinite(parameter.grad).all().item())
        ):
            raise Active14TransactionError("local active14 gradient differs")
        _reduce_gradient_raw(
            backend,
            parameter.grad,
            axis="sp",
            parameter_name=name,
            update_number=update_number,
        )
        parameter.grad.div_(float(SP_SIZE))
        _reduce_gradient_raw(
            backend,
            parameter.grad,
            axis="dp",
            parameter_name=name,
            update_number=update_number,
        )
        parameter.grad.div_(float(DP_SIZE))
    norms = {}
    for category, parameters in category_rows.items():
        squared = sum(
            float(parameter.grad.detach().float().square().sum().item())
            for parameter in parameters
        )
        norm = math.sqrt(squared)
        if not math.isfinite(norm) or norm <= 0.0:
            raise Active14TransactionError(
                f"post-bootstrap category gradient is zero/nonfinite: {category}"
            )
        norms[category] = norm
    total = math.sqrt(sum(value * value for value in norms.values()))
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-active14-gradient-sync-v1",
            "update_number": update_number,
            "training_regime": TRAINING_REGIME,
            "collective_order": ["SP4_SUM", "divide_by_4", "DP2_SUM", "divide_by_2"],
            "none_materialized_parameter_names": materialized,
            "category_l2_float64_hex": {
                name: value.hex() for name, value in norms.items()
            },
            "preclip_l2_float64_hex": total.hex(),
            "all_five_categories_finite_nonzero": True,
        }
    )


def _reduce_gradient_raw(
    backend: short_trainer.AuthenticatedDP2SP4Backend,
    value: torch.Tensor,
    *,
    axis: str,
    parameter_name: str,
    update_number: int,
) -> None:
    """Use groups authenticated once at transaction open without rehashing per tensor."""

    if axis not in {"sp", "dp"}:
        raise Active14TransactionError("active14 collective axis differs")
    if backend.test_only:
        reducer = backend._test_reduce  # noqa: SLF001
        if not callable(reducer):
            raise Active14TransactionError("active14 CPU reducer differs")
        reducer(value, axis, parameter_name, update_number)
    else:
        import torch.distributed as dist

        group = backend._sp_group if axis == "sp" else backend._dp_group  # noqa: SLF001
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    if not bool(torch.isfinite(value).all().item()):
        raise Active14TransactionError("active14 collective produced nonfinite gradient")


def _consensus_raw(
    backend: short_trainer.AuthenticatedDP2SP4Backend,
    value: str,
    *,
    scope: str,
    label: str,
) -> str:
    """Consensus on immutable groups already authenticated at transaction open."""

    if scope not in {"world", "sp"}:
        raise Active14TransactionError("active14 consensus scope differs")
    if backend.test_only:
        consensus = backend._test_consensus  # noqa: SLF001
        if not callable(consensus) or consensus(value, scope, label) != value:
            raise Active14TransactionError(f"{label} consensus differs")
        return value
    import torch.distributed as dist

    group = backend._world_group if scope == "world" else backend._sp_group  # noqa: SLF001
    count = WORLD_SIZE if scope == "world" else SP_SIZE
    gathered: list[Any] = [None] * count
    dist.all_gather_object(gathered, value, group=group)
    if any(item != value for item in gathered):
        raise Active14TransactionError(f"{label} differs across {scope}")
    return value


def _validate_native_result(
    *,
    result: Any,
    bindings: native_v2.AuthenticatedNativeBindings,
    backend: short_trainer.AuthenticatedDP2SP4Backend,
    plan: Active14CellPlan,
) -> Mapping[str, Any]:
    if type(result) is not native_v2.NativeTrainingClosureResult:
        raise Active14TransactionError("active14 native result type differs")
    receipt = validate_sealed_mapping(result.receipt, label="active14 native result")
    loss = float(result.flow_matching_loss)
    expected_target = backend.sp_rank >= 2
    local_target_rows = receipt.get("local_target_rows")
    denied = (
        "optimizer_created",
        "parameters_updated",
        "scheduler_step_called",
        "outer_clean_state_transport_used",
        "external_guided_clean_cotangent_accepted",
        "target_video_used",
        "mask_used",
        "pose_used",
        "track_used",
        "optical_flow_used",
        "motion_donor_used",
        "full_sampler_trajectory_verified",
        "training_quality_claim_authorized",
        "scientific_action_editing_claim_authorized",
        "optimizer_step_verified_by_this_core",
        "two_consecutive_steps_verified_by_this_core",
        "post_bootstrap_cuda_short_training_verified_by_this_core",
        "short_training_claim_authorized",
    )
    if (
        not math.isfinite(loss)
        or loss < 0.0
        or receipt.get("schema_version") != native_v2.SCHEMA_VERSION
        or receipt.get("binding_receipt_digest") != bindings.receipt()["digest"]
        or receipt.get("schedule_index") != plan.schedule_index
        or receipt.get("training_regime") != TRAINING_REGIME
        or receipt.get("schedule_cell_active_for_training") is not True
        or receipt.get("schedule_cell_counted_as_trained") is not True
        or receipt.get("phase_a_objective") != native_v2.FLOW_MATCHING_OBJECTIVE
        or receipt.get("supervision_pair") != "same_source_video_noop"
        or receipt.get("frame_count") != native_v2.EXPECTED_FRAMES
        or receipt.get("local_sequence_parallel_rank") != backend.sp_rank
        or receipt.get("local_sequence_parallel_size") != SP_SIZE
        or type(local_target_rows) is not int
        or local_target_rows < 0
        or (local_target_rows > 0) is not expected_target
        or receipt.get("local_adapter_graph_bearing") is not expected_target
        or receipt.get("trainable_registry_values_unchanged") is not True
        or receipt.get("exclusive_trainable_scope_is_exact_authenticated_registry")
        is not True
        or any(receipt.get(name) is not False for name in denied)
    ):
        raise Active14TransactionError("active14 native result contract differs")
    if expected_target:
        local_gradient_gate = (
            "post_bootstrap_target_rows_all_five_categories_finite_nonzero"
        )
        expected_gradient_gates = {
            "bootstrap_output_only_gate_verified": False,
            "post_bootstrap_five_category_local_gate_verified": True,
            "source_only_sp_all_five_categories_exact_zero_verified": False,
        }
    else:
        local_gradient_gate = (
            "source_only_sp_rank_all_five_categories_exact_zero"
        )
        expected_gradient_gates = {
            "bootstrap_output_only_gate_verified": False,
            "post_bootstrap_five_category_local_gate_verified": False,
            "source_only_sp_all_five_categories_exact_zero_verified": True,
        }
    if (
        sum(receipt.get(name) is True for name in expected_gradient_gates) != 1
        or any(
            receipt.get(name) is not expected
            for name, expected in expected_gradient_gates.items()
        )
    ):
        raise Active14TransactionError("active14 post-bootstrap v2 gate differs")
    assert_no_authority(receipt)
    return seal_mapping(
        {
            "schema_version": "bernini-graft-phase-a-active14-native-admission-v2",
            "update_number": plan.update_number,
            "schedule_index": plan.schedule_index,
            "training_regime": TRAINING_REGIME,
            "row_iid": plan.row_iid,
            "row_source_sha256": plan.row_source_sha256,
            "native_receipt_digest": receipt["digest"],
            "flow_matching_loss_float64_hex": loss.hex(),
            "local_sp_rank": backend.sp_rank,
            "local_target_rows": local_target_rows,
            "local_target_owner": expected_target,
            "local_gradient_gate": local_gradient_gate,
            "checkpoint_written": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )


def _restore_snapshot(
    snapshot: Sequence[tuple[torch.nn.Parameter, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
) -> None:
    failure: Optional[BaseException] = None
    for parameter, before in snapshot:
        try:
            with torch.no_grad():
                parameter.copy_(before)
            parameter.grad = None
        except BaseException as error:  # pragma: no cover - catastrophic device loss
            failure = error
    optimizer.state.clear()
    if failure is not None:
        raise Active14TransactionError("active14 rollback could not restore snapshot") from failure


def execute_active14_transaction(
    *,
    upstream_qualification: Mapping[str, Any],
    local_field14_receipt: Mapping[str, Any],
    short_result_digest: str,
    family: str,
    row_iid: str,
    row_source_sha256: str,
    bindings: native_v2.AuthenticatedNativeBindings,
    backend: short_trainer.AuthenticatedDP2SP4Backend,
    services: AuthenticatedActive14Services,
    prepare: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    finalize: Callable[
        [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ],
) -> Active14TransactionResult:
    """Execute 14 updates plus downstream prepare/finalize as one transaction."""

    qualification = validate_sealed_mapping(
        upstream_qualification, label="active14 upstream qualification"
    )
    local_field = validate_local_field14_receipt(local_field14_receipt)
    _require_sha256(short_result_digest, label="short result digest")
    _require_sha256(row_source_sha256, label="active14 row source SHA256")
    if (
        qualification.get("schema_version")
        != "bernini-graft-phase-a-active14-upstream-qualification-v1"
        or qualification.get("slurm_afterok_is_queue_gate_only") is not True
        or qualification.get("weights_inherited_from_dependency_job") is not False
        or qualification.get("optimizer_state_inherited_from_dependency_job")
        is not False
        or qualification.get("checkpoint_consumed_from_dependency_job") is not False
        or qualification.get("checkpoint_written") is not False
        or qualification.get("publication_performed") is not False
        or
        type(bindings) is not native_v2.AuthenticatedNativeBindings
        or type(backend) is not short_trainer.AuthenticatedDP2SP4Backend
        or type(services) is not AuthenticatedActive14Services
        or services.test_only is not bindings.test_only
        or services.test_only is not backend.test_only
        or not callable(prepare)
        or not callable(finalize)
        or local_field.get("short_result_digest") != short_result_digest
        or local_field.get("family") != family
        or local_field.get("wrong_owner_iid") != row_iid
    ):
        raise Active14TransactionError("active14 execution binding differs")
    assert_no_authority(qualification)
    backend.assert_live()
    services.assert_live()
    named = _trainable_registry(bindings)
    frozen = _frozen_registry(bindings)
    initial_trainable_digest = _registry_digest(named)
    initial_base_digest = _registry_digest(frozen)
    _consensus_raw(
        backend,
        initial_trainable_digest,
        scope="world",
        label="active14 initial trainables",
    )
    _consensus_raw(
        backend,
        initial_base_digest,
        scope="world",
        label="active14 initial base",
    )
    snapshot = tuple((parameter, parameter.detach().clone()) for _, parameter in named)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=OPTIMIZER_LEARNING_RATE,
        betas=OPTIMIZER_BETAS,
        eps=OPTIMIZER_EPS,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        foreach=False,
    )
    updates: list[Mapping[str, Any]] = []
    phase = "active14_updates"
    try:
        for update_number, schedule_index in enumerate(ACTIVE_INDICES, start=1):
            if any(parameter.grad is not None for _, parameter in named):
                raise Active14TransactionError("active14 inherited stale gradient")
            plan_body = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "update_number": update_number,
                "schedule_index": schedule_index,
                "training_regime": TRAINING_REGIME,
                "dp_arm": backend.dp_arm,
                "row_iid": row_iid,
                "row_source_sha256": row_source_sha256,
                "official_active14_order": True,
            }
            plan = Active14CellPlan._mint(
                update_number=update_number,
                schedule_index=schedule_index,
                training_regime=TRAINING_REGIME,
                dp_arm=backend.dp_arm,
                row_iid=row_iid,
                row_source_sha256=row_source_sha256,
                plan_digest=object_sha256(plan_body),
            )
            cell = services.make_update_cell(plan)
            if (
                type(cell) is not native_v2.PhaseANativeTrainingClosure
                or cell.bindings is not bindings
                or cell.schedule_index != schedule_index
                or cell.training_regime != TRAINING_REGIME
                or cell.phase != "new"
            ):
                raise Active14TransactionError("active14 native cell binding differs")
            before_trainable = _registry_digest(named)
            before_base = _registry_digest(frozen)
            cell.measure()
            cell.derive_phase_a_flow_matching_vjp()
            native_result = cell.replay_and_backward()
            admission = _validate_native_result(
                result=native_result,
                bindings=bindings,
                backend=backend,
                plan=plan,
            )
            if any(parameter.grad is not None for _, parameter in frozen):
                raise Active14TransactionError("frozen base acquired a gradient")
            sync = _synchronize_gradients(
                named=named, backend=backend, update_number=update_number
            )
            preclip = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], MAX_GRAD_NORM
            )
            preclip_float = float(preclip.item())
            if not math.isfinite(preclip_float) or preclip_float <= 0.0:
                raise Active14TransactionError("active14 clip norm differs")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            after_trainable = _registry_digest(named)
            after_base = _registry_digest(frozen)
            if (
                after_trainable == before_trainable
                or after_base != before_base
                or after_base != initial_base_digest
                or any(parameter.grad is not None for _, parameter in named)
                or any(parameter.grad is not None for _, parameter in frozen)
            ):
                raise Active14TransactionError("active14 update invariant differs")
            _consensus_raw(
                backend,
                after_trainable,
                scope="world",
                label=f"active14 update {update_number}",
            )
            _consensus_raw(
                backend,
                after_base,
                scope="world",
                label=f"active14 base {update_number}",
            )
            route = validate_sealed_mapping(
                services.after_update(plan, admission),
                label=f"active14 update {update_number} route",
            )
            if (
                route.get("update_number") != update_number
                or route.get("schedule_index") != schedule_index
                or route.get("row_iid") != row_iid
                or route.get("exact_four_native_forwards") is not True
                or route.get("fit_row_only") is not True
                or route.get("checkpoint_written") is not False
                or route.get("publication_performed") is not False
            ):
                raise Active14TransactionError("active14 route receipt differs")
            assert_no_authority(route)
            updates.append(
                seal_mapping(
                    {
                        "schema_version": "bernini-graft-phase-a-active14-update-v1",
                        **plan_body,
                        "plan_digest": plan.plan_digest,
                        "native_admission_digest": admission["digest"],
                        "gradient_synchronization_digest": sync["digest"],
                        "route_receipt_digest": route["digest"],
                        "preclip_l2_float64_hex": preclip_float.hex(),
                        "parameter_digest_before": before_trainable,
                        "parameter_digest_after": after_trainable,
                        "frozen_base_digest": after_base,
                        "checkpoint_written": False,
                        "publication_performed": False,
                        **_false_authority(),
                    }
                )
            )
        services.assert_schedule_unchanged()
        final_trainable_digest = _registry_digest(named)
        final_base_digest = _registry_digest(frozen)
        if (
            len(updates) != len(ACTIVE_INDICES)
            or final_trainable_digest == initial_trainable_digest
            or final_base_digest != initial_base_digest
        ):
            raise Active14TransactionError("active14 final state differs")
        phase = "downstream_prepare"
        preliminary = seal_mapping(
            {
                "schema_version": "bernini-graft-phase-a-active14-precommit-v1",
                "status": "active14_updates_complete_downstream_prepare_pending",
                "active_indices": list(ACTIVE_INDICES),
                "update_receipt_digests": [row["digest"] for row in updates],
                "initial_trainable_digest": initial_trainable_digest,
                "final_trainable_digest": final_trainable_digest,
                "frozen_base_digest": final_base_digest,
                "checkpoint_written": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        preparation_receipt = validate_sealed_mapping(
            prepare(preliminary), label="active14 downstream preparation"
        )
        assert_no_authority(preparation_receipt)
        if (
            preparation_receipt.get("preparation_completed") is not True
            or preparation_receipt.get("published") is not False
            or preparation_receipt.get("checkpoint_written") is not False
            or preparation_receipt.get("publication_performed") is not False
        ):
            raise Active14TransactionError("active14 downstream preparation differs")
        if (
            _registry_digest(named) != final_trainable_digest
            or _registry_digest(frozen) != final_base_digest
        ):
            raise Active14TransactionError("downstream preparation mutated active14 state")
        active14_commit_receipt = seal_mapping(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_in_memory_active14_transaction_no_checkpoint",
                "topology": {
                    "world_size": WORLD_SIZE,
                    "dp_size": DP_SIZE,
                    "sp_size": SP_SIZE,
                    "rank": backend.rank,
                    "dp_arm": backend.dp_arm,
                    "sp_rank": backend.sp_rank,
                },
                "family": family,
                "row_iid": row_iid,
                "row_source_sha256": row_source_sha256,
                "upstream_qualification_digest": qualification["digest"],
                "same_process_short_result_digest": short_result_digest,
                "same_process_field14_result_digest": local_field["digest"],
                "same_process_short_then_field14_then_active14": True,
                "weights_inherited_from_dependency_job": False,
                "checkpoint_reloaded_between_local_stages": False,
                "optimizer_contract": {
                    "kind": "torch.optim.AdamW",
                    "moments": "fresh-zero-state-after-short-and-field14",
                    "moments_inherited_from_short": False,
                    "learning_rate": OPTIMIZER_LEARNING_RATE,
                    "betas": list(OPTIMIZER_BETAS),
                    "eps": OPTIMIZER_EPS,
                    "weight_decay": OPTIMIZER_WEIGHT_DECAY,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "steps": 14,
                    "schedule_indices": list(ACTIVE_INDICES),
                    "training_regime": TRAINING_REGIME,
                    "gradient_collective_order": ["SP4_SUM_div_4", "DP2_SUM_div_2"],
                },
                "updates": [dict(row) for row in updates],
                "initial_trainable_digest": initial_trainable_digest,
                "final_trainable_digest": final_trainable_digest,
                "initial_frozen_base_digest": initial_base_digest,
                "final_frozen_base_digest": final_base_digest,
                "all_fourteen_updates_completed": True,
                "all_fourteen_post_bootstrap_five_category_gates_passed": True,
                "frozen_base_unchanged": True,
                "transaction_committed_in_memory": True,
                "downstream_preparation_receipt_digest": preparation_receipt[
                    "digest"
                ],
                "downstream_prepared_but_not_published": True,
                "decoded_media_used_by_this_core": False,
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        assert_no_authority(active14_commit_receipt)
        phase = "downstream_finalize"
        finalize_receipt = validate_sealed_mapping(
            finalize(active14_commit_receipt, preparation_receipt),
            label="active14 downstream finalize",
        )
        assert_no_authority(finalize_receipt)
        if (
            finalize_receipt.get("finalize_completed") is not True
            or finalize_receipt.get("active14_commit_receipt_digest")
            != active14_commit_receipt["digest"]
            or finalize_receipt.get("preparation_receipt_digest")
            != preparation_receipt["digest"]
            or finalize_receipt.get("checkpoint_written") is not False
            or finalize_receipt.get("publication_performed") is not False
        ):
            raise Active14TransactionError("active14 downstream finalize differs")
        # Finalize is the transaction's last potentially stateful operation.
        # It must perform all of its own validation before an atomic rename.
        receipt = seal_mapping(
            {
                "schema_version": (
                    "bernini-graft-phase-a-active14-downstream-transaction-v1"
                ),
                "status": "completed_active14_and_downstream_finalize",
                "active14_commit_receipt": active14_commit_receipt,
                "active14_commit_receipt_digest": active14_commit_receipt[
                    "digest"
                ],
                "preparation_receipt": preparation_receipt,
                "preparation_receipt_digest": preparation_receipt["digest"],
                "finalize_receipt": finalize_receipt,
                "finalize_receipt_digest": finalize_receipt["digest"],
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        assert_no_authority(receipt)
        return Active14TransactionResult(
            receipt=receipt,
            active14_commit_receipt=active14_commit_receipt,
            preparation_receipt=preparation_receipt,
            finalize_receipt=finalize_receipt,
        )
    except BaseException as error:
        rollback_error: Optional[BaseException] = None
        try:
            _restore_snapshot(snapshot, optimizer)
            restored = _registry_digest(named) == initial_trainable_digest
        except BaseException as restore_error:  # pragma: no cover
            rollback_error = restore_error
            restored = False
        failure = seal_mapping(
            {
                "schema_version": FAILURE_SCHEMA_VERSION,
                "status": "failed_rolled_back_no_checkpoint",
                "failure_phase": phase,
                "error": f"{type(error).__name__}:{error}",
                "rollback_error": (
                    None
                    if rollback_error is None
                    else f"{type(rollback_error).__name__}:{rollback_error}"
                ),
                "completed_update_count_before_failure": len(updates),
                "completed_schedule_indices": [
                    row["schedule_index"] for row in updates
                ],
                "trainable_parameters_restored_to_transaction_snapshot": restored,
                "optimizer_state_cleared": rollback_error is None,
                "failed_state_eligible_for_continuation": False,
                "checkpoint_written": False,
                "checkpoint_payload_returned": False,
                "publication_performed": False,
                **_false_authority(),
            }
        )
        message = "active14 transaction failed and rolled back"
        if rollback_error is not None:
            message = "active14 transaction failed and rollback could not be proven"
        raise Active14TransactionError(
            message, failure_receipt=failure
        ) from error


__all__ = [
    "ACTIVE_INDICES",
    "AUTHORITY_FIELDS",
    "Active14CellPlan",
    "Active14TransactionError",
    "Active14TransactionResult",
    "AuthenticatedActive14Services",
    "FAILURE_SCHEMA_VERSION",
    "LOCAL_FIELD14_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "UPSTREAM_SCHEMA_VERSION",
    "authenticate_cpu_test_services",
    "authenticate_official_services",
    "canonical_json_bytes",
    "execute_active14_transaction",
    "object_sha256",
    "seal_mapping",
    "validate_local_field14_receipt",
    "validate_sealed_mapping",
    "validate_upstream_field14_parent",
]
