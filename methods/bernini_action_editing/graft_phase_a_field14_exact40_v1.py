#!/usr/bin/env python3
"""Fail-closed exact40 six-field sweep after the frozen Phase-A short core.

This module does not train, select, compensate across scheduler coordinates, or
write a checkpoint.  It admits one detached six-field packet at a time in the
official UniPC order, hashes it, proves that the tensor objects were released,
and retains only canonical diagnostic receipts.  The live GPU/model operations
are supplied by the versioned field14 runner; this core owns their ordering and
the non-authoritative metric/provenance checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence
import weakref

import torch

import identity_rebinder_v1 as rebinder
import inference_sigma_strata as sigma_strata
import run_graft_phase_a_a_lite_short_gpu_v1 as short_runner


SCHEMA_VERSION = "bernini-graft-phase-a-field14-exact40-core-v1"
INDEX_SCHEMA_VERSION = "bernini-graft-phase-a-field14-index-v1"
PROVENANCE_SCHEMA_VERSION = "bernini-graft-phase-a-field14-provenance-v1"
METRICS_SCHEMA_VERSION = "bernini-graft-phase-a-field14-semantic-metrics-v1"
RELEASE_SCHEMA_VERSION = "bernini-graft-phase-a-field14-tensor-release-v1"

EXACT40_INDICES = tuple(range(40))
INACTIVE_INDICES = tuple(range(26))
ACTIVE_INDICES = tuple(range(26, 40))
FIELD_ROLES = (
    "source_noop_target_velocity",
    "correct_atlas_noop_velocity",
    "wrong_atlas_noop_velocity",
    "dropped_atlas_noop_velocity",
    "correct_atlas_action_velocity",
    "dropped_atlas_action_velocity",
)
RAW_ROLES = (
    "correct_negative",
    "correct_noop",
    "correct_action",
    "wrong_negative",
    "wrong_noop",
    "drop_negative",
    "drop_noop",
    "drop_action",
)
AUTHORITY_FIELDS = tuple(short_runner.AUTHORITY_FIELDS)


class Field14Exact40Error(RuntimeError):
    """Reject a sweep before it can acquire checkpoint or claim authority."""


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Field14Exact40Error(f"{label} must be lowercase SHA256")
    return value


_TENSOR_IDENTITY_KEYS = {
    "shape",
    "dtype",
    "device_type_at_observation",
    "finite",
    "byte_count",
    "raw_sha256",
    "content_sha256",
}
_DTYPE_ELEMENT_BYTES = {
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.complex128": 16,
}


def _validate_tensor_identity_record(
    value: Any,
    *,
    label: str,
    dtype: str,
    device_type: str,
    shape: Callable[[list[int]], bool],
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _TENSOR_IDENTITY_KEYS
        or type(value.get("shape")) is not list
        or any(type(item) is not int or item <= 0 for item in value["shape"])
        or value.get("dtype") != dtype
        or value.get("device_type_at_observation") != device_type
        or value.get("finite") is not True
        or type(value.get("byte_count")) is not int
        or not shape(value["shape"])
    ):
        raise Field14Exact40Error(f"{label} tensor identity differs")
    elements = 1
    for item in value["shape"]:
        elements *= item
    if value["byte_count"] != elements * _DTYPE_ELEMENT_BYTES[dtype]:
        raise Field14Exact40Error(f"{label} tensor byte count differs")
    _require_sha256(value.get("raw_sha256"), label=f"{label} raw SHA256")
    _require_sha256(value.get("content_sha256"), label=f"{label} content SHA256")
    return value


def _plain_json_value(value: Any) -> Any:
    """Own nested sealed mappings/tuples before canonical JSON encoding."""

    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise Field14Exact40Error("canonical JSON mapping key is not a string")
        return {key: _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise Field14Exact40Error("value contains a non-JSON runtime object")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain_json_value(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise Field14Exact40Error("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = dict(value)
    if "digest" in plain:
        raise Field14Exact40Error("sealed mapping already contains digest")
    plain["digest"] = object_sha256(plain)
    return MappingProxyType(plain)


def validate_sealed_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Field14Exact40Error(f"{label} must be a mapping")
    plain = dict(value)
    digest = plain.pop("digest", None)
    if type(digest) is not str or digest != object_sha256(plain):
        raise Field14Exact40Error(f"{label} digest differs")
    plain["digest"] = digest
    return plain


def _false_authority() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_FIELDS}


def _walk_no_authority(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                key in AUTHORITY_FIELDS
                or key.endswith("_authorized")
                or "authority" in key
            ) and item is not False:
                raise Field14Exact40Error("field14 receipt elevated authority")
            _walk_no_authority(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _walk_no_authority(item)


@dataclass(frozen=True)
class Field14TensorSet:
    source_noop_target_velocity: torch.Tensor = field(repr=False, compare=False)
    correct_atlas_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    wrong_atlas_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    dropped_atlas_noop_velocity: torch.Tensor = field(repr=False, compare=False)
    correct_atlas_action_velocity: torch.Tensor = field(repr=False, compare=False)
    dropped_atlas_action_velocity: torch.Tensor = field(repr=False, compare=False)
    provenance: Mapping[str, Any]


def _tensor_rows(value: Field14TensorSet) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple((name, getattr(value, name)) for name in FIELD_ROLES)


_TRUE_FLAGS = (
    "exactly_six_fields",
    "all_fields_detached_fp32_finite_contiguous",
    "all_field_storages_pairwise_disjoint",
    "same_confirmation_source_zs_bytes_all_fields",
    "same_native_full_source_v_pack_bytes_all_model_fields",
    "same_noisy_target_object_and_bytes_all_fields",
    "same_epsilon_bytes_all_fields",
    "same_sigma_timestep_coordinate_all_fields",
    "same_negative_condition_bytes_all_model_fields",
    "correct_atlas_from_confirmation_row",
    "wrong_atlas_from_same_family_fit_row",
    "wrong_intervention_changes_only_identity_atlas_memory",
    "drop_intervention_disables_only_identity_rebinder_residual_and_atlas_route",
    "drop_retains_native_full_source_v_pack",
    "noop_positive_condition_shared_across_correct_wrong_drop",
    "action_positive_condition_shared_across_correct_drop",
    "action_noop_pair_differs_only_in_positive_text_embedding",
    "source_noop_target_velocity_recomputed_from_same_x_sigma_and_source_zs",
    "native_v2v_apg_field_formula_used",
    "per_index_independent_measurement",
)
_FALSE_FLAGS = (
    "confirmation_row_consumed_by_optimizer",
    "wrong_atlas_is_cross_family",
    "native_source_v_pack_dropped",
    "negative_condition_changed_by_intervention",
    "noise_or_coordinate_changed_by_intervention",
    "cross_index_compensation_used",
    "cross_index_selection_used",
    "target_video_used",
    "generated_proposal_used",
    "t2v_branch_used",
    "source_retelling_used",
    "selector_used",
    "mask_pose_track_flow_or_motion_donor_used",
    "checkpoint_written",
    "checkpoint_payload_returned",
    "publication_performed",
)
_RUNTIME_EVIDENCE_KEYS = {
    "coordinate",
    "confirmation_source_state_receipt_digest",
    "wrong_fit_source_state_receipt_digest",
    "epsilon_receipt_digest",
    "noisy_target_receipt_digest",
    "same_state_identities_before_model_fields",
    "same_state_identities_after_all_model_fields",
    "atlas_identities_before_model_fields",
    "atlas_identities_after_all_model_fields",
    "same_state_tensor_identities_recomputed_byte_equal",
    "wrong_route_receipts_differ_only_in_atlas_memory",
    "drop_route_receipts_retain_v_branch_disable_only_rebinder",
    "action_noop_route_receipts_equal_with_negative_raw_reuse",
    "raw_call_order",
    "raw_tensor_identities",
    "route_receipts",
    "expected_enabled_route_gate_float64_hex_recomputed",
    "inactive_raw_parity",
    "ambient_torch_no_grad",
}


def expected_route_gate(schedule_index: int) -> float:
    if type(schedule_index) is not int or schedule_index not in EXACT40_INDICES:
        raise Field14Exact40Error("field14 schedule index differs")
    sigma = sigma_strata.PINNED_POSITIVE_SIGMAS[schedule_index]
    gate = rebinder.mid_low_sigma_gate(sigma)
    if schedule_index in INACTIVE_INDICES:
        if sigma < rebinder.DEFAULT_HIGH_SIGMA or gate != 0.0:
            raise Field14Exact40Error("inactive exact40 gate contract differs")
    elif sigma >= rebinder.DEFAULT_HIGH_SIGMA or not math.isfinite(gate) or gate <= 0.0:
        raise Field14Exact40Error("active exact40 gate contract differs")
    return gate


def build_field14_provenance(
    *,
    schedule_index: int,
    family: str,
    confirmation_iid: str,
    confirmation_source_sha256: str,
    wrong_owner_iid: str,
    wrong_owner_source_sha256: str,
    fields: Mapping[str, torch.Tensor],
    runtime_evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    if tuple(fields) != FIELD_ROLES:
        raise Field14Exact40Error("field14 field inventory/order differs")
    gate = expected_route_gate(schedule_index)
    evidence = dict(runtime_evidence)
    if set(evidence) != _RUNTIME_EVIDENCE_KEYS:
        raise Field14Exact40Error("runtime evidence key schema differs")
    if set(evidence) & (set(_TRUE_FLAGS) | set(_FALSE_FLAGS) | set(AUTHORITY_FIELDS)):
        raise Field14Exact40Error("runtime evidence overrides fixed field14 flags")
    return seal_mapping(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "schedule_index": schedule_index,
            "family": family,
            "confirmation_iid": confirmation_iid,
            "confirmation_source_sha256": confirmation_source_sha256,
            "wrong_owner_iid": wrong_owner_iid,
            "wrong_owner_source_sha256": wrong_owner_source_sha256,
            "field_roles": list(FIELD_ROLES),
            "field_tensor_identities": {
                name: short_runner.tensor_identity(fields[name]) for name in FIELD_ROLES
            },
            "route_regime": (
                "inactive_sigma_ge_0.75_exact_zero"
                if schedule_index in INACTIVE_INDICES
                else "active_sigma_lt_0.75_finite_nonzero"
            ),
            "expected_enabled_route_gate_float64_hex": gate.hex(),
            "inactive_exact_zero_raw_and_preinstall_parity_verified": (
                schedule_index in INACTIVE_INDICES
            ),
            "active_finite_nonzero_gate_provenance_verified": (
                schedule_index in ACTIVE_INDICES
            ),
            **evidence,
            **{name: True for name in _TRUE_FLAGS},
            **{name: False for name in _FALSE_FLAGS},
            **_false_authority(),
        }
    )


def _semantic_metrics(
    *, schedule_index: int, fields: Mapping[str, torch.Tensor]
) -> Mapping[str, Any]:
    target = fields["source_noop_target_velocity"].double()

    def mse(name: str) -> float:
        result = float((fields[name].double() - target).square().mean().item())
        if not math.isfinite(result) or result < 0.0:
            raise Field14Exact40Error("field14 semantic metric is non-finite")
        return result

    losses = {
        "correct_atlas": mse("correct_atlas_noop_velocity"),
        "wrong_atlas": mse("wrong_atlas_noop_velocity"),
        "dropped_atlas": mse("dropped_atlas_noop_velocity"),
    }
    correct_delta = fields["correct_atlas_action_velocity"].double() - fields[
        "correct_atlas_noop_velocity"
    ].double()
    drop_delta = fields["dropped_atlas_action_velocity"].double() - fields[
        "dropped_atlas_noop_velocity"
    ].double()
    correct_norm = float(correct_delta.square().sum().sqrt().item())
    drop_norm = float(drop_delta.square().sum().sqrt().item())
    dot = float((correct_delta * drop_delta).sum().item())
    denominator = correct_norm * drop_norm
    cosine: Optional[float] = dot / denominator if denominator > 0.0 else None
    finite = all(math.isfinite(item) for item in (*losses.values(), correct_norm, drop_norm))
    if cosine is not None:
        finite = finite and math.isfinite(cosine)
    if not finite:
        raise Field14Exact40Error("field14 action semantic metric is non-finite")
    return seal_mapping(
        {
            "schema_version": METRICS_SCHEMA_VERSION,
            "schedule_index": schedule_index,
            "noop_fm_loss_float64_hex": {
                name: value.hex() for name, value in losses.items()
            },
            "action_delta_norm_float64_hex": {
                "correct_atlas": correct_norm.hex(),
                "dropped_atlas": drop_norm.hex(),
            },
            "action_delta_correct_drop_cosine_float64_hex": (
                cosine.hex() if cosine is not None else None
            ),
            "zero_norm_cosine_policy": "record_null_no_gate",
            "semantic_metrics_are_diagnostic_only": True,
            "semantic_metrics_used_for_selection": False,
            "semantic_metrics_used_for_cross_index_compensation": False,
            **_false_authority(),
        }
    )


def _validate_runtime_evidence(
    provenance: Mapping[str, Any],
    *,
    schedule_index: int,
    expected_gate: float,
) -> None:
    for name in (
        "confirmation_source_state_receipt_digest",
        "wrong_fit_source_state_receipt_digest",
        "epsilon_receipt_digest",
        "noisy_target_receipt_digest",
    ):
        _require_sha256(provenance.get(name), label=f"field14 {name}")
    if (
        provenance.get("expected_enabled_route_gate_float64_hex_recomputed")
        != expected_gate.hex()
        or provenance.get("ambient_torch_no_grad") is not True
    ):
        raise Field14Exact40Error("field14 live gate/no-grad evidence differs")
    coordinate = validate_sealed_mapping(
        provenance.get("coordinate"), label="field14 coordinate"
    )
    if (
        coordinate.get("schedule_index") != schedule_index
        or coordinate.get("timestep") != sigma_strata.PINNED_TIMESTEPS[schedule_index]
        or coordinate.get("sigma_float32_be_hex")
        != sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index]
        or coordinate.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256
        or coordinate.get("scheduler_step_called") is not False
    ):
        raise Field14Exact40Error("field14 coordinate evidence differs")
    before = provenance.get("same_state_identities_before_model_fields")
    after = provenance.get("same_state_identities_after_all_model_fields")
    atlas_before = provenance.get("atlas_identities_before_model_fields")
    atlas_after = provenance.get("atlas_identities_after_all_model_fields")
    expected_state_keys = (
        "confirmation_source_zs",
        "epsilon",
        "noisy_target_x_sigma",
        "native_visual_pack",
        "native_rotary_pack",
        "sigma",
        "timestep",
        "negative_condition",
        "noop_positive_condition",
        "action_positive_condition",
    )
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or tuple(before) != expected_state_keys
        or dict(before) != dict(after)
        or not isinstance(atlas_before, Mapping)
        or not isinstance(atlas_after, Mapping)
        or dict(atlas_before) != dict(atlas_after)
        or provenance.get("same_state_tensor_identities_recomputed_byte_equal")
        is not True
        or before["negative_condition"] == before["noop_positive_condition"]
        or before["negative_condition"] == before["action_positive_condition"]
        or before["noop_positive_condition"] == before["action_positive_condition"]
    ):
        raise Field14Exact40Error("field14 same-state runtime evidence differs")

    state_specs = {
        "confirmation_source_zs": (
            "torch.float32",
            "cuda",
            lambda dims: len(dims) == 5 and dims[0] == 1 and dims[1] == 16,
        ),
        "epsilon": (
            "torch.float32",
            "cuda",
            lambda dims: len(dims) == 5 and dims[0] == 1 and dims[1] == 16,
        ),
        "noisy_target_x_sigma": (
            "torch.float32",
            "cuda",
            lambda dims: len(dims) == 5 and dims[0] == 1 and dims[1] == 16,
        ),
        "native_visual_pack": (
            "torch.bfloat16",
            "cuda",
            lambda dims: len(dims) == 3 and dims[0] == 1 and dims[2] == 1536,
        ),
        "native_rotary_pack": (
            "torch.complex128",
            "cuda",
            lambda dims: len(dims) == 4 and dims[0:2] == [1, 1],
        ),
        "sigma": ("torch.float32", "cpu", lambda dims: dims == []),
        "timestep": ("torch.int64", "cuda", lambda dims: dims == [1]),
        "negative_condition": (
            "torch.bfloat16",
            "cuda",
            lambda dims: dims == [1, 512, 4096],
        ),
        "noop_positive_condition": (
            "torch.bfloat16",
            "cuda",
            lambda dims: dims == [1, 512, 4096],
        ),
        "action_positive_condition": (
            "torch.bfloat16",
            "cuda",
            lambda dims: dims == [1, 512, 4096],
        ),
    }
    for name, (dtype, device_type, shape) in state_specs.items():
        _validate_tensor_identity_record(
            before[name],
            label=f"field14 {name}",
            dtype=dtype,
            device_type=device_type,
            shape=shape,
        )
    if not (
        before["confirmation_source_zs"]["shape"]
        == before["epsilon"]["shape"]
        == before["noisy_target_x_sigma"]["shape"]
    ):
        raise Field14Exact40Error("field14 source/noise/x-sigma geometry differs")
    expected_atlas_keys = (
        "correct_confirmation_atlas",
        "wrong_same_family_fit_atlas",
    )
    if (
        tuple(atlas_before) != expected_atlas_keys
        or tuple(atlas_after) != expected_atlas_keys
    ):
        raise Field14Exact40Error("field14 atlas identity inventory differs")
    for name in expected_atlas_keys:
        _validate_tensor_identity_record(
            atlas_before[name],
            label=f"field14 {name}",
            dtype="torch.float32",
            device_type="cuda",
            shape=lambda dims: (
                len(dims) == 3
                and dims[0] == 1
                and dims[1] > 0
                and dims[2] == 1536
            ),
        )
    correct_atlas_identity = atlas_before["correct_confirmation_atlas"]
    wrong_atlas_identity = atlas_before["wrong_same_family_fit_atlas"]
    if (
        correct_atlas_identity["raw_sha256"]
        == wrong_atlas_identity["raw_sha256"]
        or correct_atlas_identity["content_sha256"]
        == wrong_atlas_identity["content_sha256"]
    ):
        raise Field14Exact40Error(
            "field14 correct/wrong atlas tensor bytes do not differ"
        )

    routes = provenance.get("route_receipts")
    raw = provenance.get("raw_tensor_identities")
    if (
        not isinstance(routes, Mapping)
        or tuple(routes) != RAW_ROLES
        or not isinstance(raw, Mapping)
        or tuple(raw) != RAW_ROLES
        or provenance.get("raw_call_order") != list(RAW_ROLES)
    ):
        raise Field14Exact40Error("field14 raw/route inventory differs")
    admitted_routes = {
        name: validate_sealed_mapping(routes[name], label=f"field14 {name} route")
        for name in RAW_ROLES
    }

    for name in RAW_ROLES:
        _validate_tensor_identity_record(
            raw[name],
            label=f"field14 {name} raw",
            dtype="torch.bfloat16",
            device_type="cuda",
            shape=lambda dims: bool(dims),
        )
    route_keys = {
        "branch_name",
        "total_tokens",
        "condition_tokens",
        "target_tokens",
        "sequence_parallel_rank",
        "sequence_parallel_size",
        "sigma_hex",
        "gate_hex",
        "atlas_receipt_digest",
        "source_memory_owned_by_V_VI_only",
        "enabled",
        "digest",
    }
    if any(
        set(admitted_routes[name]) != route_keys
        or type(admitted_routes[name].get("total_tokens")) is not int
        or admitted_routes[name]["total_tokens"] <= 0
        or type(admitted_routes[name].get("condition_tokens")) is not int
        or not 0
        <= admitted_routes[name]["condition_tokens"]
        < admitted_routes[name]["total_tokens"]
        or admitted_routes[name].get("target_tokens")
        != admitted_routes[name]["total_tokens"]
        - admitted_routes[name]["condition_tokens"]
        or admitted_routes[name].get("sequence_parallel_size") != 4
        or admitted_routes[name].get("sequence_parallel_rank") not in range(4)
        or admitted_routes[name].get("sigma_hex")
        != sigma_strata.PINNED_POSITIVE_SIGMAS[schedule_index].hex()
        or admitted_routes[name].get("source_memory_owned_by_V_VI_only") is not True
        for name in RAW_ROLES
    ):
        raise Field14Exact40Error("field14 route receipt schema differs")
    if any(
        not isinstance(raw[name], Mapping)
        or type(raw[name].get("content_sha256")) is not str
        or len(raw[name]["content_sha256"]) != 64
        for name in RAW_ROLES
    ):
        raise Field14Exact40Error("field14 raw tensor identity differs")
    enabled = RAW_ROLES[:5]
    dropped = RAW_ROLES[5:]
    if any(
        admitted_routes[name].get("branch_name") != "V"
        or admitted_routes[name].get("enabled") is not True
        or admitted_routes[name].get("gate_hex") != expected_gate.hex()
        or admitted_routes[name].get("atlas_receipt_digest") is None
        for name in enabled
    ) or any(
        admitted_routes[name].get("branch_name") != "V"
        or admitted_routes[name].get("enabled") is not False
        or admitted_routes[name].get("gate_hex") != 0.0.hex()
        or admitted_routes[name].get("atlas_receipt_digest") is not None
        for name in dropped
    ):
        raise Field14Exact40Error("field14 route receipt gate differs")
    if not (
        admitted_routes["correct_negative"]
        == admitted_routes["correct_noop"]
        == admitted_routes["correct_action"]
        and admitted_routes["wrong_negative"] == admitted_routes["wrong_noop"]
        and admitted_routes["drop_negative"]
        == admitted_routes["drop_noop"]
        == admitted_routes["drop_action"]
    ):
        raise Field14Exact40Error("field14 same-route condition evidence differs")

    def without_atlas(value: Mapping[str, Any]) -> dict[str, Any]:
        plain = dict(value)
        plain.pop("digest")
        plain.pop("atlas_receipt_digest")
        return plain

    correct_atlas = admitted_routes["correct_negative"]["atlas_receipt_digest"]
    wrong_atlas = admitted_routes["wrong_negative"]["atlas_receipt_digest"]
    correct_without_adapter = without_atlas(admitted_routes["correct_negative"])
    drop_without_adapter = without_atlas(admitted_routes["drop_negative"])
    for key in ("enabled", "gate_hex"):
        correct_without_adapter.pop(key)
        drop_without_adapter.pop(key)
    if (
        any(
            admitted_routes[name]["atlas_receipt_digest"] != correct_atlas
            for name in ("correct_noop", "correct_action")
        )
        or admitted_routes["wrong_noop"]["atlas_receipt_digest"] != wrong_atlas
        or correct_atlas == wrong_atlas
        or without_atlas(admitted_routes["correct_negative"])
        != without_atlas(admitted_routes["wrong_negative"])
        or correct_without_adapter != drop_without_adapter
        or provenance.get("wrong_route_receipts_differ_only_in_atlas_memory")
        is not True
        or provenance.get(
            "drop_route_receipts_retain_v_branch_disable_only_rebinder"
        )
        is not True
        or provenance.get(
            "action_noop_route_receipts_equal_with_negative_raw_reuse"
        )
        is not True
    ):
        raise Field14Exact40Error("field14 route-only intervention evidence differs")

    inactive = provenance.get("inactive_raw_parity")
    if schedule_index in INACTIVE_INDICES:
        inactive = validate_sealed_mapping(
            inactive, label="field14 inactive raw parity"
        )
        if (
            inactive.get("schedule_index") != schedule_index
            or inactive.get("route_gate_float64_hex") != 0.0.hex()
            or any(
                inactive.get(name) is not True
                for name in (
                    "correct_wrong_drop_negative_raw_byte_exact",
                    "correct_wrong_drop_noop_raw_byte_exact",
                    "correct_drop_action_raw_byte_exact",
                    "all_same_condition_raw_equal_preinstall",
                    "source_noise_xsigma_vpack_rotary_timestep_conditions_equal_preinstall",
                )
            )
        ):
            raise Field14Exact40Error("field14 inactive parity evidence differs")
        preinstall = inactive.get("preinstall_row_sha256")
        if (
            not isinstance(preinstall, Mapping)
            or tuple(preinstall)
            != ("negative", "noop_positive", "action_positive")
            or not (
                raw["correct_negative"]
                == raw["wrong_negative"]
                == raw["drop_negative"]
            )
            or not (
                raw["correct_noop"] == raw["wrong_noop"] == raw["drop_noop"]
            )
            or raw["correct_action"] != raw["drop_action"]
            or raw["correct_negative"]["content_sha256"] != preinstall["negative"]
            or raw["correct_noop"]["content_sha256"] != preinstall["noop_positive"]
            or raw["correct_action"]["content_sha256"] != preinstall["action_positive"]
        ):
            raise Field14Exact40Error("field14 inactive raw byte identities differ")
    elif inactive is not None:
        raise Field14Exact40Error("field14 active coordinate acquired inactive parity")


def admit_field14_tensor_set(
    value: Any,
    *,
    schedule_index: int,
    family: str,
    confirmation_iid: str,
    confirmation_source_sha256: str,
    wrong_owner_iid: str,
    wrong_owner_source_sha256: str,
) -> Mapping[str, Any]:
    if family not in short_runner.FAMILY_BY_DP_ARM:
        raise Field14Exact40Error("field14 family differs")
    arm = short_runner.FAMILY_BY_DP_ARM.index(family)
    if (
        confirmation_iid != short_runner.CONFIRMATION_IID_BY_DP_ARM[arm]
        or wrong_owner_iid != short_runner.FIT_IID_BY_DP_ARM[arm]
        or confirmation_iid == wrong_owner_iid
        or confirmation_source_sha256 == wrong_owner_source_sha256
    ):
        raise Field14Exact40Error("field14 confirmation/wrong-owner binding differs")
    _require_sha256(
        confirmation_source_sha256, label="field14 confirmation source SHA256"
    )
    _require_sha256(
        wrong_owner_source_sha256, label="field14 wrong-owner source SHA256"
    )
    if type(value) is not Field14TensorSet:
        raise Field14Exact40Error("field14 producer returned a non-tensor-set")
    rows = _tensor_rows(value)
    shape: Optional[tuple[int, ...]] = None
    device: Optional[torch.device] = None
    pointers = []
    fields: dict[str, torch.Tensor] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    for name, tensor in rows:
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.float32
            or tensor.device.type == "meta"
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not tensor.is_contiguous()
            or tensor.numel() <= 0
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise Field14Exact40Error(f"field14 tensor contract differs: {name}")
        tensor_shape = tuple(int(item) for item in tensor.shape)
        if shape is None:
            shape, device = tensor_shape, tensor.device
        elif tensor_shape != shape or tensor.device != device:
            raise Field14Exact40Error("field14 tensor geometry differs")
        pointers.append(
            (tensor.device.type, tensor.device.index, int(tensor.untyped_storage().data_ptr()))
        )
        fields[name] = tensor
        identities[name] = short_runner.tensor_identity(tensor)
    if len(set(pointers)) != len(FIELD_ROLES):
        raise Field14Exact40Error("field14 tensor storages alias")

    provenance = validate_sealed_mapping(value.provenance, label="field14 provenance")
    gate = expected_route_gate(schedule_index)
    expected_provenance_keys = (
        {
            "schema_version",
            "schedule_index",
            "family",
            "confirmation_iid",
            "confirmation_source_sha256",
            "wrong_owner_iid",
            "wrong_owner_source_sha256",
            "field_roles",
            "field_tensor_identities",
            "route_regime",
            "expected_enabled_route_gate_float64_hex",
            "inactive_exact_zero_raw_and_preinstall_parity_verified",
            "active_finite_nonzero_gate_provenance_verified",
            "digest",
        }
        | _RUNTIME_EVIDENCE_KEYS
        | set(_TRUE_FLAGS)
        | set(_FALSE_FLAGS)
        | set(AUTHORITY_FIELDS)
    )
    if set(provenance) != expected_provenance_keys:
        raise Field14Exact40Error("field14 provenance key schema differs")
    expected = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "schedule_index": schedule_index,
        "family": family,
        "confirmation_iid": confirmation_iid,
        "confirmation_source_sha256": confirmation_source_sha256,
        "wrong_owner_iid": wrong_owner_iid,
        "wrong_owner_source_sha256": wrong_owner_source_sha256,
        "field_roles": list(FIELD_ROLES),
        "field_tensor_identities": identities,
        "route_regime": (
            "inactive_sigma_ge_0.75_exact_zero"
            if schedule_index in INACTIVE_INDICES
            else "active_sigma_lt_0.75_finite_nonzero"
        ),
        "expected_enabled_route_gate_float64_hex": gate.hex(),
        "inactive_exact_zero_raw_and_preinstall_parity_verified": (
            schedule_index in INACTIVE_INDICES
        ),
        "active_finite_nonzero_gate_provenance_verified": (
            schedule_index in ACTIVE_INDICES
        ),
    }
    if any(provenance.get(name) != wanted for name, wanted in expected.items()):
        raise Field14Exact40Error("field14 provenance identity/regime differs")
    if any(provenance.get(name) is not True for name in _TRUE_FLAGS):
        raise Field14Exact40Error("field14 same-state provenance is incomplete")
    if any(provenance.get(name) is not False for name in _FALSE_FLAGS):
        raise Field14Exact40Error("field14 provenance crossed a denied boundary")
    if any(provenance.get(name) is not False for name in AUTHORITY_FIELDS):
        raise Field14Exact40Error("field14 provenance elevated authority")
    _validate_runtime_evidence(
        provenance, schedule_index=schedule_index, expected_gate=gate
    )
    _walk_no_authority(provenance)
    metrics = _semantic_metrics(schedule_index=schedule_index, fields=fields)
    return seal_mapping(
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "schedule_index": schedule_index,
            "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                schedule_index
            ],
            "timestep": sigma_strata.PINNED_TIMESTEPS[schedule_index],
            "field_roles": list(FIELD_ROLES),
            "field_shape": list(shape or ()),
            "field_dtype": "torch.float32",
            "field_device_type": str(device.type if device is not None else ""),
            "field_tensor_identities": identities,
            "provenance": provenance,
            "semantic_metrics": dict(metrics),
            "per_index_canonical_hash_before_release": True,
            "tensors_embedded_in_receipt": False,
            "checkpoint_payload_returned": False,
            **_false_authority(),
        }
    )


def execute_exact40_sweep(
    *,
    family: str,
    confirmation_iid: str,
    confirmation_source_sha256: str,
    wrong_owner_iid: str,
    wrong_owner_source_sha256: str,
    short_result_digest: str,
    preinstall_baseline_digest: str,
    measure_index: Callable[[int], Field14TensorSet],
    release_index: Callable[[int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Measure, admit, hash, and release exactly one coordinate at a time."""

    if torch.is_grad_enabled():
        raise Field14Exact40Error("field14 sweep must enter under torch.no_grad")
    _require_sha256(short_result_digest, label="field14 short-result digest")
    _require_sha256(
        preinstall_baseline_digest, label="field14 preinstall-baseline digest"
    )
    if not callable(measure_index) or not callable(release_index):
        raise Field14Exact40Error("field14 callbacks differ")
    rows = []
    for expected_index in EXACT40_INDICES:
        packet = measure_index(expected_index)
        if type(packet) is not Field14TensorSet:
            raise Field14Exact40Error("field14 measurement type differs")
        tensor_references = [weakref.ref(tensor) for _, tensor in _tensor_rows(packet)]
        admission = admit_field14_tensor_set(
            packet,
            schedule_index=expected_index,
            family=family,
            confirmation_iid=confirmation_iid,
            confirmation_source_sha256=confirmation_source_sha256,
            wrong_owner_iid=wrong_owner_iid,
            wrong_owner_source_sha256=wrong_owner_source_sha256,
        )
        field_hashes = {
            name: admission["field_tensor_identities"][name]["content_sha256"]
            for name in FIELD_ROLES
        }
        del packet
        gc.collect()
        if any(reference() is not None for reference in tensor_references):
            raise Field14Exact40Error("field14 producer retained a released tensor")
        release = validate_sealed_mapping(
            release_index(expected_index), label="field14 tensor release"
        )
        if (
            release.get("schema_version") != RELEASE_SCHEMA_VERSION
            or release.get("schedule_index") != expected_index
            or release.get("all_field_tensor_objects_released") is not True
            or release.get("allocator_cache_release_requested") is not True
            or any(release.get(name) is not False for name in AUTHORITY_FIELDS)
        ):
            raise Field14Exact40Error("field14 tensor release receipt differs")
        rows.append(
            {
                "schedule_index": expected_index,
                "admission_digest": admission["digest"],
                "field_tensor_sha256": field_hashes,
                "semantic_metrics_digest": admission["semantic_metrics"]["digest"],
                "provenance_digest": admission["provenance"]["digest"],
                "release_digest": release["digest"],
                "all_field_tensor_objects_released": True,
            }
        )
    if [row["schedule_index"] for row in rows] != list(EXACT40_INDICES):
        raise Field14Exact40Error("field14 exact40 order/coverage differs")
    result = seal_mapping(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_in_memory_exact40_no_grad_no_checkpoint",
            "family": family,
            "confirmation_iid": confirmation_iid,
            "confirmation_source_sha256": confirmation_source_sha256,
            "wrong_owner_iid": wrong_owner_iid,
            "wrong_owner_source_sha256": wrong_owner_source_sha256,
            "short_result_digest": short_result_digest,
            "preinstall_baseline_digest": preinstall_baseline_digest,
            "schedule_indices": list(EXACT40_INDICES),
            "inactive_indices": list(INACTIVE_INDICES),
            "active_indices": list(ACTIVE_INDICES),
            "field_roles": list(FIELD_ROLES),
            "rows": rows,
            "exact40_official_order": True,
            "ambient_torch_no_grad": True,
            "one_index_admitted_hashed_and_released_before_next": True,
            "cross_index_tensor_retention": False,
            "cross_index_compensation_used": False,
            "cross_index_selection_used": False,
            "semantic_metrics_are_diagnostic_only": True,
            "checkpoint_written": False,
            "checkpoint_payload_returned": False,
            "publication_performed": False,
            **_false_authority(),
        }
    )
    _walk_no_authority(result)
    return result


def build_release_receipt(schedule_index: int, *, cuda_cache_requested: bool) -> Mapping[str, Any]:
    if type(cuda_cache_requested) is not bool or not cuda_cache_requested:
        raise Field14Exact40Error("field14 allocator release was not requested")
    expected_route_gate(schedule_index)
    return seal_mapping(
        {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "schedule_index": schedule_index,
            "all_field_tensor_objects_released": True,
            "allocator_cache_release_requested": True,
            "no_tensor_payload_retained": True,
            "checkpoint_payload_returned": False,
            **_false_authority(),
        }
    )


__all__ = [
    "ACTIVE_INDICES",
    "AUTHORITY_FIELDS",
    "EXACT40_INDICES",
    "FIELD_ROLES",
    "Field14Exact40Error",
    "Field14TensorSet",
    "INACTIVE_INDICES",
    "PROVENANCE_SCHEMA_VERSION",
    "RAW_ROLES",
    "RELEASE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "admit_field14_tensor_set",
    "build_field14_provenance",
    "build_release_receipt",
    "canonical_json_bytes",
    "execute_exact40_sweep",
    "expected_route_gate",
    "object_sha256",
    "seal_mapping",
    "validate_sealed_mapping",
]
