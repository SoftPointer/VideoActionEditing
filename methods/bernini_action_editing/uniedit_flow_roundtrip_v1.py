#!/usr/bin/env python3
"""Model-free contract for the UniEdit-Flow A0 roundtrip control.

This module implements only Algorithm 1 (Uni-Inv with Euler) from
UniEdit-Flow and its vanilla explicit-Euler inversion comparator.  It owns no
model, optimizer, adapter, random state, media I/O, or distributed state.

The Bernini experiment deliberately reuses the already captured exact40
``UniPCMultistepScheduler`` grid as a *coordinate registry*.  The solvers in
this file do not instantiate UniPC and do not apply ``flow_shift`` again.  A
matched descending Euler reconstruction is therefore a clean test of the
inversion rule, not a comparison confounded by different reverse solvers.

Primary authorities:

* https://openreview.net/forum?id=ArU2CeB7Tm
* https://arxiv.org/html/2504.13109v2#S4.SS1 (Algorithm 1)
* https://github.com/DSL-Lab/UniEdit-Flow
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from typing import Any, Callable, Mapping, NoReturn, Optional, Sequence

import inference_sigma_strata as exact40


class UniEditFlowRoundtripError(RuntimeError):
    """Raised before an ambiguous A0 recurrence or receipt is accepted."""


def fail(message: str) -> NoReturn:
    raise UniEditFlowRoundtripError(message)


CONTRACT_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-contract-v1"
RECEIPT_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-receipt-v1"
TRACE_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-trace-v1"
GATE_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-hard-gate-v1"
METRIC_PACKET_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-metrics-v1"
SOURCE_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-source-v1"
MODEL_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-model-v1"
PARALLEL_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-parallel-v1"
PROMPT_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-prompt-v1"
SOLVER_CONTRACT_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-solver-v1"
DEPENDENCY_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-dependencies-v1"
RUNTIME_VERSIONS_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-versions-v1"
BACKEND_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-metric-backend-v1"
MEDIA_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-media-v1"
ARM_SCHEMA = "bernini-uniedit-flow-roundtrip-a0-arm-v1"

PAPER_URL = "https://openreview.net/forum?id=ArU2CeB7Tm"
ARXIV_ALGORITHM_URL = "https://arxiv.org/html/2504.13109v2#S4.SS1"
OFFICIAL_REPOSITORY_URL = "https://github.com/DSL-Lab/UniEdit-Flow"
OFFICIAL_REPOSITORY_COMMIT = "dc9edb465545352bbd9d674010ac8683e554c97d"
OFFICIAL_UNIINV_SCHEDULER_GIT_BLOB = "dde8c59d811a2064a6b07a5d21457f4aef636a3e"

IID = "00435ad621c44fac"
SOURCE_VIDEO_SHA256 = "b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1"
SOURCE_DATASET_SPEC_SHA256 = "62468b24d4a57ec03d42ce8c006a707cbcf56588ef62d10632089eb5ad457920"
SOURCE_DATASET_RECEIPT_SHA256 = "6ed77cf7d98391c2074e5938ab50d0688d457bddfd688f9a5825d455447a20bb"
SOURCE_DATASET_RECEIPT_DIGEST = "12ede44ebab03215e19574967a9afec3c634f246f2cfd2634a48ce0e3dea8738"
ORBIT_DATASET_SPEC_SHA256 = "72c0f104b123a1b7ad69f32697a0b7f7e8c2fdf766c951f3c0bed7518f0f564f"
ORBIT_DATASET_RECEIPT_SHA256 = "c088eb0128c3c807941f60eb3e763d0e71f4c8dbb190c60b9c0dad6caeca0230"
ORBIT_DATASET_RECEIPT_DIGEST = "9000dd9dace16501587196ac8459b620529301508ee6c98662f266b3b29b8982"
ORBIT_ROW_DIGEST = "e6b48ee59d7816a03a0808d5826ddd0f405bb099b3dde12dc72ec545c90a8529"
PINNED_VAE_IDENTITY_DIGEST = "43ba8e152bc13a32538eec9f7859bc6858562c0a6cd0056a4d5d7522c3ca5784"

EXPECTED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
EXPECTED_CHECKPOINT_TREE_SHA256 = "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
EXPECTED_MODEL_STATE_SHA256 = "1c4cc090e657c926d6a4445caf386928c72cdddd5a0151802aac9f13707cbae2"
EXPECTED_VAE_FILE_SHA256 = {
    "vae/config.json": "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527",
    "vae/diffusion_pytorch_model.safetensors": "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",
}

EXPECTED_DEPENDENCY_SHA256 = {
    "inference_sigma_strata.py": "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
    "dclr_runtime_contract.py": "2c4416742372f85a6307d3d17ee31cdf8a8677cc9884f67fcd3c79064d838cfc",
    "pair_v5_native_bridge.py": "a441afd4b2df516d9332d02c928ac60eb19dcc80c356693e85f12d00080a4dd6",
}

EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.8.11",
    "torch": "2.7.1+rocm6.3",
    "torch_hip": "6.3.42131-fa1d09cbd",
    "diffusers": "0.38.0",
    "transformers": "5.5.4",
}

EXPECTED_EXACT40_SCHEDULE_SHA256 = (
    "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
)
FRAME_COUNT = 81
FPS = 25
HEIGHT = 592
WIDTH = 400
LATENT_SHAPE = (1, 16, 21, 74, 50)
SP_SIZE = 4
WORLD_SIZE = 4
ARMS = (
    "c0_vae_ceiling",
    "e0_vanilla_euler_roundtrip",
    "u0_uni_inv_roundtrip",
)
SOLVER_ARMS = ARMS[1:]
ARM_MODEL_FORWARD_CALLS = {
    # C0 is the already-encoded source latent decoded by the frozen VAE.  It
    # establishes the codec ceiling and must never query the transformer.
    "c0_vae_ceiling": 0,
    # Vanilla Euler: forty ascending inversion queries followed by the same
    # forty descending reconstruction queries.
    "e0_vanilla_euler_roundtrip": 80,
    # Uni-Inv additionally evaluates the initial sigma-zero velocity and the
    # corrected velocity at every one of the forty positive coordinates.
    "u0_uni_inv_roundtrip": 81,
}
TOTAL_MODEL_FORWARD_CALLS = sum(ARM_MODEL_FORWARD_CALLS.values())
TEMPORAL_QUINTILE_BOUNDS = (
    (0, 16),
    (16, 32),
    (32, 48),
    (48, 64),
    (64, 81),
)

_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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
        raise UniEditFlowRoundtripError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def _exact_mapping(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        fail(f"{label} keys differ")
    return value


def _validated_record(
    value: Any, keys_without_digest: set[str], *, label: str
) -> Mapping[str, Any]:
    record = _exact_mapping(
        value, keys_without_digest | {"digest"}, label=label
    )
    declared = require_sha256(record.get("digest"), label=f"{label} digest")
    unsigned = dict(record)
    unsigned.pop("digest")
    if object_sha256(unsigned) != declared:
        fail(f"{label} digest mismatch")
    return record


def finalize_evidence_record(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    """Attach one canonical digest to a nested runtime evidence record."""

    if not isinstance(unsigned, Mapping) or "digest" in unsigned:
        fail("unsigned evidence record differs")
    value = dict(unsigned)
    value["digest"] = object_sha256(value)
    return value


def _float32(value: Any, *, label: str) -> float:
    try:
        number = float(value)
        result = struct.unpack(">f", struct.pack(">f", number))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise UniEditFlowRoundtripError(f"{label} must be finite float32") from error
    if not math.isfinite(number) or not math.isfinite(result):
        fail(f"{label} must be finite float32")
    return float(result)


def _float32_hex(value: Any, *, label: str) -> str:
    number = _float32(value, label=label)
    return struct.pack(">f", number).hex()


def _float32_from_hex(value: str, *, label: str) -> float:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{8}", value) is None:
        fail(f"{label} must be eight lowercase float32 hex characters")
    return float(struct.unpack(">f", bytes.fromhex(value))[0])


def _float32_sub(left: float, right: float) -> float:
    return _float32(float(left) - float(right), label="schedule delta")


@dataclass(frozen=True)
class RoundtripCoordinate:
    """One low-to-high coordinate of the exact Bernini grid."""

    ascending_index: int
    official_denoising_index: Optional[int]
    model_timestep_int64: int
    sigma: float
    sigma_float32_be_hex: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ascending_index": self.ascending_index,
            "official_denoising_index": self.official_denoising_index,
            "model_timestep_int64": self.model_timestep_int64,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
        }


@dataclass(frozen=True)
class RoundtripSchedule:
    coordinates_ascending: tuple[RoundtripCoordinate, ...]

    @property
    def interval_count(self) -> int:
        return len(self.coordinates_ascending) - 1

    def positive_delta(self, interval_index: int) -> float:
        if type(interval_index) is not int or not 1 <= interval_index <= self.interval_count:
            fail("ascending interval index lies outside exact40")
        left = self.coordinates_ascending[interval_index - 1].sigma
        right = self.coordinates_ascending[interval_index].sigma
        delta = _float32_sub(right, left)
        if not delta > 0.0:
            fail("exact40 ascending delta is not positive")
        return delta

    def negative_delta(self, descending_index: int) -> float:
        if type(descending_index) is not int or not 1 <= descending_index <= self.interval_count:
            fail("descending interval index lies outside exact40")
        high = self.coordinates_ascending[descending_index].sigma
        low = self.coordinates_ascending[descending_index - 1].sigma
        delta = _float32_sub(low, high)
        if not delta < 0.0:
            fail("exact40 descending delta is not negative")
        return delta

    def receipt(self) -> dict[str, Any]:
        value = {
            "schema_version": CONTRACT_SCHEMA,
            "coordinate_source": "captured_bernini_unipc_exact40_registry",
            "source_schedule_sha256": exact40.SCHEDULE_SHA256,
            "num_intervals": self.interval_count,
            "coordinates_ascending": [
                coordinate.as_dict() for coordinate in self.coordinates_ascending
            ],
            "positive_delta_float32_be_hex": [
                _float32_hex(self.positive_delta(index), label="positive delta")
                for index in range(1, self.interval_count + 1)
            ],
            "negative_delta_float32_be_hex": [
                _float32_hex(self.negative_delta(index), label="negative delta")
                for index in range(self.interval_count, 0, -1)
            ],
            "flow_shift_declared_by_source_schedule": exact40.FLOW_SHIFT,
            "flow_shift_application_count_in_a0_runtime": 0,
            "double_shift_forbidden": True,
            "scheduler_object_instantiated_by_a0_solver": False,
            "model_timestep_policy": "pinned_int64_not_recomputed_from_sigma",
        }
        return {**value, "digest": object_sha256(value)}


def build_exact40_roundtrip_schedule() -> RoundtripSchedule:
    if exact40.SCHEDULE_SHA256 != EXPECTED_EXACT40_SCHEDULE_SHA256:
        fail("imported exact40 schedule SHA differs from the A0 pin")
    if (
        len(exact40.PINNED_TIMESTEPS) != 40
        or len(exact40.PINNED_POSITIVE_SIGMAS) != 40
        or len(exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX) != 40
    ):
        fail("imported exact40 schedule cardinality differs")
    coordinates: list[RoundtripCoordinate] = [
        RoundtripCoordinate(0, None, 0, 0.0, exact40.TERMINAL_SIGMA_FLOAT32_HEX)
    ]
    reversed_rows = zip(
        reversed(range(40)),
        reversed(exact40.PINNED_TIMESTEPS),
        reversed(exact40.PINNED_POSITIVE_SIGMAS),
        reversed(exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX),
    )
    for ascending_index, (denoising_index, timestep, sigma, sigma_hex) in enumerate(
        reversed_rows, start=1
    ):
        if _float32_hex(sigma, label="pinned sigma") != sigma_hex:
            fail("imported exact40 sigma decimal/byte representations differ")
        coordinates.append(
            RoundtripCoordinate(
                ascending_index=ascending_index,
                official_denoising_index=denoising_index,
                model_timestep_int64=int(timestep),
                sigma=float(sigma),
                sigma_float32_be_hex=str(sigma_hex),
            )
        )
    schedule = RoundtripSchedule(tuple(coordinates))
    if schedule.interval_count != 40:
        fail("A0 requires exactly forty exact40 intervals")
    if any(
        left.sigma >= right.sigma
        for left, right in zip(
            schedule.coordinates_ascending,
            schedule.coordinates_ascending[1:],
        )
    ):
        fail("A0 exact40 coordinates must be strictly ascending")
    # Exercise every interval and its float32 sign guard at import time.
    for index in range(1, 41):
        schedule.positive_delta(index)
        schedule.negative_delta(index)
    return schedule


EXACT40_ROUNDTRIP_SCHEDULE = build_exact40_roundtrip_schedule()


def solver_contract_receipt() -> dict[str, Any]:
    """Return the only solver contract that may authorize the A0 receipt."""

    unsigned = {
        "schema_version": SOLVER_CONTRACT_SCHEMA,
        "schedule_digest": EXACT40_ROUNDTRIP_SCHEDULE.receipt()["digest"],
        "coordinate_policy": "bernini_native_exact40_registry_plus_sigma_zero",
        "e0_inverse_recurrence": (
            "z_i=z_{i-1}+delta_sigma_i*v(z_{i-1},sigma_{i-1},blank)"
        ),
        "u0_predictor_recurrence": (
            "z_hat_i=z_{i-1}+delta_sigma_i*v_{i-1}"
        ),
        "u0_corrector_recurrence": (
            "z_i=z_{i-1}+delta_sigma_i*v(z_hat_i,sigma_i,blank)"
        ),
        "matched_reconstruction_recurrence": (
            "z_{i-1}=z_i-delta_sigma_i*v(z_i,sigma_i,blank)"
        ),
        "same_descending_reconstruction_for_e0_u0": True,
        "sigma_zero_query_required_for_u0": True,
        "model_timestep_policy": "pinned_int64_registry_value_passed_as_fp32",
        "state_update_dtype": "torch.float32",
        "model_forward_autocast_dtype": "torch.bfloat16",
        "scheduler_object_instantiated": False,
        "scheduler_step_calls": 0,
        "flow_shift_reapplied": False,
        "model_forward_calls_by_arm": dict(ARM_MODEL_FORWARD_CALLS),
        "total_model_forward_calls": TOTAL_MODEL_FORWARD_CALLS,
    }
    return finalize_evidence_record(unsigned)


def dependency_receipt(method_source_archive_sha256: str) -> dict[str, Any]:
    archive_sha = require_sha256(
        method_source_archive_sha256, label="dependency method source archive"
    )
    unsigned = {
        "schema_version": DEPENDENCY_SCHEMA,
        "method_source_archive_sha256": archive_sha,
        "local_source_sha256": dict(EXPECTED_DEPENDENCY_SHA256),
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "official_uniinv_scheduler_git_blob": OFFICIAL_UNIINV_SCHEDULER_GIT_BLOB,
        "external_method_code_executed": False,
        "native_bernini_lower_level_t2v_path_required": True,
    }
    return finalize_evidence_record(unsigned)


def runtime_versions_receipt() -> dict[str, Any]:
    return finalize_evidence_record(
        {
            "schema_version": RUNTIME_VERSIONS_SCHEMA,
            **EXPECTED_RUNTIME_VERSIONS,
        }
    )


@dataclass(frozen=True)
class VelocityQuery:
    arm: str
    phase: str
    query_index: int
    coordinate: RoundtripCoordinate

    def __post_init__(self) -> None:
        if self.arm not in SOLVER_ARMS:
            fail("velocity query arm differs from E0/U0")
        if self.phase not in {"inverse", "reconstruct"}:
            fail("velocity query phase differs")
        if type(self.query_index) is not int or self.query_index < 0:
            fail("velocity query index must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "phase": self.phase,
            "query_index": self.query_index,
            "coordinate": self.coordinate.as_dict(),
        }


@dataclass(frozen=True)
class Trajectory:
    arm: str
    phase: str
    initial_state: Any
    final_state: Any
    records: tuple[Mapping[str, Any], ...]
    model_forward_calls: int

    def receipt(self, *, digest_state: Callable[[Any], str]) -> dict[str, Any]:
        value = {
            "schema_version": TRACE_SCHEMA,
            "arm": self.arm,
            "phase": self.phase,
            "initial_state_sha256": require_sha256(
                digest_state(self.initial_state), label="trajectory initial state"
            ),
            "final_state_sha256": require_sha256(
                digest_state(self.final_state), label="trajectory final state"
            ),
            "model_forward_calls": self.model_forward_calls,
            "records": [dict(item) for item in self.records],
        }
        return {**value, "digest": object_sha256(value)}


VelocityFunction = Callable[[Any, VelocityQuery], Any]
AdvanceFunction = Callable[[Any, Any, float], Any]
DigestFunction = Callable[[Any], str]


def _default_advance(state: Any, velocity: Any, delta: float) -> Any:
    """Small model-free AXPY used by scalar/list/tuple hostile tests."""

    if isinstance(state, bool) or isinstance(velocity, bool):
        fail("default AXPY forbids booleans")
    if isinstance(state, (int, float)) and isinstance(velocity, (int, float)):
        result = float(state) + float(delta) * float(velocity)
        if not math.isfinite(result):
            fail("default AXPY produced a non-finite scalar")
        return result
    if type(state) is tuple and type(velocity) is tuple and len(state) == len(velocity):
        return tuple(
            _default_advance(left, right, delta)
            for left, right in zip(state, velocity)
        )
    if type(state) is list and type(velocity) is list and len(state) == len(velocity):
        return [
            _default_advance(left, right, delta)
            for left, right in zip(state, velocity)
        ]
    fail("default AXPY requires matching finite scalars, tuples, or lists")


def _default_digest(value: Any) -> str:
    return object_sha256(value)


def _guarded_velocity(
    state: Any,
    query: VelocityQuery,
    *,
    velocity_fn: VelocityFunction,
    digest_state: DigestFunction,
) -> tuple[Any, str, str]:
    before = require_sha256(digest_state(state), label="pre-forward state digest")
    try:
        velocity = velocity_fn(state, query)
    except UniEditFlowRoundtripError:
        raise
    except Exception as error:
        raise UniEditFlowRoundtripError(
            f"velocity callback failed for {query.arm}/{query.phase}/{query.query_index}: {error}"
        ) from error
    after = require_sha256(digest_state(state), label="post-forward state digest")
    if before != after:
        fail("velocity callback mutated the recurrence state")
    velocity_digest = require_sha256(
        digest_state(velocity), label="velocity state digest"
    )
    return velocity, before, velocity_digest


def _guarded_advance(
    state: Any,
    velocity: Any,
    delta: float,
    *,
    advance: AdvanceFunction,
    digest_state: DigestFunction,
) -> tuple[Any, str, str]:
    before = require_sha256(digest_state(state), label="pre-AXPY state digest")
    velocity_before = require_sha256(
        digest_state(velocity), label="pre-AXPY velocity digest"
    )
    try:
        result = advance(state, velocity, delta)
    except UniEditFlowRoundtripError:
        raise
    except Exception as error:
        raise UniEditFlowRoundtripError(f"AXPY callback failed: {error}") from error
    if require_sha256(digest_state(state), label="post-AXPY state digest") != before:
        fail("AXPY callback mutated its state input")
    if (
        require_sha256(digest_state(velocity), label="post-AXPY velocity digest")
        != velocity_before
    ):
        fail("AXPY callback mutated its velocity input")
    result_digest = require_sha256(digest_state(result), label="AXPY result digest")
    return result, before, result_digest


def vanilla_inverse_euler(
    initial_state: Any,
    velocity_fn: VelocityFunction,
    *,
    schedule: RoundtripSchedule = EXACT40_ROUNDTRIP_SCHEDULE,
    advance: AdvanceFunction = _default_advance,
    digest_state: DigestFunction = _default_digest,
) -> Trajectory:
    """Explicit Euler inversion using ``v(x_{i-1}, t_{i-1})``.

    This is the paper's vanilla/DDIM-style comparator.  It is intentionally
    not the alternative ``v(x_{i-1}, t_i)`` ablation.
    """

    current = initial_state
    records: list[dict[str, Any]] = []
    for interval_index in range(1, schedule.interval_count + 1):
        query = VelocityQuery(
            arm="e0_vanilla_euler_roundtrip",
            phase="inverse",
            query_index=interval_index - 1,
            coordinate=schedule.coordinates_ascending[interval_index - 1],
        )
        velocity, state_sha, velocity_sha = _guarded_velocity(
            current,
            query,
            velocity_fn=velocity_fn,
            digest_state=digest_state,
        )
        delta = schedule.positive_delta(interval_index)
        next_state, _, next_sha = _guarded_advance(
            current,
            velocity,
            delta,
            advance=advance,
            digest_state=digest_state,
        )
        records.append(
            {
                "kind": "explicit_euler_inverse",
                "interval_index_ascending": interval_index,
                "query": query.as_dict(),
                "delta_sigma_float32_be_hex": _float32_hex(
                    delta, label="E0 inverse delta"
                ),
                "state_before_sha256": state_sha,
                "velocity_sha256": velocity_sha,
                "state_after_sha256": next_sha,
            }
        )
        current = next_state
    return Trajectory(
        arm="e0_vanilla_euler_roundtrip",
        phase="inverse",
        initial_state=initial_state,
        final_state=current,
        records=tuple(records),
        model_forward_calls=schedule.interval_count,
    )


def uni_inv_predictor_corrector(
    initial_state: Any,
    velocity_fn: VelocityFunction,
    *,
    schedule: RoundtripSchedule = EXACT40_ROUNDTRIP_SCHEDULE,
    advance: AdvanceFunction = _default_advance,
    digest_state: DigestFunction = _default_digest,
) -> Trajectory:
    """Exact Algorithm-1 Euler predictor/corrector recurrence."""

    current = initial_state
    initial_query = VelocityQuery(
        arm="u0_uni_inv_roundtrip",
        phase="inverse",
        query_index=0,
        coordinate=schedule.coordinates_ascending[0],
    )
    previous_velocity, initial_sha, previous_velocity_sha = _guarded_velocity(
        current,
        initial_query,
        velocity_fn=velocity_fn,
        digest_state=digest_state,
    )
    records: list[dict[str, Any]] = [
        {
            "kind": "uni_inv_initial_velocity",
            "query": initial_query.as_dict(),
            "state_sha256": initial_sha,
            "velocity_sha256": previous_velocity_sha,
        }
    ]
    for interval_index in range(1, schedule.interval_count + 1):
        delta = schedule.positive_delta(interval_index)
        # The next model query is allowed to allocate a new velocity but not
        # to recycle and overwrite the buffer returned by the previous query.
        # Save the identity before entering the callback: digesting only when
        # the trace row is written would silently relabel the predictor with a
        # subsequently overwritten value.
        previous_velocity_sha = require_sha256(
            digest_state(previous_velocity),
            label="U0 previous velocity digest before correction query",
        )
        predictor, corrected_before_sha, predictor_sha = _guarded_advance(
            current,
            previous_velocity,
            delta,
            advance=advance,
            digest_state=digest_state,
        )
        query = VelocityQuery(
            arm="u0_uni_inv_roundtrip",
            phase="inverse",
            query_index=interval_index,
            coordinate=schedule.coordinates_ascending[interval_index],
        )
        corrected_velocity, checked_predictor_sha, corrected_velocity_sha = (
            _guarded_velocity(
                predictor,
                query,
                velocity_fn=velocity_fn,
                digest_state=digest_state,
            )
        )
        if (
            require_sha256(
                digest_state(previous_velocity),
                label="U0 previous velocity digest after correction query",
            )
            != previous_velocity_sha
        ):
            fail("Uni-Inv correction query mutated the previous velocity buffer")
        if checked_predictor_sha != predictor_sha:
            fail("Uni-Inv predictor changed before the correction query")
        corrected, checked_before_sha, corrected_sha = _guarded_advance(
            current,
            corrected_velocity,
            delta,
            advance=advance,
            digest_state=digest_state,
        )
        if checked_before_sha != corrected_before_sha:
            fail("Uni-Inv corrected base changed between predictor and corrector")
        records.append(
            {
                "kind": "uni_inv_predictor_corrector",
                "interval_index_ascending": interval_index,
                "query": query.as_dict(),
                "delta_sigma_float32_be_hex": _float32_hex(
                    delta, label="U0 inverse delta"
                ),
                "corrected_before_sha256": corrected_before_sha,
                "previous_velocity_sha256": previous_velocity_sha,
                "predictor_sha256": predictor_sha,
                "corrected_velocity_sha256": corrected_velocity_sha,
                "corrected_after_sha256": corrected_sha,
            }
        )
        current = corrected
        previous_velocity = corrected_velocity
    return Trajectory(
        arm="u0_uni_inv_roundtrip",
        phase="inverse",
        initial_state=initial_state,
        final_state=current,
        records=tuple(records),
        model_forward_calls=schedule.interval_count + 1,
    )


def matched_euler_reconstruct(
    inverted_state: Any,
    velocity_fn: VelocityFunction,
    *,
    arm: str,
    schedule: RoundtripSchedule = EXACT40_ROUNDTRIP_SCHEDULE,
    advance: AdvanceFunction = _default_advance,
    digest_state: DigestFunction = _default_digest,
) -> Trajectory:
    """Reconstruct with one identical descending explicit-Euler solver."""

    if arm not in SOLVER_ARMS:
        fail("matched reconstruction arm differs from E0/U0")
    current = inverted_state
    records: list[dict[str, Any]] = []
    for query_index, coordinate_index in enumerate(
        range(schedule.interval_count, 0, -1)
    ):
        query = VelocityQuery(
            arm=arm,
            phase="reconstruct",
            query_index=query_index,
            coordinate=schedule.coordinates_ascending[coordinate_index],
        )
        velocity, state_sha, velocity_sha = _guarded_velocity(
            current,
            query,
            velocity_fn=velocity_fn,
            digest_state=digest_state,
        )
        delta = schedule.negative_delta(coordinate_index)
        next_state, _, next_sha = _guarded_advance(
            current,
            velocity,
            delta,
            advance=advance,
            digest_state=digest_state,
        )
        records.append(
            {
                "kind": "matched_explicit_euler_reconstruct",
                "interval_index_descending": coordinate_index,
                "query": query.as_dict(),
                "delta_sigma_float32_be_hex": _float32_hex(
                    delta, label="matched reconstruction delta"
                ),
                "state_before_sha256": state_sha,
                "velocity_sha256": velocity_sha,
                "state_after_sha256": next_sha,
            }
        )
        current = next_state
    return Trajectory(
        arm=arm,
        phase="reconstruct",
        initial_state=inverted_state,
        final_state=current,
        records=tuple(records),
        model_forward_calls=schedule.interval_count,
    )


def run_solver_arm(
    initial_state: Any,
    velocity_fn: VelocityFunction,
    *,
    arm: str,
    schedule: RoundtripSchedule = EXACT40_ROUNDTRIP_SCHEDULE,
    advance: AdvanceFunction = _default_advance,
    digest_state: DigestFunction = _default_digest,
) -> tuple[Trajectory, Trajectory]:
    if arm == "e0_vanilla_euler_roundtrip":
        inversion = vanilla_inverse_euler(
            initial_state,
            velocity_fn,
            schedule=schedule,
            advance=advance,
            digest_state=digest_state,
        )
    elif arm == "u0_uni_inv_roundtrip":
        inversion = uni_inv_predictor_corrector(
            initial_state,
            velocity_fn,
            schedule=schedule,
            advance=advance,
            digest_state=digest_state,
        )
    else:
        fail("run_solver_arm accepts only E0 or U0")
    reconstruction = matched_euler_reconstruct(
        inversion.final_state,
        velocity_fn,
        arm=arm,
        schedule=schedule,
        advance=advance,
        digest_state=digest_state,
    )
    return inversion, reconstruction


def temporal_quintile_records() -> list[dict[str, int]]:
    records = [
        {
            "quintile_index": index,
            "frame_start_inclusive": start,
            "frame_end_exclusive": end,
            "frame_count": end - start,
        }
        for index, (start, end) in enumerate(TEMPORAL_QUINTILE_BOUNDS)
    ]
    flattened = [
        frame
        for start, end in TEMPORAL_QUINTILE_BOUNDS
        for frame in range(start, end)
    ]
    if flattened != list(range(FRAME_COUNT)):
        fail("temporal quintiles are not an exact disjoint cover of 81 frames")
    return records


def metric_backend_receipts(
    method_source_archive_sha256: str,
    *,
    lpips_available: bool,
    lpips_weights_sha256: Optional[str] = None,
) -> dict[str, Mapping[str, Any]]:
    archive_sha = require_sha256(
        method_source_archive_sha256, label="metric backend source archive"
    )
    if type(lpips_available) is not bool:
        fail("LPIPS backend availability must be boolean")
    if lpips_available:
        weights_sha = require_sha256(
            lpips_weights_sha256, label="LPIPS weights SHA"
        )
        package_version: Optional[str] = "0.1.4"
    else:
        if lpips_weights_sha256 is not None:
            fail("unavailable LPIPS backend cannot declare weights")
        weights_sha = None
        package_version = None
    rows = {
        "psnr": {
            "schema_version": BACKEND_SCHEMA,
            "name": "psnr",
            "available": True,
            "implementation": "a0_runtime.fp32_rgb_psnr_v1",
            "implementation_sha256": archive_sha,
            "package": None,
            "package_version": None,
            "weights_sha256": None,
        },
        "ssim": {
            "schema_version": BACKEND_SCHEMA,
            "name": "ssim",
            "available": True,
            "implementation": "a0_runtime.fp32_rgb_ssim_gaussian11_v1",
            "implementation_sha256": archive_sha,
            "package": None,
            "package_version": None,
            "weights_sha256": None,
        },
        "lpips": {
            "schema_version": BACKEND_SCHEMA,
            "name": "lpips",
            "available": lpips_available,
            "implementation": "lpips.LPIPS(net=alex,version=0.1,spatial=False)",
            "implementation_sha256": archive_sha,
            "package": "lpips",
            "package_version": package_version,
            "weights_sha256": weights_sha,
        },
    }
    return {name: finalize_evidence_record(row) for name, row in rows.items()}


def available_metric(
    name: str, value: float, *, backend_digest: str
) -> dict[str, Any]:
    directions = {"psnr": "higher", "ssim": "higher", "lpips": "lower"}
    if name not in directions or isinstance(value, bool):
        fail("metric name/value differs")
    number = float(value)
    if not math.isfinite(number):
        fail(f"{name} metric must be finite")
    return {
        "name": name,
        "status": "available",
        "value": number,
        "better": directions[name],
        "reason": None,
        "backend_digest": require_sha256(
            backend_digest, label=f"{name} backend digest"
        ),
    }


def pending_metric(
    name: str, reason: str, *, backend_digest: str
) -> dict[str, Any]:
    directions = {"psnr": "higher", "ssim": "higher", "lpips": "lower"}
    if name not in directions or type(reason) is not str or not reason.strip():
        fail("pending metric requires a known name and non-empty reason")
    return {
        "name": name,
        "status": "pending",
        "value": None,
        "better": directions[name],
        "reason": reason,
        "backend_digest": require_sha256(
            backend_digest, label=f"{name} backend digest"
        ),
    }


def _validate_backend(
    value: Any,
    *,
    name: str,
    method_source_archive_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    backend = _validated_record(
        value,
        {
            "schema_version",
            "name",
            "available",
            "implementation",
            "implementation_sha256",
            "package",
            "package_version",
            "weights_sha256",
        },
        label=f"{name} metric backend",
    )
    expected_implementation = {
        "psnr": "a0_runtime.fp32_rgb_psnr_v1",
        "ssim": "a0_runtime.fp32_rgb_ssim_gaussian11_v1",
        "lpips": "lpips.LPIPS(net=alex,version=0.1,spatial=False)",
    }
    if (
        name not in expected_implementation
        or backend.get("schema_version") != BACKEND_SCHEMA
        or backend.get("name") != name
        or type(backend.get("available")) is not bool
        or backend.get("implementation") != expected_implementation[name]
    ):
        fail(f"{name} metric backend identity differs")
    implementation_sha = require_sha256(
        backend.get("implementation_sha256"),
        label=f"{name} backend implementation SHA",
    )
    if (
        method_source_archive_sha256 is not None
        and implementation_sha != method_source_archive_sha256
    ):
        fail(f"{name} metric backend is not bound to the method archive")
    if name in {"psnr", "ssim"}:
        if (
            backend.get("available") is not True
            or backend.get("package") is not None
            or backend.get("package_version") is not None
            or backend.get("weights_sha256") is not None
        ):
            fail(f"{name} built-in backend closure differs")
    elif backend.get("available") is True:
        if (
            backend.get("package") != "lpips"
            or backend.get("package_version") != "0.1.4"
        ):
            fail("available LPIPS backend package identity differs")
        require_sha256(backend.get("weights_sha256"), label="LPIPS weights SHA")
    elif (
        backend.get("package") != "lpips"
        or backend.get("package_version") is not None
        or backend.get("weights_sha256") is not None
    ):
        fail("unavailable LPIPS backend must not claim package weights")
    return backend


def _validate_metric(
    value: Any,
    *,
    name: str,
    label: str,
    backend: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "name",
        "status",
        "value",
        "better",
        "reason",
        "backend_digest",
    }:
        fail(f"{label} {name} metric schema differs")
    expected_better = "lower" if name == "lpips" else "higher"
    if value.get("name") != name or value.get("better") != expected_better:
        fail(f"{label} {name} metric identity differs")
    if value.get("backend_digest") != backend.get("digest"):
        fail(f"{label} {name} metric backend binding differs")
    status = value.get("status")
    if status == "available":
        number = value.get("value")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            fail(f"{label} {name} available value is not numeric")
        if not math.isfinite(float(number)) or value.get("reason") is not None:
            fail(f"{label} {name} available value is non-finite/ambiguous")
        if backend.get("available") is not True:
            fail(f"{label} {name} is available while its backend is unavailable")
    elif status == "pending":
        if value.get("value") is not None or not isinstance(value.get("reason"), str):
            fail(f"{label} {name} pending value is ambiguous")
        if not str(value["reason"]).strip():
            fail(f"{label} {name} pending reason is empty")
        if backend.get("available") is not False:
            fail(f"{label} {name} is pending while its backend is available")
    else:
        fail(f"{label} {name} metric status differs")
    return value


def validate_metric_packet(
    packet: Any,
    *,
    arm: str,
    method_source_archive_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    if arm not in ARMS or not isinstance(packet, Mapping):
        fail("metric packet arm/root differs")
    expected = {
        "schema_version",
        "arm",
        "reference",
        "measurement_domain",
        "reference_rgb_tensor_sha256",
        "candidate_rgb_tensor_sha256",
        "full_video",
        "temporal_quintiles",
        "backends",
    }
    if set(packet) != expected:
        fail(f"{arm} metric packet keys differ")
    if (
        packet.get("schema_version") != METRIC_PACKET_SCHEMA
        or packet.get("arm") != arm
        or packet.get("reference") != "resized_source_rgb"
        or packet.get("measurement_domain")
        != "in_memory_vae_decode_before_video_encoding"
    ):
        fail(f"{arm} metric packet identity differs")
    require_sha256(
        packet.get("reference_rgb_tensor_sha256"),
        label=f"{arm} metric reference RGB tensor",
    )
    require_sha256(
        packet.get("candidate_rgb_tensor_sha256"),
        label=f"{arm} metric candidate RGB tensor",
    )
    backend_root = packet.get("backends")
    if not isinstance(backend_root, Mapping) or set(backend_root) != {
        "psnr",
        "ssim",
        "lpips",
    }:
        fail(f"{arm} metric backend set differs")
    backends = {
        name: _validate_backend(
            backend_root[name],
            name=name,
            method_source_archive_sha256=method_source_archive_sha256,
        )
        for name in ("psnr", "ssim", "lpips")
    }
    expected_slices = [
        {
            "frame_start_inclusive": 0,
            "frame_end_exclusive": FRAME_COUNT,
            "frame_count": FRAME_COUNT,
        }
    ] + [
        {
            "frame_start_inclusive": item["frame_start_inclusive"],
            "frame_end_exclusive": item["frame_end_exclusive"],
            "frame_count": item["frame_count"],
        }
        for item in temporal_quintile_records()
    ]
    full = packet.get("full_video")
    quintiles = packet.get("temporal_quintiles")
    if not isinstance(full, Mapping) or not isinstance(quintiles, list) or len(quintiles) != 5:
        fail(f"{arm} metric temporal structure differs")
    for index, (observed, expected_slice) in enumerate(
        zip([full, *quintiles], expected_slices)
    ):
        if not isinstance(observed, Mapping) or set(observed) != {
            *expected_slice.keys(),
            "quintile_index",
            "psnr",
            "ssim",
            "lpips",
        }:
            fail(f"{arm} metric slice {index} schema differs")
        expected_quintile = None if index == 0 else index - 1
        if observed.get("quintile_index") != expected_quintile:
            fail(f"{arm} metric slice {index} quintile index differs")
        for key, expected_value in expected_slice.items():
            if observed.get(key) != expected_value:
                fail(f"{arm} metric slice {index} frame bounds differ")
        for metric in ("psnr", "ssim", "lpips"):
            _validate_metric(
                observed.get(metric),
                name=metric,
                label=f"{arm} slice {index}",
                backend=backends[metric],
            )
    return packet


def _build_hard_gate(
    metrics_by_arm: Mapping[str, Any], *, evidence_closure_complete: bool
) -> dict[str, Any]:
    if type(evidence_closure_complete) is not bool:
        fail("A0 evidence closure flag must be boolean")
    if not isinstance(metrics_by_arm, Mapping) or set(metrics_by_arm) != set(ARMS):
        fail("hard gate requires exactly C0/E0/U0 metric packets")
    for arm in ARMS:
        validate_metric_packet(metrics_by_arm[arm], arm=arm)
    e0 = metrics_by_arm["e0_vanilla_euler_roundtrip"]["full_video"]
    u0 = metrics_by_arm["u0_uni_inv_roundtrip"]["full_video"]
    unavailable = [
        name
        for name in ("psnr", "ssim", "lpips")
        if e0[name]["status"] != "available" or u0[name]["status"] != "available"
    ]
    comparisons: dict[str, Any] = {}
    if unavailable:
        status = "pending"
        for name in ("psnr", "ssim", "lpips"):
            comparisons[name] = {
                "operator": "<" if name == "lpips" else ">",
                "e0_value": e0[name]["value"],
                "u0_value": u0[name]["value"],
                "u0_strictly_better": None,
            }
        reason = "required full-video metrics pending: " + ",".join(unavailable)
    else:
        comparisons = {
            "psnr": {
                "operator": ">",
                "e0_value": float(e0["psnr"]["value"]),
                "u0_value": float(u0["psnr"]["value"]),
                "u0_strictly_better": float(u0["psnr"]["value"])
                > float(e0["psnr"]["value"]),
            },
            "ssim": {
                "operator": ">",
                "e0_value": float(e0["ssim"]["value"]),
                "u0_value": float(u0["ssim"]["value"]),
                "u0_strictly_better": float(u0["ssim"]["value"])
                > float(e0["ssim"]["value"]),
            },
            "lpips": {
                "operator": "<",
                "e0_value": float(e0["lpips"]["value"]),
                "u0_value": float(u0["lpips"]["value"]),
                "u0_strictly_better": float(u0["lpips"]["value"])
                < float(e0["lpips"]["value"]),
            },
        }
        passed = all(item["u0_strictly_better"] is True for item in comparisons.values())
        status = "pass" if passed else "fail"
        reason = (
            "U0 is strictly better than E0 on full-video PSNR, SSIM, and LPIPS"
            if passed
            else "U0 is not strictly better than E0 on all three full-video metrics"
        )
    value = {
        "schema_version": GATE_SCHEMA,
        "status": status,
        "scope": "numerical_roundtrip_control_only",
        "comparisons": comparisons,
        "reason": reason,
        # A1 (the prompt-paired Uni-Edit correction) is a separate experiment
        # and is admitted only by a completed strict three-metric A0 pass.
        # In particular, an unavailable LPIPS backend is pending, never GO.
        "evidence_closure_complete": evidence_closure_complete,
        "a1_prompt_paired_correction_authorized": (
            status == "pass" and evidence_closure_complete
        ),
        "visual_review_status": "pending",
        "automatic_visual_claim": False,
        "semantic_editing_claim_authorized": False,
        "method_success_claimed": False,
    }
    return {**value, "digest": object_sha256(value)}


def evaluate_hard_gate(metrics_by_arm: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate metrics only; this public path can never authorize A1."""

    return _build_hard_gate(metrics_by_arm, evidence_closure_complete=False)


def _closed_hard_gate(metrics_by_arm: Mapping[str, Any]) -> dict[str, Any]:
    """Build the expected gate only inside the completed-receipt closure."""

    return _build_hard_gate(metrics_by_arm, evidence_closure_complete=True)


def _validate_source(value: Any) -> Mapping[str, Any]:
    source = _validated_record(
        value,
        {
            "schema_version",
            "iid",
            "source_video_sha256",
            "source_dataset_spec_sha256",
            "source_dataset_receipt_sha256",
            "source_dataset_receipt_digest",
            "orbit_dataset_spec_sha256",
            "orbit_dataset_receipt_sha256",
            "orbit_dataset_receipt_digest",
            "orbit_row_digest",
            "clean_latent_sha256",
            "clean_latent_shape",
            "clean_latent_dtype",
            "clean_latent_coordinate",
            "resized_source_rgb_tensor_sha256",
            "frame_count",
            "fps",
            "height",
            "width",
        },
        label="A0 source",
    )
    expected = {
        "schema_version": SOURCE_SCHEMA,
        "iid": IID,
        "source_video_sha256": SOURCE_VIDEO_SHA256,
        "source_dataset_spec_sha256": SOURCE_DATASET_SPEC_SHA256,
        "source_dataset_receipt_sha256": SOURCE_DATASET_RECEIPT_SHA256,
        "source_dataset_receipt_digest": SOURCE_DATASET_RECEIPT_DIGEST,
        "orbit_dataset_spec_sha256": ORBIT_DATASET_SPEC_SHA256,
        "orbit_dataset_receipt_sha256": ORBIT_DATASET_RECEIPT_SHA256,
        "orbit_dataset_receipt_digest": ORBIT_DATASET_RECEIPT_DIGEST,
        "orbit_row_digest": ORBIT_ROW_DIGEST,
        "clean_latent_shape": list(LATENT_SHAPE),
        "clean_latent_dtype": "torch.float32",
        "clean_latent_coordinate": "normalized_bernini_vae_V0_video",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "height": HEIGHT,
        "width": WIDTH,
    }
    if any(source.get(key) != expected_value for key, expected_value in expected.items()):
        fail("A0 source authority or geometry differs")
    require_sha256(source.get("clean_latent_sha256"), label="source clean latent")
    require_sha256(
        source.get("resized_source_rgb_tensor_sha256"),
        label="source resized RGB tensor",
    )
    return source


def _validate_model(value: Any) -> Mapping[str, Any]:
    model = _validated_record(
        value,
        {
            "schema_version",
            "renderer",
            "bernini_commit",
            "veomni_commit",
            "checkpoint_tree_sha256",
            "checkpoint_manifest_sha256",
            "model_state_sha256",
            "transformer_count",
            "transformer_block_count",
            "transformer_frozen_eval",
            "vae_identity_digest",
            "vae_file_sha256",
            "vae_frozen_eval",
        },
        label="A0 model",
    )
    expected = {
        "schema_version": MODEL_SCHEMA,
        "renderer": "Bernini-R-1.3B-transformer_1",
        "bernini_commit": EXPECTED_BERNINI_COMMIT,
        "veomni_commit": EXPECTED_VEOMNI_COMMIT,
        "checkpoint_tree_sha256": EXPECTED_CHECKPOINT_TREE_SHA256,
        "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        "transformer_count": 1,
        "transformer_block_count": 30,
        "transformer_frozen_eval": True,
        "vae_identity_digest": PINNED_VAE_IDENTITY_DIGEST,
        "vae_file_sha256": EXPECTED_VAE_FILE_SHA256,
        "vae_frozen_eval": True,
    }
    if any(model.get(key) != expected_value for key, expected_value in expected.items()):
        fail("A0 frozen model identity differs")
    return model


def _validate_parallel(value: Any) -> Mapping[str, Any]:
    parallel = _validated_record(
        value,
        {
            "schema_version",
            "topology",
            "world_size",
            "sequence_parallel_size",
            "data_parallel_size",
            "distributed_invocation_count",
            "world4_consensus",
            "single_model_replica_per_rank",
        },
        label="A0 parallel",
    )
    expected = {
        "schema_version": PARALLEL_SCHEMA,
        "topology": "WORLD4_DP1_SP4",
        "world_size": WORLD_SIZE,
        "sequence_parallel_size": SP_SIZE,
        "data_parallel_size": 1,
        "distributed_invocation_count": 1,
        "world4_consensus": True,
        "single_model_replica_per_rank": True,
    }
    if any(parallel.get(key) != expected_value for key, expected_value in expected.items()):
        fail("A0 parallel contract differs")
    return parallel


def _validate_prompt(value: Any) -> Mapping[str, Any]:
    prompt = _validated_record(
        value,
        {
            "schema_version",
            "kind",
            "raw_text",
            "utf8_sha256",
            "utf8_bytes",
            "embedding_sha256",
            "embedding_shape",
            "embedding_dtype",
            "encoder_call_count",
            "guidance_branch_count",
            "same_for_e0_u0_inverse_reconstruct",
            "cfg_used",
            "apg_used",
        },
        label="A0 prompt",
    )
    expected = {
        "schema_version": PROMPT_SCHEMA,
        "kind": "single_blank_t2v_condition",
        "raw_text": "",
        "utf8_sha256": hashlib.sha256(b"").hexdigest(),
        "utf8_bytes": 0,
        "embedding_shape": [1, 512, 4096],
        "embedding_dtype": "torch.bfloat16",
        "encoder_call_count": 1,
        "guidance_branch_count": 1,
        "same_for_e0_u0_inverse_reconstruct": True,
        "cfg_used": False,
        "apg_used": False,
    }
    if any(prompt.get(key) != expected_value for key, expected_value in expected.items()):
        fail("A0 blank prompt contract differs")
    require_sha256(prompt.get("embedding_sha256"), label="blank prompt embedding")
    return prompt


def _validate_trace_query(
    value: Any,
    *,
    arm: str,
    phase: str,
    query_index: int,
    coordinate_index: int,
) -> Mapping[str, Any]:
    query = _exact_mapping(
        value, {"arm", "phase", "query_index", "coordinate"}, label="trace query"
    )
    if query != {
        "arm": arm,
        "phase": phase,
        "query_index": query_index,
        "coordinate": EXACT40_ROUNDTRIP_SCHEDULE.coordinates_ascending[
            coordinate_index
        ].as_dict(),
    }:
        fail("trace query coordinate/order differs")
    return query


def _trace_sha(value: Any, *, label: str) -> str:
    return require_sha256(value, label=label)


def validate_trace_receipt(
    value: Any, *, arm: str, phase: str
) -> Mapping[str, Any]:
    if arm not in SOLVER_ARMS or phase not in {"inverse", "reconstruct"}:
        fail("trace arm/phase differs")
    trace = _validated_record(
        value,
        {
            "schema_version",
            "arm",
            "phase",
            "initial_state_sha256",
            "final_state_sha256",
            "model_forward_calls",
            "records",
        },
        label=f"{arm} {phase} trace",
    )
    if trace.get("schema_version") != TRACE_SCHEMA or (
        trace.get("arm"), trace.get("phase")
    ) != (arm, phase):
        fail("trace identity differs")
    initial_sha = _trace_sha(trace.get("initial_state_sha256"), label="trace initial")
    final_sha = _trace_sha(trace.get("final_state_sha256"), label="trace final")
    records = trace.get("records")
    if not isinstance(records, list):
        fail("trace records must be a list")

    if phase == "reconstruct":
        if trace.get("model_forward_calls") != 40 or len(records) != 40:
            fail("reconstruction trace count differs")
        previous = initial_sha
        for ordinal, (record, coordinate_index) in enumerate(
            zip(records, range(40, 0, -1))
        ):
            row = _exact_mapping(
                record,
                {
                    "kind",
                    "interval_index_descending",
                    "query",
                    "delta_sigma_float32_be_hex",
                    "state_before_sha256",
                    "velocity_sha256",
                    "state_after_sha256",
                },
                label="reconstruction trace row",
            )
            _validate_trace_query(
                row["query"],
                arm=arm,
                phase=phase,
                query_index=ordinal,
                coordinate_index=coordinate_index,
            )
            state_before = _trace_sha(row["state_before_sha256"], label="state before")
            state_after = _trace_sha(row["state_after_sha256"], label="state after")
            _trace_sha(row["velocity_sha256"], label="velocity")
            if (
                row.get("kind") != "matched_explicit_euler_reconstruct"
                or row.get("interval_index_descending") != coordinate_index
                or row.get("delta_sigma_float32_be_hex")
                != _float32_hex(
                    EXACT40_ROUNDTRIP_SCHEDULE.negative_delta(coordinate_index),
                    label="expected reconstruction delta",
                )
                or state_before != previous
            ):
                fail("reconstruction trace recurrence chain differs")
            previous = state_after
        if previous != final_sha:
            fail("reconstruction trace final state differs")
        return trace

    if arm == "e0_vanilla_euler_roundtrip":
        if trace.get("model_forward_calls") != 40 or len(records) != 40:
            fail("E0 inverse trace count differs")
        previous = initial_sha
        for ordinal, record in enumerate(records):
            interval_index = ordinal + 1
            row = _exact_mapping(
                record,
                {
                    "kind",
                    "interval_index_ascending",
                    "query",
                    "delta_sigma_float32_be_hex",
                    "state_before_sha256",
                    "velocity_sha256",
                    "state_after_sha256",
                },
                label="E0 inverse trace row",
            )
            _validate_trace_query(
                row["query"],
                arm=arm,
                phase=phase,
                query_index=ordinal,
                coordinate_index=ordinal,
            )
            state_before = _trace_sha(row["state_before_sha256"], label="state before")
            state_after = _trace_sha(row["state_after_sha256"], label="state after")
            _trace_sha(row["velocity_sha256"], label="velocity")
            if (
                row.get("kind") != "explicit_euler_inverse"
                or row.get("interval_index_ascending") != interval_index
                or row.get("delta_sigma_float32_be_hex")
                != _float32_hex(
                    EXACT40_ROUNDTRIP_SCHEDULE.positive_delta(interval_index),
                    label="expected E0 inverse delta",
                )
                or state_before != previous
            ):
                fail("E0 inverse trace recurrence chain differs")
            previous = state_after
        if previous != final_sha:
            fail("E0 inverse trace final state differs")
        return trace

    if trace.get("model_forward_calls") != 41 or len(records) != 41:
        fail("U0 inverse trace count differs")
    initial = _exact_mapping(
        records[0],
        {"kind", "query", "state_sha256", "velocity_sha256"},
        label="U0 initial trace row",
    )
    _validate_trace_query(
        initial["query"],
        arm=arm,
        phase=phase,
        query_index=0,
        coordinate_index=0,
    )
    if (
        initial.get("kind") != "uni_inv_initial_velocity"
        or _trace_sha(initial.get("state_sha256"), label="U0 initial state")
        != initial_sha
    ):
        fail("U0 initial trace state differs")
    previous_velocity = _trace_sha(
        initial.get("velocity_sha256"), label="U0 initial velocity"
    )
    previous_state = initial_sha
    for interval_index, record in enumerate(records[1:], start=1):
        row = _exact_mapping(
            record,
            {
                "kind",
                "interval_index_ascending",
                "query",
                "delta_sigma_float32_be_hex",
                "corrected_before_sha256",
                "previous_velocity_sha256",
                "predictor_sha256",
                "corrected_velocity_sha256",
                "corrected_after_sha256",
            },
            label="U0 corrector trace row",
        )
        _validate_trace_query(
            row["query"],
            arm=arm,
            phase=phase,
            query_index=interval_index,
            coordinate_index=interval_index,
        )
        corrected_before = _trace_sha(
            row["corrected_before_sha256"], label="U0 corrected base"
        )
        corrected_after = _trace_sha(
            row["corrected_after_sha256"], label="U0 corrected result"
        )
        _trace_sha(row["predictor_sha256"], label="U0 predictor")
        corrected_velocity = _trace_sha(
            row["corrected_velocity_sha256"], label="U0 corrected velocity"
        )
        if (
            row.get("kind") != "uni_inv_predictor_corrector"
            or row.get("interval_index_ascending") != interval_index
            or row.get("delta_sigma_float32_be_hex")
            != _float32_hex(
                EXACT40_ROUNDTRIP_SCHEDULE.positive_delta(interval_index),
                label="expected U0 inverse delta",
            )
            or corrected_before != previous_state
            or row.get("previous_velocity_sha256") != previous_velocity
        ):
            fail("U0 inverse trace recurrence chain differs")
        previous_state = corrected_after
        previous_velocity = corrected_velocity
    if previous_state != final_sha:
        fail("U0 inverse trace final state differs")
    return trace


def _validate_media(value: Any, *, arm: str) -> Mapping[str, Any]:
    media = _validated_record(
        value,
        {
            "schema_version",
            "arm",
            "decode_input_latent_sha256",
            "decoded_rgb_tensor_sha256",
            "mp4_sha256",
            "frame_count",
            "fps",
            "height",
            "width",
            "video_codec",
            "pixel_format",
            "vae_decode_count",
            "vae_frozen_eval",
            "metrics_measured_before_video_encoding",
        },
        label=f"{arm} media",
    )
    expected = {
        "schema_version": MEDIA_SCHEMA,
        "arm": arm,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "height": HEIGHT,
        "width": WIDTH,
        "video_codec": "h264",
        "pixel_format": "yuv420p",
        "vae_decode_count": 1,
        "vae_frozen_eval": True,
        "metrics_measured_before_video_encoding": True,
    }
    if any(media.get(key) != expected_value for key, expected_value in expected.items()):
        fail(f"{arm} exact81 media/decode closure differs")
    for key in (
        "decode_input_latent_sha256",
        "decoded_rgb_tensor_sha256",
        "mp4_sha256",
    ):
        require_sha256(media.get(key), label=f"{arm} media {key}")
    return media


def _validate_arm_bindings(
    value: Any,
    *,
    source: Mapping[str, Any],
    model: Mapping[str, Any],
    parallel: Mapping[str, Any],
    prompt: Mapping[str, Any],
    schedule: Mapping[str, Any],
    solver_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    bindings = _exact_mapping(
        value,
        {
            "source_digest",
            "model_digest",
            "parallel_digest",
            "prompt_digest",
            "prompt_embedding_sha256",
            "schedule_digest",
            "solver_contract_digest",
        },
        label="arm bindings",
    )
    if bindings != {
        "source_digest": source["digest"],
        "model_digest": model["digest"],
        "parallel_digest": parallel["digest"],
        "prompt_digest": prompt["digest"],
        "prompt_embedding_sha256": prompt["embedding_sha256"],
        "schedule_digest": schedule["digest"],
        "solver_contract_digest": solver_contract["digest"],
    }:
        fail("arm source/model/parallel/blank-prompt/schedule binding differs")
    return bindings


def _validate_arm(
    value: Any,
    *,
    arm: str,
    source: Mapping[str, Any],
    model: Mapping[str, Any],
    parallel: Mapping[str, Any],
    prompt: Mapping[str, Any],
    schedule: Mapping[str, Any],
    solver_contract: Mapping[str, Any],
    method_source_archive_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if arm == "c0_vae_ceiling":
        keys = {
            "schema_version",
            "arm",
            "role",
            "bindings",
            "prompt_consumed",
            "transformer_forward_calls",
            "inversion_trace",
            "reconstruction_trace",
            "input_clean_latent_sha256",
            "decode_input_latent_sha256",
            "media",
            "metrics",
        }
    else:
        keys = {
            "schema_version",
            "arm",
            "role",
            "bindings",
            "prompt_consumed",
            "transformer_forward_calls",
            "inversion_trace",
            "reconstruction_trace",
            "state_chain",
            "media",
            "metrics",
        }
    row = _validated_record(value, keys, label=f"A0 arm {arm}")
    if row.get("schema_version") != ARM_SCHEMA or row.get("arm") != arm:
        fail(f"A0 arm {arm} identity differs")
    _validate_arm_bindings(
        row.get("bindings"),
        source=source,
        model=model,
        parallel=parallel,
        prompt=prompt,
        schedule=schedule,
        solver_contract=solver_contract,
    )
    media = _validate_media(row.get("media"), arm=arm)
    metrics = validate_metric_packet(
        row.get("metrics"),
        arm=arm,
        method_source_archive_sha256=method_source_archive_sha256,
    )
    if (
        metrics.get("reference_rgb_tensor_sha256")
        != source.get("resized_source_rgb_tensor_sha256")
        or metrics.get("candidate_rgb_tensor_sha256")
        != media.get("decoded_rgb_tensor_sha256")
    ):
        fail(f"{arm} metric tensors are not bound to source/media")

    clean_sha = source["clean_latent_sha256"]
    if arm == "c0_vae_ceiling":
        if (
            row.get("role") != "frozen_vae_codec_ceiling"
            or row.get("prompt_consumed") is not False
            or row.get("transformer_forward_calls") != 0
            or row.get("inversion_trace") is not None
            or row.get("reconstruction_trace") is not None
            or row.get("input_clean_latent_sha256") != clean_sha
            or row.get("decode_input_latent_sha256") != clean_sha
            or media.get("decode_input_latent_sha256") != clean_sha
        ):
            fail("C0 zero-forward codec closure differs")
        return row, metrics

    if (
        row.get("role") != "numerical_roundtrip_control"
        or row.get("prompt_consumed") is not True
        or row.get("transformer_forward_calls")
        != ARM_MODEL_FORWARD_CALLS[arm]
    ):
        fail(f"{arm} role/prompt/forward closure differs")
    inversion = validate_trace_receipt(row.get("inversion_trace"), arm=arm, phase="inverse")
    reconstruction = validate_trace_receipt(
        row.get("reconstruction_trace"), arm=arm, phase="reconstruct"
    )
    chain = _exact_mapping(
        row.get("state_chain"),
        {
            "source_clean_latent_sha256",
            "inversion_initial_state_sha256",
            "inversion_final_state_sha256",
            "reconstruction_initial_state_sha256",
            "reconstruction_final_state_sha256",
            "decode_input_latent_sha256",
            "inversion_to_reconstruction_exact",
            "reconstruction_to_decode_exact",
        },
        label=f"{arm} state chain",
    )
    expected_chain = {
        "source_clean_latent_sha256": clean_sha,
        "inversion_initial_state_sha256": clean_sha,
        "inversion_final_state_sha256": inversion["final_state_sha256"],
        "reconstruction_initial_state_sha256": inversion["final_state_sha256"],
        "reconstruction_final_state_sha256": reconstruction["final_state_sha256"],
        "decode_input_latent_sha256": reconstruction["final_state_sha256"],
        "inversion_to_reconstruction_exact": True,
        "reconstruction_to_decode_exact": True,
    }
    if (
        chain != expected_chain
        or inversion.get("initial_state_sha256") != clean_sha
        or reconstruction.get("initial_state_sha256")
        != inversion.get("final_state_sha256")
        or media.get("decode_input_latent_sha256")
        != reconstruction.get("final_state_sha256")
    ):
        fail(f"{arm} inversion/reconstruction/decode state chain differs")
    return row, metrics


RECEIPT_KEYS = {
    "schema_version",
    "complete",
    "experiment",
    "authority",
    "source",
    "model",
    "parallel",
    "prompt",
    "schedule",
    "solver_contract",
    "execution",
    "arms",
    "hard_gate",
    "dependencies",
    "runtime_versions",
    "prohibitions",
    "visual_review",
    "method_source_revision",
    "method_source_archive_sha256",
    "receipt_digest",
}


def validate_receipt(receipt: Any) -> Mapping[str, Any]:
    """Strict model-free validator for a completed A0 receipt."""

    if not isinstance(receipt, Mapping) or set(receipt) != RECEIPT_KEYS:
        fail("A0 receipt top-level keys differ")
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("complete") is not True:
        fail("A0 receipt identity/completion differs")
    declared = require_sha256(receipt.get("receipt_digest"), label="receipt digest")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest")
    if object_sha256(unsigned) != declared:
        fail("A0 receipt digest mismatch")
    if _SHA1.fullmatch(str(receipt.get("method_source_revision"))) is None:
        fail("A0 method source revision is not a full SHA-1")
    require_sha256(
        receipt.get("method_source_archive_sha256"),
        label="method source archive SHA",
    )
    method_source_archive_sha256 = str(receipt["method_source_archive_sha256"])
    experiment = receipt.get("experiment")
    if not isinstance(experiment, Mapping) or experiment != {
        "stage": "A0",
        "scope": "roundtrip_control_only",
        "read_only": True,
        "training": False,
        "editing": False,
    }:
        fail("A0 experiment scope differs")
    authority = receipt.get("authority")
    if authority != {
        "paper_url": PAPER_URL,
        "arxiv_algorithm_url": ARXIV_ALGORITHM_URL,
        "official_repository_url": OFFICIAL_REPOSITORY_URL,
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "official_uniinv_scheduler_git_blob": OFFICIAL_UNIINV_SCHEDULER_GIT_BLOB,
    }:
        fail("A0 UniEdit-Flow authority differs")
    source = _validate_source(receipt.get("source"))
    model = _validate_model(receipt.get("model"))
    parallel = _validate_parallel(receipt.get("parallel"))
    prompt = _validate_prompt(receipt.get("prompt"))
    schedule = receipt.get("schedule")
    expected_schedule = EXACT40_ROUNDTRIP_SCHEDULE.receipt()
    if schedule != expected_schedule:
        fail("A0 receipt schedule differs from exact40 runtime authority")
    solver_contract = receipt.get("solver_contract")
    if solver_contract != solver_contract_receipt():
        fail("A0 solver contract differs")
    if receipt.get("dependencies") != dependency_receipt(
        method_source_archive_sha256
    ):
        fail("A0 dependency closure differs")
    if receipt.get("runtime_versions") != runtime_versions_receipt():
        fail("A0 runtime version closure differs")

    execution = receipt.get("execution")
    expected_execution = {
        "model_forward_calls_by_arm": dict(ARM_MODEL_FORWARD_CALLS),
        "conditional_model_forward_calls": TOTAL_MODEL_FORWARD_CALLS,
        "unconditional_model_forward_calls": 0,
        "cfg_combinations": 0,
        "apg_combinations": 0,
        "model_load_count": 1,
        "vae_load_count": 1,
        "vae_decode_count_by_arm": {arm: 1 for arm in ARMS},
        "total_vae_decode_count": 3,
        "media_output_count": 3,
        "scheduler_instance_count": 0,
        "scheduler_step_count": 0,
        "optimizer_instance_count": 0,
        "optimizer_steps": 0,
        "adapter_forward_calls": 0,
    }
    if execution != expected_execution:
        fail("A0 execution counts differ")

    arms = receipt.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        fail("A0 receipt arm set differs")
    metrics: dict[str, Any] = {}
    validated_arms: dict[str, Mapping[str, Any]] = {}
    for arm in ARMS:
        validated_arms[arm], metrics[arm] = _validate_arm(
            arms[arm],
            arm=arm,
            source=source,
            model=model,
            parallel=parallel,
            prompt=prompt,
            schedule=schedule,
            solver_contract=solver_contract,
            method_source_archive_sha256=method_source_archive_sha256,
        )
    first_backends = canonical_json_bytes(metrics[ARMS[0]]["backends"])
    if any(
        canonical_json_bytes(metrics[arm]["backends"]) != first_backends
        for arm in ARMS[1:]
    ):
        fail("A0 arms used different metric backend identities")
    prohibitions = receipt.get("prohibitions")
    expected_prohibitions = {
        "optimizer_created": False,
        "optimizer_steps": 0,
        "adapter_loaded": False,
        "adapter_parameters": 0,
        "cfg_used": False,
        "apg_used": False,
        "uni_edit_a1_used": False,
        "scheduler_object_used_by_solver": False,
        "flow_shift_reapplied": False,
        "automatic_visual_claim": False,
        "semantic_method_success_claimed": False,
    }
    if prohibitions != expected_prohibitions:
        fail("A0 prohibition closure differs")
    visual = receipt.get("visual_review")
    if visual != {
        "status": "pending",
        "automatic_claim": False,
        "human_review_required": True,
    }:
        fail("A0 visual review must remain pending")
    # Only after every identity, trace, media, metric, execution, dependency,
    # and prohibition above has closed may the metric result admit A1.
    expected_gate = _closed_hard_gate(metrics)
    if receipt.get("hard_gate") != expected_gate:
        fail("A0 hard gate differs from closed evidence evaluation")
    return receipt


def finalize_receipt(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        fail("unsigned receipt already contains receipt_digest")
    if unsigned.get("hard_gate") != {}:
        fail("unsigned receipt hard gate must be empty")
    value = dict(unsigned)
    arms = value.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        fail("unsigned receipt arm set differs")
    value["hard_gate"] = _closed_hard_gate(
        {arm: arms[arm].get("metrics") for arm in ARMS}
    )
    value["receipt_digest"] = object_sha256(value)
    validate_receipt(value)
    return value


__all__ = [
    "ARMS",
    "ARM_MODEL_FORWARD_CALLS",
    "ARXIV_ALGORITHM_URL",
    "CONTRACT_SCHEMA",
    "EXACT40_ROUNDTRIP_SCHEDULE",
    "EXPECTED_EXACT40_SCHEDULE_SHA256",
    "FPS",
    "FRAME_COUNT",
    "GATE_SCHEMA",
    "METRIC_PACKET_SCHEMA",
    "OFFICIAL_REPOSITORY_COMMIT",
    "OFFICIAL_REPOSITORY_URL",
    "OFFICIAL_UNIINV_SCHEDULER_GIT_BLOB",
    "PAPER_URL",
    "RECEIPT_KEYS",
    "RECEIPT_SCHEMA",
    "RoundtripCoordinate",
    "RoundtripSchedule",
    "SOLVER_ARMS",
    "SP_SIZE",
    "TEMPORAL_QUINTILE_BOUNDS",
    "TOTAL_MODEL_FORWARD_CALLS",
    "TRACE_SCHEMA",
    "Trajectory",
    "UniEditFlowRoundtripError",
    "VelocityQuery",
    "WORLD_SIZE",
    "available_metric",
    "build_exact40_roundtrip_schedule",
    "canonical_json_bytes",
    "evaluate_hard_gate",
    "finalize_receipt",
    "matched_euler_reconstruct",
    "object_sha256",
    "pending_metric",
    "run_solver_arm",
    "temporal_quintile_records",
    "uni_inv_predictor_corrector",
    "validate_metric_packet",
    "validate_receipt",
    "vanilla_inverse_euler",
]
