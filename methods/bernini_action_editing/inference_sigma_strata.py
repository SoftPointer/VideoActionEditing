#!/usr/bin/env python3
"""Exact Bernini 40-step UniPC sigma strata for C2FR training.

This module deliberately does *not* recreate a flow schedule analytically.
The pinned values below were captured from the released Bernini-R 1.3B
``UniPCMultistepScheduler`` after ``set_timesteps(40)`` with ``flow_shift=5``.
That distinction matters: UniPC first builds float32 flow sigmas and then
casts its public timesteps to int64.  Consequently its last positive pair is
``(117, 0.11765105277299881)``, not the often-assumed analytic ``5 / 44``.

The helper has three narrow jobs:

* fail closed if a runtime UniPC schedule differs at even one float32 bit;
* select one exact ``(timestep, sigma)`` pair from the inference grid as a
  pure function of the absolute optimizer step;
* emit a deterministic schedule hash and per-stratum histogram for receipts.

It owns no random state and imports neither torch nor Bernini.  The training
loop can therefore audit the real scheduler once, then replace an upstream
random training pair after recovering the shared epsilon/noise realization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Mapping


class InferenceSigmaStrataError(RuntimeError):
    """Raised when the pinned inference-sigma contract is violated."""


SCHEDULE_SCHEMA = "bernini-unipc-sigma-schedule-v1"
RECEIPT_SCHEMA = "bernini-c2fr-inference-sigma-strata-receipt-v1"
SCHEDULER_CLASS = "UniPCMultistepScheduler"
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 40
FLOW_SHIFT = 5.0

# Captured from the pinned Diffusers UniPC instance, *not* rounded from a
# separately evaluated formula.  Public model timesteps are int64.
PINNED_TIMESTEPS: tuple[int, ...] = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)

# Big-endian IEEE-754 float32 bytes captured from scheduler.sigmas on CPU.
# Hex, rather than decimal literals, makes exactness explicit and makes the
# schedule digest independent of JSON float rendering.
PINNED_POSITIVE_SIGMA_FLOAT32_HEX: tuple[str, ...] = (
    "3f7fffef", "3f7eb1f9", "3f7d560b", "3f7beb53", "3f7a70da",
    "3f78e594", "3f77485b", "3f7597f0", "3f73d2f4", "3f71f7e6",
    "3f70051e", "3f6df8cb", "3f6bd0e9", "3f698b3c", "3f67254a",
    "3f649c50", "3f61ed37", "3f5f148a", "3f5c0e64", "3f58d661",
    "3f556787", "3f51bc2a", "3f4dcdd4", "3f499515", "3f45095d",
    "3f4020bc", "3f3acf9b", "3f35085f", "3f2ebaf8", "3f27d446",
    "3f203d59", "3f17da71", "3f0e89a7", "3f042120", "3ef0d923",
    "3ed6539a", "3eb80796", "3e9516ea", "3e58b351", "3df0f309",
)
TERMINAL_SIGMA_FLOAT32_HEX = "00000000"


def _float_from_float32_hex(value: str) -> float:
    return float(struct.unpack(">f", bytes.fromhex(value))[0])


def _float32_hex(value: Any, *, label: str) -> str:
    try:
        numeric = float(value)
        encoded = struct.pack(">f", numeric)
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise InferenceSigmaStrataError(f"{label} must be finite float32") from error
    if not math.isfinite(numeric):
        raise InferenceSigmaStrataError(f"{label} must be finite float32")
    return encoded.hex()


PINNED_POSITIVE_SIGMAS: tuple[float, ...] = tuple(
    _float_from_float32_hex(value)
    for value in PINNED_POSITIVE_SIGMA_FLOAT32_HEX
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _schedule_digest_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEDULE_SCHEMA,
        "scheduler_class": SCHEDULER_CLASS,
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "flow_shift_float64_hex": FLOW_SHIFT.hex(),
        "timesteps_int64": list(PINNED_TIMESTEPS),
        "positive_sigmas_float32_be_hex": list(
            PINNED_POSITIVE_SIGMA_FLOAT32_HEX
        ),
        "terminal_sigma_float32_be_hex": TERMINAL_SIGMA_FLOAT32_HEX,
    }


# Pin the digest as a second, independent guard against editing one constant
# without changing the declared experiment identity.
SCHEDULE_SHA256 = "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"
_COMPUTED_SCHEDULE_SHA256 = hashlib.sha256(
    _canonical_json_bytes(_schedule_digest_payload())
).hexdigest()
if _COMPUTED_SCHEDULE_SHA256 != SCHEDULE_SHA256:  # pragma: no cover - import guard
    raise RuntimeError("pinned Bernini UniPC schedule constants differ from their hash")


@dataclass(frozen=True)
class SigmaStratum:
    """One exact inference query selected for an optimizer step."""

    optimizer_step: int
    cycle_index: int
    schedule_index: int
    timestep: int
    sigma: float
    sigma_float32_be_hex: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer_step": self.optimizer_step,
            "cycle_index": self.cycle_index,
            "schedule_index": self.schedule_index,
            "timestep": self.timestep,
            "sigma": self.sigma,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
        }


def select_sigma_stratum(optimizer_step: int) -> SigmaStratum:
    """Select ``optimizer_step % 40`` in official descending inference order.

    The absolute optimizer step is intentional: all Ulysses ranks and resumed
    runs choose the same pair without restoring an auxiliary RNG or cursor.
    Every contiguous 40 optimizer updates visits every positive inference
    sigma exactly once.
    """

    if type(optimizer_step) is not int or optimizer_step < 0:
        raise InferenceSigmaStrataError(
            "optimizer_step must be a non-negative integer"
        )
    schedule_index = optimizer_step % NUM_INFERENCE_STEPS
    return SigmaStratum(
        optimizer_step=optimizer_step,
        cycle_index=optimizer_step // NUM_INFERENCE_STEPS,
        schedule_index=schedule_index,
        timestep=PINNED_TIMESTEPS[schedule_index],
        sigma=PINNED_POSITIVE_SIGMAS[schedule_index],
        sigma_float32_be_hex=PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index],
    )


def _scalar_value(value: Any, *, label: str) -> Any:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise InferenceSigmaStrataError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        return candidate
    except InferenceSigmaStrataError:
        raise
    except Exception as error:
        raise InferenceSigmaStrataError(f"{label} must be scalar") from error


def assert_selected_timestep_sigma(
    *, timestep: Any, sigma: Any, selected: SigmaStratum
) -> None:
    """Certify that a rebuilt training batch uses one selected UniPC pair."""

    if not isinstance(selected, SigmaStratum):
        raise InferenceSigmaStrataError("selected must be a SigmaStratum")
    # Python scalars remain useful in model-free unit tests.  If the caller
    # supplies a tensor, however, keep the same CPU-fp32 scalar convention as
    # the audited UniPC/C2FR clean-field path; bit-equivalent fp64/GPU values
    # are not accepted silently.
    if hasattr(sigma, "dtype") or hasattr(sigma, "device"):
        dtype = str(getattr(sigma, "dtype", ""))
        device = getattr(sigma, "device", None)
        device_type = getattr(device, "type", None)
        if device_type is None and device is not None:
            device_type = str(device).split(":", 1)[0]
        if dtype != "torch.float32" or device_type != "cpu":
            raise InferenceSigmaStrataError(
                "selected training sigma tensor must be torch.float32 on cpu"
            )
    if hasattr(timestep, "dtype") or hasattr(timestep, "device"):
        dtype = str(getattr(timestep, "dtype", ""))
        device = getattr(timestep, "device", None)
        device_type = getattr(device, "type", None)
        if device_type is None and device is not None:
            device_type = str(device).split(":", 1)[0]
        if dtype != "torch.int64" or device_type != "cpu":
            raise InferenceSigmaStrataError(
                "selected training timestep tensor must be torch.int64 on cpu"
            )
    observed_timestep = _scalar_value(timestep, label="timestep")
    try:
        timestep_float = float(observed_timestep)
    except (TypeError, ValueError, OverflowError) as error:
        raise InferenceSigmaStrataError("timestep must be numeric") from error
    if not math.isfinite(timestep_float) or timestep_float != float(selected.timestep):
        raise InferenceSigmaStrataError(
            "training timestep differs from selected inference timestep"
        )
    observed_sigma = _scalar_value(sigma, label="sigma")
    if _float32_hex(observed_sigma, label="sigma") != selected.sigma_float32_be_hex:
        raise InferenceSigmaStrataError(
            "training sigma differs from selected inference sigma"
        )


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping) and name in config:
        return config[name]
    return getattr(config, name, None)


def _require_tensor_vector(
    value: Any,
    *,
    label: str,
    dtype: str,
    device_type: str,
) -> list[Any]:
    observed_dtype = str(getattr(value, "dtype", ""))
    device = getattr(value, "device", None)
    observed_device = getattr(device, "type", None)
    if observed_device is None and device is not None:
        observed_device = str(device).split(":", 1)[0]
    if observed_dtype != dtype or observed_device != device_type:
        raise InferenceSigmaStrataError(
            f"runtime {label} must be {dtype} on {device_type}"
        )
    ndim = getattr(value, "ndim", None)
    if ndim is not None and int(ndim) != 1:
        raise InferenceSigmaStrataError(f"runtime {label} must be one-dimensional")
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        values = candidate.tolist()
    except Exception as error:
        raise InferenceSigmaStrataError(
            f"cannot read runtime {label}"
        ) from error
    if not isinstance(values, list):
        raise InferenceSigmaStrataError(f"runtime {label} must be a vector")
    return values


def audit_runtime_unipc_schedule(
    scheduler: Any, *, initialize: bool = True
) -> dict[str, Any]:
    """Audit the real scheduler against the captured 40-step schedule.

    With ``initialize=True`` this executes the same public call made by
    Bernini inference, ``scheduler.set_timesteps(40)``.  The audit requires
    CPU int64 timesteps, CPU float32 sigmas, exact timestep equality, and exact
    IEEE-754 float32 sigma bits (including terminal zero).
    """

    if scheduler is None:
        raise InferenceSigmaStrataError("runtime UniPC scheduler is required")
    config = getattr(scheduler, "config", None)
    if config is None:
        raise InferenceSigmaStrataError("runtime UniPC scheduler must expose config")
    configured_class = _config_value(config, "_class_name")
    if (
        type(scheduler).__name__ != SCHEDULER_CLASS
        and configured_class != SCHEDULER_CLASS
    ):
        raise InferenceSigmaStrataError(
            f"runtime scheduler must be {SCHEDULER_CLASS}"
        )
    expected_config = {
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "flow_shift": FLOW_SHIFT,
        "prediction_type": "flow_prediction",
        "predict_x0": True,
        "use_flow_sigmas": True,
        "thresholding": False,
        "solver_order": 2,
        "solver_type": "bh2",
        "final_sigmas_type": "zero",
    }
    for name, expected in expected_config.items():
        observed = _config_value(config, name)
        if type(expected) is bool:
            matches = observed is expected
        elif type(expected) is int:
            matches = type(observed) is int and observed == expected
        elif type(expected) is float:
            matches = type(observed) in (int, float) and float(observed) == expected
        else:
            matches = observed == expected
        if not matches:
            raise InferenceSigmaStrataError(
                f"runtime scheduler config {name} differs: "
                f"expected {expected!r}, got {observed!r}"
            )
    if initialize:
        setter = getattr(scheduler, "set_timesteps", None)
        if not callable(setter):
            raise InferenceSigmaStrataError(
                "runtime UniPC scheduler must expose set_timesteps"
            )
        setter(NUM_INFERENCE_STEPS)
    timesteps = _require_tensor_vector(
        getattr(scheduler, "timesteps", None),
        label="timesteps",
        dtype="torch.int64",
        device_type="cpu",
    )
    sigmas = _require_tensor_vector(
        getattr(scheduler, "sigmas", None),
        label="sigmas",
        dtype="torch.float32",
        device_type="cpu",
    )
    if len(timesteps) != NUM_INFERENCE_STEPS:
        raise InferenceSigmaStrataError(
            "runtime UniPC must expose exactly 40 timesteps"
        )
    if len(sigmas) != NUM_INFERENCE_STEPS + 1:
        raise InferenceSigmaStrataError(
            "runtime UniPC must expose 40 positive sigmas plus terminal zero"
        )
    observed_timesteps: list[int] = []
    for index, value in enumerate(timesteps):
        try:
            numeric = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise InferenceSigmaStrataError(
                f"runtime timestep {index} is not integral"
            ) from error
        if numeric != value or numeric != PINNED_TIMESTEPS[index]:
            raise InferenceSigmaStrataError(
                f"runtime timestep differs at schedule index {index}"
            )
        observed_timesteps.append(numeric)
    observed_sigma_hex = tuple(
        _float32_hex(value, label=f"runtime sigma {index}")
        for index, value in enumerate(sigmas)
    )
    expected_sigma_hex = (
        *PINNED_POSITIVE_SIGMA_FLOAT32_HEX,
        TERMINAL_SIGMA_FLOAT32_HEX,
    )
    for index, (observed, expected) in enumerate(
        zip(observed_sigma_hex, expected_sigma_hex)
    ):
        if observed != expected:
            raise InferenceSigmaStrataError(
                f"runtime sigma differs at schedule index {index}: "
                f"expected float32 {expected}, got {observed}"
            )
    return {
        "schedule_sha256": SCHEDULE_SHA256,
        "timesteps": observed_timesteps,
        "positive_sigmas": [float(value) for value in sigmas[:-1]],
        "positive_sigmas_float32_be_hex": list(observed_sigma_hex[:-1]),
        "terminal_sigma": float(sigmas[-1]),
        "terminal_sigma_float32_be_hex": observed_sigma_hex[-1],
    }


def histogram_for_optimizer_range(
    *, start_step: int = 0, stop_step: int
) -> tuple[int, ...]:
    """Return exact per-stratum counts for ``range(start_step, stop_step)``."""

    for name, value in (("start_step", start_step), ("stop_step", stop_step)):
        if type(value) is not int or value < 0:
            raise InferenceSigmaStrataError(
                f"{name} must be a non-negative integer"
            )
    if stop_step < start_step:
        raise InferenceSigmaStrataError("stop_step must not precede start_step")
    counts = [0] * NUM_INFERENCE_STEPS
    for optimizer_step in range(start_step, stop_step):
        counts[optimizer_step % NUM_INFERENCE_STEPS] += 1
    return tuple(counts)


def build_sigma_strata_receipt(
    *, completed_optimizer_steps: int
) -> dict[str, Any]:
    """Build a cumulative, resume-stable receipt fragment for training."""

    if type(completed_optimizer_steps) is not int or completed_optimizer_steps < 0:
        raise InferenceSigmaStrataError(
            "completed_optimizer_steps must be a non-negative integer"
        )
    histogram = histogram_for_optimizer_range(stop_step=completed_optimizer_steps)
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "schedule": {
            **_schedule_digest_payload(),
            "schedule_sha256": SCHEDULE_SHA256,
            "positive_sigmas": list(PINNED_POSITIVE_SIGMAS),
            "terminal_sigma": 0.0,
        },
        "selection": {
            "formula": "schedule_index = absolute_optimizer_step % 40",
            "order": "official_unipc_descending_sigma",
            "rank_invariant": True,
            "resume_cursor": "absolute_optimizer_step",
        },
        "completed_optimizer_steps": completed_optimizer_steps,
        "complete_cycles": completed_optimizer_steps // NUM_INFERENCE_STEPS,
        "partial_cycle_steps": completed_optimizer_steps % NUM_INFERENCE_STEPS,
        "histogram_by_schedule_index": list(histogram),
        "histogram_by_timestep": [
            {"schedule_index": index, "timestep": timestep, "count": histogram[index]}
            for index, timestep in enumerate(PINNED_TIMESTEPS)
        ],
    }
    payload["receipt_digest"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


__all__ = [
    "FLOW_SHIFT",
    "InferenceSigmaStrataError",
    "NUM_INFERENCE_STEPS",
    "NUM_TRAIN_TIMESTEPS",
    "PINNED_POSITIVE_SIGMAS",
    "PINNED_POSITIVE_SIGMA_FLOAT32_HEX",
    "PINNED_TIMESTEPS",
    "RECEIPT_SCHEMA",
    "SCHEDULE_SCHEMA",
    "SCHEDULE_SHA256",
    "SigmaStratum",
    "assert_selected_timestep_sigma",
    "audit_runtime_unipc_schedule",
    "build_sigma_strata_receipt",
    "histogram_for_optimizer_range",
    "select_sigma_stratum",
]
