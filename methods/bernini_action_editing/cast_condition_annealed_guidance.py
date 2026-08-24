#!/usr/bin/env python3
"""Inference-only condition annealing for Bernini native RV2V guidance.

Bernini's native RV2V field is the sum of four independently meaningful
condition axes::

    eps_empty
      + omega_vid * (eps_video - eps_empty)
      + omega_img * (eps_video_image - eps_video)
      + omega_txt * (eps_video_image_text - eps_video_image)

The full-video axis preserves source dynamics but can also copy the old
action.  The independently encoded image-reference axis carries appearance
without a full temporal trajectory, while the text axis carries the requested
action.  CAST therefore tests one preregistered, inference-valid schedule:

* early steps reduce full-video authority and strengthen text;
* middle steps restore the source-video path;
* late steps strengthen source locking and reduce text authority.

The schedule multiplies the *effective* weights supplied by Bernini at each
step.  This preserves the vendor transformer's own late-model ``omega_scale``
switch.  The native-control arm forwards the original keyword values without
even rewriting them.

This module does not score endpoints, select candidates, train parameters, or
claim action-editing success.  It is kept independent from source-set initial
noise so the two interventions can be evaluated causally before any cross.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, MutableMapping, Optional


SCHEMA_VERSION = "bernini-cast-condition-annealed-guidance-v1"
NUM_INFERENCE_STEPS = 40
ARM_NATIVE_FIXED = "native_fixed"
ARM_ACTION_FIRST_SOURCE_LOCK = "action_first_source_lock"
ARM_ORDER = (ARM_NATIVE_FIXED, ARM_ACTION_FIRST_SOURCE_LOCK)
WEIGHT_KEYS = ("cur_omega_vid", "cur_omega_img", "cur_omega_txt")


class CASTConditionGuidanceError(RuntimeError):
    """Raised when the hook could change anything outside the sealed schedule."""


@dataclass(frozen=True)
class GuidanceMultipliers:
    video: float
    image: float
    text: float

    def as_dict(self) -> dict[str, float]:
        return {"video": self.video, "image": self.image, "text": self.text}


@dataclass(frozen=True)
class GuidanceStratum:
    name: str
    first_step: int
    last_step: int
    multipliers: GuidanceMultipliers

    def contains(self, step_index: int) -> bool:
        return self.first_step <= step_index <= self.last_step

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "multipliers": self.multipliers.as_dict(),
        }


# These are deliberately moderate.  The experiment asks whether condition-axis
# ordering exposes native action support; it is not an unrestricted CFG sweep.
ACTION_FIRST_SOURCE_LOCK_STRATA = (
    GuidanceStratum(
        "action_geometry",
        0,
        19,
        GuidanceMultipliers(video=0.50, image=1.00, text=1.50),
    ),
    GuidanceStratum(
        "source_reacquisition",
        20,
        32,
        GuidanceMultipliers(video=0.80, image=1.05, text=1.20),
    ),
    GuidanceStratum(
        "source_lock",
        33,
        39,
        GuidanceMultipliers(video=1.15, image=1.15, text=0.75),
    ),
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _object_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_registry() -> None:
    expected = list(range(NUM_INFERENCE_STEPS))
    actual: list[int] = []
    for stratum in ACTION_FIRST_SOURCE_LOCK_STRATA:
        if (
            not stratum.name
            or type(stratum.first_step) is not int
            or type(stratum.last_step) is not int
            or stratum.first_step > stratum.last_step
        ):
            raise RuntimeError("CAST guidance stratum geometry differs")
        actual.extend(range(stratum.first_step, stratum.last_step + 1))
        values = stratum.multipliers.as_dict().values()
        if any(
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 < value <= 2.0
            for value in values
        ):
            raise RuntimeError("CAST guidance multiplier leaves registered bounds")
    if actual != expected:
        raise RuntimeError("CAST guidance strata do not partition exact40")


_validate_registry()


def schedule_contract() -> dict[str, Any]:
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "arm_order": list(ARM_ORDER),
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "vendor_formula": (
            "eps_empty+omega_vid*(eps_video-eps_empty)+"
            "omega_img*(eps_video_image-eps_video)+"
            "omega_txt*(eps_video_image_text-eps_video_image)"
        ),
        "multiplier_input": "vendor_effective_weight_after_native_omega_scale",
        "native_fixed_exact_keyword_forward": True,
        "action_first_source_lock": [
            value.as_dict() for value in ACTION_FIRST_SOURCE_LOCK_STRATA
        ],
        "source_action_factorization": {
            "full_video_axis": "identity_background_camera_plus_old_motion",
            "image_reference_axis": "appearance_without_full_temporal_trajectory",
            "text_axis": "requested_action_semantics",
        },
        "information_flow": {
            "runtime_inputs": ["source_video", "action_instruction"],
            "target_video": False,
            "t2v_media_or_latent": False,
            "mask_flow_pose_track_trajectory": False,
            "external_initial_noise": False,
        },
        "training_performed": False,
        "optimizer_authorized": False,
        "scientific_action_editing_claim": False,
    }
    return {**unsigned, "contract_digest": _object_sha256(unsigned)}


def stratum_for_step(step_index: int) -> GuidanceStratum:
    if type(step_index) is not int or not 0 <= step_index < NUM_INFERENCE_STEPS:
        raise CASTConditionGuidanceError("step index leaves exact40 schedule")
    for stratum in ACTION_FIRST_SOURCE_LOCK_STRATA:
        if stratum.contains(step_index):
            return stratum
    raise CASTConditionGuidanceError("step index is not covered by the registry")


def scheduled_guidance_kwargs(
    values: Mapping[str, Any], *, arm: str, step_index: int
) -> dict[str, Any]:
    """Return one validated per-step kwargs mapping.

    The native arm returns a shallow copy with identical key/value objects.
    The active arm changes exactly the three scalar CFG weights and rejects a
    missing, boolean, non-finite, or non-positive vendor weight.
    """

    if not isinstance(values, Mapping):
        raise CASTConditionGuidanceError("sample_one_step kwargs must be a mapping")
    if arm not in ARM_ORDER:
        raise CASTConditionGuidanceError("guidance arm is outside the registry")
    if type(step_index) is not int or not 0 <= step_index < NUM_INFERENCE_STEPS:
        raise CASTConditionGuidanceError("step index leaves exact40 schedule")
    result = dict(values)
    for key in WEIGHT_KEYS:
        value = result.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise CASTConditionGuidanceError(f"vendor {key} differs")
    if arm == ARM_NATIVE_FIXED:
        return result
    multipliers = stratum_for_step(step_index).multipliers
    multiplier_by_key = {
        "cur_omega_vid": multipliers.video,
        "cur_omega_img": multipliers.image,
        "cur_omega_txt": multipliers.text,
    }
    for key in WEIGHT_KEYS:
        result[key] = float(result[key]) * multiplier_by_key[key]
    return result


@dataclass(frozen=True)
class GuidanceHookTrace:
    arm: str
    calls: tuple[Mapping[str, Any], ...]
    restored_original_callable: bool

    def receipt(self) -> Mapping[str, Any]:
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "arm": self.arm,
            "call_count": len(self.calls),
            "calls": [dict(value) for value in self.calls],
            "restored_original_callable": self.restored_original_callable,
            "contract_digest": schedule_contract()["contract_digest"],
            "training_performed": False,
            "scientific_action_editing_claim": False,
        }
        return {**unsigned, "trace_digest": _object_sha256(unsigned)}


@contextmanager
def install_guidance_schedule(
    diffusion: Any,
    *,
    arm: str,
    trace_sink: Optional[MutableMapping[str, Any]] = None,
) -> Iterator[None]:
    """Temporarily wrap one real Bernini ``sample_one_step`` method.

    The context fails closed unless the sampler calls the same method exactly
    forty times and the original bound method is restored in ``finally``.
    ``trace_sink`` is populated only after successful restoration/closure.
    """

    if arm not in ARM_ORDER:
        raise CASTConditionGuidanceError("guidance arm is outside the registry")
    original = getattr(diffusion, "sample_one_step", None)
    if not callable(original):
        raise CASTConditionGuidanceError("diffusion lacks callable sample_one_step")
    instance_dictionary = getattr(diffusion, "__dict__", None)
    had_instance_attribute = (
        isinstance(instance_dictionary, dict)
        and "sample_one_step" in instance_dictionary
    )
    original_instance_value = (
        instance_dictionary["sample_one_step"]
        if had_instance_attribute
        else None
    )
    calls: list[Mapping[str, Any]] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        step_index = len(calls)
        if step_index >= NUM_INFERENCE_STEPS:
            raise CASTConditionGuidanceError("sampler exceeded exact40 calls")
        changed = scheduled_guidance_kwargs(
            kwargs, arm=arm, step_index=step_index
        )
        calls.append(
            {
                "step_index": step_index,
                "stratum": (
                    "native_fixed"
                    if arm == ARM_NATIVE_FIXED
                    else stratum_for_step(step_index).name
                ),
                "input": {key: float(kwargs[key]) for key in WEIGHT_KEYS},
                "executed": {key: float(changed[key]) for key in WEIGHT_KEYS},
            }
        )
        return original(*args, **changed)

    setattr(wrapped, "_cast_condition_annealing_hook", True)
    setattr(diffusion, "sample_one_step", wrapped)
    body_error: Optional[BaseException] = None
    try:
        yield
    except BaseException as error:
        body_error = error
        raise
    finally:
        wrapper_unchanged = getattr(diffusion, "sample_one_step", None) is wrapped
        if had_instance_attribute:
            setattr(diffusion, "sample_one_step", original_instance_value)
        else:
            delattr(diffusion, "sample_one_step")
        restored_value = getattr(diffusion, "sample_one_step", None)
        restored = (
            restored_value is original
            or (
                getattr(restored_value, "__self__", None)
                is getattr(original, "__self__", None)
                and getattr(restored_value, "__func__", None)
                is getattr(original, "__func__", None)
            )
        )
        if body_error is None:
            if not wrapper_unchanged or not restored:
                raise CASTConditionGuidanceError(
                    "sample_one_step hook was mutated or not restored"
                )
            if len(calls) != NUM_INFERENCE_STEPS:
                raise CASTConditionGuidanceError(
                    "sampler did not execute exact40 sample_one_step calls"
                )
            trace = GuidanceHookTrace(arm, tuple(calls), restored)
            if trace_sink is not None:
                trace_sink.clear()
                trace_sink.update(trace.receipt())


__all__ = [
    "ACTION_FIRST_SOURCE_LOCK_STRATA",
    "ARM_ACTION_FIRST_SOURCE_LOCK",
    "ARM_NATIVE_FIXED",
    "ARM_ORDER",
    "CASTConditionGuidanceError",
    "GuidanceHookTrace",
    "GuidanceMultipliers",
    "GuidanceStratum",
    "NUM_INFERENCE_STEPS",
    "SCHEMA_VERSION",
    "WEIGHT_KEYS",
    "install_guidance_schedule",
    "schedule_contract",
    "scheduled_guidance_kwargs",
    "stratum_for_step",
]
