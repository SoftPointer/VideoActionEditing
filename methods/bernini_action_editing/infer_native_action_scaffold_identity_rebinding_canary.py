#!/usr/bin/env python3
"""Frozen exact81 Bernini action-scaffold / source-identity rebinding canary.

This runner asks whether Bernini's *native* ``rv2v`` pathway can render a
registered pure-T2V proposal as ordered temporal evidence while independently
encoded RGB frames from the exact source video bind identity.  It performs no
training and does not read a proposal MP4.  Registered proposal tensors are
loaded only through the sealed factor-bank, pre-decode FP32 loader audited by
``infer_native_multivideo_motion_donor_oracle``.

All eight arms use the same source-specific renderer prompt, target seed,
official target Gaussian, native UniPC scheduler, frozen weights and 81-frame
geometry.  The two order arms jointly change native privileged-V membership,
source IDs, and condition order; they are explicitly not a pure role swap.
The wrong-reference arm changes only the independently encoded RGB identity
references: it never reads a paired target, parquet row, or precomputed latent.
Factor-bank semantic labels are registrations, not evidence that the rendered
proposal realizes the labelled action.  Neither one-step nor exact40 output
authorizes an action, identity, or quality claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_native_multivideo_motion_donor_oracle as donor  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "frozen-bernini-native-action-scaffold-identity-rebinding-canary"
SCHEMA_VERSION = "bernini-native-action-scaffold-identity-rebinding-receipt-v3"
FRAME_COUNT = 81
LATENT_SHAPE = (1, 16, 21, 62, 60)
REFERENCE_SHAPE = (1, 16, 1, 62, 60)
REFERENCE_INDICES = (0, 27, 53, 80)
PATCH_TOKENS = 19_530
REFERENCE_PATCH_TOKENS = 930
FPS = 25
HEIGHT = 496
WIDTH = 480
TARGET_SEED = 20_260_810
ALLOWED_STEPS = (1, 40)
ULYSSES_SIZE = 4
NATIVE_INTERPOLATE_SOURCE_IDS = True
NATIVE_MAX_TRAINED_SOURCE_ID = 5
FACTOR_EXECUTION_GROUP = "sp4-a"
CDF_DOG_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
CDF_DOG_WRONG_SOURCE_SHA256 = (
    "da7e3efa6f4fabac1f1c57b9376667366ca2ad43d4710adea5892eb313cc5e7a"
)
DONOR_BRANCHES = ("full_action", "noop", "reverse_action")
RENDERER_BODY = (
    "Locked overhead camera and unchanged composition. The same muscular "
    "tan-and-white pit bull with the same black collar begins seated beside the "
    "same long bone on the same gray concrete surface. The dog deliberately lowers its "
    "head, grips the bone, lifts it upward, raises it clearly off the floor, "
    "and holds it. Preserve the source dog's exact identity, markings, collar, "
    "body proportions, background, lighting, and camera. Treat full-video "
    "conditions only as temporal action evidence and use the source image "
    "references for exact source identity and appearance. Do not copy donor "
    "identity, donor background, donor object appearance, or donor camera."
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RoleRebindingCanaryError(RuntimeError):
    """Raised before ambiguous rebinding evidence can be published."""


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    video_roles: tuple[str, ...]
    donor_branch: Optional[str]
    source_reference_indices: tuple[int, ...]
    reference_video_role: Optional[str]
    privileged_v_role: str
    diagnostic: str


ARM_SPECS = (
    ArmSpec(
        "source-video-source-refs", ("source_video",), None, REFERENCE_INDICES,
        "source_video", "source_video",
        "source_video_plus_source_identity_references_baseline",
    ),
    ArmSpec(
        "action-donor-source-refs", ("registered_full_action_proposal",),
        "full_action", REFERENCE_INDICES, "source_video",
        "registered_full_action_proposal",
        "primary_action_scaffold_plus_source_identity_references",
    ),
    ArmSpec(
        "noop-donor-source-refs", ("registered_noop_proposal",), "noop",
        REFERENCE_INDICES, "source_video", "registered_noop_proposal",
        "registered_noop_scaffold_negative_control",
    ),
    ArmSpec(
        "reverse-donor-source-refs", ("registered_reverse_action_proposal",),
        "reverse_action", REFERENCE_INDICES, "source_video",
        "registered_reverse_action_proposal",
        "registered_reverse_scaffold_negative_control",
    ),
    ArmSpec(
        "action-donor-only", ("registered_full_action_proposal",),
        "full_action", (), None, "registered_full_action_proposal",
        "identity_reference_marginal_control",
    ),
    ArmSpec(
        "source-action-source-refs",
        ("source_video", "registered_full_action_proposal"), "full_action",
        REFERENCE_INDICES, "source_video", "source_video",
        "preservation_privileged_joint_order_arm",
    ),
    ArmSpec(
        "action-source-source-refs",
        ("registered_full_action_proposal", "source_video"), "full_action",
        REFERENCE_INDICES, "source_video", "registered_full_action_proposal",
        "action_privileged_joint_order_arm",
    ),
    ArmSpec(
        "action-donor-wrong-refs", ("registered_full_action_proposal",),
        "full_action", REFERENCE_INDICES, "wrong_source_video",
        "registered_full_action_proposal",
        "matched_wrong_source_identity_reference_causal_control",
    ),
)
ARM_ORDER = tuple(spec.arm_id for spec in ARM_SPECS)
ARM_GROUPS = {
    "group-a": ARM_ORDER[:4],
    "group-b": ARM_ORDER[4:],
}


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RoleRebindingCanaryError(f"receipt is not finite canonical ASCII JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha(value: Any, *, length: int, label: str) -> str:
    text = str(value)
    pattern = _SHA1 if length == 40 else _SHA256
    if pattern.fullmatch(text) is None:
        raise RoleRebindingCanaryError(f"{label} must be lowercase SHA-{1 if length == 40 else 256}")
    return text


def arm_plan(group: Optional[str] = None) -> tuple[ArmSpec, ...]:
    """Return and self-check the immutable eight-arm design."""

    if len(set(ARM_ORDER)) != 8:
        raise RoleRebindingCanaryError("arm order/uniqueness differs")
    by_name = {spec.arm_id: spec for spec in ARM_SPECS}
    for spec in ARM_SPECS:
        if not spec.video_roles or len(spec.video_roles) not in (1, 2):
            raise RoleRebindingCanaryError("each arm must have one or two video conditions")
        if spec.source_reference_indices not in ((), REFERENCE_INDICES):
            raise RoleRebindingCanaryError("source-reference design differs")
        expected_reference_role = None if not spec.source_reference_indices else spec.reference_video_role
        if expected_reference_role not in (None, "source_video", "wrong_source_video"):
            raise RoleRebindingCanaryError("reference-video role differs")
        if bool(spec.source_reference_indices) != (spec.reference_video_role is not None):
            raise RoleRebindingCanaryError("reference-video role/indices mismatch")
        if spec.video_roles[0] != spec.privileged_v_role:
            raise RoleRebindingCanaryError("privileged-V role must be first video")
        if spec.donor_branch is not None and spec.donor_branch not in DONOR_BRANCHES:
            raise RoleRebindingCanaryError("arm requests an unregistered proposal branch")
    if group is None:
        return ARM_SPECS
    if group not in ARM_GROUPS:
        raise RoleRebindingCanaryError("unknown fixed arm group")
    return tuple(by_name[name] for name in ARM_GROUPS[group])


def _float32(value: float) -> float:
    """Match the scalar precision returned by pinned ``torch.linspace``."""

    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _native_source_ids(count: int) -> list[float]:
    """Reproduce pinned Bernini ``_make_sids`` without importing torch."""

    if count < 0:
        raise RoleRebindingCanaryError("native source count cannot be negative")
    if count == 0:
        return []
    if NATIVE_INTERPOLATE_SOURCE_IDS and count > NATIVE_MAX_TRAINED_SOURCE_ID:
        if count == 1:  # pragma: no cover - guarded by count > 5.
            return [1.0]
        span = float(NATIVE_MAX_TRAINED_SOURCE_ID - 1)
        return [
            _float32(1.0 + span * index / (count - 1))
            for index in range(count)
        ]
    return [float(index) for index in range(1, count + 1)]


def condition_source_id_contract(spec: ArmSpec) -> dict[str, Any]:
    video_count = len(spec.video_roles)
    refs = tuple(spec.source_reference_indices)
    vi_source_count = video_count + len(refs)
    vi_source_ids = _native_source_ids(vi_source_count)
    image_only_source_ids = _native_source_ids(len(refs))
    ordered: list[dict[str, Any]] = []
    for index, role in enumerate(spec.video_roles):
        ordered.append({
            "source_id": vi_source_ids[index],
            "patch_combo": "VI",
            "kind": "video",
            "role": role,
            "also_enters_v_combo": index == 0,
        })
    for offset, frame_index in enumerate(refs):
        common = {
            "kind": "independently_encoded_rgb_frame",
            "reference_video_role": spec.reference_video_role,
            "role": f"{spec.reference_video_role}_reference_frame_{frame_index}",
            "frame_index": frame_index,
        }
        # Pinned Bernini patches each image twice: once on the shared VI axis,
        # then once on the independent image-only I axis.  rv2v forwards VI,
        # not I, but both native patch calls still occur and must be audited.
        ordered.append({
            **common,
            "source_id": vi_source_ids[video_count + offset],
            "patch_combo": "VI",
        })
        ordered.append({
            **common,
            "source_id": image_only_source_ids[offset],
            "patch_combo": "I",
        })
    ordered.append({
        "source_id": 0.0,
        "patch_combo": "target_shared_by_all_guidance_branches",
        "kind": "noisy_target",
        "role": "target",
    })
    interpolation_used = (
        vi_source_count > NATIVE_MAX_TRAINED_SOURCE_ID
        or len(refs) > NATIVE_MAX_TRAINED_SOURCE_ID
    )
    patch_ids = [float(row["source_id"]) for row in ordered]
    return {
        "target_source_id": 0.0,
        "native_interpolate_src_id": NATIVE_INTERPOLATE_SOURCE_IDS,
        "native_max_trained_source_id": NATIVE_MAX_TRAINED_SOURCE_ID,
        "vi_source_count": vi_source_count,
        "image_only_source_count": len(refs),
        "vi_video_source_ids": vi_source_ids[:video_count],
        "vi_reference_source_ids": vi_source_ids[video_count:],
        "image_only_reference_source_ids": image_only_source_ids,
        "patch_vae_latent_calls_in_order": ordered,
        "patch_source_id_order_per_step": patch_ids,
        "native_source_id_interpolation_used": interpolation_used,
        "all_patch_source_ids_within_trained_interval_0_through_5": all(
            0.0 <= value <= float(NATIVE_MAX_TRAINED_SOURCE_ID)
            for value in patch_ids
        ),
        "conditioning_source_id_extrapolation_used": False,
    }


def _expected_shared_lengths(spec: ArmSpec) -> list[int]:
    # Native rv2v: none, first-video V, all-video+all-image VI uncond/action.
    vi = (
        PATCH_TOKENS
        + len(spec.video_roles) * PATCH_TOKENS
        + len(spec.source_reference_indices) * REFERENCE_PATCH_TOKENS
    )
    return [PATCH_TOKENS, 2 * PATCH_TOKENS, vi, vi]


@dataclass
class _ActiveAudit:
    completed_steps: int = 0
    patch_source_ids: list[float] = field(default_factory=list)
    shared_lengths: list[int] = field(default_factory=list)
    shared_visual_objects: list[Any] = field(default_factory=list)
    shared_rotary_objects: list[Any] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)


class NativeRoleRebindingConditionAudit:
    """Reversible observation of native video/ref source IDs and rv2v forwards."""

    def __init__(
        self,
        diffusion: Any,
        *,
        spec: ArmSpec,
        video_conditions: list[Any],
        image_references: list[Any],
        expected_steps: int,
        prompt_embeds: Any,
        uncond_prompt_embeds: Any,
    ) -> None:
        self.diffusion = sampler_contract.resolve_diffusion_core(diffusion)
        self.transformer = getattr(self.diffusion, "transformer", None)
        self.scheduler = getattr(self.diffusion, "scheduler", None)
        self.spec = spec
        self.video_conditions = video_conditions
        self.image_references = image_references
        self.expected_steps = int(expected_steps)
        self.prompt_embeds = prompt_embeds
        self.uncond_prompt_embeds = uncond_prompt_embeds
        diffusion_config = getattr(self.diffusion, "config", None)
        self.native_interpolate_src_id = getattr(
            diffusion_config, "interpolate_src_id", True
        )
        self.native_max_trained_src_id = getattr(
            diffusion_config, "max_trained_src_id", 5
        )
        self._original_sample = getattr(self.diffusion, "sample", None)
        self._original_shared = getattr(self.diffusion, "shared_step", None)
        self._original_patch = getattr(self.transformer, "patch_vae_latent", None)
        self._original_scheduler = getattr(self.scheduler, "step", None)
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._active: Optional[_ActiveAudit] = None
        self.sample_calls = 0
        self.restored = False
        self.trace: dict[str, Any] = {}
        if (
            len(video_conditions) != len(spec.video_roles)
            or len(image_references) != len(spec.source_reference_indices)
            or self.expected_steps not in ALLOWED_STEPS
            or self.prompt_embeds is None
            or self.uncond_prompt_embeds is None
            or not all(callable(value) for value in (
                self._original_sample, self._original_shared,
                self._original_patch, self._original_scheduler,
            ))
        ):
            raise RoleRebindingCanaryError("native rebinding audit construction differs")
        if (
            self.native_interpolate_src_id is not NATIVE_INTERPOLATE_SOURCE_IDS
            or self.native_max_trained_src_id != NATIVE_MAX_TRAINED_SOURCE_ID
        ):
            raise RoleRebindingCanaryError("native source-id interpolation config differs")
        if getattr(self.diffusion, "transformer_2", None) is not None:
            raise RoleRebindingCanaryError("canary supports only frozen Bernini 1.3B")
        for owner, name in (
            (self.diffusion, "sample"), (self.diffusion, "shared_step"),
            (self.transformer, "patch_vae_latent"), (self.scheduler, "step"),
        ):
            if name in vars(owner):
                raise RoleRebindingCanaryError(f"refusing stacked observer on {name}")

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had_instance = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous))

    def install(self) -> None:
        if self._patches:
            raise RoleRebindingCanaryError("rebinding observer already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared(*args, **kwargs)

        def patch_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_patch(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, patch_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_native_role_rebinding_read_only_audit", self)
        try:
            self._set_patch(self.transformer, "patch_vae_latent", patch_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous = self._patches.pop()
            try:
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:  # pragma: no cover
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise RoleRebindingCanaryError("failed to restore native observer") from errors[0]

    def _expected_patch_ids(self) -> list[float]:
        return condition_source_id_contract(self.spec)["patch_source_id_order_per_step"]

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.sample_calls:
            raise RoleRebindingCanaryError("observer permits exactly one sample")
        try:
            values = sampler_contract._bind_call(self._original_sample, args, kwargs)
        except Exception as error:
            raise RoleRebindingCanaryError(str(error)) from error
        expected = native.native_sampling_contract(
            "rv2v", steps=self.expected_steps, seed=TARGET_SEED
        )
        for name, wanted in expected.items():
            observed = values.get(name)
            if isinstance(wanted, tuple):
                observed = tuple(observed)
            if observed != wanted:
                raise RoleRebindingCanaryError(f"native sample {name} differs")
        expected_image_argument = self.image_references if self.image_references else None
        if (
            values.get("multi_video_vae_latents") is not self.video_conditions
            or values.get("multi_image_vae_latents") is not expected_image_argument
            or values.get("image_vae_latents") is not None
        ):
            raise RoleRebindingCanaryError("native video/reference list identity differs")
        for actual, wanted in zip(values["multi_video_vae_latents"], self.video_conditions):
            if actual is not wanted:
                raise RoleRebindingCanaryError("native video tensor object/order differs")
        for actual, wanted in zip(
            values.get("multi_image_vae_latents") or (), self.image_references
        ):
            if actual is not wanted:
                raise RoleRebindingCanaryError("native reference tensor object/order differs")
        if (
            values.get("prompt_embeds") is not self.prompt_embeds
            or values.get("uncond_prompt_embeds") is not self.uncond_prompt_embeds
        ):
            raise RoleRebindingCanaryError("native prompt object identity differs")
        state = _ActiveAudit()
        self._active = state
        try:
            result = self._original_sample(*args, **kwargs)
            if (
                state.completed_steps != self.expected_steps
                or state.patch_source_ids
                or state.shared_lengths
                or len(state.step_records) != self.expected_steps
            ):
                raise RoleRebindingCanaryError("native sample ended with incomplete audit")
            self.sample_calls = 1
            self.trace = {
                "sample_calls": 1,
                "step_count": self.expected_steps,
                "scheduler_calls": self.expected_steps,
                "guidance_mode": "rv2v",
                "video_roles_in_list_order": list(self.spec.video_roles),
                "source_reference_indices_in_list_order": list(self.spec.source_reference_indices),
                "source_id_order_per_step": list(state.step_records[0]["source_ids"]),
                "native_expected_source_id_order_per_step": self._expected_patch_ids(),
                "native_interpolate_src_id": self.native_interpolate_src_id,
                "native_max_trained_src_id": self.native_max_trained_src_id,
                "shared_visual_token_lengths_per_step": _expected_shared_lengths(self.spec),
                "step_records": list(state.step_records),
                "step_records_digest": object_sha256(state.step_records),
                "original_callables_received_unchanged_arguments": True,
                "original_return_objects_forwarded": True,
                "observer_modified_numerics": False,
            }
            return result
        finally:
            self._active = None

    def _wrapped_patch(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RoleRebindingCanaryError("patch_vae_latent outside observed sample")
        try:
            values = sampler_contract._bind_call(self._original_patch, args, kwargs)
        except Exception as error:
            raise RoleRebindingCanaryError(str(error)) from error
        expected = self._expected_patch_ids()
        index = len(state.patch_source_ids)
        if index >= len(expected):
            raise RoleRebindingCanaryError("too many patch_vae_latent calls")
        source_id = float(values.get("source_id"))
        if not math.isclose(source_id, expected[index], rel_tol=0.0, abs_tol=1e-6):
            raise RoleRebindingCanaryError("native source-id order differs")
        result = self._original_patch(*args, **kwargs)
        state.patch_source_ids.append(source_id)
        return result

    def _wrapped_shared(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RoleRebindingCanaryError("shared_step outside observed sample")
        try:
            values = sampler_contract._bind_call(self._original_shared, args, kwargs)
        except Exception as error:
            raise RoleRebindingCanaryError(str(error)) from error
        expected = _expected_shared_lengths(self.spec)
        index = len(state.shared_lengths)
        if index >= len(expected):
            raise RoleRebindingCanaryError("too many shared_step calls")
        lengths = values.get("batch_vae_seqlen")
        if not isinstance(lengths, (list, tuple)) or tuple(int(v) for v in lengths) != (expected[index],):
            raise RoleRebindingCanaryError("native shared visual-token length/order differs")
        visual = values.get("noisy_latents")
        rotary = values.get("rotary_embs")
        if visual is None or rotary is None:
            raise RoleRebindingCanaryError("native visual/rotary objects are absent")
        expected_prompts = [
            self.uncond_prompt_embeds, self.uncond_prompt_embeds,
            self.uncond_prompt_embeds, self.prompt_embeds,
        ]
        if values.get("cond_embeds") is not expected_prompts[index]:
            raise RoleRebindingCanaryError("native uncond/action forward order differs")
        if values.get("model_id") != "transformer_1":
            raise RoleRebindingCanaryError("native sampler left pinned 1.3B expert")
        if index == 3 and (
            visual is not state.shared_visual_objects[2]
            or rotary is not state.shared_rotary_objects[2]
        ):
            raise RoleRebindingCanaryError("VI uncond/action visual object differs")
        result = self._original_shared(*args, **kwargs)
        state.shared_lengths.append(expected[index])
        state.shared_visual_objects.append(visual)
        state.shared_rotary_objects.append(rotary)
        return result

    def _wrapped_scheduler(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RoleRebindingCanaryError("scheduler.step outside observed sample")
        if (
            state.patch_source_ids != self._expected_patch_ids()
            or state.shared_lengths != _expected_shared_lengths(self.spec)
        ):
            raise RoleRebindingCanaryError("scheduler arrived before complete native guidance step")
        result = self._original_scheduler(*args, **kwargs)
        state.step_records.append({
            "step_index": state.completed_steps,
            "source_ids": list(state.patch_source_ids),
            "video_roles": list(self.spec.video_roles),
            "source_reference_indices": list(self.spec.source_reference_indices),
            "shared_visual_token_lengths": list(state.shared_lengths),
            "target_source_id": 0,
            "image_condition_count": len(self.image_references),
            "scheduler_original_return_forwarded": True,
        })
        state.completed_steps += 1
        state.patch_source_ids.clear()
        state.shared_lengths.clear()
        state.shared_visual_objects.clear()
        state.shared_rotary_objects.clear()
        return result


def _condition_lists(
    spec: ArmSpec,
    *,
    source: Any,
    donors: Mapping[str, Any],
    source_references: Mapping[int, Any],
    wrong_source_references: Mapping[int, Any],
) -> tuple[list[Any], list[Any]]:
    videos: list[Any] = []
    for role in spec.video_roles:
        if role == "source_video":
            videos.append(source)
        elif spec.donor_branch is not None:
            videos.append(donors[spec.donor_branch])
        else:
            raise RoleRebindingCanaryError("video role lacks bound tensor")
    if spec.reference_video_role is None:
        reference_bank: Mapping[int, Any] = {}
    elif spec.reference_video_role == "source_video":
        reference_bank = source_references
    elif spec.reference_video_role == "wrong_source_video":
        reference_bank = wrong_source_references
    else:  # pragma: no cover - arm_plan validates the immutable design.
        raise RoleRebindingCanaryError("reference-video role lacks bound tensor bank")
    refs = [reference_bank[index] for index in spec.source_reference_indices]
    if len(videos) != len(spec.video_roles) or len(refs) != len(spec.source_reference_indices):
        raise RoleRebindingCanaryError("condition construction lost order")
    return videos, refs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--wrong-source-video", required=True)
    parser.add_argument("--factor-manifest", required=True)
    parser.add_argument("--expected-factor-manifest-file-sha256", required=True)
    parser.add_argument("--factor-bank-receipt", required=True)
    parser.add_argument("--expected-factor-bank-receipt-file-sha256", required=True)
    parser.add_argument("--bank-output-root", required=True)
    parser.add_argument("--factor-execution-group", default=FACTOR_EXECUTION_GROUP)
    parser.add_argument("--arm-group", required=True, choices=tuple(ARM_GROUPS))
    parser.add_argument("--arms", nargs="+", required=True, choices=ARM_ORDER)
    parser.add_argument("--num-inference-steps", type=int, required=True, choices=ALLOWED_STEPS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--expected-bernini-commit", default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=native.legacy.trainer.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=native.legacy.trainer.CHECKPOINT_TREE_SHA256)
    return parser


def _validate_cli(args: argparse.Namespace) -> tuple[ArmSpec, ...]:
    expected = ARM_GROUPS[args.arm_group]
    if tuple(args.arms) != expected:
        raise RoleRebindingCanaryError(f"{args.arm_group} must execute exactly {list(expected)}")
    specs = arm_plan(args.arm_group)
    if args.factor_execution_group != FACTOR_EXECUTION_GROUP:
        raise RoleRebindingCanaryError("all arms must bind the same sp4-a proposal cell")
    for name in ("runtime_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        _require_sha(getattr(args, name), length=40, label=name)
    for name in (
        "runtime_source_archive_sha256", "launcher_source_sha256",
        "expected_factor_manifest_file_sha256",
        "expected_factor_bank_receipt_file_sha256", "expected_checkpoint_tree_sha256",
    ):
        _require_sha(getattr(args, name), length=64, label=name)
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise RoleRebindingCanaryError("Bernini commit differs from pinned release")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise RoleRebindingCanaryError("VeOmni commit differs from pinned release")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise RoleRebindingCanaryError("checkpoint tree differs from pinned release")
    return specs


def _save_reference_latent(
    path: Path,
    value: Any,
    *,
    frame_index: int,
    reference_video_role: str,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    stored = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        path.exists() or path.is_symlink() or path.suffix != ".safetensors"
        or tuple(int(item) for item in stored.shape) != REFERENCE_SHAPE
        or not bool(torch.isfinite(stored).all().item())
    ):
        raise RoleRebindingCanaryError("source reference artifact contract differs")
    if reference_video_role not in ("source_video", "wrong_source_video"):
        raise RoleRebindingCanaryError("reference artifact video role differs")
    stem = "source" if reference_video_role == "source_video" else "wrong_source"
    tensor_key = f"{stem}_reference_latent"
    artifact_role = f"independently_encoded_{stem}_rgb_frame"
    raw_identity = native.value_audit.tensor_identity(
        stored, label=f"{stem}_reference_{frame_index}"
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            {tensor_key: stored}, str(temporary),
            metadata={
                "coordinate": "bernini_normalized_clean_vae_latent",
                "artifact_role": artifact_role,
                "reference_video_role": reference_video_role,
                "frame_index": str(frame_index),
                "temporal_video_latent_slice": "false",
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [tensor_key]:
                raise RoleRebindingCanaryError("source reference tensor key differs")
            restored = opened.get_tensor(tensor_key).contiguous()
            metadata = dict(opened.metadata() or {})
        if not torch.equal(restored, stored):
            raise RoleRebindingCanaryError("source reference artifact round trip differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": donor.file_sha256(path),
        "tensor_key": tensor_key,
        "frame_index": frame_index,
        "shape": list(REFERENCE_SHAPE),
        "stored_dtype": "torch.float32",
        "raw_storage_sha256": raw_identity["raw_storage_sha256"],
        "coordinate": metadata["coordinate"],
        "artifact_role": metadata["artifact_role"],
        "reference_video_role": metadata["reference_video_role"],
        "independent_rgb_frame_vae_encode": True,
        "temporal_video_latent_slice": False,
        "roundtrip_byte_exact_fp32": True,
    }


def _video_raw_sha(role: str, *, source_sha: str, donor_shas: Mapping[str, str], branch: Optional[str]) -> str:
    if role == "source_video":
        return source_sha
    if branch is None:
        raise RoleRebindingCanaryError("proposal role lacks branch")
    return donor_shas[branch]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    specs = _validate_cli(args)
    output_dir = donor._fresh_output_directory(args.output_dir)

    manifest, manifest_path, manifest_file_sha = donor._load_json(
        args.factor_manifest, label="factor manifest"
    )
    if manifest_file_sha != args.expected_factor_manifest_file_sha256:
        raise RoleRebindingCanaryError("factor manifest file SHA-256 differs")
    bank_receipt, bank_receipt_path, bank_receipt_file_sha = donor._load_json(
        args.factor_bank_receipt, label="factor bank receipt"
    )
    if bank_receipt_file_sha != args.expected_factor_bank_receipt_file_sha256:
        raise RoleRebindingCanaryError("factor bank receipt file SHA-256 differs")
    try:
        bound = donor.bind_registered_donors(
            manifest=manifest, bank_receipt=bank_receipt,
            execution_group=FACTOR_EXECUTION_GROUP,
        )
    except Exception as error:
        raise RoleRebindingCanaryError(str(error)) from error
    if bank_receipt.get("manifest_file_sha256") != manifest_file_sha:
        raise RoleRebindingCanaryError("bank receipt manifest binding differs")
    bank_root = donor._canonical_root(args.bank_output_root, label="bank output root")
    loaded_donors_cpu: dict[str, Any] = {}
    donor_provenance: dict[str, Any] = {}
    for branch in DONOR_BRANCHES:
        try:
            tensor, provenance = donor.load_registered_clean_donor(
                row=bound["donor_rows"][branch], bank_root=bank_root
            )
        except Exception as error:
            raise RoleRebindingCanaryError(str(error)) from error
        loaded_donors_cpu[branch] = tensor
        donor_provenance[branch] = provenance
    donor_raw_cpu = {
        branch: row["clean_latent_raw_storage_sha256"]
        for branch, row in donor_provenance.items()
    }
    if len(set(donor_raw_cpu.values())) != len(DONOR_BRANCHES):
        raise RoleRebindingCanaryError("registered proposal controls alias one tensor")

    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise RoleRebindingCanaryError("source video must be absolute and non-symlink")
    source_path = source_requested.resolve(strict=True)
    source_contract = bound["manifest"].get("source_geometry_video", {})
    if (
        source_path != source_requested or not source_path.is_file()
        or source_contract.get("sha256") != CDF_DOG_SOURCE_SHA256
        or donor.file_sha256(source_path) != CDF_DOG_SOURCE_SHA256
    ):
        raise RoleRebindingCanaryError("source is not the pinned CDF dog exact81 video")
    wrong_source_requested = Path(args.wrong_source_video).expanduser()
    if not wrong_source_requested.is_absolute() or wrong_source_requested.is_symlink():
        raise RoleRebindingCanaryError("wrong-source video must be absolute and non-symlink")
    wrong_source_path = wrong_source_requested.resolve(strict=True)
    if (
        wrong_source_path != wrong_source_requested
        or not wrong_source_path.is_file()
        or wrong_source_path == source_path
        or donor.file_sha256(wrong_source_path) != CDF_DOG_WRONG_SOURCE_SHA256
    ):
        raise RoleRebindingCanaryError("wrong source is not the pinned distinct exact81 video")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root, args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise RoleRebindingCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise RoleRebindingCanaryError("Bernini attention heads do not divide Ulysses4")
    inference_file_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.transformer_wan as transformer_wan
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != native.legacy.MV2V_SYSTEM_PROMPT:
        raise RoleRebindingCanaryError("runtime Bernini mv2v system prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise RoleRebindingCanaryError("runtime Bernini negative prompt differs")
    distributed = native.legacy.inference_distributed_contract()
    if distributed.world_size != ULYSSES_SIZE or distributed.ulysses_size != ULYSSES_SIZE:
        raise RoleRebindingCanaryError("runtime requires exact WORLD4/Ulysses4")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RoleRebindingCanaryError("runtime requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl", timeout=timedelta(minutes=240), rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise RoleRebindingCanaryError(f"checkpoint validation failed: {checkpoint_rows[0]}")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = (
        native.source_audit.prepare_hashed_source_snapshot(source_path)
    )
    if source_sha != CDF_DOG_SOURCE_SHA256:
        raise RoleRebindingCanaryError("source snapshot SHA-256 differs")
    wrong_source_tensor, wrong_source_metadata, wrong_source_sha = (
        native.source_audit.prepare_hashed_source_snapshot(wrong_source_path)
    )
    if wrong_source_sha != CDF_DOG_WRONG_SOURCE_SHA256:
        raise RoleRebindingCanaryError("wrong-source snapshot SHA-256 differs")
    if (
        source_metadata.get("frame_count") != FRAME_COUNT
        or wrong_source_metadata.get("frame_count") != FRAME_COUNT
        or tuple(source_metadata["source_derived_bucket_hw"]) != (HEIGHT, WIDTH)
        or tuple(wrong_source_metadata["source_derived_bucket_hw"]) != (HEIGHT, WIDTH)
    ):
        raise RoleRebindingCanaryError("source bucket geometry differs")

    full_prompt = native.legacy.build_training_prompt(RENDERER_BODY, prompt_cleaner=prompt_clean)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    target_ids, target_mask = native.legacy._tokenize_training_prompt(tokenizer, full_prompt)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except Exception as error:
        raise RoleRebindingCanaryError(str(error)) from error
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise RoleRebindingCanaryError("renderer is not pinned UniPC shift5")
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = native.source_audit.model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    source_pixels = source_tensor.to(device=device, dtype=torch.float32)
    wrong_source_pixels = wrong_source_tensor.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        source_latent = _vae_encode(vae, source_pixels).contiguous()
        source_references = {
            index: _vae_encode(
                vae, source_pixels[:, :, index:index + 1, :, :].contiguous()
            ).contiguous()
            for index in REFERENCE_INDICES
        }
        wrong_source_references = {
            index: _vae_encode(
                vae,
                wrong_source_pixels[:, :, index:index + 1, :, :].contiguous(),
            ).contiguous()
            for index in REFERENCE_INDICES
        }
    if tuple(int(item) for item in source_latent.shape) != LATENT_SHAPE:
        raise RoleRebindingCanaryError("source video latent geometry differs")
    if any(
        tuple(int(item) for item in value.shape) != REFERENCE_SHAPE
        for bank in (source_references, wrong_source_references)
        for value in bank.values()
    ):
        raise RoleRebindingCanaryError("independent source reference latent geometry differs")
    source_broadcast = native._broadcast_condition_from_rank_zero(
        source_latent, label="cdf_source_video", world_size=ULYSSES_SIZE
    )
    source_reference_broadcasts = {
        str(index): native._broadcast_condition_from_rank_zero(
            value, label=f"source_reference_{index}", world_size=ULYSSES_SIZE
        )
        for index, value in source_references.items()
    }
    wrong_source_reference_broadcasts = {
        str(index): native._broadcast_condition_from_rank_zero(
            value, label=f"wrong_source_reference_{index}", world_size=ULYSSES_SIZE
        )
        for index, value in wrong_source_references.items()
    }
    source_identity = native._all_rank_tensor_identity(
        source_latent, label="cdf_source_video", world_size=ULYSSES_SIZE
    )
    source_reference_identities = {
        str(index): native._all_rank_tensor_identity(
            value, label=f"source_reference_{index}", world_size=ULYSSES_SIZE
        )
        for index, value in source_references.items()
    }
    wrong_source_reference_identities = {
        str(index): native._all_rank_tensor_identity(
            value, label=f"wrong_source_reference_{index}", world_size=ULYSSES_SIZE
        )
        for index, value in wrong_source_references.items()
    }
    source_raw_sha = source_identity["identity"]["raw_storage_sha256"]
    source_ref_raw_shas = {
        row["identity"]["raw_storage_sha256"]
        for row in source_reference_identities.values()
    }
    wrong_source_ref_raw_shas = {
        row["identity"]["raw_storage_sha256"]
        for row in wrong_source_reference_identities.values()
    }
    if (
        len(source_ref_raw_shas) != len(REFERENCE_INDICES)
        or len(wrong_source_ref_raw_shas) != len(REFERENCE_INDICES)
        or source_ref_raw_shas.intersection(wrong_source_ref_raw_shas)
    ):
        raise RoleRebindingCanaryError("correct/wrong reference identity banks alias")

    proposals: dict[str, Any] = {}
    proposal_broadcasts: dict[str, Any] = {}
    proposal_identities: dict[str, Any] = {}
    for branch in DONOR_BRANCHES:
        value = loaded_donors_cpu[branch].to(device=device, dtype=torch.float32).contiguous()
        proposal_broadcasts[branch] = native._broadcast_condition_from_rank_zero(
            value, label=f"registered_{branch}_proposal", world_size=ULYSSES_SIZE
        )
        proposal_identities[branch] = native._all_rank_tensor_identity(
            value, label=f"registered_{branch}_proposal", world_size=ULYSSES_SIZE
        )
        if proposal_identities[branch]["identity"]["raw_storage_sha256"] != donor_raw_cpu[branch]:
            raise RoleRebindingCanaryError("GPU proposal differs from registered FP32 tensor")
        proposals[branch] = value
    if source_raw_sha in donor_raw_cpu.values():
        raise RoleRebindingCanaryError("source condition aliases a proposal tensor")

    vae.to("cpu")
    del (
        source_tensor, source_pixels, wrong_source_tensor, wrong_source_pixels,
        loaded_donors_cpu,
    )
    torch.cuda.empty_cache()
    model.to(device)
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        prompt_embeds = model.encode_prompt(target_ids.to(device), target_mask.to(device))
        uncond_embeds = model.encode_prompt(negative_ids.to(device), negative_mask.to(device))
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    try:
        wan_source_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        sampler_contract._validate_scheduler_contract(
            diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
        )
    except Exception as error:
        raise RoleRebindingCanaryError(str(error)) from error
    transformer_path = Path(transformer_wan.__file__).resolve()
    transformer_sha = donor.file_sha256(transformer_path)
    if transformer_sha != donor.PINNED_TRANSFORMER_WAN_SHA256:
        raise RoleRebindingCanaryError("transformer_wan.py differs from audited bytes")

    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    initial_noise_rank_identities: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    conditioning: dict[str, Any] = {}
    donor_raw_gpu = {
        branch: proposal_identities[branch]["identity"]["raw_storage_sha256"]
        for branch in DONOR_BRANCHES
    }
    with torch.inference_mode():
        for spec in specs:
            videos, refs = _condition_lists(
                spec,
                source=source_latent,
                donors=proposals,
                source_references=source_references,
                wrong_source_references=wrong_source_references,
            )
            video_raw_shas = [
                _video_raw_sha(
                    role, source_sha=source_raw_sha, donor_shas=donor_raw_gpu,
                    branch=spec.donor_branch,
                )
                for role in spec.video_roles
            ]
            selected_reference_identities = (
                wrong_source_reference_identities
                if spec.reference_video_role == "wrong_source_video"
                else source_reference_identities
            )
            reference_raw_shas = [
                selected_reference_identities[str(index)]["identity"]["raw_storage_sha256"]
                for index in spec.source_reference_indices
            ]
            conditioning[spec.arm_id] = {
                "video_roles_in_order": list(spec.video_roles),
                "video_latent_raw_storage_sha256_in_order": video_raw_shas,
                "source_reference_indices_in_order": list(spec.source_reference_indices),
                "source_reference_raw_storage_sha256_in_order": reference_raw_shas,
                "reference_video_role": spec.reference_video_role,
                "source_ids": condition_source_id_contract(spec),
                "privileged_v_role": spec.privileged_v_role,
                "first_video_alone_enters_v": True,
                "all_videos_and_source_refs_enter_vi": True,
                "reference_encoding": "independent_rgb_frame_to_wan_vae_[1,16,1,62,60]",
                "reference_from_temporal_video_latent_slice": False,
            }
            audit = NativeRoleRebindingConditionAudit(
                diffusion, spec=spec, video_conditions=videos,
                image_references=refs, expected_steps=args.num_inference_steps,
                prompt_embeds=prompt_embeds, uncond_prompt_embeds=uncond_embeds,
            )
            audit.install()
            sample_kwargs = {
                "prompt_embeds": prompt_embeds,
                "uncond_prompt_embeds": uncond_embeds,
                "image_vae_latents": None,
                "multi_video_vae_latents": videos,
                "multi_image_vae_latents": refs if refs else None,
                "width": WIDTH,
                "height": HEIGHT,
                "device": device,
                **native.native_sampling_contract(
                    "rv2v", steps=args.num_inference_steps, seed=TARGET_SEED
                ),
            }
            try:
                result, capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda kw=sample_kwargs: diffusion.sample(**kw),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=LATENT_SHAPE,
                    expected_device=device,
                    expected_seed=TARGET_SEED,
                )
            finally:
                audit.restore()
            if (
                not isinstance(result, torch.Tensor) or result.device != device
                or result.dtype != torch.float32 or result.requires_grad
                or result.grad_fn is not None or not result.is_contiguous()
                or tuple(int(item) for item in result.shape) != LATENT_SHAPE
                or not bool(torch.isfinite(result).all().item())
            ):
                raise RoleRebindingCanaryError("native sampler return contract differs")
            generated_cpu = result.detach().to(device="cpu").contiguous()
            generated[spec.arm_id] = generated_cpu
            generated_identities[spec.arm_id] = native._all_rank_tensor_identity(
                generated_cpu, label=f"generated_{spec.arm_id}", world_size=ULYSSES_SIZE
            )
            initial_noise[spec.arm_id] = capture
            initial_noise_rank_identities[spec.arm_id] = native._all_rank_tensor_identity(
                capture.tensor, label=f"official_initial_gaussian_{spec.arm_id}",
                world_size=ULYSSES_SIZE,
            )
            audits[spec.arm_id] = dict(audit.trace)

    noise_hashes = {capture.raw_value_sha256 for capture in initial_noise.values()}
    if len(noise_hashes) != 1:
        raise RoleRebindingCanaryError("group arms did not start from byte-identical Gaussian")
    freeze_after = native.source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before or any(parameter.requires_grad for parameter in model.parameters()):
        raise RoleRebindingCanaryError("frozen model certificate changed")
    model.to("cpu")
    torch.cuda.empty_cache()

    after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            after_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            after_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(after_rows, src=0)
    if not isinstance(after_rows[0], Mapping) or after_rows[0].get("identity") != checkpoint_identity:
        raise RoleRebindingCanaryError("checkpoint content changed during runtime")

    local_evidence = {
        "rank": distributed.rank,
        "audits_digest": object_sha256(audits),
        "generated_digest": object_sha256(generated_identities),
        "noise_raw_sha256": next(iter(noise_hashes)),
        "freeze_digest": object_sha256(freeze_after),
    }
    gathered: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(gathered, local_evidence)
    if sorted(row.get("rank") for row in gathered if isinstance(row, Mapping)) != [0, 1, 2, 3]:
        raise RoleRebindingCanaryError("WORLD4 rank evidence closure differs")
    for field_name in ("audits_digest", "generated_digest", "noise_raw_sha256", "freeze_digest"):
        if len({row.get(field_name) for row in gathered if isinstance(row, Mapping)}) != 1:
            raise RoleRebindingCanaryError(f"WORLD4 ranks disagree on {field_name}")

    if distributed.rank == 0:
        artifact_dir = donor._output_staging_directory(output_dir)
        noise_artifacts = {
            spec.arm_id: native._save_initial_noise_atomically(
                artifact_dir / f"{spec.arm_id}.official-initial-gaussian.safetensors",
                initial_noise[spec.arm_id],
                all_rank_identity=initial_noise_rank_identities[spec.arm_id],
            )
            for spec in specs
        }
        source_artifact = native._save_normalized_clean_latent_atomically(
            artifact_dir / "source.normalized-clean-latent.safetensors",
            source_latent, artifact_role="source_video_condition",
        )
        reference_artifacts = {
            str(index): _save_reference_latent(
                artifact_dir / f"source-reference-{index}.safetensors",
                source_references[index], frame_index=index,
                reference_video_role="source_video",
            )
            for index in REFERENCE_INDICES
        }
        wrong_source_reference_artifacts = {
            str(index): _save_reference_latent(
                artifact_dir / f"wrong-source-reference-{index}.safetensors",
                wrong_source_references[index], frame_index=index,
                reference_video_role="wrong_source_video",
            )
            for index in REFERENCE_INDICES
        }
        outputs = donor._save_outputs(
            output_dir=artifact_dir, generated=generated, vae=vae, device=device,
            save_output_fn=save_output, steps=args.num_inference_steps,
        )
        arm_receipts = {
            spec.arm_id: {
                **asdict(spec),
                "guidance_mode": "rv2v",
                "conditioning": conditioning[spec.arm_id],
                "sampling": {
                    **native.native_sampling_contract(
                        "rv2v", steps=args.num_inference_steps, seed=TARGET_SEED
                    ),
                    "target_initialization": native.TARGET_INITIALIZATION,
                    "same_target_gaussian_across_all_eight_arms": True,
                    "target_mixed_with_source_or_proposal": False,
                },
                "audit": audits[spec.arm_id],
                "factor_label_independently_verified": False,
            }
            for spec in specs
        }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "stage": "engineering_oom_callpath_canary" if args.num_inference_steps == 1 else "matched_exact40_qualitative_causal_pilot",
            "arm_group": args.arm_group,
            "arms_in_execution_order": [spec.arm_id for spec in specs],
            "fixed_global_arm_order": list(ARM_ORDER),
            "runtime_source": {
                "revision": args.runtime_source_revision,
                "archive_sha256": args.runtime_source_archive_sha256,
                "launcher_sha256": args.launcher_source_sha256,
            },
            "pinned_sources": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "wan_diffusion_path": str(Path(wan_diffusion.__file__).resolve()),
                "wan_diffusion_sha256": wan_source_sha,
                "transformer_wan_path": str(transformer_path),
                "transformer_wan_sha256": transformer_sha,
                "bernini_inference_files": inference_file_hashes,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_before_and_after": checkpoint_identity,
                "unchanged": True,
            },
            "factor_bank": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": manifest_file_sha,
                "manifest_digest": bound["manifest"]["manifest_digest"],
                "bank_receipt_path": str(bank_receipt_path),
                "bank_receipt_file_sha256": bank_receipt_file_sha,
                "bank_receipt_digest": bound["bank_receipt_digest"],
                "bank_root": str(bank_root),
                "fixed_execution_group": FACTOR_EXECUTION_GROUP,
                "proposal_cell": bound["cell"],
                "proposals": donor_provenance,
                "all_proposals_same_registered_cell": True,
                "predecode_fp32_latents_only": True,
                "proposal_mp4_consumed": False,
                "semantic_labels_independently_verified": False,
            },
            "source": {
                "video_path": str(source_path),
                "video_sha256": source_sha,
                "metadata": source_metadata,
                "full_video_identity": source_identity,
                "full_video_rank_zero_broadcast": source_broadcast,
                "normalized_clean_latent_artifact": source_artifact,
                "reference_indices": list(REFERENCE_INDICES),
                "reference_identities": source_reference_identities,
                "reference_rank_zero_broadcasts": source_reference_broadcasts,
                "reference_artifacts": reference_artifacts,
                "references_independently_vae_encoded_from_rgb": True,
                "references_sliced_from_full_video_latent": False,
            },
            "wrong_source": {
                "video_path": str(wrong_source_path),
                "video_sha256": wrong_source_sha,
                "metadata": wrong_source_metadata,
                "reference_indices": list(REFERENCE_INDICES),
                "reference_identities": wrong_source_reference_identities,
                "reference_rank_zero_broadcasts": wrong_source_reference_broadcasts,
                "reference_artifacts": wrong_source_reference_artifacts,
                "references_independently_vae_encoded_from_rgb": True,
                "full_video_vae_encode_performed": False,
                "references_sliced_from_full_video_latent": False,
                "source_video_only": True,
                "paired_target_accessed": False,
                "paired_parquet_accessed": False,
                "precomputed_latent_accessed": False,
            },
            "prompt": {
                "renderer_body_utf8_sha256": hashlib.sha256(RENDERER_BODY.encode("utf-8")).hexdigest(),
                "full_prompt_utf8_sha256": hashlib.sha256(full_prompt.encode("utf-8")).hexdigest(),
                "mv2v_system_prompt_sha256": hashlib.sha256(native.legacy.MV2V_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "same_prompt_all_eight_arms": True,
                "source_specific_action_body": True,
                "factor_bank_dual_dog_prompt_reused": False,
                "full_video_conditions_declared_temporal_evidence_only": True,
                "source_image_refs_declared_identity_evidence": True,
                "donor_identity_background_camera_copy_forbidden": True,
            },
            "matched_target": {
                "seed": TARGET_SEED,
                "frame_count": FRAME_COUNT,
                "latent_shape": list(LATENT_SHAPE),
                "height": HEIGHT,
                "width": WIDTH,
                "fps": FPS,
                "num_inference_steps": args.num_inference_steps,
                "same_target_gaussian_all_arms_in_group": True,
                "target_gaussian_raw_storage_sha256": next(iter(noise_hashes)),
                "cross_group_equality_must_be_sealed_by_compute_launcher": True,
                "external_target_or_target_latent": False,
            },
            "arms": arm_receipts,
            "proposal_condition_broadcasts": proposal_broadcasts,
            "proposal_condition_all_rank_identities": proposal_identities,
            "initial_noise_artifacts": noise_artifacts,
            "generated_identities": generated_identities,
            "outputs": outputs,
            "frozen_model": freeze_after,
            "world4_evidence": gathered,
            "runtime_versions": {
                "torch": torch.__version__, "torch_hip": str(torch.version.hip),
                "transformers": transformers_version, "diffusers": diffusers_version,
            },
            "interpretation": {
                "training_performed": False,
                "optimizer": None,
                "backward": False,
                "model_weights_written": False,
                "native_rv2v_only": True,
                "initial_gaussian_observer_only": True,
                "initial_gaussian_observer_changed_return_object": False,
                "custom_noise_or_sampler": False,
                "target_video": False,
                "mask": False,
                "flow": False,
                "pose": False,
                "track": False,
                "trajectory": False,
                "optimization": False,
                "proposal_mp4_read": False,
                "factor_label_proves_realized_motion": False,
                "action_success_evaluated": False,
                "identity_preservation_evaluated": False,
                "quality_claim": False,
                "scientific_claim_authorized": False,
                "order_swap_is_pure_role_swap": False,
                "order_swap_jointly_changes_privileged_v_source_ids_and_order": True,
                "two_video_four_ref_arms_use_native_source_id_interpolation_to_1_through_5": True,
                "conditioning_source_id_extrapolation_used": False,
                "wrong_reference_arm_uses_discrete_vi_source_ids_1_through_5": True,
                "wrong_source_target_parquet_or_latent_read": False,
                "one_step_stage_is_engineering_only": args.num_inference_steps == 1,
            },
        }
        receipt = donor._rebase_artifact_paths(receipt, old_root=artifact_dir, new_root=output_dir)
        receipt["receipt_digest"] = object_sha256(receipt)
        donor._write_receipt(artifact_dir / "receipt.json", receipt)
        donor._commit_output_transaction(staging=artifact_dir, final=output_dir)
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del (
        source_latent, source_references, wrong_source_references,
        proposals, generated, initial_noise,
    )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_STEPS", "ARM_GROUPS", "ARM_ORDER", "ARM_SPECS",
    "CDF_DOG_SOURCE_SHA256", "CDF_DOG_WRONG_SOURCE_SHA256",
    "DONOR_BRANCHES", "FACTOR_EXECUTION_GROUP",
    "FRAME_COUNT", "LATENT_SHAPE", "METHOD",
    "NativeRoleRebindingConditionAudit", "REFERENCE_INDICES",
    "REFERENCE_SHAPE", "RENDERER_BODY", "RoleRebindingCanaryError",
    "SCHEMA_VERSION", "TARGET_SEED", "arm_plan",
    "condition_source_id_contract", "main",
]
