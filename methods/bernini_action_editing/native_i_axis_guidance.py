#!/usr/bin/env python3
"""Frozen native-I-axis guidance hook for Bernini-R 1.3B ``rv2v``.

The pinned Bernini sampler already constructs four visual packs on every
RV2V denoising step: ``none``, ``V``, ``I`` and ``VI``.  Its renderer forwards
only ``none/V/VI_uncond/VI_cond``.  This module reuses the *same-step* native
``I`` pack for one additional unconditional transformer call and changes only
the velocity passed to the original UniPC scheduler:

``vG = vN + 4.5*g*((vI-v0) - (vVIu-vV))``.

``g`` is exactly ``0.25`` at exact40 indices 33--37 and zero elsewhere.  At a
zero-gate coordinate the original scheduler call and original model-output
object are forwarded unchanged; indices 38 and 39 are therefore exact native
parity controls.  There is no APG, cache, source K/V replay, mask, pose, flow,
optimizer or trainable parameter in this module.

The hook is installed only on one already-loaded ``GEN_Wanx22`` instance.  It
does not edit vendor source and restores every instance attribute in reverse
order.  Runtime checks bind the observed native patch order, four official
branch calls, shared target geometry, guidance formula and one original UniPC
call per step before an active correction is allowed to execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence

import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-native-i-axis-guidance-hook-v1"
METHOD = "frozen-bernini-native-i-axis-late-gated-canary"
PINNED_BERNINI_COMMIT = sampler_contract.PINNED_BERNINI_COMMIT
PINNED_WAN_DIFFUSION_SHA256 = sampler_contract.PINNED_WAN_DIFFUSION_SHA256

FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
REFERENCE_COUNT = 4
OMEGA_VIDEO = 1.25
OMEGA_IMAGE = 4.5
OMEGA_TEXT = 4.0
ACTIVE_STEP_INDICES = (33, 34, 35, 36, 37)
FINAL_NATIVE_PARITY_INDICES = (38, 39)
ACTIVE_GATE = 0.25

ARM_ORDER = ("N-C", "N-W", "G-C", "G-W", "G-P", "G-D", "G-S")
NATIVE_ARMS = frozenset(("N-C", "N-W"))
GATED_ARMS = frozenset(("G-C", "G-W", "G-P", "G-D", "G-S"))
CORRECT_REFERENCE_INDICES = (0, 27, 53, 80)
PERMUTED_REFERENCE_INDICES = (27, 53, 80, 0)
PHASE_SHIFT_REFERENCE_INDICES = (10, 30, 50, 70)


class NativeIAxisGuidanceError(RuntimeError):
    """Raised before an unaudited field reaches the original UniPC step."""


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
        raise NativeIAxisGuidanceError(
            f"value is not finite canonical ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sigma_gate(step_index: int) -> float:
    """Return the only registered exact40 native-I gate."""

    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise NativeIAxisGuidanceError("step index must be an integer")
    if not 0 <= step_index < NUM_INFERENCE_STEPS:
        raise NativeIAxisGuidanceError("step index must lie in exact40")
    return ACTIVE_GATE if step_index in ACTIVE_STEP_INDICES else 0.0


def native_rv2v_velocity(v0: Any, v_v: Any, v_vi_u: Any, v_vi_c: Any) -> Any:
    """Pinned raw-velocity RV2V combination; deliberately no APG."""

    return (
        v0
        + OMEGA_VIDEO * (v_v - v0)
        + OMEGA_IMAGE * (v_vi_u - v_v)
        + OMEGA_TEXT * (v_vi_c - v_vi_u)
    )


def gated_native_i_velocity(
    v0: Any,
    v_v: Any,
    v_i: Any,
    v_vi_u: Any,
    v_vi_c: Any,
    *,
    gate: float,
) -> Any:
    """Apply the registered native-I replacement residual in velocity space."""

    numeric = float(gate)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise NativeIAxisGuidanceError("I-axis gate must be finite in [0,1]")
    native = native_rv2v_velocity(v0, v_v, v_vi_u, v_vi_c)
    return native + OMEGA_IMAGE * numeric * (
        (v_i - v0) - (v_vi_u - v_v)
    )


def arm_reference_contract(arm: str) -> Mapping[str, Any]:
    """Return one immutable renderer/reference intervention.

    ``G-P`` is a reference-list/source-id-slot sensitivity control, not a
    chronological-motion shuffle.  The operator only reorders the same four
    tensor objects and must preserve their byte-exact multiset.
    """

    if arm not in ARM_ORDER:
        raise NativeIAxisGuidanceError(f"unknown native-I canary arm: {arm!r}")
    gated = arm in GATED_ARMS
    if arm in ("N-C", "G-C"):
        role, indices, control = "correct", CORRECT_REFERENCE_INDICES, "canonical"
    elif arm in ("N-W", "G-W"):
        role, indices, control = "wrong", CORRECT_REFERENCE_INDICES, "weak_wrong_source"
    elif arm == "G-P":
        role, indices, control = (
            "correct",
            PERMUTED_REFERENCE_INDICES,
            "byte_exact_same_multiset_reference_list_permutation_sensitivity",
        )
    elif arm == "G-D":
        role, indices, control = "none", (), "reference_drop_degenerate_i_axis"
    else:
        role, indices, control = (
            "correct",
            PHASE_SHIFT_REFERENCE_INDICES,
            "same_source_phase_shift_pose_leakage_control",
        )
    return {
        "arm": arm,
        "renderer": "native_rv2v" if not gated else "native_rv2v_plus_i_axis",
        "gated": gated,
        "reference_role": role,
        "reference_indices_in_list_order": list(indices),
        "reference_count": len(indices),
        "control": control,
        "chronological_shuffle_claimed": False,
        "same_full_correct_source_video": True,
    }


def arm_plan() -> tuple[Mapping[str, Any], ...]:
    rows = tuple(arm_reference_contract(arm) for arm in ARM_ORDER)
    if (
        tuple(row["arm"] for row in rows) != ARM_ORDER
        or {row["arm"] for row in rows if row["gated"]} != GATED_ARMS
        or {row["arm"] for row in rows if not row["gated"]} != NATIVE_ARMS
    ):
        raise NativeIAxisGuidanceError("native-I arm registry differs")
    return rows


def permute_reference_objects(
    canonical_references: Sequence[Any], permutation: Sequence[int] = (1, 2, 3, 0)
) -> tuple[Any, ...]:
    """Reorder, but never transform or copy, four reference tensor objects."""

    refs = tuple(canonical_references)
    order = tuple(permutation)
    if len(refs) != REFERENCE_COUNT or sorted(order) != list(range(REFERENCE_COUNT)):
        raise NativeIAxisGuidanceError("reference permutation must cover four slots once")
    result = tuple(refs[index] for index in order)
    if sorted(id(value) for value in result) != sorted(id(value) for value in refs):
        raise NativeIAxisGuidanceError("reference permutation changed the object multiset")
    return result


def hook_contract() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
        "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
        "frame_count": FRAME_COUNT,
        "latent_phases": LATENT_PHASES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_mode": "rv2v",
        "native_formula": (
            "v0+1.25*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
        ),
        "gated_formula": (
            "vG=vN+4.5*g*((vI-v0)-(vVIu-vV))"
        ),
        "coordinate": "raw_velocity_before_original_unipc_step",
        "apg": False,
        "gate": {
            "active_step_indices": list(ACTIVE_STEP_INDICES),
            "active_gate_hex": float(ACTIVE_GATE).hex(),
            "all_other_indices": 0.0,
            "final_native_parity_indices": list(FINAL_NATIVE_PARITY_INDICES),
        },
        "active_branch_forward_order": [
            "none_uncond", "V_uncond", "VI_uncond", "VI_cond", "I_uncond"
        ],
        "native_branch_forward_order": [
            "none_uncond", "V_uncond", "VI_uncond", "VI_cond"
        ],
        "i_pack": "four_independently_encoded_T1_refs_plus_same_noisy_target",
        "cache": False,
        "cross_trajectory_state_transport": False,
        "vendor_source_modified": False,
        "original_unipc_calls_per_step": 1,
        "training": False,
        "optimizer": False,
        "mask_pose_flow_track_trajectory": False,
    }
    return {**value, "digest": object_sha256(value)}


def _tensor_raw_sha256(value: Any, *, label: str) -> str:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH runtime supplies torch
        raise NativeIAxisGuidanceError("tensor hashing requires PyTorch") from error
    if not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise NativeIAxisGuidanceError(f"{label} must be a non-empty tensor")
    byte_tensor = value.detach().contiguous().view(torch.uint8).cpu().reshape(-1)
    try:
        raw = byte_tensor.numpy().tobytes()
    except RuntimeError as error:
        # Some CPU-only contract-test environments intentionally have a
        # PyTorch/NumPy ABI mismatch.  The slower fallback is byte-identical
        # and is never expected on the pinned AUH runtime.
        if "Numpy is not available" not in str(error):
            raise
        raw = bytes(byte_tensor.tolist())
    return hashlib.sha256(raw).hexdigest()


def _tensor_rms(value: Any, *, label: str) -> float:
    try:
        numeric = float(value.detach().float().square().mean().sqrt().cpu().item())
    except Exception as error:
        raise NativeIAxisGuidanceError(f"cannot compute {label} RMS") from error
    if not math.isfinite(numeric):
        raise NativeIAxisGuidanceError(f"{label} RMS is non-finite")
    return numeric


def _scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise NativeIAxisGuidanceError(f"{label} is not scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        numeric = float(candidate)
    except NativeIAxisGuidanceError:
        raise
    except Exception as error:
        raise NativeIAxisGuidanceError(f"{label} is not numeric") from error
    if not math.isfinite(numeric):
        raise NativeIAxisGuidanceError(f"{label} is non-finite")
    return numeric


@dataclass
class _SharedCall:
    name: str
    values: Mapping[str, Any]
    prediction: Any


@dataclass
class _ActiveSample:
    cond_embeds: Any
    uncond_embeds: Any
    reference_count: int
    completed_steps: int = 0
    patch_outputs: list[tuple[Any, Any]] = field(default_factory=list)
    patch_source_ids: list[float] = field(default_factory=list)
    shared_calls: list[_SharedCall] = field(default_factory=list)
    records: list[Mapping[str, Any]] = field(default_factory=list)


class NativeIAxisGuidanceHook:
    """Reversible one-sample observer/projector for pinned native RV2V."""

    def __init__(
        self,
        diffusion: Any,
        *,
        arm: str,
        expected_steps: int,
        expected_bernini_commit: str,
        observed_wan_diffusion_sha256: str,
    ) -> None:
        if arm not in ARM_ORDER:
            raise NativeIAxisGuidanceError("hook arm is outside the sealed registry")
        if expected_steps != NUM_INFERENCE_STEPS:
            raise NativeIAxisGuidanceError("native-I canary is fixed to exact40")
        if expected_bernini_commit != PINNED_BERNINI_COMMIT:
            raise NativeIAxisGuidanceError("Bernini revision differs")
        if observed_wan_diffusion_sha256 != PINNED_WAN_DIFFUSION_SHA256:
            raise NativeIAxisGuidanceError("wan_diffusion.py bytes differ")
        self.diffusion = sampler_contract.resolve_diffusion_core(diffusion)
        self.transformer = getattr(self.diffusion, "transformer", None)
        self.scheduler = getattr(self.diffusion, "scheduler", None)
        self.arm = arm
        self.arm_contract = arm_reference_contract(arm)
        self.gated = bool(self.arm_contract["gated"])
        self.reference_count = int(self.arm_contract["reference_count"])
        self.expected_steps = expected_steps
        self._original_sample = getattr(self.diffusion, "sample", None)
        self._original_shared = getattr(self.diffusion, "shared_step", None)
        self._original_patch = getattr(self.transformer, "patch_vae_latent", None)
        self._original_scheduler = getattr(self.scheduler, "step", None)
        if not all(
            callable(value)
            for value in (
                self._original_sample,
                self._original_shared,
                self._original_patch,
                self._original_scheduler,
            )
        ):
            raise NativeIAxisGuidanceError("pinned sampler call surface differs")
        if getattr(self.diffusion, "transformer_2", None) is not None:
            raise NativeIAxisGuidanceError("native-I canary supports Bernini 1.3B only")
        if getattr(self.diffusion, "use_unipc", None) is not True:
            raise NativeIAxisGuidanceError("native-I canary requires native UniPC")
        for owner, name in (
            (self.diffusion, "sample"),
            (self.diffusion, "shared_step"),
            (self.transformer, "patch_vae_latent"),
            (self.scheduler, "step"),
        ):
            if name in vars(owner):
                raise NativeIAxisGuidanceError(f"refusing stacked override on {name}")
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._active: Optional[_ActiveSample] = None
        self.sample_calls = 0
        self.trace: Mapping[str, Any] = {}
        self.restored = False

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had_instance = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous))

    def install(self) -> None:
        if self._patches:
            raise NativeIAxisGuidanceError("native-I hook is already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared(*args, **kwargs)

        def patch_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_patch(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, patch_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_native_i_axis_guidance_hook", self)
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
            except Exception as error:  # pragma: no cover - catastrophic runtime failure
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise NativeIAxisGuidanceError("failed to restore native-I hook") from errors[0]

    def _expected_patch_source_ids(self) -> tuple[float, ...]:
        if self.reference_count == 0:
            return (1.0, 0.0)
        if self.reference_count != REFERENCE_COUNT:
            raise NativeIAxisGuidanceError("only RV2V-4 or ref-drop is supported")
        return (1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0)

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.sample_calls:
            raise NativeIAxisGuidanceError("hook permits exactly one native sample")
        try:
            values = sampler_contract._bind_call(self._original_sample, args, kwargs)
        except Exception as error:
            raise NativeIAxisGuidanceError(str(error)) from error
        refs = values.get("multi_image_vae_latents")
        refs = () if refs is None else tuple(refs)
        videos = values.get("multi_video_vae_latents")
        if (
            values.get("guidance_mode") != "rv2v"
            or values.get("num_frames") != FRAME_COUNT
            or values.get("num_inference_steps") != self.expected_steps
            or _scalar(values.get("omega_vid"), label="omega_vid") != OMEGA_VIDEO
            or _scalar(values.get("omega_img"), label="omega_img") != OMEGA_IMAGE
            or _scalar(values.get("omega_txt"), label="omega_txt") != OMEGA_TEXT
            or not isinstance(videos, (list, tuple))
            or len(videos) != 1
            or len(refs) != self.reference_count
            or values.get("image_vae_latents") is not None
        ):
            raise NativeIAxisGuidanceError("native sample/condition contract differs")
        cond = values.get("prompt_embeds")
        uncond = values.get("uncond_prompt_embeds")
        if cond is None or uncond is None or cond is uncond:
            raise NativeIAxisGuidanceError("native conditional axes differ")
        state = _ActiveSample(cond, uncond, self.reference_count)
        self._active = state
        try:
            result = self._original_sample(*args, **kwargs)
            if (
                state.completed_steps != self.expected_steps
                or state.patch_outputs
                or state.patch_source_ids
                or state.shared_calls
                or len(state.records) != self.expected_steps
            ):
                raise NativeIAxisGuidanceError("native sample ended with an open step")
            self.sample_calls = 1
            unsigned = {
                "contract": hook_contract(),
                "arm": dict(self.arm_contract),
                "sample_calls": 1,
                "step_count": len(state.records),
                "expected_transformer_forwards": (
                    self.expected_steps
                    * (5 if self.gated and self.reference_count else 4)
                ),
                "observed_transformer_forwards": sum(
                    int(row["transformer_forward_count"]) for row in state.records
                ),
                "steps": list(state.records),
                "numerical_mutation_surface": "scheduler.model_output_only",
                "vendor_source_modified": False,
            }
            self.trace = {**unsigned, "trace_digest": object_sha256(unsigned)}
            return result
        finally:
            self._active = None

    def _wrapped_patch(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise NativeIAxisGuidanceError("patch_vae_latent called outside active sample")
        expected = self._expected_patch_source_ids()
        index = len(state.patch_outputs)
        if index >= len(expected):
            raise NativeIAxisGuidanceError("too many native patch calls in one step")
        try:
            values = sampler_contract._bind_call(self._original_patch, args, kwargs)
        except Exception as error:
            raise NativeIAxisGuidanceError(str(error)) from error
        sid = _scalar(values.get("source_id"), label="patch source_id")
        if not math.isclose(sid, expected[index], rel_tol=0.0, abs_tol=1.0e-6):
            raise NativeIAxisGuidanceError("native patch source-id order differs")
        result = self._original_patch(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise NativeIAxisGuidanceError("patch_vae_latent return contract differs")
        latent, rotary = result
        if getattr(latent, "ndim", None) != 3 or getattr(rotary, "ndim", 0) < 3:
            raise NativeIAxisGuidanceError("native patch output geometry differs")
        state.patch_outputs.append((latent, rotary))
        state.patch_source_ids.append(sid)
        return result

    def _expected_shared_lengths(self, state: _ActiveSample) -> tuple[int, ...]:
        expected_patches = self._expected_patch_source_ids()
        if len(state.patch_outputs) != len(expected_patches):
            raise NativeIAxisGuidanceError("shared forward began before native pack closed")
        target_tokens = int(state.patch_outputs[-1][0].shape[1])
        video_tokens = int(state.patch_outputs[0][0].shape[1])
        if target_tokens != video_tokens or target_tokens <= 0:
            raise NativeIAxisGuidanceError("source/target patch geometry differs")
        if self.reference_count:
            ref_tokens = sum(int(state.patch_outputs[index][0].shape[1]) for index in (1, 3, 5, 7))
            vi_tokens = video_tokens + ref_tokens + target_tokens
        else:
            vi_tokens = video_tokens + target_tokens
        return (target_tokens, video_tokens + target_tokens, vi_tokens, vi_tokens)

    def _wrapped_shared(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise NativeIAxisGuidanceError("shared_step called outside active sample")
        names = ("none_uncond", "V_uncond", "VI_uncond", "VI_cond")
        index = len(state.shared_calls)
        if index >= len(names):
            raise NativeIAxisGuidanceError("unexpected extra native shared_step call")
        try:
            values = sampler_contract._bind_call(self._original_shared, args, kwargs)
        except Exception as error:
            raise NativeIAxisGuidanceError(str(error)) from error
        lengths = self._expected_shared_lengths(state)
        observed_lengths = values.get("batch_vae_seqlen")
        if (
            not isinstance(observed_lengths, (list, tuple))
            or tuple(int(value) for value in observed_lengths) != (lengths[index],)
            or values.get("model_id") != "transformer_1"
        ):
            raise NativeIAxisGuidanceError("native shared branch geometry differs")
        expected_prompt = state.cond_embeds if index == 3 else state.uncond_embeds
        if values.get("cond_embeds") is not expected_prompt:
            raise NativeIAxisGuidanceError("native RV2V prompt branch order differs")
        if index == 3:
            previous = state.shared_calls[2].values
            if (
                values.get("noisy_latents") is not previous.get("noisy_latents")
                or values.get("rotary_embs") is not previous.get("rotary_embs")
                or _scalar(values.get("timesteps"), label="VI_cond timestep")
                != _scalar(previous.get("timesteps"), label="VI_uncond timestep")
            ):
                raise NativeIAxisGuidanceError("VI uncond/cond state or timestep differs")
        prediction = self._original_shared(*args, **kwargs)
        if (
            getattr(prediction, "ndim", None) != 3
            or int(prediction.shape[0]) != 1
            or int(prediction.shape[1]) != lengths[index]
        ):
            raise NativeIAxisGuidanceError("native shared prediction geometry differs")
        state.shared_calls.append(_SharedCall(names[index], values, prediction))
        return prediction

    def _i_axis_target(self, state: _ActiveSample, target_tokens: int) -> tuple[Any, int]:
        if self.reference_count == 0:
            return state.shared_calls[0].prediction[:, -target_tokens:, :], 0
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise NativeIAxisGuidanceError("native I-axis forward requires PyTorch") from error
        # Exact pinned patch order: video, ref0:VI, ref0:I, ..., ref3:I, target.
        i_parts = [state.patch_outputs[index] for index in (2, 4, 6, 8)]
        target_part = state.patch_outputs[-1]
        i_latents = torch.cat([part[0] for part in i_parts] + [target_part[0]], dim=1)
        i_rotary = torch.cat([part[1] for part in i_parts] + [target_part[1]], dim=2)
        base = state.shared_calls[0].values
        prediction = self._original_shared(
            model_id="transformer_1",
            noisy_latents=i_latents,
            timesteps=base["timesteps"],
            cond_embeds=state.uncond_embeds,
            rotary_embs=i_rotary,
            batch_vae_seqlen=[int(i_latents.shape[1])],
            batch_text_seqlen=base["batch_text_seqlen"],
        )
        if (
            getattr(prediction, "ndim", None) != 3
            or int(prediction.shape[0]) != 1
            or int(prediction.shape[1]) != int(i_latents.shape[1])
        ):
            raise NativeIAxisGuidanceError("I-only shared prediction geometry differs")
        return prediction[:, -target_tokens:, :], 1

    def _wrapped_scheduler(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise NativeIAxisGuidanceError("scheduler.step called outside active sample")
        if (
            tuple(state.patch_source_ids) != self._expected_patch_source_ids()
            or len(state.shared_calls) != 4
        ):
            raise NativeIAxisGuidanceError("scheduler reached before native RV2V closure")
        try:
            values = sampler_contract._bind_call(self._original_scheduler, args, kwargs)
        except Exception as error:
            raise NativeIAxisGuidanceError(str(error)) from error
        model_output = values.get("model_output")
        sample = values.get("sample")
        if (
            getattr(model_output, "ndim", None) != 3
            or getattr(sample, "ndim", None) != 3
            or tuple(model_output.shape) != tuple(sample.shape)
        ):
            raise NativeIAxisGuidanceError("native UniPC packed geometry differs")
        step_index = state.completed_steps
        resolved_index, sigma, sigma_float = sampler_contract._resolve_sigma(
            self.scheduler, values.get("timestep")
        )
        if resolved_index != step_index:
            raise NativeIAxisGuidanceError("native UniPC schedule index/order differs")
        target_tokens = int(sample.shape[1])
        component_names = ("none_uncond", "V_uncond", "VI_uncond", "VI_cond")
        components = {
            call.name: call.prediction[:, -target_tokens:, :]
            for call in state.shared_calls
        }
        if tuple(components) != component_names or any(
            tuple(value.shape) != tuple(model_output.shape) for value in components.values()
        ):
            raise NativeIAxisGuidanceError("native target suffix geometry differs")
        v0 = components["none_uncond"]
        v_v = components["V_uncond"]
        v_vi_u = components["VI_uncond"]
        v_vi_c = components["VI_cond"]
        rebuilt_native = native_rv2v_velocity(v0, v_v, v_vi_u, v_vi_c).to(
            device=model_output.device, dtype=model_output.dtype
        )
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise NativeIAxisGuidanceError("native parity check requires PyTorch") from error
        if not torch.equal(rebuilt_native, model_output):
            difference = rebuilt_native.float() - model_output.float()
            raise NativeIAxisGuidanceError(
                "local raw RV2V formula is not exact native model_output: "
                f"max_abs={float(difference.abs().max().cpu().item()):.9g}"
            )

        gate = sigma_gate(step_index) if self.gated else 0.0
        i_target: Optional[Any] = None
        i_forward_count = 0
        correction: Optional[Any] = None
        executed = model_output
        if self.gated:
            i_target, i_forward_count = self._i_axis_target(state, target_tokens)
            if tuple(i_target.shape) != tuple(model_output.shape):
                raise NativeIAxisGuidanceError("I target and scheduler geometry differ")
            if self.reference_count == 0 and not (
                torch.equal(i_target, v0) and torch.equal(v_vi_u, v_v)
            ):
                raise NativeIAxisGuidanceError("ref-drop axes did not degenerate exactly")
            correction = OMEGA_IMAGE * gate * ((i_target - v0) - (v_vi_u - v_v))
            if gate > 0.0:
                candidate = gated_native_i_velocity(
                    v0, v_v, i_target, v_vi_u, v_vi_c, gate=gate
                ).to(device=model_output.device, dtype=model_output.dtype)
                # Ref-drop is a bit-exact native control even on active indices.
                if self.reference_count == 0:
                    if not torch.equal(candidate, model_output):
                        raise NativeIAxisGuidanceError("ref-drop changed native velocity")
                    executed = model_output
                else:
                    executed = candidate
        if gate == 0.0 and executed is not model_output:
            raise NativeIAxisGuidanceError("zero gate did not preserve model-output object")
        if step_index in FINAL_NATIVE_PARITY_INDICES and not torch.equal(
            executed, model_output
        ):
            raise NativeIAxisGuidanceError("final exact native parity failed")

        if executed is model_output:
            scheduler_args, scheduler_kwargs = tuple(args), dict(kwargs)
        else:
            try:
                scheduler_args, scheduler_kwargs = sampler_contract._replace_argument(
                    self._original_scheduler,
                    args,
                    kwargs,
                    name="model_output",
                    value=executed,
                )
            except Exception as error:
                raise NativeIAxisGuidanceError(str(error)) from error
        result = self._original_scheduler(*scheduler_args, **scheduler_kwargs)

        branch_hashes = {
            name: _tensor_raw_sha256(value, label=name)
            for name, value in components.items()
        }
        branch_calls = {name: 1 for name in component_names}
        if i_target is not None:
            branch_hashes["I_uncond"] = _tensor_raw_sha256(
                i_target, label="I_uncond"
            )
            branch_calls["I_uncond"] = i_forward_count
        record = {
            "step_index": step_index,
            "timestep": _scalar(values.get("timestep"), label="timestep"),
            "sigma": sigma_float,
            "sigma_dtype": str(getattr(sigma, "dtype", None)),
            "gate_hex": float(gate).hex(),
            "gate_active": gate > 0.0,
            "native_branch_order": list(component_names),
            "executed_branch_order": list(component_names)
            + (["I_uncond"] if i_target is not None else []),
            "branch_call_counts": branch_calls,
            "transformer_forward_count": 4 + i_forward_count,
            "patch_call_count": len(state.patch_outputs),
            "patch_source_ids": list(state.patch_source_ids),
            "shared_visual_token_lengths": list(self._expected_shared_lengths(state)),
            "target_tokens": target_tokens,
            "branch_target_raw_sha256": branch_hashes,
            "native_velocity_raw_sha256": _tensor_raw_sha256(
                model_output, label="native_velocity"
            ),
            "executed_velocity_raw_sha256": _tensor_raw_sha256(
                executed, label="executed_velocity"
            ),
            "correction_raw_sha256": (
                None
                if correction is None
                else _tensor_raw_sha256(correction, label="i_axis_correction")
            ),
            "correction_rms": (
                None
                if correction is None
                else _tensor_rms(correction, label="i_axis_correction")
            ),
            "native_formula_exact_parity": True,
            "scheduler_received_original_model_output_object": executed is model_output,
            "original_scheduler_call_count": 1,
            "i_axis_degenerate_alias_none": self.reference_count == 0 and self.gated,
            "final_native_parity": step_index in FINAL_NATIVE_PARITY_INDICES,
        }
        state.records.append(record)
        state.completed_steps += 1
        state.patch_outputs.clear()
        state.patch_source_ids.clear()
        state.shared_calls.clear()
        return result


__all__ = [
    "ACTIVE_GATE",
    "ACTIVE_STEP_INDICES",
    "ARM_ORDER",
    "CORRECT_REFERENCE_INDICES",
    "FINAL_NATIVE_PARITY_INDICES",
    "GATED_ARMS",
    "METHOD",
    "NATIVE_ARMS",
    "NUM_INFERENCE_STEPS",
    "NativeIAxisGuidanceError",
    "NativeIAxisGuidanceHook",
    "PERMUTED_REFERENCE_INDICES",
    "PHASE_SHIFT_REFERENCE_INDICES",
    "arm_plan",
    "arm_reference_contract",
    "gated_native_i_velocity",
    "hook_contract",
    "native_rv2v_velocity",
    "object_sha256",
    "permute_reference_objects",
    "sigma_gate",
]
