#!/usr/bin/env python3
"""Audited Bernini native full-video-axis intervention for exact40 RV2V.

Bernini's pinned RV2V sampler evaluates four branches at every denoising step
and combines their target suffixes as

``vN = v0 + 1.25*(vV-v0) + 4.5*(vVIu-vV) + 4.0*(vVIc-vVIu)``.

This module changes one registered scalar only: the coefficient of the first
standalone full-video difference ``(vV-v0)``.  ``V-on`` and ``wrong-V`` retain
the native coefficient 1.25; ``V-off`` uses zero.  Notice that this is a
coordinate-localization intervention, not a claim that the complete RV2V
prediction is video-independent: the native ``(vVIu-vV)`` term remains.

The hook observes the already-computed native branches and replaces only the
``model_output`` argument passed to the original UniPC step.  It neither adds
a transformer forward nor edits vendor files or model parameters.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import native_i_axis_guidance as i_axis
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-native-v-axis-guidance-hook-v1"
METHOD = "frozen-bernini-native-full-video-axis-causal-probe-v1"
PINNED_BERNINI_COMMIT = sampler_contract.PINNED_BERNINI_COMMIT
PINNED_WAN_DIFFUSION_SHA256 = sampler_contract.PINNED_WAN_DIFFUSION_SHA256

FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
REFERENCE_COUNT = 4
OMEGA_VIDEO_NATIVE = 1.25
OMEGA_IMAGE = 4.5
OMEGA_TEXT = 4.0
ARM_ORDER = ("V-on", "V-off", "wrong-V")


class NativeVAxisGuidanceError(RuntimeError):
    """Raised before an unaudited V-axis field reaches UniPC."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return i_axis.canonical_json_bytes(value)
    except Exception as error:
        raise NativeVAxisGuidanceError(str(error)) from error


def object_sha256(value: Any) -> str:
    return i_axis.object_sha256(value)


def arm_contract(arm: str) -> Mapping[str, Any]:
    """Return the complete registered condition change for one arm."""

    if arm not in ARM_ORDER:
        raise NativeVAxisGuidanceError(f"unknown native V-axis arm: {arm!r}")
    wrong = arm == "wrong-V"
    omega_video = 0.0 if arm == "V-off" else OMEGA_VIDEO_NATIVE
    return {
        "arm": arm,
        "full_video_condition_role": "wrong" if wrong else "correct",
        "omega_video": omega_video,
        "omega_image": OMEGA_IMAGE,
        "omega_text": OMEGA_TEXT,
        "correct_image_references": True,
        "same_instruction": True,
        "same_scheduler": True,
        "same_target_geometry": True,
        "intervention": (
            "replace_full_video_condition_only"
            if wrong
            else (
                "zero_standalone_vV_minus_v0_coefficient_only"
                if arm == "V-off"
                else "native_no_numerical_intervention"
            )
        ),
        "v_vi_u_minus_v_v_term_retained": True,
    }


def v_axis_velocity(
    v0: Any,
    v_v: Any,
    v_vi_u: Any,
    v_vi_c: Any,
    *,
    omega_video: float,
) -> Any:
    """Combine the four native branches with one explicit V coefficient."""

    numeric = float(omega_video)
    if not math.isfinite(numeric) or numeric not in (0.0, OMEGA_VIDEO_NATIVE):
        raise NativeVAxisGuidanceError("omega_video must be registered 0 or 1.25")
    return (
        v0
        + numeric * (v_v - v0)
        + OMEGA_IMAGE * (v_vi_u - v_v)
        + OMEGA_TEXT * (v_vi_c - v_vi_u)
    )


def hook_contract() -> Mapping[str, Any]:
    unsigned = {
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
        "v_off_formula": (
            "v0+0.0*(vV-v0)+4.5*(vVIu-vV)+4.0*(vVIc-vVIu)"
        ),
        "coordinate": "raw_velocity_before_original_unipc_step",
        "arm_order": list(ARM_ORDER),
        "native_branch_forward_order": [
            "none_uncond", "V_uncond", "VI_uncond", "VI_cond"
        ],
        "transformer_forwards_per_step": 4,
        "original_unipc_calls_per_step": 1,
        "all_exact40_steps_intervened_for_v_off": True,
        "apg": False,
        "training": False,
        "optimizer": False,
        "feature_scorer": False,
        "selection": False,
        "vendor_source_modified": False,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


class NativeVAxisGuidanceHook(i_axis.NativeIAxisGuidanceHook):
    """Reversible exact40 observer/projector for one registered V arm."""

    def __init__(
        self,
        diffusion: Any,
        *,
        arm: str,
        expected_steps: int,
        expected_bernini_commit: str,
        observed_wan_diffusion_sha256: str,
    ) -> None:
        contract = arm_contract(arm)
        try:
            super().__init__(
                diffusion,
                arm="N-C",
                expected_steps=expected_steps,
                expected_bernini_commit=expected_bernini_commit,
                observed_wan_diffusion_sha256=observed_wan_diffusion_sha256,
            )
        except Exception as error:
            raise NativeVAxisGuidanceError(str(error)) from error
        self.arm = arm
        self.arm_contract = contract
        self.gated = False
        self.reference_count = REFERENCE_COUNT

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.sample_calls:
            raise NativeVAxisGuidanceError("hook permits exactly one native sample")
        try:
            values = sampler_contract._bind_call(self._original_sample, args, kwargs)
        except Exception as error:
            raise NativeVAxisGuidanceError(str(error)) from error
        refs = values.get("multi_image_vae_latents")
        refs = () if refs is None else tuple(refs)
        videos = values.get("multi_video_vae_latents")
        if (
            values.get("guidance_mode") != "rv2v"
            or values.get("num_frames") != FRAME_COUNT
            or values.get("num_inference_steps") != self.expected_steps
            or i_axis._scalar(values.get("omega_vid"), label="omega_vid")
            != OMEGA_VIDEO_NATIVE
            or i_axis._scalar(values.get("omega_img"), label="omega_img")
            != OMEGA_IMAGE
            or i_axis._scalar(values.get("omega_txt"), label="omega_txt")
            != OMEGA_TEXT
            or not isinstance(videos, (list, tuple))
            or len(videos) != 1
            or len(refs) != REFERENCE_COUNT
            or values.get("image_vae_latents") is not None
        ):
            raise NativeVAxisGuidanceError("native sample/condition contract differs")
        cond = values.get("prompt_embeds")
        uncond = values.get("uncond_prompt_embeds")
        if cond is None or uncond is None or cond is uncond:
            raise NativeVAxisGuidanceError("native prompt axes differ")
        state = i_axis._ActiveSample(cond, uncond, REFERENCE_COUNT)
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
                raise NativeVAxisGuidanceError("native sample ended with an open step")
            self.sample_calls = 1
            unsigned = {
                "contract": hook_contract(),
                "arm": dict(self.arm_contract),
                "sample_calls": 1,
                "step_count": len(state.records),
                "expected_transformer_forwards": self.expected_steps * 4,
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

    def _wrapped_scheduler(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise NativeVAxisGuidanceError("scheduler.step called outside sample")
        if (
            tuple(state.patch_source_ids) != self._expected_patch_source_ids()
            or len(state.shared_calls) != 4
        ):
            raise NativeVAxisGuidanceError("scheduler reached before branch closure")
        try:
            values = sampler_contract._bind_call(self._original_scheduler, args, kwargs)
        except Exception as error:
            raise NativeVAxisGuidanceError(str(error)) from error
        model_output = values.get("model_output")
        sample = values.get("sample")
        if (
            getattr(model_output, "ndim", None) != 3
            or getattr(sample, "ndim", None) != 3
            or tuple(model_output.shape) != tuple(sample.shape)
        ):
            raise NativeVAxisGuidanceError("native UniPC packed geometry differs")
        step_index = state.completed_steps
        resolved_index, sigma, sigma_float = sampler_contract._resolve_sigma(
            self.scheduler, values.get("timestep")
        )
        if resolved_index != step_index:
            raise NativeVAxisGuidanceError("native UniPC schedule order differs")
        target_tokens = int(sample.shape[1])
        names = ("none_uncond", "V_uncond", "VI_uncond", "VI_cond")
        components = {
            call.name: call.prediction[:, -target_tokens:, :]
            for call in state.shared_calls
        }
        if tuple(components) != names or any(
            tuple(value.shape) != tuple(model_output.shape)
            for value in components.values()
        ):
            raise NativeVAxisGuidanceError("native target suffix geometry differs")
        v0 = components["none_uncond"]
        v_v = components["V_uncond"]
        v_vi_u = components["VI_uncond"]
        v_vi_c = components["VI_cond"]
        rebuilt_native = v_axis_velocity(
            v0, v_v, v_vi_u, v_vi_c, omega_video=OMEGA_VIDEO_NATIVE
        ).to(device=model_output.device, dtype=model_output.dtype)
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise NativeVAxisGuidanceError("V-axis hook requires PyTorch") from error
        if not torch.equal(rebuilt_native, model_output):
            difference = rebuilt_native.float() - model_output.float()
            raise NativeVAxisGuidanceError(
                "local RV2V formula is not exact native output: "
                f"max_abs={float(difference.abs().max().cpu().item()):.9g}"
            )
        omega_video = float(self.arm_contract["omega_video"])
        if omega_video == OMEGA_VIDEO_NATIVE:
            executed = model_output
        else:
            executed = v_axis_velocity(
                v0, v_v, v_vi_u, v_vi_c, omega_video=omega_video
            ).to(device=model_output.device, dtype=model_output.dtype)
        correction = executed.float() - model_output.float()
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
                raise NativeVAxisGuidanceError(str(error)) from error
        result = self._original_scheduler(*scheduler_args, **scheduler_kwargs)
        branch_hashes = {
            name: i_axis._tensor_raw_sha256(value, label=name)
            for name, value in components.items()
        }
        record = {
            "step_index": step_index,
            "timestep": i_axis._scalar(values.get("timestep"), label="timestep"),
            "sigma": sigma_float,
            "sigma_dtype": str(getattr(sigma, "dtype", None)),
            "omega_video_hex": omega_video.hex(),
            "standalone_v_axis_active": omega_video != 0.0,
            "native_branch_order": list(names),
            "executed_branch_order": list(names),
            "branch_call_counts": {name: 1 for name in names},
            "transformer_forward_count": 4,
            "patch_call_count": len(state.patch_outputs),
            "patch_source_ids": list(state.patch_source_ids),
            "shared_visual_token_lengths": list(self._expected_shared_lengths(state)),
            "target_tokens": target_tokens,
            "branch_target_raw_sha256": branch_hashes,
            "native_velocity_raw_sha256": i_axis._tensor_raw_sha256(
                model_output, label="native_velocity"
            ),
            "executed_velocity_raw_sha256": i_axis._tensor_raw_sha256(
                executed, label="executed_velocity"
            ),
            "v_axis_correction_rms": i_axis._tensor_rms(
                correction, label="v_axis_correction"
            ),
            "native_formula_exact_parity": True,
            "scheduler_received_original_model_output_object": executed is model_output,
            "original_scheduler_call_count": 1,
            "v_vi_u_minus_v_v_term_retained": True,
        }
        state.records.append(record)
        state.completed_steps += 1
        state.patch_outputs.clear()
        state.patch_source_ids.clear()
        state.shared_calls.clear()
        return result


__all__ = [
    "ARM_ORDER",
    "METHOD",
    "NUM_INFERENCE_STEPS",
    "NativeVAxisGuidanceError",
    "NativeVAxisGuidanceHook",
    "arm_contract",
    "hook_contract",
    "v_axis_velocity",
]
