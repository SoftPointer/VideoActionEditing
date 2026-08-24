#!/usr/bin/env python3
"""Scientific-v3 primitives for Bernini source-self role learning.

This module closes three gaps in the v2 engineering canary without pretending
that the pretext task already solves action editing:

* visual conditioning is assembled through Bernini's native
  ``transformer.patch_vae_latent`` path and reproduces the four RV2V axes
  ``none/V/I/VI``.  Image references are deliberately patched twice: source
  ids 2--5 on the VI axis and source ids 1--4 on the I axis;
* noisy states are sampled only from the 40 non-terminal coordinates of the
  pinned exact-40 Bernini UniPC inference schedule;
* correct, reversed-donor, wrong-reference and reference-off cells all remain
  graph connected.  A causal ranking loss makes the correct cell lower-error
  than every intervention, then a fresh post-update evaluation emits strict
  gates instead of treating pre-update diagnostics as evidence.

The objective can still exploit negative-cell shortcuts, cross-scene wrong
references, or pose leakage in source frames.  Consequently this file is an
auditable core for a one-step scientific canary, not authorization for a long
training run and not evidence of semantic motion transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping, Optional, Sequence

import torch


SCHEMA_VERSION = "bernini-source-self-native-ref-contrastive-v4"
FRAME_COUNT = 81
LATENT_PHASES = 21
LATENT_CHANNELS = 16
REFERENCE_COUNT = 4
VI_VIDEO_SOURCE_IDS = (1.0,)
VI_IMAGE_SOURCE_IDS = (2.0, 3.0, 4.0, 5.0)
I_IMAGE_SOURCE_IDS = (1.0, 2.0, 3.0, 4.0)
PATCH_CALL_SOURCE_IDS = (1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0)
PATCH_CALL_ROLES = (
    "video:VI",
    "ref0:VI",
    "ref0:I",
    "ref1:VI",
    "ref1:I",
    "ref2:VI",
    "ref2:I",
    "ref3:VI",
    "ref3:I",
    "target",
)
BRANCH_CONCAT_ORDER = {
    "none": ("target",),
    "V": ("video", "target"),
    "I": ("ref0", "ref1", "ref2", "ref3", "target"),
    "VI": ("video", "ref0", "ref1", "ref2", "ref3", "target"),
}
LATENT_CONCAT_DIM = 1
ROTARY_CONCAT_DIM = 2
PINNED_NATIVE_RV2V4_REFERENCE_CONTRACT_DIGEST = (
    "f7ce6020ef02d536012d5e1c952ee828d08a72dd74f7fea828bcf0ec3b1ed0d4"
)
ROLE_CELL_NAMES = ("correct", "reverse", "wrong", "off")
NEGATIVE_CELL_NAMES = ROLE_CELL_NAMES[1:]

# Extracted from the pinned Bernini-R checkpoint with
# UniPCMultistepScheduler.from_pretrained(..., flow_shift=5.0), followed by
# set_timesteps(40).  The scheduler has one additional terminal sigma=0 which
# is not a transformer-forward coordinate and is therefore excluded here.
NATIVE_UNIPC40_TIMESTEPS = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)
NATIVE_UNIPC40_SIGMAS = (
    0.9999989867210388,
    0.9949031472206116,
    0.9895941615104675,
    0.9840595126152039,
    0.978284478187561,
    0.9722530841827393,
    0.9659478068351746,
    0.9593496322631836,
    0.9524376392364502,
    0.9451888799667358,
    0.9375780820846558,
    0.9295775294303894,
    0.9211564660072327,
    0.912280797958374,
    0.9029127359390259,
    0.893010139465332,
    0.8825258612632751,
    0.871407151222229,
    0.8595945835113525,
    0.8470211625099182,
    0.8336109519004822,
    0.8192774057388306,
    0.8039219379425049,
    0.7874310612678528,
    0.7696741223335266,
    0.7504994869232178,
    0.7297303080558777,
    0.7071589827537537,
    0.6825404167175293,
    0.6555827856063843,
    0.6259360909461975,
    0.5931769013404846,
    0.55678790807724,
    0.5161304473876953,
    0.4704066216945648,
    0.41860657930374146,
    0.3594328761100769,
    0.2911904454231262,
    0.21162153780460358,
    0.11765105277299881,
)
PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST = (
    "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
)


class NativeRefContrastiveV3Error(RuntimeError):
    """Raised before an ambiguous v3 pack, update, or gate is accepted."""


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
        raise NativeRefContrastiveV3Error(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def native_rv2v4_reference_contract() -> Mapping[str, Any]:
    """Return Bernini's exact one-video plus four-reference source-ID contract."""

    if (
        REFERENCE_COUNT != 4
        or len(VI_IMAGE_SOURCE_IDS) != REFERENCE_COUNT
        or len(I_IMAGE_SOURCE_IDS) != REFERENCE_COUNT
        or len(PATCH_CALL_SOURCE_IDS) != 2 + 2 * REFERENCE_COUNT
        or len(PATCH_CALL_ROLES) != len(PATCH_CALL_SOURCE_IDS)
        or set(BRANCH_CONCAT_ORDER) != {"none", "V", "I", "VI"}
    ):
        raise NativeRefContrastiveV3Error("native RV2V-4 registry differs")
    value = {
        "native_mode": "RV2V-4",
        "video_condition_count": 1,
        "image_reference_count": REFERENCE_COUNT,
        "total_visual_condition_count": 1 + REFERENCE_COUNT,
        "vi_video_source_ids": list(VI_VIDEO_SOURCE_IDS),
        "vi_image_source_ids": list(VI_IMAGE_SOURCE_IDS),
        "i_image_source_ids": list(I_IMAGE_SOURCE_IDS),
        "patch_call_source_ids": list(PATCH_CALL_SOURCE_IDS),
        "patch_call_roles": list(PATCH_CALL_ROLES),
        "branch_concat_order": {
            name: list(BRANCH_CONCAT_ORDER[name]) for name in ("none", "V", "I", "VI")
        },
        "latent_concat_dim": LATENT_CONCAT_DIM,
        "rotary_concat_dim": ROTARY_CONCAT_DIM,
        "target_source_id": 0.0,
        "source_id_interpolation_used": False,
        "maximum_source_id": 5.0,
    }
    digest = object_sha256(value)
    if digest != PINNED_NATIVE_RV2V4_REFERENCE_CONTRACT_DIGEST:
        raise NativeRefContrastiveV3Error("pinned native RV2V-4 digest differs")
    return {**value, "digest": digest}


def native_unipc40_schedule_receipt() -> Mapping[str, Any]:
    """Return an immutable, float-hex schedule contract for receipts."""

    value = {
        "scheduler": "UniPCMultistepScheduler",
        "prediction_type": "flow_prediction",
        "use_flow_sigmas": True,
        "flow_shift": 5.0,
        "num_inference_steps": 40,
        "model_forward_count": 40,
        "terminal_sigma_excluded_from_training": 0.0,
        "timesteps": list(NATIVE_UNIPC40_TIMESTEPS),
        "sigma_float64_hex": [float(value).hex() for value in NATIVE_UNIPC40_SIGMAS],
        "sampling_distribution": "uniform_without_replacement_over_exact40_per_cycle",
    }
    return {**value, "digest": object_sha256(value)}


def _validate_native_schedule() -> None:
    if len(NATIVE_UNIPC40_TIMESTEPS) != 40 or len(NATIVE_UNIPC40_SIGMAS) != 40:
        raise NativeRefContrastiveV3Error("native UniPC schedule must have 40 forwards")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in NATIVE_UNIPC40_TIMESTEPS
    ):
        raise NativeRefContrastiveV3Error("native timesteps must be exact integers")
    if any(
        left <= right
        for left, right in zip(NATIVE_UNIPC40_TIMESTEPS, NATIVE_UNIPC40_TIMESTEPS[1:])
    ):
        raise NativeRefContrastiveV3Error("native timesteps must strictly decrease")
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in NATIVE_UNIPC40_SIGMAS
    ):
        raise NativeRefContrastiveV3Error("native model-forward sigmas must lie in (0,1]")
    if any(
        left <= right
        for left, right in zip(NATIVE_UNIPC40_SIGMAS, NATIVE_UNIPC40_SIGMAS[1:])
    ):
        raise NativeRefContrastiveV3Error("native sigmas must strictly decrease")
    if native_unipc40_schedule_receipt()["digest"] != PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST:
        raise NativeRefContrastiveV3Error("pinned native UniPC40 schedule digest differs")


_validate_native_schedule()


def schedule_indices_for_step(
    *, seed: int, step: int, samples_per_step: int = 4
) -> tuple[int, ...]:
    """Draw one preregistered exact-40 stratum without replacement.

    For the default four-sigma microbatch, ten consecutive steps cover all 40
    native inference coordinates exactly once.  A new deterministic
    permutation is created for the next ten-step cycle.  Restricting the
    microbatch size to divisors of 40 avoids a short final stratum and silent
    changes in coordinate probability.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise NativeRefContrastiveV3Error("schedule seed must lie in [0,2^63)")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise NativeRefContrastiveV3Error("schedule step must be a nonnegative integer")
    if (
        isinstance(samples_per_step, bool)
        or not isinstance(samples_per_step, int)
        or samples_per_step <= 0
        or 40 % samples_per_step
    ):
        raise NativeRefContrastiveV3Error("samples_per_step must be a positive divisor of 40")
    steps_per_cycle = 40 // samples_per_step
    cycle, offset = divmod(step, steps_per_cycle)
    material = f"{seed}\0native-unipc40\0{cycle}".encode("ascii")
    permutation_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    indices = list(range(40))
    random.Random(permutation_seed).shuffle(indices)
    start = offset * samples_per_step
    return tuple(indices[start : start + samples_per_step])


@dataclass(frozen=True)
class MultiSigmaStates:
    indices: tuple[int, ...]
    sigmas: torch.Tensor
    timesteps: torch.Tensor
    noisy: torch.Tensor
    target_velocity: torch.Tensor
    weights: torch.Tensor

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schedule_digest": native_unipc40_schedule_receipt()["digest"],
            "indices": list(self.indices),
            "timesteps": [int(value) for value in self.timesteps.detach().cpu().tolist()],
            "sigma_float64_hex": [
                float(value).hex() for value in self.sigmas.detach().cpu().tolist()
            ],
            "weights_float64_hex": [
                float(value).hex() for value in self.weights.detach().cpu().tolist()
            ],
            "same_clean_and_epsilon_across_sigma_coordinates": True,
            "flow_equations": {
                "noisy": "x_sigma=(1-sigma)*clean+sigma*epsilon",
                "target_velocity": "epsilon-clean",
            },
        }
        return {**value, "digest": object_sha256(value)}


def build_multi_sigma_states(
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    indices: Sequence[int],
    device: Optional[torch.device | str] = None,
) -> MultiSigmaStates:
    """Build exact inference-coordinate rectified-flow states.

    The leading output axis is sigma.  The same clean video and Gaussian are
    used at every coordinate so condition interventions can be compared with
    common random numbers.
    """

    if (
        not isinstance(clean, torch.Tensor)
        or not isinstance(epsilon, torch.Tensor)
        or clean.shape != epsilon.shape
        or clean.ndim < 2
        or not clean.is_floating_point()
        or clean.dtype != epsilon.dtype
        or clean.device != epsilon.device
        or clean.requires_grad
        or epsilon.requires_grad
        or not bool(torch.isfinite(clean).all().item())
        or not bool(torch.isfinite(epsilon).all().item())
    ):
        raise NativeRefContrastiveV3Error(
            "clean/epsilon must be same-shape, graph-free, finite floating tensors"
        )
    exact_indices = tuple(indices)
    if (
        not exact_indices
        or len(set(exact_indices)) != len(exact_indices)
        or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 40 for index in exact_indices)
    ):
        raise NativeRefContrastiveV3Error("sigma indices must be unique integers in [0,40)")
    output_device = clean.device if device is None else torch.device(device)
    work_clean = clean.to(device=output_device, dtype=torch.float32)
    work_epsilon = epsilon.to(device=output_device, dtype=torch.float32)
    sigmas = torch.tensor(
        [NATIVE_UNIPC40_SIGMAS[index] for index in exact_indices],
        dtype=torch.float64,
        device=output_device,
    )
    timesteps = torch.tensor(
        [NATIVE_UNIPC40_TIMESTEPS[index] for index in exact_indices],
        # Pinned UniPC exposes float32 timesteps to WanDiffusion.sample even
        # though every exact-40 coordinate happens to be integer-valued.
        dtype=torch.float32,
        device=output_device,
    )
    broadcast_shape = (len(exact_indices),) + (1,) * work_clean.ndim
    sigma32 = sigmas.to(dtype=torch.float32).reshape(broadcast_shape)
    clean_s = work_clean.unsqueeze(0)
    epsilon_s = work_epsilon.unsqueeze(0)
    noisy = ((1.0 - sigma32) * clean_s + sigma32 * epsilon_s).contiguous()
    velocity = (work_epsilon - work_clean).unsqueeze(0).expand_as(noisy).contiguous()
    weights = torch.full(
        (len(exact_indices),),
        1.0 / float(len(exact_indices)),
        dtype=torch.float64,
        device=output_device,
    )
    return MultiSigmaStates(exact_indices, sigmas, timesteps, noisy, velocity, weights)


@dataclass(frozen=True)
class NativeRV2VBranch:
    name: str
    latents: torch.Tensor
    rotary: torch.Tensor
    target_mask: torch.Tensor
    total_tokens: int
    condition_tokens: int
    source_ids: tuple[float, ...]
    concat_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.name not in {"none", "V", "I", "VI"}:
            raise NativeRefContrastiveV3Error("unknown native RV2V branch")
        if (
            not isinstance(self.latents, torch.Tensor)
            or self.latents.ndim != 3
            or int(self.latents.shape[0]) != 1
            or not isinstance(self.rotary, torch.Tensor)
            or self.rotary.ndim < 3
            or not isinstance(self.target_mask, torch.Tensor)
            or self.target_mask.dtype != torch.bool
            or self.target_mask.ndim != 1
            or int(self.latents.shape[1]) != self.total_tokens
            or int(self.target_mask.numel()) != self.total_tokens
            or int(self.target_mask.sum().item()) != self.total_tokens - self.condition_tokens
            or len(self.source_ids) != len(self.concat_order)
        ):
            raise NativeRefContrastiveV3Error("native RV2V branch geometry differs")
        if bool(self.target_mask[: self.condition_tokens].any().item()) or not bool(
            self.target_mask[self.condition_tokens :].all().item()
        ):
            raise NativeRefContrastiveV3Error("native target mask is not a suffix")


@dataclass(frozen=True)
class NativeRV2VPack:
    none: NativeRV2VBranch
    video: NativeRV2VBranch
    image: NativeRV2VBranch
    video_image: NativeRV2VBranch
    reference_count: int
    patch_call_source_ids: tuple[float, ...]
    patch_call_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.reference_count != REFERENCE_COUNT
            or self.patch_call_source_ids != PATCH_CALL_SOURCE_IDS
            or self.patch_call_roles != PATCH_CALL_ROLES
            or self.none.source_ids != (0.0,)
            or self.video.source_ids != VI_VIDEO_SOURCE_IDS + (0.0,)
            or self.image.source_ids != I_IMAGE_SOURCE_IDS + (0.0,)
            or self.video_image.source_ids
            != VI_VIDEO_SOURCE_IDS + VI_IMAGE_SOURCE_IDS + (0.0,)
            or self.none.concat_order != BRANCH_CONCAT_ORDER["none"]
            or self.video.concat_order != BRANCH_CONCAT_ORDER["V"]
            or self.image.concat_order != BRANCH_CONCAT_ORDER["I"]
            or self.video_image.concat_order != BRANCH_CONCAT_ORDER["VI"]
        ):
            raise NativeRefContrastiveV3Error(
                "native RV2V-4 pack/source-ID closure differs"
            )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "native_vendor_path": "WanDiffusion.sample: none/V/I/VI",
            "native_patch_method": "transformer.patch_vae_latent",
            "reference_count": self.reference_count,
            "native_rv2v4_reference_contract_digest": (
                native_rv2v4_reference_contract()["digest"]
            ),
            "patch_call_source_ids": list(self.patch_call_source_ids),
            "patch_call_roles": list(self.patch_call_roles),
            "latent_concat_dim": LATENT_CONCAT_DIM,
            "rotary_concat_dim": ROTARY_CONCAT_DIM,
            "branches": {
                branch.name: {
                    "total_tokens": branch.total_tokens,
                    "condition_tokens": branch.condition_tokens,
                    "source_ids": list(branch.source_ids),
                    "concat_order": list(branch.concat_order),
                }
                for branch in (self.none, self.video, self.image, self.video_image)
            },
            "vi_image_source_ids": list(VI_IMAGE_SOURCE_IDS),
            "i_image_source_ids": list(I_IMAGE_SOURCE_IDS),
            "image_refs_repatched_on_i_axis": True,
            "target_source_id": 0.0,
        }
        return {**value, "digest": object_sha256(value)}


def build_native_rv2v_pack(
    transformer: Any,
    *,
    donor_video: torch.Tensor,
    image_references: Sequence[torch.Tensor],
    noisy_target: torch.Tensor,
) -> NativeRV2VPack:
    """Reproduce Bernini's native RV2V ``none/V/I/VI`` visual packing.

    This intentionally calls ``patch_vae_latent`` ten times, matching the
    pinned vendor implementation for one video and four image references:
    video once, every image once on VI and once on I, and target once.
    """

    patch = getattr(transformer, "patch_vae_latent", None)
    if not callable(patch):
        raise NativeRefContrastiveV3Error(
            "transformer must expose Bernini patch_vae_latent"
        )
    refs = tuple(image_references)
    if len(refs) != REFERENCE_COUNT:
        raise NativeRefContrastiveV3Error("native RV2V-4 requires exactly four image refs")
    tensors = (donor_video, *refs, noisy_target)
    if any(not isinstance(value, torch.Tensor) or value.ndim != 5 for value in tensors):
        raise NativeRefContrastiveV3Error("native inputs must be [B,C,T,H,W] tensors")
    if any(int(value.shape[0]) != 1 for value in tensors):
        raise NativeRefContrastiveV3Error("native Bernini pack supports batch size one")
    if (
        int(donor_video.shape[1]) != LATENT_CHANNELS
        or int(noisy_target.shape[1]) != LATENT_CHANNELS
        or int(donor_video.shape[2]) != LATENT_PHASES
        or int(noisy_target.shape[2]) != LATENT_PHASES
    ):
        raise NativeRefContrastiveV3Error(
            "native donor/target must be exact81 [1,16,21,H,W] latents"
        )
    if any(int(value.shape[1]) != LATENT_CHANNELS for value in refs):
        raise NativeRefContrastiveV3Error("every native image ref must have 16 channels")
    if any(int(value.shape[2]) != 1 for value in refs):
        raise NativeRefContrastiveV3Error("every native image ref must have one latent phase")
    if any(
        tuple(value.shape[:2]) != tuple(noisy_target.shape[:2])
        or tuple(value.shape[3:]) != tuple(noisy_target.shape[3:])
        for value in (donor_video, *refs)
    ):
        raise NativeRefContrastiveV3Error("native visual conditions have incompatible geometry")

    call_sids: list[float] = []
    call_roles: list[str] = []

    def native_patch(
        value: torch.Tensor, source_id: float, role: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        call_sids.append(float(source_id))
        call_roles.append(role)
        result = patch(value.to(dtype=getattr(transformer, "dtype", value.dtype)), source_id=source_id)
        if not isinstance(result, tuple) or len(result) != 2:
            raise NativeRefContrastiveV3Error("patch_vae_latent must return (latent, rotary)")
        latent, rotary = result
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 3
            or int(latent.shape[0]) != 1
            or not isinstance(rotary, torch.Tensor)
            or rotary.ndim < 3
        ):
            raise NativeRefContrastiveV3Error("native patch output geometry differs")
        return latent, rotary

    # Exact call and source-id order in pinned WanDiffusion.sample.
    video_latent, video_rotary = native_patch(donor_video, 1.0, "video:VI")
    vi_ref_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    i_ref_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, reference in enumerate(refs):
        vi_ref_parts.append(native_patch(reference, float(index + 2), f"ref{index}:VI"))
        i_ref_parts.append(native_patch(reference, float(index + 1), f"ref{index}:I"))
    target_latent, target_rotary = native_patch(noisy_target, 0.0, "target")

    def assemble(
        name: str,
        conditions: Sequence[tuple[torch.Tensor, torch.Tensor]],
        source_ids: tuple[float, ...],
        concat_order: tuple[str, ...],
    ) -> NativeRV2VBranch:
        latent_parts = [item[0] for item in conditions] + [target_latent]
        rotary_parts = [item[1] for item in conditions] + [target_rotary]
        latents = torch.cat(latent_parts, dim=LATENT_CONCAT_DIM)
        # Native WanDiffusion concatenates rotary tokens on dimension 2.
        rotary = torch.cat(rotary_parts, dim=ROTARY_CONCAT_DIM)
        condition_tokens = sum(int(item[0].shape[1]) for item in conditions)
        total = int(latents.shape[1])
        mask = torch.cat(
            (
                torch.zeros(condition_tokens, dtype=torch.bool, device=latents.device),
                torch.ones(total - condition_tokens, dtype=torch.bool, device=latents.device),
            )
        )
        return NativeRV2VBranch(
            name,
            latents,
            rotary,
            mask,
            total,
            condition_tokens,
            source_ids + (0.0,),
            concat_order,
        )

    pack = NativeRV2VPack(
        none=assemble("none", (), (), BRANCH_CONCAT_ORDER["none"]),
        video=assemble(
            "V", ((video_latent, video_rotary),), (1.0,), BRANCH_CONCAT_ORDER["V"]
        ),
        image=assemble("I", i_ref_parts, I_IMAGE_SOURCE_IDS, BRANCH_CONCAT_ORDER["I"]),
        video_image=assemble(
            "VI",
            ((video_latent, video_rotary), *vi_ref_parts),
            VI_VIDEO_SOURCE_IDS + VI_IMAGE_SOURCE_IDS,
            BRANCH_CONCAT_ORDER["VI"],
        ),
        reference_count=len(refs),
        patch_call_source_ids=tuple(call_sids),
        patch_call_roles=tuple(call_roles),
    )
    if pack.image.source_ids != I_IMAGE_SOURCE_IDS + (0.0,):
        raise NativeRefContrastiveV3Error("native image-only source IDs differ")
    if pack.patch_call_source_ids != PATCH_CALL_SOURCE_IDS:
        raise NativeRefContrastiveV3Error("native VI+I patch call order differs")
    if pack.patch_call_roles != PATCH_CALL_ROLES:
        raise NativeRefContrastiveV3Error("native VI+I physical patch role order differs")
    return pack


def forward_native_target_branch(
    wan_diffusion: Any,
    branch: NativeRV2VBranch,
    *,
    timestep: torch.Tensor,
    cond_embeds: torch.Tensor,
) -> torch.Tensor:
    """Forward one already-native-packed branch and select its target suffix.

    The function intentionally consumes patch embeddings returned by
    ``patch_vae_latent``.  Calling ``transformer.patch_embedding`` again here
    would depart from native RV2V and is forbidden.  Adapter route contexts,
    when used, must surround this call so target-row Q/O wrappers observe the
    same full native sequence used by the base transformer.
    """

    shared_step = getattr(wan_diffusion, "shared_step", None)
    if not callable(shared_step):
        raise NativeRefContrastiveV3Error("wan_diffusion must expose shared_step")
    if not isinstance(branch, NativeRV2VBranch):
        raise NativeRefContrastiveV3Error("forward requires a NativeRV2VBranch")
    if (
        not isinstance(timestep, torch.Tensor)
        or timestep.dtype != torch.float32
        or timestep.numel() != 1
        or timestep.requires_grad
        or timestep.device != branch.latents.device
        or not bool(torch.isfinite(timestep).all().item())
        or float(timestep.item()) not in NATIVE_UNIPC40_TIMESTEPS
    ):
        raise NativeRefContrastiveV3Error(
            "timestep must be one exact native-UniPC40 device-local FP32 coordinate"
        )
    if (
        not isinstance(cond_embeds, torch.Tensor)
        or cond_embeds.ndim != 3
        or int(cond_embeds.shape[0]) != 1
        or int(cond_embeds.shape[1]) <= 0
        or cond_embeds.device != branch.latents.device
        or cond_embeds.requires_grad
    ):
        raise NativeRefContrastiveV3Error(
            "condition embeds must be frozen device-local [1,L,D]"
        )
    prediction = shared_step(
        model_id="transformer_1",
        noisy_latents=branch.latents,
        timesteps=timestep.reshape(1),
        cond_embeds=cond_embeds,
        rotary_embs=branch.rotary,
        batch_vae_seqlen=[branch.total_tokens],
        batch_text_seqlen=[int(cond_embeds.shape[1])],
    )
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.ndim != 3
        or int(prediction.shape[0]) != 1
        or int(prediction.shape[1]) != branch.total_tokens
        or prediction.device != branch.latents.device
    ):
        raise NativeRefContrastiveV3Error("native shared_step output geometry differs")
    target = prediction[:, branch.target_mask, :]
    if int(target.shape[1]) != branch.total_tokens - branch.condition_tokens:
        raise NativeRefContrastiveV3Error("native target suffix selection differs")
    return target


@dataclass(frozen=True)
class RoleCausalObjective:
    loss: torch.Tensor
    correct_error: torch.Tensor
    error_by_cell: Mapping[str, torch.Tensor]
    gap_by_negative: Mapping[str, torch.Tensor]
    hinge_by_negative: Mapping[str, torch.Tensor]
    error_by_sigma: Mapping[str, torch.Tensor]


def _validate_role_predictions(
    predictions: Mapping[str, torch.Tensor], target: torch.Tensor
) -> None:
    if set(predictions) != set(ROLE_CELL_NAMES):
        raise NativeRefContrastiveV3Error(
            "role predictions must contain exactly correct/reverse/wrong/off"
        )
    if not isinstance(target, torch.Tensor) or target.ndim < 2 or not target.is_floating_point():
        raise NativeRefContrastiveV3Error("role target must be a floating [S,...] tensor")
    for name in ROLE_CELL_NAMES:
        value = predictions[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != target.shape
            or value.device != target.device
            or not value.is_floating_point()
            or not value.requires_grad
        ):
            raise NativeRefContrastiveV3Error(
                f"{name} prediction must be graph-connected with target geometry"
            )


def role_causal_ranking_objective(
    predictions: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    sigma_weights: torch.Tensor,
    margin: float = 0.05,
    ranking_weight: float = 1.0,
    correct_weight: float = 1.0,
) -> RoleCausalObjective:
    """Make the correct role assignment beat every single intervention."""

    _validate_role_predictions(predictions, target)
    if (
        not isinstance(sigma_weights, torch.Tensor)
        or sigma_weights.ndim != 1
        or int(sigma_weights.numel()) != int(target.shape[0])
        or sigma_weights.device != target.device
        or not sigma_weights.is_floating_point()
        or bool((sigma_weights < 0).any().item())
        or not bool(torch.isfinite(sigma_weights).all().item())
        or float(sigma_weights.sum().item()) <= 0.0
    ):
        raise NativeRefContrastiveV3Error("sigma weights must be finite shared [S] weights")
    for name, value in (
        ("margin", margin),
        ("ranking_weight", ranking_weight),
        ("correct_weight", correct_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise NativeRefContrastiveV3Error(f"{name} must be a finite scalar")
    if margin <= 0.0 or ranking_weight <= 0.0 or correct_weight <= 0.0:
        raise NativeRefContrastiveV3Error("objective scalars must be positive")

    reduce_dims = tuple(range(1, target.ndim))
    normalized_weights = sigma_weights.float() / sigma_weights.float().sum()
    per_sigma = {
        name: (predictions[name].float() - target.float()).square().mean(dim=reduce_dims)
        for name in ROLE_CELL_NAMES
    }
    errors = {
        name: (values * normalized_weights).sum()
        for name, values in per_sigma.items()
    }
    correct = errors["correct"]
    gaps = {name: errors[name] - correct for name in NEGATIVE_CELL_NAMES}
    hinges = {
        name: torch.relu(correct.new_tensor(float(margin)) - gap)
        for name, gap in gaps.items()
    }
    loss = float(correct_weight) * correct + float(ranking_weight) * sum(hinges.values())
    if not bool(torch.isfinite(loss).item()) or not loss.requires_grad:
        raise NativeRefContrastiveV3Error("causal role loss is non-finite or detached")
    return RoleCausalObjective(loss, correct, errors, gaps, hinges, per_sigma)


def detached_error_snapshot(result: RoleCausalObjective) -> Mapping[str, float]:
    return {
        name: float(value.detach().float().cpu().item())
        for name, value in result.error_by_cell.items()
    }


def post_update_role_gates(
    *,
    pre_errors: Mapping[str, float],
    post_errors: Mapping[str, float],
    margin: float,
    maximum_correct_regression: float = 0.0,
    minimum_gap_gain: float = 0.0,
) -> Mapping[str, Any]:
    """Evaluate fresh optimizer-after role metrics with fail-closed gates."""

    if set(pre_errors) != set(ROLE_CELL_NAMES) or set(post_errors) != set(ROLE_CELL_NAMES):
        raise NativeRefContrastiveV3Error("pre/post errors require the exact four role cells")
    scalars = {
        **{f"pre_{name}": value for name, value in pre_errors.items()},
        **{f"post_{name}": value for name, value in post_errors.items()},
        "margin": margin,
        "maximum_correct_regression": maximum_correct_regression,
        "minimum_gap_gain": minimum_gap_gain,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in scalars.values()
    ):
        raise NativeRefContrastiveV3Error("post-update gate inputs must be finite scalars")
    if margin <= 0.0 or maximum_correct_regression < 0.0 or minimum_gap_gain < 0.0:
        raise NativeRefContrastiveV3Error("post-update gate thresholds are invalid")

    pre_gaps = {
        name: float(pre_errors[name]) - float(pre_errors["correct"])
        for name in NEGATIVE_CELL_NAMES
    }
    post_gaps = {
        name: float(post_errors[name]) - float(post_errors["correct"])
        for name in NEGATIVE_CELL_NAMES
    }
    margin_gates = {
        name: gap >= float(margin) for name, gap in post_gaps.items()
    }
    gap_gain_gates = {
        name: post_gaps[name] >= pre_gaps[name] + float(minimum_gap_gain)
        for name in NEGATIVE_CELL_NAMES
    }
    correct_gate = float(post_errors["correct"]) <= (
        float(pre_errors["correct"]) + float(maximum_correct_regression)
    )
    accepted = correct_gate and all(margin_gates.values()) and all(gap_gain_gates.values())
    value = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_timing": "fresh_forward_after_optimizer_step",
        "pre_errors": {name: float(pre_errors[name]) for name in ROLE_CELL_NAMES},
        "post_errors": {name: float(post_errors[name]) for name in ROLE_CELL_NAMES},
        "pre_gaps": pre_gaps,
        "post_gaps": post_gaps,
        "required_margin": float(margin),
        "minimum_gap_gain": float(minimum_gap_gain),
        "maximum_correct_regression": float(maximum_correct_regression),
        "correct_nonregression_gate": correct_gate,
        "margin_gates": margin_gates,
        "gap_gain_gates": gap_gain_gates,
        "all_negative_cells_graph_connected_during_update": True,
        "accepted": accepted,
    }
    return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class CausalUpdateResult:
    loss: float
    pre_errors: Mapping[str, float]
    post_errors: Mapping[str, float]
    gates: Mapping[str, Any]
    gradient_norm: float


def run_causal_update(
    *,
    forward_cells: Callable[[], tuple[Mapping[str, torch.Tensor], torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    trainable_parameters: Sequence[torch.nn.Parameter],
    sigma_weights: torch.Tensor,
    margin: float = 0.05,
    ranking_weight: float = 1.0,
    correct_weight: float = 1.0,
    maximum_correct_regression: float = 0.0,
    minimum_gap_gain: float = 0.0,
    gradient_sync: Optional[Callable[[Sequence[torch.nn.Parameter]], None]] = None,
) -> CausalUpdateResult:
    """Execute one update and a truly post-update four-cell gate.

    Distributed callers must supply ``gradient_sync`` implementing their
    audited SP-then-DP reduction before the optimizer step.
    """

    parameters = tuple(trainable_parameters)
    if not parameters or len({id(value) for value in parameters}) != len(parameters):
        raise NativeRefContrastiveV3Error("trainable parameter set is empty or aliased")
    optimizer.zero_grad(set_to_none=True)
    pre_predictions, target = forward_cells()
    objective = role_causal_ranking_objective(
        pre_predictions,
        target,
        sigma_weights=sigma_weights,
        margin=margin,
        ranking_weight=ranking_weight,
        correct_weight=correct_weight,
    )
    pre_errors = detached_error_snapshot(objective)
    objective.loss.backward()
    if gradient_sync is not None:
        gradient_sync(parameters)
    if any(
        value.grad is None or not bool(torch.isfinite(value.grad).all().item())
        for value in parameters
    ):
        raise NativeRefContrastiveV3Error("a trainable gradient is missing or non-finite")
    squared = sum(value.grad.detach().float().square().sum() for value in parameters)
    gradient_norm = float(squared.sqrt().item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise NativeRefContrastiveV3Error("causal update gradient norm is invalid")
    optimizer.step()

    # Recompute.  Reusing pre-update tensors here would turn the gate into a lie.
    with torch.no_grad():
        post_predictions, post_target = forward_cells()
        if post_target.shape != target.shape:
            raise NativeRefContrastiveV3Error("post-update target geometry changed")
        reduce_dims = tuple(range(1, post_target.ndim))
        normalized = sigma_weights.float() / sigma_weights.float().sum()
        post_errors = {
            name: float(
                (
                    (post_predictions[name].float() - post_target.float())
                    .square()
                    .mean(dim=reduce_dims)
                    * normalized
                )
                .sum()
                .cpu()
                .item()
            )
            for name in ROLE_CELL_NAMES
        }
    gates = post_update_role_gates(
        pre_errors=pre_errors,
        post_errors=post_errors,
        margin=margin,
        maximum_correct_regression=maximum_correct_regression,
        minimum_gap_gain=minimum_gap_gain,
    )
    return CausalUpdateResult(
        loss=float(objective.loss.detach().float().cpu().item()),
        pre_errors=pre_errors,
        post_errors=post_errors,
        gates=gates,
        gradient_norm=gradient_norm,
    )


__all__ = [
    "BRANCH_CONCAT_ORDER",
    "CausalUpdateResult",
    "FRAME_COUNT",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "LATENT_CONCAT_DIM",
    "I_IMAGE_SOURCE_IDS",
    "MultiSigmaStates",
    "NATIVE_UNIPC40_SIGMAS",
    "NATIVE_UNIPC40_TIMESTEPS",
    "PINNED_NATIVE_RV2V4_REFERENCE_CONTRACT_DIGEST",
    "PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST",
    "PATCH_CALL_SOURCE_IDS",
    "PATCH_CALL_ROLES",
    "REFERENCE_COUNT",
    "ROTARY_CONCAT_DIM",
    "VI_IMAGE_SOURCE_IDS",
    "VI_VIDEO_SOURCE_IDS",
    "NativeRV2VBranch",
    "NativeRV2VPack",
    "NativeRefContrastiveV3Error",
    "RoleCausalObjective",
    "build_multi_sigma_states",
    "build_native_rv2v_pack",
    "detached_error_snapshot",
    "forward_native_target_branch",
    "native_unipc40_schedule_receipt",
    "native_rv2v4_reference_contract",
    "post_update_role_gates",
    "role_causal_ranking_objective",
    "run_causal_update",
    "schedule_indices_for_step",
]
