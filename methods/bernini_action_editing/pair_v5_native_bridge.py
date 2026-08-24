"""Executable Bernini-native model bridge for PAIR-v5.

The bridge has two deliberately separate boundaries.

``FrozenBerniniT2VScorer`` is the candidate-own-coordinate action critic.  It
accepts only ``(x_sigma, sigma, prompt)`` and caches one target-only Bernini
patch/timestep packet across the closed ten-prompt MACE registry.  There is no
slot through which a source, paired target, proposal, donor, mask, flow, pose,
or track can enter the critic.  Its output is an exact-81 spatial velocity
field; :func:`score_frozen_t2v_action_energy` also returns the frozen MACE
energy packet.

``forward_native_rv2v4_policy_pair`` is the deployment-model boundary.  It
builds Bernini's native one-video/four-image-reference pack independently for
the trainable student and frozen/reference policy, runs the registered
none/V/VI four-forward formula, and returns exact-81 spatial velocities.  T2V
proposal pixels or latents are not accepted by this boundary either.

This module is plumbing, not evidence that the learned policy edits actions.
Every receipt states that limitation explicitly.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
import struct
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn

import dclr_runtime_contract as runtime_contract
import mace_candidate_action_energy as mace
import pair_v5_phase_conjunctive_energy as phase_energy
import source_self_native_ref_contrastive_v3 as native
import source_self_native_rv2v_guidance as guidance
import source_self_native_target_adapter as cio_adapter


SCHEMA_VERSION = "bernini-pair-v5-native-bridge-v1"
T2V_SCORER_SCHEMA = "bernini-pair-v5-frozen-t2v-spatial-scorer-v1"
RV2V_POLICY_PAIR_SCHEMA = "bernini-pair-v5-native-rv2v4-policy-pair-v1"
FRAME_COUNT = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
PATCH_SIZE = (1, 2, 2)
T2V_TARGET_SOURCE_ID = 0
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

EXPANDED_GUIDANCE_COEFFICIENTS = {
    "none_uncond": -0.25,
    "V_uncond": -3.25,
    "VI_uncond": 0.5,
    "VI_cond": 4.0,
}

FORBIDDEN_SCORER_INPUT_NAMES = frozenset(
    {
        "source",
        "source_video",
        "source_latent",
        "target",
        "target_video",
        "target_latent",
        "paired_target",
        "proposal",
        "proposal_video",
        "proposal_latent",
        "proposal_noise",
        "donor",
        "donor_video",
        "donor_latent",
        "mask",
        "motion_mask",
        "flow",
        "optical_flow",
        "pose",
        "track",
        "tracks",
        "trajectory",
        "trajectories",
        "reference_video",
    }
)


class PairV5NativeBridgeError(ValueError):
    """A PAIR-v5 model call differs from the sealed Bernini contract."""


# One deliberately closed extension point is needed by the Q-MOSAIC native
# hidden-VJP module.  Keeping the registry here avoids duck-typing arbitrary
# objects with a ``route`` method while also avoiding an import cycle from the
# bridge back into the VJP implementation.
_QMOSAIC_CORE16_REGISTRY_ID = "qmosaic-core16-fixed-a-b-only-v1"


@dataclass(frozen=True)
class _ClosedActionAdapterRegistration:
    adapter_type: type
    route_factory: Callable[..., Any]
    gate_factory: Callable[..., tuple[str, float]]


_CLOSED_ACTION_ADAPTER_REGISTRY: dict[str, _ClosedActionAdapterRegistration] = {}


def register_closed_action_adapter_type(
    *,
    registry_id: str,
    adapter_type: type,
    route_factory: Callable[..., Any],
    gate_factory: Callable[..., tuple[str, float]],
) -> None:
    """Register the single audited Q-MOSAIC adapter handle type.

    This is intentionally not a general plugin registry.  The exact ID,
    module basename and class name are pinned, registration is idempotent only
    for the same type/callables, and dispatch below uses ``type(x) is T``.
    """

    module_name = getattr(adapter_type, "__module__", "")
    if (
        registry_id != _QMOSAIC_CORE16_REGISTRY_ID
        or not isinstance(adapter_type, type)
        or not module_name.endswith("self_imagined_native_rv2v_hidden_vjp_v1")
        or getattr(adapter_type, "__name__", "") != "Core16ActionLoRAHandle"
        or not callable(route_factory)
        or not callable(gate_factory)
    ):
        raise PairV5NativeBridgeError(
            "external action adapter registration is outside the closed registry"
        )
    candidate = _ClosedActionAdapterRegistration(
        adapter_type=adapter_type,
        route_factory=route_factory,
        gate_factory=gate_factory,
    )
    current = _CLOSED_ACTION_ADAPTER_REGISTRY.get(registry_id)
    if current is not None and current != candidate:
        raise PairV5NativeBridgeError("closed action adapter registration changed")
    _CLOSED_ACTION_ADAPTER_REGISTRY[registry_id] = candidate


def _closed_action_adapter_registration(
    adapter: Any,
) -> Optional[_ClosedActionAdapterRegistration]:
    registration = _CLOSED_ACTION_ADAPTER_REGISTRY.get(
        _QMOSAIC_CORE16_REGISTRY_ID
    )
    if registration is not None and type(adapter) is registration.adapter_type:
        return registration
    return None


# The same FP32 mean is also evaluated once as a ten-branch batch below for a
# diagnostic. ROCm may choose a different parallel reduction tree for that
# shape, so bit equality is not valid for the batched diagnostic. The actual
# mutation/identity audit is recomputed branch-serial with MACE's exact tensor
# shape and remains bit-exact. These auxiliary bounds are deliberately tighter
# than BF16 quantization and are recorded in every successful bridge receipt.
VELOCITY_ENERGY_CLOSURE_RTOL = 1.0e-5
VELOCITY_ENERGY_CLOSURE_ATOL = 1.0e-7


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
        raise PairV5NativeBridgeError(
            "receipt value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5NativeBridgeError(f"{label} must be lowercase SHA-256")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
    }
    # ``bytes(UntypedStorage)`` iterates one Python integer at a time in some
    # Torch releases.  Reinterpretation as uint8 preserves BF16/FP16/FP32
    # bytes while letting NumPy perform one contiguous bulk copy.
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    if len(raw) != int(cpu.numel() * cpu.element_size()):
        raise PairV5NativeBridgeError("tensor receipt storage byte count differs")
    digest = hashlib.sha256()
    digest.update(_canonical_json(metadata))
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


def _fp32_bits(value: torch.Tensor, *, label: str) -> str:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.numel() != 1
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5NativeBridgeError(f"{label} must be one detached finite FP32 value")
    return struct.pack("!f", float(value.item())).hex()


def _storage_ptr(value: torch.Tensor) -> int:
    getter = getattr(value, "untyped_storage", None)
    storage = getter() if getter is not None else value.storage()
    return int(storage.data_ptr())


def _validate_exact81_spatial(
    value: Any,
    *,
    label: str,
    detached_fp32: bool,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or int(value.shape[3]) % PATCH_SIZE[1]
        or int(value.shape[4]) % PATCH_SIZE[2]
        or not value.is_floating_point()
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5NativeBridgeError(
            f"{label} must be finite exact81 [1,16,21,H,W] with positive even H/W"
        )
    if detached_fp32 and (
        value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise PairV5NativeBridgeError(f"{label} must be detached FP32")
    return value


def _validate_image_references(
    values: Any,
    *,
    video: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if isinstance(values, (str, bytes)):
        raise PairV5NativeBridgeError("image_references must contain exactly four tensors")
    try:
        refs = tuple(values)
    except TypeError as error:
        raise PairV5NativeBridgeError(
            "image_references must contain exactly four tensors"
        ) from error
    if len(refs) != native.REFERENCE_COUNT:
        raise PairV5NativeBridgeError("native RV2V-4 requires exactly four image refs")
    expected = (1, LATENT_CHANNELS, 1, int(video.shape[3]), int(video.shape[4]))
    for index, value in enumerate(refs):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(int(item) for item in value.shape) != expected
            or value.device != video.device
            or not value.is_floating_point()
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise PairV5NativeBridgeError(
                f"image_references[{index}] must be detached [1,16,1,H,W] on the video device"
            )
    return refs


def _unpack_spatial_velocity(
    packed: Any,
    *,
    video_shape: Sequence[int],
) -> torch.Tensor:
    """Invert Wan's ``(t,h,w),(pt,ph,pw,c)`` target token order."""

    shape = tuple(int(item) for item in video_shape)
    if (
        len(shape) != 5
        or shape[:3] != (1, LATENT_CHANNELS, LATENT_PHASES)
        or shape[3] <= 0
        or shape[4] <= 0
        or shape[3] % 2
        or shape[4] % 2
    ):
        raise PairV5NativeBridgeError("video_shape is not exact81 spatial geometry")
    batch, channels, phases, height, width = shape
    token_count = phases * (height // 2) * (width // 2)
    if (
        not isinstance(packed, torch.Tensor)
        or packed.ndim != 3
        or tuple(int(item) for item in packed.shape)
        != (batch, token_count, runtime_contract.PINNED_PATCH_DIM)
        or not packed.is_floating_point()
        or not bool(torch.isfinite(packed).all().item())
    ):
        raise PairV5NativeBridgeError(
            f"packed velocity must be [1,{token_count},{runtime_contract.PINNED_PATCH_DIM}]"
        )
    patches = packed.reshape(
        batch,
        phases,
        height // 2,
        width // 2,
        PATCH_SIZE[0],
        PATCH_SIZE[1],
        PATCH_SIZE[2],
        channels,
    )
    return (
        patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
        .reshape(batch, channels, phases, height, width)
        .contiguous()
    )


def _validate_prompt_registry(value: Any) -> dict[str, str]:
    try:
        return mace.validate_prompt_closure(value)
    except mace.MACECandidateActionEnergyError as error:
        raise PairV5NativeBridgeError(str(error)) from error


def _validate_text_conditions(
    prompt_by_branch: Mapping[str, str],
    condition_by_branch: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(condition_by_branch, Mapping):
        raise PairV5NativeBridgeError("condition_by_branch must be a mapping")
    if set(condition_by_branch) != set(mace.BRANCH_ORDER):
        raise PairV5NativeBridgeError("text-condition branch closure differs")
    result: dict[str, torch.Tensor] = {}
    prior: list[tuple[str, torch.Tensor]] = []
    for branch in mace.BRANCH_ORDER:
        value = condition_by_branch[branch]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(int(item) for item in value.shape)
            != (
                1,
                runtime_contract.PINNED_TEXT_TOKENS,
                runtime_contract.PINNED_TEXT_DIM,
            )
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or value.device.type == "meta"
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            raise PairV5NativeBridgeError(
                f"condition for {branch} must be frozen [1,512,4096]"
            )
        for prior_branch, prior_value in prior:
            if torch.equal(value, prior_value):
                raise PairV5NativeBridgeError(
                    f"condition for {branch} aliases branch {prior_branch}"
                )
        result[branch] = value
        prior.append((branch, value))
    # Bind raw prompt text separately from embeddings.  This prevents a caller
    # from changing branch labels after official tokenization.
    if len(prompt_by_branch) != len(result):
        raise PairV5NativeBridgeError("prompt/text-condition cardinality differs")
    return result


class FrozenBerniniT2VScorer(nn.Module):
    """Frozen target-only Bernini denoiser with a closed prompt registry.

    A ten-call MACE packet patches ``x_sigma`` once.  All prompt branches then
    reuse the very same target token, rotary, and FP32 timestep tensor objects.
    Calls outside :data:`mace.BRANCH_ORDER` or in a different order fail.
    """

    def __init__(
        self,
        diffusion: nn.Module,
        transformer: nn.Module,
        prompt_by_branch: Mapping[str, str],
        condition_by_branch: Mapping[str, torch.Tensor],
        *,
        frozen_model_receipt_digest: str,
        model_id: str = "transformer_1",
    ) -> None:
        super().__init__()
        if not isinstance(diffusion, nn.Module) or not isinstance(transformer, nn.Module):
            raise PairV5NativeBridgeError(
                "diffusion and transformer must be auditable torch modules"
            )
        if not callable(getattr(diffusion, "shared_step", None)):
            raise PairV5NativeBridgeError("diffusion must expose shared_step")
        if not callable(getattr(transformer, "patch_vae_latent", None)):
            raise PairV5NativeBridgeError("transformer must expose patch_vae_latent")
        if model_id not in {"transformer_1", "transformer_2"}:
            raise PairV5NativeBridgeError("model_id must identify one Bernini expert")
        prompts = _validate_prompt_registry(prompt_by_branch)
        conditions = _validate_text_conditions(prompts, condition_by_branch)
        model_receipt_digest = _require_sha256(
            frozen_model_receipt_digest, label="frozen_model_receipt_digest"
        )
        if any(parameter.requires_grad for parameter in diffusion.parameters()) or any(
            parameter.requires_grad for parameter in transformer.parameters()
        ):
            raise PairV5NativeBridgeError("frozen T2V scorer contains trainable parameters")

        self.diffusion = diffusion
        self.transformer = transformer
        self.model_id = model_id
        self._prompt_by_branch = prompts
        self._branch_by_prompt = {prompt: branch for branch, prompt in prompts.items()}
        self._prompt_registry_digest = object_sha256(prompts)
        self._condition_registry_digest = object_sha256(
            {
                branch: _tensor_sha256(conditions[branch])
                for branch in mace.BRANCH_ORDER
            }
        )
        self._frozen_model_receipt_digest = model_receipt_digest
        for index, branch in enumerate(mace.BRANCH_ORDER):
            self.register_buffer(
                f"_condition_{index}", conditions[branch], persistent=False
            )
        self._packet_key: Optional[tuple[int, int, int, int]] = None
        self._packet_branch: Any = None
        self._packet_timestep: Optional[torch.Tensor] = None
        self._packet_shape: Optional[tuple[int, ...]] = None
        self._packet_position = 0
        self._last_packet_receipt: Optional[Mapping[str, Any]] = None
        self.eval()

    @property
    def prompt_registry_digest(self) -> str:
        return self._prompt_registry_digest

    @property
    def condition_registry_digest(self) -> str:
        return self._condition_registry_digest

    @property
    def last_packet_receipt(self) -> Optional[Mapping[str, Any]]:
        return self._last_packet_receipt

    def prompt_registry(self) -> Mapping[str, str]:
        return dict(self._prompt_by_branch)

    def _condition(self, branch: str) -> torch.Tensor:
        return getattr(self, f"_condition_{mace.BRANCH_ORDER.index(branch)}")

    def _current_condition_registry_digest(self) -> str:
        return object_sha256(
            {
                branch: _tensor_sha256(self._condition(branch))
                for branch in mace.BRANCH_ORDER
            }
        )

    def abort_packet(self) -> None:
        """Drop an incomplete ephemeral packet after a failed scorer call."""

        self._packet_key = None
        self._packet_branch = None
        self._packet_timestep = None
        self._packet_shape = None
        self._packet_position = 0

    def _start_packet(self, x_sigma: torch.Tensor, sigma: torch.Tensor) -> None:
        if self._packet_key is not None:
            raise PairV5NativeBridgeError("a T2V score packet is already active")
        if self._current_condition_registry_digest() != self.condition_registry_digest:
            raise PairV5NativeBridgeError(
                "frozen ten-branch text-condition registry differs from its seal"
            )
        dtype = getattr(self.transformer, "dtype", None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise PairV5NativeBridgeError("transformer exposes no supported dtype")
        condition_devices = {self._condition(name).device for name in mace.BRANCH_ORDER}
        if condition_devices != {x_sigma.device}:
            raise PairV5NativeBridgeError(
                "all frozen text conditions must share the candidate device"
            )
        with torch.no_grad():
            result = self.transformer.patch_vae_latent(
                x_sigma.to(dtype=dtype), source_id=T2V_TARGET_SOURCE_ID
            )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise PairV5NativeBridgeError(
                "patch_vae_latent must return target tokens and rotary"
            )
        try:
            branch = runtime_contract.build_t2v_target_branch(
                result[0], result[1], target_source_id=T2V_TARGET_SOURCE_ID
            )
            timestep = runtime_contract.fp32_sigma_to_timestep(sigma)
        except runtime_contract.DCLRRuntimeContractError as error:
            raise PairV5NativeBridgeError(str(error)) from error
        if tuple(timestep.shape) != (1,) or not bool(
            ((sigma > 0.0) & (sigma < 1.0)).all().item()
        ):
            raise PairV5NativeBridgeError("T2V scorer sigma must be exact [1] in (0,1)")
        expected_tokens = int(x_sigma.shape[2]) * (
            int(x_sigma.shape[3]) // 2
        ) * (int(x_sigma.shape[4]) // 2)
        if branch.target_token_count != expected_tokens:
            raise PairV5NativeBridgeError("T2V patched token geometry differs from exact81")
        self._packet_key = (
            id(x_sigma),
            int(x_sigma._version),
            id(sigma),
            int(sigma._version),
        )
        self._packet_branch = branch
        self._packet_timestep = timestep
        self._packet_shape = tuple(int(item) for item in x_sigma.shape)
        self._packet_position = 0
        self._last_packet_receipt = None

    def forward(
        self,
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        prompt: str,
    ) -> torch.Tensor:
        _validate_exact81_spatial(
            x_sigma, label="candidate-own x_sigma", detached_fp32=True
        )
        if (
            not isinstance(sigma, torch.Tensor)
            or sigma.dtype != torch.float32
            or tuple(sigma.shape) != (1,)
            or sigma.device != x_sigma.device
            or sigma.requires_grad
            or sigma.grad_fn is not None
            or not bool(torch.isfinite(sigma).all().item())
        ):
            raise PairV5NativeBridgeError("sigma must be detached device-local FP32 [1]")
        branch_name = self._branch_by_prompt.get(prompt)
        if branch_name is None:
            raise PairV5NativeBridgeError("prompt is outside the frozen ten-branch registry")
        expected_branch = mace.BRANCH_ORDER[self._packet_position]
        if branch_name != expected_branch:
            raise PairV5NativeBridgeError(
                f"T2V scorer expected branch {expected_branch}, observed {branch_name}"
            )
        if self._packet_position == 0:
            self._start_packet(x_sigma, sigma)
        expected_key = (
            id(x_sigma),
            int(x_sigma._version),
            id(sigma),
            int(sigma._version),
        )
        if self._packet_key != expected_key:
            raise PairV5NativeBridgeError(
                "all ten T2V branches must reuse the same x_sigma/sigma objects"
            )
        branch = self._packet_branch
        timestep = self._packet_timestep
        video_shape = self._packet_shape
        condition = self._condition(branch_name)
        if branch is None or timestep is None or video_shape is None:
            raise PairV5NativeBridgeError("T2V packet was not initialized")
        with torch.no_grad():
            prediction = self.diffusion.shared_step(
                model_id=self.model_id,
                noisy_latents=branch.noisy_latents,
                timesteps=timestep,
                cond_embeds=condition,
                rotary_embs=branch.rotary_embs,
                batch_vae_seqlen=list(branch.batch_vae_seqlen),
                batch_text_seqlen=[runtime_contract.PINNED_TEXT_TOKENS],
            )
        total = branch.total_token_count
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(int(item) for item in prediction.shape)
            != (1, total, runtime_contract.PINNED_PATCH_DIM)
            or prediction.device != x_sigma.device
            or prediction.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or prediction.requires_grad
            or prediction.grad_fn is not None
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise PairV5NativeBridgeError(
                "frozen shared_step must return detached [1,N,64] velocity"
            )
        # T2V is target-only, so this is a direct storage view of the complete
        # target tail rather than a copied/repacked pseudo-MV2V prediction.
        packed_velocity = prediction[:, -branch.target_token_count :, :]
        if _storage_ptr(packed_velocity) != _storage_ptr(prediction):
            raise PairV5NativeBridgeError("T2V target tail is not a direct storage view")
        spatial = _unpack_spatial_velocity(packed_velocity, video_shape=video_shape)
        self._packet_position += 1
        if self._packet_position == len(mace.BRANCH_ORDER):
            value = {
                "schema_version": T2V_SCORER_SCHEMA,
                "branch_order": list(mace.BRANCH_ORDER),
                "prompt_registry_digest": self.prompt_registry_digest,
                "condition_registry_digest": self.condition_registry_digest,
                "frozen_model_receipt_digest": self._frozen_model_receipt_digest,
                "candidate_shape": list(video_shape),
                "spatial_velocity_shape": list(video_shape),
                "target_tokens": branch.target_token_count,
                "target_source_id": T2V_TARGET_SOURCE_ID,
                "t2v_target_tail_direct_storage_view": True,
                "patch_vae_latent_calls_per_ten_branch_packet": 1,
                "shared_x_sigma_object": True,
                "shared_sigma_object": True,
                "shared_timestep_object": True,
                "condition_registry_revalidated_before_packet": True,
                "sigma_float32_bits_hex": _fp32_bits(sigma, label="sigma"),
                "timestep_float32_bits_hex": _fp32_bits(
                    timestep, label="timestep"
                ),
                "timestep_mapping": "float32_t_equals_1000_times_sigma_no_shift",
                "proposal_visual_data_consumed": False,
            }
            self._last_packet_receipt = {**value, "digest": object_sha256(value)}
            self.abort_packet()
        return spatial


class _RecordingFrozenScorer(nn.Module):
    def __init__(self, scorer: FrozenBerniniT2VScorer) -> None:
        super().__init__()
        self.scorer = scorer
        self.values: list[torch.Tensor] = []
        self.input_ids: list[tuple[int, int]] = []
        self.eval()

    def forward(
        self, x_sigma: torch.Tensor, sigma: torch.Tensor, prompt: str
    ) -> torch.Tensor:
        self.input_ids.append((id(x_sigma), id(sigma)))
        value = self.scorer(x_sigma, sigma, prompt)
        self.values.append(value.detach().float())
        return value


@dataclass(frozen=True)
class FrozenT2VActionScore:
    """Ten velocities plus global and phase-conjunctive frozen energies."""

    branch_velocities: torch.Tensor
    energy: mace.CandidateActionEnergyResult
    phase_energy: phase_energy.PhaseConjunctiveEnergyResult
    receipt: Mapping[str, Any]


def score_frozen_t2v_action_energy(
    clean_candidate: torch.Tensor,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    prompt_by_branch: Mapping[str, str],
    scorer: FrozenBerniniT2VScorer,
    phase_weight_commitment: Mapping[str, Any],
    *,
    registered_phase_weight_digest: str,
    energy_epsilon: float = mace.DEFAULT_ENERGY_EPSILON,
) -> FrozenT2VActionScore:
    """Score one candidate and immediately apply the sealed phase conjunction.

    ``x_sigma`` is constructed inside the MACE core from the three public
    candidate-state arguments.  The scorer observes that single tensor object
    across all ten serial prompts.  Its resulting spatial fields are passed
    directly to the phase core here; callers cannot substitute branch-specific
    states between model inference and milestone evaluation.
    """

    clean = _validate_exact81_spatial(
        clean_candidate, label="clean candidate", detached_fp32=True
    )
    if not isinstance(scorer, FrozenBerniniT2VScorer):
        raise PairV5NativeBridgeError("scorer must be FrozenBerniniT2VScorer")
    prompts = _validate_prompt_registry(prompt_by_branch)
    if object_sha256(prompts) != scorer.prompt_registry_digest:
        raise PairV5NativeBridgeError("prompt registry differs from frozen text conditions")
    if scorer.training or scorer._packet_key is not None:
        raise PairV5NativeBridgeError("frozen scorer must be idle and in eval mode")
    recording = _RecordingFrozenScorer(scorer)
    try:
        energy = mace.evaluate_candidate_action_energy(
            clean,
            epsilon,
            sigma,
            prompts,
            recording,
            energy_epsilon=energy_epsilon,
        )
    except (mace.MACECandidateActionEnergyError, PairV5NativeBridgeError) as error:
        scorer.abort_packet()
        if isinstance(error, PairV5NativeBridgeError):
            raise
        raise PairV5NativeBridgeError(str(error)) from error
    if (
        len(recording.values) != len(mace.BRANCH_ORDER)
        or len(set(recording.input_ids)) != 1
    ):
        raise PairV5NativeBridgeError(
            "frozen critic did not reuse one candidate-own x_sigma/sigma packet"
        )
    velocities = torch.stack(recording.values, dim=0)
    expected_shape = (len(mace.BRANCH_ORDER), *tuple(int(x) for x in clean.shape))
    if tuple(int(item) for item in velocities.shape) != expected_shape:
        raise PairV5NativeBridgeError("T2V branch spatial velocity closure differs")
    serial_recomputed = torch.stack(
        [
            (value - energy.velocity_target)
            .square()
            .flatten(start_dim=1)
            .mean(dim=1)
            for value in recording.values
        ],
        dim=0,
    )
    if not torch.equal(serial_recomputed, energy.branch_energies):
        raise PairV5NativeBridgeError(
            "branch-serial spatial velocities do not bit-close to MACE energies"
        )
    batched_recomputed = (
        velocities - energy.velocity_target.unsqueeze(0)
    ).square().flatten(start_dim=2).mean(dim=2)
    closure_error = (batched_recomputed - energy.branch_energies).abs()
    closure_max_abs_error = float(closure_error.max().item())
    if not torch.allclose(
        batched_recomputed,
        energy.branch_energies,
        rtol=VELOCITY_ENERGY_CLOSURE_RTOL,
        atol=VELOCITY_ENERGY_CLOSURE_ATOL,
    ):
        raise PairV5NativeBridgeError(
            "spatial velocities do not close to MACE energies within the "
            "registered FP32 reduction bound: "
            f"max_abs_error={closure_max_abs_error:.9g}"
        )
    packet = scorer.last_packet_receipt
    if not isinstance(packet, Mapping):
        raise PairV5NativeBridgeError("frozen T2V packet receipt is unavailable")
    prediction_by_branch = {
        branch: velocities[index]
        for index, branch in enumerate(mace.BRANCH_ORDER)
    }
    try:
        conjunctive = phase_energy.evaluate_phase_conjunctive_energy(
            clean,
            epsilon,
            sigma,
            prediction_by_branch,
            phase_weight_commitment,
            registered_phase_weight_digest=registered_phase_weight_digest,
            frozen_t2v_receipt_digest=str(packet["digest"]),
            energy_epsilon=energy_epsilon,
        )
    except phase_energy.PairV5PhaseEnergyError as error:
        raise PairV5NativeBridgeError(str(error)) from error
    if not torch.equal(conjunctive.x_sigma, energy.x_sigma) or not torch.equal(
        conjunctive.velocity_label, energy.velocity_target
    ):
        raise PairV5NativeBridgeError(
            "MACE and phase cores did not reconstruct the same candidate state"
        )
    phase_global_closure_error = (
        conjunctive.global_branch_energies - energy.branch_energies
    ).abs()
    phase_global_closure_max_abs_error = float(
        phase_global_closure_error.max().item()
    )
    if not torch.allclose(
        conjunctive.global_branch_energies,
        energy.branch_energies,
        rtol=VELOCITY_ENERGY_CLOSURE_RTOL,
        atol=VELOCITY_ENERGY_CLOSURE_ATOL,
    ):
        raise PairV5NativeBridgeError(
            "global and phase-preserving energy reductions do not close within "
            "the registered FP32 reduction bound: "
            f"max_abs_error={phase_global_closure_max_abs_error:.9g}"
        )
    value = {
        "schema_version": T2V_SCORER_SCHEMA,
        "bridge_contract_digest": bridge_contract_receipt()["digest"],
        "packet_receipt_digest": packet["digest"],
        "phase_energy_contract_digest": phase_energy.contract_receipt()["digest"],
        "phase_weight_registration_digest": registered_phase_weight_digest,
        "phase_energy_receipt_digest": conjunctive.receipt["receipt_digest"],
        "prompt_registry_digest": scorer.prompt_registry_digest,
        "condition_registry_digest": scorer.condition_registry_digest,
        "frozen_model_receipt_digest": packet["frozen_model_receipt_digest"],
        "branch_order": list(mace.BRANCH_ORDER),
        "branch_velocity_shape": list(expected_shape),
        "branch_energy_shape": list(energy.branch_energies.shape),
        "reward_shape": list(energy.reward.shape),
        "velocity_energy_closure_verified": True,
        "velocity_energy_serial_closure_bit_exact": True,
        "velocity_energy_batched_closure_verified": True,
        "velocity_energy_closure_max_abs_error": closure_max_abs_error,
        "velocity_energy_closure_rtol": VELOCITY_ENERGY_CLOSURE_RTOL,
        "velocity_energy_closure_atol": VELOCITY_ENERGY_CLOSURE_ATOL,
        "phase_global_energy_closure_verified": True,
        "phase_global_energy_closure_max_abs_error": (
            phase_global_closure_max_abs_error
        ),
        "phase_global_energy_closure_rtol": VELOCITY_ENERGY_CLOSURE_RTOL,
        "phase_global_energy_closure_atol": VELOCITY_ENERGY_CLOSURE_ATOL,
        "same_state_reconstructed_by_mace_and_phase_cores": True,
        "phase_conjunction_applied_inside_native_bridge": True,
        "candidate_own_coordinate": True,
        "frozen_no_grad": True,
        "proposal_visual_data_consumed": False,
        "paired_target_consumed": False,
        "source_or_donor_consumed": False,
        "mask_flow_pose_track_consumed": False,
        "scientific_action_editing_claim": False,
    }
    receipt = {**value, "digest": object_sha256(value)}
    return FrozenT2VActionScore(velocities.detach(), energy, conjunctive, receipt)


def _validate_text_pair(
    cond_embeds: Any,
    uncond_embeds: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = (cond_embeds, uncond_embeds)
    if any(
        not isinstance(value, torch.Tensor)
        or tuple(int(item) for item in value.shape)
        != (
            1,
            runtime_contract.PINNED_TEXT_TOKENS,
            runtime_contract.PINNED_TEXT_DIM,
        )
        or value.device != device
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        for value in values
    ):
        raise PairV5NativeBridgeError(
            "RV2V conditional/unconditional text must be frozen device-local [1,512,4096]"
        )
    if tuple(cond_embeds.shape) != tuple(uncond_embeds.shape):
        raise PairV5NativeBridgeError("RV2V text geometry differs")
    if torch.equal(cond_embeds, uncond_embeds):
        raise PairV5NativeBridgeError("conditional and unconditional text are identical")
    return cond_embeds, uncond_embeds


def _native_schedule_coordinate(
    sigma: Any,
    timestep: Any,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if (
        not isinstance(sigma, torch.Tensor)
        or not isinstance(timestep, torch.Tensor)
        or sigma.dtype != torch.float32
        or timestep.dtype != torch.float32
        or sigma.numel() != 1
        or timestep.numel() != 1
        or sigma.device != timestep.device
        or sigma.requires_grad
        or timestep.requires_grad
        or sigma.grad_fn is not None
        or timestep.grad_fn is not None
        or not bool(torch.isfinite(sigma).all().item())
        or not bool(torch.isfinite(timestep).all().item())
    ):
        raise PairV5NativeBridgeError(
            "native RV2V sigma/timestep must be detached device-local FP32 scalars"
        )
    observed_timestep = float(timestep.item())
    try:
        index = tuple(float(value) for value in native.NATIVE_UNIPC40_TIMESTEPS).index(
            observed_timestep
        )
    except ValueError as error:
        raise PairV5NativeBridgeError("timestep is outside native exact40") from error
    expected_sigma = torch.tensor(
        [native.NATIVE_UNIPC40_SIGMAS[index]],
        dtype=torch.float32,
        device=sigma.device,
    )
    if _fp32_bits(sigma, label="sigma") != _fp32_bits(
        expected_sigma, label="expected sigma"
    ):
        raise PairV5NativeBridgeError(
            "sigma/timestep do not identify the same native exact40 coordinate"
        )
    return sigma.reshape(1), timestep.reshape(1), index


def _route_context(
    adapter: Any,
    *,
    transformer: nn.Module,
    branch: native.NativeRV2VBranch,
    sequence_parallel_rank: int,
    sequence_parallel_size: int,
    sigma_schedule_index: int,
    enabled: bool,
) -> Any:
    if adapter is None:
        return nullcontext()
    if getattr(adapter, "transformer", None) is not transformer:
        raise PairV5NativeBridgeError("adapter is bound to a different transformer")
    route_method = getattr(adapter, "route", None)
    if not callable(route_method):
        raise PairV5NativeBridgeError("adapter must expose a route context")
    if isinstance(adapter, cio_adapter.NativeTargetAdapterHandle):
        route = cio_adapter.NativeTargetRoute(
            total_tokens=branch.total_tokens,
            condition_tokens=branch.condition_tokens,
            sequence_parallel_rank=sequence_parallel_rank,
            sequence_parallel_size=sequence_parallel_size,
            branch_name=branch.name,
            enabled=enabled,
        )
    else:
        registration = _closed_action_adapter_registration(adapter)
        if registration is not None:
            return registration.route_factory(
                adapter=adapter,
                branch=branch,
                sequence_parallel_rank=sequence_parallel_rank,
                sequence_parallel_size=sequence_parallel_size,
                sigma_schedule_index=sigma_schedule_index,
                enabled=enabled,
            )
        # PAIR's action adapter is kept in a separate module so this bridge can
        # be imported and tested without installing it.  Its handle advertises
        # ``gate_name`` and consumes the exact40 schedule index in its route.
        try:
            import pair_v5_action_adapter as action_adapter
        except ImportError as error:  # pragma: no cover - integration failure
            raise PairV5NativeBridgeError("PAIR-v5 action adapter is unavailable") from error
        if not isinstance(adapter, action_adapter.PairV5ActionAdapterHandle):
            raise PairV5NativeBridgeError("adapter type is outside the closed route registry")
        route = action_adapter.PairV5ActionRoute(
            total_tokens=branch.total_tokens,
            condition_tokens=branch.condition_tokens,
            sequence_parallel_rank=sequence_parallel_rank,
            sequence_parallel_size=sequence_parallel_size,
            branch_name=branch.name,
            sigma_schedule_index=sigma_schedule_index,
            enabled=enabled,
        )
    return route_method(route)


def _action_adapter_gate(
    adapter: Any,
    *,
    sigma_schedule_index: int,
) -> Optional[tuple[str, float]]:
    """Return the sealed Action-LoRA gate, or ``None`` for another route.

    The final two exact40 coordinates deliberately bypass Action-LoRA.  The
    native bridge must therefore distinguish an intentionally frozen
    low-sigma student field from an accidentally detached high/mid-sigma
    field.  Type checking remains closed and is repeated by
    :func:`_route_context` at the actual model call.
    """

    if adapter is None or isinstance(adapter, cio_adapter.NativeTargetAdapterHandle):
        return None
    registration = _closed_action_adapter_registration(adapter)
    if registration is not None:
        gate_name, gate_weight = registration.gate_factory(
            adapter=adapter,
            sigma_schedule_index=sigma_schedule_index,
        )
        if not isinstance(gate_name, str) or not math.isfinite(float(gate_weight)):
            raise PairV5NativeBridgeError("registered action adapter gate differs")
        return gate_name, float(gate_weight)
    try:
        import pair_v5_action_adapter as action_adapter
    except ImportError as error:  # pragma: no cover - integration failure
        raise PairV5NativeBridgeError("PAIR-v5 action adapter is unavailable") from error
    if not isinstance(adapter, action_adapter.PairV5ActionAdapterHandle):
        raise PairV5NativeBridgeError("adapter type is outside the closed route registry")
    try:
        gate_name, gate_weight = action_adapter.sigma_gate(sigma_schedule_index)
    except action_adapter.PairV5ActionAdapterError as error:
        raise PairV5NativeBridgeError(str(error)) from error
    return gate_name, float(gate_weight)


def _guided_packed_prediction(
    diffusion: nn.Module,
    transformer: nn.Module,
    pack: native.NativeRV2VPack,
    *,
    timestep: torch.Tensor,
    cond_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    adapter: Any,
    sequence_parallel_rank: int,
    sequence_parallel_size: int,
    sigma_schedule_index: int,
    adapter_enabled: bool,
    no_grad: bool,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    if bool(getattr(transformer, "gradient_checkpointing", False)) or bool(
        getattr(transformer, "is_gradient_checkpointing", False)
    ):
        raise PairV5NativeBridgeError(
            "gradient checkpointing is forbidden for branch-local routes"
        )
    rows = (
        ("none_uncond", pack.none, uncond_embeds),
        ("V_uncond", pack.video, uncond_embeds),
        ("VI_uncond", pack.video_image, uncond_embeds),
        ("VI_cond", pack.video_image, cond_embeds),
    )
    if tuple(name for name, _, _ in rows) != tuple(
        guidance.guidance_receipt()["forward_order"]
    ):
        raise PairV5NativeBridgeError("native RV2V forward order differs")
    context = torch.no_grad() if no_grad else nullcontext()
    components: dict[str, torch.Tensor] = {}
    with context:
        for name, branch, text in rows:
            with _route_context(
                adapter,
                transformer=transformer,
                branch=branch,
                sequence_parallel_rank=sequence_parallel_rank,
                sequence_parallel_size=sequence_parallel_size,
                sigma_schedule_index=sigma_schedule_index,
                enabled=adapter_enabled,
            ):
                components[name] = native.forward_native_target_branch(
                    diffusion,
                    branch,
                    timestep=timestep,
                    cond_embeds=text,
                )
        none = components["none_uncond"]
        video = components["V_uncond"]
        vi_u = components["VI_uncond"]
        vi_c = components["VI_cond"]
        guided = (
            none
            + guidance.OMEGA_VIDEO * (video - none)
            + guidance.OMEGA_IMAGE * (vi_u - video)
            + guidance.OMEGA_TEXT * (vi_c - vi_u)
        )
    if not isinstance(guided, torch.Tensor) or not bool(
        torch.isfinite(guided).all().item()
    ):
        raise PairV5NativeBridgeError("native RV2V guided prediction is invalid")
    return guided, components


@dataclass(frozen=True)
class NativeRV2V4PolicyPair:
    """Trainable student and detached reference exact-81 velocity fields."""

    student_velocity: torch.Tensor
    reference_velocity: torch.Tensor
    student_components: Mapping[str, torch.Tensor]
    reference_components: Mapping[str, torch.Tensor]
    receipt: Mapping[str, Any]


def forward_native_rv2v4_policy_pair(
    student_diffusion: nn.Module,
    student_transformer: nn.Module,
    frozen_reference_diffusion: nn.Module,
    frozen_reference_transformer: nn.Module,
    condition_video: torch.Tensor,
    image_references: Sequence[torch.Tensor],
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
    cond_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    *,
    student_adapter: Any = None,
    reference_adapter: Any = None,
    sequence_parallel_rank: int,
    sequence_parallel_size: int = 4,
) -> NativeRV2V4PolicyPair:
    """Run one deployment-matched RV2V-4 student/reference policy query."""

    if not all(
        isinstance(value, nn.Module)
        for value in (
            student_diffusion,
            student_transformer,
            frozen_reference_diffusion,
            frozen_reference_transformer,
        )
    ):
        raise PairV5NativeBridgeError("student/reference runtime objects must be modules")
    if not callable(getattr(student_diffusion, "shared_step", None)) or not callable(
        getattr(frozen_reference_diffusion, "shared_step", None)
    ):
        raise PairV5NativeBridgeError("both policies must expose shared_step")
    state = _validate_exact81_spatial(
        x_sigma, label="native RV2V x_sigma", detached_fp32=True
    )
    video = _validate_exact81_spatial(
        condition_video, label="condition video", detached_fp32=True
    )
    if tuple(video.shape) != tuple(state.shape) or video.device != state.device:
        raise PairV5NativeBridgeError("condition video and x_sigma geometry/device differ")
    refs = _validate_image_references(image_references, video=video)
    conditional, unconditional = _validate_text_pair(
        cond_embeds, uncond_embeds, device=state.device
    )
    sigma_value, timestep_value, schedule_index = _native_schedule_coordinate(
        sigma, timestep
    )
    student_action_gate = _action_adapter_gate(
        student_adapter, sigma_schedule_index=schedule_index
    )
    reference_action_gate = _action_adapter_gate(
        reference_adapter, sigma_schedule_index=schedule_index
    )
    if sigma_value.device != state.device:
        raise PairV5NativeBridgeError("native coordinate and RV2V state use different devices")
    if (
        isinstance(sequence_parallel_rank, bool)
        or not isinstance(sequence_parallel_rank, int)
        or isinstance(sequence_parallel_size, bool)
        or not isinstance(sequence_parallel_size, int)
        or sequence_parallel_size not in (1, 4)
        or not 0 <= sequence_parallel_rank < sequence_parallel_size
    ):
        raise PairV5NativeBridgeError("only SP1 tests and production SP4 are supported")
    if student_transformer is frozen_reference_transformer:
        if student_adapter is None or reference_adapter is not student_adapter:
            raise PairV5NativeBridgeError(
                "a shared transformer requires one explicit adapter handle for enabled/disabled routes"
            )
    elif reference_adapter is None:
        if frozen_reference_diffusion.training or frozen_reference_transformer.training:
            raise PairV5NativeBridgeError("separate frozen reference must be in eval mode")
        if any(
            parameter.requires_grad
            for parameter in frozen_reference_diffusion.parameters()
        ) or any(
            parameter.requires_grad
            for parameter in frozen_reference_transformer.parameters()
        ):
            raise PairV5NativeBridgeError("separate reference policy is trainable")

    student_pack = native.build_native_rv2v_pack(
        student_transformer,
        donor_video=video,
        image_references=refs,
        noisy_target=state,
    )
    reference_pack = native.build_native_rv2v_pack(
        frozen_reference_transformer,
        donor_video=video,
        image_references=refs,
        noisy_target=state,
    )
    student_guided, student_components = _guided_packed_prediction(
        student_diffusion,
        student_transformer,
        student_pack,
        timestep=timestep_value,
        cond_embeds=conditional,
        uncond_embeds=unconditional,
        adapter=student_adapter,
        sequence_parallel_rank=sequence_parallel_rank,
        sequence_parallel_size=sequence_parallel_size,
        sigma_schedule_index=schedule_index,
        adapter_enabled=True,
        no_grad=False,
    )
    reference_guided, reference_components = _guided_packed_prediction(
        frozen_reference_diffusion,
        frozen_reference_transformer,
        reference_pack,
        timestep=timestep_value,
        cond_embeds=conditional,
        uncond_embeds=unconditional,
        adapter=reference_adapter,
        sequence_parallel_rank=sequence_parallel_rank,
        sequence_parallel_size=sequence_parallel_size,
        sigma_schedule_index=schedule_index,
        adapter_enabled=False,
        no_grad=True,
    )
    student_velocity = _unpack_spatial_velocity(
        student_guided.float(), video_shape=state.shape
    )
    reference_velocity = _unpack_spatial_velocity(
        reference_guided.float(), video_shape=state.shape
    ).detach()
    student_action_route_active = (
        student_action_gate is None or student_action_gate[1] > 0.0
    )
    if student_action_route_active:
        if not student_velocity.requires_grad or student_velocity.grad_fn is None:
            raise PairV5NativeBridgeError(
                "student velocity is detached from its trainable policy"
            )
    elif student_velocity.requires_grad or student_velocity.grad_fn is not None:
        raise PairV5NativeBridgeError(
            "low-sigma Action-LoRA route did not return the frozen base policy"
        )
    if reference_velocity.requires_grad or reference_velocity.grad_fn is not None:
        raise PairV5NativeBridgeError("reference velocity is not detached")
    if tuple(student_velocity.shape) != tuple(state.shape) or tuple(
        reference_velocity.shape
    ) != tuple(state.shape):
        raise PairV5NativeBridgeError("RV2V spatial velocity geometry differs")
    if set(student_components) != set(EXPANDED_GUIDANCE_COEFFICIENTS) or set(
        reference_components
    ) != set(EXPANDED_GUIDANCE_COEFFICIENTS):
        raise PairV5NativeBridgeError("RV2V component closure differs")

    student_pack_receipt = student_pack.receipt()
    reference_pack_receipt = reference_pack.receipt()
    branch_geometry = {
        name: {
            "total_tokens": getattr(student_pack, attr).total_tokens,
            "condition_tokens": getattr(student_pack, attr).condition_tokens,
        }
        for name, attr in (
            ("none", "none"),
            ("V", "video"),
            ("I", "image"),
            ("VI", "video_image"),
        )
    }
    value = {
        "schema_version": RV2V_POLICY_PAIR_SCHEMA,
        "bridge_contract_digest": bridge_contract_receipt()["digest"],
        "frame_count": FRAME_COUNT,
        "latent_shape": list(state.shape),
        "reference_count": len(refs),
        "reference_latent_phases": [int(item.shape[2]) for item in refs],
        "native_rv2v4_reference_contract_digest": (
            native.native_rv2v4_reference_contract()["digest"]
        ),
        "student_pack_digest": student_pack_receipt["digest"],
        "reference_pack_digest": reference_pack_receipt["digest"],
        "branch_geometry": branch_geometry,
        "guidance_receipt_digest": guidance.guidance_receipt()["digest"],
        "expanded_guidance_coefficients_hex": {
            name: float(coefficient).hex()
            for name, coefficient in EXPANDED_GUIDANCE_COEFFICIENTS.items()
        },
        "guidance_coefficient_sum_hex": float(
            sum(EXPANDED_GUIDANCE_COEFFICIENTS.values())
        ).hex(),
        "sigma_schedule_index": schedule_index,
        "sigma_float32_bits_hex": _fp32_bits(sigma_value, label="sigma"),
        "timestep_float32_bits_hex": _fp32_bits(
            timestep_value, label="timestep"
        ),
        "student_prediction_trainable": student_action_route_active,
        "reference_prediction_detached": True,
        "student_adapter_route_enabled": (
            student_adapter is not None and student_action_route_active
        ),
        "student_action_adapter_gate": (
            None if student_action_gate is None else student_action_gate[0]
        ),
        "student_action_adapter_gate_weight_hex": (
            None if student_action_gate is None else student_action_gate[1].hex()
        ),
        "reference_action_adapter_gate": (
            None if reference_action_gate is None else reference_action_gate[0]
        ),
        "reference_action_adapter_gate_weight_hex": (
            None if reference_action_gate is None else reference_action_gate[1].hex()
        ),
        "reference_adapter_route_explicitly_disabled": reference_adapter is not None,
        "sequence_parallel_size": sequence_parallel_size,
        "proposal_visual_data_consumed": False,
        "paired_target_consumed": False,
        "mask_flow_pose_track_consumed": False,
        "scientific_action_editing_claim": False,
    }
    receipt = {**value, "digest": object_sha256(value)}
    return NativeRV2V4PolicyPair(
        student_velocity=student_velocity,
        reference_velocity=reference_velocity,
        student_components=student_components,
        reference_components={
            name: component.detach() for name, component in reference_components.items()
        },
        receipt=receipt,
    )


def bridge_contract_receipt() -> Mapping[str, Any]:
    """Return the digest-bound static geometry and information-flow contract."""

    scorer_signatures = {
        "FrozenBerniniT2VScorer.__init__": set(
            inspect.signature(FrozenBerniniT2VScorer.__init__).parameters
        ),
        "FrozenBerniniT2VScorer.forward": set(
            inspect.signature(FrozenBerniniT2VScorer.forward).parameters
        ),
        "score_frozen_t2v_action_energy": set(
            inspect.signature(score_frozen_t2v_action_energy).parameters
        ),
    }
    offending = {
        name: sorted(parameters & FORBIDDEN_SCORER_INPUT_NAMES)
        for name, parameters in scorer_signatures.items()
        if parameters & FORBIDDEN_SCORER_INPUT_NAMES
    }
    if offending:
        raise PairV5NativeBridgeError(
            f"frozen T2V scorer exposes forbidden inputs: {offending}"
        )
    if not math.isclose(
        sum(EXPANDED_GUIDANCE_COEFFICIENTS.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise PairV5NativeBridgeError("expanded RV2V coefficients do not sum to one")
    guidance_contract = guidance.guidance_receipt()
    value = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "latent_channels": LATENT_CHANNELS,
        "latent_phases": LATENT_PHASES,
        "patch_size": list(PATCH_SIZE),
        "patch_output_dim": runtime_contract.PINNED_PATCH_DIM,
        "transformer_inner_dim": runtime_contract.PINNED_INNER_DIM,
        "rotary_complex_dim": runtime_contract.PINNED_ROPE_DIM,
        "text_geometry": [
            1,
            runtime_contract.PINNED_TEXT_TOKENS,
            runtime_contract.PINNED_TEXT_DIM,
        ],
        "mace_schema": mace.SCHEMA_VERSION,
        "mace_branch_order": list(mace.BRANCH_ORDER),
        "phase_energy_schema": phase_energy.SCHEMA_VERSION,
        "phase_energy_contract_digest": phase_energy.contract_receipt()["digest"],
        "phase_milestone_order": list(phase_energy.MILESTONE_ORDER),
        "t2v_input_fields": ["clean_candidate", "epsilon", "sigma", "prompt_by_branch"],
        "t2v_model_call_fields": ["x_sigma", "sigma", "prompt"],
        "t2v_target_source_id": T2V_TARGET_SOURCE_ID,
        "t2v_target_tail": "direct_storage_view_of_target_only_full_prediction",
        "t2v_timestep": "float32_1000_times_physical_sigma_no_shift",
        "t2v_patch_calls_per_ten_prompts": 1,
        "t2v_condition_registry_revalidated_before_each_packet": True,
        "t2v_phase_energy_handoff": "inside_bridge_without_caller_prediction_slot",
        "native_pack_schema": native.SCHEMA_VERSION,
        "native_rv2v4_reference_contract_digest": (
            native.native_rv2v4_reference_contract()["digest"]
        ),
        "native_exact40_schedule_digest": (
            native.native_unipc40_schedule_receipt()["digest"]
        ),
        "rv2v_guidance_schema": guidance.SCHEMA_VERSION,
        "rv2v_guidance_receipt_digest": guidance_contract["digest"],
        "rv2v_guidance_forward_order": list(guidance_contract["forward_order"]),
        "rv2v_expanded_coefficients_hex": {
            name: float(coefficient).hex()
            for name, coefficient in EXPANDED_GUIDANCE_COEFFICIENTS.items()
        },
        "rv2v_reference_count": native.REFERENCE_COUNT,
        "rv2v_reference_frame_indices": [0, 27, 53, 80],
        "rv2v_student_reference_same_exact40_coordinate": True,
        "rv2v_action_adapter_receives_exact40_schedule_index": True,
        "rv2v_low_sigma_action_adapter_is_direct_frozen_base": True,
        "proposal_role": "offline_prompt_calibration_provenance_only",
        "proposal_visual_data_consumed": False,
        "paired_target_consumed": False,
        "mask_flow_pose_track_trajectory_consumed": False,
        "scientific_action_editing_claim": False,
    }
    return {**value, "digest": object_sha256(value)}


__all__ = [
    "EXPANDED_GUIDANCE_COEFFICIENTS",
    "FORBIDDEN_SCORER_INPUT_NAMES",
    "FRAME_COUNT",
    "FrozenBerniniT2VScorer",
    "FrozenT2VActionScore",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "NativeRV2V4PolicyPair",
    "PATCH_SIZE",
    "PairV5NativeBridgeError",
    "RV2V_POLICY_PAIR_SCHEMA",
    "SCHEMA_VERSION",
    "T2V_SCORER_SCHEMA",
    "bridge_contract_receipt",
    "forward_native_rv2v4_policy_pair",
    "object_sha256",
    "register_closed_action_adapter_type",
    "score_frozen_t2v_action_energy",
]
