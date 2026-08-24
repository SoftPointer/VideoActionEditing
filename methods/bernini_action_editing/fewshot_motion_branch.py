"""Training-only EPMC integration at Bernini's real attention-head boundary.

The V11 CPMR branch supplies frozen visual Q/K/V/O weights and a proposal
carrier.  This module adds only a 36-parameter per-example motion code:

* 20 target-phase logits (phase 0 is structurally absent); and
* 16 block logits, each tied across Bernini's 12 actual attention heads.

The code gates the direct ``varlen_attention`` result shaped
``[1, local_q, 12, 128]`` before head flattening and ``to_out``.  It never
pretends that chunks of a pre-projection 1536-vector are attention heads.

The privileged target/support videos are allowed to supervise code inversion
outside this module during training.  They are not arguments to the eventual
source+instruction predictor, and no mask, flow, pose, track, trajectory, or
target condition is introduced at inference.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

import counterfactual_proposal_motion_branch as cpmr
import fewshot_privileged_motion_code as epmc


METHOD_NAME = "fewshot-epmc-cpmr-head-gating"
SCHEMA_VERSION = "bernini-fewshot-motion-branch-v1"

GLOBAL_VISUAL_TOKENS = cpmr.GLOBAL_VISUAL_TOKENS
SOURCE_VISUAL_TOKENS = cpmr.SOURCE_VISUAL_TOKENS
TARGET_VISUAL_TOKENS = cpmr.TARGET_VISUAL_TOKENS
LATENT_PHASES = epmc.LATENT_PHASES
TARGET_TOKENS_PER_PHASE = TARGET_VISUAL_TOKENS // LATENT_PHASES
TIED_CODE_DIMENSION = epmc.NONZERO_PHASES + epmc.MOTION_BLOCKS
INITIAL_ACTION_GATE = 0.10
OUTER_CPMR_GATE = 0.10
RAW_TEXT_CONDITION_WIDTH = 4_096
OUTPUT_PATCH_WIDTH = 64

if TARGET_TOKENS_PER_PHASE * LATENT_PHASES != TARGET_VISUAL_TOKENS:
    raise RuntimeError("target token count is not divisible by 21 latent phases")
if (GLOBAL_VISUAL_TOKENS, TARGET_TOKENS_PER_PHASE, TIED_CODE_DIMENSION) != (
    39_060,
    930,
    36,
):
    raise RuntimeError("the pinned Bernini 81-frame geometry changed")


class FewShotMotionBranchContractError(RuntimeError):
    """Raised when the training-only motion-code surface is ambiguous."""


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise FewShotMotionBranchContractError(f"{label} must be a torch.Tensor")
    return tuple(int(item) for item in value.shape)


def _lengths(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.device.type == "meta":
            raise FewShotMotionBranchContractError(f"{label} cannot be meta")
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FewShotMotionBranchContractError(f"{label} must be a sequence")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FewShotMotionBranchContractError(
            f"{label} must contain exact integers"
        ) from error
    if any(item <= 0 for item in result):
        raise FewShotMotionBranchContractError(
            f"{label} must contain positive integers"
        )
    return result


def _positive_zero(name: str, value: torch.Tensor) -> None:
    detached = value.detach().contiguous()
    if int(torch.count_nonzero(detached).item()) != 0:
        raise FewShotMotionBranchContractError(f"{name} must be exact zero")
    if int(torch.count_nonzero(detached.reshape(-1).view(torch.uint8)).item()) != 0:
        raise FewShotMotionBranchContractError(
            f"{name} must be byte-exact positive zero"
        )


class TiedHeadEpisodicMotionCode(nn.Module):
    """The first-canary 36D code: 20 phase + 16 tied-head parameters."""

    def __init__(
        self,
        *,
        batch_size: int = 1,
        initial_action_gate: float = INITIAL_ACTION_GATE,
    ) -> None:
        super().__init__()
        if isinstance(batch_size, bool) or batch_size != 1:
            raise FewShotMotionBranchContractError(
                "the first canary defines exactly batch_size=1"
            )
        if isinstance(initial_action_gate, bool):
            raise FewShotMotionBranchContractError(
                "initial_action_gate must be a finite scalar"
            )
        try:
            initial = float(initial_action_gate)
        except (TypeError, ValueError, OverflowError) as error:
            raise FewShotMotionBranchContractError(
                "initial_action_gate must be a finite scalar"
            ) from error
        if not math.isfinite(initial) or not 0.0 <= initial < 1.0:
            raise FewShotMotionBranchContractError(
                "initial_action_gate must lie in [0,1)"
            )
        logit = math.atanh(initial)
        self.phase_logits_nonzero = nn.Parameter(
            torch.full((1, epmc.NONZERO_PHASES), logit, dtype=torch.float32)
        )
        self.block_logits = nn.Parameter(
            torch.full((1, epmc.MOTION_BLOCKS), logit, dtype=torch.float32)
        )

    def forward(self) -> epmc.MotionCode:
        block_head_logits = self.block_logits[:, :, None].expand(
            1, epmc.MOTION_BLOCKS, epmc.ATTENTION_HEADS
        )
        return epmc.decode_bounded_motion_code(
            self.phase_logits_nonzero, block_head_logits
        )

    def receipt(self) -> dict[str, Any]:
        code = self()
        return {
            "method": METHOD_NAME,
            "schema_version": SCHEMA_VERSION,
            "parameterization": "20_phase_plus_16_block_tied_across_12_heads",
            "trainable_dimension": sum(
                int(parameter.numel()) for parameter in self.parameters()
            ),
            "phase0_structurally_absent": True,
            "motion_code": code.audit_receipt(),
        }


def canonical_tied_noop_motion_code(
    *, device: torch.device | str | None = None
) -> epmc.MotionCode:
    """Return the byte-exact all-zero ablation; it has no trainable tensor."""

    return epmc.canonical_noop_motion_code(1, device=device)


def global_query_phase_ids(*, device: torch.device | str) -> torch.Tensor:
    """Return source=-1 then target phases 0..20 in packed Bernini order."""

    result = torch.full(
        (1, GLOBAL_VISUAL_TOKENS, 1),
        -1,
        dtype=torch.int64,
        device=device,
    )
    target = torch.arange(LATENT_PHASES, dtype=torch.int64, device=device)
    target = target.repeat_interleave(TARGET_TOKENS_PER_PHASE)
    result[:, SOURCE_VISUAL_TOKENS:, 0] = target
    return result


def local_query_phase_ids(
    *,
    device: torch.device | str,
    expected_local_queries: int,
    padding_tensor_fn: Callable[..., torch.Tensor],
    slice_input_tensor_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Use Bernini's official pad/slice functions for full or Ulysses-4 IDs."""

    if expected_local_queries not in (
        GLOBAL_VISUAL_TOKENS,
        GLOBAL_VISUAL_TOKENS // 4,
    ):
        raise FewShotMotionBranchContractError(
            "query phase IDs define only full sequence or Ulysses-4 shards"
        )
    if not callable(padding_tensor_fn) or not callable(slice_input_tensor_fn):
        raise FewShotMotionBranchContractError(
            "query phase IDs require official callable pad/slice operations"
        )
    global_ids = global_query_phase_ids(device=device)
    padded = padding_tensor_fn(global_ids, dim=1)
    local = slice_input_tensor_fn(padded, dim=1)
    if _shape(local, label="local query phase IDs") != (
        1,
        expected_local_queries,
        1,
    ):
        raise FewShotMotionBranchContractError(
            "official pad/slice returned an unexpected local phase-ID layout"
        )
    local = local[0, :, 0].contiguous()
    if local.dtype != torch.int64 or local.device != torch.device(device):
        raise FewShotMotionBranchContractError(
            "official pad/slice changed phase-ID dtype or device"
        )
    if bool(((local < -1) | (local >= LATENT_PHASES)).any().item()):
        raise FewShotMotionBranchContractError("local phase IDs escaped -1..20")
    return local


@dataclass(frozen=True)
class _FewShotCodeInvocation:
    motion_code: epmc.MotionCode
    allowed_motion_module_ids: frozenset[int]


_CURRENT_CODE: contextvars.ContextVar[Optional[_FewShotCodeInvocation]] = (
    contextvars.ContextVar("fewshot_epmc_motion_code", default=None)
)


@contextlib.contextmanager
def _fewshot_code_invocation(
    motion_code: epmc.MotionCode,
    *,
    motion_modules: Sequence[cpmr.MotionCrossAttention],
) -> Iterator[None]:
    if _CURRENT_CODE.get() is not None:
        raise FewShotMotionBranchContractError(
            "nested few-shot motion-code invocations are forbidden"
        )
    motion_code.validate()
    module_ids = frozenset(id(item) for item in motion_modules)
    if not module_ids or len(module_ids) != len(tuple(motion_modules)):
        raise FewShotMotionBranchContractError(
            "motion-code invocation requires distinct installed modules"
        )
    token = _CURRENT_CODE.set(_FewShotCodeInvocation(motion_code, module_ids))
    try:
        yield
    finally:
        _CURRENT_CODE.reset(token)


class FewShotMotionCrossAttention(cpmr.MotionCrossAttention):
    """Frozen CPMR clone with true post-attention EPMC head gating."""

    def train(self, mode: bool = True) -> "FewShotMotionCrossAttention":
        # The copied branch is a fixed motion basis.  Parent ``model.train()``
        # must not reactivate any copied dropout.
        del mode
        super().train(False)
        return self

    def _centered_output_projection(self, flattened: torch.Tensor) -> torch.Tensor:
        projection = self.to_out[0]
        if isinstance(projection, nn.Linear):
            output = F.linear(flattened, projection.weight, bias=None)
        else:
            # CPU mocks and an upstream-compatible affine wrapper may not be an
            # nn.Linear.  Subtracting f(0) removes the otherwise leaking bias.
            output = projection(flattened) - projection(torch.zeros_like(flattened))
        dropout = self.to_out[1]
        if isinstance(dropout, nn.Dropout):
            output = F.dropout(output, p=float(dropout.p), training=False)
        else:
            output = dropout(output)
        return output

    def gate_and_merge_projected_heads(
        self,
        projected_motion_heads: torch.Tensor,
        query_phase_ids: torch.Tensor,
        motion_code: epmc.MotionCode,
    ) -> torch.Tensor:
        """Gate actual heads, remove output bias, and preserve structural zeros."""

        input_dtype = projected_motion_heads.dtype
        if not torch.is_floating_point(projected_motion_heads):
            raise FewShotMotionBranchContractError(
                "projected motion heads must have floating dtype"
            )
        gated = epmc.gate_projected_motion_heads(
            projected_motion_heads.float(),
            query_phase_ids,
            motion_code,
            block_index=self.block_index,
            audit_digests=False,
        )
        flattened = gated.flattened_output().to(dtype=input_dtype)
        output = self._centered_output_projection(flattened)
        active = query_phase_ids > 0
        output = torch.where(
            active[None, :, None], output, torch.zeros_like(output)
        ).contiguous()
        if bool((~active).any().item()):
            _positive_zero(
                "source/target-phase0 projected residual", output[:, ~active]
            )
        return output

    def _merge_projected_motion_heads(
        self,
        projected_motion_heads: torch.Tensor,
        *,
        local_target_mask: torch.Tensor,
        padding_tensor_fn: Callable[..., torch.Tensor],
        slice_input_tensor_fn: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        invocation = _CURRENT_CODE.get()
        if invocation is None or id(self) not in invocation.allowed_motion_module_ids:
            raise FewShotMotionBranchContractError(
                "few-shot motion branch ran without its exact one-call code binding"
            )
        expected_local = int(projected_motion_heads.shape[1])
        phase_ids = local_query_phase_ids(
            device=projected_motion_heads.device,
            expected_local_queries=expected_local,
            padding_tensor_fn=padding_tensor_fn,
            slice_input_tensor_fn=slice_input_tensor_fn,
        )
        expected_target = phase_ids >= 0
        if not torch.equal(
            expected_target[None, :, None], local_target_mask.bool()
        ):
            raise FewShotMotionBranchContractError(
                "phase IDs and official CPMR target mask disagree"
            )
        return self.gate_and_merge_projected_heads(
            projected_motion_heads, phase_ids, invocation.motion_code
        )


def install_fewshot_motion_branch(
    model: Any,
    *,
    motion_kwargs: Optional[Mapping[str, Any]] = None,
) -> cpmr.CPMRMotionPatchHandle:
    """Install frozen few-shot branches on the pinned blocks 0..15."""

    kwargs = dict(motion_kwargs or {})

    def factory(donor_attn1: Any, block_index: int) -> FewShotMotionCrossAttention:
        return FewShotMotionCrossAttention(
            donor_attn1, block_index=block_index, **kwargs
        )

    handle = cpmr.install_cpmr_motion_branch(model, motion_factory=factory)
    for module in handle.motion_modules:
        module.requires_grad_(False)
        module.eval()
    return handle


@contextlib.contextmanager
def fewshot_motion_code_context(
    *,
    patch_handle: cpmr.CPMRMotionPatchHandle,
    motion_code: epmc.MotionCode,
) -> Iterator[epmc.MotionCode]:
    """Bind a source+instruction code across a complete sampling call.

    Inference nests this context around ``cpmr_final_render_hook`` and
    ``model.sample``.  It carries no target/support video and no spatial oracle;
    the existing CPMR runtime still authenticates each of the 40 APG steps.
    """

    if not isinstance(patch_handle, cpmr.CPMRMotionPatchHandle):
        raise FewShotMotionBranchContractError("patch_handle has the wrong type")
    if patch_handle.restored:
        raise FewShotMotionBranchContractError("patch_handle was already restored")
    if tuple(patch_handle.indices) != cpmr.MOTION_BLOCK_INDICES or any(
        not isinstance(module, FewShotMotionCrossAttention)
        for module in patch_handle.motion_modules
    ):
        raise FewShotMotionBranchContractError(
            "motion code requires the exact blocks-0..15 few-shot patch"
        )
    motion_code.validate()
    parameter_devices = {
        parameter.device
        for module in patch_handle.motion_modules
        for parameter in module.parameters()
    }
    if len(parameter_devices) != 1 or motion_code.phase_gates.device not in parameter_devices:
        raise FewShotMotionBranchContractError(
            "motion code must share the installed branch device"
        )
    with _fewshot_code_invocation(
        motion_code, motion_modules=patch_handle.motion_modules
    ):
        yield motion_code


def _validate_frozen_surface(
    diffusion: Any,
    patch_handle: cpmr.CPMRMotionPatchHandle,
) -> None:
    if not isinstance(patch_handle, cpmr.CPMRMotionPatchHandle):
        raise FewShotMotionBranchContractError("patch_handle has the wrong type")
    if patch_handle.restored:
        raise FewShotMotionBranchContractError("patch_handle was already restored")
    if getattr(diffusion, "transformer_2", None) is not None:
        raise FewShotMotionBranchContractError(
            "few-shot canary requires the 1.3B single-expert decoder"
        )
    if getattr(diffusion, "transformer", None) is not patch_handle.transformer:
        raise FewShotMotionBranchContractError(
            "shared_step transformer is not the exact patched transformer_1"
        )
    if bool(getattr(patch_handle.transformer, "gradient_checkpointing", False)):
        raise FewShotMotionBranchContractError(
            "first code-inversion canary requires gradient checkpointing disabled; "
            "checkpoint recomputation would occur after the one-call code binding"
        )
    if tuple(patch_handle.indices) != cpmr.MOTION_BLOCK_INDICES:
        raise FewShotMotionBranchContractError(
            "few-shot canary requires the complete blocks-0..15 patch"
        )
    if any(
        not isinstance(module, FewShotMotionCrossAttention)
        for module in patch_handle.motion_modules
    ):
        raise FewShotMotionBranchContractError(
            "patch contains a non-few-shot motion module"
        )
    trainable = [
        name
        for name, parameter in patch_handle.transformer.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise FewShotMotionBranchContractError(
            "base/clone parameters must be frozen; trainable=" + ",".join(trainable[:4])
        )
    for module in patch_handle.motion_modules:
        if module.training:
            raise FewShotMotionBranchContractError(
                "frozen motion clones must remain in eval mode"
            )


def _validate_shared_step_inputs(
    *,
    model_id: str,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    cond_embeds: torch.Tensor,
    rotary_embs: torch.Tensor,
    batch_vae_seqlen: Sequence[int] | torch.Tensor,
    batch_text_seqlen: Optional[Sequence[int] | torch.Tensor],
    carrier: torch.Tensor,
    activity: torch.Tensor,
    motion_code: epmc.MotionCode,
    require_code_grad: bool,
) -> None:
    if not torch.is_grad_enabled():
        raise FewShotMotionBranchContractError(
            "few-shot training shared_step requires autograd enabled"
        )
    if model_id != "transformer_1":
        raise FewShotMotionBranchContractError(
            "few-shot canary must route through transformer_1"
        )
    if _shape(noisy_latents, label="noisy_latents") != (
        1,
        GLOBAL_VISUAL_TOKENS,
        cpmr.HIDDEN_SIZE,
    ):
        raise FewShotMotionBranchContractError(
            "noisy_latents must be exact batch-1 [1,39060,1536]"
        )
    if _shape(timesteps, label="timesteps") not in ((), (1,)):
        raise FewShotMotionBranchContractError("timesteps must contain one value")
    cond_shape = _shape(cond_embeds, label="cond_embeds")
    if (
        len(cond_shape) != 3
        or cond_shape[0] != 1
        or cond_shape[2] != RAW_TEXT_CONDITION_WIDTH
    ):
        raise FewShotMotionBranchContractError(
            "raw cond_embeds must be batch-1 [1,text,4096]"
        )
    rotary_shape = _shape(rotary_embs, label="rotary_embs")
    if not rotary_shape or rotary_shape[0] != 1:
        raise FewShotMotionBranchContractError("rotary_embs must be batch-1")
    if _lengths(batch_vae_seqlen, label="batch_vae_seqlen") != (
        GLOBAL_VISUAL_TOKENS,
    ):
        raise FewShotMotionBranchContractError(
            "source/target spans require batch_vae_seqlen=[39060]"
        )
    if batch_text_seqlen is not None:
        text_lengths = _lengths(batch_text_seqlen, label="batch_text_seqlen")
        if len(text_lengths) != 1 or text_lengths[0] != cond_shape[1]:
            raise FewShotMotionBranchContractError(
                "batch_text_seqlen must exactly span cond_embeds"
            )
    if carrier.requires_grad or noisy_latents.requires_grad or cond_embeds.requires_grad:
        raise FewShotMotionBranchContractError(
            "first canary permits gradients only through the 36D motion code"
        )
    if carrier.device != noisy_latents.device or carrier.dtype != noisy_latents.dtype:
        raise FewShotMotionBranchContractError(
            "carrier must match noisy_latents device and attention dtype"
        )
    motion_code.validate()
    if motion_code.phase_gates.device != noisy_latents.device:
        raise FewShotMotionBranchContractError(
            "motion code and shared_step must use the same device"
        )
    code_requires_grad = bool(
        motion_code.phase_gates.requires_grad
        or motion_code.block_head_gates.requires_grad
    )
    if code_requires_grad != require_code_grad:
        raise FewShotMotionBranchContractError(
            "motion-code grad state differs from the requested canary mode"
        )
    # Reuse the frozen CPMR carrier/activity validator at the exact hidden
    # dtype/device boundary.  Its return value also forbids an inactive action
    # carrier that would bypass head gating and deadlock code optimization.
    probe = cpmr.CPMRMotionInvocation(
        trajectory=cpmr.ACTION_PROPOSAL,
        polarity=cpmr.POSITIVE,
        prompt_object=cond_embeds,
        positive_noop_prompt_object=cond_embeds.clone(),
        gate=0.0,
        carrier=carrier,
        activity=activity,
    )
    if not cpmr._validate_motion_payload(probe, hidden_states=noisy_latents):
        raise FewShotMotionBranchContractError(
            "training carrier must contain active non-phase0 proposal motion"
        )


@dataclass(frozen=True)
class TrainingSharedStepResult:
    prediction: torch.Tensor
    binding_receipt: Mapping[str, Any]
    source_span: tuple[int, int]
    target_span: tuple[int, int]
    target_tokens_per_phase: int
    outer_cpmr_gate: float

    def receipt(self) -> dict[str, Any]:
        return {
            "method": METHOD_NAME,
            "schema_version": SCHEMA_VERSION,
            "training_only": True,
            "shared_step_calls": 1,
            "model_id": "transformer_1",
            "batch_size": 1,
            "global_visual_tokens": GLOBAL_VISUAL_TOKENS,
            "raw_text_condition_width": RAW_TEXT_CONDITION_WIDTH,
            "output_patch_width": OUTPUT_PATCH_WIDTH,
            "source_span": list(self.source_span),
            "target_span": list(self.target_span),
            "latent_phases": LATENT_PHASES,
            "target_tokens_per_phase": self.target_tokens_per_phase,
            "head_gating_point": "varlen_output_before_flatten_and_to_out",
            "runtime_full_head_digests": False,
            "runtime_digest_policy": (
                "disabled_in_hot_path; no lazy or synthetic tensor hashes"
            ),
            "outer_cpmr_gate": self.outer_cpmr_gate,
            "base_and_motion_clone_frozen": True,
            "gradient_checkpointing": False,
            "inference_conditions": ["source_video", "instruction"],
            "forbidden_inference_conditions": list(
                epmc.FORBIDDEN_INFERENCE_ARGUMENTS
            ),
            "conditioned_encoder_binding": dict(self.binding_receipt),
        }


def run_training_shared_step(
    diffusion: Any,
    *,
    patch_handle: cpmr.CPMRMotionPatchHandle,
    motion_code: epmc.MotionCode,
    carrier: torch.Tensor,
    activity: torch.Tensor,
    model_id: str,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    cond_embeds: torch.Tensor,
    rotary_embs: torch.Tensor,
    batch_vae_seqlen: Sequence[int] | torch.Tensor,
    batch_text_seqlen: Optional[Sequence[int] | torch.Tensor] = None,
    outer_cpmr_gate: float = OUTER_CPMR_GATE,
    require_code_grad: bool = True,
) -> TrainingSharedStepResult:
    """Run exactly one authenticated training ``GEN_Wanx22.shared_step``.

    This is intentionally not the 40-step APG sampling hook.  A target video
    may define the loss outside this call, but never enters the branch as a
    condition.  The only trainable state allowed by this first canary is the
    supplied 36D code.
    """

    _validate_frozen_surface(diffusion, patch_handle)
    _validate_shared_step_inputs(
        model_id=model_id,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        cond_embeds=cond_embeds,
        rotary_embs=rotary_embs,
        batch_vae_seqlen=batch_vae_seqlen,
        batch_text_seqlen=batch_text_seqlen,
        carrier=carrier,
        activity=activity,
        motion_code=motion_code,
        require_code_grad=require_code_grad,
    )
    if outer_cpmr_gate != OUTER_CPMR_GATE:
        raise FewShotMotionBranchContractError(
            f"first canary freezes outer_cpmr_gate={OUTER_CPMR_GATE}"
        )
    shared_step = getattr(diffusion, "shared_step", None)
    if not callable(shared_step):
        raise FewShotMotionBranchContractError(
            "diffusion core lacks callable shared_step"
        )
    kwargs: dict[str, Any] = {
        "model_id": model_id,
        "noisy_latents": noisy_latents,
        "timesteps": timesteps,
        "cond_embeds": cond_embeds,
        "rotary_embs": rotary_embs,
        "batch_vae_seqlen": batch_vae_seqlen,
    }
    if batch_text_seqlen is not None:
        kwargs["batch_text_seqlen"] = batch_text_seqlen
    try:
        inspect.signature(shared_step).bind(**kwargs)
    except (TypeError, ValueError) as error:
        raise FewShotMotionBranchContractError(
            "pinned GEN_Wanx22.shared_step signature differs"
        ) from error

    binding = patch_handle.new_conditioned_encoder_binding()
    invocation = cpmr.CPMRMotionInvocation(
        trajectory=cpmr.FINAL_RENDER,
        polarity=cpmr.POSITIVE,
        prompt_object=cond_embeds,
        positive_noop_prompt_object=cond_embeds,
        conditioned_encoder_binding=binding,
        gate=OUTER_CPMR_GATE,
        carrier=carrier,
        activity=activity,
    )
    with fewshot_motion_code_context(
        patch_handle=patch_handle, motion_code=motion_code
    ):
        with cpmr.cpmr_motion_invocation(
            invocation, encoder_hidden_states=cond_embeds
        ):
            prediction = shared_step(**kwargs)
    if _shape(prediction, label="shared_step prediction") != (
        1,
        GLOBAL_VISUAL_TOKENS,
        OUTPUT_PATCH_WIDTH,
    ):
        raise FewShotMotionBranchContractError(
            "shared_step prediction must be packed patches [1,39060,64]"
        )
    receipt = binding.receipt()
    if not (
        receipt.get("completed") is True
        and receipt.get("consumed") is True
        and receipt.get("aborted") is False
        and receipt.get("bound_tensor_released") is True
        and tuple(receipt.get("observed_block_indices", ()))
        == cpmr.MOTION_BLOCK_INDICES
    ):
        raise FewShotMotionBranchContractError(
            "conditioned encoder binding did not complete exactly once"
        )
    return TrainingSharedStepResult(
        prediction=prediction,
        binding_receipt=receipt,
        source_span=(0, SOURCE_VISUAL_TOKENS),
        target_span=(SOURCE_VISUAL_TOKENS, GLOBAL_VISUAL_TOKENS),
        target_tokens_per_phase=TARGET_TOKENS_PER_PHASE,
        outer_cpmr_gate=OUTER_CPMR_GATE,
    )


__all__ = [
    "FewShotMotionBranchContractError",
    "FewShotMotionCrossAttention",
    "GLOBAL_VISUAL_TOKENS",
    "INITIAL_ACTION_GATE",
    "LATENT_PHASES",
    "METHOD_NAME",
    "OUTER_CPMR_GATE",
    "OUTPUT_PATCH_WIDTH",
    "RAW_TEXT_CONDITION_WIDTH",
    "SCHEMA_VERSION",
    "SOURCE_VISUAL_TOKENS",
    "TARGET_TOKENS_PER_PHASE",
    "TARGET_VISUAL_TOKENS",
    "TIED_CODE_DIMENSION",
    "TiedHeadEpisodicMotionCode",
    "TrainingSharedStepResult",
    "canonical_tied_noop_motion_code",
    "fewshot_motion_code_context",
    "global_query_phase_ids",
    "install_fewshot_motion_branch",
    "local_query_phase_ids",
    "run_training_shared_step",
]
