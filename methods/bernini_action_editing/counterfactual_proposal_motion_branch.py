"""Frozen CPMR motion branch for the pinned Bernini-R 1.3B runtime.

This module implements only the V11 processor boundary.  It deliberately does
not implement proposal sampling, a runner, a trainer, or LoRA.  The production
path keeps Bernini's text cross-attention untouched and adds a separately
registered visual-domain cross-attention residual after the original processor.

The implementation is dependency-injected at the Bernini kernel boundary so the
same contracts can be exercised with CPU mocks.  With no injections it imports
the pinned ``bernini.attention`` and ``bernini.parallel`` functions lazily.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterator, Optional, Sequence

import torch
from torch import nn


GLOBAL_VISUAL_TOKENS = 39_060
SOURCE_VISUAL_TOKENS = 19_530
TARGET_VISUAL_TOKENS = 19_530
CARRIER_TOKENS = 1_344
LATENT_PHASES = 21
CARRIER_TOKENS_PER_PHASE = 64
HIDDEN_SIZE = 1_536
ATTENTION_HEADS = 12
ATTENTION_HEAD_DIM = 128
EXPECTED_BLOCK_COUNT = 30
MOTION_BLOCK_INDICES = tuple(range(16))
MOTION_MODULE_NAME = "cpmr_motion_cross_attention"

ACTION_PROPOSAL = "action_proposal"
NOOP_PROPOSAL = "noop_proposal"
FINAL_RENDER = "final_render"
TRAJECTORIES = (ACTION_PROPOSAL, NOOP_PROPOSAL, FINAL_RENDER)
POSITIVE = "positive"
UNCONDITIONAL = "unconditional"
POLARITIES = (POSITIVE, UNCONDITIONAL)
FROZEN_GATES = (0.0, 0.05, 0.10, 0.20, 0.40)


class CPMRMotionBranchContractError(ValueError):
    """Raised when the frozen V11 runtime contract is ambiguous or violated."""


def _exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise CPMRMotionBranchContractError(f"{label} must be an integer, got bool")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CPMRMotionBranchContractError(f"{label} must be an integer") from error
    if converted != value:
        raise CPMRMotionBranchContractError(f"{label} must be exact, got {value!r}")
    return converted


def _scalar_int(value: Any, *, label: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise CPMRMotionBranchContractError(f"{label} must be scalar")
        value = value.detach().cpu().item()
    return _exact_int(value, label=label)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _require_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CPMRMotionBranchContractError(f"{label} must be a torch.Tensor")
    return value


def _require_phase_zero_positive_zero(carrier: torch.Tensor) -> None:
    """Reject nonzero and signed-zero phase-0 bytes at the runtime boundary."""

    phase_zero = carrier[:, :CARRIER_TOKENS_PER_PHASE].detach().contiguous()
    phase_zero_bytes = phase_zero.reshape(-1).view(torch.uint8)
    if bool(torch.count_nonzero(phase_zero_bytes).item()):
        raise CPMRMotionBranchContractError(
            "motion carrier phase 0 must be byte-exact positive zero"
        )


def _clone_module(module: Any, *, label: str) -> nn.Module:
    if not isinstance(module, nn.Module):
        raise CPMRMotionBranchContractError(f"attn1.{label} must be an nn.Module")
    source_state = module.state_dict()
    if any(value.device.type == "meta" for value in source_state.values()):
        raise CPMRMotionBranchContractError(
            f"attn1.{label} cannot be cloned before checkpoint materialization"
        )
    try:
        cloned = copy.deepcopy(module)
    except Exception as error:
        raise CPMRMotionBranchContractError(
            f"could not independently clone attn1.{label}"
        ) from error
    if cloned is module:
        raise CPMRMotionBranchContractError(f"attn1.{label} clone shares module identity")
    cloned_state = cloned.state_dict()
    if tuple(source_state) != tuple(cloned_state):
        raise CPMRMotionBranchContractError(
            f"attn1.{label} clone changed state-dict keys"
        )
    for key, source_value in source_state.items():
        cloned_value = cloned_state[key]
        if source_value.dtype != cloned_value.dtype or source_value.shape != cloned_value.shape:
            raise CPMRMotionBranchContractError(
                f"attn1.{label}.{key} clone changed dtype or shape"
            )
        if not torch.equal(source_value, cloned_value):
            raise CPMRMotionBranchContractError(
                f"attn1.{label}.{key} clone changed tensor values"
            )
    return cloned


def _integer_sequence(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise CPMRMotionBranchContractError(f"{label} tensor must be one-dimensional")
        raw = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        raise CPMRMotionBranchContractError(f"{label} must be a one-dimensional sequence")
    return tuple(_exact_int(item, label=f"{label} item") for item in raw)


@dataclass
class CPMRConditionedEncoderBinding:
    """One-use binding for the post-condition-embedder/post-SP text tensor.

    The outer runner can authenticate the raw tensor passed into the pinned
    transformer, but that object is replaced by ``condition_embedder`` and, in
    Ulysses mode, by ``prepare_inputs_for_sp``.  The first selected ``attn2``
    therefore binds the resulting internal object and every later selected
    block must observe that exact same object in strict block order.
    """

    expected_block_indices: tuple[int, ...]
    _owner_token: Any = field(repr=False)
    _bound_tensor: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _observed_block_indices: list[int] = field(
        default_factory=list, init=False, repr=False
    )
    _active: bool = field(default=False, init=False, repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)
    _completed: bool = field(default=False, init=False, repr=False)
    _aborted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._owner_token is None:
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding requires an owner token"
            )
        try:
            indices = tuple(
                _exact_int(item, label="conditioned encoder block index")
                for item in self.expected_block_indices
            )
        except TypeError as error:
            raise CPMRMotionBranchContractError(
                "conditioned encoder block inventory must be iterable"
            ) from error
        if (
            not indices
            or indices != tuple(sorted(set(indices)))
            or any(item not in MOTION_BLOCK_INDICES for item in indices)
        ):
            raise CPMRMotionBranchContractError(
                "conditioned encoder block inventory must be a non-empty "
                "strictly increasing subset of blocks 0..15"
            )
        self.expected_block_indices = indices

    def begin(self) -> None:
        if self._active or self._consumed:
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding is active or already consumed"
            )
        self._active = True

    def observe(self, owner_token: Any, block_index: int, value: Any) -> None:
        if not self._active or self._consumed:
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding is not active"
            )
        if owner_token is not self._owner_token:
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding belongs to a different patch handle"
            )
        value = _require_tensor(value, label="conditioned encoder_hidden_states")
        position = len(self._observed_block_indices)
        if position >= len(self.expected_block_indices):
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding observed too many blocks"
            )
        expected = self.expected_block_indices[position]
        observed = _exact_int(block_index, label="conditioned encoder block index")
        if observed != expected:
            raise CPMRMotionBranchContractError(
                f"conditioned encoder binding expected block {expected}, got {observed}"
            )
        if self._bound_tensor is None:
            self._bound_tensor = value
        elif value is not self._bound_tensor:
            raise CPMRMotionBranchContractError(
                "conditioned encoder_hidden_states object changed between motion blocks"
            )
        self._observed_block_indices.append(observed)

    def finish(self) -> None:
        if not self._active or self._consumed:
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding cannot be completed"
            )
        self._active = False
        self._consumed = True
        if tuple(self._observed_block_indices) != self.expected_block_indices:
            self._bound_tensor = None
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding did not observe the exact block inventory"
            )
        self._completed = True
        self._bound_tensor = None

    def abort(self) -> None:
        self._active = False
        self._consumed = True
        self._aborted = True
        self._bound_tensor = None

    def receipt(self) -> dict[str, Any]:
        return {
            "expected_block_indices": list(self.expected_block_indices),
            "observed_block_indices": list(self._observed_block_indices),
            "completed": self._completed,
            "consumed": self._consumed,
            "aborted": self._aborted,
            "bound_tensor_released": self._bound_tensor is None,
        }


def _conditioned_encoder_binding_for_processors(
    processors: Sequence[Any],
) -> CPMRConditionedEncoderBinding:
    """Derive, rather than accept, the binding inventory from one patch."""

    items = tuple(processors)
    if not items or any(not isinstance(item, CPMRTextAttnProcessor) for item in items):
        raise CPMRMotionBranchContractError(
            "conditioned encoder binding requires CPMR processors"
        )
    indices = tuple(item.block_index for item in items)
    if indices != tuple(sorted(set(indices))):
        raise CPMRMotionBranchContractError(
            "conditioned encoder binding processors must be strictly ordered"
        )
    owner_token = items[0]._patch_token
    if any(item._patch_token is not owner_token for item in items):
        raise CPMRMotionBranchContractError(
            "conditioned encoder binding processors belong to different patches"
        )
    return CPMRConditionedEncoderBinding(indices, owner_token)


@dataclass(frozen=True)
class CPMRMotionInvocation:
    """One APG branch identity, one-use encoder binding, and motion payload.

    ``prompt_object`` is the exact raw conditioned encoder tensor passed into
    the transformer call.  It is authenticated at context entry and compared
    with ``positive_noop_prompt_object`` by Python object identity.  Because the
    pinned transformer replaces that object internally, the one-use
    ``conditioned_encoder_binding`` separately authenticates the tensor seen by
    selected ``attn2`` processors after condition embedding and SP slicing.
    Text matching and APG call ordinals are never used for routing.
    """

    trajectory: str
    polarity: str
    prompt_object: Any
    positive_noop_prompt_object: Any
    conditioned_encoder_binding: Optional[CPMRConditionedEncoderBinding] = None
    gate: float = 0.0
    carrier: Optional[torch.Tensor] = None
    activity: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.trajectory not in TRAJECTORIES:
            raise CPMRMotionBranchContractError(
                f"unknown CPMR trajectory {self.trajectory!r}"
            )
        if self.polarity not in POLARITIES:
            raise CPMRMotionBranchContractError(
                f"unknown CPMR APG polarity {self.polarity!r}"
            )
        if isinstance(self.gate, bool):
            raise CPMRMotionBranchContractError("motion gate must not be bool")
        try:
            gate = float(self.gate)
        except (TypeError, ValueError, OverflowError) as error:
            raise CPMRMotionBranchContractError("motion gate must be scalar") from error
        if not math.isfinite(gate) or gate not in FROZEN_GATES:
            raise CPMRMotionBranchContractError(
                f"frozen oracle gate must be one of {FROZEN_GATES}, got {self.gate!r}"
            )
        object.__setattr__(self, "gate", gate)
        if not isinstance(self.prompt_object, torch.Tensor) or not isinstance(
            self.positive_noop_prompt_object, torch.Tensor
        ):
            raise CPMRMotionBranchContractError(
                "routing requires two raw conditioned encoder tensors"
            )
        binding = self.conditioned_encoder_binding
        if binding is not None and not isinstance(
            binding, CPMRConditionedEncoderBinding
        ):
            raise CPMRMotionBranchContractError(
                "conditioned_encoder_binding has the wrong type"
            )
        if self.routes_motion != (binding is not None):
            raise CPMRMotionBranchContractError(
                "exactly the final positive no-op route requires a conditioned "
                "encoder binding"
            )

    @property
    def branch_tag(self) -> str:
        return f"{self.trajectory}:{self.polarity}"

    @property
    def routes_motion(self) -> bool:
        return (
            self.trajectory == FINAL_RENDER
            and self.polarity == POSITIVE
            and self.prompt_object is self.positive_noop_prompt_object
        )


_CURRENT_INVOCATION: contextvars.ContextVar[Optional[CPMRMotionInvocation]] = (
    contextvars.ContextVar("cpmr_motion_invocation", default=None)
)


def current_cpmr_motion_invocation() -> Optional[CPMRMotionInvocation]:
    return _CURRENT_INVOCATION.get()


@contextlib.contextmanager
def cpmr_motion_invocation(
    invocation: CPMRMotionInvocation,
    *,
    encoder_hidden_states: torch.Tensor,
) -> Iterator[CPMRMotionInvocation]:
    """Authenticate and install one non-nestable transformer-call context."""

    if not isinstance(invocation, CPMRMotionInvocation):
        raise CPMRMotionBranchContractError("invocation has the wrong type")
    if _CURRENT_INVOCATION.get() is not None:
        raise CPMRMotionBranchContractError("nested CPMR motion invocations are forbidden")
    raw_encoder = _require_tensor(
        encoder_hidden_states, label="raw transformer encoder_hidden_states"
    )
    if raw_encoder is not invocation.prompt_object:
        raise CPMRMotionBranchContractError(
            "CPMR invocation is not bound to the transformer input encoder object"
        )
    binding = invocation.conditioned_encoder_binding
    if invocation.routes_motion:
        if binding is None:
            raise CPMRMotionBranchContractError(
                "active CPMR invocation lacks a conditioned encoder binding"
            )
        binding.begin()
    token = _CURRENT_INVOCATION.set(invocation)
    try:
        yield invocation
    except BaseException:
        if binding is not None:
            binding.abort()
        raise
    else:
        if binding is not None:
            binding.finish()
    finally:
        _CURRENT_INVOCATION.reset(token)


def _validate_motion_payload(
    invocation: CPMRMotionInvocation,
    *,
    hidden_states: torch.Tensor,
) -> bool:
    carrier = _require_tensor(invocation.carrier, label="motion carrier")
    activity = _require_tensor(invocation.activity, label="motion activity")
    if _shape(carrier) != (1, CARRIER_TOKENS, HIDDEN_SIZE):
        raise CPMRMotionBranchContractError(
            "motion carrier must be shaped [1,1344,1536]"
        )
    if _shape(activity) != (1, LATENT_PHASES) or activity.dtype != torch.bool:
        raise CPMRMotionBranchContractError(
            "motion activity must be bool [1,21]"
        )
    if carrier.device != hidden_states.device:
        raise CPMRMotionBranchContractError(
            "motion carrier and hidden_states must be on the same device"
        )
    if carrier.dtype != hidden_states.dtype:
        raise CPMRMotionBranchContractError(
            "motion carrier and hidden_states must have the same attention dtype"
        )
    if activity.device != carrier.device:
        raise CPMRMotionBranchContractError(
            "motion activity and carrier must be on the same device"
        )
    if not bool(torch.isfinite(carrier).all().item()):
        raise CPMRMotionBranchContractError("motion carrier contains non-finite values")

    phase_tokens = carrier.reshape(
        1, LATENT_PHASES, CARRIER_TOKENS_PER_PHASE, HIDDEN_SIZE
    )
    if bool(activity[:, 0].any().item()):
        raise CPMRMotionBranchContractError(
            "motion activity phase 0 must be false"
        )
    _require_phase_zero_positive_zero(carrier)
    phase_nonzero = torch.count_nonzero(phase_tokens, dim=(2, 3)).ne(0)
    if not torch.equal(phase_nonzero, activity):
        raise CPMRMotionBranchContractError(
            "motion activity must exactly equal per-phase carrier nonzero state"
        )
    return bool(activity.any().item())


class MotionCrossAttention(nn.Module):
    """Independent frozen visual-domain Q/K/V/O branch cloned from ``attn1``."""

    def __init__(
        self,
        donor_attn1: Any,
        *,
        block_index: int,
        projection_processor: Optional[Any] = None,
        varlen_attention_fn: Optional[Callable[..., torch.Tensor]] = None,
        gen_cu_seqlens_fn: Optional[Callable[..., Any]] = None,
        padding_tensor_fn: Optional[Callable[..., torch.Tensor]] = None,
        slice_input_tensor_fn: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.block_index = _exact_int(block_index, label="block index")
        if self.block_index not in MOTION_BLOCK_INDICES:
            raise CPMRMotionBranchContractError(
                "frozen motion branch is restricted to blocks 0..15"
            )

        heads = _exact_int(getattr(donor_attn1, "heads", None), label="attn1 heads")
        if heads != ATTENTION_HEADS:
            raise CPMRMotionBranchContractError(
                f"attn1 must expose exactly {ATTENTION_HEADS} heads"
            )
        self.heads = heads
        self.inner_dim = _exact_int(
            getattr(donor_attn1, "inner_dim", None), label="attn1 inner_dim"
        )
        self.inner_kv_dim = _exact_int(
            getattr(donor_attn1, "inner_kv_dim", None), label="attn1 inner_kv_dim"
        )
        self.out_dim = _exact_int(
            getattr(donor_attn1, "out_dim", None), label="attn1 out_dim"
        )
        self.cross_attention_dim = _exact_int(
            getattr(donor_attn1, "cross_attention_dim", None),
            label="attn1 cross_attention_dim",
        )
        if (
            self.inner_dim,
            self.inner_kv_dim,
            self.out_dim,
            self.cross_attention_dim,
        ) != (HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE):
            raise CPMRMotionBranchContractError(
                "attn1 visual donor dimensions must all equal 1536"
            )
        self.head_dim = self.inner_dim // self.heads
        if self.head_dim != ATTENTION_HEAD_DIM:
            raise CPMRMotionBranchContractError(
                "attn1 head_dim derived from inner_dim/heads must equal 128"
            )

        self.to_q = _clone_module(donor_attn1.to_q, label="to_q")
        self.to_k = _clone_module(donor_attn1.to_k, label="to_k")
        self.to_v = _clone_module(donor_attn1.to_v, label="to_v")
        self.norm_q = _clone_module(getattr(donor_attn1, "norm_q", None), label="norm_q")
        self.norm_k = _clone_module(getattr(donor_attn1, "norm_k", None), label="norm_k")
        donor_to_out = getattr(donor_attn1, "to_out", None)
        if not isinstance(donor_to_out, (nn.ModuleList, list, tuple)) or len(donor_to_out) != 2:
            raise CPMRMotionBranchContractError("attn1.to_out must contain projection and dropout")
        self.to_out = nn.ModuleList(
            [
                _clone_module(donor_to_out[0], label="to_out.0"),
                _clone_module(donor_to_out[1], label="to_out.1"),
            ]
        )

        for name in ("to_q", "to_k", "to_v"):
            projection = getattr(self, name)
            in_features = getattr(projection, "in_features", HIDDEN_SIZE)
            out_features = getattr(projection, "out_features", HIDDEN_SIZE)
            if (int(in_features), int(out_features)) != (HIDDEN_SIZE, HIDDEN_SIZE):
                raise CPMRMotionBranchContractError(
                    f"attn1.{name} must be a 1536 -> 1536 projection"
                )
        out_projection = self.to_out[0]
        if (
            int(getattr(out_projection, "in_features", HIDDEN_SIZE)),
            int(getattr(out_projection, "out_features", HIDDEN_SIZE)),
        ) != (HIDDEN_SIZE, HIDDEN_SIZE):
            raise CPMRMotionBranchContractError(
                "attn1.to_out.0 must be a 1536 -> 1536 projection"
            )

        source_processor = (
            projection_processor
            if projection_processor is not None
            else getattr(donor_attn1, "processor", None)
        )
        if not callable(getattr(source_processor, "_project_qkv", None)):
            raise CPMRMotionBranchContractError(
                "attn1 processor must expose official _project_qkv"
            )
        try:
            projector = copy.deepcopy(source_processor)
        except Exception as error:
            raise CPMRMotionBranchContractError(
                "could not independently clone the attn1 projection processor"
            ) from error
        object.__setattr__(self, "_projection_processor", projector)
        object.__setattr__(self, "_varlen_attention_fn", varlen_attention_fn)
        object.__setattr__(self, "_gen_cu_seqlens_fn", gen_cu_seqlens_fn)
        object.__setattr__(self, "_padding_tensor_fn", padding_tensor_fn)
        object.__setattr__(self, "_slice_input_tensor_fn", slice_input_tensor_fn)

        donor_parameter_ids = {
            id(parameter)
            for module in (
                donor_attn1.to_q,
                donor_attn1.to_k,
                donor_attn1.to_v,
                getattr(donor_attn1, "norm_q", None),
                getattr(donor_attn1, "norm_k", None),
                donor_to_out[0],
                donor_to_out[1],
            )
            if isinstance(module, nn.Module)
            for parameter in module.parameters()
        }
        cloned_parameter_ids = {id(parameter) for parameter in self.parameters()}
        if donor_parameter_ids.intersection(cloned_parameter_ids):
            raise CPMRMotionBranchContractError(
                "motion branch shares a Parameter object with original attn1"
            )
        self.requires_grad_(False)
        self.motion_calls = 0
        self.project_qkv_calls = 0
        self.varlen_calls = 0
        self.explicit_custom_collective_calls = 0
        self.last_metadata: Optional[dict[str, Any]] = None

    def _runtime_ops(
        self,
    ) -> tuple[
        Callable[..., torch.Tensor],
        Callable[..., Any],
        Callable[..., torch.Tensor],
        Callable[..., torch.Tensor],
    ]:
        varlen_fn = self._varlen_attention_fn
        gen_fn = self._gen_cu_seqlens_fn
        pad_fn = self._padding_tensor_fn
        slice_fn = self._slice_input_tensor_fn
        if varlen_fn is None:
            from bernini.attention import varlen_attention as varlen_fn
        if gen_fn is None or pad_fn is None or slice_fn is None:
            from bernini.parallel import (
                gen_cu_seqlens_for_cross_attn,
                padding_tensor_for_seqeunce_parallel,
                slice_input_tensor,
            )

            if gen_fn is None:
                gen_fn = gen_cu_seqlens_for_cross_attn
            if pad_fn is None:
                pad_fn = padding_tensor_for_seqeunce_parallel
            if slice_fn is None:
                slice_fn = slice_input_tensor
        return varlen_fn, gen_fn, pad_fn, slice_fn

    def _merge_projected_motion_heads(
        self,
        projected_motion_heads: torch.Tensor,
        *,
        local_target_mask: torch.Tensor,
        padding_tensor_fn: Callable[..., torch.Tensor],
        slice_input_tensor_fn: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        """Merge real attention heads through the frozen output projection.

        ``projected_motion_heads`` is the direct ``varlen_attention`` result
        after restoring its batch axis and before flattening/to_out.  The
        default implementation is deliberately identical to the V11 path.
        Training-only extensions may override this single hook to transform
        *actual* heads while retaining the official attention/SP machinery.
        """

        del local_target_mask, padding_tensor_fn, slice_input_tensor_fn
        output = projected_motion_heads.flatten(2, 3).contiguous()
        output = self.to_out[0](output)
        output = self.to_out[1](output)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        carrier: torch.Tensor,
        *,
        origin_hidden_states_seq_len: int,
        batch_image_vae_seqlen: Sequence[int],
    ) -> torch.Tensor:
        hidden_states = _require_tensor(hidden_states, label="motion hidden_states")
        carrier = _require_tensor(carrier, label="motion carrier")
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise CPMRMotionBranchContractError(
                "motion hidden_states must be batch-1 [1,local_q,1536]"
            )
        if int(hidden_states.shape[2]) != HIDDEN_SIZE:
            raise CPMRMotionBranchContractError("motion hidden width must be 1536")
        if _shape(carrier) != (1, CARRIER_TOKENS, HIDDEN_SIZE):
            raise CPMRMotionBranchContractError(
                "motion carrier must be replicated full [1,1344,1536]"
            )
        _require_phase_zero_positive_zero(carrier)
        if carrier.device != hidden_states.device or carrier.dtype != hidden_states.dtype:
            raise CPMRMotionBranchContractError(
                "motion carrier must match hidden_states device and dtype"
            )
        origin = _exact_int(
            origin_hidden_states_seq_len, label="origin_hidden_states_seq_len"
        )
        if origin != GLOBAL_VISUAL_TOKENS:
            raise CPMRMotionBranchContractError(
                "canonical CPMR visual sequence must contain 39060 global tokens"
            )
        batch_lengths = _integer_sequence(
            batch_image_vae_seqlen, label="batch_image_vae_seqlen"
        )
        if batch_lengths != (GLOBAL_VISUAL_TOKENS,):
            raise CPMRMotionBranchContractError(
                "frozen CPMR runtime defines only batch 1 with length 39060"
            )
        if not bool(torch.isfinite(hidden_states).all().item()):
            raise CPMRMotionBranchContractError("motion queries contain non-finite values")
        if not bool(torch.isfinite(carrier).all().item()):
            raise CPMRMotionBranchContractError("motion carrier contains non-finite values")

        varlen_fn, gen_fn, pad_fn, slice_fn = self._runtime_ops()
        metadata = gen_fn(
            GLOBAL_VISUAL_TOKENS,
            [GLOBAL_VISUAL_TOKENS],
            [CARRIER_TOKENS],
            device=hidden_states.device,
        )
        if not isinstance(metadata, (tuple, list)) or len(metadata) != 5:
            raise CPMRMotionBranchContractError(
                "official cross-attention metadata must contain five entries"
            )
        cu_k, cu_q, max_k, max_q, rank_q_len = metadata
        cu_k = _require_tensor(cu_k, label="motion cu_seqlens_k")
        cu_q = _require_tensor(cu_q, label="motion cu_seqlens_q")
        rank_q_len = _scalar_int(rank_q_len, label="motion rank_q_len")
        expected_local = int(hidden_states.shape[1])
        if expected_local not in (GLOBAL_VISUAL_TOKENS, GLOBAL_VISUAL_TOKENS // 4):
            raise CPMRMotionBranchContractError(
                "frozen runtime only defines full-sequence or Ulysses-4 local queries"
            )
        if rank_q_len != expected_local:
            raise CPMRMotionBranchContractError(
                "official rank_q_len differs from the outer local query length"
            )
        if cu_k.tolist() != [0, CARRIER_TOKENS]:
            raise CPMRMotionBranchContractError("motion cu_k must equal [0,1344]")
        if cu_q.tolist() != [0, expected_local]:
            raise CPMRMotionBranchContractError(
                "motion cu_q must span exactly the local query shard"
            )
        if _scalar_int(max_k, label="motion max_k") != CARRIER_TOKENS:
            raise CPMRMotionBranchContractError("motion max_k must equal 1344")
        if _scalar_int(max_q, label="motion max_q") != expected_local:
            raise CPMRMotionBranchContractError(
                "motion max_q must equal the local query length"
            )

        global_mask = hidden_states.new_zeros((1, GLOBAL_VISUAL_TOKENS, 1))
        global_mask[:, SOURCE_VISUAL_TOKENS:, :] = 1
        local_mask = slice_fn(pad_fn(global_mask, dim=1), dim=1)
        if _shape(local_mask) != (1, expected_local, 1):
            raise CPMRMotionBranchContractError(
                "official pad/slice target mask differs from local query layout"
            )
        if not bool(
            torch.logical_or(local_mask == 0, local_mask == 1).all().item()
        ):
            raise CPMRMotionBranchContractError("target mask must be binary")

        query, key, value = self._projection_processor._project_qkv(
            self,
            hidden_states,
            carrier,
            None,
            GLOBAL_VISUAL_TOKENS,
            True,
        )
        self.project_qkv_calls += 1
        expected_q_shape = (1, expected_local, ATTENTION_HEADS, ATTENTION_HEAD_DIM)
        expected_kv_shape = (1, CARRIER_TOKENS, ATTENTION_HEADS, ATTENTION_HEAD_DIM)
        if _shape(query) != expected_q_shape:
            raise CPMRMotionBranchContractError(
                f"motion query must be local-Q/full-head {expected_q_shape}"
            )
        if _shape(key) != expected_kv_shape or _shape(value) != expected_kv_shape:
            raise CPMRMotionBranchContractError(
                f"motion key/value must be replicated-full/full-head {expected_kv_shape}"
            )
        if query.device != hidden_states.device or key.device != hidden_states.device:
            raise CPMRMotionBranchContractError("projected motion tensors changed device")
        if not all(bool(torch.isfinite(item).all().item()) for item in (query, key, value)):
            raise CPMRMotionBranchContractError("projected motion Q/K/V is non-finite")

        query_dtype = query.dtype
        output = varlen_fn(
            query.squeeze(0).contiguous(),
            key.squeeze(0).contiguous(),
            value.squeeze(0).contiguous(),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=False,
        )
        self.varlen_calls += 1
        if _shape(output) != expected_q_shape[1:]:
            raise CPMRMotionBranchContractError(
                "motion varlen output shape differs from local query"
            )
        output = output.unsqueeze(0).contiguous().to(query_dtype)
        if _shape(output) != expected_q_shape:
            raise CPMRMotionBranchContractError(
                "motion varlen output did not preserve separated projected heads"
            )
        output = self._merge_projected_motion_heads(
            output,
            local_target_mask=local_mask,
            padding_tensor_fn=pad_fn,
            slice_input_tensor_fn=slice_fn,
        )
        if _shape(output) != (1, expected_local, HIDDEN_SIZE):
            raise CPMRMotionBranchContractError(
                "motion output projection changed local query layout"
            )
        # The mask is intentionally applied after both output stages so an
        # out-projection bias cannot leak a residual into source positions.
        output = torch.where(local_mask.bool(), output, torch.zeros_like(output))
        if not bool(torch.isfinite(output).all().item()):
            raise CPMRMotionBranchContractError("motion residual is non-finite")
        if bool(torch.count_nonzero(output.masked_select(~local_mask.bool())).item()):
            raise CPMRMotionBranchContractError("source motion residual is not exact zero")

        self.motion_calls += 1
        self.last_metadata = {
            "origin_q": GLOBAL_VISUAL_TOKENS,
            "local_q": expected_local,
            "carrier_k": CARRIER_TOKENS,
            "heads": ATTENTION_HEADS,
            "head_dim": ATTENTION_HEAD_DIM,
            "cu_q": [int(item) for item in cu_q.tolist()],
            "cu_k": [int(item) for item in cu_k.tolist()],
            "max_q": _scalar_int(max_q, label="motion max_q"),
            "max_k": _scalar_int(max_k, label="motion max_k"),
            "rank_q_len": rank_q_len,
            "explicit_custom_collectives": 0,
            "measured_custom_collectives": None,
        }
        return output

    def statistics(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "motion_calls": self.motion_calls,
            "project_qkv_calls": self.project_qkv_calls,
            "varlen_calls": self.varlen_calls,
            "explicit_custom_collective_calls": self.explicit_custom_collective_calls,
            "measured_custom_collective_calls": None,
            "last_metadata": copy.deepcopy(self.last_metadata),
        }


def _global_origin_length(
    origin_hidden_states_seq_len: Optional[int],
    batch_image_vae_seqlen: Any,
) -> int:
    if origin_hidden_states_seq_len is not None:
        origin = _exact_int(
            origin_hidden_states_seq_len, label="origin_hidden_states_seq_len"
        )
    else:
        origin = sum(
            _integer_sequence(
                batch_image_vae_seqlen, label="batch_image_vae_seqlen"
            )
        )
    if origin != GLOBAL_VISUAL_TOKENS:
        raise CPMRMotionBranchContractError(
            "canonical motion branch requires 39060 global visual tokens"
        )
    return origin


class CPMRTextAttnProcessor:
    """Plain, explicit-signature wrapper around one official ``attn2`` processor."""

    def __init__(
        self,
        base_processor: Any,
        motion_attention: MotionCrossAttention,
        *,
        block_index: int,
        patch_token: Any = None,
    ) -> None:
        if not callable(base_processor):
            raise CPMRMotionBranchContractError("base attn2 processor must be callable")
        if not isinstance(motion_attention, MotionCrossAttention):
            raise CPMRMotionBranchContractError(
                "motion_attention must be a registered MotionCrossAttention"
            )
        index = _exact_int(block_index, label="block index")
        if index != motion_attention.block_index:
            raise CPMRMotionBranchContractError(
                "processor and motion module block indices differ"
            )
        self.base_processor = base_processor
        self.motion_attention = motion_attention
        self.block_index = index
        self._patch_token = object() if patch_token is None else patch_token
        self.base_calls = 0
        self.motion_calls = 0
        self.zero_gate_delegations = 0
        self.inactive_delegations = 0
        self.no_context_delegations = 0
        self.branch_delegations = 0
        self.branch_counts: dict[str, int] = {}

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        batch_image_vae_seqlen=None,
        text_features_length=None,
        origin_hidden_states_seq_len: Optional[int] = None,
        split_hidden_states_seq_len: Optional[int] = None,
        cu_seqlens_q_cache=None,
        max_seqlen_q_cache=None,
        cu_seqlens_k_cross_cache=None,
        cu_seqlens_q_cross_cache=None,
        max_seqlen_k_cross_cache=None,
        max_seqlen_q_cross_cache=None,
    ) -> torch.Tensor:
        base_output = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            text_features_length=text_features_length,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            split_hidden_states_seq_len=split_hidden_states_seq_len,
            cu_seqlens_q_cache=cu_seqlens_q_cache,
            max_seqlen_q_cache=max_seqlen_q_cache,
            cu_seqlens_k_cross_cache=cu_seqlens_k_cross_cache,
            cu_seqlens_q_cross_cache=cu_seqlens_q_cross_cache,
            max_seqlen_k_cross_cache=max_seqlen_k_cross_cache,
            max_seqlen_q_cross_cache=max_seqlen_q_cross_cache,
        )
        self.base_calls += 1
        base_output = _require_tensor(base_output, label="base attn2 output")
        if _shape(base_output) != _shape(hidden_states):
            raise CPMRMotionBranchContractError(
                "base attn2 output must preserve rank-local hidden shape"
            )

        invocation = current_cpmr_motion_invocation()
        if invocation is None:
            self.no_context_delegations += 1
            return base_output
        self.branch_counts[invocation.branch_tag] = (
            self.branch_counts.get(invocation.branch_tag, 0) + 1
        )
        if not invocation.routes_motion:
            self.branch_delegations += 1
            return base_output
        binding = invocation.conditioned_encoder_binding
        if binding is None:
            raise CPMRMotionBranchContractError(
                "active CPMR context lacks a conditioned encoder binding"
            )
        # This visit deliberately precedes every gate/activity early return.
        # Thus Z0 and all-inactive calls authenticate the same production
        # condition path while still performing no motion tensor arithmetic.
        binding.observe(self._patch_token, self.block_index, encoder_hidden_states)

        has_activity = _validate_motion_payload(
            invocation, hidden_states=hidden_states
        )
        if invocation.gate == 0.0:
            self.zero_gate_delegations += 1
            return base_output
        if not has_activity:
            self.inactive_delegations += 1
            return base_output
        if encoder_hidden_states is None:
            raise CPMRMotionBranchContractError(
                "active CPMR wrapper must be installed on text cross-attention attn2"
            )
        if attention_mask is not None or rotary_emb is not None:
            raise CPMRMotionBranchContractError(
                "active motion cross-attention does not accept text mask or rotary_emb"
            )
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise CPMRMotionBranchContractError(
                "active CPMR wrapper only defines batch-1 hidden states"
            )
        if not bool(torch.isfinite(base_output).all().item()):
            raise CPMRMotionBranchContractError("base attn2 output is non-finite")

        origin = _global_origin_length(
            origin_hidden_states_seq_len, batch_image_vae_seqlen
        )
        residual = self.motion_attention(
            hidden_states,
            invocation.carrier,
            origin_hidden_states_seq_len=origin,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
        )
        gate_fp32 = torch.tensor(
            invocation.gate, dtype=torch.float32, device=residual.device
        )
        gated_residual = (residual.float() * gate_fp32).to(base_output.dtype)
        combined = base_output + gated_residual
        if not bool(torch.isfinite(combined).all().item()):
            raise CPMRMotionBranchContractError("combined text/motion output is non-finite")
        self.motion_calls += 1
        return combined

    def statistics(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "base_calls": self.base_calls,
            "motion_calls": self.motion_calls,
            "zero_gate_delegations": self.zero_gate_delegations,
            "inactive_delegations": self.inactive_delegations,
            "no_context_delegations": self.no_context_delegations,
            "branch_delegations": self.branch_delegations,
            "branch_counts": dict(sorted(self.branch_counts.items())),
        }


def resolve_wan_transformer(model: Any) -> Any:
    """Resolve the single pinned 30-block Wan transformer through wrappers."""

    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            if len(blocks) != EXPECTED_BLOCK_COUNT:
                raise CPMRMotionBranchContractError(
                    f"Bernini-R 1.3B must have {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
                )
            return candidate
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("diff_dec", "transformer", "base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None and nested is not candidate:
                queue.append(nested)
    raise CPMRMotionBranchContractError(
        "could not resolve the pinned 30-block Bernini-R Wan transformer"
    )


@dataclass
class CPMRMotionPatchHandle:
    transformer: Any
    indices: tuple[int, ...]
    processors: tuple[CPMRTextAttnProcessor, ...]
    motion_modules: tuple[MotionCrossAttention, ...]
    original_processors: tuple[Any, ...]
    _patch_token: Any = field(repr=False)
    restored: bool = False

    def new_conditioned_encoder_binding(self) -> CPMRConditionedEncoderBinding:
        if self.restored:
            raise CPMRMotionBranchContractError(
                "cannot bind conditioned encoder after patch restore"
            )
        binding = _conditioned_encoder_binding_for_processors(self.processors)
        if binding._owner_token is not self._patch_token:
            raise CPMRMotionBranchContractError(
                "conditioned encoder binding ownership differs from patch handle"
            )
        return binding

    def restore(self) -> None:
        if self.restored:
            return
        blocks = self.transformer.blocks
        # Validate the complete handle before mutating anything; a conflict must
        # never result in a partial restore.
        for index, processor, motion in zip(
            self.indices, self.processors, self.motion_modules
        ):
            block = blocks[index]
            if getattr(block.attn2, "processor", None) is not processor:
                raise CPMRMotionBranchContractError(
                    f"block {index} attn2 processor changed behind patch handle"
                )
            if getattr(block, MOTION_MODULE_NAME, None) is not motion:
                raise CPMRMotionBranchContractError(
                    f"block {index} motion module changed behind patch handle"
                )
        for index, original in zip(self.indices, self.original_processors):
            block = blocks[index]
            setter = getattr(block.attn2, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                block.attn2.processor = original
            delattr(block, MOTION_MODULE_NAME)
        self.restored = True

    def receipt(self) -> dict[str, Any]:
        return {
            "method": "counterfactual-proposal-motion-rebinding-v11",
            "installed_block_indices": list(self.indices),
            "installed_block_count": len(self.indices),
            "registered_module_name": MOTION_MODULE_NAME,
            "official_cross_sp": "local-q/full-12-head/replicated-full-carrier",
            "conditioned_encoder_binding": (
                "raw-transformer-input-authenticated/"
                "post-condition-post-sp-one-use-identity"
            ),
            "binding_expected_block_indices": list(self.indices),
            "frozen_eager_inference_only": True,
            "gradient_checkpoint_supported": False,
            "torch_compile_supported": False,
            "explicit_custom_collective_calls": 0,
            "measured_custom_collective_calls": None,
            "batch_size": 1,
            "global_visual_tokens": GLOBAL_VISUAL_TOKENS,
            "carrier_tokens": CARRIER_TOKENS,
            "restored": self.restored,
            "processors": [processor.statistics() for processor in self.processors],
            "motion_modules": [motion.statistics() for motion in self.motion_modules],
        }

    def __enter__(self) -> "CPMRMotionPatchHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def install_cpmr_motion_branch(
    model: Any,
    *,
    motion_factory: Optional[Callable[[Any, int], MotionCrossAttention]] = None,
    processor_factory: Optional[
        Callable[[Any, MotionCrossAttention, int, Any], CPMRTextAttnProcessor]
    ] = None,
) -> CPMRMotionPatchHandle:
    """Install V11 on blocks 0..15 and return an exact reversible handle."""

    transformer = resolve_wan_transformer(model)
    originals: list[Any] = []
    processors: list[CPMRTextAttnProcessor] = []
    motions: list[MotionCrossAttention] = []
    installed_indices: list[int] = []
    patch_token = object()
    try:
        for index in MOTION_BLOCK_INDICES:
            block = transformer.blocks[index]
            if hasattr(block, MOTION_MODULE_NAME):
                raise CPMRMotionBranchContractError(
                    f"block {index} already has {MOTION_MODULE_NAME}"
                )
            original = getattr(block.attn2, "processor", None)
            if original is None:
                raise CPMRMotionBranchContractError(
                    f"block {index} attn2 lacks a processor"
                )
            if isinstance(original, CPMRTextAttnProcessor):
                raise CPMRMotionBranchContractError(
                    f"block {index} already has a CPMR processor"
                )
            motion = (
                motion_factory(block.attn1, index)
                if motion_factory is not None
                else MotionCrossAttention(block.attn1, block_index=index)
            )
            if not isinstance(motion, MotionCrossAttention):
                raise CPMRMotionBranchContractError(
                    "motion_factory returned the wrong type"
                )
            if motion.block_index != index:
                raise CPMRMotionBranchContractError(
                    "motion_factory returned the wrong block index"
                )
            processor = (
                processor_factory(original, motion, index, patch_token)
                if processor_factory is not None
                else CPMRTextAttnProcessor(
                    original,
                    motion,
                    block_index=index,
                    patch_token=patch_token,
                )
            )
            if not isinstance(processor, CPMRTextAttnProcessor):
                raise CPMRMotionBranchContractError(
                    "processor_factory returned the wrong type"
                )
            if processor._patch_token is not patch_token:
                raise CPMRMotionBranchContractError(
                    "processor_factory returned a processor owned by another patch"
                )
            block.add_module(MOTION_MODULE_NAME, motion)
            try:
                setter = getattr(block.attn2, "set_processor", None)
                if callable(setter):
                    setter(processor)
                else:
                    block.attn2.processor = processor
            except Exception:
                if getattr(block, MOTION_MODULE_NAME, None) is motion:
                    delattr(block, MOTION_MODULE_NAME)
                raise
            originals.append(original)
            motions.append(motion)
            processors.append(processor)
            installed_indices.append(index)
    except Exception:
        for index, original, processor, motion in zip(
            reversed(installed_indices),
            reversed(originals),
            reversed(processors),
            reversed(motions),
        ):
            block = transformer.blocks[index]
            if getattr(block.attn2, "processor", None) is processor:
                setter = getattr(block.attn2, "set_processor", None)
                if callable(setter):
                    setter(original)
                else:
                    block.attn2.processor = original
            if getattr(block, MOTION_MODULE_NAME, None) is motion:
                delattr(block, MOTION_MODULE_NAME)
        raise
    return CPMRMotionPatchHandle(
        transformer=transformer,
        indices=MOTION_BLOCK_INDICES,
        processors=tuple(processors),
        motion_modules=tuple(motions),
        original_processors=tuple(originals),
        _patch_token=patch_token,
    )


__all__ = [
    "ACTION_PROPOSAL",
    "ATTENTION_HEADS",
    "ATTENTION_HEAD_DIM",
    "CARRIER_TOKENS",
    "CPMRConditionedEncoderBinding",
    "CPMRMotionBranchContractError",
    "CPMRMotionInvocation",
    "CPMRMotionPatchHandle",
    "CPMRTextAttnProcessor",
    "FINAL_RENDER",
    "GLOBAL_VISUAL_TOKENS",
    "HIDDEN_SIZE",
    "MOTION_BLOCK_INDICES",
    "MOTION_MODULE_NAME",
    "MotionCrossAttention",
    "NOOP_PROPOSAL",
    "POSITIVE",
    "SOURCE_VISUAL_TOKENS",
    "TARGET_VISUAL_TOKENS",
    "UNCONDITIONAL",
    "cpmr_motion_invocation",
    "current_cpmr_motion_invocation",
    "install_cpmr_motion_branch",
    "resolve_wan_transformer",
]
