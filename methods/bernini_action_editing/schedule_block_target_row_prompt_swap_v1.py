#!/usr/bin/env python3
"""Frozen schedule x block target-row prompt-swap diagnostic.

The transformer always receives the no-op prompt.  A branch-only capture
forward records the post-condition-embedder/post-SP text tensor seen by each
``attn2`` processor; no hidden state or output from that forward is reused.
During a mixed forward the official processor is evaluated twice at selected
blocks with the *same current no-op hidden states*: once with the current
no-op encoder tensor and once with the captured branch encoder tensor.  Only
the native visual target suffix selects the latter output.

This module has no optimizer, Parameter, scheduler, sampler, or decoder.  It
also owns the preregistered six-output C0 and 112-output full-grid plans.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterator, Mapping, NoReturn, Optional, Sequence

try:
    from . import schedule_block_causal_policy_v1 as policy
except ImportError:
    import schedule_block_causal_policy_v1 as policy


SCHEMA_VERSION = "bernini-schedule-block-target-row-prompt-swap-v1"
PLAN_SCHEMA = "bernini-schedule-block-causal-localization-plan-v1"
TOTAL_BLOCKS = 30
SP_SIZE = 4
OWNERS = ("correct_owner", "wrong_owner")
TEXT_BRANCHES = (
    "noop",
    "forward",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
)
NONNOOP_BRANCHES = TEXT_BRANCHES[1:]
PHASES = ("capture", "plain_noop", "noop_swap", "mixed")
ALL30_BAND = "all30_reference"
NONE_BAND = "none"
C0_SCHEDULE_INDEX = 29
C0_BAND = "late_middle"
PROMPT_AUTHORITY_IID = "00435ad621c44fac"
PROMPT_AUTHORITY_SHA256 = (
    "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"
)
PROMPT_AUTHORITY_MAPPING = MappingProxyType({
    "noop": "noop",
    "forward": "action",
    "reverse": "reverse",
    "incomplete": "incomplete",
    "camera_only": "camera_only",
    "appearance_only": "appearance_only",
})
BRANCH_CALIBRATION_AUTHORITY = (
    ("noop", True, "numerical_baseline_only", False,
     "failed_first_frame_already_hands_on_hips", False),
    ("forward", True, "positive_direction_candidate", None,
     "direction_pass", False),
    ("reverse", True, "directional_negative_candidate", None,
     "direction_pass", False),
    ("incomplete", True, "exploratory_calibration_failed", None,
     "failed_delayed_action_reaches_hands_on_hips", False),
    ("camera_only", True, "exploratory_nuisance_control", None,
     "cross_branch_identity_background_confounded", False),
    ("appearance_only", True, "exploratory_nuisance_control", None,
     "cross_branch_identity_background_confounded", False),
)


def branch_calibration_authority() -> Mapping[str, Any]:
    return {
        branch: {
            "run_in_fixed_grid": run,
            "role": role,
            **(
                {"semantic_negative_authorized": semantic_negative}
                if semantic_negative is not None
                else {}
            ),
            "calibration": calibration,
            "scientific_veto_authorized": veto,
        }
        for branch, run, role, semantic_negative, calibration, veto
        in BRANCH_CALIBRATION_AUTHORITY
    }
WRONG_OWNER_VARIANT = "variant_a"
WRONG_OWNER_VIDEO_SHA256 = (
    "dc03115417c6c74c2b21d8a6d18ed076cace647efb1ede85149879ef13402d0f"
)
ORBIT_ROW_DIGEST = (
    "e6b48ee59d7816a03a0808d5826ddd0f405bb099b3dde12dc72ec545c90a8529"
)
OWNER_FULL_BLOB_SHA256 = MappingProxyType({
    "correct_owner": "3805084233cd037c6988cd0dfd343162c7f9df5ba2cfa784dd2e6eb5190032ff",
    "wrong_owner": "f637c212805b3ef78e0bca7d0a3af182900073ea4afc7fd66983edaab9f75a44",
})
OWNER_REFERENCE_BLOB_SHA256 = MappingProxyType({
    "correct_owner": (
        "57215a7b93835bcef100906494c0f4ab741344163d8e57867d079f1d6514549d",
        "56c47289613b7eb9b326a496fe61bb7d3705577a00cf0332c01fbf903ef0dfb4",
        "40462eda8bba99b0f6d95db9e2429209d74b82ea3c4a39eebe35d435020ec4cf",
        "315ffe61ad6a18704461d20d5c8b89c2ff9e873e641ebdc4b44b6ac934e3d22a",
    ),
    "wrong_owner": (
        "2e8b9d6a0b66c0c0c4d23976a9d7a4974287bb60bc8713d21f905bbbd7e0c9a3",
        "0adba43ea65f3c57fc1e78c7f17585351350f47516bcae30936479e4bb15e933",
        "ba1c143f362ce14ac68f249fd6623ffa2c50b893a6673406a17b253a0039f0b2",
        "f1b4779855a5c6ef5cbd87e856236a3be8592ab2b2d4b2b28e2a43018a3cac27",
    ),
})
NATIVE_SOURCE_IDS = (1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PromptSwapError(RuntimeError):
    """Raised before an ambiguous prompt intervention can execute."""


def fail(message: str) -> NoReturn:
    raise PromptSwapError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PromptSwapError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_int(value: Any, *, lower: int, upper: int, label: str) -> int:
    if type(value) is not int or not lower <= value < upper:
        fail(f"{label} must be an exact integer in [{lower},{upper})")
    return value


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        fail(f"{label} must expose tensor shape")
    try:
        result = tuple(int(item) for item in shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise PromptSwapError(f"{label} shape differs") from error
    if not result or any(item <= 0 for item in result):
        fail(f"{label} has empty geometry")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class OwnerInputBinding:
    """Sealed owner-prefix identity carried by every processor invocation."""

    owner: str
    schedule_index: int
    timestep: int
    sigma_float32_be_hex: str
    orbit_row_digest: str
    target_source_full_blob_sha256: str
    owner_full_blob_sha256: str
    owner_reference_blob_sha256: tuple[str, str, str, str]
    decoded_target_tensor_sha256: str
    decoded_owner_full_tensor_sha256: str
    decoded_owner_reference_tensor_sha256: tuple[str, str, str, str]
    epsilon_sha256: str
    target_x_s_sha256: str
    prepared_visual_prefix_sha256: str
    prepared_prefix_rotary_sha256: str
    total_tokens: int
    condition_tokens: int
    source_ids: tuple[float, float, float, float, float, float] = NATIVE_SOURCE_IDS

    def __post_init__(self) -> None:
        if self.owner not in OWNERS:
            fail("owner input binding axis differs")
        if self.schedule_index not in policy.REGISTERED_SCHEDULE_INDICES:
            fail("owner input binding schedule differs")
        expected_timestep = policy.exact40.PINNED_TIMESTEPS[self.schedule_index]
        expected_sigma_hex = policy.exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
            self.schedule_index
        ]
        if (
            type(self.timestep) is not int
            or self.timestep != expected_timestep
            or self.sigma_float32_be_hex != expected_sigma_hex
        ):
            fail("owner input binding timestep/sigma authority differs")
        total = _exact_int(self.total_tokens, lower=1, upper=2**31, label="owner total tokens")
        condition = _exact_int(
            self.condition_tokens,
            lower=0,
            upper=total,
            label="owner condition tokens",
        )
        if condition >= total:
            fail("owner input binding requires a strict target suffix")
        if _sha256(self.orbit_row_digest, label="orbit row digest") != ORBIT_ROW_DIGEST:
            fail("owner input binding orbit row differs")
        scalar_hashes = (
            (self.target_source_full_blob_sha256, "target source full blob"),
            (self.owner_full_blob_sha256, "owner full blob"),
            (self.decoded_target_tensor_sha256, "decoded target tensor"),
            (self.decoded_owner_full_tensor_sha256, "decoded owner full tensor"),
            (self.epsilon_sha256, "epsilon tensor"),
            (self.target_x_s_sha256, "target x_s tensor"),
            (self.prepared_visual_prefix_sha256, "prepared visual prefix"),
            (self.prepared_prefix_rotary_sha256, "prepared prefix rotary"),
        )
        for value, label in scalar_hashes:
            _sha256(value, label=label)
        for label, values in (
            ("owner reference blob", self.owner_reference_blob_sha256),
            ("decoded owner reference tensor", self.decoded_owner_reference_tensor_sha256),
        ):
            if not isinstance(values, tuple) or len(values) != 4:
                fail(f"{label} closure differs")
            for value in values:
                _sha256(value, label=label)
        if self.target_source_full_blob_sha256 != OWNER_FULL_BLOB_SHA256["correct_owner"]:
            fail("target source must remain the sealed source member")
        if self.owner_full_blob_sha256 != OWNER_FULL_BLOB_SHA256[self.owner]:
            fail("owner full blob does not match the owner axis")
        if self.owner_reference_blob_sha256 != OWNER_REFERENCE_BLOB_SHA256[self.owner]:
            fail("owner reference blobs do not match the owner axis")
        if (
            self.owner == "correct_owner"
            and self.decoded_owner_full_tensor_sha256
            != self.decoded_target_tensor_sha256
        ):
            fail("correct owner full tensor must be the unchanged target source tensor")
        if tuple(self.source_ids) != NATIVE_SOURCE_IDS:
            fail("native owner/target source IDs differ")

    @property
    def digest(self) -> str:
        return object_sha256(self.receipt())

    def receipt(self) -> Mapping[str, Any]:
        return {
            "owner": self.owner,
            "schedule_index": self.schedule_index,
            "timestep": self.timestep,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
            "orbit_row_digest": self.orbit_row_digest,
            "target_source_full_blob_sha256": self.target_source_full_blob_sha256,
            "owner_full_blob_sha256": self.owner_full_blob_sha256,
            "owner_reference_blob_sha256": list(self.owner_reference_blob_sha256),
            "decoded_target_tensor_sha256": self.decoded_target_tensor_sha256,
            "decoded_owner_full_tensor_sha256": self.decoded_owner_full_tensor_sha256,
            "decoded_owner_reference_tensor_sha256": list(
                self.decoded_owner_reference_tensor_sha256
            ),
            "epsilon_sha256": self.epsilon_sha256,
            "target_x_s_sha256": self.target_x_s_sha256,
            "prepared_visual_prefix_sha256": self.prepared_visual_prefix_sha256,
            "prepared_prefix_rotary_sha256": self.prepared_prefix_rotary_sha256,
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "source_ids": list(self.source_ids),
            "owner_pair_switch_audited_by_single_binding": False,
        }


def validate_owner_pair_bindings(
    correct: OwnerInputBinding, wrong: OwnerInputBinding
) -> Mapping[str, Any]:
    if (
        not isinstance(correct, OwnerInputBinding)
        or not isinstance(wrong, OwnerInputBinding)
        or correct.owner != "correct_owner"
        or wrong.owner != "wrong_owner"
    ):
        fail("owner pair binding order differs")
    shared_fields = (
        "orbit_row_digest",
        "target_source_full_blob_sha256",
        "decoded_target_tensor_sha256",
        "epsilon_sha256",
        "target_x_s_sha256",
        "source_ids",
        "prepared_prefix_rotary_sha256",
        "schedule_index",
        "timestep",
        "sigma_float32_be_hex",
        "total_tokens",
        "condition_tokens",
    )
    if any(getattr(correct, name) != getattr(wrong, name) for name in shared_fields):
        fail("wrong owner changed target/noise/source pins")
    owner_fields = (
        "owner_full_blob_sha256",
        "owner_reference_blob_sha256",
        "decoded_owner_full_tensor_sha256",
        "decoded_owner_reference_tensor_sha256",
        "prepared_visual_prefix_sha256",
    )
    if any(getattr(correct, name) == getattr(wrong, name) for name in owner_fields):
        fail("wrong owner did not switch every visual-prefix identity")
    value = {
        "correct_owner_input_binding_digest": correct.digest,
        "wrong_owner_input_binding_digest": wrong.digest,
        "shared_target_x_s_sha256": correct.target_x_s_sha256,
        "shared_epsilon_sha256": correct.epsilon_sha256,
        "shared_target_source_full_blob_sha256": correct.target_source_full_blob_sha256,
        "wrong_owner_changes_visual_prefix_only": True,
    }
    return {**value, "digest": object_sha256(value)}


def tensor_content_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    """Hash one small post-condition tensor and bind its mutation version."""

    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        fail(f"{label} must be a materialized torch Tensor")
    tensor = value.detach().contiguous()
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all().item()):
        fail(f"{label} must be finite floating point")
    raw = tensor.view(torch.uint8).cpu().numpy().tobytes(order="C")
    metadata = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor_version": _tensor_version_identity(value, label=label),
    }
    return {**metadata, "digest": object_sha256(metadata)}


def _require_runtime_tensor(value: Any, *, label: str) -> Any:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.ndim != 3
        or not value.is_floating_point()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        fail(f"{label} must be a finite graph-free floating [B,L,D] tensor")
    return value


def _tensors_alias(left: Any, right: Any) -> bool:
    import torch

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    predicate = getattr(torch._C, "_is_alias_of", None)
    if callable(predicate):
        return bool(predicate(left, right))
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def raw_tensor_bytes_equal(left: Any, right: Any) -> bool:
    """Compare tensor contract and bytes exactly, including floating sign bits."""

    import torch

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or tuple(left.shape) != tuple(right.shape)
        or left.dtype != right.dtype
        or left.device != right.device
        or left.layout != torch.strided
        or right.layout != torch.strided
    ):
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8).reshape(-1)
    right_bytes = right.detach().contiguous().view(torch.uint8).reshape(-1)
    return bool((left_bytes == right_bytes).all().item())


def _tensor_version_identity(value: Any, *, label: str) -> int | str:
    """Return the mutation version or an explicit inference-tensor sentinel.

    PyTorch inference tensors deliberately do not expose a version counter.
    Their raw bytes remain the mutation authority; the sentinel prevents that
    expected runtime property from being mistaken for an unhandled failure.
    """

    import torch

    try:
        return int(value._version)
    except RuntimeError as error:
        if bool(torch.is_inference(value)):
            return "inference_tensor_no_version"
        raise PromptSwapError(
            f"{label} tensor version is unavailable: {error}"
        ) from error


def _snapshot_tensor(value: Any, *, label: str) -> Mapping[str, Any]:
    tensor = _require_runtime_tensor(value, label=label)
    return {
        "clone": tensor.detach().clone(),
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "device": tensor.device,
        "version": _tensor_version_identity(tensor, label=label),
    }


def _assert_tensor_unchanged(
    value: Any, snapshot: Mapping[str, Any], *, label: str
) -> None:
    import torch

    tensor = _require_runtime_tensor(value, label=label)
    if (
        tuple(tensor.shape) != snapshot["shape"]
        or tensor.dtype != snapshot["dtype"]
        or tensor.device != snapshot["device"]
        or _tensor_version_identity(tensor, label=label) != snapshot["version"]
        or not raw_tensor_bytes_equal(tensor, snapshot["clone"])
    ):
        fail(f"{label} content, version, dtype, device, or shape changed")


def _validate_attention_frozen(attn: Any) -> None:
    import torch

    if torch.is_grad_enabled():
        fail("attn2 prompt swap requires inference/no-grad execution")
    parameters = getattr(attn, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if parameter.requires_grad or parameter.grad is not None:
                fail("attn2 parameter is trainable or carries a gradient")


def _validate_attn_output(value: Any, hidden_states: Any, encoder: Any) -> Any:
    import torch

    hidden = _require_runtime_tensor(hidden_states, label="attn2 hidden states")
    output = _require_runtime_tensor(value, label="attn2 output")
    if (
        tuple(output.shape) != tuple(hidden.shape)
        or output.dtype != hidden.dtype
        or output.device != hidden.device
        or _tensors_alias(output, hidden)
        or _tensors_alias(output, encoder)
    ):
        fail("official attn2 output tensor contract or aliasing differs")
    if not isinstance(value, torch.Tensor):  # pragma: no cover - narrowed above
        fail("official attn2 output is not a tensor")
    return output


def band_blocks(name: str) -> tuple[int, ...]:
    bands = dict(policy.REGISTERED_BLOCK_BANDS)
    if name == ALL30_BAND:
        return tuple(range(TOTAL_BLOCKS))
    if name == NONE_BAND:
        return ()
    if name not in bands:
        fail("block band is not preregistered")
    return tuple(bands[name])


@dataclass(frozen=True)
class NativeTargetSuffixLayout:
    """Exact native VI global suffix and append-pad/contiguous SP selector."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int = SP_SIZE

    def __post_init__(self) -> None:
        total = _exact_int(
            self.total_tokens, lower=1, upper=2**31, label="total tokens"
        )
        condition = _exact_int(
            self.condition_tokens,
            lower=0,
            upper=total,
            label="condition tokens",
        )
        size = _exact_int(
            self.sequence_parallel_size,
            lower=1,
            upper=5,
            label="sequence parallel size",
        )
        rank = _exact_int(
            self.sequence_parallel_rank,
            lower=0,
            upper=size,
            label="sequence parallel rank",
        )
        if size not in (1, SP_SIZE) or condition >= total:
            fail("layout requires a strict target suffix under SP1 or SP4")
        object.__setattr__(self, "total_tokens", total)
        object.__setattr__(self, "condition_tokens", condition)
        object.__setattr__(self, "sequence_parallel_rank", rank)
        object.__setattr__(self, "sequence_parallel_size", size)

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def padded_tokens(self) -> int:
        return self.local_length * self.sequence_parallel_size

    def local_target_selector(self, *, device: Any) -> Any:
        import torch

        selector = torch.cat(
            (
                torch.zeros(self.condition_tokens, dtype=torch.bool, device=device),
                torch.ones(self.target_tokens, dtype=torch.bool, device=device),
                torch.zeros(
                    self.padded_tokens - self.total_tokens,
                    dtype=torch.bool,
                    device=device,
                ),
            )
        )
        start = self.sequence_parallel_rank * self.local_length
        result = selector[start : start + self.local_length].contiguous()
        if int(result.numel()) != self.local_length:
            fail("local target selector length differs")
        return result

    def receipt(self) -> Mapping[str, Any]:
        starts = [rank * self.local_length for rank in range(self.sequence_parallel_size)]
        spans = [
            [start, start + self.local_length] for start in starts
        ]
        value = {
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "local_length": self.local_length,
            "padded_tokens": self.padded_tokens,
            "rank_spans_in_padded_global_sequence": spans,
            "padding_policy": "append_false_then_contiguous_rank_chunk",
            "target_is_strict_global_suffix": True,
        }
        return {**value, "digest": object_sha256(value)}


@dataclass
class PostConditionBranchCache:
    """Reusable read-only post-condition text tensors from one capture forward."""

    branch: str
    expected_block_indices: tuple[int, ...] = tuple(range(TOTAL_BLOCKS))
    _values: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _identities: dict[int, Mapping[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _capturing: bool = field(default=False, init=False, repr=False)
    _sealed: bool = field(default=False, init=False, repr=False)
    _capture_aborted: bool = field(default=False, init=False, repr=False)
    reuse_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.branch not in NONNOOP_BRANCHES:
            fail("only a non-noop text branch may own a capture cache")
        indices = tuple(self.expected_block_indices)
        if indices != tuple(range(TOTAL_BLOCKS)):
            fail("v1 capture cache must bind all 30 blocks")

    def begin_capture(self) -> None:
        if self._capturing or self._sealed or self._values or self._identities:
            fail("capture cache is active, sealed, or dirty")
        self._capturing = True

    def capture(self, block_index: int, value: Any) -> None:
        if not self._capturing or self._sealed:
            fail("capture cache is not active")
        expected = len(self._values)
        if block_index != expected or block_index not in self.expected_block_indices:
            fail(f"capture expected block {expected}, got {block_index}")
        shape = _shape(value, label="post-condition encoder tensor")
        if len(shape) != 3 or shape[0] != 1:
            fail("post-condition encoder tensor must be [1,L,D]")
        self._values[block_index] = value
        self._identities[block_index] = tensor_content_identity(
            value, label=f"block {block_index} captured encoder"
        )

    def finish_capture(self) -> None:
        if not self._capturing or self._sealed:
            fail("capture cache cannot finish")
        self._capturing = False
        if tuple(self._values) != self.expected_block_indices:
            self._values.clear()
            self._identities.clear()
            self._capture_aborted = True
            fail("capture did not visit the exact 30-block inventory")
        self._sealed = True

    def abort_capture(self) -> None:
        self._capturing = False
        self._values.clear()
        self._identities.clear()
        self._capture_aborted = True

    def get(self, block_index: int) -> Any:
        if not self._sealed or self._capturing:
            fail("branch cache is not sealed")
        if block_index not in self._values:
            fail("branch cache lacks the selected block")
        current = tensor_content_identity(
            self._values[block_index],
            label=f"block {block_index} cached encoder reuse",
        )
        if current != self._identities[block_index]:
            fail("captured branch encoder content or mutation version changed")
        self.reuse_count += 1
        return self._values[block_index]

    def assert_unchanged(self) -> Mapping[str, Any]:
        if not self._sealed or tuple(self._values) != self.expected_block_indices:
            fail("cannot audit an unsealed branch cache")
        current = {
            index: tensor_content_identity(
                value, label=f"block {index} terminal cached encoder"
            )
            for index, value in self._values.items()
        }
        if current != self._identities:
            fail("captured branch encoder content changed after capture")
        value = {
            "branch": self.branch,
            "block_identity_digest": object_sha256(
                {str(index): dict(identity) for index, identity in current.items()}
            ),
            "all_30_content_and_versions_unchanged": True,
        }
        return {**value, "digest": object_sha256(value)}

    @property
    def sealed(self) -> bool:
        return self._sealed

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "branch": self.branch,
            "expected_block_indices": list(self.expected_block_indices),
            "captured_block_indices": list(self._values),
            "captured_shapes": {
                str(index): list(_shape(value, label="cached encoder"))
                for index, value in self._values.items()
            },
            "captured_content_identity_by_block": {
                str(index): dict(identity)
                for index, identity in self._identities.items()
            },
            "sealed": self._sealed,
            "capturing": self._capturing,
            "capture_aborted": self._capture_aborted,
            "reuse_count": self.reuse_count,
            "captured_hidden_or_output_reused": False,
            "captured_text_encoder_state_only": True,
        }
        return {**value, "digest": object_sha256(value)}


@dataclass(frozen=True)
class PromptSwapInvocation:
    phase: str
    schedule_index: int
    band_name: str
    branch: str
    owner: str
    owner_binding: OwnerInputBinding
    prompt_object: Any
    layout: NativeTargetSuffixLayout
    cache: Optional[PostConditionBranchCache] = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            fail("prompt-swap phase differs")
        if self.schedule_index not in policy.REGISTERED_SCHEDULE_INDICES:
            fail("schedule index is not registered")
        if self.branch not in TEXT_BRANCHES or self.owner not in OWNERS:
            fail("text branch or owner axis differs")
        if (
            not isinstance(self.owner_binding, OwnerInputBinding)
            or self.owner_binding.owner != self.owner
        ):
            fail("invocation owner axis is not bound to its sealed visual prefix")
        if not isinstance(self.layout, NativeTargetSuffixLayout):
            fail("invocation layout differs")
        if (
            self.owner_binding.schedule_index != self.schedule_index
            or self.owner_binding.total_tokens != self.layout.total_tokens
            or self.owner_binding.condition_tokens != self.layout.condition_tokens
        ):
            fail("invocation schedule/layout differs from its owner input binding")
        selected = band_blocks(self.band_name)
        if self.phase == "capture":
            if self.branch not in NONNOOP_BRANCHES or self.band_name != ALL30_BAND:
                fail("capture must bind one non-noop branch across all30")
            if not isinstance(self.cache, PostConditionBranchCache):
                fail("capture requires its branch cache")
            if self.cache.branch != self.branch:
                fail("capture branch/cache provenance differs")
        elif self.phase in ("plain_noop", "noop_swap"):
            if self.branch != "noop" or self.cache is not None:
                fail("noop audit may not consume a branch cache")
            if self.phase == "plain_noop" and selected:
                fail("plain noop has no selected block")
            if self.phase == "noop_swap" and not selected:
                fail("noop swap requires a registered nonempty band")
        else:
            if self.branch not in NONNOOP_BRANCHES:
                fail("mixed invocation requires a non-noop branch")
            if not isinstance(self.cache, PostConditionBranchCache):
                fail("mixed invocation requires a branch cache")
            if self.cache.branch != self.branch or not self.cache.sealed or not selected:
                fail("mixed invocation cache/band differs")

    @property
    def selected_blocks(self) -> tuple[int, ...]:
        return band_blocks(self.band_name)


_ACTIVE_INVOCATION: ContextVar[Optional[PromptSwapInvocation]] = ContextVar(
    "bernini_schedule_block_target_row_prompt_swap", default=None
)


def active_invocation() -> Optional[PromptSwapInvocation]:
    return _ACTIVE_INVOCATION.get()


@contextmanager
def activate_prompt_swap(
    invocation: PromptSwapInvocation, *, encoder_hidden_states: Any
) -> Iterator[PromptSwapInvocation]:
    if not isinstance(invocation, PromptSwapInvocation):
        fail("prompt-swap context requires PromptSwapInvocation")
    if active_invocation() is not None:
        fail("nested prompt-swap invocations are forbidden")
    if encoder_hidden_states is not invocation.prompt_object:
        fail("invocation is not bound to the raw transformer prompt object")
    if invocation.owner_binding.owner != invocation.owner:
        fail("prompt-swap context owner binding changed")
    if invocation.phase == "capture":
        assert invocation.cache is not None
        invocation.cache.begin_capture()
    token: Token[Optional[PromptSwapInvocation]] = _ACTIVE_INVOCATION.set(invocation)
    try:
        yield invocation
    except BaseException:
        if invocation.phase == "capture" and invocation.cache is not None:
            invocation.cache.abort_capture()
        raise
    else:
        if invocation.phase == "capture" and invocation.cache is not None:
            invocation.cache.finish_capture()
    finally:
        _ACTIVE_INVOCATION.reset(token)


class TargetRowPromptSwapProcessor:
    """Explicit-signature wrapper around one official Bernini attn2 processor."""

    def __init__(self, base_processor: Any, *, block_index: int) -> None:
        if not callable(base_processor):
            fail("base attn2 processor must be callable")
        self.base_processor = base_processor
        self.block_index = _exact_int(
            block_index, lower=0, upper=TOTAL_BLOCKS, label="block index"
        )
        self.base_calls = 0
        self.alternate_calls = 0
        self.capture_calls = 0
        self.plain_noop_calls = 0
        self.noop_swap_selected_calls = 0
        self.noop_swap_unselected_calls = 0
        self.mixed_selected_calls = 0
        self.mixed_unselected_calls = 0
        self.no_context_calls = 0
        self.non_target_parity_checks = 0
        self.noop_full_parity_checks = 0
        self.owner_binding_digests: set[str] = set()

    def _base(
        self,
        attn: Any,
        hidden_states: Any,
        *,
        encoder_hidden_states: Any,
        attention_mask: Any,
        rotary_emb: Any,
        batch_image_vae_seqlen: Any,
        text_features_length: Any,
        origin_hidden_states_seq_len: Any,
        split_hidden_states_seq_len: Any,
        cu_seqlens_q_cache: Any,
        max_seqlen_q_cache: Any,
        cu_seqlens_k_cross_cache: Any,
        cu_seqlens_q_cross_cache: Any,
        max_seqlen_k_cross_cache: Any,
        max_seqlen_q_cross_cache: Any,
    ) -> Any:
        import torch

        with torch.no_grad():
            value = self.base_processor(
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
        return _validate_attn_output(value, hidden_states, encoder_hidden_states)

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        rotary_emb: Optional[Any] = None,
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
    ) -> Any:
        _validate_attention_frozen(attn)
        _require_runtime_tensor(hidden_states, label="current noop hidden states")
        _require_runtime_tensor(
            encoder_hidden_states, label="current post-condition noop encoder"
        )
        kwargs = {
            "attention_mask": attention_mask,
            "rotary_emb": rotary_emb,
            "batch_image_vae_seqlen": batch_image_vae_seqlen,
            "text_features_length": text_features_length,
            "origin_hidden_states_seq_len": origin_hidden_states_seq_len,
            "split_hidden_states_seq_len": split_hidden_states_seq_len,
            "cu_seqlens_q_cache": cu_seqlens_q_cache,
            "max_seqlen_q_cache": max_seqlen_q_cache,
            "cu_seqlens_k_cross_cache": cu_seqlens_k_cross_cache,
            "cu_seqlens_q_cross_cache": cu_seqlens_q_cross_cache,
            "max_seqlen_k_cross_cache": max_seqlen_k_cross_cache,
            "max_seqlen_q_cross_cache": max_seqlen_q_cross_cache,
        }
        invocation = active_invocation()
        selected = (
            self.block_index in invocation.selected_blocks
            if invocation is not None
            else False
        )
        guard_dual_inputs = invocation is not None and (
            invocation.phase == "capture"
            or (selected and invocation.phase in ("noop_swap", "mixed"))
        )
        hidden_snapshot = (
            _snapshot_tensor(hidden_states, label="guarded noop hidden states")
            if guard_dual_inputs
            else None
        )
        encoder_snapshot = (
            _snapshot_tensor(encoder_hidden_states, label="guarded noop encoder")
            if guard_dual_inputs
            else None
        )
        if invocation is not None:
            if invocation.owner_binding.owner != invocation.owner:
                fail("active owner binding differs from invocation owner")
            self.owner_binding_digests.add(invocation.owner_binding.digest)
            if invocation.phase == "capture":
                assert invocation.cache is not None
                invocation.cache.capture(self.block_index, encoder_hidden_states)
        base_output = self._base(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        if hidden_snapshot is not None and encoder_snapshot is not None:
            _assert_tensor_unchanged(
                hidden_states, hidden_snapshot, label="noop hidden after first attn2 call"
            )
            _assert_tensor_unchanged(
                encoder_hidden_states,
                encoder_snapshot,
                label="noop encoder after first attn2 call",
            )
        if invocation is None:
            self.no_context_calls += 1
            return base_output
        if invocation.phase == "capture":
            self.capture_calls += 1
            return base_output
        if invocation.phase == "plain_noop":
            self.plain_noop_calls += 1
            return base_output
        if invocation.phase == "noop_swap":
            if not selected:
                self.noop_swap_unselected_calls += 1
                return base_output
            repeated_noop = self._base(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            self.alternate_calls += 1
            import torch

            assert hidden_snapshot is not None and encoder_snapshot is not None
            _assert_tensor_unchanged(
                hidden_states, hidden_snapshot, label="noop hidden after second attn2 call"
            )
            _assert_tensor_unchanged(
                encoder_hidden_states,
                encoder_snapshot,
                label="noop encoder after second attn2 call",
            )
            if (
                base_output.dtype != repeated_noop.dtype
                or base_output.device != repeated_noop.device
                or tuple(base_output.shape) != tuple(repeated_noop.shape)
                or _tensors_alias(base_output, repeated_noop)
            ):
                fail("noop-swap official outputs differ in tensor contract or alias")
            selector = invocation.layout.local_target_selector(
                device=hidden_states.device
            )
            if int(hidden_states.shape[1]) != int(selector.numel()):
                fail("noop-swap local hidden/selector length differs")
            mask = selector.view(1, -1, 1).expand_as(base_output)
            mixed_noop = torch.where(mask, repeated_noop, base_output)
            if not raw_tensor_bytes_equal(
                repeated_noop, base_output
            ) or not raw_tensor_bytes_equal(
                mixed_noop, base_output
            ) or mixed_noop.dtype != hidden_states.dtype or mixed_noop.device != hidden_states.device:
                fail("noop-swap selected-band output is not bit-exact plain noop")
            self.noop_swap_selected_calls += 1
            self.noop_full_parity_checks += 1
            self.non_target_parity_checks += 1
            return mixed_noop
        if not selected:
            self.mixed_unselected_calls += 1
            return base_output
        assert invocation.cache is not None
        branch_encoder = invocation.cache.get(self.block_index)
        _require_runtime_tensor(branch_encoder, label="captured branch encoder")
        if (
            _shape(branch_encoder, label="branch encoder")
            != _shape(encoder_hidden_states, label="noop encoder")
            or branch_encoder.dtype != encoder_hidden_states.dtype
            or branch_encoder.device != encoder_hidden_states.device
        ):
            fail("captured branch/noop encoder tensor contract differs")
        assert hidden_snapshot is not None and encoder_snapshot is not None
        branch_snapshot = _snapshot_tensor(
            branch_encoder, label="guarded captured branch encoder"
        )
        branch_output = self._base(
            attn,
            hidden_states,
            encoder_hidden_states=branch_encoder,
            **kwargs,
        )
        self.alternate_calls += 1
        import torch

        _assert_tensor_unchanged(
            hidden_states, hidden_snapshot, label="noop hidden after branch attn2 call"
        )
        _assert_tensor_unchanged(
            encoder_hidden_states,
            encoder_snapshot,
            label="noop encoder after branch attn2 call",
        )
        _assert_tensor_unchanged(
            branch_encoder,
            branch_snapshot,
            label="captured branch encoder after branch attn2 call",
        )
        if (
            base_output.dtype != branch_output.dtype
            or base_output.device != branch_output.device
            or tuple(base_output.shape) != tuple(branch_output.shape)
            or _tensors_alias(base_output, branch_output)
        ):
            fail("noop/branch attn2 outputs differ in tensor contract or alias")
        selector = invocation.layout.local_target_selector(
            device=hidden_states.device
        )
        if int(hidden_states.shape[1]) != int(selector.numel()):
            fail("local hidden sequence differs from native suffix selector")
        mask = selector.view(1, -1, 1).expand_as(base_output)
        mixed = torch.where(mask, branch_output, base_output)
        if not raw_tensor_bytes_equal(
            mixed.masked_select(~mask), base_output.masked_select(~mask)
        ):
            fail("non-target prompt-swap rows differ from noop output")
        if not bool(torch.isfinite(mixed).all().item()):
            fail("mixed prompt-swap output is non-finite")
        if mixed.dtype != hidden_states.dtype or mixed.device != hidden_states.device:
            fail("mixed prompt-swap dtype/device differs")
        self.non_target_parity_checks += 1
        self.mixed_selected_calls += 1
        return mixed

    def statistics(self) -> Mapping[str, Any]:
        return {
            "block_index": self.block_index,
            "base_calls": self.base_calls,
            "alternate_calls": self.alternate_calls,
            "capture_calls": self.capture_calls,
            "plain_noop_calls": self.plain_noop_calls,
            "noop_swap_selected_calls": self.noop_swap_selected_calls,
            "noop_swap_unselected_calls": self.noop_swap_unselected_calls,
            "mixed_selected_calls": self.mixed_selected_calls,
            "mixed_unselected_calls": self.mixed_unselected_calls,
            "no_context_calls": self.no_context_calls,
            "non_target_parity_checks": self.non_target_parity_checks,
            "noop_full_parity_checks": self.noop_full_parity_checks,
            "owner_input_binding_digests": sorted(self.owner_binding_digests),
        }


@dataclass
class PromptSwapPatchHandle:
    transformer: Any
    attn2_modules: tuple[Any, ...]
    processors: tuple[TargetRowPromptSwapProcessor, ...]
    originals: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS:
            fail("transformer block inventory changed before restore")
        for index, processor in enumerate(self.processors):
            if (
                blocks[index].attn2 is not self.attn2_modules[index]
                or getattr(blocks[index].attn2, "processor", None) is not processor
            ):
                fail("attn2 processor changed behind prompt-swap handle")
        attempted: list[int] = []
        try:
            for index, original in enumerate(self.originals):
                attempted.append(index)
                _assign_processor(self.attn2_modules[index], original)
        except BaseException as error:
            _rollback_processors(
                self.attn2_modules,
                self.processors,
                label="restore rollback",
            )
            raise PromptSwapError(
                f"attn2 restore failed transactionally: {type(error).__name__}: {error}"
            ) from error
        if any(
            getattr(attn, "processor", None) is not original
            for attn, original in zip(self.attn2_modules, self.originals)
        ):
            _rollback_processors(
                self.attn2_modules,
                self.processors,
                label="restore verification rollback",
            )
            fail("attn2 restore terminal verification failed")
        self.restored = True

    def statistics(self) -> list[Mapping[str, Any]]:
        return [processor.statistics() for processor in self.processors]

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "installed_block_indices": list(range(TOTAL_BLOCKS)),
            "installed_projection": "blocks.{0..29}.attn2.processor",
            "base_processor_reused_for_noop_and_branch": True,
            "same_current_hidden_states_for_both_selected_calls": True,
            "non_target_rows_select_noop_output": True,
            "unselected_blocks_single_official_call": True,
            "capture_text_condition_only": True,
            "capture_hidden_or_output_reused": False,
            "owner_binding_required_by_every_context": True,
            "installation_and_restore_transactional": True,
            "inference_no_grad_required": True,
            "attention_parameters_frozen_and_grad_free_required": True,
            "optimizer_present": False,
            "parameter_update_authorized": False,
            "restored": self.restored,
            "statistics": self.statistics(),
        }
        return {**value, "digest": object_sha256(value)}

    def __enter__(self) -> "PromptSwapPatchHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def resolve_transformer(model: Any) -> Any:
    queue = [model]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            if len(blocks) != TOTAL_BLOCKS:
                fail("Bernini-R 1.3B transformer must expose exactly 30 blocks")
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
    fail("could not resolve the single 30-block Bernini transformer")


def _assign_processor(attn: Any, value: Any) -> None:
    setter = getattr(attn, "set_processor", None)
    if callable(setter):
        setter(value)
    else:
        setattr(attn, "processor", value)
    if getattr(attn, "processor", None) is not value:
        fail("attn2 setter did not install the exact processor object")


def _rollback_processors(
    attn_modules: Sequence[Any], values: Sequence[Any], *, label: str
) -> None:
    failures: list[int] = []
    for index, (attn, value) in enumerate(zip(attn_modules, values)):
        try:
            setattr(attn, "processor", value)
        except BaseException:
            try:
                setter = getattr(attn, "set_processor", None)
                if not callable(setter):
                    raise
                setter(value)
            except BaseException:
                failures.append(index)
                continue
        if getattr(attn, "processor", None) is not value:
            failures.append(index)
    if failures:
        fail(f"{label} could not restore attn2 blocks {failures}")


def install_prompt_swap_processors(model: Any) -> PromptSwapPatchHandle:
    transformer = resolve_transformer(model)
    blocks = tuple(transformer.blocks)
    attn_modules: list[Any] = []
    originals: list[Any] = []
    processors: list[TargetRowPromptSwapProcessor] = []
    for index, block in enumerate(blocks):
        attn2 = getattr(block, "attn2", None)
        original = getattr(attn2, "processor", None)
        if original is None or isinstance(original, TargetRowPromptSwapProcessor):
            fail(f"block {index} attn2 processor is absent or already wrapped")
        setter = getattr(attn2, "set_processor", None)
        if not callable(setter) and not hasattr(attn2, "processor"):
            fail(f"block {index} attn2 cannot replace its processor")
        attn_modules.append(attn2)
        originals.append(original)
        processors.append(TargetRowPromptSwapProcessor(original, block_index=index))
    attempted: list[int] = []
    try:
        for index, (attn2, wrapper) in enumerate(zip(attn_modules, processors)):
            attempted.append(index)
            _assign_processor(attn2, wrapper)
    except BaseException as error:
        _rollback_processors(attn_modules, originals, label="installation rollback")
        raise PromptSwapError(
            f"attn2 installation failed transactionally: {type(error).__name__}: {error}"
        ) from error
    if any(
        getattr(attn, "processor", None) is not wrapper
        for attn, wrapper in zip(attn_modules, processors)
    ):
        _rollback_processors(attn_modules, originals, label="installation verification rollback")
        fail("attn2 installation terminal verification failed")
    return PromptSwapPatchHandle(
        transformer=transformer,
        attn2_modules=tuple(attn_modules),
        processors=tuple(processors),
        originals=tuple(originals),
    )


@dataclass(frozen=True)
class ExperimentCell:
    ordinal: int
    profile: str
    owner: str
    schedule_index: int
    band_name: str
    branch: str
    role: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            fail("cell ordinal differs")
        if self.profile not in ("c0-smoke", "full-grid"):
            fail("cell profile differs")
        if self.owner not in OWNERS or self.schedule_index not in policy.REGISTERED_SCHEDULE_INDICES:
            fail("cell owner/schedule differs")
        if self.branch not in TEXT_BRANCHES:
            fail("cell text branch differs")
        selected = band_blocks(self.band_name)
        if self.role == "noop_baseline":
            if self.branch != "noop" or self.band_name != NONE_BAND or selected:
                fail("noop baseline cell differs")
        elif self.role == "all30_reference":
            if self.branch != "forward" or self.band_name != ALL30_BAND:
                fail("all30 reference cell differs")
        elif self.role == "selected_band_intervention":
            if self.branch == "noop" or self.band_name in (NONE_BAND, ALL30_BAND):
                fail("selected-band intervention differs")
        else:
            fail("cell role differs")

    @property
    def cell_id(self) -> str:
        return (
            f"{self.profile}-s{self.schedule_index}-{self.owner}-"
            f"{self.band_name}-{self.branch}"
        )

    @property
    def output_name(self) -> str:
        return f"{self.ordinal:03d}_{self.cell_id}.mp4"

    def receipt(self) -> Mapping[str, Any]:
        return {
            "ordinal": self.ordinal,
            "cell_id": self.cell_id,
            "profile": self.profile,
            "owner": self.owner,
            "schedule_index": self.schedule_index,
            "band_name": self.band_name,
            "selected_blocks": list(band_blocks(self.band_name)),
            "branch": self.branch,
            "role": self.role,
            "output_name": self.output_name,
        }


def c0_smoke_cells() -> tuple[ExperimentCell, ...]:
    rows: list[ExperimentCell] = []
    for owner in OWNERS:
        rows.extend(
            (
                ExperimentCell(
                    len(rows), "c0-smoke", owner, C0_SCHEDULE_INDEX,
                    NONE_BAND, "noop", "noop_baseline",
                ),
                ExperimentCell(
                    len(rows) + 1, "c0-smoke", owner, C0_SCHEDULE_INDEX,
                    C0_BAND, "forward", "selected_band_intervention",
                ),
                ExperimentCell(
                    len(rows) + 2, "c0-smoke", owner, C0_SCHEDULE_INDEX,
                    ALL30_BAND, "forward", "all30_reference",
                ),
            )
        )
    if len(rows) != 6:
        fail("C0 output plan must contain exactly six cells")
    return tuple(rows)


def full_grid_cells() -> tuple[ExperimentCell, ...]:
    rows: list[ExperimentCell] = []
    bands = tuple(name for name, _ in policy.REGISTERED_BLOCK_BANDS)
    for schedule_index in policy.REGISTERED_SCHEDULE_INDICES:
        owner = "correct_owner"
        rows.append(ExperimentCell(len(rows), "full-grid", owner, schedule_index, NONE_BAND, "noop", "noop_baseline"))
        rows.append(ExperimentCell(len(rows), "full-grid", owner, schedule_index, ALL30_BAND, "forward", "all30_reference"))
        for band in bands:
            for branch in NONNOOP_BRANCHES:
                rows.append(ExperimentCell(len(rows), "full-grid", owner, schedule_index, band, branch, "selected_band_intervention"))
        owner = "wrong_owner"
        rows.append(ExperimentCell(len(rows), "full-grid", owner, schedule_index, NONE_BAND, "noop", "noop_baseline"))
        rows.append(ExperimentCell(len(rows), "full-grid", owner, schedule_index, ALL30_BAND, "forward", "all30_reference"))
        for band in bands:
            rows.append(ExperimentCell(len(rows), "full-grid", owner, schedule_index, band, "forward", "selected_band_intervention"))
    if len(rows) != 112:
        fail("full decoded grid must contain exactly 112 cells")
    return tuple(rows)


def build_plan(profile: str) -> Mapping[str, Any]:
    if profile == "c0-smoke":
        cells = c0_smoke_cells()
    elif profile == "full-grid":
        cells = full_grid_cells()
    else:
        fail("unknown plan profile")
    value = {
        "schema_version": PLAN_SCHEMA,
        "profile": profile,
        "policy_schema_version": policy.SCHEMA_VERSION,
        "registered_grid_sha256": policy.REGISTERED_GRID_SHA256,
        "schedule_indices": list(policy.REGISTERED_SCHEDULE_INDICES),
        "block_bands": {
            name: list(indices) for name, indices in policy.REGISTERED_BLOCK_BANDS
        },
        "owners": list(OWNERS),
        "text_branches": list(TEXT_BRANCHES),
        "branch_calibration_authority": branch_calibration_authority(),
        "only_reverse_is_directional_negative_candidate": True,
        "noop_is_numerical_not_semantic_baseline": True,
        "incomplete_is_exploratory_calibration_failed": True,
        "cross_branch_identity_background_confounded": True,
        "negative_cluster_semantically_validated": False,
        "branch_calibration_may_not_remove_or_select_cells": True,
        "branch_calibration_scientific_veto_authorized": False,
        "forward_prompt_authority_mapping": "forward<-branch_descriptions.action",
        "prompt_authority_iid": PROMPT_AUTHORITY_IID,
        "prompt_authority_sha256": PROMPT_AUTHORITY_SHA256,
        "wrong_owner_variant": WRONG_OWNER_VARIANT,
        "wrong_owner_video_sha256": WRONG_OWNER_VIDEO_SHA256,
        "decoded_output_count": len(cells),
        "internal_noop_parity_outputs_published": 0,
        "cells": [cell.receipt() for cell in cells],
        "optimizer_present": False,
        "gradient_computation": False,
        "scheduler_steps": 0,
        "decoded_stateless_x0_hat_required": True,
        "visual_adaptive_cell_selection": False,
        "method_success_claimed": False,
    }
    return {**value, "plan_digest": object_sha256(value)}


def validate_plan(value: Any, *, profile: str) -> Mapping[str, Any]:
    expected = build_plan(profile)
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != canonical_json_bytes(expected):
        fail("experiment plan differs from the deterministic preregistration")
    return value


__all__ = (
    "ALL30_BAND",
    "BRANCH_CALIBRATION_AUTHORITY",
    "C0_BAND",
    "C0_SCHEDULE_INDEX",
    "ExperimentCell",
    "NativeTargetSuffixLayout",
    "NONNOOP_BRANCHES",
    "NATIVE_SOURCE_IDS",
    "ORBIT_ROW_DIGEST",
    "OWNERS",
    "OWNER_FULL_BLOB_SHA256",
    "OWNER_REFERENCE_BLOB_SHA256",
    "OwnerInputBinding",
    "PLAN_SCHEMA",
    "PHASES",
    "PostConditionBranchCache",
    "PromptSwapError",
    "PromptSwapInvocation",
    "PromptSwapPatchHandle",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "TEXT_BRANCHES",
    "TargetRowPromptSwapProcessor",
    "activate_prompt_swap",
    "active_invocation",
    "band_blocks",
    "build_plan",
    "c0_smoke_cells",
    "full_grid_cells",
    "install_prompt_swap_processors",
    "object_sha256",
    "raw_tensor_bytes_equal",
    "resolve_transformer",
    "validate_owner_pair_bindings",
    "validate_plan",
)
