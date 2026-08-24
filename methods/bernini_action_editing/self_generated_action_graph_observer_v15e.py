#!/usr/bin/env python3
"""Observer-only self-generated-action Q/K -> role graph preflight.

This module is intentionally not an editor.  It intercepts the *existing*
official Bernini ``attn1`` ``_project_qkv`` call, retains detached post-RoPE
Q/K in process memory, and returns the exact object produced by the official
processor.  A separate pure tensor reduction integrates out the anchor's
spatial axis and emits only a role-labelled temporal relation graph.

The intended production order is::

    dynamic anchor re-forward + phase-0-static re-forward
        -> post-RoPE Q/K capture (rank-local heads, global sequence)
        -> anchor-local generic-role mask pooling
        -> G[head, 21, role_q, 21, role_k]
        -> discard raw Q/K and masks
        -> explicit SP4 head gather outside attention
        -> four-appearance consensus in a later, still observer-only stage

No public API in this file accepts target tensors, source appearance memory,
an optimizer, a route strength, or a decoder.  Receipts keep route, decode,
training, and scientific-claim authorization false even when the mechanical
representation gates pass.  In particular, a one-anchor result is a
``representation_candidate`` and is never a v15b ``AnchorRelationGraphV15B``.

The MEV 840 ``place`` action is registered here only for representation
diagnostics.  This does not register ``place`` in the v15b editor/signed-graph
ABI and therefore cannot silently make it routable.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


METHOD = "bernini-self-generated-action-graph-observer-v15e"
CAPTURE_SCHEMA = "bernini-self-action-post-rope-qk-capture-v15e"
MASK_SCHEMA = "bernini-self-action-generic-role-mask-authority-v15e"
CANDIDATE_SCHEMA = "bernini-self-action-role-graph-candidate-v15e"
RECEIPT_SCHEMA = "bernini-self-action-role-graph-preflight-v15e"

LATENT_PHASES = 21
EXPECTED_BLOCK_COUNT = 30
GENERIC_ROLES = ("human_agent", "moving_object", "recipient")
TRACE_CHANNELS = GENERIC_ROLES + ("contact",)
ARMS = ("dynamic", "phase0_static")
OFFICIAL_PROCESSOR_MODULE = "bernini.models.transformer_wan"
OFFICIAL_PROCESSOR_CLASS = "WanAttnProcessor2_0"

# These are extraction allowlists, not editor authorization.  Every other
# role pair is zeroed before the candidate can leave the anchor process.
ACTION_ALLOWED_EDGES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "pour": (
        ("human_agent", "human_agent"),
        ("moving_object", "moving_object"),
        ("recipient", "recipient"),
        ("human_agent", "moving_object"),
        ("moving_object", "recipient"),
    ),
    "place": (
        ("human_agent", "human_agent"),
        ("moving_object", "moving_object"),
        ("recipient", "recipient"),
        ("human_agent", "moving_object"),
        ("moving_object", "recipient"),
    ),
}
ACTION_REQUIRED_EDGES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "pour": (
        ("human_agent", "moving_object"),
        ("moving_object", "recipient"),
    ),
    "place": (
        ("human_agent", "moving_object"),
        ("moving_object", "recipient"),
    ),
}

MIN_ROLE_PIXELS_PER_PHASE = 1
MIN_REQUIRED_EDGE_NORM = 1.0e-4
MIN_REQUIRED_EDGE_QUERY_PHASES = 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SelfActionGraphObserverV15EError(RuntimeError):
    """Fail-closed observer, tensor, provenance, or representation error."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelfActionGraphObserverV15EError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SelfActionGraphObserverV15EError(f"{label} is not lowercase SHA256")
    return value


def _role(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or ROLE_RE.fullmatch(value) is None:
        raise SelfActionGraphObserverV15EError(f"{label} is not a canonical role/action ID")
    return value


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelfActionGraphObserverV15EError(f"{label} is outside its integer domain")
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise SelfActionGraphObserverV15EError("tensor digest requires a material tensor")
    logical = value.detach().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(logical.dtype), "shape": [int(item) for item in logical.shape]}
    )
    try:
        raw = logical.view(torch.uint8).cpu().numpy().tobytes(order="C")
    except Exception as error:
        raise SelfActionGraphObserverV15EError("cannot materialize tensor digest") from error
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class AnchorRoleMaskAuthorityV15E:
    """Anchor-local generic-role tracks on the latent grid.

    The authority may be produced by frozen SAM2 plus a separately audited
    role assignment.  Source masks are deliberately invalid here because
    source and self-generated anchor coordinates do not share authority.
    """

    schema_version: str
    anchor_slot: str
    anchor_asset_sha256: str
    producer_receipt_sha256: str
    roles: tuple[str, ...]
    masks: torch.Tensor  # [21, 3, H, W], CPU bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != MASK_SCHEMA:
            raise SelfActionGraphObserverV15EError("anchor role-mask schema differs")
        if not isinstance(self.anchor_slot, str) or not re.fullmatch(r"v[0-9]+", self.anchor_slot):
            raise SelfActionGraphObserverV15EError("anchor role-mask slot differs")
        _sha256(self.anchor_asset_sha256, label="anchor asset")
        _sha256(self.producer_receipt_sha256, label="role-mask producer receipt")
        if self.roles != GENERIC_ROLES:
            raise SelfActionGraphObserverV15EError("anchor role vocabulary differs")
        mask = self.masks
        if (
            not isinstance(mask, torch.Tensor)
            or mask.device.type != "cpu"
            or mask.dtype != torch.bool
            or mask.ndim != 4
            or tuple(mask.shape[:2]) != (LATENT_PHASES, len(GENERIC_ROLES))
            or int(mask.shape[2]) < 1
            or int(mask.shape[3]) < 1
            or not mask.is_contiguous()
        ):
            raise SelfActionGraphObserverV15EError(
                "anchor generic-role masks must be contiguous CPU bool [21,3,H,W]"
            )
        counts = mask.flatten(2).sum(2)
        if bool((counts < MIN_ROLE_PIXELS_PER_PHASE).any().item()):
            raise SelfActionGraphObserverV15EError(
                "each anchor generic role must be observed in every latent phase"
            )
        payload = {
            "schema_version": self.schema_version,
            "anchor_slot": self.anchor_slot,
            "anchor_asset_sha256": self.anchor_asset_sha256,
            "producer_receipt_sha256": self.producer_receipt_sha256,
            "roles": list(self.roles),
            "shape": [int(item) for item in mask.shape],
            "masks_sha256": tensor_sha256(mask),
        }
        if object_sha256(payload) != _sha256(self.digest, label="role-mask digest"):
            raise SelfActionGraphObserverV15EError("anchor role-mask digest differs")

    @classmethod
    def create(
        cls,
        *,
        anchor_slot: str,
        anchor_asset_sha256: str,
        producer_receipt_sha256: str,
        masks: torch.Tensor,
    ) -> "AnchorRoleMaskAuthorityV15E":
        logical = masks.detach().to(device="cpu", dtype=torch.bool).contiguous()
        payload = {
            "schema_version": MASK_SCHEMA,
            "anchor_slot": anchor_slot,
            "anchor_asset_sha256": anchor_asset_sha256,
            "producer_receipt_sha256": producer_receipt_sha256,
            "roles": list(GENERIC_ROLES),
            "shape": [int(item) for item in logical.shape],
            "masks_sha256": tensor_sha256(logical),
        }
        return cls(
            MASK_SCHEMA,
            anchor_slot,
            anchor_asset_sha256,
            producer_receipt_sha256,
            GENERIC_ROLES,
            logical,
            object_sha256(payload),
        )


@dataclass(frozen=True)
class QKCaptureInvocationV15E:
    capture_bank: "QKCaptureBankV15E"
    arm: str
    anchor_slot: str
    anchor_asset_sha256: str
    caption_sha256: str
    initial_noise_sha256: str
    timestep_sha256: str
    pairing_receipt_sha256: str
    denoise_step_index: int
    rank: int
    sp_size: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if not isinstance(self.capture_bank, QKCaptureBankV15E):
            raise SelfActionGraphObserverV15EError("capture invocation bank differs")
        if self.arm not in ARMS:
            raise SelfActionGraphObserverV15EError("capture arm differs")
        if not isinstance(self.anchor_slot, str) or not re.fullmatch(r"v[0-9]+", self.anchor_slot):
            raise SelfActionGraphObserverV15EError("capture anchor slot differs")
        for label, value in (
            ("anchor asset", self.anchor_asset_sha256),
            ("caption", self.caption_sha256),
            ("initial noise", self.initial_noise_sha256),
            ("timestep", self.timestep_sha256),
            ("dynamic/static pairing receipt", self.pairing_receipt_sha256),
        ):
            _sha256(value, label=label)
        _exact_int(self.denoise_step_index, label="denoise step index")
        _exact_int(self.rank, label="rank")
        _exact_int(self.sp_size, label="SP size", minimum=1)
        _exact_int(self.height, label="latent height", minimum=1)
        _exact_int(self.width, label="latent width", minimum=1)
        if self.rank >= self.sp_size:
            raise SelfActionGraphObserverV15EError("rank is outside SP group")

    @property
    def global_tokens(self) -> int:
        return LATENT_PHASES * self.height * self.width


@dataclass(frozen=True)
class PostRopeQKCaptureV15E:
    schema_version: str
    arm: str
    anchor_slot: str
    anchor_asset_sha256: str
    caption_sha256: str
    initial_noise_sha256: str
    timestep_sha256: str
    pairing_receipt_sha256: str
    denoise_step_index: int
    block_index: int
    rank: int
    sp_size: int
    height: int
    width: int
    query: torch.Tensor
    key: torch.Tensor

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_SCHEMA or self.arm not in ARMS:
            raise SelfActionGraphObserverV15EError("Q/K capture schema/arm differs")
        for label, value in (
            ("anchor asset", self.anchor_asset_sha256),
            ("caption", self.caption_sha256),
            ("initial noise", self.initial_noise_sha256),
            ("timestep", self.timestep_sha256),
            ("dynamic/static pairing receipt", self.pairing_receipt_sha256),
        ):
            _sha256(value, label=label)
        _exact_int(self.denoise_step_index, label="denoise step index")
        _exact_int(self.block_index, label="block index")
        _exact_int(self.rank, label="rank")
        _exact_int(self.sp_size, label="SP size", minimum=1)
        if self.block_index >= EXPECTED_BLOCK_COUNT or self.rank >= self.sp_size:
            raise SelfActionGraphObserverV15EError("Q/K capture block/rank differs")
        expected_tokens = LATENT_PHASES * self.height * self.width
        if (
            not isinstance(self.query, torch.Tensor)
            or not isinstance(self.key, torch.Tensor)
            or self.query.ndim != 4
            or tuple(self.query.shape) != tuple(self.key.shape)
            or int(self.query.shape[0]) != 1
            or int(self.query.shape[1]) != expected_tokens
            or int(self.query.shape[2]) < 1
            or int(self.query.shape[3]) < 1
            or self.query.dtype != self.key.dtype
            or self.query.device != self.key.device
            or self.query.requires_grad
            or self.key.requires_grad
            or self.query.grad_fn is not None
            or self.key.grad_fn is not None
            or not bool(torch.isfinite(self.query).all().item())
            or not bool(torch.isfinite(self.key).all().item())
        ):
            raise SelfActionGraphObserverV15EError(
                "post-RoPE Q/K must be detached finite [1,21*H*W,local_heads,D]"
            )


class QKCaptureBankV15E:
    """In-memory only bank; it has deliberately no serialization method."""

    def __init__(self, selected_block_indices: Sequence[int]) -> None:
        blocks = tuple(selected_block_indices)
        if (
            not blocks
            or blocks != tuple(sorted(set(blocks)))
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item < EXPECTED_BLOCK_COUNT
                for item in blocks
            )
        ):
            raise SelfActionGraphObserverV15EError(
                "selected blocks must be an increasing subset of 0..29"
            )
        self.selected_block_indices = blocks
        self._captures: dict[tuple[str, str, int, int], PostRopeQKCaptureV15E] = {}
        self.capture_count = 0
        self.zeroized_capture_count = 0

    def capture(self, value: PostRopeQKCaptureV15E) -> None:
        if not isinstance(value, PostRopeQKCaptureV15E):
            raise SelfActionGraphObserverV15EError("capture bank input differs")
        if value.block_index not in self.selected_block_indices:
            raise SelfActionGraphObserverV15EError("capture block is outside scope")
        key = (value.anchor_slot, value.arm, value.block_index, value.rank)
        if key in self._captures:
            raise SelfActionGraphObserverV15EError("duplicate Q/K capture")
        self._captures[key] = value
        self.capture_count += 1

    def get(
        self, *, anchor_slot: str, arm: str, block_index: int, rank: int
    ) -> PostRopeQKCaptureV15E:
        value = self._captures.get((anchor_slot, arm, block_index, rank))
        if value is None:
            raise SelfActionGraphObserverV15EError("requested Q/K capture is absent")
        return value

    def clear(self) -> None:
        devices = set()
        with torch.no_grad():
            for value in self._captures.values():
                value.query.zero_()
                value.key.zero_()
                devices.add(value.query.device)
                self.zeroized_capture_count += 1
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        self._captures.clear()

    def receipt(self) -> Mapping[str, Any]:
        return {
            "schema_version": CAPTURE_SCHEMA,
            "selected_block_indices": list(self.selected_block_indices),
            "capture_count": self.capture_count,
            "zeroized_capture_count": self.zeroized_capture_count,
            "resident_capture_count": len(self._captures),
            "persistent_tensor_artifact_created": False,
            "captured_fields": ["post_rope_query", "post_rope_key"],
            "forbidden_fields": [
                "value",
                "hidden_state",
                "attention_output",
                "rgb",
                "latent",
                "gaussian",
                "absolute_spatial_coordinate",
            ],
        }


_ACTIVE_CAPTURE: ContextVar[Optional[QKCaptureInvocationV15E]] = ContextVar(
    "bernini_self_action_graph_capture_v15e", default=None
)


@contextmanager
def observe_self_action_qk_v15e(
    invocation: QKCaptureInvocationV15E,
) -> Iterator[None]:
    if not isinstance(invocation, QKCaptureInvocationV15E):
        raise SelfActionGraphObserverV15EError("observer invocation differs")
    if _ACTIVE_CAPTURE.get() is not None:
        raise SelfActionGraphObserverV15EError("nested Q/K observation is forbidden")
    token: Token[Optional[QKCaptureInvocationV15E]] = _ACTIVE_CAPTURE.set(invocation)
    try:
        yield
    finally:
        _ACTIVE_CAPTURE.reset(token)


def _validate_frozen_attn(attn: Any) -> None:
    if not isinstance(attn, torch.nn.Module):
        return
    if attn.training or any(parameter.requires_grad for parameter in attn.parameters()):
        raise SelfActionGraphObserverV15EError("observed attention must be frozen/eval")


class SelfActionQKAttn1ObserverV15E:
    """Intercept one official projection without replaying attention.

    The official processor is delegated exactly once.  During that call only,
    its instance-level ``_project_qkv`` attribute is replaced by a closure
    which calls the original bound method and clones Q/K.  This observes the
    same post-RoPE, post-Ulysses tensors consumed by official attention and
    introduces no second projection or additional SP collective.
    """

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        capture_bank: QKCaptureBankV15E,
    ) -> None:
        processor_type = type(base_processor)
        if (
            not callable(base_processor)
            or not callable(getattr(base_processor, "_project_qkv", None))
            or processor_type.__module__ != OFFICIAL_PROCESSOR_MODULE
            or processor_type.__name__ != OFFICIAL_PROCESSOR_CLASS
        ):
            raise SelfActionGraphObserverV15EError("base attn1 processor is not official")
        block = _exact_int(block_index, label="block index")
        if block not in capture_bank.selected_block_indices:
            raise SelfActionGraphObserverV15EError("observer block is outside capture scope")
        if "_project_qkv" in getattr(base_processor, "__dict__", {}):
            raise SelfActionGraphObserverV15EError(
                "official processor already has an instance projection override"
            )
        self.base_processor = base_processor
        self.block_index = block
        self.capture_bank = capture_bank
        self.base_calls = 0
        self.observer_calls = 0
        self.output_modified = False

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
        kwargs = {
            "encoder_hidden_states": encoder_hidden_states,
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
        invocation = _ACTIVE_CAPTURE.get()
        if invocation is None:
            output = self.base_processor(attn, hidden_states, **kwargs)
            self.base_calls += 1
            return output
        if invocation.capture_bank is not self.capture_bank:
            raise SelfActionGraphObserverV15EError("observer/capture-bank ownership differs")
        if encoder_hidden_states is not None or attention_mask is not None:
            raise SelfActionGraphObserverV15EError("action graph observer requires attn1")
        if rotary_emb is None:
            raise SelfActionGraphObserverV15EError("action graph observer requires post-RoPE Q/K")
        if origin_hidden_states_seq_len != invocation.global_tokens:
            raise SelfActionGraphObserverV15EError("observer global token geometry differs")
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.requires_grad
            or hidden_states.grad_fn is not None
        ):
            raise SelfActionGraphObserverV15EError(
                "action graph observer requires a detached frozen hidden stream"
            )
        _validate_frozen_attn(attn)

        original_projection = self.base_processor._project_qkv
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []

        def intercept_projection(*args: Any, **inner_kwargs: Any):
            if captured:
                raise SelfActionGraphObserverV15EError(
                    "official attn1 called _project_qkv more than once"
                )
            query, key, value = original_projection(*args, **inner_kwargs)
            captured.append((query.detach().clone(), key.detach().clone()))
            return query, key, value

        # The class method is restored by deleting the temporary instance
        # attribute.  The finally block runs even if official attention fails.
        self.base_processor._project_qkv = intercept_projection
        try:
            output = self.base_processor(attn, hidden_states, **kwargs)
        finally:
            try:
                delattr(self.base_processor, "_project_qkv")
            except AttributeError as error:
                raise SelfActionGraphObserverV15EError(
                    "temporary projection interception was lost"
                ) from error
        self.base_calls += 1
        if len(captured) != 1:
            raise SelfActionGraphObserverV15EError("official attn1 projection was not observed once")
        query, key = captured[0]
        self.capture_bank.capture(
            PostRopeQKCaptureV15E(
                CAPTURE_SCHEMA,
                invocation.arm,
                invocation.anchor_slot,
                invocation.anchor_asset_sha256,
                invocation.caption_sha256,
                invocation.initial_noise_sha256,
                invocation.timestep_sha256,
                invocation.pairing_receipt_sha256,
                invocation.denoise_step_index,
                self.block_index,
                invocation.rank,
                invocation.sp_size,
                invocation.height,
                invocation.width,
                query,
                key,
            )
        )
        self.observer_calls += 1
        return output

    def statistics(self) -> Mapping[str, Any]:
        return {
            "block_index": self.block_index,
            "base_calls": self.base_calls,
            "observer_calls": self.observer_calls,
            "official_projection_calls_added": 0,
            "SP_collective_calls_added": 0,
            "output_modified": False,
            "parameters_added": 0,
        }


@dataclass
class SelfActionQKObserverPatchHandleV15E:
    transformer: Any
    block_indices: tuple[int, ...]
    processors: tuple[SelfActionQKAttn1ObserverV15E, ...]
    originals: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        for block, wrapper in zip(self.block_indices, self.processors):
            if getattr(self.transformer.blocks[block].attn1, "processor", None) is not wrapper:
                raise SelfActionGraphObserverV15EError("attn1 observer changed behind handle")
        for block, original in zip(self.block_indices, self.originals):
            attn1 = self.transformer.blocks[block].attn1
            setter = getattr(attn1, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn1.processor = original
        self.restored = True

    def receipt(self) -> Mapping[str, Any]:
        return {
            "schema_version": CAPTURE_SCHEMA,
            "block_indices": list(self.block_indices),
            "attn1_only": True,
            "official_processor_delegated_once": True,
            "output_modified": False,
            "parameters_added": 0,
            "route_authorized": False,
            "decode_authorized": False,
            "training_authorized": False,
            "scientific_claim_authorized": False,
            "restored": self.restored,
            "processors": [item.statistics() for item in self.processors],
        }

    def __enter__(self) -> "SelfActionQKObserverPatchHandleV15E":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def _resolve_transformer(model: Any) -> Any:
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
                raise SelfActionGraphObserverV15EError("Bernini transformer must have 30 blocks")
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
    raise SelfActionGraphObserverV15EError("cannot resolve Bernini 30-block transformer")


def install_self_action_qk_observer_v15e(
    model: Any, *, capture_bank: QKCaptureBankV15E
) -> SelfActionQKObserverPatchHandleV15E:
    if not isinstance(capture_bank, QKCaptureBankV15E):
        raise SelfActionGraphObserverV15EError("observer install bank differs")
    transformer = _resolve_transformer(model)
    originals: list[Any] = []
    wrappers: list[SelfActionQKAttn1ObserverV15E] = []
    installed: list[int] = []
    try:
        for block in capture_bank.selected_block_indices:
            attn1 = transformer.blocks[block].attn1
            original = getattr(attn1, "processor", None)
            wrapper = SelfActionQKAttn1ObserverV15E(
                original, block_index=block, capture_bank=capture_bank
            )
            setter = getattr(attn1, "set_processor", None)
            if callable(setter):
                setter(wrapper)
            else:
                attn1.processor = wrapper
            if getattr(attn1, "processor", None) is not wrapper:
                raise SelfActionGraphObserverV15EError("attn1 observer installation did not stick")
            originals.append(original)
            wrappers.append(wrapper)
            installed.append(block)
    except Exception:
        for block, original in zip(installed, originals):
            attn1 = transformer.blocks[block].attn1
            setter = getattr(attn1, "set_processor", None)
            if callable(setter):
                setter(original)
            else:
                attn1.processor = original
        raise
    return SelfActionQKObserverPatchHandleV15E(
        transformer,
        tuple(installed),
        tuple(wrappers),
        tuple(originals),
    )


def _same_pair_authority(
    dynamic: PostRopeQKCaptureV15E, static: PostRopeQKCaptureV15E
) -> None:
    if dynamic.arm != "dynamic" or static.arm != "phase0_static":
        raise SelfActionGraphObserverV15EError("Q/K pair arm order differs")
    for field in (
        "anchor_slot",
        "anchor_asset_sha256",
        "caption_sha256",
        "initial_noise_sha256",
        "timestep_sha256",
        "pairing_receipt_sha256",
        "denoise_step_index",
        "block_index",
        "rank",
        "sp_size",
        "height",
        "width",
    ):
        if getattr(dynamic, field) != getattr(static, field):
            raise SelfActionGraphObserverV15EError(f"dynamic/static {field} differs")
    if (
        tuple(dynamic.query.shape) != tuple(static.query.shape)
        or dynamic.query.dtype != static.query.dtype
        or dynamic.query.device != static.query.device
    ):
        raise SelfActionGraphObserverV15EError("dynamic/static Q/K geometry differs")


def _masked_role_mean(
    value: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    # value [1,T,S,H,D], masks [T,R,S] -> [H,T,R,D]
    weight = masks.to(device=value.device, dtype=torch.float32)
    denom = weight.sum(2).clamp_min(1.0)
    pooled = torch.einsum("btshd,trs->htrd", value.float(), weight)
    return pooled / denom.unsqueeze(0).unsqueeze(-1)


def _mask_contact_trace(mask_4d: torch.Tensor) -> torch.Tensor:
    # Contact is geometry-only and combines hand-object and object-recipient
    # adjacency.  It carries timing but no absolute coordinate in the output.
    logical = mask_4d.to(dtype=torch.float32)
    dilated = F.max_pool2d(
        logical.reshape(LATENT_PHASES * len(GENERIC_ROLES), 1, *logical.shape[2:]),
        kernel_size=3,
        stride=1,
        padding=1,
    ).reshape_as(logical)
    human, moving, recipient = range(len(GENERIC_ROLES))

    def overlap(left: int, right: int) -> torch.Tensor:
        numerator = (dilated[:, left] * logical[:, right]).flatten(1).sum(1)
        denominator = logical[:, right].flatten(1).sum(1).clamp_min(1.0)
        return numerator / denominator

    return torch.maximum(overlap(human, moving), overlap(moving, recipient))


def _project_graph(graph: torch.Tensor) -> torch.Tensor:
    result = graph.float().clone()
    result -= result.mean(dim=3, keepdim=True)
    result[:, 0].zero_()
    return result


@dataclass(frozen=True)
class LocalActionGraphCandidateV15E:
    schema_version: str
    action_id: str
    anchor_slot: str
    anchor_asset_sha256: str
    mask_authority_digest: str
    block_index: int
    rank: int
    sp_size: int
    roles: tuple[str, ...]
    graph: torch.Tensor  # CPU fp32 [local_heads,21,3,21,3]
    timing_trace: torch.Tensor  # CPU fp32 [21,4]
    required_edge_metrics: tuple[tuple[str, str, float, int], ...]
    mechanically_qualified: bool
    digest: str

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA:
            raise SelfActionGraphObserverV15EError("graph candidate schema differs")
        _role(self.action_id, label="action")
        _sha256(self.anchor_asset_sha256, label="candidate anchor asset")
        _sha256(self.mask_authority_digest, label="candidate mask authority")
        if self.roles != GENERIC_ROLES or self.action_id not in ACTION_ALLOWED_EDGES:
            raise SelfActionGraphObserverV15EError("candidate action/roles differ")
        graph = self.graph
        if (
            graph.device.type != "cpu"
            or graph.dtype != torch.float32
            or graph.ndim != 5
            or tuple(graph.shape[1:])
            != (LATENT_PHASES, len(GENERIC_ROLES), LATENT_PHASES, len(GENERIC_ROLES))
            or not graph.is_contiguous()
            or int(torch.count_nonzero(graph[:, 0]).item()) != 0
            or float(graph.sum(3).abs().max().item()) > 1.0e-5
        ):
            raise SelfActionGraphObserverV15EError("candidate graph invariant differs")
        if (
            self.timing_trace.device.type != "cpu"
            or self.timing_trace.dtype != torch.float32
            or tuple(self.timing_trace.shape) != (LATENT_PHASES, len(TRACE_CHANNELS))
            or not self.timing_trace.is_contiguous()
            or float(self.timing_trace.min().item()) < 0.0
        ):
            raise SelfActionGraphObserverV15EError("candidate timing trace differs")
        expected_edges = ACTION_REQUIRED_EDGES[self.action_id]
        if tuple((item[0], item[1]) for item in self.required_edge_metrics) != expected_edges:
            raise SelfActionGraphObserverV15EError("required-edge metrics differ")
        expected_qualified = all(
            norm >= MIN_REQUIRED_EDGE_NORM
            and phases >= MIN_REQUIRED_EDGE_QUERY_PHASES
            for _query, _key, norm, phases in self.required_edge_metrics
        )
        if self.mechanically_qualified is not expected_qualified:
            raise SelfActionGraphObserverV15EError("mechanical graph gate differs")
        payload = self._payload()
        if object_sha256(payload) != _sha256(self.digest, label="candidate digest"):
            raise SelfActionGraphObserverV15EError("candidate digest differs")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "anchor_slot": self.anchor_slot,
            "anchor_asset_sha256": self.anchor_asset_sha256,
            "mask_authority_digest": self.mask_authority_digest,
            "block_index": self.block_index,
            "rank": self.rank,
            "sp_size": self.sp_size,
            "roles": list(self.roles),
            "graph_sha256": tensor_sha256(self.graph),
            "timing_trace_channels": list(TRACE_CHANNELS),
            "timing_trace_sha256": tensor_sha256(self.timing_trace),
            "required_edge_metrics": [list(item) for item in self.required_edge_metrics],
            "mechanically_qualified": self.mechanically_qualified,
        }


def extract_local_action_graph_v15e(
    dynamic: PostRopeQKCaptureV15E,
    static: PostRopeQKCaptureV15E,
    role_masks: AnchorRoleMaskAuthorityV15E,
    *,
    action_id: str,
) -> LocalActionGraphCandidateV15E:
    """Reduce one block/rank dynamic-static Q/K pair to a role-time graph."""

    if not isinstance(dynamic, PostRopeQKCaptureV15E) or not isinstance(
        static, PostRopeQKCaptureV15E
    ):
        raise SelfActionGraphObserverV15EError("extractor requires typed Q/K captures")
    if not isinstance(role_masks, AnchorRoleMaskAuthorityV15E):
        raise SelfActionGraphObserverV15EError("extractor requires anchor role masks")
    _same_pair_authority(dynamic, static)
    action = _role(action_id, label="action")
    allowed = ACTION_ALLOWED_EDGES.get(action)
    if allowed is None:
        raise SelfActionGraphObserverV15EError("action lacks extraction edge registry")
    if (
        role_masks.anchor_slot != dynamic.anchor_slot
        or role_masks.anchor_asset_sha256 != dynamic.anchor_asset_sha256
        or tuple(role_masks.masks.shape[2:]) != (dynamic.height, dynamic.width)
    ):
        raise SelfActionGraphObserverV15EError("role masks/QK anchor authority differs")

    spatial = dynamic.height * dynamic.width

    def phase(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            1,
            LATENT_PHASES,
            spatial,
            int(value.shape[2]),
            int(value.shape[3]),
        )

    masks = role_masks.masks.flatten(2)
    # The phase-0-static arm has no later-time object coordinates.  Pooling it
    # with the dynamic trajectory would sample background at vacated/future
    # sites and turn an anchor coordinate path into the contrast.  Its role
    # authority is therefore exactly the phase-0 role support repeated in
    # time, matching the phase-0-tiled latent construction.
    static_masks = masks[:1].repeat(LATENT_PHASES, 1, 1)
    dynamic_q = F.normalize(_masked_role_mean(phase(dynamic.query), masks), dim=-1)
    dynamic_k = F.normalize(_masked_role_mean(phase(dynamic.key), masks), dim=-1)
    static_q = F.normalize(
        _masked_role_mean(phase(static.query), static_masks), dim=-1
    )
    static_k = F.normalize(
        _masked_role_mean(phase(static.key), static_masks), dim=-1
    )
    head_dim = int(dynamic.query.shape[-1])
    scale = math.sqrt(head_dim)
    dynamic_logits = torch.einsum("htrd,husd->htrus", dynamic_q, dynamic_k) / scale
    static_logits = torch.einsum("htrd,husd->htrus", static_q, static_k) / scale
    graph = torch.softmax(dynamic_logits, dim=3) - torch.softmax(static_logits, dim=3)
    edge_mask = torch.zeros(
        len(GENERIC_ROLES), len(GENERIC_ROLES), device=graph.device, dtype=graph.dtype
    )
    for query_role, key_role in allowed:
        edge_mask[GENERIC_ROLES.index(query_role), GENERIC_ROLES.index(key_role)] = 1.0
    graph *= edge_mask[None, None, :, None, :]
    graph = _project_graph(graph).detach().cpu().contiguous()

    activity_rows = []
    for role_index in range(len(GENERIC_ROLES)):
        delta_q = dynamic_q[:, :, role_index] - static_q[:, :, role_index]
        delta_k = dynamic_k[:, :, role_index] - static_k[:, :, role_index]
        energy = torch.sqrt(
            (delta_q.square().mean(dim=(0, 2)) + delta_k.square().mean(dim=(0, 2)))
            .clamp_min(0.0)
        )
        activity_rows.append(energy)
    contact = _mask_contact_trace(role_masks.masks).to(activity_rows[0].device)
    trace = torch.stack((*activity_rows, contact), dim=1).detach().cpu().float().contiguous()

    required_metrics = []
    for query_role, key_role in ACTION_REQUIRED_EDGES[action]:
        edge = graph[
            :,
            :,
            GENERIC_ROLES.index(query_role),
            :,
            GENERIC_ROLES.index(key_role),
        ]
        norm = float(torch.linalg.vector_norm(edge.double()).item())
        phases = int((edge.abs().amax(dim=(0, 2)) > 0).sum().item())
        required_metrics.append((query_role, key_role, norm, phases))
    mechanically_qualified = all(
        norm >= MIN_REQUIRED_EDGE_NORM and phases >= MIN_REQUIRED_EDGE_QUERY_PHASES
        for _query, _key, norm, phases in required_metrics
    )
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "action_id": action,
        "anchor_slot": dynamic.anchor_slot,
        "anchor_asset_sha256": dynamic.anchor_asset_sha256,
        "mask_authority_digest": role_masks.digest,
        "block_index": dynamic.block_index,
        "rank": dynamic.rank,
        "sp_size": dynamic.sp_size,
        "roles": list(GENERIC_ROLES),
        "graph_sha256": tensor_sha256(graph),
        "timing_trace_channels": list(TRACE_CHANNELS),
        "timing_trace_sha256": tensor_sha256(trace),
        "required_edge_metrics": [list(item) for item in required_metrics],
        "mechanically_qualified": mechanically_qualified,
    }
    return LocalActionGraphCandidateV15E(
        CANDIDATE_SCHEMA,
        action,
        dynamic.anchor_slot,
        dynamic.anchor_asset_sha256,
        role_masks.digest,
        dynamic.block_index,
        dynamic.rank,
        dynamic.sp_size,
        GENERIC_ROLES,
        graph,
        trace,
        tuple(required_metrics),
        mechanically_qualified,
        object_sha256(payload),
    )


def build_representation_preflight_receipt_v15e(
    candidates: Sequence[LocalActionGraphCandidateV15E],
    *,
    qk_bank: QKCaptureBankV15E,
    role_masks: AnchorRoleMaskAuthorityV15E,
) -> Mapping[str, Any]:
    """Return a JSON-safe one-anchor preflight receipt, never authorization."""

    rows = tuple(candidates)
    if not rows or not all(isinstance(item, LocalActionGraphCandidateV15E) for item in rows):
        raise SelfActionGraphObserverV15EError("preflight requires graph candidates")
    if not isinstance(qk_bank, QKCaptureBankV15E):
        raise SelfActionGraphObserverV15EError("preflight requires the live Q/K bank")
    qk_bank_receipt = qk_bank.receipt()
    first = rows[0]
    if any(
        item.action_id != first.action_id
        or item.anchor_slot != first.anchor_slot
        or item.anchor_asset_sha256 != first.anchor_asset_sha256
        or item.mask_authority_digest != role_masks.digest
        for item in rows
    ):
        raise SelfActionGraphObserverV15EError("candidate preflight authorities differ")
    identities = [(item.block_index, item.rank) for item in rows]
    if identities != sorted(set(identities)):
        raise SelfActionGraphObserverV15EError("candidate block/rank rows repeat or are unsorted")
    mechanical = all(item.mechanically_qualified for item in rows)
    raw_qk_cleared = qk_bank_receipt.get("resident_capture_count") == 0
    blockers = []
    if not mechanical:
        blockers.append("one_or_more_required_role_edges_failed_mechanical_gate")
    if not raw_qk_cleared:
        blockers.append("raw_qk_still_resident_in_anchor_process")
    blockers.extend(
        (
            "four_distinct_appearance_anchors_not_yet_compared",
            "cross_appearance_edge_consensus_not_yet_passed",
            "v15b_editor_action_registry_not_authorized_by_this_observer",
            "no_route_or_decode_executed",
        )
    )
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD,
        "status": (
            "representation_candidate_mechanical_pass_consensus_pending"
            if mechanical and raw_qk_cleared
            else "representation_candidate_preflight_no_go"
        ),
        "action_id": first.action_id,
        "anchor_slot": first.anchor_slot,
        "anchor_asset_sha256": first.anchor_asset_sha256,
        "role_mask_authority_digest": role_masks.digest,
        "candidate_rows": [
            {
                "block_index": item.block_index,
                "rank": item.rank,
                "digest": item.digest,
                "graph_sha256": tensor_sha256(item.graph),
                "timing_trace_sha256": tensor_sha256(item.timing_trace),
                "mechanically_qualified": item.mechanically_qualified,
            }
            for item in rows
        ],
        "qk_capture": dict(qk_bank_receipt),
        "anchor_process_persistent_output_allowlist": [
            "role_relation_graph",
            "role_contact_timing_trace",
            "digests",
            "gate_metrics",
            "receipt",
        ],
        "anchor_process_persistent_output_forbidden": [
            "query",
            "key",
            "value",
            "hidden_state",
            "attention_output",
            "rgb",
            "latent",
            "gaussian",
            "role_mask_coordinates",
        ],
        "raw_qk_cleared_before_target_process": raw_qk_cleared,
        "blockers": blockers,
        "representation_candidate_qualified": mechanical and raw_qk_cleared,
        "four_anchor_consensus_passed": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "scientific_claim_authorized": False,
    }
    return {**payload, "receipt_sha256": object_sha256(payload)}


__all__ = [
    "ACTION_ALLOWED_EDGES",
    "ACTION_REQUIRED_EDGES",
    "AnchorRoleMaskAuthorityV15E",
    "GENERIC_ROLES",
    "LocalActionGraphCandidateV15E",
    "PostRopeQKCaptureV15E",
    "QKCaptureBankV15E",
    "QKCaptureInvocationV15E",
    "SelfActionGraphObserverV15EError",
    "SelfActionQKAttn1ObserverV15E",
    "SelfActionQKObserverPatchHandleV15E",
    "build_representation_preflight_receipt_v15e",
    "extract_local_action_graph_v15e",
    "install_self_action_qk_observer_v15e",
    "observe_self_action_qk_v15e",
    "tensor_sha256",
]
