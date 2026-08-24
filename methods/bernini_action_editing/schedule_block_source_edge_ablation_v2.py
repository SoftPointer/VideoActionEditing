#!/usr/bin/env python3
"""Frozen target-query -> source-K/V schedule x block intervention.

This is the Stage-A intervention described by ``20260813_man/preservation.md``.
It is deliberately different from the older target-row *prompt* swap:

* the edited edge is Bernini ``attn1`` self attention, not ``attn2`` text
  cross attention;
* the intervention runs inside the native exact40 sampler;
* only target queries in one registered schedule x block cell lose access to
  the non-target visual-prefix keys and values;
* source-query outputs, target-to-target attention, all other blocks/steps,
  the prompt, scheduler, Gaussian and model parameters remain native.

The official Bernini processor first performs projection, Q/K normalization,
Ulysses gather-sequence/scatter-heads and Q/K RoPE.  At an active cell this
wrapper reuses that exact projection, evaluates native full attention once,
and evaluates target-query/target-KV attention once.  It then keeps the native
source-query rows and replaces only the target-query rows.  Slicing happens
after RoPE, without moving any token to a different positional phase.

The ``source-on`` arm delegates directly to the untouched official processor.
That arm is the numerical-parity control.  No optimizer, score, reward,
ranking, feature evaluator, or output selection exists in this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterator, Mapping, NoReturn, Optional, Sequence

try:
    from . import native_i_axis_guidance as native_i
    from . import schedule_block_causal_policy_v1 as policy
    from . import source_kv_replay as replay
except ImportError:
    import native_i_axis_guidance as native_i
    import schedule_block_causal_policy_v1 as policy
    import source_kv_replay as replay


SCHEMA_VERSION = "bernini-schedule-block-source-edge-ablation-v2"
METHOD = "frozen-target-query-source-kv-edge-causal-localization-v2"
EDGE_MODES = ("source-on", "source-off")
NATIVE_BRANCH_ORDER = ("none_uncond", "V_uncond", "VI_uncond", "VI_cond")
SOURCE_BEARING_BRANCHES = NATIVE_BRANCH_ORDER[1:]
TEXT_BRANCHES = (
    "forward",
    "noop",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
)
OWNER_ROLES = ("correct_owner", "wrong_owner")
NUM_STEPS = 40
NUM_BLOCKS = 30


class SourceEdgeAblationError(RuntimeError):
    """Raised before an ambiguous or partial edge intervention can execute."""


def fail(message: str) -> NoReturn:
    raise SourceEdgeAblationError(message)


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
        raise SourceEdgeAblationError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_int(value: Any, *, lower: int, upper: int, label: str) -> int:
    if type(value) is not int or not lower <= value < upper:
        fail(f"{label} must be an exact integer in [{lower},{upper})")
    return value


def _band(name: str) -> tuple[int, ...]:
    bands = dict(policy.REGISTERED_BLOCK_BANDS)
    if name not in bands:
        fail(f"block band must be one of {tuple(bands)}")
    return tuple(bands[name])


def intervention_contract() -> Mapping[str, Any]:
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "renderer": "Bernini-R-1.3B-transformer_1",
        "sampler": "native-exact40-UniPC-flow-shift-5",
        "frame_count": 81,
        "schedule_indices": list(policy.REGISTERED_SCHEDULE_INDICES),
        "block_bands": {
            name: list(indices) for name, indices in policy.REGISTERED_BLOCK_BANDS
        },
        "text_branches": list(TEXT_BRANCHES),
        "owner_roles": list(OWNER_ROLES),
        "edge_modes": list(EDGE_MODES),
        "attention": "attn1-self-attention",
        "projection_boundary": (
            "official-post-projection-qk-norm-Ulysses-gather-seq-scatter-heads-"
            "and-qk-RoPE"
        ),
        "source_definition": "all-non-target-visual-prefix-tokens",
        "target_definition": "native-noisy-target-visual-suffix",
        "source_off_operation": (
            "target_queries-attend-target-KV-only;source-query-native-output-retained"
        ),
        "source_on_operation": "delegate-exact-official-attn1-processor-object",
        "none_uncond_has_no_source_prefix_and_is_always_native": True,
        "token_order_or_rope_phase_changed": False,
        "optimizer": False,
        "parameter_update": False,
        "training": False,
        "reward": False,
        "feature_scalar": False,
        "ranking": False,
        "selection": False,
        "decoded_exact81_required": True,
        "manual_conjunctive_review_required": True,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


@dataclass(frozen=True)
class EdgeInvocation:
    """One immutable transformer-forward binding for every patched block."""

    edge_mode: str
    schedule_index: int
    registered_schedule_index: int
    band_name: str
    branch_name: str
    total_tokens: int
    target_tokens: int
    ulysses_rank: int
    ulysses_size: int

    def __post_init__(self) -> None:
        if self.edge_mode not in EDGE_MODES:
            fail("edge mode differs")
        _exact_int(self.schedule_index, lower=0, upper=NUM_STEPS, label="schedule index")
        if self.registered_schedule_index not in policy.REGISTERED_SCHEDULE_INDICES:
            fail("registered schedule index differs")
        _band(self.band_name)
        if self.branch_name not in NATIVE_BRANCH_ORDER:
            fail("native branch name differs")
        total = _exact_int(self.total_tokens, lower=1, upper=2**31, label="total tokens")
        target = _exact_int(self.target_tokens, lower=1, upper=2**31, label="target tokens")
        if self.branch_name == "none_uncond":
            if total != target:
                fail("none_uncond must contain target tokens only")
        elif not total > target:
            fail("source-bearing branch requires a nonempty visual prefix")
        _exact_int(self.ulysses_rank, lower=0, upper=2**16, label="Ulysses rank")
        size = _exact_int(self.ulysses_size, lower=1, upper=2**16, label="Ulysses size")
        if self.ulysses_rank >= size:
            fail("Ulysses rank is outside its group")

    @property
    def source_tokens(self) -> int:
        return self.total_tokens - self.target_tokens

    @property
    def active_schedule(self) -> bool:
        return self.schedule_index == self.registered_schedule_index

    def block_active(self, block_index: int) -> bool:
        return (
            self.edge_mode == "source-off"
            and self.active_schedule
            and self.branch_name in SOURCE_BEARING_BRANCHES
            and block_index in _band(self.band_name)
        )

    def receipt(self) -> Mapping[str, Any]:
        unsigned = {
            "edge_mode": self.edge_mode,
            "schedule_index": self.schedule_index,
            "registered_schedule_index": self.registered_schedule_index,
            "band_name": self.band_name,
            "selected_blocks": list(_band(self.band_name)),
            "branch_name": self.branch_name,
            "total_tokens": self.total_tokens,
            "source_tokens": self.source_tokens,
            "target_tokens": self.target_tokens,
            "ulysses_rank": self.ulysses_rank,
            "ulysses_size": self.ulysses_size,
        }
        return {**unsigned, "digest": object_sha256(unsigned)}


_ACTIVE: ContextVar[Optional[EdgeInvocation]] = ContextVar(
    "bernini_schedule_block_source_edge_ablation_v2", default=None
)


@contextmanager
def activate_edge(invocation: EdgeInvocation) -> Iterator[EdgeInvocation]:
    if not isinstance(invocation, EdgeInvocation):
        fail("edge invocation has the wrong type")
    if _ACTIVE.get() is not None:
        fail("nested source-edge invocations are forbidden")
    token = _ACTIVE.set(invocation)
    try:
        yield invocation
    finally:
        _ACTIVE.reset(token)


def current_edge_invocation() -> EdgeInvocation:
    value = _ACTIVE.get()
    if value is None:
        fail("patched attn1 called outside an edge invocation")
    return value


def _as_exact_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        fail(f"{label} must be a sequence")
    result = tuple(int(item) for item in value)
    if any(type(item) is bool or float(item) != int(item) for item in value):
        fail(f"{label} must contain exact integers")
    return result


class TargetQuerySourceEdgeProcessor:
    """Official-attention wrapper with one exact target->prefix edge deletion."""

    def __init__(
        self,
        base_processor: Any,
        *,
        block_index: int,
        varlen_attention_fn: Optional[Callable[..., Any]] = None,
        get_parallel_state_fn: Optional[Callable[[], Any]] = None,
        gather_heads_scatter_seq_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not callable(base_processor) or not callable(
            getattr(base_processor, "_project_qkv", None)
        ):
            fail("official attn1 processor lacks callable/_project_qkv")
        self.base_processor = base_processor
        self.block_index = _exact_int(
            block_index, lower=0, upper=NUM_BLOCKS, label="block index"
        )
        self._varlen_attention_fn = varlen_attention_fn
        self._get_parallel_state_fn = get_parallel_state_fn
        self._gather_heads_scatter_seq_fn = gather_heads_scatter_seq_fn
        self.official_delegate_calls = 0
        self.active_edge_deletion_calls = 0
        self.active_source_on_calls = 0
        self.branch_calls = {name: 0 for name in NATIVE_BRANCH_ORDER}
        self.schedule_calls = {str(index): 0 for index in range(NUM_STEPS)}
        self.last_active_geometry: Optional[Mapping[str, Any]] = None

    def _ops(self) -> tuple[Callable[..., Any], Callable[[], Any], Callable[..., Any]]:
        varlen_fn = self._varlen_attention_fn
        state_fn = self._get_parallel_state_fn
        inverse_fn = self._gather_heads_scatter_seq_fn
        if varlen_fn is None:
            from bernini.attention import varlen_attention as varlen_fn
        if state_fn is None or inverse_fn is None:
            from bernini.parallel import gather_heads_scatter_seq, get_parallel_state

            state_fn = get_parallel_state if state_fn is None else state_fn
            inverse_fn = (
                gather_heads_scatter_seq if inverse_fn is None else inverse_fn
            )
        return varlen_fn, state_fn, inverse_fn

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
        invocation = current_edge_invocation()
        self.branch_calls[invocation.branch_name] += 1
        self.schedule_calls[str(invocation.schedule_index)] += 1
        if encoder_hidden_states is not None or attention_mask is not None:
            fail("source-edge intervention is restricted to unmasked attn1")
        if getattr(hidden_states, "ndim", None) != 3 or int(hidden_states.shape[0]) != 1:
            fail("attn1 hidden states must be [1,L,D]")

        active_block = self.block_index in _band(invocation.band_name)
        active_coordinate = (
            invocation.active_schedule
            and invocation.branch_name in SOURCE_BEARING_BRANCHES
            and active_block
        )
        if invocation.edge_mode == "source-on" or not invocation.block_active(
            self.block_index
        ):
            # The parity control and every off-cell call execute the exact
            # official callable with the original objects and argument values.
            result = self.base_processor(
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
            self.official_delegate_calls += 1
            if invocation.edge_mode == "source-on" and active_coordinate:
                self.active_source_on_calls += 1
            return result

        try:
            import torch
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise SourceEdgeAblationError("active edge deletion requires PyTorch") from error
        if torch.is_grad_enabled():
            fail("Stage-A edge deletion requires inference/no-grad execution")
        replay.require_rotary_embedding(rotary_emb)
        query, key, value = self.base_processor._project_qkv(
            attn,
            hidden_states,
            None,
            rotary_emb,
            origin_hidden_states_seq_len,
            False,
        )
        projected_shape = replay.projected_qkv_shape(query, key, value)
        replay.validate_projected_rotary_embedding(
            rotary_emb,
            projected_shape=projected_shape,
            projected_device=query.device,
        )
        if projected_shape[1] != invocation.total_tokens:
            fail("projected global sequence differs from invocation geometry")
        if _as_exact_tuple(
            batch_image_vae_seqlen, label="batch_image_vae_seqlen"
        ) != (invocation.total_tokens,):
            fail("native batch length differs from invocation")
        if _as_exact_tuple(cu_seqlens_q_cache, label="cu_seqlens_q_cache") != (
            0,
            invocation.total_tokens,
        ):
            fail("native cumulative sequence differs from invocation")
        maximum = int(
            max_seqlen_q_cache.item()
            if hasattr(max_seqlen_q_cache, "item")
            else max_seqlen_q_cache
        )
        if maximum != invocation.total_tokens:
            fail("native maximum sequence differs from invocation")
        if not isinstance(cu_seqlens_q_cache, torch.Tensor):
            fail("active edge deletion requires tensor cumulative lengths")

        varlen_fn, state_fn, inverse_fn = self._ops()
        ulysses_enabled, rank, size = replay.parallel_identity(state_fn())
        if (rank, size) != (invocation.ulysses_rank, invocation.ulysses_size):
            fail("Ulysses runtime differs from edge invocation")
        q = query.squeeze(0).contiguous()
        k = key.squeeze(0).contiguous()
        v = value.squeeze(0).contiguous()
        native_output = varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q_cache,
            cu_seqlens_k=cu_seqlens_q_cache,
            max_seqlen_q=max_seqlen_q_cache,
            max_seqlen_k=max_seqlen_q_cache,
            causal=False,
        )
        source_tokens = invocation.source_tokens
        restricted_cu = torch.tensor(
            [0, invocation.target_tokens],
            dtype=cu_seqlens_q_cache.dtype,
            device=cu_seqlens_q_cache.device,
        )
        target_output = varlen_fn(
            q[source_tokens:].contiguous(),
            k[source_tokens:].contiguous(),
            v[source_tokens:].contiguous(),
            cu_seqlens_q=restricted_cu,
            cu_seqlens_k=restricted_cu,
            max_seqlen_q=invocation.target_tokens,
            max_seqlen_k=invocation.target_tokens,
            causal=False,
        )
        if (
            tuple(native_output.shape) != tuple(q.shape)
            or tuple(target_output.shape) != tuple(q[source_tokens:].shape)
            or not bool(torch.isfinite(native_output).all().item())
            or not bool(torch.isfinite(target_output).all().item())
        ):
            fail("native/restricted attention output contract differs")
        output = torch.cat(
            (native_output[:source_tokens], target_output), dim=0
        ).unsqueeze(0)
        if ulysses_enabled:
            output = inverse_fn(output, head_dim=2, seq_dim=1)
        if getattr(output, "ndim", None) != 4:
            fail("edge attention output must be [1,S,H,D]")
        output = output.flatten(2, 3).contiguous().type_as(query)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        if tuple(output.shape) != tuple(hidden_states.shape):
            fail("edge attention final geometry differs")
        self.active_edge_deletion_calls += 1
        self.last_active_geometry = {
            "schedule_index": invocation.schedule_index,
            "band_name": invocation.band_name,
            "branch_name": invocation.branch_name,
            "total_tokens": invocation.total_tokens,
            "source_tokens": source_tokens,
            "target_tokens": invocation.target_tokens,
            "source_query_rows_from_native_full_attention": True,
            "target_query_rows_from_target_KV_only_attention": True,
            "post_rope_token_order_unchanged": True,
        }
        return output

    def statistics(self) -> Mapping[str, Any]:
        return {
            "block_index": self.block_index,
            "official_delegate_calls": self.official_delegate_calls,
            "active_edge_deletion_calls": self.active_edge_deletion_calls,
            "active_source_on_calls": self.active_source_on_calls,
            "branch_calls": dict(self.branch_calls),
            "schedule_calls": dict(self.schedule_calls),
            "last_active_geometry": self.last_active_geometry,
        }


@dataclass
class SourceEdgePatchHandle:
    transformer: Any
    processors: tuple[TargetQuerySourceEdgeProcessor, ...]
    originals: tuple[Any, ...]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        blocks = self.transformer.blocks
        for index, processor in enumerate(self.processors):
            if getattr(blocks[index].attn1, "processor", None) is not processor:
                fail(f"attn1 block {index} changed behind edge handle")
        for index, original in enumerate(self.originals):
            attn = blocks[index].attn1
            setter = getattr(attn, "set_processor", None)
            setter(original) if callable(setter) else setattr(attn, "processor", original)
        self.restored = True

    def receipt(self) -> Mapping[str, Any]:
        unsigned = {
            "contract": intervention_contract(),
            "installed_block_indices": list(range(NUM_BLOCKS)),
            "installed_projection": "blocks.{0..29}.attn1.processor",
            "processors": [dict(value.statistics()) for value in self.processors],
            "restored": self.restored,
        }
        return {**unsigned, "digest": object_sha256(unsigned)}


def install_source_edge_processors(model: Any) -> SourceEdgePatchHandle:
    transformer = replay.resolve_wan_transformer(model)
    if len(transformer.blocks) != NUM_BLOCKS:
        fail("Bernini transformer block count differs")
    originals: list[Any] = []
    processors: list[TargetQuerySourceEdgeProcessor] = []
    installed: list[int] = []
    try:
        for index, block in enumerate(transformer.blocks):
            attn = block.attn1
            original = getattr(attn, "processor", None)
            if original is None or isinstance(original, TargetQuerySourceEdgeProcessor):
                fail(f"attn1 block {index} lacks a fresh official processor")
            processor = TargetQuerySourceEdgeProcessor(
                original, block_index=index
            )
            setter = getattr(attn, "set_processor", None)
            setter(processor) if callable(setter) else setattr(attn, "processor", processor)
            originals.append(original)
            processors.append(processor)
            installed.append(index)
    except Exception:
        for index, original in zip(reversed(installed), reversed(originals)):
            attn = transformer.blocks[index].attn1
            setter = getattr(attn, "set_processor", None)
            setter(original) if callable(setter) else setattr(attn, "processor", original)
        raise
    return SourceEdgePatchHandle(
        transformer=transformer,
        processors=tuple(processors),
        originals=tuple(originals),
    )


class NativeSourceEdgeHook(native_i.NativeIAxisGuidanceHook):
    """One native exact40 sample with a single registered edge cell."""

    def __init__(
        self,
        diffusion: Any,
        *,
        edge_mode: str,
        schedule_index: int,
        band_name: str,
        expected_steps: int,
        expected_bernini_commit: str,
        observed_wan_diffusion_sha256: str,
    ) -> None:
        if edge_mode not in EDGE_MODES:
            fail("edge mode differs")
        if schedule_index not in policy.REGISTERED_SCHEDULE_INDICES:
            fail("schedule index is outside the registered Stage-A grid")
        _band(band_name)
        try:
            super().__init__(
                diffusion,
                arm="N-C",
                expected_steps=expected_steps,
                expected_bernini_commit=expected_bernini_commit,
                observed_wan_diffusion_sha256=observed_wan_diffusion_sha256,
            )
        except Exception as error:
            raise SourceEdgeAblationError(str(error)) from error
        self.edge_mode = edge_mode
        self.registered_schedule_index = schedule_index
        self.band_name = band_name
        self.edge_handle: Optional[SourceEdgePatchHandle] = None
        self.edge_receipt: Mapping[str, Any] = {}

    def install(self) -> None:
        try:
            super().install()
            self.edge_handle = install_source_edge_processors(self.transformer)
        except Exception:
            try:
                if self.edge_handle is not None:
                    self.edge_handle.restore()
            finally:
                super().restore()
            raise

    def restore(self) -> None:
        edge_error: Optional[Exception] = None
        if self.edge_handle is not None:
            try:
                self.edge_handle.restore()
            except Exception as error:  # pragma: no cover - catastrophic conflict
                edge_error = error
        try:
            super().restore()
        finally:
            if edge_error is not None:
                raise SourceEdgeAblationError(
                    "failed to restore source-edge processors"
                ) from edge_error

    def _wrapped_shared(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            fail("shared_step called outside the active native sample")
        branch_ordinal = len(state.shared_calls)
        if branch_ordinal >= len(NATIVE_BRANCH_ORDER):
            fail("native shared branch closure differs")
        lengths = self._expected_shared_lengths(state)
        target_tokens = int(state.patch_outputs[-1][0].shape[1])
        _, rank, size = replay.parallel_identity(None)
        # ``parallel_identity(None)`` is only valid for unit fakes.  At runtime
        # query the exact installed Bernini parallel state before binding.
        try:
            from bernini.parallel import get_parallel_state

            _, rank, size = replay.parallel_identity(get_parallel_state())
        except ImportError:
            pass
        invocation = EdgeInvocation(
            edge_mode=self.edge_mode,
            schedule_index=state.completed_steps,
            registered_schedule_index=self.registered_schedule_index,
            band_name=self.band_name,
            branch_name=NATIVE_BRANCH_ORDER[branch_ordinal],
            total_tokens=lengths[branch_ordinal],
            target_tokens=target_tokens,
            ulysses_rank=rank,
            ulysses_size=size,
        )
        with activate_edge(invocation):
            return super()._wrapped_shared(*args, **kwargs)

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        result = super()._wrapped_sample(*args, **kwargs)
        if self.edge_handle is None:
            fail("edge processor handle disappeared during sample")
        stats = [dict(value.statistics()) for value in self.edge_handle.processors]
        selected = set(_band(self.band_name))
        expected_active_per_selected_block = len(SOURCE_BEARING_BRANCHES)
        expected_branch_calls = {
            name: self.expected_steps for name in NATIVE_BRANCH_ORDER
        }
        expected_schedule_calls = {
            str(index): len(NATIVE_BRANCH_ORDER) for index in range(self.expected_steps)
        }
        for row in stats:
            active = int(row["active_edge_deletion_calls"])
            source_on = int(row["active_source_on_calls"])
            selected_block = row["block_index"] in selected
            expected_active = (
                expected_active_per_selected_block
                if selected_block and self.edge_mode == "source-off"
                else 0
            )
            expected_source_on = (
                expected_active_per_selected_block
                if selected_block and self.edge_mode == "source-on"
                else 0
            )
            expected_delegate = (
                self.expected_steps * len(NATIVE_BRANCH_ORDER) - expected_active
            )
            if (
                row["branch_calls"] != expected_branch_calls
                or row["schedule_calls"] != expected_schedule_calls
                or int(row["official_delegate_calls"]) != expected_delegate
                or active != expected_active
                or source_on != expected_source_on
            ):
                fail("attn1 processor call/delegation closure differs")
            if row["block_index"] in selected:
                if self.edge_mode == "source-off" and not isinstance(
                    row["last_active_geometry"], Mapping
                ):
                    fail("selected source-off block lacks active geometry")
            elif row["last_active_geometry"] is not None:
                fail("unselected block reported active edge geometry")
        unsigned_edge = {
            "contract": intervention_contract(),
            "edge_mode": self.edge_mode,
            "registered_schedule_index": self.registered_schedule_index,
            "band_name": self.band_name,
            "selected_blocks": list(_band(self.band_name)),
            "source_bearing_branches": list(SOURCE_BEARING_BRANCHES),
            "expected_active_calls_per_selected_block": expected_active_per_selected_block,
            "per_block": stats,
            "native_trace_digest": self.trace.get("trace_digest"),
        }
        self.edge_receipt = {
            **unsigned_edge,
            "digest": object_sha256(unsigned_edge),
        }
        self.trace = {
            **dict(self.trace),
            "source_edge": self.edge_receipt,
            "source_edge_trace_digest": object_sha256(
                {"native": self.trace.get("trace_digest"), "edge": self.edge_receipt}
            ),
        }
        return result


def decoded_grid_contract() -> Mapping[str, Any]:
    """Return the minimum non-deduplicated evidence plan.

    Native prompt baselines and the compatible wrong-owner baseline are decoded
    once per family/seed and linked into every cell.  Every source-off prompt is
    separately decoded for all 16 schedule x block cells.  One hooked source-on
    forward at the preregistered C0 cell is compared bit-exactly with the
    unhooked native forward baseline; because ``source-on`` delegates at every
    call and has no cell-dependent arithmetic, duplicating that same output
    fifteen more times would add no intervention evidence.  This is 104
    outputs/family/seed:

      6 native prompt baselines + 1 wrong-owner baseline
      + 1 hooked source-on parity output + 16 * 6 source-off prompts.
    """

    cells = len(policy.REGISTERED_SCHEDULE_INDICES) * len(
        policy.REGISTERED_BLOCK_BANDS
    )
    per_family = len(TEXT_BRANCHES) + 1 + 1 + cells * len(TEXT_BRANCHES)
    unsigned = {
        "families": ["dog", "human"],
        "seed_count_per_family": 1,
        "schedule_block_cell_count": cells,
        "native_prompt_baselines_per_family": len(TEXT_BRANCHES),
        "wrong_owner_forward_baselines_per_family": 1,
        "source_off_prompt_outputs_per_cell": len(TEXT_BRANCHES),
        "hooked_source_on_forward_parity_outputs_per_family": 1,
        "outputs_per_family": per_family,
        "total_decoded_outputs": 2 * per_family,
        "comparison_per_cell": [
            "source-on-forward_vs_source-off-forward",
            "source-off-forward_vs_source-off-noop",
            "source-off-forward_vs_source-off-reverse",
            "source-off-forward_vs_source-off-incomplete",
            "source-off-forward_vs_source-off-camera_only",
            "source-off-forward_vs_source-off-appearance_only",
            "correct-source-on-forward_vs-compatible-wrong-owner-forward",
        ],
        "same_seed_scheduler_gaussian_decode_within_family": True,
        "scalar_score_or_reward": False,
        "automatic_ranking_or_selection": False,
    }
    return {**unsigned, "digest": object_sha256(unsigned)}


__all__ = [
    "EDGE_MODES",
    "EdgeInvocation",
    "METHOD",
    "NATIVE_BRANCH_ORDER",
    "NativeSourceEdgeHook",
    "OWNER_ROLES",
    "SCHEMA_VERSION",
    "SOURCE_BEARING_BRANCHES",
    "SourceEdgeAblationError",
    "SourceEdgePatchHandle",
    "TEXT_BRANCHES",
    "TargetQuerySourceEdgeProcessor",
    "activate_edge",
    "current_edge_invocation",
    "decoded_grid_contract",
    "install_source_edge_processors",
    "intervention_contract",
    "object_sha256",
]
