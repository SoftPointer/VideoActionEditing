from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - default lightweight environment
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import counterfactual_proposal_motion_branch as branch  # noqa: E402


class PassProjection(nn.Module):
    in_features = branch.HIDDEN_SIZE
    out_features = branch.HIDDEN_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class BiasProjection(PassProjection):
    def __init__(self, bias: float = 3.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.bias.to(dtype=value.dtype)


class Projector:
    def __init__(self) -> None:
        self.calls = 0
        self.last = None

    def _project_qkv(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        origin_hidden_states_seq_len,
        is_cross_attn,
    ):
        self.calls += 1
        self.last = {
            "hidden": tuple(hidden_states.shape),
            "encoder": tuple(encoder_hidden_states.shape),
            "rotary": rotary_emb,
            "origin": origin_hidden_states_seq_len,
            "cross": is_cross_attn,
        }
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        return query, key, value


class DonorAttention(nn.Module):
    def __init__(self, *, meta: bool = False, output_bias: bool = False) -> None:
        super().__init__()
        self.heads = branch.ATTENTION_HEADS
        self.inner_dim = branch.HIDDEN_SIZE
        self.inner_kv_dim = branch.HIDDEN_SIZE
        self.out_dim = branch.HIDDEN_SIZE
        self.cross_attention_dim = branch.HIDDEN_SIZE
        if meta:
            make = lambda: nn.Linear(  # noqa: E731
                branch.HIDDEN_SIZE,
                branch.HIDDEN_SIZE,
                device="meta",
            )
            self.to_q = make()
            self.to_k = make()
            self.to_v = make()
            output = make()
        else:
            self.to_q = PassProjection()
            self.to_k = PassProjection()
            self.to_v = PassProjection()
            output = BiasProjection() if output_bias else PassProjection()
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.to_out = nn.ModuleList([output, nn.Identity()])
        self.processor = Projector()


class CountingBaseProcessor:
    def __init__(self, value: float = 2.0) -> None:
        self.value = float(value)
        self.calls = 0
        self.last_kwargs = None

    def __call__(self, attn, hidden_states, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return hidden_states * self.value


class RaisingBaseProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, attn, hidden_states, **kwargs):
        del attn, hidden_states, kwargs
        self.calls += 1
        raise RuntimeError("base sentinel")


class StubMotion(branch.MotionCrossAttention):
    """Wrapper-only test double; production creation still clones attn1."""

    def __init__(self, block_index: int = 0, value: float = 4.0) -> None:
        nn.Module.__init__(self)
        self.block_index = block_index
        self.value = float(value)
        self.motion_calls = 0

    def forward(self, hidden_states, carrier, **kwargs):
        del carrier, kwargs
        self.motion_calls += 1
        return torch.full_like(hidden_states, self.value)

    def statistics(self):
        return {
            "block_index": self.block_index,
            "motion_calls": self.motion_calls,
            "explicit_custom_collective_calls": 0,
            "measured_custom_collective_calls": None,
        }


class Attn2Box(nn.Module):
    def __init__(self, processor) -> None:
        super().__init__()
        self.processor = processor

    def set_processor(self, processor) -> None:
        self.processor = processor


class FakeBlock(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.attn1 = DonorAttention()
        self.attn2 = Attn2Box(CountingBaseProcessor(value=index + 1))


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [FakeBlock(index) for index in range(branch.EXPECTED_BLOCK_COUNT)]
        )


def payload(*, active: bool, dtype=torch.float32):
    phase = torch.zeros((1, branch.LATENT_PHASES, 1, 1), dtype=dtype)
    activity = torch.zeros((1, branch.LATENT_PHASES), dtype=torch.bool)
    if active:
        phase[:, 1] = 1
        activity[:, 1] = True
    carrier = phase.expand(
        1,
        branch.LATENT_PHASES,
        branch.CARRIER_TOKENS_PER_PHASE,
        branch.HIDDEN_SIZE,
    ).reshape(1, branch.CARRIER_TOKENS, branch.HIDDEN_SIZE)
    return carrier, activity


def invocation(
    *,
    gate: float = 0.1,
    active: bool = True,
    trajectory: str = branch.FINAL_RENDER,
    polarity: str = branch.POSITIVE,
    same_prompt: bool = True,
    dtype=torch.float32,
    processors=(),
):
    current = torch.ones((1, 2, 3), dtype=dtype)
    expected = current if same_prompt else current.clone()
    carrier, activity = payload(active=active, dtype=dtype)
    routes_motion = (
        trajectory == branch.FINAL_RENDER
        and polarity == branch.POSITIVE
        and current is expected
    )
    binding = None
    if routes_motion:
        binding = (
            branch._conditioned_encoder_binding_for_processors(processors)
            if processors
            else branch.CPMRConditionedEncoderBinding((0,), object())
        )
    return branch.CPMRMotionInvocation(
        trajectory=trajectory,
        polarity=polarity,
        prompt_object=current,
        positive_noop_prompt_object=expected,
        conditioned_encoder_binding=binding,
        gate=gate,
        carrier=carrier,
        activity=activity,
    )


def call_wrapper(
    processor,
    hidden,
    *,
    encoder=True,
    encoder_object=None,
    origin=branch.GLOBAL_VISUAL_TOKENS,
    batch_lengths=None,
):
    if batch_lengths is None:
        batch_lengths = [branch.GLOBAL_VISUAL_TOKENS]
    if encoder and encoder_object is None:
        active = branch.current_cpmr_motion_invocation()
        binding = (
            active.conditioned_encoder_binding if active is not None else None
        )
        encoder_object = (
            binding._bound_tensor
            if binding is not None and binding._bound_tensor is not None
            else torch.full((1, 2, 3), 7.0, dtype=hidden.dtype)
        )
    return processor(
        object(),
        hidden,
        encoder_hidden_states=encoder_object if encoder else None,
        batch_image_vae_seqlen=batch_lengths,
        text_features_length=[2],
        origin_hidden_states_seq_len=origin,
        split_hidden_states_seq_len=branch.GLOBAL_VISUAL_TOKENS // 4,
    )


def official_mocks(rank: int, events: list):
    local = branch.GLOBAL_VISUAL_TOKENS // 4

    def gen(q_len, q_lengths, k_lengths, device="cpu"):
        events.append(("gen", q_len, tuple(q_lengths), tuple(k_lengths)))
        return (
            torch.tensor([0, branch.CARRIER_TOKENS], dtype=torch.int32, device=device),
            torch.tensor([0, local], dtype=torch.int32, device=device),
            torch.tensor(branch.CARRIER_TOKENS, device=device),
            torch.tensor(local, device=device),
            local,
        )

    def pad(value, dim):
        events.append(("pad", tuple(value.shape), dim))
        return value

    def slice_input(value, dim):
        events.append(("slice", tuple(value.shape), dim))
        return value[:, rank * local : (rank + 1) * local]

    def varlen(q, k, v, **kwargs):
        del k, v
        events.append(
            (
                "varlen",
                tuple(q.shape),
                kwargs["cu_seqlens_q"].tolist(),
                kwargs["cu_seqlens_k"].tolist(),
                int(kwargs["max_seqlen_q"]),
                int(kwargs["max_seqlen_k"]),
                kwargs["causal"],
            )
        )
        return q

    return gen, pad, slice_input, varlen


def runtime_motion(*, rank: int, output_bias: bool = False):
    events = []
    gen, pad, slice_input, varlen = official_mocks(rank, events)
    motion = branch.MotionCrossAttention(
        DonorAttention(output_bias=output_bias),
        block_index=0,
        varlen_attention_fn=varlen,
        gen_cu_seqlens_fn=gen,
        padding_tensor_fn=pad,
        slice_input_tensor_fn=slice_input,
    )
    return motion, events


class CPMRMotionBranchTests(unittest.TestCase):
    def test_01_frozen_constants_and_plain_explicit_processor_signature(self):
        self.assertEqual(branch.GLOBAL_VISUAL_TOKENS, 39_060)
        self.assertEqual(branch.SOURCE_VISUAL_TOKENS, 19_530)
        self.assertEqual(branch.TARGET_VISUAL_TOKENS, 19_530)
        self.assertEqual(branch.CARRIER_TOKENS, 1_344)
        self.assertEqual(branch.MOTION_BLOCK_INDICES, tuple(range(16)))
        self.assertFalse(issubclass(branch.CPMRTextAttnProcessor, nn.Module))
        parameters = list(inspect.signature(branch.CPMRTextAttnProcessor.__call__).parameters)
        self.assertEqual(
            parameters,
            [
                "self",
                "attn",
                "hidden_states",
                "encoder_hidden_states",
                "attention_mask",
                "rotary_emb",
                "batch_image_vae_seqlen",
                "text_features_length",
                "origin_hidden_states_seq_len",
                "split_hidden_states_seq_len",
                "cu_seqlens_q_cache",
                "max_seqlen_q_cache",
                "cu_seqlens_k_cross_cache",
                "cu_seqlens_q_cross_cache",
                "max_seqlen_k_cross_cache",
                "max_seqlen_q_cross_cache",
            ],
        )

    def test_02_invocation_gate_and_prompt_identity_are_fail_closed(self):
        self.assertTrue(invocation().routes_motion)
        self.assertFalse(invocation(same_prompt=False).routes_motion)
        self.assertFalse(invocation(trajectory=branch.ACTION_PROPOSAL).routes_motion)
        self.assertFalse(invocation(polarity=branch.UNCONDITIONAL).routes_motion)
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError,
            "two raw conditioned encoder tensors",
        ):
            branch.CPMRMotionInvocation(
                trajectory=branch.FINAL_RENDER,
                polarity=branch.POSITIVE,
                prompt_object=None,
                positive_noop_prompt_object=None,
                gate=0.1,
            )
        raw = torch.ones((1, 2, 3))
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError,
            "requires a conditioned encoder binding",
        ):
            branch.CPMRMotionInvocation(
                trajectory=branch.FINAL_RENDER,
                polarity=branch.POSITIVE,
                prompt_object=raw,
                positive_noop_prompt_object=raw,
                gate=0.1,
            )
        for invalid_gate in (-0.1, 0.11, 1.0, float("nan"), True):
            with self.subTest(gate=invalid_gate):
                with self.assertRaises(branch.CPMRMotionBranchContractError):
                    invocation(gate=invalid_gate)

    def test_03_context_is_non_nested_and_exception_safe(self):
        item = invocation(gate=0.0)
        self.assertIsNone(branch.current_cpmr_motion_invocation())
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                self.assertIs(branch.current_cpmr_motion_invocation(), item)
                with self.assertRaises(branch.CPMRMotionBranchContractError):
                    with branch.cpmr_motion_invocation(
                        item, encoder_hidden_states=item.prompt_object
                    ):
                        pass
                raise RuntimeError("sentinel")
        self.assertIsNone(branch.current_cpmr_motion_invocation())
        self.assertTrue(item.conditioned_encoder_binding.receipt()["aborted"])

    def test_04_visual_donor_clone_is_independent_registered_and_frozen(self):
        donor = DonorAttention()
        motion = branch.MotionCrossAttention(donor, block_index=0)
        self.assertIsInstance(motion, nn.Module)
        self.assertEqual((motion.inner_dim, motion.heads, motion.head_dim), (1536, 12, 128))
        for name in ("to_q", "to_k", "to_v", "norm_q", "norm_k", "to_out"):
            self.assertIsNot(getattr(motion, name), getattr(donor, name, None))
        donor_ids = {id(value) for value in donor.parameters()}
        motion_ids = {id(value) for value in motion.parameters()}
        self.assertTrue(donor_ids.isdisjoint(motion_ids))
        self.assertTrue(all(not value.requires_grad for value in motion.parameters()))
        self.assertTrue(any(name.startswith("to_q.") for name in motion.state_dict()))
        for name in ("to_q", "to_k", "to_v", "norm_q", "norm_k"):
            donor_state = getattr(donor, name).state_dict()
            motion_state = getattr(motion, name).state_dict()
            self.assertEqual(tuple(donor_state), tuple(motion_state))
            for key in donor_state:
                self.assertTrue(torch.equal(donor_state[key], motion_state[key]))

    def test_05_visual_donor_geometry_rejects_wrong_heads_or_width(self):
        donor = DonorAttention()
        donor.heads = 8
        with self.assertRaises(branch.CPMRMotionBranchContractError):
            branch.MotionCrossAttention(donor, block_index=0)
        donor = DonorAttention()
        donor.inner_dim = 768
        with self.assertRaises(branch.CPMRMotionBranchContractError):
            branch.MotionCrossAttention(donor, block_index=0)
        donor = DonorAttention()
        donor.norm_q = None
        with self.assertRaisesRegex(branch.CPMRMotionBranchContractError, "norm_q"):
            branch.MotionCrossAttention(donor, block_index=0)
        donor = DonorAttention(meta=True)
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError, "checkpoint materialization"
        ):
            branch.MotionCrossAttention(donor, block_index=0)

    def test_06_no_context_delegates_and_calls_base_exactly_once(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.ones((1, 3, branch.HIDDEN_SIZE))
        result = call_wrapper(processor, hidden)
        self.assertTrue(torch.equal(result, hidden * 2))
        self.assertEqual((base.calls, processor.base_calls, motion.motion_calls), (1, 1, 0))
        self.assertEqual(processor.no_context_delegations, 1)
        self.assertIn("origin_hidden_states_seq_len", base.last_kwargs)

    def test_07_zero_gate_is_byte_exact_and_never_calls_motion(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.randn((1, 3, branch.HIDDEN_SIZE))
        item = invocation(gate=0.0, processors=(processor,))
        with branch.cpmr_motion_invocation(
            item, encoder_hidden_states=item.prompt_object
        ):
            result = call_wrapper(processor, hidden)
        self.assertTrue(torch.equal(result, hidden * 2))
        self.assertEqual((base.calls, motion.motion_calls), (1, 0))
        self.assertEqual(processor.zero_gate_delegations, 1)

    def test_08_all_inactive_exact_zero_delegates_without_qkvo(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.randn((1, 2, branch.HIDDEN_SIZE))
        item = invocation(active=False, processors=(processor,))
        with branch.cpmr_motion_invocation(
            item, encoder_hidden_states=item.prompt_object
        ):
            result = call_wrapper(processor, hidden)
        self.assertTrue(torch.equal(result, hidden * 2))
        self.assertEqual(motion.motion_calls, 0)
        self.assertEqual(processor.inactive_delegations, 1)

    def test_09_proposals_and_unconditional_apg_branch_are_isolated(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.ones((1, 1, branch.HIDDEN_SIZE))
        items = (
            invocation(trajectory=branch.ACTION_PROPOSAL),
            invocation(trajectory=branch.NOOP_PROPOSAL),
            invocation(polarity=branch.UNCONDITIONAL),
        )
        for item in items:
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                self.assertTrue(torch.equal(call_wrapper(processor, hidden), hidden * 2))
        self.assertEqual(base.calls, 3)
        self.assertEqual(motion.motion_calls, 0)
        self.assertEqual(processor.branch_delegations, 3)

    def test_10_other_positive_prompt_object_cannot_activate_motion(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.ones((1, 1, branch.HIDDEN_SIZE))
        item = invocation(same_prompt=False)
        with branch.cpmr_motion_invocation(
            item, encoder_hidden_states=item.prompt_object
        ):
            result = call_wrapper(processor, hidden)
        self.assertTrue(torch.equal(result, hidden * 2))
        self.assertEqual((base.calls, motion.motion_calls), (1, 0))

    def test_10b_context_authenticates_raw_then_binds_internal_encoder(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.ones((1, 1, branch.HIDDEN_SIZE))
        item = invocation(processors=(processor,))
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError,
            "transformer input encoder object",
        ):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object.clone()
            ):
                pass
        self.assertEqual((base.calls, motion.motion_calls), (0, 0))

        item = invocation(processors=(processor,))
        internal = item.prompt_object.clone()
        self.assertIsNot(internal, item.prompt_object)
        with branch.cpmr_motion_invocation(
            item, encoder_hidden_states=item.prompt_object
        ):
            call_wrapper(processor, hidden, encoder_object=internal)
        self.assertTrue(item.conditioned_encoder_binding.receipt()["completed"])
        self.assertEqual((base.calls, motion.motion_calls), (1, 1))

    def test_10c_binding_rejects_owner_order_and_internal_object_changes(self):
        token = object()
        p0 = branch.CPMRTextAttnProcessor(
            CountingBaseProcessor(), StubMotion(block_index=0),
            block_index=0, patch_token=token,
        )
        p1 = branch.CPMRTextAttnProcessor(
            CountingBaseProcessor(), StubMotion(block_index=1),
            block_index=1, patch_token=token,
        )
        hidden = torch.ones((1, 1, branch.HIDDEN_SIZE))

        item = invocation(gate=0.0, processors=(p0, p1))
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError, "expected block 1, got 0"
        ):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                internal = item.prompt_object.clone()
                call_wrapper(p0, hidden, encoder_object=internal)
                call_wrapper(p0, hidden, encoder_object=internal)
        self.assertTrue(item.conditioned_encoder_binding.receipt()["aborted"])

        item = invocation(gate=0.0, processors=(p0, p1))
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError, "object changed between motion blocks"
        ):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                call_wrapper(p0, hidden, encoder_object=torch.ones((1, 2, 3)))
                call_wrapper(p1, hidden, encoder_object=torch.ones((1, 2, 3)))

        foreign = branch.CPMRTextAttnProcessor(
            CountingBaseProcessor(), StubMotion(block_index=0), block_index=0
        )
        item = invocation(gate=0.0, processors=(p0,))
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError, "different patch handle"
        ):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                call_wrapper(foreign, hidden)

    def test_10d_binding_is_complete_one_use_and_preserves_base_exception(self):
        processor = branch.CPMRTextAttnProcessor(
            CountingBaseProcessor(), StubMotion(), block_index=0
        )
        item = invocation(gate=0.0, processors=(processor,))
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError, "exact block inventory"
        ):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                pass
        receipt = item.conditioned_encoder_binding.receipt()
        self.assertTrue(receipt["consumed"])
        self.assertTrue(receipt["bound_tensor_released"])

        item = invocation(gate=0.0, processors=(processor,))
        with branch.cpmr_motion_invocation(
            item, encoder_hidden_states=item.prompt_object
        ):
            call_wrapper(processor, torch.ones((1, 1, branch.HIDDEN_SIZE)))
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError, "already consumed"
        ):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                pass

        raising = branch.CPMRTextAttnProcessor(
            RaisingBaseProcessor(), StubMotion(), block_index=0
        )
        item = invocation(gate=0.0, processors=(raising,))
        with self.assertRaisesRegex(RuntimeError, "base sentinel"):
            with branch.cpmr_motion_invocation(
                item, encoder_hidden_states=item.prompt_object
            ):
                call_wrapper(raising, torch.ones((1, 1, branch.HIDDEN_SIZE)))
        self.assertTrue(item.conditioned_encoder_binding.receipt()["aborted"])

    def test_11_activity_mismatch_fails_after_one_base_call_before_motion(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.ones((1, 1, branch.HIDDEN_SIZE))
        item = invocation(processors=(processor,))
        forged = branch.CPMRMotionInvocation(
            trajectory=item.trajectory,
            polarity=item.polarity,
            prompt_object=item.prompt_object,
            positive_noop_prompt_object=item.positive_noop_prompt_object,
            conditioned_encoder_binding=(
                branch._conditioned_encoder_binding_for_processors((processor,))
            ),
            gate=item.gate,
            carrier=item.carrier,
            activity=torch.zeros_like(item.activity),
        )
        with self.assertRaisesRegex(branch.CPMRMotionBranchContractError, "activity"):
            with branch.cpmr_motion_invocation(
                forged, encoder_hidden_states=forged.prompt_object
            ):
                call_wrapper(processor, hidden)
        self.assertEqual((base.calls, motion.motion_calls), (1, 0))

    def test_11b_phase_zero_must_remain_byte_exact_positive_zero(self):
        base = CountingBaseProcessor()
        motion = StubMotion()
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.ones((1, 1, branch.HIDDEN_SIZE))
        item = invocation(processors=(processor,))
        signed_zero = item.carrier.clone()
        signed_zero[:, : branch.CARRIER_TOKENS_PER_PHASE].fill_(-0.0)
        forged = branch.CPMRMotionInvocation(
            trajectory=item.trajectory,
            polarity=item.polarity,
            prompt_object=item.prompt_object,
            positive_noop_prompt_object=item.positive_noop_prompt_object,
            conditioned_encoder_binding=(
                branch._conditioned_encoder_binding_for_processors((processor,))
            ),
            gate=item.gate,
            carrier=signed_zero,
            activity=item.activity,
        )
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError,
            "phase 0.*positive zero",
        ):
            with branch.cpmr_motion_invocation(
                forged, encoder_hidden_states=forged.prompt_object
            ):
                call_wrapper(processor, hidden)
        self.assertEqual((base.calls, motion.motion_calls), (1, 0))

        nonzero, nonzero_activity = payload(active=True)
        nonzero[:, 0, 0] = 1.0
        nonzero_activity[:, 0] = True
        forged_nonzero = branch.CPMRMotionInvocation(
            trajectory=item.trajectory,
            polarity=item.polarity,
            prompt_object=item.prompt_object,
            positive_noop_prompt_object=item.positive_noop_prompt_object,
            conditioned_encoder_binding=(
                branch._conditioned_encoder_binding_for_processors((processor,))
            ),
            gate=item.gate,
            carrier=nonzero,
            activity=nonzero_activity,
        )
        with self.assertRaisesRegex(
            branch.CPMRMotionBranchContractError,
            "activity phase 0",
        ):
            with branch.cpmr_motion_invocation(
                forged_nonzero,
                encoder_hidden_states=forged_nonzero.prompt_object,
            ):
                call_wrapper(processor, hidden)
        self.assertEqual((base.calls, motion.motion_calls), (2, 0))

    def test_12_official_cross_sp_is_local_q_full_heads_and_replicated_kv(self):
        motion, events = runtime_motion(rank=2)
        local = branch.GLOBAL_VISUAL_TOKENS // 4
        hidden = torch.ones((1, local, branch.HIDDEN_SIZE), dtype=torch.bfloat16)
        carrier, _ = payload(active=True, dtype=torch.bfloat16)
        output = motion(
            hidden,
            carrier,
            origin_hidden_states_seq_len=branch.GLOBAL_VISUAL_TOKENS,
            batch_image_vae_seqlen=torch.tensor([branch.GLOBAL_VISUAL_TOKENS]),
        )
        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertTrue(bool(torch.count_nonzero(output)))
        self.assertEqual(
            motion._projection_processor.last,
            {
                "hidden": (1, local, 1536),
                "encoder": (1, 1344, 1536),
                "rotary": None,
                "origin": 39060,
                "cross": True,
            },
        )
        self.assertEqual(motion.last_metadata["cu_q"], [0, 9765])
        self.assertEqual(motion.last_metadata["cu_k"], [0, 1344])
        self.assertEqual(motion.last_metadata["heads"], 12)
        self.assertEqual(motion.last_metadata["explicit_custom_collectives"], 0)
        self.assertIsNone(motion.last_metadata["measured_custom_collectives"])
        varlen = [event for event in events if event[0] == "varlen"]
        self.assertEqual(varlen[0][1], (9765, 12, 128))
        self.assertEqual(varlen[0][2:6], ([0, 9765], [0, 1344], 9765, 1344))
        self.assertIs(varlen[0][6], False)

    def test_13_post_to_out_mask_clears_source_bias_to_exact_zero(self):
        motion, events = runtime_motion(rank=0, output_bias=True)
        local = branch.GLOBAL_VISUAL_TOKENS // 4
        hidden = torch.zeros((1, local, branch.HIDDEN_SIZE), dtype=torch.bfloat16)
        carrier, _ = payload(active=True, dtype=torch.bfloat16)
        output = motion(
            hidden,
            carrier,
            origin_hidden_states_seq_len=branch.GLOBAL_VISUAL_TOKENS,
            batch_image_vae_seqlen=[branch.GLOBAL_VISUAL_TOKENS],
        )
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        self.assertEqual([event[0] for event in events[:3]], ["gen", "pad", "slice"])
        self.assertEqual(motion.motion_calls, 1)

    def test_14_active_wrapper_adds_fp32_gated_residual_after_one_base_call(self):
        base = CountingBaseProcessor(value=2.0)
        motion = StubMotion(value=4.0)
        processor = branch.CPMRTextAttnProcessor(base, motion, block_index=0)
        hidden = torch.ones((1, 2, branch.HIDDEN_SIZE), dtype=torch.float16)
        item = invocation(
            gate=0.10, dtype=torch.float16, processors=(processor,)
        )
        with branch.cpmr_motion_invocation(
            item, encoder_hidden_states=item.prompt_object
        ):
            output = call_wrapper(
                processor,
                hidden,
                origin=None,
                batch_lengths=torch.tensor([branch.GLOBAL_VISUAL_TOKENS]),
            )
        self.assertTrue(torch.equal(output, torch.full_like(hidden, 2.4)))
        self.assertEqual((base.calls, processor.base_calls, motion.motion_calls), (1, 1, 1))
        self.assertEqual(processor.motion_calls, 1)

    def test_15_install_registers_modules_only_on_0_to_15_and_preserves_base_identity(self):
        transformer = FakeTransformer()
        attn1_ids = [id(block.attn1) for block in transformer.blocks]
        attn2_ids = [id(block.attn2) for block in transformer.blocks]
        originals = [block.attn2.processor for block in transformer.blocks]
        handle = branch.install_cpmr_motion_branch(transformer)
        self.assertEqual(handle.indices, tuple(range(16)))
        for index, block in enumerate(transformer.blocks):
            self.assertEqual(id(block.attn1), attn1_ids[index])
            self.assertEqual(id(block.attn2), attn2_ids[index])
            if index < 16:
                self.assertIsInstance(
                    getattr(block, branch.MOTION_MODULE_NAME),
                    branch.MotionCrossAttention,
                )
                self.assertIsInstance(block.attn2.processor, branch.CPMRTextAttnProcessor)
                self.assertIs(block.attn2.processor.base_processor, originals[index])
                self.assertTrue(
                    any(
                        key.startswith(
                            f"blocks.{index}.{branch.MOTION_MODULE_NAME}.to_q"
                        )
                        for key in transformer.state_dict()
                    )
                )
            else:
                self.assertFalse(hasattr(block, branch.MOTION_MODULE_NAME))
                self.assertIs(block.attn2.processor, originals[index])
        self.assertEqual(handle.receipt()["explicit_custom_collective_calls"], 0)
        self.assertIsNone(handle.receipt()["measured_custom_collective_calls"])
        self.assertEqual(
            handle.receipt()["binding_expected_block_indices"], list(range(16))
        )
        self.assertFalse(handle.receipt()["gradient_checkpoint_supported"])
        self.assertFalse(handle.receipt()["torch_compile_supported"])

        carrier, activity = payload(active=True)
        raw = torch.ones((1, 2, 3))
        item = branch.CPMRMotionInvocation(
            trajectory=branch.FINAL_RENDER,
            polarity=branch.POSITIVE,
            prompt_object=raw,
            positive_noop_prompt_object=raw,
            conditioned_encoder_binding=handle.new_conditioned_encoder_binding(),
            gate=0.0,
            carrier=carrier,
            activity=activity,
        )
        hidden = torch.ones((1, 1, branch.HIDDEN_SIZE))
        internal = raw.clone()
        with branch.cpmr_motion_invocation(item, encoder_hidden_states=raw):
            for processor in handle.processors:
                call_wrapper(processor, hidden, encoder_object=internal)
        binding_receipt = item.conditioned_encoder_binding.receipt()
        self.assertEqual(binding_receipt["observed_block_indices"], list(range(16)))
        self.assertTrue(binding_receipt["completed"])
        self.assertTrue(
            all(motion.motion_calls == 0 for motion in handle.motion_modules)
        )

    def test_16_restore_is_idempotent_and_restores_exact_state_dict_keys(self):
        transformer = FakeTransformer()
        before_keys = tuple(transformer.state_dict().keys())
        originals = tuple(block.attn2.processor for block in transformer.blocks)
        handle = branch.install_cpmr_motion_branch(transformer)
        during_keys = tuple(transformer.state_dict().keys())
        self.assertNotEqual(during_keys, before_keys)
        handle.restore()
        handle.restore()
        self.assertTrue(handle.restored)
        self.assertEqual(tuple(transformer.state_dict().keys()), before_keys)
        for index, block in enumerate(transformer.blocks):
            self.assertIs(block.attn2.processor, originals[index])
            self.assertFalse(hasattr(block, branch.MOTION_MODULE_NAME))

    def test_17_install_failure_rolls_back_without_partial_modules(self):
        transformer = FakeTransformer()
        originals = tuple(block.attn2.processor for block in transformer.blocks)

        def factory(donor, index):
            if index == 4:
                raise RuntimeError("factory failure")
            return branch.MotionCrossAttention(donor, block_index=index)

        with self.assertRaisesRegex(RuntimeError, "factory failure"):
            branch.install_cpmr_motion_branch(transformer, motion_factory=factory)
        for index, block in enumerate(transformer.blocks):
            self.assertIs(block.attn2.processor, originals[index])
            self.assertFalse(hasattr(block, branch.MOTION_MODULE_NAME))


if __name__ == "__main__":
    unittest.main()
