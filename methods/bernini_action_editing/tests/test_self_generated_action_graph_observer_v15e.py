from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_action_graph_observer_v15e as observer  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


class FakeOfficialProcessor:
    def __init__(self) -> None:
        self.base_calls = 0
        self.project_calls = 0
        self.last_output = None

    def _project_qkv(
        self,
        _attn,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        origin_hidden_states_seq_len,
        is_cross_attn,
    ):
        self.project_calls += 1
        if encoder_hidden_states is not None or rotary_emb is None or is_cross_attn:
            raise AssertionError("fake was not called as visual post-RoPE projection")
        if origin_hidden_states_seq_len != int(hidden_states.shape[1]):
            raise AssertionError("fake origin length differs")
        query = hidden_states.reshape(1, int(hidden_states.shape[1]), 2, 2)
        key = torch.flip(query, dims=(-1,))
        value = query * 0.5
        return query, key, value

    def __call__(self, attn, hidden_states, **kwargs):
        self.base_calls += 1
        self._project_qkv(
            attn,
            hidden_states,
            kwargs["encoder_hidden_states"],
            kwargs["rotary_emb"],
            kwargs["origin_hidden_states_seq_len"],
            kwargs["encoder_hidden_states"] is not None,
        )
        self.last_output = hidden_states.square() + 0.25
        return self.last_output


FakeOfficialProcessor.__name__ = observer.OFFICIAL_PROCESSOR_CLASS
FakeOfficialProcessor.__module__ = observer.OFFICIAL_PROCESSOR_MODULE


def make_masks(*, asset_sha: str = SHA_A) -> observer.AnchorRoleMaskAuthorityV15E:
    masks = torch.zeros((21, 3, 1, 3), dtype=torch.bool)
    masks[:, 0, 0, 0] = True
    masks[:, 1, 0, 1] = True
    masks[:, 2, 0, 2] = True
    return observer.AnchorRoleMaskAuthorityV15E.create(
        anchor_slot="v0",
        anchor_asset_sha256=asset_sha,
        producer_receipt_sha256=SHA_D,
        masks=masks,
    )


def make_capture(
    arm: str,
    *,
    asset_sha: str = SHA_A,
    noise_sha: str = SHA_C,
    block: int = 4,
    rank: int = 0,
) -> observer.PostRopeQKCaptureV15E:
    generator = torch.Generator().manual_seed(101 if arm == "dynamic" else 202)
    query = torch.randn((1, 63, 2, 4), generator=generator)
    key = torch.randn((1, 63, 2, 4), generator=generator)
    if arm == "phase0_static":
        query = query.reshape(1, 21, 3, 2, 4)[:, :1].repeat(1, 21, 1, 1, 1).reshape(1, 63, 2, 4)
        key = key.reshape(1, 21, 3, 2, 4)[:, :1].repeat(1, 21, 1, 1, 1).reshape(1, 63, 2, 4)
    return observer.PostRopeQKCaptureV15E(
        observer.CAPTURE_SCHEMA,
        arm,
        "v0",
        asset_sha,
        SHA_B,
        noise_sha,
        SHA_D,
        SHA_B,
        24,
        block,
        rank,
        4,
        1,
        3,
        query.detach(),
        key.detach(),
    )


class ObserverHookTests(unittest.TestCase):
    def test_active_observer_intercepts_existing_projection_once_and_returns_identity(self):
        bank = observer.QKCaptureBankV15E((4,))
        base = FakeOfficialProcessor()
        wrapper = observer.SelfActionQKAttn1ObserverV15E(
            base, block_index=4, capture_bank=bank
        )
        hidden = torch.arange(63 * 4, dtype=torch.float32).reshape(1, 63, 4)
        invocation = observer.QKCaptureInvocationV15E(
            bank, "dynamic", "v0", SHA_A, SHA_B, SHA_C, SHA_D, SHA_B, 24,
            0, 4, 1, 3
        )
        with observer.observe_self_action_qk_v15e(invocation):
            output = wrapper(
                object(),
                hidden,
                rotary_emb=torch.ones(1),
                origin_hidden_states_seq_len=63,
            )
        self.assertIs(output, base.last_output)
        self.assertEqual((base.base_calls, base.project_calls), (1, 1))
        self.assertNotIn("_project_qkv", base.__dict__)
        capture = bank.get(anchor_slot="v0", arm="dynamic", block_index=4, rank=0)
        self.assertEqual(tuple(capture.query.shape), (1, 63, 2, 2))
        self.assertFalse(capture.query.requires_grad)
        self.assertEqual(wrapper.statistics()["SP_collective_calls_added"], 0)
        self.assertFalse(wrapper.statistics()["output_modified"])

    def test_inactive_observer_is_plain_delegation(self):
        bank = observer.QKCaptureBankV15E((4,))
        base = FakeOfficialProcessor()
        wrapper = observer.SelfActionQKAttn1ObserverV15E(
            base, block_index=4, capture_bank=bank
        )
        hidden = torch.ones((1, 63, 4), dtype=torch.float32)
        output = wrapper(
            object(),
            hidden,
            rotary_emb=torch.ones(1),
            origin_hidden_states_seq_len=63,
        )
        self.assertIs(output, base.last_output)
        self.assertEqual((base.base_calls, base.project_calls), (1, 1))
        self.assertEqual(bank.capture_count, 0)

    def test_install_is_attn1_only_and_reversible(self):
        class Attn:
            def __init__(self):
                self.processor = FakeOfficialProcessor()

            def set_processor(self, value):
                self.processor = value

        model = SimpleNamespace(
            blocks=[SimpleNamespace(attn1=Attn()) for _ in range(30)]
        )
        originals = (model.blocks[4].attn1.processor, model.blocks[9].attn1.processor)
        bank = observer.QKCaptureBankV15E((4, 9))
        handle = observer.install_self_action_qk_observer_v15e(model, capture_bank=bank)
        self.assertIsInstance(
            model.blocks[4].attn1.processor, observer.SelfActionQKAttn1ObserverV15E
        )
        self.assertFalse(handle.receipt()["route_authorized"])
        handle.restore()
        self.assertIs(model.blocks[4].attn1.processor, originals[0])
        self.assertIs(model.blocks[9].attn1.processor, originals[1])


class ExtractionTests(unittest.TestCase):
    def test_dynamic_static_pair_emits_only_registered_role_edges(self):
        candidate = observer.extract_local_action_graph_v15e(
            make_capture("dynamic"),
            make_capture("phase0_static"),
            make_masks(),
            action_id="place",
        )
        self.assertEqual(tuple(candidate.graph.shape), (2, 21, 3, 21, 3))
        self.assertEqual(tuple(candidate.timing_trace.shape), (21, 4))
        self.assertEqual(int(torch.count_nonzero(candidate.graph[:, 0])), 0)
        self.assertLessEqual(float(candidate.graph.sum(3).abs().max()), 1.0e-6)
        # human -> recipient is not in the extraction allowlist.
        human = observer.GENERIC_ROLES.index("human_agent")
        recipient = observer.GENERIC_ROLES.index("recipient")
        self.assertEqual(
            int(torch.count_nonzero(candidate.graph[:, :, human, :, recipient])), 0
        )
        self.assertTrue(candidate.mechanically_qualified)

    def test_pair_provenance_and_anchor_local_masks_fail_closed(self):
        with self.assertRaises(observer.SelfActionGraphObserverV15EError):
            observer.extract_local_action_graph_v15e(
                make_capture("dynamic"),
                make_capture("phase0_static", noise_sha="e" * 64),
                make_masks(),
                action_id="place",
            )
        with self.assertRaises(observer.SelfActionGraphObserverV15EError):
            observer.extract_local_action_graph_v15e(
                make_capture("dynamic"),
                make_capture("phase0_static"),
                make_masks(asset_sha="f" * 64),
                action_id="place",
            )

    def test_masks_require_all_three_roles_in_every_phase(self):
        masks = torch.zeros((21, 3, 1, 3), dtype=torch.bool)
        masks[:, 0, 0, 0] = True
        masks[:, 1, 0, 1] = True
        masks[:-1, 2, 0, 2] = True
        with self.assertRaises(observer.SelfActionGraphObserverV15EError):
            observer.AnchorRoleMaskAuthorityV15E.create(
                anchor_slot="v0",
                anchor_asset_sha256=SHA_A,
                producer_receipt_sha256=SHA_D,
                masks=masks,
            )

    def test_preflight_never_authorizes_and_requires_raw_qk_clear(self):
        masks = make_masks()
        candidate = observer.extract_local_action_graph_v15e(
            make_capture("dynamic"),
            make_capture("phase0_static"),
            masks,
            action_id="place",
        )
        bank = observer.QKCaptureBankV15E((4,))
        bank.capture(make_capture("dynamic"))
        bank.capture(make_capture("phase0_static"))
        resident = observer.build_representation_preflight_receipt_v15e(
            (candidate,), qk_bank=bank, role_masks=masks
        )
        self.assertFalse(resident["raw_qk_cleared_before_target_process"])
        self.assertFalse(resident["representation_candidate_qualified"])
        bank.clear()
        cleared = observer.build_representation_preflight_receipt_v15e(
            (candidate,), qk_bank=bank, role_masks=masks
        )
        self.assertTrue(cleared["raw_qk_cleared_before_target_process"])
        self.assertTrue(cleared["representation_candidate_qualified"])
        for key in (
            "four_anchor_consensus_passed",
            "route_authorized",
            "decode_authorized",
            "training_authorized",
            "scientific_claim_authorized",
        ):
            self.assertFalse(cleared[key])
        self.assertIn("query", cleared["anchor_process_persistent_output_forbidden"])


if __name__ == "__main__":
    unittest.main()
