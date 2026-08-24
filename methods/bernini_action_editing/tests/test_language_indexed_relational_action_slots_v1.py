#!/usr/bin/env python3

from __future__ import annotations

import copy
import inspect
from pathlib import Path
import sys
import unittest


try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # local macOS workspace intentionally has no torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

if TORCH_AVAILABLE:
    import language_indexed_relational_action_slots_v1 as slots  # noqa: E402
else:
    slots = None  # type: ignore[assignment]


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _binding(branch: str) -> slots.LanguageRoleTokenBinding:
    prompts = {
        "action": "The dog picks up the bone and holds it in its mouth.",
        "noop": "The dog stays beside the bone with its mouth clear.",
        "reverse": "The dog lowers and releases the bone from its mouth.",
        "incomplete": "The dog approaches the bone but does not lift it.",
    }
    return slots.LanguageRoleTokenBinding(
        branch=branch,
        prompt_text=prompts[branch],
        token_ids=(101, 102, 103, 104),
        valid_token_count=4,
        actor_token_indices=(0,),
        object_token_indices=(1,),
        anatomical_anchor_token_indices=(2,),
        action_token_indices=(3,),
        tokenizer_receipt_digest=SHA_A,
        text_encoder_receipt_digest=SHA_B,
    )


def _stage_values(values: tuple[int, int, int, int, int]) -> list[int]:
    ranges = slots.CDF_DOG_PREREGISTERED_EVENT_RANGES
    result: list[int] = []
    for stage, value in zip(slots.EVENT_STAGE_ORDER, values):
        begin, stop = getattr(ranges, stage)
        result.extend([value] * (stop - begin))
    if len(result) != slots.LATENT_PHASES:
        raise AssertionError("fixture phase count differs")
    return result


def _fixture_query(branch: str) -> torch.Tensor:
    # 3x3 patch grid.  The four feature dimensions are independent actor,
    # object, anatomical-anchor and action language axes.
    query = torch.zeros(slots.LATENT_PHASES, 9, 1, 4, dtype=torch.float32)
    noop_actor = _stage_values((6, 6, 6, 6, 6))
    noop_object = _stage_values((8, 8, 8, 8, 8))
    if branch == "noop":
        actor, obj = noop_actor, noop_object
    elif branch == "action":
        actor = _stage_values((6, 7, 8, 5, 5))
        obj = _stage_values((8, 8, 8, 5, 5))
    elif branch == "reverse":
        actor = list(reversed(_stage_values((6, 7, 8, 5, 5))))
        obj = list(reversed(_stage_values((8, 8, 8, 5, 5))))
    elif branch == "incomplete":
        actor = _stage_values((6, 7, 8, 8, 8))
        obj = noop_object
    else:
        raise AssertionError(branch)
    for phase in range(slots.LATENT_PHASES):
        actor_patch = actor[phase]
        object_patch = obj[phase]
        query[phase, actor_patch, 0, 0] = 8.0
        query[phase, object_patch, 0, 1] = 8.0
        query[phase, actor_patch, 0, 2] = 8.0
        action_patch = object_patch if branch != "noop" else actor_patch
        query[phase, action_patch, 0, 3] = 8.0
    return query


def _fixture_capture(
    branch: str, *, noisy_state_sha256: str = SHA_C
) -> slots.GlobalRelationalQKCapture:
    keys = torch.eye(4, dtype=torch.float32).reshape(4, 1, 4)
    return slots._global_capture_unsafe_for_test(  # noqa: SLF001
        branch=branch,
        binding=_binding(branch),
        target_queries=_fixture_query(branch),
        text_keys=keys,
        patch_height=3,
        patch_width=3,
        shared_noisy_state_sha256=noisy_state_sha256,
    )


def _fixture_bank() -> dict[str, slots.GlobalRelationalQKCapture]:
    return {branch: _fixture_capture(branch) for branch in slots.BRANCH_ORDER}


_ModuleBase = nn.Module if TORCH_AVAILABLE else object


class _FakeAttention(_ModuleBase):
    def __init__(self, *, normalized: bool = False) -> None:
        super().__init__()
        self.heads = 2
        self.to_q = nn.Linear(8, 8, bias=False)
        self.to_k = nn.Linear(8, 8, bias=False)
        self.norm_q = nn.LayerNorm(8) if normalized else None
        self.norm_k = nn.LayerNorm(8) if normalized else None
        with torch.no_grad():
            self.to_q.weight.copy_(torch.eye(8))
            self.to_k.weight.copy_(torch.eye(8))


class _FakeBlock(_ModuleBase):
    def __init__(self, *, normalized: bool = False) -> None:
        super().__init__()
        self.attn2 = _FakeAttention(normalized=normalized)


class _FakeTransformer(_ModuleBase):
    def __init__(self, *, normalized: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock(normalized=normalized)])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable in this environment")
class LanguageIndexedRelationalActionSlotsTests(unittest.TestCase):
    def test_hook_plan_is_narrow_real_bernini_qk_surface(self) -> None:
        plan = slots.bernini_relational_hook_plan()
        self.assertEqual(len(plan), 2)
        self.assertEqual(
            [row.module_path for row in plan],
            [
                "diff_dec.transformer.blocks.15.attn2.to_q",
                "diff_dec.transformer.blocks.15.attn2.to_k",
            ],
        )
        self.assertTrue(all(row.hook_kind == "post" for row in plan))

    def test_language_roles_are_text_only_disjoint_and_sealed(self) -> None:
        binding = _binding("action")
        receipt = binding.receipt()
        self.assertFalse(receipt["visual_localization_annotation_used"])
        self.assertEqual(
            set(receipt["role_token_indices"]), set(slots.ROLE_ORDER)
        )
        with self.assertRaises(slots.RelationalActionSlotError):
            slots.LanguageRoleTokenBinding(
                branch="action",
                prompt_text="dog bone mouth lift",
                token_ids=(1, 2, 3),
                valid_token_count=3,
                actor_token_indices=(0,),
                object_token_indices=(0,),
                anatomical_anchor_token_indices=(1,),
                action_token_indices=(2,),
                tokenizer_receipt_digest=SHA_A,
                text_encoder_receipt_digest=SHA_B,
            )

    def test_exact21_five_stage_partition_is_closed(self) -> None:
        ranges = slots.CDF_DOG_PREREGISTERED_EVENT_RANGES
        self.assertEqual(
            list(ranges.receipt()),
            ["approach", "contact", "grip", "lift", "hold"],
        )
        path = torch.arange(21 * 3, dtype=torch.float32).reshape(21, 3)
        self.assertEqual(tuple(ranges.reduce(path).shape), (5, 3))
        with self.assertRaises(slots.RelationalActionSlotError):
            slots.EventPhaseRanges(
                approach=(0, 5),
                contact=(5, 8),
                grip=(9, 11),
                lift=(11, 16),
                hold=(16, 21),
            )

    def test_target_suffix_layout_restores_contiguous_sp4_exactly_once(self) -> None:
        layouts = [
            slots.build_target_suffix_sp_layout(
                patch_height=2,
                patch_width=2,
                condition_tokens=20,
                total_tokens=20 + 21 * 4,
                sp_rank=rank,
            )
            for rank in range(4)
        ]
        all_indices = torch.cat([row.target_flat_indices for row in layouts])
        self.assertTrue(
            torch.equal(torch.sort(all_indices).values, torch.arange(84))
        )
        with self.assertRaises(slots.RelationalActionSlotError):
            slots.build_target_suffix_sp_layout(
                patch_height=2,
                patch_width=2,
                condition_tokens=20,
                total_tokens=105,
                sp_rank=0,
            )

    def test_soft_slots_localize_without_threshold_topk_or_visual_annotation(self) -> None:
        capture = _fixture_capture("action")
        result = slots.compute_language_indexed_soft_slots(capture)
        self.assertEqual(tuple(result.relation_path.shape), (21, 15))
        self.assertEqual(set(result.role_weights), set(slots.ROLE_ORDER))
        for role in slots.ROLE_ORDER:
            self.assertTrue(
                torch.allclose(
                    result.role_weights[role].sum(dim=1),
                    torch.ones(21),
                    rtol=0.0,
                    atol=1.0e-6,
                )
            )
        object_centroid = result.role_centroids["object"]
        # The object moves from lower-right to middle-right in the lift phase.
        self.assertLess(float(object_centroid[12, 1]), float(object_centroid[0, 1]))

    def test_relational_quotient_uses_all_controls_and_preservation_rows(self) -> None:
        audit = slots.audit_relational_action_slots(_fixture_bank())
        self.assertEqual(tuple(audit.target_quotient.shape), (5, 15))
        self.assertEqual(tuple(audit.reverse_quotient.shape), (5, 15))
        self.assertEqual(tuple(audit.incomplete_quotient.shape), (5, 15))
        self.assertGreater(audit.metrics["action_noop_quotient_norm"], 0.0)
        self.assertGreater(
            audit.metrics["reverse_retimed_cosine"],
            audit.metrics["action_reverse_same_order_cosine"],
        )
        self.assertEqual(
            set(audit.preservation.rows), set(slots.PRESERVATION_ROW_NAMES)
        )
        self.assertEqual(
            set(audit.event_proxy_metrics),
            {
                "approach_distance_decrease_proxy",
                "contact_overlap_gain_proxy",
                "grip_anchor_object_overlap_gain_proxy",
                "lift_screen_upward_proxy",
                "hold_anchor_object_overlap_proxy",
                "hold_lift_retention_proxy",
            },
        )

    def test_counterfactuals_must_share_the_same_noisy_state(self) -> None:
        bank = _fixture_bank()
        bank["reverse"] = _fixture_capture(
            "reverse", noisy_state_sha256=SHA_D
        )
        with self.assertRaises(slots.RelationalActionSlotError):
            slots.audit_relational_action_slots(bank)

    def test_tiny_observer_hooks_once_detaches_and_does_not_mutate(self) -> None:
        transformer = _FakeTransformer()
        observer = slots.BerniniRelationalCrossAttentionObserver(
            transformer,
            block_index=0,
            expected_hidden_size=8,
            tiny_fixture=True,
        )
        layout = slots.build_target_suffix_sp_layout(
            patch_height=1,
            patch_width=2,
            condition_tokens=0,
            total_tokens=42,
            sp_rank=0,
            sp_size=1,
        )
        attention = transformer.blocks[0].attn2
        before_q = attention.to_q.weight.detach().clone()
        before_k = attention.to_k.weight.detach().clone()
        observer.install()
        try:
            with observer.capture(
                layout=layout,
                binding=_binding("action"),
                shared_noisy_state_sha256=SHA_C,
            ) as holder:
                attention.to_q(torch.randn(1, 42, 8, requires_grad=True))
                attention.to_k(torch.randn(1, 4, 8, requires_grad=True))
            self.assertEqual(len(holder), 1)
            capture = holder[0]
            self.assertEqual(tuple(capture.target_queries.shape), (42, 2, 4))
            self.assertEqual(tuple(capture.text_keys.shape), (4, 2, 4))
            self.assertFalse(capture.target_queries.requires_grad)
            self.assertFalse(capture.text_keys.requires_grad)
            self.assertEqual(capture.runtime_origin, "tiny_torch_fixture")
            self.assertTrue(torch.equal(attention.to_q.weight, before_q))
            self.assertTrue(torch.equal(attention.to_k.weight, before_k))
        finally:
            observer.remove()

    def test_sp4_assembly_restores_global_query_order(self) -> None:
        transformer = _FakeTransformer()
        observer = slots.BerniniRelationalCrossAttentionObserver(
            transformer,
            block_index=0,
            expected_hidden_size=8,
            tiny_fixture=True,
        )
        attention = transformer.blocks[0].attn2
        context = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        rows = []
        observer.install()
        try:
            for rank in range(4):
                layout = slots.build_target_suffix_sp_layout(
                    patch_height=2,
                    patch_width=2,
                    condition_tokens=0,
                    total_tokens=84,
                    sp_rank=rank,
                )
                start = rank * layout.local_length
                query = torch.arange(
                    start * 8,
                    (start + layout.local_length) * 8,
                    dtype=torch.float32,
                ).reshape(1, layout.local_length, 8)
                with observer.capture(
                    layout=layout,
                    binding=_binding("action"),
                    shared_noisy_state_sha256=SHA_C,
                ) as holder:
                    attention.to_q(query)
                    attention.to_k(context)
                rows.append(holder[0])
        finally:
            observer.remove()
        global_capture = slots.assemble_sp4_relational_capture(rows)
        self.assertEqual(tuple(global_capture.target_queries.shape), (21, 4, 2, 4))
        expected = torch.arange(84 * 8, dtype=torch.float32).reshape(21, 4, 2, 4)
        self.assertTrue(torch.equal(global_capture.target_queries, expected))
        self.assertEqual(
            global_capture.runtime_origin,
            "assembled_sp4_tiny_torch_fixture",
        )

    def test_observer_replays_flat_inner_dim_norm_before_head_unflatten(self) -> None:
        transformer = _FakeTransformer(normalized=True)
        observer = slots.BerniniRelationalCrossAttentionObserver(
            transformer,
            block_index=0,
            expected_hidden_size=8,
            tiny_fixture=True,
        )
        layout = slots.build_target_suffix_sp_layout(
            patch_height=1,
            patch_width=2,
            condition_tokens=0,
            total_tokens=42,
            sp_rank=0,
            sp_size=1,
        )
        attention = transformer.blocks[0].attn2
        query = torch.arange(42 * 8, dtype=torch.float32).reshape(1, 42, 8)
        context = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)
        expected_q = attention.norm_q(attention.to_q(query)).reshape(42, 2, 4)
        expected_k = attention.norm_k(attention.to_k(context)).reshape(4, 2, 4)
        observer.install()
        try:
            with observer.capture(
                layout=layout,
                binding=_binding("action"),
                shared_noisy_state_sha256=SHA_C,
            ) as holder:
                attention.to_q(query)
                attention.to_k(context)
        finally:
            observer.remove()
        self.assertTrue(torch.equal(holder[0].target_queries, expected_q))
        self.assertTrue(torch.equal(holder[0].text_keys, expected_k))

    def test_receipt_is_closed_and_can_never_authorize_training(self) -> None:
        bank = _fixture_bank()
        audit = slots.audit_relational_action_slots(bank)
        provenance = slots.RelationalProbeProvenance(
            probe_id="cdf-dog-relational-fixture",
            checkpoint_content_sha256=SHA_A,
            source_video_sha256=SHA_B,
            source_registry_sha256=SHA_D,
            method_revision_sha256=SHA_E,
            shared_noisy_state_sha256=SHA_C,
            query_seed=2026081701,
        )
        receipt = slots.build_relational_probe_receipt(
            provenance=provenance,
            captures=bank,
            audit=audit,
        )
        slots.validate_relational_probe_receipt(receipt)
        self.assertEqual(
            receipt["status"],
            "READ_ONLY_INTERNAL_RELATIONAL_DIAGNOSTIC_ZERO_UPDATES",
        )
        self.assertFalse(receipt["authority"]["scientific_authority"])
        self.assertFalse(receipt["authority"]["real_auh_runtime_validated"])
        self.assertEqual(receipt["authority"]["training_updates_authorized"], 0)
        self.assertEqual(receipt["authority"]["parameter_updates_executed"], 0)
        self.assertFalse(
            receipt["diagnostics"]["ordered_event_semantics_adjudicated"]
        )
        self.assertTrue(
            all(
                value is False
                for value in receipt["forbidden_inputs_and_actions"].values()
            )
        )

        changed = copy.deepcopy(dict(receipt))
        changed["authority"]["scientific_authority"] = True
        changed["receipt_digest"] = slots.object_sha256(
            {
                key: value
                for key, value in changed.items()
                if key != "receipt_digest"
            }
        )
        with self.assertRaises(slots.RelationalActionSlotError):
            slots.validate_relational_probe_receipt(changed)

    def test_public_runtime_surfaces_accept_no_visual_localizer_or_success_callback(self) -> None:
        surfaces = (
            slots.compute_language_indexed_soft_slots,
            slots.audit_relational_action_slots,
            slots.source_native_preservation_rows,
            slots.build_relational_probe_receipt,
        )
        forbidden = {
            "mask",
            "track",
            "pose",
            "flow",
            "trajectory",
            "detector",
            "segmenter",
            "callback",
            "success",
            "passed",
            "evaluator",
        }
        for surface in surfaces:
            names = set(inspect.signature(surface).parameters)
            self.assertFalse(names.intersection(forbidden), (surface.__name__, names))


if __name__ == "__main__":
    unittest.main()
