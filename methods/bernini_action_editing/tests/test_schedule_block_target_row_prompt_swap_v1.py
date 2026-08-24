from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import schedule_block_target_row_prompt_swap_v1 as swap  # noqa: E402


CORE_PATH = METHOD_ROOT / "schedule_block_target_row_prompt_swap_v1.py"


def owner_binding(owner="correct_owner", schedule_index=29):
    marker = "1" if owner == "correct_owner" else "2"
    return swap.OwnerInputBinding(
        owner=owner,
        schedule_index=schedule_index,
        timestep=swap.policy.exact40.PINNED_TIMESTEPS[schedule_index],
        sigma_float32_be_hex=swap.policy.exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule_index],
        orbit_row_digest=swap.ORBIT_ROW_DIGEST,
        target_source_full_blob_sha256=swap.OWNER_FULL_BLOB_SHA256["correct_owner"],
        owner_full_blob_sha256=swap.OWNER_FULL_BLOB_SHA256[owner],
        owner_reference_blob_sha256=swap.OWNER_REFERENCE_BLOB_SHA256[owner],
        decoded_target_tensor_sha256="1" * 64,
        decoded_owner_full_tensor_sha256=marker * 64,
        decoded_owner_reference_tensor_sha256=tuple(
            f"{value:x}" * 64
            for value in (
                (4, 5, 6, 7)
                if owner == "correct_owner"
                else (8, 9, 10, 11)
            )
        ),
        epsilon_sha256="c" * 64,
        target_x_s_sha256="d" * 64,
        prepared_visual_prefix_sha256=marker * 64,
        prepared_prefix_rotary_sha256="e" * 64,
        total_tokens=5,
        condition_tokens=3,
    )


class PromptSwapPlanTests(unittest.TestCase):
    def test_c0_is_exact_six_without_published_internal_parity_outputs(self) -> None:
        plan = swap.build_plan("c0-smoke")
        self.assertEqual(plan["decoded_output_count"], 6)
        self.assertEqual(plan["internal_noop_parity_outputs_published"], 0)
        cells = plan["cells"]
        self.assertEqual(
            [(row["owner"], row["band_name"], row["branch"]) for row in cells],
            [
                ("correct_owner", "none", "noop"),
                ("correct_owner", "late_middle", "forward"),
                ("correct_owner", "all30_reference", "forward"),
                ("wrong_owner", "none", "noop"),
                ("wrong_owner", "late_middle", "forward"),
                ("wrong_owner", "all30_reference", "forward"),
            ],
        )
        self.assertEqual({row["schedule_index"] for row in cells}, {29})
        self.assertEqual(cells[1]["selected_blocks"], list(range(16, 23)))
        self.assertFalse(plan["optimizer_present"])
        self.assertFalse(plan["gradient_computation"])
        self.assertEqual(plan["scheduler_steps"], 0)
        self.assertTrue(plan["noop_is_numerical_not_semantic_baseline"])
        self.assertTrue(plan["incomplete_is_exploratory_calibration_failed"])
        self.assertTrue(plan["only_reverse_is_directional_negative_candidate"])
        self.assertFalse(plan["negative_cluster_semantically_validated"])
        self.assertFalse(plan["branch_calibration_scientific_veto_authorized"])
        self.assertFalse(
            plan["branch_calibration_authority"]["noop"]["semantic_negative_authorized"]
        )

    def test_full_grid_is_exact_88_correct_plus_24_wrong(self) -> None:
        plan = swap.build_plan("full-grid")
        self.assertEqual(plan["decoded_output_count"], 112)
        correct = [row for row in plan["cells"] if row["owner"] == "correct_owner"]
        wrong = [row for row in plan["cells"] if row["owner"] == "wrong_owner"]
        self.assertEqual((len(correct), len(wrong)), (88, 24))
        for index in (16, 29, 35, 38):
            correct_s = [row for row in correct if row["schedule_index"] == index]
            wrong_s = [row for row in wrong if row["schedule_index"] == index]
            self.assertEqual((len(correct_s), len(wrong_s)), (22, 6))
            self.assertEqual(sum(row["branch"] == "noop" for row in correct_s), 1)
            self.assertEqual(sum(row["band_name"] == "all30_reference" for row in correct_s), 1)
            self.assertEqual(
                {row["branch"] for row in wrong_s}, {"noop", "forward"}
            )
        self.assertEqual(
            plan["forward_prompt_authority_mapping"],
            "forward<-branch_descriptions.action",
        )

    def test_plan_is_deterministic_and_hostile_mutations_fail_closed(self) -> None:
        for profile in ("c0-smoke", "full-grid"):
            first = swap.build_plan(profile)
            second = swap.build_plan(profile)
            self.assertEqual(swap.canonical_json_bytes(first), swap.canonical_json_bytes(second))
            self.assertIs(swap.validate_plan(first, profile=profile), first)
            changed = copy.deepcopy(first)
            changed["cells"][0]["schedule_index"] = 28
            with self.assertRaises(swap.PromptSwapError):
                swap.validate_plan(changed, profile=profile)
            changed = copy.deepcopy(first)
            changed["optimizer_present"] = True
            with self.assertRaises(swap.PromptSwapError):
                swap.validate_plan(changed, profile=profile)

    def test_nested_plan_mutation_cannot_poison_future_expected_plan(self) -> None:
        pristine = swap.build_plan("full-grid")
        pristine_bytes = swap.canonical_json_bytes(pristine)
        hostile = swap.build_plan("full-grid")
        hostile["branch_calibration_authority"]["noop"]["role"] = "semantic_negative"
        hostile["cells"][0]["selected_blocks"].append(29)
        hostile["block_bands"]["early"].append(29)
        with self.assertRaises(swap.PromptSwapError):
            swap.validate_plan(hostile, profile="full-grid")
        fresh = swap.build_plan("full-grid")
        self.assertEqual(swap.canonical_json_bytes(fresh), pristine_bytes)
        self.assertEqual(
            fresh["branch_calibration_authority"]["noop"]["role"],
            "numerical_baseline_only",
        )
        self.assertEqual(fresh["cells"][0]["selected_blocks"], [])
        self.assertEqual(fresh["block_bands"]["early"], list(range(8)))

    def test_policy_axes_and_wrong_owner_are_not_conflated(self) -> None:
        self.assertNotIn("wrong_owner", swap.TEXT_BRANCHES)
        self.assertEqual(swap.OWNERS, ("correct_owner", "wrong_owner"))
        self.assertEqual(swap.PROMPT_AUTHORITY_MAPPING["forward"], "action")
        self.assertNotIn("wrong_actor", swap.TEXT_BRANCHES)

    def test_ast_has_no_duplicate_dataclass_fields_or_duplicate_terminal_returns(self) -> None:
        tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            annotated = [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
            ]
            self.assertEqual(len(annotated), len(set(annotated)), node.name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for left, right in zip(node.body, node.body[1:]):
                    self.assertFalse(
                        isinstance(left, ast.Return) and isinstance(right, ast.Return),
                        node.name,
                    )

    def test_all_public_authority_maps_and_global_literals_are_immutable(self) -> None:
        pristine_plan = swap.canonical_json_bytes(swap.build_plan("full-grid"))
        pristine_correct = owner_binding().digest
        mutations = (
            (swap.PROMPT_AUTHORITY_MAPPING, "forward", "wrong_actor"),
            (swap.OWNER_FULL_BLOB_SHA256, "correct_owner", "0" * 64),
            (swap.OWNER_REFERENCE_BLOB_SHA256, "wrong_owner", ("0" * 64,) * 4),
        )
        for authority, key, value in mutations:
            with self.assertRaises(TypeError):
                authority[key] = value
        with self.assertRaises(TypeError):
            swap.OWNER_REFERENCE_BLOB_SHA256["correct_owner"][0] = "0" * 64
        self.assertEqual(swap.canonical_json_bytes(swap.build_plan("full-grid")), pristine_plan)
        self.assertEqual(owner_binding().digest, pristine_correct)

        tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            self.assertNotIsInstance(value, (ast.Dict, ast.List, ast.Set))


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class PromptSwapDynamicTests(unittest.TestCase):
    class BaseProcessor:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, attn, hidden_states, *, encoder_hidden_states, **kwargs):
            self.calls += 1
            value = encoder_hidden_states.float().mean().to(hidden_states.dtype)
            return hidden_states + value

    @staticmethod
    def call(processor, hidden, encoder):
        with torch.inference_mode():
            return PromptSwapDynamicTests.call_raw(processor, hidden, encoder)

    @staticmethod
    def call_raw(processor, hidden, encoder, attn=None):
        return processor(
            attn, hidden, encoder_hidden_states=encoder, attention_mask=None,
            rotary_emb=None, batch_image_vae_seqlen=[hidden.shape[1]],
            text_features_length=[encoder.shape[1]],
            origin_hidden_states_seq_len=hidden.shape[1],
            split_hidden_states_seq_len=hidden.shape[1],
        )

    @staticmethod
    def owner_binding(owner="correct_owner", schedule_index=29):
        return owner_binding(owner, schedule_index)

    def capture_bank(self, encoder) -> tuple[swap.PostConditionBranchCache, list]:
        processors = [
            swap.TargetRowPromptSwapProcessor(self.BaseProcessor(), block_index=index)
            for index in range(30)
        ]
        cache = swap.PostConditionBranchCache("forward")
        layout = swap.NativeTargetSuffixLayout(5, 3, 0, 1)
        invocation = swap.PromptSwapInvocation(
            "capture", 29, swap.ALL30_BAND, "forward", "correct_owner",
            self.owner_binding(), encoder, layout, cache,
        )
        hidden = torch.zeros((1, 5, 2), dtype=torch.float32)
        with swap.activate_prompt_swap(invocation, encoder_hidden_states=encoder):
            for processor in processors:
                self.call(processor, hidden, encoder)
        self.assertTrue(cache.sealed)
        return cache, processors

    def test_sp4_selector_append_padding_and_target_crosses_ranks(self) -> None:
        counts = []
        receipts = []
        for rank in range(4):
            layout = swap.NativeTargetSuffixLayout(23, 11, rank, 4)
            counts.append(int(layout.local_target_selector(device="cpu").sum().item()))
            receipts.append(layout.receipt())
        self.assertEqual(counts, [0, 1, 6, 5])
        self.assertTrue(all(row["padded_tokens"] == 24 for row in receipts))
        self.assertEqual(receipts[0]["rank_spans_in_padded_global_sequence"], [[0, 6], [6, 12], [12, 18], [18, 24]])

    def test_mixed_selected_uses_same_hidden_and_changes_target_only(self) -> None:
        branch_encoder = torch.full((1, 2, 3), 3.0)
        cache, processors = self.capture_bank(branch_encoder)
        processor = processors[16]
        noop_encoder = torch.full((1, 2, 3), 1.0)
        hidden = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
        layout = swap.NativeTargetSuffixLayout(5, 3, 0, 1)
        invocation = swap.PromptSwapInvocation(
            "mixed", 29, "late_middle", "forward", "correct_owner",
            self.owner_binding(), noop_encoder, layout, cache,
        )
        before = processor.statistics()
        with swap.activate_prompt_swap(invocation, encoder_hidden_states=noop_encoder):
            result = self.call(processor, hidden, noop_encoder)
        expected = hidden + 1.0
        expected[:, 3:] = hidden[:, 3:] + 3.0
        self.assertTrue(torch.equal(result, expected))
        after = processor.statistics()
        self.assertEqual(after["base_calls"] - before["base_calls"], 2)
        self.assertEqual(after["alternate_calls"] - before["alternate_calls"], 1)
        self.assertEqual(after["non_target_parity_checks"] - before["non_target_parity_checks"], 1)
        self.assertTrue(cache.assert_unchanged()["all_30_content_and_versions_unchanged"])

    def test_unselected_block_is_single_official_call_and_returns_base_object(self) -> None:
        branch_encoder = torch.full((1, 2, 3), 3.0)
        cache, processors = self.capture_bank(branch_encoder)
        processor = processors[0]
        noop_encoder = torch.ones((1, 2, 3))
        hidden = torch.zeros((1, 5, 2))
        invocation = swap.PromptSwapInvocation(
            "mixed", 29, "late_middle", "forward", "correct_owner",
            self.owner_binding(), noop_encoder,
            swap.NativeTargetSuffixLayout(5, 3, 0, 1), cache,
        )
        before = processor.statistics()
        with swap.activate_prompt_swap(invocation, encoder_hidden_states=noop_encoder):
            result = self.call(processor, hidden, noop_encoder)
        self.assertTrue(torch.equal(result, hidden + 1.0))
        after = processor.statistics()
        self.assertEqual(after["base_calls"] - before["base_calls"], 1)
        self.assertEqual(after["alternate_calls"] - before["alternate_calls"], 0)

    def test_noop_swap_selected_executes_two_calls_where_and_full_byte_parity(self) -> None:
        processor = swap.TargetRowPromptSwapProcessor(self.BaseProcessor(), block_index=16)
        encoder = torch.ones((1, 2, 3))
        hidden = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
        layout = swap.NativeTargetSuffixLayout(5, 3, 0, 1)
        plain = swap.PromptSwapInvocation(
            "plain_noop", 29, "none", "noop", "correct_owner",
            self.owner_binding(), encoder, layout
        )
        with swap.activate_prompt_swap(plain, encoder_hidden_states=encoder):
            plain_output = self.call(processor, hidden, encoder)
        before = processor.statistics()
        noop_swap = swap.PromptSwapInvocation(
            "noop_swap", 29, "late_middle", "noop", "correct_owner",
            self.owner_binding(), encoder, layout
        )
        with swap.activate_prompt_swap(noop_swap, encoder_hidden_states=encoder):
            swapped = self.call(processor, hidden, encoder)
        self.assertTrue(torch.equal(swapped, plain_output))
        after = processor.statistics()
        self.assertEqual(after["base_calls"] - before["base_calls"], 2)
        self.assertEqual(after["alternate_calls"] - before["alternate_calls"], 1)
        self.assertEqual(after["noop_full_parity_checks"] - before["noop_full_parity_checks"], 1)

    def test_true_inference_tensors_capture_mixed_noop_and_raw_mutation_guard(self) -> None:
        with torch.inference_mode():
            branch_encoder = torch.zeros((1, 2, 3), dtype=torch.float32)
            noop_encoder = torch.ones((1, 2, 3), dtype=torch.float32)
            hidden = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
            self.assertTrue(torch.is_inference(branch_encoder))
            identity = swap.tensor_content_identity(
                branch_encoder, label="inference branch encoder"
            )
            self.assertEqual(identity["tensor_version"], "inference_tensor_no_version")

            cache, processors = self.capture_bank(branch_encoder)
            mixed_invocation = swap.PromptSwapInvocation(
                "mixed", 29, "late_middle", "forward", "correct_owner",
                self.owner_binding(), noop_encoder,
                swap.NativeTargetSuffixLayout(5, 3, 0, 1), cache,
            )
            with swap.activate_prompt_swap(
                mixed_invocation, encoder_hidden_states=noop_encoder
            ):
                mixed = self.call_raw(processors[16], hidden, noop_encoder)
            expected = hidden + 1.0
            expected[:, 3:] = hidden[:, 3:]
            self.assertTrue(swap.raw_tensor_bytes_equal(mixed, expected))
            self.assertTrue(
                cache.assert_unchanged()["all_30_content_and_versions_unchanged"]
            )

            noop_processor = swap.TargetRowPromptSwapProcessor(
                self.BaseProcessor(), block_index=16
            )
            noop_invocation = swap.PromptSwapInvocation(
                "noop_swap", 29, "late_middle", "noop", "correct_owner",
                self.owner_binding(), noop_encoder,
                swap.NativeTargetSuffixLayout(5, 3, 0, 1), None,
            )
            with swap.activate_prompt_swap(
                noop_invocation, encoder_hidden_states=noop_encoder
            ):
                noop_result = self.call_raw(noop_processor, hidden, noop_encoder)
            self.assertTrue(
                swap.raw_tensor_bytes_equal(noop_result, hidden + 1.0)
            )

            # Inference tensors permit mutation while still inside
            # inference_mode.  Signed-zero changes only the raw sign bit and
            # must therefore be rejected even without a version counter.
            branch_encoder.data.fill_(-0.0)
            with self.assertRaisesRegex(
                swap.PromptSwapError, "content or mutation version"
            ):
                cache.get(0)

    def test_capture_mutation_and_context_hostility_fail_closed(self) -> None:
        encoder = torch.ones((1, 2, 3))
        cache, _ = self.capture_bank(encoder)
        encoder.data.add_(1.0)
        with self.assertRaisesRegex(swap.PromptSwapError, "content or mutation version"):
            cache.get(0)

        layout = swap.NativeTargetSuffixLayout(5, 3, 0, 1)
        invocation = swap.PromptSwapInvocation(
            "plain_noop", 29, "none", "noop", "correct_owner",
            self.owner_binding(), encoder, layout
        )
        with self.assertRaisesRegex(swap.PromptSwapError, "raw transformer prompt"):
            with swap.activate_prompt_swap(
                invocation, encoder_hidden_states=encoder.clone()
            ):
                pass

    def test_capture_branch_cache_and_owner_binding_mismatch_fail_closed(self) -> None:
        encoder = torch.ones((1, 2, 3))
        layout = swap.NativeTargetSuffixLayout(5, 3, 0, 1)
        with self.assertRaisesRegex(swap.PromptSwapError, "branch/cache"):
            swap.PromptSwapInvocation(
                "capture", 29, swap.ALL30_BAND, "forward", "correct_owner",
                self.owner_binding(), encoder, layout,
                swap.PostConditionBranchCache("reverse"),
            )
        with self.assertRaisesRegex(swap.PromptSwapError, "owner axis"):
            swap.PromptSwapInvocation(
                "plain_noop", 29, "none", "noop", "wrong_owner",
                self.owner_binding("correct_owner"), encoder, layout,
            )

    def test_owner_pair_binds_shared_target_and_switched_prefix_only(self) -> None:
        correct = self.owner_binding("correct_owner")
        wrong = self.owner_binding("wrong_owner")
        self.assertFalse(correct.receipt()["owner_pair_switch_audited_by_single_binding"])
        self.assertNotIn("wrong_owner_changes_visual_prefix_only", correct.receipt())
        audit = swap.validate_owner_pair_bindings(correct, wrong)
        self.assertTrue(audit["wrong_owner_changes_visual_prefix_only"])
        changed = dict(wrong.__dict__)
        changed["epsilon_sha256"] = "f" * 64
        hostile = swap.OwnerInputBinding(**changed)
        with self.assertRaisesRegex(swap.PromptSwapError, "target/noise/source"):
            swap.validate_owner_pair_bindings(correct, hostile)

    def test_raw_byte_helper_and_noop_double_call_reject_signed_zero(self) -> None:
        positive = torch.zeros((1, 5, 2), dtype=torch.float32)
        negative = torch.copysign(positive, -torch.ones_like(positive))
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(swap.raw_tensor_bytes_equal(positive, negative))

        class AlternatingSignedZero:
            def __init__(self):
                self.calls = 0

            def __call__(self, attn, hidden_states, **kwargs):
                self.calls += 1
                result = torch.zeros_like(hidden_states)
                if self.calls % 2 == 0:
                    result = torch.copysign(result, -torch.ones_like(result))
                return result

        encoder = torch.zeros((1, 2, 3))
        invocation = swap.PromptSwapInvocation(
            "noop_swap", 29, "late_middle", "noop", "correct_owner",
            self.owner_binding(), encoder,
            swap.NativeTargetSuffixLayout(5, 3, 0, 1),
        )
        processor = swap.TargetRowPromptSwapProcessor(
            AlternatingSignedZero(), block_index=16
        )
        with swap.activate_prompt_swap(invocation, encoder_hidden_states=encoder):
            with self.assertRaisesRegex(swap.PromptSwapError, "bit-exact"):
                self.call(processor, positive, encoder)

    def test_signed_zero_data_mutation_of_hidden_encoder_and_cache_is_rejected(self) -> None:
        class SignbitMutator:
            def __init__(self, target):
                self.target = target

            def __call__(self, attn, hidden_states, *, encoder_hidden_states, **kwargs):
                value = hidden_states if self.target == "hidden" else encoder_hidden_states
                value.data.copy_(torch.copysign(torch.zeros_like(value), -torch.ones_like(value)))
                return torch.ones_like(hidden_states)

        for target in ("hidden", "encoder"):
            hidden = torch.zeros((1, 5, 2))
            encoder = torch.zeros((1, 2, 3))
            invocation = swap.PromptSwapInvocation(
                "noop_swap", 29, "late_middle", "noop", "correct_owner",
                self.owner_binding(), encoder,
                swap.NativeTargetSuffixLayout(5, 3, 0, 1),
            )
            processor = swap.TargetRowPromptSwapProcessor(
                SignbitMutator(target), block_index=16
            )
            with swap.activate_prompt_swap(invocation, encoder_hidden_states=encoder):
                with self.assertRaisesRegex(swap.PromptSwapError, "changed"):
                    self.call(processor, hidden, encoder)

        encoder = torch.zeros((1, 2, 3))
        cache, _ = self.capture_bank(encoder)
        encoder.data.copy_(
            torch.copysign(torch.zeros_like(encoder), -torch.ones_like(encoder))
        )
        with self.assertRaisesRegex(swap.PromptSwapError, "content or mutation version"):
            cache.get(0)

    def test_mixed_non_target_signed_zero_corruption_is_rejected(self) -> None:
        class ZeroOrOneProcessor:
            def __call__(self, attn, hidden_states, *, encoder_hidden_states, **kwargs):
                return torch.zeros_like(hidden_states) + encoder_hidden_states.mean()

        branch_encoder = torch.ones((1, 2, 3))
        cache = swap.PostConditionBranchCache("forward")
        cache.begin_capture()
        for index in range(30):
            cache.capture(index, branch_encoder)
        cache.finish_capture()
        noop_encoder = torch.zeros((1, 2, 3))
        hidden = torch.zeros((1, 5, 2))
        processor = swap.TargetRowPromptSwapProcessor(
            ZeroOrOneProcessor(), block_index=16
        )
        invocation = swap.PromptSwapInvocation(
            "mixed", 29, "late_middle", "forward", "correct_owner",
            self.owner_binding(), noop_encoder,
            swap.NativeTargetSuffixLayout(5, 3, 0, 1), cache,
        )
        real_where = torch.where

        def corrupt_non_target(mask, branch, base):
            result = real_where(mask, branch, base)
            negative_zero = torch.copysign(
                torch.zeros_like(result), -torch.ones_like(result)
            )
            return real_where(~mask, negative_zero, result)

        with swap.activate_prompt_swap(invocation, encoder_hidden_states=noop_encoder):
            with mock.patch.object(torch, "where", side_effect=corrupt_non_target):
                with self.assertRaisesRegex(swap.PromptSwapError, "non-target"):
                    self.call(processor, hidden, noop_encoder)

    def test_dtype_alias_grad_and_mutating_processors_fail_closed(self) -> None:
        encoder = torch.ones((1, 2, 3), dtype=torch.float32)
        hidden = torch.zeros((1, 5, 2), dtype=torch.float32)
        layout = swap.NativeTargetSuffixLayout(5, 3, 0, 1)
        invocation = swap.PromptSwapInvocation(
            "noop_swap", 29, "late_middle", "noop", "correct_owner",
            self.owner_binding(), encoder, layout,
        )

        class WrongDtype:
            def __call__(self, attn, hidden_states, **kwargs):
                return hidden_states.double() + 1.0

        class Alias:
            def __call__(self, attn, hidden_states, **kwargs):
                return hidden_states

        class MutateFirst:
            def __call__(self, attn, hidden_states, **kwargs):
                hidden_states.data.add_(1.0)
                return hidden_states.clone()

        for base, pattern in (
            (WrongDtype(), "tensor contract"),
            (Alias(), "aliasing"),
            (MutateFirst(), "changed"),
        ):
            processor = swap.TargetRowPromptSwapProcessor(base, block_index=16)
            local_hidden = hidden.clone()
            with swap.activate_prompt_swap(invocation, encoder_hidden_states=encoder):
                with self.assertRaisesRegex(swap.PromptSwapError, pattern):
                    self.call(processor, local_hidden, encoder)

        processor = swap.TargetRowPromptSwapProcessor(self.BaseProcessor(), block_index=16)
        with swap.activate_prompt_swap(invocation, encoder_hidden_states=encoder):
            with self.assertRaisesRegex(swap.PromptSwapError, "inference/no-grad"):
                self.call_raw(processor, hidden, encoder)

        attn = torch.nn.Linear(2, 2)
        with swap.activate_prompt_swap(invocation, encoder_hidden_states=encoder):
            with self.assertRaisesRegex(swap.PromptSwapError, "trainable"):
                with torch.inference_mode():
                    self.call_raw(processor, hidden, encoder, attn=attn)

    def test_second_call_mutation_is_detected(self) -> None:
        class MutateSecond:
            def __init__(self):
                self.calls = 0

            def __call__(self, attn, hidden_states, *, encoder_hidden_states, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    encoder_hidden_states.data.mul_(2.0)
                return hidden_states.clone() + 1.0

        encoder = torch.ones((1, 2, 3))
        hidden = torch.zeros((1, 5, 2))
        invocation = swap.PromptSwapInvocation(
            "noop_swap", 29, "late_middle", "noop", "correct_owner",
            self.owner_binding(), encoder,
            swap.NativeTargetSuffixLayout(5, 3, 0, 1),
        )
        processor = swap.TargetRowPromptSwapProcessor(MutateSecond(), block_index=16)
        with swap.activate_prompt_swap(invocation, encoder_hidden_states=encoder):
            with self.assertRaisesRegex(swap.PromptSwapError, "after second"):
                self.call(processor, hidden, encoder)

    def test_install_restore_preserves_attn2_modules_and_processors(self) -> None:
        class Attn:
            def __init__(self, processor):
                self.processor = processor

            def set_processor(self, processor):
                self.processor = processor

        blocks = [SimpleNamespace(attn2=Attn(self.BaseProcessor())) for _ in range(30)]
        transformer = SimpleNamespace(blocks=blocks)
        original_attn = [id(block.attn2) for block in blocks]
        originals = [block.attn2.processor for block in blocks]
        handle = swap.install_prompt_swap_processors(transformer)
        self.assertEqual([id(block.attn2) for block in blocks], original_attn)
        self.assertTrue(
            all(isinstance(block.attn2.processor, swap.TargetRowPromptSwapProcessor) for block in blocks)
        )
        handle.restore()
        self.assertTrue(all(block.attn2.processor is originals[index] for index, block in enumerate(blocks)))

    def test_install_and_restore_setter_exceptions_rollback_transactionally(self) -> None:
        class HostileAttn:
            def __init__(self, processor):
                self.processor = processor
                self.throw_after_assign = False

            def set_processor(self, processor):
                self.processor = processor
                if self.throw_after_assign:
                    self.throw_after_assign = False
                    raise RuntimeError("hostile setter after assignment")

        blocks = [SimpleNamespace(attn2=HostileAttn(self.BaseProcessor())) for _ in range(30)]
        transformer = SimpleNamespace(blocks=blocks)
        originals = [block.attn2.processor for block in blocks]
        blocks[9].attn2.throw_after_assign = True
        with self.assertRaisesRegex(swap.PromptSwapError, "installation failed transactionally"):
            swap.install_prompt_swap_processors(transformer)
        self.assertTrue(all(block.attn2.processor is originals[index] for index, block in enumerate(blocks)))

        handle = swap.install_prompt_swap_processors(transformer)
        wrappers = [block.attn2.processor for block in blocks]
        blocks[12].attn2.throw_after_assign = True
        with self.assertRaisesRegex(swap.PromptSwapError, "restore failed transactionally"):
            handle.restore()
        self.assertFalse(handle.restored)
        self.assertTrue(all(block.attn2.processor is wrappers[index] for index, block in enumerate(blocks)))
        handle.restore()
        self.assertTrue(handle.restored)
        self.assertTrue(all(block.attn2.processor is originals[index] for index, block in enumerate(blocks)))


if __name__ == "__main__":
    unittest.main()
