from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = METHOD_ROOT / "clean_source_visual_context_adapter_v1.py"
TRAINING_PATH = METHOD_ROOT / "clean_source_visual_context_training_v1.py"
RUNNER_PATH = METHOD_ROOT / "train_clean_source_visual_context_stage_b_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_training_v1 as training

try:
    import torch
    from torch import nn

    import clean_source_visual_context_adapter_v1 as visual

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    visual = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class CleanSourceVisualContextStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core_source = CORE_PATH.read_text(encoding="utf-8")
        cls.training_source = TRAINING_PATH.read_text(encoding="utf-8")
        cls.runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        cls.core_tree = ast.parse(cls.core_source)
        cls.training_tree = ast.parse(cls.training_source)
        cls.runner_tree = ast.parse(cls.runner_source)

    def test_architecture_is_independent_source_only_attention(self) -> None:
        for fragment in (
            "class CleanSourceVisualEncoder",
            "self.patchifier = nn.Conv3d",
            "self.patch_norm = nn.LayerNorm",
            "self.projection = nn.Linear",
            "self.source_role = nn.Embedding",
            "class TargetQuerySourceOnlyAttention",
            "target = query_states[:, selector, :]",
            "k = self.key(memory_fp32)",
            "v = self.value(memory_fp32)",
            "nn.init.zeros_(self.output.weight)",
            '"native_self_attention_kv_replaced": False',
            '"native_self_attention_kv_replayed": False',
            '"native_text_cross_attention_changed": False',
            '"source_reads_target_noise": "declared_per_memory_receipt"',
        ):
            self.assertIn(fragment, self.core_source)
        self.assertNotIn("import source_kv_replay", self.core_source)
        self.assertNotIn("import source_kv_route", self.core_source)
        self.assertNotIn("import preservation_residual", self.core_source)
        self.assertNotIn(".attn1.to_k =", self.core_source)
        self.assertNotIn(".attn1.to_v =", self.core_source)

    def test_target_only_sp_selector_and_checkpoint_replay_are_explicit(self) -> None:
        for fragment in (
            "ALLOWED_SP_SIZES = {1, 4}",
            "def local_target_selector",
            "padded = self.local_length * self.sequence_parallel_size",
            "start = self.sequence_parallel_rank * self.local_length",
            "result[:, selector, :] = delta.to(query_states.dtype)",
            "graph_zero = self.residual_gain.to(dtype=query_states.dtype)",
            "result = query_states * graph_zero",
            "def checkpoint_route_context_fn",
            "_replay_checkpoint_route(route)",
        ):
            self.assertIn(fragment, self.core_source)
        adapter_start = self.core_source.index("    def adapter_delta(")
        adapter_body = self.core_source[
            adapter_start : self.core_source.index(
                "    def forward(", adapter_start
            )
        ]
        self.assertNotIn("result = torch.zeros_like(query_states)", adapter_body)
        self.assertIn("if not bool(selector.any().item()):", adapter_body)
        self.assertIn("return result", adapter_body)

    def test_legacy_posterior_list_is_only_subscripted_at_source_index_zero(self) -> None:
        indexed_functions = []
        for node in ast.walk(self.training_tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            indices = []
            for child in ast.walk(node):
                if not isinstance(child, ast.Subscript):
                    continue
                if (
                    not isinstance(child.value, ast.Name)
                    or child.value.id != "posterior_list"
                ):
                    continue
                slice_node = child.slice
                if isinstance(slice_node, ast.Index):  # Python 3.8 AST
                    slice_node = slice_node.value
                if isinstance(slice_node, ast.Constant):
                    indices.append(slice_node.value)
            if indices:
                indexed_functions.append((node.name, indices))
        self.assertEqual(
            indexed_functions,
            [("_extract_source_blob_from_legacy_container", [0])],
        )
        self.assertEqual(self.training_source.count("posterior_list[0]"), 1)
        for fragment in (
            'SPLIT_COUNTS = {"train": 64, "confirmation": 16, "heldout": 8}',
            "FULL644_ROWS = 644",
            "FULL644_ACTION_FAMILIES = 28",
            '"stage_b_posterior_index_1_synthetic_target_decoded": False',
            '"stage_b_posterior_index_1_synthetic_target_hashed": False',
            '"source_posterior_reused_as_clean_noop_target": True',
            '"training_use_forbidden": True',
            '"user_authorized_exploratory_training": True',
        ):
            self.assertIn(fragment, self.training_source)

    def test_stage_b_runner_only_preloads_physical_index0_store(self) -> None:
        for fragment in (
            "PinnedPhysicalSourceOnlyPosteriorStore(",
            "source_only_preload = store.preload(train_manifest_indices)",
            '"preloaded_rows") != 64',
            '"legacy_parquet_opened") is not False',
            '"synthetic_target_index1_bytes_read") is not False',
        ):
            self.assertIn(fragment, self.runner_source)
        for forbidden in (
            "pq.read_table",
            "posterior_list",
            "_extract_source_blob_from_legacy_container",
        ):
            self.assertNotIn(forbidden, self.runner_source)

    def test_adapter_receipt_is_generic_and_memory_receipt_owns_noise_provenance(self) -> None:
        for fragment in (
            '"key_value_source": "independent_source_visual_memory_only"',
            '"memory_input_kinds_supported": list(MEMORY_INPUT_KINDS)',
            '"target_noise_read_by_memory_encoder": "declared_per_memory_receipt"',
            '"source_reads_target_noise": "declared_per_memory_receipt"',
            '"contains_target_noise": (',
        ):
            self.assertIn(fragment, self.core_source)
        self.assertNotIn('"target_noise_read_by_memory_encoder": False', self.core_source)
        self.assertNotIn('"source_reads_target_noise": False', self.core_source)

    def test_checkpoint_contract_is_exact_zero_through_eighty(self) -> None:
        for fragment in (
            "CHECKPOINT_STEPS = (0, 20, 40, 60, 80)",
            "MAX_OPTIMIZER_STEPS = 80",
            "SAVE_EVERY = 20",
            "start_before_first_optimizer_step",
            "after_optimizer_step",
            "checkpoint_step_{step:08d}.pt",
            "step-0 output projection is not exactly zero",
        ):
            self.assertIn(fragment, self.training_source)

    def test_structural_scope_does_not_self_authorize_training(self) -> None:
        receipt = training.training_contract_receipt()
        self.assertEqual(
            receipt["adapter_block_scope_status"],
            "structural_candidate_not_causally_admitted",
        )
        self.assertTrue(receipt["stage_a_decoded_admission_required_before_optimizer"])
        self.assertFalse(receipt["optimizer_authorized_by_this_contract"])

    def test_create_only_json_writes_one_canonical_object(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            path = Path(value) / "manifest.json"
            payload = {"schema": "one-object", "count": 1}
            training.write_create_only_json(path, payload)
            self.assertEqual(path.read_bytes(), training.canonical_json_bytes(payload) + b"\n")


class SourceOnlySplitContractTests(unittest.TestCase):
    @staticmethod
    def _full_rows() -> list[dict[str, object]]:
        return [
            {
                "iid": f"iid-{index:04d}",
                "group_id": f"group-{index:04d}",
                "action_family": f"family-{index % 28:02d}",
                "source_video_sha256": hashlib.sha256(
                    f"source-{index}".encode("ascii")
                ).hexdigest(),
                "strict_selection_gates_all_true": True,
                "single_dynamic_actor": True,
                "heldout_action_canary_eligible": True,
            }
            for index in range(training.FULL644_ROWS)
        ]

    def test_split_is_deterministic_exact_and_hash_disjoint(self) -> None:
        rows = self._full_rows()
        first = training.deterministic_source_hash_split(rows)
        second = training.deterministic_source_hash_split(list(reversed(rows)))
        self.assertEqual(
            {
                split: tuple(row["iid"] for row in selected)
                for split, selected in first.items()
            },
            {
                split: tuple(row["iid"] for row in selected)
                for split, selected in second.items()
            },
        )
        self.assertEqual(
            {split: len(selected) for split, selected in first.items()},
            training.SPLIT_COUNTS,
        )
        selected = [row for values in first.values() for row in values]
        self.assertEqual(len(selected), 88)
        self.assertEqual(len({row["iid"] for row in selected}), 88)
        self.assertEqual(len({row["group_id"] for row in selected}), 88)
        self.assertEqual(len({row["source_video_sha256"] for row in selected}), 88)
        self.assertGreaterEqual(
            len({row["action_family"] for row in first["train"]}), 16
        )
        self.assertTrue(
            all(row["heldout_action_canary_eligible"] for row in first["heldout"])
        )
        self.assertEqual(
            len({row["action_family"] for row in first["heldout"]}), 8
        )

    def test_heldout_gate_mismatch_and_insufficient_families_fail_closed(self) -> None:
        rows = self._full_rows()
        rows[0]["heldout_action_canary_eligible"] = False
        with self.assertRaisesRegex(
            training.CleanSourceVisualTrainingError, "qualification differs"
        ):
            training.deterministic_source_hash_split(rows)

        rows = self._full_rows()
        for row in rows:
            eligible = row["action_family"] == "family-00"
            row["strict_selection_gates_all_true"] = eligible
            row["single_dynamic_actor"] = eligible
            row["heldout_action_canary_eligible"] = eligible
        with self.assertRaisesRegex(
            training.CleanSourceVisualTrainingError,
            "eight distinct heldout action families",
        ):
            training.deterministic_source_hash_split(rows)

    def test_two_sample_or_non_disjoint_data_fails_closed(self) -> None:
        rows = self._full_rows()
        rows[1]["source_video_sha256"] = rows[0]["source_video_sha256"]
        with self.assertRaisesRegex(
            training.CleanSourceVisualTrainingError, "globally unique"
        ):
            training.deterministic_source_hash_split(rows)
        with self.assertRaisesRegex(
            training.CleanSourceVisualTrainingError, "exactly 644"
        ):
            training.deterministic_source_hash_split(self._full_rows()[:2])

    def test_cadence_refuses_missing_or_posthoc_checkpoints(self) -> None:
        cadence = training.CheckpointCadence()
        with self.assertRaisesRegex(
            training.CleanSourceVisualTrainingError, "exact next step is 0"
        ):
            cadence.observe(20)
        for step in training.CHECKPOINT_STEPS:
            cadence.observe(step)
        cadence.assert_complete()
        self.assertTrue(cadence.receipt()["complete"])

        incomplete = training.CheckpointCadence()
        incomplete.observe(0)
        incomplete.observe(20)
        with self.assertRaisesRegex(
            training.CleanSourceVisualTrainingError, "cannot finish"
        ):
            incomplete.assert_complete()


if _TORCH_AVAILABLE:
    class _NativeAttention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)
            self.to_v = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.processor = object()
            self.added_kv_proj_dim = None
            self.add_k_proj = None
            self.add_v_proj = None


    class _NativeBlock(nn.Module):
        def __init__(
            self, attention: nn.Module, text_attention: nn.Module
        ) -> None:
            super().__init__()
            self.attn1 = attention
            self.attn2 = text_attention

        def forward(self, hidden: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
            return hidden


    class WanTransformer3DModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = visual.HIDDEN_SIZE_1P3B
            self.config = dict(visual.PINNED_TRANSFORMER_CONFIG)
            self.patch_embedding = nn.Conv3d(
                16,
                hidden,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                bias=False,
            )
            shared_attention = _NativeAttention(hidden)
            shared_text_attention = _NativeAttention(hidden)
            self.blocks = nn.ModuleList(
                [
                    _NativeBlock(shared_attention, shared_text_attention)
                    for _ in range(visual.TOTAL_BLOCKS_1P3B)
                ]
            )


    WanTransformer3DModel.__module__ = visual.PINNED_TRANSFORMER_CLASS_MODULE


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class CleanSourceVisualContextDynamicTests(unittest.TestCase):
    def _memory(self, *, hidden: int = 8):
        torch.manual_seed(3)
        encoder = visual.CleanSourceVisualEncoder(
            hidden_size=hidden,
            encoder_width=8,
            patch_size=(1, 2, 2),
            memory_token_cap=12,
        )
        latent = torch.randn((1, 16, 3, 8, 8), dtype=torch.float32).contiguous()
        memory = encoder.build_memory(
            latent,
            source_video_sha256="a" * 64,
            memory_input_latent_sha256="b" * 64,
        )
        return encoder, memory

    def test_encoder_preserves_temporal_phases_and_caps_spatial_tokens(self) -> None:
        encoder, memory = self._memory()
        self.assertEqual(memory.patch_grid, (3, 4, 4))
        self.assertEqual(memory.pooled_grid[0], 3)
        self.assertLessEqual(memory.token_count, 12)
        self.assertEqual(tuple(memory.tokens.shape), (1, memory.token_count, 8))
        receipt = encoder.architecture_receipt()
        self.assertFalse(receipt["temporal_pooling"])
        self.assertEqual(receipt["explicit_source_role_id"], 1)

        noised = encoder.build_memory(
            torch.randn((1, 16, 3, 8, 8), dtype=torch.float32).contiguous(),
            source_video_sha256="c" * 64,
            memory_input_latent_sha256="d" * 64,
            input_kind="same_noise_forward_noised_source",
        )
        self.assertFalse(memory.receipt()["contains_target_noise"])
        self.assertTrue(noised.receipt()["contains_target_noise"])
        self.assertFalse(noised.receipt()["clean_source_input"])

    def test_step_zero_is_exact_base_then_only_target_rows_change(self) -> None:
        _, memory = self._memory()
        attention = visual.TargetQuerySourceOnlyAttention(
            hidden_size=8, attention_width=8, num_heads=2
        )
        queries = torch.randn((1, 7, 8), dtype=torch.float32)
        frozen = torch.randn_like(queries)
        route = visual.VisualContextRoute(7, 3, 0, 1, memory)
        with visual.activate_route(route):
            initial = attention(queries, frozen)
        self.assertTrue(torch.equal(initial, frozen))

        with torch.no_grad():
            attention.output.weight.copy_(torch.eye(8))
        with visual.activate_route(route):
            adapted = attention(queries, frozen)
        self.assertTrue(torch.equal(adapted[:, :3], frozen[:, :3]))
        self.assertGreater(float((adapted[:, 3:] - frozen[:, 3:]).abs().max()), 0.0)

        changed_condition = queries.clone()
        changed_condition[:, :3] += 1000.0
        with visual.activate_route(route):
            changed = attention(changed_condition, frozen)
        self.assertTrue(torch.allclose(changed[:, 3:], adapted[:, 3:]))

    def test_sp4_chunks_equal_sp1_and_padding_is_never_written(self) -> None:
        _, memory = self._memory()
        torch.manual_seed(5)
        attention = visual.TargetQuerySourceOnlyAttention(
            hidden_size=8, attention_width=8, num_heads=2
        )
        with torch.no_grad():
            attention.output.weight.copy_(torch.eye(8))
        global_query = torch.randn((1, 11, 8), dtype=torch.float32)
        base = torch.zeros_like(global_query)
        with visual.activate_route(visual.VisualContextRoute(11, 3, 0, 1, memory)):
            sp1 = attention(global_query, base)

        padded_query = torch.cat(
            (global_query, torch.zeros((1, 1, 8), dtype=torch.float32)), dim=1
        )
        local_outputs = []
        for rank in range(4):
            query = padded_query[:, rank * 3 : (rank + 1) * 3].contiguous()
            with visual.activate_route(
                visual.VisualContextRoute(11, 3, rank, 4, memory)
            ):
                local_outputs.append(attention(query, torch.zeros_like(query)))
        joined = torch.cat(local_outputs, dim=1)
        self.assertTrue(torch.allclose(joined[:, :11], sp1, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(joined[:, 11:], torch.zeros_like(joined[:, 11:])))

    def test_selector_empty_rank_keeps_upstream_backward_graph_isomorphic(self) -> None:
        """Both empty and target-owning SP ranks must enter upstream backward."""

        _, memory = self._memory()

        class _CollectiveTrace(torch.autograd.Function):
            traces: list[str] = []

            @staticmethod
            def forward(ctx, value: torch.Tensor, label: str) -> torch.Tensor:
                ctx.label = label
                return value.clone()

            @staticmethod
            def backward(ctx, gradient: torch.Tensor):
                _CollectiveTrace.traces.append(ctx.label)
                return gradient, None

        traces_by_rank = []
        # total=11, condition=9, local_length=3: rank 0 owns no target and
        # rank 3 owns both target rows plus one append-padding row.
        for rank in (0, 3):
            first = visual.TargetQuerySourceOnlyAttention(
                hidden_size=8, attention_width=8, num_heads=2
            )
            second = visual.TargetQuerySourceOnlyAttention(
                hidden_size=8, attention_width=8, num_heads=2
            )
            query = torch.randn((1, 3, 8), dtype=torch.float32)
            frozen_first = torch.randn_like(query)
            route = visual.VisualContextRoute(11, 9, rank, 4, memory)
            with visual.activate_route(route):
                first_output = first(query, frozen_first)
            # The first adapter has exact-zero output projection.  On an
            # empty rank it must nevertheless seed the graph for downstream
            # frozen blocks, exactly as a target-owning rank does.
            self.assertTrue(torch.equal(first_output, frozen_first))
            self.assertTrue(first_output.requires_grad)

            _CollectiveTrace.traces = []
            gathered = _CollectiveTrace.apply(first_output, "ulysses-gather")
            frozen_second = gathered * 1.25
            with visual.activate_route(route):
                final = second(gathered, frozen_second)
            self.assertTrue(torch.equal(final, frozen_second))
            final.square().mean().backward()
            traces_by_rank.append(tuple(_CollectiveTrace.traces))

            self.assertIsNotNone(first.residual_gain.grad)
            self.assertIsNotNone(second.residual_gain.grad)

        self.assertEqual(
            traces_by_rank,
            [("ulysses-gather",), ("ulysses-gather",)],
        )

    def test_graph_anchor_never_writes_condition_or_padding_rows(self) -> None:
        _, memory = self._memory()
        attention = visual.TargetQuerySourceOnlyAttention(
            hidden_size=8, attention_width=8, num_heads=2
        )
        with torch.no_grad():
            attention.output.weight.copy_(torch.eye(8))
        padded = torch.randn((1, 3, 8), dtype=torch.float32)
        frozen = torch.randn_like(padded)

        # Rank 0 is condition-only and must remain bit-exact even after the
        # output projection can produce a non-zero target residual elsewhere.
        empty_route = visual.VisualContextRoute(11, 9, 0, 4, memory)
        with visual.activate_route(empty_route):
            empty_output = attention(padded, frozen)
        self.assertTrue(torch.equal(empty_output, frozen))
        self.assertTrue(empty_output.requires_grad)

        # Rank 3 owns two targets followed by one padding row.  Only those two
        # target rows may change; the final padding row stays bit-exact.
        target_route = visual.VisualContextRoute(11, 9, 3, 4, memory)
        with visual.activate_route(target_route):
            target_output = attention(padded, frozen)
        self.assertGreater(
            float((target_output[:, :2] - frozen[:, :2]).abs().max()), 0.0
        )
        self.assertTrue(torch.equal(target_output[:, 2:], frozen[:, 2:]))

    def test_install_keeps_native_blocks_qkv_and_text_path_untouched(self) -> None:
        model = WanTransformer3DModel()
        model.requires_grad_(False)
        native_blocks = tuple(id(block) for block in model.blocks)
        native_attention = tuple(
            (
                (
                    id(block.attn1),
                    id(block.attn1.to_q),
                    id(block.attn1.to_k),
                    id(block.attn1.to_v),
                    id(block.attn1.to_out[0]),
                    id(block.attn1.to_out[1]),
                    id(block.attn1.norm_q),
                    id(block.attn1.norm_k),
                    id(block.attn1.processor),
                ),
                (
                    id(block.attn2),
                    id(block.attn2.to_q),
                    id(block.attn2.to_k),
                    id(block.attn2.to_v),
                    id(block.attn2.to_out[0]),
                    id(block.attn2.to_out[1]),
                    id(block.attn2.norm_q),
                    id(block.attn2.norm_k),
                    id(block.attn2.processor),
                ),
            )
            for block in model.blocks
        )
        handle = visual.install_clean_source_visual_context_adapter_v1(
            model,
            runtime_source_commit=visual.PINNED_BERNINI_SOURCE_COMMIT,
            model_revision=visual.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256="c" * 64,
            encoder_width=8,
            visual_patch_size=(1, 4, 4),
            memory_token_cap=16,
            attention_width=8,
            attention_heads=2,
        )
        self.assertEqual(tuple(id(block) for block in model.blocks), native_blocks)
        self.assertEqual(
            tuple(
                (
                    (
                        id(block.attn1),
                        id(block.attn1.to_q),
                        id(block.attn1.to_k),
                        id(block.attn1.to_v),
                        id(block.attn1.to_out[0]),
                        id(block.attn1.to_out[1]),
                        id(block.attn1.norm_q),
                        id(block.attn1.norm_k),
                        id(block.attn1.processor),
                    ),
                    (
                        id(block.attn2),
                        id(block.attn2.to_q),
                        id(block.attn2.to_k),
                        id(block.attn2.to_v),
                        id(block.attn2.to_out[0]),
                        id(block.attn2.to_out[1]),
                        id(block.attn2.norm_q),
                        id(block.attn2.norm_k),
                        id(block.attn2.processor),
                    ),
                )
                for block in model.blocks
            ),
            native_attention,
        )
        self.assertTrue(handle.native_structure_untouched())
        self.assertTrue(handle.base_parameters_frozen())
        self.assertTrue(handle.receipt()["zero_initialized_output_projection"])
        self.assertTrue(all(parameter.requires_grad for _, parameter in handle.trainable_named_parameters()))

        original_text_k = model.blocks[0].attn2.to_k
        model.blocks[0].attn2.to_k = nn.Linear(
            visual.HIDDEN_SIZE_1P3B,
            visual.HIDDEN_SIZE_1P3B,
            bias=False,
        )
        self.assertFalse(handle.native_structure_untouched())
        model.blocks[0].attn2.to_k = original_text_k
        self.assertTrue(handle.native_structure_untouched())

        original_text_processor = model.blocks[0].attn2.processor
        model.blocks[0].attn2.processor = object()
        self.assertFalse(handle.native_structure_untouched())
        model.blocks[0].attn2.processor = original_text_processor
        self.assertTrue(handle.native_structure_untouched())

        handle_receipt = handle.receipt()
        self.assertEqual(
            handle_receipt["target_noise_read_by_memory_encoder"],
            "declared_per_memory_receipt",
        )
        self.assertEqual(
            handle_receipt["source_reads_target_noise"],
            "declared_per_memory_receipt",
        )
        self.assertEqual(
            handle_receipt["memory_input_kinds_supported"],
            list(visual.MEMORY_INPUT_KINDS),
        )
        self.assertEqual(
            handle_receipt["sp_empty_target_rank_graph_anchor"],
            "query_times_trainable_exact_zero_on_every_rank",
        )
        self.assertTrue(
            handle_receipt["sp_collective_backward_graph_isomorphic"]
        )

        latent = torch.randn((1, 16, 2, 4, 4), dtype=torch.float32).contiguous()
        memory = handle.build_memory(
            latent,
            source_video_sha256="d" * 64,
            memory_input_latent_sha256="e" * 64,
        )
        hidden = torch.randn((1, 4, visual.HIDDEN_SIZE_1P3B), dtype=torch.float32)
        with handle.route(visual.VisualContextRoute(4, 1, 0, 1, memory)):
            output = model.blocks[8](hidden)
        self.assertTrue(torch.equal(output, hidden))
        handle.restore()
        self.assertFalse(hasattr(model, "clean_source_visual_context_v1"))

    def test_physical_index0_store_preloads_without_legacy_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            iid = "physical-store-test"
            source_path = root / f"{iid}.source-posterior-index0.pt"
            parameters = torch.arange(
                1 * 32 * 21 * 2 * 2, dtype=torch.float32
            ).reshape(1, 32, 21, 2, 2).contiguous()
            torch.save(parameters, source_path)
            source_sha = training.file_sha256(source_path)
            row = training.SourceOnlySplitRow(
                iid=iid,
                split="train",
                group_id="physical-store-group",
                action_family="dog",
                source_video_sha256="a" * 64,
                strict_selection_gates_all_true=True,
                single_dynamic_actor=True,
                heldout_action_canary_eligible=True,
                source_posterior_path=str(source_path),
                source_posterior_file_sha256=source_sha,
                legacy_shard_path=str(
                    training.PINNED_FULL644_SHARDS / f"{iid}.parquet"
                ),
                legacy_shard_sha256="b" * 64,
                source_posterior_blob_sha256=source_sha,
            )
            manifest = training.SourceOnlySplitManifest(
                rows=(row,),
                manifest_digest="c" * 64,
                source_dataset={},
                source_only_materialization={"root": str(root)},
            )
            mean = torch.zeros((1, 16, 1, 1, 1), dtype=torch.float32)
            std = torch.ones((1, 16, 1, 1, 1), dtype=torch.float32)
            store = training.PinnedPhysicalSourceOnlyPosteriorStore(
                manifest,
                vae_latents_mean=mean,
                vae_latents_std=std,
                verify_files_on_first_access=True,
            )
            preload = store.preload([0])
            self.assertEqual(preload["preloaded_rows"], 1)
            self.assertFalse(preload["legacy_parquet_opened"])
            self.assertFalse(preload["synthetic_target_index1_bytes_read"])
            self.assertTrue(preload["physical_index0_files_only"])
            loaded = store.load(0)
            self.assertIs(loaded, store.load(0))
            self.assertIs(loaded.source_condition, loaded.clean_noop_target)
            self.assertTrue(torch.equal(loaded.source_condition, parameters[:, :16]))
            self.assertFalse(
                (training.PINNED_FULL644_SHARDS / f"{iid}.parquet").exists()
            )

    def test_noop_objective_is_plain_flow_matching_not_reward(self) -> None:
        prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0]])
        loss = visual.no_op_flow_matching_loss(
            prediction=prediction, target_velocity=target
        )
        self.assertEqual(float(loss.item()), 2.5)
        loss.backward()
        self.assertIsNotNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
