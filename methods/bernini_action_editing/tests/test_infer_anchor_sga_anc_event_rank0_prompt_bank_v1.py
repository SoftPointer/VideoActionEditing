from __future__ import annotations

import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = METHOD_ROOT / "infer_anchor_sga_anc_event_v1.py"
DECODE_SCRIPT_PATH = (
    METHOD_ROOT / "scripts/auh_decode_online_anchor_attention_dynaedit_v1.sh"
)
BRIDGE_SCRIPT_PATH = (
    METHOD_ROOT
    / "scripts/auh_dynaedit_dynamic_static_anchor_event01_job140846_v27.sh"
)
V14_WATCHER_PATH = (
    METHOD_ROOT
    / "scripts/auh_watch_decode_real_source_routed_teacher_delta_v14.sh"
)
V14R2_WATCHER_PATH = (
    METHOD_ROOT
    / "scripts/auh_watch_decode_target_owned_routed_teacher_v14r2.sh"
)
V14R2_SIDECAR_VALIDATOR_PATH = (
    METHOD_ROOT / "validate_v14r2_decode_sidecar.py"
)
EXACT_VIDEO_MATERIALIZER_PATH = (
    METHOD_ROOT / "exact_local_video_materializer_v1.py"
)
INFER_LORA_PATH = METHOD_ROOT / "infer_lora.py"
V14R3_ABORT_AUTHORITY_PATH = (
    METHOD_ROOT / "assets/v14r3_last_valid_checkpoint_abort_authority_v1.json"
)


def _load_runner() -> object:
    """Load the runner without requiring the AUH Torch/Bernini environment."""

    stubs = {
        "anchor_sga_anc_controller": types.ModuleType("anchor_sga_anc_controller"),
        "infer_lora": types.ModuleType("infer_lora"),
        "infer_native_identity_generation_canary": types.ModuleType(
            "infer_native_identity_generation_canary"
        ),
        "infer_source_aligned_controller_oracle": types.ModuleType(
            "infer_source_aligned_controller_oracle"
        ),
    }
    stubs["infer_source_aligned_controller_oracle"].NOOP_INSTRUCTION = "noop"
    prior = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "infer_anchor_sga_anc_event_rank0_prompt_bank_fixture", RUNNER_PATH
        )
        if spec is None or spec.loader is None:
            raise AssertionError("cannot load runner fixture")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in prior.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


runner = _load_runner()


def _load_sidecar_validator() -> object:
    name = "validate_v14r2_decode_sidecar_fixture"
    spec = importlib.util.spec_from_file_location(name, V14R2_SIDECAR_VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load v14r2 sidecar validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sidecar_validator = _load_sidecar_validator()


def _load_exact_video_materializer_runtime() -> object:
    name = "exact_local_video_materializer_v1_test_fixture"
    spec = importlib.util.spec_from_file_location(name, EXACT_VIDEO_MATERIALIZER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load exact video materializer fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exact_video_runtime = _load_exact_video_materializer_runtime()


class ExactLocalVideoMaterializerTests(unittest.TestCase):
    def test_unrelated_preloaded_tools_namespace_is_replaced_by_exact_sources(
        self,
    ) -> None:
        original = {
            name: value
            for name, value in tuple(sys.modules.items())
            if name == "tools" or name.startswith("tools.")
        }
        for name in original:
            sys.modules.pop(name, None)
        poison = types.ModuleType("tools")
        poison.materialize_vae = object()
        sys.modules["tools"] = poison
        path_snapshot = list(sys.path)
        try:
            materializer = (
                exact_video_runtime.install_exact_local_video_materializer()
            )
            package = sys.modules["tools"]
            builder = sys.modules["tools.build_renderer_dataset"]
            self.assertIsNot(package, poison)
            self.assertEqual(tuple(package.__path__), ())
            self.assertEqual(
                tuple(package.__spec__.submodule_search_locations), ()
            )
            self.assertIs(package.materialize_vae, materializer)
            self.assertIs(package.build_renderer_dataset, builder)
            self.assertIs(materializer.raw_builder, builder)
            self.assertEqual(
                Path(materializer.__file__).resolve(strict=True),
                METHOD_ROOT / "tools/materialize_vae.py",
            )
            self.assertEqual(
                Path(builder.__file__).resolve(strict=True),
                METHOD_ROOT / "tools/build_renderer_dataset.py",
            )
            imported: dict[str, object] = {}
            exec("from tools import materialize_vae", imported)
            self.assertIs(imported["materialize_vae"], materializer)
            self.assertIs(
                exact_video_runtime.install_exact_local_video_materializer(),
                materializer,
            )
            self.assertEqual(sys.path, path_snapshot)
        finally:
            for name in tuple(sys.modules):
                if name == "tools" or name.startswith("tools."):
                    sys.modules.pop(name, None)
            sys.modules.update(original)

    def test_materializer_source_pin_mismatch_fails_closed(self) -> None:
        with self.assertRaises(
            exact_video_runtime.ExactLocalVideoMaterializerError
        ):
            exact_video_runtime._read_exact_source(
                METHOD_ROOT / "tools/materialize_vae.py",
                "0" * 64,
                label="test materializer",
            )

    def test_infer_lora_has_no_ambiguous_materializer_import(self) -> None:
        source = INFER_LORA_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        ambiguous = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "tools"
            and any(alias.name == "materialize_vae" for alias in node.names)
        ]
        self.assertEqual(ambiguous, [])
        self.assertEqual(
            source.count("install_exact_local_video_materializer()"), 2
        )
        self.assertIn(
            '!= METHOD_ROOT / "exact_local_video_materializer_v1.py"', source
        )
        self.assertIn("import escaped method root", source)


class BlockParserTests(unittest.TestCase):
    def test_inclusive_range(self) -> None:
        self.assertEqual(runner._parse_blocks("4-9"), tuple(range(4, 10)))

    def test_sparse_increasing_list(self) -> None:
        self.assertEqual(runner._parse_blocks("1,3,7,29"), (1, 3, 7, 29))

    def test_sparse_list_rejects_duplicates(self) -> None:
        with self.assertRaises(runner.AnchorEventInferenceError):
            runner._parse_blocks("1,3,3")


class TrainedAttentionCheckpointTests(unittest.TestCase):
    @staticmethod
    def _fixture(
        directory: str,
        *,
        objective: str = "real_source_routed_teacher_delta",
        route: str = "cross_sparse",
        complete: bool = True,
    ) -> tuple[Path, dict[str, object], dict[str, object]]:
        root = Path(directory) / "checkpoint-00000008"
        adapter = root / "adapter"
        adapter.mkdir(parents=True)
        config_path = adapter / "adapter_config.json"
        model_path = adapter / "adapter_model.safetensors"
        config_path.write_text("{}\n", encoding="ascii")
        model_path.write_bytes(b"fixture")
        model_sha = hashlib.sha256(b"fixture").hexdigest()
        receipt = {
            "schema_version": "bernini-online-anchor-attention-training-receipt-v2",
            "complete": complete,
            "global_step": 8,
            "adapter_model_sha256": model_sha,
            "training_contract": {
                "full_attention_lora_enabled": True,
                "lora_rank": 256,
                "lora_alpha": 256,
                "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
                "lora_target_module_count": 240,
                "lora_target_modules_sha256": "0" * 64,
                "training_objective": objective,
                "route_operator": route,
            },
        }
        receipt_path = root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
        expected = {
            "expected_global_step": 8,
            "expected_training_objective": objective,
            "expected_route_operator": route,
            "expected_adapter_model_sha256": model_sha,
            "expected_receipt_sha256": runner.file_sha256(receipt_path),
        }
        return root, receipt, expected

    def test_new_routed_teacher_objective_is_bound_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, receipt, expected = self._fixture(directory)
            loaded = runner._attention_lora_checkpoint(str(root), **expected)
            self.assertEqual(loaded["schema_version"], receipt["schema_version"])
            self.assertEqual(loaded["global_step"], 8)
            self.assertEqual(
                loaded["training_objective"], "real_source_routed_teacher_delta"
            )
            self.assertEqual(loaded["binding"]["receipt_sha256"], expected["expected_receipt_sha256"])
            self.assertRegex(loaded["binding_sha256"], r"^[0-9a-f]{64}$")

    def test_paired_delta_objective_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _, expected = self._fixture(
                directory, objective="paired_delta_fm"
            )
            loaded = runner._attention_lora_checkpoint(str(root), **expected)
            self.assertEqual(loaded["training_objective"], "paired_delta_fm")

    def test_v3_target_owned_qk_checkpoint_has_a_separate_strict_abi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            route = "self_target_owned_temporal_kernel_v14r2"
            root, receipt, expected = self._fixture(
                directory,
                objective="real_source_target_owned_routed_teacher_delta_v14r2",
                route=route,
            )
            receipt["schema_version"] = (
                "bernini-online-anchor-attention-training-receipt-v3"
            )
            receipt["adapter_config_sha256"] = runner.file_sha256(
                root / "adapter/adapter_config.json"
            )
            expected["expected_adapter_config_sha256"] = receipt[
                "adapter_config_sha256"
            ]
            receipt["gradient_coverage"] = {
                "tensor_count": 480,
                "nonzero_tensor_count": 480,
            }
            full_sides = {
                "lora_A": {
                    "nonzero_tensor_count": 240,
                    "epsilon_active_tensor_count": 240,
                },
                "lora_B": {
                    "nonzero_tensor_count": 240,
                    "epsilon_active_tensor_count": 240,
                },
            }
            receipt["component_gradient_probes"] = {
                "action_objective": {
                    "tensor_count": 480,
                    "nonzero_tensor_count": 480,
                    "epsilon_active_tensor_count": 480,
                    "l2_norm_fp64": 2.0,
                    "adapter_sides": full_sides,
                },
                "raw_source_caption_trajectory_replay": {
                    "tensor_count": 480,
                    "nonzero_tensor_count": 480,
                    "epsilon_active_tensor_count": 480,
                    "l2_norm_fp64": 1.0,
                    "adapter_sides": full_sides,
                },
                "interaction": {
                    "action_l2_norm_fp64": 2.0,
                    "raw_replay_l2_norm_fp64": 1.0,
                    "combined_l2_norm_fp64": 2.0,
                    "action_gradient_dot_combined_gradient_fp64": 4.0,
                    "action_alignment_ratio": 1.0,
                    "action_replay_cosine": 0.2,
                    "weighted_replay_gradient_fraction": 0.0,
                    "effective_replay_scale": 0.0,
                    "correction_ratio_q": 0.0,
                    "replay_projection_applied": False,
                    "replay_combine_mode": "action_only",
                },
            }
            receipt["last_loss"] = None
            receipt[
                "last_reporting_scalar_is_not_a_joint_backpropagated_objective"
            ] = True
            receipt["last_objective_components"] = {
                "base_replay_scale": 0.025,
                "effective_replay_scale": 0.0,
                "effective_source_replay_scalar_for_reporting": 0.0,
            }
            receipt["memory_gate"] = {
                "capture_phase": (
                    "after_two_real_component_backwards_before_actual_update_audit_clones"
                ),
                "actual_update_audit_allocations_excluded": True,
                "true_training_tensors_only": True,
                "dummy_or_padding_allocations": False,
                "passed": True,
                "minimum_reserved_fraction": 0.75,
                "per_rank": [
                    {"rank": rank, "reserved_fraction": 0.75}
                    for rank in range(4)
                ],
            }
            receipt["actual_optimizer_update_probe"] = {
                "schema_version": "bernini-actual-optimizer-update-probe-v1",
                "step": 8,
                "replay_combine_mode": "action_only",
                "gradient_scope": (
                    "separately_allreduced_global_action_and_raw_replay"
                ),
                "optimizer_semantics_observed_not_modified": True,
                "parameter_snapshot_native_dtype": True,
                "tensor_count": 480,
                "parameter_element_count": 188743680,
                "changed_tensor_count": 480,
                "changed_element_count": 1,
                "delta_theta_l2_norm_fp64": 0.01,
                "action_descent_required": True,
                "action_descent_passed": True,
                "action_descent_fp64": 0.01,
                "source_descent_required": False,
                "source_descent_passed": False,
            }
            receipt["anchor_cache"] = {
                "capture_count": 88,
                "replay_count": 176,
                "qk_only_capture_count": 88,
                "qk_only_replay_count": 176,
                "qk_only_cached_fields": ["query", "key"],
                "pending_entries": 0,
            }
            receipt["training_contract"].update(
                {
                    "route_transport": (
                        "self_target_owned_temporal_kernel_attn_output_v14r2"
                    ),
                    "target_owned_qk_route_v14r2": True,
                    "anchor_donor_cached_fields": ["query", "key"],
                    "anchor_donor_value_cached_or_used_by_route": False,
                    "anchor_donor_hidden_or_attention_output_cached_or_used_by_route": False,
                    "anchor_donor_rgb_latent_or_absolute_spatial_coordinate_used_by_route": False,
                    "anchor_to_target_appearance_correspondence_used": False,
                    "anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel": True,
                    "anchor_qk_phase0_only_difference_produces_zero_route": True,
                    "real_source_variant_schedule": "complete_real_source",
                    "source_variant_argument": "not_applicable",
                    "micro_semantics": "different_seed_and_cross_appearance_donor",
                    "anchor_route_replay_uses_per_capture": 2,
                    "teacher_delta_mode": "raw",
                    "source_reconstruction_weight": None,
                    "source_reconstruction_weight_argument": 0.025,
                    "base_replay_scale": 0.025,
                    "effective_replay_scale": 0.0,
                    "replay_combine_mode": "action_only",
                    "routed_teacher_mode": "same_action_route_only",
                    "student_route_off_branch_stop_gradient": True,
                    "action_objective_backpropagates_only_routed_student_query": True,
                    "routed_teacher_cross_caption_source_branch": False,
                    "true_training_memory_fraction_strictly_above_half": True,
                    "training_memory_gate_capture_phase": (
                        "after_two_real_component_backwards_before_actual_update_audit_clones"
                    ),
                    "actual_update_audit_allocations_excluded_from_training_memory_gate": True,
                }
            )
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            loaded = runner._attention_lora_checkpoint(str(root), **expected)
            self.assertEqual(
                loaded["required_decode_transport"],
                "self_target_owned_temporal_kernel_attn_output_v14r2",
            )

            receipt["schema_version"] = (
                "bernini-online-anchor-attention-training-receipt-v2"
            )
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            receipt["schema_version"] = (
                "bernini-online-anchor-attention-training-receipt-v3"
            )
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)

            config_path = root / "adapter/adapter_config.json"
            config_path.write_text('{"lora_alpha":1}\n', encoding="ascii")
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            config_path.write_text("{}\n", encoding="ascii")

            expected["expected_adapter_config_sha256"] = "f" * 64
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            expected["expected_adapter_config_sha256"] = receipt[
                "adapter_config_sha256"
            ]

            receipt["component_gradient_probes"]["interaction"][
                "correction_ratio_q"
            ] = 0.1
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            receipt["component_gradient_probes"]["interaction"][
                "correction_ratio_q"
            ] = 0.0

            receipt["component_gradient_probes"]["interaction"][
                "replay_combine_mode"
            ] = "norm_balanced_025"
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            receipt["component_gradient_probes"]["interaction"][
                "replay_combine_mode"
            ] = "action_only"

            receipt["component_gradient_probes"]["interaction"][
                "replay_projection_applied"
            ] = True
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            receipt["component_gradient_probes"]["interaction"][
                "replay_projection_applied"
            ] = False

            receipt["memory_gate"]["actual_update_audit_allocations_excluded"] = False
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            receipt["memory_gate"]["actual_update_audit_allocations_excluded"] = True

            receipt["actual_optimizer_update_probe"][
                "action_descent_passed"
            ] = False
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)
            receipt["actual_optimizer_update_probe"][
                "action_descent_passed"
            ] = True

            receipt["training_contract"]["anchor_donor_cached_fields"] = [
                "query", "key", "value"
            ]
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)

    def test_v3_schema_rejects_a_legacy_objective_and_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, receipt, expected = self._fixture(directory)
            receipt["schema_version"] = (
                "bernini-online-anchor-attention-training-receipt-v3"
            )
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="ascii")
            expected["expected_receipt_sha256"] = runner.file_sha256(receipt_path)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)

    def test_incomplete_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _, expected = self._fixture(directory, complete=False)
            with self.assertRaises(runner.AnchorEventInferenceError):
                runner._attention_lora_checkpoint(str(root), **expected)

    def test_step_objective_route_and_sha_expectations_are_exact(self) -> None:
        mutation_cases = (
            {"expected_global_step": 4},
            {"expected_training_objective": "paired_delta_fm"},
            {"expected_route_operator": "self_temporal_kernel"},
            {"expected_adapter_model_sha256": "f" * 64},
            {"expected_receipt_sha256": "e" * 64},
        )
        for mutation in mutation_cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root, _, expected = self._fixture(directory)
                expected.update(mutation)
                with self.assertRaises(runner.AnchorEventInferenceError):
                    runner._attention_lora_checkpoint(str(root), **expected)

    def test_expectations_without_checkpoint_are_rejected(self) -> None:
        with self.assertRaises(runner.AnchorEventInferenceError):
            runner._attention_lora_checkpoint("", expected_global_step=8)

    def test_trained_route_off_control_is_explicit_and_same_checkpoint_only(self) -> None:
        checkpoint = {"global_step": 32}
        self.assertFalse(
            runner._trained_route_off_control(
                trained_attention=checkpoint,
                transport_steps=40,
                explicitly_allowed=False,
            )
        )
        with self.assertRaises(runner.AnchorEventInferenceError):
            runner._trained_route_off_control(
                trained_attention=checkpoint,
                transport_steps=0,
                explicitly_allowed=False,
            )
        self.assertTrue(
            runner._trained_route_off_control(
                trained_attention=checkpoint,
                transport_steps=0,
                explicitly_allowed=True,
            )
        )
        for trained_attention, steps in ((None, 0), (checkpoint, 40)):
            with self.subTest(trained_attention=trained_attention, steps=steps):
                with self.assertRaises(runner.AnchorEventInferenceError):
                    runner._trained_route_off_control(
                        trained_attention=trained_attention,
                        transport_steps=steps,
                        explicitly_allowed=True,
                    )


class _Tensor:
    def __init__(self, *, shape: tuple[int, ...], dtype: object, value: object) -> None:
        self.shape = shape
        self.dtype = dtype
        self.value = value
        self.to_calls: list[object] = []

    def to(self, device: object) -> "_Tensor":
        self.to_calls.append(device)
        return self

    def contiguous(self) -> "_Tensor":
        return self


class _Cuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _Torch:
    bfloat16 = object()

    def __init__(self) -> None:
        self.cuda = _Cuda()
        self.empty_calls: list[tuple[tuple[int, ...], object, object]] = []

    @staticmethod
    def inference_mode() -> object:
        return nullcontext()

    def empty(
        self, shape: tuple[int, ...], *, dtype: object, device: object
    ) -> _Tensor:
        self.empty_calls.append((tuple(shape), dtype, device))
        return _Tensor(shape=tuple(shape), dtype=dtype, value=None)


class _CollectiveGroup:
    def __init__(self, world_size: int) -> None:
        self.barrier = threading.Barrier(world_size)
        self.condition = threading.Condition()
        self.status: object = None
        self.tensors: dict[int, tuple[tuple[int, ...], object, object]] = {}
        self.retired_ranks: set[int] = set()
        self.rank_zero_to_retired_snapshot: set[int] | None = None


class _Dist:
    def __init__(self, group: _CollectiveGroup, rank: int) -> None:
        self.group = group
        self.rank = rank
        self.tensor_index = 0

    def barrier(self) -> None:
        self.group.barrier.wait(timeout=5.0)

    def broadcast_object_list(self, values: list[object], *, src: int) -> None:
        if src != 0:
            raise AssertionError("unexpected object broadcast source")
        with self.group.condition:
            if self.rank == 0:
                self.group.status = dict(values[0])  # type: ignore[arg-type]
                self.group.condition.notify_all()
            else:
                if not self.group.condition.wait_for(
                    lambda: self.group.status is not None, timeout=5.0
                ):
                    raise AssertionError("rank-zero status broadcast timed out")
                values[0] = dict(self.group.status)  # type: ignore[arg-type]

    def broadcast(self, value: _Tensor, *, src: int) -> None:
        if src != 0:
            raise AssertionError("unexpected tensor broadcast source")
        index = self.tensor_index
        self.tensor_index += 1
        with self.group.condition:
            if self.rank == 0:
                self.group.tensors[index] = (value.shape, value.dtype, value.value)
                self.group.condition.notify_all()
            else:
                if not self.group.condition.wait_for(
                    lambda: index in self.group.tensors, timeout=5.0
                ):
                    raise AssertionError(f"tensor broadcast {index} timed out")
                shape, dtype, payload = self.group.tensors[index]
                if value.shape != shape or value.dtype is not dtype:
                    raise AssertionError("receiver allocation contract differs")
                value.value = payload


class _Encoder:
    def __init__(self, group: _CollectiveGroup, rank: int) -> None:
        self.group = group
        self.rank = rank
        self.to_calls: list[object] = []

    def to(self, device: object) -> "_Encoder":
        with self.group.condition:
            if self.rank != 0:
                raise AssertionError("nonzero rank attempted to load T5 on device")
            self.group.rank_zero_to_retired_snapshot = set(self.group.retired_ranks)
        self.to_calls.append(device)
        return self


class _Model:
    def __init__(
        self,
        group: _CollectiveGroup,
        rank: int,
        torch_module: _Torch,
        *,
        invalid_prompt: str | None = None,
    ) -> None:
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "torch_module", torch_module)
        object.__setattr__(self, "invalid_prompt", invalid_prompt)
        object.__setattr__(self, "encode_calls", [])
        object.__setattr__(self, "t5_text_encoder", _Encoder(group, rank))

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name == "t5_text_encoder" and value is None:
            with self.group.condition:
                self.group.retired_ranks.add(self.rank)
                self.group.condition.notify_all()

    def encode_prompt(self, input_ids: _Tensor, attention_mask: _Tensor) -> _Tensor:
        if self.rank != 0:
            raise AssertionError("nonzero rank attempted prompt encoding")
        if attention_mask.value != input_ids.value:
            raise AssertionError("prompt id/mask pair differs")
        self.encode_calls.append(input_ids.value)
        shape = (
            (1, 511, 4096)
            if input_ids.value == self.invalid_prompt
            else runner.PROMPT_EMBEDDING_SHAPE
        )
        return _Tensor(
            shape=shape,
            dtype=self.torch_module.bfloat16,
            value=input_ids.value,
        )


def _token_bank(torch_module: _Torch) -> dict[str, tuple[_Tensor, _Tensor]]:
    return {
        name: (
            _Tensor(shape=(1, 512), dtype="ids", value=name),
            _Tensor(shape=(1, 512), dtype="mask", value=name),
        )
        for name in runner.PROMPT_BANK_NAMES
    }


class RankZeroPromptBankTests(unittest.TestCase):
    def _run_world4(
        self, *, invalid_prompt: str | None = None
    ) -> tuple[
        _CollectiveGroup,
        list[_Torch],
        list[_Model],
        list[dict[str, _Tensor] | None],
        list[BaseException | None],
    ]:
        group = _CollectiveGroup(runner.ULYSSES_SIZE)
        torches = [_Torch() for _ in range(runner.ULYSSES_SIZE)]
        models = [
            _Model(group, rank, torches[rank], invalid_prompt=invalid_prompt)
            for rank in range(runner.ULYSSES_SIZE)
        ]
        outputs: list[dict[str, _Tensor] | None] = [None] * runner.ULYSSES_SIZE
        errors: list[BaseException | None] = [None] * runner.ULYSSES_SIZE

        def run_rank(rank: int) -> None:
            try:
                outputs[rank] = runner._rank_zero_prompt_bank(
                    models[rank],
                    tokenized_prompts=_token_bank(torches[rank]) if rank == 0 else None,
                    distributed_rank=rank,
                    device=f"cuda:{rank}",
                    dist=_Dist(group, rank),
                    torch_module=torches[rank],
                )
            except BaseException as error:  # make collective test failures observable
                errors[rank] = error

        threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
            self.assertFalse(thread.is_alive(), "WORLD4 prompt collective deadlocked")
        return group, torches, models, outputs, errors

    def test_only_rank_zero_moves_and_encodes_t5_then_all_ranks_receive_exact_bank(self) -> None:
        group, torches, models, outputs, errors = self._run_world4()
        self.assertEqual(errors, [None, None, None, None])
        self.assertEqual(group.rank_zero_to_retired_snapshot, {1, 2, 3})
        self.assertEqual(
            models[0].encode_calls,
            list(runner.PROMPT_BANK_NAMES),
        )
        for rank in range(1, runner.ULYSSES_SIZE):
            self.assertEqual(models[rank].encode_calls, [])
        for model in models:
            self.assertIsNone(model.t5_text_encoder)
        self.assertEqual([len(torch.empty_calls) for torch in torches], [0, 7, 7, 7])
        self.assertEqual(
            [torch.cuda.empty_cache_calls for torch in torches], [1, 1, 1, 1]
        )
        for output in outputs:
            self.assertIsNotNone(output)
            assert output is not None
            self.assertEqual(tuple(output), runner.PROMPT_BANK_NAMES)
            self.assertEqual(
                [output[name].value for name in runner.PROMPT_BANK_NAMES],
                list(runner.PROMPT_BANK_NAMES),
            )
            self.assertTrue(
                all(
                    tensor.shape == runner.PROMPT_EMBEDDING_SHAPE
                    and tensor.dtype is torches[0].bfloat16
                    for tensor in output.values()
                )
            )

    def test_rank_zero_shape_failure_is_synchronized_before_tensor_collectives(self) -> None:
        group, _, models, outputs, errors = self._run_world4(invalid_prompt="anchor")
        self.assertEqual(group.tensors, {})
        self.assertEqual(outputs, [None, None, None, None])
        for error in errors:
            self.assertIsInstance(error, runner.AnchorEventInferenceError)
            self.assertIn("rank-zero prompt encoding failed", str(error))
        for model in models:
            self.assertIsNone(model.t5_text_encoder)


class ConstructorT5BypassTests(unittest.TestCase):
    class _T5Base:
        real_calls = 0

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            cls.real_calls += 1
            return ("real", args, kwargs)

    class _T5(_T5Base):
        pass

    def setUp(self) -> None:
        self._T5.real_calls = 0
        if "from_pretrained" in vars(self._T5):
            delattr(self._T5, "from_pretrained")

    def test_nonzero_rank_gets_one_placeholder_without_calling_real_loader(self) -> None:
        checkpoint = Path("/absolute/checkpoint")
        placeholder = object()
        with runner._nonzero_rank_t5_load_bypass(
            distributed_rank=2,
            t5_encoder_class=self._T5,
            expected_checkpoint=checkpoint,
            expected_dtype="bf16",
            placeholder_factory=lambda: placeholder,
        ) as audit:
            value = self._T5.from_pretrained(
                str(checkpoint), subfolder="text_encoder", torch_dtype="bf16"
            )
            self.assertIs(value, placeholder)
            self.assertEqual(self._T5.real_calls, 0)
        self.assertEqual(audit["call_count"], 1)
        self.assertFalse(audit["real_t5_load"])
        self.assertNotIn("from_pretrained", vars(self._T5))
        restored = self._T5.from_pretrained("outside-scope")
        self.assertEqual(restored[0], "real")
        self.assertEqual(self._T5.real_calls, 1)

    def test_rank_zero_never_patches_real_loader(self) -> None:
        checkpoint = Path("/absolute/checkpoint")
        with runner._nonzero_rank_t5_load_bypass(
            distributed_rank=0,
            t5_encoder_class=self._T5,
            expected_checkpoint=checkpoint,
            expected_dtype="bf16",
            placeholder_factory=lambda: object(),
        ) as audit:
            value = self._T5.from_pretrained(
                str(checkpoint), subfolder="text_encoder", torch_dtype="bf16"
            )
        self.assertEqual(value[0], "real")
        self.assertTrue(audit["real_t5_load"])
        self.assertEqual(audit["call_count"], 0)

    def test_nonzero_rank_rejects_constructor_abi_drift_and_restores_class(self) -> None:
        checkpoint = Path("/absolute/checkpoint")
        with self.assertRaises(runner.AnchorEventInferenceError):
            with runner._nonzero_rank_t5_load_bypass(
                distributed_rank=3,
                t5_encoder_class=self._T5,
                expected_checkpoint=checkpoint,
                expected_dtype="bf16",
                placeholder_factory=lambda: object(),
            ):
                self._T5.from_pretrained(
                    str(checkpoint),
                    subfolder="wrong",
                    torch_dtype="bf16",
                )
        self.assertNotIn("from_pretrained", vars(self._T5))

    def test_world4_runtime_closure_requires_exactly_one_real_load(self) -> None:
        rows = [
            {
                "rank": rank,
                "real_t5_loaded": rank == 0,
                "bypassed_t5_load": rank != 0,
                "bypass_call_count": 0 if rank == 0 else 1,
                "placeholder_retained": rank != 0,
            }
            for rank in range(4)
        ]
        self.assertIsNone(runner._validate_t5_load_closure(rows))
        hostile = [dict(row) for row in rows]
        hostile[2]["real_t5_loaded"] = True
        with self.assertRaises(runner.AnchorEventInferenceError):
            runner._validate_t5_load_closure(hostile)


class RunnerWiringTests(unittest.TestCase):
    @staticmethod
    def _v14r2_sidecar_fixture(*, route_off: bool):
        transport = "self_target_owned_temporal_kernel_attn_output_v14r2"
        expected = sidecar_validator.Expected(
            video="/tmp/v14r2.mp4",
            checkpoint="/tmp/checkpoint-00000032",
            step=32,
            route="self_target_owned_temporal_kernel_v14r2",
            transport=transport,
            transport_steps=0 if route_off else 40,
            adapter_sha256="a" * 64,
            adapter_config_sha256="b" * 64,
            receipt_sha256="c" * 64,
            source_sha256="d" * 64,
            anchor_sha256="e" * 64,
            preservation_mode="none",
            sga_score_mode="global_source_cosine",
            route_off=route_off,
        )
        binding = {
            "receipt_sha256": expected.receipt_sha256,
            "adapter_config_sha256": expected.adapter_config_sha256,
            "adapter_model_sha256": expected.adapter_sha256,
            "global_step": expected.step,
            "training_objective": sidecar_validator.OBJECTIVE,
            "route_operator": expected.route,
            "required_decode_transport": transport,
        }
        trained = {
            "path": expected.checkpoint,
            "schema_version": sidecar_validator.TRAINING_SCHEMA,
            "global_step": expected.step,
            "training_objective": sidecar_validator.OBJECTIVE,
            "route_operator": expected.route,
            "required_decode_transport": transport,
            "adapter_model_sha256": expected.adapter_sha256,
            "adapter_config_sha256": expected.adapter_config_sha256,
            "receipt_sha256": expected.receipt_sha256,
            "checkpoint_binding": binding,
            "checkpoint_binding_sha256": sidecar_validator._canonical_sha256(binding),
            "expectations_fail_closed": {
                "global_step": expected.step,
                "training_objective": sidecar_validator.OBJECTIVE,
                "route_operator": expected.route,
                "adapter_model_sha256": expected.adapter_sha256,
                "adapter_config_sha256": expected.adapter_config_sha256,
                "receipt_sha256": expected.receipt_sha256,
                "all_validated": True,
            },
            "adapter_kept_unmerged": True,
            "frozen_anchor_calls_use_disable_adapter": True,
            "target_and_source_editor_calls_keep_adapter_enabled": True,
            "adapter_enabled_for_target_source_calls": True,
            "anchor_injection_enabled": not route_off,
            "same_checkpoint_route_off_causal_control": route_off,
        }
        trace = {
            "anchor_model_forwards": 0 if route_off else 104,
            "anchor_native_trajectory_model_forwards": 0,
            "anchor_candidate_cells": 0 if route_off else 52,
            "anchor_active_schedule": [] if route_off else list(range(40)),
            "target_owned_qk_route_v14r2": not route_off,
            "anchor_donor_cached_fields": None if route_off else ["query", "key"],
            "anchor_donor_value_hidden_output_or_coordinate_used": (
                None if route_off else False
            ),
            "anchor_to_target_appearance_correspondence_used": (
                None if route_off else False
            ),
            "anchor_temporal_attention_kernel_contrast": not route_off,
            "anchor_temporal_kernel_applied_to_target_value_only": not route_off,
            "anchor_route_shared_by_target_negative_and_condition": not route_off,
            "anchor_route_target_conditional_only": False,
            "initial_latent_phase_clamped_after_every_update": True,
            "anchor_value_stream_copied": False,
            "source_value_stream_retained": True,
            "anchor_present_in_every_active_target_candidate": not route_off,
            "anchor_present_after_active_interval": False,
            "attention_cache": {
                "capture_count": 0 if route_off else 2288,
                "qk_only_capture_count": 0 if route_off else 2288,
                "replay_count": 0 if route_off else 4576,
                "qk_only_replay_count": 0 if route_off else 4576,
                "pending_entries": 0,
                "qk_only_cached_fields": ["query", "key"],
            },
        }
        receipt = {
            "schema_version": sidecar_validator.SCHEMA,
            "complete": True,
            "loaded_trained_attention_checkpoint": True,
            "trained_attention_checkpoint": trained,
            "causal_control": {
                "enabled": route_off,
                "kind": "same_trained_checkpoint_route_off" if route_off else None,
                "explicit_opt_in": route_off,
                "trained_adapter_loaded": True,
                "adapter_enabled_for_target_source_calls": True,
                "anchor_injection_enabled": not route_off,
                "transport_steps": expected.transport_steps,
            },
            "source": {
                "sha256": expected.source_sha256,
                "role": "clean_edit_state_identity_appearance_scene_authority",
            },
            "pure_t2v_anchor": {
                "sha256": expected.anchor_sha256,
                "active_solver_steps": expected.transport_steps,
                "model_forward_at_every_active_solver_step_and_candidate": (
                    not route_off
                ),
            },
            "pure_t2v_anchor_bank": [{"sha256": expected.anchor_sha256}],
            "mechanism": {
                "arm": "AQK_SGA5",
                "transport": transport,
                "transport_strength": 0.25,
                "transport_steps": expected.transport_steps,
                "initial_phase_clamp": True,
                "field_guidance": "raw_cfg",
                "field_model": "first_phase_caption_i2v",
                "source_cfg_scale": 4.5,
                "target_cfg_scale": 4.5,
                "sga_temperature": 0.01,
                "early_candidate_count": 5,
                "initial_noise_proposal_mode": "keyed_only",
                "anchor_state_mode": "clean_noised",
                "anchor_cfg_scope": "shared",
                "anchor_contrast_mode": "caption_noop_same_video",
                "anchor_sigma_cap": 1.0,
                "preservation_mode": expected.preservation_mode,
                "preservation_keep_fraction": 0.2,
                "preservation_outside_scale": 0.0,
                "preservation_dilation": 1,
                "preservation_residual_fraction": 0.0,
                "preservation_object_identity_strength": 0.0,
                "preservation_start_step": 0,
                "preservation_ramp_steps": 1,
                "sga_score_mode": expected.sga_score_mode,
                "anchor_candidate_mode": "single_shared",
                "anchor_bank_size": 1,
                "anchor_spatial_alignment": "none",
                "selected_blocks": sidecar_validator.BLOCKS,
                "pure_t2v_anchor_online_block_transport_enabled": not route_off,
                "pure_t2v_anchor_online_velocity_transport_enabled": False,
                "pure_t2v_anchor_values_or_pixels_copied_to_output": False,
                "decode_audit_contract": {
                    "transport_steps": expected.transport_steps,
                    "anchor_state_mode": "clean_noised",
                    "anchor_cfg_scope": "shared",
                    "source_cfg_scale": 4.5,
                    "target_cfg_scale": 4.5,
                    "source_and_target_cfg_equal": True,
                    "pure_t2v_teacher_adapter_policy": "disable_loaded_editor_adapter",
                    "target_source_editor_adapter_policy": "loaded_adapter_enabled",
                    "trained_route_off_control_explicitly_allowed": route_off,
                    "same_checkpoint_route_off_causal_control": route_off,
                    "anchor_injection_enabled": not route_off,
                },
                "trace": trace,
            },
            "output": {
                "path": expected.video,
                "sha256": "f" * 64,
                "frames": 81,
                "fps": 25,
            },
            "freeze_before": {
                "schema_version": sidecar_validator.EDITOR_FREEZE_SCHEMA,
                "base_and_adapter_frozen": True,
                "base_frozen": True,
                "trainable_parameter_tensors": 0,
                "trainable_parameter_elements": 0,
                "peft_model_authenticated": True,
                "adapter_disable_context_available": True,
                "adapter_disable_context_reversible": True,
                "adapter_disable_context_refreezes_parameters": True,
                "adapter_kept_unmerged": True,
                "adapter_enabled_for_editor_calls": True,
                "pure_t2v_teacher_policy": "temporary_disable_adapter_context",
                "adapter_config_names": ["default"],
                "lora_layer_count": sidecar_validator.EDITOR_LORA_LAYERS,
                "lora_parameter_tensors": (
                    sidecar_validator.EDITOR_LORA_PARAMETER_TENSORS
                ),
                "lora_parameter_elements": 188743680,
                "lora_layer_inventory_sha256": "1" * 64,
                "lora_parameter_inventory_sha256": "2" * 64,
            },
        }
        receipt["freeze_after"] = dict(receipt["freeze_before"])
        return receipt, expected

    def test_v14r2_sidecar_semantics_and_route_activity_fail_closed(self) -> None:
        for route_off in (False, True):
            receipt, expected = self._v14r2_sidecar_fixture(route_off=route_off)
            sidecar_validator.validate_receipt(receipt, expected)

        receipt, expected = self._v14r2_sidecar_fixture(route_off=False)
        mutations = {
            "wrong_schema": ("schema_version", "v46"),
            "wrong_source": ("source.sha256", "0" * 64),
            "wrong_state": ("mechanism.anchor_state_mode", "native_t2v_trajectory"),
            "wrong_cfg": ("mechanism.source_cfg_scale", 8.5),
            "wrong_noop": ("mechanism.anchor_contrast_mode", "dynamic_static_same_caption"),
            "wrong_preservation": ("mechanism.preservation_mode", "source_motion_support"),
            "missing_capture": ("mechanism.trace.attention_cache.capture_count", 0),
            "missing_replay": ("mechanism.trace.attention_cache.replay_count", 0),
            "route_not_shared": (
                "mechanism.trace.anchor_route_shared_by_target_negative_and_condition",
                False,
            ),
            "wrong_config_binding": (
                "trained_attention_checkpoint.checkpoint_binding.adapter_config_sha256",
                "0" * 64,
            ),
            "trainable_editor": ("freeze_before.trainable_parameter_tensors", 1),
            "disabled_editor": ("freeze_before.adapter_enabled_for_editor_calls", False),
            "merged_editor": ("freeze_before.adapter_kept_unmerged", False),
            "wrong_adapter_inventory": (
                "freeze_after.lora_parameter_inventory_sha256",
                "3" * 64,
            ),
        }
        for label, (path, value) in mutations.items():
            hostile = json.loads(json.dumps(receipt))
            target = hostile
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            with self.subTest(label=label), self.assertRaises(
                sidecar_validator.V14R2DecodeValidationError
            ):
                sidecar_validator.validate_receipt(hostile, expected)

        routeoff, routeoff_expected = self._v14r2_sidecar_fixture(route_off=True)
        routeoff["mechanism"]["trace"]["anchor_model_forwards"] = 1
        with self.assertRaises(sidecar_validator.V14R2DecodeValidationError):
            sidecar_validator.validate_receipt(routeoff, routeoff_expected)

    def test_every_teacher_helper_call_disables_adapter_but_student_field_does_not(self) -> None:
        controller_path = METHOD_ROOT / "anchor_sga_anc_controller.py"
        tree = ast.parse(controller_path.read_text(encoding="utf-8"))
        teacher_helpers = {
            "_capture_anchor_qk",
            "_capture_anchor_cross_attention",
            "_guided_source_free_apg_velocity",
        }
        source_free_without_adapter: list[str] = []
        for function in (
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ):
            if function.name in teacher_helpers:
                names = [item.arg for item in function.args.kwonlyargs]
                adapter_index = names.index("adapter_controller")
                self.assertIsNone(function.args.kw_defaults[adapter_index])
            for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
                name = call.func.id if isinstance(call.func, ast.Name) else ""
                keywords = {item.arg for item in call.keywords}
                if name in teacher_helpers:
                    self.assertIn("adapter_controller", keywords)
                if name == "_predict_source_free_velocity" and "adapter_controller" not in keywords:
                    source_free_without_adapter.append(function.name)
        self.assertEqual(source_free_without_adapter, ["_predict_field_velocity"])

    def test_trained_editor_remains_reversible_for_frozen_anchor_calls(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        controller_source = (METHOD_ROOT / "anchor_sga_anc_controller.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("merge_and_unload", source)
        self.assertIn('getattr(peft_model, "disable_adapter", None)', source)
        self.assertIn('"adapter_kept_unmerged": True', source)
        self.assertIn('"frozen_anchor_calls_use_disable_adapter": True', source)
        self.assertIn("def _frozen_anchor_adapter_context", controller_source)
        self.assertGreaterEqual(
            controller_source.count("adapter_controller=renderer_or_diffusion"), 10
        )
        self.assertIn(
            'trace.get("anchor_teacher_disable_adapter_context_available") is not True',
            source,
        )

    def test_decode_checkpoint_provenance_and_e04_noop_are_auditable(self) -> None:
        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        decode = DECODE_SCRIPT_PATH.read_text(encoding="utf-8")
        bridge = BRIDGE_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('transport_steps="${ONLINE_ANCHOR_DECODE_TRANSPORT_STEPS:-40}"', decode)
        self.assertIn('cfg_scope="${ONLINE_ANCHOR_DECODE_CFG_SCOPE:-shared}"', decode)
        self.assertIn('anchor_state_mode="${ONLINE_ANCHOR_DECODE_STATE_MODE:-clean_noised}"', decode)
        self.assertIn('ONLINE_ANCHOR_DECODE_DEV_OVERRIDE', decode)
        self.assertIn('ONLINE_ANCHOR_DECODE_BRIDGE_RUNNER', decode)
        self.assertNotIn("no hand enters", decode)
        self.assertIn("same light-skinned hand remains present", decode)
        for field in (
            "TRAINED_ATTENTION_EXPECTED_STEP",
            "TRAINED_ATTENTION_EXPECTED_OBJECTIVE",
            "TRAINED_ATTENTION_EXPECTED_ROUTE_OPERATOR",
            "TRAINED_ATTENTION_EXPECTED_ADAPTER_SHA256",
            "TRAINED_ATTENTION_EXPECTED_ADAPTER_CONFIG_SHA256",
            "TRAINED_ATTENTION_EXPECTED_RECEIPT_SHA256",
        ):
            self.assertIn(field, decode)
            self.assertIn(field, bridge)
        for option in (
            "--expected-trained-attention-step",
            "--expected-trained-attention-objective",
            "--expected-trained-attention-route-operator",
            "--expected-trained-attention-adapter-sha256",
            "--expected-trained-attention-adapter-config-sha256",
            "--expected-trained-attention-receipt-sha256",
        ):
            self.assertIn(option, bridge)
            self.assertIn(option, runner_source)
        self.assertIn('"checkpoint_binding_sha256"', runner_source)
        self.assertIn('"decode_audit_contract"', runner_source)
        self.assertIn("--allow-trained-route-off-control", runner_source)
        self.assertIn('"same_checkpoint_route_off_causal_control"', runner_source)
        self.assertIn('"adapter_enabled_for_target_source_calls"', runner_source)
        self.assertIn('"anchor_injection_enabled"', runner_source)
        self.assertIn("ALLOW_TRAINED_ROUTE_OFF_CONTROL", decode)
        self.assertIn("ALLOW_TRAINED_ROUTE_OFF_CONTROL", bridge)
        v14r2_import_branch = bridge[
            bridge.index(
                'if [ "$trained_attention_expected_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]'
            ) : bridge.index("else\n  export PYTHONPATH=\"$dev:$runtime\"")
        ]
        self.assertIn('export PYTHONPATH="$dev"', v14r2_import_branch)
        self.assertNotIn("$runtime", v14r2_import_branch)
        self.assertIn("V14R2_IMPORT_MODULES", v14r2_import_branch)
        self.assertIn("module.__file__", v14r2_import_branch)
        self.assertIn("exact_local_video_materializer_v1", v14r2_import_branch)
        self.assertIn("tools/build_renderer_dataset.py", v14r2_import_branch)
        self.assertIn("tools/materialize_vae.py", v14r2_import_branch)
        self.assertIn('poison=types.ModuleType("tools")', v14r2_import_branch)
        self.assertIn(
            "install_exact_local_video_materializer", v14r2_import_branch
        )

    def test_v14_watcher_is_serial_immutable_and_contains_required_controls(self) -> None:
        watcher = V14_WATCHER_PATH.read_text(encoding="utf-8")
        self.assertEqual(watcher.count("srun --jobid="), 1)
        self.assertNotIn("scancel", watcher)
        self.assertIn('for step in 8 32; do', watcher)
        self.assertIn('for event in 0 2 4 7; do', watcher)
        self.assertIn('32 "$event" 0 1 none global_source_cosine', watcher)
        self.assertIn(
            '32 "$event" 40 0 source_motion_support background_plus_anchor_action_002',
            watcher,
        )
        self.assertIn("ONLINE_ANCHOR_DECODE_DEV_OVERRIDE", watcher)
        self.assertIn("ONLINE_ANCHOR_DECODE_BRIDGE_RUNNER", watcher)
        self.assertIn("ONLINE_ANCHOR_EXPECTED_ADAPTER_SHA256", watcher)
        self.assertIn("ONLINE_ANCHOR_EXPECTED_RECEIPT_SHA256", watcher)
        self.assertIn("ONLINE_ANCHOR_ALLOW_TRAINED_ROUTE_OFF_CONTROL", watcher)
        expected_routes = {
            "self_temporal_kernel": "temporal_kernel_contrast_attn_output",
            "self_target_gated_kernel25": "target_gated_hard_kernel_top25_attn_output",
            "self_correspondence_kernel25": (
                "correspondence_gated_hard_kernel_top25_attn_output"
            ),
        }
        for route, transport in expected_routes.items():
            self.assertIn(f"{route}) echo {transport}", watcher)
        for node in (
            "auh7-1b-gpu-233",
            "auh7-1b-gpu-268",
            "auh7-1b-gpu-292",
            "auh7-1b-gpu-315",
        ):
            self.assertEqual(watcher.count(node), 1)
        self.assertIn("find \"$output_dir\"", watcher)

    def test_v14r2_watcher_validates_exact_artifacts_before_skip(self) -> None:
        watcher = V14R2_WATCHER_PATH.read_text(encoding="utf-8")
        self.assertEqual(watcher.count("srun --jobid="), 1)
        self.assertNotIn("scancel", watcher)
        self.assertIn("validate_existing_decode", watcher)
        self.assertIn('test -f "$video"', watcher)
        self.assertIn('test -f "$sidecar"', watcher)
        self.assertIn('video_sha="$(sha256sum -- "$video"', watcher)
        self.assertIn("ffprobe -v error -count_frames", watcher)
        self.assertIn('.[0].nb_read_frames == "81"', watcher)
        self.assertIn('.[0].avg_frame_rate == "25/1"', watcher)
        self.assertNotIn("-name '*.mp4' -print -quit", watcher)
        self.assertIn("fresh v14r2 tag contains an unexpected partial artifact", watcher)
        self.assertIn(
            'real_source_target_owned_routed_teacher_delta_v14r2', watcher
        )
        self.assertIn(
            'source-online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2', watcher
        )
        self.assertIn(
            "online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2.content.json",
            watcher,
        )
        self.assertIn("sameaction_global_actiononly_s8_v14r3_gradgeom", watcher)
        self.assertIn("validate_v14r2_deployment_marker.py", watcher)
        self.assertIn("--role decode", watcher)
        self.assertIn("--min-test-count 144", watcher)
        self.assertIn("--training-marker", watcher)
        self.assertIn("--shared-core methods/bernini_action_editing/anchor_qk_transport.py", watcher)
        for route, transport in runner.V14R2_ROUTE_TO_TRANSPORT.items():
            self.assertIn(route, watcher)
            self.assertIn(transport, watcher)
        for field in (
            ".trained_attention_checkpoint.checkpoint_binding.global_step",
            ".trained_attention_checkpoint.checkpoint_binding.training_objective",
            ".trained_attention_checkpoint.checkpoint_binding.route_operator",
            ".trained_attention_checkpoint.checkpoint_binding.required_decode_transport",
            ".trained_attention_checkpoint.checkpoint_binding.adapter_model_sha256",
            ".trained_attention_checkpoint.checkpoint_binding.adapter_config_sha256",
            ".trained_attention_checkpoint.checkpoint_binding.receipt_sha256",
            ".mechanism.transport_steps",
            ".output.sha256",
        ):
            self.assertIn(field, watcher)
        for arm_binding in (
            "same_action_route_only action_only 4 aborted_preoptimizer_last_valid",
            "same_action_route_only norm_balanced_025 4 aborted_preoptimizer_last_valid",
            "same_action_route_only action_priority_pcgrad_010 8 completed_s8",
        ):
            self.assertIn(arm_binding, watcher)
        launch_tail = watcher[watcher.rindex("launch sameaction_global_actiononly") :]
        self.assertNotIn("sameaction_gate25_halfspace001", launch_tail)
        self.assertNotIn("touch TRAINING_COMPLETE", watcher)
        self.assertIn("validate_aborted_last_valid", watcher)
        self.assertIn("--validate-aborted-last-valid", watcher)
        self.assertIn("diagnostic_decode_only == true", watcher)
        self.assertIn("promotion_authorized == false", watcher)
        self.assertIn("checkpoint_artifact_complete == true", watcher)
        self.assertIn("LASTVALID_S${step}_ABORTED_PREOPT_S$((step + 1))_DIAG", watcher)
        self.assertIn("v14r3_last_valid_checkpoint_abort_authority_v1.json", watcher)
        self.assertIn("tools/build_renderer_dataset.py", watcher)
        self.assertIn("tools/materialize_vae.py", watcher)
        self.assertIn("exact_local_video_materializer_v1.py", watcher)
        self.assertIn("infer_anchor_sga_anc_trained_editor_decode_v1.py", watcher)
        self.assertIn("test_infer_anchor_sga_anc_trained_editor_decode_v1.py", watcher)
        self.assertIn("v14r3d2", watcher)
        self.assertNotIn("v14r3d1", watcher)
        authority = json.loads(
            V14R3_ABORT_AUTHORITY_PATH.read_text(encoding="ascii")
        )
        self.assertEqual(
            authority["schema_version"],
            "bernini-v14r3-last-valid-preoptimizer-abort-authority-v1",
        )
        self.assertTrue(authority["diagnostic_decode_only"])
        self.assertFalse(authority["promotion_authorized"])
        self.assertEqual(authority["requested_max_steps"], 8)
        self.assertEqual(
            set(authority["arms"]),
            {
                "sameaction_global_actiononly_s8_v14r3_gradgeom",
                "sameaction_global_norm025_s8_v14r3_gradgeom",
            },
        )
        for row in authority["arms"].values():
            self.assertFalse(row["training_run_complete"])
            self.assertEqual(
                row["failure"]["failed_attempt_step"],
                row["checkpoint"]["last_completed_optimizer_step"] + 1,
            )
            self.assertEqual(row["failure"]["rank_error_count"], 4)
        self.assertIn(
            ".training_contract.routed_teacher_mode == $teacher_mode", watcher
        )
        self.assertIn(
            ".training_contract.replay_combine_mode == $combine_mode", watcher
        )
        self.assertIn(
            "after_two_real_component_backwards_before_actual_update_audit_clones",
            watcher,
        )
        self.assertIn(
            ".memory_gate.actual_update_audit_allocations_excluded == true",
            watcher,
        )
        self.assertIn("def combine_geometry_ok($mode)", watcher)
        self.assertIn("def actual_optimizer_update_ok($mode; $expected_step)", watcher)
        self.assertIn(
            ".component_gradient_probes.interaction | combine_geometry_ok($combine_mode)",
            watcher,
        )
        self.assertIn(
            "actual_optimizer_update_ok($combine_mode; $step)", watcher
        )
        self.assertNotIn(
            ".component_gradient_probes.interaction.weighted_replay_gradient_fraction >= 0.001",
            watcher,
        )

    def test_main_uses_stable_post_t5_freeze_scope_and_preserves_receipt_schema(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r'SCHEMA_VERSION = "bernini-pure-t2v-anchor-sga-anc-event-canary-v\d+"',
        )
        self.assertIn('"schema_version": SCHEMA_VERSION', source)
        self.assertEqual(source.count("model.t5_text_encoder.to(device)"), 1)
        self.assertNotIn('model.t5_text_encoder.to("cpu")', source)
        self.assertIn("model.t5_text_encoder = None", source)
        self.assertIn("with _nonzero_rank_t5_load_bypass(", source)
        self.assertIn("placeholder_factory=torch.nn.Identity", source)
        self.assertIn("_validate_t5_load_closure(t5_load_rows)", source)
        constructor = source.index("model = BerniniRendererModel(config)")
        prompt_bank = source.index("prompt_bank = _rank_zero_prompt_bank(")
        freeze_before = source.index(
            "freeze_before = source_audit.model_freeze_certificate(model)"
        )
        transformer_load = source.index("model.diff_dec.transformer.to(device)")
        freeze_after = source.index(
            "freeze_after = source_audit.model_freeze_certificate(model)"
        )
        self.assertLess(constructor, prompt_bank)
        self.assertLess(prompt_bank, freeze_before)
        self.assertLess(freeze_before, transformer_load)
        self.assertLess(transformer_load, freeze_after)
        self.assertIn('"freeze_before": freeze_before', source)
        self.assertIn('"freeze_after": freeze_after', source)


if __name__ == "__main__":
    unittest.main()
