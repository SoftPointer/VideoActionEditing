from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import generic_source_carrier_r64_heldout_contract_v1 as contract
import build_generic_source_carrier_r64_heldout_html_v1 as html_builder
import build_generic_source_carrier_r64_heldout_release_v1 as release_builder


RUNNER_SOURCE = (
    METHOD_ROOT / "infer_generic_source_carrier_r64_heldout_v1.py"
).read_text(encoding="utf-8")
HOLDER_SOURCE = (
    METHOD_ROOT / "scripts/auh_generic_source_carrier_r64_heldout_holder_v1.sh"
).read_text(encoding="utf-8")
PREFLIGHT_SOURCE = (
    METHOD_ROOT
    / "tools/preflight_generic_source_carrier_r64_heldout_sources_v1.py"
).read_text(encoding="utf-8")
MATERIALIZER_SOURCE = (
    METHOD_ROOT / "tools/materialize_vae.py"
).read_text(encoding="utf-8")
RUNTIME_REVISION = "1" * 40
RUNTIME_CLOSURE_SHA256 = "2" * 64
LAUNCHER_SHA256 = "3" * 64
LOCAL_R64_RECEIPT = (
    METHOD_ROOT.parents[1]
    / "md/action_editing/20260814_man/evidence/stage_r64_joint_136309_r2"
    / "run_receipt.json"
)


def _resign(value: dict) -> dict:
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    value["receipt_digest"] = contract.object_sha256(unsigned)
    return value


def _adapter_architecture() -> dict:
    memory_unsigned = {
        "schema_version": contract.MEMORY_SCHEMA,
        "input": "detached_registered_source_visual_latent_[1,16,F,H,W]",
        "patchifier": "trainable_conv3d",
        "patch_size": [1, 4, 4],
        "temporal_patch_stride": 1,
        "temporal_pooling": False,
        "spatial_pooling_only_when_needed": True,
        "memory_token_cap": 1024,
        "encoder_width": 256,
        "hidden_size": 1536,
        "pipeline": (
            "patchify->spatial_budget->layer_norm->projection->layer_norm"
            "+fixed_3d_fourier_phase_y_x+source_role_id"
        ),
        "position_representation": "fixed_absolute_3d_fourier_phase_y_x_v1",
        "position_parameters_trainable": False,
        "position_added_after_projection": True,
        "explicit_source_role_id": 1,
        "target_noise_argument_present": False,
        "text_argument_present": False,
    }
    memory = {
        **memory_unsigned,
        "digest": contract.object_sha256(memory_unsigned),
    }
    shapes = {
        "encoder.patchifier.weight": [256, 16, 1, 4, 4],
        "encoder.patchifier.bias": [256],
        "encoder.patch_norm.weight": [256],
        "encoder.patch_norm.bias": [256],
        "encoder.projection.weight": [1536, 256],
        "encoder.projection.bias": [1536],
        "encoder.source_role.weight": [2, 1536],
    }
    for block in contract.ADAPTER_BLOCK_INDICES:
        prefix = f"adapters.{block}"
        shapes[f"{prefix}.residual_gain"] = []
        for projection in ("query", "key", "value"):
            shapes[f"{prefix}.{projection}.weight"] = [64, 1536]
        shapes[f"{prefix}.output.weight"] = [1536, 64]
    trainable = [
        {"name": name, "shape": shape, "dtype": "torch.float32"}
        for name, shape in shapes.items()
    ]
    unsigned = {
        "schema_version": contract.ADAPTER_SCHEMA,
        "gpu_validated": False,
        "scientific_quality_claim": False,
        "runtime_source_commit": contract.BERNINI_COMMIT,
        "model_revision": contract.BERNINI_MODEL_REVISION,
        "checkpoint_manifest_sha256": (
            contract.CHECKPOINT_CONTENT_MANIFEST_SHA256
        ),
        "transformer_config_digest": (
            "3802225b715e939064da7270705031a201c4585dbe1f4d3bfa37afe0be1d475b"
        ),
        "block_indices": list(contract.ADAPTER_BLOCK_INDICES),
        "block_scope_status": "structural_candidate_not_causally_admitted",
        "optimizer_authorized_by_this_receipt": False,
        "insertion": "registered_forward_hook_on_frozen_block_output",
        "query_source": "current_frozen_block_input_target_rows",
        "key_value_source": "independent_source_visual_memory_only",
        "memory_input_kinds_supported": [
            "clean_source",
            "same_noise_forward_noised_source",
        ],
        "native_self_attention_kv_replaced": False,
        "native_self_attention_kv_replayed": False,
        "native_text_cross_attention_changed": False,
        "native_blocks_replaced": False,
        "native_structure_untouched": True,
        "condition_rows_directly_written": False,
        "sp_empty_target_rank_graph_anchor": (
            "query_times_trainable_exact_zero_on_every_rank"
        ),
        "sp_collective_backward_graph_isomorphic": True,
        "target_noise_read_by_memory_encoder": "declared_per_memory_receipt",
        "source_reads_target_noise": "declared_per_memory_receipt",
        "zero_initialized_output_projection": True,
        "multiplicative_gain_initial_value": 1.0,
        "double_zero_dead_parameterization": False,
        "checkpoint_context_fn_required": True,
        "base_parameters_frozen": True,
        "memory_encoder": memory,
        "trainable": trainable,
        "feature_reward": False,
        "vlm_reward": False,
        "synthetic_target_required": False,
    }
    return {**unsigned, "digest": contract.object_sha256(unsigned)}


def _base_freeze_certificate() -> dict:
    certificate = {
        "base_frozen": True,
        "model_eval": True,
        "adapter_modules_absent": True,
        "module_count": 128,
        "module_topology_sha256": "4" * 64,
        "parameter_tensor_count": 256,
        "parameter_byte_count": 4096,
        "buffer_tensor_count": 8,
        "buffer_byte_count": 512,
        "state_metadata_sha256": "5" * 64,
        "state_content_sha256": "6" * 64,
        "exact_parameter_and_buffer_bytes_hashed": True,
        "device_and_storage_address_excluded": True,
    }
    return {
        "before": certificate,
        "after": copy.deepcopy(certificate),
        "unchanged": True,
        "certificate_sha256": contract.object_sha256(certificate),
    }


def _rank_route_trace(*, source_sha: str, arm: str, rank: int) -> dict:
    enabled = arm == "trained-carrier-r64"
    memory_source = source_sha if enabled else None
    calls = []
    for step in range(contract.NUM_INFERENCE_STEPS):
        timestep = float(contract.NUM_INFERENCE_STEPS - step)
        memory_digest = (
            hashlib.sha256(f"{source_sha}-memory-{step}".encode()).hexdigest()
            if enabled
            else None
        )
        for branch_index, branch in enumerate(contract.NATIVE_BRANCH_ORDER):
            condition_tokens = (0, 8, 16, 16)[branch_index]
            calls.append(
                {
                    "branch": branch,
                    "step_index": step,
                    "timestep": timestep,
                    "total_tokens": 8 + condition_tokens,
                    "condition_tokens": condition_tokens,
                    "target_tokens": 8,
                    "route_enabled": enabled,
                    "memory_source_video_sha256": memory_source,
                    "memory_construction_digest": memory_digest,
                }
            )
    unsigned = {
        "schema_version": contract.ROUTE_SCHEMA,
        "source_control_arm": "correct" if enabled else "carrier-off",
        "target_source_video_sha256": source_sha,
        "memory_source_video_sha256": memory_source,
        "memory_transform": "identity" if enabled else None,
        "memory_input_kind": (
            "same_noise_forward_noised_source" if enabled else None
        ),
        "exact40": True,
        "step_count": contract.NUM_INFERENCE_STEPS,
        "shared_step_call_count": (
            contract.NUM_INFERENCE_STEPS * len(contract.NATIVE_BRANCH_ORDER)
        ),
        "native_branch_order": list(contract.NATIVE_BRANCH_ORDER),
        "target_tokens": 8,
        "sequence_parallel_size": contract.SP_SIZE,
        "sequence_parallel_rank": rank,
        "shared_step_only_wrapped": True,
        "native_guidance_changed": False,
        "scheduler_changed": False,
        "optimizer_present": False,
        "memory_build_count": contract.NUM_INFERENCE_STEPS if enabled else 0,
        "calls": calls,
    }
    return {**unsigned, "trace_digest": contract.object_sha256(unsigned)}


def _aggregate_route_trace(*, source_sha: str, arm: str) -> dict:
    ranks = [
        _rank_route_trace(source_sha=source_sha, arm=arm, rank=rank)
        for rank in range(contract.WORLD_SIZE)
    ]
    semantic = dict(ranks[0])
    semantic.pop("trace_digest")
    semantic.pop("sequence_parallel_rank")
    unsigned = {
        "schema_version": contract.ROUTE_SCHEMA,
        "world_size": contract.WORLD_SIZE,
        "sequence_parallel_size": contract.SP_SIZE,
        "rank_trace_digests": [row["trace_digest"] for row in ranks],
        "semantic_projection_digest": contract.object_sha256(semantic),
        "exact40": True,
        "shared_step_calls_per_rank": (
            contract.NUM_INFERENCE_STEPS * len(contract.NATIVE_BRANCH_ORDER)
        ),
    }
    return {
        **unsigned,
        "trace_digest": contract.object_sha256(unsigned),
        "rank_traces": ranks,
    }


def _rebind_aggregate(trace: dict) -> None:
    unsigned = dict(trace)
    unsigned.pop("trace_digest", None)
    unsigned.pop("rank_traces", None)
    trace["trace_digest"] = contract.object_sha256(unsigned)


def _rebind_full_route(value: dict, record_id: str) -> None:
    trace = value["evidence"]["route_traces"][record_id]
    for rank_trace in trace["rank_traces"]:
        unsigned = dict(rank_trace)
        unsigned.pop("trace_digest", None)
        rank_trace["trace_digest"] = contract.object_sha256(unsigned)
    trace["rank_trace_digests"] = [
        rank_trace["trace_digest"] for rank_trace in trace["rank_traces"]
    ]
    semantic = dict(trace["rank_traces"][0])
    semantic.pop("trace_digest")
    semantic.pop("sequence_parallel_rank")
    trace["semantic_projection_digest"] = contract.object_sha256(semantic)
    _rebind_aggregate(trace)
    for row in value["rows"]:
        if row["record_id"] == record_id:
            row["route_trace_digest"] = trace["trace_digest"]
            return
    raise AssertionError(f"missing fixture row: {record_id}")


def _validate(value: dict) -> dict:
    return dict(
        contract.validate_receipt(
            value,
            expected_runtime_source_revision=RUNTIME_REVISION,
            expected_runtime_source_closure_sha256=RUNTIME_CLOSURE_SHA256,
            expected_launcher_sha256=LAUNCHER_SHA256,
            verify_media=False,
        )
    )


def _valid_receipt() -> dict:
    iids = [f"{index:016x}" for index in range(1, 9)]
    source_rows = []
    rows = []
    route_traces = {}
    for index, iid in enumerate(iids):
        source_sha = hashlib.sha256(f"source-{iid}".encode()).hexdigest()
        source_rows.append(
            {
                "iid": iid,
                "group_id": f"group-{index}",
                "action_family_provenance_only": f"family-{index}",
                "source_video_sha256": source_sha,
                "seed": contract.heldout_seed(iid),
                "relative_mp4": f"media/{iid}__source.mp4",
                "mp4_sha256": source_sha,
                "frame_count": 81,
                "fps": 25,
            }
        )
        gaussian = hashlib.sha256(f"gaussian-{iid}".encode()).hexdigest()
        for arm in contract.ARMS:
            record_id = f"{iid}__{arm}"
            trace = _aggregate_route_trace(source_sha=source_sha, arm=arm)
            trace_digest = trace["trace_digest"]
            route_traces[record_id] = trace
            rows.append(
                {
                    "record_id": record_id,
                    "iid": iid,
                    "group_id": f"group-{index}",
                    "action_family_provenance_only": f"family-{index}",
                    "arm": arm,
                    "source_video_sha256": source_sha,
                    "seed": contract.heldout_seed(iid),
                    "instruction": contract.GENERIC_NOOP_INSTRUCTION,
                    "instruction_sha256": contract.GENERIC_NOOP_SHA256,
                    "initial_gaussian_sha256": gaussian,
                    "route_trace_digest": trace_digest,
                    "carrier_enabled": arm == "trained-carrier-r64",
                    "latent_shape": [1, 16, 21, 60, 62],
                    "relative_mp4": f"media/{record_id}.mp4",
                    "mp4_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
                    "frame_count": 81,
                    "fps": 25,
                }
            )
    source_digest = contract.SOURCE_MANIFEST_DIGEST
    carrier_final = contract.R64_CARRIER_FINAL_SHA256
    architecture = _adapter_architecture()
    unsigned = {
        "schema_version": contract.RECEIPT_SCHEMA,
        "method": "bernini-generic-source-carrier-r64-heldout-v1",
        "complete": True,
        "complete_action_result": False,
        "action_claim_forbidden": True,
        "quality_claimed": False,
        "r64_authority": {
            "training_receipt": "/vast/r64/run_receipt.json",
            "training_receipt_file_sha256": contract.R64_TRAINING_RECEIPT_SHA256,
            "training_receipt_digest": contract.R64_TRAINING_RECEIPT_DIGEST,
            "checkpoint": "/vast/r64/stage_r_composite_checkpoint.pt",
            "checkpoint_file_sha256": contract.R64_CHECKPOINT_SHA256,
            "source_manifest": "/vast/source/source_only_split_v3.json",
            "source_manifest_file_sha256": contract.SOURCE_MANIFEST_SHA256,
            "source_manifest_digest": source_digest,
            "carrier_initial_sha256": contract.R64_CARRIER_INITIAL_SHA256,
            "carrier_final_sha256": carrier_final,
            "checkpoint_tree_sha256": contract.CHECKPOINT_TREE_SHA256,
            "checkpoint_content_manifest_sha256": (
                contract.CHECKPOINT_CONTENT_MANIFEST_SHA256
            ),
        },
        "strict_load": {
            "carrier_parameter_sha256": carrier_final,
            "carrier_parameter_count": 2_036_996,
            "checkpoint_complete_action_result": False,
            "planner_loaded": False,
            "operator_loaded": False,
            "adapter_architecture_digest": architecture["digest"],
            "loaded_block_indices": list(contract.ADAPTER_BLOCK_INDICES),
        },
        "source_manifest": {
            "path": "/vast/source/source_only_split_v3.json",
            "file_sha256": contract.SOURCE_MANIFEST_SHA256,
            "manifest_digest": source_digest,
            "split": "heldout",
            "rows": 8,
            "row_order": "iid-lexicographic",
        },
        "sources": source_rows,
        "rows": rows,
        "execution": {
            "world_size": 4,
            "sequence_parallel_size": 4,
            "num_inference_steps": 40,
            "frame_count": 81,
            "fps": 25,
            "arms": list(contract.ARMS),
            "same_source_seed_prompt_gaussian_within_pair": True,
            "base_arm": "same-loaded-model authenticated carrier-off route",
            "trained_arm": "strictly-loaded R64 same-noise carrier route",
        },
        "evidence": {
            "runtime_source": {
                "revision": RUNTIME_REVISION,
                "closure_sha256": RUNTIME_CLOSURE_SHA256,
                "launcher_sha256": LAUNCHER_SHA256,
            },
            "pinned_sources": {
                "bernini_commit": contract.BERNINI_COMMIT,
                "veomni_commit": contract.VEOMNI_COMMIT,
                "wan_diffusion_sha256": "6" * 64,
                "bernini_inference_files": {},
            },
            "base_checkpoint": {
                "path": "/vast/base",
                "tree_sha256": contract.CHECKPOINT_TREE_SHA256,
                "content_identity": {
                    "manifest_path": "/vast/base/checkpoint.sha256",
                    "manifest_sha256_computed": (
                        contract.CHECKPOINT_CONTENT_MANIFEST_SHA256
                    ),
                    "manifest_sha256_expected": (
                        contract.CHECKPOINT_CONTENT_MANIFEST_SHA256
                    ),
                    "verified_file_count": 23,
                    "every_file_sha256_verified": True,
                    "verified_entries_digest": "8" * 64,
                },
                "opened_read_only": True,
            },
            "raw_projection": {
                "path": "/vast/raw.parquet",
                "file_sha256": contract.RAW_PARQUET_SHA256,
                "safe_columns_read": list(contract.RAW_SAFE_COLUMNS),
                "target_columns_read": False,
            },
            "source_preprocessing": {iid: {} for iid in iids},
            "native_prompt_sha256": "7" * 64,
            "adapter_architecture": architecture,
            "independent_loaded_tensor_digest_before": "9" * 64,
            "independent_loaded_tensor_digest_after": "9" * 64,
            "base_freeze_certificate": _base_freeze_certificate(),
            "route_traces": route_traces,
            "host_trim_after_load": {},
            "runtime_versions": {},
        },
        "authority": {
            "manual_preservation_review_pending": True,
            "action_evaluation_performed": False,
            "reward_present": False,
            "ranking_present": False,
            "selection_present": False,
            "optimizer_present": False,
            "backward_performed": False,
            "parameter_update": False,
            "target_video_read": False,
        },
    }
    return {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}


class R64HeldoutContractTests(unittest.TestCase):
    def test_python_media_validator_has_no_external_binary_dependency(self) -> None:
        from tools import materialize_vae

        original = materialize_vae._decode_exact_video
        previous_path = os.environ.get("PATH")

        class Frames:
            shape = (81, 32, 48, 3)
            dtype = "uint8"

        try:
            materialize_vae._decode_exact_video = lambda path: (
                Frames(), 25.0, (32, 48)
            )
            os.environ["PATH"] = ""
            media = contract.validate_exact81_media(Path(__file__).resolve())
        finally:
            materialize_vae._decode_exact_video = original
            if previous_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous_path
        self.assertTrue(media["all_frames_decoded"])
        self.assertEqual(media["frame_count"], 81)
        self.assertEqual(media["fps"], 25.0)
        self.assertEqual((media["height"], media["width"]), (32, 48))
        self.assertEqual(media["decoder_backend"], "decord")

    def test_python_media_validator_rejects_80_fps_and_shape_hostiles(self) -> None:
        from tools import materialize_vae

        original = materialize_vae._decode_exact_video

        class Frames:
            dtype = "uint8"

            def __init__(self, shape: tuple[int, ...]) -> None:
                self.shape = shape

        hostile = (
            (Frames((80, 32, 48, 3)), 25.0, (32, 48)),
            (Frames((81, 32, 48, 3)), 24.0, (32, 48)),
            (Frames((81, 0, 48, 3)), 25.0, (0, 48)),
            (Frames((81, 32, 48, 4)), 25.0, (32, 48)),
            (Frames((81, 32, 49, 3)), 25.0, (32, 48)),
        )
        try:
            for decoded in hostile:
                with self.subTest(decoded=decoded[0].shape):
                    materialize_vae._decode_exact_video = (
                        lambda path, value=decoded: value
                    )
                    with self.assertRaises(contract.R64HeldoutContractError):
                        contract.validate_exact81_media(Path(__file__).resolve())
        finally:
            materialize_vae._decode_exact_video = original

    def test_source_preflight_is_no_model_and_precedes_torchrun(self) -> None:
        self.assertNotIn("import torch", PREFLIGHT_SOURCE)
        self.assertNotIn("import transformers", PREFLIGHT_SOURCE)
        self.assertNotIn("import diffusers", PREFLIGHT_SOURCE)
        self.assertNotIn("import bernini", PREFLIGHT_SOURCE)
        self.assertNotIn("subprocess", PREFLIGHT_SOURCE)
        self.assertIn("contract.validate_exact81_media(source)", PREFLIGHT_SOURCE)
        self.assertIn("target_columns_read\": False", PREFLIGHT_SOURCE)
        self.assertIn(
            "reader.get_batch(list(range(FRAME_COUNT))).asnumpy()",
            MATERIALIZER_SOURCE,
        )
        invocation = HOLDER_SOURCE.index('"${python_bin}" -B "${source_preflight}"')
        model_start = HOLDER_SOURCE.index(
            'exec "${python_bin}" -B -m torch.distributed.run'
        )
        self.assertLess(invocation, model_start)
        self.assertIn(
            "tools/preflight_generic_source_carrier_r64_heldout_sources_v1.py",
            release_builder.FILES_AND_MODES,
        )
        self.assertEqual(
            release_builder.COMPONENT_FILES["source_preflight_sha256"],
            "tools/preflight_generic_source_carrier_r64_heldout_sources_v1.py",
        )

    def test_extracted_preflight_imports_without_model_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            release = release_builder.build_release(METHOD_ROOT, root / "release")
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(release["archive"], mode="r:") as bundle:
                bundle.extractall(extracted)
            extracted_method = extracted / release_builder.MEMBER_ROOT
            program = f"""
from pathlib import Path
import sys
method_root = Path({str(extracted_method)!r}).resolve(strict=True)
sys.path.insert(0, str(method_root))
import generic_source_carrier_r64_heldout_contract_v1 as contract
contract.bind_release_preprocessing_tools(method_root)
from tools import preflight_generic_source_carrier_r64_heldout_sources_v1 as preflight
assert preflight._forbidden_model_modules() == []
assert not any(name in sys.modules for name in (
    "infer_generic_source_carrier_r64_heldout_v1",
    "infer_native_identity_generation_canary",
    "tri_branch_unipc",
))
print("NO_MODEL_PREFLIGHT_IMPORT_PASS")
"""
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd="/",
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            self.assertIn("NO_MODEL_PREFLIGHT_IMPORT_PASS", completed.stdout)

    def test_real_r64_receipt_authority_admission(self) -> None:
        self.assertEqual(
            contract.file_sha256(LOCAL_R64_RECEIPT),
            contract.R64_TRAINING_RECEIPT_SHA256,
        )
        authority = contract.load_r64_authority(
            LOCAL_R64_RECEIPT, verify_files=False
        )
        self.assertEqual(
            authority.checkpoint_content_manifest_sha256,
            contract.CHECKPOINT_CONTENT_MANIFEST_SHA256,
        )
        self.assertEqual(
            authority.checkpoint_tree_sha256, contract.CHECKPOINT_TREE_SHA256
        )
        self.assertEqual(
            authority.carrier_final_sha256, contract.R64_CARRIER_FINAL_SHA256
        )

    def test_valid_eight_by_two_receipt_and_direct_html(self) -> None:
        receipt = _valid_receipt()
        _validate(receipt)
        page = html_builder.render_html(receipt)
        self.assertEqual(page.count('<section class="example">'), 8)
        self.assertEqual(page.count("<h3>Trained carrier · R64</h3>"), 8)
        self.assertIn("not an action-editing result", page)
        self.assertIn("same observed official Gaussian", page)

    def test_hostile_resigned_authority_and_mapping_mutations_fail(self) -> None:
        mutations = []
        value = _valid_receipt()
        value["r64_authority"]["checkpoint_file_sha256"] = "9" * 64
        mutations.append(value)
        value = _valid_receipt()
        value["rows"][0]["source_video_sha256"] = "9" * 64
        mutations.append(value)
        value = _valid_receipt()
        value["rows"][1]["carrier_enabled"] = False
        mutations.append(value)
        value = _valid_receipt()
        value["rows"][0]["extra"] = True
        mutations.append(value)
        value = _valid_receipt()
        value["evidence"]["route_traces"].pop(value["rows"][0]["record_id"])
        mutations.append(value)
        value = _valid_receipt()
        value["source_manifest"]["rows"] = 7
        mutations.append(value)
        for hostile in mutations:
            with self.subTest(hostile=mutations.index(hostile)):
                with self.assertRaises(contract.R64HeldoutContractError):
                    _validate(_resign(hostile))

    def test_hostile_resigned_deep_route_and_runtime_mutations_fail(self) -> None:
        mutations = []

        value = _valid_receipt()
        value["evidence"]["runtime_source"]["revision"] = "a" * 40
        mutations.append(value)

        value = _valid_receipt()
        value["evidence"]["runtime_source"]["closure_sha256"] = "a" * 64
        mutations.append(value)

        value = _valid_receipt()
        value["evidence"]["runtime_source"]["launcher_sha256"] = "a" * 64
        mutations.append(value)

        value = _valid_receipt()
        adapter = value["evidence"]["adapter_architecture"]
        adapter["block_indices"] = [8, 12, 16, 21]
        unsigned_adapter = dict(adapter)
        unsigned_adapter.pop("digest")
        adapter["digest"] = contract.object_sha256(unsigned_adapter)
        value["strict_load"]["adapter_architecture_digest"] = adapter["digest"]
        value["strict_load"]["loaded_block_indices"] = [8, 12, 16, 21]
        mutations.append(value)

        value = _valid_receipt()
        freeze = value["evidence"]["base_freeze_certificate"]
        freeze["after"]["parameter_byte_count"] += 4
        mutations.append(value)

        value = _valid_receipt()
        record_id = value["rows"][0]["record_id"]
        value["evidence"]["route_traces"][record_id]["rank_traces"][0][
            "trace_digest"
        ] = "a" * 64
        mutations.append(value)

        value = _valid_receipt()
        record_id = value["rows"][0]["record_id"]
        trace = value["evidence"]["route_traces"][record_id]
        trace["semantic_projection_digest"] = "a" * 64
        _rebind_aggregate(trace)
        value["rows"][0]["route_trace_digest"] = trace["trace_digest"]
        mutations.append(value)

        value = _valid_receipt()
        record_id = next(
            row["record_id"] for row in value["rows"]
            if row["arm"] == "trained-carrier-r64"
        )
        for rank_trace in value["evidence"]["route_traces"][record_id]["rank_traces"]:
            rank_trace["source_control_arm"] = "wrong-owner"
        _rebind_full_route(value, record_id)
        mutations.append(value)

        value = _valid_receipt()
        record_id = value["rows"][0]["record_id"]
        trace = value["evidence"]["route_traces"][record_id]
        trace["exact40"] = False
        for rank_trace in trace["rank_traces"]:
            rank_trace["exact40"] = False
        _rebind_full_route(value, record_id)
        mutations.append(value)

        value = _valid_receipt()
        record_id = next(
            row["record_id"] for row in value["rows"]
            if row["arm"] == "trained-carrier-r64"
        )
        for rank_trace in value["evidence"]["route_traces"][record_id]["rank_traces"]:
            rank_trace["calls"][0]["route_enabled"] = False
        _rebind_full_route(value, record_id)
        mutations.append(value)

        value = _valid_receipt()
        record_id = value["rows"][0]["record_id"]
        for rank_trace in value["evidence"]["route_traces"][record_id]["rank_traces"]:
            rank_trace["calls"].pop()
            rank_trace["shared_step_call_count"] = 159
        _rebind_full_route(value, record_id)
        mutations.append(value)

        for index, hostile in enumerate(mutations):
            with self.subTest(hostile=index):
                with self.assertRaises(contract.R64HeldoutContractError):
                    _validate(_resign(hostile))

    def test_runner_emits_only_bound_mp4_artifacts(self) -> None:
        self.assertIn("def _save_mp4_outputs(", RUNNER_SOURCE)
        self.assertNotIn("native._save_outputs(", RUNNER_SOURCE)
        self.assertNotIn("normalized-clean-latent", RUNNER_SOURCE)
        self.assertNotIn("safetensors", RUNNER_SOURCE)
        self.assertIn('source_control = "carrier-off"', RUNNER_SOURCE)
        self.assertIn('arm == "trained-carrier-r64"', RUNNER_SOURCE)
        self.assertIn("same source", RUNNER_SOURCE)
        self.assertIn("There is intentionally no collective after rank-zero media decode", RUNNER_SOURCE)

    def test_release_is_deterministic_and_exact_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            first = root / "first"
            second = root / "second"
            a = release_builder.build_release(METHOD_ROOT, first)
            b = release_builder.build_release(METHOD_ROOT, second)
            self.assertEqual(a["archive_sha256"], b["archive_sha256"])
            self.assertEqual(a["manifest_sha256"], b["manifest_sha256"])
            self.assertEqual(
                a["content_closure_sha256"], b["content_closure_sha256"]
            )
            self.assertFalse(a["remote_launch_authorized"])
            self.assertEqual(a["file_count"], len(release_builder.FILES_AND_MODES))
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(a["archive"], mode="r:") as bundle:
                bundle.extractall(extracted)
            executed = release_builder.validate_executed_release(
                method_root=extracted / release_builder.MEMBER_ROOT,
                manifest_path=Path(a["manifest"]),
                expected_manifest_sha256=a["manifest_sha256"],
            )
            self.assertEqual(
                executed["content_closure_sha1"], a["content_closure_sha1"]
            )
            self.assertEqual(
                executed["content_closure_sha256"],
                a["content_closure_sha256"],
            )

    def test_extracted_release_invokes_actual_source_snapshot_import_path(self) -> None:
        """Exercise the lazy import that failed in the first AUH run."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            release = release_builder.build_release(METHOD_ROOT, root / "release")
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(release["archive"], mode="r:") as bundle:
                bundle.extractall(extracted)
            extracted_method = extracted / release_builder.MEMBER_ROOT
            source = root / "source.mp4"
            source.write_bytes(b"heldout-source-snapshot-import-closure\n")
            shadow = root / "prepended-runtime-tree"
            (shadow / "tools").mkdir(parents=True)
            (shadow / "tools/materialize_vae.py").write_text(
                "raise AssertionError('prepended tools shadow was imported')\n",
                encoding="utf-8",
            )
            program = f"""
import hashlib
from pathlib import Path
import sys

method_root = Path({str(extracted_method)!r}).resolve(strict=True)
source = Path({str(source)!r}).resolve(strict=True)
shadow = Path({str(shadow)!r}).resolve(strict=True)
sys.path.insert(0, str(method_root))

import generic_source_carrier_r64_heldout_contract_v1 as contract
identities = contract.bind_release_preprocessing_tools(method_root)
# ``activate_source_trees`` prepends Bernini/VeOmni roots after the heldout
# bootstrap.  A hostile same-named module there must not change the binding.
sys.path.insert(0, str(shadow))
import infer_source_kv_carrier_oracle as source_audit
from tools import materialize_vae

assert Path(materialize_vae.__file__).resolve(strict=True) == method_root / "tools/materialize_vae.py"
assert set(identities) == {{"tools.build_renderer_dataset", "tools.materialize_vae"}}

calls = []
class Frames:
    shape = (81, 32, 48, 3)

class Tensor:
    def __init__(self, shape):
        self.shape = shape
    def unsqueeze(self, dimension):
        assert dimension == 0
        return Tensor((1, *self.shape))

def decode(path):
    calls.append(("decode", Path(path).name))
    return Frames(), 25.0, (32, 48)

def resize(frames, bucket_hw, crop):
    assert isinstance(frames, Frames)
    assert crop is None
    calls.append(("resize", tuple(bucket_hw)))
    return Tensor((3, 81, int(bucket_hw[0]), int(bucket_hw[1])))

materialize_vae._decode_exact_video = decode
materialize_vae._resize_video = resize
pixels, metadata, source_sha = source_audit.prepare_hashed_source_snapshot(source)
assert pixels.shape[0:3] == (1, 3, 81)
assert metadata["decoded_from_private_byte_snapshot"] is True
assert metadata["original_stat_identity_stable"] is True
assert source_sha == hashlib.sha256(source.read_bytes()).hexdigest()
assert [row[0] for row in calls] == ["decode", "resize"]
print("ACTUAL_SOURCE_SNAPSHOT_PATH_PASS")
"""
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd="/",
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            self.assertIn("ACTUAL_SOURCE_SNAPSHOT_PATH_PASS", completed.stdout)

    def test_release_tools_binding_rejects_preloaded_ambiguous_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            release = release_builder.build_release(METHOD_ROOT, root / "release")
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(release["archive"], mode="r:") as bundle:
                bundle.extractall(extracted)
            extracted_method = extracted / release_builder.MEMBER_ROOT
            program = f"""
from pathlib import Path
import sys
import types
method_root = Path({str(extracted_method)!r}).resolve(strict=True)
sys.path.insert(0, str(method_root))
import generic_source_carrier_r64_heldout_contract_v1 as contract
hostile = types.ModuleType("tools")
hostile.__path__ = ["/definitely/not/the/release/tools"]
sys.modules["tools"] = hostile
try:
    contract.bind_release_preprocessing_tools(method_root)
except contract.R64HeldoutContractError:
    print("AMBIGUOUS_TOOLS_REJECTED")
else:
    raise AssertionError("ambiguous preloaded tools package was accepted")
"""
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd="/",
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            self.assertIn("AMBIGUOUS_TOOLS_REJECTED", completed.stdout)

            materializer = extracted_method / "tools/materialize_vae.py"
            materializer.chmod(0o644)
            materializer.write_bytes(materializer.read_bytes() + b"# tampered\n")
            program = f"""
from pathlib import Path
import sys
method_root = Path({str(extracted_method)!r}).resolve(strict=True)
sys.path.insert(0, str(method_root))
import generic_source_carrier_r64_heldout_contract_v1 as contract
try:
    contract.bind_release_preprocessing_tools(method_root)
except contract.R64HeldoutContractError:
    print("PREPROCESSING_TOOL_SHA_REJECTED")
else:
    raise AssertionError("tampered preprocessing tool was accepted")
"""
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd="/",
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            self.assertIn("PREPROCESSING_TOOL_SHA_REJECTED", completed.stdout)


if __name__ == "__main__":
    unittest.main()
