from __future__ import annotations

from argparse import Namespace
import ast
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from methods.bernini_action_editing import (
    generic_action_confirmation_data_prep_controller_v1 as controller,
)
from methods.bernini_action_editing.tools import (
    build_generic_action_confirmation_release_v1 as release,
)
from methods.bernini_action_editing.tools import (
    reserve4_confirmation_generation_sp4_v1 as runner,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_generic_action_confirmation_data_prep_136309_world4_v1.sh"
)
RANK_WRAPPER = (
    METHOD_ROOT
    / "scripts/auh_generic_action_confirmation_data_prep_rank_exec_v1.sh"
)
FIT_R8 = METHOD_ROOT / "releases/generic_action_fit40_generation_v8"
BRANCHES = [
    "action",
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resign(value: dict) -> dict:
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    return {**unsigned, "receipt_digest": runner.object_sha256(unsigned)}


def _valid_resource_lifecycle() -> dict:
    from methods.bernini_action_editing import (
        infer_native_identity_generation_canary as native,
    )

    rank_evidence = [
        {
            "rank": rank,
            "local_rank": rank,
            "hostname": "test-node",
            "guard_required": True,
            "guard_active": True,
            "module_path": "model.t5_text_encoder",
            "exact_positional_cpu_offload_request_only": True,
            "cpu_offload_requests_observed": 1,
            "cpu_offload_requests_suppressed": 1,
            "successful_cpu_materializations": 0,
            "delegated_to_requests": 1,
            "parameter_device_before": f"cuda:{rank}",
            "parameter_device_after": f"cuda:{rank}",
            "storage_fingerprint_before": f"{rank + 1:064x}",
            "storage_fingerprint_after": f"{rank + 1:064x}",
            "guard_method_restored": True,
            "vmrss_kib": 1000 + rank,
            "vmhwm_kib": 2000 + rank,
            "gpu_memory_limit_gib": native.T2V_GPU_MEMORY_LIMIT_GIB,
            "gpu_memory_limit_bytes": native.T2V_GPU_MEMORY_LIMIT_BYTES,
            "gpu_total_memory_bytes": 64 * 1024**3,
            "gpu_peak_allocated_bytes": 31 * 1024**3 + rank,
            "gpu_peak_reserved_bytes": 32 * 1024**3 + rank,
            "gpu_peak_reserved_within_limit": True,
        }
        for rank in range(4)
    ]
    return {
        **native.T2V_RESOURCE_LIFECYCLE_CONTRACT,
        "world4_load_completion_gate": {
            "schema_version": native.WORLD4_LOAD_COMPLETION_GATE_SCHEMA,
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "renderer_gpu_resident_trimmed_monotonic_ns_by_rank": [101, 102, 103, 104],
            "load_completion_barrier_returned_monotonic_ns_by_rank": [201, 202, 203, 204],
            "source_tokenizer_setup_entered_monotonic_ns_by_rank": [301, 302, 303, 304],
            "native_sampling_entered_monotonic_ns_by_rank": [401, 402, 403, 404],
            "world4_barrier_completed_before_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_any_source_tokenizer_setup": True,
            "all_four_renderer_loads_complete_before_first_native_sampling": True,
        },
        "world4_t2v_text_encoder_gpu_residency_gate": {
            "schema_version": native.T2V_TEXT_ENCODER_GPU_RESIDENCY_GATE_SCHEMA,
            "world_size": 4,
            "hostname": "test-node",
            "ranks": [0, 1, 2, 3],
            "module_path": "model.t5_text_encoder",
            "rank_evidence": rank_evidence,
            "all_rank_exactly_one_cpu_offload_request_suppressed": True,
            "all_rank_zero_successful_cpu_materializations": True,
            "all_rank_gpu_resident_before_and_after_sampling": True,
            "all_rank_storage_fingerprint_unchanged": True,
            "all_rank_guard_method_restored": True,
            "all_rank_peak_reserved_within_52_gib": True,
        },
    }


def _valid_scope() -> tuple[list[dict], list[dict]]:
    tasks: list[dict] = []
    proofs: list[dict] = []
    for row in runner.CONFIRMATION_CELL_REGISTRY:
        slot = str(row["seed_slot"])
        iid = str(row["iid"])
        ids = [f"{runner.SEED_PREFIXES[slot]}{iid}-{branch}" for branch in BRANCHES]
        proof = {
            "seed_slot": slot,
            "group_id": row["group_id"],
            "calibration_group_id": f"cell-{iid}-s{row['seed']}",
            "seed": row["seed"],
            "analysis_split": "confirmation",
            "candidate_ids": ids,
            "branch_order": BRANCHES,
            "complete_ten_branch_cell": True,
        }
        proofs.append(proof)
        tasks.extend({"candidate_id": candidate_id} for candidate_id in ids)
    return tasks, proofs


class GenericActionConfirmationReleaseTests(unittest.TestCase):
    @staticmethod
    def _dummy_root(root: Path) -> Path:
        method_root = root / "methods/bernini_action_editing"
        for index, (relative, mode) in enumerate(release.FILES_AND_MODES.items()):
            path = method_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"member-{index}-{relative}\n".encode("ascii"))
            path.chmod(mode)
        return method_root.resolve()

    def test_release_is_deterministic_exact_confirmation40(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root = self._dummy_root(root)
            outputs = []
            for ordinal in range(3):
                archive = root / f"source-{ordinal}.tar"
                manifest = root / f"source-{ordinal}.json"
                receipt = release.build(method_root, archive, manifest)
                outputs.append((archive.read_bytes(), manifest.read_bytes(), receipt))
            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(outputs[1][0], outputs[2][0])
            self.assertEqual(outputs[0][1], outputs[1][1])
            self.assertEqual(outputs[1][1], outputs[2][1])
            manifest = json.loads(outputs[0][1])
            authority = manifest["authority"]
            self.assertEqual(
                manifest["schema_version"],
                "bernini-generic-action-confirmation-data-prep-release-v3",
            )
            self.assertEqual(manifest["release_generation"], "r3")
            self.assertEqual(
                manifest["release_scope"],
                "reserve4-confirmation40-media-only-pending-external-blind-review",
            )
            self.assertEqual(authority["analysis_split"], "confirmation")
            self.assertEqual(authority["candidate_count"], 40)
            self.assertEqual(authority["seed_cell_count"], 4)
            self.assertEqual(
                authority["confirmation_iids"],
                ["0c6915018a5f4d9b", "33322eb8ec1e4703"],
            )
            self.assertEqual(authority["branch_order"], BRANCHES)
            self.assertTrue(authority["confirmation_generation_authorized"])
            self.assertTrue(authority["pending_external_blind_review"])
            self.assertFalse(authority["generation_runner_has_review_authority"])
            self.assertFalse(authority["existing_core4_confirmation_media_included"])
            self.assertEqual(
                authority["future_external_blind_review_population_candidate_count"],
                80,
            )
            self.assertFalse(
                authority["generated_rgb_latent_gaussian_is_editor_input_or_target"]
            )
            row_by_path = {row["path"]: row for row in manifest["files"]}
            self.assertEqual(
                _sha(METHOD_ROOT / "tools/reserve4_fixed_generation_sp4_v1.py"),
                "be722e4020040ba446f290f07378e870e2d3c1a4228ec997c3447770fcb53d5d",
            )
            self.assertEqual(
                manifest["component_pins"]["resource_contract_sha256"],
                row_by_path["tools/reserve4_fixed_generation_sp4_v1.py"][
                    "sha256"
                ],
            )
            self.assertIn(
                "tools/reserve4_confirmation_generation_sp4_v1.py",
                row_by_path,
            )
            self.assertTrue(
                manifest["topology"][
                    "confirmation_smoke_supplemental_physical_receipt_create_only"
                ]
            )
            self.assertTrue(
                manifest["topology"][
                    "formal_candidate_boundary_checks_run_inside_rank_wrapper"
                ]
            )
            for field in (
                "independent_full81_blind_review_present",
                "same_runner_may_self_certify_visual_review",
                "phi_v1_extraction_authorized",
                "optimizer_created",
                "optimizer_authorized",
                "p_or_o_manifest_materialization_authorized",
                "training_authorized",
            ):
                self.assertFalse(authority[field], field)
            topology = manifest["topology"]
            self.assertEqual(
                topology["holder"],
                {"job_id": 136309, "node": "auh7-1b-gpu-280"},
            )
            self.assertTrue(topology["fresh_run_root_required"])
            self.assertEqual(topology["consecutive_master_port_count"], 4)
            self.assertEqual(topology["master_port_inclusive_range"], [1024, 65532])
            self.assertEqual(topology["slurm_child_gpu_count"], 8)
            self.assertEqual(topology["numbered_slurm_children"], 1)
            self.assertEqual(topology["compute_world_size"], 4)
            self.assertEqual(topology["run_sp4_shard_process_count"], 4)
            self.assertEqual(topology["world4_model_invocation_count"], 40)
            self.assertEqual(topology["total_native_model_invocation_count"], 41)
            self.assertEqual(topology["sealed_shard_order"], controller.SHARD_ORDER)
            self.assertTrue(topology["all_model_invocations_strictly_serial"])
            self.assertTrue(topology["serialized_world4_host_checkpoint_load"])
            self.assertTrue(topology["model_load_lock_node_local"])
            self.assertTrue(
                topology["model_load_lock_held_through_gpu_move_and_malloc_trim"]
            )
            for field in (
                "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup",
                "resource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling",
                "compile_smoke_asserts_world4_load_completion_ordering",
                "t2v_text_encoder_rank_gpu_residency_required",
                "t2v_text_encoder_exact_cpu_offload_suppressed_once_per_rank",
                "t2v_text_encoder_retired_only_with_renderer",
                "r10_smoke_mp4_gaussian_latent_byte_parity_required",
            ):
                self.assertTrue(topology[field], field)
            self.assertTrue(
                topology["t2v_vae_load_deferred_until_rank0_post_sampling"]
            )
            self.assertTrue(
                topology[
                    "world4_renderer_retirement_barrier_before_rank_zero_vae_load"
                ]
            )
            self.assertFalse(topology["rank_or_gpu_action_family_partition"])

    def test_r3_rank_wrapper_binds_monitor_boundaries_and_physical_reopen(self) -> None:
        source = RANK_WRAPPER.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            source.count("assert-host-memory-monitor-live"), 3
        )
        self.assertIn("seal-physical-smoke-receipt", source)
        self.assertIn('"${global_rank}" == 0', source)
        self.assertIn('"${cache_token}" == compile-smoke-*', source)
        self.assertIn("GADP_PHYSICAL_SMOKE_RECEIPT_OUTPUT", source)
        self.assertIn("GADP_RESOURCE_CONTRACT_SHA256", source)
        self.assertLess(
            source.index("pre-candidate host monitor boundary"),
            source.index('"${python_bin}" -B "$@"'),
        )
        self.assertLess(
            source.index('wait "${child_pid}"'),
            source.index("post-candidate host monitor boundary"),
        )

    def test_r3_launcher_separates_confirmation_runner_from_frozen_resource_contract(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'generator="${method_root}/tools/reserve4_confirmation_generation_sp4_v1.py"',
            source,
        )
        self.assertIn(
            'resource_contract="${method_root}/tools/reserve4_fixed_generation_sp4_v1.py"',
            source,
        )
        self.assertIn('"${generator}" smoke-sp4', source)
        self.assertIn('"${generator}" run-sp4', source)
        self.assertIn('"${resource_contract}" host-memory-monitor', source)
        self.assertIn(
            '"${resource_contract}" seal-terminal-host-memory-gate', source
        )
        self.assertIn("seal-compile-host-memory-gate", source)
        self.assertIn("compile-smoke-physical-tensor-evidence.json", source)
        self.assertIn("fuser /dev/kfd", source)
        self.assertNotIn("--host-memory-gate-output", source)

    def test_manifest_pins_registry_specs_iids_seeds_and_full_branch_order(self) -> None:
        manifest, _ = release.build_manifest(METHOD_ROOT)
        authority = manifest["authority"]
        self.assertEqual(
            authority["authoring_registry_raw_sha256"],
            "204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c",
        )
        self.assertEqual(
            authority["reserve4_selection_raw_sha256"],
            "a4baa1aea27f6497ca2dd615cc09b2b90eee37173f506e60ae7d630c41886be6",
        )
        self.assertEqual(authority["seed1_spec_raw_sha256"], runner.SPEC_AUTHORITIES["seed1"])
        self.assertEqual(authority["seed2_spec_raw_sha256"], runner.SPEC_AUTHORITIES["seed2"])
        self.assertEqual(
            authority["confirmation_seed_cells"],
            [
                ["0c6915018a5f4d9b", 2026080822, "sp4-a"],
                ["33322eb8ec1e4703", 2026080823, "sp4-b"],
                ["0c6915018a5f4d9b", 2026080922, "sp4-a"],
                ["33322eb8ec1e4703", 2026080923, "sp4-b"],
            ],
        )

    def test_scope_validator_accepts_only_exact_two_iid_two_seed_population(self) -> None:
        tasks, proofs = _valid_scope()
        runner._validate_confirmation_scope(tasks, proofs)
        hostile = [dict(row) for row in proofs]
        hostile[0] = dict(hostile[0], seed=2026080821)
        with self.assertRaisesRegex(runner.Reserve4GenerationError, "registry differs"):
            runner._validate_confirmation_scope(tasks, hostile)
        hostile = [dict(row) for row in proofs]
        hostile[1] = dict(hostile[1], analysis_split="fit")
        with self.assertRaisesRegex(runner.Reserve4GenerationError, "registry differs"):
            runner._validate_confirmation_scope(tasks, hostile)
        hostile_tasks = list(tasks)
        hostile_tasks[0] = {"candidate_id": "pair5-t2v-reserve4-v1-wrong-action"}
        with self.assertRaisesRegex(runner.Reserve4GenerationError, "order or identity"):
            runner._validate_confirmation_scope(hostile_tasks, proofs)
        with self.assertRaisesRegex(runner.Reserve4GenerationError, "size differs"):
            runner._validate_confirmation_scope(tasks[:-1], proofs)

    def test_fit_is_not_selectable_from_confirmation_generator(self) -> None:
        parser = runner.build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "build-plan",
                        "--seed1-spec", "/tmp/seed1.json",
                        "--seed2-spec", "/tmp/seed2.json",
                        "--split", "fit",
                        "--output-dir", "/tmp/output",
                    ]
                )
        with self.assertRaisesRegex(
            runner.Reserve4GenerationError, "only the confirmation split"
        ):
            runner.build_plan(
                seed1_spec="/tmp/seed1.json",
                seed2_spec="/tmp/seed2.json",
                split="fit",
                output_dir="/tmp/output",
            )

    def test_only_two_data_generation_entrypoints_are_runnable(self) -> None:
        self.assertEqual(
            set(release.ENTRYPOINTS),
            {
                "generic_action_confirmation_data_prep_controller_v1.py",
                "scripts/auh_generic_action_confirmation_data_prep_136309_world4_v1.sh",
            },
        )
        forbidden = {
            "tools/materialize_phi_v1_sidecars_sp4.py",
            "tools/generic_action_manifest_v1.py",
            "train_generic_source_anchored_action_v1.py",
            "generic_source_anchored_action_v1.py",
        }
        self.assertTrue(forbidden.isdisjoint(release.FILES_AND_MODES))
        self.assertNotIn("train_lora.py", release.ENTRYPOINTS)
        self.assertIn("train_lora.py", release.FILES_AND_MODES)

    def test_archive_is_exact_regular_file_closure_with_fixed_modes(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        raw = release.build_archive(manifest, payloads)
        release.verify_archive(raw, manifest)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
        self.assertEqual(len(members), len(release.FILES_AND_MODES))
        self.assertEqual(
            [member.name for member in members],
            [f"{release.MEMBER_ROOT}/{row['path']}" for row in manifest["files"]],
        )
        for member in members:
            self.assertTrue(member.isfile())
            self.assertFalse(member.issym())
            self.assertFalse(member.islnk())
            self.assertEqual(member.uid, 0)
            self.assertEqual(member.gid, 0)
            self.assertEqual(member.mtime, 0)

    def test_launcher_is_one_all8_child_with_four_serial_world4_shards(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("readonly holder_job=136309", source)
        self.assertIn("readonly holder_node=auh7-1b-gpu-280", source)
        self.assertIn('[[ "${split}" == confirmation ]]', source)
        self.assertIn("--gpus-per-task=8", source)
        self.assertIn("--cpus-per-task=32 --mem=60G", source)
        self.assertEqual(source.count("srun --jobid="), 1)
        self.assertEqual(source.count("run_sealed_shard "), 4)
        expected_calls = [
            "run_sealed_shard 1 seed1 sp4-a 0,1,2,3",
            "run_sealed_shard 2 seed1 sp4-b 4,5,6,7",
            "run_sealed_shard 3 seed2 sp4-a 0,1,2,3",
            "run_sealed_shard 4 seed2 sp4-b 4,5,6,7",
        ]
        offsets = [source.index(call) for call in expected_calls]
        self.assertEqual(offsets, sorted(offsets))
        self.assertLess(source.index("run_compile_smoke"), offsets[0])
        self.assertNotIn("--ntasks=8", source)
        self.assertNotIn("WORLD_SIZE=8", source)
        self.assertNotIn("DP2", source)
        self.assertNotIn("dog=", source.lower())
        self.assertNotIn("human=", source.lower())
        self.assertIn(
            'readonly model_load_lock="${task_scratch}/renderer-load.lock"', source
        )
        self.assertIn('chmod 0400 "${model_load_lock}"', source)
        self.assertIn('export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1', source)
        self.assertIn('export NATIVE_V_AXIS_LOAD_LOCK="${model_load_lock}"', source)
        self.assertIn(
            'export NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED=1', source
        )
        self.assertIn("serialized_world4_host_checkpoint_load=true", source)
        self.assertIn(
            "model_load_lock_held_through_gpu_move_and_malloc_trim=true", source
        )
        self.assertIn(
            "t2v_vae_load_deferred_until_rank0_post_sampling=true", source
        )
        self.assertIn(
            "world4_renderer_retirement_barrier_before_rank_zero_vae_load=true",
            source,
        )
        for field in (
            "world4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup=true",
            "resource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling=true",
            "compile_smoke_asserts_world4_load_completion_ordering=true",
            "t2v_text_encoder_rank_gpu_residency_required=true",
            "t2v_text_encoder_exact_cpu_offload_suppressed_once_per_rank=true",
            "t2v_text_encoder_retired_only_with_renderer=true",
            "r10_smoke_mp4_gaussian_latent_byte_parity_required=true",
            "t5_rank_gpu_residency=true",
            "r10_artifact_parity=true",
        ):
            self.assertIn(field, source)
        self.assertIn("GENERIC_ACTION_CONFIRMATION40_GENERATION_R3_COMPLETE", source)
        self.assertNotIn("GENERIC_ACTION_CONFIRMATION40_GENERATION_V1_COMPLETE", source)
        for forbidden in ("scancel", "scontrol release", "scontrol requeue"):
            self.assertNotIn(forbidden, source)

    def test_formal_generation_is_gated_by_exact_disposable_smoke_receipt(self) -> None:
        tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
        functions = {
            node.name: ast.get_source_segment(
                Path(runner.__file__).read_text(encoding="utf-8"), node
            )
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        formal = functions["run_sp4"] or ""
        self.assertLess(
            formal.index("_validate_compile_smoke_for_runtime"),
            formal.index("output.mkdir"),
        )
        smoke = functions["run_compile_smoke_sp4"] or ""
        self.assertIn("_delete_disposable_smoke_root", smoke)
        self.assertIn('"formal_candidate_count_at_gate": 0', smoke)
        self.assertIn('"compile_smoke_passed": True', smoke)

    def test_rank_wrapper_uses_private_node_local_cache_and_rejects_nfs(self) -> None:
        source = RANK_WRAPPER.read_text(encoding="utf-8")
        self.assertIn("torchrun invokes this wrapper with --no_python", source)
        self.assertIn("stat -f -c '%T'", source)
        self.assertIn("ext2/ext3|xfs|tmpfs", source)
        self.assertNotIn("nfs", source.lower().split("case", 1)[1].split("esac", 1)[0])
        for variable in (
            "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "PYTHONPYCACHEPREFIX", "MIOPEN_USER_DB_PATH",
            "MIOPEN_CUSTOM_CACHE_DIR",
        ):
            self.assertIn(f"export {variable}=", source)

    def test_runtime_binding_authenticates_and_forwards_node_local_load_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scratch = root / "scratch"
            bernini = root / "Bernini"
            veomni = root / "VeOmni"
            checkpoint = root / "checkpoint"
            for path in (scratch, bernini, veomni, checkpoint):
                path.mkdir()
            checkpoint_manifest = root / "checkpoint.sha256"
            checkpoint_manifest.write_text("sealed\n", encoding="ascii")
            load_lock = scratch / "renderer-load.lock"
            load_lock.write_bytes(b"")
            load_lock.chmod(0o400)
            args = Namespace(
                python=str(Path(sys.executable).resolve()),
                bernini_root=str(bernini),
                veomni_root=str(veomni),
                checkpoint=str(checkpoint),
                checkpoint_content_manifest=str(checkpoint_manifest),
                method_source_revision="1" * 40,
                method_source_archive_sha256="2" * 64,
                master_port=29571,
            )
            environment = {
                "GADP_NODE_LOCAL_SCRATCH": str(scratch),
                "GADP_NODE_LOCAL_SCRATCH_FSTYPE": "ext2/ext3",
                "NATIVE_SERIALIZED_HOST_LOAD_REQUIRED": "1",
                "NATIVE_V_AXIS_LOAD_LOCK": str(load_lock),
                "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED": "1",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                runner, "_filesystem_type", return_value="ext2/ext3"
            ):
                runtime, python, _worker, _rank_exec, bound_scratch = (
                    runner._runtime_binding(args)
                )
                self.assertEqual(
                    runtime["serialized_host_checkpoint_load"]["sha256"],
                    runner.EMPTY_FILE_SHA256,
                )
                candidate_environment = runner._candidate_environment(
                    expected_visible="0,1,2,3",
                    python=python,
                    scratch=bound_scratch,
                    cache_token="confirmation40-test",
                )
                self.assertEqual(
                    candidate_environment["NATIVE_V_AXIS_LOAD_LOCK"], str(load_lock)
                )
                self.assertEqual(
                    candidate_environment["NATIVE_SERIALIZED_HOST_LOAD_REQUIRED"],
                    "1",
                )
                self.assertEqual(
                    candidate_environment[
                        "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED"
                    ],
                    "1",
                )
                self.assertEqual(
                    runtime["t2v_text_encoder_rank_gpu_residency"],
                    {
                        "required": True,
                        "environment_variable": (
                            "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED"
                        ),
                        "official_model_sample_preserved": True,
                        "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
                        "all_other_to_requests_delegated": True,
                        "text_encoder_retired_only_with_renderer": True,
                    },
                )
                load_lock.chmod(0o600)
                with self.assertRaisesRegex(
                    runner.Reserve4GenerationError, "lock identity differs"
                ):
                    runner._runtime_binding(args)

    def test_confirmation_receipt_replays_native_resource_lifecycle_contract(self) -> None:
        pair_source = (
            METHOD_ROOT / "infer_pair_v5_t2v_calibration_bank.py"
        ).read_text(encoding="utf-8")
        native_source = (
            METHOD_ROOT / "infer_native_identity_generation_canary.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "native.validate_t2v_resource_lifecycle(", pair_source
        )
        self.assertIn("require_serialized_load=True", pair_source)
        self.assertIn('"resource_lifecycle": dict(resource_lifecycle)', native_source)
        manifest, _ = release.build_manifest(METHOD_ROOT)
        members = {row["path"] for row in manifest["files"]}
        self.assertIn("infer_pair_v5_t2v_calibration_bank.py", members)
        self.assertIn("infer_native_identity_generation_canary.py", members)
        self.assertEqual(
            runner.COMPILE_SMOKE_SCHEMA,
            "bernini-generic-action-confirmation40-compile-smoke-v3",
        )
        self.assertEqual(
            runner.PLAN_SCHEMA,
            "bernini-reserve4-confirmation-generation-sp4-plan-v2",
        )
        runner_source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('"reserve4-fixed-generation-plan-v2.json"', runner_source)
        self.assertIn('two-seed-complete10-v2', runner_source)
        self.assertEqual(
            controller.PLAN_SCHEMA,
            "bernini-generic-action-confirmation40-generation-136309-plan-v3",
        )
        self.assertEqual(
            controller.COMPLETION_SCHEMA,
            "bernini-generic-action-confirmation40-generation-136309-completion-v3",
        )
        controller_source = Path(controller.__file__).read_text(encoding="utf-8")
        self.assertIn(
            '"plan_id": "generic-action-confirmation40-generation-136309-r3"',
            controller_source,
        )
        runner_source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "native.validate_t2v_resource_lifecycle(", runner_source
        )
        self.assertIn("require_serialized_load=True", runner_source)
        self.assertIn("_validate_r10_smoke_artifact_parity", runner_source)
        self.assertEqual(
            list(runner._R10_SMOKE_ARTIFACT_AUTHORITY),
            list(runner._SMOKE_ARTIFACT_NAMES),
        )

    def test_compile_smoke_dynamically_validates_lifecycle_and_r10_parity(self) -> None:
        lifecycle = _valid_resource_lifecycle()
        from methods.bernini_action_editing import (
            infer_native_identity_generation_canary as native,
        )

        self.assertEqual(
            native.validate_t2v_resource_lifecycle(
                lifecycle, require_serialized_load=True
            ),
            lifecycle,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scratch = root / "scratch"
            runtime = {
                "method_root": str(root / "method"),
                "python": {"path": str(root / "python"), "sha256": "1" * 64},
                "bernini_root": str(root / "Bernini"),
                "veomni_root": str(root / "VeOmni"),
                "checkpoint": str(root / "checkpoint"),
                "checkpoint_content_manifest": {
                    "path": str(root / "checkpoint.sha256"),
                    "sha256": "2" * 64,
                },
                "method_source_revision": "3" * 40,
                "method_source_archive_sha256": "4" * 64,
                "generation_worker": {
                    "path": str(root / "worker.py"),
                    "sha256": "5" * 64,
                },
                "rank_cache_wrapper": {
                    "path": str(root / "rank-exec.sh"),
                    "sha256": "6" * 64,
                },
                "preprocessing_tools": dict(runner.PREPROCESSING_TOOL_SHA256),
                "node_local_scratch": {
                    "path": str(scratch),
                    "filesystem_type": "ext2/ext3",
                },
                "serialized_host_checkpoint_load": {
                    "required": True,
                    "environment_variable": "NATIVE_V_AXIS_LOAD_LOCK",
                    "path": str(scratch / "renderer-load.lock"),
                    "sha256": runner.EMPTY_FILE_SHA256,
                    "mode": "0400",
                    "parent_is_authenticated_node_local_scratch": True,
                    "lock_held_through_model_to_rank_gpu_and_malloc_trim": True,
                },
                "t2v_text_encoder_rank_gpu_residency": {
                    "required": True,
                    "environment_variable": "NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED",
                    "official_model_sample_preserved": True,
                    "exact_positional_cpu_offload_request_suppressed_once_per_rank": True,
                    "all_other_to_requests_delegated": True,
                    "text_encoder_retired_only_with_renderer": True,
                },
            }
            unsigned = {
                "schema_version": runner.COMPILE_SMOKE_SCHEMA,
                "plan": {
                    "path": str(root / "plan.json"),
                    "file_sha256": "7" * 64,
                    "plan_digest": "8" * 64,
                },
                "smoke_task": {
                    "candidate_id": "confirmation-smoke",
                    "seed_slot": "seed1",
                    "group_id": "sp4-a",
                    "visible_gpus": [0, 1, 2, 3],
                    "analysis_split": "confirmation",
                    "ordinal": 0,
                    "candidate_spec_path": str(root / "candidate.json"),
                    "candidate_spec_sha256": "9" * 64,
                    "root_spec_sha256": "a" * 64,
                },
                "runtime": runtime,
                "candidate_evidence": {
                    "candidate_receipt_file_sha256": "b" * 64,
                    "candidate_receipt_digest": "c" * 64,
                    "native_receipt_file_sha256": "d" * 64,
                    "native_receipt_digest": "e" * 64,
                    "resource_lifecycle": lifecycle,
                    "artifact_identities": [
                        {
                            "name": name,
                            **runner._R10_SMOKE_ARTIFACT_AUTHORITY[name],
                        }
                        for name in runner._SMOKE_ARTIFACT_NAMES
                    ],
                },
                "world_size": 4,
                "full_native_sampling_steps": 40,
                "formal_candidate_count_at_gate": 0,
                "disposable_output_deleted": True,
                "compile_smoke_passed": True,
                "training_performed": False,
                "optimizer_authorized": False,
            }
            receipt = _resign(unsigned)
            path = root / "compile-smoke.json"
            path.write_bytes(runner.canonical_json_bytes(receipt) + b"\n")
            observed, _, _ = runner.load_compile_smoke_receipt(path, _sha(path))
            self.assertEqual(observed, receipt)

            stale = json.loads(path.read_text(encoding="ascii"))
            stale["candidate_evidence"]["resource_lifecycle"][
                "schema_version"
            ] = "bernini-native-t2v-resource-lifecycle-v3"
            stale = _resign(stale)
            path.write_bytes(runner.canonical_json_bytes(stale) + b"\n")
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "did not prove WORLD4 load completion",
            ):
                runner.load_compile_smoke_receipt(path, _sha(path))

            hostile = json.loads(
                runner.canonical_json_bytes(receipt).decode("ascii")
            )
            hostile["candidate_evidence"]["artifact_identities"][0][
                "file_sha256"
            ] = "f" * 64
            hostile = _resign(hostile)
            path.write_bytes(runner.canonical_json_bytes(hostile) + b"\n")
            with self.assertRaisesRegex(
                runner.Reserve4GenerationError,
                "byte-exact r10 artifact authority",
            ):
                runner.load_compile_smoke_receipt(path, _sha(path))

    def test_release_local_import_closure_has_no_missing_local_module(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        paths = {row["path"] for row in manifest["files"]}
        local_modules = {
            Path(path).stem: path
            for path in paths
            if path.endswith(".py") and "/" not in path
        }
        tool_modules = {
            Path(path).stem: path
            for path in paths
            if path.startswith("tools/") and path.endswith(".py")
        }
        repository_modules = {
            path.stem
            for path in METHOD_ROOT.glob("*.py")
        } | {
            path.stem
            for path in (METHOD_ROOT / "tools").glob("*.py")
        }
        unresolved: set[tuple[str, str]] = set()
        for relative, raw in payloads.items():
            if not relative.endswith(".py"):
                continue
            tree = ast.parse(raw.decode("utf-8"), filename=relative)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imports.append(node.module.split(".")[0])
            for module in imports:
                if module in repository_modules and module not in local_modules and module not in tool_modules:
                    unresolved.add((relative, module))
        self.assertEqual(unresolved, set())

    def test_new_entrypoints_contain_no_optimizer_training_phi_or_review_action(self) -> None:
        for relative in release.ENTRYPOINTS + (
            "tools/reserve4_confirmation_generation_sp4_v1.py",
        ):
            source = (METHOD_ROOT / relative).read_text(encoding="utf-8")
            for forbidden in (
                "optimizer.step(",
                "backward(",
                "materialize_phi_v1",
                "generic_action_manifest_v1",
                "review_passed = True",
            ):
                self.assertNotIn(forbidden, source, f"{relative}: {forbidden}")

    def test_fit_r8_frozen_payload_was_not_modified(self) -> None:
        expected = {
            FIT_R8 / "source.tar": "36d5c6ef5405a2f2563a13f632b0dbd8aaaffae30210c02f248fcf461bdaec9f",
            FIT_R8 / "source.manifest.json": "872c571d635ddd1456058ffbb73cca73d2f4a1f720d62c182d2eecc57f6f34f6",
        }
        self.assertEqual({path: _sha(path) for path in expected}, expected)

    def test_shell_entrypoints_are_syntax_valid(self) -> None:
        for path in (LAUNCHER, RANK_WRAPPER):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
