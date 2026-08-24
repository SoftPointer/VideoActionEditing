from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
CORE = METHOD_ROOT / "source_self_role_repaint.py"
RUNTIME = METHOD_ROOT / "source_self_runtime.py"
TRAINER = METHOD_ROOT / "train_source_self_role_repaint.py"
MATERIALIZER = METHOD_ROOT / "tools" / "materialize_source_self_role_repaint.py"
SPEC = METHOD_ROOT / "assets" / "source_self_role_repaint_canary_spec_v2.json"

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
import source_self_runtime as source_runtime

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None
    save_file = None

if torch is not None:
    import source_self_role_repaint as core
    import train_source_self_role_repaint as training
else:
    core = None
    training = None


class SourceSelfStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = CORE.read_text(encoding="utf-8")
        cls.runtime = RUNTIME.read_text(encoding="utf-8")
        cls.trainer = TRAINER.read_text(encoding="utf-8")
        cls.materializer = MATERIALIZER.read_text(encoding="utf-8")
        cls.spec = SPEC.read_text(encoding="utf-8")
        ast.parse(cls.core)
        ast.parse(cls.runtime)
        ast.parse(cls.trainer)
        ast.parse(cls.materializer)

    def test_exact81_three_independent_refs_and_roles_are_frozen(self) -> None:
        for fragment in (
            "LATENT_PHASES = 21",
            "REFERENCE_COUNT = 3",
            "REFERENCE_RGB_INDICES = (0, 40, 80)",
            "ROLE_DONOR = 1",
            "ROLE_REFERENCE = 2",
            "ROLE_TARGET = 3",
        ):
            self.assertIn(fragment, self.core)
        self.assertIn("source_id=1", self.trainer)
        self.assertIn("source_id=index + 2", self.trainer)
        self.assertIn("source_id=0", self.trainer)

    def test_closed_adapter_scope_is_role_embedding_and_target_row_q_o(self) -> None:
        for fragment in (
            "TRAINABLE_BLOCK_INDICES = tuple(range(23))",
            'projection not in {"to_q", "to_out.0"}',
            '"key_value_trainable": False',
            '"cross_attention_trainable": False',
            '"target_row_only": True',
            "roles == ROLE_TARGET",
        ):
            self.assertIn(fragment, self.core)
        self.assertNotIn("attn2.to_", self.core)
        self.assertIn('choices=("early-mid-0-22", "all30-ablation")', self.trainer)

    def test_real_source_is_only_positive_and_controls_have_no_gradient(self) -> None:
        for fragment in (
            '"independent_pinned_vae_encode_of_same_raw_clean_source_rgb"',
            '"paired_dataset_accessed": False',
            '"prior_posterior_accessed": False',
            '"target_video_path_present": False',
            '"edited_target_accessed": False',
            '"action_supervision_present": False',
            "with torch.no_grad(), torch.autocast",
            '"all_controls_no_gradient": True',
            '"optimizer_supervision": "none"',
            '"donor_dc_correct_refs"',
            '"ordered_refs_absent"',
            '"reference_source_ids": []',
        ):
            self.assertIn(fragment, self.trainer)
        self.assertIn("heldout_factorial_cells", self.trainer)
        self.assertNotIn("full644", self.trainer)

    def test_absent_reference_geometry_reads_the_layout_authority(self) -> None:
        self.assertIn(
            "absent.total_tokens >= main.layout.total_tokens", self.trainer
        )
        self.assertNotIn("absent.total_tokens >= main.total_tokens", self.trainer)
        prepared = next(
            node
            for node in ast.parse(self.trainer).body
            if isinstance(node, ast.ClassDef) and node.name == "PreparedCondition"
        )
        fields = {
            node.target.id
            for node in prepared.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        }
        self.assertEqual(fields, {"input_patches", "rotary", "layout"})

    def test_legacy_world8_default_and_explicit_world4_profile_are_closed(self) -> None:
        for fragment in (
            "WORLD_SIZE = 8",
            "SP_SIZE = 4",
            "DP_SIZE = 2",
            "LOGICAL_ARM_COUNT = 2",
            "CANARY_STEPS = 1",
            "SP_GROUP_RANKS = ((0, 1, 2, 3), (4, 5, 6, 7))",
            "DP_GROUP_RANKS = ((0, 4), (1, 5), (2, 6), (3, 7))",
            '"--parallel-topology"',
            '"--method-source-revision-kind"',
            '"--method-source-manifest-sha256"',
            "default=runtime.WORLD8_DP2_SP4.profile",
            'args.rho != 0.0',
            'choices=("engineering-canary",)',
        ):
            self.assertIn(fragment, self.trainer)
        for fragment in (
            'profile="world8-dp2-sp4"',
            'profile="world4-dp1-sp4"',
            "WORLD4_DP1_SP4",
            "topology: ParallelTopology = WORLD8_DP2_SP4",
            "values[\"LOCAL_WORLD_SIZE\"] not in (2, topology.world_size)",
            "if topology.dp_size > 1:",
            "parameter.grad.div_(float(topology.sp_size))",
            "parameter.grad.div_(float(topology.dp_size))",
        ):
            self.assertIn(fragment, self.runtime)
        self.assertIn(
            "runtime.synchronize_gradients(trainable, parallel)", self.trainer
        )
        self.assertIn("import source_self_runtime as runtime", self.trainer)
        self.assertNotIn("train_ramp_c0", self.trainer)
        self.assertIn(
            '"method_source_revision_kind": args.method_source_revision_kind',
            self.trainer,
        )
        self.assertIn(
            '"method_source_manifest_sha256": args.method_source_manifest_sha256',
            self.trainer,
        )

    def test_world4_serializes_two_logical_arms_without_changing_objective(self) -> None:
        for fragment in (
            "return tuple(range(LOGICAL_ARM_COUNT))",
            "return float(topology.dp_size) / float(LOGICAL_ARM_COUNT)",
            "for logical_arm in local_logical_arms:",
            "for name, condition in prepared.controls.items():",
            "for logical in prepared_arms:",
            "scaled_loss = loss * loss_scale",
            "scaled_loss.backward()",
            '"all_logical_controls_preceded_any_backward": True',
            '"serial_logical_arm_accumulation": topology.dp_size == 1',
            '"logical_objective": "mean(logical_arm_0,logical_arm_1)"',
            '"dp_all_reduce_skipped_for_dp1": topology.dp_size == 1',
        ):
            self.assertIn(fragment, self.trainer)

    def test_output_transaction_is_rank_zero_filesystem_only(self) -> None:
        prepare = self.runtime[
            self.runtime.index("def prepare_output_transaction(") :
            self.runtime.index("def _lstat_absent(")
        ]
        publish = self.runtime[
            self.runtime.index("def publish_output_transaction(") :
            self.runtime.index("\n\n__all__ =")
        ]
        self.assertIn("dist.all_gather_object(gathered, local", prepare)
        self.assertIn("if rank == 0:", prepare)
        self.assertIn("dist.broadcast_object_list(reservation", prepare)
        self.assertNotIn("world_all_true(fresh", prepare)
        self.assertNotIn("dist.barrier", prepare)
        self.assertIn("_rename_directory_noreplace(stage, output)", publish)
        self.assertIn("dist.broadcast_object_list(publication", publish)
        self.assertIn("runtime.publish_output_transaction(", self.trainer)
        self.assertNotIn("os.replace(stage, output)", self.trainer)
        self.assertNotIn("if not output.is_dir()", self.trainer)

    def test_nfs_unsupported_noreplace_uses_inode_pinned_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source = parent / ".one_step.staging"
            destination = parent / "one_step"
            source.mkdir(mode=0o750)
            (source / "artifact").write_bytes(b"sealed")
            source_identity = (source.stat().st_dev, source.stat().st_ino)
            with mock.patch.object(
                source_runtime,
                "_try_renameat2_noreplace",
                return_value=source_runtime.errno.EINVAL,
            ):
                source_runtime._rename_directory_noreplace(source, destination)
            self.assertFalse(source.exists())
            self.assertEqual(
                (destination.stat().st_dev, destination.stat().st_ino),
                source_identity,
            )
            self.assertEqual((destination / "artifact").read_bytes(), b"sealed")

    def test_nfs_reservation_fallback_never_clobbers_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source = parent / ".one_step.staging"
            destination = parent / "one_step"
            source.mkdir(mode=0o750)
            destination.mkdir(mode=0o700)
            (destination / "owner").write_bytes(b"foreign")
            with mock.patch.object(
                source_runtime,
                "_try_renameat2_noreplace",
                return_value=source_runtime.errno.EINVAL,
            ):
                with self.assertRaises(source_runtime.SourceSelfRuntimeError):
                    source_runtime._rename_directory_noreplace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertEqual((destination / "owner").read_bytes(), b"foreign")

    def test_linux_nfs_reservation_is_made_inert_before_rename(self) -> None:
        runtime_source = RUNTIME.read_text(encoding="utf-8")
        self.assertIn(
            'inert_mode = 0o000 if sys.platform.startswith("linux") else 0o700',
            runtime_source,
        )
        self.assertLess(
            runtime_source.index("os.fchmod(reservation_fd, inert_mode)"),
            runtime_source.index("os.rename(", runtime_source.index("inert_mode =")),
        )

    def test_one_shot_text_encoder_is_released_before_training(self) -> None:
        self.assertIn("renderer.t5_text_encoder = None", self.trainer)
        self.assertIn("t5_released_after_one_frozen_embedding", self.trainer)
        self.assertNotIn("t5_moved_to_cpu_after_one_frozen_embedding", self.trainer)
        text_release = self.trainer.index("renderer.t5_text_encoder = None")
        first_control = self.trainer.index("for logical_arm in local_logical_arms:")
        self.assertLess(text_release, first_control)
        controls = self.trainer.index("for logical_arm in local_logical_arms:")
        backwards = self.trainer.index("for logical in prepared_arms:", controls)
        step = self.trainer.index("optimizer.step()", backwards)
        self.assertLess(controls, backwards)
        self.assertLess(backwards, step)
        self.assertEqual(self.trainer.count("optimizer.step()"), 1)
        self.assertNotIn("gathered[4]", self.trainer)
        self.assertIn('"logical_arm": logical.logical_arm', self.trainer)
        self.assertIn('"physical_dp_rank": contract.arm_index', self.trainer)

    def test_world4_receipt_records_cross_node_placement_and_dynamic_groups(self) -> None:
        for fragment in (
            '"nodes": nodes',
            '"local_world_size": contract.local_world_size',
            '"sp4_crosses_nodes": topology.sp_size > contract.local_world_size',
            '"preferred_world4_placement"',
            '"profile": topology.profile',
            '"world_size": topology.world_size',
            '"physical_data_parallel_size": topology.dp_size',
            '"sp_groups": [list(item) for item in topology.sp_group_ranks]',
            '"dp_groups": [list(item) for item in topology.dp_group_ranks]',
            '"logical_records": history',
        ):
            self.assertIn(fragment, self.trainer)
        self.assertIn(
            'RUN_RECEIPT_SCHEMA = "bernini-source-self-role-repaint-training-receipt-v3"',
            self.trainer,
        )
        self.assertIn(
            'HISTORY_SCHEMA = "bernini-source-self-role-repaint-step-history-v3"',
            self.trainer,
        )

    def test_clean_commit_local_import_closure_has_no_untracked_dependency(self) -> None:
        tracked = set(
            subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.splitlines()
        )
        planned_source_self = {
            "methods/bernini_action_editing/source_self_role_repaint.py",
            "methods/bernini_action_editing/source_self_runtime.py",
            "methods/bernini_action_editing/train_source_self_role_repaint.py",
            "methods/bernini_action_editing/tools/materialize_source_self_role_repaint.py",
        }
        pending = [CORE, RUNTIME, TRAINER, MATERIALIZER]
        visited: set[Path] = set()
        local_dependencies: set[Path] = set()
        while pending:
            path = pending.pop()
            if path in visited:
                continue
            visited.add(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    module_names.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    if node.module:
                        module_names.add(node.module)
                        if node.module == "tools":
                            module_names.update(
                                f"tools.{alias.name}" for alias in node.names
                            )
            for module_name in module_names:
                candidate = METHOD_ROOT / (module_name.replace(".", "/") + ".py")
                if candidate.is_file():
                    candidate = candidate.resolve()
                    local_dependencies.add(candidate)
                    pending.append(candidate)
        relative = {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            for path in local_dependencies | visited
        }
        untracked = sorted(relative - tracked - planned_source_self)
        self.assertEqual(untracked, [], f"untracked local import closure: {untracked}")
        self.assertNotIn(
            "methods/bernini_action_editing/train_ramp_c0.py", relative
        )
    def test_materializer_uses_exactly_six_raw_source_encodes(self) -> None:
        for fragment in (
            'SPEC_SCHEMA = "bernini-source-self-role-repaint-materialization-spec-v2"',
            'ROW_SCHEMA = "bernini-source-self-role-repaint-row-v2"',
            'RECEIPT_SCHEMA = "bernini-source-self-role-repaint-dataset-receipt-v2"',
            "independent_vae_encode_calls_per_row",
            '"all_six_calls_share_one_pinned_vae_identity": True',
            '"clean_target": "independent_pinned_vae_encode_of_raw_clean_source_rgb"',
            '"paired_dataset_accessed": False',
            '"prior_posterior_accessed": False',
            '"target_video_path_present": False',
            '"references_from_video_posterior_slice": False',
            '"edited_target_accessed": False',
            '"synthetic_edited_target_present": False',
            "if call_index != 6",
            "tuple(range(6))",
            '("clean_target_posterior_blob", rgb, LATENT_PHASES',
            "rgb[:, 0:1]",
            "rgb[:, 40:41]",
            "rgb[:, 80:81]",
        ):
            self.assertIn(fragment, self.materializer)
        self.assertNotIn("clean[:, :, 0]", self.materializer)
        self.assertNotIn("full644", self.materializer)
        self.assertNotIn("video_vae_latents", self.materializer)

    def test_v2_spec_has_only_raw_source_identity_fields(self) -> None:
        value = json.loads(self.spec)
        self.assertEqual(
            value["schema_version"],
            "bernini-source-self-role-repaint-materialization-spec-v2",
        )
        unsigned = dict(value)
        digest = unsigned.pop("spec_digest")
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
        for row in value["rows"]:
            self.assertEqual(
                set(row), {"iid", "source_video_path", "source_video_sha256"}
            )
        self.assertNotIn("target_video", self.spec)
        self.assertNotIn("parquet", self.spec)

    def test_adapter_checkpoint_has_strict_metadata_and_loader(self) -> None:
        for fragment in (
            "def safetensors_metadata",
            '"schema_version": SCHEMA_VERSION',
            '"block_indices_json"',
            '"conditional_base_rho_hex"',
            '"inference_requires_identical_conditional_base_rho": "true"',
            "def strict_load_source_self_adapter",
            "adapter tensor key closure differs",
            "adapter safetensors metadata differs",
            '"pre_post_stat_and_hash_stable": True',
        ):
            self.assertIn(fragment, self.core)
        self.assertIn("conditional_base_rho=conditional_base_rho", self.trainer)
        self.assertIn("conditional_base_rho=args.rho", self.trainer)

    def test_routes_cover_each_control_and_main_backward(self) -> None:
        for fragment in (
            "control_invocation = role.RouteInvocation(",
            "with adapter.route(control_invocation):",
            "main_invocation = role.RouteInvocation(",
            "with adapter.route(main_invocation):",
            "loss.backward()",
        ):
            self.assertIn(fragment, self.trainer)
        route_position = self.trainer.index("with adapter.route(main_invocation):")
        backward_position = self.trainer.index("loss.backward()", route_position)
        self.assertGreater(backward_position, route_position)

    def test_non_reentrant_checkpointing_replays_the_exact_route(self) -> None:
        for fragment in (
            "def checkpoint_route_context_fn()",
            "current is invocation",
            '"checkpoint recomputation entered a different role route"',
            '"context_fn": role.checkpoint_route_context_fn',
            '"checkpoint_context_fn_captures_exact_route_by_identity": True',
        ):
            self.assertIn(fragment, self.core + self.trainer)
        self.assertNotIn('gradient_checkpointing_kwargs={"use_reentrant": False}', self.trainer)

    def test_raw_source_and_adapter_have_pre_post_mutation_audits(self) -> None:
        for fragment in (
            "source_identity_before = _stat_identity(source_path)",
            "source_sha_before = file_sha256(source_path)",
            "source_identity_after_decode = _stat_identity(source_path)",
            "source_sha_after = file_sha256(source_path)",
        ):
            self.assertIn(fragment, self.materializer)
        for fragment in (
            "before_identity = _stat_identity(path)",
            "actual_sha = _file_sha256(path)",
            "after_read_identity = _stat_identity(path)",
            "after_read_sha = _file_sha256(path)",
        ):
            self.assertIn(fragment, self.core)


class SourceSelfRuntimeUnitTests(unittest.TestCase):
    def test_world8_rank_mapping_is_exact(self) -> None:
        contract = source_runtime.distributed_contract(
            {
                "WORLD_SIZE": "8",
                "RANK": "6",
                "LOCAL_RANK": "6",
                "LOCAL_WORLD_SIZE": "8",
            }
        )
        self.assertEqual(contract.arm_index, 1)
        self.assertEqual(contract.sp_rank, 2)
        self.assertIs(contract.topology, source_runtime.WORLD8_DP2_SP4)
        with self.assertRaises(source_runtime.SourceSelfRuntimeError):
            source_runtime.distributed_contract(
                {
                    "WORLD_SIZE": "4",
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "LOCAL_WORLD_SIZE": "4",
                }
            )

    def test_explicit_world4_accepts_preferred_two_by_two_and_test_one_by_four(self) -> None:
        topology = source_runtime.parallel_topology("world4-dp1-sp4")
        self.assertIs(topology, source_runtime.WORLD4_DP1_SP4)
        self.assertEqual(topology.sp_group_ranks, ((0, 1, 2, 3),))
        self.assertEqual(
            topology.dp_group_ranks, ((0,), (1,), (2,), (3,))
        )
        for rank in range(4):
            contract = source_runtime.distributed_contract(
                {
                    "WORLD_SIZE": "4",
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank % 2),
                    "LOCAL_WORLD_SIZE": "2",
                },
                topology=topology,
            )
            self.assertEqual(contract.arm_index, 0)
            self.assertEqual(contract.sp_rank, rank)
            self.assertEqual(contract.local_world_size, 2)
        one_node = source_runtime.distributed_contract(
            {
                "WORLD_SIZE": "4",
                "RANK": "3",
                "LOCAL_RANK": "3",
                "LOCAL_WORLD_SIZE": "4",
            },
            topology=topology,
        )
        self.assertEqual(one_node.sp_rank, 3)
        self.assertEqual(one_node.local_world_size, 4)

    def test_explicit_world4_rejects_ambiguous_rank_or_placement(self) -> None:
        topology = source_runtime.WORLD4_DP1_SP4
        for environment in (
            {
                "WORLD_SIZE": "4",
                "RANK": "2",
                "LOCAL_RANK": "1",
                "LOCAL_WORLD_SIZE": "2",
            },
            {
                "WORLD_SIZE": "4",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
            },
            {
                "WORLD_SIZE": "4",
                "RANK": "3",
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "3",
            },
        ):
            with self.assertRaises(source_runtime.SourceSelfRuntimeError):
                source_runtime.distributed_contract(
                    environment, topology=topology
                )
        with self.assertRaises(source_runtime.SourceSelfRuntimeError):
            source_runtime.distributed_contract(
                {
                    "WORLD_SIZE": "4",
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "LOCAL_WORLD_SIZE": "4",
                },
                topology=topology,
                allow_multinode_dp2_sp4=True,
            )

    def test_dp_all_reduce_is_structurally_guarded_for_dp1(self) -> None:
        source = RUNTIME.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "synchronize_gradients"
        )
        guards = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and ast.get_source_segment(source, node.test)
            == "topology.dp_size > 1"
        ]
        self.assertEqual(len(guards), 1)
        guarded = ast.get_source_segment(source, guards[0])
        assert guarded is not None
        self.assertIn("group=parallel.dp_group", guarded)
        self.assertIn("parameter.grad.div_(float(topology.dp_size))", guarded)
        self.assertNotIn("group=parallel.world_group", guarded)

    def test_atomic_receipt_and_bundle_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            artifact_names = (
                "adapter.safetensors",
                "optimizer.pt",
                "history.json",
            )
            for index, name in enumerate(artifact_names):
                (stage / name).write_bytes(
                    f"artifact-{index}".encode("ascii")
                )
            receipt = {
                "schema_version": "source-self-runtime-test-v1",
                "artifacts": {
                    name: source_runtime.file_sha256(stage / name)
                    for name in artifact_names
                },
            }
            receipt["receipt_digest"] = source_runtime.object_sha256(receipt)
            source_runtime.atomic_json(stage / "receipt.json", receipt)
            source_runtime.verify_staged_run_bundle(stage, receipt)
            (stage / "history.json").write_bytes(b"mutated")
            with self.assertRaises(source_runtime.SourceSelfRuntimeError):
                source_runtime.verify_staged_run_bundle(stage, receipt)


@unittest.skipIf(torch is None, "AUH vace tensor runtime is required")
class SourceSelfTensorTests(unittest.TestCase):
    def _transformer(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(16, 16)
                self.to_k = torch.nn.Linear(16, 16)
                self.to_v = torch.nn.Linear(16, 16)
                self.to_out = torch.nn.ModuleList(
                    [torch.nn.Linear(16, 16), torch.nn.Dropout(0.0)]
                )

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = Attention()
                self.attn2 = Attention()

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embedding = torch.nn.Conv3d(16, 16, (1, 2, 2), (1, 2, 2))
                self.blocks = torch.nn.ModuleList([Block() for _ in range(30)])

        value = Transformer()
        value.requires_grad_(False)
        return value

    def test_world4_serial_logical_arm_objective_matches_world8_dp2_mean(self) -> None:
        self.assertEqual(
            training.logical_arms_for_topology(
                source_runtime.WORLD8_DP2_SP4, 0
            ),
            (0,),
        )
        self.assertEqual(
            training.logical_arms_for_topology(
                source_runtime.WORLD8_DP2_SP4, 1
            ),
            (1,),
        )
        self.assertEqual(
            training.logical_arms_for_topology(
                source_runtime.WORLD4_DP1_SP4, 0
            ),
            (0, 1),
        )
        self.assertEqual(
            training.logical_loss_scale(source_runtime.WORLD8_DP2_SP4), 1.0
        )
        self.assertEqual(
            training.logical_loss_scale(source_runtime.WORLD4_DP1_SP4), 0.5
        )
        gradient0 = torch.tensor([1.0, -3.0, 5.0])
        gradient1 = torch.tensor([7.0, 1.0, -1.0])
        world8 = (gradient0 + gradient1) / 2.0
        world4 = 0.5 * gradient0 + 0.5 * gradient1
        self.assertTrue(torch.equal(world4, world8))

    def test_world4_two_by_two_placement_receipt_is_explicit(self) -> None:
        contract = source_runtime.distributed_contract(
            {
                "WORLD_SIZE": "4",
                "RANK": "2",
                "LOCAL_RANK": "0",
                "LOCAL_WORLD_SIZE": "2",
            },
            topology=source_runtime.WORLD4_DP1_SP4,
        )
        receipt = training.placement_receipt(contract)
        self.assertEqual(receipt["nodes"], 2)
        self.assertEqual(receipt["ranks_per_node"], 2)
        self.assertTrue(receipt["sp4_crosses_nodes"])
        self.assertTrue(receipt["preferred_world4_placement"])

    def test_conditional_base_rho0_is_exact_and_rho_positive_is_explicitly_non_gaussian(self) -> None:
        generator = torch.Generator().manual_seed(3)
        epsilon = torch.randn((2, 16, 21, 4, 4), generator=generator).contiguous()
        source = torch.randn((2, 16, 21, 4, 4), generator=generator).contiguous()
        exact, receipt0 = core.source_rich_conditional_base(epsilon, source, rho=0.0)
        self.assertIs(exact, epsilon)
        self.assertEqual(exact.data_ptr(), epsilon.data_ptr())
        self.assertTrue(receipt0["rho0_is_byte_alias_of_standard_gaussian"])
        mixed, receipt = core.source_rich_conditional_base(epsilon, source, rho=0.2)
        self.assertEqual(tuple(mixed.shape), tuple(epsilon.shape))
        self.assertTrue(bool(torch.isfinite(mixed).all()))
        self.assertFalse(receipt["gaussianity_claimed_for_rho_gt_zero"])
        self.assertTrue(receipt["gram_schmidt_verified_for_rho_gt_zero"])
        self.assertTrue(receipt["temporal_dc_carrier_verified_for_rho_gt_zero"])
        self.assertTrue(receipt["carrier_norm_match_verified"])
        self.assertTrue(receipt["realized_energy_preservation_verified"])
        self.assertLessEqual(receipt["max_absolute_carrier_epsilon_cosine"], 2.0e-5)
        self.assertLessEqual(receipt["max_observed_relative_carrier_norm_error"], 2.0e-5)
        self.assertLessEqual(receipt["max_observed_relative_energy_change"], 2.0e-5)
        self.assertEqual(receipt["max_observed_temporal_dc_error"], 0.0)
        epsilon_norm = epsilon.flatten(1).norm(dim=1)
        mixed_norm = mixed.flatten(1).norm(dim=1)
        self.assertTrue(torch.allclose(mixed_norm, epsilon_norm, rtol=2.0e-5, atol=2.0e-5))

    def test_target_row_q_and_o_delta_and_backward_are_zero_off_target(self) -> None:
        transformer = self._transformer()
        handle = core.install_source_self_adapter(transformer, rank=2, alpha=2.0)
        layout = core.TokenRoleLayout.contiguous(
            donor_tokens=4, reference_tokens=(1, 1, 1), target_tokens=5
        )
        invocation = core.RouteInvocation(layout, 0, 1)
        for wrapper in (handle.q_wrappers[0][1], handle.o_wrappers[0][1]):
            with torch.no_grad():
                wrapper.lora_a.weight.fill_(0.125)
                wrapper.lora_b.weight.fill_(0.25)
            hidden = torch.randn(
                (1, layout.total_tokens, 16), requires_grad=True
            )
            with handle.route(invocation):
                delta = wrapper.adapter_delta(hidden)
                delta.sum().backward()
            condition_delta = delta[:, : layout.condition_tokens]
            target_delta = delta[:, layout.condition_tokens :]
            self.assertTrue(torch.equal(condition_delta, torch.zeros_like(condition_delta)))
            self.assertGreater(float(target_delta.abs().sum()), 0.0)
            self.assertTrue(
                torch.equal(
                    hidden.grad[:, : layout.condition_tokens],
                    torch.zeros_like(hidden.grad[:, : layout.condition_tokens]),
                )
            )
            self.assertGreater(
                float(hidden.grad[:, layout.condition_tokens :].abs().sum()), 0.0
            )
        self.assertTrue(handle.base_parameters_frozen())

    def test_checkpoint_recompute_replays_route_after_outer_context_exits(self) -> None:
        transformer = self._transformer()
        handle = core.install_source_self_adapter(transformer, rank=2, alpha=2.0)
        q_wrapper = handle.q_wrappers[0][1]
        o_wrapper = handle.o_wrappers[0][1]
        with torch.no_grad():
            for wrapper in (q_wrapper, o_wrapper):
                wrapper.lora_a.weight.fill_(0.125)
                wrapper.lora_b.weight.fill_(0.25)
        layout = core.TokenRoleLayout.contiguous(
            donor_tokens=4, reference_tokens=(1, 1, 1), target_tokens=5
        )
        invocation = core.RouteInvocation(layout, 0, 1)
        seen = []

        class RoutedPair(torch.nn.Module):
            def forward(self, hidden):
                seen.append(core.active_route())
                return o_wrapper(torch.nn.functional.silu(q_wrapper(hidden)))

        hidden = torch.randn(
            (1, layout.total_tokens, 16), requires_grad=True
        )
        with handle.route(invocation):
            value = torch.utils.checkpoint.checkpoint(
                RoutedPair(),
                hidden,
                use_reentrant=False,
                context_fn=core.checkpoint_route_context_fn,
            )
        self.assertIsNone(core.active_route())
        value.square().mean().backward()
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(item is invocation for item in seen))
        for wrapper in (q_wrapper, o_wrapper):
            self.assertIsNotNone(wrapper.lora_a.weight.grad)
            self.assertIsNotNone(wrapper.lora_b.weight.grad)
        self.assertIsNone(core.active_route())

    def test_checkpoint_route_rejects_missing_or_different_invocation(self) -> None:
        layout = core.TokenRoleLayout.contiguous(
            donor_tokens=4, reference_tokens=(1, 1, 1), target_tokens=5
        )
        first = core.RouteInvocation(layout, 0, 1)
        second = core.RouteInvocation(layout, 0, 1)
        with self.assertRaisesRegex(core.SourceSelfRoleRepaintError, "without an active"):
            core.checkpoint_route_context_fn()
        with core.activate_route(first):
            _forward_context, recompute_context = core.checkpoint_route_context_fn()
        with core.activate_route(second):
            with self.assertRaisesRegex(core.SourceSelfRoleRepaintError, "different role"):
                with recompute_context:
                    pass

    def test_sp4_append_padding_and_target_selector(self) -> None:
        layout = core.TokenRoleLayout.contiguous(
            donor_tokens=4, reference_tokens=(1, 1, 1), target_tokens=6
        )
        invocations = [core.RouteInvocation(layout, rank, 4) for rank in range(4)]
        local = [
            invocation.local_roles(device=torch.device("cpu"))
            for invocation in invocations
        ]
        joined = torch.cat(local)
        expected = torch.tensor(
            (*layout.roles, core.ROLE_PADDING, core.ROLE_PADDING, core.ROLE_PADDING),
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(joined, expected))
        self.assertEqual([int(item.numel()) for item in local], [4, 4, 4, 4])
        self.assertEqual(int((joined == core.ROLE_TARGET).sum()), layout.target_tokens)
        self.assertEqual(int((joined == core.ROLE_PADDING).sum()), 3)
        transformer = self._transformer()
        handle = core.install_source_self_adapter(transformer, rank=2, alpha=2.0)
        wrapper = handle.q_wrappers[0][1]
        with torch.no_grad():
            wrapper.lora_a.weight.fill_(0.125)
            wrapper.lora_b.weight.fill_(0.25)
        for invocation, roles in zip(invocations, local):
            hidden = torch.randn((1, invocation.local_length, 16))
            with handle.route(invocation):
                delta = wrapper.adapter_delta(hidden)
            inactive = (roles != core.ROLE_TARGET).view(1, -1, 1).expand_as(delta)
            self.assertTrue(torch.equal(delta[inactive], torch.zeros_like(delta[inactive])))

    def test_absent_reference_condition_has_no_ref_tokens_or_source_ids(self) -> None:
        calls = []

        class FakeRope:
            def __call__(self, latent, *, source_id):
                calls.append(source_id)
                phases = int(latent.shape[2])
                tokens = phases * (int(latent.shape[3]) // 2) * (
                    int(latent.shape[4]) // 2
                )
                return torch.zeros((1, 2, tokens, 4), dtype=torch.float32)

        donor = torch.randn((16, 21, 4, 4), dtype=torch.float32).contiguous()
        noisy = torch.randn((16, 21, 4, 4), dtype=torch.float32).contiguous()
        condition = training._condition(
            donor, (), noisy, rope=FakeRope(), device=torch.device("cpu")
        )
        self.assertEqual(calls, [1, 0])
        self.assertEqual(condition.layout.reference_tokens, ())
        self.assertEqual(condition.layout.reference_token_total, 0)
        self.assertEqual(condition.layout.condition_tokens, condition.layout.donor_tokens)
        layout_receipt = condition.layout.receipt()
        self.assertFalse(layout_receipt["references_present"])
        self.assertEqual(layout_receipt["reference_rgb_indices"], [])
        self.assertNotIn(core.ROLE_REFERENCE, condition.layout.roles)

    def test_all30_receipt_marks_late_blocks_trainable(self) -> None:
        transformer = self._transformer()
        handle = core.install_source_self_adapter(
            transformer,
            rank=2,
            alpha=2.0,
            block_indices=tuple(range(core.TOTAL_BLOCKS_1P3B)),
        )
        receipt = handle.receipt()
        self.assertTrue(receipt["registered_all30_ablation"])
        self.assertTrue(receipt["late_blocks_trainable"])
        self.assertEqual(receipt["frozen_block_indices"], [])

    def test_strict_safetensors_roundtrip(self) -> None:
        transformer = self._transformer()
        handle = core.install_source_self_adapter(transformer, rank=2, alpha=2.0)
        with torch.no_grad():
            for index, (_, parameter) in enumerate(handle.trainable_named_parameters()):
                parameter.fill_(0.001 * (index + 1))
        expected = {
            name: parameter.detach().cpu().float().contiguous().clone()
            for name, parameter in handle.trainable_named_parameters()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file(
                expected,
                str(path),
                metadata=dict(
                    handle.safetensors_metadata(conditional_base_rho=0.25)
                ),
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            handle.restore()
            restored_transformer = self._transformer()
            loaded, receipt = core.strict_load_source_self_adapter(
                restored_transformer,
                path,
                expected_file_sha256=digest,
                expected_rho=0.25,
                rank=2,
                alpha=2.0,
            )
            actual = dict(loaded.trainable_named_parameters())
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                self.assertTrue(torch.equal(actual[name].detach().cpu(), expected[name]))
            self.assertTrue(receipt["strict_tensor_closure"])
            self.assertEqual(receipt["conditional_base_rho_hex"], float(0.25).hex())
            loaded.restore()
            with self.assertRaises(core.SourceSelfRoleRepaintError):
                core.strict_load_source_self_adapter(
                    self._transformer(),
                    path,
                    expected_file_sha256=digest,
                    expected_rho=0.0,
                    rank=2,
                    alpha=2.0,
                )


if __name__ == "__main__":
    unittest.main()
