#!/usr/bin/env python3
"""CPU-only tests for the ELAL-3 C2 staged training release/controller."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = REPO_ROOT / "methods" / "bernini_action_editing"
BUILDER_PATH = METHOD_ROOT / "tools" / "build_elal3_c2_role_binding_training_release_v1.py"
DEPLOY_PATH = METHOD_ROOT / "tools" / "control_elal3_c2_role_binding_training_v1.py"
GATE_PATH = METHOD_ROOT / "elal3_c2_staged_gate_controller_v1.py"
ORIGIN_VERIFIER_PATH = METHOD_ROOT / "elal3_c2_origin_receipt_verifier_v1.py"
LAUNCHER_PATH = METHOD_ROOT / "scripts" / "auh_run_elal3_c2_role_binding_stage_v1.sh"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load(BUILDER_PATH, "_test_elal3_c2_release_builder")
deploy = load(DEPLOY_PATH, "_test_elal3_c2_deployment_controller")
gate = load(GATE_PATH, "_test_elal3_c2_gate_controller")
origin = load(ORIGIN_VERIFIER_PATH, "_test_elal3_c2_origin_verifier")


PROJECTION_CONTROLS = (
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_simulator_optimizer_diagnostic_authority_v1.json",
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_real_model_authority_v1.json",
    "md/action_editing/20260817_box/evidence/"
    "elal3_c2_role_binding_experiment_contract_v1.json",
)
PROJECTION_SOURCE = "methods/bernini_action_editing/train_source.py"


def _projection_program() -> bytes:
    raw = LAUNCHER_PATH.read_text(encoding="utf-8")
    marker = "<<'PY_PROJECTION'\n"
    if raw.count(marker) != 1:
        raise AssertionError("launcher projection heredoc marker differs")
    program = raw.split(marker, 1)[1].split("\nPY_PROJECTION", 1)[0]
    return program.encode("utf-8")


def _projection_fixture(root: Path) -> tuple[Path, Path]:
    pristine = root / "source-archive"
    consumer = root / "source"
    payloads = {
        PROJECTION_CONTROLS[0]: b"derivative\n",
        PROJECTION_CONTROLS[1]: b"model-authority\n",
        PROJECTION_CONTROLS[2]: b"experiment-contract\n",
        PROJECTION_SOURCE: b"SOURCE = True\n",
    }
    for tree in (pristine, consumer):
        for relative, raw in payloads.items():
            path = tree / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o444)
        directories = sorted(
            (path for path in tree.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            directory.chmod(0o555)
        tree.chmod(0o555)
    return pristine, consumer


def _run_projection(pristine: Path, consumer: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-",
            str(pristine),
            str(consumer),
            *PROJECTION_CONTROLS,
        ],
        input=_projection_program(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _portable_checkpoint_tree(
    trainer: Any,
    *,
    steps: Sequence[int],
    parameter_shas: Sequence[str],
) -> Mapping[str, Any]:
    order = [f"block.{index}.lora_A.weight" for index in range(480)] + [
        f"block.{index}.elal3_c0_v1.weight" for index in range(188)
    ]
    inventory = [
        {"name": name, "shape": [1], "dtype": "torch.float32", "numel": 1}
        for name in order
    ]
    rows = []
    for step, parameter_sha in zip(steps, parameter_shas):
        names = ["adapter-and-elal3.pt"]
        if step:
            names.append("optimizer.pt")
        names.append("CHECKPOINT_RECEIPT.json")
        files = [
            {
                "name": name,
                "sha256": "a" * 64,
                "size": 1,
                "mode": 0o444,
                "nlink": 1,
                "held_fd_double_hash_verified": True,
                "named_identity_replayed": True,
            }
            for name in names
        ]
        optimizer_digest = "b" * 64 if step else None
        optimizer_inventory = (
            {
                "state_entry_count": 668,
                "param_group_count": 1,
                "parameter_count": 668,
                "parameter_inventory_digest": trainer.object_sha256(inventory),
                "optimizer_step": step,
                "exp_avg_nonzero_parameter_count": 668,
                "exp_avg_sq_nonzero_parameter_count": 668,
                "state_keys_by_parameter": [
                    {
                        "parameter_id": index,
                        "state_keys": ["exp_avg", "exp_avg_sq", "step"],
                    }
                    for index in range(668)
                ],
                "tree_digest": optimizer_digest,
            }
            if step
            else None
        )
        unsigned = {
            "schema_version": trainer.CHECKPOINT_SCHEMA,
            "step": step,
            "file_order": names,
            "directory_entries": names,
            "directory_mode": 0o500,
            "files": files,
            "adapter_payload_tree_digest": "c" * 64,
            "parameter_order": order,
            "parameter_inventory": inventory,
            "optimizer_payload_tree_digest": optimizer_digest,
            "optimizer_state_inventory": optimizer_inventory,
            "checkpoint_receipt_digest": "d" * 64,
            "trainable_parameter_sha256": parameter_sha,
            "strict_reload_pass": True,
        }
        rows.append(
            {**unsigned, "portable_record_digest": trainer.object_sha256(unsigned)}
        )
    return {
        "schema_version": "bernini-elal3-c2-sealed-checkpoint-tree-v1",
        "expected_steps": list(steps),
        "directory_entries": [f"checkpoint-{step:08d}" for step in steps],
        "directory_mode": 0o500,
        "portable_checkpoint_records": rows,
        "portable_checkpoint_tree_digest": trainer.object_sha256(rows),
        "physical_origin_replay_passed": True,
    }


def _fresh1_attestation(
    trainer: Any,
    *,
    arm_id: str,
    runner_sha: str,
    bundle_sha: str,
    source_pins: Mapping[str, Any],
    cross: Mapping[str, Any],
    origin_binding: Mapping[str, Any],
    gate_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    initial, final = "1" * 64, "2" * 64
    tree = _portable_checkpoint_tree(
        trainer, steps=(0, 1), parameter_shas=(initial, final)
    )
    job, node, seed = trainer.ARM_PLACEMENT[arm_id]
    unsigned = {
        "schema_version": trainer.FRESH1_ORIGIN_ATTESTATION_SCHEMA,
        "status": "FRESH1_ORIGIN_PHYSICAL_REPLAY_PASS",
        "stage": "fresh1",
        "arm_id": arm_id,
        "holder_job_id": job,
        "node": node,
        "seed": seed,
        "receipt_sha256": "3" * 64,
        "receipt_size": 1,
        "receipt_digest": "4" * 64,
        "initial_trainable_sha256": initial,
        "final_trainable_sha256": final,
        "common_comparison_payload_digest": cross[
            "common_comparison_payload_digest"
        ],
        "row_input_noise_schedule_digest": cross[
            "common_row_input_noise_schedule_digest"
        ],
        "history_digest": "5" * 64,
        "portable_checkpoint_tree": tree,
        "portable_checkpoint_tree_digest": tree[
            "portable_checkpoint_tree_digest"
        ],
        "cross_arm_gate_sha256": cross["gate_sha256"],
        "cross_arm_gate_digest": cross["gate_digest"],
        "cross_arm_recipe_version_digest": cross["recipe_version_digest"],
        "runner_source_sha256": runner_sha,
        "latent_bundle_sha256": bundle_sha,
        "source_pins": dict(source_pins),
        "experiment_contract_sha256": trainer.EXPERIMENT_CONTRACT_SHA256,
        "external_authority_sha256": trainer.EXTERNAL_AUTHORITY_SHA256,
        "model_authority_sha256": trainer.MODEL_AUTHORITY_SHA256,
        "materializer_run_complete_sha256": trainer.MATERIALIZER_RUN_COMPLETE_SHA256,
        "materializer_run_complete_digest": trainer.MATERIALIZER_RUN_COMPLETE_DIGEST,
        "checkpoint_exact23_binding_digest": "6" * 64,
        "bernini_execution_source_binding_digest": "7" * 64,
        "origin_verifier_binding": dict(origin_binding),
        "gate_controller_binding": dict(gate_binding),
        "physical_origin_replay_passed": True,
        "closed_validator_passed": True,
    }
    return {**unsigned, "attestation_digest": trainer.object_sha256(unsigned)}


class ELAL3C2RoleBindingReleaseTests(unittest.TestCase):
    def test_all_inline_node_receivers_compile(self) -> None:
        for name in ("ASSET_RECEIVER", "CONTROL_RECEIVER", "SEALED_READER"):
            compile(getattr(deploy, name), name, "exec")

    def test_frozen_release_literals_are_complete(self) -> None:
        gate.require_release_literals()
        deploy.require_release_literals()
        origin.require_release_literals()
        self.assertEqual(
            origin.TRAINER_SHA256,
            "63f35b39e60dbf2c1dd1dcecb29393c04d9f00fd0833054e7d81d40790dfe4ce",
        )
        builder._require_pins(
            runtime_pins=builder.RUNTIME_PINS,
            bundle_sha256=builder.LATENT_BUNDLE_SHA256,
            bundle_size=builder.LATENT_BUNDLE_SIZE,
            receipt_sha256=builder.LATENT_RECEIPT_SHA256,
            receipt_size=builder.LATENT_RECEIPT_SIZE,
            receipt_digest=builder.LATENT_RECEIPT_DIGEST,
            run_sha256=builder.MATERIALIZER_RUN_COMPLETE_SHA256,
            run_size=builder.MATERIALIZER_RUN_COMPLETE_SIZE,
            run_digest=builder.MATERIALIZER_RUN_COMPLETE_DIGEST,
        )

    def test_retry2_exact16_literals_are_identical_across_components(self) -> None:
        expected = "b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
        self.assertEqual(builder.LATENT_BUNDLE_SHA256, expected)
        self.assertEqual(gate.LATENT_BUNDLE_SHA256, expected)
        self.assertEqual(deploy.LATENT_BUNDLE_SHA256, expected)
        self.assertEqual(builder.LATENT_BUNDLE_SIZE, 78_277_976)
        self.assertEqual(gate.LATENT_BUNDLE_SIZE, 78_277_976)
        self.assertEqual(deploy.LATENT_BUNDLE_SIZE, 78_277_976)
        self.assertEqual(
            builder.LATENT_RECEIPT_SHA256,
            "a1ca0d3c015a54d61c8a71d00bc78688dab20d6592ba30ddf73b0ea18e7d70ee",
        )
        self.assertEqual(
            builder.MATERIALIZER_RUN_COMPLETE_SHA256,
            "c6eee4766943c7959a2c1ad9b8b6b4e823dec054b31d2fdfb5d03aacd9f7e1ac",
        )
        checkpoint_manifest = REPO_ROOT / builder.CHECKPOINT_EXACT23_RELATIVE
        self.assertEqual(checkpoint_manifest.stat().st_size, 2_350)
        self.assertEqual(
            hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest(),
            builder.CHECKPOINT_EXACT23_SHA256,
        )

    def test_assignments_and_stage_order_are_exact(self) -> None:
        self.assertEqual(
            deploy.PLACEMENTS,
            (
                ("A_duplicate_control", "141620", "auh7-1b-gpu-226", 20260821),
                ("B_paired_role", "141618", "auh7-1b-gpu-249", 20260821),
                ("B_paired_role_replica", "141619", "auh7-1b-gpu-257", 20260822),
            ),
        )
        self.assertEqual([row["holder_job_id"] for row in builder.RUN_ASSIGNMENTS], ["141620", "141618", "141619"])
        self.assertEqual([row["seed"] for row in builder.RUN_ASSIGNMENTS], [20260821, 20260821, 20260822])
        self.assertEqual(
            origin.ARM_PLACEMENT,
            {
                arm: (job, node, seed)
                for arm, job, node, seed in deploy.PLACEMENTS
            },
        )

    def test_origin_verifier_requires_the_exact_holder_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "141620", "HOSTNAME": "auh7-1b-gpu-226.cluster"},
            clear=False,
        ):
            self.assertEqual(
                origin.require_origin_placement("A_duplicate_control")["seed"],
                20260821,
            )
        with mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "141618", "HOSTNAME": "auh7-1b-gpu-226"},
            clear=False,
        ):
            with self.assertRaises(origin.ELAL3C2OriginVerifierError):
                origin.require_origin_placement("A_duplicate_control")

    def test_origin_receipt_reader_binds_outer_bytes_and_inner_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            unsigned = {"arm_id": "A_duplicate_control", "ok": True}
            digest = origin.object_digest(unsigned)
            value = {**unsigned, "receipt_digest": digest}
            raw = origin.canonical_json_bytes(value) + b"\n"
            path = root / "receipt.json"
            path.write_bytes(raw)
            os.chmod(path, 0o444)
            self.assertEqual(
                origin.held_sealed_json(
                    path,
                    expected_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_size=len(raw),
                    expected_self_digest=digest,
                    self_digest_key="receipt_digest",
                    label="origin receipt",
                ),
                value,
            )
            resigned = {**unsigned, "ok": False}
            resigned_digest = origin.object_digest(resigned)
            resigned_raw = origin.canonical_json_bytes(
                {**resigned, "receipt_digest": resigned_digest}
            ) + b"\n"
            hostile = root / "resigned.json"
            hostile.write_bytes(resigned_raw)
            os.chmod(hostile, 0o444)
            with self.assertRaises(origin.ELAL3C2OriginVerifierError):
                origin.held_sealed_json(
                    hostile,
                    expected_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_size=len(raw),
                    expected_self_digest=digest,
                    self_digest_key="receipt_digest",
                    label="origin receipt hostile",
                )

    def test_builder_closure_is_deterministic_with_injected_pending_sources(self) -> None:
        train_lora = Path("/tmp/elal3-c1-reviewed-train-lora-630c2152.py")
        if not train_lora.is_file():
            self.skipTest("reviewed remote train_lora copy is not present")
        runtime = dict(builder.RUNTIME_PINS)
        for relative in (
            "methods/bernini_action_editing/train_elal3_c2_simulator_role_pair_v1.py",
            "methods/bernini_action_editing/elal3_c2_staged_gate_controller_v1.py",
            "methods/bernini_action_editing/elal3_c2_origin_receipt_verifier_v1.py",
        ):
            raw = (REPO_ROOT / relative).read_bytes()
            runtime[relative] = (hashlib.sha256(raw).hexdigest(), len(raw))
        evidence = (
            REPO_ROOT
            / "md/action_editing/20260817_box/evidence/"
            "elal3_c2_exact16_materialization_r3_node226"
        )
        kwargs = {
            "latent_receipt_path": evidence / "latent-bundle-receipt.json",
            "materializer_run_complete_path": evidence / "RUN_COMPLETE.json",
            "train_lora_source_path": train_lora,
            "runtime_pins": runtime,
            "bundle_sha256": builder.LATENT_BUNDLE_SHA256,
            "bundle_size": builder.LATENT_BUNDLE_SIZE,
            "receipt_sha256": builder.LATENT_RECEIPT_SHA256,
            "receipt_size": builder.LATENT_RECEIPT_SIZE,
            "receipt_digest": builder.LATENT_RECEIPT_DIGEST,
            "run_sha256": builder.MATERIALIZER_RUN_COMPLETE_SHA256,
            "run_size": builder.MATERIALIZER_RUN_COMPLETE_SIZE,
            "run_digest": builder.MATERIALIZER_RUN_COMPLETE_DIGEST,
        }
        first_archive, first_manifest = builder.build_payload(REPO_ROOT, **kwargs)
        second_archive, second_manifest = builder.build_payload(REPO_ROOT, **kwargs)
        self.assertEqual(first_archive, second_archive)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["file_count"], 20)
        paths = {row["path"] for row in first_manifest["files"]}
        self.assertIn(
            "methods/bernini_action_editing/tools/materialize_vae.py", paths
        )
        self.assertIn(
            "methods/bernini_action_editing/tools/build_renderer_dataset.py",
            paths,
        )
        self.assertIn(
            "methods/bernini_action_editing/tools/__init__.py", paths
        )
        published = (
            REPO_ROOT
            / "md/action_editing/20260817_box/evidence/"
            "elal3_c2_role_binding_training_release_v10_retry2"
        )
        self.assertEqual((published / "source.tar").read_bytes(), first_archive)
        self.assertEqual(
            (published / "source.manifest.json").read_bytes(),
            builder.canonical_json_bytes(first_manifest) + b"\n",
        )
        self.assertEqual(
            deploy.ARCHIVE_SHA256,
            hashlib.sha256(first_archive).hexdigest(),
        )
        manifest_raw = builder.canonical_json_bytes(first_manifest) + b"\n"
        self.assertEqual(
            (deploy.MANIFEST_SHA256, deploy.MANIFEST_SIZE),
            (hashlib.sha256(manifest_raw).hexdigest(), len(manifest_raw)),
        )
        self.assertTrue(first_manifest["materializer_runtime_tree_reuse_forbidden"])
        self.assertEqual(
            first_manifest["authority_bindings"]["checkpoint_exact23_file_count"],
            23,
        )

    def test_srun_command_is_identity_bound_and_uses_overlap(self) -> None:
        self.assertEqual(
            deploy.srun_prefix("141620", "auh7-1b-gpu-226"),
            [
                "/usr/bin/srun",
                "--jobid=141620",
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                "--nodelist=auh7-1b-gpu-226",
                "--kill-on-bad-exit=1",
            ],
        )

    def test_asset_receiver_builds_a_separate_0444_runtime_tree(self) -> None:
        source_raw = b"VALUE = 1\n"
        source_name = "methods/bernini_action_editing/example.py"
        source_stream = io.BytesIO()
        with tarfile.open(
            fileobj=source_stream, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            archive.addfile(
                deploy._tar_info(source_name, len(source_raw)), io.BytesIO(source_raw)
            )
        source_archive = source_stream.getvalue()
        manifest_unsigned = {
            "schema_version": "bernini-elal3-c2-role-binding-training-release-v1",
            "archive_format": "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1",
            "archive_member_mode": "0444",
            "fresh_training_runtime_file_mode": "0444",
            "fresh_training_runtime_root_mode": "0555",
            "materializer_runtime_tree_reuse_forbidden": True,
            "archive_sha256": hashlib.sha256(source_archive).hexdigest(),
            "archive_size": len(source_archive),
            "file_count": 1,
            "files": [
                {
                    "path": source_name,
                    "sha256": hashlib.sha256(source_raw).hexdigest(),
                    "size": len(source_raw),
                    "mode": "0444",
                }
            ],
            "stage_sequence": [
                "exact3_preflight_no_update",
                "cross_arm_preflight_gate",
                "exact3_fresh1",
                "fresh1_acceptance_gate",
                "exact3_fresh_exact10",
            ],
            "gate_failure_stops_later_stages": True,
            "exact10_resume_from_fresh1_forbidden": True,
            "formal_c2_authorized": False,
            "exact160_authorized": False,
            "source_instruction_inference_authorized": False,
            "real_video_generalization_authorized": False,
            "scientific_claim_authorized": False,
        }
        manifest = {
            **manifest_unsigned,
            "manifest_digest": deploy.object_digest(manifest_unsigned),
        }
        manifest_raw = deploy.canonical_json_bytes(manifest) + b"\n"
        assets = {
            "assets/auh_run_elal3_c2_role_binding_stage_v1.sh": b"#!/bin/bash\n",
            "assets/c2-exact16-latents.safetensors": b"bundle",
            "assets/source.manifest.json": manifest_raw,
            "assets/source.tar": source_archive,
        }
        outer = io.BytesIO()
        with tarfile.open(fileobj=outer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(assets):
                archive.addfile(
                    deploy._tar_info(name, len(assets[name])), io.BytesIO(assets[name])
                )
        expected = {
            name: {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
            for name, raw in assets.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            node_root = Path(directory).resolve() / "fresh-node-root"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    deploy.ASSET_RECEIVER,
                    str(node_root),
                    deploy.canonical_json_bytes(expected).decode("ascii"),
                ],
                input=outer.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            runtime = node_root / "gate-runtime"
            extracted = runtime / source_name
            self.assertEqual(extracted.read_bytes(), source_raw)
            self.assertEqual(stat.S_IMODE(extracted.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(runtime.stat().st_mode), 0o555)
            self.assertNotEqual(runtime, node_root / "assets")
            collision = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    deploy.ASSET_RECEIVER,
                    str(node_root),
                    deploy.canonical_json_bytes(expected).decode("ascii"),
                ],
                input=outer.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(collision.returncode, 0)
            self.assertIn(b"not fresh", collision.stderr)

    def test_transport_tar_is_deterministic_sorted_0444(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            first = root / "first.bin"
            second = root / "second.bin"
            first.write_bytes(b"alpha")
            second.write_bytes(b"beta")
            rows = [
                ("controls/a.json", first, hashlib.sha256(b"alpha").hexdigest(), 5),
                ("controls/b.json", second, hashlib.sha256(b"beta").hexdigest(), 4),
            ]
            out1 = root / "one.tar"
            out2 = root / "two.tar"
            info1 = deploy.build_stream_tar(out1, rows)
            info2 = deploy.build_stream_tar(out2, rows)
            self.assertEqual(out1.read_bytes(), out2.read_bytes())
            self.assertEqual(info1["sha256"], info2["sha256"])
            with tarfile.open(out1, "r:") as archive:
                members = archive.getmembers()
            self.assertEqual([row.name for row in members], ["controls/a.json", "controls/b.json"])
            self.assertTrue(all(row.mode == 0o444 and row.uid == 0 and row.gid == 0 and row.mtime == 0 for row in members))

    def test_transport_rejects_unsorted_or_parent_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            item = root / "item"
            item.write_bytes(b"x")
            sha = hashlib.sha256(b"x").hexdigest()
            with self.assertRaises(deploy.ELAL3C2DeploymentError):
                deploy.build_stream_tar(root / "bad1.tar", [("b", item, sha, 1), ("a", item, sha, 1)])
            with self.assertRaises(deploy.ELAL3C2DeploymentError):
                deploy.build_stream_tar(root / "bad2.tar", [("../x", item, sha, 1)])

    def test_transport_binding_rejects_a_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            physical = root / "physical"
            physical.mkdir()
            item = physical / "item"
            item.write_bytes(b"x")
            linked = root / "linked"
            linked.symlink_to(physical, target_is_directory=True)
            with self.assertRaises((deploy.ELAL3C2DeploymentError, OSError)):
                deploy.stable_binding(
                    linked / "item",
                    expected_sha256=hashlib.sha256(b"x").hexdigest(),
                    expected_size=1,
                    expected_mode=None,
                    label="symlink-parent hostile",
                )

    def test_portable_control_rows_exclude_foreign_fresh1_receipts(self) -> None:
        row = {
            "path": Path("/physical/origin-only.json"),
            "sha256": "1" * 64,
            "size": 1,
        }
        per_arm = {arm: dict(row) for arm, *_ in deploy.PLACEMENTS}
        gate_row = dict(row)
        rows = deploy.control_rows(
            per_arm,
            cross_gate=gate_row,
            fresh1_gate=gate_row,
            fresh1_attestations=per_arm,
        )
        names = [name for name, *_ in rows]
        self.assertEqual(
            sum(name.startswith("controls/fresh1-attestations/") for name in names),
            3,
        )
        self.assertFalse(any(name.startswith("controls/fresh1/") for name in names))

    def test_control_receiver_creates_0444_and_replays_identical_existing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "node-root"
            root.mkdir()
            (root / "controls" / "preflight").mkdir(parents=True)
            raw = b'{"ok":true}\n'
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                info = deploy._tar_info("controls/preflight/A.json", len(raw))
                archive.addfile(info, io.BytesIO(raw))
            expected = {
                "controls/preflight/A.json": {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }
            }
            argv = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                deploy.CONTROL_RECEIVER,
                str(root),
                deploy.canonical_json_bytes(expected).decode("ascii"),
            ]
            for _ in range(2):
                completed = subprocess.run(argv, input=stream.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            target = root / "controls" / "preflight" / "A.json"
            self.assertEqual(target.read_bytes(), raw)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
            self.assertEqual(target.stat().st_nlink, 1)

    def test_control_receiver_rejects_different_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "node-root"
            target = root / "controls" / "preflight" / "A.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"wrong")
            os.chmod(target, 0o444)
            raw = b'{"ok":true}\n'
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                archive.addfile(deploy._tar_info("controls/preflight/A.json", len(raw)), io.BytesIO(raw))
            expected = {"controls/preflight/A.json": {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}}
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "-c", deploy.CONTROL_RECEIVER, str(root), deploy.canonical_json_bytes(expected).decode("ascii")],
                input=stream.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_control_receiver_rejects_a_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "node-root"
            root.mkdir()
            outside = base / "outside" / "preflight"
            outside.mkdir(parents=True)
            (root / "controls").symlink_to(outside.parent, target_is_directory=True)
            raw = b'{"ok":true}\n'
            stream = io.BytesIO()
            with tarfile.open(
                fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                archive.addfile(
                    deploy._tar_info("controls/preflight/A.json", len(raw)),
                    io.BytesIO(raw),
                )
            expected = {
                "controls/preflight/A.json": {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    deploy.CONTROL_RECEIVER,
                    str(root),
                    deploy.canonical_json_bytes(expected).decode("ascii"),
                ],
                input=stream.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((outside / "A.json").exists())

    def test_gate_held_fd_reader_requires_0444_and_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # macOS exposes tempfile paths through /var -> /private/var.  The
            # release reader intentionally rejects every symlinked parent, so
            # test the canonical physical path.
            path = Path(directory).resolve() / "receipt.json"
            value = {"a": 1, "b": True}
            raw = gate.canonical_json_bytes(value) + b"\n"
            path.write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            with self.assertRaises(gate.ELAL3C2GateControllerError):
                gate.read_sealed_json(path, expected_sha256=sha, label="test")
            os.chmod(path, 0o444)
            self.assertEqual(gate.read_sealed_json(path, expected_sha256=sha, label="test"), value)

    def test_gate_comparator_removes_only_final_whitelist(self) -> None:
        base = {
            "arm_id": "A",
            "branch_recipe": "a",
            "holder_job_id": "1",
            "node": "n1",
            "second_branch_descriptor": {"x": 1},
            "receipt_digest": "d",
            "actual_shape_preflight": {"same": 1, "runtime_telemetry": [1]},
            "step0_gain_safety": {"same": 2, "runtime_telemetry": [2]},
            "pre_publish_closure_replays": {
                "same": 3,
                "runtime_telemetry": {"path": "node-a"},
            },
            "strict": 3,
        }
        other = json.loads(gate.canonical_json_bytes(base))
        other.update({"arm_id": "B", "branch_recipe": "b", "holder_job_id": "2", "node": "n2", "second_branch_descriptor": {"x": 9}, "receipt_digest": "e"})
        other["actual_shape_preflight"]["runtime_telemetry"] = [9]
        other["step0_gain_safety"]["runtime_telemetry"] = [9]
        other["pre_publish_closure_replays"]["runtime_telemetry"] = {
            "path": "node-b"
        }
        self.assertEqual(gate._comparable_preflight(base), gate._comparable_preflight(other))
        other["strict"] = 4
        self.assertNotEqual(gate._comparable_preflight(base), gate._comparable_preflight(other))

    def test_gate_fresh1_portable_call_matches_real_trainer_abi(self) -> None:
        trainer = load(
            METHOD_ROOT / "train_elal3_c2_simulator_role_pair_v1.py",
            "_test_elal3_c2_real_trainer_gate_abi",
        )
        parameters = inspect.signature(
            trainer.validate_fresh1_origin_attestation_v1
        ).parameters
        required = {
            "expected_sha256",
            "arm_id",
            "expected_runner_sha256",
            "expected_bundle_sha256",
            "expected_source_pins",
            "cross_gate",
            "expected_origin_verifier_binding",
            "expected_gate_controller_binding",
        }
        self.assertTrue(required.issubset(parameters))
        controller_source = inspect.getsource(gate.build_fresh1_gate)
        for name in required:
            self.assertIn(f"{name}=", controller_source)
        self.assertNotIn("fresh1_receipt", controller_source)
        self.assertIn('"fresh1_origin_attestations": rows', controller_source)

    def test_real_trainer_portable_gate_survives_disjoint_roots_and_rejects_resign(self) -> None:
        trainer = load(
            METHOD_ROOT / "train_elal3_c2_simulator_role_pair_v1.py",
            "_test_elal3_c2_real_trainer_disjoint_portable",
        )
        runner_sha = hashlib.sha256(
            (METHOD_ROOT / "train_elal3_c2_simulator_role_pair_v1.py").read_bytes()
        ).hexdigest()
        bundle_sha = builder.LATENT_BUNDLE_SHA256
        source_pins = {"portable_fixture": True}
        cross = {
            "gate_sha256": "8" * 64,
            "gate_digest": "9" * 64,
            "recipe_version_digest": "a" * 64,
            "common_initial_trainable_sha256": "1" * 64,
            "common_row_input_noise_schedule_digest": "b" * 64,
            "common_comparison_payload_digest": "c" * 64,
        }
        origin_binding = {
            "name": deploy.ORIGIN_VERIFIER_BASENAME,
            "sha256": "d" * 64,
            "size": 11,
            "mode": 0o444,
            "nlink": 1,
        }
        controller_binding = {
            "name": deploy.GATE_CONTROLLER_BASENAME,
            "sha256": "e" * 64,
            "size": 12,
            "mode": 0o444,
            "nlink": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            rows = []
            origin_paths = []
            for index, arm_id in enumerate(trainer.ARM_IDS):
                node_root = root / f"disjoint-node-root-{index}"
                node_root.mkdir()
                value = _fresh1_attestation(
                    trainer,
                    arm_id=arm_id,
                    runner_sha=runner_sha,
                    bundle_sha=bundle_sha,
                    source_pins=source_pins,
                    cross=cross,
                    origin_binding=origin_binding,
                    gate_binding=controller_binding,
                )
                raw = trainer.canonical_json_bytes(value) + b"\n"
                path = node_root / "portable-attestation.json"
                path.write_bytes(raw)
                os.chmod(path, 0o444)
                sha = hashlib.sha256(raw).hexdigest()
                validated = trainer.validate_fresh1_origin_attestation_v1(
                    path,
                    expected_sha256=sha,
                    arm_id=arm_id,
                    expected_runner_sha256=runner_sha,
                    expected_bundle_sha256=bundle_sha,
                    expected_source_pins=source_pins,
                    cross_gate=cross,
                    expected_origin_verifier_binding=origin_binding,
                    expected_gate_controller_binding=controller_binding,
                )
                rows.append(
                    {
                        "arm_id": arm_id,
                        "attestation_sha256": sha,
                        "attestation_digest": validated["attestation_digest"],
                        "attestation": validated,
                    }
                )
                origin_paths.append(path)
            unsigned = {
                "schema_version": trainer.FRESH1_ACCEPTANCE_GATE_SCHEMA,
                "status": "FRESH1_ACCEPTANCE_GATE_PASS",
                "experiment_contract_sha256": trainer.EXPERIMENT_CONTRACT_SHA256,
                "external_authority_sha256": trainer.EXTERNAL_AUTHORITY_SHA256,
                "model_authority_sha256": trainer.MODEL_AUTHORITY_SHA256,
                "latent_bundle_sha256": bundle_sha,
                "runner_source_sha256": runner_sha,
                "source_pins": source_pins,
                "cross_arm_gate_sha256": cross["gate_sha256"],
                "cross_arm_gate_digest": cross["gate_digest"],
                "cross_arm_recipe_version_digest": cross["recipe_version_digest"],
                "origin_verifier_binding": origin_binding,
                "gate_controller_binding": controller_binding,
                "fresh1_origin_attestations": rows,
                "exact_fresh1_attestation_count": 3,
                "all_three_origin_physical_replays_passed": True,
                "exact10_resume_from_fresh1_forbidden": True,
            }
            gate_value = {
                **unsigned,
                "gate_digest": trainer.object_sha256(unsigned),
            }
            gate_raw = trainer.canonical_json_bytes(gate_value) + b"\n"
            gate_path = root / "central-portable-gate.json"
            gate_path.write_bytes(gate_raw)
            os.chmod(gate_path, 0o444)
            gate_sha = hashlib.sha256(gate_raw).hexdigest()
            for path in origin_paths:
                path.unlink()
                path.parent.rmdir()
            validated_gate = trainer.validate_fresh1_acceptance_gate_v1(
                gate_path,
                expected_sha256=gate_sha,
                expected_runner_sha256=runner_sha,
                expected_bundle_sha256=bundle_sha,
                expected_source_pins=source_pins,
                cross_gate=cross,
                expected_origin_verifier_binding=origin_binding,
                expected_gate_controller_binding=controller_binding,
            )
            self.assertTrue(
                validated_gate["all_three_portable_origin_attestations_replayed"]
            )

            hostile = json.loads(trainer.canonical_json_bytes(gate_value))
            bad = hostile["fresh1_origin_attestations"][1]["attestation"]
            bad["node"] = "auh7-1b-gpu-226"
            bad_unsigned = dict(bad)
            bad_unsigned.pop("attestation_digest")
            bad["attestation_digest"] = trainer.object_sha256(bad_unsigned)
            row = hostile["fresh1_origin_attestations"][1]
            row["attestation_digest"] = bad["attestation_digest"]
            row["attestation_sha256"] = hashlib.sha256(
                trainer.canonical_json_bytes(bad) + b"\n"
            ).hexdigest()
            hostile_unsigned = dict(hostile)
            hostile_unsigned.pop("gate_digest")
            hostile["gate_digest"] = trainer.object_sha256(hostile_unsigned)
            hostile_raw = trainer.canonical_json_bytes(hostile) + b"\n"
            hostile_path = root / "resigned-hostile-gate.json"
            hostile_path.write_bytes(hostile_raw)
            os.chmod(hostile_path, 0o444)
            with self.assertRaises(trainer.ELAL3C2TrainingError):
                trainer.validate_fresh1_acceptance_gate_v1(
                    hostile_path,
                    expected_sha256=hashlib.sha256(hostile_raw).hexdigest(),
                    expected_runner_sha256=runner_sha,
                    expected_bundle_sha256=bundle_sha,
                    expected_source_pins=source_pins,
                    cross_gate=cross,
                    expected_origin_verifier_binding=origin_binding,
                    expected_gate_controller_binding=controller_binding,
                )

    def test_origin_exact10_calls_real_final_closed_abi(self) -> None:
        trainer = load(
            METHOD_ROOT / "train_elal3_c2_simulator_role_pair_v1.py",
            "_test_elal3_c2_real_trainer_exact10_origin_abi",
        )
        required = {
            "expected_cross_gate_binding",
            "expected_fresh1_gate_binding",
            "expected_origin_verifier_binding",
            "expected_gate_controller_binding",
        }
        self.assertTrue(
            required.issubset(
                inspect.signature(
                    trainer._validate_exact10_origin_attestation_value_v1
                ).parameters
            )
        )
        self.assertTrue(
            {"expected_cross_gate_binding", "expected_fresh1_gate_binding"}.issubset(
                inspect.signature(
                    trainer.build_exact10_origin_attestation_v1
                ).parameters
            )
        )
        source = inspect.getsource(origin.build_origin_attestation)
        self.assertIn("trainer.build_exact10_origin_attestation_v1(", source)
        self.assertIn("trainer._validate_exact10_origin_attestation_value_v1(", source)
        for name in required:
            self.assertIn(f"{name}=", source)

    def test_gate_source_pin_envelope_matches_real_trainer_abi(self) -> None:
        trainer_path = METHOD_ROOT / "train_elal3_c2_simulator_role_pair_v1.py"
        trainer = load(trainer_path, "_test_elal3_c2_real_trainer_source_pin_abi")
        trainer_raw = trainer_path.read_bytes()
        rows = {
            "c2_trainer": (
                hashlib.sha256(trainer_raw).hexdigest(),
                len(trainer_raw),
            ),
            **{
                name: (sha, size)
                for name, (_relative, sha, size) in gate.STATIC_SOURCE_ROWS.items()
            },
        }
        source_closure = {
            "sources": {
                name: {
                    "path": f"/different/runtime/path/{name}.py",
                    "sha256": sha,
                    "size": size,
                    "mode": 0o444,
                    "nlink": 1,
                    "held_fd_double_hash_verified": True,
                    "held_openat_parent_chain_replayed": True,
                    "actual_imported_module_file_verified": True,
                }
                for name, (sha, size) in rows.items()
            },
            "callable_ownership_verified": True,
        }
        actual = trainer.source_pin_map_v1(source_closure)
        old_sha, old_size = gate.TRAINER_SHA256, gate.TRAINER_SIZE
        old_origin_sha = gate.ORIGIN_VERIFIER_SHA256
        old_origin_size = gate.ORIGIN_VERIFIER_SIZE
        try:
            gate.TRAINER_SHA256 = rows["c2_trainer"][0]
            gate.TRAINER_SIZE = rows["c2_trainer"][1]
            gate.ORIGIN_VERIFIER_SHA256 = "1" * 64
            gate.ORIGIN_VERIFIER_SIZE = 1
            expected = gate.source_pins()
        finally:
            gate.TRAINER_SHA256, gate.TRAINER_SIZE = old_sha, old_size
            gate.ORIGIN_VERIFIER_SHA256 = old_origin_sha
            gate.ORIGIN_VERIFIER_SIZE = old_origin_size
        self.assertEqual(actual, expected)
        self.assertTrue(actual["runtime_absolute_paths_devices_inodes_excluded"])

    def test_sealed_reader_rejects_symlink_and_accepts_held_0444(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "node"
            target = root / "runs" / "receipt.json"
            target.parent.mkdir(parents=True)
            raw = b'{"ok":true}\n'
            target.write_bytes(raw)
            os.chmod(target, 0o444)
            argv = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                deploy.SEALED_READER,
                str(root),
                str(target),
                "1024",
            ]
            completed = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(completed.stdout, raw)
            link = root / "runs" / "link.json"
            link.symlink_to(target)
            rejected = subprocess.run(
                [*argv[:-2], str(link), "1024"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_launcher_exact3_control_projection_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pristine, consumer = _projection_fixture(Path(temporary))
            completed = _run_projection(pristine, consumer)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            for relative in PROJECTION_CONTROLS:
                self.assertEqual(stat.S_IMODE((consumer / relative).stat().st_mode), 0o644)
                self.assertEqual(stat.S_IMODE((pristine / relative).stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE((consumer / PROJECTION_SOURCE).stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE((pristine / PROJECTION_SOURCE).stat().st_mode), 0o444)

    def test_launcher_rejects_fourth_consumer_file_chmod(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pristine, consumer = _projection_fixture(Path(temporary))
            (consumer / PROJECTION_SOURCE).chmod(0o644)
            completed = _run_projection(pristine, consumer)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"pre-projection source/archive file mode differs", completed.stderr)

    def test_launcher_rejects_projection_prebypass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pristine, consumer = _projection_fixture(Path(temporary))
            (consumer / PROJECTION_CONTROLS[0]).chmod(0o644)
            completed = _run_projection(pristine, consumer)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"pre-projection source/archive file mode differs", completed.stderr)

    def test_launcher_rejects_mutable_pristine_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pristine, consumer = _projection_fixture(Path(temporary))
            (pristine / PROJECTION_SOURCE).chmod(0o644)
            completed = _run_projection(pristine, consumer)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"pre-projection source/archive file mode differs", completed.stderr)

    def test_launcher_syntax_and_frozen_fail_closed_order(self) -> None:
        completed = subprocess.run(["/bin/bash", "-n", str(LAUNCHER_PATH)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        raw = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'expected_runner_sha256="63f35b39e60dbf2c1dd1dcecb29393c04d9f00fd0833054e7d81d40790dfe4ce"',
            raw,
        )
        self.assertIn(
            'expected_archive_sha256="e6ccc7c55c50d03d6df57cb8a9a3d85bb2dc1b0977ef1905105944757b720e61"',
            raw,
        )
        self.assertIn("--nproc-per-node=8", raw)
        self.assertIn("--preflight-only", raw)
        self.assertIn("--fresh1-acceptance-gate", raw)
        self.assertIn("--fresh1-origin-verifier-name", raw)
        self.assertIn("--expected-fresh1-origin-verifier-sha256", raw)
        self.assertIn("--expected-fresh1-origin-verifier-size", raw)
        self.assertIn("--fresh1-gate-controller-name", raw)
        self.assertIn("--expected-fresh1-gate-controller-sha256", raw)
        self.assertIn("--expected-fresh1-gate-controller-size", raw)
        self.assertIn("--checkpoint-exact23-manifest", raw)
        self.assertIn("--materializer-run-complete", raw)
        self.assertIn("fresh1_checkpoint_consumed", raw)
        self.assertIn("materializer", raw)
        self.assertIn("source-archive", raw)
        self.assertIn("os.fchmod(consumer_fd, 0o644)", raw)
        self.assertIn("consumer exact-three-only mode closure differs", raw)
        execution = subprocess.run(["/bin/bash", str(LAUNCHER_PATH)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(execution.returncode, 2)
        self.assertIn(b"ELAL3_C2_STAGE", execution.stderr)

    def test_controller_source_has_no_shared_vast_assumption(self) -> None:
        self.assertEqual(
            str(deploy.NODE_ROOT),
            "/tmp/elal3-c2-role-e6ccc7c5-v10",
        )
        self.assertEqual(deploy.NODE_ROOT.parts[1], "tmp")
        self.assertIn("stdin", deploy.__doc__ or "")
        self.assertIn("login_compute_shared_vast_assumed", DEPLOY_PATH.read_text(encoding="utf-8"))
        self.assertIn("--expected-controller-sha256", DEPLOY_PATH.read_text(encoding="utf-8"))
        self.assertIn("outer_controller_sha256", DEPLOY_PATH.read_text(encoding="utf-8"))
        self.assertIn("--expected-controller-source-sha256", GATE_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("fresh1_checkpoint", deploy.launch_stage_parallel.__code__.co_names)
        execute_source = inspect.getsource(deploy.execute)
        fresh_origin = execute_source.index('run_origin_verifiers(\n        "fresh1"')
        fresh_gate = execute_source.index('"fresh1_acceptance_gate.json"')
        exact10_origin = execute_source.index('run_origin_verifiers(\n        "exact10"')
        completion = execute_source.index('"exact3_origin_physical_postflight_pass"')
        self.assertLess(fresh_origin, fresh_gate)
        self.assertLess(fresh_gate, exact10_origin)
        self.assertLess(exact10_origin, completion)


if __name__ == "__main__":
    unittest.main()
