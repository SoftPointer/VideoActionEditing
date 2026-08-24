#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = REPO_ROOT / "methods" / "bernini_action_editing"
CONSUMER_PATH = METHOD_ROOT / "action_edit_checkpoint_consumer_0817_v1.py"
PRODUCT_PATH = METHOD_ROOT / "infer_action_edit_product_abi_0817_v1.py"
REAL_R2_EVIDENCE = (
    REPO_ROOT
    / "md"
    / "action_editing"
    / "20260817_man"
    / "evidence"
    / "pre_d0_paired2_edf3d1d2a77c_r2"
)
REAL_R2_RECEIPT_SHA256 = (
    "8014b7b71413318d80162fba12b73d83d6b9d9de5ea57ad295643a238b0f8c0e"
)


def load_subject(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


consumer = load_subject(CONSUMER_PATH, "action_edit_checkpoint_consumer_0817_v1_test")
product = load_subject(PRODUCT_PATH, "infer_action_edit_product_abi_0817_v1_test")

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> str:
    payload = consumer.canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return sha_bytes(payload)


def campaign_payload(checkpoint_paths: list[Path] | None = None) -> dict[str, object]:
    checkpoint_paths = checkpoint_paths or [
        Path(f"/tmp/checkpoint-{step:08d}") for step in range(3)
    ]
    dummy = [f"{index + 1:064x}" for index in range(16)]
    release = {
        "path": "/tmp/release/manifest.json",
        "sha256": consumer.PINNED_R2_RELEASE_MANIFEST_SHA256,
        "schema_version": consumer.RELEASE_MANIFEST_SCHEMA,
        "member_root": consumer.RELEASE_MEMBER_ROOT,
        "members": [],
        "member_count": 8,
        "member_set_sha256": consumer.PINNED_R2_RELEASE_MEMBER_SET_SHA256,
        "regular_non_symlink_exact_modes_sizes_hashes_verified": True,
    }
    unsigned: dict[str, object] = {
        "schema_version": consumer.TRAIN_RECEIPT_SCHEMA,
        "method": consumer.TRAIN_RUNNER_METHOD,
        "authority": consumer.AUTHORITY,
        "status": "complete_pre_d0_two_update_engineering_smoke",
        "complete": True,
        "promotable": False,
        "formal_training_started": False,
        "counts_as_d0": False,
        "counts_as_d1": False,
        "counts_as_d2": False,
        "scientific_claim_authorized": False,
        "action_quality_claim_authorized": False,
        "optimizer_steps": 2,
        "fresh_official_base": True,
        "resume_consumed": False,
        "checkpoint_steps": [0, 1, 2],
        "all_checkpoints_rank0_full_trainable_optimizer_roundtrip_reloaded": True,
        "all_checkpoints_all8_runtime_state_bytes_persisted": True,
        "all_checkpoints_rank0_runtime_state_roundtrip_reloaded": True,
        "terminal_world8_consensus_precedes_receipt_publication": True,
        "parent_allocation_released": False,
        "parameter_digests": {
            str(step): consumer.PINNED_R2_P_STATE_SHA256[step]
            for step in range(3)
        },
        "checkpoints": [
            {
                "step": step,
                "path": str(checkpoint_paths[step]),
                "adapter_sha256": dummy[3],
                "optimizer_sha256": dummy[4],
                "runtime_state_sha256": dummy[5],
                "metadata_sha256": dummy[6],
                "rank0_full_trainable_optimizer_roundtrip_reload_verified": True,
                "all8_runtime_state_bytes_persisted_verified": True,
                "rank0_runtime_state_roundtrip_reload_verified": True,
            }
            for step in range(3)
        ],
        "architecture": {
            "action_plan_predictor_source_sha256": (
                consumer.PINNED_PREDICTOR_SOURCE_SHA256
            ),
            "exact30_post_block_injection": True,
            "target_plan_input": False,
            "action_plan_state_dict_abi": {
                "abi_sha256": consumer.PINNED_CONDITIONER_ABI_SHA256
            },
        },
        "provenance": {
            "runner_source_sha256": consumer.PINNED_TRAIN_RUNNER_SHA256,
            "predictor_source_sha256": consumer.PINNED_PREDICTOR_SOURCE_SHA256,
            "release_closure": release,
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "fresh_state": True,
            "scheduler": "constant_lr_no_scheduler_object",
            "topology": "engineering_equivalent_replicated_not_formal_sharded",
        },
        "distributed": {
            "world_size": 8,
            "dp_size": 2,
            "sp_size": 4,
            "pre_sp_complete_source_predictor": True,
            "source_and_padding_bit_exact_under_injection": True,
            "checkpoint_forward_and_recompute_calls_per_block": 2,
        },
        "dataset": {
            "formal_0817_manifest_consumed": False,
            "effective_scientific_sample_size_claimed": False,
            "teacher_anchor_qualification_claimed": False,
        },
        "trainable_inventory_sha256": dummy[12],
    }
    return {**unsigned, "receipt_digest": consumer.object_sha256(unsigned)}


class CampaignReceiptTests(unittest.TestCase):
    def test_validates_p0_p1_p2_and_keeps_nonpromotable_authority(self) -> None:
        path = (REAL_R2_EVIDENCE / "receipt.json").resolve()
        receipt = consumer.validate_campaign_receipt(
            path, expected_sha256=consumer.PINNED_R2_CAMPAIGN_RECEIPT_SHA256
        )
        self.assertEqual(tuple(receipt.parameter_digests), (0, 1, 2))
        self.assertEqual(len(set(receipt.parameter_digests.values())), 3)
        self.assertFalse(receipt.raw["promotable"])
        self.assertFalse(receipt.raw["formal_training_started"])

    def test_rejects_p_state_alias_or_authority_promotion(self) -> None:
        for mutation in ("alias", "promote", "digest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                value = campaign_payload()
                if mutation == "alias":
                    value["parameter_digests"]["2"] = value["parameter_digests"]["1"]
                elif mutation == "promote":
                    value["promotable"] = True
                else:
                    value["receipt_digest"] = "0" * 64
                path = Path(temporary).resolve() / "receipt.json"
                sha = write_json(path, value)
                with self.assertRaises(consumer.CheckpointConsumerError):
                    consumer.validate_campaign_receipt(path, expected_sha256=sha)

    @unittest.skipUnless(
        (REAL_R2_EVIDENCE / "receipt.json").is_file(),
        "downloaded r2 evidence is not present in this checkout",
    )
    def test_replays_real_r2_receipt_and_all_three_metadata_schemas(self) -> None:
        receipt_path = (REAL_R2_EVIDENCE / "receipt.json").resolve()
        receipt = consumer.validate_campaign_receipt(
            receipt_path, expected_sha256=REAL_R2_RECEIPT_SHA256
        )
        self.assertEqual(
            receipt.parameter_digests,
            {
                0: "e26c5fd00a581e7710b60eef29a691763b03915ee73c25ffec82cb0bc8bba891",
                1: "d40391c7a2c9fa72e02b9dedc44f835b9eb3ce0b8f626cf0e36e576efb961970",
                2: "5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e",
            },
        )
        for step in (0, 1, 2):
            metadata_path = (
                REAL_R2_EVIDENCE
                / f"checkpoint-{step:08d}"
                / "metadata.json"
            ).resolve()
            self.assertEqual(
                consumer.file_sha256(metadata_path),
                receipt.checkpoint_records[step]["metadata_sha256"],
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["step"], step)
            self.assertEqual(
                metadata["parameter_sha256"], receipt.parameter_digests[step]
            )
            self.assertIsNone(
                consumer.validate_training_reference_checkpoint_binding(
                    metadata,
                    step=step,
                    parameter_sha256=receipt.parameter_digests[step],
                )
            )


class CheckpointPreflightTests(unittest.TestCase):
    def _materialize(self, root: Path):
        checkpoint_paths = [
            root / f"checkpoint-{step:08d}" for step in range(3)
        ]
        selected = checkpoint_paths[1]
        selected.mkdir()
        state_path = selected / "full_trainable_state.pt"
        optimizer_path = selected / "optimizer.pt"
        runtime_path = selected / "runtime_state.pt"
        state_path.write_bytes(b"state-bytes")
        optimizer_path.write_bytes(b"optimizer-bytes")
        runtime_path.write_bytes(b"runtime-bytes")

        campaign_value = campaign_payload(checkpoint_paths)
        record = campaign_value["checkpoints"][1]
        record["adapter_sha256"] = consumer.file_sha256(state_path)
        record["optimizer_sha256"] = consumer.file_sha256(optimizer_path)
        record["runtime_state_sha256"] = consumer.file_sha256(runtime_path)
        release = campaign_value["provenance"]["release_closure"]
        metadata = {
            "schema_version": consumer.TRAIN_RECEIPT_SCHEMA,
            "method": consumer.TRAIN_RUNNER_METHOD,
            "authority": consumer.AUTHORITY,
            "promotable": False,
            "formal_d0_dataset": False,
            "scientific_claim_authorized": False,
            "target_quality_qualified_for_0817": False,
            "fresh_official_base": True,
            "resume_consumed": False,
            "step": 1,
            "rank0_full_trainable_state_roundtrip_reload_verified": True,
            "rank0_optimizer_roundtrip_reload_verified": True,
            "all8_rng_sampler_scheduler_state_bytes_persisted_verified": True,
            "rank0_rng_state_roundtrip_reload_verified": True,
            "parameter_sha256": campaign_value["parameter_digests"]["1"],
            "roundtrip_parameter_sha256": campaign_value["parameter_digests"]["1"],
            "trainable_inventory_sha256": campaign_value[
                "trainable_inventory_sha256"
            ],
            "architecture": campaign_value["architecture"],
            "method_source_file_sha256": consumer.PINNED_TRAIN_RUNNER_SHA256,
            "release_closure": release,
            "adapter_file": state_path.name,
            "adapter_sha256": record["adapter_sha256"],
            "optimizer_file": optimizer_path.name,
            "optimizer_sha256": record["optimizer_sha256"],
            "runtime_state_file": runtime_path.name,
            "runtime_state_sha256": record["runtime_state_sha256"],
        }
        record["metadata_sha256"] = write_json(selected / "metadata.json", metadata)
        unsigned = dict(campaign_value)
        unsigned.pop("receipt_digest")
        campaign_value["receipt_digest"] = consumer.object_sha256(unsigned)
        campaign_path = root / "receipt.json"
        campaign_sha = write_json(campaign_path, campaign_value)
        campaign = consumer.CampaignReceipt(
            path=campaign_path,
            sha256=campaign_sha,
            raw=campaign_value,
            parameter_digests={
                step: campaign_value["parameter_digests"][str(step)]
                for step in range(3)
            },
            checkpoint_records={
                step: campaign_value["checkpoints"][step] for step in range(3)
            },
        )
        return selected, campaign, release["sha256"]

    def test_exact_four_file_checkpoint_is_bound_to_campaign_and_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selected, campaign, release_sha = self._materialize(root)
            result = consumer.validate_checkpoint_preflight(
                selected,
                step=1,
                campaign=campaign,
                expected_release_manifest_sha256=release_sha,
            )
            self.assertEqual(result.parameter_sha256, campaign.parameter_digests[1])
            self.assertEqual(result.step, 1)

    def test_rejects_file_bytes_or_extra_member_after_receipt(self) -> None:
        for mutation in ("bytes", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                selected, campaign, release_sha = self._materialize(root)
                if mutation == "bytes":
                    (selected / "full_trainable_state.pt").write_bytes(b"changed")
                else:
                    (selected / "unrecorded.bin").write_bytes(b"x")
                with self.assertRaises(consumer.CheckpointConsumerError):
                    consumer.validate_checkpoint_preflight(
                        selected,
                        step=1,
                        campaign=campaign,
                        expected_release_manifest_sha256=release_sha,
                    )


class ReleasePreflightTests(unittest.TestCase):
    def _release(self, root: Path) -> tuple[Path, str]:
        rows = []
        for relative in sorted(consumer.RELEASE_FILES_AND_MODES):
            source = METHOD_ROOT / relative
            target = root / relative
            shutil.copyfile(source, target)
            target.chmod(0o444)
            rows.append(
                {
                    "path": relative,
                    "mode": 0o444,
                    "size": target.stat().st_size,
                    "sha256": consumer.file_sha256(target),
                }
            )
        manifest = root / "manifest.json"
        sha = write_json(
            manifest,
            {
                "schema_version": consumer.RELEASE_MANIFEST_SCHEMA,
                "member_root": consumer.RELEASE_MEMBER_ROOT,
                "files": rows,
            },
        )
        return manifest, sha

    def test_authenticates_frozen_eight_member_release_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, sha = self._release(root)
            receipt = consumer.authenticate_release_before_import(
                manifest, expected_sha256=sha, method_root=root
            )
            self.assertEqual(len(receipt.members), 8)
            self.assertEqual(receipt.manifest_sha256, sha)

    def test_rejects_member_mode_or_bytes_drift(self) -> None:
        for mutation in ("mode", "bytes"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest, sha = self._release(root)
                target = root / "train_lora.py"
                if mutation == "mode":
                    target.chmod(0o644)
                else:
                    target.chmod(0o644)
                    target.write_bytes(target.read_bytes() + b"\n")
                    target.chmod(0o444)
                with self.assertRaises(consumer.CheckpointConsumerError):
                    consumer.authenticate_release_before_import(
                        manifest, expected_sha256=sha, method_root=root
                    )

    def test_authenticated_source_bytes_bypass_hostile_import_shadow(self) -> None:
        target_names = tuple(
            Path(filename).stem
            for filename in sorted(consumer.RELEASE_FILES_AND_MODES)
        )

        class HostileFinder:
            def __init__(self) -> None:
                self.seen = []

            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname in target_names:
                    self.seen.append(fullname)
                    raise AssertionError(f"hostile import shadow executed: {fullname}")
                return None

        saved = {name: sys.modules.pop(name) for name in target_names if name in sys.modules}
        saved_method_root = consumer.METHOD_ROOT
        finder = HostileFinder()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest, sha = self._release(root)
                release = consumer.authenticate_release_before_import(
                    manifest, expected_sha256=sha, method_root=root
                )
                cache = root / "__pycache__"
                cache.mkdir()
                (cache / "action_plan_predictor_v1.cpython-hostile.pyc").write_bytes(
                    b"hostile-pyc-must-not-be-read"
                )
                (root / "train_action_edit_large_lora_0817_v1.so").write_bytes(
                    b"hostile-extension-must-not-be-loaded"
                )
                consumer.METHOD_ROOT = root
                sys.meta_path.insert(0, finder)
                runner, predictor, closure = (
                    consumer.import_authenticated_training_modules(release)
                )
                late = {}
                for name in (
                    "inference_sigma_strata",
                    "clean_source_visual_context_stage_b_contract_v1",
                    "packed_preservation_lora_v2",
                    "packed_preservation_release_v2",
                    "source_self_runtime",
                    "train_lora",
                ):
                    late[name] = consumer._load_authenticated_release_module(
                        release, name
                    )
                self.assertEqual(
                    Path(runner.__file__),
                    root / "train_action_edit_large_lora_0817_v1.py",
                )
                self.assertEqual(
                    Path(predictor.__file__), root / "action_plan_predictor_v1.py"
                )
                self.assertEqual(closure["sha256"], sha)
                self.assertIs(
                    late["clean_source_visual_context_stage_b_contract_v1"].exact40,
                    late["inference_sigma_strata"],
                )
                imported = runner.validate_imported_release_modules(
                    closure,
                    {
                        "action_plan_predictor_v1.py": predictor,
                        "clean_source_visual_context_stage_b_contract_v1.py": late[
                            "clean_source_visual_context_stage_b_contract_v1"
                        ],
                        "inference_sigma_strata.py": late[
                            "inference_sigma_strata"
                        ],
                        "packed_preservation_lora_v2.py": late[
                            "packed_preservation_lora_v2"
                        ],
                        "packed_preservation_release_v2.py": late[
                            "packed_preservation_release_v2"
                        ],
                        "source_self_runtime.py": late["source_self_runtime"],
                        "train_action_edit_large_lora_0817_v1.py": runner,
                        "train_lora.py": late["train_lora"],
                    },
                    method_root=root,
                )
                self.assertTrue(
                    imported[
                        "resolved_under_method_root_and_rehashed_after_import"
                    ]
                )
                self.assertEqual(finder.seen, [])
        finally:
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
            for name in target_names:
                sys.modules.pop(name, None)
            sys.modules.update(saved)
            consumer.METHOD_ROOT = saved_method_root

    def test_public_fresh_entry_orders_auth_import_receipt_checkpoint_build(self) -> None:
        names = tuple(
            inspect.signature(
                consumer.consume_frozen_r2_world8_checkpoint
            ).parameters
        )
        self.assertEqual(
            names,
            (
                "release_manifest_path",
                "campaign_receipt_path",
                "checkpoint_dir",
                "checkpoint_step",
                "bernini_root",
                "veomni_root",
                "base_checkpoint",
                "checkpoint_content_manifest",
                "expected_consumer_source_sha256",
                "expected_product_source_sha256",
            ),
        )
        source = inspect.getsource(consumer.consume_frozen_r2_world8_checkpoint)
        positions = [
            source.index(fragment)
            for fragment in (
                "authenticate_release_before_import(",
                "import_authenticated_training_modules(",
                "validate_campaign_receipt(",
                "validate_checkpoint_preflight(",
                "build_and_load_fresh_world8_model(",
            )
        ]
        self.assertEqual(positions, sorted(positions))


class ParityAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameter_sha = "a" * 64
        self.load = {
            "authority": consumer.AUTHORITY,
            "promotable": False,
            "checkpoint_parameter_sha256": self.parameter_sha,
            "training_attached_reference_present": False,
            "training_attached_reference_absent": True,
            "training_attached_full_renderer_reference_absent": True,
            "training_to_fresh_forward_parity_verified": False,
            "fresh_a_b_parity_verified": False,
            "promotion_authorized": False,
            "training_attached_reference_binding": None,
        }

    def test_fresh_a_b_cannot_be_relabelled_training_attached(self) -> None:
        fresh = {
            "schema_version": consumer.FRESH_WORLD8_PARITY_SCHEMA,
            "authority": consumer.AUTHORITY,
            "promotable": False,
            "checkpoint_parameter_sha256": self.parameter_sha,
            "exact_or_bounded_parity_pass": True,
            "origin": "posthoc_two_independent_world8_fresh_consumer_launches",
            "os_process_independence_proven": True,
            "disjoint_object_and_parameter_storage_verified": True,
        }
        result = consumer.bind_parity_result(
            self.load, fresh_a_b=fresh, training_attached=None
        )
        self.assertTrue(result["fresh_a_b_parity_verified"])
        self.assertTrue(result["training_attached_reference_absent"])
        self.assertFalse(result["training_to_fresh_forward_parity_verified"])
        self.assertFalse(result["promotion_authorized"])

    def test_only_true_pre_save_origin_can_close_training_parity(self) -> None:
        reference = {
            "schema_version": consumer.TRAINING_REFERENCE_SCHEMA,
            "authority": consumer.AUTHORITY,
            "promotable": False,
            "checkpoint_parameter_sha256": self.parameter_sha,
            "origin": "training_process_pre_checkpoint_export",
            "fresh_consumer_parity_pass": True,
            "training_to_fresh_forward_parity_verified": True,
            "conditioner_cell_training_to_fresh_forward_parity_verified": True,
            "full_bernini_renderer_training_to_fresh_forward_parity_verified": False,
            "metadata_file_sha256": "b" * 64,
            "tensor_file_sha256": "c" * 64,
        }
        self.load["training_attached_reference_binding"] = {
            "metadata_file_sha256": "b" * 64,
            "tensor_file_sha256": "c" * 64,
        }
        result = consumer.bind_parity_result(
            self.load, fresh_a_b=None, training_attached=reference
        )
        self.assertTrue(result["training_to_fresh_forward_parity_verified"])
        self.assertFalse(result["training_attached_reference_absent"])
        self.assertFalse(result["promotable"])
        forged = {**reference, "origin": "posthoc_two_disjoint_fresh_object_instances"}
        with self.assertRaises(consumer.CheckpointConsumerError):
            consumer.bind_parity_result(
                self.load, fresh_a_b=None, training_attached=forged
            )


class WorldConsensusTests(unittest.TestCase):
    class Dist:
        def __init__(self, mismatch=False):
            self.mismatch = mismatch

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size(group=None):
            return 8

        @staticmethod
        def get_rank(group=None):
            return 0

        def all_gather_object(self, rows, local, group=None):
            for rank in range(8):
                rows[rank] = {
                    "world_rank": rank,
                    "receipt_sha256": (
                        "f" * 64
                        if self.mismatch and rank == 7
                        else local["receipt_sha256"]
                    ),
                    "rank_local": {
                        "fresh_process_session_id": f"{rank + 1:064x}"
                    },
                }

    def test_requires_exact_all8_receipt_consensus(self) -> None:
        result = consumer.world8_consensus_receipt(
            {"checkpoint": "P2"}, distributed_module=self.Dist(), group=object()
        )
        self.assertTrue(result["all8_exact_consensus"])
        self.assertEqual(result["rank_order"], list(range(8)))
        with self.assertRaises(consumer.CheckpointConsumerError):
            consumer.world8_consensus_receipt(
                {"checkpoint": "P2"},
                distributed_module=self.Dist(mismatch=True),
                group=object(),
            )

    def test_world8_process_sessions_and_two_launch_a_b_are_distinct(self) -> None:
        first_consensus = consumer.world8_consensus_receipt(
            {
                "checkpoint": "P2",
                "fresh_process_session_id": f"{1:064x}",
            },
            distributed_module=self.Dist(),
            group=object(),
            rank_local_fields=("fresh_process_session_id",),
        )
        fingerprint = {
            "schema_version": "bernini-action-edit-fixed-forward-fingerprint-v1",
            "tensor_set_sha256": "9" * 64,
        }

        def launch(offset: int):
            sessions = [f"{offset + rank:064x}" for rank in range(8)]
            return {
                "schema_version": consumer.CONSUMER_SCHEMA,
                "authority": consumer.AUTHORITY,
                "promotable": False,
                "promotion_authorized": False,
                "checkpoint_step": 2,
                "checkpoint_parameter_sha256": consumer.PINNED_R2_P_STATE_SHA256[2],
                "loaded_parameter_sha256": consumer.PINNED_R2_P_STATE_SHA256[2],
                "campaign_receipt_sha256": consumer.PINNED_R2_CAMPAIGN_RECEIPT_SHA256,
                "checkpoint_metadata_sha256": "3" * 64,
                "release_manifest_sha256": consumer.PINNED_R2_RELEASE_MANIFEST_SHA256,
                "runner_source_sha256": consumer.PINNED_TRAIN_RUNNER_SHA256,
                "predictor_source_sha256": consumer.PINNED_PREDICTOR_SOURCE_SHA256,
                "conditioner_state_abi_sha256": consumer.PINNED_CONDITIONER_ABI_SHA256,
                "consumer_source_sha256": "4" * 64,
                "product_bridge_source_sha256": "5" * 64,
                "fresh_process_session_id": sessions[0],
                "fresh_loaded_fixed_forward_executed": True,
                "fresh_loaded_fixed_forward_fingerprint": fingerprint,
                "world8_consumer_complete": True,
                "fresh_world8_process_forward_exact_consensus_verified": True,
                "fresh_world8_process_forward_scope": (
                    "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
                ),
                "full_bernini_renderer_forward_executed": False,
                "training_to_fresh_forward_parity_verified": False,
                "world8_consensus": {
                    "world_size": 8,
                    "all8_exact_consensus": True,
                    "eight_distinct_fresh_process_sessions": True,
                    "rank_local_fresh_process_sessions": sessions,
                },
            }

        self.assertTrue(first_consensus["eight_distinct_fresh_process_sessions"])
        parity = consumer.compare_fresh_world8_consumer_receipts(
            launch(100), launch(200)
        )
        self.assertTrue(parity["os_process_independence_proven"])
        self.assertFalse(parity["training_attached_reference"])


class ProductPureContractTests(unittest.TestCase):
    def test_public_product_boundary_has_no_target_anchor_teacher_argument(self) -> None:
        receipt = product.validate_public_product_signatures()
        self.assertTrue(receipt["source_and_instruction_required"])
        self.assertFalse(
            receipt["target_anchor_teacher_external_annotations_accepted"]
        )
        joined = " ".join(
            name
            for names in receipt["public_signatures"].values()
            for name in names
        ).lower()
        for forbidden in product.FORBIDDEN_PRODUCT_ARGUMENT_FRAGMENTS:
            self.assertNotIn(forbidden, joined)

    def test_inference_policy_is_exact_and_does_not_restore_training_state(self) -> None:
        policy = product.OfflineInferencePolicyV1(seed=170817)
        policy.validate()
        self.assertFalse(policy.training_rng_restored)
        self.assertFalse(policy.training_sampler_cursor_consumed)
        self.assertFalse(policy.training_scheduler_object_consumed)
        for kwargs in (
            {"num_inference_steps": 39},
            {"flow_shift": 4.0},
            {"training_rng_restored": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(product.ProductABIError):
                product.OfflineInferencePolicyV1(seed=170817, **kwargs).validate()

    def test_live_scheduler_replays_authenticated_exact40_float32_grid(self) -> None:
        sigma = load_subject(
            METHOD_ROOT / "inference_sigma_strata.py",
            "inference_sigma_strata_product_test",
        )

        class Device:
            type = "cpu"

        class Vector:
            def __init__(self, values, dtype):
                self.values = list(values)
                self.dtype = dtype
                self.device = Device()
                self.ndim = 1

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self.values)

        class Scheduler:
            def __init__(self):
                self.config = {
                    "_class_name": "UniPCMultistepScheduler",
                    "num_train_timesteps": 1000,
                    "flow_shift": 5.0,
                    "prediction_type": "flow_prediction",
                    "predict_x0": True,
                    "use_flow_sigmas": True,
                    "thresholding": False,
                    "solver_order": 2,
                    "solver_type": "bh2",
                    "final_sigmas_type": "zero",
                }

            def set_timesteps(self, count):
                if count != 40:
                    raise AssertionError(count)
                self.timesteps = Vector(sigma.PINNED_TIMESTEPS, "torch.int64")
                self.sigmas = Vector(
                    (*sigma.PINNED_POSITIVE_SIGMAS, 0.0), "torch.float32"
                )

        scheduler = Scheduler()
        receipt = product.audit_live_inference_scheduler(
            scheduler=scheduler,
            inference_policy=product.OfflineInferencePolicyV1(seed=7),
            sigma_contract_module=sigma,
            initialize=True,
        )
        self.assertEqual(
            receipt["schedule_sha256"], product.PINNED_UNIPC_SCHEDULE_SHA256
        )
        self.assertTrue(receipt["exact_runtime_schedule_verified"])
        scheduler.config["flow_shift"] = 4.0
        with self.assertRaises(product.ProductABIError):
            product.audit_live_inference_scheduler(
                scheduler=scheduler,
                inference_policy=product.OfflineInferencePolicyV1(seed=7),
                sigma_contract_module=sigma,
                initialize=True,
            )

    def test_external_request_is_exactly_source_video_instruction_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve() / "source.mp4"
            source.write_bytes(b"not-decoded-in-pure-contract-test")
            request = product.ProductRequestV1(
                source_video_path=str(source),
                expected_source_video_sha256=product.file_sha256(source),
                instruction="Make the person jump over the box.",
                inference_policy=product.OfflineInferencePolicyV1(seed=170817),
            )
            receipt = request.validate()
        self.assertFalse(receipt["clean_target_present"])
        self.assertFalse(receipt["anchor_present"])
        self.assertFalse(receipt["teacher_or_external_annotation_present"])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve() / "source.mp4"
            source.write_bytes(b"x")
            with self.assertRaises(product.ProductABIError):
                product.ProductRequestV1(
                    source_video_path=str(source),
                    expected_source_video_sha256="0" * 64,
                    instruction="edit",
                    inference_policy=product.OfflineInferencePolicyV1(seed=1),
                ).validate()

    def test_nonpromotable_product_receipt_preserves_r2_reference_absence(self) -> None:
        consensus = {
            "world_size": 8,
            "all8_exact_consensus": True,
            "eight_distinct_fresh_process_sessions": True,
        }
        checkpoint = {
            "schema_version": consumer.CONSUMER_SCHEMA,
            "authority": product.AUTHORITY,
            "promotable": False,
            "promotion_authorized": False,
            "world8_consumer_complete": True,
            "fresh_world8_process_forward_exact_consensus_verified": True,
            "fresh_world8_process_forward_scope": (
                "conditioner_predictor_plus_exact30_cell_only_not_bernini_renderer"
            ),
            "full_bernini_renderer_forward_executed": False,
            "checkpoint_parameter_sha256": "1" * 64,
            "loaded_parameter_sha256": "1" * 64,
            "training_attached_reference_absent": True,
            "training_attached_full_renderer_reference_absent": True,
            "training_to_fresh_forward_parity_verified": False,
            "world8_consensus": consensus,
        }
        routes = [
            {
                "row_identity": "product-row",
                "source_tokens": 21,
                "inference_tokens": 21,
                "sequence_parallel_rank": 0,
                "sequence_parallel_size": 1,
                "local_phase_indices_sha256": "2" * 64,
                "exact_block_indices": list(range(30)),
                "clean_target_present": False,
                "anchor_present": False,
                "inference_step_index": index,
                "scheduler_schedule_sha256": (
                    product.PINNED_UNIPC_SCHEDULE_SHA256
                ),
                "scheduler_timestep_bound_to_step": True,
                "runtime_timestep": product.PINNED_UNIPC_TIMESTEPS[index],
                "live_scheduler_verified_before_denoiser": True,
            }
            for index in range(40)
        ]
        receipt = product.build_nonpromotable_product_receipt(
            checkpoint_consumer_receipt=checkpoint,
            inference_policy=product.OfflineInferencePolicyV1(seed=170817),
            route_receipts=routes,
            checkpoint_world8_consensus=consensus,
        )
        self.assertFalse(receipt["promotable"])
        self.assertTrue(receipt["training_attached_reference_absent"])
        self.assertTrue(receipt["training_attached_full_renderer_reference_absent"])
        self.assertFalse(receipt["clean_target_or_anchor_consumed"])
        self.assertTrue(receipt["engineering_bridge_smoke_only"])
        self.assertFalse(receipt["materialized_product_request_bound"])
        self.assertIn("callback_bridge_smoke", receipt["completion_scope"])
        self.assertFalse(checkpoint["full_bernini_renderer_forward_executed"])
        self.assertFalse(receipt["offline_product_inference_completed"])
        self.assertFalse(receipt["mp4_emitted"])
        unsigned = dict(receipt)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(digest, product.object_sha256(unsigned))
        drifted = copy.deepcopy(routes)
        drifted[7]["runtime_timestep"] += 1
        with self.assertRaises(product.ProductABIError):
            product.build_nonpromotable_product_receipt(
                checkpoint_consumer_receipt=checkpoint,
                inference_policy=product.OfflineInferencePolicyV1(seed=170817),
                route_receipts=drifted,
                checkpoint_world8_consensus=consensus,
            )


@unittest.skipIf(torch is None, "PyTorch is unavailable in this local workspace")
class ProductTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        import action_plan_predictor_v1 as predictor

        cls.predictor = predictor

    def conditioner(self):
        config = self.predictor.ActionPlanPredictorConfig(
            profile=self.predictor.CPU_TEST_PROFILE,
            source_token_width=12,
            instruction_token_width=16,
            model_width=16,
            attention_heads=4,
            mlp_width=32,
            layer_count=2,
        )
        return self.predictor.ActionPlanConditionerV1(
            config, renderer_hidden_width=8
        )

    def encoder_receipt(self):
        return {
            "schema_version": "bernini-frozen-source-instruction-encoders-v1",
            "vae_source_code_sha256": "1" * 64,
            "vae_weights_sha256": "2" * 64,
            "t5_tokenizer_code_sha256": "3" * 64,
            "t5_encoder_code_sha256": "4" * 64,
            "t5_weights_sha256": "5" * 64,
            "noise_factory_code_sha256": "6" * 64,
            "vae_frozen": True,
            "t5_frozen": True,
            "source_token_width": 12,
            "instruction_token_width": 16,
            "source_preprocessing": (
                "exact81_rgb_to_normalized_clean_vae_patch_tokens"
            ),
            "instruction_preprocessing": (
                "complete_unpadded_contextual_frozen_t5_tokens"
            ),
            "noise_factory_semantics": (
                "counter_based_torch_Generator_cpu_no_global_rng_mutation"
            ),
        }

    def test_materializes_only_source_instruction_and_counter_based_noise(self) -> None:
        conditioner = self.conditioner()
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary).resolve() / "source.mp4"
            source_path.write_bytes(b"source-video-bytes")
            request = product.ProductRequestV1(
                source_video_path=str(source_path),
                expected_source_video_sha256=product.file_sha256(source_path),
                instruction="Make the subject wave.",
                inference_policy=product.OfflineInferencePolicyV1(seed=817),
            )

            def source_encoder(_path):
                return torch.linspace(-1.0, 1.0, 21 * 12).reshape(
                    1, 21, 1, 1, 12
                )

            def instruction_encoder(_instruction):
                return torch.linspace(-0.5, 0.5, 3 * 16).reshape(1, 3, 16)

            def isolated_noise(source, policy):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(policy.seed)
                return torch.randn(
                    1, 21, 1, 8,
                    generator=generator,
                    dtype=source.dtype,
                )

            inputs, receipt = product.materialize_product_request(
                request=request,
                conditioner=conditioner,
                source_video_to_clean_source_tokens=source_encoder,
                instruction_to_contextual_tokens=instruction_encoder,
                inference_noise_factory=isolated_noise,
                encoder_receipt=self.encoder_receipt(),
                torch_module=torch,
            )
            self.assertEqual(tuple(inputs.clean_source_tokens.shape), (1, 21, 1, 1, 12))
            self.assertTrue(receipt["source_video_plus_instruction_only"])
            self.assertFalse(receipt["clean_target_or_anchor_consumed"])

            def ambient_noise(source, _policy):
                return torch.randn(1, 21, 1, 8, dtype=source.dtype)

            with self.assertRaises(product.ProductABIError):
                product.materialize_product_request(
                    request=request,
                    conditioner=conditioner,
                    source_video_to_clean_source_tokens=source_encoder,
                    instruction_to_contextual_tokens=instruction_encoder,
                    inference_noise_factory=ambient_noise,
                    encoder_receipt=self.encoder_receipt(),
                    torch_module=torch,
                )

    def test_two_independent_fresh_instances_have_exact_fixed_forward_parity(self) -> None:
        torch.manual_seed(817)
        left = self.conditioner()
        right = self.conditioner()
        right.load_state_dict(copy.deepcopy(left.state_dict()), strict=True)
        receipt = product.compare_fresh_a_b(
            conditioner_a=left,
            conditioner_b=right,
            predictor_module=self.predictor,
            torch_module=torch,
            checkpoint_parameter_sha256="1" * 64,
        )
        self.assertTrue(receipt["exact_parity"])
        self.assertFalse(receipt["training_attached_reference"])
        self.assertFalse(receipt["training_to_fresh_forward_parity_claimed"])

    def test_training_attached_writer_and_fresh_verifier_are_distinct(self) -> None:
        torch.manual_seed(818)
        training = self.conditioner()
        fresh = self.conditioner()
        fresh.load_state_dict(copy.deepcopy(training.state_dict()), strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            writer = product.write_training_attached_fixed_forward_reference(
                output_dir=root,
                conditioner=training,
                predictor_module=self.predictor,
                torch_module=torch,
                checkpoint_step=2,
                checkpoint_parameter_sha256="2" * 64,
                execution_phase="immediately_before_save_checkpoint",
            )
            binding = product.training_reference_checkpoint_binding(writer)
            verified = product.verify_training_attached_reference(
                reference_metadata_path=root / writer["metadata_file"],
                checkpoint_binding=binding,
                conditioner=fresh,
                predictor_module=self.predictor,
                torch_module=torch,
                checkpoint_parameter_sha256="2" * 64,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            )
        self.assertEqual(
            verified["origin"], "training_process_pre_checkpoint_export"
        )
        self.assertTrue(verified["training_to_fresh_forward_parity_verified"])
        self.assertFalse(verified["promotable"])

    def test_future_full_renderer_reference_callback_protocol_synthetic_roundtrip(self) -> None:
        torch.manual_seed(820)
        training = self.conditioner()
        fresh = self.conditioner()
        fresh.load_state_dict(copy.deepcopy(training.state_dict()), strict=True)
        training_renderer = torch.nn.Linear(8, 8, bias=False)
        fresh_renderer = torch.nn.Linear(8, 8, bias=False)
        fresh_renderer.load_state_dict(
            copy.deepcopy(training_renderer.state_dict()), strict=True
        )
        parameter_sha = "8" * 64
        callback_source_sha = product.file_sha256(Path(__file__).resolve())
        contract = {
            "schema_version": product.FULL_RENDERER_CALLBACK_CONTRACT_SCHEMA,
            "training_callback_source_sha256": callback_source_sha,
            "fresh_callback_source_sha256": callback_source_sha,
            "input_semantics": (
                "deterministic_source_instruction_inference_noise_embeddings_v1"
            ),
            "model_surface": (
                "full_persisted_bernini_renderer_plus_ActionPlanConditionerV1"
            ),
            "forward_coordinate": "one_fixed_denoiser_step_pre_decode",
            "source_instruction_only": True,
            "clean_target_or_anchor_consumed": False,
            "exact30_injection_required": True,
        }

        def callback(conditioner, renderer, role):
            def run():
                with torch.no_grad():
                    fixed = product.fixed_forward_tensors(
                        conditioner=conditioner,
                        predictor_module=self.predictor,
                        torch_module=torch,
                    )
                    output = renderer(fixed["conditioned_inference_hidden"])
                return {
                    "schema_version": product.FULL_RENDERER_CALLBACK_RESULT_SCHEMA,
                    "tensors": {"denoiser.hidden": output},
                    "execution_receipt": {
                        "schema_version": (
                            "bernini-action-edit-full-renderer-forward-execution-v1"
                        ),
                        "callback_role": role,
                        "callback_source_sha256": callback_source_sha,
                        "checkpoint_parameter_sha256": parameter_sha,
                        "source_instruction_inference_noise_only": True,
                        "clean_target_or_anchor_consumed": False,
                        "same_persisted_conditioner_and_30_heads": True,
                        "persisted_trainable_bytes_unchanged": True,
                        "exact_block_indices": list(range(30)),
                        "full_bernini_renderer_forward_executed": True,
                        "forward_coordinate": "one_fixed_denoiser_step_pre_decode",
                        "model_mode": (
                            "eval_fixed_forward_with_training_state_bytes"
                        ),
                    },
                }

            return run

        forged_contract = dict(contract)
        forged_contract["training_callback_source_sha256"] = "9" * 64
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(product.ProductABIError):
                product.write_training_attached_full_renderer_reference(
                    output_dir=Path(temporary).resolve(),
                    full_forward_callback=callback(
                        training, training_renderer, "training_pre_save"
                    ),
                    callback_contract=forged_contract,
                    torch_module=torch,
                    checkpoint_step=10,
                    checkpoint_parameter_sha256=parameter_sha,
                    execution_phase="immediately_before_save_checkpoint",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            writer = product.write_training_attached_full_renderer_reference(
                output_dir=root,
                full_forward_callback=callback(
                    training, training_renderer, "training_pre_save"
                ),
                callback_contract=contract,
                torch_module=torch,
                checkpoint_step=10,
                checkpoint_parameter_sha256=parameter_sha,
                execution_phase="immediately_before_save_checkpoint",
            )
            binding = product.full_renderer_reference_checkpoint_binding(writer)
            verified = product.verify_training_attached_full_renderer_reference(
                reference_metadata_path=root / writer["metadata_file"],
                checkpoint_binding=binding,
                fresh_full_forward_callback=callback(
                    fresh, fresh_renderer, "fresh_consumer"
                ),
                expected_callback_contract=contract,
                torch_module=torch,
                checkpoint_parameter_sha256=parameter_sha,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            )
        self.assertTrue(
            verified[
                "full_bernini_renderer_training_to_fresh_forward_parity_verified"
            ]
        )
        self.assertFalse(
            verified[
                "conditioner_cell_training_to_fresh_forward_parity_verified"
            ]
        )
        self.assertTrue(
            verified["callback_protocol_requires_future_runner_release_pin"]
        )
        self.assertFalse(verified["offline_full40_product_inference_completed"])
        self.assertFalse(verified["promotable"])

    def test_exact30_hook_runtime_changes_only_inference_suffix(self) -> None:
        torch.manual_seed(819)
        conditioner = self.conditioner()
        # Make the action branch observably nonzero while retaining the exact
        # persisted module implementation.
        with torch.no_grad():
            for projection in conditioner.injection.projections:
                projection.weight.fill_(0.001)
                projection.bias.fill_(0.002)

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(30)]
                )

            def forward(self, value):
                for block in self.blocks:
                    value = block(value)
                return value

        transformer = Transformer()
        hooks = product.install_offline_action_plan_hooks(
            transformer=transformer,
            conditioner=conditioner,
            torch_module=torch,
        )
        inputs = product.deterministic_fixed_inputs(
            conditioner=conditioner, torch_module=torch, device=torch.device("cpu")
        )
        route_plan = product.prepare_product_action_route(
            conditioner=conditioner,
            inputs=inputs,
            predictor_module=self.predictor,
            torch_module=torch,
        )
        route = product.OfflineLocalActionRoute(
            plan=route_plan,
            source_tokens=21,
            inference_tokens=21,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
            row_identity="cpu-product-row",
        )

        class SigmaContract:
            SCHEDULE_SHA256 = product.PINNED_UNIPC_SCHEDULE_SHA256
            SCHEDULER_CLASS = "UniPCMultistepScheduler"
            NUM_INFERENCE_STEPS = 40
            FLOW_SHIFT = 5.0

            @staticmethod
            def audit_runtime_unipc_schedule(scheduler, initialize=True):
                del scheduler, initialize
                return {
                    "schedule_sha256": product.PINNED_UNIPC_SCHEDULE_SHA256,
                    "timesteps": list(product.PINNED_UNIPC_TIMESTEPS),
                    "positive_sigmas_float32_be_hex": ["00000001"] * 40,
                    "terminal_sigma_float32_be_hex": "00000000",
                }

        source = torch.linspace(-0.5, 0.5, 21 * 8).reshape(1, 21, 8)
        noise = inputs.inference_noise_hidden.reshape(1, 21, 8)
        packed = torch.cat((source, noise), dim=1)
        output, route_receipt = product.execute_offline_denoiser_step(
            route=route,
            denoiser_step=transformer,
            denoiser_kwargs={"value": packed},
            inference_policy=product.OfflineInferencePolicyV1(seed=819),
            scheduler=object(),
            sigma_contract_module=SigmaContract,
            inference_step_index=0,
            runtime_timestep=product.PINNED_UNIPC_TIMESTEPS[0],
        )
        self.assertTrue(torch.equal(output[:, :21], source))
        self.assertFalse(torch.equal(output[:, 21:], noise))
        self.assertEqual(route_receipt["exact_block_indices"], list(range(30)))
        hooks.restore()

    def test_sp4_hook_preserves_source_and_append_padding_bytes(self) -> None:
        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(30)]
                )

            def forward(self, value):
                for block in self.blocks:
                    value = block(value)
                return value

        class SigmaContract:
            SCHEDULE_SHA256 = product.PINNED_UNIPC_SCHEDULE_SHA256
            SCHEDULER_CLASS = "UniPCMultistepScheduler"
            NUM_INFERENCE_STEPS = 40
            FLOW_SHIFT = 5.0

            @staticmethod
            def audit_runtime_unipc_schedule(scheduler, initialize=True):
                del scheduler, initialize
                return {
                    "schedule_sha256": product.PINNED_UNIPC_SCHEDULE_SHA256,
                    "timesteps": list(product.PINNED_UNIPC_TIMESTEPS),
                    "positive_sigmas_float32_be_hex": ["00000001"] * 40,
                    "terminal_sigma_float32_be_hex": "00000000",
                }

        for rank in (1, 3):
            with self.subTest(sp_rank=rank):
                conditioner = self.conditioner()
                with torch.no_grad():
                    for projection in conditioner.injection.projections:
                        projection.weight.fill_(0.001)
                        projection.bias.fill_(0.002)
                transformer = Transformer()
                hooks = product.install_offline_action_plan_hooks(
                    transformer=transformer,
                    conditioner=conditioner,
                    torch_module=torch,
                )
                inputs = product.deterministic_fixed_inputs(
                    conditioner=conditioner,
                    torch_module=torch,
                    device=torch.device("cpu"),
                )
                plan = product.prepare_product_action_route(
                    conditioner=conditioner,
                    inputs=inputs,
                    predictor_module=self.predictor,
                    torch_module=torch,
                )
                route = product.OfflineLocalActionRoute(
                    plan=plan,
                    source_tokens=21,
                    inference_tokens=21,
                    sequence_parallel_rank=rank,
                    sequence_parallel_size=4,
                    row_identity=f"cpu-product-sp4-rank-{rank}",
                )
                source = torch.linspace(-0.5, 0.5, 21 * 8).reshape(1, 21, 8)
                noise = inputs.inference_noise_hidden.reshape(1, 21, 8)
                padding = torch.full((1, 2, 8), -0.25)
                padded = torch.cat((source, noise, padding), dim=1)
                start = rank * route.local_length
                local = padded[:, start : start + route.local_length, :].clone()
                selector = torch.tensor(route.local_phase_indices_tuple()) >= 0
                output, _ = product.execute_offline_denoiser_step(
                    route=route,
                    denoiser_step=transformer,
                    denoiser_kwargs={"value": local},
                    inference_policy=product.OfflineInferencePolicyV1(seed=819),
                    scheduler=object(),
                    sigma_contract_module=SigmaContract,
                    inference_step_index=0,
                    runtime_timestep=product.PINNED_UNIPC_TIMESTEPS[0],
                )
                self.assertTrue(torch.equal(output[:, ~selector, :], local[:, ~selector, :]))
                self.assertFalse(torch.equal(output[:, selector, :], local[:, selector, :]))
                hooks.restore()


@unittest.skipIf(torch is None, "PyTorch is unavailable in this local workspace")
class StrictCheckpointLoadTensorTests(unittest.TestCase):
    def test_authenticated_cpu_state_optimizer_runtime_load_is_rng_neutral(self) -> None:
        if str(METHOD_ROOT) not in sys.path:
            sys.path.insert(0, str(METHOD_ROOT))
        import action_plan_predictor_v1 as predictor

        config = predictor.ActionPlanPredictorConfig(
            profile=predictor.CPU_TEST_PROFILE,
            source_token_width=12,
            instruction_token_width=16,
            model_width=16,
            attention_heads=4,
            mlp_width=32,
            layer_count=2,
        )
        conditioner = predictor.ActionPlanConditionerV1(
            config, renderer_hidden_width=8
        )
        model = torch.nn.Module()
        model.add_module("action_plan_conditioner_v1", conditioner)

        class Runner:
            @staticmethod
            def exact_trainable_named_parameters(model, _conditioner):
                return tuple(
                    (name, value)
                    for name, value in model.named_parameters()
                    if value.requires_grad
                )

            @staticmethod
            def trainable_inventory(named):
                return tuple(
                    {
                        "name": name,
                        "shape": [int(value) for value in parameter.shape],
                        "dtype": str(parameter.dtype),
                        "numel": int(parameter.numel()),
                    }
                    for name, parameter in named
                )

            object_sha256 = staticmethod(consumer.object_sha256)

            @staticmethod
            def load_conditioner_state_strict(conditioner, state):
                conditioner.load_state_dict(dict(state), strict=True)

            @staticmethod
            def load_trainable_state_strict(named, state):
                with torch.no_grad():
                    for name, parameter in named:
                        parameter.copy_(state[name])

            @staticmethod
            def tensor_digest(_named):
                return consumer.PINNED_R2_P_STATE_SHA256[2]

            @staticmethod
            def validate_adamw_state_abi(state, named, step):
                if state != {"fake_adamw_state": True} or not named or step != 2:
                    raise AssertionError("fake optimizer ABI differs")

        runner = Runner()
        named = runner.exact_trainable_named_parameters(model, conditioner)
        inventory_sha = consumer.object_sha256(
            list(runner.trainable_inventory(named))
        )
        abi_sha = predictor.exact_state_dict_abi(conditioner)["abi_sha256"]
        random_state = __import__("random").getstate()
        runtime = {
            "schema_version": consumer.RUNTIME_STATE_SCHEMA,
            "completed_optimizer_steps": 2,
            "next_sampler_cursor": None,
            "scheduler": {
                "object": None,
                "policy": "constant_lr_no_scheduler_object",
                "learning_rate": 1.0e-4,
                "completed_steps": 2,
            },
            "stochasticity": {
                "training_noise": "counter_based_per_row_torch_Generator_cpu",
                "dropout": 0.0,
                "rng_snapshots_retained_for_full_replay_abi": True,
            },
            "per_rank": [
                {
                    "world_rank": rank,
                    "dp_arm": rank // 4,
                    "sp_rank": rank % 4,
                    "python_random_state": random_state,
                    "torch_cpu_rng_state": torch.arange(16, dtype=torch.uint8),
                    "torch_cuda_rng_state": torch.arange(8, dtype=torch.uint8),
                }
                for rank in range(8)
            ],
        }
        state = {
            "schema_version": consumer.FULL_STATE_SCHEMA,
            "trainable_parameters": {
                name: parameter.detach().cpu().contiguous().clone()
                for name, parameter in named
            },
            "action_plan_conditioner": {
                name: value.detach().cpu().contiguous().clone()
                for name, value in conditioner.state_dict().items()
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state_path = root / "full_trainable_state.pt"
            optimizer_path = root / "optimizer.pt"
            runtime_path = root / "runtime_state.pt"
            torch.save(state, state_path)
            torch.save({"fake_adamw_state": True}, optimizer_path)
            torch.save(runtime, runtime_path)
            metadata = {
                "adapter_sha256": consumer.file_sha256(state_path),
                "optimizer_sha256": consumer.file_sha256(optimizer_path),
                "runtime_state_sha256": consumer.file_sha256(runtime_path),
                "trainable_inventory_sha256": inventory_sha,
                "architecture": {
                    "action_plan_state_dict_abi": {"abi_sha256": abi_sha}
                },
                "release_closure": {
                    "sha256": consumer.PINNED_R2_RELEASE_MANIFEST_SHA256,
                    "member_set_sha256": consumer.PINNED_R2_RELEASE_MEMBER_SET_SHA256,
                },
            }
            checkpoint = consumer.CheckpointPreflight(
                directory=root,
                step=2,
                metadata=metadata,
                metadata_sha256="7" * 64,
                adapter_path=state_path,
                optimizer_path=optimizer_path,
                runtime_path=runtime_path,
                parameter_sha256=consumer.PINNED_R2_P_STATE_SHA256[2],
            )
            record = {
                "metadata_sha256": checkpoint.metadata_sha256,
                "adapter_sha256": metadata["adapter_sha256"],
                "optimizer_sha256": metadata["optimizer_sha256"],
                "runtime_state_sha256": metadata["runtime_state_sha256"],
            }
            campaign = consumer.CampaignReceipt(
                path=root / "receipt.json",
                sha256=consumer.PINNED_R2_CAMPAIGN_RECEIPT_SHA256,
                raw={},
                parameter_digests=dict(consumer.PINNED_R2_P_STATE_SHA256),
                checkpoint_records={0: {}, 1: {}, 2: record},
            )
            python_before = __import__("random").getstate()
            torch_before = torch.get_rng_state().clone()
            loaded = consumer.load_fresh_checkpoint_strict(
                model=model,
                conditioner=conditioner,
                checkpoint=checkpoint,
                campaign=campaign,
                runner=runner,
                predictor_module=predictor,
                release_closure=metadata["release_closure"],
                torch_module=torch,
                require_formal_profile=False,
            )
        self.assertEqual(
            loaded.receipt["loaded_parameter_sha256"],
            consumer.PINNED_R2_P_STATE_SHA256[2],
        )
        self.assertTrue(loaded.receipt["training_attached_reference_absent"])
        self.assertFalse(loaded.receipt["training_to_fresh_forward_parity_verified"])
        self.assertEqual(__import__("random").getstate(), python_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))


if __name__ == "__main__":
    unittest.main()
