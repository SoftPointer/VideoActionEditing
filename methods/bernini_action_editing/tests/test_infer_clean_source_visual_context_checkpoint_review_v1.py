#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_checkpoint_review_contract_v1 as review  # noqa: E402
import clean_source_visual_context_stage_b_contract_v1 as stage_contract  # noqa: E402
import clean_source_visual_context_adapter_v1 as visual  # noqa: E402
import infer_clean_source_visual_context_checkpoint_review_v1 as runner  # noqa: E402


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _authority_fixture(root: Path) -> tuple[Path, str, dict, dict]:
    source_manifest = root / "source-only.json"
    source_sha = _write(source_manifest, b"sealed-source-only-v3\n")
    source_digest = "1" * 64
    admission_digest = "2" * 64
    checkpoints = []
    for step in stage_contract.CHECKPOINT_STEPS:
        path = root / f"checkpoint-{step:08d}.pt"
        file_sha = _write(path, f"checkpoint-{step}".encode())
        checkpoints.append(
            {
                "step": step,
                "logical_records_seen": step * stage_contract.GLOBAL_BATCH,
                "path": str(path),
                "file_sha256": file_sha,
                "adapter_parameter_digest": hashlib.sha256(
                    f"adapter-{step}".encode()
                ).hexdigest(),
            }
        )
    memory_kind = "clean_source"
    chain = stage_contract.checkpoint_decode_chain(
        checkpoints,
        manifest_digest=source_digest,
        admission_digest=admission_digest,
        memory_input_kind=memory_kind,
    )
    unsigned = {
        "schema_version": runner.TRAINING_RECEIPT_SCHEMA,
        "complete": True,
        "optimizer_steps": 80,
        "continuous_trajectory": True,
        "checkpoint_steps": list(stage_contract.CHECKPOINT_STEPS),
        "memory_input_kind": memory_kind,
        "dataset": {
            "manifest_digest": source_digest,
            "manifest_path": str(source_manifest),
            "manifest_file_sha256": source_sha,
            "optimizer_split": "train",
            "optimizer_rows": 64,
            "heldout_action_canary_rows": 8,
            "posterior_index_0_accessed": True,
            "posterior_index_1_synthetic_target_accessed": False,
        },
        "stage_a_admission": {
            "receipt_digest": admission_digest,
            "installed_sparse_block_indices": list(
                stage_contract.PREREGISTERED_SPARSE_BLOCK_INDICES
            ),
        },
        "model": {
            "bernini_commit": stage_contract.EXPECTED_BERNINI_COMMIT,
            "veomni_commit": stage_contract.EXPECTED_VEOMNI_COMMIT,
            "model_revision": visual.PINNED_BERNINI_MODEL_REVISION,
            "checkpoint_tree_sha256": stage_contract.EXPECTED_CHECKPOINT_TREE_SHA256,
            "checkpoint_content_manifest_sha256": (
                stage_contract.EXPECTED_CHECKPOINT_MANIFEST_SHA256
            ),
        },
        "adapter": {
            "runtime_memory_input_binding": {"input_kind": memory_kind}
        },
        "checkpoint_decode_chain": chain,
        "checkpoint_records": checkpoints,
        "post_training_review_integration": {
            "all_checkpoints_strictly_loadable": True,
            "checkpoint_videos_decoded": False,
            "html_review_generated": False,
        },
        "authority": {
            "gpu_runtime_executed": True,
            "decoded_checkpoint_inference_executed": False,
        },
    }
    receipt = {**unsigned, "receipt_digest": stage_contract.object_sha256(unsigned)}
    receipt_path = root / "training-receipt.json"
    receipt_path.write_bytes(stage_contract.canonical_json_bytes(receipt) + b"\n")
    review_manifest = {
        "source_only_manifest": {
            "path": str(source_manifest),
            "file_sha256": source_sha,
            "manifest_digest": source_digest,
        }
    }
    return receipt_path, stage_contract.file_sha256(receipt_path), review_manifest, receipt


class CheckpointDecodeRunnerTests(unittest.TestCase):
    def test_training_authority_binds_exact_checkpoint_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt, receipt_sha, manifest, _ = _authority_fixture(root)
            authority = runner.load_training_decode_authority(
                receipt,
                expected_file_sha256=receipt_sha,
                review_manifest=manifest,
                checkpoint_step=40,
            )
            self.assertEqual(authority.checkpoint_step, 40)
            self.assertEqual(authority.memory_input_kind, "clean_source")
            self.assertEqual(
                authority.block_indices,
                stage_contract.PREREGISTERED_SPARSE_BLOCK_INDICES,
            )
            self.assertTrue(authority.checkpoint_path.is_file())

    def test_training_authority_rejects_manifest_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt, receipt_sha, manifest, _ = _authority_fixture(root)
            manifest["source_only_manifest"]["manifest_digest"] = "9" * 64
            with self.assertRaisesRegex(runner.CheckpointDecodeError, "source-only"):
                runner.load_training_decode_authority(
                    receipt,
                    expected_file_sha256=receipt_sha,
                    review_manifest=manifest,
                    checkpoint_step=20,
                )

    def test_training_authority_rejects_checkpoint_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt, receipt_sha, manifest, value = _authority_fixture(root)
            selected = next(row for row in value["checkpoint_records"] if row["step"] == 60)
            Path(selected["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(runner.CheckpointDecodeError, "bytes changed"):
                runner.load_training_decode_authority(
                    receipt,
                    expected_file_sha256=receipt_sha,
                    review_manifest=manifest,
                    checkpoint_step=60,
                )

    def test_physical_plan_is_nine_samples_for_ten_logical_rows(self) -> None:
        plan = runner.physical_decode_plan()
        self.assertEqual(len(plan), 9)
        self.assertEqual(
            tuple(row["physical_arm"] for row in plan), review.PHYSICAL_DECODE_ARMS
        )
        self.assertEqual(len(review.LOGICAL_ARM_ORDER), 10)
        self.assertEqual(plan[0]["text_branch"], "forward")
        self.assertEqual(plan[1]["source_control"], "carrier-off")

    def test_logical_rows_bind_real_provider_semantics_and_alias(self) -> None:
        source_sha = {name: hashlib.sha256(name.encode()).hexdigest() for name in review.SENTINEL_ORDER}
        sentinels = []
        by_iid = {
            fixed["iid"]: sentinel_id
            for sentinel_id, fixed in review.SENTINEL_IDENTITIES.items()
        }
        pairs = {
            sentinel_id: by_iid[fixed["wrong_owner_iid"]]
            for sentinel_id, fixed in review.SENTINEL_IDENTITIES.items()
        }
        for index, sentinel_id in enumerate(review.SENTINEL_ORDER):
            fixed = review.SENTINEL_IDENTITIES[sentinel_id]
            branches = {branch: f"{sentinel_id} complete {branch}" for branch in review.TEXT_BRANCHES}
            sentinels.append(
                {
                    "sentinel_id": sentinel_id,
                    "iid": fixed["iid"],
                    "diversity_role": fixed["diversity_role"],
                    "source_entity_type": fixed["source_entity_type"],
                    "source_video_sha256": source_sha[sentinel_id],
                    "wrong_owner_source_video_sha256": source_sha[pairs[sentinel_id]],
                    "seed": 100 + index,
                    "instructions": branches,
                    "instruction_sha256": {
                        branch: hashlib.sha256(text.encode()).hexdigest()
                        for branch, text in branches.items()
                    },
                }
            )
        physical = {}
        for sentinel_id in review.SENTINEL_ORDER:
            for arm in review.PHYSICAL_DECODE_ARMS:
                digest = hashlib.sha256(f"{sentinel_id}-{arm}".encode()).hexdigest()
                physical[(sentinel_id, arm)] = {
                    "route_trace_digest": digest,
                    "initial_gaussian_sha256": source_sha[sentinel_id],
                    "relative_mp4": f"media/{sentinel_id}__{arm}.mp4",
                    "mp4_sha256": digest,
                    "physical_decode_id": f"step-00000020__{sentinel_id}__{arm}",
                }
        rows = runner._logical_rows(
            checkpoint_step=20,
            checkpoint_sha256="a" * 64,
            manifest={"sentinels": sentinels},
            physical=physical,
        )
        self.assertEqual(len(rows), 40)
        by_key = {(row["sentinel_id"], row["arm"]): row for row in rows}
        for sentinel_id in review.SENTINEL_ORDER:
            correct = by_key[(sentinel_id, "correct")]
            forward = by_key[(sentinel_id, "forward")]
            wrong = by_key[(sentinel_id, "wrong-owner")]
            order = by_key[(sentinel_id, "order-permutation")]
            self.assertEqual(correct["physical_decode_id"], forward["physical_decode_id"])
            self.assertEqual(wrong["memory_source_video_sha256"], source_sha[pairs[sentinel_id]])
            self.assertEqual(order["memory_source_video_sha256"], source_sha[sentinel_id])
            self.assertEqual(order["memory_transform"], "reverse-phase-order-20-to-0")


if __name__ == "__main__":
    unittest.main()
