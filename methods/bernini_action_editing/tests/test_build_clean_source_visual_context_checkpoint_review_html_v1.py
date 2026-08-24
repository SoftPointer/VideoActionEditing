#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import clean_source_visual_context_checkpoint_review_contract_v1 as contract  # noqa: E402
import build_clean_source_visual_context_checkpoint_review_html_v1 as builder  # noqa: E402


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _sha_bytes(value)


def _manifest(root: Path) -> tuple[Path, dict, dict[str, dict]]:
    sentinels = []
    source_sha = {
        sentinel: hashlib.sha256(f"source-{sentinel}".encode()).hexdigest()
        for sentinel in contract.SENTINEL_ORDER
    }
    identities: dict[str, dict] = {}
    for index, sentinel in enumerate(contract.SENTINEL_ORDER):
        fixed = dict(contract.SENTINEL_IDENTITIES[sentinel])
        instructions = {
            branch: f"Complete {branch} instruction for {sentinel}; preserve identity and scene."
            for branch in contract.TEXT_BRANCHES
        }
        fixed["source_video_sha256"] = source_sha[sentinel]
        fixed["forward_instruction_sha256"] = hashlib.sha256(
            instructions["forward"].encode("utf-8")
        ).hexdigest()
        identities[sentinel] = fixed
        sentinels.append(
            {
                "sentinel_id": sentinel,
                "diversity_role": fixed["diversity_role"],
                "source_entity_type": fixed["source_entity_type"],
                "iid": fixed["iid"],
                "action_family": fixed["action_family"],
                "source_video": str(root / f"unopened-{sentinel}.mp4"),
                "source_video_sha256": source_sha[sentinel],
                "source_caption": f"Full source caption for {sentinel}.",
                "seed": fixed["seed"],
                "wrong_owner_iid": fixed["wrong_owner_iid"],
                "wrong_owner_source_video_sha256": next(
                    source_sha[name]
                    for name in contract.SENTINEL_ORDER
                    if contract.SENTINEL_IDENTITIES[name]["iid"] == fixed["wrong_owner_iid"]
                ),
                "instructions": instructions,
                "instruction_sha256": {
                    branch: hashlib.sha256(text.encode("utf-8")).hexdigest()
                    for branch, text in instructions.items()
                },
                "latent_shape": fixed["latent_shape"],
                "source_posterior_path": str(root / f"unopened-{sentinel}.pt"),
                "source_posterior_file_sha256": "1" * 64,
                "source_media": {"frame_count": 81, "fps": 25, "not_verified": True},
            }
        )
    unsigned = {
        "schema_version": contract.MANIFEST_SCHEMA,
        "manifest_id": "unit-test-fixed-four",
        "source_only_manifest": {
            "path": str(root / "source-only.json"),
            "file_sha256": "2" * 64,
            "manifest_digest": "3" * 64,
            "selected_split": "heldout",
            "train_overlap_count": 0,
        },
        "authoring": {
            "path": str(root / "authoring.json"),
            "file_sha256": "4" * 64,
            "authoring_digest": "5" * 64,
            "fixed_before_checkpoint_decode": True,
            "raw_full644_file_sha256": contract.source_data.PINNED_RAW_PARQUET_SHA256,
            "target_video_bytes_read": False,
        },
        "checkpoint_steps": list(contract.CHECKPOINT_STEPS),
        "sentinel_order": list(contract.SENTINEL_ORDER),
        "source_controls": list(contract.SOURCE_CONTROLS),
        "text_branches": list(contract.TEXT_BRANCHES),
        "logical_arm_order": list(contract.LOGICAL_ARM_ORDER),
        "physical_decode_arms": list(contract.PHYSICAL_DECODE_ARMS),
        "correct_forward_alias": {
            "logical_alias": True,
            "same_instruction": "forward",
            "same_source_control": "correct",
            "same_seed": True,
            "same_physical_mp4_required": True,
        },
        "sentinels": sentinels,
        "sampling": {
            "frame_count": 81,
            "fps": 25,
            "num_inference_steps": 40,
            "world_size": 4,
            "sequence_parallel_size": 4,
            "same_seed_within_sentinel_all_checkpoints_and_arms": True,
        },
        "authority": {
            "training_performed_by_review": False,
            "optimizer_present": False,
            "target_video_available": False,
            "feature_evaluator_present": False,
            "vlm_evaluator_present": False,
            "manual_video_review_required": True,
            "decoded_quality_claimed": False,
        },
    }
    value = {**unsigned, "manifest_digest": contract.object_sha256(unsigned)}
    path = root / "review-manifest.json"
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")
    return path, value, identities


def _media_record(relative: str, sha: str) -> dict:
    return {
        "relative_mp4": relative,
        "mp4_sha256": sha,
        "frame_count": 81,
        "fps": 25,
    }


def _build_shards(root: Path, manifest: dict) -> None:
    sentinel_by_id = {row["sentinel_id"]: row for row in manifest["sentinels"]}
    sentinel_by_iid = {row["iid"]: row for row in manifest["sentinels"]}
    base_bytes = {
        sentinel: f"base-output-{sentinel}".encode()
        for sentinel in contract.SENTINEL_ORDER
    }
    for step in contract.CHECKPOINT_STEPS:
        shard = root / f"step-{step:08d}"
        shard.mkdir(parents=True)
        source_records = []
        native_records = []
        logical = []
        checkpoint_sha = hashlib.sha256(f"checkpoint-{step}".encode()).hexdigest()
        for sentinel in contract.SENTINEL_ORDER:
            fixed = sentinel_by_id[sentinel]
            source_rel = f"media/{sentinel}__source.mp4"
            source_sha = _write(shard / source_rel, f"source-{sentinel}".encode())
            fixed["source_video_sha256"] = source_sha
            source_records.append(
                {
                    "sentinel_id": sentinel,
                    "iid": fixed["iid"],
                    "diversity_role": fixed["diversity_role"],
                    "source_entity_type": fixed["source_entity_type"],
                    "source_caption": fixed["source_caption"],
                    "source_video_sha256": source_sha,
                    "wrong_owner_source_video_sha256": fixed[
                        "wrong_owner_source_video_sha256"
                    ],
                    "seed": fixed["seed"],
                    **_media_record(source_rel, source_sha),
                }
            )
            if step == 0:
                native_rel = f"media/{sentinel}__native.mp4"
                native_sha = _write(shard / native_rel, base_bytes[sentinel])
                text = fixed["instructions"]["forward"]
                native_records.append(
                    {
                        "sentinel_id": sentinel,
                        "iid": fixed["iid"],
                        "source_video_sha256": source_sha,
                        "seed": fixed["seed"],
                        "instruction": text,
                        "instruction_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "route_trace_digest": "5" * 64,
                        "initial_gaussian_sha256": hashlib.sha256(
                            f"gaussian-{sentinel}".encode()
                        ).hexdigest(),
                        **_media_record(native_rel, native_sha),
                    }
                )
            physical: dict[str, tuple[str, str, str]] = {}
            for arm in contract.PHYSICAL_DECODE_ARMS:
                if arm == "carrier-off" or step == 0 and arm == "correct":
                    payload = base_bytes[sentinel]
                else:
                    payload = f"{step}-{sentinel}-{arm}".encode()
                rel = f"media/{sentinel}__{arm}.mp4"
                sha = _write(shard / rel, payload)
                physical[arm] = (rel, sha, hashlib.sha256(payload + b"trace").hexdigest())
            physical["forward"] = physical["correct"]
            for arm in contract.LOGICAL_ARM_ORDER:
                if arm in contract.SOURCE_CONTROLS:
                    axis, source_control, text_branch = "source-control", arm, "forward"
                else:
                    axis, source_control, text_branch = "typed-instruction", "correct", arm
                if source_control == "carrier-off":
                    memory_sha, transform = None, None
                elif source_control == "wrong-owner":
                    memory_sha, transform = fixed["wrong_owner_source_video_sha256"], "identity"
                elif source_control == "order-permutation":
                    memory_sha, transform = source_sha, "reverse-phase-order-20-to-0"
                else:
                    memory_sha, transform = source_sha, "identity"
                text = fixed["instructions"][text_branch]
                rel, sha, trace = physical[arm]
                logical.append(
                    {
                        "record_id": contract.logical_record_key(step, sentinel, arm),
                        "checkpoint_step": step,
                        "checkpoint_file_sha256": checkpoint_sha,
                        "sentinel_id": sentinel,
                        "iid": fixed["iid"],
                        "diversity_role": fixed["diversity_role"],
                        "source_entity_type": fixed["source_entity_type"],
                        "source_video_sha256": source_sha,
                        "seed": fixed["seed"],
                        "arm": arm,
                        "axis": axis,
                        "source_control": source_control,
                        "text_branch": text_branch,
                        "instruction": text,
                        "instruction_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "memory_source_video_sha256": memory_sha,
                        "memory_transform": transform,
                        "route_trace_digest": trace,
                        "initial_gaussian_sha256": hashlib.sha256(
                            f"gaussian-{sentinel}".encode()
                        ).hexdigest(),
                        **_media_record(rel, sha),
                        "physical_decode_id": f"{step}-{sentinel}-{physical[arm][1]}",
                    }
                )
        # Update manifest's wrong-owner hashes after the source bytes above are
        # known, then bring source/logical rows into exact agreement.
        for sentinel in contract.SENTINEL_ORDER:
            fixed = sentinel_by_id[sentinel]
            wrong = sentinel_by_iid[fixed["wrong_owner_iid"]]
            fixed["wrong_owner_source_video_sha256"] = wrong["source_video_sha256"]
        for source in source_records:
            fixed = sentinel_by_id[source["sentinel_id"]]
            source["wrong_owner_source_video_sha256"] = fixed[
                "wrong_owner_source_video_sha256"
            ]
        for row in logical:
            fixed = sentinel_by_id[row["sentinel_id"]]
            if row["source_control"] == "wrong-owner":
                row["memory_source_video_sha256"] = fixed[
                    "wrong_owner_source_video_sha256"
                ]
        unsigned = {
            "schema_version": contract.SHARD_SCHEMA,
            "complete": True,
            "checkpoint": {
                "step": step,
                "logical_records_seen": step * 8,
                "path": str(root / f"checkpoint-{step}.pt"),
                "file_sha256": checkpoint_sha,
                "adapter_parameter_digest": "6" * 64,
                "strict_load_succeeded": True,
            },
            "review_manifest_digest": manifest["manifest_digest"],
            "memory_input_kind": "clean_source",
            "source_records": source_records,
            "native_records": native_records,
            "logical_records": logical,
            "execution": {
                "world_size": 4,
                "sequence_parallel_size": 4,
                "num_inference_steps": 40,
                "frame_count": 81,
                "fps": 25,
                "same_seed_all_arms_within_sentinel": True,
                "same_source_all_checkpoints": True,
                "parent_allocation_released": False,
            },
            "authority": {
                "decoded_checkpoint_inference_executed": True,
                "optimizer_present": False,
                "backward_performed": False,
                "parameter_update": False,
                "feature_evaluator_present": False,
                "vlm_evaluator_present": False,
                "manual_review_pending": True,
                "quality_claimed": False,
            },
        }
        receipt = {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}
        (shard / "receipt.json").write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
    # The test mutates source SHA fields after initial manifest construction.
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_digest")
    manifest["manifest_digest"] = contract.object_sha256(unsigned_manifest)

    # Rebind receipts to the final manifest digest and recompute their seals.
    for step in contract.CHECKPOINT_STEPS:
        path = root / f"step-{step:08d}" / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt["review_manifest_digest"] = manifest["manifest_digest"]
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = contract.object_sha256(receipt)
        path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")


class HtmlBuilderTests(unittest.TestCase):
    def _fixture(self) -> tuple[tempfile.TemporaryDirectory, Path, dict, Path]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name).resolve()
        manifest_path, manifest, identities = _manifest(root)
        identity_patch = mock.patch.object(contract, "SENTINEL_IDENTITIES", identities)
        identity_patch.start()
        self.addCleanup(identity_patch.stop)
        shard_root = root / "shards"
        shard_root.mkdir()
        _build_shards(shard_root, manifest)
        manifest_path.write_bytes(contract.canonical_json_bytes(manifest) + b"\n")
        return temp, manifest_path, manifest, shard_root

    def test_builds_complete_clear_self_contained_review(self) -> None:
        temp, manifest_path, manifest, shard_root = self._fixture()
        self.addCleanup(temp.cleanup)
        output = Path(temp.name).resolve() / "review"
        result = builder.build_review(
            manifest_path=manifest_path,
            expected_manifest_sha256=contract.file_sha256(manifest_path),
            shard_root=shard_root,
            output_dir=output,
            verify_manifest_files=False,
            verify_media=False,
        )
        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertEqual(result["logical_records"], 200)
        self.assertIn("What “carrier-off” removes", page)
        self.assertIn("reverses its 21 latent phases", page)
        self.assertIn("Registered wrong-owner memory", page)
        self.assertIn("hand-object-blueprint-roll", page)
        self.assertIn("equal latent geometry, different entity and scene", page)
        self.assertIn("Complete incomplete instruction for emitter-fireworks-explode", page)
        self.assertIn("Checkpoint step 80", page)
        self.assertIn("Source (unchanged input)", page)
        self.assertIn("Native frozen Bernini", page)
        self.assertNotIn("<script", page.lower())
        self.assertNotIn("https://", page.lower())
        self.assertTrue((output / "evidence.json").is_file())
        self.assertTrue(all(not path.is_symlink() for path in output.rglob("*")))
        # Content-addressed aliases prevent 200 redundant copies.
        self.assertLess(len(list((output / "media").glob("*.mp4"))), 184)

    def test_rejects_correct_forward_alias_drift(self) -> None:
        temp, manifest_path, manifest, shard_root = self._fixture()
        self.addCleanup(temp.cleanup)
        receipt_path = shard_root / "step-00000020" / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        row = next(row for row in receipt["logical_records"] if row["sentinel_id"] == contract.SENTINEL_ORDER[0] and row["arm"] == "forward")
        row["physical_decode_id"] = "drift"
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = contract.object_sha256(receipt)
        receipt_path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
        with self.assertRaisesRegex(builder.CheckpointReviewHtmlError, "alias"):
            builder.build_review(
                manifest_path=manifest_path,
                expected_manifest_sha256=contract.file_sha256(manifest_path),
                shard_root=shard_root,
                output_dir=Path(temp.name).resolve() / "review",
                verify_manifest_files=False,
                verify_media=False,
            )

    def test_rejects_wrong_owner_label_with_correct_memory(self) -> None:
        temp, manifest_path, manifest, shard_root = self._fixture()
        self.addCleanup(temp.cleanup)
        receipt_path = shard_root / "step-00000040" / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        row = next(row for row in receipt["logical_records"] if row["sentinel_id"] == contract.SENTINEL_ORDER[3] and row["arm"] == "wrong-owner")
        row["memory_source_video_sha256"] = row["source_video_sha256"]
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = contract.object_sha256(receipt)
        receipt_path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
        with self.assertRaisesRegex(builder.CheckpointReviewHtmlError, "memory owner"):
            builder.build_review(
                manifest_path=manifest_path,
                expected_manifest_sha256=contract.file_sha256(manifest_path),
                shard_root=shard_root,
                output_dir=Path(temp.name).resolve() / "review",
                verify_manifest_files=False,
                verify_media=False,
            )

    def test_rejects_initial_gaussian_drift_across_checkpoint_or_arm(self) -> None:
        temp, manifest_path, manifest, shard_root = self._fixture()
        self.addCleanup(temp.cleanup)
        receipt_path = shard_root / "step-00000040" / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        row = next(
            item
            for item in receipt["logical_records"]
            if item["sentinel_id"] == contract.SENTINEL_ORDER[0] and item["arm"] == "reverse"
        )
        row["initial_gaussian_sha256"] = "9" * 64
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = contract.object_sha256(receipt)
        receipt_path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
        with self.assertRaisesRegex(
            builder.CheckpointReviewHtmlError, "same official Gaussian"
        ):
            builder.build_review(
                manifest_path=manifest_path,
                expected_manifest_sha256=contract.file_sha256(manifest_path),
                shard_root=shard_root,
                output_dir=Path(temp.name).resolve() / "review",
                verify_manifest_files=False,
                verify_media=False,
            )

    def test_rejects_frozen_carrier_drift_across_checkpoints(self) -> None:
        temp, manifest_path, manifest, shard_root = self._fixture()
        self.addCleanup(temp.cleanup)
        receipt_path = shard_root / "step-00000080" / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        row = next(row for row in receipt["logical_records"] if row["sentinel_id"] == contract.SENTINEL_ORDER[1] and row["arm"] == "carrier-off")
        row["mp4_sha256"] = "9" * 64
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = contract.object_sha256(receipt)
        receipt_path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
        with self.assertRaisesRegex(builder.CheckpointReviewHtmlError, "drifted"):
            builder.build_review(
                manifest_path=manifest_path,
                expected_manifest_sha256=contract.file_sha256(manifest_path),
                shard_root=shard_root,
                output_dir=Path(temp.name).resolve() / "review",
                verify_manifest_files=False,
                verify_media=False,
            )

    def test_rejects_any_evaluator_field_even_when_false(self) -> None:
        temp, manifest_path, manifest, shard_root = self._fixture()
        self.addCleanup(temp.cleanup)
        receipt_path = shard_root / "step-00000060" / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["score"] = None
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = contract.object_sha256(receipt)
        receipt_path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
        with self.assertRaisesRegex(builder.CheckpointReviewHtmlError, "forbidden evaluator"):
            builder.build_review(
                manifest_path=manifest_path,
                expected_manifest_sha256=contract.file_sha256(manifest_path),
                shard_root=shard_root,
                output_dir=Path(temp.name).resolve() / "review",
                verify_manifest_files=False,
                verify_media=False,
            )


if __name__ == "__main__":
    unittest.main()
