from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_preservation_checkpoint_dynamics_html_v1 as builder  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed(value: dict) -> dict:
    result = dict(value)
    result["receipt_digest"] = hashlib.sha256(
        builder.canonical_json_bytes(result)
    ).hexdigest()
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )


class CheckpointDynamicsHTMLTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "dog" / "source.mp4"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source-video")
        self.source_caption = "A grey dog stands in a fixed autumn park scene."
        self.instruction = (
            "A grey dog bends its hind legs, lowers its hips, settles into a "
            "stable seated pose, and holds the pose while the camera and scene remain fixed."
        )
        self.seed = 2026081601
        self.training_iids = ["train-source-a", "train-source-b"]
        self.dataset_receipt_path = self.root / "training" / "dataset-receipt.json"
        self.dataset_receipt = _signed(
            {
                "schema_version": builder.DATASET_RECEIPT_SCHEMA,
                "complete": True,
                "action_supervision_present": False,
                "edited_target_accessed": False,
                "paired_dataset_accessed": False,
                "prior_posterior_accessed": False,
                "synthetic_edited_target_present": False,
                "target_video_accessed": False,
                "target_video_path_present": False,
                "scientific_claim_authorized": False,
                "semantic_motion_preservation_claimed": False,
                "dataset": {
                    "iids": list(self.training_iids),
                    "rows": len(self.training_iids),
                    "sha256": "1" * 64,
                },
            }
        )
        _write_json(self.dataset_receipt_path, self.dataset_receipt)
        self.manifest = {
            "schema_version": builder.SCHEMA_VERSION,
            "authority": dict(builder.AUTHORITY),
            "ranks": {
                name: {
                    "adapter_rank": rank,
                    "training_dataset_receipt": str(
                        self.dataset_receipt_path.relative_to(self.root)
                    ),
                }
                for name, rank in builder.RANKS.items()
            },
            "cells": [
                {
                    "cell_id": "dog",
                    "source_iid": "heldout-dog",
                    "source_video": "dog/source.mp4",
                    "source_action_caption": self.source_caption,
                    "full_instruction": self.instruction,
                    "seed": self.seed,
                    "variants": {},
                }
            ],
        }
        for rank_name, rank in builder.RANKS.items():
            self.manifest["cells"][0]["variants"][rank_name] = (
                self._rank_fixture(rank_name, rank)
            )
        self.manifest_path = self.root / "manifest.json"
        _write_json(self.manifest_path, self.manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _training_receipt(
        self,
        rank: int,
        step: int,
        adapter_file_sha: str,
        *,
        checkpoint_bundles: list[dict] | None = None,
    ) -> dict:
        value = {
            "schema_version": builder.TRAINING_RECEIPT_SCHEMA,
            "method": "bernini-preservation-only-base-prior-residual-v1",
            "complete": True,
            "mode": "preservation-residual-v1",
            "optimizer_steps": step,
            "adapter_rank": rank,
            "registered_schedule_indices": list(range(40)),
            "base_frozen": True,
            "frozen_base_action_prior_not_retrained": True,
            "scientific_claim_authorized": False,
            "action_editing_claim_authorized": False,
            "method_success_claimed": False,
            "training_schedule_indices": list(range(step)),
            "dataset": {
                "rows": 2,
                "parquet_sha256": "1" * 64,
                "receipt_sha256": _sha(self.dataset_receipt_path),
                "receipt_digest": self.dataset_receipt["receipt_digest"],
                "synthetic_target_consumed": False,
                "target_is_same_real_source": True,
            },
            "objective": {
                "name": "single_preservation_residual_mse",
                "feature_reward": False,
                "vlm_reward": False,
                "action_reward": False,
                "synthetic_target": False,
            },
            "initial_adapter_sha256": ("7" if rank == 8 else "8") * 64,
            "final_adapter_sha256": ("3" if rank == 8 else "4") * 64,
            "artifacts": {"adapter.safetensors": adapter_file_sha},
            "method_source_revision": "9" * 40,
            "method_source_archive_sha256": "e" * 64,
            "method_source_manifest_sha256": "f" * 64,
        }
        if step == 20:
            value.update(
                {
                    "checkpoint_bundle": True,
                    "continuous_trajectory": True,
                    "trajectory_optimizer_steps": 40,
                    "checkpoint_interval": 20,
                }
            )
        else:
            value.update(
                {
                    "positive_gradient_steps": 40,
                    "checkpoint_interval": 20,
                    "formal_exact40_complete": True,
                    "checkpoint_bundles": checkpoint_bundles,
                }
            )
        return _signed(value)

    def _inference_receipt(
        self,
        *,
        rank: int,
        native_video: Path,
        residual_video: Path,
        training_receipt_path: Path,
        adapter_file_sha: str,
        parameter_sha: str,
    ) -> dict:
        value = {
            "schema_version": builder.INFERENCE_RECEIPT_SCHEMA,
            "method": "bernini-preservation-residual-action-canary-v1",
            "cell_id": "dog",
            "action_reward_consumed": False,
            "feature_reward_consumed": False,
            "vlm_reward_consumed": False,
            "synthetic_target_consumed": False,
            "scientific_or_action_editing_claim_authorized": False,
            "input": {
                "source_video_sha256": _sha(self.source),
                "source_action_caption": self.source_caption,
                "source_action_caption_sha256": hashlib.sha256(
                    self.source_caption.encode("utf-8")
                ).hexdigest(),
                "target_action_caption": self.instruction,
                "target_action_caption_sha256": hashlib.sha256(
                    self.instruction.encode("utf-8")
                ).hexdigest(),
            },
            "sampling": {
                "seed": self.seed,
                "num_inference_steps": 40,
                "frame_count": 81,
                "same_official_gaussian_all_arms": True,
            },
            "outputs": {
                "native-rv2v": {
                    "sha256": _sha(native_video),
                    "frame_count": 81,
                    "fps": 25,
                },
                "preservation-residual": {
                    "sha256": _sha(residual_video),
                    "frame_count": 81,
                    "fps": 25,
                },
            },
            "preservation_residual": {
                "native-rv2v": {
                    "native_baseline": True,
                    "preservation_residual_applied": False,
                },
                "preservation-residual": {
                    "composition": "v_native_action+(v_adapted_noop-v_frozen_noop)",
                    "feature_reward": False,
                    "unit_gain": True,
                    "scheduler_steps": 40,
                },
            },
            "checkpoint": {
                "every_file_sha256_verified": True,
                "verified_entries_digest": "5" * 64,
                "manifest_sha256_computed": "6" * 64,
            },
            "training_bundle": {
                "adapter_rank": rank,
                "adapter_sha256": adapter_file_sha,
                "receipt_sha256": _sha(training_receipt_path),
                "strict_load": {
                    "parameter_digest": parameter_sha,
                    "adapter_file_sha256": adapter_file_sha,
                    "strict_tensor_and_metadata_closure": True,
                    "all_adapter_parameters_frozen_for_inference": True,
                },
            },
            "source_revisions": {
                "runtime_method": "9" * 40,
                "runtime_source_archive_sha256": "e" * 64,
            },
        }
        return _signed(value)

    def _rank_fixture(self, rank_name: str, rank: int) -> dict:
        base = self.root / "dog" / rank_name
        base.mkdir(parents=True)
        result: dict[str, dict[str, str]] = {}
        native20 = base / "step20-native.mp4"
        residual20 = base / "step20-preservation.mp4"
        native20.write_bytes(b"shared-native-step0")
        residual20.write_bytes(f"{rank_name}-residual20".encode())
        adapter20_sha = ("a" if rank == 8 else "b") * 64
        training20 = base / "step20-training-receipt.json"
        training20_value = self._training_receipt(rank, 20, adapter20_sha)
        _write_json(training20, training20_value)
        inference20 = base / "step20-inference-receipt.json"
        _write_json(
            inference20,
            self._inference_receipt(
                rank=rank,
                native_video=native20,
                residual_video=residual20,
                training_receipt_path=training20,
                adapter_file_sha=adapter20_sha,
                parameter_sha=training20_value["final_adapter_sha256"],
            ),
        )
        result["step0"] = {
            "video": str(native20.relative_to(self.root)),
            "inference_receipt": str(inference20.relative_to(self.root)),
        }
        result["step20"] = {
            "video": str(residual20.relative_to(self.root)),
            "paired_native_video": str(native20.relative_to(self.root)),
            "inference_receipt": str(inference20.relative_to(self.root)),
            "training_receipt": str(training20.relative_to(self.root)),
        }

        native40 = base / "step40-native.mp4"
        residual40 = base / "step40-preservation.mp4"
        native40.write_bytes(b"shared-native-step0")
        residual40.write_bytes(f"{rank_name}-residual40".encode())
        adapter40_sha = ("c" if rank == 8 else "d") * 64
        training40 = base / "step40-training-receipt.json"
        step20_receipt_sha = _sha(training20)
        training40_value = self._training_receipt(
            rank,
            40,
            adapter40_sha,
            checkpoint_bundles=[
                {
                    "ok": True,
                    "optimizer_step": 0,
                    "adapter_parameter_sha256": ("7" if rank == 8 else "8")
                    * 64,
                    "adapter_file_sha256": "0" * 64,
                    "receipt_sha256": "1" * 64,
                    "receipt_digest": "2" * 64,
                },
                {
                    "ok": True,
                    "optimizer_step": 20,
                    "adapter_parameter_sha256": training20_value[
                        "final_adapter_sha256"
                    ],
                    "adapter_file_sha256": adapter20_sha,
                    "receipt_sha256": step20_receipt_sha,
                    "receipt_digest": training20_value["receipt_digest"],
                },
            ],
        )
        _write_json(training40, training40_value)
        inference40 = base / "step40-inference-receipt.json"
        _write_json(
            inference40,
            self._inference_receipt(
                rank=rank,
                native_video=native40,
                residual_video=residual40,
                training_receipt_path=training40,
                adapter_file_sha=adapter40_sha,
                parameter_sha=training40_value["final_adapter_sha256"],
            ),
        )
        result["step40"] = {
            "video": str(residual40.relative_to(self.root)),
            "paired_native_video": str(native40.relative_to(self.root)),
            "inference_receipt": str(inference40.relative_to(self.root)),
            "training_receipt": str(training40.relative_to(self.root)),
        }
        return result

    def _rewrite_manifest(self, value: dict) -> None:
        self.manifest = value
        _write_json(self.manifest_path, value)

    def test_builds_source_plus_two_rank_three_checkpoint_full_video_grid(self) -> None:
        output = self.root / "index.html"
        result = builder.build(
            manifest_path=self.manifest_path,
            media_root=self.root,
            output=output,
        )
        self.assertEqual(result, output)
        text = output.read_text(encoding="utf-8")
        self.assertEqual(text.count("<video controls"), 11)
        for expected in (
            "Step 0 · native",
            "Step 20",
            "Step 40",
            "Rank 8",
            "Rank 2",
            self.instruction,
            str(self.seed),
            "checkpoint parameter/tree SHA-256",
            "checkpoint artifact/file SHA-256",
            "唯一训练 source 数：</b>2",
            "F0 旧两样本 preservation-residual 诊断",
            "不是新的 Stage-A V-axis 方法",
            "Source 是什么",
            "Instruction 是什么",
            "Step 是什么",
            "它不是视频帧编号，也不是质量分数",
            "F0 唯一训练 source 数：</b>2",
            "train-source-a · train-source-b",
            "Paired native · same inference process",
            "跨独立推理进程 native byte parity",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("0.61", text)
        self.assertNotIn("success score", text.lower())
        self.assertNotIn("<dt>value</dt>", text.lower())

    def test_missing_any_video_fails_before_publication(self) -> None:
        missing = self.root / self.manifest["cells"][0]["variants"]["rank8"]["step20"]["video"]
        missing.unlink()
        output = self.root / "missing-video.html"
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=output,
            )
        self.assertFalse(output.exists())

    def test_missing_paired_native_fails_before_publication(self) -> None:
        relative = self.manifest["cells"][0]["variants"]["rank8"]["step40"][
            "paired_native_video"
        ]
        (self.root / relative).unlink()
        output = self.root / "missing-paired-native.html"
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=output,
            )
        self.assertFalse(output.exists())

    def test_cross_process_native_drift_is_exposed_not_hidden(self) -> None:
        rank = self.manifest["cells"][0]["variants"]["rank8"]
        native = self.root / rank["step40"]["paired_native_video"]
        native.write_bytes(b"different-native-process")
        inference_path = self.root / rank["step40"]["inference_receipt"]
        receipt = json.loads(inference_path.read_text(encoding="ascii"))
        receipt.pop("receipt_digest")
        receipt["outputs"]["native-rv2v"]["sha256"] = _sha(native)
        _write_json(inference_path, _signed(receipt))
        output = self.root / "native-drift.html"
        builder.build(
            manifest_path=self.manifest_path,
            media_root=self.root,
            output=output,
        )
        text = output.read_text(encoding="utf-8")
        self.assertIn(
            "不一致；必须使用每个 checkpoint 自己的 paired native 比较",
            text,
        )
        self.assertIn('class="native-parity warn"', text)
        self.assertIn("#ff465a", text)

    def test_paired_native_sha_mismatch_fails(self) -> None:
        rank = self.manifest["cells"][0]["variants"]["rank2"]
        native = self.root / rank["step40"]["paired_native_video"]
        native.write_bytes(b"unreceipted-native-bytes")
        with self.assertRaisesRegex(
            builder.CheckpointDynamicsHTMLError, "paired native MP4 bytes"
        ):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "native-sha-mismatch.html",
            )

    def test_paired_native_metadata_must_be_exact81(self) -> None:
        rank = self.manifest["cells"][0]["variants"]["rank2"]
        inference_path = self.root / rank["step40"]["inference_receipt"]
        receipt = json.loads(inference_path.read_text(encoding="ascii"))
        receipt.pop("receipt_digest")
        receipt["outputs"]["native-rv2v"]["frame_count"] = 80
        _write_json(inference_path, _signed(receipt))
        with self.assertRaisesRegex(
            builder.CheckpointDynamicsHTMLError, "native MP4 metadata"
        ):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "native-metadata.html",
            )

    def test_step0_and_step20_must_share_exact_receipt_file(self) -> None:
        rank = self.manifest["cells"][0]["variants"]["rank8"]
        original = self.root / rank["step0"]["inference_receipt"]
        copy_path = original.with_name("step0-byte-distinct-receipt.json")
        copy_path.write_bytes(original.read_bytes() + b"\n")
        rank["step0"]["inference_receipt"] = str(copy_path.relative_to(self.root))
        self._rewrite_manifest(self.manifest)
        with self.assertRaisesRegex(
            builder.CheckpointDynamicsHTMLError,
            "process-paired native closure",
        ):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "step0-receipt-drift.html",
            )

    def test_output_must_be_inside_self_contained_media_root(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(
            builder.CheckpointDynamicsHTMLError, "self-contained media root"
        ):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=nested / "index.html",
            )

    def test_ancestor_symlink_alias_cannot_publish_html(self) -> None:
        alias = self.root.parent / f"{self.root.name}-alias-parent"
        alias.symlink_to(self.root.parent, target_is_directory=True)
        try:
            aliased_root = alias / self.root.name
            with self.assertRaisesRegex(
                builder.CheckpointDynamicsHTMLError, "self-contained media root"
            ):
                builder.build(
                    manifest_path=self.manifest_path,
                    media_root=self.root,
                    output=aliased_root / "alias-index.html",
                )
        finally:
            alias.unlink(missing_ok=True)

    def test_missing_any_receipt_fails_before_publication(self) -> None:
        missing = self.root / self.manifest["cells"][0]["variants"]["rank2"]["step40"]["training_receipt"]
        missing.unlink()
        output = self.root / "missing-receipt.html"
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=output,
            )
        self.assertFalse(output.exists())

    def test_missing_inference_receipt_fails_before_publication(self) -> None:
        missing = self.root / self.manifest["cells"][0]["variants"]["rank8"][
            "step20"
        ]["inference_receipt"]
        missing.unlink()
        output = self.root / "missing-inference-receipt.html"
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=output,
            )
        self.assertFalse(output.exists())

    def test_missing_dataset_receipt_fails_before_publication(self) -> None:
        self.dataset_receipt_path.unlink()
        output = self.root / "missing-dataset-receipt.html"
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=output,
            )
        self.assertFalse(output.exists())

    def test_dataset_receipt_must_prove_unique_training_sources(self) -> None:
        value = dict(self.dataset_receipt)
        value.pop("receipt_digest")
        value["dataset"] = dict(value["dataset"])
        value["dataset"]["iids"] = ["train-source-a", "train-source-a"]
        _write_json(self.dataset_receipt_path, _signed(value))
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "duplicate-source.html",
            )

    def test_feature_scalar_field_or_success_authority_is_rejected(self) -> None:
        value = copy.deepcopy(self.manifest)
        value["cells"][0]["variants"]["rank8"]["step20"]["feature_score"] = 0.9
        self._rewrite_manifest(value)
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "feature.html",
            )
        value = copy.deepcopy(self.manifest)
        value["cells"][0]["variants"]["rank8"]["step20"].pop("feature_score")
        value["authority"]["method_success_claimed"] = True
        self._rewrite_manifest(value)
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "claim.html",
            )

    def test_checkpoint_hash_mismatch_fails(self) -> None:
        value = copy.deepcopy(self.manifest)
        training_relative = value["cells"][0]["variants"]["rank8"]["step20"]["training_receipt"]
        path = self.root / training_relative
        receipt = json.loads(path.read_text(encoding="ascii"))
        receipt["final_adapter_sha256"] = "f" * 64
        receipt.pop("receipt_digest")
        _write_json(path, _signed(receipt))
        self._rewrite_manifest(value)
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "hash-mismatch.html",
            )

    def test_independent_exact20_cannot_be_spliced_to_old_step40(self) -> None:
        relative = self.manifest["cells"][0]["variants"]["rank8"]["step20"][
            "training_receipt"
        ]
        path = self.root / relative
        receipt = json.loads(path.read_text(encoding="ascii"))
        receipt.pop("receipt_digest")
        for field in (
            "checkpoint_bundle",
            "continuous_trajectory",
            "trajectory_optimizer_steps",
        ):
            receipt.pop(field)
        receipt["positive_gradient_steps"] = 20
        _write_json(path, _signed(receipt))
        dataset = builder._validate_dataset_receipt(self.dataset_receipt_path)
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder._validate_training_receipt(
                path,
                expected_rank=8,
                expected_step=20,
                dataset_receipt=dataset,
            )

    def test_old_step40_without_cadence_bindings_is_rejected(self) -> None:
        relative = self.manifest["cells"][0]["variants"]["rank2"]["step40"][
            "training_receipt"
        ]
        path = self.root / relative
        receipt = json.loads(path.read_text(encoding="ascii"))
        receipt.pop("receipt_digest")
        receipt.pop("checkpoint_bundles")
        _write_json(path, _signed(receipt))
        with self.assertRaises(builder.CheckpointDynamicsHTMLError):
            builder.build(
                manifest_path=self.manifest_path,
                media_root=self.root,
                output=self.root / "old-step40.html",
            )


if __name__ == "__main__":
    unittest.main()
