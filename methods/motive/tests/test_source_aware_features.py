from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from motive.descriptor import DescriptorConfig, encode_action_descriptor
from motive.source_aware_features import (
    R5_ENDPOINT_LAYOUT,
    R5_FINAL_SCHEMA,
    R5_PHASH_SPLIT_VERSION,
    R5FeatureConfig,
    _commit_task,
    _extract_one,
    _feature_config_digest,
    _file_digest,
    _implementation_digest,
    _normalized_row,
    _object_digest,
    _root_contract,
    _validate_final,
    cluster_source_hashes,
    endpoint_blocks,
    finalize_tasks,
    instruction_hash_features,
    source_perceptual_fingerprint,
)
from motive.source_aware_repr import R5EndpointBatch


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EndpointPrimitiveTests(unittest.TestCase):
    def test_extract_one_decodes_pair_and_emits_raw_factorized_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, moving in (("source.mp4", False), ("target.mp4", True)):
                path = root / name
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    8.0,
                    (64, 64),
                )
                self.assertTrue(writer.isOpened())
                for frame_index in range(8):
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)
                    x = 12 + (3 * frame_index if moving else 0)
                    cv2.rectangle(frame, (x, 24), (x + 12, 36), (255, 255, 255), -1)
                    writer.write(frame)
                writer.release()
                self.assertTrue(path.is_file())
            raw = {
                "iid": "decode-pair",
                "prompt": "move the square to the right",
                "src_video": "source.mp4",
                "tgt_video": "target.mp4",
                "r5_pilot_label": {
                    "class": "positive",
                    "negative_type": "",
                    "action_signature": "move right",
                    "production_eligible": False,
                },
            }
            row = _normalized_row(raw, input_index=0, context="decode-test")
            result = _extract_one(
                {
                    "row": row,
                    "data_root": str(root),
                    "config": asdict(
                        R5FeatureConfig(
                            analysis_frames=8,
                            resize_width=64,
                            instruction_dim=32,
                        )
                    ),
                }
            )
            self.assertEqual(np.asarray(result["source_camera"]).shape, (8,))
            self.assertEqual(np.asarray(result["target_camera"]).shape, (8,))
            self.assertGreater(len(np.asarray(result["source_actor"])), 8)
            self.assertTrue(result["manifest"]["source_perceptual_hash"])
            self.assertEqual(
                len(result["manifest"]["source_video_sha256"]),
                64,
            )

    def test_endpoint_blocks_are_raw_descriptor_tail_split(self) -> None:
        residual = np.zeros((3, 8, 8, 2), dtype=np.float32)
        residual[:, 2:5, 2:5, 0] = 0.75
        camera = np.zeros_like(residual)
        camera[..., 1] = 0.25
        analysis = SimpleNamespace(
            residual_flows=residual,
            global_flows=camera,
            frame_times=np.asarray([0.0, 0.2, 0.4, 0.6], dtype=np.float32),
            frames_gray=np.zeros((4, 8, 8), dtype=np.uint8),
        )
        config = DescriptorConfig(
            temporal_bins=2,
            grid_rows=1,
            grid_cols=1,
            orientation_bins=4,
        )
        raw = encode_action_descriptor(
            residual,
            analysis.frame_times,
            8,
            global_flows=camera,
            config=config,
            normalize=False,
        )
        actor, camera_endpoint = endpoint_blocks(
            analysis,
            descriptor_config=config,
        )
        np.testing.assert_array_equal(actor, raw[:-8])
        np.testing.assert_array_equal(camera_endpoint, raw[-8:])
        self.assertEqual(camera_endpoint.shape, (8,))
        self.assertNotAlmostEqual(float(np.linalg.norm(raw)), 1.0)

    def test_fingerprint_and_prompt_hash_are_deterministic(self) -> None:
        frames = np.stack(
            [
                np.tile(np.arange(48, dtype=np.uint8), (32, 1)),
                np.tile(np.arange(48, dtype=np.uint8), (32, 1)) + 3,
            ]
        )
        first = source_perceptual_fingerprint(frames, max_frames=2)
        second = source_perceptual_fingerprint(frames, max_frames=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first["sampled_frame_digest"]), 64)
        self.assertGreater(len(first["perceptual_hash"]), 0)
        hashed = instruction_hash_features(["a dog runs", "a dog runs"], feature_dim=32)
        np.testing.assert_array_equal(hashed[0], hashed[1])
        self.assertAlmostEqual(float(np.linalg.norm(hashed[0])), 1.0, places=6)

    def test_dsu_merges_near_hashes_transitively(self) -> None:
        result = cluster_source_hashes(
            exact_digests=[_sha(str(index)) for index in range(4)],
            perceptual_hashes=["00", "01", "03", "f0"],
            maximum_hamming_fraction=0.125,
        )
        self.assertEqual(result.groups, 2)
        self.assertEqual(result.group_ids[0], result.group_ids[1])
        self.assertEqual(result.group_ids[1], result.group_ids[2])
        self.assertNotEqual(result.group_ids[2], result.group_ids[3])
        self.assertGreaterEqual(result.near_phash_unions, 2)

    def test_pilot_labels_are_fail_closed_and_negative_is_audit_only(self) -> None:
        base = {
            "iid": "x",
            "prompt": "make the dog run",
            "src_video": "source.mp4",
            "tgt_video": "target.mp4",
        }
        negative = _normalized_row(
            {
                **base,
                "r5_pilot_label": {
                    "class": "negative",
                    "negative_type": "static",
                    "action_signature": "run",
                    "production_eligible": False,
                },
            },
            input_index=0,
            context="test",
        )
        self.assertEqual(negative["label_role"], "negative_audit")
        self.assertFalse(negative["eligible_positive"])
        self.assertEqual(negative["action_signature"], "negative:static")
        positive = _normalized_row(
            {
                **base,
                "r5_pilot_label": {
                    "class": "positive",
                    "negative_type": "",
                    "action_signature": "run",
                    "production_eligible": False,
                },
            },
            input_index=0,
            context="test",
        )
        self.assertEqual(positive["action_signature"], "run")
        self.assertTrue(positive["eligible_positive"])
        with self.assertRaisesRegex(ValueError, "cannot be production_eligible"):
            _normalized_row(
                {
                    **base,
                    "r5_pilot_label": {
                        "class": "positive",
                        "negative_type": "",
                        "action_signature": "run",
                        "production_eligible": True,
                    },
                },
                input_index=0,
                context="test",
            )


class CommittedPipelineTests(unittest.TestCase):
    def _result(
        self,
        row: dict[str, object],
        *,
        phash: str,
        exact: str,
        offset: float,
    ) -> dict[str, object]:
        manifest = {
            **row,
            "source_resolved_path": "/data/source.mp4",
            "target_resolved_path": "/data/target.mp4",
            "source_video_sha256": _sha(f"source-{offset}"),
            "target_video_sha256": _sha(f"target-{offset}"),
            "source_sampled_frame_digest": exact,
            "source_perceptual_hash": phash,
            "source_motion_label": "actor_motion",
            "target_motion_label": "actor_motion",
        }
        actor = np.asarray([offset, offset + 1.0, offset + 2.0], dtype=np.float32)
        camera = np.arange(8, dtype=np.float32) + offset
        return {
            "manifest": manifest,
            "source_actor": actor,
            "source_camera": camera,
            "target_actor": actor + 0.25,
            "target_camera": camera + 0.5,
        }

    def test_two_task_finalize_is_r5_loadable_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            data_root.mkdir()
            raw_rows: list[dict[str, object]] = []
            for index in range(4):
                label = (
                    {
                        "class": "positive",
                        "negative_type": "",
                        "action_signature": "run",
                        "production_eligible": False,
                    }
                    if index % 2 == 0
                    else {
                        "class": "negative",
                        "negative_type": "static",
                        "action_signature": "run",
                        "production_eligible": False,
                    }
                )
                raw_rows.append(
                    {
                        "iid": f"sample-{index}",
                        "prompt": f"prompt {index}",
                        "src_video": f"source-{index}.mp4",
                        "tgt_video": f"target-{index}.mp4",
                        "human_review": {"verdict": "valid_action"},
                        "qwen_evidence": {"visual": {"status": "ok"}},
                        "r5_pilot_label": label,
                    }
                )
            source_manifest = root / "source.jsonl"
            source_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            normalized = [
                _normalized_row(
                    row,
                    input_index=index,
                    context=f"test:{index}",
                )
                for index, row in enumerate(raw_rows)
            ]
            config = R5FeatureConfig(instruction_dim=32)
            root_contract = _root_contract(data_root)
            common = {
                "source_manifest": str(source_manifest.resolve()),
                "source_manifest_sha256": _file_digest(source_manifest),
                "source_rows": 4,
                "data_root_contract": root_contract,
                "data_root_digest": _object_digest(root_contract),
                "config": asdict(config),
                "config_digest": _feature_config_digest(config),
                "implementation_digest": _implementation_digest(),
                "task_count": 2,
                "partition": "input-index-modulo-task-count-v1",
            }
            output = root / "output"
            hashes = ["00", "01", "f0", "f1"]
            for task_index in range(2):
                indices = list(range(task_index, 4, 2))
                results = [
                    self._result(
                        normalized[index],
                        phash=hashes[index],
                        exact=_sha(f"frame-{index}"),
                        offset=float(index),
                    )
                    for index in indices
                ]
                _commit_task(
                    output / "tasks" / f"task-{task_index:03d}",
                    results=results,
                    provenance={
                        **common,
                        "task_index": task_index,
                    },
                )
            arguments = argparse.Namespace(
                output_dir=output,
                task_count=2,
                data_seed=7,
                train_fraction=0.7,
                validation_fraction=0.1,
                maximum_hamming_fraction=0.125,
                minimum_positive_train=0,
                minimum_positive_validation=0,
                minimum_positive_test=1,
                resume=False,
            )
            with self.assertRaisesRegex(ValueError, "lacks positive rows"):
                finalize_tasks(arguments)
            self.assertFalse((output / "final.npz").exists())
            arguments.minimum_positive_test = 0
            self.assertEqual(finalize_tasks(arguments), 0)
            validated = _validate_final(output)
            arrays = validated["arrays"]
            self.assertEqual(
                json.loads(str(arrays["metadata_json"].item()))[
                    "schema_version"
                ],
                R5_FINAL_SCHEMA,
            )
            self.assertEqual(
                set(str(value) for value in arrays["split_versions"]),
                {R5_PHASH_SPLIT_VERSION},
            )
            self.assertEqual(
                list(arrays["label_role"]),
                [
                    "positive_delta",
                    "negative_audit",
                    "positive_delta",
                    "negative_audit",
                ],
            )
            self.assertEqual(str(arrays["action_signatures"][1]), "negative:static")
            self.assertFalse(bool(np.any(arrays["production_eligible"])))
            batch = R5EndpointBatch.from_mapping(
                arrays,
                require_visual_clusters=False,
            )
            self.assertEqual(batch.source_actor.shape, (4, 3))
            self.assertEqual(batch.source_camera.shape, (4, 8))
            manifest_rows = validated["rows"]
            self.assertIn("human_review", manifest_rows[0])
            self.assertIn("qwen_evidence", manifest_rows[0])
            self.assertIn("input_digest", manifest_rows[0])
            self.assertEqual(
                json.loads(
                    (output / "summary.json").read_text(encoding="utf-8")
                )["endpoint_layout"],
                R5_ENDPOINT_LAYOUT,
            )
            arguments.resume = True
            self.assertEqual(finalize_tasks(arguments), 0)
            arguments.maximum_hamming_fraction = 0.10
            with self.assertRaisesRegex(ValueError, "different cluster/split"):
                finalize_tasks(arguments)

            with (output / "manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "manifest digest mismatch"):
                _validate_final(output)

    def test_empty_task_fails_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "at least one row"):
                _commit_task(
                    Path(temporary) / "task",
                    results=[],
                    provenance={"config": asdict(R5FeatureConfig())},
                )


if __name__ == "__main__":
    unittest.main()
