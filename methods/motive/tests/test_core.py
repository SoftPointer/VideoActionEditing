from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from motive.archive import (
    assert_archives_compatible,
    build_feature_metadata,
    load_feature_archive,
    save_feature_archive,
)
from motive.audit import main as audit_main
from motive.descriptor import encode_action_delta, encode_action_descriptor
from motive.geometry import (
    MotionConfig,
    MotionMetrics,
    analyze_video,
    classify_motion,
    delta_motion_mask,
    normalize_motion_magnitude,
)
from motive.goku_manifest import main as goku_manifest_main
from motive.rank import main as rank_main
from motive.selection import majority_vote_select, rank_by_query
from motive.semantics import classify_instruction


HAS_TORCH = importlib.util.find_spec("torch") is not None


def metrics(**overrides: float | int) -> MotionMetrics:
    values = {
        "raw_speed_mean": 0.0,
        "raw_speed_p90": 0.0,
        "residual_speed_mean": 0.0,
        "residual_speed_p90": 0.0,
        "residual_speed_p99": 0.0,
        "active_pixel_fraction": 0.0,
        "active_frame_fraction": 0.0,
        "camera_explained_ratio": 0.0,
        "affine_inlier_ratio": 0.0,
        "scene_cut_ratio": 0.0,
        "temporal_energy_cv": 0.0,
        "sampled_frames": 8,
        "duration_seconds": 1.0,
        "source_fps": 8.0,
        "source_frame_count": 8,
        "source_width": 64,
        "source_height": 48,
    }
    values.update(overrides)
    return MotionMetrics(**values)


class GeometryTests(unittest.TestCase):
    def test_zero_motion_guard(self) -> None:
        values = np.zeros((3, 4, 5), dtype=np.float32)
        normalized = normalize_motion_magnitude(values, mode="motive")
        np.testing.assert_array_equal(normalized, values)

    def test_robust_normalization_and_delta(self) -> None:
        source = np.zeros((2, 3, 4, 2), dtype=np.float32)
        target = source.copy()
        target[:, 1:, 2:, 0] = 2.0
        mask = delta_motion_mask(source, target)
        self.assertEqual(mask.shape, source.shape[:-1])
        self.assertGreater(float(mask.max()), 0.9)
        self.assertEqual(float(mask[:, 0, :].max()), 0.0)

        sparse = np.zeros((10, 10), dtype=np.float32)
        sparse[4, 7] = 1.0
        sparse_mask = normalize_motion_magnitude(sparse, mode="robust")
        self.assertGreater(float(sparse_mask[4, 7]), 0.9)

    def test_conservative_classification(self) -> None:
        self.assertEqual(classify_motion(metrics()), "static")
        self.assertEqual(
            classify_motion(
                metrics(
                    raw_speed_mean=0.01,
                    camera_explained_ratio=0.9,
                    residual_speed_p90=0.001,
                )
            ),
            "camera_only",
        )
        self.assertEqual(
            classify_motion(
                metrics(
                    residual_speed_p90=0.02,
                    active_pixel_fraction=0.2,
                )
            ),
            "dynamic_object",
        )
        self.assertEqual(
            classify_motion(metrics(scene_cut_ratio=0.5)),
            "cut_or_decode_artifact",
        )
        self.assertEqual(
            classify_motion(metrics(scene_cut_ratio=1.0 / 7.0)),
            "cut_or_decode_artifact",
        )

    def test_action_descriptor_and_delta(self) -> None:
        flows = np.zeros((8, 12, 16, 2), dtype=np.float32)
        flows[:4, :, :, 0] = 1.0
        flows[4:, :, :, 1] = 2.0
        times = np.linspace(0.0, 1.0, num=9, dtype=np.float32)
        descriptor = encode_action_descriptor(flows, times, 16)
        self.assertGreater(len(descriptor), 128)
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0, places=5)
        opposite = encode_action_descriptor(-flows, times, 16)
        delta = encode_action_delta(descriptor, opposite)
        self.assertAlmostEqual(float(np.linalg.norm(delta)), 1.0, places=5)
        self.assertLess(float(descriptor @ opposite), 0.95)

    def test_near_zero_flow_does_not_become_a_unit_action(self) -> None:
        rng = np.random.default_rng(17)
        first = rng.normal(0.0, 1e-12, size=(8, 12, 16, 2)).astype(np.float32)
        second = rng.normal(0.0, 1e-12, size=(8, 12, 16, 2)).astype(np.float32)
        times = np.linspace(0.0, 1.0, num=9, dtype=np.float32)
        first_descriptor = encode_action_descriptor(first, times, 16)
        second_descriptor = encode_action_descriptor(second, times, 16)
        self.assertEqual(float(np.linalg.norm(first_descriptor)), 0.0)
        self.assertEqual(float(np.linalg.norm(second_descriptor)), 0.0)
        self.assertEqual(
            float(np.linalg.norm(first_descriptor - second_descriptor)),
            0.0,
        )

        one_pixel_outlier = np.zeros((8, 12, 16, 2), dtype=np.float32)
        one_pixel_outlier[0, 0, 0, 0] = 1.0
        outlier_descriptor = encode_action_descriptor(
            one_pixel_outlier,
            times,
            16,
        )
        self.assertEqual(float(np.linalg.norm(outlier_descriptor)), 0.0)


class SelectionTests(unittest.TestCase):
    def test_single_and_majority_query_ranking(self) -> None:
        features = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices, scores = rank_by_query(features, np.asarray([1.0, 0.0]), top_k=2)
        self.assertEqual(indices.tolist(), [0, 1])
        self.assertGreater(float(scores[0]), float(scores[1]))

        result = majority_vote_select(
            features,
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            vote_percentile=50.0,
            top_k=2,
        )
        self.assertEqual(len(result.indices), 2)
        self.assertEqual(result.score_matrix.shape, (2, 4))

    def test_tied_scores_get_rank_budget_and_zero_features_are_excluded(self) -> None:
        features = np.vstack(
            (
                np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1)),
                np.zeros((1, 2), dtype=np.float32),
            )
        )
        result = majority_vote_select(
            features,
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            vote_percentile=90.0,
            top_k=2,
        )
        self.assertEqual(result.indices.tolist(), [0, 1])
        self.assertEqual(result.votes[:2].tolist(), [2, 2])
        self.assertEqual(int(result.votes[-1]), 0)
        self.assertNotIn(20, result.indices.tolist())


class SemanticTests(unittest.TestCase):
    def test_action_endpoint_and_suppression(self) -> None:
        self.assertEqual(
            classify_instruction(
                "Make the person walk up the stairs continuously."
            ).label,
            "continuous_action",
        )
        self.assertEqual(
            classify_instruction(
                "Make the skier appear to be jumping, suspended in mid-air."
            ).label,
            "endpoint_pose",
        )
        self.assertEqual(
            classify_instruction(
                "Make the runner stop and stand still on the path."
            ).label,
            "motion_suppression",
        )
        self.assertEqual(
            classify_instruction(
                "Reshape the chain into a thicker circular loop."
            ).label,
            "shape_appearance",
        )
        self.assertEqual(
            classify_instruction(
                "Increase the river flow and make the waterfall more turbulent."
            ).label,
            "environmental_motion",
        )
        self.assertEqual(
            classify_instruction(
                "Lower the car's ride height significantly."
            ).label,
            "rigid_transform",
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch optional dependency is not installed")
class AttributionTests(unittest.TestCase):
    def test_motion_weighted_loss_and_projection(self) -> None:
        import torch

        from motive.attribution import (
            CountSketchProjector,
            action_edit_measurement_loss,
            align_motion_mask_to_latents,
            factorial_edit_tangents,
            motion_weighted_mse,
            normalize_motion_magnitude_torch,
            project_parameter_gradients,
        )

        prediction = torch.ones(1, 2, 3, 4, 4)
        target = torch.zeros_like(prediction)
        mask = torch.zeros(1, 3, 4, 4)
        mask[:, :, :, :2] = 1.0
        paper_loss = motion_weighted_mse(prediction, target, mask)
        active_loss = motion_weighted_mse(
            prediction,
            target,
            mask,
            reduction="active_mean",
        )
        self.assertAlmostEqual(float(paper_loss), 0.5, places=6)
        self.assertAlmostEqual(float(active_loss), 1.0, places=6)
        normalized = normalize_motion_magnitude_torch(torch.zeros(2, 3, 4, 4))
        self.assertEqual(float(normalized.sum()), 0.0)
        integer_motion = torch.tensor([[[0, 1], [2, 3]]])
        integer_normalized = normalize_motion_magnitude_torch(
            integer_motion,
            mode="motive",
        )
        self.assertTrue(integer_normalized.is_floating_point())
        self.assertGreater(len(torch.unique(integer_normalized)), 2)

        tensors = [torch.arange(10.0), torch.arange(5.0)]
        projector = CountSketchProjector(output_dim=8, seed=7, chunk_size=4)
        first = projector.project(tensors)
        second = projector.project(tensors)
        self.assertTrue(torch.equal(first, second))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(first)), 1.0, places=6)

        concatenated = torch.cat(tensors)
        different_chunks = CountSketchProjector(
            output_dim=8,
            seed=7,
            chunk_size=7,
        ).project([concatenated])
        self.assertTrue(torch.allclose(first, different_chunks, atol=1e-6))

        other_seed = CountSketchProjector(
            output_dim=8,
            seed=8,
            chunk_size=4,
        ).project(tensors)
        self.assertFalse(torch.allclose(first, other_seed))
        self.assertFalse(
            any(
                torch.allclose(torch.roll(first, shifts), other_seed)
                for shifts in range(len(first))
            )
        )

        low_indices = torch.arange(32, dtype=torch.int64)
        high_indices = low_indices + 2**32
        self.assertFalse(
            torch.equal(
                projector._hash32(low_indices, salt=3),
                projector._hash32(high_indices, salt=3),
            )
        )

        raw = projector.project_raw(tensors)
        self.assertGreater(float(torch.linalg.vector_norm(raw)), 0.0)
        self.assertTrue(
            torch.allclose(
                raw / torch.linalg.vector_norm(raw),
                first,
                atol=1e-6,
            )
        )

        class SlotModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = torch.nn.Parameter(torch.arange(4.0))
                self.unused = torch.nn.Parameter(torch.ones(3))
                self.last = torch.nn.Parameter(torch.arange(2.0))

        slot_model = SlotModel()
        (slot_model.first.sum() + slot_model.last.sum()).backward()
        slotted, diagnostics = project_parameter_gradients(
            slot_model,
            projector=CountSketchProjector(output_dim=8, seed=7),
            normalize=False,
        )
        explicit = CountSketchProjector(output_dim=8, seed=7).project_slots(
            [
                (slot_model.first.grad, slot_model.first.numel()),
                (None, slot_model.unused.numel()),
                (slot_model.last.grad, slot_model.last.numel()),
            ],
            normalize=False,
        )
        self.assertTrue(torch.equal(slotted, explicit))
        self.assertEqual(diagnostics["missing_gradient_names"], ["unused"])

        cells = {
            "tc": torch.tensor([3.0, 2.0]),
            "sc": torch.tensor([1.0, 1.0]),
            "t0": torch.tensor([2.0, 3.0]),
            "s0": torch.tensor([1.0, 1.0]),
        }
        tangents = factorial_edit_tangents(cells)
        expected_did = torch.tensor([1.0, -1.0])
        expected_did = expected_did / torch.linalg.vector_norm(expected_did)
        self.assertTrue(torch.allclose(tangents["factorial_did"], expected_did))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(tangents["paired_delta"])),
            1.0,
            places=6,
        )

        pairwise_mask = torch.zeros(1, 80, 16, 24)
        pairwise_mask[:, 39, 7:9, 10:12] = 1.0
        latent_mask = align_motion_mask_to_latents(
            pairwise_mask,
            target_frames=21,
            target_height=2,
            target_width=3,
            input_timing="pairwise",
        )
        self.assertEqual(tuple(latent_mask.shape), (1, 21, 2, 3))
        self.assertGreater(float(latent_mask.max()), 0.0)

        tiny_mask = torch.zeros(1, 3, 4, 4)
        tiny_mask[:, 1, 1, 1] = 1.0
        balanced = action_edit_measurement_loss(
            prediction,
            target,
            tiny_mask,
            preservation_prediction=prediction,
            preservation_target=target,
        )
        self.assertAlmostEqual(float(balanced), 1.25, places=5)
        with self.assertRaises(ValueError):
            action_edit_measurement_loss(
                prediction,
                target,
                torch.zeros_like(tiny_mask),
            )


class VideoAuditTests(unittest.TestCase):
    @staticmethod
    def _write_video(path: Path, moving: bool) -> None:
        width, height, fps, frames = 96, 64, 12.0, 24
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("test VideoWriter could not be opened")
        rng = np.random.default_rng(3)
        background = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
        for index in range(frames):
            frame = background.copy()
            x_coordinate = 8 + (2 * index if moving else 0)
            x_coordinate = min(x_coordinate, width - 20)
            cv2.rectangle(
                frame,
                (x_coordinate, 22),
                (x_coordinate + 16, 38),
                (255, 255, 255),
                thickness=-1,
            )
            writer.write(frame)
        writer.release()

    def test_end_to_end_directory_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static_path = root / "static.avi"
            moving_path = root / "moving.avi"
            self._write_video(static_path, moving=False)
            self._write_video(moving_path, moving=True)
            config = MotionConfig(analysis_frames=20, resize_width=128)
            self.assertEqual(analyze_video(static_path, config).label, "static")
            self.assertEqual(analyze_video(moving_path, config).label, "dynamic_object")

            output = root / "audit"
            status = audit_main(
                [
                    "--input",
                    str(root),
                    "--output-dir",
                    str(output),
                    "--analysis-frames",
                    "20",
                    "--resize-width",
                    "128",
                ]
            )
            self.assertEqual(status, 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["selected"], 1)
            self.assertTrue((output / "descriptors.npz").is_file())
            _, _, metadata = load_feature_archive(output / "descriptors.npz")
            self.assertEqual(
                metadata["feature_kind"],
                "geometry_action_descriptor",
            )


class GokuManifestTests(unittest.TestCase):
    def test_manifest_normalizes_relative_pair_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "jsons" / "combine_json"
            combined.mkdir(parents=True)
            raw = {
                "case_id": "case-7",
                "source_video": "videos/case-7/source.mp4",
                "edited_video": "videos/case-7/edited.mp4",
                "instruction_en": "Make the person walk across the room.",
                "source_caption": "source",
                "edited_caption": "edited",
            }
            (combined / "case-7_all.json").write_text(json.dumps(raw))
            output = root / "manifest.jsonl"
            status = goku_manifest_main(
                [
                    "--dataset-root",
                    str(root),
                    "--output",
                    str(output),
                    "--sample-size",
                    "1",
                ]
            )
            self.assertEqual(status, 0)
            row = json.loads(output.read_text())
            self.assertEqual(row["iid"], "case-7")
            self.assertEqual(row["src_video"], raw["source_video"])
            self.assertEqual(row["tgt_video"], raw["edited_video"])
            self.assertEqual(
                row["instruction_semantics"]["label"],
                "continuous_action",
            )


class ArchiveTests(unittest.TestCase):
    def test_archive_provenance_is_required_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = build_feature_metadata(
                feature_kind="geometry_action_descriptor",
                dimension=2,
                provenance={"descriptor_version": "test-v1"},
            )
            path = root / "features.npz"
            save_feature_archive(
                path,
                features=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ids=np.asarray(["one"]),
                metadata=metadata,
            )
            _, _, loaded = load_feature_archive(path)
            assert_archives_compatible(metadata, loaded)
            incompatible = build_feature_metadata(
                feature_kind="geometry_action_descriptor",
                dimension=2,
                provenance={"descriptor_version": "test-v2"},
            )
            with self.assertRaises(ValueError):
                assert_archives_compatible(metadata, incompatible)

            legacy = root / "legacy.npz"
            np.savez_compressed(
                legacy,
                features=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ids=np.asarray(["one"]),
            )
            with self.assertRaises(ValueError):
                load_feature_archive(legacy)

    def test_single_query_rank_fraction_uses_only_valid_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = build_feature_metadata(
                feature_kind="geometry_action_descriptor",
                dimension=2,
                provenance={"descriptor_version": "test-v1"},
            )
            candidates = np.zeros((100, 2), dtype=np.float32)
            candidates[:5, 0] = 1.0
            feature_path = root / "features.npz"
            query_path = root / "query.npz"
            save_feature_archive(
                feature_path,
                features=candidates,
                ids=np.asarray([str(index) for index in range(100)]),
                metadata=metadata,
            )
            save_feature_archive(
                query_path,
                features=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ids=np.asarray(["query"]),
                metadata=metadata,
            )
            output = root / "rank.jsonl"
            status = rank_main(
                [
                    "--features",
                    str(feature_path),
                    "--queries",
                    str(query_path),
                    "--output",
                    str(output),
                    "--single-query",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_mixed_single_and_paired_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "mixed.jsonl"
            rows = [
                {"iid": "single", "tgt_video": "one.mp4"},
                {
                    "iid": "paired",
                    "src_video": "source.mp4",
                    "tgt_video": "target.mp4",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaises(ValueError):
                audit_main(
                    [
                        "--input",
                        str(manifest),
                        "--output-dir",
                        str(root / "output"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
