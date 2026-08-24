from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

import numpy as np

from motive.r10b_tangent_core import (
    R10BTangentError,
    build_smoke_manifest,
    split_directional_family,
    strict_atomic_family,
    track_delta_components,
    track_delta_saliency,
    validate_smoke_rows,
    validate_track_cache_arrays,
)


class R10BTangentCoreTests(unittest.TestCase):
    def test_directional_family_split_is_fail_closed(self) -> None:
        self.assertEqual(
            split_directional_family("open_close", "Open the wooden door."),
            "open",
        )
        self.assertEqual(
            split_directional_family("open_close", "Shut the wooden door."),
            "close",
        )
        self.assertIsNone(
            split_directional_family(
                "open_close",
                "Open the box and then close it.",
            )
        )
        self.assertEqual(split_directional_family("reach", "Reach left."), "reach")
        self.assertEqual(
            strict_atomic_family("sit_down", "Make the person sit down on the chair."),
            "sit_down",
        )
        self.assertIsNone(
            strict_atomic_family(
                "stand_up",
                "Make the person stand up and then turn to the left.",
            )
        )
        self.assertIsNone(
            strict_atomic_family(
                "open_close",
                "Make the person open the door and close it again.",
            )
        )

    def test_track_delta_saliency_rejects_global_camera_and_keeps_actor(self) -> None:
        frames, tracks = 8, 16
        grid = np.stack(
            np.meshgrid(
                np.linspace(0.1, 0.9, 4),
                np.linspace(0.1, 0.9, 4),
                indexing="xy",
            ),
            axis=-1,
        ).reshape(tracks, 2)
        source = np.repeat(grid[None], frames, axis=0)
        target = source.copy()
        # A global translation should be removed by robust coordinate median.
        target[:, :, 0] += np.arange(frames)[:, None] * 0.01
        # Four actor tracks move differently and must survive.
        target[:, :4, 1] += np.arange(frames)[:, None] * 0.03
        visibility = np.ones((frames, tracks), dtype=np.float32)
        velocity, magnitude, midpoint = track_delta_components(
            source,
            target,
            visibility,
            visibility,
        )
        self.assertEqual(velocity.shape, (frames - 1, tracks, 2))
        self.assertEqual(midpoint.shape, (frames - 1, tracks, 2))
        self.assertGreater(float(magnitude[:, :4].mean()), 0.01)
        self.assertLess(float(magnitude[:, 4:].mean()), 1e-5)

        mask, diagnostics = track_delta_saliency(
            source,
            target,
            visibility,
            visibility,
            height=16,
            width=16,
        )
        self.assertEqual(mask.shape, (frames - 1, 16, 16))
        self.assertGreater(float(mask.max()), 0.9)
        self.assertGreater(diagnostics["normalized_active_fraction"], 0.0)
        self.assertLess(diagnostics["normalized_active_fraction"], 0.5)

    def test_track_geometry_shape_validation(self) -> None:
        with self.assertRaises(R10BTangentError):
            track_delta_components(
                np.zeros((1, 2, 2)),
                np.zeros((1, 2, 2)),
                np.ones((1, 2)),
                np.ones((1, 2)),
            )

    def test_smoke_manifest_is_fresh_component_disjoint_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidates.jsonl"
            track_manifest_path = root / "track_manifest.jsonl"
            track_cache_path = root / "tracks.npz"
            rows = []
            track_rows = []
            frames, tracks = 6, 4
            source_tracks = np.zeros((2, frames, tracks, 2), dtype=np.float32)
            source_tracks[..., 0] = np.linspace(0.2, 0.8, tracks)
            source_tracks[..., 1] = 0.5
            target_tracks = source_tracks.copy()
            target_tracks[0, :, :2, 1] += np.arange(frames)[:, None] * 0.03
            target_tracks[1, :, 2:, 1] -= np.arange(frames)[:, None] * 0.03
            visibility = np.ones((2, frames, tracks), dtype=np.float32)
            prompts = (
                ("sit_down", "Make the person sit down on the chair."),
                ("lie_down", "Make the person lie down on the bed."),
            )
            for index, (family, prompt) in enumerate(prompts):
                iid = f"case-{index}"
                rows.append(
                    {
                        "iid": iid,
                        "input_digest": "a" * 64,
                        "prompt": prompt,
                        "label": {
                            "class": "positive",
                            "primary_family": family,
                            "provenance_kind": "synthetic-test",
                            "human_label": False,
                        },
                        "assignment": {
                            "fresh": True,
                            "split": "train",
                            "component_id": f"component-{index}",
                        },
                        "source_bindings": {
                            "media": {
                                "data_root": str(root),
                                "src_video": {
                                    "relative_path": f"{iid}/source.mp4",
                                    "sha256": "b" * 64,
                                },
                                "tgt_video": {
                                    "relative_path": f"{iid}/target.mp4",
                                    "sha256": "c" * 64,
                                },
                            }
                        },
                    }
                )
                track_rows.append(
                    {
                        "iid": iid,
                        "input_index": index,
                        "paired_camera_valid": True,
                    }
                )
            candidate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            track_manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in track_rows),
                encoding="utf-8",
            )
            np.savez_compressed(
                track_cache_path,
                input_indices=np.arange(2),
                source_stabilized_tracks=source_tracks,
                target_stabilized_tracks=target_tracks,
                source_visibility=visibility,
                target_visibility=visibility,
                source_camera_valid=np.ones(2, dtype=bool),
                target_camera_valid=np.ones(2, dtype=bool),
                source_track_valid=np.ones(2, dtype=bool),
                target_track_valid=np.ones(2, dtype=bool),
            )
            payload = build_smoke_manifest(
                candidate_manifest=candidate_path,
                track_cache=track_cache_path,
                track_manifest=track_manifest_path,
                families=("sit_down", "lie_down"),
                per_family=1,
            )
            self.assertEqual(payload["summary"]["rows"], 2)
            self.assertEqual(payload["summary"]["unique_components"], 2)
            self.assertEqual(payload["summary"]["legacy_test_rows"], 0)
            self.assertFalse(payload["summary"]["formal_evidence"])
            validate_smoke_rows(payload["rows"])

            invalid_rows = [dict(row) for row in payload["rows"]]
            invalid_rows[0]["src_video"] = "../escape.mp4"
            with self.assertRaises(R10BTangentError):
                validate_smoke_rows(invalid_rows)

            duplicate_components = [dict(row) for row in payload["rows"]]
            duplicate_components[1]["component_id"] = duplicate_components[0][
                "component_id"
            ]
            with self.assertRaises(R10BTangentError):
                validate_smoke_rows(duplicate_components)

            rows[0]["source_bindings"]["media"]["src_video"] = "not-a-record"
            candidate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                R10BTangentError,
                "family support shortfall",
            ):
                build_smoke_manifest(
                    candidate_manifest=candidate_path,
                    track_cache=track_cache_path,
                    track_manifest=track_manifest_path,
                    families=("sit_down", "lie_down"),
                    per_family=1,
                )

            invalid_cache = {
                "input_indices": np.asarray([0, 0]),
                "source_stabilized_tracks": source_tracks,
                "target_stabilized_tracks": target_tracks,
                "source_visibility": visibility,
                "target_visibility": visibility,
                "source_camera_valid": np.ones(2, dtype=bool),
                "target_camera_valid": np.ones(2, dtype=bool),
                "source_track_valid": np.ones(2, dtype=bool),
                "target_track_valid": np.ones(2, dtype=bool),
            }
            with self.assertRaises(R10BTangentError):
                validate_track_cache_arrays(invalid_cache)


if __name__ == "__main__":
    unittest.main()
