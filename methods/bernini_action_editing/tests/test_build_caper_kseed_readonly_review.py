#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_caper_kseed_readonly_review as review  # noqa: E402


SOURCE_IDS = (
    "1111111111111111",
    "2222222222222222",
    "3333333333333333",
    "4444444444444444",
)
SEEDS = (101, 102, 103, 104)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(review.canonical_json_bytes(value) + b"\n")


def _seal(value: dict[str, object]) -> dict[str, object]:
    return {**value, "receipt_digest": review.object_sha256(value)}


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): review.file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _ffprobe_result(
    _command: tuple[str, ...], *, frames: int = 81, fps: str = "25/1"
) -> subprocess.CompletedProcess[bytes]:
    payload = {
        "streams": [
            {
                "width": 496,
                "height": 480,
                "avg_frame_rate": fps,
                "nb_read_frames": str(frames),
            }
        ]
    }
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(payload).encode("utf-8"), stderr=b""
    )


class PopulationFixture:
    def __init__(self, parent: Path) -> None:
        self.input_root = parent / "input"
        self.output_root = parent / "portable-review"
        self.input_root.mkdir()
        self.registry_path = self.input_root / review.REGISTRY_NAME
        self.master_path = self.input_root / review.MASTER_NAME
        self.cells = [
            f"fit-{source_id}-s{seed}" for source_id in SOURCE_IDS for seed in SEEDS
        ]
        self._write_registry()
        children = [self._write_cell(cell_id) for cell_id in self.cells]
        self._write_master(children)

    def _write_registry(self) -> None:
        sources = []
        for source_index, source_id in enumerate(SOURCE_IDS, start=1):
            source = self.input_root / f"source-{source_id}.mp4"
            source.write_bytes(f"source-video-{source_id}".encode("ascii"))
            sources.append(
                {
                    "split": "fit",
                    "source_id": source_id,
                    "actor_kind": "dog" if source_index != 3 else "cat",
                    "identity_id": f"fixture-identity-{source_index}",
                    "scene_id": f"fixture-scene-{source_index}",
                    "source_video": f"/original/source-{source_id}.mp4",
                    "source_video_sha256": review.file_sha256(source),
                    "target_action_caption": f"Fixture action {source_index}.",
                }
            )
        registry = {
            "schema_version": review.REGISTRY_SCHEMA,
            "population_design": {
                "fit": {
                    "source_ids": list(SOURCE_IDS),
                    "seeds": list(SEEDS),
                    "cell_order": self.cells,
                    "expected_cell_count": len(self.cells),
                    "cartesian_population_required": True,
                    "seed_filtering_or_best_of_k_authorized": False,
                }
            },
            "sources": sources,
        }
        _write_json(self.registry_path, registry)

    def _write_cell(self, cell_id: str) -> dict[str, object]:
        _phase, source_id, seed_text = cell_id.split("-")
        seed = int(seed_text[1:])
        cell_root = self.input_root / cell_id
        cell_root.mkdir()
        video = cell_root / f"{review.NATIVE_ARM}.mp4"
        video.write_bytes(f"candidate-video-{cell_id}".encode("ascii"))
        video_sha = review.file_sha256(video)
        receipt = _seal(
            {
                "schema_version": review.CELL_SCHEMA,
                "cell_id": cell_id,
                "population_phase": "fit",
                "input": {"source_id": source_id},
                "sampling": {"seed": seed, "frame_count": 81, "fps": 25},
                "outputs": {
                    review.NATIVE_ARM: {
                        "path": f"/original/{cell_id}/{video.name}",
                        "sha256": video_sha,
                        "frame_count": 81,
                        "fps": 25,
                    }
                },
                "seed_filtering_or_best_of_k_authorized": False,
                "scientific_or_action_editing_claim_authorized": False,
                "training_performed": False,
                "optimizer_created": False,
                "parameter_update": False,
            }
        )
        receipt_path = cell_root / "receipt.json"
        _write_json(receipt_path, receipt)
        attempt = _seal(
            {
                "schema_version": review.ATTEMPT_SCHEMA,
                "cell_id": cell_id,
                "population_phase": "fit",
                "source_id": source_id,
                "seed": seed,
                "process_exit_code": 0,
                "attempt_success": True,
                "attempt_status": "completed_success",
                "seed_discarded": False,
                "retry_or_replacement_seed_authorized": False,
                "child_receipt_path": f"/original/{cell_id}/receipt.json",
                "child_receipt_file_sha256": review.file_sha256(receipt_path),
                "child_receipt_digest": receipt["receipt_digest"],
            }
        )
        attempt_path = self.input_root / "attempts" / f"{cell_id}.json"
        _write_json(attempt_path, attempt)
        return {
            "cell_id": cell_id,
            "source_id": source_id,
            "seed": seed,
            "attempt_receipt_sha256": review.file_sha256(attempt_path),
            "child_receipt_file_sha256": review.file_sha256(receipt_path),
            "child_receipt_digest": receipt["receipt_digest"],
            "mp4_sha256": video_sha,
        }

    def _write_master(self, children: list[dict[str, object]]) -> None:
        master = _seal(
            {
                "schema_version": review.MASTER_SCHEMA,
                "registered_cell_order": self.cells,
                "registered_cell_count": len(self.cells),
                "successful_cell_count": len(self.cells),
                "failed_cell_count": 0,
                "failed_attempts": [],
                "children": children,
                "population_complete": True,
                "population_decision": "PASS_COMPLETE",
                "seed_filtering_or_best_of_k_authorized": False,
                "retry_or_replacement_seed_authorized": False,
                "partial_population_scientific_claim_authorized": False,
                "training_performed": False,
                "optimizer_created": False,
                "parameter_update": False,
                "exact81": True,
                "fps": 25,
                "registry_file_sha256": review.file_sha256(self.registry_path),
            }
        )
        _write_json(self.master_path, master)

    def build(self) -> dict[str, object]:
        with mock.patch.object(
            review.subprocess, "run", side_effect=lambda command, **_kwargs: _ffprobe_result(command)
        ):
            return review.build_review(
                input_root=self.input_root,
                output_root=self.output_root,
                job_id="131678",
            )


class CaperKseedReadonlyReviewTests(unittest.TestCase):
    def test_builds_portable_exact_registered_order_without_semantic_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PopulationFixture(Path(temporary))
            before = _tree_snapshot(fixture.input_root)
            audit = fixture.build()
            self.assertEqual(_tree_snapshot(fixture.input_root), before)

            self.assertEqual(audit["source_count"], 4)
            self.assertEqual(audit["candidate_count"], 16)
            self.assertEqual(audit["registered_cell_order"], fixture.cells)
            self.assertEqual(audit["semantic_status"], "UNASSESSED")
            self.assertFalse(audit["semantic_pass_assigned"])
            self.assertFalse(audit["seed_filtering_or_best_of_k"])
            self.assertFalse(audit["candidate_ranking_or_selection_performed"])
            self.assertEqual(
                [[candidate["seed"] for candidate in row["candidates"]] for row in audit["rows"]],
                [list(SEEDS)] * 4,
            )

            page = (fixture.output_root / "index.html").read_text(encoding="utf-8")
            self.assertIn("Population completion is not a semantic pass", page)
            self.assertIn("best-of-K selection: <strong>disabled</strong>", page)
            self.assertIn("semantic status: <strong>UNASSESSED</strong>", page)
            self.assertEqual(page.count("Play together from 0"), 4)
            self.assertEqual(page.count("<video "), 20)
            self.assertNotIn(str(fixture.input_root), page)
            positions = [page.index(f'id="source-{source_id}"') for source_id in SOURCE_IDS]
            self.assertEqual(positions, sorted(positions))
            seed_positions = [page.index(f"Seed {seed}") for seed in SEEDS]
            self.assertEqual(seed_positions, sorted(seed_positions))

            copied = fixture.output_root / audit["rows"][0]["candidates"][0]["portable_video"]
            expected_sha = audit["rows"][0]["candidates"][0]["video_sha256"]
            self.assertEqual(review.file_sha256(copied), expected_sha)
            shutil.rmtree(fixture.input_root)
            self.assertTrue(copied.is_file())
            self.assertTrue((fixture.output_root / "review-audit.json").is_file())

    def test_refuses_master_reordering_even_when_receipt_is_resealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PopulationFixture(Path(temporary))
            master = json.loads(fixture.master_path.read_text(encoding="utf-8"))
            master.pop("receipt_digest")
            master["registered_cell_order"] = list(reversed(master["registered_cell_order"]))
            _write_json(fixture.master_path, _seal(master))
            with self.assertRaisesRegex(review.ReviewError, "reordered cells"):
                fixture.build()
            self.assertFalse(fixture.output_root.exists())

    def test_refuses_registry_that_authorizes_best_of_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PopulationFixture(Path(temporary))
            registry = json.loads(fixture.registry_path.read_text(encoding="utf-8"))
            registry["population_design"]["fit"]["seed_filtering_or_best_of_k_authorized"] = True
            _write_json(fixture.registry_path, registry)
            with self.assertRaisesRegex(review.ReviewError, "best-of-K is authorized"):
                fixture.build()
            self.assertFalse(fixture.output_root.exists())

    def test_probe_rejects_non_exact81(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"fixture")
            with mock.patch.object(
                review.subprocess,
                "run",
                side_effect=lambda command, **_kwargs: _ffprobe_result(command, frames=80),
            ):
                with self.assertRaisesRegex(review.ReviewError, "not exact81/25fps"):
                    review.probe_exact81(video)


if __name__ == "__main__":
    unittest.main()
