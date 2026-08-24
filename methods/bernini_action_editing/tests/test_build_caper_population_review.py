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

import build_caper_population_review as review  # noqa: E402


SOURCE_ID = "7b88a1ca1f804f41"
SEED = 2026081801
CELL_ID = f"fit-{SOURCE_ID}-s{SEED}"
GAUSSIAN_SHA = "9" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(review.canonical_json_bytes(value) + b"\n")


def _seal(unsigned: dict[str, object]) -> dict[str, object]:
    return {**unsigned, "receipt_digest": review.object_sha256(unsigned)}


class PopulationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.source_root = self.root / "sources"
        self.source = self.source_root / SOURCE_ID / "source_video.mp4"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"source-exact81-fixture")

        self.native_root = self.root / "native"
        self.sibling_root = self.root / "sibling"
        self.native_root.mkdir()
        self.sibling_root.mkdir()
        self.native_registry = self.root / "native-registry.json"
        self.sibling_registry = self.root / "sibling-registry.json"
        self.output = self.root / "review" / "index.html"
        self._write_registries()
        self._write_native_population()
        self._write_sibling_population()

    def _write_registries(self) -> None:
        source_row = {
            "split": "fit",
            "source_id": SOURCE_ID,
            "actor_kind": "dog",
            "identity_id": "fixture-dog",
            "scene_id": "fixture-park",
            "source_video": "/relocated/original/source_video.mp4",
            "source_video_sha256": review.file_sha256(self.source),
        }
        native = {
            "schema_version": "bernini-caper-native-kseed-population-sit-v1",
            "population_design": {
                "fit": {
                    "cell_order": [CELL_ID],
                    "expected_cell_count": 1,
                    "cartesian_population_required": True,
                    "seed_filtering_or_best_of_k_authorized": False,
                }
            },
            "sources": [source_row],
        }
        sibling_source = {
            **source_row,
            "captions": {arm: f"fixture {arm}" for arm in review.ARM_ORDER},
        }
        sibling = {
            "schema_version": (
                "bernini-caper-native-counterfactual-sibling-population-sit-v1"
            ),
            "arm_order": list(review.ARM_ORDER),
            "population_design": {
                "split": "fit",
                "cell_order": [CELL_ID],
                "expected_cell_count": 1,
                "arms_per_cell": 4,
                "cartesian_population_required": True,
                "seed_filtering_or_best_of_k_authorized": False,
                "replacement_seed_authorized": False,
            },
            "source": sibling_source,
        }
        _write_json(self.native_registry, native)
        _write_json(self.sibling_registry, sibling)

    def _write_native_population(self) -> None:
        cell_root = self.native_root / CELL_ID
        cell_root.mkdir(parents=True)
        target = cell_root / f"{review.NATIVE_ARM}.mp4"
        target.write_bytes(b"independent-native-target")
        receipt = _seal(
            {
                "schema_version": review.NATIVE_CELL_SCHEMA,
                "cell_id": CELL_ID,
                "population_phase": "fit",
                "seed_filtering_or_best_of_k_authorized": False,
                "input": {"source_id": SOURCE_ID},
                "sampling": {
                    "seed": SEED,
                    "frame_count": 81,
                    "fps": 25,
                    "official_gaussian_raw_sha256": GAUSSIAN_SHA,
                },
                "outputs": {
                    review.NATIVE_ARM: {
                        "path": f"/original/run/{CELL_ID}/{target.name}",
                        "sha256": review.file_sha256(target),
                        "frame_count": 81,
                        "fps": 25,
                    }
                },
                "training_performed": False,
                "optimizer_created": False,
                "parameter_update": False,
            }
        )
        receipt_path = cell_root / "receipt.json"
        _write_json(receipt_path, receipt)
        attempt = _seal(
            {
                "schema_version": review.NATIVE_ATTEMPT_SCHEMA,
                "cell_id": CELL_ID,
                "population_phase": "fit",
                "source_id": SOURCE_ID,
                "seed": SEED,
                "process_exit_code": 0,
                "attempt_success": True,
                "attempt_status": "completed_success",
                "seed_attempt_recorded_even_on_failure": True,
                "seed_discarded": False,
                "retry_or_replacement_seed_authorized": False,
                "child_receipt_path": f"/original/run/{CELL_ID}/receipt.json",
                "child_receipt_file_sha256": review.file_sha256(receipt_path),
                "child_receipt_digest": receipt["receipt_digest"],
            }
        )
        _write_json(self.native_root / "attempts" / f"{CELL_ID}.json", attempt)
        master = _seal(
            {
                "schema_version": review.NATIVE_MASTER_SCHEMA,
                "registered_cell_order": [CELL_ID],
                "population_complete": True,
                "population_decision": "PASS_COMPLETE",
                "failed_attempts": [],
                "seed_filtering_or_best_of_k_authorized": False,
            }
        )
        _write_json(self.native_root / "fit-population-receipt.json", master)

    def _write_sibling_population(self) -> None:
        cell_root = self.sibling_root / CELL_ID
        cell_root.mkdir(parents=True)
        pointers = []
        for arm_index, arm in enumerate(review.ARM_ORDER):
            video = cell_root / f"{arm}.mp4"
            video.write_bytes(f"sibling-{arm}".encode("ascii"))
            arm_receipt = _seal(
                {
                    "schema_version": review.SIBLING_ARM_SCHEMA,
                    "cell_id": CELL_ID,
                    "arm": arm,
                    "arm_index": arm_index,
                    "source_id": SOURCE_ID,
                    "seed": SEED,
                    "sampling": {
                        "seed": SEED,
                        "frame_count": 81,
                        "fps": 25,
                        "official_gaussian_raw_sha256": GAUSSIAN_SHA,
                    },
                    "output": {
                        "path": f"/original/run/{CELL_ID}/{video.name}",
                        "sha256": review.file_sha256(video),
                        "frame_count": 81,
                        "fps": 25,
                    },
                    "training_performed": False,
                    "optimizer_created": False,
                    "parameter_update": False,
                    "preference_admission_performed": False,
                }
            )
            arm_path = cell_root / f"{arm}.receipt.json"
            _write_json(arm_path, arm_receipt)
            pointers.append(
                {
                    "arm": arm,
                    "path": f"/original/run/{CELL_ID}/{arm_path.name}",
                    "file_sha256": review.file_sha256(arm_path),
                    "receipt_digest": arm_receipt["receipt_digest"],
                }
            )
        cell_receipt = _seal(
            {
                "schema_version": review.SIBLING_CELL_SCHEMA,
                "cell_id": CELL_ID,
                "source_id": SOURCE_ID,
                "seed": SEED,
                "arm_order": list(review.ARM_ORDER),
                "expected_arm_count": 4,
                "complete_arm_count": 4,
                "all_four_sibling_arms_complete": True,
                "shared_contract": {
                    "official_gaussian_raw_sha256": GAUSSIAN_SHA,
                },
                "arm_receipts": pointers,
                "training_performed": False,
                "optimizer_created": False,
                "parameter_update": False,
                "preference_admission_performed": False,
                "partial_population_scientific_claim_authorized": False,
            }
        )
        cell_receipt_path = cell_root / "receipt.json"
        _write_json(cell_receipt_path, cell_receipt)
        attempt = self.sibling_attempt(
            success=True,
            exit_code=0,
            receipt_path=cell_receipt_path,
            receipt_digest=str(cell_receipt["receipt_digest"]),
        )
        _write_json(self.sibling_root / "attempts" / f"{CELL_ID}.json", attempt)
        self.write_sibling_master(complete=True)

    def sibling_attempt(
        self,
        *,
        success: bool,
        exit_code: int,
        receipt_path: Path | None = None,
        receipt_digest: str | None = None,
    ) -> dict[str, object]:
        return _seal(
            {
                "schema_version": review.SIBLING_ATTEMPT_SCHEMA,
                "cell_id": CELL_ID,
                "source_id": SOURCE_ID,
                "seed": SEED,
                "arm_order": list(review.ARM_ORDER),
                "expected_arm_count": 4,
                "process_exit_code": exit_code,
                "attempt_success": success,
                "attempt_status": (
                    "completed_success" if success else "completed_failure"
                ),
                "cell_process_attempt_recorded_even_on_failure": True,
                "all_four_arm_outcomes_closed": success,
                "unobserved_or_incomplete_arm_outcomes_possible": not success,
                "seed_discarded": False,
                "retry_or_replacement_seed_authorized": False,
                "partial_or_complete_cell_artifacts": [
                    {"name": "partial.marker", "size_bytes": 1}
                ]
                if not success
                else [],
                "cell_receipt_path": str(receipt_path) if receipt_path else None,
                "cell_receipt_file_sha256": (
                    review.file_sha256(receipt_path) if receipt_path else None
                ),
                "cell_receipt_digest": receipt_digest,
            }
        )

    def write_sibling_master(self, *, complete: bool) -> None:
        master = _seal(
            {
                "schema_version": review.SIBLING_MASTER_SCHEMA,
                "cell_order": [CELL_ID],
                "population_complete": complete,
                "population_decision": (
                    "PASS_COMPLETE"
                    if complete
                    else "NO_GO_INCOMPLETE_OR_FAILED_ATTEMPTS"
                ),
                "failed_attempts": [] if complete else [{"cell_id": CELL_ID}],
                "seed_filtering_or_best_of_k_authorized": False,
            }
        )
        _write_json(self.sibling_root / "population-receipt.json", master)

    def build(self, *, ffprobe: str = "ffprobe") -> dict[str, object]:
        return review.build_review(
            native_registry_path=self.native_registry,
            native_root=self.native_root,
            sibling_registry_path=self.sibling_registry,
            sibling_root=self.sibling_root,
            phase="fit",
            source_root=self.source_root,
            output_html=self.output,
            ffprobe=ffprobe,
        )


def _ffprobe_result(command: tuple[str, ...], *, bad_suffix: str | None = None):
    frames = "80" if bad_suffix and str(command[-1]).endswith(bad_suffix) else "81"
    payload = {
        "streams": [
            {
                "width": 496,
                "height": 480,
                "avg_frame_rate": "25/1",
                "nb_read_frames": frames,
            }
        ]
    }
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=b"",
    )


class CaperPopulationReviewTests(unittest.TestCase):
    def test_real_ffprobe_accepts_81_frames_and_rejects_80(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            self.skipTest("ffmpeg/ffprobe are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for count in (81, 80):
                path = root / f"frames-{count}.mp4"
                completed = subprocess.run(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=black:s=16x16:r=25",
                        "-frames:v",
                        str(count),
                        "-an",
                        "-c:v",
                        "mpeg4",
                        "-q:v",
                        "2",
                        str(path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr.decode("utf-8", errors="replace"),
                )
                videos.append(path)
            probed = review.probe_exact81_video(videos[0], ffprobe=ffprobe)
            self.assertEqual(probed["frame_count"], 81)
            self.assertEqual(probed["fps"], 25)
            with self.assertRaisesRegex(review.CaperReviewError, "not exact81/25fps"):
                review.probe_exact81_video(videos[1], ffprobe=ffprobe)

    def test_complete_population_validates_every_video_and_renders_sync_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PopulationFixture(Path(directory))
            calls: list[tuple[str, ...]] = []

            def fake_run(command, **_kwargs):
                calls.append(tuple(command))
                return _ffprobe_result(tuple(command))

            with mock.patch.object(review.subprocess, "run", side_effect=fake_run):
                audit = fixture.build()

            self.assertTrue(audit["review_complete"])
            self.assertEqual(audit["selection_policy"]["seed_filtering_or_best_of_k"], False)
            self.assertFalse(
                audit["selection_policy"][
                    "fallback_to_native_target_when_sibling_attempt_fails"
                ]
            )
            self.assertEqual(len(audit["cells"]), 1)
            cell = audit["cells"][0]
            self.assertEqual(cell["source"]["status"], "valid")
            self.assertEqual(
                [cell["roles"][arm]["status"] for arm in review.ARM_ORDER],
                ["valid"] * 4,
            )
            self.assertEqual(cell["native_target_duplicate"]["status"], "valid")
            self.assertEqual(cell["native_sibling_target_coordinate"]["status"], "valid")
            # Source + independent native target + all four registered sibling arms.
            self.assertEqual(len(calls), 6)
            page = fixture.output.read_text(encoding="utf-8")
            self.assertIn('id="play-all"', page)
            self.assertIn('id="pause-all"', page)
            self.assertIn('id="seek"', page)
            self.assertIn("Phase-order violation", page)
            self.assertIn("no best-of-K", page)
            self.assertEqual(page.count("<video "), 5)
            audit_path = fixture.output.with_name("index.audit.json")
            self.assertTrue(audit_path.is_file())
            persisted = json.loads(audit_path.read_text(encoding="ascii"))
            self.assertEqual(persisted["audit_digest"], audit["audit_digest"])

    def test_failed_sibling_attempt_is_visible_and_never_falls_back_to_native_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PopulationFixture(Path(directory))
            failed = fixture.sibling_attempt(success=False, exit_code=17)
            _write_json(
                fixture.sibling_root / "attempts" / f"{CELL_ID}.json", failed
            )
            fixture.write_sibling_master(complete=False)
            with mock.patch.object(
                review.subprocess,
                "run",
                side_effect=lambda command, **_kwargs: _ffprobe_result(tuple(command)),
            ):
                audit = fixture.build()

            self.assertFalse(audit["review_complete"])
            cell = audit["cells"][0]
            self.assertEqual(cell["native_target_duplicate"]["status"], "valid")
            self.assertEqual(
                [cell["roles"][arm]["status"] for arm in review.ARM_ORDER],
                ["failed"] * 4,
            )
            self.assertEqual(
                cell["roles"]["target"]["details"]["process_exit_code"], 17
            )
            page = fixture.output.read_text(encoding="utf-8")
            self.assertIn("completed_failure", page)
            self.assertIn("partial.marker", page)
            # Only source is primary; the independent target remains a link/audit.
            self.assertEqual(page.count("<video "), 1)
            self.assertIn("native target MP4", page)

    def test_ffprobe_non_exact81_is_an_invalid_arm_and_cli_signals_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PopulationFixture(Path(directory))

            def fake_run(command, **_kwargs):
                return _ffprobe_result(tuple(command), bad_suffix="incomplete.mp4")

            with mock.patch.object(review.subprocess, "run", side_effect=fake_run):
                audit = fixture.build()
                common = [
                    "--native-registry",
                    str(fixture.native_registry),
                    "--native-root",
                    str(fixture.native_root),
                    "--sibling-registry",
                    str(fixture.sibling_registry),
                    "--sibling-root",
                    str(fixture.sibling_root),
                    "--source-root",
                    str(fixture.source_root),
                    "--output",
                    str(fixture.root / "review-cli.html"),
                ]
                self.assertEqual(review.main(common), 3)
                self.assertEqual(review.main([*common, "--allow-incomplete"]), 0)

            self.assertFalse(audit["review_complete"])
            self.assertEqual(
                audit["cells"][0]["roles"]["incomplete"]["status"], "invalid"
            )
            self.assertIn(
                "not exact81/25fps",
                audit["cells"][0]["roles"]["incomplete"]["message"],
            )

    def test_unexpected_attempt_receipt_prevents_complete_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = PopulationFixture(Path(directory))
            extra = fixture.native_root / "attempts" / "fit-deadbeef-s999.json"
            _write_json(extra, _seal({"unexpected": True}))
            with mock.patch.object(
                review.subprocess,
                "run",
                side_effect=lambda command, **_kwargs: _ffprobe_result(tuple(command)),
            ):
                audit = fixture.build()
            self.assertFalse(audit["review_complete"])
            self.assertEqual(
                audit["unexpected_attempt_receipts"]["native"], [extra.name]
            )


if __name__ == "__main__":
    unittest.main()
