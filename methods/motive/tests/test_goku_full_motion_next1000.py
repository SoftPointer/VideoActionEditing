from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive.goku_full_motion_next1000 import (
    CANDIDATES_NAME,
    DONE_NAME,
    EXPECTED_EXCLUDED_ROWS,
    EXPECTED_PARENT_ROWS,
    OUTPUT_ROWS,
    SUMMARY_NAME,
    Next1000MaterializeError,
    materialize_next1000_candidates,
)
from motive.goku_full_motion_prepare import PREFILTER_SCHEMA, _object_digest


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _parent_line(value: object, index: int) -> bytes:
    # Deliberately use two non-canonical encodings.  The output must preserve
    # these exact parent bytes rather than serializing the parsed objects.
    separators = (", ", ": ") if index % 2 else (",", ":")
    return (
        json.dumps(value, ensure_ascii=False, separators=separators) + "\n"
    ).encode("utf-8")


class _Fixture:
    def __init__(
        self,
        root: Path,
        *,
        geometry_overrides: dict[int, dict[str, object]] | None = None,
        group_overrides: dict[int, str] | None = None,
        rank_overrides: dict[int, int] | None = None,
    ) -> None:
        geometry_overrides = geometry_overrides or {}
        group_overrides = group_overrides or {}
        rank_overrides = rank_overrides or {}
        self.prefilter = root / "prefilter"
        self.sources = root / "sources"
        anchors = self.prefilter / "anchors"
        anchors.mkdir(parents=True)
        self.sources.mkdir()
        self.rows: list[dict[str, object]] = []
        self.lines: list[bytes] = []
        anchor_map: dict[str, str] = {}
        for index in range(EXPECTED_PARENT_ROWS):
            iid = f"iid-{index:04d}"
            source = self.sources / f"{iid}.mp4"
            source_raw = f"source-video-bytes-{iid}".encode("ascii")
            source.write_bytes(source_raw)
            source_stat = source.stat()
            anchor_relative = f"anchors/{iid}.png"
            anchor = self.prefilter / anchor_relative
            anchor_raw = f"lossless-anchor-bytes-{iid}".encode("ascii")
            anchor.write_bytes(anchor_raw)
            anchor_map[anchor_relative] = _sha(anchor_raw)
            geometry: dict[str, object] = {
                "frame_count": 81,
                "fps": 25.0,
                "duration_seconds": 3.2,
                "width": 704,
                "height": 1_280,
                "short_side": 704,
                "file_size_bytes": source_stat.st_size,
                "mtime_ns_at_analysis": source_stat.st_mtime_ns,
            }
            geometry.update(geometry_overrides.get(index, {}))
            row: dict[str, object] = {
                "schema_version": PREFILTER_SCHEMA,
                "iid": iid,
                "group_id": group_overrides.get(index, f"group-{index:04d}"),
                "family": "people",
                "src_video": f"videos/{iid}/source.mp4",
                "resolved_src_video": str(source.resolve()),
                "source_caption": "A person performs an action.",
                "edited_caption": "The person performs another action.",
                "prompt": "Legacy prompt is not authoritative.",
                "anchor_image": anchor_relative,
                "resolved_anchor_image": str(anchor.resolve()),
                "anchor_sha256": _sha(anchor_raw),
                "source_video_sha256": _sha(source_raw),
                "prefilter_score": 1.0 - index / 10_000,
                "media": geometry,
                "motion": {
                    "label": "dynamic_object",
                    "scene_cut_ratio": 0.0,
                },
                "eligible": True,
                "selected": True,
                "selection_rank": rank_overrides.get(index, index + 1),
            }
            self.rows.append(row)
            self.lines.append(_parent_line(row, index))
        selected_raw = b"".join(self.lines)
        evaluated_raw = selected_raw
        (self.prefilter / "selected.jsonl").write_bytes(selected_raw)
        (self.prefilter / "evaluated.jsonl").write_bytes(evaluated_raw)
        summary = {
            "schema_version": PREFILTER_SCHEMA,
            "status": "complete",
            "config": {"sample_size": EXPECTED_PARENT_ROWS},
            "counts": {"selected": EXPECTED_PARENT_ROWS},
        }
        summary_raw = _pretty(summary)
        (self.prefilter / "summary.json").write_bytes(summary_raw)
        artifacts = {
            "selected.jsonl": _sha(selected_raw),
            "evaluated.jsonl": _sha(evaluated_raw),
            "summary.json": _sha(summary_raw),
            "anchors": _object_digest(anchor_map),
        }
        done = {
            "schema_version": PREFILTER_SCHEMA,
            "status": "complete",
            "selected_rows": EXPECTED_PARENT_ROWS,
            "artifacts": artifacts,
            "anchor_sha256": anchor_map,
        }
        (self.prefilter / "done.json").write_bytes(_pretty(done))
        self.parent_sha256 = _sha(selected_raw)
        self.exclude = root / "exact128.jsonl"
        self.exclude_indices = list(range(EXPECTED_EXCLUDED_ROWS))
        self.exclude.write_bytes(
            b"".join(self.lines[index] for index in self.exclude_indices)
        )
        self.exclude_sha256 = _sha(self.exclude.read_bytes())

    def run(self, output: Path, *, hash_workers: int = 4):
        return materialize_next1000_candidates(
            prefilter_dir=self.prefilter,
            exclude_manifest=self.exclude,
            output_dir=output,
            expected_parent_selected_sha256=self.parent_sha256,
            expected_exclude_manifest_sha256=self.exclude_sha256,
            hash_workers=hash_workers,
        )


class GokuFullMotionNext1000Tests(unittest.TestCase):
    def test_emits_exact_ranked_parent_lines_and_closed_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            output = root / "next1000"
            summary = fixture.run(output)

            expected = b"".join(
                fixture.lines[
                    EXPECTED_EXCLUDED_ROWS : EXPECTED_EXCLUDED_ROWS + OUTPUT_ROWS
                ]
            )
            self.assertEqual((output / CANDIDATES_NAME).read_bytes(), expected)
            self.assertEqual(summary["output"]["rows"], OUTPUT_ROWS)
            self.assertEqual(summary["output"]["sha256"], _sha(expected))
            self.assertEqual(
                summary["selection"]["candidate_first_parent_rank"], 129
            )
            self.assertEqual(
                summary["selection"]["candidate_last_parent_rank"], 1_128
            )
            self.assertTrue(summary["selection"]["parent_row_bytes_preserved"])
            self.assertEqual(
                summary["inputs"]["parent_selected_sha256"],
                fixture.parent_sha256,
            )
            self.assertEqual(
                summary["inputs"]["exclude_manifest_sha256"],
                fixture.exclude_sha256,
            )

            summary_raw = (output / SUMMARY_NAME).read_bytes()
            done = json.loads((output / DONE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(
                done["output_sha256"],
                {
                    CANDIDATES_NAME: _sha(expected),
                    SUMMARY_NAME: _sha(summary_raw),
                },
            )
            self.assertEqual(done["counts"]["remaining"], 1_107)
            self.assertEqual(done["counts"]["tail"], 107)
            self.assertFalse(done["training_eligible"])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {CANDIDATES_NAME, SUMMARY_NAME, DONE_NAME},
            )

    def test_create_only_rejects_existing_output_before_rehashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            output = root / "next1000"
            fixture.run(output, hash_workers=1)
            before = {
                path.name: path.read_bytes() for path in output.iterdir()
            }
            # If the second call got past the create-only guard, this tamper
            # would make media validation fail instead.
            Path(fixture.rows[128]["resolved_src_video"]).write_bytes(b"tampered")
            with self.assertRaises(FileExistsError):
                fixture.run(output, hash_workers=1)
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()},
                before,
            )

    def test_wrong_parent_or_exclusion_digest_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            output = root / "next1000"
            with self.assertRaisesRegex(
                Next1000MaterializeError, "parent selected.jsonl SHA-256 differs"
            ):
                materialize_next1000_candidates(
                    prefilter_dir=fixture.prefilter,
                    exclude_manifest=fixture.exclude,
                    output_dir=output,
                    expected_parent_selected_sha256="0" * 64,
                    expected_exclude_manifest_sha256=fixture.exclude_sha256,
                    hash_workers=1,
                )
            self.assertFalse(output.exists())

            with self.assertRaisesRegex(
                Next1000MaterializeError, "exclusion manifest SHA-256 differs"
            ):
                materialize_next1000_candidates(
                    prefilter_dir=fixture.prefilter,
                    exclude_manifest=fixture.exclude,
                    output_dir=output,
                    expected_parent_selected_sha256=fixture.parent_sha256,
                    expected_exclude_manifest_sha256="f" * 64,
                    hash_workers=1,
                )
            self.assertFalse(output.exists())

    def test_exclusion_row_must_be_verbatim_parent_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            changed = dict(fixture.rows[0])
            changed["prompt"] = "A changed row with the same IID."
            fixture.exclude.write_bytes(
                _parent_line(changed, 0)
                + b"".join(fixture.lines[1:EXPECTED_EXCLUDED_ROWS])
            )
            fixture.exclude_sha256 = _sha(fixture.exclude.read_bytes())
            with self.assertRaisesRegex(
                Next1000MaterializeError, "not byte-identical"
            ):
                fixture.run(root / "next1000", hash_workers=1)
            self.assertFalse((root / "next1000").exists())

    def test_invalid_fixed_geometry_fails_before_publication(self) -> None:
        cases = {
            "frame_count": {"frame_count": 80},
            "fps": {"fps": 24.0},
            "short_side": {"short_side": 640},
        }
        for label, override in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = _Fixture(
                    root,
                    geometry_overrides={EXPECTED_EXCLUDED_ROWS: override},
                )
                with self.assertRaises(Next1000MaterializeError):
                    fixture.run(root / "next1000", hash_workers=1)
                self.assertFalse((root / "next1000").exists())

    def test_tampered_candidate_source_fails_actual_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            source = Path(
                str(fixture.rows[EXPECTED_EXCLUDED_ROWS]["resolved_src_video"])
            )
            source.write_bytes(b"tampered-after-prefilter")
            with self.assertRaisesRegex(
                Next1000MaterializeError, "source video SHA-256 differs"
            ):
                fixture.run(root / "next1000", hash_workers=1)
            self.assertFalse((root / "next1000").exists())

    def test_parent_rank_and_group_contracts_fail_closed(self) -> None:
        cases = {
            "rank": {
                "rank_overrides": {EXPECTED_PARENT_ROWS - 1: 1},
            },
            "group": {
                "group_overrides": {
                    EXPECTED_PARENT_ROWS - 1: "group-0000",
                },
            },
        }
        for label, kwargs in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = _Fixture(root, **kwargs)
                with self.assertRaises(Next1000MaterializeError):
                    fixture.run(root / "next1000", hash_workers=1)
                self.assertFalse((root / "next1000").exists())


if __name__ == "__main__":
    unittest.main()
