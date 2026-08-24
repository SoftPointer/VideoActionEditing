from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive.goku_full_motion_prepare import (
    CANDIDATES_NAME,
    DONE_NAME,
    PREFILTER_SCHEMA,
    SUMMARY_NAME,
    FullMotionPrepareError,
    _object_digest,
    prepare_candidates,
)


def _canonical_line(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _row(index: int, *, short_side: int = 704) -> dict[str, object]:
    iid = f"iid{index:03d}"
    anchor = f"anchors/{iid}.png"
    anchor_raw = f"anchor-{iid}".encode()
    return {
        "schema_version": PREFILTER_SCHEMA,
        "iid": iid,
        "group_id": f"group-{index:03d}",
        "family": "wave",
        "src_video": f"videos/{iid}/source.mp4",
        "resolved_src_video": f"/data/videos/{iid}/source.mp4",
        "source_caption": "source",
        "edited_caption": "edited",
        "prompt": "legacy seed only",
        "anchor_image": anchor,
        "resolved_anchor_image": f"/run/prefilter/{anchor}",
        "anchor_sha256": _sha(anchor_raw),
        "source_video_sha256": _sha(f"source-{iid}".encode()),
        "prefilter_score": 0.9,
        "media": {
            "frame_count": 81,
            "fps": 25.0,
            "duration_seconds": 3.2,
            "short_side": short_side,
        },
        "motion": {"scene_cut_ratio": 0.0},
        "eligible": True,
        "selected": True,
        "selection_rank": index + 1,
        "_anchor_raw": anchor_raw,
    }


def _make_prefilter(root: Path, rows: list[dict[str, object]]) -> Path:
    prefilter = root / "prefilter"
    anchors = prefilter / "anchors"
    anchors.mkdir(parents=True)
    clean_rows: list[dict[str, object]] = []
    anchor_map: dict[str, str] = {}
    for row in rows:
        clean = dict(row)
        raw = clean.pop("_anchor_raw")
        relative = str(clean["anchor_image"])
        (prefilter / relative).write_bytes(raw)  # type: ignore[arg-type]
        anchor_map[relative] = _sha(raw)  # type: ignore[arg-type]
        clean_rows.append(clean)
    selected_raw = b"".join(_canonical_line(row) for row in clean_rows)
    evaluated_raw = selected_raw
    (prefilter / "selected.jsonl").write_bytes(selected_raw)
    (prefilter / "evaluated.jsonl").write_bytes(evaluated_raw)
    summary = {
        "schema_version": PREFILTER_SCHEMA,
        "status": "complete",
        "config": {"sample_size": len(clean_rows)},
        "counts": {"selected": len(clean_rows)},
    }
    summary_raw = _pretty(summary)
    (prefilter / "summary.json").write_bytes(summary_raw)
    artifacts = {
        "selected.jsonl": _sha(selected_raw),
        "evaluated.jsonl": _sha(evaluated_raw),
        "summary.json": _sha(summary_raw),
        "anchors": _object_digest(anchor_map),
    }
    done = {
        "schema_version": PREFILTER_SCHEMA,
        "status": "complete",
        "selected_rows": len(clean_rows),
        "artifacts": artifacts,
        "anchor_sha256": anchor_map,
    }
    (prefilter / "done.json").write_bytes(_pretty(done))
    return prefilter


class FullMotionPrepareTest(unittest.TestCase):
    def test_preserves_required_row_bytes_and_orders_it_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefilter = _make_prefilter(root, [_row(i) for i in range(5)])
            output = root / "prepared"
            summary = prepare_candidates(
                prefilter_dir=prefilter,
                output_dir=output,
                sample_size=3,
                required_iids=["iid004"],
            )
            rows = [
                json.loads(line)
                for line in (output / CANDIDATES_NAME).read_text().splitlines()
            ]
            self.assertEqual([row["iid"] for row in rows], ["iid004", "iid000", "iid001"])
            self.assertEqual(summary["semantics"]["legacy_qwen_decisions_reused"], False)
            self.assertTrue((output / SUMMARY_NAME).is_file())
            self.assertTrue((output / DONE_NAME).is_file())

    def test_filters_low_resolution_before_counting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefilter = _make_prefilter(
                root,
                [_row(0, short_side=480), _row(1), _row(2)],
            )
            output = root / "prepared"
            prepare_candidates(
                prefilter_dir=prefilter,
                output_dir=output,
                sample_size=2,
            )
            rows = [json.loads(line) for line in (output / CANDIDATES_NAME).read_text().splitlines()]
            self.assertEqual([row["iid"] for row in rows], ["iid001", "iid002"])

    def test_missing_required_iid_fails_closed_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefilter = _make_prefilter(root, [_row(0)])
            output = root / "prepared"
            with self.assertRaises(FullMotionPrepareError):
                prepare_candidates(
                    prefilter_dir=prefilter,
                    output_dir=output,
                    sample_size=1,
                    required_iids=["absent"],
                )
            self.assertFalse(output.exists())

    def test_tampered_anchor_fails_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefilter = _make_prefilter(root, [_row(0)])
            (prefilter / "anchors/iid000.png").write_bytes(b"tampered")
            with self.assertRaises(FullMotionPrepareError):
                prepare_candidates(
                    prefilter_dir=prefilter,
                    output_dir=root / "prepared",
                    sample_size=1,
                )

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefilter = _make_prefilter(root, [_row(0)])
            output = root / "prepared"
            prepare_candidates(
                prefilter_dir=prefilter,
                output_dir=output,
                sample_size=1,
            )
            with self.assertRaises(FileExistsError):
                prepare_candidates(
                    prefilter_dir=prefilter,
                    output_dir=output,
                    sample_size=1,
                )


if __name__ == "__main__":
    unittest.main()
