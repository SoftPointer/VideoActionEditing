from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motive import goku_atomic_candidate_epoch_slicer as slicer


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parent(
    path: Path,
    count: int,
    *,
    duplicate_iid: bool = False,
    duplicate_group: bool = False,
) -> tuple[bytes, str, list[bytes]]:
    lines: list[bytes] = []
    for index in range(count):
        iid_index = 0 if duplicate_iid and index == count - 1 else index
        group_index = 0 if duplicate_group and index == count - 1 else index
        # Deliberately retain non-canonical spacing to test byte preservation.
        line = (
            json.dumps(
                {
                    "iid": f"iid-{iid_index:03d}",
                    "group_id": f"group-{group_index:03d}",
                    "selection_rank": index + 1,
                    "payload": {"z": index, "a": "keep exact"},
                },
                sort_keys=False,
                separators=(", ", ": "),
            )
            + "\n"
        ).encode()
        lines.append(line)
    raw = b"".join(lines)
    path.write_bytes(raw)
    return raw, _sha(raw), lines


class CandidateEpochSlicerTest(unittest.TestCase):
    def test_contiguous_byte_exact_epochs_and_receipt_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "selected.jsonl"
            parent_raw, parent_sha, lines = _parent(parent, 17)
            output = root / "epochs"
            summary = slicer.slice_epochs(
                parent_selected=parent,
                expected_parent_sha256=parent_sha,
                output_dir=output,
                epoch_size=2,
                min_epochs=8,
            )
            self.assertEqual(summary["epoch_count"], 9)
            self.assertEqual(summary["parent_selected_sha256"], parent_sha)
            self.assertTrue(summary["iid_global_unique"])
            self.assertTrue(summary["group_id_global_unique"])
            reconstructed: list[bytes] = []
            for epoch_index in range(1, 10):
                epoch = output / f"epoch_{epoch_index:04d}"
                selected_raw = (epoch / slicer.SELECTED_NAME).read_bytes()
                reconstructed.append(selected_raw)
                start = (epoch_index - 1) * 2
                end = min(start + 2, len(lines))
                self.assertEqual(selected_raw, b"".join(lines[start:end]))
                output_sha = _sha(selected_raw)
                epoch_summary = json.loads(
                    (epoch / slicer.SUMMARY_NAME).read_text()
                )
                epoch_done = json.loads((epoch / slicer.DONE_NAME).read_text())
                for receipt in (epoch_summary, epoch_done):
                    self.assertEqual(receipt["parent_sha256"], parent_sha)
                    self.assertEqual(receipt["start"], start)
                    self.assertEqual(receipt["end"], end)
                    self.assertEqual(receipt["rows"], end - start)
                    self.assertEqual(receipt["output_sha256"], output_sha)
                    self.assertEqual(receipt["binding"]["start"], start)
                    self.assertEqual(receipt["binding"]["end"], end)
                self.assertEqual(
                    {path.name for path in epoch.iterdir()}, slicer.EPOCH_ENTRIES
                )
            self.assertEqual(b"".join(reconstructed), parent_raw)
            self.assertEqual(summary["concatenated_output_sha256"], parent_sha)
            self.assertEqual(summary["epochs"][-1]["rows"], 1)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    slicer.SUMMARY_NAME,
                    slicer.DONE_NAME,
                    *(f"epoch_{index:04d}" for index in range(1, 10)),
                },
            )

    def test_parent_sha_and_create_only_publication_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "selected.jsonl"
            unused_raw, parent_sha, unused_lines = _parent(parent, 4)
            with self.assertRaisesRegex(slicer.EpochSlicerError, "SHA differs"):
                slicer.slice_epochs(
                    parent_selected=parent,
                    expected_parent_sha256="0" * 64,
                    output_dir=root / "bad",
                    epoch_size=2,
                    min_epochs=1,
                )
            output = root / "epochs"
            slicer.slice_epochs(
                parent_selected=parent,
                expected_parent_sha256=parent_sha,
                output_dir=output,
                epoch_size=2,
                min_epochs=1,
            )
            frozen = (output / slicer.DONE_NAME).read_bytes()
            with self.assertRaisesRegex(FileExistsError, "create-only"):
                slicer.slice_epochs(
                    parent_selected=parent,
                    expected_parent_sha256=parent_sha,
                    output_dir=output,
                    epoch_size=2,
                    min_epochs=1,
                )
            self.assertEqual((output / slicer.DONE_NAME).read_bytes(), frozen)

    def test_global_iid_and_group_uniqueness_are_hard_gates(self) -> None:
        for field in ("iid", "group"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                parent = root / "selected.jsonl"
                unused_raw, parent_sha, unused_lines = _parent(
                    parent,
                    4,
                    duplicate_iid=field == "iid",
                    duplicate_group=field == "group",
                )
                with self.assertRaisesRegex(
                    slicer.EpochSlicerError,
                    "duplicate IID" if field == "iid" else "duplicate group_id",
                ):
                    slicer.slice_epochs(
                        parent_selected=parent,
                        expected_parent_sha256=parent_sha,
                        output_dir=root / "epochs",
                        epoch_size=2,
                        min_epochs=1,
                    )
                self.assertFalse((root / "epochs").exists())

    def test_minimum_epoch_count_is_enforced_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "selected.jsonl"
            unused_raw, parent_sha, unused_lines = _parent(parent, 14)
            output = root / "epochs"
            with self.assertRaisesRegex(slicer.EpochSlicerError, "fewer than required 8"):
                slicer.slice_epochs(
                    parent_selected=parent,
                    expected_parent_sha256=parent_sha,
                    output_dir=output,
                    epoch_size=2,
                    min_epochs=8,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
