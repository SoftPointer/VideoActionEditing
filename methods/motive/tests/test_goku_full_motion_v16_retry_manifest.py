from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive.goku_full_motion_qwen_v16 import (
    RECORD_SCHEMA,
    ROW_RECEIPT_SCHEMA,
    object_sha256,
)
from motive.goku_full_motion_v16_retry_manifest import (
    GokuFullMotionV16RetryManifestError,
    derive_error_retry_manifest,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _input_row(iid: str) -> dict:
    return {
        "iid": iid,
        "group_id": f"group-{iid}",
        "family": "people",
        "src_video": "source.mp4",
        "resolved_src_video": "/frozen/source.mp4",
        "source_caption": "A person moves one hand.",
        "edited_caption": "The person claps overhead.",
        "prompt": "Change the action to overhead clapping.",
        "anchor_image": "anchor.png",
        "resolved_anchor_image": "/frozen/anchor.png",
        "anchor_sha256": "1" * 64,
        "source_video_sha256": "2" * 64,
        "prefilter_score": 8.0,
        "media": {},
        "motion": {},
    }


def _write_terminal(qwen_root: Path, row: dict, *, status: str) -> None:
    iid = row["iid"]
    result_dir = qwen_root / "rows" / iid
    result_dir.mkdir(parents=True)
    (qwen_root / "terminal").mkdir(exist_ok=True)
    (qwen_root / "passed").mkdir(exist_ok=True)
    result = {
        "schema_version": RECORD_SCHEMA,
        "iid": iid,
        "status": status,
        "input_digest": object_sha256(row),
        "input_row": copy.deepcopy(row),
        "model": {
            "model_path": "/frozen/Qwen3-VL-32B-Instruct",
            "model_revision": "fixture",
            "transformers_version": "fixture",
        },
        "runtime": {
            "nframes": 16,
            "max_pixels": 2_359_296,
            "tile_width": 512,
            "mosaic_columns": 4,
        },
        "media_verification": None,
        "visual_input_digest": None,
        "source_stage": None,
        "target_stage": None,
        "source_census": None,
        "target_plan": None,
        "compiled_instruction": None,
        "error": (
            {"type": "FixtureSemanticError", "message": "strict reject"}
            if status == "error"
            else None
        ),
        "record_digest": None,
    }
    result["record_digest"] = object_sha256(result)
    result_path = result_dir / "result.json"
    result_bytes = (
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    result_path.write_bytes(result_bytes)

    passed_path = qwen_root / "passed" / f"{iid}.jsonl"
    passed_sha = None
    if status == "ok":
        passed_bytes = b"{}\n"
        passed_path.write_bytes(passed_bytes)
        passed_sha = _sha_bytes(passed_bytes)
    receipt = {
        "schema_version": ROW_RECEIPT_SCHEMA,
        "iid": iid,
        "status": status,
        "input_digest": object_sha256(row),
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha_bytes(result_bytes),
        "passed_path": str(passed_path.resolve()) if status == "ok" else None,
        "passed_sha256": passed_sha,
        "receipt_digest": None,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    (qwen_root / "terminal" / f"{iid}.receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


class GokuFullMotionV16RetryManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.input = self.root / "candidates.jsonl"
        self.qwen = self.root / "qwen-v16r2"
        self.qwen.mkdir()
        self.rows = [_input_row(f"iid-{index:03d}") for index in range(128)]
        self.lines = [
            (json.dumps(row, ensure_ascii=False, separators=(", ", ": ")) + "\n").encode()
            for row in self.rows
        ]
        self.input.write_bytes(b"".join(self.lines))
        self.errors = {1, 9, 127}
        for index, row in enumerate(self.rows):
            _write_terminal(
                self.qwen,
                row,
                status="error" if index in self.errors else "ok",
            )
        self.output = self.root / "retry.jsonl"
        self.receipt = self.root / "retry.receipt.json"

    def _derive(self, *, resume: bool = False):
        return derive_error_retry_manifest(
            input_manifest=self.input,
            input_sha256=_sha_bytes(self.input.read_bytes()),
            qwen_root=self.qwen,
            output_manifest=self.output,
            receipt_path=self.receipt,
            expected_rows=128,
            resume=resume,
        )

    def test_exact_original_error_lines_are_published_in_source_order(self) -> None:
        receipt = self._derive()
        expected = b"".join(self.lines[index] for index in sorted(self.errors))
        self.assertEqual(self.output.read_bytes(), expected)
        self.assertEqual(
            receipt["error_iids"], ["iid-001", "iid-009", "iid-127"]
        )
        self.assertEqual(receipt["terminal_count"], 128)
        self.assertEqual(receipt["error_count"], 3)
        self.assertEqual(receipt["retry_manifest_sha256"], _sha_bytes(expected))
        self.assertEqual(len(receipt["terminal_evidence"]), 128)

    def test_create_only_rejects_republication_but_resume_is_idempotent(self) -> None:
        first = self._derive()
        with self.assertRaisesRegex(
            GokuFullMotionV16RetryManifestError, "create-only target"
        ):
            self._derive()
        second = self._derive(resume=True)
        self.assertEqual(second, first)

    def test_resume_closes_manifest_only_interrupted_publication(self) -> None:
        expected = b"".join(self.lines[index] for index in sorted(self.errors))
        self.output.write_bytes(expected)
        receipt = self._derive(resume=True)
        self.assertEqual(receipt["error_count"], 3)
        self.assertTrue(self.receipt.is_file())

    def test_missing_terminal_fails_before_any_publication(self) -> None:
        (self.qwen / "terminal" / "iid-064.receipt.json").unlink()
        with self.assertRaises(Exception):
            self._derive()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.receipt.exists())

    def test_tampered_receipt_input_binding_is_rejected(self) -> None:
        path = self.qwen / "terminal" / "iid-009.receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["input_digest"] = "f" * 64
        receipt["receipt_digest"] = None
        receipt["receipt_digest"] = object_sha256(receipt)
        path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        with self.assertRaisesRegex(Exception, "terminal receipt identity differs"):
            self._derive()

    def test_source_manifest_digest_and_row_count_are_closed(self) -> None:
        with self.assertRaisesRegex(
            GokuFullMotionV16RetryManifestError, "SHA-256 differs"
        ):
            derive_error_retry_manifest(
                input_manifest=self.input,
                input_sha256="0" * 64,
                qwen_root=self.qwen,
                output_manifest=self.output,
                receipt_path=self.receipt,
                expected_rows=128,
            )
        with self.assertRaisesRegex(
            GokuFullMotionV16RetryManifestError, "row count differs"
        ):
            derive_error_retry_manifest(
                input_manifest=self.input,
                input_sha256=_sha_bytes(self.input.read_bytes()),
                qwen_root=self.qwen,
                output_manifest=self.output,
                receipt_path=self.receipt,
                expected_rows=127,
            )


if __name__ == "__main__":
    unittest.main()
