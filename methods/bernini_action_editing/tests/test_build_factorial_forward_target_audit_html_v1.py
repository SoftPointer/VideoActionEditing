from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_factorial_forward_target_audit_html_v1.py"
)
SPEC = importlib.util.spec_from_file_location("build_factorial_forward_target_audit_html_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_id = "source0000000001"
    media_root = tmp_path / "media"
    source_root = media_root / source_id
    source_root.mkdir(parents=True)
    (source_root / "comparison_source_target.mp4").write_bytes(b"video")
    (source_root / "review_source_target_f0_20_40_60_80.jpg").write_bytes(b"image")
    (source_root / "raw_candidate.json").write_text(
        json.dumps({"iid": source_id, "prompt": "Make the dog sit down."}),
        encoding="utf-8",
    )
    review = {
        "schema_version": MODULE.SCHEMA,
        "authority": {
            "factorial_negatives_present": False,
            "same_seed_pairing_verified": False,
            "training_target_authorized": False,
            "optimizer_step_authorized": False,
            "method_success_claimed": False,
        },
        "summary": {
            "reviewed": 1,
            "strict_eligible": 1,
            "compound_instruction": 0,
            "action_failure": 0,
            "wrong_target_family": 0,
        },
        "rows": [
            {
                "source_id": source_id,
                "action_family": "dog-stand-to-sit",
                "split": "fit",
                "status": "strict_eligible",
                "raw_prompt": "Make the dog sit down.",
                "review_note": "The intended action completes without a second event.",
            }
        ],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return review_path, media_root, tmp_path / "index.html"


class ForwardTargetAuditHTMLTests(unittest.TestCase):
    def test_builds_bound_create_only_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review_path, media_root, output = _fixture(Path(directory))

            result = MODULE.build(
                review_path=review_path, media_root=media_root, output=output
            )

            page = output.read_text(encoding="utf-8")
            self.assertEqual(result["row_count"], 1)
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertIn("source on the left", page)
            self.assertIn("source0000000001/comparison_source_target.mp4", page)
            self.assertIn("Authority remains closed", page)

    def test_rejects_training_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review_path, media_root, output = _fixture(Path(directory))
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["authority"]["training_target_authorized"] = True
            review_path.write_text(json.dumps(review), encoding="utf-8")

            with self.assertRaisesRegex(
                MODULE.ForwardTargetAuditHTMLError, "fail-closed"
            ):
                MODULE.build(
                    review_path=review_path, media_root=media_root, output=output
                )

    def test_rejects_raw_prompt_binding_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review_path, media_root, output = _fixture(Path(directory))
            metadata = media_root / "source0000000001" / "raw_candidate.json"
            metadata.write_text(
                json.dumps(
                    {"iid": "source0000000001", "prompt": "A different action."}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                MODULE.ForwardTargetAuditHTMLError, "raw prompt binding"
            ):
                MODULE.build(
                    review_path=review_path, media_root=media_root, output=output
                )

    def test_rejects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review_path, media_root, output = _fixture(Path(directory))
            output.write_text("pre-existing", encoding="utf-8")

            with self.assertRaisesRegex(
                MODULE.ForwardTargetAuditHTMLError, "fresh absolute path"
            ):
                MODULE.build(
                    review_path=review_path, media_root=media_root, output=output
                )


if __name__ == "__main__":
    unittest.main()
