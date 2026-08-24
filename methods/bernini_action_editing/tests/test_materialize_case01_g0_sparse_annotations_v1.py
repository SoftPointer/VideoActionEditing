from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


BERNINI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BERNINI_ROOT.parents[1]
SOURCE = BERNINI_ROOT / "materialize_case01_g0_sparse_annotations_v1.py"
SPEC = (
    BERNINI_ROOT
    / "assets/case01_288545b9c031491a_g0_sparse_annotations_v1.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("case01_sparse_g0", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Case01SparseG0ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def build_manifest(self):
        return self.module._build_manifest(REPO_ROOT, SPEC, SOURCE)

    def test_frozen_spec_and_exact_schedule_pass(self):
        manifest = self.build_manifest()
        self.assertEqual(manifest["schema_version"], self.module.SCHEMA_VERSION)
        self.assertEqual(tuple(manifest["frame_schedule"]), tuple(range(0, 81, 10)))
        self.assertEqual(len(manifest["frames"]), 9)
        self.assertEqual(
            manifest["manifest_digest"],
            self.module._canonical_digest(
                {key: value for key, value in manifest.items() if key != "manifest_digest"}
            ),
        )

    def test_every_frame_has_complete_conservative_labels(self):
        manifest = self.build_manifest()
        expected = {
            "dog#1",
            "dog#1.head",
            "dog#1.mouth",
            "bone#1",
            "bone#1.support",
            "safe-background#1",
        }
        for record in manifest["frames"]:
            labels = record["annotations"]
            self.assertEqual(set(labels), expected)
            self.assertEqual(labels["dog#1"]["geometry_type"], "binary_mask_reference")
            self.assertEqual(labels["bone#1"]["instruction_role"], "patient")
            self.assertEqual(labels["dog#1.head"]["geometry_type"], "reviewed_box")
            self.assertEqual(labels["dog#1.mouth"]["confidence"], "medium")
            self.assertIn("ambiguity", labels["dog#1.mouth"])
            self.assertEqual(
                labels["bone#1.support"]["geometry_type"],
                "derived_ground_context_box",
            )
            self.assertIn("not target RGB", labels["bone#1.support"]["ambiguity"])

    def test_safe_background_is_constant_and_mask_disjoint(self):
        manifest = self.build_manifest()
        boxes = {
            tuple(record["annotations"]["safe-background#1"]["bbox_xyxy"])
            for record in manifest["frames"]
        }
        self.assertEqual(boxes, {(560, 560, 660, 660)})
        self.assertTrue(
            manifest["validation"][
                "all_safe_background_boxes_disjoint_from_dog_and_bone_masks"
            ]
        )
        self.assertGreaterEqual(
            manifest["validation"][
                "minimum_head_box_dog_mask_coverage_fraction"
            ],
            0.70,
        )
        self.assertGreaterEqual(
            manifest["validation"][
                "minimum_mouth_box_dog_mask_coverage_fraction"
            ],
            0.70,
        )

    def test_claim_boundary_stays_fail_closed(self):
        manifest = self.build_manifest()
        self.assertEqual(
            manifest["validation"]["full_g0"],
            "PENDING_INDEPENDENT_SECOND_REVIEW",
        )
        self.assertEqual(manifest["validation"]["independent_second_review"], "PENDING")
        self.assertFalse(manifest["claim_limits"]["full_g0_authorized"])
        self.assertFalse(manifest["claim_limits"]["renderer_inference_performed"])
        self.assertFalse(manifest["claim_limits"]["training_performed"])
        self.assertEqual(manifest["claim_limits"]["optimizer_updates"], 0)

    def test_materialize_verify_and_tamper_rejection(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not installed")
        with tempfile.TemporaryDirectory(prefix="case01-sparse-g0-test-") as temporary:
            output_root = Path(temporary) / "fresh-output"
            receipt = self.module.materialize(
                REPO_ROOT, SPEC, output_root, "ffmpeg", materializer_path=SOURCE
            )
            self.assertEqual(
                receipt["status"], "COMPLETE_PRIMARY_SPARSE_ANNOTATION_HALF"
            )
            self.assertEqual(
                self.module._png_dimensions(
                    output_root / self.module.OVERLAY_SHEET_NAME
                ),
                (1056, 1104),
            )
            verified = self.module.verify_output(
                REPO_ROOT, SPEC, output_root, materializer_path=SOURCE
            )
            self.assertEqual(verified["receipt_digest"], receipt["receipt_digest"])

            manifest_path = output_root / "manifest.json"
            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered["validation"]["full_g0"] = "PASS"
            manifest_path.write_text(
                json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(self.module.SparseAnnotationError):
                self.module.verify_output(
                    REPO_ROOT, SPEC, output_root, materializer_path=SOURCE
                )


if __name__ == "__main__":
    unittest.main()
