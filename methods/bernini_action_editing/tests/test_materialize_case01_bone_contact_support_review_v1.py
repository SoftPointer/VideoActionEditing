from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
MODULE_PATH = (
    METHOD_ROOT / "tools/materialize_case01_bone_contact_support_review_v1.py"
)
SPEC = importlib.util.spec_from_file_location("case01_support_review_packet", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MorphologyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"OpenCV/NumPy unavailable: {error}")

    def test_exact_halo8_and_directional_contact_apron(self) -> None:
        import numpy as np

        bone = np.zeros((MODULE.HEIGHT, MODULE.WIDTH), dtype=bool)
        dog = np.zeros_like(bone)
        bone[200, 500] = True
        dog[500, 100] = True
        parts = MODULE.build_support_components(bone, dog)

        inclusive = parts["halo8_inclusive"]
        self.assertEqual(int(inclusive.sum()), 17 * 17)
        self.assertTrue(inclusive[192:209, 492:509].all())
        self.assertTrue(parts["candidate_support"][216, 488])
        self.assertTrue(parts["candidate_support"][204, 512])
        self.assertFalse(parts["candidate_support"][191, 500])
        self.assertFalse(parts["candidate_support"][217, 500])
        self.assertEqual(
            int(np.logical_and(parts["candidate_support"], parts["dog_guard12"]).sum()),
            0,
        )

    def test_dog_guard_collision_fails_closed(self) -> None:
        import numpy as np

        bone = np.zeros((MODULE.HEIGHT, MODULE.WIDTH), dtype=bool)
        dog = np.zeros_like(bone)
        bone[200, 500] = True
        dog[200, 524] = True
        with self.assertRaisesRegex(MODULE.SupportPacketError, "protected dog guard"):
            MODULE.build_support_components(bone, dog)

    def test_radius_must_be_builtin_nonnegative_int(self) -> None:
        import numpy as np

        mask = np.zeros((MODULE.HEIGHT, MODULE.WIDTH), dtype=bool)
        with self.assertRaisesRegex(MODULE.SupportPacketError, "builtin int"):
            MODULE.dilate_square(mask, True)
        with self.assertRaisesRegex(MODULE.SupportPacketError, "builtin int"):
            MODULE.dilate_square(mask, -1)


class ReviewClaimBoundaryTest(unittest.TestCase):
    def test_both_templates_are_entirely_pending(self) -> None:
        for slot in (1, 2):
            value = MODULE.review_template(slot)
            self.assertEqual(value["reviewer_slot"], slot)
            self.assertEqual(value["overall_decision"], "PENDING")
            self.assertFalse(value["all_81_native_frames_reviewed"])
            self.assertFalse(value["claim_limits_acknowledged"])
            self.assertEqual(len(value["frames"]), MODULE.FRAME_COUNT)
            for row in value["frames"]:
                self.assertEqual(row["decision"], "PENDING")
                self.assertEqual(row["contact_shadow_coverage"], "PENDING")

    def test_html_never_represents_a_review_pass(self) -> None:
        rows = [
            {
                "frame_index": index,
                "areas": {"candidate_support": 123},
                "crop_bbox_xyxy": [1, 2, 3, 4],
            }
            for index in range(MODULE.FRAME_COUNT)
        ]
        rendered = MODULE.build_html(rows)
        self.assertEqual(rendered.count('class="card"'), MODULE.FRAME_COUNT)
        self.assertIn("UNSIGNED — HOLD", rendered)
        self.assertIn("remains PENDING until two independent external reviewers", rendered)
        self.assertNotIn("support PASS granted", rendered)

    def test_canonical_json_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.canonical_json_bytes({"bad": float("nan")})


class FrozenAuthorityTest(unittest.TestCase):
    def authority_kwargs(self) -> dict:
        return {
            "source_video": REPO_ROOT
            / "artifacts/current_full644_r64_heldout8_20260820/assets/media/case01-source.mp4",
            "masklet_root": REPO_ROOT
            / "artifacts/object_grounded_case01_0821_sam2_masklets_r2",
            "sparse_spec_path": REPO_ROOT
            / "methods/bernini_action_editing/assets/case01_288545b9c031491a_g0_sparse_annotations_v1.json",
            "sparse_manifest_path": REPO_ROOT
            / "artifacts/object_grounded_case01_0821_sparse_g0_v1/manifest.json",
            "sparse_receipt_path": REPO_ROOT
            / "artifacts/object_grounded_case01_0821_sparse_g0_v1/receipt.json",
            "revoked_r4_root": REPO_ROOT
            / "artifacts/object_grounded_case01_0821_bone_interventions_r4",
        }

    def test_all_frozen_inputs_validate_and_r4_stays_negative_only(self) -> None:
        value = MODULE.validate_authorities(**self.authority_kwargs())
        self.assertEqual(
            len(
                [
                    key
                    for key in value["output_by_path"]
                    if key.startswith("masks/bone/")
                ]
            ),
            MODULE.FRAME_COUNT,
        )
        self.assertFalse(
            value["sparse_receipt"]["claim_limits"]["full_g0_authorized"]
        )
        self.assertEqual(
            value["sparse_receipt"]["validation"]["independent_second_review"],
            "PENDING",
        )

    def test_one_byte_authority_tamper_is_rejected(self) -> None:
        original = self.authority_kwargs()["sparse_receipt_path"]
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "receipt.json"
            payload = bytearray(original.read_bytes())
            payload[-2] ^= 1
            tampered.write_bytes(payload)
            kwargs = self.authority_kwargs()
            kwargs["sparse_receipt_path"] = tampered
            with self.assertRaisesRegex(MODULE.SupportPacketError, "SHA-256 differs"):
                MODULE.validate_authorities(**kwargs)


class MaterializedPacketTest(unittest.TestCase):
    PACKET = REPO_ROOT / "artifacts/case01_bone_contact_support_review_v1_unsigned_20260822"

    def test_packet_if_materialized_is_unsigned_and_exact_tree_bound(self) -> None:
        if not self.PACKET.is_dir():
            self.skipTest("local review packet has not been materialized")
        manifest_path = self.PACKET / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], MODULE.SCHEMA)
        self.assertEqual(
            manifest["status"],
            "UNSIGNED_CANDIDATE_HOLD_PENDING_TWO_EXTERNAL_REVIEWS",
        )
        self.assertFalse(manifest["candidate_is_review_passed"])
        self.assertFalse(manifest["claim_limits"]["support_pass_authorized"])
        self.assertFalse(manifest["claim_limits"]["renderer_or_vace_run_authorized"])
        self.assertEqual(manifest["review_gate"]["current_external_receipts"], 0)
        self.assertEqual(len(manifest["frames"]), MODULE.FRAME_COUNT)
        self.assertTrue(
            all(
                row["checks"]["contact_shadow_visual_coverage"]
                == "PENDING_TWO_EXTERNAL_REVIEWS"
                for row in manifest["frames"]
            )
        )
        sums: dict[str, str] = {}
        for line in (self.PACKET / "SHA256SUMS").read_text(encoding="ascii").splitlines():
            digest, relative = line.split("  ", 1)
            sums[relative] = digest
        self.assertEqual(sums["manifest.json"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        actual_files = {
            path.relative_to(self.PACKET).as_posix()
            for path in self.PACKET.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, set(sums) | {"SHA256SUMS"})


if __name__ == "__main__":
    unittest.main()
