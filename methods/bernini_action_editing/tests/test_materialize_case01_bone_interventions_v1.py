from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "materialize_case01_bone_interventions_v1.py"
SPEC = importlib.util.spec_from_file_location("case01_bone_interventions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrameDiffMetricsTest(unittest.TestCase):
    def test_single_outside_support_pixel_tamper_is_rejected(self) -> None:
        source = bytes(MODULE.RGB_FRAME_BYTES)
        tampered = bytearray(source)
        outside_pixel = MODULE.FRAME_PIXELS - 1
        tampered[outside_pixel * 3] = 1
        with self.assertRaisesRegex(
            RuntimeError, "pixels changed outside declared support"
        ):
            MODULE.frame_diff_metrics(source, bytes(tampered), {0})

    def test_inside_support_pixel_is_counted(self) -> None:
        source = bytes(MODULE.RGB_FRAME_BYTES)
        changed = bytearray(source)
        inside_pixel = 123
        changed[inside_pixel * 3 : inside_pixel * 3 + 3] = bytes((1, 2, 3))
        metrics = MODULE.frame_diff_metrics(source, bytes(changed), {inside_pixel})
        self.assertEqual(metrics["changed_pixels"], 1)
        self.assertEqual(metrics["outside_expected_tube_changed_pixels"], 0)
        self.assertEqual(metrics["rgb_l1_sum_inside_tube"], 6)


class TranslatedMatchedRelationsTest(unittest.TestCase):
    def build_fixture(self) -> dict:
        source = bytearray(MODULE.RGB_FRAME_BYTES)
        source_bone_index = 300 * MODULE.WIDTH + 500
        source[source_bone_index * 3 : source_bone_index * 3 + 3] = bytes(
            (101, 102, 103)
        )
        source = bytes(source)
        removal_support = {source_bone_index, source_bone_index + 1}
        target_bone_index = (300 + MODULE.TRANSLATE_DY) * MODULE.WIDTH + 500
        target_nonbone_index = target_bone_index + 1
        sham_support = {target_bone_index, target_nonbone_index}

        removed = bytearray(source)
        for index in removal_support:
            removed[index * 3 : index * 3 + 3] = bytes((10, 11, 12))
        removed = bytes(removed)
        sham = bytearray(source)
        for index in sham_support:
            sham[index * 3 : index * 3 + 3] = bytes((20, 21, 22))
        sham = bytes(sham)
        matched_background = MODULE.copy_support_pixels(
            removed, sham, sham_support
        )
        translated, translated_bone = MODULE.translate_bone_pixels(
            source,
            matched_background,
            {source_bone_index},
            MODULE.TRANSLATE_DY,
        )
        return {
            "source": source,
            "removed": removed,
            "translated": translated,
            "sham": sham,
            "source_bone": {source_bone_index},
            "removal_support": removal_support,
            "sham_support": sham_support,
            "translated_bone": translated_bone,
            "target_nonbone_index": target_nonbone_index,
        }

    def assert_fixture(self, fixture: dict) -> dict:
        return MODULE.assert_translated_matched_relations(
            source=fixture["source"],
            removed=fixture["removed"],
            translated=fixture["translated"],
            sham=fixture["sham"],
            source_bone=fixture["source_bone"],
            removal_support=fixture["removal_support"],
            sham_support=fixture["sham_support"],
            translated_bone=fixture["translated_bone"],
            dy=MODULE.TRANSLATE_DY,
        )

    def test_matched_relations_pass(self) -> None:
        result = self.assert_fixture(self.build_fixture())
        self.assertTrue(result["original_support_equals_removed"])
        self.assertTrue(result["target_nonbone_support_equals_sham"])
        self.assertTrue(result["target_bone_equals_shifted_source"])

    def test_target_nonbone_tamper_is_rejected(self) -> None:
        fixture = self.build_fixture()
        translated = bytearray(fixture["translated"])
        offset = fixture["target_nonbone_index"] * 3
        translated[offset] ^= 1
        fixture["translated"] = bytes(translated)
        with self.assertRaisesRegex(RuntimeError, "target non-bone byte mismatch"):
            self.assert_fixture(fixture)

    def test_original_support_tamper_is_rejected(self) -> None:
        fixture = self.build_fixture()
        translated = bytearray(fixture["translated"])
        index = next(iter(fixture["removal_support"]))
        translated[index * 3] ^= 1
        fixture["translated"] = bytes(translated)
        with self.assertRaisesRegex(RuntimeError, "original-support byte mismatch"):
            self.assert_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
