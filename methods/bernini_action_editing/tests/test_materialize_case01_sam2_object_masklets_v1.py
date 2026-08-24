from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "materialize_case01_sam2_object_masklets_v1.py"
SPEC = ROOT / "assets/case01_288545b9c031491a_sam2_boxes_v1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("case01_masklets", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Case01OracleMaskletContractTest(unittest.TestCase):
    def test_frozen_spec_passes(self):
        module = load_module()
        value = json.loads(SPEC.read_text(encoding="utf-8"))
        module.validate_spec(value)

    def test_wrong_patient_box_is_rejected(self):
        module = load_module()
        value = json.loads(SPEC.read_text(encoding="utf-8"))
        value["frame0_objects"][1]["box_xyxy"] = [700.0, 10.0, 705.0, 20.0]
        with self.assertRaises(module.OracleMaskletError):
            module.validate_spec(value)

    def test_claim_expansion_is_rejected(self):
        module = load_module()
        value = json.loads(SPEC.read_text(encoding="utf-8"))
        value["claim_limits"]["training_performed"] = True
        with self.assertRaises(module.OracleMaskletError):
            module.validate_spec(value)

    def test_geometry_and_iou(self):
        module = load_module()
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed in the local control environment")
        left = np.zeros((5, 5), dtype=bool)
        right = np.zeros((5, 5), dtype=bool)
        left[1:3, 1:3] = True
        right[2:4, 2:4] = True
        self.assertEqual(module.geometry(left)["area"], 4)
        self.assertAlmostEqual(module.mask_iou(left, right), 1.0 / 7.0)


if __name__ == "__main__":
    unittest.main()
