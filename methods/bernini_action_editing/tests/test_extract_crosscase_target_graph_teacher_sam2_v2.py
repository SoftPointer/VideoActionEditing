import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "extract_crosscase_target_graph_teacher_sam2_v2.py"
ASSETS = ROOT / "assets"


def load_program():
    # The local lightweight test environment does not ship NumPy.  Validation
    # and phase-state tests do not execute numerical observer code.
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    spec = importlib.util.spec_from_file_location("graph_teacher_v2", PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GraphTeacherV2HostileContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_program()
        cls.spec_407 = json.loads(
            (ASSETS / "40712e1341dc_target_graph_teacher_sam2_spec_v2.json")
            .read_text(encoding="utf-8")
        )

    def test_sealed_spec_is_valid(self):
        for path in sorted(ASSETS.glob("*_target_graph_teacher_sam2_spec_v2.json")):
            with self.subTest(path=path.name):
                self.module._validate_spec(json.loads(path.read_text(encoding="utf-8")))

    def test_tracked_node_with_both_prompts_missing_is_rejected(self):
        value = copy.deepcopy(self.spec_407)
        phone = next(node for node in value["nodes"] if node["node_id"] == "phone")
        phone["source_prompt"] = None
        phone["real_target_teacher_prompt"] = None
        with self.assertRaisesRegex(Exception, "tracked mode requires at least one prompt"):
            self.module._validate_spec(value)

    def test_persistent_unknown_node_with_any_prompt_is_rejected(self):
        value = copy.deepcopy(self.spec_407)
        support = next(
            node for node in value["nodes"] if node["node_id"] == "nightstand_support"
        )
        support["source_prompt"] = copy.deepcopy(
            next(node for node in value["nodes"] if node["node_id"] == "phone")[
                "source_prompt"
            ]
        )
        with self.assertRaisesRegex(Exception, "persistent-unknown mode forbids prompts"):
            self.module._validate_spec(value)

    def test_unknown_mode_is_rejected(self):
        value = copy.deepcopy(self.spec_407)
        value["nodes"][0]["observation_mode"] = "implicit_fallback"
        with self.assertRaisesRegex(Exception, "observation mode differs"):
            self.module._validate_spec(value)

    def test_persistent_unknown_phase_is_always_unresolved(self):
        for frame in self.module.PHASE_FRAMES:
            self.assertEqual(
                self.module._phase_status(None, frame, None, "persistent_unknown"),
                "unresolved_persistent_unknown",
            )

    def test_relative_geometry_channels_are_explicit(self):
        source = PROGRAM.read_text(encoding="utf-8")
        for name in (
            "relative_dx_norm",
            "relative_dy_norm",
            "relative_dx_delta_from_first_joint_observation",
            "relative_dy_delta_from_first_joint_observation",
            "relative_dx_phase_velocity",
            "relative_dy_phase_velocity",
        ):
            self.assertIn(name, source)


if __name__ == "__main__":
    unittest.main()
