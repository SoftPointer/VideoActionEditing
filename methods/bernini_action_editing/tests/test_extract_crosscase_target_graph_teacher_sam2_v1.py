import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "extract_crosscase_target_graph_teacher_sam2_v1.py"
ASSETS = ROOT / "assets"
LAUNCHER = ROOT / "scripts" / "auh_extract_crosscase_target_graph_teacher_sam2_v1.sh"
CASES = ("8b05aaf463db", "40712e1341dc", "5e83a9279951")


class CrosscaseTargetGraphTeacherContractTests(unittest.TestCase):
    def setUp(self):
        self.source = PROGRAM.read_text(encoding="utf-8")
        self.launcher = LAUNCHER.read_text(encoding="utf-8")
        self.specs = {
            case_id: json.loads(
                (ASSETS / f"{case_id}_target_graph_teacher_sam2_spec_v1.json")
                .read_text(encoding="utf-8")
            )
            for case_id in CASES
        }

    def test_program_parses_and_has_no_renderer_dependency(self):
        ast.parse(self.source)
        self.assertNotIn("diffsynth", self.source)
        self.assertNotIn("optimizer.step", self.source)
        self.assertIn('"teacher_observation_scaffold_not_oceg": True', self.source)
        self.assertIn('"generator_read_authorized": False', self.source)
        self.assertIn('"training_authorized": False', self.source)

    def test_all_specs_are_review_only_frozen_observers(self):
        for case_id, spec in self.specs.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(spec["case_id"], case_id)
                self.assertTrue(spec["sam2"]["frozen"])
                self.assertTrue(spec["sam2"]["separate_node_states"])
                self.assertFalse(spec["claim_limits"]["generator_read_authorized"])
                self.assertFalse(spec["claim_limits"]["renderer_condition_authorized"])
                self.assertFalse(spec["claim_limits"]["training_authorized"])
                self.assertFalse(spec["claim_limits"]["selection_authorized"])
                self.assertEqual(spec["claim_limits"]["optimizer_updates"], 0)

    def test_407_phone_disappearance_fails_closed(self):
        spec = self.specs["40712e1341dc"]
        phone = next(node for node in spec["nodes"] if node["node_id"] == "phone")
        self.assertEqual(phone["real_target_teacher_prompt"]["review_reliable_end"], 48)
        events = {event["event_id"]: event for event in spec["manual_event_program"]}
        self.assertEqual(events["phone_support_and_release"]["status"], "review_unresolved")
        self.assertEqual(events["terminal_phone_support"]["status"], "review_unresolved")

    def test_5e83_phone_is_not_fabricated_and_tablet_is_separate(self):
        spec = self.specs["5e83a9279951"]
        nodes = {node["node_id"]: node for node in spec["nodes"]}
        self.assertIsNone(nodes["phone"]["source_prompt"])
        self.assertEqual(nodes["phone"]["real_target_teacher_prompt"]["review_reliable_start"], 56)
        self.assertIsNotNone(nodes["tablet"]["source_prompt"])
        self.assertIsNotNone(nodes["tablet"]["real_target_teacher_prompt"])
        self.assertNotEqual(nodes["tablet"]["object_id"], nodes["phone"]["object_id"])
        events = {event["event_id"]: event for event in spec["manual_event_program"]}
        self.assertEqual(events["phone_initial_identity"]["status"], "review_unresolved")

    def test_launcher_uses_only_authorized_holder_steps(self):
        for job_id, node in (
            ("147881", "auh7-1b-gpu-213"),
            ("147873", "auh7-1b-gpu-284"),
            ("147871", "auh7-1b-gpu-232"),
        ):
            self.assertIn(job_id, self.launcher)
            self.assertIn(node, self.launcher)
        self.assertNotIn("scancel", self.launcher)
        self.assertNotIn("scontrol", self.launcher)


if __name__ == "__main__":
    unittest.main()
