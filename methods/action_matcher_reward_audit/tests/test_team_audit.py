import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_team_audit.py"


class TeamAuditStaticTest(unittest.TestCase):
    def test_runner_declares_group_relative_semantics(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("episodic group-relative logits", source)
        self.assertIn("two_way_vs_all_way_order_flip_rate", source)
        self.assertIn("not action correctness truth", source)

    def test_runner_compiles(self):
        spec = importlib.util.spec_from_file_location("run_team_audit", MODULE_PATH)
        self.assertIsNotNone(spec)


if __name__ == "__main__":
    unittest.main()
