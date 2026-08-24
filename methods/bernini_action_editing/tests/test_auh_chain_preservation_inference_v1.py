from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_chain_preservation_inference_after_training_v1.sh"
)


class PreservationInferenceChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_waits_without_allocating_a_child(self) -> None:
        waiting = self.text.split("done", 1)[0]
        self.assertIn("sleep", waiting)
        self.assertNotIn("srun", waiting)
        self.assertIn("controller.COMPLETE", waiting)

    def test_reuses_same_two_parent_allowlist(self) -> None:
        self.assertIn("135407:auh7-1b-gpu-260", self.text)
        self.assertIn("135411:auh7-1b-gpu-214", self.text)
        self.assertNotIn("scancel", self.text.lower())
        self.assertIn("auh_infer_preservation_residual_single_holder_v1.sh", self.text)


if __name__ == "__main__":
    unittest.main()
