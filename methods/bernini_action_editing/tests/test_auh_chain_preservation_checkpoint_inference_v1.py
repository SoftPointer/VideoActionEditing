from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_chain_preservation_checkpoint_inference_v1.sh"
)


class PreservationCheckpointChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_waits_for_one_continuous_trajectory(self) -> None:
        self.assertIn('while [[ ! -f "${training_run}/controller.COMPLETE" ]]', self.text)
        self.assertIn('step-00000000', self.text)
        self.assertIn('step-00000020', self.text)
        self.assertIn('assert final["checkpoint_steps"] == [0, 20, 40]', self.text)
        self.assertIn('== [0, 20]', self.text)

    def test_inference_is_ordered_twenty_then_forty(self) -> None:
        first = self.text.index('state=running_step20')
        second = self.text.index('state=running_step40')
        complete = self.text.index('state=complete')
        self.assertLess(first, second)
        self.assertLess(second, complete)
        self.assertIn('PRESERVATION_INFER_TRAINING_BUNDLE="${step20_bundle}"', self.text)
        self.assertIn('PRESERVATION_INFER_TRAINING_BUNDLE="${step40_bundle}"', self.text)

    def test_parent_is_never_released_or_signalled(self) -> None:
        lowered = self.text.lower()
        for forbidden in ("scancel", "scontrol release", "scontrol requeue", "kill -"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn('parent_not_released=true', self.text)


if __name__ == "__main__":
    unittest.main()
