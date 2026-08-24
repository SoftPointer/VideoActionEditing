from __future__ import annotations

from pathlib import Path
import re
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_train_preservation_residual_single_holder_v1.sh"
)


class PreservationSingleHolderLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_closed_holder_allowlist_and_parent_retention(self) -> None:
        self.assertIn("135407:auh7-1b-gpu-260", self.text)
        self.assertIn("135411:auh7-1b-gpu-214", self.text)
        lowered = self.text.lower()
        self.assertNotIn("scancel", lowered)
        self.assertNotIn("scontrol release", lowered)
        self.assertNotRegex(lowered, r"kill[^\n]*(135407|135411)")
        self.assertIn("parent_not_released=true", self.text)

    def test_world8_dp2_sp4_and_two_registered_adapter_ranks(self) -> None:
        self.assertIn("--parallel-topology world8-dp2-sp4", self.text)
        self.assertIn("--nproc_per_node=8", self.text)
        self.assertIn("--gres=gpu:mi210:8", self.text)
        self.assertRegex(self.text, re.compile(r'case "\$\{adapter_rank\}" in 2\|8\)'))

    def test_preservation_objective_entrypoint_is_the_only_trainer(self) -> None:
        self.assertIn("train_preservation_residual_v1.py", self.text)
        self.assertNotIn("train_source_noised_carrier_strata_v1.py", self.text)
        self.assertNotIn("source_kv_route_objective", self.text)

    def test_checkpoint_cadence_is_exact20_or_exact40(self) -> None:
        self.assertIn(
            'case "${optimizer_steps}" in 20|40)',
            self.text,
        )

    def test_exact20_must_reproduce_historical_prefix(self) -> None:
        self.assertIn(
            "20af97615bf51aba46c59795f21330a5563426826043faa8ad5626ad17c5f42a",
            self.text,
        )
        self.assertIn(
            "2a5f775212796fbe7836f206ef3a0e9f49dced7c544b08323cba599f6900ffc9",
            self.text,
        )
        self.assertIn("exact20_prefix_reproduced=true", self.text)

    def test_exact40_publishes_immutable_zero_and_twenty_bundles(self) -> None:
        self.assertIn(
            'checkpoint_args=(--checkpoint-output-root "${checkpoint_output_root}")',
            self.text,
        )
        self.assertIn("for step in 00000000 00000020", self.text)
        self.assertIn("checkpoint_steps=0,20,40", self.text)


if __name__ == "__main__":
    unittest.main()
