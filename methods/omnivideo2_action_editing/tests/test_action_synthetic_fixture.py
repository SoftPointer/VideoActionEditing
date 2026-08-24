from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action import ActionLatentDataset, load_action_config  # noqa: E402
from tools.build_action_synthetic_fixture import build_fixture  # noqa: E402


class ActionSyntheticFixtureTest(unittest.TestCase):
    def test_four_row_preview_fixture_is_mask_free_and_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            receipt = build_fixture(output, seed=7)
            self.assertEqual(receipt["sample_count"], 4)
            self.assertFalse(receipt["mask_or_tube_inputs"])
            config = load_action_config(output / receipt["config"])
            self.assertTrue(config.training.allow_preview)
            dataset = ActionLatentDataset(
                output / "manifest.jsonl",
                payload_root=output / "payloads",
                expected_motion_tokens=config.planner.num_tokens,
                allowed_task_types=config.training.allowed_task_types,
                allow_preview=True,
            )
            self.assertEqual(len(dataset), 4)
            self.assertEqual(
                {dataset[index]["task_type"] for index in range(4)},
                {
                    "action_edit",
                    "identity_reconstruction",
                    "native_replay",
                    "native_isolation_probe",
                },
            )
            forbidden = {"mask", "tube", "track", "source_erasure"}
            self.assertFalse(forbidden & set(dataset[0]))
            with self.assertRaisesRegex(ValueError, "preview-only"):
                ActionLatentDataset(
                    output / "manifest.jsonl",
                    payload_root=output / "payloads",
                    expected_motion_tokens=config.planner.num_tokens,
                    allowed_task_types=config.training.allowed_task_types,
                    allow_preview=False,
                )


if __name__ == "__main__":
    unittest.main()
