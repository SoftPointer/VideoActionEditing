from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_lora as trainer  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - exercised on the AUH runtime.
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class RewardPreferenceObjectiveTests(unittest.TestCase):
    def sample(self):
        source = torch.arange(1 * 32 * 21 * 2 * 2, dtype=torch.float32).reshape(
            1, 32, 21, 2, 2
        )
        target = source + 10000
        messages = [
            {"type": "video", "has_loss": 0},
            {"type": "text", "has_loss": 0, "text": "Make the person stand up."},
            {"type": "video_gen", "has_loss": 1},
        ]
        return {
            "inputs": json.dumps(messages),
            "video_vae_latents": [
                trainer._tensor_blob(source),
                trainer._tensor_blob(target),
            ],
            "source_name": trainer.TASK_SOURCE_NAME,
        }, source, target

    def test_negative_rotation_is_deterministic(self) -> None:
        self.assertEqual(
            [trainer.contrastive_negative_kind(step) for step in range(6)],
            ["noop", "reverse", "incomplete", "noop", "reverse", "incomplete"],
        )
        for kind in trainer.CONTRASTIVE_NEGATIVE_KINDS:
            self.assertEqual(
                [
                    trainer.contrastive_negative_kind(step, schedule=kind)
                    for step in range(6)
                ],
                [kind] * 6,
            )
        self.assertEqual(
            [
                trainer.contrastive_negative_kind(
                    step, schedule="noop_incomplete"
                )
                for step in range(6)
            ],
            ["noop", "incomplete", "noop", "incomplete", "noop", "incomplete"],
        )

    def test_three_high_contrast_targets_are_exact(self) -> None:
        sample, source, target = self.sample()
        noop = trainer.build_contrastive_sample(sample, negative_kind="noop")
        reverse = trainer.build_contrastive_sample(sample, negative_kind="reverse")
        incomplete = trainer.build_contrastive_sample(sample, negative_kind="incomplete")
        self.assertTrue(torch.equal(trainer._load_tensor_blob(noop["video_vae_latents"][1]), source))
        self.assertTrue(
            torch.equal(
                trainer._load_tensor_blob(reverse["video_vae_latents"][1]),
                target.flip(dims=(2,)),
            )
        )
        incomplete_tensor = trainer._load_tensor_blob(incomplete["video_vae_latents"][1])
        self.assertTrue(torch.equal(incomplete_tensor[:, :, :11], target[:, :, :11]))
        self.assertTrue(
            torch.equal(
                incomplete_tensor[:, :, 11:],
                target[:, :, 10:11].expand_as(incomplete_tensor[:, :, 11:]),
            )
        )

    def test_identity_branch_changes_only_instruction_and_target(self) -> None:
        sample, source, _ = self.sample()
        identity = trainer.build_identity_preservation_sample(sample)
        messages = json.loads(identity["inputs"])
        self.assertEqual(messages[1]["text"], trainer.IDENTITY_PRESERVATION_INSTRUCTION)
        self.assertTrue(
            torch.equal(trainer._load_tensor_blob(identity["video_vae_latents"][0]), source)
        )
        self.assertTrue(
            torch.equal(trainer._load_tensor_blob(identity["video_vae_latents"][1]), source)
        )

    def test_margin_loss_rewards_a_larger_rejected_minus_chosen_gap(self) -> None:
        chosen = torch.tensor(0.2, requires_grad=True)
        rejected_bad = torch.tensor(0.21, requires_grad=True)
        rejected_good = torch.tensor(0.50)
        bad = trainer.high_contrast_preference_loss(
            chosen, rejected_bad, margin=0.05, temperature=20.0
        )
        good = trainer.high_contrast_preference_loss(
            chosen.detach(), rejected_good, margin=0.05, temperature=20.0
        )
        self.assertGreater(float(bad), float(good))
        bad.backward()
        self.assertGreater(float(chosen.grad), 0.0)
        self.assertLess(float(rejected_bad.grad), 0.0)

    def test_reference_dpo_uses_improvement_over_frozen_base(self) -> None:
        reference_chosen = torch.tensor(0.20)
        reference_rejected = torch.tensor(0.30)
        unchanged = trainer.reference_dpo_loss(
            torch.tensor(0.20),
            torch.tensor(0.30),
            reference_chosen,
            reference_rejected,
            beta=10.0,
        )
        improved = trainer.reference_dpo_loss(
            torch.tensor(0.15),
            torch.tensor(0.40),
            reference_chosen,
            reference_rejected,
            beta=10.0,
        )
        self.assertGreater(float(unchanged), float(improved))

    def test_detached_rejected_loss_cannot_push_rejected_error_up(self) -> None:
        chosen = torch.tensor(0.20, requires_grad=True)
        rejected = torch.tensor(0.21, requires_grad=True)
        loss = trainer.detached_rejected_preference_loss(
            chosen, rejected, margin=0.05, temperature=20.0
        )
        loss.backward()
        self.assertGreater(float(chosen.grad), 0.0)
        self.assertIsNone(rejected.grad)


if __name__ == "__main__":
    unittest.main()
