from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_candidate_evaluator_packet as packet  # noqa: E402
import pair_v5_safe_pareto as safe  # noqa: E402


def _sha(character: str) -> str:
    return character * 64


class EvaluatorPacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = packet.make_registry(
            {
                axis: packet.make_evaluator_binding(
                    f"{axis}-evaluator-v1",
                    evaluator_sha256=_sha(str(index + 1)),
                    model_digest=_sha(chr(ord("a") + index)),
                )
                for index, axis in enumerate(packet.EVALUATOR_AXES)
            }
        )
        self.flags = {name: False for name in packet.HARD_NEGATIVE_FLAGS}
        self.action_score = {
            "raw_candidate_own_score": 0.75,
            "calibrated_action_score": 0.80,
            "candidate_evaluator_receipt_digest": _sha("9"),
        }
        upstream = {
            axis: _sha(str(index + 5))
            for index, axis in enumerate(packet.EVALUATOR_AXES)
        }
        upstream["action"] = self.action_score["candidate_evaluator_receipt_digest"]
        self.value = packet.make_packet(
            "candidate-a",
            rollout_receipt_digest=_sha("1"),
            mp4_sha256=_sha("2"),
            source_video_sha256=_sha("3"),
            complete_caption_sha256=_sha("4"),
            evaluator_registry_digest=self.registry["registry_digest"],
            upstream_evaluator_receipt_digest_by_axis=upstream,
            raw_scores={
                "action": 0.75,
                "identity": 0.90,
                "consistency": 0.85,
                "quality": 0.88,
            },
            reported_scores={
                "action": 0.80,
                "identity": 0.90,
                "consistency": 0.85,
                "quality": 0.88,
            },
            hard_negative_flags=self.flags,
        )
        self.candidate = safe.make_candidate(
            "candidate-a",
            action_score=0.80,
            identity_score=0.90,
            consistency_score=0.85,
            quality_score=0.88,
            hard_negative_flags=self.flags,
            evaluator_packet_digest=self.value["packet_digest"],
            rollout_receipt_digest=_sha("1"),
        )

    def verify(self, value=None, registry=None, candidate=None, action_score=None):
        return packet.verify_packet(
            self.value if value is None else value,
            registry=self.registry if registry is None else registry,
            safe_candidate=self.candidate if candidate is None else candidate,
            action_score_receipt=(
                self.action_score if action_score is None else action_score
            ),
            rollout_receipt_digest=_sha("1"),
            mp4_sha256=_sha("2"),
            source_video_sha256=_sha("3"),
            complete_caption_sha256=_sha("4"),
            expected_action_evaluator_sha256=_sha("1"),
            expected_action_model_digest=_sha("a"),
        )

    def test_every_axis_and_physical_candidate_replay_exactly(self) -> None:
        checked = self.verify()
        self.assertEqual(checked["packet_digest"], self.value["packet_digest"])
        self.assertEqual(set(packet.EVALUATOR_AXES), set(self.registry["evaluators"]))

    def test_mp4_cross_swap_fails_even_when_packet_is_resealed(self) -> None:
        changed = deepcopy(self.value)
        changed["mp4_sha256"] = _sha("f")
        changed.pop("packet_digest")
        changed["packet_digest"] = packet.object_sha256(changed)
        with self.assertRaisesRegex(
            packet.PairV5EvaluatorPacketError, "physical candidate binding differs"
        ):
            self.verify(value=changed)

    def test_fake_action_receipt_digest_fails(self) -> None:
        action_score = dict(self.action_score)
        action_score["candidate_evaluator_receipt_digest"] = _sha("e")
        with self.assertRaisesRegex(
            packet.PairV5EvaluatorPacketError, "packet action evidence"
        ):
            self.verify(action_score=action_score)

    def test_wrong_evaluator_or_model_registry_fails(self) -> None:
        registry = deepcopy(self.registry)
        registry["evaluators"]["action"]["model_digest"] = _sha("f")
        registry.pop("registry_digest")
        registry["registry_digest"] = packet.object_sha256(registry)
        changed = deepcopy(self.value)
        changed["evaluator_registry_digest"] = registry["registry_digest"]
        changed.pop("packet_digest")
        changed["packet_digest"] = packet.object_sha256(changed)
        candidate = safe.make_candidate(
            "candidate-a",
            action_score=0.80,
            identity_score=0.90,
            consistency_score=0.85,
            quality_score=0.88,
            hard_negative_flags=self.flags,
            evaluator_packet_digest=changed["packet_digest"],
            rollout_receipt_digest=_sha("1"),
        )
        with self.assertRaisesRegex(
            packet.PairV5EvaluatorPacketError, "action evaluator/model registry differs"
        ):
            self.verify(value=changed, registry=registry, candidate=candidate)


if __name__ == "__main__":
    unittest.main()
