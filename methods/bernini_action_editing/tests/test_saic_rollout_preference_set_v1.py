from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import saic_rollout_preference_set_v1 as preference  # noqa: E402


def _sha(character: str) -> str:
    return character * 64


def _seal(row: dict[str, object], field: str) -> dict[str, object]:
    result = deepcopy(row)
    result[field] = preference.object_sha256(result)
    return result


def _candidate(
    candidate_id: str,
    *,
    arm: str,
    event: float,
    axis: float = 0.9,
    source: str | None = None,
) -> dict[str, object]:
    source_id = source or f"{arm}-source"
    character = format((sum(candidate_id.encode("ascii")) % 14) + 1, "x")
    output_sha = _sha(character)
    endpoint_sha = _sha("e" if candidate_id.endswith("chosen") else "f")
    rollout = _seal(
        {
            "schema_version": preference.ROLLOUT_RECEIPT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "generation_mode": preference.GENERATION_MODE,
            "on_policy": True,
            "weights_frozen_during_rollout": True,
            "source_conditioned": True,
            "policy_sha256": _sha("a"),
            "source_id": source_id,
            "source_media_sha256": _sha("b" if arm == "dog" else "c"),
            "output_media_sha256": output_sha,
            "frame_count": 81,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "exact40_step_count": 40,
            "preference_update_indices": list(preference.EXACT40_UPDATE_INDICES),
            "pure_t2v_media_read": False,
            "pure_t2v_latent_read": False,
            "pure_t2v_noise_read": False,
            "target_media_read": False,
            "paired_target_read": False,
        },
        "receipt_digest",
    )
    codec = _seal(
        {
            "schema_version": preference.CODEC_RECEIPT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "input_output_media_sha256": output_sha,
            "decoded_rgb24_sha256": _sha("1"),
            "codec_name": "h264-crf18",
            "codec_bitstream_sha256": _sha("2"),
            "codec_decoded_rgb24_sha256": _sha("3"),
            "vae_id": "bernini-vae",
            "vae_weights_sha256": _sha("4"),
            "reencoded_latent_sha256": endpoint_sha,
            "frame_count": 81,
            "fps_numerator": 25,
            "fps_denominator": 1,
            "endpoint_detached": True,
        },
        "receipt_digest",
    )
    return _seal(
        {
            "schema_version": preference.SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "arm": arm,
            "source_id": source_id,
            "instruction_id": f"{arm}-action",
            "policy_sha256": _sha("a"),
            "source_media_sha256": _sha("b" if arm == "dog" else "c"),
            "output_media_sha256": output_sha,
            "endpoint_latent_sha256": endpoint_sha,
            "declared_role": preference.ENDPOINT_ROLE,
            "event_score": event,
            "axis_scores": {name: axis for name in preference.PRESERVATION_AXES},
            "rollout_receipt": rollout,
            "codec_reencode_receipt": codec,
        },
        "candidate_digest",
    )


def _floor(value: float) -> dict[str, float]:
    return {axis: value for axis in preference.PRESERVATION_AXES}


class SAICRolloutPreferenceSetTests(unittest.TestCase):
    def _population(self) -> list[dict[str, object]]:
        return [
            _candidate("dog-chosen", arm="dog", event=0.90),
            _candidate("dog-rejected", arm="dog", event=0.50),
            _candidate("human-chosen", arm="human", event=0.85),
            _candidate("human-rejected", arm="human", event=0.45),
        ]

    def test_two_arm_hard_pareto_set_authorizes_exactly_two_pairs(self) -> None:
        result = preference.build_preference_set(
            self._population(), minimum_event_gain=0.2, axis_floors=_floor(0.8)
        )
        self.assertTrue(result.optimizer_step_allowed)
        self.assertEqual([row["arm"] for row in result.authorized_pairs], ["dog", "human"])
        self.assertEqual(len(result.authorized_pairs), 2)
        self.assertFalse(result.receipt["scalar_reward_or_weighted_compensation_used"])
        self.assertFalse(result.receipt["pure_t2v_endpoint_used"])
        self.assertEqual(
            tuple(result.receipt["preference_update_indices"]),
            preference.EXACT40_UPDATE_INDICES,
        )

    def test_missing_human_pair_forces_global_zero_update(self) -> None:
        result = preference.build_preference_set(
            self._population()[:2], minimum_event_gain=0.2, axis_floors=_floor(0.8)
        )
        self.assertFalse(result.optimizer_step_allowed)
        self.assertEqual(result.authorized_pairs, ())
        self.assertEqual(result.diagnostic_admissible_pair_count_by_arm["dog"], 1)
        self.assertEqual(result.diagnostic_admissible_pair_count_by_arm["human"], 0)
        self.assertIsNotNone(result.zero_update_reason)

    def test_single_candidate_can_never_authorize_a_pair(self) -> None:
        result = preference.build_preference_set(
            [self._population()[0]],
            minimum_event_gain=0.2,
            axis_floors=_floor(0.8),
        )
        self.assertFalse(result.optimizer_step_allowed)
        self.assertEqual(result.authorized_pairs, ())
        self.assertEqual(
            result.diagnostic_admissible_pair_count_by_arm,
            {"dog": 0, "human": 0},
        )

    def test_one_axis_failure_cannot_be_compensated(self) -> None:
        rows = self._population()
        # Improve every other score and event, but degrade one preservation axis.
        rows[0]["axis_scores"] = _floor(1.0)
        rows[0]["axis_scores"]["camera"] = 0.89
        rows[1]["axis_scores"] = _floor(0.90)
        rows[0]["candidate_digest"] = preference.object_sha256(
            {key: value for key, value in rows[0].items() if key != "candidate_digest"}
        )
        rows[1]["candidate_digest"] = preference.object_sha256(
            {key: value for key, value in rows[1].items() if key != "candidate_digest"}
        )
        result = preference.build_preference_set(
            rows, minimum_event_gain=0.2, axis_floors=_floor(0.8)
        )
        self.assertFalse(result.optimizer_step_allowed)
        self.assertEqual(result.diagnostic_admissible_pair_count_by_arm["dog"], 0)

    def test_both_endpoints_must_pass_each_absolute_floor(self) -> None:
        rows = self._population()
        rows[1]["axis_scores"]["source_bind"] = 0.1
        rows[1]["candidate_digest"] = preference.object_sha256(
            {key: value for key, value in rows[1].items() if key != "candidate_digest"}
        )
        result = preference.build_preference_set(
            rows, minimum_event_gain=0.2, axis_floors=_floor(0.8)
        )
        self.assertFalse(result.optimizer_step_allowed)

    def test_pure_t2v_endpoint_and_media_reads_are_rejected(self) -> None:
        row = _candidate("dog-chosen", arm="dog", event=0.9)
        row["rollout_receipt"]["generation_mode"] = "pure_t2v"
        row["rollout_receipt"]["pure_t2v_media_read"] = True
        row["rollout_receipt"]["receipt_digest"] = preference.object_sha256(
            {
                key: value
                for key, value in row["rollout_receipt"].items()
                if key != "receipt_digest"
            }
        )
        row["candidate_digest"] = preference.object_sha256(
            {key: value for key, value in row.items() if key != "candidate_digest"}
        )
        with self.assertRaisesRegex(preference.SAICRolloutPreferenceError, "pure-T2V"):
            preference.validate_candidate(row)

    def test_codec_reencoded_endpoint_binding_and_seals_fail_closed(self) -> None:
        row = _candidate("dog-chosen", arm="dog", event=0.9)
        row["codec_reencode_receipt"]["reencoded_latent_sha256"] = _sha("9")
        row["codec_reencode_receipt"]["receipt_digest"] = preference.object_sha256(
            {
                key: value
                for key, value in row["codec_reencode_receipt"].items()
                if key != "receipt_digest"
            }
        )
        row["candidate_digest"] = preference.object_sha256(
            {key: value for key, value in row.items() if key != "candidate_digest"}
        )
        with self.assertRaisesRegex(preference.SAICRolloutPreferenceError, "codec-reencoded"):
            preference.validate_candidate(row)
        row = _candidate("dog-chosen", arm="dog", event=0.9)
        row["event_score"] = 0.91
        with self.assertRaisesRegex(preference.SAICRolloutPreferenceError, "digest differs"):
            preference.validate_candidate(row)

    def test_exact40_registered_set_excludes_last_two(self) -> None:
        self.assertEqual(
            preference.EXACT40_UPDATE_INDICES,
            (4, 12, 20, 28, 33, 34, 35, 37),
        )
        for index in preference.EXACT40_UPDATE_INDICES:
            self.assertEqual(preference.validate_update_index(index), index)
        for index in (38, 39):
            with self.assertRaisesRegex(preference.SAICRolloutPreferenceError, "38/39"):
                preference.validate_update_index(index)


if __name__ == "__main__":
    unittest.main()
