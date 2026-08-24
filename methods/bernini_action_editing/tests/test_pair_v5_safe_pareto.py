from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_safe_pareto as pair_v5  # noqa: E402


def _sha(character: str) -> str:
    return character * 64


def _flags(**updates: bool) -> dict[str, bool]:
    value = {name: False for name in pair_v5.HARD_NEGATIVE_FLAGS}
    value.update(updates)
    return value


def _candidate(
    candidate_id: str,
    action: float,
    identity: float,
    consistency: float,
    quality: float,
    **flags: bool,
) -> dict:
    return pair_v5.make_candidate(
        candidate_id,
        action_score=action,
        identity_score=identity,
        consistency_score=consistency,
        quality_score=quality,
        hard_negative_flags=_flags(**flags),
        evaluator_packet_digest=pair_v5.object_sha256(
            {"candidate_id": candidate_id, "binding": "evaluator-packet"}
        ),
        rollout_receipt_digest=pair_v5.object_sha256(
            {"candidate_id": candidate_id, "binding": "native-rollout"}
        ),
    )


class PairV5Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = pair_v5.make_policy(
            "cdf-dog-pair-v5",
            bootstrap_action_delta=0.10,
            max_identity_degradation=0.05,
            max_consistency_degradation=0.05,
            max_quality_degradation=0.05,
            min_action_score=0.80,
            min_identity_score=0.80,
            min_consistency_score=0.80,
            min_quality_score=0.80,
        )
        self.provenance = pair_v5.make_calibrator_provenance(
            "mace-candidate-own-coordinate-v1",
            action_evaluator_sha256=_sha("a"),
            calibration_receipt_sha256=_sha("b"),
            calibration_receipt_digest=_sha("c"),
        )
        self.initial = pair_v5.initial_state(self.policy)

    def advance(self, candidates, state=None):
        return pair_v5.advance_pair_selector(
            state=self.initial if state is None else state,
            candidates=candidates,
            policy=self.policy,
            calibrator_provenance=self.provenance,
        )


class SafeParetoBootstrapTests(PairV5Fixture):
    def test_bootstrap_selects_action_gain_with_bounded_preservation_loss(self) -> None:
        loser = _candidate("loser", 0.30, 0.92, 0.91, 0.90)
        winner = _candidate("winner", 0.52, 0.89, 0.88, 0.87)
        receipt = self.advance([loser, winner])

        self.assertEqual(receipt["stage_before"], pair_v5.BOOTSTRAP_STAGE)
        self.assertEqual(receipt["stage_after"], pair_v5.BOOTSTRAP_STAGE)
        self.assertFalse(receipt["transitioned_to_strict"])
        self.assertEqual(
            receipt["decision"], "selected_safe_pareto_bootstrap_pair"
        )
        selected = receipt["selected_pair"]
        self.assertEqual(selected["winner_candidate_id"], "winner")
        self.assertEqual(selected["loser_candidate_id"], "loser")
        self.assertAlmostEqual(selected["action_improvement"], 0.22)
        self.assertAlmostEqual(selected["identity_degradation"], 0.03)
        self.assertTrue(selected["safe_pareto_pass"])
        self.assertFalse(selected["winner_absolute_feasible"])
        self.assertFalse(selected["strict_feasible_pass"])

    def test_action_gain_and_every_preservation_bound_are_mandatory(self) -> None:
        loser = _candidate("loser", 0.30, 0.92, 0.91, 0.90)
        weak_action = _candidate("weak-action", 0.39, 0.92, 0.91, 0.90)
        identity_drop = _candidate("identity-drop", 0.60, 0.86, 0.91, 0.90)
        consistency_drop = _candidate(
            "consistency-drop", 0.60, 0.92, 0.85, 0.90
        )
        quality_drop = _candidate("quality-drop", 0.60, 0.92, 0.91, 0.84)
        for winner in (
            weak_action,
            identity_drop,
            consistency_drop,
            quality_drop,
        ):
            with self.subTest(winner=winner["candidate_id"]):
                receipt = self.advance([loser, winner])
                self.assertEqual(receipt["eligible_pair_count"], 0)
                self.assertIsNone(receipt["selected_pair"])

    def test_every_registered_hard_negative_is_forbidden_as_winner(self) -> None:
        clean_loser = _candidate("clean-loser", 0.20, 0.90, 0.90, 0.90)
        for flag in pair_v5.HARD_NEGATIVE_FLAGS:
            with self.subTest(flag=flag):
                flagged = _candidate(
                    f"flagged-{flag.replace('_', '-')}",
                    0.95,
                    0.90,
                    0.90,
                    0.90,
                    **{flag: True},
                )
                receipt = self.advance([clean_loser, flagged])
                selected_winner = (
                    None
                    if receipt["selected_pair"] is None
                    else receipt["selected_pair"]["winner_candidate_id"]
                )
                self.assertNotEqual(
                    selected_winner,
                    flagged["candidate_id"],
                )
                self.assertNotIn(
                    flagged["candidate_id"],
                    receipt["absolute_feasible_candidate_ids"],
                )

    def test_hard_negative_cannot_be_loser_either(self) -> None:
        winner = _candidate("clean-winner", 0.65, 0.90, 0.90, 0.90)
        for flag in pair_v5.HARD_NEGATIVE_FLAGS:
            with self.subTest(flag=flag):
                flagged_loser = _candidate(
                    f"flagged-loser-{flag.replace('_', '-')}",
                    0.20,
                    0.92,
                    0.92,
                    0.92,
                    **{flag: True},
                )
                receipt = self.advance([winner, flagged_loser])
                self.assertEqual(receipt["eligible_pair_count"], 0)
                self.assertIsNone(receipt["selected_pair"])

    def test_deterministic_ranking_prefers_decisive_gain(self) -> None:
        loser = _candidate("loser", 0.20, 0.90, 0.90, 0.90)
        medium = _candidate("medium", 0.45, 0.88, 0.88, 0.88)
        strongest = _candidate("strongest", 0.65, 0.87, 0.87, 0.87)
        receipt = self.advance([medium, loser, strongest])
        self.assertEqual(
            receipt["selected_pair"]["winner_candidate_id"], "strongest"
        )
        self.assertEqual(
            receipt["selected_pair"]["loser_candidate_id"], "loser"
        )


class IrreversibleStageTests(PairV5Fixture):
    def _strict_transition(self):
        loser = _candidate("near-miss", 0.55, 0.94, 0.93, 0.92)
        feasible = _candidate("feasible", 0.85, 0.90, 0.89, 0.88)
        return self.advance([loser, feasible])

    def test_first_absolute_feasible_candidate_switches_same_event_to_strict(self) -> None:
        receipt = self._strict_transition()
        self.assertEqual(receipt["stage_before"], pair_v5.BOOTSTRAP_STAGE)
        self.assertEqual(receipt["stage_after"], pair_v5.STRICT_STAGE)
        self.assertTrue(receipt["transitioned_to_strict"])
        self.assertEqual(receipt["absolute_feasible_candidate_ids"], ["feasible"])
        self.assertEqual(
            receipt["decision"], "selected_strict_feasible_pair"
        )
        selected = receipt["selected_pair"]
        self.assertEqual(selected["winner_candidate_id"], "feasible")
        self.assertEqual(selected["loser_candidate_id"], "near-miss")
        self.assertTrue(selected["winner_absolute_feasible"])
        self.assertTrue(selected["winner_hard_negative_flags_all_false"])
        self.assertTrue(selected["loser_hard_negative_flags_all_false"])
        self.assertTrue(selected["strict_feasible_pass"])
        # Feasible-only is winner eligibility: the clean loser deliberately
        # fails the absolute action threshold and remains a useful near miss.
        self.assertNotIn(
            selected["loser_candidate_id"],
            receipt["absolute_feasible_candidate_ids"],
        )
        closure = receipt["condition_closure"]
        self.assertTrue(closure["strict_feasible_only_means_winner_eligibility"])
        self.assertTrue(closure["both_pair_endpoints_hard_negative_free"])
        self.assertTrue(
            closure["relative_safe_pareto_constraints_retained_in_strict"]
        )
        state = receipt["next_state"]
        self.assertEqual(state["stage"], pair_v5.STRICT_STAGE)
        self.assertEqual(state["strict_since_revision"], 1)

    def test_strict_state_rejects_later_bootstrap_only_pair_and_never_reverts(self) -> None:
        first = self._strict_transition()
        loser = _candidate("later-loser", 0.20, 0.91, 0.91, 0.91)
        nonfeasible = _candidate("later-nonfeasible", 0.55, 0.88, 0.88, 0.88)
        second = self.advance(
            [nonfeasible, loser], state=first["next_state"]
        )

        self.assertEqual(second["stage_before"], pair_v5.STRICT_STAGE)
        self.assertEqual(second["stage_after"], pair_v5.STRICT_STAGE)
        self.assertFalse(second["transitioned_to_strict"])
        self.assertEqual(second["eligible_pair_count"], 0)
        self.assertIsNone(second["selected_pair"])
        self.assertEqual(
            second["next_state"]["strict_since_revision"],
            first["next_state"]["strict_since_revision"],
        )

    def test_strict_stage_still_rejects_relative_preservation_violation(self) -> None:
        # The winner is absolutely feasible, so this event enters strict, but
        # it loses too much identity relative to the clean near miss.
        near_miss = _candidate("identity-strong-near-miss", 0.55, 0.95, 0.90, 0.90)
        feasible_but_unsafe = _candidate(
            "feasible-but-unsafe", 0.85, 0.80, 0.90, 0.90
        )
        receipt = self.advance([near_miss, feasible_but_unsafe])
        self.assertEqual(receipt["stage_after"], pair_v5.STRICT_STAGE)
        self.assertEqual(
            receipt["absolute_feasible_candidate_ids"], ["feasible-but-unsafe"]
        )
        self.assertEqual(receipt["eligible_pair_count"], 0)
        self.assertIsNone(receipt["selected_pair"])

    def test_correctly_rehashed_strict_to_bootstrap_transition_still_fails(self) -> None:
        strict = self._strict_transition()["next_state"]
        forged = dict(strict)
        forged.pop("state_digest")
        forged["stage"] = pair_v5.BOOTSTRAP_STAGE
        forged["strict_since_revision"] = None
        forged["revision"] = strict["revision"] + 1
        forged["last_event_digest"] = _sha("d")
        forged["state_digest"] = pair_v5.object_sha256(forged)
        # The forged state is internally self-consistent in isolation; the
        # transition validator still knows a strict predecessor can never
        # return to bootstrap.
        pair_v5.validate_state(forged, self.policy)
        with self.assertRaisesRegex(
            pair_v5.PairV5ContractError, "irreversible"
        ):
            pair_v5.validate_state_transition(strict, forged, self.policy)

    def test_flagged_high_score_does_not_trigger_strict(self) -> None:
        loser = _candidate("loser", 0.20, 0.90, 0.90, 0.90)
        contaminated = _candidate(
            "contaminated", 0.95, 0.95, 0.95, 0.95, camera=True
        )
        receipt = self.advance([loser, contaminated])
        self.assertEqual(receipt["stage_after"], pair_v5.BOOTSTRAP_STAGE)
        self.assertEqual(receipt["absolute_feasible_candidate_ids"], [])


class OrderAndReplayTests(PairV5Fixture):
    def test_candidate_order_does_not_change_any_receipt_byte(self) -> None:
        candidates = [
            _candidate("zulu", 0.52, 0.89, 0.88, 0.87),
            _candidate("alpha", 0.30, 0.92, 0.91, 0.90),
            _candidate("middle", 0.45, 0.90, 0.89, 0.88),
        ]
        forward = self.advance(candidates)
        reverse = self.advance(list(reversed(candidates)))
        rotated = self.advance(candidates[1:] + candidates[:1])
        self.assertEqual(
            pair_v5.canonical_json_bytes(forward),
            pair_v5.canonical_json_bytes(reverse),
        )
        self.assertEqual(
            pair_v5.canonical_json_bytes(forward),
            pair_v5.canonical_json_bytes(rotated),
        )

    def test_replay_binds_state_candidates_policy_and_calibrator(self) -> None:
        candidates = [
            _candidate("loser", 0.30, 0.92, 0.91, 0.90),
            _candidate("winner", 0.52, 0.89, 0.88, 0.87),
        ]
        receipt = self.advance(candidates)
        observed = pair_v5.replay_and_verify_receipt(
            receipt,
            state=self.initial,
            candidates=list(reversed(candidates)),
            policy=self.policy,
            calibrator_provenance=self.provenance,
        )
        self.assertEqual(observed["receipt_digest"], receipt["receipt_digest"])

        changed = [
            candidates[0],
            _candidate("winner", 0.53, 0.89, 0.88, 0.87),
        ]
        with self.assertRaisesRegex(
            pair_v5.PairV5ContractError, "does not replay exactly"
        ):
            pair_v5.replay_and_verify_receipt(
                receipt,
                state=self.initial,
                candidates=changed,
                policy=self.policy,
                calibrator_provenance=self.provenance,
            )


class TamperAndClosureTests(PairV5Fixture):
    def test_candidate_and_state_digest_tampering_fail(self) -> None:
        candidate = _candidate("candidate", 0.50, 0.90, 0.90, 0.90)
        candidate["action_score"] = 0.51
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "digest mismatch"):
            pair_v5.validate_candidate(candidate)

        state = deepcopy(self.initial)
        state["state_digest"] = "f" * 64
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "digest mismatch"):
            pair_v5.validate_state(state, self.policy)

    def test_inner_and_outer_receipt_tamper_fail(self) -> None:
        receipt = self.advance(
            [
                _candidate("loser", 0.30, 0.92, 0.91, 0.90),
                _candidate("winner", 0.52, 0.89, 0.88, 0.87),
            ]
        )
        tampered = deepcopy(receipt)
        tampered["selected_pair"]["action_improvement"] = 0.90
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "digest mismatch"):
            pair_v5.validate_selection_receipt(tampered, self.policy)

        tampered = deepcopy(receipt)
        tampered["candidate_count"] = 3
        with self.assertRaises(pair_v5.PairV5ContractError):
            pair_v5.validate_selection_receipt(tampered, self.policy)

    def test_even_fully_resealed_semantic_tamper_fails_replay(self) -> None:
        candidates = [
            _candidate("loser", 0.30, 0.92, 0.91, 0.90),
            _candidate("winner", 0.52, 0.89, 0.88, 0.87),
        ]
        receipt = self.advance(candidates)
        tampered = deepcopy(receipt)
        pair = tampered["selected_pair"]
        pair.pop("pair_digest")
        pair["action_improvement"] = 0.21
        pair["pair_digest"] = pair_v5.object_sha256(pair)
        tampered["event_digest"] = pair_v5.object_sha256(
            {
                "schema_version": pair_v5.EVENT_SCHEMA,
                "policy_digest": tampered["policy_digest"],
                "calibrator_provenance_digest": tampered[
                    "calibrator_provenance_digest"
                ],
                "state_before_digest": tampered["state_before_digest"],
                "candidate_set_digest": tampered["candidate_set_digest"],
                "stage_before": tampered["stage_before"],
                "stage_after": tampered["stage_after"],
                "transitioned_to_strict": tampered["transitioned_to_strict"],
                "absolute_feasible_candidate_ids": tampered[
                    "absolute_feasible_candidate_ids"
                ],
                "selected_pair_digest": pair["pair_digest"],
                "eligible_pair_count": tampered["eligible_pair_count"],
            }
        )
        tampered["next_state"].pop("state_digest")
        tampered["next_state"]["last_event_digest"] = tampered["event_digest"]
        tampered["next_state"]["state_digest"] = pair_v5.object_sha256(
            tampered["next_state"]
        )
        tampered.pop("receipt_digest")
        tampered["receipt_digest"] = pair_v5.object_sha256(tampered)
        # Standalone form is well-digested but replay reconstructs the actual
        # score delta and rejects the semantic rewrite.
        pair_v5.validate_selection_receipt(tampered, self.policy)
        with self.assertRaisesRegex(
            pair_v5.PairV5ContractError, "does not replay exactly"
        ):
            pair_v5.replay_and_verify_receipt(
                tampered,
                state=self.initial,
                candidates=candidates,
                policy=self.policy,
                calibrator_provenance=self.provenance,
            )

    def test_candidate_policy_flags_and_provenance_are_closed(self) -> None:
        candidate = _candidate("candidate", 0.50, 0.90, 0.90, 0.90)
        leaked = deepcopy(candidate)
        leaked["donor_latent"] = _sha("d")
        leaked["candidate_digest"] = pair_v5.object_sha256(
            {key: value for key, value in leaked.items() if key != "candidate_digest"}
        )
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "extra=.*donor_latent"):
            pair_v5.validate_candidate(leaked)

        missing_flag = deepcopy(candidate)
        missing_flag["hard_negative_flags"].pop("reverse")
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "missing=.*reverse"):
            pair_v5.validate_candidate(missing_flag)

        extra_flag = deepcopy(candidate)
        extra_flag["hard_negative_flags"]["custom_mask_gate"] = False
        with self.assertRaisesRegex(
            pair_v5.PairV5ContractError, "extra=.*custom_mask_gate"
        ):
            pair_v5.validate_candidate(extra_flag)

        leaked_provenance = deepcopy(self.provenance)
        leaked_provenance["proposal_video_sha256"] = _sha("e")
        leaked_provenance["provenance_digest"] = pair_v5.object_sha256(
            {
                key: value
                for key, value in leaked_provenance.items()
                if key != "provenance_digest"
            }
        )
        with self.assertRaisesRegex(
            pair_v5.PairV5ContractError, "extra=.*proposal_video_sha256"
        ):
            pair_v5.validate_calibrator_provenance(leaked_provenance)

        leaked_policy = deepcopy(self.policy)
        leaked_policy["target_video"] = "/forbidden/target.mp4"
        leaked_policy["policy_digest"] = pair_v5.object_sha256(
            {
                key: value
                for key, value in leaked_policy.items()
                if key != "policy_digest"
            }
        )
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "extra=.*target_video"):
            pair_v5.validate_policy(leaked_policy)

    def test_main_api_has_no_media_tensor_or_privileged_input_slot(self) -> None:
        parameters = set(
            inspect.signature(pair_v5.advance_pair_selector).parameters
        )
        self.assertEqual(
            parameters,
            {"state", "candidates", "policy", "calibrator_provenance"},
        )
        serialized_provenance = pair_v5.canonical_json_bytes(
            self.provenance
        ).decode("utf-8")
        for forbidden in (
            "video",
            "latent",
            "noise",
            "donor",
            "source",
            "target",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ):
            self.assertNotIn(forbidden, serialized_provenance)

        receipt = self.advance(
            [
                _candidate("loser", 0.30, 0.92, 0.91, 0.90),
                _candidate("winner", 0.52, 0.89, 0.88, 0.87),
            ]
        )
        closure = receipt["condition_closure"]
        self.assertTrue(closure["selector_consumes_scores_only"])
        for key, value in closure.items():
            if key.endswith("_consumed"):
                self.assertFalse(value, key)

    def test_noncanonical_duplicate_and_nonfinite_json_fail(self) -> None:
        canonical = pair_v5.canonical_json_bytes({"a": 1, "b": 2})
        self.assertEqual(pair_v5.parse_canonical_json_bytes(canonical), {"a": 1, "b": 2})
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "canonical form"):
            pair_v5.parse_canonical_json_bytes(b'{"b":2,"a":1}')
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "duplicate JSON key"):
            pair_v5.parse_canonical_json_bytes(b'{"a":1,"a":2}')
        with self.assertRaisesRegex(pair_v5.PairV5ContractError, "non-finite"):
            pair_v5.parse_canonical_json_bytes(b'{"a":NaN}')


if __name__ == "__main__":
    unittest.main()
