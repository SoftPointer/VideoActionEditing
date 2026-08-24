from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "saic_event_reward_v1.py"
SPEC = importlib.util.spec_from_file_location("saic_event_reward_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reward = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reward
SPEC.loader.exec_module(reward)


class SAICEventRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.checkpoint = self.root / "critic.safetensors"
        self.checkpoint.write_bytes(b"frozen-saic-four-stage-critic-v1\x00weights")
        self.checkpoint_sha = __import__("hashlib").sha256(self.checkpoint.read_bytes()).hexdigest()
        self.qualification_path = self.root / "qualification.json"
        self.qualification = self._qualification()
        self._write_qualification(self.qualification)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _qualification(self) -> dict:
        phase_margins = {phase: 0.30 for phase in reward.PHASE_ORDER}
        cells = []
        for dimension in reward.HOLDOUT_ORDER:
            for negative in reward.NEGATIVE_ORDER:
                cells.append(
                    {
                        "holdout_dimension": dimension,
                        "negative_kind": negative,
                        "sample_count": 3,
                        "stage_margins": dict(phase_margins),
                        "weakest_margin": 0.30,
                        "passed": True,
                    }
                )
        unsigned = {
            "schema_version": reward.QUALIFICATION_SCHEMA_VERSION,
            "critic_checkpoint": {
                "content_sha256": self.checkpoint_sha,
                "byte_size": self.checkpoint.stat().st_size,
                "state_dict_kind": "critic_only_no_optimizer",
                "eval_mode": True,
                "requires_grad_parameter_count": 0,
                "optimizer_state_present": False,
            },
            "phase_order": list(reward.PHASE_ORDER),
            "negative_order": list(reward.NEGATIVE_ORDER),
            "holdout_order": list(reward.HOLDOUT_ORDER),
            "holdout_summary": {
                dimension: {
                    "held_out_unit_count": 2,
                    "fit_overlap_count": 0,
                    "passed": True,
                }
                for dimension in reward.HOLDOUT_ORDER
            },
            "coverage_cells": cells,
            "thresholds": {
                "qualification_margin_floor": 0.20,
                "bootstrap_relative_margin_floor": 0.10,
                "absolute_action_score_floors": {
                    "onset": 0.70,
                    "transition": 0.75,
                    "completion": 0.80,
                    "hold": 0.80,
                },
                "absolute_margin_floors": {
                    "onset": 0.20,
                    "transition": 0.20,
                    "completion": 0.25,
                    "hold": 0.25,
                },
            },
            "authority_contract": {
                "score_only_runtime_boundary": True,
                "receipt_alone_authorizes_optimizer": False,
                "receipt_alone_authorizes_inverse": False,
                "receipt_alone_authorizes_publication": False,
                "bootstrap_scope": "same_round_relative_pairing_only",
                "absolute_four_stage_pass_required_for_inverse": True,
                "absolute_four_stage_pass_required_for_publication": True,
                "external_source_constraints_still_required": True,
            },
        }
        return {**unsigned, "receipt_digest": reward.object_sha256(unsigned)}

    def _write_qualification(self, value: dict) -> None:
        self.qualification_path.write_bytes(reward.canonical_json_bytes(value) + b"\n")

    def _reseal_qualification(self, value: dict) -> dict:
        unsigned = copy.deepcopy(value)
        unsigned.pop("receipt_digest", None)
        return {**unsigned, "receipt_digest": reward.object_sha256(unsigned)}

    def _candidate(self, *, action: dict[str, float] | None = None) -> dict:
        action_scores = action or {
            "onset": 0.90,
            "transition": 0.91,
            "completion": 0.92,
            "hold": 0.93,
        }
        scores = {"action": action_scores}
        offsets = {
            "reverse": 0.55,
            "incomplete": 0.60,
            "camera_only": 0.30,
            "appearance_only": 0.20,
        }
        for negative, score in offsets.items():
            scores[negative] = {phase: score for phase in reward.PHASE_ORDER}
        unsigned = {
            "schema_version": reward.CANDIDATE_SCORE_SCHEMA_VERSION,
            "candidate_id": "candidate-0001",
            "rollout_id": "rollout-0001",
            "action_family": "dog-sit-hold",
            "policy_checkpoint_sha256": "1" * 64,
            "critic_checkpoint_sha256": self.checkpoint_sha,
            "qualification_receipt_digest": self.qualification["receipt_digest"],
            "rollout_contract": {
                "on_policy": True,
                "fresh_after_latest_update": True,
                "source_coordinate": "current_policy_rv2v",
                "decoded_exact81": True,
                "frame_count": 81,
                "scores_computed_by_frozen_critic": True,
                "event_bank_candidate": False,
                "payload_kind": "scalar_stage_scores_only",
                "media_or_path_attached": False,
                "latent_attached": False,
                "noise_attached": False,
                "target_attached": False,
                "proposal_attached": False,
            },
            "phase_order": list(reward.PHASE_ORDER),
            "negative_order": list(reward.NEGATIVE_ORDER),
            "scores": scores,
        }
        return {**unsigned, "score_packet_digest": reward.object_sha256(unsigned)}

    @staticmethod
    def _reseal_candidate(value: dict) -> dict:
        unsigned = copy.deepcopy(value)
        unsigned.pop("score_packet_digest", None)
        return {**unsigned, "score_packet_digest": reward.object_sha256(unsigned)}

    def test_bootstrap_only_makes_candidate_eligible_for_external_relative_pairing(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        decision = boundary.evaluate(self._candidate(), mode="bootstrap")
        self.assertAlmostEqual(decision["weakest_margin"], 0.30)
        self.assertEqual(decision["weakest_phase"], "onset")
        self.assertTrue(decision["relative_action_margin_pass"])
        self.assertTrue(decision["relative_pairing_eligible"])
        self.assertTrue(decision["absolute_four_stage_pass"])
        authority = decision["authority"]
        self.assertFalse(authority["optimizer_update_authorized"])
        self.assertEqual(authority["optimizer_scope"], "none_single_candidate_consumer")
        self.assertTrue(authority["same_round_y_plus_y_minus_pair_builder_required"])
        self.assertTrue(authority["seven_noncompensating_axes_required_for_optimizer"])
        self.assertFalse(authority["inverse_cycle_entry_authorized"])
        self.assertFalse(authority["event_side_checkpoint_publication_authorized"])
        self.assertFalse(authority["global_checkpoint_publication_authorized"])
        unsigned = dict(decision)
        digest = unsigned.pop("decision_digest")
        self.assertEqual(digest, reward.object_sha256(unsigned))

    def test_strict_absolute_pass_is_required_for_inverse_and_event_publication(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        decision = boundary.evaluate(self._candidate(), mode="strict")
        self.assertTrue(decision["absolute_four_stage_pass"])
        authority = decision["authority"]
        self.assertFalse(authority["optimizer_update_authorized"])
        self.assertEqual(authority["optimizer_scope"], "none_single_candidate_consumer")
        self.assertTrue(authority["inverse_cycle_entry_authorized"])
        self.assertTrue(authority["event_side_checkpoint_publication_authorized"])
        # This module cannot compensate for the separate source/identity gates.
        self.assertFalse(authority["global_checkpoint_publication_authorized"])
        self.assertTrue(authority["external_source_constraints_still_required"])

    def test_bootstrap_relative_pass_does_not_spoof_absolute_pass(self) -> None:
        action = {phase: 0.60 for phase in reward.PHASE_ORDER}
        candidate = self._candidate(action=action)
        for negative in reward.NEGATIVE_ORDER:
            candidate["scores"][negative] = {phase: 0.20 for phase in reward.PHASE_ORDER}
        candidate = self._reseal_candidate(candidate)
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        bootstrap = boundary.evaluate(candidate, mode="bootstrap")
        strict = boundary.evaluate(candidate, mode="strict")
        self.assertTrue(bootstrap["relative_action_margin_pass"])
        self.assertFalse(bootstrap["absolute_four_stage_pass"])
        self.assertTrue(bootstrap["relative_pairing_eligible"])
        self.assertFalse(bootstrap["authority"]["optimizer_update_authorized"])
        self.assertFalse(bootstrap["authority"]["inverse_cycle_entry_authorized"])
        self.assertFalse(strict["authority"]["optimizer_update_authorized"])
        self.assertFalse(strict["authority"]["inverse_cycle_entry_authorized"])
        self.assertFalse(strict["authority"]["event_side_checkpoint_publication_authorized"])

    def test_reward_uses_weakest_stage_against_strongest_negative_not_mean(self) -> None:
        candidate = self._candidate()
        candidate["scores"]["camera_only"]["completion"] = 0.89
        candidate = self._reseal_candidate(candidate)
        decision = reward.load_event_reward_boundary(
            self.checkpoint, self.qualification_path
        ).evaluate(candidate, mode="bootstrap")
        self.assertAlmostEqual(decision["stage_margins"]["completion"], 0.03)
        self.assertEqual(decision["strongest_negative_kinds"]["completion"], "camera_only")
        self.assertEqual(decision["weakest_phase"], "completion")
        self.assertAlmostEqual(decision["weakest_margin"], 0.03)
        self.assertFalse(decision["relative_action_margin_pass"])
        self.assertFalse(decision["relative_pairing_eligible"])
        self.assertFalse(decision["authority"]["optimizer_update_authorized"])

    def test_each_holdout_and_negative_cross_cell_is_mandatory(self) -> None:
        for dimension in reward.HOLDOUT_ORDER:
            damaged = copy.deepcopy(self.qualification)
            damaged["coverage_cells"] = [
                row for row in damaged["coverage_cells"] if row["holdout_dimension"] != dimension
            ]
            damaged = self._reseal_qualification(damaged)
            self._write_qualification(damaged)
            with self.subTest(missing_holdout=dimension), self.assertRaises(reward.SAICEventRewardError):
                reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        for negative in reward.NEGATIVE_ORDER:
            damaged = copy.deepcopy(self.qualification)
            damaged["coverage_cells"] = [
                row for row in damaged["coverage_cells"] if row["negative_kind"] != negative
            ]
            damaged = self._reseal_qualification(damaged)
            self._write_qualification(damaged)
            with self.subTest(missing_negative=negative), self.assertRaises(reward.SAICEventRewardError):
                reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)

    def test_failed_or_overlapping_qualification_never_loads(self) -> None:
        mutations = []
        failed_cell = copy.deepcopy(self.qualification)
        failed_cell["coverage_cells"][0]["stage_margins"]["hold"] = 0.01
        failed_cell["coverage_cells"][0]["weakest_margin"] = 0.01
        failed_cell["coverage_cells"][0]["passed"] = False
        mutations.append(failed_cell)
        overlap = copy.deepcopy(self.qualification)
        overlap["holdout_summary"]["action_family"]["fit_overlap_count"] = 1
        mutations.append(overlap)
        unfrozen = copy.deepcopy(self.qualification)
        unfrozen["critic_checkpoint"]["requires_grad_parameter_count"] = 1
        mutations.append(unfrozen)
        self_authorizing = copy.deepcopy(self.qualification)
        self_authorizing["authority_contract"]["receipt_alone_authorizes_publication"] = True
        mutations.append(self_authorizing)
        for index, damaged in enumerate(mutations):
            self._write_qualification(self._reseal_qualification(damaged))
            with self.subTest(case=index), self.assertRaises(reward.SAICEventRewardError):
                reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)

    def test_receipt_and_candidate_digests_are_not_advisory(self) -> None:
        damaged_receipt = copy.deepcopy(self.qualification)
        damaged_receipt["thresholds"]["bootstrap_relative_margin_floor"] = 0.0
        self._write_qualification(damaged_receipt)
        with self.assertRaises(reward.SAICEventRewardError):
            reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        self._write_qualification(self.qualification)
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        damaged_candidate = self._candidate()
        damaged_candidate["scores"]["action"]["hold"] = 999.0
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(damaged_candidate, mode="strict")

    def test_checkpoint_and_receipt_are_revalidated_before_every_decision(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        self.checkpoint.write_bytes(self.checkpoint.read_bytes() + b"tampered")
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(self._candidate(), mode="strict")

        # Restore both inputs, load again, then mutate the receipt with valid JSON.
        self.checkpoint.write_bytes(b"frozen-saic-four-stage-critic-v1\x00weights")
        self._write_qualification(self.qualification)
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        self.qualification_path.write_bytes(self.qualification_path.read_bytes() + b" ")
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(self._candidate(), mode="strict")

    def test_public_snapshot_mutation_cannot_lower_revalidated_thresholds(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        for phase in reward.PHASE_ORDER:
            boundary.qualification["thresholds"]["absolute_action_score_floors"][phase] = -999.0
            boundary.qualification["thresholds"]["absolute_margin_floors"][phase] = -999.0
        action = {phase: 0.60 for phase in reward.PHASE_ORDER}
        candidate = self._candidate(action=action)
        for negative in reward.NEGATIVE_ORDER:
            candidate["scores"][negative] = {phase: 0.20 for phase in reward.PHASE_ORDER}
        candidate = self._reseal_candidate(candidate)
        decision = boundary.evaluate(candidate, mode="strict")
        self.assertFalse(decision["absolute_four_stage_pass"])
        self.assertFalse(decision["authority"]["inverse_cycle_entry_authorized"])
        self.assertFalse(
            decision["authority"]["event_side_checkpoint_publication_authorized"]
        )

    def test_score_packet_rejects_every_forbidden_media_or_teacher_channel(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        forbidden_top_level = (
            "event_bank_video_path",
            "video_path",
            "media",
            "frames",
            "latent",
            "noise",
            "target",
            "proposal",
            "motion_donor",
        )
        for field in forbidden_top_level:
            candidate = self._candidate()
            candidate[field] = "/forbidden/event-bank/candidate.mp4"
            candidate = self._reseal_candidate(candidate)
            with self.subTest(field=field), self.assertRaises(reward.SAICEventRewardError):
                boundary.evaluate(candidate, mode="bootstrap")

        contract_flags = (
            "media_or_path_attached",
            "latent_attached",
            "noise_attached",
            "target_attached",
            "proposal_attached",
        )
        for field in contract_flags:
            candidate = self._candidate()
            candidate["rollout_contract"][field] = True
            candidate = self._reseal_candidate(candidate)
            with self.subTest(contract=field), self.assertRaises(reward.SAICEventRewardError):
                boundary.evaluate(candidate, mode="bootstrap")

    def test_only_fresh_current_on_policy_exact81_scores_are_accepted(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        mutations = (
            ("on_policy", False),
            ("fresh_after_latest_update", False),
            ("source_coordinate", "pure_t2v_event_bank"),
            ("decoded_exact81", False),
            ("frame_count", 41),
            ("scores_computed_by_frozen_critic", False),
            ("event_bank_candidate", True),
            ("payload_kind", "latent_and_scores"),
        )
        for field, value in mutations:
            candidate = self._candidate()
            candidate["rollout_contract"][field] = value
            candidate = self._reseal_candidate(candidate)
            with self.subTest(field=field), self.assertRaises(reward.SAICEventRewardError):
                boundary.evaluate(candidate, mode="bootstrap")
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(self.root / "event-bank-candidate.json", mode="bootstrap")

    def test_phase_and_negative_sets_are_exact_and_nan_is_rejected(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        missing_negative = self._candidate()
        del missing_negative["scores"]["appearance_only"]
        missing_negative = self._reseal_candidate(missing_negative)
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(missing_negative, mode="bootstrap")
        extra_phase = self._candidate()
        extra_phase["scores"]["action"]["average"] = 1.0
        extra_phase = self._reseal_candidate(extra_phase)
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(extra_phase, mode="bootstrap")
        nan_score = self._candidate()
        nan_score["scores"]["action"]["hold"] = float("nan")
        # It cannot even be digest-sealed as canonical JSON.
        with self.assertRaises(reward.SAICEventRewardError):
            self._reseal_candidate(nan_score)

    def test_wrong_critic_or_qualification_binding_and_unknown_mode_fail(self) -> None:
        boundary = reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)
        wrong_critic = self._candidate()
        wrong_critic["critic_checkpoint_sha256"] = "2" * 64
        wrong_critic = self._reseal_candidate(wrong_critic)
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(wrong_critic, mode="strict")
        wrong_receipt = self._candidate()
        wrong_receipt["qualification_receipt_digest"] = "3" * 64
        wrong_receipt = self._reseal_candidate(wrong_receipt)
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(wrong_receipt, mode="strict")
        with self.assertRaises(reward.SAICEventRewardError):
            boundary.evaluate(self._candidate(), mode="auto")

    def test_symlinks_and_duplicate_json_keys_are_rejected(self) -> None:
        checkpoint_link = self.root / "critic-link.safetensors"
        checkpoint_link.symlink_to(self.checkpoint)
        with self.assertRaises(reward.SAICEventRewardError):
            reward.load_event_reward_boundary(checkpoint_link, self.qualification_path)
        receipt_link = self.root / "qualification-link.json"
        receipt_link.symlink_to(self.qualification_path)
        with self.assertRaises(reward.SAICEventRewardError):
            reward.load_event_reward_boundary(self.checkpoint, receipt_link)
        duplicate = reward.canonical_json_bytes(self.qualification).decode("ascii")
        duplicate = duplicate[:-1] + ',"receipt_digest":"' + self.qualification["receipt_digest"] + '"}'
        self.qualification_path.write_text(duplicate, encoding="utf-8")
        with self.assertRaises(reward.SAICEventRewardError):
            reward.load_event_reward_boundary(self.checkpoint, self.qualification_path)


if __name__ == "__main__":
    unittest.main()
