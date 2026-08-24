from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import latent_temporal_event_critic_dataset as dataset  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def usage(bank: str = "bank") -> dict:
    return dataset.make_critic_usage_authority(
        bank_receipt_digest=sha(bank),
        authorization_source="archived_user_research_direction_20260808",
        authorization_evidence_sha256=sha("authorization"),
    )


def candidate_rows(
    *,
    cell: str = "cell-a",
    split: str = "fit",
    family: str = "family-a",
    actor: str = "actor-a",
    scene: str = "scene-a",
    action_group: str = "action-a",
    seed: int = 101,
    bank: str = "bank",
) -> list[dict]:
    rows = []
    for branch in dataset.SEMANTIC_BRANCHES:
        caption = (
            f"A complete static-camera scene description for {cell} shows the main actor "
            f"performing the registered {branch} branch with a continuous coherent ending."
        )
        positive = branch == "action"
        rows.append(
            {
                "candidate_id": f"candidate-{cell}-{branch}",
                "bank_receipt_digest": sha(bank),
                "cell_id": cell,
                "analysis_split": split,
                "action_family_id": family,
                "actor_group_id": actor,
                "scene_group_id": scene,
                "action_group_id": action_group,
                "seed": seed,
                "official_gaussian_tensor_sha256": sha(f"gaussian-{cell}"),
                "semantic_branch": branch,
                "full_t2v_caption": caption,
                "full_t2v_caption_utf8_sha256": sha(caption),
                "clean_latent_artifact_path": f"/tmp/{cell}-{branch}.safetensors",
                "clean_latent_artifact_sha256": sha(f"artifact-{cell}-{branch}"),
                "clean_latent_tensor_sha256": sha(f"tensor-{cell}-{branch}"),
                "clean_latent_shape": [1, 16, 21, 60, 62],
                "generation_receipt_digest": sha(f"generation-{cell}-{branch}"),
                "event_audit_artifact_sha256": sha("audit"),
                "complete_target_transition_observed": positive,
                "terminal_hold_observed": positive,
                "full_target_action_observed": positive,
                "full_target_action_false_confirmed": not positive,
            }
        )
    return rows


def make_episode(**overrides) -> dict:
    return dataset.build_episode_plan(
        candidate_rows(**overrides), usage_authority=usage()
    )


class EpisodeClosureTests(unittest.TestCase):
    def test_event_qualified_episode_has_exact_hidden_query_arms(self) -> None:
        episode = make_episode()
        self.assertEqual(episode["arm_order"], list(dataset.ARM_ROLES))
        self.assertFalse(episode["generated_media_editor_use_authorized"])
        self.assertEqual(
            episode["hidden_query_contract"]["hook_coordinate"],
            "block.15.output",
        )
        self.assertEqual(episode["native_geometry"]["patch_grid_height_width"], [30, 31])
        self.assertEqual(episode["native_geometry"]["patch_positions"], 930)
        self.assertTrue(
            episode["hidden_query_contract"]["patch_positions_are_episode_specific"]
        )
        arms = {row["role"]: row for row in episode["arms"]}
        self.assertEqual(
            arms["same_video_reverse"]["source_candidate_id"],
            arms["positive"]["source_candidate_id"],
        )
        self.assertEqual(
            arms["same_video_reverse"]["temporal_transform"], "reverse"
        )
        self.assertEqual(
            arms["same_video_freeze_first"]["temporal_transform"],
            "freeze_first",
        )
        self.assertEqual(
            arms["semantic_camera_only"]["source_semantic_branch"],
            "camera_only",
        )
        self.assertTrue(
            all(row["same_state_x_sigma_proof_required"] for row in episode["arms"])
        )

    def test_native_geometry_is_derived_without_resize_or_orientation_swap(self) -> None:
        expected = {
            (60, 62): ([30, 31], 930),
            (64, 58): ([32, 29], 928),
            (68, 54): ([34, 27], 918),
        }
        for (height, width), (grid, positions) in expected.items():
            with self.subTest(shape=(height, width)):
                binding = dataset.derive_native_geometry(
                    [1, 16, 21, height, width]
                )
                self.assertEqual(binding["patch_grid_height_width"], grid)
                self.assertEqual(binding["patch_positions"], positions)
                self.assertFalse(binding["resize_or_crop_applied"])
        with self.assertRaisesRegex(
            dataset.LatentTemporalEventDatasetError, "must divide"
        ):
            dataset.derive_native_geometry([1, 16, 21, 61, 62])

    def test_positive_must_have_transition_and_terminal_hold(self) -> None:
        rows = candidate_rows()
        rows[0]["terminal_hold_observed"] = False
        with self.assertRaisesRegex(
            dataset.LatentTemporalEventDatasetError, "event-qualified"
        ):
            dataset.build_episode_plan(rows, usage_authority=usage())

    def test_ambiguous_or_successful_semantic_negative_fails_closed(self) -> None:
        rows = candidate_rows()
        noop = rows[1]
        noop["full_target_action_false_confirmed"] = False
        with self.assertRaisesRegex(
            dataset.LatentTemporalEventDatasetError, "ambiguous"
        ):
            dataset.build_episode_plan(rows, usage_authority=usage())

    def test_usage_authority_cannot_enable_an_editor_target(self) -> None:
        authority = usage()
        authority["authorized_use"]["generated_rgb_or_latent_may_be_editor_target"] = True
        unsigned = dict(authority)
        unsigned.pop("receipt_digest")
        authority["receipt_digest"] = dataset.object_sha256(unsigned)
        with self.assertRaisesRegex(
            dataset.LatentTemporalEventDatasetError, "critic-only"
        ):
            dataset.validate_critic_usage_authority(authority)


class PopulationAndPilotTests(unittest.TestCase):
    def pilot_episodes(self):
        episodes = []
        for split, suffix, offset in (
            ("fit", "fit", 0),
            ("confirmation", "confirm", 10),
        ):
            for family_index, family in enumerate(("dog-sit", "human-rise")):
                episodes.append(
                    make_episode(
                        cell=f"cell-{suffix}-{family_index}",
                        split=split,
                        family=family,
                        actor=f"actor-{suffix}-{family_index}",
                        scene=f"scene-{suffix}-{family_index}",
                        action_group=f"action-{suffix}-{family_index}",
                        seed=101 + offset + family_index,
                    )
                )
        return episodes

    def test_core4_population_only_authorizes_topup_pilot(self) -> None:
        audit = dataset.audit_episode_population(
            self.pilot_episodes(), protocol="core4_pilot"
        )
        self.assertTrue(audit["population_eligible"])
        self.assertTrue(audit["critic_head_pilot_training_authorized"])
        self.assertTrue(audit["worth_topup_evaluation_authorized"])
        self.assertFalse(audit["scientific_critic_claim_authorized"])
        self.assertFalse(audit["editor_optimizer_authorized"])

    def test_seed_or_identity_leak_between_pilot_splits_is_rejected(self) -> None:
        episodes = self.pilot_episodes()
        mutated = deepcopy(episodes[-1])
        unsigned = dict(mutated)
        unsigned.pop("receipt_digest")
        unsigned["actor_group_id"] = episodes[0]["actor_group_id"]
        mutated = {**unsigned, "receipt_digest": dataset.object_sha256(unsigned)}
        episodes[-1] = mutated
        audit = dataset.audit_episode_population(episodes, protocol="core4_pilot")
        self.assertFalse(audit["population_eligible"])
        self.assertIn(
            "pilot_split_overlap:actor_group_id", audit["failure_reasons"]
        )

    def test_pilot_gate_is_conjunctive_and_never_authorizes_editor(self) -> None:
        episode_ids = ("confirm-dog", "confirm-human")
        scores = {}
        for episode_id in episode_ids:
            scores[episode_id] = {"positive": 1.0}
            scores[episode_id].update(
                {role: 0.0 for role in dataset.NEGATIVE_ROLES}
            )
        passed = dataset.evaluate_core4_pilot_gate(
            scores,
            expected_confirmation_episode_ids=episode_ids,
            input_gradient_audit_passed=True,
        )
        self.assertTrue(passed["worth_fixed_topup_generation"])
        self.assertFalse(passed["editor_optimizer_authorized"])

        scores["confirm-human"]["semantic_wrong_actor"] = 0.95
        failed = dataset.evaluate_core4_pilot_gate(
            scores,
            expected_confirmation_episode_ids=episode_ids,
            input_gradient_audit_passed=True,
        )
        self.assertFalse(failed["worth_fixed_topup_generation"])
        self.assertIn(
            "margin:confirm-human:semantic_wrong_actor",
            failed["failure_reasons"],
        )

    def test_existing_local_core4_is_inventory_only_not_train_authority(self) -> None:
        path = METHOD_ROOT / "assets" / "pair_v5_t2v_calibration_core4_bank_v2.json"
        root = json.loads(path.read_text(encoding="utf-8"))
        audit = dataset.audit_core4_spec_inventory(root)
        self.assertEqual(audit["candidate_count"], 40)
        self.assertEqual(audit["cell_count"], 4)
        self.assertEqual(audit["action_family_count"], 2)
        self.assertEqual(audit["registered_seed_count"], 4)
        self.assertTrue(audit["old_artifact_scope_is_calibration_only"])
        self.assertTrue(audit["core4_geometry_can_support_fit_confirmation_pilot"])
        self.assertFalse(audit["scientific_three_way_family_holdout_possible"])
        self.assertFalse(audit["current_critic_training_authorized"])
        self.assertFalse(audit["editor_optimizer_authorized"])
        self.assertIn(
            "same_state_action_noop_hidden_pairs_not_materialized",
            audit["failure_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
