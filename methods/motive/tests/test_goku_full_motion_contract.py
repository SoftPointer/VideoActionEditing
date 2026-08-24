from __future__ import annotations

import copy
import unittest

from motive import goku_full_motion_contract as contract


def _evidence(description: str, start: int = 0, end: int = 80) -> dict:
    return {
        "schema_version": contract.MOTION_EVIDENCE_SCHEMA,
        "start_frame": start,
        "end_frame": end,
        "description": description,
    }


def source_census(*, static_person: bool = False) -> dict:
    registry = [
        {
            "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
            "entity_id": "entity_01",
            "entity_type": "person",
            "stable_reference": "the person on the left",
            "i0_bbox_xyxy_1000": [50, 350, 300, 900],
            "viewer_region": "center_left",
            "region_ordinal": 1,
            "role": "dynamic_subject",
            "visible_at_i0": True,
            "reachable_at_i0": True,
            "confidence": "high",
        },
        {
            "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
            "entity_id": "entity_02",
            "entity_type": "person",
            "stable_reference": "the person on the right",
            "i0_bbox_xyxy_1000": [700, 350, 950, 900],
            "viewer_region": "center_right",
            "region_ordinal": 1,
            "role": "dynamic_subject",
            "visible_at_i0": True,
            "reachable_at_i0": True,
            "confidence": "high",
        },
    ]
    static = []
    if static_person:
        registry.append(
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_03",
                "entity_type": "person",
                "stable_reference": "the seated bystander in the center",
                "i0_bbox_xyxy_1000": [400, 350, 600, 900],
                "viewer_region": "center",
                "region_ordinal": 1,
                "role": "static_salient",
                "visible_at_i0": True,
                "reachable_at_i0": False,
                "confidence": "high",
            }
        )
        static.append(
            {
                "schema_version": contract.SOURCE_STATIC_PERSON_SCHEMA,
                "unit_id": "static_person_01",
                "entity_id": "entity_03",
                "entity_type": "person",
                "stable_reference": "the seated bystander in the center",
                "visible_at_i0": True,
                "i0_state": "A seated bystander faces the two standing people",
                "source_state": "remain_still",
                "motion_evidence": [
                    _evidence("the seated bystander remains in the same pose")
                ],
                "confidence": "high",
            }
        )
    return {
        "schema_version": contract.SOURCE_CENSUS_SCHEMA,
        "iid": "two-people-wave-001",
        "clip": {
            "schema_version": contract.CLIP_SCHEMA,
            "frame_count": 81,
            "fps": "25/1",
            "timeline_span_seconds": 3.2,
            "single_continuous_shot": True,
        },
        "source_quality": "high",
        "scene_description": "Two standing people are visible against a plain wall",
        "i0_visible_entities": [
            "the person on the left",
            "the person on the right",
            *[row["stable_reference"] for row in static],
        ],
        "i0_entity_registry": registry,
        "motion_inventory_complete": True,
        "crowd_or_unresolved_motion": False,
        "diffuse_unresolved_motion": False,
        "dynamic_units": [
            {
                "schema_version": contract.SOURCE_DYNAMIC_UNIT_SCHEMA,
                "unit_id": "unit_01",
                "entity_id": "entity_01",
                "entity_type": "person",
                "stable_reference": "the person on the left",
                "visible_at_i0": True,
                "independent_motion": True,
                "i0_state": "The left person holds a peace sign near the face",
                "source_action_signature": "raise_peace_sign_left",
                "source_motion": "raises two fingers in a peace-sign gesture",
                "source_motion_components": [
                    {
                        "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                        "component_id": "component_01",
                        "component_type": "gesture",
                        "motion_signature": "raise_peace_sign_left",
                        "motion_description": (
                            "raises two fingers in a peace-sign gesture"
                        ),
                        "dependent_entity_ids": [],
                        "motion_evidence": [
                            _evidence(
                                "the left person's raised hand changes position",
                                0,
                                60,
                            )
                        ],
                    }
                ],
                "motion_evidence": [
                    _evidence(
                        "the left person's raised hand changes position",
                        0,
                        60,
                    )
                ],
                "confidence": "high",
            },
            {
                "schema_version": contract.SOURCE_DYNAMIC_UNIT_SCHEMA,
                "unit_id": "unit_02",
                "entity_id": "entity_02",
                "entity_type": "person",
                "stable_reference": "the person on the right",
                "visible_at_i0": True,
                "independent_motion": True,
                "i0_state": "The right person holds a peace sign near the shoulder",
                "source_action_signature": "raise_peace_sign_right",
                "source_motion": "moves a two-finger peace-sign gesture",
                "source_motion_components": [
                    {
                        "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                        "component_id": "component_01",
                        "component_type": "gesture",
                        "motion_signature": "raise_peace_sign_right",
                        "motion_description": (
                            "moves a two-finger peace-sign gesture"
                        ),
                        "dependent_entity_ids": [],
                        "motion_evidence": [
                            _evidence(
                                "the right person's raised hand changes position",
                                0,
                                64,
                            )
                        ],
                    }
                ],
                "motion_evidence": [
                    _evidence(
                        "the right person's raised hand changes position",
                        0,
                        64,
                    )
                ],
                "confidence": "high",
            },
        ],
        "static_salient_people": static,
        "camera": {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "locked_off",
            "motion_signature": "locked_off",
            "motion_description": "locked off",
            "dynamic": False,
            "motion_evidence": [
                _evidence("the background remains aligned across all frames")
            ],
            "confidence": "high",
        },
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _target_unit(
    unit_id: str,
    entity_id: str,
    reference: str,
    side: str,
) -> dict:
    novel = "lower the peace sign and wave with an open palm"
    return {
        "schema_version": contract.TARGET_DYNAMIC_UNIT_SCHEMA,
        "unit_id": unit_id,
        "entity_id": entity_id,
        "stable_reference": reference,
        "target_action_signature": f"open_palm_wave_{side}",
        "motion_relation": "replace",
        "source_motion_suppressed": True,
        "explicit_shared_base_motion": None,
        "source_component_dispositions": [
            {
                "schema_version": contract.TARGET_COMPONENT_DISPOSITION_SCHEMA,
                "component_id": "component_01",
                "disposition": "suppress",
                "explicit_target_motion": None,
            }
        ],
        "novel_target_motion": novel,
        "target_clause": f"have {reference} {novel}",
        "substantive_change": True,
        "starts_at_i0": True,
        "i0_executable": True,
        "complete_within_clip": True,
        "completion_time_seconds": 3.0,
        "ordered_stages": [
            f"{reference} lowers the two raised fingers",
            f"{reference} opens the palm and waves",
        ],
        "interaction_entity_ids": [],
        "required_i0_entity_ids": [entity_id],
    }


def target_plan(source: dict) -> dict:
    dynamic_ids = [row["unit_id"] for row in source["dynamic_units"]]
    static_ids = [row["unit_id"] for row in source["static_salient_people"]]
    static_targets = [
        {
            "schema_version": contract.TARGET_STATIC_PERSON_SCHEMA,
            "unit_id": row["unit_id"],
            "entity_id": row["entity_id"],
            "entity_type": row["entity_type"],
            "stable_reference": row["stable_reference"],
            "target_state": "remain_still",
            "target_clause": f"have {row['stable_reference']} remain still",
        }
        for row in source["static_salient_people"]
    ]
    return {
        "schema_version": contract.TARGET_PLAN_SCHEMA,
        "iid": source["iid"],
        "source_census_sha256": contract.object_sha256(source),
        "dynamic_unit_targets": [
            _target_unit(
                "unit_01", "entity_01", "the person on the left", "left"
            ),
            _target_unit(
                "unit_02", "entity_02", "the person on the right", "right"
            ),
        ],
        "static_person_targets": static_targets,
        "camera_target": {
            "schema_version": contract.TARGET_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_relation": "preserve_static",
            "target_motion_class": "locked_off",
            "target_motion_signature": "locked_off",
            "target_motion_description": "locked off",
            "target_clause": "keep the camera locked off",
            "source_motion_suppressed": False,
            "substantive_change": False,
            "starts_at_i0": True,
            "i0_executable": True,
            "complete_within_clip": True,
            "completion_time_seconds": 3.2,
            "ordered_stages": ["keep the camera locked off for the full clip"],
        },
        "preservation": {
            "schema_version": contract.TARGET_PRESERVATION_SCHEMA,
            "preserve_identity": True,
            "preserve_appearance": True,
            "preserve_scene": True,
            "allow_new_entities": False,
            "allow_removed_entities": False,
        },
        "coverage": {
            "schema_version": contract.TARGET_COVERAGE_SCHEMA,
            "required_dynamic_unit_ids": dynamic_ids,
            "planned_changed_unit_ids": dynamic_ids,
            "missing_unit_ids": [],
            "extra_unit_ids": [],
            "required_static_person_ids": static_ids,
            "constrained_static_person_ids": static_ids,
            "camera_clause_present": True,
        },
        "i0_executable": True,
        "no_new_prerequisites": True,
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _set_shared_base(target: dict, motion: str) -> None:
    target["motion_relation"] = "explicit_shared_base_with_novel_action"
    target["source_motion_suppressed"] = False
    target["explicit_shared_base_motion"] = motion
    target["source_component_dispositions"][0].update(
        {
            "disposition": "explicit_shared_base",
            "explicit_target_motion": motion,
        }
    )


class GokuFullMotionContractTests(unittest.TestCase):
    def test_passive_reachable_vehicle_is_a_valid_i0_interaction_entity(self) -> None:
        census = source_census()
        census["i0_visible_entities"].append("the stationary trailer below")
        census["i0_entity_registry"].append(
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_03",
                "entity_type": "vehicle",
                "stable_reference": "the stationary trailer below",
                "i0_bbox_xyxy_1000": [400, 920, 600, 990],
                "viewer_region": "lower_center",
                "region_ordinal": 1,
                "role": "passive_interaction_object",
                "visible_at_i0": True,
                "reachable_at_i0": True,
                "confidence": "high",
            }
        )
        validated = contract.validate_source_census(census)
        self.assertEqual(
            validated["i0_entity_registry"][-1]["entity_type"], "vehicle"
        )

    def test_two_people_all_changed_and_camera_explicit(self) -> None:
        source = source_census()
        plan = target_plan(source)
        validated_source = contract.validate_source_census(source)
        validated_plan = contract.validate_target_plan(
            plan, source_census=source
        )
        binding = contract.build_contract(
            source_census=source, target_plan=plan
        )
        self.assertEqual(
            [unit["unit_id"] for unit in validated_source["dynamic_units"]],
            ["unit_01", "unit_02"],
        )
        self.assertEqual(
            [unit["unit_id"] for unit in validated_plan["dynamic_unit_targets"]],
            ["unit_01", "unit_02"],
        )
        self.assertTrue(binding["all_dynamic_units_changed"])
        self.assertTrue(binding["camera_explicit"])
        self.assertEqual(
            contract.validate_contract_binding(
                binding, source_census=source, target_plan=plan
            ),
            binding,
        )

    def test_missing_right_person_target_fails_closed(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"].pop()
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "exactly one target"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_missing_source_component_disposition_fails_closed(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0][
            "source_component_dispositions"
        ].clear()
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "dispose every source motion component exactly once",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_unbound_pick_up_red_ball_fails_closed(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
            "lower the raised hand, then pick up a red ball and hold it"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "interaction motion must bind its I0 entity",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_independent_inventory_alignment_rejects_missing_actor(self) -> None:
        primary = source_census()
        secondary = source_census()
        secondary["i0_visible_entities"].pop()
        secondary["i0_entity_registry"].pop()
        secondary["dynamic_units"].pop()
        contract.validate_source_census(primary)
        contract.validate_source_census(secondary)
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "independent source inventory projections differ",
        ):
            contract.build_source_inventory_alignment(
                primary=primary,
                secondary=secondary,
            )

    def test_independent_inventory_alignment_rejects_missing_passive_object(
        self,
    ) -> None:
        primary = source_census()
        basket_reference = "the woven basket held by the left person"
        primary["i0_visible_entities"].append(basket_reference)
        primary["i0_entity_registry"].append(
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_03",
                "entity_type": "rigid_object",
                "stable_reference": basket_reference,
                "i0_bbox_xyxy_1000": [310, 500, 390, 680],
                "viewer_region": "center",
                "region_ordinal": 1,
                "role": "passive_interaction_object",
                "visible_at_i0": True,
                "reachable_at_i0": True,
                "confidence": "high",
            }
        )
        primary["dynamic_units"][0]["source_motion_components"].append(
            {
                "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                "component_id": "component_02",
                "component_type": "object_interaction",
                "motion_signature": "carry_basket_with_gait",
                "motion_description": (
                    "carries the woven basket while walking"
                ),
                "dependent_entity_ids": ["entity_03"],
                "motion_evidence": [
                    _evidence("the held basket moves with the walking person")
                ],
            }
        )
        secondary = copy.deepcopy(primary)
        secondary["i0_visible_entities"].pop()
        secondary["i0_entity_registry"].pop()
        secondary["dynamic_units"][0]["source_motion_components"].pop()

        contract.validate_source_census(primary)
        contract.validate_source_census(secondary)
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "independent source inventory projections differ",
        ):
            contract.build_source_inventory_alignment(
                primary=primary,
                secondary=secondary,
            )

    def test_independent_inventory_alignment_accepts_bbox_jitter_and_prose(self) -> None:
        primary = source_census()
        secondary = source_census()
        for index, entity in enumerate(secondary["i0_entity_registry"]):
            reference = "the left blue actor" if index == 0 else "the right dark actor"
            entity["stable_reference"] = reference
            entity["i0_bbox_xyxy_1000"] = [
                coordinate + delta
                for coordinate, delta in zip(
                    entity["i0_bbox_xyxy_1000"],
                    (10, -10, 10, -10),
                    strict=True,
                )
            ]
            secondary["dynamic_units"][index]["stable_reference"] = reference
        secondary["i0_visible_entities"] = [
            entity["stable_reference"]
            for entity in secondary["i0_entity_registry"]
        ]
        alignment = contract.build_source_inventory_alignment(
            primary=primary,
            secondary=secondary,
        )
        self.assertTrue(alignment["projections_equal"])
        self.assertEqual(len(alignment["entity_matches"]), 2)
        self.assertTrue(
            all(item["bbox_iou_milli"] >= 250 for item in alignment["entity_matches"])
        )

    def test_independent_inventory_alignment_accepts_limb_first_raise_prose(
        self,
    ) -> None:
        primary = source_census()
        secondary = copy.deepcopy(primary)
        primary["dynamic_units"][0]["source_motion_components"][0][
            "motion_description"
        ] = "raises the right hand from hip to chest and forms a peace sign"
        primary["dynamic_units"][1]["source_motion_components"][0][
            "motion_description"
        ] = (
            "raises the left gloved hand from waistband to chest and forms "
            "a peace sign"
        )
        secondary["dynamic_units"][0]["source_motion_components"][0][
            "motion_description"
        ] = "the right hand raises from hip to chest and forms a peace sign"
        secondary["dynamic_units"][1]["source_motion_components"][0][
            "motion_description"
        ] = (
            "the left hand (gloved) raises from waistband to chest and forms "
            "a peace sign"
        )

        alignment = contract.build_source_inventory_alignment(
            primary=primary,
            secondary=secondary,
        )

        self.assertTrue(alignment["projections_equal"])
        for match in alignment["dynamic_unit_matches"]:
            self.assertEqual(
                match["semantic_primitives"],
                ["limb_raise", "peace_sign"],
            )

    def test_independent_inventory_alignment_matches_mount_paraphrases(
        self,
    ) -> None:
        primary = source_census()
        secondary = copy.deepcopy(primary)
        primary_component = primary["dynamic_units"][0][
            "source_motion_components"
        ][0]
        secondary_component = secondary["dynamic_units"][0][
            "source_motion_components"
        ][0]
        primary_component.update(
            {
                "component_type": "object_interaction",
                "motion_signature": "jumps_onto_other_actor",
                "motion_description": (
                    "jumps onto the other actor's back and wraps both arms "
                    "around the torso"
                ),
                "dependent_entity_ids": ["entity_02"],
            }
        )
        secondary_component.update(
            {
                "component_type": "object_interaction",
                "motion_signature": "mounts_other_actor",
                "motion_description": (
                    "mounts the other actor and maintains physical contact"
                ),
                "dependent_entity_ids": ["entity_02"],
            }
        )

        alignment = contract.build_source_inventory_alignment(
            primary=primary,
            secondary=secondary,
        )

        self.assertTrue(alignment["projections_equal"])
        self.assertEqual(
            alignment["dynamic_unit_matches"][0]["semantic_primitives"],
            ["mount"],
        )

    def test_independent_inventory_alignment_keeps_ball_actions_distinct(
        self,
    ) -> None:
        primary = source_census()
        ball = {
            "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
            "entity_id": "entity_03",
            "entity_type": "rigid_object",
            "stable_reference": "the soccer ball between the two players",
            "i0_bbox_xyxy_1000": [450, 700, 550, 800],
            "viewer_region": "lower_center",
            "region_ordinal": 1,
            "role": "passive_interaction_object",
            "visible_at_i0": True,
            "reachable_at_i0": True,
            "confidence": "high",
        }
        primary["i0_entity_registry"].append(ball)
        primary["i0_visible_entities"].append(ball["stable_reference"])
        primary_component = primary["dynamic_units"][0][
            "source_motion_components"
        ][0]
        primary_component.update(
            {
                "component_type": "object_interaction",
                "motion_signature": "strikes_soccer_ball",
                "motion_description": "strikes the soccer ball with the right foot",
                "dependent_entity_ids": ["entity_03"],
            }
        )

        for signature, description in (
            ("dribbles_soccer_ball", "dribbles the soccer ball with the left foot"),
            ("tackles_for_soccer_ball", "tackles to dispossess the soccer ball"),
        ):
            with self.subTest(signature=signature):
                secondary = copy.deepcopy(primary)
                secondary_component = secondary["dynamic_units"][0][
                    "source_motion_components"
                ][0]
                secondary_component["motion_signature"] = signature
                secondary_component["motion_description"] = description
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "semantic primitives differ",
                ):
                    contract.build_source_inventory_alignment(
                        primary=primary,
                        secondary=secondary,
                    )

    def test_independent_inventory_alignment_filters_cross_channel_context(
        self,
    ) -> None:
        primary = source_census()
        secondary = copy.deepcopy(primary)
        primary["dynamic_units"][0]["source_motion_components"][0][
            "motion_description"
        ] = "raises two fingers in a peace-sign gesture while standing"
        alignment = contract.build_source_inventory_alignment(
            primary=primary,
            secondary=secondary,
        )
        self.assertTrue(alignment["projections_equal"])
        self.assertEqual(
            alignment["dynamic_unit_matches"][0]["semantic_primitives"],
            ["hand_sign", "peace_sign"],
        )

    def test_independent_inventory_alignment_normalizes_locomotion_trajectory(
        self,
    ) -> None:
        primary = source_census()
        secondary = copy.deepcopy(primary)
        primary_unit = primary["dynamic_units"][0]
        primary_unit.update(
            {
                "source_action_signature": "walks_left_to_right",
                "source_motion": (
                    "walks from left to right while keeping the head slightly "
                    "lowered"
                ),
            }
        )
        primary_component = primary_unit["source_motion_components"][0]
        primary_component.update(
            {
                "component_type": "locomotion",
                "motion_signature": "walks_left_to_right",
                "motion_description": (
                    "walks from left to right with alternating leg motion"
                ),
            }
        )
        secondary_unit = secondary["dynamic_units"][0]
        secondary_unit.update(
            {
                "source_action_signature": "rightward_walk",
                "source_motion": (
                    "walks steadily rightward while maintaining a "
                    "forward-facing head orientation"
                ),
            }
        )
        secondary_component = secondary_unit["source_motion_components"][0]
        secondary_component.update(
            {
                "component_type": "locomotion",
                "motion_signature": "rightward_walk",
                "motion_description": (
                    "walks rightward with alternating leg motion; the front "
                    "leg lifts and extends forward before retracting"
                ),
            }
        )

        primary_projection = contract.source_inventory_projection(primary)
        secondary_projection = contract.source_inventory_projection(secondary)
        self.assertEqual(
            primary_projection["dynamic_units"][0]["semantic_primitives"],
            ["trajectory_right", "walk"],
        )
        self.assertEqual(
            secondary_projection["dynamic_units"][0]["semantic_primitives"],
            ["trajectory_right", "walk"],
        )
        alignment = contract.build_source_inventory_alignment(
            primary=primary,
            secondary=secondary,
        )
        self.assertEqual(
            alignment["schema_version"],
            "motive-goku-full-motion-source-inventory-alignment-v4",
        )

    def test_independent_inventory_alignment_rejects_opposite_trajectory(
        self,
    ) -> None:
        primary = source_census()
        secondary = copy.deepcopy(primary)
        for census, signature, motion in (
            (
                primary,
                "walks_left_to_right",
                "walks from left to right across the frame",
            ),
            (
                secondary,
                "walks_right_to_left",
                "walks from right to left across the frame",
            ),
        ):
            unit = census["dynamic_units"][0]
            unit["source_action_signature"] = signature
            unit["source_motion"] = motion
            component = unit["source_motion_components"][0]
            component.update(
                {
                    "component_type": "locomotion",
                    "motion_signature": signature,
                    "motion_description": motion,
                }
            )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "semantic primitives differ",
        ):
            contract.build_source_inventory_alignment(
                primary=primary,
                secondary=secondary,
            )

    def test_independent_inventory_alignment_rejects_unknown_component_actions(
        self,
    ) -> None:
        primary = source_census()
        secondary = copy.deepcopy(primary)
        primary_unit = primary["dynamic_units"][0]
        primary_unit["source_action_signature"] = "snap_fingers_twice"
        primary_unit["source_motion"] = "snaps the fingers twice"
        primary_unit["source_motion_components"][0].update(
            {
                "motion_signature": "snap_fingers_twice",
                "motion_description": "snaps the fingers twice",
            }
        )
        secondary_unit = secondary["dynamic_units"][0]
        secondary_unit["source_action_signature"] = "trace_finger_circle"
        secondary_unit["source_motion"] = (
            "traces a circle in the air with one finger"
        )
        secondary_unit["source_motion_components"][0].update(
            {
                "motion_signature": "trace_finger_circle",
                "motion_description": (
                    "traces a circle in the air with one finger"
                ),
            }
        )

        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "no recognized semantic primitives",
        ):
            contract.build_source_inventory_alignment(
                primary=primary,
                secondary=secondary,
            )

    def test_independent_inventory_alignment_rejects_component_granularity(
        self,
    ) -> None:
        primary = source_census()
        primary["dynamic_units"][0]["source_motion_components"].append(
            {
                "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                "component_id": "component_02",
                "component_type": "articulation",
                "motion_signature": "elbow_bend_and_extend",
                "motion_description": (
                    "raises and lowers the arm by bending the elbow"
                ),
                "dependent_entity_ids": [],
                "motion_evidence": [
                    _evidence("the elbow bends and extends during the gesture")
                ],
            }
        )
        secondary = copy.deepcopy(primary)
        secondary["dynamic_units"][0]["source_motion_components"].pop()
        contract.validate_source_census(primary)
        contract.validate_source_census(secondary)
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "motion-component inventories differ",
        ):
            contract.build_source_inventory_alignment(
                primary=primary,
                secondary=secondary,
            )

    def test_source_inventory_alignment_v3_replay_is_rejected(self) -> None:
        primary = source_census()
        secondary = copy.deepcopy(primary)
        alignment = contract.build_source_inventory_alignment(
            primary=primary,
            secondary=secondary,
        )
        stale = copy.deepcopy(alignment)
        stale["schema_version"] = (
            "motive-goku-full-motion-source-inventory-alignment-v3"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "source inventory alignment differs",
        ):
            contract.validate_source_inventory_alignment(
                stale,
                primary=primary,
                secondary=secondary,
            )

    def test_independent_inventory_alignment_rejects_opposite_motion(self) -> None:
        primary = source_census()
        secondary = source_census()
        unit = secondary["dynamic_units"][0]
        unit["source_action_signature"] = "lower_hand_to_waist"
        unit["source_motion"] = "lowers the raised hand from chest to waist"
        component = unit["source_motion_components"][0]
        component["motion_signature"] = "lower_hand_to_waist"
        component["motion_description"] = (
            "lowers the raised hand from chest to waist"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "semantic primitives differ",
        ):
            contract.build_source_inventory_alignment(
                primary=primary,
                secondary=secondary,
            )

    def test_missing_camera_fails_closed(self) -> None:
        source = source_census()
        plan = target_plan(source)
        del plan["camera_target"]
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "closed schema"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_dynamic_motion_cannot_be_a_pure_stillness_assertion(self) -> None:
        mutations = {
            "unit_signature": lambda unit: unit.__setitem__(
                "source_action_signature", "remains_completely_still"
            ),
            "unit_motion": lambda unit: unit.__setitem__(
                "source_motion", "there is no visible motion"
            ),
            "unit_evidence": lambda unit: unit.__setitem__(
                "motion_evidence",
                [_evidence("the person remains completely still")],
            ),
            "component_signature": lambda unit: unit[
                "source_motion_components"
            ][0].__setitem__("motion_signature", "no_gesture_change"),
            "component_description": lambda unit: unit[
                "source_motion_components"
            ][0].__setitem__(
                "motion_description", "there is no visible gesture change"
            ),
            "component_evidence": lambda unit: unit[
                "source_motion_components"
            ][0].__setitem__(
                "motion_evidence",
                [_evidence("the hand shows no visible gesture change")],
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                source = source_census()
                mutate(source["dynamic_units"][1])
                with self.assertRaises(contract.GokuFullMotionContractError):
                    contract.validate_source_census(source)

    def test_local_stillness_with_positive_gesture_motion_is_allowed(self) -> None:
        source = source_census()
        unit = source["dynamic_units"][1]
        mixed = "the torso remains completely still while the right hand waves"
        unit["source_motion"] = mixed
        unit["motion_evidence"] = [_evidence(mixed)]
        unit["source_motion_components"][0]["motion_description"] = mixed
        unit["source_motion_components"][0]["motion_evidence"] = [
            _evidence(mixed)
        ]
        self.assertEqual(
            contract.validate_source_census(source)["dynamic_units"][1][
                "source_motion"
            ],
            mixed,
        )

    def test_past_participle_requires_a_finite_motion_verb(self) -> None:
        source = source_census()
        unit = source["dynamic_units"][1]
        trajectory = (
            "the right arm raised from the abdomen to shoulder height while "
            "the left arm remained stationary"
        )
        unit["source_motion"] = trajectory
        unit["motion_evidence"] = [_evidence(trajectory)]
        unit["source_motion_components"][0]["motion_description"] = trajectory
        unit["source_motion_components"][0]["motion_evidence"] = [
            _evidence(trajectory)
        ]
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "no positive dynamic motion",
        ):
            contract.validate_source_census(source)

    def test_static_raised_participle_without_trajectory_is_rejected(self) -> None:
        for description in (
            "the right arm is raised at shoulder height while the body "
            "remains stationary",
            "the right arm is raised at shoulder height from frame 0 to "
            "frame 80 while the body remains stationary",
            "the right arm is raised at shoulder height while lighting "
            "changes from dark to bright and the body remains stationary",
            "the right arm raised from frame 0 to frame 80 while the body "
            "remained stationary",
        ):
            with self.subTest(description=description):
                source = source_census()
                source["dynamic_units"][1]["source_motion_components"][0][
                    "motion_description"
                ] = description
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "no positive dynamic motion",
                ):
                    contract.validate_source_census(source)

    def test_copular_still_as_temporal_adverb_is_not_static_motion(self) -> None:
        source = source_census()
        component = source["dynamic_units"][1]["source_motion_components"][0]
        component["motion_description"] = (
            "the right arm is still extended as the elbow shifts inward"
        )
        component["motion_evidence"] = [
            _evidence(
                "the hand is still in contact while its angle changes visibly"
            )
        ]
        self.assertEqual(contract.validate_source_census(source), source)

    def test_unambiguous_copular_or_remain_still_motion_is_rejected(self) -> None:
        for description in (
            "the hand is completely still",
            "the hand remains still",
            "the hand is stationary",
            "there is no significant turning",
            "the head shows no visible head turn",
        ):
            with self.subTest(description=description):
                source = source_census()
                source["dynamic_units"][1]["source_motion_components"][0][
                    "motion_description"
                ] = description
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "no positive dynamic motion",
                ):
                    contract.validate_source_census(source)

    def test_static_component_signatures_fail_without_rejecting_steady_gait(
        self,
    ) -> None:
        for signature in (
            "stable_riding_posture",
            "steady_handlebar_grip",
            "head_steady_forward",
        ):
            with self.subTest(signature=signature):
                source = source_census()
                source["dynamic_units"][1]["source_motion_components"][0][
                    "motion_signature"
                ] = signature
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "not a dynamic action",
                ):
                    contract.validate_source_census(source)

        source = source_census()
        source["dynamic_units"][1]["source_motion_components"][0][
            "motion_signature"
        ] = "steady_walk_left_to_right"
        self.assertEqual(contract.validate_source_census(source), source)

    def test_stable_reference_hard_limit_preserves_full_identity_text(self) -> None:
        source = source_census()
        reference = "the uniquely dressed left actor " + "x" * (
            contract.MAX_STABLE_REFERENCE_CHARS
            - len("the uniquely dressed left actor ")
        )
        self.assertEqual(len(reference), contract.MAX_STABLE_REFERENCE_CHARS)
        source["i0_entity_registry"][0]["stable_reference"] = reference
        source["dynamic_units"][0]["stable_reference"] = reference
        source["i0_visible_entities"][0] = reference
        self.assertEqual(
            contract.validate_source_census(source)["dynamic_units"][0][
                "stable_reference"
            ],
            reference,
        )

        too_long = reference + "x"
        source["i0_entity_registry"][0]["stable_reference"] = too_long
        source["dynamic_units"][0]["stable_reference"] = too_long
        source["i0_visible_entities"][0] = too_long
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "too long"
        ):
            contract.validate_source_census(source)

    def test_same_source_action_signature_is_not_an_edit(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["target_action_signature"] = source[
            "dynamic_units"
        ][0]["source_action_signature"]
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "identical to the source"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_same_source_motion_text_is_not_novel(self) -> None:
        source = source_census()
        plan = target_plan(source)
        source_motion = source["dynamic_units"][0]["source_motion"]
        # Sentence punctuation is formatting, not a substantive edit.
        plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
            source_motion + "."
        )
        plan["dynamic_unit_targets"][0]["target_clause"] = (
            f"have the person on the left {source_motion}"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "restates source motion"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_semantic_source_motion_paraphrase_is_not_novel(self) -> None:
        source = source_census()
        unit = source["dynamic_units"][1]
        unit["source_action_signature"] = "raise_glove_into_hand_sign"
        unit["source_motion"] = (
            "raises his gloved hand from his waist into a hand sign"
        )
        unit["source_motion_components"][0]["motion_signature"] = (
            "raise_glove_into_hand_sign"
        )
        unit["source_motion_components"][0]["motion_description"] = (
            "raises his gloved hand from his waist into a hand sign"
        )
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][1]
        target["target_action_signature"] = "lift_glove_into_hand_sign"
        target["novel_target_motion"] = (
            "moves his black-gloved hand upward from his waist and shapes "
            "it into a hand sign"
        )
        target["ordered_stages"] = [
            "immediately lift the black-gloved hand upward from the waist",
            "then shape the raised black-gloved hand into a hand sign",
        ]
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "semantic restatement",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_nonrendered_stage_cannot_manufacture_semantic_novelty(self) -> None:
        source = source_census()
        plan = target_plan(source)
        executable_restatements = (
            "display a V-sign gesture while moving upward",
            "display a V-sign gesture",
        )
        for index, (target, restatement) in enumerate(
            zip(
                plan["dynamic_unit_targets"],
                executable_restatements,
                strict=True,
            )
        ):
            target["target_action_signature"] = (
                f"reworded_peace_sign_{index}"
            )
            target["novel_target_motion"] = restatement
            # This primitive is not rendered by the deterministic compiler and
            # therefore must not turn the V-sign restatement into an edit.
            target["ordered_stages"] = [restatement, "nod the head once"]

        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "semantic restatement",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_unit_cannot_borrow_novelty_from_another_dynamic_subject(self) -> None:
        source = source_census()
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        target["target_action_signature"] = "repeat_v_sign_plus_other_nod"
        target["novel_target_motion"] = (
            "display a V-sign gesture while [[entity_02]] nods once"
        )
        target["ordered_stages"] = [target["novel_target_motion"]]
        target["interaction_entity_ids"] = ["entity_02"]
        target["required_i0_entity_ids"] = ["entity_01", "entity_02"]

        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "cannot borrow novelty from another dynamic subject",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_each_dynamic_unit_may_describe_its_own_new_action(self) -> None:
        source = source_census()
        plan = target_plan(source)

        validated = contract.validate_target_plan(
            plan, source_census=source
        )

        self.assertEqual(
            [
                target["target_action_signature"]
                for target in validated["dynamic_unit_targets"]
            ],
            ["open_palm_wave_left", "open_palm_wave_right"],
        )

    def test_dynamic_unit_may_interact_with_static_i0_object(self) -> None:
        source = source_census()
        source["i0_visible_entities"].append("the small red box below")
        source["i0_entity_registry"].append(
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_03",
                "entity_type": "rigid_object",
                "stable_reference": "the small red box below",
                "i0_bbox_xyxy_1000": [320, 760, 430, 910],
                "viewer_region": "lower_center",
                "region_ordinal": 1,
                "role": "passive_interaction_object",
                "visible_at_i0": True,
                "reachable_at_i0": True,
                "confidence": "high",
            }
        )
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        target["target_action_signature"] = "pick_up_red_box_and_wave"
        target["novel_target_motion"] = (
            "lower the peace sign, pick up [[entity_03]], and wave with an "
            "open palm while holding the box"
        )
        target["target_clause"] = (
            "have the left person pick up the red box and wave"
        )
        target["ordered_stages"] = [
            "lower the raised fingers",
            "pick up [[entity_03]] and wave with an open palm",
        ]
        target["interaction_entity_ids"] = ["entity_03"]
        target["required_i0_entity_ids"] = ["entity_01", "entity_03"]

        validated = contract.validate_target_plan(
            plan, source_census=source
        )
        self.assertEqual(
            validated["dynamic_unit_targets"][0]["interaction_entity_ids"],
            ["entity_03"],
        )

    def test_generic_source_future_anaphora_is_forbidden(self) -> None:
        source = source_census()
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        target["novel_target_motion"] = (
            "keep doing what he does while nodding twice"
        )
        target["ordered_stages"] = [
            "immediately keep doing what he does",
            "then nod twice while doing it",
        ]
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "source-future",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_crowd_and_unclear_inventory_fail_closed(self) -> None:
        crowd = source_census()
        crowd["crowd_or_unresolved_motion"] = True
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "must be exactly false"
        ):
            contract.validate_source_census(crowd)

        unclear = source_census()
        unclear["confidence"] = "unclear"
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "exactly 'high'"
        ):
            contract.validate_source_census(unclear)

    def test_non_contiguous_dynamic_ids_fail_closed(self) -> None:
        source = source_census()
        source["dynamic_units"][1]["unit_id"] = "unit_03"
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "contiguous"
        ):
            contract.validate_source_census(source)

    def test_ambiguous_source_future_wording_is_forbidden(self) -> None:
        source = source_census()
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        target["novel_target_motion"] = "continue the original hand action"
        target["target_clause"] = (
            "have the person on the left continue the original hand action"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "source-future"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_non_executable_target_clauses_only_require_basic_text(self) -> None:
        source = source_census(static_person=True)
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["target_clause"] = (
            "preserve identity and appearance while continuing the original "
            "motion as in the source video"
        )
        plan["static_person_targets"][0]["target_clause"] = (
            "keep the current pose shown at left like in the original video"
        )
        plan["camera_target"]["target_clause"] = (
            "retain the previous trajectory as in source video"
        )
        self.assertEqual(
            contract.validate_target_plan(plan, source_census=source), plan
        )

        malformed = copy.deepcopy(plan)
        malformed["dynamic_unit_targets"][0]["target_clause"] = (
            " target clause with leading whitespace"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "trimmed single-line text",
        ):
            contract.validate_target_plan(malformed, source_census=source)

    def test_executable_target_prose_allows_absolute_i0_wording(self) -> None:
        allowed_novel_motions = (
            "preserve identity and appearance while beginning a new "
            "open-palm wave",
            "keep the torso upright while opening the raised hand",
            "from the current pose, lower the raised hand and wave",
            "at the same moment, open the raised hand and wave",
            "the actor shown at left opens the raised hand and waves",
            "begin moving clockwise around the visible marker, then continue "
            "moving clockwise around that marker",
        )
        for novel_motion in allowed_novel_motions:
            with self.subTest(novel_motion=novel_motion):
                source = source_census()
                plan = target_plan(source)
                plan["dynamic_unit_targets"][0][
                    "novel_target_motion"
                ] = novel_motion
                contract.validate_target_plan(plan, source_census=source)

    def test_executable_target_fields_reject_source_future_shortcuts(self) -> None:
        def set_novel(plan: dict) -> None:
            plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
                "continue the original motion"
            )

        def set_shared(plan: dict) -> None:
            target = plan["dynamic_unit_targets"][0]
            _set_shared_base(target, "retain it")

        def set_shared_with_descriptors(plan: dict) -> None:
            target = plan["dynamic_unit_targets"][0]
            _set_shared_base(
                target,
                "retain the existing slow walking action"
            )

        def set_dynamic_stage(plan: dict) -> None:
            plan["dynamic_unit_targets"][0]["ordered_stages"][0] = (
                "continue the motion"
            )

        def set_camera_description(plan: dict) -> None:
            plan["camera_target"]["target_motion_description"] = (
                "move as in source video"
            )

        def set_camera_stage(plan: dict) -> None:
            plan["camera_target"]["ordered_stages"][0] = (
                "retain the previous trajectory"
            )

        cases = {
            "novel motion": set_novel,
            "shared base pronoun": set_shared,
            "shared base source marker plus descriptors": (
                set_shared_with_descriptors
            ),
            "dynamic stage": set_dynamic_stage,
            "camera description": set_camera_description,
            "camera stage": set_camera_stage,
        }
        for name, mutate in cases.items():
            with self.subTest(field=name):
                source = source_census()
                plan = target_plan(source)
                mutate(plan)
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError, "source-future"
                ):
                    contract.validate_target_plan(plan, source_census=source)

    def test_replace_rejects_hidden_continuing_locomotion_base(self) -> None:
        hidden_bases = (
            "keep walking forward while raising the open palm to wave",
            "keep riding rightward while standing upright on the footpegs",
            "stand upright while continuing to accelerate leftward along the "
            "curved dirt track",
        )
        for novel_motion in hidden_bases:
            with self.subTest(novel_motion=novel_motion):
                source = source_census()
                plan = target_plan(source)
                plan["dynamic_unit_targets"][0][
                    "novel_target_motion"
                ] = novel_motion
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "hides continuing locomotion under replace",
                ):
                    contract.validate_target_plan(
                        plan,
                        source_census=source,
                    )

    def test_motorcycle_continuation_requires_explicit_shared_base(self) -> None:
        source = source_census()
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        target["novel_target_motion"] = (
            "Immediately, the rider on the orange and white dirt bike with "
            "number 31 begins to stand up on the footpegs, lifting their body "
            "upright over the bike while maintaining balance. Then, as the "
            "bike continues moving leftward along the curved dirt path, the "
            "rider leans slightly forward, extending their arms for stability, "
            "and actively steers with subtle body shifts. By the end, the rider "
            "is fully upright on the footpegs, riding at a steady pace with "
            "controlled motion, kicking up a moderate dust cloud behind the "
            "rear wheel."
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "explicit_shared_base_with_novel_action",
        ):
            contract.validate_target_plan(plan, source_census=source)

        _set_shared_base(
            target,
            "From I0, the orange and white dirt bike moves leftward along the "
            "curved dirt path at a steady pace and raises a moderate dust "
            "trail"
        )
        target["novel_target_motion"] = (
            "Immediately, the rider rises from the seat onto the footpegs, "
            "then leans slightly forward with both arms extended for balance, "
            "and by the end remains fully upright while actively steering"
        )
        contract.validate_target_plan(plan, source_census=source)

    def test_replace_allows_target_internal_locomotion_continuation(self) -> None:
        source = source_census()
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        target["novel_target_motion"] = (
            "Immediately, the person begins walking rightward from I0. Then, "
            "the person continues walking rightward while opening the raised "
            "hand into a wave."
        )
        target["ordered_stages"] = [
            "The person begins walking rightward from I0",
            "The person continues walking rightward while opening the hand",
        ]
        contract.validate_target_plan(plan, source_census=source)

    def test_replace_stage_cannot_hide_unestablished_locomotion(self) -> None:
        source = source_census()
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        target["ordered_stages"] = [
            "The rider rises while keeping the motorcycle moving leftward",
            "The rider reaches an upright stance and opens one hand",
        ]
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "ordered_stages hides continuing locomotion under replace",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_source_future_path_comparisons_are_forbidden(self) -> None:
        forbidden = (
            "move right in the opposite direction of its original path",
            "turn around relative to the source route",
            "move away from the previous leftward course",
            "switch direction from the existing travel line",
        )
        for novel_motion in forbidden:
            with self.subTest(novel_motion=novel_motion):
                source = source_census()
                plan = target_plan(source)
                plan["dynamic_unit_targets"][0][
                    "novel_target_motion"
                ] = novel_motion
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "source-future",
                ):
                    contract.validate_target_plan(
                        plan,
                        source_census=source,
                    )

    def test_as_in_source_comparison_is_forbidden_in_executable_prose(self) -> None:
        source = source_census()

        novel_plan = target_plan(source)
        novel_plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
            "walk steadily rightward as in the source"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "source-future",
        ):
            contract.validate_target_plan(novel_plan, source_census=source)

        shared_plan = target_plan(source)
        shared = shared_plan["dynamic_unit_targets"][0]
        _set_shared_base(
            shared,
            "walk steadily while maintaining the same general path and "
            "direction as in the source"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "source-future",
        ):
            contract.validate_target_plan(shared_plan, source_census=source)

        stage_plan = target_plan(source)
        stage_plan["dynamic_unit_targets"][0]["ordered_stages"][0] = (
            "The person lowers the hand as in the original"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "source-future",
        ):
            contract.validate_target_plan(stage_plan, source_census=source)

        camera_plan = target_plan(source)
        camera_plan["camera_target"]["target_motion_description"] = (
            "Keep the camera locked off as in the original video"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "source-future",
        ):
            contract.validate_target_plan(camera_plan, source_census=source)

    def test_directional_locomotion_in_place_is_forbidden(self) -> None:
        contradictions = (
            "trot forward in place while opening the raised hand",
            "walk slowly backward in place while waving",
            "run leftward while remaining in place",
            "the bike moves rightward while staying completely in place",
            "ride forward in place while standing on the footpegs",
            "drive the motorcycle backward in place while leaning forward",
        )
        for novel_motion in contradictions:
            with self.subTest(novel_motion=novel_motion):
                source = source_census()
                plan = target_plan(source)
                plan["dynamic_unit_targets"][0][
                    "novel_target_motion"
                ] = novel_motion
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "contradictory directional locomotion in place",
                ):
                    contract.validate_target_plan(
                        plan,
                        source_census=source,
                    )

    def test_directional_locomotion_in_place_is_forbidden_in_stages(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["ordered_stages"][0] = (
            "The person trots forward in place while lowering the hand"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            r"ordered_stages\[0\].*contradictory directional locomotion",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_noncontradictory_in_place_chronology_is_allowed(self) -> None:
        allowed = (
            "turn in place while facing forward and open the raised hand",
            "walk forward, then stop and jog in place while waving",
        )
        for novel_motion in allowed:
            with self.subTest(novel_motion=novel_motion):
                source = source_census()
                plan = target_plan(source)
                plan["dynamic_unit_targets"][0][
                    "novel_target_motion"
                ] = novel_motion
                contract.validate_target_plan(plan, source_census=source)

    def test_same_gait_after_target_trot_is_target_internal(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
            "Immediately, the horse pivots clockwise and begins trotting "
            "steadily toward the right. By the end, the horse is still "
            "trotting with the same gait near the visible pole."
        )
        contract.validate_target_plan(plan, source_census=source)

    def test_complete_noun_and_from_i0_novel_prose_are_allowed(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
            "the left person's raised hand lowers before the open palm waves "
            "twice"
        )
        plan["dynamic_unit_targets"][0]["target_clause"] = (
            "have the left person lower the hand and wave twice"
        )
        plan["dynamic_unit_targets"][1]["novel_target_motion"] = (
            "from I0, the right person opens the raised fingers and completes "
            "two side-to-side waves"
        )
        plan["dynamic_unit_targets"][1]["target_clause"] = (
            "have the right person open the hand and wave twice from I0"
        )
        self.assertEqual(
            contract.validate_target_plan(plan, source_census=source), plan
        )

    def test_visible_inventory_must_exactly_repeat_registry_references(self) -> None:
        source = source_census()
        source["i0_visible_entities"][0] = "the left-side standing person"
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "must exactly equal the I0 registry stable references",
        ):
            contract.validate_source_census(source)

    def test_explicit_shared_base_is_structured_and_clause_may_paraphrase(self) -> None:
        source = source_census()
        plan = target_plan(source)
        target = plan["dynamic_unit_targets"][0]
        _set_shared_base(target, "walk forward")
        target["target_clause"] = (
            "have the person on the left walk forward and lower the peace sign "
            "and wave with an open palm"
        )
        contract.validate_target_plan(plan, source_census=source)

        paraphrased = copy.deepcopy(plan)
        paraphrased["dynamic_unit_targets"][0]["target_clause"] = (
            "have the person on the left advance while changing to a friendly "
            "open-hand gesture"
        )
        contract.validate_target_plan(paraphrased, source_census=source)

        missing = copy.deepcopy(plan)
        missing["dynamic_unit_targets"][0]["explicit_shared_base_motion"] = None
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "non-empty string"
        ):
            contract.validate_target_plan(missing, source_census=source)

    def test_paraphrased_actor_clause_keeps_exact_structured_reference(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["target_clause"] = (
            "have the left-side actor switch to a friendly open-hand wave"
        )
        contract.validate_target_plan(plan, source_census=source)

        wrong_reference = copy.deepcopy(plan)
        wrong_reference["dynamic_unit_targets"][0]["stable_reference"] = (
            "the left-side actor"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "stable_reference differs from source census",
        ):
            contract.validate_target_plan(
                wrong_reference, source_census=source
            )

    def test_static_clause_may_paraphrase_actor_but_must_say_remain_still(self) -> None:
        source = source_census(static_person=True)
        plan = target_plan(source)
        target = plan["static_person_targets"][0]
        target["target_clause"] = (
            "the central observer stays completely motionless."
        )
        self.assertEqual(
            contract.validate_target_plan(plan, source_census=source), plan
        )

        target["target_state"] = "hold_position"
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "remain_still"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_all_model_target_prose_accepts_terminal_punctuation(self) -> None:
        for punctuation in ".!?;:":
            with self.subTest(punctuation=punctuation):
                source = source_census(static_person=True)
                source["camera"] = {
                    "schema_version": contract.SOURCE_CAMERA_SCHEMA,
                    "camera_id": "camera",
                    "motion_class": "pan_left",
                    "motion_signature": "slow_pan_left",
                    "motion_description": "slow pan left",
                    "dynamic": True,
                    "motion_evidence": [
                        _evidence(
                            "the full background shifts right across the clip"
                        )
                    ],
                    "confidence": "high",
                }
                plan = target_plan(source)
                dynamic = plan["dynamic_unit_targets"][0]
                shared_motion = (
                    "walk forward from the exact visible stance" + punctuation
                )
                _set_shared_base(dynamic, shared_motion)
                dynamic.update(
                    {
                        "novel_target_motion": (
                            "from I0, open the raised hand and wave twice"
                            + punctuation
                        ),
                        "target_clause": (
                            "have the left person walk and wave twice"
                            + punctuation
                        ),
                    }
                )
                plan["static_person_targets"][0]["target_clause"] = (
                    "the central observer stays completely motionless"
                    + punctuation
                )
                plan["camera_target"].update(
                    {
                        "motion_relation": "replace_motion",
                        "target_motion_class": "dolly_in",
                        "target_motion_signature": "steady_dolly_in",
                        "target_motion_description": (
                            "a steady forward dolly toward the actors"
                            + punctuation
                        ),
                        "target_clause": (
                            "a gradual move closer to the actors" + punctuation
                        ),
                        "source_motion_suppressed": True,
                        "substantive_change": True,
                    }
                )
                self.assertEqual(
                    contract.validate_target_plan(plan, source_census=source),
                    plan,
                )

    def test_camera_crosscheck_may_paraphrase_structured_trajectory(self) -> None:
        source = source_census()
        source["camera"] = {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "pan_left",
            "motion_signature": "slow_pan_left",
            "motion_description": "slow pan left",
            "dynamic": True,
            "motion_evidence": [
                _evidence("the full background shifts right across the clip")
            ],
            "confidence": "high",
        }
        plan = target_plan(source)
        plan["camera_target"].update(
            {
                "motion_relation": "replace_motion",
                "target_motion_class": "dolly_in",
                "target_motion_signature": "steady_dolly_in",
                "target_motion_description": "a steady forward dolly toward the actors",
                "target_clause": "move the camera gradually closer to the actors",
                "source_motion_suppressed": True,
                "substantive_change": True,
            }
        )
        contract.validate_target_plan(plan, source_census=source)

    def test_static_camera_description_may_be_detailed_absolute_prose(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["camera_target"]["target_motion_description"] = (
            "stationary camera with no detectable panning tilting zooming or "
            "translation throughout"
        )
        plan["camera_target"]["target_clause"] = (
            "a fixed locked-off wide shot."
        )
        contract.validate_target_plan(plan, source_census=source)

    def test_static_camera_still_requires_locked_off_structure(self) -> None:
        source = source_census()
        base = target_plan(source)
        base["camera_target"]["target_motion_description"] = (
            "stationary camera with no detectable panning tilting zooming or "
            "translation throughout"
        )
        invalid_fields = (
            ("motion_relation", "replace_motion"),
            ("target_motion_class", "pan_left"),
            ("target_motion_signature", "stationary_no_pan"),
        )
        for field, value in invalid_fields:
            with self.subTest(field=field, value=value):
                plan = copy.deepcopy(base)
                plan["camera_target"][field] = value
                with self.assertRaisesRegex(
                    contract.GokuFullMotionContractError,
                    "preserve a static camera as locked off",
                ):
                    contract.validate_target_plan(plan, source_census=source)

    def test_static_salient_person_must_remain_still(self) -> None:
        source = source_census(static_person=True)
        plan = target_plan(source)
        contract.validate_target_plan(plan, source_census=source)
        plan["static_person_targets"][0]["target_state"] = "wave"
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "remain_still"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_static_salient_animal_is_also_constrained_to_remain_still(self) -> None:
        source = source_census(static_person=True)
        animal = source["static_salient_people"][0]
        animal.update(
            {
                "entity_type": "animal",
                "stable_reference": "the seated dog in the center",
                "i0_state": "A seated dog faces the two standing people",
                "motion_evidence": [
                    _evidence("the seated dog holds the same resting pose")
                ],
            }
        )
        source["i0_entity_registry"][-1].update(
            {
                "entity_type": "animal",
                "stable_reference": "the seated dog in the center",
            }
        )
        source["i0_visible_entities"][-1] = "the seated dog in the center"
        plan = target_plan(source)
        contract.validate_target_plan(plan, source_census=source)
        self.assertEqual(
            plan["static_person_targets"][0]["entity_type"], "animal"
        )
        self.assertIn(
            "remain still", plan["static_person_targets"][0]["target_clause"]
        )

    def test_dynamic_camera_must_have_different_target(self) -> None:
        source = source_census()
        source["camera"] = {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "pan_left",
            "motion_signature": "slow_pan_left",
            "motion_description": "slow pan left",
            "dynamic": True,
            "motion_evidence": [
                _evidence("the full background shifts right across the clip")
            ],
            "confidence": "high",
        }
        plan = target_plan(source)
        plan["camera_target"].update(
            {
                "motion_relation": "replace_motion",
                "target_motion_class": "pan_left",
                # Even a different label cannot make punctuation-only prose
                # variation into a substantively different trajectory.
                "target_motion_signature": "alternate_slow_pan_left",
                "target_motion_description": "slow pan left.",
                "target_clause": "move the camera in a slow pan left",
                "source_motion_suppressed": True,
                "substantive_change": True,
            }
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "different explicit trajectory"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_dynamic_camera_same_motion_class_rephrase_fails_closed(self) -> None:
        source = source_census()
        source["camera"] = {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "pan_left",
            "motion_signature": "slow_pan_left",
            "motion_description": "slow pan left",
            "dynamic": True,
            "motion_evidence": [
                _evidence("the full background shifts right across the clip")
            ],
            "confidence": "high",
        }
        plan = target_plan(source)
        plan["camera_target"].update(
            {
                "motion_relation": "replace_motion",
                "target_motion_class": "pan_left",
                "target_motion_signature": "smooth_horizontal_sweep_left",
                "target_motion_description": (
                    "a smooth horizontal sweep toward viewer-left"
                ),
                "target_clause": "sweep the camera smoothly toward the left",
                "source_motion_suppressed": True,
                "substantive_change": True,
            }
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "different explicit trajectory",
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_dynamic_camera_can_be_explicitly_replaced_by_locked_off(self) -> None:
        source = source_census()
        source["camera"] = {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "pan_left",
            "motion_signature": "slow_pan_left",
            "motion_description": "slow pan left",
            "dynamic": True,
            "motion_evidence": [
                _evidence("the full background shifts right across the clip")
            ],
            "confidence": "high",
        }
        plan = target_plan(source)
        plan["camera_target"].update(
            {
                "motion_relation": "replace_motion",
                "source_motion_suppressed": True,
                "substantive_change": True,
            }
        )
        validated = contract.validate_target_plan(plan, source_census=source)
        self.assertEqual(
            validated["camera_target"]["target_motion_signature"],
            "locked_off",
        )

    def test_completion_must_fit_three_point_two_seconds(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["completion_time_seconds"] = 3.21
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "3.2"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_required_entity_absent_at_i0_fails_closed(self) -> None:
        source = source_census()
        plan = target_plan(source)
        plan["dynamic_unit_targets"][0]["interaction_entity_ids"].append(
            "entity_99"
        )
        plan["dynamic_unit_targets"][0]["required_i0_entity_ids"].append(
            "entity_99"
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "not a distinct I0 entity"
        ):
            contract.validate_target_plan(plan, source_census=source)

    def test_source_model_output_canonicalizes_only_safe_redundancy(self) -> None:
        raw = source_census(static_person=True)
        del raw["i0_visible_entities"]
        del raw["i0_entity_registry"][0]["region_ordinal"]
        for entity in raw["i0_entity_registry"]:
            entity["viewer_region"] = "upper_left"
            entity["region_ordinal"] = 24
        # Keep one missing field as well as wrong fields in the model output.
        del raw["i0_entity_registry"][0]["region_ordinal"]
        del raw["dynamic_units"][0]["entity_type"]
        del raw["dynamic_units"][0]["stable_reference"]
        del raw["dynamic_units"][0]["visible_at_i0"]
        del raw["dynamic_units"][1]["stable_reference"]
        del raw["static_salient_people"][0]["entity_type"]
        del raw["static_salient_people"][0]["stable_reference"]
        del raw["static_salient_people"][0]["visible_at_i0"]
        raw_before = copy.deepcopy(raw)

        canonical, receipt = (
            contract.canonicalize_source_census_model_output(
                raw, "two-people-wave-001"
            )
        )

        self.assertEqual(raw, raw_before)
        self.assertEqual(
            canonical["i0_visible_entities"],
            [
                entity["stable_reference"]
                for entity in canonical["i0_entity_registry"]
            ],
        )
        self.assertEqual(
            [
                entity["viewer_region"]
                for entity in canonical["i0_entity_registry"]
            ],
            ["center_left", "center_right", "center"],
        )
        self.assertEqual(
            [
                entity["region_ordinal"]
                for entity in canonical["i0_entity_registry"]
            ],
            [1, 1, 1],
        )
        for unit in canonical["dynamic_units"]:
            registry = canonical["i0_entity_registry"][
                int(unit["entity_id"].removeprefix("entity_")) - 1
            ]
            self.assertEqual(unit["entity_type"], registry["entity_type"])
            self.assertEqual(
                unit["stable_reference"], registry["stable_reference"]
            )
            self.assertIs(unit["visible_at_i0"], registry["visible_at_i0"])
        static = canonical["static_salient_people"][0]
        registry_static = canonical["i0_entity_registry"][2]
        self.assertEqual(static["entity_type"], registry_static["entity_type"])
        self.assertEqual(
            static["stable_reference"], registry_static["stable_reference"]
        )
        self.assertIs(static["visible_at_i0"], True)
        self.assertFalse(receipt["semantic_repair"])
        self.assertEqual(receipt["raw_sha256"], contract.object_sha256(raw))
        self.assertEqual(
            receipt["canonical_sha256"], contract.object_sha256(canonical)
        )
        receipt_body = {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
        self.assertEqual(
            receipt["receipt_sha256"], contract.object_sha256(receipt_body)
        )
        self.assertEqual(
            contract.validate_source_census_canonicalization(
                raw,
                canonical,
                receipt,
                "two-people-wave-001",
            ),
            receipt,
        )

    def test_source_model_output_rejects_present_grounding_conflicts(self) -> None:
        cases: dict[str, tuple[dict, str]] = {}

        dynamic_type = source_census()
        dynamic_type["dynamic_units"][0]["entity_type"] = "animal"
        cases["dynamic_type"] = (
            dynamic_type,
            r"dynamic_units\[0\]\.entity_type",
        )

        dynamic_reference = source_census()
        dynamic_reference["dynamic_units"][0]["stable_reference"] = (
            "the person on the right"
        )
        cases["dynamic_reference"] = (
            dynamic_reference,
            r"dynamic_units\[0\]\.stable_reference",
        )

        dynamic_visibility = source_census()
        dynamic_visibility["dynamic_units"][0]["visible_at_i0"] = False
        cases["dynamic_visibility"] = (
            dynamic_visibility,
            r"dynamic_units\[0\]\.visible_at_i0",
        )

        static_reference = source_census(static_person=True)
        static_reference["static_salient_people"][0]["stable_reference"] = (
            "the person on the left"
        )
        cases["static_reference"] = (
            static_reference,
            r"static_salient_people\[0\]\.stable_reference",
        )

        for name, (raw, path_pattern) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                contract.GokuFullMotionContractError,
                rf"{path_pattern} conflicts with its authoritative value",
            ):
                contract.canonicalize_source_census_model_output(
                    raw, raw["iid"]
                )

    def test_source_model_output_rebuilds_review_inventory_from_registry(
        self,
    ) -> None:
        raw = source_census()
        raw["i0_visible_entities"] = [
            "an aggregate scene summary containing unstructured scenery"
        ]
        raw_before = copy.deepcopy(raw)

        canonical, receipt = contract.canonicalize_source_census_model_output(
            raw, raw["iid"]
        )

        self.assertEqual(raw, raw_before)
        self.assertEqual(
            canonical["i0_visible_entities"],
            [
                entity["stable_reference"]
                for entity in raw["i0_entity_registry"]
            ],
        )
        self.assertIn(
            "i0_visible_entities", receipt["normalized_field_paths"]
        )
        self.assertIn("i0_visible_entities", receipt["changed_field_paths"])
        self.assertFalse(receipt["semantic_repair"])
        self.assertEqual(
            contract.validate_source_census_canonicalization(
                raw, canonical, receipt, raw["iid"]
            ),
            receipt,
        )

    def test_source_model_output_noop_has_empty_changed_paths(self) -> None:
        raw = source_census(static_person=True)
        canonical, receipt = (
            contract.canonicalize_source_census_model_output(raw, raw["iid"])
        )
        self.assertEqual(canonical, raw)
        self.assertEqual(receipt["changed_field_paths"], [])
        self.assertGreater(len(receipt["normalized_field_paths"]), 0)

    def test_source_model_output_region_ordinals_follow_bbox_and_order(self) -> None:
        raw = source_census()
        for index, reference in enumerate(("the first red box", "the second red box")):
            raw["i0_visible_entities"].append(reference)
            raw["i0_entity_registry"].append(
                {
                    "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                    "entity_id": f"entity_{index + 3:02d}",
                    "entity_type": "rigid_object",
                    "stable_reference": reference,
                    "i0_bbox_xyxy_1000": [20 + index * 30, 20, 80 + index * 30, 90],
                    "viewer_region": "lower_right",
                    "region_ordinal": 24,
                    "role": "passive_interaction_object",
                    "visible_at_i0": True,
                    "reachable_at_i0": True,
                    "confidence": "high",
                }
            )
        canonical, _ = contract.canonicalize_source_census_model_output(
            raw, raw["iid"]
        )
        added = canonical["i0_entity_registry"][-2:]
        self.assertEqual(
            [(row["viewer_region"], row["region_ordinal"]) for row in added],
            [("upper_left", 1), ("upper_left", 2)],
        )

    def test_source_model_output_never_repairs_semantic_structure(self) -> None:
        cases = {}

        role = source_census()
        role["i0_entity_registry"][0]["role"] = "static_salient"
        cases["role"] = role

        bbox = source_census()
        bbox["i0_entity_registry"][0]["i0_bbox_xyxy_1000"] = [50, 50, 50, 90]
        cases["bbox"] = bbox

        component = source_census()
        duplicate = copy.deepcopy(
            component["dynamic_units"][0]["source_motion_components"][0]
        )
        duplicate["component_id"] = "component_02"
        component["dynamic_units"][0]["source_motion_components"].append(
            duplicate
        )
        cases["component"] = component

        camera = source_census()
        camera["camera"]["dynamic"] = True
        cases["camera"] = camera

        missing_actor = source_census()
        missing_actor["dynamic_units"].pop()
        cases["subject_count"] = missing_actor

        for name, raw in cases.items():
            with self.subTest(name=name), self.assertRaises(
                contract.GokuFullMotionContractError
            ):
                contract.canonicalize_source_census_model_output(
                    raw, "two-people-wave-001"
                )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "must be an object"
        ):
            contract.canonicalize_source_census_model_output(
                '{"malformed":', "two-people-wave-001"
            )

    def test_source_canonicalization_receipt_and_object_tampering_fail(self) -> None:
        raw = source_census()
        canonical, receipt = (
            contract.canonicalize_source_census_model_output(raw, raw["iid"])
        )
        forged_canonical = copy.deepcopy(canonical)
        forged_canonical["scene_description"] = "A forged scene description"
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "differs from deterministic reconstruction",
        ):
            contract.validate_source_census_canonicalization(
                raw, forged_canonical, receipt, raw["iid"]
            )
        forged_receipt = copy.deepcopy(receipt)
        forged_receipt["semantic_repair"] = True
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "receipt differs from reconstruction",
        ):
            contract.validate_source_census_canonicalization(
                raw, canonical, forged_receipt, raw["iid"]
            )

    def test_target_model_output_canonicalizes_only_identity_redundancy(self) -> None:
        source = source_census(static_person=True)
        raw = target_plan(source)
        del raw["dynamic_unit_targets"][0]["entity_id"]
        del raw["dynamic_unit_targets"][0]["stable_reference"]
        del raw["dynamic_unit_targets"][1]["stable_reference"]
        static = raw["static_person_targets"][0]
        del static["entity_id"]
        del static["entity_type"]
        del static["stable_reference"]
        raw_before = copy.deepcopy(raw)

        canonical, receipt = contract.canonicalize_target_plan_model_output(
            raw, source
        )

        self.assertEqual(raw, raw_before)
        for target, source_unit in zip(
            canonical["dynamic_unit_targets"],
            source["dynamic_units"],
            strict=True,
        ):
            self.assertEqual(target["entity_id"], source_unit["entity_id"])
            self.assertEqual(
                target["stable_reference"], source_unit["stable_reference"]
            )
        target_static = canonical["static_person_targets"][0]
        source_static = source["static_salient_people"][0]
        for field in ("entity_id", "entity_type", "stable_reference"):
            self.assertEqual(target_static[field], source_static[field])
        self.assertFalse(receipt["semantic_repair"])
        self.assertEqual(
            receipt["context"],
            {
                "iid": source["iid"],
                "source_census_sha256": contract.object_sha256(source),
            },
        )
        self.assertEqual(
            contract.validate_target_plan_canonicalization(
                raw, canonical, receipt, source
            ),
            receipt,
        )

    def test_target_model_output_rejects_present_actor_identity_conflicts(
        self,
    ) -> None:
        source = source_census(static_person=True)
        cases: dict[str, tuple[dict, str]] = {}

        dynamic_entity = target_plan(source)
        dynamic_entity["dynamic_unit_targets"][0]["entity_id"] = "entity_02"
        cases["dynamic_entity"] = (
            dynamic_entity,
            r"dynamic_unit_targets\[0\]\.entity_id",
        )

        dynamic_reference = target_plan(source)
        dynamic_reference["dynamic_unit_targets"][0]["stable_reference"] = (
            "the person on the right"
        )
        cases["dynamic_reference"] = (
            dynamic_reference,
            r"dynamic_unit_targets\[0\]\.stable_reference",
        )

        static_entity = target_plan(source)
        static_entity["static_person_targets"][0]["entity_id"] = "entity_01"
        cases["static_entity"] = (
            static_entity,
            r"static_person_targets\[0\]\.entity_id",
        )

        static_type = target_plan(source)
        static_type["static_person_targets"][0]["entity_type"] = "animal"
        cases["static_type"] = (
            static_type,
            r"static_person_targets\[0\]\.entity_type",
        )

        static_reference = target_plan(source)
        static_reference["static_person_targets"][0]["stable_reference"] = (
            "the person on the left"
        )
        cases["static_reference"] = (
            static_reference,
            r"static_person_targets\[0\]\.stable_reference",
        )

        for name, (raw, path_pattern) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                contract.GokuFullMotionContractError,
                rf"{path_pattern} conflicts with its authoritative value",
            ):
                contract.canonicalize_target_plan_model_output(raw, source)

    def test_target_model_output_noop_has_empty_changed_paths(self) -> None:
        source = source_census(static_person=True)
        raw = target_plan(source)
        canonical, receipt = contract.canonicalize_target_plan_model_output(
            raw, source
        )
        self.assertEqual(canonical, raw)
        self.assertEqual(receipt["changed_field_paths"], [])

    def test_target_model_output_requires_exact_list_and_unit_ids_first(self) -> None:
        source = source_census(static_person=True)
        cases = {}
        missing_dynamic = target_plan(source)
        missing_dynamic["dynamic_unit_targets"].pop()
        cases["missing_dynamic"] = missing_dynamic
        wrong_dynamic_id = target_plan(source)
        wrong_dynamic_id["dynamic_unit_targets"][0]["unit_id"] = "unit_02"
        cases["wrong_dynamic_id"] = wrong_dynamic_id
        missing_static = target_plan(source)
        missing_static["static_person_targets"].pop()
        cases["missing_static"] = missing_static
        wrong_static_id = target_plan(source)
        wrong_static_id["static_person_targets"][0]["unit_id"] = (
            "static_person_02"
        )
        cases["wrong_static_id"] = wrong_static_id

        for name, raw in cases.items():
            with self.subTest(name=name), self.assertRaises(
                contract.GokuFullMotionContractError
            ):
                contract.canonicalize_target_plan_model_output(raw, source)

    def test_target_model_output_never_repairs_components_or_camera(self) -> None:
        source = source_census()

        component = target_plan(source)
        component["dynamic_unit_targets"][0][
            "source_component_dispositions"
        ].append(
            copy.deepcopy(
                component["dynamic_unit_targets"][0][
                    "source_component_dispositions"
                ][0]
            )
        )
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "dispose every source motion component",
        ):
            contract.canonicalize_target_plan_model_output(component, source)

        camera = target_plan(source)
        camera["camera_target"]["target_motion_class"] = "pan_left"
        with self.assertRaises(contract.GokuFullMotionContractError):
            contract.canonicalize_target_plan_model_output(camera, source)

        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "must be an object"
        ):
            contract.canonicalize_target_plan_model_output("not JSON", source)

    def test_target_canonicalization_receipt_tampering_fails(self) -> None:
        source = source_census()
        raw = target_plan(source)
        canonical, receipt = contract.canonicalize_target_plan_model_output(
            raw, source
        )
        forged = copy.deepcopy(receipt)
        forged["canonical_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError,
            "receipt differs from reconstruction",
        ):
            contract.validate_target_plan_canonicalization(
                raw, canonical, forged, source
            )


if __name__ == "__main__":
    unittest.main()
