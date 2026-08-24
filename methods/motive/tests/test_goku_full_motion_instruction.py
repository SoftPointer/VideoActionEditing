from __future__ import annotations

import copy
import hashlib
import unittest

from motive import goku_full_motion_contract as contract
from motive import goku_full_motion_instruction as instruction


def _evidence(description: str) -> dict:
    return {
        "schema_version": contract.MOTION_EVIDENCE_SCHEMA,
        "start_frame": 0,
        "end_frame": 80,
        "description": description,
    }


def _source(*, with_static: bool = False) -> dict:
    registry = []
    units = []
    for index, side in enumerate(("left", "right"), start=1):
        entity_id = f"entity_{index:02d}"
        registry.append(
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": entity_id,
                "entity_type": "person",
                "stable_reference": f"the person on the {side}",
                "i0_bbox_xyxy_1000": (
                    [50, 350, 300, 900]
                    if side == "left"
                    else [700, 350, 950, 900]
                ),
                "viewer_region": f"center_{side}",
                "region_ordinal": 1,
                "role": "dynamic_subject",
                "visible_at_i0": True,
                "reachable_at_i0": True,
                "confidence": "high",
            }
        )
        units.append(
            {
                "schema_version": contract.SOURCE_DYNAMIC_UNIT_SCHEMA,
                "unit_id": f"unit_{index:02d}",
                "entity_id": entity_id,
                "entity_type": "person",
                "stable_reference": f"the person on the {side}",
                "visible_at_i0": True,
                "independent_motion": True,
                "i0_state": f"The {side} person holds a peace sign",
                "source_action_signature": f"peace_sign_{side}",
                "source_motion": f"moves a peace-sign gesture with the {side} hand",
                "source_motion_components": [
                    {
                        "schema_version": contract.SOURCE_MOTION_COMPONENT_SCHEMA,
                        "component_id": "component_01",
                        "component_type": "gesture",
                        "motion_signature": f"peace_sign_{side}",
                        "motion_description": (
                            f"moves a peace-sign gesture with the {side} hand"
                        ),
                        "dependent_entity_ids": [],
                        "motion_evidence": [
                            _evidence(f"the {side} person's raised hand moves")
                        ],
                    }
                ],
                "motion_evidence": [
                    _evidence(f"the {side} person's raised hand moves")
                ],
                "confidence": "high",
            }
        )
    static = []
    if with_static:
        registry.append(
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_03",
                "entity_type": "person",
                "stable_reference": "the seated observer",
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
                "stable_reference": "the seated observer",
                "visible_at_i0": True,
                "i0_state": "The seated observer rests both hands on the lap",
                "source_state": "remain_still",
                "motion_evidence": [
                    _evidence("the seated observer holds the same pose")
                ],
                "confidence": "high",
            }
        )
    return {
        "schema_version": contract.SOURCE_CENSUS_SCHEMA,
        "iid": "render-two-people-001",
        "clip": {
            "schema_version": contract.CLIP_SCHEMA,
            "frame_count": 81,
            "fps": "25/1",
            "timeline_span_seconds": 3.2,
            "single_continuous_shot": True,
        },
        "source_quality": "high",
        "scene_description": "Two standing people are framed against a wall",
        "i0_visible_entities": [
            "the person on the left",
            "the person on the right",
            *[row["stable_reference"] for row in static],
        ],
        "i0_entity_registry": registry,
        "motion_inventory_complete": True,
        "crowd_or_unresolved_motion": False,
        "diffuse_unresolved_motion": False,
        "dynamic_units": units,
        "static_salient_people": static,
        "camera": {
            "schema_version": contract.SOURCE_CAMERA_SCHEMA,
            "camera_id": "camera",
            "motion_class": "locked_off",
            "motion_signature": "locked_off",
            "motion_description": "locked off",
            "dynamic": False,
            "motion_evidence": [_evidence("the background remains aligned")],
            "confidence": "high",
        },
        "uncertainty_codes": [],
        "confidence": "high",
    }


def _plan(source: dict) -> dict:
    targets = []
    for index, side in enumerate(("left", "right"), start=1):
        entity_id = f"entity_{index:02d}"
        reference = f"the person on the {side}"
        novel = "lower the peace sign and wave with an open palm"
        targets.append(
            {
                "schema_version": contract.TARGET_DYNAMIC_UNIT_SCHEMA,
                "unit_id": f"unit_{index:02d}",
                "entity_id": entity_id,
                "stable_reference": reference,
                "target_action_signature": f"open_palm_wave_{side}",
                "motion_relation": "replace",
                "source_motion_suppressed": True,
                "explicit_shared_base_motion": None,
                "source_component_dispositions": [
                    {
                        "schema_version": (
                            contract.TARGET_COMPONENT_DISPOSITION_SCHEMA
                        ),
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
                    f"{reference} lowers the raised fingers",
                    f"{reference} opens the palm and waves",
                ],
                "interaction_entity_ids": [],
                "required_i0_entity_ids": [entity_id],
            }
        )
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
    dynamic_ids = [row["unit_id"] for row in source["dynamic_units"]]
    static_ids = [row["unit_id"] for row in source["static_salient_people"]]
    return {
        "schema_version": contract.TARGET_PLAN_SCHEMA,
        "iid": source["iid"],
        "source_census_sha256": contract.object_sha256(source),
        "dynamic_unit_targets": targets,
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


def _critic(source: dict, plan: dict, compiled: dict) -> dict:
    dynamic_ids = [row["unit_id"] for row in source["dynamic_units"]]
    static_ids = [row["unit_id"] for row in source["static_salient_people"]]
    return {
        "schema_version": contract.COVERAGE_CRITIC_SCHEMA,
        "iid": source["iid"],
        "source_census_sha256": contract.object_sha256(source),
        "target_plan_sha256": contract.object_sha256(plan),
        "instruction_sha256": compiled["instruction_sha256"],
        "required_dynamic_unit_ids": dynamic_ids,
        "plan_covered_dynamic_unit_ids": dynamic_ids,
        "instruction_covered_dynamic_unit_ids": dynamic_ids,
        "missing_unit_ids": [],
        "extra_unit_ids": [],
        "ambiguous_unit_ids": [],
        "per_unit_substantive_change": {
            unit_id: True for unit_id in dynamic_ids
        },
        "source_future_suppressed_or_explicit": {
            unit_id: True for unit_id in dynamic_ids
        },
        "camera_clause_present": True,
        "camera_target_valid": True,
        "required_static_person_ids": static_ids,
        "static_people_preserved": {
            unit_id: True for unit_id in static_ids
        },
        "i0_executable": True,
        "no_new_prerequisites": True,
        "no_unrequested_action": True,
        "verdict": "pass",
        "uncertainty_codes": [],
        "confidence": "high",
    }


class GokuFullMotionInstructionTests(unittest.TestCase):
    def test_clause_subject_references_resolve_only_by_registry_entity_id(
        self,
    ) -> None:
        source = _source(with_static=True)
        plan = _plan(source)
        registry_references = {
            row["entity_id"]: row["stable_reference"]
            for row in source["i0_entity_registry"]
        }
        for row in source["dynamic_units"]:
            row["stable_reference"] = "POISON_DYNAMIC_NON_REGISTRY_REFERENCE"
        for row in source["static_salient_people"]:
            row["stable_reference"] = "POISON_STATIC_NON_REGISTRY_REFERENCE"

        rendered = instruction._clause_sources(source, plan)
        text = "\n".join(row[3] for row in rendered)
        self.assertNotIn("POISON_", text)
        for entity_id in (
            [row["entity_id"] for row in source["dynamic_units"]]
            + [row["entity_id"] for row in source["static_salient_people"]]
        ):
            self.assertIn(registry_references[entity_id], text)

    def test_coverage_critic_binds_every_instruction_clause(self) -> None:
        source = _source(with_static=True)
        plan = _plan(source)
        compiled = instruction.compile_full_motion_instruction(source, plan)
        critic = _critic(source, plan, compiled)
        self.assertEqual(
            contract.validate_coverage_critic(
                critic,
                source_census=source,
                target_plan=plan,
                compiled_instruction=compiled,
            ),
            critic,
        )

    def test_coverage_critic_cannot_omit_right_person_or_camera(self) -> None:
        source = _source()
        plan = _plan(source)
        compiled = instruction.compile_full_motion_instruction(source, plan)
        missing_right = _critic(source, plan, compiled)
        missing_right["instruction_covered_dynamic_unit_ids"] = ["unit_01"]
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "unit_02"
        ):
            contract.validate_coverage_critic(
                missing_right,
                source_census=source,
                target_plan=plan,
                compiled_instruction=compiled,
            )

        missing_camera = _critic(source, plan, compiled)
        missing_camera["camera_clause_present"] = False
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "must be exactly true"
        ):
            contract.validate_coverage_critic(
                missing_camera,
                source_census=source,
                target_plan=plan,
                compiled_instruction=compiled,
            )

    def test_two_person_renderer_is_deterministic_and_complete(self) -> None:
        source = _source()
        plan = _plan(source)
        first = instruction.compile_full_motion_instruction(source, plan)
        second = instruction.compile_full_motion_instruction(
            copy.deepcopy(source), copy.deepcopy(plan)
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["ordered_clause_ids"],
            ["dynamic:unit_01", "dynamic:unit_02", "camera:camera"],
        )
        self.assertEqual(
            set(first["entity_clauses"]), {"unit_01", "unit_02"}
        )
        self.assertIn("the person on the left", first["edit_instruction"])
        self.assertIn("the person on the right", first["edit_instruction"])
        self.assertTrue(first["edit_instruction"].endswith("camera locked off."))
        self.assertEqual(
            first["instruction_sha256"],
            hashlib.sha256(first["edit_instruction"].encode("utf-8")).hexdigest(),
        )
        instruction.validate_compiled_instruction(
            first, source_census=source, target_plan=plan
        )

    def test_i0_entity_marker_resolves_to_registry_stable_reference(self) -> None:
        source = _source()
        object_reference = "the red ball beside the left person's foot"
        source["i0_visible_entities"].append(object_reference)
        source["i0_entity_registry"].append(
            {
                "schema_version": contract.SOURCE_I0_ENTITY_SCHEMA,
                "entity_id": "entity_03",
                "entity_type": "rigid_object",
                "stable_reference": object_reference,
                "i0_bbox_xyxy_1000": [100, 750, 250, 900],
                "viewer_region": "lower_left",
                "region_ordinal": 1,
                "role": "passive_interaction_object",
                "visible_at_i0": True,
                "reachable_at_i0": True,
                "confidence": "high",
            }
        )
        plan = _plan(source)
        target = plan["dynamic_unit_targets"][0]
        target.update(
            {
                "novel_target_motion": (
                    "lower the raised hand, then pick up [[entity_03]] and "
                    "hold it near the torso"
                ),
                "target_clause": (
                    "have the left person lower the hand and lift the visible "
                    "ball"
                ),
                "ordered_stages": [
                    "the left person lowers the raised fingers",
                    "the left person picks up [[entity_03]] and holds it",
                ],
                "interaction_entity_ids": ["entity_03"],
                "required_i0_entity_ids": ["entity_01", "entity_03"],
            }
        )

        compiled = instruction.compile_full_motion_instruction(source, plan)
        clause = compiled["entity_clauses"]["unit_01"]
        self.assertIn(object_reference, clause)
        self.assertNotIn("[[entity_03]]", clause)
        self.assertNotIn("[[entity_03]]", compiled["edit_instruction"])
        instruction.validate_compiled_instruction(
            compiled,
            source_census=source,
            target_plan=plan,
        )

    def test_model_target_clause_is_not_executable_instruction_text(self) -> None:
        source = _source()
        plan = _plan(source)
        plan["dynamic_unit_targets"][0]["target_clause"] = (
            "have the person on the left perform a broad friendly hand gesture"
        )
        compiled = instruction.compile_full_motion_instruction(source, plan)
        self.assertIn(
            "Have the person on the left perform this complete target motion: "
            "lower the peace sign and wave with an open palm",
            compiled["edit_instruction"],
        )
        self.assertNotIn(
            "perform a broad friendly hand gesture",
            compiled["edit_instruction"],
        )
        self.assertFalse(
            compiled["compiler_contract"]["model_target_clause_executable"]
        )

    def test_static_and_camera_model_clauses_are_also_non_executable(self) -> None:
        source = _source(with_static=True)
        plan = _plan(source)
        plan["static_person_targets"][0]["target_clause"] = (
            "please have the seated observer remain still"
        )
        plan["camera_target"]["target_clause"] = (
            "please keep the camera locked off"
        )
        compiled = instruction.compile_full_motion_instruction(source, plan)
        self.assertIn(
            "Have the seated observer remain still", compiled["edit_instruction"]
        )
        self.assertIn("Keep the camera locked off", compiled["edit_instruction"])
        self.assertNotIn("please", compiled["edit_instruction"])

    def test_complete_noun_and_from_i0_prose_use_labelled_replace_template(
        self,
    ) -> None:
        source = _source()
        horse_reference = "the chestnut horse in the center"
        source["dynamic_units"][0].update(
            {
                "entity_type": "animal",
                "stable_reference": horse_reference,
                "i0_state": "The horse stands with all four hooves grounded",
                "source_action_signature": "stand_and_shift_weight",
                "source_motion": "shifts its weight while standing in place",
                "motion_evidence": [
                    _evidence("the horse shifts its body weight between hooves")
                ],
            }
        )
        source["i0_entity_registry"][0].update(
            {
                "entity_type": "animal",
                "stable_reference": horse_reference,
            }
        )
        source["i0_visible_entities"][0] = horse_reference
        source["scene_description"] = (
            "A chestnut horse and a person are visible in the same shot"
        )
        plan = _plan(source)
        noun_start_raw = (
            "the horse begins by lifting its front right hoof, then steps "
            "forward twice."
        )
        noun_start = noun_start_raw[:-1]
        from_i0_start_raw = (
            "from I0, the right person opens the raised fingers and completes "
            "two side-to-side waves!"
        )
        from_i0_start = from_i0_start_raw[:-1]
        plan["dynamic_unit_targets"][0].update(
            {
                "stable_reference": horse_reference,
                "novel_target_motion": noun_start_raw,
                "target_clause": (
                    "have the horse lift one hoof and step forward."
                ),
                "ordered_stages": [
                    "the chestnut horse lifts its front right hoof",
                    "the chestnut horse steps forward twice",
                ],
                "required_i0_entity_ids": ["entity_01"],
            }
        )
        plan["dynamic_unit_targets"][1]["novel_target_motion"] = (
            from_i0_start_raw
        )
        plan["dynamic_unit_targets"][1]["target_clause"] = (
            "have the right person open the hand and wave twice from I0!"
        )

        raw_plan = copy.deepcopy(plan)
        compiled = instruction.compile_full_motion_instruction(source, plan)
        self.assertEqual(plan, raw_plan)
        self.assertEqual(
            compiled["target_plan_sha256"], contract.object_sha256(raw_plan)
        )
        left = compiled["entity_clauses"]["unit_01"]
        right = compiled["entity_clauses"]["unit_02"]
        self.assertEqual(
            left,
            "Have the chestnut horse in the center perform this complete "
            "target motion: " + noun_start,
        )
        self.assertEqual(
            right,
            "Have the person on the right perform this complete target motion: "
            + from_i0_start,
        )
        self.assertIn(noun_start, compiled["edit_instruction"])
        self.assertIn(from_i0_start, compiled["edit_instruction"])
        self.assertIn("Keep the camera locked off", compiled["edit_instruction"])
        self.assertNotIn("Have the chestnut horse in the center the horse", left)
        instruction.validate_compiled_instruction(
            compiled, source_census=source, target_plan=plan
        )

    def test_shared_base_and_novel_action_are_separately_labelled(self) -> None:
        source = _source()
        plan = _plan(source)
        target = plan["dynamic_unit_targets"][0]
        shared_raw = "walk forward from the exact visible stance."
        shared = shared_raw[:-1]
        novel_raw = "from I0, the left hand opens and waves overhead twice:"
        novel = novel_raw[:-1]
        _set_shared_base(target, shared_raw)
        target["novel_target_motion"] = novel_raw
        target["target_clause"] = (
            "have the left person walk forward while waving overhead"
        )

        compiled = instruction.compile_full_motion_instruction(source, plan)
        clause = compiled["entity_clauses"]["unit_01"]
        self.assertEqual(
            clause,
            "Have the person on the left perform this explicitly specified "
            "base motion: walk forward from the exact visible stance. "
            "Concurrently, have the person on the left perform this complete "
            "novel action: from I0, the left hand opens and waves overhead "
            "twice",
        )
        self.assertIn(shared, clause)
        self.assertIn(novel, clause)
        instruction.validate_compiled_instruction(
            compiled, source_census=source, target_plan=plan
        )

    def test_dynamic_camera_terminal_period_is_rendered_once(self) -> None:
        source = _source()
        source["camera"].update(
            {
                "motion_class": "pan_left",
                "motion_signature": "slow_pan_left",
                "motion_description": "slow pan left",
                "dynamic": True,
                "motion_evidence": [
                    _evidence("the background shifts right across the clip")
                ],
            }
        )
        plan = _plan(source)
        plan["camera_target"].update(
            {
                "motion_relation": "replace_motion",
                "target_motion_class": "dolly_in",
                "target_motion_signature": "steady_dolly_in",
                "target_motion_description": (
                    "a steady forward dolly toward the actors."
                ),
                "target_clause": "a gradual move closer to the actors.",
                "source_motion_suppressed": True,
                "substantive_change": True,
            }
        )
        raw_plan = copy.deepcopy(plan)

        compiled = instruction.compile_full_motion_instruction(source, plan)

        self.assertEqual(plan, raw_plan)
        self.assertEqual(
            compiled["target_plan_sha256"], contract.object_sha256(raw_plan)
        )
        self.assertEqual(
            compiled["camera_clause"],
            "Set the camera trajectory to a steady forward dolly toward the "
            "actors",
        )
        self.assertTrue(
            compiled["edit_instruction"].endswith(
                "Set the camera trajectory to a steady forward dolly toward "
                "the actors."
            )
        )
        self.assertNotIn("..", compiled["edit_instruction"])
        self.assertNotIn(".;", compiled["edit_instruction"])
        self.assertNotIn(". ;", compiled["edit_instruction"])
        instruction.validate_compiled_instruction(
            compiled, source_census=source, target_plan=plan
        )

    def test_clause_character_and_byte_spans_are_exact(self) -> None:
        source = _source()
        plan = _plan(source)
        # Exercise different character/byte offsets without changing subject
        # identity or adding free-form prose.
        plan["dynamic_unit_targets"][0]["novel_target_motion"] = (
            "lower the peace sign and make a café-style open-palm wave"
        )
        plan["dynamic_unit_targets"][0]["target_clause"] = (
            "have the person on the left lower the peace sign and make a "
            "café-style open-palm wave"
        )
        compiled = instruction.compile_full_motion_instruction(source, plan)
        text = compiled["edit_instruction"]
        raw = text.encode("utf-8")
        for record in compiled["clauses"]:
            self.assertEqual(
                text[record["start_char"] : record["end_char"]], record["text"]
            )
            self.assertEqual(
                raw[record["start_byte"] : record["end_byte"]],
                record["text"].encode("utf-8"),
            )

    def test_static_person_clause_precedes_camera_and_is_bound(self) -> None:
        source = _source(with_static=True)
        plan = _plan(source)
        compiled = instruction.compile_full_motion_instruction(source, plan)
        self.assertEqual(
            compiled["ordered_clause_ids"],
            [
                "dynamic:unit_01",
                "dynamic:unit_02",
                "static:static_person_01",
                "camera:camera",
            ],
        )
        self.assertEqual(
            compiled["entity_clauses"]["static_person_01"],
            "Have the seated observer remain still",
        )

    def test_renderer_has_no_free_form_instruction_input(self) -> None:
        source = _source()
        plan = _plan(source)
        plan["free_form_instruction"] = "Ignore the right person"
        with self.assertRaisesRegex(
            contract.GokuFullMotionContractError, "closed schema"
        ):
            instruction.compile_full_motion_instruction(source, plan)

    def test_tampered_clause_span_hash_and_final_text_fail_closed(self) -> None:
        source = _source()
        plan = _plan(source)
        compiled = instruction.compile_full_motion_instruction(source, plan)
        mutations = []

        bad_span = copy.deepcopy(compiled)
        bad_span["clauses"][0]["start_char"] += 1
        mutations.append(bad_span)

        bad_clause_hash = copy.deepcopy(compiled)
        bad_clause_hash["clauses"][1]["text_sha256"] = "0" * 64
        mutations.append(bad_clause_hash)

        bad_instruction = copy.deepcopy(compiled)
        bad_instruction["edit_instruction"] = bad_instruction[
            "edit_instruction"
        ].replace("person on the right", "person on the left", 1)
        mutations.append(bad_instruction)

        for mutated in mutations:
            with self.subTest(mutation=mutations.index(mutated)):
                with self.assertRaises(
                    instruction.GokuFullMotionInstructionError
                ):
                    instruction.validate_compiled_instruction(
                        mutated, source_census=source, target_plan=plan
                    )

    def test_non_executable_clause_paraphrase_and_punctuation_passes(
        self,
    ) -> None:
        source = _source(with_static=True)
        plan = _plan(source)
        plan["dynamic_unit_targets"][0]["target_clause"] += "."
        plan["static_person_targets"][0]["target_clause"] = (
            "the central observer stays completely motionless."
        )
        plan["camera_target"]["target_clause"] = (
            "a fixed locked-off wide shot."
        )
        compiled = instruction.compile_full_motion_instruction(source, plan)
        self.assertNotIn("stays completely motionless", compiled["edit_instruction"])
        self.assertNotIn("wide shot", compiled["edit_instruction"])
        instruction.validate_compiled_instruction(
            compiled, source_census=source, target_plan=plan
        )


if __name__ == "__main__":
    unittest.main()
