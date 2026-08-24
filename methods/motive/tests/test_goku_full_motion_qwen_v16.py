from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive.goku_full_motion_qwen_v16 import (
    COMPILED_INSTRUCTION_SCHEMA,
    GokuFullMotionQwenV16Error,
    PASSED_SCHEMA,
    ROW_RECEIPT_SCHEMA,
    SOURCE_CENSUS_SCHEMA,
    SOURCE_CAMERA_SCHEMA,
    SOURCE_SUBJECT_SCHEMA,
    TARGET_CAMERA_SCHEMA,
    TARGET_COVERAGE_SCHEMA,
    TARGET_PLAN_SCHEMA,
    TARGET_PLAN_SYSTEM,
    TARGET_SUBJECT_SCHEMA,
    _loads_object,
    build_parser,
    build_target_plan_prompt,
    canonicalize_source_census,
    canonicalize_target_plan,
    compile_instruction,
    object_sha256,
    run_one,
    run_worker,
    validate_compiled_instruction,
    validate_passed_row,
    validate_source_census,
    validate_target_plan,
)


def _evidence(description: str) -> dict:
    return {
        "start_frame": 0,
        "end_frame": 80,
        "description": description,
    }


def _census(iid: str = "sample01", *, mechanical: bool = False) -> dict:
    value = {
        "iid": iid,
        "dynamic_subjects": [
            {
                "subject_id": "subject_01",
                "entity_type": "person",
                "stable_reference": "the man in a blue shirt on viewer-left",
                "i0_bbox_xyxy_1000": [40, 130, 450, 970],
                "i0_state": "standing with both hands beside his waist",
                "source_action_signature": "raise_left_hand_peace_sign",
                "source_motion": (
                    "raises his left hand from his waist and forms a peace sign"
                ),
                "motion_evidence": [
                    _evidence("his left hand rises and two fingers extend")
                ],
                "dynamic": True,
            },
            {
                "subject_id": "subject_02",
                "entity_type": "person",
                "stable_reference": "the woman in black on viewer-right",
                "i0_bbox_xyxy_1000": [520, 120, 960, 980],
                "i0_state": "standing with her right hand lowered",
                "source_action_signature": "wave_right_hand",
                "source_motion": "raises her right hand and waves side to side",
                "motion_evidence": [
                    _evidence("her right hand rises and oscillates laterally")
                ],
                "dynamic": True,
            },
        ],
        "camera": {
            "motion_class": "pan_left",
            "source_motion": "the camera pans steadily left across the scene",
            "motion_evidence": [
                _evidence("the full background shifts right across ordered frames")
            ],
        },
        "all_dynamic_subjects_enumerated": True,
        "crowd_or_unresolved_motion": False,
        "confidence": "high",
    }
    if not mechanical:
        value["schema_version"] = SOURCE_CENSUS_SCHEMA
        for subject in value["dynamic_subjects"]:
            subject["schema_version"] = SOURCE_SUBJECT_SCHEMA
            for evidence in subject["motion_evidence"]:
                evidence["schema_version"] = (
                    "motive-goku-full-motion-v16-evidence-v1"
                )
        value["camera"]["schema_version"] = SOURCE_CAMERA_SCHEMA
        for evidence in value["camera"]["motion_evidence"]:
            evidence["schema_version"] = (
                "motive-goku-full-motion-v16-evidence-v1"
            )
    return value


def _plan(iid: str = "sample01", *, reverse: bool = False, mechanical: bool = False) -> dict:
    targets = [
        {
            "subject_id": "subject_01",
            "target_action_signature": "clap_both_hands_overhead",
            "target_motion": (
                "immediately raises both hands above his head and repeatedly claps"
            ),
            "substantive_change": True,
        },
        {
            "subject_id": "subject_02",
            "target_action_signature": "crouch_and_touch_floor",
            "target_motion": (
                "immediately bends both knees into a crouch and touches the floor"
            ),
            "substantive_change": True,
        },
    ]
    if reverse:
        targets.reverse()
    value = {
        "iid": iid,
        "dynamic_subject_targets": targets,
        "camera_target": {
            "relation": "replace_motion",
            "motion_class": "locked_off",
            "target_motion": "the camera remains completely locked off",
        },
        "coverage": {
            "dynamic_subject_ids": ["subject_01", "subject_02"],
            "camera_covered": True,
        },
        "confidence": "high",
    }
    if not mechanical:
        value["schema_version"] = TARGET_PLAN_SCHEMA
        for target in value["dynamic_subject_targets"]:
            target["schema_version"] = TARGET_SUBJECT_SCHEMA
        value["camera_target"]["schema_version"] = TARGET_CAMERA_SCHEMA
        value["coverage"]["schema_version"] = TARGET_COVERAGE_SCHEMA
    return value


def _input_row(iid: str = "sample01") -> dict:
    return {
        "iid": iid,
        "group_id": "group01",
        "family": "people",
        "src_video": "source.mp4",
        "resolved_src_video": "/fake/source.mp4",
        "source_caption": "Two people gesture.",
        "edited_caption": "Two people perform different actions.",
        "prompt": "Change both people's actions.",
        "anchor_image": "anchor.png",
        "resolved_anchor_image": "/fake/anchor.png",
        "anchor_sha256": "a" * 64,
        "source_video_sha256": "b" * 64,
        "prefilter_score": 9.0,
        "media": {},
        "motion": {},
    }


class _Backend:
    model_path = "/fake/Qwen3-VL-32B-Instruct"
    model_revision = "test-revision"
    transformers_version = "test"

    def __init__(self, census: dict, plan: dict) -> None:
        self.census = census
        self.plan = plan
        self.calls: list[str] = []

    def generate_source_motion_census_v16(self, **kwargs):
        self.calls.append("census")
        raw = "```json\n" + json.dumps(self.census) + "\n```"
        return raw, kwargs["expected_visual_input_digest"]

    def generate_target_plan_v16(self, **kwargs):
        self.calls.append("target")
        return json.dumps(self.plan), kwargs["expected_visual_input_digest"]


def _args(input_path: Path, output_root: Path, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=input_path,
        output_root=output_root,
        model="/fake/model",
        root=root,
        row_index=0,
        num_rows=1,
        max_new_tokens=4096,
        nframes=16,
        max_pixels=2_359_296,
        tile_width=512,
        mosaic_columns=4,
        attn_implementation="sdpa",
        allow_download=False,
        allow_errors=False,
    )


def _worker_args(
    input_path: Path,
    output_root: Path,
    root: Path,
    *,
    num_rows: int,
    worker_index: int = 0,
    num_workers: int = 1,
    allow_errors: bool = True,
) -> argparse.Namespace:
    args = _args(input_path, output_root, root)
    args.row_index = None
    args.worker_index = worker_index
    args.num_workers = num_workers
    args.num_rows = num_rows
    args.allow_errors = allow_errors
    return args


def _fake_prepare(row, *, root, runtime):
    return (
        root / f"{row['iid']}.mp4",
        root / f"{row['iid']}.png",
        {
            "exact_i0": True,
            "temporal_geometry": {
                "frame_count": 81,
                "fps": "25/1",
                "timeline_span_seconds": 3.2,
                "width": 1280,
                "height": 720,
            },
        },
        (object(), object(), object(), object(), object()),
        "f" * 64,
    )


class _AdaptiveBackend:
    model_path = "/fake/Qwen3-VL-32B-Instruct"
    model_revision = "test-revision"
    transformers_version = "test"

    def __init__(
        self,
        *,
        semantic_failure_iids: set[str] | None = None,
        infrastructure_failure_iids: set[str] | None = None,
    ) -> None:
        self.semantic_failure_iids = semantic_failure_iids or set()
        self.infrastructure_failure_iids = infrastructure_failure_iids or set()
        self.current_iid: str | None = None
        self.calls: list[tuple[str, str]] = []

    def generate_source_motion_census_v16(self, **kwargs):
        marker = "Annotate iid='"
        user = kwargs["user"]
        iid = user.split(marker, 1)[1].split("'", 1)[0]
        self.current_iid = iid
        self.calls.append(("census", iid))
        if iid in self.infrastructure_failure_iids:
            raise RuntimeError(f"simulated backend failure for {iid}")
        return (
            json.dumps(_census(iid, mechanical=True)),
            kwargs["expected_visual_input_digest"],
        )

    def generate_target_plan_v16(self, **kwargs):
        assert self.current_iid is not None
        iid = self.current_iid
        self.calls.append(("target", iid))
        plan = _plan(iid, mechanical=True)
        if iid in self.semantic_failure_iids:
            plan["dynamic_subject_targets"] = plan["dynamic_subject_targets"][:1]
        return json.dumps(plan), kwargs["expected_visual_input_digest"]


class GokuFullMotionQwenV16Tests(unittest.TestCase):
    def _validated_census(self) -> dict:
        raw = _census()
        value, _ = canonicalize_source_census(raw, expected_iid="sample01")
        return validate_source_census(value, expected_iid="sample01")

    def _validated_plan(self, census: dict, *, reverse: bool = False) -> dict:
        raw = _plan(reverse=reverse)
        value, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        return validate_target_plan(
            value, expected_iid="sample01", source_census=census
        )

    def test_mechanical_repair_is_local_and_closed(self) -> None:
        raw, parse_repairs = _loads_object(
            "```json\n" + json.dumps(_census(mechanical=True)) + "\n```",
            stage="census",
        )
        canonical, receipt = canonicalize_source_census(
            raw, expected_iid="sample01"
        )
        validated = validate_source_census(canonical, expected_iid="sample01")
        self.assertEqual(parse_repairs, ["removed_markdown_json_fence"])
        self.assertFalse(receipt["semantic_fields_invented"])
        self.assertEqual(validated["schema_version"], SOURCE_CENSUS_SCHEMA)
        self.assertEqual(
            validated["dynamic_subjects"][0]["schema_version"],
            SOURCE_SUBJECT_SCHEMA,
        )

    def test_literal_entity_nouns_map_to_closed_ontology_parents(self) -> None:
        cases = {
            "dog": "animal",
            "bird": "animal",
            "bear": "animal",
            "boat": "vehicle",
            "water": "fluid_or_emitter",
            "clouds": "fluid_or_emitter",
            "human": "person",
        }
        for literal, expected in cases.items():
            with self.subTest(literal=literal):
                raw = _census()
                raw["dynamic_subjects"] = [raw["dynamic_subjects"][0]]
                raw["dynamic_subjects"][0]["entity_type"] = literal
                canonical, receipt = canonicalize_source_census(
                    raw, expected_iid="sample01"
                )
                validated = validate_source_census(
                    canonical, expected_iid="sample01"
                )
                self.assertEqual(
                    validated["dynamic_subjects"][0]["entity_type"], expected
                )
                self.assertIn(
                    "normalized_subject_01.entity_type",
                    receipt["operations"],
                )
                self.assertFalse(receipt["semantic_fields_invented"])

    def test_semantic_completeness_flags_are_never_repaired(self) -> None:
        raw = _census(mechanical=True)
        del raw["all_dynamic_subjects_enumerated"]
        canonical, receipt = canonicalize_source_census(
            raw, expected_iid="sample01"
        )
        self.assertNotIn("all_dynamic_subjects_enumerated", canonical)
        self.assertFalse(receipt["semantic_fields_invented"])
        with self.assertRaisesRegex(GokuFullMotionQwenV16Error, "keys differ"):
            validate_source_census(canonical, expected_iid="sample01")

    def test_target_reordered_to_source_and_compiler_covers_all_motion(self) -> None:
        census = self._validated_census()
        plan = self._validated_plan(census, reverse=True)
        self.assertEqual(
            [item["subject_id"] for item in plan["dynamic_subject_targets"]],
            ["subject_01", "subject_02"],
        )
        compiled = compile_instruction(census, plan)
        self.assertEqual(compiled["schema_version"], COMPILED_INSTRUCTION_SCHEMA)
        self.assertEqual(
            compiled["covered_dynamic_subject_ids"],
            ["subject_01", "subject_02"],
        )
        self.assertTrue(compiled["camera_covered"])
        self.assertEqual(
            [item["kind"] for item in compiled["clauses"]],
            ["dynamic_subject", "dynamic_subject", "preservation", "camera"],
        )
        self.assertIn("the man in a blue shirt on viewer-left", compiled["instruction"])
        self.assertIn("the woman in black on viewer-right", compiled["instruction"])
        self.assertIn("Preserve every subject's identity and appearance", compiled["instruction"])
        self.assertTrue(compiled["instruction"].endswith("camera locked off."))

    def test_missing_dynamic_target_is_not_repaired(self) -> None:
        census = self._validated_census()
        raw = _plan(mechanical=True)
        raw["dynamic_subject_targets"] = raw["dynamic_subject_targets"][:1]
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "exactly one target"
        ):
            validate_target_plan(
                canonical, expected_iid="sample01", source_census=census
            )

    def test_moving_camera_cannot_be_implicitly_preserved(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["camera_target"] = {
            "schema_version": TARGET_CAMERA_SCHEMA,
            "relation": "preserve_static",
            "motion_class": "pan_left",
            "target_motion": "the camera pans steadily left across the scene",
        }
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "replacement trajectory"
        ):
            validate_target_plan(
                canonical, expected_iid="sample01", source_census=census
            )

    def test_replace_camera_relation_alias_is_mechanically_normalized(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["camera_target"].update(
            {
                "relation": "replace_zoom_with_static",
                "motion_class": "locked_off",
                "target_motion": (
                    "the camera remains completely locked off with no zoom or pan"
                ),
            }
        )
        original_motion = raw["camera_target"]["target_motion"]
        canonical, repair = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        self.assertEqual(canonical["camera_target"]["relation"], "replace_motion")
        self.assertEqual(canonical["camera_target"]["target_motion"], original_motion)
        self.assertIn(
            "normalized_camera.relation_replace_alias_to_replace_motion",
            repair["operations"],
        )
        validate_target_plan(
            canonical, expected_iid="sample01", source_census=census
        )

    def test_explicit_locked_target_corrects_contradictory_preserve_enum(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["camera_target"].update(
            {
                "relation": "preserve_static",
                "motion_class": "locked_off",
                "target_motion": (
                    "the camera stays fixed at the initial framing with no "
                    "camera motion"
                ),
            }
        )
        original_motion = raw["camera_target"]["target_motion"]
        canonical, repair = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        self.assertEqual(canonical["camera_target"]["relation"], "replace_motion")
        self.assertEqual(canonical["camera_target"]["target_motion"], original_motion)
        self.assertIn(
            "normalized_camera.relation_preserve_static_to_replace_motion_"
            "from_explicit_locked_target",
            repair["operations"],
        )
        validate_target_plan(
            canonical, expected_iid="sample01", source_census=census
        )

    def test_locked_camera_prose_cannot_hide_affirmative_motion(self) -> None:
        census = self._validated_census()
        motions = (
            "there is no zoom, but the camera pans steadily right",
            "the camera remains completely locked off while the shot pushes slowly in",
            "the camera remains completely locked off while framing slides steadily right",
            "the camera remains completely locked off while the viewpoint sweeps left",
            "the camera remains completely locked off while its position translates smoothly right",
            "the camera remains completely locked off while the shot glides smoothly right",
            "the camera stays fixed while the view creeps slowly forward",
            "the camera remains completely locked off while framing sways from side to side",
            "the camera remains completely locked off while the shot bobs gently up and down",
            "the camera remains completely locked off while the view closes in on the subject",
            "the camera remains completely locked off while framing widens steadily",
        )
        for relation in ("preserve_static", "replace_motion"):
            for motion in motions:
                with self.subTest(relation=relation, motion=motion):
                    raw = _plan()
                    raw["camera_target"].update(
                        {
                            "relation": relation,
                            "motion_class": "locked_off",
                            "target_motion": motion,
                        }
                    )
                    canonical, _ = canonicalize_target_plan(
                        raw, expected_iid="sample01", source_census=census
                    )
                    expected_error = (
                        "replacement trajectory"
                        if relation == "preserve_static"
                        else "locked_off target camera requires explicit"
                    )
                    with self.assertRaisesRegex(
                        GokuFullMotionQwenV16Error, expected_error
                    ):
                        validate_target_plan(
                            canonical,
                            expected_iid="sample01",
                            source_census=census,
                        )

    def test_moving_camera_class_rejects_locked_or_motionless_prose(self) -> None:
        census = self._validated_census()
        for motion, error in (
            (
                "the camera remains completely locked off, but pans steadily right",
                "contradicts locked-camera prose",
            ),
            (
                "the composition remains visually coherent throughout",
                "requires explicit camera-motion prose",
            ),
        ):
            with self.subTest(motion=motion):
                raw = _plan()
                raw["camera_target"].update(
                    {
                        "relation": "replace_motion",
                        "motion_class": "pan_right",
                        "target_motion": motion,
                    }
                )
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(
                    GokuFullMotionQwenV16Error, error
                ):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_missing_camera_relation_cannot_hide_class_prose_conflict(self) -> None:
        census = self._validated_census()
        raw = _plan()
        del raw["camera_target"]["relation"]
        raw["camera_target"].update(
            {
                "motion_class": "locked_off",
                "target_motion": "the camera pans steadily right throughout",
            }
        )
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error,
            "locked_off target camera requires explicit",
        ):
            validate_target_plan(
                canonical, expected_iid="sample01", source_census=census
            )

    def test_ambiguous_camera_target_never_repairs_preserve_static(self) -> None:
        census = self._validated_census()
        for motion in (
            "keep the framing visually stable",
            "preserve the camera presentation",
            "the camera is not fixed but may settle",
        ):
            with self.subTest(motion=motion):
                raw = _plan()
                raw["camera_target"].update(
                    {
                        "relation": "preserve_static",
                        "motion_class": "locked_off",
                        "target_motion": motion,
                    }
                )
                canonical, repair = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                self.assertEqual(
                    canonical["camera_target"]["relation"], "preserve_static"
                )
                self.assertFalse(
                    any(
                        "preserve_static_to_replace_motion" in operation
                        for operation in repair["operations"]
                    )
                )
                with self.assertRaisesRegex(
                    GokuFullMotionQwenV16Error, "replacement trajectory"
                ):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_unknown_replace_relation_alias_remains_invalid(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["camera_target"].update(
            {
                "relation": "replace_behavior_with_static",
                "motion_class": "locked_off",
                "target_motion": "the camera remains completely locked off",
            }
        )
        canonical, repair = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        self.assertEqual(
            canonical["camera_target"]["relation"],
            "replace_behavior_with_static",
        )
        self.assertNotIn(
            "normalized_camera.relation_replace_alias_to_replace_motion",
            repair["operations"],
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "camera relation is invalid"
        ):
            validate_target_plan(
                canonical, expected_iid="sample01", source_census=census
            )

    def test_source_future_shortcut_is_rejected(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["dynamic_subject_targets"][0]["target_motion"] = (
            "continues the original motion while smiling"
        )
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "unavailable source future"
        ):
            validate_target_plan(
                canonical, expected_iid="sample01", source_census=census
            )

    def test_visible_same_gravel_path_is_not_source_future_anaphora(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["dynamic_subject_targets"][0]["target_motion"] = (
            "immediately turns left and walks along the same gravel path "
            "toward the visible gate"
        )
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        validated = validate_target_plan(
            canonical, expected_iid="sample01", source_census=census
        )
        self.assertIn(
            "same gravel path",
            validated["dynamic_subject_targets"][0]["target_motion"],
        )

    def test_target_motion_rejects_time_beyond_three_point_two_seconds(self) -> None:
        census = self._validated_census()
        motions = (
            "immediately begins one full rotation and completes it in 8 seconds",
            "reaches toward the water in 6 seconds, then remains submerged for 3 seconds",
            "immediately spins for eight seconds and stops facing forward",
            "immediately turns in place for 3201 milliseconds and then stops",
            "immediately waves continuously for 3.21s and lowers his arm",
            "holds still for one full minute",
            "waves continuously for five whole seconds",
        )
        for motion in motions:
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(
                    GokuFullMotionQwenV16Error,
                    "exceeds the 3.2-second target timeline",
                ):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_target_motion_rejects_out_of_range_frame_timing(self) -> None:
        census = self._validated_census()
        for motion in (
            "immediately turns left and completes the turn by frame 81",
            "immediately waves for 81 frames and then lowers his arm",
            "immediately turns left and completes the turn by the 81st frame",
            "immediately waves for eighty-one frames and then lowers his arm",
            "immediately turns left and completes the turn at frame index: 81",
            "immediately turns left and completes the turn at frame no. 81",
            "moves at frame indices 0, 40, and 81",
            "immediately completes the turn at the eighty-first frame",
            "immediately stops at the 81-frame mark",
        ):
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(
                    GokuFullMotionQwenV16Error,
                    "(?:outside integer frame indices 0..80|exceeds the 80-frame)",
                ):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_target_motion_accepts_exact_timeline_boundaries(self) -> None:
        census = self._validated_census()
        for motion in (
            "immediately rotates for 3.2 seconds and stops facing forward",
            "immediately waves for 3 seconds and stops at frame 80",
            "immediately moves continuously for 80 frames and then stops",
            "immediately moves continuously for eighty frames and then stops",
            "immediately turns left and completes the turn at frame 80",
            "immediately turns left and completes the turn by the 80th frame",
            "completes one moderate turn within all 81 frames",
        ):
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                validated = validate_target_plan(
                    canonical,
                    expected_iid="sample01",
                    source_census=census,
                )
                self.assertEqual(
                    validated["dynamic_subject_targets"][0]["target_motion"],
                    motion,
                )

    def test_target_motion_rejects_cumulative_stage_overflow(self) -> None:
        census = self._validated_census()
        for motion in (
            "claps for 2 seconds, then waves for 2 seconds",
            "waves for 3 seconds, then claps for 3 seconds",
            "waits until frame 50, then waves for 40 frames",
            "waves for 50 frames, then claps for 40 frames",
            "After she waves for 2 seconds, she claps for 2 seconds",
            "Immediately after he waves for 2 seconds, he claps for 2 seconds",
            "waves for 2 seconds before clapping for 2 seconds",
            "waves for 2 seconds, after which claps for 2 seconds",
            "waves for 2 seconds, two seconds later claps for 1 second",
            "First, waves for 2 seconds. Second, claps for 2 seconds",
            "waves for 2 seconds prior to clapping for 2 seconds",
            "waves for 2 seconds followed immediately with clapping for 2 seconds",
        ):
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(
                    GokuFullMotionQwenV16Error,
                    "explicit stages require at least",
                ):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_parallel_durations_are_not_added_across_motion_or_subjects(self) -> None:
        census = self._validated_census()
        raw = _plan()
        motions = (
            "waves for 2 seconds while running after the ball for 2 seconds",
            "crouches for 3 seconds while holding both arms forward",
        )
        for target, motion in zip(
            raw["dynamic_subject_targets"], motions, strict=True
        ):
            target["target_motion"] = motion
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        validated = validate_target_plan(
            canonical, expected_iid="sample01", source_census=census
        )
        self.assertEqual(
            [item["target_motion"] for item in validated["dynamic_subject_targets"]],
            list(motions),
        )
        compiled = compile_instruction(census, validated)
        self.assertEqual(compiled["instruction"].count("for 2 seconds"), 2)
        self.assertEqual(compiled["instruction"].count("for 3 seconds"), 1)

    def test_frame_count_and_frame_index_have_distinct_limits(self) -> None:
        census = self._validated_census()
        for motion, error in (
            ("completes between frames 0 and 81", "frame indices 0..80"),
            ("completes at frame index #81", "frame indices 0..80"),
            ("completes at frame number eighty-first", "frame indices 0..80"),
            ("completes at timestep 81", "frame indices 0..80"),
            ("completes within all 82 frames", "exceeds the 81-frame clip"),
            ("begins at frame -1", "negative temporal amount"),
            (
                "waits one frame after frame 80 before moving",
                "delayed motion.*exceeds the 3.2-second",
            ),
        ):
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(GokuFullMotionQwenV16Error, error):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_sampling_aliases_and_excessive_complexity_are_rejected(self) -> None:
        census = self._validated_census()
        cases = (
            ("raises both arms at F20", "source-view sample marker"),
            ("raises both arms at F_20", "source-view sample marker"),
            ("starts turning at C0", "source-view sample marker"),
            ("raises both arms at S 5", "source-view sample marker"),
            (
                "raises both arms, then spins, then jumps, finally crouches",
                "too many sequential stages",
            ),
            ("turns in place 7 cycles", "repetition count"),
        )
        for motion, error in cases:
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(GokuFullMotionQwenV16Error, error):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_noncanonical_or_additive_timing_cannot_bypass_limits(self) -> None:
        census = self._validated_census()
        cases = (
            ("spins for 3 and 1/2 seconds", "unsupported temporal notation"),
            ("spins for 3½ seconds", "unsupported temporal notation"),
            ("spins for 3,500 milliseconds", "unsupported temporal notation"),
            ("spins for four and a quarter seconds", "unsupported temporal notation"),
            (
                "spins for four and one quarter seconds",
                "unsupported temporal notation",
            ),
            ("spins for four thousand milliseconds", "unsupported temporal notation"),
            ("spins for three and one half seconds", "unsupported temporal notation"),
            ("spins for three and one-half seconds", "unsupported temporal notation"),
            ("spins for three and a half seconds", "unsupported temporal notation"),
            ("spins for three and one third seconds", "unsupported temporal notation"),
            ("spins for three and two fifths seconds", "unsupported temporal notation"),
            ("spins for three point three seconds", "unsupported temporal notation"),
            ("spins for three point two one seconds", "unsupported temporal notation"),
            (
                "spins for three thousand four hundred milliseconds",
                "unsupported temporal notation",
            ),
            (
                "spins for three thousand and five hundred milliseconds",
                "unsupported temporal notation",
            ),
            (
                "spins for three thousand two hundred and one milliseconds",
                "unsupported temporal notation",
            ),
            (
                "spins for three thousand, four hundred milliseconds",
                "unsupported temporal notation",
            ),
            ("spins for 4k milliseconds", "unsupported temporal notation"),
            (
                "turns through eighty and one quarter frames",
                "unsupported temporal notation",
            ),
            (
                "works for 3 seconds and 500 milliseconds",
                "additive duration",
            ),
            ("completes at frame 8e1", "unsupported temporal notation"),
        )
        for motion, error in cases:
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(GokuFullMotionQwenV16Error, error):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_source_future_anaphora_variants_are_rejected(self) -> None:
        census = self._validated_census()
        motions = (
            "maintains its prior gait while raising one paw",
            "keeps doing what it was doing while looking left",
            "continues its earlier trajectory while lowering its head",
            "follows what happens next in the source video",
            "resumes the movement visible in later frames",
            "continues walking just as shown later in the source clip",
            "keeps the action shown after the first frame and smiles",
            "carries on with the gesture from the rest of the clip",
            "reproduces the remainder of his recorded action",
            "does whatever he does after I0 and then smiles",
            "proceeds as shown in the later frames",
            "keeps walking as the remaining footage shows",
            "matches the frames that follow I0",
            "copies the action in the rest of this footage",
            "performs whatever occurs beyond frame zero",
            "performs whatever is shown beyond I0",
            "repeats whatever was shown beyond the initial frame",
        )
        for motion in motions:
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(
                    GokuFullMotionQwenV16Error,
                    "unavailable source future",
                ):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_target_signature_cannot_hide_invalid_time_or_sample_code(self) -> None:
        census = self._validated_census()
        for signature, error in (
            ("spin_for_8_seconds", "3.2-second target timeline"),
            ("raise_hand_at_s5", "source-view sample marker"),
        ):
            with self.subTest(signature=signature):
                raw = _plan()
                raw["dynamic_subject_targets"][0][
                    "target_action_signature"
                ] = signature
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                with self.assertRaisesRegex(GokuFullMotionQwenV16Error, error):
                    validate_target_plan(
                        canonical,
                        expected_iid="sample01",
                        source_census=census,
                    )

    def test_three_moderate_stages_are_not_double_counted(self) -> None:
        census = self._validated_census()
        motions = (
            "waves for 1 second, then claps for 1 second, then finally "
            "crouches for 1 second",
            "First, waves for 1 second. Second, claps for 1 second. Third, "
            "crouches for 1 second",
        )
        for motion in motions:
            with self.subTest(motion=motion):
                raw = _plan()
                raw["dynamic_subject_targets"][0]["target_motion"] = motion
                canonical, _ = canonicalize_target_plan(
                    raw, expected_iid="sample01", source_census=census
                )
                validated = validate_target_plan(
                    canonical, expected_iid="sample01", source_census=census
                )
                self.assertEqual(
                    validated["dynamic_subject_targets"][0]["target_motion"],
                    motion,
                )

    def test_absolute_marks_and_legitimate_f1_entity_are_not_false_rejected(self) -> None:
        census = self._validated_census()
        raw = _plan()
        motion = (
            "At the one-second mark, waves for one second, then stops at the "
            "three-second mark"
        )
        raw["dynamic_subject_targets"][0]["target_motion"] = motion
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        validated = validate_target_plan(
            canonical, expected_iid="sample01", source_census=census
        )
        self.assertEqual(
            validated["dynamic_subject_targets"][0]["target_motion"], motion
        )

        source_raw = _census()
        source_raw["dynamic_subjects"][0]["stable_reference"] = (
            "the red F1 race car on viewer-left"
        )
        source_canonical, _ = canonicalize_source_census(
            source_raw, expected_iid="sample01"
        )
        f1_census = validate_source_census(
            source_canonical, expected_iid="sample01"
        )
        compile_instruction(f1_census, self._validated_plan(f1_census))

    def test_compiled_instruction_surface_cannot_inject_overlong_timing(self) -> None:
        census = self._validated_census()
        plan = self._validated_plan(census)
        compiled = compile_instruction(census, plan)
        tampered = copy.deepcopy(compiled)
        tampered["instruction"] += " Continue for 8 seconds."
        tampered["instruction_sha256"] = hashlib.sha256(
            tampered["instruction"].encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error,
            "exceeds the 3.2-second target timeline",
        ):
            validate_compiled_instruction(
                tampered,
                source_census=census,
                target_plan=plan,
            )

        untimed_injection = copy.deepcopy(compiled)
        untimed_injection["instruction"] += " Also turn the sky green."
        untimed_injection["instruction_sha256"] = hashlib.sha256(
            untimed_injection["instruction"].encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "deterministic clause rendering"
        ):
            validate_compiled_instruction(
                untimed_injection,
                source_census=census,
                target_plan=plan,
            )

        clause_edit = copy.deepcopy(compiled)
        original_text = clause_edit["clauses"][0]["text"]
        changed_text = original_text + " while turning the sky green"
        clause_edit["clauses"][0]["text"] = changed_text
        clause_edit["instruction"] = clause_edit["instruction"].replace(
            original_text, changed_text
        )
        clause_edit["instruction_sha256"] = hashlib.sha256(
            clause_edit["instruction"].encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "signed target plan"
        ):
            validate_compiled_instruction(
                clause_edit,
                source_census=census,
                target_plan=plan,
            )

    def test_target_prompt_states_exact_moderate_timeline_contract(self) -> None:
        census = self._validated_census()
        user_prompt = build_target_plan_prompt(census, legacy_prompt="")
        for prompt in (TARGET_PLAN_SYSTEM, user_prompt):
            self.assertIn("exactly 81 frames", prompt)
            self.assertIn("frame index 80", prompt)
            self.assertIn("3.2 seconds", prompt)
            self.assertIn("moderate", prompt)
            self.assertRegex(prompt, r"F(?:20|<number>)")

    def test_source_sample_labels_never_reach_target_or_instruction(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["dynamic_subject_targets"][0]["target_motion"] = (
            "immediately raises both hands and completes the gesture by frame S5"
        )
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "source-view sample marker"
        ):
            validate_target_plan(
                canonical, expected_iid="sample01", source_census=census
            )

        source_raw = _census()
        source_raw["dynamic_subjects"][0]["stable_reference"] += " at S15"
        source_canonical, _ = canonicalize_source_census(
            source_raw, expected_iid="sample01"
        )
        source_with_marker = validate_source_census(
            source_canonical, expected_iid="sample01"
        )
        valid_plan = self._validated_plan(source_with_marker)
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "source-view sample marker"
        ):
            compile_instruction(source_with_marker, valid_plan)

    def test_ambiguous_target_time_is_rejected_not_guessed(self) -> None:
        census = self._validated_census()
        raw = _plan()
        raw["camera_target"]["target_motion"] = (
            "the camera remains locked off for several seconds"
        )
        canonical, _ = canonicalize_target_plan(
            raw, expected_iid="sample01", source_census=census
        )
        with self.assertRaisesRegex(
            GokuFullMotionQwenV16Error, "ambiguous temporal amount"
        ):
            validate_target_plan(
                canonical, expected_iid="sample01", source_census=census
            )

    def test_model_semicolons_are_mechanically_normalized_in_all_prose(self) -> None:
        source_raw = _census(mechanical=True)
        source_raw["dynamic_subjects"][0]["stable_reference"] += "; with a beard"
        source_raw["dynamic_subjects"][0]["i0_state"] += "; feet apart"
        source_raw["dynamic_subjects"][0]["source_motion"] += "; then holds it"
        source_raw["dynamic_subjects"][0]["motion_evidence"][0][
            "description"
        ] += "; the arm then stays raised"
        source_raw["camera"]["source_motion"] += "； framing shifts"
        source_raw["camera"]["motion_evidence"][0]["description"] += (
            "; background displacement is coherent"
        )
        source_canonical, source_repair = canonicalize_source_census(
            source_raw, expected_iid="sample01"
        )
        census = validate_source_census(
            source_canonical, expected_iid="sample01"
        )
        self.assertGreaterEqual(len(source_repair["operations"]), 6)

        target_raw = _plan(mechanical=True)
        target_raw["dynamic_subject_targets"][0]["target_motion"] += (
            "; then holds both palms together"
        )
        target_raw["camera_target"]["target_motion"] += (
            "； no reframing occurs"
        )
        target_canonical, target_repair = canonicalize_target_plan(
            target_raw,
            expected_iid="sample01",
            source_census=census,
        )
        plan = validate_target_plan(
            target_canonical,
            expected_iid="sample01",
            source_census=census,
        )
        self.assertTrue(
            any("target_motion_semicolon_to_comma" in operation for operation in target_repair["operations"])
        )
        compiled = compile_instruction(census, plan)
        self.assertNotIn("；", compiled["instruction"])
        # Only compiler-owned ASCII separators remain: one between each
        # adjacent closed clause, never inside a model-authored leaf.
        self.assertEqual(
            compiled["instruction"].count("; "),
            len(compiled["clauses"]) - 1,
        )

    def test_parser_constructs_once_and_accepts_mosaic_columns(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--input",
                "input.jsonl",
                "--output-root",
                "out",
                "--model",
                "model",
                "--root",
                ".",
                "--row-index",
                "0",
                "--num-rows",
                "1",
                "--mosaic-columns",
                "5",
            ]
        )
        self.assertEqual(args.mosaic_columns, 5)

    def test_run_one_uses_two_calls_and_stream_publishes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "out"
            row = _input_row()
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            backend = _Backend(_census(mechanical=True), _plan(mechanical=True))

            def factory(**kwargs):
                return backend

            def prepare(row, *, root, runtime):
                return (
                    root / "source.mp4",
                    root / "anchor.png",
                    {
                        "exact_i0": True,
                        "temporal_geometry": {
                            "frame_count": 81,
                            "fps": "25/1",
                            "timeline_span_seconds": 3.2,
                            "width": 1280,
                            "height": 720,
                        },
                    },
                    (object(), object(), object(), object(), object()),
                    "c" * 64,
                )

            status = run_one(
                _args(input_path, output_root, root),
                backend_factory=factory,
                prepare=prepare,
            )
            self.assertEqual(status, 0)
            self.assertEqual(backend.calls, ["census", "target"])
            result = json.loads(
                (output_root / "rows/sample01/result.json").read_text()
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(
                result["record_digest"],
                object_sha256({**result, "record_digest": None}),
            )
            passed_path = output_root / "passed/sample01.jsonl"
            self.assertTrue(passed_path.read_bytes().endswith(b"\n"))
            passed = json.loads(passed_path.read_text())
            self.assertEqual(passed["schema_version"], PASSED_SCHEMA)
            self.assertEqual(validate_passed_row(passed), passed)
            self.assertFalse(passed["generation_authorized"])
            self.assertTrue(passed["all_dynamic_subjects_covered"])
            receipt = json.loads(
                (output_root / "terminal/sample01.receipt.json").read_text()
            )
            self.assertEqual(receipt["schema_version"], ROW_RECEIPT_SCHEMA)
            self.assertEqual(receipt["status"], "ok")

            def forbidden_factory(**kwargs):
                raise AssertionError("terminal resume must not reload Qwen")

            resumed = run_one(
                _args(input_path, output_root, root),
                backend_factory=forbidden_factory,
                prepare=prepare,
            )
            self.assertEqual(resumed, 0)

    def test_run_one_failure_is_terminal_but_does_not_publish_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "out"
            input_path.write_text(
                json.dumps(_input_row()) + "\n", encoding="utf-8"
            )
            incomplete = _plan(mechanical=True)
            incomplete["dynamic_subject_targets"] = incomplete[
                "dynamic_subject_targets"
            ][:1]
            backend = _Backend(_census(mechanical=True), incomplete)

            def prepare(row, *, root, runtime):
                return (
                    root / "source.mp4",
                    root / "anchor.png",
                    {"exact_i0": True},
                    (object(), object(), object(), object(), object()),
                    "d" * 64,
                )

            status = run_one(
                _args(input_path, output_root, root),
                backend_factory=lambda **kwargs: backend,
                prepare=prepare,
            )
            self.assertEqual(status, 2)
            self.assertEqual(backend.calls, ["census", "target"])
            result = json.loads(
                (output_root / "rows/sample01/result.json").read_text()
            )
            self.assertEqual(result["status"], "error")
            self.assertIn("exactly one target", result["error"]["message"])
            self.assertFalse((output_root / "passed/sample01.jsonl").exists())
            receipt = json.loads(
                (output_root / "terminal/sample01.receipt.json").read_text()
            )
            self.assertEqual(receipt["status"], "error")

    def test_census_schema_failure_gets_one_local_retry_only(self) -> None:
        class RetryBackend(_Backend):
            def __init__(self) -> None:
                super().__init__(_census(mechanical=True), _plan(mechanical=True))
                self.census_attempt = 0

            def generate_source_motion_census_v16(self, **kwargs):
                self.calls.append("census")
                self.census_attempt += 1
                raw = (
                    "not JSON"
                    if self.census_attempt == 1
                    else json.dumps(self.census)
                )
                return raw, kwargs["expected_visual_input_digest"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "out"
            input_path.write_text(
                json.dumps(_input_row()) + "\n", encoding="utf-8"
            )
            backend = RetryBackend()

            def prepare(row, *, root, runtime):
                return (
                    root / "source.mp4",
                    root / "anchor.png",
                    {
                        "exact_i0": True,
                        "temporal_geometry": {
                            "frame_count": 81,
                            "fps": "25/1",
                            "timeline_span_seconds": 3.2,
                            "width": 640,
                            "height": 480,
                        },
                    },
                    (object(), object(), object(), object(), object()),
                    "e" * 64,
                )

            status = run_one(
                _args(input_path, output_root, root),
                backend_factory=lambda **kwargs: backend,
                prepare=prepare,
            )
            self.assertEqual(status, 0)
            self.assertEqual(backend.calls, ["census", "census", "target"])
            result = json.loads(
                (output_root / "rows/sample01/result.json").read_text()
            )
            attempts = result["source_stage"]["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertIsNotNone(attempts[0]["error"])
            self.assertIsNone(attempts[1]["error"])
            self.assertEqual(result["source_stage"]["selected_attempt"], 2)

    def test_persistent_worker_loads_once_and_continues_after_semantic_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "out"
            rows = [_input_row(f"sample{index:02d}") for index in range(3)]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            backend = _AdaptiveBackend(semantic_failure_iids={"sample01"})
            factory_calls = 0

            def factory(**kwargs):
                nonlocal factory_calls
                factory_calls += 1
                return backend

            status = run_worker(
                _worker_args(
                    input_path,
                    output_root,
                    root,
                    num_rows=3,
                    allow_errors=True,
                ),
                backend_factory=factory,
                prepare=_fake_prepare,
            )
            self.assertEqual(status, 0)
            self.assertEqual(factory_calls, 1)
            self.assertEqual(
                backend.calls,
                [
                    ("census", "sample00"),
                    ("target", "sample00"),
                    ("census", "sample01"),
                    ("target", "sample01"),
                    ("census", "sample02"),
                    ("target", "sample02"),
                ],
            )
            statuses = {
                iid: json.loads(
                    (output_root / "terminal" / f"{iid}.receipt.json").read_text()
                )["status"]
                for iid in ("sample00", "sample01", "sample02")
            }
            self.assertEqual(
                statuses,
                {"sample00": "ok", "sample01": "error", "sample02": "ok"},
            )
            self.assertTrue((output_root / "passed/sample00.jsonl").is_file())
            self.assertFalse((output_root / "passed/sample01.jsonl").exists())
            self.assertTrue((output_root / "passed/sample02.jsonl").is_file())

            def forbidden_factory(**kwargs):
                raise AssertionError("terminal worker resume must not reload Qwen")

            resumed = run_worker(
                _worker_args(
                    input_path,
                    output_root,
                    root,
                    num_rows=3,
                    allow_errors=True,
                ),
                backend_factory=forbidden_factory,
                prepare=_fake_prepare,
            )
            self.assertEqual(resumed, 0)

    def test_persistent_workers_use_deterministic_strided_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "out"
            rows = [_input_row(f"stride{index:02d}") for index in range(5)]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            backend = _AdaptiveBackend()
            factory_calls = 0

            def factory(**kwargs):
                nonlocal factory_calls
                factory_calls += 1
                return backend

            status = run_worker(
                _worker_args(
                    input_path,
                    output_root,
                    root,
                    num_rows=5,
                    worker_index=1,
                    num_workers=2,
                ),
                backend_factory=factory,
                prepare=_fake_prepare,
            )
            self.assertEqual(status, 0)
            self.assertEqual(factory_calls, 1)
            self.assertEqual(
                backend.calls,
                [
                    ("census", "stride01"),
                    ("target", "stride01"),
                    ("census", "stride03"),
                    ("target", "stride03"),
                ],
            )
            self.assertEqual(
                sorted(path.name for path in (output_root / "terminal").iterdir()),
                ["stride01.receipt.json", "stride03.receipt.json"],
            )

    def test_worker_backend_failure_is_not_published_as_semantic_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "out"
            rows = [_input_row(f"infra{index:02d}") for index in range(3)]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            backend = _AdaptiveBackend(
                infrastructure_failure_iids={"infra01"}
            )
            with self.assertRaisesRegex(RuntimeError, "simulated backend failure"):
                run_worker(
                    _worker_args(
                        input_path,
                        output_root,
                        root,
                        num_rows=3,
                        allow_errors=True,
                    ),
                    backend_factory=lambda **kwargs: backend,
                    prepare=_fake_prepare,
                )
            self.assertTrue(
                (output_root / "terminal/infra00.receipt.json").is_file()
            )
            self.assertFalse(
                (output_root / "terminal/infra01.receipt.json").exists()
            )
            self.assertFalse(
                (output_root / "rows/infra01/result.json").exists()
            )
            self.assertFalse(
                (output_root / "terminal/infra02.receipt.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
