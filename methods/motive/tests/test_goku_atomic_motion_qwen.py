from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

from motive import goku_atomic_motion_qwen as atomic
from motive import goku_full_motion_qwen_v16 as v16


def _evidence(description: str) -> dict:
    return {
        "schema_version": v16.MOTION_EVIDENCE_SCHEMA,
        "start_frame": 0,
        "end_frame": 80,
        "description": description,
    }


def _census(iid: str = "sample01") -> dict:
    return {
        "schema_version": v16.SOURCE_CENSUS_SCHEMA,
        "iid": iid,
        "dynamic_subjects": [
            {
                "schema_version": v16.SOURCE_SUBJECT_SCHEMA,
                "subject_id": "subject_01",
                "entity_type": "person",
                "stable_reference": "the person in blue on viewer-left holding a ball",
                "i0_bbox_xyxy_1000": [40, 130, 450, 970],
                "i0_state": "standing while holding a red ball at waist height",
                "source_action_signature": "raise_left_hand_peace_sign",
                "source_motion": "raises the left hand and forms a peace sign",
                "motion_evidence": [_evidence("the left hand rises and fingers extend")],
                "dynamic": True,
            },
            {
                "schema_version": v16.SOURCE_SUBJECT_SCHEMA,
                "subject_id": "subject_02",
                "entity_type": "person",
                "stable_reference": "the person in black on viewer-right",
                "i0_bbox_xyxy_1000": [520, 120, 960, 980],
                "i0_state": "standing with both hands open near the waist",
                "source_action_signature": "wave_right_hand",
                "source_motion": "raises the right hand and waves side to side",
                "motion_evidence": [_evidence("the right hand rises and moves laterally")],
                "dynamic": True,
            },
        ],
        "camera": {
            "schema_version": v16.SOURCE_CAMERA_SCHEMA,
            "motion_class": "pan_left",
            "source_motion": "the camera pans steadily left across the scene",
            "motion_evidence": [_evidence("the full background shifts to viewer-right")],
        },
        "all_dynamic_subjects_enumerated": True,
        "crowd_or_unresolved_motion": False,
        "confidence": "high",
    }


def _plan(iid: str = "sample01") -> dict:
    return {
        "schema_version": v16.TARGET_PLAN_SCHEMA,
        "iid": iid,
        "dynamic_subject_targets": [
            {
                "schema_version": v16.TARGET_SUBJECT_SCHEMA,
                "subject_id": "subject_01",
                "target_action_signature": "hand_ball_to_right_person",
                "target_motion": (
                    "immediately extends the red ball toward the person on the right "
                    "and releases it into the open hands by frame index 64"
                ),
                "substantive_change": True,
            },
            {
                "schema_version": v16.TARGET_SUBJECT_SCHEMA,
                "subject_id": "subject_02",
                "target_action_signature": "receive_ball_from_left_person",
                "target_motion": (
                    "immediately reaches both open hands toward the red ball and securely "
                    "receives it from the person on the left by frame index 64"
                ),
                "substantive_change": True,
            },
        ],
        "camera_target": {
            "schema_version": v16.TARGET_CAMERA_SCHEMA,
            "relation": "replace_motion",
            "motion_class": "locked_off",
            "target_motion": "camera remains locked off",
        },
        "coverage": {
            "schema_version": v16.TARGET_COVERAGE_SCHEMA,
            "dynamic_subject_ids": ["subject_01", "subject_02"],
            "camera_covered": True,
        },
        "confidence": "high",
    }


def _response(iid: str = "sample01") -> dict:
    plan = _plan(iid)
    return {
        "schema_version": atomic.ATOMIC_TARGET_RESPONSE_SCHEMA,
        "iid": iid,
        "atomic_event": {
            "schema_version": atomic.ATOMIC_EVENT_SCHEMA,
            "event_id": "event_01",
            "event_action_signature": "hand_ball_between_people",
            "event_summary": "the left person handing the visible ball to the right person",
            "participants": [
                {
                    "schema_version": atomic.ATOMIC_PARTICIPANT_SCHEMA,
                    "subject_id": "subject_01",
                    "role": "agent",
                    "event_contribution": "hands the visible ball to the other person",
                    "target_action_signature": plan["dynamic_subject_targets"][0][
                        "target_action_signature"
                    ],
                },
                {
                    "schema_version": atomic.ATOMIC_PARTICIPANT_SCHEMA,
                    "subject_id": "subject_02",
                    "role": "patient",
                    "event_contribution": "receives the same visible ball",
                    "target_action_signature": plan["dynamic_subject_targets"][1][
                        "target_action_signature"
                    ],
                },
            ],
            "causal_edges": [
                {
                    "schema_version": atomic.ATOMIC_CAUSAL_EDGE_SCHEMA,
                    "from_subject_id": "subject_01",
                    "to_subject_id": "subject_02",
                    "relation": "acts_on",
                }
            ],
            "independent_event_count": 1,
            "single_causal_event": True,
            "all_dynamic_subjects_in_event": True,
            "no_independent_action_threads": True,
        },
        "target_plan": plan,
        "confidence": "high",
    }


def _input_row(iid: str = "sample01") -> dict:
    return {
        "iid": iid,
        "group_id": "group01",
        "family": "people",
        "src_video": f"{iid}.mp4",
        "resolved_src_video": f"/fake/{iid}.mp4",
        "source_caption": "Two people make unrelated gestures.",
        "edited_caption": "One person hands a ball to the other person.",
        "prompt": "Make both people complete a ball handoff, then make them wave.",
        "anchor_image": f"{iid}.png",
        "resolved_anchor_image": f"/fake/{iid}.png",
        "anchor_sha256": "a" * 64,
        "source_video_sha256": "b" * 64,
        "prefilter_score": 9.0,
        "media": {},
        "motion": {},
    }


def _args(input_path: Path, output_root: Path, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=input_path,
        output_root=output_root,
        model="/fake/model",
        root=root,
        row_index=0,
        worker_index=None,
        num_workers=None,
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
        "c" * 64,
    )


def _write_source_census_cache(
    cache_root: Path, row: dict, *, census: dict | None = None
) -> Path:
    iid = row["iid"]
    record = {
        "schema_version": v16.RECORD_SCHEMA,
        "iid": iid,
        "status": "error",
        "input_digest": atomic.object_sha256(row),
        "input_row": copy.deepcopy(row),
        "visual_input_digest": "d" * 64,
        "source_stage": {
            "attempts": [{"attempt": 1, "error": None}],
            "selected_attempt": 1,
            "mechanical_repair": {
                "schema_version": v16.MECHANICAL_REPAIR_SCHEMA,
                "stage": "source_census",
                "operations": [],
                "semantic_fields_invented": False,
            },
        },
        "source_census": copy.deepcopy(census or _census(iid)),
        "record_digest": None,
    }
    record["record_digest"] = v16._digest_object_with_field(
        record, "record_digest"
    )
    path = cache_root / "rows" / iid / "result.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(v16._pretty_bytes(record))
    return path


class _Backend:
    model_path = "/fake/Qwen3-VL-32B-Instruct"
    model_revision = "test-revision"
    transformers_version = "test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _iid(user: str) -> str:
        for candidate in ("sample01", "sample02", "sample03"):
            if candidate in user:
                return candidate
        raise AssertionError("test prompt has no known IID")

    def generate_source_motion_census_v16(self, **kwargs):
        iid = self._iid(kwargs["user"])
        self.calls.append(("census", iid))
        return json.dumps(_census(iid)), kwargs["expected_visual_input_digest"]

    def generate_atomic_target_plan_v1(self, **kwargs):
        iid = self._iid(kwargs["user"])
        self.calls.append(("target", iid))
        return json.dumps(_response(iid)), kwargs["expected_visual_input_digest"]


class _SemanticRetryBackend(_Backend):
    def __init__(self, *, always_invalid: bool = False) -> None:
        super().__init__()
        self.always_invalid = always_invalid
        self.target_prompts: list[str] = []

    def generate_atomic_target_plan_v1(self, **kwargs):
        iid = self._iid(kwargs["user"])
        self.calls.append(("target", iid))
        self.target_prompts.append(kwargs["user"])
        response = _response(iid)
        if self.always_invalid or len(self.target_prompts) == 1:
            source_signature = _census(iid)["dynamic_subjects"][0][
                "source_action_signature"
            ]
            response["target_plan"]["dynamic_subject_targets"][0][
                "target_action_signature"
            ] = source_signature
            response["atomic_event"]["participants"][0][
                "target_action_signature"
            ] = source_signature
        return json.dumps(response), kwargs["expected_visual_input_digest"]


class AtomicMotionQwenTests(unittest.TestCase):
    def test_prompt_requires_one_graph_and_marks_frames_private(self) -> None:
        prompt = atomic.build_atomic_target_plan_prompt(
            _census(), legacy_prompt="Wave, then crouch."
        )
        self.assertIn("exactly one focal causal action", prompt.casefold())
        self.assertIn("causal_edges", prompt)
        self.assertIn("not a training label", prompt)
        self.assertIn(atomic.ATOMIC_TARGET_RESPONSE_SCHEMA, prompt)
        self.assertIn("two events and is forbidden", atomic.ATOMIC_TARGET_SYSTEM)

    def test_connected_atomic_wrapper_validates_and_extracts_v16_plan(self) -> None:
        result = atomic.validate_atomic_target_response(
            _response(), source_census=_census()
        )
        self.assertEqual(result["target_plan"]["schema_version"], v16.TARGET_PLAN_SCHEMA)
        self.assertEqual(
            [item["subject_id"] for item in result["atomic_event"]["participants"]],
            ["subject_01", "subject_02"],
        )

    def test_unambiguous_edge_aliases_normalize_but_conflicts_fail_closed(self) -> None:
        aliased = _response()
        edge = aliased["atomic_event"]["causal_edges"][0]
        edge["from"] = edge["from_subject_id"]
        edge["to"] = edge.pop("to_subject_id")
        repairs: list[str] = []

        result = atomic.validate_atomic_target_response(
            aliased,
            source_census=_census(),
            repair_operations=repairs,
        )
        canonical_edge = result["atomic_event"]["causal_edges"][0]
        self.assertEqual(
            set(canonical_edge),
            {"schema_version", "from_subject_id", "to_subject_id", "relation"},
        )
        self.assertEqual(canonical_edge["from_subject_id"], "subject_01")
        self.assertEqual(canonical_edge["to_subject_id"], "subject_02")
        self.assertEqual(len(repairs), 2)
        self.assertTrue(any("removed_redundant" in item for item in repairs))
        self.assertTrue(any("renamed" in item for item in repairs))

        conflicting = _response()
        conflicting["atomic_event"]["causal_edges"][0]["from"] = "subject_02"
        with self.assertRaisesRegex(
            atomic.GokuAtomicMotionQwenError, "ambiguous.*from"
        ):
            atomic.validate_atomic_target_response(
                conflicting, source_census=_census()
            )

    def test_disconnected_or_multiple_event_wrapper_is_rejected(self) -> None:
        disconnected = _response()
        disconnected["atomic_event"]["causal_edges"] = []
        with self.assertRaisesRegex(atomic.AtomicEventRejected, "disconnected"):
            atomic.validate_atomic_target_response(
                disconnected, source_census=_census()
            )

        multiple = _response()
        multiple["atomic_event"]["independent_event_count"] = 2
        with self.assertRaisesRegex(atomic.AtomicEventRejected, "not one"):
            atomic.validate_atomic_target_response(multiple, source_census=_census())

        stitched = _response()
        stitched["atomic_event"]["event_summary"] = (
            "the left person hands over the ball then waves"
        )
        with self.assertRaisesRegex(atomic.AtomicEventRejected, "independent-action"):
            atomic.validate_atomic_target_response(stitched, source_census=_census())

    def test_source_action_repeat_gets_one_semantic_feedback_retry(self) -> None:
        backend = _SemanticRetryBackend()
        trace: dict = {}
        result = atomic.annotate_prepared_row(
            _input_row(),
            backend=backend,
            source_path=Path("/fake/sample01.mp4"),
            anchor_path=Path("/fake/sample01.png"),
            media_verification={"exact_i0": True},
            visuals=(object(),),
            visual_input_digest="c" * 64,
            runtime={
                "max_new_tokens": 4096,
                "nframes": 16,
                "max_pixels": 2_359_296,
            },
            trace=trace,
        )

        self.assertEqual(
            backend.calls,
            [("census", "sample01"), ("target", "sample01"), ("target", "sample01")],
        )
        self.assertEqual(trace["atomic_target_stage"]["selected_attempt"], 2)
        attempts = trace["atomic_target_stage"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertIn("repeats the source action", attempts[0]["error"])
        self.assertIsNone(attempts[1]["error"])
        self.assertNotIn("SEMANTIC VALIDATION RETRY", backend.target_prompts[0])
        self.assertIn("SEMANTIC VALIDATION RETRY", backend.target_prompts[1])
        self.assertIn("repeats the source action", backend.target_prompts[1])
        atomic.validate_atomic_target_response(
            result["atomic_target_response"], source_census=_census()
        )

    def test_hash_closed_source_census_cache_hit_skips_first_visual_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "old_v17_qwen"
            row = _input_row()
            _write_source_census_cache(cache_root, row)
            backend = _Backend()
            trace: dict = {}
            result = atomic.annotate_prepared_row(
                row,
                backend=backend,
                source_path=Path("/fake/sample01.mp4"),
                anchor_path=Path("/fake/sample01.png"),
                media_verification={"exact_i0": True},
                visuals=(object(),),
                visual_input_digest="c" * 64,
                runtime={"max_new_tokens": 4096, "nframes": 16, "max_pixels": 1},
                trace=trace,
                source_census_cache_root=cache_root,
            )

            self.assertEqual(backend.calls, [("target", "sample01")])
            cache = trace["source_stage"]["cache"]
            self.assertEqual(trace["source_stage"]["selected_attempt"], "cache")
            self.assertEqual(cache["status"], "hit")
            self.assertEqual(cache["input_row_sha256"], atomic.object_sha256(row))
            self.assertEqual(cache["source_video_sha256"], "b" * 64)
            self.assertEqual(
                cache["source_census_sha256"], atomic.object_sha256(_census())
            )
            self.assertEqual(result["source_census"], _census())

    def test_missing_or_hash_mismatched_census_cache_falls_back_to_visual(self) -> None:
        for case in ("missing", "source_hash_mismatch"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                cache_root = Path(directory) / "old_v17_qwen"
                cache_root.mkdir()
                row = _input_row()
                if case == "source_hash_mismatch":
                    stale_row = copy.deepcopy(row)
                    stale_row["source_video_sha256"] = "e" * 64
                    _write_source_census_cache(cache_root, stale_row)
                backend = _Backend()
                trace: dict = {}
                atomic.annotate_prepared_row(
                    row,
                    backend=backend,
                    source_path=Path("/fake/sample01.mp4"),
                    anchor_path=Path("/fake/sample01.png"),
                    media_verification={"exact_i0": True},
                    visuals=(object(),),
                    visual_input_digest="c" * 64,
                    runtime={
                        "max_new_tokens": 4096,
                        "nframes": 16,
                        "max_pixels": 1,
                    },
                    trace=trace,
                    source_census_cache_root=cache_root,
                )
                self.assertEqual(
                    backend.calls,
                    [("census", "sample01"), ("target", "sample01")],
                )
                cache = trace["source_stage"]["cache"]
                self.assertEqual(
                    cache["status"],
                    "miss" if case == "missing" else "rejected",
                )
                if case == "source_hash_mismatch":
                    self.assertIn("input row hash differs", cache["rejection"])

    def test_semantic_retry_is_bounded_and_second_failure_stays_rejected(self) -> None:
        backend = _SemanticRetryBackend(always_invalid=True)
        trace: dict = {}
        with self.assertRaises(atomic.GokuAtomicMotionQwenStageError) as raised:
            atomic.annotate_prepared_row(
                _input_row(),
                backend=backend,
                source_path=Path("/fake/sample01.mp4"),
                anchor_path=Path("/fake/sample01.png"),
                media_verification={"exact_i0": True},
                visuals=(object(),),
                visual_input_digest="c" * 64,
                runtime={
                    "max_new_tokens": 4096,
                    "nframes": 16,
                    "max_pixels": 2_359_296,
                },
                trace=trace,
            )
        self.assertEqual(raised.exception.stage, "atomic_target_plan")
        self.assertEqual(len(raised.exception.attempts), 2)
        self.assertEqual(backend.calls.count(("target", "sample01")), 2)
        self.assertEqual(trace["atomic_target_stage"]["selected_attempt"], None)

    def test_run_one_publishes_atomic_provenance_and_old_v16_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "output"
            row = _input_row()
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            backend = _Backend()

            status = atomic.run_one(
                _args(input_path, output_root, root),
                backend_factory=lambda **kwargs: backend,
                prepare=_fake_prepare,
            )
            self.assertEqual(status, 0)
            self.assertEqual(backend.calls, [("census", "sample01"), ("target", "sample01")])

            result = json.loads(
                (output_root / "rows/sample01/result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["schema_version"], atomic.RECORD_SCHEMA)
            self.assertEqual(
                result["atomic_target_response"]["schema_version"],
                atomic.ATOMIC_TARGET_RESPONSE_SCHEMA,
            )
            provenance = result["planner_provenance"]
            self.assertFalse(provenance["wan_generation_prompt_is_training_label"])
            self.assertEqual(
                provenance["training_label_owner"],
                "motive.goku_atomic_motion_instruction.atomic_action_instruction",
            )

            passed_path = output_root / "passed/sample01.jsonl"
            passed = json.loads(passed_path.read_text(encoding="utf-8"))
            self.assertEqual(passed["schema_version"], v16.PASSED_SCHEMA)
            self.assertEqual(v16.validate_passed_row(passed), passed)
            self.assertEqual(passed["target_plan"], result["target_plan"])
            self.assertEqual(
                passed["edit_instruction"], result["compiled_instruction"]["instruction"]
            )

            receipt_path = output_root / "terminal/sample01.receipt.json"
            receipt = atomic._validate_terminal_receipt(
                receipt_path,
                output_root=output_root,
                iid="sample01",
                input_digest=atomic.object_sha256(row),
            )
            self.assertEqual(receipt["schema_version"], atomic.ROW_RECEIPT_SCHEMA)

            calls_before_resume = list(backend.calls)
            resumed = atomic.run_one(
                _args(input_path, output_root, root),
                backend_factory=lambda **kwargs: (_ for _ in ()).throw(
                    AssertionError("resume must not reload Qwen")
                ),
                prepare=_fake_prepare,
            )
            self.assertEqual(resumed, 0)
            self.assertEqual(backend.calls, calls_before_resume)

    def test_worker_loads_one_backend_for_its_strided_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output_root = root / "output"
            rows = [_input_row("sample01"), _input_row("sample02")]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            args = _args(input_path, output_root, root)
            args.row_index = None
            args.worker_index = 0
            args.num_workers = 1
            args.num_rows = 2
            backend = _Backend()
            factory_calls = 0

            def factory(**kwargs):
                nonlocal factory_calls
                factory_calls += 1
                return backend

            status = atomic.run_worker(
                args, backend_factory=factory, prepare=_fake_prepare
            )
            self.assertEqual(status, 0)
            self.assertEqual(factory_calls, 1)
            self.assertEqual(
                backend.calls,
                [
                    ("census", "sample01"),
                    ("target", "sample01"),
                    ("census", "sample02"),
                    ("target", "sample02"),
                ],
            )
            for iid in ("sample01", "sample02"):
                passed = json.loads(
                    (output_root / "passed" / f"{iid}.jsonl").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(v16.validate_passed_row(passed), passed)


if __name__ == "__main__":
    unittest.main()
