#!/usr/bin/env python3

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import case01_source_bone_intervention_plan_v1 as plan_v1  # noqa: E402


def _reference() -> dict:
    checkpoint = {**plan_v1.EXPECTED_CHECKPOINT, "path": "/authority/cp644/checkpoint_manifest.json"}
    producer = {
        **plan_v1.EXPECTED_PRODUCER,
        "infer_lora_path": "/release/infer_lora.py",
        "ffprobe_path": "/runtime/ffprobe",
    }
    tasks = []
    for index in range(8):
        for arm in ("base", "full644"):
            tasks.append(
                {
                    "case_index": index,
                    "arm": arm,
                    "iid": plan_v1.IID if index == 1 else f"dummy-{index}",
                    "instruction": plan_v1.INSTRUCTION if index == 1 else "dummy",
                    "instruction_sha256": (
                        plan_v1.INSTRUCTION_SHA256 if index == 1 else "0" * 64
                    ),
                    "seed": 2026 + index,
                    "num_inference_steps": 40,
                    "source_onset_policy": "none",
                }
            )
    value = {
        "schema_version": "bernini-full644-exploratory-matched-eval-plan-v1",
        "checkpoint_manifest": checkpoint,
        "producer": producer,
        "tasks": tasks,
    }
    # The fixture carries the real authority digest while keeping irrelevant
    # shared8 fields small.  Patch object_sha256 only inside tests that invoke
    # validate_reference_plan.
    value["plan_digest"] = plan_v1.REFERENCE_PLAN_DIGEST
    return value


class Case01SourceBoneInterventionPlanTests(unittest.TestCase):
    def test_four_variant_hashes_and_stop_boundary_are_closed(self) -> None:
        self.assertEqual(
            tuple(plan_v1.CANDIDATE_VARIANT_SHA256), plan_v1.VARIANT_ORDER
        )
        self.assertTrue(plan_v1.ASSET_AUTHORITY_STATUS.startswith("REJECTED_"))
        self.assertEqual(plan_v1.EXPECTED_VIDEO["frame_count"], 81)
        self.assertEqual(plan_v1.EXPECTED_VIDEO["fps_num"], 25)
        self.assertIn("ASSET_AUTHORITY_REJECTED_DO_NOT_LAUNCH", plan_v1.STOP_TUPLE)
        self.assertIn(
            "LARGE_SCALE_OBJECT_ADAPTER_TRAINING_FORBIDDEN_AT_STAGE0",
            plan_v1.STOP_TUPLE,
        )

    def test_variant_semantics_isolate_presence_position_and_sham(self) -> None:
        semantics = plan_v1._variant_semantics()  # noqa: SLF001
        self.assertTrue(semantics["original"]["bone_present"])
        self.assertFalse(semantics["removed"]["bone_present"])
        self.assertTrue(semantics["translated"]["bone_present"])
        self.assertEqual(
            semantics["translated"]["original_bone_region_treatment"],
            "same_deterministic_ffmpeg_removelogo_spatial_interpolation_r4_as_removed",
        )
        self.assertEqual(semantics["sham"]["bone_position"], "source_original")

    def test_source_argument_parser_rejects_partial_or_duplicate_factorial(self) -> None:
        with self.assertRaises(plan_v1.InterventionPlanError):
            plan_v1._parse_sources(["original=/a"])  # noqa: SLF001
        with self.assertRaises(plan_v1.InterventionPlanError):
            plan_v1._parse_sources(  # noqa: SLF001
                [
                    "original=/a",
                    "removed=/b",
                    "translated=/c",
                    "original=/d",
                ]
            )

    def test_build_plan_is_r64_only_by_default_and_matched(self) -> None:
        reference = _reference()
        real_object_sha = plan_v1.object_sha256
        paths = {variant: Path(f"/sources/{variant}.mp4") for variant in plan_v1.VARIANT_ORDER}

        def fake_stable(path, *, expected_sha256=None, return_bytes=False):
            del return_bytes
            variant = Path(path).stem
            self.assertEqual(
                expected_sha256, plan_v1.CANDIDATE_VARIANT_SHA256[variant]
            )
            return None, expected_sha256, 1234

        with tempfile.TemporaryDirectory() as root:
            output_root = Path(root).resolve()
            try:
                plan_v1.object_sha256 = lambda value: (  # type: ignore[assignment]
                    plan_v1.REFERENCE_PLAN_DIGEST
                    if isinstance(value, dict)
                    and value.get("schema_version")
                    == "bernini-full644-exploratory-matched-eval-plan-v1"
                    else real_object_sha(value)
                )
                original_stable = plan_v1._stable_file
                plan_v1._stable_file = fake_stable  # type: ignore[assignment]
                plan = plan_v1.build_plan(
                    reference_plan=reference,
                    reference_plan_path=Path("/authority/reference-plan.json"),
                    sources=paths,
                    output_root=output_root,
                    probe=lambda path: dict(plan_v1.EXPECTED_VIDEO),
                )
            finally:
                plan_v1.object_sha256 = real_object_sha  # type: ignore[assignment]
                plan_v1._stable_file = original_stable  # type: ignore[assignment]
        self.assertEqual(plan["arms"], ["full644"])
        self.assertEqual(plan["task_count"], 4)
        self.assertEqual(
            [task["intervention_variant"] for task in plan["tasks"]],
            list(plan_v1.VARIANT_ORDER),
        )
        conditions = {
            (
                task["instruction_sha256"],
                task["seed"],
                task["num_inference_steps"],
                task["source_onset_policy"],
            )
            for task in plan["tasks"]
        }
        self.assertEqual(len(conditions), 1)
        self.assertFalse(plan["execution"]["existing_top_level_runner_accepts_plan"])
        self.assertFalse(plan["asset_authority"]["launch_allowed"])
        self.assertTrue(
            plan["asset_authority"][
                "codec_only_transcoded_present_control_required"
            ]
        )
        self.assertTrue(
            plan["execution"][
                "lower_level_retained_source_path_supports_alternate_absolute_paths"
            ]
        )

    def test_frozen_runner_has_generic_lower_source_path_but_closed_top_plan(self) -> None:
        runner = (METHOD_ROOT / "full644_exploratory_matched_runner_auh_r5.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('path = Path(task["source_video"])', runner)
        self.assertIn('expected_sha256=task["source_video_sha256"]', runner)
        self.assertIn('"--source-video",\n        task["source_video"]', runner)
        self.assertIn("v2.validate_plan(plan)", runner)
        self.assertIn("or len(tasks) != 16", runner)
        self.assertIn(
            'or tuple(task.get("task_id") for task in tasks) != TASK_IDS', runner
        )

    def test_strict_json_reader_fails_closed_when_bytes_are_unavailable(self) -> None:
        original_stable = plan_v1._stable_file
        plan_v1._stable_file = lambda *args, **kwargs: (None, "0" * 64, 0)  # type: ignore[assignment]
        try:
            with self.assertRaisesRegex(
                plan_v1.InterventionPlanError,
                "stable JSON reader returned no bytes",
            ):
                plan_v1._strict_json_file(  # noqa: SLF001
                    "/authority/reference-plan.json", expected_sha256="0" * 64
                )
        finally:
            plan_v1._stable_file = original_stable  # type: ignore[assignment]

    def test_plan_digest_detects_treatment_mutation(self) -> None:
        payload = {"schema_version": "x", "seed": 2027}
        digest = plan_v1.object_sha256(payload)
        changed = copy.deepcopy(payload)
        changed["seed"] = 2028
        self.assertNotEqual(digest, plan_v1.object_sha256(changed))


if __name__ == "__main__":
    unittest.main()
