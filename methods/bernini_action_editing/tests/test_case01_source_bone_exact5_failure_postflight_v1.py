#!/usr/bin/env python3
"""Renderer-free contract tests for the exact5 failure postflight."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from methods.bernini_action_editing.tests import (
    test_build_case01_source_bone_exact5_r64_html_v1 as success_fixture,
)
from methods.bernini_action_editing.tools import (
    case01_source_bone_exact5_failure_postflight_v1 as postflight,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures/case01_source_bone_exact5_failure_postflight_contract_v1.json"
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def seal(value: dict, field: str) -> dict:
    result = copy.deepcopy(value)
    result[field] = postflight.contract.object_sha256(result)
    return result


def failure_attestation(plan_sha256: str) -> dict:
    return seal(
        {
            "schema_version": postflight.contract.FAILURE_SCHEMA,
            "status": "FAILED_NO_RETRY",
            "error_type": postflight.FAILURE_ERROR_TYPE,
            "error": postflight.FAILURE_ERROR,
            "plan_path": f"/run/plan/{postflight.contract.PLAN_REL.name}",
            "plan_sha256": plan_sha256,
            "runner_path": "/release/case01_source_bone_exact5_runner_v1.py",
            "retry_allowed": False,
            "partial_outputs_are_not_results": True,
            "scientific_claim_authorized": False,
        },
        "failure_digest",
    )


def synthetic_observation() -> dict:
    plan = success_fixture.make_plan()
    cases = postflight.contract.validate_plan(plan)
    rows = []
    for index, case in enumerate(cases):
        task_id = case["task_id"]
        rows.append(
            {
                **case,
                "source_path": Path(f"/bundle/sources/{case['id']}.mp4"),
                "output_path": Path(f"/bundle/outputs/media/{task_id}.mp4"),
                "receipt_path": Path(
                    f"/bundle/outputs/media/{task_id}.mp4.receipt.json"
                ),
                "runner_task_path": Path(
                    f"/bundle/outputs/media/.matched-v2-{index:02d}-{task_id}"
                    "-runner-task.json"
                ),
                "source_sha256": case["source_sha256"],
                "source_size": case["source_size"],
                "source_mode": {"mode": 0o644, "nlink": 1},
                "source_probe": {
                    "codec": "h264", **postflight.contract.EXPECTED_SOURCE_VIDEO,
                },
                "output_sha256": sha(f"failed-output-{index}"),
                "output_size": 1000 + index,
                "output_mode": {"mode": 0o644, "nlink": 1},
                "output_probe": {
                    "codec": "h264", "frame_count": 81, "fps_num": 25,
                    "fps_den": 1, "width": 480, "height": 496,
                },
                "receipt_sha256": sha(f"receipt-file-{index}"),
                "receipt_size": 2000 + index,
                "receipt_mode": {"mode": 0o600, "nlink": 1},
                "receipt_digest": sha(f"receipt-digest-{index}"),
                "runner_task_sha256": sha(f"runner-task-file-{index}"),
                "runner_task_size": 3000 + index,
                "runner_task_mode": {"mode": 0o600, "nlink": 1},
                "task_replay": {
                    "task_result_digest": sha(f"task-result-{index}"),
                    "task_input_digest": sha(f"task-input-{index}"),
                    "artifact_rows_digest": sha(f"artifact-rows-{index}"),
                    "consumption_digest": sha(f"consumption-{index}"),
                    "model_capture_digest": sha("shared-model-capture"),
                    "adapter_capture_digest": sha(f"adapter-capture-{index}"),
                    "consumption_input_digest": sha(f"consumption-input-{index}"),
                },
                "artifact_rows": [
                    {
                        "role": role,
                        "path": f"outputs/media/{role}-{index}.json",
                        "sha256": sha(f"{role}-file-{index}"), "size": 50,
                        "embedded_digest": sha(f"{role}-digest-{index}"),
                    }
                    for role in sorted(postflight.ARTIFACT_SUFFIXES)
                ],
                "log_path": Path(f"/bundle/outputs/media/log-{index}"),
                "log_sha256": sha(f"log-{index}"), "log_size": 99,
                "log_mode": {"mode": 0o600, "nlink": 1},
                "receipt": {
                    "sampling": {"seed": postflight.contract.SEED},
                    "prompt_contract": {"task": "mv2v"},
                    "input": {
                        "source_video_physical_authority": {"mode": 0o644},
                    },
                },
            }
        )
    plan_sha = sha("plan-file")
    failure = failure_attestation(plan_sha)
    return {
        "plan": plan, "plan_sha256": plan_sha, "plan_size": 1234,
        "plan_mode": {"mode": 0o644, "nlink": 1},
        "failure": failure, "failure_sha256": sha("failure-file"),
        "failure_size": 456,
        "failure_mode": {"mode": 0o644, "nlink": 1}, "cases": rows,
        "reference_path": Path("/bundle") / postflight.REFERENCE_REL,
        "reference_sha256": postflight.contract.REFERENCE_OUTPUT_SHA256,
        "reference_size": 7531886,
        "reference_mode": {"mode": 0o444, "nlink": 1},
        "reference_receipt": {
            "receipt_digest": postflight.REFERENCE_RECEIPT_DIGEST,
        },
        "reference_receipt_sha256": postflight.REFERENCE_RECEIPT_SHA256,
        "reference_receipt_size": 7809,
        "reference_receipt_mode": {"mode": 0o444, "nlink": 1},
        "reference_probe": {
            "codec": "h264", "frame_count": 81, "fps_num": 25,
            "fps_den": 1, "width": 480, "height": 496,
        },
        "exact_decode": {
            "pixel_format": "rgb24", "byte_count": 57_153_600,
            "sha256": sha("failed-decoded"),
        },
        "reference_decode": {
            "pixel_format": "rgb24", "byte_count": 57_153_600,
            "sha256": sha("reference-decoded"),
        },
        "ffprobe_authority": {
            "path": "/local/ffprobe", "sha256": sha("ffprobe"), "size": 10,
        },
        "ffmpeg_authority": {
            "path": "/local/ffmpeg", "sha256": sha("ffmpeg"), "size": 11,
        },
        "auxiliary_root_entries": ["evidence"],
    }


class FailurePostflightTests(unittest.TestCase):
    def test_frozen_fixture_matches_contract(self) -> None:
        raw = FIXTURE.read_bytes()
        fixture = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            raw, postflight.contract.canonical_json_bytes(fixture) + b"\n"
        )
        self.assertEqual(fixture["postflight_schema"], postflight.POSTFLIGHT_SCHEMA)
        self.assertEqual(fixture["postflight_status"], postflight.POSTFLIGHT_STATUS)
        self.assertEqual(fixture["task_ids"], list(postflight.contract.TASK_IDS))
        self.assertEqual(
            fixture["historical_reference_sha256"],
            postflight.contract.REFERENCE_OUTPUT_SHA256,
        )
        self.assertFalse(fixture["historical_reference_is_current_task_arm"])

    def test_failure_attestation_exact_error_and_digest(self) -> None:
        plan_sha = sha("plan")
        value = failure_attestation(plan_sha)
        postflight.validate_failure_attestation(value, plan_sha256=plan_sha)
        bad = copy.deepcopy(value)
        bad["error"] = "some other failure"
        bad = seal(
            {key: item for key, item in bad.items() if key != "failure_digest"},
            "failure_digest",
        )
        with self.assertRaisesRegex(postflight.PostflightError, "parity gate"):
            postflight.validate_failure_attestation(bad, plan_sha256=plan_sha)

    def test_manifest_is_exactly_reconstructed_and_claim_limited(self) -> None:
        observed = synthetic_observation()
        manifest = postflight.build_manifest(
            observed, observed_at_utc="2026-08-21T12:00:00+00:00",
        )
        postflight.validate_manifest(manifest, observed)
        self.assertEqual(manifest["reference_parity"]["status"], "FAIL")
        self.assertTrue(
            manifest["reference_parity"][
                "historical_reference_is_not_a_current_task_arm"
            ]
        )
        self.assertTrue(manifest["claim_limits"]["partial_outputs_are_not_results"])
        self.assertFalse(manifest["claim_limits"]["scientific_claim_authorized"])
        self.assertFalse(manifest["claim_limits"]["formal_claim_authorized"])
        self.assertFalse(manifest["artifact_inventory"]["success_report_present"])

        bad = copy.deepcopy(manifest)
        bad["task_rows"][0]["output"]["sha256"] = sha("forged-output")
        bad["task_rows_digest"] = postflight.contract.object_sha256(
            bad["task_rows"]
        )
        bad = seal(
            {key: item for key, item in bad.items() if key != "manifest_digest"},
            "manifest_digest",
        )
        with self.assertRaisesRegex(postflight.PostflightError, "cross-link"):
            postflight.validate_manifest(bad, observed)

    def test_bundle_closure_requires_reference_and_all_persistent_files(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            bundle = Path(value) / "bundle"
            for directory in (
                bundle / "plan", bundle / "final", bundle / "sources",
                bundle / "outputs/media", bundle / "reference",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (bundle / postflight.contract.PLAN_REL).touch()
            (bundle / postflight.contract.ATTESTATION_REL).touch()
            for variant in postflight.contract.VARIANT_ORDER:
                (bundle / "sources" / f"{variant}.mp4").touch()
            for name in postflight._expected_media_names():
                (bundle / "outputs/media" / name).touch()
            reference = bundle / postflight.REFERENCE_REL
            reference.touch()
            (bundle / postflight.REFERENCE_RECEIPT_REL).touch()
            resolved = postflight.bundle_paths(bundle, require_manifest=False)
            self.assertEqual(resolved["reference"], reference)
            reference.unlink()
            with self.assertRaisesRegex(postflight.PostflightError, "missing"):
                postflight.bundle_paths(bundle, require_manifest=False)

    def test_missing_bundle_leaf_never_creates_postflight(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            bundle = Path(value) / "bundle"
            for directory in (
                bundle / "plan", bundle / "final", bundle / "sources",
                bundle / "outputs/media", bundle / "reference",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (bundle / postflight.contract.PLAN_REL).touch()
            (bundle / postflight.contract.ATTESTATION_REL).touch()
            for variant in postflight.contract.VARIANT_ORDER:
                (bundle / "sources" / f"{variant}.mp4").touch()
            names = sorted(postflight._expected_media_names())
            for name in names[:-1]:
                (bundle / "outputs/media" / name).touch()
            (bundle / postflight.REFERENCE_REL).touch()
            (bundle / postflight.REFERENCE_RECEIPT_REL).touch()
            with self.assertRaises(postflight.PostflightError):
                postflight.produce_manifest(
                    bundle=bundle, ffprobe=Path("/not-used/ffprobe"),
                    ffmpeg=Path("/not-used/ffmpeg"),
                )
            self.assertFalse((bundle / "postflight").exists())


if __name__ == "__main__":
    unittest.main()
