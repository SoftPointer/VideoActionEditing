from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import postflight_seer_full160_core4_eval_v1 as postflight  # noqa: E402
import seer_full160_review_verdict_v1 as verdict  # noqa: E402


POSTFLIGHT_LAUNCHER = (
    METHOD_ROOT / "scripts" / "auh_postflight_seer_full160_core4_eval_20260813.sbatch"
)
EVAL_LAUNCHER = (
    METHOD_ROOT / "scripts" / "auh_eval_seer_full160_core4_array_20260813.sbatch"
)


def _signed(unsigned: dict[str, object]) -> dict[str, object]:
    return {**unsigned, "receipt_digest": verdict.object_sha256(unsigned)}


def _master(*, identical_iids: frozenset[str] = frozenset()) -> dict[str, object]:
    cases = [
        {
            "iid": iid,
            "path": f"/evidence/{iid}/eval-execution-binding.json",
            "sha256": str(index + 1) * 64,
            "receipt_digest": str(index + 2) * 64,
            "decoded_outputs_byte_identical": iid in identical_iids,
            "pair_receipt_path": f"/evidence/results/{iid}/paired-receipt.json",
            "pair_receipt_sha256": str(index + 3) * 64,
            "pair_receipt_digest": str(index + 4) * 64,
        }
        for index, iid in enumerate(verdict.CORE4_IIDS)
    ]
    return _signed(
        {
            "schema_version": postflight.SCHEMA_VERSION,
            "status": (
                "full160_core4_execution_closed_pending_strict_paired_"
                "full_video_review"
            ),
            "array_job_id": 135999,
            "array_task_count": 2,
            "task_completions": [
                {
                    "task": task,
                    "node": f"auh7-1b-gpu-{200 + task}",
                    "path": f"/evidence/task-{task}.COMPLETE",
                    "sha256": str(8 + task) * 64,
                }
                for task in (0, 1)
            ],
            "training_method_source": {
                "revision": "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a",
                "archive_sha256": (
                    "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822"
                ),
            },
            "inference_runtime_overlay": {
                "archive_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "manifest_digest": "c" * 64,
                "heldout_runner_sha256": "d" * 64,
                "is_training_method_archive": False,
            },
            "trained_adapter": {
                "adapter_model_sha256": "e" * 64,
                "training_receipt_sha256": "f" * 64,
                "training_receipt_digest": "0" * 64,
                "training_global_step": 160,
                "training_max_steps": 160,
            },
            "cases": cases,
            "decision_contract": copy.deepcopy(verdict.MASTER_DECISION_CONTRACT),
            "decoded_outputs_byte_identical_count": len(identical_iids),
            "core4_master_receipt": {
                "path": "/evidence/results/core4-master-receipt.json",
                "sha256": "a" * 64,
                "receipt_digest": "b" * 64,
            },
            "full_video_action_and_preservation_review_complete": False,
            "method_success_claimed": False,
            "method_success_authorized": False,
        }
    )


def _completed_review(master: dict[str, object], states: list[tuple[bool, bool]]) -> dict[str, object]:
    blank = verdict.build_blank_review(master)
    rows = copy.deepcopy(blank["cases"])
    for row, (base_safe, trained_safe) in zip(rows, states):
        row["frozen_base"] = {
            "requested_action_reaches_terminal_and_holds": base_safe,
            "identity_scene_camera_inventory_preserved": base_safe,
            "joint_safe": base_safe,
        }
        row["trained_adapter"] = {
            "requested_action_reaches_terminal_and_holds": trained_safe,
            "identity_scene_camera_inventory_preserved": trained_safe,
            "joint_safe": trained_safe,
        }
        row["full_video_review_complete"] = True
    unsigned = dict(blank)
    unsigned.pop("receipt_digest")
    unsigned["status"] = "completed_strict_paired_full_video_review"
    unsigned["cases"] = rows
    unsigned["all_four_cases_review_complete"] = True
    return _signed(unsigned)


class SeerFull160PostflightTests(unittest.TestCase):
    def test_executable_resolver_allows_absolute_symlink_to_regular_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "python-real"
            target.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            target.chmod(0o500)
            link = root / "python"
            link.symlink_to(target.name)
            self.assertEqual(
                postflight._executable_file(link, label="fixture Python"), target
            )

    def test_blank_review_freezes_exact_joint_safe_schema_and_thresholds(self) -> None:
        blank = verdict.build_blank_review(_master())
        self.assertEqual(blank["decision_thresholds"], verdict.THRESHOLDS)
        self.assertEqual([row["iid"] for row in blank["cases"]], list(verdict.CORE4_IIDS))
        self.assertTrue(all(row["frozen_base"]["joint_safe"] is None for row in blank["cases"]))
        self.assertFalse(blank["all_four_cases_review_complete"])
        self.assertFalse(blank["method_success_authorized"])

    def test_master_admission_cross_binds_full_decision_contract(self) -> None:
        hostile = copy.deepcopy(_master())
        hostile["decision_contract"]["minimum_trained_joint_safe"] = 2
        unsigned = dict(hostile)
        unsigned.pop("receipt_digest")
        hostile = _signed(unsigned)
        with self.assertRaisesRegex(verdict.VerdictError, "execution contract"):
            verdict.build_blank_review(hostile)

    def test_unique_go_rule_and_no_go_paths_are_deterministic(self) -> None:
        master = _master()
        # Three trained-safe cases, three flips, both families, zero regression.
        review = _completed_review(master, [(False, True), (False, True), (False, True), (False, False)])
        result = verdict.reduce_review(review, master)
        self.assertEqual(result["status"], "GO")
        self.assertTrue(result["method_success"])
        self.assertTrue(all(result["checks"].values()))

        # Equal arm labels remain legitimate zero-effect evidence and NO_GO.
        identical = _completed_review(master, [(False, False)] * 4)
        result = verdict.reduce_review(identical, master)
        self.assertEqual(result["status"], "NO_GO")
        self.assertFalse(result["method_success"])
        self.assertEqual(result["counts"]["base_fail_to_trained_joint_safe_flips"], 0)

        # A safe-to-unsafe regression forces NO_GO even when three flips exist.
        regression = _completed_review(master, [(False, True), (False, True), (False, True), (True, False)])
        result = verdict.reduce_review(regression, master)
        self.assertEqual(result["status"], "NO_GO")
        self.assertFalse(result["checks"]["maximum_joint_safe_to_unsafe_regressions_met"])

    def test_byte_identical_outputs_accept_equal_labels_and_force_nonflip(self) -> None:
        iid = verdict.CORE4_IIDS[0]
        master = _master(identical_iids=frozenset({iid}))
        review = _completed_review(
            master, [(True, True), (False, True), (False, True), (False, True)]
        )
        result = verdict.reduce_review(review, master)
        row = result["cases"][0]
        self.assertTrue(row["decoded_outputs_byte_identical"])
        self.assertTrue(row["identical_output_forced_nonflip"])
        self.assertFalse(row["base_fail_to_trained_joint_safe"])
        self.assertFalse(row["base_safe_to_trained_unsafe"])

    def test_byte_identical_outputs_reject_different_arm_labels(self) -> None:
        iid = verdict.CORE4_IIDS[0]
        master = _master(identical_iids=frozenset({iid}))
        review = _completed_review(master, [(False, True)] * 4)
        with self.assertRaisesRegex(
            verdict.VerdictError, "identical decoded outputs require identical arm labels"
        ):
            verdict.reduce_review(review, master)

    def test_completed_review_requires_exact_root_and_status(self) -> None:
        master = _master()
        review = _completed_review(master, [(False, True)] * 4)
        hostile = dict(review)
        hostile["unregistered_field"] = True
        unsigned = dict(hostile)
        unsigned.pop("receipt_digest")
        hostile = _signed(unsigned)
        with self.assertRaisesRegex(verdict.VerdictError, "root contract"):
            verdict.reduce_review(hostile, master)

        hostile = dict(review)
        hostile["status"] = "blind_full_video_review_complete"
        unsigned = dict(hostile)
        unsigned.pop("receipt_digest")
        hostile = _signed(unsigned)
        with self.assertRaisesRegex(verdict.VerdictError, "root contract"):
            verdict.reduce_review(hostile, master)

    def test_joint_safe_must_be_exact_boolean_and(self) -> None:
        master = _master()
        review = _completed_review(master, [(False, True)] * 4)
        hostile = copy.deepcopy(review)
        hostile["cases"][0]["trained_adapter"]["joint_safe"] = False
        unsigned = dict(hostile)
        unsigned.pop("receipt_digest")
        hostile = _signed(unsigned)
        with self.assertRaisesRegex(verdict.VerdictError, "required AND"):
            verdict.reduce_review(hostile, master)

    def test_launchers_pin_afterok_components_and_never_claim_method_success(self) -> None:
        eval_text = EVAL_LAUNCHER.read_text(encoding="utf-8")
        post_text = POSTFLIGHT_LAUNCHER.read_text(encoding="utf-8")
        for token in (
            "postflight_seer_full160_core4_eval_v1.py",
            "seer_full160_review_verdict_v1.py",
            "SEER_FULL160_POSTFLIGHT_SHA256",
            "SEER_FULL160_POSTFLIGHT_LAUNCHER_SHA256",
            "SEER_FULL160_VERDICT_REDUCER_SHA256",
        ):
            self.assertIn(token, eval_text)
        self.assertIn("afterok:<full160-array-job-id>", post_text)
        self.assertIn("PASS_EXECUTION_CLOSURE_PENDING_STRICT_PAIRED_FULL_VIDEO_REVIEW", post_text)
        for token in (
            "SEER_FULL160_SOURCE_BINDER_SHA256",
            "SEER_FULL160_OVERLAY_BUILDER_SHA256",
            "SEER_FULL160_POSTFLIGHT_LAUNCHER_SHA256",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONNOUSERSITE=1",
            '"${python_bin}" -I -B',
        ):
            self.assertIn(token, post_text)
        self.assertNotIn("METHOD_SUCCESS", post_text)
        postflight_text = Path(postflight.__file__).read_text(encoding="utf-8")
        self.assertIn("verify-core4", postflight_text)
        self.assertIn("runtime heldout runner/overlay cross-bind differs", postflight_text)


if __name__ == "__main__":
    unittest.main()
