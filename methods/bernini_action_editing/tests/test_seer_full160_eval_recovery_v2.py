from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS = METHOD_ROOT / "tools"
for root in (METHOD_ROOT, TOOLS):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bind_seer_full160_eval_source_v2 as binder  # noqa: E402
import postflight_seer_full160_core4_eval_v2 as postflight  # noqa: E402
import seer_full160_review_verdict_v2 as verdict  # noqa: E402


def _signed(value: dict[str, object]) -> dict[str, object]:
    return {**value, "receipt_digest": verdict.object_sha256(value)}


def _master() -> dict[str, object]:
    recovery = {
        "receipt_path": "/e/recovery.json",
        "receipt_sha256": "1" * 64,
        "receipt_digest": "2" * 64,
        "final_checkpoint_path": "/e/checkpoint-00000160",
        "final_training_receipt_digest": "3" * 64,
        "final_adapter_model_sha256": "4" * 64,
        "job_id": 135313,
        "slurm_job_success": False,
        "checkpoint_heldout_eligible": True,
    }
    cases = []
    for index, iid in enumerate(verdict.CORE4_IIDS):
        cases.append({
            "iid": iid, "path": f"/e/{iid}.json", "sha256": f"{index + 5:x}" * 64,
            "receipt_digest": f"{index + 6:x}" * 64,
            "decoded_outputs_byte_identical": False,
            "pair_receipt_path": f"/e/{iid}-pair.json",
            "pair_receipt_sha256": f"{index + 7:x}" * 64,
            "pair_receipt_digest": f"{index + 8:x}" * 64,
            "recovery_receipt_sha256": recovery["receipt_sha256"],
            "recovery_receipt_digest": recovery["receipt_digest"],
        })
    return _signed({
        "schema_version": postflight.SCHEMA_VERSION,
        "status": "full160_core4_execution_closed_pending_strict_paired_full_video_review",
        "array_job_id": 136000, "array_task_count": 2,
        "task_completions": [{"task": i, "node": f"auh7-1b-gpu-20{i}", "path": f"/e/task-{i}", "sha256": f"{i + 8:x}" * 64} for i in (0, 1)],
        "training_method_source": {"revision": binder.TRAINING_REVISION, "archive_sha256": binder.TRAINING_ARCHIVE_SHA256},
        "inference_runtime_overlay": {"archive_sha256": "a" * 64, "manifest_sha256": "b" * 64, "manifest_digest": "c" * 64, "heldout_runner_sha256": "d" * 64, "is_training_method_archive": False},
        "trained_adapter": {"adapter_model_sha256": "4" * 64, "training_receipt_sha256": "e" * 64, "training_receipt_digest": "3" * 64, "training_global_step": 160, "training_max_steps": 160},
        "checkpoint_recovery": recovery,
        "cases": cases, "decision_contract": copy.deepcopy(verdict.MASTER_DECISION_CONTRACT),
        "decoded_outputs_byte_identical_count": 0,
        "core4_master_receipt": {"path": "/e/core4.json", "sha256": "e" * 64, "receipt_digest": "f" * 64},
        "full_video_action_and_preservation_review_complete": False,
        "method_success_claimed": False, "method_success_authorized": False,
    })


class EvalRecoveryV2Tests(unittest.TestCase):
    def test_recovery_raw_sha_and_digest_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "recovery.json"
            path.write_bytes(b"sealed\n")
            fake = {
                "schema_version": binder.recovery.SCHEMA_VERSION,
                "receipt_digest": "2" * 64,
                "final_checkpoint": {"path": "/e/checkpoint-00000160", "receipt": {"sha256": "3" * 64, "digest": "4" * 64}, "adapter_model": {"sha256": "5" * 64}},
            }
            with mock.patch.object(binder.recovery, "verify_receipt", return_value=fake):
                row = binder._verify_recovery_binding(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), expected_digest="2" * 64)
                self.assertTrue(row["checkpoint_heldout_eligible"])
                with self.assertRaisesRegex(binder.SourceBindingError, "digest"):
                    binder._verify_recovery_binding(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), expected_digest="9" * 64)
                with self.assertRaisesRegex(binder.SourceBindingError, "SHA"):
                    binder._verify_recovery_binding(path, expected_sha256="9" * 64, expected_digest="2" * 64)

    def test_blank_and_verdict_propagate_recovery_identity(self) -> None:
        master = _master()
        blank = verdict.build_blank_review(master)
        self.assertEqual(blank["checkpoint_recovery"], master["checkpoint_recovery"])
        rows = copy.deepcopy(blank["cases"])
        for row in rows:
            for arm in ("frozen_base", "trained_adapter"):
                row[arm] = {"requested_action_reaches_terminal_and_holds": False, "identity_scene_camera_inventory_preserved": False, "joint_safe": False}
            row["full_video_review_complete"] = True
        unsigned = dict(blank); unsigned.pop("receipt_digest")
        unsigned.update(status="completed_strict_paired_full_video_review", cases=rows, all_four_cases_review_complete=True)
        result = verdict.reduce_review(_signed(unsigned), master)
        self.assertEqual(result["checkpoint_recovery"], master["checkpoint_recovery"])
        self.assertEqual(result["status"], "NO_GO")

    def test_master_rejects_recovery_adapter_mismatch_and_case_split(self) -> None:
        for mutation in ("master", "case"):
            hostile = copy.deepcopy(_master())
            if mutation == "master":
                hostile["checkpoint_recovery"]["final_adapter_model_sha256"] = "9" * 64
            else:
                hostile["cases"][2]["recovery_receipt_digest"] = "9" * 64
            hostile.pop("receipt_digest")
            with self.assertRaisesRegex(verdict.VerdictError, "recovery|case closure"):
                verdict.build_blank_review(_signed(hostile))

    def test_array_checks_recovery_before_output_creation(self) -> None:
        text = (METHOD_ROOT / "scripts" / "auh_eval_seer_full160_core4_array_v2_20260813.sbatch").read_text()
        self.assertLess(text.index('"${recovery_verifier}" verify'), text.index('mkdir -m 0700 -p "${logs_root}"'))
        for token in ("SEER_FM160_RECOVERY_RECEIPT_SHA256", "SEER_FM160_RECOVERY_RECEIPT_DIGEST", "--expected-recovery-receipt-digest"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
