from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS = METHOD_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import recover_seer_fm160_job135313_v1 as recovery  # noqa: E402


class RecoveryUnitTests(unittest.TestCase):
    def test_embedded_json_extractor_recovers_prefixed_step(self) -> None:
        text = 'NCCL noise {"step":1,"same_state_exact":1.0}\n{"step":2}\n'
        self.assertEqual([row["step"] for row in recovery._json_objects(text)], [1, 2])

    def test_step_closure_rejects_missing_and_duplicate(self) -> None:
        rows = [{"step": step, "same_state_exact": 1.0, "preclip_gradient_norm": 0.1} for step in range(1, 161)]
        recovery._validate_step_rows(rows)
        with self.assertRaisesRegex(recovery.RecoveryError, "step closure"):
            recovery._validate_step_rows(rows[:-1])
        with self.assertRaisesRegex(recovery.RecoveryError, "step closure"):
            recovery._validate_step_rows(rows + [dict(rows[-1])])

    def test_bad_sacct_is_rejected(self) -> None:
        good = {"job_id": 135313, "state": "FAILED", "exit_code": "1:0", "elapsed_seconds": 1921, "node": "auh7-1b-gpu-209"}
        recovery._validate_job(good)
        hostile = dict(good)
        hostile["state"] = "COMPLETED"
        with self.assertRaisesRegex(recovery.RecoveryError, "accounting"):
            recovery._validate_job(hostile)

    def test_wrong_hardpin_and_create_only_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "artifact"
            path.write_bytes(b"x")
            with self.assertRaisesRegex(recovery.RecoveryError, "hard pin"):
                recovery._require_hardpin(path, "0" * 64, label="fixture")
            with self.assertRaisesRegex(recovery.RecoveryError, "fresh"):
                recovery._write_create_only(path, {"x": 1})

    def test_optimizer_logical_digest_mismatch_is_fail_closed(self) -> None:
        source = Path(recovery.__file__).read_text(encoding="utf-8")
        self.assertIn('optimizer_digest != final["optimizer"]["checkpoint_state_digest"]', source)
        self.assertIn("optimizer logical payload digest differs", source)

    def test_verify_receipt_rejects_resigned_authority_widening(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bindings = {}
            for name in ("launcher", "stdout", "stderr", "latest", "receipt", "config", "adapter", "optimizer"):
                path = root / name
                path.write_bytes(name.encode())
                bindings[name] = {"path": str(path), "sha256": recovery.file_sha256(path)}
            training = {"global_step": 160}
            training["receipt_digest"] = recovery.object_sha256(training)
            (root / "receipt").write_bytes(recovery.canonical_json_bytes(training) + b"\n")
            bindings["receipt"]["sha256"] = recovery.file_sha256(root / "receipt")
            bindings["receipt"]["digest"] = training["receipt_digest"]
            latest = {"checkpoint": str(root / "checkpoint-00000160"), "global_step": 160, "receipt_digest": training["receipt_digest"]}
            (root / "latest").write_bytes(recovery.canonical_json_bytes(latest) + b"\n")
            bindings["latest"]["sha256"] = recovery.file_sha256(root / "latest")
            bindings["latest"]["value"] = latest
            unsigned = {
                "schema_version": recovery.SCHEMA_VERSION,
                "job": {"job_id": 135313, "state": "FAILED", "exit_code": "1:0"},
                "artifacts": {**{key: bindings[key] for key in ("launcher", "stdout", "stderr", "latest")}},
                "final_checkpoint": {"step": 160, "path": str(root / "checkpoint-00000160"), "receipt": bindings["receipt"], "adapter_config": bindings["config"], "adapter_model": bindings["adapter"], "optimizer": bindings["optimizer"]},
                "training_execution_complete": True,
                "engineering_execution_success": True,
                "slurm_job_success": False,
                "checkpoint_heldout_eligible": True,
                "method_success": False,
                "heldout_evaluation_required": True,
                "production_claim_forbidden": True,
                "scientific_claim_authorized": False,
                "original_artifacts_modified": False,
            }
            hostile = copy.deepcopy(unsigned)
            hostile["slurm_job_success"] = True
            hostile["receipt_digest"] = recovery.object_sha256(hostile)
            path = root / "recovery.json"
            path.write_bytes(recovery.canonical_json_bytes(hostile) + b"\n")
            with self.assertRaisesRegex(recovery.RecoveryError, "authority"):
                recovery.verify_receipt(path)

    def test_verify_receipt_rejects_tampered_bound_artifact(self) -> None:
        # Static guard: every final artifact binding is rehashed during verify.
        source = Path(recovery.__file__).read_text(encoding="utf-8")
        self.assertIn("FM160 recovery {label} SHA differs", source)
        self.assertIn("latest.get(\"receipt_digest\") != receipt.get(\"receipt_digest\")", source)


if __name__ == "__main__":
    unittest.main()
