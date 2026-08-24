#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_run_graft_phase_a_active14_transaction_world8_v1.sbatch"
WRAPPER = METHOD_ROOT / "scripts/auh_submit_graft_phase_a_active14_transaction_world8_v1.sh"
PLAN = METHOD_ROOT / "assets/graft_phase_a_active14_transaction_world8_plan_v1.json"
CORE = METHOD_ROOT / "train_graft_phase_a_active14_transaction_v1.py"
RUNNER = METHOD_ROOT / "run_graft_phase_a_active14_transaction_gpu_v1.py"
FIELD_CORE = METHOD_ROOT / "graft_phase_a_field14_exact40_v1.py"
FIELD_RUNNER = METHOD_ROOT / "run_graft_phase_a_field14_exact40_gpu_v1.py"
SHORT_RUNNER = METHOD_ROOT / "run_graft_phase_a_a_lite_short_gpu_v1.py"


class Active14LauncherTests(unittest.TestCase):
    def test_shell_syntax_resources_and_dependency_are_exact(self) -> None:
        for path in (LAUNCHER, WRAPPER):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        for line in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=64",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --time=72:00:00",
        ):
            self.assertIn(line, launcher)
        self.assertIn('f"--dependency=afterok:{field_job_id}"', wrapper)
        self.assertIn('"--cpus-per-task=64"', wrapper)
        self.assertIn('"--time=72:00:00"', wrapper)
        self.assertNotIn("--export=ALL", wrapper)
        self.assertIn("launcher_transport", wrapper)
        self.assertIn("sbatch_transport", wrapper)
        self.assertIn("pass_fds=(launcher_fd,sbatch_fd)", wrapper)
        self.assertIn("reserve_receipt(output_fd,submission_name)", wrapper)
        self.assertIn("publish_reserved_receipt", wrapper)
        self.assertIn("provisional_non_success_inode_prevents_ambiguous_resubmission", wrapper)

    def test_launcher_and_wrapper_hash_cascade_is_current(self) -> None:
        launcher_sha = hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()
        wrapper = WRAPPER.read_text(encoding="utf-8")
        match = re.search(r"readonly required_launcher_sha256=([0-9a-f]{64})", wrapper)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), launcher_sha)
        self.assertEqual(
            launcher_sha,
            "d896b87dbc95dbcb65b80a0d635bc1dfd577f6a30a0dfc1d726ca23e1432efdb",
        )

    def test_runtime_source_pins_and_exact27_closure_are_explicit(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "readonly required_source_commit=037f7a8e296bf2d0bf7da63ccc9eb8b7ed6608fd",
            launcher,
        )
        expected = {
            "required_core_sha": hashlib.sha256(CORE.read_bytes()).hexdigest(),
            "required_runner_sha": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            "required_field_core_sha": hashlib.sha256(FIELD_CORE.read_bytes()).hexdigest(),
            "required_field_runner_sha": hashlib.sha256(FIELD_RUNNER.read_bytes()).hexdigest(),
            "required_short_runner_sha": hashlib.sha256(SHORT_RUNNER.read_bytes()).hexdigest(),
        }
        for name, digest in expected.items():
            self.assertIn(f"readonly {name}={digest}", launcher)
        self.assertIn(
            '"bernini-graft-phase-a-active14-runtime-python-closure-v1"',
            launcher,
        )
        self.assertIn('manifest.get("source_git_commit")!=source_commit', launcher)
        self.assertIn(
            'manifest.get("selection")!="commit-037f7a8-recursive-runtime-import-closure-v1"',
            launcher,
        )
        self.assertIn('len(files)!=27', launcher)
        self.assertIn('len(members)!=27', launcher)
        expected_names = {
            "train_graft_phase_a_active14_transaction_v1.py",
            "run_graft_phase_a_active14_transaction_gpu_v1.py",
            "graft_phase_a_field14_exact40_v1.py",
            "run_graft_phase_a_field14_exact40_gpu_v1.py",
            "run_graft_phase_a_a_lite_short_gpu_v1.py",
            "train_graft_phase_a_a_lite_short_v1.py",
        }
        for name in expected_names:
            self.assertIn(f'"{name}"', launcher)
        self.assertIn(
            "--expected-field14-source-commit \"${required_field_source_commit}\"",
            launcher,
        )
        self.assertIn(
            'UPSTREAM_SCHEMA_VERSION = "bernini-graft-phase-a-field14-world8-parent-v2"',
            CORE.read_text(encoding="utf-8"),
        )

    def test_plan_is_canonical_and_numeric_dependency_is_final(self) -> None:
        raw = PLAN.read_bytes()
        value = json.loads(raw.decode("ascii"))
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
        self.assertEqual(raw, canonical)
        dependency = value["field14_dependency"]
        self.assertEqual(
            value["runtime"]["short_runner_sha256"],
            hashlib.sha256(SHORT_RUNNER.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            value["runtime"]["field14_source_commit"],
            "f9ef982e976ad19ed81ed075d33c9221952945e4",
        )
        self.assertEqual(dependency["kind"], "afterok")
        self.assertEqual(
            dependency["receipt_sha256_policy"],
            "derive-from-stable-sealed-file-after-afterok",
        )
        job_id = dependency["job_id"]
        receipt_path = dependency["receipt_path"]
        self.assertEqual(job_id, "133530")
        self.assertTrue(job_id.isdecimal())
        self.assertEqual(
            receipt_path,
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/"
            "phase_a_field14_exact40_world8_v1/runs/"
            "source-f9ef982-launcher-05813e6-r2/receipt.json",
        )
        self.assertNotIn("__FIELD14_", raw.decode("ascii"))
        runner_source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("args.expected_upstream_field14_job_id.isdecimal()", runner_source)
        self.assertIn("derive-from-stable-sealed-file-after-afterok", runner_source)

    def test_runtime_derives_unknown_receipt_sha_only_after_afterok(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('field_receipt_sha="$(file_sha256 "${field_receipt}")"', launcher)
        self.assertIn(
            '--expected-upstream-field14-receipt-sha256 "${field_receipt_sha}"',
            launcher,
        )
        self.assertIn('"receipt_may_be_absent_at_submission":True', wrapper)
        self.assertIn("field_anchor,field_anchor_fd,field_anchor_identity", wrapper)
        self.assertIn('field_receipt.name!="receipt.json"', wrapper)
        self.assertIn('"weights_inherited":False', wrapper)
        self.assertIn('"dependency_is_queue_gate_only"', wrapper)

    def test_no_checkpoint_writer_or_false_job_success(self) -> None:
        combined = LAUNCHER.read_text(encoding="utf-8") + WRAPPER.read_text(
            encoding="utf-8"
        )
        self.assertNotIn("torch.save", combined)
        self.assertIn('"job_success":None', combined)
        self.assertIn('"training_authority":False', combined)
        self.assertIn('"scientific_success_claimed":False', combined)
        self.assertIn('"checkpoint_written":False', combined)


if __name__ == "__main__":
    unittest.main()
