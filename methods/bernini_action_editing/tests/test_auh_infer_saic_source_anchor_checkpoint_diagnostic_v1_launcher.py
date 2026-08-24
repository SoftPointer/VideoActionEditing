#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import postflight_saic_source_anchor_checkpoint_diagnostic_v1 as postflight  # noqa: E402
import submit_saic_source_anchor_checkpoint_diagnostic_v1 as submitter  # noqa: E402


LAUNCHER = METHOD_ROOT / "scripts/auh_infer_saic_source_anchor_checkpoint_diagnostic_v1.sbatch"
SUBMITTER = TOOLS_ROOT / "submit_saic_source_anchor_checkpoint_diagnostic_v1.py"
POSTFLIGHT = TOOLS_ROOT / "postflight_saic_source_anchor_checkpoint_diagnostic_v1.py"


class SourceAnchorDiagnosticLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.submitter = SUBMITTER.read_text(encoding="utf-8")
        cls.postflight = POSTFLIGHT.read_text(encoding="utf-8")

    def test_first_run_is_one_world4_dynamic_rendezvous_canary(self) -> None:
        self.assertTrue(self.launcher.startswith("#!/usr/bin/bash\n"))
        self.assertIn("#SBATCH --gres=gpu:mi210:4", self.launcher)
        self.assertIn("--nproc_per_node=4", self.launcher)
        self.assertIn("--rdzv_endpoint=127.0.0.1:0", self.launcher)
        self.assertIn("--rdzv_backend=c10d", self.launcher)
        self.assertNotIn("master_port=$((", self.launcher)
        self.assertNotIn("--master_port=", self.launcher)
        self.assertIn("cells=7 exact81=true exact40=true", self.launcher)
        self.assertNotIn("#SBATCH --qos=", self.launcher)

    def test_slurm_gpu_visibility_is_preserved_and_receipted(self) -> None:
        self.assertIn("gpu_visibility_source=ROCR_VISIBLE_DEVICES", self.launcher)
        self.assertIn("gpu_visibility_source=CUDA_VISIBLE_DEVICES", self.launcher)
        self.assertIn('export ROCR_VISIBLE_DEVICES="${gpu_visibility}"', self.launcher)
        self.assertIn(
            "unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL",
            self.launcher,
        )
        self.assertNotIn(
            "unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES", self.launcher
        )
        self.assertIn("all_four_gpu_mappings_distinct", self.postflight)

    def test_archive_is_retained_recursively_closed_and_origin_checked(self) -> None:
        for anchor in (
            'exec {archive_fd}<"${source_archive}"',
            '"/proc/$$/fd/${archive_fd}"',
            "source archive member safety/uniqueness differs",
            "source archive six-leaf release closure differs",
            "recursive project import-origin closure differs",
            'importlib.import_module("infer_saic_source_anchor_checkpoint_diagnostic_v1")',
            "test_infer_saic_source_anchor_checkpoint_diagnostic_v1.py",
            "test_auh_infer_saic_source_anchor_checkpoint_diagnostic_v1_launcher.py",
        ):
            self.assertIn(anchor, self.launcher)
        self.assertNotIn("tar --no-same-owner", self.launcher)

    def test_release_manifest_is_the_only_scientific_submit_anchor(self) -> None:
        self.assertEqual(submitter.SCHEMA_VERSION, postflight.SUBMISSION_SCHEMA)
        self.assertEqual(len(submitter.EXPORT_NAMES), len(set(submitter.EXPORT_NAMES)))
        self.assertGreaterEqual(len(submitter.EXPORT_NAMES), 50)
        self.assertEqual(
            submitter.EXPECTED_RELEASE_MANIFEST_SHA256,
            "UNRESOLVED_AFTER_AUH_RELEASE_MATERIALIZATION",
        )
        self.assertIn("UNRESOLVED_RELEASE_PIN", self.submitter)
        self.assertIn("resolved_release", self.submitter)
        self.assertIn("release directory recursive file closure differs", self.submitter)
        self.assertIn("/proc/self/fd/{wrapper_descriptor}", self.submitter)
        self.assertIn("retained_wrapper_device", self.submitter)
        self.assertIn("retained_wrapper_inode", self.submitter)

    def test_submitter_is_exactly_once_and_receipt_binds_all_exports(self) -> None:
        self.assertLess(
            self.submitter.index("reserved_before_sbatch_ambiguous_never_retry"),
            self.submitter.index("completed = subprocess.run(\n            command,"),
        )
        self.assertEqual(self.submitter.count("completed = subprocess.run("), 2)
        self.assertIn('"exports": exports', self.submitter)
        self.assertIn('"automatic_retry_allowed": False', self.submitter)
        self.assertIn("terminal submission receipt retained-FD reread differs", self.submitter)
        self.assertNotIn("unlink(", self.submitter)
        self.assertIn("os._exit(0)", self.submitter)

    def test_postflight_requires_exact_accounting_logs_sentinel_and_rendezvous(self) -> None:
        for anchor in (
            "SubmitLine%8192",
            '"billing": "32"',
            '"cpu": "32"',
            '"mem": "256G"',
            '"node": "1"',
            "retained_wrapper_fd",
            "terminal Slurm stdout/stderr/sentinel closure differs",
            "sentinel_exact_once_and_final_line",
            "validate_rendezvous_evidence",
            "formal Stage-A exact32 history changed",
            "action raw/clean/full/token/embedding collapsed to no-op",
        ):
            self.assertIn(anchor, self.postflight)
        self.assertEqual(len(postflight.EXPECTED_NAMES), 19)
        self.assertFalse(postflight.RUNTIME_AUTHORITY["stage_b_runtime_available"])
        self.assertFalse(postflight.POSTFLIGHT_AUTHORITY["scientific_success"])

    def test_sacct_accepts_only_exact_terminal_row_and_submit_line(self) -> None:
        exports = {
            "SAIC_ANCHOR_DIAG_SLURM_LOG_DIR": "/logs",
            "SAIC_ANCHOR_DIAG_RELEASE_MANIFEST": "/release.json",
        }
        submission = {
            "request": {"qos": "bgqos"},
            "exports": exports,
            "outputs": {},
            "single_attempt_boundary": {
                "exact_export_names": list(exports),
            },
        }
        submit_line = " ".join(
            [
                "/usr/bin/sbatch",
                "--parsable",
                "--qos=bgqos",
                "--output=/logs/saic-anchor-diag-v2-%j.out",
                "--error=/logs/saic-anchor-diag-v2-%j.err",
                "--export=NONE,"
                + ",".join(f"{name}={value}" for name, value in exports.items()),
                "/proc/self/fd/17",
            ]
        )
        good = mock.Mock(
            returncode=0,
            stdout=(
                "123|saic-anchor-diag-v2|faculty|bgqos|COMPLETED|0:0|"
                "billing=32,cpu=32,gres/gpu:mi210=4,gres/gpu=4,mem=256G,node=1|"
                f"auh-node|start|end|01:00:00|{submit_line}\n"
            ).encode("ascii"),
            stderr=b"",
        )
        with mock.patch.object(postflight.subprocess, "run", return_value=good):
            observed = postflight.observe_sacct(
                Path("/usr/bin/sacct"), job_id="123", submission=submission
            )
        self.assertTrue(observed["terminal_success"])
        self.assertTrue(observed["exact_submit_line"])
        self.assertEqual(observed["retained_wrapper_fd"], 17)

        bad = mock.Mock(
            returncode=0,
            stdout=good.stdout.replace(b"cpu=32", b"cpu=31"),
            stderr=b"",
        )
        with mock.patch.object(postflight.subprocess, "run", return_value=bad):
            with self.assertRaisesRegex(SystemExit, "topology/SubmitLine"):
                postflight.observe_sacct(
                    Path("/usr/bin/sacct"), job_id="123", submission=submission
                )


if __name__ == "__main__":
    unittest.main()
