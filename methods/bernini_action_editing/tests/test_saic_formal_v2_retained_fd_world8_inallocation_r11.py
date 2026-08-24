"""Static/hostile contract gates for the r11 in-allocation WORLD8 canary."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "scripts/auh_canary_saic_formal_v2_retained_fd_world8_payload_inallocation_r11.sh"
WRAPPER = ROOT / "scripts/auh_canary_saic_formal_v2_retained_fd_world8_inallocation_r11.sh"
LAUNCHER = ROOT / "tools/launch_saic_formal_v2_retained_fd_world8_canary_inallocation_r11.py"
POSTFLIGHT = ROOT / "tools/postflight_saic_formal_v2_retained_fd_world8_canary_inallocation_r11.py"
MATERIALIZER = ROOT / "tools/materialize_saic_formal_v2_retained_fd_world8_canary_inallocation_r11.py"
GUARD_SHA = "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
EXTERNAL_GUARD = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/releases/"
    "saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10/inputs/"
    "saic_t2v_rendezvous_guard_v2.py"
)
STEP_SCHEMA = "saic-formal-v2-retained-fd-world8-inallocation-step-launch-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InallocationR11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = load(LAUNCHER, "r11_inallocation_launcher")
        cls.postflight = load(POSTFLIGHT, "r11_inallocation_postflight")
        cls.materializer = load(MATERIALIZER, "r11_inallocation_materializer")

    def test_python_sources_parse(self) -> None:
        for path in (LAUNCHER, POSTFLIGHT, MATERIALIZER):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_transitive_hash_pins(self) -> None:
        self.assertEqual(self.launcher.EXPECTED_PAYLOAD_SHA256, sha(PAYLOAD))
        self.assertEqual(self.launcher.EXPECTED_WRAPPER_SHA256, sha(WRAPPER))
        self.assertEqual(self.launcher.EXPECTED_POSTFLIGHT_SHA256, sha(POSTFLIGHT))
        self.assertEqual(self.postflight.EXPECTED_PAYLOAD_SHA256, sha(PAYLOAD))
        self.assertEqual(self.postflight.EXPECTED_WRAPPER_SHA256, sha(WRAPPER))
        self.assertEqual(self.materializer.EXPECTED["payload"][1], sha(PAYLOAD))
        self.assertEqual(self.materializer.EXPECTED["wrapper"][1], sha(WRAPPER))
        self.assertEqual(self.materializer.EXPECTED["launcher"][1], sha(LAUNCHER))
        self.assertEqual(self.materializer.EXPECTED["postflight"][1], sha(POSTFLIGHT))

    def test_external_immutable_guard_only(self) -> None:
        materializer = MATERIALIZER.read_text(encoding="utf-8")
        self.assertEqual(str(self.materializer.EXTERNAL_GUARD), EXTERNAL_GUARD)
        self.assertEqual(self.materializer.EXTERNAL_GUARD_SHA256, GUARD_SHA)
        self.assertIn("local_guard_source_forbidden", materializer)
        for path in (PAYLOAD, WRAPPER, LAUNCHER, POSTFLIGHT, MATERIALIZER):
            self.assertIn(GUARD_SHA, path.read_text(encoding="utf-8"))
        postflight = POSTFLIGHT.read_text(encoding="utf-8")
        self.assertIn("os.O_RDONLY | os.O_NOFOLLOW", postflight)
        self.assertIn("identity(before) != identity(after)", postflight)
        self.assertIn("retained_source", materializer)
        self.assertIn("source_before = os.fstat(source_fd)", materializer)

    def test_single_full8_srun_and_pre_reservation(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertEqual(source.count('str(SRUN), f"--jobid={PARENT_ALLOCATION_JOB_ID}"'), 1)
        for token in (
            '"--nodes=1"', '"--ntasks=1"', '"--cpus-per-task=16"',
            '"--mem=32G"', '"--gres=gpu:mi210:8"', '"--overlap"',
            '"--exact"', '"--export=NONE"',
            '"--input=0"',
            '"--job-name=saic-fv2-fd-w8-inalloc-r11"',
        ):
            self.assertIn(token, source)
        reservation = source.index("os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW")
        invocation = source.index("completed = subprocess.run")
        self.assertLess(reservation, invocation)
        self.assertLess(
            source.index("command.extend(bootstrap_args)"),
            source.index("command_digest = digest(command)"),
        )
        self.assertIn('"exact_srun_argv": command', source)
        self.assertIn('"exact_srun_argv_digest": command_digest', source)
        self.assertIn('len(" ".join(command).encode("ascii")) >= 8192', source)
        self.assertNotIn("/usr/bin/sbatch", source)

    def test_compute_bootstrap_retains_wrapper_and_receipt(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        for token in (
            "wrapper_fd=os.open(wrapper_path,os.O_RDONLY|os.O_NOFOLLOW)",
            "receipt_fd=os.open(receipt_path,os.O_RDWR|os.O_NOFOLLOW)",
            'os.fchmod(receipt_fd,0o444)',
            'f"/proc/self/fd/{wrapper_fd}"',
            'str(COMPUTE_BOOTSTRAP_PYTHON), "-I", "-B", "-c", STDIN_LOADER',
            'input=BOOTSTRAP.encode("ascii")',
            '"compute_bootstrap_transported_over_srun_stdin":True',
            '"compute_bootstrap_stdin_sha256_verified_inside_step":True',
            '"compute_bootstrap_pathname_execution":False',
            '"compute_bootstrap_interpreter":"/usr/bin/python3"',
            'python_fd=os.open(python,os.O_RDONLY|os.O_NOFOLLOW)',
            '"science_python_retained_fd_prepared_for_wrapper":True',
            '"parent_allocation_job_id":parent_job',
            '"step_id":step_id', '"job_step_id":job_step_id',
            '"exact_srun_argv_digest":argv_digest',
        ):
            self.assertIn(token, source)
        self.assertIn(STEP_SCHEMA, source)
        self.assertNotIn('"--compute-bootstrap"', source)
        self.assertNotIn('"-c", BOOTSTRAP', source)
        self.assertIn("hashlib.sha256(raw).hexdigest()!=expected", source)
        self.assertIn("len(raw)!=expected_size", source)
        self.assertIn("sys.argv=[sys.argv[0],*sys.argv[3:]]", source)
        self.assertIn("exec(compile(raw,'inallocation-step-bootstrap','exec')", source)
        self.assertIn("stdout=subprocess.PIPE", source)
        self.assertIn("stderr=subprocess.PIPE", source)
        self.assertIn("inallocation-srun-client-receipt.json", source)
        self.assertIn('"srun_client_returncode": completed.returncode', source)
        self.assertIn("os.umask(0o077)", source)
        self.assertIn('"--open-mode=truncate"', source)

    def test_payload_evidence_is_honestly_step_scoped(self) -> None:
        raw = PAYLOAD.read_bytes()
        self.assertTrue(raw.startswith(b"#!/usr/bin/bash\n"))
        self.assertNotIn(b"perl: warning", raw)
        source = raw.decode("utf-8")
        for token in (
            '"inallocation_step_wrapper_executed_from_retained_fd": True',
            '"step_success": None', '"parent_job_success": None',
            '"step_id": step_id', '"job_step_id": job_step_id',
            '"step_terminal_verified": False',
            '"canonical_runtime_origin_verified": True',
            '"canonical_runtime_origin_sha256": runtime_sha',
            '"canonical_runtime_origin_path"',
            'if module_name == canonical_module_name:',
            'continue',
            'os.execve(fd,[sys.argv[2],*sys.argv[3:]],os.environ)',
            '"science_python_fd_exec_argv0_preserved": True',
            '"torchrun_no_python_worker_entrypoint_retained_fd": True',
            '--no-python --nproc-per-node=4',
            '"${stage0_python_fd_path}" -I -B "${guard_fd_path}" worker',
        ):
            self.assertIn(token, source)
        self.assertNotIn("submitted_job", source)
        self.assertNotIn("submission_success", source)

    def test_postflight_queries_exact_step_without_dash_x(self) -> None:
        source = POSTFLIGHT.read_text(encoding="utf-8")
        self.assertIn('str(EXPECTED_SACCT), "-j", job_step_id', source)
        self.assertIn('if "-X" in command:', source)
        self.assertNotIn('"-X",', source)
        self.assertIn('expected_submit_line = " ".join(expected_command)', source)
        self.assertIn(
            'expected_basename_submit_line = " ".join(["srun", *expected_command[1:]])',
            source,
        )
        self.assertNotIn('" ".join(expected_command[14:])', source)
        for token in (
            'accounting["State"] != "COMPLETED"',
            'accounting["JobName"] != "saic-fv2-fd-w8-inalloc-r11"',
            'accounting["ExitCode"] != "0:0"',
            'tres.get("cpu") != "16"', 'tres.get("mem") != "32G"',
            'tres.get("gres/gpu:mi210") != "8"',
            '"step_success": True', '"parent_job_success": None',
            '"parent_job_terminal_state_not_claimed": True',
            "SAIC_FV2_FD_WORLD8_INALLOCATION_R11_PASS",
            "os.O_RDONLY | os.O_NOFOLLOW",
            "both_logs_sealed_0400_after_retained_read",
            "srun_client_receipt_file_sha256",
        ):
            self.assertIn(token, source)

    def test_r10_science_closure_preserved(self) -> None:
        for path in (PAYLOAD, POSTFLIGHT):
            source = path.read_text(encoding="utf-8")
            for token in (
                "archive_member_count", "864", "archive_regular_file_count",
                "853", "archive_directory_count", "11",
                "runtime_origin_project_module_count", "14",
                "two_concurrent_world4_on_one_requested_8mi210_node",
            ):
                self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
