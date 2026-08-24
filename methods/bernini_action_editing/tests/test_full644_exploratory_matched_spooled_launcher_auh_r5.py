from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_spooled_launcher_auh_r5 as launcher


MINIMAL_CAPTURED_RUNNER = b'''from __future__ import annotations
import hashlib,json,os,sys

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)

def identity(value):
    return {"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"mode":value.st_mode,"nlink":value.st_nlink,"rdev":value.st_rdev,"size":value.st_size,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}

expected_environment={"FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY","SLURM_JOB_ID","SLURM_STEP_ID","SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_NODE","SLURM_STEP_GPUS","SLURM_NNODES","SLURM_STEP_NUM_NODES","SLURM_JOB_NODELIST","SLURM_STEP_NODELIST"}
if set(os.environ)!=expected_environment:
    raise RuntimeError("captured runner environment differs")
if not (sys.flags.isolated==1 and sys.flags.no_site==1 and sys.flags.dont_write_bytecode==1 and sys.flags.ignore_environment==1):
    raise RuntimeError("captured runner interpreter flags differ")
entry=json.loads(os.environ["FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY"])
expected_fields={"schema_version","runner_fd","runner_path","runner_sha256","runner_identity","python_fd","python_path","python_sha256","python_identity","release_digest","bootstrap_sha256","entry_method","slurm_export_none_required","bash_privileged_startup_required","captured_source_entry","authority_digest"}
if set(entry)!=expected_fields or entry["schema_version"]!="full644-exploratory-matched-captured-runner-entry-authority-v1" or entry["entry_method"]!="slurm-spooled-or-trusted-stdin-held-python-fd-v1" or entry["captured_source_entry"] is not True or entry["slurm_export_none_required"] is not True or entry["bash_privileged_startup_required"] is not True:
    raise RuntimeError("captured runner entry schema differs")
unsigned=dict(entry); observed_digest=unsigned.pop("authority_digest")
if hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest()!=observed_digest:
    raise RuntimeError("captured runner entry digest differs")
for role in ("runner","python"):
    descriptor=entry[role+"_fd"]
    if type(descriptor) is not int or descriptor<3 or os.get_inheritable(descriptor):
        raise RuntimeError("captured runner FD seal differs")
    if identity(os.fstat(descriptor))!=entry[role+"_identity"]:
        raise RuntimeError("captured runner FD identity differs")
if entry["runner_path"]!=__file__ or sys.argv[0]!=__file__:
    raise RuntimeError("captured runner source entry differs")
print("FULL644_ROOT_BOOTSTRAP_E2E_OK",flush=True)
'''


class SpooledLauncherV2Tests(unittest.TestCase):
    def test_production_source_pins_match_current_repo_bytes(self) -> None:
        sources = {
            "runner": "full644_exploratory_matched_runner_auh_r5.py",
            "bridge": "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
            "adapter": "full644_exploratory_matched_infer_adapter_v2.py",
            "eval_v1": "full644_exploratory_matched_eval_v1.py",
            "eval_v2": "full644_exploratory_matched_eval_v2.py",
            "model_authority": (
                "action_preservation_decoded_eval_model_authority_v2.py"
            ),
        }
        for role, basename in sources.items():
            with self.subTest(role=role):
                observed = hashlib.sha256(
                    (MODULE_ROOT / basename).read_bytes()
                ).hexdigest()
                self.assertEqual(observed, launcher.EXPECTED_STATIC_SHA256[role])

    def fixture(self, root: Path) -> tuple[dict, dict[str, str]]:
        files: dict[str, str] = {}
        static_roles = tuple(launcher.EXPECTED_STATIC_SHA256)
        for index, role in enumerate(static_roles):
            path = (root / f"{role}.bin").resolve()
            path.write_bytes(
                MINIMAL_CAPTURED_RUNNER
                if role == "runner"
                else f"pinned-{role}-{index}\n".encode("utf-8")
            )
            path.chmod(0o444)
            files[role] = str(path)
        python_path = (root / "python").resolve()
        shutil.copyfile(Path(sys.executable).resolve(strict=True), python_path)
        python_path.chmod(0o555)
        ffmpeg = (root / "ffmpeg").resolve()
        ffmpeg.write_bytes(b"fixture ffmpeg\n")
        ffmpeg.chmod(0o555)
        plan = (root / "plan.json").resolve()
        plan.write_bytes(b'{"fixture":true}\n')
        plan.chmod(0o444)
        for name in ("model", "bernini", "veomni"):
            (root / name).mkdir()
        value = {
            "schema_version": launcher.INPUT_SCHEMA,
            "entry_mode": "trusted_stdin",
            **files,
            "python": str(python_path),
            "ffmpeg": str(ffmpeg),
            "plan": str(plan),
            "output_report": str((root / "report.json").resolve()),
            "runner_attestation": str((root / "attestation.json").resolve()),
            "model_root": str((root / "model").resolve()),
            "bernini_root": str((root / "bernini").resolve()),
            "veomni_root": str((root / "veomni").resolve()),
            "authority_root": str((root / "authority").resolve()),
            "rank_cache_root": str((root / "rank-cache").resolve()),
            "holder_job_id": "143812",
            "expected_node": "auh7-1b-gpu-293",
            "campaign_mode": launcher.CASE00_CANARY_CAMPAIGN,
        }
        expected = {
            role: hashlib.sha256(Path(value[role]).read_bytes()).hexdigest()
            for role in static_roles
        }
        return value, expected

    def test_payload_hard_pins_release_and_forbids_named_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            with mock.patch.object(
                launcher, "EXPECTED_STATIC_SHA256", expected
            ):
                release, payload = launcher.build_release(value)
            text = payload.decode("utf-8")
            self.assertEqual(release["expected_allocation_gpu_count"], 8)
            self.assertEqual(release["selected_task_ids"], list(launcher.CANARY_TASK_IDS))
            self.assertFalse(release["formal_full16_report"])
            self.assertTrue(
                release["slurm_environment_contract"][
                    "caller_synthesized_slurm_facts_forbidden"
                ]
            )
            self.assertTrue(release["runner_is_compiled_from_captured_fd_bytes"])
            for role, option in (
                ("torch_local_agent_source", "--torch-local-agent-source"),
                (
                    "torch_dynamic_rendezvous_source",
                    "--torch-dynamic-rendezvous-source",
                ),
                (
                    "torch_multiprocessing_api_source",
                    "--torch-multiprocessing-api-source",
                ),
            ):
                self.assertIn(role, release["identities"])
                self.assertEqual(
                    release["runner_arguments"][
                        release["runner_arguments"].index(option) + 1
                    ],
                    value[role],
                )
            self.assertIn("exec -c", text)
            self.assertIn("-I -S -B -c", text)
            self.assertNotIn("$@", text)
            self.assertIn("named payload execution forbidden", text)
            self.assertIn(
                "if shopt -q varredir_close 2>/dev/null; then "
                "shopt -u varredir_close; fi",
                text,
            )
            self.assertIn("if len(sys.argv)!=15", launcher.ROOT_BOOTSTRAP)
            self.assertIn(
                'spec.get("schema_version")!="' + launcher.SCHEMA + '"',
                launcher.ROOT_BOOTSTRAP,
            )
            checked = subprocess.run(
                ["/bin/bash", "-n"],
                input=payload,
                check=False,
                capture_output=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_unsupported_varredir_close_is_a_safe_noop(self) -> None:
        checked = subprocess.run(
            [
                "/bin/bash",
                "-c",
                "set -e; if shopt -q varredir_close 2>/dev/null; then "
                "shopt -u varredir_close; fi; printf PASS",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(checked.stdout, "PASS")

    def test_materialize_is_create_only_and_receipt_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            launch_input = root / "launch-input.json"
            launch_input.write_bytes(launcher.canonical_json_bytes(value) + b"\n")
            launch_input.chmod(0o444)
            payload = root / "payload.sh"
            receipt = root / "payload.receipt.json"
            with mock.patch.object(
                launcher, "EXPECTED_STATIC_SHA256", expected
            ):
                observed = launcher.materialize(
                    str(launch_input), str(payload), str(receipt)
                )
                with self.assertRaises(launcher.RootLaunchReleaseError):
                    launcher.materialize(
                        str(launch_input), str(payload), str(receipt)
                    )
            self.assertEqual(stat.S_IMODE(payload.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
            self.assertEqual(
                receipt.read_bytes(),
                launcher.canonical_json_bytes(observed) + b"\n",
            )
            self.assertFalse(observed["submission_or_execution_performed"])
            environment = {
                "SLURM_JOB_ID": "143812",
                "SLURM_STEP_ID": "1",
                "SLURM_GPUS_ON_NODE": "8",
                "SLURM_GPUS_PER_NODE": "8",
                "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
                "SLURM_NNODES": "1",
                "SLURM_STEP_NUM_NODES": "1",
                "SLURM_JOB_NODELIST": "auh7-1b-gpu-293",
                "SLURM_STEP_NODELIST": "auh7-1b-gpu-293",
            }
            named = subprocess.run(
                ["/bin/bash", "-p", str(payload)],
                check=False,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(named.returncode, 92)

    def test_shell_gate_requires_real_auh_fields_and_rejects_legacy_injection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            with mock.patch.object(
                launcher, "EXPECTED_STATIC_SHA256", expected
            ):
                _, payload = launcher.build_release(value)
            environment = {
                "SLURM_JOB_ID": "143812",
                "SLURM_STEP_ID": "60",
                "SLURM_GPUS_ON_NODE": "8",
                "SLURM_GPUS_PER_NODE": "8",
                "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
                "SLURM_NNODES": "1",
                "SLURM_STEP_NUM_NODES": "1",
                "SLURM_JOB_NODELIST": "auh7-1b-gpu-293",
                "SLURM_STEP_NODELIST": "auh7-1b-gpu-293",
            }
            for legacy in ("SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"):
                hostile = dict(environment)
                hostile[legacy] = "1"
                observed = subprocess.run(
                    ["/bin/bash", "-p", "-s"],
                    input=payload,
                    check=False,
                    capture_output=True,
                    env=hostile,
                )
                self.assertEqual(observed.returncode, 99, legacy)
            for missing, expected_exit in (
                ("SLURM_GPUS_PER_NODE", 95),
                ("SLURM_STEP_GPUS", 96),
                ("SLURM_NNODES", 96),
                ("SLURM_STEP_NUM_NODES", 96),
                ("SLURM_JOB_NODELIST", 97),
                ("SLURM_STEP_NODELIST", 97),
            ):
                hostile = dict(environment)
                hostile.pop(missing)
                observed = subprocess.run(
                    ["/bin/bash", "-p", "-s"],
                    input=payload,
                    check=False,
                    capture_output=True,
                    env=hostile,
                )
                self.assertEqual(observed.returncode, expected_exit, missing)
            for hostile_step_id in (
                "",
                "batch",
                "extern",
                "01",
                "+1",
                "-1",
                "1.2",
                "1 2",
            ):
                hostile = dict(environment)
                hostile["SLURM_STEP_ID"] = hostile_step_id
                observed = subprocess.run(
                    ["/bin/bash", "-p", "-s"],
                    input=payload,
                    check=False,
                    capture_output=True,
                    env=hostile,
                )
                self.assertEqual(observed.returncode, 95, hostile_step_id)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and Path("/proc/self/fd").is_dir(),
        "root bootstrap entry requires Linux /proc",
    )
    def test_trusted_stdin_executes_captured_runner_from_held_fds(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            with mock.patch.object(
                launcher, "EXPECTED_STATIC_SHA256", expected
            ):
                _, payload = launcher.build_release(value)
            environment = {
                "SLURM_JOB_ID": "143812",
                "SLURM_STEP_ID": "1",
                "SLURM_GPUS_ON_NODE": "8",
                "SLURM_GPUS_PER_NODE": "8",
                "SLURM_STEP_GPUS": "0,1,2,3,4,5,6,7",
                "SLURM_NNODES": "1",
                "SLURM_STEP_NUM_NODES": "1",
                "SLURM_JOB_NODELIST": "auh7-1b-gpu-293",
                "SLURM_STEP_NODELIST": "auh7-1b-gpu-293",
            }
            streamed = subprocess.run(
                ["/bin/bash", "-p", "-s"],
                input=payload,
                check=False,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(streamed.returncode, 0, streamed.stderr)
            self.assertEqual(
                streamed.stdout.decode("utf-8").strip(),
                "FULL644_ROOT_BOOTSTRAP_E2E_OK",
            )

    def test_static_pin_and_fresh_path_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            value, expected = self.fixture(root)
            runner_path = Path(value["runner"])
            runner_path.chmod(0o600)
            runner_path.write_bytes(b"hostile runner\n")
            runner_path.chmod(0o444)
            with mock.patch.object(
                launcher, "EXPECTED_STATIC_SHA256", expected
            ), self.assertRaisesRegex(
                launcher.RootLaunchReleaseError, "runner.*SHA"
            ):
                launcher.build_release(value)


if __name__ == "__main__":
    unittest.main()
