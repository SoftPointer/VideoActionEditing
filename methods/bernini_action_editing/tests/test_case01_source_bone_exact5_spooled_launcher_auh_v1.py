#!/usr/bin/env python3
"""Hostile contract tests for the independent case01 exact5 launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "case01_source_bone_exact5_spooled_launcher_auh_v1.py"
)


def load_launcher():
    name = "_test_exact5_launcher_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("launcher import specification differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MOCK_RUNNER = b'''import hashlib,json,os,sys
raw=os.environ["FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY"]
entry=json.loads(raw)
unsigned=dict(entry); claimed=unsigned.pop("authority_digest")
canonical=lambda value: json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
assert claimed==hashlib.sha256(canonical(unsigned)).hexdigest()
assert entry["entry_method"]=="slurm-spooled-or-trusted-stdin-held-python-fd-v1"
assert entry["runner_path"]==__file__ and os.fstat(entry["runner_fd"]).st_ino==os.lstat(__file__).st_ino
assert sys.flags.isolated==1 and sys.flags.no_site==1 and sys.dont_write_bytecode
assert len(sys.argv)>2 and len(sys.argv[1:])%2==0
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
assert args["--campaign-mode"]=="case01-source-bone-exact5-r64-canary"
payload=b"MOCK_EXACT5_CAPTURED_WRAPPER_REACHED\\n"
fd=os.open(args["--output-report"],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
try: os.write(fd,payload); os.fsync(fd)
finally: os.close(fd)
'''


class Fixture:
    def __init__(self, module, base: Path):
        self.module = module
        self.root = base.resolve()
        self.method = self.root / "release/methods/bernini_action_editing"
        self.external = self.root / "external"
        self.final = self.root / "final"
        self.runtime = self.root / "runtime"
        for directory in (self.method, self.external, self.final, self.runtime):
            directory.mkdir(parents=True, exist_ok=True)

        names = {
            "runner": "case01_source_bone_exact5_runner_v1.py",
            "frozen_runner": "full644_exploratory_matched_runner_auh_r5.py",
            "exact5_eval": "case01_source_bone_exact5_eval_v1.py",
            "bridge": "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
            "adapter": "full644_exploratory_matched_infer_adapter_auh_r5f.py",
            "base_adapter": "full644_exploratory_matched_infer_adapter_v2.py",
            "eval_v1": "full644_exploratory_matched_eval_v1.py",
            "eval_v2": "full644_exploratory_matched_eval_v2.py",
            "model_authority": "action_preservation_decoded_eval_model_authority_v2.py",
        }
        self.paths = {}
        for role, name in names.items():
            path = self.method / name
            raw = MOCK_RUNNER if role == "runner" else (f"# {role}\n".encode())
            path.write_bytes(raw)
            path.chmod(0o444)
            self.paths[role] = path

        for role in (
            "torchrun_source",
            "torchrun_handler_source",
            "torch_local_agent_source",
            "torch_dynamic_rendezvous_source",
            "torch_multiprocessing_api_source",
            "model_manifest",
        ):
            path = self.external / (role + ".blob")
            path.write_bytes((role + "\n").encode())
            path.chmod(0o444)
            self.paths[role] = path

        self.python = Path(os.path.realpath(sys.executable))
        self.ffmpeg = self.external / "ffmpeg"
        self.ffmpeg.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.ffmpeg.chmod(0o555)
        self.paths["python"] = self.python
        self.paths["ffmpeg"] = self.ffmpeg

        tasks = [{"task_id": task_id} for task_id in module.TASK_IDS]
        self.plan_value = {
            "schema_version": "case01-source-bone-exact5-r64-plan-v1",
            "experiment_id": "case01-288545b9c031491a-source-bone-exact5-r64-v1",
            "production_ready": True,
            "launch_allowed": True,
            "task_count": 5,
            "tasks": tasks,
        }
        self.plan = self.root / "plan.json"
        self.plan.write_bytes(module.canonical_json_bytes(self.plan_value) + b"\n")
        self.plan.chmod(0o444)
        self.paths["plan"] = self.plan

        for role in module.EXPECTED_STATIC_SHA256:
            module.EXPECTED_STATIC_SHA256[role] = hashlib.sha256(
                self.paths[role].read_bytes()
            ).hexdigest()

        self.model_root = self.runtime / "model"
        self.bernini_root = self.runtime / "bernini"
        self.veomni_root = self.runtime / "veomni"
        for path in (self.model_root, self.bernini_root, self.veomni_root):
            path.mkdir()
        self.output_report = self.final / "report.json"
        self.runner_attestation = self.final / "attestation.json"
        self.authority_root = self.runtime / "authority"
        self.rank_cache_root = self.runtime / "rank-cache"

    def input_value(self):
        value = {
            "schema_version": self.module.INPUT_SCHEMA,
            "entry_mode": "trusted_stdin",
            "output_report": str(self.output_report),
            "runner_attestation": str(self.runner_attestation),
            "model_root": str(self.model_root),
            "bernini_root": str(self.bernini_root),
            "veomni_root": str(self.veomni_root),
            "authority_root": str(self.authority_root),
            "rank_cache_root": str(self.rank_cache_root),
            "holder_job_id": "143808",
            "expected_node": "auh7-1b-gpu-292",
            "campaign_mode": self.module.CAMPAIGN,
        }
        value.update({role: str(self.paths[role]) for role in self.module.IDENTITY_ROLES})
        return value

    def input_file(self, value=None, name="input.json"):
        path = self.root / name
        path.write_bytes(
            self.module.canonical_json_bytes(self.input_value() if value is None else value)
            + b"\n"
        )
        return path

    def release(self):
        return self.module.build_release(self.input_value())[0]

    def run_bootstrap(self, release, *, swap_role=None):
        if swap_role is not None:
            path = self.paths[swap_role]
            replacement = path.with_name(path.name + ".replacement")
            replacement.write_bytes(path.read_bytes() + b"# named swap\n")
            replacement.chmod(path.stat().st_mode & 0o777)
            os.replace(replacement, path)
        bootstrap = self.module.ROOT_BOOTSTRAP.replace(
            'python_process=os.stat("/proc/self/exe")',
            "python_process=os.fstat(python_fd)",
            1,
        )
        # Darwin injects process-launch variables even when subprocess receives
        # an empty mapping.  Production Linux reaches the same empty-env state
        # through ``exec -c`` in the generated payload.
        bootstrap = "import os;os.environ.clear()\n" + bootstrap
        descriptor = os.open(self.python, os.O_RDONLY)
        try:
            command = [
                str(self.python), "-I", "-S", "-B", "-c", bootstrap,
                str(descriptor),
                self.module.canonical_json_bytes(release).decode("utf-8"),
                self.module.object_sha256(release),
                hashlib.sha256(self.module.ROOT_BOOTSTRAP.encode()).hexdigest(),
                "trusted_stdin", "143808", "7", "8", "8",
                "0,1,2,3,4,5,6,7", "1", "1",
                "auh7-1b-gpu-292", "auh7-1b-gpu-292",
            ]
            return subprocess.run(
                command,
                env={"LC_CTYPE": "C.UTF-8"},
                pass_fds=(descriptor,),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        finally:
            os.close(descriptor)


class Exact5LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.module = load_launcher()
        self.fixture = Fixture(self.module, Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact18_and_no_historical_campaign_surface(self):
        self.assertEqual(len(self.module.IDENTITY_ROLES), 18)
        self.assertEqual(len(set(self.module.IDENTITY_ROLES)), 18)
        source = SOURCE.read_text("utf-8")
        for forbidden in (
            "full16-production", "case00-pair-canary", "shared8-",
            "stops_after_pair", "build_release_r5f",
        ):
            self.assertNotIn(forbidden, source)

    def test_build_release_exact_contract(self):
        release, payload = self.module.build_release(self.fixture.input_value())
        self.assertEqual(release["campaign_mode"], self.module.CAMPAIGN)
        self.assertEqual(release["selected_task_ids"], list(self.module.TASK_IDS))
        self.assertEqual(set(release["identities"]), set(self.module.IDENTITY_ROLES))
        self.assertEqual(len(release["identities"]), 18)
        self.assertTrue(payload.endswith(b"\n"))
        self.assertIn(b"-I -S -B -c", payload)
        self.assertIn(b"exec -c", payload)

    def test_materialize_is_create_only_and_canonical(self):
        launch_input = self.fixture.input_file()
        payload = self.fixture.root / "payload.sh"
        receipt = self.fixture.root / "receipt.json"
        value = self.module.materialize(str(launch_input), str(payload), str(receipt))
        self.assertEqual(payload.stat().st_mode & 0o777, 0o444)
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o400)
        self.assertEqual(
            receipt.read_bytes(), self.module.canonical_json_bytes(value) + b"\n"
        )
        with self.assertRaises(self.module.Exact5RootLaunchError):
            self.module.materialize(str(launch_input), str(payload), str(receipt))
        subprocess.run(["/bin/bash", "-n", str(payload)], check=True)

    def test_duplicate_json_key_and_extra_field_fail(self):
        raw = self.module.canonical_json_bytes(self.fixture.input_value())
        duplicate = self.fixture.root / "duplicate.json"
        duplicate.write_bytes(b'{"schema_version":"x",' + raw[1:] + b"\n")
        with self.assertRaises(self.module.Exact5RootLaunchError):
            self.module._load_input(duplicate)
        value = self.fixture.input_value()
        value["unexpected"] = True
        with self.assertRaises(self.module.Exact5RootLaunchError):
            self.module._load_input(self.fixture.input_file(value, "extra.json"))

    def test_wrong_campaign_and_noncanonical_plan_fail(self):
        value = self.fixture.input_value()
        value["campaign_mode"] = "wrong"
        with self.assertRaises(self.module.Exact5RootLaunchError):
            self.module._load_input(self.fixture.input_file(value, "campaign.json"))
        self.fixture.plan.chmod(0o644)
        with self.assertRaises(self.module.Exact5RootLaunchError):
            self.module.build_release(self.fixture.input_value())

    def test_identity_path_alias_and_output_alias_fail(self):
        value = self.fixture.input_value()
        value["torchrun_handler_source"] = value["torchrun_source"]
        self.module.EXPECTED_STATIC_SHA256["torchrun_handler_source"] = (
            self.module.EXPECTED_STATIC_SHA256["torchrun_source"]
        )
        with self.assertRaises(self.module.Exact5RootLaunchError):
            self.module.build_release(value)
        value = self.fixture.input_value()
        value["runner_attestation"] = value["output_report"]
        with self.assertRaises(self.module.Exact5RootLaunchError):
            self.module.build_release(value)

    def test_bootstrap_reaches_captured_mock_wrapper(self):
        result = self.fixture.run_bootstrap(self.fixture.release())
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(
            self.fixture.output_report.read_bytes(),
            b"MOCK_EXACT5_CAPTURED_WRAPPER_REACHED\n",
        )

    def test_bootstrap_rejects_named_swap_before_wrapper(self):
        result = self.fixture.run_bootstrap(
            self.fixture.release(), swap_role="frozen_runner"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.fixture.output_report.exists())
        self.assertIn(b"named role identity differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
