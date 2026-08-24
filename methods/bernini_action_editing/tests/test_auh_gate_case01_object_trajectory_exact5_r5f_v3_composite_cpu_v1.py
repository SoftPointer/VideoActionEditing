from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HOLD_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_gate_case01_object_trajectory_exact5_r5f_v3_"
    "composite_cpu_once_v1.HOLD.py"
)
READY_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_gate_case01_object_trajectory_exact5_r5f_v3_"
    "composite_cpu_once_v1.READY.py"
)
PACKAGE_CONTROLLER_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_materialize_case01_object_trajectory_exact5_r64_"
    "overlay_package_once_v2.HOLD.py"
)
MATERIALIZER_PATH = ROOT / (
    "methods/bernini_action_editing/tools/"
    "materialize_case01_object_trajectory_exact5_r64_overlay_package_v2.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError("module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in statement.targets):
                return ast.literal_eval(statement.value)
    raise AssertionError("assignment not found: " + name)


class CompositeCPUGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hold = load(HOLD_PATH, "composite_cpu_gate_hold_test")
        cls.ready = load(READY_PATH, "composite_cpu_gate_ready_test")
        cls.hold_source = HOLD_PATH.read_text(encoding="utf-8")
        cls.ready_source = READY_PATH.read_text(encoding="utf-8")

    def test_source_and_embedded_programs_compile_all_optimization_levels(self):
        for path, source in (
            (HOLD_PATH, self.hold_source), (READY_PATH, self.ready_source),
        ):
            for optimize in (0, 1, 2):
                compile(source, str(path), "exec", optimize=optimize)
        for module in (self.hold, self.ready):
            for name in ("ROOT_BOOTSTRAP", "CHILD_BOOTSTRAP"):
                raw = getattr(module, name)
                for optimize in (0, 1, 2):
                    compile(raw, name, "exec", optimize=optimize)

    def test_ready_is_one_exact_state_line_diff(self):
        old = 'CONTROLLER_STATE = "HOLD_PENDING_FRESH_PACKAGE_PINS"'
        new = 'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_CPU_SRUN_NO_RETRY"'
        self.assertEqual(self.hold_source.count(old), 1)
        self.assertEqual(self.ready_source.count(new), 1)
        self.assertEqual(self.hold_source.replace(old, new, 1), self.ready_source)

    def test_hold_and_wrong_token_are_before_every_io(self):
        for module, argv in ((self.hold, []), (self.ready, ["--execute", "bad"])):
            with mock.patch.object(
                module.os, "lstat", side_effect=AssertionError("filesystem touched")
            ), mock.patch.object(
                module.os, "open", side_effect=AssertionError("filesystem touched")
            ), mock.patch.object(
                module.subprocess, "Popen", side_effect=AssertionError("process touched")
            ):
                self.assertEqual(module.main(argv), 88)

    def test_final_package_pins_and_authorization_are_closed(self):
        self.assertEqual(self.hold.blocked_dynamic_pins(), ())
        self.assertEqual(self.ready.blocked_dynamic_pins(), ())
        self.assertEqual(self.hold.dynamic_pin_values(), self.ready.dynamic_pin_values())
        token = self.ready.authorization_token()
        self.assertRegex(token, r"^[0-9a-f]{64}$")
        self.assertEqual(token, self.ready.authorization_token())
        self.assertEqual(self.ready.PUBLICATION_SIZE, 2530)
        self.assertEqual(self.ready.MATERIALIZATION_SIZE, 41351)
        self.assertEqual(self.ready.PACKAGE_CONTROLLER_SIZE, 8113)
        self.assertEqual(len(self.ready.PACKAGE_ROOT_IDENTITY), 11)

    def test_fresh_names_and_cache_are_disjoint_from_gpu_production(self):
        paths = {
            str(self.ready.ATTEMPT_PATH), str(self.ready.RECEIPT_PATH),
            str(self.ready.EVIDENCE_PATH), str(self.ready.STDOUT_PATH),
            str(self.ready.STDERR_PATH),
        }
        self.assertEqual(len(paths), 5)
        self.assertTrue(all("canary_v2" in path and "composite_cpu" in path
                            for path in paths))
        self.assertIn("node292-r2-rank-cache", str(self.ready.PRODUCTION_RANK_CACHE))
        self.assertIn("composite-cpu-job", self.ready.ROOT_BOOTSTRAP)
        self.assertNotIn("node292-r2-rank-cache\";os.mkdir", self.ready.ROOT_BOOTSTRAP)
        self.assertIn("production_rank_cache_untouched", self.ready.ROOT_BOOTSTRAP)

    def test_single_cpu_only_srun_exact_argv(self):
        command = self.ready.build_srun_argv()
        self.assertEqual(command[0], "/usr/bin/srun")
        self.assertEqual(command.count("--ntasks=1"), 1)
        self.assertEqual(command.count("--gres=none"), 1)
        self.assertEqual(command.count("--nodes=1"), 1)
        self.assertIn("--nodelist=auh7-1b-gpu-292", command)
        self.assertNotIn("--gpus=1", command)
        self.assertEqual(command[-3:], ["/bin/bash", "-p", "-s"])

    def test_real_embedded_payload_width_and_bash_syntax(self):
        rows = {}
        for index, role in enumerate(self.ready.IDENTITY_ROLES):
            rows[role] = {
                "path": str(self.ready.PACKAGE_ROOT / "release" / (role + ".py")),
                "sha256": hashlib.sha256(role.encode("utf-8")).hexdigest(),
                "size": 1000 + index,
                "identity": [48, index + 10, 2012, 2000, 33060, 1, 0,
                             1000 + index, 8, 100 + index, 200 + index],
                "mode": 0o444, "nlink": 1,
            }
        release = {
            "schema_version": (
                "case01-object-trajectory-exact5-r5f-v3-composite-cpu-release-v1"
            ),
            "package": {"identity_count": 26}, "identities": rows,
            "production_rank_cache": str(self.ready.PRODUCTION_RANK_CACHE),
            "world_size": 4, "gpu_count": 0,
            "root_bootstrap_sha256": hashlib.sha256(
                self.ready.ROOT_BOOTSTRAP.encode("utf-8")
            ).hexdigest(),
            "child_bootstrap_sha256": hashlib.sha256(
                self.ready.CHILD_BOOTSTRAP.encode("utf-8")
            ).hexdigest(),
        }
        payload = self.ready.build_payload(release)
        transport = self.ready.transport_preflight(
            self.ready.build_srun_argv(), payload,
        )
        self.assertGreater(len(payload), len(self.ready.ROOT_BOOTSTRAP.encode()))
        self.assertEqual(transport["stdin_bytes"], len(payload))
        self.assertLess(transport["stdin_bytes"], transport["stdin_bound"])
        self.assertLess(transport["argv_bytes"], transport["argv_bound"])
        checked = subprocess.run(
            ["/bin/bash", "-n"], input=payload, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr.decode())

    def test_controller_order_is_receipt_first_attempt_before_one_srun(self):
        source = self.hold_source
        publication = source.index("publication_held = open_authority(")
        materialization = source.index("materialization_held = open_authority(")
        package_controller = source.index("package_controller_held = open_authority(")
        root = source.index("root = open_directory(PACKAGE_ROOT")
        attempt = source.index("attempt_raw = create_json(ATTEMPT_PATH")
        launch = source.index("returncode, stdout, stderr, process_group = run_srun(")
        self.assertLess(publication, materialization)
        self.assertLess(materialization, package_controller)
        self.assertLess(package_controller, root)
        self.assertLess(root, attempt)
        self.assertLess(attempt, launch)
        tree = ast.parse(source)
        popen_calls = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
        ]
        self.assertEqual(len(popen_calls), 1)

    def test_embedded_four_real_process_private_parent_and_object_contract(self):
        root = self.ready.ROOT_BOOTSTRAP
        child = self.ready.CHILD_BOOTSTRAP
        self.assertIn("for rank in range(4):", root)
        self.assertIn("subprocess.Popen", root)
        self.assertIn("close_fds=True", root)
        self.assertIn("pass_fds=pass_fds", root)
        self.assertIn("private_fd in inherited", root)
        self.assertIn("write_all(fd,payload)", root)
        self.assertIn("child_process.terminate()", root)
        self.assertIn("child_process.kill()", root)
        self.assertIn("shutil.rmtree(cache_root)", root)
        self.assertIn("_FROZEN_VALIDATE_INHERITED", child)
        self.assertIn("capture private-parent FD replay differs", child)
        self.assertIn("validate_inherited_fd_binding_r5f", child)
        self.assertIn("read_fd_with_pread_r5f", child)
        self.assertIn("with module.base.translated_publication", child)
        self.assertIn("inner._patched_legacy(legacy,assets)", child)
        self.assertIn('legacy is not sys.modules.get("infer_lora")', child)
        self.assertIn('"torch" in sys.modules', child)

    def test_receipt_exact_schema_and_hostile_tamper(self):
        package = {"root": "sealed"}
        rows = []
        for rank in range(4):
            row = {
                "rank": rank, "pid": 5000 + rank,
                "private_parent_fd_number": 40,
                "private_parent_replacement_inode": 100 + rank,
                "pread_bytes_sha256": self.ready.SHARED_OFD_PAYLOAD_SHA256,
                "pread_offset_before": 13, "pread_offset_after": 13,
            }
            row["rank_digest"] = self.ready.object_digest(row)
            rows.append(row)
        value = {
            "schema_version": self.ready.RECEIPT_SCHEMA,
            "status": "PASS_COMPOSITE_CPU_EXACT26_HOLD",
            "holder_job_id": self.ready.HOLDER_JOB_ID, "node": self.ready.NODE,
            "slurm_step_id": "511", "package": package,
            "world_size": 4, "rank_count": 4, "rank_rows": rows,
            "isolated_runtime": {
                "python_flags": ["-I", "-S", "-B"], "isolated": 1,
                "no_site": 1, "dont_write_bytecode": True,
                "entry_via_proc_self_fd": True,
            },
            "private_parent_fd": {
                "synthetic_model_capture": True, "captured_parent_omitted": True,
                "captured_parent_closed_or_reused": True,
                "frozen_validator_rejected": True,
                "r5f_validator_accepted": True,
                "r5f_pread_path_exercised": True,
            },
            "shared_ofd_pread": {
                "rank_count": 4, "all_reads_exact": True,
                "offsets_unchanged": True,
            },
            "module_binding": {
                "module_name": "infer_lora", "base_infer_lora_same_object": True,
                "object_cli_applied_to_base_module": True,
                "translated_publication_applied_to_base_module": True,
                "legacy_module_instance_count": 1,
                "duplicate_legacy_module_loaded": False,
            },
            "side_effects": {
                "gpu_requested": False, "torch_imported": False,
                "renderer_or_vae_loaded": False, "publication_performed": False,
            },
            "cache_lifecycle": {
                "admission_cache_root": (
                    "/tmp/bernini-case01-object-trajectory-r5f-v3-composite-"
                    "cpu-job143808-step511-cache"
                ),
                "admission_cache_fresh": True,
                "admission_cache_cleanup_performed": True,
                "admission_cache_absent_terminal": True,
                "production_rank_cache": str(self.ready.PRODUCTION_RANK_CACHE),
                "production_rank_cache_untouched": True,
                "production_rank_cache_absent_before_and_after": True,
            },
            "process_cleanup": {
                "all_rank_returncodes_zero": True, "rank_processes_zero": True,
                "torchrun_processes_zero": True, "child_processes_terminal": True,
            },
            "launch_allowed": False,
        }
        value["receipt_digest"] = self.ready.object_digest(value)
        raw = self.ready.canonical(value) + b"\n"
        self.assertEqual(self.ready.validate_receipt(raw, package), value)
        for mutate in ("extra", "pread", "cache"):
            hostile = json.loads(json.dumps(value))
            if mutate == "extra":
                hostile["unexpected"] = False
            elif mutate == "pread":
                hostile["rank_rows"][0]["pread_bytes_sha256"] = "0" * 64
                unsigned = dict(hostile["rank_rows"][0])
                unsigned.pop("rank_digest")
                hostile["rank_rows"][0]["rank_digest"] = self.ready.object_digest(unsigned)
            else:
                hostile["cache_lifecycle"]["production_rank_cache_untouched"] = False
            unsigned = dict(hostile); unsigned.pop("receipt_digest", None)
            hostile["receipt_digest"] = self.ready.object_digest(unsigned)
            with self.assertRaises(self.ready.CompositeCPUError):
                self.ready.validate_receipt(self.ready.canonical(hostile) + b"\n", package)

    def test_package_producer_exact_field_contracts_match(self):
        pairs = (
            ("PUBLICATION_FIELDS", "PACKAGE_RECEIPT_FIELDS"),
            ("MATERIALIZATION_FIELDS", "REPORT_FIELDS"),
            ("LAUNCH_FIELDS", "LAUNCH_FIELDS"),
            ("LAUNCH_RELEASE_FIELDS", "LAUNCH_RELEASE_FIELDS"),
        )
        for gate_name, producer_name in pairs:
            self.assertEqual(
                set(getattr(self.ready, gate_name)),
                set(literal_assignment(PACKAGE_CONTROLLER_PATH, producer_name)),
            )
        materializer = MATERIALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn('"runtime/model-authority-v3"', materializer)
        self.assertIn('"outputs/media_v3"', materializer)
        self.assertNotIn('"runtime/model-authority-v2"', materializer)

    def test_create_immutable_replays_zero_and_nonzero_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for name, raw in (("zero", b""), ("payload", b"authority\n")):
                path = parent / name
                self.ready.create_immutable(path, raw, 0o400)
                self.assertEqual(path.read_bytes(), raw)
                self.assertEqual(path.stat().st_mode & 0o777, 0o400)

    def test_process_group_cleanup_does_not_return_when_leader_already_exited(self):
        process = mock.Mock()
        process.pid = 424242
        process.poll.return_value = 0
        with mock.patch.object(
            self.ready, "_process_group_absent", side_effect=[False, True]
        ), mock.patch.object(self.ready.os, "killpg") as killpg:
            self.ready._terminate_process_group(process)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(424242, signal.SIGTERM), mock.call(424242, signal.SIGKILL)],
        )
        process.wait.assert_not_called()

    def test_evidence_schema_is_honest_about_srun_and_real_ranks(self):
        self.assertIn("srun_ntasks", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertIn("real_rank_process_count", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertNotIn("ntasks", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertIn('"srun_ntasks": 1', self.ready_source)
        self.assertIn('"real_rank_process_count": 4', self.ready_source)
        self.assertNotIn('"process_group_zero":True', self.ready.ROOT_BOOTSTRAP)


if __name__ == "__main__":
    unittest.main()
