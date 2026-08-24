from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
HOLD_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_gate_case01_object_trajectory_exact5_r5f_v4_"
    "composite_cpu_once_v4.HOLD.py"
)
PACKAGE_CONTROLLER_PATH = ROOT / (
    "methods/bernini_action_editing/scripts/"
    "auh_materialize_case01_object_trajectory_exact5_r64_"
    "overlay_package_once_v3.READY.py"
)
MATERIALIZER_PATH = ROOT / (
    "methods/bernini_action_editing/tools/"
    "materialize_case01_object_trajectory_exact5_r64_overlay_package_v3.py"
)
COMPOSITE_SOURCE = ROOT / (
    "methods/bernini_action_editing/"
    "infer_case01_object_trajectory_oracle_auh_r5f_v4.py"
)
BASE_ADAPTER_SOURCE = ROOT / (
    "methods/bernini_action_editing/"
    "full644_exploratory_matched_infer_adapter_v3.py"
)
SEALED_METHOD_FIXTURE = Path(
    "/tmp/case01_object_trajectory_v1_sealed_methods_fixture"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError("module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source(source: str, path: Path, name: str):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = None
    module.__loader__ = None
    module.__cached__ = None
    module.__builtins__ = __builtins__
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in statement.targets):
                return ast.literal_eval(statement.value)
    raise AssertionError("assignment not found: " + name)


class CompositeCPUGateV4V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hold = load(HOLD_PATH, "composite_cpu_gate_hold_test")
        cls.hold_source = HOLD_PATH.read_text(encoding="utf-8")
        old = 'CONTROLLER_STATE = "HOLD_PENDING_FRESH_CANARY_V3_CPU_V4_AUDIT"'
        new = 'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_CPU_SRUN_NO_RETRY"'
        if cls.hold_source.count(old) != 1:
            raise AssertionError("HOLD state transition source differs")
        cls.ready_source = cls.hold_source.replace(old, new, 1)
        cls.ready = load_source(
            cls.ready_source, HOLD_PATH, "composite_cpu_gate_ready_memory_test"
        )

    def test_source_and_embedded_programs_compile_all_optimization_levels(self):
        for path, source in (
            (HOLD_PATH, self.hold_source), (HOLD_PATH, self.ready_source),
        ):
            for optimize in (0, 1, 2):
                compile(source, str(path), "exec", optimize=optimize)
        for module in (self.hold, self.ready):
            for name in ("ROOT_BOOTSTRAP", "CHILD_BOOTSTRAP"):
                raw = getattr(module, name)
                for optimize in (0, 1, 2):
                    compile(raw, name, "exec", optimize=optimize)

    def test_ready_projection_is_one_exact_in_memory_state_line_diff(self):
        old = 'CONTROLLER_STATE = "HOLD_PENDING_FRESH_CANARY_V3_CPU_V4_AUDIT"'
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
        self.assertEqual(self.ready.PUBLICATION_SIZE, 2528)
        self.assertEqual(self.ready.MATERIALIZATION_SIZE, 41726)
        self.assertEqual(self.ready.PACKAGE_CONTROLLER_SIZE, 8099)
        self.assertEqual(len(self.ready.PACKAGE_ROOT_IDENTITY), 11)
        self.assertEqual(len(self.ready.CORE4_RELEASE_AUTHORITIES), 5)
        self.assertEqual(
            self.ready.BASE_ADAPTER_SHA256,
            "7b72e8dc88d95daa34d93604dddacf6dcf4f75a2f92f356f743183cf06fa7120",
        )

    def test_fresh_names_and_cache_are_disjoint_from_gpu_production(self):
        paths = {
            str(self.ready.ATTEMPT_PATH), str(self.ready.RECEIPT_PATH),
            str(self.ready.EVIDENCE_PATH), str(self.ready.STDOUT_PATH),
            str(self.ready.STDERR_PATH),
        }
        self.assertEqual(len(paths), 5)
        self.assertTrue(all("canary_v3" in path and "composite_cpu" in path
                            for path in paths))
        self.assertTrue(all("_v4." in path for path in paths))
        self.assertIn("node292-r3-rank-cache", str(self.ready.PRODUCTION_RANK_CACHE))
        self.assertIn("composite-cpu-v4-job", self.ready.ROOT_BOOTSTRAP)
        self.assertNotIn(
            "node292-" + "r2-rank-cache\";os.mkdir", self.ready.ROOT_BOOTSTRAP
        )
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
                "case01-object-trajectory-exact5-r5f-v4-composite-cpu-release-v4"
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
        launch = source.index(") = run_srun(command, payload)")
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
        self.assertIn("remove_tree(cache_root)", root)
        self.assertIn("shutil.rmtree(path)", root)
        self.assertIn("_FROZEN_VALIDATE_INHERITED", child)
        self.assertIn("capture private-parent FD replay differs", child)
        self.assertIn("validate_inherited_fd_binding_r5f", child)
        self.assertIn("read_fd_with_pread_r5f", child)
        self.assertIn("with module.base.translated_publication", child)
        self.assertIn("inner._patched_legacy(legacy,assets)", child)
        self.assertIn('legacy is not sys.modules.get("infer_lora")', child)
        self.assertIn('p0_base.importlib.import_module("bernini.pipeline")', child)
        self.assertIn("p0_base._CapturedVendorFinder", child)
        self.assertIn("p0_base._CapturedVendorLoader", child)
        self.assertIn('getattr(spec,"loader",None) is not loader', child)
        self.assertIn('getattr(imported,"__cached__","non-none") is not None', child)
        self.assertIn('"activation_import_before_callback_return":True', child)
        self.assertIn("rows[\"base_adapter\"][\"path\"]", root)
        self.assertIn('"torch" in sys.modules', child)

    def test_child_module_attribute_surface_matches_real_composite_v4(self):
        tree = ast.parse(self.ready.CHILD_BOOTSTRAP, "CHILD_BOOTSTRAP")
        accessed = {
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "module"
        }
        self.assertEqual(accessed, {
            "BASE_ADAPTER_SHA256", "OBJECT_WRAPPER_INNER_SHA256", "Path",
            "_FROZEN_VALIDATE_INHERITED", "__builtins__", "__cached__",
            "__dict__", "__file__", "__loader__", "__name__", "__package__",
            "_bind_composite_producer_hashes", "base", "held_object_sources",
            "model_authority", "read_fd_with_pread_r5f",
            "validate_inherited_fd_binding_r5f",
        })
        composite_tree = ast.parse(
            COMPOSITE_SOURCE.read_bytes(), filename=str(COMPOSITE_SOURCE),
        )
        exported = {
            target.id
            for node in composite_tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
            if isinstance(target, ast.Name)
        }
        self.assertIn("BASE_ADAPTER_SHA256", exported)
        self.assertNotIn("_BASE_ADAPTER_SHA256", exported)
        self.assertNotIn("module._BASE_ADAPTER_SHA256", self.ready.CHILD_BOOTSTRAP)
        self.assertIn("module.BASE_ADAPTER_SHA256", self.ready.CHILD_BOOTSTRAP)

    def test_real_root_bootstrap_runs_four_real_children_on_sealed_fixture(self):
        self.assertTrue(SEALED_METHOD_FIXTURE.is_dir())

        def identity_row(path: Path):
            raw = path.read_bytes()
            info = path.stat()
            return {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "identity": list(self.ready.identity(info)),
                "mode": stat.S_IMODE(info.st_mode),
                "nlink": info.st_nlink,
            }

        with tempfile.TemporaryDirectory(prefix="cpu-v4-root-fixture-") as temporary:
            fixture_root = Path(temporary).resolve()
            methods_root = fixture_root / "methods"
            shutil.copytree(SEALED_METHOD_FIXTURE, methods_root)
            composite_path = methods_root / COMPOSITE_SOURCE.name
            base_adapter_path = methods_root / BASE_ADAPTER_SOURCE.name
            shutil.copy2(COMPOSITE_SOURCE, composite_path)
            shutil.copy2(BASE_ADAPTER_SOURCE, base_adapter_path)
            composite_path.chmod(0o444)
            base_adapter_path.chmod(0o444)

            authority_path = (
                methods_root
                / "action_preservation_decoded_eval_model_authority_v2.py"
            )
            relative_files = literal_assignment(
                authority_path, "MODEL_RELATIVE_FILES",
            )
            manifest_path = fixture_root / "base-model.sha256"
            manifest_path.write_text(
                "".join("0" * 64 + "  ./" + value + "\n"
                        for value in relative_files),
                encoding="utf-8",
            )
            manifest_path.chmod(0o444)
            dummy_path = fixture_root / "held-authority.bin"
            dummy_path.write_bytes(b"sealed CPU-v4 held authority\n")
            dummy_path.chmod(0o444)
            python_path = Path(sys.executable).resolve(strict=True)

            role_paths = {
                role: dummy_path for role in self.ready.IDENTITY_ROLES
            }
            role_paths.update({
                "adapter": composite_path,
                "base_adapter": base_adapter_path,
                "model_authority": authority_path,
                "base_model_manifest": manifest_path,
                "python": python_path,
                "ffmpeg": python_path,
                "ffprobe": python_path,
            })
            identities = {
                role: identity_row(role_paths[role])
                for role in self.ready.IDENTITY_ROLES
            }

            # Darwin has no /proc and /tmp resolves through /private/tmp.
            # These platform-only transport/path substitutions leave the
            # embedded CHILD_BOOTSTRAP byte-exact.
            portable_root = self.ready.ROOT_BOOTSTRAP
            proc_identity = 'ident(os.stat("/proc/self/exe"))'
            proc_exec = '"/proc/self/fd/%d"%python_fd'
            cache_template = (
                '"/tmp/bernini-case01-object-trajectory-r5f-v4-composite-'
                'cpu-v4-job%s-step%s-cache"'
            )
            capture_call = (
                "expected_manifest_sha256=hashlib.sha256(manifest_raw)."
                "hexdigest())"
            )
            self.assertEqual(portable_root.count(proc_identity), 1)
            self.assertEqual(portable_root.count(proc_exec), 1)
            self.assertEqual(portable_root.count(cache_template), 1)
            self.assertEqual(portable_root.count(capture_call), 1)
            portable_cache_template = str(
                fixture_root / "cpu-v4-job%s-step%s-cache"
            )
            portable_root = portable_root.replace(
                proc_identity, "ident(os.stat(sys.executable))", 1,
            ).replace(proc_exec, "sys.executable", 1).replace(
                cache_template, repr(portable_cache_template), 1,
            ).replace(
                capture_call,
                capture_call[:-1] + ',proc_fd_prefix="/dev/fd")',
                1,
            )
            self.assertEqual(
                base64.b64decode(
                    self.ready.ROOT_BOOTSTRAP.split(
                        'base64.b64decode("', 1,
                    )[1].split('".encode("ascii")', 1)[0].encode("ascii"),
                    validate=True,
                ).decode("utf-8", "strict"),
                self.ready.CHILD_BOOTSTRAP,
            )

            root_sha = hashlib.sha256(
                self.ready.ROOT_BOOTSTRAP.encode("utf-8")
            ).hexdigest()
            release = {
                "schema_version": (
                    "case01-object-trajectory-exact5-r5f-v4-"
                    "composite-cpu-release-v4"
                ),
                "package": {"fixture": "sealed-composite-v4"},
                "identities": identities,
                "production_rank_cache": str(self.ready.PRODUCTION_RANK_CACHE),
                "root_bootstrap_sha256": root_sha,
                "child_bootstrap_sha256": hashlib.sha256(
                    self.ready.CHILD_BOOTSTRAP.encode("utf-8")
                ).hexdigest(),
            }
            release_b64 = base64.b64encode(
                self.ready.canonical(release)
            ).decode("ascii")
            step_id = str(900_000_000 + os.getpid())
            cache_root = Path(portable_cache_template % (
                self.ready.HOLDER_JOB_ID, step_id,
            ))
            self.assertFalse(cache_root.exists())
            self.assertFalse(self.ready.PRODUCTION_RANK_CACHE.exists())

            python_fd = os.open(str(python_path), os.O_RDONLY)
            try:
                process = subprocess.Popen(
                    [
                        str(python_path), "-I", "-S", "-B", "-c",
                        portable_root, str(python_fd), release_b64,
                        self.ready.object_digest(release), root_sha,
                        self.ready.HOLDER_JOB_ID, step_id, self.ready.NODE,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                    pass_fds=(python_fd,),
                    env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                )
                stdout, stderr = process.communicate(timeout=240)
            finally:
                os.close(python_fd)
            self.assertEqual(process.returncode, 0, stderr.decode("utf-8", "replace"))
            self.assertEqual(stderr, b"")
            self.assertEqual(stdout.count(b"\n"), 1)
            receipt = json.loads(stdout.decode("utf-8", "strict"))
            self.assertEqual(
                receipt["status"],
                "PASS_COMPOSITE_CPU_EXACT26_ACTIVATION_IMPORT_V4_HOLD",
            )
            self.assertEqual([row["rank"] for row in receipt["rank_rows"]],
                             [0, 1, 2, 3])
            self.assertEqual(len({row["pid"] for row in receipt["rank_rows"]}), 4)
            self.assertEqual(
                receipt["activation_import"]["base_adapter_sha256"],
                self.ready.BASE_ADAPTER_SHA256,
            )
            self.assertEqual(
                receipt["activation_import"]["base_adapter_path"],
                str(base_adapter_path),
            )
            self.assertTrue(all(
                row["activation_import_before_callback_return"]
                and row["captured_vendor_finder_preinstalled"]
                for row in receipt["rank_rows"]
            ))
            self.assertFalse(cache_root.exists())

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
                "activation_callback_import_module": "bernini.pipeline",
                "activation_import_before_callback_return": True,
                "captured_vendor_finder_preinstalled": True,
                "captured_vendor_finder_count": 1,
                "captured_vendor_loader_type": "_CapturedVendorLoader",
                "captured_vendor_spec_loader_type": "_CapturedVendorLoader",
                "captured_vendor_loader_is_spec_loader": True,
                "captured_vendor_cached_is_none": True,
            }
            row["rank_digest"] = self.ready.object_digest(row)
            rows.append(row)
        value = {
            "schema_version": self.ready.RECEIPT_SCHEMA,
            "status": "PASS_COMPOSITE_CPU_EXACT26_ACTIVATION_IMPORT_V4_HOLD",
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
            "activation_import": {
                "module": "bernini.pipeline",
                "callback_phase": "inside_original_activate_before_return",
                "finder_installed_before_callback": True,
                "finder_count_per_rank": [1, 1, 1, 1],
                "loader_type": "_CapturedVendorLoader",
                "spec_loader_type": "_CapturedVendorLoader",
                "loader_is_spec_loader": True,
                "cached_is_none": True,
                "base_adapter_role": "base_adapter",
                "base_adapter_path": str(
                    self.ready.PACKAGE_ROOT
                    / "release/methods/bernini_action_editing/"
                    "full644_exploratory_matched_infer_adapter_v3.py"
                ),
                "base_adapter_sha256": self.ready.BASE_ADAPTER_SHA256,
                "rank_count": 4,
            },
            "side_effects": {
                "gpu_requested": False, "torch_imported": False,
                "renderer_or_vae_loaded": False, "publication_performed": False,
            },
            "cache_lifecycle": {
                "admission_cache_root": (
                    "/tmp/bernini-case01-object-trajectory-r5f-v4-composite-"
                    "cpu-v4-job143808-step511-cache"
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
        for mutate in (
            "extra", "pread", "cache", "finder", "loader", "callback",
            "top_path", "top_cache",
        ):
            hostile = json.loads(json.dumps(value))
            if mutate == "extra":
                hostile["unexpected"] = False
            elif mutate == "pread":
                hostile["rank_rows"][0]["pread_bytes_sha256"] = "0" * 64
                unsigned = dict(hostile["rank_rows"][0])
                unsigned.pop("rank_digest")
                hostile["rank_rows"][0]["rank_digest"] = self.ready.object_digest(unsigned)
            elif mutate == "cache":
                hostile["cache_lifecycle"]["production_rank_cache_untouched"] = False
            elif mutate == "finder":
                hostile["rank_rows"][0]["captured_vendor_finder_count"] = 0
                unsigned_row = dict(hostile["rank_rows"][0])
                unsigned_row.pop("rank_digest")
                hostile["rank_rows"][0]["rank_digest"] = self.ready.object_digest(
                    unsigned_row
                )
            elif mutate == "loader":
                hostile["rank_rows"][2]["captured_vendor_loader_type"] = (
                    "SourceFileLoader"
                )
                unsigned_row = dict(hostile["rank_rows"][2])
                unsigned_row.pop("rank_digest")
                hostile["rank_rows"][2]["rank_digest"] = self.ready.object_digest(
                    unsigned_row
                )
            elif mutate == "callback":
                hostile["rank_rows"][3][
                    "activation_import_before_callback_return"
                ] = False
                unsigned_row = dict(hostile["rank_rows"][3])
                unsigned_row.pop("rank_digest")
                hostile["rank_rows"][3]["rank_digest"] = self.ready.object_digest(
                    unsigned_row
                )
            elif mutate == "top_path":
                hostile["activation_import"]["base_adapter_path"] += ".wrong"
            else:
                hostile["activation_import"]["cached_is_none"] = False
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
        self.assertIn('"receipt_gated_exact6_overlay"', materializer)
        for relative, (digest, size) in self.ready.CORE4_RELEASE_AUTHORITIES.items():
            path = ROOT / relative
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            self.assertEqual(len(raw), size)

    def test_fresh_schema_and_p0_names_have_no_failed_generation_tokens(self):
        self.assertNotIn("canary_" + "v2", self.hold_source)
        self.assertNotIn("r5f-" + "v3", self.hold_source)
        self.assertIn("r5f-v4", self.hold.RECEIPT_SCHEMA)
        self.assertTrue(str(self.hold.RECEIPT_PATH).endswith(
            ".composite_cpu_admission_receipt_v4.json"
        ))
        self.assertTrue(str(self.hold.EVIDENCE_PATH).endswith(
            ".composite_cpu_admission_controller_evidence_v4.json"
        ))
        self.assertEqual(self.hold.PLAN_SIZE, 32050)
        self.assertEqual(self.hold.LAUNCH_RECEIPT_SIZE, 10292)
        self.assertEqual(self.hold.LAUNCH_INPUT_SIZE, 9788)
        self.assertEqual(self.hold.LAUNCH_PAYLOAD_SIZE, 12783)
        for value in (
            self.hold.PLAN_SHA256, self.hold.PLAN_DIGEST,
            self.hold.LAUNCH_RECEIPT_SHA256,
            self.hold.LAUNCH_RECEIPT_DIGEST,
            self.hold.LAUNCH_INPUT_SHA256, self.hold.LAUNCH_PAYLOAD_SHA256,
        ):
            self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_create_immutable_replays_zero_and_nonzero_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for name, raw in (("zero", b""), ("payload", b"authority\n")):
                path = parent / name
                self.ready.create_immutable(path, raw, 0o400)
                self.assertEqual(path.read_bytes(), raw)
                self.assertEqual(path.stat().st_mode & 0o777, 0o400)

    def test_transient_terminal_group_naturally_disappears_without_signal(self):
        class ClosedPipe:
            closed = True

        class Process:
            pid = 4141
            returncode = 0
            stdin = ClosedPipe()
            stdout = ClosedPipe()
            stderr = ClosedPipe()

            def communicate(self, *, input, timeout):
                self.input = input
                self.timeout = timeout
                return b"receipt\n", b""

            def poll(self):
                return self.returncode

        process = Process()
        with mock.patch.object(
            self.ready.subprocess, "Popen", return_value=process,
        ), mock.patch.object(
            self.ready, "_process_group_present", side_effect=[True, False],
        ), mock.patch.object(
            self.ready.time, "sleep",
        ), mock.patch.object(
            self.ready, "_signal_process_group",
        ) as signal_group, mock.patch.object(
            self.ready, "_seal_process_group",
        ) as seal:
            result = self.ready.run_srun(
                self.ready.build_srun_argv(), b"payload",
            )
        self.assertEqual(result[:4], (0, b"receipt\n", b"", 4141))
        self.assertEqual(result[4], {
            "normal_exit_passive_grace_performed": True,
            "normal_exit_signal_sent": False,
            "terminal_pipes_closed": True,
            "process_group_zero": True,
        })
        signal_group.assert_not_called()
        seal.assert_not_called()

    def test_persistent_terminal_group_is_sealed_then_fails_closed(self):
        class ClosedPipe:
            closed = True

        process = mock.Mock()
        process.pid = 4242
        process.returncode = 0
        process.stdin = ClosedPipe()
        process.stdout = ClosedPipe()
        process.stderr = ClosedPipe()
        process.communicate.return_value = (b"receipt\n", b"")
        with mock.patch.object(
            self.ready.subprocess, "Popen", return_value=process,
        ), mock.patch.object(
            self.ready, "_process_group_absent", return_value=False,
        ), mock.patch.object(
            self.ready, "_seal_process_group",
        ) as seal, self.assertRaisesRegex(
            self.ready.CompositeCPUError, "required cleanup after passive grace",
        ):
            self.ready.run_srun(self.ready.build_srun_argv(), b"payload")
        seal.assert_called_once_with(process, 4242)

    def test_run_srun_timeout_seals_saved_process_group_without_retry(self):
        process = mock.Mock()
        process.pid = 5151
        process.returncode = -signal.SIGTERM
        timeout = subprocess.TimeoutExpired(["srun"], 1)
        process.communicate.side_effect = timeout
        with mock.patch.object(
            self.ready.subprocess, "Popen", return_value=process,
        ) as popen, mock.patch.object(
            self.ready, "_process_group_present", return_value=False,
        ), mock.patch.object(self.ready, "_seal_process_group") as seal, \
             self.assertRaisesRegex(
                 self.ready.CompositeCPUError, "srun timed out",
             ) as caught:
            self.ready.run_srun(self.ready.build_srun_argv(), b"payload")
        popen.assert_called_once()
        seal.assert_called_once_with(process, 5151)
        self.assertIs(caught.exception.__cause__, timeout)

    def test_run_srun_communicate_error_is_primary_when_cleanup_also_fails(self):
        process = mock.Mock()
        process.pid = 6161
        process.returncode = None
        primary = RuntimeError("hostile communicate error")
        cleanup = OSError("hostile cleanup error")
        process.communicate.side_effect = primary
        with mock.patch.object(
            self.ready.subprocess, "Popen", return_value=process,
        ), mock.patch.object(
            self.ready, "_process_group_present", return_value=True,
        ), mock.patch.object(
            self.ready, "_seal_process_group", side_effect=cleanup,
        ) as seal, self.assertRaisesRegex(
            self.ready.CompositeCPUError, "process/pipe zero gate differs",
        ) as caught:
            self.ready.run_srun(self.ready.build_srun_argv(), b"payload")
        seal.assert_called_once_with(process, 6161)
        self.assertIs(caught.exception.__cause__, primary)

    def test_early_leader_and_term_ignoring_descendant_reach_esrch(self):
        source = (
            "import os,signal,sys,time\n"
            "child=os.fork()\n"
            "if child == 0:\n"
            " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            " sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            " while True: time.sleep(1)\n"
            "time.sleep(0.05)\n"
            "os._exit(0)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", source], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True,
        )
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.35)
            process.poll()
            with mock.patch.object(
                self.ready, "PROCESS_TERM_GRACE_SECONDS", 0.10,
            ), mock.patch.object(
                self.ready, "PROCESS_KILL_GRACE_SECONDS", 1.50,
            ):
                self.ready._seal_process_group(process, process.pid)
            self.assertTrue(self.ready._process_group_absent(process.pid, 0.5))
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=1)

    def test_pipe_close_error_does_not_bypass_process_group_zero_gate(self):
        source = (
            "import signal,sys,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", source], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True,
        )
        original_stdout = process.stdout

        class BrokenClose:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.closed = False

            def close(self):
                if not self.closed:
                    self.closed = True
                    self.wrapped.close()
                raise OSError("synthetic pipe close failure")

        try:
            self.assertIsNotNone(original_stdout)
            self.assertEqual(original_stdout.readline(), b"READY\n")
            process.stdout = BrokenClose(original_stdout)
            with mock.patch.object(
                self.ready, "PROCESS_TERM_GRACE_SECONDS", 0.10,
            ), mock.patch.object(
                self.ready, "PROCESS_KILL_GRACE_SECONDS", 1.50,
            ), self.assertRaisesRegex(
                self.ready.CompositeCPUError, "terminal pipe seal",
            ):
                self.ready._seal_process_group(process, process.pid)
            self.assertTrue(self.ready._process_group_absent(process.pid, 0.5))
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=1)

    def test_evidence_schema_is_honest_about_srun_and_real_ranks(self):
        self.assertEqual(self.ready.EVIDENCE_SCHEMA, self.ready.SCHEMA + "-evidence")
        self.assertIn('"schema_version": EVIDENCE_SCHEMA', self.ready_source)
        self.assertIn("srun_ntasks", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertIn("real_rank_process_count", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertIn(
            "normal_exit_passive_grace_performed", self.ready.CPU_EVIDENCE_FIELDS
        )
        self.assertIn("normal_exit_signal_sent", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertIn("terminal_pipes_closed", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertNotIn("ntasks", self.ready.CPU_EVIDENCE_FIELDS)
        self.assertIn('"srun_ntasks": 1', self.ready_source)
        self.assertIn('"real_rank_process_count": 4', self.ready_source)
        self.assertIn('"normal_exit_signal_sent": False', self.ready_source)
        self.assertNotIn('"process_group_zero":True', self.ready.ROOT_BOOTSTRAP)


if __name__ == "__main__":
    unittest.main()
