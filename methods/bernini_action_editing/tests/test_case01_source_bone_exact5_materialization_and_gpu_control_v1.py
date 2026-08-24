#!/usr/bin/env python3
"""Hostile closure tests for the case01 exact5 materialization/control chain."""

from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
SNAPSHOT_SOURCE = METHOD / "tools/build_case01_source_bone_exact5_source_snapshot_v1.py"
MATERIALIZER_SOURCE = METHOD / "tools/materialize_case01_source_bone_exact5_r64_package_v1.py"
STATIC_SOURCE = METHOD / "case01_source_bone_exact5_static_probe_v1.py"
FAKE_SOURCE = METHOD / "case01_source_bone_exact5_root_fake_runner_v1.py"
EVAL_SOURCE = METHOD / "case01_source_bone_exact5_eval_v1.py"
LAUNCHER_SOURCE = METHOD / "case01_source_bone_exact5_spooled_launcher_auh_v1.py"
GPU_CONTROLLER = (
    METHOD / "scripts/"
    "auh_launch_case01_source_bone_exact5_r64_gpu_job143808_node292_once_v1.HOLD.sh"
)
STATIC_GATE = (
    METHOD / "scripts/"
    "auh_gate_case01_source_bone_exact5_static_job143808_node292_once_v1.sh"
)
FAKE_GATE = (
    METHOD / "scripts/"
    "auh_gate_case01_source_bone_exact5_root_fake_job143808_node292_once_v1.sh"
)
ASSET_ROOT = REPO / "artifacts/object_grounded_case01_0821_bone_interventions_r4"
AUDIT = (
    REPO / "md/action_editing/20260821_man/evidence/"
    "case01_exact5_intervention_asset_independent_audit_v1.json"
)


def load(path: Path, stem: str) -> types.ModuleType:
    name = f"_test_{stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


snapshot = load(SNAPSHOT_SOURCE, "snapshot")
materializer = load(MATERIALIZER_SOURCE, "materializer")
static_probe = load(STATIC_SOURCE, "static")
exact_eval = load(EVAL_SOURCE, "eval")
launcher = load(LAUNCHER_SOURCE, "launcher")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def heredocs(source: str) -> list[str]:
    return [part.split("\nPY", 1)[0] for part in source.split("<<'PY'\n")[1:]]


class OwnedStat:
    """Portable proxy for AUH's guangyi.chen:camdpi ownership contract."""

    def __init__(self, value: os.stat_result):
        self._value = value

    def __getattr__(self, name):
        if name == "st_uid":
            return 2012
        if name == "st_gid":
            return 2000
        return getattr(self._value, name)


class ModeStat:
    def __init__(self, value, mode: int):
        self._value = value
        self._mode = mode

    def __getattr__(self, name):
        if name == "st_mode":
            return self._mode
        return getattr(self._value, name)


@contextmanager
def auh_owner_view(module):
    real_fstat, real_stat, real_lstat = os.fstat, os.stat, os.lstat
    real_path_lstat = Path.lstat
    identity_name = "identity" if hasattr(module, "identity") else "ident"
    real_identity = getattr(module, identity_name)

    def owned_fstat(*args, **kwargs):
        return OwnedStat(real_fstat(*args, **kwargs))

    def owned_stat(*args, **kwargs):
        return OwnedStat(real_stat(*args, **kwargs))

    def owned_lstat(*args, **kwargs):
        return OwnedStat(real_lstat(*args, **kwargs))

    def owned_identity(value):
        return real_identity(OwnedStat(value))

    def owned_path_lstat(path):
        return OwnedStat(real_path_lstat(path))

    with mock.patch.object(module.os, "fstat", owned_fstat), \
         mock.patch.object(module.os, "stat", owned_stat), \
         mock.patch.object(module.os, "lstat", owned_lstat), \
         mock.patch.object(module, identity_name, owned_identity), \
         mock.patch.object(Path, "lstat", owned_path_lstat):
        yield


class SnapshotFixture:
    def __init__(self, base: Path):
        self.root = (base / "snapshot").resolve()
        self.root.mkdir()
        self.old = "/sealed/r5f"
        self.staging = "/fresh/exact5"
        self.release = {}
        self.diagnostics = {}
        self.raw = {}
        for relative in materializer.RELEASE_FILES:
            value = ("release:" + relative + "\n").encode()
            self.raw[relative] = value
            self.release[relative] = sha(value)
        for relative in materializer.DIAGNOSTIC_SOURCE_FILES:
            value = ("diagnostic:" + relative + "\n").encode()
            self.raw[relative] = value
            self.diagnostics[relative] = sha(value)
        audit_raw = b'{"fixture":"audit"}\n'
        self.raw[materializer.AUDIT_RELATIVE] = audit_raw
        self.audit_sha = sha(audit_raw)
        self.materializer_relative = (
            "methods/bernini_action_editing/tools/"
            "materialize_case01_source_bone_exact5_r64_package_v1.py"
        )
        materializer_raw = b"# captured fixture materializer\n"
        self.raw[self.materializer_relative] = materializer_raw
        self.materializer_sha = sha(materializer_raw)
        for relative, value in self.raw.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            path.chmod(0o444)
        self.rows = []
        fresh_release = {
            "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py",
            "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py",
            "methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py",
        }
        for relative in sorted(self.raw):
            provenance = (
                "sealed_r5f_snapshot"
                if relative in self.release and relative not in fresh_release
                else "new_exact5_staging"
            )
            self.rows.append({
                "path": relative, "sha256": sha(self.raw[relative]),
                "size": len(self.raw[relative]), "mode": 0o444,
                "provenance": provenance,
            })
        self.manifest = {
            "schema_version": "case01-source-bone-exact5-source-snapshot-v1",
            "status": "SEALED_NOT_EXECUTED", "target_root": str(self.root),
            "old_r5f_snapshot_root": self.old,
            "new_staging_root": self.staging, "file_count": 23,
            "old_reused_file_count": 16, "new_staged_file_count": 7,
            "physical_file_count_including_manifest": 24,
            "release_file_count": 19, "sealed_r5f_infer_lora_reused": True,
            "working_tree_infer_lora_read": False,
            "slurm_step_launched": False, "files": self.rows,
        }
        self.write_manifest(self.manifest)
        self.seal_directories()

    def write_manifest(self, value: dict) -> None:
        value = copy.deepcopy(value)
        value.pop("snapshot_digest", None)
        value["snapshot_digest"] = materializer.object_sha256(value)
        path = self.root / materializer.SNAPSHOT_MANIFEST
        if path.exists():
            path.chmod(0o644)
        path.write_bytes(materializer.canonical_json_bytes(value) + b"\n")
        path.chmod(0o444)
        self.manifest = value

    def unseal_directories(self) -> None:
        for path in [self.root, *self.root.rglob("*")]:
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)

    def seal_directories(self) -> None:
        directories = [path for path in self.root.rglob("*") if path.is_dir()]
        for path in sorted(directories, reverse=True):
            path.chmod(0o555)
        self.root.chmod(0o555)

    @contextmanager
    def patched(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(materializer, "SOURCE_SNAPSHOT_ROOT", self.root))
            stack.enter_context(mock.patch.object(materializer, "OLD_R5F_SNAPSHOT_ROOT", Path(self.old)))
            stack.enter_context(mock.patch.object(materializer, "SOURCE_STAGING_ROOT", Path(self.staging)))
            stack.enter_context(mock.patch.object(materializer, "RELEASE_FILES", self.release))
            stack.enter_context(mock.patch.object(materializer, "DIAGNOSTIC_SOURCE_FILES", self.diagnostics))
            stack.enter_context(mock.patch.object(materializer, "AUDIT_SHA256", self.audit_sha))
            stack.enter_context(auh_owner_view(materializer))
            yield

    def replay(self):
        with self.patched():
            return materializer._preflight_snapshot(self.root, self.materializer_sha)


class SnapshotAndPackageTests(unittest.TestCase):
    def _assert_exact_tree_hostiles_precede_fchmod(self, module, error_type) -> None:
        for kind in ("fifo", "synthetic-socket", "symlink-dir"):
            with self.subTest(module=module.__name__, kind=kind), \
                 tempfile.TemporaryDirectory(dir="/tmp") as value:
                base = Path(value)
                root = base / "tree"
                root.mkdir(mode=0o700)
                leaf = root / "file"
                leaf.write_bytes(b"sealed\n")
                leaf.chmod(0o444)
                hostile = root / "hostile"
                if kind == "fifo":
                    os.mkfifo(hostile)
                elif kind == "symlink-dir":
                    outside = base / "outside"
                    outside.mkdir()
                    hostile.symlink_to(outside, target_is_directory=True)
                else:
                    hostile.write_bytes(b"socket placeholder\n")
                    hostile.chmod(0o444)
                with auh_owner_view(module), mock.patch.object(
                    module.os, "fchmod", wraps=os.fchmod,
                ) as fchmod_spy:
                    if kind == "synthetic-socket":
                        owned_stat = module.os.stat

                        def socket_stat(path, *args, **kwargs):
                            result = owned_stat(path, *args, **kwargs)
                            if path == "hostile" and kwargs.get("dir_fd") is not None:
                                return ModeStat(result, stat.S_IFSOCK | 0o600)
                            return result

                        stat_patch = mock.patch.object(module.os, "stat", socket_stat)
                    else:
                        stat_patch = mock.patch.object(module.os, "stat", module.os.stat)
                    with stat_patch, self.assertRaises(error_type):
                        module.open_exact_tree(
                            root, {"file": 0o444}, {".": 0o700},
                        )
                    fchmod_spy.assert_not_called()

    def test_builder_target_and_package_final_specials_reject_before_fchmod(self) -> None:
        self._assert_exact_tree_hostiles_precede_fchmod(
            snapshot, snapshot.Exact5SnapshotError,
        )
        self._assert_exact_tree_hostiles_precede_fchmod(
            materializer, materializer.Exact5PackageError,
        )

    def test_staging_closure_rejects_fifo_and_symlink_directory(self) -> None:
        for kind in ("fifo", "symlink-dir"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir="/tmp") as value:
                base = Path(value)
                staging = (base / "staging").resolve()
                staging.mkdir()
                expected = set(snapshot.NEW_STAGED_FILES) | {snapshot.BUILDER_RELATIVE}
                for relative in expected:
                    path = staging / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((relative + "\n").encode())
                    path.chmod(0o444)
                hostile = staging / "hostile"
                if kind == "fifo":
                    os.mkfifo(hostile)
                else:
                    outside = base / "outside"
                    outside.mkdir()
                    hostile.symlink_to(outside, target_is_directory=True)
                with mock.patch.object(snapshot, "STAGING_ROOT", staging), \
                     auh_owner_view(snapshot), \
                     mock.patch.object(snapshot.os, "fchmod", wraps=os.fchmod) as fchmod_spy:
                    with self.assertRaises(snapshot.Exact5SnapshotError):
                        snapshot._exact_staging_closure()
                    fchmod_spy.assert_not_called()

    def test_exact_counts_launch18_and_cross_file_pins(self) -> None:
        self.assertEqual(len(snapshot.OLD_REUSED_FILES), 16)
        self.assertEqual(len(snapshot.NEW_STAGED_FILES), 7)
        self.assertEqual(len(snapshot.NEW_STAGED_FILES) + 1, 8)
        self.assertEqual(len(materializer.RELEASE_FILES), 19)
        self.assertEqual(len(materializer._expected_snapshot_files()), 24)
        self.assertEqual(len(launcher.IDENTITY_ROLES), 18)
        self.assertEqual(len(set(launcher.IDENTITY_ROLES)), 18)
        launch = materializer.launch_input(
            Path("/tmp/exact5-package").resolve(), "143808",
            "auh7-1b-gpu-292", Path("/tmp/exact5-package/plan.json").resolve(),
        )
        self.assertEqual({role for role in launcher.IDENTITY_ROLES if role in launch}, set(launcher.IDENTITY_ROLES))
        self.assertEqual(tuple(materializer.TASK_IDS), tuple(exact_eval.TASK_IDS))
        self.assertEqual(tuple(launcher.TASK_IDS), tuple(exact_eval.TASK_IDS))

        for relative, expected in snapshot.NEW_STAGED_FILES.items():
            local = REPO / relative
            self.assertTrue(local.is_file(), relative)
            self.assertEqual(sha(local.read_bytes()), expected, relative)
        for relative, expected in materializer.DIAGNOSTIC_SOURCE_FILES.items():
            self.assertEqual(sha((REPO / relative).read_bytes()), expected, relative)
        for relative in (
            "methods/bernini_action_editing/case01_source_bone_exact5_runner_v1.py",
            "methods/bernini_action_editing/case01_source_bone_exact5_eval_v1.py",
            "methods/bernini_action_editing/case01_source_bone_exact5_spooled_launcher_auh_v1.py",
        ):
            self.assertEqual(materializer.RELEASE_FILES[relative], sha((REPO / relative).read_bytes()))

    def test_snapshot_manifest_baseline_and_hostile_schema_rows(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            fixture = SnapshotFixture(Path(value))
            self.assertEqual(set(fixture.replay()), set(fixture.raw))
            fixture.unseal_directories()

        mutators = {
            "extra-top": lambda row: row.update(unexpected=True),
            "missing-top": lambda row: row.pop("status"),
            "wrong-count": lambda row: row.update(file_count=22),
            "wrong-target": lambda row: row.update(target_root="/wrong"),
            "duplicate-row": lambda row: row["files"].__setitem__(-1, copy.deepcopy(row["files"][0])),
            "wrong-provenance": lambda row: row["files"][0].update(provenance="new_exact5_staging" if row["files"][0]["provenance"] == "sealed_r5f_snapshot" else "sealed_r5f_snapshot"),
            "reordered": lambda row: row.update(files=list(reversed(row["files"]))),
        }
        for label, mutate in mutators.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as value:
                fixture = SnapshotFixture(Path(value))
                fixture.unseal_directories()
                changed = copy.deepcopy(fixture.manifest)
                mutate(changed)
                fixture.write_manifest(changed)
                fixture.seal_directories()
                with self.assertRaises(materializer.Exact5PackageError):
                    fixture.replay()
                fixture.unseal_directories()

    def test_snapshot_rejects_fifo_socket_and_symlink_directory(self) -> None:
        for kind in ("fifo", "synthetic-socket", "symlink-dir"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir="/tmp") as value:
                base = Path(value)
                fixture = SnapshotFixture(base)
                fixture.unseal_directories()
                hostile = fixture.root / "hostile"
                listener = None
                if kind == "fifo":
                    os.mkfifo(hostile)
                elif kind == "synthetic-socket":
                    hostile.write_bytes(b"socket placeholder\n")
                    hostile.chmod(0o444)
                else:
                    outside = base / "outside"
                    outside.mkdir()
                    hostile.symlink_to(outside, target_is_directory=True)
                fixture.seal_directories()
                try:
                    if kind != "synthetic-socket":
                        with self.assertRaises(materializer.Exact5PackageError):
                            fixture.replay()
                    else:
                        with fixture.patched():
                            owned_stat = materializer.os.stat

                            def socket_stat(path, *args, **kwargs):
                                result = owned_stat(path, *args, **kwargs)
                                if path == "hostile" and kwargs.get("dir_fd") is not None:
                                    return ModeStat(result, stat.S_IFSOCK | 0o600)
                                return result

                            with mock.patch.object(materializer.os, "stat", socket_stat):
                                with self.assertRaises(materializer.Exact5PackageError):
                                    materializer._preflight_snapshot(
                                        fixture.root, fixture.materializer_sha,
                                    )
                finally:
                    if listener is not None:
                        listener.close()
                    fixture.unseal_directories()

    def test_stable_file_rejects_mode_symlink_path_and_named_swap(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value).resolve()
            target = root / "source.py"
            target.write_bytes(b"source\n")
            target.chmod(0o444)
            expected = sha(target.read_bytes())
            self.assertEqual(materializer.stable_file(target, expected, 0o444), b"source\n")
            target.chmod(0o644)
            with self.assertRaises(materializer.Exact5PackageError):
                materializer.stable_file(target, expected, 0o444)
            target.chmod(0o444)
            alias = root / "alias.py"
            alias.symlink_to(target)
            with self.assertRaises(materializer.Exact5PackageError):
                materializer.stable_file(alias, expected, 0o444)
            with self.assertRaises(materializer.Exact5PackageError):
                materializer.stable_file(Path("relative.py"), expected, 0o444)

            replacement = root / "replacement.py"
            replacement.write_bytes(b"source\n")
            replacement.chmod(0o444)
            real_pread = os.pread
            swapped = False

            def racing_pread(*args, **kwargs):
                nonlocal swapped
                raw = real_pread(*args, **kwargs)
                if not swapped:
                    swapped = True
                    os.replace(replacement, target)
                return raw

            with mock.patch.object(materializer.os, "pread", racing_pread):
                with self.assertRaises(materializer.Exact5PackageError):
                    materializer.stable_file(target, expected, 0o444)

    def test_no_named_directory_chmod_and_special_tree_guards(self) -> None:
        for path in (SNAPSHOT_SOURCE, MATERIALIZER_SOURCE):
            source = path.read_text("utf-8")
            tree = ast.parse(source)
            named_chmods = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os" and node.func.attr == "chmod"
            ]
            self.assertEqual(named_chmods, [], path.name)
            self.assertIn("os.fchmod", source)
            self.assertIn("O_DIRECTORY", source)
            self.assertIn("O_NOFOLLOW", source)
            self.assertIn("special entry", source)


class PlanAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = exact_eval.build_asset_authority(
            ASSET_ROOT / "manifest.json", ASSET_ROOT, AUDIT,
        )

    def plan(self, output: Path) -> dict:
        return exact_eval.build_plan(
            asset_authority=self.authority,
            checkpoint_manifest={
                **exact_eval.EXPECTED_CHECKPOINT,
                "path": "/authority/checkpoint-00000644/checkpoint_manifest.json",
            },
            producer={
                **exact_eval.EXPECTED_PRODUCER,
                "infer_lora_path": "/release/infer_lora.py",
                "ffprobe_path": "/runtime/ffprobe",
            },
            output_root=output,
        )

    def test_audit_and_plan_pins_plus_4_6_reordered_rejections(self) -> None:
        audit_raw = AUDIT.read_bytes()
        self.assertEqual(sha(audit_raw), exact_eval.INDEPENDENT_AUDIT_SHA256)
        self.assertEqual(len(audit_raw), exact_eval.INDEPENDENT_AUDIT_SIZE)
        self.assertEqual(sha(audit_raw), materializer.AUDIT_SHA256)
        self.assertEqual(len(audit_raw), materializer.AUDIT_SIZE)
        with tempfile.TemporaryDirectory() as value:
            plan = self.plan(Path(value).resolve())
        self.assertEqual(plan["task_count"], 5)
        self.assertEqual([row["task_id"] for row in plan["tasks"]], list(exact_eval.TASK_IDS))
        mutations = []
        four = copy.deepcopy(plan); four["tasks"].pop(); four["task_count"] = 4; mutations.append(four)
        six = copy.deepcopy(plan); six["tasks"].append(copy.deepcopy(six["tasks"][-1])); six["task_count"] = 6; mutations.append(six)
        reordered = copy.deepcopy(plan); reordered["tasks"].reverse(); mutations.append(reordered)
        for changed in mutations:
            changed["plan_digest"] = exact_eval.object_sha256({k: v for k, v in changed.items() if k != "plan_digest"})
            with self.assertRaises(exact_eval.Exact5EvalError):
                exact_eval.validate_plan(changed)

    def test_ffprobe_drift_and_rank_cache_guards_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value).resolve()
            ffprobe = root / "ffprobe"
            ffprobe.write_bytes(b"#!/bin/sh\nexit 0\n")
            ffprobe.chmod(0o555)
            with self.assertRaises(static_probe.StaticProbeError):
                static_probe.stable(ffprobe, "0" * 64)
        static_text = STATIC_SOURCE.read_text("utf-8")
        self.assertIn("os.path.lexists(str(RANK_CACHE_ROOT))", static_text)
        for gate in (STATIC_GATE, FAKE_GATE, GPU_CONTROLLER):
            source = gate.read_text("utf-8")
            self.assertTrue(
                ('! -e "$CACHE"' in source and '! -L "$CACHE"' in source)
                or (
                    '"$CACHE"' in source and '! -e "$fresh"' in source
                    and '! -L "$fresh"' in source
                )
            )


class ControllerTests(unittest.TestCase):
    def test_shell_and_embedded_python_normal_and_optimized_compile(self) -> None:
        for path in (STATIC_GATE, FAKE_GATE, GPU_CONTROLLER):
            result = subprocess.run(
                ["/bin/bash", "-n", str(path)], capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            blocks = heredocs(path.read_text("utf-8"))
            self.assertGreaterEqual(len(blocks), 2)
            for optimize in (0, 2):
                for index, block in enumerate(blocks):
                    compile(block, f"{path.name}:heredoc{index}", "exec", optimize=optimize)

    def test_gpu_controller_is_held_exact8_single_srun_and_fd_replayed(self) -> None:
        source = GPU_CONTROLLER.read_text("utf-8")
        self.assertIn("readonly CONTROLLER_STATE=HOLD_PENDING_EXACT_CPU_GATE_PINS", source)
        self.assertLess(source.index('[[ "$CONTROLLER_STATE" == READY ]]'), source.index("exec {ROOT_PYTHON_FD}"))
        self.assertLess(source.index('[[ "$CONTROLLER_STATE" == READY ]]'), source.index("/usr/bin/srun"))
        self.assertEqual(source.count("/usr/bin/srun"), 1)
        self.assertLess(source.index('"status": "ATTEMPT_CLAIMED_BEFORE_SRUN"'), source.index("/usr/bin/srun"))
        for token in (
            "--nodes=1 --ntasks=1", "--nodelist=auh7-1b-gpu-292",
            "--cpus-per-task=64 --mem=64G --gpus-per-node=8",
            "--exclusive --exact", "--export=NONE", "exec /bin/bash -p -s",
            '<&"$PAYLOAD_FD"', "MAX_GATE_STEP", "len(set(gate_rows)) != 2",
            'for relative in ("outputs/media", "final", "runtime")',
            "package_fd", "payload_fd", "post-use held authority replay differs",
        ):
            self.assertIn(token, source)
        self.assertNotIn("/usr/bin/chmod", source)
        self.assertNotIn("os.chmod(", source)
        self.assertIn("os.fchmod(descriptor, 0o400)", source)
        self.assertGreaterEqual(source.count("HOLD_"), 15)

        result = subprocess.run(
            ["/bin/bash", "-p", "-s"], input=GPU_CONTROLLER.read_bytes(),
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(result.returncode, 88)
        self.assertIn(b"HOLD pending exact CPU receipt/evidence pins", result.stderr)

    def test_gpu_parent_object_identity_survives_expected_child_create(self) -> None:
        preflight = heredocs(GPU_CONTROLLER.read_text("utf-8"))[0]
        tree = ast.parse(preflight)
        node = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "object_id"
        )
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "object_id", "exec"), namespace)
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "sentinel").write_bytes(b"existing\n")
            before = root.stat()
            (root / "attempt").write_bytes(b"claimed\n")
            after = root.stat()
            before_id = namespace["object_id"](before)
            after_id = namespace["object_id"](after)
            if sys.platform == "linux":
                self.assertEqual(before_id, after_id)
            else:
                # APFS reports a directory nlink change for each new leaf;
                # AUH is Linux, where a regular-file child leaves nlink stable.
                self.assertEqual(before_id[:5] + before_id[6:], after_id[:5] + after_id[6:])
            self.assertNotEqual(
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns),
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            )

    def test_cpu_controllers_lock_receipt_attempt_evidence_and_held_logs(self) -> None:
        for path in (STATIC_GATE, FAKE_GATE):
            source = path.read_text("utf-8")
            self.assertEqual(source.count("/usr/bin/srun"), 1)
            self.assertLess(source.index("ATTEMPT_CLAIMED_BEFORE_SRUN"), source.index("/usr/bin/srun"))
            self.assertIn("set(receipt)!=receipt_fields", source)
            self.assertIn("claimed!=digest(unsigned)", source)
            self.assertIn("set(attempt)!=attempt_fields", source)
            self.assertIn("attempt_claimed!=digest(attempt_unsigned)", source)
            self.assertIn("stdout_raw!=b\"CASE01_EXACT5_", source)
            self.assertIn("O_DIRECTORY", source)
            self.assertIn("O_NOFOLLOW", source)
            self.assertIn("os.fchmod", source)
            self.assertNotIn("/usr/bin/chmod", source)
            self.assertNotIn("os.chmod(", source)
            self.assertIn("canonical_json_plus_lf", source)
            self.assertIn("single_srun_attempt", source)
            self.assertIn("retry_allowed", source)


if __name__ == "__main__":
    unittest.main()
