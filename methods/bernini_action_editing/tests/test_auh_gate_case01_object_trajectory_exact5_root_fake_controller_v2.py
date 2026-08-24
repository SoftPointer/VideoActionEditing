#!/usr/bin/env python3
"""Hostile tests for receipt-first exact5 captured-root HOLD/READY."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = METHOD_ROOT / (
    "scripts/auh_gate_case01_object_trajectory_exact5_"
    "root_fake_once_v2.HOLD.py"
)
READY_PATH = METHOD_ROOT / (
    "scripts/auh_gate_case01_object_trajectory_exact5_"
    "root_fake_once_v2.READY.py"
)
ROOT_FAKE_PATH = (
    METHOD_ROOT / "case01_object_trajectory_exact5_root_fake_runner_v1.py"
)
LAUNCHER_PATH = (
    METHOD_ROOT / "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> types.ModuleType:
    name = "_test_root_fake_hold_controller_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load(CONTROLLER_PATH)


def root_spec() -> dict[str, object]:
    identities = {
        role: {
            "path": f"/authority/{role}",
            "sha256": (format(index + 1, "x") * 64)[:64],
            "size": index + 1,
        }
        for index, role in enumerate(controller.IDENTITY_ROLES)
    }
    identities["runner"] = {
        "path": "/authority/case01_object_trajectory_exact5_runner_v1.py",
        "sha256": controller.PRODUCTION_RUNNER_SHA256,
        "size": controller.PRODUCTION_RUNNER_SIZE,
    }
    return {
        "schema_version": controller.ROOT_SPEC_SCHEMA,
        "campaign_mode": controller.CAMPAIGN,
        "launch_allowed": False,
        "identities": identities,
        "captured_runner": {
            "path": str(controller.ROOT_FAKE_RUNNER_PATH),
            "sha256": controller.ROOT_FAKE_RUNNER_SHA256,
            "size": controller.ROOT_FAKE_RUNNER_SIZE,
        },
        "launch_input": {
            "path": str(controller.LAUNCH_INPUT_PATH),
            "sha256": "a" * 64,
            "size": 123,
        },
        "result_path": str(controller.ROOT_FAKE_OUTPUT_PATH),
    }


def root_entry(spec: dict[str, object]) -> dict[str, object]:
    identities = spec["identities"]
    assert isinstance(identities, dict)
    entry: dict[str, object] = {
        "schema_version": controller.ROOT_ENTRY_SCHEMA,
        "release_digest": controller.object_digest(spec),
        "identity_roles": list(controller.IDENTITY_ROLES),
        "identity_set_digest": controller.object_digest(identities),
        "launch_input_sha256": "a" * 64,
        "production_runner": identities["runner"],
        "captured_runner": spec["captured_runner"],
        "captured_runner_identity": list(range(11)),
        "plan_sha256": identities["plan"]["sha256"],
        "task_ids": list(controller.TASK_IDS),
        "arm_order": list(controller.ARM_ORDER),
        "all_exact25_named_identities_replayed": True,
        "captured_runner_outside_exact25": True,
        "captured_runner_bytes_compiled": True,
        "publication_performed": False,
    }
    entry["authority_digest"] = controller.object_digest(entry)
    return entry


def root_result(
    spec: dict[str, object], entry: dict[str, object],
) -> dict[str, object]:
    identities = spec["identities"]
    assert isinstance(identities, dict)
    value: dict[str, object] = {
        "schema_version": controller.ROOT_FAKE_SCHEMA,
        "status": "PASS_CAPTURED_ROOT_FAKE_HOLD",
        "campaign_mode": controller.CAMPAIGN,
        "launch_allowed": False,
        "exact_identity_count": 25,
        "identity_roles": list(controller.IDENTITY_ROLES),
        "task_ids": list(controller.TASK_IDS),
        "arm_order": list(controller.ARM_ORDER),
        "release_digest": controller.object_digest(spec),
        "identity_set_digest": controller.object_digest(identities),
        "launch_input_sha256": "a" * 64,
        "entry_authority_digest": entry["authority_digest"],
        "plan_sha256": identities["plan"]["sha256"],
        "production_runner_sha256": controller.PRODUCTION_RUNNER_SHA256,
        "captured_runner_sha256": controller.ROOT_FAKE_RUNNER_SHA256,
        "all_exact25_named_identities_replayed": True,
        "captured_runner_outside_exact25": True,
        "captured_runner_bytes_compiled": True,
        "torch_imported": False,
        "renderer_imported": False,
        "publication_performed": False,
    }
    value["receipt_digest"] = controller.object_digest(value)
    return value


def valid_plan() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    identities = {
        role: {"path": f"/x/{role}", "sha256": "b" * 64, "size": 7}
        for role in controller.IDENTITY_ROLES
    }
    checkpoint = {
        "path": identities["r64_checkpoint_manifest"]["path"],
        "sha256": identities["r64_checkpoint_manifest"]["sha256"],
    }
    producer: dict[str, object] = {}
    for role, keys in {
        "legacy_infer_alias": (
            "infer_lora_path", "infer_lora_sha256", "infer_lora_size",
        ),
        "adapter": (
            "inference_wrapper_path", "inference_wrapper_sha256",
            "inference_wrapper_size",
        ),
        "trajectory_projection": (
            "trajectory_projection_module_path",
            "trajectory_projection_module_sha256",
            "trajectory_projection_module_size",
        ),
        "trajectory_scaffold_module": (
            "trajectory_scaffold_module_path",
            "trajectory_scaffold_module_sha256",
            "trajectory_scaffold_module_size",
        ),
        "ffprobe": ("ffprobe_path", "ffprobe_sha256", "ffprobe_size"),
    }.items():
        row = identities[role]
        producer[keys[0]] = row["path"]
        producer[keys[1]] = row["sha256"]
        producer[keys[2]] = row["size"]
    tasks = []
    for arm, task_id in zip(controller.ARM_ORDER, controller.TASK_IDS):
        tasks.append({
            "task_id": task_id,
            "oracle_arm": arm,
            "source_onset_policy": "hard1_every_step",
            "external_conditions": (
                {} if arm in {"null_before", "null_after"}
                else {name: {} for name in controller.EXTERNAL_KEYS}
            ),
            "adapter": {"checkpoint_manifest": checkpoint},
        })
    return ({
        "status": "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY",
        "production_ready": False,
        "launch_allowed": False,
        "hold_reasons": ["fixture remains HOLD"],
        "producer": producer,
        "checkpoint_manifest": checkpoint,
        "tasks": tasks,
    }, identities)


class ExplosiveArgv:
    def __iter__(self):
        raise AssertionError("HOLD iterated argv")


class FakeHeld:
    def __init__(self, path: Path, raw: bytes, events: list[str]) -> None:
        self.path = path
        self.raw = raw
        self.events = events
        self.descriptor = 101
        self.held_identity = tuple(range(11))

    def replay(self) -> None:
        self.events.append("replay:" + self.path.name)

    def close(self) -> None:
        self.events.append("close:" + self.path.name)


class FakeDirectory:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def replay(self) -> None:
        self.events.append("replay:package-root")

    def close(self) -> None:
        self.events.append("close:package-root")


class RootFakeControllerV2Test(unittest.TestCase):
    def test_hold_gate_precedes_argv_pins_paths_and_controller(self) -> None:
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("called")
            raise AssertionError("HOLD crossed first gate")

        stderr = io.StringIO()
        with mock.patch.object(controller, "blocked_dynamic_pins", forbidden), \
             mock.patch.object(controller, "controller", forbidden), \
             mock.patch.object(controller.os, "open", forbidden), \
             mock.patch.object(controller.os, "lstat", forbidden), \
             redirect_stderr(stderr):
            result = controller.main(ExplosiveArgv())
        self.assertEqual(result, 88)
        self.assertEqual(touched, [])
        self.assertIn("HOLD", stderr.getvalue())

    def test_final_package_static_and_root_pins_are_exact_unblocked(self) -> None:
        self.assertEqual(controller.blocked_dynamic_pins(), ())
        self.assertEqual(controller.dynamic_pin_values(), {
            "package_publication_receipt_sha256":
                "b3766694f24ead6d7da04e5a1da077de69a9dbbf06df8f06ff0c9db77d84c533",
            "package_publication_receipt_size": 2_209,
            "package_publication_receipt_digest":
                "5cab7d2db0079d4b6960273e681c20b60941b892c3a42bfdbd70be819d991cb9",
            "materialization_report_sha256":
                "e1e4d7ae266828f27f77f39528672cd7ccae9aa067fdee291d4e5e32f9a9bf2f",
            "materialization_report_size": 21_743,
            "materialization_report_digest":
                "99ba2595bde82371257a46b08ef55f77f54cb5b86877aa791daf6976237868c4",
            "static_receipt_sha256":
                "3e65f4342f33a0d4264fa7f09759bad3aa2f4c4622a6965db675f2c551fb07b8",
            "static_receipt_size": 1_035,
            "static_receipt_digest":
                "7ed16825624ca99dc7f2cbbea3c9a5a991122108aff4867796a3ac01456ab6be",
            "package_root_identity": [
                48, 12_038_280_342_419_913_116, 2012, 2000, 16_832, 2,
                0, 4096, 0, 1_787_357_728_317_453_482,
                1_787_357_728_652_385_810,
            ],
        })

    def test_ready_is_strict_single_state_line_overlay(self) -> None:
        hold_lines = CONTROLLER_PATH.read_bytes().splitlines(keepends=True)
        ready_lines = READY_PATH.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(hold_lines), len(ready_lines))
        changed = [
            index for index, (hold, ready) in enumerate(
                zip(hold_lines, ready_lines)
            ) if hold != ready
        ]
        self.assertEqual(len(changed), 1)
        line = changed[0]
        self.assertEqual(
            hold_lines[line],
            b'CONTROLLER_STATE = "HOLD_PENDING_PACKAGE_STATIC_AND_ROOT_PINS"\n',
        )
        self.assertEqual(
            ready_lines[line],
            b'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_IN_PROCESS_CAPTURED_ROOT"\n',
        )
        ready = load(READY_PATH)
        self.assertEqual(ready.blocked_dynamic_pins(), ())
        self.assertEqual(ready.CONTROLLER_STATE, ready.READY_STATE)
        spec = root_spec()
        entry = root_entry(spec)
        expected = root_result(spec, entry)
        called = mock.Mock(return_value=expected)
        stdout = io.StringIO()
        with mock.patch.object(ready, "controller", called), \
             redirect_stdout(stdout):
            code = ready.main(["--execute", ready.authorization_token()])
        self.assertEqual(code, 0)
        called.assert_called_once_with()
        self.assertEqual(json.loads(stdout.getvalue()), expected)

    def test_source_has_one_in_process_call_and_receipt_first_order(self) -> None:
        raw = CONTROLLER_PATH.read_bytes()
        source = raw.decode("utf-8")
        tree = ast.parse(raw, filename=str(CONTROLLER_PATH))
        functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        main = functions["main"]
        self.assertIsInstance(main.body[0], ast.If)
        self.assertEqual(
            {
                node.id for node in ast.walk(main.body[0].test)
                if isinstance(node, ast.Name)
            },
            {"CONTROLLER_STATE", "READY_STATE"},
        )
        captured_calls = [
            node for node in ast.walk(functions["run_isolated_root_fake"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "captured_main"
        ]
        self.assertEqual(len(captured_calls), 1)
        imports = {
            alias.name.split(".")[0]
            for node in tree.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue({"subprocess", "socket"}.isdisjoint(imports))
        self.assertNotIn("subprocess.run(", source)
        self.assertNotIn("subprocess.Popen(", source)
        self.assertNotIn("/usr/bin/ssh", source)
        self.assertNotIn("/usr/bin/srun", source)
        self.assertLess(
            source.index("publication_authority = open_authority("),
            source.index("report_authority = open_authority("),
        )
        self.assertLess(
            source.index("report_authority = open_authority("),
            source.index("static_authority = open_authority("),
        )
        self.assertLess(
            source.index("static_authority = open_authority("),
            source.index("package_root = open_package_root("),
        )
        self.assertLess(
            source.index("result = run_isolated_root_fake("),
            source.index("output_raw = create_immutable_receipt("),
        )
        self.assertEqual(
            (sha256(ROOT_FAKE_PATH.read_bytes()), ROOT_FAKE_PATH.stat().st_size),
            (controller.ROOT_FAKE_RUNNER_SHA256,
             controller.ROOT_FAKE_RUNNER_SIZE),
        )
        self.assertEqual(
            (sha256(LAUNCHER_PATH.read_bytes()), LAUNCHER_PATH.stat().st_size),
            (controller.LAUNCHER_SHA256, controller.LAUNCHER_SIZE),
        )
        for optimize in (0, 2):
            self.assertIsNotNone(
                compile(raw, str(CONTROLLER_PATH), "exec", optimize=optimize)
            )
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertNotIn(b"\r", raw)

    def test_strict_json_rejects_duplicate_nan_and_noncanonical(self) -> None:
        self.assertEqual(
            controller.strict_json(b'{"a":1,"b":2}\n', label="valid"),
            {"a": 1, "b": 2},
        )
        for raw in (
            b'{"a":1,"a":2}\n', b'{"b":2, "a":1}\n',
            b'{"a":NaN}\n', b'{"a":1}',
        ):
            with self.subTest(raw=raw), self.assertRaises(
                controller.RootFakeControllerError
            ):
                controller.strict_json(raw, label="hostile")

    def test_plan_truth_rejects_onset_external_and_producer_tamper(self) -> None:
        plan, identities = valid_plan()
        controller.validate_plan_and_crosslinks(plan, identities)
        for mutation in ("onset", "null-external", "producer"):
            hostile = __import__("copy").deepcopy(plan)
            if mutation == "onset":
                hostile["tasks"][2]["source_onset_policy"] = "none"
            elif mutation == "null-external":
                hostile["tasks"][0]["external_conditions"] = {
                    "stage0_masks": {}
                }
            else:
                hostile["producer"]["inference_wrapper_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(
                controller.RootFakeControllerError
            ):
                controller.validate_plan_and_crosslinks(hostile, identities)

    def test_exact25_duplicate_path_is_rejected_before_any_open(self) -> None:
        spec = root_spec()
        identities = spec["identities"]
        self.assertIsInstance(identities, dict)
        identities["legacy_exact5_runner"] = dict(identities["runner"])
        opened = mock.Mock(side_effect=AssertionError("identity path opened"))
        with mock.patch.object(controller, "open_authority", opened), \
             self.assertRaisesRegex(
                 controller.RootFakeControllerError,
                 "exact25 identity closure differs",
             ):
            controller.open_exact25(identities)
        opened.assert_not_called()

    def test_static_receipt_crosslink_and_five_arm_truth_are_strict(self) -> None:
        report = {"launch": {"input": {"sha256": "a" * 64}}}
        value: dict[str, object] = {
            "schema_version": controller.STATIC_SCHEMA,
            "status": "ADMITTED_STATIC_HOLD_ONLY",
            "launch_allowed": False,
            "blocked_roles": [],
            "final_source_pins_complete": True,
            "exact_identity_count": 25,
            "task_ids": list(controller.TASK_IDS),
            "arm_order": list(controller.ARM_ORDER),
            "all_tasks_hard1_every_step": True,
            "null_arms_have_no_external_conditions": True,
            "route_and_active_arms_have_external_conditions": True,
            "torch_imported": False,
            "renderer_imported": False,
            "publication_performed": False,
            "input_sha256": "a" * 64,
            "launcher_sha256": controller.LAUNCHER_SHA256,
        }
        value["receipt_digest"] = controller.object_digest(value)
        raw = controller.canonical(value) + b"\n"
        held = types.SimpleNamespace(raw=raw)
        with mock.patch.object(
            controller, "STATIC_RECEIPT_SHA256", sha256(raw),
        ), mock.patch.object(
            controller, "STATIC_RECEIPT_SIZE", len(raw),
        ), mock.patch.object(
            controller, "STATIC_RECEIPT_DIGEST", value["receipt_digest"],
        ):
            self.assertEqual(
                controller.validate_static_receipt(held, report), value,
            )
            for field in (
                "input_sha256", "all_tasks_hard1_every_step",
                "null_arms_have_no_external_conditions",
            ):
                hostile = dict(value)
                hostile[field] = (
                    "0" * 64 if field == "input_sha256" else False
                )
                hostile.pop("receipt_digest")
                hostile["receipt_digest"] = controller.object_digest(hostile)
                hostile_raw = controller.canonical(hostile) + b"\n"
                hostile_held = types.SimpleNamespace(raw=hostile_raw)
                with self.subTest(field=field), mock.patch.object(
                    controller, "STATIC_RECEIPT_SHA256", sha256(hostile_raw),
                ), mock.patch.object(
                    controller, "STATIC_RECEIPT_SIZE", len(hostile_raw),
                ), mock.patch.object(
                    controller, "STATIC_RECEIPT_DIGEST",
                    hostile["receipt_digest"],
                ), self.assertRaises(
                    controller.RootFakeControllerError
                ):
                    controller.validate_static_receipt(hostile_held, report)

    def test_held_authority_rejects_links_and_inode_replacement(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            path = base / "authority.json"
            raw = b'{"authority":true}\n'
            path.write_bytes(raw)
            path.chmod(0o400)
            info = path.stat()
            held = controller.open_authority(
                path, expected_sha256=sha256(raw), expected_size=len(raw),
                expected_mode=0o400, expected_uid=info.st_uid,
                expected_gid=info.st_gid,
            )
            held.replay()
            replacement = base / "replacement.json"
            replacement.write_bytes(raw)
            replacement.chmod(0o400)
            os.replace(replacement, path)
            with self.assertRaisesRegex(
                controller.RootFakeControllerError, "held authority changed",
            ):
                held.replay()
            held.close()
            link = base / "link.json"
            link.symlink_to(path)
            with self.assertRaises(controller.RootFakeControllerError):
                controller.open_authority(
                    link, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=info.st_uid,
                    expected_gid=info.st_gid,
                )
            hardlink = base / "hardlink.json"
            os.link(path, hardlink)
            with self.assertRaises(controller.RootFakeControllerError):
                controller.open_authority(
                    path, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=info.st_uid,
                    expected_gid=info.st_gid,
                )

    def test_real_runner_loader_installs_refusing_process_shim(self) -> None:
        module = controller.load_root_fake_runner(ROOT_FAKE_PATH.read_bytes())
        self.assertIsInstance(module.subprocess, types.SimpleNamespace)
        with self.assertRaisesRegex(
            controller.RootFakeControllerError, "subprocess is forbidden",
        ):
            module.subprocess.run(["forbidden"])
        self.assertTrue(callable(module.captured_main))

    def test_real_captured_main_abi_runs_in_process_with_memory_output(self) -> None:
        root_fake_path = ROOT_FAKE_PATH.resolve()
        raw = root_fake_path.read_bytes()
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            target = Path(value).resolve() / "captured-result.json"
            with mock.patch.object(
                controller, "ROOT_FAKE_RUNNER_PATH", root_fake_path,
            ), mock.patch.object(
                controller, "ROOT_FAKE_OUTPUT_PATH", target,
            ):
                held = controller.open_authority(
                    root_fake_path,
                    expected_sha256=controller.ROOT_FAKE_RUNNER_SHA256,
                    expected_size=controller.ROOT_FAKE_RUNNER_SIZE,
                    expected_mode=stat.S_IMODE(root_fake_path.stat().st_mode),
                    expected_uid=root_fake_path.stat().st_uid,
                    expected_gid=root_fake_path.stat().st_gid,
                    maximum_size=controller.MAX_SOURCE_SIZE,
                )
                try:
                    module = controller.load_root_fake_runner(raw)
                    spec = root_spec()
                    entry = root_entry(spec)
                    entry["captured_runner_identity"] = list(
                        held.held_identity
                    )
                    entry.pop("authority_digest")
                    entry["authority_digest"] = controller.object_digest(entry)
                    result = controller.run_isolated_root_fake(
                        module, spec=spec, entry=entry,
                    )
                finally:
                    held.close()
            self.assertEqual(result["status"], "PASS_CAPTURED_ROOT_FAKE_HOLD")
            self.assertFalse(os.path.lexists(target))

    def test_isolated_real_abi_call_once_restores_process_state(self) -> None:
        spec = root_spec()
        entry = root_entry(spec)
        result = root_result(spec, entry)
        calls = 0
        original_environment = dict(os.environ)
        original_argv = sys.argv
        original_main = sys.modules.get("__main__")

        class FakeModule:
            @staticmethod
            def create(*_args):
                raise AssertionError("controller did not install capture create")

            @staticmethod
            def captured_main(values):
                nonlocal calls
                calls += 1
                self.assertEqual(
                    values,
                    ["--captured-result", str(controller.ROOT_FAKE_OUTPUT_PATH)],
                )
                self.assertEqual(
                    set(os.environ),
                    {"CASE01_OBJECT_TRAJECTORY_CAPTURED_ROOT_ENTRY"},
                )
                FakeModule.create(controller.ROOT_FAKE_OUTPUT_PATH, result)
                print(controller.ROOT_MARKER + str(result["receipt_digest"]))
                return 0

        observed = controller.run_isolated_root_fake(
            FakeModule, spec=spec, entry=entry,
        )
        self.assertEqual(observed, result)
        self.assertEqual(calls, 1)
        self.assertEqual(dict(os.environ), original_environment)
        self.assertIs(sys.argv, original_argv)
        self.assertIs(sys.modules.get("__main__"), original_main)

    def test_isolated_refusal_calls_once_and_has_zero_final_output(self) -> None:
        calls = 0

        class RefusingModule:
            @staticmethod
            def create(*_args):
                raise AssertionError("create called")

            @staticmethod
            def captured_main(_values):
                nonlocal calls
                calls += 1
                raise RuntimeError("hostile refusal")

        with mock.patch.object(
            controller, "ROOT_FAKE_OUTPUT_PATH", Path("/tmp/never-created-root.json"),
        ), self.assertRaisesRegex(
            controller.RootFakeControllerError, "zero output and no retry",
        ):
            controller.run_isolated_root_fake(
                RefusingModule, spec=root_spec(), entry=root_entry(root_spec()),
            )
        self.assertEqual(calls, 1)
        self.assertFalse(os.path.lexists("/tmp/never-created-root.json"))

    def test_create_only_0400_self_digest_and_postflight(self) -> None:
        spec = root_spec()
        entry = root_entry(spec)
        result = root_result(spec, entry)
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            evidence = base / "evidence"
            evidence.mkdir(mode=0o700)
            target = evidence / "exact5_root_fake_receipt_v1.json"
            owner = evidence.stat()
            with mock.patch.object(controller, "PACKAGE_ROOT", base), \
                 mock.patch.object(controller, "ROOT_FAKE_OUTPUT_PATH", target), \
                 mock.patch.object(controller, "REMOTE_UID", owner.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", owner.st_gid):
                raw = controller.create_immutable_receipt(target, result)
                self.assertEqual(target.read_bytes(), raw)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
                held = controller.postflight_output(result, raw)
                held.close()
                with self.assertRaises(FileExistsError):
                    controller.create_immutable_receipt(target, result)

    def test_bad_prospective_digest_creates_zero_output(self) -> None:
        spec = root_spec()
        entry = root_entry(spec)
        result = root_result(spec, entry)
        result["receipt_digest"] = "0" * 64
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            evidence = base / "evidence"
            evidence.mkdir(mode=0o700)
            target = evidence / "exact5_root_fake_receipt_v1.json"
            owner = evidence.stat()
            with mock.patch.object(controller, "PACKAGE_ROOT", base), \
                 mock.patch.object(controller, "ROOT_FAKE_OUTPUT_PATH", target), \
                 mock.patch.object(controller, "REMOTE_UID", owner.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", owner.st_gid), \
                 self.assertRaisesRegex(
                     controller.RootFakeControllerError,
                     "prospective root-fake receipt differs",
                 ):
                controller.create_immutable_receipt(target, result)
            self.assertFalse(os.path.lexists(target))

    def _mocked_controller(self, *, refusal: bool = False):
        events: list[str] = []
        plan_raw = b'{"plan":true}\n'
        held_by_path = {
            controller.PACKAGE_PUBLICATION_RECEIPT_PATH: FakeHeld(
                controller.PACKAGE_PUBLICATION_RECEIPT_PATH, b"publication", events,
            ),
            controller.MATERIALIZATION_REPORT_PATH: FakeHeld(
                controller.MATERIALIZATION_REPORT_PATH, b"report", events,
            ),
            controller.STATIC_RECEIPT_PATH: FakeHeld(
                controller.STATIC_RECEIPT_PATH, b"static", events,
            ),
            controller.LAUNCHER_PATH: FakeHeld(
                controller.LAUNCHER_PATH, b"launcher", events,
            ),
            controller.LAUNCH_INPUT_PATH: FakeHeld(
                controller.LAUNCH_INPUT_PATH, b"input", events,
            ),
            controller.ROOT_FAKE_RUNNER_PATH: FakeHeld(
                controller.ROOT_FAKE_RUNNER_PATH, b"root-fake", events,
            ),
            controller.PLAN_PATH: FakeHeld(
                controller.PLAN_PATH, plan_raw, events,
            ),
        }

        def opened(path, **_kwargs):
            events.append("open:" + Path(path).name)
            return held_by_path[path]

        report = {
            "launch": {
                "input": {"sha256": "a" * 64, "size": 5},
                "release": {"identities": {"plan": {
                    "path": str(controller.PLAN_PATH),
                    "sha256": "b" * 64, "size": len(plan_raw),
                }}},
            },
        }
        spec = root_spec()
        entry = root_entry(spec)
        result = root_result(spec, entry)
        exact = [
            FakeHeld(Path(f"/exact/{index}"), b"exact", events)
            for index in range(25)
        ]
        output = FakeHeld(controller.ROOT_FAKE_OUTPUT_PATH, b"output", events)
        calls = 0

        def run(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            events.append("captured:call")
            if refusal:
                raise RuntimeError("refused")
            return result

        def create(*_args):
            events.append("create:output")
            return controller.canonical(result) + b"\n"

        def postflight(*_args):
            events.append("postflight:output")
            return output

        patches = (
            mock.patch.object(controller, "open_authority", side_effect=opened),
            mock.patch.object(controller, "validate_publication_receipt",
                              return_value={}),
            mock.patch.object(controller, "validate_materialization_report",
                              return_value=report),
            mock.patch.object(controller, "validate_static_receipt",
                              return_value={}),
            mock.patch.object(controller, "open_package_root",
                              return_value=FakeDirectory(events)),
            mock.patch.object(controller, "load_launcher",
                              return_value=types.SimpleNamespace()),
            mock.patch.object(controller, "load_root_fake_runner",
                              return_value=types.SimpleNamespace()),
            mock.patch.object(controller, "strict_json", return_value={}),
            mock.patch.object(controller, "build_root_spec_and_entry",
                              return_value=(spec, entry)),
            mock.patch.object(controller, "open_exact25", return_value=exact),
            mock.patch.object(controller, "validate_running_python",
                              return_value=None),
            mock.patch.object(controller, "require_fresh_output",
                              side_effect=lambda: events.append("fresh:output")),
            mock.patch.object(controller, "run_isolated_root_fake",
                              side_effect=run),
            mock.patch.object(controller, "create_immutable_receipt",
                              side_effect=create),
            mock.patch.object(controller, "postflight_output",
                              side_effect=postflight),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], patches[11], patches[12], patches[13], patches[14]:
            if refusal:
                with self.assertRaisesRegex(
                    controller.RootFakeControllerError,
                    "zero output and no retry",
                ):
                    controller.controller()
                return None, events, calls
            return controller.controller(), events, calls

    def test_controller_receipt_order_one_call_create_then_postflight(self) -> None:
        result, events, calls = self._mocked_controller()
        self.assertIsNotNone(result)
        self.assertEqual(calls, 1)
        publication = "open:" + controller.PACKAGE_PUBLICATION_RECEIPT_PATH.name
        report = "open:" + controller.MATERIALIZATION_REPORT_PATH.name
        static = "open:" + controller.STATIC_RECEIPT_PATH.name
        self.assertLess(events.index(publication), events.index(report))
        self.assertLess(events.index(report), events.index(static))
        self.assertLess(events.index(static), events.index("replay:package-root"))
        self.assertLess(
            events.index(
                "replay:" + controller.PACKAGE_PUBLICATION_RECEIPT_PATH.name
            ),
            events.index(
                "replay:" + controller.MATERIALIZATION_REPORT_PATH.name
            ),
        )
        self.assertLess(
            events.index("replay:" + controller.MATERIALIZATION_REPORT_PATH.name),
            events.index("replay:" + controller.STATIC_RECEIPT_PATH.name),
        )
        self.assertLess(
            events.index("replay:" + controller.STATIC_RECEIPT_PATH.name),
            events.index("replay:package-root"),
        )
        self.assertLess(events.index("captured:call"), events.index("create:output"))
        self.assertLess(events.index("create:output"),
                        events.index("postflight:output"))
        self.assertEqual(events.count("captured:call"), 1)
        self.assertEqual(events.count("create:output"), 1)

    def test_controller_refusal_calls_once_and_creates_zero_output(self) -> None:
        result, events, calls = self._mocked_controller(refusal=True)
        self.assertIsNone(result)
        self.assertEqual(calls, 1)
        self.assertNotIn("create:output", events)
        self.assertNotIn("postflight:output", events)


if __name__ == "__main__":
    unittest.main()
