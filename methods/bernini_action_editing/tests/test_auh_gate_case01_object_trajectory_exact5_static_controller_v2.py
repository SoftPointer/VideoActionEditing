#!/usr/bin/env python3
"""Hostile tests for the receipt-first exact5 static HOLD controller."""

from __future__ import annotations

import ast
import copy
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
    "scripts/auh_gate_case01_object_trajectory_exact5_static_once_v2.HOLD.py"
)
READY_PATH = METHOD_ROOT / (
    "scripts/auh_gate_case01_object_trajectory_exact5_static_once_v2.READY.py"
)
STATIC_PROBE_PATH = (
    METHOD_ROOT / "case01_object_trajectory_exact5_static_probe_v1.py"
)
LAUNCHER_PATH = (
    METHOD_ROOT / "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"
)
STATIC_PROBE_SHA256 = (
    "071256da47635fc3481f51b48e7e5eddddc963a5345b1dda405473744d2c01a9"
)
STATIC_PROBE_SIZE = 5_887
LAUNCHER_SHA256 = (
    "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f"
)
LAUNCHER_SIZE = 27_492


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> types.ModuleType:
    name = "_test_static_hold_controller_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load(CONTROLLER_PATH)


def static_result(input_sha256: str = "1" * 64) -> dict[str, object]:
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
        "input_sha256": input_sha256,
        "launcher_sha256": controller.LAUNCHER_SHA256,
    }
    value["receipt_digest"] = controller.object_digest(value)
    return value


def reseal_report(report: dict[str, object]) -> None:
    launch = report["launch"]
    release = launch["release"]
    release.pop("release_digest", None)
    release["release_digest"] = controller.object_digest(release)
    launch.pop("receipt_digest", None)
    launch["receipt_digest"] = controller.object_digest(launch)
    report.pop("receipt_digest", None)
    report["receipt_digest"] = controller.object_digest(report)


def launch_closure_fixture():
    launcher = controller.load_launcher(LAUNCHER_PATH.read_bytes())
    identities: dict[str, dict[str, object]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    method_root = (
        controller.PACKAGE_ROOT / "release/methods/bernini_action_editing"
    )
    for index, role in enumerate(launcher.IDENTITY_ROLES):
        if role in launcher.METHOD_ROLE_BASENAMES:
            path = method_root / launcher.METHOD_ROLE_BASENAMES[role]
        elif role == "plan":
            path = controller.PLAN_PATH
        else:
            path = Path("/external") / f"{index:02d}-{role}"
        digest = launcher.EXPECTED_STATIC_SHA256.get(role, f"{index + 1:064x}")
        row = {"path": str(path), "sha256": digest, "size": index + 101}
        identities[role] = row
        if role in launcher.METHOD_ROLE_BASENAMES:
            relative = str(path.relative_to(controller.PACKAGE_ROOT))
            artifacts[relative] = {"sha256": digest, "size": index + 101}
    launcher_relative = (
        "release/methods/bernini_action_editing/"
        "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"
    )
    artifacts[launcher_relative] = {
        "sha256": controller.LAUNCHER_SHA256,
        "size": controller.LAUNCHER_SIZE,
    }
    launch_input = {
        "schema_version": launcher.INPUT_SCHEMA,
        "entry_mode": "trusted_stdin",
        "campaign_mode": controller.CAMPAIGN,
        "holder_job_id": controller.JOB_ID,
        "expected_node": controller.NODE,
        "expected_allocation_gpu_count": 8,
        "identities": copy.deepcopy(identities),
        "output_report": str(controller.OUTPUT_REPORT_PATH),
        "runner_attestation": str(controller.RUNNER_ATTESTATION_PATH),
        "model_root": "/external/model",
        "bernini_root": "/external/bernini",
        "veomni_root": "/external/veomni",
        "authority_root": str(controller.AUTHORITY_ROOT),
        "rank_cache_root": str(controller.RANK_CACHE_ROOT),
    }
    report: dict[str, object] = {
        "plan": {
            "path": identities["plan"]["path"],
            "sha256": identities["plan"]["sha256"],
            "plan_digest": "f" * 64,
        },
        "artifacts": artifacts,
        "launch": {
            "input": {"path": str(controller.LAUNCH_INPUT_PATH)},
            "release": {
                "identity_roles": list(launcher.IDENTITY_ROLES),
                "identities": copy.deepcopy(identities),
            },
        },
    }
    reseal_report(report)
    return report, launch_input, launcher


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


class StaticControllerV2Test(unittest.TestCase):
    def test_hold_gate_precedes_argv_path_open_mkdir_and_controller(self) -> None:
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("called")
            raise AssertionError("HOLD crossed its first gate")

        stderr = io.StringIO()
        with mock.patch.object(controller, "blocked_dynamic_pins", forbidden), \
             mock.patch.object(controller, "controller", forbidden), \
             mock.patch.object(controller.os, "open", forbidden), \
             mock.patch.object(controller.os, "lstat", forbidden), \
             mock.patch.object(controller.os, "mkdir", forbidden), \
             redirect_stderr(stderr):
            result = controller.main(ExplosiveArgv())
        self.assertEqual(result, 88)
        self.assertEqual(touched, [])
        self.assertIn("HOLD", stderr.getvalue())

    def test_final_package_pins_are_exact_and_unblocked(self) -> None:
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
        self.assertEqual(changed, [28])
        self.assertEqual(
            hold_lines[28],
            b'CONTROLLER_STATE = "HOLD_PENDING_PACKAGE_PUBLICATION_AND_ROOT_PINS"\n',
        )
        self.assertEqual(
            ready_lines[28],
            b'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_STATIC_PROBE"\n',
        )
        ready = load(READY_PATH)
        self.assertEqual(ready.blocked_dynamic_pins(), ())
        self.assertEqual(ready.CONTROLLER_STATE, ready.READY_STATE)
        expected = static_result()
        stdout = io.StringIO()
        called = mock.Mock(return_value=expected)
        with mock.patch.object(ready, "controller", called), \
             redirect_stdout(stdout):
            result = ready.main(["--execute", ready.authorization_token()])
        self.assertEqual(result, 0)
        called.assert_called_once_with()
        self.assertEqual(
            json.loads(stdout.getvalue()), expected,
        )

    def test_static_source_pins_single_probe_and_no_process_transport(self) -> None:
        raw = CONTROLLER_PATH.read_bytes()
        source = raw.decode("utf-8")
        tree = ast.parse(raw, filename=str(CONTROLLER_PATH))
        functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        main = functions["main"]
        self.assertIsInstance(main.body[0], ast.If)
        first_names = {
            node.id for node in ast.walk(main.body[0].test)
            if isinstance(node, ast.Name)
        }
        self.assertEqual(first_names, {"CONTROLLER_STATE", "READY_STATE"})
        probe_calls = [
            node for node in ast.walk(functions["controller"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "probe"
        ]
        self.assertEqual(len(probe_calls), 1)
        imports = {
            alias.name.split(".")[0]
            for node in tree.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue(
            {"subprocess", "socket", "tempfile"}.isdisjoint(imports)
        )
        self.assertNotIn("/usr/bin/ssh", source)
        self.assertNotIn("/usr/bin/srun", source)
        self.assertNotIn("os.rename(", source)
        self.assertNotIn("os.replace(", source)
        self.assertNotIn("os.mkdir(", source)
        self.assertLess(
            source.index("publication_authority = open_authority("),
            source.index("package_root = open_package_root("),
        )
        controller_body = source[source.index("def controller()") :]
        self.assertLess(
            controller_body.index("launch_input_value = strict_json("),
            controller_body.index("validate_launch_identity_closure("),
        )
        self.assertLess(
            controller_body.index("validate_launch_identity_closure("),
            controller_body.index("probed = probe_module.probe("),
        )
        self.assertLess(
            source.index("probed = probe_module.probe("),
            source.index("output_raw = create_immutable_receipt("),
        )
        static_raw = STATIC_PROBE_PATH.read_bytes()
        launcher_raw = LAUNCHER_PATH.read_bytes()
        self.assertEqual(
            (sha256(static_raw), len(static_raw)),
            (STATIC_PROBE_SHA256, STATIC_PROBE_SIZE),
        )
        self.assertEqual(
            (sha256(launcher_raw), len(launcher_raw)),
            (LAUNCHER_SHA256, LAUNCHER_SIZE),
        )
        for optimize in (0, 2):
            self.assertIsNotNone(
                compile(raw, str(CONTROLLER_PATH), "exec", optimize=optimize)
            )
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertNotIn(b"\r", raw)

    def test_strict_json_rejects_duplicates_nan_and_noncanonical_bytes(self) -> None:
        self.assertEqual(
            controller.strict_json(b'{"a":1,"b":2}\n', label="valid"),
            {"a": 1, "b": 2},
        )
        for raw in (
            b'{"a":1,"a":2}\n', b'{"b":2, "a":1}\n',
            b'{"a":NaN}\n', b'{"a":1}',
        ):
            with self.subTest(raw=raw), self.assertRaises(
                controller.StaticControllerError
            ):
                controller.strict_json(raw, label="hostile")

    def test_pinned_order_and_held_input_identity_closure_passes(self) -> None:
        report, launch_input, launcher = launch_closure_fixture()
        raw = controller.canonical(launch_input) + b"\n"
        parsed = controller.strict_json(raw, label="held launch input")
        controller.validate_launch_identity_closure(report, parsed, launcher)

    def test_resealed_report_input_mismatch_and_role_reorder_fail_zero_output(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            output = Path(value) / "static-output.json"
            for hostile_name in ("identity-mismatch", "role-reorder"):
                report, launch_input, launcher = launch_closure_fixture()
                if hostile_name == "identity-mismatch":
                    report["launch"]["release"]["identities"]["runner"][
                        "sha256"
                    ] = "0" * 64
                else:
                    roles = report["launch"]["release"]["identity_roles"]
                    roles[0], roles[1] = roles[1], roles[0]
                reseal_report(report)
                with self.subTest(hostile=hostile_name), self.assertRaisesRegex(
                    controller.StaticControllerError,
                    "held launch input/report role closure differs",
                ):
                    controller.validate_launch_identity_closure(
                        report, launch_input, launcher,
                    )
                self.assertFalse(os.path.lexists(output))

    def test_resealed_internal_paths_outside_package_fail_zero_output(self) -> None:
        cases = ("method", "plan", "launcher", "input")
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            output = Path(value) / "static-output.json"
            for hostile_name in cases:
                report, launch_input, launcher = launch_closure_fixture()
                if hostile_name == "method":
                    outside = "/tmp/outside-runner.py"
                    report["launch"]["release"]["identities"]["runner"][
                        "path"
                    ] = outside
                    launch_input["identities"]["runner"]["path"] = outside
                elif hostile_name == "plan":
                    outside = "/tmp/outside-plan.json"
                    report["launch"]["release"]["identities"]["plan"][
                        "path"
                    ] = outside
                    launch_input["identities"]["plan"]["path"] = outside
                    report["plan"]["path"] = outside
                elif hostile_name == "launcher":
                    relative = (
                        "release/methods/bernini_action_editing/"
                        "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"
                    )
                    row = report["artifacts"].pop(relative)
                    report["artifacts"]["../outside-launcher.py"] = row
                else:
                    report["launch"]["input"]["path"] = (
                        "/tmp/outside-launch-input.json"
                    )
                reseal_report(report)
                with self.subTest(hostile=hostile_name), self.assertRaises(
                    controller.StaticControllerError,
                ):
                    controller.validate_launch_identity_closure(
                        report, launch_input, launcher,
                    )
                self.assertFalse(os.path.lexists(output))

    def test_held_authority_rejects_links_and_named_inode_replacement(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            path = base / "authority.json"
            raw = b'{"authority":true}\n'
            path.write_bytes(raw); path.chmod(0o400)
            info = path.stat()
            held = controller.open_authority(
                path, expected_sha256=sha256(raw), expected_size=len(raw),
                expected_mode=0o400, expected_uid=info.st_uid,
                expected_gid=info.st_gid,
            )
            held.replay()
            replacement = base / "replacement.json"
            replacement.write_bytes(raw); replacement.chmod(0o400)
            os.replace(replacement, path)
            with self.assertRaisesRegex(
                controller.StaticControllerError, "held authority changed",
            ):
                held.replay()
            held.close()

            link = base / "link.json"
            link.symlink_to(path)
            with self.assertRaises(controller.StaticControllerError):
                controller.open_authority(
                    link, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=info.st_uid,
                    expected_gid=info.st_gid,
                )
            hardlink = base / "hardlink.json"
            os.link(path, hardlink)
            with self.assertRaises(controller.StaticControllerError):
                controller.open_authority(
                    path, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=info.st_uid,
                    expected_gid=info.st_gid,
                )

    def test_create_only_0400_output_self_digest_and_postflight(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            evidence = base / "evidence"
            evidence.mkdir(mode=0o700)
            target = evidence / "exact5_static_probe_receipt_v1.json"
            owner = evidence.stat()
            result = static_result()
            with mock.patch.object(controller, "PACKAGE_ROOT", base), \
                 mock.patch.object(controller, "STATIC_OUTPUT_PATH", target), \
                 mock.patch.object(controller, "REMOTE_UID", owner.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", owner.st_gid):
                raw = controller.create_immutable_receipt(target, result)
                self.assertEqual(target.read_bytes(), raw)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
                held = controller.postflight_output(result, raw)
                held.close()
                with self.assertRaises(FileExistsError):
                    controller.create_immutable_receipt(target, result)
            self.assertEqual(target.read_bytes(), raw)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)

    def test_bad_prospective_digest_creates_zero_output(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            evidence = base / "evidence"
            evidence.mkdir(mode=0o700)
            target = evidence / "exact5_static_probe_receipt_v1.json"
            owner = evidence.stat()
            result = static_result()
            result["receipt_digest"] = "0" * 64
            with mock.patch.object(controller, "PACKAGE_ROOT", base), \
                 mock.patch.object(controller, "STATIC_OUTPUT_PATH", target), \
                 mock.patch.object(controller, "REMOTE_UID", owner.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", owner.st_gid), \
                 self.assertRaisesRegex(
                     controller.StaticControllerError,
                     "static probe result differs",
                 ):
                controller.create_immutable_receipt(target, result)
            self.assertFalse(os.path.lexists(target))

    def _run_mocked_controller(
        self, *, probe_error: Exception | None = None,
        closure_error: Exception | None = None,
    ):
        events: list[str] = []
        input_raw = b'{"launch":"input"}\n'
        publication = FakeHeld(
            controller.PACKAGE_PUBLICATION_RECEIPT_PATH, b"publication", events,
        )
        report_held = FakeHeld(
            controller.MATERIALIZATION_REPORT_PATH, b"report", events,
        )
        runtime = FakeHeld(controller.VACE_PYTHON, b"python", events)
        self_held = FakeHeld(CONTROLLER_PATH, b"controller", events)
        probe_held = FakeHeld(controller.STATIC_PROBE_PATH, b"probe", events)
        launcher = FakeHeld(controller.LAUNCHER_PATH, b"launcher", events)
        launch_input = FakeHeld(controller.LAUNCH_INPUT_PATH, input_raw, events)
        output = FakeHeld(controller.STATIC_OUTPUT_PATH, b"output", events)
        directory = FakeDirectory(events)
        report = {
            "launch": {"input": {
                "path": str(controller.LAUNCH_INPUT_PATH),
                "sha256": sha256(input_raw), "size": len(input_raw),
                "mode": 0o444, "nlink": 1,
            }}
        }
        held_by_path = {
            controller.PACKAGE_PUBLICATION_RECEIPT_PATH: publication,
            controller.MATERIALIZATION_REPORT_PATH: report_held,
            controller.STATIC_PROBE_PATH: probe_held,
            controller.LAUNCHER_PATH: launcher,
            controller.LAUNCH_INPUT_PATH: launch_input,
        }

        def open_side_effect(path, **_kwargs):
            events.append("open:" + Path(path).name)
            return held_by_path[path]

        calls = 0

        class ProbeModule:
            @staticmethod
            def probe(*_args):
                nonlocal calls
                calls += 1
                events.append("probe:call")
                if probe_error is not None:
                    raise probe_error
                return static_result(sha256(input_raw))

        class LauncherModule:
            pass

        def closure(*_args):
            events.append("validate:launch-closure")
            if closure_error is not None:
                raise closure_error

        def fresh() -> None:
            events.append("fresh:output")

        def create(path, result):
            events.append("create:output")
            return controller.canonical(result) + b"\n"

        def postflight(result, raw):
            events.append("postflight:output")
            self.assertEqual(raw, controller.canonical(result) + b"\n")
            return output

        patches = (
            mock.patch.object(
                controller, "open_authority", side_effect=open_side_effect,
            ),
            mock.patch.object(
                controller, "validate_publication_receipt",
                return_value={"status": "PUBLISHED_RECEIPT_GATED"},
            ),
            mock.patch.object(
                controller, "open_package_root", return_value=directory,
            ),
            mock.patch.object(
                controller, "validate_materialization_report",
                return_value=report,
            ),
            mock.patch.object(
                controller, "open_runtime_authority", return_value=runtime,
            ),
            mock.patch.object(
                controller, "open_self_authority", return_value=self_held,
            ),
            mock.patch.object(
                controller, "load_static_probe", return_value=ProbeModule,
            ),
            mock.patch.object(
                controller, "load_launcher", return_value=LauncherModule,
            ),
            mock.patch.object(
                controller, "validate_launch_identity_closure",
                side_effect=closure,
            ),
            mock.patch.object(
                controller, "require_fresh_output", side_effect=fresh,
            ),
            mock.patch.object(
                controller, "create_immutable_receipt", side_effect=create,
            ),
            mock.patch.object(
                controller, "postflight_output", side_effect=postflight,
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], patches[11]:
            if probe_error is None and closure_error is None:
                result = controller.controller()
                return result, events, calls
            if closure_error is not None:
                with self.assertRaisesRegex(
                    controller.StaticControllerError, "hostile closure",
                ):
                    controller.controller()
                return None, events, calls
            with self.assertRaisesRegex(
                controller.StaticControllerError, "zero output and no retry",
            ):
                controller.controller()
        return None, events, calls

    def test_receipt_first_exactly_once_probe_then_create_and_postflight(self) -> None:
        result, events, calls = self._run_mocked_controller()
        self.assertEqual(calls, 1)
        self.assertFalse(result["launch_allowed"])
        self.assertLess(
            events.index(
                "open:" + controller.PACKAGE_PUBLICATION_RECEIPT_PATH.name
            ),
            events.index("replay:package-root"),
        )
        self.assertLess(events.index("probe:call"), events.index("create:output"))
        self.assertLess(
            events.index("create:output"), events.index("postflight:output")
        )
        self.assertEqual(events.count("probe:call"), 1)
        self.assertEqual(events.count("create:output"), 1)
        self.assertEqual(events.count("postflight:output"), 1)

    def test_probe_failure_calls_once_and_creates_zero_output(self) -> None:
        result, events, calls = self._run_mocked_controller(
            probe_error=RuntimeError("hostile probe refusal")
        )
        self.assertIsNone(result)
        self.assertEqual(calls, 1)
        self.assertEqual(events.count("probe:call"), 1)
        self.assertNotIn("create:output", events)
        self.assertNotIn("postflight:output", events)

    def test_launch_closure_failure_precedes_probe_and_creates_zero_output(self) -> None:
        result, events, calls = self._run_mocked_controller(
            closure_error=controller.StaticControllerError(
                "hostile closure mismatch"
            )
        )
        self.assertIsNone(result)
        self.assertEqual(calls, 0)
        self.assertEqual(events.count("validate:launch-closure"), 1)
        self.assertNotIn("probe:call", events)
        self.assertNotIn("create:output", events)
        self.assertNotIn("postflight:output", events)


if __name__ == "__main__":
    unittest.main()
