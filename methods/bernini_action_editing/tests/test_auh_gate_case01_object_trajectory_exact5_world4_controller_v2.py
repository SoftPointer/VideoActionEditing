#!/usr/bin/env python3
"""Hostile tests for the receipt-first package world4 HOLD controller."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = METHOD_ROOT / (
    "scripts/auh_gate_case01_object_trajectory_exact5_world4_once_v2.HOLD.py"
)
READY_CONTROLLER_PATH = METHOD_ROOT / (
    "scripts/auh_gate_case01_object_trajectory_exact5_world4_once_v2.READY.py"
)
ENGINE_PATH = METHOD_ROOT / (
    "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py"
)
ROOT_FAKE_CONTROLLER_PATH = METHOD_ROOT / (
    "scripts/auh_gate_case01_object_trajectory_exact5_root_fake_once_v2.READY.py"
)
LOCAL_RECEIPT_ROOT = METHOD_ROOT.parents[1] / (
    "artifacts/case01_object_trajectory_exact5_r64_canary_v1"
)
HOLD_SHA256 = "4215b495d400b9c2af565f07ad776c01093f09aaefc2357404dd7e73102ac9e4"
HOLD_SIZE = 57_483
READY_SHA256 = "7484c02823194f0dee96dde53f06d2469ee6efc6e7ba43126dbfa0a1211fa3c5"
READY_SIZE = 57_484
HOLD_STATE = "HOLD_PENDING_PACKAGE_STATIC_ROOTFAKE_AND_ROOT_PINS"
READY_STATE = "READY_EXPLICIT_SINGLE_SRUN_PACKAGE_WORLD4_ADMISSION"
HOLD_ASSIGNMENT = f'CONTROLLER_STATE = "{HOLD_STATE}"\n'.encode("ascii")
READY_ASSIGNMENT = f'CONTROLLER_STATE = "{READY_STATE}"\n'.encode("ascii")


def load(path: Path) -> types.ModuleType:
    name = "_test_package_world4_hold_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load(CONTROLLER_PATH)


class ExplosiveArgv:
    def __iter__(self):
        raise AssertionError("HOLD iterated argv")


class FakeAuthority:
    def __init__(self, path: Path, raw: bytes, events: list[str]) -> None:
        self.path = path
        self.raw = raw
        self.events = events
        self.descriptor = 17

    def replay(self) -> None:
        self.events.append("authority:" + self.path.name)

    def row(self):
        return {"path": str(self.path), "sha256": hashlib.sha256(self.raw).hexdigest(),
                "size": len(self.raw)}

    def close(self) -> None:
        self.events.append("close:" + self.path.name)


class FakeRoot:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.held_identity = tuple(range(11))

    def replay(self) -> None:
        self.events.append("root")

    def close(self) -> None:
        self.events.append("close:root")


def canonical_value(value: dict[str, object]) -> tuple[bytes, str]:
    unsigned = dict(value)
    digest = controller.object_digest(unsigned)
    value["receipt_digest"] = digest
    return controller.canonical(value) + b"\n", digest


def static_fixture() -> tuple[dict[str, object], dict[str, object]]:
    report = {"launch": {"input": {"sha256": "a" * 64}}}
    value: dict[str, object] = {
        "schema_version": controller.STATIC_SCHEMA,
        "status": "ADMITTED_STATIC_HOLD_ONLY", "launch_allowed": False,
        "blocked_roles": [], "final_source_pins_complete": True,
        "exact_identity_count": 25,
        "task_ids": list(controller.TASK_IDS),
        "arm_order": list(controller.ARM_ORDER),
        "all_tasks_hard1_every_step": True,
        "null_arms_have_no_external_conditions": True,
        "route_and_active_arms_have_external_conditions": True,
        "torch_imported": False, "renderer_imported": False,
        "publication_performed": False, "input_sha256": "a" * 64,
        "launcher_sha256":
        "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f",
    }
    return report, value


def root_fake_fixture() -> tuple[dict[str, object], dict[str, object]]:
    identities = {
        role: {"path": f"/authority/{role}", "sha256": f"{index + 1:064x}",
               "size": index + 1}
        for index, role in enumerate(controller.IDENTITY_ROLES)
    }
    report = {
        "launch": {
            "input": {"sha256": "b" * 64},
            "release": {"identities": identities},
        },
        "plan": {"sha256": "c" * 64},
    }
    value: dict[str, object] = {
        "schema_version": controller.ROOT_FAKE_SCHEMA,
        "status": "PASS_CAPTURED_ROOT_FAKE_HOLD",
        "campaign_mode": controller.CAMPAIGN, "launch_allowed": False,
        "exact_identity_count": 25,
        "identity_roles": list(controller.IDENTITY_ROLES),
        "task_ids": list(controller.TASK_IDS),
        "arm_order": list(controller.ARM_ORDER),
        "release_digest": "d" * 64,
        "identity_set_digest": controller.object_digest(identities),
        "launch_input_sha256": "b" * 64,
        "entry_authority_digest": "e" * 64,
        "plan_sha256": "c" * 64,
        "production_runner_sha256":
        "e47b81643c1d17e5099a9b33f16ca75521001ad52d2df2305b46b7e8c4d5ac4c",
        "captured_runner_sha256":
        "0d73fdaa4a4f1817f572eea471661850098ffe5aa54f54a9927c37a7e3f2a872",
        "all_exact25_named_identities_replayed": True,
        "captured_runner_outside_exact25": True,
        "captured_runner_bytes_compiled": True,
        "torch_imported": False, "renderer_imported": False,
        "publication_performed": False,
    }
    return report, value


class PackageWorld4ControllerV2Test(unittest.TestCase):
    def test_ready_is_frozen_exactly_one_state_assignment_from_hold(self) -> None:
        hold = CONTROLLER_PATH.read_bytes()
        ready = READY_CONTROLLER_PATH.read_bytes()
        self.assertEqual((hashlib.sha256(hold).hexdigest(), len(hold)),
                         (HOLD_SHA256, HOLD_SIZE))
        self.assertEqual((hashlib.sha256(ready).hexdigest(), len(ready)),
                         (READY_SHA256, READY_SIZE))
        self.assertEqual(hold.count(HOLD_ASSIGNMENT), 1)
        self.assertNotIn(READY_ASSIGNMENT, hold)
        self.assertNotIn(HOLD_ASSIGNMENT, ready)
        self.assertEqual(ready.count(READY_ASSIGNMENT), 1)
        self.assertEqual(ready, hold.replace(HOLD_ASSIGNMENT, READY_ASSIGNMENT, 1))
        differing = [
            (before, after)
            for before, after in zip(
                hold.splitlines(keepends=True), ready.splitlines(keepends=True),
            )
            if before != after
        ]
        self.assertEqual(differing, [(HOLD_ASSIGNMENT, READY_ASSIGNMENT)])
        self.assertEqual(len(hold.splitlines()), len(ready.splitlines()))
        self.assertTrue(ready.endswith(b"\n"))
        self.assertFalse(ready.endswith(b"\n\n"))
        self.assertNotIn(b"\r", ready)
        ready_module = load(READY_CONTROLLER_PATH)
        self.assertEqual(controller.CONTROLLER_STATE, HOLD_STATE)
        self.assertEqual(ready_module.CONTROLLER_STATE, READY_STATE)
        self.assertEqual(ready_module.CONTROLLER_STATE, ready_module.READY_STATE)
        self.assertEqual(ready_module.blocked_dynamic_pins(), ())

    def test_hold_gate_precedes_argv_pins_paths_and_controller(self) -> None:
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("called")
            raise AssertionError("HOLD crossed its first gate")

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

    def test_all_thirteen_final_dynamic_pins_are_valid_and_unblocked(self) -> None:
        expected = {
            "package_publication_receipt_sha256",
            "package_publication_receipt_size",
            "package_publication_receipt_digest",
            "materialization_report_sha256", "materialization_report_size",
            "materialization_report_digest", "static_receipt_sha256",
            "static_receipt_size", "static_receipt_digest",
            "root_fake_receipt_sha256", "root_fake_receipt_size",
            "root_fake_receipt_digest", "package_root_identity",
        }
        self.assertEqual(set(controller.dynamic_pin_values()), expected)
        self.assertEqual(controller.blocked_dynamic_pins(), ())
        self.assertEqual(controller.dynamic_pin_values(), {
            "package_publication_receipt_sha256":
            "b3766694f24ead6d7da04e5a1da077de69a9dbbf06df8f06ff0c9db77d84c533",
            "package_publication_receipt_size": 2209,
            "package_publication_receipt_digest":
            "5cab7d2db0079d4b6960273e681c20b60941b892c3a42bfdbd70be819d991cb9",
            "materialization_report_sha256":
            "e1e4d7ae266828f27f77f39528672cd7ccae9aa067fdee291d4e5e32f9a9bf2f",
            "materialization_report_size": 21743,
            "materialization_report_digest":
            "99ba2595bde82371257a46b08ef55f77f54cb5b86877aa791daf6976237868c4",
            "static_receipt_sha256":
            "3e65f4342f33a0d4264fa7f09759bad3aa2f4c4622a6965db675f2c551fb07b8",
            "static_receipt_size": 1035,
            "static_receipt_digest":
            "7ed16825624ca99dc7f2cbbea3c9a5a991122108aff4867796a3ac01456ab6be",
            "root_fake_receipt_sha256":
            "af4cb28c23bc9e7a8355133f2068d02af5f97eda16083fa8b591e5131062f619",
            "root_fake_receipt_size": 1975,
            "root_fake_receipt_digest":
            "4a65b5dab48904fced093fd0bff0c16a50c13b5caa30b6a70dc4e4ae9c6b170a",
            "package_root_identity": [
                48, 12038280342419913116, 2012, 2000, 16832, 2, 0,
                4096, 0, 1787357728317453482, 1787357728652385810,
            ],
        })
        called = mock.Mock(side_effect=AssertionError("controller called"))
        with mock.patch.object(controller, "CONTROLLER_STATE", controller.READY_STATE), \
             mock.patch.object(controller, "controller", called):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = controller.main(["--execute", "wrong"])
        self.assertEqual(result, 96)
        called.assert_not_called()
        self.assertIn("authorization argv differs", stderr.getvalue())

    def test_root_fake_path_matches_final_controller_and_local_artifact(self) -> None:
        root_fake_controller = load(ROOT_FAKE_CONTROLLER_PATH)
        local = LOCAL_RECEIPT_ROOT / (
            "exact5_root_fake_runner_probe_receipt_v1.json"
        )
        self.assertEqual(
            controller.ROOT_FAKE_RECEIPT_PATH,
            root_fake_controller.ROOT_FAKE_OUTPUT_PATH,
        )
        self.assertEqual(controller.ROOT_FAKE_RECEIPT_PATH.name, local.name)
        self.assertTrue(local.is_file())

    def test_real_local_four_receipt_chain_validates_in_order(self) -> None:
        local_paths = {
            "publication": LOCAL_RECEIPT_ROOT / (
                "bernini_case01_object_trajectory_exact5_r64_canary_v1."
                "publication_receipt_v2.json"
            ),
            "materialization": LOCAL_RECEIPT_ROOT / (
                "package_materialization_receipt_v1.json"
            ),
            "static": LOCAL_RECEIPT_ROOT / "exact5_static_probe_receipt_v1.json",
            "root_fake": LOCAL_RECEIPT_ROOT / (
                "exact5_root_fake_runner_probe_receipt_v1.json"
            ),
        }
        raws = {key: path.read_bytes() for key, path in local_paths.items()}
        for key, prefix in (
            ("publication", "PACKAGE_PUBLICATION_RECEIPT"),
            ("materialization", "MATERIALIZATION_REPORT"),
            ("static", "STATIC_RECEIPT"),
            ("root_fake", "ROOT_FAKE_RECEIPT"),
        ):
            self.assertEqual(
                hashlib.sha256(raws[key]).hexdigest(),
                getattr(controller, prefix + "_SHA256"),
            )
            self.assertEqual(len(raws[key]), getattr(controller, prefix + "_SIZE"))
        publication_json = json.loads(raws["publication"])
        anchor = publication_json["receipt_inode_anchor"]
        remote_info = types.SimpleNamespace(
            st_dev=anchor[0], st_ino=anchor[1], st_uid=anchor[2],
            st_gid=anchor[3], st_mode=anchor[4] | controller.RECEIPT_MODE,
        )
        held = {
            key: types.SimpleNamespace(raw=raw, descriptor=71 + index)
            for index, (key, raw) in enumerate(raws.items())
        }
        events: list[str] = []
        with mock.patch.object(controller.os, "fstat", return_value=remote_info):
            publication = controller.validate_publication_receipt(
                held["publication"]
            )
            events.append("publication")
        report = controller.validate_materialization_report(
            held["materialization"], publication,
        )
        events.append("materialization")
        controller.validate_static_receipt(held["static"], report)
        events.append("static")
        controller.validate_root_fake_receipt(held["root_fake"], report)
        events.append("root_fake")
        self.assertEqual(
            events, ["publication", "materialization", "static", "root_fake"],
        )

    def test_source_proves_receipt_first_attempt_before_single_srun(self) -> None:
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
            {node.id for node in ast.walk(main.body[0].test)
             if isinstance(node, ast.Name)},
            {"CONTROLLER_STATE", "READY_STATE"},
        )
        body = source[source.index("def controller()") : source.index("def main(")]
        gate = body.index("package_gate = open_package_gate()")
        engine = body.index("engine, engine_authority = load_engine()")
        fresh = body.index("_fresh_outputs(engine)")
        attempt = body.index("attempt_raw = engine._create_json(ATTEMPT_PATH, attempt)")
        srun = body.index("returncode, stdout_raw, stderr_raw = engine._run_single_srun(")
        self.assertLess(gate, engine)
        self.assertLess(engine, fresh)
        self.assertLess(fresh, attempt)
        self.assertLess(attempt, srun)
        calls = [
            node for node in ast.walk(functions["controller"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_single_srun"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("world4_cpu_admission_v1", source)
        self.assertNotIn("world4_receipt_v1.json", source)
        self.assertNotIn('"retry_allowed": True', source)

    def test_exact_package_project_and_runtime_authority_closure(self) -> None:
        self.assertEqual(
            set(controller.PACKAGE_PROJECT_AUTHORITIES),
            {"wrapper", "projection", "scaffold_module", "scaffold", "world4"},
        )
        for role, row in controller.PACKAGE_PROJECT_AUTHORITIES.items():
            self.assertFalse(Path(row["relative"]).is_absolute(), role)
            self.assertNotIn("..", Path(row["relative"]).parts)
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(row["size"], 0)
        local_engine = load(ENGINE_PATH)
        self.assertEqual(controller.RUNTIME_AUTHORITIES, local_engine.RUNTIME_AUTHORITIES)
        self.assertEqual(controller.ENGINE_SHA256, hashlib.sha256(ENGINE_PATH.read_bytes()).hexdigest())
        self.assertEqual(controller.ENGINE_SIZE, ENGINE_PATH.stat().st_size)

    def test_configure_engine_substitutes_package_not_old_cpu_gate(self) -> None:
        engine = types.SimpleNamespace()
        sentinel = object()
        controller.configure_engine(engine, sentinel)
        self.assertEqual(engine.SOURCE_ROOT, controller.PACKAGE_ROOT)
        self.assertIs(engine.open_source_stage_gate, sentinel)
        self.assertEqual(engine.PROJECT_AUTHORITIES, controller.PACKAGE_PROJECT_AUTHORITIES)
        self.assertEqual(engine.WORLD4_RECEIPT_PATH, controller.WORLD4_RECEIPT_PATH)
        self.assertEqual(engine.ATTEMPT_PATH, controller.ATTEMPT_PATH)
        self.assertEqual(engine.STAGE_ROOT, controller.STAGE_ROOT)
        self.assertNotIn("cpu_admission", str(engine.WORLD4_RECEIPT_PATH))

    def test_cpu_only_single_srun_argv_has_bounded_contract(self) -> None:
        engine = load(ENGINE_PATH)
        controller.configure_engine(engine, lambda: None)
        evidence = {
            "schema_version": controller.SCHEMA + "-package-gate",
            "root": str(controller.PACKAGE_ROOT),
        }
        plan = engine.build_compute_plan(evidence)
        transport = base64_encode(engine.canonical(plan))
        argv = engine.build_srun_argv(transport)
        self.assertEqual(argv.count("/usr/bin/srun"), 1)
        self.assertIn("--jobid=143808", argv)
        self.assertIn("--nodelist=auh7-1b-gpu-292", argv)
        self.assertIn("--nodes=1", argv)
        self.assertIn("--ntasks=1", argv)
        self.assertIn("--cpus-per-task=16", argv)
        self.assertIn("--gres=none", argv)
        export = next(value for value in argv if value.startswith("--export="))
        for text in (
            "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=",
            "HIP_VISIBLE_DEVICES=", "ROCR_VISIBLE_DEVICES=-1",
        ):
            self.assertIn(text, export)
        self.assertEqual(plan["gpu_count"], 0)
        self.assertEqual(plan["per_scenario_timeout_seconds"], 30)
        self.assertEqual(len(plan["world4_scenarios"]), 7)
        self.assertFalse(plan["retry_allowed"])

    def test_strict_json_rejects_duplicate_noncanonical_and_nan(self) -> None:
        with self.assertRaises(controller.PackageWorld4Error):
            controller.strict_json(b'{"a":1,"a":2}\n', label="duplicate")
        with self.assertRaises(controller.PackageWorld4Error):
            controller.strict_json(b'{"b":2, "a":1}\n', label="spaced")
        with self.assertRaises(controller.PackageWorld4Error):
            controller.strict_json(b'{"a":NaN}\n', label="nan")

    def test_static_receipt_exact_pin_and_hostile_mutation(self) -> None:
        report, value = static_fixture()
        raw, digest = canonical_value(value)
        held = types.SimpleNamespace(raw=raw)
        patches = (
            mock.patch.object(controller, "STATIC_RECEIPT_SHA256", hashlib.sha256(raw).hexdigest()),
            mock.patch.object(controller, "STATIC_RECEIPT_SIZE", len(raw)),
            mock.patch.object(controller, "STATIC_RECEIPT_DIGEST", digest),
        )
        with patches[0], patches[1], patches[2]:
            self.assertEqual(controller.validate_static_receipt(held, report), value)
            hostile = dict(value)
            hostile["torch_imported"] = True
            hostile.pop("receipt_digest")
            hostile["receipt_digest"] = controller.object_digest(hostile)
            held.raw = controller.canonical(hostile) + b"\n"
            with self.assertRaises(controller.PackageWorld4Error):
                controller.validate_static_receipt(held, report)

    def test_root_fake_receipt_exact_pin_and_hostile_mutation(self) -> None:
        report, value = root_fake_fixture()
        raw, digest = canonical_value(value)
        held = types.SimpleNamespace(raw=raw)
        with mock.patch.object(controller, "ROOT_FAKE_RECEIPT_SHA256", hashlib.sha256(raw).hexdigest()), \
             mock.patch.object(controller, "ROOT_FAKE_RECEIPT_SIZE", len(raw)), \
             mock.patch.object(controller, "ROOT_FAKE_RECEIPT_DIGEST", digest):
            self.assertEqual(controller.validate_root_fake_receipt(held, report), value)
            hostile = copy.deepcopy(value)
            hostile["plan_sha256"] = "0" * 64
            hostile.pop("receipt_digest")
            hostile["receipt_digest"] = controller.object_digest(hostile)
            held.raw = controller.canonical(hostile) + b"\n"
            with self.assertRaises(controller.PackageWorld4Error):
                controller.validate_root_fake_receipt(held, report)

    def test_gate_replays_all_receipts_before_root_and_projects(self) -> None:
        events: list[str] = []
        authorities = [
            FakeAuthority(Path(f"/receipt-{index}.json"), b"{}\n", events)
            for index in range(4)
        ]
        projects = {
            "wrapper": FakeAuthority(Path("/package/wrapper.py"), b"x", events),
            "world4": FakeAuthority(Path("/package/world4.py"), b"y", events),
        }
        values = {
            "publication": {"receipt_digest": "a" * 64},
            "materialization": {"receipt_digest": "b" * 64},
            "static": {"receipt_digest": "c" * 64},
            "root_fake": {"receipt_digest": "d" * 64},
        }
        gate = controller.HeldPackageGate(authorities, FakeRoot(events), values, projects)
        with mock.patch.object(controller, "validate_publication_receipt", return_value=values["publication"]), \
             mock.patch.object(controller, "validate_materialization_report", return_value=values["materialization"]), \
             mock.patch.object(controller, "validate_static_receipt", return_value=values["static"]), \
             mock.patch.object(controller, "validate_root_fake_receipt", return_value=values["root_fake"]):
            gate.replay()
        self.assertEqual(events[:4], [
            "authority:receipt-0.json", "authority:receipt-1.json",
            "authority:receipt-2.json", "authority:receipt-3.json",
        ])
        self.assertEqual(events[4], "root")
        self.assertTrue(all(event.startswith("authority:") for event in events[5:]))

    def test_open_gate_orders_four_receipts_root_then_project(self) -> None:
        events: list[str] = []
        values = {
            "publication": {"receipt_digest": "a" * 64},
            "materialization": {"receipt_digest": "b" * 64},
            "static": {"receipt_digest": "c" * 64},
            "root_fake": {"receipt_digest": "d" * 64},
        }

        def fake_open(path, **_kwargs):
            events.append("open:" + path.name)
            return FakeAuthority(path, b"{}\n", events)

        def fake_root(_identity):
            events.append("open:package-root")
            return FakeRoot(events)

        with mock.patch.object(controller, "open_authority", side_effect=fake_open), \
             mock.patch.object(controller, "open_package_root", side_effect=fake_root), \
             mock.patch.object(controller, "validate_publication_receipt", return_value=values["publication"]), \
             mock.patch.object(controller, "validate_materialization_report", return_value=values["materialization"]), \
             mock.patch.object(controller, "validate_static_receipt", return_value=values["static"]), \
             mock.patch.object(controller, "validate_root_fake_receipt", return_value=values["root_fake"]), \
             mock.patch.object(controller.HeldPackageGate, "replay"):
            gate = controller.open_package_gate()
        self.assertEqual(events[:5], [
            "open:" + controller.PACKAGE_PUBLICATION_RECEIPT_PATH.name,
            "open:" + controller.MATERIALIZATION_REPORT_PATH.name,
            "open:" + controller.STATIC_RECEIPT_PATH.name,
            "open:" + controller.ROOT_FAKE_RECEIPT_PATH.name,
            "open:package-root",
        ])
        self.assertEqual(len(events), 5 + len(controller.PACKAGE_PROJECT_AUTHORITIES))
        gate.close()

    def test_normal_and_optimized_compile_without_execution(self) -> None:
        raw = CONTROLLER_PATH.read_bytes()
        for path in (CONTROLLER_PATH, READY_CONTROLLER_PATH):
            candidate = path.read_bytes()
            for optimize in (0, 2):
                self.assertIsNotNone(
                    compile(candidate, str(path), "exec", optimize=optimize),
                )
        tree = ast.parse(raw, filename=str(CONTROLLER_PATH))
        guards = [
            node for node in tree.body if isinstance(node, ast.If)
            and any(isinstance(child, ast.Name) and child.id == "__name__"
                    for child in ast.walk(node.test))
        ]
        self.assertEqual(len(guards), 1)


def base64_encode(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii")


if __name__ == "__main__":
    unittest.main()
