#!/usr/bin/env python3
"""Hostile fail-closed tests for the fresh core-v4 exact6 package scaffold."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = METHOD_ROOT / (
    "tools/materialize_case01_object_trajectory_exact5_r64_overlay_package_v3.py"
)
BASE_MATERIALIZER_PATH = METHOD_ROOT / (
    "tools/materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py"
)
CONTROLLER_PATH = METHOD_ROOT / (
    "scripts/auh_materialize_case01_object_trajectory_exact5_r64_"
    "overlay_package_once_v3.HOLD.py"
)
READY_PATH = METHOD_ROOT / (
    "scripts/auh_materialize_case01_object_trajectory_exact5_r64_"
    "overlay_package_once_v3.READY.py"
)
LAUNCHER_PATH = METHOD_ROOT / (
    "case01_object_trajectory_exact5_spooled_launcher_auh_v4.py"
)


def load(path: Path) -> types.ModuleType:
    name = "_test_case01_exact6_hold_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Exact6BlockedScaffoldTest(unittest.TestCase):
    def test_publication_primitives_are_ast_identical_to_reviewed_v1(self) -> None:
        names = {
            "_identity", "_inode_anchor", "_read_fd", "open_held_parent",
            "HeldPublicationReservation", "create_publication_reservation",
            "publish_under_reservation", "_audit_sealed_publication_receipt",
            "seal_publication_receipt", "fsync_shadow_directories",
        }

        def definitions(path: Path) -> dict[str, str]:
            tree = ast.parse(path.read_bytes(), filename=str(path))
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names
            }

        reviewed = definitions(BASE_MATERIALIZER_PATH)
        overlay = definitions(MATERIALIZER_PATH)
        self.assertEqual(set(reviewed), names)
        self.assertEqual(overlay, reviewed)

    def test_release25_uses_five_replacements_and_five_new_release_leaves(self) -> None:
        materializer = load(MATERIALIZER_PATH)
        prefix = "methods/bernini_action_editing/"
        self.assertEqual(len(materializer.BASE_RELEASE_FILES), 25)
        self.assertEqual(len(materializer.REPLACED_BASE_RELEASE_FILES), 5)
        self.assertEqual(len(materializer.OVERLAY_RELEASE_FILES), 5)
        self.assertEqual(len(materializer.RELEASE_FILES), 25)
        self.assertEqual(
            materializer.REPLACED_BASE_RELEASE_FILES,
            {
                prefix + "full644_exploratory_matched_infer_adapter_v2.py",
                prefix + "full644_exploratory_matched_infer_adapter_auh_r5f.py",
                prefix + "case01_object_trajectory_exact5_runner_v1.py",
                prefix + "case01_object_trajectory_exact5_eval_v1.py",
                prefix + "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py",
            },
        )
        self.assertEqual(
            set(materializer.OVERLAY_RELEASE_FILES),
            {
                prefix + "full644_exploratory_matched_infer_adapter_v3.py",
                prefix + "infer_case01_object_trajectory_oracle_auh_r5f_v4.py",
                prefix + "case01_object_trajectory_exact5_runner_v4.py",
                prefix + "case01_object_trajectory_exact5_eval_v4.py",
                prefix + "case01_object_trajectory_exact5_spooled_launcher_auh_v4.py",
            },
        )
        self.assertTrue(
            materializer.REPLACED_BASE_RELEASE_FILES.isdisjoint(
                materializer.RELEASE_FILES
            )
        )
        self.assertEqual(
            len(
                set(materializer.OVERLAY_RELEASE_FILES)
                | {materializer.OVERLAY_MATERIALIZER_RELATIVE}
            ),
            6,
        )
        source = MATERIALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn('"source_file_count": 6', source)
        self.assertNotIn('"source_file_count": 5', source)

    def test_stable_core_tuple_is_exact_pinned_to_local_bytes(self) -> None:
        materializer = load(MATERIALIZER_PATH)
        self.assertEqual(materializer.blocked_sources(), ())
        expected_sizes = {
            "full644_exploratory_matched_infer_adapter_v3.py": 124_612,
            "infer_case01_object_trajectory_oracle_auh_r5f_v4.py": 42_184,
            "case01_object_trajectory_exact5_eval_v4.py": 116_371,
            "case01_object_trajectory_exact5_runner_v4.py": 21_712,
            "case01_object_trajectory_exact5_spooled_launcher_auh_v4.py": 27_878,
        }
        prefix = "methods/bernini_action_editing/"
        for basename, size in expected_sizes.items():
            raw = (METHOD_ROOT / basename).read_bytes()
            self.assertEqual(len(raw), size)
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(),
                materializer.OVERLAY_RELEASE_FILES[prefix + basename],
            )
        materializer.require_final_pins()

    def test_exact26_runtime_rows_close_new_base_outer_and_frozen_inner(self) -> None:
        materializer = load(MATERIALIZER_PATH)
        launcher = load(LAUNCHER_PATH)
        prefix = "methods/bernini_action_editing/"
        internal = {
            "runner": prefix + "case01_object_trajectory_exact5_runner_v4.py",
            "legacy_exact5_runner": prefix + "case01_source_bone_exact5_runner_v1.py",
            "object_eval": prefix + "case01_object_trajectory_exact5_eval_v4.py",
            "legacy_exact5_eval": prefix + "case01_source_bone_exact5_eval_v1.py",
            "frozen_runner": prefix + "full644_exploratory_matched_runner_auh_r5.py",
            "bridge": prefix + "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
            "adapter": prefix + "infer_case01_object_trajectory_oracle_auh_r5f_v4.py",
            "object_wrapper_inner": prefix + "infer_case01_object_trajectory_oracle_v1.py",
            "legacy_infer_alias": prefix + "infer_lora_full644_r5_frozen_acc46.py",
            "trajectory_projection": prefix + "object_trajectory_projection_v1.py",
            "trajectory_scaffold_module": prefix + "case01_oracle_object_trajectory_v1.py",
            "base_adapter": prefix + "full644_exploratory_matched_infer_adapter_v3.py",
            "eval_v1": prefix + "full644_exploratory_matched_eval_v1.py",
            "eval_v2": prefix + "full644_exploratory_matched_eval_v2.py",
            "model_authority": prefix + "action_preservation_decoded_eval_model_authority_v2.py",
            "base_model_manifest": prefix + "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
        }
        self.assertEqual(
            {
                role: Path(relative).name
                for role, relative in internal.items()
                if role != "base_model_manifest"
            },
            launcher.METHOD_ROLE_BASENAMES,
        )
        release_bytes = {
            relative: ("synthetic:" + relative).encode("utf-8")
            for relative in materializer.RELEASE_FILES
        }
        for role in ("adapter", "object_wrapper_inner", "base_adapter"):
            relative = internal[role]
            release_bytes[relative] = (METHOD_ROOT.parents[1] / relative).read_bytes()
        expected_static = dict(launcher.EXPECTED_STATIC_SHA256)
        for role, relative in internal.items():
            expected_static[role] = hashlib.sha256(
                release_bytes[relative]
            ).hexdigest()
        launcher.EXPECTED_STATIC_SHA256 = expected_static
        runtime_preflight = {}
        for index, role in enumerate(launcher.IDENTITY_ROLES):
            if role in internal or role == "plan":
                continue
            digest = launcher.EXPECTED_STATIC_SHA256.get(role)
            if digest is None:
                digest = hashlib.sha256(role.encode("utf-8")).hexdigest()
            runtime_preflight[role] = {
                "path": f"/runtime/{role}", "sha256": digest,
                "size": index + 1,
            }
        package_root = Path("/fresh/canary_v3")
        plan_path = package_root / "plan/hold_plan_v3.json"
        plan_raw = b'{"status":"HOLD"}\n'
        identities = materializer._runtime_identities_from_preflight(
            package_root, plan_path, plan_raw, launcher,
            release_bytes, runtime_preflight,
        )
        self.assertEqual(tuple(identities), launcher.IDENTITY_ROLES)
        self.assertEqual(len(identities), 26)
        for role, relative in internal.items():
            row = identities[role]
            self.assertEqual(row["path"], str(package_root / "release" / relative))
            self.assertEqual(row["sha256"], hashlib.sha256(
                release_bytes[relative]
            ).hexdigest())
            self.assertEqual(row["size"], len(release_bytes[relative]))
        self.assertEqual(
            identities["adapter"]["sha256"], materializer.FINAL_WRAPPER_SHA256
        )
        self.assertEqual(
            identities["base_adapter"]["sha256"],
            materializer.FINAL_BASE_ADAPTER_SHA256,
        )
        self.assertEqual(
            identities["object_wrapper_inner"]["sha256"],
            materializer.OBJECT_WRAPPER_INNER_SHA256,
        )
        self.assertEqual(len(materializer.RELEASE_FILES), 25)

    def test_seal_refuses_before_any_filesystem_or_publication_primitive(self) -> None:
        materializer = load(MATERIALIZER_PATH)
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("crossed")
            raise AssertionError("blocked seal crossed into filesystem")

        blocked_relative = next(iter(materializer.OVERLAY_RELEASE_FILES))
        with mock.patch.dict(
                 materializer.OVERLAY_RELEASE_FILES,
                 {blocked_relative: "BLOCKED_TEST_PIN"},
             ), \
             mock.patch.dict(
                 materializer.RELEASE_FILES,
                 {blocked_relative: "BLOCKED_TEST_PIN"},
             ), \
             mock.patch.object(materializer.os.path, "lexists", forbidden), \
             mock.patch.object(materializer.os, "lstat", forbidden), \
             mock.patch.object(materializer.os, "open", forbidden):
            with self.assertRaisesRegex(
                materializer.HoldPackageError, "HOLD: final source pins blocked"
            ):
                materializer.seal_overlay_source_root(
                    materializer.OVERLAY_ROOT,
                    materializer.OVERLAY_RECEIPT_PATH,
                    {},
                )
        self.assertEqual(touched, [])

    def test_controller_hold_is_first_gate_and_dynamic_pins_are_closed(self) -> None:
        controller = load(CONTROLLER_PATH)
        self.assertEqual(
            controller.CONTROLLER_STATE,
            "HOLD_BLOCKED_PENDING_CORE4_FREEZE_AND_OVERLAY_PUBLICATION",
        )
        self.assertEqual(controller.blocked_dynamic_pins(), ())
        self.assertEqual(
            (
                controller.OVERLAY_RECEIPT_SHA256,
                controller.OVERLAY_RECEIPT_SIZE,
                controller.OVERLAY_RECEIPT_DIGEST,
                controller.OVERLAY_ROOT_IDENTITY,
            ),
            (
                "5827df34b30496b3b768f26e4be91de71b4a54dda0da000f0b00373549146be4",
                3_915,
                "463bf54d848730ac8b13a625c8a66e436c5523b7595dcda29095fe194986baa2",
                [
                    48, 5704346356003806783, 2012, 2000, 16749, 2, 0,
                    4096, 0, 1787377341057464968, 1787377456971845705,
                ],
            ),
        )
        raw = MATERIALIZER_PATH.read_bytes()
        self.assertEqual(len(raw), controller.MATERIALIZER_SIZE)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), controller.MATERIALIZER_SHA256
        )
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("crossed")
            raise AssertionError("HOLD state crossed")

        stderr = io.StringIO()
        with mock.patch.object(controller, "blocked_dynamic_pins", forbidden), \
             mock.patch.object(controller, "controller", forbidden), \
             redirect_stderr(stderr):
            result = controller.main(["--execute", "malicious"])
        self.assertEqual(result, 88)
        self.assertEqual(touched, [])
        self.assertIn("HOLD", stderr.getvalue())

    def test_fresh_paths_and_abi_preservation_are_explicit(self) -> None:
        materializer = load(MATERIALIZER_PATH)
        controller = load(CONTROLLER_PATH)
        self.assertEqual(materializer.TARGET_ROOT, controller.PACKAGE_ROOT)
        self.assertEqual(materializer.OVERLAY_ROOT, controller.OVERLAY_ROOT)
        self.assertEqual(
            materializer.OVERLAY_MATERIALIZER_RELATIVE,
            controller.MATERIALIZER_RELATIVE,
        )
        self.assertIn("canary_v3", str(controller.PACKAGE_ROOT))
        self.assertIn("r5f_v4_source_overlay_6", str(controller.OVERLAY_ROOT))
        self.assertTrue(str(controller.RANK_CACHE_ROOT).endswith("r3-rank-cache"))
        self.assertEqual(
            controller.INTERNAL_RECEIPT_RELATIVE,
            "authority/package_materialization_receipt_v4.json",
        )
        self.assertEqual(
            set(controller.OVERLAY_RELATIVES),
            set(materializer.OVERLAY_RELEASE_FILES)
            | {materializer.OVERLAY_MATERIALIZER_RELATIVE},
        )
        source = MATERIALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'plan_relative = "plan/case01_object_trajectory_exact5_r64_HOLD_plan_v3.json"',
            source,
        )
        self.assertIn("object_trajectory_exact5_report_v3.json", source)
        self.assertIn("object_trajectory_exact5_runner_attestation_v3.json", source)
        self.assertEqual(
            materializer.CAMPAIGN,
            "case01-object-trajectory-exact5-r64-engineering-oracle-v3",
        )
        self.assertIn('input_relative = "launch/root_launch_input_HOLD_v3.json"', source)
        self.assertIn('payload_relative = "launch/root_launch_payload_HOLD_v3.sh"', source)

    def test_controller_loads_only_the_new_materializer_contract(self) -> None:
        materializer = load(MATERIALIZER_PATH)
        controller = load(CONTROLLER_PATH)
        loaded = controller.load_materializer(MATERIALIZER_PATH.read_bytes())
        self.assertEqual(loaded.TARGET_ROOT, controller.PACKAGE_ROOT)
        self.assertEqual(loaded.OVERLAY_ROOT, controller.OVERLAY_ROOT)
        self.assertEqual(
            loaded.OVERLAY_RECEIPT_PATH, controller.OVERLAY_RECEIPT_PATH
        )
        self.assertEqual(
            loaded.OVERLAY_MATERIALIZER_RELATIVE,
            controller.MATERIALIZER_RELATIVE,
        )
        self.assertEqual(loaded.RELEASE_FILES, materializer.RELEASE_FILES)
        self.assertEqual(len(loaded.OVERLAY_RELEASE_FILES), 5)
        self.assertEqual(len(loaded.RELEASE_FILES), 25)

    def test_controller_has_no_scheduler_or_network_surface(self) -> None:
        tree = ast.parse(CONTROLLER_PATH.read_bytes(), filename=str(CONTROLLER_PATH))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse({"socket", "requests", "paramiko"} & imports)
        source = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("sbatch", source)
        self.assertNotIn("ssh ", source)
        popen_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "Popen"
        ]
        self.assertEqual(len(popen_calls), 1)

    def test_ready_is_one_exact_state_only_copy_and_is_not_invoked(self) -> None:
        hold_lines = CONTROLLER_PATH.read_bytes().splitlines(keepends=True)
        ready_lines = READY_PATH.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(hold_lines), len(ready_lines))
        differences = [
            index
            for index, (hold, ready) in enumerate(zip(hold_lines, ready_lines))
            if hold != ready
        ]
        self.assertEqual(len(differences), 1)
        changed = differences[0]
        self.assertEqual(
            hold_lines[changed],
            b'CONTROLLER_STATE = "HOLD_BLOCKED_PENDING_CORE4_FREEZE_AND_OVERLAY_PUBLICATION"\n',
        )
        self.assertEqual(
            ready_lines[changed],
            b'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_ATTEMPT_R64_HOLD_PACKAGE"\n',
        )
        hold = load(CONTROLLER_PATH)
        ready = load(READY_PATH)
        self.assertNotEqual(hold.CONTROLLER_STATE, hold.READY_STATE)
        self.assertEqual(ready.CONTROLLER_STATE, ready.READY_STATE)
        self.assertEqual(hold.dynamic_pin_values(), ready.dynamic_pin_values())
        self.assertEqual(hold.blocked_dynamic_pins(), ())
        self.assertEqual(ready.blocked_dynamic_pins(), ())
        self.assertEqual(hold.authorization_token(), ready.authorization_token())


if __name__ == "__main__":
    unittest.main()
