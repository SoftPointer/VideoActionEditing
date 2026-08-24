#!/usr/bin/env python3
"""Hostile tests for the fresh exact5-overlay package materializer."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = METHOD_ROOT / (
    "tools/materialize_case01_object_trajectory_exact5_r64_overlay_package_v2.py"
)
BASE_MATERIALIZER_PATH = METHOD_ROOT / (
    "tools/materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py"
)
LAUNCHER_PATH = METHOD_ROOT / (
    "case01_object_trajectory_exact5_spooled_launcher_auh_v3.py"
)


def load(path: Path) -> types.ModuleType:
    name = "_test_case01_overlay_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class OverlayMaterializerTest(unittest.TestCase):
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

    def test_release25_path_semantics_keep_base_inner_distinct(self) -> None:
        module = load(MATERIALIZER_PATH)
        prefix = "methods/bernini_action_editing/"
        self.assertEqual(len(module.BASE_RELEASE_FILES), 25)
        self.assertEqual(len(module.RELEASE_FILES), 25)
        self.assertEqual(len(module.OVERLAY_RELEASE_FILES), 4)
        self.assertEqual(
            module.BASE_RELEASE_FILES[prefix + "infer_case01_object_trajectory_oracle_v1.py"],
            "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9",
        )
        self.assertEqual(
            module.BASE_RELEASE_FILES[prefix + "case01_object_trajectory_exact5_runner_v1.py"],
            "e47b81643c1d17e5099a9b33f16ca75521001ad52d2df2305b46b7e8c4d5ac4c",
        )
        self.assertEqual(
            module.BASE_RELEASE_FILES[prefix + "case01_object_trajectory_exact5_eval_v1.py"],
            "47cc871b82b8cf7762db9183997440eeabd287b1c702d9cd7421fd43e0a555e0",
        )
        expected_overlay = {
            prefix + "infer_case01_object_trajectory_oracle_auh_r5f_v3.py":
                "b30bba5c9cd233d412ffd88d8413311e9ffbb79d3ddf69aaf6eb2ee96183b489",
            prefix + "case01_object_trajectory_exact5_runner_v3.py":
                "02207e64a129444b26adf8bd92307102c4a91e85d2a029fa60030a7e9e6f45c8",
            prefix + "case01_object_trajectory_exact5_eval_v3.py":
                "cfdfc5fec04243265b6c122649fed9144d89510d17184a77782c0ec0ddc5ed8a",
            prefix + "case01_object_trajectory_exact5_spooled_launcher_auh_v3.py":
                "0073a3b549bdbecc49866471c43f6882e4e494876312a7082560b0cc29e0f913",
        }
        self.assertEqual(module.OVERLAY_RELEASE_FILES, expected_overlay)
        self.assertEqual(
            module.RELEASE_FILES[prefix + "infer_case01_object_trajectory_oracle_v1.py"],
            module.OBJECT_WRAPPER_INNER_SHA256,
        )
        self.assertTrue(module.REPLACED_BASE_RELEASE_FILES.isdisjoint(module.RELEASE_FILES))
        snapshot_expected = module.snapshot_expected_files(
            module.BASE_MATERIALIZER_SHA256
        )
        self.assertEqual(len(snapshot_expected), 33)
        self.assertEqual(
            snapshot_expected[
                prefix + "full644_exploratory_matched_infer_adapter_auh_r5f.py"
            ],
            "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
        )
        self.assertEqual(
            snapshot_expected[
                prefix + "case01_object_trajectory_exact5_spooled_launcher_auh_v1.py"
            ],
            "a81e812627125a24d72ec956b384e30479df379b66b5a94da871021c3e14267f",
        )
        self.assertEqual(
            snapshot_expected[module.SNAPSHOT_MATERIALIZER_RELATIVE],
            module.BASE_MATERIALIZER_SHA256,
        )

    def test_exact26_runtime_rows_close_outer_inner_paths_and_hashes(self) -> None:
        module = load(MATERIALIZER_PATH)
        launcher = load(LAUNCHER_PATH)
        prefix = "methods/bernini_action_editing/"
        internal = {
            "runner": prefix + "case01_object_trajectory_exact5_runner_v3.py",
            "legacy_exact5_runner": prefix + "case01_source_bone_exact5_runner_v1.py",
            "object_eval": prefix + "case01_object_trajectory_exact5_eval_v3.py",
            "legacy_exact5_eval": prefix + "case01_source_bone_exact5_eval_v1.py",
            "frozen_runner": prefix + "full644_exploratory_matched_runner_auh_r5.py",
            "bridge": prefix + "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
            "adapter": prefix + "infer_case01_object_trajectory_oracle_auh_r5f_v3.py",
            "object_wrapper_inner": prefix + "infer_case01_object_trajectory_oracle_v1.py",
            "legacy_infer_alias": prefix + "infer_lora_full644_r5_frozen_acc46.py",
            "trajectory_projection": prefix + "object_trajectory_projection_v1.py",
            "trajectory_scaffold_module": prefix + "case01_oracle_object_trajectory_v1.py",
            "base_adapter": prefix + "full644_exploratory_matched_infer_adapter_v2.py",
            "eval_v1": prefix + "full644_exploratory_matched_eval_v1.py",
            "eval_v2": prefix + "full644_exploratory_matched_eval_v2.py",
            "model_authority": prefix + "action_preservation_decoded_eval_model_authority_v2.py",
            "base_model_manifest": prefix + "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
        }
        self.assertEqual(
            set(internal) - {"base_model_manifest"},
            set(launcher.METHOD_ROLE_BASENAMES),
        )
        self.assertEqual(
            {
                role: Path(relative).name for role, relative in internal.items()
                if role != "base_model_manifest"
            },
            launcher.METHOD_ROLE_BASENAMES,
        )
        self.assertEqual(
            internal["base_model_manifest"],
            prefix + "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
        )

        release_bytes = {
            relative: ("synthetic:" + relative).encode("utf-8")
            for relative in module.RELEASE_FILES
        }
        for role in ("adapter", "object_wrapper_inner"):
            relative = internal[role]
            release_bytes[relative] = (METHOD_ROOT.parents[1] / relative).read_bytes()
        expected_static = dict(launcher.EXPECTED_STATIC_SHA256)
        for role, relative in internal.items():
            expected_static[role] = sha(release_bytes[relative])
        launcher.EXPECTED_STATIC_SHA256 = expected_static

        runtime_preflight = {}
        for index, role in enumerate(launcher.IDENTITY_ROLES):
            if role in internal or role == "plan":
                continue
            digest = launcher.EXPECTED_STATIC_SHA256.get(role)
            if digest is None:
                digest = sha(role.encode("utf-8"))
            runtime_preflight[role] = {
                "path": f"/runtime/{role}", "sha256": digest,
                "size": index + 1,
            }
        package_root = Path("/fresh/canary_v2")
        plan_path = package_root / "plan/hold_plan_v3.json"
        plan_raw = b'{"status":"HOLD"}\n'
        identities = module._runtime_identities_from_preflight(
            package_root, plan_path, plan_raw, launcher,
            release_bytes, runtime_preflight,
        )
        self.assertEqual(tuple(identities), launcher.IDENTITY_ROLES)
        self.assertEqual(len(identities), 26)
        for role, relative in internal.items():
            row = identities[role]
            self.assertEqual(row["path"], str(package_root / "release" / relative))
            self.assertEqual(row["sha256"], sha(release_bytes[relative]))
            self.assertEqual(row["size"], len(release_bytes[relative]))

        outer = identities["adapter"]
        inner = identities["object_wrapper_inner"]
        self.assertEqual(
            outer["sha256"],
            "b30bba5c9cd233d412ffd88d8413311e9ffbb79d3ddf69aaf6eb2ee96183b489",
        )
        self.assertEqual(
            inner["sha256"],
            "20ee1447148cfc60c6cb745316ce972180070d50b6431a8f4d254ee5dfff7db9",
        )
        self.assertNotEqual(outer["path"], inner["path"])
        producer_outer = dict(outer)
        producer_inner = dict(inner)
        self.assertEqual(outer, producer_outer)
        self.assertEqual(inner, producer_inner)
        self.assertEqual(len(module.RELEASE_FILES), 25)
        self.assertEqual(len(module.object_sha(identities)), 64)

    def test_blocked_pins_fail_before_any_filesystem_probe(self) -> None:
        module = load(MATERIALIZER_PATH)
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("fs")
            raise AssertionError("blocked pins crossed the state gate")

        with mock.patch.object(module.os.path, "lexists", forbidden), \
             mock.patch.object(module.os, "lstat", forbidden):
            with self.assertRaisesRegex(module.HoldPackageError, r"^HOLD:"):
                module.materialize(
                    module.TARGET_ROOT, module.SOURCE_SNAPSHOT_ROOT,
                    module.OVERLAY_ROOT, module.OVERLAY_RECEIPT_PATH,
                    "143808", "auh7-1b-gpu-292",
                    snapshot_manifest_sha256="0" * 64,
                    snapshot_materializer_sha256=module.BASE_MATERIALIZER_SHA256,
                    overlay_materializer_sha256="1" * 64,
                    overlay_receipt_sha256="BLOCKED",
                    overlay_receipt_size=0,
                    overlay_receipt_digest="BLOCKED",
                    overlay_root_identity=[],
                )
        self.assertEqual(touched, [])

    def test_v3_output_namespace_is_single_and_fresh(self) -> None:
        raw = MATERIALIZER_PATH.read_text(encoding="utf-8")
        self.assertIn('"outputs/media_v3"', raw)
        self.assertIn('"runtime/model-authority-v3"', raw)
        self.assertIn("object_trajectory_exact5_report_v3.json", raw)
        self.assertIn("object_trajectory_exact5_runner_attestation_v3.json", raw)
        self.assertIn("case01_object_trajectory_exact5_r64_HOLD_plan_v3.json", raw)
        self.assertNotIn('"runtime/model-authority-v2"', raw)

    def test_seal_exact5_then_receipt_first_preflight(self) -> None:
        module = load(MATERIALIZER_PATH)
        canonical_tmp = Path(tempfile.gettempdir()).resolve(strict=True)
        base = Path(tempfile.mkdtemp(prefix="case01-overlay-v2-", dir=canonical_tmp))
        root = base / "overlay"
        receipt = base / "overlay.receipt_v1.json"
        raw_by_relative = {
            "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_auh_r5f_v3.py": b"# composite\n",
            "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v3.py": b"# runner\n",
            "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v3.py": b"# eval\n",
            "methods/bernini_action_editing/case01_object_trajectory_exact5_spooled_launcher_auh_v3.py": b"# launcher\n",
            module.OVERLAY_MATERIALIZER_RELATIVE: MATERIALIZER_PATH.read_bytes(),
        }
        try:
            root.mkdir(mode=0o755)
            for relative, raw in raw_by_relative.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                target.write_bytes(raw)
                os.chmod(target, 0o644)
            for directory in sorted(
                {root, *(path.parent for path in root.rglob("*") if path.is_file())},
                key=lambda path: len(path.parts),
            ):
                os.chmod(directory, 0o755)
            overlay_release = {
                relative: sha(raw)
                for relative, raw in raw_by_relative.items()
                if relative != module.OVERLAY_MATERIALIZER_RELATIVE
            }
            module.OVERLAY_RELEASE_FILES = overlay_release
            module.RELEASE_FILES.update(overlay_release)
            module.OVERLAY_ROOT = root
            module.OVERLAY_RECEIPT_PATH = receipt
            module.__file__ = str(root / module.OVERLAY_MATERIALIZER_RELATIVE)
            specs = {
                relative: {"sha256": sha(raw), "size": len(raw)}
                for relative, raw in raw_by_relative.items()
            }
            value = module.seal_overlay_source_root(root, receipt, specs)
            self.assertEqual(value["source_file_count"], 5)
            self.assertEqual(value["receipt_digest"], module.object_sha({
                key: item for key, item in value.items() if key != "receipt_digest"
            }))
            self.assertEqual(stat.S_IMODE(os.lstat(root).st_mode), 0o555)
            self.assertEqual(stat.S_IMODE(os.lstat(receipt).st_mode), 0o400)
            self.assertTrue(all(
                stat.S_IMODE(os.lstat(root / relative).st_mode) == 0o444
                for relative in raw_by_relative
            ))

            receipt_raw = receipt.read_bytes()
            receipt_sha = sha(receipt_raw)
            root_identity = list(module._identity(os.lstat(root)))
            replayed, evidence = module.preflight_overlay(
                root, receipt,
                materializer_sha256=sha(
                    raw_by_relative[module.OVERLAY_MATERIALIZER_RELATIVE]
                ),
                receipt_sha256=receipt_sha,
                receipt_size=len(receipt_raw),
                receipt_digest=value["receipt_digest"],
                root_identity=root_identity,
            )
            self.assertEqual(replayed, raw_by_relative)
            self.assertEqual(evidence["source_file_count"], 5)
            self.assertEqual(evidence["receipt"]["sha256"], sha(receipt_raw))

            with self.assertRaisesRegex(
                module.HoldPackageError, "stable authority differs"
            ):
                module.preflight_overlay(
                    root, receipt,
                    materializer_sha256=sha(
                        raw_by_relative[module.OVERLAY_MATERIALIZER_RELATIVE]
                    ),
                    receipt_sha256="0" * 64,
                    receipt_size=len(receipt_raw),
                    receipt_digest=value["receipt_digest"],
                    root_identity=root_identity,
                )
        finally:
            if receipt.exists():
                os.chmod(receipt, 0o600)
            if root.exists():
                for path in sorted(
                    (item for item in root.rglob("*") if item.is_dir()),
                    key=lambda item: len(item.parts), reverse=True,
                ):
                    os.chmod(path, 0o755)
                os.chmod(root, 0o755)
            shutil.rmtree(base)

    def test_overlay_receipt_root_and_leaf_inode_replacements_refused(self) -> None:
        canonical_tmp = Path(tempfile.gettempdir()).resolve(strict=True)
        for variant in ("receipt", "root", "leaf"):
            with self.subTest(variant=variant):
                module = load(MATERIALIZER_PATH)
                base = Path(tempfile.mkdtemp(
                    prefix=f"case01-overlay-replace-{variant}-",
                    dir=canonical_tmp,
                ))
                root = base / "overlay"
                receipt = base / "overlay.receipt_v1.json"
                raw_by_relative = {
                    "methods/bernini_action_editing/infer_case01_object_trajectory_oracle_auh_r5f_v3.py": b"# composite\n",
                    "methods/bernini_action_editing/case01_object_trajectory_exact5_runner_v3.py": b"# runner\n",
                    "methods/bernini_action_editing/case01_object_trajectory_exact5_eval_v3.py": b"# eval\n",
                    "methods/bernini_action_editing/case01_object_trajectory_exact5_spooled_launcher_auh_v3.py": b"# launcher\n",
                    module.OVERLAY_MATERIALIZER_RELATIVE: MATERIALIZER_PATH.read_bytes(),
                }

                def populate(target_root: Path, *, sealed: bool) -> None:
                    target_root.mkdir(mode=0o755)
                    for relative, raw in raw_by_relative.items():
                        target = target_root / relative
                        target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                        target.write_bytes(raw)
                        os.chmod(target, 0o444 if sealed else 0o644)
                    for directory in sorted(
                        {
                            target_root,
                            *(path.parent for path in target_root.rglob("*") if path.is_file()),
                        },
                        key=lambda path: len(path.parts), reverse=sealed,
                    ):
                        os.chmod(directory, 0o555 if sealed else 0o755)

                try:
                    populate(root, sealed=False)
                    overlay_release = {
                        relative: sha(raw)
                        for relative, raw in raw_by_relative.items()
                        if relative != module.OVERLAY_MATERIALIZER_RELATIVE
                    }
                    module.OVERLAY_RELEASE_FILES = overlay_release
                    module.RELEASE_FILES.update(overlay_release)
                    module.OVERLAY_ROOT = root
                    module.OVERLAY_RECEIPT_PATH = receipt
                    module.__file__ = str(
                        root / module.OVERLAY_MATERIALIZER_RELATIVE
                    )
                    specs = {
                        relative: {"sha256": sha(raw), "size": len(raw)}
                        for relative, raw in raw_by_relative.items()
                    }
                    receipt_value = module.seal_overlay_source_root(
                        root, receipt, specs,
                    )
                    receipt_raw = receipt.read_bytes()
                    receipt_sha = sha(receipt_raw)
                    root_identity = list(module._identity(os.lstat(root)))

                    if variant == "receipt":
                        before = os.lstat(receipt)
                        replacement = base / "replacement.receipt.json"
                        replacement.write_bytes(receipt_raw)
                        os.chmod(replacement, 0o400)
                        os.replace(replacement, receipt)
                        self.assertEqual(receipt.read_bytes(), receipt_raw)
                        self.assertNotEqual(before.st_ino, os.lstat(receipt).st_ino)
                    elif variant == "root":
                        before = os.lstat(root)
                        replacement_root = base / "replacement-root"
                        populate(replacement_root, sealed=True)
                        old_root = base / "old-root"
                        # Darwin refuses renaming a 0555 source directory even
                        # when its writable parent owns the namespace.  This
                        # chmod is a hostile-test portability seam only; the
                        # replacement is restored to the receipt's 0555 mode.
                        os.chmod(root, 0o755)
                        os.rename(root, old_root)
                        os.chmod(old_root, 0o555)
                        os.chmod(replacement_root, 0o755)
                        os.rename(replacement_root, root)
                        os.chmod(root, 0o555)
                        self.assertEqual(
                            {
                                relative: (root / relative).read_bytes()
                                for relative in raw_by_relative
                            },
                            raw_by_relative,
                        )
                        self.assertNotEqual(before.st_ino, os.lstat(root).st_ino)
                    else:
                        relative = next(iter(sorted(raw_by_relative)))
                        target = root / relative
                        before = os.lstat(target)
                        os.chmod(target.parent, 0o755)
                        replacement = target.with_name(target.name + ".replacement")
                        replacement.write_bytes(raw_by_relative[relative])
                        os.chmod(replacement, 0o444)
                        os.replace(replacement, target)
                        os.chmod(target.parent, 0o555)
                        self.assertEqual(target.read_bytes(), raw_by_relative[relative])
                        self.assertNotEqual(before.st_ino, os.lstat(target).st_ino)

                    with self.assertRaises(module.HoldPackageError):
                        module.preflight_overlay(
                            root, receipt,
                            materializer_sha256=sha(
                                raw_by_relative[module.OVERLAY_MATERIALIZER_RELATIVE]
                            ),
                            receipt_sha256=receipt_sha,
                            receipt_size=len(receipt_raw),
                            receipt_digest=receipt_value["receipt_digest"],
                            root_identity=root_identity,
                        )
                finally:
                    if base.exists():
                        for directory, _subdirs, _files in os.walk(
                            base, topdown=False,
                        ):
                            try:
                                os.chmod(directory, 0o755)
                            except FileNotFoundError:
                                pass
                        shutil.rmtree(base)

    def test_publication_applied_error_and_postseal_error_stay_0400(self) -> None:
        module = load(MATERIALIZER_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            target = parent / "target"
            receipt = parent / "receipt.json"
            shadow = parent / "shadow"
            shadow.mkdir(); shadow.chmod(0o555)
            parent_fd = os.open(
                parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            reservation = module.create_publication_reservation(
                parent_fd, receipt_path=receipt, target_root=target,
            )
            real_rename = module.os.rename

            def applied_then_error(*args, **_kwargs):
                (parent / args[0]).chmod(0o755)
                real_rename(parent / args[0], parent / args[1])
                (parent / args[1]).chmod(0o555)
                raise OSError(5, "synthetic NFS applied-then-error")

            try:
                with mock.patch.object(
                    module.os, "rename", side_effect=applied_then_error,
                ), self.assertRaises(module.PublicationCommittedError) as caught:
                    module.publish_under_reservation(
                        parent_fd, shadow.name, target.name, reservation,
                    )
                self.assertEqual(
                    caught.exception.observation["namespace_state"],
                    "target_same_inode_source_absent",
                )
                self.assertFalse(
                    caught.exception.observation["rename_returned_zero"],
                )
                _raw, terminal = module.seal_publication_receipt(
                    parent_fd, reservation,
                    {
                        "schema_version": "test-terminal",
                        "status": "PUBLISHED_COMMIT_ERROR_NOT_ADMISSION",
                        "target_root": str(target),
                        "receipt_path": str(receipt),
                        "receipt_is_consumption_gate": False,
                        "receipt_is_admission": False,
                        "target_root_identity": list(caught.exception.identity),
                        "publication_observation": caught.exception.observation,
                    },
                )
                self.assertEqual(
                    terminal["status"], "PUBLISHED_COMMIT_ERROR_NOT_ADMISSION",
                )
                self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
            finally:
                reservation.close(); os.close(parent_fd)

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            target = parent / "target"; target.mkdir(); target.chmod(0o555)
            receipt = parent / "receipt.json"
            parent_fd = os.open(
                parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            reservation = module.create_publication_reservation(
                parent_fd, receipt_path=receipt, target_root=target,
            )
            target_identity = module._identity(target.stat())
            real_fsync = module.os.fsync

            def fail_parent_fsync(descriptor: int) -> None:
                if descriptor == parent_fd:
                    raise OSError(5, "synthetic postseal parent fsync")
                real_fsync(descriptor)

            try:
                with mock.patch.object(
                    module.os, "fsync", side_effect=fail_parent_fsync,
                ):
                    _raw, value = module.seal_publication_receipt(
                        parent_fd, reservation,
                        {
                            "schema_version": "test",
                            "status": "PASS",
                            "target_root": str(target),
                            "receipt_path": str(receipt),
                            "target_root_identity": list(target_identity),
                            "receipt_is_consumption_gate": True,
                            "receipt_is_admission": True,
                        },
                    )
                self.assertEqual(value["status"], "PASS")
                self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
            finally:
                reservation.close(); os.close(parent_fd)


class OverlayLauncherTest(unittest.TestCase):
    def test_exact26_inner_outer_contract_and_hold_plan(self) -> None:
        module = load(LAUNCHER_PATH)
        self.assertEqual(len(module.IDENTITY_ROLES), 26)
        self.assertIn("object_wrapper_inner", module.IDENTITY_ROLES)
        self.assertEqual(
            module.METHOD_ROLE_BASENAMES["adapter"],
            "infer_case01_object_trajectory_oracle_auh_r5f_v3.py",
        )
        self.assertEqual(
            module.METHOD_ROLE_BASENAMES["object_wrapper_inner"],
            "infer_case01_object_trajectory_oracle_v1.py",
        )
        self.assertEqual(module.blocked_roles(), ())
        self.assertEqual(
            module.EXPECTED_STATIC_SHA256["runner"],
            "02207e64a129444b26adf8bd92307102c4a91e85d2a029fa60030a7e9e6f45c8",
        )
        self.assertEqual(
            module.EXPECTED_STATIC_SHA256["object_eval"],
            "cfdfc5fec04243265b6c122649fed9144d89510d17184a77782c0ec0ddc5ed8a",
        )
        self.assertEqual(
            module.EXPECTED_STATIC_SHA256["adapter"],
            "b30bba5c9cd233d412ffd88d8413311e9ffbb79d3ddf69aaf6eb2ee96183b489",
        )

        plan = {
            "status": "HOLD_INCOMPLETE_PRODUCER_OR_AUTHORITY",
            "launch_allowed": False, "production_ready": False,
            "hold_reasons": ["test"],
            "tasks": [
                {
                    "task_id": task_id, "oracle_arm": arm,
                    "source_onset_policy": "hard1_every_step",
                }
                for arm, task_id in zip(module.ARM_ORDER, module.TASK_IDS)
            ],
        }
        plan_raw = module.canonical_json_bytes(plan) + b"\n"
        method = Path("/authority/release/methods/bernini_action_editing")
        identities: dict[str, dict[str, object]] = {}
        for index, role in enumerate(module.IDENTITY_ROLES):
            if role in module.METHOD_ROLE_BASENAMES:
                path = method / module.METHOD_ROLE_BASENAMES[role]
            elif role == "plan":
                path = Path("/authority/plan/hold-v2.json")
            else:
                path = Path(f"/authority/runtime/{role}-{index}")
            expected = module.EXPECTED_STATIC_SHA256.get(role)
            digest = expected if isinstance(expected, str) and len(expected) == 64 else sha(role.encode())
            identities[role] = {"path": str(path), "sha256": digest, "size": index + 1}
        identities["plan"] = {
            "path": "/authority/plan/hold-v2.json",
            "sha256": sha(plan_raw), "size": len(plan_raw),
        }
        value = {
            "schema_version": module.INPUT_SCHEMA,
            "entry_mode": "trusted_stdin", "campaign_mode": module.CAMPAIGN,
            "holder_job_id": "143808", "expected_node": "node292",
            "expected_allocation_gpu_count": 8, "identities": identities,
            "output_report": "/authority/final/report.json",
            "runner_attestation": "/authority/final/attestation.json",
            "model_root": "/authority/model", "bernini_root": "/authority/bernini",
            "veomni_root": "/authority/veomni",
            "authority_root": "/authority/runtime/model-authority-v3",
            "rank_cache_root": "/tmp/case01-r2-cache",
        }
        self.assertEqual(
            module.validate_input(
                value, reopen=False, allow_blocked_pins=True,
                plan_override=plan,
            ),
            value,
        )


if __name__ == "__main__":
    unittest.main()
