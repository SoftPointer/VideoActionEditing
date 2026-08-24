#!/usr/bin/env python3
"""Hostile real-temp tests for AUHv2 receipt-gated downstream consumers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD = Path(__file__).resolve().parents[1]


def load(path: Path, prefix: str) -> types.ModuleType:
    name = prefix + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load(
    METHOD / "tools/build_case01_object_trajectory_exact5_source_snapshot_v1.py",
    "_auh_v2_builder_",
)
materializer = load(
    METHOD / "tools/materialize_case01_object_trajectory_exact5_r64_hold_package_v1.py",
    "_auh_v2_materializer_",
)
cpu = load(
    METHOD / "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py",
    "_auh_v2_cpu_",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def seal_tree(root: Path) -> None:
    for directory, subdirs, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(Path(directory) / name, 0o444)
        for name in subdirs:
            os.chmod(Path(directory) / name, 0o555)
    os.chmod(root, 0o555)


class StageFixture:
    def __init__(self, base: Path):
        self.base = base.resolve()
        self.stage = self.base / "physical15"
        self.receipt = self.base / "physical15.receipt.json"
        self.target = self.base / "snapshot-target"
        self.snapshot_receipt = self.base / "snapshot-target.receipt.json"
        self.old = self.base / "old"
        self.stage.mkdir(); self.old.mkdir()
        self.builder_raw = Path(builder.__file__).read_bytes()
        self.builder_sha = sha(self.builder_raw)
        self.fake_pins: dict[str, str] = {}
        for index, relative in enumerate(builder.STAGED_FILES):
            raw = (
                Path(materializer.__file__).read_bytes()
                if relative == materializer.MATERIALIZER_RELATIVE
                else f"leaf-{index}:{relative}\n".encode()
            )
            path = self.stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            self.fake_pins[relative] = sha(raw)
        path = self.stage / builder.BUILDER_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.builder_raw)
        seal_tree(self.stage)
        self.uid = os.geteuid(); self.gid = os.getegid()
        self.python = Path(sys.executable).resolve()
        self.python_raw = self.python.read_bytes()
        self.bootstrap_sha = "b" * 64

    @contextmanager
    def patched_builder(self):
        with ExitStack() as stack:
            for name, value in (
                ("STAGING_ROOT", self.stage),
                ("STAGING_RECEIPT_PATH", self.receipt),
                ("TARGET_ROOT", self.target),
                ("SNAPSHOT_PUBLICATION_RECEIPT_PATH", self.snapshot_receipt),
                ("OLD_EXACT5_SNAPSHOT", self.old),
                ("STAGED_FILES", self.fake_pins),
                ("STAGING_REMOTE_UID", self.uid),
                ("STAGING_REMOTE_GID", self.gid),
                ("STAGING_REMOTE_PYTHON", self.python),
                ("STAGING_REMOTE_PYTHON_SHA256", sha(self.python_raw)),
                ("STAGING_REMOTE_PYTHON_SIZE", len(self.python_raw)),
                ("STAGING_BOOTSTRAP_SHA256", self.bootstrap_sha),
            ):
                stack.enter_context(mock.patch.object(builder, name, value))
            yield

    @contextmanager
    def patched_materializer(self):
        rows = self.rows()
        with ExitStack() as stack:
            for name, value in (
                ("SNAPSHOT_STAGING_ROOT", self.stage),
                ("SNAPSHOT_STAGING_RECEIPT_PATH", self.receipt),
                ("STAGING_REMOTE_UID", self.uid),
                ("STAGING_REMOTE_GID", self.gid),
                ("STAGING_REMOTE_PYTHON", self.python),
                ("STAGING_REMOTE_PYTHON_SHA256", sha(self.python_raw)),
                ("STAGING_REMOTE_PYTHON_SIZE", len(self.python_raw)),
                ("STAGING_BOOTSTRAP_SHA256", self.bootstrap_sha),
            ):
                stack.enter_context(mock.patch.object(materializer, name, value))
            yield rows

    @contextmanager
    def patched_cpu(self):
        rows = tuple(self.rows())
        cpu_target = self.base / "cpu-target"
        cpu_stage = self.base / "cpu-stage"
        with ExitStack() as stack:
            for name, value in (
                ("SOURCE_ROOT", self.stage),
                ("SOURCE_RECEIPT_PATH", self.receipt),
                ("SOURCE_STAGE_AUTHORITIES", rows),
                ("SOURCE_STAGE_UID", self.uid),
                ("SOURCE_STAGE_GID", self.gid),
                ("VACE_PYTHON", self.python),
                ("SOURCE_STAGE_REMOTE_PYTHON_SHA256", sha(self.python_raw)),
                ("SOURCE_STAGE_REMOTE_PYTHON_SIZE", len(self.python_raw)),
                ("SOURCE_STAGE_BOOTSTRAP_SHA256", self.bootstrap_sha),
                ("TARGET_ROOT", cpu_target),
                ("EVIDENCE_DIR", cpu_target / "evidence"),
                ("LOGS_DIR", cpu_target / "logs"),
                ("STAGE_ROOT", cpu_stage),
                ("PUBLICATION_ROOT", cpu_stage / "publication"),
            ):
                stack.enter_context(mock.patch.object(cpu, name, value))
            yield cpu_target, cpu_stage

    def rows(self) -> list[dict[str, object]]:
        result = []
        for relative, pin in {**self.fake_pins, builder.BUILDER_RELATIVE: self.builder_sha}.items():
            result.append({
                "relative": relative, "sha256": pin,
                "size": (self.stage / relative).stat().st_size,
            })
        return sorted(result, key=lambda row: str(row["relative"]))

    def write_receipt(
        self, *, operation: str = "stage", mutation: str | None = None,
    ) -> bytes:
        rows = self.rows()
        manifest = builder.expected_staging_manifest(rows)
        root_identity = list(builder.ident(os.lstat(self.stage)))
        if mutation == "stale":
            root_identity[1] += 1
        descriptor = os.open(self.receipt, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            receipt_anchor = builder._receipt_inode_anchor(os.fstat(descriptor))
            if operation == "stage":
                status = "STAGED_RECEIPT_GATED"
                observation = {
                    "kind": "live_posix_rename_under_held_receipt_reservation",
                    "root_identity": root_identity,
                    "held_inode_continuity": True,
                    "ordinary_posix_rename_performed_this_operation": True,
                    "rename_noreplace_performed_this_operation": False,
                    "target_absent_rechecked_before_rename": True,
                    "whole_tree_atomically_visible": True,
                    "historical_replacement_claim": "not_made",
                }
                request = stage = "a" * 64
                terminal = None
            else:
                status = "RECOVERED_RECEIPT_ONLY"
                observation = {
                    "kind": "recovered_existing_exact15_current_inode",
                    "root_identity": root_identity,
                    "held_inode_continuity": True,
                    "ordinary_posix_rename_performed_this_operation": False,
                    "rename_noreplace_performed_this_operation": False,
                    "target_absent_rechecked_before_rename": False,
                    "whole_tree_atomically_visible": True,
                    "historical_replacement_claim": "not_made",
                }
                request, stage, terminal = "c" * 64, "a" * 64, "d" * 64
            value = {
                "schema_version": builder.STAGING_RECEIPT_SCHEMA,
                "status": status, "operation": operation,
                "target_root": str(self.stage),
                "receipt_path": str(self.receipt),
                "manifest_digest": manifest["manifest_digest"],
                "request_payload_sha256": request,
                "stage_payload_sha256": stage,
                "bootstrap_source_sha256": self.bootstrap_sha,
                "file_count": 15,
                "files": [
                    {**row, "mode": 0o444, "nlink": 1} for row in rows
                ],
                "directories": manifest["directories"],
                "file_mode": 0o444, "directory_mode": 0o555,
                "receipt_mode": 0o400,
                "held_parent_identity_replayed": True,
                "ancestor_chain_nofollow": True,
                "publication_protocol": builder.STAGING_PUBLICATION_PROTOCOL,
                "rename_noreplace": False,
                "cooperative_writer_exclusion": True,
                "receipt_is_consumption_gate": True,
                "receipt_is_admission": True,
                "uncooperative_same_uid_race_out_of_scope": True,
                "target_observation": observation,
                "commit_terminal_digest": terminal,
                "receipt_inode_anchor": receipt_anchor,
                "launch_allowed": False,
            }
            if mutation == "not_admission":
                value["receipt_is_admission"] = False
            value["receipt_digest"] = builder.object_digest(value)
            raw = builder.canonical(value) + b"\n"
            os.write(descriptor, raw); os.fsync(descriptor)
            os.fchmod(descriptor, 0o400); os.fsync(descriptor)
            return raw
        finally:
            os.close(descriptor)


class ReceiptConsumerTests(unittest.TestCase):
    def test_stage_and_recovered_final_receipts_are_consumable_by_all_three(self) -> None:
        for operation in ("stage", "recover-receipt"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                fixture = StageFixture(Path(temporary))
                with fixture.patched_builder():
                    raw = fixture.write_receipt(operation=operation)
                    gate = builder.open_staging_gate(fixture.builder_sha)
                    gate.replay(); gate.close()
                with fixture.patched_materializer() as rows:
                    evidence = materializer.replay_staging_receipt_authority(raw, rows)
                    self.assertEqual(evidence["staging_file_count"], 15)
                with fixture.patched_cpu():
                    gate = cpu.open_source_stage_gate()
                    gate.replay()
                    self.assertEqual(gate.evidence()["file_count"], 15)
                    gate.close()

    def test_missing_0600_partial_tampered_and_stale_fail_before_targets(self) -> None:
        for mutation in ("missing", "reservation0600", "partial", "not_admission", "stale"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = StageFixture(Path(temporary))
                with fixture.patched_builder():
                    if mutation == "reservation0600":
                        fixture.receipt.write_bytes(b"{}\n"); fixture.receipt.chmod(0o600)
                    elif mutation == "partial":
                        fixture.receipt.write_bytes(b'{"schema_version":'); fixture.receipt.chmod(0o400)
                    elif mutation != "missing":
                        fixture.write_receipt(mutation=mutation)
                    with self.assertRaises(builder.SnapshotError):
                        builder.build(
                            fixture.old, fixture.stage, fixture.target,
                            builder_sha256=fixture.builder_sha,
                        )
                    self.assertFalse(fixture.target.exists())
                    self.assertFalse(fixture.snapshot_receipt.exists())
                with fixture.patched_cpu() as (cpu_target, cpu_stage):
                    with self.assertRaises(cpu.CpuAdmissionError):
                        cpu.controller()
                    self.assertFalse(cpu_target.exists())
                    self.assertFalse(cpu_stage.exists())

    def test_materializer_accepts_exact35_then_rejects_0600_live_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = StageFixture(Path(temporary))
            with fixture.patched_builder():
                staging_receipt_raw = fixture.write_receipt()
            with fixture.patched_materializer() as staging_rows, ExitStack() as stack:
                fresh = set(fixture.fake_pins)
                release_raw = {
                    relative: (
                        (fixture.stage / relative).read_bytes()
                        if relative in fresh else f"legacy:{relative}\n".encode()
                    )
                    for relative in materializer.RELEASE_FILES
                }
                diagnostic_raw = {
                    relative: (fixture.stage / relative).read_bytes()
                    for relative in materializer.DIAGNOSTIC_FILES
                }
                authority_raw = {
                    relative: (fixture.stage / relative).read_bytes()
                    for relative in materializer.SNAPSHOT_AUTHORITY_FILES
                }
                release_pins = {relative: sha(raw) for relative, raw in release_raw.items()}
                diagnostic_pins = {relative: sha(raw) for relative, raw in diagnostic_raw.items()}
                authority_pins = {relative: sha(raw) for relative, raw in authority_raw.items()}
                stack.enter_context(mock.patch.object(materializer, "RELEASE_FILES", release_pins))
                stack.enter_context(mock.patch.object(materializer, "DIAGNOSTIC_FILES", diagnostic_pins))
                stack.enter_context(mock.patch.object(materializer, "SNAPSHOT_AUTHORITY_FILES", authority_pins))
                snapshot_root = fixture.base / "exact35"
                snapshot_receipt = fixture.base / "exact35.receipt_v2.json"
                stack.enter_context(mock.patch.object(
                    materializer, "SNAPSHOT_PUBLICATION_RECEIPT_PATH", snapshot_receipt,
                ))
                snapshot_root.mkdir()
                materializer_raw = Path(materializer.__file__).read_bytes()
                materializer_sha = sha(materializer_raw)
                base_raw = {
                    **release_raw, **diagnostic_raw, **authority_raw,
                    materializer.MATERIALIZER_RELATIVE: materializer_raw,
                }
                expected_base = {
                    **release_pins, **diagnostic_pins, **authority_pins,
                    materializer.MATERIALIZER_RELATIVE: materializer_sha,
                }
                staging_authority = materializer.replay_staging_receipt_authority(
                    staging_receipt_raw, staging_rows,
                )
                all_raw = {
                    **base_raw,
                    materializer.STAGING_RECEIPT_COPY_RELATIVE: staging_receipt_raw,
                }
                rows = []
                for relative in sorted(all_raw):
                    provenance = (
                        "copied_exact_auh_v2_staging_receipt_authority"
                        if relative == materializer.STAGING_RECEIPT_COPY_RELATIVE
                        else "independent_inode_copy_of_sealed_legacy_infer"
                        if relative == materializer.LEGACY_ALIAS_RELATIVE
                        else "sealed_legacy_exact5_snapshot"
                        if relative in materializer.LEGACY_REUSED_PATHS
                        else "fresh_pinned_staging"
                    )
                    mode = 0o400 if relative == materializer.STAGING_RECEIPT_COPY_RELATIVE else 0o444
                    rows.append({
                        "path": relative,
                        "sha256": sha(all_raw[relative]),
                        "size": len(all_raw[relative]),
                        "mode": mode, "provenance": provenance,
                    })
                    path = snapshot_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(all_raw[relative]); path.chmod(mode)
                manifest = {
                    "schema_version": "case01-object-trajectory-exact5-source-snapshot-v2",
                    "status": "SEALED_SOURCE_ONLY_NOT_LAUNCHABLE",
                    "launch_allowed": False,
                    "old_snapshot_root": str(materializer.OLD_EXACT5_SNAPSHOT),
                    "staging_root": str(fixture.stage),
                    "staging_receipt_path": str(fixture.receipt),
                    "snapshot_publication_receipt_path": str(snapshot_receipt),
                    "target_root": str(snapshot_root),
                    "content_leaf_count": 34,
                    "physical_file_count_including_manifest": 35,
                    "release_file_count": 25,
                    "legacy_alias_is_distinct_regular_inode": True,
                    "builder_authority": {
                        "path": str(fixture.stage / materializer.SNAPSHOT_BUILDER_RELATIVE),
                        "sha256": fixture.builder_sha,
                        "size": len(fixture.builder_raw),
                        "sealed_bytes_in_snapshot": False,
                    },
                    "staging_receipt_authority": staging_authority,
                    "publication_protocol": materializer.STAGING_PUBLICATION_PROTOCOL,
                    "rename_noreplace": False,
                    "cooperative_writer_exclusion": True,
                    "target_absent_rechecked": True,
                    "whole_tree_atomically_visible": True,
                    "uncooperative_same_uid_race_out_of_scope": True,
                    "retry_allowed": False,
                    "formal_review_test": materializer.FORMAL_REVIEW_TEST,
                    "files": rows,
                }
                manifest["manifest_digest"] = materializer.object_sha(manifest)
                manifest_raw = materializer.canonical(manifest) + b"\n"
                manifest_path = snapshot_root / materializer.SNAPSHOT_MANIFEST_NAME
                manifest_path.write_bytes(manifest_raw); manifest_path.chmod(0o444)
                seal_tree(snapshot_root)
                (snapshot_root / materializer.STAGING_RECEIPT_COPY_RELATIVE).chmod(0o400)
                root_identity = materializer._identity(snapshot_root.stat())
                descriptor = os.open(
                    snapshot_receipt, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600,
                )
                try:
                    publication = {
                        "schema_version": "case01-object-trajectory-exact5-source-snapshot-publication-v2-receipt",
                        "status": "PUBLISHED_RECEIPT_GATED",
                        "target_root": str(snapshot_root),
                        "receipt_path": str(snapshot_receipt),
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": sha(manifest_raw),
                        "manifest_digest": manifest["manifest_digest"],
                        "staging_receipt_sha256": sha(staging_receipt_raw),
                        "staging_receipt_digest": staging_authority["receipt_digest"],
                        "content_leaf_count": 34,
                        "physical_file_count_including_manifest": 35,
                        "publication_protocol": materializer.STAGING_PUBLICATION_PROTOCOL,
                        "rename_noreplace": False,
                        "cooperative_writer_exclusion": True,
                        "target_absent_rechecked_before_rename": True,
                        "ordinary_posix_rename_performed": True,
                        "publication_observation": {
                            "namespace_state": "target_same_inode_source_absent",
                            "rename_returned_zero": True,
                            "rename_error_errno": None,
                            "parent_fsync_returned_zero": True,
                            "parent_fsync_error_errno": None,
                        },
                        "whole_tree_atomically_visible": True,
                        "uncooperative_same_uid_race_out_of_scope": True,
                        "retry_allowed": False,
                        "target_root_identity": list(root_identity),
                        "receipt_mode": 0o400,
                        "receipt_is_consumption_gate": True,
                        "receipt_is_admission": True,
                        "launch_allowed": False,
                        "receipt_inode_anchor": materializer._inode_anchor(os.fstat(descriptor)),
                    }
                    publication["receipt_digest"] = materializer.object_sha(publication)
                    publication_raw = materializer.canonical(publication) + b"\n"
                    os.write(descriptor, publication_raw); os.fsync(descriptor)
                    os.fchmod(descriptor, 0o400); os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                raw_by_path, evidence = materializer.preflight_snapshot(
                    snapshot_root, manifest_sha256=sha(manifest_raw),
                    materializer_sha256=materializer_sha,
                    require_configured_root=False,
                )
                self.assertEqual(len(raw_by_path), 34)
                self.assertEqual(evidence["physical_file_count"], 35)
                fixture.receipt.chmod(0o600)
                with self.assertRaises(materializer.HoldPackageError):
                    materializer.preflight_snapshot(
                        snapshot_root, manifest_sha256=sha(manifest_raw),
                        materializer_sha256=materializer_sha,
                        require_configured_root=False,
                    )


class NfsPublicationTests(unittest.TestCase):
    def test_shadow_directory_fsync_failure_precedes_rename(self) -> None:
        for module in (builder, materializer):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary).resolve()
                target = parent / "target"; receipt = parent / "receipt.json"
                shadow = parent / "shadow"; (shadow / "nested").mkdir(parents=True)
                parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                reservation = module.create_publication_reservation(
                    parent_fd, receipt_path=receipt, target_root=target,
                )
                try:
                    with mock.patch.object(
                        module.os, "fsync",
                        side_effect=OSError(5, "synthetic shadow fsync"),
                    ), mock.patch.object(module.os, "rename") as renamed:
                        with self.assertRaises(OSError):
                            module.fsync_shadow_directories(shadow)
                    renamed.assert_not_called()
                    self.assertFalse(target.exists())
                    self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o600)
                finally:
                    reservation.close(); os.close(parent_fd)

    def test_single_ordinary_rename_and_final_0400_receipt(self) -> None:
        for module, error in (
            (builder, builder.SnapshotError),
            (materializer, materializer.HoldPackageError),
        ):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary).resolve()
                target = parent / "target"; receipt = parent / "receipt.json"
                shadow = parent / "shadow"; shadow.mkdir(); shadow.chmod(0o555)
                parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                reservation = module.create_publication_reservation(
                    parent_fd, receipt_path=receipt, target_root=target,
                )
                calls = 0
                real_rename = module.os.rename

                def counted(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    # Darwin's sandbox rejects renameat-style dirfd use in a
                    # temporary directory and rejects renaming a 0555 source.
                    # Preserve the one os.rename call while translating it to
                    # the same absolute parent; the transient chmod is only a
                    # host-test accommodation, never consumer behavior.
                    (parent / args[0]).chmod(0o755)
                    real_rename(parent / args[0], parent / args[1])
                    (parent / args[1]).chmod(0o555)

                try:
                    with mock.patch.object(module.os, "rename", side_effect=counted):
                        identity, observation = module.publish_under_reservation(
                            parent_fd, shadow.name, target.name, reservation,
                        )
                    self.assertEqual(calls, 1)
                    self.assertTrue(observation["parent_fsync_returned_zero"])
                    _raw, value = module.seal_publication_receipt(
                        parent_fd, reservation,
                        {"schema_version": "test", "status": "PASS",
                         "target_root": str(target),
                         "receipt_path": str(receipt),
                         "target_root_identity": list(identity)},
                    )
                    self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
                    anchor = (
                        module._receipt_inode_anchor(receipt.stat())
                        if module is builder else module._inode_anchor(receipt.stat())
                    )
                    self.assertEqual(value["receipt_inode_anchor"], anchor)
                    self.assertTrue(target.is_dir())
                    self.assertNotIn("renameat2", Path(module.__file__).read_text())
                finally:
                    reservation.close(); os.close(parent_fd)

    def test_applied_then_reported_error_is_classified_committed(self) -> None:
        for module in (builder, materializer):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary).resolve()
                target = parent / "target"; receipt = parent / "receipt.json"
                shadow = parent / "shadow"; shadow.mkdir(); shadow.chmod(0o555)
                parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                reservation = module.create_publication_reservation(
                    parent_fd, receipt_path=receipt, target_root=target,
                )
                real_rename = module.os.rename

                def applied_then_error(*args, **kwargs):
                    (parent / args[0]).chmod(0o755)
                    real_rename(parent / args[0], parent / args[1])
                    (parent / args[1]).chmod(0o555)
                    raise OSError(5, "synthetic NFS applied-then-error")

                try:
                    with mock.patch.object(module.os, "rename", side_effect=applied_then_error), \
                         self.assertRaises(module.PublicationCommittedError) as caught:
                        module.publish_under_reservation(
                            parent_fd, shadow.name, target.name, reservation,
                        )
                    self.assertTrue(target.is_dir())
                    self.assertFalse(caught.exception.observation["rename_returned_zero"])
                    self.assertEqual(
                        caught.exception.observation["namespace_state"],
                        "target_same_inode_source_absent",
                    )
                    module.seal_publication_receipt(
                        parent_fd, reservation,
                        {"schema_version": "test-terminal",
                         "status": "PUBLISHED_COMMIT_ERROR_NOT_ADMISSION",
                         "target_root": str(target),
                         "receipt_path": str(receipt),
                         "receipt_is_consumption_gate": False,
                         "target_root_identity": list(caught.exception.identity),
                         "publication_observation": caught.exception.observation},
                    )
                    self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
                finally:
                    reservation.close(); os.close(parent_fd)

    def test_post_commit_parent_fsync_error_is_classified_committed(self) -> None:
        for module in (builder, materializer):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary).resolve()
                target = parent / "target"; receipt = parent / "receipt.json"
                shadow = parent / "shadow"; shadow.mkdir(); shadow.chmod(0o555)
                parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                reservation = module.create_publication_reservation(
                    parent_fd, receipt_path=receipt, target_root=target,
                )
                real_rename = module.os.rename

                def same_parent_rename(*args, **kwargs):
                    (parent / args[0]).chmod(0o755)
                    real_rename(parent / args[0], parent / args[1])
                    (parent / args[1]).chmod(0o555)

                try:
                    with mock.patch.object(module.os, "rename", side_effect=same_parent_rename), \
                         mock.patch.object(module.os, "fsync", side_effect=OSError(5, "synthetic fsync")), \
                         self.assertRaises(module.PublicationCommittedError) as caught:
                        module.publish_under_reservation(
                            parent_fd, shadow.name, target.name, reservation,
                        )
                    self.assertTrue(target.is_dir())
                    self.assertTrue(caught.exception.observation["rename_returned_zero"])
                    self.assertFalse(
                        caught.exception.observation["parent_fsync_returned_zero"],
                    )
                    self.assertEqual(
                        caught.exception.observation["namespace_state"],
                        "target_same_inode_source_absent",
                    )
                finally:
                    reservation.close(); os.close(parent_fd)

    def test_postseal_parent_fsync_error_keeps_immutable_0400(self) -> None:
        for module in (builder, materializer):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary).resolve()
                target = parent / "target"; receipt = parent / "receipt.json"
                target.mkdir(); target.chmod(0o555)
                identity = (
                    module.ident(target.stat())
                    if module is builder else module._identity(target.stat())
                )
                parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                reservation = module.create_publication_reservation(
                    parent_fd, receipt_path=receipt, target_root=target,
                )
                real_fsync = module.os.fsync

                def fail_parent(descriptor: int) -> None:
                    if descriptor == parent_fd:
                        raise OSError(5, "synthetic final receipt parent fsync")
                    real_fsync(descriptor)

                try:
                    with mock.patch.object(module.os, "fsync", side_effect=fail_parent):
                        _raw, value = module.seal_publication_receipt(
                            parent_fd, reservation,
                            {"schema_version": "test", "status": "PASS",
                             "target_root": str(target),
                             "receipt_path": str(receipt),
                             "target_root_identity": list(identity),
                             "receipt_is_consumption_gate": True,
                             "receipt_is_admission": True},
                        )
                    self.assertEqual(value["status"], "PASS")
                    self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
                    self.assertTrue(target.exists())
                finally:
                    reservation.close(); os.close(parent_fd)

    def test_fchmod_applied_then_error_keeps_immutable_0400(self) -> None:
        for module in (builder, materializer):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary).resolve()
                target = parent / "target"; receipt = parent / "receipt.json"
                target.mkdir(); target.chmod(0o555)
                identity = (
                    module.ident(target.stat())
                    if module is builder else module._identity(target.stat())
                )
                parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                reservation = module.create_publication_reservation(
                    parent_fd, receipt_path=receipt, target_root=target,
                )
                real_fchmod = module.os.fchmod

                def applied_then_error(descriptor: int, mode: int) -> None:
                    real_fchmod(descriptor, mode)
                    if mode == module.PUBLICATION_RECEIPT_MODE:
                        raise OSError(5, "synthetic applied fchmod error")

                try:
                    with mock.patch.object(module.os, "fchmod", side_effect=applied_then_error):
                        _raw, value = module.seal_publication_receipt(
                            parent_fd, reservation,
                            {"schema_version": "test", "status": "PASS",
                             "target_root": str(target),
                             "receipt_path": str(receipt),
                             "target_root_identity": list(identity),
                             "receipt_is_consumption_gate": True,
                             "receipt_is_admission": True},
                        )
                    self.assertEqual(value["status"], "PASS")
                    self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)
                finally:
                    reservation.close(); os.close(parent_fd)


if __name__ == "__main__":
    unittest.main()
