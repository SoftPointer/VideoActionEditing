from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

from methods.bernini_action_editing import (
    recover_v4g_scientific_no_go_attestation_v1 as recovery,
)


class RecoveryAttestationTests(unittest.TestCase):
    def test_released_guard_is_first_and_source_has_no_assert(self) -> None:
        source = Path(recovery.__file__).read_text()
        tree = ast.parse(source)
        self.assertEqual(sum(isinstance(node, ast.Assert) for node in ast.walk(tree)), 0)
        self.assertTrue(recovery.RELEASE_SEALED)
        self.assertNotIn("renameat2", source)
        self.assertNotIn("RENAME_NOREPLACE", source)
        with mock.patch.object(recovery, "RELEASE_SEALED", False):
            with mock.patch.object(recovery, "_parser", side_effect=AssertionError("parsed")):
                with self.assertRaisesRegex(SystemExit, "intentional NO-GO"):
                    recovery.main([])
            with self.assertRaisesRegex(RuntimeError, "intentional NO-GO"):
                recovery.recover(recovery.ORIGINAL_RUN_ROOT, recovery.RECOVERY_ROOT)
            with self.assertRaisesRegex(RuntimeError, "intentional NO-GO"):
                recovery._publish_attestation_create_only(recovery.RECOVERY_ROOT)

    def test_canonical_json_rejects_duplicate_and_noncanonical(self) -> None:
        value = {"a": 1, "b": [True, None]}
        raw = recovery._canonical_bytes(value) + b"\n"
        self.assertEqual(recovery._json_no_duplicates(raw), value)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            recovery._json_no_duplicates(b'{"a":1,"a":2}\n')
        with self.assertRaisesRegex(RuntimeError, "not canonical"):
            recovery._json_no_duplicates(b'{"b":[], "a":1}\n')

    def test_scan_original_exact_manifest_and_hostile_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve() / "run"
            root.mkdir(mode=0o700)
            (root / "fold0").mkdir(mode=0o700)
            (root / "fold0/a").write_bytes(b"alpha")
            os.chmod(root / "fold0/a", 0o444)
            binding, _ = recovery._read_regular(root / "fold0/a", mode=0o444, nlink=1)
            rows = [{
                "path": "fold0/a", "sha256": binding["sha256"],
                "size_bytes": binding["size_bytes"],
                "mode_octal": "0444", "nlink": 1,
                "device": binding["device"], "inode": binding["inode"],
            }]
            stable_signature = recovery._object_sha({
                "directories": [{"path": "fold0", "mode_octal": "0700"}],
                "files": [{
                    key: rows[0][key] for key in (
                        "path", "sha256", "size_bytes", "mode_octal", "nlink",
                    )
                }],
            })
            with mock.patch.object(
                recovery, "EXPECTED_DIRS", {"fold0"}
            ), mock.patch.object(
                recovery, "EXPECTED_FILES", {"fold0/a"}
            ), mock.patch.object(
                recovery, "EXACT26_MANIFEST_SHA256", recovery._object_sha(rows)
            ), mock.patch.object(
                recovery, "EXPECTED_FILE_COUNT", 1
            ), mock.patch.object(
                recovery, "PARENT_STABLE_SIGNATURE_SHA256", stable_signature
            ):
                actual, bindings = recovery._scan_original(root)
                self.assertEqual(actual, rows)
                self.assertEqual(bindings["fold0/a"]["sha256"], binding["sha256"])
                (root / "extra").write_bytes(b"x")
                os.chmod(root / "extra", 0o444)
                with self.assertRaisesRegex(RuntimeError, "exact directory/file set"):
                    recovery._scan_original(root)

    def test_scan_rejects_symlink_hardlink_and_wrong_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            target = root / "target"
            target.write_bytes(b"payload")
            os.chmod(target, 0o444)
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "canonical absolute/non-symlink"):
                recovery._read_regular(link, mode=0o444, nlink=1)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            nested = real_parent / "nested"
            nested.write_bytes(b"nested")
            os.chmod(nested, 0o444)
            alias_parent = root / "alias-parent"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "canonical absolute/non-symlink"):
                recovery._read_regular(alias_parent / "nested", mode=0o444, nlink=1)
            hard = root / "hard"
            os.link(target, hard)
            with self.assertRaisesRegex(RuntimeError, "nlink differs"):
                recovery._read_regular(target, mode=0o444, nlink=1)
            hard.unlink()
            os.chmod(target, 0o644)
            with self.assertRaisesRegex(RuntimeError, "mode differs"):
                recovery._read_regular(target, mode=0o444, nlink=1)

    @staticmethod
    def _binding(
        path: Path, sha256: str, *, size: int = 17, mode: str = "0444",
        nlink: int = 1, device: int = 48, inode: int = 100,
    ) -> dict[str, object]:
        return {
            "path": str(path), "sha256": sha256, "size_bytes": size,
            "mode_octal": mode, "nlink": nlink, "device": device,
            "inode": inode, "mtime_ns": 1000 + inode,
            "ctime_ns": 2000 + inode,
            "single_fd_pre_post_identity_and_sha_exact": True,
        }

    @classmethod
    def _publish_fixture(
        cls, original: Path, root: Path,
    ) -> tuple[
        dict[str, object], list[dict[str, object]],
        dict[str, dict[str, object]], str, str,
    ]:
        rows: list[dict[str, object]] = []
        bindings: dict[str, dict[str, object]] = {}
        for index, relative in enumerate(sorted(recovery.EXPECTED_FILES)):
            raw = f"burned:{relative}".encode()
            row = {
                "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw), "mode_octal": "0444", "nlink": 1,
                "device": 48, "inode": 10000 + index,
            }
            rows.append(row)
            bindings[relative] = {
                **row, "path": str(original / relative),
                "mtime_ns": 3000 + index, "ctime_ns": 4000 + index,
                "single_fd_pre_post_identity_and_sha_exact": True,
            }
        manifest_sha = recovery._object_sha(rows)
        folds = []
        counts = ((400, 113, 131), (402, 115, 127), (401, 115, 128),
                  (403, 112, 129), (403, 112, 129))
        by_path = {row["path"]: row for row in rows}
        for fold, (model_fit, inner, oof) in enumerate(counts):
            pre = by_path[f"fold{fold}/preselection.pt"]
            fixed = by_path[f"fold{fold}/fixed1200.pt"]
            folds.append({
                "fold_index": fold,
                "inner_receipt_sha256": recovery.INNER_RECEIPT_SHA256[fold],
                "inner_receipt_digest": recovery.INNER_RECEIPT_DIGEST[fold],
                "inner_status": "V4G_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD",
                "inner_pass": False,
                "oof_semantic_tensor_read_count": 0,
                "oof_semantic_tensor_materialized_count": 0,
                "model_fit_count": model_fit, "inner_count": inner,
                "oof_count": oof,
                "preselection_checkpoint": {
                    "sha256": pre["sha256"], "size_bytes": pre["size_bytes"],
                    "device": pre["device"], "inode": pre["inode"],
                    "metadata_digest": hashlib.sha256(f"pre{fold}".encode()).hexdigest(),
                },
                "fixed1200_checkpoint": {
                    "sha256": fixed["sha256"], "size_bytes": fixed["size_bytes"],
                    "device": fixed["device"], "inode": fixed["inode"],
                    "metadata_digest": hashlib.sha256(f"fixed{fold}".encode()).hexdigest(),
                    "model_state_sha256": hashlib.sha256(f"state{fold}".encode()).hexdigest(),
                },
                "three_field_runtime_physical_identity_projection_exact": True,
                "mode_and_nlink_verified_separately": True,
                "fidelity_gate": False,
                "all_three_negative_full_gates": False,
                "complete_gate": False,
            })

        release_manifest = cls._binding(
            recovery.RELEASE_ROOT / "release-manifest-v4g.json",
            recovery.RELEASE_MANIFEST_SHA256, inode=201,
        )
        controller = cls._binding(
            recovery.CONTROLLER_PATH, recovery.CONTROLLER_SHA256,
            mode="0555", inode=202,
        )
        python = cls._binding(
            recovery.PYTHON_PATH, recovery.PYTHON_SHA256,
            mode="0755", inode=203,
        )
        input_receipts = [
            cls._binding(path, sha256, inode=210 + index)
            for index, (path, sha256) in enumerate(recovery.AUTHORITY_FILES)
        ]
        shard_bindings = [
            cls._binding(
                Path(f"/authority/feature-shard-{index}.pt"),
                hashlib.sha256(f"shard{index}".encode()).hexdigest(),
                inode=240 + index,
            )
            for index in range(6)
        ]
        input_rows = [{
            "label": ("feature", "v4a", "v4c", "v4d")[index],
            "path": binding["path"], "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        } for index, binding in enumerate(input_receipts)]
        input_rows.extend({
            "label": f"feature-shard-{index}", "path": binding["path"],
            "sha256": binding["sha256"], "size_bytes": binding["size_bytes"],
        } for index, binding in enumerate(shard_bindings))
        input_sha = recovery._object_sha(input_rows)
        stdout = cls._binding(
            recovery.OUTER_STDOUT_PATH, recovery.OUTER_STDOUT["sha256"],
            size=recovery.OUTER_STDOUT["size_bytes"], mode="0600",
            device=recovery.OUTER_STDOUT["device"], inode=recovery.OUTER_STDOUT["inode"],
        )
        stderr = cls._binding(
            recovery.OUTER_STDERR_PATH, recovery.OUTER_STDERR["sha256"],
            size=recovery.OUTER_STDERR["size_bytes"], mode="0600",
            device=recovery.OUTER_STDERR["device"], inode=recovery.OUTER_STDERR["inode"],
        )
        sacct = cls._binding(
            recovery.SACCT_PATH, recovery.SACCT_SHA256, mode="0755", inode=220,
        )
        recovery_manifest = cls._binding(
            recovery.RECOVERY_RELEASE_ROOT / recovery.RECOVERY_MANIFEST_NAME,
            "b" * 64, inode=301,
        )
        recovery_runtime = cls._binding(
            recovery.RECOVERY_RELEASE_ROOT / recovery.RECOVERY_RUNTIME_RELATIVE_PATH,
            "d" * 64, inode=302,
        )
        recovery_tests = cls._binding(
            recovery.RECOVERY_RELEASE_ROOT / recovery.RECOVERY_TESTS_RELATIVE_PATH,
            "e" * 64, inode=303,
        )
        recovery_controller = cls._binding(
            recovery.RECOVERY_CONTROLLER_PATH, "f" * 64,
            mode="0555", inode=304,
        )
        recovery_tree_rows = [{
            "path": relative, "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        } for relative, binding in sorted({
            recovery.RECOVERY_MANIFEST_NAME: recovery_manifest,
            recovery.RECOVERY_RUNTIME_RELATIVE_PATH: recovery_runtime,
            recovery.RECOVERY_TESTS_RELATIVE_PATH: recovery_tests,
        }.items())]
        value: dict[str, object] = {
            "schema_version": recovery.SCHEMA,
            "authority": "burned_known_transform_development_scientific_no_go_only",
            "original_run_root": str(original),
            "recovery_root": str(root),
            "original_run_mutated_by_recovery": False,
            "original_run_postverified_unchanged": True,
            "original_controller_complete": False,
            "original_controller_exit_nonzero": True,
            "scientifically_verified_all_inner_no_go": True,
            "global_inner_barrier_created": False,
            "evaluate_fold_executed": False,
            "aggregate_executed": False,
            "all_fold_oof_semantic_tensor_read_count": 0,
            "all_fold_oof_semantic_tensor_materialized_count": 0,
            "recovery_ledger_reconstructed_from_burned_exact26": True,
            "burned_exact26_file_count": 26,
            "burned_exact26_manifest_sha256": manifest_sha,
            "burned_exact26_manifest": rows,
            "burned_parent_stable_signature_sha256": recovery.PARENT_STABLE_SIGNATURE_SHA256,
            "original_run_root_binding": {
                "path": str(original), "mode_octal": "0700", "nlink": 8,
                "device": 48, "inode": 900, "mtime_ns": 901,
                "ctime_ns": 902,
                "members": [
                    "fold0", "fold1", "fold2", "fold3", "fold4",
                    "launch-plan.json", "logs",
                ],
                "single_fd_pre_post_identity_and_membership_exact": True,
            },
            "old_controller_identity_schema_bug": {
                "runtime_receipt_identity_keys": ["device", "inode", "size_bytes"],
                "phase_binding_additional_keys": [
                    "mode_octal", "nlink", "mtime_ns", "ctime_ns",
                ],
                "all_ten_checkpoint_three_field_projections_exact": True,
                "all_ten_checkpoint_full_object_equal": False,
                "mode_and_nlink_verified_independently": True,
                "bug_affected_scientific_values": False,
                "bug_only_blocked_final_controller_seal": True,
            },
            "source_authority": {
                "authority_snapshot": recovery.AUTHORITY_SNAPSHOT,
                "release": {
                    "root": str(recovery.RELEASE_ROOT),
                    "tree_sha256": recovery.RELEASE_TREE_SHA256,
                    "manifest": release_manifest,
                    "manifest_digest": recovery.RELEASE_MANIFEST_DIGEST,
                    "file_count": 11, "directory_count": 3,
                },
                "controller": controller, "python": python,
                "process_python": dict(python), "input_receipts": input_receipts,
                "recovery_parser_torch": {
                    "torch_version": "2.7.1+rocm6.3",
                    "torch_hip_version": "6.3.test",
                    "torch_version_exact_2_7_1_rocm6_3": True,
                    "torch_hip_release_6_3": True,
                    "torch_package_root": str(
                        recovery.PYTHON_SITE_PACKAGES / "torch"
                    ),
                    "torch_module_origins": {
                        "torch": str(
                            recovery.PYTHON_SITE_PACKAGES / "torch/__init__.py"
                        ),
                        "torch.nn": str(
                            recovery.PYTHON_SITE_PACKAGES / "torch/nn/__init__.py"
                        ),
                        "torch.nn.functional": str(
                            recovery.PYTHON_SITE_PACKAGES
                            / "torch/nn/functional.py"
                        ),
                    },
                    "v4g_torch_module_identities_exact": True,
                    "torch_standard_source_loaders_and_package_path_exact": True,
                },
                "input_snapshot": {
                    "sha256": input_sha, "ordered_rows": input_rows,
                    "feature_shard_bindings": shard_bindings,
                    "exact_receipt_count": 4, "exact_feature_shard_count": 6,
                    "all_ten_files_single_fd_reverified": True,
                },
                "runtime_sha256": recovery.RUNTIME_SHA256,
                "tests_sha256": recovery.TESTS_SHA256,
                "recovery": {
                    "release_root": str(recovery.RECOVERY_RELEASE_ROOT),
                    "release_root_binding": {
                        "path": str(recovery.RECOVERY_RELEASE_ROOT),
                        "mode_octal": "0555", "nlink": 4,
                        "device": 48, "inode": 300, "mtime_ns": 301,
                        "ctime_ns": 302,
                        "members": ["methods", recovery.RECOVERY_MANIFEST_NAME],
                        "single_fd_pre_post_identity_and_membership_exact": True,
                    },
                    "release_tree_sha256": recovery._object_sha(recovery_tree_rows),
                    "tree_rows": recovery_tree_rows,
                    "manifest": recovery_manifest,
                    "manifest_digest": "c" * 64,
                    "runtime": recovery_runtime,
                    "tests": recovery_tests,
                    "controller": recovery_controller,
                    "exact_file_count": 3,
                    "exact_directory_count_below_root": 3,
                    "one_way_sha256_dag_reverified": True,
                    "controller_identity_recorded_not_runtime_reverse_pinned": True,
                },
            },
            "launch_plan": {
                "binding": cls._binding(
                    original / "launch-plan.json", recovery.LAUNCH_PLAN_SHA256,
                    size=3298, inode=230,
                ),
                "schema_version": "v4g-launch-plan-v1",
                "normal_tests_run": 36, "normal_tests_skipped": 0,
                "optimized_tests_run": 36, "optimized_tests_skipped": 0,
            },
            "original_controller_logs": {"stdout": stdout, "stderr": stderr},
            "failed_seal_child_accounting": {
                "record": {
                    "job_id_raw": "143811.94", "state": "FAILED",
                    "exit_code": "1:0", "node": "auh7-1b-gpu-306",
                    "alloc_cpus": "8", "req_mem": "", "elapsed": "00:00:03",
                    "start": "2026-08-21T09:43:49",
                    "end": "2026-08-21T09:43:52",
                },
                "sacct_executable": sacct,
                "query_columns": [
                    "job_id_raw", "state", "exit_code", "node", "alloc_cpus",
                    "req_mem", "elapsed", "start", "end",
                ],
                "exact_row_replayed": True,
            },
            "folds": folds,
            "all_qualification_claims_false": True,
            "qualification_scope": dict(
                recovery.EXPECTED_ATTESTATION_QUALIFICATION_SCOPE
            ),
        }
        return value, rows, bindings, manifest_sha, input_sha

    def _create_only_context(self, original: Path, root: Path):
        value, rows, bindings, manifest_sha, input_sha = self._publish_fixture(
            original, root,
        )
        patches = (
            mock.patch.object(recovery, "RELEASE_SEALED", True),
            mock.patch.object(recovery, "ORIGINAL_RUN_ROOT", original),
            mock.patch.object(recovery, "RECOVERY_ROOT", root),
            mock.patch.object(recovery, "TRUSTED_PATH_ANCHOR", root.parent),
            mock.patch.object(recovery, "EXACT26_MANIFEST_SHA256", manifest_sha),
            mock.patch.object(recovery, "INPUT_SNAPSHOT_SHA256", input_sha),
            mock.patch.object(
                recovery, "_collect_verified_attestation",
                return_value=(value, rows, bindings),
            ),
            mock.patch.object(
                recovery, "_scan_original", return_value=(rows, bindings),
            ),
            mock.patch.object(
                recovery, "_read_directory_binding",
                return_value=value["original_run_root_binding"],
            ),
            mock.patch.object(
                recovery, "_reverify_prepublication_authorities",
                return_value=None,
            ),
        )
        return value, rows, bindings, patches

    def test_create_only_commit_exact1_and_self_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            _, _, _, patches = self._create_only_context(original, root)
            for patch in patches:
                patch.start()
            try:
                result = recovery._publish_attestation_create_only(root)
            finally:
                for patch in reversed(patches):
                    patch.stop()
            self.assertEqual(result["exact_file_count"], 1)
            self.assertEqual(
                result["schema_version"],
                "v4g-scientific-no-go-sibling-recovery-result-v4",
            )
            self.assertEqual(set(result), {
                "path", "file_sha256", "size_bytes", "receipt_digest",
                "mode_octal", "nlink", "root_mode_octal",
                "exact_file_count", "create_only_name_claim",
                "failure_tombstone_root_mode_octal",
                "original_run_and_source_authorities_reverified_after_name_claim",
                "root_and_file_same_fd_precommit_verified_and_parent_fsynced",
                "producer_root_precommit_binding",
                "producer_attestation_final_binding",
                "root_creation_to_precommit_device_inode_exact",
                "file_creation_to_final_device_inode_exact",
                "final_mode_commit", "final_mode_commit_order",
                "schema_version", "original_run_postverified_unchanged",
                "original_run_exact26_manifest_sha256", "scientific_result",
                "original_controller_complete",
            })
            self.assertTrue(result["create_only_name_claim"])
            self.assertTrue(result["final_mode_commit"])
            self.assertTrue(
                result["root_creation_to_precommit_device_inode_exact"]
            )
            self.assertTrue(
                result["file_creation_to_final_device_inode_exact"]
            )
            self.assertEqual(
                result["final_mode_commit_order"],
                ["file_0444", "root_0555"],
            )
            self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o555)
            path = root / "recovery-attestation.json"
            self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o444)
            self.assertEqual(path.lstat().st_nlink, 1)
            loaded = recovery._json_no_duplicates(path.read_bytes())
            digest = loaded.pop("receipt_digest")
            self.assertEqual(digest, recovery._object_sha(loaded))
            root_binding = result["producer_root_precommit_binding"]
            self.assertEqual(set(root_binding), {
                "path", "mode_octal", "nlink", "device", "inode",
                "mtime_ns", "ctime_ns", "members",
                "single_fd_pre_post_identity_and_membership_exact",
            })
            self.assertEqual(root_binding["path"], str(root))
            self.assertEqual(str(root.resolve(strict=True)), str(root))
            self.assertEqual(root_binding["mode_octal"], "0700")
            self.assertEqual(
                root_binding["members"], ["recovery-attestation.json"],
            )
            self.assertTrue(
                root_binding[
                    "single_fd_pre_post_identity_and_membership_exact"
                ]
            )
            current_root = root.lstat()
            self.assertEqual(
                (root_binding["device"], root_binding["inode"],
                 root_binding["nlink"]),
                (current_root.st_dev, current_root.st_ino, current_root.st_nlink),
            )
            file_binding = result["producer_attestation_final_binding"]
            self.assertEqual(set(file_binding), {
                "path", "sha256", "size_bytes", "mode_octal", "nlink",
                "device", "inode", "mtime_ns", "ctime_ns",
                "single_fd_pre_post_identity_and_sha_exact",
            })
            self.assertEqual(file_binding["path"], str(path))
            self.assertEqual(file_binding["mode_octal"], "0444")
            self.assertEqual(file_binding["nlink"], 1)
            self.assertEqual(file_binding["sha256"], result["file_sha256"])
            self.assertEqual(file_binding["size_bytes"], result["size_bytes"])
            self.assertTrue(
                file_binding["single_fd_pre_post_identity_and_sha_exact"]
            )
            current_file = path.lstat()
            self.assertEqual(
                (
                    file_binding["device"], file_binding["inode"],
                    file_binding["size_bytes"], file_binding["nlink"],
                    file_binding["mtime_ns"], file_binding["ctime_ns"],
                ),
                (
                    current_file.st_dev, current_file.st_ino,
                    current_file.st_size, current_file.st_nlink,
                    current_file.st_mtime_ns, current_file.st_ctime_ns,
                ),
            )
            with mock.patch.object(recovery, "RELEASE_SEALED", True), mock.patch.object(
                recovery, "RECOVERY_ROOT", root,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "path/capabilities|not fresh",
                ):
                    recovery._publish_attestation_create_only(root)

    def test_postwrite_authority_failure_leaves_nonaccepted_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            _, _, _, patches = self._create_only_context(original, root)
            for patch in patches:
                patch.start()
            try:
                with mock.patch.object(
                    recovery, "_reverify_prepublication_authorities",
                    side_effect=RuntimeError("source recheck failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "source recheck failed"):
                        recovery._publish_attestation_create_only(root)
                with self.assertRaisesRegex(
                    RuntimeError, "path/capabilities|not fresh",
                ):
                    recovery._publish_attestation_create_only(root)
            finally:
                for patch in reversed(patches):
                    patch.stop()
            self.assertTrue(os.path.lexists(root))
            self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)
            path = root / "recovery-attestation.json"
            self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0)
            self.assertEqual(path.lstat().st_nlink, 1)

    def test_original_final_postscan_change_leaves_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            _, rows, bindings, patches = self._create_only_context(original, root)
            for patch in patches:
                patch.start()
            try:
                with mock.patch.object(
                    recovery, "_scan_original",
                    side_effect=[
                        (rows, bindings),
                        (rows + [{"path": "hostile"}], bindings),
                    ],
                ):
                    with self.assertRaisesRegex(RuntimeError, "before final mode"):
                        recovery._publish_attestation_create_only(root)
            finally:
                for patch in reversed(patches):
                    patch.stop()
            self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE((root / "recovery-attestation.json").lstat().st_mode),
                0,
            )

    def test_publisher_rejects_forged_nested_payload_and_has_no_authority_seams(self) -> None:
        signature = inspect.signature(recovery._publish_attestation_create_only)
        self.assertEqual(list(signature.parameters), ["root"])
        collect_signature = inspect.signature(recovery._collect_verified_attestation)
        self.assertEqual(list(collect_signature.parameters), ["original_root", "recovery_root"])
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            value, _, _, manifest_sha, input_sha = self._publish_fixture(
                original, root,
            )
            with mock.patch.object(recovery, "ORIGINAL_RUN_ROOT", original), mock.patch.object(
                recovery, "EXACT26_MANIFEST_SHA256", manifest_sha,
            ), mock.patch.object(
                recovery, "INPUT_SNAPSHOT_SHA256", input_sha,
            ):
                forged = dict(value)
                forged["source_authority"] = {}
                with self.assertRaisesRegex(RuntimeError, "schema/claims"):
                    recovery._validate_publish_value(root, forged)
                forged = dict(value)
                forged["qualification_scope"] = {
                    **recovery.EXPECTED_ATTESTATION_QUALIFICATION_SCOPE,
                    "forged_qualified": True,
                }
                with self.assertRaisesRegex(RuntimeError, "schema/claims"):
                    recovery._validate_publish_value(root, forged)

    def test_prepublication_authority_reverify_accepts_exact_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            value, _, _, _, _ = self._publish_fixture(original, root)
            source = value["source_authority"]
            target_module = (
                "methods.bernini_action_editing."
                "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g"
            )

            def reader(path: Path, **kwargs):
                capture = kwargs.get("capture", False)
                if path == recovery.CONTROLLER_PATH:
                    binding = source["controller"]
                elif path == recovery.PYTHON_PATH:
                    binding = source["python"]
                elif path == recovery.OUTER_STDOUT_PATH:
                    binding = value["original_controller_logs"]["stdout"]
                elif path == recovery.OUTER_STDERR_PATH:
                    binding = value["original_controller_logs"]["stderr"]
                else:
                    index = [item[0] for item in recovery.AUTHORITY_FILES].index(path)
                    binding = source["input_receipts"][index]
                return binding, (b"captured\n" if capture else None)

            patches = (
                mock.patch.dict(sys.modules, {target_module: object()}),
                mock.patch.object(
                    recovery, "_verify_release", return_value=source["release"],
                ),
                mock.patch.object(
                    recovery, "_verify_recovery_release_and_controller",
                    return_value=source["recovery"],
                ),
                mock.patch.object(recovery, "_read_regular", side_effect=reader),
                mock.patch.object(
                    recovery, "_verify_process_python",
                    return_value=source["process_python"],
                ),
                mock.patch.object(
                    recovery, "_verify_recovery_parser_torch",
                    return_value={
                        key: source["recovery_parser_torch"][key]
                        for key in (
                            "torch_version", "torch_hip_version",
                            "torch_version_exact_2_7_1_rocm6_3",
                            "torch_hip_release_6_3",
                        )
                    },
                ),
                mock.patch.object(
                    recovery, "_verify_recovery_parser_torch_origins",
                    return_value={
                        key: source["recovery_parser_torch"][key]
                        for key in (
                            "torch_package_root", "torch_module_origins",
                            "v4g_torch_module_identities_exact",
                            "torch_standard_source_loaders_and_package_path_exact",
                        )
                    },
                ),
                mock.patch.object(
                    recovery, "_verify_input_snapshot",
                    return_value=source["input_snapshot"],
                ),
                mock.patch.object(
                    recovery, "_query_failed_step",
                    return_value=value["failed_seal_child_accounting"],
                ),
            )
            for patch in patches:
                patch.start()
            try:
                recovery._reverify_prepublication_authorities(value)
                with mock.patch.object(
                    recovery, "_verify_release", return_value={},
                ):
                    with self.assertRaisesRegex(RuntimeError, "historical release"):
                        recovery._reverify_prepublication_authorities(value)
            finally:
                for patch in reversed(patches):
                    patch.stop()

    def test_preoccupied_final_root_is_unchanged(self) -> None:
        for occupied_kind in ("file", "directory", "symlink"):
            with self.subTest(occupied_kind=occupied_kind), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp).resolve()
                original = parent / "original"
                original.mkdir(mode=0o700)
                root = parent / "recovery"
                target = parent / "target"
                if occupied_kind == "file":
                    root.write_bytes(b"occupied-root\n")
                    os.chmod(root, 0o600)
                    expected = (root.read_bytes(), stat.S_IMODE(root.lstat().st_mode))
                elif occupied_kind == "directory":
                    root.mkdir(mode=0o700)
                    (root / "marker").write_bytes(b"occupied-directory\n")
                    expected = (root.joinpath("marker").read_bytes(), sorted(os.listdir(root)))
                else:
                    target.mkdir(mode=0o700)
                    root.symlink_to(target, target_is_directory=True)
                    expected = os.readlink(root)
                _, _, _, patches = self._create_only_context(original, root)
                for patch in patches:
                    patch.start()
                try:
                    with self.assertRaisesRegex(RuntimeError, "path/capabilities"):
                        recovery._publish_attestation_create_only(root)
                finally:
                    for patch in reversed(patches):
                        patch.stop()
                if occupied_kind == "file":
                    self.assertEqual(
                        (root.read_bytes(), stat.S_IMODE(root.lstat().st_mode)),
                        expected,
                    )
                elif occupied_kind == "directory":
                    self.assertEqual(
                        (root.joinpath("marker").read_bytes(), sorted(os.listdir(root))),
                        expected,
                    )
                else:
                    self.assertTrue(root.is_symlink())
                    self.assertEqual(os.readlink(root), expected)

    def test_preoccupied_final_file_name_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            _, _, _, patches = self._create_only_context(original, root)
            real_join = recovery._require_child_directory_fd
            inserted = False

            def occupy_name(parent_fd: int, name: str, descriptor: int, *, mode: int):
                nonlocal inserted
                result = real_join(parent_fd, name, descriptor, mode=mode)
                if name == root.name and not inserted:
                    inserted = True
                    marker_fd = os.open(
                        "recovery-attestation.json",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=descriptor,
                    )
                    try:
                        os.write(marker_fd, b"occupied-file\n")
                    finally:
                        os.close(marker_fd)
                return result

            for patch in patches:
                patch.start()
            try:
                with mock.patch.object(
                    recovery, "_require_child_directory_fd",
                    side_effect=occupy_name,
                ):
                    with self.assertRaisesRegex(RuntimeError, "final name is not fresh"):
                        recovery._publish_attestation_create_only(root)
            finally:
                for patch in reversed(patches):
                    patch.stop()
            marker = root / "recovery-attestation.json"
            self.assertEqual(marker.read_bytes(), b"occupied-file\n")
            self.assertEqual(stat.S_IMODE(marker.lstat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)

    def test_precommit_extra_symlink_and_new_inode_are_rejected(self) -> None:
        for attack in ("extra", "symlink", "new_inode"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp).resolve()
                original = parent / "original"
                original.mkdir(mode=0o700)
                root = parent / "recovery"
                target = parent / "target"
                target.write_bytes(b"target\n")
                _, _, _, patches = self._create_only_context(original, root)

                def hostile_authority(_value: object) -> None:
                    path = root / "recovery-attestation.json"
                    if attack == "extra":
                        (root / "extra").write_bytes(b"extra\n")
                    elif attack == "symlink":
                        path.unlink()
                        path.symlink_to(target)
                    else:
                        os.chmod(path, 0o600)
                        raw = path.read_bytes()
                        path.unlink()
                        path.write_bytes(raw)
                        os.chmod(path, 0)

                for patch in patches:
                    patch.start()
                try:
                    with mock.patch.object(
                        recovery, "_reverify_prepublication_authorities",
                        side_effect=hostile_authority,
                    ):
                        with self.assertRaises(RuntimeError):
                            recovery._publish_attestation_create_only(root)
                finally:
                    for patch in reversed(patches):
                        patch.stop()
                self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)

    def test_same_bytes_new_inode_before_final_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            _, _, _, patches = self._create_only_context(original, root)
            real_fchmod = os.fchmod
            replaced = False
            old_inode = 0

            def replace_after_file_commit(descriptor: int, mode: int) -> None:
                nonlocal replaced, old_inode
                real_fchmod(descriptor, mode)
                if mode == 0o444 and not replaced:
                    replaced = True
                    path = root / "recovery-attestation.json"
                    old_inode = path.lstat().st_ino
                    raw = path.read_bytes()
                    path.unlink()
                    path.write_bytes(raw)
                    os.chmod(path, 0o444)

            for patch in patches:
                patch.start()
            try:
                with mock.patch.object(
                    recovery.os, "fchmod", side_effect=replace_after_file_commit,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "creation-to-final identity",
                    ):
                        recovery._publish_attestation_create_only(root)
            finally:
                for patch in reversed(patches):
                    patch.stop()
            path = root / "recovery-attestation.json"
            self.assertTrue(replaced)
            self.assertNotEqual(path.lstat().st_ino, old_inode)
            self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)

    def test_parent_symlink_or_root_inode_swap_is_rejected(self) -> None:
        for attack in ("root_inode", "parent_symlink"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temp:
                trusted = Path(temp).resolve()
                parent = trusted / "runs"
                parent.mkdir(mode=0o700)
                original = parent / "original"
                original.mkdir(mode=0o700)
                root = parent / "recovery"
                attacker = trusted / "attacker"
                attacker.mkdir(mode=0o700)
                _, _, _, patches = self._create_only_context(original, root)

                def hostile_authority(_value: object) -> None:
                    if attack == "root_inode":
                        os.rename(root, parent / "recovery.held")
                        root.mkdir(mode=0o700)
                        (root / "marker").write_bytes(b"replacement\n")
                    else:
                        os.rename(parent, trusted / "runs.held")
                        parent.symlink_to(attacker, target_is_directory=True)
                        forged = attacker / root.name
                        forged.mkdir(mode=0o700)
                        (forged / "marker").write_bytes(b"replacement\n")

                for patch in patches:
                    patch.start()
                try:
                    with mock.patch.object(
                        recovery, "TRUSTED_PATH_ANCHOR", trusted,
                    ), mock.patch.object(
                        recovery, "_reverify_prepublication_authorities",
                        side_effect=hostile_authority,
                    ):
                        with self.assertRaises(RuntimeError):
                            recovery._publish_attestation_create_only(root)
                finally:
                    for patch in reversed(patches):
                        patch.stop()
                self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)

    def test_file_and_root_commit_failures_leave_rejected_modes(self) -> None:
        for failed_mode, expected_file_mode in ((0o444, 0), (0o555, 0o444)):
            with self.subTest(failed_mode=oct(failed_mode)), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp).resolve()
                original = parent / "original"
                original.mkdir(mode=0o700)
                root = parent / "recovery"
                _, _, _, patches = self._create_only_context(original, root)
                real_fchmod = os.fchmod

                def fail_commit(descriptor: int, mode: int) -> None:
                    if mode == failed_mode:
                        raise OSError("synthetic commit failure")
                    real_fchmod(descriptor, mode)

                for patch in patches:
                    patch.start()
                try:
                    with mock.patch.object(
                        recovery.os, "fchmod", side_effect=fail_commit,
                    ):
                        with self.assertRaisesRegex(OSError, "synthetic commit"):
                            recovery._publish_attestation_create_only(root)
                finally:
                    for patch in reversed(patches):
                        patch.stop()
                self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)
                self.assertEqual(
                    stat.S_IMODE(
                        (root / "recovery-attestation.json").lstat().st_mode
                    ),
                    expected_file_mode,
                )

    def test_write_fsync_and_replay_failures_leave_mode000_tombstone(self) -> None:
        for failure in ("write", "fsync", "replay"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp).resolve()
                original = parent / "original"
                original.mkdir(mode=0o700)
                root = parent / "recovery"
                _, _, _, patches = self._create_only_context(original, root)
                nested: mock._patch
                if failure == "write":
                    nested = mock.patch.object(recovery.os, "write", return_value=0)
                elif failure == "fsync":
                    real_fsync = os.fsync
                    calls = 0

                    def fail_file_fsync(descriptor: int) -> None:
                        nonlocal calls
                        calls += 1
                        if calls == 2:
                            raise OSError("synthetic fsync failure")
                        real_fsync(descriptor)

                    nested = mock.patch.object(
                        recovery.os, "fsync", side_effect=fail_file_fsync,
                    )
                else:
                    real_read = os.read
                    corrupted = False

                    def corrupt_replay(descriptor: int, count: int) -> bytes:
                        nonlocal corrupted
                        raw = real_read(descriptor, count)
                        if raw and not corrupted:
                            corrupted = True
                            return bytes([raw[0] ^ 1]) + raw[1:]
                        return raw

                    nested = mock.patch.object(
                        recovery.os, "read", side_effect=corrupt_replay,
                    )
                for patch in patches:
                    patch.start()
                try:
                    with nested:
                        with self.assertRaises((RuntimeError, OSError)):
                            recovery._publish_attestation_create_only(root)
                finally:
                    for patch in reversed(patches):
                        patch.stop()
                self.assertEqual(stat.S_IMODE(root.lstat().st_mode), 0o700)
                path = root / "recovery-attestation.json"
                self.assertEqual(stat.S_IMODE(path.lstat().st_mode), 0)

    def test_final_file_open_contract_and_no_fs_after_root_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            root = parent / "recovery"
            _, _, _, patches = self._create_only_context(original, root)
            committed = False
            late_operations: list[str] = []
            postcommit_close_count = 0
            file_opens: list[tuple[int, int]] = []
            real_open = os.open
            real_close = os.close
            real_fchmod = os.fchmod

            def tracked_open(path: object, flags: int, mode: int = 0o777, **kwargs: object) -> int:
                if committed:
                    late_operations.append("open")
                if path == "recovery-attestation.json":
                    file_opens.append((flags, mode))
                return real_open(path, flags, mode, **kwargs)

            def tracked_fchmod(descriptor: int, mode: int) -> None:
                nonlocal committed
                if committed:
                    late_operations.append("fchmod")
                real_fchmod(descriptor, mode)
                if mode == 0o555:
                    committed = True

            def tracked_close(descriptor: int) -> None:
                nonlocal postcommit_close_count
                real_close(descriptor)
                if committed:
                    postcommit_close_count += 1
                    raise OSError("ignored close after commit")

            operation_patches = [
                mock.patch.object(recovery.os, "open", side_effect=tracked_open),
                mock.patch.object(recovery.os, "fchmod", side_effect=tracked_fchmod),
                mock.patch.object(recovery.os, "close", side_effect=tracked_close),
            ]
            for name in (
                "stat", "lstat", "fstat", "fsync", "listdir", "lseek",
                "read", "write", "mkdir",
            ):
                real_operation = getattr(os, name)

                def track_operation(*args: object, _name: str = name,
                                    _real: object = real_operation, **kwargs: object):
                    if committed:
                        late_operations.append(_name)
                    return _real(*args, **kwargs)

                operation_patches.append(
                    mock.patch.object(
                        recovery.os, name, side_effect=track_operation,
                    )
                )
            for patch in patches:
                patch.start()
            for patch in operation_patches:
                patch.start()
            try:
                result = recovery._publish_attestation_create_only(root)
            finally:
                for patch in reversed(operation_patches):
                    patch.stop()
                for patch in reversed(patches):
                    patch.stop()
            self.assertTrue(result["final_mode_commit"])
            self.assertTrue(committed)
            self.assertEqual(late_operations, [])
            self.assertGreater(postcommit_close_count, 0)
            self.assertEqual(len(file_opens), 1)
            flags, mode = file_opens[0]
            self.assertEqual(mode, 0)
            self.assertTrue(flags & os.O_EXCL)
            self.assertTrue(flags & os.O_NOFOLLOW)
            self.assertTrue(flags & os.O_CLOEXEC)
            self.assertTrue(flags & os.O_RDWR)

    def test_import_rejects_preloaded_frozen_runtime(self) -> None:
        name = (
            "methods.bernini_action_editing."
            "semantic_anchor_vjepa2_role_directed_teacher_margin_v4g"
        )
        self.assertNotIn(name, sys.modules)
        sys.modules[name] = types.SimpleNamespace(__file__=str(Path(recovery.__file__)))
        try:
            with self.assertRaisesRegex(RuntimeError, "preloaded"):
                recovery._import_frozen_v4g(Path("/does/not/matter"))
        finally:
            sys.modules.pop(name, None)

    def test_recovery_parser_requires_exact_torch_and_rocm_release(self) -> None:
        exact = types.SimpleNamespace(
            torch=types.SimpleNamespace(
                __version__="2.7.1+rocm6.3",
                version=types.SimpleNamespace(hip="6.3.test"),
            )
        )
        self.assertTrue(
            recovery._verify_recovery_parser_torch(exact)[
                "torch_version_exact_2_7_1_rocm6_3"
            ]
        )
        wrong = types.SimpleNamespace(
            torch=types.SimpleNamespace(
                __version__="2.7.0",
                version=types.SimpleNamespace(hip="6.3.test"),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Torch/ROCm"):
            recovery._verify_recovery_parser_torch(wrong)
        with mock.patch.dict(sys.modules, {"torch": object()}):
            with self.assertRaisesRegex(RuntimeError, "preloaded"):
                recovery._require_no_preloaded_torch()
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp).resolve() / "site-packages"
            paths = {
                "torch": site / "torch/__init__.py",
                "torch.nn": site / "torch/nn/__init__.py",
                "torch.nn.functional": site / "torch/nn/functional.py",
            }
            modules = {}
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# pinned module\n")
                module = types.ModuleType(name)
                module.__file__ = str(path)
                loader = importlib.machinery.SourceFileLoader(name, str(path))
                module.__spec__ = importlib.util.spec_from_loader(
                    name, loader, origin=str(path),
                    is_package=name in {"torch", "torch.nn"},
                )
                modules[name] = module
            modules["torch"].__path__ = [str(site / "torch")]
            v4g = types.SimpleNamespace(
                torch=modules["torch"], nn=modules["torch.nn"],
                F=modules["torch.nn.functional"],
            )
            with mock.patch.object(
                recovery, "PYTHON_SITE_PACKAGES", site,
            ), mock.patch.dict(sys.modules, modules):
                origins = recovery._verify_recovery_parser_torch_origins(v4g)
            self.assertTrue(origins["v4g_torch_module_identities_exact"])

    def test_import_rejects_nonstandard_meta_path_before_loading_source(self) -> None:
        removed = {
            name: sys.modules.pop(name)
            for name in ("methods.bernini_action_editing", "methods")
            if name in sys.modules
        }
        try:
            with mock.patch.object(sys, "meta_path", [object()]):
                with self.assertRaisesRegex(RuntimeError, "import machinery"):
                    recovery._import_frozen_v4g(Path("/does/not/matter"))
        finally:
            sys.modules.update(removed)

    def test_captured_source_loader_executes_fd_bytes_and_rejects_path_aba(self) -> None:
        protected = ("methods.bernini_action_editing", "methods")
        removed = {
            name: sys.modules.pop(name) for name in protected if name in sys.modules
        }
        standard_meta_path = [
            importlib.machinery.BuiltinImporter,
            importlib.machinery.FrozenImporter,
            importlib.machinery.PathFinder,
        ]
        try:
            with tempfile.TemporaryDirectory() as temp:
                package = Path(temp).resolve() / "methods/bernini_action_editing"
                package.mkdir(parents=True)
                source = package / "synthetic_capture_test.py"
                trusted_raw = b"MARKER = 'trusted'\n"
                source.write_bytes(trusted_raw)
                os.chmod(source, 0o444)
                expected = hashlib.sha256(trusted_raw).hexdigest()
                name = "methods.bernini_action_editing.synthetic_capture_test"
                with mock.patch.object(sys, "meta_path", list(standard_meta_path)):
                    loaded = recovery._import_captured_source_graph(
                        package, {name: (source.name, expected)}, name,
                    )
                self.assertEqual(loaded.MARKER, "trusted")
                for module_name in (name, *protected):
                    sys.modules.pop(module_name, None)

                side_effect = Path(temp) / "malicious-executed"
                real_read = recovery._read_regular
                mutated = False

                def capture_then_swap(path: Path, **kwargs):
                    nonlocal mutated
                    result = real_read(path, **kwargs)
                    if path == source and kwargs.get("capture") and not mutated:
                        mutated = True
                        os.chmod(source, 0o644)
                        source.write_text(
                            f"open({str(side_effect)!r}, 'w').write('bad')\n"
                        )
                        os.chmod(source, 0o444)
                    return result

                with mock.patch.object(
                    sys, "meta_path", list(standard_meta_path),
                ), mock.patch.object(
                    recovery, "_read_regular", side_effect=capture_then_swap,
                ):
                    with self.assertRaisesRegex(RuntimeError, "executed-byte"):
                        recovery._import_captured_source_graph(
                            package, {name: (source.name, expected)}, name,
                        )
                self.assertFalse(side_effect.exists())

                cleanup_raw = (
                    b"import sys, types\n"
                    b"sys.modules['torch.synthetic_cleanup'] = "
                    b"types.ModuleType('torch.synthetic_cleanup')\n"
                    b"raise RuntimeError('synthetic import failure')\n"
                )
                os.chmod(source, 0o644)
                source.write_bytes(cleanup_raw)
                os.chmod(source, 0o444)
                cleanup_sha = hashlib.sha256(cleanup_raw).hexdigest()
                with mock.patch.object(
                    sys, "meta_path", list(standard_meta_path),
                ):
                    with self.assertRaisesRegex(RuntimeError, "synthetic import failure"):
                        recovery._import_captured_source_graph(
                            package, {name: (source.name, cleanup_sha)}, name,
                        )
                self.assertFalse(
                    any(
                        module_name == "torch" or module_name.startswith("torch.")
                        for module_name in sys.modules
                    )
                )
        finally:
            for name in list(sys.modules):
                if name.startswith("methods.bernini_action_editing.synthetic_capture_test"):
                    sys.modules.pop(name, None)
            for name in protected:
                sys.modules.pop(name, None)
            sys.modules.update(removed)

    def test_feature_exact6_input_snapshot_is_independently_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            shard_rows = []
            for index in range(6):
                path = root / f"shard{index}.json"
                path.write_bytes(f"shard-{index}".encode())
                os.chmod(path, 0o444)
                binding, _ = recovery._read_regular(path, mode=0o444, nlink=1)
                shard_rows.append({
                    "index": index, "path": str(path),
                    "sha256": binding["sha256"],
                })
            receipt_values = [
                {"shards": shard_rows}, {"receipt": "v4a"},
                {"receipt": "v4c"}, {"receipt": "v4d"},
            ]
            receipts = []
            for index, value in enumerate(receipt_values):
                path = root / f"receipt{index}.json"
                raw = recovery._canonical_bytes(value) + b"\n"
                path.write_bytes(raw)
                os.chmod(path, 0o444)
                binding, captured = recovery._read_regular(
                    path, mode=0o444, nlink=1, capture=True,
                )
                self.assertEqual(captured, raw)
                receipts.append((binding, raw))
            labels = ("feature", "v4a", "v4c", "v4d")
            expected_rows = [{
                "label": labels[index], "path": binding["path"],
                "sha256": binding["sha256"], "size_bytes": binding["size_bytes"],
            } for index, (binding, _) in enumerate(receipts)]
            for index, shard in enumerate(shard_rows):
                binding, _ = recovery._read_regular(
                    Path(shard["path"]), mode=0o444, nlink=1,
                )
                expected_rows.append({
                    "label": f"feature-shard-{index}", "path": binding["path"],
                    "sha256": binding["sha256"], "size_bytes": binding["size_bytes"],
                })
            expected_sha = recovery._object_sha(expected_rows)
            with mock.patch.object(recovery, "INPUT_SNAPSHOT_SHA256", expected_sha):
                result = recovery._verify_input_snapshot(receipts)
            self.assertEqual(result["sha256"], expected_sha)
            forged_feature = dict(receipt_values[0])
            forged_feature["shards"] = list(reversed(shard_rows))
            forged = list(receipts)
            forged[0] = (
                forged[0][0], recovery._canonical_bytes(forged_feature) + b"\n",
            )
            with mock.patch.object(recovery, "INPUT_SNAPSHOT_SHA256", expected_sha):
                with self.assertRaisesRegex(RuntimeError, "shard order"):
                    recovery._verify_input_snapshot(forged)

    def test_recovery_exact3_manifest_and_detached_controller(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            root = parent / "release"
            runtime = root / recovery.RECOVERY_RUNTIME_RELATIVE_PATH
            tests = root / recovery.RECOVERY_TESTS_RELATIVE_PATH
            runtime.parent.mkdir(parents=True)
            tests.parent.mkdir(parents=True)
            runtime.write_bytes(b"runtime\n")
            tests.write_bytes(b"tests\n")
            os.chmod(runtime, 0o444)
            os.chmod(tests, 0o444)
            runtime_sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
            tests_sha = hashlib.sha256(tests.read_bytes()).hexdigest()
            manifest = {
                "schema_version": (
                    "v4g-scientific-no-go-recovery-detached-release-manifest-v1"
                ),
                "status": "SEALED",
                "payload": [
                    {
                        "relative_path": recovery.RECOVERY_RUNTIME_RELATIVE_PATH,
                        "role": "recovery_runtime", "sha256": runtime_sha,
                    },
                    {
                        "relative_path": recovery.RECOVERY_TESTS_RELATIVE_PATH,
                        "role": "recovery_runtime_tests", "sha256": tests_sha,
                    },
                ],
                "payload_count": 2,
                "manifest_target_relative_path": recovery.RECOVERY_MANIFEST_NAME,
                "release_tree_contract": {
                    "exact_file_count_including_manifest": 3,
                    "exact_directory_count_below_root": 3,
                    "all_files_mode_0444_nlink1": True,
                    "all_directories_mode_0555": True,
                },
                "authority_graph": {
                    "sha256_graph_is_directed_and_acyclic": True,
                    "manifest_pins_runtime_and_tests": True,
                    "runtime_pins_controller_or_manifest": False,
                    "detached_controller_is_outside_release_tree": True,
                },
            }
            manifest["manifest_digest"] = recovery._object_sha(manifest)
            manifest_path = root / recovery.RECOVERY_MANIFEST_NAME
            manifest_path.write_bytes(recovery._canonical_bytes(manifest) + b"\n")
            os.chmod(manifest_path, 0o444)
            for directory in sorted(
                (path for path in root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts), reverse=True,
            ):
                os.chmod(directory, 0o555)
            os.chmod(root, 0o555)
            controller = parent / "controller.sh"
            controller.write_bytes(b"#!/bin/sh\n")
            os.chmod(controller, 0o555)
            with mock.patch.object(
                recovery, "RECOVERY_RELEASE_ROOT", root,
            ), mock.patch.object(
                recovery, "RECOVERY_CONTROLLER_PATH", controller,
            ), mock.patch.object(
                recovery, "__file__", str(runtime),
            ):
                result = recovery._verify_recovery_release_and_controller()
            self.assertEqual(result["exact_file_count"], 3)
            self.assertEqual(result["runtime"]["sha256"], runtime_sha)
            self.assertEqual(result["tests"]["sha256"], tests_sha)

    @staticmethod
    def _receipt_fixture(fold: int, bindings: dict[str, dict[str, object]]) -> dict[str, object]:
        pre = bindings[f"fold{fold}/preselection.pt"]
        fixed = bindings[f"fold{fold}/fixed1200.pt"]
        state_sha = hashlib.sha256(f"state{fold}".encode()).hexdigest()
        artifact_common = {
            "mode_octal": "0444", "nlink": 1,
            "model_state_sha256": state_sha,
        }
        pre_artifact = {
            **artifact_common, "file_sha256": pre["sha256"],
            "size_bytes": pre["size_bytes"], "metadata_digest": "1" * 64,
            "physical_identity": {
                "device": pre["device"], "inode": pre["inode"],
                "size_bytes": pre["size_bytes"],
            },
        }
        fixed_artifact = {
            **artifact_common, "file_sha256": fixed["sha256"],
            "size_bytes": fixed["size_bytes"], "metadata_digest": "2" * 64,
            "physical_identity": {
                "device": fixed["device"], "inode": fixed["inode"],
                "size_bytes": fixed["size_bytes"],
            },
        }
        return {
            "fold_index": fold,
            "receipt_digest": recovery.INNER_RECEIPT_DIGEST[fold],
            "status": "V4G_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD",
            "inner_pass": False,
            "global_barrier_required_before_any_fold_oof": True,
            "oof_semantic_tensor_materialized_count": 0,
            "oof_semantic_tensor_read_count_exact0": True,
            "oof_used_for_training_checkpoint_or_inner_gate": False,
            "model_fit_original_count": 400 + fold,
            "inner_validation_original_count": 110 + fold,
            "oof_original_count": 130 - fold,
            "training": {
                "full_budget_steps_executed": 1200, "fixed_step": 1200,
                "early_stopped": False,
            },
            "fixed_candidate": {
                "inner_pass": False, "pass": False,
                "gate": {
                    "complete_candidate_dependent_inner_gate": False,
                    "five_view_fidelity_gate": False,
                    "all_three_negative_full_gates": False,
                },
            },
            "qualification_scope": dict(recovery.EXPECTED_QUALIFICATION_SCOPE),
            "selective_feature_materialization_before_global_barrier": {
                "stage1_oof_semantic_tensor_count": 0,
                "stage2_oof_semantic_tensor_count": 0,
            },
            "preselection_checkpoint_artifact": pre_artifact,
            "fixed1200_checkpoint_artifact": fixed_artifact,
            "preselection_fixed1200_checkpoint_pair_join": {
                "distinct_device_inode_pair": True,
                "same_model_state_sha256": True,
            },
        }

    def test_receipt_projection_accepts_exact_three_fields_and_rejects_inode_drift(self) -> None:
        bindings: dict[str, dict[str, object]] = {}
        for fold in range(5):
            for index, name in enumerate(("preselection.pt", "fixed1200.pt")):
                bindings[f"fold{fold}/{name}"] = {
                    "sha256": hashlib.sha256(f"{fold}-{name}".encode()).hexdigest(),
                    "size_bytes": 1000 + fold * 10 + index,
                    "mode_octal": "0444", "nlink": 1,
                    "device": 48, "inode": 100 + fold * 2 + index,
                    "mtime_ns": 1, "ctime_ns": 2,
                }
            bindings[f"fold{fold}/inner.json"] = {
                "sha256": recovery.INNER_RECEIPT_SHA256[fold],
            }
        receipts = {
            fold: self._receipt_fixture(fold, bindings) for fold in range(5)
        }

        def loader(root: str, expected_sha: str, run_binding: dict[str, str]):
            del expected_sha, run_binding
            fold = int(Path(root).name[-1])
            return receipts[fold], {
                "receipt_digest": recovery.INNER_RECEIPT_DIGEST[fold],
                "file_sha256": recovery.INNER_RECEIPT_SHA256[fold],
            }

        rows = recovery._verify_receipts(Path("/authority"), bindings, loader)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["three_field_runtime_physical_identity_projection_exact"] for row in rows))
        receipts[2]["fixed1200_checkpoint_artifact"]["physical_identity"]["inode"] += 1
        with self.assertRaisesRegex(RuntimeError, "fold2"):
            recovery._verify_receipts(Path("/authority"), bindings, loader)
        receipts[2] = self._receipt_fixture(2, bindings)
        receipts[2]["qualification_scope"]["action_representation_qualified"] = True
        with self.assertRaisesRegex(RuntimeError, "fold2"):
            recovery._verify_receipts(Path("/authority"), bindings, loader)

    def test_public_recover_has_only_frozen_paths_and_internal_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp).resolve()
            original = parent / "original"
            original.mkdir(mode=0o700)
            recovery_root = parent / "sibling"
            result_stub = {
                "path": str(recovery_root / "recovery-attestation.json"),
                "file_sha256": "f" * 64, "size_bytes": 1,
                "receipt_digest": "e" * 64, "mode_octal": "0444",
                "nlink": 1, "root_mode_octal": "0555", "exact_file_count": 1,
            }
            with mock.patch.object(recovery, "RELEASE_SEALED", True), mock.patch.object(
                recovery, "ORIGINAL_RUN_ROOT", original,
            ), mock.patch.object(
                recovery, "RECOVERY_ROOT", recovery_root,
            ), mock.patch.object(
                recovery, "_publish_attestation_create_only", return_value=result_stub,
            ) as publisher:
                result = recovery.recover(original, recovery_root)
                publisher.assert_called_once_with(recovery_root)
                self.assertEqual(result, result_stub)
                with self.assertRaisesRegex(RuntimeError, "official recovery paths"):
                    recovery.recover(original, parent / "wrong")


if __name__ == "__main__":
    unittest.main()
