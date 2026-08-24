#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import e00_legacy_infer_fixed_rng_wrapper_r5 as wrapper
import validate_e00_three_vessel_clean_diag_r5 as validator
from tools import build_e00_three_vessel_clean_diag_r5_package as package


class E00CleanDiagnosticR5BootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = validator.DEFAULT_PROTOCOL
        self.protocol = copy.deepcopy(validator.load_protocol(self.protocol_path))
        self.bootstrap_path = REPO_ROOT / package.BOOTSTRAP_ROOT_FILE
        self.bootstrap_sha = package._sha256(self.bootstrap_path)

    @staticmethod
    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def make_core_tree(self, root: Path) -> Path:
        dfix2 = root / "dfix2"
        for relative in package.CORE_PINS:
            source = REPO_ROOT / relative
            target = dfix2 / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return dfix2

    def build_actual_package(self, root: Path) -> Path:
        output = root / "package"
        package.build_package(
            dfix2_source_tree=self.make_core_tree(root),
            overlay_root=REPO_ROOT,
            output=output,
            expected_bootstrap_root_sha256=self.bootstrap_sha,
        )
        return output

    def test_protocol_has_fresh_identity_exact_gate_and_honest_placeholder(self) -> None:
        value = validator.validate_protocol(self.protocol)
        self.assertEqual(value["schema_version"], validator.SCHEMA)
        self.assertEqual(value["revision_tag"], validator.REVISION_TAG)
        self.assertEqual(self.protocol["ab_gate_contract"]["required_equalities"], validator.REQUIRED_EQUALITIES)
        self.assertEqual(
            self.protocol["ab_gate_contract"]["current_artifact_revalidation_required"],
            validator.REQUIRED_CURRENT_BINDINGS,
        )
        placeholder = self.protocol["phase_a_placeholder_contract"]
        self.assertEqual(placeholder["classification"], "content_static_lossy_decode_reencode_placeholder")
        self.assertFalse(placeholder["decoded_bit_exact_source_video_claimed"])
        self.assertFalse(placeholder["encoded_bytes_equal_source_video_claimed"])

    def test_bootstrap_root_is_minimal_and_current(self) -> None:
        root = package.validate_bootstrap_root(
            content_root=REPO_ROOT,
            bootstrap_path=self.bootstrap_path,
            expected_bootstrap_sha256=self.bootstrap_sha,
        )
        self.assertEqual(tuple(root["pins"]), package.ROOT_PINNED_FILES)
        self.assertEqual(root["bootstrap_consumers"], list(package.BOOTSTRAP_CONSUMERS))
        self.assertFalse(set(root["pins"]) & set(package.BOOTSTRAP_CONSUMERS))
        self.assertNotIn(package.BOOTSTRAP_ROOT_FILE, root["pins"])
        _, pins = package._load_authority(REPO_ROOT, root)
        self.assertFalse(set(pins) & package.UNPINNED_BY_AUTHORITY)
        package._verify_overlay_pins(REPO_ROOT, pins)

    def test_launchers_and_bridge_check_root_before_python_and_trap_immediately(self) -> None:
        texts = {
            relative: (REPO_ROOT / relative).read_text(encoding="utf-8")
            for relative in package.BOOTSTRAP_CONSUMERS
        }
        for relative, text in texts.items():
            self.assertIn(self.bootstrap_sha, text, relative)
            root_check = text.index('verify_sha "$bootstrap_root"')
            self.assertLess(root_check, text.index('"$python_bin" -B "$builder"'), relative)
            self.assertLess(root_check, text.index('"$python_bin" -B "$validator"'), relative)
            self.assertIn("check_cache_free", text)
            self.assertNotIn("sbatch ", text)
            self.assertNotIn("scancel ", text)
            self.assertNotIn("export CUDA_VISIBLE_DEVICES=", text)
            self.assertNotIn("export ROCR_VISIBLE_DEVICES=", text)
        for relative in (package.A_LAUNCHER, package.BC_LAUNCHER):
            lines = [line.strip() for line in texts[relative].splitlines()]
            lock_index = lines.index('mkdir "$lock"')
            self.assertEqual(lines[lock_index + 1], "trap finalize EXIT")

    def test_current_native_and_four_rng_receipt_bytes_are_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"; video.write_bytes(b"video")
            native = root / "native.json"; native.write_bytes(b"native")
            rng_paths = []
            for rank in range(4):
                path = root / f"rng-{rank}.json"; path.write_bytes(f"rng-{rank}".encode())
                rng_paths.append(path)
            artifacts = {
                "video": {"path": str(video), "sha256": validator.file_sha256(video)},
                "native_receipt": {"path": str(native), "sha256": validator.file_sha256(native)},
                "rng_receipts_rank_order": [
                    {"rank": rank, "path": str(path), "sha256": validator.file_sha256(path)}
                    for rank, path in enumerate(rng_paths)
                ],
            }
            value = validator.validate_current_artifact_bytes(artifacts)
            self.assertEqual(len(value["artifacts"]["rng_receipts_rank_order"]), 4)
            rng_paths[1].write_bytes(b"changed")
            with self.assertRaises(Exception):
                validator.validate_current_artifact_bytes(artifacts)

    def test_final_closure_uses_fresh_r5_schemas(self) -> None:
        observed: dict[str, str] = {}

        def record(**_: object) -> dict[str, bool]:
            observed["revision"] = validator.r4.REVISION_TAG
            observed["final_schema"] = validator.r4.FINAL_SCHEMA
            observed["rng_schema"] = validator.r4.RNG_SCHEMA
            return {"complete": True}

        with mock.patch.object(validator.r4, "build_final_audit", side_effect=record):
            self.assertTrue(validator.build_final_audit()["complete"])
        self.assertEqual(observed["revision"], validator.REVISION_TAG)
        self.assertEqual(observed["final_schema"], validator.FINAL_SCHEMA)
        self.assertEqual(observed["rng_schema"], validator.RNG_SCHEMA)

    def test_phase_a_capability_is_token_and_role_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"; manifest.write_text("{}\n", encoding="utf-8")
            token_sha = self.sha(b"r5-a-token")
            protocol_id = validator.protocol_identity(self.protocol, self.protocol_path)
            authorization = {
                "schema_version": "bernini-e00-clean-diagnostic-r5-phase-a-authorization-v5",
                "execution_authorized": True,
                "authorized_phase": "A_ONLY_THEN_STOP",
                "package_manifest_sha256": validator.file_sha256(manifest),
                "protocol_file_sha256": protocol_id["file_sha256"],
                "protocol_canonical_sha256": protocol_id["canonical_sha256"],
                "bridge_capability_token_sha256": token_sha,
                "parent_job_id": "143808",
                "compute_node": "auh7-1b-gpu-292",
                "only_authorized_arm": validator.ARM_ROLES[0],
                "must_stop_after_a": True,
                "bc_execution_authorized": False,
                "sp4_observer_released_node292": True,
                "authorized_by": "integrity-review",
            }
            auth = root / "authorization.json"; auth.write_text(json.dumps(authorization), encoding="utf-8")
            value = validator.validate_bridge_capability(
                phase="A", arm_role=validator.ARM_ROLES[0], protocol=self.protocol,
                protocol_path=self.protocol_path, package_manifest_path=manifest,
                authorization_path=auth, capability_token_sha256=token_sha,
            )
            self.assertTrue(value["only_this_arm_admitted"])
            with self.assertRaises(validator.E00R5Error):
                validator.validate_bridge_capability(
                    phase="A", arm_role=validator.ARM_ROLES[1], protocol=self.protocol,
                    protocol_path=self.protocol_path, package_manifest_path=manifest,
                    authorization_path=auth, capability_token_sha256=token_sha,
                )

    def test_package_rejects_cache_and_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self.build_actual_package(root)
            manifest = output / package.MANIFEST_NAME
            cache = output / "nested" / "__pycache__"; cache.mkdir(parents=True)
            with self.assertRaises(package.PackageError):
                package.verify_package(
                    package_root=output, manifest_path=manifest,
                    expected_bootstrap_root_sha256=self.bootstrap_sha,
                )
            cache.rmdir(); cache.parent.rmdir()
            bytecode = output / "payload.pyc"; bytecode.write_bytes(b"compiled")
            with self.assertRaises(package.PackageError):
                package.verify_package(
                    package_root=output, manifest_path=manifest,
                    expected_bootstrap_root_sha256=self.bootstrap_sha,
                )

    def test_synchronized_authority_overlay_manifest_changes_fail_normal_and_optimized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self.build_actual_package(root)
            target_relative = "methods/bernini_action_editing/assets/e00_three_vessel_clean_diag_r5_protocol_20260821.json"
            target = output / target_relative
            old_target_sha = package._sha256(target)
            target.write_bytes(target.read_bytes() + b"\n")
            new_target_sha = package._sha256(target)

            authority = output / package.PIN_AUTHORITY_FILE
            authority_text = authority.read_text(encoding="utf-8")
            self.assertIn(old_target_sha, authority_text)
            authority.write_text(authority_text.replace(old_target_sha, new_target_sha, 1), encoding="utf-8")
            new_authority_sha = package._sha256(authority)

            manifest_path = output / package.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["overlay_pins"][target_relative] = new_target_sha
            manifest["overlay_pin_authority"]["sha256"] = new_authority_sha
            for row in manifest["files"]:
                if row["path"] == target_relative:
                    row["sha256"] = new_target_sha; row["bytes"] = target.stat().st_size
                elif row["path"] == package.PIN_AUTHORITY_FILE:
                    row["sha256"] = new_authority_sha; row["bytes"] = authority.stat().st_size
            manifest["content_digest"] = hashlib.sha256(package._canonical(manifest["files"])).hexdigest()
            manifest_path.write_bytes(package._canonical(manifest) + b"\n")

            builder = output / package.BUILDER_FILE
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            common = [
                str(builder), "verify", "--package-root", str(output),
                "--manifest", str(manifest_path),
                "--expected-bootstrap-root-sha256", self.bootstrap_sha,
            ]
            for flags in (["-B"], ["-O", "-B"]):
                result = subprocess.run(
                    [sys.executable, *flags, *common],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False,
                )
                self.assertNotEqual(result.returncode, 0, flags)
                self.assertIn("bootstrap-pinned current bytes differ", result.stderr, flags)

    def test_fixed_rng_repeatability_and_caller_restore(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        caller = torch.random.get_rng_state().clone()
        first, first_fork, first_fixed = wrapper.run_with_fixed_initial_rng(
            torch, lambda: torch.rand(8), cuda_device=None, cpu_seed=83002700, cuda_seed=None
        )
        second, second_fork, second_fixed = wrapper.run_with_fixed_initial_rng(
            torch, lambda: torch.rand(8), cuda_device=None, cpu_seed=83002700, cuda_seed=None
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_fixed["seeded_initial"], second_fixed["seeded_initial"])
        self.assertTrue(first_fork["cpu_state_restored"])
        self.assertTrue(second_fork["cpu_state_restored"])
        self.assertTrue(torch.equal(caller, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
