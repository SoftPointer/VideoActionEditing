#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import e00_legacy_infer_fixed_rng_wrapper_r4 as wrapper
import validate_e00_three_vessel_clean_diag_r4 as validator
from tools import build_e00_three_vessel_clean_diag_r4_package as package


class E00CleanDiagnosticR4IntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = validator.DEFAULT_PROTOCOL
        self.protocol = copy.deepcopy(validator.load_protocol(self.protocol_path))

    @staticmethod
    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def test_protocol_has_fresh_r4_identity_and_exact_contracts(self) -> None:
        value = validator.validate_protocol(self.protocol)
        self.assertEqual(value["schema_version"], validator.SCHEMA)
        self.assertEqual(value["revision_tag"], validator.REVISION_TAG)
        self.assertEqual(
            self.protocol["ab_gate_contract"]["required_equalities"],
            validator.REQUIRED_EQUALITIES,
        )
        self.assertEqual(
            self.protocol["ab_gate_contract"]["current_artifact_revalidation_required"],
            validator.REQUIRED_CURRENT_BINDINGS,
        )
        integrity = self.protocol["package_integrity_contract"]
        self.assertTrue(integrity["overlay_pins_required"])
        self.assertTrue(integrity["package_pycache_directories_forbidden"])
        self.assertTrue(integrity["package_pyc_files_forbidden"])
        self.assertTrue(integrity["cache_scan_required_at_build_verify_and_each_phase"])

    def test_protocol_rejects_changed_required_equalities(self) -> None:
        for change in ("remove", "append", "reorder"):
            changed = copy.deepcopy(self.protocol)
            rows = changed["ab_gate_contract"]["required_equalities"]
            if change == "remove":
                rows.pop()
            elif change == "append":
                rows.append("unregistered_equality")
            else:
                rows[0], rows[1] = rows[1], rows[0]
            with self.assertRaises(validator.E00R4Error):
                validator.validate_protocol(changed)

    def test_current_native_and_four_rng_receipts_are_bound_to_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            native = root / "video.mp4.receipt.json"
            video.write_bytes(b"video")
            native.write_bytes(b"native")
            rng_paths = []
            for rank in range(4):
                path = root / f"rng.rank{rank}.json"
                path.write_bytes(f"rng-{rank}".encode("utf-8"))
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
            self.assertEqual([row["rank"] for row in value["artifacts"]["rng_receipts_rank_order"]], list(range(4)))
            native.write_bytes(b"changed-native")
            with self.assertRaises(Exception):
                validator.validate_current_artifact_bytes(artifacts)
            native.write_bytes(b"native")
            rng_paths[3].write_bytes(b"changed-rng")
            with self.assertRaises(Exception):
                validator.validate_current_artifact_bytes(artifacts)

    def test_r4_final_closure_uses_r4_schemas_around_retained_core(self) -> None:
        observed: dict[str, str] = {}

        def record(**_: object) -> dict[str, bool]:
            observed["revision"] = validator.r3.REVISION_TAG
            observed["final_schema"] = validator.r3.FINAL_SCHEMA
            observed["rng_schema"] = validator.r3.RNG_SCHEMA
            return {"complete": True}

        with mock.patch.object(validator.r3, "build_final_audit", side_effect=record):
            self.assertEqual(validator.build_final_audit()["complete"], True)
        self.assertEqual(observed["revision"], validator.REVISION_TAG)
        self.assertEqual(observed["final_schema"], validator.FINAL_SCHEMA)
        self.assertEqual(observed["rng_schema"], validator.RNG_SCHEMA)

    def test_phase_a_capability_is_token_and_role_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            token_sha = self.sha(b"r4-a-token")
            protocol_id = validator.protocol_identity(self.protocol, self.protocol_path)
            authorization = {
                "schema_version": "bernini-e00-clean-diagnostic-r4-phase-a-authorization-v4",
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
            auth_path = root / "authorization.json"
            auth_path.write_text(json.dumps(authorization), encoding="utf-8")
            value = validator.validate_bridge_capability(
                phase="A",
                arm_role=validator.ARM_ROLES[0],
                protocol=self.protocol,
                protocol_path=self.protocol_path,
                package_manifest_path=manifest,
                authorization_path=auth_path,
                capability_token_sha256=token_sha,
            )
            self.assertTrue(value["only_this_arm_admitted"])
            with self.assertRaises(validator.E00R4Error):
                validator.validate_bridge_capability(
                    phase="A",
                    arm_role=validator.ARM_ROLES[1],
                    protocol=self.protocol,
                    protocol_path=self.protocol_path,
                    package_manifest_path=manifest,
                    authorization_path=auth_path,
                    capability_token_sha256=token_sha,
                )
            with self.assertRaises(validator.E00R4Error):
                validator.validate_bridge_capability(
                    phase="A",
                    arm_role=validator.ARM_ROLES[0],
                    protocol=self.protocol,
                    protocol_path=self.protocol_path,
                    package_manifest_path=manifest,
                    authorization_path=auth_path,
                    capability_token_sha256="f" * 64,
                )

    def test_cache_and_bytecode_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain.py").write_text("x = 1\n", encoding="utf-8")
            package.reject_cache_bytecode(root)
            cache = root / "__pycache__"
            cache.mkdir()
            with self.assertRaises(package.PackageError):
                package.reject_cache_bytecode(root)
            cache.rmdir()
            bytecode = root / "plain.pyc"
            bytecode.write_bytes(b"compiled")
            with self.assertRaises(package.PackageError):
                package.reject_cache_bytecode(root)

    def test_builder_checks_pins_manifest_constants_and_current_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dfix2 = root / "dfix2"
            overlay = root / "overlay"
            output = root / "package"
            core_relative = "methods/core.py"
            a_relative = "methods/a-launcher.sh"
            bc_relative = "methods/bc-launcher.sh"
            extra_relative = "methods/overlay.py"
            pin_relative = package.PIN_AUTHORITY_FILE
            (dfix2 / "methods").mkdir(parents=True)
            (dfix2 / core_relative).write_text("core\n", encoding="utf-8")
            for relative in (a_relative, bc_relative, extra_relative, pin_relative):
                path = overlay / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative + "\n", encoding="utf-8")
            core_pins = {core_relative: self.sha(b"core\n")}
            overlay_pins = {
                relative: package._sha256(overlay / relative)
                for relative in (a_relative, bc_relative, extra_relative)
            }
            overlay_files = tuple(overlay_pins) + (pin_relative,)
            patches = (
                mock.patch.object(package, "CORE_PINS", core_pins),
                mock.patch.object(package, "OVERLAY_PINS", overlay_pins),
                mock.patch.object(package, "OVERLAY_FILES", overlay_files),
                mock.patch.object(package, "A_LAUNCHER", a_relative),
                mock.patch.object(package, "BC_LAUNCHER", bc_relative),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                package.build_package(dfix2_source_tree=dfix2, overlay_root=overlay, output=output)
                manifest_path = output / package.MANIFEST_NAME
                original_manifest_bytes = manifest_path.read_bytes()
                original = json.loads(original_manifest_bytes)
                verified = package.verify_package(package_root=output, manifest_path=manifest_path)
                self.assertEqual(verified["dfix2_revision"], package.DFIX2_REVISION)
                self.assertEqual(verified["dfix2_core_pins"], core_pins)
                self.assertEqual(verified["overlay_files"], list(overlay_files))
                self.assertEqual(verified["overlay_pins"], overlay_pins)

                changed_file = output / extra_relative
                changed_file.write_text("changed overlay\n", encoding="utf-8")
                with self.assertRaises(package.PackageError):
                    package.verify_package(package_root=output, manifest_path=manifest_path)
                changed_file.write_text(extra_relative + "\n", encoding="utf-8")

                for field, replacement in (
                    ("dfix2_revision", "different-revision"),
                    ("dfix2_core_pins", {core_relative: "0" * 64}),
                    ("overlay_files", list(reversed(overlay_files))),
                    ("overlay_pins", {**overlay_pins, extra_relative: "0" * 64}),
                ):
                    changed = copy.deepcopy(original)
                    changed[field] = replacement
                    manifest_path.write_bytes(package._canonical(changed) + b"\n")
                    with self.assertRaises(package.PackageError):
                        package.verify_package(package_root=output, manifest_path=manifest_path)
                manifest_path.write_bytes(original_manifest_bytes)

                bytecode = output / "payload.pyc"
                bytecode.write_bytes(b"compiled")
                with self.assertRaises(package.PackageError):
                    package.verify_package(package_root=output, manifest_path=manifest_path)
                bytecode.unlink()
                cache = output / "nested" / "__pycache__"
                cache.mkdir(parents=True)
                with self.assertRaises(package.PackageError):
                    package.verify_package(package_root=output, manifest_path=manifest_path)

    def test_registered_overlay_pins_match_workspace_bytes(self) -> None:
        self.assertTrue(package.OVERLAY_PINS)
        self.assertEqual(
            set(package.OVERLAY_PINS),
            set(package.OVERLAY_FILES) - {package.PIN_AUTHORITY_FILE},
        )
        for relative, expected in package.OVERLAY_PINS.items():
            path = REPO_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertFalse(path.is_symlink(), relative)
            self.assertEqual(package._sha256(path), expected, relative)

    def test_launchers_pin_builder_and_recheck_cache_without_hiding_slurm(self) -> None:
        builder_sha = package._sha256(METHOD_ROOT / "tools/build_e00_three_vessel_clean_diag_r4_package.py")
        bridge = (METHOD_ROOT / "scripts/auh_e00_three_vessel_clean_diag_r4_bridge.sh").read_text(encoding="utf-8")
        a_launcher = (METHOD_ROOT / "scripts/auh_launch_e00_three_vessel_clean_diag_r4_phase_a_only_node292.sh").read_text(encoding="utf-8")
        bc_launcher = (METHOD_ROOT / "scripts/auh_launch_e00_three_vessel_clean_diag_r4_phase_bc_node292.sh").read_text(encoding="utf-8")
        for text in (bridge, a_launcher, bc_launcher):
            self.assertIn(builder_sha, text)
            self.assertIn("check_cache_free", text)
            self.assertIn("-name __pycache__", text)
            self.assertIn("-name '*.pyc'", text)
            self.assertNotIn("sbatch ", text)
            self.assertNotIn("scancel ", text)
            self.assertNotIn("export CUDA_VISIBLE_DEVICES=", text)
            self.assertNotIn("export ROCR_VISIBLE_DEVICES=", text)
        self.assertIn("fixed_parent_job=143808", bridge)
        self.assertLess(bridge.index("bridge-capability"), bridge.index("torch.distributed.run"))
        self.assertEqual(a_launcher.count("srun --jobid="), 1)
        self.assertLess(
            bc_launcher.index('run_arm B "$role_b"'),
            bc_launcher.index('"$python_bin" -B "$validator" ab-gate'),
        )
        self.assertLess(
            bc_launcher.index('"$python_bin" -B "$validator" verify-ab-gate'),
            bc_launcher.index('run_arm C "$role_c"'),
        )
        self.assertIn('E00_R4_AB_GATE="$gate"', bc_launcher)

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
