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
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import e00_legacy_infer_fixed_rng_wrapper_r3 as wrapper
import validate_e00_three_vessel_clean_diag_r3 as validator
from tools import build_e00_three_vessel_clean_diag_r3_package as package


class E00CleanDiagnosticR3Test(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = validator.DEFAULT_PROTOCOL
        self.protocol = copy.deepcopy(validator.load_protocol(self.protocol_path))

    @staticmethod
    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def test_protocol_hard_closes_required_equalities_and_capability(self) -> None:
        validator.validate_protocol(self.protocol)
        self.assertEqual(
            self.protocol["ab_gate_contract"]["required_equalities"],
            validator.REQUIRED_EQUALITIES,
        )
        self.assertEqual(
            self.protocol["ab_gate_contract"]["current_artifact_revalidation_required"],
            validator.REQUIRED_CURRENT_BINDINGS,
        )
        self.assertTrue(
            self.protocol["bridge_capability_contract"][
                "direct_bridge_without_capability_forbidden"
            ]
        )

    def test_protocol_rejects_required_equality_mutations(self) -> None:
        for mutation in ("drop", "append", "swap"):
            bad = copy.deepcopy(self.protocol)
            rows = bad["ab_gate_contract"]["required_equalities"]
            if mutation == "drop":
                rows.pop()
            elif mutation == "append":
                rows.append("unregistered_equality")
            else:
                rows[0], rows[1] = rows[1], rows[0]
            with self.assertRaises(validator.E00R3Error):
                validator.validate_protocol(bad)

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

    def test_current_native_and_four_rng_receipt_bytes_are_reloaded(self) -> None:
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
            self.assertEqual(len(value["artifacts"]["rng_receipts_rank_order"]), 4)
            native.write_bytes(b"mutated-native")
            with self.assertRaises(validator.E00R3Error):
                validator.validate_current_artifact_bytes(artifacts)
            native.write_bytes(b"native")
            rng_paths[2].unlink()
            with self.assertRaises(validator.E00R3Error):
                validator.validate_current_artifact_bytes(artifacts)

    def phase_a_authorization(self, manifest: Path, token_sha: str) -> dict:
        protocol_id = validator.protocol_identity(self.protocol, self.protocol_path)
        return {
            "schema_version": "bernini-e00-clean-diagnostic-r3-phase-a-authorization-v3",
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
            "authorized_by": "independent-test-reviewer",
        }

    def phase_bc_authorization(self, manifest: Path, token_sha: str) -> dict:
        protocol_id = validator.protocol_identity(self.protocol, self.protocol_path)
        return {
            "schema_version": "bernini-e00-clean-diagnostic-r3-phase-bc-authorization-v3",
            "execution_authorized": True,
            "authorized_phase": "B_THEN_CURRENT_AB_GATE_THEN_C",
            "package_manifest_sha256": validator.file_sha256(manifest),
            "protocol_file_sha256": protocol_id["file_sha256"],
            "protocol_canonical_sha256": protocol_id["canonical_sha256"],
            "bridge_capability_token_sha256": token_sha,
            "phase_a_stopped_marker_sha256": "1" * 64,
            "phase_a_arm_audit_sha256": "2" * 64,
            "phase_a_mp4_sha256": "3" * 64,
            "parent_job_id": "143808",
            "compute_node": "auh7-1b-gpu-292",
            "authorized_arm_order": [validator.ARM_ROLES[1], validator.ARM_ROLES[2]],
            "c_requires_bridge_revalidated_current_ab_gate": True,
            "stop_without_c_on_gate_failure": True,
            "sp4_observer_released_node292": True,
            "authorized_by": "independent-test-reviewer",
        }

    def test_bridge_capability_rejects_wrong_token_role_and_direct_bc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            token_sha = self.sha(b"fresh-a-token")
            auth_path = root / "a-auth.json"
            auth_path.write_text(json.dumps(self.phase_a_authorization(manifest, token_sha)), encoding="utf-8")
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
            with self.assertRaises(validator.E00R3Error):
                validator.validate_bridge_capability(
                    phase="A",
                    arm_role=validator.ARM_ROLES[1],
                    protocol=self.protocol,
                    protocol_path=self.protocol_path,
                    package_manifest_path=manifest,
                    authorization_path=auth_path,
                    capability_token_sha256=token_sha,
                )
            with self.assertRaises(validator.E00R3Error):
                validator.validate_bridge_capability(
                    phase="A",
                    arm_role=validator.ARM_ROLES[0],
                    protocol=self.protocol,
                    protocol_path=self.protocol_path,
                    package_manifest_path=manifest,
                    authorization_path=auth_path,
                    capability_token_sha256="f" * 64,
                )

            bc_path = root / "bc-auth.json"
            bc_path.write_text(json.dumps(self.phase_bc_authorization(manifest, self.sha(b"bc"))), encoding="utf-8")
            with self.assertRaises(validator.E00R3Error):
                validator.validate_bridge_capability(
                    phase="B",
                    arm_role=validator.ARM_ROLES[1],
                    protocol=self.protocol,
                    protocol_path=self.protocol_path,
                    package_manifest_path=manifest,
                    authorization_path=bc_path,
                    capability_token_sha256=self.sha(b"bc"),
                )
            with self.assertRaises(validator.E00R3Error):
                validator.validate_bridge_capability(
                    phase="C",
                    arm_role=validator.ARM_ROLES[2],
                    protocol=self.protocol,
                    protocol_path=self.protocol_path,
                    package_manifest_path=manifest,
                    authorization_path=bc_path,
                    capability_token_sha256=self.sha(b"bc"),
                )

    def test_builder_verifier_rejects_resealed_core_revision_and_overlay_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dfix2 = root / "dfix2"
            overlay = root / "overlay"
            output = root / "package"
            core_relative = "methods/core.py"
            overlays = (package.A_LAUNCHER, package.BC_LAUNCHER, "methods/new.py")
            (dfix2 / "methods").mkdir(parents=True)
            (dfix2 / core_relative).write_text("core\n", encoding="utf-8")
            for relative in overlays:
                path = overlay / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative + "\n", encoding="utf-8")
            core_pins = {core_relative: self.sha(b"core\n")}
            with mock.patch.object(package, "CORE_PINS", core_pins), mock.patch.object(
                package, "OVERLAY_FILES", overlays
            ):
                package.build_package(dfix2_source_tree=dfix2, overlay_root=overlay, output=output)
                manifest_path = output / package.MANIFEST_NAME
                original = json.loads(manifest_path.read_text(encoding="utf-8"))
                package.verify_package(package_root=output, manifest_path=manifest_path)
                mutations = []
                bad = copy.deepcopy(original)
                bad["dfix2_core_pins"] = {core_relative: "0" * 64}
                mutations.append(bad)
                bad = copy.deepcopy(original)
                bad["dfix2_revision"] = "attacker-resealed-revision"
                mutations.append(bad)
                bad = copy.deepcopy(original)
                bad["overlay_files"] = list(reversed(overlays))
                mutations.append(bad)
                for bad in mutations:
                    manifest_path.write_bytes(package._canonical(bad) + b"\n")
                    with self.assertRaises(package.PackageError):
                        package.verify_package(package_root=output, manifest_path=manifest_path)

    def test_bridge_and_launchers_encode_non_bypassable_order(self) -> None:
        bridge = (METHOD_ROOT / "scripts/auh_e00_three_vessel_clean_diag_r3_bridge.sh").read_text(encoding="utf-8")
        a_launcher = (METHOD_ROOT / "scripts/auh_launch_e00_three_vessel_clean_diag_r3_phase_a_only_node292.sh").read_text(encoding="utf-8")
        bc_launcher = (METHOD_ROOT / "scripts/auh_launch_e00_three_vessel_clean_diag_r3_phase_bc_node292.sh").read_text(encoding="utf-8")
        self.assertIn("fixed_parent_job=143808", bridge)
        self.assertNotIn("EXPECTED_PARENT_JOB", bridge)
        self.assertIn("E00_R3_CAPABILITY_TOKEN", bridge)
        self.assertLess(bridge.index('bridge-capability'), bridge.index('torch.distributed.run'))
        self.assertIn('phase C requires E00_R3_AB_GATE', bridge)
        self.assertEqual(a_launcher.count("srun --jobid="), 1)
        self.assertLess(
            bc_launcher.index('run_arm B "$role_b"'),
            bc_launcher.index('"$python_bin" -B "$validator" ab-gate'),
        )
        self.assertLess(
            bc_launcher.index('"$python_bin" -B "$validator" verify-ab-gate'),
            bc_launcher.index('run_arm C "$role_c"'),
        )
        for text in (bridge, a_launcher, bc_launcher):
            self.assertNotIn("sbatch ", text)
            self.assertNotIn("scancel ", text)
            self.assertNotIn("export CUDA_VISIBLE_DEVICES=", text)
            self.assertNotIn("export ROCR_VISIBLE_DEVICES=", text)


if __name__ == "__main__":
    unittest.main()
