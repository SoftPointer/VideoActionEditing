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

import e00_legacy_infer_fixed_rng_wrapper_v2 as wrapper
import validate_e00_three_vessel_fresh_keyed_two_phase_diagnostic_v2 as validator
from tools import build_e00_three_vessel_clean_diag_r2_package_v2 as package


class E00CleanDiagnosticR2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = validator.DEFAULT_PROTOCOL
        self.protocol = copy.deepcopy(validator.load_protocol(self.protocol_path))

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def fixed_rng_receipts(self, role: str) -> list[dict]:
        protocol_id = validator.protocol_identity(self.protocol, self.protocol_path)
        rows = []
        for rank, seeds in enumerate(validator.EXPECTED_SEEDS):
            rows.append(
                {
                    "schema_version": validator.RNG_SCHEMA,
                    "revision_tag": validator.REVISION_TAG,
                    "arm_role": role,
                    "rank": rank,
                    "protocol": copy.deepcopy(protocol_id),
                    "fixed_initial_rng": {
                        "enabled": True,
                        "scheme": "explicit_rank_owned_cpu_cuda_manual_seed_v2",
                        "scope": (
                            "inside_fork_rng_immediately_before_entire_legacy_inference_entrypoint"
                        ),
                        "cpu_seed": seeds["cpu_seed"],
                        "cuda_seed": seeds["cuda_seed"],
                        "seeded_initial": {
                            "cpu_sha256": self._digest(f"cpu-{rank}"),
                            "cuda_sha256": self._digest(f"cuda-{rank}"),
                        },
                        "terminal_before_restore": {
                            "cpu_sha256": self._digest(f"terminal-cpu-{rank}"),
                            "cuda_sha256": self._digest(f"terminal-cuda-{rank}"),
                        },
                    },
                }
            )
        return rows

    @staticmethod
    def fixed_rows() -> list[dict]:
        return [
            {
                "rank": rank,
                "cpu_seed": seeds["cpu_seed"],
                "cuda_seed": seeds["cuda_seed"],
                "cpu_initial_sha256": hashlib.sha256(
                    f"cpu-{rank}".encode("utf-8")
                ).hexdigest(),
                "cuda_initial_sha256": hashlib.sha256(
                    f"cuda-{rank}".encode("utf-8")
                ).hexdigest(),
            }
            for rank, seeds in enumerate(validator.EXPECTED_SEEDS)
        ]

    def arm_audit(
        self,
        role: str,
        *,
        video: Path,
        latent_sha: str,
        fixed_rows: list[dict] | None = None,
    ) -> dict:
        rows = copy.deepcopy(fixed_rows if fixed_rows is not None else self.fixed_rows())
        video_sha = validator.file_sha256(video)
        return {
            "schema_version": validator.ARM_AUDIT_SCHEMA,
            "revision_tag": validator.REVISION_TAG,
            "complete": True,
            "arm_role": role,
            "anchor_free": False,
            "training_performed": False,
            "optimization_steps": 0,
            "protocol": validator.protocol_identity(self.protocol, self.protocol_path),
            "spec_canonical_sha256": "1" * 64,
            "artifacts": {
                "video_path": str(video),
                "video_sha256": video_sha,
            },
            "native": {
                "native_output_sha256": video_sha,
                "outer_schedule_digest": "schedule-v1",
                "frozen_certificate_sha256": "2" * 64,
                "target_route_replay_steps": (
                    40 if role == validator.ARM_ROLES[2] else 0
                ),
            },
            "rng_and_noise": {
                "explicit_initial_rng": True,
                "per_rank_fixed_initial_rng": rows,
                "per_rank_fixed_initial_rng_digest": validator.canonical_sha256(rows),
                "raw_noise_rows": [{"step": step, "sha256": self._digest(str(step))} for step in range(40)],
                "raw_noise_bank_sha256": "3" * 64,
                "predecode_latent_sha256": latent_sha,
            },
        }

    def test_protocol_closes_full_two_phase_order_and_static_placeholder(self) -> None:
        value = validator.validate_protocol(self.protocol)
        self.assertEqual(value["revision_tag"], validator.REVISION_TAG)
        self.assertEqual(self.protocol["arm_order"], list(validator.ARM_ROLES))
        states = [row["state"] for row in self.protocol["two_phase_state_machine"]]
        self.assertEqual(states[1], "EXTERNAL_A_ONLY_AUTHORIZATION")
        self.assertEqual(states[4], "A_B_BIT_EXACT_GATE")
        self.assertEqual(
            self.protocol["two_phase_state_machine"][4]["on_failure"],
            "STOP_WITHOUT_C",
        )
        base = validator.load_bound_base_spec(self.protocol)
        self.assertNotEqual(
            base["data_and_prompt_contract"]["source_video_sha256"],
            base["data_and_prompt_contract"]["pure_noobserver_placeholder"]["sha256"],
        )

    def test_protocol_mutations_fail_closed(self) -> None:
        bad = copy.deepcopy(self.protocol)
        bad["arm_order"][1], bad["arm_order"][2] = bad["arm_order"][2], bad["arm_order"][1]
        with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
            validator.validate_protocol(bad)
        bad = copy.deepcopy(self.protocol)
        bad["fixed_initial_rng"]["per_rank"][2]["cuda_seed"] += 1
        with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
            validator.validate_protocol(bad)
        bad = copy.deepcopy(self.protocol)
        bad["two_phase_state_machine"][4]["on_failure"] = "RUN_C_ANYWAY"
        with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
            validator.validate_protocol(bad)

    def test_cpu_fixed_rng_is_repeatable_and_restores_caller(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        caller = torch.random.get_rng_state().clone()
        first, first_fork, first_fixed = wrapper.run_with_fixed_initial_rng(
            torch,
            lambda: torch.rand(8),
            cuda_device=None,
            cpu_seed=82002700,
            cuda_seed=None,
        )
        second, second_fork, second_fixed = wrapper.run_with_fixed_initial_rng(
            torch,
            lambda: torch.rand(8),
            cuda_device=None,
            cpu_seed=82002700,
            cuda_seed=None,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(
            first_fixed["seeded_initial"], second_fixed["seeded_initial"]
        )
        self.assertTrue(first_fork["cpu_state_restored"])
        self.assertTrue(second_fork["cpu_state_restored"])
        self.assertTrue(torch.equal(caller, torch.random.get_rng_state()))

    def test_fixed_rng_receipt_binds_seed_and_initial_state(self) -> None:
        base_result = {
            "rank_count": 4,
            "all_rank_rng_state_restored": True,
            "raw_noise_rows": [],
            "raw_noise_bank_sha256": "4" * 64,
            "route_application_enabled": False,
            "predecode_latent_sha256": "5" * 64,
        }
        rows = self.fixed_rng_receipts(validator.ARM_ROLES[0])
        with mock.patch.object(
            validator.legacy, "validate_rng_receipts", return_value=base_result
        ):
            value = validator.validate_fixed_rng_receipts(
                rows,
                protocol=self.protocol,
                protocol_path=self.protocol_path,
                arm_role=validator.ARM_ROLES[0],
                expected_output_path="/out/a.mp4",
                native_receipt_sha256="6" * 64,
            )
        self.assertTrue(value["explicit_initial_rng"])
        self.assertEqual(value["per_rank_fixed_initial_rng"], self.fixed_rows())
        rows[1]["fixed_initial_rng"]["cpu_seed"] += 1
        with mock.patch.object(
            validator.legacy, "validate_rng_receipts", return_value=base_result
        ):
            with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
                validator.validate_fixed_rng_receipts(
                    rows,
                    protocol=self.protocol,
                    protocol_path=self.protocol_path,
                    arm_role=validator.ARM_ROLES[0],
                    expected_output_path="/out/a.mp4",
                    native_receipt_sha256="6" * 64,
                )

    def test_ab_gate_requires_latent_mp4_and_fixed_rng_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_a = root / "a.mp4"
            video_b = root / "b.mp4"
            video_a.write_bytes(b"same-mp4-bytes")
            video_b.write_bytes(b"same-mp4-bytes")
            latent = "7" * 64
            a = self.arm_audit(validator.ARM_ROLES[0], video=video_a, latent_sha=latent)
            b = self.arm_audit(validator.ARM_ROLES[1], video=video_b, latent_sha=latent)
            audit_a = root / "a.audit.json"
            audit_b = root / "b.audit.json"
            audit_a.write_text(json.dumps(a, sort_keys=True), encoding="utf-8")
            audit_b.write_text(json.dumps(b, sort_keys=True), encoding="utf-8")
            gate = validator.build_ab_gate(
                a_audit=a,
                b_audit=b,
                a_audit_path=audit_a,
                b_audit_path=audit_b,
                a_video=video_a,
                b_video=video_b,
            )
            self.assertTrue(gate["c_execution_gate_passed"])

            changed = copy.deepcopy(b)
            changed["rng_and_noise"]["predecode_latent_sha256"] = "8" * 64
            with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
                validator.build_ab_gate(
                    a_audit=a,
                    b_audit=changed,
                    a_audit_path=audit_a,
                    b_audit_path=audit_b,
                    a_video=video_a,
                    b_video=video_b,
                )

            changed = copy.deepcopy(b)
            changed["rng_and_noise"]["per_rank_fixed_initial_rng"][0][
                "cpu_initial_sha256"
            ] = "9" * 64
            changed["rng_and_noise"]["per_rank_fixed_initial_rng_digest"] = (
                validator.canonical_sha256(
                    changed["rng_and_noise"]["per_rank_fixed_initial_rng"]
                )
            )
            with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
                validator.build_ab_gate(
                    a_audit=a,
                    b_audit=changed,
                    a_audit_path=audit_a,
                    b_audit_path=audit_b,
                    a_video=video_a,
                    b_video=video_b,
                )

            video_b.write_bytes(b"different-mp4-bytes")
            with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
                validator.build_ab_gate(
                    a_audit=a,
                    b_audit=b,
                    a_audit_path=audit_a,
                    b_audit_path=audit_b,
                    a_video=video_a,
                    b_video=video_b,
                )

    def test_gate_digest_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_a = root / "a.mp4"
            video_b = root / "b.mp4"
            video_a.write_bytes(b"same")
            video_b.write_bytes(b"same")
            a = self.arm_audit(validator.ARM_ROLES[0], video=video_a, latent_sha="a" * 64)
            b = self.arm_audit(validator.ARM_ROLES[1], video=video_b, latent_sha="a" * 64)
            audit_a = root / "a.json"
            audit_b = root / "b.json"
            audit_a.write_text(json.dumps(a), encoding="utf-8")
            audit_b.write_text(json.dumps(b), encoding="utf-8")
            gate = validator.build_ab_gate(
                a_audit=a,
                b_audit=b,
                a_audit_path=audit_a,
                b_audit_path=audit_b,
                a_video=video_a,
                b_video=video_b,
            )
            gate["only_admitted_next_arm"] = validator.ARM_ROLES[1]
            with self.assertRaises(validator.E00TwoPhaseDiagnosticError):
                validator.validate_existing_ab_gate(
                    gate,
                    a_audit=a,
                    b_audit=b,
                    a_audit_path=audit_a,
                    b_audit_path=audit_b,
                    a_video=video_a,
                    b_video=video_b,
                )

    def test_launchers_enforce_a_stop_then_b_gate_c_and_slurm_visibility(self) -> None:
        a_launcher = (METHOD_ROOT / "scripts/auh_launch_e00_three_vessel_clean_diag_r2_phase_a_only_node292.sh").read_text(encoding="utf-8")
        bc_launcher = (METHOD_ROOT / "scripts/auh_launch_e00_three_vessel_clean_diag_r2_phase_bc_node292.sh").read_text(encoding="utf-8")
        bridge = (METHOD_ROOT / "scripts/auh_e00_three_vessel_clean_diag_r2_bridge.sh").read_text(encoding="utf-8")
        self.assertEqual(a_launcher.count("srun --jobid="), 1)
        self.assertIn("A_STOPPED_REVIEW_REQUIRED", a_launcher)
        self.assertIn("bc_execution_authorized == false", a_launcher)
        self.assertLess(
            bc_launcher.index('run_arm "$role_b"'),
            bc_launcher.index('"$python_bin" -B "$validator" ab-gate'),
        )
        self.assertLess(
            bc_launcher.index('"$python_bin" -B "$validator" ab-gate'),
            bc_launcher.index('run_arm "$role_c"'),
        )
        self.assertIn("phase_a_stopped_marker_sha256", bc_launcher)
        self.assertIn("phase_a_arm_audit_sha256", bc_launcher)
        self.assertIn("phase_a_mp4_sha256", bc_launcher)
        for text in (a_launcher, bc_launcher, bridge):
            self.assertNotIn("sbatch ", text)
            self.assertNotIn("scancel ", text)
            self.assertNotIn("export CUDA_VISIBLE_DEVICES=", text)
            self.assertNotIn("export ROCR_VISIBLE_DEVICES=", text)
            self.assertNotIn("unset CUDA_VISIBLE_DEVICES", text)
            self.assertNotIn("unset ROCR_VISIBLE_DEVICES", text)

    def test_package_marker_names_full_order_and_resists_semantic_reseal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dfix2 = root / "dfix2"
            overlay = root / "overlay"
            output = root / "package"
            core_relative = "methods/core.py"
            extra_relative = "methods/new.py"
            paths = (extra_relative, package.A_LAUNCHER, package.BC_LAUNCHER)
            (dfix2 / "methods").mkdir(parents=True)
            (dfix2 / core_relative).write_text("core\n", encoding="utf-8")
            for relative in paths:
                path = overlay / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative}\n", encoding="utf-8")
            expected = {core_relative: hashlib.sha256(b"core\n").hexdigest()}
            package.build_package(
                dfix2_source_tree=dfix2,
                overlay_root=overlay,
                output=output,
                expected_core=expected,
                overlay_files=paths,
            )
            manifest_path = output / package.MANIFEST_NAME
            value = package.verify_package(
                package_root=output, manifest_path=manifest_path
            )
            self.assertEqual(value["full_arm_order"], package.ARM_ORDER)
            marker_path = output / package.REVIEW_MARKER
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["full_arm_order"], package.ARM_ORDER)
            self.assertTrue(marker["phase_a_must_stop"])
            self.assertTrue(marker["ab_mp4_bit_exact_required_before_c"])

            # Attempt to reseal file hashes after swapping B/C.  The semantic
            # review-marker validator must still reject the package.
            marker["full_arm_order"][1], marker["full_arm_order"][2] = (
                marker["full_arm_order"][2],
                marker["full_arm_order"][1],
            )
            marker_path.write_bytes(package._canonical(marker) + b"\n")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in manifest["files"]:
                if row["path"] == package.REVIEW_MARKER:
                    row["sha256"] = package._sha256(marker_path)
                    row["bytes"] = marker_path.stat().st_size
            manifest["review_marker_sha256"] = package._sha256(marker_path)
            manifest["content_digest"] = hashlib.sha256(
                package._canonical(manifest["files"])
            ).hexdigest()
            manifest_path.write_bytes(package._canonical(manifest) + b"\n")
            with self.assertRaises(package.PackageError):
                package.verify_package(package_root=output, manifest_path=manifest_path)


if __name__ == "__main__":
    unittest.main()
