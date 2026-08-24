from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import hashlib
import sys
import tempfile
import unittest

from methods.bernini_action_editing import (
    generic_source_anchored_action_pair_controller_v1 as pair,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


class GenericSourceAnchoredActionPairControllerTests(unittest.TestCase):
    @staticmethod
    def _receipt(
        *,
        profile: str,
        steps: int,
        action_complete: bool,
        invariants: dict | None = None,
        gpu: list[float] | None = None,
        host: list[float] | None = None,
    ) -> dict:
        unsigned = {
            "schema_version": pair.TRAINING_RECEIPT_SCHEMA,
            "complete": True,
            "execution_profile": profile,
            "stage_r_updates": steps if profile in {"smoke-r", "stage-r64"} else 0,
            "planner_updates": steps if profile == "smoke-p" else (24 if profile in {"resume-po40", "action-only40"} else 0),
            "operator_updates": steps if profile == "smoke-o" else (16 if profile in {"resume-po40", "action-only40"} else 0),
            "complete_action_result": action_complete,
            "pair_invariants": invariants or {
                "base": "a" * 64,
                "action_rows": "b" * 64,
                "planner_initial": "c" * 64,
                "operator_initial": "d" * 64,
            },
            "resources": {
                "gpu_peak_reserved_gib_by_rank": gpu or [40.0, 41.0, 42.0, 43.0],
                "host_peak_rss_gib_by_rank": host or [40.0, 41.0, 42.0, 43.0],
                "host_cgroup_peak_gib_by_rank": host or [40.0, 41.0, 42.0, 43.0],
            },
            "distributed": {
                "topology": pair.PARALLEL_TOPOLOGY,
                "world_size": pair.WORLD_SIZE,
                "one_shared_model": True,
                "same_logical_row_on_all_ranks": True,
                "rank_action_family_partition": False,
            },
        }
        return {**unsigned, "receipt_digest": pair.object_sha256(unsigned)}

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.write_bytes(pair.canonical_json_bytes(value) + b"\n")

    def test_registered_runs_are_single_model_world4_and_not_rank_partitioned(self) -> None:
        self.assertEqual(pair.WORLD_SIZE, 4)
        self.assertEqual(pair.PARALLEL_TOPOLOGY, "world4-dp1-sp4")
        self.assertEqual(
            pair.ARM_BINDINGS["joint_stage_r64"]["holder_job"], 136309
        )
        self.assertEqual(
            pair.ARM_BINDINGS["joint_resume_po40"]["holder_node"],
            "auh7-1b-gpu-280",
        )
        self.assertEqual(
            pair.ARM_BINDINGS["action_only_no_carrier"]["holder_job"], 136141
        )
        self.assertEqual(
            pair.ARM_BINDINGS["action_only_no_carrier"]["holder_node"],
            "auh7-1b-gpu-299",
        )
        self.assertEqual(
            pair.ARM_BINDINGS["action_only_no_carrier"]["carrier_policy"],
            "not_installed_or_exact_zero_frozen",
        )

    def test_r_smoke_and_r64_do_not_require_action_manifests(self) -> None:
        for arm_id in ("smoke_r", "joint_stage_r64"):
            row = pair.ARM_BINDINGS[arm_id]
            self.assertIs(row["requires_action_manifests"], False)
            self.assertIs(row["requires_resume"], False)
            self.assertIs(row["complete_action_result"], False)
        for arm_id in (
            "smoke_p",
            "smoke_o",
            "joint_resume_po40",
            "action_only_no_carrier",
        ):
            self.assertIs(
                pair.ARM_BINDINGS[arm_id]["requires_action_manifests"], True
            )

    def test_r_input_closure_does_not_require_validator_or_action_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            method_root = root / "methods" / "bernini_action_editing"
            (method_root / "scripts").mkdir(parents=True)
            files = {
                method_root / pair.TRAINER: b"trainer\n",
                method_root / pair.CORE: b"core\n",
                method_root / pair.COMMON_LAUNCHER: b"launcher\n",
                root / "source.json": b"source\n",
                root / "source.tar": b"archive\n",
                root / "source.manifest.json": b"manifest\n",
            }
            for path, raw in files.items():
                path.write_bytes(raw)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            args = Namespace(
                method_root=str(method_root),
                python_bin=str(Path(sys.executable).resolve()),
                expected_trainer_sha256=digest(method_root / pair.TRAINER),
                expected_core_sha256=digest(method_root / pair.CORE),
                expected_launcher_sha256=digest(method_root / pair.COMMON_LAUNCHER),
                expected_manifest_validator_sha256=None,
                source_manifest=str(root / "source.json"),
                expected_source_manifest_sha256=pair.SOURCE_MANIFEST_SHA256,
                method_archive=str(root / "source.tar"),
                expected_method_archive_sha256=digest(root / "source.tar"),
                method_manifest=str(root / "source.manifest.json"),
                expected_method_manifest_sha256=digest(
                    root / "source.manifest.json"
                ),
                representation_manifest=None,
                expected_representation_manifest_sha256=None,
                source_pair_manifest=None,
                expected_source_pair_manifest_sha256=None,
            )
            # Rebind only the test source bytes while retaining the production
            # requirement that the user-provided expected pin equals authority.
            original = pair.SOURCE_MANIFEST_SHA256
            observed = digest(root / "source.json")
            try:
                pair.SOURCE_MANIFEST_SHA256 = observed
                args.expected_source_manifest_sha256 = observed
                closure = pair.validate_inputs(
                    args, require_action_manifests=False
                )
            finally:
                pair.SOURCE_MANIFEST_SHA256 = original
            self.assertIsNone(closure["manifest_validator_sha256"])
            self.assertIsNone(closure["representation_manifest"])
            self.assertIsNone(closure["source_pair_manifest"])

    def test_receipt_memory_limits_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            valid = self._receipt(
                profile="smoke-r", steps=1, action_complete=False
            )
            self._write(path, valid)
            pair.validate_training_receipt(
                path,
                expected_profile="smoke-r",
                expected_steps=1,
                expected_complete_action_result=False,
            )
            invalid = self._receipt(
                profile="smoke-r",
                steps=1,
                action_complete=False,
                gpu=[40.0, 41.0, 42.0, 52.0],
            )
            self._write(path, invalid)
            with self.assertRaisesRegex(
                pair.GenericActionPairError, "memory gate differs"
            ):
                pair.validate_training_receipt(
                    path,
                    expected_profile="smoke-r",
                    expected_steps=1,
                    expected_complete_action_result=False,
                )
            invalid = self._receipt(
                profile="smoke-r",
                steps=1,
                action_complete=False,
                host=[40.0, 41.0, 42.0, 60.0],
            )
            self._write(path, invalid)
            with self.assertRaises(pair.GenericActionPairError):
                pair.validate_training_receipt(
                    path,
                    expected_profile="smoke-r",
                    expected_steps=1,
                    expected_complete_action_result=False,
                )

    def test_formal_pair_requires_identical_po_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main = root / "main.json"
            control = root / "control.json"
            invariants = {
                "base": "a" * 64,
                "representation": "b" * 64,
                "pairs": "c" * 64,
                "po_row_order": "d" * 64,
                "seed_sigma_optimizer": "e" * 64,
                "planner_initial": "f" * 64,
                "operator_initial": "0" * 64,
            }
            self._write(
                main,
                self._receipt(
                    profile="resume-po40",
                    steps=40,
                    action_complete=True,
                    invariants=invariants,
                ),
            )
            self._write(
                control,
                self._receipt(
                    profile="action-only40",
                    steps=40,
                    action_complete=True,
                    invariants=dict(invariants),
                ),
            )
            sealed = pair.seal_formal_pair(main, control)
            self.assertEqual(sealed["schema_version"], pair.PAIR_RECEIPT_SCHEMA)
            self.assertIs(sealed["parent_allocations_released"], False)
            self.assertIs(sealed["decoded_review_complete"], False)
            changed = dict(invariants)
            changed["po_row_order"] = "1" * 64
            self._write(
                control,
                self._receipt(
                    profile="action-only40",
                    steps=40,
                    action_complete=True,
                    invariants=changed,
                ),
            )
            with self.assertRaisesRegex(
                pair.GenericActionPairError, "shared base, manifests"
            ):
                pair.seal_formal_pair(main, control)

    def test_controller_signals_only_spawned_wrapper_process_groups(self) -> None:
        source = (
            METHOD_ROOT / "generic_source_anchored_action_pair_controller_v1.py"
        ).read_text(encoding="utf-8")
        lowered = source.lower()
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg(group, signal.SIGTERM)", source)
        self.assertIn("os.killpg(group, signal.SIGKILL)", source)
        self.assertNotIn("scancel", lowered)
        self.assertNotIn("scontrol release", lowered)
        self.assertNotIn("scontrol requeue", lowered)
        self.assertNotRegex(lowered, r"kill[^\n]*(136309|136141)")


if __name__ == "__main__":
    unittest.main()
