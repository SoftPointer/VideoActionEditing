from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from methods.bernini_action_editing import (
    clean_source_visual_context_pair_controller_v1 as pair,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
RUNNER = METHOD_ROOT / "train_clean_source_visual_context_stage_b_v1.py"


class CleanSourceVisualContextPairTests(unittest.TestCase):
    @staticmethod
    def _sealed(unsigned: dict) -> dict:
        return {**unsigned, "receipt_digest": pair.object_sha256(unsigned)}

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.write_bytes(pair.canonical_json_bytes(value) + b"\n")

    def _preflight(self, kind: str, invariants: dict) -> dict:
        return {
            "schema_version": pair.PREFLIGHT_RECEIPT_SCHEMA,
            "complete": True,
            "memory_input_kind": kind,
            "pair_invariants": invariants,
            "authority": {
                "optimizer_constructed": False,
                "backward_executed": False,
                "checkpoint_written": False,
            },
        }

    def test_preflight_pair_requires_identical_shared_initialization(self) -> None:
        shared = {
            "initial_parameter_digest": "a" * 64,
            "stage_a_admission_digest": None,
            "seed": 20260814,
            "digest": "b" * 64,
        }
        clean = self._preflight("clean_source", shared)
        noised = self._preflight("same_noise_forward_noised_source", dict(shared))
        clean["step0_exact_base_parity"] = {
            "local_prediction_sha256": "1" * 64
        }
        noised["step0_exact_base_parity"] = {
            "local_prediction_sha256": "2" * 64
        }
        self.assertEqual(
            pair._assert_pair_receipts(clean, noised, formal=False), shared
        )
        noised["pair_invariants"] = {**shared, "initial_parameter_digest": "c" * 64}
        with self.assertRaisesRegex(pair.CleanSourceVisualPairError, "invariants differ"):
            pair._assert_pair_receipts(clean, noised, formal=False)

    def test_formal_admission_binds_expected_initial_before_optimizer(self) -> None:
        preflight_shared_unsigned = {
            "initial_parameter_digest": "a" * 64,
            "stage_a_admission_digest": None,
            "seed": 20260814,
        }
        preflight_shared = {
            **preflight_shared_unsigned,
            "digest": pair.object_sha256(preflight_shared_unsigned),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arm_rows = []
            for index, arm in enumerate(pair.ARMS):
                run_root = root / f"preflight_arm_{index}"
                training = run_root / "training"
                training.mkdir(parents=True)
                preflight = self._sealed(
                    self._preflight(arm["memory_input_kind"], preflight_shared)
                )
                receipt_path = training / "receipt.json"
                self._write(receipt_path, preflight)
                arm_rows.append(
                    {
                        "memory_input_kind": arm["memory_input_kind"],
                        "holder_job": arm["holder_job"],
                        "holder_node": arm["holder_node"],
                        "run_root": str(run_root),
                        "master_port": 24110 + index,
                        "receipt_path": str(receipt_path),
                        "receipt_file_sha256": pair.file_sha256(receipt_path),
                        "receipt_digest": preflight["receipt_digest"],
                    }
                )
            preflight_pair = self._sealed(
                {
                    "schema_version": pair.PREFLIGHT_PAIR_SCHEMA,
                    "complete": True,
                    "decision": (
                        "shared_initialization_verified_formal_still_requires_stage_a"
                    ),
                    "optimizer_authorized": False,
                    "shared_pair_invariants": preflight_shared,
                    "allowed_arm_differences": list(pair.ALLOWED_ARM_DIFFERENCES),
                    "arms": arm_rows,
                    "parent_allocations_released": False,
                }
            )
            preflight_path = root / "preflight_pair.json"
            self._write(preflight_path, preflight_pair)
            shared_unsigned = {
                **preflight_shared_unsigned,
                "stage_a_admission_digest": "b" * 64,
            }
            shared = {
                **shared_unsigned,
                "digest": pair.object_sha256(shared_unsigned),
            }
            unsigned = {
                "schema_version": pair.FORMAL_PAIR_ADMISSION_SCHEMA,
                "complete": True,
                "decision": "admit_both_or_neither",
                "optimizer_authorized": True,
                "optimizer_authorized_by_pair_alone": False,
                "preflight_pair_receipt_digest": preflight_pair["receipt_digest"],
                "preflight_pair_receipt_path": str(preflight_path),
                "preflight_pair_receipt_file_sha256": pair.file_sha256(
                    preflight_path
                ),
                "stage_a_admission": {"receipt_digest": "b" * 64},
                "stage_a_admission_file_sha256": "d" * 64,
                "shared_pair_invariants": shared,
                "allowed_arm_differences": list(pair.ALLOWED_ARM_DIFFERENCES),
                "arms": [
                    {
                        "memory_input_kind": arm["memory_input_kind"],
                        "holder_job": arm["holder_job"],
                        "holder_node": arm["holder_node"],
                        "run_root": str(root / f"formal_arm_{index}"),
                        "master_port": 24210 + index,
                    }
                    for index, arm in enumerate(pair.ARMS)
                ],
                "synthetic_target_accessed": False,
                "reward_used": False,
            }
            receipt = self._sealed(unsigned)
            path = root / "formal_pair.json"
            self._write(path, receipt)
            loaded = pair.load_formal_pair_admission(
                path,
                expected_file_sha256=pair.file_sha256(path),
                memory_input_kind="clean_source",
                expected_shared_invariants_without_initial=shared,
            )
            self.assertEqual(loaded["expected_initial_parameter_digest"], "a" * 64)
            with self.assertRaisesRegex(
                pair.CleanSourceVisualPairError, "shared invariants differ"
            ):
                pair.load_formal_pair_admission(
                    path,
                    expected_file_sha256=pair.file_sha256(path),
                    memory_input_kind="clean_source",
                    expected_shared_invariants_without_initial={
                        **shared,
                        "seed": 7,
                    },
                )

            forged = dict(preflight_pair)
            forged["optimizer_authorized"] = True
            forged_unsigned = dict(forged)
            forged_unsigned.pop("receipt_digest")
            forged["receipt_digest"] = pair.object_sha256(forged_unsigned)
            with self.assertRaisesRegex(
                pair.CleanSourceVisualPairError,
                "decision/shared schema differs",
            ):
                pair.validate_preflight_pair_receipt(forged)

    def test_runner_gates_pair_before_optimizer_and_logs_each_step(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        main = source[source.index("def main(") :]
        self.assertLess(
            main.index("pair_contract.load_formal_pair_admission("),
            main.index("optimizer = torch.optim.AdamW("),
        )
        self.assertLess(
            main.index("actual adapter initialization differs"),
            main.index("optimizer = torch.optim.AdamW("),
        )
        self.assertIn("args.expected_initial_parameter_digest", main)
        for fragment in (
            '"global_step": step_zero_based + 1',
            '"microbatch_mean_raw_losses"',
            '"preclip_component_gradient_norms"',
            '"cuda_peak_allocated_gib"',
            '"elapsed_seconds"',
            "print(json.dumps(progress, sort_keys=True), flush=True)",
        ):
            self.assertIn(fragment, source)

    def test_controller_never_writes_bytecode_and_kills_child_sessions(self) -> None:
        source = (METHOD_ROOT / "clean_source_visual_context_pair_controller_v1.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            source.index("sys.dont_write_bytecode = True"),
            source.index("import clean_source_visual_context_stage_b_contract_v1"),
        )
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg(process_group_id, signal.SIGTERM)", source)
        self.assertIn("os.killpg(process_group_id, signal.SIGKILL)", source)
        self.assertNotIn("scancel", source.lower())


if __name__ == "__main__":
    unittest.main()
