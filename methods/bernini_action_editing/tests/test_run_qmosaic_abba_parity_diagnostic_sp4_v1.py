#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_qmosaic_abba_parity_diagnostic_sp4_v1 as diagnostic


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def projection_receipt(role: str, sp_rank: int) -> dict:
    unsigned = {
        "schema_version": diagnostic.qmosaic.ZERO_ROUTE_PROOF_SCHEMA_VERSION,
        "role": role,
        "sp_rank": sp_rank,
        "wrapper_count": len(diagnostic.qmosaic.CANONICAL_B_PARAMETER_NAMES),
        "canonical_wrapper_order_sha256": sha("order"),
        "call_evidence_sha256": sha("calls"),
        "b_state_before_sha256": sha("b"),
        "b_state_after_sha256": sha("b"),
        "total_local_row_count": 32,
        "total_selected_row_count": 16,
        "missing_wrapper_count": 0,
        "repeated_wrapper_count": 0,
        "all_selected_deltas_numerically_exact_zero": True,
        "all_base_result_raw_bytes_equal": True,
        "b_unchanged": True,
    }
    return {**unsigned, "digest": diagnostic.qmosaic.object_sha256(unsigned)}


def metric(*, exact: bool, maximum: float, relative_l2: float) -> dict:
    return {
        "exact_equal": exact,
        "max_abs": maximum,
        "relative_l2": relative_l2,
    }


def classified_pairs(
    *,
    within: tuple[float, float] = (0.0, 0.0),
    mode: float = 0.0,
    cross: float = 0.0,
    failed_gate: float = 0.0,
) -> dict:
    rows = {}
    for name, _, _, category in diagnostic.PAIR_SPECS:
        if category == "within_off":
            value = within[0]
        elif category == "within_on0":
            value = within[1]
        elif category == "mode":
            value = mode
        elif category == "cross":
            value = cross
        else:
            value = failed_gate
        rows[name] = metric(exact=value == 0.0, maximum=value, relative_l2=value)
    return rows


def call_execution() -> dict:
    rows = {}
    for call in diagnostic.COMPLETE_CALL_ORDER:
        rows[call] = {
            "adapter_enabled": call.startswith("N") or call == "P",
            "grad_enabled": call != "M",
            "inference_mode_enabled": False,
            "detach_observer": call == "M",
            "source": "unit-test",
        }
    return rows


def tiny_role(role: str, sp_rank: int) -> dict:
    value = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 4)
    tensors = {name: value.clone() for name in diagnostic.COMPLETE_CALL_ORDER}
    raw = {
        name: value.to(dtype=torch.bfloat16).reshape(2, 4).clone()
        for name in diagnostic.COMPLETE_CALL_ORDER
    }
    return dict(
        diagnostic.build_role_diagnostic(
            role=role,
            sp_rank=sp_rank,
            tensors=tensors,
            raw_target_tensors=raw,
            call_execution=call_execution(),
            projection_proof=projection_receipt(role, sp_rank),
        )
    )


def provenance() -> dict:
    return {
        "method_source_revision": "a" * 40,
        "method_source_archive_sha256": sha("archive"),
        "checkpoint_content_receipt_digest": sha("checkpoint"),
        "owner_packet_receipt_digest": sha("owner"),
        "editor_runtime_input_receipt_digest": sha("editor"),
        "runner_contract_digest": sha("runner"),
        "collective_receipt_digest": sha("collective"),
    }


class PairMetricTests(unittest.TestCase):
    def test_exact_and_signed_zero_raw_mismatch_are_distinguished(self) -> None:
        exact = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        row = diagnostic.compare_fp32_tensors(
            exact, exact.clone(), left_label="left", right_label="right"
        )
        self.assertTrue(row["exact_equal"])
        self.assertEqual(row["raw_mismatch_count"], 0)
        self.assertEqual(row["max_abs"], 0.0)
        self.assertEqual(row["rms"], 0.0)
        self.assertEqual(row["relative_l2"], 0.0)

        negative_zero = torch.tensor([-0.0, 1.0], dtype=torch.float32)
        positive_zero = torch.tensor([0.0, 1.0], dtype=torch.float32)
        signed = diagnostic.compare_fp32_tensors(
            negative_zero,
            positive_zero,
            left_label="negative-zero",
            right_label="positive-zero",
        )
        self.assertTrue(signed["torch_equal"])
        self.assertFalse(signed["raw_exact_equal"])
        self.assertFalse(signed["exact_equal"])
        self.assertEqual(signed["raw_mismatch_count"], 1)
        self.assertEqual(signed["numerical_mismatch_count"], 0)
        self.assertEqual(signed["max_abs"], 0.0)

    def test_metric_records_existing_replay_bound_without_using_it_as_verdict(self) -> None:
        left = torch.tensor([1.0, 2.0], dtype=torch.float32)
        right = torch.tensor([1.0, 2.0 + 1.0e-6], dtype=torch.float32)
        row = diagnostic.compare_fp32_tensors(
            left, right, left_label="left", right_label="right"
        )
        self.assertFalse(row["exact_equal"])
        self.assertGreater(row["max_abs"], 0.0)
        self.assertGreater(row["rms"], 0.0)
        self.assertGreater(row["relative_l2"], 0.0)
        expected = float(
            torch.linalg.vector_norm((left.double() - right.double())).item()
            / max(
                float(torch.linalg.vector_norm(left.double()).item()),
                float(torch.linalg.vector_norm(right.double()).item()),
            )
        )
        self.assertAlmostEqual(row["relative_l2"], expected)
        self.assertEqual(
            row["relative_l2_policy"],
            "l2_difference/max(l2_left,l2_right,1e-30)",
        )
        self.assertTrue(row["within_existing_replay_bound"])


class VerdictTests(unittest.TestCase):
    def test_exact_wrapper_repeatability_and_inconclusive_verdicts(self) -> None:
        verdict, _ = diagnostic.classify_role(
            classified_pairs(), projection_passed=True
        )
        self.assertEqual(verdict, diagnostic.EXACT_TRANSIENT)

        verdict, _ = diagnostic.classify_role(
            classified_pairs(), projection_passed=False
        )
        self.assertEqual(verdict, diagnostic.A_WRAPPER_ROUTE)

        verdict, evidence = diagnostic.classify_role(
            classified_pairs(
                within=(2.0e-6, 1.0e-6),
                mode=1.5e-6,
                cross=1.8e-6,
                failed_gate=1.7e-6,
            ),
            projection_passed=True,
        )
        self.assertEqual(verdict, diagnostic.B_REPEATABILITY)
        self.assertTrue(evidence["cross_route_inside_predeclared_envelope"])
        self.assertEqual(evidence["combined_envelope"]["inflation_factor"], 1.0)

        verdict, evidence = diagnostic.classify_role(
            classified_pairs(
                within=(1.0e-6, 1.0e-6),
                mode=1.0e-6,
                cross=2.0e-6,
                failed_gate=1.0e-6,
            ),
            projection_passed=True,
        )
        self.assertEqual(verdict, diagnostic.INCONCLUSIVE)
        self.assertFalse(evidence["cross_route_inside_predeclared_envelope"])

    def test_combination_is_fail_closed(self) -> None:
        self.assertEqual(
            diagnostic.combine_verdicts(
                [diagnostic.EXACT_TRANSIENT, diagnostic.EXACT_TRANSIENT]
            ),
            diagnostic.EXACT_TRANSIENT,
        )
        self.assertEqual(
            diagnostic.combine_verdicts(
                [diagnostic.B_REPEATABILITY, diagnostic.EXACT_TRANSIENT]
            ),
            diagnostic.B_REPEATABILITY,
        )
        self.assertEqual(
            diagnostic.combine_verdicts(
                [diagnostic.B_REPEATABILITY, diagnostic.INCONCLUSIVE]
            ),
            diagnostic.INCONCLUSIVE,
        )
        self.assertEqual(
            diagnostic.combine_verdicts(
                [diagnostic.B_REPEATABILITY, diagnostic.A_WRAPPER_ROUTE]
            ),
            diagnostic.A_WRAPPER_ROUTE,
        )

    def test_role_packet_closes_M_ABBA_P_and_excludes_P_from_envelope(self) -> None:
        row = tiny_role("action", 0)
        self.assertEqual(row["call_order"], list(diagnostic.COMPLETE_CALL_ORDER))
        self.assertEqual(row["abba_envelope_call_order"], ["O0", "N0", "N1", "O1"])
        self.assertTrue(row["projection_call_excluded_from_envelope"])
        self.assertTrue(row["projection_proof_passed"])
        self.assertEqual(row["verdict"], diagnostic.EXACT_TRANSIENT)
        self.assertEqual(set(row["pairs"]), {name for name, *_ in diagnostic.PAIR_SPECS})
        self.assertEqual(
            set(row["raw_block15_target_pairs"]),
            {name for name, *_ in diagnostic.PAIR_SPECS},
        )
        self.assertEqual(
            row["raw_vs_sketch_attribution"]["classification"],
            "NO_NONEXACT_OBSERVED",
        )
        self.assertFalse(row["call_execution"]["M"]["adapter_enabled"])
        self.assertFalse(row["call_execution"]["M"]["grad_enabled"])
        self.assertTrue(row["call_execution"]["M"]["detach_observer"])
        for call in diagnostic.ABBA_CALL_ORDER + ("P",):
            self.assertTrue(row["call_execution"][call]["grad_enabled"])
            self.assertFalse(row["call_execution"][call]["detach_observer"])

    def test_raw_exact_sketch_nonexact_is_explicitly_attributed(self) -> None:
        base = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 4)
        tensors = {name: base.clone() for name in diagnostic.COMPLETE_CALL_ORDER}
        tensors["O1"][0, 0, 0] += 1.0e-5
        tensors["N0"][0, 0, 0] += 5.0e-6
        raw_base = base.to(dtype=torch.bfloat16).reshape(2, 4)
        raw = {name: raw_base.clone() for name in diagnostic.COMPLETE_CALL_ORDER}
        row = diagnostic.build_role_diagnostic(
            role="action",
            sp_rank=0,
            tensors=tensors,
            raw_target_tensors=raw,
            call_execution=call_execution(),
            projection_proof=projection_receipt("action", 0),
        )
        attribution = row["raw_vs_sketch_attribution"]
        self.assertTrue(attribution["raw_exact_while_sketch_nonexact"])
        self.assertEqual(attribution["raw_nonexact_pair_names"], [])
        self.assertEqual(attribution["classification"], "RAW_EXACT_SKETCH_NONEXACT")

    def test_empty_raw_target_shard_has_finite_zero_metrics(self) -> None:
        left = torch.empty(0, 1536, dtype=torch.bfloat16)
        row = diagnostic.compare_floating_tensors(
            left,
            left.clone(),
            left_label="empty-left",
            right_label="empty-right",
        )
        self.assertTrue(row["empty_tensor"])
        self.assertTrue(row["exact_equal"])
        self.assertEqual(row["max_abs"], 0.0)
        self.assertEqual(row["rms"], 0.0)
        self.assertEqual(row["relative_l2"], 0.0)


class ReceiptAndAggregationTests(unittest.TestCase):
    def rank_receipt(self, root: Path, *, cell: str, seed: int, rank: int) -> dict:
        roles = {role: tiny_role(role, rank) for role in diagnostic.ROLE_ORDER}
        path = root / diagnostic.RANK_RECEIPT_BASENAME.format(sp_rank=rank)
        return dict(
            diagnostic.build_rank_receipt(
                cell_id=cell,
                query_seed=seed,
                sp_rank=rank,
                world_rank=rank,
                role_diagnostics=roles,
                provenance=provenance(),
                runtime_environment={"torch_version": torch.__version__},
                parameter_invariance={
                    "parameter_bytes_unchanged": True,
                    "lora_b_exact_zero_before": True,
                    "lora_b_exact_zero_after": True,
                    "optimizer_created": False,
                    "parameter_update_performed": False,
                },
                terminal_full_seal={"sp_rank": rank, "digest": sha(f"terminal-{rank}")},
                output_path=str(path),
            )
        )

    def write_world4(self, root: Path, *, cell: str, seed: int) -> Path:
        root.mkdir()
        receipts = [
            self.rank_receipt(root, cell=cell, seed=seed, rank=rank)
            for rank in range(diagnostic.SP_SIZE)
        ]
        artifacts = []
        for rank, receipt in enumerate(receipts):
            path = root / diagnostic.RANK_RECEIPT_BASENAME.format(sp_rank=rank)
            diagnostic.write_create_only_json(path, receipt)
            artifacts.append(
                {
                    "sp_rank": rank,
                    "path": str(path),
                    "file_sha256": diagnostic.file_sha256(path),
                    "receipt_digest": receipt["receipt_digest"],
                }
            )
        world4 = diagnostic.build_world4_receipt(
            rank_receipts=receipts, rank_artifacts=artifacts
        )
        path = root / diagnostic.WORLD4_RECEIPT_BASENAME
        diagnostic.write_create_only_json(path, world4)
        return path

    def test_create_only_rank_world4_and_all8_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dog = self.write_world4(
                root / "dog", cell="dog", seed=diagnostic.FIXED_QUERY_SEEDS["dog"][0]
            )
            human = self.write_world4(
                root / "human",
                cell="human",
                seed=diagnostic.FIXED_QUERY_SEEDS["human"][0],
            )
            output = root / diagnostic.ALL8_MANIFEST_BASENAME
            manifest = diagnostic.aggregate_all8(
                dog_world4_receipt=dog,
                human_world4_receipt=human,
                output=output,
            )
            self.assertEqual(manifest["rank_receipt_count"], 8)
            self.assertEqual(manifest["verdict"], diagnostic.EXACT_TRANSIENT)
            self.assertFalse(manifest["execution_authority"]["decode_executed"])
            self.assertFalse(manifest["execution_authority"]["vjp_executed"])
            self.assertFalse(
                manifest["execution_authority"]["parameter_update_authorized"]
            )
            with self.assertRaises(diagnostic.QMosaicABBAParityDiagnosticError):
                diagnostic.aggregate_all8(
                    dog_world4_receipt=dog,
                    human_world4_receipt=human,
                    output=output,
                )

    def test_tampered_rank_receipt_blocks_all8_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            dog = self.write_world4(
                root / "dog", cell="dog", seed=diagnostic.FIXED_QUERY_SEEDS["dog"][0]
            )
            human = self.write_world4(
                root / "human",
                cell="human",
                seed=diagnostic.FIXED_QUERY_SEEDS["human"][0],
            )
            rank = dog.parent / diagnostic.RANK_RECEIPT_BASENAME.format(sp_rank=0)
            rank.write_bytes(rank.read_bytes() + b" ")
            with self.assertRaisesRegex(
                diagnostic.QMosaicABBAParityDiagnosticError, "artifact changed"
            ):
                diagnostic.aggregate_all8(
                    dog_world4_receipt=dog,
                    human_world4_receipt=human,
                    output=root / diagnostic.ALL8_MANIFEST_BASENAME,
                )


class ClosedCLIContractTests(unittest.TestCase):
    def test_runtime_explicitly_reproduces_historical_M_without_editor_seal(self) -> None:
        measurement = inspect.getsource(diagnostic._historical_measure_to_cpu)
        replay = inspect.getsource(diagnostic._replay_to_cpu)
        runtime = inspect.getsource(diagnostic.run_world4)
        self.assertIn("with torch.no_grad()", measurement)
        self.assertIn("adapter_enabled=False", measurement)
        self.assertIn("detach=True", measurement)
        self.assertIn("with torch.enable_grad()", replay)
        self.assertIn("detach=False", replay)
        self.assertNotIn("seal_editor_packet", runtime)
        self.assertNotIn("replay_session", runtime)

    def test_only_diagnostic_and_aggregation_subcommands_exist(self) -> None:
        parser = diagnostic.build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        )
        self.assertEqual(set(subparsers.choices), {"run-world4", "aggregate-all8"})
        runtime_options = {
            option
            for action in subparsers.choices["run-world4"]._actions
            for option in action.option_strings
        }
        self.assertIn("--diagnostic-only", runtime_options)
        for forbidden in ("--decode", "--vjp", "--optimizer", "--train", "--update"):
            self.assertNotIn(forbidden, runtime_options)


if __name__ == "__main__":
    unittest.main()
