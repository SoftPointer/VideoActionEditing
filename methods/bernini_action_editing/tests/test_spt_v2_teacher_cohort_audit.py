#!/usr/bin/env python3
"""Contracts for the read-only hardened-SPT trust cohort selector."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SPT_ROOT = METHOD_ROOT / "spt_v2"
for root in (METHOD_ROOT, SPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import audit_teacher_cohort as cohort  # noqa: E402
import phase_transport as spt  # noqa: E402
import train_lora as legacy  # noqa: E402


class _Dataset:
    def __init__(self, count: int):
        self.signature = "dataset-signature"
        self.rows = [
            {"iid": f"iid-{index:03d}", "inputs": {"instruction": str(index)}}
            for index in range(count)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _summary() -> dict[str, str]:
    return {
        "sha256": "a" * 64,
        "summary_digest": "b" * 64,
        "index_sha256": "c" * 64,
    }


def _report(index: int, *, selected: bool) -> dict:
    row = {"iid": f"iid-{index:03d}", "inputs": {"instruction": str(index)}}
    return {
        "row_index": index,
        "iid": row["iid"],
        "identity_sha256": legacy.dataset_identity(row, index),
        "selection": {
            "selected": selected,
            "criteria": {
                "prebudget_generate_fraction": selected,
                "postbudget_transport_fraction": selected,
                "proxy_relative_improvement_over_copy": selected,
                "postbudget_generate_fraction_per_phase": selected,
            },
            "rejection_reasons": [] if selected else ["synthetic-test"],
        },
    }


class TeacherCohortPureTests(unittest.TestCase):
    def _argv(self) -> list[str]:
        return [
            "--checkpoint", "/checkpoint",
            "--preprocessed-parquet-dir", "/data",
            "--dataset-summary", "/summary.json",
            "--output-dir", "/output",
            "--method-source-revision", "1" * 40,
            "--method-source-archive-sha256", "2" * 64,
        ]

    def test_defaults_are_the_requested_four_hard_thresholds(self) -> None:
        args = cohort.build_parser().parse_args(self._argv())
        thresholds = cohort.validate_cli(args)
        self.assertEqual(
            thresholds,
            cohort.SelectorThresholds(
                max_prebudget_generate_fraction=0.25,
                min_postbudget_transport_fraction=0.03,
                min_proxy_relative_improvement_over_copy=0.40,
                max_postbudget_generate_fraction_per_phase=0.12,
            ),
        )
        self.assertEqual(args.minimum_selected, 8)
        self.assertFalse(args.allow_insufficient_selection)
        self.assertFalse(hasattr(args, "max_steps"))
        self.assertFalse(hasattr(args, "learning_rate"))

    def test_scan_supports_default_prefix_explicit_prefix_and_range(self) -> None:
        parser = cohort.build_parser()
        default = parser.parse_args(self._argv())
        self.assertEqual(
            cohort.resolve_scan_window(default, 200),
            cohort.ScanWindow("ordered_prefix", 0, 64),
        )
        prefix = parser.parse_args(self._argv() + ["--prefix-rows", "128"])
        self.assertEqual(
            cohort.resolve_scan_window(prefix, 200).row_indices,
            tuple(range(128)),
        )
        explicit = parser.parse_args(
            self._argv() + ["--row-range", "64", "128"]
        )
        window = cohort.resolve_scan_window(explicit, 200)
        self.assertEqual(window.receipt()["mode"], "explicit_half_open_range")
        self.assertEqual(window.row_indices, tuple(range(64, 128)))
        invalid = parser.parse_args(
            self._argv() + ["--row-range", "128", "129"]
        )
        with self.assertRaisesRegex(cohort.TeacherCohortAuditError, "minimum-selected"):
            cohort.resolve_scan_window(invalid, 200)

    def test_selection_is_an_explicit_conjunction(self) -> None:
        thresholds = cohort.SelectorThresholds()
        report = {
            "prebudget_generate_fraction": 0.25,
            "postbudget_gate_fraction": {"transport": 0.03},
            "proxy_relative_improvement_over_copy": 0.40,
            "observed_max_postbudget_generate_fraction_per_phase": 0.12,
        }
        decision = cohort.selection_decision(report, thresholds)
        self.assertTrue(decision["selected"])
        self.assertEqual(decision["rejection_reasons"], [])
        for key, bad_value in (
            ("prebudget_generate_fraction", 0.250001),
            ("proxy_relative_improvement_over_copy", 0.399999),
            ("observed_max_postbudget_generate_fraction_per_phase", 0.12001),
        ):
            candidate = dict(report)
            candidate[key] = bad_value
            self.assertFalse(
                cohort.selection_decision(candidate, thresholds)["selected"]
            )
        candidate = dict(report)
        candidate["postbudget_gate_fraction"] = {"transport": 0.029999}
        self.assertFalse(
            cohort.selection_decision(candidate, thresholds)["selected"]
        )

    def test_membership_is_ordered_hash_bound_and_strict_loadable(self) -> None:
        dataset = _Dataset(3)
        reports = [_report(2, selected=True), _report(0, selected=True), _report(1, selected=False)]
        membership = cohort.build_selected_membership(
            reports=reports,
            dataset=dataset,
            dataset_summary=_summary(),
            scan=cohort.ScanWindow("ordered_prefix", 0, 3),
            thresholds=cohort.SelectorThresholds(),
            minimum_selected=2,
        )
        self.assertEqual(membership["ordered_selected_row_indices"], [0, 2])
        self.assertEqual(
            [member["iid"] for member in membership["members"]],
            ["iid-000", "iid-002"],
        )
        self.assertTrue(membership["sufficient"])
        self.assertRegex(membership["membership_digest"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            cohort.validate_selected_membership(
                membership, dataset=dataset, dataset_summary=_summary()
            ),
            (0, 2),
        )
        self.assertTrue(
            membership["trainer_load_contract"][
                "recompute_each_row_identity_sha256"
            ]
        )

        tampered = json.loads(json.dumps(membership))
        tampered["members"][0]["iid"] = "wrong"
        with self.assertRaisesRegex(cohort.TeacherCohortAuditError, "digest"):
            cohort.validate_selected_membership(
                tampered, dataset=dataset, dataset_summary=_summary()
            )

    def test_membership_loader_recomputes_dataset_row_identity(self) -> None:
        dataset = _Dataset(2)
        membership = cohort.build_selected_membership(
            reports=[_report(0, selected=True), _report(1, selected=True)],
            dataset=dataset,
            dataset_summary=_summary(),
            scan=cohort.ScanWindow("ordered_prefix", 0, 2),
            thresholds=cohort.SelectorThresholds(),
            minimum_selected=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected_membership.json"
            path.write_text(json.dumps(membership), encoding="utf-8")
            loaded, rows = cohort.load_selected_membership(
                path, dataset=dataset, dataset_summary=_summary()
            )
            self.assertEqual(loaded["membership_digest"], membership["membership_digest"])
            self.assertEqual(rows, (0, 1))
            dataset.rows[1]["inputs"]["instruction"] = "changed"
            with self.assertRaisesRegex(cohort.TeacherCohortAuditError, "identity"):
                cohort.load_selected_membership(
                    path, dataset=dataset, dataset_summary=_summary()
                )

    def test_main_is_read_only_disjoint_four_rank_oracle_scan(self) -> None:
        source = inspect.getsource(cohort.main)
        self.assertIn("distributed.world_size != 4", source)
        self.assertIn("distributed.rank :: distributed.world_size", source)
        self.assertIn("spt.build_oracle_plan", source)
        self.assertIn("torch.inference_mode", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)

    def test_generate_ceiling_cannot_be_relaxed(self) -> None:
        args = cohort.build_parser().parse_args(
            self._argv()
            + ["--max-postbudget-generate-fraction-per-phase", "0.120001"]
        )
        with self.assertRaisesRegex(cohort.TeacherCohortAuditError, "0.12"):
            cohort.validate_cli(args)


try:
    import torch
except ImportError:  # pragma: no cover - local contract environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local contract environment")
class TeacherCohortTensorTests(unittest.TestCase):
    def test_real_hardened_oracle_report_exposes_pairwise_metrics(self) -> None:
        torch.manual_seed(7)
        source = torch.randn(1, 21, 2, 2, 64)
        target = source.clone()
        config = spt.PhaseTransportConfig(
            latent_channels=64,
            max_generate_fraction_per_phase=0.12,
            teacher_require_cycle=True,
        )
        oracle = spt.build_oracle_plan(
            source, target, config, feature_channels=64
        )
        row = {"iid": "identity", "inputs": {"instruction": "do nothing"}}
        report = cohort.row_report(
            row_index=0,
            row=row,
            source=source,
            target=target,
            oracle=oracle,
            thresholds=cohort.SelectorThresholds(),
        )
        self.assertEqual(report["mse"]["source_to_target"], 0.0)
        self.assertEqual(report["mse"]["proxy_to_target"], 0.0)
        self.assertEqual(report["mse"]["source_to_proxy"], 0.0)
        self.assertLessEqual(
            report["observed_max_postbudget_generate_fraction_per_phase"],
            0.120001,
        )
        self.assertAlmostEqual(
            sum(report["postbudget_gate_fraction"].values()), 1.0, places=6
        )
        self.assertFalse(report["selection"]["selected"])


if __name__ == "__main__":
    unittest.main()
