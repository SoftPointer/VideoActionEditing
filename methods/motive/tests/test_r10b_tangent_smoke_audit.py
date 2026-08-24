from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from motive import r10b_lucy_tangent_extract as lucy
from motive import r10b_bernini_tangent_extract as bernini
from motive import r10b_tangent_smoke_audit as audit
from motive.r10b_tangent_smoke_audit import (
    R10BTangentSmokeAuditError,
    _atomic_write_json,
    _cross_seed_scalar_diagnostics,
    audit_published_smoke,
)


class R10BTangentSmokeAuditTests(unittest.TestCase):
    @staticmethod
    def _write_fake_smoke(
        root: Path,
        *,
        schema: str = lucy.EXTRACT_SCHEMA,
    ) -> None:
        seeds = (101, 202)
        roles = tuple(lucy.ROLE_NAMES)
        rows = [
            {
                "iid": "sit-a",
                "family": "sit_down",
                "component_id": "component-a",
                "cells": {
                    "tc": {"loss": {"combined_loss": 2.0}},
                    "t0": {"loss": {"combined_loss": 2.1}},
                },
            },
            {
                "iid": "lie-b",
                "family": "lie_down",
                "component_id": "component-b",
                "cells": {
                    "tc": {"loss": {"combined_loss": 3.3}},
                    "t0": {"loss": {"combined_loss": 3.0}},
                },
            },
        ]
        summary = {
            "schema_version": schema,
            "measurement": {
                "projection_seeds": list(seeds),
                "projection_dimension_per_role": 3,
            },
        }
        (root / "summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        (root / "rows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        base_cells = {
            "tc": np.asarray([[3.0, 2.0, 0.0], [5.0, 1.0, 0.0]]),
            "sc": np.asarray([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
            "t0": np.asarray([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]),
            "s0": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        }
        arrays: dict[str, np.ndarray] = {
            "ids": np.asarray(["sit-a", "lie-b"]),
            "metadata_json": np.asarray("{}"),
        }
        for role_index, role in enumerate(roles, start=1):
            for seed_index, seed in enumerate(seeds, start=1):
                scale = float(role_index * seed_index)
                for cell, values in base_cells.items():
                    arrays[f"raw__{role}__{cell}__p{seed}"] = (
                        values * scale
                    ).astype(np.float32)
        np.savez_compressed(root / "features.npz", **arrays)

    @staticmethod
    def _valid_result(root: Path) -> dict[str, object]:
        return {
            "status": "VALID",
            "output_dir": str(root),
            "artifact_digest": "a" * 64,
            "rows": 2,
            "feature_arrays": 1,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        }

    def test_audit_recomputes_raw_factorial_diagnostics_and_keeps_gates_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fake_smoke(root)
            validation = self._valid_result(root)
            with mock.patch.object(
                lucy,
                "validate_published_extract",
                return_value=validation,
            ) as validator:
                result = audit_published_smoke(root)

            self.assertEqual(validator.call_count, 2)
            first = result["rows"][0]["by_role"]["self_motion"][
                "by_projection_seed"
            ]["101"]
            self.assertAlmostEqual(
                first["quotient_norms"]["paired"],
                math.sqrt(5.0),
            )
            self.assertAlmostEqual(
                first["quotient_norms"]["noop"],
                math.sqrt(5.0),
            )
            self.assertAlmostEqual(
                first["quotient_norms"]["did"],
                math.sqrt(2.0),
            )
            self.assertAlmostEqual(first["paired_vs_noop_cosine"], 0.8)
            self.assertAlmostEqual(
                first["quotient_norm_ratios"][
                    "did_over_paired_plus_noop"
                ],
                math.sqrt(2.0) / (2.0 * math.sqrt(5.0)),
            )
            sensitivity = result["rows"][0][
                "instruction_target_loss_sensitivity"
            ]
            self.assertAlmostEqual(
                sensitivity["relative_to_noop"],
                (2.0 - 2.1) / 2.1,
            )
            self.assertFalse(result["representation_gate_passed"])
            self.assertFalse(result["renderer_probe_authorized"])
            self.assertFalse(result["editor_training_authorized"])
            self.assertFalse(
                result["limitations"][
                    "two_row_smoke_is_retrieval_evidence"
                ]
            )
            policy = result["cross_projection_seed_diagnostics"]["policy"]
            self.assertFalse(
                policy["projected_coordinates_comparable_across_seeds"]
            )
            self.assertFalse(policy["cross_seed_vector_dot_products_computed"])
            self.assertFalse(policy["cross_seed_vector_cosines_computed"])

    def test_validator_runs_before_features_are_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fake_smoke(root)
            with mock.patch.object(
                lucy,
                "validate_published_extract",
                side_effect=RuntimeError("closure rejected"),
            ), mock.patch.object(
                audit,
                "_load_raw_features",
            ) as load_features:
                with self.assertRaisesRegex(RuntimeError, "closure rejected"):
                    audit_published_smoke(root)
            load_features.assert_not_called()

    def test_bernini_schema_dispatches_to_bernini_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fake_smoke(
                root,
                schema=bernini.EXTRACT_SCHEMA,
            )
            validation = self._valid_result(root)
            with mock.patch.object(
                bernini,
                "validate_published_extract",
                return_value=validation,
            ) as validator:
                result = audit_published_smoke(root)
            self.assertEqual(result["backend"], "bernini_r_1_3b")
            self.assertEqual(validator.call_count, 2)

    def test_changed_validation_result_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fake_smoke(root)
            before = self._valid_result(root)
            after = {**before, "artifact_digest": "b" * 64}
            with mock.patch.object(
                lucy,
                "validate_published_extract",
                side_effect=[before, after],
            ):
                with self.assertRaisesRegex(
                    R10BTangentSmokeAuditError,
                    "changed during read-only audit",
                ):
                    audit_published_smoke(root)

    def test_atomic_json_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            value = {
                "representation_gate_passed": False,
                "editor_training_authorized": False,
            }
            _atomic_write_json(path, value)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                value,
            )
            with self.assertRaises(FileExistsError):
                _atomic_write_json(path, value)


class R10BCrossSeedScalarOnlyTests(unittest.TestCase):
    def test_rank_diagnostics_accept_scalars_and_report_no_vector_metric(
        self,
    ) -> None:
        metrics = tuple(audit._SCALAR_METRICS)
        table = {
            "self_motion": {
                11: {
                    metric: [1.0, 2.0, 3.0]
                    for metric in metrics
                },
                29: {
                    metric: [10.0, 20.0, 30.0]
                    for metric in metrics
                },
            }
        }
        result = _cross_seed_scalar_diagnostics(
            table,
            row_ids=["a", "b", "c"],
            seeds=[11, 29],
        )
        pair = result["self_motion"][
            "pairwise_seed_row_rank_spearman"
        ][0]
        self.assertEqual(
            pair["metric_correlations"]["did_norm"],
            1.0,
        )
        self.assertNotIn("cosine_between_seed_vectors", result)


if __name__ == "__main__":
    unittest.main()
