from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motive.r7_coherent_actor import R7_COHERENT_ACTOR_SCHEMA
from motive.r7_p1_diagnostic import (
    R7_P1_DIAGNOSTIC_DONE_SCHEMA,
    R7_P1_DIAGNOSTIC_ROW_SCHEMA,
    R7_P1_DIAGNOSTIC_SCHEMA,
    R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
)
from motive.r7_preflight_extract import (
    _atomic_json,
    _atomic_jsonl,
    _file_digest,
    _object_digest,
)
from motive.r8_p1_mechanism_diagnostic import (
    R8_P1_MECHANISM_ROW_SCHEMA,
    load_p1_commit,
    run_mechanism_diagnostic,
)
from motive.r8_stable_motion import (
    R8_SMOOTHED_CENTER_DEFAULT_ABSOLUTE_POSITION_TOLERANCE,
    R8_SMOOTHED_CENTER_DEFAULT_DELTA,
    R8_STABLE_MOTION_PHASE_STEPS,
)


PHASES = 32


def _not_ready_side() -> dict[str, object]:
    return {
        "diagnostic_ready": False,
        "selector_ready": False,
        "event_ready": False,
        "failure_stage": "coherent_actor_selector",
        "failure_reason": "no_coherent_component",
        "failure_detail": "synthetic not-ready record",
        "selector": {
            "schema_version": R7_COHERENT_ACTOR_SCHEMA,
            "actor_track_indices": [],
            "coordinate_space": "normalized-max-side-isotropic",
        },
        "score": 0.0,
        "actor_track_mask": [False] * 16,
        "actor_trajectory": [[0.0, 0.0]] * PHASES,
        "actor_track_trajectories": [],
        "actor_track_phase_mask": [],
        "phase_times": [0.0] * PHASES,
        "selector_phase_energy": [0.0] * PHASES,
        "actor_phase_speed": [],
        "event_transition_energy": [],
        "phase_visibility": [0.0] * PHASES,
        "event_window": None,
    }


def _ready_side(
    *,
    membership: tuple[int, ...] = (2, 7, 11, 19),
    permutation: tuple[int, ...] | None = None,
    aggregate_shift: bool = False,
    transition_failure: bool = False,
) -> dict[str, object]:
    count = len(membership)
    phase = np.arange(PHASES, dtype=np.float64)
    translation = np.stack((0.003 * phase, -0.001 * phase), axis=1)
    trajectories = np.repeat(translation[None], count, axis=0)
    mask = np.ones((count, PHASES), dtype=bool)
    if transition_failure:
        mask[:] = False
        half = count // 2
        mask[:half, ::2] = True
        mask[half:, 1::2] = True
    indices = np.asarray(membership, dtype=np.int64)
    if permutation is not None:
        order = np.asarray(permutation, dtype=np.int64)
        trajectories = trajectories[order]
        mask = mask[order]
        indices = indices[order]
    aggregate = translation.copy()
    if aggregate_shift:
        aggregate[16:, 0] += 0.05
    energy = np.linspace(0.01, 0.20, PHASES, dtype=np.float64)
    return {
        "diagnostic_ready": True,
        "selector_ready": True,
        "event_ready": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_detail": None,
        "selector": {
            "schema_version": R7_COHERENT_ACTOR_SCHEMA,
            "diagnostic_ready": True,
            "actor_track_indices": indices.tolist(),
            "coordinate_space": "normalized-max-side-isotropic",
        },
        "score": 0.1,
        "actor_track_mask": [index in set(membership) for index in range(32)],
        "actor_trajectory": aggregate.tolist(),
        "actor_track_trajectories": trajectories.tolist(),
        "actor_track_phase_mask": mask.tolist(),
        "phase_times": np.linspace(0.0, 1.24, PHASES).tolist(),
        "selector_phase_energy": energy.tolist(),
        "actor_phase_speed": np.linalg.norm(
            np.gradient(aggregate, axis=0), axis=1
        ).tolist(),
        "event_transition_energy": (
            0.5 * (energy[:-1] + energy[1:])
        ).tolist(),
        "phase_visibility": np.mean(mask, axis=0).tolist(),
        "event_window": {
            "transition_start": 0,
            "transition_stop": 31,
            "frame_start": 0,
            "frame_stop": 32,
            "start_time": 0.0,
            "end_time": 1.24,
            "duration": 1.24,
            "normalized_start": 0.0,
            "normalized_end": 1.0,
            "captured_energy_fraction": 1.0,
        },
    }


def _audit(
    *,
    eligible: bool,
    base: dict[str, object],
    perturbed: dict[str, object] | None,
) -> dict[str, object]:
    available = bool(
        eligible
        and base["diagnostic_ready"]
        and isinstance(perturbed, dict)
        and perturbed["diagnostic_ready"]
    )
    if available:
        first = np.asarray(base["actor_trajectory"], dtype=np.float64)
        second = np.asarray(
            perturbed["actor_trajectory"], dtype=np.float64
        )
        trajectory_rmse = float(np.sqrt(np.mean((first - second) ** 2)))
    else:
        trajectory_rmse = 1.0
    return {
        "eligible": eligible,
        "performed": eligible,
        "seed": 123 if eligible else None,
        "seed_derivation": "synthetic" if eligible else None,
        "perturbation": {} if eligible else None,
        "ready_consistent": bool(
            isinstance(perturbed, dict)
            and base["diagnostic_ready"] == perturbed["diagnostic_ready"]
        ),
        "comparison_available": available,
        "metrics": {
            "actor_mask_iou": 1.0 if available else 0.0,
            "event_window_iou": 1.0 if available else 0.0,
            "trajectory_rmse": trajectory_rmse,
            "shared_actor_track_fraction": 1.0 if available else 0.0,
            "per_track_trajectory_rmse": 0.0 if available else 1.0,
            "energy_cosine": 1.0 if available else 0.0,
            "shape_profile_cosine": 1.0 if available else 0.0,
            "event_duration_relative_error": 0.0 if available else 1.0,
        },
        "joint_pass": bool(available and trajectory_rmse <= 0.01),
        "failure_reason": (
            None
            if available and trajectory_rmse <= 0.01
            else "joint_threshold_failed" if available else "base_not_ready"
        ),
        "perturbed": perturbed,
    }


def _rows(
    *,
    rescue: bool = True,
    transition_failure: bool = False,
    permuted_perturbed: bool = False,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index in range(181):
        positive = index < 99
        target_camera = index < 97
        target = _not_ready_side()
        perturbed: dict[str, object] | None = (
            _not_ready_side() if target_camera else None
        )
        if index == 0 and rescue:
            target = _ready_side(transition_failure=transition_failure)
            perturbed = _ready_side(
                aggregate_shift=True,
                transition_failure=transition_failure,
                permutation=(2, 0, 3, 1)
                if permuted_perturbed
                else None,
            )
        eligible = bool(positive and target_camera)
        output.append(
            {
                "schema_version": R7_P1_DIAGNOSTIC_ROW_SCHEMA,
                "input_index": index,
                "iid": f"synthetic-{index:03d}",
                "cache_row_sha256": _object_digest({"index": index}),
                "positive": positive,
                "label_type": "positive" if positive else "negative",
                "negative_type": None if positive else "static",
                "action_signature": "walk" if positive else "negative:static",
                "source_camera_valid": True,
                "target_camera_valid": target_camera,
                "source": _not_ready_side(),
                "target": target,
                "target_audit": _audit(
                    eligible=eligible,
                    base=target,
                    perturbed=perturbed,
                ),
                "formal_status": "INSUFFICIENT",
                "production_decision": False,
                "generation_authorized": False,
            }
        )
    return output


def _contract() -> dict[str, object]:
    audit_config = {
        "trajectory_rmse_threshold": 0.01,
        "energy_cosine_threshold": 0.85,
    }
    gate_config = {"synthetic": True}
    return {
        "schema_version": R7_P1_DIAGNOSTIC_SCHEMA,
        "input_manifest": "/synthetic/input.jsonl",
        "input_manifest_sha256": "1" * 64,
        "cache": {"synthetic": True},
        "seed": 20260727,
        "selector": {"synthetic": True},
        "continuous_event_locator": {"synthetic": True},
        "independent_audit": {
            "domain": "synthetic",
            "implementation_sha256": "2" * 64,
            "config": audit_config,
            "config_sha256": _object_digest(audit_config),
            "base_selection_is_audit_mask": False,
            "missing_comparison_credit": 0,
        },
        "diagnostic_gate": {
            "schema_version": "motive-r7-p1-frozen-development-gate-v1",
            "config": gate_config,
            "config_sha256": _object_digest(gate_config),
            "thresholds_frozen_before_p1_cache_results": True,
            "not_independent_preregistration": True,
            "design_was_driven_by_prior_p0_failure": True,
            "thresholds_may_not_be_adjusted_from_results": True,
        },
        "implementation_sha256": {
            "r7_p1_diagnostic.py": "3" * 64,
            "r7_coherent_actor.py": "4" * 64,
        },
        "development_scope": "synthetic old-181 development fixture",
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
    }


def _write_p1(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _contract()
    available = sum(
        bool(row["target_audit"]["comparison_available"]) for row in rows
    )
    gate = {
        "schema_version": "motive-r7-p1-frozen-development-gate-v1",
        "diagnostic_gate_passed": False,
        "counts": {
            "rows": 181,
            "positive_target_camera_valid": 97,
            "positive_target_audit_joint_pass": sum(
                bool(row["target_audit"]["joint_pass"]) for row in rows
            ),
            "synthetic_available": available,
        },
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
    }
    summary = {
        "schema_version": R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
        "rows": 181,
        "contract": contract,
        "contract_sha256": _object_digest(contract),
        "rows_object_sha256": _object_digest(rows),
        "failure_counts": {},
        "gate": gate,
        "formal_status": "INSUFFICIENT",
        "formal_reason": "synthetic",
        "production_decision": False,
        "generation_authorized": False,
    }
    _atomic_jsonl(directory / "rows.jsonl", rows)
    _atomic_json(directory / "summary.json", summary)
    done = {
        "schema_version": R7_P1_DIAGNOSTIC_DONE_SCHEMA,
        "committed": True,
        "rows": 181,
        "rows_sha256": _file_digest(directory / "rows.jsonl"),
        "summary_sha256": _file_digest(directory / "summary.json"),
        "contract_sha256": _object_digest(contract),
        "diagnostic_gate_passed": False,
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
    }
    _atomic_json(directory / "done.json", done)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _rewrite_summary_commit(
    directory: Path,
    summary: dict[str, object],
) -> None:
    contract = summary["contract"]
    assert isinstance(contract, dict)
    summary["contract_sha256"] = _object_digest(contract)
    _atomic_json(directory / "summary.json", summary)
    done = _read_json(directory / "done.json")
    done["summary_sha256"] = _file_digest(directory / "summary.json")
    done["contract_sha256"] = summary["contract_sha256"]
    _atomic_json(directory / "done.json", done)


class R8MechanismDiagnosticTests(unittest.TestCase):
    def test_rescue_full_denominator_and_permutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "p1"
            output = root / "r8"
            _write_p1(
                source,
                _rows(rescue=True, permuted_perturbed=True),
            )
            done = run_mechanism_diagnostic(
                input_directory=source,
                output_directory=output,
            )
            self.assertEqual(done["rows"], 181)
            summary = _read_json(output / "summary.json")
            self.assertEqual(
                summary["counts"][
                    "audit_denominator_all_positive_target_camera_valid"
                ],
                97,
            )
            self.assertEqual(
                summary["counts"]["upstream_comparison_available"], 1
            )
            self.assertEqual(
                summary["counts"]["r8_comparison_available"], 1
            )
            self.assertEqual(summary["counts"]["reference_rescue"], 1)
            self.assertFalse(summary["creates_new_gate"])
            row = _read_rows(output / "rows.jsonl")[0]
            self.assertEqual(
                row["schema_version"], R8_P1_MECHANISM_ROW_SCHEMA
            )
            self.assertGreater(
                row["comparison"]["old_aggregate_trajectory_rmse"],
                0.01,
            )
            self.assertAlmostEqual(
                row["comparison"]["r8_global_trajectory_rmse"], 0.0
            )
            self.assertEqual(
                row["base"]["representation"][
                    "component_track_indices"
                ],
                row["perturbed"]["representation"][
                    "component_track_indices"
                ],
            )
            center_fields = {
                "center_certificate_kind",
                "center_position_error_upper_bound",
                "center_global_curvature_lower_bound",
                "center_gradient_upper_bound",
            }
            for side in ("base", "perturbed"):
                representation = row[side]["representation"]
                self.assertTrue(
                    center_fields.issubset(representation),
                    f"{side} omitted center-certificate evidence",
                )
                for field in center_fields:
                    self.assertEqual(
                        len(representation[field]),
                        R8_STABLE_MOTION_PHASE_STEPS,
                    )
                representation_summary = representation["summary"]
                self.assertEqual(
                    representation_summary["smoothed_center_delta"],
                    R8_SMOOTHED_CENTER_DEFAULT_DELTA,
                )
                self.assertEqual(
                    representation_summary[
                        "smoothed_center_absolute_position_tolerance"
                    ],
                    (
                        R8_SMOOTHED_CENTER_DEFAULT_ABSOLUTE_POSITION_TOLERANCE
                    ),
                )
                self.assertEqual(
                    representation_summary["center_storage_dtype"],
                    "float64",
                )
                self.assertEqual(
                    representation_summary[
                        "max_center_position_error_upper_bound"
                    ],
                    max(
                        representation[
                            "center_position_error_upper_bound"
                        ]
                    ),
                )
            comparison = row["comparison"]
            for side in ("base", "perturbed"):
                key = (
                    f"{side}_max_center_position_error_upper_bound"
                )
                self.assertEqual(
                    comparison[key],
                    row[side]["representation"]["summary"][
                        "max_center_position_error_upper_bound"
                    ],
                )
                statistics = summary["descriptive_only"][key]
                self.assertEqual(statistics["count"], 1)
                self.assertEqual(statistics["maximum"], comparison[key])
            serialized_evidence = summary["contract"]["stable_motion"][
                "serialized_center_evidence"
            ]
            self.assertEqual(
                set(serialized_evidence["per_phase_fields"]),
                center_fields,
            )
            self.assertEqual(
                serialized_evidence["center_storage_dtype"], "float64"
            )

    def test_not_ready_and_transition_support_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "p1"
            output = root / "r8"
            _write_p1(source, _rows(transition_failure=True))
            run_mechanism_diagnostic(
                input_directory=source,
                output_directory=output,
            )
            rows = _read_rows(output / "rows.jsonl")
            self.assertTrue(rows[0]["base"]["attempted"])
            self.assertFalse(rows[0]["base"]["r8_diagnostic_ready"])
            self.assertEqual(
                rows[0]["base"]["failure_reason"],
                "insufficient_transition_support",
            )
            self.assertFalse(
                rows[0]["comparison"]["r8_comparison_available"]
            )
            self.assertFalse(rows[1]["base"]["attempted"])
            self.assertEqual(
                rows[1]["base"]["failure_reason"], "upstream_not_ready"
            )
            summary = _read_json(output / "summary.json")
            self.assertEqual(
                summary["r8_failure_counts"][
                    "insufficient_transition_support"
                ],
                2,
            )

    def test_input_tamper_and_nonunique_order_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "p1"
            _write_p1(source, _rows())
            rows = _read_rows(source / "rows.jsonl")
            rows[0]["iid"] = "tampered"
            _atomic_jsonl(source / "rows.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "binding differs"):
                load_p1_commit(source)

            source2 = root / "p1-duplicate"
            rows = _rows()
            rows[1]["input_index"] = 0
            _write_p1(source2, rows)
            with self.assertRaisesRegex(ValueError, "order/index"):
                load_p1_commit(source2)

            source3 = root / "p1-unsafe"
            rows = _rows()
            rows[0]["generation_authorized"] = True
            _write_p1(source3, rows)
            with self.assertRaisesRegex(
                ValueError, "generation_authorized is unsafe"
            ):
                load_p1_commit(source3)

    def test_input_safety_fields_require_real_false_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            row_source = root / "p1-row-zero"
            rows = _rows()
            rows[0]["generation_authorized"] = 0
            _write_p1(row_source, rows)
            with self.assertRaisesRegex(
                ValueError, "generation_authorized is unsafe"
            ):
                load_p1_commit(row_source)

            summary_source = root / "p1-summary-zero"
            _write_p1(summary_source, _rows())
            summary = _read_json(summary_source / "summary.json")
            summary["production_decision"] = 0
            _rewrite_summary_commit(summary_source, summary)
            with self.assertRaisesRegex(
                ValueError, "production_decision is unsafe"
            ):
                load_p1_commit(summary_source)

            contract_source = root / "p1-contract-zero"
            _write_p1(contract_source, _rows())
            summary = _read_json(contract_source / "summary.json")
            contract = summary["contract"]
            assert isinstance(contract, dict)
            contract["generation_authorized"] = 0
            _rewrite_summary_commit(contract_source, summary)
            with self.assertRaisesRegex(
                ValueError, "generation_authorized is unsafe"
            ):
                load_p1_commit(contract_source)

            done_source = root / "p1-done-zero"
            _write_p1(done_source, _rows())
            done = _read_json(done_source / "done.json")
            done["production_decision"] = 0
            _atomic_json(done_source / "done.json", done)
            with self.assertRaisesRegex(
                ValueError, "production_decision is unsafe"
            ):
                load_p1_commit(done_source)

    def test_gate_decisions_require_matching_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, invalid in enumerate((None, 0, "", "false")):
                source = root / f"p1-gate-{index}"
                _write_p1(source, _rows())
                summary = _read_json(source / "summary.json")
                gate = summary["gate"]
                assert isinstance(gate, dict)
                if invalid is None:
                    del gate["diagnostic_gate_passed"]
                else:
                    gate["diagnostic_gate_passed"] = invalid
                _rewrite_summary_commit(source, summary)
                with self.subTest(gate_decision=invalid):
                    with self.assertRaisesRegex(
                        ValueError, "gate binding differs"
                    ):
                        load_p1_commit(source)

            source = root / "p1-done-nonbool"
            _write_p1(source, _rows())
            done = _read_json(source / "done.json")
            done["diagnostic_gate_passed"] = 0
            _atomic_json(source / "done.json", done)
            with self.assertRaisesRegex(
                ValueError, "gate binding differs"
            ):
                load_p1_commit(source)

            source = root / "p1-done-mismatch"
            _write_p1(source, _rows())
            done = _read_json(source / "done.json")
            done["diagnostic_gate_passed"] = True
            _atomic_json(source / "done.json", done)
            with self.assertRaisesRegex(
                ValueError, "gate binding differs"
            ):
                load_p1_commit(source)

    def test_overwrite_resume_and_output_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "p1"
            output = root / "r8"
            _write_p1(source, _rows())
            first = run_mechanism_diagnostic(
                input_directory=source,
                output_directory=output,
            )
            resumed = run_mechanism_diagnostic(
                input_directory=source,
                output_directory=output,
                resume=True,
            )
            self.assertEqual(first, resumed)
            with self.assertRaises(FileExistsError):
                run_mechanism_diagnostic(
                    input_directory=source,
                    output_directory=output,
                )
            rows = _read_rows(output / "rows.jsonl")
            del rows[0]["base"]["representation"][
                "center_gradient_upper_bound"
            ]
            _atomic_jsonl(output / "rows.jsonl", rows)
            with self.assertRaisesRegex(
                ValueError, "deterministic recomputation"
            ):
                run_mechanism_diagnostic(
                    input_directory=source,
                    output_directory=output,
                    resume=True,
                )

            output2 = root / "r8-summary"
            run_mechanism_diagnostic(
                input_directory=source,
                output_directory=output2,
            )
            summary = _read_json(output2 / "summary.json")
            summary["descriptive_only"][
                "base_max_center_position_error_upper_bound"
            ]["maximum"] = 0.5
            _atomic_json(output2 / "summary.json", summary)
            with self.assertRaisesRegex(
                ValueError, "summary differs"
            ):
                run_mechanism_diagnostic(
                    input_directory=source,
                    output_directory=output2,
                    resume=True,
                )

            output3 = root / "r8-done"
            run_mechanism_diagnostic(
                input_directory=source,
                output_directory=output3,
            )
            done = _read_json(output3 / "done.json")
            done["generation_authorized"] = 0
            _atomic_json(output3 / "done.json", done)
            with self.assertRaisesRegex(
                ValueError, "done fields/hash binding differ"
            ):
                run_mechanism_diagnostic(
                    input_directory=source,
                    output_directory=output3,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
