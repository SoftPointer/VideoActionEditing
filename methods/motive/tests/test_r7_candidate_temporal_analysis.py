from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import numpy as np

from motive import r7_artifact_permissions as permissions
from motive import r7_candidate_temporal_analysis as analysis
from motive import r7_candidate_temporal_screen as screen


def _feature_map(vector: tuple[float, float]) -> dict[str, np.ndarray]:
    return {
        modality: np.asarray(vector, dtype=np.float64)
        for modality in screen.MODALITIES
    }


def _example(
    iid: str,
    *,
    label: str,
    family: str,
    split: str,
    component: str,
    vector: tuple[float, float],
    overrides: dict[str, tuple[float, float]] | None = None,
    sampling_weight: float = 1.0,
) -> screen._Example:
    features = _feature_map(vector)
    for modality, value in (overrides or {}).items():
        features[modality] = np.asarray(value, dtype=np.float64)
    return screen._Example(
        iid=iid,
        label_class=label,
        family=family,
        split=split,
        component_id=component,
        fresh=True,
        sampling_weight=sampling_weight,
        features=features,
        motion_energy=1.0 if label == "positive" else 0.1,
    )


def _screen_commit(
    root: Path,
    *,
    dino_confounded: bool = True,
) -> tuple[Path, str]:
    wave = (1.0, 0.0)
    jump = (0.0, 1.0)
    examples = [
        _example(
            f"wave-train-{index}",
            label="positive",
            family="wave",
            split="train",
            component=f"wave-train-component-{index}",
            vector=wave,
        )
        for index in range(5)
    ]
    examples.extend(
        [
            _example(
                f"jump-train-{index}",
                label="positive",
                family="jump",
                split="train",
                component=f"jump-train-component-{index}",
                vector=jump,
            )
            for index in range(5)
        ]
    )
    for split in ("validation", "test"):
        examples.extend(
            [
                _example(
                    f"wave-{split}",
                    label="positive",
                    family="wave",
                    split=split,
                    component=f"wave-{split}-component",
                    vector=wave,
                    overrides={
                        screen.TARGET_ENDPOINT: jump,
                        screen.ORDERLESS_TEMPORAL: jump,
                        screen.CAMERA_NUISANCE: jump,
                        screen.SHUFFLED_QUERY: jump,
                        screen.REVERSED_QUERY: jump,
                    },
                ),
                _example(
                    f"jump-{split}",
                    label="positive",
                    family="jump",
                    split=split,
                    component=f"jump-{split}-component",
                    vector=jump,
                    overrides={
                        screen.TARGET_ENDPOINT: wave,
                        screen.ORDERLESS_TEMPORAL: wave,
                        screen.CAMERA_NUISANCE: wave,
                        screen.SHUFFLED_QUERY: wave,
                        screen.REVERSED_QUERY: wave,
                    },
                ),
                _example(
                    f"negative-{split}",
                    label="negative",
                    family="no_action",
                    split=split,
                    component=f"negative-{split}-component",
                    vector=(-1.0, 0.0),
                    sampling_weight=9.25,
                ),
            ]
        )
    rows, retrieval, binary, diagnostics = screen._evaluate(
        examples,
        eligible_families={"wave", "jump"},
    )
    contract = {
        "schema_version": screen.SCREEN_SCHEMA,
        "retrieval": {
            "split_bias": {
                "relative_motion_vs_dino_diagnostic_is_split_confounded":
                    dino_confounded,
            }
        },
        "representation": {"learned_parameters": False},
        "semantics": {
            "labels_are_pseudo": True,
            "split_is_provisional_diagnostic_only": True,
            "no_gradient": True,
            "no_optimization": True,
            **screen._safety_flags(),
        },
    }
    row_bytes = screen._jsonl_bytes(rows)
    summary = {
        "schema_version": screen.SCREEN_SCHEMA,
        "status": "complete",
        "contract": contract,
        "contract_sha256": screen._object_digest(contract),
        "retrieval": retrieval,
        "positive_vs_sampled_negative": {
            "protocol": screen.BINARY_PROTOCOL,
            "metrics": binary,
        },
        "leakage_control": diagnostics["leakage_control"],
        "decision": {
            "formal_status": "INSUFFICIENT",
            "diagnostic_completed": True,
            **screen._safety_flags(),
        },
        "formal_status": "INSUFFICIENT",
        **screen._safety_flags(),
        "output": {
            "rows_name": screen.ROWS_NAME,
            "rows": len(rows),
            "rows_sha256": hashlib.sha256(row_bytes).hexdigest(),
            "row_order": "ascending_iid",
            "row_encoding": "canonical_json_utf8_lf",
        },
    }
    summary_bytes = screen._pretty_json_bytes(summary)
    payload_files = {
        screen.ROWS_NAME: {
            "sha256": hashlib.sha256(row_bytes).hexdigest(),
            "bytes": len(row_bytes),
            "mode_octal": "0444",
        },
        screen.SUMMARY_NAME: {
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "bytes": len(summary_bytes),
            "mode_octal": "0444",
        },
    }
    done = screen._done_payload(
        rows=len(rows),
        contract_sha256=summary["contract_sha256"],
        payload_files=payload_files,
    )
    directory = root / "screen"
    directory.mkdir()
    (directory / screen.ROWS_NAME).write_bytes(row_bytes)
    (directory / screen.SUMMARY_NAME).write_bytes(summary_bytes)
    (directory / screen.DONE_NAME).write_bytes(
        screen._pretty_json_bytes(done)
    )
    permissions.seal_staging_tree(directory)
    done_sha = analysis._file_digest(directory / screen.DONE_NAME)
    screen._validate_candidate_temporal_screen_envelope(directory)
    return directory, done_sha


class BootstrapPrimitiveTests(unittest.TestCase):
    def _row(
        self,
        iid: str,
        component: str,
        family: str,
        *,
        hit: bool,
    ) -> dict[str, object]:
        modalities = {}
        for modality in screen.MODALITIES:
            modalities[modality] = {
                "valid_for_retrieval": True,
                "correct_at_1": hit,
                "correct_at_5": hit,
            }
        return {
            "iid": iid,
            "component_id": component,
            "family": family,
            "modalities": modalities,
        }

    def test_component_rows_share_multiplicity_and_pairing_is_exact(
        self,
    ) -> None:
        rows = [
            self._row("a-1", "component-a", "wave", hit=True),
            self._row("a-2", "component-a", "wave", hit=False),
            self._row("b-1", "component-b", "jump", hit=True),
        ]
        plan = analysis._bootstrap_plan(
            rows,
            scope="overall",
            repetitions=64,
            seed=19,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        np.testing.assert_array_equal(
            plan.row_weights[:, 0],
            plan.row_weights[:, 1],
        )
        left = analysis._bootstrap_metric_series(
            rows,
            modality=screen.TARGET_TEMPORAL,
            plan=plan,
        )
        right = analysis._bootstrap_metric_series(
            rows,
            modality=screen.TARGET_ENDPOINT,
            plan=plan,
        )
        for metric in analysis.METRICS:
            interval = analysis._difference_interval(
                left[metric],
                right[metric],
                point=0.0,
            )
            self.assertEqual(interval["lower"], 0.0)
            self.assertEqual(interval["upper"], 0.0)
            self.assertEqual(interval["bootstrap_mean"], 0.0)

    def test_invalid_query_counts_as_miss_but_coverage_is_retained(
        self,
    ) -> None:
        rows = [
            self._row("valid", "component-a", "wave", hit=True),
            self._row("invalid", "component-b", "wave", hit=False),
        ]
        rows[1]["modalities"][screen.TARGET_TEMPORAL][
            "valid_for_retrieval"
        ] = False
        rows[1]["modalities"][screen.TARGET_TEMPORAL][
            "correct_at_1"
        ] = None
        rows[1]["modalities"][screen.TARGET_TEMPORAL][
            "correct_at_5"
        ] = None
        metric = analysis._point_metrics(
            rows,
            modality=screen.TARGET_TEMPORAL,
        )
        self.assertEqual(metric["valid_fraction"], 0.5)
        self.assertEqual(metric["metrics"]["micro_r_at_1"], 0.5)
        self.assertEqual(
            metric["metrics"]["macro_family_r_at_1"],
            0.5,
        )

    def test_validation_test_direction_reports_disagreement(self) -> None:
        def scope(point: float) -> dict[str, object]:
            return {
                "paired_component_bootstrap": {
                    "intervals": {
                        metric: {
                            "point": point,
                            "lower": point,
                            "upper": point,
                        }
                        for metric in analysis.METRICS
                    }
                }
            }

        result = analysis._val_test_direction(
            scope(0.25),
            scope(-0.25),
        )
        self.assertEqual(
            result["metrics"]["macro_family_r_at_1"][
                "consistency"
            ],
            "DISAGREE",
        )
        self.assertEqual(
            result["overall"],
            "MIXED_DISAGREE_OR_ZERO",
        )

    def test_component_crossing_validation_and_test_fails_closed(
        self,
    ) -> None:
        rows = [
            {
                "component_id": "shared",
                "split": "validation",
                "label_class": "positive",
                "eligible_positive_query": True,
                "family": "wave",
            },
            {
                "component_id": "shared",
                "split": "test",
                "label_class": "positive",
                "eligible_positive_query": True,
                "family": "wave",
            },
        ]
        with self.assertRaisesRegex(
            analysis.CandidateTemporalAnalysisError,
            "cross validation/test",
        ):
            analysis._validate_component_topology(rows)

    def test_family_stratified_draws_preserve_equal_family_macro(
        self,
    ) -> None:
        rows = [
            self._row(
                "rare",
                "rare-component",
                "rare-family",
                hit=True,
            )
        ]
        rows.extend(
            self._row(
                f"common-{index}",
                f"common-component-{index}",
                "common-family",
                hit=False,
            )
            for index in range(9)
        )
        plan = analysis._bootstrap_plan(
            rows,
            scope="overall",
            repetitions=4096,
            seed=41,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            dict(plan.family_component_counts),
            {"common-family": 9, "rare-family": 1},
        )
        series = analysis._bootstrap_metric_series(
            rows,
            modality=screen.TARGET_TEMPORAL,
            plan=plan,
        )
        np.testing.assert_array_equal(
            series["macro_family_r_at_1"],
            np.full(4096, 0.5, dtype=np.float64),
        )

    def test_component_crossing_action_families_fails_closed(
        self,
    ) -> None:
        rows = [
            {
                "component_id": "shared",
                "split": "validation",
                "label_class": "positive",
                "eligible_positive_query": True,
                "family": family,
            }
            for family in ("wave", "jump")
        ]
        with self.assertRaisesRegex(
            analysis.CandidateTemporalAnalysisError,
            "cross action families",
        ):
            analysis._validate_component_topology(rows)


class CandidateTemporalAnalysisCommitTests(unittest.TestCase):
    def test_analysis_is_deterministic_sealed_and_safety_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            first = root / "analysis-first"
            second = root / "analysis-second"
            result = analysis.run_candidate_temporal_analysis(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                output_dir=first,
                bootstrap_repetitions=64,
                seed=31,
            )
            analysis.run_candidate_temporal_analysis(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                output_dir=second,
                bootstrap_repetitions=64,
                seed=31,
            )
            for name in analysis.OUTPUT_NAMES:
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )
                self.assertEqual(
                    stat.S_IMODE((first / name).stat().st_mode),
                    0o444,
                )
            self.assertEqual(
                stat.S_IMODE(first.stat().st_mode),
                0o555,
            )
            summary = result["summary"]
            self.assertEqual(summary["formal_status"], "INSUFFICIENT")
            self.assertTrue(
                summary["evidence_limitations"][
                    "fixed_train_bank_not_resampled"
                ]
            )
            for field in analysis.SAFETY_FIELDS:
                self.assertIs(summary[field], False)
                self.assertIs(summary["decision"][field], False)
            comparisons = {
                row["name"]: row for row in result["comparisons"]
            }
            temporal = comparisons["target_temporal_vs_endpoint"]
            interval = temporal["scopes"]["overall"][
                "paired_component_bootstrap"
            ]["intervals"]["macro_family_r_at_1"]
            self.assertEqual(interval["point"], 1.0)
            self.assertEqual(interval["lower"], 1.0)
            self.assertEqual(interval["upper"], 1.0)
            self.assertEqual(
                temporal["validation_test_direction"]["metrics"][
                    "macro_family_r_at_1"
                ]["consistency"],
                "SAME_POSITIVE",
            )
            self.assertTrue(
                comparisons["target_temporal_vs_pooled_dino"][
                    "dino_split_confounded"
                ]
            )
            self.assertFalse(
                comparisons["target_temporal_vs_endpoint"][
                    "dino_split_confounded"
                ]
            )
            analysis_done_sha = analysis._file_digest(
                first / analysis.DONE_NAME
            )
            replay = analysis.validate_candidate_temporal_analysis(
                first,
                expected_done_sha256=analysis_done_sha,
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                bootstrap_repetitions=64,
                seed=31,
            )
            self.assertTrue(replay["input_screen_verified"])

    def test_create_only_and_strict_resume(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            output = root / "analysis"
            analysis.run_candidate_temporal_analysis(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                output_dir=output,
                bootstrap_repetitions=16,
                seed=7,
            )
            original = {
                name: (output / name).read_bytes()
                for name in analysis.OUTPUT_NAMES
            }
            with self.assertRaises(FileExistsError):
                analysis.run_candidate_temporal_analysis(
                    screen_dir=screen_dir,
                    expected_screen_done_sha256=screen_done_sha,
                    output_dir=output,
                    bootstrap_repetitions=16,
                    seed=7,
                )
            analysis.run_candidate_temporal_analysis(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                output_dir=output,
                bootstrap_repetitions=16,
                seed=7,
                resume=True,
            )
            self.assertEqual(
                original,
                {
                    name: (output / name).read_bytes()
                    for name in analysis.OUTPUT_NAMES
                },
            )

    def test_resume_rejects_self_consistent_location_with_changed_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            output = root / "analysis"
            analysis.run_candidate_temporal_analysis(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                output_dir=output,
                bootstrap_repetitions=8,
                seed=3,
            )
            permissions.make_staging_tree_removable(output)
            path = output / analysis.COMPARISONS_NAME
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["interpretation"] = "tampered"
            path.write_bytes(analysis._jsonl_bytes(rows))
            permissions.seal_staging_tree(output)
            with self.assertRaisesRegex(
                analysis.CandidateTemporalAnalysisError,
                "resume payload differs",
            ):
                analysis.run_candidate_temporal_analysis(
                    screen_dir=screen_dir,
                    expected_screen_done_sha256=screen_done_sha,
                    output_dir=output,
                    bootstrap_repetitions=8,
                    seed=3,
                    resume=True,
                )

    def test_external_screen_done_anchor_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, _screen_done_sha = _screen_commit(root)
            with self.assertRaisesRegex(
                analysis.CandidateTemporalAnalysisError,
                "external done SHA differs",
            ):
                analysis.run_candidate_temporal_analysis(
                    screen_dir=screen_dir,
                    expected_screen_done_sha256="0" * 64,
                    output_dir=root / "analysis",
                    bootstrap_repetitions=8,
                    seed=3,
                )

    def test_atomic_claim_never_replaces_an_existing_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            output = root / "analysis"
            output.mkdir()
            sentinel = output / "owned-by-another-writer"
            sentinel.write_text("keep\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                analysis.run_candidate_temporal_analysis(
                    screen_dir=screen_dir,
                    expected_screen_done_sha256=screen_done_sha,
                    output_dir=output,
                    bootstrap_repetitions=8,
                    seed=3,
                )
            self.assertEqual(
                {entry.name for entry in output.iterdir()},
                {sentinel.name},
            )
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "keep\n",
            )

    def test_path_replacement_after_claim_fails_identity_check(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            output = root / "analysis"
            displaced = root / "displaced-claim"
            original = analysis._write_file_at
            replaced = False

            def replace_then_write(
                directory_fd: int,
                name: str,
                payload: bytes,
            ) -> tuple[int, tuple[int, ...]]:
                nonlocal replaced
                if not replaced:
                    replaced = True
                    os.rename(output, displaced)
                    output.mkdir(mode=0o700)
                return original(directory_fd, name, payload)

            with mock.patch.object(
                analysis,
                "_write_file_at",
                side_effect=replace_then_write,
            ):
                with self.assertRaisesRegex(
                    analysis.CandidateTemporalAnalysisError,
                    "path identity changed",
                ):
                    analysis.run_candidate_temporal_analysis(
                        screen_dir=screen_dir,
                        expected_screen_done_sha256=screen_done_sha,
                        output_dir=output,
                        bootstrap_repetitions=8,
                        seed=3,
                    )
            self.assertTrue(replaced)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode),
                0o700,
            )

    def test_payload_replacement_after_write_fails_identity_check(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            output = root / "analysis"
            original = analysis._write_file_at
            replaced = False

            def replace_payload_after_write(
                directory_fd: int,
                name: str,
                payload: bytes,
            ) -> tuple[int, tuple[int, ...]]:
                nonlocal replaced
                bound_payload = original(directory_fd, name, payload)
                if not replaced:
                    replaced = True
                    os.unlink(name, dir_fd=directory_fd)
                    replacement = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | os.O_NOFOLLOW,
                        0o444,
                        dir_fd=directory_fd,
                    )
                    try:
                        os.write(replacement, payload)
                        os.fsync(replacement)
                    finally:
                        os.close(replacement)
                return bound_payload

            with mock.patch.object(
                analysis,
                "_write_file_at",
                side_effect=replace_payload_after_write,
            ):
                with self.assertRaisesRegex(
                    analysis.CandidateTemporalAnalysisError,
                    "payload path identity changed",
                ):
                    analysis.run_candidate_temporal_analysis(
                        screen_dir=screen_dir,
                        expected_screen_done_sha256=screen_done_sha,
                        output_dir=output,
                        bootstrap_repetitions=8,
                        seed=3,
                    )
            self.assertTrue(replaced)
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode),
                0o700,
            )

    def test_payload_in_place_rewrite_fails_sealed_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            output = root / "analysis"
            original = analysis._write_file_at
            rewritten = False

            def rewrite_payload_after_write(
                directory_fd: int,
                name: str,
                payload: bytes,
            ) -> tuple[int, tuple[int, ...]]:
                nonlocal rewritten
                bound_payload = original(directory_fd, name, payload)
                if not rewritten:
                    rewritten = True
                    os.chmod(
                        name,
                        0o600,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    replacement = os.open(
                        name,
                        os.O_WRONLY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        changed = (
                            bytes([payload[0] ^ 1]) + payload[1:]
                        )
                        os.pwrite(replacement, changed, 0)
                        os.fsync(replacement)
                    finally:
                        os.close(replacement)
                    os.chmod(
                        name,
                        0o444,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                return bound_payload

            with mock.patch.object(
                analysis,
                "_write_file_at",
                side_effect=rewrite_payload_after_write,
            ):
                with self.assertRaisesRegex(
                    analysis.CandidateTemporalAnalysisError,
                    "payload path identity changed",
                ):
                    analysis.run_candidate_temporal_analysis(
                        screen_dir=screen_dir,
                        expected_screen_done_sha256=screen_done_sha,
                        output_dir=output,
                        bootstrap_repetitions=8,
                        seed=3,
                    )
            self.assertTrue(rewritten)
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode),
                0o700,
            )

    def test_parent_path_replacement_cannot_rebind_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            payloads, input_value = analysis._derive(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                bootstrap_repetitions=8,
                seed=3,
            )
            parent = root / "published"
            parent.mkdir()
            output = parent / "analysis"
            displaced_parent = root / "displaced-published"
            original = analysis._assert_input_stable
            calls = 0

            def replace_parent_after_seal(value: object) -> None:
                nonlocal calls
                calls += 1
                original(value)
                if calls != 2:
                    return
                os.rename(parent, displaced_parent)
                parent.mkdir()
                output.mkdir(mode=0o700)
                for name, payload in payloads.items():
                    path = output / name
                    path.write_bytes(payload)
                    path.chmod(0o444)
                output.chmod(0o555)

            with mock.patch.object(
                analysis,
                "_assert_input_stable",
                side_effect=replace_parent_after_seal,
            ):
                identities = analysis._publish(
                    output,
                    payloads=payloads,
                    input_value=input_value,
                )
            self.assertEqual(calls, 2)
            self.assertNotEqual(
                analysis._capture_identities_for_names(
                    output,
                    analysis.OUTPUT_NAMES,
                ),
                identities,
            )
            with self.assertRaisesRegex(
                analysis.CandidateTemporalAnalysisError,
                "identities differ from the bound operation",
            ):
                analysis._validate_analysis_envelope(
                    output,
                    expected_identities=identities,
                )

    def test_resume_never_repairs_an_unsealed_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            output = root / "analysis"
            analysis.run_candidate_temporal_analysis(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                output_dir=output,
                bootstrap_repetitions=8,
                seed=3,
            )
            output.chmod(0o700)
            with self.assertRaisesRegex(
                analysis.CandidateTemporalAnalysisError,
                "requires a sealed output root",
            ):
                analysis.run_candidate_temporal_analysis(
                    screen_dir=screen_dir,
                    expected_screen_done_sha256=screen_done_sha,
                    output_dir=output,
                    bootstrap_repetitions=8,
                    seed=3,
                    resume=True,
                )
            output.chmod(0o555)

    def test_failure_after_temporary_root_seal_reverts_to_0700(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            screen_dir, screen_done_sha = _screen_commit(root)
            payloads, input_value = analysis._derive(
                screen_dir=screen_dir,
                expected_screen_done_sha256=screen_done_sha,
                bootstrap_repetitions=8,
                seed=3,
            )
            output = root / "analysis"
            original = analysis._assert_input_stable
            calls = 0

            def fail_after_root_seal(value: object) -> None:
                nonlocal calls
                calls += 1
                original(value)
                if calls == 2:
                    raise analysis.CandidateTemporalAnalysisError(
                        "injected late input failure"
                    )

            with mock.patch.object(
                analysis,
                "_assert_input_stable",
                side_effect=fail_after_root_seal,
            ):
                with self.assertRaisesRegex(
                    analysis.CandidateTemporalAnalysisError,
                    "injected late input failure",
                ):
                    analysis._publish(
                        output,
                        payloads=payloads,
                        input_value=input_value,
                    )
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode),
                0o700,
            )


if __name__ == "__main__":
    unittest.main()
