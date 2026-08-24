from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from motive.r7_preflight_extract import (
    R7_ROW_SCHEMA,
    _commit_final,
    _empty_arrays,
    compute_p0_gate,
)
from motive.r7_representation_probe import (
    MODALITIES,
    RepresentationProbeConfig,
    RepresentationProbeError,
    run_representation_probe,
    validate_probe_output,
)
from motive.r7_visual_split import (
    R7_FRESHNESS_POLICY_VERSION,
    R7_VISUAL_SPLIT_SCHEMA,
)


def _specification() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in ("a", "b"):
        for index in range(3):
            rows.append(
                {
                    "iid": f"{family}-train-{index}",
                    "family": family,
                    "split": "train",
                    "fresh": False,
                }
            )
    for family in ("a", "b"):
        for index in range(2):
            rows.append(
                {
                    "iid": f"{family}-test-{index}",
                    "family": family,
                    "split": "test",
                    "fresh": True,
                }
            )
    return rows


def _write_preflight(
    root: Path,
    rows: list[dict[str, object]],
    *,
    name: str = "preflight",
) -> Path:
    directory = root / name
    arrays = _empty_arrays(len(rows))
    arrays["input_indices"][:] = np.arange(len(rows), dtype=np.int64)
    arrays["positive"][:] = True
    manifest_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        family = str(row["family"])
        arrays["target_base_valid"][index] = bool(
            row.get("base_valid", True)
        )
        arrays["target_dino_valid"][index] = True
        if arrays["target_base_valid"][index]:
            teacher_coordinate = 0 if family == "a" else 1
            arrays["target_teacher_embedding"][index, teacher_coordinate] = 1.0
        # Appearance is deliberately anti-correlated on evaluation examples.
        if str(row["split"]) == "train":
            dino_coordinate = 0 if family == "a" else 1
        else:
            dino_coordinate = 1 if family == "a" else 0
        arrays["target_dino_cls"][index, :, dino_coordinate] = 1.0
        # Camera stays identically zero and is retained as a nuisance baseline.
        manifest_rows.append(
            {
                "schema_version": R7_ROW_SCHEMA,
                "input_index": index,
                "iid": row["iid"],
                "positive": True,
            }
        )
    input_manifest = root / f"{name}-input.jsonl"
    input_manifest.write_text(
        "".join(
            json.dumps({"iid": row["iid"]}, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    gate = compute_p0_gate(manifest_rows, arrays)
    _commit_final(
        directory=directory,
        rows=manifest_rows,
        arrays=arrays,
        input_manifest=input_manifest,
        shard_done_sha256=[],
        gate=gate,
    )
    return directory


def _write_split(
    path: Path,
    rows: list[dict[str, object]],
    *,
    audit_passed: bool = True,
    same_component: tuple[str, str] | None = None,
) -> None:
    component_by_iid = {
        str(row["iid"]): f"component-{row['iid']}" for row in rows
    }
    if same_component is not None:
        left, right = same_component
        component_by_iid[right] = component_by_iid[left]
    assignments = [
        {
            "iid": row["iid"],
            "split": row["split"],
            "component_id": component_by_iid[str(row["iid"])],
            "evaluation_fresh": row["fresh"],
            "forced_train_by_seen_component": not bool(row["fresh"]),
        }
        for row in rows
    ]
    components = [
        {
            "component_id": component,
            "member_nodes": [],
            "member_iids": sorted(
                str(row["iid"])
                for row in rows
                if component_by_iid[str(row["iid"])] == component
            ),
            "split": next(
                str(row["split"])
                for row in rows
                if component_by_iid[str(row["iid"])] == component
            ),
            "seen_iids": [],
            "forced_train_by_seen_component": False,
        }
        for component in sorted(set(component_by_iid.values()))
    ]
    audit = {
        "samples": len(rows),
        "assets": 2 * len(rows),
        "components": len(components),
        "split_counts": [],
        "component_split_counts": [],
        "edge_counts": [],
        "cross_split_component_ids": [],
        "cross_split_relation_edges": [],
        "assignment_component_mismatches": [],
        "stable_split_mismatches": [],
        "seen_component_evaluation_iids": [],
        "passed": audit_passed,
    }

    def digest(value: object) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    placeholder = "0" * 64
    provenance = {
        "schema_version": R7_VISUAL_SPLIT_SCHEMA,
        "implementation_version": "r7-visual-split-numpy-v1",
        "split_version": "joint-source-target-exact-phash-dino-dsu-v1",
        "freshness_policy_version": R7_FRESHNESS_POLICY_VERSION,
        "config_digest": placeholder,
        "dino_provenance_digest": placeholder,
        "input_pairs_digest": placeholder,
        "prior_seen_iid_ledger_digest": placeholder,
        "prior_seen_iid_count": 0,
        "matched_prior_seen_iids": [],
        "edges_digest": digest([]),
        "components_digest": digest(components),
        "assignments_digest": digest(assignments),
        "audit_digest": digest(audit),
    }
    provenance["provenance_digest"] = digest(provenance)
    payload = {
        "assignments": assignments,
        "components": components,
        "edges": [],
        "audit": audit,
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_labels(
    path: Path,
    rows: list[dict[str, object]],
    *,
    provenance: str = "human",
    pair_overrides: dict[str, str] | None = None,
    incomplete_human: bool = False,
) -> None:
    output: list[str] = []
    for index, row in enumerate(rows):
        if provenance == "human":
            label_provenance: dict[str, object] = {"kind": "human"}
            if not incomplete_human:
                label_provenance.update(
                    {
                        "annotation_id": f"annotation-{index}",
                        "annotator_id": "reviewer-1",
                    }
                )
        else:
            label_provenance = {
                "kind": "pseudo",
                "method": "qwen-rule-fusion-v1",
            }
        iid = str(row["iid"])
        value = {
            "iid": iid,
            "pair_id": (pair_overrides or {}).get(iid, iid),
            "action_signature": row["family"],
            "label_provenance": label_provenance,
        }
        output.append(json.dumps(value, sort_keys=True) + "\n")
    path.write_text("".join(output), encoding="utf-8")


def _config() -> RepresentationProbeConfig:
    return RepresentationProbeConfig(
        eval_splits=("test",),
        minimum_total_train_references=5,
        minimum_train_references_per_family=3,
        minimum_train_components_per_family=3,
        minimum_eval_queries_per_family=2,
        minimum_eval_components=2,
        bootstrap_repetitions=32,
        permutation_repetitions=32,
        seed=1701,
    )


class R7RepresentationProbeTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        rows: list[dict[str, object]] | None = None,
        provenance: str = "human",
        audit_passed: bool = True,
        incomplete_human: bool = False,
        pair_overrides: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, object]], Path, Path, Path]:
        specification = rows or _specification()
        preflight = _write_preflight(root, specification)
        split = root / "split.json"
        labels = root / "labels.jsonl"
        _write_split(
            split,
            specification,
            audit_passed=audit_passed,
        )
        _write_labels(
            labels,
            specification,
            provenance=provenance,
            incomplete_human=incomplete_human,
            pair_overrides=pair_overrides,
        )
        return specification, preflight, split, labels

    def test_primary_retrieval_baselines_bootstrap_and_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, preflight, split, labels = self._fixture(root)
            output = root / "output"
            done = run_representation_probe(
                preflight_final_directories=[preflight],
                visual_split=split,
                labels=labels,
                output_dir=output,
                config=_config(),
            )
            self.assertEqual(done["formal_status"], "FORMAL_EVALUABLE")
            self.assertIsNone(done["formal_probe_passed"])
            self.assertFalse(done["generation_authorized"])
            validated = validate_probe_output(output)
            summary = validated["summary"]
            teacher = summary["metrics"]["target_actor_teacher"]
            appearance = summary["metrics"]["target_pooled_dino"]
            self.assertEqual(teacher["micro"]["r_at_1"], 1.0)
            self.assertEqual(teacher["macro_family"]["r_at_1"], 1.0)
            self.assertEqual(appearance["micro"]["r_at_1"], 0.0)
            self.assertEqual(
                set(teacher["per_family"]),
                {"a", "b"},
            )
            bootstrap = teacher["component_bootstrap_95"]
            self.assertEqual(bootstrap["repetitions"], 32)
            self.assertEqual(
                set(bootstrap["per_family_intervals"]), {"a", "b"}
            )
            null = teacher["label_permutation_null"]
            self.assertEqual(null["repetitions"], 32)
            self.assertIn(
                "one_sided_p_value",
                null["statistics"]["micro_r_at_1"],
            )
            self.assertTrue(
                summary["coverage"][
                    "comparison_modalities_share_exact_cohort"
                ]
            )
            self.assertEqual(
                summary["coverage"]["zero_camera_trajectory_rows"], 10
            )
            for row in validated["per_query"]:
                for modality in MODALITIES:
                    result = row["modalities"][modality]
                    self.assertNotIn(
                        row["iid"], result["top_reference_iids"]
                    )
                    self.assertNotIn(
                        row["pair_id"], result["top_reference_pair_ids"]
                    )
                    self.assertNotIn(
                        row["component_id"],
                        result["top_reference_components"],
                    )

    def test_pseudo_or_incomplete_human_labels_are_insufficient(self) -> None:
        for provenance, incomplete in (("pseudo", False), ("human", True)):
            with self.subTest(provenance=provenance, incomplete=incomplete):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    _, preflight, split, labels = self._fixture(
                        root,
                        provenance=provenance,
                        incomplete_human=incomplete,
                    )
                    output = root / "output"
                    run_representation_probe(
                        preflight_final_directories=[preflight],
                        visual_split=split,
                        labels=labels,
                        output_dir=output,
                        config=_config(),
                    )
                    summary = validate_probe_output(output)["summary"]
                    self.assertEqual(
                        summary["formal_status"], "INSUFFICIENT"
                    )
                    self.assertFalse(
                        summary["decision"]["formal_evaluable"]
                    )
                    self.assertFalse(summary["generation_authorized"])

    def test_unattested_or_nonfresh_split_is_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, preflight, split, labels = self._fixture(
                root,
                audit_passed=False,
            )
            rows[-1]["fresh"] = False
            _write_split(split, rows, audit_passed=False)
            output = root / "output"
            run_representation_probe(
                preflight_final_directories=[preflight],
                visual_split=split,
                labels=labels,
                output_dir=output,
                config=_config(),
            )
            summary = validate_probe_output(output)["summary"]
            self.assertEqual(summary["formal_status"], "INSUFFICIENT")
            reasons = " ".join(summary["decision"]["formal_reasons"])
            self.assertIn("fresh", reasons)

    def test_tampered_split_provenance_cannot_be_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, preflight, split, labels = self._fixture(root)
            payload = json.loads(split.read_text(encoding="utf-8"))
            payload["provenance"]["assignments_digest"] = "f" * 64
            split.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            output = root / "output"
            run_representation_probe(
                preflight_final_directories=[preflight],
                visual_split=split,
                labels=labels,
                output_dir=output,
                config=_config(),
            )
            summary = validate_probe_output(output)["summary"]
            self.assertEqual(summary["formal_status"], "INSUFFICIENT")
            self.assertFalse(
                summary["contract"]["visual_split"][
                    "artifact_digests_valid"
                ]
            )

    def test_cross_split_visual_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _specification()
            preflight = _write_preflight(root, rows)
            split = root / "split.json"
            labels = root / "labels.jsonl"
            _write_split(
                split,
                rows,
                same_component=("a-train-0", "a-test-0"),
            )
            _write_labels(labels, rows)
            with self.assertRaisesRegex(
                RepresentationProbeError, "crosses"
            ):
                run_representation_probe(
                    preflight_final_directories=[preflight],
                    visual_split=split,
                    labels=labels,
                    output_dir=root / "output",
                    config=_config(),
                )

    def test_insufficient_train_family_support_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _specification()
            rows[0]["family"] = "other"
            _, preflight, split, labels = self._fixture(root, rows=rows)
            with self.assertRaisesRegex(
                RepresentationProbeError, "insufficient family support"
            ):
                run_representation_probe(
                    preflight_final_directories=[preflight],
                    visual_split=split,
                    labels=labels,
                    output_dir=root / "output",
                    config=_config(),
                )

    def test_same_logical_pair_is_masked_and_blocks_formal_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _specification()
            pair_overrides = {
                "a-train-0": "shared-pair",
                "a-test-0": "shared-pair",
            }
            _, preflight, split, labels = self._fixture(
                root,
                rows=rows,
                pair_overrides=pair_overrides,
            )
            output = root / "output"
            run_representation_probe(
                preflight_final_directories=[preflight],
                visual_split=split,
                labels=labels,
                output_dir=output,
                config=replace(
                    _config(),
                    minimum_train_references_per_family=2,
                    minimum_train_components_per_family=2,
                ),
            )
            validated = validate_probe_output(output)
            self.assertEqual(
                validated["summary"]["formal_status"], "INSUFFICIENT"
            )
            query = next(
                row
                for row in validated["per_query"]
                if row["iid"] == "a-test-0"
            )
            for modality in MODALITIES:
                self.assertNotIn(
                    "shared-pair",
                    query["modalities"][modality][
                        "top_reference_pair_ids"
                    ],
                )

    def test_base_invalid_rows_are_excluded_with_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _specification()
            rows.append(
                {
                    "iid": "dropped-train",
                    "family": "a",
                    "split": "train",
                    "fresh": False,
                    "base_valid": False,
                }
            )
            _, preflight, split, labels = self._fixture(root, rows=rows)
            output = root / "output"
            run_representation_probe(
                preflight_final_directories=[preflight],
                visual_split=split,
                labels=labels,
                output_dir=output,
                config=_config(),
            )
            coverage = validate_probe_output(output)["summary"]["coverage"]
            self.assertEqual(coverage["preflight_rows"], 11)
            self.assertEqual(coverage["target_base_invalid"], 1)
            self.assertEqual(coverage["common_cohort"], 10)

    def test_resume_is_exact_and_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, preflight, split, labels = self._fixture(root)
            output = root / "output"
            arguments = {
                "preflight_final_directories": [preflight],
                "visual_split": split,
                "labels": labels,
                "output_dir": output,
                "config": _config(),
            }
            first = run_representation_probe(**arguments)
            with self.assertRaises(FileExistsError):
                run_representation_probe(**arguments)
            resumed = run_representation_probe(**arguments, resume=True)
            self.assertEqual(first, resumed)
            per_query = output / "per_query.jsonl"
            per_query.write_text(
                per_query.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RepresentationProbeError, "digest mismatch"
            ):
                validate_probe_output(output)

    def test_multiple_preflights_may_not_duplicate_iids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = _specification()
            first = _write_preflight(root, rows, name="first")
            second = _write_preflight(root, rows, name="second")
            split = root / "split.json"
            labels = root / "labels.jsonl"
            _write_split(split, rows)
            _write_labels(labels, rows)
            with self.assertRaisesRegex(
                RepresentationProbeError, "duplicate iid"
            ):
                run_representation_probe(
                    preflight_final_directories=[first, second],
                    visual_split=split,
                    labels=labels,
                    output_dir=root / "output",
                    config=_config(),
                )


if __name__ == "__main__":
    unittest.main()
