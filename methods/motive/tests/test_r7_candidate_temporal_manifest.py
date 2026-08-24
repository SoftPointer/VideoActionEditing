from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from motive import r7_candidate_temporal_manifest as temporal


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@contextmanager
def _small_contract() -> Iterator[None]:
    with patch.multiple(
        temporal,
        POSITIVE_CENSUS=2,
        TRUSTED_NEGATIVE_POPULATION=4,
        TRUSTED_NEGATIVE_SAMPLE=2,
        CANDIDATE_POPULATION=6,
        EXPECTED_ANCHOR_ROWS=1,
        EXPECTED_INDEXED_ROWS=7,
    ):
        yield


def _candidate(
    iid: str,
    *,
    cohort: str,
    expansion_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    positive = cohort == "pseudo_positive"
    source_label: dict[str, Any] = {
        "bucket": "positive" if positive else "negative",
        "primary_family": "walk",
    }
    if positive:
        source_label["action_signature"] = "walk_right"
    else:
        source_label["negative_role"] = "pseudo_negative"
        source_label["negative_type"] = "static"
    source = {
        "iid": iid,
        "input_digest": _sha(f"input:{iid}".encode()),
        "prompt": f"make the person move for {iid}",
        "src_video": f"videos/{iid}/source.mp4",
        "tgt_video": f"videos/{iid}/target.mp4",
        "r7_expansion_manifest": source_label,
    }
    row = {
        "schema_version": temporal.candidate_module.ROW_SCHEMA,
        "iid": iid,
        "input_digest": source["input_digest"],
        "prompt": source["prompt"],
        "src_video": source["src_video"],
        "tgt_video": source["tgt_video"],
        "cohort": cohort,
        "primary_family": "walk",
        "source_row_sha256": _digest(source),
        "source_artifact_digest": expansion_digest,
        "split_assigned": False,
        "human_label": False,
        "training_eligible": False,
    }
    return row, source


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.expansion_digest = "a" * 64
        self.candidate_digest = "b" * 64
        self.indexed_digest = "c" * 64
        self.upstream = {
            "dino_edges": "d" * 64,
            "visual_graph": "e" * 64,
        }
        self.expansion = root / "expansion"
        self.candidates = root / "candidates"
        self.indexed = root / "indexed"
        self.data = root / "data"
        for directory, names in (
            (self.expansion, temporal._SOURCE_NAMES),
            (self.candidates, temporal._CANDIDATE_NAMES),
            (self.indexed, temporal._INDEXED_NAMES),
        ):
            directory.mkdir()
            for name in names:
                (directory / name).write_bytes(
                    f"fixture:{directory.name}:{name}\n".encode()
                )
        self.data.mkdir()

        specifications = [
            ("pos-000", "pseudo_positive"),
            ("neg-000", "pseudo_negative"),
            ("neg-001", "pseudo_negative"),
            ("pos-001", "pseudo_positive"),
            ("neg-002", "pseudo_negative"),
            ("neg-003", "pseudo_negative"),
        ]
        self.rows: list[dict[str, Any]] = []
        self.source_by_iid: dict[str, dict[str, Any]] = {}
        for index, (iid, cohort) in enumerate(specifications):
            row, source = _candidate(
                iid,
                cohort=cohort,
                expansion_digest=self.expansion_digest,
            )
            self.rows.append(row)
            self.source_by_iid[iid] = {
                "row": source,
                "line_number": index + 1,
            }
            for relative in (row["src_video"], row["tgt_video"]):
                path = self.data / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"media:{relative}\n".encode())

        self.assignments = {}
        self.assignment_index = {}
        for index, row in enumerate(sorted(self.rows, key=lambda x: x["iid"])):
            iid = row["iid"]
            self.assignments[iid] = {
                "schema_version": (
                    temporal.indexed_io.ASSIGNMENT_ROW_SCHEMA
                ),
                "iid": iid,
                "component_id": f"component-{iid}",
                "split": ("validation" if iid == "pos-001" else "train"),
                "fresh": True,
                "forced_train": False,
                "forced_by_anchor": False,
                "forced_by_previously_seen": False,
                "anchor": False,
                "cohort": row["cohort"],
            }
            self.assignment_index[iid] = index

    def candidate_result(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "source_by_iid": self.source_by_iid,
            "expansion_artifact_digest": self.expansion_digest,
            "candidate_artifact_digest": self.candidate_digest,
            "candidate_done": {
                "input_artifact_digest": self.expansion_digest
            },
        }

    def indexed_result(self) -> dict[str, Any]:
        return {
            "assignment_by_iid": self.assignments,
            "assignment_index": self.assignment_index,
            "artifact_digest": self.indexed_digest,
            "upstream_artifact_digests": self.upstream,
        }

    def build(self, output: Path, *, resume: bool = False) -> dict[str, Any]:
        with patch.object(
            temporal,
            "_validate_candidate_commit",
            return_value=self.candidate_result(),
        ), patch.object(
            temporal,
            "_validate_indexed_commit",
            return_value=self.indexed_result(),
        ):
            return temporal.build_candidate_temporal_manifest(
                expansion_manifest_dir=self.expansion,
                candidate_manifest_dir=self.candidates,
                indexed_graph_dir=self.indexed,
                data_root=self.data,
                expected_expansion_artifact_digest=self.expansion_digest,
                expected_candidate_artifact_digest=self.candidate_digest,
                expected_indexed_artifact_digest=self.indexed_digest,
                output_dir=output,
                resume=resume,
            )


def _component(
    iid: str,
    *,
    anchor: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = ((iid, "source"), (iid, "target"))
    component_id = temporal.indexed_core._component_id(assets)
    forced = anchor
    component = {
        "schema_version": temporal.indexed_io.COMPONENT_ROW_SCHEMA,
        "component_id": component_id,
        "member_assets": [[iid, "source"], [iid, "target"]],
        "member_iids": [iid],
        "split": "train" if forced else "validation",
        "fresh": not forced,
        "forced_train": forced,
        "anchor_iids": [iid] if anchor else [],
        "previously_seen_iids": [],
    }
    assignment = {
        "schema_version": temporal.indexed_io.ASSIGNMENT_ROW_SCHEMA,
        "iid": iid,
        "component_id": component_id,
        "split": component["split"],
        "fresh": component["fresh"],
        "forced_train": component["forced_train"],
        "forced_by_anchor": anchor,
        "forced_by_previously_seen": False,
        "anchor": anchor,
        "cohort": "anchor_positive" if anchor else "pseudo_positive",
    }
    return component, assignment


def _publish_indexed(
    directory: Path,
    *,
    candidate_rows: list[dict[str, Any]],
) -> str:
    components: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for row in candidate_rows:
        component, assignment = _component(row["iid"], anchor=False)
        assignment["cohort"] = row["cohort"]
        components.append(component)
        assignments.append(assignment)
    anchor_component, anchor_assignment = _component(
        "anchor-000", anchor=True
    )
    components.append(anchor_component)
    assignments.append(anchor_assignment)
    components.sort(key=lambda row: row["component_id"])
    assignments.sort(key=lambda row: row["iid"])
    edges: list[dict[str, Any]] = []

    directory.mkdir()
    assignments_raw = _jsonl(assignments)
    components_raw = _jsonl(components)
    edges_raw = _jsonl(edges)
    (directory / temporal.indexed_io.ASSIGNMENTS_NAME).write_bytes(
        assignments_raw
    )
    (directory / temporal.indexed_io.COMPONENTS_NAME).write_bytes(
        components_raw
    )
    (directory / temporal.indexed_io.SPANNING_EDGES_NAME).write_bytes(
        edges_raw
    )
    split_counts = {
        split: sum(row["split"] == split for row in assignments)
        for split in temporal._SPLITS
    }
    freshness = {
        "fresh_iids": sum(row["fresh"] for row in assignments),
        "nonfresh_iids": sum(not row["fresh"] for row in assignments),
        "forced_train_iids": sum(
            row["forced_train"] for row in assignments
        ),
        "forced_by_anchor_iids": sum(
            row["forced_by_anchor"] for row in assignments
        ),
        "forced_by_previously_seen_iids": 0,
    }
    outputs = {
        temporal.indexed_io.ASSIGNMENTS_NAME: {
            "rows": len(assignments),
            "sha256": _sha(assignments_raw),
            "order": "iid",
        },
        temporal.indexed_io.COMPONENTS_NAME: {
            "rows": len(components),
            "sha256": _sha(components_raw),
            "order": "component_id",
        },
        temporal.indexed_io.SPANNING_EDGES_NAME: {
            "rows": 0,
            "sha256": _sha(edges_raw),
            "order": "canonical-endpoints-relation-value",
        },
    }
    summary = {
        "schema_version": temporal.indexed_io.SUMMARY_SCHEMA,
        "status": "complete",
        "assignment_semantics":
            "diagnostic-provisional-component-split-v1",
        "input_bindings": {
            "dino_edges": {"artifact_digest": "d" * 64},
            "visual_graph": {"artifact_digest": "e" * 64},
        },
        "counts": {
            "candidate_iids": len(candidate_rows),
            "anchor_iids": 1,
            "total_iids": len(assignments),
            "assets": 2 * len(assignments),
            "components": len(components),
            "spanning_edges": 0,
            "hard_dino_input_edges": 0,
        },
        "split_iid_counts": split_counts,
        "freshness_counts": freshness,
        "thresholds_human_calibrated": False,
        "formal_split": False,
        "training_authorized": False,
        "outputs": outputs,
    }
    summary_raw = _pretty(summary)
    (directory / temporal.indexed_io.SUMMARY_NAME).write_bytes(summary_raw)
    artifact_hashes = {
        name: record["sha256"] for name, record in outputs.items()
    }
    artifact_hashes[temporal.indexed_io.SUMMARY_NAME] = _sha(summary_raw)
    done = {
        "schema_version": temporal.indexed_io.DONE_SCHEMA,
        "status": "complete",
        "iids": len(assignments),
        "components": len(components),
        "input_artifact_digests": {
            "dino_edges": "d" * 64,
            "visual_graph": "e" * 64,
        },
        "artifacts": {
            name: {"filename": name, "sha256": digest}
            for name, digest in sorted(artifact_hashes.items())
        },
        "artifact_digest": _digest(artifact_hashes),
        "giant_component_warning": False,
        "thresholds_human_calibrated": False,
        "formal_split": False,
        "training_authorized": False,
    }
    (directory / temporal.indexed_io.DONE_NAME).write_bytes(_pretty(done))
    return done["artifact_digest"]


def _rechain_indexed(directory: Path) -> str:
    names = (
        temporal.indexed_io.ASSIGNMENTS_NAME,
        temporal.indexed_io.COMPONENTS_NAME,
        temporal.indexed_io.SPANNING_EDGES_NAME,
    )
    summary_path = directory / temporal.indexed_io.SUMMARY_NAME
    summary = json.loads(summary_path.read_text())
    for name in names:
        raw = (directory / name).read_bytes()
        summary["outputs"][name]["rows"] = len(raw.splitlines())
        summary["outputs"][name]["sha256"] = _sha(raw)
    summary_raw = _pretty(summary)
    summary_path.write_bytes(summary_raw)
    hashes = {
        name: summary["outputs"][name]["sha256"] for name in names
    }
    hashes[temporal.indexed_io.SUMMARY_NAME] = _sha(summary_raw)
    done_path = directory / temporal.indexed_io.DONE_NAME
    done = json.loads(done_path.read_text())
    done["artifacts"] = {
        name: {"filename": name, "sha256": digest}
        for name, digest in sorted(hashes.items())
    }
    done["artifact_digest"] = _digest(hashes)
    done_path.write_bytes(_pretty(done))
    return done["artifact_digest"]


class R7CandidateTemporalManifestTests(unittest.TestCase):
    def test_frozen_selection_key_vector(self) -> None:
        self.assertEqual(temporal.SELECTION_SEED, 260108834)
        self.assertEqual(
            temporal._selection_key(iid="iid-00000"),
            "07c7e25d5d45674d45e11ed8ab56071f"
            "90fb213248692ea04e3f3d27ee1c579c",
        )

    def test_build_validate_bottom_k_resume_and_external_anchors(
        self,
    ) -> None:
        with _small_contract(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            output = root / "temporal"
            result = fixture.build(output)

            self.assertFalse(result["resume_verified"])
            rows = result["rows"]
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                sum(row["label"]["class"] == "positive" for row in rows),
                2,
            )
            negatives = [
                row for row in rows if row["label"]["class"] == "negative"
            ]
            expected_negative_iids = {
                iid
                for _, iid in sorted(
                    (
                        temporal._selection_key(iid=row["iid"]),
                        row["iid"],
                    )
                    for row in fixture.rows
                    if row["cohort"] == "pseudo_negative"
                )[:2]
            }
            self.assertEqual(
                {row["iid"] for row in negatives},
                expected_negative_iids,
            )
            self.assertEqual(
                sorted(row["sampling"]["selection_rank"] for row in negatives),
                [1, 2],
            )
            self.assertTrue(
                all(
                    row["label"]["action_signature"] is None
                    for row in negatives
                )
            )
            self.assertTrue(
                all(
                    isinstance(row["label"]["action_signature"], str)
                    and row["label"]["action_signature"]
                    for row in rows
                    if row["label"]["class"] == "positive"
                )
            )
            for row in rows:
                self.assertFalse(row["assignment"]["anchor"])
                self.assertFalse(row["formal_evidence"])
                self.assertFalse(row["formal_split"])
                self.assertFalse(row["human_labels_asserted"])
                self.assertFalse(row["training_authorized"])
                self.assertFalse(row["generation_authorized"])
                self.assertNotIn("r5_pilot_label", _canonical(row).decode())

            summary_anchor = root / "expected-summary.json"
            done_anchor = root / "expected-done.json"
            shutil.copyfile(output / temporal.SUMMARY_NAME, summary_anchor)
            shutil.copyfile(output / temporal.DONE_NAME, done_anchor)
            checked = temporal.validate_candidate_temporal_manifest(
                output,
                expected_summary_path=summary_anchor,
                expected_done_path=done_anchor,
            )
            self.assertEqual(checked["rows"], rows)
            anchored_summary = summary_anchor.read_bytes()
            summary_anchor.write_bytes(anchored_summary + b" ")
            with self.assertRaisesRegex(ValueError, "external expected"):
                temporal.validate_candidate_temporal_manifest(
                    output,
                    expected_summary_path=summary_anchor,
                )
            summary_anchor.write_bytes(anchored_summary)
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode), 0o555
            )
            for name in temporal.OUTPUT_NAMES:
                self.assertEqual(
                    stat.S_IMODE((output / name).stat().st_mode),
                    0o444,
                )

            resumed = fixture.build(output, resume=True)
            self.assertTrue(resumed["resume_verified"])
            with self.assertRaises(FileExistsError):
                fixture.build(output)

    def test_sampling_tamper_extra_artifact_and_media_change_fail_closed(
        self,
    ) -> None:
        with _small_contract(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            output = root / "temporal"
            fixture.build(output)
            manifest_path = output / temporal.MANIFEST_NAME
            original = manifest_path.read_bytes()
            rows = [
                json.loads(line) for line in original.splitlines()
            ]
            negative = next(
                row for row in rows if row["label"]["class"] == "negative"
            )
            negative["sampling"]["selection_key_sha256"] = "0" * 64
            os.chmod(output, 0o755)
            os.chmod(manifest_path, 0o644)
            manifest_path.write_bytes(_jsonl(rows))
            os.chmod(manifest_path, 0o444)
            os.chmod(output, 0o555)
            with self.assertRaisesRegex(ValueError, "selection key"):
                temporal.validate_candidate_temporal_manifest(output)
            os.chmod(output, 0o755)
            os.chmod(manifest_path, 0o644)
            manifest_path.write_bytes(original)
            os.chmod(manifest_path, 0o444)

            extra = output / "unexpected"
            os.chmod(output, 0o755)
            extra.write_text("x")
            os.chmod(extra, 0o444)
            os.chmod(output, 0o555)
            with self.assertRaisesRegex(ValueError, "artifact set"):
                temporal.validate_candidate_temporal_manifest(output)
            os.chmod(output, 0o755)
            os.chmod(extra, 0o644)
            extra.unlink()
            os.chmod(output, 0o555)

            os.chmod(output / temporal.DONE_NAME, 0o644)
            with self.assertRaisesRegex(ValueError, "0444"):
                temporal.validate_candidate_temporal_manifest(output)
            os.chmod(output / temporal.DONE_NAME, 0o444)

            selected_path = fixture.data / rows[0]["src_video"]
            selected_path.write_bytes(b"changed media\n")
            with self.assertRaises(RuntimeError):
                fixture.build(output, resume=True)

    def test_media_traversal_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            outside = root / "outside.mp4"
            outside.write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "traversal"):
                temporal._hash_media(data, "../outside.mp4")
            link = data / "link.mp4"
            link.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "safely"):
                temporal._hash_media(data, "link.mp4")

    def test_indexed_commit_rejects_anchor_injection_duplicate_and_cohort(
        self,
    ) -> None:
        with _small_contract(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            indexed = root / "real-indexed"
            digest = _publish_indexed(
                indexed,
                candidate_rows=fixture.rows,
            )
            valid = temporal._validate_indexed_commit(
                indexed_graph_dir=indexed,
                candidate_rows=fixture.rows,
                expected_indexed_artifact_digest=digest,
            )
            self.assertEqual(len(valid["anchor_iids"]), 1)

            assignment_path = (
                indexed / temporal.indexed_io.ASSIGNMENTS_NAME
            )
            clean = assignment_path.read_bytes()
            rows = [
                json.loads(line) for line in clean.splitlines()
            ]
            anchor = next(row for row in rows if row["anchor"])
            anchor["cohort"] = "anchor_negative"
            assignment_path.write_bytes(_jsonl(rows))
            changed_digest = _rechain_indexed(indexed)
            temporal._validate_indexed_commit(
                indexed_graph_dir=indexed,
                candidate_rows=fixture.rows,
                expected_indexed_artifact_digest=changed_digest,
            )

            rows = [
                json.loads(line) for line in clean.splitlines()
            ]
            anchor = next(row for row in rows if row["anchor"])
            anchor["cohort"] = "anchor"
            assignment_path.write_bytes(_jsonl(rows))
            changed_digest = _rechain_indexed(indexed)
            with self.assertRaisesRegex(ValueError, "invalid cohort"):
                temporal._validate_indexed_commit(
                    indexed_graph_dir=indexed,
                    candidate_rows=fixture.rows,
                    expected_indexed_artifact_digest=changed_digest,
                )

            assignment_path.write_bytes(clean)
            _rechain_indexed(indexed)
            rows = [
                json.loads(line) for line in clean.splitlines()
            ]
            target = next(row for row in rows if not row["anchor"])
            target["anchor"] = True
            assignment_path.write_bytes(_jsonl(rows))
            changed_digest = _rechain_indexed(indexed)
            with self.assertRaisesRegex(
                ValueError, "non-anchor|anchor census"
            ):
                temporal._validate_indexed_commit(
                    indexed_graph_dir=indexed,
                    candidate_rows=fixture.rows,
                    expected_indexed_artifact_digest=changed_digest,
                )

            assignment_path.write_bytes(clean)
            _rechain_indexed(indexed)
            rows = [
                json.loads(line) for line in clean.splitlines()
            ]
            rows[1]["iid"] = rows[0]["iid"]
            assignment_path.write_bytes(_jsonl(rows))
            changed_digest = _rechain_indexed(indexed)
            with self.assertRaisesRegex(ValueError, "IID ordered|duplicate"):
                temporal._validate_indexed_commit(
                    indexed_graph_dir=indexed,
                    candidate_rows=fixture.rows,
                    expected_indexed_artifact_digest=changed_digest,
                )

            assignment_path.write_bytes(clean)
            _rechain_indexed(indexed)
            rows = [
                json.loads(line) for line in clean.splitlines()
            ]
            target = next(
                row
                for row in rows
                if not row["anchor"]
                and row["cohort"] == "pseudo_positive"
            )
            target["cohort"] = "pseudo_negative"
            assignment_path.write_bytes(_jsonl(rows))
            changed_digest = _rechain_indexed(indexed)
            with self.assertRaisesRegex(ValueError, "cohort mismatch"):
                temporal._validate_indexed_commit(
                    indexed_graph_dir=indexed,
                    candidate_rows=fixture.rows,
                    expected_indexed_artifact_digest=changed_digest,
                )

    def test_indexed_done_upstream_mismatch_is_rejected(self) -> None:
        with _small_contract(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            indexed = root / "indexed-real"
            digest = _publish_indexed(
                indexed,
                candidate_rows=fixture.rows,
            )
            done_path = indexed / temporal.indexed_io.DONE_NAME
            done = json.loads(done_path.read_text())
            done["input_artifact_digests"]["visual_graph"] = "f" * 64
            done_path.write_bytes(_pretty(done))
            with self.assertRaisesRegex(ValueError, "hash/input chain"):
                temporal._validate_indexed_commit(
                    indexed_graph_dir=indexed,
                    candidate_rows=fixture.rows,
                    expected_indexed_artifact_digest=digest,
                )


if __name__ == "__main__":
    unittest.main()
