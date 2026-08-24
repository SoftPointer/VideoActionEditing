from __future__ import annotations

import hashlib
import json
import stat
import struct
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest.mock import patch

import numpy as np

from motive import r7_artifact_permissions as artifact_permissions
from motive import r7_expansion_dino_edges as dino_edges
from motive import r7_dino_quotient_calibration as quotient_calibration
from motive import r7_indexed_visual_graph_io as graph_io
from motive import r7_visual_graph_input as graph_input


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _dino_contract() -> dict[str, Any]:
    values = {
        "encoder_id": "facebook/dinov2-base",
        "encoder_revision": "a" * 40,
        "model_tree_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "model_file_count": 14,
        "frame_sampling_version": "uniform-6-from-uniform-32-v1",
        "preprocessing_version":
            "transformers-auto-image-processor-local-v1",
        "pooling": "last-hidden-state-cls-token-v1",
        "embedding_dim": 768,
        "dtype": "float32",
        "normalization": "l2-per-frame",
        "frozen_encoder": True,
        "local_files_only": True,
    }
    self_expected = set(graph_input.DINO_COMPARISON_FIELDS)
    if set(values) != self_expected:
        raise AssertionError("test DINO fixture differs from public contract")
    return values


def _publish_graph_directory(
    root: Path,
    *,
    rows: Sequence[dict[str, Any]],
    anchors: Sequence[str],
    candidates: Sequence[str],
    dino_contract: dict[str, Any],
) -> Path:
    directory = root / "graph-input"
    features = np.zeros((len(rows), 6, 768), dtype=np.float32)
    for index in range(len(rows)):
        features[index, :, index % 768] = 1.0
    derived = graph_input._Derived(
        rows=tuple(rows),
        arrays={
            "asset_indices": np.arange(len(rows), dtype=np.int64),
            "dino_cls": features,
        },
        summary_base={
            "schema_version": graph_input.SUMMARY_SCHEMA,
            "status": "complete",
            "assets": len(rows),
            "iids": len(rows) // 2,
            "candidate_iids": {
                "count": len(candidates),
                "sha256": _digest(list(candidates)),
            },
            "anchor_iids": {
                "count": len(anchors),
                "sha256": _digest(list(anchors)),
            },
            "asset_order":
                "lexicographic-iid-source-before-target-v1",
            "source_artifacts": {
                "candidate": {
                    "artifact_digest": "b" * 64,
                    "input_manifest_sha256": "1" * 64,
                    "final_done_sha256": "2" * 64,
                },
                "anchor": {
                    "artifact_digest": "a" * 64,
                    "input_manifest_sha256": "3" * 64,
                    "final_done_sha256": "4" * 64,
                },
            },
            "dino_contract": dino_contract,
            "split_assigned": False,
            "human_labels_asserted": False,
            "training_authorized": False,
        },
        paths={},
        snapshot={},
    )
    graph_input._atomic_publish(
        directory=directory,
        derived=derived,
        pre_publish_check=lambda: None,
    )
    graph_input.validate_graph_input_commit(directory)
    return directory


def _make_dino_directory(root: Path) -> Path:
    final = root / "dino-edges" / "final"
    final.mkdir(parents=True)
    names = (
        dino_edges.HARD_EDGES_NAME,
        dino_edges.AUDIT_EDGES_NAME,
        dino_edges.CALIBRATION_EDGES_NAME,
        dino_edges.SUMMARY_NAME,
        dino_edges.DONE_NAME,
    )
    for name in names:
        (final / name).write_bytes(f"dino:final:{name}\n".encode())
    final_quotient = final / dino_edges.IID_PAIR_MAXIMA_NAME
    final_quotient.mkdir()
    for name in (
        quotient_calibration.ARTIFACT_METADATA_NAME,
        quotient_calibration.ARTIFACT_ARRAYS_NAME,
        quotient_calibration.ARTIFACT_DONE_NAME,
    ):
        (final_quotient / name).write_bytes(
            f"dino:final:quotient:{name}\n".encode()
        )
    shards = final.parent / "shards"
    for rank in range(8):
        shard = shards / f"rank-{rank:05d}-of-00008"
        shard.mkdir(parents=True)
        for name in names:
            (shard / name).write_bytes(
                f"dino:rank:{rank}:{name}\n".encode()
            )
        rank_quotient = shard / dino_edges.QUOTIENT_RANK_PARTIAL_NAME
        rank_quotient.mkdir()
        for name in (
            quotient_calibration.ARTIFACT_METADATA_NAME,
            quotient_calibration.ARTIFACT_ARRAYS_NAME,
            quotient_calibration.ARTIFACT_DONE_NAME,
        ):
            (rank_quotient / name).write_bytes(
                f"dino:rank:{rank}:quotient:{name}\n".encode()
            )
    return final


def _row(
    iid: str,
    *,
    role: str,
    asset_index: int,
    anchor: bool,
    cohort: str,
    source_input_index: int,
) -> dict[str, Any]:
    video_sha = hashlib.sha256(
        f"video:{iid}:{role}".encode()
    ).hexdigest()
    # SHA-derived values are overwhelmingly farther than Hamming radius six
    # across different IIDs, while a pair intentionally shares its hashes.
    dhash = hashlib.sha256(f"dhash:{iid}".encode()).hexdigest()[:16]
    source_artifact = ("a" if anchor else "b") * 64
    index_digest = _digest(
        {
            "source_artifact_digest": source_artifact,
            "iid": iid,
            "role": role,
            "video_sha256": video_sha,
        }
    )
    return {
        "schema_version": graph_input.ROW_SCHEMA,
        "asset_index": asset_index,
        "iid": iid,
        "role": role,
        "anchor": anchor,
        "cohort": cohort,
        "video_sha256": video_sha,
        "dhashes": [dhash] * 6,
        "source_artifact_digest": source_artifact,
        "source_input_index": source_input_index,
        "source_index_digest": index_digest,
    }


class Fixture:
    def __init__(
        self,
        root: Path,
        *,
        anchor_iids: Sequence[str] = ("anchor",),
        candidate_iids: Sequence[str] = ("candidate", "fresh"),
        hard_links: Sequence[
            tuple[tuple[str, str], tuple[str, str]]
        ] = (),
    ) -> None:
        anchor_set = set(anchor_iids)
        candidate_set = set(candidate_iids)
        if anchor_set & candidate_set:
            raise ValueError("test anchor/candidate IID overlap")
        all_iids = sorted(anchor_set | candidate_set)
        rows: list[dict[str, Any]] = []
        source_indices = {"anchor": 0, "candidate": 0}
        for iid in all_iids:
            anchor = iid in anchor_set
            kind = "anchor" if anchor else "candidate"
            source_index = source_indices[kind]
            source_indices[kind] += 1
            cohort = "anchor_positive" if anchor else "pseudo_positive"
            for role in ("source", "target"):
                rows.append(
                    _row(
                        iid,
                        role=role,
                        asset_index=len(rows),
                        anchor=anchor,
                        cohort=cohort,
                        source_input_index=source_index,
                    )
                )
        anchors = sorted(anchor_set)
        candidates = sorted(candidate_set)
        dino_contract = _dino_contract()
        self.graph_directory = _publish_graph_directory(
            root,
            rows=rows,
            anchors=anchors,
            candidates=candidates,
            dino_contract=dino_contract,
        )
        self.dino_directory = _make_dino_directory(root)
        self.graph_result = graph_input.validate_graph_input_commit(
            self.graph_directory
        )
        graph_hashes = self.graph_result["artifact_hashes"]
        algorithm = dino_edges._algorithm_contract(
            block_size=32,
            audit_top_k=20,
        )
        contract = {
            "schema_version": dino_edges.MATCHER_SCHEMA,
            "input_directory": str(self.graph_directory.resolve()),
            "input_artifact_digest":
                self.graph_result["artifact_digest"],
            "input_artifacts": graph_hashes,
            "input_rows": len(rows),
            "dino_contract": dino_contract,
            "dino_contract_sha256": _digest(dino_contract),
            "algorithm": algorithm,
            "algorithm_sha256": _digest(algorithm),
            "implementation": {
                "module": "r7_expansion_dino_edges.py",
                "module_sha256": "d" * 64,
                "python": "3.12.0",
                "numpy": "1.26.4",
            },
            "runtime": {
                "torch_version": "2.7.1",
                "torch_cuda_version": None,
                "torch_hip_version": "6.3",
                "device_type": "cuda",
                "device_name": "MI210",
                "tf32_allowed": False,
            },
            "world_size": 8,
        }
        node_index = {
            (row["iid"], row["role"]): row["asset_index"] for row in rows
        }
        hard_edges: list[dict[str, Any]] = []
        for left, right in hard_links:
            asset_a, asset_b = sorted(
                (node_index[left], node_index[right])
            )
            first = rows[asset_a]
            second = rows[asset_b]
            score = np.float32(0.99)
            hard_edges.append(
                {
                    "schema_version": dino_edges.HARD_EDGE_SCHEMA,
                    "edge_type": "hard_dino",
                    "hard_edge": True,
                    "asset_a": asset_a,
                    "asset_b": asset_b,
                    "iid_a": first["iid"],
                    "role_a": first["role"],
                    "video_sha256_a": first["video_sha256"],
                    "iid_b": second["iid"],
                    "role_b": second["role"],
                    "video_sha256_b": second["video_sha256"],
                    "cosine": round(
                        float(score),
                        dino_edges.COSINE_ROUND_DECIMALS,
                    ),
                    "cosine_float32_hex":
                        struct.pack(">f", float(score)).hex(),
                    "frame_a": 0,
                    "frame_b": 1,
                    "owner_rank": asset_a % 8,
                    "world_size": 8,
                }
            )
        hard_edges.sort(key=lambda row: (row["asset_a"], row["asset_b"]))
        contract_sha = _digest(contract)
        expected_iid_pairs = len(rows) // 2 * (len(rows) // 2 - 1) // 2
        self.dino_result = {
            "paths": {
                name: self.dino_directory / name
                for name in (
                    dino_edges.HARD_EDGES_NAME,
                    dino_edges.AUDIT_EDGES_NAME,
                    dino_edges.CALIBRATION_EDGES_NAME,
                    dino_edges.SUMMARY_NAME,
                    dino_edges.DONE_NAME,
                )
            },
            "done": {
                "contract_sha256": contract_sha,
                "calibration_intended_use":
                    "diagnostic_not_threshold_calibrating",
                "thresholds_human_calibrated": False,
                "human_labels_asserted": False,
                "training_authorized": False,
            },
            "summary": {"contract_sha256": contract_sha},
            "contract": contract,
            "hard_edges": hard_edges,
            "audit_edges": [],
            "calibration_edges": [],
            "iid_pair_maxima": {
                "schema_version":
                    quotient_calibration.IID_PAIR_MAXIMA_SCHEMA,
                "contract": {
                    "expected_iid_pairs": expected_iid_pairs,
                    "observed_iid_pairs": expected_iid_pairs,
                    "partials_per_iid_pair": 2,
                    "observed_partials": 2 * expected_iid_pairs,
                    "coverage_complete": True,
                    "training_authorized": False,
                },
                "artifact_digest": "9" * 64,
            },
        }

    @contextmanager
    def patched(self) -> Iterator[None]:
        real_graph_validator = (
            graph_input.validate_graph_input_commit
        )
        with (
            patch(
                "motive.r7_indexed_visual_graph_io."
                "graph_input_module.validate_graph_input_commit",
                wraps=real_graph_validator,
            ) as graph_validator,
            patch(
                "motive.r7_indexed_visual_graph_io."
                "dino_edges.validate_final",
                return_value=self.dino_result,
            ) as dino_validator,
        ):
            yield
            self.graph_validator = graph_validator
            self.dino_validator = dino_validator

    def build(self, output: Path, *, resume: bool = False) -> dict[str, Any]:
        return graph_io.build_indexed_visual_graph_result(
            graph_input_dir=self.graph_directory,
            dino_edges_dir=self.dino_directory,
            output_dir=output,
            resume=resume,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class IndexedVisualGraphIoTests(unittest.TestCase):
    def test_anchor_relation_forces_candidate_train_and_warns_giant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(
                root,
                hard_links=(
                    (
                        ("anchor", "source"),
                        ("candidate", "source"),
                    ),
                ),
            )
            output = root / "result"
            with fixture.patched():
                done = fixture.build(output)
            artifact_permissions.assert_sealed_tree(output)
            self.assertEqual(
                stat.S_IMODE(output.stat().st_mode),
                0o555,
            )
            assignments = {
                row["iid"]: row
                for row in _read_jsonl(output / graph_io.ASSIGNMENTS_NAME)
            }
            candidate = assignments["candidate"]
            self.assertEqual(candidate["split"], "train")
            self.assertFalse(candidate["fresh"])
            self.assertTrue(candidate["forced_train"])
            self.assertTrue(candidate["forced_by_anchor"])
            self.assertFalse(candidate["anchor"])
            self.assertTrue(assignments["anchor"]["anchor"])
            summary = json.loads(
                (output / graph_io.SUMMARY_NAME).read_text()
            )
            self.assertTrue(summary["giant_component_warning"])
            self.assertEqual(
                summary["counts"]["candidate_iids"],
                2,
            )
            self.assertEqual(summary["counts"]["anchor_iids"], 1)
            self.assertFalse(summary["thresholds_human_calibrated"])
            self.assertFalse(summary["formal_split"])
            self.assertFalse(summary["training_authorized"])
            self.assertFalse(done["training_authorized"])
            self.assertEqual(fixture.graph_validator.call_count, 1)
            self.assertEqual(fixture.dino_validator.call_count, 1)

    def test_fixed_split_contains_fresh_evaluation_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(
                root,
                anchor_iids=(),
                candidate_iids=tuple(
                    f"candidate-{index:03d}" for index in range(64)
                ),
            )
            output = root / "result"
            with fixture.patched():
                fixture.build(output)
            rows = _read_jsonl(output / graph_io.ASSIGNMENTS_NAME)
            evaluation = [
                row
                for row in rows
                if row["split"] in {"validation", "test"}
            ]
            self.assertTrue(evaluation)
            self.assertTrue(all(row["fresh"] for row in rows))
            self.assertTrue(all(not row["forced_train"] for row in rows))
            summary = json.loads(
                (output / graph_io.SUMMARY_NAME).read_text()
            )
            self.assertEqual(
                summary["config"]["minimum_dino_cosine"],
                0.96,
            )
            self.assertEqual(
                summary["config"]["maximum_dhash_hamming"],
                6,
            )
            self.assertEqual(summary["config"]["data_seed"], 260108828)
            self.assertEqual(
                sum(summary["split_iid_counts"].values()),
                64,
            )

    def test_resume_is_byte_verification_and_nonresume_never_overwrites(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            first = root / "first"
            second = root / "second"
            with fixture.patched():
                fixture.build(first)
                artifact_permissions.assert_sealed_tree(first)
                mode_probe = first / graph_io.ASSIGNMENTS_NAME
                mode_probe.chmod(0o644)
                with self.assertRaisesRegex(ValueError, "mode differs"):
                    fixture.build(first, resume=True)
                mode_probe.chmod(0o444)
                before = {
                    path.name: (
                        path.read_bytes(),
                        path.stat().st_mtime_ns,
                    )
                    for path in first.iterdir()
                }
                with self.assertRaises(FileExistsError):
                    fixture.build(first)
                resumed = fixture.build(first, resume=True)
                fixture.build(second)
            after = {
                path.name: (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
                for path in first.iterdir()
            }
            self.assertEqual(before, after)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in first.iterdir()
                },
                {
                    path.name: path.read_bytes()
                    for path in second.iterdir()
                },
            )

    def test_atomic_publish_failure_thaws_and_removes_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "failed"
            payloads = {
                name: f"{name}\n".encode("utf-8")
                for name in graph_io.OUTPUT_NAMES
            }
            with (
                patch.object(
                    graph_io.os,
                    "rename",
                    side_effect=OSError("injected rename failure"),
                ),
                self.assertRaisesRegex(OSError, "injected"),
            ):
                graph_io._atomic_publish(
                    target,
                    payloads=payloads,
                    pre_publish_check=lambda: None,
                )
            self.assertFalse(target.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_graph_input_and_dino_tamper_are_rejected_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "result"
            with fixture.patched():
                fixture.build(output)
            with (
                fixture.graph_directory / graph_input.MANIFEST_NAME
            ).open("ab") as handle:
                handle.write(b"tamper")
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "digest|hash",
            ):
                fixture.build(output, resume=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "result"
            with fixture.patched():
                fixture.build(output)
            with (
                fixture.dino_directory / dino_edges.HARD_EDGES_NAME
            ).open("ab") as handle:
                handle.write(b"tamper")
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "byte-for-byte",
            ):
                fixture.build(output, resume=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "result"
            with fixture.patched():
                fixture.build(output)
            nested_arrays = (
                fixture.dino_directory
                / dino_edges.IID_PAIR_MAXIMA_NAME
                / quotient_calibration.ARTIFACT_ARRAYS_NAME
            )
            with nested_arrays.open("ab") as handle:
                handle.write(b"tamper")
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "byte-for-byte",
            ):
                fixture.build(output, resume=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            output = root / "result"
            with fixture.patched():
                fixture.build(output)
            with (
                fixture.dino_directory
                / dino_edges.CALIBRATION_EDGES_NAME
            ).open("ab") as handle:
                handle.write(b"tamper")
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "byte-for-byte",
            ):
                fixture.build(output, resume=True)

    def test_dino_endpoint_and_threshold_contract_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(
                root,
                hard_links=(
                    (
                        ("anchor", "source"),
                        ("candidate", "source"),
                    ),
                ),
            )
            fixture.dino_result["hard_edges"][0]["iid_a"] = "wrong"
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "identity binding differs",
            ):
                fixture.build(root / "result")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root)
            fixture.dino_result["contract"]["algorithm"][
                "hard_threshold"
            ] = 0.95
            # Keep the outer contract chain internally self-consistent so
            # the explicit threshold check is the reason for rejection.
            contract = fixture.dino_result["contract"]
            fixture.dino_result["summary"]["contract_sha256"] = _digest(
                contract
            )
            fixture.dino_result["done"]["contract_sha256"] = _digest(
                contract
            )
            with fixture.patched(), self.assertRaisesRegex(
                ValueError,
                "threshold binding differs",
            ):
                fixture.build(root / "result")


if __name__ == "__main__":
    unittest.main()
