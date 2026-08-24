from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest

import numpy as np

from motive import r7_artifact_permissions as artifact_permissions
from motive import r7_expansion_dino_edges as dino_edges
from motive import r7_visual_graph_input as graph_input
from motive.r7_dino_quotient_calibration import (
    RankQuotientAccumulator,
    make_graph_commit_binding,
    merge_exact8_rank_partials,
    publish_artifact_directory,
)
from motive.r7_quotient_evidence_io import (
    BASE_COMPONENTS_NAME,
    CALIBRATION_SAMPLE_DIR,
    COMPONENT_MAXIMA_DIR,
    DONE_NAME,
    OUTPUT_ENTRIES,
    SUMMARY_NAME,
    build_quotient_evidence,
    validate_quotient_evidence,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _far_dhash(label: str, frame: int) -> str:
    return hashlib.sha256(
        f"dhash:{label}:{frame}".encode("utf-8")
    ).hexdigest()[:16]


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for asset_index in range(12):
        iid_index = asset_index // 2
        iid = f"iid-{iid_index:02d}"
        role = "source" if asset_index % 2 == 0 else "target"
        anchor = iid_index in {0, 2}
        video_sha = _sha(f"video:{asset_index}")
        # One exact-SHA relation joins IID 0 and IID 1.
        if asset_index in {0, 2}:
            video_sha = _sha("shared-exact-video")
        dhashes = [
            _far_dhash(str(asset_index), frame)
            for frame in range(6)
        ]
        # One <=6 Hamming relation joins IID 2 and IID 3.
        if asset_index == 4:
            dhashes[0] = "1111111111111111"
        if asset_index == 7:
            dhashes[0] = "1111111111111116"
        rows.append(
            {
                "schema_version": graph_input.ROW_SCHEMA,
                "asset_index": asset_index,
                "iid": iid,
                "role": role,
                "anchor": anchor,
                "cohort": (
                    "anchor_positive"
                    if anchor
                    else "pseudo_positive"
                ),
                "video_sha256": video_sha,
                "dhashes": dhashes,
                "source_artifact_digest":
                    "a" * 64 if anchor else "b" * 64,
                "source_input_index": iid_index,
                "source_index_digest": _sha(
                    f"source-index:{asset_index}"
                ),
            }
        )
    return rows


def _features(assets: int, seed: int = 29) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(
        size=(assets, 6, 768)
    ).astype(np.float32)
    values /= np.linalg.norm(values, axis=2, keepdims=True)
    return values.astype(np.float32)


def _encoder() -> dict[str, object]:
    return {
        "encoder_id": "facebook/dinov2-base",
        "encoder_revision": "a" * 40,
        "model_tree_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "model_file_count": 14,
        "embedding_dim": 768,
        "dtype": "float32",
        "normalization": "l2-per-frame",
        "frozen_encoder": True,
        "local_files_only": True,
        "frame_sampling_version": "uniform-6-from-uniform-32-v1",
        "preprocessing_version":
            "transformers-auto-image-processor-local-v1",
        "pooling": "last-hidden-state-cls-token-v1",
    }


def _build_graph_commit(
    directory: Path,
    rows: list[dict[str, object]],
    *,
    feature_seed: int = 29,
) -> None:
    anchor_iids = sorted(
        {
            str(row["iid"])
            for row in rows
            if row["anchor"] is True
        }
    )
    candidate_iids = sorted(
        {
            str(row["iid"])
            for row in rows
            if row["anchor"] is False
        }
    )
    arrays = {
        "asset_indices": np.arange(len(rows), dtype=np.int64),
        "dino_cls": _features(len(rows), seed=feature_seed),
    }
    summary_base = {
        "schema_version": graph_input.SUMMARY_SCHEMA,
        "status": "complete",
        "assets": len(rows),
        "iids": len(rows) // 2,
        "candidate_iids": {
            "count": len(candidate_iids),
            "sha256": graph_input._object_digest(candidate_iids),
        },
        "anchor_iids": {
            "count": len(anchor_iids),
            "sha256": graph_input._object_digest(anchor_iids),
        },
        "asset_order": "lexicographic-iid-source-before-target-v1",
        "source_artifacts": {
            "candidate": {
                "artifact_digest": "b" * 64,
                "input_manifest_sha256": "d" * 64,
                "final_done_sha256": "e" * 64,
            },
            "anchor": {
                "artifact_digest": "a" * 64,
                "input_manifest_sha256": "f" * 64,
                "final_done_sha256": "1" * 64,
            },
        },
        "dino_contract": _encoder(),
        "split_assigned": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    derived = graph_input._Derived(
        rows=tuple(rows),
        arrays=arrays,
        summary_base=summary_base,
        paths={},
        snapshot={},
    )
    graph_input._atomic_publish(
        directory=directory,
        derived=derived,
        pre_publish_check=lambda: None,
    )
    graph_input.validate_graph_input_commit(directory)


def _score_matrix(assets: int, bias: float = 0.0) -> np.ndarray:
    scores = np.zeros((assets, assets), dtype=np.float32)
    for left in range(assets):
        for right in range(left + 1, assets):
            scores[left, right] = np.float32(
                min(0.999, 0.55 + bias + 0.01 * left + 0.007 * right)
            )
    # High DINO evidence across IID 4/5 must not form a base component.
    scores[8:10, 10:12] = np.float32(0.999)
    return scores


class _MatrixMatcher:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)

    def __call__(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        query_index: int,
        candidate_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del query, candidates
        indices = np.asarray(candidate_indices, dtype=np.int64)
        return (
            np.ascontiguousarray(
                self.scores[query_index, indices],
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                (query_index + indices) % 6,
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                (query_index + 2 * indices) % 6,
                dtype=np.int64,
            ),
        )


def _build_dino_commit(
    *,
    graph_directory: Path,
    output_root: Path,
    scores: np.ndarray,
) -> Path:
    matcher = _MatrixMatcher(scores)
    for rank in range(dino_edges.WORLD_SIZE):
        dino_edges.extract_rank(
            input_directory=graph_directory,
            output_root=output_root,
            rank=rank,
            block_size=4,
            audit_top_k=2,
            calibration_per_stratum=2,
            block_matcher=matcher,
        )
    dino_edges.finalize_shards(
        input_directory=graph_directory,
        output_root=output_root,
    )
    final = output_root / "final"
    dino_edges.validate_final(
        final,
        input_directory=graph_directory,
        output_root=output_root,
    )
    return final


def _iid_artifact(
    rows: list[dict[str, object]],
    scores: np.ndarray,
    graph_binding: dict[str, object],
) -> dict[str, object]:
    partials: list[dict[str, object]] = []
    assets = len(rows)
    for rank in range(8):
        accumulator = RankQuotientAccumulator(
            rows,
            graph_binding=graph_binding,
            rank=rank,
        )
        for asset_a in range(rank, assets, 8):
            for begin in range(asset_a + 1, assets, 4):
                stop = min(assets, begin + 4)
                candidates = np.arange(begin, stop, dtype=np.int64)
                accumulator.consume_block(
                    asset_a=asset_a,
                    candidate_indices=candidates,
                    scores=np.ascontiguousarray(
                        scores[asset_a, begin:stop],
                        dtype=np.float32,
                    ),
                    frames_a=np.ascontiguousarray(
                        (asset_a + candidates) % 6,
                        dtype=np.int64,
                    ),
                    frames_b=np.ascontiguousarray(
                        (asset_a + 2 * candidates) % 6,
                        dtype=np.int64,
                    ),
                )
        partials.append(accumulator.finalize())
    return merge_exact8_rank_partials(
        rows,
        partials,
        graph_binding=graph_binding,
    )


def _tree_files(directory: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for root, directory_names, file_names in os.walk(directory):
        directory_names.sort()
        file_names.sort()
        relative_root = Path(root).relative_to(directory)
        for name in file_names:
            path = Path(root) / name
            result[str(relative_root / name)] = path.read_bytes()
    return result


class QuotientEvidenceIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = _rows()
        self.graph_dir = self.root / "graph"
        _build_graph_commit(self.graph_dir, self.rows)
        graph = graph_input.validate_graph_input_commit(self.graph_dir)
        self.graph_binding = make_graph_commit_binding(
            artifact_digest=graph["artifact_digest"],
            artifact_hashes=graph["artifact_hashes"],
        )
        self.scores = _score_matrix(len(self.rows))
        self.dino_root = self.root / "dino"
        self.dino_final_dir = _build_dino_commit(
            graph_directory=self.graph_dir,
            output_root=self.dino_root,
            scores=self.scores,
        )
        validated_dino = dino_edges.validate_final(
            self.dino_final_dir,
            input_directory=self.graph_dir,
            output_root=self.dino_root,
        )
        self.iid_artifact = validated_dino["iid_pair_maxima"]
        self.iid_dir = self.root / "standalone-iid-maxima"
        publish_artifact_directory(
            self.iid_dir,
            self.iid_artifact,
            graph_binding=self.graph_binding,
            rows=self.rows,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, name: str) -> Path:
        output = self.root / name
        build_quotient_evidence(
            graph_input_dir=self.graph_dir,
            dino_final_dir=self.dino_final_dir,
            output_dir=output,
        )
        return output

    def test_deterministic_exact_closure_and_conservation(self) -> None:
        first = self._build("evidence-first")
        second = self._build("evidence-second")
        self.assertEqual(_tree_files(first), _tree_files(second))
        result = validate_quotient_evidence(
            first,
            graph_input_dir=self.graph_dir,
            dino_final_dir=self.dino_final_dir,
        )
        done = result["done"]
        base = result["base_components"]
        summary = result["summary"]
        self.assertEqual(set(entry.name for entry in first.iterdir()),
                         set(OUTPUT_ENTRIES))
        self.assertEqual(done["iids"], 6)
        self.assertEqual(done["iid_pairs"], 15)
        self.assertEqual(
            done["component_pairs"],
            done["base_components"]
            * (done["base_components"] - 1)
            // 2,
        )
        self.assertEqual(
            summary["calibration"]["strata"]
            and sum(
                row["N_h"]
                for row in summary["calibration"]["strata"]
            ),
            done["component_pairs"],
        )
        self.assertTrue(
            base["conservation"]["source_target_pair_atomic"]
        )
        self.assertFalse(done["training_authorized"])
        self.assertFalse(done["thresholds_human_calibrated"])
        artifact_permissions.assert_sealed_tree(first)
        self.assertEqual(
            stat.S_IMODE(first.stat().st_mode),
            0o555,
        )
        mode_probe = first / SUMMARY_NAME
        mode_probe.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "mode differs"):
            validate_quotient_evidence(
                first,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )
        mode_probe.chmod(0o444)
        resumed = build_quotient_evidence(
            graph_input_dir=self.graph_dir,
            dino_final_dir=self.dino_final_dir,
            output_dir=first,
            resume=True,
        )
        self.assertEqual(resumed, done)

    def test_base_components_are_pre_dino_only(self) -> None:
        output = self._build("pre-dino")
        base = json.loads(
            (output / BASE_COMPONENTS_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(base["algorithm"]["dino_input_edge_count"], 0)
        self.assertEqual(
            base["algorithm"]["dino_edges"],
            "forbidden-empty-tuple",
        )
        self.assertEqual(base["relation_counts"]["dino_cosine"], 0)
        self.assertEqual(
            base["indexed_candidate_relation_counts"]["dino_cosine"],
            0,
        )
        mapping = {
            row["iid"]: row["base_component"]
            for row in base["iid_to_base_component"]
        }
        self.assertEqual(mapping["iid-00"], mapping["iid-01"])
        self.assertEqual(mapping["iid-02"], mapping["iid-03"])
        # A 0.999 quotient score is evidence, never a base relation.
        self.assertNotEqual(mapping["iid-04"], mapping["iid-05"])
        with self.assertRaises(TypeError):
            build_quotient_evidence(
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
                iid_pair_maxima_dir=self.iid_dir,
                output_dir=self.root / "forbidden-argument",
            )

    def test_standalone_and_parent_iid_artifacts_cannot_bypass_dino(
        self,
    ) -> None:
        output = self._build("bound")
        alternate_artifact = _iid_artifact(
            self.rows,
            _score_matrix(len(self.rows), bias=0.02),
            self.graph_binding,
        )
        alternate_dir = self.root / "alternate-iid"
        publish_artifact_directory(
            alternate_dir,
            alternate_artifact,
            graph_binding=self.graph_binding,
            rows=self.rows,
        )
        # A standalone artifact, even with self-consistent logical/storage
        # digests, is not an accepted formal API input.
        with self.assertRaises(TypeError):
            build_quotient_evidence(
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
                iid_pair_maxima_dir=alternate_dir,
                output_dir=self.root / "standalone-must-not-build",
            )
        # Replacing the fixed final child by that internally valid artifact
        # is rejected when validate_final remerges the eight ranks.
        final_child = (
            self.dino_final_dir / dino_edges.IID_PAIR_MAXIMA_NAME
        )
        self.dino_final_dir.chmod(0o755)
        artifact_permissions.remove_staging_tree(final_child)
        artifact_permissions.make_staging_tree_removable(alternate_dir)
        os.rename(alternate_dir, final_child)
        artifact_permissions.seal_staging_tree(final_child)
        self.dino_final_dir.chmod(0o555)
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )

    def test_audit_reproduction_rejects_visual_and_archive_rebinding(
        self,
    ) -> None:
        visually_different_rows = copy.deepcopy(self.rows)
        visually_different_rows[0]["video_sha256"] = _sha(
            "audit-different-valid-video"
        )
        visually_different_rows[0]["dhashes"][0] = "fedcba9876543210"
        visual_graph = self.root / "visual-graph"
        _build_graph_commit(visual_graph, visually_different_rows)
        with self.assertRaisesRegex(ValueError, "binding|contract"):
            build_quotient_evidence(
                graph_input_dir=visual_graph,
                dino_final_dir=self.dino_final_dir,
                output_dir=self.root / "must-not-build-visual",
            )

        # Rows are byte-identical, but a different committed feature archive
        # is a distinct validated graph commit and cannot reuse the maxima.
        archive_graph = self.root / "archive-graph"
        _build_graph_commit(
            archive_graph,
            copy.deepcopy(self.rows),
            feature_seed=31,
        )
        original = graph_input.validate_graph_input_commit(self.graph_dir)
        alternate = graph_input.validate_graph_input_commit(archive_graph)
        self.assertEqual(original["rows"], alternate["rows"])
        self.assertNotEqual(
            original["artifact_hashes"]["archive"],
            alternate["artifact_hashes"]["archive"],
        )
        with self.assertRaisesRegex(ValueError, "binding|contract"):
            build_quotient_evidence(
                graph_input_dir=archive_graph,
                dino_final_dir=self.dino_final_dir,
                output_dir=self.root / "must-not-build-archive",
            )

    def test_dino_final_parent_and_rank_tamper_are_rejected(self) -> None:
        output = self._build("dino-bound")
        parent_copy = self.root / "dino-parent-tamper"
        rank_copy = self.root / "dino-rank-tamper"
        shutil.copytree(self.dino_root, parent_copy)
        shutil.copytree(self.dino_root, rank_copy)

        final_done = parent_copy / "final" / dino_edges.DONE_NAME
        final_done.chmod(0o644)
        final_done.write_bytes(final_done.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=parent_copy / "final",
            )

        rank_summary = (
            rank_copy
            / "shards"
            / "rank-00000-of-00008"
            / dino_edges.SUMMARY_NAME
        )
        rank_summary.chmod(0o644)
        rank_summary.write_bytes(rank_summary.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=rank_copy / "final",
            )

    def test_mapping_relation_and_directory_tamper_fail_closed(self) -> None:
        mapping_output = self._build("mapping-tamper")
        base_path = mapping_output / BASE_COMPONENTS_NAME
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base["iid_to_base_component"][0]["base_component"] = "forged"
        base_path.chmod(0o644)
        base_path.write_text(
            json.dumps(
                base,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                mapping_output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )

        relation_output = self._build("relation-tamper")
        relation_path = relation_output / BASE_COMPONENTS_NAME
        relation = json.loads(
            relation_path.read_text(encoding="utf-8")
        )
        relation["relation_counts"]["dino_cosine"] = 1
        relation_path.chmod(0o644)
        relation_path.write_text(
            json.dumps(
                relation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                relation_output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )

        extra_output = self._build("extra")
        extra_output.chmod(0o755)
        (extra_output / "unexpected").write_text(
            "not in closure",
            encoding="utf-8",
        )
        (extra_output / "unexpected").chmod(0o444)
        extra_output.chmod(0o555)
        with self.assertRaisesRegex(ValueError, "closure"):
            validate_quotient_evidence(
                extra_output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )

    def test_subartifact_tamper_and_type_changes_fail_closed(self) -> None:
        component_output = self._build("component-tamper")
        arrays_path = (
            component_output
            / COMPONENT_MAXIMA_DIR
            / "arrays.npz"
        )
        arrays_path.chmod(0o644)
        payload = bytearray(arrays_path.read_bytes())
        payload[len(payload) // 2] ^= 1
        arrays_path.write_bytes(payload)
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                component_output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )

        sample_output = self._build("sample-tamper")
        metadata = (
            sample_output
            / CALIBRATION_SAMPLE_DIR
            / "metadata.json"
        )
        external = self.root / "external-metadata"
        external.write_bytes(metadata.read_bytes())
        metadata.parent.chmod(0o755)
        metadata.unlink()
        metadata.symlink_to(external)
        metadata.parent.chmod(0o555)
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                sample_output,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )

        root_type = self._build("root-type")
        done_path = root_type / DONE_NAME
        root_type.chmod(0o755)
        done_path.unlink()
        done_path.mkdir()
        root_type.chmod(0o555)
        with self.assertRaises(ValueError):
            validate_quotient_evidence(
                root_type,
                graph_input_dir=self.graph_dir,
                dino_final_dir=self.dino_final_dir,
            )


if __name__ == "__main__":
    unittest.main()
