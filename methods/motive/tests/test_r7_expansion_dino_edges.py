from __future__ import annotations

import ast
import hashlib
import inspect
import json
import stat
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motive import r7_artifact_permissions as artifact_permissions
from motive import r7_dino_quotient_calibration as quotient_calibration
from motive import r7_visual_graph_input as visual_graph_input
from motive.r7_expansion_dino_edges import (
    AUDIT_THRESHOLD,
    CALIBRATION_EDGES_NAME,
    CALIBRATION_SAMPLING_SEED,
    DINO_DIM,
    DINO_FRAMES,
    HARD_THRESHOLD,
    QUOTIENT_RANK_PARTIAL_NAME,
    WORLD_SIZE,
    _algorithm_contract,
    _calibration_hash_priority,
    _load_quotient_artifact,
    _pair_identifier,
    _score_stratum_index,
    extract_rank,
    finalize_shards,
    match_rank_arrays,
    numpy_block_matcher,
    validate_final,
    validate_graph_input,
    validate_shard,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _graph_binding(label: str = "test-graph") -> dict[str, object]:
    return quotient_calibration.make_graph_commit_binding(
        artifact_digest=_sha(f"{label}:artifact"),
        artifact_hashes={
            name: _sha(f"{label}:{name}")
            for name in ("manifest", "archive", "summary", "done")
        },
    )


def _encoder() -> dict[str, object]:
    return {
        "encoder_id": "facebook/dinov2-base",
        "encoder_revision": "a" * 40,
        "model_tree_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "model_file_count": 14,
        "embedding_dim": DINO_DIM,
        "dtype": "float32",
        "normalization": "l2-per-frame",
        "frozen_encoder": True,
        "local_files_only": True,
        "frame_sampling_version": "uniform-6-from-uniform-32-v1",
        "preprocessing_version":
            "transformers-auto-image-processor-local-v1",
        "pooling": "last-hidden-state-cls-token-v1",
    }


def _rows(count: int) -> list[dict[str, object]]:
    if count <= 0 or count % 2:
        raise ValueError("test graph rows must be non-empty pairs")
    rows: list[dict[str, object]] = []
    for index in range(count):
        pair_index = index // 2
        anchor = pair_index == 0
        rows.append(
            {
                "schema_version": visual_graph_input.ROW_SCHEMA,
                "asset_index": index,
                "iid": f"iid-{pair_index:03d}",
                "role": "source" if index % 2 == 0 else "target",
                "anchor": anchor,
                "cohort": (
                    "anchor_positive"
                    if anchor
                    else (
                        "pseudo_positive"
                        if pair_index % 2 == 0
                        else "pseudo_negative"
                    )
                ),
                "video_sha256": _sha(f"video:{index}"),
                "dhashes": [
                    f"{index * DINO_FRAMES + frame:016x}"
                    for frame in range(DINO_FRAMES)
                ],
                "source_artifact_digest": (
                    "a" * 64 if anchor else "b" * 64
                ),
                "source_input_index": (
                    0 if anchor else pair_index - 1
                ),
                "source_index_digest": _sha(
                    f"source-index:{index}"
                ),
            }
        )
    return rows


def _features(count: int) -> np.ndarray:
    generator = np.random.default_rng(19)
    value = generator.normal(
        size=(count, DINO_FRAMES, DINO_DIM)
    ).astype(np.float32)
    value /= np.linalg.norm(value, axis=2, keepdims=True)
    return value.astype(np.float32)


def _build_external_graph_commit(
    directory: Path,
    *,
    rows: list[dict[str, object]],
    features: np.ndarray,
) -> None:
    candidate_iids = sorted(
        {
            str(row["iid"])
            for row in rows
            if row["anchor"] is False
        }
    )
    anchor_iids = sorted(
        {
            str(row["iid"])
            for row in rows
            if row["anchor"] is True
        }
    )
    arrays = {
        "asset_indices": np.arange(len(rows), dtype=np.int64),
        "dino_cls": np.ascontiguousarray(features),
    }
    summary_base = {
        "schema_version": visual_graph_input.SUMMARY_SCHEMA,
        "status": "complete",
        "assets": len(rows),
        "iids": len(rows) // 2,
        "candidate_iids": {
            "count": len(candidate_iids),
            "sha256":
                visual_graph_input._object_digest(candidate_iids),
        },
        "anchor_iids": {
            "count": len(anchor_iids),
            "sha256": visual_graph_input._object_digest(anchor_iids),
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
    derived = visual_graph_input._Derived(
        rows=tuple(rows),
        arrays=arrays,
        summary_base=summary_base,
        paths={},
        snapshot={},
    )
    visual_graph_input._atomic_publish(
        directory=directory,
        derived=derived,
        pre_publish_check=lambda: None,
    )
    visual_graph_input.validate_graph_input_commit(directory)


def _rebind_external_commit(directory: Path) -> None:
    manifest = directory / visual_graph_input.MANIFEST_NAME
    archive = directory / visual_graph_input.ARCHIVE_NAME
    summary_path = directory / visual_graph_input.SUMMARY_NAME
    done_path = directory / visual_graph_input.DONE_NAME

    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manifest_sha256"] = file_sha(manifest)
    summary["archive_sha256"] = file_sha(archive)
    summary_path.write_bytes(
        visual_graph_input._pretty_json_bytes(summary)
    )
    output_sha = {
        "manifest": file_sha(manifest),
        "archive": file_sha(archive),
        "summary": file_sha(summary_path),
    }
    done = json.loads(done_path.read_text(encoding="utf-8"))
    for name, digest in output_sha.items():
        done["artifacts"][name]["sha256"] = digest
    done["artifact_digest"] = visual_graph_input._object_digest(output_sha)
    done_path.write_bytes(visual_graph_input._pretty_json_bytes(done))


def _rebind_dino_output(directory: Path) -> None:
    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    summary_path = directory / "summary.json"
    done_path = directory / "done.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for field, filename in (
        ("hard_edges_sha256", "hard_edges.jsonl"),
        ("audit_edges_sha256", "audit_edges.jsonl"),
        ("calibration_edges_sha256", CALIBRATION_EDGES_NAME),
    ):
        summary[field] = file_sha(directory / filename)
    summary_path.chmod(0o644)
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.chmod(0o444)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    for name, filename in (
        ("hard_edges", "hard_edges.jsonl"),
        ("audit_edges", "audit_edges.jsonl"),
        ("calibration_edges", CALIBRATION_EDGES_NAME),
        ("summary", "summary.json"),
    ):
        done["artifacts"][name]["sha256"] = file_sha(
            directory / filename
        )
    done_path.chmod(0o644)
    done_path.write_text(
        json.dumps(
            done,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    done_path.chmod(0o444)


class MatrixMatcher:
    def __init__(self, scores: np.ndarray) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls = 0

    def __call__(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        query_index: int,
        candidate_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del query, candidates
        self.calls += 1
        values = self.scores[query_index, candidate_indices]
        frames_a = (
            np.asarray(candidate_indices, dtype=np.int64) + query_index
        ) % DINO_FRAMES
        frames_b = (
            2 * np.asarray(candidate_indices, dtype=np.int64) + query_index
        ) % DINO_FRAMES
        return values, frames_a, frames_b


class DinoEdgeMatcherTests(unittest.TestCase):
    def test_calibration_strata_are_fixed_nonoverlapping_and_complete(
        self,
    ) -> None:
        algorithm = _algorithm_contract(
            block_size=3,
            audit_top_k=2,
            calibration_per_stratum=7,
        )
        score_strata = algorithm["calibration_score_strata"]
        self.assertEqual(score_strata[0]["lower"], -1.0)
        self.assertEqual(score_strata[-1]["upper"], 1.0)
        self.assertEqual(score_strata[-1]["upper_operator"], "<=")
        for left, right in zip(score_strata, score_strata[1:]):
            self.assertEqual(left["upper_operator"], "<")
            self.assertEqual(left["upper"], right["lower"])
        sampling_strata = algorithm["calibration_sampling_strata"]
        self.assertEqual(len(sampling_strata), 2 * len(score_strata))
        self.assertEqual(
            {
                stratum["pair_relation"]
                for stratum in sampling_strata
            },
            {"same_iid", "cross_iid"},
        )
        self.assertEqual(
            algorithm["calibration_statistical_unit"],
            "asset_pair",
        )
        self.assertEqual(
            algorithm["calibration_intended_use"],
            "diagnostic_not_threshold_calibrating",
        )
        self.assertTrue(algorithm["quotient_accumulation_enabled"])
        self.assertFalse(algorithm["quotient_accumulation_optional"])
        self.assertEqual(
            algorithm["quotient_partials_per_iid_pair"],
            2,
        )
        self.assertEqual(
            algorithm["quotient_graph_commit_binding_schema"],
            quotient_calibration.GRAPH_COMMIT_BINDING_SCHEMA,
        )
        self.assertEqual(3348 * 3347 // 2, 5_602_878)
        self.assertFalse(algorithm["thresholds_human_calibrated"])
        self.assertFalse(algorithm["calibration_training_authorized"])

    def _input(
        self,
        root: Path,
        *,
        count: int = 12,
    ) -> tuple[Path, list[dict[str, object]], np.ndarray]:
        rows = _rows(count)
        features = _features(count)
        input_directory = root / "graph-input"
        _build_external_graph_commit(
            input_directory,
            rows=rows,
            features=features,
        )
        return input_directory, rows, features

    def _score_matrix(self, count: int) -> np.ndarray:
        scores = np.zeros((count, count), dtype=np.float32)
        assignments = {
            (0, 1): 0.97,
            (0, 2): 0.95,
            (0, 3): 0.94,
            (0, 4): 0.93,
            (0, 9): 0.945,
            (1, 9): 0.95,
            (2, 9): 0.949,
            (3, 8): 0.98,
            (7, 11): 0.921,
        }
        for (a, b), value in assignments.items():
            if b < count:
                scores[a, b] = np.float32(value)
                scores[b, a] = np.float32(value)
        return scores

    def _extract_all(
        self,
        *,
        input_directory: Path,
        output_root: Path,
        scores: np.ndarray,
        order: list[int],
        top_k: int = 2,
        calibration_per_stratum: int = 2,
    ) -> None:
        matcher = MatrixMatcher(scores)
        for rank in order:
            extract_rank(
                input_directory=input_directory,
                output_root=output_root,
                rank=rank,
                world_size=WORLD_SIZE,
                block_size=3,
                audit_top_k=top_k,
                calibration_per_stratum=calibration_per_stratum,
                block_matcher=matcher,
            )

    def test_exact_eight_rank_merge_hard_exhaustive_and_order_invariant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory, rows, _ = self._input(root)
            scores = self._score_matrix(len(rows))
            forward = root / "forward"
            reverse = root / "reverse"
            self._extract_all(
                input_directory=input_directory,
                output_root=forward,
                scores=scores,
                order=list(range(WORLD_SIZE)),
            )
            self._extract_all(
                input_directory=input_directory,
                output_root=reverse,
                scores=scores,
                order=list(reversed(range(WORLD_SIZE))),
            )
            finalize_shards(
                input_directory=input_directory,
                output_root=forward,
            )
            finalize_shards(
                input_directory=input_directory,
                output_root=reverse,
            )
            first = validate_final(
                forward / "final",
                input_directory=input_directory,
                output_root=forward,
            )
            second = validate_final(
                reverse / "final",
                input_directory=input_directory,
                output_root=reverse,
            )
            for output_root in (forward, reverse):
                artifact_permissions.assert_sealed_tree(
                    output_root / "final"
                )
                for rank in range(WORLD_SIZE):
                    artifact_permissions.assert_sealed_tree(
                        output_root
                        / "shards"
                        / f"rank-{rank:05d}-of-{WORLD_SIZE:05d}"
                    )
            self.assertEqual(
                stat.S_IMODE(
                    (forward / "final").stat().st_mode
                ),
                0o555,
            )
            graph_commit = (
                visual_graph_input.validate_graph_input_commit(
                    input_directory
                )
            )
            self.assertEqual(
                first["contract"]["input_artifact_digest"],
                graph_commit["done"]["artifact_digest"],
            )
            self.assertEqual(
                first["contract"]["input_artifacts"],
                graph_commit["artifact_hashes"],
            )
            self.assertEqual(
                first["contract"]["dino_contract"],
                graph_commit["summary"]["dino_contract"],
            )
            expected_hard = {
                (a, b)
                for a in range(len(rows))
                for b in range(a + 1, len(rows))
                if scores[a, b] >= np.float32(HARD_THRESHOLD)
            }
            self.assertEqual(
                {
                    (row["asset_a"], row["asset_b"])
                    for row in first["hard_edges"]
                },
                expected_hard,
            )
            self.assertEqual(
                first["hard_edges"],
                second["hard_edges"],
            )
            self.assertEqual(
                first["audit_edges"],
                second["audit_edges"],
            )
            self.assertEqual(
                first["calibration_edges"],
                second["calibration_edges"],
            )
            self.assertEqual(
                first["iid_pair_maxima"]["artifact_digest"],
                second["iid_pair_maxima"]["artifact_digest"],
            )
            self.assertEqual(
                (forward / "final" / "hard_edges.jsonl").read_bytes(),
                (reverse / "final" / "hard_edges.jsonl").read_bytes(),
            )
            self.assertEqual(
                (forward / "final" / "audit_edges.jsonl").read_bytes(),
                (reverse / "final" / "audit_edges.jsonl").read_bytes(),
            )
            self.assertEqual(
                (
                    forward / "final" / CALIBRATION_EDGES_NAME
                ).read_bytes(),
                (
                    reverse / "final" / CALIBRATION_EDGES_NAME
                ).read_bytes(),
            )
            self.assertEqual(
                first["summary"]["coverage_proof"][
                    "observed_compared_pairs"
                ],
                len(rows) * (len(rows) - 1) // 2,
            )
            self.assertTrue(
                first["summary"]["coverage_proof"][
                    "complete_upper_triangle"
                ]
            )
            self.assertTrue(
                all(
                    row["hard_edge"] is False
                    and float(row["cosine"]) < HARD_THRESHOLD
                    for row in first["audit_edges"]
                )
            )
            selections = {
                (row["asset_a"], row["asset_b"]):
                row["selected_for_asset_indices"]
                for row in first["audit_edges"]
            }
            # (0,4) survives because asset 4 selected it, but it is not in
            # asset 0's two strongest calibration neighbors.
            self.assertEqual(selections[(0, 4)], [4])
            # Candidates incident to asset 9 were owned by three ranks.
            self.assertNotIn(9, selections[(0, 9)])
            self.assertIn(9, selections[(1, 9)])
            self.assertIn(9, selections[(2, 9)])
            calibration = first["summary"]["calibration_sampling"]
            self.assertEqual(
                calibration["population_count"],
                len(rows) * (len(rows) - 1) // 2,
            )
            self.assertEqual(
                calibration["sampling_seed"],
                CALIBRATION_SAMPLING_SEED,
            )
            self.assertEqual(
                calibration["input_artifact_digest"],
                graph_commit["artifact_digest"],
            )
            self.assertEqual(
                first["summary"]["score_histogram"][
                    "population_count"
                ],
                calibration["population_count"],
            )
            histogram = first["summary"]["score_histogram"]["counts"]
            self.assertEqual(sum(histogram["same_iid"]), len(rows) // 2)
            self.assertEqual(
                sum(histogram["cross_iid"]),
                len(rows) * (len(rows) - 1) // 2 - len(rows) // 2,
            )
            self.assertEqual(
                [
                    row["pair_relation"]
                    for row in first["calibration_edges"]
                    if row["pair_relation"] == "same_iid"
                ],
                ["same_iid"]
                * sum(
                    stratum["n_h"]
                    for stratum in calibration["strata"]
                    if stratum["pair_relation"] == "same_iid"
                ),
            )
            for name in (
                quotient_calibration.ARTIFACT_METADATA_NAME,
                quotient_calibration.ARTIFACT_ARRAYS_NAME,
                quotient_calibration.ARTIFACT_DONE_NAME,
            ):
                self.assertEqual(
                    (
                        forward
                        / "final"
                        / "iid_pair_maxima"
                        / name
                    ).read_bytes(),
                    (
                        reverse
                        / "final"
                        / "iid_pair_maxima"
                        / name
                    ).read_bytes(),
                )
            self.assertTrue(
                all(
                    row["training_authorized"] is False
                    and row["human_labels_asserted"] is False
                    for row in first["calibration_edges"]
                )
            )
            self.assertEqual(
                first["done"]["calibration_intended_use"],
                "diagnostic_not_threshold_calibrating",
            )

            expected_bottom_k: dict[int, list[tuple[int, int]]] = {}
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    relation_index = (
                        0 if rows[a]["iid"] == rows[b]["iid"] else 1
                    )
                    stratum_index = (
                        2 * _score_stratum_index(scores[a, b])
                        + relation_index
                    )
                    _pair_id, pair_digest = _pair_identifier(a, b)
                    priority = _calibration_hash_priority(
                        pair_id_sha256=pair_digest,
                        population_digest=graph_commit["artifact_digest"],
                    )
                    expected_bottom_k.setdefault(
                        stratum_index,
                        [],
                    ).append((priority, a, b))
            expected_pairs = {
                (index, a, b)
                for index, candidates in expected_bottom_k.items()
                for _priority, a, b in sorted(candidates)[:2]
            }
            observed_pairs = {
                (
                    row["sampling_stratum_index"],
                    row["asset_a"],
                    row["asset_b"],
                )
                for row in first["calibration_edges"]
            }
            self.assertEqual(observed_pairs, expected_pairs)
            _, pair_digest = _pair_identifier(0, 1)
            self.assertNotEqual(
                _calibration_hash_priority(
                    pair_id_sha256=pair_digest,
                    population_digest="a" * 64,
                ),
                _calibration_hash_priority(
                    pair_id_sha256=pair_digest,
                    population_digest="b" * 64,
                ),
            )
            iid_maxima = first["iid_pair_maxima"]
            expected_iid_pairs = (
                len(rows) // 2 * (len(rows) // 2 - 1) // 2
            )
            self.assertEqual(
                len(iid_maxima["arrays"]["score"]),
                expected_iid_pairs,
            )
            self.assertEqual(
                iid_maxima["contract"]["observed_iid_pairs"],
                expected_iid_pairs,
            )
            self.assertEqual(
                iid_maxima["contract"]["partials_per_iid_pair"],
                2,
            )
            self.assertEqual(
                iid_maxima["contract"]["observed_partials"],
                2 * expected_iid_pairs,
            )
            observed_iid_maxima = {
                (
                    int(iid_maxima["arrays"]["iid_a"][index]),
                    int(iid_maxima["arrays"]["iid_b"][index]),
                ): (
                    iid_maxima["arrays"]["score"][index],
                    (
                        int(iid_maxima["arrays"]["asset_a"][index]),
                        int(iid_maxima["arrays"]["asset_b"][index]),
                        int(iid_maxima["arrays"]["frame_a"][index]),
                        int(iid_maxima["arrays"]["frame_b"][index]),
                    ),
                )
                for index in range(expected_iid_pairs)
            }
            for iid_a in range(len(rows) // 2):
                for iid_b in range(iid_a + 1, len(rows) // 2):
                    candidates = []
                    for asset_a in (2 * iid_a, 2 * iid_a + 1):
                        for asset_b in (2 * iid_b, 2 * iid_b + 1):
                            candidates.append(
                                (
                                    scores[asset_a, asset_b],
                                    (
                                        asset_a,
                                        asset_b,
                                        (asset_a + asset_b) % DINO_FRAMES,
                                        (
                                            2 * asset_b + asset_a
                                        )
                                        % DINO_FRAMES,
                                    ),
                                )
                            )
                    maximum = max(item[0] for item in candidates)
                    witness = min(
                        item[1]
                        for item in candidates
                        if item[0] == maximum
                    )
                    observed_score, observed_witness = (
                        observed_iid_maxima[(iid_a, iid_b)]
                    )
                    self.assertEqual(
                        observed_score.tobytes(),
                        np.float32(maximum).tobytes(),
                    )
                    self.assertEqual(observed_witness, witness)

    def test_calibration_bottom_k_is_block_partition_invariant(
        self,
    ) -> None:
        rows = _rows(12)
        features = _features(12)
        scores = self._score_matrix(12)
        observed = []
        for block_size in (2, 5):
            observed.append(
                match_rank_arrays(
                    rows=rows,
                    dino_cls=features,
                    graph_binding=_graph_binding(),
                    rank=0,
                    world_size=WORLD_SIZE,
                    block_size=block_size,
                    audit_top_k=2,
                    calibration_per_stratum=2,
                    calibration_population_digest="d" * 64,
                    block_matcher=MatrixMatcher(scores),
                )
            )
        self.assertEqual(
            observed[0]["calibration_edges"],
            observed[1]["calibration_edges"],
        )
        self.assertEqual(
            observed[0]["calibration_sampling"],
            observed[1]["calibration_sampling"],
        )
        self.assertEqual(
            observed[0]["score_histogram"],
            observed[1]["score_histogram"],
        )
        self.assertEqual(
            observed[0]["quotient_rank_partial"]["artifact_digest"],
            observed[1]["quotient_rank_partial"]["artifact_digest"],
        )

    def test_quotient_uses_one_accumulator_and_no_extra_matcher_calls(
        self,
    ) -> None:
        rows = _rows(12)
        features = _features(12)
        matcher = MatrixMatcher(self._score_matrix(12))
        real_accumulator = quotient_calibration.RankQuotientAccumulator
        with patch.object(
            quotient_calibration,
            "RankQuotientAccumulator",
            wraps=real_accumulator,
        ) as constructor:
            result = match_rank_arrays(
                rows=rows,
                dino_cls=features,
                graph_binding=_graph_binding(),
                rank=0,
                world_size=WORLD_SIZE,
                block_size=3,
                audit_top_k=2,
                calibration_per_stratum=2,
                calibration_population_digest="e" * 64,
                block_matcher=matcher,
            )
        self.assertEqual(constructor.call_count, 1)
        expected_matcher_calls = sum(
            (len(rows) - asset - 1 + 2) // 3
            for asset in range(0, len(rows), WORLD_SIZE)
        )
        self.assertEqual(matcher.calls, expected_matcher_calls)
        expected_partials = sum(
            len(rows) // 2 - asset // 2 - 1
            for asset in range(0, len(rows), WORLD_SIZE)
        )
        self.assertEqual(
            len(result["quotient_rank_partial"]["arrays"]["score"]),
            expected_partials,
        )

    def test_quotient_consume_is_immediately_after_normalisation(
        self,
    ) -> None:
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(match_rank_arrays))
        )
        matches: list[tuple[list[ast.stmt], int]] = []
        for node in ast.walk(tree):
            for _unused_field, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                for index, statement in enumerate(value):
                    call = (
                        statement.value
                        if isinstance(statement, ast.Assign)
                        else None
                    )
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "_normalise_matcher_output"
                    ):
                        matches.append((value, index))
        self.assertEqual(len(matches), 1)
        statements, index = matches[0]
        self.assertLess(index + 1, len(statements))
        next_statement = statements[index + 1]
        self.assertIsInstance(next_statement, ast.Expr)
        consume = next_statement.value
        self.assertIsInstance(consume, ast.Call)
        self.assertIsInstance(consume.func, ast.Attribute)
        self.assertEqual(consume.func.attr, "consume_block")
        self.assertIsInstance(consume.func.value, ast.Name)
        self.assertEqual(
            consume.func.value.id,
            "quotient_accumulator",
        )

    def test_quotient_rejects_same_rows_from_different_graph_commit(
        self,
    ) -> None:
        rows = _rows(12)
        features = _features(12)
        first_binding = _graph_binding("first")
        second_binding = _graph_binding("second")
        result = match_rank_arrays(
            rows=rows,
            dino_cls=features,
            graph_binding=first_binding,
            rank=0,
            world_size=WORLD_SIZE,
            block_size=3,
            audit_top_k=2,
            calibration_per_stratum=2,
            calibration_population_digest="f" * 64,
            block_matcher=MatrixMatcher(self._score_matrix(12)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = (
                Path(temporary) / QUOTIENT_RANK_PARTIAL_NAME
            )
            quotient_calibration.publish_artifact_directory(
                directory,
                result["quotient_rank_partial"],
                graph_binding=first_binding,
                rows=rows,
            )
            loaded, unused_binding = _load_quotient_artifact(
                directory,
                graph_rows=rows,
                graph_binding=first_binding,
                expected_schema=quotient_calibration.RANK_PARTIAL_SCHEMA,
            )
            del unused_binding
            self.assertEqual(
                loaded["artifact_digest"],
                result["quotient_rank_partial"]["artifact_digest"],
            )
            with self.assertRaisesRegex(
                ValueError,
                "graph commit binding differs",
            ):
                _load_quotient_artifact(
                    directory,
                    graph_rows=rows,
                    graph_binding=second_binding,
                    expected_schema=(
                        quotient_calibration.RANK_PARTIAL_SCHEMA
                    ),
                )

    def test_exact_float32_threshold_boundaries(self) -> None:
        rows = _rows(6)
        features = _features(6)
        scores = np.zeros((6, 6), dtype=np.float32)
        hard = np.float32(HARD_THRESHOLD)
        audit = np.float32(AUDIT_THRESHOLD)
        below_hard = np.nextafter(hard, np.float32(-np.inf))
        below_audit = np.nextafter(audit, np.float32(-np.inf))
        values = [hard, below_hard, audit, below_audit]
        for asset_b, value in enumerate(values, start=1):
            scores[0, asset_b] = value
            scores[asset_b, 0] = value
        result = match_rank_arrays(
            rows=rows,
            dino_cls=features,
            graph_binding=_graph_binding(),
            rank=0,
            world_size=WORLD_SIZE,
            block_size=2,
            audit_top_k=10,
            block_matcher=MatrixMatcher(scores),
        )
        self.assertEqual(
            {
                (row["asset_a"], row["asset_b"])
                for row in result["hard_edges"]
            },
            {(0, 1)},
        )
        self.assertEqual(
            {
                (row["asset_a"], row["asset_b"])
                for row in result["audit_edges"]
            },
            {(0, 2), (0, 3)},
        )
        calibration = {
            (row["asset_a"], row["asset_b"]): row
            for row in result["calibration_edges"]
        }
        self.assertEqual(
            calibration[(0, 1)]["score_stratum"],
            "hard",
        )
        self.assertEqual(
            calibration[(0, 2)]["score_stratum"],
            "audit_upper",
        )
        self.assertEqual(
            calibration[(0, 3)]["score_stratum"],
            "audit_lower",
        )
        self.assertEqual(
            calibration[(0, 4)]["score_stratum"],
            "near_audit_complement",
        )
        self.assertEqual(
            calibration[(0, 1)]["pair_relation"],
            "same_iid",
        )
        self.assertEqual(
            calibration[(0, 2)]["pair_relation"],
            "cross_iid",
        )
        semantic = result["semantic_band_counts"]["all"]
        self.assertEqual(semantic["below_0p92"], 2)
        self.assertEqual(
            semantic["at_least_0p92_below_0p96"],
            2,
        )
        self.assertEqual(semantic["at_least_0p96"], 1)

    def test_numpy_matcher_hard_edges_equal_full_bruteforce(self) -> None:
        rows = _rows(10)
        features = _features(10)
        features[1] = features[0]
        features[7] = features[3]
        observed: set[tuple[int, int]] = set()
        for rank in range(WORLD_SIZE):
            result = match_rank_arrays(
                rows=rows,
                dino_cls=features,
                graph_binding=_graph_binding(),
                rank=rank,
                world_size=WORLD_SIZE,
                block_size=4,
                audit_top_k=3,
                block_matcher=numpy_block_matcher,
            )
            observed.update(
                (row["asset_a"], row["asset_b"])
                for row in result["hard_edges"]
            )
        expected: set[tuple[int, int]] = set()
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                matrix = np.einsum(
                    "fd,gd->fg",
                    features[a],
                    features[b],
                    dtype=np.float32,
                    optimize=False,
                )
                if np.max(matrix) >= np.float32(HARD_THRESHOLD):
                    expected.add((a, b))
        self.assertEqual(observed, expected)
        self.assertEqual(expected, {(0, 1), (3, 7)})

    def test_external_row_schema_mismatch_fails_even_when_rehashed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory, rows, _ = self._input(root)
            manifest = input_directory / visual_graph_input.MANIFEST_NAME
            committed_rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            committed_rows[0]["schema_version"] = "wrong-row-schema"
            manifest.write_text(
                "".join(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                    for row in committed_rows
                ),
                encoding="utf-8",
            )
            _rebind_external_commit(input_directory)
            with self.assertRaisesRegex(
                ValueError,
                "pair/schema binding",
            ):
                validate_graph_input(input_directory)
            with self.assertRaises(ValueError):
                extract_rank(
                    input_directory=input_directory,
                    output_root=root / "must-not-exist",
                    rank=0,
                    block_size=3,
                    audit_top_k=2,
                    block_matcher=MatrixMatcher(
                        self._score_matrix(len(rows))
                    ),
                )

    def test_resume_no_overwrite_tamper_and_partial_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory, rows, _ = self._input(root)
            scores = self._score_matrix(len(rows))
            output = root / "output"
            matcher = MatrixMatcher(scores)
            extract_rank(
                input_directory=input_directory,
                output_root=output,
                rank=0,
                block_size=3,
                audit_top_k=2,
                block_matcher=matcher,
            )
            with self.assertRaises(FileExistsError):
                extract_rank(
                    input_directory=input_directory,
                    output_root=output,
                    rank=0,
                    block_size=3,
                    audit_top_k=2,
                    block_matcher=matcher,
                )
            extract_rank(
                input_directory=input_directory,
                output_root=output,
                rank=0,
                block_size=3,
                audit_top_k=2,
                block_matcher=matcher,
                resume=True,
            )
            shard_directory = (
                output / "shards" / "rank-00000-of-00008"
            )
            shard_hard_mode_probe = (
                shard_directory / "hard_edges.jsonl"
            )
            shard_hard_mode_probe.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "mode differs"):
                validate_shard(
                    shard_directory,
                    input_directory=input_directory,
                )
            shard_hard_mode_probe.chmod(0o444)
            partial_arrays = (
                output
                / "shards"
                / "rank-00000-of-00008"
                / "quotient_rank_partial"
                / quotient_calibration.ARTIFACT_ARRAYS_NAME
            )
            original_partial_arrays = partial_arrays.read_bytes()
            partial_arrays.chmod(0o644)
            payload = bytearray(original_partial_arrays)
            payload[len(payload) // 2] ^= 1
            partial_arrays.write_bytes(bytes(payload))
            with self.assertRaises(ValueError):
                extract_rank(
                    input_directory=input_directory,
                    output_root=output,
                    rank=0,
                    block_size=3,
                    audit_top_k=2,
                    block_matcher=matcher,
                    resume=True,
                )
            partial_arrays.write_bytes(original_partial_arrays)
            partial_arrays.chmod(0o444)
            with self.assertRaises((FileNotFoundError, ValueError)):
                finalize_shards(
                    input_directory=input_directory,
                    output_root=output,
                )
            for rank in range(1, WORLD_SIZE):
                extract_rank(
                    input_directory=input_directory,
                    output_root=output,
                    rank=rank,
                    block_size=3,
                    audit_top_k=2,
                    block_matcher=matcher,
                )
            finalize_shards(
                input_directory=input_directory,
                output_root=output,
            )
            with self.assertRaises(FileExistsError):
                finalize_shards(
                    input_directory=input_directory,
                    output_root=output,
                )
            finalize_shards(
                input_directory=input_directory,
                output_root=output,
                resume=True,
            )
            final_quotient = output / "final" / "iid_pair_maxima"
            unexpected = final_quotient / "unexpected"
            final_quotient.chmod(0o755)
            unexpected.write_text("tamper\n", encoding="utf-8")
            unexpected.chmod(0o444)
            final_quotient.chmod(0o555)
            with self.assertRaises(ValueError):
                validate_final(
                    output / "final",
                    input_directory=input_directory,
                    output_root=output,
                )
            final_quotient.chmod(0o755)
            unexpected.unlink()
            final_quotient.chmod(0o555)
            calibration_path = (
                output / "final" / CALIBRATION_EDGES_NAME
            )
            original_calibration = calibration_path.read_bytes()
            original_summary = (
                output / "final" / "summary.json"
            ).read_bytes()
            original_done = (
                output / "final" / "done.json"
            ).read_bytes()
            calibration_rows = [
                json.loads(line)
                for line in original_calibration.decode(
                    "utf-8"
                ).splitlines()
            ]
            calibration_rows[0]["hash_priority_sha256"] = "0" * 64
            calibration_path.chmod(0o644)
            calibration_path.write_text(
                "".join(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                    for row in calibration_rows
                ),
                encoding="utf-8",
            )
            _rebind_dino_output(calibration_path.parent)
            calibration_path.chmod(0o444)
            with self.assertRaises(ValueError):
                validate_final(
                    calibration_path.parent,
                    input_directory=input_directory,
                    output_root=output,
                )
            calibration_path.chmod(0o644)
            calibration_path.write_bytes(original_calibration)
            calibration_path.chmod(0o444)
            final_summary = output / "final" / "summary.json"
            final_done = output / "final" / "done.json"
            final_summary.chmod(0o644)
            final_summary.write_bytes(original_summary)
            final_summary.chmod(0o444)
            final_done.chmod(0o644)
            final_done.write_bytes(original_done)
            final_done.chmod(0o444)
            shard_hard = (
                output
                / "shards"
                / "rank-00000-of-00008"
                / "hard_edges.jsonl"
            )
            shard_hard.chmod(0o644)
            shard_hard.write_bytes(shard_hard.read_bytes() + b"\n")
            shard_hard.chmod(0o444)
            with self.assertRaises(ValueError):
                validate_shard(
                    shard_hard.parent,
                    input_directory=input_directory,
                )

    def test_quotient_nested_directory_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory, rows, _ = self._input(root)
            output = root / "output"
            extract_rank(
                input_directory=input_directory,
                output_root=output,
                rank=0,
                block_size=3,
                audit_top_k=2,
                block_matcher=MatrixMatcher(
                    self._score_matrix(len(rows))
                ),
            )
            shard = (
                output / "shards" / "rank-00000-of-00008"
            )
            nested = shard / "quotient_rank_partial"
            external = root / "moved-rank-partial"
            shard.chmod(0o755)
            artifact_permissions.make_staging_tree_removable(nested)
            nested.rename(external)
            nested.symlink_to(external, target_is_directory=True)
            shard.chmod(0o555)
            with self.assertRaises(ValueError):
                validate_shard(
                    shard,
                    input_directory=input_directory,
                )

    def test_graph_input_is_hash_bound_and_rejects_feature_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory, rows, _ = self._input(root, count=4)
            validated = validate_graph_input(input_directory)
            self.assertEqual(len(validated["rows"]), len(rows))
            done = json.loads(
                (input_directory / "done.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                validated["artifact_digest"],
                done["artifact_digest"],
            )
            features_path = input_directory / "features.npz"
            payload = bytearray(features_path.read_bytes())
            payload[-1] ^= 1
            features_path.write_bytes(bytes(payload))
            with self.assertRaises(ValueError):
                validate_graph_input(input_directory)


if __name__ == "__main__":
    unittest.main()
