from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from motive import r7_artifact_permissions as permissions
from motive import r7_candidate_temporal_screen as screen


def _features(value: np.ndarray) -> dict[str, np.ndarray]:
    vector = screen._normalise_feature(value)
    return {
        modality: vector.copy() for modality in screen.MODALITIES
    }


def _example(
    iid: str,
    *,
    label: str,
    family: str,
    split: str,
    component: str,
    vector: tuple[float, float] = (1.0, 0.0),
    energy: float = 1.0,
) -> screen._Example:
    return screen._Example(
        iid=iid,
        label_class=label,
        family=family,
        split=split,
        component_id=component,
        fresh=split != "train",
        sampling_weight=1.0 if label == "positive" else 9.25,
        features=_features(np.asarray(vector, dtype=np.float64)),
        motion_energy=energy,
    )


class MotionDescriptorTests(unittest.TestCase):
    def test_static_and_linear_motion_have_expected_fixed_statistics(self) -> None:
        frames = 8
        tracks = 16
        visibility = np.ones((frames, tracks), dtype=np.float32)
        static = np.zeros((frames, tracks, 2), dtype=np.float32)
        temporal, endpoint, orderless, energy = (
            screen._motion_descriptors(static, visibility)
        )
        self.assertEqual(temporal.shape, ((frames - 1) * 15,))
        self.assertEqual(endpoint.shape, (15,))
        self.assertEqual(orderless.shape, (90,))
        self.assertTrue(np.array_equal(temporal, np.zeros_like(temporal)))
        self.assertTrue(np.array_equal(endpoint, np.zeros_like(endpoint)))
        self.assertEqual(energy, 0.0)

        moving = static.copy()
        moving[:, :, 0] = np.linspace(0.0, 1.0, frames)[:, None]
        temporal, endpoint, _orderless, energy = (
            screen._motion_descriptors(moving, visibility)
        )
        sequence = temporal.reshape(frames - 1, 15)
        np.testing.assert_allclose(sequence[:, :5], 1.0, atol=1e-6)
        np.testing.assert_allclose(sequence[:, 5:10], 0.0, atol=1e-6)
        np.testing.assert_allclose(sequence[:, 10:], 1.0, atol=1e-6)
        np.testing.assert_allclose(endpoint[:5], 1.0, atol=1e-6)
        self.assertAlmostEqual(energy, 1.0, places=6)

    def test_query_controls_are_deterministic_and_change_order(self) -> None:
        iid = "candidate-control"
        first = screen._shuffle_indices(iid, 32, seed=17)
        second = screen._shuffle_indices(iid, 32, seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(sorted(first.tolist()), list(range(32)))
        self.assertNotEqual(first.tolist(), list(range(32)))
        self.assertNotEqual(first.tolist(), list(reversed(range(32))))
        blocks = np.arange(32 * 15, dtype=np.float64).reshape(32, 15)
        shuffled_blocks = blocks[first]
        self.assertEqual(
            sorted(map(tuple, blocks.tolist())),
            sorted(map(tuple, shuffled_blocks.tolist())),
        )

        frames = 8
        tracks = np.zeros((frames, 16, 2), dtype=np.float32)
        tracks[:, :, 0] = np.linspace(0.0, 1.0, frames)[:, None] ** 2
        visibility = np.ones((frames, 16), dtype=np.float32)
        clean, _, _, _ = screen._motion_descriptors(tracks, visibility)
        reverse, _, _, _ = screen._motion_descriptors(
            tracks[::-1],
            visibility[::-1],
        )
        self.assertFalse(np.allclose(clean, reverse))

    def test_camera_descriptor_separates_identity_and_translation(self) -> None:
        identity = np.zeros((8, 2, 3), dtype=np.float32)
        identity[:, 0, 0] = 1.0
        identity[:, 1, 1] = 1.0
        descriptor = screen._camera_descriptor(identity)
        self.assertEqual(descriptor.shape, (32,))
        self.assertTrue(np.array_equal(descriptor, np.zeros_like(descriptor)))
        translated = identity.copy()
        translated[:, 0, 2] = np.linspace(0.0, 0.25, 8)
        translated_descriptor = screen._camera_descriptor(translated)
        self.assertGreater(float(np.linalg.norm(translated_descriptor)), 0.0)


class EvaluationTests(unittest.TestCase):
    def test_realized_split_topology_reads_bound_spanning_edges(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            name = screen.indexed_graph_io.SPANNING_EDGES_NAME
            path = root / name
            summary_name = screen.indexed_graph_io.SUMMARY_NAME
            summary_path = root / summary_name

            def edge(
                left_iid: str,
                left_role: str,
                right_iid: str,
                right_role: str,
                relation: str,
                value: object,
            ) -> dict[str, object]:
                return {
                    "schema_version":
                        screen.indexed_graph_io.EDGE_ROW_SCHEMA,
                    "left_iid": left_iid,
                    "left_role": left_role,
                    "right_iid": right_iid,
                    "right_role": right_role,
                    "relation": relation,
                    "value": value,
                }

            def candidate_for(
                rows: list[dict[str, object]],
                *,
                payload: bytes | None = None,
                output_rows: int | None = None,
                spanning_count: int | None = None,
            ) -> dict:
                canonical_payload = screen._jsonl_bytes(rows)
                payload = (
                    canonical_payload if payload is None else payload
                )
                path.write_bytes(payload)
                nodes = {
                    (row["left_iid"], row["left_role"])
                    for row in rows
                } | {
                    (row["right_iid"], row["right_role"])
                    for row in rows
                }
                iids = {
                    str(row[field])
                    for row in rows
                    for field in ("left_iid", "right_iid")
                }
                relations = {
                    relation: sum(
                        row.get("relation") == relation for row in rows
                    )
                    for relation in (
                        "paired_sample",
                        "exact_sha256",
                        "dhash_hamming",
                        "dino_cosine",
                    )
                }
                summary = {
                    "schema_version":
                        screen.indexed_graph_io.SUMMARY_SCHEMA,
                    "status": "complete",
                    "counts": {
                        "total_iids": len(iids),
                        "assets": len(nodes),
                        "components": len(nodes) - len(rows),
                        "spanning_edges": (
                            len(rows)
                            if spanning_count is None
                            else spanning_count
                        ),
                    },
                    "statistics": {"relation_counts": relations},
                    "outputs": {
                        name: {
                            "rows": (
                                len(rows)
                                if output_rows is None
                                else output_rows
                            ),
                            "sha256": hashlib.sha256(
                                payload
                            ).hexdigest(),
                            "order":
                                "canonical-endpoints-relation-value",
                        }
                    },
                }
                summary_payload = screen._pretty_json_bytes(summary)
                summary_path.write_bytes(summary_payload)
                return {
                    "summary": {
                        "input_bindings": {
                            "indexed_graph": {
                                "path": str(root),
                                "artifact_digest": "a" * 64,
                                "files": {
                                    name: {
                                        "sha256": hashlib.sha256(
                                            payload
                                        ).hexdigest(),
                                        "bytes": len(payload),
                                    },
                                    summary_name: {
                                        "sha256": hashlib.sha256(
                                            summary_payload
                                        ).hexdigest(),
                                        "bytes": len(summary_payload),
                                    },
                                },
                            }
                        }
                    }
                }

            base_rows = [
                edge(
                    "a",
                    "source",
                    "a",
                    "target",
                    "paired_sample",
                    None,
                ),
                edge(
                    "a",
                    "target",
                    "b",
                    "source",
                    "dhash_hamming",
                    3,
                ),
                edge(
                    "b",
                    "source",
                    "b",
                    "target",
                    "paired_sample",
                    None,
                ),
            ]
            candidate = candidate_for(base_rows)
            real_open = os.open
            open_flags: list[int] = []

            def tracked_open(
                target: os.PathLike[str] | str,
                flags: int,
                *args: object,
                **kwargs: object,
            ) -> int:
                open_flags.append(flags)
                return real_open(target, flags, *args, **kwargs)

            with (
                patch.object(
                    screen,
                    "_file_digest",
                    side_effect=AssertionError("separate hash forbidden"),
                ),
                patch.object(
                    screen,
                    "_load_canonical_jsonl",
                    side_effect=AssertionError("second read forbidden"),
                ),
                patch.object(screen.os, "open", side_effect=tracked_open),
            ):
                audit, bound_path = (
                    screen._validate_split_topology_binding(candidate)
                )
            self.assertEqual(bound_path, path.resolve())
            self.assertTrue(open_flags)
            self.assertTrue(
                all(flags & os.O_NOFOLLOW for flags in open_flags)
            )
            self.assertEqual(audit["retained_dino_spanning_edges"], 0)
            self.assertTrue(audit["realized_partition_is_dino_edge_free"])

            dino_rows = list(base_rows)
            dino_rows[1] = edge(
                "a",
                "target",
                "b",
                "source",
                "dino_cosine",
                0.97,
            )
            audit, _ = screen._validate_split_topology_binding(
                candidate_for(dino_rows)
            )
            self.assertEqual(audit["retained_dino_spanning_edges"], 1)
            self.assertTrue(
                audit["dino_changed_realized_component_topology"]
            )

    def test_realized_split_topology_rejects_invalid_edges_and_receipts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            name = screen.indexed_graph_io.SPANNING_EDGES_NAME
            path = root / name
            summary_name = screen.indexed_graph_io.SUMMARY_NAME
            summary_path = root / summary_name

            def edge(
                left_iid: str,
                left_role: str,
                right_iid: str,
                right_role: str,
                relation: str,
                value: object,
            ) -> dict[str, object]:
                return {
                    "schema_version":
                        screen.indexed_graph_io.EDGE_ROW_SCHEMA,
                    "left_iid": left_iid,
                    "left_role": left_role,
                    "right_iid": right_iid,
                    "right_role": right_role,
                    "relation": relation,
                    "value": value,
                }

            valid = [
                edge(
                    "a",
                    "source",
                    "a",
                    "target",
                    "paired_sample",
                    None,
                ),
                edge(
                    "a",
                    "target",
                    "b",
                    "source",
                    "dino_cosine",
                    0.97,
                ),
                edge(
                    "b",
                    "source",
                    "b",
                    "target",
                    "paired_sample",
                    None,
                ),
            ]

            def candidate_for(
                rows: list[dict[str, object]],
                *,
                payload: bytes | None = None,
                output_rows: int | None = None,
                spanning_count: int | None = None,
            ) -> dict:
                payload = (
                    screen._jsonl_bytes(rows)
                    if payload is None
                    else payload
                )
                path.write_bytes(payload)
                relations = {
                    relation: sum(
                        row.get("relation") == relation for row in rows
                    )
                    for relation in (
                        "paired_sample",
                        "exact_sha256",
                        "dhash_hamming",
                        "dino_cosine",
                    )
                }
                summary = {
                    "schema_version":
                        screen.indexed_graph_io.SUMMARY_SCHEMA,
                    "status": "complete",
                    "counts": {
                        "total_iids": 2,
                        "assets": 4,
                        "components": 1,
                        "spanning_edges": (
                            len(rows)
                            if spanning_count is None
                            else spanning_count
                        ),
                    },
                    "statistics": {"relation_counts": relations},
                    "outputs": {
                        name: {
                            "rows": (
                                len(rows)
                                if output_rows is None
                                else output_rows
                            ),
                            "sha256": hashlib.sha256(
                                payload
                            ).hexdigest(),
                            "order":
                                "canonical-endpoints-relation-value",
                        }
                    },
                }
                summary_payload = screen._pretty_json_bytes(summary)
                summary_path.write_bytes(summary_payload)
                return {
                    "summary": {
                        "input_bindings": {
                            "indexed_graph": {
                                "path": str(root),
                                "artifact_digest": "a" * 64,
                                "files": {
                                    name: {
                                        "sha256": hashlib.sha256(
                                            payload
                                        ).hexdigest(),
                                        "bytes": len(payload),
                                    },
                                    summary_name: {
                                        "sha256": hashlib.sha256(
                                            summary_payload
                                        ).hexdigest(),
                                        "bytes": len(summary_payload),
                                    },
                                },
                            }
                        }
                    }
                }

            malformed_rows: list[list[dict[str, object]]] = []
            wrong_schema = [dict(row) for row in valid]
            wrong_schema[0]["schema_version"] = "wrong"
            malformed_rows.append(wrong_schema)
            reversed_endpoint = [dict(row) for row in valid]
            reversed_endpoint[1]["left_iid"] = "b"
            reversed_endpoint[1]["right_iid"] = "a"
            malformed_rows.append(reversed_endpoint)
            wrong_role = [dict(row) for row in valid]
            wrong_role[1]["left_role"] = "query"
            malformed_rows.append(wrong_role)
            wrong_relation = [dict(row) for row in valid]
            wrong_relation[1]["relation"] = "unknown"
            malformed_rows.append(wrong_relation)
            wrong_value = [dict(row) for row in valid]
            wrong_value[1]["value"] = 0.5
            malformed_rows.append(wrong_value)
            malformed_rows.append(list(reversed(valid)))
            for rows in malformed_rows:
                with self.subTest(rows=rows):
                    with self.assertRaises(
                        screen.CandidateTemporalScreenError
                    ):
                        screen._validate_split_topology_binding(
                            candidate_for(rows)
                        )

            noncanonical = b" " + screen._jsonl_bytes(valid)
            with self.assertRaises(screen.CandidateTemporalScreenError):
                screen._validate_split_topology_binding(
                    candidate_for(valid, payload=noncanonical)
                )
            with self.assertRaises(screen.CandidateTemporalScreenError):
                screen._validate_split_topology_binding(
                    candidate_for(valid, output_rows=2)
                )
            with self.assertRaises(screen.CandidateTemporalScreenError):
                screen._validate_split_topology_binding(
                    candidate_for(valid, spanning_count=2)
                )

            candidate = candidate_for(valid)
            path.write_bytes(screen._jsonl_bytes(valid[:-1]))
            with self.assertRaises(screen.CandidateTemporalScreenError):
                screen._validate_split_topology_binding(candidate)

            candidate = candidate_for(valid)
            target = root / "replacement.jsonl"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
            with self.assertRaises(screen.CandidateTemporalScreenError):
                screen._validate_split_topology_binding(candidate)

    def test_support_filter_is_exact(self) -> None:
        examples = [
            _example(
                f"a-train-{index}",
                label="positive",
                family="wave",
                split="train",
                component=f"a-component-{index}",
            )
            for index in range(5)
        ]
        examples.extend(
            [
                _example(
                    "a-validation",
                    label="positive",
                    family="wave",
                    split="validation",
                    component="a-eval-1",
                ),
            ]
        )
        examples.extend(
            [
                _example(
                    f"b-train-{index}",
                    label="positive",
                    family="jump",
                    split="train",
                    component=f"b-component-{index % 3}",
                )
                for index in range(4)
            ]
        )
        examples.extend(
            [
                _example(
                    "b-validation",
                    label="positive",
                    family="jump",
                    split="validation",
                    component="b-eval-1",
                ),
                _example(
                    "b-test",
                    label="positive",
                    family="jump",
                    split="test",
                    component="b-eval-2",
                ),
            ]
        )
        examples.extend(
            [
                _example(
                    f"c-train-{index}",
                    label="positive",
                    family="clap",
                    split="train",
                    component=f"c-component-{index}",
                )
                for index in range(5)
            ]
        )
        support, eligible = screen._family_support(examples)
        self.assertEqual(eligible, {"wave", "clap"})
        self.assertTrue(support["families"]["wave"]["eligible"])
        self.assertFalse(
            support["families"]["wave"][
                "evaluation_support_sufficient_for_precision_reporting"
            ]
        )
        self.assertTrue(support["families"]["clap"]["eligible"])
        self.assertEqual(
            support["families"]["clap"]["evaluation_queries"],
            0,
        )
        self.assertEqual(
            support["families"]["jump"]["exclusion_reasons"],
            [
                "train_references_below_5",
                "train_components_below_5",
            ],
        )

    def test_retrieval_and_binary_auc_use_clean_independent_bank(self) -> None:
        examples = [
            _example(
                f"train-{index}",
                label="positive",
                family="wave",
                split="train",
                component=f"train-component-{index}",
                vector=(1.0, 0.0),
                energy=2.0,
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
                    vector=(0.0, 1.0),
                    energy=2.0,
                )
                for index in range(5)
            ]
        )
        examples.extend(
            [
                _example(
                    "positive-validation",
                    label="positive",
                    family="wave",
                    split="validation",
                    component="eval-component-1",
                    vector=(1.0, 0.0),
                    energy=2.0,
                ),
                _example(
                    "positive-test",
                    label="positive",
                    family="wave",
                    split="test",
                    component="eval-component-2",
                    vector=(1.0, 0.0),
                    energy=2.0,
                ),
                _example(
                    "jump-positive-validation",
                    label="positive",
                    family="jump",
                    split="validation",
                    component="jump-eval-component-1",
                    vector=(0.0, 1.0),
                    energy=2.0,
                ),
                _example(
                    "jump-positive-test",
                    label="positive",
                    family="jump",
                    split="test",
                    component="jump-eval-component-2",
                    vector=(0.0, 1.0),
                    energy=2.0,
                ),
                _example(
                    "negative-validation",
                    label="negative",
                    family="no_action",
                    split="validation",
                    component="eval-component-3",
                    vector=(-1.0, 0.0),
                    energy=0.0,
                ),
                _example(
                    "negative-test",
                    label="negative",
                    family="no_action",
                    split="test",
                    component="eval-component-4",
                    vector=(-1.0, 0.0),
                    energy=0.0,
                ),
            ]
        )
        rows, retrieval, binary, diagnostics = screen._evaluate(
            examples,
            eligible_families={"wave", "jump"},
        )
        self.assertEqual(len(rows), 6)
        for modality in screen.MODALITIES:
            self.assertEqual(
                retrieval[modality]["overall"]["micro"]["r_at_1"],
                1.0,
            )
            for row in rows:
                result = row["modalities"][modality]
                self.assertEqual(
                    len(set(result["top_reference_components"])),
                    5,
                )
            self.assertEqual(
                binary[modality]["overall"]["sampled_auroc"],
                1.0,
            )
        self.assertEqual(
            binary["target_motion_energy"]["overall"]["sampled_auroc"],
            1.0,
        )
        self.assertFalse(
            diagnostics["leakage_control"][
                "source_video_as_independent_reference"
            ]
        )
        self.assertEqual(
            diagnostics["coverage"]["train_reference_rows"],
            10,
        )
        screen._validate_metric_rows(rows)

    def test_zero_query_is_not_ranked_and_counts_as_conservative_miss(
        self,
    ) -> None:
        examples = [
            _example(
                f"train-{index}",
                label="positive",
                family="wave",
                split="train",
                component=f"train-component-{index}",
                vector=(1.0, 0.0),
            )
            for index in range(5)
        ]
        examples.extend(
            [
                _example(
                    "zero-positive-validation",
                    label="positive",
                    family="wave",
                    split="validation",
                    component="zero-eval-component",
                    vector=(0.0, 0.0),
                    energy=0.0,
                ),
                _example(
                    "negative-validation",
                    label="negative",
                    family="none",
                    split="validation",
                    component="negative-eval-component",
                    vector=(-1.0, 0.0),
                    energy=0.0,
                ),
            ]
        )
        rows, retrieval, binary, _ = screen._evaluate(
            examples,
            eligible_families={"wave"},
        )
        zero = next(row for row in rows if row["iid"].startswith("zero-"))
        for modality in screen.MODALITIES:
            result = zero["modalities"][modality]
            self.assertFalse(result["valid_for_retrieval"])
            self.assertEqual(result["invalid_reason"], "zero_query")
            self.assertEqual(result["top_reference_iids"], [])
            self.assertIsNone(zero["binary_scores"][modality])
            self.assertEqual(
                retrieval[modality]["overall"]["micro"]["r_at_1"],
                0.0,
            )
            self.assertIsNone(
                retrieval[modality]["overall"]["micro"][
                    "valid_only_r_at_1"
                ]
            )
            self.assertEqual(
                binary[modality]["overall"]["valid_positive_rows"],
                0,
            )
        screen._validate_metric_rows(rows)


class ImmutableOutputTests(unittest.TestCase):
    def _commit(self, root: Path) -> Path:
        examples = [
            _example(
                f"train-{index}",
                label="positive",
                family="wave",
                split="train",
                component=f"train-component-{index}",
            )
            for index in range(5)
        ]
        examples.extend(
            [
                _example(
                    "positive-validation",
                    label="positive",
                    family="wave",
                    split="validation",
                    component="eval-1",
                ),
                _example(
                    "positive-test",
                    label="positive",
                    family="wave",
                    split="test",
                    component="eval-2",
                ),
                _example(
                    "negative-validation",
                    label="negative",
                    family="none",
                    split="validation",
                    component="eval-3",
                    vector=(-1.0, 0.0),
                    energy=0.0,
                ),
            ]
        )
        rows, retrieval, binary, diagnostics = screen._evaluate(
            examples,
            eligible_families={"wave"},
        )
        contract = {
            "schema_version": screen.SCREEN_SCHEMA,
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
                "sha256":
                    hashlib.sha256(summary_bytes).hexdigest(),
                "bytes": len(summary_bytes),
                "mode_octal": "0444",
            },
        }
        done = screen._done_payload(
            rows=len(rows),
            contract_sha256=screen._object_digest(contract),
            payload_files=payload_files,
        )
        output = root / "screen"
        output.mkdir()
        (output / screen.ROWS_NAME).write_bytes(row_bytes)
        (output / screen.SUMMARY_NAME).write_bytes(summary_bytes)
        (output / screen.DONE_NAME).write_bytes(
            screen._pretty_json_bytes(done)
        )
        permissions.seal_staging_tree(output)
        return output

    def test_output_replays_and_rejects_mode_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = self._commit(root)
            result = screen._validate_candidate_temporal_screen_envelope(
                output
            )
            self.assertEqual(result["summary"]["formal_status"], "INSUFFICIENT")
            os.chmod(output, 0o755)
            with self.assertRaisesRegex(ValueError, "mode differs"):
                screen._validate_candidate_temporal_screen_envelope(output)
            os.chmod(output, 0o555)

    def test_resume_repairs_interrupted_post_rename_root_mode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = self._commit(root)
            payloads = {
                name: (source / name).read_bytes()
                for name in screen.OUTPUT_NAMES
            }
            target = root / "interrupted"
            inputs = screen._Inputs(
                candidate={},
                cache={},
                visual={},
                original_rows=(),
                binding={},
                identities={},
                media_identities={},
            )
            with patch.object(
                screen.artifact_permissions,
                "seal_published_root",
                side_effect=RuntimeError("injected seal interruption"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "injected seal interruption",
                ):
                    screen._publish(
                        target,
                        payloads=payloads,
                        inputs=inputs,
                    )
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)
            screen._strict_resume(target, payloads=payloads)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555)
            screen._validate_candidate_temporal_screen_envelope(target)

    def test_output_rejects_metric_tampering_even_with_updated_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = self._commit(root)
            permissions.make_staging_tree_removable(output)
            summary_path = output / screen.SUMMARY_NAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["retrieval"][screen.TARGET_TEMPORAL]["overall"][
                "micro"
            ]["r_at_1"] = 0.0
            summary_path.write_bytes(screen._pretty_json_bytes(summary))
            done_path = output / screen.DONE_NAME
            done = json.loads(done_path.read_text(encoding="utf-8"))
            record = done["payload_files"][screen.SUMMARY_NAME]
            record["sha256"] = screen._file_digest(summary_path)
            record["bytes"] = summary_path.stat().st_size
            done["artifact_digest"] = screen._object_digest(
                done["payload_files"]
            )
            done_path.write_bytes(screen._pretty_json_bytes(done))
            permissions.seal_staging_tree(output)
            with self.assertRaisesRegex(
                screen.CandidateTemporalScreenError,
                "do not replay",
            ):
                screen._validate_candidate_temporal_screen_envelope(output)

    def test_strict_replay_rejects_self_consistent_row_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = self._commit(root)
            original_payloads = {
                name: (output / name).read_bytes()
                for name in screen.OUTPUT_NAMES
            }
            permissions.make_staging_tree_removable(output)
            rows = [
                json.loads(line)
                for line in (
                    output / screen.ROWS_NAME
                ).read_text(encoding="utf-8").splitlines()
            ]
            negative = next(
                row for row in rows if row["label_class"] == "negative"
            )
            negative["family"] = "forged-family"
            row_bytes = screen._jsonl_bytes(rows)
            (output / screen.ROWS_NAME).write_bytes(row_bytes)

            summary_path = output / screen.SUMMARY_NAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["output"]["rows_sha256"] = hashlib.sha256(
                row_bytes
            ).hexdigest()
            summary_bytes = screen._pretty_json_bytes(summary)
            summary_path.write_bytes(summary_bytes)
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
            (output / screen.DONE_NAME).write_bytes(
                screen._pretty_json_bytes(done)
            )
            permissions.seal_staging_tree(output)
            # The old envelope-only check cannot know that this family was
            # changed; the strict validator must reproduce from upstream.
            screen._validate_candidate_temporal_screen_envelope(output)
            fake_inputs = screen._Inputs(
                candidate={},
                cache={},
                visual={},
                original_rows=(),
                binding={},
                identities={},
                media_identities={},
            )
            with patch.object(
                screen,
                "_derive",
                return_value=(original_payloads, fake_inputs),
            ), self.assertRaisesRegex(
                screen.CandidateTemporalScreenError,
                "does not replay from bound inputs",
            ):
                screen.validate_candidate_temporal_screen(
                    output,
                    expected_done_sha256=screen._file_digest(
                        output / screen.DONE_NAME
                    ),
                    candidate_manifest_dir=Path("/unused/candidate"),
                    expected_candidate_manifest_done_sha256="a" * 64,
                    track_cache_final=Path("/unused/cache"),
                    expected_track_cache_done_sha256="b" * 64,
                    visual_features_final=Path("/unused/visual"),
                    expected_visual_features_done_sha256="c" * 64,
                    visual_candidates_manifest=Path(
                        "/unused/candidates.jsonl"
                    ),
                    expected_visual_candidates_sha256="d" * 64,
                )

    def test_fresh_derivation_payload_validates_end_to_end(self) -> None:
        specifications = [
            (
                f"train-{index}",
                "positive",
                "train",
                f"train-component-{index}",
                "wave",
            )
            for index in range(5)
        ]
        specifications.extend(
            [
                (
                    f"jump-train-{index}",
                    "positive",
                    "train",
                    f"jump-train-component-{index}",
                    "jump",
                )
                for index in range(5)
            ]
        )
        specifications.extend(
            [
                (
                    "positive-validation",
                    "positive",
                    "validation",
                    "eval-1",
                    "wave",
                ),
                ("positive-test", "positive", "test", "eval-2", "wave"),
                (
                    "jump-positive-validation",
                    "positive",
                    "validation",
                    "jump-eval-1",
                    "jump",
                ),
                (
                    "jump-positive-test",
                    "positive",
                    "test",
                    "jump-eval-2",
                    "jump",
                ),
                (
                    "negative-validation",
                    "negative",
                    "validation",
                    "eval-3",
                    "none",
                ),
                ("negative-test", "negative", "test", "eval-4", "none"),
            ]
        )
        row_count = len(specifications)
        candidate_rows = []
        cache_rows = []
        visual_rows = []
        for index, (iid, label, split, component, family) in enumerate(
            specifications
        ):
            candidate_rows.append(
                {
                    "iid": iid,
                    "label": {
                        "class": label,
                        "primary_family": family,
                    },
                    "assignment": {
                        "split": split,
                        "component_id": component,
                        "fresh": split != "train",
                    },
                    "sampling": {
                        "inverse_probability_weight":
                            1.0 if label == "positive" else 9.25,
                    },
                }
            )
            cache_rows.append({"iid": iid})
            visual_rows.append({"iid": iid})
        frames = 8
        tracks = 16
        source = np.zeros((row_count, frames, tracks, 2), dtype=np.float32)
        target = source.copy()
        for index, (
            _iid,
            label,
            _split,
            _component,
            family,
        ) in enumerate(
            specifications
        ):
            if label == "negative":
                target[index, :, :, 0] = (
                    -np.linspace(0.0, 1.0, frames)[:, None]
                )
            elif family == "jump":
                target[index, :, :, 1] = np.linspace(
                    0.0, 1.0, frames
                )[:, None]
            else:
                target[index, :, :, 0] = np.linspace(
                    0.0, 1.0, frames
                )[:, None]
        visibility = np.ones(
            (row_count, frames, tracks),
            dtype=np.float32,
        )
        cumulative = np.zeros(
            (row_count, frames, 2, 3),
            dtype=np.float32,
        )
        cumulative[:, :, 0, 0] = 1.0
        cumulative[:, :, 1, 1] = 1.0
        dino = np.zeros((row_count, 6, 4), dtype=np.float32)
        dino[:, :, 0] = 1.0
        fake = screen._Inputs(
            candidate={"rows": candidate_rows},
            cache={
                "rows": cache_rows,
                "arrays": {
                    "source_camera_valid":
                        np.ones(row_count, dtype=np.bool_),
                    "target_camera_valid":
                        np.ones(row_count, dtype=np.bool_),
                    "source_stabilized_tracks": source,
                    "source_visibility": visibility,
                    "target_stabilized_tracks": target,
                    "target_visibility": visibility,
                    "target_cumulative_affines": cumulative,
                },
            },
            visual={
                "rows": visual_rows,
                "arrays": {
                    "target_valid": np.ones(row_count, dtype=np.bool_),
                    "target_dino_cls": dino,
                },
            },
            original_rows=(),
            binding={
                "fixture": "fully-bound-by-mocked-validator",
                "realized_split_topology": {
                    "retained_dino_spanning_edges": 0,
                    "dino_changed_realized_component_topology": False,
                    "realized_partition_is_dino_edge_free": True,
                },
            },
            identities={},
            media_identities={},
        )
        with patch.object(screen, "_validate_inputs", return_value=fake):
            payloads, inputs = screen._derive(
                candidate_manifest_dir=Path("/unused/candidate"),
                expected_candidate_manifest_done_sha256="a" * 64,
                track_cache_final=Path("/unused/cache"),
                expected_track_cache_done_sha256="b" * 64,
                visual_features_final=Path("/unused/visual"),
                expected_visual_features_done_sha256="c" * 64,
                visual_candidates_manifest=Path("/unused/candidates.jsonl"),
                expected_visual_candidates_sha256="d" * 64,
                seed=screen.DEFAULT_SEED,
                verify_source_shards=True,
                rehash_videos=True,
            )
        self.assertIs(inputs, fake)
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "derived"
            output.mkdir()
            for name, payload in payloads.items():
                (output / name).write_bytes(payload)
            permissions.seal_staging_tree(output)
            arguments = {
                "candidate_manifest_dir": Path("/unused/candidate"),
                "expected_candidate_manifest_done_sha256": "a" * 64,
                "track_cache_final": Path("/unused/cache"),
                "expected_track_cache_done_sha256": "b" * 64,
                "visual_features_final": Path("/unused/visual"),
                "expected_visual_features_done_sha256": "c" * 64,
                "visual_candidates_manifest":
                    Path("/unused/candidates.jsonl"),
                "expected_visual_candidates_sha256": "d" * 64,
            }
            with self.assertRaisesRegex(
                screen.CandidateTemporalScreenError,
                "external done SHA",
            ):
                screen.validate_candidate_temporal_screen(
                    output,
                    expected_done_sha256="0" * 64,
                    **arguments,
                )
            with patch.object(
                screen,
                "_derive",
                return_value=(payloads, fake),
            ):
                result = screen.validate_candidate_temporal_screen(
                    output,
                    expected_done_sha256=screen._file_digest(
                        output / screen.DONE_NAME
                    ),
                    **arguments,
                )
            self.assertTrue(result["upstream_replay_verified"])
        self.assertEqual(
            result["summary"]["support"]["eligible_families"],
            ["jump", "wave"],
        )
        self.assertEqual(result["done"]["formal_status"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
